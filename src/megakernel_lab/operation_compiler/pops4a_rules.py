"""P-OPS4-A hierarchical leading-axis reductions (parameter gradients and clipping norms).

The frozen generic reduction gives one CTA to each result and strides its 256 threads across the reduced rows, so a
bias or normalization-weight gradient over B*T rows reads one 4-byte element per row and lane (uncoalesced), and a
full reduction (a clipping squared norm) runs on one CTA. P-OPS3-C's exact column engine covers <= 4096 rows and
C-OPS2-C's split covers >= 16384-element results with an exclusive two-barrier phase. This rule covers the rest:

* ``column``: reduced axes are a leading prefix, kept trailing axes contiguous (>= 32 results, > 4096 rows when the
  exact P-OPS3-C engine is enabled). A CTA item owns one row tile and 32 adjacent columns; warp w sums rows
  r0 + w, r0 + w + 8, ... of its lane's column (every load instruction reads 128 contiguous bytes), the eight warp
  sums are combined in warp order through shared memory and stored as partial[tile][column].
* ``full``: every axis reduced (clipping norms). A CTA item owns CH consecutive elements in the storage order of
  the source's dense (possibly permuted) operand view, so transposed weight gradients are read contiguously;
  butterfly per warp, warp-order combination, partial[chunk].
* ``direct``: a full reduction of at most DIRECT_MAX elements is summed by one CTA without partials.

A second item (``finalize``) sums each result's partials in tile order (column: one thread per result; full: one
warp) and applies the mean scale / BF16 rounding. Partial and finalize items are scheduled like any dependent work
and may join existing phases (pinned/selected products, other reductions, other P-OPS4-A items) instead of
occupying an exclusive phase; the partial buffer is an ordinary arena value. Deterministic fixed order, but a
different summation order from the frozen CTA tree: admitted only with the declared numerical checks.
Dispatch uses only axes, extents, layouts and dtypes.
"""
from __future__ import annotations

COLUMN_MIN_RESULTS = 32
OVER_EXACT_MIN_ELEMENTS = 1 << 21   # sources this large prefer this engine over P-OPS3-C's exact <=4096-row engine
COLUMN_TARGET_ITEMS = 256        # CTA items per column reduction (~2 per CTA of the 128-CTA grid)
COLUMN_MIN_ROWS_PER_ITEM = 64    # >= 8 rows per warp
FULL_CHUNK = 4096                # elements per CTA item (16 per thread)
DIRECT_MAX = 8192
UNROLL = 8


def _ceil(a, b):
    return -(-a // b)


def column_eligible(src_shape, src_strides, dims, kept, extent, results, exact_column_covers):
    if not dims or list(dims) != list(range(len(dims))) or not kept:
        return False
    if results < COLUMN_MIN_RESULTS or extent < 2 * COLUMN_MIN_ROWS_PER_ITEM or exact_column_covers:
        return False
    if extent < COLUMN_MIN_ROWS_PER_ITEM:
        return False
    span = 1 + sum((d - 1) * s for d, s in zip(src_shape, src_strides))
    if span >= 2**31:
        return False
    stride = 1
    for axis in reversed(kept):
        if src_strides[axis] != stride:
            return False
        stride *= src_shape[axis]
    return True


def column_plan(extent, results, cols=1, target_items=COLUMN_TARGET_ITEMS, floor=False):
    blocks = _ceil(results, 32 * cols)
    # floor: never exceed target_items (one item per CTA when target_items equals the grid; no partial round)
    tiles = max(1, min(target_items // blocks if floor else _ceil(target_items, blocks),
                       extent // COLUMN_MIN_ROWS_PER_ITEM))
    rows = _ceil(extent, tiles)
    tiles = _ceil(extent, rows)
    return {'blocks': blocks, 'tiles': tiles, 'rows': rows, 'items': tiles * blocks, 'cols': cols}


def column_partials(*, part, extent, results, leaves, plan, rotate, unroll=UNROLL):
    """leaves[j] reads source element (reduced flat index pr, kept flat index o<j>). A warp covers 32c adjacent
    columns: lane l owns columns ob + l + 32j (j < c), so each row issues c coalesced 128-byte loads."""
    b, t, rows, c = plan['blocks'], plan['tiles'], plan['rows'], plan['cols']
    decl = "\n    ".join(f"const unsigned o{j} = ob + {32 * j}u + lane; float acc{j} = 0.0f;" for j in range(c))
    loads = "\n".join(f"      if (o{j} < {results}u) acc{j} += {leaves[j]};" for j in range(c))
    keep = "\n    ".join(f"smem[{256 * j}u + threadIdx.x] = acc{j};" for j in range(c))
    store = "\n".join(f"""      {{
        float s = smem[{256 * j}u + lane];
#pragma unroll
        for (unsigned k = 1; k < 8u; ++k) s += smem[{256 * j}u + k * 32u + lane];
        if (o{j} < {results}u) {part}[t * {results}u + o{j}] = s;
      }}""" for j in range(c))
    return f'''{{ // P-OPS4-A column partials: {results} columns x {extent} rows -> {t} tiles x {rows} rows, {c} column(s) per lane
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + {rotate % 4096}u) % gridDim.x; it < {t * b}u; it += gridDim.x) {{
    const unsigned t = it / {b}u, ob = (it % {b}u) * {32 * c}u;
    const unsigned r0 = t * {rows}u, r1 = min(r0 + {rows}u, {extent}u);
    {decl}
#pragma unroll {unroll}
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {{
{loads}
    }}
    {keep}
    __syncthreads();
    if (w == 0u) {{
{store}
    }}
    __syncthreads();
  }}
}}'''


def column_finalize(*, part, out, results, plan, rounded, scale, rotate):
    t = plan['tiles']
    return f'''{{ // P-OPS4-A column finalize: {results} results, {t} tile partials each (tile order)
  const unsigned g = ((blockIdx.x + {rotate % 4096}u) % gridDim.x) * blockDim.x + threadIdx.x;
  for (unsigned o = g; o < {results}u; o += gridDim.x * blockDim.x) {{
    float s = {part}[o];
#pragma unroll
    for (unsigned k = 1; k < {t}u; ++k) s += {part}[k * {results}u + o];
    {out}[o] = {rounded('s' + scale)};
  }}
}}'''


ARRIVE = '''    __syncthreads();  // partial writers fenced their own stores (threadFenceReduction pattern)
    if (threadIdx.x == 0u) {{
      const unsigned old = atomicAdd(latch_cnt + {slot}, 1u);
      latch_last = old == {expected}u;
      if (latch_last) atomicExch(latch_cnt + {slot}, 0u);
    }}
    __syncthreads();'''


def column_last(*, out, extent, results, leaves, plan, rotate, rounded, scale, unroll=UNROLL):
    """Column partials with last-arriver combination in the same phase (P-OPS2-A split-K counter protocol):
    the CTA that completes a column block's last tile sums that block's tile partials in tile order and stores
    the results. Counters (/*P2_CNT*/) start at zero and are restored by every last arriver; partials live in
    the phase's split scratch (/*P2_PART*/)."""
    b, t, rows, c = plan['blocks'], plan['tiles'], plan['rows'], plan['cols']
    decl = "\n    ".join(f"const unsigned o{j} = ob + {32 * j}u + lane; float acc{j} = 0.0f;" for j in range(c))
    loads = "\n".join(f"      if (o{j} < {results}u) acc{j} += {leaves[j]};" for j in range(c))
    keep = "\n    ".join(f"smem[{256 * j}u + threadIdx.x] = acc{j};" for j in range(c))
    store = "\n".join(f"""      {{
        float s = smem[{256 * j}u + lane];
#pragma unroll
        for (unsigned k = 1; k < 8u; ++k) s += smem[{256 * j}u + k * 32u + lane];
        if (o{j} < {results}u) latch_part[t * {results}u + o{j}] = s;
      }}""" for j in range(c)) + "\n      __threadfence();"
    combine = "\n".join(f"""      if (o{j} < {results}u) {{
        float s = __ldcg(latch_part + o{j});
        for (unsigned k = 1; k < {t}u; ++k) s += __ldcg(latch_part + k * {results}u + o{j});
        {out}[o{j}] = {rounded('s' + scale)};
      }}""" for j in range(c))
    arrive = ARRIVE.format(slot='(it % ' + str(b) + 'u)', expected=t - 1)
    return f'''{{ // P-OPS4-A column partials + last-arriver combination: {results} columns x {extent} rows -> {t} tiles x {rows} rows, {c} column(s) per lane
  unsigned* latch_cnt = /*P2_CNT*/;
  float* latch_part = /*P2_PART*/;
  __shared__ bool latch_last;
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + {rotate % 4096}u) % gridDim.x; it < {t * b}u; it += gridDim.x) {{
    const unsigned t = it / {b}u, ob = (it % {b}u) * {32 * c}u;
    const unsigned r0 = t * {rows}u, r1 = min(r0 + {rows}u, {extent}u);
    {decl}
#pragma unroll {unroll}
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {{
{loads}
    }}
    {keep}
    __syncthreads();
    if (w == 0u) {{
{store}
    }}
{arrive}
    if (latch_last && w == 0u) {{
      __threadfence();
{combine}
    }}
    __syncthreads();
  }}
}}'''


def full_last(*, out, extent, leaf, plan, rotate, rounded, scale, unroll=UNROLL):
    c, chunk = plan['chunks'], plan['chunk']
    leaves = chunk // 256
    arrive = ARRIVE.format(slot='0u', expected=c - 1)
    return f'''{{ // P-OPS4-A full partials + last-arriver combination: {extent} elements -> {c} chunks of {chunk} (storage order)
  unsigned* latch_cnt = /*P2_CNT*/;
  float* latch_part = /*P2_PART*/;
  __shared__ bool latch_last;
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + {rotate % 4096}u) % gridDim.x; it < {c}u; it += gridDim.x) {{
    const unsigned base = it * {chunk}u + threadIdx.x;
    float acc = 0.0f;
#pragma unroll {unroll}
    for (unsigned j = 0; j < {leaves}u; ++j) {{
      const unsigned s = base + 256u * j;
      if (s < {extent}u) acc += {leaf};
    }}
    acc = s101h::warp_sum(acc);
    if (lane == 0u) smem[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {{
      float total = smem[0];
#pragma unroll
      for (unsigned k = 1; k < 8u; ++k) total += smem[k];
      latch_part[it] = total;
      __threadfence();
    }}
{arrive}
    if (latch_last && w == 0u) {{
      __threadfence();
      float s = 0.0f;
      for (unsigned k = lane; k < {c}u; k += 32u) s += __ldcg(latch_part + k);
      s = s101h::warp_sum(s);
      if (lane == 0u) {out}[0] = {rounded('s' + scale)};
    }}
    __syncthreads();
  }}
}}'''


def storage_order_position(view_shape, view_strides, var):
    """Logical (row-major) flat index of the element at storage offset `var` of a dense permuted view."""
    contiguous, s = [], 1
    for d in reversed(view_shape):
        contiguous.append(s)
        s *= d
    contiguous = list(reversed(contiguous))
    terms = []
    for d, (extent, stride, logical) in enumerate(zip(view_shape, view_strides, contiguous)):
        if extent == 1:
            continue
        coord = var if stride == 1 else f"({var} / {stride}u)"
        if stride * extent != s:
            coord = f"({coord} % {extent}u)"
        terms.append(coord if logical == 1 else f"{coord} * {logical}u")
    return " + ".join(terms) or "0u"


def dense_permutation(shape, strides, offset):
    """True for an offset-0 view whose strides are a permutation of some contiguous stride set."""
    if offset:
        return False
    dims = sorted((s, d) for s, d in zip(strides, shape) if d != 1)
    expect = 1
    for s, d in dims:
        if s != expect:
            return False
        expect *= d
    return True


def full_plan(extent, chunk=FULL_CHUNK):
    if extent <= DIRECT_MAX:
        return {'mode': 'direct', 'items': 1}
    chunks = _ceil(extent, chunk)
    return {'mode': 'full', 'chunks': chunks, 'items': chunks, 'chunk': chunk}


def full_partials(*, part, extent, leaf, plan, rotate, unroll=UNROLL):
    """`leaf` reads the source element whose storage-order index is s."""
    c, chunk = plan['chunks'], plan['chunk']
    leaves = chunk // 256
    return f'''{{ // P-OPS4-A full partials: {extent} elements -> {c} chunks of {chunk} (storage order)
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + {rotate % 4096}u) % gridDim.x; it < {c}u; it += gridDim.x) {{
    const unsigned base = it * {chunk}u + threadIdx.x;
    float acc = 0.0f;
#pragma unroll {unroll}
    for (unsigned j = 0; j < {leaves}u; ++j) {{
      const unsigned s = base + 256u * j;
      if (s < {extent}u) acc += {leaf};
    }}
    acc = s101h::warp_sum(acc);
    if (lane == 0u) smem[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {{
      float total = smem[0];
#pragma unroll
      for (unsigned k = 1; k < 8u; ++k) total += smem[k];
      {part}[it] = total;
    }}
    __syncthreads();
  }}
}}'''


def full_finalize(*, part, out, plan, rounded, scale, rotate):
    c = plan['chunks']
    return f'''{{ // P-OPS4-A full finalize: {c} chunk partials (lane-strided, butterfly)
  if (blockIdx.x == {rotate % 4096}u % gridDim.x && threadIdx.x < 32u) {{
    float s = 0.0f;
    for (unsigned k = threadIdx.x; k < {c}u; k += 32u) s += {part}[k];
    s = s101h::warp_sum(s);
    if (threadIdx.x == 0u) {out}[0] = {rounded('s' + scale)};
  }}
}}'''


def direct(*, out, extent, leaf, rounded, scale, rotate):
    leaves = _ceil(extent, 256)
    return f'''{{ // P-OPS4-A direct full reduction: {extent} elements on one CTA (storage order)
  if (blockIdx.x == {rotate % 4096}u % gridDim.x) {{
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    float acc = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < {leaves}u; ++j) {{
      const unsigned s = threadIdx.x + 256u * j;
      if (s < {extent}u) acc += {leaf};
    }}
    acc = s101h::warp_sum(acc);
    if (lane == 0u) smem[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {{
      float total = smem[0];
#pragma unroll
      for (unsigned k = 1; k < 8u; ++k) total += smem[k];
      {out}[0] = {rounded('total' + scale)};
    }}
    __syncthreads();
  }}
}}'''


# ------------------------------------------------------------------------------------------------ outlined bodies
# One out-of-line copy per (leaf functor, K, C) instead of an unrolled inline copy per reduction: the generated
# program's phases otherwise fetch cold, distinct instruction streams (P-OPS3-C finding). FP32-only programs (no
# warpgroup MMA, whose callers must avoid non-inlined calls). Same per-element arithmetic and summation order as
# the inline bodies with the same parameters.
OUTLINED_HEADER = r'''
#ifndef LATCH_P4A_DEFINED
#define LATCH_P4A_DEFINED
struct latch_p4a_nopf { __device__ __forceinline__ void operator()(unsigned, unsigned) const {} };
template <class L, class P, int K, int C, int PD>
__device__ __noinline__ void latch_p4a_column(const L leaf, const P pf, float* p0, float* p1, float* p2, float* p3,
                                              unsigned E, unsigned N, unsigned T, unsigned RT, unsigned B,
                                              unsigned rot, float* sm) {
  float* const parts[4] = {p0, p1, p2, p3};
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < T * B; it += gridDim.x) {
    const unsigned t = it / B, ob = (it % B) * (32u * C);
    const unsigned r0 = t * RT, r1 = min(r0 + RT, E);
    float acc[K][C];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) acc[k][j] = 0.0f;
    if (PD > 0 && lane == 0u) {
      // L2 prefetch of the first PD row steps of this warp (one 128-byte line per operand, column group)
      for (unsigned q = r0 + w; q < r1 && q < r0 + w + 8u * PD; q += 8u)
#pragma unroll
        for (int j = 0; j < C; ++j) if (ob + 32u * j < N) pf(q, ob + 32u * j);
    }
#pragma unroll 8
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {
      if (PD > 0 && lane == 0u && pr + 8u * PD < r1) {
#pragma unroll
        for (int j = 0; j < C; ++j) if (ob + 32u * j < N) pf(pr + 8u * PD, ob + 32u * j);
      }
#pragma unroll
      for (int j = 0; j < C; ++j) {
        const unsigned o = ob + 32u * j + lane;
        if (o < N) {
          float v[K];
          leaf(pr, o, v);
#pragma unroll
          for (int k = 0; k < K; ++k) acc[k][j] += v[k];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) sm[(k * C + j) * 256u + threadIdx.x] = acc[k][j];
    __syncthreads();
    if (w == 0u) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const unsigned o = ob + 32u * j + lane;
          float s = sm[(k * C + j) * 256u + lane];
#pragma unroll
          for (unsigned q = 1; q < 8u; ++q) s += sm[(k * C + j) * 256u + q * 32u + lane];
          if (o < N) parts[k][t * N + o] = s;
        }
    }
    __syncthreads();
  }
}

template <class L>
__device__ __noinline__ void latch_p4a_full(const L leaf, float* part, unsigned S, unsigned CH, unsigned NC,
                                            unsigned rot, float* sm) {
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < NC; it += gridDim.x) {
    const unsigned base = it * CH + threadIdx.x;
    float acc = 0.0f;
#pragma unroll 8
    for (unsigned j = 0; j < CH; j += 256u) {
      const unsigned s = base + j;
      if (s < S) acc += leaf(s);
    }
    acc = s101h::warp_sum(acc);
    if (lane == 0u) sm[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {
      float total = sm[0];
#pragma unroll
      for (unsigned k = 1; k < 8u; ++k) total += sm[k];
      part[it] = total;
    }
    __syncthreads();
  }
}

template <class L>
__device__ __noinline__ float latch_p4a_direct(const L leaf, unsigned S, float* sm) {
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  float acc = 0.0f;
  for (unsigned s = threadIdx.x; s < S; s += 256u) acc += leaf(s);
  acc = s101h::warp_sum(acc);
  if (lane == 0u) sm[w] = acc;
  __syncthreads();
  // Frozen second stage (generic CTA tree): warp butterfly over the 8 warp sums with zero lanes; exact in lane 0.
  float total = 0.0f;
  if (threadIdx.x < 32u) {
    total = lane < 8u ? sm[lane] : 0.0f;
    total = s101h::warp_sum(total);
  }
  __syncthreads();
  return total;
}

__device__ __noinline__ void latch_p4a_colfin(const float* part, float* out, unsigned N, unsigned T, unsigned rot,
                                              float div) {
  const unsigned g = ((blockIdx.x + rot) % gridDim.x) * blockDim.x + threadIdx.x;
  for (unsigned o = g; o < N; o += gridDim.x * blockDim.x) {
    float s = part[o];
    for (unsigned k = 1; k < T; ++k) s += part[k * N + o];
    out[o] = div != 0.0f ? s / div : s;
  }
}

__device__ __noinline__ void latch_p4a_fullfin(const float* part, float* out, unsigned NC, unsigned rot, float div) {
  if (blockIdx.x == rot % gridDim.x && threadIdx.x < 32u) {
    float s = 0.0f;
    for (unsigned k = threadIdx.x; k < NC; k += 32u) s += part[k];
    s = s101h::warp_sum(s);
    if (threadIdx.x == 0u) out[0] = div != 0.0f ? s / div : s;
  }
}
#endif
'''

# Inline variant of the same templates (measured: non-inlined calls in the 255-register program cost more than
# they save; gpu-05). Identical bodies, distinct names, __forceinline__.
INLINE_HEADER = (OUTLINED_HEADER.replace('LATCH_P4A_DEFINED', 'LATCH_P4A_INLINE_DEFINED')
                 .replace('__noinline__', '__forceinline__')
                 .replace('latch_p4a_', 'latch_p4i_'))

PF_HEADER = r'''
#ifndef LATCH_PF_DEFINED
#define LATCH_PF_DEFINED
__device__ __forceinline__ float latch_pf(const float* p) {
  asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
  return 0.0f;
}
#endif
'''

# cp.async-staged column body: every memory load of the leaf is issued as a 4-byte cp.async into the lane's own
# ring slot S row steps ahead (no registers held while in flight); the compute functor reads the slots in the same
# order. Each lane reads only what it staged, so only cp.async.wait_group orders it. Same leaf arithmetic, same
# per-lane row order and CTA combination as latch_p4i_column, so results are bitwise identical to it.
STAGED_HEADER = r'''
#ifndef LATCH_P4S_DEFINED
#define LATCH_P4S_DEFINED
__device__ __forceinline__ float latch_cpa(float* dst, const float* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src)
               : "memory");
  return 0.0f;
}
template <class I, class L, int K, int C, int S, int NL>
__device__ __forceinline__ void latch_p4s_column(const I issue, const L leaf, float* p0, float* p1, float* p2,
                                                 float* p3, unsigned E, unsigned N, unsigned T, unsigned RT,
                                                 unsigned B, unsigned rot, float* sm) {
  float* const parts[4] = {p0, p1, p2, p3};
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  float* ring = sm + w * (S * NL * C * 32);
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < T * B; it += gridDim.x) {
    const unsigned t = it / B, ob = (it % B) * (32u * C);
    const unsigned r0 = t * RT, r1 = min(r0 + RT, E);
    const unsigned nk = r1 > r0 + w ? (r1 - r0 - w + 7u) / 8u : 0u;
    float acc[K][C];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) acc[k][j] = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) {
      if ((unsigned)s < nk) {
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const unsigned o = ob + 32u * j + lane;
          if (o < N) issue(r0 + w + 8u * s, o, ring + (s * NL * C + j) * 32 + lane);
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (unsigned q = 0; q < nk; ++q) {
      asm volatile("cp.async.wait_group %0;" :: "n"(S - 1) : "memory");
      const unsigned slot = q % S;
#pragma unroll
      for (int j = 0; j < C; ++j) {
        const unsigned o = ob + 32u * j + lane;
        if (o < N) {
          float v[K];
          leaf(ring + (slot * NL * C + j) * 32 + lane, v);
#pragma unroll
          for (int k = 0; k < K; ++k) acc[k][j] += v[k];
        }
      }
      const unsigned next = q + S;
      if (next < nk) {
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const unsigned o = ob + 32u * j + lane;
          if (o < N) issue(r0 + w + 8u * next, o, ring + (slot * NL * C + j) * 32 + lane);
        }
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) sm[(k * C + j) * 256u + threadIdx.x] = acc[k][j];
    __syncthreads();
    if (w == 0u) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const unsigned o = ob + 32u * j + lane;
          float s = sm[(k * C + j) * 256u + lane];
#pragma unroll
          for (unsigned q = 1; q < 8u; ++q) s += sm[(k * C + j) * 256u + q * 32u + lane];
          if (o < N) parts[k][t * N + o] = s;
        }
    }
    __syncthreads();
  }
}
#endif
'''
STAGE_BUDGET_BYTES = 65536


def stage_depth(loads, cols, budget=STAGE_BUDGET_BYTES):
    per_step = 8 * loads * cols * 32 * 4
    return max(2, min(16, budget // per_step))

# Vectorized column body: lane l owns 4 adjacent columns of each 128-column group (one 16-byte load per contiguous
# operand and row; row-broadcast operands stay scalar). Same per-column row order and warp-order CTA combination as
# the scalar body; only the column-to-lane assignment and item tiling differ.
VECTOR_HEADER = r'''
#ifndef LATCH_P4V_DEFINED
#define LATCH_P4V_DEFINED
template <class L, int K, int C>
__device__ __forceinline__ void latch_p4v_column(const L leaf, float* p0, float* p1, float* p2, float* p3, unsigned E,
                                                 unsigned N, unsigned T, unsigned RT, unsigned B, unsigned rot,
                                                 float* sm) {
  float* const parts[4] = {p0, p1, p2, p3};
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < T * B; it += gridDim.x) {
    const unsigned t = it / B, ob = (it % B) * (128u * C);
    const unsigned r0 = t * RT, r1 = min(r0 + RT, E);
    float acc[K][C][4];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[k][j][e] = 0.0f;
#pragma unroll 4
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {
#pragma unroll
      for (int j = 0; j < C; ++j) {
        const unsigned o = ob + 128u * j + 4u * lane;
        if (o < N) {
          float v[K][4];
          leaf(pr, o, v);
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[k][j][e] += v[k][e];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) sm[((k * C + j) * 4 + e) * 256u + threadIdx.x] = acc[k][j][e];
    __syncthreads();
    if (w == 0u) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < C; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const unsigned o = ob + 128u * j + 4u * lane + e;
            float s = sm[((k * C + j) * 4 + e) * 256u + lane];
#pragma unroll
            for (unsigned q = 1; q < 8u; ++q) s += sm[((k * C + j) * 4 + e) * 256u + q * 32u + lane];
            if (o < N) parts[k][t * N + o] = s;
          }
    }
    __syncthreads();
  }
}
#endif
'''


def _index_value(text, pr, o):
    import re
    expression = re.sub(r'(\d+)u\b', r'\1', text).replace('(unsigned)', '').replace('/', '//')
    return eval(expression, {'__builtins__': {}}, {'pr': pr, 'o': o})


def classify_load(index_text, extent, results):
    """'vec' if the index is unit-stride and 4-aligned in o for o = 4m + e, 'bcast' if independent of e, else None."""
    samples = [(pr, o4) for pr in sorted({0, 1, 7, 65, extent - 1}) for o4 in sorted({0, 4, 128, results - 4})]
    try:
        values = [[_index_value(index_text, pr, o4 + e) for e in range(4)] for pr, o4 in samples]
    except Exception:
        return None
    if all(v[e] == v[0] + e and v[0] % 4 == 0 for v in values for e in range(4)):
        return 'vec'
    if all(v[e] == v[0] for v in values for e in range(4)):
        return 'bcast'
    return None

VECTOR_FULL_HEADER = r'''
#ifndef LATCH_P4VF_DEFINED
#define LATCH_P4VF_DEFINED
template <class L>
__device__ __forceinline__ void latch_p4v_full(const L leaf, float* part, unsigned S, unsigned CH, unsigned NC,
                                               unsigned rot, float* sm) {
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < NC; it += gridDim.x) {
    const unsigned base = it * CH;
    float acc = 0.0f;
#pragma unroll 4
    for (unsigned j = 4u * threadIdx.x; j < CH; j += 1024u) {
      const unsigned s = base + j;
      if (s < S) {
        float v[4];
        leaf(s, v);
        acc += v[0]; acc += v[1]; acc += v[2]; acc += v[3];
      }
    }
    acc = s101h::warp_sum(acc);
    if (lane == 0u) sm[w] = acc;
    __syncthreads();
    if (threadIdx.x == 0u) {
      float total = sm[0];
#pragma unroll
      for (unsigned k = 1; k < 8u; ++k) total += sm[k];
      part[it] = total;
    }
    __syncthreads();
  }
}
#endif
'''


def classify_full_load(index_text, extent):
    samples = sorted({0, 4, 1024, 4096, extent - 4})
    try:
        import re
        expression = re.sub(r'(\d+)u\b', r'\1', index_text).replace('(unsigned)', '').replace('/', '//')
        values = [[eval(expression, {'__builtins__': {}}, {'s': s + e}) for e in range(4)] for s in samples]
    except Exception:
        return None
    if all(v[e] == v[0] + e and v[0] % 4 == 0 for v in values for e in range(4)):
        return 'vec'
    if all(v[e] == v[0] for v in values for e in range(4)):
        return 'bcast'
    return None

# Last-arriver variants of the column bodies: the CTA completing a column block's last tile sums that block's tile
# partials in tile order (the finalize item's order, so results equal the two-phase form bit for bit) and stores the
# K results. Counters (/*P2_CNT*/, one per column block) start at zero and are restored by every last arriver;
# partials (K x T x N) live in the phase's split scratch (/*P2_PART*/). Only partial writers fence their stores.
LAST_HEADER = r'''
#ifndef LATCH_P4L_DEFINED
#define LATCH_P4L_DEFINED
template <int K, int G, int W>
__device__ __forceinline__ void latch_p4l_finish(float* const* outs, float div, unsigned* cnt, const float* part,
                                                 unsigned N, unsigned T, unsigned t, unsigned b, unsigned ob,
                                                 unsigned lane, unsigned w) {
  // G columns per lane-group step, W adjacent columns per lane; ob = first column of this block.
  __shared__ bool latch_last;
  __syncthreads();
  if (threadIdx.x == 0u) {
    const unsigned old = atomicAdd(cnt + b, 1u);
    latch_last = old == T - 1u;
    if (latch_last) atomicExch(cnt + b, 0u);
  }
  __syncthreads();
  if (latch_last && w == 0u) {
    __threadfence();
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < G; ++j)
#pragma unroll
        for (int e = 0; e < W; ++e) {
          const unsigned o = ob + (32u * W) * j + W * lane + e;
          if (o < N) {
            float s = __ldcg(part + (size_t)k * T * N + o);
            for (unsigned q = 1; q < T; ++q) s += __ldcg(part + (size_t)k * T * N + (size_t)q * N + o);
            outs[k][o] = div != 0.0f ? s / div : s;
          }
        }
  }
  __syncthreads();
}

template <class L, int K, int C>
__device__ __forceinline__ void latch_p4l_vector(const L leaf, float* o0, float* o1, float* o2, float* o3, float div,
                                                 unsigned* cnt, float* part, unsigned E, unsigned N, unsigned T,
                                                 unsigned RT, unsigned B, unsigned rot, float* sm) {
  float* const outs[4] = {o0, o1, o2, o3};
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < T * B; it += gridDim.x) {
    const unsigned t = it / B, b = it % B, ob = b * (128u * C);
    const unsigned r0 = t * RT, r1 = min(r0 + RT, E);
    float acc[K][C][4];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[k][j][e] = 0.0f;
#pragma unroll 4
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {
#pragma unroll
      for (int j = 0; j < C; ++j) {
        const unsigned o = ob + 128u * j + 4u * lane;
        if (o < N) {
          float v[K][4];
          leaf(pr, o, v);
#pragma unroll
          for (int k = 0; k < K; ++k)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[k][j][e] += v[k][e];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) sm[((k * C + j) * 4 + e) * 256u + threadIdx.x] = acc[k][j][e];
    __syncthreads();
    if (w == 0u) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < C; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const unsigned o = ob + 128u * j + 4u * lane + e;
            float s = sm[((k * C + j) * 4 + e) * 256u + lane];
#pragma unroll
            for (unsigned q = 1; q < 8u; ++q) s += sm[((k * C + j) * 4 + e) * 256u + q * 32u + lane];
            if (o < N) part[(size_t)k * T * N + (size_t)t * N + o] = s;
          }
      __threadfence();
    }
    latch_p4l_finish<K, C, 4>(outs, div, cnt, part, N, T, t, b, ob, lane, w);
  }
}

template <class L, int K, int C>
__device__ __forceinline__ void latch_p4l_scalar(const L leaf, float* o0, float* o1, float* o2, float* o3, float div,
                                                 unsigned* cnt, float* part, unsigned E, unsigned N, unsigned T,
                                                 unsigned RT, unsigned B, unsigned rot, float* sm) {
  float* const outs[4] = {o0, o1, o2, o3};
  const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
  for (unsigned it = (blockIdx.x + rot) % gridDim.x; it < T * B; it += gridDim.x) {
    const unsigned t = it / B, b = it % B, ob = b * (32u * C);
    const unsigned r0 = t * RT, r1 = min(r0 + RT, E);
    float acc[K][C];
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) acc[k][j] = 0.0f;
#pragma unroll 8
    for (unsigned pr = r0 + w; pr < r1; pr += 8u) {
#pragma unroll
      for (int j = 0; j < C; ++j) {
        const unsigned o = ob + 32u * j + lane;
        if (o < N) {
          float v[K];
          leaf(pr, o, v);
#pragma unroll
          for (int k = 0; k < K; ++k) acc[k][j] += v[k];
        }
      }
    }
#pragma unroll
    for (int k = 0; k < K; ++k)
#pragma unroll
      for (int j = 0; j < C; ++j) sm[(k * C + j) * 256u + threadIdx.x] = acc[k][j];
    __syncthreads();
    if (w == 0u) {
#pragma unroll
      for (int k = 0; k < K; ++k)
#pragma unroll
        for (int j = 0; j < C; ++j) {
          const unsigned o = ob + 32u * j + lane;
          float s = sm[(k * C + j) * 256u + lane];
#pragma unroll
          for (unsigned q = 1; q < 8u; ++q) s += sm[(k * C + j) * 256u + q * 32u + lane];
          if (o < N) part[(size_t)k * T * N + (size_t)t * N + o] = s;
        }
      __threadfence();
    }
    latch_p4l_finish<K, C, 1>(outs, div, cnt, part, N, T, t, b, ob, lane, w);
  }
}
#endif
'''
