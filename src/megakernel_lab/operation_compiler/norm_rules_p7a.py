# Adapted from frozen P7-A f36b4737cee732e6915d11e233bcc068126b3e64; private scheduling/resource protocol.
"""P-OPS3-C normalization rules: exact row/column reduction engines and row-consumer attachment.

Normalization layers lower to a reduction over a row (RMSNorm statistics, the row term of its derivative),
a reduction over the leading rows (the normalization weight gradient) and elementwise consumers that broadcast
the row statistic back over the row. The fused schedule previously gave every non-matrix heavy operation its own
grid phase: a CTA-per-row reduction, then a separate elementwise phase that re-read the row from memory, and the
weight-gradient reduction serialized next to it with lanes striding across rows (uncoalesced).

These rules change only scheduling and the assignment of work to threads, never the arithmetic:

* ``row engine``: a warp owns one output row. Lane l accumulates the leaves of the frozen CTA threads
  t = 32 w + l (w = 0..7) exactly as the frozen loop does (``acc = 0; acc += leaf(t + 256 j)``), then replays the
  frozen tree: an xor butterfly per w (the frozen per-warp ``warp_sum``) and the second-stage butterfly over the
  eight warp partials with zero lanes. The stored statistic is bitwise identical to the frozen CTA tree.
* ``column engine``: a reduction over leading rows (kept trailing columns). Lanes own columns, so every load is a
  contiguous segment; each thread evaluates the lane-0 butterfly subtree of four frozen threads per group
  (levels 16 and 8 in registers), levels 4/2/1 by register and two xor shuffles, and the second stage from shared
  memory with the frozen zero lanes. Bitwise identical to the frozen tree.
* ``row attachment``: an elementwise operation over the same [rows, extent] index space that reads the row
  statistic only at its own row (and otherwise only earlier values, graph inputs or members already attached) is
  evaluated by the owning warp right after the statistic, from a register. A same-shaped operation in the
  statistic's phase that shares a loaded operand with the reduction leaf may attach too (sibling), so its
  consumers can follow. Attached expressions are the unchanged per-operation expressions.
* A row reduction and one column reduction may share a phase on disjoint block ranges.

Dispatch uses only reduction axes, extents, layouts and dependencies; no model names or layer roles.
"""
from __future__ import annotations

import re

ROW_MIN_ROWS = 64          # fewer rows keep the CTA tree (a warp per row would leave the grid idle)
ROW_MAX_LEAVES = 16        # leaves per frozen thread (extent <= 4096)
COL_MIN_COLUMNS = 32
COL_MAX_LEAVES = 16        # rows per frozen thread for the column engine (reduced extent <= 4096)
COL_GROUP = 8              # columns per CTA iteration (one 32-byte segment per row and lane group)
GRID_CAP = 128


def position(src_shape, src_strides, dims, kept, rvar, ovar):
    """Storage offset of source element (reduced flat index rvar, kept flat index ovar); frozen formula.

    Emitted in 32-bit unsigned arithmetic (callers admit only sources below 2**31 elements)."""
    from .lower import _numel
    reduction_shape = tuple(src_shape[i] for i in dims)
    kept_shape = tuple(src_shape[i] for i in kept)
    terms = []
    for sequence, subshape, index in [(dims, reduction_shape, rvar), (kept, kept_shape, ovar)]:
        for j, axis in enumerate(sequence):
            inner, extent, stride = _numel(subshape[j+1:]), src_shape[axis], src_strides[axis]
            coord = index if inner == 1 else f"({index} / {inner}u)"
            if j != 0:
                coord = f"({coord} % {extent}u)"
            terms.append(coord if stride == 1 else f"{coord} * {stride}u")
    return " + ".join(terms) or "0u"


def _span(shape, strides):
    return 1 + sum((d - 1) * s for d, s in zip(shape, strides))


def warps_per_row(rows, blocks):
    """Warps cooperating on one row so a block's eight warps are busy in one round (power of two)."""
    per_block = -(-rows // max(1, blocks))
    group = 1
    while group < 8 and group < per_block:
        group *= 2
    return 8 // group


def row_eligible(src_shape, src_strides, dims, results, extent):
    """Reduced axes are the trailing axes, innermost unit stride; enough rows; bounded extent."""
    n = len(src_shape)
    if _span(src_shape, src_strides) >= 2**31:
        return False
    if not dims or list(dims) != list(range(n - len(dims), n)):
        return False
    if src_strides[-1] != 1 or extent <= 32 or -(-extent // 256) > ROW_MAX_LEAVES or results < ROW_MIN_ROWS:
        return False
    # Reduced axes merge into one contiguous run (coalesced rows).
    stride = 1
    for axis in reversed(dims):
        if src_strides[axis] != stride:
            return False
        stride *= src_shape[axis]
    return True


def column_eligible(src_shape, src_strides, dims, results, extent):
    """Reduced axes are the leading axes; kept trailing axes are contiguous with unit innermost stride."""
    n = len(src_shape)
    if _span(src_shape, src_strides) >= 2**31:
        return False
    if not dims or list(dims) != list(range(len(dims))) or len(dims) == n:
        return False
    if extent <= 32 or -(-extent // 256) > COL_MAX_LEAVES or results < COL_MIN_COLUMNS:
        return False
    stride = 1
    for axis in reversed(range(len(dims), n)):
        if src_strides[axis] != stride:
            return False
        stride *= src_shape[axis]
    return True


def row_code(*, out, rows, extent, leaf, rounded, scale, blocks=GRID_CAP):
    """Row engine. W warps share a row; warp k of a row group owns frozen threads 32 w + lane for its 8/W values
    of w. `leaf` reads source element (prow, pcol). Markers: /*ROW_LO*/, /*ROW_MEMBERS*/ (members see g, lane,
    latch_w0, latch_nw; one unrolled element loop per owned w)."""
    leaves = -(-extent // 256)
    # P6-A: /*ROW_RB*/ rows per warp slot and iteration (default 1 = the P-OPS3-C engine). Every leaf load of the
    # RB rows is issued before their butterflies; each row keeps its own frozen tree, partial buffer and members.
    return f'''{{ // P-OPS3-C row engine: {rows} rows x {extent} (frozen CTA tree, /*ROW_W*/ warps per row)
  const unsigned latch_lo = /*ROW_LO*/0u;
  if (blockIdx.x >= latch_lo) {{
    constexpr unsigned latch_wr = /*ROW_WR*/8u, latch_nw = 8u / latch_wr, latch_rb = 8u / latch_wr;
    constexpr unsigned latch_rbn = /*ROW_RB*/1u;
    const unsigned latch_nb = gridDim.x - latch_lo;
    const unsigned lane = threadIdx.x & 31u, latch_slot = (threadIdx.x >> 5) / latch_wr;
    const unsigned latch_w0 = ((threadIdx.x >> 5) % latch_wr) * latch_nw;
    unsigned latch_buf = 0u;
    for (unsigned latch_base = (blockIdx.x - latch_lo) * latch_rb * latch_rbn; latch_base < {rows}u; latch_base += latch_nb * latch_rb * latch_rbn) {{
      float latch_acc[latch_rbn][latch_nw];
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        const unsigned prow = g;
#pragma unroll
        for (unsigned v = 0; v < latch_nw; ++v) {{
          float acc = 0.0f;
          if (g < {rows}u) {{
#pragma unroll
            for (int j = 0; j < {leaves}; ++j) {{
              const unsigned pcol = 32u * (latch_w0 + v) + lane + 256u * j;
              if (pcol < {extent}u) acc += {leaf};
            }}
          }}
          latch_acc[latch_r][v] = acc;
        }}
      }}
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        float* latch_part = smem + (latch_buf * latch_rbn + latch_r) * 64u + latch_slot * 8u;
        if (g < {rows}u) {{
#pragma unroll
          for (unsigned v = 0; v < latch_nw; ++v) {{
            const float acc = s101h::warp_sum(latch_acc[latch_r][v]);
            if (lane == 0u) latch_part[latch_w0 + v] = acc;
          }}
        }}
      }}
      if (latch_wr == 1u) __syncwarp(); else __syncthreads();  // P6-A: a warp owns its rows' partials
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        float* latch_part = smem + (latch_buf * latch_rbn + latch_r) * 64u + latch_slot * 8u;
        if (g < {rows}u) {{
          float second = lane < 8u ? latch_part[lane] : 0.0f;
          second = s101h::warp_sum(second);
          if (latch_w0 == 0u && lane == 0u) {out}[g] = {rounded('second' + scale)};
/*ROW_MEMBERS*/
        }}
      }}
      latch_buf ^= 1u;
    }}
  }}
}}'''


def column_code(*, out, columns, extent, leaf, rounded, scale, group=None):
    """Coalesced exact column engine. `leaf` reads source element (reduced pr, kept o). Marker: /*COL_SPAN*/.

    group 8: lanes = 4 sub-groups x 8 columns, each thread two lane-0 subtrees of four frozen threads.
    group 4: lanes = 8 sub-groups x 4 columns, each thread one subtree (half the leaves, twice the blocks)."""
    COL_GROUP = group or globals()['COL_GROUP']
    if COL_GROUP == 4:
        return _column_code4(out=out, columns=columns, extent=extent, leaf=leaf, rounded=rounded, scale=scale)
    leaves = -(-extent // 256)
    groups = -(-columns // COL_GROUP)
    tail = rounded('s0' + scale)
    return f'''{{ // P-OPS3-C column engine: {columns} columns x {extent} rows (frozen CTA tree, lanes over columns)
  const unsigned latch_cn = /*COL_SPAN*/min({groups}u, gridDim.x);
  if (blockIdx.x < latch_cn) {{
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5, sub = lane >> 3, cl = lane & 7u;
    for (unsigned cg = blockIdx.x; cg < {groups}u; cg += latch_cn) {{
      const unsigned o = cg * {COL_GROUP}u + cl;
      float t2[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {{
        float x[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {{
          const unsigned t = 32u * w + sub + 4u * h + 8u * q;
          float acc = 0.0f;
#pragma unroll
          for (int j = 0; j < {leaves}; ++j) {{
            const unsigned pr = t + 256u * j;
            if (pr < {extent}u && o < {columns}u) acc += {leaf};
          }}
          x[q] = acc;
        }}
        t2[h] = (x[0] + x[2]) + (x[1] + x[3]);
      }}
      float v = t2[0] + t2[1];
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      if (sub == 0u) smem[w * {COL_GROUP}u + cl] = v;
      __syncthreads();
      if (threadIdx.x < {COL_GROUP}u && cg * {COL_GROUP}u + threadIdx.x < {columns}u) {{
        float s[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] = (smem[l * {COL_GROUP} + threadIdx.x] + 0.0f) + 0.0f;
        const float s0 = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
        {out}[cg * {COL_GROUP}u + threadIdx.x] = {tail};
      }}
      __syncthreads();
    }}
  }}
}}'''


def _column_code4(*, out, columns, extent, leaf, rounded, scale):
    leaves = -(-extent // 256)
    groups = -(-columns // 4)
    tail = rounded('s0' + scale)
    return f'''{{ // P-OPS3-C column engine: {columns} columns x {extent} rows (frozen CTA tree, 4 columns per CTA)
  const unsigned latch_cn = /*COL_SPAN*/min({groups}u, gridDim.x);
  if (blockIdx.x < latch_cn) {{
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5, sub = lane >> 2, cl = lane & 3u;
    for (unsigned cg = blockIdx.x; cg < {groups}u; cg += latch_cn) {{
      const unsigned o = cg * 4u + cl;
      float x[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {{
        const unsigned t = 32u * w + sub + 8u * q;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < {leaves}; ++j) {{
          const unsigned pr = t + 256u * j;
          if (pr < {extent}u && o < {columns}u) acc += {leaf};
        }}
        x[q] = acc;
      }}
      float v = (x[0] + x[2]) + (x[1] + x[3]);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      if (sub == 0u) smem[w * 4u + cl] = v;
      __syncthreads();
      if (threadIdx.x < 4u && cg * 4u + threadIdx.x < {columns}u) {{
        float s[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] = (smem[l * 4 + threadIdx.x] + 0.0f) + 0.0f;
        const float s0 = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
        {out}[cg * 4u + threadIdx.x] = {tail};
      }}
      __syncthreads();
    }}
  }}
}}'''


def normalized_index(text):
    return re.sub(r"[\s()]", "", text)


def row_index_text(extent):
    return f"i/{extent}u"


OUTLINED = r"""
#ifndef LATCH_OPS_OUTLINED
#define LATCH_OPS_OUTLINED
__device__ __noinline__ float latch_powf(float x, float y) { return powf(x, y); }
__device__ __noinline__ float latch_expf(float x) { return expf(x); }
__device__ __noinline__ float latch_logf(float x) { return logf(x); }
__device__ __noinline__ float latch_fdiv(float a, float b) { return a / b; }
__device__ __noinline__ float latch_sigmoid(float x) { return (1.0f / (1.0f + expf(-x))); }
__device__ __noinline__ float latch_silu(float x) { return (x / (1.0f + expf(-x))); }
__device__ __noinline__ float latch_silu_backward(float g, float x) {
  const float s = (1.0f / (1.0f + expf(-x)));
  return (g * s * (1.0f + x * (1.0f - s)));
}
#endif
"""
_OUTLINE = re.compile(r"(?<![\w:.])(powf|expf|logf)\(")


def outline(body):
    """Route generated libdevice transcendental calls to their single out-of-line copies."""
    count = len(_OUTLINE.findall(body)) + body.count("latch_fdiv(") + body.count("latch_sigmoid(") + body.count("latch_silu")
    return _OUTLINE.sub(lambda m: "latch_" + m.group(1) + "(", body), count


FACTORABLE = {"add", "sub", "mul", "div", "neg", "pow", "rsqrt", "sqrt", "exp", "expm1", "log1p", "log", "reciprocal",
              "tanh", "sigmoid", "silu", "relu", "abs", "clone", "_to_copy"}


PURE_CALLS = ("latch_sigmoid", "latch_silu_backward", "latch_silu", "latch_fdiv", "powf", "expf", "logf")


def cse_calls(expression, prefix):
    """Evaluate each repeated pure outlined call of one element expression once (its value is unchanged)."""
    prelude = []
    while True:
        counts = {}
        for name in PURE_CALLS:
            for match in re.finditer(r"(?<![\w:.])" + re.escape(name) + r"\(", expression):
                depth, k = 0, match.end() - 1
                while k < len(expression):
                    depth += expression[k] == "("
                    depth -= expression[k] == ")"
                    if depth == 0:
                        break
                    k += 1
                call = expression[match.start():k + 1]
                counts[call] = counts.get(call, 0) + 1
        repeated = [c for c, n in counts.items() if n > 1]
        if not repeated:
            return prelude, expression
        call = min(repeated, key=len)            # innermost first; outer calls are found again afterwards
        temp = f"{prefix}_{len(prelude)}"
        prelude.append(f"const float {temp} = {call};")
        expression = expression.replace(call, temp)


def row_code_multi(*, rows, extent, entries):
    """P6-A fused row engine: several independent statistics of the same [rows, extent] rows in one pass. Each
    statistic keeps its own accumulator, frozen per-warp butterflies and second stage (bitwise per statistic, as
    row_code). entries: [(leaf, store_statement)], the store using `second`. Markers /*ROW_MEMBERS_k*/ receive each
    engine's members after every statistic of the row group is stored (block barrier in between)."""
    leaves = -(-extent // 256)
    K = len(entries)
    acc_lines = "\n".join(
        f"              if (pcol < {extent}u) latch_acc[latch_r][{k}][v] += {leaf};" for k, (leaf, _) in enumerate(entries))
    stores = "\n".join(
        f"          {{ const float second = latch_sec[latch_r][{k}]; if (latch_w0 == 0u && lane == 0u) {store} }}"
        for k, (_, store) in enumerate(entries))
    members = "\n".join(
        f"          {{ const float second = latch_sec[latch_r][{k}];\n/*ROW_MEMBERS_{k}*/\n          }}"
        for k in range(K))
    return f'''{{ // P6-A fused row engine: {K} statistics of {rows} rows x {extent} (frozen CTA tree each, /*ROW_W*/ warps per row)
  const unsigned latch_lo = /*ROW_LO*/0u;
  if (blockIdx.x >= latch_lo) {{
    constexpr unsigned latch_wr = /*ROW_WR*/8u, latch_nw = 8u / latch_wr, latch_rb = 8u / latch_wr;
    constexpr unsigned latch_rbn = /*ROW_RB*/1u;
    const unsigned latch_nb = gridDim.x - latch_lo;
    const unsigned lane = threadIdx.x & 31u, latch_slot = (threadIdx.x >> 5) / latch_wr;
    const unsigned latch_w0 = ((threadIdx.x >> 5) % latch_wr) * latch_nw;
    unsigned latch_buf = 0u;
    for (unsigned latch_base = (blockIdx.x - latch_lo) * latch_rb * latch_rbn; latch_base < {rows}u; latch_base += latch_nb * latch_rb * latch_rbn) {{
      float latch_acc[latch_rbn][{K}][latch_nw];
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        const unsigned prow = g;
#pragma unroll
        for (unsigned v = 0; v < latch_nw; ++v) {{
#pragma unroll
          for (int k = 0; k < {K}; ++k) latch_acc[latch_r][k][v] = 0.0f;
          if (g < {rows}u) {{
#pragma unroll
            for (int j = 0; j < {leaves}; ++j) {{
              const unsigned pcol = 32u * (latch_w0 + v) + lane + 256u * j;
{acc_lines}
            }}
          }}
        }}
      }}
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        if (g < {rows}u) {{
#pragma unroll
          for (int k = 0; k < {K}; ++k) {{
            float* latch_part = smem + ((latch_buf * latch_rbn + latch_r) * {K} + k) * 64u + latch_slot * 8u;
#pragma unroll
            for (unsigned v = 0; v < latch_nw; ++v) {{
              const float acc = s101h::warp_sum(latch_acc[latch_r][k][v]);
              if (lane == 0u) latch_part[latch_w0 + v] = acc;
            }}
          }}
        }}
      }}
      if (latch_wr == 1u) __syncwarp(); else __syncthreads();  // P6-A: a warp owns its rows' partials
      float latch_sec[latch_rbn][{K}];
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
#pragma unroll
        for (int k = 0; k < {K}; ++k) {{
          float* latch_part = smem + ((latch_buf * latch_rbn + latch_r) * {K} + k) * 64u + latch_slot * 8u;
          float second = (g < {rows}u && lane < 8u) ? latch_part[lane] : 0.0f;
          latch_sec[latch_r][k] = s101h::warp_sum(second);
        }}
        if (g < {rows}u) {{
{stores}
        }}
      }}
      if (latch_wr == 1u) __syncwarp(); else __syncthreads();  // P6-A: a warp owns its rows' partials
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        if (g < {rows}u) {{
{members}
        }}
      }}
      latch_buf ^= 1u;
    }}
  }}
}}'''


def row_code_chain(*, rows, extent, entries):
    """P6-A chained row engine (one warp per row only): statistic k's leaf may read statistics 0..k-1 of the same
    row, which the same warp stored before a __syncwarp. Each statistic keeps its frozen per-warp butterflies and
    second stage (bitwise per statistic, as row_code). entries: [(leaf, store_statement)], the store using `second`.
    Markers /*ROW_MEMBERS_k*/ receive engine k's members right after its statistic is stored."""
    leaves = -(-extent // 256)
    blocks = []
    for k, (leaf, store) in enumerate(entries):
        blocks.append(f'''
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        const unsigned prow = g;
#pragma unroll
        for (unsigned v = 0; v < latch_nw; ++v) {{
          float acc = 0.0f;
          if (g < {rows}u) {{
#pragma unroll
            for (int j = 0; j < {leaves}; ++j) {{
              const unsigned pcol = 32u * (latch_w0 + v) + lane + 256u * j;
              if (pcol < {extent}u) acc += {leaf};
            }}
          }}
          latch_acc[latch_r][v] = acc;
        }}
      }}
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        float* latch_part = smem + ((latch_buf * latch_rbn + latch_r) * {len(entries)} + {k}) * 64u + latch_slot * 8u;
#pragma unroll
        for (unsigned v = 0; v < latch_nw; ++v) {{
          const float acc = s101h::warp_sum(latch_acc[latch_r][v]);
          if (lane == 0u) latch_part[latch_w0 + v] = acc;
        }}
      }}
      __syncwarp();
#pragma unroll
      for (unsigned latch_r = 0; latch_r < latch_rbn; ++latch_r) {{
        const unsigned g = latch_base + latch_r * latch_rb + latch_slot;
        float* latch_part = smem + ((latch_buf * latch_rbn + latch_r) * {len(entries)} + {k}) * 64u + latch_slot * 8u;
        float second = lane < 8u ? latch_part[lane] : 0.0f;
        second = s101h::warp_sum(second);
        if (g < {rows}u) {{
          if (lane == 0u) {store}
/*ROW_MEMBERS_{k}*/
        }}
      }}
      __syncwarp();''')
    body = "".join(blocks)
    return f'''{{ // P6-A chained row engine: {len(entries)} statistics of {rows} rows x {extent} (one warp per row)
  const unsigned latch_lo = /*ROW_LO*/0u;
  if (blockIdx.x >= latch_lo) {{
    constexpr unsigned latch_wr = /*ROW_WR*/8u, latch_nw = 8u / latch_wr, latch_rb = 8u / latch_wr;
    static_assert(latch_wr == 1u, "chained row engines need one warp per row");
    constexpr unsigned latch_rbn = /*ROW_RB*/1u;
    const unsigned latch_nb = gridDim.x - latch_lo;
    const unsigned lane = threadIdx.x & 31u, latch_slot = (threadIdx.x >> 5) / latch_wr;
    const unsigned latch_w0 = ((threadIdx.x >> 5) % latch_wr) * latch_nw;
    unsigned latch_buf = 0u;
    for (unsigned latch_base = (blockIdx.x - latch_lo) * latch_rb * latch_rbn; latch_base < {rows}u; latch_base += latch_nb * latch_rb * latch_rbn) {{
      float latch_acc[latch_rbn][latch_nw];
{body}
      latch_buf ^= 1u;
    }}
  }}
}}'''
