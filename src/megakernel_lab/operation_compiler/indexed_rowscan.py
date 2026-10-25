"""LATCH-STEP-04 `rowscan_v1`: single-phase deterministic FP32 embedding gradients.

Generic lowering of `aten.embedding_dense_backward` for index lists and tables small
enough that every CTA can hold a row histogram in shared memory. Selected from shape,
dtype and layout only (see `supported`). One phase, no grid barrier, no atomics on
floating data, no inter-CTA communication:

1. every CTA builds the same integer row histogram of the N indices (padding
   occurrences excluded) in shared memory;
2. active rows become work units in row order: a light row (<= FMAX occurrences) is
   one full-width unit; a heavy row is split into column slices. Every CTA computes
   the same unit cost prefix and takes the units whose cost start falls in its window;
3. the CTA gathers the ordinals of occurrences in its row range in ascending order
   (each warp compacts a contiguous ordinal slice; slices concatenate in warp order);
4. a warp sums each unit: full-width units add occurrences in ascending ordinal order
   from +0.0 (bitwise the ordered policy); column slices accumulate occurrences
   lane-strided (match g -> lane g mod 32, ascending) and combine lanes by a fixed
   xor butterfly. The association depends only on (indices, N, D, constants);
5. rows without occurrences (including the padding row) are written as +0.0.

Declared arithmetic policy `reassociated_fp32_v1` (heavy slices); repeated runs are
bitwise identical. Work: O(G*(N + V) + N*D + V*D); shared memory V + N + small words.
"""
from __future__ import annotations

WARPS = 8
FMAX = 64          # occurrences summed in order by one full-width unit
BUDGET = 32        # per-lane vector loads targeted per column-slice unit
WSMAX = 8          # vectors per lane accumulated by a column-slice unit
HCOST = 4          # per-unit overhead in the balancing cost
UCAP = 512         # units materialized per round (rounds repeat for small grids)
SMEM_LIMIT = 200 * 1024

HELPERS = r"""#include <cassert>
// LATCH-STEP-04 rowscan_v1 helpers (FP32 lanes, no FMA).
__device__ __forceinline__ void rs_zero(float& a) { a = 0.0f; }
__device__ __forceinline__ void rs_zero(float4& a) { a = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
__device__ __forceinline__ void rs_add(float& a, const float b) { a = __fadd_rn(a, b); }
__device__ __forceinline__ void rs_add(float4& a, const float4 b) {
  a.x = __fadd_rn(a.x, b.x); a.y = __fadd_rn(a.y, b.y); a.z = __fadd_rn(a.z, b.z); a.w = __fadd_rn(a.w, b.w);
}
__device__ __forceinline__ float rs_xor(float a, int o) { return __shfl_xor_sync(0xffffffffu, a, o); }
__device__ __forceinline__ float4 rs_xor(float4 a, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, a.x, o), __shfl_xor_sync(0xffffffffu, a.y, o),
                     __shfl_xor_sync(0xffffffffu, a.z, o), __shfl_xor_sync(0xffffffffu, a.w, o));
}
// Block-wide exclusive scan over 256 threads; tmp holds >= 9 words.
__device__ __forceinline__ long long rs_scan(long long v, long long* tmp, long long& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  long long x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) { const long long y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
  if (lane == 31) tmp[warp] = x;
  __syncthreads();
  if (threadIdx.x == 0) { long long run = 0; for (int w = 0; w < 8; ++w) { const long long t = tmp[w]; tmp[w] = run; run += t; } tmp[8] = run; }
  __syncthreads();
  const long long result = tmp[warp] + x - v;
  total = tmp[8];
  __syncthreads();
  return result;
}
// Index of the (j+1)-th set bit of m (j < popc(m)).
__device__ __forceinline__ int rs_nth(unsigned m, int j) {
  for (int i = 0; i < j; ++i) m &= m - 1u;
  return __ffs(m) - 1;
}
"""


def _pad(x, unit=4):
    return -(-x // unit) * unit


def shared_bytes(n, rows):
    # cnt[V] int + units[UCAP] int4 + packed list[N] u32 + sorted ordinals[N] u16 + seg lengths + scan scratch
    return 4 * _pad(rows) + 16 * UCAP + 4 * _pad(n) + 2 * _pad(n, 8) + 4 * 8 + 8 * 18 + 64


def supported(n, rows, width):
    if n < 1 or rows < 1 or width < 1:
        return False, "empty index or row dimension"
    if n > 65535 or rows > 32768:
        return False, "rowscan packs 16-bit ordinals and in-range rows (N <= 65535, V <= 32768)"
    if shared_bytes(n, rows) > SMEM_LIMIT:
        return False, f"row histogram + occurrence list need {shared_bytes(n, rows)} shared bytes > {SMEM_LIMIT}"
    return True, "rowscan engine admitted"


def _unravel(shape, strides, var):
    terms, inner = [], 1
    for extent, stride in reversed(list(zip(shape, strides))):
        if extent != 1 and stride != 0:
            coordinate = f"((unsigned){var} / {inner}u % {extent}u)" if inner != 1 else f"((unsigned){var} % {extent}u)"
            terms.append(f"(long long){coordinate} * {stride}LL")
        inner *= extent
    return " + ".join(reversed(terms)) or "0LL"


def emit(name, grad, ids, lg, li, rows, padding, width, ident, debug_stop=99):
    """Return (phases, decision) for one embedding gradient."""
    n, V, D = li.numel, rows, width
    G, I, OUT = ident(grad), ident(ids), ident(name)
    id_at = _unravel(li.shape, li.strides, "o")
    grad_base = _unravel(lg.shape[:-1], lg.strides[:-1], "o")
    sc = lg.strides[-1]
    vector_ok = sc == 1 and D % 4 == 0 and all(s % 4 == 0 for s, e in zip(lg.strides[:-1], lg.shape[:-1]) if e != 1)
    smem = shared_bytes(n, V)
    seg = -(-n // WARPS)

    def body(w):
        T = "float4" if w == 4 else "float"
        DV = D // w                          # vectors per row
        J = -(-DV // 32)                     # vectors per lane in a full-width unit
        load = (lambda base, vec: f"reinterpret_cast<const float4*>({G} + ({base}))[{vec}]") if w == 4 else \
               (lambda base, vec: f"{G}[({base}) + (long long)({vec}) * {sc}LL]")
        return f"""{{
  __syncthreads();
  int* cnt_ = reinterpret_cast<int*>(smem);
  int4* units_ = reinterpret_cast<int4*>(cnt_ + {_pad(V)});
  unsigned* list_ = reinterpret_cast<unsigned*>(units_ + {UCAP});          // (row - rlo) << 16 | ordinal
  unsigned short* sorted_ = reinterpret_cast<unsigned short*>(list_ + {_pad(n)});
  int* seglen_ = reinterpret_cast<int*>(sorted_ + {_pad(n, 8)});
  long long* tmp_ = reinterpret_cast<long long*>(seglen_ + 8);
  int* meta_ = reinterpret_cast<int*>(tmp_ + 18);
  {T}* out_ = reinterpret_cast<{T}*>({OUT});
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned below_ = (1u << lane) - 1u;
  // (1) identical integer row histogram in every CTA (padding excluded); index loads batched.
  for (int r = tid; r < {V}; r += 256) cnt_[r] = 0;
  __syncthreads();
  for (int b0 = tid; b0 < {n}; b0 += 2048) {{
    long long idv[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {{ const int o = b0 + 256 * u; idv[u] = o < {n} ? {I}[{id_at}] : -1LL; }}
#pragma unroll
    for (int u = 0; u < 8; ++u) {{
      if (b0 + 256 * u < {n}) {{
        assert(idv[u] >= 0 && idv[u] < {V}LL);
        if (idv[u] != {padding}LL) atomicAdd(&cnt_[(int)idv[u]], 1);
      }}
    }}
  }}
  __syncthreads();
  // (2) rows without occurrences, including the padding row, written once as +0.0.
  if ({debug_stop} >= 2)
  for (int r = blockIdx.x * {WARPS} + warp; r < {V}; r += gridDim.x * {WARPS}) {{
    if (cnt_[r] == 0) {{
      {T} z; rs_zero(z);
      for (int vec = lane; vec < {DV}; vec += 32) out_[(long long)r * {DV} + vec] = z;
    }}
  }}
  // Units of one row: full width (k = 1) or column slices of ws vectors.
  auto units_of = [&](int c, int& k, int& ws, int& cost) {{
    if (c <= 0) {{ k = 0; ws = 0; cost = 0; return; }}
    if (c <= {FMAX}) {{ k = 1; ws = {DV}; cost = c * {J} + {HCOST}; return; }}
    const int q = (c + 31) >> 5;
    int v = {BUDGET} / q; v = v < 1 ? 1 : (v > {WSMAX} ? {WSMAX} : v); v = v > {DV} ? {DV} : v;
    k = ({DV} + v - 1) / v; ws = v; cost = q * v + {HCOST};
  }};
  // (3) identical unit cost prefix; this CTA owns units whose cost start lies in its window.
  const int per_ = ({V} + 255) / 256, r0_ = tid * per_, r1_ = min({V}, r0_ + per_);
  long long mine = 0;
  for (int r = r0_; r < r1_; ++r) {{ int k, ws, cost; units_of(cnt_[r], k, ws, cost); mine += (long long)k * cost; }}
  long long total;
  const long long start = rs_scan(mine, tmp_, total);
  const long long G_ = gridDim.x, cta = blockIdx.x;
  long long owned = 0;
  int lo = 0x7fffffff, hi = -1;
  if (total > 0) {{
    long long a = start;
    for (int r = r0_; r < r1_; ++r) {{
      int k, ws, cost; units_of(cnt_[r], k, ws, cost);
      for (int s = 0; s < k; ++s, a += cost)
        if ((a * G_) / total == cta) {{ ++owned; lo = min(lo, r); hi = max(hi, r); }}
    }}
  }}
  if (tid == 0) {{ meta_[0] = 0x7fffffff; meta_[1] = -1; }}
  __syncthreads();
  if (owned) {{ atomicMin(&meta_[0], lo); atomicMax(&meta_[1], hi); }}
  long long all_owned;
  const long long first_ = rs_scan(owned, tmp_, all_owned);
  const int rlo_ = meta_[0], rhi_ = meta_[1];
  if ({debug_stop} >= 3 && all_owned) {{
    // (4) packed occurrences in [rlo_, rhi_], ascending: warp w compacts its ordinal slice in order.
    {{
      const int p0 = warp * {seg}, p1 = min({n}, p0 + {seg});
      int len = 0;
      for (int b = p0; b < p1; b += 128) {{
        long long idv[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {{ const int o = b + 32 * u + lane; idv[u] = o < p1 ? {I}[{id_at}] : -1LL; }}
#pragma unroll
        for (int u = 0; u < 4; ++u) {{
          const int o = b + 32 * u + lane;
          const bool take = o < p1 && idv[u] >= rlo_ && idv[u] <= rhi_ && idv[u] != {padding}LL;
          const unsigned m = __ballot_sync(0xffffffffu, take);
          if (take) list_[p0 + len + __popc(m & below_)] = ((unsigned)(idv[u] - rlo_) << 16) | (unsigned)o;
          len += __popc(m);
        }}
      }}
      if (lane == 0) seglen_[warp] = len;
    }}
    __syncthreads();
    // (5) stable counting sort of this CTA's occurrences by row (cnt_ becomes end offsets in range).
    if ({debug_stop} >= 4 && warp == 0) {{
      int run = 0;
      for (int b = rlo_; b <= rhi_; b += 32) {{
        const int r = b + lane;
        const int c = r <= rhi_ ? cnt_[r] : 0;
        int x = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {{ const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }}
        if (r <= rhi_) cnt_[r] = run + x - c;                      // exclusive start (cursor)
        run += __shfl_sync(0xffffffffu, x, 31);
      }}
      __syncwarp();
      for (int g = 0; g < {WARPS}; ++g) {{
        const int base = g * {seg}, len = seglen_[g];
        for (int e = 0; e < len; e += 32) {{
          const bool live = e + lane < len;
          const unsigned packed = live ? list_[base + e + lane] : 0u;
          const int rr = live ? (int)(packed >> 16) : -1;
          const unsigned peers = __match_any_sync(0xffffffffu, rr);
          const int rank = __popc(peers & below_);
          const int pos = live ? cnt_[rlo_ + rr] + rank : 0;
          __syncwarp();
          if (live) sorted_[pos] = (unsigned short)(packed & 0xffffu);
          if (live && rank == __popc(peers) - 1) cnt_[rlo_ + rr] = pos + 1;   // cursor -> end
          __syncwarp();
        }}
      }}
    }}
    __syncthreads();
  }}
  // In [rlo_, rhi_] cnt_ now holds end offsets into sorted_; elsewhere counts.
  auto range_of = [&](int r, int& begin, int& c) {{
    if (r >= rlo_ && r <= rhi_) {{ begin = r > rlo_ ? cnt_[r - 1] : 0; c = cnt_[r] - begin; }}
    else {{ begin = 0; c = cnt_[r]; }}
  }};
  // (6) units in rounds of {UCAP}: materialize (row, slice, begin, count), then warps sum them.
  for (long long round = 0; {debug_stop} >= 5 && round < all_owned; round += {UCAP}) {{
    if (owned) {{
      long long a = start, local = first_;
      for (int r = r0_; r < r1_; ++r) {{
        int begin, c; range_of(r, begin, c);
        int k, ws, cost; units_of(c, k, ws, cost);
        for (int s = 0; s < k; ++s, a += cost) if ((a * G_) / total == cta) {{
          if (local >= round && local < round + {UCAP}) units_[local - round] = make_int4(r, s, begin, c);
          ++local;
        }}
      }}
    }}
    __syncthreads();
    const int count = (int)min((long long){UCAP}, all_owned - round);
    for (int u = warp; u < count; u += {WARPS}) {{
      const int4 unit = units_[u];
      const int r = unit.x, s = unit.y, begin = unit.z, c = unit.w;
      int k, ws, cost; units_of(c, k, ws, cost);
      if (k == 1) {{
        {T} acc_[{J}];
#pragma unroll
        for (int j = 0; j < {J}; ++j) rs_zero(acc_[j]);
        for (int i = 0; i < c; i += 8) {{
          {T} v_[8][{J}];
#pragma unroll
          for (int t = 0; t < 8; ++t) {{
            const bool live = i + t < c;
            const int o = live ? (int)sorted_[begin + i + t] : 0;
            const long long gb_ = live ? {grad_base} : 0LL;
#pragma unroll
            for (int j = 0; j < {J}; ++j) {{
              const int vec = lane + 32 * j;
              if (live && vec < {DV}) v_[t][j] = {load("gb_", "vec")}; else rs_zero(v_[t][j]);
            }}
          }}
#pragma unroll
          for (int t = 0; t < 8; ++t)
            if (i + t < c) {{
#pragma unroll
              for (int j = 0; j < {J}; ++j) rs_add(acc_[j], v_[t][j]);
            }}
        }}
#pragma unroll
        for (int j = 0; j < {J}; ++j) {{ const int vec = lane + 32 * j; if (vec < {DV}) out_[(long long)r * {DV} + vec] = acc_[j]; }}
      }} else {{
        const int v0 = s * ws, nv = min(ws, {DV} - v0);
        {T} acc_[{WSMAX}];
#pragma unroll
        for (int j = 0; j < {WSMAX}; ++j) rs_zero(acc_[j]);
        for (int i = lane; i < c; i += 32) {{             // occurrence g -> lane g mod 32, ascending
          const int o = (int)sorted_[begin + i];
          const long long gb_ = {grad_base};
          {T} v_[{WSMAX}];
#pragma unroll
          for (int j = 0; j < {WSMAX}; ++j) if (j < nv) v_[j] = {load("gb_", "v0 + j")};
#pragma unroll
          for (int j = 0; j < {WSMAX}; ++j) if (j < nv) rs_add(acc_[j], v_[j]);
        }}
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {{
#pragma unroll
          for (int j = 0; j < {WSMAX}; ++j) {{ const {T} y = rs_xor(acc_[j], off); rs_add(acc_[j], y); }}
        }}
#pragma unroll
        for (int j = 0; j < {WSMAX}; ++j) if (lane == j && j < nv) out_[(long long)r * {DV} + v0 + j] = acc_[j];
      }}
    }}
    __syncthreads();
  }}
}}"""

    if vector_ok:
        code = (f"if ((((unsigned long long)({G}) | (unsigned long long)({OUT})) & 15ULL) == 0ULL) {body(4)} "
                f"else {body(1)}")
    else:
        code = body(1)
    phase = dict(kind="embedding_gradient", writes=[name], reads=[grad, ids], code=code, shared_bytes=smem,
                 reduction_elements="rowscan occurrence units", accumulation_dtype="float32",
                 indexed_engine="rowscan_v1", indexed_reduction_policy="rowscan_v1",
                 arithmetic_policy="reassociated_fp32_v1")
    decision = dict(engine="rowscan_v1", vector_columns=vector_ok, shared_bytes=smem, full_width_max=FMAX,
                    slice_budget=BUDGET, slice_vectors_max=WSMAX, reused_occurrences=False,
                    arithmetic_policy="reassociated_fp32_v1",
                    accumulation_order=("rows with <= %d occurrences: ascending ordinals from +0.0 (ordered); heavier rows: "
                                        "column slices, occurrence g -> lane g mod 32 ascending, fixed xor butterfly" % FMAX),
                    work_bound="G*(N+V) + N*D + V*D", engine_phases=1)
    return [phase], decision
