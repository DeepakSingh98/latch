"""LATCH-STEP-04 `owner_v1`: single-phase deterministic FP32 embedding gradients by row ownership.

Generic lowering of `aten.embedding_dense_backward`, selected from shape, dtype and layout
only. One phase, no grid barrier, no atomics, no inter-CTA communication:

1. row r belongs to CTA r mod P (P = largest power of two <= launched grid; CTAs >= P own
   nothing), spreading neighbouring hot rows; ownership uses only bit operations;
2. each CTA scans the N indices once (batched loads); warp w stably compacts the
   occurrences of its CTA's rows from a contiguous ordinal slice (ballot + popc);
3. a local stable counting sort over the CTA's ceil(V / G) rows (per-warp bins, warp
   leaders only, no atomics) leaves every owned row's ordinals contiguous and ascending;
4. owned rows are summed: a light row (<= FMAX occurrences) by one warp, adding the
   occurrences in ascending ordinal order from +0.0 (bitwise the ordered policy); a heavy
   row in column slices spread over the CTA's warps, occurrence g -> lane g mod 32
   (ascending), lanes combined by a fixed xor butterfly;
5. owned rows without occurrences, including the padding row, are written as +0.0.

Declared arithmetic policy `reassociated_fp32_v1` (heavy-row slices); repeated runs are
bitwise identical for a fixed grid. Work: O(G*N + N*D + V*D). One extremely hot row is
summed by a single CTA (its SM bandwidth bounds that case).
"""
from __future__ import annotations

WARPS = 8
FMAX = 32          # occurrences summed in order by one full-width warp
BUDGET = 32        # per-lane vector loads targeted per column-slice unit (16 measured slower)
WSMAX = 8          # vectors per lane accumulated by a column-slice unit
UCAP = 256         # units materialized per round
SMEM_LIMIT = 160 * 1024
MAX_N = 16384      # every CTA scans all indices; beyond this the segmented engine is selected

HELPERS = r"""#include <cassert>
// LATCH-STEP-04 owner_v1 helpers (FP32 lanes, no FMA).
__device__ __forceinline__ void ow_zero(float& a) { a = 0.0f; }
__device__ __forceinline__ void ow_zero(float4& a) { a = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
__device__ __forceinline__ void ow_add(float& a, const float b) { a = __fadd_rn(a, b); }
__device__ __forceinline__ void ow_add(float4& a, const float4 b) {
  a.x = __fadd_rn(a.x, b.x); a.y = __fadd_rn(a.y, b.y); a.z = __fadd_rn(a.z, b.z); a.w = __fadd_rn(a.w, b.w);
}
__device__ __forceinline__ float ow_sq(const float a) { return __fmul_rn(a, a); }
__device__ __forceinline__ float ow_sq(const float4 a) {
  return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(a.x, a.x), __fmul_rn(a.y, a.y)), __fmul_rn(a.z, a.z)), __fmul_rn(a.w, a.w));
}
__device__ __forceinline__ float ow_xor(float a, int o) { return __shfl_xor_sync(0xffffffffu, a, o); }
__device__ __forceinline__ float4 ow_xor(float4 a, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, a.x, o), __shfl_xor_sync(0xffffffffu, a.y, o),
                     __shfl_xor_sync(0xffffffffu, a.z, o), __shfl_xor_sync(0xffffffffu, a.w, o));
}
"""


def _pad(x, unit=4):
    return -(-x // unit) * unit


GRID_CHUNK = 32    # owned local rows per pass are sized for >= 32 CTAs; smaller grids loop over row chunks


def local_rows(rows):
    return -(-rows // GRID_CHUNK)


def shared_bytes(n, rows):
    lmax = _pad(local_rows(rows) + 1)
    # [8][lmax] warp bins + [lmax] starts + packed entries u32[N] + sorted ordinals u16[N] + units + misc
    return 4 * (WARPS * lmax + 2 * lmax) + 4 * _pad(n) + 2 * _pad(n, 8) + 8 * UCAP + 4 * 32


def supported(n, rows, width):
    if n < 1 or rows < 1 or width < 1:
        return False, "empty index or row dimension"
    if n > MAX_N:
        return False, f"owner engine scans all indices in every CTA; N={n} > {MAX_N}"
    if rows >= (1 << 31) or local_rows(rows) >= (1 << 16):
        return False, "local row ids exceed 16 bits"
    if shared_bytes(n, rows) > SMEM_LIMIT:
        return False, f"local bins/lists need {shared_bytes(n, rows)} shared bytes > {SMEM_LIMIT}"
    return True, "owner engine admitted"


def _unravel(shape, strides, var):
    terms, inner = [], 1
    for extent, stride in reversed(list(zip(shape, strides))):
        if extent != 1 and stride != 0:
            coordinate = f"((unsigned){var} / {inner}u % {extent}u)" if inner != 1 else f"((unsigned){var} % {extent}u)"
            terms.append(f"(long long){coordinate} * {stride}LL")
        inner *= extent
    return " + ".join(reversed(terms)) or "0LL"


def emit(name, grad, ids, lg, li, rows, padding, width, ident, norm=None, flags=None, debug_stop=99, read=None):
    """read(position, lanes): optional gradient element reader (the static emitter's storage tokens, so an
    elementwise producer may be inlined); lanes 4 = float4 at position, 1 = one element, 0 = address bits."""
    n, V, D = li.numel, rows, width
    G, I, OUT = ident(grad), ident(ids), ident(name)
    id_at = _unravel(li.shape, li.strides, "o")
    grad_base = _unravel(lg.shape[:-1], lg.strides[:-1], "o")
    sc = lg.strides[-1]
    vector_ok = sc == 1 and D % 4 == 0 and all(s % 4 == 0 for s, e in zip(lg.strides[:-1], lg.shape[:-1]) if e != 1)
    smem = shared_bytes(n, V)
    seg = -(-n // WARPS)
    LMAX = local_rows(V)
    binstride = _pad(LMAX + 1)

    def body(w):
        T = "float4" if w == 4 else "float"
        DV = D // w
        J = -(-DV // 32)
        if read is not None:
            load = (lambda base, vec: read(f"({base}) + 4LL * (long long)({vec})", 4)) if w == 4 else \
                   (lambda base, vec: read(f"({base}) + (long long)({vec}) * {sc}LL", 1))
        else:
            load = (lambda base, vec: f"reinterpret_cast<const float4*>({G} + ({base}))[{vec}]") if w == 4 else \
                   (lambda base, vec: f"{G}[({base}) + (long long)({vec}) * {sc}LL]")

        def sq_add(value):
            return f"sq_ = __fadd_rn(sq_, ow_sq({value}));" if norm else ""

        def slice_block(W):
            # Column slice of W vectors: occurrence g -> lane g mod 32 (ascending); PF occurrences per lane in flight.
            PF = max(1, 16 // W)              # <= 16 vectors in flight per lane (register budget)
            return f"""if (ws == {W}) {{
          {T} acc_[{W}];
#pragma unroll
          for (int t = 0; t < {W}; ++t) ow_zero(acc_[t]);
          for (int i0 = lane; i0 < c; i0 += {32 * PF}) {{
            {T} v_[{PF}][{W}];
#pragma unroll
            for (int f = 0; f < {PF}; ++f) {{
              const int i = i0 + 32 * f;
              const bool live = i < c;
              const int o = live ? (int)sorted_[begin + i] : 0;
              const long long gb_ = live ? {grad_base} : 0LL;
#pragma unroll
              for (int t = 0; t < {W}; ++t) if (live && t < nv) v_[f][t] = {load("gb_", "v0 + t")}; else ow_zero(v_[f][t]);
            }}
#pragma unroll
            for (int f = 0; f < {PF}; ++f)
              if (i0 + 32 * f < c) {{
#pragma unroll
                for (int t = 0; t < {W}; ++t) ow_add(acc_[t], v_[f][t]);
              }}
          }}
#pragma unroll
          for (int off = 16; off > 0; off >>= 1) {{
#pragma unroll
            for (int t = 0; t < {W}; ++t) {{ const {T} y = ow_xor(acc_[t], off); ow_add(acc_[t], y); }}
          }}
#pragma unroll
          for (int t = 0; t < {W}; ++t) if (lane == t && t < nv) {{ out_[r * {DV} + v0 + t] = acc_[t]; {sq_add("acc_[t]")} }}
        }}"""
        slices = " else ".join(slice_block(W) for W in (1, 2, 4, 8) if W <= max(1, DV))
        # Row activity (1 = summed occurrences, 0 = zero gradient row) for a later inactive-row optimizer split.
        flag_store = (f"{ident(flags)}[(long long)cta + (long long)P * (jc + j)] = s > 0 ? 1.0f : 0.0f;"
                      if flags else "")
        # Optional fused clip-norm term sum(g * g): CTA tree in fixed order, then the last-arriving CTA adds
        # the per-CTA partials in CTA order (prologue-zeroed arrival counter, restored to zero).
        finalize = "" if not norm else f"""  {{
    __shared__ unsigned ow_last_;
    float x = sq_;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = __fadd_rn(x, __shfl_down_sync(0xffffffffu, x, o));
    float* wsq_ = reinterpret_cast<float*>(smem);
    __syncthreads();
    if (lane == 0) wsq_[warp] = x;
    __syncthreads();
    if (tid == 0) {{
      float s = 0.0f;
      for (int w = 0; w < {WARPS}; ++w) s = __fadd_rn(s, wsq_[w]);
      (/*P2_PART*/)[blockIdx.x] = s;
      __threadfence();
      ow_last_ = atomicAdd(/*P2_CNT*/, 1u) == gridDim.x - 1u;
    }}
    __syncthreads();
    if (ow_last_ && warp == 0) {{
      __threadfence();
      float s = 0.0f;
      for (int b = lane; b < Gc; b += 32) s = __fadd_rn(s, __ldcg((/*P2_PART*/) + b));
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, o));
      if (lane == 0) {{ {ident(norm)}[0] = s; *(/*P2_CNT*/) = 0u; }}
    }}
    __syncthreads();
  }}"""
        return f"""{{
  __syncthreads();
  const int Gc = gridDim.x, cta = blockIdx.x;
  const int LOGP = 31 - __clz(Gc), P = 1 << LOGP;             // owners: largest power of two <= grid
  int* bins_ = reinterpret_cast<int*>(smem);                 // [8][LMAX] warp counts, then cursors
  int* start_ = bins_ + {WARPS} * {binstride};               // [LMAX + 1] row starts in sorted_
  int* uoff_ = start_ + {binstride};                         // [LMAX + 1] unit offsets
  unsigned* ent_ = reinterpret_cast<unsigned*>(uoff_ + {binstride});        // [N] (local row << 16 | ordinal)
  unsigned short* sorted_ = reinterpret_cast<unsigned short*>(ent_ + {_pad(n)});  // [N] ordinals by row
  int2* units_ = reinterpret_cast<int2*>(sorted_ + {_pad(n, 8)});           // [UCAP] (local row, slice)
  int* misc_ = reinterpret_cast<int*>(units_ + {UCAP});
  {T}* out_ = reinterpret_cast<{T}*>({OUT});
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const unsigned below_ = (1u << lane) - 1u;
  const int Lall = cta < P ? ({V} - cta + P - 1) >> LOGP : 0;   // owned rows r = cta + P * j
  float sq_ = 0.0f;                                          // squares of values this thread wrote (fixed order)
  auto slicing = [&](int c, int& k, int& ws) {{
    if (c <= 0) {{ k = 0; ws = 0; return; }}
    if (c <= {FMAX}) {{ k = 1; ws = {DV}; return; }}
    const int q = (c + 31) >> 5;
    int v = {WSMAX};
    while (v > 1 && q * v > {BUDGET}) v >>= 1;               // power-of-two slice widths 8, 4, 2, 1
    while (v > {DV}) v >>= 1;
    ws = v; k = ({DV} + v - 1) / v;
  }};
  for (int jc = 0; jc < Lall; jc += {LMAX}) {{               // one pass for grids >= {GRID_CHUNK} CTAs
  const int L = min({LMAX}, Lall - jc);
  for (int i = tid; i < {WARPS} * {binstride}; i += 256) bins_[i] = 0;
  __syncthreads();
  // (1) stable compaction of owned occurrences (ascending within each warp slice); warp-private counts.
  int len = 0;
  if ({debug_stop} >= 1) {{
    const int p0 = warp * {seg}, p1 = min({n}, p0 + {seg});
    for (int b = p0; b < p1; b += 512) {{
      long long idv[16];
#pragma unroll
      for (int u = 0; u < 16; ++u) {{ const int o = b + 32 * u + lane; idv[u] = o < p1 ? {I}[{id_at}] : -1LL; }}
#pragma unroll
      for (int u = 0; u < 16; ++u) {{
        const int o = b + 32 * u + lane;
        int j = -1;
        if (o < p1) {{
          assert(idv[u] >= 0 && idv[u] < {V}LL);
          const int r = (int)idv[u];
          const int jl = (r >> LOGP) - jc;
          if (r != {padding} && (r & (P - 1)) == cta && jl >= 0 && jl < L) j = jl;
        }}
        const unsigned m = __ballot_sync(0xffffffffu, j >= 0);
        if (j >= 0) {{
          ent_[p0 + len + __popc(m & below_)] = ((unsigned)j << 16) | (unsigned)o;
          atomicAdd(&bins_[warp * {binstride} + j], 1);
        }}
        len += __popc(m);
      }}
    }}
  }}
  __syncthreads();
  // (2) row starts (row-major, warp-minor => stable), warp cursors and unit offsets via block scans.
  if ({debug_stop} >= 2) {{
    const int per = (L + 255) / 256, j0 = tid * per, j1 = min(L, j0 + per);
    int rows_total = 0, units_total = 0;
    for (int j = j0; j < j1; ++j) {{
      int s = 0;
#pragma unroll
      for (int w = 0; w < {WARPS}; ++w) s += bins_[w * {binstride} + j];
      int k, ws; slicing(s, k, ws);
      rows_total += s; units_total += k;
    }}
    int x = rows_total, y = units_total;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {{
      const int a = __shfl_up_sync(0xffffffffu, x, o), b = __shfl_up_sync(0xffffffffu, y, o);
      if (lane >= o) {{ x += a; y += b; }}
    }}
    if (lane == 31) {{ misc_[warp] = x; misc_[8 + warp] = y; }}
    __syncthreads();
    if (tid == 0) {{
      int ra = 0, ua = 0;
      for (int w = 0; w < {WARPS}; ++w) {{ const int a = misc_[w], b = misc_[8 + w]; misc_[w] = ra; misc_[8 + w] = ua; ra += a; ua += b; }}
      misc_[16] = ra; misc_[17] = ua;
    }}
    __syncthreads();
    int run = misc_[warp] + x - rows_total, urun = misc_[8 + warp] + y - units_total;
    for (int j = j0; j < j1; ++j) {{
      start_[j] = run; uoff_[j] = urun;
      int s = 0;
#pragma unroll
      for (int w = 0; w < {WARPS}; ++w) {{ const int c = bins_[w * {binstride} + j]; bins_[w * {binstride} + j] = run + s; s += c; }}
      int k, ws; slicing(s, k, ws);
      run += s; urun += k;
      {flag_store}
    }}
    if (tid == 0) {{ start_[L] = misc_[16]; uoff_[L] = misc_[17]; }}
  }}
  __syncthreads();
  // (3) stable scatter: each warp places its own ascending segment using its bin cursors.
  if ({debug_stop} >= 3) {{
    const int p0 = warp * {seg};
    for (int e = 0; e < len; e += 32) {{
      const bool live = e + lane < len;
      const unsigned packed = live ? ent_[p0 + e + lane] : 0u;
      const int ej = live ? (int)(packed >> 16) : -1;
      const unsigned peers = __match_any_sync(0xffffffffu, ej);
      const int rank = __popc(peers & below_);
      const int pos = live ? bins_[warp * {binstride} + ej] + rank : 0;
      __syncwarp();
      if (live) sorted_[pos] = (unsigned short)(packed & 0xffffu);
      if (live && rank == __popc(peers) - 1) bins_[warp * {binstride} + ej] = pos + 1;
      __syncwarp();
    }}
  }}
  __syncthreads();
  // (4) units: light rows whole, heavy rows in column slices; rounds of {UCAP}.
  const int U = {debug_stop} >= 4 ? uoff_[L] : 0;
  for (int ub = 0; ub < U; ub += {UCAP}) {{
    {{
      const int per = (L + 255) / 256, j0 = tid * per, j1 = min(L, j0 + per);
      for (int j = j0; j < j1; ++j) {{
        int k, ws; slicing(start_[j + 1] - start_[j], k, ws);
        for (int s = 0; s < k; ++s) {{ const int u = uoff_[j] + s - ub; if (u >= 0 && u < {UCAP}) units_[u] = make_int2(j, s); }}
      }}
    }}
    __syncthreads();
    const int count = min({UCAP}, U - ub);
    for (int u = warp; u < count; u += {WARPS}) {{
      const int j = units_[u].x, s = units_[u].y;
      const int begin = start_[j], c = start_[j + 1] - begin;
      int k, ws; slicing(c, k, ws);
      const long long r = (long long)cta + (long long)P * (jc + j);
      if (k == 1) {{
        {T} acc_[{J}];
#pragma unroll
        for (int q = 0; q < {J}; ++q) ow_zero(acc_[q]);
        for (int i = 0; i < c; i += 8) {{
          {T} v_[8][{J}];
#pragma unroll
          for (int t = 0; t < 8; ++t) {{
            const bool live = i + t < c;
            const int o = live ? (int)sorted_[begin + i + t] : 0;
            const long long gb_ = live ? {grad_base} : 0LL;
#pragma unroll
            for (int q = 0; q < {J}; ++q) {{
              const int vec = lane + 32 * q;
              if (live && vec < {DV}) v_[t][q] = {load("gb_", "vec")}; else ow_zero(v_[t][q]);
            }}
          }}
#pragma unroll
          for (int t = 0; t < 8; ++t)
            if (i + t < c) {{
#pragma unroll
              for (int q = 0; q < {J}; ++q) ow_add(acc_[q], v_[t][q]);
            }}
        }}
#pragma unroll
        for (int q = 0; q < {J}; ++q) {{ const int vec = lane + 32 * q; if (vec < {DV}) {{ out_[r * {DV} + vec] = acc_[q]; {sq_add("acc_[q]")} }} }}
      }} else {{
        const int v0 = s * ws, nv = min(ws, {DV} - v0);
        {slices}
      }}
    }}
    __syncthreads();
  }}
  // (5) owned rows without occurrences (including the padding row) written as +0.0.
  for (int j = warp; {debug_stop} >= 5 && j < L; j += {WARPS}) {{
    if (start_[j + 1] == start_[j]) {{
      const long long r = (long long)cta + (long long)P * (jc + j);
      {T} z; ow_zero(z);
      for (int vec = lane; vec < {DV}; vec += 32) out_[r * {DV} + vec] = z;
    }}
  }}
  /*LATCH_INACTIVE_ROWS_HOOK*/
  __syncthreads();
  }}
{finalize}
}}"""

    if vector_ok:
        g_addr = read("0", 0) if read is not None else f"(unsigned long long)({G})"
        code = (f"if ((({g_addr} | (unsigned long long)({OUT})) & 15ULL) == 0ULL) {body(4)} "
                f"else {body(1)}")
    else:
        code = body(1)
    phase = dict(kind="embedding_gradient", writes=[name] + ([norm] if norm else []) + ([flags] if flags else []),
                 reads=[grad, ids], code=code, row_active=flags, inactive_row_hook="/*LATCH_INACTIVE_ROWS_HOOK*/",
                 shared_bytes=smem,
                 **({"p2_split": {"counter_words": 1, "partial_words": 1024}, "fused_norm": norm,
                     "norm_arithmetic": "per-thread squares of written values in processing order; CTA tree; "
                                        "last-arriver sum of CTA partials in CTA order"} if norm else {}),
                 reduction_elements="owner row units", accumulation_dtype="float32",
                 indexed_engine="owner_v1", indexed_reduction_policy="owner_v1",
                 arithmetic_policy="reassociated_fp32_v1")
    decision = dict(engine="owner_v1", vector_columns=vector_ok, shared_bytes=smem, full_width_max=FMAX,
                    slice_budget=BUDGET, slice_vectors_max=WSMAX, reused_occurrences=False,
                    arithmetic_policy="reassociated_fp32_v1",
                    accumulation_order=("rows with <= %d occurrences: ascending ordinals from +0.0 (ordered); heavier rows: "
                                        "column slices, occurrence g -> lane g mod 32 ascending, fixed xor butterfly" % FMAX),
                    ownership="row r -> CTA r mod P, P = largest power of two <= gridDim.x", work_bound="G*N + N*D + V*D", engine_phases=1)
    return [phase], decision
