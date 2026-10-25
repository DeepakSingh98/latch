"""Segmented, load-balanced FP32 embedding gradients (OMNI-P1-A `segmented_v1`).

Generic lowering of `aten.embedding_dense_backward` selected from operation,
shape, dtype and layout only. Occurrences are ordered by the unique composite
key (row, ordinal) -- padding occurrences receive the sentinel row V and are
excluded -- using a stable shared-memory LSD radix sort of row digits per tile
(ordinals stay ascending inside a row), followed by global rank-merge passes
(log2(N/T) of them, none when one tile holds every occurrence). Positions of the
sorted list are then split into fixed chunks of S positions (8 chunks = one CTA
segment), so reduction work is balanced by occurrence count, independent of how
concentrated the indices are:

* a row whose occurrences lie inside one chunk is summed directly, in ascending
  ordinal order from +0.0 (bitwise the ordered policy), by the final phase;
* a row crossing chunk boundaries inside one segment is summed piecewise by the
  segment's warps (ascending within each piece) and the pieces are added in
  position order by the segment CTA;
* a row crossing segment boundaries additionally combines its per-segment sums
  in segment order: the piece list is split into at most eight contiguous ranges
  (one per warp, each summed in order) whose sums are added in range order.

Every contribution is used exactly once in FP32; no atomics touch floating data,
the association depends only on (N, S) and the input indices, so repeated runs
are bitwise identical. Declared arithmetic policy: `reassociated_fp32_v1`.
Inactive and padding rows are written exactly once as positive zero by the only
phase that declares the output. Scratch: tile keys plus V int2 row bounds, one
N-key buffer per merge pass, one partial row per chunk and per segment, and one
row id per segment. Column work uses float4 when the layout and the runtime
pointers admit it, otherwise scalar FP32 with identical association.
"""
from __future__ import annotations

WARPS = 8                # 256-thread CTAs (runtime-fixed)
MAX_WIDTH = 1024         # widest row held in shared-memory piece buffers
TILE32, TILE64 = 16384, 8192   # radix tile capacity for 32-bit / 64-bit packed keys

HELPERS = r"""#include <cassert>
// OMNI-P1-A segmented embedding gradients: lane element helpers (FP32 lanes, no FMA).
__device__ __forceinline__ void seg_zero(float& a) { a = 0.0f; }
__device__ __forceinline__ void seg_zero(float4& a) { a = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
__device__ __forceinline__ void seg_add(float& a, const float b) { a = __fadd_rn(a, b); }
__device__ __forceinline__ void seg_add(float4& a, const float4 b) {
  a.x = __fadd_rn(a.x, b.x); a.y = __fadd_rn(a.y, b.y); a.z = __fadd_rn(a.z, b.z); a.w = __fadd_rn(a.w, b.w);
}
// LATCH-MOESUP row-scaled contributions: float(bf16 x) * s, one rounding (the stored FP32 product's value).
__device__ __forceinline__ float seg_scaled(const __nv_bfloat16 x, const float s) { return __fmul_rn(__bfloat162float(x), s); }
__device__ __forceinline__ float4 seg_scaled4(const uint2 x, const float s) {
  return make_float4(__fmul_rn(__uint_as_float(x.x << 16), s), __fmul_rn(__uint_as_float(x.x & 0xffff0000u), s),
                     __fmul_rn(__uint_as_float(x.y << 16), s), __fmul_rn(__uint_as_float(x.y & 0xffff0000u), s));
}
// LATCH-MOESUP-ROW widened contributions: float(bf16 x), exact (the elided FP32 copy's value).
__device__ __forceinline__ float4 seg_widen4(const uint2 x) {
  return make_float4(__uint_as_float(x.x << 16), __uint_as_float(x.x & 0xffff0000u),
                     __uint_as_float(x.y << 16), __uint_as_float(x.y & 0xffff0000u));
}
"""


# LATCH-MOESUP-ROW: rows (positions) whose 16-byte column units are loaded together before their ordered adds in the
# vector walk (exactly the same ordered sum; more loads in flight per lane). The established value is 4.
SEG_VEC_BATCH = 4
# LATCH-MOESUP-ROW: the per-chunk walk of the final pass (vector form) loads its next SEG_PIPE_BATCH positions before
# adding (and storing) the current ones. The contributions and the output are distinct values, so no load can observe
# a row store; every row still adds its positions in sorted order (bitwise). Measured standalone at the MoE shape
# (rowbench seg.cu, 524288 positions x 384): 0.328 -> 0.244 ms with batch 2 (batch 4 0.287, batch 1 0.328).
SEG_PIPE = True
SEG_PIPE_BATCH = 2


def supported(n, rows, width):
    if width > MAX_WIDTH:
        return False, f"width {width} exceeds the {MAX_WIDTH}-column shared piece buffer"
    if rows + 1 >= (1 << 31) or n >= (1 << 31):
        return False, "row sentinel or ordinal exceeds 32-bit key fields"
    return True, "segmented engine admitted"


def chunk_size(n):
    """Positions per warp chunk: about 1024 chunks, at least 4 and at most 64 positions."""
    return max(4, min(64, -(-n // 1024)))


def tile_plan(n, rows):
    """Radix tile: one tile when possible; 32-bit packed (row, local index) keys when they fit."""
    rbits = max(1, rows.bit_length())          # rows 0..V (V = padding sentinel)
    for capacity, key_bits in ((TILE32, 32), (TILE64, 64)):
        tile = min(capacity, -(-n // 256) * 256)
        lbits = max(1, (tile - 1).bit_length())
        if key_bits == 64 or rbits + lbits <= 32:
            shift = lbits if key_bits == 32 else 32
            return dict(tile=tile, key_bits=key_bits, shift=shift, passes=-(-rbits // 8), row_bits=rbits)


def _unravel(shape, strides, var):
    """Flat ordinal `var` over `shape` -> element offset with the given strides (32-bit coordinates)."""
    terms, inner = [], 1
    for extent, stride in reversed(list(zip(shape, strides))):
        if extent != 1 and stride != 0:
            coordinate = f"((unsigned){var} / {inner}u % {extent}u)" if inner != 1 else f"((unsigned){var} % {extent}u)"
            terms.append(f"(long long){coordinate} * {stride}LL")
        inner *= extent
    return " + ".join(reversed(terms)) or "0LL"


def _peers(d, bits):
    """Lanes holding the same digit, from one ballot per digit bit (deterministic warp ranking)."""
    lines = [f"unsigned m_ = 0xffffffffu;"]
    for b in range(bits):
        lines.append(f"{{ const unsigned q_ = __ballot_sync(0xffffffffu, ({d} >> {b}) & 1u); m_ &= (({d} >> {b}) & 1u) ? q_ : ~q_; }}")
    return " ".join(lines)


def _tile_sort(index, keys, ids, n, V, padding, plan, ident, id_at):
    T, P, SH = plan["tile"], plan["passes"], plan["shift"]
    KT = "unsigned" if plan["key_bits"] == 32 else "unsigned long long"
    PAD = "0xffffffffu" if plan["key_bits"] == 32 else "~0ULL"
    LMASK = f"{(1 << SH) - 1}u" if plan["key_bits"] == 32 else "0xffffffffULL"
    tiles = -(-n // T)
    per_warp = T // WARPS
    return f"""{{
  int2* bounds_ = reinterpret_cast<int2*>({ident(index)} + {n});
  for (long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x; r < {V}LL; r += (long long)gridDim.x * blockDim.x)
    bounds_[r] = make_int2(0, 0);
  unsigned long long* out_ = reinterpret_cast<unsigned long long*>({ident(keys)});
  {KT}* A_ = reinterpret_cast<{KT}*>(smem);
  {KT}* B_ = A_ + {T};
  int* hist_ = reinterpret_cast<int*>(B_ + {T});      // [8][256] warp digit offsets, then 8 warp totals
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned below_ = (1u << lane) - 1u;
  for (int tile = blockIdx.x; tile < {tiles}; tile += gridDim.x) {{
    const int base = tile * {T};
    const int count = min({T}, {n} - base);
    for (int e = threadIdx.x; e < {T}; e += blockDim.x) {{
      {KT} k = {PAD};
      if (e < count) {{
        const unsigned o = (unsigned)(base + e);
        const long long id = {ident(ids)}[{id_at}];
        assert(id >= 0 && id < {V}LL);
        const {KT} row = id == {padding}LL ? ({KT}){V} : ({KT})id;
        k = (row << {SH}) | ({KT})e;
      }}
      A_[e] = k;
    }}
    __syncthreads();
    // Stable LSD radix passes over the row digits; warps own contiguous slices in tile order.
    for (int pass = 0; pass < {P}; ++pass) {{
      const int shift = {SH} + 8 * pass;
      for (int i = threadIdx.x; i < {WARPS * 256}; i += blockDim.x) hist_[i] = 0;
      __syncthreads();
      for (int it = 0; it < {per_warp // 32}; ++it) {{
        const unsigned d = (unsigned)(A_[warp * {per_warp} + it * 32 + lane] >> shift) & 255u;
        {_peers("d", 8)}
        const unsigned peers = m_;
        if (lane == __ffs(peers) - 1) hist_[warp * 256 + d] += __popc(peers);
      }}
      __syncthreads();
      {{
        const int t = threadIdx.x;   // one digit per thread
        int v[{WARPS}], run = 0;
#pragma unroll
        for (int w = 0; w < {WARPS}; ++w) {{ v[w] = run; run += hist_[w * 256 + t]; }}
        int x = run;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {{ const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }}
        if (lane == 31) hist_[{WARPS * 256} + warp] = x;
        __syncthreads();
        int before = 0;
        for (int w = 0; w < warp; ++w) before += hist_[{WARPS * 256} + w];
        const int exclusive = before + x - run;
#pragma unroll
        for (int w = 0; w < {WARPS}; ++w) hist_[w * 256 + t] = exclusive + v[w];
      }}
      __syncthreads();
      for (int it = 0; it < {per_warp // 32}; ++it) {{
        const {KT} k = A_[warp * {per_warp} + it * 32 + lane];
        const unsigned d = (unsigned)(k >> shift) & 255u;
        {_peers("d", 8)}
        const unsigned peers = m_;
        const int at = hist_[warp * 256 + d];
        B_[at + __popc(peers & below_)] = k;
        __syncwarp();
        if (lane == __ffs(peers) - 1) hist_[warp * 256 + d] = at + __popc(peers);
        __syncwarp();
      }}
      __syncthreads();
      {KT}* swap_ = A_; A_ = B_; B_ = swap_;
    }}
    for (int e = threadIdx.x; e < count; e += blockDim.x) {{
      const {KT} k = A_[e];
      out_[base + e] = ((unsigned long long)(k >> {SH}) << 32) | (unsigned long long)(base + (int)(k & {LMASK}));
    }}
    __syncthreads();
    if ({P} & 1) {{ {KT}* swap_ = A_; A_ = B_; B_ = swap_; }}
  }}
}}""", (2 * T * (4 if plan["key_bits"] == 32 else 8) + 4 * (WARPS * 256 + WARPS)), tiles


def radix_plan(n, rows, padding):
    """Global stable LSD radix over row digits: <=128 position tiles, <=256 bins per pass."""
    top = rows if padding >= 0 else rows - 1         # largest row value present (V = padding sentinel)
    rbits = max(1, top.bit_length())
    passes = -(-rbits // 8)
    bits = -(-rbits // passes)
    tiles = min(128, max(1, -(-n // 32)))
    span = -(-n // tiles)
    tiles = -(-n // span)
    per_warp = -(-span // (WARPS * 32)) * 32        # contiguous warp slice, whole 32-lane batches
    return dict(passes=passes, digit_bits=bits, bins=1 << bits, tiles=tiles, tile_positions=span,
                warp_positions=per_warp, batches=per_warp // 32, row_bits=rbits)


def _radix_phases(index, ids, n, V, padding, plan, ident, id_at, allocate):
    """Q0 histogram phase plus one scatter phase per pass; returns (phases, final keys value)."""
    P, DB, B, TT, T, TW, NB = (plan[k] for k in ("passes", "digit_bits", "bins", "tiles", "tile_positions",
                                                   "warp_positions", "batches"))
    H = f"(reinterpret_cast<int*>({ident(index)} + {V}))"
    key_from_ids = f"""const long long id = {ident(ids)}[{id_at}];
        assert(id >= 0 && id < {V}LL);
        k = ((unsigned long long)(id == {padding}LL ? {V}LL : id) << 32) | (unsigned long long)o;"""
    zero_hist = f"for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < {(P - 1) * TT * B}; i += gridDim.x * blockDim.x) {H}[{TT * B} + i] = 0;" if P > 1 else ""
    q0 = f"""{{
  int2* bounds_ = reinterpret_cast<int2*>({ident(index)});
  for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < {V}; r += gridDim.x * blockDim.x) bounds_[r] = make_int2(0, 0);
  {zero_hist}
  int* bins_ = reinterpret_cast<int*>(smem);
  for (int tile = blockIdx.x; tile < {TT}; tile += gridDim.x) {{
    for (int i = threadIdx.x; i < {B}; i += blockDim.x) bins_[i] = 0;
    __syncthreads();
    for (int q = threadIdx.x; q < {T}; q += blockDim.x) {{
      const int p = tile * {T} + q;
      if (p < {n}) {{
        const unsigned o = (unsigned)p; unsigned long long k;
        {key_from_ids}
        atomicAdd(&bins_[(unsigned)(k >> 32) & {B - 1}u], 1);
      }}
    }}
    __syncthreads();
    for (int i = threadIdx.x; i < {B}; i += blockDim.x) {H}[tile * {B} + i] = bins_[i];
    __syncthreads();
  }}
}}"""
    phases = [dict(kind="index_radix_histogram", writes=[index], reads=[ids], code=q0, shared_bytes=4 * B,
                   work_items=TT, radix=plan)]
    source = None
    for p_ in range(P):
        out = allocate(f"seg_keys_{p_ + 1}", (n,), "int64")
        shift = 32 + DB * p_
        load = (f"const unsigned o = (unsigned)p; {key_from_ids}" if source is None
                else f"k = reinterpret_cast<const unsigned long long*>({ident(source)})[p];")
        nxt = (f"atomicAdd(&{H}[{(p_ + 1) * TT * B} + (pos / {T}) * {B} + ((unsigned)(k_[j] >> {shift + DB}) & {B - 1}u)], 1);"
               if p_ + 1 < P else "")
        code = f"""{{
  const int* hist_ = {H} + {p_ * TT * B};
  unsigned long long* out_ = reinterpret_cast<unsigned long long*>({ident(out)});
  int* wc_ = reinterpret_cast<int*>(smem);     // [8][{B}] warp digit cursors
  int* wt_ = wc_ + {WARPS * B};                // 8 warp scan totals
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const unsigned below_ = (1u << lane) - 1u;
  for (int tile = blockIdx.x; tile < {TT}; tile += gridDim.x) {{
    unsigned long long k_[{NB}]; unsigned d_[{NB}], m_s[{NB}];
#pragma unroll
    for (int j = 0; j < {NB}; ++j) {{
      const int q = warp * {TW} + j * 32 + lane, p = tile * {T} + q;
      const bool valid = q < {T} && p < {n};
      unsigned long long k = ~0ULL;
      if (valid) {{ {load} }}
      k_[j] = k;
      d_[j] = (unsigned)(k >> {shift}) & {B - 1}u;
      {_peers("d_[j]", DB)}
      m_s[j] = m_ & __ballot_sync(0xffffffffu, valid) & (valid ? 0xffffffffu : 0u);
    }}
    for (int i = threadIdx.x; i < {WARPS * B}; i += blockDim.x) wc_[i] = 0;
    __syncthreads();
#pragma unroll
    for (int j = 0; j < {NB}; ++j)
      if (m_s[j] && lane == __ffs(m_s[j]) - 1) wc_[warp * {B} + d_[j]] += __popc(m_s[j]);
    __syncthreads();
    {{
      const int t = threadIdx.x;
      int total = 0, before = 0;
      if (t < {B}) {{
#pragma unroll 16
        for (int c = 0; c < {TT}; ++c) {{ const int h = hist_[c * {B} + t]; total += h; before += c < tile ? h : 0; }}
      }}
      int x = total;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {{ const int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }}
      if (lane == 31) wt_[warp] = x;
      __syncthreads();
      int prior = 0;
      for (int w = 0; w < warp; ++w) prior += wt_[w];
      if (t < {B}) {{
        int run = prior + x - total + before;
#pragma unroll
        for (int w = 0; w < {WARPS}; ++w) {{ const int c = wc_[w * {B} + t]; wc_[w * {B} + t] = run; run += c; }}
      }}
    }}
    __syncthreads();
#pragma unroll
    for (int j = 0; j < {NB}; ++j) {{
      const int at = m_s[j] ? wc_[warp * {B} + d_[j]] : 0;
      if (m_s[j]) {{
        const int pos = at + __popc(m_s[j] & below_);
        out_[pos] = k_[j];
        {nxt}
      }}
      __syncwarp();
      if (m_s[j] && lane == __ffs(m_s[j]) - 1) wc_[warp * {B} + d_[j]] = at + __popc(m_s[j]);
      __syncwarp();
    }}
    __syncthreads();
  }}
}}"""
        reads = [index] + ([ids] if source is None else [source])
        phases.append(dict(kind="index_radix_scatter", writes=[out], reads=reads, code=code,
                           shared_bytes=4 * (WARPS * B + WARPS), work_items=TT, radix_pass=p_, digit_shift=shift))
        source = out
    return phases, source


def emit(name, grad, ids, lg, li, rows, padding, width, ident, allocate, cache, sorter="radix", scale=None, widen=False):
    """Return (phases, decision) for the segmented engine; `allocate(suffix, extent, dtype)` adds scratch.

    scale (LATCH-MOESUP): (value, layout) of an FP32 per-occurrence factor; `grad` is then BF16 and each
    contribution is float(grad) * factor rounded once, the value of the elided FP32 product it replaces.
    widen (LATCH-MOESUP-ROW): `grad` is BF16 and each contribution is float(grad), the exact value of the elided
    FP32 copy it replaces."""
    n = li.numel
    V, D = rows, width
    S = chunk_size(n)
    SEG = WARPS * S
    C = -(-n // S)
    GS = -(-n // SEG)
    G = ident(grad)
    id_at = _unravel(li.shape, li.strides, "o")
    grad_base = _unravel(lg.shape[:-1], lg.strides[:-1], "o")
    sc = lg.strides[-1]
    # float4 columns: unit column stride, width and row strides multiples of four.
    vector_ok = sc == 1 and D % 4 == 0 and all(s % 4 == 0 for s, e in zip(lg.strides[:-1], lg.shape[:-1]) if e != 1)
    phases = []
    key = ("segmented", sorter, ids, rows, padding)
    reused = cache is not None and key in cache
    plan = None
    if reused:
        index, keys, ready = cache[key]
    elif sorter == "radix":
        plan = radix_plan(n, rows, padding)
        # Row bounds (int2 per row) then one int histogram per radix pass, one value per phase.
        index = allocate("seg_index", (V + -(-plan["passes"] * plan["tiles"] * plan["bins"] // 2),), "int64")
        sort_phases, keys = _radix_phases(index, ids, n, V, padding, plan, ident, id_at, allocate)
        phases.extend(sort_phases)
        ready = None
    else:
        plan = tile_plan(n, rows)
        # One value per phase (fusion hooks register a phase by its first write): the
        # sorted tile keys occupy words [0, N) and the int2 row bounds words [N, N + V).
        index = keys = allocate("seg_index", (n + V,), "int64")
        sort, smem_bytes, tiles = _tile_sort(index, keys, ids, n, V, padding, plan, ident, id_at)
        phases.append(dict(kind="index_tile_sort", writes=[index], reads=[ids], code=sort,
                           shared_bytes=smem_bytes, tile=plan["tile"], tiles=tiles, radix_passes=plan["passes"],
                           key_bits=plan["key_bits"], work_items=tiles))
        span, number = plan["tile"], 1
        while span < n:
            merged = allocate(f"seg_keys_{number}", (n,), "int64")
            code = f"""{{
  const unsigned long long* in_ = reinterpret_cast<const unsigned long long*>({ident(keys)});
  unsigned long long* out_ = reinterpret_cast<unsigned long long*>({ident(merged)});
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < {n}; i += gridDim.x * blockDim.x) {{
    const int begin = i / {2 * span} * {2 * span};
    const int middle = min(begin + {span}, {n}), end = min(begin + {2 * span}, {n});
    const bool left = i < middle;
    const unsigned long long k = in_[i];
    int lo = left ? middle : begin, hi = left ? end : middle;
    const int first = lo;
    while (lo < hi) {{ const int m = (lo + hi) >> 1; if (in_[m] < k) lo = m + 1; else hi = m; }}
    out_[begin + (i - (left ? begin : middle)) + (lo - first)] = k;
  }}
}}"""
            phases.append(dict(kind="index_merge", writes=[merged], reads=[keys], code=code, merge_span=span))
            keys, span, number = merged, span * 2, number + 1
        ready = None
    bounds_at = ident(index) if sorter == "radix" else f"{ident(index)} + {n}"
    # Partial rows [(C + GS) * D floats] followed by one int32 row id per segment.
    partials = allocate("seg_partials", ((C + GS) * D + GS,), "float32")
    K, P, out = ident(keys), ident(partials), ident(name)

    def lanes(w):
        """Column units per lane for element width w (4 -> float4)."""
        units = D // w
        return units, -(-units // 32)

    def vt(w):
        return "float4" if w == 4 else "float"

    def gload(w, base, unit):
        if scale is not None:
            factor = f"{ident(scale[0])}[{_unravel(scale[1].shape, scale[1].strides, 'o')}]"
            if w == 4:
                return f"seg_scaled4(reinterpret_cast<const uint2*>({G} + {base})[{unit}], {factor})"
            return f"seg_scaled({G}[{base} + (long long)({unit}) * {sc}LL], {factor})"
        if widen:
            if w == 4:
                return f"seg_widen4(reinterpret_cast<const uint2*>({G} + {base})[{unit}])"
            return f"__bfloat162float({G}[{base} + (long long)({unit}) * {sc}LL])"
        if w == 4:
            return f"reinterpret_cast<const float4*>({G} + {base})[{unit}]"
        return f"{G}[{base} + (long long)({unit}) * {sc}LL]"

    def walk(w, lo, hi):
        """Ordered sum of sorted positions [lo, hi) into acc_ (keys loaded 32 at a time)."""
        units, J = lanes(w)
        batch = max(1, min(8, 32 // J)) if w == 1 else max(1, min(SEG_VEC_BATCH, 24 // J))
        return f"""{{
    for (int j = 0; j < {J}; ++j) seg_zero(acc_[j]);
    for (int kb_ = {lo}; kb_ < {hi}; kb_ += 32) {{
      const int lo_ = kb_, hi_ = min({hi}, kb_ + 32);
      const int ord_ = lane < hi_ - lo_ ? (int)(unsigned)({K}[lo_ + lane] & 0xffffffffULL) : 0;
#pragma unroll 1
      for (int q = lo_; q < hi_; q += {batch}) {{
        {vt(w)} v_[{batch}][{J}];
#pragma unroll
        for (int u = 0; u < {batch}; ++u) {{
          const unsigned o = (unsigned)__shfl_sync(0xffffffffu, ord_, (q + u - kb_) & 31);
          const bool live_ = q + u < hi_;
          const long long gb_ = live_ ? {grad_base} : 0LL;
#pragma unroll
          for (int j = 0; j < {J}; ++j) {{
            const int c = lane + 32 * j;
            if (live_ && c < {units}) v_[u][j] = {gload(w, "gb_", "c")}; else seg_zero(v_[u][j]);
          }}
        }}
#pragma unroll
        for (int u = 0; u < {batch}; ++u)
          if (q + u < hi_) {{
#pragma unroll
            for (int j = 0; j < {J}; ++j) seg_add(acc_[j], v_[u][j]);
          }}
      }}
    }}
  }}"""

    def partials_body(w):
        units, J = lanes(w)
        T = vt(w)
        return f"""{{
  {T}* piece_ = reinterpret_cast<{T}*>(smem);          // [16][{units}]
  int* meta_ = reinterpret_cast<int*>(smem + 16 * {D});  // prow[16], pcont[16], groups
  {T}* part_ = reinterpret_cast<{T}*>({P});
  for (int g = blockIdx.x; g < {GS}; g += gridDim.x) {{
    const int p0 = (g * {WARPS} + warp) * {S};
    int hrow = -1, trow = -1, hcont = 0;
    if (p0 < {n}) {{
      const int p1 = min(p0 + {S}, {n});
      const unsigned rl = lane < p1 - p0 ? (unsigned)(K_[p0 + lane] >> 32) : 0xffffffffu;
      const unsigned rh = lane + 32 < p1 - p0 ? (unsigned)(K_[p0 + 32 + lane] >> 32) : 0xffffffffu;
      const unsigned first = __shfl_sync(0xffffffffu, rl, 0);
      const unsigned last = (p1 - p0 > 32) ? __shfl_sync(0xffffffffu, rh, (p1 - p0 - 33) & 31)
                                           : __shfl_sync(0xffffffffu, rl, (p1 - p0 - 1) & 31);
      const bool before = p0 > 0 && first < {V}u && (unsigned)(K_[p0 - 1] >> 32) == first;
      const bool after = p1 < {n} && last < {V}u && (unsigned)(K_[p1] >> 32) == last;
      // Extent of the first run and start of the last run inside the chunk.
      const unsigned diff_first = __ballot_sync(0xffffffffu, rl != first);
      const unsigned diff_first_h = __ballot_sync(0xffffffffu, rh != first);
      const int hend = diff_first ? p0 + __ffs(diff_first) - 1 : (diff_first_h && p1 - p0 > 32 ? p0 + 32 + __ffs(diff_first_h) - 1 : p1);
      const unsigned other_l = __ballot_sync(0xffffffffu, rl != last && lane < p1 - p0);
      const unsigned other_h = __ballot_sync(0xffffffffu, rh != last && lane + 32 < p1 - p0);
      const int tbeg = other_h ? p0 + 32 + 32 - __clz(other_h) : (other_l ? p0 + 32 - __clz(other_l) : p0);
      const bool single = first == last;
      {T} acc_[{J}];
      if (before) {{
        hrow = (int)first; hcont = single && after;
        {walk(w, "p0", "hend")}
        for (int j = 0; j < {J}; ++j) if (lane + 32 * j < {units}) piece_[(2 * warp) * {units} + lane + 32 * j] = acc_[j];
      }}
      if (after && !(single && before)) {{
        trow = (int)last;
        {walk(w, "tbeg", "p1")}
        for (int j = 0; j < {J}; ++j) if (lane + 32 * j < {units}) piece_[(2 * warp + 1) * {units} + lane + 32 * j] = acc_[j];
      }}
    }}
    if (lane == 0) {{ meta_[2 * warp] = hrow; meta_[2 * warp + 1] = trow; meta_[16 + 2 * warp] = hcont; meta_[17 + 2 * warp] = 1; }}
    __syncthreads();
    if (threadIdx.x == 0) {{
      // Group the segment's pieces by row (sorted rows: one row's pieces are consecutive).
      int count = 0, open = -1, listed = -1;
      for (int i = 0; i < 16; ++i) {{
        const int r = meta_[i];
        if (r < 0) continue;
        if (open >= 0 && meta_[32 + 3 * open] == r) {{ meta_[32 + 3 * open + 2] = i; continue; }}
        open = count++;
        meta_[32 + 3 * open] = r; meta_[32 + 3 * open + 1] = i; meta_[32 + 3 * open + 2] = i;
      }}
      // A group continues past the segment iff its last piece continues out of its chunk.
      for (int q = 0; q < count; ++q) {{
        const int i0 = meta_[32 + 3 * q + 1], i1 = meta_[32 + 3 * q + 2];
        const bool head = i0 == 0;                         // continues from the previous segment
        const bool cont = (i1 & 1) ? true : (meta_[16 + i1] != 0);
        if (!head && cont) listed = meta_[32 + 3 * q];
        meta_[80 + q] = head ? {C} + g : g * {WARPS} + (i0 >> 1);
      }}
      meta_[96] = count;
      seg_rows_[g] = listed;
    }}
    __syncthreads();
    const int groups = meta_[96];
    for (int c = threadIdx.x; c < {units}; c += blockDim.x)
      for (int q = 0; q < groups; ++q) {{
        {T} acc; seg_zero(acc);
        for (int i = meta_[32 + 3 * q + 1]; i <= meta_[32 + 3 * q + 2]; ++i)
          if (meta_[i] >= 0) seg_add(acc, piece_[i * {units} + c]);
        part_[(long long)meta_[80 + q] * {units} + c] = acc;
      }}
    __syncthreads();
  }}
}}"""

    def final_body(w):
        units, J = lanes(w)
        T = vt(w)
        fbatch = max(1, min(8, 32 // J)) if w == 1 else max(1, min(SEG_VEC_BATCH, 24 // J))
        piped = SEG_PIPE and w == 4
        if piped:
            fbatch = SEG_PIPE_BATCH

        def load(dst, rws, q):
            return f"""#pragma unroll
      for (int u = 0; u < {fbatch}; ++u) {{
        const int at = ({q} + u - p0) & 63;
        const unsigned long long key_ = at < 32 ? __shfl_sync(0xffffffffu, kl, at & 31) : __shfl_sync(0xffffffffu, kh, at & 31);
        const unsigned o = (unsigned)(key_ & 0xffffffffULL);
        {rws}[u] = (int)(unsigned)(key_ >> 32);
        const bool live_ = {q} + u < hi;
        const long long gb_ = live_ ? {grad_base} : 0LL;
#pragma unroll
        for (int j = 0; j < {J}; ++j) {{
          const int c = lane + 32 * j;
          if (live_ && c < {units}) {dst}[u][j] = {gload(w, "gb_", "c")}; else seg_zero({dst}[u][j]);
        }}
      }}"""
        if piped:
            walk_head = f"""    {T} n_[{fbatch}][{J}]; int nr_[{fbatch}];
    if (lo < hi) {{
      {load("n_", "nr_", "lo")}
    }}
#pragma unroll 1
    for (int q = lo; q < hi; q += {fbatch}) {{
      {T} v_[{fbatch}][{J}]; int rw_[{fbatch}];
#pragma unroll
      for (int u = 0; u < {fbatch}; ++u) {{ rw_[u] = nr_[u]; for (int j = 0; j < {J}; ++j) v_[u][j] = n_[u][j]; }}
      if (q + {fbatch} < hi) {{
      {load("n_", "nr_", f"(q + {fbatch})")}
      }}"""
        else:
            walk_head = f"""#pragma unroll 1
    for (int q = lo; q < hi; q += {fbatch}) {{
      {T} v_[{fbatch}][{J}]; int rw_[{fbatch}];
      {load("v_", "rw_", "q")}"""
        return f"""{{
  {T}* wsum_ = reinterpret_cast<{T}*>(smem);          // [8][{units}]
  int* wlive_ = reinterpret_cast<int*>(smem + {WARPS} * {D});
  const {T}* part_ = reinterpret_cast<const {T}*>({P});
  {T}* out_ = reinterpret_cast<{T}*>({out});
  // (a) rows spanning several segments: whole CTA, ordered range sums.
  for (int g = blockIdx.x; g < {GS}; g += gridDim.x) {{
    const int r = seg_rows_[g];
    if (r < 0) continue;
    const int2 be = bounds_[r];
    const int gb = be.x / {SEG}, ge = (be.y - 1) / {SEG}, F = ge - gb + 1;
    const int lo = warp * F / {WARPS}, hi = (warp + 1) * F / {WARPS};
    {T} acc_[{J}];
    for (int j = 0; j < {J}; ++j) seg_zero(acc_[j]);
    for (int f = lo; f < hi; ++f) {{
      const long long slot = f == 0 ? (long long)(be.x / {S}) : (long long)({C} + gb + f);
      for (int j = 0; j < {J}; ++j) {{ const int c = lane + 32 * j; if (c < {units}) seg_add(acc_[j], part_[slot * {units} + c]); }}
    }}
    for (int j = 0; j < {J}; ++j) if (lane + 32 * j < {units}) wsum_[warp * {units} + lane + 32 * j] = acc_[j];
    if (lane == 0) wlive_[warp] = hi > lo;
    __syncthreads();
    for (int c = threadIdx.x; c < {units}; c += blockDim.x) {{
      {T} acc; seg_zero(acc);
      for (int w = 0; w < {WARPS}; ++w) if (wlive_[w]) seg_add(acc, wsum_[w * {units} + c]);
      out_[(long long)r * {units} + c] = acc;
    }}
    __syncthreads();
  }}
  // (b1) zero rows and in-segment crossing rows (copied from their segment sum): rows grid-strided.
  const int warps = gridDim.x * {WARPS}, gw = blockIdx.x * {WARPS} + warp;
  const int per = ({V} + warps - 1) / warps, r0 = gw * per, r1 = min({V}, r0 + per);
  for (int rb = r0; rb < r1; rb += 32) {{
    const int mine = rb + lane;
    const bool here = mine < r1;
    const int2 own = here ? bounds_[mine] : make_int2(0, 0);
    const bool act = here && own.x != own.y && mine != {padding};
    const bool copy = act && own.x / {S} != (own.y - 1) / {S} && own.x / {SEG} == (own.y - 1) / {SEG};
    unsigned zm = __ballot_sync(0xffffffffu, here && !act), cm = __ballot_sync(0xffffffffu, copy);
    while (zm) {{
      const int t = __ffs(zm) - 1; zm &= zm - 1;
      {T} z; seg_zero(z);
      for (int j = 0; j < {J}; ++j) {{ const int c = lane + 32 * j; if (c < {units}) out_[(long long)(rb + t) * {units} + c] = z; }}
    }}
    while (cm) {{
      const int t = __ffs(cm) - 1; cm &= cm - 1;
      const long long slot = __shfl_sync(0xffffffffu, own.x, t) / {S};
      for (int j = 0; j < {J}; ++j) {{ const int c = lane + 32 * j; if (c < {units}) out_[(long long)(rb + t) * {units} + c] = part_[slot * {units} + c]; }}
    }}
  }}
  // (b2) rows inside one chunk: warp per chunk (balanced by occurrences); the chunk's inner
  // runs stream as one position list so loads overlap across rows; each row adds in order.
  for (int k = gw; k < {C}; k += warps) {{
    const int p0 = k * {S}, p1 = min(p0 + {S}, {n});
    const unsigned long long kl = lane < p1 - p0 ? (unsigned long long){K}[p0 + lane] : ~0ULL;
    const unsigned long long kh = lane + 32 < p1 - p0 ? (unsigned long long){K}[p0 + 32 + lane] : ~0ULL;
    const unsigned rl = (unsigned)(kl >> 32), rh = (unsigned)(kh >> 32);
    const unsigned first = __shfl_sync(0xffffffffu, rl, 0);
    const unsigned last = (p1 - p0 > 32) ? __shfl_sync(0xffffffffu, rh, (p1 - p0 - 33) & 31)
                                         : __shfl_sync(0xffffffffu, rl, (p1 - p0 - 1) & 31);
    const bool before = p0 > 0 && first < {V}u && (unsigned)({K}[p0 - 1] >> 32) == first;
    const bool after = p1 < {n} && last < {V}u && (unsigned)({K}[p1] >> 32) == last;
    const unsigned df = __ballot_sync(0xffffffffu, rl != first), dfh = __ballot_sync(0xffffffffu, rh != first);
    const int hend = df ? p0 + __ffs(df) - 1 : (dfh && p1 - p0 > 32 ? p0 + 32 + __ffs(dfh) - 1 : p1);
    const unsigned ol = __ballot_sync(0xffffffffu, rl != last && lane < p1 - p0);
    const unsigned oh = __ballot_sync(0xffffffffu, rh != last && lane + 32 < p1 - p0);
    const int tbeg = oh ? p0 + 32 + 32 - __clz(oh) : (ol ? p0 + 32 - __clz(ol) : p0);
    const unsigned pl = __ballot_sync(0xffffffffu, rl >= {V}u && lane < p1 - p0);
    const unsigned ph = __ballot_sync(0xffffffffu, rh >= {V}u && lane + 32 < p1 - p0);
    const int pad = pl ? p0 + __ffs(pl) - 1 : (ph ? p0 + 32 + __ffs(ph) - 1 : p1);
    const int lo = before ? hend : p0;
    const int hi = min(after ? tbeg : p1, pad);
    {T} acc_[{J}];
    for (int j = 0; j < {J}; ++j) seg_zero(acc_[j]);
    int cur = -1;
{walk_head}
#pragma unroll
      for (int u = 0; u < {fbatch}; ++u)
        if (q + u < hi) {{
          if (rw_[u] != cur) {{
            if (cur >= 0)
              for (int j = 0; j < {J}; ++j) {{ const int c = lane + 32 * j; if (c < {units}) out_[(long long)cur * {units} + c] = acc_[j]; }}
            for (int j = 0; j < {J}; ++j) seg_zero(acc_[j]);
            cur = rw_[u];
          }}
          for (int j = 0; j < {J}; ++j) seg_add(acc_[j], v_[u][j]);
        }}
    }}
    if (cur >= 0)
      for (int j = 0; j < {J}; ++j) {{ const int c = lane + 32 * j; if (c < {units}) out_[(long long)cur * {units} + c] = acc_[j]; }}
  }}
}}"""

    def choose(body, pointers):
        if not vector_ok:
            return body(1)
        aligned = " | ".join(f"(unsigned long long)({p})" for p in pointers)
        return f"if ((({aligned}) & 15ULL) == 0ULL) {body(4)} else {body(1)}"

    # ---- per-chunk pieces and per-segment ordered combination ----
    scatter = "" if reused else f"""for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < {n}; p += gridDim.x * blockDim.x) {{
    const unsigned r = (unsigned)(K_[p] >> 32);
    if (r < {V}u) {{
      if (p == 0 || (unsigned)(K_[p - 1] >> 32) != r) bounds_[r].x = p;
      if (p == {n - 1} || (unsigned)(K_[p + 1] >> 32) != r) bounds_[r].y = p + 1;
    }}
  }}"""
    code = f"""{{
  int2* bounds_ = reinterpret_cast<int2*>({bounds_at});
  int* seg_rows_ = reinterpret_cast<int*>({P} + {(C + GS) * D});
  const unsigned long long* K_ = reinterpret_cast<const unsigned long long*>({K});
  {scatter}
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  {choose(partials_body, (G, P))}
}}"""
    phases.append(dict(kind="index_segment_partials", writes=[partials], reads=[keys, grad, index, *([scale[0]] if scale else [])], code=code,
                       shared_bytes=16 * D * 4 + 4 * 128, work_items=GS, chunk_positions=S, segment_positions=SEG))
    if not reused:
        # A later table reusing these occurrences reads this value: its final phase
        # then waits for the row bounds scattered above.
        ready = partials
        cache[key] = (index, keys, ready)
    # ---- final dense materialization: the only writer of the output ----
    final = f"""{{
  const int2* bounds_ = reinterpret_cast<const int2*>({bounds_at});
  const int* seg_rows_ = reinterpret_cast<const int*>({P} + {(C + GS) * D});
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  {choose(final_body, (G, P, out))}
}}"""
    reads_f = list(dict.fromkeys([keys, grad, index, partials, ready, *([scale[0]] if scale else [])]))
    # LATCH-MOESUP-ROW: the occurrence order (keys = row << 32 | ordinal, ascending) is recorded so other row loops
    # over the same index may visit their rows grouped by index value (ops/row_fusion.py gather order).
    phases.append(dict(kind="embedding_gradient", writes=[name], reads=reads_f, code=final,
                       sorted_keys=keys, sorted_ids=ids, sorted_positions=n, sorted_rows=V,
                       shared_bytes=WARPS * D * 4 + 4 * WARPS,
                       reduction_elements="segmented occurrence pieces", accumulation_dtype="float32",
                       indexed_engine="segmented_v1", indexed_reduction_policy="segmented_v1",
                       arithmetic_policy="reassociated_fp32_v1"))
    for phase in phases:
        # Constant-trip column loops stay in registers (no local-memory accumulator arrays).
        phase["code"] = phase["code"].replace("for (int j = 0; j <", "_Pragma(\"unroll\") for (int j = 0; j <")
    decision = dict(engine="segmented_v1", chunk_positions=S, segment_positions=SEG, chunks=C, segments=GS,
                    sorter=sorter, sort_plan=plan, vector_columns=vector_ok, reused_occurrences=reused,
                    row_scale=scale[0] if scale else None, row_widen=bool(widen),
                    arithmetic_policy="reassociated_fp32_v1",
                    accumulation_order=("single-chunk rows: ascending ordinals from +0.0; otherwise ascending pieces, "
                                        "pieces added in position order; multi-segment rows: <=8 contiguous ordered "
                                        "ranges of segment sums added in range order"),
                    work_bound="N*passes + N log^2(N/T) + V*D + N*D",
                    scratch=dict(partial_floats=(C + GS) * D + GS,
                                 sort_buffers=None if plan is None else (plan["passes"] if sorter == "radix" else
                                                                         -(-n // plan["tile"]) - 1)))
    from . import indexed_segmented_outline as outline
    if outline.current_preference() != "off":
        bindings = [(G, "__nv_bfloat16*" if scale is not None or widen else "float*"),
                    (K, "int64_t*"), (ident(index), "int64_t*"), (P, "float*"),
                    (out, "float*"), ("smem", "float*")]
        if scale is not None:
            bindings.append((ident(scale[0]), "float*"))
        helpers = []
        for i, phase in enumerate(phases):
            if phase["kind"] in ("index_segment_partials", "embedding_gradient"):
                phases[i], helper = outline.outline(phase, bindings)
                if helper not in helpers:
                    helpers.append(helper)
        decision["outlined_helpers"] = helpers
    return phases, decision
