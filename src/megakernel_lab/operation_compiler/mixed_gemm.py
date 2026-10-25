"""Explicit BF16 ATen mm/bmm using unchanged pinned Latch matrix bodies.

Donor: private Latch, commit 0987d803f7b8be0d41cc653efd928e6581a97026,
src/megakernel_releases/s108/_current/ordinary/native/lib/:
  bgemm.cuh git blob 7399bbe145064763b06a8a50026341a606c261b6
  tgemm.cuh git blob d331fd5f462bebb6c7bda527dadcc4bbcc75f78c

The two files in vendor/ are unchanged. Only the operation-to-Pair adapter and
BF16-result epilogue below are new. No model generator or backward schedule is
imported. The body uses fixed ascending-K BF16 MMA with FP32 accumulation; the
output rounds once to BF16 to match ordinary BF16 ATen mm semantics. FP32 master
parameters, gradients after cast-back, and optimizer arithmetic remain FP32.
"""
from __future__ import annotations


HEADERS = ('tgemm.cuh', 'bgemm.cuh')
CUDA_HELPERS = r'''
#ifndef LATCH_OPS_BF16_GEMM_HELPERS
#define LATCH_OPS_BF16_GEMM_HELPERS
#include "bgemm.cuh"
struct LatchOpsStoreBF16 {
  __nv_bfloat16* out;
  long long ld, batch_stride;
  __device__ __forceinline__ void operator()(int batch, int row, int col, float4 value) const {
    s101h::st4bf(out + (long long)batch * batch_stride + (long long)row * ld + col, value);
  }
};
#endif
'''


# P8-E split generic: a generic-fallback product whose few output tiles each run a long K serially (e.g. narrow,
# 8-byte-row operands of a patch-2 embedding against 12,544 token rows) splits K into parts so up to `grid` tasks
# run at once. Each part accumulates its K range ascending with fmaf from +0 (as the unsplit tile); the last part to
# arrive at its tile sums the FP32 part results in part order and rounds once to BF16 (the split-K arithmetic of the
# TMA body: FP32 reassociation only), then restores its arrival counter to zero.
SPLIT_MIN_K = 2048
SPLIT_PART_K = 256


def _split_parts(tiles, k, grid):
    if not grid or k < SPLIT_MIN_K or tiles * 4 > grid:
        return 1
    return max(1, min(-(-k // SPLIT_PART_K), grid // tiles))


# P9-E skinny products (generic-fallback shapes the 16x16 SIMT tile serves poorly), arithmetic unchanged:
#  * short K (K <= 16): every output element is the generic tile's ascending-k fmaf chain from +0 over the same
#    exactly widened BF16 operands, rounded once -- an elementwise loop (B staged once per CTA as FP32);
#  * split products with one output dimension <= 16 (S) and the other contiguous in its operand (L): each split task
#    keeps the generic split's K part [k0, k1) and its ascending fmaf order per output, but a CTA covers 256 / S
#    contiguous L positions x all S, streaming K in 128-row slabs (16-byte loads, double-buffered in shared memory);
#    the last arriving part sums the FP32 parts in part order and rounds once (the same parts, sum order and rounding
#    as the generic split: bitwise identical results).
P9E_SKINNY = True
SKINNY_KC = 128
SKINNY_MIN_TASKS = 128
# P9-E (candidate A9): skinny split products choose their own K-part count so that (chunks x parts) fills the grid
# (the generic split sizes parts for 16x16 tiles, leaving half the CTAs idle on a long serial K). Changes only the
# association of the FP32 part sum (parts of ascending fmaf chains, summed in part order).
P9E_SKINNY_PARTS = False


def _skinny_k(a, b, out, m, n, k, sam, sak, sbk, sbn, ident):
    return f"""{{ // P9-E skinny-K product: {m}x{n} = ({m}x{k})({k}x{n}), ascending-k fmaf from +0, one BF16 rounding
  __syncthreads();                                   // earlier users of shared memory are done
  for (int i = threadIdx.x; i < {k * n}; i += blockDim.x)
    smem[i] = __bfloat162float({ident(b)}[(i / {n}) * {sbk}LL + (i % {n}) * {sbn}LL]);
  __syncthreads();
  for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < {m * n}LL; e += (long long)gridDim.x * blockDim.x) {{
    const long long row = e / {n}LL;
    const int col = (int)(e % {n}LL);
    float acc = 0.0f;
#pragma unroll
    for (int z = 0; z < {k}; ++z) acc = fmaf(__bfloat162float({ident(a)}[row * {sam}LL + z * {sak}LL]), smem[z * {n} + col], acc);
    {out}[row * {n}LL + col] = __float2bfloat16_rn(acc);
  }}
  __syncthreads();
}}"""


def _skinny_split(a, b, out, m, n, k, sam, sak, sbk, sbn, parts, kp, ident):
    small_n = n <= 16
    S = n if small_n else m
    L = m if small_n else n
    sp = 1
    while sp < S:
        sp *= 2
    LC = 256 // sp
    while -(-L // LC) * parts < SKINNY_MIN_TASKS and LC > 16:   # narrower chunks until the grid is half full
        LC //= 2
    chunks = -(-L // LC)
    KC = SKINNY_KC
    per = -(-KC * LC // 8 // 256)                     # 16-byte big-operand loads per thread per slab
    big, small = (a, b) if small_n else (b, a)
    big_k = sak if small_n else sbk                   # big operand stride along K (unit stride along L)
    # small operand element (k, s)
    small_at = (f"{ident(b)}[(kk) * {sbk}LL + (ss) * {sbn}LL]" if small_n
                else f"{ident(a)}[(ss) * {sam}LL + (kk) * {sak}LL]")
    row, col = ("l", "sidx") if small_n else ("sidx", "l")
    return f"""{{ // P9-E skinny split: {L} x {S} outputs, {parts} K parts of {kp}, {chunks} chunks of {LC}, {KC}-row slabs
  __shared__ int latch_last;
  unsigned* latch_cnt = /*P2_CNT*/;
  float* latch_part = /*P2_PART*/;                   // [{parts}][{m * n}] FP32 part results
  __nv_bfloat16* bigs = reinterpret_cast<__nv_bfloat16*>(smem);   // [2][{KC}][{LC}]
  float* smalls = smem + {KC * LC};                  // [2][{KC}][{sp}]
  const int sidx = threadIdx.x / {LC}, lt = threadIdx.x % {LC};
  for (long long task = blockIdx.x; task < {chunks * parts}LL; task += gridDim.x) {{
    const int chunk = (int)(task / {parts}LL), part = (int)(task % {parts}LL);
    const int l = chunk * {LC} + lt;
    const int k0 = part * {kp}, k1 = min({k}, k0 + {kp});
    const int slabs = (k1 - k0 + {KC - 1}) / {KC};
    uint4 rb[{per}];
    float rs[{-(-KC * sp // 256)}];
    auto fetch = [&](int sl) {{
      const int kb = k0 + sl * {KC};
#pragma unroll
      for (int u = 0; u < {per}; ++u) {{
        const int c = threadIdx.x + 256 * u, kk = c / {LC // 8}, l8 = (c % {LC // 8}) * 8;
        const int gl = chunk * {LC} + l8;
        rb[u] = (c < {KC * LC // 8} && kb + kk < k1 && gl < {L}) ? __ldcg(reinterpret_cast<const uint4*>({ident(big)} + (long long)(kb + kk) * {big_k}LL + gl))
                                          : make_uint4(0u, 0u, 0u, 0u);
      }}
#pragma unroll
      for (int u = 0; u < {-(-KC * sp // 256)}; ++u) {{
        const int c = threadIdx.x + 256 * u, kq = c / {sp}, ss = c % {sp}, kk = kb + kq;
        rs[u] = (c < {KC * sp} && kk < k1 && ss < {S}) ? __bfloat162float({small_at}) : 0.0f;
      }}
    }};
    auto place = [&](int buf) {{
#pragma unroll
      for (int u = 0; u < {per}; ++u) {{
        const int c = threadIdx.x + 256 * u, kk = c / {LC // 8}, l8 = (c % {LC // 8}) * 8;
        if (c < {KC * LC // 8}) *reinterpret_cast<uint4*>(bigs + (buf * {KC} + kk) * {LC} + l8) = rb[u];
      }}
#pragma unroll
      for (int u = 0; u < {-(-KC * sp // 256)}; ++u) {{
        const int c = threadIdx.x + 256 * u;
        if (c < {KC * sp}) smalls[buf * {KC * sp} + c] = rs[u];
      }}
    }};
    __syncthreads();                                 // shared memory of the previous task is free
    fetch(0);
    place(0);
    __syncthreads();
    float acc = 0.0f;
    for (int sl = 0; sl < slabs; ++sl) {{
      if (sl + 1 < slabs) fetch(sl + 1);
      const int n_k = min({KC}, k1 - (k0 + sl * {KC})), buf = sl & 1;
      const __nv_bfloat16* bk = bigs + buf * {KC * LC} + lt;
      const float* sk = smalls + buf * {KC * sp} + (sidx < {sp} ? sidx : 0);
      if (sidx < {sp})
        for (int z = 0; z < n_k; ++z) acc = fmaf(__bfloat162float(bk[z * {LC}]), sk[z * {sp}], acc);
      if (sl + 1 < slabs) place((sl + 1) & 1);
      __syncthreads();
    }}
    if (sidx < {S} && l < {L}) __stcg(latch_part + (long long)part * {m * n}LL + (long long){row} * {n}LL + {col}, acc);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) latch_last = atomicAdd(latch_cnt + chunk, 1u) == {parts - 1}u;
    __syncthreads();
    if (latch_last) {{
      __threadfence();
      if (sidx < {S} && l < {L}) {{
        const long long o = (long long){row} * {n}LL + {col};
        float total = 0.0f;
        for (int q = 0; q < {parts}; ++q) total += __ldcg(latch_part + (long long)q * {m * n}LL + o);
        {out}[o] = __float2bfloat16_rn(total);
      }}
      if (threadIdx.x == 0) latch_cnt[chunk] = 0u;   // counters are restored to zero for later phases
    }}
  }}
}}"""


def _skinny_split_ok(m, n, k, sam, sak, sbk, sbn, la, lb):
    small_n = n <= 16
    if not (small_n or m <= 16):
        return False
    L = m if small_n else n
    unit, kstride = ((sam, sak) if small_n else (sbn, sbk))
    offset = la.offset if small_n else lb.offset
    return unit == 1 and kstride % 8 == 0 and offset % 8 == 0 and L % 8 == 0 and L >= 64


def emit(a, b, out, la, lb, shape, split_grid=0):
    from .lower import LoweringError, _ident
    if len(la.shape) not in (2, 3) or len(lb.shape) != len(la.shape):
        raise LoweringError('BF16 mm/bmm requires equal rank-2 or rank-3 operands')
    m, k = la.shape[-2:]
    kb, n = lb.shape[-2:]
    batched = len(la.shape) == 3
    batch = la.shape[0] if batched else 1
    if k != kb or (batched and lb.shape[0] != batch) or shape != ((batch, m, n) if batched else (m, n)):
        raise LoweringError('BF16 matrix shape mismatch')
    sam, sak = la.strides[-2:]
    sbk, sbn = lb.strides[-2:]
    bsa, bsb = (la.strides[0], lb.strides[0]) if batched else (0, 0)
    vector_safe = ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)
                   and all(s % 8 == 0 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk,
                                                bsa, bsb, la.offset, lb.offset)))
    if m % 64 or n % 64 or k % 32 or not vector_safe:
        mt, nt = (m + 15) // 16, (n + 15) // 16
        tiles = batch * mt * nt
        parts = _split_parts(tiles, k, split_grid)
        if P9E_SKINNY and not batched and k <= 16 and k * n * 4 <= 16384:
            return _skinny_k(a, b, out, m, n, k, sam, sak, sbk, sbn, _ident), {
                'tiles': tiles, 'engine': 'p9e_skinny_k', 'batch': batch, 'shared_bytes': max(2048, k * n * 4),
                'arithmetic': 'explicit BF16 operands/output; FP32 ascending-K fmaf from +0 per element; one rounding '
                              '(the generic tile arithmetic)'}
        if parts > 1 and P9E_SKINNY and not batched and _skinny_split_ok(m, n, k, sam, sak, sbk, sbn, la, lb):
            from .vendor import plan as arena_plan
            kp = -(-k // parts)
            S = n if n <= 16 else m
            L = m if n <= 16 else n
            sp = 1
            while sp < S:
                sp *= 2
            lc = 256 // sp
            while -(-L // lc) * parts < SKINNY_MIN_TASKS and lc > 16:
                lc //= 2
            chunks = -(-L // lc)
            if P9E_SKINNY_PARTS and split_grid:
                parts = max(parts, min(-(-k // SPLIT_PART_K), split_grid // chunks))
                kp = -(-k // parts)
            code = _skinny_split(a, b, out, m, n, k, sam, sak, sbk, sbn, parts, kp, _ident)
            return code, {'tiles': chunks * parts, 'engine': 'p9e_skinny_split', 'batch': batch,
                          'shared_bytes': 2 * SKINNY_KC * (256 // sp) * 2 + 2 * SKINNY_KC * sp * 4,
                          'work_items': chunks * parts,
                          'p2_split': {'counter_words': arena_plan.a64(chunks), 'partial_words': arena_plan.a64(parts * m * n)},
                          'arithmetic': f'explicit BF16 operands/output; FP32 ascending-K fmaf from +0 per K part ({parts} parts '
                                        f'of {kp}, as the generic split); fixed-order part sum; one rounding to BF16'}
        if parts > 1:
            from .vendor import plan as arena_plan
            kp = -(-k // parts)
            code = f'''{{ // P8-E split generic: {tiles} tiles x {parts} K parts of {kp}
  __shared__ int latch_last;
  unsigned* latch_cnt = /*P2_CNT*/;
  float* latch_part = /*P2_PART*/;
  for (long long task = blockIdx.x; task < {tiles * parts}LL; task += gridDim.x) {{
    const long long tile = task / {parts}LL;
    const int part = (int)(task % {parts}LL);
    const long long batch = tile / {mt*nt}LL;
    const int row = (tile / {nt}LL % {mt}LL) * 16 + threadIdx.x / 16;
    const int col = (tile % {nt}LL) * 16 + threadIdx.x % 16;
    const int k0 = part * {kp}, k1 = min({k}, k0 + {kp});
    float acc = 0.0f;
    for (int kk = k0; kk < k1; kk += 16) {{
      const int ak = kk + threadIdx.x % 16, bk = kk + threadIdx.x / 16;
      smem[threadIdx.x] = row < {m} && ak < k1 ? __bfloat162float({_ident(a)}[batch * {bsa}LL + row * {sam}LL + ak * {sak}LL]) : 0.0f;
      smem[256 + threadIdx.x] = bk < k1 && col < {n} ? __bfloat162float({_ident(b)}[batch * {bsb}LL + bk * {sbk}LL + col * {sbn}LL]) : 0.0f;
      __syncthreads();
      for (int z = 0; z < 16 && kk + z < k1; ++z)
        acc = fmaf(smem[(threadIdx.x / 16) * 16 + z], smem[256 + z * 16 + threadIdx.x % 16], acc);
      __syncthreads();
    }}
    __stcg(latch_part + task * 256LL + threadIdx.x, acc);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) latch_last = atomicAdd(latch_cnt + tile, 1u) == {parts - 1}u;
    __syncthreads();
    if (latch_last) {{
      __threadfence();
      float total = 0.0f;
      for (int q = 0; q < {parts}; ++q) total += __ldcg(latch_part + (tile * {parts}LL + q) * 256LL + threadIdx.x);
      if (row < {m} && col < {n}) {out}[batch * {m*n}LL + row * {n}LL + col] = __float2bfloat16_rn(total);
      if (threadIdx.x == 0) latch_cnt[tile] = 0u;   // counters are restored to zero for later phases
    }}
  }}
}}'''
            return code, {'tiles': tiles * parts, 'engine': 'generic_bf16_simt_16_split', 'batch': batch,
                          'shared_bytes': 2048, 'work_items': tiles * parts,
                          'p2_split': {'counter_words': arena_plan.a64(tiles), 'partial_words': arena_plan.a64(tiles * parts * 256)},
                          'arithmetic': f'explicit BF16 operands/output; FP32 ascending-K fmaf from +0 per K part ({parts} parts '
                                        f'of {kp}); fixed-order part sum; one rounding to BF16'}
        # Generic actual-shape fallback, shared with the FP32 SIMT schedule.
        # It never reads padding. Every BF16 operand is converted exactly to
        # FP32, accumulated with ascending-K fmaf, then rounded once to BF16.
        code = f'''for (long long tile = blockIdx.x; tile < {tiles}LL; tile += gridDim.x) {{
  const long long batch = tile / {mt*nt}LL;
  const int row = (tile / {nt}LL % {mt}LL) * 16 + threadIdx.x / 16;
  const int col = (tile % {nt}LL) * 16 + threadIdx.x % 16;
  float acc = 0.0f;
  for (int kk = 0; kk < {k}; kk += 16) {{
    const int ak = kk + threadIdx.x % 16, bk = kk + threadIdx.x / 16;
    smem[threadIdx.x] = row < {m} && ak < {k} ? __bfloat162float({_ident(a)}[batch * {bsa}LL + row * {sam}LL + ak * {sak}LL]) : 0.0f;
    smem[256 + threadIdx.x] = bk < {k} && col < {n} ? __bfloat162float({_ident(b)}[batch * {bsb}LL + bk * {sbk}LL + col * {sbn}LL]) : 0.0f;
    __syncthreads();
    for (int z = 0; z < 16 && kk + z < {k}; ++z)
      acc = fmaf(smem[(threadIdx.x / 16) * 16 + z], smem[256 + z * 16 + threadIdx.x % 16], acc);
    __syncthreads();
  }}
  if (row < {m} && col < {n}) {out}[batch * {m*n}LL + row * {n}LL + col] = __float2bfloat16_rn(acc);
}}'''
        return code, {'tiles': tiles, 'engine': 'generic_bf16_simt_16', 'batch': batch, 'shared_bytes': 2048,
                      'arithmetic': 'explicit BF16 operands/output; FP32 ascending-K fmaf; no padding reads or split-K'}
    # KS32 satisfies R*KS/8 % NTHREADS == 0 for both 64-row operands.
    # Three BF16 cp.async stages alias the later FP32 epilogue staging.
    a_elements = 64 * 40 if sak == 1 else 32 * 72
    b_elements = 64 * 40 if sbk == 1 else 32 * 72
    shared_bytes = max(2 * 3 * (a_elements + b_elements), 4 * 64 * 72)
    code = f'''{{
  s101h::GemmArgs<1> g{{}};
  g.p[0] = {{reinterpret_cast<const float*>({_ident(a)}), {sam}LL, {sak}LL, {bsa}LL, 0,
             reinterpret_cast<const float*>({_ident(b)}), {sbk}LL, {sbn}LL, {bsb}LL, 0, {k}}};
  g.M = {m}; g.N = {n}; g.nb = {batch}; g.nb2 = 1; g.ksplit = 1;
  s101h::bgemm_phase<64,64,{str(sak == 1).lower()},{str(sbk == 1).lower()},1,32,false>(
      g, LatchOpsStoreBF16{{{out}, {n}LL, {m*n}LL}}, smem);
}}'''
    return code, {'tiles': batch * (m // 64) * (n // 64), 'engine': 'pinned_bf16_64',
                  'batch': batch, 'shared_bytes': shared_bytes,
                  'arithmetic': 'explicit BF16 operands/output; FP32 MMA accumulation; ascending K; no split-K'}
