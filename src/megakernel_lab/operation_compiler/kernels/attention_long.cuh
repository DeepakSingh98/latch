// P7-C long-context training attention work items for the LATCH-OPS cooperative program (new code in this lane).
//
// Scope: the P-OPS3-B/P-OPS4-E BF16-boundary attention contract (pshort::Args, attention_short.cuh) at context
// T % 128 == 0 (T <= 4096) and head width 64: query/key/value/output-gradient operands are FP32 widenings of BF16
// values, an additive FP32 mask tensor with broadcast strides, a captured scalar score scale; outputs rounded once to
// BF16 (their consumers are BF16 casts) plus the FP32 output O and row log-sum-exp for the loop tape.
//
// Arithmetic (declared, same class as attention_short.cuh): tensor-core mma.sync m16n8k16 BF16 x BF16 products with
// FP32 accumulation. Q K^T and dO V^T have BF16-exact operands (exact products). FP32 operands (probabilities P and
// scaled score gradients dS) are split exactly into three BF16 pieces, so every elementary product is exact and only
// the FP32 accumulation order differs from the conventional kernels. Scores x = fl(fl(dot) op c) + mask. Forward:
// row max m and l = sum exp(x - m) in one sweep with the online rescaling l <- l exp(m_old - m_new) (FP32-level
// difference from the captured single sum, as the selected split-K short forward); then P = exp(x - m) / l with
// expf and a correctly rounded division, O = P V, lse = m + log(l). (Resident forward: row max first, then
// e = exp(x - m) once per element with l = sum e and O = (sum split3(e) V) / l, one division per output element.) Backward recomputes P = exp(x - lse); dS = P (dP -
// delta) with delta = rowsum(dO * O_fp32) in a fixed order (identical code in both backward item kinds); dS0 = dS op
// c. No atomics: every output element has one writer.
//
// Exact tile skipping (runtime guard, any mask): a 16-row x 64-column tile is skipped when, for every row i of the
// tile, |c| |q_i| max_j |k_j| (1 + 2^-10) + max(mask tile) < m_i - 120 (forward, m_i the row max over the computed
// tiles, which only grows) or < lse_i - 120 (backward). Then every x_ij - m_i < -120, so exp underflows to exactly 0:
// the skipped probabilities, their score gradients and every product contribution are exactly zero, and the computed
// values equal the dense computation up to the order of exact zero additions. A tile that fails the guard is computed.
//
// Work decomposition: forward item (bh, 128-query block), 8 warps x 16 rows, key blocks of 64 streamed; backward has
// key/value items (bh, 128-key block: dK and dV rows, query blocks of 64 streamed) and query items (bh, 128-query
// block: dQ rows, key blocks of 64 streamed), both kinds in one phase. Items run in a compiler-supplied
// decreasing-cost order with boustrophedon CTA assignment. Requires the program's 256-thread CTA.
#pragma once
#include "attention_short.cuh"

namespace plong {

#define PLONG_DEV __device__ __forceinline__
using pshort::Args;
using pshort::NW;
using bf16 = __nv_bfloat16;
constexpr int HD = 64, LD = HD + 8, KS = HD / 16;
constexpr float MARGIN = 120.f;           // exp(-120) underflows to +0 in FP32 (subnormal limit ~ -103.97)
constexpr float NORM_SLACK = 1.0009765625f;  // 1 + 2^-10: covers FP32 rounding of norms, dot and products

PLONG_DEV float scale_abs(const Args& a) {
  return a.scale_mode == 1 ? fabsf(a.scale) : (a.scale_mode == 2 ? 1.f / fabsf(a.scale) : 1.f);
}
PLONG_DEV float mask_at(const Args& a, int bh, int i, int j) {
  return a.has_mask ? __ldg(a.mask + (long long)bh * a.m_b + (long long)i * a.m_i + (long long)j * a.m_j) : 0.f;
}

// Euclidean norms of ROWS BF16 rows of a shared [ROWS][LD] block into out[ROWS] (fixed order per row).
template <int ROWS>
PLONG_DEV void row_norms(const bf16* s, float* out) {
  constexpr int PER = 256 / ROWS;           // threads per row (2 for 128 rows, 4 for 64 rows)
  constexpr int EACH = HD / PER;
  const int row = threadIdx.x / PER, part = threadIdx.x % PER;
  float acc = 0.f;
#pragma unroll
  for (int d = 0; d < EACH; ++d) {
    const float x = __bfloat162float(s[row * LD + part * EACH + d]);
    acc = __fadd_rn(acc, __fmul_rn(x, x));
  }
#pragma unroll
  for (int o = 1; o < PER; o <<= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
  if (part == 0) out[row] = sqrtf(acc);
}

// delta_i = sum_d fl(dO_id * O_id) for 64 query rows starting at q0 into out[64]. Fixed order: four contiguous
// quarter sums, combined by an xor butterfly (identical bits in every lane and in both backward item kinds).
PLONG_DEV void row_delta(const Args& a, int bh, int q0, float* out) {
  const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
  const bf16* d = a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)(q0 + row) * a.ld.sr + part * 16;
  const float* o = a.o32 + ((long long)bh * a.t + q0 + row) * HD + part * 16;
  float acc = 0.f;
#pragma unroll
  for (int k = 0; k < 16; ++k) acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(d[k]), __ldcg(o + k)));
  acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
  acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
  if (part == 0) out[row] = acc;
}

// row_delta with 16-byte loads (2 x uint4 of dO, 4 x float4 of O per thread); same products and addition order,
// identical bits. BLOCKS consecutive 64-row blocks are loaded before any is reduced (memory-level parallelism).
template <int BLOCKS>
PLONG_DEV void row_delta_vec(const Args& a, int bh, int q0, float* out) {
  const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
  uint4 dv[BLOCKS][2];
  float4 ov[BLOCKS][4];
#pragma unroll
  for (int b = 0; b < BLOCKS; ++b) {
    const bf16* d = a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)(q0 + 64 * b + row) * a.ld.sr + part * 16;
    const float* o = a.o32 + ((long long)bh * a.t + q0 + 64 * b + row) * HD + part * 16;
    dv[b][0] = __ldcg(reinterpret_cast<const uint4*>(d));
    dv[b][1] = __ldcg(reinterpret_cast<const uint4*>(d) + 1);
#pragma unroll
    for (int k = 0; k < 4; ++k) ov[b][k] = __ldcg(reinterpret_cast<const float4*>(o) + k);
  }
#pragma unroll
  for (int b = 0; b < BLOCKS; ++b) {
    const uint32_t w[8] = {dv[b][0].x, dv[b][0].y, dv[b][0].z, dv[b][0].w, dv[b][1].x, dv[b][1].y, dv[b][1].z, dv[b][1].w};
    const float of[16] = {ov[b][0].x, ov[b][0].y, ov[b][0].z, ov[b][0].w, ov[b][1].x, ov[b][1].y, ov[b][1].z, ov[b][1].w,
                          ov[b][2].x, ov[b][2].y, ov[b][2].z, ov[b][2].w, ov[b][3].x, ov[b][3].y, ov[b][3].z, ov[b][3].w};
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
      acc = __fadd_rn(acc, __fmul_rn(dk, of[k]));
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    if (part == 0) out[64 * b + row] = acc;
  }
}

// Max over the warp of a value (every lane receives it).
PLONG_DEV float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
PLONG_DEV float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}
PLONG_DEV float quad_sum(float v) {
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

// Load ROWS BF16 rows of HD units (strided source) into shared [ROWS][LD] with 16-byte cp.async copies.
template <int ROWS>
PLONG_DEV void load_block(bf16* dst, const bf16* src, long long row_stride) {
  constexpr int CH = HD / 8, TOTAL = ROWS * CH;
#pragma unroll
  for (int k = 0; k < (TOTAL + 255) / 256; ++k) {
    const int idx = (int)threadIdx.x + 256 * k;
    if (TOTAL % 256 == 0 || idx < TOTAL) {
      const int r = idx / CH, c = idx % CH;
      pshort::cp16(dst + r * LD + 8 * c, src + (long long)r * row_stride + 8 * c);
    }
  }
}

// Scores of a 16-row x 64-column tile: acc[nt] (8 n8 tiles) = A (registers, 16 rows x HD) times B rows (shared
// [64][LD], the column operand stored row-major by column index).
PLONG_DEV void tile_scores(float (&acc)[8][4], const uint32_t (&af)[KS][4], const bf16* sb, int lane) {
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      uint32_t b[2];
      pshort::ldsm_x2(b, sb + (nt * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
      pshort::mma(acc[nt], af[ks], b);
    }
  }
}

// acc[nh] (8 head n-tiles) += sum over 64 columns of split3(P tile) times rows of sv ([64][LD], k = column index).
PLONG_DEV void tile_split_product(float (&acc)[8][4], const float (&p)[8][4], const bf16* sv, int lane) {
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) {
    uint32_t a0[3], a1[3], a2[3], a3[3];
    pshort::split2<3>(a0, p[2 * k2][0], p[2 * k2][1]);
    pshort::split2<3>(a1, p[2 * k2][2], p[2 * k2][3]);
    pshort::split2<3>(a2, p[2 * k2 + 1][0], p[2 * k2 + 1][1]);
    pshort::split2<3>(a3, p[2 * k2 + 1][2], p[2 * k2 + 1][3]);
#pragma unroll
    for (int nh = 0; nh < 8; ++nh) {
      uint32_t b[2];
      pshort::ldsm_x2_t(b, sv + (k2 * 16 + (lane & 15)) * LD + nh * 8);
#pragma unroll
      for (int s = 0; s < 3; ++s) {
        const uint32_t af[4] = {a0[s], a1[s], a2[s], a3[s]};
        pshort::mma(acc[nh], af, b);
      }
    }
  }
}

PLONG_DEV void load_afrag(uint32_t (&af)[KS][4], const bf16* s, int lane) {
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) pshort::ldsm_x4(af[ks], s + (lane & 15) * LD + ks * 16 + (lane >> 4) * 8);
}

// ---------------------------------------------------------------------------------------------------- forward
// Shared: Q [128][LD], K [64][LD], V [64][LD] (BF16); q norms [128], key-norm scratch [16], per-warp tile mask
// max [NW][64] (FP32).
__host__ __device__ constexpr int forward_shared_bytes() { return 2 * (128 + 64 + 64) * LD + 4 * (128 + 16 + NW * 64); }

PLONG_DEV void forward_item(const Args& a, int bh, int qblk, float* smem) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + 64 * LD;
  float* qn = reinterpret_cast<float*>(sv + 64 * LD);
  float* kn = qn + 128;
  float* tmax = kn + 16;
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  load_block<128>(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sq, qn);
  uint32_t qa[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  __syncthreads();
  const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  unsigned long long computed = 0ull;

  // One computed tile of the statistics sweep: row max and online-rescaled exponential sum.
  auto stats_tile = [&](int kb) {
    float acc[8][4];
    tile_scores(acc, qa, sk, lane);
    float x[8][4], tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = kb * 64 + nt * 8 + 2 * tig + (e & 1);
        x[nt][e] = pshort::scaled(a, acc[nt][e]) + mask_at(a, bh, e < 2 ? r0 : r1, col);
        if (e < 2) tm0 = fmaxf(tm0, x[nt][e]); else tm1 = fmaxf(tm1, x[nt][e]);
      }
    const float n0 = fmaxf(m0, quad_max(tm0)), n1 = fmaxf(m1, quad_max(tm1));
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        if (e < 2) s0 = __fadd_rn(s0, expf(x[nt][e] - n0)); else s1 = __fadd_rn(s1, expf(x[nt][e] - n1));
      }
    s0 = quad_sum(s0);
    s1 = quad_sum(s1);
    l0 = __fadd_rn(m0 == -INFINITY ? 0.f : __fmul_rn(l0, expf(m0 - n0)), s0);
    l1 = __fadd_rn(m1 == -INFINITY ? 0.f : __fmul_rn(l1, expf(m1 - n1)), s1);
    m0 = n0;
    m1 = n1;
    computed |= 1ull << kb;
  };

  // Sweep A: tiles whose mask maximum exceeds -1000 (every unmasked and partially masked tile).
  for (int kb = 0; kb < nkb; ++kb) {
    float mt = -INFINITY;
    if (a.has_mask) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          mt = fmaxf(mt, mask_at(a, bh, e < 2 ? r0 : r1, kb * 64 + nt * 8 + 2 * tig + (e & 1)));
      mt = warp_max(mt);
    } else {
      mt = 0.f;
    }
    if (lane == 0) tmax[warp * 64 + kb] = mt;
    const bool live = mt > -1000.f;
    if (__syncthreads_or(live)) {
      load_block<64>(sk, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr);
      pshort::cp_commit_wait();
      __syncthreads();
      if (live) stats_tile(kb);
      __syncthreads();
    }
  }
  // Sweep B: the remaining tiles, skipped under the exact-zero guard, otherwise computed.
  for (int kb = 0; kb < nkb; ++kb) {
    const bool candidate = !((computed >> kb) & 1ull);
    if (!__syncthreads_or(candidate)) continue;
    load_block<64>(sk, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr);
    pshort::cp_commit_wait();
    __syncthreads();
    {  // block max key norm: 64 rows, 4 threads per row, warp maxima, then a CTA max into kn[NW]
      const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        const float x = __bfloat162float(sk[row * LD + part * 16 + d]);
        acc = __fadd_rn(acc, __fmul_rn(x, x));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
      const float mx = warp_max(sqrtf(acc));
      if (lane == 0) kn[warp] = mx;
      __syncthreads();
      if (threadIdx.x == 0) {
        float m = kn[0];
#pragma unroll
        for (int w = 1; w < NW; ++w) m = fmaxf(m, kn[w]);
        kn[NW] = m;
      }
      __syncthreads();
    }
    if (candidate) {
      const float mt = tmax[warp * 64 + kb], knb = kn[NW];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), knb), NORM_SLACK) + mt;
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), knb), NORM_SLACK) + mt;
      const bool safe = (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN);
      if (!__all_sync(0xffffffffu, safe)) stats_tile(kb);
    }
    __syncthreads();
  }
  // Output sweep over the computed tiles: P = exp(x - m) / l, O += split3(P) V.
  float oacc[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  for (int kb = 0; kb < nkb; ++kb) {
    const bool mine = (computed >> kb) & 1ull;
    if (!__syncthreads_or(mine)) continue;
    load_block<64>(sk, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr);
    load_block<64>(sv, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr);
    pshort::cp_commit_wait();
    __syncthreads();
    if (mine) {
      float acc[8][4];
      tile_scores(acc, qa, sk, lane);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int col = kb * 64 + nt * 8 + 2 * tig + (e & 1);
          const float x = pshort::scaled(a, acc[nt][e]) + mask_at(a, bh, e < 2 ? r0 : r1, col);
          acc[nt][e] = __fdiv_rn(expf(x - (e < 2 ? m0 : m1)), e < 2 ? l0 : l1);
        }
      tile_split_product(oacc, acc, sv, lane);
    }
    __syncthreads();
  }
  bf16* ob = a.o + pshort::head_offset(a.lo, bh, a.heads);
  float* o32 = a.o32 + (long long)bh * t * HD;
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(ob + (long long)r0 * a.lo.sr + col, oacc[nh][0], oacc[nh][1]);
    pshort::store_bf16x2(ob + (long long)r1 * a.lo.sr + col, oacc[nh][2], oacc[nh][3]);
    *reinterpret_cast<float2*>(o32 + (long long)r0 * HD + col) = make_float2(oacc[nh][0], oacc[nh][1]);
    *reinterpret_cast<float2*>(o32 + (long long)r1 * HD + col) = make_float2(oacc[nh][2], oacc[nh][3]);
  }
  if (tig == 0) {
    a.lse[(long long)bh * t + r0] = m0 + logf(l0);
    a.lse[(long long)bh * t + r1] = m1 + logf(l1);
  }
}

// --------------------------------------------------------------------------------------------------- backward
// Shared (largest item): two [128][LD] and two [64][LD] BF16 blocks; lse, delta and norms [128] each; key-norm and
// reduction scratch (FP32).
__host__ __device__ constexpr int backward_shared_bytes() { return 2 * (2 * 128 + 2 * 64) * LD + 4 * (4 * 128 + 16); }

// ---------------------------------------------------------------------------------------- key/value item
PLONG_DEV void kv_item(const Args& a, int bh, int kblk, float* smem) {
  bf16* sk = reinterpret_cast<bf16*>(smem);   // [128][LD] keys of this item
  bf16* sv = sk + 128 * LD;                    // [128][LD] values
  bf16* sq = sv + 128 * LD;                    // [64][LD] query block
  bf16* sd = sq + 64 * LD;                     // [64][LD] output-gradient block
  float* slse = reinterpret_cast<float*>(sd + 64 * LD);  // [64]
  float* sdel = slse + 128;                    // [64]
  float* sqn = sdel + 128;                     // [64]
  float* kn = sqn + 128;                       // [128] key norms
  const int t = a.t, nqb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = kblk * 128, j0 = k0 + warp * 16 + g, j1 = j0 + 8;
  const float sabs = scale_abs(a);
  load_block<128>(sk, a.k + pshort::head_offset(a.lk, bh, a.heads) + (long long)k0 * a.lk.sr, a.lk.sr);
  load_block<128>(sv, a.v + pshort::head_offset(a.lv, bh, a.heads) + (long long)k0 * a.lv.sr, a.lv.sr);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sk, kn);
  uint32_t ka[KS][4], va[KS][4];
  load_afrag(ka, sk + warp * 16 * LD, lane);
  load_afrag(va, sv + warp * 16 * LD, lane);
  __syncthreads();
  const float knw = warp_max(fmaxf(kn[warp * 16 + g], kn[warp * 16 + g + 8]));
  float dv[8][4], dk[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh)
#pragma unroll
    for (int e = 0; e < 4; ++e) dv[nh][e] = dk[nh][e] = 0.f;
  const bf16* qbase = a.q + pshort::head_offset(a.lq, bh, a.heads);
  const bf16* dbase = a.dout + pshort::head_offset(a.ld, bh, a.heads);
  for (int qb = 0; qb < nqb; ++qb) {
    const int q0 = qb * 64;
    load_block<64>(sq, qbase + (long long)q0 * a.lq.sr, a.lq.sr);
    load_block<64>(sd, dbase + (long long)q0 * a.ld.sr, a.ld.sr);
    if (threadIdx.x < 64) slse[threadIdx.x] = __ldcg(a.lse + (long long)bh * t + q0 + threadIdx.x);
    row_delta(a, bh, q0, sdel);
    pshort::cp_commit_wait();
    __syncthreads();
    row_norms<64>(sq, sqn);
    __syncthreads();
    // Guard: tile (warp's 16 keys x 64 queries) skipped when every element's bound is below lse_i - MARGIN.
    float mt = 0.f, mk[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        mk[nt][e] = mask_at(a, bh, q0 + nt * 8 + 2 * tig + (e & 1), e < 2 ? j0 : j1);
        mt = (nt == 0 && e == 0) ? mk[nt][e] : fmaxf(mt, mk[nt][e]);
      }
    mt = warp_max(mt);
    bool safe = true;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int qi = nt * 8 + 2 * tig + e;
        const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, sqn[qi]), knw), NORM_SLACK) + mt;
        safe = safe && (b < slse[qi] - MARGIN);
      }
    if (!__all_sync(0xffffffffu, safe)) {
      float s[8][4], dp[8][4];
      tile_scores(s, ka, sq, lane);   // S^T: rows = keys, columns = queries
      tile_scores(dp, va, sd, lane);  // dP^T = V dO^T
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int qi = nt * 8 + 2 * tig + (e & 1);
          const float x = pshort::scaled(a, s[nt][e]) + mk[nt][e];
          const float p = expf(x - slse[qi]);
          s[nt][e] = p;
          dp[nt][e] = pshort::scaled(a, __fmul_rn(p, dp[nt][e] - sdel[qi]));
        }
      tile_split_product(dv, s, sd, lane);   // dV += P^T dO
      tile_split_product(dk, dp, sq, lane);  // dK += dS0^T Q
    }
    __syncthreads();
  }
  bf16* dkb = a.dk + pshort::head_offset(a.ldk, bh, a.heads);
  bf16* dvb = a.dv + pshort::head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dkb + (long long)j0 * a.ldk.sr + col, dk[nh][0], dk[nh][1]);
    pshort::store_bf16x2(dkb + (long long)j1 * a.ldk.sr + col, dk[nh][2], dk[nh][3]);
    pshort::store_bf16x2(dvb + (long long)j0 * a.ldv.sr + col, dv[nh][0], dv[nh][1]);
    pshort::store_bf16x2(dvb + (long long)j1 * a.ldv.sr + col, dv[nh][2], dv[nh][3]);
  }
}

// ------------------------------------------------------------------------------------------------- query item
PLONG_DEV void q_item(const Args& a, int bh, int qblk, float* smem) {
  bf16* sq = reinterpret_cast<bf16*>(smem);   // [128][LD]
  bf16* sd = sq + 128 * LD;                    // [128][LD]
  bf16* sk = sd + 128 * LD;                    // [64][LD]
  bf16* sv = sk + 64 * LD;                     // [64][LD]
  float* sdel = reinterpret_cast<float*>(sv + 64 * LD);  // [128]
  float* sqn = sdel + 128;                     // [128]
  float* red = sqn + 128;                      // [NW + 1]
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  load_block<128>(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr);
  load_block<128>(sd, a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)q0 * a.ld.sr, a.ld.sr);
  row_delta(a, bh, q0, sdel);
  row_delta(a, bh, q0 + 64, sdel + 64);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sq, sqn);
  uint32_t qa[KS][4], da[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  load_afrag(da, sd + warp * 16 * LD, lane);
  __syncthreads();
  const float lse0 = __ldcg(a.lse + (long long)bh * t + r0), lse1 = __ldcg(a.lse + (long long)bh * t + r1);
  const float del0 = sdel[warp * 16 + g], del1 = sdel[warp * 16 + g + 8];
  const float qn0 = sqn[warp * 16 + g], qn1 = sqn[warp * 16 + g + 8];
  float dq[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) dq[nh][0] = dq[nh][1] = dq[nh][2] = dq[nh][3] = 0.f;
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  for (int kb = 0; kb < nkb; ++kb) {
    float mt = -INFINITY, mk[8][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        mk[nt][e] = mask_at(a, bh, e < 2 ? r0 : r1, kb * 64 + nt * 8 + 2 * tig + (e & 1));
        mt = fmaxf(mt, mk[nt][e]);
      }
    mt = warp_max(mt);
    load_block<64>(sk, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr);
    load_block<64>(sv, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr);
    pshort::cp_commit_wait();
    __syncthreads();
    {
      const int row = threadIdx.x >> 2, part = threadIdx.x & 3;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        const float x = __bfloat162float(sk[row * LD + part * 16 + d]);
        acc = __fadd_rn(acc, __fmul_rn(x, x));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
      const float mx = warp_max(sqrtf(acc));
      if (lane == 0) red[warp] = mx;
    }
    __syncthreads();
    float knb = red[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) knb = fmaxf(knb, red[w]);
    const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), knb), NORM_SLACK) + mt;
    const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), knb), NORM_SLACK) + mt;
    const bool safe = (b0 < lse0 - MARGIN) && (b1 < lse1 - MARGIN);
    if (!__all_sync(0xffffffffu, safe)) {
      float s[8][4], dp[8][4];
      tile_scores(s, qa, sk, lane);   // S = Q K^T
      tile_scores(dp, da, sv, lane);  // dP = dO V^T
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x = pshort::scaled(a, s[nt][e]) + mk[nt][e];
          const float p = expf(x - (e < 2 ? lse0 : lse1));
          dp[nt][e] = pshort::scaled(a, __fmul_rn(p, dp[nt][e] - (e < 2 ? del0 : del1)));
        }
      tile_split_product(dq, dp, sk, lane);  // dQ += dS0 K
    }
    __syncthreads();
  }
  bf16* dqb = a.dq + pshort::head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dqb + (long long)r0 * a.ldq.sr + col, dq[nh][0], dq[nh][1]);
    pshort::store_bf16x2(dqb + (long long)r1 * a.ldq.sr + col, dq[nh][2], dq[nh][3]);
  }
}

// ------------------------------------------------------------------------- resident variants (T <= RES_MAX)
// Same arithmetic, guard and outputs as the streaming items above. The whole head's streamed operand (K and V for
// forward/query items, Q and dO for key/value items) is copied to shared memory once per item, so warps proceed
// through their tiles without CTA barriers. Tiles whose mask values are all +0 skip the mask loads and the +0
// addition (x + 0 == x exactly; only -0 becomes +0, which exp maps identically).
constexpr int RES_MAX = 512;
__host__ __device__ constexpr int res_forward_shared_bytes() { return 2 * (128 + 2 * RES_MAX) * LD + 4 * (128 + RES_MAX / 64 + 16); }
__host__ __device__ constexpr int res_backward_shared_bytes() { return 2 * (256 + 2 * RES_MAX) * LD + 4 * (3 * RES_MAX + 128 + 16); }

// Mask fragment of a 16 x 64 tile (rows r0/r1, columns kb*64 + ...) or its transpose (rows are keys j0/j1, columns
// queries q0 + ...). Returns the warp-wide maximum and whether every value is exactly +0 (sign included).
template <bool TRANS>
PLONG_DEV float tile_mask(const Args& a, int bh, int ra, int rb, int c0, float (&mk)[8][4], bool& zero) {
  float mt = -INFINITY;
  unsigned nz = 0u;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = c0 + nt * 8 + 2 * (threadIdx.x & 3) + (e & 1), row = e < 2 ? ra : rb;
      const float m = TRANS ? mask_at(a, bh, col, row) : mask_at(a, bh, row, col);
      mk[nt][e] = m;
      mt = fmaxf(mt, m);
      nz |= __float_as_uint(m);
    }
  zero = __all_sync(0xffffffffu, nz == 0u);
  return warp_max(mt);
}

// Warp-tile mask classification with 16-byte loads: (max, all +0) of mask rows [r0, r0 + R) x columns [c0, c0 + C)
// for R x C = 16 x 64 (query-row tiles) or 64 x 16 (transposed key/value tiles). Requires unit column stride and
// 16-byte aligned rows (checked by the caller: vec_ok); exact same set of values as the fragment loads.
template <int R, int C>
PLONG_DEV void tile_class(const Args& a, int bh, int r0, int c0, float& mt, bool& zero) {
  constexpr int PER = R * C / 32, ROWS_PER_LANE = R / 32 > 0 ? R / 32 : 1, LANES_PER_ROW = 32 / (R < 32 ? R : 32);
  const int lane = threadIdx.x & 31;
  const int row = R >= 32 ? lane * ROWS_PER_LANE : lane / LANES_PER_ROW;
  const int col = R >= 32 ? 0 : (lane % LANES_PER_ROW) * (C / LANES_PER_ROW);
  const float* base = a.mask + (long long)bh * a.m_b + (long long)(r0 + row) * a.m_i + c0 + col;
  float m = -INFINITY;
  unsigned nz = 0u;
#pragma unroll
  for (int k = 0; k < PER / 4; ++k) {
    const int rr = R >= 32 ? k / (C / 4) : 0, cc = R >= 32 ? 4 * (k % (C / 4)) : 4 * k;
    const float4 v = __ldg(reinterpret_cast<const float4*>(base + (long long)rr * a.m_i + cc));
    m = fmaxf(fmaxf(m, fmaxf(v.x, v.y)), fmaxf(v.z, v.w));
    nz |= __float_as_uint(v.x) | __float_as_uint(v.y) | __float_as_uint(v.z) | __float_as_uint(v.w);
  }
  mt = warp_max(m);
  zero = __all_sync(0xffffffffu, nz == 0u);
}
PLONG_DEV bool vec_ok(const Args& a) {
  return a.has_mask && a.m_j == 1 && a.m_i % 4 == 0 && a.m_b % 4 == 0 && (reinterpret_cast<uintptr_t>(a.mask) & 15) == 0;
}
// Fragment mask values of a tile (only needed for tiles that are computed and not all-zero).
template <bool TRANS>
PLONG_DEV void tile_frag(const Args& a, int bh, int ra, int rb, int c0, float (&mk)[8][4]) {
  const int tig = threadIdx.x & 3;
  if (!TRANS && a.m_j == 1 && a.m_i % 2 == 0 && a.m_b % 2 == 0) {
    const float* base = a.mask + (long long)bh * a.m_b;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = c0 + nt * 8 + 2 * tig;
      const float2 u = __ldg(reinterpret_cast<const float2*>(base + (long long)ra * a.m_i + col));
      const float2 w = __ldg(reinterpret_cast<const float2*>(base + (long long)rb * a.m_i + col));
      mk[nt][0] = u.x; mk[nt][1] = u.y; mk[nt][2] = w.x; mk[nt][3] = w.y;
    }
    return;
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = c0 + nt * 8 + 2 * tig + (e & 1), row = e < 2 ? ra : rb;
      mk[nt][e] = TRANS ? mask_at(a, bh, col, row) : mask_at(a, bh, row, col);
    }
}
// (max, zero) and, when needed, fragment values: classification by 16-byte loads when admitted, else the fragment
// loads themselves (tile_mask).
template <bool TRANS>
PLONG_DEV float tile_mask2(const Args& a, int bh, int ra, int rb, int tile_r0, int c0, float (&mk)[8][4], bool& zero,
                           bool& have) {
  if (vec_ok(a)) {
    float mt;
    if (TRANS) tile_class<64, 16>(a, bh, c0, tile_r0, mt, zero);
    else tile_class<16, 64>(a, bh, tile_r0, c0, mt, zero);
    have = false;
    return mt;
  }
  have = true;
  return tile_mask<TRANS>(a, bh, ra, rb, c0, mk, zero);
}

PLONG_DEV void load_rows_res(bf16* dst, const bf16* src, long long row_stride, int rows) {
  for (int idx = (int)threadIdx.x; idx < rows * (HD / 8); idx += 256) {
    const int r = idx >> 3, c = idx & 7;
    pshort::cp16(dst + r * LD + 8 * c, src + (long long)r * row_stride + 8 * c);
  }
}

// Block maxima of row norms of `rows` shared rows (64-row blocks) into out[rows / 64]; scratch[rows] row norms.
PLONG_DEV void block_norms(const bf16* s, int rows, float* scratch, float* out) {
  for (int idx = (int)threadIdx.x; idx < rows * 4; idx += 256) {
    const int row = idx >> 2, part = idx & 3;
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      const float x = __bfloat162float(s[row * LD + part * 16 + d]);
      acc = __fadd_rn(acc, __fmul_rn(x, x));
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    if (part == 0) scratch[row] = sqrtf(acc);
  }
  __syncthreads();
  if ((int)threadIdx.x < rows / 64) {
    float m = 0.f;
    for (int r = 0; r < 64; ++r) m = fmaxf(m, scratch[threadIdx.x * 64 + r]);
    out[threadIdx.x] = m;
  }
}

PLONG_DEV void forward_item_res(const Args& a, int bh, int qblk, float* smem) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + RES_MAX * LD;
  float* qn = reinterpret_cast<float*>(sv + RES_MAX * LD);
  float* kn = qn + 128;                        // [RES_MAX / 64]
  float* scratch = reinterpret_cast<float*>(sv);  // key row norms, before V arrives (V is copied after them)
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads), a.lk.sr, t);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sq, qn);
  block_norms(sk, t, scratch, kn);
  __syncthreads();
  load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads), a.lv.sr, t);  // overlaps the score sweeps
  pshort::cp_commit();
  uint32_t qa[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
  // Pass 1: row maxima over the computed tiles (scores only). Pass 2: e = exp(x - m) once per element, l = sum e and
  // the unnormalized O = sum split3(e) V; O = O / l per output element at the end (FP32-level reassociation of the
  // normalization, the class of the selected split-K short forward).
  float m0 = -INFINITY, m1 = -INFINITY;
  unsigned long long computed = 0ull, zeros = 0ull;
  float tmx[RES_MAX / 64];
  auto max_tile = [&](int kb, const float (&mk)[8][4], bool zero) {
    float acc[8][4];
    tile_scores(acc, qa, sk + kb * 64 * LD, lane);
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, acc[nt][e]) : pshort::scaled(a, acc[nt][e]) + mk[nt][e];
        if (e < 2) tm0 = fmaxf(tm0, x); else tm1 = fmaxf(tm1, x);
      }
    m0 = fmaxf(m0, quad_max(tm0));
    m1 = fmaxf(m1, quad_max(tm1));
    computed |= 1ull << kb;
  };
  for (int kb = 0; kb < nkb; ++kb) {          // sweep A: tiles with mask maximum above -1000
    float mk[8][4];
    bool zero = true, have = false;
    const float mt = a.has_mask ? tile_mask2<false>(a, bh, r0, r1, q0 + warp * 16, kb * 64, mk, zero, have) : 0.f;
    tmx[kb] = mt;
    if (zero) zeros |= 1ull << kb;
    if (mt > -1000.f) {
      if (!zero && !have) tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
      max_tile(kb, mk, zero);
    }
  }
  for (int kb = 0; kb < nkb; ++kb) {          // sweep B: guarded skip, otherwise compute
    if ((computed >> kb) & 1ull) continue;
    const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn[kb]), NORM_SLACK) + tmx[kb];
    const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn[kb]), NORM_SLACK) + tmx[kb];
    if (__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) continue;
    float mk[8][4];
    const bool zero = (zeros >> kb) & 1ull;
    if (!zero) tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    max_tile(kb, mk, zero);
  }
  pshort::cp_commit_wait();                   // V
  __syncthreads();
  float oacc[8][4], l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  for (int kb = 0; kb < nkb; ++kb) {
    if (!((computed >> kb) & 1ull)) continue;
    const bool zero = (zeros >> kb) & 1ull;
    float mk[8][4];
    if (!zero) tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    float acc[8][4];
    tile_scores(acc, qa, sk + kb * 64 * LD, lane);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, acc[nt][e]) : pshort::scaled(a, acc[nt][e]) + mk[nt][e];
        const float ex = expf(x - (e < 2 ? m0 : m1));
        acc[nt][e] = ex;
        if (e < 2) l0 = __fadd_rn(l0, ex); else l1 = __fadd_rn(l1, ex);
      }
    tile_split_product(oacc, acc, sv + kb * 64 * LD, lane);
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  bf16* ob = a.o + pshort::head_offset(a.lo, bh, a.heads);
  float* o32 = a.o32 + (long long)bh * t * HD;
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    const float o00 = __fdiv_rn(oacc[nh][0], l0), o01 = __fdiv_rn(oacc[nh][1], l0);
    const float o10 = __fdiv_rn(oacc[nh][2], l1), o11 = __fdiv_rn(oacc[nh][3], l1);
    pshort::store_bf16x2(ob + (long long)r0 * a.lo.sr + col, o00, o01);
    pshort::store_bf16x2(ob + (long long)r1 * a.lo.sr + col, o10, o11);
    *reinterpret_cast<float2*>(o32 + (long long)r0 * HD + col) = make_float2(o00, o01);
    *reinterpret_cast<float2*>(o32 + (long long)r1 * HD + col) = make_float2(o10, o11);
  }
  if (tig == 0) {
    a.lse[(long long)bh * t + r0] = m0 + logf(l0);
    a.lse[(long long)bh * t + r1] = m1 + logf(l1);
  }
}

PLONG_DEV void kv_item_res(const Args& a, int bh, int kblk, float* smem) {
  bf16* sk = reinterpret_cast<bf16*>(smem);   // [128][LD] keys of this item
  bf16* sv = sk + 128 * LD;
  bf16* sq = sv + 128 * LD;                    // [T][LD] all queries
  bf16* sd = sq + RES_MAX * LD;                // [T][LD] all output gradients
  float* slse = reinterpret_cast<float*>(sd + RES_MAX * LD);
  float* sdel = slse + RES_MAX;
  float* sqn = sdel + RES_MAX;
  float* kn = sqn + RES_MAX;                   // [128]
  const int t = a.t, nqb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = kblk * 128, j0 = k0 + warp * 16 + g, j1 = j0 + 8;
  const float sabs = scale_abs(a);
  load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads) + (long long)k0 * a.lk.sr, a.lk.sr, 128);
  load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads) + (long long)k0 * a.lv.sr, a.lv.sr, 128);
  load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads), a.lq.sr, t);
  load_rows_res(sd, a.dout + pshort::head_offset(a.ld, bh, a.heads), a.ld.sr, t);
  pshort::cp_commit();
  for (int i = (int)threadIdx.x; i < t; i += 256) slse[i] = __ldcg(a.lse + (long long)bh * t + i);
  for (int qb = 0; qb + 1 < nqb; qb += 2) row_delta_vec<2>(a, bh, qb * 64, sdel + qb * 64);
  if (nqb & 1) row_delta_vec<1>(a, bh, (nqb - 1) * 64, sdel + (nqb - 1) * 64);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sk, kn);
  for (int qb = 0; qb < nqb; ++qb) row_norms<64>(sq + qb * 64 * LD, sqn + qb * 64);
  uint32_t ka[KS][4], va[KS][4];
  load_afrag(ka, sk + warp * 16 * LD, lane);
  load_afrag(va, sv + warp * 16 * LD, lane);
  __syncthreads();
  const float knw = warp_max(fmaxf(kn[warp * 16 + g], kn[warp * 16 + g + 8]));
  float dv[8][4], dk[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh)
#pragma unroll
    for (int e = 0; e < 4; ++e) dv[nh][e] = dk[nh][e] = 0.f;
  for (int qb = 0; qb < nqb; ++qb) {
    const int q0 = qb * 64;
    bool zero = true;
    float mt = 0.f;
    if (a.has_mask) {
      if (vec_ok(a)) {
        tile_class<64, 16>(a, bh, q0, k0 + warp * 16, mt, zero);
      } else {
        float mk[8][4];
        mt = tile_mask<true>(a, bh, j0, j1, q0, mk, zero);
      }
    }
    bool safe = true;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int qi = q0 + nt * 8 + 2 * tig + e;
        const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, sqn[qi]), knw), NORM_SLACK) + mt;
        safe = safe && (b < slse[qi] - MARGIN);
      }
    if (__all_sync(0xffffffffu, safe)) continue;
    float s[8][4], dp[8][4];
    tile_scores(s, ka, sq + q0 * LD, lane);
    tile_scores(dp, va, sd + q0 * LD, lane);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int qi = q0 + nt * 8 + 2 * tig + (e & 1);
        const float x = zero ? pshort::scaled(a, s[nt][e])
                             : pshort::scaled(a, s[nt][e]) + mask_at(a, bh, qi, e < 2 ? j0 : j1);
        const float pr = expf(x - slse[qi]);
        s[nt][e] = pr;
        dp[nt][e] = pshort::scaled(a, __fmul_rn(pr, dp[nt][e] - sdel[qi]));
      }
    tile_split_product(dv, s, sd + q0 * LD, lane);
    tile_split_product(dk, dp, sq + q0 * LD, lane);
  }
  bf16* dkb = a.dk + pshort::head_offset(a.ldk, bh, a.heads);
  bf16* dvb = a.dv + pshort::head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dkb + (long long)j0 * a.ldk.sr + col, dk[nh][0], dk[nh][1]);
    pshort::store_bf16x2(dkb + (long long)j1 * a.ldk.sr + col, dk[nh][2], dk[nh][3]);
    pshort::store_bf16x2(dvb + (long long)j0 * a.ldv.sr + col, dv[nh][0], dv[nh][1]);
    pshort::store_bf16x2(dvb + (long long)j1 * a.ldv.sr + col, dv[nh][2], dv[nh][3]);
  }
}

PLONG_DEV void q_item_res(const Args& a, int bh, int qblk, float* smem) {
  bf16* sq = reinterpret_cast<bf16*>(smem);   // [128][LD]
  bf16* sd = sq + 128 * LD;                    // [128][LD]
  bf16* sk = sd + 128 * LD;                    // [T][LD]
  bf16* sv = sk + RES_MAX * LD;                // [T][LD]
  float* sdel = reinterpret_cast<float*>(sv + RES_MAX * LD);  // [128]
  float* sqn = sdel + 128;                     // [128]
  float* kn = sqn + 128;                       // [RES_MAX / 64]
  float* scratch = kn + RES_MAX / 64;          // [RES_MAX] key row norms
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  load_rows_res(sd, a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)q0 * a.ld.sr, a.ld.sr, 128);
  load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads), a.lk.sr, t);
  load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads), a.lv.sr, t);
  pshort::cp_commit();
  row_delta_vec<2>(a, bh, q0, sdel);
  pshort::cp_commit_wait();
  __syncthreads();
  row_norms<128>(sq, sqn);
  block_norms(sk, t, scratch, kn);
  uint32_t qa[KS][4], da[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  load_afrag(da, sd + warp * 16 * LD, lane);
  __syncthreads();
  const float lse0 = __ldcg(a.lse + (long long)bh * t + r0), lse1 = __ldcg(a.lse + (long long)bh * t + r1);
  const float del0 = sdel[warp * 16 + g], del1 = sdel[warp * 16 + g + 8];
  const float qn0 = sqn[warp * 16 + g], qn1 = sqn[warp * 16 + g + 8];
  float dq[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) dq[nh][0] = dq[nh][1] = dq[nh][2] = dq[nh][3] = 0.f;
  for (int kb = 0; kb < nkb; ++kb) {
    bool zero = true;
    float mt = 0.f;
    if (a.has_mask) {
      if (vec_ok(a)) {
        tile_class<16, 64>(a, bh, q0 + warp * 16, kb * 64, mt, zero);
      } else {
        float mk[8][4];
        mt = tile_mask<false>(a, bh, r0, r1, kb * 64, mk, zero);
      }
    }
    const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn[kb]), NORM_SLACK) + mt;
    const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn[kb]), NORM_SLACK) + mt;
    if (__all_sync(0xffffffffu, (b0 < lse0 - MARGIN) && (b1 < lse1 - MARGIN))) continue;
    float s[8][4], dp[8][4];
    tile_scores(s, qa, sk + kb * 64 * LD, lane);
    tile_scores(dp, da, sv + kb * 64 * LD, lane);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, s[nt][e])
                             : pshort::scaled(a, s[nt][e]) + mask_at(a, bh, e < 2 ? r0 : r1, kb * 64 + nt * 8 + 2 * tig + (e & 1));
        const float pr = expf(x - (e < 2 ? lse0 : lse1));
        dp[nt][e] = pshort::scaled(a, __fmul_rn(pr, dp[nt][e] - (e < 2 ? del0 : del1)));
      }
    tile_split_product(dq, dp, sk + kb * 64 * LD, lane);
  }
  bf16* dqb = a.dq + pshort::head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dqb + (long long)r0 * a.ldq.sr + col, dq[nh][0], dq[nh][1]);
    pshort::store_bf16x2(dqb + (long long)r1 * a.ldq.sr + col, dq[nh][2], dq[nh][3]);
  }
}

// Phase drivers, boustrophedon over the CTAs. Forward items run in decreasing query-block index (decreasing causal
// cost), head batch fastest. Backward items take the compiler's per-head-batch list of (kind << 7 | block) entries
// in decreasing estimated cost (kind 0 key/value, 1 query), head batch fastest.
template <bool RES = false>
PLONG_DEV void forward_phase(const Args& a, int blocks, int bhs, float* smem) {
  const int G = gridDim.x, c = blockIdx.x, items = blocks * bhs;
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if constexpr (RES) forward_item_res(a, j % bhs, blocks - 1 - j / bhs, smem);
    else forward_item(a, j % bhs, blocks - 1 - j / bhs, smem);
  }
}
template <bool RES = false>
PLONG_DEV void backward_phase(const Args& a, const unsigned char* list, int entries, int bhs, float* smem) {
  const int G = gridDim.x, c = blockIdx.x, items = entries * bhs;
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    const unsigned e = list[j / bhs];
    if constexpr (RES) {
      if (e >> 7) q_item_res(a, j % bhs, (int)(e & 0x7fu), smem);
      else kv_item_res(a, j % bhs, (int)(e & 0x7fu), smem);
    } else {
      if (e >> 7) q_item(a, j % bhs, (int)(e & 0x7fu), smem);
      else kv_item(a, j % bhs, (int)(e & 0x7fu), smem);
    }
  }
}


// ------------------------------------------------------- P8-B v2 items: pipelined, single-pass, NS-piece products
// Same contract, outputs, mask/guard semantics and resident footprint (T <= RES_MAX) as the resident items above.
// Changes: (1) every streamed 64-row operand tile is its own cp.async group, issued at item start, and each tile is
// consumed as soon as it lands (one CTA barrier per tile) instead of waiting for the whole head; (2) the forward is
// one sweep with online normalization (running row max m, l <- l exp(m_old - m_new), O <- O exp(m_old - m_new),
// e = exp(x - m_new) once per element, O = O / l at the end: FP32-level reassociation, same class as the streaming
// forward); (3) tiles with mask maximum > -1000 are computed without evaluating the norm guard; the guard (same bound
// and margin) is evaluated only for the remaining tiles after the sweep; (4) the probability / score-gradient
// operand of the value, dV, dK and dQ products is split into NS BF16 pieces: NS = 3 is the exact split of the
// FP32 operand (every elementary product exact, as above); NS = 1 rounds that operand once to BF16 (FP32
// accumulation) -- the declared BF16-operand attention arithmetic, selected only by the compiler's precision
// contract for graphs that carry those roundings. The key/value and query backward items with NS = 3 compute
// exactly the products and FP32 operations of kv_item_res / q_item_res, tile by tile.
template <int NS>
PLONG_DEV void tile_product(float (&acc)[8][4], const float (&p)[8][4], const bf16* sv, int lane) {
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) {
    uint32_t a0[NS], a1[NS], a2[NS], a3[NS];
    pshort::split2<NS>(a0, p[2 * k2][0], p[2 * k2][1]);
    pshort::split2<NS>(a1, p[2 * k2][2], p[2 * k2][3]);
    pshort::split2<NS>(a2, p[2 * k2 + 1][0], p[2 * k2 + 1][1]);
    pshort::split2<NS>(a3, p[2 * k2 + 1][2], p[2 * k2 + 1][3]);
#pragma unroll
    for (int nh = 0; nh < 8; ++nh) {
      uint32_t b[2];
      pshort::ldsm_x2_t(b, sv + (k2 * 16 + (lane & 15)) * LD + nh * 8);
#pragma unroll
      for (int s = 0; s < NS; ++s) {
        const uint32_t af[4] = {a0[s], a1[s], a2[s], a3[s]};
        pshort::mma(acc[nh], af, b);
      }
    }
  }
}

// Wait until at most n of this thread's cp.async groups are pending (n clamped to [0, 8]).
PLONG_DEV void cp_wait_dyn(int n) {
  switch (n <= 0 ? 0 : (n >= 8 ? 8 : n)) {
    case 0: asm volatile("cp.async.wait_group 0;\n" ::: "memory"); break;
    case 1: asm volatile("cp.async.wait_group 1;\n" ::: "memory"); break;
    case 2: asm volatile("cp.async.wait_group 2;\n" ::: "memory"); break;
    case 3: asm volatile("cp.async.wait_group 3;\n" ::: "memory"); break;
    case 4: asm volatile("cp.async.wait_group 4;\n" ::: "memory"); break;
    case 5: asm volatile("cp.async.wait_group 5;\n" ::: "memory"); break;
    case 6: asm volatile("cp.async.wait_group 6;\n" ::: "memory"); break;
    case 7: asm volatile("cp.async.wait_group 7;\n" ::: "memory"); break;
    default: asm volatile("cp.async.wait_group 8;\n" ::: "memory"); break;
  }
}

__host__ __device__ constexpr int v2_forward_shared_bytes() { return 2 * (128 + 2 * RES_MAX) * LD + 4 * (128 + RES_MAX / 64 + RES_MAX); }
__host__ __device__ constexpr int v2_backward_shared_bytes() {
  return 2 * (256 + 2 * RES_MAX) * LD + 4 * (3 * RES_MAX + 128 + RES_MAX / 64 + 16);
}

// P8-B mask tile summaries. The additive mask is an input of the phase; with a head-broadcast mask (m_b == 0) its
// tile classifications are identical for every item. Each v2 phase builds them once, cooperatively (every CTA,
// exact maxima and +0 tests over every value, same predicates as tile_class / tile_mask), into a per-phase slot,
// synchronizes the grid once, and items read 8-byte entries instead of re-reading the mask tiles. Kind 0: 16 query
// rows x 64 keys (forward and query items); kind 1: 64 queries x 16 keys (key/value items).
constexpr int MT_SLOTS = 32, MT_ENTRIES = (RES_MAX / 16) * (RES_MAX / 64);
struct MaskTile { float mx; unsigned zero; };
__device__ MaskTile plong_mtab[MT_SLOTS][2][MT_ENTRIES];
__device__ unsigned long long plong_mcnt[MT_SLOTS];

// P8-B row term delta_i = rowsum(dO_i * O32_i) of every (head batch, row), built once per backward phase with the
// products and addition order of row_delta / row_delta_vec (16 sequential terms per quarter, xor-1 then xor-2
// butterfly: identical bits), so items read one FP32 per row instead of recomputing it from dO and O32.
constexpr int DELTA_MAX = 65536;   // head batches x rows per slot (static device scratch, 8 MiB total)
__device__ float plong_delta[MT_SLOTS][DELTA_MAX];

PLONG_DEV void delta_build(const Args& a, int slot, int bhs) {
  const int t = a.t, rows = bhs * t, lane = threadIdx.x & 31, part = lane & 3;
  const int gq = (blockIdx.x * 256 + (int)threadIdx.x) >> 2, nq = (gridDim.x * 256) >> 2;
  for (int r = gq; r - (gq - (gq & 7)) < rows; r += nq) {          // warp-uniform trip count (8 rows per warp)
    float acc = 0.f;
    if (r < rows) {
      const int bh = r / t, i = r % t;
      const bf16* d = a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)i * a.ld.sr + part * 16;
      const float* o = a.o32 + ((long long)bh * t + i) * HD + part * 16;
      const uint4 d0 = __ldcg(reinterpret_cast<const uint4*>(d)), d1 = __ldcg(reinterpret_cast<const uint4*>(d) + 1);
      const uint32_t w[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
        acc = __fadd_rn(acc, __fmul_rn(dk, __ldcg(o + k)));
      }
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    if (part == 0 && r < rows) plong_delta[slot][r] = acc;
  }
}

PLONG_DEV bool delta_ok(int bhs, int t) { return bhs * t <= DELTA_MAX; }

PLONG_DEV bool mtab_ok(const Args& a) { return a.has_mask && a.m_b == 0 && a.t <= RES_MAX && a.t % 64 == 0; }

PLONG_DEV void plong_grid_sync(unsigned long long* cnt) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    const unsigned long long old = atomicAdd(cnt, 1ull);
    const unsigned long long target = (old / gridDim.x + 1ull) * gridDim.x;
    unsigned long long seen;
    do {
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(seen) : "l"(cnt) : "memory");
    } while (seen < target);
  }
  __syncthreads();
}

// Builds kinds selected by the mask (bit 0: kind 0, bit 1: kind 1) into slot (caller synchronizes the grid).
PLONG_DEV void mtab_build(const Args& a, int slot, int kinds) {
  const int t = a.t, n0 = (t / 16) * (t / 64), warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int total = 2 * n0, gw = blockIdx.x * NW + warp, stride = gridDim.x * NW;
  for (int e = gw; e < total; e += stride) {
    const int kind = e >= n0, idx = kind ? e - n0 : e;
    if (!((kinds >> kind) & 1)) continue;
    int r0, c0, cols;
    if (kind == 0) { r0 = (idx / (t / 64)) * 16; c0 = (idx % (t / 64)) * 64; cols = 64; }
    else { r0 = (idx / (t / 16)) * 64; c0 = (idx % (t / 16)) * 16; cols = 16; }
    float m = -INFINITY;
    unsigned nz = 0u;
#pragma unroll 4
    for (int k = 0; k < 32; ++k) {
      const int v = lane + 32 * k, row = v / cols, col = v % cols;
      const float x = mask_at(a, 0, r0 + row, c0 + col);
      m = fmaxf(m, x);
      nz |= __float_as_uint(x);
    }
    m = warp_max(m);
    const bool zero = __all_sync(0xffffffffu, nz == 0u);
    if (lane == 0) plong_mtab[slot][kind][idx] = MaskTile{m, zero ? 1u : 0u};
  }
}

// P8-B per-tile arrival barriers: each thread's cp.async copies of tile g arrive (noinc) on barrier g when they
// complete; a warp consuming tile g waits on that barrier only, never for the other warps' computation.
constexpr int TB_COUNT = RES_MAX / 64;
__shared__ __align__(8) unsigned long long plong_tbar[TB_COUNT];
__shared__ int plong_tbar_live;            // barriers currently initialized (thread 0 bookkeeping)
PLONG_DEV unsigned tbar(int g) { return pshort::smem_u32(&plong_tbar[g]); }
PLONG_DEV void tbar_release() {            // caller: every thread passed its waits, then a CTA barrier
  if (threadIdx.x == 0) {
    for (int g = 0; g < plong_tbar_live; ++g) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(tbar(g)) : "memory");
    plong_tbar_live = 0;
  }
}
PLONG_DEV void tbar_init(int groups) {     // caller: CTA barrier before (previous users done) and after
  if (threadIdx.x == 0) {
    for (int g = 0; g < plong_tbar_live; ++g) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(tbar(g)) : "memory");
    for (int g = 0; g < groups; ++g) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tbar(g)), "r"(256) : "memory");
    plong_tbar_live = groups;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
}
PLONG_DEV void tbar_arrive(int g) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(tbar(g)) : "memory");
}
PLONG_DEV void tbar_wait(int g) {
  asm volatile("{\n.reg .pred P1;\nTB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], 0;\n@P1 bra TB_DONE;\n"
               "bra TB_WAIT;\nTB_DONE:\n}\n" ::"r"(tbar(g)) : "memory");
}

template <int NS>
PLONG_DEV void forward_item2(const Args& a, int bh, int qblk, float* smem, int slot = -1) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + RES_MAX * LD;
  float* qn = reinterpret_cast<float*>(sv + RES_MAX * LD);
  float* kn = qn + 128;                        // [RES_MAX / 64]
  float* scratch = kn + RES_MAX / 64;          // [RES_MAX]
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  tbar_init(nkb);
  __syncthreads();
  load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  pshort::cp_commit();
  for (int kb = 0; kb < nkb; ++kb) {
    load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
    load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
    tbar_arrive(kb);
  }
  float tmx[RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  const bool tab = slot >= 0 && mtab_ok(a);
  for (int kb = 0; kb < nkb; ++kb) {          // classification of this warp's 16 x 64 mask tiles (overlaps copies)
    float mk[8][4];
    bool zero = true, have = false;
    float mt = 0.f;
    if (tab) {
      const MaskTile e = plong_mtab[slot][0][((q0 + warp * 16) / 16) * nkb + kb];
      mt = e.mx;
      zero = e.zero != 0u;
    } else if (a.has_mask) {
      mt = tile_mask2<false>(a, bh, r0, r1, q0 + warp * 16, kb * 64, mk, zero, have);
    }
    tmx[kb] = mt;
    if (zero) zeros |= 1u << kb;
    if (mt > -1000.f) live |= 1u << kb;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // Q (this thread's group; tile copies arrive on their barriers) (tile copies stay uncommitted)
  __syncthreads();
  uint32_t qa[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, oacc[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  unsigned computed = 0u;
  auto tile = [&](int kb) {
    const bool zero = (zeros >> kb) & 1u;
    float mk[8][4];
    if (!zero) tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    float s[8][4];
    tile_scores(s, qa, sk + kb * 64 * LD, lane);
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, s[nt][e]) : pshort::scaled(a, s[nt][e]) + mk[nt][e];
        s[nt][e] = x;
        if (e < 2) tm0 = fmaxf(tm0, x); else tm1 = fmaxf(tm1, x);
      }
    const float n0 = fmaxf(m0, quad_max(tm0)), n1 = fmaxf(m1, quad_max(tm1));
    const float al0 = m0 == -INFINITY ? 0.f : expf(m0 - n0), al1 = m1 == -INFINITY ? 0.f : expf(m1 - n1);
    l0 = __fmul_rn(l0, al0);
    l1 = __fmul_rn(l1, al1);
#pragma unroll
    for (int nh = 0; nh < 8; ++nh) {
      oacc[nh][0] = __fmul_rn(oacc[nh][0], al0); oacc[nh][1] = __fmul_rn(oacc[nh][1], al0);
      oacc[nh][2] = __fmul_rn(oacc[nh][2], al1); oacc[nh][3] = __fmul_rn(oacc[nh][3], al1);
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float ex = expf(s[nt][e] - (e < 2 ? n0 : n1));
        s[nt][e] = ex;
        if (e < 2) l0 = __fadd_rn(l0, ex); else l1 = __fadd_rn(l1, ex);
      }
    tile_product<NS>(oacc, s, sv + kb * 64 * LD, lane);
    m0 = n0;
    m1 = n1;
    computed |= 1u << kb;
  };
  for (int kb = 0; kb < nkb; ++kb) {           // sweep: each warp computes its live tiles as they land
    if ((live >> kb) & 1u) {
      tbar_wait(kb);
      tile(kb);
    }
  }
  for (int kb = 0; kb < nkb; ++kb) tbar_wait(kb);   // every tile landed (guarded remainder may read any)
  const unsigned all = nkb >= 32 ? ~0u : ((1u << nkb) - 1u);
  if (__syncthreads_or((computed & all) != all)) {   // guarded remainder (exact-zero skip or compute)
    row_norms<128>(sq, qn);
    block_norms(sk, t, scratch, kn);
    __syncthreads();
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn[kb]), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn[kb]), NORM_SLACK) + tmx[kb];
      if (__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) continue;
      tile(kb);
    }
  }
  l0 = quad_sum(l0);
  l1 = quad_sum(l1);
  bf16* ob = a.o + pshort::head_offset(a.lo, bh, a.heads);
  float* o32 = a.o32 + (long long)bh * t * HD;
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    const float o00 = __fdiv_rn(oacc[nh][0], l0), o01 = __fdiv_rn(oacc[nh][1], l0);
    const float o10 = __fdiv_rn(oacc[nh][2], l1), o11 = __fdiv_rn(oacc[nh][3], l1);
    pshort::store_bf16x2(ob + (long long)r0 * a.lo.sr + col, o00, o01);
    pshort::store_bf16x2(ob + (long long)r1 * a.lo.sr + col, o10, o11);
    *reinterpret_cast<float2*>(o32 + (long long)r0 * HD + col) = make_float2(o00, o01);
    *reinterpret_cast<float2*>(o32 + (long long)r1 * HD + col) = make_float2(o10, o11);
  }
  if (tig == 0) {
    a.lse[(long long)bh * t + r0] = m0 + logf(l0);
    a.lse[(long long)bh * t + r1] = m1 + logf(l1);
  }
}

template <int NS>
PLONG_DEV void kv_item2(const Args& a, int bh, int kblk, float* smem, int slot = -1, bool dtab = false) {
  bf16* sk = reinterpret_cast<bf16*>(smem);   // [128][LD] keys of this item
  bf16* sv = sk + 128 * LD;
  bf16* sq = sv + 128 * LD;                    // [T][LD] all queries
  bf16* sd = sq + RES_MAX * LD;                // [T][LD] all output gradients
  float* slse = reinterpret_cast<float*>(sd + RES_MAX * LD);
  float* sdel = slse + RES_MAX;
  float* sqn = sdel + RES_MAX;
  float* kn = sqn + RES_MAX;                   // [128]
  const int t = a.t, nqb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = kblk * 128, j0 = k0 + warp * 16 + g, j1 = j0 + 8;
  const float sabs = scale_abs(a);
  const bf16* qbase = a.q + pshort::head_offset(a.lq, bh, a.heads);
  const bf16* dbase = a.dout + pshort::head_offset(a.ld, bh, a.heads);
  tbar_init(nqb);
  __syncthreads();
  load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads) + (long long)k0 * a.lk.sr, a.lk.sr, 128);
  load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads) + (long long)k0 * a.lv.sr, a.lv.sr, 128);
  pshort::cp_commit();
  for (int qb = 0; qb < nqb; ++qb) {
    load_rows_res(sq + qb * 64 * LD, qbase + (long long)qb * 64 * a.lq.sr, a.lq.sr, 64);
    load_rows_res(sd + qb * 64 * LD, dbase + (long long)qb * 64 * a.ld.sr, a.ld.sr, 64);
    tbar_arrive(qb);
  }
  for (int i = (int)threadIdx.x; i < t; i += 256) slse[i] = __ldcg(a.lse + (long long)bh * t + i);
  if (slot >= 0 && dtab) {
    for (int i = (int)threadIdx.x; i < t; i += 256) sdel[i] = __ldcg(&plong_delta[slot][(long long)bh * t + i]);
  } else {
    for (int qb = 0; qb + 1 < nqb; qb += 2) row_delta_vec<2>(a, bh, qb * 64, sdel + qb * 64);
    if (nqb & 1) row_delta_vec<1>(a, bh, (nqb - 1) * 64, sdel + (nqb - 1) * 64);
  }
  float tmx[RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  const bool tab = slot >= 0 && mtab_ok(a);
  for (int qb = 0; qb < nqb; ++qb) {
    bool zero = true;
    float mt = 0.f;
    if (tab) {
      const MaskTile e = plong_mtab[slot][1][qb * (t / 16) + (k0 + warp * 16) / 16];
      mt = e.mx;
      zero = e.zero != 0u;
    } else if (a.has_mask) {
      if (vec_ok(a)) {
        tile_class<64, 16>(a, bh, qb * 64, k0 + warp * 16, mt, zero);
      } else {
        float mk[8][4];
        mt = tile_mask<true>(a, bh, j0, j1, qb * 64, mk, zero);
      }
    }
    tmx[qb] = mt;
    if (zero) zeros |= 1u << qb;
    if (mt > -1000.f) live |= 1u << qb;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // K, V (and the lse / delta stores above, after the barrier) (tile copies stay uncommitted)
  __syncthreads();
  uint32_t ka[KS][4], va[KS][4];
  load_afrag(ka, sk + warp * 16 * LD, lane);
  load_afrag(va, sv + warp * 16 * LD, lane);
  float dv[8][4], dk[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh)
#pragma unroll
    for (int e = 0; e < 4; ++e) dv[nh][e] = dk[nh][e] = 0.f;
  unsigned computed = 0u;
  auto tile = [&](int qb) {
    const int q0 = qb * 64;
    const bool zero = (zeros >> qb) & 1u;
    float s[8][4], dp[8][4];
    tile_scores(s, ka, sq + q0 * LD, lane);
    tile_scores(dp, va, sd + q0 * LD, lane);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int qi = q0 + nt * 8 + 2 * tig + (e & 1);
        const float x = zero ? pshort::scaled(a, s[nt][e])
                             : pshort::scaled(a, s[nt][e]) + mask_at(a, bh, qi, e < 2 ? j0 : j1);
        const float pr = expf(x - slse[qi]);
        s[nt][e] = pr;
        dp[nt][e] = pshort::scaled(a, __fmul_rn(pr, dp[nt][e] - sdel[qi]));
      }
    tile_product<NS>(dv, s, sd + q0 * LD, lane);
    tile_product<NS>(dk, dp, sq + q0 * LD, lane);
    computed |= 1u << qb;
  };
  for (int qb = 0; qb < nqb; ++qb) {
    if ((live >> qb) & 1u) {
      tbar_wait(qb);
      tile(qb);
    }
  }
  for (int qb = 0; qb < nqb; ++qb) tbar_wait(qb);
  const unsigned all = nqb >= 32 ? ~0u : ((1u << nqb) - 1u);
  if (__syncthreads_or((computed & all) != all)) {
    row_norms<128>(sk, kn);
    for (int qb = 0; qb < nqb; ++qb) row_norms<64>(sq + qb * 64 * LD, sqn + qb * 64);
    __syncthreads();
    const float knw = warp_max(fmaxf(kn[warp * 16 + g], kn[warp * 16 + g + 8]));
    for (int qb = 0; qb < nqb; ++qb) {
      if ((computed >> qb) & 1u) continue;
      const int q0 = qb * 64;
      bool safe = true;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qi = q0 + nt * 8 + 2 * tig + e;
          const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, sqn[qi]), knw), NORM_SLACK) + tmx[qb];
          safe = safe && (b < slse[qi] - MARGIN);
        }
      if (__all_sync(0xffffffffu, safe)) continue;
      tile(qb);
    }
  }
  bf16* dkb = a.dk + pshort::head_offset(a.ldk, bh, a.heads);
  bf16* dvb = a.dv + pshort::head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dkb + (long long)j0 * a.ldk.sr + col, dk[nh][0], dk[nh][1]);
    pshort::store_bf16x2(dkb + (long long)j1 * a.ldk.sr + col, dk[nh][2], dk[nh][3]);
    pshort::store_bf16x2(dvb + (long long)j0 * a.ldv.sr + col, dv[nh][0], dv[nh][1]);
    pshort::store_bf16x2(dvb + (long long)j1 * a.ldv.sr + col, dv[nh][2], dv[nh][3]);
  }
}

template <int NS>
PLONG_DEV void q_item2(const Args& a, int bh, int qblk, float* smem, int slot = -1, bool dtab = false) {
  bf16* sq = reinterpret_cast<bf16*>(smem);   // [128][LD]
  bf16* sd = sq + 128 * LD;                    // [128][LD]
  bf16* sk = sd + 128 * LD;                    // [T][LD]
  bf16* sv = sk + RES_MAX * LD;                // [T][LD]
  float* sdel = reinterpret_cast<float*>(sv + RES_MAX * LD);  // [128]
  float* sqn = sdel + 128;                     // [128]
  float* kn = sqn + 128;                       // [RES_MAX / 64]
  float* scratch = kn + RES_MAX / 64;          // [RES_MAX] key row norms
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  tbar_init(nkb);
  __syncthreads();
  load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  load_rows_res(sd, a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)q0 * a.ld.sr, a.ld.sr, 128);
  pshort::cp_commit();
  for (int kb = 0; kb < nkb; ++kb) {
    load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
    load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
    tbar_arrive(kb);
  }
  if (slot >= 0 && dtab) {
    if (threadIdx.x < 128) sdel[threadIdx.x] = __ldcg(&plong_delta[slot][(long long)bh * t + q0 + threadIdx.x]);
  } else {
    row_delta_vec<2>(a, bh, q0, sdel);
  }
  const float lse0 = __ldcg(a.lse + (long long)bh * t + r0), lse1 = __ldcg(a.lse + (long long)bh * t + r1);
  float tmx[RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  const bool tab = slot >= 0 && mtab_ok(a);
  for (int kb = 0; kb < nkb; ++kb) {
    bool zero = true;
    float mt = 0.f;
    if (tab) {
      const MaskTile e = plong_mtab[slot][0][((q0 + warp * 16) / 16) * nkb + kb];
      mt = e.mx;
      zero = e.zero != 0u;
    } else if (a.has_mask) {
      if (vec_ok(a)) {
        tile_class<16, 64>(a, bh, q0 + warp * 16, kb * 64, mt, zero);
      } else {
        float mk[8][4];
        mt = tile_mask<false>(a, bh, r0, r1, kb * 64, mk, zero);
      }
    }
    tmx[kb] = mt;
    if (zero) zeros |= 1u << kb;
    if (mt > -1000.f) live |= 1u << kb;
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");   // Q, dO (tile copies stay uncommitted)
  __syncthreads();
  uint32_t qa[KS][4], da[KS][4];
  load_afrag(qa, sq + warp * 16 * LD, lane);
  load_afrag(da, sd + warp * 16 * LD, lane);
  const float del0 = sdel[warp * 16 + g], del1 = sdel[warp * 16 + g + 8];
  float dq[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) dq[nh][0] = dq[nh][1] = dq[nh][2] = dq[nh][3] = 0.f;
  unsigned computed = 0u;
  auto tile = [&](int kb) {
    const bool zero = (zeros >> kb) & 1u;
    float mk[8][4];
    if (!zero) tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    float s[8][4], dp[8][4];
    tile_scores(s, qa, sk + kb * 64 * LD, lane);
    tile_scores(dp, da, sv + kb * 64 * LD, lane);
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, s[nt][e]) : pshort::scaled(a, s[nt][e]) + mk[nt][e];
        const float pr = expf(x - (e < 2 ? lse0 : lse1));
        dp[nt][e] = pshort::scaled(a, __fmul_rn(pr, dp[nt][e] - (e < 2 ? del0 : del1)));
      }
    tile_product<NS>(dq, dp, sk + kb * 64 * LD, lane);
    computed |= 1u << kb;
  };
  for (int kb = 0; kb < nkb; ++kb) {
    if ((live >> kb) & 1u) {
      tbar_wait(kb);
      tile(kb);
    }
  }
  for (int kb = 0; kb < nkb; ++kb) tbar_wait(kb);
  const unsigned all = nkb >= 32 ? ~0u : ((1u << nkb) - 1u);
  if (__syncthreads_or((computed & all) != all)) {
    row_norms<128>(sq, sqn);
    block_norms(sk, t, scratch, kn);
    __syncthreads();
    const float qn0 = sqn[warp * 16 + g], qn1 = sqn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn[kb]), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn[kb]), NORM_SLACK) + tmx[kb];
      if (__all_sync(0xffffffffu, (b0 < lse0 - MARGIN) && (b1 < lse1 - MARGIN))) continue;
      tile(kb);
    }
  }
  bf16* dqb = a.dq + pshort::head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dqb + (long long)r0 * a.ldq.sr + col, dq[nh][0], dq[nh][1]);
    pshort::store_bf16x2(dqb + (long long)r1 * a.ldq.sr + col, dq[nh][2], dq[nh][3]);
  }
}

template <int NS>
PLONG_DEV void forward_phase2(const Args& a, int blocks, int bhs, float* smem, int slot = -1) {
  const int G = gridDim.x, c = blockIdx.x, items = blocks * bhs;
  if (threadIdx.x == 0) plong_tbar_live = 0;
  if (slot >= 0 && mtab_ok(a)) {
    mtab_build(a, slot, 1);
    plong_grid_sync(&plong_mcnt[slot]);
  }
  __syncthreads();
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    forward_item2<NS>(a, j % bhs, blocks - 1 - j / bhs, smem, slot);
  }
  __syncthreads();
  tbar_release();
}
template <int NS>
PLONG_DEV void backward_phase2(const Args& a, const unsigned char* list, int entries, int bhs, float* smem,
                               int slot = -1) {
  const int G = gridDim.x, c = blockIdx.x, items = entries * bhs;
  if (threadIdx.x == 0) plong_tbar_live = 0;
  if (slot >= 0) {
    if (mtab_ok(a)) mtab_build(a, slot, 3);
    if (delta_ok(bhs, a.t)) delta_build(a, slot, bhs);
    plong_grid_sync(&plong_mcnt[slot]);
  }
  __syncthreads();
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    const unsigned e = list[j / bhs];
    const bool dtab = delta_ok(bhs, a.t);
    if (e >> 7) q_item2<NS>(a, j % bhs, (int)(e & 0x7fu), smem, slot, dtab);
    else kv_item2<NS>(a, j % bhs, (int)(e & 0x7fu), smem, slot, dtab);
  }
  __syncthreads();
  tbar_release();
}

}  // namespace plong
