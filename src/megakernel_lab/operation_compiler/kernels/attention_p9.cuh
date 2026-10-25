// P9-B long-context training attention items (new code in this lane), derived from P8-B's v2 items
// (attention_long.cuh @ 909db2a40: forward_item2 / kv_item2 / q_item2) under the same Args contract.
//
// Arithmetic: per computed tile, exactly the operations and order of the v2 items (tile_scores, scaled + mask,
// online max/rescale with expf, tile_product<NS>, the same final division and lse), so the outputs are bitwise those
// of v2 whenever the same tiles contribute in the same order. Tiles are visited in the same ascending order; a tile
// that v2 skips by its exact-zero guard contributes exact zeros if computed (exp underflows to +0, the running max
// is unchanged, the rescale factor is exp(0) = 1), so guard decisions only change work, never values.
//
// What changes (data movement and work only):
//   * The phase prologue builds, besides v2's mask-tile table, a table of key-block (and query-block) maxima of
//     row norms per head batch (knmax / qnmax), with the row-norm arithmetic of v2's block_norms. The exact-zero
//     guard of a masked tile then needs no copy of the masked keys.
//   * Forward items copy only the key/value tiles that some warp of the CTA computes (mask-table live tiles);
//     a masked tile whose guard fails is copied on demand (slow path).
#pragma once
#include "attention_long.cuh"

namespace p9att {

using plong::Args;
using plong::HD;
using plong::KS;
using plong::LD;
using plong::MARGIN;
using plong::NORM_SLACK;
using bf16 = __nv_bfloat16;
#define P9A_DEV __device__ __forceinline__

constexpr int SLOTS = plong::MT_SLOTS;
constexpr int NMAX = 4096;                          // head batches x (T / 64) entries per slot
__device__ float p9_knmax[SLOTS][NMAX];             // max_j |k_j| over each 64-key block of each head batch
__device__ unsigned p9_kncnt[SLOTS][1024];          // P9-B: per head batch, key-norm entries published (monotonic)
__device__ unsigned p9_fcall[SLOTS][256];           // P9-B: per CTA, forward calls of this slot (same on every CTA)
__device__ float p9_qnmax[SLOTS][NMAX];             // max_i |q_i| over each 64-query block

P9A_DEV bool norms_ok(int bhs, int t) { return bhs * (t / 64) <= NMAX; }

// Row norms (v2 block_norms arithmetic: 4 lanes x 16 squares in order, xor-1 then xor-2, sqrtf) of 64-row blocks
// of a [bh][row] BF16 operand, block maximum by one warp-wide max per block.
P9A_DEV void block_norm_table(const bf16* base, const pshort::Layout4& l, int heads, int bhs, int t, float* out,
                              unsigned* published = nullptr) {
  const int nb = t / 64, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * 8 + warp, nw = gridDim.x * 8;
  for (int e = gw; e < bhs * nb; e += nw) {                        // one warp per (bh, 64-row block)
    const int bh = e / nb, blk = e % nb;
    const bf16* src = base + pshort::head_offset(l, bh, heads) + (long long)(blk * 64) * l.sr;
    uint4 u[8][2];                                                 // all loads first (one memory latency)
#pragma unroll
    for (int r8 = 0; r8 < 8; ++r8) {
      const int row = r8 * 8 + (lane >> 2), part = lane & 3;
      u[r8][0] = __ldg(reinterpret_cast<const uint4*>(src + (long long)row * l.sr + part * 16));
      u[r8][1] = __ldg(reinterpret_cast<const uint4*>(src + (long long)row * l.sr + part * 16) + 1);
    }
    float m = 0.f;
#pragma unroll
    for (int r8 = 0; r8 < 8; ++r8) {                               // 8 rows per pass, 4 lanes per row
      const uint32_t w[8] = {u[r8][0].x, u[r8][0].y, u[r8][0].z, u[r8][0].w, u[r8][1].x, u[r8][1].y, u[r8][1].z, u[r8][1].w};
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 16; ++d) {
        const float x = __uint_as_float((d & 1) ? (w[d >> 1] & 0xffff0000u) : (w[d >> 1] << 16));
        acc = __fadd_rn(acc, __fmul_rn(x, x));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
      m = fmaxf(m, sqrtf(acc));
    }
    m = plong::warp_max(m);
    if (lane == 0) {
      out[e] = m;
      if (published) {                                             // P9-B: release this entry to its head batch
        __threadfence();
        atomicAdd(&published[bh], 1u);
      }
    }
  }
}

// Forward item: v2 forward_item2 arithmetic; copies only CTA-live key/value tiles.
template <int NS>
P9A_DEV void forward_item3(const Args& a, int bh, int qblk, float* smem, int slot) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + plong::RES_MAX * LD;
  float* qn = reinterpret_cast<float*>(sv + plong::RES_MAX * LD);
  __shared__ unsigned p9_live_or;
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = plong::scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  // Classification of this warp's 16 x 64 mask tiles from the phase table (built by the prologue).
  float tmx[plong::RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  for (int kb = 0; kb < nkb; ++kb) {
    const plong::MaskTile e = plong::plong_mtab[slot][0][((q0 + warp * 16) / 16) * nkb + kb];
    tmx[kb] = e.mx;
    if (e.zero) zeros |= 1u << kb;
    if (e.mx > -1000.f) live |= 1u << kb;
  }
  if (threadIdx.x == 0) p9_live_or = 0u;
  plong::tbar_init(nkb);
  __syncthreads();
  if (lane == 0) atomicOr(&p9_live_or, live);
  plong::load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  pshort::cp_commit();
  __syncthreads();
  const unsigned cta_live = p9_live_or;
  for (int kb = 0; kb < nkb; ++kb) {
    if ((cta_live >> kb) & 1u) {
      plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
      plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
      plong::tbar_arrive(kb);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  uint32_t qa[KS][4];
  plong::load_afrag(qa, sq + warp * 16 * LD, lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, oacc[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  unsigned computed = 0u;
  auto tile = [&](int kb) {
    const bool zero = (zeros >> kb) & 1u;
    float mk[8][4];
    if (!zero) plong::tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    float s[8][4];
    plong::tile_scores(s, qa, sk + kb * 64 * LD, lane);
    float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float x = zero ? pshort::scaled(a, s[nt][e]) : pshort::scaled(a, s[nt][e]) + mk[nt][e];
        s[nt][e] = x;
        if (e < 2) tm0 = fmaxf(tm0, x); else tm1 = fmaxf(tm1, x);
      }
    const float n0 = fmaxf(m0, plong::quad_max(tm0)), n1 = fmaxf(m1, plong::quad_max(tm1));
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
    plong::tile_product<NS>(oacc, s, sv + kb * 64 * LD, lane);
    m0 = n0;
    m1 = n1;
    computed |= 1u << kb;
  };
  for (int kb = 0; kb < nkb; ++kb) {
    if ((live >> kb) & 1u) {
      plong::tbar_wait(kb);
      tile(kb);
    }
  }
  // Guarded remainder: v2 bound with the key-block norm maxima from the prologue table and this item's q norms.
  for (int kb = 0; kb < nkb; ++kb) if ((cta_live >> kb) & 1u) plong::tbar_wait(kb);   // every copied tile landed
  const unsigned all = nkb >= 32 ? ~0u : ((1u << nkb) - 1u);
  if (__syncthreads_or((computed & all) != all)) {    // CTA-uniform from here
    plong::row_norms<128>(sq, qn);
    __syncthreads();
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    bool need = false;
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float kn = p9_knmax[slot][bh * nkb + kb];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
      if (!__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) need = true;
    }
    // If every guard holds at the current row maxima, v2 skips every remaining tile too (its maxima only change
    // when it computes a tile). Otherwise replay v2's sequential remainder on fully copied operands.
    if (__syncthreads_or(need)) {
      for (int kb = 0; kb < nkb; ++kb) {
        if ((cta_live >> kb) & 1u) continue;
        plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
        plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
      }
      asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();
      for (int kb = 0; kb < nkb; ++kb) {
        if ((computed >> kb) & 1u) continue;
        const float kn = p9_knmax[slot][bh * nkb + kb];
        const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
        const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
        if (__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) continue;
        tile(kb);
      }
    }
  }
  l0 = plong::quad_sum(l0);
  l1 = plong::quad_sum(l1);
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

// Forward phase: prologue (mask table kind 0 + key-block norm table), one grid synchronization, then items in
// v2's decreasing-cost boustrophedon order.
template <int NS>
P9A_DEV void forward_phase3(const Args& a, int blocks, int bhs, float* smem, int slot) {
  const int G = gridDim.x, c = blockIdx.x, items = blocks * bhs;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  plong::mtab_build(a, slot, 1);
  block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot]);
  plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    forward_item3<NS>(a, j % bhs, blocks - 1 - j / bhs, smem, slot);
  }
  __syncthreads();
  plong::tbar_release();
}


#ifdef P9A_TRACE
#ifndef P9A_TRACE_N
#define P9A_TRACE_N 128
#endif
__device__ unsigned long long p9a_trace[256][P9A_TRACE_N];   // [CTA][event]: globaltimer stamps of warp 0 / warp 7 lane 0
__device__ int p9a_trace_n[256];
P9A_DEV void tstamp(int code) {
  if ((threadIdx.x & 31) == 0 && (threadIdx.x == 0 || threadIdx.x == 224) && blockIdx.x < 256) {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    const int i = atomicAdd(&p9a_trace_n[blockIdx.x], 1);
    if (i < P9A_TRACE_N) p9a_trace[blockIdx.x][i] = (t << 8) | ((unsigned long long)(threadIdx.x >> 5) << 5) | (unsigned)code;
  }
}
#else
P9A_DEV void tstamp(int) {}
#endif
// ------------------------------------------------------------------------------------------------- lean tiles (v4)
// Same per-element operations and per-accumulator mma chains as v2's tile functions; differences are only code
// shape: the score scaling mode and the all-zero-mask case are template parameters (no per-element selects), and
// B fragments are loaded two n-tiles at a time (ldmatrix .x4: the same 8x8 matrices as two .x2 loads).
template <int MODE> P9A_DEV float scl(float d, float s) { return MODE == 1 ? d * s : (MODE == 2 ? d / s : d); }

P9A_DEV void scores_x4(float (&acc)[8][4], const uint32_t (&af)[KS][4], const bf16* sb, int lane) {
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks)
#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      uint32_t b4[4];
      pshort::ldsm_x4(b4, sb + ((2 * n2 + (lane >> 4)) * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
      const uint32_t b0[2] = {b4[0], b4[1]}, b1[2] = {b4[2], b4[3]};
      pshort::mma(acc[2 * n2], af[ks], b0);
      pshort::mma(acc[2 * n2 + 1], af[ks], b1);
    }
}
// acc[nh] += P (bf16 rounding of p, NS = 1 only) times rows of sv ([64][LD], k = row index).
P9A_DEV void product_x4(float (&acc)[8][4], const float (&p)[8][4], const bf16* sv, int lane) {
#pragma unroll
  for (int k2 = 0; k2 < 4; ++k2) {
    uint32_t a0[1], a1[1], a2[1], a3[1];
    pshort::split2<1>(a0, p[2 * k2][0], p[2 * k2][1]);
    pshort::split2<1>(a1, p[2 * k2][2], p[2 * k2][3]);
    pshort::split2<1>(a2, p[2 * k2 + 1][0], p[2 * k2 + 1][1]);
    pshort::split2<1>(a3, p[2 * k2 + 1][2], p[2 * k2 + 1][3]);
    const uint32_t af[4] = {a0[0], a1[0], a2[0], a3[0]};
#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      uint32_t b4[4];
      pshort::ldsm_x4_t(b4, sv + (k2 * 16 + (lane & 15)) * LD + (2 * n2 + (lane >> 4)) * 8);
      const uint32_t b0[2] = {b4[0], b4[1]}, b1[2] = {b4[2], b4[3]};
      pshort::mma(acc[2 * n2], af, b0);
      pshort::mma(acc[2 * n2 + 1], af, b1);
    }
  }
}

// P9-B practical exponentials require an explicit per-call precision declaration. The generated
// shape wrapper gives Args.practical_exp a constant value; strict and practical helpers can coexist.
// Approximate mode uses ex2.approx.ftz(d * log2(e)); subnormal results flush to zero.
P9A_DEV float p9a_exp(float d, bool practical) {
  if (!practical) return expf(d);
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(d, 1.4426950408889634f)));
  return y;
}
#define P9A_EXP(d) p9a_exp((d), a.practical_exp)
// One forward tile, v2 forward_item2::tile arithmetic (NS = 1).
template <int MODE, bool DIAG>
P9A_DEV void fwd_tile(const Args& a, int bh, int r0, int r1, int kb, const uint32_t (&qa)[KS][4], const bf16* sk,
                      const bf16* sv, int lane, float& m0, float& m1, float& l0, float& l1, float (&oacc)[8][4],
                      const float* staged = nullptr) {
  float mk[8][4];
  if (DIAG) {
    if (staged) {                                  // this warp's [16][64] mask rows of the tile (same values)
      const int g = lane >> 2, tig = lane & 3;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const float2 u = *reinterpret_cast<const float2*>(staged + g * 64 + nt * 8 + 2 * tig);
        const float2 w = *reinterpret_cast<const float2*>(staged + (g + 8) * 64 + nt * 8 + 2 * tig);
        mk[nt][0] = u.x; mk[nt][1] = u.y; mk[nt][2] = w.x; mk[nt][3] = w.y;
      }
    } else {
      plong::tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    }
  }
  float s[8][4];
  scores_x4(s, qa, sk + kb * 64 * LD, lane);
  float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x = DIAG ? scl<MODE>(s[nt][e], a.scale) + mk[nt][e] : scl<MODE>(s[nt][e], a.scale);
      s[nt][e] = x;
      if (e < 2) tm0 = fmaxf(tm0, x); else tm1 = fmaxf(tm1, x);
    }
  const float n0 = fmaxf(m0, plong::quad_max(tm0)), n1 = fmaxf(m1, plong::quad_max(tm1));
  const float al0 = m0 == -INFINITY ? 0.f : P9A_EXP(m0 - n0), al1 = m1 == -INFINITY ? 0.f : P9A_EXP(m1 - n1);
#ifdef P9A_SKIP_RESCALE
  // P9-B: when no row maximum of this warp changed (n == m), every rescale factor is exp(0) = 1 and multiplying by it
  // is exact: skip the rescale of l and the 32 output accumulators (warp-uniform branch, same values).
  if (__any_sync(0xffffffffu, (n0 != m0) | (n1 != m1)))
#endif
  {
  l0 = __fmul_rn(l0, al0);
  l1 = __fmul_rn(l1, al1);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    oacc[nh][0] = __fmul_rn(oacc[nh][0], al0); oacc[nh][1] = __fmul_rn(oacc[nh][1], al0);
    oacc[nh][2] = __fmul_rn(oacc[nh][2], al1); oacc[nh][3] = __fmul_rn(oacc[nh][3], al1);
  }
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float ex = P9A_EXP(s[nt][e] - (e < 2 ? n0 : n1));
      s[nt][e] = ex;
      if (e < 2) l0 = __fadd_rn(l0, ex); else l1 = __fadd_rn(l1, ex);
    }
  product_x4(oacc, s, sv + kb * 64 * LD, lane);
  m0 = n0;
  m1 = n1;
}

// Forward item v4: forward_item3 data movement with the lean tiles (NS = 1, scale mode MODE); item tables, q norms
// and the first non-zero live mask tile of each warp are staged in shared memory at item start.
// P9-B: the item tables live in the dynamic shared memory after the item's operands (static __shared__ arrays of every
// phase function add up in the program's single kernel and pushed dynamic + static shared memory past the 196 KiB
// carve-out, halving L1 for every other phase). The diagonal mask tile staging (32 KiB) is off by default for the same
// reason (P9A_STAGE_MASK).
#ifndef P9A_STAGE_MASK
#define P9A_STAGE_MASK 0
#endif
template <class T> P9A_DEV T* tail_at(float* smem, int bytes) { return reinterpret_cast<T*>(reinterpret_cast<char*>(smem) + bytes); }
constexpr int FWD_TAIL = 2 * (128 + 2 * plong::RES_MAX) * LD + 4 * 128 + (P9A_STAGE_MASK ? 4 * 8 * 16 * 64 : 0);
__host__ __device__ constexpr int v4_forward_shared_bytes() {
  return FWD_TAIL + 8 * 8 * (plong::RES_MAX / 64) + 4 * (plong::RES_MAX / 64) + 16 + 16;   // + call epoch (last 16)
}
// `have` (P9-B pair items): key/value tiles already resident and complete from the previous item of the same head
// batch; they are neither copied nor re-armed. Returns the tiles resident after this item's copies.
template <int MODE>
P9A_DEV unsigned forward_item4(const Args& a, int bh, int qblk, float* smem, int slot, int mslot, unsigned kn_target = 0u,
                               bool reuse = false, unsigned have = 0u) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + plong::RES_MAX * LD;
  float* qn = reinterpret_cast<float*>(sv + plong::RES_MAX * LD);
  float* smask = qn + 128;                                         // [8 warps][16][64] (P9A_STAGE_MASK)
  plong::MaskTile* p9_mt4 = tail_at<plong::MaskTile>(smem, FWD_TAIL);
  float* p9_kn4 = tail_at<float>(smem, FWD_TAIL + 8 * 8 * (plong::RES_MAX / 64));
  unsigned& p9_live_or4 = *tail_at<unsigned>(smem, FWD_TAIL + 8 * 8 * (plong::RES_MAX / 64) + 4 * (plong::RES_MAX / 64));
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = plong::scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  tstamp(1);
  if (threadIdx.x == 0) p9_live_or4 = 0u;
  if ((int)threadIdx.x < 8 * nkb) p9_mt4[threadIdx.x] = plong::plong_mtab[mslot][0][(q0 / 16) * nkb + threadIdx.x];
  else if (kn_target == 0u && (int)threadIdx.x < 9 * nkb) p9_kn4[threadIdx.x - 8 * nkb] = p9_knmax[slot][bh * nkb + threadIdx.x - 8 * nkb];
  if (!reuse) plong::tbar_init(nkb);
  __syncthreads();
  tstamp(21);
  float tmx[plong::RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  for (int kb = 0; kb < nkb; ++kb) {
    const plong::MaskTile e = p9_mt4[warp * nkb + kb];
    tmx[kb] = e.mx;
    if (e.zero) zeros |= 1u << kb;
    if (e.mx > -1000.f) live |= 1u << kb;
  }
  if (lane == 0) atomicOr(&p9_live_or4, live);
  // Q, then this warp's first live non-zero mask tile (16 rows x 64 FP32), one committed group.
  plong::load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  const unsigned nz = live & ~zeros;
  const int stage_kb = nz ? __ffs(nz) - 1 : -1;
  const bool can_stage = P9A_STAGE_MASK && a.has_mask && a.m_j == 1 && (a.m_i % 4) == 0 && (a.m_b % 4) == 0 &&
                         ((reinterpret_cast<uintptr_t>(a.mask) & 15) == 0);
  if (stage_kb >= 0 && can_stage) {
    const float* src = a.mask + (long long)bh * a.m_b + (long long)(q0 + warp * 16) * a.m_i + stage_kb * 64;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int idx = lane + 32 * c, row = idx >> 4, col = (idx & 15) * 4;
      pshort::cp16(smask + warp * 1024 + row * 64 + col, src + (long long)row * a.m_i + col);
    }
  }
  pshort::cp_commit();
  __syncthreads();
  tstamp(22);
  const unsigned cta_live = p9_live_or4;
  for (int kb = 0; kb < nkb; ++kb) {
    if (((cta_live & ~have) >> kb) & 1u) {
      plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
      plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
      plong::tbar_arrive(kb);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  tstamp(2);
  plong::row_norms<128>(sq, qn);                                   // read only after a later CTA barrier
  uint32_t qa[KS][4];
  plong::load_afrag(qa, sq + warp * 16 * LD, lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, oacc[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  unsigned computed = 0u;
  const float* my_mask = smask + warp * 1024;
  auto tile = [&](int kb) {
    tstamp(3);
    if ((zeros >> kb) & 1u) fwd_tile<MODE, false>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc);
    else fwd_tile<MODE, true>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc,
                              (kb == stage_kb && can_stage) ? my_mask : nullptr);
    computed |= 1u << kb;
  };
#pragma unroll 1
  for (int kb = 0; kb < nkb; ++kb) {
    if ((live >> kb) & 1u) {
      plong::tbar_wait(kb);
      tile(kb);
    }
  }
  tstamp(4);
  for (int kb = 0; kb < nkb; ++kb) if ((cta_live >> kb) & 1u) plong::tbar_wait(kb);
  if (kn_target != 0u) {
    // P9-B: the key-norm entries of this head batch were published by this call's prologue warps (counter release);
    // read them only now, after the item's tiles (no grid synchronization before the items).
    if (threadIdx.x == 0) {
      unsigned seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(&p9_kncnt[slot][bh]) : "memory");
      } while ((int)(seen - kn_target) < 0);
    }
    __syncthreads();
    if ((int)threadIdx.x < nkb) p9_kn4[threadIdx.x] = __ldcg(&p9_knmax[slot][bh * nkb + threadIdx.x]);
    __syncthreads();
  }
  // Remainder guard (v2 bound; key-block norm maxima from the prologue table, q norms computed above).
  bool need = false;
  {
    __syncwarp();                                                  // row_norms<128>: this warp's 32 threads computed its 16 rows
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float kn = p9_kn4[kb];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
      if (!__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) need = true;
    }
  }
  // If every guard holds at the current row maxima, v2 skips every remaining tile too (its maxima only change when
  // it computes a tile). Otherwise replay v2's sequential remainder on fully copied operands.
  if (__syncthreads_or(need)) {
    for (int kb = 0; kb < nkb; ++kb) {
      if (((cta_live | have) >> kb) & 1u) continue;
      plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
      plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float kn = p9_kn4[kb];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
      if (__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) continue;
      if ((zeros >> kb) & 1u) fwd_tile<MODE, false>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc);
      else fwd_tile<MODE, true>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc);
    }
  }
  l0 = plong::quad_sum(l0);
  l1 = plong::quad_sum(l1);
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
  tstamp(5);
  return cta_live | have;
}

#ifndef P9A_KV_QUEUE
#define P9A_KV_QUEUE 0
#endif
#ifndef P9A_FWD_QUEUE
#define P9A_FWD_QUEUE 0
#endif
#if P9A_KV_QUEUE || P9A_FWD_QUEUE
// P9-B: item work queues. Every CTA runs every call of a slot and makes exactly one failing claim per call, so the
// claims of call e start at (e - 1) * (items + gridDim.x); the call count is kept per CTA.
__device__ unsigned p9_wq[2][SLOTS];
__device__ unsigned p9_wqcalls[2][SLOTS][256];
__shared__ unsigned p9_wqcall;
__shared__ int p9_wqitem;
P9A_DEV void wq_begin(int q, int slot) {
  if (threadIdx.x == 0) p9_wqcall = ++p9_wqcalls[q][slot][blockIdx.x];
}
P9A_DEV int wq_claim(int q, int slot, int items) {
  __syncthreads();
  if (threadIdx.x == 0)
    p9_wqitem = (int)(atomicAdd(&p9_wq[q][slot], 1u) - (p9_wqcall - 1u) * (unsigned)(items + (int)gridDim.x));
  __syncthreads();
  return p9_wqitem;
}
#endif
// PRE: 0 = prologue builds mask table + key norms + grid sync; 1 = key norms only (mask table prebuilt for the
// whole update) + grid sync.
// Phase entries are out of line (P9A_OUTLINE, default on): inlined into the program's single kernel function the
// item bodies spill (P9-B c-01: +2387 spill operations); mma.sync/cp.async callees do not serialize wgmma elsewhere.
#ifndef P9A_OUTLINE
#define P9A_OUTLINE 0       // the compiler's per-shape wrappers (long_attention.P9_WRAPPER) are the out-of-line level
#endif
#if P9A_OUTLINE
#define P9A_PHASE __device__ __noinline__
#else
#define P9A_PHASE __device__ __forceinline__
#endif
template <int PRE>
P9A_PHASE void forward_phase4(const Args& a, int blocks, int bhs, float* smem, int slot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, items = blocks * bhs;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  tstamp(6);
#ifndef P9A_NO_KNORM
#define P9A_NO_KNORM 0     // diagnostic only: PRE 1 phases skip the key-norm table and its grid sync (timing study)
#endif
  unsigned kn_target = 0u;
#ifndef P9A_KN_FLAGS
#define P9A_KN_FLAGS 1
#endif
  if (PRE == 1 && P9A_KN_FLAGS && bhs <= 1024 && G <= 256) {
    // P9-B: no grid synchronization: each CTA counts this slot's calls (every CTA runs every call), the key-norm
    // warps release a per-head-batch counter, and items wait for it only before their remainder guard.
    unsigned* ep = tail_at<unsigned>(smem, v4_forward_shared_bytes() - 16);
    if (threadIdx.x == 0) *ep = ++p9_fcall[slot][c];
    __syncthreads();
    kn_target = *ep * (unsigned)(a.t / 64);
    block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot], p9_kncnt[slot]);
    __syncthreads();
  } else if (PRE == 0 || !P9A_NO_KNORM) {
    if (PRE == 0) plong::mtab_build(a, mslot, 3);   // both kinds: later phases read this slot
    block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot]);
    tstamp(7);
    plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  }
  tstamp(8);
#if P9A_FWD_QUEUE
  wq_begin(1, slot);
#endif
  for (int r = 0;; ++r) {
#if P9A_FWD_QUEUE
    const int j = wq_claim(1, slot, items);
    if (j >= items) break;
#else
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
#endif
    if (a.scale_mode == 1) forward_item4<1>(a, j % bhs, blocks - 1 - j / bhs, smem, slot, mslot, kn_target);
    else if (a.scale_mode == 2) forward_item4<2>(a, j % bhs, blocks - 1 - j / bhs, smem, slot, mslot, kn_target);
    else forward_item4<0>(a, j % bhs, blocks - 1 - j / bhs, smem, slot, mslot, kn_target);
  }
  __syncthreads();
  plong::tbar_release();
}


// P9-B forward v7: pair items. A head batch's 128-query blocks are paired as (blocks - 1 - p, p), so every pair has
// the same number of causal key tiles and a CTA runs one item (one setup, one tail) instead of up to two. The second
// block of a pair reuses the key/value tiles the first one copied (its live tiles are a subset under a causal mask;
// any other live tile is copied). Per-row arithmetic and tile order are those of forward_item4: outputs are bitwise.
template <int PRE>
P9A_PHASE void forward_phase7(const Args& a, int blocks, int bhs, float* smem, int slot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, npair = blocks / 2, kinds = npair + (blocks & 1), items = kinds * bhs;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  unsigned kn_target = 0u;
  if (PRE == 1 && P9A_KN_FLAGS && bhs <= 1024 && G <= 256) {
    unsigned* ep = tail_at<unsigned>(smem, v4_forward_shared_bytes() - 16);
    if (threadIdx.x == 0) *ep = ++p9_fcall[slot][c];
    __syncthreads();
    kn_target = *ep * (unsigned)(a.t / 64);
    block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot], p9_kncnt[slot]);
    __syncthreads();
  } else {
    if (PRE == 0) plong::mtab_build(a, mslot, 3);
    block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot]);
    plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  }
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    const int bh = j % bhs, p = j / bhs;
    const int first = p < npair ? blocks - 1 - p : npair;
    unsigned have = 0u;
#define P9A_F7(M)                                                                                                   \
    have = forward_item4<M>(a, bh, first, smem, slot, mslot, kn_target);                                             \
    if (p < npair) { __syncthreads(); forward_item4<M>(a, bh, p, smem, slot, mslot, kn_target, true, have); }
    if (a.scale_mode == 1) { P9A_F7(1) }
    else if (a.scale_mode == 2) { P9A_F7(2) }
    else { P9A_F7(0) }
#undef P9A_F7
  }
  __syncthreads();
  plong::tbar_release();
}

// ------------------------------------------------------------------------------------------------ backward (v4)
// Key/value items (bh, 128-key block): v2 kv_item2 arithmetic (NS = 1) on the CTA-live query tiles only; besides dK
// and dV they publish the BF16 score gradient dS^T[key][query] (the cvt.rn.bf16x2 values the query side rounds for
// its dQ product) for every CTA-live 64-query block, zeros for warps whose 16 keys are not live there, and a
// published flag per (bh, 64-query block, 128-key block). dQ items then accumulate dQ = dS K over published blocks in
// ascending key order with v2 q_item2's product chain. A published block that v2's query item would not compute holds
// exact zeros (exp underflow), adding nothing.
constexpr int DS_SLOTS = 8;
// Backward item tables (dynamic shared memory after kv_item4's operands; dq_item4 uses less): see FWD_TAIL.
constexpr int BWD_TAIL = 2 * (256 + 2 * plong::RES_MAX) * LD + 4 * 3 * plong::RES_MAX;
#ifndef P9A_DS_ELEMS
#define P9A_DS_ELEMS (64LL * 512LL * 512LL)
#endif
__device__ bf16 p9_dsbuf[P9A_DS_ELEMS];                            // program-wide dS^T scratch (phases are sequential)
__device__ unsigned char p9_pub[DS_SLOTS][NMAX * 4];               // [bh][query tile 64][key block 128]
__device__ unsigned p9_pub32[DS_SLOTS][NMAX * 4];   // LATCH-STEP-06 (P9_BWD_TAGS): call-epoch publication tags

// delta_i = rowsum(dO_i * O32_i): plong::delta_build arithmetic (per row quarter 16 sequential products/sums in d
// order, xor-1 then xor-2 butterfly), all of a thread's loads issued before any reduction.
// P9-B (P9A_DELTA_PF): L2 prefetch of the rows delta_table will read (this thread's quad rows of dO and O32), issued at
// phase entry so their DRAM latency overlaps the q-norm table. No semantic effect.
P9A_DEV void delta_prefetch(const Args& a, int bhs) {
  const int t = a.t, rows = bhs * t, part = threadIdx.x & 3;
  const int gq = (blockIdx.x * 256 + (int)threadIdx.x) >> 2, nq = (gridDim.x * 256) >> 2;
  for (int row = gq; row < rows; row += nq) {
    const int bh = row / t, i = row % t;
    const bf16* d = a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)i * a.ld.sr + part * 16;
    const float* o = a.o32 + ((long long)bh * t + i) * HD + part * 16;
    asm volatile("prefetch.global.L2 [%0];" :: "l"(o));
    if (part == 0) asm volatile("prefetch.global.L2 [%0];" :: "l"(d));
  }
}
P9A_DEV void delta_table(const Args& a, int slot, int bhs) {
  const int t = a.t, rows = bhs * t, lane = threadIdx.x & 31, part = lane & 3;
  const int gq = (blockIdx.x * 256 + (int)threadIdx.x) >> 2, nq = (gridDim.x * 256) >> 2;
  constexpr int R = 4;                                             // rows per quad per pass
  const int wq = gq - (gq & 7);                                    // first quad of this warp: uniform trip count
  for (int base = gq; base - (gq - wq) < rows; base += R * nq) {
    uint4 dv[R][2];
    float4 ov[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = base + r * nq;
      if (row < rows) {
        const int bh = row / t, i = row % t;
        const bf16* d = a.dout + pshort::head_offset(a.ld, bh, a.heads) + (long long)i * a.ld.sr + part * 16;
        const float* o = a.o32 + ((long long)bh * t + i) * HD + part * 16;
        dv[r][0] = __ldcg(reinterpret_cast<const uint4*>(d));
        dv[r][1] = __ldcg(reinterpret_cast<const uint4*>(d) + 1);
#pragma unroll
        for (int k = 0; k < 4; ++k) ov[r][k] = __ldcg(reinterpret_cast<const float4*>(o) + k);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int row = base + r * nq;
      const uint32_t w[8] = {dv[r][0].x, dv[r][0].y, dv[r][0].z, dv[r][0].w, dv[r][1].x, dv[r][1].y, dv[r][1].z, dv[r][1].w};
      const float of[16] = {ov[r][0].x, ov[r][0].y, ov[r][0].z, ov[r][0].w, ov[r][1].x, ov[r][1].y, ov[r][1].z, ov[r][1].w,
                            ov[r][2].x, ov[r][2].y, ov[r][2].z, ov[r][2].w, ov[r][3].x, ov[r][3].y, ov[r][3].z, ov[r][3].w};
      float acc = 0.f;
#pragma unroll
      for (int k = 0; k < 16; ++k) {
        const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
        acc = __fadd_rn(acc, __fmul_rn(dk, of[k]));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
      if (part == 0 && row < rows) plong::plong_delta[slot][row] = acc;
    }
  }
}

template <int MODE, bool DIAG>
P9A_DEV void kv_tile(const Args& a, int bh, int j0, int j1, int q0, const uint32_t (&ka)[KS][4],
                     const uint32_t (&va)[KS][4], const bf16* sq, const bf16* sd, const float* slse,
                     const float* sdel, int lane, float (&dv)[8][4], float (&dk)[8][4], bf16* ds_rows) {
  const int tig = lane & 3;
  float mk[8][4];
  if (DIAG) plong::tile_frag<true>(a, bh, j0, j1, q0, mk);
  float s[8][4], dp[8][4];
  scores_x4(s, ka, sq + q0 * LD, lane);
  scores_x4(dp, va, sd + q0 * LD, lane);
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int qi = q0 + nt * 8 + 2 * tig + (e & 1);
      const float x = DIAG ? scl<MODE>(s[nt][e], a.scale) + mk[nt][e] : scl<MODE>(s[nt][e], a.scale);
      const float pr = P9A_EXP(x - slse[qi]);
      s[nt][e] = pr;
      dp[nt][e] = scl<MODE>(__fmul_rn(pr, dp[nt][e] - sdel[qi]), a.scale);
    }
  if (ds_rows) {                                    // rows j0 / j1 of dS^T, row stride a.t
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int col = q0 + nt * 8 + 2 * tig;
      pshort::store_bf16x2(ds_rows + col, dp[nt][0], dp[nt][1]);
      pshort::store_bf16x2(ds_rows + 8LL * a.t + col, dp[nt][2], dp[nt][3]);
    }
  }
  product_x4(dv, s, sd + q0 * LD, lane);
  product_x4(dk, dp, sq + q0 * LD, lane);
}

template <int MODE>
P9A_DEV void kv_item4(const Args& a, int bh, int kblk, float* smem, int slot, bf16* dsT, int dslot, int mslot,
                      unsigned tag32 = 0u) {
  bf16* sk = reinterpret_cast<bf16*>(smem);   // [128][LD]
  bf16* sv = sk + 128 * LD;
  bf16* sq = sv + 128 * LD;                    // [T][LD] (CTA-live tiles only)
  bf16* sd = sq + plong::RES_MAX * LD;
  float* slse = reinterpret_cast<float*>(sd + plong::RES_MAX * LD);
  float* sdel = slse + plong::RES_MAX;
  plong::MaskTile* p9_kvmt = tail_at<plong::MaskTile>(smem, BWD_TAIL);
  float* p9_kvqn = tail_at<float>(smem, BWD_TAIL + 512);
  unsigned& p9_kv_live = *tail_at<unsigned>(smem, BWD_TAIL + 1056);
  const int t = a.t, nqb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = kblk * 128, j0 = k0 + warp * 16 + g, j1 = j0 + 8;
  const float sabs = plong::scale_abs(a);
  const bf16* qbase = a.q + pshort::head_offset(a.lq, bh, a.heads);
  const bf16* dbase = a.dout + pshort::head_offset(a.ld, bh, a.heads);
  tstamp(13);
  if (threadIdx.x == 0) p9_kv_live = 0u;
  {                                             // kind-1 entries: [qb][warp] = table[qb * (t / 16) + (k0 / 16) + warp]
    const int i = threadIdx.x;
    if (i < 8 * nqb) p9_kvmt[i] = plong::plong_mtab[mslot][1][(i / 8) * (t / 16) + k0 / 16 + (i % 8)];
    else if (i < 9 * nqb) p9_kvqn[i - 8 * nqb] = p9_qnmax[slot][bh * nqb + i - 8 * nqb];
  }
  plong::tbar_init(nqb);
  __syncthreads();
  tstamp(19);
  float tmx[plong::RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  for (int qb = 0; qb < nqb; ++qb) {
    const plong::MaskTile e = p9_kvmt[qb * 8 + warp];
    tmx[qb] = e.mx;
    if (e.zero) zeros |= 1u << qb;
    if (e.mx > -1000.f) live |= 1u << qb;
  }
  if (lane == 0) atomicOr(&p9_kv_live, live);
  plong::load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads) + (long long)k0 * a.lk.sr, a.lk.sr, 128);
  plong::load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads) + (long long)k0 * a.lv.sr, a.lv.sr, 128);
  pshort::cp_commit();
  for (int i = (int)threadIdx.x; i < t; i += 256) {
    slse[i] = __ldcg(a.lse + (long long)bh * t + i);
    sdel[i] = __ldcg(&plong::plong_delta[slot][(long long)bh * t + i]);
  }
  __syncthreads();
  tstamp(20);
  const unsigned cta_live = p9_kv_live;
  for (int qb = 0; qb < nqb; ++qb) {
    if ((cta_live >> qb) & 1u) {
      plong::load_rows_res(sq + qb * 64 * LD, qbase + (long long)qb * 64 * a.lq.sr, a.lq.sr, 64);
      plong::load_rows_res(sd + qb * 64 * LD, dbase + (long long)qb * 64 * a.ld.sr, a.ld.sr, 64);
      plong::tbar_arrive(qb);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  tstamp(14);
  uint32_t ka[KS][4], va[KS][4];
  plong::load_afrag(ka, sk + warp * 16 * LD, lane);
  plong::load_afrag(va, sv + warp * 16 * LD, lane);
  float dv[8][4], dk[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh)
#pragma unroll
    for (int e = 0; e < 4; ++e) dv[nh][e] = dk[nh][e] = 0.f;
  unsigned computed = 0u;
  bf16* myrows = dsT + ((long long)bh * t + j0) * t;                // dS^T row j0 (row j1 = j0 + 8)
  auto tile = [&](int qb) {
    if ((zeros >> qb) & 1u) kv_tile<MODE, false>(a, bh, j0, j1, qb * 64, ka, va, sq, sd, slse, sdel, lane, dv, dk, myrows);
    else kv_tile<MODE, true>(a, bh, j0, j1, qb * 64, ka, va, sq, sd, slse, sdel, lane, dv, dk, myrows);
    computed |= 1u << qb;
  };
#pragma unroll 1
  for (int qb = 0; qb < nqb; ++qb) {
    if ((live >> qb) & 1u) {
      plong::tbar_wait(qb);
      tile(qb);
    } else if ((cta_live >> qb) & 1u) {          // published block: this warp's rows are exact zeros
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = qb * 64 + nt * 8 + 2 * tig;
        pshort::store_bf16x2(myrows + col, 0.f, 0.f);
        pshort::store_bf16x2(myrows + 8LL * t + col, 0.f, 0.f);
      }
    }
  }
  for (int qb = 0; qb < nqb; ++qb) if ((cta_live >> qb) & 1u) plong::tbar_wait(qb);
  // Remainder guard: per-row v2 bound with block maxima of query norms (conservative: never skips a tile v2 computes).
  bool need = false;
  {
    float knv = 0.f;
    {
      // this warp's 16 key norms (v2 row_norms<128> arithmetic on sk rows, 2 threads per row: warp w holds rows 16w..)
      const int row = threadIdx.x >> 1, part = threadIdx.x & 1;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        const float x = __bfloat162float(sk[row * LD + part * 32 + d]);
        acc = __fadd_rn(acc, __fmul_rn(x, x));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      knv = plong::warp_max(sqrtf(acc));
    }
    for (int qb = 0; qb < nqb; ++qb) {
      if ((computed >> qb) & 1u) continue;
      const float qn = p9_kvqn[qb];
      bool safe = true;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qi = qb * 64 + nt * 8 + 2 * tig + e;
          const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn), knv), NORM_SLACK) + tmx[qb];
          safe = safe && (b < slse[qi] - MARGIN);
        }
      if (!__all_sync(0xffffffffu, safe)) need = true;
    }
  }
  unsigned extra = 0u;                          // query tiles computed by the slow path (published as well)
  if (__syncthreads_or(need)) {                 // slow path: v2 remainder on fully copied operands
    float* sqn = sdel + plong::RES_MAX;         // [T] (inside the shared budget: sk/sv/sq/sd/lse/del + T floats)
    for (int qb = 0; qb < nqb; ++qb) {
      if ((cta_live >> qb) & 1u) continue;
      plong::load_rows_res(sq + qb * 64 * LD, qbase + (long long)qb * 64 * a.lq.sr, a.lq.sr, 64);
      plong::load_rows_res(sd + qb * 64 * LD, dbase + (long long)qb * 64 * a.ld.sr, a.ld.sr, 64);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    float* kn = reinterpret_cast<float*>(smem) ;  // reuse? no: key norms via row_norms into scratch below
    (void)kn;
    float* p9_kvkn = tail_at<float>(smem, BWD_TAIL + 544);
    plong::row_norms<128>(sk, p9_kvkn);
    for (int qb = 0; qb < nqb; ++qb) plong::row_norms<64>(sq + qb * 64 * LD, sqn + qb * 64);
    __syncthreads();
    const float knw = plong::warp_max(fmaxf(p9_kvkn[warp * 16 + g], p9_kvkn[warp * 16 + g + 8]));
    for (int qb = 0; qb < nqb; ++qb) {
      if ((computed >> qb) & 1u) continue;
      bool safe = true;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qi = qb * 64 + nt * 8 + 2 * tig + e;
          const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, sqn[qi]), knw), NORM_SLACK) + tmx[qb];
          safe = safe && (b < slse[qi] - MARGIN);
        }
      if (__all_sync(0xffffffffu, safe)) continue;
      tile(qb);
      extra |= 1u << qb;
    }
    // Rows of slow-path tiles not computed by some warps: zeros (the tile's other warps published values).
    const unsigned any_extra = __reduce_or_sync(0xffffffffu, extra);
    unsigned& p9_kv_extra = *tail_at<unsigned>(smem, BWD_TAIL + 1060);
    if (threadIdx.x == 0) p9_kv_extra = 0u;
    __syncthreads();
    if (lane == 0) atomicOr(&p9_kv_extra, any_extra);
    __syncthreads();
    const unsigned all_extra = p9_kv_extra;
    for (int qb = 0; qb < nqb; ++qb) {
      if (!((all_extra >> qb) & 1u) || ((computed >> qb) & 1u)) continue;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = qb * 64 + nt * 8 + 2 * tig;
        pshort::store_bf16x2(myrows + col, 0.f, 0.f);
        pshort::store_bf16x2(myrows + 8LL * t + col, 0.f, 0.f);
      }
    }
    if (threadIdx.x < nqb && ((all_extra >> threadIdx.x) & 1u)) {
      if (tag32) p9_pub32[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = tag32;
      else p9_pub[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = 1;
    }
  }
  if ((int)threadIdx.x < nqb) {
    if ((cta_live >> threadIdx.x) & 1u) {
      if (tag32) p9_pub32[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = tag32;
      else p9_pub[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = 1;
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
  tstamp(15);
}

// ------------------------------------------------------------------------------------------------ backward (v8)
// P9-B v8 backward: no phase prologue and no in-phase grid synchronization. Key/value items compute the delta rows of
// their staged query blocks (delta_table arithmetic on the staged dO and the FP32 output) and the head batch's
// query-block norm maxima (block_norm_table arithmetic); published dS^T flags carry a per-call tag and dQ items wait
// for an epoch completion target (every CTA runs every call of a slot, so per-CTA call counts agree).
// delta_i for the 64-row query blocks in `blocks` (rows from the staged dO `sd`, [T][LD]); all 256 threads.
P9A_DEV void delta_blocks8(const Args& a, int bh, unsigned blocks, const bf16* sd, float* sdel) {
  const int t = a.t, row = threadIdx.x >> 2, part = threadIdx.x & 3;
  for (int qb = 0; qb < t / 64; ++qb) {
    if (!((blocks >> qb) & 1u)) continue;                           // CTA-uniform
    const int i = qb * 64 + row;
    const float* o = a.o32 + ((long long)bh * t + i) * HD + part * 16;
    float4 ov[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) ov[k] = __ldcg(reinterpret_cast<const float4*>(o) + k);
    const uint4 d0 = *reinterpret_cast<const uint4*>(sd + i * LD + part * 16);
    const uint4 d1 = *reinterpret_cast<const uint4*>(sd + i * LD + part * 16 + 8);
    const uint32_t w[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    const float of[16] = {ov[0].x, ov[0].y, ov[0].z, ov[0].w, ov[1].x, ov[1].y, ov[1].z, ov[1].w,
                          ov[2].x, ov[2].y, ov[2].z, ov[2].w, ov[3].x, ov[3].y, ov[3].z, ov[3].w};
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
      acc = __fadd_rn(acc, __fmul_rn(dk, of[k]));
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    if (part == 0) sdel[i] = acc;
  }
}
// One warp: max row norm of query block qb of head batch bh (block_norm_table arithmetic).
P9A_DEV float block_norm8(const bf16* base, const pshort::Layout4& l, int heads, int bh, int qb) {
  const int lane = threadIdx.x & 31;
  const bf16* src = base + pshort::head_offset(l, bh, heads) + (long long)(qb * 64) * l.sr;
  uint4 u[8][2];
#pragma unroll
  for (int r8 = 0; r8 < 8; ++r8) {
    const int row = r8 * 8 + (lane >> 2), part = lane & 3;
    u[r8][0] = __ldg(reinterpret_cast<const uint4*>(src + (long long)row * l.sr + part * 16));
    u[r8][1] = __ldg(reinterpret_cast<const uint4*>(src + (long long)row * l.sr + part * 16) + 1);
  }
  float m = 0.f;
#pragma unroll
  for (int r8 = 0; r8 < 8; ++r8) {
    const uint32_t w[8] = {u[r8][0].x, u[r8][0].y, u[r8][0].z, u[r8][0].w, u[r8][1].x, u[r8][1].y, u[r8][1].z, u[r8][1].w};
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < 16; ++d) {
      const float x = __uint_as_float((d & 1) ? (w[d >> 1] & 0xffff0000u) : (w[d >> 1] << 16));
      acc = __fadd_rn(acc, __fmul_rn(x, x));
    }
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
    acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
    m = fmaxf(m, sqrtf(acc));
  }
  return plong::warp_max(m);
}

template <int MODE>
P9A_DEV void kv_item8(const Args& a, int bh, int kblk, float* smem, int slot, bf16* dsT, int dslot, int mslot,
                      unsigned char tag) {
  bf16* sk = reinterpret_cast<bf16*>(smem);   // [128][LD]
  bf16* sv = sk + 128 * LD;
  bf16* sq = sv + 128 * LD;                    // [T][LD] (CTA-live tiles only)
  bf16* sd = sq + plong::RES_MAX * LD;
  float* slse = reinterpret_cast<float*>(sd + plong::RES_MAX * LD);
  float* sdel = slse + plong::RES_MAX;
  plong::MaskTile* p9_kvmt = tail_at<plong::MaskTile>(smem, BWD_TAIL);
  float* p9_kvqn = tail_at<float>(smem, BWD_TAIL + 512);
  unsigned& p9_kv_live = *tail_at<unsigned>(smem, BWD_TAIL + 1056);
  const int t = a.t, nqb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = kblk * 128, j0 = k0 + warp * 16 + g, j1 = j0 + 8;
  const float sabs = plong::scale_abs(a);
  const bf16* qbase = a.q + pshort::head_offset(a.lq, bh, a.heads);
  const bf16* dbase = a.dout + pshort::head_offset(a.ld, bh, a.heads);
  tstamp(13);
  if (threadIdx.x == 0) p9_kv_live = 0u;
  {                                             // kind-1 entries: [qb][warp] = table[qb * (t / 16) + (k0 / 16) + warp]
    const int i = threadIdx.x;
    if (i < 8 * nqb) p9_kvmt[i] = plong::plong_mtab[mslot][1][(i / 8) * (t / 16) + k0 / 16 + (i % 8)];
  }
  plong::tbar_init(nqb);
  __syncthreads();
  tstamp(19);
  float tmx[plong::RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  for (int qb = 0; qb < nqb; ++qb) {
    const plong::MaskTile e = p9_kvmt[qb * 8 + warp];
    tmx[qb] = e.mx;
    if (e.zero) zeros |= 1u << qb;
    if (e.mx > -1000.f) live |= 1u << qb;
  }
  if (lane == 0) atomicOr(&p9_kv_live, live);
  plong::load_rows_res(sk, a.k + pshort::head_offset(a.lk, bh, a.heads) + (long long)k0 * a.lk.sr, a.lk.sr, 128);
  plong::load_rows_res(sv, a.v + pshort::head_offset(a.lv, bh, a.heads) + (long long)k0 * a.lv.sr, a.lv.sr, 128);
  pshort::cp_commit();
  for (int i = (int)threadIdx.x; i < t; i += 256) slse[i] = __ldcg(a.lse + (long long)bh * t + i);
  __syncthreads();
  tstamp(20);
  const unsigned cta_live = p9_kv_live;
  for (int qb = 0; qb < nqb; ++qb) {
    if ((cta_live >> qb) & 1u) {
      plong::load_rows_res(sq + qb * 64 * LD, qbase + (long long)qb * 64 * a.lq.sr, a.lq.sr, 64);
      plong::load_rows_res(sd + qb * 64 * LD, dbase + (long long)qb * 64 * a.ld.sr, a.ld.sr, 64);
      plong::tbar_arrive(qb);
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  __syncthreads();
  // P9-B v8: this item's delta rows (staged dO, FP32 O) and the head batch's query-block norm maxima, computed here
  // with the arithmetic of delta_table / block_norm_table (same values) instead of in a phase prologue.
  delta_blocks8(a, bh, cta_live, sd, sdel);
  for (int qb = warp; qb < nqb; qb += 8) p9_kvqn[qb] = block_norm8(a.q, a.lq, a.heads, bh, qb);
  __syncthreads();
  tstamp(14);
  uint32_t ka[KS][4], va[KS][4];
  plong::load_afrag(ka, sk + warp * 16 * LD, lane);
  plong::load_afrag(va, sv + warp * 16 * LD, lane);
  float dv[8][4], dk[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh)
#pragma unroll
    for (int e = 0; e < 4; ++e) dv[nh][e] = dk[nh][e] = 0.f;
  unsigned computed = 0u;
  bf16* myrows = dsT + ((long long)bh * t + j0) * t;                // dS^T row j0 (row j1 = j0 + 8)
  auto tile = [&](int qb) {
    if ((zeros >> qb) & 1u) kv_tile<MODE, false>(a, bh, j0, j1, qb * 64, ka, va, sq, sd, slse, sdel, lane, dv, dk, myrows);
    else kv_tile<MODE, true>(a, bh, j0, j1, qb * 64, ka, va, sq, sd, slse, sdel, lane, dv, dk, myrows);
    computed |= 1u << qb;
  };
#pragma unroll 1
  for (int qb = 0; qb < nqb; ++qb) {
    if ((live >> qb) & 1u) {
      plong::tbar_wait(qb);
      tile(qb);
    } else if ((cta_live >> qb) & 1u) {          // published block: this warp's rows are exact zeros
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = qb * 64 + nt * 8 + 2 * tig;
        pshort::store_bf16x2(myrows + col, 0.f, 0.f);
        pshort::store_bf16x2(myrows + 8LL * t + col, 0.f, 0.f);
      }
    }
  }
  for (int qb = 0; qb < nqb; ++qb) if ((cta_live >> qb) & 1u) plong::tbar_wait(qb);
  // Remainder guard: per-row v2 bound with block maxima of query norms (conservative: never skips a tile v2 computes).
  bool need = false;
  {
    float knv = 0.f;
    {
      // this warp's 16 key norms (v2 row_norms<128> arithmetic on sk rows, 2 threads per row: warp w holds rows 16w..)
      const int row = threadIdx.x >> 1, part = threadIdx.x & 1;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        const float x = __bfloat162float(sk[row * LD + part * 32 + d]);
        acc = __fadd_rn(acc, __fmul_rn(x, x));
      }
      acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
      knv = plong::warp_max(sqrtf(acc));
    }
    for (int qb = 0; qb < nqb; ++qb) {
      if ((computed >> qb) & 1u) continue;
      const float qn = p9_kvqn[qb];
      bool safe = true;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qi = qb * 64 + nt * 8 + 2 * tig + e;
          const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn), knv), NORM_SLACK) + tmx[qb];
          safe = safe && (b < slse[qi] - MARGIN);
        }
      if (!__all_sync(0xffffffffu, safe)) need = true;
    }
  }
  unsigned extra = 0u;                          // query tiles computed by the slow path (published as well)
  if (__syncthreads_or(need)) {                 // slow path: v2 remainder on fully copied operands
    float* sqn = sdel + plong::RES_MAX;         // [T] (inside the shared budget: sk/sv/sq/sd/lse/del + T floats)
    for (int qb = 0; qb < nqb; ++qb) {
      if ((cta_live >> qb) & 1u) continue;
      plong::load_rows_res(sq + qb * 64 * LD, qbase + (long long)qb * 64 * a.lq.sr, a.lq.sr, 64);
      plong::load_rows_res(sd + qb * 64 * LD, dbase + (long long)qb * 64 * a.ld.sr, a.ld.sr, 64);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    delta_blocks8(a, bh, ~cta_live & ((1u << nqb) - 1u), sd, sdel);
    __syncthreads();
    float* kn = reinterpret_cast<float*>(smem) ;  // reuse? no: key norms via row_norms into scratch below
    (void)kn;
    float* p9_kvkn = tail_at<float>(smem, BWD_TAIL + 544);
    plong::row_norms<128>(sk, p9_kvkn);
    for (int qb = 0; qb < nqb; ++qb) plong::row_norms<64>(sq + qb * 64 * LD, sqn + qb * 64);
    __syncthreads();
    const float knw = plong::warp_max(fmaxf(p9_kvkn[warp * 16 + g], p9_kvkn[warp * 16 + g + 8]));
    for (int qb = 0; qb < nqb; ++qb) {
      if ((computed >> qb) & 1u) continue;
      bool safe = true;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int qi = qb * 64 + nt * 8 + 2 * tig + e;
          const float b = __fmul_rn(__fmul_rn(__fmul_rn(sabs, sqn[qi]), knw), NORM_SLACK) + tmx[qb];
          safe = safe && (b < slse[qi] - MARGIN);
        }
      if (__all_sync(0xffffffffu, safe)) continue;
      tile(qb);
      extra |= 1u << qb;
    }
    // Rows of slow-path tiles not computed by some warps: zeros (the tile's other warps published values).
    const unsigned any_extra = __reduce_or_sync(0xffffffffu, extra);
    unsigned& p9_kv_extra = *tail_at<unsigned>(smem, BWD_TAIL + 1060);
    if (threadIdx.x == 0) p9_kv_extra = 0u;
    __syncthreads();
    if (lane == 0) atomicOr(&p9_kv_extra, any_extra);
    __syncthreads();
    const unsigned all_extra = p9_kv_extra;
    for (int qb = 0; qb < nqb; ++qb) {
      if (!((all_extra >> qb) & 1u) || ((computed >> qb) & 1u)) continue;
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const int col = qb * 64 + nt * 8 + 2 * tig;
        pshort::store_bf16x2(myrows + col, 0.f, 0.f);
        pshort::store_bf16x2(myrows + 8LL * t + col, 0.f, 0.f);
      }
    }
    if (threadIdx.x < nqb && ((all_extra >> threadIdx.x) & 1u))
      p9_pub[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = tag;
  }
  if ((int)threadIdx.x < nqb) {
    if ((cta_live >> threadIdx.x) & 1u) p9_pub[dslot][(bh * nqb + threadIdx.x) * 4 + kblk] = tag;
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
  tstamp(15);
}

// dQ item (bh, 128-query block): per warp 16 query rows; for each 128-key block in ascending order whose dS^T block is
// published for the warp's query tile, dq += dS (BF16, from shared via ldmatrix.trans) times K rows, chained per
// 64-key tile and k16 step exactly as v2 q_item2's tile_product<1>.
constexpr int DS_LD = 128 + 8;                     // dS^T staging row stride (queries)
__host__ __device__ constexpr int dq_shared_bytes() { return 2 * (128 * LD + 128 * DS_LD) * 2; }

template <int TAGGED = 0>
P9A_DEV void dq_item4(const Args& a, int bh, int qblk, float* smem, const bf16* dsT, int dslot,
                      unsigned tag = 1) {
  const int t = a.t, nqb = t / 64, nkb2 = t / 128, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8, qt = (q0 + warp * 16) / 64;
  bf16* ring = reinterpret_cast<bf16*>(smem);   // 2 x { K [128][LD], dS^T [128 keys][DS_LD] }
  constexpr int SLOT = 128 * LD + 128 * DS_LD;
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  unsigned char (*p9_dqpub)[4] = reinterpret_cast<unsigned char (*)[4]>(tail_at<unsigned char>(smem, BWD_TAIL + 1064));
  tstamp(17);
  if ((int)threadIdx.x < 2 * nkb2) {
    const int h = threadIdx.x / nkb2, kb2 = threadIdx.x % nkb2;
    const int pi_ = (bh * nqb + qblk * 2 + h) * 4 + kb2;
    p9_dqpub[h][kb2] = TAGGED == 2 ? (unsigned char)(__ldcg(&p9_pub32[dslot][pi_]) == tag)
                     : TAGGED == 1 ? (unsigned char)(p9_pub[dslot][pi_] == (unsigned char)tag)
                                   : p9_pub[dslot][pi_];
  }
  __syncthreads();
  unsigned cta = 0u, mine = 0u;
  for (int kb2 = 0; kb2 < nkb2; ++kb2) {
    if (p9_dqpub[0][kb2] | p9_dqpub[1][kb2]) cta |= 1u << kb2;
    if (p9_dqpub[(warp * 16) / 64][kb2]) mine |= 1u << kb2;
  }
  int list[4], n = 0;
  for (int kb2 = 0; kb2 < nkb2; ++kb2) if ((cta >> kb2) & 1u) list[n++] = kb2;
  auto load = [&](int i) {
    bf16* dst = ring + (i & 1) * SLOT;
    const int kb2 = list[i];
    plong::load_rows_res(dst, kbase + (long long)kb2 * 128 * a.lk.sr, a.lk.sr, 128);
    bf16* dd = dst + 128 * LD;
    const bf16* src = dsT + ((long long)bh * t + kb2 * 128) * t + q0;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int idx = (int)threadIdx.x + 256 * c, row = idx >> 4, col = (idx & 15) * 8;
      pshort::cp16(dd + row * DS_LD + col, src + (long long)row * t + col);
    }
    pshort::cp_commit();
  };
  float dq[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) dq[nh][0] = dq[nh][1] = dq[nh][2] = dq[nh][3] = 0.f;
  if (n > 0) load(0);
  for (int i = 0; i < n; ++i) {
    if (i + 1 < n) { load(i + 1); asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }
    else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if ((mine >> list[i]) & 1u) {
      const bf16* sk = ring + (i & 1) * SLOT;
      const bf16* sds = sk + 128 * LD;
#pragma unroll
      for (int half = 0; half < 2; ++half) {     // 64-key tiles 2 kb2 and 2 kb2 + 1
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
          const int kr = half * 64 + k2 * 16;    // key row of this k16 step
          uint32_t af[4];
          const int mat = lane >> 3, rr = lane & 7;
          pshort::ldsm_x4_t(af, sds + (kr + (mat >> 1) * 8 + rr) * DS_LD + warp * 16 + (mat & 1) * 8);
#pragma unroll
          for (int n2 = 0; n2 < 4; ++n2) {
            uint32_t b4[4];
            pshort::ldsm_x4_t(b4, sk + (kr + (lane & 15)) * LD + (2 * n2 + (lane >> 4)) * 8);
            const uint32_t b0[2] = {b4[0], b4[1]}, b1[2] = {b4[2], b4[3]};
            pshort::mma(dq[2 * n2], af, b0);
            pshort::mma(dq[2 * n2 + 1], af, b1);
          }
        }
      }
    }
    __syncthreads();                              // slot reusable
  }
  bf16* dqb = a.dq + pshort::head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) {
    const int col = nh * 8 + 2 * tig;
    pshort::store_bf16x2(dqb + (long long)r0 * a.ldq.sr + col, dq[nh][0], dq[nh][1]);
    pshort::store_bf16x2(dqb + (long long)r1 * a.ldq.sr + col, dq[nh][2], dq[nh][3]);
  }
  tstamp(18);
}

__host__ __device__ constexpr int v4_backward_shared_bytes() { return BWD_TAIL + 1152; }

// Backward phase: prologue (mask tables if PRE == 0, query-block norm maxima, delta table, published flags cleared),
// grid sync, key/value items, grid sync, dQ items. dsT: [bh][T][T] BF16 scratch.
template <int PRE>
P9A_PHASE void backward_phase4(const Args& a, int bhs, float* smem, int slot, bf16* dsT, int dslot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, nkb2 = a.t / 128, nqb = a.t / 64;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  tstamp(10);
#ifdef P9A_DELTA_PF
  delta_prefetch(a, bhs);
#endif
  if (PRE == 0) plong::mtab_build(a, mslot, 3);
  if (!a.qnorm_ready) block_norm_table(a.q, a.lq, a.heads, bhs, a.t, p9_qnmax[slot]);
  tstamp(23);
  if (!a.delta_ready) delta_table(a, slot, bhs);
  tstamp(24);
  for (int i = blockIdx.x * 256 + threadIdx.x; i < bhs * nqb * 4; i += G * 256) p9_pub[dslot][i] = 0;
  tstamp(11);
  plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  tstamp(12);
  const int items = nkb2 * bhs;
  for (int r = 0;; ++r) {                         // key/value items, decreasing cost (low key blocks first)
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (a.scale_mode == 1) kv_item4<1>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
    else if (a.scale_mode == 2) kv_item4<2>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
    else kv_item4<0>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
  }
  __syncthreads();
  plong::tbar_release();
  plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  tstamp(16);
  for (int r = 0;; ++r) {                         // dQ items, decreasing cost (high query blocks first)
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    dq_item4(a, j % bhs, nkb2 - 1 - j / bhs, smem, dsT, dslot);
  }
  __syncthreads();
}

// Backward v6: backward_phase4's items with one grid synchronization. Each key/value item increments its head
// batch's completion counter (release) after publishing dS^T and its flags; each CTA reads, before the grid
// synchronization (so before any key/value item of this call runs), the counters of the head batches of its own dQ
// items, and a dQ item waits (acquire) for base + T/128 completions of its head batch. dQ work of finished head
// batches overlaps the remaining key/value items. Item outputs are identical to v4 (same items, same arithmetic).
__device__ unsigned p9_kvdone[SLOTS][1024];
#ifndef P9A_DQ_QUEUE
#define P9A_DQ_QUEUE 0
#endif
#if P9A_DQ_QUEUE
__device__ unsigned p9_dqq[SLOTS];                 // P9-B: dQ work-queue claims (monotonic over calls)
__device__ unsigned p9_dqcalls[SLOTS][256];        // P9-B: per CTA, backward calls of this slot
__shared__ unsigned p9_dqbase[128];                // key/value completion snapshot per head batch (bhs <= 128)
__shared__ unsigned p9_dqcall;
__shared__ int p9_dqitem;
#endif


template <int PRE>
P9A_PHASE void backward_phase6(const Args& a, int bhs, float* smem, int slot, bf16* dsT, int dslot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, nkb2 = a.t / 128, nqb = a.t / 64;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  const int items = nkb2 * bhs;
  unsigned* p9_base6 = tail_at<unsigned>(smem, BWD_TAIL + 1088);
#if P9A_DQ_QUEUE
  // P9-B: dQ items from a call-epoch work queue; every head batch's key/value completion count is snapshotted before
  // the phase's grid synchronization (no completion of this call can precede it).
  if (bhs > 128) __trap();
  for (int b = threadIdx.x; b < bhs; b += 256) p9_dqbase[b] = *reinterpret_cast<volatile unsigned*>(&p9_kvdone[slot][b]);
  if (threadIdx.x == 0) p9_dqcall = ++p9_dqcalls[slot][c];
#else
  if (threadIdx.x < 8) {                         // this CTA's dQ items (rounds r = threadIdx.x)
    const int r = threadIdx.x;
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    p9_base6[r] = j < items ? *reinterpret_cast<volatile unsigned*>(&p9_kvdone[slot][j % bhs]) : 0u;
  }
#endif
  tstamp(10);
#ifdef P9A_DELTA_PF
  delta_prefetch(a, bhs);
#endif
  if (PRE == 0) plong::mtab_build(a, mslot, 3);
  if (!a.qnorm_ready) block_norm_table(a.q, a.lq, a.heads, bhs, a.t, p9_qnmax[slot]);
  tstamp(23);
  if (!a.delta_ready) delta_table(a, slot, bhs);
  tstamp(24);
  for (int i = blockIdx.x * 256 + threadIdx.x; i < bhs * nqb * 4; i += G * 256) p9_pub[dslot][i] = 0;
  tstamp(11);
  plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  tstamp(12);
#if P9A_KV_QUEUE
  wq_begin(0, slot);
#endif
  for (int r = 0;; ++r) {                         // key/value items, decreasing cost (low key blocks first)
#if P9A_KV_QUEUE
    const int j = wq_claim(0, slot, items);
    if (j >= items) break;
#else
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
#endif
    if (a.scale_mode == 1) kv_item4<1>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
    else if (a.scale_mode == 2) kv_item4<2>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
    else kv_item4<0>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot);
    __syncthreads();                              // every thread's dS^T, flag and dK/dV stores issued
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(&p9_kvdone[slot][j % bhs], 1u);
    }
  }
  __syncthreads();
  plong::tbar_release();
  tstamp(16);
  for (int r = 0;; ++r) {                         // dQ items, decreasing cost (high query blocks first)
#if P9A_DQ_QUEUE
    if (r) __syncthreads();
    if (threadIdx.x == 0) {
      // claims of a call: items + gridDim.x (one failing claim per CTA), so the call's base is fixed
      const unsigned base = (p9_dqcall - 1u) * (unsigned)(items + G);
      p9_dqitem = (int)(atomicAdd(&p9_dqq[slot], 1u) - base);
    }
    __syncthreads();
    const int j = p9_dqitem;
    if (j >= items) break;
    if (threadIdx.x == 0) {
      const unsigned target = p9_dqbase[j % bhs] + (unsigned)nkb2;
#else
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned target = p9_base6[r] + (unsigned)nkb2;
#endif
      unsigned seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(&p9_kvdone[slot][j % bhs]) : "memory");
      } while ((int)(seen - target) < 0);
    }
    __syncthreads();
    dq_item4(a, j % bhs, nkb2 - 1 - j / bhs, smem, dsT, dslot);
  }
  __syncthreads();
}

// LATCH-STEP-06 (long_attention.P9_BWD_TAGS): backward_phase6 without its in-phase grid synchronization. The
// publication flags carry the call epoch (32-bit, never reused) instead of being cleared, and dQ items wait for the
// head batch's key/value completion count epoch x (key blocks): every call of a slot runs every key/value item once,
// so the count is exact without a snapshot. Requires the prebuilt mask table (PRE 1), the ready delta/q-norm tables
// and the dQ work queue. Item arithmetic, item order and published tiles are those of backward_phase6.
#if P9A_DQ_QUEUE
template <int PRE>
P9A_PHASE void backward_phase6t(const Args& a, int bhs, float* smem, int slot, bf16* dsT, int dslot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, nkb2 = a.t / 128;
  if (PRE == 0 || bhs > 128 || !a.qnorm_ready || !a.delta_ready) __trap();
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  const int items = nkb2 * bhs;
  if (threadIdx.x == 0) p9_dqcall = ++p9_dqcalls[slot][c];
  __syncthreads();
  const unsigned epoch = p9_dqcall;
  tstamp(12);
  for (int r = 0;; ++r) {                         // key/value items, decreasing cost (low key blocks first)
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (a.scale_mode == 1) kv_item4<1>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, epoch);
    else if (a.scale_mode == 2) kv_item4<2>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, epoch);
    else kv_item4<0>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, epoch);
    __syncthreads();                              // every thread's dS^T, flag and dK/dV stores issued
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(&p9_kvdone[slot][j % bhs], 1u);
    }
  }
  __syncthreads();
  plong::tbar_release();
  tstamp(16);
  for (int r = 0;; ++r) {                         // dQ items, decreasing cost (high query blocks first)
    if (r) __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned base = (epoch - 1u) * (unsigned)(items + G);
      p9_dqitem = (int)(atomicAdd(&p9_dqq[slot], 1u) - base);
    }
    __syncthreads();
    const int j = p9_dqitem;
    if (j >= items) break;
    if (threadIdx.x == 0) {
      const unsigned target = epoch * (unsigned)nkb2;
      unsigned seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(&p9_kvdone[slot][j % bhs]) : "memory");
      } while ((int)(seen - target) < 0);
    }
    __syncthreads();
    dq_item4<2>(a, j % bhs, nkb2 - 1 - j / bhs, smem, dsT, dslot, epoch);
  }
  __syncthreads();
}
#endif


__device__ unsigned p9_bcall8[SLOTS][256];          // P9-B v8: per CTA, backward calls of this slot
__device__ unsigned p9_kvdone8[SLOTS][1024];        // P9-B v8: per head batch, key/value items completed (monotonic)
template <int PRE>
P9A_PHASE void backward_phase8(const Args& a, int bhs, float* smem, int slot, bf16* dsT, int dslot, int mslot = -1) {
  if (mslot < 0) mslot = slot;
  const int G = gridDim.x, c = blockIdx.x, nkb2 = a.t / 128;
  if (PRE == 0 || bhs > 1024 || G > 256) __trap();                // v8 requires the prebuilt mask table
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  const int items = nkb2 * bhs;
  unsigned* ep = tail_at<unsigned>(smem, BWD_TAIL + 1124);
  if (threadIdx.x == 0) *ep = ++p9_bcall8[slot][c];
  __syncthreads();
  const unsigned epoch = *ep;
  const unsigned char tag = (unsigned char)(epoch % 255u + 1u);
  const unsigned kv_target = epoch * (unsigned)nkb2;
  tstamp(12);
  for (int r = 0;; ++r) {                         // key/value items, decreasing cost (low key blocks first)
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (a.scale_mode == 1) kv_item8<1>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, tag);
    else if (a.scale_mode == 2) kv_item8<2>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, tag);
    else kv_item8<0>(a, j % bhs, j / bhs, smem, slot, dsT, dslot, mslot, tag);
    __syncthreads();                              // every thread's dS^T, flag and dK/dV stores issued
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(&p9_kvdone8[slot][j % bhs], 1u);
    }
  }
  __syncthreads();
  plong::tbar_release();
  tstamp(16);
  for (int r = 0;; ++r) {                         // dQ items, decreasing cost (high query blocks first)
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (threadIdx.x == 0) {
      unsigned seen;
      do {
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(seen) : "l"(&p9_kvdone8[slot][j % bhs]) : "memory");
      } while ((int)(seen - kv_target) < 0);
    }
    __syncthreads();
    dq_item4<1>(a, j % bhs, nkb2 - 1 - j / bhs, smem, dsT, dslot, tag);
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------------ forward (v5)
// v4 plus: Q is copied and waited before the key/value copies are issued, and the score products of the next live
// tile are issued before the softmax and value product of the current one (software pipelining; same per-tile
// operations in the same order per row group, so outputs stay bitwise those of v2).
template <int MODE>
P9A_DEV void fwd_soft_prod(const Args& a, float (&s)[8][4], const float (&mk)[8][4], const bf16* svt, int lane,
                           float& m0, float& m1, float& l0, float& l1, float (&oacc)[8][4]) {
  float tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x = scl<MODE>(s[nt][e], a.scale) + mk[nt][e];
      s[nt][e] = x;
      if (e < 2) tm0 = fmaxf(tm0, x); else tm1 = fmaxf(tm1, x);
    }
  const float n0 = fmaxf(m0, plong::quad_max(tm0)), n1 = fmaxf(m1, plong::quad_max(tm1));
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
  product_x4(oacc, s, svt, lane);
  m0 = n0;
  m1 = n1;
}

template <int MODE>
P9A_DEV void forward_item5(const Args& a, int bh, int qblk, float* smem, int slot) {
  bf16* sq = reinterpret_cast<bf16*>(smem);
  bf16* sk = sq + 128 * LD;
  bf16* sv = sk + plong::RES_MAX * LD;
  float* qn = reinterpret_cast<float*>(sv + plong::RES_MAX * LD);
  float* smask = qn + 128;                                         // [8 warps][16][64]
  __shared__ unsigned p9_live_or5;
  __shared__ plong::MaskTile p9_mt5[8 * (plong::RES_MAX / 64)];
  __shared__ float p9_kn5[plong::RES_MAX / 64];
  const int t = a.t, nkb = t / 64, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 128, r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const float sabs = plong::scale_abs(a);
  const bf16* kbase = a.k + pshort::head_offset(a.lk, bh, a.heads);
  const bf16* vbase = a.v + pshort::head_offset(a.lv, bh, a.heads);
  tstamp(1);
  // Q first (own group), then tables.
  plong::load_rows_res(sq, a.q + pshort::head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, a.lq.sr, 128);
  pshort::cp_commit();
  if (threadIdx.x == 0) p9_live_or5 = 0u;
  if ((int)threadIdx.x < 8 * nkb) p9_mt5[threadIdx.x] = plong::plong_mtab[slot][0][(q0 / 16) * nkb + threadIdx.x];
  else if ((int)threadIdx.x < 9 * nkb) p9_kn5[threadIdx.x - 8 * nkb] = p9_knmax[slot][bh * nkb + threadIdx.x - 8 * nkb];
  plong::tbar_init(nkb);
  __syncthreads();
  float tmx[plong::RES_MAX / 64];
  unsigned live = 0u, zeros = 0u;
  for (int kb = 0; kb < nkb; ++kb) {
    const plong::MaskTile e = p9_mt5[warp * nkb + kb];
    tmx[kb] = e.mx;
    if (e.zero) zeros |= 1u << kb;
    if (e.mx > -1000.f) live |= 1u << kb;
  }
  if (lane == 0) atomicOr(&p9_live_or5, live);
  const unsigned nz = live & ~zeros;
  const int stage_kb = nz ? __ffs(nz) - 1 : -1;
  const bool can_stage = a.has_mask && a.m_j == 1 && (a.m_i % 4) == 0 && (a.m_b % 4) == 0 &&
                         ((reinterpret_cast<uintptr_t>(a.mask) & 15) == 0);
  if (stage_kb >= 0 && can_stage) {
    const float* src = a.mask + (long long)bh * a.m_b + (long long)(q0 + warp * 16) * a.m_i + stage_kb * 64;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int idx = lane + 32 * c, row = idx >> 4, col = (idx & 15) * 4;
      pshort::cp16(smask + warp * 1024 + row * 64 + col, src + (long long)row * a.m_i + col);
    }
  }
  pshort::cp_commit();
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");          // Q and the staged mask rows
  __syncthreads();
  const unsigned cta_live = p9_live_or5;
  for (int kb = 0; kb < nkb; ++kb) {
    if ((cta_live >> kb) & 1u) {
      plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
      plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
      plong::tbar_arrive(kb);
    }
  }
  tstamp(2);
  plong::row_norms<128>(sq, qn);
  uint32_t qa[KS][4];
  plong::load_afrag(qa, sq + warp * 16 * LD, lane);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f, oacc[8][4];
#pragma unroll
  for (int nh = 0; nh < 8; ++nh) oacc[nh][0] = oacc[nh][1] = oacc[nh][2] = oacc[nh][3] = 0.f;
  const float* my_mask = smask + warp * 1024;
  auto mask_frag = [&](int kb, float (&mk)[8][4]) {   // +0 for all-zero tiles (adding +0 leaves every value equal)
    if ((zeros >> kb) & 1u) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) mk[nt][0] = mk[nt][1] = mk[nt][2] = mk[nt][3] = 0.f;
    } else if (kb == stage_kb && can_stage) {
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const float2 u = *reinterpret_cast<const float2*>(my_mask + g * 64 + nt * 8 + 2 * tig);
        const float2 w = *reinterpret_cast<const float2*>(my_mask + (g + 8) * 64 + nt * 8 + 2 * tig);
        mk[nt][0] = u.x; mk[nt][1] = u.y; mk[nt][2] = w.x; mk[nt][3] = w.y;
      }
    } else {
      plong::tile_frag<false>(a, bh, r0, r1, kb * 64, mk);
    }
  };
  // Software-pipelined sweep over this warp's live tiles.
  unsigned rest = live;
  int cur = rest ? __ffs(rest) - 1 : -1;
  if (cur >= 0) rest &= rest - 1;
  float sc[8][4], mkc[8][4];
  if (cur >= 0) {
    plong::tbar_wait(cur);
    scores_x4(sc, qa, sk + cur * 64 * LD, lane);
    mask_frag(cur, mkc);
  }
#pragma unroll 1
  while (cur >= 0) {
    tstamp(3);
    const int nxt = rest ? __ffs(rest) - 1 : -1;
    if (nxt >= 0) rest &= rest - 1;
    float sn[8][4], mkn[8][4];
    if (nxt >= 0) {
      plong::tbar_wait(nxt);
      scores_x4(sn, qa, sk + nxt * 64 * LD, lane);
      mask_frag(nxt, mkn);
    }
    fwd_soft_prod<MODE>(a, sc, mkc, sv + cur * 64 * LD, lane, m0, m1, l0, l1, oacc);
    cur = nxt;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) { sc[nt][e] = sn[nt][e]; mkc[nt][e] = mkn[nt][e]; }
  }
  const unsigned computed = live;
  tstamp(4);
  for (int kb = 0; kb < nkb; ++kb) if ((cta_live >> kb) & 1u) plong::tbar_wait(kb);
  bool need = false;
  {
    __syncwarp();
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float kn = p9_kn5[kb];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
      if (!__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) need = true;
    }
  }
  if (__syncthreads_or(need)) {
    for (int kb = 0; kb < nkb; ++kb) {
      if ((cta_live >> kb) & 1u) continue;
      plong::load_rows_res(sk + kb * 64 * LD, kbase + (long long)kb * 64 * a.lk.sr, a.lk.sr, 64);
      plong::load_rows_res(sv + kb * 64 * LD, vbase + (long long)kb * 64 * a.lv.sr, a.lv.sr, 64);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    const float qn0 = qn[warp * 16 + g], qn1 = qn[warp * 16 + g + 8];
    for (int kb = 0; kb < nkb; ++kb) {
      if ((computed >> kb) & 1u) continue;
      const float kn = p9_kn5[kb];
      const float b0 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn0), kn), NORM_SLACK) + tmx[kb];
      const float b1 = __fmul_rn(__fmul_rn(__fmul_rn(sabs, qn1), kn), NORM_SLACK) + tmx[kb];
      if (__all_sync(0xffffffffu, (b0 < m0 - MARGIN) && (b1 < m1 - MARGIN))) continue;
      if ((zeros >> kb) & 1u) fwd_tile<MODE, false>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc);
      else fwd_tile<MODE, true>(a, bh, r0, r1, kb, qa, sk, sv, lane, m0, m1, l0, l1, oacc);
    }
  }
  l0 = plong::quad_sum(l0);
  l1 = plong::quad_sum(l1);
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
  tstamp(5);
}

template <int PRE>
P9A_DEV void forward_phase5(const Args& a, int blocks, int bhs, float* smem, int slot) {
  const int G = gridDim.x, c = blockIdx.x, items = blocks * bhs;
  if (threadIdx.x == 0) plong::plong_tbar_live = 0;
  tstamp(6);
  if (PRE == 0) plong::mtab_build(a, slot, 1);
  block_norm_table(a.k, a.lk, a.heads, bhs, a.t, p9_knmax[slot]);
  tstamp(7);
  plong::plong_grid_sync(&plong::plong_mcnt[slot]);
  tstamp(8);
  for (int r = 0;; ++r) {
    const int j = r * G + ((r & 1) ? G - 1 - c : c);
    if (j >= items) break;
    if (r) __syncthreads();
    if (a.scale_mode == 1) forward_item5<1>(a, j % bhs, blocks - 1 - j / bhs, smem, slot);
    else if (a.scale_mode == 2) forward_item5<2>(a, j % bhs, blocks - 1 - j / bhs, smem, slot);
    else forward_item5<0>(a, j % bhs, blocks - 1 - j / bhs, smem, slot);
  }
  __syncthreads();
  plong::tbar_release();
}

}  // namespace p9att

#if defined(P9A_TRACE) && defined(P9A_TRACE_EXPORT)
// P9-B diagnostic: copy the per-CTA attention event stamps to the host (dst: 256 x P9A_TRACE_N, counts: 256) and,
// when reset != 0, clear the counters for the next update.
extern "C" int latch_p9a_trace(unsigned long long* dst, int* counts, int reset) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess && dst) e = cudaMemcpyFromSymbol(dst, p9att::p9a_trace, sizeof(p9att::p9a_trace));
  if (e == cudaSuccess && counts) e = cudaMemcpyFromSymbol(counts, p9att::p9a_trace_n, sizeof(p9att::p9a_trace_n));
  if (e == cudaSuccess && reset) {
    static int zeros[256] = {0};
    e = cudaMemcpyToSymbol(p9att::p9a_trace_n, zeros, sizeof(zeros));
  }
  return (int)e;
}
extern "C" int latch_p9a_trace_n() { return P9A_TRACE_N; }
#endif
