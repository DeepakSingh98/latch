// P-OPS4-D long-context training attention work items for the LATCH-OPS cooperative program.
//
// Scope: self-attention with BF16-exact query/key/value/output-gradient operands (FP32 widenings of BF16 values in
// the captured graph), head width 64, context T a multiple of 128 in [256, 4096], an additive FP32 mask tensor with
// broadcast strides and a captured scalar score scale. Outputs are rounded once to BF16 (their captured consumers
// are BF16 casts). Operand/output layouts are arbitrary 4-D (batch, head, row, unit) strides, so head split/merge
// copies and dtype conversions around the attention are absorbed (P-OPS3-B admission). Requires 256-thread CTAs.
//
// Execution (no T x T storage): tiled online-softmax forward saving the BF16 output and a log2-domain row
// log-sum-exp; backward in two deterministic passes without atomics: a query-block pass computing delta = rowsum(dO*O)
// and dQ, then a key-block pass computing dK and dV. Probabilities are recomputed from the saved log-sum-exp.
//
// Arithmetic (declared; compile-time NS): tensor-core mma.sync m16n8k16 BF16 x BF16 products with FP32 accumulation
// from +0. Q K^T and dO V^T read BF16-exact operands (exact products). Probabilities P and score gradients dS enter
// the P V, P^T dO, dS K and dS^T Q products as NS BF16 pieces (NS = 1: one round-to-nearest BF16 value, the class of
// BF16 FlashAttention/SDPA; NS = 3: an exact split, every product equals the FP32 product). Scores
// x = fl(fl(dot) op c) + mask; exponentials are exp2f of an FMA-prescaled argument (FP32). Row sums use the unrounded
// FP32 probabilities. delta uses the stored BF16 output (NS = 1) or a saved FP32 output (PFLATLONG_O32). Summation orders
// differ from conventional kernels (FP32 association only); every output element has one writer (bitwise
// run-to-run determinism).
//
// Skipping: a 64-key tile is skipped for a 16-row group only with a proof that every probability in it is exactly zero:
// x_ij - lse_i <= |q_i| |k|_tile c + max(mask tile) - (rowmax_j(mask_ij) - |q_i| max|k| c) < -120 (with rounding
// margins), so exp2f of the prescaled argument underflows to +0. Proof tables (per-tile key norms, mask tile maxima,
// all-zero mask flags, per-tile mask row maxima) come from a prepare phase that reads every key and mask element.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// P9-A diagnostic stamps inside attention items (CTA 0, worker leaders; P9_TRACE builds only).
#ifndef P9AX
#ifdef P9_TRACE
#ifndef P9_ATTN_DECL
#define P9_ATTN_DECL
__device__ unsigned long long p9_attn_buf[2][512][6];
__device__ unsigned p9_attn_seq[2];
#endif
#define P9AX(slot, value) do { if (blockIdx.x == 0 && (threadIdx.x & 127) == 0) { const int w_ = threadIdx.x >> 7; \
  const unsigned s_ = p9_attn_seq[w_]; if (s_ < 512) p9_attn_buf[w_][s_][slot] = (value); } } while (0)
#define P9NOW() ([]() { unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); return t_; }())
#else
#define P9AX(slot, value) do { } while (0)
#endif
#endif

namespace pflatlong {

#define PFLATLONG_DEV __device__ __forceinline__

constexpr int NT = 256;
constexpr int NW = 8;
constexpr int HD = 64;
constexpr int LD = 72;    // shared row stride (BF16 elements): 144 bytes, conflict-free ldmatrix
constexpr int BR = 128;   // rows per forward/dQ item, keys per dK/dV item
constexpr int BT = 64;    // tile width: keys (forward/dQ) or queries (dK/dV)
constexpr float L2E = 1.4426950408889634f;
constexpr float DEAD = -120.0f;

struct Layout4 {
  long long sb, sh, sr;  // element strides of (batch, head, row); unit stride along the head width
};

struct Args {
  const __nv_bfloat16 *q, *k, *v, *dout;
  Layout4 lq, lk, lv, ld;
  __nv_bfloat16 *o, *dq, *dk, *dv;
  Layout4 lo, ldq, ldk, ldv;
  const float* mask;
  long long m_b, m_i, m_j;  // mask strides over (batch*heads, query, key); zero for broadcast axes
  int has_mask, mask_batched;
  float* kn;       // [BH][T/64] max key L2 norm per 64-key tile (rounded up)
  float* mmax;     // [MB][T/64][T/64] mask tile maxima
  float* mzero;    // [MB][T/64][T/64] 1 when the mask tile is all zero, else 0
  float* rm;       // [MB][T/64 key tiles][T] mask row maxima per key tile
  float* lse2;     // [BH][T] log2-domain row log-sum-exp (forward)
  float* delta;    // [BH][T] rowsum(dO * O) (dQ pass)
  float* o32;      // [BH][T][64] FP32 output (PFLATLONG_O32 recipes only)
  unsigned long long* live;  // [BH][T/16] live 64-key tiles per 16-row group (forward proof)
  __nv_bfloat16* dsbuf;      // [BH][T/64][T/64][64][64] stored BF16 dS^T tiles (v9 storage choice)
  float* vn;       // [BH][T/64] max value L2 norm per 64-key tile (rounded up; v11 fixed-point bound)
  float* dqf;      // [BH][T] power-of-two dQ fixed-point scale per query row (v11)
  int* dqacc;      // [BH][T/64][64 d][64 queries] int32 dQ^T accumulator tiles, 16-byte chunks swizzled (v11)
  int heads, bh, t;
  float scale;
  int scale_mode;  // 0 none, 1 x*scale, 2 x/scale
  // Release step 09 (defaults keep the original semantics): mask batch = bh / m_heads (per-example masks shared by
  // every head; mask tables are per mask batch); safe_rows: a row with no allowed key gets output 0 and
  // lse2 = +inf, so every backward probability ex2(x * k - lse2) is exactly 0 (v6/v7 items only).
  int m_heads = 1;
  int safe_rows = 0;
};

PFLATLONG_DEV long long head_offset(const Layout4& l, int bh, int heads) {
  return (long long)(bh / heads) * l.sb + (long long)(bh % heads) * l.sh;
}
PFLATLONG_DEV float scaled(const Args& a, float dot) {
  return a.scale_mode == 1 ? dot * a.scale : (a.scale_mode == 2 ? dot / a.scale : dot);
}
// Upper bound of |x * scale| / |x| (rounded up).
PFLATLONG_DEV float scale_bound(const Args& a) {
  const float c = a.scale_mode == 1 ? fabsf(a.scale) : (a.scale_mode == 2 ? 1.0f / fabsf(a.scale) : 1.0f);
  return c * (1.0f + 1.0f / 256.0f);
}
PFLATLONG_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
PFLATLONG_DEV void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
PFLATLONG_DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
PFLATLONG_DEV void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
PFLATLONG_DEV void cp_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }
PFLATLONG_DEV void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PFLATLONG_DEV void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
// Register-only tensor-core product: not volatile, so independent accumulators interleave.
PFLATLONG_DEV void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
PFLATLONG_DEV uint32_t pack(float x0, float x1) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
  return *reinterpret_cast<const uint32_t*>(&h);
}
// NS BF16x2 pieces of two FP32 values (x = sum of pieces; exact for NS = 3).
template <int NS>
PFLATLONG_DEV void split2(uint32_t (&out)[NS], float x0, float x1) {
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    out[s] = *reinterpret_cast<const uint32_t*>(&h);
    x0 -= __low2float(h);
    x1 -= __high2float(h);
  }
}
PFLATLONG_DEV void store_bf16x2(__nv_bfloat16* p, float x0, float x1) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(x0, x1);
}
PFLATLONG_DEV float warp_max(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
  return x;
}
PFLATLONG_DEV float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
PFLATLONG_DEV float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Copy ROWS x 64 BF16 rows (row stride `stride` elements) into shared [ROWS][LD]; all 256 threads.
template <int ROWS>
PFLATLONG_DEV void load_rows(__nv_bfloat16* dst, const __nv_bfloat16* src, long long stride) {
  constexpr int ITER = ROWS * 8 / NT;
#pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int idx = (int)threadIdx.x + k * NT, r = idx >> 3, c = idx & 7;
    cp16(dst + r * LD + 8 * c, src + (long long)r * stride + 8 * c);
  }
}
// A fragments (m16 x k16 steps over 64 units) of shared rows [row0, row0 + 16).
PFLATLONG_DEV void a_frags(uint32_t (&a)[4][4], const __nv_bfloat16* s, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks)
    ldsm_x4(a[ks], s + (row0 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + 16 * ks + (lane >> 4) * 8);
}
// acc[n] (+)= A(16 x 64) * B where B[k][n] = S[n][k] (rows of S are the n dimension); n-tiles [n0, n0 + NN).
template <int NN>
PFLATLONG_DEV void mma_nt(float (&acc)[NN][4], const uint32_t (&a)[4][4], const __nv_bfloat16* s, int n0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
    for (int p = 0; p < NN / 2; ++p) {
      uint32_t b[4];
      ldsm_x4(b, s + ((n0 + 2 * p + (lane >> 4)) * 8 + (lane & 7)) * LD + 16 * ks + ((lane >> 3) & 1) * 8);
      mma(acc[2 * p], a[ks], b[0], b[1]);
      mma(acc[2 * p + 1], a[ks], b[2], b[3]);
    }
  }
}
// out[8 n-tiles over 64 units] += P(16 x 16*KK pieces) * S rows [r0, r0 + 16*KK) (B[k][n] = S[k][n], transposed load).
template <int NS, int KK>
PFLATLONG_DEV void mma_pv(float (&out)[8][4], const uint32_t (&pa)[KK][NS][4], const __nv_bfloat16* s, int r0, int lane) {
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint32_t b[4];
      ldsm_x4_t(b, s + (r0 + 16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + 8 * (2 * p + (lane >> 4)));
#pragma unroll
      for (int piece = 0; piece < NS; ++piece) {
        mma(out[2 * p], pa[kk][piece], b[0], b[1]);
        mma(out[2 * p + 1], pa[kk][piece], b[2], b[3]);
      }
    }
  }
}
// Pack C fragments of n-tiles [2kk, 2kk + 1] (kk < KK) into A-operand pieces.
template <int NS, int KK, int NN>
PFLATLONG_DEV void to_a(uint32_t (&pa)[KK][NS][4], const float (&c)[NN][4]) {
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
    uint32_t t0[NS], t1[NS], t2[NS], t3[NS];
    split2<NS>(t0, c[2 * kk][0], c[2 * kk][1]);
    split2<NS>(t1, c[2 * kk][2], c[2 * kk][3]);
    split2<NS>(t2, c[2 * kk + 1][0], c[2 * kk + 1][1]);
    split2<NS>(t3, c[2 * kk + 1][2], c[2 * kk + 1][3]);
#pragma unroll
    for (int s = 0; s < NS; ++s) {
      pa[kk][s][0] = t0[s];
      pa[kk][s][1] = t1[s];
      pa[kk][s][2] = t2[s];
      pa[kk][s][3] = t3[s];
    }
  }
}

// ---------------------------------------------------------------------------------------------------- prepare
// One warp per unit: key-norm units (bh, key tile) then mask units (mask batch, query tile, key tile). Loads are
// row-contiguous across lanes (8 lanes per 128-byte key row, 16 lanes per 256-byte mask row) with shuffle reductions
// in a fixed order (results are order-independent maxima, flags and per-row sums of squares).
PFLATLONG_DEV void prepare_unit(const Args& a, long long u) {
  const int lane = threadIdx.x & 31, nt = a.t >> 6;
  const long long nk = (long long)a.bh * nt;
  if (u < nk) {
    const int bh = (int)(u / nt), kt = (int)(u % nt);
    const __nv_bfloat16* base = a.k + head_offset(a.lk, bh, a.heads);
    const int sub = lane & 7, rsub = lane >> 3;  // 8 lanes x 16 bytes per key row, 4 rows per instruction
    uint4 w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      w[i] = __ldcg(reinterpret_cast<const uint4*>(base + (long long)(kt * 64 + 4 * i + rsub) * a.lk.sr) + sub);
    float best = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc = fmaf(bf_lo(ws[e]), bf_lo(ws[e]), acc);
        acc = fmaf(bf_hi(ws[e]), bf_hi(ws[e]), acc);
      }
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      best = fmaxf(best, sqrtf(acc));
    }
    best = warp_max(best);
    if (lane == 0) a.kn[u] = best * (1.0f + 1.0f / 256.0f);
    if (a.vn) {  // v11: value norms bound the fixed-point dQ accumulation (same reduction as kn)
      const __nv_bfloat16* vb = a.v + head_offset(a.lv, bh, a.heads);
#pragma unroll
      for (int i = 0; i < 16; ++i)
        w[i] = __ldcg(reinterpret_cast<const uint4*>(vb + (long long)(kt * 64 + 4 * i + rsub) * a.lv.sr) + sub);
      float vbest = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          acc = fmaf(bf_lo(ws[e]), bf_lo(ws[e]), acc);
          acc = fmaf(bf_hi(ws[e]), bf_hi(ws[e]), acc);
        }
#pragma unroll
        for (int o = 1; o < 8; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        vbest = fmaxf(vbest, sqrtf(acc));
      }
      vbest = warp_max(vbest);
      if (lane == 0) a.vn[u] = vbest * (1.0f + 1.0f / 256.0f);
    }
    return;
  }
  u -= nk;
  const int mb = (int)(u / ((long long)nt * nt)), rem = (int)(u % ((long long)nt * nt));
  const int qt = rem / nt, kt = rem % nt;
  const float* mbase = a.mask + (long long)mb * a.m_b;
  float tmax = -INFINITY;
  bool zero = true;
  const int sub = lane & 15, rsub = lane >> 4;  // 16 lanes x 16 bytes per mask row, 2 rows per instruction
  const bool vec = a.m_j == 1 && (a.m_i & 3) == 0 && (a.m_b & 3) == 0 &&
                   ((reinterpret_cast<uintptr_t>(a.mask) & 15) == 0);
  float4 w[32];  // all 32 loads in flight before any reduction
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const float* rp = mbase + (long long)(qt * 64 + 2 * i + rsub) * a.m_i + (long long)(kt * 64) * a.m_j;
    w[i] = vec ? __ldcg(reinterpret_cast<const float4*>(rp) + sub)
               : make_float4(__ldcg(rp + (long long)(4 * sub) * a.m_j), __ldcg(rp + (long long)(4 * sub + 1) * a.m_j),
                             __ldcg(rp + (long long)(4 * sub + 2) * a.m_j), __ldcg(rp + (long long)(4 * sub + 3) * a.m_j));
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int row = qt * 64 + 2 * i + rsub;
    float rmax = fmaxf(fmaxf(w[i].x, w[i].y), fmaxf(w[i].z, w[i].w));
    zero = zero && w[i].x == 0.f && w[i].y == 0.f && w[i].z == 0.f && w[i].w == 0.f;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
    if (sub == 0) a.rm[((long long)mb * nt + kt) * a.t + row] = rmax;
    tmax = fmaxf(tmax, rmax);
  }
  tmax = warp_max(tmax);
  zero = __all_sync(0xffffffffu, zero);
  if (lane == 0) {
    a.mmax[u] = tmax;
    a.mzero[u] = zero ? 1.f : 0.f;
  }
}

// P9-A: the key-norm value of prepare_unit unit (bh, kt), evaluated by one warp with the identical arithmetic.
PFLATLONG_DEV float kn_tile(const Args& a, int bh, int kt) {
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* base = a.k + head_offset(a.lk, bh, a.heads);
  const int sub = lane & 7, rsub = lane >> 3;
  uint4 w[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    w[i] = __ldcg(reinterpret_cast<const uint4*>(base + (long long)(kt * 64 + 4 * i + rsub) * a.lk.sr) + sub);
  float best = 0.f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const uint32_t ws[4] = {w[i].x, w[i].y, w[i].z, w[i].w};
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc = fmaf(bf_lo(ws[e]), bf_lo(ws[e]), acc);
      acc = fmaf(bf_hi(ws[e]), bf_hi(ws[e]), acc);
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    best = fmaxf(best, sqrtf(acc));
  }
  best = warp_max(best);
  return best * (1.0f + 1.0f / 256.0f);
}

// Live 64-key tiles for this warp's 16 rows [row0, row0 + 16) of head bh, whose Q rows are in shared Qs[r0..].
PFLATLONG_DEV unsigned long long live_tiles(const Args& a, int bh, int row0, const __nv_bfloat16* qs, int lane,
                                            const float* knt = nullptr) {
  const int nt = a.t >> 6, qt = row0 >> 6;
  const int mb = a.mask_batched ? bh / a.m_heads : 0;
  const float* kn = knt ? knt : a.kn + (long long)bh * nt;
  if (a.has_mask && a.kn == nullptr && knt == nullptr) {
    // P9-A mask-only liveness (no key-norm tables): a tile is skipped only when every mask entry is -inf, where every
    // probability is exactly zero; every other tile is evaluated.
    unsigned long long bits = 0ull;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int t = lane + 32 * half;
      const bool live = t < nt && !(a.mmax[((long long)mb * nt + qt) * nt + t] == -INFINITY);
      bits |= (unsigned long long)__ballot_sync(0xffffffffu, live) << (32 * half);
    }
    return bits;
  }
  // |q_i| for row lane>>1 (two lanes per row, 32 units each, fixed order then pair sum).
  const int r = lane >> 1, h = lane & 1;
  float sq = 0.f;
#pragma unroll
  for (int c = 0; c < 32; c += 2) {
    const uint32_t w = *reinterpret_cast<const uint32_t*>(qs + r * LD + 32 * h + c);
    sq = fmaf(bf_lo(w), bf_lo(w), sq);
    sq = fmaf(bf_hi(w), bf_hi(w), sq);
  }
  const float other = __shfl_xor_sync(0xffffffffu, sq, 1);
  const float qn = sqrtf(h ? other + sq : sq + other) * scale_bound(a) + 1e-30f;
  float knall = 0.f;
  for (int t = lane; t < nt; t += 32) knall = fmaxf(knall, kn[t]);
  knall = warp_max(knall);
  float negm = -INFINITY, amax = warp_max(qn);
  if (a.has_mask) {
    float rmax = -INFINITY;
    for (int t = h; t < nt; t += 2) rmax = fmaxf(rmax, a.rm[((long long)mb * nt + t) * a.t + row0 + r]);
    rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, 1));
    // m_i >= rmax_i - |q_i| max|k| c (rounding margins included in qn, kn and the absolute 1.0).
    negm = warp_max(-(rmax - qn * knall - 1.0f));
  } else {
    negm = warp_max(qn * knall + 1.0f);
  }
  unsigned long long bits = 0ull;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int t = lane + 32 * half;
    bool live = false;
    if (t < nt) {
      const float mt = a.has_mask ? a.mmax[((long long)mb * nt + qt) * nt + t] : 0.f;
      const float bound = amax * kn[t] + mt + negm;
      live = !(bound < DEAD);  // NaN or infinite statistics keep the tile
    }
    const unsigned b = __ballot_sync(0xffffffffu, live);
    bits |= (unsigned long long)b << (32 * half);
  }
  return bits;
}

PFLATLONG_DEV bool tile_mixed(const Args& a, int bh, int qt, int kt) {
  if (!a.has_mask) return false;
  const int nt = a.t >> 6, mb = a.mask_batched ? bh / a.m_heads : 0;
  return a.mzero[((long long)mb * nt + qt) * nt + kt] == 0.f;
}

PFLATLONG_DEV const float* mask_row(const Args& a, int bh, int row) {
  return a.mask + (long long)(a.mask_batched ? bh / a.m_heads : 0) * a.m_b + (long long)row * a.m_i;
}

// Index of the (n+1)-th set bit at or after `from`; 64 when none.
PFLATLONG_DEV int next_bit(unsigned long long bits, int from) {
  if (from >= 64) return 64;
  const unsigned long long rest = bits >> from;
  return rest ? from + __ffsll((long long)rest) - 1 : 64;
}

// ---------------------------------------------------------------------------------------------------- forward
// Shared: Q [128][LD], K/V stages [2][2][64][LD] (BF16), warp live words [8].
__host__ __device__ constexpr int forward_shared_bytes() { return 2 * (BR * LD + 4 * BT * LD) + 8 * NW + 16; }

template <int NS>
PFLATLONG_DEV void forward_item(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + BR * LD;
  unsigned long long* wl = reinterpret_cast<unsigned long long*>(kvs + 4 * BT * LD);
  const int t = a.t, row0 = qb * BR;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  cp_commit();
  cp_wait0();
  __syncthreads();
  const int wrow = row0 + 16 * warp, qt = wrow >> 6;
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * warp * LD, lane);
  if (lane == 0) {
    wl[warp] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  __syncthreads();
  unsigned long long all = 0ull;
#pragma unroll
  for (int w = 0; w < NW; ++w) all |= wl[w];
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  int kt = next_bit(all, 0);
  if (kt < 64) {
    load_rows<BT>(kvs, kg + (long long)(kt * BT) * a.lk.sr, a.lk.sr);
    load_rows<BT>(kvs + BT * LD, vg + (long long)(kt * BT) * a.lv.sr, a.lv.sr);
  }
  cp_commit();
  int stage = 0;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  while (kt < 64) {
    const int nk = next_bit(all, kt + 1);
    if (nk < 64) {
      __nv_bfloat16* dst = kvs + (stage ^ 1) * 2 * BT * LD;
      load_rows<BT>(dst, kg + (long long)(nk * BT) * a.lk.sr, a.lk.sr);
      load_rows<BT>(dst + BT * LD, vg + (long long)(nk * BT) * a.lv.sr, a.lv.sr);
    }
    cp_commit();
    cp_wait1();
    __syncthreads();
    const __nv_bfloat16* ks = kvs + stage * 2 * BT * LD;
    const __nv_bfloat16* vs = ks + BT * LD;
    if ((mine >> kt) & 1ull) {
      const bool mixed = tile_mixed(a, bh, qt, kt);
      float s[8][4];
#pragma unroll
      for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
      mma_nt<8>(s, qa, ks, 0, lane);
      float x0 = -INFINITY, x1 = -INFINITY;
      const float* mlo = mixed ? mask_row(a, bh, r_lo) : nullptr;
      const float* mhi = mixed ? mask_row(a, bh, r_hi) : nullptr;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const long long col = (long long)kt * BT + 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float x = scaled(a, s[n][e]);
          if (mixed) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
          s[n][e] = x;
        }
        x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
        x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
      }
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
        x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
      }
      const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
      const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
      const float c0 = m0 == -INFINITY ? 0.f : exp2f(fmaf(m0, L2E, -ml0));
      const float c1 = m1 == -INFINITY ? 0.f : exp2f(fmaf(m1, L2E, -ml1));
      m0 = n0;
      m1 = n1;
      float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        s[n][0] = exp2f(fmaf(s[n][0], L2E, -ml0));
        s[n][1] = exp2f(fmaf(s[n][1], L2E, -ml0));
        s[n][2] = exp2f(fmaf(s[n][2], L2E, -ml1));
        s[n][3] = exp2f(fmaf(s[n][3], L2E, -ml1));
        sum0 += s[n][0] + s[n][1];
        sum1 += s[n][2] + s[n][3];
        o[n][0] *= c0;
        o[n][1] *= c0;
        o[n][2] *= c1;
        o[n][3] *= c1;
      }
      l0 = fmaf(l0, c0, sum0);
      l1 = fmaf(l1, c1, sum1);
      uint32_t pa[4][NS][4];
      to_a<NS, 4, 8>(pa, s);
      mma_pv<NS, 4>(o, pa, vs, 0, lane);
    }
    __syncthreads();  // the next iteration's copies overwrite this stage
    stage ^= 1;
    kt = nk;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
}

// --------------------------------------------------------------------------------------------- dQ (+ delta) pass
// Shared: Q, dO [128][LD], K/V stages [2][2][64][LD], warp live words [8].
__host__ __device__ constexpr int dq_shared_bytes() { return 2 * (2 * BR * LD + 4 * BT * LD) + 8 * NW + 16; }

PFLATLONG_DEV float grad_scale(const Args& a, float g) {
  return a.scale_mode == 1 ? g * a.scale : (a.scale_mode == 2 ? g / a.scale : g);
}

template <int NS>
PFLATLONG_DEV void dq_item(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* ds = qs + BR * LD;
  __nv_bfloat16* kvs = ds + BR * LD;
  unsigned long long* wl = reinterpret_cast<unsigned long long*>(kvs + 4 * BT * LD);
  const int t = a.t, row0 = qb * BR;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  load_rows<BR>(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr);
  cp_commit();
  const int wrow = row0 + 16 * warp, qt = wrow >> 6;
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  if (lane == 0) wl[warp] = mine;
  // delta_i = sum_d dO_id O_id for row wrow + (lane >> 1): two lanes per row, fixed order, pair sum.
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc = fmaf(bf_lo(ws[e]), __ldcg(op + 8 * c + 2 * e), acc);
        acc = fmaf(bf_hi(ws[e]), __ldcg(op + 8 * c + 2 * e + 1), acc);
      }
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      const uint32_t ws[4] = {w.x, w.y, w.z, w.w}, zs[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc = fmaf(bf_lo(ws[e]), bf_lo(zs[e]), acc);
        acc = fmaf(bf_hi(ws[e]), bf_hi(zs[e]), acc);
      }
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  __syncthreads();
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  a_frags(da, ds, 16 * warp, lane);
  unsigned long long all = 0ull;
#pragma unroll
  for (int w = 0; w < NW; ++w) all |= wl[w];
  float dq[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
  int kt = next_bit(all, 0);
  if (kt < 64) {
    load_rows<BT>(kvs, kg + (long long)(kt * BT) * a.lk.sr, a.lk.sr);
    load_rows<BT>(kvs + BT * LD, vg + (long long)(kt * BT) * a.lv.sr, a.lv.sr);
  }
  cp_commit();
  int stage = 0;
  while (kt < 64) {
    const int nk = next_bit(all, kt + 1);
    if (nk < 64) {
      __nv_bfloat16* dst = kvs + (stage ^ 1) * 2 * BT * LD;
      load_rows<BT>(dst, kg + (long long)(nk * BT) * a.lk.sr, a.lk.sr);
      load_rows<BT>(dst + BT * LD, vg + (long long)(nk * BT) * a.lv.sr, a.lv.sr);
    }
    cp_commit();
    cp_wait1();
    __syncthreads();
    const __nv_bfloat16* ks = kvs + stage * 2 * BT * LD;
    const __nv_bfloat16* vs = ks + BT * LD;
    if ((mine >> kt) & 1ull) {
      const bool mixed = tile_mixed(a, bh, qt, kt);
      const float* mlo = mixed ? mask_row(a, bh, r_lo) : nullptr;
      const float* mhi = mixed ? mask_row(a, bh, r_hi) : nullptr;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s[4][4], dp[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.f;
        mma_nt<4>(s, qa, ks, 4 * h, lane);
        mma_nt<4>(dp, da, vs, 4 * h, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const long long col = (long long)kt * BT + 32 * h + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float x = scaled(a, s[n][e]);
            if (mixed) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
            const float p = exp2f(fmaf(x, L2E, -(e < 2 ? s_lo : s_hi)));
            s[n][e] = grad_scale(a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
          }
        }
        uint32_t sa[2][NS][4];
        to_a<NS, 2, 4>(sa, s);
        mma_pv<NS, 2>(dq, sa, ks, 32 * h, lane);
      }
    }
    __syncthreads();
    stage ^= 1;
    kt = nk;
  }
  cp_wait0();
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
}

// ---------------------------------------------------------------------------------------------- dK / dV pass
// Shared: K, V [128][LD]; stages [2] x (Q [64][LD], dO [64][LD]) BF16 + (lse2 [64], delta [64]) FP32; live words.
__host__ __device__ constexpr int kv_shared_bytes() {
  return 2 * (2 * BR * LD) + 2 * (2 * 2 * BT * LD + 4 * 2 * BT) + 2 * 8 * 2 + 16;
}

template <int NS>
PFLATLONG_DEV void kv_item(const Args& a, int bh, int kb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* vs = ks + BR * LD;
  char* stages = reinterpret_cast<char*>(vs + BR * LD);
  constexpr int STAGE = 2 * 2 * BT * LD + 4 * 2 * BT;
  unsigned long long* wl = reinterpret_cast<unsigned long long*>(stages + 2 * STAGE);
  const int t = a.t, nt = t >> 6, key0 = kb * BR;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows<BR>(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr);
  load_rows<BR>(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr);
  cp_commit();
  // Query tiles with a live 16-row group for key tiles 2kb (warps 0-3) and 2kb + 1 (warps 4-7).
  if (warp < 2) {
    const int kt = 2 * kb + warp;
    unsigned long long bits = 0ull;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int qt = lane + 32 * half;
      bool live = false;
      if (qt < nt) {
#pragma unroll
        for (int g = 0; g < 4; ++g) live = live || ((a.live[(long long)bh * (t >> 4) + 4 * qt + g] >> kt) & 1ull);
      }
      bits |= (unsigned long long)__ballot_sync(0xffffffffu, live) << (32 * half);
    }
    if (lane == 0) wl[warp] = bits;
  }
  cp_wait0();
  __syncthreads();
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * warp, lane);
  a_frags(va, vs, 16 * warp, lane);
  const unsigned long long mine = wl[warp >> 2], all = wl[0] | wl[1];
  const int ktw = 2 * kb + (warp >> 2);
  const int k_lo = key0 + 16 * warp + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
    dk[n][0] = dk[n][1] = dk[n][2] = dk[n][3] = dv[n][0] = dv[n][1] = dv[n][2] = dv[n][3] = 0.f;
  auto load_stage = [&](int qt, int stage) {
    char* base = stages + stage * STAGE;
    __nv_bfloat16* qd = reinterpret_cast<__nv_bfloat16*>(base);
    load_rows<BT>(qd, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr);
    load_rows<BT>(qd + BT * LD, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr);
    float* sl = reinterpret_cast<float*>(qd + 2 * BT * LD);
    if (threadIdx.x < 16) cp16(sl + 4 * threadIdx.x, a.lse2 + (long long)bh * t + qt * BT + 4 * threadIdx.x);
    else if (threadIdx.x < 32)
      cp16(sl + BT + 4 * (threadIdx.x - 16), a.delta + (long long)bh * t + qt * BT + 4 * (threadIdx.x - 16));
  };
  int qt = next_bit(all, 0);
  if (qt < 64) load_stage(qt, 0);
  cp_commit();
  int stage = 0;
  __syncthreads();  // K/V shared rows are dead after the fragment loads; stages do not alias them
  while (qt < 64) {
    const int nq = next_bit(all, qt + 1);
    if (nq < 64) load_stage(nq, stage ^ 1);
    cp_commit();
    cp_wait1();
    __syncthreads();
    const __nv_bfloat16* qd = reinterpret_cast<const __nv_bfloat16*>(stages + stage * STAGE);
    const __nv_bfloat16* dd = qd + BT * LD;
    const float* sl = reinterpret_cast<const float*>(qd + 2 * BT * LD);
    const float* dl = sl + BT;
    if ((mine >> qt) & 1ull) {
      const bool mixed = tile_mixed(a, bh, qt, ktw);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s[4][4], dp[4][4];
#pragma unroll
        for (int n = 0; n < 4; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.f;
        mma_nt<4>(s, ka, qd, 4 * h, lane);
        mma_nt<4>(dp, va, dd, 4 * h, lane);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int j = 32 * h + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int jj = j + (e & 1);
            float x = scaled(a, s[n][e]);
            if (mixed) x += __ldg(mask_row(a, bh, qt * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
            const float p = exp2f(fmaf(x, L2E, -sl[jj]));
            s[n][e] = p;
            dp[n][e] = grad_scale(a, p * (dp[n][e] - dl[jj]));
          }
        }
        uint32_t pa[2][NS][4];
        to_a<NS, 2, 4>(pa, s);
        mma_pv<NS, 2>(dv, pa, dd, 32 * h, lane);
        to_a<NS, 2, 4>(pa, dp);
        mma_pv<NS, 2>(dk, pa, qd, 32 * h, lane);
      }
    }
    __syncthreads();
    stage ^= 1;
    qt = nq;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
}

// ====================================================================================================== v2 schedule
// Same arithmetic and outputs as v1. Each warp owns two m16 row tiles (32 rows or keys), so every B fragment loaded
// from shared memory feeds two products; the eight warps form four row groups x two halves of the opposite axis
// (split-K in the forward/dQ items, split-Q in the dK/dV items) whose partial results combine once, in a fixed order,
// through shared memory. Live/mixed tile bits are computed once per item; operand tiles stream through a
// three-stage (forward, dQ) or two-stage (dK/dV) cp.async pipeline, two 64-wide tiles per stage.

// Mixed (mask read needed) bits over the 64-key tiles of query tile qt; lanes cover tiles, one ballot per half.
PFLATLONG_DEV unsigned long long mixed_rows(const Args& a, int bh, int qt, int lane) {
  if (!a.has_mask) return 0ull;
  const int nt = a.t >> 6, mb = a.mask_batched ? bh / a.m_heads : 0;
  unsigned long long bits = 0ull;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int t = lane + 32 * half;
    const bool m = t < nt && a.mzero[((long long)mb * nt + qt) * nt + t] == 0.f;
    bits |= (unsigned long long)__ballot_sync(0xffffffffu, m) << (32 * half);
  }
  return bits;
}
// Mixed bits over query tiles for one key tile kt (dK/dV items).
PFLATLONG_DEV unsigned long long mixed_cols(const Args& a, int bh, int kt, int lane) {
  if (!a.has_mask) return 0ull;
  const int nt = a.t >> 6, mb = a.mask_batched ? bh / a.m_heads : 0;
  unsigned long long bits = 0ull;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int q = lane + 32 * half;
    const bool m = q < nt && a.mzero[((long long)mb * nt + q) * nt + kt] == 0.f;
    bits |= (unsigned long long)__ballot_sync(0xffffffffu, m) << (32 * half);
  }
  return bits;
}
// Query tiles with a live 16-row group for key tile kt (forward proof table).
PFLATLONG_DEV unsigned long long live_cols(const Args& a, int bh, int kt, int lane) {
  const int nt = a.t >> 6;
  unsigned long long bits = 0ull;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int qt = lane + 32 * half;
    bool live = false;
    if (qt < nt) {
#pragma unroll
      for (int g = 0; g < 4; ++g) live = live || ((a.live[(long long)bh * (a.t >> 4) + 4 * qt + g] >> kt) & 1ull);
    }
    bits |= (unsigned long long)__ballot_sync(0xffffffffu, live) << (32 * half);
  }
  return bits;
}

// c0/c1 (n-tiles [n0, n0 + NN)) += A0/A1 (16 x 64 each) * B, B[k][n] = S[n][k]; one B load feeds both row tiles.
template <int NN>
PFLATLONG_DEV void mma_nt_2m(float (&c0)[NN][4], float (&c1)[NN][4], const uint32_t (&a0)[4][4],
                         const uint32_t (&a1)[4][4], const __nv_bfloat16* s, int n0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
    for (int p = 0; p < NN / 2; ++p) {
      uint32_t b[4];
      ldsm_x4(b, s + ((n0 + 2 * p + (lane >> 4)) * 8 + (lane & 7)) * LD + 16 * ks + ((lane >> 3) & 1) * 8);
      mma(c0[2 * p], a0[ks], b[0], b[1]);
      mma(c0[2 * p + 1], a0[ks], b[2], b[3]);
      mma(c1[2 * p], a1[ks], b[0], b[1]);
      mma(c1[2 * p + 1], a1[ks], b[2], b[3]);
    }
  }
}
// o0/o1 (8 n-tiles over 64 units) += P0/P1 (16 x 16*KK pieces) * S rows [r0, r0 + 16*KK) (transposed load).
template <int NS, int KK>
PFLATLONG_DEV void mma_pv_2m(float (&o0)[8][4], float (&o1)[8][4], const uint32_t (&p0)[KK][NS][4],
                         const uint32_t (&p1)[KK][NS][4], const __nv_bfloat16* s, int r0, int lane) {
#pragma unroll
  for (int kk = 0; kk < KK; ++kk) {
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      uint32_t b[4];
      ldsm_x4_t(b, s + (r0 + 16 * kk + ((lane >> 3) & 1) * 8 + (lane & 7)) * LD + 8 * (2 * p + (lane >> 4)));
#pragma unroll
      for (int piece = 0; piece < NS; ++piece) {
        mma(o0[2 * p], p0[kk][piece], b[0], b[1]);
        mma(o0[2 * p + 1], p0[kk][piece], b[2], b[3]);
        mma(o1[2 * p], p1[kk][piece], b[0], b[1]);
        mma(o1[2 * p + 1], p1[kk][piece], b[2], b[3]);
      }
    }
  }
}

// Online-softmax update of one m16 x 64 score tile (rows r_lo, r_hi; key columns from col0) into (m, l, o).
PFLATLONG_DEV void softmax_tile(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                            float (&o)[8][4], const float* mlo, const float* mhi, long long col0, int lane) {
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = scaled(a, s[n][e]);
      if (mlo) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
      s[n][e] = x;
    }
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  const float c0 = m0 == -INFINITY ? 0.f : exp2f(fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : exp2f(fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = exp2f(fmaf(s[n][0], L2E, -ml0));
    s[n][1] = exp2f(fmaf(s[n][1], L2E, -ml0));
    s[n][2] = exp2f(fmaf(s[n][2], L2E, -ml1));
    s[n][3] = exp2f(fmaf(s[n][3], L2E, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
    o[n][0] *= c0;
    o[n][1] *= c0;
    o[n][2] *= c1;
    o[n][3] *= c1;
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

// Pairs of live 64-wide tiles in increasing order: (first, second) starting at bit `from`.
PFLATLONG_DEV void next_pair(unsigned long long bits, int from, int& first, int& second) {
  first = next_bit(bits, from);
  second = next_bit(bits, first + 1);
}

// Stage of two 64-row operand tiles: rows [0, 64) tile ta, [64, 128) tile tb of X, then the same of Y at +128 rows.
PFLATLONG_DEV void load_pair(__nv_bfloat16* d, const __nv_bfloat16* xg, long long xs, const __nv_bfloat16* yg,
                         long long ys, int ta, int tb) {
  if (ta < 64) {
    load_rows<BT>(d, xg + (long long)(ta * BT) * xs, xs);
    load_rows<BT>(d + 2 * BT * LD, yg + (long long)(ta * BT) * ys, ys);
  }
  if (tb < 64) {
    load_rows<BT>(d + BT * LD, xg + (long long)(tb * BT) * xs, xs);
    load_rows<BT>(d + 3 * BT * LD, yg + (long long)(tb * BT) * ys, ys);
  }
}

constexpr int PAIR = 4 * BT * LD;  // BF16 elements of one two-tile stage of two operands
__host__ __device__ constexpr int forward2_shared_bytes() { return 2 * 3 * PAIR + 16 * 8; }
__host__ __device__ constexpr int dq2_shared_bytes() { return 2 * 3 * PAIR + 16 * 8; }
__host__ __device__ constexpr int kv2_shared_bytes() { return 2 * 2 * BR * LD + 2 * (2 * PAIR + 4 * 2 * 2 * BT) + 16 * 8; }

template <int NS>
PFLATLONG_DEV void forward_item2(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rg = warp & 3, kh = warp >> 2;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(st + 3 * PAIR);
  const int t = a.t, row0 = qb * BR, r0 = 32 * rg, wrow = row0 + r0, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  __nv_bfloat16* qs = st + 2 * PAIR;  // Q staging aliases the third stage
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  cp_commit();
  cp_wait0();
  __syncthreads();
  if (kh == 0) {
    const unsigned long long l0 = live_tiles(a, bh, wrow, qs + r0 * LD, lane);
    const unsigned long long l1 = live_tiles(a, bh, wrow + 16, qs + (r0 + 16) * LD, lane);
    const unsigned long long mx = mixed_rows(a, bh, qt, lane);
    if (lane == 0) {
      a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = l0;
      a.live[(long long)bh * (t >> 4) + (wrow >> 4) + 1] = l1;
      flags[rg] = l0 | l1;
      flags[4 + rg] = mx;
    }
  }
  uint32_t qa0[4][4], qa1[4][4];
  a_frags(qa0, qs, r0, lane);
  a_frags(qa1, qs, r0 + 16, lane);
  __syncthreads();
  const unsigned long long mine = flags[rg], mixed = flags[4 + rg];
  const unsigned long long all = flags[0] | flags[1] | flags[2] | flags[3];
  float o0[8][4], o1[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    o0[n][0] = o0[n][1] = o0[n][2] = o0[n][3] = 0.f;
    o1[n][0] = o1[n][1] = o1[n][2] = o1[n][3] = 0.f;
  }
  float m[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY}, l[4] = {0.f, 0.f, 0.f, 0.f};
  const int r_lo = wrow + (lane >> 2);  // rows r_lo, r_lo + 8 (tile 0) and r_lo + 16, r_lo + 24 (tile 1)
  int ca, cb, na, nb;
  next_pair(all, 0, ca, cb);
  next_pair(all, cb + 1, na, nb);
  load_pair(st, kg, a.lk.sr, vg, a.lv.sr, ca, cb);
  cp_commit();
  load_pair(st + PAIR, kg, a.lk.sr, vg, a.lv.sr, na, nb);
  cp_commit();
  int stage = 0;
  while (ca < 64) {
    int fa, fb;
    next_pair(all, nb + 1, fa, fb);
    cp_wait1();
    __syncthreads();
    load_pair(st + ((stage + 2) % 3) * PAIR, kg, a.lk.sr, vg, a.lv.sr, fa, fb);
    cp_commit();
    const int kt = kh ? cb : ca;
    if (kt < 64 && ((mine >> kt) & 1ull)) {
      const __nv_bfloat16* ks = st + stage * PAIR + kh * BT * LD;
      const __nv_bfloat16* vs = ks + 2 * BT * LD;
      const bool mx = (mixed >> kt) & 1ull;
      float s0[8][4], s1[8][4];
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        s0[n][0] = s0[n][1] = s0[n][2] = s0[n][3] = 0.f;
        s1[n][0] = s1[n][1] = s1[n][2] = s1[n][3] = 0.f;
      }
      mma_nt_2m<8>(s0, s1, qa0, qa1, ks, 0, lane);
      const long long col0 = (long long)kt * BT;
      softmax_tile(a, s0, m[0], m[1], l[0], l[1], o0, mx ? mask_row(a, bh, r_lo) : nullptr,
                   mx ? mask_row(a, bh, r_lo + 8) : nullptr, col0, lane);
      softmax_tile(a, s1, m[2], m[3], l[2], l[3], o1, mx ? mask_row(a, bh, r_lo + 16) : nullptr,
                   mx ? mask_row(a, bh, r_lo + 24) : nullptr, col0, lane);
      uint32_t p0[4][NS][4], p1[4][NS][4];
      to_a<NS, 4, 8>(p0, s0);
      to_a<NS, 4, 8>(p1, s1);
      mma_pv_2m<NS, 4>(o0, o1, p0, p1, vs, 0, lane);
    }
    ca = na;
    cb = nb;
    na = fa;
    nb = fb;
    stage = stage == 2 ? 0 : stage + 1;
  }
  cp_wait0();
  __syncthreads();  // every warp is done with the stages; reuse them for the split-K exchange
  float* cmb = reinterpret_cast<float*>(smem) + (rg * 32 + lane) * 72;
  if (kh == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      *reinterpret_cast<float4*>(cmb + 4 * n) = make_float4(o0[n][0], o0[n][1], o0[n][2], o0[n][3]);
      *reinterpret_cast<float4*>(cmb + 32 + 4 * n) = make_float4(o1[n][0], o1[n][1], o1[n][2], o1[n][3]);
    }
    *reinterpret_cast<float4*>(cmb + 64) = make_float4(m[0], m[1], m[2], m[3]);
    *reinterpret_cast<float4*>(cmb + 68) = make_float4(l[0], l[1], l[2], l[3]);
  }
  __syncthreads();
  if (kh == 1) return;
  const float4 pm = *reinterpret_cast<const float4*>(cmb + 64), pl = *reinterpret_cast<const float4*>(cmb + 68);
  const float om[4] = {pm.x, pm.y, pm.z, pm.w}, ol[4] = {pl.x, pl.y, pl.z, pl.w};
  float ca4[4], cb4[4], lse[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float mn = fmaxf(m[r], om[r]);
    const float ml = mn == -INFINITY ? 0.f : mn * L2E;
    ca4[r] = m[r] == -INFINITY ? 0.f : exp2f(fmaf(m[r], L2E, -ml));
    cb4[r] = om[r] == -INFINITY ? 0.f : exp2f(fmaf(om[r], L2E, -ml));
    float lt = l[r] * ca4[r] + ol[r] * cb4[r];
    lt += __shfl_xor_sync(0xffffffffu, lt, 1);
    lt += __shfl_xor_sync(0xffffffffu, lt, 2);
    l[r] = lt;
    lse[r] = ml + log2f(lt);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float4 q0 = *reinterpret_cast<const float4*>(cmb + 4 * n), q1 = *reinterpret_cast<const float4*>(cmb + 32 + 4 * n);
    const float y[8] = {(o0[n][0] * ca4[0] + q0.x * cb4[0]) / l[0], (o0[n][1] * ca4[0] + q0.y * cb4[0]) / l[0],
                        (o0[n][2] * ca4[1] + q0.z * cb4[1]) / l[1], (o0[n][3] * ca4[1] + q0.w * cb4[1]) / l[1],
                        (o1[n][0] * ca4[2] + q1.x * cb4[2]) / l[2], (o1[n][1] * ca4[2] + q1.y * cb4[2]) / l[2],
                        (o1[n][2] * ca4[3] + q1.z * cb4[3]) / l[3], (o1[n][3] * ca4[3] + q1.w * cb4[3]) / l[3]};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int row = r_lo + 8 * r;
      store_bf16x2(og + (long long)row * a.lo.sr + d, y[2 * r], y[2 * r + 1]);
#ifdef PFLATLONG_O32
      *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + row) * HD + d) = make_float2(y[2 * r], y[2 * r + 1]);
#endif
    }
  }
  if ((lane & 3) == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) a.lse2[(long long)bh * t + r_lo + 8 * r] = lse[r];
  }
}

template <int NS>
PFLATLONG_DEV void dq_item2(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, rg = warp & 3, kh = warp >> 2;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(st + 3 * PAIR);
  const int t = a.t, row0 = qb * BR, r0 = 32 * rg, wrow = row0 + r0, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  __nv_bfloat16* qs = st + 2 * PAIR;
  __nv_bfloat16* ds = qs + BR * LD;
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  load_rows<BR>(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr);
  cp_commit();
  if (kh == 0 && lane == 0) {
    flags[rg] = a.live[(long long)bh * (t >> 4) + (wrow >> 4)] | a.live[(long long)bh * (t >> 4) + (wrow >> 4) + 1];
  }
  if (kh == 0) {
    const unsigned long long mx = mixed_rows(a, bh, qt, lane);
    if (lane == 0) flags[4 + rg] = mx;
  }
  // delta for row wrow + lane: 64 units in fixed order.
  float dl = 0.f;
  {
    const int r = wrow + lane;
    const uint4* dp = reinterpret_cast<const uint4*>(a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr);
#ifdef PFLATLONG_O32
    const float4* op = reinterpret_cast<const float4*>(a.o32 + ((long long)bh * t + r) * HD);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 w = __ldcg(dp + c);
      const float4 z0 = __ldcg(op + 2 * c), z1 = __ldcg(op + 2 * c + 1);
      dl = fmaf(bf_lo(w.x), z0.x, dl); dl = fmaf(bf_hi(w.x), z0.y, dl);
      dl = fmaf(bf_lo(w.y), z0.z, dl); dl = fmaf(bf_hi(w.y), z0.w, dl);
      dl = fmaf(bf_lo(w.z), z1.x, dl); dl = fmaf(bf_hi(w.z), z1.y, dl);
      dl = fmaf(bf_lo(w.w), z1.z, dl); dl = fmaf(bf_hi(w.w), z1.w, dl);
    }
#else
    const uint4* op = reinterpret_cast<const uint4*>(a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr);
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint4 w = __ldcg(dp + c), z = __ldcg(op + c);
      dl = fmaf(bf_lo(w.x), bf_lo(z.x), dl); dl = fmaf(bf_hi(w.x), bf_hi(z.x), dl);
      dl = fmaf(bf_lo(w.y), bf_lo(z.y), dl); dl = fmaf(bf_hi(w.y), bf_hi(z.y), dl);
      dl = fmaf(bf_lo(w.z), bf_lo(z.z), dl); dl = fmaf(bf_hi(w.z), bf_hi(z.z), dl);
      dl = fmaf(bf_lo(w.w), bf_lo(z.w), dl); dl = fmaf(bf_hi(w.w), bf_hi(z.w), dl);
    }
#endif
    if (kh == 0) a.delta[(long long)bh * t + r] = dl;
  }
  float dlr[4], lsr[4];
  const int r_lo = wrow + (lane >> 2);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    dlr[r] = __shfl_sync(0xffffffffu, dl, (lane >> 2) + 8 * r);
    lsr[r] = a.lse2[(long long)bh * t + r_lo + 8 * r];
  }
  cp_wait0();
  __syncthreads();
  uint32_t qa0[4][4], qa1[4][4], da0[4][4], da1[4][4];
  a_frags(qa0, qs, r0, lane);
  a_frags(qa1, qs, r0 + 16, lane);
  a_frags(da0, ds, r0, lane);
  a_frags(da1, ds, r0 + 16, lane);
  __syncthreads();
  const unsigned long long mine = flags[rg], mixed = flags[4 + rg];
  const unsigned long long all = flags[0] | flags[1] | flags[2] | flags[3];
  float dq0[8][4], dq1[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    dq0[n][0] = dq0[n][1] = dq0[n][2] = dq0[n][3] = 0.f;
    dq1[n][0] = dq1[n][1] = dq1[n][2] = dq1[n][3] = 0.f;
  }
  int ca, cb, na, nb;
  next_pair(all, 0, ca, cb);
  next_pair(all, cb + 1, na, nb);
  load_pair(st, kg, a.lk.sr, vg, a.lv.sr, ca, cb);
  cp_commit();
  load_pair(st + PAIR, kg, a.lk.sr, vg, a.lv.sr, na, nb);
  cp_commit();
  int stage = 0;
  while (ca < 64) {
    int fa, fb;
    next_pair(all, nb + 1, fa, fb);
    cp_wait1();
    __syncthreads();
    load_pair(st + ((stage + 2) % 3) * PAIR, kg, a.lk.sr, vg, a.lv.sr, fa, fb);
    cp_commit();
    const int kt = kh ? cb : ca;
    if (kt < 64 && ((mine >> kt) & 1ull)) {
      const __nv_bfloat16* ks = st + stage * PAIR + kh * BT * LD;
      const __nv_bfloat16* vs = ks + 2 * BT * LD;
      const bool mx = (mixed >> kt) & 1ull;
      const float* mrow[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) mrow[r] = mx ? mask_row(a, bh, r_lo + 8 * r) : nullptr;
#pragma unroll
      for (int c = 0; c < 4; ++c) {  // 16-key chunks
        float s[2][2][4], dp[2][2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[j][n][e] = dp[j][n][e] = 0.f;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
          uint32_t bk[4], bv[4];
          const int off = (16 * c + (lane >> 4) * 8 + (lane & 7)) * LD + 16 * k16 + ((lane >> 3) & 1) * 8;
          ldsm_x4(bk, ks + off);
          ldsm_x4(bv, vs + off);
          mma(s[0][0], qa0[k16], bk[0], bk[1]);
          mma(s[0][1], qa0[k16], bk[2], bk[3]);
          mma(s[1][0], qa1[k16], bk[0], bk[1]);
          mma(s[1][1], qa1[k16], bk[2], bk[3]);
          mma(dp[0][0], da0[k16], bv[0], bv[1]);
          mma(dp[0][1], da0[k16], bv[2], bv[3]);
          mma(dp[1][0], da1[k16], bv[0], bv[1]);
          mma(dp[1][1], da1[k16], bv[2], bv[3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const long long col = (long long)kt * BT + 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int r = 2 * j + (e >> 1);
              float x = scaled(a, s[j][n][e]);
              if (mx) x += __ldg(mrow[r] + (col + (e & 1)) * a.m_j);
              const float p = exp2f(fmaf(x, L2E, -lsr[r]));
              s[j][n][e] = grad_scale(a, p * (dp[j][n][e] - dlr[r]));
            }
          }
        uint32_t sa0[1][NS][4], sa1[1][NS][4];
        to_a<NS, 1, 2>(sa0, s[0]);
        to_a<NS, 1, 2>(sa1, s[1]);
        mma_pv_2m<NS, 1>(dq0, dq1, sa0, sa1, ks, 16 * c, lane);
      }
    }
    ca = na;
    cb = nb;
    na = fa;
    nb = fb;
    stage = stage == 2 ? 0 : stage + 1;
  }
  cp_wait0();
  __syncthreads();
  float* cmb = reinterpret_cast<float*>(smem) + (rg * 32 + lane) * 64;
  if (kh == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      *reinterpret_cast<float4*>(cmb + 4 * n) = make_float4(dq0[n][0], dq0[n][1], dq0[n][2], dq0[n][3]);
      *reinterpret_cast<float4*>(cmb + 32 + 4 * n) = make_float4(dq1[n][0], dq1[n][1], dq1[n][2], dq1[n][3]);
    }
  }
  __syncthreads();
  if (kh == 1) return;
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float4 q0 = *reinterpret_cast<const float4*>(cmb + 4 * n), q1 = *reinterpret_cast<const float4*>(cmb + 32 + 4 * n);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq0[n][0] + q0.x, dq0[n][1] + q0.y);
    store_bf16x2(qg + (long long)(r_lo + 8) * a.ldq.sr + d, dq0[n][2] + q0.z, dq0[n][3] + q0.w);
    store_bf16x2(qg + (long long)(r_lo + 16) * a.ldq.sr + d, dq1[n][0] + q1.x, dq1[n][1] + q1.y);
    store_bf16x2(qg + (long long)(r_lo + 24) * a.ldq.sr + d, dq1[n][2] + q1.z, dq1[n][3] + q1.w);
  }
}

// dK/dV: warp (kg = warp & 3, qh = warp >> 2) owns keys [32 kg, 32 kg + 32) of the 128-key block and the qh-th query
// tile of every two-tile stage. K fragments stay in registers; V fragments reload per 16-query chunk.
template <int NS>
PFLATLONG_DEV void kv_item2(const Args& a, int bh, int kb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, kgp = warp & 3, qh = warp >> 2;
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(smem);   // K [128][LD]
  __nv_bfloat16* vs = ks + BR * LD;                              // V [128][LD]
  char* stages = reinterpret_cast<char*>(vs + BR * LD);
  constexpr int STAGE = 2 * PAIR + 4 * 2 * 2 * BT;               // bytes: Q/dO pair, then lse2[128], delta[128]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(stages + 2 * STAGE);
  const int t = a.t, key0 = kb * BR, k0 = 32 * kgp, kt = 2 * kb + (kgp >> 1);
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows<BR>(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr);
  load_rows<BR>(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr);
  cp_commit();
  if (warp < 2) {
    const unsigned long long lv = live_cols(a, bh, 2 * kb + warp, lane);
    const unsigned long long mx = mixed_cols(a, bh, 2 * kb + warp, lane);
    if (lane == 0) {
      flags[warp] = lv;
      flags[2 + warp] = mx;
    }
  }
  cp_wait0();
  __syncthreads();
  uint32_t ka0[4][4], ka1[4][4];
  a_frags(ka0, ks, k0, lane);
  a_frags(ka1, ks, k0 + 16, lane);
  const unsigned long long mine = flags[kgp >> 1], mixed = flags[2 + (kgp >> 1)], all = flags[0] | flags[1];
  const int kr = key0 + k0 + (lane >> 2);  // keys kr, kr + 8 (tile 0), kr + 16, kr + 24 (tile 1)
  float dk0[8][4], dk1[8][4], dv0[8][4], dv1[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk0[n][e] = dk1[n][e] = dv0[n][e] = dv1[n][e] = 0.f;
  auto issue = [&](int qa, int qb2, int stage) {
    char* base = stages + stage * STAGE;
    __nv_bfloat16* d = reinterpret_cast<__nv_bfloat16*>(base);
    load_pair(d, qg, a.lq.sr, dg, a.ld.sr, qa, qb2);
    float* sl = reinterpret_cast<float*>(d + PAIR);
    const int tq = threadIdx.x & 31, which = threadIdx.x >> 5;  // warps 0-3: lse2/delta of qa then qb2
    if (which < 4) {
      const int tile = which & 1 ? qb2 : qa;
      if (tile < 64 && tq < 16) {
        const float* src = (which < 2 ? a.lse2 : a.delta) + (long long)bh * t + tile * BT + 4 * tq;
        cp16(sl + (which < 2 ? 0 : 2 * BT) + (which & 1) * BT + 4 * tq, src);
      }
    }
  };
  int ca, cb, na, nb;
  next_pair(all, 0, ca, cb);
  issue(ca, cb, 0);
  cp_commit();
  int stage = 0;
  while (ca < 64) {
    next_pair(all, cb + 1, na, nb);
    if (na < 64) issue(na, nb, stage ^ 1);
    cp_commit();
    cp_wait1();
    __syncthreads();
    const int qt = qh ? cb : ca;
    if (qt < 64 && ((mine >> qt) & 1ull)) {
      const __nv_bfloat16* qd = reinterpret_cast<const __nv_bfloat16*>(stages + stage * STAGE) + qh * BT * LD;
      const __nv_bfloat16* dd = qd + 2 * BT * LD;
      const float* sl = reinterpret_cast<const float*>(reinterpret_cast<const __nv_bfloat16*>(stages + stage * STAGE) + PAIR)
                        + qh * BT;
      const float* dl = sl + 2 * BT;
      const bool mx = (mixed >> qt) & 1ull;
#pragma unroll
      for (int c = 0; c < 4; ++c) {  // 16-query chunks
        float s[2][2][4], dp[2][2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int n = 0; n < 2; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[j][n][e] = dp[j][n][e] = 0.f;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
          uint32_t bq[4], bo[4], va0[4], va1[4];
          const int off = (16 * c + (lane >> 4) * 8 + (lane & 7)) * LD + 16 * k16 + ((lane >> 3) & 1) * 8;
          ldsm_x4(bq, qd + off);
          ldsm_x4(bo, dd + off);
          ldsm_x4(va0, vs + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + 16 * k16 + (lane >> 4) * 8);
          ldsm_x4(va1, vs + (k0 + 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + 16 * k16 + (lane >> 4) * 8);
          mma(s[0][0], ka0[k16], bq[0], bq[1]);
          mma(s[0][1], ka0[k16], bq[2], bq[3]);
          mma(s[1][0], ka1[k16], bq[0], bq[1]);
          mma(s[1][1], ka1[k16], bq[2], bq[3]);
          mma(dp[0][0], va0, bo[0], bo[1]);
          mma(dp[0][1], va0, bo[2], bo[3]);
          mma(dp[1][0], va1, bo[0], bo[1]);
          mma(dp[1][1], va1, bo[2], bo[3]);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const int jq = 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int qq = jq + (e & 1), key = kr + 16 * j + 8 * (e >> 1);
              float x = scaled(a, s[j][n][e]);
              if (mx) x += __ldg(mask_row(a, bh, qt * BT + qq) + (long long)key * a.m_j);
              const float p = exp2f(fmaf(x, L2E, -sl[qq]));
              s[j][n][e] = p;
              dp[j][n][e] = grad_scale(a, p * (dp[j][n][e] - dl[qq]));
            }
          }
        uint32_t pa0[1][NS][4], pa1[1][NS][4];
        to_a<NS, 1, 2>(pa0, s[0]);
        to_a<NS, 1, 2>(pa1, s[1]);
        mma_pv_2m<NS, 1>(dv0, dv1, pa0, pa1, dd, 16 * c, lane);
        to_a<NS, 1, 2>(pa0, dp[0]);
        to_a<NS, 1, 2>(pa1, dp[1]);
        mma_pv_2m<NS, 1>(dk0, dk1, pa0, pa1, qd, 16 * c, lane);
      }
    }
    __syncthreads();  // the next iteration's copies overwrite this stage
    stage ^= 1;
    ca = na;
    cb = nb;
  }
  cp_wait0();
  float* cmb = reinterpret_cast<float*>(stages) + (kgp * 32 + lane) * 128;
  if (qh == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      *reinterpret_cast<float4*>(cmb + 4 * n) = make_float4(dk0[n][0], dk0[n][1], dk0[n][2], dk0[n][3]);
      *reinterpret_cast<float4*>(cmb + 32 + 4 * n) = make_float4(dk1[n][0], dk1[n][1], dk1[n][2], dk1[n][3]);
      *reinterpret_cast<float4*>(cmb + 64 + 4 * n) = make_float4(dv0[n][0], dv0[n][1], dv0[n][2], dv0[n][3]);
      *reinterpret_cast<float4*>(cmb + 96 + 4 * n) = make_float4(dv1[n][0], dv1[n][1], dv1[n][2], dv1[n][3]);
    }
  }
  __syncthreads();
  if (qh == 1) return;
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float4 k0v = *reinterpret_cast<const float4*>(cmb + 4 * n), k1v = *reinterpret_cast<const float4*>(cmb + 32 + 4 * n);
    const float4 v0v = *reinterpret_cast<const float4*>(cmb + 64 + 4 * n), v1v = *reinterpret_cast<const float4*>(cmb + 96 + 4 * n);
    store_bf16x2(kgo + (long long)kr * a.ldk.sr + d, dk0[n][0] + k0v.x, dk0[n][1] + k0v.y);
    store_bf16x2(kgo + (long long)(kr + 8) * a.ldk.sr + d, dk0[n][2] + k0v.z, dk0[n][3] + k0v.w);
    store_bf16x2(kgo + (long long)(kr + 16) * a.ldk.sr + d, dk1[n][0] + k1v.x, dk1[n][1] + k1v.y);
    store_bf16x2(kgo + (long long)(kr + 24) * a.ldk.sr + d, dk1[n][2] + k1v.z, dk1[n][3] + k1v.w);
    store_bf16x2(vgo + (long long)kr * a.ldv.sr + d, dv0[n][0] + v0v.x, dv0[n][1] + v0v.y);
    store_bf16x2(vgo + (long long)(kr + 8) * a.ldv.sr + d, dv0[n][2] + v0v.z, dv0[n][3] + v0v.w);
    store_bf16x2(vgo + (long long)(kr + 16) * a.ldv.sr + d, dv1[n][0] + v1v.x, dv1[n][1] + v1v.y);
    store_bf16x2(vgo + (long long)(kr + 24) * a.ldv.sr + d, dv1[n][2] + v1v.z, dv1[n][3] + v1v.w);
  }
}

// ====================================================================================================== v3 schedule
// Same arithmetic and outputs as v1 (one m16 row tile per warp, 128-row/128-key items, 64-wide tiles). Differences:
// live/mixed tile bits are computed once per item; tiles stream through a three-stage cp.async ring with one CTA
// barrier per tile; each warp issues the tensor-core products of the next tile (forward) or next 16-wide chunk
// (backward) before the softmax/elementwise work of the current one, so that work overlaps tensor execution.
constexpr int KV1 = 2 * BT * LD;  // BF16 elements of one stage holding two 64-row operand tiles
__host__ __device__ constexpr int forward3_shared_bytes() { return 2 * 3 * KV1 + 16 * 8; }
__host__ __device__ constexpr int dq3_shared_bytes() { return 2 * 3 * KV1 + 16 * 8; }
__host__ __device__ constexpr int kv3_shared_bytes() { return 2 * 3 * KV1 + 3 * 4 * 2 * BT + 16 * 8; }

PFLATLONG_DEV void load_two(__nv_bfloat16* d, const __nv_bfloat16* xg, long long xs, const __nv_bfloat16* yg,
                        long long ys, int tile) {
  if (tile < 64) {
    load_rows<BT>(d, xg + (long long)(tile * BT) * xs, xs);
    load_rows<BT>(d + BT * LD, yg + (long long)(tile * BT) * ys, ys);
  }
}

template <int NS>
PFLATLONG_DEV void forward_item3(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(st + 3 * KV1);
  const int t = a.t, row0 = qb * BR, wrow = row0 + 16 * warp, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  __nv_bfloat16* qs = st + KV1;  // Q staging (128 rows) aliases stages 1 and 2
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  cp_commit();
  cp_wait0();
  __syncthreads();
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * warp * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[warp] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  __syncthreads();
  unsigned long long all = 0ull;
#pragma unroll
  for (int w = 0; w < NW; ++w) all |= flags[w];
  float o[8][4], sc[8][4], sn[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[n][e] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  int t0 = next_bit(all, 0), t1 = next_bit(all, t0 + 1);
  load_two(st, kg, a.lk.sr, vg, a.lv.sr, t0);
  cp_commit();
  load_two(st + KV1, kg, a.lk.sr, vg, a.lv.sr, t1);
  cp_commit();
  int stage = 0;
  // S of the first tile.
  cp_wait1();
  __syncthreads();
  bool have = t0 < 64 && ((mine >> t0) & 1ull);
  if (have) {
#pragma unroll
    for (int n = 0; n < 8; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
    mma_nt<8>(sc, qa, st, 0, lane);
  }
  int cur = t0, nxt = t1;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait0();
    __syncthreads();  // tile nxt has landed; stage (stage + 2) % 3 is no longer read
    load_two(st + ((stage + 2) % 3) * KV1, kg, a.lk.sr, vg, a.lv.sr, after);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    const bool have_next = nxt < 64 && ((mine >> nxt) & 1ull);
    if (have_next) {  // next tile's scores first: tensor work overlapping this tile's softmax
#pragma unroll
      for (int n = 0; n < 8; ++n) sn[n][0] = sn[n][1] = sn[n][2] = sn[n][3] = 0.f;
      mma_nt<8>(sn, qa, st + ns * KV1, 0, lane);
    }
    if (have) {
      const bool mx = (mixed >> cur) & 1ull;
      softmax_tile(a, sc, m0, m1, l0, l1, o, mx ? mlo : nullptr, mx ? mhi : nullptr, (long long)cur * BT, lane);
      uint32_t pa[4][NS][4];
      to_a<NS, 4, 8>(pa, sc);
      mma_pv<NS, 4>(o, pa, st + stage * KV1 + BT * LD, 0, lane);
    }
    if (have_next) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    have = have_next;
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  __syncthreads();  // stages are reused by the next item
}

// ===================================================================================================== v13 forward
// Explicit BF16 SDPA only; the emitter proves an unmasked or canonical causal
// boundary. This uses the existing native MMA/layout helpers with the forward
// arithmetic order of PyTorch 2.9.1's hdim64/dropout0 Flash specialization:
// 128-key tiles in reverse order, raw-dot row maxima, unfused score exponent,
// sequential per-thread probability sums, quad XOR 2 then 1, reciprocal scale.
// Source attribution (algorithm/order, no external runtime dependency):
// Dao-AILab/flash-attention@979702c87a8713a8e0a5e9fee122b90d2ef13be5,
// flash_fwd_launch_template.h, flash_fwd_kernel.h, softmax.h and utils.h.
// PyTorch v2.9.1 aten/src/ATen/CMakeLists.txt defines UNFUSE_FMA for Flash.
// Our CTA still has eight independent m16 row warps. The paired v13 backward
// below reuses v3 staging/MMA with the pinned deterministic arithmetic order.
template <int NS>
PFLATLONG_DEV void forward_item13(const Args& a, int bh, int qb, void* smem_v) {
  static_assert(NS == 1, "v13 requires the explicit BF16 SDPA contract");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = qb * BR, wrow = row0 + 16 * warp;
  const int r0 = wrow + (lane >> 2), r1 = r0 + 8;
  __nv_bfloat16* qs = static_cast<__nv_bfloat16*>(smem_v);
  __nv_bfloat16* ks = qs + BR * LD;
  __nv_bfloat16* vs = ks + BR * LD;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  cp_commit();
  cp_wait0();
  __syncthreads();
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  // Backward uses 64-key tile words. Retain every potentially nonzero causal
  // tile, without the older numerical underflow-pruning optimization.
  const int live_count = a.has_mask ? ((wrow + 15) / BT + 1) : (a.t / BT);
  if (lane == 0)
    a.live[(long long)bh * (a.t >> 4) + (wrow >> 4)] =
        live_count == 64 ? ~0ull : ((1ull << live_count) - 1ull);
  float o[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[n][e] = 0.f;
  float maximum0 = -INFINITY, maximum1 = -INFINITY;
  float sum0 = 0.f, sum1 = 0.f;
  const float scale_log2 = __fmul_rn(a.scale, L2E);
  const int first = a.has_mask ? qb : (a.t / BR - 1);
  for (int tile = first; tile >= 0; --tile) {
    load_rows<BR>(ks, kg + (long long)(tile * BR) * a.lk.sr, a.lk.sr);
    load_rows<BR>(vs, vg + (long long)(tile * BR) * a.lv.sr, a.lv.sr);
    cp_commit();
    cp_wait0();
    __syncthreads();
    float scores[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) scores[n][e] = 0.f;
    mma_nt<16>(scores, qa, ks, 0, lane);
    const float old0 = maximum0, old1 = maximum1;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      const int column = tile * BR + n * 8 + 2 * (lane & 3);
      if (a.has_mask) {
        if (column > r0) scores[n][0] = -INFINITY;
        if (column + 1 > r0) scores[n][1] = -INFINITY;
        if (column > r1) scores[n][2] = -INFINITY;
        if (column + 1 > r1) scores[n][3] = -INFINITY;
      }
      maximum0 = fmaxf(maximum0, scores[n][0]);
      maximum0 = fmaxf(maximum0, scores[n][1]);
      maximum1 = fmaxf(maximum1, scores[n][2]);
      maximum1 = fmaxf(maximum1, scores[n][3]);
    }
#pragma unroll
    for (int offset = 2; offset >= 1; offset >>= 1) {
      maximum0 = fmaxf(maximum0, __shfl_xor_sync(0xffffffffu, maximum0, offset));
      maximum1 = fmaxf(maximum1, __shfl_xor_sync(0xffffffffu, maximum1, offset));
    }
    float rescale0 = 1.f, rescale1 = 1.f;
    if (tile != first) {
      rescale0 = exp2f(__fmul_rn(old0 - maximum0, scale_log2));
      rescale1 = exp2f(__fmul_rn(old1 - maximum1, scale_log2));
      // Flash contracts the sum rescale with its first probability add.
      // Defer that operation; O rescaling remains a separate multiply.
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        o[n][0] *= rescale0; o[n][1] *= rescale0;
        o[n][2] *= rescale1; o[n][3] *= rescale1;
      }
    }
    const float max_scaled0 = __fmul_rn(maximum0, scale_log2);
    const float max_scaled1 = __fmul_rn(maximum1, scale_log2);
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      scores[n][0] = exp2f(__fmul_rn(scores[n][0], scale_log2) - max_scaled0);
      scores[n][1] = exp2f(__fmul_rn(scores[n][1], scale_log2) - max_scaled0);
      scores[n][2] = exp2f(__fmul_rn(scores[n][2], scale_log2) - max_scaled1);
      scores[n][3] = exp2f(__fmul_rn(scores[n][3], scale_log2) - max_scaled1);
      if (n == 0 && tile != first) {
        sum0 = fmaf(sum0, rescale0, scores[n][0]);
        sum1 = fmaf(sum1, rescale1, scores[n][2]);
      } else { sum0 += scores[n][0]; sum1 += scores[n][2]; }
      sum0 += scores[n][1]; sum1 += scores[n][3];
    }
    uint32_t pa[8][1][4];
    to_a<1, 8, 16>(pa, scores);
    mma_pv<1, 8>(o, pa, vs, 0, lane);
    __syncthreads();  // K/V shared storage can now be overwritten.
  }
#pragma unroll
  for (int offset = 2; offset >= 1; offset >>= 1) {
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
  }
  const float inverse0 = 1.f / sum0, inverse1 = 1.f / sum1;
  __nv_bfloat16* output = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(output + (long long)r0 * a.lo.sr + d, o[n][0] * inverse0, o[n][1] * inverse0);
    store_bf16x2(output + (long long)r1 * a.lo.sr + d, o[n][2] * inverse1, o[n][3] * inverse1);
  }
  if ((lane & 3) == 0) {
    // Store the existing private log2 tape ABI after forming Flash's natural
    // LSE. The BF16 saved output remains the backward delta operand.
    a.lse2[(long long)bh * a.t + r0] = fmaf(maximum0, a.scale, __logf(sum0)) * L2E;
    a.lse2[(long long)bh * a.t + r1] = fmaf(maximum1, a.scale, __logf(sum1)) * L2E;
  }
  __syncthreads();
}

// Products of one 16-wide chunk: S (A rows x chunk rows of X) and dP (B rows x chunk rows of Y), both [2 n-tiles].
PFLATLONG_DEV void chunk_products(float (&s)[2][4], float (&dp)[2][4], const uint32_t (&xa)[4][4],
                              const uint32_t (&ya)[4][4], const __nv_bfloat16* xs, const __nv_bfloat16* ys, int c,
                              int lane) {
#pragma unroll
  for (int n = 0; n < 2; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = dp[n][0] = dp[n][1] = dp[n][2] = dp[n][3] = 0.f;
#pragma unroll
  for (int k16 = 0; k16 < 4; ++k16) {
    uint32_t bx[4], by[4];
    const int off = (16 * c + (lane >> 4) * 8 + (lane & 7)) * LD + 16 * k16 + ((lane >> 3) & 1) * 8;
    ldsm_x4(bx, xs + off);
    ldsm_x4(by, ys + off);
    mma(s[0], xa[k16], bx[0], bx[1]);
    mma(s[1], xa[k16], bx[2], bx[3]);
    mma(dp[0], ya[k16], by[0], by[1]);
    mma(dp[1], ya[k16], by[2], by[3]);
  }
}

// v13 backward: pinned Flash BF16 arithmetic, reusing v3 staging/MMA.
// UNFUSE_FMA exponent, eight-lane delta, reverse128 dK/dV, and deterministic
// per128-key dQ partials. SPLITS is bound by the emitter target/shape policy.
// Algorithm/order attribution: the Flash commit pinned above,
// flash_bwd_kernel.h, flash_bwd_preprocess_kernel.h, flash_bwd_launch_template.h.
template <int NS, int SPLITS>
PFLATLONG_DEV void dq_item13(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(st + 3 * KV1);
  const int t = a.t, row0 = qb * BR, wrow = row0 + 16 * warp, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  __nv_bfloat16* qs = st + KV1;   // Q staging (stage 1)
  __nv_bfloat16* ds = st + 2 * KV1;  // dO staging (stage 2)
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  load_rows<BR>(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[warp] = mine;
  // Pinned Flash preprocessing: eight consecutive features per lane,
  // eight lanes per row, sequential local dot then XOR4/2/1 reduction.
#pragma unroll
  for (int group = 0; group < 4; ++group) {
    const int r = wrow + 4 * group + (lane >> 3), sub = lane & 7;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 8 * sub;
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 8 * sub;
    const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp));
    const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op));
    float acc = bf_lo(w.x) * bf_lo(z.x);
    acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
    acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
    acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
    acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    if (sub == 0) a.delta[(long long)bh * t + r] = acc;
  }
  __syncwarp();
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = a.delta[(long long)bh * t + r_lo];
  const float d_hi = a.delta[(long long)bh * t + r_hi];
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  cp_wait0();
  __syncthreads();
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  a_frags(da, ds, 16 * warp, lane);
  __syncthreads();  // staging stages are reused
  unsigned long long all = 0ull;
#pragma unroll
  for (int w = 0; w < NW; ++w) all |= flags[w];
  float dq[8][4];
  static_assert(NS == 1 && SPLITS > 0 && SPLITS <= 32, "Flash BF16 deterministic split regime");
  float partitions[SPLITS][8][4] = {};
#pragma unroll
  for (int n = 0; n < 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  load_two(st, kg, a.lk.sr, vg, a.lv.sr, cur);
  cp_commit();
  load_two(st + KV1, kg, a.lk.sr, vg, a.lv.sr, nxt);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait1();
    __syncthreads();  // tile cur has landed; stage (stage + 2) % 3 is no longer read
    load_two(st + ((stage + 2) % 3) * KV1, kg, a.lk.sr, vg, a.lv.sr, after);
    cp_commit();
    if ((mine >> cur) & 1ull) {
      const __nv_bfloat16* ks = st + stage * KV1;
      const __nv_bfloat16* vs = ks + BT * LD;
      const bool mx = (mixed >> cur) & 1ull;
      float s[2][2][4], dp[2][2][4];
      chunk_products(s[0], dp[0], qa, da, ks, vs, 0, lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int b = c & 1;
        if (c + 1 < 4) chunk_products(s[b ^ 1], dp[b ^ 1], qa, da, ks, vs, c + 1, lane);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const long long col = (long long)cur * BT + 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float x = scaled(a, s[b][n][e]);
            if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
            const float p = exp2f(__fsub_rn(__fmul_rn(x, L2E), (e < 2 ? s_lo : s_hi)));
            s[b][n][e] = p * (dp[b][n][e] - (e < 2 ? d_lo : d_hi));
          }
        }
        uint32_t sa[1][NS][4];
        to_a<NS, 1, 2>(sa, s[b]);
        mma_pv<NS, 1>(dq, sa, ks, 16 * c, lane);
      }
    }
    if ((nxt >> 1) != (cur >> 1)) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          partitions[(cur >> 1) % SPLITS][n][e] += dq[n][e];
          dq[n][e] = 0.f;
        }
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float sum = 0.f;
#pragma unroll
      for (int split = 0; split < SPLITS; ++split) sum += partitions[split][n][e];
      dq[n][e] = sum * a.scale;
    }
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  __syncthreads();
}

template <int NS>
PFLATLONG_DEV void kv_item13(const Args& a, int bh, int kb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  float* sv = reinterpret_cast<float*>(st + 3 * KV1);  // [3 stages][lse2 64, delta 64]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  const int t = a.t, key0 = kb * BR, ktw = 2 * kb + (warp >> 2);
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  __nv_bfloat16* ks = st + KV1;      // K staging (stage 1)
  __nv_bfloat16* vs = st + 2 * KV1;  // V staging (stage 2)
  load_rows<BR>(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr);
  load_rows<BR>(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr);
  cp_commit();
  if (warp < 2) {
    const unsigned long long lv = live_cols(a, bh, 2 * kb + warp, lane);
    const unsigned long long mx = mixed_cols(a, bh, 2 * kb + warp, lane);
    if (lane == 0) {
      flags[warp] = lv;
      flags[2 + warp] = mx;
    }
  }
  cp_wait0();
  __syncthreads();
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * warp, lane);
  a_frags(va, vs, 16 * warp, lane);
  const unsigned long long mine = flags[warp >> 2], mixed = flags[2 + (warp >> 2)], all = flags[0] | flags[1];
  __syncthreads();  // staging stages are reused
  const int k_lo = key0 + 16 * warp + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk[n][e] = dv[n][e] = 0.f;
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_two(st + stg * KV1, qg, a.lq.sr, dg, a.ld.sr, qt);
    float* d = sv + stg * 2 * BT;
    if (threadIdx.x < 16) cp16(d + 4 * threadIdx.x, a.lse2 + (long long)bh * t + qt * BT + 4 * threadIdx.x);
    else if (threadIdx.x < 32)
      cp16(d + BT + 4 * (threadIdx.x - 16), a.delta + (long long)bh * t + qt * BT + 4 * (threadIdx.x - 16));
  };
  // Flash dK/dV visits 128-query tiles descending, with the
  // ordinary ascending k16 products within each 128-query tile.
  auto next_query = [&](int previous) {
    int candidate = previous < 0 ? (t / 128 - 1) * 2
        : ((previous & 1) ? previous - 3 : previous + 1);
    while (candidate >= 0) {
      if ((all >> candidate) & 1ull) return candidate;
      candidate = (candidate & 1) ? candidate - 3 : candidate + 1;
    }
    return 64;
  };
  int cur = next_query(-1), nxt = cur == 64 ? 64 : next_query(cur);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = nxt == 64 ? 64 : next_query(nxt);
    cp_wait1();
    __syncthreads();
    issue(after, (stage + 2) % 3);
    cp_commit();
    if ((mine >> cur) & 1ull) {
      const __nv_bfloat16* qd = st + stage * KV1;
      const __nv_bfloat16* dd = qd + BT * LD;
      const float* sl = sv + stage * 2 * BT;
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      float s[2][2][4], dp[2][2][4];
      chunk_products(s[0], dp[0], ka, va, qd, dd, 0, lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int b = c & 1;
        if (c + 1 < 4) chunk_products(s[b ^ 1], dp[b ^ 1], ka, va, qd, dd, c + 1, lane);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int j = 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int jj = j + (e & 1);
            float x = scaled(a, s[b][n][e]);
            if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
            const float p = exp2f(__fsub_rn(__fmul_rn(x, L2E), sl[jj]));
            s[b][n][e] = p;
            dp[b][n][e] = p * (dp[b][n][e] - dl[jj]);
          }
        }
        uint32_t pa[1][NS][4];
        to_a<NS, 1, 2>(pa, s[b]);
        mma_pv<NS, 1>(dv, pa, dd, 16 * c, lane);
        to_a<NS, 1, 2>(pa, dp[b]);
        mma_pv<NS, 1>(dk, pa, qd, 16 * c, lane);
      }
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0] * a.scale, dk[n][1] * a.scale);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2] * a.scale, dk[n][3] * a.scale);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  __syncthreads();
}

template <int NS>
PFLATLONG_DEV void dq_item3(const Args& a, int bh, int qb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(st + 3 * KV1);
  const int t = a.t, row0 = qb * BR, wrow = row0 + 16 * warp, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  __nv_bfloat16* qs = st + KV1;   // Q staging (stage 1)
  __nv_bfloat16* ds = st + 2 * KV1;  // dO staging (stage 2)
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  load_rows<BR>(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[warp] = mine;
  // delta for row wrow + (lane >> 1): two lanes per row, fixed order, pair sum (as v1).
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  cp_wait0();
  __syncthreads();
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  a_frags(da, ds, 16 * warp, lane);
  __syncthreads();  // staging stages are reused
  unsigned long long all = 0ull;
#pragma unroll
  for (int w = 0; w < NW; ++w) all |= flags[w];
  float dq[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  load_two(st, kg, a.lk.sr, vg, a.lv.sr, cur);
  cp_commit();
  load_two(st + KV1, kg, a.lk.sr, vg, a.lv.sr, nxt);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait1();
    __syncthreads();  // tile cur has landed; stage (stage + 2) % 3 is no longer read
    load_two(st + ((stage + 2) % 3) * KV1, kg, a.lk.sr, vg, a.lv.sr, after);
    cp_commit();
    if ((mine >> cur) & 1ull) {
      const __nv_bfloat16* ks = st + stage * KV1;
      const __nv_bfloat16* vs = ks + BT * LD;
      const bool mx = (mixed >> cur) & 1ull;
      float s[2][2][4], dp[2][2][4];
      chunk_products(s[0], dp[0], qa, da, ks, vs, 0, lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int b = c & 1;
        if (c + 1 < 4) chunk_products(s[b ^ 1], dp[b ^ 1], qa, da, ks, vs, c + 1, lane);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const long long col = (long long)cur * BT + 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            float x = scaled(a, s[b][n][e]);
            if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
            const float p = exp2f(fmaf(x, L2E, -(e < 2 ? s_lo : s_hi)));
            s[b][n][e] = grad_scale(a, p * (dp[b][n][e] - (e < 2 ? d_lo : d_hi)));
          }
        }
        uint32_t sa[1][NS][4];
        to_a<NS, 1, 2>(sa, s[b]);
        mma_pv<NS, 1>(dq, sa, ks, 16 * c, lane);
      }
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  __syncthreads();
}

template <int NS>
PFLATLONG_DEV void kv_item3(const Args& a, int bh, int kb, void* smem_v) {
  char* smem = static_cast<char*>(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __nv_bfloat16* st = reinterpret_cast<__nv_bfloat16*>(smem);
  float* sv = reinterpret_cast<float*>(st + 3 * KV1);  // [3 stages][lse2 64, delta 64]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  const int t = a.t, key0 = kb * BR, ktw = 2 * kb + (warp >> 2);
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  __nv_bfloat16* ks = st + KV1;      // K staging (stage 1)
  __nv_bfloat16* vs = st + 2 * KV1;  // V staging (stage 2)
  load_rows<BR>(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr);
  load_rows<BR>(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr);
  cp_commit();
  if (warp < 2) {
    const unsigned long long lv = live_cols(a, bh, 2 * kb + warp, lane);
    const unsigned long long mx = mixed_cols(a, bh, 2 * kb + warp, lane);
    if (lane == 0) {
      flags[warp] = lv;
      flags[2 + warp] = mx;
    }
  }
  cp_wait0();
  __syncthreads();
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * warp, lane);
  a_frags(va, vs, 16 * warp, lane);
  const unsigned long long mine = flags[warp >> 2], mixed = flags[2 + (warp >> 2)], all = flags[0] | flags[1];
  __syncthreads();  // staging stages are reused
  const int k_lo = key0 + 16 * warp + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4];
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dk[n][e] = dv[n][e] = 0.f;
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_two(st + stg * KV1, qg, a.lq.sr, dg, a.ld.sr, qt);
    float* d = sv + stg * 2 * BT;
    if (threadIdx.x < 16) cp16(d + 4 * threadIdx.x, a.lse2 + (long long)bh * t + qt * BT + 4 * threadIdx.x);
    else if (threadIdx.x < 32)
      cp16(d + BT + 4 * (threadIdx.x - 16), a.delta + (long long)bh * t + qt * BT + 4 * (threadIdx.x - 16));
  };
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait1();
    __syncthreads();
    issue(after, (stage + 2) % 3);
    cp_commit();
    if ((mine >> cur) & 1ull) {
      const __nv_bfloat16* qd = st + stage * KV1;
      const __nv_bfloat16* dd = qd + BT * LD;
      const float* sl = sv + stage * 2 * BT;
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      float s[2][2][4], dp[2][2][4];
      chunk_products(s[0], dp[0], ka, va, qd, dd, 0, lane);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int b = c & 1;
        if (c + 1 < 4) chunk_products(s[b ^ 1], dp[b ^ 1], ka, va, qd, dd, c + 1, lane);
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          const int j = 16 * c + 8 * n + 2 * (lane & 3);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int jj = j + (e & 1);
            float x = scaled(a, s[b][n][e]);
            if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
            const float p = exp2f(fmaf(x, L2E, -sl[jj]));
            s[b][n][e] = p;
            dp[b][n][e] = grad_scale(a, p * (dp[b][n][e] - dl[jj]));
          }
        }
        uint32_t pa[1][NS][4];
        to_a<NS, 1, 2>(pa, s[b]);
        mma_pv<NS, 1>(dv, pa, dd, 16 * c, lane);
        to_a<NS, 1, 2>(pa, dp[b]);
        mma_pv<NS, 1>(dk, pa, qd, 16 * c, lane);
      }
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  __syncthreads();
}

#if defined(PFLATLONG_WGMMA)
// ====================================================================================================== v4 schedule
// Hopper warpgroup products (requires sm_90a): the CTA's two warpgroups each own 64 rows (forward, dQ) or 64 keys
// (dK/dV). Every product is wgmma.mma_async m64n64k16 with the A operand in registers (the same per-warp fragment
// layout as the mma.sync schedules: warp w of a warpgroup holds rows 16w..16w+15) and the B operand read by the
// tensor cores from a 128-byte-swizzled shared tile (K-major for Q K^T / dO V^T / K Q^T / V dO^T, transposed for
// P V / dS K / P^T dO / dS^T Q). Arithmetic is that of the mma.sync schedules: BF16 x BF16 products accumulated in
// FP32 from +0 (accumulation order inside a k16 step and across k16 steps may differ), P/dS pieces as NS.
// The next tile's score products are issued asynchronously before the current tile's softmax/elementwise work.
#ifndef PFLATLONG_TLBO
#define PFLATLONG_TLBO 1
#endif
#ifndef PFLATLONG_TSBO
#define PFLATLONG_TSBO 64
#endif
constexpr int SWT = 64 * 128;  // bytes of one swizzled 64 x 64 BF16 tile

PFLATLONG_DEV char* align1024(void* p) {
  const uint32_t s = smem_u32(p);
  return static_cast<char*>(p) + ((1024u - (s & 1023u)) & 1023u);
}
// Copy 64 rows x 64 BF16 (row stride `stride` elements) into a 128B-swizzled tile; all 256 threads.
PFLATLONG_DEV void load_sw(char* dst, const __nv_bfloat16* src, long long stride) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = (int)threadIdx.x + k * NT, r = idx >> 3, j = idx & 7;
    cp16(dst + r * 128 + ((j ^ (r & 7)) << 4), src + (long long)r * stride + 8 * j);
  }
}
PFLATLONG_DEV uint64_t sw_desc(const void* p, uint32_t lbo, uint32_t sbo) {
  uint64_t d = (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4);
  d |= (uint64_t)(lbo & 0x3FFF) << 16;
  d |= (uint64_t)(sbo & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}
// K-major tile (rows = n, 64 k units per row): k16 step ks starts 32 bytes further along the swizzled row.
PFLATLONG_DEV uint64_t kdesc(const char* tile, int ks) { return sw_desc(tile + 32 * ks, 1, 64); }
// Transposed (MN-major) tile (rows = k, 64 n units per row): k16 step ks starts 16 rows further.
PFLATLONG_DEV uint64_t tdesc(const char* tile, int ks) { return sw_desc(tile + 2048 * ks, PFLATLONG_TLBO, PFLATLONG_TSBO); }
PFLATLONG_DEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
PFLATLONG_DEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
PFLATLONG_DEV void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
PFLATLONG_DEV void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

// Register pinning (CUTLASS warpgroup_fence_operand): wgmma reads its A registers and writes its accumulators
// asynchronously; empty volatile asm uses after the wait keep the compiler from reusing or moving them earlier.
PFLATLONG_DEV void pin(float (&x)[8][4]) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(x[n][e])::"memory");
}
template <int K, int P>
PFLATLONG_DEV void pin(uint32_t (&x)[K][P][4]) {
#pragma unroll
  for (int k = 0; k < K; ++k)
#pragma unroll
    for (int q = 0; q < P; ++q)
#pragma unroll
      for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(x[k][q][e])::"memory");
}
PFLATLONG_DEV void pin(uint32_t (&x)[4][4]) {
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(x[k][e])::"memory");
}
template <int TNSP>
PFLATLONG_DEV void wgmma64(float (&d)[8][4], const uint32_t (&a)[4], uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %36, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
      "%23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %37, p, 1, 1, %38;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
        "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(b), "n"(TNSP));
}
// acc (64 rows x 64) (+)= A (64 x 64, registers, 4 k16 steps) * B where B[k][n] = tile[n][k] (K-major tile).
PFLATLONG_DEV void wg_nt(float (&acc)[8][4], const uint32_t (&a)[4][4], const char* tile, bool accumulate) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) wgmma64<0>(acc, a[ks], kdesc(tile, ks), accumulate || ks > 0);
}
// acc += P (64 x 64, NS pieces per k16 step) * B where B[k][n] = tile[k][n] (transposed tile).
template <int NS>
PFLATLONG_DEV void wg_pv(float (&acc)[8][4], const uint32_t (&p)[4][NS][4], const char* tile) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
#pragma unroll
    for (int piece = 0; piece < NS; ++piece) wgmma64<1>(acc, p[kk][piece], tdesc(tile, kk), 1);
}
PFLATLONG_DEV void zero8(float (&x)[8][4]) {
#pragma unroll
  for (int n = 0; n < 8; ++n) x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f;
}

constexpr int ST4 = 2 * SWT;  // one stage: two swizzled 64 x 64 tiles
__host__ __device__ constexpr int forward4_shared_bytes() { return 1024 + 3 * ST4 + 2 * BR * LD + 16 * 8; }
__host__ __device__ constexpr int dq4_shared_bytes() { return 1024 + 3 * ST4 + 2 * 2 * BR * LD + 16 * 8; }
__host__ __device__ constexpr int kv4_shared_bytes() { return 1024 + 3 * ST4 + 3 * 4 * 2 * BT + 2 * 2 * BR * LD + 16 * 8; }

template <int NS>
PFLATLONG_DEV void forward_item4(const Args& a, int bh, int qb, void* smem_v) {
  char* base = align1024(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
  char* ring = base;                                                     // 3 stages x (K, V) swizzled tiles
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(base + 3 * ST4);  // Q [128][LD]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(qs + BR * LD);
  const int t = a.t, row0 = qb * BR, wrow = row0 + 16 * warp, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  cp_commit();
  cp_wait0();
  __syncthreads();
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * warp * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[warp] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  __syncthreads();
  const unsigned long long group = flags[4 * wg] | flags[4 * wg + 1] | flags[4 * wg + 2] | flags[4 * wg + 3];
  const unsigned long long all = flags[0] | flags[1] | flags[2] | flags[3] | flags[4] | flags[5] | flags[6] | flags[7];
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr);
      load_sw(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  __syncthreads();
  bool have = cur < 64 && ((group >> cur) & 1ull);
  if (have) {
    wg_fence();
    wg_nt(sc, qa, ring, false);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait0();
    proxy_fence();
    __syncthreads();  // tile nxt landed and is visible to the tensor cores; stage (stage + 2) % 3 is free
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    const bool have_next = nxt < 64 && ((group >> nxt) & 1ull);
    if (have_next) {
      wg_fence();
      wg_nt(sn, qa, ring + ns * ST4, false);  // asynchronous: overlaps the softmax below
      wg_commit();
    }
    if (have) {
      const bool mx = (mixed >> cur) & 1ull;
      softmax_tile(a, sc, m0, m1, l0, l1, o, mx ? mlo : nullptr, mx ? mhi : nullptr, (long long)cur * BT, lane);
      uint32_t pa[4][NS][4];
      to_a<NS, 4, 8>(pa, sc);
      wg_fence();
      wg_pv<NS>(o, pa, ring + stage * ST4 + SWT);
      wg_commit();
      pin(pa);
    }
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(qa);
    if (have_next) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    have = have_next;
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  __syncthreads();
}

template <int NS>
PFLATLONG_DEV void dq_item4(const Args& a, int bh, int qb, void* smem_v) {
  char* base = align1024(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
  char* ring = base;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(base + 3 * ST4);
  __nv_bfloat16* ds = qs + BR * LD;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ds + BR * LD);
  const int t = a.t, row0 = qb * BR, wrow = row0 + 16 * warp, qt = wrow >> 6;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows<BR>(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr);
  load_rows<BR>(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[warp] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  cp_wait0();
  __syncthreads();
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * warp, lane);
  a_frags(da, ds, 16 * warp, lane);
  const unsigned long long group = flags[4 * wg] | flags[4 * wg + 1] | flags[4 * wg + 2] | flags[4 * wg + 3];
  const unsigned long long all = flags[0] | flags[1] | flags[2] | flags[3] | flags[4] | flags[5] | flags[6] | flags[7];
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr);
      load_sw(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait1();
    proxy_fence();
    __syncthreads();
    issue(after, (stage + 2) % 3);
    cp_commit();
    if ((group >> cur) & 1ull) {
      const char* kt = ring + stage * ST4;
      const char* vt = kt + SWT;
      const bool mx = (mixed >> cur) & 1ull;
      wg_fence();
      wg_nt(s, qa, kt, false);
      wg_nt(dp, da, vt, false);
      wg_commit();
      wg_wait<0>();
      pin(s);
      pin(dp);
      pin(qa);
      pin(da);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const long long col = (long long)cur * BT + 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float x = scaled(a, s[n][e]);
          if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
          const float p = exp2f(fmaf(x, L2E, -(e < 2 ? s_lo : s_hi)));
          s[n][e] = grad_scale(a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
        }
      }
      uint32_t sa[4][NS][4];
      to_a<NS, 4, 8>(sa, s);
      wg_fence();
      wg_pv<NS>(dq, sa, kt);
      wg_commit();
      wg_wait<0>();
      pin(dq);
      pin(sa);
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  __syncthreads();
}

template <int NS>
PFLATLONG_DEV void kv_item4(const Args& a, int bh, int kb, void* smem_v) {
  char* base = align1024(smem_v);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2;
  char* ring = base;                                              // 3 stages x (Q, dO) swizzled tiles
  float* sv = reinterpret_cast<float*>(base + 3 * ST4);           // [3][lse2 64, delta 64]
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(sv + 3 * 2 * BT);
  __nv_bfloat16* vs = ks + BR * LD;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(vs + BR * LD);
  const int t = a.t, key0 = kb * BR;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows<BR>(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr);
  load_rows<BR>(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr);
  cp_commit();
  if (warp < 2) {
    const unsigned long long lv = live_cols(a, bh, 2 * kb + warp, lane);
    const unsigned long long mx = mixed_cols(a, bh, 2 * kb + warp, lane);
    if (lane == 0) {
      flags[warp] = lv;
      flags[2 + warp] = mx;
    }
  }
  cp_wait0();
  __syncthreads();
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * warp, lane);
  a_frags(va, vs, 16 * warp, lane);
  const unsigned long long mine = flags[wg], mixed = flags[2 + wg], all = flags[0] | flags[1];
  const int k_lo = key0 + 16 * warp + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr);
    load_sw(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr);
    float* d = sv + stg * 2 * BT;
    if (threadIdx.x < 16) cp16(d + 4 * threadIdx.x, a.lse2 + (long long)bh * t + qt * BT + 4 * threadIdx.x);
    else if (threadIdx.x < 32)
      cp16(d + BT + 4 * (threadIdx.x - 16), a.delta + (long long)bh * t + qt * BT + 4 * (threadIdx.x - 16));
  };
  int cur = next_bit(all, 0), nxt = next_bit(all, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(all, nxt + 1);
    cp_wait1();
    proxy_fence();
    __syncthreads();
    issue(after, (stage + 2) % 3);
    cp_commit();
    if ((mine >> cur) & 1ull) {
      const char* qt_ = ring + stage * ST4;
      const char* dt = qt_ + SWT;
      const float* sl = sv + stage * 2 * BT;
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      wg_fence();
      wg_nt(s, ka, qt_, false);
      wg_nt(dp, va, dt, false);
      wg_commit();
      wg_wait<0>();
      pin(s);
      pin(dp);
      pin(ka);
      pin(va);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = exp2f(fmaf(x, L2E, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = grad_scale(a, p * (dp[n][e] - dl[jj]));
        }
      }
      uint32_t pa[4][NS][4], sa[4][NS][4];
      to_a<NS, 4, 8>(pa, s);
      to_a<NS, 4, 8>(sa, dp);
      wg_fence();
      wg_pv<NS>(dv, pa, dt);
      wg_pv<NS>(dk, sa, qt_);
      wg_commit();
      wg_wait<0>();
      pin(dv);
      pin(dk);
      pin(pa);
      pin(sa);
    }
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  __syncthreads();
}

// ====================================================================================================== v5 schedule
// v4 products with independent warpgroups: each warpgroup is its own worker on 64-row (forward, dQ) or 64-key (dK/dV)
// items, with a private three-stage operand ring and a named 128-thread barrier (ids 1 and 2), so the two
// warpgroups drift apart and one's softmax/elementwise work overlaps the other's tensor-core products. Items are
// distributed over 2 x gridDim workers in boustrophedon order. Arithmetic is identical to v4.
constexpr int WGB = 50 * 1024;  // bytes of one warpgroup's region (1024-aligned): ring + small state
__host__ __device__ constexpr int v5_shared_bytes() { return 1024 + 2 * WGB; }

PFLATLONG_DEV void wg_bar(int wg) { asm volatile("bar.sync %0, 128;\n" ::"r"(1 + wg) : "memory"); }
// Copy 64 rows x 64 BF16 into a 128B-swizzled tile with the 128 threads of one warpgroup.
PFLATLONG_DEV void load_sw_wg(char* dst, const __nv_bfloat16* src, long long stride, int tid) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int idx = tid + k * 128, r = idx >> 3, j = idx & 7;
    cp16(dst + r * 128 + ((j ^ (r & 7)) << 4), src + (long long)r * stride + 8 * j);
  }
}
// Copy 64 rows x 64 BF16 into shared [64][LD] with one warpgroup.
PFLATLONG_DEV void load_rows_wg(__nv_bfloat16* dst, const __nv_bfloat16* src, long long stride, int tid) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int idx = tid + k * 128, r = idx >> 3, c = idx & 7;
    cp16(dst + r * LD + 8 * c, src + (long long)r * stride + 8 * c);
  }
}
PFLATLONG_DEV long long snake2(long long r) {
  const long long g = 2LL * gridDim.x, b = 2LL * blockIdx.x + (threadIdx.x >> 7);
  return r * g + ((r & 1) ? g - 1 - b : b);
}

template <int NS>
PFLATLONG_DEV void forward_item5(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);  // Q staging aliases stage 2
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt(sc, qa, ring, false);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait0();
    proxy_fence();
    wg_bar(wg);  // tile nxt landed and is visible to the tensor cores; stage (stage + 2) % 3 is free
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    if (nxt < 64) {
      wg_fence();
      wg_nt(sn, qa, ring + ns * ST4, false);
      wg_commit();
    }
    const bool mx = (mixed >> cur) & 1ull;
    softmax_tile(a, sc, m0, m1, l0, l1, o, mx ? mlo : nullptr, mx ? mhi : nullptr, (long long)cur * BT, lane);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv<NS>(o, pa, ring + stage * ST4 + SWT);
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

template <int NS>
PFLATLONG_DEV void dq_item5(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);      // Q staging aliases stage 1
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);  // dO staging aliases stage 2
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);  // staging stages are reused
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const char* kt = ring + stage * ST4;
    const char* vt = kt + SWT;
    const bool mx = (mixed >> cur) & 1ull;
    wg_fence();
    wg_nt(s, qa, kt, false);
    wg_nt(dp, da, vt, false);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(qa);
    pin(da);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const long long col = (long long)cur * BT + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = scaled(a, s[n][e]);
        if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
        const float p = exp2f(fmaf(x, L2E, -(e < 2 ? s_lo : s_hi)));
        s[n][e] = grad_scale(a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
      }
    }
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv<NS>(dq, sa, kt);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

template <int NS>
PFLATLONG_DEV void kv_item5(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);  // [3][lse2 64, delta 64]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);      // K staging aliases stage 1
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);  // V staging aliases stage 2
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);  // staging stages are reused
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const char* qt_ = ring + stage * ST4;
    const char* dt = qt_ + SWT;
    const float* sl = sv + stage * 2 * BT;
    const float* dl = sl + BT;
    const bool mx = (mixed >> cur) & 1ull;
    wg_fence();
    wg_nt(s, ka, qt_, false);
    wg_nt(dp, va, dt, false);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jj = j + (e & 1);
        float x = scaled(a, s[n][e]);
        if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
        const float p = exp2f(fmaf(x, L2E, -sl[jj]));
        s[n][e] = p;
        dp[n][e] = grad_scale(a, p * (dp[n][e] - dl[jj]));
      }
    }
    uint32_t pa[4][NS][4], sa[4][NS][4];
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
    wg_pv<NS>(dv, pa, dt);
    wg_pv<NS>(dk, sa, qt_);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v6 schedule
// v5 structure with fewer instructions per tile: shared-tile descriptors are formed once per stage and advanced by
// constants; mask reads are compiled into a separate tile body chosen once per tile (no predicated per-element
// loads); per-column statistics load as pairs; and when the captured score scale is an exact power of two (POW2,
// decided by the compiler from the captured constant) the scale is folded into the exponent's FMA:
// fmaf(s, c * log2e, -m) == fmaf(fl(s * c), log2e, -m) bit for bit because s * c and c * log2e are exact.
// Unmasked tiles therefore produce the same bits as v5; masked tiles use the v5 formula.
// v7 options of the v6 items. PFLATLONG_FTZ: probabilities use ex2.approx.ftz (results below 2^-126 flush to +0; the
// non-FTZ exp2f adds a subnormal fix-up sequence). PFLATLONG_DEFER_GSCALE (power-of-two scales only): dS2 * c enters the
// dQ/dK products unscaled and the accumulators are multiplied by c once (exact: power-of-two scaling commutes with
// BF16 rounding, piece splits and FP32 accumulation away from subnormals).
PFLATLONG_DEV float ex2(float x) {
#ifdef PFLATLONG_FTZ
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return exp2f(x);
#endif
}
#ifdef PFLATLONG_DEFER_GSCALE
#define PFLATLONG_GS(POW2, a, g) ((POW2) ? (g) : grad_scale((a), (g)))
#else
#define PFLATLONG_GS(POW2, a, g) grad_scale((a), (g))
#endif
PFLATLONG_DEV uint64_t desc_add(uint64_t d, uint32_t bytes) { return d + (uint64_t)(bytes >> 4); }

template <bool POW2>
PFLATLONG_DEV float sc2(const Args& a) { return POW2 ? a.scale * L2E : L2E; }

// Online softmax of one m16 x 64 tile (v5 softmax_tile semantics). MASKED reads the mask; POW2 folds the scale.
template <bool MASKED, bool POW2>
PFLATLONG_DEV void softmax_tile6(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                             float (&o)[8][4], const float* mlo, const float* mhi, long long col0, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    if (!FOLD) {
      const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = scaled(a, s[n][e]);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
        s[n][e] = x;
      }
    }
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  if (FOLD) {  // max of the scaled scores = scale * max of the products (positive power of two: exact)
    x0 *= a.scale;
    x1 *= a.scale;
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  // A finite additive mask may use -FLT_MAX. In a leading fully masked row/tile,
  // multiplying that maximum by log2(e) overflows even though score - maximum
  // is finite. Subtract first only in that exceptional case; ordinary online
  // softmax retains its existing fused arithmetic and rounding.
  const bool shift0 = isinf(ml0) && isfinite(n0), shift1 = isinf(ml1) && isfinite(n1);
  const float c0 = m0 == -INFINITY ? 0.f : ex2(shift0 ? (m0 - n0) * L2E : fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : ex2(shift1 ? (m1 - n1) * L2E : fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = FOLD ? sc2<true>(a) : L2E;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = ex2(shift0 ? ((FOLD ? scaled(a, s[n][0]) : s[n][0]) - n0) * L2E : fmaf(s[n][0], k, -ml0));
    s[n][1] = ex2(shift0 ? ((FOLD ? scaled(a, s[n][1]) : s[n][1]) - n0) * L2E : fmaf(s[n][1], k, -ml0));
    s[n][2] = ex2(shift1 ? ((FOLD ? scaled(a, s[n][2]) : s[n][2]) - n1) * L2E : fmaf(s[n][2], k, -ml1));
    s[n][3] = ex2(shift1 ? ((FOLD ? scaled(a, s[n][3]) : s[n][3]) - n1) * L2E : fmaf(s[n][3], k, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
    o[n][0] *= c0;
    o[n][1] *= c0;
    o[n][2] *= c1;
    o[n][3] *= c1;
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

// acc (+)= A (registers) * B (K-major tile with descriptor d0 at its k = 0 origin).
PFLATLONG_DEV void wg_nt6(float (&acc)[8][4], const uint32_t (&a)[4][4], uint64_t d0) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) wgmma64<0>(acc, a[ks], desc_add(d0, 32 * ks), ks > 0);
}
template <int NS>
PFLATLONG_DEV void wg_pv6(float (&acc)[8][4], const uint32_t (&p)[4][NS][4], uint64_t d0) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk)
#pragma unroll
    for (int piece = 0; piece < NS; ++piece) wgmma64<1>(acc, p[kk][piece], desc_add(d0, 2048 * kk), 1);
}

template <int NS, bool POW2>
PFLATLONG_DEV void forward_item6(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  P9AX(3, P9NOW());
  P9AX(4, (unsigned long long)__popcll(group));
  const uint64_t ring_desc = sw_desc(ring, 1, 64);  // K-major and transposed descriptors share these fields
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait0();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));
      wg_commit();
    }
    if ((mixed >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, sc, m0, m1, l0, l1, o, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, sc, m0, m1, l0, l1, o, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
  P9AX(5, P9NOW());
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;  // no allowed key: o is 0
  if (empty0) l0 = 1.f;
  if (empty1) l1 = 1.f;
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

// P9-A: forward_item6 over 128 query rows (two m64 halves h = 0, 1 of one warpgroup) sharing every K/V stage. Per key
// tile: both QK products are issued, half 0's softmax overlaps half 1's QK, half 1's softmax overlaps half 0's PV.
// Every row sees the same tiles in the same order with the same operations as forward_item6 (bitwise equal); a tile
// live for either half is evaluated for both (exactly as forward_item6 evaluates the union of its four warps).
template <int NS, bool POW2>
PFLATLONG_DEV void forward_item6w(const Args& a, int bh, int qb128, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);  // 128 x LD rows over stages 1-2 (prologue only)
  const int t = a.t, row0 = qb128 * 128;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr;
  load_rows_wg(qs, qg, a.lq.sr, tid);
  load_rows_wg(qs + 64 * LD, qg + 64LL * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  P9AX(3, P9NOW());
  const int wrow0 = row0 + 16 * w4, wrow1 = wrow0 + 64;
  const unsigned long long mine0 = live_tiles(a, bh, wrow0, qs + 16 * w4 * LD, lane);
  const unsigned long long mine1 = live_tiles(a, bh, wrow1, qs + (64 + 16 * w4) * LD, lane);
  const unsigned long long mixed0 = mixed_rows(a, bh, 2 * qb128, lane);
  const unsigned long long mixed1 = mixed_rows(a, bh, 2 * qb128 + 1, lane);
  if (lane == 0) {
    flags[w4] = mine0;
    flags[4 + w4] = mine1;
    a.live[(long long)bh * (t >> 4) + (wrow0 >> 4)] = mine0;
    a.live[(long long)bh * (t >> 4) + (wrow1 >> 4)] = mine1;
  }
  uint32_t qa0[4][4], qa1[4][4];
  a_frags(qa0, qs, 16 * w4, lane);
  a_frags(qa1, qs, 64 + 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3] | flags[4] | flags[5] | flags[6] | flags[7];
  P9AX(4, (unsigned long long)__popcll(group));
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o0[8][4], o1[8][4], s0[8][4], s1[8][4];
  zero8(o0);
  zero8(o1);
  float m00 = -INFINITY, m01 = -INFINITY, l00 = 0.f, l01 = 0.f;
  float m10 = -INFINITY, m11 = -INFINITY, l10 = 0.f, l11 = 0.f;
  const int r0_lo = wrow0 + (lane >> 2), r0_hi = r0_lo + 8, r1_lo = wrow1 + (lane >> 2), r1_hi = r1_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();                                                    // this tile's stage has landed
    proxy_fence();
    wg_bar(wg);                                                    // ... for every thread; stage + 2 is free
    issue(after, stage == 0 ? 2 : stage - 1);                      // (stage + 2) % 3
    cp_commit();
    const uint64_t dk = desc_add(ring_desc, stage * ST4), dv = desc_add(ring_desc, stage * ST4 + SWT);
    wg_fence();
    wg_nt6(s0, qa0, dk);
    wg_commit();
    wg_nt6(s1, qa1, dk);
    wg_commit();
    wg_wait<1>();
    pin(s0);
    if ((mixed0 >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, s0, m00, m01, l00, l01, o0, mask_row(a, bh, r0_lo), mask_row(a, bh, r0_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, s0, m00, m01, l00, l01, o0, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa0[4][NS][4];
    to_a<NS, 4, 8>(pa0, s0);
    wg_fence();
    wg_pv6<NS>(o0, pa0, dv);
    wg_commit();
    wg_wait<1>();
    pin(s1);
    if ((mixed1 >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, s1, m10, m11, l10, l11, o1, mask_row(a, bh, r1_lo), mask_row(a, bh, r1_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, s1, m10, m11, l10, l11, o1, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa1[4][NS][4];
    to_a<NS, 4, 8>(pa1, s1);
    wg_fence();
    wg_pv6<NS>(o1, pa1, dv);
    wg_commit();
    wg_wait<0>();
    pin(o0);
    pin(o1);
    pin(pa0);
    pin(pa1);
    pin(qa0);
    pin(qa1);
    cur = nxt;
    nxt = after;
    stage = stage == 2 ? 0 : stage + 1;
  }
  cp_wait0();
  P9AX(5, P9NOW());
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l00 += __shfl_xor_sync(0xffffffffu, l00, off);
    l01 += __shfl_xor_sync(0xffffffffu, l01, off);
    l10 += __shfl_xor_sync(0xffffffffu, l10, off);
    l11 += __shfl_xor_sync(0xffffffffu, l11, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(og + (long long)r0_lo * a.lo.sr + d, o0[n][0] / l00, o0[n][1] / l00);
    store_bf16x2(og + (long long)r0_hi * a.lo.sr + d, o0[n][2] / l01, o0[n][3] / l01);
    store_bf16x2(og + (long long)r1_lo * a.lo.sr + d, o1[n][0] / l10, o1[n][1] / l10);
    store_bf16x2(og + (long long)r1_hi * a.lo.sr + d, o1[n][2] / l11, o1[n][3] / l11);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r0_lo) * HD + d) = make_float2(o0[n][0] / l00, o0[n][1] / l00);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r0_hi) * HD + d) = make_float2(o0[n][2] / l01, o0[n][3] / l01);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r1_lo) * HD + d) = make_float2(o1[n][0] / l10, o1[n][1] / l10);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r1_hi) * HD + d) = make_float2(o1[n][2] / l11, o1[n][3] / l11);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r0_lo] = (m00 == -INFINITY ? 0.f : m00 * L2E) + log2f(l00);
    a.lse2[(long long)bh * t + r0_hi] = (m01 == -INFINITY ? 0.f : m01 * L2E) + log2f(l01);
    a.lse2[(long long)bh * t + r1_lo] = (m10 == -INFINITY ? 0.f : m10 * L2E) + log2f(l10);
    a.lse2[(long long)bh * t + r1_hi] = (m11 == -INFINITY ? 0.f : m11 * L2E) + log2f(l11);
  }
  wg_bar(wg);
}


// P9-A: forward_item6 with a four-stage K/V ring and prefetch distance two (the next tile's QK and this tile's
// softmax/PV overlap the loads of the two following tiles). Same tiles, same arithmetic order: bitwise equal.
constexpr int WGB6D = 66 * 1024;
PFLATLONG_DEV void cp_wait2() { asm volatile("cp.async.wait_group 2;\n" ::); }
template <int NS, bool POW2>
PFLATLONG_DEV void forward_item6d(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB6D;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 4 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);  // K-major and transposed descriptors share these fields
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1), aft = next_bit(group, nxt + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  issue(aft, 2);
  cp_commit();
  cp_wait2();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, aft + 1);   // two tiles ahead of nxt (4-stage ring)
    cp_wait1();                                   // nxt has landed; aft may still be in flight
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 3) & 3);
    cp_commit();
    const int ns = (stage + 1) & 3;
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));
      wg_commit();
    }
    if ((mixed >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, sc, m0, m1, l0, l1, o, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, sc, m0, m1, l0, l1, o, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = aft;
    aft = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void forward_item6k(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  // P9-A: this item's head key norms (prepare_unit arithmetic), one warp per key tile, into shared memory.
  float* knt = reinterpret_cast<float*>(ring + 3 * ST4 + 64);
  for (int kt = w4; kt < (t >> 6); kt += 4) {
    const float v = kn_tile(a, bh, kt);
    if (lane == 0) knt[kt] = v;
  }
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane, knt);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);  // K-major and transposed descriptors share these fields
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait0();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));
      wg_commit();
    }
    if ((mixed >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, sc, m0, m1, l0, l1, o, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, sc, m0, m1, l0, l1, o, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}


// P9-A: forward_item6 with speculative first K/V tiles (see the comment inside); same tiles and arithmetic.
template <int NS, bool POW2>
PFLATLONG_DEV void forward_item6s(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  // P9-A: speculative K/V tiles 0 and 1 into stages 0 and 1 while Q lands and the live tiles are proven (they are the
  // first two live tiles of every query block under a causal mask); on a miss they drain and are re-issued.
  const bool spec = (t >> 6) >= 2;
  if (spec) {
    load_sw_wg(ring, kg, a.lk.sr, tid);
    load_sw_wg(ring + SWT, vg, a.lv.sr, tid);
    cp_commit();
    load_sw_wg(ring + ST4, kg + (long long)BT * a.lk.sr, a.lk.sr, tid);
    load_sw_wg(ring + ST4 + SWT, vg + (long long)BT * a.lv.sr, a.lv.sr, tid);
    cp_commit();
    cp_wait2();
  } else {
    cp_wait0();
  }
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);  // K-major and transposed descriptors share these fields
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  if (!(spec && cur == 0 && (nxt == 1 || nxt >= 64))) {
    if (spec) {
      cp_wait0();                                                   // speculative copies drain before reuse
      wg_bar(wg);
    }
    issue(cur, 0);
    cp_commit();
    issue(nxt, 1);
    cp_commit();
  }
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait0();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));
      wg_commit();
    }
    if ((mixed >> cur) & 1ull)
      softmax_tile6<true, POW2>(a, sc, m0, m1, l0, l1, o, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi),
                                (long long)cur * BT, lane);
    else
      softmax_tile6<false, POW2>(a, sc, m0, m1, l0, l1, o, nullptr, nullptr, (long long)cur * BT, lane);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

// dS of one 64 x 64 tile in place: s holds scores, dp holds dP; MASKED reads the mask; POW2 folds the scale.
template <bool MASKED, bool POW2>
PFLATLONG_DEV void ds_rows6(const Args& a, float (&s)[8][4], const float (&dp)[8][4], const float* mlo, const float* mhi,
                        long long col0, float s_lo, float s_hi, float d_lo, float d_hi, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  const float k = FOLD ? sc2<true>(a) : L2E;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = s[n][e];
      if (!FOLD) {
        x = scaled(a, x);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
      }
      const float p = ex2(fmaf(x, k, -(e < 2 ? s_lo : s_hi)));
      s[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
    }
  }
}

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item6(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    wg_fence();
    wg_nt6(s, qa, kd);
    wg_nt6(dp, da, vd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(qa);
    pin(da);
    if ((mixed >> cur) & 1ull)
      ds_rows6<true, POW2>(a, s, dp, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi), (long long)cur * BT, s_lo, s_hi,
                           d_lo, d_hi, lane);
    else
      ds_rows6<false, POW2>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}


// P9-A: delta = rowsum(dO * O) for 16 rows of head bh per warp, with dq_item6's exact per-row arithmetic.
PFLATLONG_DEV void delta_unit6(const Args& a, long long u) {
  const int lane = threadIdx.x & 31, t = a.t;
  const int bh = (int)(u / (t >> 4)), wrow = (int)(u % (t >> 4)) * 16;
  const int r = wrow + (lane >> 1), h = lane & 1;
  const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
  float acc = 0.f;
#ifdef PFLATLONG_O32
  const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
    const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
    const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
    acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
    acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
    acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
    acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
  }
#else
  const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
    const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
    acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
    acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
    acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
    acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
  }
#endif
  const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
  const float dl = h ? other + acc : acc + other;
  if (h == 0) a.delta[(long long)bh * t + r] = dl;
}

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item6n(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    // P9-A merged backward: delta is stored by delta_unit6 (identical arithmetic); this item only uses it.
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    wg_fence();
    wg_nt6(s, qa, kd);
    wg_nt6(dp, da, vd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(qa);
    pin(da);
    if ((mixed >> cur) & 1ull)
      ds_rows6<true, POW2>(a, s, dp, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi), (long long)cur * BT, s_lo, s_hi,
                           d_lo, d_hi, lane);
    else
      ds_rows6<false, POW2>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

// P^T and dS^T of one 64-key x 64-query tile: rows are keys (k_lo, k_hi), columns queries with statistics sl/dl.
template <bool MASKED, bool POW2>
PFLATLONG_DEV void kv_tile6(const Args& a, float (&s)[8][4], float (&dp)[8][4], const float* sl, const float* dl, int bh,
                        int q0, int k_lo, int k_hi, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  const float k = FOLD ? sc2<true>(a) : L2E;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int j = 8 * n + 2 * (lane & 3);
    const float2 lse = *reinterpret_cast<const float2*>(sl + j), del = *reinterpret_cast<const float2*>(dl + j);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = s[n][e];
      if (!FOLD) {
        x = scaled(a, x);
        if (MASKED) x += __ldg(mask_row(a, bh, q0 + j + (e & 1)) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
      }
      const float p = ex2(fmaf(x, k, -((e & 1) ? lse.y : lse.x)));
      s[n][e] = p;
      dp[n][e] = grad_scale(a, p * (dp[n][e] - ((e & 1) ? del.y : del.x)));
    }
  }
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item6(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
#ifdef PFLATLONG_KV6_PTR
    wg_nt(s, ka, ring + stage * ST4, false);
    wg_nt(dp, va, ring + stage * ST4 + SWT, false);
#else
    wg_nt6(s, ka, qd);
    wg_nt6(dp, va, dd);
#endif
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    {
      // Branch-free body: one loop for masked and unmasked tiles (selects, predicated mask loads). Every variant that
      // branched once per tile around code updating both wgmma accumulators produced wrong dK/dV and clobbered dQ
      // (gpu-10..13, kv_tile6 and a per-tile-branch inline body); this form matches v5 bit for bit when POW2 is false.
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    uint32_t pa[4][NS][4], sa[4][NS][4];
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
#ifdef PFLATLONG_KV6_PTR
    wg_pv<NS>(dv, pa, ring + stage * ST4 + SWT);
    wg_pv<NS>(dk, sa, ring + stage * ST4);
#else
    wg_pv6<NS>(dv, pa, dd);
    wg_pv6<NS>(dk, sa, qd);
#endif
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}


// P9-A: kv_item6 with the K and V operands of S = K Q^T and dP = V dO^T read by wgmma from shared memory (swizzled
// tiles kept beside the ring) instead of register fragments: the same products, 32 fewer live registers per thread.
constexpr int WGB6A = 66 * 1024;
PFLATLONG_DEV void wgmma64a(float (&d)[8][4], uint64_t a, uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %32, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
      "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %33, %34, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
        "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "r"(scale_d), "l"(a), "l"(b));
}
PFLATLONG_DEV void wg_nt6a(float (&acc)[8][4], uint64_t a0, uint64_t d0) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) wgmma64a(acc, desc_add(a0, 32 * ks), desc_add(d0, 32 * ks), ks > 0);
}
template <int NS, bool POW2>
PFLATLONG_DEV void kv_item6a(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB6A;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  char* kA = ring + 3 * ST4 + 2048;                                // swizzled K / V tiles: the wgmma A operands
  char* vA = kA + SWT;
  (void)ks;
  (void)vs;
  load_sw_wg(kA, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_sw_wg(vA, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ka_desc = sw_desc(kA, 1, 64), va_desc = sw_desc(vA, 1, 64);
  const unsigned long long mine = flags[0], mixed = flags[1];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
    wg_nt6a(s, ka_desc, qd);
    wg_nt6a(dp, va_desc, dd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    {
      // Branch-free body: one loop for masked and unmasked tiles (selects, predicated mask loads). Every variant that
      // branched once per tile around code updating both wgmma accumulators produced wrong dK/dV and clobbered dQ
      // (gpu-10..13, kv_tile6 and a per-tile-branch inline body); this form matches v5 bit for bit when POW2 is false.
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    uint32_t pa[4][NS][4], sa[4][NS][4];
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
    wg_pv6<NS>(dv, pa, dd);
    wg_pv6<NS>(dk, sa, qd);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// P9-A: dq_item6 with the Q and dO operands read by wgmma from shared memory (same products, fewer registers).
template <int NS, bool POW2>
PFLATLONG_DEV void dq_item6a(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB6A;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  char* qA = ring + 3 * ST4 + 2048;                                // swizzled Q / dO tiles: the wgmma A operands
  char* dA = qA + SWT;
  (void)qs;
  (void)ds;
  load_sw_wg(qA, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_sw_wg(dA, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  proxy_fence();
  wg_bar(wg);
  const uint64_t qa_desc = sw_desc(qA, 1, 64), da_desc = sw_desc(dA, 1, 64);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    wg_fence();
    wg_nt6a(s, qa_desc, kd);
    wg_nt6a(dp, da_desc, vd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    if ((mixed >> cur) & 1ull)
      ds_rows6<true, POW2>(a, s, dp, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi), (long long)cur * BT, s_lo, s_hi,
                           d_lo, d_hi, lane);
    else
      ds_rows6<false, POW2>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}



// P9-A: dq_item6 / kv_item6 with a speculative first tile in the free stage 0 (same tiles and arithmetic).
template <int NS, bool POW2>
PFLATLONG_DEV void dq_item6s(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  // P9-A: speculative key tile 0 into the free stage 0 (the first live tile of every query block under a causal mask).
  if ((t >> 6) >= 1) {
    load_sw_wg(ring, kg, a.lk.sr, tid);
    load_sw_wg(ring + SWT, vg, a.lv.sr, tid);
  }
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  if (cur != 0) issue(cur, 0);                                      // tile 0 already landed in stage 0 (hit)
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    wg_fence();
    wg_nt6(s, qa, kd);
    wg_nt6(dp, da, vd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(qa);
    pin(da);
    if ((mixed >> cur) & 1ull)
      ds_rows6<true, POW2>(a, s, dp, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi), (long long)cur * BT, s_lo, s_hi,
                           d_lo, d_hi, lane);
    else
      ds_rows6<false, POW2>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item6s(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  // P9-A: speculative diagonal query tile kt (Q, dO, lse and delta) into the free stage 0: the first live query tile
  // of every key block under a causal mask.
  {
    const int qt0 = kt;
    load_sw_wg(ring, qg + (long long)(qt0 * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + SWT, dg + (long long)(qt0 * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt0 * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt0 * BT + 4 * (tid - 16));
  }
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  if (cur != kt) issue(cur, 0);                                     // diagonal tile already landed in stage 0 (hit)
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
#ifdef PFLATLONG_KV6_PTR
    wg_nt(s, ka, ring + stage * ST4, false);
    wg_nt(dp, va, ring + stage * ST4 + SWT, false);
#else
    wg_nt6(s, ka, qd);
    wg_nt6(dp, va, dd);
#endif
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    {
      // Branch-free body: one loop for masked and unmasked tiles (selects, predicated mask loads). Every variant that
      // branched once per tile around code updating both wgmma accumulators produced wrong dK/dV and clobbered dQ
      // (gpu-10..13, kv_tile6 and a per-tile-branch inline body); this form matches v5 bit for bit when POW2 is false.
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    uint32_t pa[4][NS][4], sa[4][NS][4];
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
#ifdef PFLATLONG_KV6_PTR
    wg_pv<NS>(dv, pa, ring + stage * ST4 + SWT);
    wg_pv<NS>(dk, sa, ring + stage * ST4);
#else
    wg_pv6<NS>(dv, pa, dd);
    wg_pv6<NS>(dk, sa, qd);
#endif
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v8 backward
// v7 forward; backward items split each 64-wide tile into two 32-wide halves (wgmma m64n32k16 score products) so a
// warpgroup issues the second half's score products, then works on the first half's elementwise step, then issues
// the first half's gradient products before the second half's elementwise step: tensor work overlaps elementwise work
// inside a warpgroup. Arithmetic per element and accumulation order of every output are those of v7 (each k16 step of
// the gradient products is the same wgmma instruction on the same operands, in the same order).
PFLATLONG_DEV void zero4(float (&x)[4][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n) x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f;
}
PFLATLONG_DEV void pin4(float (&x)[4][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(x[n][e])::"memory");
}
PFLATLONG_DEV void wgmma32(float (&d)[4][4], const uint32_t (&a)[4], uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %20, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %21, p, 1, 1, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(b));
}
// acc (64 x 32) = A (64 x 64 registers) * B, B[k][n] = tile rows [32h, 32h + 32) (K-major, 64 k units).
PFLATLONG_DEV void wg_nt_half(float (&acc)[4][4], const uint32_t (&a)[4][4], uint64_t tile, int h) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) wgmma32(acc, a[ks], desc_add(tile, 4096 * h + 32 * ks), ks > 0);
}
// acc (64 x 64) += P (64 x 32, two k16 steps of NS pieces) * B, B[k][n] = tile rows [32h, 32h + 32) (transposed).
template <int NS>
PFLATLONG_DEV void wg_pv_half(float (&acc)[8][4], const uint32_t (&p)[2][NS][4], uint64_t tile, int h) {
#pragma unroll
  for (int kk = 0; kk < 2; ++kk)
#pragma unroll
    for (int piece = 0; piece < NS; ++piece) wgmma64<1>(acc, p[kk][piece], desc_add(tile, 4096 * h + 2048 * kk), 1);
}
template <int K, int P>
PFLATLONG_DEV void pin_a(uint32_t (&x)[K][P][4]) { pin(x); }

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item8(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s0[4][4], dp0[4][4], s1[4][4], dp1[4][4];
  zero8(dq);
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  // Elementwise dS of one half (branch-free; per-tile fold/mask by selects).
  auto ds_half = [&](float (&s)[4][4], const float (&dp)[4][4], bool mx, int cur, int h) {
    const bool fold = POW2 && !mx;
    const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const long long col = (long long)cur * BT + 32 * h + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = fold ? s[n][e] : scaled(a, s[n][e]);
        if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
        const float p = ex2(fmaf(x, kf, -(e < 2 ? s_lo : s_hi)));
        s[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
      }
    }
  };
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    const bool mx = (mixed >> cur) & 1ull;
    wg_fence();
    wg_nt_half(s0, qa, kd, 0);
    wg_nt_half(dp0, da, vd, 0);
    wg_commit();
    wg_nt_half(s1, qa, kd, 1);
    wg_nt_half(dp1, da, vd, 1);
    wg_commit();
    wg_wait<1>();  // first half's scores ready; second half in flight
    pin4(s0);
    pin4(dp0);
    ds_half(s0, dp0, mx, cur, 0);
    uint32_t sa0[2][NS][4];
    to_a<NS, 2, 4>(sa0, s0);
    wg_fence();
    wg_pv_half<NS>(dq, sa0, kd, 0);
    wg_commit();
    wg_wait<1>();  // second half's scores ready; first half's dQ products in flight
    pin4(s1);
    pin4(dp1);
    ds_half(s1, dp1, mx, cur, 1);
    uint32_t sa1[2][NS][4];
    to_a<NS, 2, 4>(sa1, s1);
    wg_fence();
    wg_pv_half<NS>(dq, sa1, kd, 1);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin_a(sa0);
    pin_a(sa1);
    pin(qa);
    pin(da);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item8(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s0[4][4], dp0[4][4], s1[4][4], dp1[4][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  // P^T and dS^T of one 32-query half (branch-free; per-tile fold/mask by selects).
  auto pds_half = [&](float (&s)[4][4], float (&dp)[4][4], const float* sl, const float* dl, bool mx, int cur, int h) {
    const bool fold = POW2 && !mx;
    const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int j = 32 * h + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jj = j + (e & 1);
        float x = fold ? s[n][e] : scaled(a, s[n][e]);
        if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
        const float p = ex2(fmaf(x, kf, -sl[jj]));
        s[n][e] = p;
        dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
      }
    }
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    const bool mx = (mixed >> cur) & 1ull;
    wg_fence();
    wg_nt_half(s0, ka, qd, 0);
    wg_nt_half(dp0, va, dd, 0);
    wg_commit();
    wg_nt_half(s1, ka, qd, 1);
    wg_nt_half(dp1, va, dd, 1);
    wg_commit();
    wg_wait<1>();
    pin4(s0);
    pin4(dp0);
    pds_half(s0, dp0, sl, sl + BT, mx, cur, 0);
    uint32_t pa0[2][NS][4], sa0[2][NS][4];
    to_a<NS, 2, 4>(pa0, s0);
    to_a<NS, 2, 4>(sa0, dp0);
    wg_fence();
    wg_pv_half<NS>(dv, pa0, dd, 0);
    wg_pv_half<NS>(dk, sa0, qd, 0);
    wg_commit();
    wg_wait<1>();
    pin4(s1);
    pin4(dp1);
    pds_half(s1, dp1, sl, sl + BT, mx, cur, 1);
    uint32_t pa1[2][NS][4], sa1[2][NS][4];
    to_a<NS, 2, 4>(pa1, s1);
    to_a<NS, 2, 4>(sa1, dp1);
    wg_fence();
    wg_pv_half<NS>(dv, pa1, dd, 1);
    wg_pv_half<NS>(dk, sa1, qd, 1);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin_a(pa0);
    pin_a(sa0);
    pin_a(pa1);
    pin_a(sa1);
    pin(ka);
    pin(va);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v9 backward
// Stored-dS storage choice (BF16 recipe, NS = 1). Backward = delta phase -> dK/dV phase (v7 item that also stores
// each live BF16 dS^T tile, exactly the values its own dK product consumes) -> dQ phase (dQ = sum over live key tiles
// of dS K from the stored tiles: one product per tile instead of three, no softmax recomputation). Scratch: one
// [BH][T/64][T/64][64][64] BF16 buffer (only live tiles are written or read). Deterministic: every output element
// has one writer and a fixed accumulation order. Under PFLATLONG_DEFER_GSCALE the stored tiles are unscaled dS2 and both
// consumers multiply once by the power-of-two scale, as in v7.
constexpr int DS_TILE = 64 * 64;  // BF16 elements per stored dS^T tile, [key][query] row-major

PFLATLONG_DEV __nv_bfloat16* ds_tile(const Args& a, int bh, int kt, int qt) {
  const int nt = a.t >> 6;
  return a.dsbuf + (((long long)bh * nt + kt) * nt + qt) * DS_TILE;
}

// delta = rowsum(dO * O) for 16 rows per warp unit: two lanes per row, fixed order, pair sum (v1 order).
PFLATLONG_DEV void delta_unit(const Args& a, long long u) {
  const int lane = threadIdx.x & 31, rows = a.t >> 4;
  const int bh = (int)(u / rows), r = (int)(u % rows) * 16 + (lane >> 1), h = lane & 1;
  const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
  const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
    const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
    acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
    acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
    acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
    acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
  }
  const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
  if (h == 0) a.delta[(long long)bh * a.t + r] = acc + other;
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item9(const Args& a, int bh, int kb64, void* smem_v) {
  static_assert(NS == 1, "stored dS is defined for the BF16 recipe");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  const int kr = 16 * w4 + (lane >> 2);  // tile-local key rows kr, kr + 8
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
    wg_nt6(s, ka, qd);
    wg_nt6(dp, va, dd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    {
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    uint32_t pa[4][1][4], sa[4][1][4];
    to_a<1, 4, 8>(pa, s);
    to_a<1, 4, 8>(sa, dp);
    {  // store this tile's BF16 dS^T (the dK operand values) as [key][query]
      __nv_bfloat16* dst = ds_tile(a, bh, kt, cur);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        const int q0 = 16 * kk + 2 * (lane & 3);
        *reinterpret_cast<uint32_t*>(dst + kr * 64 + q0) = sa[kk][0][0];
        *reinterpret_cast<uint32_t*>(dst + (kr + 8) * 64 + q0) = sa[kk][0][1];
        *reinterpret_cast<uint32_t*>(dst + kr * 64 + q0 + 8) = sa[kk][0][2];
        *reinterpret_cast<uint32_t*>(dst + (kr + 8) * 64 + q0 + 8) = sa[kk][0][3];
      }
    }
    wg_fence();
    wg_pv6<1>(dv, pa, dd);
    wg_pv6<1>(dk, sa, qd);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// dQ from stored tiles. Warpgroup region: 3 swizzled K tiles, then 3 padded dS^T tiles [64][LD].
constexpr int WGB9 = 54 * 1024;
__host__ __device__ constexpr int dq9_shared_bytes() { return 1024 + 2 * WGB9; }

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item9(const Args& a, int bh, int qb64, void* smem_v) {
  static_assert(NS == 1, "stored dS is defined for the BF16 recipe");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB9;
  __nv_bfloat16* dsr = reinterpret_cast<__nv_bfloat16*>(ring + 3 * SWT);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(dsr + 3 * BT * LD);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  if (lane == 0) flags[w4] = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * SWT, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_rows_wg(dsr + stage * BT * LD, ds_tile(a, bh, tile, qt), 64, tid);
    }
  };
  float dq[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);  // tile cur landed; the stage read two tiles ago is free
    issue(after, (stage + 2) % 3);
    cp_commit();
    // A = dS (queries x keys) from the stored [key][query] tile: transposed 8x8 loads.
    const __nv_bfloat16* st = dsr + stage * BT * LD;
    uint32_t sa[4][1][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const int mi = lane >> 3;
      ldsm_x4_t(sa[kk][0], st + (16 * kk + (mi >> 1) * 8 + (lane & 7)) * LD + 16 * w4 + (mi & 1) * 8);
    }
    wg_fence();
    wg_pv6<1>(dq, sa, desc_add(ring_desc, stage * SWT));
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v10 schedule
// v7 items with FA3-style intra-warpgroup pipelining (same arithmetic, same accumulation orders). Forward: the value
// products of tile i stay in flight while the softmax statistics of tile i + 1 are computed (the running output is
// rescaled only after those products complete). dQ: the next tile's score/dP products are issued before the current
// tile's elementwise step and the dQ products are waited for one tile later. dK/dV: the gradient products of tile i
// are waited for at the start of tile i + 1. A ring stage is refilled only after every product reading it completed.
template <bool POW2>
PFLATLONG_DEV void softmax_stats(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                             float& c0, float& c1, bool mx, const float* mlo, const float* mhi, long long col0, int lane) {
  const bool fold = POW2 && !mx;
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = fold ? s[n][e] : scaled(a, s[n][e]);
      if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
      s[n][e] = x;
    }
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  if (fold) {
    x0 *= a.scale;
    x1 *= a.scale;
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  c0 = m0 == -INFINITY ? 0.f : ex2(fmaf(m0, L2E, -ml0));
  c1 = m1 == -INFINITY ? 0.f : ex2(fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = fold ? sc2<true>(a) : L2E;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = ex2(fmaf(s[n][0], k, -ml0));
    s[n][1] = ex2(fmaf(s[n][1], k, -ml0));
    s[n][2] = ex2(fmaf(s[n][2], k, -ml1));
    s[n][3] = ex2(fmaf(s[n][3], k, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

template <int NS, bool POW2>
PFLATLONG_DEV void forward_item10(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  uint32_t pa[4][NS][4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int q = 0; q < NS; ++q) pa[k][q][0] = pa[k][q][1] = pa[k][q][2] = pa[k][q][3] = 0u;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
  }
  int stage = 0;
  bool pv_pending = false;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    const int ns = stage == 2 ? 0 : stage + 1;
    cp_wait0();
    proxy_fence();
    wg_bar(wg);  // tile nxt landed and is visible to the tensor cores
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));  // in flight during this tile's softmax statistics
      wg_commit();
    }
    float c0, c1;
    softmax_stats<POW2>(a, sc, m0, m1, l0, l1, c0, c1, (mixed >> cur) & 1ull, mlo, mhi, (long long)cur * BT, lane);
    if (pv_pending) {  // previous tile's value products (committed before the scores above) must finish
      if (nxt < 64) wg_wait<1>();
      else wg_wait<0>();
      pin(o);
      pin(pa);
    }
    wg_bar(wg);  // every warp's products reading the previous stage are complete: refill it
    issue(after, (stage + 2) % 3);
    cp_commit();
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      o[n][0] *= c0;
      o[n][1] *= c0;
      o[n][2] *= c1;
      o[n][3] *= c1;
    }
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    pv_pending = true;
    if (nxt < 64) {
      wg_wait<1>();  // next tile's scores (committed before these value products)
      pin(sn);
      pin(qa);
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  wg_wait<0>();
  pin(o);
  pin(pa);
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
#ifdef PFLATLONG_O32
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_lo) * HD + d) = make_float2(y0, y1);
    *reinterpret_cast<float2*>(a.o32 + ((long long)bh * t + r_hi) * HD + d) = make_float2(y2, y3);
#endif
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item10(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
#ifdef PFLATLONG_O32
    const float* op = a.o32 + ((long long)bh * t + r) * HD + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const float4 z0 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c);
      const float4 z1 = __ldcg(reinterpret_cast<const float4*>(op) + 2 * c + 1);
      acc = fmaf(bf_lo(w.x), z0.x, acc); acc = fmaf(bf_hi(w.x), z0.y, acc);
      acc = fmaf(bf_lo(w.y), z0.z, acc); acc = fmaf(bf_hi(w.y), z0.w, acc);
      acc = fmaf(bf_lo(w.z), z1.x, acc); acc = fmaf(bf_hi(w.z), z1.y, acc);
      acc = fmaf(bf_lo(w.w), z1.z, acc); acc = fmaf(bf_hi(w.w), z1.w, acc);
    }
#else
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
#endif
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_wg(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid);
      load_sw_wg(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float dq[8][4], s[8][4], dp[8][4], sn[8][4], dpn[8][4];
  zero8(dq);
  uint32_t sa[4][NS][4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int q = 0; q < NS; ++q) sa[k][q][0] = sa[k][q][1] = sa[k][q][2] = sa[k][q][3] = 0u;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(s, qa, ring_desc);
    wg_nt6(dp, da, desc_add(ring_desc, SWT));
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
  }
  int stage = 0;
  bool dq_pending = false;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    const int ns = stage == 2 ? 0 : stage + 1;
    cp_wait0();
    proxy_fence();
    wg_bar(wg);  // tile nxt landed
    if (nxt < 64) {
      const uint64_t kn = desc_add(ring_desc, ns * ST4);
      wg_fence();
      wg_nt6(sn, qa, kn);
      wg_nt6(dpn, da, desc_add(kn, SWT));
      wg_commit();
    }
    {
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const long long col = (long long)cur * BT + 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
          const float p = ex2(fmaf(x, kf, -(e < 2 ? s_lo : s_hi)));
          s[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
        }
      }
    }
    if (dq_pending) {
      if (nxt < 64) wg_wait<1>();
      else wg_wait<0>();
      pin(dq);
      pin(sa);
    }
    wg_bar(wg);  // the previous tile's dQ products (reading its K tile) are complete: refill that stage
    issue(after, (stage + 2) % 3);
    cp_commit();
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, desc_add(ring_desc, stage * ST4));
    wg_commit();
    dq_pending = true;
    if (nxt < 64) {
      wg_wait<1>();
      pin(sn);
      pin(dpn);
      pin(qa);
      pin(da);
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          s[n][e] = sn[n][e];
          dp[n][e] = dpn[n][e];
        }
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  wg_wait<0>();
  pin(dq);
  pin(sa);
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item10(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  uint32_t pa[4][NS][4], sa[4][NS][4];
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg * 2 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);  // tile cur landed
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
    wg_nt6(s, ka, qd);   // queued behind the previous tile's gradient products, which were not waited for
    wg_nt6(dp, va, dd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    pin(dk);
    pin(dv);
    wg_bar(wg);  // the previous tile's gradient products are complete: refill its stage
    issue(after, (stage + 2) % 3);
    cp_commit();
    {
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
    wg_pv6<NS>(dv, pa, dd);
    wg_pv6<NS>(dk, sa, qd);
    wg_commit();  // waited for at the next tile (or after the loop)
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  wg_wait<0>();
  pin(dv);
  pin(dk);
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v11 backward
// Single-pass backward (BF16 recipe, NS = 1): delta phase -> dK/dV/dQ phase -> dQ conversion phase; no score
// recomputation for dQ. The dK/dV item (v7 arithmetic for dK and dV) also forms the dQ^T partial K^T dS^T of each live
// tile from the same BF16 dS^T values its dK product consumes, rounds it to int32 fixed point at a per-query
// power-of-two scale f_i and adds it to an accumulator tile with one cp.reduce.async.bulk .add.s32. Integer addition
// is associative, so the sum does not depend on the order in which key blocks arrive: deterministic without ordered
// accumulation or stored dS. Overflow is excluded by construction: f_i = 2^(30 - x_i) with 2^x_i > U_i =
// 4 |dO_i| max|V_k| max|K_k| (norms rounded up; times the score scale bound when above 1), while
// sum_k |dS_ik K_kd| <= (sum_k P_ik)(max_k |dP_ik| + |delta_i|) max_k |K_k| <= ~2.1 |dO_i| max|V| max|K| < U_i, so every
// partial sum stays below 2^30 plus one rounding unit per key block. Each partial rounds once at resolution 1/f_i
// (< 2^-29 U_i). The conversion writes dQ = BF16(sum / f_i [* scale]) with a single rounding (via FP64).
constexpr int WGB11 = 74 * 1024;  // ring 48 KiB, row statistics, dS^T tile at 50 KiB, int32 staging at 58 KiB
__host__ __device__ constexpr int v11_kv_shared_bytes() { return 1024 + 2 * WGB11; }
constexpr int DQ_TILE = 64 * 64;
PFLATLONG_DEV int dq_swz(int d, int q) { return d * 64 + (((q >> 2) ^ (d & 15)) << 2) + (q & 3); }
// Issued by the one thread with `leader` set, as a predicated instruction (no divergent branch near the wgmma
// products: divergent paths can serialize them, see C7520). Waits are executed by every thread (no-ops without
// outstanding bulk groups).
PFLATLONG_DEV void bulk_add_s32(int* g, const void* s, uint32_t bytes, bool leader) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %3, 0;\n"
               "@p cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32 [%0], [%1], %2;\n"
               "@p cp.async.bulk.commit_group;\n}\n"
               ::"l"(g), "r"(smem_u32(s)), "r"(bytes), "r"((int)leader) : "memory");
}
PFLATLONG_DEV void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory"); }
PFLATLONG_DEV void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
PFLATLONG_DEV void proxy_fence_all() { asm volatile("fence.proxy.async;\n" ::: "memory"); }

// delta_i and the fixed-point scale f_i (16 rows per warp unit, v9 delta order); zeroes 1024 accumulator words.
PFLATLONG_DEV void delta_unit11(const Args& a, long long u) {
  const int lane = threadIdx.x & 31, rows = a.t >> 4, nt = a.t >> 6;
  const int bh = (int)(u / rows), r = (int)(u % rows) * 16 + (lane >> 1), h = lane & 1;
  int4* z = reinterpret_cast<int4*>(a.dqacc) + u * 256;
#pragma unroll
  for (int i = 0; i < 8; ++i) z[lane + 32 * i] = make_int4(0, 0, 0, 0);
  const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
  const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
  float acc = 0.f, sq = 0.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
    const uint4 zz = __ldcg(reinterpret_cast<const uint4*>(op) + c);
    acc = fmaf(bf_lo(w.x), bf_lo(zz.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(zz.x), acc);
    acc = fmaf(bf_lo(w.y), bf_lo(zz.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(zz.y), acc);
    acc = fmaf(bf_lo(w.z), bf_lo(zz.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(zz.z), acc);
    acc = fmaf(bf_lo(w.w), bf_lo(zz.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(zz.w), acc);
    sq = fmaf(bf_lo(w.x), bf_lo(w.x), sq); sq = fmaf(bf_hi(w.x), bf_hi(w.x), sq);
    sq = fmaf(bf_lo(w.y), bf_lo(w.y), sq); sq = fmaf(bf_hi(w.y), bf_hi(w.y), sq);
    sq = fmaf(bf_lo(w.z), bf_lo(w.z), sq); sq = fmaf(bf_hi(w.z), bf_hi(w.z), sq);
    sq = fmaf(bf_lo(w.w), bf_lo(w.w), sq); sq = fmaf(bf_hi(w.w), bf_hi(w.w), sq);
  }
  const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
  sq += __shfl_xor_sync(0xffffffffu, sq, 1);
  float km = 0.f, vm = 0.f;
  for (int i = lane; i < nt; i += 32) {
    km = fmaxf(km, a.kn[(long long)bh * nt + i]);
    vm = fmaxf(vm, a.vn[(long long)bh * nt + i]);
  }
  km = warp_max(km);
  vm = warp_max(vm);
  const float bound = 4.0f * (sqrtf(sq) * (1.0f + 1.0f / 256.0f)) * vm * km * fmaxf(1.0f, scale_bound(a));
  int x = 0;
  if (bound > 0.f && bound < INFINITY) frexpf(bound, &x);  // bound < 2^x
  const int e = min(120, max(-120, 30 - x));
  if (h == 0) {
    a.delta[(long long)bh * a.t + r] = acc + other;
    a.dqf[(long long)bh * a.t + r] = ldexpf(1.0f, e);
  }
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item11(const Args& a, int bh, int kb64, void* smem_v) {
  static_assert(NS == 1, "the fixed-point single-pass backward is defined for the BF16 recipe");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB11;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);  // [3][lse2 64, delta 64, f 64]
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 3 * BT);
  char* dst = ring + 50 * 1024;                           // swizzled BF16 dS^T [key][query]
  int* stg = reinterpret_cast<int*>(ring + 58 * 1024);    // int32 dQ^T partial [d][query]
  __nv_bfloat16* kts = reinterpret_cast<__nv_bfloat16*>(ring);  // K^T [d][LD], before the ring starts
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, nt = t >> 6, key0 = kb64 * 64, kt = kb64;
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  proxy_fence_all();  // accumulator zeroing (generic proxy, delta phase) before bulk additions
  load_rows_wg(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid);
  load_rows_wg(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = live_cols(a, bh, kt, lane);
    const unsigned long long mx = mixed_cols(a, bh, kt, lane);
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4], kta[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
#pragma unroll 4
  for (int i = 0; i < 32; ++i) {
    const int idx = tid + 128 * i, key = idx & 63, d = idx >> 6;
    kts[d * LD + key] = ks[key * LD + d];
  }
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  a_frags(kta, kts, 16 * w4, lane);
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64), dst_desc = sw_desc(dst, PFLATLONG_TLBO, PFLATLONG_TSBO);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8, r0 = 16 * w4 + (lane >> 2);
  float dk[8][4], dv[8][4], s[8][4], dp[8][4], dqt[8][4];
  zero8(dk);
  zero8(dv);
  zero8(dqt);
  auto issue = [&](int qt, int stg_i) {
    if (qt >= 64) return;
    load_sw_wg(ring + stg_i * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid);
    load_sw_wg(ring + stg_i * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid);
    float* d = sv + stg_i * 3 * BT;
    if (tid < 16) cp16(d + 4 * tid, a.lse2 + (long long)bh * t + qt * BT + 4 * tid);
    else if (tid < 32) cp16(d + BT + 4 * (tid - 16), a.delta + (long long)bh * t + qt * BT + 4 * (tid - 16));
    else if (tid < 48) cp16(d + 2 * BT + 4 * (tid - 32), a.dqf + (long long)bh * t + qt * BT + 4 * (tid - 32));
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 3 * BT;
    wg_fence();
    wg_nt6(s, ka, qd);
    wg_nt6(dp, va, dd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    {
      // v7 branch-free body (selects, predicated mask loads).
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          if (mx) x += __ldg(mask_row(a, bh, cur * BT + jj) + (long long)(e < 2 ? k_lo : k_hi) * a.m_j);
          const float p = ex2(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    uint32_t pa[4][1][4], sa[4][1][4];
    to_a<1, 4, 8>(pa, s);
    to_a<1, 4, 8>(sa, dp);
    // dS^T (the BF16 values of sa) to the swizzled [key][query] tile: B operand of dQ^T = K^T dS^T.
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int off = ((n ^ (r0 & 7)) << 4) + 4 * (lane & 3);
      *reinterpret_cast<uint32_t*>(dst + r0 * 128 + off) = sa[n >> 1][0][(n & 1) ? 2 : 0];
      *reinterpret_cast<uint32_t*>(dst + (r0 + 8) * 128 + off) = sa[n >> 1][0][(n & 1) ? 3 : 1];
    }
    bulk_wait_read();  // the previous tile's staging has been read by its bulk addition
    proxy_fence();
    wg_bar(wg);
    wg_fence();
    wg_pv6<1>(dv, pa, dd);
    wg_pv6<1>(dk, sa, qd);
#pragma unroll
    for (int ks4 = 0; ks4 < 4; ++ks4) wgmma64<1>(dqt, kta[ks4], desc_add(dst_desc, 2048 * ks4), ks4 > 0);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(dqt);
    pin(pa);
    pin(sa);
    pin(kta);
    {
      const float* fq = sl + 2 * BT;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int q = 8 * n + 2 * (lane & 3);
        const float f0 = fq[q], f1 = fq[q + 1];
        *reinterpret_cast<int2*>(stg + dq_swz(r0, q)) =
            make_int2(__float2int_rn(dqt[n][0] * f0), __float2int_rn(dqt[n][1] * f1));
        *reinterpret_cast<int2*>(stg + dq_swz(r0 + 8, q)) =
            make_int2(__float2int_rn(dqt[n][2] * f0), __float2int_rn(dqt[n][3] * f1));
      }
    }
    proxy_fence();
    wg_bar(wg);
    bulk_add_s32(a.dqacc + ((long long)bh * nt + cur) * DQ_TILE, stg, DQ_TILE * 4, tid == 0);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  bulk_wait_all();
  proxy_fence_all();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dk[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// ====================================================================================================== v12 schedule
// v7 arithmetic class (BF16 recipe, NS = 1) with 128-key steps in the forward and dQ items: score products are
// wgmma m64n128k16 over two adjacent 64-key tiles, the value/dQ products run 8 k16 steps, so the per-step fixed cost
// (pipeline barrier, product waits, row-statistic shuffles, rescale) is paid once per 128 keys instead of per 64.
// A pair is processed when either 64-key tile is live; a dead half contributes exactly +0 (the live-tile proof
// guarantees its exponentials underflow to +0 and that it cannot hold the row maximum). The mask is read for the
// whole pair when either half is mixed (adding a zero mask value is exact). dK/dV keep the v7 item.
constexpr int ST12 = 4 * 64 * 128;  // one ring stage: K rows [128] + V rows [128], swizzled BF16
constexpr int WGB12 = 97 * 1024;    // ring 3 x 32 KiB + flags
__host__ __device__ constexpr int v12_shared_bytes() { return 1024 + 2 * WGB12; }

template <int TNSP>
PFLATLONG_DEV void wgmma128(float (&d)[16][4], const uint32_t (&a)[4], uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %68, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
      "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, "
      "%45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
      "{%64, %65, %66, %67}, %69, p, 1, 1, %70;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
        "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3]), "+f"(d[8][0]), "+f"(d[8][1]), "+f"(d[8][2]),
        "+f"(d[8][3]), "+f"(d[9][0]), "+f"(d[9][1]), "+f"(d[9][2]), "+f"(d[9][3]), "+f"(d[10][0]), "+f"(d[10][1]),
        "+f"(d[10][2]), "+f"(d[10][3]), "+f"(d[11][0]), "+f"(d[11][1]), "+f"(d[11][2]), "+f"(d[11][3]),
        "+f"(d[12][0]), "+f"(d[12][1]), "+f"(d[12][2]), "+f"(d[12][3]), "+f"(d[13][0]), "+f"(d[13][1]),
        "+f"(d[13][2]), "+f"(d[13][3]), "+f"(d[14][0]), "+f"(d[14][1]), "+f"(d[14][2]), "+f"(d[14][3]),
        "+f"(d[15][0]), "+f"(d[15][1]), "+f"(d[15][2]), "+f"(d[15][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(b), "n"(TNSP));
}
PFLATLONG_DEV void pin16(float (&x)[16][4]) {
#pragma unroll
  for (int n = 0; n < 16; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(x[n][e])::"memory");
}
// acc (64 x 128) = A (64 x 64 registers) * B, B[k][n] = rows n of a 128-row K-major tile.
PFLATLONG_DEV void wg_nt12(float (&acc)[16][4], const uint32_t (&a)[4][4], uint64_t d0) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) wgmma128<0>(acc, a[ks], desc_add(d0, 32 * ks), ks > 0);
}
// acc (64 x 64) += P (64 x 128, 8 k16 steps) * B, B[k][n] = rows k of a 128-row transposed tile.
PFLATLONG_DEV void wg_pv12(float (&acc)[8][4], const uint32_t (&p)[8][1][4], uint64_t d0) {
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) wgmma64<1>(acc, p[kk][0], desc_add(d0, 2048 * kk), 1);
}
// 64-key-tile bits -> 128-key pair bits (pair j live when tile 2j or 2j + 1 is).
PFLATLONG_DEV unsigned long long pair_bits(unsigned long long g) {
  unsigned long long p = 0ull;
#pragma unroll
  for (int j = 0; j < 32; ++j) p |= (unsigned long long)(((g >> (2 * j)) & 3ull) != 0ull) << j;
  return p;
}
// 128 swizzled rows (two consecutive 64-row tiles; row r at r * 128 bytes, chunk j at j ^ (r & 7)).
PFLATLONG_DEV void load_sw128_wg(char* dst, const __nv_bfloat16* src, long long stride, int tid) {
  load_sw_wg(dst, src, stride, tid);
  load_sw_wg(dst + 64 * 128, src + 64 * stride, stride, tid);
}

// Online softmax of one m16 x 128 tile pair (softmax_tile6 semantics over 16 n-blocks).
template <bool MASKED, bool POW2>
PFLATLONG_DEV void softmax_tile12(const Args& a, float (&s)[16][4], float& m0, float& m1, float& l0, float& l1,
                              float (&o)[8][4], const float* mlo, const float* mhi, long long col0, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    if (!FOLD) {
      const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = scaled(a, s[n][e]);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
        s[n][e] = x;
      }
    }
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  if (FOLD) {
    x0 *= a.scale;
    x1 *= a.scale;
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  const float c0 = m0 == -INFINITY ? 0.f : ex2(fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : ex2(fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = FOLD ? sc2<true>(a) : L2E;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    s[n][0] = ex2(fmaf(s[n][0], k, -ml0));
    s[n][1] = ex2(fmaf(s[n][1], k, -ml0));
    s[n][2] = ex2(fmaf(s[n][2], k, -ml1));
    s[n][3] = ex2(fmaf(s[n][3], k, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
  }
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    o[n][0] *= c0;
    o[n][1] *= c0;
    o[n][2] *= c1;
    o[n][3] *= c1;
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

template <int NS, bool POW2>
PFLATLONG_DEV void forward_item12(const Args& a, int bh, int qb64, void* smem_v) {
  static_assert(NS == 1, "v12 is defined for the BF16 recipe");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB12;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST12);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST12);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = live_tiles(a, bh, wrow, qs + 16 * w4 * LD, lane);
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) {
    flags[w4] = mine;
    a.live[(long long)bh * (t >> 4) + (wrow >> 4)] = mine;
  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long pairs = pair_bits(flags[0] | flags[1] | flags[2] | flags[3]);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int pr, int stage) {
    if (pr < 64) {
      load_sw128_wg(ring + stage * ST12, kg + (long long)(pr * 128) * a.lk.sr, a.lk.sr, tid);
      load_sw128_wg(ring + stage * ST12 + ST12 / 2, vg + (long long)(pr * 128) * a.lv.sr, a.lv.sr, tid);
    }
  };
  float o[8][4], s[16][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(pairs, 0), nxt = next_bit(pairs, cur + 1);
  wg_bar(wg);  // Q fragments taken before stage 2 (Q staging) is refilled
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(pairs, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST12), vd = desc_add(kd, ST12 / 2);
    wg_fence();
    wg_nt12(s, qa, kd);
    wg_commit();
    wg_wait<0>();
    pin16(s);
    pin(qa);
    if ((mixed >> (2 * cur)) & 3ull)
      softmax_tile12<true, POW2>(a, s, m0, m1, l0, l1, o, mask_row(a, bh, r_lo), mask_row(a, bh, r_hi),
                                 (long long)cur * 128, lane);
    else
      softmax_tile12<false, POW2>(a, s, m0, m1, l0, l1, o, nullptr, nullptr, (long long)cur * 128, lane);
    uint32_t pa[8][1][4];
    to_a<1, 8, 16>(pa, s);
    wg_fence();
    wg_pv12(o, pa, vd);
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(pa);
    pin(qa);
    cur = nxt;
    nxt = after;
    stage = stage == 2 ? 0 : stage + 1;
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(og + (long long)r_lo * a.lo.sr + d, o[n][0] / l0, o[n][1] / l0);
    store_bf16x2(og + (long long)r_hi * a.lo.sr + d, o[n][2] / l1, o[n][3] / l1);
  }
  if ((lane & 3) == 0) {
    a.lse2[(long long)bh * t + r_lo] = (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    a.lse2[(long long)bh * t + r_hi] = (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

// dS of one 64 x 128 tile pair in place (ds_rows6 semantics over 16 n-blocks).
template <bool MASKED, bool POW2>
PFLATLONG_DEV void ds_rows12(const Args& a, float (&s)[16][4], const float (&dp)[16][4], const float* mlo, const float* mhi,
                         long long col0, float s_lo, float s_hi, float d_lo, float d_hi, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  const float k = FOLD ? sc2<true>(a) : L2E;
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = s[n][e];
      if (!FOLD) {
        x = scaled(a, x);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
      }
      const float p = ex2(fmaf(x, k, -(e < 2 ? s_lo : s_hi)));
      s[n][e] = PFLATLONG_GS(POW2, a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
    }
  }
}

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item12(const Args& a, int bh, int qb64, void* smem_v) {
  static_assert(NS == 1, "v12 is defined for the BF16 recipe");
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB12;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST12);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST12);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST12 + ST12 / 2);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_wg(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid);
  load_rows_wg(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid);
  cp_commit();
  const unsigned long long mine = a.live[(long long)bh * (t >> 4) + (wrow >> 4)];
  const unsigned long long mixed = mixed_rows(a, bh, qt, lane);
  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0) a.delta[(long long)bh * t + r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = a.lse2[(long long)bh * t + r_lo], s_hi = a.lse2[(long long)bh * t + r_hi];
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long pairs = pair_bits(flags[0] | flags[1] | flags[2] | flags[3]);
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int pr, int stage) {
    if (pr < 64) {
      load_sw128_wg(ring + stage * ST12, kg + (long long)(pr * 128) * a.lk.sr, a.lk.sr, tid);
      load_sw128_wg(ring + stage * ST12 + ST12 / 2, vg + (long long)(pr * 128) * a.lv.sr, a.lv.sr, tid);
    }
  };
  const float* mlo = mask_row(a, bh, r_lo);
  const float* mhi = mask_row(a, bh, r_hi);
  float dq[8][4], s[16][4], dp[16][4];
  zero8(dq);
  int cur = next_bit(pairs, 0), nxt = next_bit(pairs, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(pairs, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST12), vd = desc_add(kd, ST12 / 2);
    wg_fence();
    wg_nt12(s, qa, kd);
    wg_nt12(dp, da, vd);
    wg_commit();
    wg_wait<0>();
    pin16(s);
    pin16(dp);
    pin(qa);
    pin(da);
    if ((mixed >> (2 * cur)) & 3ull)
      ds_rows12<true, POW2>(a, s, dp, mlo, mhi, (long long)cur * 128, s_lo, s_hi, d_lo, d_hi, lane);
    else
      ds_rows12<false, POW2>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    uint32_t sa[8][1][4];
    to_a<1, 8, 16>(sa, s);
    wg_fence();
    wg_pv12(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    pin(qa);
    pin(da);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
#ifdef PFLATLONG_DEFER_GSCALE
  if (POW2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dq[n][e] *= a.scale;
  }
#endif
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

// dQ = BF16(sum / f_i [* power-of-two scale]) for one (head batch, 64-query tile) per warp unit.
template <bool POW2>
PFLATLONG_DEV void dq_convert_unit11(const Args& a, long long u) {
  const int lane = threadIdx.x & 31, nt = a.t >> 6;
  const int bh = (int)(u / nt), qt = (int)(u % nt);
  const int* acc = a.dqacc + u * DQ_TILE;
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#ifdef PFLATLONG_DEFER_GSCALE
  const double sc = POW2 ? (double)a.scale : 1.0;
#else
  const double sc = 1.0;
#endif
#pragma unroll 1
  for (int half = 0; half < 2; ++half) {
    const int q = lane + 32 * half, row = qt * 64 + q;
    const double inv = (double)(1.0f / a.dqf[(long long)bh * a.t + row]) * sc;
    __nv_bfloat16* out = qg + (long long)row * a.ldq.sr;
#pragma unroll 8
    for (int d = 0; d < 64; d += 2) {
      const double x0 = (double)__ldcg(acc + dq_swz(d, q)) * inv, x1 = (double)__ldcg(acc + dq_swz(d + 1, q)) * inv;
      __nv_bfloat162 hv;
      hv.x = __double2bfloat16(x0);
      hv.y = __double2bfloat16(x1);
      *reinterpret_cast<__nv_bfloat162*>(out + d) = hv;
    }
  }
}

#endif  // PFLATLONG_WGMMA

// Boustrophedon item order over CTAs; items are numbered in decreasing causal cost.
PFLATLONG_DEV long long snake(long long r) {
  const long long g = gridDim.x, b = blockIdx.x;
  return r * g + ((r & 1) ? g - 1 - b : b);
}

}  // namespace plong
