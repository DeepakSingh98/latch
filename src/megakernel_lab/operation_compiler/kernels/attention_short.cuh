// P-OPS3-B short-context training attention work items for the LATCH-OPS cooperative program.
//
// Scope: self-attention with BF16-exact query/key/value/output-gradient operands (they are FP32 widenings of
// BF16 values in the captured graph), head width HD in {32, 64}, context T <= 128, an additive mask tensor with
// broadcast strides and a captured scalar score scale. Outputs are rounded once to BF16 (their captured
// consumers are BF16 casts). The layouts of every operand/output are arbitrary 4-D (batch, head, row, unit)
// strides, so head split/merge copies and dtype conversions around the attention are absorbed.
//
// Arithmetic (declared): tensor-core mma.sync m16n8k16 BF16 x BF16 products with FP32 accumulation from +0.
// Products whose operands are both BF16-exact (Q K^T, dO V^T) are exact. FP32 operands (probabilities P and
// score gradients dS) are split exactly into NS BF16 pieces (NS = 3 reproduces every FP32 product exactly; NS < 3
// is a declared reduced-precision option). Scores are x = fl(fl(dot) op c) + mask; the forward softmax uses expf
// and a correctly rounded division like the captured _softmax and saves the row log-sum-exp and the FP32 output;
// the backward recomputes P = exp(x - lse) (FP32-level difference from the forward P). The softmax derivative is
// dS = P * (dP - delta) as _softmax_backward_data, followed by the captured score scaling. For the dQ rows delta
// is the captured sum_j P_j dP_j over the full row (fixed order); the dK/dV columns, which need every row's delta,
// use delta = rowsum(dO * O_fp32) (mathematically identical, FP32-level difference). Summation orders differ from
// the conventional kernels (FP32 association only). No atomics: every output element has one writer.
//
// Work decomposition: forward item (bh, 16-query block); backward item (bh, 16-row block b) computes the score
// tiles of key block b (all queries) and of query block b (all keys) only, then dV/dK rows of key block b and
// dQ rows of query block b. Requires the program's 256-thread CTA.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace pshort {

#define PSHORT_DEV __device__ __forceinline__
constexpr int NT = 256;
constexpr int NW = 8;
// Warp w's u-th assignment (index w + NW*u) exists; constant when every warp has one, so no convergence
// barriers are needed around the following .sync.aligned matrix instructions.
#define PSHORT_HAS(u, count) (NW * ((u) + 1) <= (count) || warp + NW * (u) < (count))

struct Layout4 {
  long long sb, sh, sr;  // element strides of (batch, head, row); unit stride along the head width
};

struct Args {
  const __nv_bfloat16 *q, *k, *v, *dout;
  Layout4 lq, lk, lv, ld;
  __nv_bfloat16 *o, *dq, *dk, *dv;
  Layout4 lo, ldq, ldk, ldv;
  float* o32;  // forward FP32 output, contiguous [BH][T][HD] (column delta = rowsum(dO * O))
  float* lse;  // forward row log-sum-exp, contiguous [BH][T]
  const float* mask;
  long long m_b, m_i, m_j;  // mask strides over (batch*heads, query, key); zero for broadcast axes
  int has_mask;
  // Release step 09 (defaults keep the original semantics): mask batch = bh / m_heads (per-example masks shared
  // by every head); safe_rows: a row with no allowed key gets output 0 and lse = +inf (backward P = 0 exactly).
  int m_heads = 1;
  int safe_rows = 0;
  int heads, t;
  float scale;
  int scale_mode;  // 0 none, 1 x*scale, 2 x/scale
  int delta_ready;  // P9-B: the delta table of this backward call was written by the dO product's epilogue
  bool practical_exp = false;  // explicit per-helper precision; never selected by a TU-global macro
  int qnorm_ready;  // P9-B: ... and its q-norm block maxima
  int64_t* trace;       // diagnostic SM-cycle stamps [phase items][16] (PSHORT_TRACE builds only)
  int trace_item, trace_offset;
};

#ifndef PSHORT_TRACE
#define PSHORT_TRACE 0
#endif
#define PSHORT_STAMP(slot)                                                                         \
  do {                                                                                             \
    if (PSHORT_TRACE && threadIdx.x == 0)                                                          \
      a.trace[((long long)a.trace_offset + a.trace_item) * 16 + (slot)] = (int64_t)clock64();      \
  } while (0)

PSHORT_DEV long long head_offset(const Layout4& l, int bh, int heads) {
  return (long long)(bh / heads) * l.sb + (long long)(bh % heads) * l.sh;
}

PSHORT_DEV float scaled(const Args& a, float dot) {
  return a.scale_mode == 1 ? dot * a.scale : (a.scale_mode == 2 ? dot / a.scale : dot);
}

PSHORT_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

PSHORT_DEV void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
PSHORT_DEV void cp_commit_wait() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
}
PSHORT_DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
PSHORT_DEV void cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_u32(dst)), "l"(src));
}
PSHORT_DEV void cp_wait_all_but_one() { asm volatile("cp.async.wait_group 1;\n" ::); }

#ifndef PSHORT_CHAINS
#define PSHORT_CHAINS 1  // independent accumulator chains over the k-steps of the P V and stage-B products
#endif

#ifndef PSHORT_SPLIT_GROUPS
#define PSHORT_SPLIT_GROUPS 1  // 1: backward waits for Q/K/V/dO before its products, the rest afterwards
#endif

PSHORT_DEV void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PSHORT_DEV void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PSHORT_DEV void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}
PSHORT_DEV void ldsm_x2_t(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(smem_u32(p)));
}

// Register-only tensor-core product: not volatile, so independent accumulator chains can be interleaved.
PSHORT_DEV void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Exact split of two FP32 values into NS BF16x2 pieces: x = sum_s piece_s (exact for NS = 3).
template <int NS>
PSHORT_DEV void split2(uint32_t (&out)[NS], float x0, float x1) {
#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    out[s] = *reinterpret_cast<const uint32_t*>(&h);
    x0 -= __low2float(h);
    x1 -= __high2float(h);
  }
}

PSHORT_DEV void store_bf16x2(__nv_bfloat16* p, float x0, float x1) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(x0, x1);
}

// Copy rows [0, valid) of a strided BF16 head block (HD units per row) into shared [ROWS][HD + 8]; zero the rest.
// Trip counts are compile-time constants; offsets are 32-bit within the block.
template <int HD, int ROWS, int T0 = 0, int TN = NT>
PSHORT_DEV void load_rows(__nv_bfloat16* dst, const __nv_bfloat16* src, int row_stride, int valid) {
  constexpr int LD = HD + 8, CH = HD / 8, TOTAL = ROWS * CH, ITER = (TOTAL + TN - 1) / TN;
  if (T0 > 0 && (int)threadIdx.x < T0) return;
#pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int idx = (int)threadIdx.x - T0 + k * TN;
    if (TOTAL % TN == 0 || idx < TOTAL) {
      const int r = idx / CH, c = idx % CH;
      if (valid >= ROWS || r < valid) cp16(dst + r * LD + 8 * c, src + r * row_stride + 8 * c);
      else *reinterpret_cast<uint4*>(dst + r * LD + 8 * c) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
}

// Stage mask[row0 + r][col0 + c] (r < ROWS, c < COLS) into shared FP32 [ROWS][COLS + 4]; entries outside the
// context are zero (the score code excludes them separately). 16-byte copies when rows are contiguous.
template <int ROWS, int COLS, int T0 = 0, int TN = NT>
PSHORT_DEV void stage_mask(float* dst, const Args& a, int bh, int row0, int col0) {
  constexpr int LM = COLS + 4;
  const int t = a.t;
  if (!a.has_mask || (T0 > 0 && (int)threadIdx.x < T0)) return;
  const float* base = a.mask + (long long)(bh / a.m_heads) * a.m_b;
  if (a.m_j == 1 && a.m_i % 4 == 0 && a.m_b % 4 == 0 && t % 4 == 0) {
    constexpr int CH = COLS / 4, TOTAL = ROWS * CH, ITER = (TOTAL + TN - 1) / TN;
#pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int idx = (int)threadIdx.x - T0 + k * TN;
      if (TOTAL % TN == 0 || idx < TOTAL) {
        const int r = idx / CH, c = 4 * (idx % CH);
        if (row0 + r < t && col0 + c < t) cp16(dst + r * LM + c, base + (long long)(row0 + r) * a.m_i + col0 + c);
        else *reinterpret_cast<float4*>(dst + r * LM + c) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  } else {
    for (int idx = (int)threadIdx.x - T0; idx < ROWS * COLS; idx += TN) {
      const int r = idx / COLS, c = idx % COLS;
      dst[r * LM + c] = (row0 + r < t && col0 + c < t)
          ? __ldcg(base + (long long)(row0 + r) * a.m_i + (long long)(col0 + c) * a.m_j) : 0.f;
    }
  }
}

// ---------------------------------------------------------------------------------------------------- forward
// Shared bytes: K, V [TP][HD+8], Q block [16][HD+8], P pieces [NS][16][TP+8] (BF16); mask [16][TP+4] and row
// partials [2][8][16] (FP32).
template <int HD, int TP, int NS>
__host__ __device__ constexpr int forward_shared_bytes() {
  return 2 * (2 * TP * (HD + 8) + 16 * (HD + 8) + NS * 16 * (TP + 8)) + 4 * (16 * (TP + 4) + 2 * NW * 16);
}

// Split-K forward (selected per attention by the compiler; PSHORT_FWD_SPLITK=1 forces it): warp (kg, hh) owns keys [16 kg, +16) and head-width half hh. It keeps its
// unnormalized probabilities exp(x - m_kg) in registers as the A operand of its value product and publishes
// (m_kg, l_kg, O_kg); one exchange combines O = sum_kg exp(m_kg - m) O_kg / sum_kg exp(m_kg - m) l_kg (fixed order),
// as split-K/flash-decoding attention does. Two CTA barriers instead of four; FP32-level rounding differences from
// the single-pass softmax (exp of a shifted argument, one extra scale per key group).
#ifndef PSHORT_FWD_SPLITK
#define PSHORT_FWD_SPLITK 0
#endif

template <int HD, int TP>
__host__ __device__ constexpr bool splitk_forward_ok() {
  return TP % 16 == 0 && NW % (TP / 16) == 0 && (HD / 8) % (NW / (TP / 16)) == 0;
}

template <int HD, int TP>
__host__ __device__ constexpr int splitk_forward_shared_bytes() {
  return 2 * (2 * TP * (HD + 8) + 16 * (HD + 8)) + 4 * (16 * (TP + 4) + (TP / 16) * 16 * (HD + 4) + 2 * (TP / 16) * 16);
}

template <int HD, int TP, int NS>
PSHORT_DEV void forward_item_splitk(const Args& a, int bh, int qblk, float* smem) {
  constexpr int LD = HD + 8, LM = TP + 4, LO = HD + 4, KS = HD / 16;
  constexpr int KG = TP / 16, HH = NW / KG, HNT = HD / 8 / HH;  // key groups, head-width parts, n-tiles per part
  __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sv = sk + TP * LD;
  __nv_bfloat16* sq = sv + TP * LD;
  float* sm = reinterpret_cast<float*>(sq + 16 * LD);  // mask rows [16][TP+4]
  float* spo = sm + 16 * LM;                           // partial outputs [KG][16][HD+4]
  float* sml = spo + KG * 16 * LO;                     // [KG][16] maxima, then [KG][16] sums
  const int t = a.t, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int kg = warp % KG, hh = warp / KG, q0 = qblk * 16;
  PSHORT_STAMP(0);
  // Copy groups: [Q block, K, mask rows] for the scores, then [V].
  load_rows<HD, 16>(sq, a.q + head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, (int)a.lq.sr, t - q0);
  load_rows<HD, TP>(sk, a.k + head_offset(a.lk, bh, a.heads), (int)a.lk.sr, t);
  stage_mask<16, TP>(sm, a, bh, q0, 0);
  cp_commit();
  load_rows<HD, TP>(sv, a.v + head_offset(a.lv, bh, a.heads), (int)a.lv.sr, t);
  cp_commit();
  PSHORT_STAMP(1);
  cp_wait_all_but_one();
  __syncthreads();
  PSHORT_STAMP(2);
  // Scores for this warp's 16 keys: two n-tiles, C-fragment order.
  float s[2][4];
  {
    uint32_t qa[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
      ldsm_x4(qa[ks], sq + (lane & 15) * LD + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int nt = 2 * kg + u;
      s[u][0] = s[u][1] = s[u][2] = s[u][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t kb[2];
        ldsm_x2(kb, sk + (nt * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
        mma(s[u], qa[ks], kb);
      }
    }
  }
  float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = (2 * kg + u) * 8 + 2 * tig + (e & 1);
      const float m = a.has_mask ? sm[(g + (e < 2 ? 0 : 8)) * LM + col] : 0.f;
      const float x = col < t ? scaled(a, s[u][e]) + m : -INFINITY;
      s[u][e] = x;
      if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
    }
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int u = 0; u < 2; ++u) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x = s[u][e];
      const float ex = x == -INFINITY ? 0.f : expf(x - (e < 2 ? mx0 : mx1));
      s[u][e] = ex;
      if (e < 2) l0 += ex; else l1 += ex;
    }
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  // A operand (16 rows x 16 keys) of the unnormalized probabilities, split exactly into NS BF16 pieces.
  uint32_t pa[NS][4];
  {
    uint32_t r0[NS], r1[NS], r2[NS], r3[NS];
    split2<NS>(r0, s[0][0], s[0][1]);
    split2<NS>(r1, s[0][2], s[0][3]);
    split2<NS>(r2, s[1][0], s[1][1]);
    split2<NS>(r3, s[1][2], s[1][3]);
#pragma unroll
    for (int sp_i = 0; sp_i < NS; ++sp_i) {
      pa[sp_i][0] = r0[sp_i]; pa[sp_i][1] = r1[sp_i]; pa[sp_i][2] = r2[sp_i]; pa[sp_i][3] = r3[sp_i];
    }
  }
  PSHORT_STAMP(3);
  cp_commit_wait();  // V
  __syncthreads();
  PSHORT_STAMP(4);
  // Partial value product over this warp's keys for its head-width part.
#pragma unroll
  for (int v = 0; v < HNT; ++v) {
    const int nt = hh * HNT + v;
    uint32_t vb[2];
    ldsm_x2_t(vb, sv + (16 * kg + (lane & 15)) * LD + nt * 8);
    float acc[NS][4];
#pragma unroll
    for (int sp_i = 0; sp_i < NS; ++sp_i) {
      acc[sp_i][0] = acc[sp_i][1] = acc[sp_i][2] = acc[sp_i][3] = 0.f;
      mma(acc[sp_i], pa[sp_i], vb);
    }
#pragma unroll
    for (int sp_i = 1; sp_i < NS; ++sp_i)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[0][e] += acc[sp_i][e];
    float* dst = spo + (kg * 16) * LO + nt * 8 + 2 * tig;
    *reinterpret_cast<float2*>(dst + g * LO) = make_float2(acc[0][0], acc[0][1]);
    *reinterpret_cast<float2*>(dst + (g + 8) * LO) = make_float2(acc[0][2], acc[0][3]);
  }
  if (hh == 0 && tig == 0) {
    sml[kg * 16 + g] = mx0; sml[kg * 16 + g + 8] = mx1;
    sml[KG * 16 + kg * 16 + g] = l0; sml[KG * 16 + kg * 16 + g + 8] = l1;
  }
  PSHORT_STAMP(5);
  __syncthreads();
  PSHORT_STAMP(6);
  // Combine: thread handles one row and HD / 16 consecutive head-width units (fixed key-group order).
  constexpr int CPT = HD / 16;
  const int row = threadIdx.x >> 4, c0 = (threadIdx.x & 15) * CPT, qrow = q0 + row;
  float m = sml[row];
#pragma unroll
  for (int k = 1; k < KG; ++k) m = fmaxf(m, sml[k * 16 + row]);
  float scale_k[KG], total = 0.f;
#pragma unroll
  for (int k = 0; k < KG; ++k) {
    const float mk = sml[k * 16 + row];
    scale_k[k] = mk == -INFINITY ? 0.f : expf(mk - m);
    total += scale_k[k] * sml[KG * 16 + k * 16 + row];
  }
  float o[CPT];
#pragma unroll
  for (int c = 0; c < CPT; ++c) o[c] = 0.f;
#pragma unroll
  for (int k = 0; k < KG; ++k) {
#pragma unroll
    for (int c = 0; c < CPT; ++c) o[c] += scale_k[k] * spo[(k * 16 + row) * LO + c0 + c];
  }
  const bool empty = a.safe_rows && total == 0.f;  // every key forbidden: o is exactly 0
  if (empty) total = 1.f;
#pragma unroll
  for (int c = 0; c < CPT; ++c) o[c] /= total;
  if (qrow < t) {
    __nv_bfloat16* ob = a.o + head_offset(a.lo, bh, a.heads) + (long long)qrow * a.lo.sr + c0;
    float* o32 = a.o32 + ((long long)bh * t + qrow) * HD + c0;
#pragma unroll
    for (int c = 0; c < CPT; c += 2) {
      store_bf16x2(ob + c, o[c], o[c + 1]);
      *reinterpret_cast<float2*>(o32 + c) = make_float2(o[c], o[c + 1]);
    }
    // Row log-sum-exp for the backward recomputation P = exp(x - lse).
    if ((threadIdx.x & 15) == 0) a.lse[(long long)bh * t + qrow] = empty ? INFINITY : m + logf(total);
  }
  PSHORT_STAMP(7);
  if (PSHORT_TRACE && threadIdx.x == 0) a.trace[((long long)a.trace_offset + a.trace_item) * 16 + 15] = 7;
}

template <int HD, int TP, int NS, bool SPLITK = false>
PSHORT_DEV void forward_item(const Args& a, int bh, int qblk, float* smem) {
  if constexpr ((SPLITK || PSHORT_FWD_SPLITK) && splitk_forward_ok<HD, TP>()) {
    forward_item_splitk<HD, TP, NS>(a, bh, qblk, smem);
    return;
  }
  constexpr int LD = HD + 8, LP = TP + 8, LM = TP + 4, NKT = TP / 8, NHT = HD / 8, KS = HD / 16;
  __nv_bfloat16* sk = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* sv = sk + TP * LD;
  __nv_bfloat16* sq = sv + TP * LD;
  __nv_bfloat16* sp = sq + 16 * LD;
  float* sm = reinterpret_cast<float*>(sp + NS * 16 * LP);  // mask rows [16][TP+4]
  float* red = sm + 16 * LM;                               // [2][NW][16]
  const int t = a.t, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int q0 = qblk * 16;
  PSHORT_STAMP(0);
  // Copy groups: [Q block, K, mask rows] for the scores, then [V] for the value product.
  load_rows<HD, 16>(sq, a.q + head_offset(a.lq, bh, a.heads) + (long long)q0 * a.lq.sr, (int)a.lq.sr, t - q0);
  load_rows<HD, TP>(sk, a.k + head_offset(a.lk, bh, a.heads), (int)a.lk.sr, t);
  stage_mask<16, TP>(sm, a, bh, q0, 0);
  cp_commit();
  PSHORT_STAMP(1);
  load_rows<HD, TP>(sv, a.v + head_offset(a.lv, bh, a.heads), (int)a.lv.sr, t);
  cp_commit();
  PSHORT_STAMP(2);
  cp_wait_all_but_one();
  __syncthreads();
  PSHORT_STAMP(3);
  constexpr int MAXT = (NKT + NW - 1) / NW;
  float s[MAXT][4];
  const int r0 = q0 + g, r1 = q0 + g + 8;
  float mx0 = -INFINITY, mx1 = -INFINITY;
  {
    uint32_t qa[KS][4];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
      ldsm_x4(qa[ks], sq + (lane & 15) * LD + ks * 16 + (lane >> 4) * 8);
#pragma unroll
    for (int u = 0; u < MAXT; ++u) {
      const int nt = warp + NW * u;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      if (PSHORT_HAS(u, NKT)) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
          uint32_t kb[2];
          ldsm_x2(kb, sk + (nt * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
          mma(acc, qa[ks], kb);
        }
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = nt * 8 + 2 * tig + (e & 1);
        const float m = a.has_mask ? sm[(g + (e < 2 ? 0 : 8)) * LM + col] : 0.f;
        const float x = (PSHORT_HAS(u, NKT) && col < t) ? scaled(a, acc[e]) + m : -INFINITY;
        s[u][e] = x;
        if (e < 2) mx0 = fmaxf(mx0, x); else mx1 = fmaxf(mx1, x);
      }
    }
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) {
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
  }
  if (tig == 0) { red[warp * 16 + g] = mx0; red[warp * 16 + g + 8] = mx1; }
  PSHORT_STAMP(4);
  __syncthreads();
  PSHORT_STAMP(5);
  mx0 = red[g]; mx1 = red[g + 8];
#pragma unroll
  for (int w = 1; w < NW; ++w) { mx0 = fmaxf(mx0, red[w * 16 + g]); mx1 = fmaxf(mx1, red[w * 16 + g + 8]); }
  if (a.safe_rows) {  // a row with no allowed key: every x is -inf; shift by 0 so each exp is exactly 0
    if (mx0 == -INFINITY) mx0 = 0.f;
    if (mx1 == -INFINITY) mx1 = 0.f;
  }
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int u = 0; u < MAXT; ++u) {
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float ex = expf(s[u][e] - (e < 2 ? mx0 : mx1));
      s[u][e] = ex;
      if (e < 2) sum0 += ex; else sum1 += ex;
    }
  }
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) {
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, o);
    sum1 += __shfl_xor_sync(0xffffffffu, sum1, o);
  }
  if (tig == 0) { red[NW * 16 + warp * 16 + g] = sum0; red[NW * 16 + warp * 16 + g + 8] = sum1; }
  PSHORT_STAMP(6);
  __syncthreads();
  PSHORT_STAMP(7);
  sum0 = red[NW * 16 + g]; sum1 = red[NW * 16 + g + 8];
#pragma unroll
  for (int w = 1; w < NW; ++w) { sum0 += red[NW * 16 + w * 16 + g]; sum1 += red[NW * 16 + w * 16 + g + 8]; }
  const bool empty0 = a.safe_rows && sum0 == 0.f, empty1 = a.safe_rows && sum1 == 0.f;
  if (empty0) sum0 = 1.f;  // probabilities are exactly 0; the output row is 0
  if (empty1) sum1 = 1.f;
#pragma unroll
  for (int u = 0; u < MAXT; ++u) {
    const int nt = warp + NW * u;
    if (PSHORT_HAS(u, NKT)) {
      uint32_t p0[NS], p1[NS];
      split2<NS>(p0, s[u][0] / sum0, s[u][1] / sum0);
      split2<NS>(p1, s[u][2] / sum1, s[u][3] / sum1);
#pragma unroll
      for (int sp_i = 0; sp_i < NS; ++sp_i) {
        *reinterpret_cast<uint32_t*>(sp + sp_i * 16 * LP + g * LP + nt * 8 + 2 * tig) = p0[sp_i];
        *reinterpret_cast<uint32_t*>(sp + sp_i * 16 * LP + (g + 8) * LP + nt * 8 + 2 * tig) = p1[sp_i];
      }
    }
  }
  PSHORT_STAMP(8);
  cp_commit_wait();  // V
  __syncthreads();
  PSHORT_STAMP(9);
  // O block = P V: warp owns head-width n-tiles warp, warp + 8; one accumulator chain per BF16 piece.
  __nv_bfloat16* ob = a.o + head_offset(a.lo, bh, a.heads);
  float* o32 = a.o32 + (long long)bh * t * HD;
#pragma unroll
  for (int v = 0; v < (NHT + NW - 1) / NW; ++v) {
    const int nt = warp + NW * v;
    if (!PSHORT_HAS(v, NHT)) continue;
    constexpr int KC = PSHORT_CHAINS;
    float acc[NS * KC][4];
#pragma unroll
    for (int c = 0; c < NS * KC; ++c) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < TP / 16; ++ks) {
      uint32_t vb[2];
      ldsm_x2_t(vb, sv + (ks * 16 + (lane & 15)) * LD + nt * 8);
#pragma unroll
      for (int sp_i = 0; sp_i < NS; ++sp_i) {
        uint32_t pa[4];
        ldsm_x4(pa, sp + sp_i * 16 * LP + (lane & 15) * LP + ks * 16 + (lane >> 4) * 8);
        mma(acc[sp_i * KC + ks % KC], pa, vb);
      }
    }
#pragma unroll
    for (int c = 1; c < NS * KC; ++c)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[0][e] += acc[c][e];
    PSHORT_STAMP(10);
    const int col = nt * 8 + 2 * tig;
    if (r0 < t) {
      store_bf16x2(ob + (long long)r0 * a.lo.sr + col, acc[0][0], acc[0][1]);
      *reinterpret_cast<float2*>(o32 + r0 * HD + col) = make_float2(acc[0][0], acc[0][1]);
    }
    if (r1 < t) {
      store_bf16x2(ob + (long long)r1 * a.lo.sr + col, acc[0][2], acc[0][3]);
      *reinterpret_cast<float2*>(o32 + r1 * HD + col) = make_float2(acc[0][2], acc[0][3]);
    }
  }
  PSHORT_STAMP(11);
  if (PSHORT_TRACE && threadIdx.x == 0) a.trace[((long long)a.trace_offset + a.trace_item) * 16 + 15] = 11;
  if (warp == NW - 1 && tig == 0) {
    // Row log-sum-exp for the backward recomputation P = exp(x - lse) (off the P barrier's critical path).
    if (r0 < t) a.lse[(long long)bh * t + r0] = empty0 ? INFINITY : mx0 + logf(sum0);
    if (r1 < t) a.lse[(long long)bh * t + r1] = empty1 ? INFINITY : mx1 + logf(sum1);
  }
}

// --------------------------------------------------------------------------------------------------- backward
template <int TP, int HD>
__host__ __device__ constexpr int delta_threads() {
  int r = 1;
  while (r * 2 * TP <= NT && r * 2 <= HD / 4) r *= 2;
  return r;
}

// Shared (bytes): Q, K, V, dO [TP][HD+8] BF16; one region holding first the FP32 forward output [TP][HD+4] (dead
// after the column delta) and then the P and dS key-block columns [NS][TP][24] and dS query-block rows
// [NS][16][TP+8] (BF16); masks (staged for TP <= 96): key-block columns [TP][20], query-block rows [16][TP+4];
// per-row lse and delta [2][TP]; exact query-row delta partials [TP/8][16].
template <int HD, int TP, int NS>
__host__ __device__ constexpr int piece_bytes() {
  return 2 * (2 * NS * TP * 24 + NS * 16 * (TP + 8));
}
template <int HD, int TP, int NS>
__host__ __device__ constexpr int union_bytes() {
  return piece_bytes<HD, TP, NS>() > 4 * TP * (HD + 4) ? piece_bytes<HD, TP, NS>() : 4 * TP * (HD + 4);
}
template <int TP>
__host__ __device__ constexpr bool stage_backward_mask() { return TP <= 96; }
template <int HD, int TP, int NS>
__host__ __device__ constexpr int backward_shared_bytes() {
  return 2 * 4 * TP * (HD + 8) + union_bytes<HD, TP, NS>()
       + 4 * ((stage_backward_mask<TP>() ? TP * 20 + 16 * (TP + 4) : 0) + 2 * TP + (TP / 8) * 16);
}

// MMA part of a 16x8 score tile (query rows [16*rb, +16), keys [8*nt, +8)): raw scores and dP = dO V^T.
template <int HD>
PSHORT_DEV void tile_products(const __nv_bfloat16* sq, const __nv_bfloat16* sk, const __nv_bfloat16* sv,
                              const __nv_bfloat16* sd, int rb, int nt, float (&sc)[4], float (&dp)[4]) {
  constexpr int LD = HD + 8, KS = HD / 16;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int e = 0; e < 4; ++e) sc[e] = dp[e] = 0.f;
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t qa[4], da[4], kb[2], vb[2];
    ldsm_x4(qa, sq + (16 * rb + (lane & 15)) * LD + ks * 16 + (lane >> 4) * 8);
    ldsm_x4(da, sd + (16 * rb + (lane & 15)) * LD + ks * 16 + (lane >> 4) * 8);
    ldsm_x2(kb, sk + (nt * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
    ldsm_x2(vb, sv + (nt * 8 + (lane & 7)) * LD + ks * 16 + ((lane >> 3) & 1) * 8);
    mma(sc, qa, kb);
    mma(dp, da, vb);
  }
}

// Backward shared-memory layout (see backward_shared_bytes).
template <int HD, int TP, int NS>
struct BwdSmem {
  static constexpr int LD = HD + 8, LP = TP + 8, LC = 24, LO = HD + 4, LMC = 20, LMR = TP + 4;
  static constexpr bool SM = stage_backward_mask<TP>();
  __nv_bfloat16 *sq, *sk, *sv, *sd, *spc, *ssc, *ssr;
  float *so, *smc, *smr, *slse, *sdel, *sred;
  PSHORT_DEV explicit BwdSmem(float* smem) {
    sq = reinterpret_cast<__nv_bfloat16*>(smem);
    sk = sq + TP * LD;
    sv = sk + TP * LD;
    sd = sv + TP * LD;
    spc = sd + TP * LD;             // P[:, block] pieces, rows = queries
    ssc = spc + NS * TP * LC;       // dS[:, block] pieces
    ssr = ssc + NS * TP * LC;       // dS[block, :] pieces
    so = reinterpret_cast<float*>(spc);  // FP32 forward output rows (aliases the pieces; dead before them)
    smc = reinterpret_cast<float*>(reinterpret_cast<char*>(spc) + union_bytes<HD, TP, NS>());
    smr = smc + (SM ? TP * LMC : 0);     // mask, query-block rows
    slse = smr + (SM ? 16 * LMR : 0);
    sdel = slse + TP;                    // column delta (rowsum dO * O)
    sred = sdel + TP;                    // exact delta partials of the query-block rows [NKT][16]
  }
};

// Operand bits: forward-independent backward operands that a preceding phase may copy early.
enum { PF_Q = 1, PF_K = 2, PF_V = 4, PF_DOUT = 8, PF_MASK = 16, PF_TILES = 15 };

template <int HD, int TP, int NS, int T0 = 0, int TN = NT>
PSHORT_DEV void backward_operands(const Args& a, int bh, int blk, const BwdSmem<HD, TP, NS>& L, int bits) {
  const int t = a.t;
  if (bits & PF_Q) load_rows<HD, TP, T0, TN>(L.sq, a.q + head_offset(a.lq, bh, a.heads), (int)a.lq.sr, t);
  if (bits & PF_K) load_rows<HD, TP, T0, TN>(L.sk, a.k + head_offset(a.lk, bh, a.heads), (int)a.lk.sr, t);
  if (bits & PF_V) load_rows<HD, TP, T0, TN>(L.sv, a.v + head_offset(a.lv, bh, a.heads), (int)a.lv.sr, t);
  if (bits & PF_DOUT) load_rows<HD, TP, T0, TN>(L.sd, a.dout + head_offset(a.ld, bh, a.heads), (int)a.ld.sr, t);
  if ((bits & PF_MASK) && L.SM) {
    stage_mask<TP, 16, T0, TN>(L.smc, a, bh, 0, 16 * blk);
    stage_mask<16, TP, T0, TN>(L.smr, a, bh, 16 * blk, 0);
  }
}

// Issued at the end of the preceding phase for this CTA's first backward item; completes during the barrier.
// Warp 0 issues nothing: the grid barrier's release fence is executed by thread 0 and would otherwise wait for
// its own outstanding copies, moving their latency into the barrier.
template <int HD, int TP, int NS>
PSHORT_DEV void backward_prefetch(const Args& a, int bh, int blk, float* smem, int bits) {
  const BwdSmem<HD, TP, NS> L(smem);
  backward_operands<HD, TP, NS, 32, NT - 32>(a, bh, blk, L, bits);
  cp_commit();
}

template <int HD, int TP, int NS>
PSHORT_DEV void backward_item(const Args& a, int bh, int blk, float* smem, int prefetched) {
  constexpr int LD = HD + 8, LP = TP + 8, LC = 24, LO = HD + 4, LMC = 20, LMR = TP + 4;
  constexpr int NHT = HD / 8, NB = TP / 16, NKT = TP / 8;
  constexpr int RPW = (NKT + NW - 1) / NW, CPW = (2 * NB + NW - 1) / NW;  // row / column tiles per warp
  const BwdSmem<HD, TP, NS> L(smem);
  __nv_bfloat16 *sq = L.sq, *sk = L.sk, *sv = L.sv, *sd = L.sd, *spc = L.spc, *ssc = L.ssc, *ssr = L.ssr;
  float *so = L.so, *smc = L.smc, *smr = L.smr, *slse = L.slse, *sdel = L.sdel, *sred = L.sred;
  constexpr bool SM = BwdSmem<HD, TP, NS>::SM;
  const int t = a.t, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, tig = lane & 3;
  const int k0 = 16 * blk;
  PSHORT_STAMP(0);
  // Commit groups: [prefetched operands (earlier phase)] [remaining Q/K/V/dO] [O32, masks, lse].
  backward_operands<HD, TP, NS>(a, bh, blk, L, PF_TILES & ~prefetched);
  if (PSHORT_SPLIT_GROUPS) cp_commit();
  PSHORT_STAMP(1);
  {
    constexpr int CH = HD / 4, TOTAL = TP * CH, ITER = (TOTAL + NT - 1) / NT;
    const float* src = a.o32 + (long long)bh * t * HD;
#pragma unroll
    for (int k = 0; k < ITER; ++k) {
      const int idx = threadIdx.x + k * NT;
      if (TOTAL % NT == 0 || idx < TOTAL) {
        const int r = idx / CH, c = 4 * (idx % CH);
        if (t >= TP || r < t) cp16(so + r * LO + c, src + r * HD + c);
        else *reinterpret_cast<float4*>(so + r * LO + c) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }
  PSHORT_STAMP(2);
  backward_operands<HD, TP, NS>(a, bh, blk, L, PF_MASK & ~prefetched);
  PSHORT_STAMP(3);
  const float* gmask = a.mask + (long long)(bh / a.m_heads) * a.m_b;
  {
    // Row log-sum-exp joins the second copy group (a synchronous load would queue behind every copy in flight).
    const float* lrow = a.lse + (long long)bh * t;
    if (t % 4 == 0) {
      if ((int)threadIdx.x < TP / 4) {
        if (4 * (int)threadIdx.x < t) cp16(slse + 4 * threadIdx.x, lrow + 4 * threadIdx.x);
        else *reinterpret_cast<float4*>(slse + 4 * threadIdx.x) = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    } else if ((int)threadIdx.x < TP) {
      if ((int)threadIdx.x < t) cp4(slse + threadIdx.x, lrow + threadIdx.x);
      else slse[threadIdx.x] = 0.f;
    }
  }
  PSHORT_STAMP(4);
  if (PSHORT_SPLIT_GROUPS) {
    cp_commit();
    cp_wait_all_but_one();
  } else {
    cp_commit_wait();
  }
  __syncthreads();
  PSHORT_STAMP(5);
  // Score/gradient products of this warp's tiles (independent of delta).
  float rs[RPW][4], rd[RPW][4], cs[CPW][4], cd[CPW][4];
#pragma unroll
  for (int u = 0; u < RPW; ++u)
    if (PSHORT_HAS(u, NKT)) tile_products<HD>(sq, sk, sv, sd, blk, warp + NW * u, rs[u], rd[u]);
#pragma unroll
  for (int u = 0; u < CPW; ++u) {
    const int tile = warp + NW * u;
    if (PSHORT_HAS(u, 2 * NB)) tile_products<HD>(sq, sk, sv, sd, tile % NB, 2 * blk + tile / NB, cs[u], cd[u]);
  }
  PSHORT_STAMP(6);
  if (PSHORT_SPLIT_GROUPS) {
    cp_commit_wait();
    __syncthreads();
  }
  PSHORT_STAMP(7);
  {
    // Column delta_r = sum_d dO[r][d] * O[r][d] (equals sum_j P_rj dP_rj); FP32 products, fixed order.
    constexpr int TPR = delta_threads<TP, HD>(), EPT = HD / TPR;
    const int drow = threadIdx.x / TPR, dpart = threadIdx.x % TPR;
    float acc = 0.f;
    if (drow < TP) {
      const __nv_bfloat16* dsrc = sd + drow * LD + dpart * EPT;
      const float* osrc = so + drow * LO + dpart * EPT;
#pragma unroll
      for (int e = 0; e < EPT; ++e) acc += __bfloat162float(dsrc[e]) * osrc[e];
    }
#pragma unroll
    for (int o = 1; o < TPR; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (dpart == 0 && drow < TP) sdel[drow] = acc;
  }
  // Query-block rows: P = exp(x - lse) and exact partial sums of P * dP over each row tile's keys.
  const int qr0 = k0 + g, qr1 = qr0 + 8;
#pragma unroll
  for (int u = 0; u < RPW; ++u) {
    const int nt = warp + NW * u;
    if (!PSHORT_HAS(u, NKT)) continue;
    float d0 = 0.f, d1 = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = e < 2 ? qr0 : qr1, col = nt * 8 + 2 * tig + (e & 1);
      const float m = !a.has_mask ? 0.f : SM ? smr[(g + (e < 2 ? 0 : 8)) * LMR + col]
          : (row < t && col < t ? __ldcg(gmask + (long long)row * a.m_i + (long long)col * a.m_j) : 0.f);
      const float x = scaled(a, rs[u][e]) + m;
      const float p = (col < t && row < t) ? expf(x - slse[row]) : 0.f;
      rs[u][e] = p;
      if (e < 2) d0 += p * rd[u][e]; else d1 += p * rd[u][e];
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      d0 += __shfl_xor_sync(0xffffffffu, d0, o);
      d1 += __shfl_xor_sync(0xffffffffu, d1, o);
    }
    if (tig == 0) { sred[nt * 16 + g] = d0; sred[nt * 16 + g + 8] = d1; }
  }
  PSHORT_STAMP(8);
  __syncthreads();
  PSHORT_STAMP(9);
  {
    float dq0 = 0.f, dq1 = 0.f;  // exact query-row delta, fixed tile order
#pragma unroll
    for (int nt = 0; nt < NKT; ++nt) { dq0 += sred[nt * 16 + g]; dq1 += sred[nt * 16 + g + 8]; }
#pragma unroll
    for (int u = 0; u < RPW; ++u) {
      const int nt = warp + NW * u;
      if (!PSHORT_HAS(u, NKT)) continue;
      uint32_t ga[NS], gb[NS];
      split2<NS>(ga, scaled(a, rs[u][0] * (rd[u][0] - dq0)), scaled(a, rs[u][1] * (rd[u][1] - dq0)));
      split2<NS>(gb, scaled(a, rs[u][2] * (rd[u][2] - dq1)), scaled(a, rs[u][3] * (rd[u][3] - dq1)));
      const int col = nt * 8 + 2 * tig;
#pragma unroll
      for (int sp_i = 0; sp_i < NS; ++sp_i) {
        *reinterpret_cast<uint32_t*>(ssr + sp_i * 16 * LP + g * LP + col) = ga[sp_i];
        *reinterpret_cast<uint32_t*>(ssr + sp_i * 16 * LP + (g + 8) * LP + col) = gb[sp_i];
      }
    }
  }
#pragma unroll
  for (int u = 0; u < CPW; ++u) {
    const int tile = warp + NW * u;
    if (!PSHORT_HAS(u, 2 * NB)) continue;
    const int rb = tile % NB, nt = 2 * blk + tile / NB, q0 = 16 * rb + g;
    float p[4], ds[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = q0 + (e < 2 ? 0 : 8), col = nt * 8 + 2 * tig + (e & 1);
      const float m = !a.has_mask ? 0.f : SM ? smc[row * LMC + col - k0]
          : (row < t && col < t ? __ldcg(gmask + (long long)row * a.m_i + (long long)col * a.m_j) : 0.f);
      const float x = scaled(a, cs[u][e]) + m;
      p[e] = (col < t && row < t) ? expf(x - slse[row]) : 0.f;
      ds[e] = scaled(a, p[e] * (cd[u][e] - sdel[row]));
    }
    uint32_t pa[NS], pb[NS], ga[NS], gb[NS];
    split2<NS>(pa, p[0], p[1]);
    split2<NS>(pb, p[2], p[3]);
    split2<NS>(ga, ds[0], ds[1]);
    split2<NS>(gb, ds[2], ds[3]);
    const int cc = nt * 8 - k0 + 2 * tig;
#pragma unroll
    for (int sp_i = 0; sp_i < NS; ++sp_i) {
      *reinterpret_cast<uint32_t*>(spc + sp_i * TP * LC + q0 * LC + cc) = pa[sp_i];
      *reinterpret_cast<uint32_t*>(spc + sp_i * TP * LC + (q0 + 8) * LC + cc) = pb[sp_i];
      *reinterpret_cast<uint32_t*>(ssc + sp_i * TP * LC + q0 * LC + cc) = ga[sp_i];
      *reinterpret_cast<uint32_t*>(ssc + sp_i * TP * LC + (q0 + 8) * LC + cc) = gb[sp_i];
    }
  }
  PSHORT_STAMP(10);
  __syncthreads();
  PSHORT_STAMP(11);
  // Stage B: jobs (product, head-width n-tile): 0 = dV rows of the key block, 1 = dK rows of the key block,
  // 2 = dQ rows of the query block. Warp w takes jobs w, w + 8, ...; all of a warp's jobs and BF16 pieces run as
  // independent accumulator chains.
  constexpr int NJOB = 3 * NHT, JPW = (NJOB + NW - 1) / NW, KC = PSHORT_CHAINS;
  float acc[JPW][NS * KC][4];
#pragma unroll
  for (int j = 0; j < JPW; ++j)
#pragma unroll
    for (int c = 0; c < NS * KC; ++c) acc[j][c][0] = acc[j][c][1] = acc[j][c][2] = acc[j][c][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < NB; ++ks) {
#pragma unroll
    for (int j = 0; j < JPW; ++j) {
      const int job = warp + NW * j;
      if (!PSHORT_HAS(j, NJOB)) continue;
      // With HD = 64 every warp owns one head-width tile of each product: compile-time product index.
      const int product = NHT == NW ? j : job / NHT, nt = NHT == NW ? warp : job % NHT;
      uint32_t bb[2];
      const __nv_bfloat16* rhs = product == 0 ? sd : (product == 1 ? sq : sk);
      ldsm_x2_t(bb, rhs + (ks * 16 + (lane & 15)) * LD + nt * 8);
#pragma unroll
      for (int sp_i = 0; sp_i < NS; ++sp_i) {
        uint32_t aa[4];
        if (product < 2) {
          ldsm_x4_t(aa, (product == 0 ? spc : ssc) + sp_i * TP * LC
                        + (ks * 16 + (lane & 7) + ((lane >> 4) << 3)) * LC + ((lane >> 3) & 1) * 8);
        } else {
          ldsm_x4(aa, ssr + sp_i * 16 * LP + (lane & 15) * LP + ks * 16 + (lane >> 4) * 8);
        }
        mma(acc[j][sp_i * KC + ks % KC], aa, bb);
      }
    }
  }
  PSHORT_STAMP(12);
#pragma unroll
  for (int j = 0; j < JPW; ++j) {
    const int job = warp + NW * j;
    if (!PSHORT_HAS(j, NJOB)) continue;
    const int product = NHT == NW ? j : job / NHT, nt = NHT == NW ? warp : job % NHT;
#pragma unroll
    for (int c = 1; c < NS * KC; ++c)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[j][0][e] += acc[j][c][e];
    __nv_bfloat16* out;
    long long rs_;
    if (product == 0) { out = a.dv + head_offset(a.ldv, bh, a.heads); rs_ = a.ldv.sr; }
    else if (product == 1) { out = a.dk + head_offset(a.ldk, bh, a.heads); rs_ = a.ldk.sr; }
    else { out = a.dq + head_offset(a.ldq, bh, a.heads); rs_ = a.ldq.sr; }
    const int row0 = k0 + g, row1 = k0 + g + 8, col = nt * 8 + 2 * tig;
    if (row0 < t) store_bf16x2(out + (long long)row0 * rs_ + col, acc[j][0][0], acc[j][0][1]);
    if (row1 < t) store_bf16x2(out + (long long)row1 * rs_ + col, acc[j][0][2], acc[j][0][3]);
  }
  PSHORT_STAMP(13);
  if (PSHORT_TRACE && threadIdx.x == 0) a.trace[((long long)a.trace_offset + a.trace_item) * 16 + 15] = 13;
}

}  // namespace pshort
