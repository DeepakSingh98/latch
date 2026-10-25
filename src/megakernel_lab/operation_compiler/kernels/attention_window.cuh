// P8-C rectangular masked mixed-precision attention work items for the LATCH-OPS cooperative program (new code).
//
// Scope: the latch_ops.mattn_* operations (ops/window_attention.py) that the capture rewrites from the attention the
// compiler's `mixed_bf16` precision policy produces: Z independent (batch*head*block) attention problems with M
// query rows and N key rows, head width HD = 32, BF16 query/key/value/output-gradient operands with unit stride
// along the head width, an optional additive FP32 mask broadcast over up to two leading batch dimensions, and a
// captured scalar score scale. Query and key extents may differ (block-local / sliding-window attention).
//
// Arithmetic (declared; the rounding points of the captured mixed graph are reproduced):
//   s   = bf16(q k^T)                         FP32 accumulation (mma.sync m16n8k16), one RNE rounding
//   x   = fl(s op c) + mask                   the captured scale, then the additive mask
//   m   = max_j x_j,  l = sum_j expf(x_j - m) (fixed order of this kernel)
//   P   = fl(expf(x - m) / l)                 recomputed bit-identically in every phase from the stored (m, l)
//   O   = bf16(bf16(P) v)
//   dP  = bf16(dO v^T);  delta = sum_j P_j dP_j (fixed order, as _softmax_backward_data)
//   dS  = bf16(fl(fl(P * fl(dP - delta)) op c))
//   dQ  = bf16(dS k),  dK = bf16(dS^T q),  dV = bf16(bf16(P)^T dO)
// Only FP32 summation order differs from the eager ATen graph (FP32 association). dK/dV items form the transposed
// products (S^T = K Q^T, dP^T = V dO^T); every element is the same 32-term exact-product dot product.
//
// Exact tile skipping: the mask table (mattn_mask_table) holds, per (mask batch, 16-row block, 64-key block), the
// maximum mask value and whether the block is all +0/-0. All-zero blocks are added without reading the mask (x + 0 is
// the identity on these values up to the sign of zero, which expf ignores). A block is skipped when, for every row
// i, |c| |q_i| max_j |k_j| (1 + 2^-6) + max(mask block) < m_i - 120 with m_i the row max over computed blocks (it only
// grows): every skipped x_ij - m_i < -120 so expf underflows to exactly +0 and every contribution is exactly zero.
// Rows without a computed finite entry never skip. No atomics: every output element has exactly one writer.
// Requires the program's 256-thread CTA.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace pwin {

#define PWIN_DEV __device__ __forceinline__
constexpr int NT = 256, NW = 8, HD = 32, LD = HD + 8, KB = 64;   // key block of the mask table / skip test
constexpr float MARGIN = 120.f;
constexpr float SLACK = 1.015625f;                               // 1 + 2^-6: bf16 rounding of s plus FP32 slack
using bf16 = __nv_bfloat16;

struct Op {            // operand: element (z, r, d) at p[z*sz + r*sr + d*sd]; BF16, or FP32 rounded to BF16 on load
  const bf16* p;
  long long sz, sr, sd;
  int fp32;
};
struct Res {           // result: element (z, r, d) at p[z*sz + r*sr + d*sd]; BF16, or its exact FP32 widening
  bf16* p;
  long long sz, sr, sd;
  int fp32;
};

struct Args {
  Op q, k, v, dout;    // q [Z][M][HD], k [Z][N][HD] (the K of K^T), v [Z][N][HD], dout [Z][M][HD]
  Res out;             // O, dQ, dK or dV depending on the phase
  int Z, M, N;
  int g1;              // z = z0 * g1 + z1 (leading batch dimensions of the mask view)
  const float* mask;   // nullptr: no mask
  long long mb0, mb1, mi, mj;   // mask element strides of (z0, z1, row, key)
  long long* table;    // [tb][M/16][N/64][TE] mask block classes and row bitmaps (see table_phase); nullptr: none
  long long tb0, tb1;  // table batch strides (elements of [M/16][N/64][2] blocks) of z0, z1; 0 broadcast
  float* stats;        // [Z][M][2] (m, l)
  float* delta;        // [Z][M]
  float scale;
  int scale_mode;      // 0 none, 1 x*scale, 2 x/scale
  float rscale;        // RN(1 / scale), set by the phases from scale
  long long* prof;     // diagnostic builds (PWIN_PROF): per (CTA, warp) cycle counters [8]
  int online;          // declared online-softmax arithmetic (see "online arithmetic" below); 0: exact arithmetic
  Op o;                // online backward: the forward output O [Z][M][HD] (delta_i = dO_i . O_i)
  // P9-C absorbed operands (mattn_fwd2 / mattn_bwd2, attention_window_v4.cuh): when absorbed != 0, item
  // z = (b * H + h) * nb + n reads q / k / v rows from the BF16 projection src [B][L][3][H][HD] (rotation applied on
  // load; key/value row r of the item is sequence position n * W + r - W, zero before 0), dO from the head-merged
  // BF16 dsrc [B][L][H][HD], and stores O head-merged and the gradients as described in the v4 header.
  int absorbed;
  const bf16* src;
  const bf16* dsrc;
  const float* rcos;   // [L][HD/2]
  const float* rsin;
  int gB, gL, gH, gW, gnb;
  bf16* o_m;           // O head-merged [B][L][H][HD]
  bf16 *dq_o, *dkc_o, *dkp_o, *dvc_o, *dvp_o;   // [B][H][L][HD]
  bf16* proj_grad;     // mattn_dsrc result [B][L][3][H][HD]
};
#ifndef PWIN_PROF
#define PWIN_PROF 0
#endif
// PWIN_TABSMEM: each item's mask-table entries are copied to shared memory once (no global reads per tile).
// PWIN_PAIR: warps evaluate two adjacent 16-row (or 16-key) blocks together, sharing the other operand's fragments.
// Neither changes any value: every row block sees the same tiles, in the same order, with the same arithmetic.
#ifndef PWIN_TABSMEM
#define PWIN_TABSMEM 0
#endif
#ifndef PWIN_PAIR
#define PWIN_PAIR 0
#endif
#define PWIN_T0(v) long long v = PWIN_PROF ? clock64() : 0
#define PWIN_ACC(a, slot, v) do { if (PWIN_PROF && (a).prof && (threadIdx.x & 31) == 0) \
    (a).prof[((long long)blockIdx.x * NW + (threadIdx.x >> 5)) * 8 + (slot)] += clock64() - (v); } while (0)

PWIN_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
PWIN_DEV void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PWIN_DEV void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PWIN_DEV void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
PWIN_DEV uint32_t pack(float x0, float x1) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
  return *reinterpret_cast<const uint32_t*>(&h);
}
PWIN_DEV float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
#ifndef PWIN_MARKSTEIN
#define PWIN_MARKSTEIN 0
#endif
// Correctly rounded x / b from y = RN(1/b) (Markstein: q = RN(x y), r = x - b q exact by FMA, RN(q + r y) = RN(x / b)
// for normal operands and quotients); outside a safe magnitude range the IEEE division is used instead.
#ifndef PWIN_DIAG_NODIV
#define PWIN_DIAG_NODIV 0     // diagnostic timing builds only: multiply by the reciprocal (wrong rounding)
#endif
#ifndef PWIN_DIAG_NOMASK
#define PWIN_DIAG_NOMASK 0    // diagnostic timing builds only: never read the mask (wrong results)
#endif
#if defined(PWIN_DIAG_FASTEXP) && PWIN_DIAG_FASTEXP
#define expf __expf           // diagnostic timing builds only
#endif
PWIN_DEV float div_rn(float x, float b, float y) {
#if PWIN_DIAG_NODIV
  return __fmul_rn(x, y);
#endif
#if PWIN_MARKSTEIN
  const float q = __fmul_rn(x, y);
  const float r = __fmaf_rn(-q, b, x);
  const float ax = fabsf(x), aq = fabsf(q);
  if ((ax > 1e-30f && ax < 1e30f && aq > 1e-30f && aq < 1e30f) || x == 0.f) return x == 0.f ? __fdiv_rn(x, b) : __fmaf_rn(r, y, q);
  return __fdiv_rn(x, b);
#else
  (void)y;
  return __fdiv_rn(x, b);
#endif
}
PWIN_DEV float apply_scale(const Args& a, float x) {
  return a.scale_mode == 1 ? __fmul_rn(x, a.scale) : (a.scale_mode == 2 ? div_rn(x, a.scale, a.rscale) : x);
}
// Markstein quotient without range checks: RN(x / b) for normal x, b and quotient (y = RN(1 / b)).
PWIN_DEV float mk_div(float x, float b, float y) {
  const float q = __fmul_rn(x, y);
  return __fmaf_rn(__fmaf_rn(-q, b, x), y, q);
}
// Operands for which mk_div is exact here: zero, or 2^-90 <= |x| <= 2^90 with |b| in [2^-30, 2^30] (normal quotient
// and residual).
PWIN_DEV bool mk_safe(float x) {
  const float ax = fabsf(x);
  return x == 0.f || (ax >= 8.077935669463161e-28f && ax <= 1.2379400392853803e27f);
}
PWIN_DEV bool mk_divisor(float b) {
  const float ab = fabsf(b);
  return ab >= 9.313225746154785e-10f && ab <= 1073741824.f;
}
// Tile of quotients by a per-row divisor (rows of e: 0 for e < 2, 1 otherwise): Markstein when every element of the
// warp's tile and both divisors are in the exact range, else IEEE division for the whole tile (warp-uniform choice).
PWIN_DEV void tile_div_rows(float (&v)[8][4], const float (&b)[2], const float (&y)[2]) {
  bool ok = mk_divisor(b[0]) && mk_divisor(b[1]);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) ok = ok && mk_safe(v[n][e]);
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[n][e] = mk_div(v[n][e], b[e >> 1], y[e >> 1]);
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[n][e] = __fdiv_rn(v[n][e], b[e >> 1]);
  }
}
// Tile scaled by the captured score scale (x * c, or x / c with the exact division rule above).
PWIN_DEV void tile_scale(const Args& a, float (&v)[8][4]) {
  if (a.scale_mode == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[n][e] = __fmul_rn(v[n][e], a.scale);
  } else if (a.scale_mode == 2) {
    const float b[2] = {a.scale, a.scale}, y[2] = {a.rscale, a.rscale};
    tile_div_rows(v, b, y);
  }
}
PWIN_DEV float scale_abs(const Args& a) {
  return a.scale_mode == 1 ? fabsf(a.scale) : (a.scale_mode == 2 ? 1.f / fabsf(a.scale) : 1.f);
}
PWIN_DEV long long mask_base(const Args& a, int z) {
  return (long long)(z / a.g1) * a.mb0 + (long long)(z % a.g1) * a.mb1;
}
constexpr int TE = 20;   // table entry words: max bits, all-zero, binary, value bits, 16 row bitmaps
PWIN_DEV const long long* table_at(const Args& a, int z, int rb, int kb) {
  const int nrb = a.M / 16, nkb = a.N / KB;
  return a.table + TE * ((((long long)(z / a.g1) * a.tb0 + (long long)(z % a.g1) * a.tb1) * nrb + rb) * nkb + kb);
}
PWIN_DEV float tab_max(const long long* t) { return __int_as_float((int)(t[0] & 0xffffffffLL)); }
PWIN_DEV bool tab_zero(const long long* t) { return t[1] != 0; }
PWIN_DEV bool tab_binary(const long long* t) { return t[2] != 0; }
PWIN_DEV float tab_value(const long long* t) { return __int_as_float((int)(t[3] & 0xffffffffLL)); }
PWIN_DEV unsigned long long tab_bits(const long long* t, int row) { return (unsigned long long)t[4 + row]; }

// rows [r0, r0 + rows) of a BF16 operand of item z into shared [rows][LD]; rows beyond `limit` are zero.
// Unit head-width stride with 16-byte aligned rows: 16-byte loads; otherwise element loads ordered so that
// consecutive threads read consecutive rows (coalesced for row-unit layouts such as a materialized K^T).
PWIN_DEV void load_rows(bf16* dst, const Op& op, int z, int r0, int rows, int limit) {
  constexpr int CH = HD / 8;
  const long long base = (long long)z * op.sz;
  if (op.fp32) {
    // FP32 source: each element rounded to BF16 (RNE) exactly as the absorbed cast; 2 x 16-byte loads per 8 values.
    const float* src = reinterpret_cast<const float*>(op.p);
    const bool vec = op.sd == 1 && (op.sr % 4) == 0 && (op.sz % 4) == 0 && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    for (int idx = (int)threadIdx.x; idx < rows * CH; idx += NT) {
      const int r = idx / CH, c = 8 * (idx % CH);
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (r0 + r < limit) {
        const float* row = src + base + (long long)(r0 + r) * op.sr;
        float x[8];
        if (vec) {
          const float4 a = __ldg(reinterpret_cast<const float4*>(row + c)), b = __ldg(reinterpret_cast<const float4*>(row + c + 4));
          x[0] = a.x; x[1] = a.y; x[2] = a.z; x[3] = a.w; x[4] = b.x; x[5] = b.y; x[6] = b.z; x[7] = b.w;
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) x[e] = __ldg(row + (long long)(c + e) * op.sd);
        }
        val = make_uint4(pack(x[0], x[1]), pack(x[2], x[3]), pack(x[4], x[5]), pack(x[6], x[7]));
      }
      *reinterpret_cast<uint4*>(dst + r * LD + c) = val;
    }
    return;
  }
  const bool vec = op.sd == 1 && (op.sr % 8) == 0 && (op.sz % 8) == 0 && ((reinterpret_cast<uintptr_t>(op.p) & 15) == 0);
  if (vec) {
    for (int idx = (int)threadIdx.x; idx < rows * CH; idx += NT) {
      const int r = idx / CH, c = 8 * (idx % CH);
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (r0 + r < limit) val = __ldg(reinterpret_cast<const uint4*>(op.p + base + (long long)(r0 + r) * op.sr + c));
      *reinterpret_cast<uint4*>(dst + r * LD + c) = val;
    }
  } else if (op.sr == 1 && (op.sd % 8) == 0 && (op.sz % 8) == 0 && (r0 % 8) == 0 && (rows % 8) == 0 &&
             ((reinterpret_cast<uintptr_t>(op.p) & 15) == 0)) {
    // Row-unit layout (e.g. a materialized K^T): 16-byte loads of 8 consecutive rows of one column, scattered into
    // the row-major shared tile. Same elements, same zero rows beyond `limit`.
    const int groups = rows / 8;
    for (int idx = (int)threadIdx.x; idx < groups * HD; idx += NT) {
      const int d = idx % HD, g = idx / HD, r = 8 * g;   // lanes over columns: conflict-free shared stores
      uint4 val = make_uint4(0u, 0u, 0u, 0u);
      if (r0 + r + 7 < limit) {
        val = __ldg(reinterpret_cast<const uint4*>(op.p + base + (long long)(r0 + r) + (long long)d * op.sd));
      } else {
        unsigned short e[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
          e[k] = (r0 + r + k < limit) ? __bfloat16_as_ushort(op.p[base + (long long)(r0 + r + k) + (long long)d * op.sd])
                                      : (unsigned short)0;
        val = make_uint4(e[0] | (unsigned)e[1] << 16, e[2] | (unsigned)e[3] << 16, e[4] | (unsigned)e[5] << 16,
                         e[6] | (unsigned)e[7] << 16);
      }
      const unsigned w[4] = {val.x, val.y, val.z, val.w};
#pragma unroll
      for (int k = 0; k < 8; ++k)
        dst[(r + k) * LD + d] = __ushort_as_bfloat16((unsigned short)((k & 1) ? (w[k >> 1] >> 16) : (w[k >> 1] & 0xffffu)));
    }
  } else {
    for (int idx = (int)threadIdx.x; idx < rows * HD; idx += NT) {
      const int r = idx % rows, d = idx / rows;
      bf16 val = __float2bfloat16_rn(0.f);
      if (r0 + r < limit) val = op.p[base + (long long)(r0 + r) * op.sr + (long long)d * op.sd];
      dst[r * LD + d] = val;
    }
  }
}
// Euclidean norm of each of `rows` shared rows (fixed order).
PWIN_DEV void row_norms(const bf16* s, int rows, float* out) {
  for (int r = (int)threadIdx.x; r < rows; r += NT) {
    float acc = 0.f;
#pragma unroll
    for (int d = 0; d < HD; ++d) {
      const float x = __bfloat162float(s[r * LD + d]);
      acc = __fadd_rn(acc, __fmul_rn(x, x));
    }
    out[r] = sqrtf(acc);
  }
}
// Max of norms over each 64-row block (keys) -> out[blocks].
PWIN_DEV void block_max(const float* norms, int rows, float* out) {
  for (int b = (int)threadIdx.x; b < rows / KB; b += NT) {
    float mx = 0.f;
    for (int r = 0; r < KB; ++r) mx = fmaxf(mx, norms[b * KB + r]);
    out[b] = mx;
  }
}

// acc[8][4] = A(16 rows of As from row a0) x B(64 rows of Bs from row b0)^T over HD: (16 x 64) FP32.
PWIN_DEV void tile_nt(float (&acc)[8][4], const bf16* As, int a0, const bf16* Bs, int b0) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks) {
    uint32_t a[4];
    ldsm_x4(a, As + (a0 + (lane & 15)) * LD + 16 * ks + 8 * (lane >> 4));
#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      uint32_t bb[4];
      ldsm_x4(bb, Bs + (b0 + 16 * n2 + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * ks + 8 * ((lane >> 3) & 1));
      mma(acc[2 * n2], a, bb[0], bb[1]);
      mma(acc[2 * n2 + 1], a, bb[2], bb[3]);
    }
  }
}
// Two adjacent 16-row A blocks (a0, a0 + 16) against the same B rows: each B fragment is loaded once.
PWIN_DEV void tile_nt2(float (&x0)[8][4], float (&x1)[8][4], const bf16* As, int a0, const bf16* Bs, int b0) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    x0[n][0] = x0[n][1] = x0[n][2] = x0[n][3] = 0.f;
    x1[n][0] = x1[n][1] = x1[n][2] = x1[n][3] = 0.f;
  }
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks) {
    uint32_t f0[4], f1[4];
    ldsm_x4(f0, As + (a0 + (lane & 15)) * LD + 16 * ks + 8 * (lane >> 4));
    ldsm_x4(f1, As + (a0 + 16 + (lane & 15)) * LD + 16 * ks + 8 * (lane >> 4));
#pragma unroll
    for (int n2 = 0; n2 < 4; ++n2) {
      uint32_t bb[4];
      ldsm_x4(bb, Bs + (b0 + 16 * n2 + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * ks + 8 * ((lane >> 3) & 1));
      mma(x0[2 * n2], f0, bb[0], bb[1]);
      mma(x0[2 * n2 + 1], f0, bb[2], bb[3]);
      mma(x1[2 * n2], f1, bb[0], bb[1]);
      mma(x1[2 * n2 + 1], f1, bb[2], bb[3]);
    }
  }
}
// out[HD/8][4] += P(16 x 64, accumulator layout, values already BF16-exact) x Bs(64 rows from b0, HD columns).
PWIN_DEV void tile_pv(float (&out)[HD / 8][4], const float (&p)[8][4], const bf16* Bs, int b0) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const uint32_t pa[4] = {pack(p[2 * kk][0], p[2 * kk][1]), pack(p[2 * kk][2], p[2 * kk][3]),
                            pack(p[2 * kk + 1][0], p[2 * kk + 1][1]), pack(p[2 * kk + 1][2], p[2 * kk + 1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Bs + (b0 + 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(out[2 * n2], pa, bb[0], bb[1]);
      mma(out[2 * n2 + 1], pa, bb[2], bb[3]);
    }
  }
}

// Score tile x for 16 rows (global rows ri0 + lane rows, ri0 a multiple of 16) x 64 keys (global key kj0, a multiple
// of 64): scale, then the mask. t: the block's mask-table entry (nullptr: no mask). All-zero blocks add nothing; binary
// blocks add +0 or v from the row bitmaps (fl(x + 0) = x and fl(x + v) are exactly what reading the mask gives);
// general blocks read the mask. Row-major orientation: element (n, e) is row ri0 + lane/4 + 8*(e/2), key
// kj0 + 8n + 2*(lane%4) + (e%2).
PWIN_DEV void finish_scores(const Args& a, float (&s)[8][4], int z, int ri0, int kj0, const long long* t) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = rbf(s[n][e]);
  tile_scale(a, s);
  if (!a.mask || PWIN_DIAG_NOMASK || tab_zero(t)) return;
  if (tab_binary(t)) {
    const float v = tab_value(t);
    const unsigned long long bits[2] = {tab_bits(t, lane >> 2), tab_bits(t, (lane >> 2) + 8)};
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = 8 * n + 2 * (lane & 3) + (e & 1);
        if (!((bits[e >> 1] >> j) & 1ull)) s[n][e] = __fadd_rn(s[n][e], v);
      }
    return;
  }
  const long long mb = mask_base(a, z);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = ri0 + (lane >> 2) + 8 * (e >> 1), j = kj0 + 8 * n + 2 * (lane & 3) + (e & 1);
      s[n][e] = __fadd_rn(s[n][e], __ldg(a.mask + mb + (long long)i * a.mi + (long long)j * a.mj));
    }
}
// Transposed tile (16 keys from kj0 x 64 query rows from ri0, ri0 a multiple of 64): element (n, e) is key
// kj0 + lane/4 + 8*(e/2), row ri0 + 8n + 2*(lane%4) + (e%2); query row block q = n / 2 uses table entry t4[q].
PWIN_DEV void finish_scores_t(const Args& a, float (&s)[8][4], int z, int kj0, int ri0, const long long* const (&t4)[4]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = rbf(s[n][e]);
  tile_scale(a, s);
  if (!a.mask || PWIN_DIAG_NOMASK) return;
  bool zero = true, binary = true;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    zero = zero && tab_zero(t4[q]);
    binary = binary && (tab_zero(t4[q]) || tab_binary(t4[q]));
  }
  if (zero) return;
  if (binary) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const long long* t = t4[n >> 1];
      if (tab_zero(t)) continue;
      const float v = tab_value(t);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jl = (kj0 % KB) + (lane >> 2) + 8 * (e >> 1);
        const int il = (8 * n + 2 * (lane & 3) + (e & 1)) & 15;
        if (!((tab_bits(t, il) >> jl) & 1ull)) s[n][e] = __fadd_rn(s[n][e], v);
      }
    }
    return;
  }
  const long long mb = mask_base(a, z);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = kj0 + (lane >> 2) + 8 * (e >> 1), i = ri0 + 8 * n + 2 * (lane & 3) + (e & 1);
      s[n][e] = __fadd_rn(s[n][e], __ldg(a.mask + mb + (long long)i * a.mi + (long long)j * a.mj));
    }
}

// Skip test of key block kb for the warp's 16 rows (row-major items). qn: norms of the rows (shared, item rows),
// kmax: max key norm of the block. m: this thread's two row maxima; reduced over the warp (all 16 rows).
PWIN_DEV bool skip_rows(const Args& a, float tmax, float qn0, float qn1, float kmax, const float (&m)[2]) {
  const float bound0 = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(qn0, kmax)), SLACK);
  const float bound1 = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(qn1, kmax)), SLACK);
  bool ok = (tmax + bound0 < m[0] - MARGIN) && (tmax + bound1 < m[1] - MARGIN);
  return __all_sync(0xffffffffu, ok);
}

// ------------------------------------------------------------------------------------- mask table (one phase)
// Entry per (table batch t, 16-row block rb, 64-key block kb): word 0 the bits of the block's maximum mask value,
// word 1 = 1 if every element is +-0, word 2 = 1 if every element is +-0 or equal to the block minimum v (binary),
// word 3 the bits of v, words 4..19 one bitmap per row (bit j set iff element (row, j) is +-0).
PWIN_DEV void table_phase(const Args& a, int t0n, int t1n, float* smem) {
  const int nrb = a.M / 16, nkb = a.N / KB;
  const long long blocks = (long long)t0n * t1n * nrb * nkb;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (long long blk = (long long)blockIdx.x * NW + warp; blk < blocks; blk += (long long)gridDim.x * NW) {
    const int kb = (int)(blk % nkb), rb = (int)((blk / nkb) % nrb);
    const long long t = blk / ((long long)nkb * nrb);
    const int t1 = (int)(t % t1n), t0 = (int)(t / t1n);
    const long long base = (long long)t0 * a.mb0 + (long long)t1 * a.mb1;
    float x0[16], x1[16];
    float mx = -INFINITY, mn = INFINITY;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      const long long row = base + (long long)(16 * rb + r) * a.mi;
      x0[r] = __ldg(a.mask + row + (long long)(KB * kb + lane) * a.mj);
      x1[r] = __ldg(a.mask + row + (long long)(KB * kb + 32 + lane) * a.mj);
      mx = fmaxf(mx, fmaxf(x0[r], x1[r]));
      mn = fminf(mn, fminf(x0[r], x1[r]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
    }
    bool zero = true, binary = true;
    long long* out = a.table + TE * blk;
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      zero = zero && x0[r] == 0.f && x1[r] == 0.f;
      binary = binary && (x0[r] == 0.f || x0[r] == mn) && (x1[r] == 0.f || x1[r] == mn);
      const unsigned lo = __ballot_sync(0xffffffffu, x0[r] == 0.f), hi = __ballot_sync(0xffffffffu, x1[r] == 0.f);
      if (lane == 0) out[4 + r] = (long long)(((unsigned long long)hi << 32) | lo);
    }
    zero = __all_sync(0xffffffffu, zero);
    binary = __all_sync(0xffffffffu, binary);
    if (lane == 0) {
      out[0] = (long long)(unsigned)__float_as_int(mx);
      out[1] = zero ? 1 : 0;
      out[2] = binary ? 1 : 0;
      out[3] = (long long)(unsigned)__float_as_int(mn);
    }
  }
  (void)smem;
}

// Two P tiles (16 rows each) times the same 64 B rows: each B fragment is loaded once; per-output accumulation order
// is that of tile_pv.
PWIN_DEV void tile_pv2(float (&o0)[HD / 8][4], const float (&p0)[8][4], float (&o1)[HD / 8][4], const float (&p1)[8][4],
                       const bf16* Bs, int b0) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) {
    const uint32_t pa[4] = {pack(p0[2 * kk][0], p0[2 * kk][1]), pack(p0[2 * kk][2], p0[2 * kk][3]),
                            pack(p0[2 * kk + 1][0], p0[2 * kk + 1][1]), pack(p0[2 * kk + 1][2], p0[2 * kk + 1][3])};
    const uint32_t pb[4] = {pack(p1[2 * kk][0], p1[2 * kk][1]), pack(p1[2 * kk][2], p1[2 * kk][3]),
                            pack(p1[2 * kk + 1][0], p1[2 * kk + 1][1]), pack(p1[2 * kk + 1][2], p1[2 * kk + 1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Bs + (b0 + 16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(o0[2 * n2], pa, bb[0], bb[1]);
      mma(o0[2 * n2 + 1], pa, bb[2], bb[3]);
      mma(o1[2 * n2], pb, bb[0], bb[1]);
      mma(o1[2 * n2 + 1], pb, bb[2], bb[3]);
    }
  }
}

// ------------------------------------------------------------------------------------------ item layout
// One work item per attention problem z: every operand row is loaded once. Warp w owns 16-row query blocks
// w, w + 8, ... (row phases) or 16-key blocks w, w + 8, ... (dK/dV phase).
struct Smem {
  bf16 *Qs, *Ks, *Vs, *Ds;
  float *qn, *kn, *kmax, *ms, *ls, *dl, *rl;
  long long* tab;          // PWIN_TABSMEM: shared copy of the item's table entries
  const long long* tb;     // the item's table entries [M/16][N/64][TE] (shared or global); set per item
};
PWIN_DEV Smem item_smem(const Args& a, float* smem) {
  Smem s;
  s.Qs = reinterpret_cast<bf16*>(smem);
  s.Ks = s.Qs + a.M * LD;
  s.Vs = s.Ks + a.N * LD;
  s.Ds = s.Vs + a.N * LD;
  s.qn = reinterpret_cast<float*>(s.Ds + a.M * LD);
  s.kn = s.qn + a.M;
  s.kmax = s.kn + a.N;
  s.ms = s.kmax + a.N / KB;
  s.ls = s.ms + a.M;
  s.dl = s.ls + a.M;
  s.rl = s.dl + a.M;
  s.tab = reinterpret_cast<long long*>(smem) + (2LL * (2 * a.M + 2 * a.N) * LD + 4LL * (5 * a.M + a.N + a.N / KB) + 15) / 16 * 2;
  s.tb = nullptr;
  return s;
}
__host__ __device__ constexpr long long item_shared_bytes(int M, int N) {
  return (2LL * (2 * M + 2 * N) * LD + 4LL * (5 * M + N + N / KB) + 15) / 16 * 16
         + (PWIN_TABSMEM ? 8LL * TE * (M / 16) * (N / KB) : 0LL);
}
// The item's table entries: copied to shared memory (PWIN_TABSMEM; synchronized by load_item) or read in place.
PWIN_DEV void item_table(const Args& a, Smem& s, int z) {
  if (!a.table) { s.tb = nullptr; return; }
  const int nrb = a.M / 16, nkb = a.N / KB;
  const long long* src = a.table + TE * (((long long)(z / a.g1) * a.tb0 + (long long)(z % a.g1) * a.tb1) * nrb * nkb);
  if (!PWIN_TABSMEM) { s.tb = src; return; }
  const int words = TE * nrb * nkb;   // multiple of 2: 16-byte chunks (entries are 160 bytes)
  for (int w = 2 * (int)threadIdx.x; w < words; w += 2 * NT)
    *reinterpret_cast<int4*>(s.tab + w) = __ldcg(reinterpret_cast<const int4*>(src + w));
  s.tb = s.tab;
}
PWIN_DEV const long long* tab_entry(const Args& a, const Smem& s, int rb, int kb) {
  return s.tb + TE * (rb * (a.N / KB) + kb);
}

PWIN_DEV void load_item(const Args& a, Smem& s, int z, bool v, bool d, bool st, bool dl) {
  item_table(a, s, z);
  load_rows(s.Qs, a.q, z, 0, a.M, a.M);
  load_rows(s.Ks, a.k, z, 0, a.N, a.N);
  if (v) load_rows(s.Vs, a.v, z, 0, a.N, a.N);
  if (d) load_rows(s.Ds, a.dout, z, 0, a.M, a.M);
  if (st || dl) {
    for (int r = (int)threadIdx.x; r < a.M; r += NT) {
      if (st) {
        const float2 x = __ldcg(reinterpret_cast<const float2*>(a.stats) + (long long)z * a.M + r);
        s.ms[r] = x.x;
        s.ls[r] = x.y;
        s.rl[r] = __frcp_rn(x.y);
      }
      if (dl) s.dl[r] = __ldcg(a.delta + (long long)z * a.M + r);
    }
  }
  __syncthreads();
  row_norms(s.Qs, a.M, s.qn);
  row_norms(s.Ks, a.N, s.kn);
  __syncthreads();
  block_max(s.kn, a.N, s.kmax);
  __syncthreads();
}

// Scores of row block (rows r0..r0+15) against key block kb, scaled and masked.
PWIN_DEV void scores(const Args& a, const Smem& s, float (&x)[8][4], int z, int r0, int kb, const long long* t) {
  tile_nt(x, s.Qs, r0, s.Ks, kb * KB);
  finish_scores(a, x, z, r0, kb * KB, t);
}

// Mask-table class of (row block, key block): 0 compute, 2 skip candidate (every mask value below -1e30).
PWIN_DEV int block_class(const Args& a, const Smem& s, int rb, int kb, float& tmax, const long long*& t) {
  if (!a.table) { tmax = 0.f; t = nullptr; return 0; }
  t = tab_entry(a, s, rb, kb);
  tmax = tab_max(t);
  return tmax < -1e30f ? 2 : 0;
}

PWIN_DEV void quad_max2(float (&v)[2]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    v[h] = fmaxf(v[h], __shfl_xor_sync(0xffffffffu, v[h], 1));
    v[h] = fmaxf(v[h], __shfl_xor_sync(0xffffffffu, v[h], 2));
  }
}
PWIN_DEV void quad_sum2(float (&v)[2]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    v[h] = __fadd_rn(v[h], __shfl_xor_sync(0xffffffffu, v[h], 1));
    v[h] = __fadd_rn(v[h], __shfl_xor_sync(0xffffffffu, v[h], 2));
  }
}

// Online row statistics over one computed block: m <- max(m, block max); l <- l * expf(m_old - m) + sum expf(x - m).
PWIN_DEV void online(const float (&x)[8][4], float (&m)[2], float (&l)[2]) {
  float mt[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    mt[0] = fmaxf(mt[0], fmaxf(x[n][0], x[n][1]));
    mt[1] = fmaxf(mt[1], fmaxf(x[n][2], x[n][3]));
  }
  quad_max2(mt);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float mn = fmaxf(m[h], mt[h]);
    if (mn != m[h]) {
      l[h] = m[h] == -INFINITY ? 0.f : __fmul_rn(l[h], expf(__fsub_rn(m[h], mn)));
      m[h] = mn;
    }
  }
  if (m[0] != -INFINITY) {
#pragma unroll
    for (int n = 0; n < 8; ++n) l[0] = __fadd_rn(__fadd_rn(l[0], expf(__fsub_rn(x[n][0], m[0]))), expf(__fsub_rn(x[n][1], m[0])));
  }
  if (m[1] != -INFINITY) {
#pragma unroll
    for (int n = 0; n < 8; ++n) l[1] = __fadd_rn(__fadd_rn(l[1], expf(__fsub_rn(x[n][2], m[1]))), expf(__fsub_rn(x[n][3], m[1])));
  }
}

// Exact skip test of (row block at r0, key block kb) against row maxima m (thread rows r, r+8; warp-uniform result).
PWIN_DEV bool skippable(const Args& a, const Smem& s, int r0, int kb, float tmax, const float (&m)[2]) {
  const int lane = threadIdx.x & 31;
  return skip_rows(a, tmax, s.qn[r0 + (lane >> 2)], s.qn[r0 + (lane >> 2) + 8], s.kmax[kb], m);
}

// P tile from the row statistics (m, l): fl(expf(x - m) / l). Returns false when the block is skipped.
PWIN_DEV bool probs(const Args& a, const Smem& s, float (&p)[8][4], int z, int r0, int kb, const float (&m)[2],
                    const float (&l)[2]) {
  float tmax;
  const long long* t;
  PWIN_T0(t_c);
  const int cls = block_class(a, s, r0 / 16, kb, tmax, t);
  if (cls == 2 && skippable(a, s, r0, kb, tmax, m)) { PWIN_ACC(a, 3, t_c); return false; }
  PWIN_ACC(a, 3, t_c);
  PWIN_T0(t_mma);
  tile_nt(p, s.Qs, r0, s.Ks, kb * KB);
  PWIN_ACC(a, 4, t_mma);
  PWIN_T0(t_fin);
  finish_scores(a, p, z, r0, kb * KB, t);
  PWIN_ACC(a, 5, t_fin);
  PWIN_T0(t_exp);
  const float rl[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) p[n][e] = expf(__fsub_rn(p[n][e], m[e >> 1]));
  tile_div_rows(p, l, rl);
  PWIN_ACC(a, 6, t_exp);
  return true;
}

// probs() in two parts for paired row blocks: whether (r0, kb) is computed, then P from the raw score tile.
PWIN_DEV bool probs_needed(const Args& a, const Smem& s, int r0, int kb, const float (&m)[2], const long long*& t) {
  float tmax;
  const int cls = block_class(a, s, r0 / 16, kb, tmax, t);
  return !(cls == 2 && skippable(a, s, r0, kb, tmax, m));
}
PWIN_DEV void probs_finish(const Args& a, float (&p)[8][4], int z, int r0, int kb, const long long* t,
                           const float (&m)[2], const float (&l)[2]) {
  finish_scores(a, p, z, r0, kb * KB, t);
  const float rl[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) p[n][e] = expf(__fsub_rn(p[n][e], m[e >> 1]));
  tile_div_rows(p, l, rl);
}

PWIN_DEV void stats_of(const Smem& s, int r0, float (&m)[2], float (&l)[2]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    m[h] = s.ms[r0 + (lane >> 2) + 8 * h];
    l[h] = s.ls[r0 + (lane >> 2) + 8 * h];
  }
}

PWIN_DEV void store_rows(const Res& r, const float (&acc)[HD / 8][4], int z, int row0, int limit) {
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = row0 + (lane >> 2) + 8 * h;
    if (row >= limit) continue;
    const long long base = (long long)z * r.sz + (long long)row * r.sr;
#pragma unroll
    for (int nt = 0; nt < HD / 8; ++nt) {
      const int col = 8 * nt + c;
      const float x0 = rbf(acc[nt][2 * h]), x1 = rbf(acc[nt][2 * h + 1]);
      if (r.fp32) {
        float* f = reinterpret_cast<float*>(r.p);
        if (r.sd == 1) {
          *reinterpret_cast<float2*>(f + base + col) = make_float2(x0, x1);
        } else {
          f[base + (long long)col * r.sd] = x0;
          f[base + (long long)(col + 1) * r.sd] = x1;
        }
      } else if (r.sd == 1) {
        *reinterpret_cast<uint32_t*>(r.p + base + col) = pack(x0, x1);
      } else {
        r.p[base + (long long)col * r.sd] = __float2bfloat16_rn(x0);
        r.p[base + (long long)(col + 1) * r.sd] = __float2bfloat16_rn(x1);
      }
    }
  }
}

PWIN_DEV void zero_acc(float (&o)[HD / 8][4]) {
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
}

// --------------------------------------------------------------------------------------------- forward
// Phase: statistics (m, l) of every row (one online sweep) and O = bf16(bf16(P) v) -> a.out, a.stats.
PWIN_DEV void store_stats(const Args& a, int z, int r0, const float (&m)[2], const float (&l)[2]) {
  const int lane = threadIdx.x & 31;
  if ((lane & 3) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      reinterpret_cast<float2*>(a.stats)[(long long)z * a.M + r0 + (lane >> 2) + 8 * h] = make_float2(m[h], l[h]);
  }
}
// Forward of the row blocks at r0 and r0 + 16 (paired): each block computes exactly the tiles, in exactly the order,
// of the unpaired sweep; a tile needed by only one block is still formed for both (the other result is unused).
PWIN_DEV void fwd_pair(const Args& a, const Smem& s, int z, int r0) {
  const int nkb = a.N / KB, rbA = r0 / 16;
  float m[2][2] = {{-INFINITY, -INFINITY}, {-INFINITY, -INFINITY}}, l[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int pass = 0; pass < 2; ++pass) {
    for (int kb = 0; kb < nkb; ++kb) {
      bool use[2];
      const long long* t[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float tmax;
        const int cls = block_class(a, s, rbA + h, kb, tmax, t[h]);
        use[h] = (cls == 2) == (pass == 1) && !(cls == 2 && skippable(a, s, r0 + 16 * h, kb, tmax, m[h]));
      }
      if (!use[0] && !use[1]) continue;
      float x0[8][4], x1[8][4];
      tile_nt2(x0, x1, s.Qs, r0, s.Ks, kb * KB);
      if (use[0]) { finish_scores(a, x0, z, r0, kb * KB, t[0]); online(x0, m[0], l[0]); }
      if (use[1]) { finish_scores(a, x1, z, r0 + 16, kb * KB, t[1]); online(x1, m[1], l[1]); }
    }
  }
  quad_sum2(l[0]);
  quad_sum2(l[1]);
  float o0[HD / 8][4], o1[HD / 8][4];
  zero_acc(o0);
  zero_acc(o1);
  for (int kb = 0; kb < nkb; ++kb) {
    const long long* t[2];
    const bool use0 = probs_needed(a, s, r0, kb, m[0], t[0]), use1 = probs_needed(a, s, r0 + 16, kb, m[1], t[1]);
    if (!use0 && !use1) continue;
    float p0[8][4], p1[8][4];
    tile_nt2(p0, p1, s.Qs, r0, s.Ks, kb * KB);
    if (use0) probs_finish(a, p0, z, r0, kb, t[0], m[0], l[0]);
    if (use1) probs_finish(a, p1, z, r0 + 16, kb, t[1], m[1], l[1]);
    if (use0 && use1) tile_pv2(o0, p0, o1, p1, s.Vs, kb * KB);
    else if (use0) tile_pv(o0, p0, s.Vs, kb * KB);
    else tile_pv(o1, p1, s.Vs, kb * KB);
  }
  store_rows(a.out, o0, z, r0, a.M);
  store_rows(a.out, o1, z, r0 + 16, a.M);
  store_stats(a, z, r0, m[0], l[0]);
  store_stats(a, z, r0 + 16, m[1], l[1]);
}

PWIN_DEV void fwd_phase_v3(const Args& a, float* smem) {
  Smem s = item_smem(a, smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nkb = a.N / KB;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    PWIN_T0(t_load);
    load_item(a, s, z, true, false, false, false);
    PWIN_ACC(a, 0, t_load);
    if (PWIN_PAIR) {
      for (int rp = warp; rp < a.M / 32; rp += NW) fwd_pair(a, s, z, 32 * rp);
      continue;
    }
    for (int rb = warp; rb < a.M / 16; rb += NW) {
      const int r0 = 16 * rb;
      PWIN_T0(t_s1);
      float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
      // Sweep 1: blocks that must be computed, then skip candidates against the running maxima.
      for (int pass = 0; pass < 2; ++pass) {
        for (int kb = 0; kb < nkb; ++kb) {
          float tmax;
          const long long* t;
          const int cls = block_class(a, s, rb, kb, tmax, t);
          if ((cls == 2) != (pass == 1)) continue;
          if (cls == 2 && skippable(a, s, r0, kb, tmax, m)) continue;
          float x[8][4];
          scores(a, s, x, z, r0, kb, t);
          online(x, m, l);
        }
      }
      quad_sum2(l);
      PWIN_ACC(a, 1, t_s1);
      PWIN_T0(t_s2);
      // Sweep 2: P and the value product.
      float o[HD / 8][4];
      zero_acc(o);
      for (int kb = 0; kb < nkb; ++kb) {
        float p[8][4];
        if (!probs(a, s, p, z, r0, kb, m, l)) continue;
        tile_pv(o, p, s.Vs, kb * KB);
      }
      PWIN_ACC(a, 2, t_s2);
      store_rows(a.out, o, z, r0, a.M);
      if ((lane & 3) == 0) {
#pragma unroll
        for (int h = 0; h < 2; ++h)
          reinterpret_cast<float2*>(a.stats)[(long long)z * a.M + r0 + (lane >> 2) + 8 * h] = make_float2(m[h], l[h]);
      }
    }
  }
}

// dP tile (BF16-rounded): dO rows r0.. x V block^T.
PWIN_DEV void dp_tile(const Smem& s, float (&dp)[8][4], int r0, int kb) {
  tile_nt(dp, s.Ds, r0, s.Vs, kb * KB);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dp[n][e] = rbf(dp[n][e]);
}

// Sweep: delta_i = sum_j P_ij dP_ij (fixed order) of row block rb -> dst[row] (shared or global).
PWIN_DEV void delta_rows(const Args& a, const Smem& s, int z, int rb, float* dst) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, r0 = 16 * rb;
  float m[2], l[2];
  stats_of(s, r0, m, l);
  float sum[2] = {0.f, 0.f};
  for (int kb = 0; kb < nkb; ++kb) {
    float p[8][4], dp[8][4];
    if (!probs(a, s, p, z, r0, kb, m, l)) continue;
    dp_tile(s, dp, r0, kb);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sum[0] = __fadd_rn(__fadd_rn(sum[0], __fmul_rn(p[n][0], dp[n][0])), __fmul_rn(p[n][1], dp[n][1]));
      sum[1] = __fadd_rn(__fadd_rn(sum[1], __fmul_rn(p[n][2], dp[n][2])), __fmul_rn(p[n][3], dp[n][3]));
    }
  }
  quad_sum2(sum);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h) dst[r0 + (lane >> 2) + 8 * h] = sum[h];
  }
}

// Sweep: dQ rows of row block rb = bf16(bf16(dS) k) -> out (delta from s.dl).
PWIN_DEV void dq_rows(const Args& a, const Smem& s, const Res& out, int z, int rb) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, r0 = 16 * rb;
  float m[2], l[2], dl[2];
  stats_of(s, r0, m, l);
#pragma unroll
  for (int h = 0; h < 2; ++h) dl[h] = s.dl[r0 + (lane >> 2) + 8 * h];
  float dq[HD / 8][4];
  zero_acc(dq);
  for (int kb = 0; kb < nkb; ++kb) {
    float p[8][4], dp[8][4];
    if (!probs(a, s, p, z, r0, kb, m, l)) continue;
    dp_tile(s, dp, r0, kb);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(dp[n][e], dl[e >> 1]));
    tile_scale(a, dp);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dp[n][e] = rbf(dp[n][e]);
    tile_pv(dq, dp, s.Ks, kb * KB);
  }
  store_rows(out, dq, z, r0, a.M);
}

// Sweep: dK (= bf16(bf16(dS)^T q)) and dV (= bf16(bf16(P)^T dO)) of the 16 keys of block kb16, transposed tiles.
PWIN_DEV void dkv_keys(const Args& a, const Smem& s, const Res& dk_out, const Res& dv_out, int z, int kb16) {
  const int lane = threadIdx.x & 31, k0 = 16 * kb16;
  float kmax = 0.f;
  for (int r = 0; r < 16; ++r) kmax = fmaxf(kmax, s.kn[k0 + r]);
  float dk[HD / 8][4], dv[HD / 8][4];
  zero_acc(dk);
  zero_acc(dv);
  for (int i0 = 0; i0 < a.M; i0 += 64) {
    const long long* t4[4] = {nullptr, nullptr, nullptr, nullptr};
    if (a.table) {
      float tmax = -INFINITY;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        t4[q] = tab_entry(a, s, i0 / 16 + q, k0 / KB);
        tmax = fmaxf(tmax, tab_max(t4[q]));
      }
      if (tmax < -1e30f) {
        bool ok = true;
        for (int c = lane; c < 64; c += 32) {
          const float bound = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(s.qn[i0 + c], kmax)), SLACK);
          ok = ok && (tmax + bound < s.ms[i0 + c] - MARGIN);
        }
        if (__all_sync(0xffffffffu, ok)) continue;
      }
    }
    float p[8][4], dp[8][4];
    tile_nt(p, s.Ks, k0, s.Qs, i0);           // S^T: keys x 64 query rows
    finish_scores_t(a, p, z, k0, i0, t4);
    tile_nt(dp, s.Vs, k0, s.Ds, i0);          // dP^T = V dO^T
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
        p[n][e] = expf(__fsub_rn(p[n][e], s.ms[i]));
      }
    // P = e / l with a per-column divisor: Markstein when every element and divisor is in range, else IEEE.
    {
      bool ok = true;
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
          ok = ok && mk_safe(p[n][e]) && mk_divisor(s.ls[i]);
        }
      if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
            p[n][e] = mk_div(p[n][e], s.ls[i], s.rl[i]);
          }
      } else {
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
            p[n][e] = __fdiv_rn(p[n][e], s.ls[i]);
          }
      }
    }
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
        dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(rbf(dp[n][e]), s.dl[i]));
      }
    tile_scale(a, dp);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dp[n][e] = rbf(dp[n][e]);
    tile_pv(dv, p, s.Ds, i0);                 // dV += bf16(P)^T dO
    tile_pv(dk, dp, s.Qs, i0);                // dK += dS^T q
  }
  store_rows(dk_out, dk, z, k0, a.N);
  store_rows(dv_out, dv, z, k0, a.N);
}

// ---- paired backward sweeps (PWIN_PAIR): same tiles, order and arithmetic per block as the unpaired sweeps above.
PWIN_DEV void delta_pair(const Args& a, const Smem& s, int z, int r0, float* dst) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB;
  float m[2][2], l[2][2];
  stats_of(s, r0, m[0], l[0]);
  stats_of(s, r0 + 16, m[1], l[1]);
  float sum[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int kb = 0; kb < nkb; ++kb) {
    const long long* t[2];
    const bool use0 = probs_needed(a, s, r0, kb, m[0], t[0]), use1 = probs_needed(a, s, r0 + 16, kb, m[1], t[1]);
    if (!use0 && !use1) continue;
    float p0[8][4], p1[8][4], d0[8][4], d1[8][4];
    tile_nt2(p0, p1, s.Qs, r0, s.Ks, kb * KB);
    tile_nt2(d0, d1, s.Ds, r0, s.Vs, kb * KB);
    if (use0) {
      probs_finish(a, p0, z, r0, kb, t[0], m[0], l[0]);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        sum[0][0] = __fadd_rn(__fadd_rn(sum[0][0], __fmul_rn(p0[n][0], rbf(d0[n][0]))), __fmul_rn(p0[n][1], rbf(d0[n][1])));
        sum[0][1] = __fadd_rn(__fadd_rn(sum[0][1], __fmul_rn(p0[n][2], rbf(d0[n][2]))), __fmul_rn(p0[n][3], rbf(d0[n][3])));
      }
    }
    if (use1) {
      probs_finish(a, p1, z, r0 + 16, kb, t[1], m[1], l[1]);
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        sum[1][0] = __fadd_rn(__fadd_rn(sum[1][0], __fmul_rn(p1[n][0], rbf(d1[n][0]))), __fmul_rn(p1[n][1], rbf(d1[n][1])));
        sum[1][1] = __fadd_rn(__fadd_rn(sum[1][1], __fmul_rn(p1[n][2], rbf(d1[n][2]))), __fmul_rn(p1[n][3], rbf(d1[n][3])));
      }
    }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    quad_sum2(sum[h]);
    if ((lane & 3) == 0) {
      dst[r0 + 16 * h + (lane >> 2)] = sum[h][0];
      dst[r0 + 16 * h + (lane >> 2) + 8] = sum[h][1];
    }
  }
}

PWIN_DEV void ds_tile(const Args& a, float (&dp)[8][4], const float (&p)[8][4], const float (&dl)[2]) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(rbf(dp[n][e]), dl[e >> 1]));
  tile_scale(a, dp);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dp[n][e] = rbf(dp[n][e]);
}

PWIN_DEV void dq_pair(const Args& a, const Smem& s, const Res& out, int z, int r0) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB;
  float m[2][2], l[2][2], dl[2][2];
  stats_of(s, r0, m[0], l[0]);
  stats_of(s, r0 + 16, m[1], l[1]);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    dl[0][h] = s.dl[r0 + (lane >> 2) + 8 * h];
    dl[1][h] = s.dl[r0 + 16 + (lane >> 2) + 8 * h];
  }
  float q0[HD / 8][4], q1[HD / 8][4];
  zero_acc(q0);
  zero_acc(q1);
  for (int kb = 0; kb < nkb; ++kb) {
    const long long* t[2];
    const bool use0 = probs_needed(a, s, r0, kb, m[0], t[0]), use1 = probs_needed(a, s, r0 + 16, kb, m[1], t[1]);
    if (!use0 && !use1) continue;
    float p0[8][4], p1[8][4], d0[8][4], d1[8][4];
    tile_nt2(p0, p1, s.Qs, r0, s.Ks, kb * KB);
    tile_nt2(d0, d1, s.Ds, r0, s.Vs, kb * KB);
    if (use0) { probs_finish(a, p0, z, r0, kb, t[0], m[0], l[0]); ds_tile(a, d0, p0, dl[0]); }
    if (use1) { probs_finish(a, p1, z, r0 + 16, kb, t[1], m[1], l[1]); ds_tile(a, d1, p1, dl[1]); }
    if (use0 && use1) tile_pv2(q0, d0, q1, d1, s.Ks, kb * KB);
    else if (use0) tile_pv(q0, d0, s.Ks, kb * KB);
    else tile_pv(q1, d1, s.Ks, kb * KB);
  }
  store_rows(out, q0, z, r0, a.M);
  store_rows(out, q1, z, r0 + 16, a.M);
}

// Transposed P / dS tiles of one 16-key block (from the raw S^T and dP^T tiles), exactly as dkv_keys.
PWIN_DEV void dkv_tile(const Args& a, const Smem& s, float (&p)[8][4], float (&dp)[8][4], int i0) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
      p[n][e] = expf(__fsub_rn(p[n][e], s.ms[i]));
    }
  bool ok = true;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
      ok = ok && mk_safe(p[n][e]) && mk_divisor(s.ls[i]);
    }
  if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
        p[n][e] = mk_div(p[n][e], s.ls[i], s.rl[i]);
      }
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
        p[n][e] = __fdiv_rn(p[n][e], s.ls[i]);
      }
  }
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
      dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(rbf(dp[n][e]), s.dl[i]));
    }
  tile_scale(a, dp);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dp[n][e] = rbf(dp[n][e]);
}

// dK/dV of the 16-key blocks at k0 and k0 + 16 (k0 a multiple of 32: one table key block).
PWIN_DEV void dkv_pair(const Args& a, const Smem& s, const Res& dk_out, const Res& dv_out, int z, int k0) {
  const int lane = threadIdx.x & 31;
  float kmax[2] = {0.f, 0.f};
  for (int r = 0; r < 16; ++r) {
    kmax[0] = fmaxf(kmax[0], s.kn[k0 + r]);
    kmax[1] = fmaxf(kmax[1], s.kn[k0 + 16 + r]);
  }
  float dk[2][HD / 8][4], dv[2][HD / 8][4];
  zero_acc(dk[0]); zero_acc(dk[1]); zero_acc(dv[0]); zero_acc(dv[1]);
  for (int i0 = 0; i0 < a.M; i0 += 64) {
    const long long* t4[4] = {nullptr, nullptr, nullptr, nullptr};
    bool use[2] = {true, true};
    if (a.table) {
      float tmax = -INFINITY;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        t4[q] = tab_entry(a, s, i0 / 16 + q, k0 / KB);
        tmax = fmaxf(tmax, tab_max(t4[q]));
      }
      if (tmax < -1e30f) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          bool ok = true;
          for (int c = lane; c < 64; c += 32) {
            const float bound = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(s.qn[i0 + c], kmax[h])), SLACK);
            ok = ok && (tmax + bound < s.ms[i0 + c] - MARGIN);
          }
          use[h] = !__all_sync(0xffffffffu, ok);
        }
      }
    }
    if (!use[0] && !use[1]) continue;
    float p0[8][4], p1[8][4], d0[8][4], d1[8][4];
    tile_nt2(p0, p1, s.Ks, k0, s.Qs, i0);       // S^T of both key blocks
    tile_nt2(d0, d1, s.Vs, k0, s.Ds, i0);       // dP^T
    if (use[0]) { finish_scores_t(a, p0, z, k0, i0, t4); dkv_tile(a, s, p0, d0, i0); }
    if (use[1]) { finish_scores_t(a, p1, z, k0 + 16, i0, t4); dkv_tile(a, s, p1, d1, i0); }
    if (use[0] && use[1]) {
      tile_pv2(dv[0], p0, dv[1], p1, s.Ds, i0);
      tile_pv2(dk[0], d0, dk[1], d1, s.Qs, i0);
    } else if (use[0]) {
      tile_pv(dv[0], p0, s.Ds, i0);
      tile_pv(dk[0], d0, s.Qs, i0);
    } else {
      tile_pv(dv[1], p1, s.Ds, i0);
      tile_pv(dk[1], d1, s.Qs, i0);
    }
  }
  store_rows(dk_out, dk[0], z, k0, a.N);
  store_rows(dv_out, dv[0], z, k0, a.N);
  store_rows(dk_out, dk[1], z, k0 + 16, a.N);
  store_rows(dv_out, dv[1], z, k0 + 16, a.N);
}

// Phase: complete backward of every problem z in one item: delta rows (shared), then dQ rows and dK/dV keys.
PWIN_DEV void bwd_phase_v3(const Args& a, const Res& dq_out, const Res& dk_out, const Res& dv_out, float* smem) {
  Smem s = item_smem(a, smem);
  const int warp = threadIdx.x >> 5;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    load_item(a, s, z, true, true, true, false);
    if (PWIN_PAIR) {
      for (int rp = warp; rp < a.M / 32; rp += NW) delta_pair(a, s, z, 32 * rp, s.dl);
      __syncthreads();
      const int nrp = a.M / 32, nkp = a.N / 32;
      for (int w = warp; w < nrp + nkp; w += NW) {
        if (w < nrp) dq_pair(a, s, dq_out, z, 32 * w);
        else dkv_pair(a, s, dk_out, dv_out, z, 32 * (w - nrp));
      }
      continue;
    }
    for (int rb = warp; rb < a.M / 16; rb += NW) delta_rows(a, s, z, rb, s.dl);
    __syncthreads();
    // dQ row blocks and dK/dV key blocks as one work list (balanced over the warps).
    const int nrb = a.M / 16, nkb16 = a.N / 16;
    for (int w = warp; w < nrb + nkb16; w += NW) {
      if (w < nrb) dq_rows(a, s, dq_out, z, w);
      else dkv_keys(a, s, dk_out, dv_out, z, w - nrb);
    }
  }
}

// ===================================================================================== online arithmetic
// v2 (declared 2026-10-05 after v1 failed the w32 trained-state loss gate, P8-C gpu-27): exponentials are the
// accurate expf; PWIN_ONLINE_FASTEXP=1 reproduces v1 (__expf, hardware ex2).
#ifndef PWIN_ONLINE_FASTEXP
#define PWIN_ONLINE_FASTEXP 0
#endif
PWIN_DEV float oexp(float x) { return PWIN_ONLINE_FASTEXP ? __expf(x) : expf(x); }
// Declared by the precision policy `mixed_bf16_online` (attention='bfloat16_online'); ops/window_attention.py holds
// its eager definition. Rounding points kept from the mixed graph: s = bf16(q k^T), dP = bf16(dO v^T), bf16(P) as
// the value-product operand, dS = bf16(.), BF16 results. Changed (declared):
//   x   = fl(s * c') + mask                c' = c (x*c) or RN(1/c) (x/c)
//   per 16-row block, computed key blocks in the order of the exact kernel (unmasked blocks ascending, then masked
//   candidates ascending); running max m; p = oexp(x - m) (hardware ex2); l and O = sum bf16(p) v rescaled by
//   oexp(m_old - m_new) at each block (online softmax); O = bf16(O * RN(1/l)); stats (m, l) final.
//   backward: P = oexp(x - m) * RN(1/l); delta_i = sum_d dO_id O_id (FP32, d ascending);
//   dS = bf16(fl(P * fl(dP - delta)) * c'); dQ, dK, dV as in the exact kernel.
// Tile skipping is the exact kernel's test (skipped tiles contribute exactly zero here as well).
PWIN_DEV float scale_mul(const Args& a) {
  return a.scale_mode == 1 ? a.scale : (a.scale_mode == 2 ? a.rscale : 1.f);
}
PWIN_DEV void mask_tile(const Args& a, float (&s)[8][4], int z, int ri0, int kj0, const long long* t) {
  const int lane = threadIdx.x & 31;
  if (!a.mask || tab_zero(t)) return;
  if (tab_binary(t)) {
    const float v = tab_value(t);
    const unsigned long long bits[2] = {tab_bits(t, lane >> 2), tab_bits(t, (lane >> 2) + 8)};
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = 8 * n + 2 * (lane & 3) + (e & 1);
        if (!((bits[e >> 1] >> j) & 1ull)) s[n][e] = __fadd_rn(s[n][e], v);
      }
    return;
  }
  const long long mb = mask_base(a, z);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = ri0 + (lane >> 2) + 8 * (e >> 1), j = kj0 + 8 * n + 2 * (lane & 3) + (e & 1);
      s[n][e] = __fadd_rn(s[n][e], __ldg(a.mask + mb + (long long)i * a.mi + (long long)j * a.mj));
    }
}
PWIN_DEV void online_scores(const Args& a, float (&s)[8][4], int z, int ri0, int kj0, const long long* t) {
  const float c = scale_mul(a);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = __fmul_rn(rbf(s[n][e]), c);
  if (a.mask) mask_tile(a, s, z, ri0, kj0, t);
}
// Transposed tile (keys x 64 query rows), as finish_scores_t.
PWIN_DEV void online_scores_t(const Args& a, float (&s)[8][4], int z, int kj0, int ri0, const long long* const (&t4)[4]) {
  const int lane = threadIdx.x & 31;
  const float c = scale_mul(a);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = __fmul_rn(rbf(s[n][e]), c);
  if (!a.mask) return;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const long long* t = t4[n >> 1];
    if (tab_zero(t)) continue;
    if (tab_binary(t)) {
      const float v = tab_value(t);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jl = (kj0 % KB) + (lane >> 2) + 8 * (e >> 1);
        const int il = (8 * n + 2 * (lane & 3) + (e & 1)) & 15;
        if (!((tab_bits(t, il) >> jl) & 1ull)) s[n][e] = __fadd_rn(s[n][e], v);
      }
    } else {
      const long long mb = mask_base(a, z);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = kj0 + (lane >> 2) + 8 * (e >> 1), i = ri0 + 8 * n + 2 * (lane & 3) + (e & 1);
        s[n][e] = __fadd_rn(s[n][e], __ldg(a.mask + mb + (long long)i * a.mi + (long long)j * a.mj));
      }
    }
  }
}

PWIN_DEV void fwd_online_rows(const Args& a, const Smem& s, int z, int r0) {
  const int rb = r0 / 16, nkb = a.N / KB;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  float o[HD / 8][4];
  zero_acc(o);
  for (int pass = 0; pass < 2; ++pass) {
    for (int kb = 0; kb < nkb; ++kb) {
      float tmax;
      const long long* t;
      const int cls = block_class(a, s, rb, kb, tmax, t);
      if ((cls == 2) != (pass == 1)) continue;
      if (cls == 2 && skippable(a, s, r0, kb, tmax, m)) continue;
      float x[8][4];
      tile_nt(x, s.Qs, r0, s.Ks, kb * KB);
      online_scores(a, x, z, r0, kb * KB, t);
      float mt[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        mt[0] = fmaxf(mt[0], fmaxf(x[n][0], x[n][1]));
        mt[1] = fmaxf(mt[1], fmaxf(x[n][2], x[n][3]));
      }
      quad_max2(mt);
      float alpha[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float mn = fmaxf(m[h], mt[h]);
        alpha[h] = m[h] == -INFINITY ? 0.f : oexp(__fsub_rn(m[h], mn));
        m[h] = mn;
      }
      float ps[2] = {0.f, 0.f};
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          x[n][e] = oexp(__fsub_rn(x[n][e], m[e >> 1]));
          ps[e >> 1] = __fadd_rn(ps[e >> 1], x[n][e]);
        }
#pragma unroll
      for (int h = 0; h < 2; ++h) l[h] = __fadd_rn(__fmul_rn(l[h], alpha[h]), ps[h]);
#pragma unroll
      for (int nt = 0; nt < HD / 8; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[nt][e] = __fmul_rn(o[nt][e], alpha[e >> 1]);
      tile_pv(o, x, s.Vs, kb * KB);
    }
  }
  quad_sum2(l);
  const float rl[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
#pragma unroll
  for (int nt = 0; nt < HD / 8; ++nt)
#pragma unroll
    for (int e = 0; e < 4; ++e) o[nt][e] = __fmul_rn(o[nt][e], rl[e >> 1]);
  store_rows(a.out, o, z, r0, a.M);
  store_stats(a, z, r0, m, l);
}

PWIN_DEV void fwd_online_phase(const Args& a, float* smem) {
  Smem s = item_smem(a, smem);
  const int warp = threadIdx.x >> 5;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    load_item(a, s, z, true, false, false, false);
    for (int rb = warp; rb < a.M / 16; rb += NW) fwd_online_rows(a, s, z, 16 * rb);
  }
}

PWIN_DEV void dq_online_rows(const Args& a, const Smem& s, const Res& out, int z, int r0) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB;
  float m[2], l[2], rl[2], dl[2];
  stats_of(s, r0, m, l);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    rl[h] = s.rl[r0 + (lane >> 2) + 8 * h];
    dl[h] = s.dl[r0 + (lane >> 2) + 8 * h];
  }
  const float c = scale_mul(a);
  float dq[HD / 8][4];
  zero_acc(dq);
  for (int kb = 0; kb < nkb; ++kb) {
    const long long* t;
    if (!probs_needed(a, s, r0, kb, m, t)) continue;
    float p[8][4], dp[8][4];
    tile_nt(p, s.Qs, r0, s.Ks, kb * KB);
    online_scores(a, p, z, r0, kb * KB, t);
    tile_nt(dp, s.Ds, r0, s.Vs, kb * KB);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pr = __fmul_rn(oexp(__fsub_rn(p[n][e], m[e >> 1])), rl[e >> 1]);
        dp[n][e] = rbf(__fmul_rn(__fmul_rn(pr, __fsub_rn(rbf(dp[n][e]), dl[e >> 1])), c));
      }
    tile_pv(dq, dp, s.Ks, kb * KB);
  }
  store_rows(out, dq, z, r0, a.M);
}

PWIN_DEV void dkv_online_keys(const Args& a, const Smem& s, const Res& dk_out, const Res& dv_out, int z, int kb16) {
  const int lane = threadIdx.x & 31, k0 = 16 * kb16;
  const float c = scale_mul(a);
  float kmax = 0.f;
  for (int r = 0; r < 16; ++r) kmax = fmaxf(kmax, s.kn[k0 + r]);
  float dk[HD / 8][4], dv[HD / 8][4];
  zero_acc(dk);
  zero_acc(dv);
  for (int i0 = 0; i0 < a.M; i0 += 64) {
    const long long* t4[4] = {nullptr, nullptr, nullptr, nullptr};
    if (a.table) {
      float tmax = -INFINITY;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        t4[q] = tab_entry(a, s, i0 / 16 + q, k0 / KB);
        tmax = fmaxf(tmax, tab_max(t4[q]));
      }
      if (tmax < -1e30f) {
        bool ok = true;
        for (int cc = lane; cc < 64; cc += 32) {
          const float bound = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(s.qn[i0 + cc], kmax)), SLACK);
          ok = ok && (tmax + bound < s.ms[i0 + cc] - MARGIN);
        }
        if (__all_sync(0xffffffffu, ok)) continue;
      }
    }
    float p[8][4], dp[8][4];
    tile_nt(p, s.Ks, k0, s.Qs, i0);           // S^T: keys x 64 query rows
    online_scores_t(a, p, z, k0, i0, t4);
    tile_nt(dp, s.Vs, k0, s.Ds, i0);          // dP^T = V dO^T
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = i0 + 8 * n + 2 * (lane & 3) + (e & 1);
        p[n][e] = __fmul_rn(oexp(__fsub_rn(p[n][e], s.ms[i])), s.rl[i]);
        dp[n][e] = rbf(__fmul_rn(__fmul_rn(p[n][e], __fsub_rn(rbf(dp[n][e]), s.dl[i])), c));
      }
    tile_pv(dv, p, s.Ds, i0);                 // dV += bf16(P)^T dO
    tile_pv(dk, dp, s.Qs, i0);                // dK += dS^T q
  }
  store_rows(dk_out, dk, z, k0, a.N);
  store_rows(dv_out, dv, z, k0, a.N);
}

PWIN_DEV void bwd_online_phase(const Args& a, const Res& dq_out, const Res& dk_out, const Res& dv_out, float* smem) {
  Smem s = item_smem(a, smem);
  const int warp = threadIdx.x >> 5;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    load_item(a, s, z, true, true, true, false);
    // delta_i = sum_d dO_id O_id (dO from shared, exactly BF16; O rounded to BF16 on load when stored as FP32).
    for (int r = (int)threadIdx.x; r < a.M; r += NT) {
      const long long base = (long long)z * a.o.sz + (long long)r * a.o.sr;
      float acc = 0.f;
#pragma unroll 8
      for (int d = 0; d < HD; ++d) {
        const float ov = a.o.fp32 ? rbf(reinterpret_cast<const float*>(a.o.p)[base + (long long)d * a.o.sd])
                                  : __bfloat162float(a.o.p[base + (long long)d * a.o.sd]);
        acc = __fadd_rn(acc, __fmul_rn(__bfloat162float(s.Ds[r * LD + d]), ov));
      }
      s.dl[r] = acc;
    }
    __syncthreads();
    const int nrb = a.M / 16, nkb16 = a.N / 16;
    for (int w = warp; w < nrb + nkb16; w += NW) {
      if (w < nrb) dq_online_rows(a, s, dq_out, z, 16 * w);
      else dkv_online_keys(a, s, dk_out, dv_out, z, w - nrb);
    }
  }
}

#include "attention_window_v4.cuh"

}  // namespace pwin
