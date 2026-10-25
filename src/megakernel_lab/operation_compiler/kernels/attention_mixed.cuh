// P-OPS5-D fused mixed-precision short attention work items for the LATCH-OPS cooperative program.
//
// Scope: self-attention training groups produced by the compiler's `mixed_bf16` precision policy (and identical
// explicit-cast programs): BF16 query/key/value/output-gradient operands, BF16 products with FP32 accumulation
// and one BF16 result rounding, FP32 scale and softmax. Head width HD in {32, 64}, context T <= 128 (zero padded
// to TP = 16 * NB rows), no mask, a captured scalar score scale. One work item is one (batch, head); the whole
// head lives in shared memory, so no score, probability or gradient matrix is ever stored globally.
//
// Arithmetic (declared; the rounding points of the captured graph are reproduced exactly):
//   s   = bf16(q k^T)                    FP32 accumulation from +0 (mma.sync m16n8k16), one RNE rounding
//   x   = fl(s * c) | fl(s / c)          the captured scale
//   P   = fl(expf(x - m) / l)            m = row max, l = sum_j expf(x_j - m) (fixed order); FP32 softmax
//   O   = bf16(bf16(P) v)
//   dP  = bf16(dO v^T)
//   dS  = fl(fl(P * fl(dP - delta)) op c), delta = sum_j P_j dP_j (fixed order) as _softmax_backward_data
//   dQ  = bf16(bf16(dS) k), dK = bf16(bf16(dS)^T q), dV = bf16(bf16(P)^T dO)
// The backward recomputes P from the saved row (m, l) with the forward's own code, so it equals the forward P
// bit for bit. Every product of BF16 operands is exact; only FP32 summation order differs from other kernels
// (FP32 association). Absorbed bias-gradient column sums use per-item partials finished either by a separate
// fixed-tree bias phase (default) or, with counters, by two fixed-order levels finished after each CTA's last
// item by the last arriver (counter atomics only, one fence per CTA). Operand sources: a BF16 value, a BF16 product plus an FP32 bias vector rounded to BF16
// (bf16(fl(widen(x) + b)), the policy's biased linear), or an FP32 value rounded to BF16. Results are written
// rounded to BF16, or widened exactly to FP32, at arbitrary 4-D (batch, head, row, unit) strides, so the head
// split/merge copies, casts and concatenations around the attention are absorbed. Every data element has
// exactly one writer. Requires the program's 256-thread CTA.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace pmix {

#define PMIX_DEV __device__ __forceinline__
constexpr int NT = 256;
constexpr int NW = 8;

struct Operand {
  const void* src;      // BF16 or FP32 value pointer (element 0 of the value)
  const float* bias;    // kind 1: FP32 bias value pointer
  long long sb, sh, sr; // element strides of (batch, head, row); unit stride along the head width
  long long bb, bh;     // kind 1: bias strides of (batch, head); unit stride along the head width
  int kind;             // 0 BF16, 1 BF16 + FP32 bias rounded to BF16, 2 FP32 rounded to BF16
};

struct Result {
  void* dst;            // BF16 or FP32 value pointer
  long long sb, sh, sr;
  int fp32;             // 1: exact FP32 widening of the BF16 result
};

struct Reduce {
  float* dst;           // column sum over (batch, row) of one gradient (its bias gradient); nullptr: none
  long long o, sh;      // element offset of (head, unit) in dst: o + sh * head + unit
};

struct Args {
  Operand q, k, v, dout;
  Result o, dq, dk, dv;
  float* stats;         // [BH][T][2]: row max m and row sum l of the forward softmax
  int heads, t;
  float scale;
  int scale_mode;       // 0 none, 1 x*scale, 2 x/scale
  // Absorbed bias-gradient reductions of dq/dk/dv (deterministic two-level sums, fixed order):
  Reduce red[3];
  float* partials;               // [batches][3][heads][HD] per-item column sums
  float* gpartials;              // [groups][3][heads][HD] per-group sums
  unsigned long long* counters;  // [heads][groups] then [heads]; zeroed by the forward phase
  int batches, group, ncounters;
  long long* trace;              // diagnostic builds (PMIX_TRACE): CTA 0 SM-cycle stamps [phase][item<8][slot<16]
  float* probs;                  // optional saved FP32 probabilities, fragment-native [BH][NB][TP/8][32 lanes][4]
  __nv_bfloat16* pds;            // P8-E long backward: bf16(P)^T and bf16(dS)^T per item, [BH][2][TP keys][TP queries]
  unsigned* flags;               // P8-E long backward stage split: [BH] row stage done (zero at entry, restored to zero)
  float* halves;                 // P9-E v3 backward: per-warp bias column partials [BH][2 halves][3][NW][HD]
  Operand ofwd;                  // P9-E PMIX_FA_DELTA: the forward output O read by the long backward
};

// Address of one lane's 4 accumulator values of n-tile n in row block blk of item bh (fragment-native layout:
// a warp's 32 lanes x 4 floats of one n-tile are 512 contiguous bytes, so stores and loads are 16-byte coalesced).
template <int TP>
PMIX_DEV float4* probs_at(const Args& a, int bh, int blk, int n) {
  const int lane = threadIdx.x & 31;
  return reinterpret_cast<float4*>(a.probs) + ((((long long)bh * (TP / 16) + blk) * (TP / 8) + n) * 32 + lane);
}

#ifndef PMIX_TRACE
#define PMIX_TRACE 0
#endif
// Stamp by lane 0 of `warp_id` in CTA 0 (diagnostic builds only; never the metric).
#define PMIX_STAMP(a, phase, j, slot, warp_id)                                                              \
  do {                                                                                                   \
    if (PMIX_TRACE && (a).trace && blockIdx.x == 0 && (j) < 8 && threadIdx.x == 32 * (warp_id))           \
      (a).trace[((phase) * 8 + (j)) * 16 + (slot)] = (long long)clock64();                               \
  } while (0)

PMIX_DEV uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

PMIX_DEV void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
PMIX_DEV void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}

PMIX_DEV void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

PMIX_DEV uint32_t pack(float x0, float x1) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
  return *reinterpret_cast<const uint32_t*>(&h);
}
PMIX_DEV float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

PMIX_DEV float apply_scale(const Args& a, float x) {
  return a.scale_mode == 1 ? __fmul_rn(x, a.scale) : (a.scale_mode == 2 ? __fdiv_rn(x, a.scale) : x);
}

// P9-E declared precision option (p9e_fast_softmax, SELECTION amendment): softmax exponentials as the hardware
// ex2.approx.ftz.f32 of fl(d * log2(e)) and probabilities as fl(e * fl(1 / l)) (one correctly rounded reciprocal per
// row) instead of libm expf and IEEE division. Forward and backward recompute P with the same code, so the backward
// still sees the forward's P bit for bit. Default 0: the declared exact arithmetic above.
#ifndef PMIX_FAST_SOFTMAX
#define PMIX_FAST_SOFTMAX 0
#endif
PMIX_DEV float pmix_exp(float d) {
#if PMIX_FAST_SOFTMAX
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(__fmul_rn(d, 1.4426950408889634f)));
  return y;
#else
  return expf(d);
#endif
}
PMIX_DEV float pmix_rinv(float l) {
#if PMIX_FAST_SOFTMAX
  return __frcp_rn(l);
#else
  return l;
#endif
}
// fl(e / l) (exact) or fl(e * li) with li = pmix_rinv(l) (fast)
PMIX_DEV float pmix_div(float e, float l, float li) {
#if PMIX_FAST_SOFTMAX
  (void)l;
  return __fmul_rn(e, li);
#else
  (void)li;
  return __fdiv_rn(e, l);
#endif
}

// Rows [0, t) of one (batch, head) operand into shared BF16 [TP][HD + 8]; rows [t, TP) zero.
template <int HD, int TP>
PMIX_DEV void load_operand(__nv_bfloat16* dst, const Operand& op, int b, int h, int t) {
  constexpr int LD = HD + 8, CH = HD / 8, TOTAL = TP * CH;
  const long long base = (long long)b * op.sb + (long long)h * op.sh;
  const long long bias_base = (long long)b * op.bb + (long long)h * op.bh;
  for (int idx = (int)threadIdx.x; idx < TOTAL; idx += NT) {
    const int r = idx / CH, c = 8 * (idx % CH);
    uint4 out = make_uint4(0u, 0u, 0u, 0u);
    if (r < t) {
      const long long at = base + (long long)r * op.sr + c;
      if (op.kind == 2) {
        const float4 x0 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(op.src) + at));
        const float4 x1 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(op.src) + at + 4));
        out = make_uint4(pack(x0.x, x0.y), pack(x0.z, x0.w), pack(x1.x, x1.y), pack(x1.z, x1.w));
      } else {
        out = __ldg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(op.src) + at));
        if (op.kind == 1) {
          const float4 b0 = __ldg(reinterpret_cast<const float4*>(op.bias + bias_base + c));
          const float4 b1 = __ldg(reinterpret_cast<const float4*>(op.bias + bias_base + c + 4));
          const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&out);
          const float2 x0 = __bfloat1622float2(x[0]), x1 = __bfloat1622float2(x[1]);
          const float2 x2 = __bfloat1622float2(x[2]), x3 = __bfloat1622float2(x[3]);
          out = make_uint4(pack(__fadd_rn(x0.x, b0.x), __fadd_rn(x0.y, b0.y)),
                           pack(__fadd_rn(x1.x, b0.z), __fadd_rn(x1.y, b0.w)),
                           pack(__fadd_rn(x2.x, b1.x), __fadd_rn(x2.y, b1.y)),
                           pack(__fadd_rn(x3.x, b1.z), __fadd_rn(x3.y, b1.w)));
        }
      }
    }
    *reinterpret_cast<uint4*>(dst + r * LD + c) = out;
  }
}

// One warp's 16 rows x HD/8 tiles of a BF16 result (accumulators rounded once) into a strided destination.
template <int HD>
PMIX_DEV void store_rows(const Result& res, const float (&acc)[HD / 8][4], int b, int h, int row0, int t) {
  const int lane = threadIdx.x & 31, r = row0 + (lane >> 2), c = 2 * (lane & 3);
  const long long base = (long long)b * res.sb + (long long)h * res.sh;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = r + 8 * half;
    if (row >= t) continue;
    const long long at = base + (long long)row * res.sr + c;
#pragma unroll
    for (int nt = 0; nt < HD / 8; ++nt) {
      const float x0 = round_bf16(acc[nt][2 * half]), x1 = round_bf16(acc[nt][2 * half + 1]);
      if (res.fp32) *reinterpret_cast<float2*>(static_cast<float*>(res.dst) + at + 8 * nt) = make_float2(x0, x1);
      else *reinterpret_cast<uint32_t*>(static_cast<__nv_bfloat16*>(res.dst) + at + 8 * nt) = pack(x0, x1);
    }
  }
}

// Scores of the warp's 16 rows against all TP keys: acc[n][.] = q_rows k^T (FP32 accumulation).
template <int HD, int TP>
PMIX_DEV void row_products(float (&acc)[TP / 8][4], const __nv_bfloat16* A, const __nv_bfloat16* B, int row0) {
  constexpr int LD = HD + 8;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < TP / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks) {
    uint32_t a[4];
    ldsm_x4(a, A + (row0 + (lane & 15)) * LD + 16 * ks + 8 * (lane >> 4));
#pragma unroll
    for (int n2 = 0; n2 < TP / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4(bb, B + (16 * n2 + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * ks + 8 * ((lane >> 3) & 1));
      mma(acc[2 * n2], a, bb[0], bb[1]);
      mma(acc[2 * n2 + 1], a, bb[2], bb[3]);
    }
  }
}

// Softmax of the warp's rows (scores already in acc): P (FP32) in place; returns per-thread (m, l) of rows r, r+8.
template <int TP>
PMIX_DEV void softmax_rows(const Args& a, float (&s)[TP / 8][4], float (&m)[2], float (&l)[2], bool given) {
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
#pragma unroll
  for (int n = 0; n < TP / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = 8 * n + c + (e & 1);
      s[n][e] = col < a.t ? apply_scale(a, round_bf16(s[n][e])) : -INFINITY;
    }
  if (!given) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float mx = -INFINITY;
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * half], s[n][2 * half + 1]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      m[half] = mx;
    }
  }
#pragma unroll
  for (int n = 0; n < TP / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = pmix_exp(__fsub_rn(s[n][e], m[e >> 1]));
  if (!given) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) sum = __fadd_rn(__fadd_rn(sum, s[n][2 * half]), s[n][2 * half + 1]);
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
      l[half] = sum;
    }
  }
  const float li[2] = {pmix_rinv(l[0]), pmix_rinv(l[1])};
#pragma unroll
  for (int n = 0; n < TP / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = pmix_div(s[n][e], l[e >> 1], li[e >> 1]);
}

// Column sums of one warp's 16 rows (BF16-rounded values, rows < t) into colsum[HD] (lanes 0..3 write).
#ifndef PMIX_DIAG_NO_COLSUM
#define PMIX_DIAG_NO_COLSUM 0  // diagnostic timing builds only: skip the column sums (invalid bias gradients)
#endif
#ifndef PMIX_DIAG_NO_FINISH
#define PMIX_DIAG_NO_FINISH 0  // diagnostic timing builds only: skip the cross-CTA reduction protocol
#endif
template <int HD>
PMIX_DEV void column_sums(float* colsum, const float (&acc)[HD / 8][4], int row0, int t) {
  if (PMIX_DIAG_NO_COLSUM) return;
  const int lane = threadIdx.x & 31, r = row0 + (lane >> 2);
#pragma unroll
  for (int nt = 0; nt < HD / 8; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float x = __fadd_rn(r < t ? round_bf16(acc[nt][e]) : 0.f, r + 8 < t ? round_bf16(acc[nt][2 + e]) : 0.f);
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 4));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 8));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 16));
      if (lane < 4) colsum[8 * nt + 2 * lane + e] = x;
    }
  }
}

// Fixed-order sum of src[i * stride] for i in [begin, end): loads issued in batches of 8 independent requests.
// Reduction slot k by value selection (never a runtime-indexed struct access, which would move Args to local memory).
PMIX_DEV float* red_dst(const Args& a, int k) { return k == 0 ? a.red[0].dst : (k == 1 ? a.red[1].dst : a.red[2].dst); }
PMIX_DEV long long red_off(const Args& a, int k, int h) {
  return k == 0 ? a.red[0].o + a.red[0].sh * h : (k == 1 ? a.red[1].o + a.red[1].sh * h : a.red[2].o + a.red[2].sh * h);
}

PMIX_DEV float ordered_sum(const float* src, long long stride, int begin, int end) {
  float x = 0.f;
  for (int i0 = begin; i0 < end; i0 += 8) {
    float v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) v[u] = i0 + u < end ? __ldcg(src + (long long)(i0 + u) * stride) : 0.f;
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (i0 + u < end) x = __fadd_rn(x, v[u]);
  }
  return x;
}

// After the item's column sums are in colsum[3][NW][HD] (all threads, after a CTA barrier): write the item's
// partial column sums. No fence here: the cross-CTA protocol runs once per CTA at the end of the phase.
template <int HD, int NB>
PMIX_DEV void reduce_item(const Args& a, int b, int h, const float* colsum, int* /*flag*/, int stride = NT,
                          int tbase = 0, int kmask = 7) {
  if (!a.red[0].dst && !a.red[1].dst && !a.red[2].dst) return;
  const long long width = 3LL * a.heads * HD;
  for (int tid = (int)threadIdx.x - tbase; tid < 3 * HD; tid += stride) {
    const int k = tid / HD, d = tid % HD;
    if (red_dst(a, k) == nullptr || !((kmask >> k) & 1)) continue;
    float x = 0.f;
#pragma unroll
    for (int w = 0; w < NB; ++w) x = __fadd_rn(x, colsum[(k * NW + w) * HD + d]);
    a.partials[(long long)b * width + ((long long)k * a.heads + h) * HD + d] = x;
  }
}

// End of the backward phase: one fence per CTA, then for each item this CTA processed, count it in its batch group;
// the last item of a group sums the group's partials (fixed batch order), and the last group of a head sums the
// group partials (fixed group order) into the reduction output. Deterministic and independent of the grid.
template <int HD>
PMIX_DEV void finish_reductions(const Args& a, long long items, int* flag) {
#if PMIX_DIAG_NO_FINISH
  return;
#endif
  if ((!a.red[0].dst && !a.red[1].dst && !a.red[2].dst) || !a.counters) return;
  const int tid = threadIdx.x;
  const int ng = (a.batches + a.group - 1) / a.group;
  const long long width = 3LL * a.heads * HD;
  const int k = tid / HD, d = tid % HD;
  const bool on = tid < 3 * HD && red_dst(a, k) != nullptr;
  const int mine = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;  // this CTA's items
  __threadfence();
  __syncthreads();
  // Level 1: every item of this CTA is counted in its batch group at once (thread j handles item j).
  for (int j0 = 0; j0 < mine; j0 += NT) {
    const int j = j0 + tid;
    if (j < mine && j < 64) {
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads), g = b / a.group;
      const int members = min(a.group, a.batches - g * a.group);
      flag[2 + j] = atomicAdd(a.counters + (long long)h * ng + g, 1ULL) == (unsigned long long)(members - 1);
    }
  }
  __syncthreads();
  for (int j = 0; j < mine && j < 64; ++j) {
    if (!flag[2 + j]) continue;   // block-uniform (shared flag)
    const long long item = blockIdx.x + (long long)j * gridDim.x;
    const int b = (int)(item / a.heads), h = (int)(item % a.heads), g = b / a.group;
    const long long col = ((long long)k * a.heads + h) * HD + d;
    __threadfence();
    if (on) a.gpartials[(long long)g * width + col] =
        ordered_sum(a.partials + col, width, g * a.group, min(a.batches, (g + 1) * a.group));
    __threadfence();
    __syncthreads();
    if (tid == 0)
      flag[1] = atomicAdd(a.counters + (long long)a.heads * ng + h, 1ULL) == (unsigned long long)(ng - 1);
    __syncthreads();
    if (flag[1]) {
      __threadfence();
      if (on) red_dst(a, k)[red_off(a, k, h) + d] = ordered_sum(a.gpartials + col, width, 0, ng);
    }
    __syncthreads();
  }
}

// Bias-gradient phase (after the backward phase's grid barrier): column sums of the per-item partials
// [batches][3][heads][HD] in a fixed tree order independent of the grid (8 columns per item: 32 batch lanes x
// strided batches in order, then the 32 lane sums in order). No atomics, no fences.
template <int HD>
PMIX_DEV void bias_phase(const Args& a, float* smem) {
  const long long width = 3LL * a.heads * HD;
  const int tid = threadIdx.x, c8 = tid & 7, lane32 = tid >> 3;   // 32 batch lanes x 8 columns
  const long long groups = width / 8;
  for (long long g8 = blockIdx.x; g8 < groups; g8 += gridDim.x) {
    const long long col = 8 * g8 + c8;
    const int k = (int)(col / ((long long)a.heads * HD)), h = (int)((col / HD) % a.heads), d = (int)(col % HD);
    float* dst = red_dst(a, k);
    if (dst != nullptr) {
      float x = 0.f;
      for (int b = lane32; b < a.batches; b += 32) x = __fadd_rn(x, a.partials[(long long)b * width + col]);
      smem[lane32 * 8 + c8] = x;
    }
    __syncthreads();
    if (tid < 8 && dst != nullptr) {
      float x = 0.f;
#pragma unroll 8
      for (int j = 0; j < 32; ++j) x = __fadd_rn(x, smem[j * 8 + c8]);
      dst[red_off(a, k, h) + d] = x;
    }
    __syncthreads();
  }
}

PMIX_DEV void zero_counters(const Args& a) {
  if (a.counters && blockIdx.x == 0)
    for (int i = (int)threadIdx.x; i < a.ncounters; i += NT) a.counters[i] = 0ULL;
}

// ---------------------------------------------------------------------------------------------------- forward
template <int HD, int TP>
__host__ __device__ constexpr int forward_shared_bytes() { return 2 * 3 * TP * (HD + 8); }

template <int HD, int TP>
PMIX_DEV void forward_compute(const Args& a, int bh, const __nv_bfloat16* Qs, const __nv_bfloat16* Ks,
                              const __nv_bfloat16* Vs, int wbase = 0) {
  constexpr int LD = HD + 8, NB = TP / 16;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = (int)(threadIdx.x >> 5) - wbase, lane = threadIdx.x & 31;
  if (warp < NB) {
    const int row0 = 16 * warp;
    float s[TP / 8][4];
    row_products<HD, TP>(s, Qs, Ks, row0);
    float m[2], l[2];
    softmax_rows<TP>(a, s, m, l, false);
    if (a.probs) {
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) *probs_at<TP>(a, bh, warp, n) = make_float4(s[n][0], s[n][1], s[n][2], s[n][3]);
    }
    float o[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < TP / 16; ++kk) {
      const uint32_t p[4] = {pack(s[2 * kk][0], s[2 * kk][1]), pack(s[2 * kk][2], s[2 * kk][3]),
                             pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t bb[4];
        ldsm_x4_t(bb, Vs + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(o[2 * n2], p, bb[0], bb[1]);
        mma(o[2 * n2 + 1], p, bb[2], bb[3]);
      }
    }
    store_rows<HD>(a.o, o, b, h, row0, t);
    if ((lane & 3) == 0) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = row0 + (lane >> 2) + 8 * half;
        if (row < t) {
          float2* stat = reinterpret_cast<float2*>(a.stats) + (long long)bh * t + row;
          *stat = make_float2(m[half], l[half]);
        }
      }
    }
  }
}

template <int HD, int TP>
PMIX_DEV void forward_item(const Args& a, int bh, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16;
  static_assert(NB <= NW && TP % 16 == 0 && (HD == 32 || HD == 64), "unsupported mixed attention shape");
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + TP * LD;
  __nv_bfloat16* Vs = Ks + TP * LD;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  load_operand<HD, TP>(Qs, a.q, b, h, t);
  load_operand<HD, TP>(Ks, a.k, b, h, t);
  load_operand<HD, TP>(Vs, a.v, b, h, t);
  __syncthreads();
  forward_compute<HD, TP>(a, bh, Qs, Ks, Vs);
}

// --------------------------------------------------------------------------------------------------- backward
template <int HD, int TP>
__host__ __device__ constexpr int backward_shared_bytes() {
  return 2 * (4 * TP * (HD + 8) + 2 * TP * (TP + 8)) + 4 * 2 * TP;
}

PMIX_DEV void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
PMIX_DEV void cp8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_u32(dst)), "l"(src));
}
PMIX_DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
PMIX_DEV void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// ------------------------------------------------------------------------------------- producer/consumer roles
// PMIX_PRODUCER=1 (when NB < 8): warps NB..7 stage the next items' operands with cp.async and complete biased
// operands in shared memory, handing buffers over with named barriers (1, 2: full; 3, 4: empty; 5: producers;
// 6: consumers); warps 0..NB-1 only compute. Every arrival is matched by a wait on the same barrier.
#ifndef PMIX_PRODUCER
#define PMIX_PRODUCER 1
#endif
// P8-E: two (batch, head) items per CTA at once when TP == 64 (see "dual items" below).
#ifndef PMIX_DUAL
#define PMIX_DUAL 1
#endif
// P8-E: the dual backward has no room for a second operand buffer; it prefetches the half's next item's operand
// rows into L2 (prefetch.global.L2) while the current item computes. Data movement hint only.
#ifndef PMIX_DUAL_PREFETCH
#define PMIX_DUAL_PREFETCH 0  // measured +12 us on DiT-S (gpu-04): off by default
#endif
#ifndef PMIX_FWD_SPLIT
#define PMIX_FWD_SPLIT 0  // measured (gpu-12): forward 24.3 vs 20.2 us on ViT-S; kept as an option
#endif
PMIX_DEV void nbar_sync(int id, int count) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }
PMIX_DEV void nbar_arrive(int id, int count) { asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }

template <bool PC, int NB>
PMIX_DEV void csync(int bar = 6) {
  if (PC) nbar_sync(bar, 32 * NB);
  else __syncthreads();
}

// Rows [0, t) of one operand into shared BF16 [TP][HD + 8] by a thread subset: cp.async for BF16 sources
// (bias completed later by finish_operand_sub), register conversion for FP32 sources; rows [t, TP) zero.
template <int HD, int TP>
PMIX_DEV void stage_operand_sub(__nv_bfloat16* dst, float* bias_dst, const Operand& op, int b, int h, int t,
                                int tid, int nth) {
  constexpr int LD = HD + 8, CH = HD / 8, TOTAL = TP * CH;
  const long long base = (long long)b * op.sb + (long long)h * op.sh;
  for (int idx = tid; idx < TOTAL; idx += nth) {
    const int r = idx / CH, c = 8 * (idx % CH);
    if (r >= t) {
      *reinterpret_cast<uint4*>(dst + r * LD + c) = make_uint4(0u, 0u, 0u, 0u);
    } else if (op.kind == 2) {
      const long long at = base + (long long)r * op.sr + c;
      const float4 x0 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(op.src) + at));
      const float4 x1 = __ldg(reinterpret_cast<const float4*>(static_cast<const float*>(op.src) + at + 4));
      *reinterpret_cast<uint4*>(dst + r * LD + c) =
          make_uint4(pack(x0.x, x0.y), pack(x0.z, x0.w), pack(x1.x, x1.y), pack(x1.z, x1.w));
    } else {
      cp16(dst + r * LD + c, static_cast<const __nv_bfloat16*>(op.src) + base + (long long)r * op.sr + c);
    }
  }
  if (op.kind == 1)
    for (int i = tid; i < HD / 4; i += nth)
      cp16(bias_dst + 4 * i, op.bias + (long long)b * op.bb + (long long)h * op.bh + 4 * i);
}

// P9-E PMIX_V3_DIRECT: rows [0, t) of one operand into shared BF16 [TP][HD + 8] in a single pass: every thread's
// 16-byte chunks are loaded first (all in flight), then completed (bias added and rounded exactly as
// finish_operand_sub does) and stored; rows [t, TP) zero. Operand fields are read once into registers (a generic
// shared-memory store could otherwise alias the operand struct and force reloads). Same values as
// stage_operand_sub + finish_operand_sub.
template <int HD, int TP>
PMIX_DEV void stage_operand_direct(__nv_bfloat16* dst, const Operand& op, int b, int h, int t, int tid) {
  constexpr int LD = HD + 8, CH = HD / 8, TOTAL = TP * CH, PER = (TOTAL + NT - 1) / NT;
  const int kind = op.kind;
  if (kind == 2) {   // FP32 sources convert in registers already (no completion pass)
    stage_operand_sub<HD, TP>(dst, nullptr, op, b, h, t, tid, NT);
    return;
  }
  const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(op.src) + (long long)b * op.sb + (long long)h * op.sh;
  const long long sr = op.sr;
  const float* bias = kind == 1 ? op.bias + (long long)b * op.bb + (long long)h * op.bh : nullptr;
  uint4 v[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int idx = tid + u * NT, r = idx / CH, c = 8 * (idx % CH);
    v[u] = make_uint4(0u, 0u, 0u, 0u);
    if (idx < TOTAL && r < t) v[u] = __ldg(reinterpret_cast<const uint4*>(src + (long long)r * sr + c));
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int idx = tid + u * NT, r = idx / CH, c = 8 * (idx % CH);
    if (idx >= TOTAL) continue;
    uint4 w = v[u];
    if (kind == 1 && r < t) {
      const float4 b0 = __ldg(reinterpret_cast<const float4*>(bias + c));
      const float4 b1 = __ldg(reinterpret_cast<const float4*>(bias + c + 4));
      const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&w);
      const float2 x0 = __bfloat1622float2(x[0]), x1 = __bfloat1622float2(x[1]);
      const float2 x2 = __bfloat1622float2(x[2]), x3 = __bfloat1622float2(x[3]);
      w = make_uint4(pack(__fadd_rn(x0.x, b0.x), __fadd_rn(x0.y, b0.y)), pack(__fadd_rn(x1.x, b0.z), __fadd_rn(x1.y, b0.w)),
                     pack(__fadd_rn(x2.x, b1.x), __fadd_rn(x2.y, b1.y)), pack(__fadd_rn(x3.x, b1.z), __fadd_rn(x3.y, b1.w)));
    }
    *reinterpret_cast<uint4*>(dst + r * LD + c) = w;
  }
}

template <int HD, int TP>
PMIX_DEV void finish_operand_sub(__nv_bfloat16* dst, const float* bias, const Operand& op, int t, int tid, int nth) {
  constexpr int LD = HD + 8, CH = HD / 8;
  if (op.kind != 1) return;
  for (int idx = tid; idx < t * CH; idx += nth) {
    const int r = idx / CH, c = 8 * (idx % CH);
    uint4* p = reinterpret_cast<uint4*>(dst + r * LD + c);
    const uint4 raw = *p;
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 x0 = __bfloat1622float2(x[0]), x1 = __bfloat1622float2(x[1]);
    const float2 x2 = __bfloat1622float2(x[2]), x3 = __bfloat1622float2(x[3]);
    *p = make_uint4(pack(__fadd_rn(x0.x, bias[c]), __fadd_rn(x0.y, bias[c + 1])),
                    pack(__fadd_rn(x1.x, bias[c + 2]), __fadd_rn(x1.y, bias[c + 3])),
                    pack(__fadd_rn(x2.x, bias[c + 4]), __fadd_rn(x2.y, bias[c + 5])),
                    pack(__fadd_rn(x3.x, bias[c + 6]), __fadd_rn(x3.y, bias[c + 7])));
  }
}

// ------------------------------------------------------------------------------ backward, split across all warps
// PMIX_BWD_SPLIT=1 (when NB < 8): stage A1 - row warps recompute P while the remaining warps compute the BF16
// probability gradient dP = dO v^T of every row block into the dS staging buffer; A2 - row warps form dS in place
// and bf16(P); B - dQ, dK and dV are 3 * NB * (HD / 32) equal tasks (16 rows x 32 columns) over all eight warps.
// Arithmetic and summation orders are those of backward_compute (same products, same k order).
#ifndef PMIX_BWD_SPLIT
#define PMIX_BWD_SPLIT 0  // measured (gpu-07): 1.5-2 us slower than the row/key-block backward on ViT-S/DiT-S
#endif

// acc[NN][4] = A(16 x 16 KS) B(16 KS x 8 NN); B stored [k][n]; A stored [m][k] (AT=false) or [k][m] (AT=true).
template <int KS, int NN, bool AT>
PMIX_DEV void tile_product(float (&acc)[NN][4], const __nv_bfloat16* A, int lda, const __nv_bfloat16* B, int ldb) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int n = 0; n < NN; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  const int mi = lane >> 3, kr = (lane & 7) + 8 * (mi >> 1), kc = 8 * (mi & 1);
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    uint32_t a[4];
    if (AT) ldsm_x4_t(a, A + (16 * ks + kr) * lda + kc);
    else ldsm_x4(a, A + (lane & 15) * lda + 16 * ks + 8 * (lane >> 4));
#pragma unroll
    for (int n2 = 0; n2 < NN / 2; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, B + (16 * ks + (lane & 7) + 8 * ((lane >> 3) & 1)) * ldb + 16 * n2 + 8 * (lane >> 4));
      mma(acc[2 * n2], a, bb[0], bb[1]);
      mma(acc[2 * n2 + 1], a, bb[2], bb[3]);
    }
  }
}

// 16 rows x 8 NN columns (from column d0) of a BF16-rounded result into a strided destination.
template <int NN>
PMIX_DEV void store_cols(const Result& res, const float (&acc)[NN][4], int b, int h, int row0, int t, int d0) {
  const int lane = threadIdx.x & 31, r = row0 + (lane >> 2), c = d0 + 2 * (lane & 3);
  const long long base = (long long)b * res.sb + (long long)h * res.sh;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = r + 8 * half;
    if (row >= t) continue;
    const long long at = base + (long long)row * res.sr + c;
#pragma unroll
    for (int nt = 0; nt < NN; ++nt) {
      const float x0 = round_bf16(acc[nt][2 * half]), x1 = round_bf16(acc[nt][2 * half + 1]);
      if (res.fp32) *reinterpret_cast<float2*>(static_cast<float*>(res.dst) + at + 8 * nt) = make_float2(x0, x1);
      else *reinterpret_cast<uint32_t*>(static_cast<__nv_bfloat16*>(res.dst) + at + 8 * nt) = pack(x0, x1);
    }
  }
}

template <int NN>
PMIX_DEV void column_sums_cols(float* colsum, const float (&acc)[NN][4], int row0, int t) {
  if (PMIX_DIAG_NO_COLSUM) return;
  const int lane = threadIdx.x & 31, r = row0 + (lane >> 2);
#pragma unroll
  for (int nt = 0; nt < NN; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float x = __fadd_rn(r < t ? round_bf16(acc[nt][e]) : 0.f, r + 8 < t ? round_bf16(acc[nt][2 + e]) : 0.f);
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 4));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 8));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 16));
      if (lane < 4) colsum[8 * nt + 2 * lane + e] = x;
    }
  }
}

template <int HD, int TP>
PMIX_DEV void backward_compute_split(const Args& a, int bh, const __nv_bfloat16* Qs, const __nv_bfloat16* Ks,
                                     const __nv_bfloat16* Vs, const __nv_bfloat16* Ds, const float* stat,
                                     __nv_bfloat16* Ps, __nv_bfloat16* Ss, float* colsum) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16, HALVES = HD / 32;
  static_assert(NB < NW, "split backward needs a spare warp");
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = 2 * (lane & 3);
  float p[TP / 8][4];
  // A1: P on row warps; dP = bf16(dO v^T) of every row block on the spare warps (into the dS staging buffer).
  if (warp < NB) {
    const int r = 16 * warp + (lane >> 2);
    row_products<HD, TP>(p, Qs, Ks, 16 * warp);
    float m[2] = {stat[2 * r], stat[2 * (r + 8)]}, l[2] = {stat[2 * r + 1], stat[2 * (r + 8) + 1]};
    softmax_rows<TP>(a, p, m, l, true);
  } else {
    for (int blk = warp - NB; blk < NB; blk += NW - NB) {
      float dp[TP / 8][4];
      row_products<HD, TP>(dp, Ds, Vs, 16 * blk);
      const int r = 16 * blk + (lane >> 2);
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) {
        *reinterpret_cast<uint32_t*>(Ss + r * LP + 8 * n + c) = pack(dp[n][0], dp[n][1]);
        *reinterpret_cast<uint32_t*>(Ss + (r + 8) * LP + 8 * n + c) = pack(dp[n][2], dp[n][3]);
      }
    }
  }
  __syncthreads();
  // A2: row warps form delta, dS (in place in the staging buffer) and bf16(P).
  if (warp < NB) {
    const int r = 16 * warp + (lane >> 2);
    float delta[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = r + 8 * half;
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) {
        const float2 d2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(Ss + row * LP + 8 * n + c));
        if (8 * n + c < t) sum = __fadd_rn(sum, __fmul_rn(d2.x, p[n][2 * half]));
        if (8 * n + c + 1 < t) sum = __fadd_rn(sum, __fmul_rn(d2.y, p[n][2 * half + 1]));
      }
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
      delta[half] = sum;
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = r + 8 * half;
      const bool live = row < t;   // padded query rows contribute nothing
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) {
        __nv_bfloat162* slot = reinterpret_cast<__nv_bfloat162*>(Ss + row * LP + 8 * n + c);
        const float2 d2 = __bfloat1622float2(*slot);
        const float p0 = p[n][2 * half], p1 = p[n][2 * half + 1];
        const float s0 = apply_scale(a, __fmul_rn(p0, __fsub_rn(d2.x, delta[half])));
        const float s1 = apply_scale(a, __fmul_rn(p1, __fsub_rn(d2.y, delta[half])));
        *reinterpret_cast<uint32_t*>(slot) = live ? pack(s0, s1) : 0u;
        *reinterpret_cast<uint32_t*>(Ps + row * LP + 8 * n + c) = live ? pack(p0, p1) : 0u;
      }
    }
  }
  __syncthreads();
  // B: dQ (rows), dK, dV (key blocks) as equal 16 x 32 tasks over all warps.
  constexpr int TASKS = 3 * NB * HALVES;
  for (int task = warp; task < TASKS; task += NW) {
    const int kind = task % 3, blk = (task / 3) / HALVES, d0 = 32 * ((task / 3) % HALVES);
    float acc[4][4];
    if (kind == 0) {
      tile_product<TP / 16, 4, false>(acc, Ss + 16 * blk * LP, LP, Ks + d0, LD);
      store_cols<4>(a.dq, acc, b, h, 16 * blk, t, d0);
      if (a.red[0].dst) column_sums_cols<4>(colsum + (0 * NW + blk) * HD + d0, acc, 16 * blk, t);
    } else if (kind == 1) {
      tile_product<TP / 16, 4, true>(acc, Ss + 16 * blk, LP, Qs + d0, LD);
      store_cols<4>(a.dk, acc, b, h, 16 * blk, t, d0);
      if (a.red[1].dst) column_sums_cols<4>(colsum + (1 * NW + blk) * HD + d0, acc, 16 * blk, t);
    } else {
      tile_product<TP / 16, 4, true>(acc, Ps + 16 * blk, LP, Ds + d0, LD);
      store_cols<4>(a.dv, acc, b, h, 16 * blk, t, d0);
      if (a.red[2].dst) column_sums_cols<4>(colsum + (2 * NW + blk) * HD + d0, acc, 16 * blk, t);
    }
  }
  __syncthreads();
  reduce_item<HD, NB>(a, b, h, colsum, nullptr);
}

template <int HD, int TP, bool PC = false>
PMIX_DEV void backward_compute(const Args& a, int bh, const __nv_bfloat16* Qs, const __nv_bfloat16* Ks,
                               const __nv_bfloat16* Vs, const __nv_bfloat16* Ds, const float* stat,
                               __nv_bfloat16* Ps, __nv_bfloat16* Ss, float* colsum, int wbase = 0, int bar = 6) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16;
  if constexpr (PMIX_BWD_SPLIT && NB < NW && !PC) {
    backward_compute_split<HD, TP>(a, bh, Qs, Ks, Vs, Ds, stat, Ps, Ss, colsum);
    return;
  }
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = (int)(threadIdx.x >> 5) - wbase, lane = threadIdx.x & 31, c = 2 * (lane & 3);
  if (warp < NB) {
    const int row0 = 16 * warp, r = row0 + (lane >> 2);
    float p[TP / 8][4];
    if (a.probs) {
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) {
        const float4 x = __ldcg(probs_at<TP>(a, bh, warp, n));
        p[n][0] = x.x; p[n][1] = x.y; p[n][2] = x.z; p[n][3] = x.w;
      }
    } else {
      row_products<HD, TP>(p, Qs, Ks, row0);
      float m[2] = {stat[2 * r], stat[2 * (r + 8)]}, l[2] = {stat[2 * r + 1], stat[2 * (r + 8) + 1]};
      softmax_rows<TP>(a, p, m, l, true);
    }
    float dp[TP / 8][4];
    row_products<HD, TP>(dp, Ds, Vs, row0);
    float delta[2];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      float sum = 0.f;
#pragma unroll
      for (int n = 0; n < TP / 8; ++n) {
        dp[n][2 * half] = round_bf16(dp[n][2 * half]);
        dp[n][2 * half + 1] = round_bf16(dp[n][2 * half + 1]);
        if (8 * n + c < t) sum = __fadd_rn(sum, __fmul_rn(dp[n][2 * half], p[n][2 * half]));
        if (8 * n + c + 1 < t) sum = __fadd_rn(sum, __fmul_rn(dp[n][2 * half + 1], p[n][2 * half + 1]));
      }
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
      sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
      delta[half] = sum;
    }
#pragma unroll
    for (int n = 0; n < TP / 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool live = (r + 8 * (e >> 1)) < t;   // padded query rows contribute nothing
        const float ds = apply_scale(a, __fmul_rn(p[n][e], __fsub_rn(dp[n][e], delta[e >> 1])));
        dp[n][e] = live ? round_bf16(ds) : 0.f;
        p[n][e] = live ? p[n][e] : 0.f;
      }
#pragma unroll
    for (int n = 0; n < TP / 8; ++n) {
      *reinterpret_cast<uint32_t*>(Ps + r * LP + 8 * n + c) = pack(p[n][0], p[n][1]);
      *reinterpret_cast<uint32_t*>(Ps + (r + 8) * LP + 8 * n + c) = pack(p[n][2], p[n][3]);
      *reinterpret_cast<uint32_t*>(Ss + r * LP + 8 * n + c) = pack(dp[n][0], dp[n][1]);
      *reinterpret_cast<uint32_t*>(Ss + (r + 8) * LP + 8 * n + c) = pack(dp[n][2], dp[n][3]);
    }
    // dQ rows = bf16(dS) k
    float dq[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < TP / 16; ++kk) {
      const uint32_t s4[4] = {pack(dp[2 * kk][0], dp[2 * kk][1]), pack(dp[2 * kk][2], dp[2 * kk][3]),
                              pack(dp[2 * kk + 1][0], dp[2 * kk + 1][1]), pack(dp[2 * kk + 1][2], dp[2 * kk + 1][3])};
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t bb[4];
        ldsm_x4_t(bb, Ks + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(dq[2 * n2], s4, bb[0], bb[1]);
        mma(dq[2 * n2 + 1], s4, bb[2], bb[3]);
      }
    }
    store_rows<HD>(a.dq, dq, b, h, row0, t);
    if (a.red[0].dst) column_sums<HD>(colsum + (0 * NW + warp) * HD, dq, row0, t);
  }
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 4, 0);
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 5, 4);
  csync<PC, NB>(bar);
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 6, 0);
  if (warp < NB) {
    // Key block `warp`: dV = bf16(P)^T dO and dK = bf16(dS)^T q over all query rows.
    const int key0 = 16 * warp;
    float dv[HD / 8][4], dk[HD / 8][4];
#pragma unroll
    for (int n = 0; n < HD / 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dv[n][e] = dk[n][e] = 0.f;
    const int mi = lane >> 3, kr = (lane & 7) + 8 * (mi >> 1), kc = 8 * (mi & 1);
#pragma unroll
    for (int kk = 0; kk < TP / 16; ++kk) {
      uint32_t pa[4], sa[4];
      ldsm_x4_t(pa, Ps + (16 * kk + kr) * LP + key0 + kc);
      ldsm_x4_t(sa, Ss + (16 * kk + kr) * LP + key0 + kc);
      // ldmatrix.trans of a [k][m] block returns the A fragment order (m0k0, m8k0, m0k8, m8k8) for this lane map
      const uint32_t pA[4] = {pa[0], pa[1], pa[2], pa[3]};
      const uint32_t sA[4] = {sa[0], sa[1], sa[2], sa[3]};
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t db[4], qb[4];
        ldsm_x4_t(db, Ds + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        ldsm_x4_t(qb, Qs + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(dv[2 * n2], pA, db[0], db[1]);
        mma(dv[2 * n2 + 1], pA, db[2], db[3]);
        mma(dk[2 * n2], sA, qb[0], qb[1]);
        mma(dk[2 * n2 + 1], sA, qb[2], qb[3]);
      }
    }
    store_rows<HD>(a.dv, dv, b, h, key0, t);
    store_rows<HD>(a.dk, dk, b, h, key0, t);
    if (a.red[1].dst) column_sums<HD>(colsum + (1 * NW + warp) * HD, dk, key0, t);
    if (a.red[2].dst) column_sums<HD>(colsum + (2 * NW + warp) * HD, dv, key0, t);
  }
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 7, 0);
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 11, 4);
  csync<PC, NB>(bar);
  reduce_item<HD, NB>(a, b, h, colsum, nullptr, PC ? 32 * NB : NT, 32 * wbase);
  if (PC) csync<PC, NB>(bar);  // P/dS staging and column sums are reused by the next item
  PMIX_STAMP(a, 1, (bh - (int)blockIdx.x) / (int)gridDim.x, 8, 0);
}

template <int HD, int TP>
PMIX_DEV void backward_item(const Args& a, int bh, float* smem) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16;
  static_assert(NB <= NW && TP % 16 == 0 && (HD == 32 || HD == 64), "unsupported mixed attention shape");
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks = Qs + TP * LD;
  __nv_bfloat16* Vs = Ks + TP * LD;
  __nv_bfloat16* Ds = Vs + TP * LD;
  __nv_bfloat16* Ps = Ds + TP * LD;   // bf16(P) [query row][key]
  __nv_bfloat16* Ss = Ps + TP * LP;   // bf16(dS * c) [query row][key]
  float* stat = reinterpret_cast<float*>(Ss + TP * LP);  // [TP][2]
  float* colsum = stat + 2 * TP;                          // [3][NW][HD] + flag
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  load_operand<HD, TP>(Qs, a.q, b, h, t);
  load_operand<HD, TP>(Ks, a.k, b, h, t);
  load_operand<HD, TP>(Vs, a.v, b, h, t);
  load_operand<HD, TP>(Ds, a.dout, b, h, t);
  for (int r = (int)threadIdx.x; r < TP; r += NT) {
    const float2 x = r < t ? reinterpret_cast<const float2*>(a.stats)[(long long)bh * t + r] : make_float2(0.f, 1.f);
    stat[2 * r] = x.x;
    stat[2 * r + 1] = x.y;
  }
  __syncthreads();
  backward_compute<HD, TP>(a, bh, Qs, Ks, Vs, Ds, stat, Ps, Ss, colsum);
}

// ------------------------------------------------------------------------------------------- pipelined phases
// PMIX_PIPELINE=1: the next item's BF16 operands (and bias vectors, statistics) are copied with cp.async into
// the alternate shared buffer while the current item computes; biased operands are completed in shared memory
// (x = bf16(fl(widen(x) + b))) after the copy lands. FP32-source operands (kind 2) use the register path.
#ifndef PMIX_PIPELINE
#define PMIX_PIPELINE 1
#endif

// Issue copies of one operand (kinds 0/1) for one (batch, head); kind 2 is converted synchronously.
template <int HD, int TP>
PMIX_DEV void stage_operand(__nv_bfloat16* dst, float* bias_dst, const Operand& op, int b, int h, int t) {
  constexpr int LD = HD + 8, CH = HD / 8, TOTAL = TP * CH, ITER = (TOTAL + NT - 1) / NT;
  if (op.kind == 2) {
    load_operand<HD, TP>(dst, op, b, h, t);
    return;
  }
  const __nv_bfloat16* src = static_cast<const __nv_bfloat16*>(op.src) + (long long)b * op.sb + (long long)h * op.sh;
#pragma unroll
  for (int k = 0; k < ITER; ++k) {
    const int idx = (int)threadIdx.x + k * NT;
    if (TOTAL % NT == 0 || idx < TOTAL) {
      const int r = idx / CH, c = 8 * (idx % CH);
      if (r < t) cp16(dst + r * LD + c, src + (long long)r * op.sr + c);
      else *reinterpret_cast<uint4*>(dst + r * LD + c) = make_uint4(0u, 0u, 0u, 0u);
    }
  }
  if (op.kind == 1 && (int)threadIdx.x < HD / 4)
    cp16(bias_dst + 4 * threadIdx.x, op.bias + (long long)b * op.bb + (long long)h * op.bh + 4 * threadIdx.x);
}

// Complete a staged biased operand in shared memory: rows < t become bf16(fl(widen(x) + b)).
template <int HD, int TP>
PMIX_DEV void finish_operand(__nv_bfloat16* dst, const float* bias, const Operand& op, int t) {
  constexpr int LD = HD + 8, CH = HD / 8;
  if (op.kind != 1) return;
  for (int idx = (int)threadIdx.x; idx < t * CH; idx += NT) {
    const int r = idx / CH, c = 8 * (idx % CH);
    uint4* p = reinterpret_cast<uint4*>(dst + r * LD + c);
    const uint4 raw = *p;
    const __nv_bfloat162* x = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float2 x0 = __bfloat1622float2(x[0]), x1 = __bfloat1622float2(x[1]);
    const float2 x2 = __bfloat1622float2(x[2]), x3 = __bfloat1622float2(x[3]);
    *p = make_uint4(pack(__fadd_rn(x0.x, bias[c]), __fadd_rn(x0.y, bias[c + 1])),
                    pack(__fadd_rn(x1.x, bias[c + 2]), __fadd_rn(x1.y, bias[c + 3])),
                    pack(__fadd_rn(x2.x, bias[c + 4]), __fadd_rn(x2.y, bias[c + 5])),
                    pack(__fadd_rn(x3.x, bias[c + 6]), __fadd_rn(x3.y, bias[c + 7])));
  }
}

template <int HD, int TP>
__host__ __device__ constexpr int forward_buffer_elems() { return 3 * TP * (HD + 8) + 2 * 3 * HD; }  // BF16 units

template <int HD, int TP>
PMIX_DEV void forward_phase_pc(const Args& a, long long items, float* smem);
template <int HD, int TP>
PMIX_DEV void forward_phase_dual(const Args& a, long long items, float* smem);
template <int HD, int TP>
PMIX_DEV void backward_phase_dual(const Args& a, long long items, float* smem);
template <int HD, int TP>
PMIX_DEV void forward_phase_split(const Args& a, long long items, float* smem);

template <int HD, int TP>
PMIX_DEV void forward_phase(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8;
  zero_counters(a);
  if constexpr (PMIX_DUAL && TP == 64) {
    forward_phase_dual<HD, TP>(a, items, smem);
    return;
  } else if constexpr (PMIX_PRODUCER && PMIX_FWD_SPLIT && TP / 16 < NW) {
    forward_phase_split<HD, TP>(a, items, smem);
    return;
  } else if constexpr (PMIX_PRODUCER && TP / 16 < NW) {
    forward_phase_pc<HD, TP>(a, items, smem);
    return;
  }
#if PMIX_PIPELINE
  constexpr int BUF = forward_buffer_elems<HD, TP>();
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem);
  const int t = a.t;
  auto issue = [&](long long item, int which) {
    __nv_bfloat16* Q = base + which * BUF;
    float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    stage_operand<HD, TP>(Q, bias, a.q, b, h, t);
    stage_operand<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t);
    stage_operand<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t);
    cp_commit();
  };
  long long item = blockIdx.x;
  if (item < items) issue(item, 0);
  for (int j = 0; item < items; ++j, item += gridDim.x) {
    __nv_bfloat16* Q = base + (j & 1) * BUF;
    float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
    PMIX_STAMP(a, 0, j, 0, 0);
    cp_wait_all();
    __syncthreads();
    PMIX_STAMP(a, 0, j, 1, 0);
    finish_operand<HD, TP>(Q, bias, a.q, t);
    finish_operand<HD, TP>(Q + TP * LD, bias + HD, a.k, t);
    finish_operand<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t);
    __syncthreads();
    PMIX_STAMP(a, 0, j, 2, 0);
    if (item + gridDim.x < items) issue(item + gridDim.x, (j & 1) ^ 1);
    PMIX_STAMP(a, 0, j, 3, 0);
    forward_compute<HD, TP>(a, (int)item, Q, Q + TP * LD, Q + 2 * TP * LD);
    PMIX_STAMP(a, 0, j, 4, 0);
    PMIX_STAMP(a, 0, j, 5, 4);
  }
#else
  for (long long item = blockIdx.x; item < items; item += gridDim.x) {
    if (item != blockIdx.x) __syncthreads();
    forward_item<HD, TP>(a, (int)item, smem);
  }
#endif
}

template <int HD, int TP>
__host__ __device__ constexpr int backward_buffer_elems() { return 4 * TP * (HD + 8) + 2 * 3 * HD + 4 * TP; }

template <int HD, int TP>
PMIX_DEV void backward_phase_pc(const Args& a, long long items, float* smem);

// P6-B: PIPE=false selects the single-buffer item schedule (less shared memory, e.g. for a two-CTA program).
template <int HD, int TP, bool PIPE = true>
PMIX_DEV void backward_phase(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, LP = TP + 8;
  if constexpr (PIPE && PMIX_DUAL && TP == 64 && HD == 32) {
    backward_phase_dual<HD, TP>(a, items, smem);
    return;
  }
  if constexpr (PIPE && PMIX_PRODUCER && TP / 16 < NW) {
    backward_phase_pc<HD, TP>(a, items, smem);
    return;
  }
  if constexpr (PIPE && PMIX_PIPELINE) {
  constexpr int BUF = backward_buffer_elems<HD, TP>();
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ps = base + 2 * BUF;
  __nv_bfloat16* Ss = Ps + TP * LP;
  float* colsum = reinterpret_cast<float*>(Ss + TP * LP);
  const int t = a.t;
  auto issue = [&](long long item, int which) {
    __nv_bfloat16* Q = base + which * BUF;
    float* bias = reinterpret_cast<float*>(Q + 4 * TP * LD);
    float* stat = bias + 3 * HD;
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    stage_operand<HD, TP>(Q, bias, a.q, b, h, t);
    stage_operand<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t);
    stage_operand<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t);
    stage_operand<HD, TP>(Q + 3 * TP * LD, nullptr, a.dout, b, h, t);
    for (int r = (int)threadIdx.x; r < TP; r += NT) {
      if (r < t) cp8(stat + 2 * r, a.stats + 2 * ((long long)item * t + r));
      else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
    }
    cp_commit();
  };
  long long item = blockIdx.x;
  if (item < items) issue(item, 0);
  int a_trace_item = 0;
  for (int j = 0; item < items; ++j, item += gridDim.x) {
    __nv_bfloat16* Q = base + (j & 1) * BUF;
    float* bias = reinterpret_cast<float*>(Q + 4 * TP * LD);
    float* stat = bias + 3 * HD;
    PMIX_STAMP(a, 1, j, 0, 0);
    cp_wait_all();
    __syncthreads();
    PMIX_STAMP(a, 1, j, 1, 0);
    finish_operand<HD, TP>(Q, bias, a.q, t);
    finish_operand<HD, TP>(Q + TP * LD, bias + HD, a.k, t);
    finish_operand<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t);
    __syncthreads();
    PMIX_STAMP(a, 1, j, 2, 0);
    if (item + gridDim.x < items) issue(item + gridDim.x, (j & 1) ^ 1);
    PMIX_STAMP(a, 1, j, 3, 0);
    a_trace_item = j;
    backward_compute<HD, TP>(a, (int)item, Q, Q + TP * LD, Q + 2 * TP * LD, Q + 3 * TP * LD, stat, Ps, Ss, colsum);
    PMIX_STAMP(a, 1, j, 9, 0);
    __syncthreads();  // P/dS staging is reused by the next item
    PMIX_STAMP(a, 1, j, 10, 0);
  }
  finish_reductions<HD>(a, items, reinterpret_cast<int*>(colsum + 3 * NW * HD));
  } else {
  for (long long item = blockIdx.x; item < items; item += gridDim.x) {
    if (item != blockIdx.x) __syncthreads();
    backward_item<HD, TP>(a, (int)item, smem);
  }
  __syncthreads();
  finish_reductions<HD>(a, items, reinterpret_cast<int*>(smem));
  }
}


// ------------------------------------------------------------------- forward with value products on producers
// PMIX_FWD_SPLIT=1 (producer/consumer mode): row warps compute scores and the softmax, save statistics (and P
// when requested) and write bf16(P) to a double-buffered staging area; the producer warps, besides staging
// operands, compute O = bf16(bf16(P) v) for the row blocks (block p, p + NP, ...). Same products and k order as
// forward_compute. Named barriers 7, 8 (P full) and 9, 10 (P empty).

template <int HD, int TP>
__host__ __device__ constexpr int forward_split_shared_bytes() {
  return 2 * (2 * forward_buffer_elems<HD, TP>() + 2 * TP * (TP + 8));
}

template <int HD, int TP>
PMIX_DEV void forward_phase_split(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16, BUF = forward_buffer_elems<HD, TP>(), CT = 32 * NB;
  constexpr int PT = NT - CT, NP = NW - NB;
  static_assert(NB < NW, "producer warps required");
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Pbase = base + 2 * BUF;   // [2][TP][LP] bf16(P)
  const int t = a.t, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;
  if (warp >= NB) {
    const int tid = threadIdx.x - CT, pw = warp - NB;
    auto stage = [&](int j) {
      const int i = j & 1;
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads);
      __nv_bfloat16* Q = base + i * BUF;
      float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
      stage_operand_sub<HD, TP>(Q, bias, a.q, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t, tid, PT);
      cp_commit();
      cp_wait_all();
      nbar_sync(5, PT);
      finish_operand_sub<HD, TP>(Q, bias, a.q, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t, tid, PT);
      nbar_arrive(1 + i, NT);
    };
    if (n > 0) stage(0);
    if (n > 1) stage(1);
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads);
      const __nv_bfloat16* Vs = base + i * BUF + 2 * TP * LD;
      const __nv_bfloat16* Pb = Pbase + i * TP * LP;
      nbar_sync(7 + i, NT);   // bf16(P) of item j staged
      for (int blk = pw; blk < NB; blk += NP) {
        float o[HD / 8][4];
        tile_product<TP / 16, HD / 8, false>(o, Pb + 16 * blk * LP, LP, Vs, LD);
        store_cols<HD / 8>(a.o, o, b, h, 16 * blk, t, 0);
      }
      if (j + 2 < n) {
        nbar_arrive(9 + i, NT);   // P staging i free for item j + 2
        nbar_sync(3 + i, NT);     // row warps released Q/K of buffer i (V is read above, in program order)
        stage(j + 2);
      }
    }
  } else {
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      PMIX_STAMP(a, 0, j, 0, 0);
      nbar_sync(1 + i, NT);
      PMIX_STAMP(a, 0, j, 1, 0);
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int bh = (int)item;
      const __nv_bfloat16* Q = base + i * BUF;
      __nv_bfloat16* Pb = Pbase + i * TP * LP;
      const int row0 = 16 * warp, c = 2 * (lane & 3);
      float s[TP / 8][4];
      row_products<HD, TP>(s, Q, Q + TP * LD, row0);
      float m[2], l[2];
      softmax_rows<TP>(a, s, m, l, false);
      if (a.probs) {
#pragma unroll
        for (int k = 0; k < TP / 8; ++k) *probs_at<TP>(a, bh, warp, k) = make_float4(s[k][0], s[k][1], s[k][2], s[k][3]);
      }
      if (j >= 2) nbar_sync(9 + i, NT);   // value products of item j - 2 finished reading this P staging
      const int r = row0 + (lane >> 2);
#pragma unroll
      for (int k = 0; k < TP / 8; ++k) {
        *reinterpret_cast<uint32_t*>(Pb + r * LP + 8 * k + c) = pack(s[k][0], s[k][1]);
        *reinterpret_cast<uint32_t*>(Pb + (r + 8) * LP + 8 * k + c) = pack(s[k][2], s[k][3]);
      }
      nbar_arrive(7 + i, NT);
      if (j + 2 < n) nbar_arrive(3 + i, NT);
      if ((lane & 3) == 0) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int row = row0 + (lane >> 2) + 8 * half;
          if (row < t) {
            float2* stat = reinterpret_cast<float2*>(a.stats) + (long long)bh * t + row;
            *stat = make_float2(m[half], l[half]);
          }
        }
      }
      PMIX_STAMP(a, 0, j, 4, 0);
    }
  }
}

// ------------------------------------------------------------------------------- producer/consumer phases
template <int HD, int TP>
PMIX_DEV void forward_phase_pc(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16, BUF = forward_buffer_elems<HD, TP>(), CT = 32 * NB, PT = NT - CT;
  static_assert(NB < NW, "producer warps required");
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem);
  const int t = a.t, warp = threadIdx.x >> 5;
  const int n = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;
  if (warp >= NB) {
    const int tid = threadIdx.x - CT;
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      if (j >= 2) nbar_sync(3 + i, NT);
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads);
      __nv_bfloat16* Q = base + i * BUF;
      float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
      stage_operand_sub<HD, TP>(Q, bias, a.q, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t, tid, PT);
      cp_commit();
      cp_wait_all();
      nbar_sync(5, PT);
      finish_operand_sub<HD, TP>(Q, bias, a.q, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t, tid, PT);
      nbar_arrive(1 + i, NT);
    }
  } else {
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      PMIX_STAMP(a, 0, j, 0, 0);
      nbar_sync(1 + i, NT);
      PMIX_STAMP(a, 0, j, 1, 0);
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      __nv_bfloat16* Q = base + i * BUF;
      forward_compute<HD, TP>(a, (int)item, Q, Q + TP * LD, Q + 2 * TP * LD);
      PMIX_STAMP(a, 0, j, 4, 0);
      if (j + 2 < n) nbar_arrive(3 + i, NT);
    }
  }
}

template <int HD, int TP>
PMIX_DEV void backward_phase_pc(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16, BUF = backward_buffer_elems<HD, TP>(), CT = 32 * NB;
  constexpr int PT = NT - CT;
  static_assert(NB < NW, "producer warps required");
  __nv_bfloat16* base = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ps = base + 2 * BUF;
  __nv_bfloat16* Ss = Ps + TP * LP;
  float* colsum = reinterpret_cast<float*>(Ss + TP * LP);
  const int t = a.t, warp = threadIdx.x >> 5;
  const int n = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;
  if (warp >= NB) {
    const int tid = threadIdx.x - CT;
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      if (j >= 2) nbar_sync(3 + i, NT);
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads);
      __nv_bfloat16* Q = base + i * BUF;
      float* bias = reinterpret_cast<float*>(Q + 4 * TP * LD);
      float* stat = bias + 3 * HD;
      stage_operand_sub<HD, TP>(Q, bias, a.q, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t, tid, PT);
      stage_operand_sub<HD, TP>(Q + 3 * TP * LD, nullptr, a.dout, b, h, t, tid, PT);
      for (int r = tid; r < TP; r += PT) {
        if (r < t) cp8(stat + 2 * r, a.stats + 2 * ((long long)item * t + r));
        else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
      }
      cp_commit();
      cp_wait_all();
      nbar_sync(5, PT);
      finish_operand_sub<HD, TP>(Q, bias, a.q, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, t, tid, PT);
      finish_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t, tid, PT);
      nbar_arrive(1 + i, NT);
    }
  } else {
    for (int j = 0; j < n; ++j) {
      const int i = j & 1;
      PMIX_STAMP(a, 1, j, 0, 0);
      nbar_sync(1 + i, NT);
      PMIX_STAMP(a, 1, j, 1, 0);
      const long long item = blockIdx.x + (long long)j * gridDim.x;
      __nv_bfloat16* Q = base + i * BUF;
      float* stat = reinterpret_cast<float*>(Q + 4 * TP * LD) + 3 * HD;
      backward_compute<HD, TP, true>(a, (int)item, Q, Q + TP * LD, Q + 2 * TP * LD, Q + 3 * TP * LD, stat, Ps, Ss,
                                     colsum);
      PMIX_STAMP(a, 1, j, 9, 0);
      if (j + 2 < n) nbar_arrive(3 + i, NT);
    }
  }
  __syncthreads();
  finish_reductions<HD>(a, items, reinterpret_cast<int*>(colsum + 3 * NW * HD));
}

// ================================================================================ dual items (P8-E, TP == 64)
// PMIX_DUAL=1: with four 16-row blocks per (batch, head), the CTA runs two items at once, one per half (warps 0-3
// and 4-7), instead of four consumer warps fed by four producer warps. Each half stages its own operands (cp.async
// and bias completion by its 128 threads; forward double-buffered when it fits) and runs forward_compute /
// backward_compute with its own warp base, shared tiles, column sums and named barriers (8 + half, 10 + half).
// Items keep the round-robin CTA assignment (half h takes the CTA's items j = h, h + 2, ...); every item's
// arithmetic and summation order are those of the single-item schedules (bitwise identical results).

template <int HD, int TP>
__host__ __device__ constexpr int dual_forward_nbuf() { return HD == 32 ? 2 : 1; }
template <int HD, int TP>
__host__ __device__ constexpr int dual_forward_bytes() {
  return 2 * dual_forward_nbuf<HD, TP>() * (3 * TP * (HD + 8) + 6 * HD) * 2;
}
template <int HD, int TP>
__host__ __device__ constexpr int dual_backward_half_elems() {   // BF16 units
  return 4 * TP * (HD + 8) + 6 * HD + 4 * TP + 2 * TP * (TP + 8) + 2 * (3 * NW * HD + 72);
}
template <int HD, int TP>
__host__ __device__ constexpr int dual_backward_bytes() { return 2 * dual_backward_half_elems<HD, TP>() * 2; }

template <int HD, int TP>
PMIX_DEV void forward_phase_dual(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, ELEMS = 3 * TP * LD + 6 * HD, NBUF = dual_forward_nbuf<HD, TP>();
  static_assert(TP == 64, "dual items need exactly four row blocks");
  const int warp = threadIdx.x >> 5, half = warp >> 2, tid = threadIdx.x & 127, t = a.t;
  __nv_bfloat16* hb = reinterpret_cast<__nv_bfloat16*>(smem) + half * NBUF * ELEMS;
  const int n = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;
  auto stage = [&](int j, __nv_bfloat16* Q) {
    const long long item = blockIdx.x + (long long)j * gridDim.x;
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
    stage_operand_sub<HD, TP>(Q, bias, a.q, b, h, t, tid, 128);
    stage_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t, tid, 128);
    stage_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t, tid, 128);
    cp_commit();
  };
  if (half < n) stage(half, hb);
  int k = 0;
  for (int j = half; j < n; j += 2, ++k) {
    __nv_bfloat16* Q = hb + (NBUF == 2 ? (k & 1) * ELEMS : 0);
    float* bias = reinterpret_cast<float*>(Q + 3 * TP * LD);
    cp_wait_all();
    nbar_sync(8 + half, 128);   // data landed for all threads; the other buffer's readers (item j - 2) are done
    finish_operand_sub<HD, TP>(Q, bias, a.q, t, tid, 128);
    finish_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, t, tid, 128);
    finish_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t, tid, 128);
    nbar_sync(8 + half, 128);
    if (NBUF == 2 && j + 2 < n) stage(j + 2, hb + ((k + 1) & 1) * ELEMS);
    forward_compute<HD, TP>(a, (int)(blockIdx.x + (long long)j * gridDim.x), Q, Q + TP * LD, Q + 2 * TP * LD,
                            4 * half);
    if (NBUF == 1 && j + 2 < n) {
      nbar_sync(8 + half, 128);
      stage(j + 2, hb);
    }
  }
}

template <int HD, int TP>
PMIX_DEV void backward_phase_dual(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, LP = TP + 8, NB = TP / 16, HALF = dual_backward_half_elems<HD, TP>();
  constexpr int OPS = 4 * TP * LD + 6 * HD + 4 * TP;
  static_assert(NB == 4, "dual items need exactly four row blocks");
  const int warp = threadIdx.x >> 5, half = warp >> 2, tid = threadIdx.x & 127, t = a.t;
  __nv_bfloat16* Q = reinterpret_cast<__nv_bfloat16*>(smem) + half * HALF;
  float* bias = reinterpret_cast<float*>(Q + 4 * TP * LD);
  float* stat = bias + 3 * HD;
  __nv_bfloat16* Ps = Q + OPS;
  __nv_bfloat16* Ss = Ps + TP * LP;
  float* colsum = reinterpret_cast<float*>(Ss + TP * LP);
  const int n = items > blockIdx.x ? (int)((items - 1 - blockIdx.x) / gridDim.x) + 1 : 0;
  for (int j = half; j < n; j += 2) {
    const long long item = blockIdx.x + (long long)j * gridDim.x;
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    stage_operand_sub<HD, TP>(Q, bias, a.q, b, h, t, tid, 128);
    stage_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, b, h, t, tid, 128);
    stage_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, b, h, t, tid, 128);
    stage_operand_sub<HD, TP>(Q + 3 * TP * LD, nullptr, a.dout, b, h, t, tid, 128);
    for (int r = tid; r < TP; r += 128) {
      if (r < t) cp8(stat + 2 * r, a.stats + 2 * (item * t + r));
      else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
    }
    cp_commit();
    if (PMIX_DUAL_PREFETCH && j + 2 < n) {   // next item of this half: operand rows into L2
      const long long nx = blockIdx.x + (long long)(j + 2) * gridDim.x;
      const int nb = (int)(nx / a.heads), nh = (int)(nx % a.heads);
      for (int idx = tid; idx < 4 * t; idx += 128) {
        const int which = idx / t, r = idx % t;
        const Operand& op = which == 0 ? a.q : which == 1 ? a.k : which == 2 ? a.v : a.dout;
        const long long at = (long long)nb * op.sb + (long long)nh * op.sh + (long long)r * op.sr;
        const void* row = op.kind == 2 ? static_cast<const void*>(static_cast<const float*>(op.src) + at)
                                       : static_cast<const void*>(static_cast<const __nv_bfloat16*>(op.src) + at);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(row));
      }
    }
    cp_wait_all();
    nbar_sync(8 + half, 128);
    finish_operand_sub<HD, TP>(Q, bias, a.q, t, tid, 128);
    finish_operand_sub<HD, TP>(Q + TP * LD, bias + HD, a.k, t, tid, 128);
    finish_operand_sub<HD, TP>(Q + 2 * TP * LD, bias + 2 * HD, a.v, t, tid, 128);
    nbar_sync(8 + half, 128);
    // ends with a sync of this half on barrier 10 + half: operands, P/dS staging and column sums are free after it
    backward_compute<HD, TP, true>(a, (int)item, Q, Q + TP * LD, Q + 2 * TP * LD, Q + 3 * TP * LD, stat, Ps, Ss,
                                   colsum, 4 * half, 10 + half);
  }
  __syncthreads();
  float* colsum0 = reinterpret_cast<float*>(reinterpret_cast<__nv_bfloat16*>(smem) + OPS + 2 * TP * LP);
  finish_reductions<HD>(a, items, reinterpret_cast<int*>(colsum0 + 3 * NW * HD));
}

// P9-E: long items v2 (see "long context v2" below): register scores and staged operands, bitwise identical.
#ifndef PMIX_LONG_V2
#define PMIX_LONG_V2 1
#endif
// P9-E: unroll factors of the 16-key step loops of the long rows (instruction scheduling only; every accumulation
// keeps its order).
#ifndef PMIX_LONG_UNROLL
#define PMIX_LONG_UNROLL 0   // 0: no pragma (the compiler's choice, as before)
#endif
#define PMIX_STR2(x) #x
#define PMIX_STR(x) PMIX_STR2(x)
#if PMIX_LONG_UNROLL > 0
#define PMIX_UNROLL_STEPS _Pragma(PMIX_STR(unroll PMIX_LONG_UNROLL))
#else
#define PMIX_UNROLL_STEPS
#endif
// P9-E declared precision option (amendment A5): forward row statistics (m, l) in one pass with running rescaling.
#ifndef PMIX_ONLINE_STATS
#define PMIX_ONLINE_STATS 0
#endif
// P9-E declared precision option (amendment A4): the long backward's delta = fl(sum_d dO[d] * O[d]) over the head
// width (BF16 dO and O widened exactly, FP32 products and sums in k16-chunk order, then the row's four lanes) -- the
// FlashAttention D -- instead of sum_j P_j bf16(dP_j) over the keys, which removes the row stage's first pass.
#ifndef PMIX_FA_DELTA
#define PMIX_FA_DELTA 0
#endif
// P9-E: half-item claimed backward units (see "long backward v3"); needs the planner's halves scratch.
#ifndef PMIX_LONG_V3
#define PMIX_LONG_V3 1
#endif
// P9-E: v3 units are half items (1) or whole items (0: each warp takes blocks w and w + NW, the v1/v2 per-warp
// accumulation, no cross-unit bias partials); both are claimed in order.
#ifndef PMIX_KEYS_RP
#define PMIX_KEYS_RP 0     // P9-E: key units recompute P^T (row stage stores only dS^T); v3 whole-item units
#endif
#ifndef PMIX_KEYS_CHUNK
#define PMIX_KEYS_CHUNK 0  // P9-E: >0 loads the key stage's stored fragments in double-buffered chunks (bitwise identical)
#endif
#ifndef PMIX_ROWS_W
#define PMIX_ROWS_W 0      // P9-E: v2 row stage issues two 16-key steps' products per iteration (bitwise identical)
#endif
#ifndef PMIX_V3_DIRECT
#define PMIX_V3_DIRECT 0   // P9-E: v3 backward stages operands in one direct pass (stage_operand_direct)
#endif
#ifndef PMIX_DIAG_SKIP_ROWS
#define PMIX_DIAG_SKIP_ROWS 0   // diagnostic timing builds only: v3 whole-item row units do no work
#endif
#ifndef PMIX_DIAG_SKIP_KEYS
#define PMIX_DIAG_SKIP_KEYS 0   // diagnostic timing builds only: v3 whole-item key units do no work
#endif
#ifndef PMIX_LONG_V3_HALVES
#define PMIX_LONG_V3_HALVES 1
#endif
// P9-E: mixed v3 units -- whole-item units for the first gridDim.x items, half-item units for the items beyond one
// wave (their row stages start a wave late, so halving them shortens the row -> key dependency tail). Same unit
// bodies and orders as the whole/half modes: bitwise identical results.
#ifndef PMIX_LONG_V3_MIXED
#define PMIX_LONG_V3_MIXED 0
#endif
// P9-E: the v2 forward phase (staged operands) is measured slower than the v1 forward items (gpu-04a): default off.
#ifndef PMIX_LONG_FWD_V2
#define PMIX_LONG_FWD_V2 0
#endif
// P9-E: forward rows of v2 keep the BF16 scores in registers (1) or recompute them as the v1 rows (0).
#ifndef PMIX_LONG_FWD_REG
#define PMIX_LONG_FWD_REG 1
#endif
// P9-E: the v2 phase bodies as called (noinline) functions, so their register allocation does not raise the
// pressure of the rest of the program (mma.sync bodies only; no warpgroup MMA crosses the call).
#ifndef PMIX_LONG_NOINLINE
#define PMIX_LONG_NOINLINE 0
#endif
#if PMIX_LONG_NOINLINE
#define PMIX_LONG_ENTRY __device__ __noinline__
#else
#define PMIX_LONG_ENTRY PMIX_DEV
#endif

// ============================================================================== long context (P8-E, 128 < T <= 256)
// The whole (batch, head) is still one work item and the declared arithmetic above is unchanged, but no T x T tile
// is held on chip (TP = 16 * ceil(T / 16) <= 256). Keys are processed in 16-key steps against shared-memory-resident
// key/value rows (forward, backward stage 1) or query/output-gradient rows (backward stage 2); the operands of a
// warp's own 16-row block are register fragments loaded straight from their sources with load_operand's conversion
// and bias rounding. Row block `blk` belongs to warp blk % 8.
//   forward: pass 1 row max (exact), pass 2 l = sum_j exp(x_j - m) in softmax_rows' per-thread key order (so m and l
//            are bitwise those of the whole-row computation), pass 3 P = fl(exp(x - m) / l) -> bf16 -> O += P v.
//   backward stage 1 (row blocks, K/V resident): delta = sum_j P_j dP_j in backward_compute's key order, then
//            dQ = bf16(dS) k over keys in increasing 16-key steps.
//   backward stage 2 (key blocks, Q/dO resident): S^T = k q^T and dP^T = v dO^T (the same BF16 products and head
//            order per element), P^T and dS^T elementwise from the saved (m, l) and delta, then dV = bf16(P)^T dO
//            and dK = bf16(dS)^T q with the accumulator fragments reused as A operands (queries in increasing
//            16-row steps, the summation order of backward_compute).
// Scores and probability gradients are recomputed instead of stored: forward 3 score products; backward 4 score and
// 3 probability-gradient products. Absorbed bias-gradient column sums accumulate a warp's blocks in block order,
// then reduce_item sums the 8 warps in order (fixed and deterministic; another association than T <= 128).

template <int HD>
PMIX_DEV void load_afrag(uint32_t (&f)[HD / 16][4], const Operand& op, int b, int h, int row0, int t) {
  const int lane = threadIdx.x & 31, g = lane >> 2, c = 2 * (lane & 3);
  const long long base = (long long)b * op.sb + (long long)h * op.sh;
  const long long bias_base = (long long)b * op.bb + (long long)h * op.bh;
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int row = row0 + g + 8 * (j & 1), col = 16 * ks + c + 8 * (j >> 1);
      uint32_t x = 0u;
      if (row < t) {
        const long long at = base + (long long)row * op.sr + col;
        if (op.kind == 2) {
          const float2 v = __ldg(reinterpret_cast<const float2*>(static_cast<const float*>(op.src) + at));
          x = pack(v.x, v.y);
        } else {
          x = __ldg(reinterpret_cast<const unsigned int*>(static_cast<const __nv_bfloat16*>(op.src) + at));
          if (op.kind == 1) {
            const float2 bv = __ldg(reinterpret_cast<const float2*>(op.bias + bias_base + col));
            const float2 v = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&x));
            x = pack(__fadd_rn(v.x, bv.x), __fadd_rn(v.y, bv.y));
          }
        }
      }
      f[ks][j] = x;
    }
}

// One warp's 16 fragment rows against 16 resident rows [key0, key0 + 16) of a [TP][HD + 8] operand (row_products'
// products for n-tiles 2 * (key0 / 16) and the next, in the same head order).
template <int HD>
PMIX_DEV void step_products(float (&s)[2][4], const uint32_t (&f)[HD / 16][4], const __nv_bfloat16* B, int key0) {
  constexpr int LD = HD + 8;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int j = 0; j < 2; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks) {
    uint32_t bb[4];
    ldsm_x4(bb, B + (key0 + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * ks + 8 * ((lane >> 3) & 1));
    mma(s[0], f[ks], bb[0], bb[1]);
    mma(s[1], f[ks], bb[2], bb[3]);
  }
}

// Masked, rounded and scaled scores x of one 16-key step (keys >= t: -inf), as softmax_rows.
PMIX_DEV void step_scores(const Args& a, float (&s)[2][4], int key0) {
  const int c = 2 * (threadIdx.x & 3);
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = key0 + 8 * j + c + (e & 1);
      s[j][e] = col < a.t ? apply_scale(a, round_bf16(s[j][e])) : -INFINITY;
    }
}

// FP32 probabilities of one 16-key step from the saved row statistics (softmax_rows with given = true).
template <int HD>
PMIX_DEV void step_probs(const Args& a, float (&p)[2][4], const uint32_t (&qf)[HD / 16][4], const __nv_bfloat16* Ks,
                         int key0, const float (&m)[2], const float (&l)[2]) {
  step_products<HD>(p, qf, Ks, key0);
  step_scores(a, p, key0);
  const float li[2] = {pmix_rinv(l[0]), pmix_rinv(l[1])};
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) p[j][e] = pmix_div(pmix_exp(__fsub_rn(p[j][e], m[e >> 1])), l[e >> 1], li[e >> 1]);
}

// Bias-gradient column sums of one warp's 16 rows, accumulated over the warp's blocks in block order.
template <int HD>
PMIX_DEV void column_sums_acc(float* colsum, const float (&acc)[HD / 8][4], int row0, int t, bool first) {
  if (PMIX_DIAG_NO_COLSUM) return;
  const int lane = threadIdx.x & 31, r = row0 + (lane >> 2);
#pragma unroll
  for (int nt = 0; nt < HD / 8; ++nt) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float x = __fadd_rn(r < t ? round_bf16(acc[nt][e]) : 0.f, r + 8 < t ? round_bf16(acc[nt][2 + e]) : 0.f);
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 4));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 8));
      x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, 16));
      float* slot = colsum + 8 * nt + 2 * lane + e;
      if (lane < 4) *slot = first ? x : __fadd_rn(*slot, x);
    }
  }
}

template <int HD, int TP>
__host__ __device__ constexpr int forward_long_shared_bytes() { return 2 * 2 * TP * (HD + 8); }
template <int HD, int TP>
__host__ __device__ constexpr int backward_long_shared_bytes() {
  return 2 * 2 * TP * (HD + 8) + 4 * 3 * TP + 4 * 3 * NW * HD + 4 * 72;
}

template <int HD, int TP>
PMIX_DEV void forward_long_rows(const Args& a, int bh, int row0, const __nv_bfloat16* Ks, const __nv_bfloat16* Vs) {
  constexpr int LD = HD + 8;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t, lane = threadIdx.x & 31;
  uint32_t qf[HD / 16][4];
  load_afrag<HD>(qf, a.q, b, h, row0, t);
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#if PMIX_ONLINE_STATS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float s[2][4];
    step_products<HD>(s, qf, Ks, 16 * kk);
    step_scores(a, s, 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float x = s[j][2 * half + e];
          if (x != -INFINITY) {
            const float mn = fmaxf(m[half], x);
            l[half] = __fadd_rn(__fmul_rn(l[half], pmix_exp(__fsub_rn(m[half], mn))), pmix_exp(__fsub_rn(x, mn)));
            m[half] = mn;
          }
        }
  }
#pragma unroll
  for (int half = 0; half < 2; ++half)
#pragma unroll
    for (int sh = 1; sh <= 2; sh <<= 1) {
      // lanes combine in lane order: the lower lane's pair first, then the upper lane's pair
      const float mo = __shfl_xor_sync(0xffffffffu, m[half], sh), lo = __shfl_xor_sync(0xffffffffu, l[half], sh);
      const bool low = !(threadIdx.x & sh);
      const float m0 = low ? m[half] : mo, l0 = low ? l[half] : lo, m1 = low ? mo : m[half], l1 = low ? lo : l[half];
      const float mn = fmaxf(m0, m1);
      l[half] = __fadd_rn(__fmul_rn(l0, pmix_exp(__fsub_rn(m0, mn))), __fmul_rn(l1, pmix_exp(__fsub_rn(m1, mn))));
      m[half] = mn;
    }
#else
  PMIX_UNROLL_STEPS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float s[2][4];
    step_products<HD>(s, qf, Ks, 16 * kk);
    step_scores(a, s, 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j) m[half] = fmaxf(m[half], fmaxf(s[j][2 * half], s[j][2 * half + 1]));
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    m[half] = fmaxf(m[half], __shfl_xor_sync(0xffffffffu, m[half], 1));
    m[half] = fmaxf(m[half], __shfl_xor_sync(0xffffffffu, m[half], 2));
  }
  PMIX_UNROLL_STEPS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float s[2][4];
    step_products<HD>(s, qf, Ks, 16 * kk);
    step_scores(a, s, 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        l[half] = __fadd_rn(__fadd_rn(l[half], pmix_exp(__fsub_rn(s[j][2 * half], m[half]))),
                            pmix_exp(__fsub_rn(s[j][2 * half + 1], m[half])));
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 1));
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 2));
  }
#endif
  float o[HD / 8][4];
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  PMIX_UNROLL_STEPS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float s[2][4];
    step_probs<HD>(a, s, qf, Ks, 16 * kk, m, l);
    const uint32_t p[4] = {pack(s[0][0], s[0][1]), pack(s[0][2], s[0][3]), pack(s[1][0], s[1][1]), pack(s[1][2], s[1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Vs + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(o[2 * n2], p, bb[0], bb[1]);
      mma(o[2 * n2 + 1], p, bb[2], bb[3]);
    }
  }
  store_rows<HD>(a.o, o, b, h, row0, t);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = row0 + (lane >> 2) + 8 * half;
      if (row < t) reinterpret_cast<float2*>(a.stats)[(long long)bh * t + row] = make_float2(m[half], l[half]);
    }
  }
}

template <int HD, int TP>
PMIX_LONG_ENTRY void forward_long_phase_v2(const Args& a, long long items, float* smem);
template <int HD, int TP>
PMIX_LONG_ENTRY void backward_long_phase_v2(const Args& a, long long items, float* smem);
template <int HD, int TP>
PMIX_LONG_ENTRY void backward_long_phase_v3(const Args& a, long long items, float* smem);

template <int HD, int TP>
PMIX_DEV void forward_long_phase(const Args& a, long long items, float* smem) {
  if (PMIX_LONG_V2 && PMIX_LONG_FWD_V2) {   // P9-E: same units and arithmetic (see "long context v2")
    forward_long_phase_v2<HD, TP>(a, items, smem);
    return;
  }
  constexpr int LD = HD + 8, NB = TP / 16;
  static_assert(TP % 16 == 0 && TP <= 256 && (HD == 32 || HD == 64), "unsupported long mixed attention");
  zero_counters(a);
  __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Vs = Ks + TP * LD;
  const int warp = threadIdx.x >> 5;
  // P8-E: a head with more row blocks than warps is two items (row blocks [0, NW) and [NW, NB)), so 2 x BH items
  // spread over the grid instead of one round of BH items plus a sparse second round. Same per-row arithmetic.
  constexpr int G = NB > NW ? 2 : 1;
  for (long long work = blockIdx.x; work < G * items; work += gridDim.x) {
    const long long item = work / G;
    const int grp = (int)(work % G);
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    __syncthreads();
    load_operand<HD, TP>(Ks, a.k, b, h, a.t);
    load_operand<HD, TP>(Vs, a.v, b, h, a.t);
    __syncthreads();
    for (int blk = (G == 1 ? 0 : grp * NW) + warp; blk < (G == 1 || grp == 1 ? NB : NW); blk += NW)
      forward_long_rows<HD, TP>(a, (int)item, 16 * blk, Ks, Vs);
  }
}

// Stage 1: delta and dQ of one row block.
template <int HD, int TP>
PMIX_DEV void backward_long_rows(const Args& a, int bh, int row0, const __nv_bfloat16* Ks, const __nv_bfloat16* Vs,
                                 const float* stat, float* dlt, float* colsum, bool first) {
  constexpr int LD = HD + 8;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = 2 * (lane & 3), r = row0 + (lane >> 2);
  uint32_t qf[HD / 16][4], df[HD / 16][4];
  load_afrag<HD>(qf, a.q, b, h, row0, t);
  load_afrag<HD>(df, a.dout, b, h, row0, t);
  const float m[2] = {stat[2 * r], stat[2 * (r + 8)]}, l[2] = {stat[2 * r + 1], stat[2 * (r + 8) + 1]};
  float delta[2] = {0.f, 0.f};
  for (int kk = 0; kk < TP / 16; ++kk) {
    float p[2][4], dp[2][4];
    step_probs<HD>(a, p, qf, Ks, 16 * kk, m, l);
    step_products<HD>(dp, df, Vs, 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = 16 * kk + 8 * j + c;
        const float d0 = round_bf16(dp[j][2 * half]), d1 = round_bf16(dp[j][2 * half + 1]);
        if (col < t) delta[half] = __fadd_rn(delta[half], __fmul_rn(d0, p[j][2 * half]));
        if (col + 1 < t) delta[half] = __fadd_rn(delta[half], __fmul_rn(d1, p[j][2 * half + 1]));
      }
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    delta[half] = __fadd_rn(delta[half], __shfl_xor_sync(0xffffffffu, delta[half], 1));
    delta[half] = __fadd_rn(delta[half], __shfl_xor_sync(0xffffffffu, delta[half], 2));
  }
  if ((lane & 3) == 0) {
    dlt[r] = delta[0];
    dlt[r + 8] = delta[1];
  }
  float dq[HD / 8][4];
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
  for (int kk = 0; kk < TP / 16; ++kk) {
    float p[2][4], dp[2][4];
    step_probs<HD>(a, p, qf, Ks, 16 * kk, m, l);
    step_products<HD>(dp, df, Vs, 16 * kk);
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool live = (r + 8 * (e >> 1)) < t;   // padded query rows contribute nothing
        const float ds = apply_scale(a, __fmul_rn(p[j][e], __fsub_rn(round_bf16(dp[j][e]), delta[e >> 1])));
        dp[j][e] = live ? round_bf16(ds) : 0.f;
        if (a.pds) {   // transposed bf16(P) and bf16(dS) for the key stage: [key][query]
          const int key = 16 * kk + 8 * j + c + (e & 1), q = r + 8 * (e >> 1);
          __nv_bfloat16* base = a.pds + (long long)bh * 2 * TP * TP;
          base[(long long)key * TP + q] = __float2bfloat16_rn(live ? p[j][e] : 0.f);
          base[(long long)TP * TP + (long long)key * TP + q] = __float2bfloat16_rn(dp[j][e]);
        }
      }
    const uint32_t s4[4] = {pack(dp[0][0], dp[0][1]), pack(dp[0][2], dp[0][3]),
                            pack(dp[1][0], dp[1][1]), pack(dp[1][2], dp[1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Ks + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(dq[2 * n2], s4, bb[0], bb[1]);
      mma(dq[2 * n2 + 1], s4, bb[2], bb[3]);
    }
  }
  store_rows<HD>(a.dq, dq, b, h, row0, t);
  if (a.red[0].dst) column_sums_acc<HD>(colsum + (0 * NW + warp) * HD, dq, row0, t, first);
}

// Transposed probabilities P^T (and, with DS, bf16(dS)^T) of 16 queries [q0, q0 + 16) for one warp's 16 keys.
template <int HD, bool DS>
PMIX_DEV void step_transposed(const Args& a, float (&pv)[2][4], float (&sv)[2][4], const uint32_t (&kf)[HD / 16][4],
                              const uint32_t (&vf)[HD / 16][4], const __nv_bfloat16* Qs, const __nv_bfloat16* Ds,
                              const float* stat, const float* dlt, int key0, int q0) {
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3), kr = key0 + (lane >> 2), t = a.t;
  float dpt[2][4];
  step_products<HD>(pv, kf, Qs, q0);
  if (DS) step_products<HD>(dpt, vf, Ds, q0);
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int key = kr + 8 * (e >> 1), q = q0 + 8 * j + c + (e & 1);
      const float x = key < t ? apply_scale(a, round_bf16(pv[j][e])) : -INFINITY;
      const float p = pmix_div(pmix_exp(__fsub_rn(x, stat[2 * q])), stat[2 * q + 1], pmix_rinv(stat[2 * q + 1]));
      const bool live = q < t;   // padded query rows contribute nothing
      if (DS) {
        const float ds = apply_scale(a, __fmul_rn(p, __fsub_rn(round_bf16(dpt[j][e]), dlt[q])));
        sv[j][e] = live ? round_bf16(ds) : 0.f;
      }
      pv[j][e] = live ? p : 0.f;
    }
}

// Stage 2: dV and dK of one key block (two passes keep the register footprint under the two-CTA budget).
template <int HD, int TP>
PMIX_DEV void backward_long_keys(const Args& a, int bh, int key0, const __nv_bfloat16* Qs, const __nv_bfloat16* Ds,
                                 const float* stat, const float* dlt, float* colsum, bool first) {
  constexpr int LD = HD + 8;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t kf[HD / 16][4], vf[HD / 16][4];
  load_afrag<HD>(kf, a.k, b, h, key0, t);
  float acc[HD / 8][4];
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  const __nv_bfloat16* stored = a.pds ? a.pds + (long long)bh * 2 * TP * TP : nullptr;
  // A fragment (16 keys x 16 queries) of a stored transposed tile: rows key0 + g (+8), queries q0 + c (+8), pairs.
  auto frag = [&](const __nv_bfloat16* tile, int q0, uint32_t (&f)[4]) {
    const int g = lane >> 2, c = 2 * (lane & 3);
    const __nv_bfloat16* at = tile + (long long)(key0 + g) * TP + q0 + c;
    f[0] = __ldcg(reinterpret_cast<const unsigned int*>(at));
    f[1] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP));
    f[2] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8));
    f[3] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP + 8));
  };
  for (int qq = 0; qq < TP / 16; ++qq) {   // dV = bf16(P)^T dO
    float pv[2][4], sv[2][4];
    uint32_t pA[4];
    if (stored) {
      frag(stored, 16 * qq, pA);
    } else {
      step_transposed<HD, false>(a, pv, sv, kf, vf, Qs, Ds, stat, dlt, key0, 16 * qq);
      pA[0] = pack(pv[0][0], pv[0][1]); pA[1] = pack(pv[0][2], pv[0][3]);
      pA[2] = pack(pv[1][0], pv[1][1]); pA[3] = pack(pv[1][2], pv[1][3]);
    }
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t db[4];
      ldsm_x4_t(db, Ds + (16 * qq + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(acc[2 * n2], pA, db[0], db[1]);
      mma(acc[2 * n2 + 1], pA, db[2], db[3]);
    }
  }
  store_rows<HD>(a.dv, acc, b, h, key0, t);
  if (a.red[2].dst) column_sums_acc<HD>(colsum + (2 * NW + warp) * HD, acc, key0, t, first);
  load_afrag<HD>(vf, a.v, b, h, key0, t);
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  for (int qq = 0; qq < TP / 16; ++qq) {   // dK = bf16(dS)^T q
    float pv[2][4], sv[2][4];
    uint32_t sA[4];
    if (stored) {
      frag(stored + (long long)TP * TP, 16 * qq, sA);
    } else {
      step_transposed<HD, true>(a, pv, sv, kf, vf, Qs, Ds, stat, dlt, key0, 16 * qq);
      sA[0] = pack(sv[0][0], sv[0][1]); sA[1] = pack(sv[0][2], sv[0][3]);
      sA[2] = pack(sv[1][0], sv[1][1]); sA[3] = pack(sv[1][2], sv[1][3]);
    }
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t qb[4];
      ldsm_x4_t(qb, Qs + (16 * qq + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(acc[2 * n2], sA, qb[0], qb[1]);
      mma(acc[2 * n2 + 1], sA, qb[2], qb[3]);
    }
  }
  store_rows<HD>(a.dk, acc, b, h, key0, t);
  if (a.red[1].dst) column_sums_acc<HD>(colsum + (1 * NW + warp) * HD, acc, key0, t, first);
}

template <int HD, int TP>
PMIX_DEV void backward_long_phase(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16;
  static_assert(TP % 16 == 0 && TP <= 256 && (HD == 32 || HD == 64), "unsupported long mixed attention");
  __nv_bfloat16* X = reinterpret_cast<__nv_bfloat16*>(smem);   // stage 1: K, stage 2: Q
  __nv_bfloat16* Y = X + TP * LD;                              // stage 1: V, stage 2: dO
  float* stat = reinterpret_cast<float*>(Y + TP * LD);         // [TP][2] (m, l); padded rows (0, 1)
  float* dlt = stat + 2 * TP;                                  // [TP] delta
  float* colsum = dlt + TP;                                    // [3][NW][HD] + flags
  const int warp = threadIdx.x >> 5, t = a.t;
  // P8-E stage split (stored P/dS tiles, bias phase): the row stage and the key stage of an item are separate work
  // units, units [0, BH) row stages then [BH, 2 BH) key stages, round robin over the grid, so 2 BH half items balance
  // over the resident CTAs instead of BH whole items. A key stage reads only its item's stored tiles, Q and dO; it
  // waits for the item's row stage (release: fence + flag; acquire: flag + fence, tiles read from L2). Row stages
  // never wait and every CTA takes units in increasing order, so the (co-resident) grid cannot deadlock. Each bias
  // column sum is formed in one stage (dq: rows; dk, dv: keys) in the same warp/block order: bitwise unchanged.
  if constexpr (TP / 16 > NW) {   // P9-E: see "long backward v3"
    if (PMIX_LONG_V3 && a.flags && a.pds && a.halves && !a.counters) {
      backward_long_phase_v3<HD, TP>(a, items, smem);
      return;
    }
  }
  if (PMIX_LONG_V2 && a.flags && a.pds && !a.counters) {   // P9-E: staged operands (see "long context v2")
    backward_long_phase_v2<HD, TP>(a, items, smem);
    return;
  }
  if (a.flags && a.pds && !a.counters) {
    for (long long w = blockIdx.x; w < 2 * items; w += gridDim.x) {
      const bool rows = w < items;
      const long long item = rows ? w : w - items;
      const int b = (int)(item / a.heads), h = (int)(item % a.heads);
      __syncthreads();
      if (rows) {
        load_operand<HD, TP>(X, a.k, b, h, t);
        load_operand<HD, TP>(Y, a.v, b, h, t);
        for (int r = (int)threadIdx.x; r < TP; r += NT) {
          const float2 x = r < t ? reinterpret_cast<const float2*>(a.stats)[item * t + r] : make_float2(0.f, 1.f);
          stat[2 * r] = x.x;
          stat[2 * r + 1] = x.y;
        }
        __syncthreads();
        for (int blk = warp; blk < NB; blk += NW)
          backward_long_rows<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, blk < NW);
        if (warp >= NB && (threadIdx.x & 31) < 4)
          for (int d = 2 * (threadIdx.x & 31); d < HD; d += 8) colsum[warp * HD + d] = colsum[warp * HD + d + 1] = 0.f;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicExch(a.flags + item, 1u);
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 1);
      } else {
        if (threadIdx.x == 0) {
          while (atomicAdd(a.flags + item, 0u) == 0u) __nanosleep(64);
          a.flags[item] = 0u;   // sole reader: restore zero for later phases
        }
        __syncthreads();
        __threadfence();
        load_operand<HD, TP>(X, a.q, b, h, t);
        load_operand<HD, TP>(Y, a.dout, b, h, t);
        __syncthreads();
        for (int blk = warp; blk < NB; blk += NW)
          backward_long_keys<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, blk < NW);
        if (warp >= NB && (threadIdx.x & 31) < 4) {
          for (int k = 1; k < 3; ++k)
            for (int d = 2 * (threadIdx.x & 31); d < HD; d += 8) colsum[(k * NW + warp) * HD + d] = colsum[(k * NW + warp) * HD + d + 1] = 0.f;
        }
        __syncthreads();
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 6);
      }
    }
    return;
  }
  for (long long item = blockIdx.x; item < items; item += gridDim.x) {
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    __syncthreads();
    load_operand<HD, TP>(X, a.k, b, h, t);
    load_operand<HD, TP>(Y, a.v, b, h, t);
    for (int r = (int)threadIdx.x; r < TP; r += NT) {
      const float2 x = r < t ? reinterpret_cast<const float2*>(a.stats)[item * t + r] : make_float2(0.f, 1.f);
      stat[2 * r] = x.x;
      stat[2 * r + 1] = x.y;
    }
    __syncthreads();
    for (int blk = warp; blk < NB; blk += NW)
      backward_long_rows<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, blk < NW);
    if (a.pds) __threadfence_block();   // stored transposed tiles are read by other warps of this CTA
    __syncthreads();
    load_operand<HD, TP>(X, a.q, b, h, t);
    load_operand<HD, TP>(Y, a.dout, b, h, t);
    __syncthreads();
    for (int blk = warp; blk < NB; blk += NW)
      backward_long_keys<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, blk < NW);
    if (warp >= NB && (threadIdx.x & 31) < 4) {   // diagnostic T <= 112: idle warps contribute exact zeros
      for (int k = 0; k < 3; ++k)
        for (int d = 2 * (threadIdx.x & 31); d < HD; d += 8) colsum[(k * NW + warp) * HD + d] = colsum[(k * NW + warp) * HD + d + 1] = 0.f;
    }
    __syncthreads();
    reduce_item<HD, NW>(a, b, h, colsum, nullptr);
  }
  __syncthreads();
  finish_reductions<HD>(a, items, reinterpret_cast<int*>(colsum + 3 * NW * HD));
}

// ================================================================== long context v2 (P9-E, register scores, staged)
// Same work units, declared arithmetic and summation orders as the long items above (bitwise identical results);
// only data movement and instruction scheduling change:
//   forward: the warp's 16 x TP BF16-rounded scores are computed once (one q k^T pass instead of three) and kept
//            packed in registers; passes 2 (l) and 3 (P, O) unpack them, so x = fl(c * bf16(s)) is the same value.
//            K and V are staged by cp.async (bias completed in shared memory, as the dual items do) and the next
//            unit's V (into a second buffer) and K (after the CTA's score pass) load while this unit computes.
//   backward: operands staged by cp.async (bias completed in shared memory) instead of per-chunk synchronous loads;
//            the key stage loads all of a warp's stored P^T / dS^T fragments before its product chain.
// x of one 16-key step from the packed BF16-rounded scores: step_scores' values bit for bit.
PMIX_DEV void step_x(const Args& a, float (&x)[2][4], const uint32_t (&w)[4], int key0) {
  const int c = 2 * (threadIdx.x & 3);
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const uint32_t u = w[2 * j + (e >> 1)];
      const float v = __uint_as_float((e & 1) ? (u & 0xffff0000u) : (u << 16));
      const int col = key0 + 8 * j + c + (e & 1);
      x[j][e] = col < a.t ? apply_scale(a, v) : -INFINITY;
    }
}

template <int HD, int TP>
__host__ __device__ constexpr int forward_long_v2_shared_bytes() { return 2 * 3 * TP * (HD + 8) + 4 * 3 * HD; }

// One warp's 16 rows: pass 1 (scores, max), then `between()` (a CTA barrier point), then passes 2 and 3.
template <int HD, int TP, class F>
PMIX_DEV void forward_long_rows_v2(const Args& a, int bh, int row0, bool active, const __nv_bfloat16* Ks,
                                   const __nv_bfloat16* Vs, F between) {
  constexpr int LD = HD + 8, NS = TP / 16;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t, lane = threadIdx.x & 31;
  uint32_t sp[NS][4];
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  if (active) {
    uint32_t qf[HD / 16][4];
    load_afrag<HD>(qf, a.q, b, h, row0, t);
#pragma unroll
    for (int kk = 0; kk < NS; ++kk) {
      float s[2][4];
      step_products<HD>(s, qf, Ks, 16 * kk);
      sp[kk][0] = pack(s[0][0], s[0][1]);
      sp[kk][1] = pack(s[0][2], s[0][3]);
      sp[kk][2] = pack(s[1][0], s[1][1]);
      sp[kk][3] = pack(s[1][2], s[1][3]);
      float x[2][4];
      step_x(a, x, sp[kk], 16 * kk);
#pragma unroll
      for (int half = 0; half < 2; ++half)
#pragma unroll
        for (int j = 0; j < 2; ++j) m[half] = fmaxf(m[half], fmaxf(x[j][2 * half], x[j][2 * half + 1]));
    }
  }
  between();
  if (!active) return;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    m[half] = fmaxf(m[half], __shfl_xor_sync(0xffffffffu, m[half], 1));
    m[half] = fmaxf(m[half], __shfl_xor_sync(0xffffffffu, m[half], 2));
  }
#pragma unroll
  for (int kk = 0; kk < NS; ++kk) {
    float x[2][4];
    step_x(a, x, sp[kk], 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        l[half] = __fadd_rn(__fadd_rn(l[half], pmix_exp(__fsub_rn(x[j][2 * half], m[half]))),
                            pmix_exp(__fsub_rn(x[j][2 * half + 1], m[half])));
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 1));
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 2));
  }
  float o[HD / 8][4];
  const float li[2] = {pmix_rinv(l[0]), pmix_rinv(l[1])};
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < NS; ++kk) {
    float x[2][4];
    step_x(a, x, sp[kk], 16 * kk);
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) x[j][e] = pmix_div(pmix_exp(__fsub_rn(x[j][e], m[e >> 1])), l[e >> 1], li[e >> 1]);
    const uint32_t p[4] = {pack(x[0][0], x[0][1]), pack(x[0][2], x[0][3]), pack(x[1][0], x[1][1]), pack(x[1][2], x[1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Vs + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(o[2 * n2], p, bb[0], bb[1]);
      mma(o[2 * n2 + 1], p, bb[2], bb[3]);
    }
  }
  store_rows<HD>(a.o, o, b, h, row0, t);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = row0 + (lane >> 2) + 8 * half;
      if (row < t) reinterpret_cast<float2*>(a.stats)[(long long)bh * t + row] = make_float2(m[half], l[half]);
    }
  }
}

// P9-E PMIX_FA_FWD (declared amendment A7, flash forward arithmetic): one pass over 64-key chunks per warp row block.
// Scores are the same BF16-rounded, scaled values x as step_scores; the running row max m and the per-lane partial
// sums l and outputs O are rescaled by exp(m_old - m_new) at each chunk; O accumulates bf16(exp(x - m_run)) v in FP32
// and is divided by l once at the end. The saved (m, l) are the final row max and sum relative to it, so the backward
// (which recomputes P = exp(x - m) / l from them) is unchanged. Eight independent n8 tiles per k16 step (64 keys)
// instead of two; one score product per element instead of three.
#ifndef PMIX_FA_FWD
#define PMIX_FA_FWD 0
#endif
#ifndef PMIX_FA_UNITS
#define PMIX_FA_UNITS 0   // 0: half-item units (as v2); 1: whole items, warps loop over the head's row blocks
#endif
template <int HD, int J>
PMIX_DEV void fa_chunk(const Args& a, const uint32_t (&qf)[HD / 16][4], const __nv_bfloat16* Ks,
                       const __nv_bfloat16* Vs, int key0, float (&m)[2], float (&l)[2], float (&o)[HD / 8][4]) {
  constexpr int LD = HD + 8;
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
  float s[2 * J][4];
#pragma unroll
  for (int j = 0; j < 2 * J; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
  for (int ks = 0; ks < HD / 16; ++ks)
#pragma unroll
    for (int jj = 0; jj < J; ++jj) {
      uint32_t bb[4];
      ldsm_x4(bb, Ks + (key0 + 16 * jj + (lane & 7) + 8 * (lane >> 4)) * LD + 16 * ks + 8 * ((lane >> 3) & 1));
      mma(s[2 * jj], qf[ks], bb[0], bb[1]);
      mma(s[2 * jj + 1], qf[ks], bb[2], bb[3]);
    }
  float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
  for (int j = 0; j < 2 * J; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int col = key0 + 8 * j + c + (e & 1);
      s[j][e] = col < a.t ? apply_scale(a, round_bf16(s[j][e])) : -INFINITY;
      mx[e >> 1] = fmaxf(mx[e >> 1], s[j][e]);
    }
  float alpha[2], mn[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffffu, mx[half], 1));
    mx[half] = fmaxf(mx[half], __shfl_xor_sync(0xffffffffu, mx[half], 2));
    mn[half] = fmaxf(m[half], mx[half]);
    // a fully masked chunk keeps m = -inf only before any valid key; key 0 is always valid, so mn is finite here
    alpha[half] = m[half] == -INFINITY ? 0.f : pmix_exp(__fsub_rn(m[half], mn[half]));
    m[half] = mn[half];
  }
  float rs[2] = {0.f, 0.f};
#pragma unroll
  for (int j = 0; j < 2 * J; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      s[j][e] = pmix_exp(__fsub_rn(s[j][e], mn[e >> 1]));
      rs[e >> 1] = __fadd_rn(rs[e >> 1], s[j][e]);
    }
#pragma unroll
  for (int half = 0; half < 2; ++half) l[half] = __fadd_rn(__fmul_rn(l[half], alpha[half]), rs[half]);
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) {
    o[n][0] = __fmul_rn(o[n][0], alpha[0]);
    o[n][1] = __fmul_rn(o[n][1], alpha[0]);
    o[n][2] = __fmul_rn(o[n][2], alpha[1]);
    o[n][3] = __fmul_rn(o[n][3], alpha[1]);
  }
#pragma unroll
  for (int jj = 0; jj < J; ++jj) {
    const uint32_t pf[4] = {pack(s[2 * jj][0], s[2 * jj][1]), pack(s[2 * jj][2], s[2 * jj][3]),
                            pack(s[2 * jj + 1][0], s[2 * jj + 1][1]), pack(s[2 * jj + 1][2], s[2 * jj + 1][3])};
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Vs + (key0 + 16 * jj + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(o[2 * n2], pf, bb[0], bb[1]);
      mma(o[2 * n2 + 1], pf, bb[2], bb[3]);
    }
  }
}

template <int HD, int TP>
PMIX_DEV void forward_long_rows_fa(const Args& a, int bh, int row0, const __nv_bfloat16* Ks, const __nv_bfloat16* Vs) {
  constexpr int NS = TP / 16, NC = NS / 4, TAIL = NS % 4;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t, lane = threadIdx.x & 31;
  uint32_t qf[HD / 16][4];
  load_afrag<HD>(qf, a.q, b, h, row0, t);
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  float o[HD / 8][4];
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
#pragma unroll 1
  for (int cc = 0; cc < NC; ++cc) fa_chunk<HD, 4>(a, qf, Ks, Vs, 64 * cc, m, l, o);
  if constexpr (TAIL > 0) fa_chunk<HD, TAIL>(a, qf, Ks, Vs, 64 * NC, m, l, o);
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 1));
    l[half] = __fadd_rn(l[half], __shfl_xor_sync(0xffffffffu, l[half], 2));
  }
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) {
    o[n][0] = __fdiv_rn(o[n][0], l[0]);
    o[n][1] = __fdiv_rn(o[n][1], l[0]);
    o[n][2] = __fdiv_rn(o[n][2], l[1]);
    o[n][3] = __fdiv_rn(o[n][3], l[1]);
  }
  store_rows<HD>(a.o, o, b, h, row0, t);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = row0 + (lane >> 2) + 8 * half;
      if (row < t) reinterpret_cast<float2*>(a.stats)[(long long)bh * t + row] = make_float2(m[half], l[half]);
    }
  }
}

// Units as forward_long_phase (G row groups per head, round robin). Shared: K, V[2], bias[3][HD].
template <int HD, int TP>
PMIX_LONG_ENTRY void forward_long_phase_v2(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16, G = (NB > NW && !(PMIX_FA_FWD && PMIX_FA_UNITS == 1)) ? 2 : 1;
  static_assert(TP % 16 == 0 && TP <= 256 && (HD == 32 || HD == 64) && NB <= 2 * NW, "unsupported long mixed attention");
  static_assert(!PMIX_FA_FWD || !PMIX_LONG_FWD_REG, "flash forward rows use the staged v1 placement");
  zero_counters(a);
  __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* V0 = Ks + TP * LD;
  float* bias = reinterpret_cast<float*>(V0 + 2 * TP * LD);   // [K, V0, V1][HD]
  const int warp = threadIdx.x >> 5, tid = threadIdx.x, t = a.t;
  // P9-E PMIX_FA_UNITS == 2 (mixed units): the first W items are whole units, the rest half units, with
  // W = clamp(2 items - 2 grid, 0, items) so every CTA draws at most two units (round robin).
  constexpr bool MIXED = PMIX_FA_FWD && PMIX_FA_UNITS == 2 && NB > NW;
  const long long W = MIXED ? max(0LL, min(items, 2 * items - 2 * (long long)gridDim.x)) : 0;
  const long long total = MIXED ? W + 2 * (items - W) : G * items;
  auto decode = [&](long long work, int& lo, int& hi) -> long long {
    if (MIXED) {
      if (work < W) { lo = 0; hi = NB; return work; }
      const long long v = work - W;
      lo = (int)(v & 1) * NW;
      hi = (v & 1) ? NB : NW;
      return W + v / 2;
    }
    const int grp = (int)(work % G);
    lo = G == 1 ? 0 : grp * NW;
    hi = (G == 1 || grp == 1) ? NB : NW;
    return work / G;
  };
  auto stage = [&](const Operand& op, long long work, __nv_bfloat16* dst, float* bdst) {
    int lo_, hi_;
    const long long item = decode(work, lo_, hi_);
    stage_operand_sub<HD, TP>(dst, bdst, op, (int)(item / a.heads), (int)(item % a.heads), t, tid, NT);
  };
  long long work = blockIdx.x;
  if (work < total) {
    __syncthreads();                                   // earlier users of shared memory are done
    stage(a.k, work, Ks, bias);
    stage(a.v, work, V0, bias + HD);
    cp_commit();
  }
  for (int i = 0; work < total; ++i, work += gridDim.x) {
    int lo, hi;
    const long long item = decode(work, lo, hi), next = work + gridDim.x;
    __nv_bfloat16* Vs = V0 + (i & 1) * TP * LD;
    cp_wait_all();
    __syncthreads();
    finish_operand_sub<HD, TP>(Ks, bias, a.k, t, tid, NT);
    finish_operand_sub<HD, TP>(Vs, bias + HD * (1 + (i & 1)), a.v, t, tid, NT);
    __syncthreads();
    if (next < total) {                                // the other V buffer was last read by unit i - 1
      stage(a.v, next, V0 + ((i + 1) & 1) * TP * LD, bias + HD * (1 + ((i + 1) & 1)));
      cp_commit();
    }
    const int blk = lo + warp;
    if (PMIX_LONG_FWD_REG) {
      forward_long_rows_v2<HD, TP>(a, (int)item, 16 * blk, blk < hi, Ks, Vs, [&]() {
        __syncthreads();                               // every warp's score pass has read K
        if (next < total) {
          stage(a.k, next, Ks, bias);
          cp_commit();
        }
      });
    } else {
      if (PMIX_FA_FWD) {
        for (int bb_ = blk; bb_ < hi; bb_ += NW) forward_long_rows_fa<HD, TP>(a, (int)item, 16 * bb_, Ks, Vs);
      } else if (blk < hi) {
        forward_long_rows<HD, TP>(a, (int)item, 16 * blk, Ks, Vs);
      }
      __syncthreads();                                 // K is read by every pass of the v1 rows
      if (next < total) {
        stage(a.k, next, Ks, bias);
        cp_commit();
      }
    }
  }
}

template <int HD, int TP>
__host__ __device__ constexpr int backward_long_v2_shared_bytes() {
  return backward_long_shared_bytes<HD, TP>() + 4 * 2 * HD;
}

PMIX_DEV uint32_t movt(uint32_t x) {   // 8x8 BF16 transpose across the warp (row lane/4, pair 2 (lane % 4))
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}

// Stage 1 of the v2 backward: backward_long_rows with the transposed bf16(P) / bf16(dS) tiles written as 4-byte
// query pairs (packed fragments transposed in registers by movmatrix) instead of one 2-byte store per element.
// Values, layout ([key][query]) and every other operation are those of backward_long_rows.
template <int HD, int TP>
PMIX_DEV void backward_long_rows_v2(const Args& a, int bh, int row0, const __nv_bfloat16* Ks, const __nv_bfloat16* Vs,
                                    const float* stat, float* dlt, float* colsum, bool first) {
  constexpr int LD = HD + 8;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, c = 2 * (lane & 3), r = row0 + (lane >> 2);
  uint32_t qf[HD / 16][4], df[HD / 16][4];
  load_afrag<HD>(qf, a.q, b, h, row0, t);
  load_afrag<HD>(df, a.dout, b, h, row0, t);
  const float m[2] = {stat[2 * r], stat[2 * (r + 8)]}, l[2] = {stat[2 * r + 1], stat[2 * (r + 8) + 1]};
  float delta[2] = {0.f, 0.f};
#if PMIX_FA_DELTA
  {
    uint32_t of[HD / 16][4];
    load_afrag<HD>(of, a.ofwd, b, h, row0, t);
#pragma unroll
    for (int ks = 0; ks < HD / 16; ++ks)
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int half = jj & 1;
        const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&df[ks][jj]));
        const float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&of[ks][jj]));
        delta[half] = __fadd_rn(__fadd_rn(delta[half], __fmul_rn(x.x, y.x)), __fmul_rn(x.y, y.y));
      }
  }
#else
  PMIX_UNROLL_STEPS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float p[2][4], dp[2][4];
    step_probs<HD>(a, p, qf, Ks, 16 * kk, m, l);
    step_products<HD>(dp, df, Vs, 16 * kk);
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = 16 * kk + 8 * j + c;
        const float d0 = round_bf16(dp[j][2 * half]), d1 = round_bf16(dp[j][2 * half + 1]);
        if (col < t) delta[half] = __fadd_rn(delta[half], __fmul_rn(d0, p[j][2 * half]));
        if (col + 1 < t) delta[half] = __fadd_rn(delta[half], __fmul_rn(d1, p[j][2 * half + 1]));
      }
  }
#endif
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    delta[half] = __fadd_rn(delta[half], __shfl_xor_sync(0xffffffffu, delta[half], 1));
    delta[half] = __fadd_rn(delta[half], __shfl_xor_sync(0xffffffffu, delta[half], 2));
  }
  if ((lane & 3) == 0) {
    dlt[r] = delta[0];
    dlt[r + 8] = delta[1];
  }
  float dq[HD / 8][4];
#pragma unroll
  for (int n = 0; n < HD / 8; ++n) dq[n][0] = dq[n][1] = dq[n][2] = dq[n][3] = 0.f;
  __nv_bfloat16* base = a.pds + (long long)bh * 2 * TP * TP;
  const bool live0 = r < t, live1 = r + 8 < t;   // padded query rows contribute nothing
#if PMIX_ROWS_W
  // P9-E: two 16-key steps per iteration -- both steps' score and dP products are issued before either's
  // elementwise work (four independent n8 chains each); stores and the dQ products keep increasing key order,
  // so every value and summation order is that of the loop below (bitwise identical).
  for (int kk = 0; kk < TP / 16; kk += 2) {
    const int nsub = kk + 1 < TP / 16 ? 2 : 1;
    float p2[2][2][4], dp2[2][2][4];
    step_probs<HD>(a, p2[0], qf, Ks, 16 * kk, m, l);
    if (nsub == 2) step_probs<HD>(a, p2[1], qf, Ks, 16 * (kk + 1), m, l);
    step_products<HD>(dp2[0], df, Vs, 16 * kk);
    if (nsub == 2) step_products<HD>(dp2[1], df, Vs, 16 * (kk + 1));
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      if (u >= nsub) break;
      const int k1 = kk + u;
      float (&p)[2][4] = p2[u];
      float (&dp)[2][4] = dp2[u];
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const bool live = (e >> 1) ? live1 : live0;
          const float ds = apply_scale(a, __fmul_rn(p[j][e], __fsub_rn(round_bf16(dp[j][e]), delta[e >> 1])));
          dp[j][e] = live ? round_bf16(ds) : 0.f;
          p[j][e] = live ? p[j][e] : 0.f;
        }
      const uint32_t s4[4] = {pack(dp[0][0], dp[0][1]), pack(dp[0][2], dp[0][3]),
                              pack(dp[1][0], dp[1][1]), pack(dp[1][2], dp[1][3])};
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const long long at = (long long)(16 * k1 + 8 * j + (lane >> 2)) * TP + row0 + 8 * half + c;
          *reinterpret_cast<uint32_t*>(base + at) = movt(pack(p[j][2 * half], p[j][2 * half + 1]));
          *reinterpret_cast<uint32_t*>(base + (long long)TP * TP + at) = movt(s4[2 * j + half]);
        }
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t bb[4];
        ldsm_x4_t(bb, Ks + (16 * k1 + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(dq[2 * n2], s4, bb[0], bb[1]);
        mma(dq[2 * n2 + 1], s4, bb[2], bb[3]);
      }
    }
  }
  if (false)
#endif
  PMIX_UNROLL_STEPS
  for (int kk = 0; kk < TP / 16; ++kk) {
    float p[2][4], dp[2][4];
    step_probs<HD>(a, p, qf, Ks, 16 * kk, m, l);
    step_products<HD>(dp, df, Vs, 16 * kk);
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool live = (e >> 1) ? live1 : live0;
        const float ds = apply_scale(a, __fmul_rn(p[j][e], __fsub_rn(round_bf16(dp[j][e]), delta[e >> 1])));
        dp[j][e] = live ? round_bf16(ds) : 0.f;
        p[j][e] = live ? p[j][e] : 0.f;
      }
    const uint32_t s4[4] = {pack(dp[0][0], dp[0][1]), pack(dp[0][2], dp[0][3]),
                            pack(dp[1][0], dp[1][1]), pack(dp[1][2], dp[1][3])};
    // [key][query] tiles: key 16 kk + 8 j + lane / 4, queries row0 + 8 half + 2 (lane % 4) and the next
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const long long at = (long long)(16 * kk + 8 * j + (lane >> 2)) * TP + row0 + 8 * half + c;
        if (!PMIX_KEYS_RP) *reinterpret_cast<uint32_t*>(base + at) = movt(pack(p[j][2 * half], p[j][2 * half + 1]));
        *reinterpret_cast<uint32_t*>(base + (long long)TP * TP + at) = movt(s4[2 * j + half]);
      }
#pragma unroll
    for (int n2 = 0; n2 < HD / 16; ++n2) {
      uint32_t bb[4];
      ldsm_x4_t(bb, Ks + (16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
      mma(dq[2 * n2], s4, bb[0], bb[1]);
      mma(dq[2 * n2 + 1], s4, bb[2], bb[3]);
    }
  }
  store_rows<HD>(a.dq, dq, b, h, row0, t);
  if (a.red[0].dst) column_sums_acc<HD>(colsum + (0 * NW + warp) * HD, dq, row0, t, first);
}

// Stage 2 of the v2 backward: backward_long_keys with every stored fragment of a pass loaded before its products.
template <int HD, int TP>
PMIX_DEV void backward_long_keys_v2(const Args& a, int bh, int key0, const __nv_bfloat16* Qs, const __nv_bfloat16* Ds,
                                    float* colsum, bool first, const float* stat = nullptr) {
  constexpr int LD = HD + 8, NS = TP / 16;
  const int b = bh / a.heads, h = bh % a.heads, t = a.t;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = 2 * (lane & 3);
  const __nv_bfloat16* stored = a.pds + (long long)bh * 2 * TP * TP;
  float acc[HD / 8][4];
#if PMIX_KEYS_RP
  {   // P9-E: dV = bf16(P)^T dO with P^T recomputed from q (staged), this warp's k and the saved row statistics
      // (backward_long_keys' recompute path: step_transposed), instead of reading a stored P^T tile.
    uint32_t kf[HD / 16][4];
    load_afrag<HD>(kf, a.k, b, h, key0, t);
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
    for (int qq = 0; qq < NS; ++qq) {
      float pv[2][4], sv[2][4];
      step_transposed<HD, false>(a, pv, sv, kf, kf, Qs, Ds, stat, nullptr, key0, 16 * qq);
      const uint32_t pA[4] = {pack(pv[0][0], pv[0][1]), pack(pv[0][2], pv[0][3]),
                              pack(pv[1][0], pv[1][1]), pack(pv[1][2], pv[1][3])};
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t db[4];
        ldsm_x4_t(db, Ds + (16 * qq + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(acc[2 * n2], pA, db[0], db[1]);
        mma(acc[2 * n2 + 1], pA, db[2], db[3]);
      }
    }
    store_rows<HD>(a.dv, acc, b, h, key0, t);
    if (a.red[2].dst) column_sums_acc<HD>(colsum + (2 * NW + warp) * HD, acc, key0, t, first);
  }
#endif
#pragma unroll
  for (int pass = PMIX_KEYS_RP ? 1 : 0; pass < 2; ++pass) {   // 0: dV = bf16(P)^T dO, 1: dK = bf16(dS)^T q
    const __nv_bfloat16* tile = stored + (long long)pass * TP * TP + (long long)(key0 + g) * TP + c;
    const __nv_bfloat16* Bs = pass == 0 ? Ds : Qs;
#pragma unroll
    for (int n = 0; n < HD / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#if PMIX_KEYS_CHUNK > 0
    // P9-E: stored fragments in chunks of PMIX_KEYS_CHUNK query steps, the next chunk loading while this one's
    // products run (2 x CHUNK x 4 registers instead of NS x 4); same products in the same increasing order.
    constexpr int KC = PMIX_KEYS_CHUNK;
    uint32_t fb[2][KC][4];
    auto load_chunk = [&](uint32_t (&dst)[KC][4], int q0) {
#pragma unroll
      for (int u = 0; u < KC; ++u) {
        if (q0 + u < NS) {
          const __nv_bfloat16* at = tile + 16 * (q0 + u);
          dst[u][0] = __ldcg(reinterpret_cast<const unsigned int*>(at));
          dst[u][1] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP));
          dst[u][2] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8));
          dst[u][3] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP + 8));
        }
      }
    };
    load_chunk(fb[0], 0);
#pragma unroll
    for (int q0 = 0; q0 < NS; q0 += KC) {
      const int cur = (q0 / KC) & 1;
      if (q0 + KC < NS) load_chunk(fb[cur ^ 1], q0 + KC);
#pragma unroll
      for (int u = 0; u < KC; ++u) {
        const int qq = q0 + u;
        if (qq < NS) {
#pragma unroll
          for (int n2 = 0; n2 < HD / 16; ++n2) {
            uint32_t db[4];
            ldsm_x4_t(db, Bs + (16 * qq + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
            mma(acc[2 * n2], fb[cur][u], db[0], db[1]);
            mma(acc[2 * n2 + 1], fb[cur][u], db[2], db[3]);
          }
        }
      }
    }
#else
    uint32_t f[NS][4];
#pragma unroll
    for (int qq = 0; qq < NS; ++qq) {
      const __nv_bfloat16* at = tile + 16 * qq;
      f[qq][0] = __ldcg(reinterpret_cast<const unsigned int*>(at));
      f[qq][1] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP));
      f[qq][2] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8));
      f[qq][3] = __ldcg(reinterpret_cast<const unsigned int*>(at + 8LL * TP + 8));
    }
#pragma unroll
    for (int qq = 0; qq < NS; ++qq) {
#pragma unroll
      for (int n2 = 0; n2 < HD / 16; ++n2) {
        uint32_t db[4];
        ldsm_x4_t(db, Bs + (16 * qq + (lane & 7) + 8 * ((lane >> 3) & 1)) * LD + 16 * n2 + 8 * (lane >> 4));
        mma(acc[2 * n2], f[qq], db[0], db[1]);
        mma(acc[2 * n2 + 1], f[qq], db[2], db[3]);
      }
    }
#endif
    if (pass == 0) {
      store_rows<HD>(a.dv, acc, b, h, key0, t);
      if (a.red[2].dst) column_sums_acc<HD>(colsum + (2 * NW + warp) * HD, acc, key0, t, first);
    } else {
      store_rows<HD>(a.dk, acc, b, h, key0, t);
      if (a.red[1].dst) column_sums_acc<HD>(colsum + (1 * NW + warp) * HD, acc, key0, t, first);
    }
  }
}

// Stage-split backward (stored P/dS, flags, bias phase) with staged operands; same units and orders as
// backward_long_phase's split branch.
template <int HD, int TP>
PMIX_LONG_ENTRY void backward_long_phase_v2(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16;
  static_assert(TP % 16 == 0 && TP <= 256 && (HD == 32 || HD == 64), "unsupported long mixed attention");
  __nv_bfloat16* X = reinterpret_cast<__nv_bfloat16*>(smem);   // row stage: K, key stage: Q
  __nv_bfloat16* Y = X + TP * LD;                              // row stage: V, key stage: dO
  float* stat = reinterpret_cast<float*>(Y + TP * LD);
  float* dlt = stat + 2 * TP;
  float* colsum = dlt + TP;                                    // [3][NW][HD] + flags (72 words)
  float* bias = colsum + 3 * NW * HD + 72;                     // [2][HD]
  const int warp = threadIdx.x >> 5, tid = threadIdx.x, t = a.t;
  for (long long w = blockIdx.x; w < 2 * items; w += gridDim.x) {
    const bool rows = w < items;
    const long long item = rows ? w : w - items;
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    __syncthreads();
    if (rows) {
      stage_operand_sub<HD, TP>(X, bias, a.k, b, h, t, tid, NT);
      stage_operand_sub<HD, TP>(Y, bias + HD, a.v, b, h, t, tid, NT);
      for (int r = tid; r < TP; r += NT) {
        if (r < t) cp8(stat + 2 * r, a.stats + 2 * (item * t + r));
        else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
      }
      cp_commit();
      cp_wait_all();
      __syncthreads();
      finish_operand_sub<HD, TP>(X, bias, a.k, t, tid, NT);
      finish_operand_sub<HD, TP>(Y, bias + HD, a.v, t, tid, NT);
      __syncthreads();
      for (int blk = warp; blk < NB; blk += NW)
        backward_long_rows_v2<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, blk < NW);
      if (warp >= NB && (threadIdx.x & 31) < 4)
        for (int d = 2 * (threadIdx.x & 31); d < HD; d += 8) colsum[warp * HD + d] = colsum[warp * HD + d + 1] = 0.f;
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) atomicExch(a.flags + item, 1u);
      reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 1);
    } else {
      // Operands of the key stage do not depend on the row stage: stage them before waiting for its flag.
      stage_operand_sub<HD, TP>(X, bias, a.q, b, h, t, tid, NT);
      stage_operand_sub<HD, TP>(Y, bias + HD, a.dout, b, h, t, tid, NT);
      cp_commit();
      if (threadIdx.x == 0) {
        while (atomicAdd(a.flags + item, 0u) == 0u) __nanosleep(64);
        a.flags[item] = 0u;
      }
      cp_wait_all();
      __syncthreads();
      __threadfence();
      finish_operand_sub<HD, TP>(X, bias, a.q, t, tid, NT);
      finish_operand_sub<HD, TP>(Y, bias + HD, a.dout, t, tid, NT);
      __syncthreads();
      for (int blk = warp; blk < NB; blk += NW)
        backward_long_keys_v2<HD, TP>(a, (int)item, 16 * blk, X, Y, colsum, blk < NW);
      if (warp >= NB && (threadIdx.x & 31) < 4) {
        for (int k = 1; k < 3; ++k)
          for (int d = 2 * (threadIdx.x & 31); d < HD; d += 8) colsum[(k * NW + warp) * HD + d] = colsum[(k * NW + warp) * HD + d + 1] = 0.f;
      }
      __syncthreads();
      reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 6);
    }
  }
}

// ================================================================== long backward v3 (P9-E, half-item units, claimed)
// Units: [0, 2 BH) row halves, [2 BH, 4 BH) key halves; half 0 = blocks [0, NW), half 1 = blocks [NW, NB); each
// warp owns at most one block of a unit. Units are claimed in increasing order from a counter (rows before keys),
// so a key half only waits for row halves already held by running CTAs. Each half writes its warps' bias column
// sums (one block each, `first`) to global partials; the second finisher of an item forms
// fl(partial(half 0) + partial(half 1)) per warp (warps with one block keep their half-0 value) -- exactly the
// accumulation of the v1 warps over blocks w and w + NW -- then reduce_item sums the warps in order. Products,
// stored tiles and every summation order are those of the v1/v2 items: bitwise identical results.
// Counters (zero at entry, restored to zero): flags[BH] rows published, rows_done[BH], keys_done[BH], claim, leave.
template <int HD, int TP>
PMIX_LONG_ENTRY void backward_long_phase_v3(const Args& a, long long items, float* smem) {
  constexpr int LD = HD + 8, NB = TP / 16, H1 = NB - NW;
  static_assert(NB > NW && NB <= 2 * NW && (HD == 32 || HD == 64), "half units need NW < NB <= 2 NW");
  __nv_bfloat16* X = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Y = X + TP * LD;
  float* stat = reinterpret_cast<float*>(Y + TP * LD);
  float* dlt = stat + 2 * TP;
  float* colsum = dlt + TP;                                    // [3][NW][HD] + 72 words (slot[0..1] below)
  float* bias = colsum + 3 * NW * HD + 72;                     // [2][HD]
  volatile unsigned* slot = reinterpret_cast<volatile unsigned*>(colsum + 3 * NW * HD);
  unsigned* flags = a.flags;
  unsigned* rows_done = a.flags + items;
  unsigned* keys_done = a.flags + 2 * items;
  unsigned* claim = a.flags + 3 * items;
  unsigned* leave = claim + 1;
  const int warp = threadIdx.x >> 5, tid = threadIdx.x, t = a.t;
  const bool any = a.red[0].dst || a.red[1].dst || a.red[2].dst;
  constexpr int HALVES = PMIX_LONG_V3_HALVES;
  // whole-item units for items [0, W), half-item units for [W, items)
  const long long W = PMIX_LONG_V3_MIXED ? (items < (long long)gridDim.x ? items : (long long)gridDim.x)
                                         : (HALVES ? 0LL : items);
  const long long per_stage = W + 2 * (items - W);
  const unsigned total = (unsigned)(2 * per_stage);
  for (;;) {
    __syncthreads();                                           // shared operands and slots of the previous unit
    if (tid == 0) slot[0] = atomicAdd(claim, 1u);
    __syncthreads();
    const unsigned u = slot[0];
    if (u >= total) break;
    const bool rows = u < per_stage;
    const long long v = rows ? (long long)u : (long long)u - per_stage;
    const bool whole = v < W;
    const long long item = whole ? v : W + ((v - W) >> 1);
    const int half = whole ? 0 : (int)((v - W) & 1);
    const int b = (int)(item / a.heads), h = (int)(item % a.heads);
    const int blk = half * NW + warp;
    const bool active = blk < NB;
    float* hp = a.halves + ((item * 2 + half) * 3) * (long long)(NW * HD);   // this half's [3][NW][HD]
    float* hq = a.halves + ((item * 2 + (1 - half)) * 3) * (long long)(NW * HD);
    if (PMIX_DIAG_SKIP_ROWS && rows && whole) {      // diagnostic timing builds only (invalid gradients)
      __syncthreads();
      if (tid == 0) atomicExch(flags + item, 1u);
      continue;
    }
    if (PMIX_DIAG_SKIP_KEYS && !rows && whole) {     // diagnostic timing builds only (invalid gradients)
      if (tid == 0) {
        while (atomicAdd(flags + item, 0u) == 0u) __nanosleep(64);
        flags[item] = 0u;
      }
      continue;
    }
    if (rows) {
      if (PMIX_V3_DIRECT) {
        for (int r = tid; r < TP; r += NT) {
          if (r < t) cp8(stat + 2 * r, a.stats + 2 * (item * t + r));
          else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
        }
        cp_commit();
        stage_operand_direct<HD, TP>(X, a.k, b, h, t, tid);
        stage_operand_direct<HD, TP>(Y, a.v, b, h, t, tid);
        cp_wait_all();
        __syncthreads();
      } else {
      stage_operand_sub<HD, TP>(X, bias, a.k, b, h, t, tid, NT);
      stage_operand_sub<HD, TP>(Y, bias + HD, a.v, b, h, t, tid, NT);
      for (int r = tid; r < TP; r += NT) {
        if (r < t) cp8(stat + 2 * r, a.stats + 2 * (item * t + r));
        else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
      }
      cp_commit();
      cp_wait_all();
      __syncthreads();
      finish_operand_sub<HD, TP>(X, bias, a.k, t, tid, NT);
      finish_operand_sub<HD, TP>(Y, bias + HD, a.v, t, tid, NT);
      __syncthreads();
      }
      if (whole) {
        for (int bb = warp; bb < NB; bb += NW)
          backward_long_rows_v2<HD, TP>(a, (int)item, 16 * bb, X, Y, stat, dlt, colsum, bb < NW);
        __threadfence();
        __syncthreads();
        if (tid == 0) atomicExch(flags + item, 1u);
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 1);
        continue;
      }
      if (active) backward_long_rows_v2<HD, TP>(a, (int)item, 16 * blk, X, Y, stat, dlt, colsum, true);
      __syncthreads();
      if (a.red[0].dst)
        for (int i = tid; i < NW * HD; i += NT) hp[i] = colsum[i];
      __threadfence();                                         // stored tiles and partials before the count
      __syncthreads();
      if (tid == 0) slot[1] = atomicAdd(rows_done + item, 1u);
      __syncthreads();
      if (slot[1] == 1u) {                                     // second row half of this item
        __threadfence();
        if (a.red[0].dst) {
          const float* p0 = half == 0 ? hp : hq;
          const float* p1 = half == 0 ? hq : hp;
          for (int i = tid; i < NW * HD; i += NT) {
            const int w = i / HD;
            colsum[i] = w < H1 ? __fadd_rn(__ldcg(p0 + i), __ldcg(p1 + i)) : __ldcg(p0 + i);
          }
        }
        __syncthreads();
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 1);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
          rows_done[item] = 0u;
          atomicExch(flags + item, 1u);
        }
      }
    } else {
      // Operands of the key stage do not depend on the row stage: stage them before waiting for its flag.
      if (PMIX_V3_DIRECT) {
        stage_operand_direct<HD, TP>(X, a.q, b, h, t, tid);
        stage_operand_direct<HD, TP>(Y, a.dout, b, h, t, tid);
        if (tid == 0)
          while (atomicAdd(flags + item, 0u) == 0u) __nanosleep(64);
        __syncthreads();
        __threadfence();
      } else {
      stage_operand_sub<HD, TP>(X, bias, a.q, b, h, t, tid, NT);
      stage_operand_sub<HD, TP>(Y, bias + HD, a.dout, b, h, t, tid, NT);
      if (PMIX_KEYS_RP)
        for (int r = tid; r < TP; r += NT) {
          if (r < t) cp8(stat + 2 * r, a.stats + 2 * (item * t + r));
          else { stat[2 * r] = 0.f; stat[2 * r + 1] = 1.f; }
        }
      cp_commit();
      if (tid == 0)
        while (atomicAdd(flags + item, 0u) == 0u) __nanosleep(64);
      cp_wait_all();
      __syncthreads();
      __threadfence();
      finish_operand_sub<HD, TP>(X, bias, a.q, t, tid, NT);
      finish_operand_sub<HD, TP>(Y, bias + HD, a.dout, t, tid, NT);
      __syncthreads();
      }
      if (whole) {
        for (int bb = warp; bb < NB; bb += NW)
          backward_long_keys_v2<HD, TP>(a, (int)item, 16 * bb, X, Y, colsum, bb < NW, stat);
        __syncthreads();
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 6);
        if (tid == 0) flags[item] = 0u;   // sole reader of this flag
        continue;
      }
      if (active) backward_long_keys_v2<HD, TP>(a, (int)item, 16 * blk, X, Y, colsum, true);
      __syncthreads();
      if (any)
        for (int i = NW * HD + tid; i < 3 * NW * HD; i += NT) hp[i] = colsum[i];
      __threadfence();
      __syncthreads();
      if (tid == 0) slot[1] = atomicAdd(keys_done + item, 1u);
      __syncthreads();
      if (slot[1] == 1u) {                                     // second key half: both halves passed the flag
        __threadfence();
        if (any) {
          const float* p0 = half == 0 ? hp : hq;
          const float* p1 = half == 0 ? hq : hp;
          for (int i = NW * HD + tid; i < 3 * NW * HD; i += NT) {
            const int w = (i / HD) % NW;
            colsum[i] = w < H1 ? __fadd_rn(__ldcg(p0 + i), __ldcg(p1 + i)) : __ldcg(p0 + i);
          }
        }
        __syncthreads();
        reduce_item<HD, NW>(a, b, h, colsum, nullptr, NT, 0, 6);
        if (tid == 0) {
          keys_done[item] = 0u;
          flags[item] = 0u;
        }
      }
    }
  }
  if (tid == 0 && atomicAdd(leave, 1u) == gridDim.x - 1) {     // every CTA has drawn an exhausted ticket
    *claim = 0u;
    *leave = 0u;
  }
}

}  // namespace pmix
