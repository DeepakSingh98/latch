// P-OPS-C matrix bodies for the operation compiler (new code in this lane; not a vendored file).
//
// Operand staging (cp.async into padded natural-layout BF16 tiles) and fragment formation (ldmatrix / ldmatrix.trans,
// mma.sync m16n8k16) follow the pinned Latch body ops/vendor/bgemm.cuh (latch-v0.6.0, git blob 7399bbe1), whose
// helpers are reused unchanged. What changes is only the work decomposition:
//   * output tiles from 32x32 upward, so a 256-row product can occupy the whole resident grid;
//   * an NST-deep stage ring sized from the static K, so short K slices issue every load before the first wait;
//   * a register-direct BF16 store (no FP32 staging round trip through shared memory).
//
// Arithmetic contract (identical to s101h::bgemm_tile with ksplit 1, for any tile shape or stage depth): every output
// element owns one FP32 accumulator that starts at +0 and receives one mma.sync.m16n8k16 BF16 product per k16 block in
// ascending k; the accumulator is rounded once to BF16 with cvt.rn.bf16x2 (s101h::bpk). Hence bitwise equal results.
//
// The FP32 SIMT body (sgemm) has the contract of s101h::gemm_tile (ops/vendor/gemm.cuh): one fmaf chain per output
// element from +0, k ascending, FP32 result stored unrounded. Only its tile size and staging differ.
#pragma once
#include <type_traits>
#include <utility>
#include "bgemm.cuh"

namespace popsc {

using s101h::bf16;

DEV void bldsm2(uint32_t (&r)[2], const bf16* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
DEV void bldsm2t(uint32_t (&r)[2], const bf16* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

// C_b = A_b B_b with A_b(m, k) = A[b*bsa + m*sam + k*sak], B_b(k, n) = B[b*bsb + k*sbk + n*sbn] (one of each pair of
// strides is 1), C_b(m, n) = C[b*bsc + m*ldc + n]. Strides in elements.
struct BArgs {
  const bf16* A; long long sam, sak, bsa;
  const bf16* B; long long sbk, sbn, bsb;
  bf16* C; long long ldc, bsc;
  int M, N, K, nb;
};

template <int BM, int BN, bool AK, bool BKC, int KS>
struct BLay {
  static constexpr int A_EL = AK ? BM * (KS + 8) : KS * (BM + 8);
  static constexpr int B_EL = BKC ? BN * (KS + 8) : KS * (BN + 8);
  static constexpr int STAGE = A_EL + B_EL;                       // BF16 elements per stage
};

// R x KS tile (RK: k-contiguous rows [r][KS+8]) or KS x R tile (r-contiguous rows [k][R+8]) of 16-byte chunks
template <int R, bool RK, int KS>
DEV void bstage(const bf16* __restrict__ X, long long sr, long long sk, int r0, int k, bf16* __restrict__ S) {
  constexpr int CH = R * KS / 8;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      if (RK) {
        const int r = c / (KS / 8), kc = (c % (KS / 8)) * 8;
        s101h::cpa16(reinterpret_cast<float*>(S + r * (KS + 8) + kc),
                     reinterpret_cast<const float*>(X + (long long)(r0 + r) * sr + k + kc));
      } else {
        const int kr = c / (R / 8), rc = (c % (R / 8)) * 8;
        s101h::cpa16(reinterpret_cast<float*>(S + kr * (R + 8) + rc),
                     reinterpret_cast<const float*>(X + (long long)(k + kr) * sk + r0 + rc));
      }
    }
  }
}

// P-OPS5-B: optional same-index consumer of the rounded product (flat index m * ldc + n of a one-batch
// row-major result). STORE=false skips the product store when only the epilogue reads the product.
struct NoEpi {};
// P-OPS5-B: optional epilogue operand prefetch. pre(flat index) loads the epilogue's one outside operand for every
// fragment position before the K loop; epi then receives it as a third argument (same value, read earlier).
template <class P> struct PreValue { using type = decltype(std::declval<const P&>()(0LL)); };
template <> struct PreValue<NoEpi> { using type = char; };

// P6-C LATE: the epilogue's outside operands are loaded for every fragment position after the K loop (when the
// operand-fragment registers are dead) instead of before it; values and epilogue arithmetic are unchanged.
// P6-C PAIR: the epilogue is called once per adjacent column pair (i even, i + 1) with both rounded values (and
// both prefetched operands), so its stores can be 8-byte FP32 / 4-byte BF16 pairs; per-element expressions unchanged.
// P6-C shk: the K loop of btile/btile_se as one non-inlined body per tile configuration. Every product call site has
// its own epilogue lambda type, so an inlined K loop is a separate code copy per call site that runs cold once per
// level (instruction-fetch stalls); with POPSC_SHK the call sites of one configuration share this body. It returns the
// cvt.rn.bf16x2 pairs of exactly btile's accumulators (same staging, ring, mma order, rounding): bitwise identical.
template <int MT, int NT> struct BPk { uint32_t v[MT][NT][2]; };
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST>
__device__ __noinline__ BPk<BM / WM / 16, BN / WN / 8> bkloop(const bf16* __restrict__ A, long long sam, long long sak,
                                                            const bf16* __restrict__ B, long long sbk, long long sbn,
                                                            int K, int m0, int n0, bf16* S) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = K / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
    bstage<BM, AK, KS>(A, sam, sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
    bstage<BN, BKC, KS>(B, sbn, sbk, n0, kk, As + L::A_EL);
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();                                            // only empty groups can remain
  BPk<MT, NT> r;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      r.v[i][j][0] = s101h::bpk(acc[i][j][0], acc[i][j][1]);
      r.v[i][j][1] = s101h::bpk(acc[i][j][2], acc[i][j][3]);
    }
  return r;
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE = true, bool LATE = false,
          bool PAIR = false, class EPI = NoEpi, class PRE = NoEpi>
DEV void btile(const BArgs& g, int b, int m0, int n0, bf16* S, const EPI& epi = EPI{}, const PRE& pre = PRE{}) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
#ifndef POPSC_SHK
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = g.K / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
    bstage<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
    bstage<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#endif
  [[maybe_unused]] typename PreValue<PRE>::type pf[MT][NT][4];
  [[maybe_unused]] auto prefetch = [&](const auto& pre_) {                    // generic: instantiated only when called
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const long long mm = m0 + wm0 + 16 * i + gq, nn = n0 + wn0 + 8 * j + 2 * tq;
        const long long bo = (long long)b * g.bsc;   // P9-C: batch offset of the flat epilogue index (0 for batch 1)
        pf[i][j][0] = pre_(bo + mm * g.ldc + nn);
        pf[i][j][1] = pre_(bo + mm * g.ldc + nn + 1);
        pf[i][j][2] = pre_(bo + (mm + 8) * g.ldc + nn);
        pf[i][j][3] = pre_(bo + (mm + 8) * g.ldc + nn + 1);
      }
  };
  if constexpr (LATE) {
  } else if constexpr (!std::is_same<PRE, NoEpi>::value) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const long long mm = m0 + wm0 + 16 * i + gq, nn = n0 + wn0 + 8 * j + 2 * tq;
        const long long bo = (long long)b * g.bsc;   // P9-C: batch offset (0 for batch 1)
        pf[i][j][0] = pre(bo + mm * g.ldc + nn);
        pf[i][j][1] = pre(bo + mm * g.ldc + nn + 1);
        pf[i][j][2] = pre(bo + (mm + 8) * g.ldc + nn);
        pf[i][j][3] = pre(bo + (mm + 8) * g.ldc + nn + 1);
      }
  }
#ifdef POPSC_SHK
  const auto pk = bkloop<BM, BN, WM, WN, AK, BKC, KS, NST>(A, g.sam, g.sak, B, g.sbk, g.sbn, g.K, m0, n0, S);
#else
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();                                               // stage it visible; stage it-1 fully consumed
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();                                            // only empty groups can remain
#endif
  if constexpr (LATE && !std::is_same<PRE, NoEpi>::value) prefetch(pre);
  bf16* C = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
#ifdef POPSC_SHK
      const uint32_t p0 = pk.v[i][j][0], p1 = pk.v[i][j][1];
#else
      const uint32_t p0 = s101h::bpk(acc[i][j][0], acc[i][j][1]), p1 = s101h::bpk(acc[i][j][2], acc[i][j][3]);
#endif
      if constexpr (STORE) {
        *reinterpret_cast<uint32_t*>(C + (long long)m * g.ldc + n) = p0;
        *reinterpret_cast<uint32_t*>(C + (long long)(m + 8) * g.ldc + n) = p1;
      }
      if constexpr (!std::is_same<EPI, NoEpi>::value) {
        const long long bo = (long long)b * g.bsc;   // P9-C: flat epilogue index includes the batch offset
        const long long i0 = bo + (long long)m * g.ldc + n, i1 = bo + (long long)(m + 8) * g.ldc + n;
        if constexpr (PAIR && std::is_same<PRE, NoEpi>::value) {
          epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
          epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p1 >> 16)));
        } else if constexpr (PAIR) {
          epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p0 >> 16)),
              pf[i][j][0], pf[i][j][1]);
          epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p1 >> 16)),
              pf[i][j][2], pf[i][j][3]);
        } else if constexpr (std::is_same<PRE, NoEpi>::value) {
          epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)));
          epi(i0 + 1, __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
          epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)));
          epi(i1 + 1, __ushort_as_bfloat16((unsigned short)(p1 >> 16)));
        } else {
          epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)), pf[i][j][0]);
          epi(i0 + 1, __ushort_as_bfloat16((unsigned short)(p0 >> 16)), pf[i][j][1]);
          epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)), pf[i][j][2]);
          epi(i1 + 1, __ushort_as_bfloat16((unsigned short)(p1 >> 16)), pf[i][j][3]);
        }
      }
    }
  __syncthreads();                                                 // every fragment read done before smem reuse
}

// CTA slice [G*lo/den, G*hi/den) of the resident grid G (at least one CTA) that executes this product's tasks;
// (0, 1, 1) is the whole grid. Grouped phases give independent products disjoint slices so they run concurrently;
// rot rotates the first task within the slice. Scheduling only: every task's arithmetic is unchanged.
struct Slice { int first, count, me; };
DEV Slice slice_of(int lo, int hi, int den, int rot) {
  const int G = (int)gridDim.x;
  int b0 = (int)((long long)G * lo / den), b1 = (int)((long long)G * hi / den);
  if (b0 > G - 1) b0 = G - 1;
  if (b1 <= b0) b1 = b0 + 1;
  const int n = b1 - b0, local = (int)blockIdx.x - b0;
  return {b0, n, (local % n + n - rot % n) % n};
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE = true, bool LATE = false,
          bool PAIR = false, class EPI = NoEpi, class PRE = NoEpi>
__device__ __noinline__ void bmm_phase(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0,
                                       EPI epi = EPI{}, PRE pre = PRE{}) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = g.nb * tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile<BM, BN, WM, WN, AK, BKC, KS, NST, STORE, LATE, PAIR, EPI, PRE>(g, r / tm, (r % tm) * BM, j * BN, S, epi, pre);
  }
}

// P6-C staged epilogue (epism). The K loop is btile's; afterwards the rounded BF16 pairs (cvt.rn.bf16x2 of the
// same accumulators) are staged in shared memory and the epilogue runs as a compact loop over row-major column pairs:
// consecutive threads take consecutive pairs (coalesced 4-byte product stores and pair-wide epilogue stores), and the
// epilogue's outside operands of U pairs are loaded before any of their stores. Values and per-element expressions
// are those of btile with a pair epilogue (PAIR): bitwise identical results; only store layout/code size change.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, int U, class EPI, class PRE>
DEV void btile_se(const BArgs& g, int b, int m0, int n0, bf16* S, const EPI& epi, const PRE& pre) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  constexpr int PW = BN / 2 + 1;                                    // padded uint32 pairs per staged row
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2 && BM * PW * 4 <= NST * L::STAGE * 2, "staged tile reuses the stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
#ifdef POPSC_SHK
  const auto pk = bkloop<BM, BN, WM, WN, AK, BKC, KS, NST>(A, g.sam, g.sak, B, g.sbk, g.sbn, g.K, m0, n0, S);
  __syncthreads();
#else
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = g.K / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
    bstage<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
    bstage<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
  __syncthreads();                                                 // every ring read done before staging reuses it
#endif
  uint32_t* T = reinterpret_cast<uint32_t*>(S);
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = wm0 + 16 * i + gq, n2 = (wn0 + 8 * j + 2 * tq) >> 1;
#ifdef POPSC_SHK
      T[m * PW + n2] = pk.v[i][j][0];
      T[(m + 8) * PW + n2] = pk.v[i][j][1];
#else
      T[m * PW + n2] = s101h::bpk(acc[i][j][0], acc[i][j][1]);
      T[(m + 8) * PW + n2] = s101h::bpk(acc[i][j][2], acc[i][j][3]);
#endif
    }
  __syncthreads();
  bf16* C = g.C + (long long)b * g.bsc;
  constexpr int PAIRS = BM * BN / 2;
#pragma unroll 1
  for (int q0 = threadIdx.x; q0 < PAIRS; q0 += NTHREADS * U) {
    long long idx[U];
    uint32_t val[U];
    [[maybe_unused]] typename PreValue<PRE>::type pa[U], pb[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int q = q0 + u * NTHREADS < PAIRS ? q0 + u * NTHREADS : q0;
      const int m = q / (BN / 2), n2 = q % (BN / 2);
      idx[u] = (long long)(m0 + m) * g.ldc + n0 + 2 * n2;
      val[u] = T[m * PW + n2];
      if constexpr (!std::is_same<PRE, NoEpi>::value) {
        pa[u] = pre((long long)b * g.bsc + idx[u]);
        pb[u] = pre((long long)b * g.bsc + idx[u] + 1);
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if (q0 + u * NTHREADS >= PAIRS) continue;
      if constexpr (STORE) *reinterpret_cast<uint32_t*>(C + idx[u]) = val[u];
      if constexpr (!std::is_same<EPI, NoEpi>::value) {
        const bf16 lo = __ushort_as_bfloat16((unsigned short)(val[u] & 0xffffu));
        const bf16 hi = __ushort_as_bfloat16((unsigned short)(val[u] >> 16));
        if constexpr (std::is_same<PRE, NoEpi>::value) epi((long long)b * g.bsc + idx[u], lo, hi);
        else epi((long long)b * g.bsc + idx[u], lo, hi, pa[u], pb[u]);
      }
    }
  }
  __syncthreads();                                                 // staged reads done before smem reuse
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE = true, int U = 4,
          class EPI = NoEpi, class PRE = NoEpi>
__device__ __noinline__ void bmm_phase_se(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0,
                                          EPI epi = EPI{}, PRE pre = PRE{}) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = g.nb * tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile_se<BM, BN, WM, WN, AK, BKC, KS, NST, STORE, U, EPI, PRE>(g, r / tm, (r % tm) * BM, j * BN, S, epi, pre);
  }
}

// P6-C sliced-K tile. Latency regime (one output tile per CTA, small M/N): btile's warps each own a 16x8..32x16
// piece and walk every k16 block of K serially, so a K-step chain of mma.sync dominates. Here KG warp groups split
// each stage's k16 blocks (group g takes k16 indices congruent to g mod KG) while each group's WM x WN warps cover
// the whole BM x BN tile with wider warp tiles (more independent accumulators per k16). Every output element thus
// owns KG FP32 accumulators, each starting at +0 and receiving its k16 products in ascending k; after the K loop the
// partials meet in shared memory and are summed in fixed group order ((p0 + p1) + p2) + ..., then rounded once to
// BF16 (cvt.rn.bf16x2). Arithmetic change versus btile: FP32 reassociation of the K sum only (declared; not
// bitwise). Operand staging, stage ring and the epilogue contract (flat index, rounded BF16 value) are unchanged.
template <int BM, int BN, int WM, int WN, int KG, bool AK, bool BKC, int KS, int NST, bool STORE = true,
          class EPI = NoEpi>
DEV void btile_kg(const BArgs& g, int b, int m0, int n0, bf16* S, const EPI& epi = EPI{}) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN * KG == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert((KS / 16) % KG == 0 && NST >= 2, "k groups / stage ring");
  constexpr int PN = BN + 4;                                       // padded FP32 partial row
  static_assert(KG * BM * PN * 4 <= NST * L::STAGE * 2, "partials reuse the stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int kg = w / (WM * WN), sw = w % (WM * WN);
  const int wm0 = (sw / WN) * WTM, wn0 = (sw % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = g.K / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
    bstage<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
    bstage<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int kb = 0; kb < KS / 16 / KG; ++kb) {
      const int k16 = (kb * KG + kg) * 16;
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
  __syncthreads();                                                 // every stage read before partials reuse it
  float* P = reinterpret_cast<float*>(S);
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = wm0 + 16 * i + gq, n = wn0 + 8 * j + 2 * tq;
      float* p = P + (kg * BM + m) * PN + n;
      p[0] = acc[i][j][0];
      p[1] = acc[i][j][1];
      p[8 * PN] = acc[i][j][2];
      p[8 * PN + 1] = acc[i][j][3];
    }
  __syncthreads();
  bf16* C = g.C + (long long)b * g.bsc;
  for (int e = 2 * threadIdx.x; e < BM * BN; e += 2 * NTHREADS) {
    const int m = e / BN, n = e % BN;
    float v0 = P[m * PN + n], v1 = P[m * PN + n + 1];
#pragma unroll
    for (int q = 1; q < KG; ++q) {
      v0 = v0 + P[(q * BM + m) * PN + n];
      v1 = v1 + P[(q * BM + m) * PN + n + 1];
    }
    const uint32_t p0 = s101h::bpk(v0, v1);
    const long long i0 = (long long)(m0 + m) * g.ldc + n0 + n;
    if constexpr (STORE) *reinterpret_cast<uint32_t*>(C + i0) = p0;
    if constexpr (!std::is_same<EPI, NoEpi>::value) {
      epi((long long)b * g.bsc + i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)));
      epi((long long)b * g.bsc + i0 + 1, __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
    }
  }
  __syncthreads();                                                 // partial reads done before smem reuse
}

template <int BM, int BN, int WM, int WN, int KG, bool AK, bool BKC, int KS, int NST, bool STORE = true,
          class EPI = NoEpi>
__device__ __noinline__ void bmm_phase_kg(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0,
                                          EPI epi = EPI{}) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = g.nb * tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile_kg<BM, BN, WM, WN, KG, AK, BKC, KS, NST, STORE, EPI>(g, r / tm, (r % tm) * BM, j * BN, S, epi);
  }
}

// P-OPS5-B row-prologue product: the task's A rows (BM x K, k-contiguous) are produced directly in shared memory
// by a generated row prologue pro(m0, As) -- the producer's per-row operations replayed for these rows (same
// expressions, same reduction tree) with only the operand value kept -- instead of being read from a stored
// operand written one dependency level earlier. B streams through the NST-deep cp.async ring, issued before the
// prologue so its latency overlaps the row work. MMA order, accumulation and rounding equal btile's.
template <int BM, int BN, int WM, int WN, bool BKC, int KS, int NST, int K, bool STORE, class PRO, class EPI>
DEV void btile_rp(const BArgs& g, int m0, int n0, bf16* S, const PRO& pro, const EPI& epi) {
  constexpr int AS = K + 8;                                       // padded resident row: conflict-free ldmatrix
  constexpr int B_EL = BKC ? BN * (KS + 8) : KS * (BN + 8);
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2 && K % KS == 0 && K % 8 == 0, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  bf16* As = S;
  bf16* Bs0 = S + BM * AS;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  constexpr int nk = K / KS;
  auto load = [&](int kk, int st) { bstage<BN, BKC, KS>(g.B, g.sbn, g.sbk, n0, kk, Bs0 + st * B_EL); };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
  pro((long long)m0, As);                                          // visible after the first loop barrier
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* Bs = Bs0 + (it % NST) * B_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i)
        s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * AS + it * KS + k16 + (lane >> 4) * 8);
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
      const uint32_t p0 = s101h::bpk(acc[i][j][0], acc[i][j][1]), p1 = s101h::bpk(acc[i][j][2], acc[i][j][3]);
      if constexpr (STORE) {
        *reinterpret_cast<uint32_t*>(g.C + (long long)m * g.ldc + n) = p0;
        *reinterpret_cast<uint32_t*>(g.C + (long long)(m + 8) * g.ldc + n) = p1;
      }
      if constexpr (!std::is_same<EPI, NoEpi>::value) {
        const long long i0 = (long long)m * g.ldc + n, i1 = (long long)(m + 8) * g.ldc + n;
        epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)));
        epi(i0 + 1, __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
        epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)));
        epi(i1 + 1, __ushort_as_bfloat16((unsigned short)(p1 >> 16)));
      }
    }
  __syncthreads();                                                 // fragment reads done before smem reuse
}

// One-batch row-major products only (the generator checks M, K, layouts and the producer's row width).
template <int BM, int BN, int WM, int WN, bool BKC, int KS, int NST, int K, bool STORE, class PRO, class EPI>
__device__ __noinline__ void bmm_phase_rp(const BArgs g, float* smem, int lo, int hi, int den, int rot,
                                          PRO pro, EPI epi) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile_rp<BM, BN, WM, WN, BKC, KS, NST, K, STORE, PRO, EPI>(g, r * BM, j * BN, S, pro, epi);
  }
}

// P-OPS5-B row-epilogue product: a task owns BM complete rows (BN = N). After the K loop the rounded BF16 tile is
// kept in shared memory (sC, BM x BN, at byte 1024) and the consuming row group runs on it for those rows (its
// unchanged per-row operations and exact reduction tree), instead of a separate grid level re-reading the product.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, class ROW>
__device__ __noinline__ void bmm_phase_re(const BArgs g, float* smem, int lo, int hi, int den, int rot, ROW row) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  // P9-C: batched products (contiguous [nb, M, N] result): task -> (batch b, row block); the row group sees the
  // flat row b * M + m0.
  const int mt = g.M / BM, ntask = g.nb * mt;
  bf16* S = reinterpret_cast<bf16*>(smem);
  bf16* sC = reinterpret_cast<bf16*>(reinterpret_cast<char*>(smem) + 1024);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  for (int task = s.me; task < ntask; task += s.count) {
    const int b = task / mt, m0 = (task % mt) * BM;
    const bf16* Ab = g.A + (long long)b * g.bsa;
    const bf16* Bb = g.B + (long long)b * g.bsb;
    bf16* Cb = g.C + (long long)b * g.bsc;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    const int nk = g.K / KS;
    auto load = [&](int kk, int st) {
      bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
      bstage<BM, AK, KS>(Ab, g.sam, g.sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
      bstage<BN, BKC, KS>(Bb, g.sbn, g.sbk, 0, kk, As + L::A_EL);
#endif
    };
#pragma unroll
    for (int st = 0; st < NST - 1; ++st) {
      if (st < nk) load(st * KS, st);
      s101h::cpa_commit();
    }
#pragma unroll 1
    for (int it = 0; it < nk; ++it) {
      s101h::cpa_wait<NST - 2>();
      __syncthreads();
      if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
      s101h::cpa_commit();
      const bf16* As = S + (it % NST) * L::STAGE;
      const bf16* Bs = As + L::A_EL;
#pragma unroll
      for (int k16 = 0; k16 < KS; k16 += 16) {
        uint32_t af[MT][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
          else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                         (((lane >> 3) & 1) << 3));
        }
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
    s101h::cpa_wait<0>();
    __syncthreads();                                               // ring reads done: sC may overlap it
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int m = wm0 + 16 * i + gq, n = wn0 + 8 * j + 2 * tq;
        const uint32_t p0 = s101h::bpk(acc[i][j][0], acc[i][j][1]), p1 = s101h::bpk(acc[i][j][2], acc[i][j][3]);
        *reinterpret_cast<uint32_t*>(sC + m * BN + n) = p0;
        *reinterpret_cast<uint32_t*>(sC + (m + 8) * BN + n) = p1;
        if constexpr (STORE) {
          *reinterpret_cast<uint32_t*>(Cb + (long long)(m0 + m) * g.ldc + n) = p0;
          *reinterpret_cast<uint32_t*>(Cb + (long long)(m0 + m + 8) * g.ldc + n) = p1;
        }
      }
    __syncthreads();
    row((long long)b * g.M + m0, (const bf16*)sC);
    __syncthreads();
  }
}

// P-OPS5-B deterministic split-K: task t covers tile t % tiles and K slice t / tiles. Each slice writes its FP32
// accumulators (fragment order) to part; the last-arriving slice of a tile (counter in the persistent area, reset to
// zero by it) sums the slices in ascending slice order, rounds once to BF16 and runs the store/epilogue. Only the FP32
// association of the K sum changes (slice partials instead of one chain); the rounding point is unchanged.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, int SPLIT, bool STORE = true,
          class EPI = NoEpi>
__device__ __noinline__ void bmm_phase_sk(const BArgs g, float* smem, int lo, int hi, int den, int rot,
                                          float* __restrict__ part, unsigned* cnt, EPI epi = EPI{}) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8, FR = MT * NT * 4;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  __shared__ unsigned latch_sk_last;
  const int tm = g.M / BM, tn = g.N / BN, tiles = tm * tn, ntask = tiles * SPLIT;
  const int kslice = g.K / SPLIT;
  bf16* S = reinterpret_cast<bf16*>(smem);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  for (int task = s.me; task < ntask; task += s.count) {
    const int tile = task % tiles, slice = task / tiles;
    const int m0 = (tile / tn) * BM, n0 = (tile % tn) * BN, k0 = slice * kslice;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    const int nk = kslice / KS;
    auto load = [&](int kk, int st) {
      bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
      bstage<BM, AK, KS>(g.A, g.sam, g.sak, m0, k0 + kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
      bstage<BN, BKC, KS>(g.B, g.sbn, g.sbk, n0, k0 + kk, As + L::A_EL);
#endif
    };
#pragma unroll
    for (int st = 0; st < NST - 1; ++st) {
      if (st < nk) load(st * KS, st);
      s101h::cpa_commit();
    }
#pragma unroll 1
    for (int it = 0; it < nk; ++it) {
      s101h::cpa_wait<NST - 2>();
      __syncthreads();
      if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
      s101h::cpa_commit();
      const bf16* As = S + (it % NST) * L::STAGE;
      const bf16* Bs = As + L::A_EL;
#pragma unroll
      for (int k16 = 0; k16 < KS; k16 += 16) {
        uint32_t af[MT][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
          else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                         (((lane >> 3) & 1) << 3));
        }
        if constexpr (NT == 1) {
          uint32_t bb[2];
          if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
          for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
        } else {
#pragma unroll
          for (int j = 0; j < NT; j += 2) {
            uint32_t bb[4];
            if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
            else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
              s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
            }
          }
        }
      }
    }
    s101h::cpa_wait<0>();
    float* mine = part + ((long long)tile * SPLIT + slice) * (BM * BN) + threadIdx.x * FR;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) mine[(i * NT + j) * 4 + q] = acc[i][j][q];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned prev = atomicAdd(cnt + tile, 1u);
      latch_sk_last = prev == (unsigned)(SPLIT - 1);
      if (latch_sk_last) cnt[tile] = 0u;                           // restored for the next use in this launch
    }
    __syncthreads();
    if (latch_sk_last) {
      __threadfence();
      const float* base = part + (long long)tile * SPLIT * (BM * BN) + threadIdx.x * FR;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float sum = __ldcg(base + (i * NT + j) * 4 + q);
#pragma unroll
            for (int sl = 1; sl < SPLIT; ++sl) sum += __ldcg(base + (long long)sl * (BM * BN) + (i * NT + j) * 4 + q);
            acc[i][j][q] = sum;
          }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
          const uint32_t p0 = s101h::bpk(acc[i][j][0], acc[i][j][1]), p1 = s101h::bpk(acc[i][j][2], acc[i][j][3]);
          if constexpr (STORE) {
            *reinterpret_cast<uint32_t*>(g.C + (long long)m * g.ldc + n) = p0;
            *reinterpret_cast<uint32_t*>(g.C + (long long)(m + 8) * g.ldc + n) = p1;
          }
          if constexpr (std::is_invocable<EPI, long long, __nv_bfloat16, __nv_bfloat16>::value) {
            // P8-C: pair-form generated epilogues (two adjacent columns per call), as the unsplit body.
            const long long i0 = (long long)m * g.ldc + n, i1 = (long long)(m + 8) * g.ldc + n;
            epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
            epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)), __ushort_as_bfloat16((unsigned short)(p1 >> 16)));
          } else if constexpr (!std::is_same<EPI, NoEpi>::value) {
            const long long i0 = (long long)m * g.ldc + n, i1 = (long long)(m + 8) * g.ldc + n;
            epi(i0, __ushort_as_bfloat16((unsigned short)(p0 & 0xffffu)));
            epi(i0 + 1, __ushort_as_bfloat16((unsigned short)(p0 >> 16)));
            epi(i1, __ushort_as_bfloat16((unsigned short)(p1 & 0xffffu)));
            epi(i1 + 1, __ushort_as_bfloat16((unsigned short)(p1 >> 16)));
          }
        }
    }
    __syncthreads();
  }
}

// P-OPS5-B cluster row product: N = 8 * BN; the eight column tiles of a BM-row block run on the eight CTAs of one
// thread-block cluster (column tile = cluster rank). After the K loop each CTA keeps its rounded BF16 tile in shared
// memory (sC at byte 1024; row partials at byte 1024 + 2*BM*BN) and runs the generated cluster row group, which
// exchanges per-row partial sums through distributed shared memory. Requires a launch with cluster dimension 8.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, class ROW>
__device__ __noinline__ void bmm_phase_cl(const BArgs g, float* smem, int rot, ROW row) {
  constexpr int CL = 8;
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  const int nclusters = (int)gridDim.x / CL, c = (int)blockIdx.x / CL, rank = (int)blockIdx.x % CL;
  if (c >= nclusters) return;
  const int tm = g.M / BM, rc = ((rot / CL) % nclusters + nclusters) % nclusters;
  const int me = (c - rc + nclusters) % nclusters;
  bf16* S = reinterpret_cast<bf16*>(smem);
  bf16* sC = reinterpret_cast<bf16*>(reinterpret_cast<char*>(smem) + 1024);
  float* part = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + 1024 + 2 * BM * BN);
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  for (int rb = me; rb < tm; rb += nclusters) {
    const int m0 = rb * BM, n0 = rank * BN;
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    const int nk = g.K / KS;
    auto load = [&](int kk, int st) {
      bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
      bstage<BM, AK, KS>(g.A, g.sam, g.sak, m0, kk, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
      bstage<BN, BKC, KS>(g.B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
#endif
    };
#pragma unroll
    for (int st = 0; st < NST - 1; ++st) {
      if (st < nk) load(st * KS, st);
      s101h::cpa_commit();
    }
#pragma unroll 1
    for (int it = 0; it < nk; ++it) {
      s101h::cpa_wait<NST - 2>();
      __syncthreads();
      if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
      s101h::cpa_commit();
      const bf16* As = S + (it % NST) * L::STAGE;
      const bf16* Bs = As + L::A_EL;
#pragma unroll
      for (int k16 = 0; k16 < KS; k16 += 16) {
        uint32_t af[MT][4];
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
          else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                         (((lane >> 3) & 1) << 3));
        }
        if constexpr (NT == 1) {
          uint32_t bb[2];
          if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
          for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
        } else {
#pragma unroll
          for (int j = 0; j < NT; j += 2) {
            uint32_t bb[4];
            if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
            else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
            for (int i = 0; i < MT; ++i) {
              s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
              s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
            }
          }
        }
      }
    }
    s101h::cpa_wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int m = wm0 + 16 * i + gq, n = wn0 + 8 * j + 2 * tq;
        const uint32_t p0 = s101h::bpk(acc[i][j][0], acc[i][j][1]), p1 = s101h::bpk(acc[i][j][2], acc[i][j][3]);
        *reinterpret_cast<uint32_t*>(sC + m * BN + n) = p0;
        *reinterpret_cast<uint32_t*>(sC + (m + 8) * BN + n) = p1;
        if constexpr (STORE) {
          *reinterpret_cast<uint32_t*>(g.C + (long long)(m0 + m) * g.ldc + n0 + n) = p0;
          *reinterpret_cast<uint32_t*>(g.C + (long long)(m0 + m + 8) * g.ldc + n0 + n) = p1;
        }
      }
    __syncthreads();
    row((long long)m0, (const bf16*)sC, part);                     // contains the cluster barriers
    __syncthreads();
  }
}

// ------------------------------------------------------------------ FP32 SIMT small-tile body (batched products)
struct SArgs {
  const float* A; long long sam, sak, bsa;
  const float* B; long long sbk, sbn, bsb;
  float* C; long long ldc, bsc;
  int M, N, K, nb;
};

template <int R, bool RK, int KC>
DEV void sstage(const float* __restrict__ X, long long sr, long long sk, int r0, int k, float* __restrict__ S) {
  constexpr int CH = R * KC / 4;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      if (RK) {
        const int r = c / (KC / 4), kc = (c % (KC / 4)) * 4;
        s101h::cpa16(S + r * (KC + 4) + kc, X + (long long)(r0 + r) * sr + k + kc);
      } else {
        const int kr = c / (R / 4), rc = (c % (R / 4)) * 4;
        s101h::cpa16(S + kr * (R + 4) + rc, X + (long long)(k + kr) * sk + r0 + rc);
      }
    }
  }
}

// 16 x 16 threads; thread (ty, tx) owns rows ty + 16 i and columns tx + 16 j of a BM x BN tile.
template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
DEV void stile(const SArgs& g, int b, int m0, int n0, float* S) {
  constexpr int TM = BM / 16, TN = BN / 16;
  constexpr int A_EL = AK ? BM * (KC + 4) : KC * (BM + 4);
  constexpr int B_EL = BKC ? BN * (KC + 4) : KC * (BN + 4);
  constexpr int STAGE = A_EL + B_EL;
  static_assert(BM % 16 == 0 && BN % 16 == 0 && KC % 4 == 0 && NST >= 2, "sgemm tile");
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  const float* A = g.A + (long long)b * g.bsa;
  const float* B = g.B + (long long)b * g.bsb;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  const int nk = g.K / KC;
  auto load = [&](int kk, int st) {
    float* As = S + st * STAGE;
    sstage<BM, AK, KC>(A, g.sam, g.sak, m0, kk, As);
    sstage<BN, BKC, KC>(B, g.sbn, g.sbk, n0, kk, As + A_EL);
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KC, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KC, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const float* As = S + (it % NST) * STAGE;
    const float* Bs = As + A_EL;
#pragma unroll 8
    for (int kk = 0; kk < KC; ++kk) {
      float a[TM], bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = AK ? As[(ty + 16 * i) * (KC + 4) + kk] : As[kk * (BM + 4) + ty + 16 * i];
#pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = BKC ? Bs[(tx + 16 * j) * (KC + 4) + kk] : Bs[kk * (BN + 4) + tx + 16 * j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
    }
  }
  s101h::cpa_wait<0>();
  float* C = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) C[(long long)(m0 + ty + 16 * i) * g.ldc + n0 + tx + 16 * j] = acc[i][j];
  __syncthreads();
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void smm_phase(const SArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = g.nb * tm * tn;
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    stile<BM, BN, AK, BKC, KC, NST>(g, r / tm, (r % tm) * BM, j * BN, smem);
  }
}

// C-OPS4-B diagnostics (never defined by the compiler's normal lowering): POPSC_DIAG_SKIP_A / POPSC_DIAG_SKIP_B
// omit the A / B operand stage loads of the BF16 tile bodies. Results are meaningless; the timing bounds what
// operand delivery costs inside product phases (e.g. what retaining loop-invariant weights could save).
// C-OPS4-B: BF16 product with M/N/K tails (each a multiple of 16; 16-byte aligned operand rows). Operand chunks
// outside [0, M) x [0, K) or [0, K) x [0, N) are zero-filled in the stage instead of loaded, the K loop runs
// ceil(K / KS) steps and out-of-range outputs are not stored. A zero-padded k16 block adds +0 to every in-range
// accumulator, so in-range results equal btile's for the same tile shape. Separate entry point: the aligned
// bmm_phase/btile and their P-OPS5-B/P6-C rewrites are unchanged.
template <int R, bool RK, int KS>
DEV void bstage_tail(const bf16* __restrict__ X, long long sr, long long sk, int r0, int k, int rlim, int klim,
                     bf16* __restrict__ S) {
  constexpr int CH = R * KS / 8;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      if (RK) {
        const int r = c / (KS / 8), kc = (c % (KS / 8)) * 8;
        bf16* dst = S + r * (KS + 8) + kc;
        if (r0 + r < rlim && k + kc < klim)
          s101h::cpa16(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(X + (long long)(r0 + r) * sr + k + kc));
        else
          *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
      } else {
        const int kr = c / (R / 8), rc = (c % (R / 8)) * 8;
        bf16* dst = S + kr * (R + 8) + rc;
        if (k + kr < klim && r0 + rc < rlim)
          s101h::cpa16(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(X + (long long)(k + kr) * sk + r0 + rc));
        else
          *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  }
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST>
DEV void btile_tail(const BArgs& g, int b, int m0, int n0, bf16* S) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = (g.K + KS - 1) / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
#ifndef POPSC_DIAG_SKIP_A
    bstage_tail<BM, AK, KS>(A, g.sam, g.sak, m0, kk, g.M, g.K, As);
#endif
#ifndef POPSC_DIAG_SKIP_B
    bstage_tail<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, g.N, g.K, As + L::A_EL);
#endif
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
  __syncthreads();                                                 // stages reusable by the next tile
  bf16* C = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
      if (n < g.N) {
        if (m < g.M) *reinterpret_cast<uint32_t*>(C + (long long)m * g.ldc + n) = s101h::bpk(acc[i][j][0], acc[i][j][1]);
        if (m + 8 < g.M) *reinterpret_cast<uint32_t*>(C + (long long)(m + 8) * g.ldc + n) = s101h::bpk(acc[i][j][2], acc[i][j][3]);
      }
    }
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST>
__device__ __noinline__ void bmm_tail(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = g.nb * tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile_tail<BM, BN, WM, WN, AK, BKC, KS, NST>(g, r / tm, (r % tm) * BM, j * BN, S);
  }
}

// P7-B/P6-E masked-edge and FP32 split result bodies, unchanged pinned arithmetic.
// P-OPS4-B masked edges: cp.async with a source size (0..16 bytes) zero-fills the rest of the 16-byte chunk and
// reads no global memory beyond it. Used only for tiles that cross the M, N or K extent.
DEV void cpa16z(bf16* dst, const bf16* src, int bytes) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(bytes) : "memory");
}

// bstage with extents: operand element (r, kk) exists for r0 + r < rlim and k + kk < klim; absent elements stage as
// +0. Chunks along the contiguous dimension are partially filled at element granularity, so any extent is admitted;
// the leading stride/offset alignment (16 bytes) of every chunk start is the caller's existing layout contract.
template <int R, bool RK, int KS>
DEV void bstage_edge(const bf16* __restrict__ X, long long sr, long long sk, int r0, int k, bf16* __restrict__ S,
                     int rlim, int klim) {
  constexpr int CH = R * KS / 8;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      if (RK) {
        const int r = c / (KS / 8), kc = (c % (KS / 8)) * 8;
        const int left = klim - (k + kc), e = (r0 + r < rlim) ? (left < 0 ? 0 : (left > 8 ? 8 : left)) : 0;
        cpa16z(S + r * (KS + 8) + kc, e ? X + (long long)(r0 + r) * sr + k + kc : X, 2 * e);
      } else {
        const int kr = c / (R / 8), rc = (c % (R / 8)) * 8;
        const int left = rlim - (r0 + rc), e = (k + kr < klim) ? (left < 0 ? 0 : (left > 8 ? 8 : left)) : 0;
        cpa16z(S + kr * (R + 8) + rc, e ? X + (long long)(k + kr) * sk + r0 + rc : X, 2 * e);
      }
    }
  }
}

// EDGE=false is the unchanged aligned body. EDGE=true covers ceil(K/KS) stages with zero-filled absent operand
// elements and stores only existing outputs (PAIR: N even, so a bf16x2 store at even n is in range and aligned).
// Per element: one FP32 accumulator from +0, one mma.sync.m16n8k16 per k16 block in ascending k (absent k terms are
// exact +0 products), one cvt.rn rounding: identical to the aligned body applied to zero-padded operands.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool EDGE = false, bool PAIR = true,
          bool F32OUT = false>
DEV void btile_edge(const BArgs& g, int b, int m0, int n0, bf16* S) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const int nk = EDGE ? (g.K + KS - 1) / KS : g.K / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
    if constexpr (EDGE) {
      bstage_edge<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As, g.M, g.K);
      bstage_edge<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL, g.N, g.K);
    } else {
      bstage<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As);
      bstage<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
    }
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();                                               // stage it visible; stage it-1 fully consumed
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();                                            // only empty groups can remain
  bf16* C = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
      if constexpr (EDGE) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = m + 8 * h;
          if (mm >= g.M || n >= g.N) continue;
          const uint32_t v = s101h::bpk(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          if (PAIR) {
            *reinterpret_cast<uint32_t*>(C + (long long)mm * g.ldc + n) = v;
          } else {
            unsigned short* c = reinterpret_cast<unsigned short*>(C + (long long)mm * g.ldc + n);
            c[0] = (unsigned short)(v & 0xffffu);
            if (n + 1 < g.N) c[1] = (unsigned short)(v >> 16);
          }
        }
      } else if constexpr (F32OUT) {
        // P6-B split-K slice: the FP32 accumulators themselves (C points at FP32 storage, batch stride in floats).
        float* Cf = reinterpret_cast<float*>(g.C) + (long long)b * g.bsc;
        *reinterpret_cast<float2*>(Cf + (long long)m * g.ldc + n) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(Cf + (long long)(m + 8) * g.ldc + n) = make_float2(acc[i][j][2], acc[i][j][3]);
      } else {
        *reinterpret_cast<uint32_t*>(C + (long long)m * g.ldc + n) = s101h::bpk(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<uint32_t*>(C + (long long)(m + 8) * g.ldc + n) = s101h::bpk(acc[i][j][2], acc[i][j][3]);
      }
    }
  __syncthreads();                                                 // every fragment read done before smem reuse
}

// CTA slice [G*lo/den, G*hi/den) of the resident grid G (at least one CTA) that executes this product's tasks;
// (0, 1, 1) is the whole grid. Grouped phases give independent products disjoint slices so they run concurrently;
// rot rotates the first task within the slice. Scheduling only: every task's arithmetic is unchanged.

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST>
__device__ __noinline__ void bmm_split_phase(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1,
                                             int rot = 0) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = g.nb * tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile_edge<BM, BN, WM, WN, AK, BKC, KS, NST, false, true, true>(g, r / tm, (r % tm) * BM, j * BN, S);
  }
}

// Masked-edge variant: ceil(M/BM) x ceil(N/BN) tiles; with an even N only tiles that cross an extent take the masked
// path, so interior tiles run the unchanged aligned staging/store code.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool PAIR>
__device__ __noinline__ void bmm_edge_phase(const BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1,
                                            int rot = 0) {
  const Slice s = slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = g.nb * tm * tn;
  const bool kfull = g.K % KS == 0;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn, m0 = (r % tm) * BM, n0 = j * BN;
    if (PAIR && kfull && m0 + BM <= g.M && n0 + BN <= g.N)       // odd N: every store is scalar (odd row stride)
      btile_edge<BM, BN, WM, WN, AK, BKC, KS, NST>(g, r / tm, m0, n0, S);
    else
      btile_edge<BM, BN, WM, WN, AK, BKC, KS, NST, true, PAIR>(g, r / tm, m0, n0, S);
  }
}

// ------------------------------------------------------------------ FP32 SIMT small-tile body (batched products)

}  // namespace popsc
