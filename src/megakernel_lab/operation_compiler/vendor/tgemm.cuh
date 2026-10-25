// S103F option t3: GEMM body on tensor cores with 3xTF32 arithmetic (numerical policy P-3T, PLAN.md).
//
// C_b[m, n] = sum over pairs p, k of A_p,b(m, k) B_p,b(k, n): same GemmArgs / Pair / epilogue interface and the same
// split-K partial layout and last-arriver reduction as gemm.cuh. Each FP32 operand x is split x = hi + lo with
// hi = tf32_rna(x), lo = tf32_rna(x - hi); a product is a_lo b_hi + a_hi b_lo + a_hi b_hi (mma.sync m16n8k8 tf32, FP32
// accumulation, small terms first). Fixed per-element order (k ascending in steps of 8 through one accumulator), so a
// launch is deterministic. Operands are copied with cp.async into NST-stage shared buffers in their natural layout
// (k-contiguous rows [r][KS+4] or k-major rows [k][R+8]); every fragment load is bank-conflict free.
#pragma once
#include "gemm.cuh"

namespace s101h {

// round to the nearest TF32 value, ties away from zero (= cvt.rna.tf32.f32 for every finite input): add half a TF32
// unit to the magnitude bits and clear the 13 low bits (two integer ops; cvt.rna expands to a NaN-checking sequence)
DEV uint32_t tf32_rna(float x) { return (__float_as_uint(x) + 0x1000u) & 0xffffe000u; }
#if defined(S103F_SPLIT3)
// option t3s: hi rounded to nearest; lo = x - hi passed as FP32 bits (the MMA reads the TF32 part, i.e. lo truncated
// toward zero, |error| <= 2^-11 |lo| <= 2^-22 |x|), three ops per element
DEV void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
  hi = tf32_rna(x);
  lo = __float_as_uint(x - __uint_as_float(hi));
}
#else
DEV void split_tf32(float x, uint32_t& hi, uint32_t& lo) {
  hi = tf32_rna(x);
  lo = tf32_rna(x - __uint_as_float(hi));                          // x - hi is exact in FP32
}
#endif
DEV void mma_tf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// d = a b (+0): the first product of a promotion chunk takes a zero C operand (no register zeroing)
DEV void mma_tf32_z(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
}
// one k8 promotion chunk of 3xTF32: d = lo.hi + hi.lo + hi.hi from zero (tensor-core accumulation), caller adds d to its
// FP32 running sum with one round-to-nearest FADD per element (options pk8 / pa8)
DEV void mma3z(float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0, uint32_t bh1, uint32_t bl0,
               uint32_t bl1) {
#if !defined(S103F_FX_1T)
  mma_tf32_z(d, al[0], al[1], al[2], al[3], bh0, bh1);
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
#else
  mma_tf32_z(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
#endif
}
DEV void mma3c(float (&d)[4], const uint32_t (&ah)[4], const uint32_t (&al)[4], uint32_t bh0, uint32_t bh1, uint32_t bl0,
               uint32_t bl1) {                                      // continue a promotion chunk (accumulate into d)
#if !defined(S103F_FX_1T)
  mma_tf32(d, al[0], al[1], al[2], al[3], bh0, bh1);
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bl0, bl1);
#endif
  mma_tf32(d, ah[0], ah[1], ah[2], ah[3], bh0, bh1);
}
DEV void add4(float (&acc)[4], const float (&d)[4]) { acc[0] += d[0]; acc[1] += d[1]; acc[2] += d[2]; acc[3] += d[3]; }
DEV void cpa16(float* dst, const float* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src) : "memory");
}
DEV void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
DEV void ldsm_x4(uint32_t (&r)[4], const float* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
template <int N>
DEV void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// S105A policy B16 (option bg): one BF16 product per k16 (mma.sync m16n8k16, FP32 accumulation), operands rounded to
// nearest BF16 from the FP32 staged tiles when the fragment is formed (standard mixed precision; no split products)
DEV uint32_t bpk(float lo, float hi) {         // lo -> bits 0..15 (lower k index), hi -> bits 16..31
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
DEV void bmma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// option epf: the epilogue prefetches (to L1) the global inputs its functor reads, two iterations ahead (scheduling
// only; the epilogue's arithmetic is unchanged)
DEV void pf1(const void* p) { asm volatile("prefetch.global.L1 [%0];" ::"l"(p)); }
template <class E> DEV void epi_pre(const E&, int, int, int) {}
DEV void epi_pre(const EpiResid& e, int, int m, int n) { pf1(e.X + (long long)m * e.ld + n); }
DEV void epi_pre(const EpiDSilu& e, int, int m, int n) { pf1(e.HP + (long long)m * e.ld + n); }
DEV void epi_pre(const EpiQKV& e, int, int m, int n) {
  if (n / e.d < 2) {
    const int j = (n % e.d) % e.dh, s = m % e.S;
    pf1(e.cosT + s * (e.dh / 2) + (j >> 1));
    pf1(e.sinT + s * (e.dh / 2) + (j >> 1));
  }
}
DEV void epi_pre(const EpiChainR& e, int b, int m, int n) {
  pf1(e.U + e.R.row(b, m) * e.ld + n);
  if (e.MS) pf1(e.MS + e.R.row(b, m) * e.ld + n);
}
DEV void epi_pre(const EpiDaf& e, int b, int m, int n) { pf1(e.SP + e.R.row(b, m) * e.ld + n); }
DEV void epi_pre(const EpiZt& e, int b, int m, int n) { pf1(e.SP + e.R.row(b, m) * e.ld + n); }
DEV void epi_pre(const EpiWrite& e, int b, int m, int n) { pf1(e.Wc + (long long)b * e.bsw + (long long)m * e.ld + n); }

#if defined(S105A_BG)
constexpr int T3P = 8;                                              // k-contiguous row pad: conflict-free float2 reads
#else
constexpr int T3P = 4;
#endif
template <int BM, int BN, bool AK, bool BKC, int KS>
struct TSmem {
  static constexpr int A_FL = AK ? BM * (KS + T3P) : KS * (BM + 8);
  static constexpr int B_FL = BKC ? BN * (KS + T3P) : KS * (BN + 8);
  static constexpr int STAGE = A_FL + B_FL;
#if defined(S105A_ST4)
  static constexpr int NST = 4;                                     // S105A option st4: four-stage cp.async ring
#else
  static constexpr int NST = 3;
#endif
  static constexpr int CS = BN + 8;                                 // staged result row stride
#if defined(S103F_PS)
  static constexpr int MAIN = NST * STAGE + 2 * STAGE;               // option ps: + hi / lo planes of one stage
#else
  static constexpr int MAIN = NST * STAGE;
#endif
  static constexpr int FLOATS = (MAIN > BM * CS) ? MAIN : BM * CS;
};

// one operand tile (R rows of the output dimension x KS of k) -> shared, natural layout
template <int R, bool RK, int KS>
DEV void t3_load(const float* __restrict__ X, long long sr, long long sk, int r0, int k, float* __restrict__ S) {
  constexpr int CH = R * KS / 4;
  static_assert(CH % NTHREADS == 0, "t3 tile / k step");
#pragma unroll
  for (int i = 0; i < CH / NTHREADS; ++i) {
    const int c = threadIdx.x + i * NTHREADS;
    if (RK) {
      const int r = c / (KS / 4), kc = (c % (KS / 4)) * 4;
      cpa16(S + r * (KS + T3P) + kc, X + (long long)(r0 + r) * sr + k + kc);
    } else {
      const int kr = c / (R / 4), rc = (c % (R / 4)) * 4;
      cpa16(S + kr * (R + 8) + rc, X + (long long)(k + kr) * sk + r0 + rc);
    }
  }
}

// fragment element (row r of the output dimension, k index kk) of a staged operand tile
template <int R, bool RK, int KS>
DEV float t3_at(const float* __restrict__ S, int r, int kk) {
  return RK ? S[r * (KS + T3P) + kk] : S[kk * (R + 8) + r];
}
// the BF16 pair (k = kk, kk + 1) of row r of a staged operand tile (kk even)
template <int R, bool RK, int KS>
DEV uint32_t bg_at(const float* __restrict__ S, int r, int kk) {
  if (RK) {
    const float2 v = *reinterpret_cast<const float2*>(S + r * (KS + T3P) + kk);
    return bpk(v.x, v.y);
  }
  return bpk(S[kk * (R + 8) + r], S[(kk + 1) * (R + 8) + r]);
}

template <int BM, int BN, bool AK, bool BKC, int NP, int KS, class Epi, bool T1 = false>
DEV void tgemm_tile(const GemmArgs<NP>& g, int b, int m0, int n0, int kbeg, int kend, float* smem, const Epi& epi,
                    int slice) {
  using SM = TSmem<BM, BN, AK, BKC, KS>;
  constexpr int WN = BN >= 128 ? 4 : (BM >= 128 ? 2 : 4), WM = 8 / WN;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(MT >= 1 && NT >= 1 && WTM % 16 == 0 && WTN % 8 == 0, "t3 warp tile");
#if !defined(S103F_PK8) && !defined(S103F_PK16) && !defined(S105A_BG)
  static_assert(MT * NT <= 16, "tiles above 64 accumulators per thread need register-free promotion (pk8 / pk16)");
#endif
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const float* Ab[NP]; const float* Bb[NP]; long long sam[NP], sak[NP], sbn[NP], sbk[NP]; int kend_p[NP];
  {
    int ksum = 0;
#pragma unroll
    for (int q = 0; q < NP; ++q) {
      Ab[q] = g.p[q].A + boff(b, g.nb2, g.p[q].bsa, g.p[q].bsa2);
      Bb[q] = g.p[q].B + boff(b, g.nb2, g.p[q].bsb, g.p[q].bsb2);
      sam[q] = g.p[q].sam; sak[q] = g.p[q].sak; sbn[q] = g.p[q].sbn; sbk[q] = g.p[q].sbk;
      ksum += g.p[q].K;
      kend_p[q] = ksum;
    }
  }
  auto load_stage = [&](int kk, int st) {
    float* As = smem + st * SM::STAGE;
    float* Bs = As + SM::A_FL;
    int base = 0;
#pragma unroll
    for (int q = 0; q < NP; ++q) {
      const int lo = base;
      base = kend_p[q];
      if (kk >= lo && kk < kend_p[q]) {
        t3_load<BM, AK, KS>(Ab[q], sam[q], sak[q], m0, kk - lo, As);
        t3_load<BN, BKC, KS>(Bb[q], sbn[q], sbk[q], n0, kk - lo, Bs);
      }
    }
  };
  const int nk = (kend - kbeg) / KS;
#pragma unroll
  for (int s = 0; s < SM::NST - 1; ++s) {
    if (s < nk) load_stage(kbeg + s * KS, s);
    cpa_commit();
  }
  for (int it = 0; it < nk; ++it) {
    cpa_wait<SM::NST - 2>();
    __syncthreads();                                               // stage it visible; stage it-1 fully consumed
    if (it + SM::NST - 1 < nk) load_stage(kbeg + (it + SM::NST - 1) * KS, (it + SM::NST - 1) % SM::NST);
    cpa_commit();
    const float* As = smem + (it % SM::NST) * SM::STAGE;
    const float* Bs = As + SM::A_FL;
    // chunked accumulation: the stage's MMAs accumulate from zero; the chunk is then added to acc with FP32 FADD (round
    // to nearest), bounding the tensor-core accumulator's truncation to one K-stage
#if defined(S103F_FX_NOPROMO) || defined(S103F_PK8) || defined(S103F_PK16) || defined(S103F_PS) || defined(S105A_BG)
    float (&tmp)[MT][NT][4] = acc;
#else
    float tmp[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) tmp[i][j][q] = 0.f;
#endif
#if defined(S105A_BG)
    if constexpr (T1) {
      // S105A option tfc: one TF32 product per k8 (operands rounded to nearest TF32), FP32 tensor-core accumulation
#if defined(S105A_TFR)
#pragma unroll 1                                                     // option tfr: rolled k loop (smaller code)
#else
#pragma unroll
#endif
      for (int k8 = 0; k8 < KS; k8 += 8) {
        uint32_t bt[NT][2];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int n = wn0 + j * 8 + gq;
          bt[j][0] = tf32_rna(t3_at<BN, BKC, KS>(Bs, n, k8 + tq));
          bt[j][1] = tf32_rna(t3_at<BN, BKC, KS>(Bs, n, k8 + tq + 4));
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int m = wm0 + i * 16 + gq;
          const uint32_t a0 = tf32_rna(t3_at<BM, AK, KS>(As, m, k8 + tq)), a1 = tf32_rna(t3_at<BM, AK, KS>(As, m + 8, k8 + tq));
          const uint32_t a2 = tf32_rna(t3_at<BM, AK, KS>(As, m, k8 + tq + 4)), a3 = tf32_rna(t3_at<BM, AK, KS>(As, m + 8, k8 + tq + 4));
#pragma unroll
          for (int j = 0; j < NT; ++j) mma_tf32(acc[i][j], a0, a1, a2, a3, bt[j][0], bt[j][1]);
        }
      }
    } else {
    // option bg (policy B16): per k16 one BF16 MMA per (i, j) accumulating in FP32 (tensor-core accumulator)
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t bf[NT][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int n = wn0 + j * 8 + gq;
        bf[j][0] = bg_at<BN, BKC, KS>(Bs, n, k16 + 2 * tq);
        bf[j][1] = bg_at<BN, BKC, KS>(Bs, n, k16 + 8 + 2 * tq);
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int m = wm0 + i * 16 + gq;
        const uint32_t a0 = bg_at<BM, AK, KS>(As, m, k16 + 2 * tq), a1 = bg_at<BM, AK, KS>(As, m + 8, k16 + 2 * tq);
        const uint32_t a2 = bg_at<BM, AK, KS>(As, m, k16 + 8 + 2 * tq), a3 = bg_at<BM, AK, KS>(As, m + 8, k16 + 8 + 2 * tq);
#pragma unroll
        for (int j = 0; j < NT; ++j) bmma(acc[i][j], a0, a1, a2, a3, bf[j][0], bf[j][1]);
      }
    }
    }
#elif defined(S103F_PS)
    // option ps (with pk8 arithmetic): the stage is split once per CTA into TF32 hi / lo planes (same layouts), then every
    // warp loads its fragments from the planes (ldmatrix for k-contiguous operands); same values, same MMA sequence
    static_assert(true, "");
    {
      float* Ph = smem + SM::NST * SM::STAGE;
      float* Pl = Ph + SM::STAGE;
      for (int e4 = threadIdx.x; e4 < SM::STAGE / 4; e4 += NTHREADS) {
        const float4 v = reinterpret_cast<const float4*>(As)[e4];
        uint32_t h0, l0, h1, l1, h2, l2, h3, l3;
        split_tf32(v.x, h0, l0); split_tf32(v.y, h1, l1); split_tf32(v.z, h2, l2); split_tf32(v.w, h3, l3);
        reinterpret_cast<uint4*>(Ph)[e4] = make_uint4(h0, h1, h2, h3);
        reinterpret_cast<uint4*>(Pl)[e4] = make_uint4(l0, l1, l2, l3);
      }
      __syncthreads();
      const float* AH = Ph; const float* AL = Pl; const float* BH = Ph + SM::A_FL; const float* BL = Pl + SM::A_FL;
#pragma unroll
      for (int k8 = 0; k8 < KS; k8 += 8) {
        uint32_t bh[NT][2], bl[NT][2];
        if (BKC && NT % 2 == 0) {
#pragma unroll
          for (int j = 0; j < NT; j += 2) {
            const int off = (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 4) + k8 + ((lane >> 3) & 1) * 4;
            uint32_t r[4];
            ldsm_x4(r, BH + off);
            bh[j][0] = r[0]; bh[j][1] = r[1]; bh[j + 1][0] = r[2]; bh[j + 1][1] = r[3];
            ldsm_x4(r, BL + off);
            bl[j][0] = r[0]; bl[j][1] = r[1]; bl[j + 1][0] = r[2]; bl[j + 1][1] = r[3];
          }
        } else {
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int n = wn0 + j * 8 + gq;
            bh[j][0] = __float_as_uint(t3_at<BN, BKC, KS>(BH, n, k8 + tq));
            bh[j][1] = __float_as_uint(t3_at<BN, BKC, KS>(BH, n, k8 + tq + 4));
            bl[j][0] = __float_as_uint(t3_at<BN, BKC, KS>(BL, n, k8 + tq));
            bl[j][1] = __float_as_uint(t3_at<BN, BKC, KS>(BL, n, k8 + tq + 4));
          }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          uint32_t ah[4], al[4];
          if (AK) {
            const int off = (wm0 + 16 * i + (lane & 7) + ((lane >> 3) & 1) * 8) * (KS + 4) + k8 + (lane >> 4) * 4;
            ldsm_x4(ah, AH + off);
            ldsm_x4(al, AL + off);
          } else {
            const int m = wm0 + i * 16 + gq;
            ah[0] = __float_as_uint(t3_at<BM, AK, KS>(AH, m, k8 + tq)); ah[1] = __float_as_uint(t3_at<BM, AK, KS>(AH, m + 8, k8 + tq));
            ah[2] = __float_as_uint(t3_at<BM, AK, KS>(AH, m, k8 + tq + 4)); ah[3] = __float_as_uint(t3_at<BM, AK, KS>(AH, m + 8, k8 + tq + 4));
            al[0] = __float_as_uint(t3_at<BM, AK, KS>(AL, m, k8 + tq)); al[1] = __float_as_uint(t3_at<BM, AK, KS>(AL, m + 8, k8 + tq));
            al[2] = __float_as_uint(t3_at<BM, AK, KS>(AL, m, k8 + tq + 4)); al[3] = __float_as_uint(t3_at<BM, AK, KS>(AL, m + 8, k8 + tq + 4));
          }
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            float d[4];
            mma3z(d, ah, al, bh[j][0], bh[j][1], bl[j][0], bl[j][1]);
            add4(acc[i][j], d);
          }
        }
      }
    }
#elif defined(S103F_PK16)
    // option pk16: one promotion chunk per two k8 steps (six MMAs from zero, then one FP32 add)
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t bh[2][NT][2], bl[2][NT][2];
#pragma unroll
      for (int hh = 0; hh < 2; ++hh)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int n = wn0 + j * 8 + gq;
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k16 + 8 * hh + tq), bh[hh][j][0], bl[hh][j][0]);
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k16 + 8 * hh + tq + 4), bh[hh][j][1], bl[hh][j][1]);
        }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int m = wm0 + i * 16 + gq;
        uint32_t ah[2][4], al[2][4];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          split_tf32(t3_at<BM, AK, KS>(As, m, k16 + 8 * hh + tq), ah[hh][0], al[hh][0]);
          split_tf32(t3_at<BM, AK, KS>(As, m + 8, k16 + 8 * hh + tq), ah[hh][1], al[hh][1]);
          split_tf32(t3_at<BM, AK, KS>(As, m, k16 + 8 * hh + tq + 4), ah[hh][2], al[hh][2]);
          split_tf32(t3_at<BM, AK, KS>(As, m + 8, k16 + 8 * hh + tq + 4), ah[hh][3], al[hh][3]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          float d[4];
          mma3z(d, ah[0], al[0], bh[0][j][0], bh[0][j][1], bl[0][j][0], bl[0][j][1]);
          mma3c(d, ah[1], al[1], bh[1][j][0], bh[1][j][1], bl[1][j][0], bl[1][j][1]);
          add4(acc[i][j], d);
        }
      }
    }
#else
#pragma unroll
    for (int k8 = 0; k8 < KS; k8 += 8) {
#if defined(S103F_T3P)
      // option t3p: all fragments of the k8 step first, then the three product passes over every (i, j) (16+
      // independent MMAs between dependent ones); per accumulator the same sequence lo.hi, hi.lo, hi.hi (bitwise = t3)
      uint32_t bh[NT][2], bl[NT][2], ah[MT][4], al[MT][4];
#if defined(S103F_T3L)
      if (BKC && NT % 2 == 0) {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t r[4];
          ldsm_x4(r, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 4) + k8 + ((lane >> 3) & 1) * 4);
          split_tf32(__uint_as_float(r[0]), bh[j][0], bl[j][0]); split_tf32(__uint_as_float(r[1]), bh[j][1], bl[j][1]);
          split_tf32(__uint_as_float(r[2]), bh[j + 1][0], bl[j + 1][0]);
          split_tf32(__uint_as_float(r[3]), bh[j + 1][1], bl[j + 1][1]);
        }
      } else
#endif
      {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int n = wn0 + j * 8 + gq;
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k8 + tq), bh[j][0], bl[j][0]);
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k8 + tq + 4), bh[j][1], bl[j][1]);
        }
      }
#if defined(S103F_T3L)
      if (AK) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          uint32_t r[4];
          ldsm_x4(r, As + (wm0 + 16 * i + (lane & 7) + ((lane >> 3) & 1) * 8) * (KS + 4) + k8 + (lane >> 4) * 4);
#pragma unroll
          for (int q = 0; q < 4; ++q) split_tf32(__uint_as_float(r[q]), ah[i][q], al[i][q]);
        }
      } else
#endif
      {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const int m = wm0 + i * 16 + gq;
          split_tf32(t3_at<BM, AK, KS>(As, m, k8 + tq), ah[i][0], al[i][0]);
          split_tf32(t3_at<BM, AK, KS>(As, m + 8, k8 + tq), ah[i][1], al[i][1]);
          split_tf32(t3_at<BM, AK, KS>(As, m, k8 + tq + 4), ah[i][2], al[i][2]);
          split_tf32(t3_at<BM, AK, KS>(As, m + 8, k8 + tq + 4), ah[i][3], al[i][3]);
        }
      }
#if !defined(S103F_BENCH_T1)                                         // bench-only: one TF32 product (timing probe)
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_tf32(tmp[i][j], al[i][0], al[i][1], al[i][2], al[i][3], bh[j][0], bh[j][1]);
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_tf32(tmp[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bl[j][0], bl[j][1]);
#endif
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_tf32(tmp[i][j], ah[i][0], ah[i][1], ah[i][2], ah[i][3], bh[j][0], bh[j][1]);
#else
      uint32_t bh[NT][2], bl[NT][2];
#if defined(S103F_TLDM)
      if (BKC && NT % 2 == 0) {                                      // option tldm: ldmatrix fragment loads
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t r[4];
          ldsm_x4(r, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 4) + k8 + ((lane >> 3) & 1) * 4);
          split_tf32(__uint_as_float(r[0]), bh[j][0], bl[j][0]); split_tf32(__uint_as_float(r[1]), bh[j][1], bl[j][1]);
          split_tf32(__uint_as_float(r[2]), bh[j + 1][0], bl[j + 1][0]);
          split_tf32(__uint_as_float(r[3]), bh[j + 1][1], bl[j + 1][1]);
        }
      } else
#endif
      {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          const int n = wn0 + j * 8 + gq;
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k8 + tq), bh[j][0], bl[j][0]);
          split_tf32(t3_at<BN, BKC, KS>(Bs, n, k8 + tq + 4), bh[j][1], bl[j][1]);
        }
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const int m = wm0 + i * 16 + gq;
        uint32_t ah[4], al[4];
#if defined(S103F_TLDM)
        if (AK) {
          uint32_t r[4];
          ldsm_x4(r, As + (wm0 + 16 * i + (lane & 7) + ((lane >> 3) & 1) * 8) * (KS + 4) + k8 + (lane >> 4) * 4);
#pragma unroll
          for (int q = 0; q < 4; ++q) split_tf32(__uint_as_float(r[q]), ah[q], al[q]);
        } else
#endif
        {
        split_tf32(t3_at<BM, AK, KS>(As, m, k8 + tq), ah[0], al[0]);
        split_tf32(t3_at<BM, AK, KS>(As, m + 8, k8 + tq), ah[1], al[1]);
        split_tf32(t3_at<BM, AK, KS>(As, m, k8 + tq + 4), ah[2], al[2]);
        split_tf32(t3_at<BM, AK, KS>(As, m + 8, k8 + tq + 4), ah[3], al[3]);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#if defined(S103F_PK8)
          // option pk8: one promotion chunk per k8 step (three MMAs from zero, then one FP32 add)
          float d[4];
          mma3z(d, ah, al, bh[j][0], bh[j][1], bl[j][0], bl[j][1]);
          add4(acc[i][j], d);
#else
#if !defined(S103F_FX_1T)                                          // semantic fault fx_1t: one TF32 product only
          mma_tf32(tmp[i][j], al[0], al[1], al[2], al[3], bh[j][0], bh[j][1]);
          mma_tf32(tmp[i][j], ah[0], ah[1], ah[2], ah[3], bl[j][0], bl[j][1]);
#endif
          mma_tf32(tmp[i][j], ah[0], ah[1], ah[2], ah[3], bh[j][0], bh[j][1]);
#endif
        }
      }
#endif
    }
#endif
#if !defined(S103F_FX_NOPROMO) && !defined(S103F_PK8) && !defined(S103F_PK16) && !defined(S103F_PS) && !defined(S105A_BG)
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] += tmp[i][j][q];
#endif
  }
  cpa_wait<0>();
  __syncthreads();                                                 // every warp's last fragment read is done
  // stage the tile: C fragment (row gq / gq + 8, cols 2tq, 2tq + 1) of every (i, j) -> Cs[m][n]
  float* Cs = smem;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = wm0 + i * 16 + gq, n = wn0 + j * 8 + 2 * tq;
      *reinterpret_cast<float2*>(Cs + m * SM::CS + n) = make_float2(acc[i][j][0], acc[i][j][1]);
      *reinterpret_cast<float2*>(Cs + (m + 8) * SM::CS + n) = make_float2(acc[i][j][2], acc[i][j][3]);
    }
  __syncthreads();
  if (g.ksplit == 1) {
#if defined(S103F_EPF)
    for (int f = threadIdx.x; f < 2 * NTHREADS && f < BM * BN / 4; f += NTHREADS)
      epi_pre(epi, b, m0 + f / (BN / 4), n0 + (f % (BN / 4)) * 4);
#endif
#if defined(S105A_EU)
#pragma unroll 4
#else
#pragma unroll 1
#endif
    for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
      const int mm = f / (BN / 4), nn = (f % (BN / 4)) * 4;
#if defined(S103F_EPF)
      const int f2 = f + 2 * NTHREADS;
      if (f2 < BM * BN / 4) epi_pre(epi, b, m0 + f2 / (BN / 4), n0 + (f2 % (BN / 4)) * 4);
#endif
      epi(b, m0 + mm, n0 + nn, *reinterpret_cast<const float4*>(Cs + mm * SM::CS + nn));
    }
    __syncthreads();                                               // Cs readers done before the next tile's loads
    return;
  }
#pragma unroll 1
  for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
    const int mm = f / (BN / 4), nn = (f % (BN / 4)) * 4;
    *reinterpret_cast<float4*>(g.part + (((long long)slice * g.nb + b) * g.M + m0 + mm) * g.N + n0 + nn) =
        *reinterpret_cast<const float4*>(Cs + mm * SM::CS + nn);
  }
  if (g.cnt) {
    // last-arriver reduction (gemm.cuh): the CTA completing a tile's final slice sums all slices in slice order
    __shared__ unsigned t3_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const int tile = (b * (g.M / BM) + m0 / BM) * (g.N / BN) + n0 / BN;
      const unsigned old = atomicAdd(g.cnt + tile, 1u);
      t3_last = (old == (unsigned)g.ksplit - 1u);
      if (t3_last) atomicExch(g.cnt + tile, 0u);
    }
    __syncthreads();
    if (t3_last) {
      __threadfence();
#if defined(S103F_EPF)
      for (int f = threadIdx.x; f < 2 * NTHREADS && f < BM * BN / 4; f += NTHREADS)
        epi_pre(epi, b, m0 + f / (BN / 4), n0 + (f % (BN / 4)) * 4);
#endif
#if defined(S105A_EU)
#pragma unroll 4
#else
#pragma unroll 1
#endif
      for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
        const int m = m0 + f / (BN / 4), n = n0 + (f % (BN / 4)) * 4;
#if defined(S103F_EPF)
        const int f2 = f + 2 * NTHREADS;
        if (f2 < BM * BN / 4) epi_pre(epi, b, m0 + f2 / (BN / 4), n0 + (f2 % (BN / 4)) * 4);
#endif
        float4 q[8];
#pragma unroll
        for (int s = 0; s < 8; ++s)
          if (s < g.ksplit)
            q[s] = __ldcg(reinterpret_cast<const float4*>(g.part + (((long long)s * g.nb + b) * g.M + m) * g.N + n));
        float4 v = q[0];
#pragma unroll
        for (int s = 1; s < 8; ++s)
          if (s < g.ksplit) { v.x += q[s].x; v.y += q[s].y; v.z += q[s].z; v.w += q[s].w; }
        epi(b, m, n, v);
      }
    }
  }
  __syncthreads();
}

template <int BM, int BN, bool AK, bool BKC, int NP, int KS, bool T1 = false, class Epi>
__device__ __noinline__ void tgemm_phase(const GemmArgs<NP> g, const Epi epi, float* smem, int vbase = 0) {
  const int tm = g.M / BM, tn = g.N / BN;
  int ktot = 0;
#pragma unroll
  for (int p = 0; p < NP; ++p) ktot += g.p[p].K;
  const int per = ktot / g.ksplit;                                 // generator guarantees divisibility by KS
  const int ntask = g.nb * tm * tn * g.ksplit;
  const int vb = ((int)blockIdx.x + (int)gridDim.x - (vbase % (int)gridDim.x)) % (int)gridDim.x;
  for (int task = vb; task < ntask; task += gridDim.x) {
    int r = task;
    const int s = r % g.ksplit; r /= g.ksplit;
    const int j = r % tn; r /= tn;
    const int i = r % tm; r /= tm;
    const int b = r;
    tgemm_tile<BM, BN, AK, BKC, NP, KS, Epi, T1>(g, b, i * BM, j * BN, s * per, (s + 1) * per, smem, epi, s);
  }
}

}  // namespace s101h
