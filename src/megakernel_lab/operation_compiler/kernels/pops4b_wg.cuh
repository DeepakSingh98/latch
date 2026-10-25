// P-OPS4-B warpgroup-MMA (wgmma) BF16 body for the operation compiler (new code in this lane).
//
// Reused unchanged: the m64nNk16 wgmma wrappers of the Latch 0.5 S107A matrix engine (vendor/s107a_wgmma.cuh, byte
// copy of P4-A f9765574a latch05/.../native/lib/s107a_wgmma.cuh, blob cff84a14) and that engine's shared-memory
// operand layouts and matrix-descriptor encodings (s107a_mm.cuh mme_dk / mme_dmn, 128-byte swizzle). What differs
// from S107A: operands are staged by cp.async from the operation's own strided BF16 operands (no CUtensorMap, which the
// operation compiler's runtime does not encode), the tile loop is the P-OPS-C task loop over CTA slices, and absent
// operand elements (M, N or K tails) are zero-filled by cp.async source sizes while stores are masked.
//
// Per CTA (256 threads = two warpgroups) and output tile BM x BN: WM = 2 splits M between the warpgroups (each 64 x BN
// per 64-row slab), WM = 1 splits N. Each 64-wide k stage issues four k16 wgmma steps per accumulator slab.
// Arithmetic: BF16 x BF16 products accumulated in FP32 by the tensor core, ascending k16 steps, one FP32 accumulator
// set per tile from the first step (scale-d = 0), absent k terms are exact +0 products; one cvt.rn.bf16x2 rounding.
// Not bitwise identical to the mma.sync bodies (different tensor-core instruction); gated by tolerance.
#pragma once
#include "popsc_mm.cuh"
#include "s107a_wgmma.cuh"

namespace pops4b {

using s101h::bf16;

DEV unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
// 128-byte swizzle descriptors: K-major rows of 64 k (SBO 1024 = 8 rows x 128 B); M/N-major 64-wide chunks
// (LBO 8192 between chunks, SBO 1024 = 8 k rows). Identical encodings to S107A mme_dk / mme_dmn.
DEV uint64_t dk(unsigned a) {
  return (uint64_t)((a & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
DEV uint64_t dmn(unsigned a) {
  return (uint64_t)((a & 0x3FFFFu) >> 4) | ((uint64_t)(8192 >> 4) << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
DEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
DEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N> DEV void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
DEV void async_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int R> DEV void keep(float (&d)[R]) {
#pragma unroll
  for (int i = 0; i < R; ++i) asm volatile("" : "+f"(d[i])::"memory");
}

template <int BM, int BN, int WM>
struct Cfg {
  static constexpr int WN = 2 / WM, WTM = BM / WM, WTN = BN / WN, MT = WTM / 64, R = WTN / 2;
  static constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
  static_assert(WM == 1 || WM == 2, "warpgroup split");
  static_assert(WTM % 64 == 0 && MT <= 2 && WTN % 8 == 0 && WTN <= 256, "warpgroup tile");
  static_assert(BM % 64 == 0 && BN % 32 == 0, "64-wide swizzle chunks (BN 96: K-major B only, planner-enforced)");
};

// One 64-k stage of an R-row operand tile. RK: k-contiguous rows, smem [R][64] with 16-byte chunk c of row r at
// r * 128 + ((c ^ (r & 7)) << 4). !RK: r-contiguous, smem [R / 64][64 k][64 r] with chunk (k, rc) at
// (rc / 64) * 8192 + k * 128 + ((((rc % 64) / 8) ^ (k & 7)) << 4). Absent elements (r >= rlim or k >= klim) stage +0.
template <int R, bool RK, bool EDGE>
DEV void stage(const bf16* __restrict__ X, long long sr, long long sk, int r0, int k, unsigned dst, int rlim, int klim) {
  constexpr int CH = R * 8;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      unsigned off;
      const bf16* src;
      int e = 8;
      if (RK) {
        const int r = c >> 3, cc = c & 7;
        off = r * 128 + ((cc ^ (r & 7)) << 4);
        src = X + (long long)(r0 + r) * sr + k + cc * 8;
        if (EDGE) {
          const int left = klim - (k + cc * 8);
          e = (r0 + r < rlim) ? (left < 0 ? 0 : (left > 8 ? 8 : left)) : 0;
        }
      } else {
        const int kr = c / (R / 8), rc = (c % (R / 8)) * 8;
        off = (rc >> 6) * 8192 + kr * 128 + ((((rc & 63) >> 3) ^ (kr & 7)) << 4);
        src = X + (long long)(k + kr) * sk + r0 + rc;
        if (EDGE) {
          const int left = rlim - (r0 + rc);
          e = (k + kr < klim) ? (left < 0 ? 0 : (left > 8 ? 8 : left)) : 0;
        }
      }
      if (EDGE) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst + off), "l"(e ? src : X), "r"(2 * e)
                     : "memory");
      } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst + off), "l"(src) : "memory");
      }
    }
  }
}

// DIAG (timing diagnostics only; results are wrong): 1 = no operand loads after the prologue, 2 = no wgmma.
// WIDE (LATCH-TINYMOE-EGRAD): g.C addresses FP32 storage; each result is rounded once to BF16 exactly as below and
// stored exactly widened, i.e. the bits a separate BF16 -> FP32 cast of the BF16 result would write. Requires PAIR.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool EDGE, bool PAIR, int DIAG = 0, bool WIDE = false>
DEV void tile(const popsc::BArgs& g, int b, int m0, int n0, unsigned base) {
  static_assert(!WIDE || PAIR, "widened stores write whole even-column pairs");
  using C = Cfg<BM, BN, WM>;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
  const int nk = EDGE ? (g.K + 63) / 64 : g.K / 64;
  auto load = [&](int kk, int st) {
    const unsigned dA = base + st * C::STAGE;
    stage<BM, AK, EDGE>(A, g.sam, g.sak, m0, kk, dA, g.M, g.K);          // (row stride, k stride); one is unit
    stage<BN, BKC, EDGE>(B, g.sbn, g.sbk, n0, kk, dA + C::A_BYTES, g.N, g.K);
  };
  float acc[C::MT][C::R];
#pragma unroll
  for (int i = 0; i < C::MT; ++i)
#pragma unroll
    for (int j = 0; j < C::R; ++j) acc[i][j] = 0.f;
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * 64, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    async_fence();                                                 // cp.async (generic proxy) -> wgmma (async proxy)
    __syncthreads();
#pragma unroll
    for (int i = 0; i < C::MT; ++i) keep(acc[i]);
    wg_fence();
    const unsigned sA = base + (it % NST) * C::STAGE, sB = sA + C::A_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      const uint64_t db = BKC ? dk(sB + wgn * C::WTN * 128 + k16 * 32) : dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
        const int r0 = wgm * C::WTM + i * 64;
        const uint64_t da = AK ? dk(sA + r0 * 128 + k16 * 32) : dmn(sA + (r0 / 64) * 8192 + k16 * 2048);
        if (DIAG != 2) s101h::MmeWg<C::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (it | k16) != 0);
      }
    }
    wg_commit();
    wg_wait<1>();                                                  // this warpgroup's step it-1 is complete
#pragma unroll
    for (int i = 0; i < C::MT; ++i) keep(acc[i]);
    __syncthreads();                                               // both warpgroups: stage (it-1) % NST is free
    if (DIAG != 1 && it + NST - 1 < nk) load((it + NST - 1) * 64, (it + NST - 1) % NST);
    s101h::cpa_commit();
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < C::MT; ++i) keep(acc[i]);
  s101h::cpa_wait<0>();
  bf16* Cp = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < C::MT; ++i) {
    const int row = m0 + wgm * C::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
    for (int j = 0; j < C::WTN / 8; ++j) {
      const int col = n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row + 8 * h;
        const uint32_t v = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
        if (WIDE) {
          if (!EDGE || (r < g.M && col < g.N)) {
            float* Cw = reinterpret_cast<float*>(g.C) + (long long)b * g.bsc;
            *reinterpret_cast<float2*>(Cw + (long long)r * g.ldc + col) =
                make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
          }
        } else if (!EDGE) {
          *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) = v;
        } else if (r < g.M && col < g.N) {
          if (PAIR) {
            *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) = v;
          } else {
            unsigned short* c = reinterpret_cast<unsigned short*>(Cp + (long long)r * g.ldc + col);
            c[0] = (unsigned short)(v & 0xffffu);
            if (col + 1 < g.N) c[1] = (unsigned short)(v >> 16);
          }
        }
      }
    }
  }
  __syncthreads();                                                 // every wgmma/cp.async of this tile is done
}

// ceil(M/BM) x ceil(N/BN) tiles over the CTA slice (popsc::slice_of). Interior tiles of even-N products with a whole
// number of 64-k stages run without masks.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, int DIAG = 0>
DEV void wmm_body(const popsc::BArgs& g, float* smem, int lo, int hi, int den, int rot) {
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = g.nb * tm * tn;
  const unsigned base = (sa(smem) + 1023u) & ~1023u;
  const bool kfull = g.K % 64 == 0;
  __syncthreads();                                                 // earlier users of shared memory are done
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn, m0 = (r % tm) * BM, n0 = j * BN;
    if (PAIR && kfull && m0 + BM <= g.M && n0 + BN <= g.N)
      tile<BM, BN, WM, AK, BKC, NST, false, true, DIAG>(g, r / tm, m0, n0, base);
    else
      tile<BM, BN, WM, AK, BKC, NST, true, PAIR, DIAG>(g, r / tm, m0, n0, base);
  }
}

// Flattened form (S107A-style): the CTA's tasks x 64-k stages form one sequence; stage loads run NST - 1 steps ahead
// across task boundaries, so the next tile's operands arrive while this tile's last MMAs and its register-direct
// epilogue run. Masked staging only for stages of tiles that cross M/N or for the K-tail stage; masked stores only for
// tiles crossing M/N (or odd N). Per-element arithmetic is identical to `tile`.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR>
DEV void wmm_body_flat(const popsc::BArgs& g, float* smem, int lo, int hi, int den, int rot) {
  using C = Cfg<BM, BN, WM>;
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = g.nb * tm * tn;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  const int nk = (g.K + 63) / 64, Q = ntm * nk;
  if (Q == 0) return;
  const bool ktail = g.K % 64 != 0;
  const unsigned base = (sa(smem) + 1023u) & ~1023u;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  auto where = [&](int i, int& b, int& m0, int& n0) {
    const int task = s.me + i * s.count, j = task % tn, r = task / tn;
    b = r / tm; m0 = (r % tm) * BM; n0 = j * BN;
  };
  auto load = [&](int q) {
    int b, m0, n0;
    where(q / nk, b, m0, n0);
    const int kk = (q % nk) * 64;
    const bf16* A = g.A + (long long)b * g.bsa;
    const bf16* B = g.B + (long long)b * g.bsb;
    const unsigned dA = base + (q % NST) * C::STAGE;
    if (m0 + BM <= g.M && n0 + BN <= g.N && !(ktail && kk + 64 > g.K)) {
      stage<BM, AK, false>(A, g.sam, g.sak, m0, kk, dA, g.M, g.K);
      stage<BN, BKC, false>(B, g.sbn, g.sbk, n0, kk, dA + C::A_BYTES, g.N, g.K);
    } else {
      stage<BM, AK, true>(A, g.sam, g.sak, m0, kk, dA, g.M, g.K);
      stage<BN, BKC, true>(B, g.sbn, g.sbk, n0, kk, dA + C::A_BYTES, g.N, g.K);
    }
  };
  float acc[C::MT][C::R];
#pragma unroll
  for (int i = 0; i < C::MT; ++i)
#pragma unroll
    for (int j = 0; j < C::R; ++j) acc[i][j] = 0.f;
  __syncthreads();                                                 // earlier users of shared memory are done
#pragma unroll
  for (int q = 0; q < NST - 1; ++q) {
    if (q < Q) load(q);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int q = 0; q < Q; ++q) {
    const int kk = q % nk;
    s101h::cpa_wait<NST - 2>();
    async_fence();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < C::MT; ++i) keep(acc[i]);
    wg_fence();
    const unsigned sA = base + (q % NST) * C::STAGE, sB = sA + C::A_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      const uint64_t db = BKC ? dk(sB + wgn * C::WTN * 128 + k16 * 32) : dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
        const int r0 = wgm * C::WTM + i * 64;
        const uint64_t da = AK ? dk(sA + r0 * 128 + k16 * 32) : dmn(sA + (r0 / 64) * 8192 + k16 * 2048);
        s101h::MmeWg<C::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (kk | k16) != 0);
      }
    }
    wg_commit();
    wg_wait<1>();
#pragma unroll
    for (int i = 0; i < C::MT; ++i) keep(acc[i]);
    __syncthreads();                                               // stage (q - 1) % NST is free in both warpgroups
    if (q + NST - 1 < Q) load(q + NST - 1);
    s101h::cpa_commit();
    if (kk == nk - 1) {
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < C::MT; ++i) keep(acc[i]);
      int b, m0, n0;
      where(q / nk, b, m0, n0);
      const bool masked = !PAIR || m0 + BM > g.M || n0 + BN > g.N;
      bf16* Cp = g.C + (long long)b * g.bsc;
#pragma unroll
      for (int i = 0; i < C::MT; ++i) {
        const int row = m0 + wgm * C::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < C::WTN / 8; ++j) {
          const int col = n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = row + 8 * h;
            const uint32_t v = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
            if (!masked) {
              *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) = v;
            } else if (r < g.M && col < g.N) {
              if (PAIR) {
                *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) = v;
              } else {
                unsigned short* c = reinterpret_cast<unsigned short*>(Cp + (long long)r * g.ldc + col);
                c[0] = (unsigned short)(v & 0xffffu);
                if (col + 1 < g.N) c[1] = (unsigned short)(v >> 16);
              }
            }
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
  __syncthreads();                                                 // every wgmma/cp.async of the phase is done
}

// Called form (one function per configuration) and an entry-resident form (inlined into the kernel; ptxas otherwise
// may serialize wgmma inside called functions, see P4-A).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool FLAT = false>
__device__ __noinline__ void wmm_phase(const popsc::BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1,
                                       int rot = 0) {
  if constexpr (FLAT) wmm_body_flat<BM, BN, WM, AK, BKC, NST, PAIR>(g, smem, lo, hi, den, rot);
  else wmm_body<BM, BN, WM, AK, BKC, NST, PAIR>(g, smem, lo, hi, den, rot);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool FLAT = false, int DIAG = 0>
DEV void wmm_phase_inline(const popsc::BArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  if constexpr (FLAT) wmm_body_flat<BM, BN, WM, AK, BKC, NST, PAIR>(g, smem, lo, hi, den, rot);
  else wmm_body<BM, BN, WM, AK, BKC, NST, PAIR, DIAG>(g, smem, lo, hi, den, rot);
}

}  // namespace pops4b
