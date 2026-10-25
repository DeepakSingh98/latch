// LATCH-STEP19 bounded-edge adapter of popsc_mm.cuh's FP32 sstage/stile/smm_phase.
// Preserves its 16x16 thread arrangement, cp.async stage ring, ascending-K fmaf
// chain and output layout. Changes only runtime extent checks, a partial final K
// stage, and live-output stores. The original static body is unchanged.
#pragma once
#include "popsc_mm.cuh"

namespace latch_dynamic {
template <int R, bool RK, int KC>
__device__ __forceinline__ void stage(const float* __restrict__ X, long long sr, long long sk,
                                    int r0, int k, int rlim, int klim, float* __restrict__ S) {
  constexpr int CH = R * KC / 4;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += 256) {
    const int c = c0 + threadIdx.x;
    if (CH % 256 == 0 || c < CH) {
      const int r = RK ? c / (KC / 4) : (c % (R / 4)) * 4;
      const int kk = RK ? (c % (KC / 4)) * 4 : c / (R / 4);
      const int remaining = RK ? klim - (k + kk) : rlim - (r0 + r);
      const bool row_live = RK ? r0 + r < rlim : k + kk < klim;
      const int elements = row_live ? max(0, min(4, remaining)) : 0;
      float* dst = RK ? S + r * (KC + 4) + kk : S + kk * (R + 4) + r;
      const float* src = elements ? X + (long long)(r0 + r) * sr + (long long)(k + kk) * sk : X;
      const unsigned address = (unsigned)__cvta_generic_to_shared(dst);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                   :: "r"(address), "l"(src), "r"(elements * 4) : "memory");
    }
  }
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __forceinline__ void tile(const popsc::SArgs& g, int m0, int n0, float* S) {
  constexpr int TM = BM / 16, TN = BN / 16;
  constexpr int A_EL = AK ? BM * (KC + 4) : KC * (BM + 4);
  constexpr int B_EL = BKC ? BN * (KC + 4) : KC * (BN + 4);
  constexpr int STAGE = A_EL + B_EL;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  const int nk = (g.K + KC - 1) / KC;
  auto load = [&](int kk, int st) {
    float* As = S + st * STAGE;
    stage<BM, AK, KC>(g.A, g.sam, g.sak, m0, kk, g.M, g.K, As);
    stage<BN, BKC, KC>(g.B, g.sbn, g.sbk, n0, kk, g.N, g.K, As + A_EL);
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
      if (it * KC + kk >= g.K) break;
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
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int m = m0 + ty + 16 * i, n = n0 + tx + 16 * j;
      if (m < g.M && n < g.N) g.C[(long long)m * g.ldc + n] = acc[i][j];
    }
  __syncthreads();
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void smm_phase(const popsc::SArgs g, float* smem) {
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN;
  for (int task = blockIdx.x; task < tm * tn; task += gridDim.x) {
    const int m0 = (task / tn) * BM, n0 = (task % tn) * BN;
    if (g.K % KC == 0 && m0 + BM <= g.M && n0 + BN <= g.N)
      popsc::stile<BM, BN, AK, BKC, KC, NST>(g, 0, m0, n0, smem);
    else
      tile<BM, BN, AK, BKC, KC, NST>(g, m0, n0, smem);
  }
}

// The ordinary static matrix selector's fixed split-K count applied to a
// bounded-M product. Each task defines one active partial tile. Publication
// and the fixed ascending-slice sum are separate compiler dependency phases,
// so no arrival counters, floating atomics or internal grid fence are needed.
template <int BM, int BN, bool AK, bool BKC, int KC, int NST, int SPLITS>
__device__ __noinline__ void smm_split_phase(const popsc::SArgs g, float* partial, float* smem) {
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN;
  const int per = g.K / SPLITS;
  for (int task = blockIdx.x; task < tm * tn * SPLITS; task += gridDim.x) {
    const int s = task % SPLITS, tile_index = task / SPLITS;
    const int m0 = (tile_index / tn) * BM, n0 = (tile_index % tn) * BN;
    popsc::SArgs slice = g;
    slice.A += (long long)s * per * g.sak;
    slice.B += (long long)s * per * g.sbk;
    slice.C = partial + (long long)s * g.N;
    slice.ldc = (long long)g.N * SPLITS;
    slice.bsc = (long long)g.M * slice.ldc;
    slice.K = per;
    if (m0 + BM <= g.M && n0 + BN <= g.N)
      popsc::stile<BM, BN, AK, BKC, KC, NST>(slice, 0, m0, n0, smem);
    else
      tile<BM, BN, AK, BKC, KC, NST>(slice, m0, n0, smem);
  }
}
}  // namespace latch_dynamic
