// Release step 14: deterministic split-K for the FP32 SIMT small-tile body (popsc::stile) on underfilled batched
// products with long K. Task = (K slice, batch, tile). Slice s accumulates its contiguous K range
// [s * K / S, (s + 1) * K / S) with the unchanged tile body (ascending-k fmaf per element from +0) and stores the
// unrounded FP32 tile to part[s][batch][M][N]; a separate reduce phase adds the slices in ascending slice order.
// FP32 reassociation of the K sum (declared, as the compiler's existing ordered FP32 split-K); grid independent.
#pragma once
#include "popsc_mm.cuh"

namespace smmsplit {

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void phase(const popsc::SArgs g, float* part, int S, int ks, float* smem, int lo = 0,
                                   int hi = 1, int den = 1, int rot = 0) {
  // Slice s covers [s ks, min(K, (s + 1) ks)); ks % KC == 0 and K % KC == 0 (planner); the last slice may be shorter.
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, per = g.nb * tm * tn, ntask = S * per;
  const long long total = (long long)g.nb * g.M * g.N;
  for (int task = s.me; task < ntask; task += s.count) {
    const int sl = task / per, rest = task % per;
    const int j = rest % tn, r = rest / tn;
    popsc::SArgs h = g;
    h.A = g.A + (long long)sl * ks * g.sak;
    h.B = g.B + (long long)sl * ks * g.sbk;
    h.K = g.K - sl * ks < ks ? g.K - sl * ks : ks;
    h.C = part + (long long)sl * total;
    h.ldc = g.N;
    h.bsc = (long long)g.M * g.N;
    popsc::stile<BM, BN, AK, BKC, KC, NST>(h, r / tm, (r % tm) * BM, j * BN, smem);
  }
}

}  // namespace smmsplit

namespace smmsplit {

// Masked-edge form of popsc::stile (release step 14): M and N need not divide the tile. Operand chunks outside
// [0, M) x [0, K) or [0, K) x [0, N) are zero-filled (cp.async src-size 0) and out-of-range results are not stored.
// K is divided exactly by KC (planner), so no k term is padded: every stored element is the same ascending-k fmaf
// chain from +0 as the generic tile (bitwise). Requires 16-byte aligned chunks: K % 4, and M % 4 (N % 4) for
// row-contiguous A (column-contiguous B).
__device__ __forceinline__ void cpa16z(float* dst, const float* src, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

template <int R, bool RK, int KC>
__device__ __forceinline__ void sstage_edge(const float* __restrict__ X, long long sr, long long sk, int r0, int k,
                                            int rlim, float* __restrict__ S) {
  constexpr int CH = R * KC / 4;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += 256) {
    const int c = c0 + threadIdx.x;
    if (CH % 256 == 0 || c < CH) {
      if (RK) {
        const int r = c / (KC / 4), kc = (c % (KC / 4)) * 4;
        const bool ok = r0 + r < rlim;
        cpa16z(S + r * (KC + 4) + kc, ok ? X + (long long)(r0 + r) * sr + k + kc : X, ok);
      } else {
        const int kr = c / (R / 4), rc = (c % (R / 4)) * 4;
        const bool ok = r0 + rc < rlim;
        cpa16z(S + kr * (R + 4) + rc, ok ? X + (long long)(k + kr) * sk + r0 + rc : X, ok);
      }
    }
  }
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __forceinline__ void stile_edge(const popsc::SArgs& g, int b, int m0, int n0, float* S) {
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
    sstage_edge<BM, AK, KC>(A, g.sam, g.sak, m0, kk, g.M, As);
    sstage_edge<BN, BKC, KC>(B, g.sbn, g.sbk, n0, kk, g.N, As + A_EL);
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
    for (int j = 0; j < TN; ++j) {
      const int r = m0 + ty + 16 * i, c = n0 + tx + 16 * j;
      if (r < g.M && c < g.N) C[(long long)r * g.ldc + c] = acc[i][j];
    }
  __syncthreads();
}

// Masked tiles (ceil-divided extents) with an optional ordered K split (S == 1: results straight to g.C).
template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void edge_phase(const popsc::SArgs g, float* part, int S, int ks, float* smem, int lo = 0,
                                        int hi = 1, int den = 1, int rot = 0) {
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, per = g.nb * tm * tn, ntask = S * per;
  const long long total = (long long)g.nb * g.M * g.N;
  for (int task = s.me; task < ntask; task += s.count) {
    const int sl = task / per, rest = task % per;
    const int j = rest % tn, r = rest / tn;
    popsc::SArgs h = g;
    if (S > 1) {
      h.A = g.A + (long long)sl * ks * g.sak;
      h.B = g.B + (long long)sl * ks * g.sbk;
      h.K = g.K - sl * ks < ks ? g.K - sl * ks : ks;
      h.C = part + (long long)sl * total;
      h.ldc = g.N;
      h.bsc = (long long)g.M * g.N;
    }
    stile_edge<BM, BN, AK, BKC, KC, NST>(h, r / tm, (r % tm) * BM, j * BN, smem);
  }
}

}  // namespace smmsplit

namespace smmsplit {

// Step-14 speed ('ectaq'): the same tasks as phase / edge_phase, claimed from a per-launch zeroed counter instead of a
// static CTA slice, so CTAs whose interval work (pinned matrix tiles) finishes early take more tasks. Thread 0
// requests the next ticket before running the current task (the atomic's latency overlaps the task). Every ticket
// below the task count is run exactly once by the CTA that drew it; each task writes only its own outputs (S == 1:
// its C tile; S > 1: its private K-slice partial), so every stored value is independent of the claim order.
template <class RUN>
__device__ __forceinline__ void claim_loop(unsigned* ctr, unsigned ntask, RUN run) {
  __shared__ unsigned smmq_slot;
  __syncthreads();
  if (threadIdx.x == 0) smmq_slot = atomicAdd(ctr, 1u);
  __syncthreads();
  unsigned task = smmq_slot;
  while (task < ntask) {
    unsigned next = 0u;
    if (threadIdx.x == 0) next = atomicAdd(ctr, 1u);
    run((int)task);
    __syncthreads();
    if (threadIdx.x == 0) smmq_slot = next;
    __syncthreads();
    task = smmq_slot;
  }
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void phase_q(const popsc::SArgs g, float* part, int S, int ks, float* smem, unsigned* ctr) {
  const int tm = g.M / BM, tn = g.N / BN, per = g.nb * tm * tn, ntask = S * per;
  const long long total = (long long)g.nb * g.M * g.N;
  claim_loop(ctr, (unsigned)ntask, [&](int task) {
    const int sl = task / per, rest = task % per;
    const int j = rest % tn, r = rest / tn;
    popsc::SArgs h = g;
    h.A = g.A + (long long)sl * ks * g.sak;
    h.B = g.B + (long long)sl * ks * g.sbk;
    h.K = g.K - sl * ks < ks ? g.K - sl * ks : ks;
    h.C = part + (long long)sl * total;
    h.ldc = g.N;
    h.bsc = (long long)g.M * g.N;
    popsc::stile<BM, BN, AK, BKC, KC, NST>(h, r / tm, (r % tm) * BM, j * BN, smem);
  });
}

template <int BM, int BN, bool AK, bool BKC, int KC, int NST>
__device__ __noinline__ void edge_phase_q(const popsc::SArgs g, float* part, int S, int ks, float* smem,
                                          unsigned* ctr) {
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, per = g.nb * tm * tn, ntask = S * per;
  const long long total = (long long)g.nb * g.M * g.N;
  claim_loop(ctr, (unsigned)ntask, [&](int task) {
    const int sl = task / per, rest = task % per;
    const int j = rest % tn, r = rest / tn;
    popsc::SArgs h = g;
    if (S > 1) {
      h.A = g.A + (long long)sl * ks * g.sak;
      h.B = g.B + (long long)sl * ks * g.sbk;
      h.K = g.K - sl * ks < ks ? g.K - sl * ks : ks;
      h.C = part + (long long)sl * total;
      h.ldc = g.N;
      h.bsc = (long long)g.M * g.N;
    }
    stile_edge<BM, BN, AK, BKC, KC, NST>(h, r / tm, (r % tm) * BM, j * BN, smem);
  });
}

}  // namespace smmsplit
