// S101H FP32 SIMT GEMM tile engine.
//
// C_b[m, n] = sum over pairs p of sum_k A_p,b(m, k) B_p,b(k, n), b = batch (e.g. document), fixed per-element order:
// pairs in order, k ascending in steps of GBK, one fmaf chain per output element (per split-K slice). Split-K slices
// are written to a partial buffer and summed in slice order by a separate reduction (deterministic).
//
// Operand layouts: A(m, k) = A[m*sam + k*sak] with AK (sak == 1) or m-contiguous (sam == 1); B(k, n) = B[k*sbk + n*sbn]
// with BKC (sbk == 1) or n-contiguous (sbn == 1). All M, N multiples of the tile, K multiples of GBK, 16-byte aligned.
#pragma once
#include "core.cuh"

namespace s101h {

constexpr int GBK = 16;

template <int BM, int BN>
struct GemmSmem {
  static constexpr int SA = BM + 4, SB = BN + 4;                 // padded k-major rows
  static constexpr int FLOATS = 2 * GBK * SA + 2 * GBK * SB;
};

// Batch b = (b / nb2, b % nb2): operand offset = (b / nb2) * bs + (b % nb2) * bs2 (floats).
struct Pair {
  const float* A; long long sam, sak, bsa, bsa2;
  const float* B; long long sbk, sbn, bsb, bsb2;
  int K;
};

template <int NP>
struct GemmArgs {
  Pair p[NP];
  int M, N, nb, nb2;                                              // per-batch output M x N, nb batches (nb2 inner)
  int ksplit;                                                     // 1 = no split
  float* part;                                                    // [ksplit][nb][M][N] when ksplit > 1
  unsigned* cnt;                                                  // split-K "last arriver" tile counters (or null)
};

DEV long long boff(long long b, int nb2, long long bs, long long bs2) { return (b / nb2) * bs + (b % nb2) * bs2; }

template <int BM, bool AK>
DEV void load_a(const float* __restrict__ A, long long sam, long long sak, int m0, int k, float4 (&r)[BM / 64]) {
  const int t = threadIdx.x;
#pragma unroll
  for (int i = 0; i < BM / 64; ++i) {
    if (AK) {
      const int row = (t >> 2) + 64 * i, kc = (t & 3) * 4;
      r[i] = *reinterpret_cast<const float4*>(A + (long long)(m0 + row) * sam + k + kc);
    } else {
      const int kr = t / (BM / 4 / (BM / 64)) , mc = (t % (BM / 4 / (BM / 64))) * 4;   // 16 k-rows x BM/(BM/64) cols
      const int kk = (BM == 64) ? kr : kr;                          // BM=64: kr in 0..15, mc in 0..60
      const int m = (BM == 64) ? mc : mc + 64 * i;                  // BM=128: two passes over column halves
      r[i] = *reinterpret_cast<const float4*>(A + (long long)(k + kk) * sak + m0 + m);
    }
  }
}

template <int BM, bool AK>
DEV void store_a(float* __restrict__ S, const float4 (&r)[BM / 64]) {
  constexpr int SA = BM + 4;
  const int t = threadIdx.x;
#pragma unroll
  for (int i = 0; i < BM / 64; ++i) {
    if (AK) {
      const int row = (t >> 2) + 64 * i, kc = (t & 3) * 4;
      S[(kc + 0) * SA + row] = r[i].x;
      S[(kc + 1) * SA + row] = r[i].y;
      S[(kc + 2) * SA + row] = r[i].z;
      S[(kc + 3) * SA + row] = r[i].w;
    } else {
      const int kr = t / 16, mc = (t % 16) * 4;
      *reinterpret_cast<float4*>(S + kr * SA + mc + 64 * i) = r[i];
    }
  }
}

// B is loaded exactly like A with (n, k) roles: B(k, n) = B[n*sbn + k*sbk]; BKC == "k-contiguous".
template <int BM, int BN, bool AK, bool BKC, int NP, class Epi>
DEV void gemm_tile(const GemmArgs<NP>& g, int b, int m0, int n0, int kbeg, int kend, float* smem, const Epi& epi,
                   int slice) {
  using SM = GemmSmem<BM, BN>;
  constexpr int TM = BM / 16, TN = BN / 16;
  float* As = smem;
  float* Bs = smem + 2 * GBK * SM::SA;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  gt_stamp(2);
  // flatten the (pair, k) iteration space [kbeg, kend) over the concatenated K
  // Operand descriptors resolved once per tile into registers (compile-time pair indices only: a runtime index into
  // g.p[] forces local-memory reads in every k-step).
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
  auto fetch_to = [&](int kk, float4 (&xa)[BM / 64], float4 (&xb)[BN / 64]) {
    int base = 0;
#pragma unroll
    for (int q = 0; q < NP; ++q) {
      const int lo = base;
      base = kend_p[q];
      if (kk >= lo && kk < kend_p[q]) {
        load_a<BM, AK>(Ab[q], sam[q], sak[q], m0, kk - lo, xa);
        load_a<BN, BKC>(Bb[q], sbn[q], sbk[q], n0, kk - lo, xb);
      }
    }
  };
  auto compute = [&](int buf) {
    const float* as = As + buf * GBK * SM::SA;
    const float* bs = Bs + buf * GBK * SM::SB;
#if defined(S102G_GK1)
    // option gk1: 64x64 tiles (the chain's latency-bound GEMMs) keep the k loop rolled (smaller code footprint)
    constexpr int GK_UNR = (BM == 64 && BN == 64) ? 1 : 4;
#pragma unroll GK_UNR
#elif defined(S102G_GKU)
#pragma unroll 4
#else
#pragma unroll
#endif
    for (int kk = 0; kk < GBK; ++kk) {
      float a[TM], bv[TN];
#pragma unroll
      for (int h = 0; h < TM / 4; ++h) {
        const float4 x = *reinterpret_cast<const float4*>(as + kk * SM::SA + 64 * h + ty * 4);
        a[4 * h + 0] = x.x; a[4 * h + 1] = x.y; a[4 * h + 2] = x.z; a[4 * h + 3] = x.w;
      }
#pragma unroll
      for (int h = 0; h < TN / 4; ++h) {
        const float4 x = *reinterpret_cast<const float4*>(bs + kk * SM::SB + 64 * h + tx * 4);
        bv[4 * h + 0] = x.x; bv[4 * h + 1] = x.y; bv[4 * h + 2] = x.z; bv[4 * h + 3] = x.w;
      }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
    }
  };
#if defined(S101H_PIPE3)
  // three register stages: loads for step s+3 are issued before computing step s (two compute steps of latency cover)
  float4 r1a[BM / 64], r1b[BN / 64], r2a[BM / 64], r2b[BN / 64], r3a[BM / 64], r3b[BN / 64];
  fetch_to(kbeg, r1a, r1b);
  store_a<BM, AK>(As, r1a);
  store_a<BN, BKC>(Bs, r1b);
  if (kbeg + GBK < kend) fetch_to(kbeg + GBK, r1a, r1b);
  if (kbeg + 2 * GBK < kend) fetch_to(kbeg + 2 * GBK, r2a, r2b);
  __syncthreads();
  int buf = 0;
  for (int k = kbeg; k < kend; k += GBK) {
    if (k + 3 * GBK < kend) fetch_to(k + 3 * GBK, r3a, r3b);
    compute(buf);
    if (k + GBK < kend) {
      store_a<BM, AK>(As + (buf ^ 1) * GBK * SM::SA, r1a);
      store_a<BN, BKC>(Bs + (buf ^ 1) * GBK * SM::SB, r1b);
    }
#pragma unroll
    for (int i = 0; i < BM / 64; ++i) { r1a[i] = r2a[i]; r2a[i] = r3a[i]; }
#pragma unroll
    for (int i = 0; i < BN / 64; ++i) { r1b[i] = r2b[i]; r2b[i] = r3b[i]; }
    __syncthreads();
    buf ^= 1;
  }
#else
#if defined(S102G_PFA)
  // option pfa: a task whose K slice is at most PFA_STEPS k-steps (64x64 tiles) issues every global load up front,
  // stages all steps into their own shared buffers, and computes them after ONE barrier (the slice no longer pays one
  // load latency per k-step). Same per-element fmaf order (k ascending): bitwise identical.
  constexpr int PFA_STEPS = 4;
  if (BM == 64 && BN == 64 && kend - kbeg <= PFA_STEPS * GBK) {
    const int ns = (kend - kbeg) / GBK;
    float4 pa[PFA_STEPS][BM / 64], pb[PFA_STEPS][BN / 64];
#pragma unroll
    for (int st = 0; st < PFA_STEPS; ++st)
      if (st < ns) fetch_to(kbeg + st * GBK, pa[st], pb[st]);
    float* As4 = smem;
    float* Bs4 = smem + PFA_STEPS * GBK * SM::SA;
#pragma unroll
    for (int st = 0; st < PFA_STEPS; ++st)
      if (st < ns) {
        store_a<BM, AK>(As4 + st * GBK * SM::SA, pa[st]);
        store_a<BN, BKC>(Bs4 + st * GBK * SM::SB, pb[st]);
      }
    __syncthreads();
#pragma unroll
    for (int st = 0; st < PFA_STEPS; ++st) {
      if (st >= ns) break;
      const float* as = As4 + st * GBK * SM::SA;
      const float* bs = Bs4 + st * GBK * SM::SB;
#pragma unroll
      for (int kk = 0; kk < GBK; ++kk) {
        float a[TM], bv[TN];
#pragma unroll
        for (int h = 0; h < TM / 4; ++h) {
          const float4 x = *reinterpret_cast<const float4*>(as + kk * SM::SA + 64 * h + ty * 4);
          a[4 * h + 0] = x.x; a[4 * h + 1] = x.y; a[4 * h + 2] = x.z; a[4 * h + 3] = x.w;
        }
#pragma unroll
        for (int h = 0; h < TN / 4; ++h) {
          const float4 x = *reinterpret_cast<const float4*>(bs + kk * SM::SB + 64 * h + tx * 4);
          bv[4 * h + 0] = x.x; bv[4 * h + 1] = x.y; bv[4 * h + 2] = x.z; bv[4 * h + 3] = x.w;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
      }
    }
    __syncthreads();
  } else
#endif
  {
  float4 ra[BM / 64], rb[BN / 64];
  fetch_to(kbeg, ra, rb);
  store_a<BM, AK>(As, ra);
  store_a<BN, BKC>(Bs, rb);
  __syncthreads();
  int buf = 0;
  for (int k = kbeg; k < kend; k += GBK) {
    const bool more = k + GBK < kend;
    if (more) fetch_to(k + GBK, ra, rb);
    compute(buf);
    if (more) {
      store_a<BM, AK>(As + (buf ^ 1) * GBK * SM::SA, ra);
      store_a<BN, BKC>(Bs + (buf ^ 1) * GBK * SM::SB, rb);
    }
    __syncthreads();
    buf ^= 1;
  }
  }
#endif
  gt_stamp(3);
  // epilogue (or split-K partial store): rows m0 + 64h + 4ty + i, cols n0 + 64w + 4tx + j
#if defined(S102G_EPIS)
  // option epis: a non-split tile's accumulators go to shared memory first; the epilogue then runs from shared memory
  // with few live registers (epilogues with IEEE division / expf slow-path calls no longer force the 64 accumulators
  // to be preserved across calls). Same values, same epilogue expressions: bitwise identical.
  if (g.ksplit == 1) {
    constexpr int SC = BN + 4;
    static_assert(BM * SC <= 28672, "epis staging exceeds the 112 KB dynamic shared memory");
    float* Cs = smem;
    __syncthreads();                                               // every warp's last compute(buf) read is done
#pragma unroll
    for (int h = 0; h < TM / 4; ++h)
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int w = 0; w < TN / 4; ++w)
          *reinterpret_cast<float4*>(Cs + (64 * h + 4 * ty + i) * SC + 64 * w + 4 * tx) =
              make_float4(acc[4 * h + i][4 * w + 0], acc[4 * h + i][4 * w + 1], acc[4 * h + i][4 * w + 2],
                          acc[4 * h + i][4 * w + 3]);
    __syncthreads();
#pragma unroll 1
    for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
      const int mm = f / (BN / 4), nn = (f % (BN / 4)) * 4;
      epi(b, m0 + mm, n0 + nn, *reinterpret_cast<const float4*>(Cs + mm * SC + nn));
    }
    __syncthreads();                                               // Cs readers done before the next tile's stores
    return;
  }
#endif
#pragma unroll
  for (int h = 0; h < TM / 4; ++h)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int m = m0 + 64 * h + 4 * ty + i;
#pragma unroll
      for (int w = 0; w < TN / 4; ++w) {
        const int n = n0 + 64 * w + 4 * tx;
        const float4 v = make_float4(acc[4 * h + i][4 * w + 0], acc[4 * h + i][4 * w + 1], acc[4 * h + i][4 * w + 2],
                                     acc[4 * h + i][4 * w + 3]);
        if (g.ksplit > 1) {
          *reinterpret_cast<float4*>(g.part + (((long long)slice * g.nb + b) * g.M + m) * g.N + n) = v;
        } else {
          epi(b, m, n, v);
        }
      }
    }
  gt_stamp(4);
  if (g.ksplit > 1 && g.cnt) {
    // last-arriver reduction: the CTA completing a tile's final slice sums all slices in slice order (deterministic;
    // the order does not depend on which slice finished last) and applies the epilogue. No extra grid barrier.
    __shared__ unsigned last_flag;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const int tile = (b * (g.M / BM) + m0 / BM) * (g.N / BN) + n0 / BN;
      const unsigned old = atomicAdd(g.cnt + tile, 1u);
      last_flag = (old == (unsigned)g.ksplit - 1u);
      if (last_flag) atomicExch(g.cnt + tile, 0u);
    }
    __syncthreads();
    if (last_flag) {
      __threadfence();
#pragma unroll
      for (int h = 0; h < TM / 4; ++h)
#if defined(S102G_LAR1)
#pragma unroll 1
#else
#pragma unroll
#endif
        for (int i = 0; i < 4; ++i) {
          const int m = m0 + 64 * h + 4 * ty + i;
#pragma unroll
          for (int w = 0; w < TN / 4; ++w) {
            const int n = n0 + 64 * w + 4 * tx;
            float4 q[8];                                             // issue every slice load first, then sum in order
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
  }
  gt_stamp(5);
  __syncthreads();
}

// One phase: every (batch, tile, slice) task, strided over the grid.
// vbase rotates the task-to-CTA mapping so several independent GEMMs of one phase spread over different CTAs.
template <int BM, int BN, bool AK, bool BKC, int NP, class Epi>
__device__ __noinline__ void gemm_phase(const GemmArgs<NP> g, const Epi epi, float* smem, int vbase = 0) {
  const int tm = g.M / BM, tn = g.N / BN;
  int ktot = 0;
#pragma unroll
  for (int p = 0; p < NP; ++p) ktot += g.p[p].K;
  const int per = ktot / g.ksplit;                                 // generator guarantees divisibility by GBK
  const int ntask = g.nb * tm * tn * g.ksplit;
  const int vb = ((int)blockIdx.x + (int)gridDim.x - (vbase % (int)gridDim.x)) % (int)gridDim.x;
  gt_stamp(1);
  for (int task = vb; task < ntask; task += gridDim.x) {
    int r = task;
    const int s = r % g.ksplit; r /= g.ksplit;
    const int j = r % tn; r /= tn;
    const int i = r % tm; r /= tm;
    const int b = r;
    gemm_tile<BM, BN, AK, BKC, NP, Epi>(g, b, i * BM, j * BN, s * per, (s + 1) * per, smem, epi, s);
  }
  gt_stamp(6);
}

// Split-K reduction + epilogue: v = sum over slices in slice order; one float4 per thread-iteration.
template <class Epi>
__device__ __noinline__ void splitk_phase(const float* __restrict__ part, int ksplit, int nb, int M, int N, const Epi epi) {
  const long long per = (long long)M * N;
  const long long total = (long long)nb * per / 4;
  for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < total; q += (long long)gridDim.x * blockDim.x) {
    const long long e = q * 4;
    const int b = (int)(e / per);
    const long long r = e % per;
    const int m = (int)(r / N), n = (int)(r % N);
    float4 v = *reinterpret_cast<const float4*>(part + e);
    for (int s = 1; s < ksplit; ++s) {
      const float4 w = *reinterpret_cast<const float4*>(part + (long long)s * nb * per + e);
      v.x += w.x; v.y += w.y; v.z += w.z; v.w += w.w;
    }
    epi(b, m, n, v);
  }
}

}  // namespace s101h
