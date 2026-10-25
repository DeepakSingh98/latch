// S105A option bst (policy B16 with BF16 storage of GEMM operands): C_b[m, n] = sum over pairs p, k of A_p,b(m, k)
// B_p,b(k, n) with BOTH operands stored as BF16 in global memory (weights cast once per update at the stage phase;
// activations written as BF16 copies by their producers). Tiles are copied with cp.async into NST-stage BF16 shared
// buffers in their natural layout and fragments are formed with ldmatrix (.trans for m- / n-contiguous operands).
// Arithmetic: one mma.sync m16n8k16 BF16 product per k16 accumulated in FP32, k ascending -- for every output element
// the same MMA sequence on the same BF16 values as option bg (which rounds the FP32 operands to BF16 when forming
// fragments), hence bitwise equal to bg for the same split-K partition. Same GemmArgs / epilogue / split-K interface
// as tgemm.cuh (pair pointers carry BF16 buffers; strides in elements).
// [S109A component copy of the pinned donor header: identical code unless a timing-only diagnostic define
// S109A_DIAG_NOSTORE / S109A_DIAG_NOSIDE is set; then, while an engine epilogue runs (s109a_diag_epi), the helpers skip
// global stores / return a constant instead of loading the side input. Timing-only; results are invalid.]
#pragma once
#include "tgemm.cuh"

namespace s101h {

typedef __nv_bfloat16 bf16;

DEV void bldsm4(uint32_t (&r)[4], const bf16* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
DEV void bldsm4t(uint32_t (&r)[4], const bf16* p) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
DEV void st4bf(bf16* p, float4 v) { S109A_DIAG_ST(*reinterpret_cast<uint2*>(p) = make_uint2(bpk(v.x, v.y), bpk(v.z, v.w))); }

// FP32 -> BF16 copy (round to nearest), grid-stride, n a multiple of 4
__device__ __noinline__ void cast_bf16(long long n, const float* __restrict__ src, bf16* __restrict__ dst) {
  const long long step = (long long)gridDim.x * NTHREADS * 4;
  for (long long i = ((long long)blockIdx.x * NTHREADS + threadIdx.x) * 4; i < n; i += step)
    st4bf(dst + i, *reinterpret_cast<const float4*>(src + i));
}

struct EpiSilu2B {                     // HP = v, HA = silu(v), HAb = bf16(HA)
  float* HP; float* HA; bf16* HAb; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    st4(HP + o, v);
    const float4 a = make_float4(silu_(v.x), silu_(v.y), silu_(v.z), silu_(v.w));
    st4(HA + o, a);
    st4bf(HAb + o, a);
  }
};
struct EpiSilu2Bo {                    // HP = v, HAb = bf16(silu(v)) (FP32 HA not needed)
  float* HP; bf16* HAb; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    st4(HP + o, v);
    st4bf(HAb + o, make_float4(silu_(v.x), silu_(v.y), silu_(v.z), silu_(v.w)));
  }
};
struct EpiDSiluB {                     // dHP = v * silu'(HP), and its BF16 copy
  const float* HP; float* DH; bf16* DHb; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    const float4 z = ld4(HP + o);
    const float4 r = make_float4(v.x * silu1_(z.x), v.y * silu1_(z.y), v.z * silu1_(z.z), v.w * silu1_(z.w));
    st4(DH + o, r);
    st4bf(DHb + o, r);
  }
};
struct EpiStoreB {                     // C = v and its BF16 copy
  float* C; bf16* Cb; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    st4(C + o, v);
    st4bf(Cb + o, v);
  }
};

template <int BM, int BN, bool AK, bool BKC, int KS>
struct BSmem {
  static constexpr int A_EL = AK ? BM * (KS + 8) : KS * (BM + 8);
  static constexpr int B_EL = BKC ? BN * (KS + 8) : KS * (BN + 8);
  static constexpr int STAGE = A_EL + B_EL;                         // BF16 elements
#if defined(S106F_BST4)
  static constexpr int NST = 4;                                     // S106F option bst4: four cp.async stages
#else
  static constexpr int NST = 3;
#endif
  static constexpr int CS = BN + 8;                                 // staged FP32 result row stride
};

template <int R, bool RK, int KS>
DEV void bgl(const bf16* __restrict__ X, long long sr, long long sk, int r0, int k, bf16* __restrict__ S) {
  constexpr int CH = R * KS / 8;
  static_assert(CH % NTHREADS == 0, "bst tile / k step");
#pragma unroll
  for (int i = 0; i < CH / NTHREADS; ++i) {
    const int c = threadIdx.x + i * NTHREADS;
    if (RK) {
      const int r = c / (KS / 8), kc = (c % (KS / 8)) * 8;
      cpa16(reinterpret_cast<float*>(S + r * (KS + 8) + kc), reinterpret_cast<const float*>(X + (long long)(r0 + r) * sr + k + kc));
    } else {
      const int kr = c / (R / 8), rc = (c % (R / 8)) * 8;
      cpa16(reinterpret_cast<float*>(S + kr * (R + 8) + rc), reinterpret_cast<const float*>(X + (long long)(k + kr) * sk + r0 + rc));
    }
  }
}

// one k stage (k index kk of the concatenated pair K) of tile (b, m0, n0) -> stage buffer st
template <int BM, int BN, bool AK, bool BKC, int NP, int KS>
DEV void bg_stage(const GemmArgs<NP>& g, int b, int m0, int n0, int kk, int st, bf16* S) {
  using SM = BSmem<BM, BN, AK, BKC, KS>;
  bf16* As = S + st * SM::STAGE;
  bf16* Bs = As + SM::A_EL;
  int base = 0;
#pragma unroll
  for (int q = 0; q < NP; ++q) {
    const int lo = base;
    base += g.p[q].K;
    if (kk >= lo && kk < base) {
      bgl<BM, AK, KS>(reinterpret_cast<const bf16*>(g.p[q].A) + boff(b, g.nb2, g.p[q].bsa, g.p[q].bsa2), g.p[q].sam,
                      g.p[q].sak, m0, kk - lo, As);
      bgl<BN, BKC, KS>(reinterpret_cast<const bf16*>(g.p[q].B) + boff(b, g.nb2, g.p[q].bsb, g.p[q].bsb2), g.p[q].sbn,
                       g.p[q].sbk, n0, kk - lo, Bs);
    }
  }
}
template <int BM, int BN, bool AK, bool BKC, int NP, int KS>
DEV void bg_prologue(const GemmArgs<NP>& g, int b, int m0, int n0, int kbeg, int kend, bf16* S) {
  const int nk = (kend - kbeg) / KS;
#pragma unroll
  for (int s = 0; s < BSmem<BM, BN, AK, BKC, KS>::NST - 1; ++s) {
    if (s < nk) bg_stage<BM, BN, AK, BKC, NP, KS>(g, b, m0, n0, kbeg + s * KS, s, S);
    cpa_commit();
  }
}

// OV (option bov): the result is staged outside the stage ring, and the next tile's prologue loads (nb_, nm_, nn_,
// nkb_, nke_; nb_ < 0: none) are issued before this tile's epilogue; `pre`: this tile's prologue was already issued
template <int BM, int BN, bool AK, bool BKC, int NP, int KS, class Epi, bool OV = false>
DEV void bgemm_tile(const GemmArgs<NP>& g, int b, int m0, int n0, int kbeg, int kend, float* smem, const Epi& epi,
                    int slice, bool pre = false, int nb_ = -1, int nm_ = 0, int nn_ = 0, int nkb_ = 0, int nke_ = 0) {
  using SM = BSmem<BM, BN, AK, BKC, KS>;
  constexpr int WN = BN >= 128 ? 4 : (BM >= 128 ? 2 : 4), WM = 8 / WN;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(MT >= 1 && NT >= 2 && NT % 2 == 0 && WTM % 16 == 0, "bst warp tile");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  bf16* S = reinterpret_cast<bf16*>(smem);
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  auto load_stage = [&](int kk, int st) { bg_stage<BM, BN, AK, BKC, NP, KS>(g, b, m0, n0, kk, st, S); };
  const int nk = (kend - kbeg) / KS;
  if (!pre) bg_prologue<BM, BN, AK, BKC, NP, KS>(g, b, m0, n0, kbeg, kend, S);
  for (int it = 0; it < nk; ++it) {
    cpa_wait<SM::NST - 2>();
    __syncthreads();                                               // stage it visible; stage it-1 fully consumed
    if (it + SM::NST - 1 < nk) load_stage(kbeg + (it + SM::NST - 1) * KS, (it + SM::NST - 1) % SM::NST);
    cpa_commit();
    const bf16* As = S + (it % SM::NST) * SM::STAGE;
    const bf16* Bs = As + SM::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i + (((lane >> 3) & 1) << 3));
      }
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t bb[4];
        if (BKC) bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
          bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
        }
      }
    }
  }
  cpa_wait<0>();
  __syncthreads();                                                 // every warp's last fragment read is done
  float* Cs = OV ? reinterpret_cast<float*>(S + SM::NST * SM::STAGE) : smem;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = wm0 + i * 16 + gq, n = wn0 + j * 8 + 2 * tq;
      *reinterpret_cast<float2*>(Cs + m * SM::CS + n) = make_float2(acc[i][j][0], acc[i][j][1]);
      *reinterpret_cast<float2*>(Cs + (m + 8) * SM::CS + n) = make_float2(acc[i][j][2], acc[i][j][3]);
    }
  __syncthreads();
  if (OV && nb_ >= 0) bg_prologue<BM, BN, AK, BKC, NP, KS>(g, nb_, nm_, nn_, nkb_, nke_, S);   // next tile's first stages
  if (g.ksplit == 1) {
#if defined(S103F_EPF)
    for (int f = threadIdx.x; f < 2 * NTHREADS && f < BM * BN / 4; f += NTHREADS)
      epi_pre(epi, b, m0 + f / (BN / 4), n0 + (f % (BN / 4)) * 4);
#endif
#pragma unroll 1
    for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
      const int mm = f / (BN / 4), nn = (f % (BN / 4)) * 4;
#if defined(S103F_EPF)
      const int f2 = f + 2 * NTHREADS;
      if (f2 < BM * BN / 4) epi_pre(epi, b, m0 + f2 / (BN / 4), n0 + (f2 % (BN / 4)) * 4);
#endif
      epi(b, m0 + mm, n0 + nn, *reinterpret_cast<const float4*>(Cs + mm * SM::CS + nn));
    }
    __syncthreads();
    return;
  }
#pragma unroll 1
  for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
    const int mm = f / (BN / 4), nn = (f % (BN / 4)) * 4;
    *reinterpret_cast<float4*>(g.part + (((long long)slice * g.nb + b) * g.M + m0 + mm) * g.N + n0 + nn) =
        *reinterpret_cast<const float4*>(Cs + mm * SM::CS + nn);
  }
  if (g.cnt) {
    __shared__ unsigned bg_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const int tile = (b * (g.M / BM) + m0 / BM) * (g.N / BN) + n0 / BN;
      const unsigned old = atomicAdd(g.cnt + tile, 1u);
      bg_last = (old == (unsigned)g.ksplit - 1u);
      if (bg_last) atomicExch(g.cnt + tile, 0u);
    }
    __syncthreads();
    if (bg_last) {
      __threadfence();
#pragma unroll 1
      for (int f = threadIdx.x; f < BM * BN / 4; f += NTHREADS) {
        const int m = m0 + f / (BN / 4), n = n0 + (f % (BN / 4)) * 4;
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

template <int BM, int BN, bool AK, bool BKC, int NP, int KS, bool OV = false, class Epi>
__device__ __noinline__ void bgemm_phase(const GemmArgs<NP> g, const Epi epi, float* smem, int vbase = 0) {
  const int tm = g.M / BM, tn = g.N / BN;
  int ktot = 0;
#pragma unroll
  for (int p = 0; p < NP; ++p) ktot += g.p[p].K;
  const int per = ktot / g.ksplit;
  const int ntask = g.nb * tm * tn * g.ksplit;
  const int vb = ((int)blockIdx.x + (int)gridDim.x - (vbase % (int)gridDim.x)) % (int)gridDim.x;
  bool pre = false;
  for (int task = vb; task < ntask; task += gridDim.x) {
    int r = task;
    const int s = r % g.ksplit; r /= g.ksplit;
    const int j = r % tn; r /= tn;
    const int i = r % tm; r /= tm;
    const int b = r;
    int nb_ = -1, nm_ = 0, nn_ = 0, ns_ = 0;
    if (OV && task + (int)gridDim.x < ntask) {
      int q = task + (int)gridDim.x;
      ns_ = q % g.ksplit; q /= g.ksplit;
      nn_ = (q % tn) * BN; q /= tn;
      nm_ = (q % tm) * BM; q /= tm;
      nb_ = q;
    }
    bgemm_tile<BM, BN, AK, BKC, NP, KS, Epi, OV>(g, b, i * BM, j * BN, s * per, (s + 1) * per, smem, epi, s, pre, nb_, nm_,
                                                 nn_, ns_ * per, (ns_ + 1) * per);
    pre = OV && nb_ >= 0;
  }
}

}  // namespace s101h
