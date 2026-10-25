// P-OPS2-A FP32 SIMT sliced-K matrix engine for the latch-ops operation compiler (new code).
//
// C_b[m, n] = sum_k A_b(m, k) B_b(k, n), strict FP32 (fmaf; no TF32, tensor cores or floating atomics).
// A CTA (256 threads) owns one BM x BN output tile. Its threads form S slices of 256/S threads; each slice thread
// accumulates an 8 x 8 block (rows ty*4+{0..3} and BM/2+ty*4+{0..3}, columns likewise), so every shared fragment load
// feeds 64 FMAs: an H100 warp-wide LDS.128 occupies the shared data path for ~3-4 cycles even when broadcast, and
// smaller register tiles are shared-memory bound. Slices divide each BK-deep stage: slice q consumes k rows
// [q*BK/S, (q+1)*BK/S) of every stage, ascending, as one fmaf chain started at 0.0f. Slice partials are then summed
// in fixed order ((p0 + p1) + p2) + ... through shared memory by all 256 threads, so the result is a deterministic function of the
// inputs, independent of scheduling (the numerical class of a fixed-order split-K). With S == 1 and KSPLIT == 1 the
// value is the single ascending chain of the Stage 3 / P-OPS-B bodies. Cross-CTA split-K (KSPLIT > 1) is applied on
// top with the P-OPS-B last-arriver fixed-order reduction of the CTA results.
// Operands are staged k-major ([k][m], [k][n]) in an NST-deep cp.async ring (16-byte copies for m/n-contiguous
// operands; 4-byte transposing copies for k-contiguous ones, arranged 8 k x 4 rows per warp over 32 distinct banks
// since leading dimensions are 4 mod 32 words). One CTA barrier per stage.
#ifndef POPS2_FP32_SLICE_CUH
#define POPS2_FP32_SLICE_CUH

namespace pops2s {

template <int NB2, long long S1, long long S2>
__device__ __forceinline__ long long boff(int b) {
  return (long long)(b / NB2) * S1 + (long long)(b % NB2) * S2;
}
template <int XI, long long S0, long long S1>
__device__ __forceinline__ long long cs(int x) {
  return (long long)(x / XI) * S1 + (long long)(x % XI) * S0;
}
__device__ __forceinline__ void cp4(float* dst, const float* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16(float* dst, const float* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

template <int BM, int BN, int S, int BK, int NST>
struct SliceShape {
  static constexpr int TPS = 256 / S, TY = BM / 8, TX = BN / 8, WX = TX / 8, KPS = BK / S;
  static constexpr int LDA = BM + 4, LDB = BN + 4;
  static constexpr int AF = BK * LDA, BF = BK * LDB, STAGE = AF + BF;
  static constexpr int RING = NST * STAGE, RED = S * BM * BN;
  static constexpr int FLOATS = RING > RED ? RING : RED;  // reduction reuses the drained ring
  static_assert(S == 1 || S == 2 || S == 4 || S == 8, "slices");
  static_assert(TY * TX == TPS && TX % 8 == 0 && TY % 4 == 0, "8x8 per thread in 4x8-lane warps");
  static_assert(BK % S == 0 && KPS >= 1, "each slice takes whole k rows of every stage");
  static_assert((BM * BK) % 256 == 0 && (BN * BK) % 256 == 0 && (BM * BK / 4) % 256 == 0 && (BN * BK / 4) % 256 == 0,
                "whole copies per thread");
};

template <int BM, int BN, int S, int BK, int NST, bool AK, bool BKC, class L>
__device__ __forceinline__ void slice_phase(const float* __restrict__ A0, const float* __restrict__ B0,
                                            float* __restrict__ C0, float* __restrict__ part, unsigned* cnt, int vbase,
                                            float* smem) {
  using Sh = SliceShape<BM, BN, S, BK, NST>;
  constexpr int TPS = Sh::TPS, PER = L::K / L::KSPLIT, NK = PER / BK;
  static_assert(NST >= 2 && PER % BK == 0 && NK >= 1, "k slice must be whole stages");
  __shared__ unsigned last_flag;
  const int t = threadIdx.x, q = t / TPS, st = t % TPS, lane = st & 31, warp = st >> 5;
  const int tx = (warp % Sh::WX) * 8 + (lane & 7), ty = (warp / Sh::WX) * 4 + (lane >> 3);
  const int ar = ty * 4, bc = tx * 4;
  const int G = (int)gridDim.x;
  const int start = ((int)blockIdx.x + G - vbase % G) % G;
  for (int task = start; task < L::TASKS; task += G) {
    int r = task;
    const int s = r % L::KSPLIT; r /= L::KSPLIT;
    const int tn = r % L::TILES_N; r /= L::TILES_N;
    const int tm = r % L::TILES_M;
    const int b = r / L::TILES_M;
    const int m0 = tm * BM, n0 = tn * BN, kbeg = s * PER;
    const float* __restrict__ A = A0 + boff<L::NB2, L::BSA, L::BSA2>(b) + cs<L::MIA, (AK ? L::SAM : 1), L::SAM1>(m0);
    const float* __restrict__ B = B0 + boff<L::NB2, L::BSB, L::BSB2>(b) + cs<L::NIB, (BKC ? L::SBN : 1), L::SBN1>(n0);
    auto load = [&](int slot, int k) {
      float* As = smem + slot * Sh::STAGE;
      float* Bs = As + Sh::AF;
      const float* Ak = A + cs<L::KIA, (AK ? 1 : L::SAK), L::SAK1>(k);
      const float* Bk = B + cs<L::KIB, (BKC ? 1 : L::SBK), L::SBK1>(k);
      if constexpr (AK) {
#pragma unroll
        for (int c = t; c < BM * BK; c += 256) {
          const int rest = c >> 5, kk = (rest % (BK / 8)) * 8 + (c & 7), mm = (rest / (BK / 8)) * 4 + ((c >> 3) & 3);
          cp4(As + kk * Sh::LDA + mm, Ak + (long long)mm * L::SAM + kk);
        }
      } else {
#pragma unroll
        for (int c = t; c < BM * BK / 4; c += 256) {
          const int kr = c / (BM / 4), mc = c % (BM / 4);
          cp16(As + kr * Sh::LDA + mc * 4, Ak + (long long)kr * L::SAK + mc * 4);
        }
      }
      if constexpr (BKC) {
#pragma unroll
        for (int c = t; c < BN * BK; c += 256) {
          const int rest = c >> 5, kk = (rest % (BK / 8)) * 8 + (c & 7), nn = (rest / (BK / 8)) * 4 + ((c >> 3) & 3);
          cp4(Bs + kk * Sh::LDB + nn, Bk + (long long)nn * L::SBN + kk);
        }
      } else {
#pragma unroll
        for (int c = t; c < BN * BK / 4; c += 256) {
          const int kr = c / (BN / 4), nc = c % (BN / 4);
          cp16(Bs + kr * Sh::LDB + nc * 4, Bk + (long long)kr * L::SBK + nc * 4);
        }
      }
    };
    float acc[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
#pragma unroll
    for (int p = 0; p < NST - 1; ++p) {
      if (p < NK) load(p, kbeg + p * BK);
      commit();
    }
#pragma unroll 1
    for (int it = 0; it < NK; ++it) {
      wait_group<NST - 2>();
      __syncthreads();  // stage `it` visible; stage it-1 fully consumed by every slice
      {
        const int nx = it + NST - 1;
        if (nx < NK) load(nx % NST, kbeg + nx * BK);
        commit();
      }
      const float* As = smem + (it % NST) * Sh::STAGE + q * Sh::KPS * Sh::LDA;
      const float* Bs = smem + (it % NST) * Sh::STAGE + Sh::AF + q * Sh::KPS * Sh::LDB;
      float fa[2][8], fb[2][8];
      auto frag = [&](int k, int p) {
        const float4 a0 = *reinterpret_cast<const float4*>(As + k * Sh::LDA + ar);
        const float4 a1 = *reinterpret_cast<const float4*>(As + k * Sh::LDA + ar + BM / 2);
        const float4 b0 = *reinterpret_cast<const float4*>(Bs + k * Sh::LDB + bc);
        const float4 b1 = *reinterpret_cast<const float4*>(Bs + k * Sh::LDB + bc + BN / 2);
        fa[p][0] = a0.x; fa[p][1] = a0.y; fa[p][2] = a0.z; fa[p][3] = a0.w;
        fa[p][4] = a1.x; fa[p][5] = a1.y; fa[p][6] = a1.z; fa[p][7] = a1.w;
        fb[p][0] = b0.x; fb[p][1] = b0.y; fb[p][2] = b0.z; fb[p][3] = b0.w;
        fb[p][4] = b1.x; fb[p][5] = b1.y; fb[p][6] = b1.z; fb[p][7] = b1.w;
      };
      frag(0, 0);
#pragma unroll
      for (int k = 0; k < Sh::KPS; ++k) {
        if (k + 1 < Sh::KPS) frag(k + 1, (k + 1) & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(fa[k & 1][i], fb[k & 1][j], acc[i][j]);
      }
    }
    wait_group<0>();
    float* C = C0 + boff<L::NB2, L::BSC, L::BSC2>(b);
    if constexpr (S == 1 && L::KSPLIT == 1) {
      // Single chain: store straight from registers.
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float* Cm = C + cs<L::MIC, L::SCM, L::SCM1>(m0 + (i >> 2) * (BM / 2) + ar + (i & 3));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (L::SCN == 1) {
            *reinterpret_cast<float4*>(Cm + cs<L::NIC, 1, L::SCN1>(n0 + h * (BN / 2) + bc)) =
                make_float4(acc[i][4 * h + 0], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
          } else {
#pragma unroll
            for (int j = 0; j < 4; ++j) Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + h * (BN / 2) + bc + j)] = acc[i][4 * h + j];
          }
        }
      }
    } else {
      // Cooperative epilogue: every slice writes its 8x8 block into a row-major BM x BN shared tile; all 256 threads
      // then sum slices in fixed order 0..S-1 (float4 lanes), and, for cross-CTA split-K, the CTA results in fixed
      // order 0..KSPLIT-1 with all KSPLIT loads of an element group in flight together.
      __syncthreads();  // ring drained by every slice
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float* row = smem + q * (BM * BN) + ((i >> 2) * (BM / 2) + ar + (i & 3)) * BN;
        *reinterpret_cast<float4*>(row + bc) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(row + bc + BN / 2) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
      }
      __syncthreads();
      constexpr int V = BM * BN / 4 / 256;  // float4 element groups per thread
      static_assert(V * 1024 == BM * BN, "whole float4 groups per thread");
      float4 v[V];
#pragma unroll
      for (int u = 0; u < V; ++u) {
        const int e = (u * 256 + t) * 4;
        float4 x = *reinterpret_cast<const float4*>(smem + e);
#pragma unroll
        for (int p = 1; p < S; ++p) {
          const float4 y = *reinterpret_cast<const float4*>(smem + p * (BM * BN) + e);
          x.x += y.x; x.y += y.y; x.z += y.z; x.w += y.w;
        }
        v[u] = x;
      }
      bool store = true;
      if constexpr (L::KSPLIT > 1) {
        const int tile = (b * L::TILES_M + tm) * L::TILES_N + tn;
        float* P = part + (long long)tile * L::KSPLIT * (BM * BN);
#pragma unroll
        for (int u = 0; u < V; ++u) __stcg(reinterpret_cast<float4*>(P + (long long)s * (BM * BN) + (u * 256 + t) * 4), v[u]);
        // CTA barrier orders all partial stores before thread 0's gpu-scope acq_rel arrival (cumulative release);
        // the last arriver's acquire plus the following barrier orders its loads (P-OPS-B relacq scheme).
        __syncthreads();
        if (t == 0) {
          unsigned old;
          asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt + tile) : "memory");
          last_flag = old == (unsigned)L::KSPLIT - 1u;
          if (last_flag) atomicExch(cnt + tile, 0u);
        }
        __syncthreads();
        store = last_flag;
        if (store) {
#pragma unroll
          for (int u = 0; u < V; ++u) {
            const int e = (u * 256 + t) * 4;
            float4 y[L::KSPLIT];
#pragma unroll
            for (int ss = 0; ss < L::KSPLIT; ++ss)
              y[ss] = ss == s ? v[u] : __ldcg(reinterpret_cast<const float4*>(P + (long long)ss * (BM * BN) + e));
            float4 x = y[0];
#pragma unroll
            for (int ss = 1; ss < L::KSPLIT; ++ss) { x.x += y[ss].x; x.y += y[ss].y; x.z += y[ss].z; x.w += y[ss].w; }
            v[u] = x;
          }
        }
      }
      if (store) {
#pragma unroll
        for (int u = 0; u < V; ++u) {
          const int e = (u * 256 + t) * 4, rr = e / BN, cc = e % BN;
          float* Cm = C + cs<L::MIC, L::SCM, L::SCM1>(m0 + rr);
          if constexpr (L::SCN == 1) {
            *reinterpret_cast<float4*>(Cm + cs<L::NIC, 1, L::SCN1>(n0 + cc)) = v[u];
          } else {
            Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + cc + 0)] = v[u].x; Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + cc + 1)] = v[u].y;
            Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + cc + 2)] = v[u].z; Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + cc + 3)] = v[u].w;
          }
        }
      }
    }
    __syncthreads();  // ring / reduction storage and last_flag are reused by the next task
  }
}

}  // namespace pops2s
#endif
