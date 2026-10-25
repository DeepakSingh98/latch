// P-OPS2-A FP32 SIMT matrix engine for the latch-ops operation compiler (new code).
//
// C_b[m, n] = sum_k A_b(m, k) B_b(k, n), strict FP32: one fmaf chain per output element and per split-K slice,
// k ascending, starting from 0.0f; no TF32, no tensor cores, no floating atomics. With KSPLIT > 1 each slice stores
// its partial tile; the CTA completing a tile's last slice sums slices in fixed order 0..KSPLIT-1 (P-OPS-B scheme),
// so the result never depends on arrival order. With KSPLIT == 1 the value equals the P-OPS-B / Stage 3 bodies.
// LATCH-LOOP-EXECUTOR: optional ACC adds the completed FP32 result to its destination in the final store;
// it never seeds the FMA chain or split-slice sum. Default-false calls retain the ordinary store behavior.
//
// Execution (differs from P-OPS-B fp32_matrix.cuh, same layout struct L and call signature):
//  * both operands are staged k-major in shared memory (A as [k][m], B as [k][n]); k-contiguous operands are
//    transposed while storing from registers, m/n-contiguous operands are stored as float4;
//  * a 16x16 thread grid (4x8-lane warp blocks) owns TM x TN outputs (TM = BM/16, TN = BN/16), split into 4-wide
//    groups BM/2 (BN/2) apart for 8-wide tiles, so every fragment LDS.128 is one shared wavefront;
//  * one k-tile of global loads is prefetched into registers while the current shared tile is consumed
//    (two shared buffers, one CTA barrier per k-tile), fragments are double-buffered in registers.
// Layout admission is the P-OPS-B contract (two-factor m/n/k indices, 16-byte aligned rows/columns).
#ifndef POPS2_FP32_SIMT_CUH
#define POPS2_FP32_SIMT_CUH

namespace pops2 {

template <int NB2, long long S1, long long S2>
__device__ __forceinline__ long long boff(int b) {
  return (long long)(b / NB2) * S1 + (long long)(b % NB2) * S2;
}
template <int XI, long long S0, long long S1>
__device__ __forceinline__ long long cs(int x) {
  return (long long)(x / XI) * S1 + (long long)(x % XI) * S0;
}

template <int BM, int BN, int BK, bool AK, bool BKC>
struct SimtShape {
  static constexpr int TM = BM / 16, TN = BN / 16;
  static constexpr int LDA = BM + (AK ? 4 : 0);   // transposed stores: pad rotates banks between k rows
  static constexpr int LDB = BN + (BKC ? 4 : 0);
  static constexpr int AF = BK * LDA, BF = BK * LDB, STAGE = AF + BF;
  static constexpr int AV = (BM * BK / 4 + 255) / 256;  // float4 global loads per thread per k-tile
  static constexpr int BV = (BN * BK / 4 + 255) / 256;
  static constexpr bool AFULL = (BM * BK / 4) % 256 == 0, BFULL = (BN * BK / 4) % 256 == 0;
  // Row (column) of fragment element i of thread y: 4-groups BM/2 apart for 8-wide fragments.
  __device__ static __forceinline__ int row(int ty, int i) {
    return TM == 8 ? (i >> 2) * (BM / 2) + ty * 4 + (i & 3) : ty * TM + i;
  }
  __device__ static __forceinline__ int col(int tx, int j) {
    return TN == 8 ? (j >> 2) * (BN / 2) + tx * 4 + (j & 3) : tx * TN + j;
  }
};

#if defined(POPS2_NOINLINE)
#define POPS2_PHASE_FN __device__ __noinline__
#else
#define POPS2_PHASE_FN __device__ __forceinline__
#endif

template <int N>
__device__ __forceinline__ void frag_load(float (&dst)[N], const float* src, int half_stride) {
  if constexpr (N == 8) {
    const float4 x = *reinterpret_cast<const float4*>(src);
    const float4 y = *reinterpret_cast<const float4*>(src + half_stride);
    dst[0] = x.x; dst[1] = x.y; dst[2] = x.z; dst[3] = x.w; dst[4] = y.x; dst[5] = y.y; dst[6] = y.z; dst[7] = y.w;
  } else if constexpr (N == 4) {
    const float4 x = *reinterpret_cast<const float4*>(src);
    dst[0] = x.x; dst[1] = x.y; dst[2] = x.z; dst[3] = x.w;
  } else {
    const float2 x = *reinterpret_cast<const float2*>(src);
    dst[0] = x.x; dst[1] = x.y;
  }
}

// Runtime-or-constant index helpers: with compile-time layout structs (inlined) these fold to constants.
__device__ __forceinline__ long long rboff(int b, int nb2, long long s1, long long s2) {
  return (long long)(b / nb2) * s1 + (long long)(b % nb2) * s2;
}
__device__ __forceinline__ long long rcs(int x, int xi, long long s0, long long s1) {
  return (long long)(x / xi) * s1 + (long long)(x % xi) * s0;
}

template <int BM, int BN, int BK, bool AK, bool BKC, class L, bool ACC = false>
__device__ __forceinline__ void simt_body(const L& lay, const float* __restrict__ A0, const float* __restrict__ B0, float* __restrict__ C0,
                               float* __restrict__ part, unsigned* cnt, int vbase, float* smem) {
  using S = SimtShape<BM, BN, BK, AK, BKC>;
  constexpr int TM = S::TM, TN = S::TN;
  static_assert((BM == 32 || BM == 64 || BM == 128) && (BN == 32 || BN == 64 || BN == 128), "unsupported tile");
  static_assert(BK == 8 || BK == 16, "k-tile depth");
  const int PER = lay.K / lay.KSPLIT, NK = PER / BK;  // generator guarantees PER % BK == 0, NK >= 1
  __shared__ unsigned last_flag;
  // Logical 16x16 thread grid; each warp owns a 4 (ty) x 8 (tx) block so a fragment LDS.128 touches 4 (A) or
  // 8 (B) distinct float4 = one shared wavefront each (a 2 x 16 warp block needs two for B).
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int tx = (warp & 1) * 8 + (lane & 7), ty = (warp >> 1) * 4 + (lane >> 3);
  const int G = (int)gridDim.x;
  const int start = ((int)blockIdx.x + G - vbase % G) % G;
  for (int task = start; task < lay.TASKS; task += G) {
    int r = task;
    const int s = r % lay.KSPLIT; r /= lay.KSPLIT;
    const int tn = r % lay.TILES_N; r /= lay.TILES_N;
    const int tm = r % lay.TILES_M;
    const int b = r / lay.TILES_M;
    const int m0 = tm * BM, n0 = tn * BN, kbeg = s * PER;
    const float* __restrict__ A = A0 + rboff(b, lay.NB2, lay.BSA, lay.BSA2) + rcs(m0, lay.MIA, (AK ? lay.SAM : 1), lay.SAM1);
    const float* __restrict__ B = B0 + rboff(b, lay.NB2, lay.BSB, lay.BSB2) + rcs(n0, lay.NIB, (BKC ? lay.SBN : 1), lay.SBN1);
    float4 ra[S::AV], rb[S::BV];
    auto gload = [&](int k) {
      const float* Ak = A + rcs(k, lay.KIA, (AK ? 1 : lay.SAK), lay.SAK1);
      const float* Bk = B + rcs(k, lay.KIB, (BKC ? 1 : lay.SBK), lay.SBK1);
#pragma unroll
      for (int v = 0; v < S::AV; ++v) {
        const int c = t + v * 256;
        if (S::AFULL || c < BM * BK / 4) {
          if constexpr (AK) ra[v] = __ldg(reinterpret_cast<const float4*>(Ak + (long long)(c / (BK / 4)) * lay.SAM + (c % (BK / 4)) * 4));
          else ra[v] = __ldg(reinterpret_cast<const float4*>(Ak + (long long)(c / (BM / 4)) * lay.SAK + (c % (BM / 4)) * 4));
        }
      }
#pragma unroll
      for (int v = 0; v < S::BV; ++v) {
        const int c = t + v * 256;
        if (S::BFULL || c < BN * BK / 4) {
          if constexpr (BKC) rb[v] = __ldg(reinterpret_cast<const float4*>(Bk + (long long)(c / (BK / 4)) * lay.SBN + (c % (BK / 4)) * 4));
          else rb[v] = __ldg(reinterpret_cast<const float4*>(Bk + (long long)(c / (BN / 4)) * lay.SBK + (c % (BN / 4)) * 4));
        }
      }
    };
    auto sstore = [&](int slot) {
      float* As = smem + slot * S::STAGE;
      float* Bs = As + S::AF;
#pragma unroll
      for (int v = 0; v < S::AV; ++v) {
        const int c = t + v * 256;
        if (S::AFULL || c < BM * BK / 4) {
          if constexpr (AK) {
            const int m = c / (BK / 4), k = (c % (BK / 4)) * 4;
            As[(k + 0) * S::LDA + m] = ra[v].x; As[(k + 1) * S::LDA + m] = ra[v].y;
            As[(k + 2) * S::LDA + m] = ra[v].z; As[(k + 3) * S::LDA + m] = ra[v].w;
          } else {
            *reinterpret_cast<float4*>(As + (c / (BM / 4)) * S::LDA + (c % (BM / 4)) * 4) = ra[v];
          }
        }
      }
#pragma unroll
      for (int v = 0; v < S::BV; ++v) {
        const int c = t + v * 256;
        if (S::BFULL || c < BN * BK / 4) {
          if constexpr (BKC) {
            const int n = c / (BK / 4), k = (c % (BK / 4)) * 4;
            Bs[(k + 0) * S::LDB + n] = rb[v].x; Bs[(k + 1) * S::LDB + n] = rb[v].y;
            Bs[(k + 2) * S::LDB + n] = rb[v].z; Bs[(k + 3) * S::LDB + n] = rb[v].w;
          } else {
            *reinterpret_cast<float4*>(Bs + (c / (BN / 4)) * S::LDB + (c % (BN / 4)) * 4) = rb[v];
          }
        }
      }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    const int arow = S::row(ty, 0), bcol = S::col(tx, 0);
    gload(kbeg);
    sstore(0);
    __syncthreads();
#pragma unroll 1
    for (int it = 0; it < NK; ++it) {
      if (it + 1 < NK) gload(kbeg + (it + 1) * BK);
      const float* As = smem + (it & 1) * S::STAGE;
      const float* Bs = As + S::AF;
      float fa[2][TM], fb[2][TN];
      frag_load(fa[0], As + arow, BM / 2);
      frag_load(fb[0], Bs + bcol, BN / 2);
#pragma unroll
      for (int k = 0; k < BK; ++k) {
        if (k + 1 < BK) {
          frag_load(fa[(k + 1) & 1], As + (k + 1) * S::LDA + arow, BM / 2);
          frag_load(fb[(k + 1) & 1], Bs + (k + 1) * S::LDB + bcol, BN / 2);
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(fa[k & 1][i], fb[k & 1][j], acc[i][j]);
      }
      if (it + 1 < NK) sstore((it + 1) & 1);
      __syncthreads();  // next stage visible; this stage's fragments consumed before it is overwritten
    }
    float* C = C0 + rboff(b, lay.NB2, lay.BSC, lay.BSC2);
    bool store = true;
    if (lay.KSPLIT > 1) {
      const int tile = (b * lay.TILES_M + tm) * lay.TILES_N + tn;
      float* P = part + (long long)tile * lay.KSPLIT * (BM * BN);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) __stcg(P + (long long)s * (BM * BN) + (i * TN + j) * 256 + t, acc[i][j]);
      // CTA barrier orders all partial stores before thread 0's gpu-scope acq_rel arrival (cumulative
      // release); the last arriver's acquire plus the next barrier orders its slice loads (P-OPS-B relacq).
      __syncthreads();
      if (t == 0) {
        unsigned old;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt + tile) : "memory");
        last_flag = old == (unsigned)lay.KSPLIT - 1u;
        if (last_flag) atomicExch(cnt + tile, 0u);
      }
      __syncthreads();
      store = last_flag;
      if (store) {
        constexpr int RG = (48 / (TM * TN)) < 1 ? 1 : ((48 / (TM * TN)) > 4 ? 4 : 48 / (TM * TN));
        float sum[TM][TN];
#pragma unroll 1
        for (int s0 = 0; s0 < lay.KSPLIT; s0 += RG) {
          float x[RG][TM][TN];
#pragma unroll
          for (int g = 0; g < RG; ++g)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                const int ss = s0 + g;
                x[g][i][j] = ss >= lay.KSPLIT ? 0.0f : ss == s ? acc[i][j]
                           : __ldcg(P + (long long)ss * (BM * BN) + (i * TN + j) * 256 + t);
              }
#pragma unroll
          for (int g = 0; g < RG; ++g)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                const int ss = s0 + g;
                if (ss < lay.KSPLIT) sum[i][j] = ss == 0 ? x[g][i][j] : sum[i][j] + x[g][i][j];
              }
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = sum[i][j];
      }
    }
    if (store) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float* Cm = C + rcs(m0 + S::row(ty, i), lay.MIC, lay.SCM, lay.SCM1);
        if (lay.SCN == 1 && TN >= 4) {
#pragma unroll
          for (int h = 0; h < TN / 4; ++h) {
            float4* target = reinterpret_cast<float4*>(Cm + rcs(n0 + S::col(tx, 4 * h), lay.NIC, 1, lay.SCN1));
            if constexpr (ACC) {
              const float4 old = *target;
              *target = make_float4(old.x + acc[i][4 * h + 0], old.y + acc[i][4 * h + 1],
                                    old.z + acc[i][4 * h + 2], old.w + acc[i][4 * h + 3]);
            } else *target =
                make_float4(acc[i][4 * h + 0], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
          }
        } else {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            float& target = Cm[rcs(n0 + S::col(tx, j), lay.NIC, lay.SCN, lay.SCN1)];
            if constexpr (ACC) target = target + acc[i][j];
            else target = acc[i][j];
          }
        }
      }
    }
    __syncthreads();  // shared buffers and last_flag are reused by this CTA's next task
  }
}


// Compile-time layout (one specialized inlined copy per product).
template <int BM, int BN, int BK, bool AK, bool BKC, class L, bool ACC = false>
POPS2_PHASE_FN void simt_phase(const float* __restrict__ A0, const float* __restrict__ B0, float* __restrict__ C0,
                               float* __restrict__ part, unsigned* cnt, int vbase, float* smem) {
  simt_body<BM, BN, BK, AK, BKC, L, ACC>(L{}, A0, B0, C0, part, cnt, vbase, smem);
}

// Runtime layout: one shared out-of-line body per (tile, k depth, operand layout class), so the instruction
// stream of every product phase with that class is the same code (instruction-cache reuse across phases).
struct RtLayout {
  long long SAM, SAK, SAM1; int MIA, KIA; long long SAK1, BSA, BSA2;
  long long SBK, SBN, SBK1, SBN1; int KIB, NIB; long long BSB, BSB2;
  long long SCM, SCN, SCM1, SCN1; int MIC, NIC; long long BSC, BSC2;
  int K, NB2, TILES_M, TILES_N, KSPLIT, TASKS;
};
template <int BM, int BN, int BK, bool AK, bool BKC>
__device__ __noinline__ void simt_rt(const RtLayout lay, const float* __restrict__ A0, const float* __restrict__ B0,
                                     float* __restrict__ C0, float* __restrict__ part, unsigned* cnt, int vbase,
                                     float* smem) {
  simt_body<BM, BN, BK, AK, BKC>(lay, A0, B0, C0, part, cnt, vbase, smem);
}

}  // namespace pops2
#endif
