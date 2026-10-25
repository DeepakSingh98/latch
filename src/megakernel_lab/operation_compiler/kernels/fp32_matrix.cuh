// P-OPS-B FP32 SIMT matrix engine for the latch-ops operation compiler (new code; not a pinned donor file).
//
// C_b[m, n] = sum_k A_b(m, k) B_b(k, n) with strict FP32 arithmetic: one fmaf chain per output element and per split-K
// slice, k ascending; no TF32, no tensor cores, no floating atomics. With ksplit > 1 every slice stores its partial
// tile and the CTA that completes a tile's last slice sums the slices in slice order 0..ksplit-1 (the value of the
// running sum never depends on which CTA arrives last), so a launch is a deterministic function of its inputs.
//
// Differences from the pinned s101h gemm.cuh body (same arithmetic per slice, different execution):
//  * operands are copied with cp.async into an NST-stage shared ring in their natural global layout, so several k-steps
//    of global latency are in flight (the pinned body has one register-staged step and stores k-contiguous operands
//    transposed into shared memory);
//  * k is consumed in groups of four (one float4 per operand row), with a thread-to-column mapping chosen per B
//    layout so that every shared load is a broadcast or bank-conflict free;
//  * a phase may hold several independent products; `vbase` rotates task ownership so they spread over the grid.
// Layout admission (checked by the generator): A(m,k) is k-contiguous (AK) or m-contiguous; B(k,n) is k-contiguous
// (BKC) or n-contiguous; batch b addresses (b / nb2) * s1 + (b % nb2) * s2. Each of m, n, k may be a two-factor index
// x -> (x / xi) * s_outer + (x % xi) * s_inner (a view through a permuted copy) when the tile extent divides xi
// (BM | mi, 64 | ni, 16 | ki). M % BM == 0, N % 64 == 0, K % (16 * ksplit) == 0; 16-byte aligned rows.
#ifndef POPS_FP32_MATRIX_CUH
#define POPS_FP32_MATRIX_CUH
#include <type_traits>

namespace pops {

// P-OPS2-D: optional per-element epilogue (a device functor receiving the flat
// row-major output index and the final value). NoEpilogue keeps the pinned store.
struct NoEpilogue {};

constexpr int BK = 16;
constexpr int NT = 256;

// Layout L (generated per distinct product, all compile-time constants):
//   SAM, SAK, SAM1, MIA, KIA, SAK1, BSA, BSA2   A(m, k) strides, two-factor extents, batch strides
//   SBK, SBN, SBK1, SBN1, KIB, NIB, BSB, BSB2   B(k, n)
//   SCM, SCN, SCM1, SCN1, MIC, NIC, BSC, BSC2   C(m, n)
//   K, NB2, TILES_M, TILES_N, KSPLIT, TASKS
// Runtime arguments are only the five storage pointers. Partials: [tile][KSPLIT][BM*BN] (thread-major);
// counters: one zero-initialized arrival counter per output tile (KSPLIT > 1 only).

template <int NB2, long long S1, long long S2>
__device__ __forceinline__ long long boff(int b) {
  return (long long)(b / NB2) * S1 + (long long)(b % NB2) * S2;
}
template <int XI, long long S0, long long S1>
__device__ __forceinline__ long long cs(int x) {
  return (long long)(x / XI) * S1 + (long long)(x % XI) * S0;
}
__device__ __forceinline__ void cp16(float* dst, const float* src) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }

// Thread (tx, ty) = (t % 16, t / 16) owns rows ty*TM + i and columns
//   n-contiguous B: h*64 + tx*4 + c (h < TN/4, c < 4)   -> float4 shared loads/stores, contiguous lanes
//   k-contiguous B: tx + 16*j (j < TN)                  -> conflict-free rows of the [n][BK+4] tile
template <int BM, int BN, bool AK, bool BKC, int KB>
struct Shape {
  static constexpr int TM = BM / 16, TN = BN / 16;
  static constexpr int SA = AK ? KB + 4 : BM + 4;            // A rows: m (AK) or k
  static constexpr int SB = BKC ? KB + 4 : BN + 4;           // B rows: n (BKC) or k
  static constexpr int AF = AK ? BM * SA : KB * SA;
  static constexpr int BF = BKC ? BN * SB : KB * SB;
  static constexpr int STAGE = AF + BF;
  __device__ static __forceinline__ int col(int tx, int j) { return BKC ? tx + 16 * j : (j / 4) * 64 + tx * 4 + (j % 4); }
};

#if defined(POPS_INLINE)
#define POPS_PHASE_FN __device__ __forceinline__
#else
#define POPS_PHASE_FN __device__ __noinline__
#endif

// P-OPS2-D: ACC=true stores C = C + result (the single FP32 addition of an
// accumulating epilogue) instead of C = result; ACC=false is the pinned store.
template <int BM, int BN, bool AK, bool BKC, int NST, class L, int KB = BK, bool ACC = false, class EPI = NoEpilogue>
POPS_PHASE_FN void gemm_phase(const float* __restrict__ A0, const float* __restrict__ B0, float* __restrict__ C0,
                              float* __restrict__ part, unsigned* cnt, int vbase, float* smem, EPI epi = EPI{}) {
  using S = Shape<BM, BN, AK, BKC, KB>;
  static_assert(KB == 16 || KB == 32 || KB == 64, "stage depth");
  constexpr int TM = S::TM, TN = S::TN;
  static_assert((BM == 32 || BM == 64 || BM == 128) && (BN == 64 || BN == 128), "unsupported tile");
  static_assert(NST >= 2, "pipeline depth");
  __shared__ unsigned last_flag;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4;
  const int G = (int)gridDim.x;
  const int start = ((int)blockIdx.x + G - vbase % G) % G;
  for (int task = start; task < L::TASKS; task += G) {
    int r = task;
    const int s = r % L::KSPLIT; r /= L::KSPLIT;
    const int tn = r % L::TILES_N; r /= L::TILES_N;
    const int tm = r % L::TILES_M;
    const int b = r / L::TILES_M;
    const int m0 = tm * BM, n0 = tn * BN;
    constexpr int PER = L::K / L::KSPLIT, NK = PER / KB;
    const int kbeg = s * PER;
    const float* __restrict__ A = A0 + boff<L::NB2, L::BSA, L::BSA2>(b) + cs<L::MIA, (AK ? L::SAM : 1), L::SAM1>(m0);
    const float* __restrict__ B = B0 + boff<L::NB2, L::BSB, L::BSB2>(b) + cs<L::NIB, (BKC ? L::SBN : 1), L::SBN1>(n0);
    auto load = [&](int slot, int k) {
      float* As = smem + slot * S::STAGE;
      float* Bs = As + S::AF;
      const float* Ak = A + cs<L::KIA, (AK ? 1 : L::SAK), L::SAK1>(k);
      const float* Bk = B + cs<L::KIB, (BKC ? 1 : L::SBK), L::SBK1>(k);
#pragma unroll
      for (int c = t; c < BM * KB / 4; c += NT) {
        if constexpr (AK) {
          const int row = c / (KB / 4), ch = c % (KB / 4);
          cp16(As + row * S::SA + ch * 4, Ak + (long long)row * L::SAM + ch * 4);
        } else {
          const int kr = c / (BM / 4), ch = c % (BM / 4);
          cp16(As + kr * S::SA + ch * 4, Ak + (long long)kr * L::SAK + ch * 4);
        }
      }
#pragma unroll
      for (int c = t; c < BN * KB / 4; c += NT) {
        if constexpr (BKC) {
          const int row = c / (KB / 4), ch = c % (KB / 4);
          cp16(Bs + row * S::SB + ch * 4, Bk + (long long)row * L::SBN + ch * 4);
        } else {
          const int kr = c / (BN / 4), ch = c % (BN / 4);
          cp16(Bs + kr * S::SB + ch * 4, Bk + (long long)kr * L::SBK + ch * 4);
        }
      }
    };
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
#pragma unroll
    for (int st = 0; st < NST - 1; ++st) {
      if (st < NK) load(st, kbeg + st * KB);
      cp_commit();
    }
#pragma unroll 1
    for (int it = 0; it < NK; ++it) {
      cp_wait<NST - 2>();
      __syncthreads();  // stage `it` visible to all threads; slot (it-1)%NST is no longer read
      {
        const int nx = it + NST - 1;
        if (nx < NK) load(nx % NST, kbeg + nx * KB);
        cp_commit();
      }
      const float* As = smem + (it % NST) * S::STAGE;
      const float* Bs = As + S::AF;
#pragma unroll
      for (int q = 0; q < KB / 4; ++q) {
        float a[TM][4], bv[4][TN];
        if constexpr (AK) {
#pragma unroll
          for (int i = 0; i < TM; ++i) {
            const float4 x = *reinterpret_cast<const float4*>(As + (ty * TM + i) * S::SA + q * 4);
            a[i][0] = x.x; a[i][1] = x.y; a[i][2] = x.z; a[i][3] = x.w;
          }
        } else {
#pragma unroll
          for (int kq = 0; kq < 4; ++kq) {
            if constexpr (TM == 2) {
              const float2 x = *reinterpret_cast<const float2*>(As + (q * 4 + kq) * S::SA + ty * 2);
              a[0][kq] = x.x; a[1][kq] = x.y;
            } else {
#pragma unroll
              for (int h = 0; h < TM / 4; ++h) {
                const float4 x = *reinterpret_cast<const float4*>(As + (q * 4 + kq) * S::SA + ty * TM + 4 * h);
                a[4 * h + 0][kq] = x.x; a[4 * h + 1][kq] = x.y; a[4 * h + 2][kq] = x.z; a[4 * h + 3][kq] = x.w;
              }
            }
          }
        }
        if constexpr (BKC) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const float4 x = *reinterpret_cast<const float4*>(Bs + (tx + 16 * j) * S::SB + q * 4);
            bv[0][j] = x.x; bv[1][j] = x.y; bv[2][j] = x.z; bv[3][j] = x.w;
          }
        } else {
#pragma unroll
          for (int kq = 0; kq < 4; ++kq)
#pragma unroll
            for (int h = 0; h < TN / 4; ++h) {
              const float4 x = *reinterpret_cast<const float4*>(Bs + (q * 4 + kq) * S::SB + h * 64 + tx * 4);
              bv[kq][4 * h + 0] = x.x; bv[kq][4 * h + 1] = x.y; bv[kq][4 * h + 2] = x.z; bv[kq][4 * h + 3] = x.w;
            }
        }
#pragma unroll
        for (int kq = 0; kq < 4; ++kq)
#pragma unroll
          for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i][kq], bv[kq][j], acc[i][j]);
      }
    }
    cp_wait<0>();
    float* C = C0 + boff<L::NB2, L::BSC, L::BSC2>(b);
    bool store = true;
    if constexpr (L::KSPLIT > 1) {
      const int tile = (b * L::TILES_M + tm) * L::TILES_N + tn;
      float* P = part + (long long)tile * L::KSPLIT * (BM * BN);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) __stcg(P + (long long)s * (BM * BN) + (i * TN + j) * NT + t, acc[i][j]);
#if defined(POPS_RELACQ)
      // Option relacq (synchronization only): the CTA barrier orders every thread's partial stores before
      // thread 0's gpu-scope acquire-release arrival (cumulative release, as in Latch S102G bar2); the last
      // arriver's acquire plus the following CTA barrier orders its slice loads. No per-thread fences.
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
#else
      __threadfence();
      __syncthreads();
      if (t == 0) {
        const unsigned old = atomicAdd(cnt + tile, 1u);
        last_flag = old == (unsigned)L::KSPLIT - 1u;
        if (last_flag) atomicExch(cnt + tile, 0u);
      }
      __syncthreads();
      store = last_flag;
      if (store) {
        __threadfence();
#endif
        // Fixed slice order 0..KSPLIT-1; this CTA's own slice comes from registers (same value as stored).
        // Slices are loaded in groups of RG (a group's loads in flight together) and summed in fixed
        // slice order 0..KSPLIT-1, so the value is identical to one-slice-at-a-time summation.
        constexpr int RG = (48 / (TM * TN)) < 1 ? 1 : ((48 / (TM * TN)) > 4 ? 4 : 48 / (TM * TN));
        float sum[TM][TN];
#pragma unroll 1
        for (int s0 = 0; s0 < L::KSPLIT; s0 += RG) {
          float x[RG][TM][TN];
#pragma unroll
          for (int g = 0; g < RG; ++g)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                const int ss = s0 + g;
                x[g][i][j] = ss >= L::KSPLIT ? 0.0f : ss == s ? acc[i][j]
                           : __ldcg(P + (long long)ss * (BM * BN) + (i * TN + j) * NT + t);
              }
#pragma unroll
          for (int g = 0; g < RG; ++g)
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
              for (int j = 0; j < TN; ++j) {
                const int ss = s0 + g;
                if (ss < L::KSPLIT) sum[i][j] = ss == 0 ? x[g][i][j] : sum[i][j] + x[g][i][j];
              }
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j) acc[i][j] = sum[i][j];
      }
    }
    if constexpr (!std::is_same<EPI, NoEpilogue>::value) {
      // Plain row-major output (checked by the generator): flat index m * SCM + n.
      if (store) {
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
          for (int j = 0; j < TN; ++j)
            epi((long long)(m0 + ty * TM + i) * L::SCM + (n0 + S::col(tx, j)) + (long long)b * L::BSC2, acc[i][j]);
      }
    } else
    if (store) {
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        float* Cm = C + cs<L::MIC, L::SCM, L::SCM1>(m0 + ty * TM + i);
        if constexpr (!BKC && L::SCN == 1) {
#pragma unroll
          for (int h = 0; h < TN / 4; ++h) {
            float4* target = reinterpret_cast<float4*>(Cm + cs<L::NIC, 1, L::SCN1>(n0 + h * 64) + tx * 4);
            if constexpr (ACC) {
              const float4 c = *target;
              *target = make_float4(c.x + acc[i][4 * h + 0], c.y + acc[i][4 * h + 1],
                                    c.z + acc[i][4 * h + 2], c.w + acc[i][4 * h + 3]);
            } else {
              *target = make_float4(acc[i][4 * h + 0], acc[i][4 * h + 1], acc[i][4 * h + 2], acc[i][4 * h + 3]);
            }
          }
        } else {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            float& target = Cm[cs<L::NIC, L::SCN, L::SCN1>(n0 + S::col(tx, j))];
            if constexpr (ACC) target = target + acc[i][j];
            else target = acc[i][j];
          }
        }
      }
    }
    __syncthreads();  // shared ring and last_flag are reused by this CTA's next task
  }
}

}  // namespace pops
#endif
