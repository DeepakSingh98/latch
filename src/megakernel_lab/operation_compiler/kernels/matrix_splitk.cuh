#pragma once
// Explicit arithmetic wrapper around the existing P6-B raw-FP32 partial body.
// No new MMA implementation: every interval uses popsc::btile_edge F32OUT.
// CTA-local scratch avoids global counters and preserves exact reduction order.
namespace latch_matrix_splitk {
constexpr int BM = 64, BN = 32, KS = 32, NST = 3;
constexpr int STAGE_BYTES = NST * popsc::BLay<BM,BN,false,false,KS>::STAGE * 2;
constexpr int SHARED_BYTES = STAGE_BYTES + BM * BN * sizeof(float);

__device__ __noinline__ void phase(const popsc::BArgs g, float* smem) {
  static_assert(NTHREADS == 256, "split6 requires the admitted 256-thread CTA");
  auto* stage = reinterpret_cast<popsc::bf16*>(smem);
  auto* partial = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + STAGE_BYTES);
  const int chunk = ((g.K + 383) / 384) * 64;
  for (int tile = blockIdx.x; tile < g.M / BM; tile += gridDim.x) {
    float sum[BM * BN / NTHREADS] = {};
    #pragma unroll 1
    for (int split = 0; split < 6; ++split) {
      const int start = split * chunk;
      popsc::BArgs part = g;
      part.A = g.A + (long long)(tile * BM) * g.sam + (long long)start * g.sak;
      part.B = g.B + (long long)start * g.sbk;
      part.C = reinterpret_cast<popsc::bf16*>(partial);
      part.M = BM; part.N = BN; part.K = min(chunk, g.K - start);
      part.nb = 1; part.ldc = BN; part.bsa = part.bsb = part.bsc = 0;
      popsc::btile_edge<BM,BN,4,2,false,false,KS,NST,false,true,true>(part,0,0,0,stage);
      #pragma unroll
      for (int i = 0; i < BM * BN / NTHREADS; ++i)
        sum[i] = __fadd_rn(sum[i], partial[threadIdx.x + i * NTHREADS]);
      __syncthreads(); // every partial read is finished before its next reuse
    }
    #pragma unroll
    for (int i = 0; i < BM * BN / NTHREADS; ++i) {
      const int index = threadIdx.x + i * NTHREADS;
      g.C[(long long)(tile * BM + index / BN) * g.ldc + index % BN] = __float2bfloat16_rn(sum[i]);
    }
    __syncthreads();
  }
}

// LATCH-MOE4-160: the six K intervals run concurrently on six warps of one CTA.
// Warp s owns interval s for the whole WBM x 32 tile: one FP32 accumulator per
// output from +0, one mma.sync.m16n8k16 per k16 block in ascending k (the same
// fragments and instruction as btile_edge F32OUT). The CTA then adds the six
// partials in ascending interval order from +0 and rounds once to BF16, exactly
// as phase() above. Only the assignment of work to warps and CTAs changes.
constexpr int WBM = 16, WKS = 32, WNST = 4;
constexpr int W_A_EL = WKS * (WBM + 8), W_B_EL = WKS * (BN + 8), W_STAGE = W_A_EL + W_B_EL;
constexpr int WARP_STAGE_BYTES = 6 * WNST * W_STAGE * 2;
constexpr int WARPS_SHARED_BYTES = WARP_STAGE_BYTES + 6 * WBM * BN * sizeof(float);

template <int R>
__device__ __forceinline__ void wstage(const popsc::bf16* __restrict__ X, long long sk, int k,
                                       popsc::bf16* __restrict__ S, int lane) {
  // KS x R tile with r-contiguous rows [k][R+8] (operand stride along r is one element).
  constexpr int CH = R * WKS / 8;
  static_assert(CH % 32 == 0, "warp-staged chunk count");
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += 32) {
    const int c = c0 + lane, kr = c / (R / 8), rc = (c % (R / 8)) * 8;
    s101h::cpa16(reinterpret_cast<float*>(S + kr * (R + 8) + rc),
                 reinterpret_cast<const float*>(X + (long long)(k + kr) * sk + rc));
  }
}

__device__ __noinline__ void phase_warps(const popsc::BArgs g, float* smem) {
  static_assert(NTHREADS == 256, "split6 requires the admitted 256-thread CTA");
  constexpr int NT = BN / 8;
  auto* stages = reinterpret_cast<popsc::bf16*>(smem);
  auto* partial = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + WARP_STAGE_BYTES);
  const int chunk = ((g.K + 383) / 384) * 64;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  for (int tile = blockIdx.x; tile < g.M / WBM; tile += gridDim.x) {
    if (w < 6) {
      const int start = w * chunk, length = min(chunk, g.K - start), nk = length / WKS;
      const popsc::bf16* A = g.A + (long long)(tile * WBM) * g.sam + (long long)start * g.sak;
      const popsc::bf16* B = g.B + (long long)start * g.sbk;
      popsc::bf16* S = stages + w * WNST * W_STAGE;
      float acc[NT][4];
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[j][q] = 0.f;
      auto load = [&](int kk, int st) {
        popsc::bf16* As = S + st * W_STAGE;
        wstage<WBM>(A, g.sak, kk, As, lane);
        wstage<BN>(B, g.sbk, kk, As + W_A_EL, lane);
      };
#pragma unroll
      for (int s = 0; s < WNST - 1; ++s) {
        if (s < nk) load(s * WKS, s);
        s101h::cpa_commit();
      }
#pragma unroll 1
      for (int it = 0; it < nk; ++it) {
        s101h::cpa_wait<WNST - 2>();
        __syncwarp();                                    // stage it visible to the warp; stage it-1 consumed
        if (it + WNST - 1 < nk) load((it + WNST - 1) * WKS, (it + WNST - 1) % WNST);
        s101h::cpa_commit();
        const popsc::bf16* As = S + (it % WNST) * W_STAGE;
        const popsc::bf16* Bs = As + W_A_EL;
#pragma unroll
        for (int k16 = 0; k16 < WKS; k16 += 16) {
          uint32_t af[4];
          s101h::bldsm4t(af, As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (WBM + 8) + (((lane >> 3) & 1) << 3));
#pragma unroll
          for (int j = 0; j < NT; j += 2) {
            uint32_t bb[4];
            s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + 8 * (j + (lane >> 4)));
            s101h::bmma(acc[j], af[0], af[1], af[2], af[3], bb[0], bb[1]);
            s101h::bmma(acc[j + 1], af[0], af[1], af[2], af[3], bb[2], bb[3]);
          }
        }
      }
      s101h::cpa_wait<0>();
      float* P = partial + w * WBM * BN;
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int m = gq, n = 8 * j + 2 * tq;
        P[m * BN + n] = acc[j][0]; P[m * BN + n + 1] = acc[j][1];
        P[(m + 8) * BN + n] = acc[j][2]; P[(m + 8) * BN + n + 1] = acc[j][3];
      }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < WBM * BN; index += NTHREADS) {
      float sum = 0.f;
#pragma unroll
      for (int split = 0; split < 6; ++split) sum = __fadd_rn(sum, partial[split * WBM * BN + index]);
      g.C[(long long)(tile * WBM + index / BN) * g.ldc + index % BN] = __float2bfloat16_rn(sum);
    }
    __syncthreads();                                     // partials and stages are reused by the next tile
  }
}
} // namespace latch_matrix_splitk
