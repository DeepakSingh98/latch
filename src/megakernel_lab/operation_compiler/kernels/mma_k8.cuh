// Explicit BF16 mma.sync k8 arithmetic body. Defaults never select this header.
// Reuses owned popsc_mm.cuh edge staging, fragment loads and phase scheduling.
// That staging is attributed there to pinned Latch bgemm.cuh (blob7399bbe1).
// K8 fragment coordinates/instruction: NVIDIA CUDA12.8 PTX ISA8.7,
// https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
// Measured identical-input forward/readout-VJP evidence: LATCH-MOE4
// readout-mma-order-{fresh,aux}-v1. This declares arithmetic, not a complete
// workload qualification. Ascending k8; optional FP32 partial reset and BF16
// RN(partial + widened previous BF16) at graph-bound K partitions.
// Optional initial-residue grouping is independently measured by
// LATCH-MOE4/readout-full-residue-aux-v1 (all four full input-VJP matrices).
// The grouping hypothesis follows the public initial-residue iterator rule:
// NVIDIA CUTLASS v2.11.0, include/cutlass/transform/threadblock/
// predicated_tile_access_iterator.h, set_predicates/add_tile_offset.
// This is not an attribution of proprietary cuBLAS implementation internals.
#pragma once
#include "popsc_mm.cuh"
namespace latch_mma_k8 {
using namespace popsc;
#ifndef LATCH_K8_FULL_STAGES
#define LATCH_K8_FULL_STAGES 1
#endif
constexpr bool K8_FULL_STAGES = LATCH_K8_FULL_STAGES;
#ifndef LATCH_K8_PARTITION_WIDE
#define LATCH_K8_PARTITION_WIDE 0
#endif
constexpr bool K8_PARTITION_WIDE = LATCH_K8_PARTITION_WIDE;
DEV void mma8(float (&d)[4],uint32_t a0,uint32_t a1,uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a0),"r"(a1),"r"(b0));
}
// Virtual K inserts at most seven zero products after the first short k8
// group of the final partition. Unshifted stages retain cp.async staging.
// Shifted stages use scalar BF16 copies: the original source K is no longer
// vector-aligned, so a shifted cp.async address would violate its ABI.
template <int R, bool RK, int KS>
DEV void bstage_initial_residue(const bf16* X, long long sr, long long sk,
                               int r0, int kk, bf16* S, int rlim, int klim,
                               int cut, int padding) {
#pragma unroll
  for (int index = threadIdx.x; index < R * KS; index += NTHREADS) {
    const int r = RK ? index / KS : index % R;
    const int kc = RK ? index % KS : index / R;
    const int virtual_k = kk + kc;
    const bool hole = virtual_k >= cut && virtual_k < cut + padding;
    const int source_k = virtual_k < cut ? virtual_k : virtual_k - padding;
    const int shared_index = RK ? r * (KS + 8) + kc : kc * (R + 8) + r;
    S[shared_index] = (!hole && r0 + r < rlim && source_k >= 0 && source_k < klim)
      ? X[(long long)(r0 + r) * sr + (long long)source_k * sk]
      : __float2bfloat16_rn(0.f);
  }
}
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool EDGE = false, bool PAIR = true,
          bool F32OUT = false, int PARTITION = 0, bool INITIAL_RESIDUE = false, bool ROUNDED_F32 = false>
DEV void tile(const BArgs& g, int b, int m0, int n0, bf16* S) {
  using L = BLay<BM, BN, AK, BKC, KS>;
  constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  static_assert(NST >= 2, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * WTM, wn0 = (w % WN) * WTN;
  const bf16* A = g.A + (long long)b * g.bsa;
  const bf16* B = g.B + (long long)b * g.bsb;
  float acc[MT][NT][4];
  // Partition subtotals are exact BF16 values; hold them as packed BF16 pairs
  // (widened on use) so wide partitioned tiles fit in registers. Bitwise equal.
  uint32_t previous[MT][NT][2] = {};
  auto prev = [&](int i, int j, int q) {
    const uint32_t w = previous[i][j][q >> 1];
    return __uint_as_float((q & 1) ? (w & 0xffff0000u) : (w << 16));
  };
  static_assert(PARTITION == 0 || PARTITION % 16 == 0, "serial partition alignment");
  static_assert(!INITIAL_RESIDUE || (EDGE && PARTITION > 0), "initial residue needs an edge serial partition");
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  int residue_cut = g.K, residue_padding = 0;
  if constexpr (INITIAL_RESIDUE) {
    const int tail_start = ((g.K - 1) / PARTITION) * PARTITION;
    const int short_group = (g.K - tail_start) % 8;
    residue_cut = tail_start + short_group;
    residue_padding = short_group == 0 ? 0 : 8 - short_group;
  }
  const int effective_k = g.K + residue_padding;
  const int nk = EDGE ? (effective_k + KS - 1) / KS : effective_k / KS;
  auto load = [&](int kk, int st) {
    bf16* As = S + st * L::STAGE;
    if constexpr (INITIAL_RESIDUE) {
      if (residue_padding && kk + KS > residue_cut) {
        bstage_initial_residue<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As, g.M, g.K, residue_cut, residue_padding);
        bstage_initial_residue<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL, g.N, g.K, residue_cut, residue_padding);
        return;
      }
    }
    if constexpr (EDGE) {
      bstage_edge<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As, g.M, g.K);
      bstage_edge<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL, g.N, g.K);
    } else {
      bstage<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As);
      bstage<BN, BKC, KS>(B, g.sbn, g.sbk, n0, kk, As + L::A_EL);
    }
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) load(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();                                               // stage it visible; stage it-1 fully consumed
    if (it + NST - 1 < nk) load((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = S + (it % NST) * L::STAGE;
    const bf16* Bs = As + L::A_EL;
    // LATCH-MOE4-160: a stage wholly inside K runs a branch-free copy of the
    // same instruction sequence (every guard below is true there), so the k8
    // halves of all fragments form one schedulable block. Arithmetic unchanged.
    auto stage_body = [&](auto guarded) {
    constexpr bool G = decltype(guarded)::value;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      if (G && it * KS + k16 >= effective_k) continue;
      const bool second = !G || it * KS + k16 + 8 < effective_k;
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          mma8(acc[i][0], af[i][0], af[i][1], bb[0]);
          if (second) mma8(acc[i][0], af[i][2], af[i][3], bb[1]);
        }
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            mma8(acc[i][j], af[i][0], af[i][1], bb[0]);
            mma8(acc[i][j + 1], af[i][0], af[i][1], bb[2]);
            if (second) {
              mma8(acc[i][j], af[i][2], af[i][3], bb[1]);
              mma8(acc[i][j + 1], af[i][2], af[i][3], bb[3]);
            }
          }
        }
      }
      if constexpr (PARTITION > 0) {
        const int end = it * KS + k16 + 16;
        if (end < effective_k && end % PARTITION == 0) {
#pragma unroll
          for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
              for (int q = 0; q < 4; q += 2) {
                previous[i][j][q >> 1] = s101h::bpk(__fadd_rn(acc[i][j][q], prev(i, j, q)),
                                                    __fadd_rn(acc[i][j][q + 1], prev(i, j, q + 1)));
                acc[i][j][q] = 0.f;
                acc[i][j][q + 1] = 0.f;
              }
        }
      }
    }
    };
    if (K8_FULL_STAGES && (it + 1) * KS <= effective_k) stage_body(std::false_type{});
    else stage_body(std::true_type{});
  }
  if constexpr (PARTITION > 0) {
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = __fadd_rn(acc[i][j][q], prev(i, j, q));
  }
  s101h::cpa_wait<0>();                                            // only empty groups can remain
  bf16* C = g.C + (long long)b * g.bsc;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
      if constexpr (ROUNDED_F32) {
        // Exactly the materialized BF16 product followed by FP32 widening.
        // No accumulator precision or rounding boundary changes.
        float* Cf = reinterpret_cast<float*>(g.C) + (long long)b * g.bsc;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = m + 8 * h;
          if (mm >= g.M || n >= g.N) continue;
          const float x = __bfloat162float(__float2bfloat16_rn(acc[i][j][2 * h]));
          const float y = __bfloat162float(__float2bfloat16_rn(acc[i][j][2 * h + 1]));
          if (n + 1 < g.N && g.ldc % 2 == 0)
            *reinterpret_cast<float2*>(Cf + (long long)mm * g.ldc + n) = make_float2(x, y);
          else {
            Cf[(long long)mm * g.ldc + n] = x;
            if (n + 1 < g.N) Cf[(long long)mm * g.ldc + n + 1] = y;
          }
        }
      } else if constexpr (EDGE) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = m + 8 * h;
          if (mm >= g.M || n >= g.N) continue;
          const uint32_t v = s101h::bpk(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
          if (PAIR) {
            *reinterpret_cast<uint32_t*>(C + (long long)mm * g.ldc + n) = v;
          } else {
            unsigned short* c = reinterpret_cast<unsigned short*>(C + (long long)mm * g.ldc + n);
            c[0] = (unsigned short)(v & 0xffffu);
            if (n + 1 < g.N) c[1] = (unsigned short)(v >> 16);
          }
        }
      } else if constexpr (F32OUT) {
        // P6-B split-K slice: the FP32 accumulators themselves (C points at FP32 storage, batch stride in floats).
        float* Cf = reinterpret_cast<float*>(g.C) + (long long)b * g.bsc;
        *reinterpret_cast<float2*>(Cf + (long long)m * g.ldc + n) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(Cf + (long long)(m + 8) * g.ldc + n) = make_float2(acc[i][j][2], acc[i][j][3]);
      } else {
        *reinterpret_cast<uint32_t*>(C + (long long)m * g.ldc + n) = s101h::bpk(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<uint32_t*>(C + (long long)(m + 8) * g.ldc + n) = s101h::bpk(acc[i][j][2], acc[i][j][3]);
      }
    }
  __syncthreads();                                                 // every fragment read done before smem reuse
}


// LATCH-MOE4-PARITY wide schedule of the identical arithmetic. Every output still owns
// one FP32 accumulator that receives m16n8k8 products in ascending (virtual) K from +0
// per partition, with the same partition epilogue and one final cvt.rn rounding; only
// the CTA/warp tile (128x256 or 128x128, 64x64 / 64x32 warp tiles), stage depth and
// the task raster change. Tasks are rastered in groups of GROUP m-tiles so concurrently
// resident CTAs share operand tiles in L2. Scheduling only: bitwise identical results.
template <bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE = false, int KS = 32, bool ROUNDED_F32 = false,
          bool PARTITION_WIDE = K8_PARTITION_WIDE>
__device__ __noinline__ void phase_wide(const BArgs g,float* smem,int rot=0) {
  // PARTITION_WIDE: partitioned products also use 128x256 tiles (packed BF16 subtotals fit).
  constexpr int BM = 128, BN = PARTITION > 0 && !PARTITION_WIDE ? 128 : 256, WM = 2, WN = 4, NST = 4, GROUP = 8;
  static_assert(KS == 32 || KS == 64, "supported k8 wide shared stage");
  const Slice slice = slice_of(0,1,1,rot);
  const int tm=(g.M+BM-1)/BM,tn=(g.N+BN-1)/BN,per=tm*tn,ntask=g.nb*per;
  for(int task=slice.me;task<ntask;task+=slice.count) {
    const int b=task/per,t=task%per,span=GROUP*tn,first=(t/span)*GROUP;
    const int size=tm-first<GROUP?tm-first:GROUP,local=t%span;
    const int mi=first+local%size,nj=local/size;
    tile<BM,BN,WM,WN,AK,BKC,KS,NST,true,false,false,PARTITION,INITIAL_RESIDUE,ROUNDED_F32>(
      g,b,mi*BM,nj*BN,reinterpret_cast<bf16*>(smem));
  }
}

template <bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE = false, bool ROUNDED_F32 = false>
__device__ __noinline__ void phase(const BArgs g,float* smem,int rot=0) {
  const Slice slice = slice_of(0,1,1,rot);
  const int tm=(g.M+63)/64,tn=(g.N+63)/64,ntask=g.nb*tm*tn;
  for(int task=slice.me;task<ntask;task+=slice.count) {
    const int j=task%tn,r=task/tn;
    tile<64,64,4,2,AK,BKC,32,3,true,false,false,PARTITION,INITIAL_RESIDUE,ROUNDED_F32>(
      g,r/tm,(r%tm)*64,j*64,reinterpret_cast<bf16*>(smem));
  }
}
} // namespace latch_mma_k8
