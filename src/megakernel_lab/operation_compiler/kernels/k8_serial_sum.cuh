// Ordered sum of individually BF16-rounded k8 partial products. Reuses the
// owned mma_k8.cuh arithmetic/staging body; see its source attribution.
#pragma once
#include "mma_k8.cuh"

namespace latch_mma_k8 {
template <bool AK, bool BKC, int CHUNK>
__device__ __noinline__ void phase_serial_sum(const popsc::BArgs g, float* smem) {
  constexpr int BM=128, BN=128, PITCH=BN+8, WM=2, WN=4, KS=32, NST=4, GROUP=8;
  static_assert(CHUNK > 0 && CHUNK % 16 == 0, "whole k8 partial boundaries");
  using L = popsc::BLay<BM,BN,AK,BKC,KS>;
  constexpr int STAGING_FLOATS = NST * L::STAGE / 2;
  float* partial = smem + STAGING_FLOATS;
  float* total = partial + BM * PITCH;
  const popsc::Slice slice = popsc::slice_of(0,1,1,0);
  const int tm=(g.M+BM-1)/BM, tn=(g.N+BN-1)/BN, span=GROUP*tn;
  for (int task=slice.me; task<tm*tn; task+=slice.count) {
    const int first=(task/span)*GROUP, size=tm-first<GROUP?tm-first:GROUP;
    const int local=task%span, m0=(first+local%size)*BM, n0=(local/size)*BN;
    const int mc=g.M-m0<BM?g.M-m0:BM, nc=g.N-n0<BN?g.N-n0:BN;
    for (int start=0; start<g.K; start+=CHUNK) {
      popsc::BArgs part=g;
      part.A = g.A + (long long)m0*g.sam + (long long)start*g.sak;
      part.B = g.B + (long long)start*g.sbk + (long long)n0*g.sbn;
      part.C = reinterpret_cast<s101h::bf16*>(partial);
      part.M=mc; part.N=nc; part.K=g.K-start<CHUNK?g.K-start:CHUNK;
      part.ldc=PITCH; part.nb=1; part.bsa=part.bsb=part.bsc=0;
      // This is exactly one original partial: FP32 +0, ascending k8, BF16
      // RN, exact widening. Generic stores target this CTA's shared tile.
      tile<BM,BN,WM,WN,AK,BKC,KS,NST,true,false,false,0,false,true>(
          part,0,0,0,reinterpret_cast<s101h::bf16*>(smem));
      for (int j=threadIdx.x; j<BM*BN/4; j+=NTHREADS) {
        const int row=j/(BN/4), col=(j%(BN/4))*4;
        const int sj=row*(PITCH/4)+col/4;
        if (row>=mc || col>=nc) continue;
        if (col+3<nc) {
          const float4 x = reinterpret_cast<const float4*>(partial)[sj];
          if (start==0) reinterpret_cast<float4*>(total)[sj]=x;
          else {
            const float4 prev=reinterpret_cast<const float4*>(total)[sj];
            reinterpret_cast<float4*>(total)[sj]=make_float4(__fadd_rn(prev.x,x.x),__fadd_rn(prev.y,x.y),
                                                          __fadd_rn(prev.z,x.z),__fadd_rn(prev.w,x.w));
          }
        } else {
#pragma unroll
          for (int q=0; q<4; ++q) if (col+q<nc) {
            const int i=sj*4+q;
            total[i]=start==0?partial[i]:__fadd_rn(total[i],partial[i]);
          }
        }
      }
      __syncthreads(); // every reader finished before the next partial overwrites the shared tile
    }
    float* out=reinterpret_cast<float*>(g.C);
    for (int j=threadIdx.x; j<BM*BN/4; j+=NTHREADS) {
      const int row=j/(BN/4), col=(j%(BN/4))*4;
      const int sj=row*(PITCH/4)+col/4;
      if (row>=mc || col>=nc) continue;
      const long long where=(long long)(m0+row)*g.ldc+n0+col;
      if (col+3<nc && g.ldc%4==0)
        *reinterpret_cast<float4*>(out+where)=reinterpret_cast<const float4*>(total)[sj];
      else {
#pragma unroll
        for (int q=0; q<4; ++q) if (col+q<nc) out[where+q]=total[sj*4+q];
      }
    }
    __syncthreads();
  }
}
} // namespace latch_mma_k8
