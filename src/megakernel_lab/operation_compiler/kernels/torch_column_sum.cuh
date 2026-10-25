#pragma once
#include <cuda_runtime.h>
// Independently expressed PyTorch2.9.1 Reduce.cuh / ReduceSumProdKernel.cu
// arithmetic (BSD-3-Clause). The measured standalone component's virtual CTA
// arithmetic is unchanged; cooperative dependency phases replace two launches.
namespace latch_torch_column_sum {
struct __align__(16) Vec {float v[4];};
__device__ __forceinline__ Vec add(Vec a,const Vec b) {
  #pragma unroll
  for(int j=0;j<4;++j) a.v[j]=__fadd_rn(a.v[j],b.v[j]);
  return a;
}
template<int BX,int BY>
__device__ __forceinline__ Vec reduce_y(Vec value,Vec* shared) {
  const int tid=threadIdx.x,ty=tid/BX;
  if(tid<BX*BY) shared[tid]=value;
  for(int offset=BY/2;offset>0;offset>>=1) {
    __syncthreads();
    if(tid<BX*BY && ty<offset) {
      value=add(value,shared[tid+offset*BX]);
      shared[tid]=value;
    }
  }
  return value;
}
template<int R,int C,int BX,int BY,int GX,int CTAS,int REDUCE_Y,int STEP>
__device__ __forceinline__ void partial(const float* input,float* output,int task,float* smem) {
  const int tid=threadIdx.x,tx=tid%BX,ty=tid/BX;
  const int block_x=task/CTAS,block_y=task%CTAS;
  const int column=(block_x*STEP+tx+(REDUCE_Y?0:ty*BX))*4;
  constexpr int stride=(REDUCE_Y?BY:1)*CTAS;
  int row=(REDUCE_Y?ty:0)+block_y*(REDUCE_Y?BY:1);
  Vec accum[4]={};
  if(tid<BX*BY && column<C) {
    while(row+3*stride<R) {
      #pragma unroll
      for(int a=0;a<4;++a) {
        const Vec loaded=*reinterpret_cast<const Vec*>(input+(long long)(row+a*stride)*C+column);
        accum[a]=add(accum[a],loaded);
      }
      row+=4*stride;
    }
    #pragma unroll
    for(int a=0;a<4;++a) {
      if(row<R) {
        const Vec loaded=*reinterpret_cast<const Vec*>(input+(long long)row*C+column);
        accum[a]=add(accum[a],loaded);row+=stride;
      }
    }
  }
  Vec value=accum[0];
  #pragma unroll
  for(int a=1;a<4;++a) value=add(value,accum[a]);
  if(REDUCE_Y) value=reduce_y<BX,BY>(value,reinterpret_cast<Vec*>(smem));
  if(tid<BX*BY && column<C && (!REDUCE_Y || ty==0)) {
    if(CTAS>1) reinterpret_cast<Vec*>(output)[tx+(block_y+block_x*CTAS)*BX]=value;
    else *reinterpret_cast<Vec*>(output+column)=value;
  }
  __syncthreads();
}
template<int C,int BX,int BY,int CTAS>
__device__ __forceinline__ void finish(const float* partial,float* output,int task,float* smem) {
  const int tid=threadIdx.x,tx=tid%BX,ty=tid/BX,column=(task*BX+tx)*4;
  Vec value={};
  if(tid<BX*BY && column<C) {
    for(int cta=ty;cta<CTAS;cta+=BY)
      value=add(value,reinterpret_cast<const Vec*>(partial)[tx+(cta+task*CTAS)*BX]);
  }
  value=reduce_y<BX,BY>(value,reinterpret_cast<Vec*>(smem));
  if(tid<BX*BY && column<C && ty==0) *reinterpret_cast<Vec*>(output+column)=value;
  __syncthreads();
}
}
