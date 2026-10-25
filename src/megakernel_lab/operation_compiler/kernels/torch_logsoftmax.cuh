#pragma once
// Promoted only after exact four-rank primary FP32/BF16 CE component evidence.

#include <cuda_runtime.h>
// Independent operation-order implementation, attributed to PyTorch v2.9.1
// (BSD-3-Clause), SoftMax.cu: ilpReduce, blockReduce, blockReduceWarp and the
// LogSoftMax epilogues. Only FP32, last-axis, contiguous finite input admitted.
namespace latch_torch_logsoftmax {
template<bool Maximum> __device__ __forceinline__ float combine(float a,float b) {
  if constexpr(Maximum) return fmaxf(a,b);
  else return __fadd_rn(a,b);
}
template<bool Maximum> __device__ __forceinline__ float warp(float a) {
  #pragma unroll
  for(int d=16;d>0;d>>=1) a=combine<Maximum>(a,__shfl_down_sync(0xffffffffu,a,d));
  return a;
}
template<int Threads,int Kind> __device__ __forceinline__ float ilp(
    const float* row,int width,int tid,float maximum=0.f) {
  float result=Kind==0?-3.40282346638528859812e38f:0.f;
  const int shift=(reinterpret_cast<unsigned long long>(row)%16)/4;
  int size=width;
  if(shift>0) {
    row-=shift;size+=shift;
    if(tid>=shift && tid<size) {
      float value=row[tid];
      if constexpr(Kind==0) result=fmaxf(result,value);
      else if constexpr(Kind==1) result=__fadd_rn(result,expf(__fsub_rn(value,maximum)));
      else result=__fadd_rn(result,value);
    }
    size-=Threads>size?size:Threads;row+=Threads;
  }
  const int last=size%(4*Threads);
  for(int offset=tid;offset*4<size-last;offset+=Threads) {
    const float4 v=reinterpret_cast<const float4*>(row)[offset];
    const float a[4]={v.x,v.y,v.z,v.w};
    #pragma unroll
    for(int j=0;j<4;++j) {
      if constexpr(Kind==0) result=fmaxf(result,a[j]);
      else if constexpr(Kind==1) result=__fadd_rn(result,expf(__fsub_rn(a[j],maximum)));
      else result=__fadd_rn(result,a[j]);
    }
  }
  for(int offset=size-last+tid;offset<size;offset+=Threads) {
    if constexpr(Kind==0) result=fmaxf(result,row[offset]);
    else if constexpr(Kind==1) result=__fadd_rn(result,expf(__fsub_rn(row[offset],maximum)));
    else result=__fadd_rn(result,row[offset]);
  }
  return result;
}
template<bool Maximum> __device__ __forceinline__ float reduce1024(float (&a)[4],float* shared) {
  const int tid=threadIdx.x,lane=tid&31,warp_id=tid>>5;
  #pragma unroll
  for(int q=0;q<4;++q) {
    a[q]=warp<Maximum>(a[q]);
    if(lane==0) shared[q*8+warp_id]=a[q];
  }
  __syncthreads();
  if(warp_id==0) {
    float x=warp<Maximum>(shared[lane]);
    if(lane==0) shared[0]=x;
  }
  __syncthreads();
  float result=shared[0];
  __syncthreads();
  return result;
}
__device__ __forceinline__ void forward_row(const float* x,float* y,int width,long long row,float* shared) {
  x+=row*width;y+=row*width;float a[4];
  #pragma unroll
  for(int q=0;q<4;++q) a[q]=ilp<1024,0>(x,width,threadIdx.x+q*256);
  const float maximum=reduce1024<true>(a,shared);
  #pragma unroll
  for(int q=0;q<4;++q) a[q]=ilp<1024,1>(x,width,threadIdx.x+q*256,maximum);
  const float logsum=logf(reduce1024<false>(a,shared));
  for(int k=threadIdx.x;k<width;k+=256) y[k]=__fsub_rn(__fsub_rn(x[k],maximum),logsum);
  __syncthreads();
}
__device__ __forceinline__ void backward_row(const float* g,const float* y,float* dx,
    int width,long long row,bool fused,float* shared) {
  g+=row*width;y+=row*width;dx+=row*width;
  #pragma unroll
  for(int q=0;q<2;++q) shared[threadIdx.x+q*256]=ilp<512,2>(g,width,threadIdx.x+q*256);
  __syncthreads();
  float total=0.f;
  if(threadIdx.x<16) {
    #pragma unroll
    for(int j=0;j<32;++j) total=__fadd_rn(total,shared[threadIdx.x*32+j]);
  }
  __syncthreads();
  if(threadIdx.x<16) shared[threadIdx.x]=total;
  __syncthreads();
  if(threadIdx.x==0) {
    total=0.f;
    #pragma unroll
    for(int j=0;j<16;++j) total=__fadd_rn(total,shared[j]);
    shared[0]=total;
  }
  __syncthreads();total=shared[0];
  for(int k=threadIdx.x;k<width;k+=256) {
    const float probability=expf(y[k]);
    dx[k]=fused?__fmaf_rn(-probability,total,g[k]):__fsub_rn(g[k],__fmul_rn(probability,total));
  }
  __syncthreads();
}
}
