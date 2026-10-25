#pragma once
#include "exact_vocab.cuh"

namespace latch_exact_vocab {
// All four virtual ILP1024 leaves owned by one thread, interleaved. Each leaf
// keeps its original element sequence and four-element accumulation order;
// only independent leaves are interleaved to expose four dependency chains.
template<int Kind,class Load> __device__ __forceinline__ void ilp1024_four(
    Load load,int width,int shift,float (&result)[4],float maximum=0.f) {
  #pragma unroll
  for(int q=0;q<4;++q) result[q]=Kind==0?-3.40282346638528859812e38f:0.f;
  int size=width,base=0;
  if(shift>0) {
    base-=shift;size+=shift;
    #pragma unroll
    for(int q=0;q<4;++q) {
      const int tid=threadIdx.x+q*256;
      if(tid>=shift && tid<size) result[q]=consume<Kind>(result[q],load(base+tid),maximum);
    }
    size-=1024>size?size:1024;base+=1024;
  }
  const int last=size%4096;
  // size-last is a multiple of 4096, so every leaf q shares the q=0 trip count.
  for(int offset=threadIdx.x;offset*4<size-last;offset+=1024) {
    float values[4][4];
    #pragma unroll
    for(int q=0;q<4;++q) load.four(base+4*(offset+q*256),values[q]);
    #pragma unroll
    for(int j=0;j<4;++j) {
      #pragma unroll
      for(int q=0;q<4;++q) result[q]=consume<Kind>(result[q],values[q][j],maximum);
    }
  }
  for(int offset=size-last+threadIdx.x;offset<size;offset+=1024) {
    #pragma unroll
    for(int q=0;q<4;++q) {
      const int k=offset+q*256;
      if(k<size) result[q]=consume<Kind>(result[q],load(base+k),maximum);
    }
  }
}

// Selected only when this storage fits the program's existing shared-memory
// allocation. The copy preserves the original row's virtual FP32 alignment.
__device__ __forceinline__ void stats_row_cached(const __nv_bfloat16* x,float* stats,
    int width,long long row,float* shared) {
  const int shift=(row*width)%4;
  auto* cache=reinterpret_cast<__nv_bfloat16*>(shared+32);
  const auto* source=x+row*width-shift;
  const int size=width+shift,vector_end=size/4*4;
  // Sixteen (LATCH-MOE-EMA-FAST; was eight) independent 8-byte loads in flight per thread before their stores.
  int k=threadIdx.x*4;
  for(;k+15*1024<vector_end;k+=16*1024) {
    uint2 words[16];
    #pragma unroll
    for(int u=0;u<16;++u) words[u]=*reinterpret_cast<const uint2*>(source+k+u*1024);
    #pragma unroll
    for(int u=0;u<16;++u) *reinterpret_cast<uint2*>(cache+k+u*1024)=words[u];
  }
  for(;k+7*1024<vector_end;k+=8*1024) {
    uint2 words[8];
    #pragma unroll
    for(int u=0;u<8;++u) words[u]=*reinterpret_cast<const uint2*>(source+k+u*1024);
    #pragma unroll
    for(int u=0;u<8;++u) *reinterpret_cast<uint2*>(cache+k+u*1024)=words[u];
  }
  for(;k<vector_end;k+=1024)
    *reinterpret_cast<uint2*>(cache+k)=*reinterpret_cast<const uint2*>(source+k);
  const int tail=vector_end+threadIdx.x;
  if(tail<size) cache[tail]=source[tail];
  __syncthreads();
  BF16Row load{cache+shift};float a[4];
  ilp1024_four<0>(load,width,shift,a);
  const float maximum=latch_torch_logsoftmax::reduce1024<true>(a,shared);
  ilp1024_four<1>(load,width,shift,a,maximum);
  const float logsum=logf(latch_torch_logsoftmax::reduce1024<false>(a,shared));
  if(threadIdx.x==0) {stats[2*row]=maximum;stats[2*row+1]=logsum;}
  __syncthreads();
}
}
