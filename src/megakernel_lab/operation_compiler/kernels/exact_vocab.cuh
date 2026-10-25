#pragma once
#include <cuda_bf16.h>
#include <cassert>
#include "torch_logsoftmax.cuh"

// Bounded unpadded counterpart of vocab_head's sparse cross-entropy lowering.
// Reuses the pinned PyTorch ILP4 arithmetic from torch_logsoftmax.cuh. The
// virtual row emulates the eliminated FP32 allocation's alignment, NOT the
// address alignment of the BF16 input. No padded contraction is introduced.
namespace latch_exact_vocab {
struct BF16Row {
  const __nv_bfloat16* x;
  __device__ __forceinline__ float operator()(int k) const {return __bfloat162float(x[k]);}
  __device__ __forceinline__ void four(int k,float (&a)[4]) const {
    // The ILP4 body's virtual FP32 alignment also makes this BF16 word
    // 8-byte aligned. Move the word once, retaining all four conversions.
    const uint2 bits=*reinterpret_cast<const uint2*>(x+k);
    a[0]=__uint_as_float(bits.x<<16);a[1]=__uint_as_float(bits.x&0xffff0000u);
    a[2]=__uint_as_float(bits.y<<16);a[3]=__uint_as_float(bits.y&0xffff0000u);
  }
};
struct SparseRow {
  int target; float selected,other;
  __device__ __forceinline__ float operator()(int k) const {return k==target?selected:other;}
  __device__ __forceinline__ void four(int k,float (&a)[4]) const {
    #pragma unroll
    for(int j=0;j<4;++j) a[j]=(*this)(k+j);
  }
};
template<int Kind> __device__ __forceinline__ float consume(float result,float value,float maximum) {
  if constexpr(Kind==0) return fmaxf(result,value);
  else if constexpr(Kind==1) return __fadd_rn(result,expf(__fsub_rn(value,maximum)));
  else return __fadd_rn(result,value);
}
template<int Threads,int Kind,class Load> __device__ __forceinline__ float ilp(
    Load load,int width,int tid,int shift,float maximum=0.f) {
  float result=Kind==0?-3.40282346638528859812e38f:0.f;
  int size=width,base=0;
  if(shift>0) {
    base-=shift;size+=shift;
    if(tid>=shift && tid<size) result=consume<Kind>(result,load(base+tid),maximum);
    size-=Threads>size?size:Threads;base+=Threads;
  }
  const int last=size%(4*Threads);
  for(int offset=tid;offset*4<size-last;offset+=Threads) {
    // Independent loads in flight, same four-element accumulation order.
    float a[4];load.four(base+4*offset,a);
    #pragma unroll
    for(int j=0;j<4;++j) result=consume<Kind>(result,a[j],maximum);
  }
  for(int offset=size-last+tid;offset<size;offset+=Threads)
    result=consume<Kind>(result,load(base+offset),maximum);
  return result;
}
__device__ __forceinline__ void stats_row(const __nv_bfloat16* x,float* stats,
    int width,long long row,float* shared) {
  BF16Row load{x+row*width};float a[4];
  const int shift=(row*width)%4;
  #pragma unroll
  for(int q=0;q<4;++q) a[q]=ilp<1024,0>(load,width,threadIdx.x+q*256,shift);
  const float maximum=latch_torch_logsoftmax::reduce1024<true>(a,shared);
  #pragma unroll
  for(int q=0;q<4;++q) a[q]=ilp<1024,1>(load,width,threadIdx.x+q*256,shift,maximum);
  const float logsum=logf(latch_torch_logsoftmax::reduce1024<false>(a,shared));
  if(threadIdx.x==0) {stats[2*row]=maximum;stats[2*row+1]=logsum;}
  __syncthreads();
}
__device__ __forceinline__ float sparse_leaf(SparseRow load,int width,int shift,int tid) {
  // For a finite singleton over signed zeros, every original ILP leaf is
  // exactly +0 except its owner's RN(+0 + selected). Compute those identical
  // leaves directly; the 512 -> 16 -> 1 reduction below remains unchanged.
  // Nonzero or nonfinite cotangents retain the original ordered traversal.
  const bool singleton=load.other==0.f && isfinite(load.selected);
  const int base=shift?512-shift:0;
  const int delta=load.target-base;
  const int body=(width-base)/2048*2048;
  const int owner=load.target<base?load.target+shift:
                  (delta<body?(delta/4)%512:delta%512);
  return singleton?(tid==owner?__fadd_rn(0.f,load.selected):0.f):
                    ilp<512,2>(load,width,tid,shift);
}
__device__ __forceinline__ float sparse_sum(SparseRow load,int width,long long row,float* shared) {
  const int shift=(row*width)%4;
  #pragma unroll
  for(int q=0;q<2;++q)
    shared[threadIdx.x+q*256]=sparse_leaf(load,width,shift,threadIdx.x+q*256);
  __syncthreads();float total=0.f;
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
  __syncthreads();total=shared[0];__syncthreads();return total;
}
__device__ __forceinline__ void grad_row(const __nv_bfloat16* x,const float* stats,
    __nv_bfloat16* dx,int width,int output_pitch,long long row,int target,float selected,float other,float* shared) {
  SparseRow cotangent{target,selected,other};
  const float total=sparse_sum(cotangent,width,row,shared);
  // The existing matrix-only operand planner may align the output row pitch.
  // Only logical columns are written/read; the matrix K extent stays width.
  x+=row*width;dx+=row*output_pitch;
  const float maximum=stats[2*row],logsum=stats[2*row+1];
  // Align output words independently of the unpadded input rows. Each lane
  // owns four contiguous results. Two aligned source words cover every
  // possible BF16 input/output pitch mismatch; only bit routing changes.
  const int lead=(4-((row*output_pitch)%4))%4;
  const int input_shift=(row*width+lead)%4;
  const int vector_end=lead+((width-lead-4)/4)*4;
  auto fetch=[&](int k) {
    const uint2* p=reinterpret_cast<const uint2*>(x+k-input_shift);
    const uint2 a=p[0];uint2 bits=a;
    if(input_shift) {
      const uint2 b=p[1];
      if(input_shift==1) {bits.x=(a.x>>16)|(a.y<<16);bits.y=(a.y>>16)|(b.x<<16);}
      else if(input_shift==2) {bits.x=a.y;bits.y=b.x;}
      else {bits.x=(a.y>>16)|(b.x<<16);bits.y=(b.x>>16)|(b.y<<16);}
    }
    return bits;
  };
  auto emit=[&](int k,uint2 bits) {
    const float v[4]={__uint_as_float(bits.x<<16),__uint_as_float(bits.x&0xffff0000u),
                      __uint_as_float(bits.y<<16),__uint_as_float(bits.y&0xffff0000u)};
    unsigned packed[2]={0,0};
    #pragma unroll
    for(int q=0;q<4;++q) {
      const float logp=__fsub_rn(__fsub_rn(v[q],maximum),logsum);
      const __nv_bfloat16 value=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(k+q)));
      packed[q/2]|=((unsigned)__bfloat16_as_ushort(value))<<((q%2)*16);
    }
    *reinterpret_cast<uint2*>(dx+k)=make_uint2(packed[0],packed[1]);
  };
  // LATCH-MOE-EMA-FAST: eight independent row words in flight per thread (was four; latency-bound at one
  // 256-thread CTA per SM). Each element's arithmetic, its word and its store are unchanged.
  int kv=lead+threadIdx.x*4;
  for(;kv+7*1024<vector_end;kv+=8*1024) {
    uint2 words[8];
    #pragma unroll
    for(int u=0;u<8;++u) words[u]=fetch(kv+u*1024);
    #pragma unroll
    for(int u=0;u<8;++u) emit(kv+u*1024,words[u]);
  }
  for(;kv+3*1024<vector_end;kv+=4*1024) {
    uint2 words[4];
    #pragma unroll
    for(int u=0;u<4;++u) words[u]=fetch(kv+u*1024);
    #pragma unroll
    for(int u=0;u<4;++u) emit(kv+u*1024,words[u]);
  }
  for(;kv<vector_end;kv+=4*256) emit(kv,fetch(kv));
  // Peel only logical elements, and keep the aligned lookahead in bounds at
  // the final row. No padding value participates in arithmetic or is written.
  if(threadIdx.x<lead) {
    const int k=threadIdx.x;
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(x[k]),maximum),logsum);
    dx[k]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(k)));
  }
  const int k=vector_end+threadIdx.x;
  if(k<width) {
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(x[k]),maximum),logsum);
    dx[k]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(k)));
  }
  __syncthreads();
}
}
