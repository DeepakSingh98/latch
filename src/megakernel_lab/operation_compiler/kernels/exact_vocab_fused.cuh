#pragma once
#include "exact_vocab_cached.cuh"

namespace latch_exact_vocab {
// LATCH-MOE-EMA-FAST: one row pass for the unpadded CE statistics and its sparse-cotangent gradient.
// The row is read from global memory once into program shared storage (as stats_row_cached), its statistics
// are computed by the same ILP1024 leaves and reduction tree and written, and every gradient element is then
// emitted from the cached BF16 row with grad_row's expression, word routing and stores. Row values, statistics,
// cotangent leaves and every rounding are those of stats_row_cached followed by grad_row.
__device__ __forceinline__ void stats_grad_row(const __nv_bfloat16* x,float* stats,__nv_bfloat16* dx,
    int width,int output_pitch,long long row,int target,float selected,float other,float* shared) {
  SparseRow cotangent{target,selected,other};
  const float total=sparse_sum(cotangent,width,row,shared);   // shared[0,512) scratch, released synchronized
  const int shift=(row*width)%4;
  auto* cache=reinterpret_cast<__nv_bfloat16*>(shared+32);
  const auto* source=x+row*width-shift;
  const int size=width+shift,vector_words=size/4*4;
  int k=threadIdx.x*4;
  for(;k+15*1024<vector_words;k+=16*1024) {
    uint2 words[16];
    #pragma unroll
    for(int u=0;u<16;++u) words[u]=*reinterpret_cast<const uint2*>(source+k+u*1024);
    #pragma unroll
    for(int u=0;u<16;++u) *reinterpret_cast<uint2*>(cache+k+u*1024)=words[u];
  }
  for(;k+7*1024<vector_words;k+=8*1024) {
    uint2 words[8];
    #pragma unroll
    for(int u=0;u<8;++u) words[u]=*reinterpret_cast<const uint2*>(source+k+u*1024);
    #pragma unroll
    for(int u=0;u<8;++u) *reinterpret_cast<uint2*>(cache+k+u*1024)=words[u];
  }
  for(;k<vector_words;k+=1024)
    *reinterpret_cast<uint2*>(cache+k)=*reinterpret_cast<const uint2*>(source+k);
  const int tail=vector_words+threadIdx.x;
  if(tail<size) cache[tail]=source[tail];
  __syncthreads();
  BF16Row load{cache+shift};float a[4];
  ilp1024_four<0>(load,width,shift,a);
  const float maximum=latch_torch_logsoftmax::reduce1024<true>(a,shared);
  ilp1024_four<1>(load,width,shift,a,maximum);
  const float logsum=logf(latch_torch_logsoftmax::reduce1024<false>(a,shared));
  if(threadIdx.x==0) {stats[2*row]=maximum;stats[2*row+1]=logsum;}
  // Gradient from the cached row (grad_row's element ownership, routing and stores).
  const __nv_bfloat16* xr=cache+shift;
  dx+=row*output_pitch;
  const int lead=(4-((row*output_pitch)%4))%4;
  const int input_shift=(row*width+lead)%4;
  const int vector_end=lead+((width-lead-4)/4)*4;
  auto fetch=[&](int k) {
    const uint2* p=reinterpret_cast<const uint2*>(xr+k-input_shift);
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
  int kv=lead+threadIdx.x*4;
  for(;kv+3*1024<vector_end;kv+=4*1024) {
    uint2 words[4];
    #pragma unroll
    for(int u=0;u<4;++u) words[u]=fetch(kv+u*1024);
    #pragma unroll
    for(int u=0;u<4;++u) emit(kv+u*1024,words[u]);
  }
  for(;kv<vector_end;kv+=4*256) emit(kv,fetch(kv));
  if(threadIdx.x<lead) {
    const int k=threadIdx.x;
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(xr[k]),maximum),logsum);
    dx[k]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(k)));
  }
  const int kt=vector_end+threadIdx.x;
  if(kt<width) {
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(xr[kt]),maximum),logsum);
    dx[kt]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(kt)));
  }
  __syncthreads();   // the next row's sparse scratch and cache overwrite this row's storage
}

// LATCH-MOE-EMA-FAST: the same row pass with two row buffers. While a row is reduced and its gradient emitted,
// the CTA's next row (row + stride, the caller's loop stride) is already being copied by cp.async into the
// other buffer. Storage: [32 reduction floats][buffer 0][buffer 1][512 sparse floats]. Each buffer holds the
// row from its virtual FP32 alignment as stats_row_cached does; every value and operation is stats_grad_row's.
__host__ __device__ constexpr long long fused_row_buffer_bytes(int width) {
  return ((long long)(width + 3) * 2 + 127) / 128 * 128;
}
__device__ __forceinline__ void fused_row_copy(const __nv_bfloat16* x, int width, long long row,
                                               __nv_bfloat16* cache) {
  const int shift = (row * width) % 4;
  const auto* source = x + row * width - shift;
  const int size = width + shift, words = size / 4 * 4;
  for (int k = threadIdx.x * 4; k < words; k += 1024) {
    const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(cache + k));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(dst), "l"(source + k) : "memory");
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void stats_grad_row_db(const __nv_bfloat16* x,float* stats,__nv_bfloat16* dx,
    int width,int output_pitch,long long row,int target,float selected,float other,float* shared,
    long long stride,long long rows) {
  const long long buffer = fused_row_buffer_bytes(width);
  char* base = reinterpret_cast<char*>(shared);
  const long long ordinal = row / stride;
  auto* cache = reinterpret_cast<__nv_bfloat16*>(base + 128 + (ordinal & 1) * buffer);
  float* sparse = reinterpret_cast<float*>(base + 128 + 2 * buffer);
  if (row < stride) fused_row_copy(x, width, row, cache);            // a CTA's first row
  const bool ahead = row + stride < rows;
  if (ahead) fused_row_copy(x, width, row + stride,
                            reinterpret_cast<__nv_bfloat16*>(base + 128 + ((ordinal + 1) & 1) * buffer));
  if (ahead) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
  else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
  const int shift = (row * width) % 4;
  {
    const int size = width + shift, words = size / 4 * 4, tail = words + threadIdx.x;
    if (tail < size) cache[tail] = (x + row * width - shift)[tail];
  }
  SparseRow cotangent{target,selected,other};
  const float total=sparse_sum(cotangent,width,row,sparse);          // synchronizes: the buffer is complete
  BF16Row load{cache+shift};float a[4];
  ilp1024_four<0>(load,width,shift,a);
  const float maximum=latch_torch_logsoftmax::reduce1024<true>(a,shared);
  ilp1024_four<1>(load,width,shift,a,maximum);
  const float logsum=logf(latch_torch_logsoftmax::reduce1024<false>(a,shared));
  if(threadIdx.x==0) {stats[2*row]=maximum;stats[2*row+1]=logsum;}
  const __nv_bfloat16* xr=cache+shift;
  dx+=row*output_pitch;
  const int lead=(4-((row*output_pitch)%4))%4;
  const int input_shift=(row*width+lead)%4;
  const int vector_end=lead+((width-lead-4)/4)*4;
  auto fetch=[&](int k) {
    const uint2* p=reinterpret_cast<const uint2*>(xr+k-input_shift);
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
  int kv=lead+threadIdx.x*4;
  for(;kv+3*1024<vector_end;kv+=4*1024) {
    uint2 words[4];
    #pragma unroll
    for(int u=0;u<4;++u) words[u]=fetch(kv+u*1024);
    #pragma unroll
    for(int u=0;u<4;++u) emit(kv+u*1024,words[u]);
  }
  for(;kv<vector_end;kv+=4*256) emit(kv,fetch(kv));
  if(threadIdx.x<lead) {
    const int k=threadIdx.x;
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(xr[k]),maximum),logsum);
    dx[k]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(k)));
  }
  const int kt=vector_end+threadIdx.x;
  if(kt<width) {
    const float logp=__fsub_rn(__fsub_rn(__bfloat162float(xr[kt]),maximum),logsum);
    dx[kt]=__float2bfloat16_rn(__fmaf_rn(-expf(logp),total,cotangent(kt)));
  }
  __syncthreads();   // this buffer is refilled two rows later; the reduction floats by the next row
}
}
