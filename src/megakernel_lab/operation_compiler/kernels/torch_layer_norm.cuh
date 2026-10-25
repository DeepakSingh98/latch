#pragma once
// FP32 vector4 Welford affine LayerNorm, explicit left-product FMA variant.
// Independently expressed from PyTorch v2.9.1 (BSD-3-Clause):
// https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
// https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/thread_constants.h
// Selected from the unchanged CPU-built explicit_left component:
// source SHA256 eca5403d53b2a85b16b58bf218b06113b123702dcccc51c36b3747a81e36e7b5
// binary SHA256 c87fc5eee15728966b01cf65be47aca94d7200f34a3d1cfc2740ce9c7b17d813
// Evidence: research/latch_moe4/results/dev/t-latch-moe4-gpu-17/
// primary-layernorm-components-fresh-v2/rank-{0,1,2,3}.json.
// All 96 same-input comparisons (4 ranks x 8 layers x output/mean/rstd)
// were bitwise equal at rows16384,width768,epsilon1e-5 on H100.
// This component result is not complete-model qualification or a speed claim.
// The common caller uses physical256 threads,128 logical workers and24 shared
// bytes. Every physical thread must participate in this call and its barriers.
#define MODE 1

#include <cuda_runtime.h>
// Algorithm/operation-order attribution: PyTorch v2.9.1 (BSD-3-Clause),
// aten/src/ATen/native/cuda/layer_norm_kernel.cu lines118-299,958-985.
// Independent FP32-only diagnostic. MODE0: ordinary C++ spelling; MODE1/2:
// explicit RN arithmetic, differing only in which mean product forms an FMA.
namespace torch_ln_vector4 {
struct Moments { float mean, m2, count; };
struct __align__(16) Vector4 { float v[4]; };
__device__ __forceinline__ Moments append(Moments a, float value) {
  const float difference = value-a.mean;
  const float count = a.count+1.f;
#if MODE == 0
  const float mean = a.mean+difference*(1.f/count);
  return {mean, a.m2+difference*(value-mean), count};
#else
  const float mean = __fmaf_rn(difference, 1.f/count, a.mean);
  return {mean, __fmaf_rn(difference, __fsub_rn(value,mean), a.m2), count};
#endif
}
// The pinned reduction passes current as B, shuffled/shared partner as A.
// Swapping these is algebraically valid but changes floating-point execution.
__device__ __forceinline__ Moments join(Moments b, Moments a) {
  const float difference = b.mean-a.mean;
  const float count = a.count+b.count;
  if (!(count > 0.f)) return {0.f,0.f,count};
  const float inverse = 1.f/count;
  const float fraction_a = a.count*inverse, fraction_b = b.count*inverse;
#if MODE == 0
  const float mean = fraction_a*a.mean+fraction_b*b.mean;
  const float m2 = a.m2+b.m2+difference*difference*a.count*fraction_b;
#else
#if MODE == 1
  const float mean = __fmaf_rn(fraction_a,a.mean,__fmul_rn(fraction_b,b.mean));
#else
  const float mean = __fmaf_rn(fraction_b,b.mean,__fmul_rn(fraction_a,a.mean));
#endif
  const float cross = __fmul_rn(__fmul_rn(difference,difference),a.count);
  const float m2 = __fmaf_rn(cross,fraction_b,__fadd_rn(a.m2,b.m2));
#endif
  return {mean,m2,count};
}
__device__ __forceinline__ void torch_ln_row(const float* __restrict__ input,
                          const float* __restrict__ gain,
                          const float* __restrict__ bias,
                          float* __restrict__ output,
                          float* __restrict__ mean_out,
                          float* __restrict__ rstd_out,
                          long long row, int width, float epsilon, float* shared) {
  // PyTorch CUDA num_threads() is128 (ROCm uses256). The four logical warps
  // may run in a physical128/256-thread block; inactive threads still join
  // every barrier. Caller supplies6 shared floats and flat lane/warp mapping.
  const int thread=threadIdx.x+threadIdx.y*blockDim.x;
  const int lane=thread&31, warp=thread>>5;
  const int threads=128;
  const long long base=row*width;
  const Vector4* x=reinterpret_cast<const Vector4*>(input+base);
  Moments value{0.f,0.f,0.f};
  for (int i=thread;thread<threads && i<width/4;i+=threads) {
    const Vector4 vector=x[i];
    #pragma unroll
    for(int k=0;k<4;++k) value=append(value,vector.v[k]);
  }
  for(int offset=16;offset>0;offset>>=1) {
    Moments partner{__shfl_down_sync(0xffffffffu,value.mean,offset),
                    __shfl_down_sync(0xffffffffu,value.m2,offset),
                    __shfl_down_sync(0xffffffffu,value.count,offset)};
    value=join(value,partner);
  }
  float* counts=shared+4;
  for(int offset=2;offset>0;offset/=2) {
    if(lane==0 && warp>=offset && warp<2*offset) {
      const int slot=warp-offset;
      shared[2*slot]=value.mean;
      shared[2*slot+1]=value.m2;
      counts[slot]=value.count;
    }
    __syncthreads();
    if(lane==0 && warp<offset) {
      const Moments partner{shared[2*warp],shared[2*warp+1],counts[warp]};
      value=join(value,partner);
    }
    __syncthreads();
  }
  if(thread==0) { shared[0]=value.mean; shared[1]=value.m2/float(width); }
  __syncthreads();
  const float mean=shared[0], rstd=rsqrtf(shared[1]+epsilon);
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  const Vector4* b=reinterpret_cast<const Vector4*>(bias);
  Vector4* y=reinterpret_cast<Vector4*>(output+base);
  for(int i=thread;thread<threads && i<width/4;i+=threads) {
    const Vector4 vector=x[i], scale=g[i], shift=b[i];
    Vector4 result;
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      result.v[k]=scale.v[k]*(rstd*(vector.v[k]-mean))+shift.v[k];
#else
      const float normalized=__fmul_rn(rstd,__fsub_rn(vector.v[k],mean));
      result.v[k]=__fmaf_rn(scale.v[k],normalized,shift.v[k]);
#endif
    }
    y[i]=result;
  }
  if(thread==0) { mean_out[row]=mean; rstd_out[row]=rstd; }
}
// LATCH-MOE-EMA-FAST: two rows per 256-thread CTA, one per 128-thread half. Each half executes exactly the
// torch_ln_row arithmetic for its row (thread t of the half takes torch thread t's vector4 elements, the same
// shuffle tree, the same two-level shared combine and the same output expressions); only the physical thread
// offset and the shared slots (12 floats per half) differ. Both halves join every CTA barrier; a half whose row
// is >= rows computes nothing and stores nothing. Caller supplies 24 shared floats.
__device__ __forceinline__ void torch_ln_row2(const float* __restrict__ input,
                          const float* __restrict__ gain,
                          const float* __restrict__ bias,
                          float* __restrict__ output,
                          float* __restrict__ mean_out,
                          float* __restrict__ rstd_out,
                          long long row0, long long rows, int width, float epsilon, float* shared_all) {
  const int physical=threadIdx.x+threadIdx.y*blockDim.x;
  const int half=physical>>7, thread=physical&127;
  const int lane=thread&31, warp=thread>>5;
  const long long row=row0+half;
  const bool live=row<rows;
  float* shared=shared_all+12*half;
  const long long base=(live?row:0)*width;
  const Vector4* x=reinterpret_cast<const Vector4*>(input+base);
  Moments value{0.f,0.f,0.f};
  for (int i=thread;live && i<width/4;i+=128) {
    const Vector4 vector=x[i];
    #pragma unroll
    for(int k=0;k<4;++k) value=append(value,vector.v[k]);
  }
  for(int offset=16;offset>0;offset>>=1) {
    Moments partner{__shfl_down_sync(0xffffffffu,value.mean,offset),
                    __shfl_down_sync(0xffffffffu,value.m2,offset),
                    __shfl_down_sync(0xffffffffu,value.count,offset)};
    value=join(value,partner);
  }
  float* counts=shared+4;
  for(int offset=2;offset>0;offset/=2) {
    if(lane==0 && warp>=offset && warp<2*offset) {
      const int slot=warp-offset;
      shared[2*slot]=value.mean;
      shared[2*slot+1]=value.m2;
      counts[slot]=value.count;
    }
    __syncthreads();
    if(lane==0 && warp<offset) {
      const Moments partner{shared[2*warp],shared[2*warp+1],counts[warp]};
      value=join(value,partner);
    }
    __syncthreads();
  }
  if(thread==0) { shared[0]=value.mean; shared[1]=value.m2/float(width); }
  __syncthreads();
  const float mean=shared[0], rstd=rsqrtf(shared[1]+epsilon);
  if(!live) return;
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  const Vector4* b=reinterpret_cast<const Vector4*>(bias);
  Vector4* y=reinterpret_cast<Vector4*>(output+base);
  for(int i=thread;i<width/4;i+=128) {
    const Vector4 vector=x[i], scale=g[i], shift=b[i];
    Vector4 result;
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      result.v[k]=scale.v[k]*(rstd*(vector.v[k]-mean))+shift.v[k];
#else
      const float normalized=__fmul_rn(rstd,__fsub_rn(vector.v[k],mean));
      result.v[k]=__fmaf_rn(scale.v[k],normalized,shift.v[k]);
#endif
    }
    y[i]=result;
  }
  if(thread==0) { mean_out[row]=mean; rstd_out[row]=rstd; }
}
// LATCH-MOE-EMA-100 (n01): R rows per 128-thread half (2R per 256-thread CTA). Each row is exactly the
// torch_ln_row2 computation for that row: thread t of the half takes torch thread t's vector4 elements in the same
// order, the same shuffle tree (join(current, partner)), the same two-level shared combine (12 shared floats per row)
// and the same output expressions. Rows are only interleaved, so independent chains overlap their latency (the
// regions run one 256-thread CTA per SM, where the one-row form is latency bound). Caller supplies 24*R floats.
template <int R>
__device__ __forceinline__ void torch_ln_rows(const float* __restrict__ input,
                          const float* __restrict__ gain,
                          const float* __restrict__ bias,
                          float* __restrict__ output,
                          float* __restrict__ mean_out,
                          float* __restrict__ rstd_out,
                          long long row0, long long rows, int width, float epsilon, float* shared_all) {
  const int physical=threadIdx.x+threadIdx.y*blockDim.x;
  const int half=physical>>7, thread=physical&127;
  const int lane=thread&31, warp=thread>>5;
  float* shared=shared_all+12*R*half;
  long long base[R];
  bool live[R];
  Moments value[R];
  #pragma unroll
  for (int j=0;j<R;++j) {
    const long long row=row0+(long long)half*R+j;
    live[j]=row<rows;
    base[j]=(live[j]?row:0)*width;
    value[j]=Moments{0.f,0.f,0.f};
  }
  for (int i=thread;i<width/4;i+=128) {
    Vector4 vector[R];
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) vector[j]=reinterpret_cast<const Vector4*>(input+base[j])[i];
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) {
      #pragma unroll
      for(int k=0;k<4;++k) value[j]=append(value[j],vector[j].v[k]);
    }
  }
  for(int offset=16;offset>0;offset>>=1) {
    #pragma unroll
    for (int j=0;j<R;++j) {
      Moments partner{__shfl_down_sync(0xffffffffu,value[j].mean,offset),
                      __shfl_down_sync(0xffffffffu,value[j].m2,offset),
                      __shfl_down_sync(0xffffffffu,value[j].count,offset)};
      value[j]=join(value[j],partner);
    }
  }
  for(int offset=2;offset>0;offset/=2) {
    if(lane==0 && warp>=offset && warp<2*offset) {
      const int slot=warp-offset;
      #pragma unroll
      for (int j=0;j<R;++j) {
        float* s=shared+12*j;
        s[2*slot]=value[j].mean;
        s[2*slot+1]=value[j].m2;
        s[4+slot]=value[j].count;
      }
    }
    __syncthreads();
    if(lane==0 && warp<offset) {
      #pragma unroll
      for (int j=0;j<R;++j) {
        const float* s=shared+12*j;
        const Moments partner{s[2*warp],s[2*warp+1],s[4+warp]};
        value[j]=join(value[j],partner);
      }
    }
    __syncthreads();
  }
  if(thread==0) {
    #pragma unroll
    for (int j=0;j<R;++j) { shared[12*j]=value[j].mean; shared[12*j+1]=value[j].m2/float(width); }
  }
  __syncthreads();
  float mean[R], rstd[R];
  #pragma unroll
  for (int j=0;j<R;++j) { mean[j]=shared[12*j]; rstd[j]=rsqrtf(shared[12*j+1]+epsilon); }
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  const Vector4* b=reinterpret_cast<const Vector4*>(bias);
  for(int i=thread;i<width/4;i+=128) {
    const Vector4 scale=g[i], shift=b[i];
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) {
      const Vector4 vector=reinterpret_cast<const Vector4*>(input+base[j])[i];
      Vector4 result;
      #pragma unroll
      for(int k=0;k<4;++k) {
#if MODE == 0
        result.v[k]=scale.v[k]*(rstd[j]*(vector.v[k]-mean[j]))+shift.v[k];
#else
        const float normalized=__fmul_rn(rstd[j],__fsub_rn(vector.v[k],mean[j]));
        result.v[k]=__fmaf_rn(scale.v[k],normalized,shift.v[k]);
#endif
      }
      reinterpret_cast<Vector4*>(output+base[j])[i]=result;
    }
  }
  if(thread==0) {
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) { mean_out[base[j]/width]=mean[j]; rstd_out[base[j]/width]=rstd[j]; }
  }
}
} // namespace torch_ln_vector4

#undef MODE
