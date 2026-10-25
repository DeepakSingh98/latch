#pragma once
// Measured shared LayerNorm dX arithmetic, explicit FP32 contraction only.
// Body below is byte-identical to the CPU-built shared_left candidate from
// latch_moe4/layernorm_dx_component.py v1. All32 primary aux rank/layer dX
// and weighted BF16 cotangent comparisons were bitwise equal to Torch2.9.1.
// PyTorch algorithm attribution (BSD-3-Clause):
// https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
// https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/block_reduce.cuh
// Caller: physical128/256 threads,128 logical,4 shared floats, aligned FP32
// vector4 rows; numerical proof presently width768, not every admitted width.
#define MODE 4

#include <cuda_runtime.h>
// Independent algorithm/order implementation attributed to PyTorch v2.9.1
// (BSD-3-Clause): CUDA vectorized LayerNorm dX and BlockReduceSum.
namespace torch_ln_dx {
struct __align__(16) Vector4 { float v[4]; };
__device__ __forceinline__ float warp_sum(float x) {
  #pragma unroll
  for (int offset=16;offset>0;offset>>=1)
    x=__fadd_rn(x,__shfl_down_sync(0xffffffffu,x,offset));
  return x;
}
__device__ __forceinline__ float block_sum(float x,float* shared,int thread) {
  const int lane=thread&31,warp=thread>>5;
  x=warp_sum(x);
  __syncthreads();
  if(lane==0 && thread<128) shared[warp]=x;
  __syncthreads();
  x=thread<4?shared[lane]:0.f;
  if(warp==0) x=warp_sum(x);
  return x;
}
__device__ __forceinline__ void row(const float* input,const float* dy,const float* gain,
    const float* mean,const float* rstd,float* dx,long long row,int width,float* shared) {
  const int thread=threadIdx.x+threadIdx.y*blockDim.x;
  const long long base=row*width;
  const Vector4* x=reinterpret_cast<const Vector4*>(input+base);
  const Vector4* y=reinterpret_cast<const Vector4*>(dy+base);
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  Vector4* result=reinterpret_cast<Vector4*>(dx+base);
  const float mu=mean[row],rs=rstd[row];
  float sum1=0.f,sum2=0.f;
  for(int index=thread;thread<128 && index<width/4;index+=128) {
    const Vector4 xv=x[index],yv=y[index],gv=g[index];
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      sum1+=yv.v[k]*gv.v[k];
      sum2+=yv.v[k]*gv.v[k]*(xv.v[k]-mu)*rs;
#else
      const float product=__fmul_rn(yv.v[k],gv.v[k]);
#if MODE < 3
      sum1=__fmaf_rn(yv.v[k],gv.v[k],sum1);
#else
      sum1=__fadd_rn(product,sum1);
#endif
      sum2=__fmaf_rn(__fmul_rn(product,__fsub_rn(xv.v[k],mu)),rs,sum2);
#endif
    }
  }
  sum1=block_sum(sum1,shared,thread);
  sum2=block_sum(sum2,shared,thread);
  if(thread==0) {shared[0]=sum1;shared[1]=sum2;}
  __syncthreads();
  sum1=shared[0];sum2=shared[1];
  const float scale=(1.f/float(width))*rs;
  for(int index=thread;thread<128 && index<width/4;index+=128) {
    const Vector4 xv=x[index],yv=y[index],gv=g[index];
    Vector4 out;
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      float value=float(width)*gv.v[k]*yv.v[k];
      value-=(xv.v[k]-mu)*rs*sum2;
      value-=sum1;
      out.v[k]=value*scale;
#else
      const float leading=__fmul_rn(float(width),gv.v[k]);
      const float centered=__fmul_rn(__fsub_rn(xv.v[k],mu),rs);
#if MODE == 1 || MODE == 3
      const float difference=__fmaf_rn(-centered,sum2,__fmul_rn(leading,yv.v[k]));
#else
      const float difference=__fmaf_rn(leading,yv.v[k],-__fmul_rn(centered,sum2));
#endif
      out.v[k]=__fmul_rn(__fsub_rn(difference,sum1),scale);
#endif
    }
    result[index]=out;
  }
}
// LATCH-MOE-EMA-FAST: two rows per 256-thread CTA, one per 128-thread half; each half runs exactly row()'s
// arithmetic (thread t of the half is torch thread t, same block_sum tree) with its own 8 shared floats.
// Every barrier is joined by both halves; a half with row >= rows computes and stores nothing.
__device__ __forceinline__ float block_sum2(float x,float* shared,int thread) {
  const int lane=thread&31,warp=thread>>5;
  x=warp_sum(x);
  __syncthreads();
  if(lane==0) shared[warp]=x;
  __syncthreads();
  x=thread<4?shared[lane]:0.f;
  if(warp==0) x=warp_sum(x);
  return x;
}
__device__ __forceinline__ void row2(const float* input,const float* dy,const float* gain,
    const float* mean,const float* rstd,float* dx,long long row0,long long rows,int width,float* shared_all) {
  const int physical=threadIdx.x+threadIdx.y*blockDim.x;
  const int half=physical>>7, thread=physical&127;
  const long long row=row0+half;
  const bool live=row<rows;
  float* shared=shared_all+8*half;
  const long long base=(live?row:0)*width;
  const Vector4* x=reinterpret_cast<const Vector4*>(input+base);
  const Vector4* y=reinterpret_cast<const Vector4*>(dy+base);
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  Vector4* result=reinterpret_cast<Vector4*>(dx+base);
  const float mu=live?mean[row]:0.f,rs=live?rstd[row]:0.f;
  float sum1=0.f,sum2=0.f;
  for(int index=thread;live && index<width/4;index+=128) {
    const Vector4 xv=x[index],yv=y[index],gv=g[index];
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      sum1+=yv.v[k]*gv.v[k];
      sum2+=yv.v[k]*gv.v[k]*(xv.v[k]-mu)*rs;
#else
      const float product=__fmul_rn(yv.v[k],gv.v[k]);
#if MODE < 3
      sum1=__fmaf_rn(yv.v[k],gv.v[k],sum1);
#else
      sum1=__fadd_rn(product,sum1);
#endif
      sum2=__fmaf_rn(__fmul_rn(product,__fsub_rn(xv.v[k],mu)),rs,sum2);
#endif
    }
  }
  sum1=block_sum2(sum1,shared,thread);
  sum2=block_sum2(sum2,shared,thread);
  if(thread==0) {shared[0]=sum1;shared[1]=sum2;}
  __syncthreads();
  sum1=shared[0];sum2=shared[1];
  if(!live) return;
  const float scale=(1.f/float(width))*rs;
  for(int index=thread;index<width/4;index+=128) {
    const Vector4 xv=x[index],yv=y[index],gv=g[index];
    Vector4 out;
    #pragma unroll
    for(int k=0;k<4;++k) {
#if MODE == 0
      float value=float(width)*gv.v[k]*yv.v[k];
      value-=(xv.v[k]-mu)*rs*sum2;
      value-=sum1;
      out.v[k]=value*scale;
#else
      const float leading=__fmul_rn(float(width),gv.v[k]);
      const float centered=__fmul_rn(__fsub_rn(xv.v[k],mu),rs);
#if MODE == 1 || MODE == 3
      const float difference=__fmaf_rn(-centered,sum2,__fmul_rn(leading,yv.v[k]));
#else
      const float difference=__fmaf_rn(leading,yv.v[k],-__fmul_rn(centered,sum2));
#endif
      out.v[k]=__fmul_rn(__fsub_rn(difference,sum1),scale);
#endif
    }
    result[index]=out;
  }
}
// LATCH-MOE-EMA-100 (n01): R rows per 128-thread half. Each row is exactly row2()'s arithmetic (thread t of the half
// is torch thread t, the same accumulation order, the same warp_sum / four-partial / warp-0 warp_sum tree, the same
// output expressions); rows are interleaved so their latency overlaps. 12 shared floats per row (partials of the two
// statistics and their broadcast), so no slot is reused within a call. Caller supplies 24*R floats.
template <int R>
__device__ __forceinline__ void rows(const float* input,const float* dy,const float* gain,
    const float* mean,const float* rstd,float* dx,long long row0,long long rows_total,int width,float* shared_all) {
  const int physical=threadIdx.x+threadIdx.y*blockDim.x;
  const int half=physical>>7, thread=physical&127;
  const int lane=thread&31, warp=thread>>5;
  float* shared=shared_all+12*R*half;
  long long base[R];
  bool live[R];
  float mu[R], rs[R], sum1[R], sum2[R];
  #pragma unroll
  for (int j=0;j<R;++j) {
    const long long row=row0+(long long)half*R+j;
    live[j]=row<rows_total;
    base[j]=(live[j]?row:0)*width;
    mu[j]=live[j]?mean[row]:0.f;
    rs[j]=live[j]?rstd[row]:0.f;
    sum1[j]=0.f; sum2[j]=0.f;
  }
  const Vector4* g=reinterpret_cast<const Vector4*>(gain);
  for(int index=thread;index<width/4;index+=128) {
    const Vector4 gv=g[index];
    Vector4 xv[R], yv[R];
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) {
      xv[j]=reinterpret_cast<const Vector4*>(input+base[j])[index];
      yv[j]=reinterpret_cast<const Vector4*>(dy+base[j])[index];
    }
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) {
      #pragma unroll
      for(int k=0;k<4;++k) {
#if MODE == 0
        sum1[j]+=yv[j].v[k]*gv.v[k];
        sum2[j]+=yv[j].v[k]*gv.v[k]*(xv[j].v[k]-mu[j])*rs[j];
#else
        const float product=__fmul_rn(yv[j].v[k],gv.v[k]);
#if MODE < 3
        sum1[j]=__fmaf_rn(yv[j].v[k],gv.v[k],sum1[j]);
#else
        sum1[j]=__fadd_rn(product,sum1[j]);
#endif
        sum2[j]=__fmaf_rn(__fmul_rn(product,__fsub_rn(xv[j].v[k],mu[j])),rs[j],sum2[j]);
#endif
      }
    }
  }
  #pragma unroll
  for (int j=0;j<R;++j) { sum1[j]=warp_sum(sum1[j]); sum2[j]=warp_sum(sum2[j]); }
  __syncthreads();
  if(lane==0) {
    #pragma unroll
    for (int j=0;j<R;++j) { shared[12*j+warp]=sum1[j]; shared[12*j+4+warp]=sum2[j]; }
  }
  __syncthreads();
  #pragma unroll
  for (int j=0;j<R;++j) {
    float a=thread<4?shared[12*j+lane]:0.f, b=thread<4?shared[12*j+4+lane]:0.f;
    if(warp==0) { a=warp_sum(a); b=warp_sum(b); }
    sum1[j]=a; sum2[j]=b;
  }
  if(thread==0) {
    #pragma unroll
    for (int j=0;j<R;++j) { shared[12*j+8]=sum1[j]; shared[12*j+9]=sum2[j]; }
  }
  __syncthreads();
  #pragma unroll
  for (int j=0;j<R;++j) { sum1[j]=shared[12*j+8]; sum2[j]=shared[12*j+9]; }
  for(int index=thread;index<width/4;index+=128) {
    const Vector4 gv=g[index];
    #pragma unroll
    for (int j=0;j<R;++j) if (live[j]) {
      const float scale=(1.f/float(width))*rs[j];
      const Vector4 xv=reinterpret_cast<const Vector4*>(input+base[j])[index];
      const Vector4 yv=reinterpret_cast<const Vector4*>(dy+base[j])[index];
      Vector4 out;
      #pragma unroll
      for(int k=0;k<4;++k) {
#if MODE == 0
        float value=float(width)*gv.v[k]*yv.v[k];
        value-=(xv.v[k]-mu[j])*rs[j]*sum2[j];
        value-=sum1[j];
        out.v[k]=value*scale;
#else
        const float leading=__fmul_rn(float(width),gv.v[k]);
        const float centered=__fmul_rn(__fsub_rn(xv.v[k],mu[j]),rs[j]);
#if MODE == 1 || MODE == 3
        const float difference=__fmaf_rn(-centered,sum2[j],__fmul_rn(leading,yv.v[k]));
#else
        const float difference=__fmaf_rn(leading,yv.v[k],-__fmul_rn(centered,sum2[j]));
#endif
        out.v[k]=__fmul_rn(__fsub_rn(difference,sum1[j]),scale);
#endif
      }
      reinterpret_cast<Vector4*>(dx+base[j])[index]=out;
    }
  }
}
}

#undef MODE
