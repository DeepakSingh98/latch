#pragma once
#include <cuda_runtime.h>
// PyTorch v2.9.1 algorithm attribution (BSD-3-Clause):
// https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu
// Explicit affine-gradient contract. Algorithm/order independently expressed
// from PyTorch2.9.1 GammaBetaBackwardCUDAKernelTemplate<float,float,32,32,256,
// false,true,false> (BSD-3-Clause source above). The standalone 1024
// thread final-FMA body matched both original derivatives bitwise on all32
// primary aux rank/layer tapes. Here256 physical threads each retain four
// independent logical thread accumulators; the original tree is unchanged.
namespace torch_ln_affine {
__device__ __forceinline__ void tile(const float* x,const float* dy,
    const float* mean,const float* rstd,float* dg,float* db,int column_tile,float* shared) {
  constexpr int WIDTH=768,ROWS=16384,LOGICAL_Y=32,STRIDE=33,SLOTS=4;
  const int thread=threadIdx.x+threadIdx.y*blockDim.x;
  const int tx=thread&31,warp=thread>>5,column=column_tile*32+tx;
  float gs[SLOTS]={0.f,0.f,0.f,0.f},bs[SLOTS]={0.f,0.f,0.f,0.f};
  float* gains=shared;float* biases=shared+LOGICAL_Y*STRIDE;
  for(int row_tile=0;row_tile<ROWS;row_tile+=256) {
    #pragma unroll
    for(int slot=0;slot<SLOTS;++slot) {
      const int ty=warp+slot*8,start=row_tile+ty*8;
      #pragma unroll
      for(int i=0;i<8;++i) {
        const int row=start+i,index=row*WIDTH+column;
        const float delta=__fsub_rn(x[index],mean[row]);
        const float weighted=__fmul_rn(dy[index],delta);
        gs[slot]=__fmaf_rn(weighted,rstd[row],gs[slot]);
        bs[slot]=__fadd_rn(bs[slot],dy[index]);
      }
    }
  }
  #pragma unroll
  for(int slot=0;slot<SLOTS;++slot) {
    const int ty=warp+slot*8;
    gains[ty*STRIDE+tx]=gs[slot];biases[ty*STRIDE+tx]=bs[slot];
  }
  __syncthreads();
  #pragma unroll
  for(int slot=0;slot<SLOTS;++slot) {
    const int output_column=warp+slot*8;
    float g=gains[tx*STRIDE+output_column],b=biases[tx*STRIDE+output_column];
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1) {
      g=__fadd_rn(g,__shfl_xor_sync(0xffffffffu,g,offset));
      b=__fadd_rn(b,__shfl_xor_sync(0xffffffffu,b,offset));
    }
    if(tx==0) {dg[column_tile*32+output_column]=g;db[column_tile*32+output_column]=b;}
  }
  // Shared storage is reusable only after all virtual columns have read it.
  __syncthreads();
}

// LATCH-MOE-EMA-FAST split form of tile(): the 32x32 logical threads' serial row chains of a column tile are
// computed by four CTAs (one logical y slot each, item = column_tile * 4 + slot) instead of one, and written to
// `chains` ([2][tiles][32 y][32 x]); combine() then runs tile()'s unchanged transpose + XOR tree on them.
// Every chain visits the same rows in the same order with the same three operations as tile().
__device__ __forceinline__ void chains(const float* x,const float* dy,const float* mean,const float* rstd,
    float* out,int item) {
  constexpr int WIDTH=768,ROWS=16384,TILES=WIDTH/32;
  const int thread=threadIdx.x+threadIdx.y*blockDim.x;
  const int tx=thread&31,warp=thread>>5,column_tile=item>>2,slot=item&3;
  const int ty=warp+slot*8,column=column_tile*32+tx;
  float g=0.f,b=0.f;
  for(int row_tile=0;row_tile<ROWS;row_tile+=256) {
    const int start=row_tile+ty*8;
    #pragma unroll
    for(int i=0;i<8;++i) {
      const int row=start+i,index=row*WIDTH+column;
      const float delta=__fsub_rn(x[index],mean[row]);
      const float weighted=__fmul_rn(dy[index],delta);
      g=__fmaf_rn(weighted,rstd[row],g);
      b=__fadd_rn(b,dy[index]);
    }
  }
  out[(column_tile*32+ty)*32+tx]=g;
  out[((TILES+column_tile)*32+ty)*32+tx]=b;
}
__device__ __forceinline__ void combine(const float* in,float* dg,float* db,int column_tile,float* shared) {
  constexpr int WIDTH=768,TILES=WIDTH/32,LOGICAL_Y=32,STRIDE=33,SLOTS=4;
  const int thread=threadIdx.x+threadIdx.y*blockDim.x;
  const int tx=thread&31,warp=thread>>5;
  float* gains=shared;float* biases=shared+LOGICAL_Y*STRIDE;
  #pragma unroll
  for(int slot=0;slot<SLOTS;++slot) {
    const int ty=warp+slot*8;
    gains[ty*STRIDE+tx]=in[(column_tile*32+ty)*32+tx];
    biases[ty*STRIDE+tx]=in[((TILES+column_tile)*32+ty)*32+tx];
  }
  __syncthreads();
  #pragma unroll
  for(int slot=0;slot<SLOTS;++slot) {
    const int output_column=warp+slot*8;
    float g=gains[tx*STRIDE+output_column],b=biases[tx*STRIDE+output_column];
    #pragma unroll
    for(int offset=16;offset>0;offset>>=1) {
      g=__fadd_rn(g,__shfl_xor_sync(0xffffffffu,g,offset));
      b=__fadd_rn(b,__shfl_xor_sync(0xffffffffu,b,offset));
    }
    if(tx==0) {dg[column_tile*32+output_column]=g;db[column_tile*32+output_column]=b;}
  }
  __syncthreads();
}
}
