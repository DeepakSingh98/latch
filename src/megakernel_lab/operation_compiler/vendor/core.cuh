// [S109A component copy of the pinned donor core.cuh: unchanged unless option xrcp (S109A_XRCP) is set]
// S101H native core: grid barrier, warp reductions, small math helpers.
// Numerical policy: FP32 everywhere (FFMA SIMT, no TF32); BF16 only where the recipe rounds to BF16 (Muon NS5).
// Every reduction has a fixed order (no atomics on data), so a launch is a deterministic function of its inputs.
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define DEV __device__ __forceinline__
#define NTHREADS 256

namespace s101h {

// ------------------------------------------------------------------------------------------ grid barrier
// ctl[0] = arrival counter (always returns to 0), ctl[1] = generation (monotonic). The control words live in a
// separate, zero-initialized buffer (never poisoned). Sense is the generation snapshot taken before arriving.
struct Grid {
  unsigned* ctl;
  unsigned nblocks;
};

DEV void grid_sync(const Grid& g) {
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile unsigned* gen = g.ctl + 1;
    const unsigned snap = *gen;
    __threadfence();
    const unsigned arrived = atomicAdd(g.ctl, 1u);
    if (arrived == g.nblocks - 1) {
      atomicExch(g.ctl, 0u);
      __threadfence();
      atomicAdd(g.ctl + 1, 1u);
    } else {
      while (*gen == snap) { __nanosleep(32); }
    }
    __threadfence();
  }
  __syncthreads();
}

// S102G option bar2 (transferred from S100I rule B; synchronization only, no arithmetic change): one monotone 64-bit
// arrival counter at ctl[0..1]; barrier k of a launch completes when the counter reaches base + k * nblocks. Arrival is
// a release reduction by thread 0 after the CTA barrier (cumulative over the CTA's prior writes), the wait an acquire
// load (orders every later access of the CTA via the trailing CTA barrier). base (ctl[2..3]) is read by every CTA at
// kernel start and advanced by CTA 0 after the launch's last barrier, when every CTA has already read it.
DEV unsigned long long ld_acq_u64(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
DEV void grid_sync2(unsigned* ctl, unsigned long long target) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long* ctr = reinterpret_cast<unsigned long long*>(ctl);
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" ::"l"(ctr) : "memory");
    while (ld_acq_u64(ctr) < target) {}
  }
  __syncthreads();
}

// S102G option lr: an opaque copy of a pointer (empty volatile asm) so that addresses derived from it inside one phase
// are not hoisted or kept live across phases (keeps the kernel body's live set small at __noinline__ call sites).
template <class T> DEV T* opq(T* p) { asm volatile("" : "+l"(p)); return p; }

// ------------------------------------------------------------------------------------------ warp helpers
DEV float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEV float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// sum over the 16-lane half-warp group containing this lane
DEV float half_sum(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEV float half_max(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// optional in-GEMM task timestamps (S101H_GTIME): generated code sets d_gt_base[blockIdx.x] to this phase's slot
#if defined(S101H_GTIME)
__device__ unsigned long long* d_gt_base[264];
DEV void gt_stamp(int j) {
  if (threadIdx.x == 0 && d_gt_base[blockIdx.x]) {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    d_gt_base[blockIdx.x][j] = t;
  }
}
#else
DEV void gt_stamp(int) {}
#endif

#ifdef S109A_XRCP
// [S109A option xrcp] the donor's IEEE `1.0f / x` inside sigmoidf_ compiles to MUFU.RCP + FFMA + a per-element slow-path
// CALL; that call structure splits unrolled epilogues into per-element blocks. s109a_rcp_rn is branch-free and was checked
// bit for bit against `1.0f / x` on all 2^32 FP32 inputs, and sigmoidf_ against the donor's on all 2^32 inputs
// (s109a/exact.py candidate c5, receipt attempts/S109A-exact-3: 0 mismatches incl. NaN bits), so every sigmoidf_
// result (and every value derived from it) is unchanged.
DEV float s109a_rcp_rn(float x) {
  const double d = (double)x;
  double r;
  asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(d));
  double e = __fma_rn(-d, r, 1.0);
  r = __fma_rn(r, e, r);
  e = __fma_rn(-d, r, 1.0);
  r = __fma_rn(r, e, r);
  float f = __double2float_rn(r);
  f = isinf(x) ? copysignf(0.0f, x) : f;
  f = (x == 0.0f) ? copysignf(__int_as_float(0x7f800000), x) : f;
  f = isnan(x) ? __int_as_float(0x7fffffff) : f;                  // the donor division's NaN result bits
  return f;
}
DEV float sigmoidf_(float z) { return s109a_rcp_rn(1.0f + expf(-z)); }
#else
DEV float sigmoidf_(float z) { return 1.0f / (1.0f + expf(-z)); }
#endif
DEV float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

}  // namespace s101h
