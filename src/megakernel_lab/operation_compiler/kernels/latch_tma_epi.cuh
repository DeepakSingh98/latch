// Release step 14: generated same-index epilogues on the staged pops4b TMA + wgmma body (pops4b::tma_body STAGED).
//
// The body stages the BF16-rounded tile in shared memory and calls epi.vec8(i, x) for every 8-column chunk of a
// row: i is the flat index of the chunk's first element (row * ldc + column, column % 8 == 0) and x[0..7] are the
// rounded product values widened to FP32 (what a load of the stored BF16 result yields). The generated consumer
// lines are the compose planner's own elementwise lines (the P7-C p5ce run(e, i0, gv) / ld(e, i0) protocol), so
// every expression, operand bit and rounding is unchanged; only the product's write/read round trip, the consumer's
// grid phase and its barrier disappear. Data-movement helpers are copies of p5c_wg.cuh's (bit-identical packing).
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

// Step-14 speed ('xdiv'): IEEE round-to-nearest division and reciprocal as the straight-line fast paths that
// ptxas emits for div.rn.f32 / rcp.rn.f32 on sm_90 (MUFU.RCP, then the same FFMA sequence), without the per-element
// slow-path branch region that serializes an epilogue's independent elements. ok_ is cleared whenever an operand
// leaves a conservative exponent domain in which that fast path is the correctly rounded result; the epilogue then
// recomputes its whole chunk with the original expressions (a / b, __frcp_rn), so every stored bit is unchanged.
#ifndef LATCH_XDIV_DEFINED
#define LATCH_XDIV_DEFINED
__device__ __forceinline__ float latch_rcp_approx(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
// rcp.rn fast path (taken by ptxas when ((bits(x) + 0x1800000) & 0x7f800000) > 0x1ffffff): MUFU.RCP, e = x r - 1,
// r + r (-e).
__device__ __forceinline__ float latch_rcp_f(bool& ok, float x) {
  ok &= ((__float_as_uint(x) + 0x1800000u) & 0x7f800000u) > 0x1ffffffu;
  const float r = latch_rcp_approx(x);
  const float e = fmaf(x, r, -1.0f);
  return fmaf(r, -e, r);
}
// div.rn fast path: r = MUFU.RCP(b); r = r + r (1 - b r); q = a r; q + r (a - b q). Domain: a and b normal with
// unbiased exponents in [-40, 40] (no intermediate overflow, underflow or subnormal quotient).
__device__ __forceinline__ float latch_div_f(bool& ok, float a, float b) {
  const unsigned ea = (__float_as_uint(a) >> 23) & 0xffu, eb = (__float_as_uint(b) >> 23) & 0xffu;
  ok &= (ea - 87u <= 80u) & (eb - 87u <= 80u);
  float r = latch_rcp_approx(b);
  const float e = fmaf(-b, r, 1.0f);
  r = fmaf(r, e, r);
  const float q = fmaf(a, r, 0.0f);
  const float rem = fmaf(-b, q, a);
  return fmaf(r, rem, q);
}
#endif

namespace tmaepi {

__device__ __forceinline__ __nv_bfloat16 upkb(const uint4& q, int j) {   // 8 BF16 values in one 16-byte chunk
  const uint32_t w = (j >> 1) == 0 ? q.x : (j >> 1) == 1 ? q.y : (j >> 1) == 2 ? q.z : q.w;
  const unsigned short b = (j & 1) ? (unsigned short)(w >> 16) : (unsigned short)(w & 0xffffu);
  return __ushort_as_bfloat16(b);
}
__device__ __forceinline__ void st8(float* p, const float (&v)[8]) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void st8(__nv_bfloat16* p, const __nv_bfloat16 (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k)
    w[k] = (uint32_t)__bfloat16_as_ushort(v[2 * k]) | ((uint32_t)__bfloat16_as_ushort(v[2 * k + 1]) << 16);
  *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// ld(e, i0) loads the chunk's identity-indexed side inputs into slot e; run(e, i0, gv) evaluates the 8 elements.
template <class LD, class RUN> struct Vec8 {
  LD ld;
  RUN run;
  __device__ __forceinline__ void vec8(unsigned i, const float (&g)[8]) const {
    ld(0, i);
    run(0, i, g);
  }
};
template <class LD, class RUN> __device__ __forceinline__ Vec8<LD, RUN> vec8(LD ld, RUN run) { return {ld, run}; }

// Step-14 speed ('tepf'): the same epilogue with pre(i), the pops4b P6-B hook called for the task's output positions
// when the task starts; pf(i) issues L2 prefetches of the identity-indexed side inputs at flat index i.
template <class LD, class RUN, class PF> struct Vec8P {
  LD ld;
  RUN run;
  PF pf;
  __device__ __forceinline__ void vec8(unsigned i, const float (&g)[8]) const {
    ld(0, i);
    run(0, i, g);
  }
  __device__ __forceinline__ void pre(unsigned i) const { pf(i); }
};
template <class LD, class RUN, class PF>
__device__ __forceinline__ Vec8P<LD, RUN, PF> vec8p(LD ld, RUN run, PF pf) { return {ld, run, pf}; }

// Step-14 speed ('tepr'): the same epilogue with prow(i, n): one bulk L2 prefetch per tile row and older side input.
template <class LD, class RUN, class PR> struct Vec8R {
  LD ld;
  RUN run;
  PR pr;
  __device__ __forceinline__ void vec8(unsigned i, const float (&g)[8]) const {
    ld(0, i);
    run(0, i, g);
  }
  __device__ __forceinline__ void prow(unsigned i, unsigned n) const { pr(i, n); }
};
template <class LD, class RUN, class PR>
__device__ __forceinline__ Vec8R<LD, RUN, PR> vec8r(LD ld, RUN run, PR pr) { return {ld, run, pr}; }

}  // namespace tmaepi
