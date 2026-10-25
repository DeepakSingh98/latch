// LATCH-STEP10-SPEED: branch-free correctly rounded FP32 division for generated element code.
//
// div.rn.f32 compiles to a reciprocal-refinement fast path (MUFU.RCP, one Newton step, q0 = x*r, exact remainder by
// FMA, one corrected quotient) guarded per division by a range check and a branch to a slow-path subroutine. The
// branch ends the basic block, so a chunk's independent divisions cannot be interleaved with each other or with the
// surrounding transcendental chains. latch_xdiv evaluates the same fast-path sequence without a branch and clears
// `ok` when either operand lies outside a window in which the sequence is the correctly rounded quotient (normal
// operands with |x|, |y| in [2^-60, 2^61): no overflow, underflow or subnormal anywhere). Callers evaluate a whole
// chunk this way and recompute it with the original `/` expressions when any division cleared `ok`, so every stored
// result is the IEEE quotient (verified bitwise against `/` by research/latch_release/probes/xdiv_probe.cu).
#pragma once

static __device__ __forceinline__ float latch_xdiv(float x, float y, bool& ok) {
  const unsigned ex = (__float_as_uint(x) >> 23) & 0xffu, ey = (__float_as_uint(y) >> 23) & 0xffu;
  ok = ok & (ex - 67u <= 120u) & (ey - 67u <= 120u);
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y));
  r = __fmaf_rn(__fmaf_rn(-y, r, 1.0f), r, r);
  const float q = __fmul_rn(x, r);
  return __fmaf_rn(__fmaf_rn(-y, q, x), r, q);
}
