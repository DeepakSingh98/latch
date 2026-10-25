// P6-E TMA + warpgroup-MMA BF16 body with masked tails (new code in this lane).
//
// Main loop from P-OPS5-C p5c::body (kernels/p5c_wg.cuh @ 4e2b74545, pinned donor; helpers below copied from it):
//   * per-tile loop with a peeled first 64-k stage whose first k16 step writes the accumulators only (scale-d 0,
//     "=f" outputs), then an inner k loop entered and re-entered with exactly one committed wgmma batch in flight and
//     wait<0> before the epilogue: ptxas keeps the wgmma pipeline (no C7518 serialization of P-OPS4-B's flattened
//     task x k loop with the epilogue inside a conditional);
//   * branch-free producer: TMA issue / expect-tx predicated inside asm on one elected thread; CUTLASS-form wait.
// Tails as P-OPS4-B tma_body (kernels/pops4b_tma.cuh): ceil-divided tiles and 64-k stages; TMA fills absent operand
// elements with +0 (exact +0 products); stores are masked. Results are 16-byte quad-transposed row chunks, so N % 8 == 0
// (absent or present chunks are whole).
// Arithmetic: identical per element to pops4b::tma_body (FP32 wgmma accumulation from +0 in ascending k16 steps, one
// cvt.rn.bf16x2 rounding); bitwise equal to it for every admitted product.
#pragma once
#include <type_traits>
#include <utility>
#include "pops4b_tma.cuh"

namespace p6e {

using s101h::bf16;
using pops4b::Cfg;
using pops4b::sa;

DEV void expect_if(bool leader, unsigned bar, unsigned bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %2;\n}\n"
               ::"r"((unsigned)leader), "r"(bar), "r"(bytes) : "memory");
}
DEV void tma2d_if(bool leader, unsigned dst, const void* map, int c0, int c1, unsigned bar) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%1], [%2, {%3, %4}], [%5];\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DEV void wait_parity(unsigned bar, unsigned parity) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\n"
               "bra LAB_WAIT;\nDONE:\n}\n" ::"r"(bar), "r"(parity) : "memory");
}

// First k16 step of a tile with write-only "=f" accumulators (scale-d = 0; generated as P-OPS5-C gen_first.py).
template <int TA, int TB> DEV void first_wg32(float (&d)[16], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1, %18, %19;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg64(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, %34, %35;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg96(float (&d)[48], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, %48, %49, 0, 1, 1, %50, %51;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg128(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1, %66, %67;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg192(float (&d)[96], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 0, 1, 1, %98, %99;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg256(float (&d)[128], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 0, 1, 1, %130, %131;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]), "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]), "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]), "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]), "=f"(d[110]), "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]), "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]), "=f"(d[120]), "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]), "=f"(d[126]), "=f"(d[127])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int N> struct FirstWg;
template <> struct FirstWg<32> { template <int TA, int TB> DEV static void run(float (&d)[16], uint64_t da, uint64_t db) { first_wg32<TA, TB>(d, da, db); } };
template <> struct FirstWg<64> { template <int TA, int TB> DEV static void run(float (&d)[32], uint64_t da, uint64_t db) { first_wg64<TA, TB>(d, da, db); } };
template <> struct FirstWg<96> { template <int TA, int TB> DEV static void run(float (&d)[48], uint64_t da, uint64_t db) { first_wg96<TA, TB>(d, da, db); } };
template <> struct FirstWg<128> { template <int TA, int TB> DEV static void run(float (&d)[64], uint64_t da, uint64_t db) { first_wg128<TA, TB>(d, da, db); } };
template <> struct FirstWg<192> { template <int TA, int TB> DEV static void run(float (&d)[96], uint64_t da, uint64_t db) { first_wg192<TA, TB>(d, da, db); } };
template <> struct FirstWg<256> { template <int TA, int TB> DEV static void run(float (&d)[128], uint64_t da, uint64_t db) { first_wg256<TA, TB>(d, da, db); } };

// Generated epilogues (P6-E, after P-OPS5-C p5c::body) see 8 contiguous outputs of the rounded product widened to
// FP32; Ld prefetches their identity operands for up to 8 chunks before any is computed.
struct NoEpi {
  DEV void operator()(int, unsigned, const float (&)[8]) const {}
};
struct NoLd {
  DEV void operator()(int, unsigned) const {}
};
template <class T> struct IsNoEpi { static constexpr bool value = false; };
template <> struct IsNoEpi<NoEpi> { static constexpr bool value = true; };
DEV float upk(const uint4 (&r)[2], int j) {                       // 8 FP32 values in two chunks
  const uint4 q = r[j >> 2];
  const uint32_t w = (j & 3) == 0 ? q.x : (j & 3) == 1 ? q.y : (j & 3) == 2 ? q.z : q.w;
  return __uint_as_float(w);
}
DEV bf16 upkb(const uint4& q, int j) {                            // 8 BF16 values in one chunk
  const uint32_t w = (j >> 1) == 0 ? q.x : (j >> 1) == 1 ? q.y : (j >> 1) == 2 ? q.z : q.w;
  const unsigned short b = (j & 1) ? (unsigned short)(w >> 16) : (unsigned short)(w & 0xffffu);
  return __ushort_as_bfloat16(b);
}
DEV void st8(float* p, const float (&v)[8]) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
DEV void st8(bf16* p, const bf16 (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_bfloat162 h; h.x = v[2 * k]; h.y = v[2 * k + 1];
    w[k] = *reinterpret_cast<const uint32_t*>(&h);
  }
  *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// P6E_QUAD: branch-free form of pops4b::quad_chunk (lane t of a quad receives word t of every quad lane: out[l] =
// x_l[t]) as two xor-shuffle 2x2 block transposes with bitwise lane-bit selects. pops4b::quad_chunk's runtime-indexed
// ternary chains compile to divergent branch regions (BSSY/BSYNC) per chunk; same words, same order here.
#ifndef P6E_QUAD
#define P6E_QUAD 0
#endif
DEV uint32_t lsel(uint32_t m, uint32_t a, uint32_t b) { return (a & m) | (b & ~m); }   // m all-ones -> a
DEV uint4 quad_chunk2(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, int t) {
  const uint32_t m1 = 0u - (uint32_t)((t >> 1) & 1), m0 = 0u - (uint32_t)(t & 1);
  // stage 1 (xor 2): lanes 0,1 keep words 0,1 and receive the partner's 0,1; lanes 2,3 keep 2,3 and receive 2,3
  const uint32_t ra = __shfl_xor_sync(0xffffffffu, lsel(m1, x0, x2), 2);
  const uint32_t rb = __shfl_xor_sync(0xffffffffu, lsel(m1, x1, x3), 2);
  const uint32_t y0 = lsel(m1, ra, x0), y1 = lsel(m1, rb, x1), y2 = lsel(m1, x2, ra), y3 = lsel(m1, x3, rb);
  // stage 2 (xor 1): 2x2 transposes inside lane pairs
  const uint32_t sa = __shfl_xor_sync(0xffffffffu, lsel(m0, y0, y1), 1);
  const uint32_t sb = __shfl_xor_sync(0xffffffffu, lsel(m0, y2, y3), 1);
  return make_uint4(lsel(m0, sa, y0), lsel(m0, y1, sa), lsel(m0, sb, y2), lsel(m0, y3, sb));
}
DEV uint4 qchunk(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, int t) {
  if constexpr (P6E_QUAD != 0) return quad_chunk2(x0, x1, x2, x3, t);
  else return pops4b::quad_chunk(x0, x1, x2, x3, t);
}

template <int MT, int WTN, bool AK, bool BKC>
DEV void stage_batch(float (&acc)[MT][WTN / 2], unsigned full_bar, unsigned parity, unsigned sA, unsigned sB,
                     int a_row0, int b_col0, bool first) {
  wait_parity(full_bar, parity);
  if (!first) {
#pragma unroll
    for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
  }
  pops4b::wg_fence();
#pragma unroll
  for (int k16 = 0; k16 < 4; ++k16) {
    const uint64_t db = BKC ? pops4b::dk(sB + b_col0 * 128 + k16 * 32) : pops4b::dmn(sB + (b_col0 / 64) * 8192 + k16 * 2048);
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int r0 = a_row0 + i * 64;
      const uint64_t da = AK ? pops4b::dk(sA + r0 * 128 + k16 * 32) : pops4b::dmn(sA + (r0 / 64) * 8192 + k16 * 2048);
      if (first && k16 == 0) FirstWg<WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db);
      else s101h::MmeWg<WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, 1);
    }
  }
  pops4b::wg_commit();
}

template <int BM, int BN, bool AK, bool BKC>
DEV void stage_issue(const pops4b::TArgs& g, bool leader, int m0, int n0, int kk, unsigned dA, unsigned bar) {
  const unsigned dB = dA + BM * 128;
  expect_if(leader, bar, (BM + BN) * 128);
  if (AK) tma2d_if(leader, dA, g.tA, kk, m0, bar);
  else {
#pragma unroll
    for (int c = 0; c < BM / 64; ++c) tma2d_if(leader, dA + c * 8192, g.tA, m0 + 64 * c, kk, bar);
  }
  if (BKC) tma2d_if(leader, dB, g.tB, kk, n0, bar);
  else {
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) tma2d_if(leader, dB + c * 8192, g.tB, n0 + 64 * c, kk, bar);
  }
}

// SPLIT > 1 (long-K products with few output tiles): task = (tile, split); split sp accumulates k-steps
// [sp nk/SPLIT, (sp+1) nk/SPLIT) from +0 and stores its FP32 partial tile (thread-major) to part; the CTA that
// arrives last at the tile's counter (cnt) sums the SPLIT partials in split order 0..SPLIT-1, restores the counter
// to zero and runs the epilogue (rounding, generated chains, stores). Deterministic fixed order; different
// summation grouping from SPLIT = 1 (tolerance-gated).
#ifndef P6E_EARLY_LD
#define P6E_EARLY_LD 0
#endif
__shared__ unsigned p6e_last;
// P6E_LAST_ISSUE: a consumed stage is refilled by the warpgroup that releases it second (shared counter per stage,
// one elected thread per warpgroup after its wgmma.wait), instead of a CTA barrier after every 64-k step; neither
// warpgroup waits for the other except through the full barriers. Same TMA loads, same wgmma order and operands.
#ifndef P6E_LAST_ISSUE
#define P6E_LAST_ISSUE 0
#endif
__shared__ unsigned p6e_cnt[8];
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool STORE = true, class Epi = NoEpi, class Ld = NoLd,
          int SPLIT = 1>
DEV void body(const pops4b::TArgs& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld(),
              unsigned* cnt = nullptr, float* part = nullptr) {
  using Cf = Cfg<BM, BN, WM>;
  static_assert(NST >= 2 && NST <= 8, "stages");
  static_assert(!AK || BM <= 256, "TMA box rows");
  static_assert(!BKC || BN <= 256, "TMA box rows");
  static_assert(Cf::WTN % 32 == 0, "quad-transposed row chunks");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn * SPLIT;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  const int nk = (g.K + 63) / 64 / SPLIT, Q = ntm * nk;          // k-steps per task (SPLIT divides them)
  if (Q == 0) return;                                              // CTA-uniform
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&pops4b::tma_bars[0]);
  const bool leader = tid == 0;
  __syncthreads();                                                 // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
  }
  if (P6E_LAST_ISSUE && tid < NST) p6e_cnt[tid] = 0u;
  __syncthreads();
  auto issue_by = [&](bool who, int qq) {
    const int task = s.me + (qq / nk) * s.count, tile = task / SPLIT, sp = task % SPLIT;
    stage_issue<BM, BN, AK, BKC>(g, who, (tile / tn) * BM, (tile % tn) * BN, (sp * nk + qq % nk) * 64,
                                 base + (qq % NST) * Cf::STAGE, bar0 + 8 * (qq % NST));
  };
  // P6E_QUAD also confines the CTA's TMA issue to warp 0 (uniform branch): the other warps skip the issue code
  auto issue = [&](int qq) {
    if (P6E_QUAD == 0 || tid < 32) issue_by(leader, qq);
  };
  // stage of step qq released by this warpgroup (its wgmma reads of it are complete)
  auto release = [&](int qq) {
    if ((tid & 127) == 0) {
      __threadfence_block();
      if (atomicAdd(&p6e_cnt[qq % NST], 1u) == 1u) {
        p6e_cnt[qq % NST] = 0u;
        if (qq + NST < Q) issue_by(true, qq + NST);
      }
    }
  };
  {
    const int pre = Q < NST ? Q : NST;
#pragma unroll 1
    for (int q = 0; q < pre; ++q) issue(q);
  }
  float acc[Cf::MT][Cf::R];
  int q = 0;
#pragma unroll 1
  for (int task = 0; task < ntm; ++task) {
    stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * Cf::STAGE,
                                         base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, true);
    ++q;
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * Cf::STAGE,
                                           base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();                                         // step q-1 complete in this warpgroup
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
      if constexpr (P6E_LAST_ISSUE != 0) {
        release(q - 1);
      } else {
        __syncthreads();                                            // ... and in the other one: stage reusable
        if (q - 1 + NST < Q) issue(q - 1 + NST);                   // CTA-uniform condition
      }
    }
    // P6E_EARLY_LD: generated epilogues' first chunk group of operand loads is issued while the tile's last wgmma
    // batch is still in flight (full tiles only; registers distinct from the accumulators). Same loads, same values.
    constexpr bool EARLY = P6E_EARLY_LD && !IsNoEpi<Epi>::value && SPLIT == 1;
    bool early_done = false;
    if constexpr (EARLY) {
      const int tk0 = s.me + task * s.count, m00 = (tk0 / tn) * BM, n00 = (tk0 % tn) * BN;
      if (m00 + BM <= g.M && n00 + BN <= g.N) {
        // mode 2 prefetches every chunk of the tile (slots 0..NCH-1; generated slot arrays hold 16 entries)
        constexpr int JJ0 = Cf::WTN / 32, NCH0 = Cf::MT * JJ0 * 2,
                      EG1 = NCH0 % 8 == 0 ? 8 : NCH0 % 6 == 0 ? 6 : NCH0 % 4 == 0 ? 4 : NCH0 % 2 == 0 ? 2 : 1,
                      EG0 = (P6E_EARLY_LD >= 2 && NCH0 <= 16) ? NCH0 : EG1;
#pragma unroll
        for (int e = 0; e < EG0; ++e) {
          const int i = e / (JJ0 * 2), jj = (e / 2) % JJ0, h = e % 2;
          const int row = m00 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
          const int col = n00 + wgn * Cf::WTN + 8 * (4 * jj + (lane & 3));
          ld(e, (unsigned)((long long)row * g.ldc + col));
        }
        early_done = true;
      }
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
    if constexpr (P6E_LAST_ISSUE != 0) {
      release(q - 1);
    } else {
      __syncthreads();
      if (q - 1 + NST < Q) issue(q - 1 + NST);                     // last stage of this tile is free
    }
    const int tk = s.me + task * s.count, tile = tk / SPLIT;
    if constexpr (SPLIT > 1) {
      float* slab = part + (size_t)tk * (BM * BN);                   // tk = tile * SPLIT + sp
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int r = 0; r < Cf::R; ++r) __stcg(slab + (i * Cf::R + r) * 256 + tid, acc[i][r]);
      __threadfence();
      __syncthreads();
      if (tid == 0) p6e_last = atomicAdd(cnt + tile, 1u) == (unsigned)(SPLIT - 1);
      __syncthreads();
      if (!p6e_last) continue;                                      // CTA-uniform; no wgmma in flight
      __threadfence();
      const float* tp = part + (size_t)tile * SPLIT * (BM * BN);
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int r = 0; r < Cf::R; ++r) {
          float t = __ldcg(tp + (i * Cf::R + r) * 256 + tid);
#pragma unroll
          for (int sp = 1; sp < SPLIT; ++sp) t += __ldcg(tp + (size_t)sp * (BM * BN) + (i * Cf::R + r) * 256 + tid);
          acc[i][r] = t;
        }
      if (tid == 0) cnt[tile] = 0u;                                 // every split arrived: reusable from zero
    }
    {
      const int m0 = (tile / tn) * BM, n0 = (tile % tn) * BN;
      const bool masked = m0 + BM > g.M || n0 + BN > g.N;
      const int t = lane & 3;
      // EG (chunks prefetched/computed per group) divides NCH (12 for 192-wide warpgroup tiles).
      constexpr int JJ = Cf::WTN / 32, NCH = Cf::MT * JJ * 2,
                    EG = NCH % 8 == 0 ? 8 : NCH % 6 == 0 ? 6 : NCH % 4 == 0 ? 4 : NCH % 2 == 0 ? 2 : 1;
      static_assert(NCH % EG == 0, "whole chunk groups");
      auto where = [&](int c, int& row, int& col) {
        const int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
        row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
        col = n0 + wgn * Cf::WTN + 8 * (4 * jj + t);
      };
#pragma unroll
      for (int c0 = 0; c0 < NCH; c0 += EG) {
        constexpr bool ALL = P6E_EARLY_LD >= 2 && NCH <= 16;
        if constexpr (!IsNoEpi<Epi>::value) {
          if (!((c0 == 0 || ALL) && early_done)) {
#pragma unroll
            for (int e = 0; e < EG; ++e) {
              int row, col;
              where(c0 + e, row, col);
              if (!masked || (row < g.M && col < g.N)) ld(ALL ? c0 + e : e, (unsigned)((long long)row * g.ldc + col));
            }
          }
        }
#pragma unroll
        for (int e = 0; e < EG; ++e) {
          const int c = c0 + e, i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
          const uint4 v = qchunk(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
          int row, col;
          where(c, row, col);
          if (masked && !(row < g.M && col < g.N)) continue;
          const long long off = (long long)row * g.ldc + col;
          if constexpr (!IsNoEpi<Epi>::value) {
            // Exact widening of the rounded BF16 results (8 contiguous outputs at flat index off; ldc == N).
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(ALL ? c : e, (unsigned)off, gv);
          }
          if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + off) = v;
        }
      }
    }
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_inval(bar0 + 8 * st);
    pops4b::proxy_fence();
  }
}

// P6E_DEFER = d > 0: deferred epilogue for unsplit tiles with at most 96 accumulators per thread. After a tile's main
// loop each thread packs its rounded results (the same s101h::bpk pairs body() transposes) into registers; that
// tile's epilogue then runs d chunks at a time right after each of the next tile's wgmma batches is committed, while
// the batch computes (the rest after the last tile). Every chunk is loaded, computed and stored exactly as in body():
// same values, same stores, only later.
#ifndef P6E_DEFER
#define P6E_DEFER 0
#endif
template <int N, class F, int... I> DEV void static_for_impl(F&& f, std::integer_sequence<int, I...>) {
  (f(std::integral_constant<int, I>{}), ...);
}
template <int N, class F> DEV void static_for(F&& f) { static_for_impl<N>(f, std::make_integer_sequence<int, N>{}); }
template <int BM, int BN, int WM>
inline constexpr bool defer_ok = P6E_DEFER > 0 && Cfg<BM, BN, WM>::MT * Cfg<BM, BN, WM>::R <= 96;

template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool STORE = true, class Epi = NoEpi, class Ld = NoLd>
DEV void body_defer(const pops4b::TArgs& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld()) {
  using Cf = Cfg<BM, BN, WM>;
  constexpr int MT = Cf::MT, R = Cf::R, JJ = Cf::WTN / 32, NCH = MT * JJ * 2, DCH = P6E_DEFER < 8 ? P6E_DEFER : 8,
                NG = (NCH + DCH - 1) / DCH;
  static_assert(NST >= 2 && NST <= 8 && Cf::WTN % 32 == 0, "stages / chunks");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  const int nk = (g.K + 63) / 64, Q = ntm * nk;
  if (Q == 0) return;                                              // CTA-uniform
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, t = lane & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&pops4b::tma_bars[0]);
  const bool leader = tid == 0;
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();
  }
  if (P6E_LAST_ISSUE && tid < NST) p6e_cnt[tid] = 0u;
  __syncthreads();
  auto issue_by = [&](bool who, int qq) {
    const int tile = s.me + (qq / nk) * s.count;
    stage_issue<BM, BN, AK, BKC>(g, who, (tile / tn) * BM, (tile % tn) * BN, (qq % nk) * 64,
                                 base + (qq % NST) * Cf::STAGE, bar0 + 8 * (qq % NST));
  };
  auto issue = [&](int qq) {
    if (P6E_QUAD == 0 || tid < 32) issue_by(leader, qq);
  };
  auto release = [&](int qq) {
    if ((tid & 127) == 0) {
      __threadfence_block();
      if (atomicAdd(&p6e_cnt[qq % NST], 1u) == 1u) {
        p6e_cnt[qq % NST] = 0u;
        if (qq + NST < Q) issue_by(true, qq + NST);
      }
    }
  };
  auto free_stage = [&](int qq) {                                  // step qq's stage is no longer read
    if constexpr (P6E_LAST_ISSUE != 0) {
      release(qq);
    } else {
      __syncthreads();
      if (qq + NST < Q) issue(qq + NST);
    }
  };
  {
    const int pre = Q < NST ? Q : NST;
#pragma unroll 1
    for (int q = 0; q < pre; ++q) issue(q);
  }
  float acc[MT][R];
  uint32_t pk[MT][R / 2];
  int pm0 = 0, pn0 = 0;
  bool pmask = false, pend = false;
  // chunk group gg of the pending tile: operand loads for its chunks, then results, epilogue and stores
  auto group = [&](auto gg) {
    constexpr int c0 = decltype(gg)::value * DCH;
    int rows[DCH], cols[DCH];
    static_for<DCH>([&](auto dd) {
      constexpr int c = c0 + decltype(dd)::value;
      if constexpr (c < NCH) {
        constexpr int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
        rows[decltype(dd)::value] = pm0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
        cols[decltype(dd)::value] = pn0 + wgn * Cf::WTN + 8 * (4 * jj + t);
        if constexpr (!IsNoEpi<Epi>::value) {
          const int row = rows[decltype(dd)::value], col = cols[decltype(dd)::value];
          if (!pmask || (row < g.M && col < g.N)) ld(decltype(dd)::value, (unsigned)((long long)row * g.ldc + col));
        }
      }
    });
    static_for<DCH>([&](auto dd) {
      constexpr int e = decltype(dd)::value, c = c0 + e;
      if constexpr (c < NCH) {
        constexpr int i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
        const int row = rows[e], col = cols[e];
        // every lane takes part in the transpose's shuffles before the masked check (as body())
        const uint4 v = qchunk(pk[i][2 * j0 + h], pk[i][2 * (j0 + 1) + h], pk[i][2 * (j0 + 2) + h],
                               pk[i][2 * (j0 + 3) + h], t);
        if (!pmask || (row < g.M && col < g.N)) {
          const long long off = (long long)row * g.ldc + col;
          if constexpr (!IsNoEpi<Epi>::value) {
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(e, (unsigned)off, gv);
          }
          if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + off) = v;
        }
      }
    });
  };
  auto run_group = [&](int k) {                                    // k is warp-uniform
    static_for<NG>([&](auto gg) { if (k == decltype(gg)::value) group(gg); });
  };
  int q = 0;
#pragma unroll 1
  for (int task = 0; task < ntm; ++task) {
    stage_batch<MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * Cf::STAGE,
                                     base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, true);
    ++q;
    if (pend) run_group(0);
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
      stage_batch<MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * Cf::STAGE,
                                       base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      if (pend && kk < NG) run_group(kk);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
      free_stage(q - 1);
    }
    if (pend) {
#pragma unroll 1
      for (int k = nk; k < NG; ++k) run_group(k);                  // short K: the rest of the pending tile
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
    free_stage(q - 1);
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int k = 0; k < R / 2; ++k) pk[i][k] = s101h::bpk(acc[i][2 * k], acc[i][2 * k + 1]);
    const int tk = s.me + task * s.count;
    pm0 = (tk / tn) * BM;
    pn0 = (tk % tn) * BN;
    pmask = pm0 + BM > g.M || pn0 + BN > g.N;
    pend = true;
  }
  if (pend) {
#pragma unroll 1
    for (int k = 0; k < NG; ++k) run_group(k);
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_inval(bar0 + 8 * st);
    pops4b::proxy_fence();
  }
}

// Entry-resident form only (wgmma inside called functions is serialized by ptxas, C7510).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST>
DEV void phase(const pops4b::TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  if constexpr (defer_ok<BM, BN, WM>) body_defer<BM, BN, WM, AK, BKC, NST>(g, smem, lo, hi, den, rot);
  else body<BM, BN, WM, AK, BKC, NST>(g, smem, lo, hi, den, rot);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool STORE, class Epi, class Ld, int SPLIT = 1>
DEV void phase_epi(const pops4b::TArgs g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld,
                   unsigned* cnt = nullptr, float* part = nullptr) {
  if constexpr (SPLIT == 1 && defer_ok<BM, BN, WM>)
    body_defer<BM, BN, WM, AK, BKC, NST, STORE, Epi, Ld>(g, smem, lo, hi, den, rot, epi, ld);
  else
    body<BM, BN, WM, AK, BKC, NST, STORE, Epi, Ld, SPLIT>(g, smem, lo, hi, den, rot, epi, ld, cnt, part);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int SPLIT>
DEV void phase_split(const pops4b::TArgs g, float* smem, int lo, int hi, int den, int rot, unsigned* cnt, float* part) {
  body<BM, BN, WM, AK, BKC, NST, true, NoEpi, NoLd, SPLIT>(g, smem, lo, hi, den, rot, NoEpi(), NoLd(), cnt, part);
}

// P6-E ping-pong body (plans with 'pp'): each warpgroup owns whole BM x BN tiles (MT = BM / 64 m64 slabs, N = BN per
// wgmma) and runs its own NST-stage TMA ring (private stages and full barriers, refilled by its own warp 0 after a
// warpgroup named barrier), so the two warpgroups meet only at the phase boundaries and one warpgroup's epilogue
// overlaps the other's wgmma main loop. Warpgroup w computes the CTA's tasks w, w + 2, ... of the cooperative task
// order. Per-tile arithmetic, rounding, epilogue chunks and stores are those of body(): bitwise equal results.
DEV void wg_bar(int id) { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }
DEV void wg_turn_wait(int id) { asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory"); }
DEV void wg_turn_arrive(int id) { asm volatile("bar.arrive %0, 256;" ::"r"(id) : "memory"); }
#ifndef P6E_PP_ORDER
#define P6E_PP_ORDER 1
#endif
template <int BM, int BN, bool AK, bool BKC, int NST, bool STORE = true, class Epi = NoEpi, class Ld = NoLd>
DEV void body_pp(const pops4b::TArgs& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld()) {
  constexpr int MT = BM / 64, WTN = BN, R = WTN / 2, A_BYTES = BM * 128, STAGE = (BM + BN) * 128;
  static_assert(MT >= 1 && MT <= 2 && MT * R <= 128 && WTN % 32 == 0, "warpgroup tile");
  static_assert(NST >= 2 && 2 * NST <= 8, "stages");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  const int nk = (g.K + 63) / 64;
  if (ntm == 0) return;                                            // CTA-uniform
  const int tid = threadIdx.x, wg = tid >> 7, wtid = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;
  const int mine = ntm > wg ? (ntm - wg + 1) / 2 : 0, Q = mine * nk;
  const unsigned base = ((sa(smem) + 1023u) & ~1023u) + (unsigned)(wg * NST * STAGE);
  const unsigned bar0 = sa(&pops4b::tma_bars[0]), wbar = bar0 + 8u * (unsigned)(NST * wg);
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < 2 * NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();
  }
  __syncthreads();
  auto task_of = [&](int it) { return s.me + (wg + 2 * it) * s.count; };
  auto issue = [&](int qq) {                                       // warp 0 of this warpgroup; lane 0 issues
    const int task = task_of(qq / nk);
    stage_issue<BM, BN, AK, BKC>(g, lane == 0, (task / tn) * BM, (task % tn) * BN, (qq % nk) * 64,
                                 base + (qq % NST) * STAGE, wbar + 8 * (qq % NST));
  };
  if (wtid < 32) {
    const int pre = Q < NST ? Q : NST;
#pragma unroll 1
    for (int q = 0; q < pre; ++q) issue(q);
  }
  float acc[MT][R];
  int q = 0;
  // Ordered main loops (P6E_PP_ORDER): the warpgroups take turns issuing their tiles' wgmma (named barriers 3/4,
  // arrive by the finishing warpgroup, sync by the next), so a tile's epilogue runs while the other warpgroup's main
  // loop has the tensor cores to itself. Arrivals are matched to the partner's remaining waits.
  const int other = ntm > 1 - wg ? (ntm - (1 - wg) + 1) / 2 : 0;
#pragma unroll 1
  for (int it = 0; it < mine; ++it) {
    if (P6E_PP_ORDER && (wg == 1 || it > 0)) wg_turn_wait(3 + wg);
    stage_batch<MT, WTN, AK, BKC>(acc, wbar + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * STAGE,
                                  base + (q % NST) * STAGE + A_BYTES, 0, 0, true);
    ++q;
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
      stage_batch<MT, WTN, AK, BKC>(acc, wbar + 8 * (q % NST), (unsigned)(q / NST) & 1u, base + (q % NST) * STAGE,
                                    base + (q % NST) * STAGE + A_BYTES, 0, 0, false);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
      wg_bar(1 + wg);                                              // the warpgroup's reads of step q-1 are done
      if (wtid < 32 && q - 1 + NST < Q) issue(q - 1 + NST);
    }
    const int tk = task_of(it), m0 = (tk / tn) * BM, n0 = (tk % tn) * BN;
    const bool masked = m0 + BM > g.M || n0 + BN > g.N;
    constexpr int JJ = WTN / 32, NCH = MT * JJ * 2,
                  EG = NCH % 8 == 0 ? 8 : NCH % 6 == 0 ? 6 : NCH % 4 == 0 ? 4 : NCH % 2 == 0 ? 2 : 1;
    static_assert(NCH % EG == 0, "whole chunk groups");
    const int t = lane & 3;
    auto where = [&](int c, int& row, int& col) {
      const int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
      row = m0 + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
      col = n0 + 8 * (4 * jj + t);
    };
    constexpr bool EARLY = P6E_EARLY_LD && !IsNoEpi<Epi>::value;
    bool early_done = false;
    if constexpr (EARLY) {
      if (!masked) {
#pragma unroll
        for (int e = 0; e < EG; ++e) {
          int row, col;
          where(e, row, col);
          ld(e, (unsigned)((long long)row * g.ldc + col));
        }
        early_done = true;
      }
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
    wg_bar(1 + wg);
    if (wtid < 32 && q - 1 + NST < Q) issue(q - 1 + NST);           // the tile's last stage is free
    // pass the turn: warpgroup 1 waits before each of its `other` (wg 0) / own tiles; warpgroup 0 waits before tiles 1..
    if (P6E_PP_ORDER && (wg == 0 ? it < other : it + 1 < other)) wg_turn_arrive(4 - wg);
#pragma unroll
    for (int c0 = 0; c0 < NCH; c0 += EG) {
      if constexpr (!IsNoEpi<Epi>::value) {
        if (!(c0 == 0 && early_done)) {
#pragma unroll
          for (int e = 0; e < EG; ++e) {
            int row, col;
            where(c0 + e, row, col);
            if (!masked || (row < g.M && col < g.N)) ld(e, (unsigned)((long long)row * g.ldc + col));
          }
        }
      }
#pragma unroll
      for (int e = 0; e < EG; ++e) {
        const int c = c0 + e, i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
        const uint4 v = qchunk(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                               s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                               s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                               s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
        int row, col;
        where(c, row, col);
        if (masked && !(row < g.M && col < g.N)) continue;
        const long long off = (long long)row * g.ldc + col;
        if constexpr (!IsNoEpi<Epi>::value) {
          const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                               __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                               __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                               __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
          epi(e, (unsigned)off, gv);
        }
        if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + off) = v;
      }
    }
  }
  __syncthreads();                                                 // both warpgroups' wgmma reads are complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < 2 * NST; ++st) pops4b::bar_inval(bar0 + 8 * st);
    pops4b::proxy_fence();
  }
}
template <int BM, int BN, bool AK, bool BKC, int NST, bool STORE, class Epi, class Ld>
DEV void phase_pp(const pops4b::TArgs g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  body_pp<BM, BN, AK, BKC, NST, STORE, Epi, Ld>(g, smem, lo, hi, den, rot, epi, ld);
}
template <int BM, int BN, bool AK, bool BKC, int NST>
DEV void phase_pp(const pops4b::TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  body_pp<BM, BN, AK, BKC, NST>(g, smem, lo, hi, den, rot);
}

}  // namespace p6e
