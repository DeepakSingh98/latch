// P-OPS5-C TMA + warpgroup-MMA BF16 body for aligned products (new code in this lane).
//
// Derived from P-OPS4-B's tma_body (kernels/pops4b_tma.cuh @ 92bb997a7, itself a port of the Latch 0.5 S107A
// matrix engine main loop); helpers (descriptors, wgmma wrappers, quad transpose, mbarrier/TMA wrappers) are reused
// from that header unchanged. Changes, each selectable for measurement:
//   * OUT = 1 stores the FP32 value float(bf16_rn(acc)) instead of the BF16 result: the exact value a separate
//     BF16 -> FP32 conversion of the stored product would produce (one rounding, exact widening), written as
//     32-byte row chunks. Used when the product's only reader is that conversion (rule in ops/p5c_matrix.py).
//   * REL = true releases a stage through a per-stage "empty" mbarrier (one arrival per warpgroup after its wgmma
//     reading the stage completed) instead of a CTA-wide barrier per 64-k step; thread 0 refills the stage after
//     that barrier completes. Stages are filled in the CTA's flattened task x k order by thread 0 only.
// Aligned extents only (M % BM == N % BN == K % 64 == 0): no masking.
// Arithmetic: BF16 x BF16 products accumulated in FP32 by wgmma in ascending k16 steps from +0 (scale-d = 0 on the
// first step of each tile), one cvt.rn.bf16x2 rounding. Identical per element to pops4b::tma_body.
#pragma once
#include <cuda_bf16.h>
#include "pops4b_tma.cuh"

namespace p5c {

using s101h::bf16;
using pops4b::Cfg;
using pops4b::sa;

__shared__ __align__(8) unsigned long long p5c_empty[8];
#ifdef P5C_DECOUPLED
// P6-D: per-stage consumption counters. Each warpgroup's elected thread adds 1 (acq_rel) after its wgmma reading the
// stage completed; the second arrival (odd previous value) issues the refill. No thread waits for the other group.
__shared__ unsigned p5c_cnt[8];
DEV bool second_arrival(bool elected, unsigned addr) {
  unsigned old;
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %1, 0;\nmov.u32 %0, 0;\n"
               "@p atom.acq_rel.cta.shared::cta.add.u32 %0, [%2], 1;\n}\n"
               : "=r"(old) : "r"((unsigned)elected), "r"(addr) : "memory");
  return elected && (old & 1u);
}
#endif

DEV void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }

// Branch-free producer primitives: every thread executes them; only the elected thread's instruction is predicated
// on. No C++ thread-dependent branch exists inside the main loop, so ptxas never needs a warpgroup wait (WG.DP)
// inside a divergent path (C7518/C7520 serialization of every wgmma).
DEV void expect_if(bool leader, unsigned bar, unsigned bytes) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n@p mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %2;\n}\n"
               ::"r"((unsigned)leader), "r"(bar), "r"(bytes) : "memory");
}
DEV void tma2d_if(bool leader, unsigned dst, const void* map, int c0, int c1, unsigned bar) {
#if defined(P9_TMA_EVICT_LAST) && P9_TMA_EVICT_LAST
  // P9-A: product operands are re-read by many CTAs; keep them in L2 ahead of streamed epilogue/optimizer traffic.
  asm volatile("{\n.reg .pred p;\n.reg .b64 pol;\nsetp.ne.u32 p, %0, 0;\n"
               "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
               "@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%1], [%2, {%3, %4}], [%5], pol;\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%1], [%2, {%3, %4}], [%5];\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
#endif
}
// CUTLASS ClusterBarrier::wait form: the retry loop lives inside one asm statement.
DEV void wait_parity(unsigned bar, unsigned parity) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\n"
               "bra LAB_WAIT;\nDONE:\n}\n" ::"r"(bar), "r"(parity) : "memory");
}

// GENERATED (P-OPS5-C scratch gen_first.py): first k16 step of a tile with write-only "=f" accumulators
// (scale-d = 0, the accumulator inputs are not read), so no wgmma accumulator is live across any call.
template <int TA, int TB> DEV void first_wg64(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, 0, 1, 1, %34, %35;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg128(float (&d)[64], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1, %66, %67;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int TA, int TB> DEV void first_wg256(float (&d)[128], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127}, %128, %129, 0, 1, 1, %130, %131;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]), "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]), "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]), "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]), "=f"(d[110]), "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]), "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]), "=f"(d[120]), "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]), "=f"(d[126]), "=f"(d[127])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <int N> struct FirstWg;
template <> struct FirstWg<64> { template <int TA, int TB> DEV static void run(float (&d)[32], uint64_t da, uint64_t db) { first_wg64<TA, TB>(d, da, db); } };
template <> struct FirstWg<128> { template <int TA, int TB> DEV static void run(float (&d)[64], uint64_t da, uint64_t db) { first_wg128<TA, TB>(d, da, db); } };
template <> struct FirstWg<256> { template <int TA, int TB> DEV static void run(float (&d)[128], uint64_t da, uint64_t db) { first_wg256<TA, TB>(d, da, db); } };
// P9-B: 96-wide first step (128 x 96 tiles with K-major B; shared with kernels/p9ws.cuh).
template <int TA, int TB> DEV void first_wg96(float (&d)[48], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, %48, %49, 0, 1, 1, %50, %51;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <> struct FirstWg<96> { template <int TA, int TB> DEV static void run(float (&d)[48], uint64_t da, uint64_t db) { first_wg96<TA, TB>(d, da, db); } };

// P9-B: 192-wide first step (128 x 192 tiles: whole waves for 3072-wide products).
template <int TA, int TB> DEV void first_wg192(float (&d)[96], uint64_t da, uint64_t db) {
  asm volatile("wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95}, %96, %97, 0, 1, 1, %98, %99;\n"
               : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]), "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]), "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]), "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]), "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]), "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]), "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]), "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]), "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]), "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]), "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]), "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]), "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]), "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]), "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95])
               : "l"(da), "l"(db), "n"(TA), "n"(TB));
}
template <> struct FirstWg<192> { template <int TA, int TB> DEV static void run(float (&d)[96], uint64_t da, uint64_t db) { first_wg192<TA, TB>(d, da, db); } };


// P9-B diagnostic (P5C_TRACE, timing only): CTA-local globaltimer stamps of p5c bodies -- 1 body start (after barrier
// init), 2 tile start, 3 tile accumulators complete (after the drain and CTA barrier), 4 tile epilogue done, 5 body end.
#ifdef P5C_TRACE
#ifndef P5C_TRACE_N
#define P5C_TRACE_N 4096
#endif
__device__ unsigned long long p5c_trace[128][P5C_TRACE_N];
__device__ int p5c_trace_n[128];
DEV void p5c_stamp(int code) {
  if (threadIdx.x == 0 && blockIdx.x < 128) {
    unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    const int i = p5c_trace_n[blockIdx.x]++;
    if (i < P5C_TRACE_N) p5c_trace[blockIdx.x][i] = (t << 8) | (unsigned)code;
  }
}
#else
DEV void p5c_stamp(int) {}
#endif

// Diagnostic only (P5C_DIAG_CALL): an out-of-line call placed into chosen phases to localize ptxas C7510.
__device__ __noinline__ void diag_call(float* p) { if (threadIdx.x == 999u) p[0] = 1.f; }
__device__ __noinline__ void diag_sync(float* p) { __syncthreads(); if (threadIdx.x == 999u) p[0] = 1.f; }
__device__ __noinline__ void diag_cpasync(float* p) {
  if (threadIdx.x == 999u) s101h::cpa16(p, p + 4);
  s101h::cpa_commit();
  s101h::cpa_wait<0>();
}
__device__ __noinline__ void diag_bmm(float* smem) {
  popsc::BArgs g{nullptr, 0, 1, 0, nullptr, 1, 0, 0, nullptr, 0, 0, 0, 0, 0, 0};
  popsc::bmm_phase<32, 32, 2, 4, true, true, 32, 2>(g, smem);
}

// Generated epilogues (ops/p5c_epilogue.py) see 8 contiguous outputs; 8-wide operand/result vectors.
struct NoEpi {
  DEV void operator()(int, unsigned, const float (&)[8]) const {}
};
struct NoLd {
  DEV void operator()(int, unsigned) const {}
};
template <class T> struct IsNoEpi { static constexpr bool value = false; };
template <> struct IsNoEpi<NoEpi> { static constexpr bool value = true; };
#ifndef P5C_STAGED_EPI
#define P5C_STAGED_EPI 0
#endif
#ifndef P5C_BULK_EPI
#define P5C_BULK_EPI 0   // LATCH-STEP-06: plain staged tiles leave through per-row bulk async copies (ops/p5c_matrix)
#endif
// LATCH-STEP-06: L2 prefetch of a tile's epilogue operand rows at tile start (ops/p5c_matrix.P5C_EPI_PREFETCH).
// `pf(r0)` is called once per tile row with the flat element offset of the row's first output (ldc == N); the
// generated functor issues one bulk L2 prefetch per operand row segment. Loads, values and order are unchanged.
struct NoPf {
  DEV void operator()(unsigned) const {}
};
template <class T> struct IsNoPf { static constexpr bool value = false; };
template <> struct IsNoPf<NoPf> { static constexpr bool value = true; };
DEV void pf_l2(const void* p, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
DEV void ld8(const float* p, float (&v)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
DEV void ld8(const bf16* p, bf16 (&v)[8]) {
  const uint4 t = *reinterpret_cast<const uint4*>(p);
  const uint32_t w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const __nv_bfloat162 h = *reinterpret_cast<const __nv_bfloat162*>(&w[k]);
    v[2 * k] = h.x; v[2 * k + 1] = h.y;
  }
}
// Raw 16-byte operand chunks prefetched by generated epilogues, unpacked at use.
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
DEV void st8cs(float* p, const float (&v)[8]) {                 // P8-A: evict-first (streaming) stores
  __stcs(reinterpret_cast<float4*>(p), make_float4(v[0], v[1], v[2], v[3]));
  __stcs(reinterpret_cast<float4*>(p) + 1, make_float4(v[4], v[5], v[6], v[7]));
}
DEV void st8cs(bf16* p, const bf16 (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_bfloat162 h; h.x = v[2 * k]; h.y = v[2 * k + 1];
    w[k] = *reinterpret_cast<const uint32_t*>(&h);
  }
  __stcs(reinterpret_cast<uint4*>(p), make_uint4(w[0], w[1], w[2], w[3]));
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

// Branch-free quad transpose (P6-E p6e_quad, commit 155bc03e5): lane t of a quad receives word t of every quad
// lane, as two xor-shuffle 2x2 block transposes with bitwise lane-bit selects. pops4b::quad_chunk's runtime-indexed
// ternary chains compile to divergent branch regions per chunk; same words, same order here (data movement only).
DEV uint32_t lsel(uint32_t m, uint32_t a, uint32_t b) { return (a & m) | (b & ~m); }
DEV uint4 quad_chunk2(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, int t) {
  const uint32_t m1 = 0u - (uint32_t)((t >> 1) & 1), m0 = 0u - (uint32_t)(t & 1);
  const uint32_t ra = __shfl_xor_sync(0xffffffffu, lsel(m1, x0, x2), 2);
  const uint32_t rb = __shfl_xor_sync(0xffffffffu, lsel(m1, x1, x3), 2);
  const uint32_t y0 = lsel(m1, ra, x0), y1 = lsel(m1, rb, x1), y2 = lsel(m1, x2, ra), y3 = lsel(m1, x3, rb);
  const uint32_t sa = __shfl_xor_sync(0xffffffffu, lsel(m0, y0, y1), 1);
  const uint32_t sb = __shfl_xor_sync(0xffffffffu, lsel(m0, y2, y3), 1);
  return make_uint4(lsel(m0, sa, y0), lsel(m0, y1, sa), lsel(m0, sb, y2), lsel(m0, y3, sb));
}

struct Args {
  const void* tA;
  const void* tB;
  const bf16* A;          // operand views the maps describe (not dereferenced; keep storage and ordering explicit)
  const bf16* B;
  void* C;                // BF16 [M, ldc] (OUT 0) or FP32 [M, ldc] (OUT 1)
  long long ldc;
  int M, N, K;
  float* part;            // P7-C split-K (SK > 1): FP32 slice partials [SK][tiles][256 threads][fragment]
  unsigned* cnt;          // P7-C split-K: per-tile arrival counters (zero on entry; the last arriver restores zero)
  // P9-B stage schedule (optional): the products of one stage share a compile-time longest-first tile table over the
  // planned grid: sch = [sgrid][scount + 1] per-CTA prefix offsets, then the per-CTA tile lists; this product is
  // member sidx. Used only on the planned grid with the whole-grid slice (otherwise the rotation share).
  const unsigned short* sch;
  int sidx, scount, sgrid;
  // P9-B prefetched schedule entry of this CTA (bit 63 set when valid): count (bits 48-62), first tile (32-47) and
  // the offset of the CTA's tile list in stl (bits 0-31). Loaded at stage entry, so the first tile issues at once.
  unsigned long long srow;
  const unsigned short* stl;
  // P9-B (loop_pipeline P9_ACC_FIRST): this FP32-accumulating product is the only writer of a zero-initialized
  // parameter-gradient buffer in the reverse body; in reverse step 0 (p5c_first) it stores instead of adding to +0.
  int acc_first;
  // P9-B (lower.P9_DELTA_EPI): this product writes an attention backward's dO; its staged epilogue also writes the
  // attention delta table entries of its tile (v6 delta_table arithmetic on the staged BF16 values).
  const float* dlt_o32;
  float* dlt_out;
  int dlt_t, dlt_heads;
  // P9-B (lower.P9_QNORM_EPI): the same epilogue writes the attention's q-norm block maxima of its rows
  // (block_norm_table arithmetic per row; the block maximum is order-independent).
  const bf16* qn_q;
  float* qn_out;
  long long qn_sb, qn_sh, qn_sr;
  // Step 10 (ops/conv_ops.py): tap-shifted B operand of an overlapping 3x3 convolution, read in place from three
  // column-shifted padded copies (3C rows, M + Lp + M columns) through a {64, 64} map. vb_c = C (0: ordinary
  // operand), vb_w = padded row width Wp, vb_m = margin M, vb_s = +1 (patches) or -1 (flipped taps). Rows (tap, c)
  // of the logical operand are tap-major. Every box starts at a non-negative multiple of 8 columns.
  int vb_c, vb_w, vb_m, vb_s;
  // LATCH-STEP10-SPEED (ops/conv_ops.py, pixel-major form): one operand is the (pixels, 9C) tap matrix of a dense
  // pixel-major (N*H*W, C) BF16 image, read in place through a TMA im2col map (implicit GEMM). vi_c = C (0: none),
  // vi_h/vi_w = image height/width, vi_s = +1 (taps) or -1 (flipped taps), vi_b = 0: the A operand (K-major, rows
  // are pixels, K = (tap, c)); 1: the B operand (N-major, columns are (tap, c), K = pixels). Each box is one
  // 64-channel slab of one tap over consecutive pixels; zero fill outside the image supplies the padding.
  int vi_c, vi_h, vi_w, vi_s, vi_b;
};
__shared__ int p5c_sk_last;
__shared__ unsigned p5c_qn_sh[4];
__shared__ int p5c_first;       // set by the program before every reverse-body stage call (step == 0)
__shared__ int p5c_skipzero;    // set by the program: count > 0 (step-0 stores replace the zero fill)

// One 64-k stage batch: MT accumulator slabs x 4 k16 steps of m64nWTNk16 per warpgroup.
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
// TMA loads of one BM x 64 A slab and one 64 x BN B slab into stage dA (B follows at dA + BM * 128).
// Step 10: stored column of a tap's slab (whole padded rows only) and its row block (the column-shifted copy).
DEV int vb_column(const Args& g, int tap) { return g.vb_m + g.vb_s * (tap / 3 - 1) * g.vb_w; }
DEV int vb_block(const Args& g, int tap) { return g.vb_s > 0 ? tap % 3 : 2 - tap % 3; }
// Step 10: B stage of a tap-shifted operand. Each 64-row slab (one tap, 64 channels) is one {64, 64} box of the
// stored copies. K-major B (BKC): rows are output (tap, c) and K is padded pixels.
template <int BN, bool BKC>
DEV void vb_issue(const Args& g, bool leader, int n0, int kk, unsigned dB, unsigned bar) {
  if (BKC) {
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) {
      const int row = n0 + 64 * c, tap = row / g.vb_c;
      tma2d_if(leader, dB + c * 8192, g.tB, kk + vb_column(g, tap), vb_block(g, tap) * g.vb_c + row - tap * g.vb_c, bar);
    }
  } else {
    const int tap = kk / g.vb_c, column = vb_column(g, tap), row = vb_block(g, tap) * g.vb_c + kk - tap * g.vb_c;
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) tma2d_if(leader, dB + c * 8192, g.tB, n0 + 64 * c + column, row, bar);
  }
}

// LATCH-STEP10-SPEED: im2col operand loads (see Args::vi_*). The box of pixel p starts at (w - 1, h - 1) of its image
// and the tap selects the im2col offsets (kw, kh) (flipped: (2 - kw, 2 - kh)); the map's pixel walk covers whole rows
// and images in raster order, with zeros outside the image.
DEV void im2col_if(bool leader, unsigned dst, const void* map, int c0, int w0, int h0, int n0, unsigned bar, int ow,
                   int oh) {
#if defined(P9_TMA_EVICT_LAST) && P9_TMA_EVICT_LAST
  asm volatile("{\n.reg .pred p;\n.reg .b64 pol;\nsetp.ne.u32 p, %0, 0;\n"
               "createpolicy.fractional.L2::evict_last.b64 pol, 1.0;\n"
               "@p cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.L2::cache_hint [%1], [%2, {%3, %4, %5, %6}], [%7], {%8, %9}, pol;\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(w0), "r"(h0), "r"(n0), "r"(bar),
                 "h"((unsigned short)ow), "h"((unsigned short)oh) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%1], [%2, {%3, %4, %5, %6}], [%7], {%8, %9};\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(w0), "r"(h0), "r"(n0), "r"(bar),
                 "h"((unsigned short)ow), "h"((unsigned short)oh) : "memory");
#endif
}
DEV void vi_load(const Args& g, bool leader, unsigned dst, const void* map, int pixel, int col, unsigned bar) {
  const int hw = g.vi_h * g.vi_w, n = pixel / hw, r = pixel - n * hw, y = r / g.vi_w, x = r - y * g.vi_w;
  const int tap = col / g.vi_c, kh = tap / 3, kw = tap - 3 * kh;
  const int c0 = tap < 9 ? col - tap * g.vi_c : g.vi_c;            // padded columns: a box beyond the channels (zeros)
  im2col_if(leader, dst, map, c0, x - 1, y - 1, n, bar, tap < 9 ? (g.vi_s > 0 ? kw : 2 - kw) : 0,
            tap < 9 ? (g.vi_s > 0 ? kh : 2 - kh) : 0);
}

template <int BM, int BN, bool AK, bool BKC>
DEV void stage_issue(const Args& g, bool leader, int m0, int n0, int kk, unsigned dA, unsigned bar) {
  const unsigned dB = dA + BM * 128;
  expect_if(leader, bar, (BM + BN) * 128);
  if (AK && g.vi_c && !g.vi_b) vi_load(g, leader, dA, g.tA, m0, kk, bar);    // uniform per product
  else if (AK) tma2d_if(leader, dA, g.tA, kk, m0, bar);
  else {
#pragma unroll
    for (int c = 0; c < BM / 64; ++c) tma2d_if(leader, dA + c * 8192, g.tA, m0 + 64 * c, kk, bar);
  }
  if (g.vb_c) vb_issue<BN, BKC>(g, leader, n0, kk, dB, bar);   // uniform per product (step 10)
  else if (!BKC && g.vi_c && g.vi_b) {
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) vi_load(g, leader, dB + c * 8192, g.tB, kk, n0 + 64 * c, bar);
  }
  else if (BKC) tma2d_if(leader, dB, g.tB, kk, n0, bar);
  else {
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) tma2d_if(leader, dB + c * 8192, g.tB, n0 + 64 * c, kk, bar);
  }
}

// ---- 2-CTA cluster multicast (MC = 2): a CTA pair computes tiles that share A rows; each CTA TMA-loads half of the
// A slab with .multicast::cluster into both CTAs' identical stage offsets and its own B slab. A stage is refilled only
// after both CTAs released it (per-stage "empty" mbarrier: one local and one remote arrival).
DEV unsigned cluster_rank() { unsigned r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
DEV unsigned peer_addr(unsigned local, unsigned rank) {
  unsigned r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(local), "r"(rank)); return r;
}
// P9-B: the default (CTA-scope release) arrive forms, as CUTLASS's ClusterBarrier::arrive(cta_id): the consumer's
// stage reads are wgmma (async proxy) reads completed by its wgmma wait, and the peer's refill is a TMA write ordered
// by the barrier itself. P5C_MC_CLUSTER_RELEASE=1 restores the explicit cluster-scope release of P8-B/P9-B probes.
#ifndef P5C_MC_CLUSTER_RELEASE
#define P5C_MC_CLUSTER_RELEASE 0
#endif
DEV void release_pair_if(bool leader, unsigned local_bar, unsigned remote_bar) {
#if P5C_MC_CLUSTER_RELEASE
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p mbarrier.arrive.release.cluster.shared::cta.b64 _, [%1];\n"
               "@p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%2];\n}\n"
               ::"r"((unsigned)leader), "r"(local_bar), "r"(remote_bar) : "memory");
#else
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p mbarrier.arrive.shared::cta.b64 _, [%1];\n"
               "@p mbarrier.arrive.shared::cluster.b64 _, [%2];\n}\n"
               ::"r"((unsigned)leader), "r"(local_bar), "r"(remote_bar) : "memory");
#endif
}
DEV void tma2d_mc_if(bool leader, unsigned dst, const void* map, int c0, int c1, unsigned bar, unsigned short mask) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %0, 0;\n"
               "@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
               " [%1], [%2, {%3, %4}], [%5], %6;\n}\n"
               ::"r"((unsigned)leader), "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar), "h"(mask) : "memory");
}
template <int BM, int BN, bool AK, bool BKC>
DEV void stage_issue_mc(const Args& g, bool leader, unsigned rank, int m0, int n0, int kk, unsigned dA, unsigned bar) {
  const unsigned dB = dA + BM * 128;
  expect_if(leader, bar, (BM + BN) * 128);                          // both A halves + own B land on this barrier
  if (AK) {
    tma2d_mc_if(leader, dA + rank * (BM / 2) * 128, g.tA, kk, m0 + (int)rank * (BM / 2), bar, (unsigned short)3);
  } else {
#pragma unroll
    for (int c = 0; c < BM / 128; ++c) {
      const int chunk = (int)rank * (BM / 128) + c;
      tma2d_mc_if(leader, dA + chunk * 8192, g.tA, m0 + 64 * chunk, kk, bar, (unsigned short)3);
    }
  }
  if (g.vb_c) vb_issue<BN, BKC>(g, leader, n0, kk, dB, bar);   // uniform per product (step 10)
  else if (!BKC && g.vi_c && g.vi_b) {                          // im2col B (A operands are never multicast)
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) vi_load(g, leader, dB + c * 8192, g.tB, kk, n0 + 64 * c, bar);
  }
  else if (BKC) tma2d_if(leader, dB, g.tB, kk, n0, bar);
  else {
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) tma2d_if(leader, dB + c * 8192, g.tB, n0 + 64 * c, kk, bar);
  }
}



template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, bool STORE = true, class Epi = NoEpi,
          class Ld = NoLd, int MC = 1, int SK = 1, int PK = 0, class Pf = NoPf>
DEV void body(const Args& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld(),
              Pf pf = Pf()) {
  using Cf = Cfg<BM, BN, WM>;
  // PK > 0 (release step 3 repair): chunked accumulation, see the K loop below.
  static_assert(PK == 0 || (Cf::MT == 1 && MC == 1 && !REL), "chunked accumulation: one accumulator slab per warpgroup");

  // LATCH-STEP-06 (P5C_BULK_EPI): a plain staged 128 x 128 tile (no generated epilogue, BF16 store) is written by
  // 128 per-row bulk async copies from an unswizzled staging buffer (272-byte rows) while the next tile's main
  // loop runs; the buffer is reused only after those copies have read it. Same bytes, same addresses.
  constexpr bool BULK_ = P5C_STAGED_EPI && P5C_BULK_EPI && BM == 128 && BN == 128 && MC == 1 && SK == 1 &&
                         IsNoEpi<Epi>::value && OUT == 0 && STORE;
  bool bulk_pending = false;
  static_assert(NST >= 2 && NST <= 8, "stages");
  static_assert(!AK || BM <= 256, "TMA box rows");
  static_assert(!BKC || BN <= 256, "TMA box rows");
  static_assert(Cf::WTN % 32 == 0, "quad-transposed row chunks");
  static_assert(MC == 1 || (MC == 2 && BM % 128 == 0), "multicast splits A into whole halves");
  static_assert(OUT != 3 || (MC == 1 && SK == 1), "raw FP32 output: unstaged single-CTA tiles, no split-K");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = tm * tn;
  // MC = 2: whole grid only (codegen), even grid (cluster launch); cluster c owns N-tile pairs (2j, 2j + 1) of one M row.
  const unsigned rank = MC == 2 ? cluster_rank() : 0u;
  const int unit = MC == 2 ? (int)blockIdx.x >> 1 : s.me, units = MC == 2 ? (int)gridDim.x >> 1 : s.count;
  // SK > 1 (P7-C split-K): work units are (tile, K slice), the slices of one tile on neighbouring units.
  const int nunit = MC == 2 ? tm * (tn / 2) : ntask * SK;
  int ntm = unit < nunit ? (nunit - unit + units - 1) / units : 0;
  const unsigned short* tl = nullptr;
  int first = 0;
  if constexpr (MC == 1 && SK == 1) {
    if ((g.srow >> 63) && lo == 0 && hi == 1 && den == 1) {
      tl = g.stl + (unsigned)(g.srow & 0xffffffffull);
      ntm = (int)((g.srow >> 48) & 0x7fffull);
      first = (int)((g.srow >> 32) & 0xffffull);
    } else if (g.sch != nullptr && lo == 0 && hi == 1 && den == 1 && (int)gridDim.x == g.sgrid) {
      const unsigned short* row = g.sch + blockIdx.x * (g.scount + 1) + g.sidx;
      tl = g.sch + g.sgrid * (g.scount + 1) + row[0];
      ntm = row[1] - row[0];
      first = ntm > 0 ? (int)tl[0] : 0;
    }
  }
  const int nk = g.K / 64 / SK, Q = ntm * nk;
  if (Q == 0) return;                                              // CTA-uniform (and cluster-uniform for MC = 2)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&pops4b::tma_bars[0]), ebar0 = sa(&p5c_empty[0]);
  auto where = [&](int i, int& m0, int& n0) {
    const int u = tl ? (i == 0 ? first : (int)tl[i]) : unit + i * units;
    if (MC == 2) {
      m0 = (u / (tn / 2)) * BM;
      n0 = (2 * (u % (tn / 2)) + (int)rank) * BN;
    } else {
      m0 = ((u / SK) / tn) * BM;
      n0 = ((u / SK) % tn) * BN;
    }
  };
  const bool leader = tid == 0;
  __syncthreads();                                                 // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_init(bar0 + 8 * st, 1);
      if (REL || MC == 2) pops4b::bar_init(ebar0 + 8 * st, 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
#ifdef P5C_DECOUPLED
#pragma unroll
    for (int st = 0; st < 8; ++st) p5c_cnt[st] = 0u;
#endif
  }
  __syncthreads();
  p5c_stamp(1);
  if (MC == 2) cluster_sync();                                     // peer barriers exist before any multicast lands
#define P5C_ISSUE(QQ) P5C_ISSUE_BY(QQ, leader)
#define P5C_ISSUE_BY(QQ, LEAD)                                                                                       \
  { const bool lead_ = (LEAD); const int qq_ = (QQ), i_ = qq_ / nk, u_ = tl ? (i_ == 0 ? first : (int)tl[i_]) : unit + i_ * units; int m_, n_;                                      \
    if (MC == 2) { m_ = (u_ / (tn / 2)) * BM; n_ = (2 * (u_ % (tn / 2)) + (int)rank) * BN;                        \
      stage_issue_mc<BM, BN, AK, BKC>(g, lead_, rank, m_, n_, (qq_ % nk) * 64, base + (qq_ % NST) * Cf::STAGE,     \
                                      bar0 + 8 * (qq_ % NST)); }                                                   \
    else { m_ = ((u_ / SK) / tn) * BM; n_ = ((u_ / SK) % tn) * BN;                                                  \
      stage_issue<BM, BN, AK, BKC>(g, lead_, m_, n_, ((qq_ % nk) + (u_ % SK) * nk) * 64,                         \
                                   base + (qq_ % NST) * Cf::STAGE, bar0 + 8 * (qq_ % NST)); } }
  // Release of stage (QQ) % NST by this CTA; for MC = 2 also to the peer, then wait until both released it.
#define P5C_RELEASE_AND_REFILL(QQ)                                                                                   \
  { const int rq_ = (QQ);                                                                                          \
    if (MC == 2) {                                                                                                 \
      const unsigned eb_ = ebar0 + 8 * (rq_ % NST);                                                                 \
      release_pair_if(leader, eb_, peer_addr(eb_, rank ^ 1u));                                                      \
      if (rq_ + NST < Q) { wait_parity(eb_, (unsigned)(rq_ / NST) & 1u); P5C_ISSUE(rq_ + NST) }                    \
    } else if (rq_ + NST < Q) P5C_ISSUE(rq_ + NST) }
  {
    const int pre = Q < NST ? Q : NST;
#ifdef P5C_PAR_PRE
    // P9-B: lanes 0..pre-1 of warp 0 issue one prologue stage each in a single predicated pass (same loads).
    { const int qp_ = tid < pre ? tid : 0; P5C_ISSUE_BY(qp_, tid < pre) }
#else
#pragma unroll 1
    for (int q = 0; q < pre; ++q) P5C_ISSUE(q)
#endif
  }
  float acc[Cf::MT][Cf::R];
  // Batch of one 64-k stage: 4 k16 steps x MT accumulator slabs per warpgroup.
  // CUTLASS-style invariant: the inner k loop is entered and re-entered with exactly one committed batch in flight;
  // every control-flow merge sees the same asynchronous accumulator state (no compiler-inserted wait in a divergent
  // path), and the accumulators complete (wait 0) before the epilogue and before the next task.
  int q = 0;
#pragma unroll 1
  for (int task = 0; task < ntm; ++task) {
    p5c_stamp(2);
#ifdef P5C_EPI_PF
    // LATCH-STEP-06: the epilogue's saved operands (reverse-tape tensors, usually in DRAM) are requested into L2 while
    // the tile's main loop runs: generated operand rows (pf) and the attention delta / q-norm rows of the dO product.
    if constexpr (!IsNoPf<Pf>::value) {
      int pm0, pn0;
      where(task, pm0, pn0);
      for (int r = tid; r < BM; r += 256) pf((unsigned)((long long)(pm0 + r) * g.ldc + pn0));
    }
    if constexpr (BM == 128 && BN == 128 && MC == 1 && SK == 1) {
      if (g.dlt_out != nullptr) {
        int pm0, pn0;
        where(task, pm0, pn0);
        const int r = pm0 + (tid >> 1), b = r / g.dlt_t, i = r - b * g.dlt_t;
        const int bh = b * g.dlt_heads + (pn0 >> 6) + (tid & 1);
        pf_l2(g.dlt_o32 + ((long long)bh * g.dlt_t + i) * 64, 256u);             // delta: 64 FP32 of O per (row, head)
        if (g.qn_out != nullptr)                                                   // q-norm: 64 BF16 of Q per (row, head)
          pf_l2(g.qn_q + (long long)(bh / g.dlt_heads) * g.qn_sb + (long long)(bh % g.dlt_heads) * g.qn_sh
                + (long long)i * g.qn_sr, 128u);
      }
    }
#endif
#ifdef P5C_ACC_PREFETCH
    // P9-B: L2 prefetch of this tile's FP32 accumulate destination (read by the epilogue after the mainloop).
    if constexpr (OUT == 2 && STORE && SK == 1) {
      if (!(g.acc_first && p5c_first)) {
        int pm0, pn0;
        where(task, pm0, pn0);
        // 128-byte lines of the tile: BM rows x (BN * 4 / 128) lines, spread over the CTA's threads.
        constexpr int LPR = BN * 4 / 128, NL = BM * LPR;
#pragma unroll
        for (int l = tid; l < NL; l += 256) {
          const float* a_ = reinterpret_cast<const float*>(g.C) + (long long)(pm0 + l / LPR) * g.ldc + pn0 + (l % LPR) * 32;
          asm volatile("prefetch.global.L2 [%0];" :: "l"(a_));
        }
      }
    }
#endif
    if constexpr (PK == 0) {
    stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, true);                                                 // peeled first stage of this tile
    ++q;
#ifdef P5C_SYNC2
    // P6-D: one CTA barrier and stage refill per two 64-k steps (NST >= 4 keeps a refill lead of >= 2 steps).
    int kk = 1;
#pragma unroll 1
    for (; kk + 1 < nk; kk += 2, q += 2) {
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q + 1) % NST), (unsigned)((q + 1) / NST) & 1u, base + ((q + 1) % NST) * Cf::STAGE, base + ((q + 1) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();                                         // steps q-1 and q complete in this warpgroup
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
      __syncthreads();                                              // ... and in the other one: both stages reusable
      if (q - 1 + NST < Q) P5C_ISSUE(q - 1 + NST)                   // P9-B: table-aware tile mapping
      if (q + NST < Q) P5C_ISSUE(q + NST)
    }
#pragma unroll 1
    for (; kk < nk; ++kk, ++q) {
#else
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
#endif
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();                                         // step q-1 complete in this warpgroup
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
#ifdef P5C_DECOUPLED
      if (q - 1 + NST < Q) { const bool issue_ = second_arrival((tid & 127) == 0, sa(&p5c_cnt[(q - 1) % NST])); P5C_ISSUE_BY(q - 1 + NST, issue_) }
#else
      __syncthreads();                                              // ... and in the other one: stage reusable
      P5C_RELEASE_AND_REFILL(q - 1)                                 // CTA-uniform conditions
#endif
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
#ifdef P5C_DECOUPLED
    if (q - 1 + NST < Q) { const bool issue_ = second_arrival((tid & 127) == 0, sa(&p5c_cnt[(q - 1) % NST])); P5C_ISSUE_BY(q - 1 + NST, issue_) }
#else
    __syncthreads();
    P5C_RELEASE_AND_REFILL(q - 1)                                   // last stage of this tile is free
#endif
    } else {
      // Release step 3 repair, chunked accumulation (same arithmetic as p8ws::mma_item with PK): the K loop is a
      // sequence of wgmma chains of at most PK 64-k stages, each from scale-d = 0; their FP32 results are summed in
      // chunk order with round-to-nearest FP32 additions, ((c0 + c1) + c2) + ..., in a second register slab.
      float sum[Cf::MT][Cf::R];
#pragma unroll 1
      for (int c0 = 0; c0 < nk; c0 += PK) {
        const int ce = c0 + PK < nk ? c0 + PK : nk;
        stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, true);
        ++q;
#pragma unroll 1
        for (int kk = c0 + 1; kk < ce; ++kk, ++q) {
          stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
          pops4b::wg_wait<1>();
#pragma unroll
          for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
          __syncthreads();
          P5C_RELEASE_AND_REFILL(q - 1)
        }
        pops4b::wg_wait<0>();
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
        __syncthreads();
        P5C_RELEASE_AND_REFILL(q - 1)
        const bool head = c0 == 0;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int r = 0; r < Cf::R; ++r) sum[i][r] = head ? acc[i][r] : __fadd_rn(sum[i][r], acc[i][r]);
      }
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int r = 0; r < Cf::R; ++r) acc[i][r] = sum[i][r];
    }
    p5c_stamp(3);
    if constexpr (SK > 1) {
      // Deterministic split-K: every slice stores its FP32 fragment; the last slice to arrive sums the SK partials
      // in slice order 0..SK-1 (FP32 association of the K sum changes; one BF16 rounding of the full sum follows).
      const int u = unit + task * units, tile = u / SK, slice = u % SK;
      constexpr int FR = Cf::MT * Cf::R;
      float* mine = g.part + ((long long)(slice * ntask + tile) * 256 + tid) * FR;
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int r = 0; r < Cf::R; r += 4)
          *reinterpret_cast<float4*>(mine + i * Cf::R + r) = make_float4(acc[i][r], acc[i][r + 1], acc[i][r + 2], acc[i][r + 3]);
      // LATCH-STEP10-SPEED: without counters (cnt == nullptr) the slice partials are summed in slice order by a later
      // phase over the whole grid (lower.py conv split-K reduce) instead of by each tile's last arriver.
      if (g.cnt == nullptr) continue;                               // uniform per product
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const unsigned prev = atomicAdd(g.cnt + tile, 1u);
        p5c_sk_last = prev == (unsigned)(SK - 1);
        if (p5c_sk_last) g.cnt[tile] = 0u;                         // reusable by the next instance of this product
      }
      __syncthreads();
      if (!p5c_sk_last) continue;                                   // CTA-uniform
      __threadfence();
#pragma unroll
      for (int sl = 0; sl < SK; ++sl) {
        const float* src = g.part + ((long long)(sl * ntask + tile) * 256 + tid) * FR;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int r = 0; r < Cf::R; r += 4) {
            const float4 v4 = __ldcg(reinterpret_cast<const float4*>(src + i * Cf::R + r));
            if (sl == 0) { acc[i][r] = v4.x; acc[i][r + 1] = v4.y; acc[i][r + 2] = v4.z; acc[i][r + 3] = v4.w; }
            else { acc[i][r] += v4.x; acc[i][r + 1] += v4.y; acc[i][r + 2] += v4.z; acc[i][r + 3] += v4.w; }
          }
      }
    }
    // P9-B (P5C_STAGED_EPI): 128 x 128 tiles stage their BF16-rounded results in a swizzled 32 KB buffer after the
    // stage ring and run the epilogue from a rolled loop over row-contiguous 16-byte chunks (half the epilogue code,
    // coalesced stores). The values each chunk sees are the same BF16 roundings; outputs are bitwise unchanged.
    // LATCH-DENSE-EXEC (OUT == 3): unrounded FP32 accumulators are never staged as BF16.
    constexpr bool STG = P5C_STAGED_EPI && BM == 128 && BN == 128 && MC == 1 && SK == 1 && OUT != 3;
    if constexpr (STG) {
      int m0, n0;
      where(task, m0, n0);
      const int t = lane & 3;
      const unsigned stg = base + NST * Cf::STAGE;
      auto swz = [](int row, int col) -> unsigned {                // byte offset of bf16 (row, col), 256 B rows
        if constexpr (BULK_) return (unsigned)(row * 272 + (col << 1));   // bulk: 272 B rows (conflict-free stores)
        return (unsigned)(row * 256 + ((((col >> 3) ^ (row & 15))) << 4) + ((col & 7) << 1));
      };
      if constexpr (BULK_) {
        if (bulk_pending) {                                        // previous tile's copies have read the buffer
          if (tid < BM) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          __syncthreads();
        }
      }
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
          const int r0 = wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2), col = wgn * Cf::WTN + nb * 8 + 2 * t;
          const uint32_t u0 = s101h::bpk(acc[i][4 * nb], acc[i][4 * nb + 1]);
          const uint32_t u1 = s101h::bpk(acc[i][4 * nb + 2], acc[i][4 * nb + 3]);
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(stg + swz(r0, col)), "r"(u0) : "memory");
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(stg + swz(r0 + 8, col)), "r"(u1) : "memory");
        }
      if constexpr (BULK_) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic -> async
      __syncthreads();
      constexpr int CPR = BN / 8, PER = BM * CPR / 256, G = 4;
      if constexpr (BULK_) {
        if (tid < BM) {
          const bf16* dst_ = reinterpret_cast<const bf16*>(g.C) + (long long)(m0 + tid) * g.ldc + n0;
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                       :: "l"(dst_), "r"(stg + (unsigned)tid * 272u), "r"((unsigned)(BN * 2)) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
        bulk_pending = true;
      } else {
#pragma unroll 1
      for (int it = 0; it < PER; it += G) {
        uint4 vv[G];
        long long oo[G];
#pragma unroll
        for (int e = 0; e < G; ++e) {
          const int id = (it + e) * 256 + tid, row = id / CPR, cc = id % CPR;
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(vv[e].x), "=r"(vv[e].y), "=r"(vv[e].z), "=r"(vv[e].w) : "r"(stg + swz(row, cc * 8)));
          oo[e] = (long long)(m0 + row) * g.ldc + n0 + cc * 8;
          if constexpr (!IsNoEpi<Epi>::value) ld(e, (unsigned)oo[e]);
        }
#pragma unroll
        for (int e = 0; e < G; ++e) {
          const uint4 v = vv[e];
          const long long off = oo[e];
          if constexpr (!IsNoEpi<Epi>::value) {
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(e, (unsigned)off, gv);
          }
          if constexpr (!STORE) {
          } else if constexpr (OUT == 0) {
            *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(g.C) + off) = v;
          } else if constexpr (OUT == 2) {
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
            if (g.acc_first && p5c_first) {
              dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
              dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
            } else {
            const float4 a0 = dst[0], a1 = dst[1];
            dst[0] = make_float4(a0.x + __uint_as_float(v.x << 16), a0.y + __uint_as_float(v.x & 0xffff0000u),
                                 a0.z + __uint_as_float(v.y << 16), a0.w + __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(a1.x + __uint_as_float(v.z << 16), a1.y + __uint_as_float(v.z & 0xffff0000u),
                                 a1.z + __uint_as_float(v.w << 16), a1.w + __uint_as_float(v.w & 0xffff0000u));
            }
          } else {
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
            dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
          }
        }
      }
      }
      if (g.dlt_out != nullptr) {
        // P9-B: delta[bh][i] = rowsum(dO * O32) for the tile's 128 rows x 2 heads, one quad per (row, head):
        // part p sums its 16 consecutive head elements in order, then xor-1 and xor-2 butterflies (delta_table).
        float4 dov[4][4];
        int dbh[4], di[4];
#pragma unroll
        for (int ps = 0; ps < 4; ++ps) {
          const int q = ps * 64 + (tid >> 2), part = tid & 3, row = q >> 1, hh = q & 1;
          const int r = m0 + row, b = r / g.dlt_t;
          di[ps] = r - b * g.dlt_t;
          dbh[ps] = b * g.dlt_heads + (n0 >> 6) + hh;
          const float4* o4 = reinterpret_cast<const float4*>(g.dlt_o32 + ((long long)dbh[ps] * g.dlt_t + di[ps]) * 64 + part * 16);
#pragma unroll
          for (int k = 0; k < 4; ++k) dov[ps][k] = __ldcg(o4 + k);
        }
#pragma unroll
        for (int ps = 0; ps < 4; ++ps) {
          const int q = ps * 64 + (tid >> 2), part = tid & 3, row = q >> 1, hh = q & 1;
          uint4 d0, d1;
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(d0.x), "=r"(d0.y), "=r"(d0.z), "=r"(d0.w) : "r"(stg + swz(row, hh * 64 + part * 16)));
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(d1.x), "=r"(d1.y), "=r"(d1.z), "=r"(d1.w) : "r"(stg + swz(row, hh * 64 + part * 16 + 8)));
          const uint32_t w[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
          const float of[16] = {dov[ps][0].x, dov[ps][0].y, dov[ps][0].z, dov[ps][0].w, dov[ps][1].x, dov[ps][1].y,
                                dov[ps][1].z, dov[ps][1].w, dov[ps][2].x, dov[ps][2].y, dov[ps][2].z, dov[ps][2].w,
                                dov[ps][3].x, dov[ps][3].y, dov[ps][3].z, dov[ps][3].w};
          float acc = 0.f;
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
            acc = __fadd_rn(acc, __fmul_rn(dk, of[k]));
          }
          acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
          acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
          if (part == 0) g.dlt_out[(long long)dbh[ps] * g.dlt_t + di[ps]] = acc;
        }
        if (g.qn_out != nullptr) {
          if (tid < 4) p5c_qn_sh[tid] = 0u;                         // block_norm_table starts its maximum at +0
          __syncthreads();
          uint4 qv[4][2];
#pragma unroll
          for (int ps = 0; ps < 4; ++ps) {
            const int part = tid & 3, hh = ((ps * 64 + (tid >> 2)) & 1), bh = dbh[ps];
            const bf16* src = g.qn_q + (long long)(bh / g.dlt_heads) * g.qn_sb + (long long)(bh % g.dlt_heads) * g.qn_sh
                              + (long long)di[ps] * g.qn_sr + part * 16;
            (void)hh;
            qv[ps][0] = __ldg(reinterpret_cast<const uint4*>(src));
            qv[ps][1] = __ldg(reinterpret_cast<const uint4*>(src) + 1);
          }
#pragma unroll
          for (int ps = 0; ps < 4; ++ps) {
            const int q = ps * 64 + (tid >> 2), row = q >> 1, hh = q & 1;
            const uint32_t w[8] = {qv[ps][0].x, qv[ps][0].y, qv[ps][0].z, qv[ps][0].w,
                                   qv[ps][1].x, qv[ps][1].y, qv[ps][1].z, qv[ps][1].w};
            float acc = 0.f;
#pragma unroll
            for (int d = 0; d < 16; ++d) {
              const float x = __uint_as_float((d & 1) ? (w[d >> 1] & 0xffff0000u) : (w[d >> 1] << 16));
              acc = __fadd_rn(acc, __fmul_rn(x, x));
            }
            acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
            acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
            if ((tid & 3) == 0) atomicMax(&p5c_qn_sh[(row >> 6) * 2 + hh], __float_as_uint(sqrtf(acc)));
          }
          __syncthreads();
          if (tid < 4) {
            const int blk = tid >> 1, hh = tid & 1, r = m0 + blk * 64, b = r / g.dlt_t, i = r - b * g.dlt_t;
            const int bh = b * g.dlt_heads + (n0 >> 6) + hh;
            g.qn_out[bh * (g.dlt_t / 64) + i / 64] = __uint_as_float(p5c_qn_sh[tid]);
          }
        }
      }
      __syncthreads();                                              // staging reusable by the next tile
    } else
    {
      int m0, n0;
      where(task, m0, n0);
      const int t = lane & 3;
      // Output chunks of this thread: (i, jj, h) -> 8 contiguous columns of one row. Generated epilogues prefetch
      // their identity operands for EG chunks before computing any of them (memory-level parallelism).
      constexpr int JJ = Cf::WTN / 32, NCH = Cf::MT * JJ * 2, EG = NCH < 8 ? NCH : (NCH % 8 == 0 ? 8 : (NCH % 6 == 0 ? 6 : 4));   // P9-B: 192-wide tiles (NCH 12)
      auto offset = [&](int c) {
        const int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
        const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
        const int col = n0 + wgn * Cf::WTN + 8 * (4 * jj + t);
        return (long long)row * g.ldc + col;
      };
#pragma unroll
      for (int c0 = 0; c0 < NCH; c0 += EG) {
        if constexpr (!IsNoEpi<Epi>::value) {
#pragma unroll
          for (int e = 0; e < EG; ++e) ld(e, (unsigned)offset(c0 + e));
        }
#ifdef P5C_ACC_BATCH
        // P9-B: FP32-accumulating stores read the EG chunks' old values first (independent loads in flight together;
        // the stores below could alias later loads, so the compiler would otherwise serialize one DRAM latency per
        // chunk). Same values, same adds, same stores.
        float4 p5c_old[OUT == 2 ? EG : 1][2];
        if constexpr (OUT == 2 && STORE) {
          if (!(g.acc_first && p5c_first)) {
#pragma unroll
            for (int e = 0; e < EG; ++e) {
              const float4* src = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(g.C) + offset(c0 + e));
              p5c_old[e][0] = src[0];
              p5c_old[e][1] = src[1];
            }
          }
        }
#endif
#pragma unroll
        for (int e = 0; e < EG; ++e) {
          const int c = c0 + e, i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
          if constexpr (OUT == 3) {
            // LATCH-DENSE-EXEC: the unrounded FP32 accumulators (the FP32 product of BF16-exact operands). Even and
            // odd columns are quad-transposed separately (data movement only), giving 8 consecutive FP32 columns.
            const uint4 lo4 = quad_chunk2(__float_as_uint(acc[i][4 * j0 + 2 * h]),
                                          __float_as_uint(acc[i][4 * (j0 + 1) + 2 * h]),
                                          __float_as_uint(acc[i][4 * (j0 + 2) + 2 * h]),
                                          __float_as_uint(acc[i][4 * (j0 + 3) + 2 * h]), t);
            const uint4 hi4 = quad_chunk2(__float_as_uint(acc[i][4 * j0 + 2 * h + 1]),
                                          __float_as_uint(acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                          __float_as_uint(acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                          __float_as_uint(acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
            const float gv[8] = {__uint_as_float(lo4.x), __uint_as_float(hi4.x), __uint_as_float(lo4.y),
                                 __uint_as_float(hi4.y), __uint_as_float(lo4.z), __uint_as_float(hi4.z),
                                 __uint_as_float(lo4.w), __uint_as_float(hi4.w)};
            const long long off = offset(c);
            if constexpr (!IsNoEpi<Epi>::value) epi(e, (unsigned)off, gv);
            if constexpr (STORE) {
              float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
              dst[0] = make_float4(gv[0], gv[1], gv[2], gv[3]);
              dst[1] = make_float4(gv[4], gv[5], gv[6], gv[7]);
            }
          } else {
          const uint4 v = quad_chunk2(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
          const long long off = offset(c);
          if constexpr (!IsNoEpi<Epi>::value) {
            // Exact widening of the rounded BF16 results, 8 contiguous outputs at flat index off (ldc == N).
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(e, (unsigned)off, gv);
          }
          if constexpr (!STORE) {
          } else if constexpr (OUT == 0) {
            *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(g.C) + off) = v;
          } else if constexpr (OUT == 2) {
            // P7-C: accumulate the exact widening into FP32 storage (one FP32 add per element, as the separate
            // conversion-and-accumulate it replaces).
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
            if (g.acc_first && p5c_first) {
              dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
              dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
            } else {
#ifdef P5C_ACC_BATCH
            const float4 a0 = p5c_old[e][0], a1 = p5c_old[e][1];
#else
            const float4 a0 = dst[0], a1 = dst[1];
#endif
            dst[0] = make_float4(a0.x + __uint_as_float(v.x << 16), a0.y + __uint_as_float(v.x & 0xffff0000u),
                                 a0.z + __uint_as_float(v.y << 16), a0.w + __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(a1.x + __uint_as_float(v.z << 16), a1.y + __uint_as_float(v.z & 0xffff0000u),
                                 a1.z + __uint_as_float(v.w << 16), a1.w + __uint_as_float(v.w & 0xffff0000u));
            }
          } else {
            // Exact widening of the rounded BF16 pair (bits << 16), 8 consecutive FP32 columns.
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
            dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
          }
          }
        }
      }
    }
    p5c_stamp(4);
  }
  if constexpr (BULK_) {
    if (bulk_pending && tid < BM) {                                // writes complete, then visible to generic loads
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (MC == 2) cluster_sync();                                     // peer's remote arrivals and multicasts landed
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_inval(bar0 + 8 * st);
      if (REL || MC == 2) pops4b::bar_inval(ebar0 + 8 * st);
    }
    pops4b::proxy_fence();
  }
  p5c_stamp(5);
#undef P5C_ISSUE
#undef P5C_ISSUE_BY
#undef P5C_RELEASE_AND_REFILL
}

// P9-B chained products (lower.P9_CHAIN): member maps, outputs and stage-table entries of the running chain.
#ifndef P5C_CH_MAX
#define P5C_CH_MAX 6
#endif
struct P5cCh { const void* tA; const void* tB; void* C; unsigned long long srow; };
__shared__ P5cCh p5c_ch[P5C_CH_MAX];
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, bool STORE = true, class Epi = NoEpi,
          class Ld = NoLd, int MC = 1, int SK = 1, int NCH = 2>
DEV void chain_body(const Args (&gs)[NCH], float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld()) {
  static_assert(OUT != 3, "raw FP32 output (LATCH-DENSE-EXEC) is emitted only by body");
  // P9-B chained products: NCH products with the same shape and configuration run as one pipeline over the
  // concatenation of their stage-table tile lists (member maps / outputs / entries in p5c_ch); every tile's
  // arithmetic is the member's own (bitwise to running the members one after another).
  static_assert(MC == 1 && SK == 1 && NCH >= 2 && NCH <= P5C_CH_MAX, "chain");
  const Args& g = gs[0];
  using Cf = Cfg<BM, BN, WM>;
  // LATCH-STEP-06 (P5C_BULK_EPI): a plain staged 128 x 128 tile (no generated epilogue, BF16 store) is written by
  // 128 per-row bulk async copies from an unswizzled staging buffer (272-byte rows) while the next tile's main
  // loop runs; the buffer is reused only after those copies have read it. Same bytes, same addresses.
  constexpr bool BULK_ = P5C_STAGED_EPI && P5C_BULK_EPI && BM == 128 && BN == 128 && MC == 1 && SK == 1 &&
                         IsNoEpi<Epi>::value && OUT == 0 && STORE;
  bool bulk_pending = false;
  static_assert(NST >= 2 && NST <= 8, "stages");
  static_assert(!AK || BM <= 256, "TMA box rows");
  static_assert(!BKC || BN <= 256, "TMA box rows");
  static_assert(Cf::WTN % 32 == 0, "quad-transposed row chunks");
  static_assert(MC == 1 || (MC == 2 && BM % 128 == 0), "multicast splits A into whole halves");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = tm * tn;
  // MC = 2: whole grid only (codegen), even grid (cluster launch); cluster c owns N-tile pairs (2j, 2j + 1) of one M row.
  const unsigned rank = MC == 2 ? cluster_rank() : 0u;
  const int unit = MC == 2 ? (int)blockIdx.x >> 1 : s.me, units = MC == 2 ? (int)gridDim.x >> 1 : s.count;
  // SK > 1 (P7-C split-K): work units are (tile, K slice), the slices of one tile on neighbouring units.
  const int nunit = MC == 2 ? tm * (tn / 2) : ntask * SK;
  int ntm = unit < nunit ? (nunit - unit + units - 1) / units : 0;
  const unsigned short* tl = nullptr;
  int first = 0;
  if constexpr (MC == 1 && SK == 1) {
    bool tab_ = true;
#pragma unroll
    for (int p_ = 0; p_ < NCH; ++p_) tab_ = tab_ && (gs[p_].srow >> 63);
    if (tab_ && lo == 0 && hi == 1 && den == 1) {
      tl = g.stl;
      ntm = 0;
#pragma unroll
      for (int p_ = 0; p_ < NCH; ++p_) ntm += (int)((gs[p_].srow >> 48) & 0x7fffull);
    } else if (true) {
      // off the planned grid: every member takes the rotation share of this CTA (same tiles for each member)
      tl = nullptr;
      ntm *= NCH;
    } else if (g.sch != nullptr && lo == 0 && hi == 1 && den == 1 && (int)gridDim.x == g.sgrid) {
      const unsigned short* row = g.sch + blockIdx.x * (g.scount + 1) + g.sidx;
      tl = g.sch + g.sgrid * (g.scount + 1) + row[0];
      ntm = row[1] - row[0];
      first = ntm > 0 ? (int)tl[0] : 0;
    }
  }
  int ccnt_[NCH];
#pragma unroll
  for (int p_ = 0; p_ < NCH; ++p_) ccnt_[p_] = tl ? (int)((gs[p_].srow >> 48) & 0x7fffull) : ntm / NCH;
  // task i -> (member, tile) through the members' table lists in order
  auto chtile = [&](int i, int& cp, int& u) {
    int j = i;
    cp = 0;
#pragma unroll
    for (int k_ = 0; k_ < NCH - 1; ++k_)
      if (cp == k_ && j >= ccnt_[k_]) { j -= ccnt_[k_]; cp = k_ + 1; }
    const unsigned long long sr = p5c_ch[cp].srow;
    u = !tl ? unit + j * units
            : (j == 0 ? (int)((sr >> 32) & 0xffffull) : (int)g.stl[(unsigned)(sr & 0xffffffffull) + j]);
  };
  auto chargs = [&](int cp) { Args a_ = g; a_.tA = p5c_ch[cp].tA; a_.tB = p5c_ch[cp].tB; return a_; };
  const int nk = g.K / 64 / SK, Q = ntm * nk;
  if (Q == 0) return;                                              // CTA-uniform (and cluster-uniform for MC = 2)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&pops4b::tma_bars[0]), ebar0 = sa(&p5c_empty[0]);
  auto where = [&](int i, int& m0, int& n0) {
    int cp_w, u;
    chtile(i, cp_w, u);
    if (MC == 2) {
      m0 = (u / (tn / 2)) * BM;
      n0 = (2 * (u % (tn / 2)) + (int)rank) * BN;
    } else {
      m0 = ((u / SK) / tn) * BM;
      n0 = ((u / SK) % tn) * BN;
    }
  };
  const bool leader = tid == 0;
  __syncthreads();                                                 // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int p_ = 0; p_ < NCH; ++p_) {
      p5c_ch[p_].tA = gs[p_].tA; p5c_ch[p_].tB = gs[p_].tB; p5c_ch[p_].C = gs[p_].C; p5c_ch[p_].srow = gs[p_].srow;
    }
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_init(bar0 + 8 * st, 1);
      if (REL || MC == 2) pops4b::bar_init(ebar0 + 8 * st, 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
#ifdef P5C_DECOUPLED
#pragma unroll
    for (int st = 0; st < 8; ++st) p5c_cnt[st] = 0u;
#endif
  }
  __syncthreads();
  p5c_stamp(1);
  if (MC == 2) cluster_sync();                                     // peer barriers exist before any multicast lands
#define P5C_ISSUE(QQ) P5C_ISSUE_BY(QQ, leader)
#define P5C_ISSUE_BY(QQ, LEAD)                                                                                       \
  { const bool lead_ = (LEAD); const int qq_ = (QQ), i_ = qq_ / nk; int cp_, u_; chtile(i_, cp_, u_); int m_, n_;      \
    if (MC == 2) { m_ = (u_ / (tn / 2)) * BM; n_ = (2 * (u_ % (tn / 2)) + (int)rank) * BN;                        \
      stage_issue_mc<BM, BN, AK, BKC>(g, lead_, rank, m_, n_, (qq_ % nk) * 64, base + (qq_ % NST) * Cf::STAGE,     \
                                      bar0 + 8 * (qq_ % NST)); }                                                   \
    else { m_ = ((u_ / SK) / tn) * BM; n_ = ((u_ / SK) % tn) * BN;                                                  \
      stage_issue<BM, BN, AK, BKC>(chargs(cp_), lead_, m_, n_, ((qq_ % nk) + (u_ % SK) * nk) * 64,            \
                                   base + (qq_ % NST) * Cf::STAGE, bar0 + 8 * (qq_ % NST)); } }
  // Release of stage (QQ) % NST by this CTA; for MC = 2 also to the peer, then wait until both released it.
#define P5C_RELEASE_AND_REFILL(QQ)                                                                                   \
  { const int rq_ = (QQ);                                                                                          \
    if (MC == 2) {                                                                                                 \
      const unsigned eb_ = ebar0 + 8 * (rq_ % NST);                                                                 \
      release_pair_if(leader, eb_, peer_addr(eb_, rank ^ 1u));                                                      \
      if (rq_ + NST < Q) { wait_parity(eb_, (unsigned)(rq_ / NST) & 1u); P5C_ISSUE(rq_ + NST) }                    \
    } else if (rq_ + NST < Q) P5C_ISSUE(rq_ + NST) }
  {
    const int pre = Q < NST ? Q : NST;
#ifdef P5C_PAR_PRE
    // P9-B: lanes 0..pre-1 of warp 0 issue one prologue stage each in a single predicated pass (same loads).
    { const int qp_ = tid < pre ? tid : 0; P5C_ISSUE_BY(qp_, tid < pre) }
#else
#pragma unroll 1
    for (int q = 0; q < pre; ++q) P5C_ISSUE(q)
#endif
  }
  float acc[Cf::MT][Cf::R];
  // Batch of one 64-k stage: 4 k16 steps x MT accumulator slabs per warpgroup.
  // CUTLASS-style invariant: the inner k loop is entered and re-entered with exactly one committed batch in flight;
  // every control-flow merge sees the same asynchronous accumulator state (no compiler-inserted wait in a divergent
  // path), and the accumulators complete (wait 0) before the epilogue and before the next task.
  int q = 0;
#pragma unroll 1
  for (int task = 0; task < ntm; ++task) {
    p5c_stamp(2);

    stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, true);                                                 // peeled first stage of this tile
    ++q;
#ifdef P5C_SYNC2
    // P6-D: one CTA barrier and stage refill per two 64-k steps (NST >= 4 keeps a refill lead of >= 2 steps).
    int kk = 1;
#pragma unroll 1
    for (; kk + 1 < nk; kk += 2, q += 2) {
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q + 1) % NST), (unsigned)((q + 1) / NST) & 1u, base + ((q + 1) % NST) * Cf::STAGE, base + ((q + 1) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();                                         // steps q-1 and q complete in this warpgroup
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
      __syncthreads();                                              // ... and in the other one: both stages reusable
      if (q - 1 + NST < Q) P5C_ISSUE(q - 1 + NST)                   // P9-B: table-aware tile mapping
      if (q + NST < Q) P5C_ISSUE(q + NST)
    }
#pragma unroll 1
    for (; kk < nk; ++kk, ++q) {
#else
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
#endif
      stage_batch<Cf::MT, Cf::WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NST), (unsigned)((q) / NST) & 1u, base + ((q) % NST) * Cf::STAGE, base + ((q) % NST) * Cf::STAGE + Cf::A_BYTES, wgm * Cf::WTM, wgn * Cf::WTN, false);
      pops4b::wg_wait<1>();                                         // step q-1 complete in this warpgroup
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
#ifdef P5C_DECOUPLED
      if (q - 1 + NST < Q) { const bool issue_ = second_arrival((tid & 127) == 0, sa(&p5c_cnt[(q - 1) % NST])); P5C_ISSUE_BY(q - 1 + NST, issue_) }
#else
      __syncthreads();                                              // ... and in the other one: stage reusable
      P5C_RELEASE_AND_REFILL(q - 1)                                 // CTA-uniform conditions
#endif
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) pops4b::keep(acc[i]);
#ifdef P5C_DECOUPLED
    if (q - 1 + NST < Q) { const bool issue_ = second_arrival((tid & 127) == 0, sa(&p5c_cnt[(q - 1) % NST])); P5C_ISSUE_BY(q - 1 + NST, issue_) }
#else
    __syncthreads();
    P5C_RELEASE_AND_REFILL(q - 1)                                   // last stage of this tile is free
#endif
    p5c_stamp(3);
    int tcp_, tcu_;
    chtile(task, tcp_, tcu_);
    void* const cC_ = p5c_ch[tcp_].C;
    if constexpr (SK > 1) {
      // Deterministic split-K: every slice stores its FP32 fragment; the last slice to arrive sums the SK partials
      // in slice order 0..SK-1 (FP32 association of the K sum changes; one BF16 rounding of the full sum follows).
      const int u = unit + task * units, tile = u / SK, slice = u % SK;
      constexpr int FR = Cf::MT * Cf::R;
      float* mine = g.part + ((long long)(slice * ntask + tile) * 256 + tid) * FR;
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int r = 0; r < Cf::R; r += 4)
          *reinterpret_cast<float4*>(mine + i * Cf::R + r) = make_float4(acc[i][r], acc[i][r + 1], acc[i][r + 2], acc[i][r + 3]);
      // LATCH-STEP10-SPEED: without counters (cnt == nullptr) the slice partials are summed in slice order by a later
      // phase over the whole grid (lower.py conv split-K reduce) instead of by each tile's last arriver.
      if (g.cnt == nullptr) continue;                               // uniform per product
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const unsigned prev = atomicAdd(g.cnt + tile, 1u);
        p5c_sk_last = prev == (unsigned)(SK - 1);
        if (p5c_sk_last) g.cnt[tile] = 0u;                         // reusable by the next instance of this product
      }
      __syncthreads();
      if (!p5c_sk_last) continue;                                   // CTA-uniform
      __threadfence();
#pragma unroll
      for (int sl = 0; sl < SK; ++sl) {
        const float* src = g.part + ((long long)(sl * ntask + tile) * 256 + tid) * FR;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int r = 0; r < Cf::R; r += 4) {
            const float4 v4 = __ldcg(reinterpret_cast<const float4*>(src + i * Cf::R + r));
            if (sl == 0) { acc[i][r] = v4.x; acc[i][r + 1] = v4.y; acc[i][r + 2] = v4.z; acc[i][r + 3] = v4.w; }
            else { acc[i][r] += v4.x; acc[i][r + 1] += v4.y; acc[i][r + 2] += v4.z; acc[i][r + 3] += v4.w; }
          }
      }
    }
    // P9-B (P5C_STAGED_EPI): 128 x 128 tiles stage their BF16-rounded results in a swizzled 32 KB buffer after the
    // stage ring and run the epilogue from a rolled loop over row-contiguous 16-byte chunks (half the epilogue code,
    // coalesced stores). The values each chunk sees are the same BF16 roundings; outputs are bitwise unchanged.
    constexpr bool STG = P5C_STAGED_EPI && BM == 128 && BN == 128 && MC == 1 && SK == 1;
    if constexpr (STG) {
      int m0, n0;
      where(task, m0, n0);
      const int t = lane & 3;
      const unsigned stg = base + NST * Cf::STAGE;
      auto swz = [](int row, int col) -> unsigned {                // byte offset of bf16 (row, col), 256 B rows
        if constexpr (BULK_) return (unsigned)(row * 272 + (col << 1));   // bulk: 272 B rows (conflict-free stores)
        return (unsigned)(row * 256 + ((((col >> 3) ^ (row & 15))) << 4) + ((col & 7) << 1));
      };
      if constexpr (BULK_) {
        if (bulk_pending) {                                        // previous tile's copies have read the buffer
          if (tid < BM) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          __syncthreads();
        }
      }
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
          const int r0 = wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2), col = wgn * Cf::WTN + nb * 8 + 2 * t;
          const uint32_t u0 = s101h::bpk(acc[i][4 * nb], acc[i][4 * nb + 1]);
          const uint32_t u1 = s101h::bpk(acc[i][4 * nb + 2], acc[i][4 * nb + 3]);
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(stg + swz(r0, col)), "r"(u0) : "memory");
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(stg + swz(r0 + 8, col)), "r"(u1) : "memory");
        }
      if constexpr (BULK_) asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic -> async
      __syncthreads();
      constexpr int CPR = BN / 8, PER = BM * CPR / 256, G = 4;
      if constexpr (BULK_) {
        if (tid < BM) {
          const bf16* dst_ = reinterpret_cast<const bf16*>(cC_) + (long long)(m0 + tid) * g.ldc + n0;
          asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                       :: "l"(dst_), "r"(stg + (unsigned)tid * 272u), "r"((unsigned)(BN * 2)) : "memory");
          asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
        bulk_pending = true;
      } else {
#pragma unroll 1
      for (int it = 0; it < PER; it += G) {
        uint4 vv[G];
        long long oo[G];
#pragma unroll
        for (int e = 0; e < G; ++e) {
          const int id = (it + e) * 256 + tid, row = id / CPR, cc = id % CPR;
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(vv[e].x), "=r"(vv[e].y), "=r"(vv[e].z), "=r"(vv[e].w) : "r"(stg + swz(row, cc * 8)));
          oo[e] = (long long)(m0 + row) * g.ldc + n0 + cc * 8;
          if constexpr (!IsNoEpi<Epi>::value) ld(e, (unsigned)oo[e]);
        }
#pragma unroll
        for (int e = 0; e < G; ++e) {
          const uint4 v = vv[e];
          const long long off = oo[e];
          if constexpr (!IsNoEpi<Epi>::value) {
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(e, (unsigned)off, gv);
          }
          if constexpr (!STORE) {
          } else if constexpr (OUT == 0) {
            *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(cC_) + off) = v;
          } else if constexpr (OUT == 2) {
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(cC_) + off);
            if (g.acc_first && p5c_first) {
              dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
              dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
            } else {
            const float4 a0 = dst[0], a1 = dst[1];
            dst[0] = make_float4(a0.x + __uint_as_float(v.x << 16), a0.y + __uint_as_float(v.x & 0xffff0000u),
                                 a0.z + __uint_as_float(v.y << 16), a0.w + __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(a1.x + __uint_as_float(v.z << 16), a1.y + __uint_as_float(v.z & 0xffff0000u),
                                 a1.z + __uint_as_float(v.w << 16), a1.w + __uint_as_float(v.w & 0xffff0000u));
            }
          } else {
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(cC_) + off);
            dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
          }
        }
      }
      }
      if (g.dlt_out != nullptr) {
        // P9-B: delta[bh][i] = rowsum(dO * O32) for the tile's 128 rows x 2 heads, one quad per (row, head):
        // part p sums its 16 consecutive head elements in order, then xor-1 and xor-2 butterflies (delta_table).
        float4 dov[4][4];
        int dbh[4], di[4];
#pragma unroll
        for (int ps = 0; ps < 4; ++ps) {
          const int q = ps * 64 + (tid >> 2), part = tid & 3, row = q >> 1, hh = q & 1;
          const int r = m0 + row, b = r / g.dlt_t;
          di[ps] = r - b * g.dlt_t;
          dbh[ps] = b * g.dlt_heads + (n0 >> 6) + hh;
          const float4* o4 = reinterpret_cast<const float4*>(g.dlt_o32 + ((long long)dbh[ps] * g.dlt_t + di[ps]) * 64 + part * 16);
#pragma unroll
          for (int k = 0; k < 4; ++k) dov[ps][k] = __ldcg(o4 + k);
        }
#pragma unroll
        for (int ps = 0; ps < 4; ++ps) {
          const int q = ps * 64 + (tid >> 2), part = tid & 3, row = q >> 1, hh = q & 1;
          uint4 d0, d1;
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(d0.x), "=r"(d0.y), "=r"(d0.z), "=r"(d0.w) : "r"(stg + swz(row, hh * 64 + part * 16)));
          asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(d1.x), "=r"(d1.y), "=r"(d1.z), "=r"(d1.w) : "r"(stg + swz(row, hh * 64 + part * 16 + 8)));
          const uint32_t w[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
          const float of[16] = {dov[ps][0].x, dov[ps][0].y, dov[ps][0].z, dov[ps][0].w, dov[ps][1].x, dov[ps][1].y,
                                dov[ps][1].z, dov[ps][1].w, dov[ps][2].x, dov[ps][2].y, dov[ps][2].z, dov[ps][2].w,
                                dov[ps][3].x, dov[ps][3].y, dov[ps][3].z, dov[ps][3].w};
          float acc = 0.f;
#pragma unroll
          for (int k = 0; k < 16; ++k) {
            const float dk = __uint_as_float((k & 1) ? (w[k >> 1] & 0xffff0000u) : (w[k >> 1] << 16));
            acc = __fadd_rn(acc, __fmul_rn(dk, of[k]));
          }
          acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
          acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
          if (part == 0) g.dlt_out[(long long)dbh[ps] * g.dlt_t + di[ps]] = acc;
        }
        if (g.qn_out != nullptr) {
          if (tid < 4) p5c_qn_sh[tid] = 0u;                         // block_norm_table starts its maximum at +0
          __syncthreads();
          uint4 qv[4][2];
#pragma unroll
          for (int ps = 0; ps < 4; ++ps) {
            const int part = tid & 3, hh = ((ps * 64 + (tid >> 2)) & 1), bh = dbh[ps];
            const bf16* src = g.qn_q + (long long)(bh / g.dlt_heads) * g.qn_sb + (long long)(bh % g.dlt_heads) * g.qn_sh
                              + (long long)di[ps] * g.qn_sr + part * 16;
            (void)hh;
            qv[ps][0] = __ldg(reinterpret_cast<const uint4*>(src));
            qv[ps][1] = __ldg(reinterpret_cast<const uint4*>(src) + 1);
          }
#pragma unroll
          for (int ps = 0; ps < 4; ++ps) {
            const int q = ps * 64 + (tid >> 2), row = q >> 1, hh = q & 1;
            const uint32_t w[8] = {qv[ps][0].x, qv[ps][0].y, qv[ps][0].z, qv[ps][0].w,
                                   qv[ps][1].x, qv[ps][1].y, qv[ps][1].z, qv[ps][1].w};
            float acc = 0.f;
#pragma unroll
            for (int d = 0; d < 16; ++d) {
              const float x = __uint_as_float((d & 1) ? (w[d >> 1] & 0xffff0000u) : (w[d >> 1] << 16));
              acc = __fadd_rn(acc, __fmul_rn(x, x));
            }
            acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 1));
            acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, 2));
            if ((tid & 3) == 0) atomicMax(&p5c_qn_sh[(row >> 6) * 2 + hh], __float_as_uint(sqrtf(acc)));
          }
          __syncthreads();
          if (tid < 4) {
            const int blk = tid >> 1, hh = tid & 1, r = m0 + blk * 64, b = r / g.dlt_t, i = r - b * g.dlt_t;
            const int bh = b * g.dlt_heads + (n0 >> 6) + hh;
            g.qn_out[bh * (g.dlt_t / 64) + i / 64] = __uint_as_float(p5c_qn_sh[tid]);
          }
        }
      }
      __syncthreads();                                              // staging reusable by the next tile
    } else
    {
      int m0, n0;
      where(task, m0, n0);
      const int t = lane & 3;
      // Output chunks of this thread: (i, jj, h) -> 8 contiguous columns of one row. Generated epilogues prefetch
      // their identity operands for EG chunks before computing any of them (memory-level parallelism).
      constexpr int JJ = Cf::WTN / 32, NCH = Cf::MT * JJ * 2, EG = NCH < 8 ? NCH : (NCH % 8 == 0 ? 8 : (NCH % 6 == 0 ? 6 : 4));   // P9-B: 192-wide tiles (NCH 12)
      auto offset = [&](int c) {
        const int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
        const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
        const int col = n0 + wgn * Cf::WTN + 8 * (4 * jj + t);
        return (long long)row * g.ldc + col;
      };
#pragma unroll
      for (int c0 = 0; c0 < NCH; c0 += EG) {
        if constexpr (!IsNoEpi<Epi>::value) {
#pragma unroll
          for (int e = 0; e < EG; ++e) ld(e, (unsigned)offset(c0 + e));
        }
#ifdef P5C_ACC_BATCH
        // P9-B: FP32-accumulating stores read the EG chunks' old values first (independent loads in flight together;
        // the stores below could alias later loads, so the compiler would otherwise serialize one DRAM latency per
        // chunk). Same values, same adds, same stores.
        float4 p5c_old[OUT == 2 ? EG : 1][2];
        if constexpr (OUT == 2 && STORE) {
          if (!(g.acc_first && p5c_first)) {
#pragma unroll
            for (int e = 0; e < EG; ++e) {
              const float4* src = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(cC_) + offset(c0 + e));
              p5c_old[e][0] = src[0];
              p5c_old[e][1] = src[1];
            }
          }
        }
#endif
#pragma unroll
        for (int e = 0; e < EG; ++e) {
          const int c = c0 + e, i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
          const uint4 v = quad_chunk2(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                             s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
          const long long off = offset(c);
          if constexpr (!IsNoEpi<Epi>::value) {
            // Exact widening of the rounded BF16 results, 8 contiguous outputs at flat index off (ldc == N).
            const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                 __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
            epi(e, (unsigned)off, gv);
          }
          if constexpr (!STORE) {
          } else if constexpr (OUT == 0) {
            *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(cC_) + off) = v;
          } else if constexpr (OUT == 2) {
            // P7-C: accumulate the exact widening into FP32 storage (one FP32 add per element, as the separate
            // conversion-and-accumulate it replaces).
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(cC_) + off);
            if (g.acc_first && p5c_first) {
              dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
              dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
            } else {
#ifdef P5C_ACC_BATCH
            const float4 a0 = p5c_old[e][0], a1 = p5c_old[e][1];
#else
            const float4 a0 = dst[0], a1 = dst[1];
#endif
            dst[0] = make_float4(a0.x + __uint_as_float(v.x << 16), a0.y + __uint_as_float(v.x & 0xffff0000u),
                                 a0.z + __uint_as_float(v.y << 16), a0.w + __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(a1.x + __uint_as_float(v.z << 16), a1.y + __uint_as_float(v.z & 0xffff0000u),
                                 a1.z + __uint_as_float(v.w << 16), a1.w + __uint_as_float(v.w & 0xffff0000u));
            }
          } else {
            // Exact widening of the rounded BF16 pair (bits << 16), 8 consecutive FP32 columns.
            float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(cC_) + off);
            dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                 __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
            dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                 __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
          }
        }
      }
    }
    p5c_stamp(4);
  }
  if constexpr (BULK_) {
    if (bulk_pending && tid < BM) {                                // writes complete, then visible to generic loads
      asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      asm volatile("fence.proxy.async.global;" ::: "memory");
    }
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (MC == 2) cluster_sync();                                     // peer's remote arrivals and multicasts landed
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_inval(bar0 + 8 * st);
      if (REL || MC == 2) pops4b::bar_inval(ebar0 + 8 * st);
    }
    pops4b::proxy_fence();
  }
  p5c_stamp(5);
#undef P5C_ISSUE
#undef P5C_ISSUE_BY
#undef P5C_RELEASE_AND_REFILL
}

template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, int NCH>
DEV void phase_chain(const Args (&gs)[NCH], float* smem) {
  chain_body<BM, BN, WM, AK, BKC, NST, OUT, REL, true, NoEpi, NoLd, 1, 1, NCH>(gs, smem, 0, 1, 1, 0);
}
// Entry-resident form only (wgmma inside called functions is serialized by ptxas, C7510).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, int MC = 1>
DEV void phase(const Args g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  body<BM, BN, WM, AK, BKC, NST, OUT, REL, true, NoEpi, NoLd, MC>(g, smem, lo, hi, den, rot);
}
// P7-C split-K product (no generated epilogue): SK slices per tile, deterministic last-arriver slice sum.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, int SK, int PK = 0>
DEV void phase_sk(const Args g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  body<BM, BN, WM, AK, BKC, NST, OUT, false, true, NoEpi, NoLd, 1, SK, PK>(g, smem, lo, hi, den, rot);
}
// Release step 3 repair: chunked-accumulation forms (PK 64-k stages per wgmma chain; no multicast, no split-K).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, int PK>
DEV void phase_k(const Args g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  body<BM, BN, WM, AK, BKC, NST, OUT, false, true, NoEpi, NoLd, 1, 1, PK>(g, smem, lo, hi, den, rot);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool STORE, int PK, class Epi, class Ld>
DEV void phase_epi_k(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  body<BM, BN, WM, AK, BKC, NST, OUT, false, STORE, Epi, Ld, 1, 1, PK>(g, smem, lo, hi, den, rot, epi, ld);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool REL, bool STORE, int OUT = 0, class Epi, class Ld,
          int MC = 1>
DEV void phase_epi(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  body<BM, BN, WM, AK, BKC, NST, OUT, REL, STORE, Epi, Ld, MC>(g, smem, lo, hi, den, rot, epi, ld);
}
// P6-D: register epilogue with an explicit output form (OUT=1: exact FP32-converting store of the rounded result).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, bool STORE, class Epi, class Ld>
DEV void phase_epi_out(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  body<BM, BN, WM, AK, BKC, NST, OUT, REL, STORE, Epi, Ld>(g, smem, lo, hi, den, rot, epi, ld);
}
// LATCH-STEP-06: the same forms with an epilogue operand prefetch functor (P5C_EPI_PF).
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool REL, bool STORE, int OUT, class Epi, class Ld,
          class Pf>
DEV void phase_epi_pf(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld, Pf pf) {
  body<BM, BN, WM, AK, BKC, NST, OUT, REL, STORE, Epi, Ld, 1, 1, 0, Pf>(g, smem, lo, hi, den, rot, epi, ld, pf);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int OUT, bool REL, bool STORE, class Epi, class Ld,
          class Pf>
DEV void phase_epi_out_pf(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld, Pf pf) {
  body<BM, BN, WM, AK, BKC, NST, OUT, REL, STORE, Epi, Ld, 1, 1, 0, Pf>(g, smem, lo, hi, den, rot, epi, ld, pf);
}

// ------------------------------------------------------------------------------------------------ ping-pong body
// Each warpgroup owns a BM x BN tile (MT = BM / 64 slabs of m64nBNk16), its own NSTW-stage ring and its own TMA
// issue (elected thread, predicated). Main loops alternate between the warpgroups through two named barriers
// (CUTLASS ping-pong ordering): while one warpgroup runs its tile's MMAs at the full tensor rate, the other
// completes its accumulators and runs its epilogue (stores, conversions, generated elementwise chains).
// Both warpgroups execute the same number of iterations (a warpgroup without a task recomputes the CTA's last tile
// and stores nothing), and every warpgroup difference is an operand or a warp-uniform predicate, never a branch
// around wgmma. Arithmetic per element is identical to `body` (FP32 wgmma accumulation in ascending k16 steps).
DEV void nbar_sync_if(bool p, unsigned id) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %0, 0;\n@q bar.sync %1, 256;\n}\n" ::"r"((unsigned)p), "r"(id) : "memory");
}
DEV void nbar_arrive_if(bool p, unsigned id) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %0, 0;\n@q bar.arrive %1, 256;\n}\n" ::"r"((unsigned)p), "r"(id) : "memory");
}
DEV void nbar_sync(unsigned id) { asm volatile("bar.sync %0, 256;" ::"r"(id) : "memory"); }
DEV void nbar_arrive(unsigned id) { asm volatile("bar.arrive %0, 256;" ::"r"(id) : "memory"); }
DEV void wg_bar(unsigned id) { asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory"); }

template <int BM, int BN, bool AK, bool BKC, int NSTW, int OUT, bool STORE = true, class Epi = NoEpi, class Ld = NoLd>
DEV void pp_body(const Args& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld()) {
  static_assert(OUT != 3, "raw FP32 output (LATCH-DENSE-EXEC) is emitted only by body");
  constexpr int MT = BM / 64, WTN = BN, R = WTN / 2;
  constexpr int A_BYTES = BM * 128, STAGE = (BM + BN) * 128;
  static_assert(BM % 64 == 0 && MT <= 2 && WTN % 32 == 0 && WTN <= 256 && 2 * NSTW <= 8, "ping-pong tile");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = g.M / BM, tn = g.N / BN, ntask = tm * tn;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  if (ntm == 0) return;                                             // CTA-uniform
  const int nloc = (ntm + 1) / 2, nk = g.K / 64, Q = nloc * nk;     // per-warpgroup iterations (uniform)
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const bool leader = (tid & 127) == 0;
  const unsigned ring = ((sa(smem) + 1023u) & ~1023u) + (unsigned)(wg * NSTW * STAGE);
  const unsigned bar0 = sa(&pops4b::tma_bars[0]) + 8u * (unsigned)(wg * NSTW);
  // Local iteration i -> CTA task 2 i + wg; a warpgroup past the CTA's tasks recomputes task ntm - 1 (no stores).
  auto task_of = [&](int i) { const int c = 2 * i + wg; return c < ntm ? c : ntm - 1; };
  auto where = [&](int i, int& m0, int& n0) {
    const int task = s.me + task_of(i) * s.count;
    m0 = (task / tn) * BM;
    n0 = (task % tn) * BN;
  };
  __syncthreads();                                                  // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < 2 * NSTW; ++st) pops4b::bar_init(sa(&pops4b::tma_bars[0]) + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();
  }
  __syncthreads();
  {
    const int pre = Q < NSTW ? Q : NSTW;
#pragma unroll 1
    for (int q = 0; q < pre; ++q) { const int qq_ = (q), c_ = 2 * (qq_ / nk) + wg, task_ = s.me + (c_ < ntm ? c_ : ntm - 1) * s.count; stage_issue<BM, BN, AK, BKC>(g, leader, (task_ / tn) * BM, (task_ % tn) * BN, (qq_ % nk) * 64, ring + (qq_ % NSTW) * STAGE, bar0 + 8 * (qq_ % NSTW)); }
  }
  // Turn barriers: id 1 = warpgroup 0 may run MMAs, id 2 = warpgroup 1. Warpgroup 1 opens warpgroup 0's first turn.
  const unsigned my_turn = 1u + (unsigned)wg, other_turn = 2u - (unsigned)wg;
  nbar_arrive_if(wg == 1, 1u);
  float acc[MT][R];
  int q = 0;
#pragma unroll 1
  for (int it = 0; it < nloc; ++it) {
    nbar_sync(my_turn);                                             // my turn to use the tensor cores
    stage_batch<MT, WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NSTW), (unsigned)((q) / NSTW) & 1u, ring + ((q) % NSTW) * STAGE, ring + ((q) % NSTW) * STAGE + A_BYTES, 0, 0, true);
    ++q;
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
      stage_batch<MT, WTN, AK, BKC>(acc, bar0 + 8 * ((q) % NSTW), (unsigned)((q) / NSTW) & 1u, ring + ((q) % NSTW) * STAGE, ring + ((q) % NSTW) * STAGE + A_BYTES, 0, 0, false);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
      wg_bar(3u + (unsigned)wg);                                    // stage of step q-1 consumed by all 4 warps
      if (q - 1 + NSTW < Q) { const int qq_ = (q - 1 + NSTW), c_ = 2 * (qq_ / nk) + wg, task_ = s.me + (c_ < ntm ? c_ : ntm - 1) * s.count; stage_issue<BM, BN, AK, BKC>(g, leader, (task_ / tn) * BM, (task_ % tn) * BN, (qq_ % nk) * 64, ring + (qq_ % NSTW) * STAGE, bar0 + 8 * (qq_ % NSTW)); }
    }
    nbar_arrive(other_turn);                                        // other warpgroup may start its main loop
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
    wg_bar(3u + (unsigned)wg);
    if (q - 1 + NSTW < Q) { const int qq_ = (q - 1 + NSTW), c_ = 2 * (qq_ / nk) + wg, task_ = s.me + (c_ < ntm ? c_ : ntm - 1) * s.count; stage_issue<BM, BN, AK, BKC>(g, leader, (task_ / tn) * BM, (task_ % tn) * BN, (qq_ % nk) * 64, ring + (qq_ % NSTW) * STAGE, bar0 + 8 * (qq_ % NSTW)); }
    const bool valid = 2 * it + wg < ntm;
    {
      int m0, n0;
      where(it, m0, n0);
      const int t = lane & 3;
      constexpr int JJ = WTN / 32, NCH = MT * JJ * 2, EG = NCH < 8 ? NCH : (NCH % 8 == 0 ? 8 : (NCH % 6 == 0 ? 6 : 4));   // P9-B: 192-wide tiles (NCH 12)
      auto offset = [&](int c) {
        const int i = c / (JJ * 2), jj = (c / 2) % JJ, h = c % 2;
        const int row = m0 + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
        const int col = n0 + 8 * (4 * jj + t);
        return (long long)row * g.ldc + col;
      };
      if (valid) {                                                  // warpgroup-uniform; accumulators complete
#pragma unroll
        for (int c0 = 0; c0 < NCH; c0 += EG) {
          if constexpr (!IsNoEpi<Epi>::value) {
#pragma unroll
            for (int e = 0; e < EG; ++e) ld(e, (unsigned)offset(c0 + e));
          }
#pragma unroll
          for (int e = 0; e < EG; ++e) {
            const int c = c0 + e, i = c / (JJ * 2), j0 = 4 * ((c / 2) % JJ), h = c % 2;
            const uint4 v = quad_chunk2(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
            const long long off = offset(c);
            if constexpr (!IsNoEpi<Epi>::value) {
              const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                                   __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
              epi(e, (unsigned)off, gv);
            }
            if constexpr (!STORE) {
            } else if constexpr (OUT == 0) {
              *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(g.C) + off) = v;
            } else {
              float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(g.C) + off);
              dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                   __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
              dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                                   __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
            }
          }
        }
      }
    }
  }
  nbar_sync_if(wg == 0, 1u);                                        // balance warpgroup 1's final turn arrival
  __syncthreads();                                                  // every wgmma read of the stages is complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < 2 * NSTW; ++st) pops4b::bar_inval(sa(&pops4b::tma_bars[0]) + 8 * st);
    pops4b::proxy_fence();
  }
}

template <int BM, int BN, bool AK, bool BKC, int NSTW, int OUT>
DEV void pp_phase(const Args g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  pp_body<BM, BN, AK, BKC, NSTW, OUT>(g, smem, lo, hi, den, rot);
}
template <int BM, int BN, bool AK, bool BKC, int NSTW, bool STORE, class Epi, class Ld>
DEV void pp_phase_epi(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  pp_body<BM, BN, AK, BKC, NSTW, 0, STORE, Epi, Ld>(g, smem, lo, hi, den, rot, epi, ld);
}
// P6-D: ping-pong register epilogue with an explicit output form (OUT=1: exact FP32-converting store).
template <int BM, int BN, bool AK, bool BKC, int NSTW, int OUT, bool STORE, class Epi, class Ld>
DEV void pp_phase_epi_out(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  pp_body<BM, BN, AK, BKC, NSTW, OUT, STORE, Epi, Ld>(g, smem, lo, hi, den, rot, epi, ld);
}

}  // namespace p5c

#if defined(P5C_TRACE)
// P9-B diagnostic export (generated programs with P5C_TRACE): copy the per-CTA stamps (128 x P5C_TRACE_N) and counts.
extern "C" int latch_p5c_trace(unsigned long long* dst, int* counts, int reset) {
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess && dst) e = cudaMemcpyFromSymbol(dst, p5c::p5c_trace, sizeof(p5c::p5c_trace));
  if (e == cudaSuccess && counts) e = cudaMemcpyFromSymbol(counts, p5c::p5c_trace_n, sizeof(p5c::p5c_trace_n));
  if (e == cudaSuccess && reset) {
    static int zeros[128] = {0};
    e = cudaMemcpyToSymbol(p5c::p5c_trace_n, zeros, sizeof(zeros));
  }
  return (int)e;
}
extern "C" int latch_p5c_trace_n() { return P5C_TRACE_N; }
#endif
