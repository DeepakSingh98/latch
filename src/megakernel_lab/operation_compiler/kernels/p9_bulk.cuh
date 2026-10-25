// P9-B: bulk-copy streaming for large elementwise phases.
// A program runs one 256-thread CTA per SM (eight warps), too few for register-based loads to keep DRAM busy on pure
// streams (the optimizer and weight casts reach ~2 TB/s). Each CTA instead streams chunks of every FP32 operand into
// shared memory with cp.async.bulk (one thread issues, a transaction-count mbarrier completes), computes the
// unchanged per-element expressions from shared memory and stores results directly. Values are identical: every
// element is read once and computed by the same expression as the register path.
#pragma once
#include <cstdint>

namespace p9bulk {

constexpr int STAGES = 3;
__shared__ __align__(8) unsigned long long p9_bulk_bars[STAGES];

__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void init() {
  __syncthreads();                                   // earlier generic users of shared memory are done
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s)
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sa(&p9_bulk_bars[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("fence.proxy.async;" ::: "memory");   // generic global writes of earlier levels -> bulk copies
  }
  __syncthreads();
}

__device__ __forceinline__ void finish() {
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(sa(&p9_bulk_bars[s])) : "memory");
  }
  __syncthreads();
}

// Thread 0: expect `bytes` on stage s's barrier.
__device__ __forceinline__ void expect(int s, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(&p9_bulk_bars[s])), "r"(bytes) : "memory");
}
// Thread 0: bulk global -> shared copy of `bytes` (multiple of 16, 16-byte aligned) completing on stage s's barrier.
__device__ __forceinline__ void copy(void* dst, const void* src, unsigned bytes, int s) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(&p9_bulk_bars[s])) : "memory");
}
__device__ __forceinline__ void wait(int s, unsigned parity) {
  asm volatile("{\n.reg .pred P1;\nLAB_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra DONE;\n"
               "bra LAB_WAIT;\nDONE:\n}\n" ::"r"(sa(&p9_bulk_bars[s])), "r"(parity) : "memory");
}

}  // namespace p9bulk
