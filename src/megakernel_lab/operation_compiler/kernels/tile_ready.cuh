// Per-row-block readiness inside the persistent cooperative kernel (opt-in: tile_readiness, ops/tile_ready.py).
//
// A dependency boundary whose every cross-boundary read is "producer output rows -> row-local consumer" keeps no grid
// barrier. Instead:
//   * Producer (a width-32 segmented attention item, kernels/attention_segmented.cuh): after the item's rows are
//     stored, the CTA publishes their element count into one counter per RB-row block they cover. Every thread's
//     stores precede a CTA barrier; thread 0 then issues a gpu-scope fence and relaxed atomic adds (the
//     cooperative-groups grid-barrier release pattern). Rows are published only after their final stores.
//   * Consumer (a row-local elementwise phase): CTAs take row blocks by static round robin over the grid in ascending
//     order; thread 0 spins with ld.acquire.gpu until EVERY producer's counter for that block equals rows x columns
//     (all of the block's rows visible), issues a gpu-scope fence, then a CTA barrier admits the block's threads. The
//     block's elements are computed by the phase's unchanged per-element body.
//   * Consumer (a claimed segmented right product, ReadyT<false, true>): before issuing a task's first operand stage,
//     the issuing thread waits for the task's A rows (ReadyT::wait_a).
// Counters are per-launch words past the planned arena, zeroed at kernel entry (the existing 'ectaq' area: ordered
// before any use by the first grid barrier); a counter exceeding its target traps (double publication), and a wait
// that makes no progress for ~1 s traps (fail-stop, never a silent hang). Waiting CTAs never hold producer work: a
// producer's items are all handed out before any CTA leaves its item loop, and every holder completes its item
// without waiting, so all resident CTAs cannot wait on each other.
#pragma once

namespace latch_tr {

__shared__ volatile int tr_ring[1];

// Optional diagnostic stamps (trace pointers; generated programs pass nullptr): globaltimer samples.
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

// Readiness functor bound to a claimed segmented product (latch_ragged::tclaim_impl / latch_pack scatter body):
//   cnt != nullptr: publish each stored tile (tile_tag, via latch_ragged::tile_consumers);
//   wcnt != nullptr: before issuing a task's first operand stage, the issuing thread waits until the task's A rows
//   [r0, r0 + rows) are complete in the producer's block counters (wcols elements per row), then fences generic ->
//   async proxy so the TMA loads of this thread observe the other CTAs' generic stores.
template<bool PUB, bool WAIT>
struct ReadyT {
  static constexpr bool publishes = PUB, waits = WAIT;   // unused paths are compiled out (if constexpr)
  using tile_tag = void;
  using wait_tag = void;
  unsigned* cnt;          // one word per RB-row block of the product's [rows, n] results (PUB)
  int rb;                 // rows per block (publication and waits)
  unsigned long long* trace = nullptr;   // optional: [blocks] last-publication stamp
  int delay_ns = 0;       // optional delay of every publication (values unchanged; generated programs pass 0)
  const unsigned* wcnt = nullptr;        // producer block counters of the A operand (WAIT)
  unsigned wcols = 0u;                   // A elements per row (K)
  int wrb = 128;                         // rows per block of the A producer's counters
  unsigned long long* wtrace = nullptr;  // optional: [A blocks] earliest task admission, stored as ~globaltimer
  int policy = 0;                        // waiting scatter products: bit 0 descending task order, bit 1 ready-first
  __device__ __forceinline__ bool probe_a(long long r0, int rows, long long total_rows) const {
    if (rows <= 0) return true;
    const long long r1 = r0 + rows;
#pragma unroll 1
    for (long long j = r0 / wrb; j * wrb < r1; ++j) {
      const long long hi = (j + 1) * wrb < total_rows ? (j + 1) * wrb : total_rows;
      unsigned v;
      asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(wcnt + j) : "memory");
      if (v != (unsigned)((hi - j * wrb) * (long long)wcols)) return false;
    }
    return true;
  }
  // One thread, after a CTA- or warpgroup-scope barrier that follows the counted stores.
  __device__ __forceinline__ void publish_rows(long long r0, int rows, int cols) const {
    if (delay_ns) __nanosleep((unsigned)delay_ns);
    __threadfence();                                         // release (cumulative over the preceding barrier)
    const long long r1 = r0 + rows;
#pragma unroll 1
    for (long long j = r0 / rb; j * rb < r1; ++j) {
      const long long lo = r0 > j * rb ? r0 : j * rb;
      const long long hi = r1 < (j + 1) * rb ? r1 : (j + 1) * rb;
      atomicAdd(cnt + j, (unsigned)((hi - lo) * cols));
      if (trace) trace[j] = now_ns();
    }
  }
  // CTA-wide variant for bodies that publish through latch_ragged::tile_consumers.
  __device__ __forceinline__ void tile_done(long long r0, int rows, int cols) const {
    __syncthreads();
    if (threadIdx.x == 0) publish_rows(r0, rows, cols);
  }
  __device__ __forceinline__ void wait_a(long long r0, int rows, long long total_rows) const;   // below
};

// Thread 0 only: wait until cnt[j] == need (all of the block's producer elements published).
__device__ __forceinline__ void wait_rows(const unsigned* cnt, long long j, unsigned need) {
  if (need == 0u) return;
  unsigned spins = 0u;
#pragma unroll 1
  while (true) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cnt + j) : "memory");
    if (v == need) break;
    if (v > need) __trap();                                  // over-publication: a producer/consumer contract bug
    __nanosleep(64);
    if (++spins > (1u << 24)) __trap();                      // no progress for ~1 s: fail-stop
  }
}

// Issuing thread only (no CTA barrier: the other threads are inside the product's pipeline).
template<bool PUB, bool WAIT>
__device__ __forceinline__ void ReadyT<PUB, WAIT>::wait_a(long long r0, int rows, long long total_rows) const {
  if (rows <= 0) return;
  const long long r1 = r0 + rows;
#pragma unroll 1
  for (long long j = r0 / wrb; j * wrb < r1; ++j) {
    const long long hi = (j + 1) * wrb < total_rows ? (j + 1) * wrb : total_rows;
    wait_rows(wcnt, j, (unsigned)((hi - j * wrb) * (long long)wcols));
  }
  // The acquire loads order this thread after the producers' releases; the proxy fence orders its following TMA
  // (async-proxy) reads after them. No gpu-scope fence is needed on the issuing thread.
  asm volatile("fence.proxy.async.global;" ::: "memory");   // generic-proxy data -> this thread's TMA reads
  if (wtrace) atomicMax(wtrace + r0 / wrb, ~now_ns());       // optional stamp: earliest admission per A block
}

__device__ __forceinline__ void elided_barrier() { __syncthreads(); }

}  // namespace latch_tr
