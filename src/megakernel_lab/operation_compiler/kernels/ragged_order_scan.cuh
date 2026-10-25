// LATCH-SELECTIVE-10 (ops.accumulation_rules ``rorder``): the stable bucket permutation of latch_ragged::order<E>
// in one pass per bucket. Thread t owns rows [t*span, (t+1)*span) (span = ceil(rows / 256)); a block exclusive
// scan of the per-thread match counts gives each thread the packed position of its first match, and its matches
// follow in ascending row order. The packed order (ascending original row within each bucket, buckets at their
// offsets) is unique, so perm and inverse are the integers order<E> writes.
#pragma once
#include <cstdint>

namespace latch_ragged {
template<int E>
__device__ __noinline__ void order_scan(const int64_t* ids, const int64_t* offsets,
                                       int64_t* perm, int64_t* inverse, int rows, float* smem) {
  int* warp_totals = reinterpret_cast<int*>(smem);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (blockDim.x != 256) __trap();
  const int span = (rows + 255) / 256;
  const int lo = t * span < rows ? t * span : rows;
  const int hi = lo + span < rows ? lo + span : rows;
  for (int e = blockIdx.x; e < E; e += gridDim.x) {
    int mine = 0;
    for (int row = lo; row < hi; ++row) mine += ids[row] == e;
    int inclusive = mine;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const int v = __shfl_up_sync(0xffffffffu, inclusive, d);
      if (lane >= d) inclusive += v;
    }
    if (lane == 31) warp_totals[warp] = inclusive;
    __syncthreads();
    int before = 0;
    for (int w = 0; w < warp; ++w) before += warp_totals[w];
    int packed = (int)offsets[e] + before + inclusive - mine;
    for (int row = lo; row < hi; ++row) {
      if (ids[row] == e) {
        perm[packed] = row;
        inverse[row] = packed;
        ++packed;
      }
    }
    __syncthreads();
  }
}
}  // namespace latch_ragged
