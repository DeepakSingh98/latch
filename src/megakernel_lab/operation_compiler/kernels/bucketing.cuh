// Routing preparation kernels (segmented_products feature "route"): route_order counts / stable order on every CTA;
// same integers. Its own header (only ragged_experts.cuh): including it never pulls the claimed product bodies
// (ragged_claim.cuh) ahead of a program's LATCH_CLAIM_SEPI definition.
#pragma once
#include <cstdint>
#include "ragged_experts.cuh"

namespace latch_grp {

// ------------------------------------------------------------------------------------------------------------------
// Routing preparation: the stable bucket order of latch_ragged::counts / order with every CTA working. Outputs are the
// same integers: offsets = exclusive prefix of exact bucket counts; perm lists each bucket's rows in ascending original
// order at the bucket's offset; inverse is perm's inverse. Out-of-range ids trap.
//   route_counts: each CTA counts a contiguous row chunk in shared memory and adds its counts to entry-zeroed global
//     words (integer adds: order-independent); the last CTA to arrive writes the prefix offsets.
//   route_order: each CTA counts its chunk again and publishes the counts in its own row of a G x E table held in the
//     perm output (which no other value occupies during this phase); after every CTA has published (all CTAs are
//     resident: full grid), each CTA sums its predecessors' rows for its bucket bases; after every CTA has read the
//     table, rows are placed in ascending order (256-row rounds; warp match ranks; per-warp counts) and the table is
//     overwritten by the placement. Grids with G x E > rows use an inline copy of latch_ragged::order_scan.
__device__ __forceinline__ void grid_arrive_wait(unsigned* ctr) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(ctr, 1u);
    while (atomicAdd(ctr, 0u) < gridDim.x) __nanosleep(32);
    __threadfence();
  }
  __syncthreads();
}

template<int E>
__device__ __forceinline__ void chunk_counts(const int64_t* ids, int lo, int hi, int* h) {
  __syncthreads();                                                 // earlier users of shared memory are done
  for (int e = threadIdx.x; e < E; e += blockDim.x) h[e] = 0;
  __syncthreads();
  for (int r = lo + (int)threadIdx.x; r < hi; r += blockDim.x) {
    const int64_t e = ids[r];
    if (e < 0 || e >= E) __trap();
    atomicAdd(&h[(int)e], 1);
  }
  __syncthreads();
}

template<int E>
DEV void route_counts(const int64_t* ids, int64_t* offsets, int rows, float* smem, unsigned* words) {
  static_assert(E >= 1 && E <= 256, "bucket count");
  int* h = reinterpret_cast<int*>(smem);
  __shared__ int grp_last;
  const int G = (int)gridDim.x, R = (rows + G - 1) / G;
  const int lo = min(rows, (int)blockIdx.x * R), hi = min(rows, lo + R);
  chunk_counts<E>(ids, lo, hi, h);
  for (int e = threadIdx.x; e < E; e += blockDim.x)
    if (h[e]) atomicAdd(&words[e], (unsigned)h[e]);
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    grp_last = atomicAdd(&words[E], 1u) == (unsigned)(G - 1);
  }
  __syncthreads();
  if (grp_last) {
    __threadfence();
    if (threadIdx.x < 32) {
      long long carry = 0;
      if (threadIdx.x == 0) offsets[0] = 0;
      for (int b = 0; b < E; b += 32) {
        long long v = b + (int)threadIdx.x < E ? (long long)__ldcg(&words[b + threadIdx.x]) : 0LL;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
          const long long n = __shfl_up_sync(0xffffffffu, v, d);
          if ((int)threadIdx.x >= d) v += n;
        }
        if (b + (int)threadIdx.x < E) offsets[b + threadIdx.x + 1] = carry + v;
        carry += __shfl_sync(0xffffffffu, v, 31);
      }
    }
  }
  __syncthreads();
}

template<int E>
DEV void route_order(const int64_t* ids, const int64_t* offsets, int64_t* perm, int64_t* inverse, int rows,
                     float* smem, unsigned* words) {
  static_assert(E >= 1 && E <= 256, "bucket count");
  const int G = (int)gridDim.x;
  if ((long long)G * E > rows) {                                   // CTA-uniform: no room for the table
    // Inline copy of latch_ragged::order_scan (one pass per bucket; same integers). No call: a call in a function
    // makes ptxas serialize every warpgroup MMA of that function (C7509).
    int* warp_totals = reinterpret_cast<int*>(smem);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    __syncthreads();
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
    return;
  }
  int* h = reinterpret_cast<int*>(smem);                           // [E] chunk counts, then bucket bases
  int* wc = h + 256;                                               // [8][E] per-warp counts -> exclusive prefixes
  int* tot = wc + 8 * 256;                                         // [E] round totals
  const int R = (rows + G - 1) / G, c = (int)blockIdx.x;
  const int lo = min(rows, c * R), hi = min(rows, lo + R);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  if (blockDim.x != 256) __trap();
  chunk_counts<E>(ids, lo, hi, h);
  for (int e = t; e < E; e += 256) perm[(long long)c * E + e] = h[e];
  grid_arrive_wait(&words[0]);                                     // every CTA's counts are published
  for (int e = t; e < E; e += 256) {
    long long base = offsets[e];
#pragma unroll 8
    for (int p = 0; p < c; ++p) base += __ldcg(reinterpret_cast<const long long*>(perm) + (long long)p * E + e);
    h[e] = (int)base;
  }
  grid_arrive_wait(&words[1]);                                     // every CTA has read the table
  for (int i = t; i < 8 * 256; i += 256) wc[i] = 0;
  __syncthreads();
  const unsigned lt = (1u << lane) - 1u;
  for (int s = lo; s < hi; s += 256) {
    const int r = s + t;
    const bool valid = r < hi;
    const int e = valid ? (int)ids[r] : -1;                        // range checked by chunk_counts
    const unsigned peers = __match_any_sync(0xffffffffu, e);
    const int rank = __popc(peers & lt);
    if (valid && rank == 0) wc[warp * 256 + e] = __popc(peers);
    __syncthreads();
    for (int b = t; b < E; b += 256) {
      int run = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) { const int x = wc[w * 256 + b]; wc[w * 256 + b] = run; run += x; }
      tot[b] = run;
    }
    __syncthreads();
    if (valid) {
      const int pos = h[e] + wc[warp * 256 + e] + rank;
      perm[pos] = r;
      inverse[r] = pos;
    }
    __syncthreads();
    for (int b = t; b < E; b += 256) {
      h[b] += tot[b];
#pragma unroll
      for (int w = 0; w < 8; ++w) wc[w * 256 + b] = 0;
    }
    __syncthreads();
  }
}
}  // namespace latch_grp
