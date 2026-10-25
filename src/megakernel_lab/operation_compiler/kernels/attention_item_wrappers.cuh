// Explicit completed-item boundary; the original v7 body and its waits stay unchanged.
#pragma once
#include "attention_long_flat.cuh"
namespace latch_attention_v7 {
template <int NS, bool POW2>
static __device__ __noinline__ void forward_item(const pflatlong::Args& a, int bh, int tile, void* smem) {
  pflatlong::forward_item6<NS, POW2>(a, bh, tile, smem);
}
template <int NS, bool POW2>
static __device__ __noinline__ void dq_item(const pflatlong::Args& a, int bh, int tile, void* smem) {
  pflatlong::dq_item6<NS, POW2>(a, bh, tile, smem);
}
template <int NS, bool POW2>
static __device__ __noinline__ void kv_item(const pflatlong::Args& a, int bh, int tile, void* smem) {
  pflatlong::kv_item6<NS, POW2>(a, bh, tile, smem);
}
}  // namespace latch_attention_v7
