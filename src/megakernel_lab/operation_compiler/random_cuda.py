"""Torch-free CUDA helpers for the explicit Philox ABI in ops.random.

Independent arithmetic implementation of Philox4x32-10; algorithm/constants:
https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h
No Random123 source/header is vendored. Caller supplies integer state storage and
commits the four next-state words only after every use of the old state.
"""

CUDA_HELPERS = r'''
#ifndef LATCH_OPS_PHILOX_HELPERS
#define LATCH_OPS_PHILOX_HELPERS
// Philox4x32-10; state[0:2] key words, state[2:4] 64-bit block counter.
// Each int64 word is interpreted modulo 2^32. No signed-overflow arithmetic.
__device__ __forceinline__ uint32_t latch_philox_word(
    const int64_t* state, uint64_t element_index) {
  const uint64_t counter = ((uint64_t)(uint32_t)state[3] << 32)
                           | (uint64_t)(uint32_t)state[2];
  const uint64_t block = counter + element_index / 4ULL;
  uint32_t a = (uint32_t)block, b = (uint32_t)(block >> 32), c = 0, d = 0;
  uint32_t key0 = (uint32_t)state[0], key1 = (uint32_t)state[1];
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint64_t product0 = 0xD2511F53ULL * (uint64_t)a;
    const uint64_t product1 = 0xCD9E8D57ULL * (uint64_t)c;
    const uint32_t next_a = (uint32_t)(product1 >> 32) ^ b ^ key0;
    const uint32_t next_c = (uint32_t)(product0 >> 32) ^ d ^ key1;
    b = (uint32_t)product1;
    d = (uint32_t)product0;
    a = next_a;
    c = next_c;
    key0 += 0x9E3779B9U;
    key1 += 0xBB67AE85U;
  }
  switch ((unsigned int)(element_index & 3ULL)) {
    case 0: return a;
    case 1: return b;
    case 2: return c;
    default: return d;
  }
}

__device__ __forceinline__ float latch_philox_mask(
    const int64_t* state, uint64_t element_index, uint64_t threshold, float scale) {
  return (uint64_t)latch_philox_word(state, element_index) >= threshold ? scale : 0.0f;
}

// P8-B: the four words of one Philox block (identical arithmetic to latch_philox_word), and a lane-strided row
// consumer: lane L of a warp computed block (row_block0 + 32 m + L) into q[m]; element 32 w + lane of the row takes
// word (lane & 3) of block 8 w + (lane >> 2), i.e. q[w >> 2] of lane 8 (w & 3) + (lane >> 2) (bit-identical masks).
__device__ __forceinline__ void latch_philox_block(const int64_t* state, uint64_t block_index, uint32_t (&w)[4]) {
  const uint64_t counter = ((uint64_t)(uint32_t)state[3] << 32)
                           | (uint64_t)(uint32_t)state[2];
  const uint64_t block = counter + block_index;
  uint32_t a = (uint32_t)block, b = (uint32_t)(block >> 32), c = 0, d = 0;
  uint32_t key0 = (uint32_t)state[0], key1 = (uint32_t)state[1];
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint64_t product0 = 0xD2511F53ULL * (uint64_t)a;
    const uint64_t product1 = 0xCD9E8D57ULL * (uint64_t)c;
    const uint32_t next_a = (uint32_t)(product1 >> 32) ^ b ^ key0;
    const uint32_t next_c = (uint32_t)(product0 >> 32) ^ d ^ key1;
    b = (uint32_t)product1;
    d = (uint32_t)product0;
    a = next_a;
    c = next_c;
    key0 += 0x9E3779B9U;
    key1 += 0xBB67AE85U;
  }
  w[0] = a; w[1] = b; w[2] = c; w[3] = d;
}

__device__ __forceinline__ float latch_philox_mask_pre(const uint32_t (&q)[4], int src, int word,
                                                       uint64_t threshold, float scale) {
  const uint32_t a = __shfl_sync(0xffffffffu, q[0], src), b = __shfl_sync(0xffffffffu, q[1], src);
  const uint32_t c = __shfl_sync(0xffffffffu, q[2], src), d = __shfl_sync(0xffffffffu, q[3], src);
  const uint32_t x = word == 0 ? a : (word == 1 ? b : (word == 2 ? c : d));
  return (uint64_t)x >= threshold ? scale : 0.0f;
}

__device__ __forceinline__ int64_t latch_philox_advance_word(
    const int64_t* state, uint64_t blocks, int word) {
  if (word < 2) return (int64_t)(uint32_t)state[word];
  const uint64_t counter = ((uint64_t)(uint32_t)state[3] << 32)
                           | (uint64_t)(uint32_t)state[2];
  const uint64_t next = counter + blocks;
  return word == 2 ? (int64_t)(uint32_t)next : (int64_t)(uint32_t)(next >> 32);
}
#endif
'''
