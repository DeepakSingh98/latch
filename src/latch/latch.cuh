// The device side of a generated training kernel.
//
// A generated program is one kernel, launched once per training step with
// cooperative groups so that every thread block is resident at the same time.
// Its phases are separated by grid-wide barriers. The helpers below are the
// items that need more than a flat loop: a tiled matrix product, a reduction
// by a whole block, and an atomic accumulation.
//
// Without CUDA the same source compiles as host C++: every CUDA thread becomes
// an OS thread and the barriers are real barriers. This is slow, and it exists
// so the compiler can be tested on a machine without a GPU.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#define LATCH_FN __device__ __forceinline__
#define LATCH_KERNEL extern "C" __global__
#define LATCH_SIDE 16
#define LATCH_SHARED(name, bytes) __shared__ __align__(16) char name[bytes]
#define LATCH_GRID_SYNC() cooperative_groups::this_grid().sync()
#else
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#define LATCH_FN static inline
#define LATCH_KERNEL static
#define LATCH_SIDE 4
#define LATCH_SHARED(name, bytes) char* const name = latch::host::shared
#define LATCH_GRID_SYNC() latch::host::grid->wait()
#define __syncthreads() latch::host::block->wait()
#endif

namespace latch {

constexpr int SIDE = LATCH_SIDE;      // a block's threads form a SIDE x SIDE grid over a matrix tile
constexpr int THREADS = SIDE * SIDE;  // threads per block
constexpr int KC = 64;                // the most matrix-product depth staged in shared memory at a time

// Bytes of shared memory for a kernel whose largest matrix tile is TILE = TM + TN (see gemm).
template <int TILE>
struct Shared {
  static constexpr int bytes = SIDE * TILE * KC * 4 > THREADS * 8 ? SIDE * TILE * KC * 4 : THREADS * 8;
};

#ifndef __CUDACC__
namespace host {
struct Barrier {
  explicit Barrier(int count) : count(count) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    const long round = rounds;
    if (++arrived == count) { arrived = 0; ++rounds; released.notify_all(); }
    else released.wait(lock, [&] { return rounds != round; });
  }
  std::mutex mutex;
  std::condition_variable released;
  int count, arrived = 0;
  long rounds = 0;
};
struct Index { unsigned x; };
inline thread_local Index threadIdx, blockIdx;
inline Index gridDim;
inline thread_local Barrier* block;
inline Barrier* grid;
inline thread_local char* shared;

template <class Kernel>
void launch(Kernel kernel, int blocks, int shared_size) {
  Barrier everyone(blocks * THREADS);
  grid = &everyone;
  gridDim.x = blocks;
  std::vector<std::unique_ptr<Barrier>> barriers;
  std::vector<std::vector<long long>> memory(blocks, std::vector<long long>(shared_size / 8 + 1));
  for (int b = 0; b < blocks; ++b) barriers.emplace_back(new Barrier(THREADS));
  std::vector<std::thread> threads;
  for (int b = 0; b < blocks; ++b)
    for (int t = 0; t < THREADS; ++t)
      threads.emplace_back([&kernel, b, t, barrier = barriers[b].get(), smem = memory[b].data()] {
        blockIdx.x = b; threadIdx.x = t;
        block = barrier;
        shared = reinterpret_cast<char*>(smem);
        kernel();
      });
  for (auto& thread : threads) thread.join();
}
}  // namespace host
}  // namespace latch
using latch::host::blockIdx;  // the names CUDA provides
using latch::host::gridDim;
using latch::host::threadIdx;
namespace latch {
#endif

// Combine one value per thread across the block; every thread gets the result.
// The order of combination is fixed, so the result does not depend on timing.
template <class T, class Combine>
LATCH_FN T block_reduce(T value, char* shared, Combine combine) {
  T* slot = reinterpret_cast<T*>(shared);
  const int t = threadIdx.x;
  slot[t] = value;
  __syncthreads();
  for (int width = THREADS / 2; width > 0; width /= 2) {
    if (t < width) slot[t] = combine(slot[t], slot[t + width]);
    __syncthreads();
  }
  value = slot[0];
  __syncthreads();
  return value;
}

// C[b][m][n] = sum over k of A[b][m][k] * B[b][k][n]: one fused multiply-add chain per element, k ascending.
// A and B are strided (in elements); C is dense. The output is cut into tiles of
// (SIDE * TM) x (SIDE * TN); a block takes every gridDim-th tile, stages DEPTH columns
// of A and DEPTH rows of B in shared memory, and each thread accumulates a TM x TN patch.
// DEPTH is at most KC. Positions past the edge of a matrix are staged as zeros.
template <int TM, int TN, int DEPTH>
LATCH_FN void gemm(const float* A, long long sab, long long sam, long long sak,
                   const float* B, long long sbb, long long sbk, long long sbn,
                   float* C, int batch, int M, int K, int N, char* shared) {
  constexpr int BM = SIDE * TM, BN = SIDE * TN;
  float* As = reinterpret_cast<float*>(shared);  // [BM][DEPTH]
  float* Bs = As + BM * DEPTH;                   // [DEPTH][BN]
  const int t = threadIdx.x, ty = t / SIDE, tx = t % SIDE;
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  for (int task = blockIdx.x; task < batch * tiles_m * tiles_n; task += gridDim.x) {
    const int b = task / (tiles_m * tiles_n);
    const int m0 = task / tiles_n % tiles_m * BM, n0 = task % tiles_n * BN;
    const float* Ab = A + b * sab;
    const float* Bb = B + b * sbb;
    float acc[TM][TN];
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += DEPTH) {
      for (int e = t; e < BM * DEPTH; e += THREADS) {
        const int m = m0 + e / DEPTH, k = k0 + e % DEPTH;
        As[e] = m < M && k < K ? Ab[m * sam + k * sak] : 0.0f;
      }
      for (int e = t; e < DEPTH * BN; e += THREADS) {
        const int k = k0 + e / BN, n = n0 + e % BN;
        Bs[e] = k < K && n < N ? Bb[k * sbk + n * sbn] : 0.0f;
      }
      __syncthreads();
      for (int k = 0; k < DEPTH; ++k)
        for (int i = 0; i < TM; ++i) {
          const float a = As[(ty * TM + i) * DEPTH + k];
          for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a, Bs[k * BN + tx * TN + j], acc[i][j]);
        }
      __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j) {
        const int m = m0 + ty * TM + i, n = n0 + tx * TN + j;
        if (m < M && n < N) C[((long long)b * M + m) * N + n] = acc[i][j];
      }
  }
}

// *target += value, safe against other threads adding to the same element.
LATCH_FN void atomic_add(float* target, float value) {
#ifdef __CUDACC__
  atomicAdd(target, value);
#else
  uint32_t* bits = reinterpret_cast<uint32_t*>(target);
  uint32_t seen = __atomic_load_n(bits, __ATOMIC_RELAXED), next;
  do {
    float sum;
    std::memcpy(&sum, &seen, 4);
    sum += value;
    std::memcpy(&next, &sum, 4);
  } while (!__atomic_compare_exchange_n(bits, &seen, next, true, __ATOMIC_SEQ_CST, __ATOMIC_RELAXED));
#endif
}

}  // namespace latch

// What the Python runtime calls, for a kernel `void kernel(void** args, char* arena)`.
// latch_blocks(0) is the default grid, one block per multiprocessor; latch_blocks(1) is the most that can be
// resident at once, which a cooperative launch may not exceed.
#ifdef __CUDACC__
#define LATCH_EXPORTS(kernel)                                                                              \
  extern "C" int latch_blocks(int most) {                                                                  \
    int device, per_sm;                                                                                    \
    cudaDeviceProp properties;                                                                             \
    if (cudaGetDevice(&device) || cudaGetDeviceProperties(&properties, device)) return -1;                 \
    if (!properties.cooperativeLaunch) return -1;                                                          \
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, latch::THREADS, 0)) return -1;      \
    return (most ? per_sm : per_sm > 0) * properties.multiProcessorCount;                                  \
  }                                                                                                        \
  extern "C" int latch_launch(void** args, char* arena, int blocks, void* stream) {                        \
    void* arguments[] = {&args, &arena};                                                                   \
    return cudaLaunchCooperativeKernel((const void*)kernel, dim3(blocks), dim3(latch::THREADS), arguments, \
                                       0, (cudaStream_t)stream);                                           \
  }                                                                                                        \
  extern "C" const char* latch_error(int code) { return cudaGetErrorString((cudaError_t)code); }
#else
#define LATCH_EXPORTS(kernel)                                                         \
  extern "C" int latch_blocks(int most) { return most ? 8 : 2; }                      \
  extern "C" int latch_launch(void** args, char* arena, int blocks, void*) {          \
    latch::host::launch([=] { kernel(args, arena); }, blocks, LATCH_SHARED_BYTES);    \
    return 0;                                                                         \
  }                                                                                   \
  extern "C" const char* latch_error(int) { return "host emulation does not fail"; }
#endif
