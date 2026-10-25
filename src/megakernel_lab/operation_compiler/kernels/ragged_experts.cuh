// Generic stable bucket permutation and segmented products. Matrix arithmetic
// reuses the existing P-OPS4-B WGMMA tile, including its edge masks and rounding.
// No capacity is assigned per bucket: the packed row count is exactly tokens*K.
#pragma once
#include <type_traits>
#include "pops4b_wg.cuh"

namespace latch_ragged {

template<int E>
__device__ __noinline__ void counts(const int64_t* ids, int64_t* offsets, int rows, float* smem) {
  // LATCH-TINYMOE-25: all 256 threads read disjoint rows and add exact integer counts in shared
  // memory (integer addition is associative, so every count and offset is bitwise the same as the
  // former 16-thread serial scan, without its one-dependent-load-per-row latency chain).
  static_assert(E >= 1 && E <= 256, "bucket count must fit the 256-int shared table");
  if (blockIdx.x != 0) return;
  int* counts_ = reinterpret_cast<int*>(smem);
  const int t = threadIdx.x;
  counts_[t] = 0;
  __syncthreads();
  for (int r = t; r < rows; r += blockDim.x) {
    const int64_t e = ids[r];
    if (e < 0 || e >= E) __trap();
    atomicAdd(&counts_[(int)e], 1);
  }
  __syncthreads();
  if (t == 0) {
    int64_t total = 0;
    offsets[0] = 0;
    for (int e = 0; e < E; ++e) { total += counts_[e]; offsets[e + 1] = total; }
  }
  __syncthreads();
}

template<int E>
__device__ __noinline__ void order(const int64_t* ids, const int64_t* offsets,
                                  int64_t* perm, int64_t* inverse, int rows, float* smem) {
  int* scan = reinterpret_cast<int*>(smem);
  const int t = threadIdx.x;
  for (int e = blockIdx.x; e < E; e += gridDim.x) {
    int base = (int)offsets[e];
    for (int chunk = 0; chunk < rows; chunk += 256) {
      const int row = chunk + t;
      const int match = row < rows && ids[row] == e;
      scan[t] = match;
      __syncthreads();
      // Integer inclusive scan gives ascending original-choice ordering.
      #pragma unroll
      for (int d = 1; d < 256; d *= 2) {
        const int value = t >= d ? scan[t - d] : 0;
        __syncthreads();
        scan[t] += value;
        __syncthreads();
      }
      if (match) {
        const int packed = base + scan[t] - 1;
        perm[packed] = row;
        inverse[row] = packed;
      }
      base += scan[255];
      __syncthreads();
    }
  }
}

// LATCH-TINYMOE-XMAT: L2 prefetch hint of one operand's storage span (16-byte aligned, size a multiple of 16), spread
// over the grid in 32 KiB pieces issued by each CTA's thread 0. Nothing is read into the SM or written.
__device__ __forceinline__ void l2_prefetch(const void* p, long long bytes) {
  if (threadIdx.x != 0) return;
  const char* base = static_cast<const char*>(p);
  const long long n = (bytes + 32767) / 32768;
  for (long long c = blockIdx.x; c < n; c += gridDim.x) {
    const long long o = c * 32768;
    const unsigned size = (unsigned)(bytes - o < 32768 ? bytes - o : 32768);
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(base + o), "r"(size) : "memory");
  }
}

// LATCH-TINYMOE-XMAT "rotate": task count of a flat product (as its body enumerates them) and the rotation that
// starts the next product's round-robin at the CTA after the last task of products totalling `tasks`.
template<int E, int BM, int BN, bool OUTER>
__device__ __forceinline__ int tasks(const int64_t* offsets, int m, int n) {
  const int tn = (n + BN - 1) / BN;
  int total = 0;
#pragma unroll 1
  for (int e = 0; e < E; ++e) total += ((OUTER ? m : (int)(offsets[e + 1] - offsets[e])) + BM - 1) / BM * tn;
  return total;
}
__device__ __forceinline__ int rot_after(int tasks) {
  const int g = (int)gridDim.x;
  return (g - tasks % g) % g;
}

// Runtime segment checks are CTA-uniform; an invalid external ABI traps before
// reading an operand. The compiler-created routing offsets always satisfy them.
__device__ __forceinline__ void bounds(const int64_t* offsets, int e, int rows, int& lo, int& hi) {
  const int64_t l = offsets[e], h = offsets[e + 1];
  if (l < 0 || h < l || h > rows) __trap();
  lo = (int)l; hi = (int)h;
}

// WIDE (outer products only, LATCH-TINYMOE-EGRAD): g.C is FP32 storage receiving the exactly widened
// BF16-rounded results, i.e. the bits a separate BF16 -> FP32 cast of the BF16 product would write.
template<int E, int BM, int BN, int WM, bool AK, bool BKC, int NST, bool OUTER, bool WIDE = false>
__device__ __noinline__ void product(popsc::BArgs g, const int64_t* offsets, int rows, float* smem) {
  static_assert(!WIDE || OUTER, "widened storage is admitted for segmented outer products");
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  const int tn = (g.N + BN - 1) / BN;
  int total = 0;
  for (int e = 0; e < E; ++e) {
    int lo, hi; bounds(offsets, e, rows, lo, hi);
    total += ((OUTER ? g.M : hi - lo) + BM - 1) / BM * tn;
  }
  __syncthreads();
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int local = task, e = 0, lo = 0, hi = 0;
    for (; e < E; ++e) {
      bounds(offsets, e, rows, lo, hi);
      const int count = ((OUTER ? g.M : hi - lo) + BM - 1) / BM * tn;
      if (local < count) break;
      local -= count;
    }
    popsc::BArgs seg = g;
    if (OUTER) {
      seg.A += (long long)lo * g.sak;
      seg.B += (long long)lo * g.sbk;
      seg.C = WIDE ? reinterpret_cast<s101h::bf16*>(reinterpret_cast<float*>(g.C) + (long long)e * g.bsc)
                   : g.C + (long long)e * g.bsc;
      seg.K = hi - lo;
    } else {
      seg.A += (long long)lo * g.sam;
      seg.B += (long long)e * g.bsb;
      seg.C += (long long)lo * g.ldc;
      seg.M = hi - lo;
    }
    // Always masked: segment boundaries may occur at any row and K can be 0.
    pops4b::tile<BM, BN, WM, AK, BKC, NST, true, true, 0, WIDE>(seg, 0,
        (local / tn) * BM, (local % tn) * BN, base);
  }
}

// LATCH-TINYMOE-XMAT: flattened segmented products. The CTA's (segment, tile) tasks and their 64-wide k stages form
// one sequence: operand stages are issued NST - 1 steps ahead across task boundaries (the next tile's operands
// arrive while this tile's last MMAs and its epilogue run), each task keeps the per-tile inner k loop with one wgmma
// group in flight and an unconditional drain before its register-direct store (no conditional wait/epilogue inside
// a flattened loop, which ptxas serializes, C7518). The segment table (task prefix and row offsets) is built once per
// call in shared memory after the operand ring. Task enumeration, staging (cp.async with zero-filled absent rows/k)
// and per-element arithmetic are those of `product`: BF16 x BF16 wgmma, FP32 accumulation over ascending k16 steps
// starting with scale-d = 0 within each task, +0 for absent k terms, one cvt.rn.bf16x2 rounding, masked pair stores;
// an empty OUTER segment stores +0. Results are therefore bitwise those of `product` for every tile shape
// (BM, BN, WM) and stage count. ROT rotates the CTA's position in the round-robin task assignment (balance between
// consecutive phases), never the tasks' arithmetic.
// P = 2 (flat2): two products of one interval with the same A operand, segment offsets and extents (e.g. the gate
// and up projections of an expert, or their weight gradients) share each A stage; every output element keeps the
// arithmetic of its own single product.
// Tile-local consumers (LATCH-TINYMOE-XMAT): an identity-indexed elementwise consumer of a right product runs in the
// CTA that stored the tile, right after the store (CTA barrier: the tile's global stores are visible to the CTA),
// over the tile's rows (one warp per row, lanes over columns) with its unchanged per-element body epi(i), i the flat
// index of the [rows, N] result. NoEpi: no consumer.
struct NoEpi {};
template<class EPI>
__device__ __forceinline__ void tile_consumers(const EPI& epi, int row0, int m0, int n0, int M, int N, long long ldc,
                                               int BM, int BN) {
  if constexpr (!std::is_same<EPI, NoEpi>::value) {
    // Uniform trip counts and no thread-dependent branch (a divergent path or a call would make ptxas serialize the
    // product's warpgroup MMA): lanes past the tile's last row/column repeat that in-tile element, an idempotent
    // recomputation (the generator admits only consumers that never read their own results from memory).
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll 1
    for (int rr = warp; rr < BM; rr += 8) {
      const int r = m0 + rr < M ? m0 + rr : M - 1;
#pragma unroll 1
      for (int cc = lane; cc < BN; cc += 32) {
        const int c = n0 + cc < N ? n0 + cc : N - 1;
        epi((long long)(row0 + r) * ldc + c);
      }
    }
  }
}

// Diagnostics only (microbenchmark builds; results wrong): 1 = no operand loads after the prologue, 2 = no wgmma.
#ifndef LATCH_RAGGED_DIAG
#define LATCH_RAGGED_DIAG 0
#endif
template<int E, int BM, int BN, int WM, int NST, int P = 1>
struct FlatCfg {
  using C = pops4b::Cfg<BM, BN, WM>;
  static constexpr int STAGE = C::A_BYTES + P * C::B_BYTES;
  static constexpr int SHARED = NST * STAGE + 1024 + 8 * (E + 2);
};

template<int E, int BM, int BN, int WM, bool AK, bool BKC, int NST, bool OUTER, int P, class EPI = NoEpi>
DEV void flat_impl(const popsc::BArgs* gs, const int64_t* offsets, int rows, float* smem, int rot,
                   const EPI& epi = EPI{}) {
  static_assert(std::is_same<EPI, NoEpi>::value || !OUTER, "tile-local consumers follow right products");
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = FlatCfg<E, BM, BN, WM, NST, P>::STAGE;
  static_assert(NST >= 2 && NST <= 8 && (P == 1 || P == 2), "stages / products");
  const popsc::BArgs& g = gs[0];
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  int* pre = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE);
  int* off = pre + (E + 1);
  const int tid = threadIdx.x, tn = (g.N + BN - 1) / BN;
  __syncthreads();                                                 // earlier users of shared memory are done
  if (tid < E) {
    int lo, hi; bounds(offsets, tid, rows, lo, hi);
    off[tid] = lo;
    if (tid == E - 1) off[E] = hi;
    pre[tid + 1] = ((OUTER ? g.M : hi - lo) + BM - 1) / BM * tn;
  }
  __syncthreads();
  if (tid == 0) {
    int total = 0;
    pre[0] = 0;
#pragma unroll 1
    for (int e = 0; e < E; ++e) { total += pre[e + 1]; pre[e + 1] = total; }
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = pre[E];
  const int me = (int)(((unsigned)blockIdx.x + (unsigned)rot) % (unsigned)G);
  const int ntask = me < T ? (T - me + G - 1) / G : 0;
  if (ntask == 0) return;                                          // CTA-uniform
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const int kfix = (g.K + 63) / 64;
  struct Task { const s101h::bf16* A; const s101h::bf16* B[P]; s101h::bf16* Cp[P]; int lo, M, K, m0, n0, nk; };
  auto task = [&](int i, Task& t) {
    const int id = me + i * G;
    int e = 0;
#pragma unroll 1
    while (e < E - 1 && id >= pre[e + 1]) ++e;
    const int local = id - pre[e], lo = off[e], hi = off[e + 1];
    t.lo = lo;
    t.m0 = (local / tn) * BM;
    t.n0 = (local % tn) * BN;
    if (OUTER) {
      t.A = g.A + (long long)lo * g.sak;
#pragma unroll
      for (int p = 0; p < P; ++p) { t.B[p] = gs[p].B + (long long)lo * g.sbk; t.Cp[p] = gs[p].C + (long long)e * g.bsc; }
      t.M = g.M; t.K = hi - lo; t.nk = (t.K + 63) / 64;
    } else {
      t.A = g.A + (long long)lo * g.sam;
#pragma unroll
      for (int p = 0; p < P; ++p) { t.B[p] = gs[p].B + (long long)e * g.bsb; t.Cp[p] = gs[p].C + (long long)lo * g.ldc; }
      t.M = hi - lo; t.K = g.K; t.nk = kfix;
    }
  };
  // Issue cursor: task li and its step lk; stage slots follow the issued-step count lq (the consumer's q).
  Task lt;
  int li = 0, lk = 0, lq = 0;
  task(0, lt);
  auto issue = [&]() {
#pragma unroll 1
    while (li < ntask && lk >= lt.nk) { ++li; lk = 0; if (li < ntask) task(li, lt); }
    if (li < ntask && !(LATCH_RAGGED_DIAG == 1 && lq >= NST - 1)) {
      const int kk = lk * 64;
      const unsigned dA = base + (lq % NST) * STAGE;
      if (lt.m0 + BM <= lt.M && lt.n0 + BN <= g.N && kk + 64 <= lt.K) {
        pops4b::stage<BM, AK, false>(lt.A, g.sam, g.sak, lt.m0, kk, dA, lt.M, lt.K);
#pragma unroll
        for (int p = 0; p < P; ++p)
          pops4b::stage<BN, BKC, false>(lt.B[p], g.sbn, g.sbk, lt.n0, kk, dA + C::A_BYTES + p * C::B_BYTES, g.N, lt.K);
      } else {
        pops4b::stage<BM, AK, true>(lt.A, g.sam, g.sak, lt.m0, kk, dA, lt.M, lt.K);
#pragma unroll
        for (int p = 0; p < P; ++p)
          pops4b::stage<BN, BKC, true>(lt.B[p], g.sbn, g.sbk, lt.n0, kk, dA + C::A_BYTES + p * C::B_BYTES, g.N, lt.K);
      }
      ++lk; ++lq;
    }
    s101h::cpa_commit();
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) issue();
  float acc[P][C::MT][C::R];
  int q = 0;
#pragma unroll 1
  for (int i = 0; i < ntask; ++i) {
    Task t;
    task(i, t);
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int a = 0; a < C::MT; ++a)
#pragma unroll
        for (int j = 0; j < C::R; ++j) acc[p][a][j] = 0.f;         // only an empty OUTER segment stores these
#pragma unroll 1
    for (int kk = 0; kk < t.nk; ++kk, ++q) {
      s101h::cpa_wait<NST - 2>();
      pops4b::async_fence();                                       // cp.async (generic proxy) -> wgmma (async proxy)
      __syncthreads();
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      pops4b::wg_fence();
      const unsigned sA = base + (q % NST) * STAGE;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
#pragma unroll
        for (int p = 0; p < P; ++p) {
          const unsigned sB = sA + C::A_BYTES + p * C::B_BYTES;
          const uint64_t db = BKC ? pops4b::dk(sB + wgn * C::WTN * 128 + k16 * 32)
                                  : pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
          for (int a = 0; a < C::MT; ++a) {
            const int r0 = wgm * C::WTM + a * 64;
            const uint64_t da = AK ? pops4b::dk(sA + r0 * 128 + k16 * 32) : pops4b::dmn(sA + (r0 / 64) * 8192 + k16 * 2048);
            if (LATCH_RAGGED_DIAG != 2)
              s101h::MmeWg<C::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[p][a], da, db, (kk | k16) != 0);
          }
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      __syncthreads();                                             // stage (q - 1) % NST is free in both warpgroups
      issue();
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int a = 0; a < C::MT; ++a) {
        const int row = t.m0 + wgm * C::WTM + a * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < C::WTN / 8; ++j) {
          const int col = t.n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = row + 8 * h;
            if (r < t.M && col < g.N)
              *reinterpret_cast<uint32_t*>(t.Cp[p] + (long long)r * g.ldc + col) =
                  s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
          }
        }
      }
    tile_consumers(epi, t.lo, t.m0, t.n0, t.M, g.N, g.ldc, BM, BN);
  }
  s101h::cpa_wait<0>();
  __syncthreads();                                                 // every wgmma/cp.async of the phase is done
}

template<int E, int BM, int BN, int WM, bool AK, bool BKC, int NST, bool OUTER, class EPI = NoEpi>
DEV void flat(const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem, int rot, const EPI& epi = EPI{}) {
  flat_impl<E, BM, BN, WM, AK, BKC, NST, OUTER, 1, EPI>(&g, offsets, rows, smem, rot, epi);
}

// g1 must equal g0 except for B and C (same A view, strides, extents); the generator checks this statically.
template<int E, int BM, int BN, int WM, bool AK, bool BKC, int NST, bool OUTER, class EPI = NoEpi>
DEV void flat2(const popsc::BArgs& g0, const popsc::BArgs& g1, const int64_t* offsets, int rows, float* smem, int rot,
               const EPI& epi = EPI{}) {
  const popsc::BArgs gs[2] = {g0, g1};
  flat_impl<E, BM, BN, WM, AK, BKC, NST, OUTER, 2, EPI>(gs, offsets, rows, smem, rot, epi);
}

// FP32 and awkward BF16 shapes use a small shared-memory SIMT tile. This is
// independently useful as a generated reference and preserves all edge cases.
template<typename T> __device__ __forceinline__ float widen(T x) { return float(x); }
template<> __device__ __forceinline__ float widen<s101h::bf16>(s101h::bf16 x) { return __bfloat162float(x); }
template<typename T> __device__ __forceinline__ T round(float x) { return T(x); }
template<> __device__ __forceinline__ s101h::bf16 round<s101h::bf16>(float x) { return __float2bfloat16_rn(x); }
// Stored result: rounded to T; with O = float and T = bf16 that rounding is then exactly widened.
template<typename O, typename T> struct Store { static __device__ __forceinline__ O at(float x) { return widen(round<T>(x)); } };
template<typename T> struct Store<T, T> { static __device__ __forceinline__ T at(float x) { return round<T>(x); } };

template<typename T, int E, bool OUTER, typename O = T>
__device__ __noinline__ void simt(const T* a, const T* b, const int64_t* offsets, O* out,
    int rows, int m, int n, int k, long long sam, long long sak, long long sbk,
    long long sbn, long long bsb, long long bsc, float* smem) {
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const int tn = (n + 15) / 16;
  int total = 0;
  for (int e = 0; e < E; ++e) {
    int lo, hi; bounds(offsets, e, rows, lo, hi);
    total += ((OUTER ? m : hi - lo) + 15) / 16 * tn;
  }
  const int tr = threadIdx.x / 16, tc = threadIdx.x % 16;
  for (int task = blockIdx.x; task < total; task += gridDim.x) {
    int local = task, e = 0, lo = 0, hi = 0;
    for (; e < E; ++e) {
      bounds(offsets, e, rows, lo, hi);
      const int count = ((OUTER ? m : hi - lo) + 15) / 16 * tn;
      if (local < count) break;
      local -= count;
    }
    const int mm = OUTER ? m : hi - lo, kk = OUTER ? hi - lo : k;
    const T* aa = a + (long long)lo * (OUTER ? sak : sam);
    const T* bb = b + (OUTER ? (long long)lo * sbk : (long long)e * bsb);
    O* cc = out + (OUTER ? (long long)e * bsc : (long long)lo * n);
    const int r = (local / tn) * 16 + tr, c = (local % tn) * 16 + tc;
    float acc = 0.f;
    for (int z = 0; z < kk; z += 16) {
      smem[threadIdx.x] = r < mm && z + tc < kk ? widen(aa[(long long)r * sam + (z + tc) * sak]) : 0.f;
      smem[256 + threadIdx.x] = c < n && z + tr < kk ? widen(bb[(long long)(z + tr) * sbk + c * sbn]) : 0.f;
      __syncthreads();
      for (int q = 0; q < 16 && z + q < kk; ++q) acc = fmaf(smem[tr * 16 + q], smem[256 + q * 16 + tc], acc);
      __syncthreads();
    }
    if (r < mm && c < n) cc[(long long)r * n + c] = Store<O, T>::at(acc);
    __syncthreads();
  }
}
}  // namespace latch_ragged
