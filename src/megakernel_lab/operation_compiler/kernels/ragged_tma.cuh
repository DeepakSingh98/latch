// LATCH-TINYMOE-XMAT: TMA-fed flattened segmented right products (ragged_mm: packed rows x per-segment matrices).
//
// Same task enumeration, flattened stage sequence and per-element arithmetic as latch_ragged::flat (and therefore as
// latch_ragged::product): BF16 x BF16 wgmma, FP32 accumulation over ascending k16 steps starting with scale-d = 0 in
// each task, one cvt.rn.bf16x2 rounding, masked pair stores. Only operand staging differs: thread 0 issues bulk tensor
// copies (one A box, one or two B boxes per stage) that complete the stage's mbarrier, instead of 256-thread cp.async.
// Tensor maps (128-byte swizzle, zero fill outside the tensor) are encoded by the runtime from the operand views:
//   A  [rows, K] K-major, box {64, BM}; a tile's rows past its segment read the next segments' rows, whose
//      results are never stored (stores are masked to the segment), so stored values are unchanged;
//   B  BKC:  the stacked per-segment matrices as [groups * N, K] K-major, box {64, BN} (a tile's columns past N read
//            the next segment's rows; never stored);
//      !BKC: as [groups * K, N] N-major, box {64, 64}; requires K % 64 == 0 so a stage never mixes segments' k rows.
// Stage barriers follow the pops4b TMA body (pops4b::tma_bars; POPS4B_PERSIST carries completed phases across calls).
#pragma once
#include "pops4b_tma.cuh"
#include "ragged_experts.cuh"

namespace latch_ragged {

// RAW (LATCH-TRAIN-PRECISION): g.C addresses FP32 storage receiving the unrounded FP32 accumulators (no BF16
// rounding of the result); operands, task enumeration and accumulation order are unchanged. No tile consumers.
template<int E, int BM, int BN, int WM, bool BKC, int NST, int P, class EPI = NoEpi, bool RAW = false>
DEV void tflat_impl(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs* gs, const int64_t* offsets,
                    int rows, float* smem, int rot, const EPI& epi = EPI{}) {
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = C::A_BYTES + P * C::B_BYTES;
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
    pre[tid + 1] = (hi - lo + BM - 1) / BM * tn;
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
  if (ntask == 0) return;                                          // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const int nk = (g.K + 63) / 64;
  struct Task { int e, lo, M, m0, n0; };
  auto task = [&](int i, Task& t) {
    const int id = me + i * G;
    int e = 0;
#pragma unroll 1
    while (e < E - 1 && id >= pre[e + 1]) ++e;
    const int local = id - pre[e];
    t.e = e; t.lo = off[e]; t.M = off[e + 1] - off[e];
    t.m0 = (local / tn) * BM;
    t.n0 = (local % tn) * BN;
  };
  const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
  unsigned uses0[NST];                                             // completed phases of each stage barrier
#pragma unroll
  for (int st = 0; st < NST; ++st) uses0[st] = POPS4B_PERSIST ? pops4b::tma_uses[st] : 0u;
  if (tid == 0) {
    if (!POPS4B_PERSIST) {
#pragma unroll
      for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
  }
  __syncthreads();
  // Issue cursor (thread 0 only): task li, its step lk, issued-step count lq (the consumers' q).
  Task lt;
  int li = 0, lk = 0, lq = 0;
  task(0, lt);
  auto issue = [&]() {
    if (lk >= nk) { ++li; lk = 0; if (li < ntask) task(li, lt); }
    if (li >= ntask) return;
    const int kk = lk * 64, st = lq % NST;
    const unsigned dA = base + st * STAGE, bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
    pops4b::tma2d(dA, tA, kk, lt.lo + lt.m0, bar);
#pragma unroll
    for (int p = 0; p < P; ++p) {
      const void* map = p == 0 ? tB0 : tB1;
      const unsigned dB = dA + C::A_BYTES + p * C::B_BYTES;
      if (BKC) pops4b::tma2d(dB, map, kk, lt.e * g.N + lt.n0, bar);
      else {
#pragma unroll
        for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, map, lt.n0 + 64 * c, lt.e * g.K + kk, bar);
      }
    }
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  float acc[P][C::MT][C::R];
#pragma unroll
  for (int p = 0; p < P; ++p)
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[p][a][j] = 0.f;            // every task's first k16 step overwrites
  int q = 0;
#pragma unroll 1
  for (int i = 0; i < ntask; ++i) {
    Task t;
    task(i, t);
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      pops4b::wg_fence();
      const unsigned sA = base + st * STAGE;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
#pragma unroll
        for (int p = 0; p < P; ++p) {
          const unsigned sB = sA + C::A_BYTES + p * C::B_BYTES;
          const uint64_t db = BKC ? pops4b::dk(sB + wgn * C::WTN * 128 + k16 * 32)
                                  : pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
          for (int a = 0; a < C::MT; ++a) {
            const uint64_t da = pops4b::dk(sA + (wgm * C::WTM + a * 64) * 128 + k16 * 32);
            s101h::MmeWg<C::WTN>::template run<0, BKC ? 0 : 1>(acc[p][a], da, db, (kk | k16) != 0);
          }
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();                                        // step q-1 complete: its stage may be refilled
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      __syncthreads();
      if (tid == 0) issue();
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
#pragma unroll
    for (int p = 0; p < P; ++p) {
      s101h::bf16* Cp = gs[p].C + (long long)t.lo * g.ldc;
      float* Cf = reinterpret_cast<float*>(gs[p].C) + (long long)t.lo * g.ldc;
#pragma unroll
      for (int a = 0; a < C::MT; ++a) {
        const int row = t.m0 + wgm * C::WTM + a * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < C::WTN / 8; ++j) {
          const int col = t.n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = row + 8 * h;
            if (r < t.M && col < g.N) {
              if (RAW)
                *reinterpret_cast<float2*>(Cf + (long long)r * g.ldc + col) =
                    make_float2(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
              else
                *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) =
                    s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
            }
          }
        }
      }
    }
    tile_consumers(epi, t.lo, t.m0, t.n0, t.M, g.N, g.ldc, BM, BN);
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      if (POPS4B_PERSIST) pops4b::tma_uses[st] = uses0[st] + (unsigned)((q - st + NST - 1) / NST);
      else pops4b::bar_inval(bar0 + 8 * st);
    }
    pops4b::proxy_fence();
  }
}

// Segmented outer products (ragged_outer: per segment C[e] = A[lo:hi]^T B[lo:hi]) fed by TMA. A [rows, M] and
// B [rows, N] are M-/N-contiguous maps (dims {M | N, rows}, box {64, 64}); a stage's k rows are packed rows
// lo + kk .. lo + kk + 63. The last stage of a segment whose length is not a multiple of 64 would read the next
// segment's rows, so after it lands and before any MMA reads it, its rows at or past the segment end are zeroed
// (whole 128-byte smem rows, so the swizzle is irrelevant) and made visible to the async proxy: every MMA sees
// exactly the operand values of the cp.async body (absent k rows +0). Same tasks, k order, accumulation from
// scale-d = 0, rounding and masked stores as latch_ragged::flat with OUTER; an empty segment stores +0.
template<int E, int BM, int BN, int WM, int NST>
DEV void touter(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem,
                int rot) {
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = C::A_BYTES + C::B_BYTES;
  static_assert(NST >= 2 && NST <= 8, "stages");
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  int* pre = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE);
  int* off = pre + (E + 1);
  const int tid = threadIdx.x, tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN;
  __syncthreads();                                                 // earlier users of shared memory are done
  if (tid < E) {
    int lo, hi; bounds(offsets, tid, rows, lo, hi);
    off[tid] = lo;
    if (tid == E - 1) off[E] = hi;
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = E * tm * tn;
  const int me = (int)(((unsigned)blockIdx.x + (unsigned)rot) % (unsigned)G);
  const int ntask = me < T ? (T - me + G - 1) / G : 0;
  if (ntask == 0) return;                                          // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  struct Task { int e, lo, K, m0, n0, nk; };
  auto task = [&](int i, Task& t) {
    const int id = me + i * G, e = id / (tm * tn), local = id % (tm * tn);
    t.e = e; t.lo = off[e]; t.K = off[e + 1] - off[e]; t.nk = (t.K + 63) / 64;
    t.m0 = (local / tn) * BM;
    t.n0 = (local % tn) * BN;
  };
  const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
  unsigned uses0[NST];
#pragma unroll
  for (int st = 0; st < NST; ++st) uses0[st] = POPS4B_PERSIST ? pops4b::tma_uses[st] : 0u;
  if (tid == 0) {
    if (!POPS4B_PERSIST) {
#pragma unroll
      for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    pops4b::proxy_fence();
  }
  __syncthreads();
  Task lt;
  int li = 0, lk = 0, lq = 0;
  task(0, lt);
  auto issue = [&]() {                                             // thread 0 only
#pragma unroll 1
    while (li < ntask && lk >= lt.nk) { ++li; lk = 0; if (li < ntask) task(li, lt); }
    if (li >= ntask) return;
    const int kk = lk * 64, st = lq % NST;
    const unsigned dA = base + st * STAGE, dB = dA + C::A_BYTES, bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
#pragma unroll
    for (int c = 0; c < BM / 64; ++c) pops4b::tma2d(dA + c * 8192, tA, lt.m0 + 64 * c, lt.lo + kk, bar);
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tB, lt.n0 + 64 * c, lt.lo + kk, bar);
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  float acc[C::MT][C::R];
  int q = 0;
#pragma unroll 1
  for (int i = 0; i < ntask; ++i) {
    Task t;
    task(i, t);
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[a][j] = 0.f;              // an empty segment stores these
#pragma unroll 1
    for (int kk = 0; kk < t.nk; ++kk, ++q) {
      const int st = q % NST;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
      const unsigned sA = base + st * STAGE, sB = sA + C::A_BYTES;
      // Rows past the segment end (only in a segment's last, partial stage; CTA-uniform): fixed trip count with
      // predicated 16-byte stores, so no thread-dependent branch surrounds the MMA code.
      const int keep_rows = t.K - kk * 64 < 64 ? t.K - kk * 64 : 64;
      if (keep_rows < 64) {
        constexpr int CHUNKS = BM / 64 + BN / 64;
#pragma unroll
        for (int it = 0; it < CHUNKS * 64 * 8 / 256; ++it) {
          const int idx = tid + it * 256, chunk = idx / 512, row = (idx % 512) / 8, col = idx % 8;
          const unsigned dst = (chunk < BM / 64 ? sA + chunk * 8192 : sB + (chunk - BM / 64) * 8192) + row * 128 + col * 16;
          if (row >= keep_rows) asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u) : "memory");
        }
        pops4b::async_fence();                                     // generic zero stores -> wgmma (async proxy)
        __syncthreads();
      }
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
      pops4b::wg_fence();
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        const uint64_t db = pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
        for (int a = 0; a < C::MT; ++a) {
          const uint64_t da = pops4b::dmn(sA + ((wgm * C::WTM + a * 64) / 64) * 8192 + k16 * 2048);
          s101h::MmeWg<C::WTN>::template run<1, 1>(acc[a], da, db, (kk | k16) != 0);
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
      __syncthreads();
      if (tid == 0) issue();
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
    s101h::bf16* Cp = g.C + (long long)t.e * g.bsc;
#pragma unroll
    for (int a = 0; a < C::MT; ++a) {
      const int row = t.m0 + wgm * C::WTM + a * 64 + warp * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < C::WTN / 8; ++j) {
        const int col = t.n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = row + 8 * h;
          if (r < g.M && col < g.N)
            *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) =
                s101h::bpk(acc[a][4 * j + 2 * h], acc[a][4 * j + 2 * h + 1]);
        }
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      if (POPS4B_PERSIST) pops4b::tma_uses[st] = uses0[st] + (unsigned)((q - st + NST - 1) / NST);
      else pops4b::bar_inval(bar0 + 8 * st);
    }
    pops4b::proxy_fence();
  }
}

template<int E, int BM, int BN, int WM, bool BKC, int NST, class EPI = NoEpi>
DEV void tflat(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem,
               int rot, const EPI& epi = EPI{}) {
  tflat_impl<E, BM, BN, WM, BKC, NST, 1, EPI>(tA, tB, tB, &g, offsets, rows, smem, rot, epi);
}

// Paired form (one A map shared by both products; g1 differs from g0 only in B and C).
template<int E, int BM, int BN, int WM, bool BKC, int NST, class EPI = NoEpi>
DEV void tflat2(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs& g0, const popsc::BArgs& g1,
                const int64_t* offsets, int rows, float* smem, int rot, const EPI& epi = EPI{}) {
  const popsc::BArgs gs[2] = {g0, g1};
  tflat_impl<E, BM, BN, WM, BKC, NST, 2, EPI>(tA, tB0, tB1, gs, offsets, rows, smem, rot, epi);
}

// RAW forms (LATCH-TRAIN-PRECISION): unrounded FP32 results (ragged_mm_raw); g.C is the FP32 storage.
template<int E, int BM, int BN, int WM, bool BKC, int NST>
DEV void tflat_raw(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem,
                   int rot) {
  tflat_impl<E, BM, BN, WM, BKC, NST, 1, NoEpi, true>(tA, tB, tB, &g, offsets, rows, smem, rot, NoEpi{});
}

template<int E, int BM, int BN, int WM, bool BKC, int NST>
DEV void tflat2_raw(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs& g0, const popsc::BArgs& g1,
                    const int64_t* offsets, int rows, float* smem, int rot) {
  const popsc::BArgs gs[2] = {g0, g1};
  tflat_impl<E, BM, BN, WM, BKC, NST, 2, NoEpi, true>(tA, tB0, tB1, gs, offsets, rows, smem, rot, NoEpi{});
}

}  // namespace latch_ragged
