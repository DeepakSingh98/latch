// Binary-search claimed segmented-product bodies (segmented_products features "bsearch", "scatter" and "kcat").
// Every body here keeps the task enumeration, task order, TMA staging, MMA order, rounding and work counters of the
// latch_ragged / latch_pack body it replaces, unless stated otherwise (the K-concatenated kcat bodies).
#pragma once
#include "pack_fusion.cuh"

namespace latch_grp {

using latch_ragged::bounds;
using latch_ragged::xclaim_ring;

using latch_ragged::NoEpi;
using latch_ragged::epi_reg;
using latch_ragged::epi_wait;
using latch_ragged::epi_tile;
using latch_ragged::tile_consumers;
using latch_ragged::work_task;
using latch_ragged::zero_tail_rows;

// tclaim_impl_h / tflat_claim_h / tflat2_claim_h / tflat_claim_scatter_rb: latch_ragged::tclaim_impl (and
// latch_pack::tflat_claim_scatter) with the ticket -> expert lookup done by binary search over the per-launch prefix
// table instead of a linear scan (up to E - 1 dependent shared loads per task switch, by every thread and by the
// TMA-issuing thread).
// The lookup returns the same expert for every ticket, so tasks, their order and every stored bit are unchanged.
// Their per-call task table (prefix sums of per-expert task counts) is built by a one-warp shuffle scan instead of a
// serial loop in thread 0 (exact integer sums, same table).
template<int E, int BM, int BN, int WM, bool BKC, int NST, int P, bool STATIC0, class EPI = NoEpi, int KS = 1>
DEV void tclaim_impl_h(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs* gs, const int64_t* offsets,
                     int rows, float* smem, unsigned* claim, const EPI& epi = EPI{}, int64_t* work0 = nullptr,
                     int64_t* work1 = nullptr) {
  using C = pops4b::Cfg<BM, BN, WM>;
  // N-major B tiles of BN % 64 != 0 (e.g. 160) are staged as whole 64-column TMA chunks (zero-filled
  // past N by the tensor map); the warpgroup MMA reads its BN columns of them.
  constexpr int BNC = BKC ? BN : (BN + 63) / 64 * 64, BBYTES = BKC ? C::B_BYTES : BNC * 128;
  constexpr int SUB = C::A_BYTES + P * BBYTES, STAGE = KS * SUB;
  static_assert(NST >= 3 && NST <= 8 && (P == 1 || P == 2) && (KS == 1 || KS == 2), "stages / products / k boxes");
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
  if (tid < 32) {                                                 // one-warp exact integer prefix scan
    int carry = 0;
#pragma unroll 1
    for (int b_ = 0; b_ < E; b_ += 32) {
      int v = b_ + tid < E ? pre[b_ + tid + 1] : 0;
#pragma unroll
      for (int d_ = 1; d_ < 32; d_ <<= 1) { const int n_ = __shfl_up_sync(0xffffffffu, v, d_); if (tid >= d_) v += n_; }
      if (b_ + tid < E) pre[b_ + tid + 1] = carry + v;
      carry += __shfl_sync(0xffffffffu, v, 31);
    }
    if (tid == 0) {
      pre[0] = 0;
      const int T0 = carry;
      const int first = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
      xclaim_ring[0] = first < T0 ? first : -1;
    }
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = pre[E];
  if (xclaim_ring[0] < 0) return;                                  // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const int nk = (g.K + 64 * KS - 1) / (64 * KS);
  if (KS == 2 && (g.K & 127)) __trap();                            // generator admits KS 2 only for whole 128-k stages
  struct Task { int e, lo, M, m0, n0; };
  auto task = [&](int id, Task& t) LATCH_PACK_INLINE {
    // Binary search for the largest e with pre[e] <= id (pre is non-decreasing; an empty expert repeats
    // its successor's prefix, so it is never the largest such index): the same expert as the linear scan.
    int lo_ = 0, hi_ = E - 1;
#pragma unroll 1
    while (lo_ < hi_) { const int mid_ = (lo_ + hi_ + 1) >> 1; if (pre[mid_] <= id) lo_ = mid_; else hi_ = mid_ - 1; }
    const int e = lo_;
    const int local = id - pre[e];
    t.e = e; t.lo = off[e]; t.M = off[e + 1] - off[e];
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
  // Issue cursor (thread 0 only): task sequence li with id lid (-1: exhausted), its step lk, issued-step count lq,
  // and the pending next ticket.
  Task lt;
  int li = 0, lk = 0, lq = 0, lid = xclaim_ring[0];
  unsigned nxt = 0u;
  if (tid == 0) {
    task(lid, lt);
    nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
  }
  auto issue = [&]() LATCH_PACK_INLINE {
    if (lid < 0) return;
    if (lk >= nk) {
      ++li; lk = 0;
      const int id = nxt < (unsigned)T ? (int)nxt : -1;
      xclaim_ring[li & 15] = id;
      lid = id;
      if (id < 0) return;
      task(id, lt);
      nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
    }
    if constexpr (epi_wait<EPI>::value) {
      // Tile readiness (ops/tile_ready.py): the task's A rows are complete before its first operand stage is issued.
      if (lk == 0) epi.wait_a((long long)lt.lo + lt.m0, lt.M - lt.m0 < BM ? lt.M - lt.m0 : BM, (long long)rows);
    }
    const int st = lq % NST;
    const unsigned bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
#pragma unroll
    for (int sub = 0; sub < KS; ++sub) {
      const int kk = (lk * KS + sub) * 64;
      const unsigned dA = base + st * STAGE + sub * SUB;
      pops4b::tma2d(dA, tA, kk, lt.lo + lt.m0, bar);
#pragma unroll
      for (int p = 0; p < P; ++p) {
        const void* map = p == 0 ? tB0 : tB1;
        const unsigned dB = dA + C::A_BYTES + p * BBYTES;
        if (BKC) pops4b::tma2d(dB, map, kk, lt.e * g.N + lt.n0, bar);
        else {
#pragma unroll
          for (int c = 0; c < BNC / 64; ++c) pops4b::tma2d(dB + c * 8192, map, lt.n0 + 64 * c, lt.e * g.K + kk, bar);
        }
      }
    }
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  __syncthreads();                                                 // ring entries of the prologue's tasks
  float acc[P][C::MT][C::R];
#pragma unroll
  for (int p = 0; p < P; ++p)
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[p][a][j] = 0.f;
  int q = 0;
#pragma unroll 1
  for (int i = 0;; ++i) {
    const int id = xclaim_ring[i & 15];
    if (id < 0) break;                                             // CTA-uniform (written before a CTA barrier)
    Task t;
    task(id, t);
    int steps = 0;                                                 // executed 64-k stages (work counters)
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      steps += KS;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
      if constexpr (!BKC && KS == 1) {
        if ((g.K & 63) && kk == nk - 1)                            // CTA-uniform: N-major B k tail
          zero_tail_rows<P * (BNC / 64)>(base + st * STAGE + C::A_BYTES, g.K & 63);
      }
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      pops4b::wg_fence();
#pragma unroll
      for (int sub = 0; sub < KS; ++sub) {
        const unsigned sA = base + st * STAGE + sub * SUB;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
#pragma unroll
          for (int p = 0; p < P; ++p) {
            const unsigned sB = sA + C::A_BYTES + p * BBYTES;
            const uint64_t db = BKC ? pops4b::dk(sB + wgn * C::WTN * 128 + k16 * 32)
                                    : pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
            for (int a = 0; a < C::MT; ++a) {
              const uint64_t da = pops4b::dk(sA + (wgm * C::WTM + a * 64) * 128 + k16 * 32);
              s101h::MmeWg<C::WTN>::template run<0, BKC ? 0 : 1>(acc[p][a], da, db, ((kk * KS + sub) | k16) != 0);
            }
          }
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();
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
    // Staged epilogue (screening, LATCH_CLAIM_SEPI): each warpgroup writes its BF16 tile (the same cvt.rn.bf16x2 bits)
    // to its own shared region, then stores whole 16-byte row chunks; rows past the segment / columns past N are skipped.
    constexpr bool SEPI_ = LATCH_CLAIM_SEPI && WM == 2 &&
        NST * STAGE + 128 + ((8 * (E + 2) + 127) & ~127) + 2 * P * C::WTM * (C::WTN * 2 + 16) <= 220 * 1024;
    if constexpr (SEPI_) {
      constexpr int PITCH = C::WTN * 2 + 16;
      constexpr int ROWS = C::WTM;                                 // this warpgroup's rows: MT 64-row slabs
      char* sp = reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE + ((8 * (E + 2) + 127) & ~127)
                 + wg * P * ROWS * PITCH;
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a)
#pragma unroll
          for (int j = 0; j < C::WTN / 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
              *reinterpret_cast<uint32_t*>(sp + p * ROWS * PITCH + (a * 64 + warp * 16 + (lane >> 2) + 8 * h) * PITCH
                                           + (8 * j + 2 * (lane & 3)) * 2) =
                  s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      const int lt_ = tid & 127;
#pragma unroll
      for (int p = 0; p < P; ++p) {
        s101h::bf16* Cp = gs[p].C + (long long)t.lo * g.ldc;
#pragma unroll
        for (int it = 0; it < ROWS * C::WTN / 8 / 128; ++it) {
          const int idx = lt_ + it * 128, rl = idx / (C::WTN / 8), c8 = idx % (C::WTN / 8);
          const int r = t.m0 + wgm * ROWS + rl, col = t.n0 + 8 * c8;
          if (r < t.M && col < g.N)
            *reinterpret_cast<uint4*>(Cp + (long long)r * g.ldc + col) =
                *reinterpret_cast<const uint4*>(sp + p * ROWS * PITCH + rl * PITCH + 16 * c8);
        }
      }
    } else {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      s101h::bf16* Cp = gs[p].C + (long long)t.lo * g.ldc;
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
              *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) =
                  s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
          }
        }
      }
    }
    }
    if constexpr (epi_reg<EPI>::value) {
      // Same elements, rows, columns and BF16 bits as the stores above (cvt.rn.bf16x2 of the same accumulators).
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
              const uint32_t w0 = s101h::bpk(acc[0][a][4 * j + 2 * h], acc[0][a][4 * j + 2 * h + 1]);
              const uint32_t w1 = s101h::bpk(acc[P - 1][a][4 * j + 2 * h], acc[P - 1][a][4 * j + 2 * h + 1]);
              s101h::bf16 x0, x1, y0, y1;
              const unsigned short b00 = (unsigned short)(w0 & 0xffffu), b01 = (unsigned short)(w0 >> 16);
              const unsigned short b10 = (unsigned short)(w1 & 0xffffu), b11 = (unsigned short)(w1 >> 16);
              memcpy(&x0, &b00, 2); memcpy(&x1, &b01, 2); memcpy(&y0, &b10, 2); memcpy(&y1, &b11, 2);
              const long long i = (long long)(t.lo + r) * g.ldc + col;
              epi(i, x0, y0);
              epi(i + 1, x1, y1);
            }
          }
        }
      }
    } else if constexpr (epi_tile<EPI>::value) {
      // Tile readiness (ops/tile_ready.py): per-warpgroup publication. The warpgroup's 128 threads finished their
      // stores of this task (rows wgm * WTM.., columns wgn * WTN.. of the tile, for every product of the pair); after
      // their named barrier one thread publishes exactly that element count (release: fence + atomic adds). No CTA
      // barrier.
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      if ((tid & 127) == 0) {
        const int r_lo = t.m0 + wgm * C::WTM, c_lo = t.n0 + wgn * C::WTN;
        const int r_hi = r_lo + C::WTM < t.M ? r_lo + C::WTM : t.M, c_hi = c_lo + C::WTN < g.N ? c_lo + C::WTN : g.N;
        if (r_hi > r_lo && c_hi > c_lo) epi.publish_rows((long long)t.lo + r_lo, r_hi - r_lo, c_hi - c_lo);
      }
    } else {
      tile_consumers(epi, t.lo, t.m0, t.n0, t.M, g.N, g.ldc, BM, BN);
    }
    work_task<false>(work0, t.e, t.m0, t.n0, t.M, g.N, g.K, BM, BN, steps);
    if (P == 2) work_task<false>(work1, t.e, t.m0, t.n0, t.M, g.N, g.K, BM, BN, steps);
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

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0, int KS = 1, class EPI = NoEpi>
DEV void tflat_claim_h(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows,
                     float* smem, int /*rot: unused with claims*/, unsigned* claim, const EPI& epi = EPI{},
                     int64_t* work = nullptr) {
  tclaim_impl_h<E, BM, BN, WM, BKC, NST, 1, STATIC0, EPI, KS>(tA, tB, tB, &g, offsets, rows, smem, claim, epi, work);
}

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0, class EPI = NoEpi>
DEV void tflat2_claim_h(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs& g0, const popsc::BArgs& g1,
                      const int64_t* offsets, int rows, float* smem, int /*rot*/, unsigned* claim,
                      const EPI& epi = EPI{}, int64_t* work0 = nullptr, int64_t* work1 = nullptr) {
  const popsc::BArgs gs[2] = {g0, g1};
  tclaim_impl_h<E, BM, BN, WM, BKC, NST, 2, STATIC0, EPI>(tA, tB0, tB1, gs, offsets, rows, smem, claim, epi, work0, work1);
}

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0, class EPI = latch_ragged::NoEpi>
DEV void tflat_claim_scatter_rb(const int64_t* __restrict__ dest, const int64_t* __restrict__ check, const void* tA,
                             const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem,
                             int /*rot*/, unsigned* claim, const EPI& epi = EPI{}, int64_t* work0 = nullptr) {
  // Tile readiness (ops/tile_ready.py): EPI is NoEpi or a waiting ReadyT (results are scattered, not published).
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = C::A_BYTES + C::B_BYTES;
  static_assert(NST >= 3 && NST <= 8, "stages");
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  int* pre = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE);
  int* off = pre + (E + 1);
  const int tid = threadIdx.x, tn = (g.N + BN - 1) / BN;
  __syncthreads();
  if (tid < E) {
    int lo, hi; bounds(offsets, tid, rows, lo, hi);
    off[tid] = lo;
    if (tid == E - 1) off[E] = hi;
    pre[tid + 1] = (hi - lo + BM - 1) / BM * tn;
  }
  __syncthreads();
  if (tid < 32) {                                                 // one-warp exact integer prefix scan
    int carry = 0;
#pragma unroll 1
    for (int b_ = 0; b_ < E; b_ += 32) {
      int v = b_ + tid < E ? pre[b_ + tid + 1] : 0;
#pragma unroll
      for (int d_ = 1; d_ < 32; d_ <<= 1) { const int n_ = __shfl_up_sync(0xffffffffu, v, d_); if (tid >= d_) v += n_; }
      if (b_ + tid < E) pre[b_ + tid + 1] = carry + v;
      carry += __shfl_sync(0xffffffffu, v, 31);
    }
    if (tid == 0) {
      pre[0] = 0;
      const int first = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
      xclaim_ring[0] = first < carry ? first : -1;
    }
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = pre[E];
  if (xclaim_ring[0] < 0) return;
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const int nk = (g.K + 63) / 64;
  struct Task { int e, lo, M, m0, n0; };
  auto task = [&](int id0, Task& t) LATCH_PACK_INLINE {
    // Readiness policy bit 0: tickets map to tasks in descending order (a bijection of tasks).
    int id = id0;
    if constexpr (latch_ragged::epi_wait<EPI>::value) {
      if (epi.policy & 1) id = T - 1 - id0;
    }
    // Binary search for the largest e with pre[e] <= id (pre is non-decreasing; an empty expert repeats
    // its successor's prefix, so it is never the largest such index): the same expert as the linear scan.
    int lo_ = 0, hi_ = E - 1;
#pragma unroll 1
    while (lo_ < hi_) { const int mid_ = (lo_ + hi_ + 1) >> 1; if (pre[mid_] <= id) lo_ = mid_; else hi_ = mid_ - 1; }
    const int e = lo_;
    const int local = id - pre[e];
    t.e = e; t.lo = off[e]; t.M = off[e + 1] - off[e];
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
  int li = 0, lk = 0, lq = 0, lid = xclaim_ring[0];
  unsigned nxt = 0u;
  int spare = -1;                                                  // ready-first: one extra claimed ticket
  if (tid == 0) {
    task(lid, lt);
    nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
  }
  auto ready = [&](int id) LATCH_PACK_INLINE {                    // non-blocking: is this task's A input complete?
    bool ok = true;
    if constexpr (latch_ragged::epi_wait<EPI>::value) {
      Task tt;
      task(id, tt);
      ok = epi.probe_a((long long)tt.lo + tt.m0, tt.M - tt.m0 < BM ? tt.M - tt.m0 : BM, (long long)rows);
    }
    return ok;
  };
  auto issue = [&]() LATCH_PACK_INLINE {                                             // thread 0 only (tclaim_impl)
    if (lid < 0) return;
    if (lk >= nk) {
      ++li; lk = 0;
      int id = nxt < (unsigned)T ? (int)nxt : -1;
      bool took_spare = false;
      if constexpr (latch_ragged::epi_wait<EPI>::value) {
        // Readiness policy bit 1 (ready-first): if the next ticket's input rows are not complete, claim at most one
        // spare ticket and run whichever is ready; otherwise keep the next ticket (its wait blocks below). Every claimed
        // ticket is run by this CTA, so no task is lost and producers (which never wait on consumers) keep progressing.
        if (epi.policy & 2) {
          if (id < 0) {
            if (spare >= 0) { id = spare; spare = -1; took_spare = true; }
          } else if (!ready(id)) {
            if (spare < 0) {
              const unsigned s2 = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
              spare = s2 < (unsigned)T ? (int)s2 : -1;
            }
            if (spare >= 0 && ready(spare)) { id = spare; spare = -1; took_spare = true; }
          }
        }
      }
      xclaim_ring[li & 15] = id;
      lid = id;
      if (id < 0) return;
      task(id, lt);
      if (!took_spare) nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
    }
    if constexpr (latch_ragged::epi_wait<EPI>::value) {
      if (lk == 0) epi.wait_a((long long)lt.lo + lt.m0, lt.M - lt.m0 < BM ? lt.M - lt.m0 : BM, (long long)rows);
    }
    const int st = lq % NST;
    const unsigned bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
    const int kk = lk * 64;
    const unsigned dA = base + st * STAGE, dB = dA + C::A_BYTES;
    pops4b::tma2d(dA, tA, kk, lt.lo + lt.m0, bar);
    if (BKC) pops4b::tma2d(dB, tB, kk, lt.e * g.N + lt.n0, bar);
    else {
#pragma unroll
      for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tB, lt.n0 + 64 * c, lt.e * g.K + kk, bar);
    }
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  __syncthreads();
  float acc[C::MT][C::R];
#pragma unroll
  for (int a = 0; a < C::MT; ++a)
#pragma unroll
    for (int j = 0; j < C::R; ++j) acc[a][j] = 0.f;
  int q = 0;
  bool bad = false;
  // Destination row of packed row `packed`, or -1 when the bijection check fails (the row is then not stored; the body
  // traps after its MMA loop).
  // Staged epilogue plus a per-warpgroup table of resolved destination rows (int, -1 = not stored).
  constexpr bool SEPI_ = LATCH_CLAIM_SEPI && WM == 2 &&
      NST * STAGE + 128 + ((8 * (E + 2) + 127) & ~127) + 2 * C::WTM * (C::WTN * 2 + 16) + 2 * C::WTM * 4 <= 220 * 1024;
  auto unpacked = [&](int packed) LATCH_PACK_INLINE {
    const int64_t d = dest[packed];
    const bool ok = d >= 0 && d < rows && check[d < 0 || d >= rows ? 0 : d] == packed;
    bad |= !ok;
    return ok ? (long long)d : -1LL;
  };
#pragma unroll 1
  for (int i = 0;; ++i) {
    const int id = xclaim_ring[i & 15];
    if (id < 0) break;
    Task t;
    task(id, t);
    int steps = 0;
    // The first WTM threads of each warpgroup own one result row each; its destination dest[packed] is
    // loaded once here (before the MMA loop) and its bijection check check[dest] during the first k step, instead of
    // both being reloaded for every 16-byte chunk of the row after the MMA loop (16-20 dependent load pairs per row).
    const int lt0 = tid & 127;
    int rpk = -1;
    long long rd = -1LL, rck = -2LL;
    if constexpr (SEPI_) {
      const int r = t.m0 + wgm * C::WTM + lt0;
      if (lt0 < C::WTM && r < t.M) { rpk = t.lo + r; rd = dest[rpk]; }
    }
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      ++steps;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
      if constexpr (!BKC) {
        if ((g.K & 63) && kk == nk - 1)
          latch_ragged::zero_tail_rows<BN / 64>(base + st * STAGE + C::A_BYTES, g.K & 63);
      }
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
      pops4b::wg_fence();
      const unsigned sA = base + st * STAGE;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        const unsigned sB = sA + C::A_BYTES;
        const uint64_t db = BKC ? pops4b::dk(sB + wgn * C::WTN * 128 + k16 * 32)
                                : pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
        for (int a = 0; a < C::MT; ++a) {
          const uint64_t da = pops4b::dk(sA + (wgm * C::WTM + a * 64) * 128 + k16 * 32);
          s101h::MmeWg<C::WTN>::template run<0, BKC ? 0 : 1>(acc[a], da, db, (kk | k16) != 0);
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();
#pragma unroll
      for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
      __syncthreads();
      if (tid == 0) issue();
      if constexpr (SEPI_) {
        if (kk == 0 && rpk >= 0) rck = check[rd < 0 || rd >= rows ? 0 : rd];
      }
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
    if constexpr (SEPI_) {
      constexpr int PITCH = C::WTN * 2 + 16;
      constexpr int ROWS = C::WTM;
      char* sp = reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE + ((8 * (E + 2) + 127) & ~127)
                 + wg * ROWS * PITCH;
#pragma unroll
      for (int a = 0; a < C::MT; ++a)
#pragma unroll
        for (int j = 0; j < C::WTN / 8; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            *reinterpret_cast<uint32_t*>(sp + (a * 64 + warp * 16 + (lane >> 2) + 8 * h) * PITCH
                                         + (8 * j + 2 * (lane & 3)) * 2) =
                s101h::bpk(acc[a][4 * j + 2 * h], acc[a][4 * j + 2 * h + 1]);
      int* drow = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE
                                         + ((8 * (E + 2) + 127) & ~127) + 2 * ROWS * PITCH) + wg * ROWS;
      if (lt0 < ROWS) {
        const bool ok = rpk >= 0 && rd >= 0 && rd < rows && rck == rpk;
        bad |= rpk >= 0 && !ok;                                    // the same rows the reference body checks
        drow[lt0] = ok ? (int)rd : -1;
      }
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      const int lt_ = tid & 127;
#pragma unroll
      for (int it = 0; it < ROWS * C::WTN / 8 / 128; ++it) {
        const int idx = lt_ + it * 128, rl = idx / (C::WTN / 8), cc8 = idx % (C::WTN / 8);
        const int col = t.n0 + 8 * cc8;
        const int d = drow[rl];
        if (d >= 0 && col < g.N)
          *reinterpret_cast<uint4*>(g.C + (long long)d * g.ldc + col) =
              *reinterpret_cast<const uint4*>(sp + rl * PITCH + 16 * cc8);
      }
    } else {
#pragma unroll
      for (int a = 0; a < C::MT; ++a) {
        const int row = t.m0 + wgm * C::WTM + a * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = row + 8 * h;
          const long long d = r < t.M ? unpacked(t.lo + r) : -1LL;
          if (d >= 0) {
            s101h::bf16* Cr = g.C + d * g.ldc;
#pragma unroll
            for (int j = 0; j < C::WTN / 8; ++j) {
              const int col = t.n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
              if (col < g.N)
                *reinterpret_cast<uint32_t*>(Cr + col) = s101h::bpk(acc[a][4 * j + 2 * h], acc[a][4 * j + 2 * h + 1]);
            }
          }
        }
      }
    }
    latch_ragged::work_task<false>(work0, t.e, t.m0, t.n0, t.M, g.N, g.K, BM, BN, steps);
  }
  if (bad) __trap();                                               // a failed unpack bijection check
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


// K-concatenated products (segmented_products feature "kcat"): two claimed right products with the same rows,
// segments, N and tile whose results are only ever added (dX = dY_a W_a^T + dY_b W_b^T) computed as ONE
// product over the concatenated reduction axis: the k loop runs ceil(K0/64) stages of (A0, B0) and then ceil(K1/64)
// stages of (A1, B1) into the same FP32 accumulators (TMA zero-fills k past each K), and the tile is rounded to BF16
// once. The two separate BF16 results and the elementwise BF16 sum that read them back are not materialized. Same
// mathematical sum; one rounding instead of three (not bitwise to the separate form).
template<int E, int BM, int BN, int WM, bool BKC, int NST, int P, bool STATIC0, class EPI = NoEpi, int KS = 1>
DEV void tclaim_kcat_h(const void* tA1k, const void* tB1k, int K1k, const void* tA2k, const void* tB2k, int K2k,
                       const void* tA, const void* tB0, const void* tB1, const popsc::BArgs* gs, const int64_t* offsets,
                     int rows, float* smem, unsigned* claim, const EPI& epi = EPI{}, int64_t* work0 = nullptr,
                     int64_t* work1 = nullptr) {
  using C = pops4b::Cfg<BM, BN, WM>;
  // N-major B tiles of BN % 64 != 0 (e.g. 160) are staged as whole 64-column TMA chunks (zero-filled
  // past N by the tensor map); the warpgroup MMA reads its BN columns of them.
  constexpr int BNC = BKC ? BN : (BN + 63) / 64 * 64, BBYTES = BKC ? C::B_BYTES : BNC * 128;
  constexpr int SUB = C::A_BYTES + P * BBYTES, STAGE = KS * SUB;
  static_assert(NST >= 3 && NST <= 8 && (P == 1 || P == 2) && (KS == 1 || KS == 2), "stages / products / k boxes");
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
  if (tid < 32) {                                                 // one-warp exact integer prefix scan
    int carry = 0;
#pragma unroll 1
    for (int b_ = 0; b_ < E; b_ += 32) {
      int v = b_ + tid < E ? pre[b_ + tid + 1] : 0;
#pragma unroll
      for (int d_ = 1; d_ < 32; d_ <<= 1) { const int n_ = __shfl_up_sync(0xffffffffu, v, d_); if (tid >= d_) v += n_; }
      if (b_ + tid < E) pre[b_ + tid + 1] = carry + v;
      carry += __shfl_sync(0xffffffffu, v, 31);
    }
    if (tid == 0) {
      pre[0] = 0;
      const int T0 = carry;
      const int first = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
      xclaim_ring[0] = first < T0 ? first : -1;
    }
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = pre[E];
  if (xclaim_ring[0] < 0) return;                                  // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  static_assert(P == 1 && KS == 1, "kcat bodies: one product, 64-k stages");
  // N-major B (!BKC): each pair addresses its expert rows as e * K_pair + k and zeroes its own last stage's rows
  // past K_pair (the next expert's rows), exactly as the single-product body does for its one tail.
  const int nk0k = (g.K + 63) / 64;                               // stages of the first operand pair
  const int nk1k = nk0k + (K1k + 63) / 64;                        // + stages of the second operand pair
  const int nk = nk1k + (K2k + 63) / 64;                          // + stages of the third (K2k = 0: none)
  if (KS == 2 && (g.K & 127)) __trap();                            // generator admits KS 2 only for whole 128-k stages
  struct Task { int e, lo, M, m0, n0; };
  auto task = [&](int id, Task& t) LATCH_PACK_INLINE {
    // Binary search for the largest e with pre[e] <= id (pre is non-decreasing; an empty expert repeats
    // its successor's prefix, so it is never the largest such index): the same expert as the linear scan.
    int lo_ = 0, hi_ = E - 1;
#pragma unroll 1
    while (lo_ < hi_) { const int mid_ = (lo_ + hi_ + 1) >> 1; if (pre[mid_] <= id) lo_ = mid_; else hi_ = mid_ - 1; }
    const int e = lo_;
    const int local = id - pre[e];
    t.e = e; t.lo = off[e]; t.M = off[e + 1] - off[e];
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
  // Issue cursor (thread 0 only): task sequence li with id lid (-1: exhausted), its step lk, issued-step count lq,
  // and the pending next ticket.
  Task lt;
  int li = 0, lk = 0, lq = 0, lid = xclaim_ring[0];
  unsigned nxt = 0u;
  if (tid == 0) {
    task(lid, lt);
    nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
  }
  auto issue = [&]() LATCH_PACK_INLINE {
    if (lid < 0) return;
    if (lk >= nk) {
      ++li; lk = 0;
      const int id = nxt < (unsigned)T ? (int)nxt : -1;
      xclaim_ring[li & 15] = id;
      lid = id;
      if (id < 0) return;
      task(id, lt);
      nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
    }
    if constexpr (epi_wait<EPI>::value) {
      // Tile readiness (ops/tile_ready.py): the task's A rows are complete before its first operand stage is issued.
      if (lk == 0) epi.wait_a((long long)lt.lo + lt.m0, lt.M - lt.m0 < BM ? lt.M - lt.m0 : BM, (long long)rows);
    }
    const int st = lq % NST;
    const unsigned bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
#pragma unroll
    for (int sub = 0; sub < KS; ++sub) {
      const bool h1k = lk >= nk0k, h2k = lk >= nk1k;                // second / third operand pair
      const int kk = (h2k ? lk - nk1k : h1k ? lk - nk0k : lk) * 64;
      const unsigned dA = base + st * STAGE + sub * SUB;
      pops4b::tma2d(dA, h2k ? tA2k : h1k ? tA1k : tA, kk, lt.lo + lt.m0, bar);
#pragma unroll
      for (int p = 0; p < P; ++p) {
        const void* map = h2k ? tB2k : h1k ? tB1k : (p == 0 ? tB0 : tB1);
        const unsigned dB = dA + C::A_BYTES + p * BBYTES;
        if (BKC) pops4b::tma2d(dB, map, kk, lt.e * g.N + lt.n0, bar);
        else {
#pragma unroll
          for (int c = 0; c < BNC / 64; ++c) pops4b::tma2d(dB + c * 8192, map, lt.n0 + 64 * c, lt.e * (h2k ? K2k : h1k ? K1k : g.K) + kk, bar);
        }
      }
    }
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  __syncthreads();                                                 // ring entries of the prologue's tasks
  float acc[P][C::MT][C::R];
#pragma unroll
  for (int p = 0; p < P; ++p)
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[p][a][j] = 0.f;
  int q = 0;
#pragma unroll 1
  for (int i = 0;; ++i) {
    const int id = xclaim_ring[i & 15];
    if (id < 0) break;                                             // CTA-uniform (written before a CTA barrier)
    Task t;
    task(id, t);
    int steps = 0;                                                 // executed 64-k stages (work counters)
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      steps += KS;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
      if constexpr (!BKC && KS == 1) {
        const int ktail_ = kk == nk0k - 1 ? (g.K & 63) : kk == nk1k - 1 ? (K1k & 63) : kk == nk - 1 ? (K2k & 63) : 0;
        if (ktail_)                                                // CTA-uniform: each pair's N-major B k tail
          zero_tail_rows<P * (BNC / 64)>(base + st * STAGE + C::A_BYTES, ktail_);
      }
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[p][a]);
      pops4b::wg_fence();
#pragma unroll
      for (int sub = 0; sub < KS; ++sub) {
        const unsigned sA = base + st * STAGE + sub * SUB;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
#pragma unroll
          for (int p = 0; p < P; ++p) {
            const unsigned sB = sA + C::A_BYTES + p * BBYTES;
            const uint64_t db = BKC ? pops4b::dk(sB + wgn * C::WTN * 128 + k16 * 32)
                                    : pops4b::dmn(sB + (wgn * C::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
            for (int a = 0; a < C::MT; ++a) {
              const uint64_t da = pops4b::dk(sA + (wgm * C::WTM + a * 64) * 128 + k16 * 32);
              s101h::MmeWg<C::WTN>::template run<0, BKC ? 0 : 1>(acc[p][a], da, db, ((kk * KS + sub) | k16) != 0);
            }
          }
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();
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
    // Staged epilogue (screening, LATCH_CLAIM_SEPI): each warpgroup writes its BF16 tile (the same cvt.rn.bf16x2 bits)
    // to its own shared region, then stores whole 16-byte row chunks; rows past the segment / columns past N are skipped.
    constexpr bool SEPI_ = LATCH_CLAIM_SEPI && WM == 2 &&
        NST * STAGE + 128 + ((8 * (E + 2) + 127) & ~127) + 2 * P * C::WTM * (C::WTN * 2 + 16) <= 220 * 1024;
    if constexpr (SEPI_) {
      constexpr int PITCH = C::WTN * 2 + 16;
      constexpr int ROWS = C::WTM;                                 // this warpgroup's rows: MT 64-row slabs
      char* sp = reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE + ((8 * (E + 2) + 127) & ~127)
                 + wg * P * ROWS * PITCH;
#pragma unroll
      for (int p = 0; p < P; ++p)
#pragma unroll
        for (int a = 0; a < C::MT; ++a)
#pragma unroll
          for (int j = 0; j < C::WTN / 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
              *reinterpret_cast<uint32_t*>(sp + p * ROWS * PITCH + (a * 64 + warp * 16 + (lane >> 2) + 8 * h) * PITCH
                                           + (8 * j + 2 * (lane & 3)) * 2) =
                  s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      const int lt_ = tid & 127;
#pragma unroll
      for (int p = 0; p < P; ++p) {
        s101h::bf16* Cp = gs[p].C + (long long)t.lo * g.ldc;
#pragma unroll
        for (int it = 0; it < ROWS * C::WTN / 8 / 128; ++it) {
          const int idx = lt_ + it * 128, rl = idx / (C::WTN / 8), c8 = idx % (C::WTN / 8);
          const int r = t.m0 + wgm * ROWS + rl, col = t.n0 + 8 * c8;
          if (r < t.M && col < g.N)
            *reinterpret_cast<uint4*>(Cp + (long long)r * g.ldc + col) =
                *reinterpret_cast<const uint4*>(sp + p * ROWS * PITCH + rl * PITCH + 16 * c8);
        }
      }
    } else {
#pragma unroll
    for (int p = 0; p < P; ++p) {
      s101h::bf16* Cp = gs[p].C + (long long)t.lo * g.ldc;
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
              *reinterpret_cast<uint32_t*>(Cp + (long long)r * g.ldc + col) =
                  s101h::bpk(acc[p][a][4 * j + 2 * h], acc[p][a][4 * j + 2 * h + 1]);
          }
        }
      }
    }
    }
    if constexpr (epi_reg<EPI>::value) {
      // Same elements, rows, columns and BF16 bits as the stores above (cvt.rn.bf16x2 of the same accumulators).
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
              const uint32_t w0 = s101h::bpk(acc[0][a][4 * j + 2 * h], acc[0][a][4 * j + 2 * h + 1]);
              const uint32_t w1 = s101h::bpk(acc[P - 1][a][4 * j + 2 * h], acc[P - 1][a][4 * j + 2 * h + 1]);
              s101h::bf16 x0, x1, y0, y1;
              const unsigned short b00 = (unsigned short)(w0 & 0xffffu), b01 = (unsigned short)(w0 >> 16);
              const unsigned short b10 = (unsigned short)(w1 & 0xffffu), b11 = (unsigned short)(w1 >> 16);
              memcpy(&x0, &b00, 2); memcpy(&x1, &b01, 2); memcpy(&y0, &b10, 2); memcpy(&y1, &b11, 2);
              const long long i = (long long)(t.lo + r) * g.ldc + col;
              epi(i, x0, y0);
              epi(i + 1, x1, y1);
            }
          }
        }
      }
    } else if constexpr (epi_tile<EPI>::value) {
      // Tile readiness (ops/tile_ready.py): per-warpgroup publication. The warpgroup's 128 threads finished their
      // stores of this task (rows wgm * WTM.., columns wgn * WTN.. of the tile, for every product of the pair); after
      // their named barrier one thread publishes exactly that element count (release: fence + atomic adds). No CTA
      // barrier.
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      if ((tid & 127) == 0) {
        const int r_lo = t.m0 + wgm * C::WTM, c_lo = t.n0 + wgn * C::WTN;
        const int r_hi = r_lo + C::WTM < t.M ? r_lo + C::WTM : t.M, c_hi = c_lo + C::WTN < g.N ? c_lo + C::WTN : g.N;
        if (r_hi > r_lo && c_hi > c_lo) epi.publish_rows((long long)t.lo + r_lo, r_hi - r_lo, c_hi - c_lo);
      }
    } else {
      tile_consumers(epi, t.lo, t.m0, t.n0, t.M, g.N, g.ldc, BM, BN);
    }
    work_task<false>(work0, t.e, t.m0, t.n0, t.M, g.N, g.K, BM, BN, steps);
    if (P == 2) work_task<false>(work1, t.e, t.m0, t.n0, t.M, g.N, g.K, BM, BN, steps);
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

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0>
DEV void tflat_kcat_claim_h(const void* tA0, const void* tB0, const void* tA1, const void* tB1, const popsc::BArgs& g0,
                            const popsc::BArgs& g1, const int64_t* offsets, int rows, float* smem, int /*rot*/,
                            unsigned* claim) {
  tclaim_kcat_h<E, BM, BN, WM, BKC, NST, 1, STATIC0, NoEpi, 1>(tA1, tB1, g1.K, nullptr, nullptr, 0, tA0, tB0, tB0, &g0,
                                                               offsets, rows, smem, claim);
}

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0>
DEV void tflat_kcat3_claim_h(const void* tA0, const void* tB0, const void* tA1, const void* tB1, const void* tA2,
                             const void* tB2, const popsc::BArgs& g0, const popsc::BArgs& g1, const popsc::BArgs& g2,
                             const int64_t* offsets, int rows, float* smem, int /*rot*/, unsigned* claim) {
  tclaim_kcat_h<E, BM, BN, WM, BKC, NST, 1, STATIC0, NoEpi, 1>(tA1, tB1, g1.K, tA2, tB2, g2.K, tA0, tB0, tB0, &g0,
                                                               offsets, rows, smem, claim);
}

}  // namespace latch_grp
