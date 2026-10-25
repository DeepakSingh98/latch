// LATCH-TRAIN-EXPERTS: dynamically claimed (balanced) TMA-fed segmented products.
//
// The bodies of ragged_tma.cuh (latch_ragged::tflat_impl / touter) with one change: which CTA computes a task.
// Instead of the static round robin (task id = CTA + i * grid), a CTA takes its next task from a per-launch counter
// (atomicAdd, one ticket requested one task ahead, so the atomic's latency overlaps the current task's k steps).
// CTAs that finish early take more tasks; a CTA whose product queue is empty moves on to the interval's next phase,
// so independent products of one dependency interval balance against each other.
//   * Weight-gradient (outer) tasks are claimed longest segment first (experts ordered by descending row count, ties
//     by expert index), the list-scheduling order that bounds the critical path by one task. Experts with no rows have
//     no tasks: their +0 results are stored up front by all CTAs (the bits the static body stores for them).
//   * STATIC0: the first task of each CTA is the static one (id = blockIdx.x), tickets start at gridDim.x; the generator
//     selects it for the first claimed product of an interval (every CTA arrives together), and the dynamic first ticket
//     for later products (CTAs arrive at different times).
// Every task is computed whole by one CTA with the unchanged per-element arithmetic: BF16 x BF16 wgmma, FP32
// accumulation over ascending k16 steps from scale-d = 0, +0 for absent k rows (zero fill / zeroed tail rows), one
// cvt.rn.bf16x2 rounding, masked pair stores. Results are therefore bitwise those of tflat / touter for any claim order.
// Task ids are handed from the issuing thread (thread 0) to the consumers through a shared ring written before the CTA
// barrier that precedes their first read (the issuer runs NST - 1 >= 2 stages ahead; every task has >= 1 stage).
// Counter: one zeroed word per product phase per launch (generator: arena words past the planned arena, zeroed at
// kernel entry and ordered by the first grid barrier).
#pragma once
#include "ragged_tma.cuh"

namespace latch_ragged {

__shared__ volatile int xclaim_ring[16];
#ifndef LATCH_CLAIM_SEPI
#define LATCH_CLAIM_SEPI 0
#endif

// Register consumers (segmented_products "repi"): the consumer functor declares reg_tag and takes (flat index, BF16 result of
// product 0, of product 1); it runs in the store loop instead of latch_ragged::tile_consumers.
template<class T, class = void> struct epi_reg : std::false_type {};
template<class T> struct epi_reg<T, std::void_t<typename T::reg_tag>> : std::true_type {};   // local classes: no statics

// KS (1 or 2): 64-wide k boxes per stage. KS 2 stages 128 k per CTA barrier and issue round (the generator admits it
// only for K % 128 == 0, so no stage holds absent k terms); every element keeps its ascending k16 chain.
template<int E, int BM, int BN, int WM, bool BKC, int NST, int P, bool STATIC0, class EPI = NoEpi, int KS = 1>
DEV void tclaim_impl(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs* gs, const int64_t* offsets,
                     int rows, float* smem, unsigned* claim, const EPI& epi = EPI{}) {
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int SUB = C::A_BYTES + P * C::B_BYTES, STAGE = KS * SUB;
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
  if (tid == 0) {
    int total = 0;
    pre[0] = 0;
#pragma unroll 1
    for (int e = 0; e < E; ++e) { total += pre[e + 1]; pre[e + 1] = total; }
    const int T0 = total;
    const int first = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
    xclaim_ring[0] = first < T0 ? first : -1;
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = pre[E];
  if (xclaim_ring[0] < 0) return;                                  // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const int nk = (g.K + 64 * KS - 1) / (64 * KS);
  if (KS == 2 && (g.K & 127)) __trap();                            // generator admits KS 2 only for whole 128-k stages
  struct Task { int e, lo, M, m0, n0; };
  auto task = [&](int id, Task& t) {
    int e = 0;
#pragma unroll 1
    while (e < E - 1 && id >= pre[e + 1]) ++e;
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
  auto issue = [&]() {
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
        const unsigned dB = dA + C::A_BYTES + p * C::B_BYTES;
        if (BKC) pops4b::tma2d(dB, map, kk, lt.e * g.N + lt.n0, bar);
        else {
#pragma unroll
          for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, map, lt.n0 + 64 * c, lt.e * g.K + kk, bar);
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
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
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
            const unsigned sB = sA + C::A_BYTES + p * C::B_BYTES;
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
    } else {
      tile_consumers(epi, t.lo, t.m0, t.n0, t.M, g.N, g.ldc, BM, BN);
    }
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
DEV void tflat_claim(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows,
                     float* smem, int /*rot: unused with claims*/, unsigned* claim, const EPI& epi = EPI{}) {
  tclaim_impl<E, BM, BN, WM, BKC, NST, 1, STATIC0, EPI, KS>(tA, tB, tB, &g, offsets, rows, smem, claim, epi);
}

template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0, class EPI = NoEpi>
DEV void tflat2_claim(const void* tA, const void* tB0, const void* tB1, const popsc::BArgs& g0, const popsc::BArgs& g1,
                      const int64_t* offsets, int rows, float* smem, int /*rot*/, unsigned* claim,
                      const EPI& epi = EPI{}) {
  const popsc::BArgs gs[2] = {g0, g1};
  tclaim_impl<E, BM, BN, WM, BKC, NST, 2, STATIC0, EPI>(tA, tB0, tB1, gs, offsets, rows, smem, claim, epi);
}

// Claimed segmented outer products (ragged_outer, C[e] = A[lo:hi]^T B[lo:hi]); see touter for staging and the tail
// rows. One ticket space per product: experts ordered by descending row count (ties by index); with HOT > 0 the
// experts holding at least HOT x the mean rows ("hot", always a prefix of that order) are tiled HBM x HBN and claimed
// first, the remaining nonempty experts BM x BN. A hot expert's tiles then occupy more CTAs: its k chain (the expert's
// row count) cannot be split without reassociation, so its tasks bound the product, and smaller tiles shorten them.
// Shared memory after the larger stage ring: off[E + 1], ord[E], cnt[4] (nonempty, hot, carried ticket).
// Q (1 or 2) products with the same segments and extents (e.g. the gate and up weight gradients) share one queue: an
// expert's tiles of every product are consecutive tickets, so each product's largest-expert tiles are claimed first.
template<int E, int BM, int BN, int WM, int NST, bool STATIC0, int Q = 1>
DEV int touter_claim_range(const void* const* tAs, const void* const* tBs, const popsc::BArgs* gs, float* smem,
                           const int* off, const int* ord, int e0, int tlo, int thi, int first, unsigned* claim) {
  const popsc::BArgs& g = gs[0];
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = C::A_BYTES + C::B_BYTES;
  static_assert(NST >= 3 && NST <= 8, "stages (>= 3: ring hand-off)");
  static_assert(C::WTN % 64 == 0 && C::WTM % 64 == 0, "warpgroup tiles on whole 64-wide MN-major swizzle chunks");
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  const int tid = threadIdx.x, tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, G = (int)gridDim.x;
  if (first >= thi) return first;                                  // CTA-uniform: no task of this class
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  struct Task { int e, lo, K, m0, n0, nk, p; };
  auto task = [&](int id, Task& t) {
    const int x = id - tlo, e = ord[e0 + x / (Q * tm * tn)], r = x % (Q * tm * tn), local = r % (tm * tn);
    t.p = Q == 1 ? 0 : r / (tm * tn);
    t.e = e; t.lo = off[e]; t.K = off[e + 1] - off[e]; t.nk = (t.K + 63) / 64;
    t.m0 = (local / tn) * BM;
    t.n0 = (local % tn) * BN;
  };
  const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
  unsigned uses0[NST];
#pragma unroll
  for (int st = 0; st < NST; ++st) uses0[st] = POPS4B_PERSIST ? pops4b::tma_uses[st] : 0u;
  if (tid == 0) {
    xclaim_ring[0] = first;
    if (!POPS4B_PERSIST) {
#pragma unroll
      for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    pops4b::proxy_fence();
  }
  __syncthreads();
  Task lt;
  int li = 0, lk = 0, lq = 0, lid = first, carry = 0;
  unsigned nxt = 0u;
  if (tid == 0) {
    task(lid, lt);
    nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
  }
  auto issue = [&]() {                                             // thread 0 only; every claimed task has nk >= 1
    if (lid < 0) return;
    if (lk >= lt.nk) {
      ++li; lk = 0;
      const int id = nxt < (unsigned)thi ? (int)nxt : -1;
      if (id < 0) carry = (int)min(nxt, 0x7fffffffu);              // the first ticket past this class
      xclaim_ring[li & 15] = id;
      lid = id;
      if (id < 0) return;
      task(id, lt);
      nxt = atomicAdd(claim, 1u) + (STATIC0 ? (unsigned)G : 0u);
    }
    const int kk = lk * 64, st = lq % NST;
    const unsigned dA = base + st * STAGE, dB = dA + C::A_BYTES, bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
#pragma unroll
    for (int c = 0; c < BM / 64; ++c) pops4b::tma2d(dA + c * 8192, tAs[lt.p], lt.m0 + 64 * c, lt.lo + kk, bar);
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tBs[lt.p], lt.n0 + 64 * c, lt.lo + kk, bar);
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  __syncthreads();                                                 // ring entries of the prologue's tasks
  float acc[C::MT][C::R];
  int q = 0;
#pragma unroll 1
  for (int i = 0;; ++i) {
    const int id = xclaim_ring[i & 15];
    if (id < 0) break;                                             // CTA-uniform (written before a CTA barrier)
    Task t;
    task(id, t);
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[a][j] = 0.f;
#pragma unroll 1
    for (int kk = 0; kk < t.nk; ++kk, ++q) {
      const int st = q % NST;
      pops4b::bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
      const unsigned sA = base + st * STAGE, sB = sA + C::A_BYTES;
      const int keep_rows = t.K - kk * 64 < 64 ? t.K - kk * 64 : 64;
      if (keep_rows < 64) {
        constexpr int CHUNKS = BM / 64 + BN / 64;
#pragma unroll
        for (int it = 0; it < CHUNKS * 64 * 8 / 256; ++it) {
          const int idx = tid + it * 256, chunk = idx / 512, row = (idx % 512) / 8, col = idx % 8;
          const unsigned dst = (chunk < BM / 64 ? sA + chunk * 8192 : sB + (chunk - BM / 64) * 8192) + row * 128 + col * 16;
          if (row >= keep_rows) asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};" ::"r"(dst), "r"(0u) : "memory");
        }
        pops4b::async_fence();
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
    s101h::bf16* Cp = gs[t.p].C + (long long)t.e * g.bsc;
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
  return carry;                                                    // meaningful in thread 0
}

template<int E, int BM, int BN, int WM, int NST, bool STATIC0, int HOT, int HBM, int HBN, int HWM, int HNST, int Q>
DEV void touter_claim_q(const void* const* tAs, const void* const* tBs, const popsc::BArgs* gs, const int64_t* offsets,
                        int rows, float* smem, unsigned* claim) {
  const popsc::BArgs& g = gs[0];
  constexpr int RING = NST * (pops4b::Cfg<BM, BN, WM>::A_BYTES + pops4b::Cfg<BM, BN, WM>::B_BYTES);
  constexpr int HRING = HOT ? HNST * (pops4b::Cfg<HBM, HBN, HWM>::A_BYTES + pops4b::Cfg<HBM, HBN, HWM>::B_BYTES) : 0;
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  int* off = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + (RING > HRING ? RING : HRING));
  int* ord = off + (E + 1);
  int* cnt = ord + E;                                              // [0] nonempty, [1] hot, [2] carried ticket
  const int tid = threadIdx.x, G = (int)gridDim.x;
  __syncthreads();                                                 // earlier users of shared memory are done
  if (tid < E) {
    int lo, hi; bounds(offsets, tid, rows, lo, hi);
    off[tid] = lo;
    if (tid == E - 1) off[E] = hi;
  }
  __syncthreads();
  if (tid < E) {                                                   // rank by (length desc, index asc)
    const int len = off[tid + 1] - off[tid];
    int rank = 0;
#pragma unroll 1
    for (int j = 0; j < E; ++j) {
      const int lj = off[j + 1] - off[j];
      rank += (lj > len) || (lj == len && j < tid);
    }
    ord[rank] = tid;
  }
  if (tid == 0) {
    int ne = 0, nh = 0;
#pragma unroll 1
    for (int e = 0; e < E; ++e) {
      const int len = off[e + 1] - off[e];
      ne += len > 0;
      nh += HOT > 0 && len > 0 && (long long)len * E >= (long long)HOT * rows;
    }
    cnt[0] = ne; cnt[1] = nh;
  }
  __syncthreads();
  const int NE = cnt[0], NH = cnt[1];
  const int TH = Q * NH * ((g.M + HBM - 1) / HBM) * ((g.N + HBN - 1) / HBN);
  const int T = TH + Q * (NE - NH) * ((g.M + BM - 1) / BM) * ((g.N + BN - 1) / BN);
  // Experts without rows: +0 results (bitwise the static body's empty-segment store), spread over the grid. The
  // generator admits only contiguous [M, N] results (ldc == N, bsc == M * N), so each is one flat run of words; plain
  // 32-bit indexing keeps the body free of division calls (a call would serialize its warpgroup MMA, C7509).
  if (NE < E) {
    const int words = g.M * g.N / 2, stride = G * (int)blockDim.x;
#pragma unroll 1
    for (int j = NE; j < E; ++j)
#pragma unroll 1
      for (int p = 0; p < Q; ++p) {
        uint32_t* out = reinterpret_cast<uint32_t*>(gs[p].C + (long long)ord[j] * g.bsc);
#pragma unroll 1
        for (int w = (int)blockIdx.x * (int)blockDim.x + tid; w < words; w += stride) out[w] = 0u;
      }
  }
  if (tid == 0) cnt[2] = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
  __syncthreads();
  int first = cnt[2];
  if (HOT && TH > 0) {
    // Outer operand maps hold 64 x 64 boxes for every tile shape, so the hot range reuses them.
    const int carry = touter_claim_range<E, HBM, HBN, HWM, HNST, STATIC0, Q>(tAs, tBs, gs, smem, off, ord, 0, 0, TH,
                                                                            first, claim);
    if (tid == 0) cnt[2] = carry;
    __syncthreads();
    first = cnt[2];
  }
  touter_claim_range<E, BM, BN, WM, NST, STATIC0, Q>(tAs, tBs, gs, smem, off, ord, NH, TH, T, first, claim);
}

template<int E, int BM, int BN, int WM, int NST, bool STATIC0, int HOT = 0, int HBM = 128, int HBN = 64, int HWM = 2,
         int HNST = 6>
DEV void touter_claim(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows,
                      float* smem, int /*rot*/, unsigned* claim) {
  const void* tAs[1] = {tA};
  const void* tBs[1] = {tB};
  touter_claim_q<E, BM, BN, WM, NST, STATIC0, HOT, HBM, HBN, HWM, HNST, 1>(tAs, tBs, &g, offsets, rows, smem, claim);
}

// Two weight-gradient products with the same segment offsets, row extent, [M, N] results and tile (g1 differs from g0
// in A, B and C only; the generator checks), claimed from one queue.
template<int E, int BM, int BN, int WM, int NST, bool STATIC0, int HOT = 0, int HBM = 128, int HBN = 64, int HWM = 2,
         int HNST = 6>
DEV void touter2_claim(const void* tA0, const void* tB0, const void* tA1, const void* tB1, const popsc::BArgs& g0,
                       const popsc::BArgs& g1, const int64_t* offsets, int rows, float* smem, int /*rot*/,
                       unsigned* claim) {
  const void* tAs[2] = {tA0, tA1};
  const void* tBs[2] = {tB0, tB1};
  const popsc::BArgs gs[2] = {g0, g1};
  touter_claim_q<E, BM, BN, WM, NST, STATIC0, HOT, HBM, HBN, HWM, HNST, 2>(tAs, tBs, gs, offsets, rows, smem, claim);
}

}  // namespace latch_ragged
