// Row-scattered claimed segmented products (segmented_products feature "scatter"; ops/pack_fusion.py).
//
// latch_pack::tflat_claim_scatter is the claimed right body of kernels/ragged_claim.cuh (latch_ragged::tclaim_impl,
// P = 1, KS = 1) whose result rows are stored at a bijective unpack's destination rows; see the comment at the body.
// pk_bars / pk_ticket are namespace-scope shared storage; they stay so the static shared layout of programs that
// include this header is unchanged.
#pragma once
#include "ragged_claim.cuh"

// Every lambda below is force-inlined: in large device units nvcc may otherwise keep one out of line, and that call
// makes ptxas serialize every warpgroup MMA of the unit (C7510/C7509). Index checks inside the MMA loops are
// branch-free (clamped addresses, a per-thread flag, one trap after the loop).
#define LATCH_PACK_INLINE __attribute__((always_inline))

namespace latch_pack {

using latch_ragged::bounds;
using latch_ragged::xclaim_ring;

__shared__ __align__(8) unsigned long long pk_bars[8];
__shared__ volatile unsigned pk_ticket[2];

// GEMM -> unpack (row scatter): a claimed right product whose result rows are a bijective unpack's source writes each
// result row r straight to its unpacked row dest[r] (U[dest[r]] = C[r]) instead of storing C and copying it.
// Bijection check of the unpack it replaces, per stored row: 0 <= dest[r] < rows and check[dest[r]] == r
// (latch_ops.permutation_gather traps on the same condition; for an embedding by route_order's inverse, check = that
// inverse and dest = its order). Every value is the product's BF16 result bits; only the row a
// result is stored at changes, exactly the row the unpack copied it to. tclaim_impl (P = 1, KS = 1) otherwise:
// TMA A and B, claims, MMA order, rounding, staged epilogue, work counters. No tile consumers (they index C's rows).
template<int E, int BM, int BN, int WM, bool BKC, int NST, bool STATIC0, class EPI = latch_ragged::NoEpi>
DEV void tflat_claim_scatter(const int64_t* __restrict__ dest, const int64_t* __restrict__ check, const void* tA,
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
  if (tid == 0) {
    int total = 0;
    pre[0] = 0;
#pragma unroll 1
    for (int e = 0; e < E; ++e) { total += pre[e + 1]; pre[e + 1] = total; }
    const int first = STATIC0 ? (int)blockIdx.x : (int)atomicAdd(claim, 1u);
    xclaim_ring[0] = first < total ? first : -1;
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
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int a = 0; a < C::MT; ++a) pops4b::keep(acc[a]);
    constexpr bool SEPI_ = LATCH_CLAIM_SEPI && WM == 2 &&
        NST * STAGE + 128 + ((8 * (E + 2) + 127) & ~127) + 2 * C::WTM * (C::WTN * 2 + 16) <= 220 * 1024;
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
      asm volatile("bar.sync %0, 128;" ::"r"(13 + wg) : "memory");
      const int lt_ = tid & 127;
#pragma unroll
      for (int it = 0; it < ROWS * C::WTN / 8 / 128; ++it) {
        const int idx = lt_ + it * 128, rl = idx / (C::WTN / 8), cc8 = idx % (C::WTN / 8);
        const int r = t.m0 + wgm * ROWS + rl, col = t.n0 + 8 * cc8;
        const long long d = r < t.M ? unpacked(t.lo + r) : -1LL;
        if (d >= 0 && col < g.N)
          *reinterpret_cast<uint4*>(g.C + d * g.ldc + col) = *reinterpret_cast<const uint4*>(sp + rl * PITCH + 16 * cc8);
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

}  // namespace latch_pack
