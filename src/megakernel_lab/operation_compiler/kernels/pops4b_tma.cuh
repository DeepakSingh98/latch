// P-OPS4-B TMA + warpgroup-MMA BF16 body for the operation compiler (new code in this lane).
//
// Port of the Latch 0.5 S107A matrix engine main loop (P4-A f9765574a latch05/.../native/lib/s107a_mm.cuh:
// mme_issue / mme_phase_fi): thread 0 keeps NST stages of TMA loads in flight across the CTA's flattened task x 64-k
// sequence, each stage completes an mbarrier (expect_tx), both warpgroups issue m64nNk16 wgmma (vendor/s107a_wgmma.cuh)
// on 128-byte swizzled stages, one wgmma group stays in flight. Changes from S107A: tasks come from the P-OPS-C CTA
// slice (popsc::slice_of) instead of a vbase rotation; no split-K; the epilogue is a register-direct BF16 store; tensor
// maps are encoded by the operation compiler's Runtime for the operand views of this product (manifest
// "tensor_maps"); M, N and K tails need no masking on load: TMA fills out-of-bounds box elements with zeros
// (absent k terms are exact +0 products), and stores are masked.
// Default arithmetic: identical to pops4b_wg.cuh (FP32 wgmma accumulation, ascending k16, one cvt.rn.bf16x2 rounding).
// Explicit serial_bf16_k opt-in adds independent K partials and BF16 partition epilogues; see ops/serial_bf16.py.
#pragma once
#include <type_traits>
#include "pops4b_wg.cuh"

// P7-B: POPS4B_LAST_ARRIVER 1 refills a stage from the last of the CTA's 8 warps to finish with it (shared counter)
// instead of a CTA barrier per k step + thread 0; warpgroups then wait only on data (full mbarriers).
#ifndef POPS4B_LAST_ARRIVER
#define POPS4B_LAST_ARRIVER 0
#endif
// P7-B: POPS4B_PERSIST 1 initializes the stage mbarriers once per kernel (tma_persist_init) and carries each
// barrier's completed-phase count across product calls (parity), instead of init + release fence + inval per call.
#ifndef POPS4B_PERSIST
#define POPS4B_PERSIST 0
#endif
// P7-B: POPS4B_CLAIM_ASYNC 1 (claim mode) takes the first three tickets with one atomic and publishes later tickets
// after the task's k steps instead of waiting for each atomic at the task start.
#ifndef POPS4B_CLAIM_ASYNC
#define POPS4B_CLAIM_ASYNC 0
#endif
// Step-14 speed ('tmaq'): POPS4B_CLAIM_AHEAD 1 claims one task ahead instead of three, so products with about as many
// tasks as CTAs are spread over all CTAs that are free (needs >= NST k steps per task: the prologue's stages all
// belong to the first task, and the next ticket is known before the ring reaches its stages).
#ifndef POPS4B_CLAIM_AHEAD
#define POPS4B_CLAIM_AHEAD 3
#endif
// P9-E: unroll of the staged epilogue's 8-column chunk loop (side-input loads in flight per thread). The accumulators
// are dead once the tile is staged, so registers are free; the per-thread chunk order (and colsum order) is unchanged.
// P9-E: K-segmented products (TArgs::kseg) are compiled in only when the program has one.
#ifndef POPS4B_KSEG
#define POPS4B_KSEG 0
#endif
#ifndef POPS4B_EPI_UNROLL
#define POPS4B_EPI_UNROLL 2
#endif
namespace pops4b { constexpr int kEpiUnroll = POPS4B_EPI_UNROLL; }   // #pragma unroll does not expand macros
// Step-14 speed diagnostic: POPS4B_TRACE 1 records per-CTA globaltimer stamps of each body call (call order is the
// same on every CTA): 0 entry, 2 first task's k steps complete, 3 its tile staged, 4 its epilogue done, 5 exit.
#ifndef POPS4B_TRACE
#define POPS4B_TRACE 0
#endif
namespace pops4b {
#if POPS4B_TRACE
__shared__ unsigned long long* pops4b_tbuf;
__shared__ unsigned pops4b_tcall;
__device__ __forceinline__ unsigned long long tstamp() {
  unsigned long long x; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x)); return x;
}
__device__ __forceinline__ void trace_begin(unsigned long long* buf) {
  if (threadIdx.x == 0) { pops4b_tbuf = buf; pops4b_tcall = 0u; }
  __syncthreads();
}
#define POPS4B_T(call, e) do { if (threadIdx.x == 0 && pops4b_tbuf && blockIdx.x < 256u) \
  pops4b_tbuf[((call) * 6u + (e)) * 256u + blockIdx.x] = pops4b::tstamp(); } while (0)
#else
#define POPS4B_T(call, e) do {} while (0)
#endif
}  // namespace pops4b

namespace pops4b {

__shared__ __align__(8) unsigned long long tma_bars[8];
__shared__ unsigned tma_free[8];
__shared__ volatile int tma_slot[8];
__shared__ unsigned tma_uses[8];

DEV void bar_init(unsigned b, unsigned n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n) : "memory"); }
DEV void bar_inval(unsigned b) { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(b) : "memory"); }
DEV void bar_expect(unsigned b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
DEV bool bar_try(unsigned b, unsigned parity) {
  unsigned ok;
  asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
               : "=r"(ok) : "r"(b), "r"(parity) : "memory");
  return ok != 0;
}
// C++ retry loop (no branch inside inline asm, the CUTLASS form): ptxas then keeps the wgmma pipeline across the wait.
DEV void bar_wait(unsigned b, unsigned parity) {
  while (!bar_try(b, parity)) {
  }
}
DEV void tma2d(unsigned dst, const void* map, int c0, int c1, unsigned bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
DEV void proxy_fence() { asm volatile("fence.proxy.async;" ::: "memory"); }
DEV void tma_persist_init() {
  if (threadIdx.x == 0) {
    const unsigned bar0 = sa(&tma_bars[0]);
#pragma unroll
    for (int st = 0; st < 8; ++st) {
      bar_init(bar0 + 8 * st, 1);
      tma_uses[st] = 0u;
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
}

// tA / tB: CUtensorMap of the A / B operand view. K-major: dims {K, rows}, box {64, BM | BN}; M/N-major: dims
// {rows, K}, box {64, 64}. C is the contiguous BF16 result [M, ldc].
struct TArgs {
  const void* tA;
  const void* tB;
  const bf16* A;          // the operand views the maps describe (not dereferenced; keeps storage and ordering explicit)
  const bf16* B;
  bf16* C;
  long long ldc;
  int M, N, K;
  float* CF = nullptr;    // P6-B split-K: FP32 K-slice results [SPLIT][M][ldc] (SPLIT > 1 only)
  // P7-B: static cost-balanced (LPT) task table for a resident grid of tg CTAs: tt[c], tt[c + 1] bound CTA c's
  // entries in tt[tg + 1 + ...] (task ids = slice * tiles + tile, in processing order). Other grids: round robin.
  const unsigned short* tt = nullptr;
  int tg = 0;
  // P7-B claim mode: tasks are claimed from this per-launch zeroed counter (atomicAdd) three sequences ahead by
  // thread 0 (ring of 8 task-id slots), so CTAs finishing early take more tasks; operand stages still prefetch across
  // task boundaries. Requires >= 2 64-wide k steps per task.
  unsigned* claim = nullptr;
  // P9-E (COLSUM): FP32 column partials of one epilogue member per 64-row slab, [ceil(M / 64)][N]
  float* colpart = nullptr;
  // P9-E (K segments): the product is sum_s A_s B_s over up to three (A, B) map pairs of kseg each (K = segments x
  // kseg, kseg % 64 == 0); k step kk reads segment kk / kseg at kk % kseg. One FP32 accumulator, one rounding.
  const void* tA2 = nullptr;
  const void* tB2 = nullptr;
  const void* tA3 = nullptr;
  const void* tB3 = nullptr;
  int kseg = 0;
  // Explicit arithmetic contract only: independent FP32 partials, ascending K;
  // each intermediate partial adds the previous BF16 result and rounds BF16.
  // Zero preserves the original one-round product. Bound by shared IR proof.
  int serial_bf16_k = 0;
  // LATCH-MOESUP batch-stacked products: A and C are the batches' rows stacked ([batch * M_b, K] / [batch * M_b, N],
  // M = batch * M_b) and B's map stacks the batches' B rows; tiles never cross a batch (M_b % BM == 0, and N_b % BN
  // == 0 when B is K-contiguous). A tile at row m0 reads B rows offset by (m0 / bgrp) * bstep (bgrp = M_b; bstep =
  // K_b for N-major B, N_b for K-contiguous B). bgrp == 0: one batch.
  int bgrp = 0;
  int bstep = 0;
  // LATCH-MOE-EMA-FAST: an M-major A whose batches are stacked along K (map dims {M_b, batch * K_b}): a tile at row
  // m0 reads A columns m0 % bgrp and K rows offset by (m0 / bgrp) * astep (astep = K_b). 0: A rows are stacked.
  int astep = 0;
  // LATCH-MOESUP (row softmax partials, STAGED with BM 128): for every output row and N tile, (max, sum expf(x - max))
  // of the BF16-rounded tile values over columns < rowvalid, written to rowpart[row * rowtiles + n0 / BN]. Each of a
  // row's two threads scans its 64-column chunks in ascending order (max pass, then exp-sum pass), merges chunks in
  // half order, then the pair merges (lower columns first): fixed order, grid independent. rowpart == nullptr: off.
  float2* rowpart = nullptr;
  int rowvalid = 0;
  int rowtiles = 0;
};

// Online-softmax merge of (m, s) with (m2, s2); -INFINITY maxima carry no mass.
DEV void row_merge(float& m, float& s, const float m2, const float s2) {
  if (m2 == -INFINITY) return;
  if (m == -INFINITY) { m = m2; s = s2; return; }
  const float mm = fmaxf(m, m2);
  s = s * expf(m - mm) + s2 * expf(m2 - mm);
  m = mm;
}

// P9-E COLSUM: epilogues with vec8c(i, g, cs) also return one FP32 member value per element (cs[0..7]); the staged
// epilogue sums them per column over each 64-row slab of the tile: every thread adds its rows of a slab in ascending
// order, then the slab's 256 / (BN / 8) threads are added in ascending thread order. Fixed order, grid independent.
template <class T, class = void> struct has_vec8c : std::false_type {};
template <class T> struct has_vec8c<T, std::void_t<decltype(&T::vec8c)>> : std::true_type {};

// Quad transpose of packed BF16 pairs: lane t of a quad holds x[t'] = columns 8 (4 jj + t') + 2 t + {0, 1}; afterwards
// it holds y[t'] = columns 8 (4 jj + t) + 2 t' + {0, 1}, i.e. one 16-byte chunk (8 consecutive columns). Data movement
// only (shfl.xor within the quad); values are unchanged.
DEV uint4 quad_chunk(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, int t) {
  auto pick = [&](int i) { return i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3; };
  const uint32_t r0 = pick(t);                                     // from lane t ^ 0
  const uint32_t r1 = __shfl_xor_sync(0xffffffffu, pick(t ^ 1), 1);
  const uint32_t r2 = __shfl_xor_sync(0xffffffffu, pick(t ^ 2), 2);
  const uint32_t r3 = __shfl_xor_sync(0xffffffffu, pick(t ^ 3), 3);
  auto from = [&](int lane) { const int sft = t ^ lane; return sft == 0 ? r0 : sft == 1 ? r1 : sft == 2 ? r2 : r3; };
  return make_uint4(from(0), from(1), from(2), from(3));
}

// P6-B: optional same-index epilogue on the rounded product: epi(flat index of an even column pair, v0, v1) receives
// the BF16-rounded and widened values (what a load of the stored result yields); STORE=false omits the product store
// when only the epilogue reads it.
struct NoEpi {
  DEV void operator()(unsigned, float, float) const {}
};

// P6-B: epilogues with `pre(i)` (L2 prefetch of their memory operands) are prefetched when a task starts, so the
// epilogue's operand reads after the last k step hit L2.
template <class T, class = void> struct has_pre : std::false_type {};
template <class T> struct has_pre<T, std::void_t<decltype(&T::pre)>> : std::true_type {};
// Step-14 speed ('tepr'): epilogues with prow(i, n) get one call per tile row when a task starts (i = flat index of
// the row's first output column, n = the row's columns in the tile): a bulk L2 prefetch of older side inputs.
template <class T, class = void> struct has_prow : std::false_type {};
template <class T> struct has_prow<T, std::void_t<decltype(&T::prow)>> : std::true_type {};

// P6-B SPLIT > 1: deterministic split-K. Task = (K slice, tile); each slice accumulates its K/SPLIT range in FP32 from
// +0 (ascending k16, as the unsplit body) and stores the unrounded FP32 tile to CF[slice]; the consumer sums the
// slices in fixed order and rounds once.
// P7-B STAGED: the BF16-rounded tile is staged in shared memory after the operand ring ([BM][BN + 8], padded rows)
// and the epilogue runs over 8-column row chunks (epi.vec8(i, g[8]); 16-byte result stores): coalesced side-input
// loads and vector stores instead of per-pair accesses in the accumulator layout. Same rounded values, same
// per-element expressions.
template <int BM, int BN> struct Staging {
  static constexpr int LDS = BN + 8;
  static constexpr int BYTES = BM * LDS * 2;
};

// P7-B ROWS (with STAGED, BN = N: every tile holds whole rows): the staged tile aliases the drained operand ring and
// epi.rows(st, lds, m0, rows) runs a generated row engine (warp per row) over the tile's rows, reading the rounded
// product values from shared memory. Operand stages are issued per task (the next task's first stages after the
// previous task's row engine), so the ring is free while the tile is staged.
// LATCH-RESEARCH-EDITS RAW (SPLIT == 1, no epilogue, even N): the unrounded FP32 accumulator tile is stored to CF
// [M][ldc] with the split-K slice stores (masked float2), the BF16 result is not written. Every k16 wgmma step, the
// +0 staging of absent elements and the task order are those of the BF16-storing body; only the final store differs.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false, class EPI = NoEpi,
          bool STORE = true, int SPLIT = 1, bool STAGED = false, bool ROWS = false, bool ALIAS = false,
          bool HALF = false, bool RAW = false>
DEV void tma_body(const TArgs& g, float* smem, int lo, int hi, int den, int rot, const EPI& epi = EPI{}) {
  using Cf = Cfg<BM, BN, WM>;
  static_assert(NST >= 2 && NST <= 8, "stages");
  static_assert(!AK || BM <= 256, "TMA box rows");
  static_assert(!BKC || BN <= 256, "TMA box rows");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  const int ks = g.K / SPLIT;                                      // SPLIT > 1: K % (64 SPLIT) == 0 (planner)
  const int nk = (ks + 63) / 64;
  const bool claim = g.claim != nullptr && nk >= 2;
  const bool table = !claim && g.tt != nullptr && (int)gridDim.x == g.tg;
#if POPS4B_TRACE
  const unsigned tcall_ = pops4b_tcall;
  __syncthreads();
  if (threadIdx.x == 0) pops4b_tcall = tcall_ + 1u;
  POPS4B_T(tcall_, 0);
#endif
  if (!claim && !table && ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count)) return;
  static_assert(SPLIT == 1 || (std::is_same<EPI, NoEpi>::value && PAIR), "split-K: no epilogue, even N");
  static_assert(!RAW || (SPLIT == 1 && std::is_same<EPI, NoEpi>::value && PAIR && !STAGED && !STORE),
                "raw FP32 result: unsplit, no epilogue, even N, no BF16 store");
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = SPLIT * tm * tn;
  const int tbase = table ? (int)g.tt[blockIdx.x] : 0;
  const int ntm = claim ? 0x7fffffff : table ? (int)g.tt[blockIdx.x + 1] - tbase
                        : (s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0);
  auto task_id = [&](int i) -> int {
    if (claim) {
      int t;
      while ((t = tma_slot[i & 7]) == -2) {
      }
      return t;
    }
    return table ? (int)g.tt[g.tg + 1 + tbase + i] : s.me + i * s.count;
  };
  auto claim_next = [&]() -> int {
    const unsigned v = atomicAdd(g.claim, 1u);
    return v < (unsigned)ntask ? (int)v : -1;
  };
  const int Q = claim ? 0x7fffffff : ntm * nk;
  if (Q == 0) return;                                              // CTA-uniform
  auto step_valid = [&](int qq) { return claim ? task_id(qq / nk) >= 0 : qq < Q; };
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&tma_bars[0]);
  auto where = [&](int i, int& m0, int& n0) {
    const int task = task_id(i) % (tm * tn);
    m0 = (task / tn) * BM;
    n0 = (task % tn) * BN;
  };
  auto slice_of_task = [&](int i) { return SPLIT == 1 ? 0 : task_id(i) / (tm * tn); };
  auto issue = [&](int q) {
    int m0, n0;
    where(q / nk, m0, n0);
    int kk = slice_of_task(q / nk) * ks + (q % nk) * 64;
    const int st = q % NST;
    const void* mA = g.tA;
    const void* mB = g.tB;
#if POPS4B_KSEG
    if (g.kseg) {
      const int sg = kk / g.kseg;
      kk -= sg * g.kseg;
      if (sg == 1) { mA = g.tA2; mB = g.tB2; }
      else if (sg == 2) { mA = g.tA3; mB = g.tB3; }
    }
#endif
    const unsigned dA = base + st * Cf::STAGE, dB = dA + Cf::A_BYTES, bar = bar0 + 8 * st;
    bar_expect(bar, Cf::STAGE);
    if (AK) tma2d(dA, mA, kk, m0, bar);
    else {
      const int ma = g.astep ? m0 % g.bgrp : m0, ka = g.astep ? kk + (m0 / g.bgrp) * g.astep : kk;
#pragma unroll
      for (int c = 0; c < BM / 64; ++c) tma2d(dA + c * 8192, mA, ma + 64 * c, ka, bar);
    }
    const int bo = g.bgrp ? (m0 / g.bgrp) * g.bstep : 0;            // batch-stacked B rows (0: one batch)
    if (BKC) tma2d(dB, mB, kk, n0 + bo, bar);
    else {
#pragma unroll
      for (int c = 0; c < BN / 64; ++c) tma2d(dB + c * 8192, mB, n0 + 64 * c, kk + bo, bar);
    }
  };
  __syncthreads();                                                 // earlier generic users of smem are done
  unsigned uses0[NST];                                             // completed phases of each stage barrier
#pragma unroll
  for (int st = 0; st < NST; ++st) uses0[st] = POPS4B_PERSIST ? tma_uses[st] : 0u;
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      if (!POPS4B_PERSIST) bar_init(bar0 + 8 * st, 1);
      tma_free[st] = 0u;
    }
    if (claim) {
#if POPS4B_CLAIM_ASYNC
      const unsigned v = atomicAdd(g.claim, 3u);                   // three consecutive tickets, one round trip
      const int t0 = v < (unsigned)ntask ? (int)v : -1, t1 = v + 1u < (unsigned)ntask ? (int)v + 1 : -1;
      const int t2 = v + 2u < (unsigned)ntask ? (int)v + 2 : -1;
#elif POPS4B_CLAIM_AHEAD == 1
      const int t0 = claim_next(), t1 = -2, t2 = -2;
#else
      const int t0 = claim_next(), t1 = t0 >= 0 ? claim_next() : -1, t2 = t1 >= 0 ? claim_next() : -1;
#endif
      tma_slot[0] = t0;
      tma_slot[1] = t1;
      tma_slot[2] = t2;
#pragma unroll
      for (int k = 3; k < 8; ++k) tma_slot[k] = -2;
    }
    if (!POPS4B_PERSIST) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    proxy_fence();                                                 // generic global/shared writes -> TMA
  }
  __syncthreads();
  // P7-B ALIAS (with STAGED): the staged tile aliases the drained operand ring as in ROWS (per-task operand issue),
  // so tiles whose ring and staging together exceed the shared budget still stage their epilogue.
  constexpr bool RING = ROWS || ALIAS;
  static_assert(!RING || (STAGED && Staging<BM, BN>::BYTES <= NST * Cf::STAGE), "ring-aliased staging");
  if (RING && claim) __trap();   // per-task issue needs known task ids (round robin or a table; P9-E: tables allowed)
  if (tid == 0) {
    const int pre = RING ? (nk < NST ? nk : NST) : Q < NST ? Q : NST;
    for (int q = 0; q < pre; ++q)
      if (step_valid(q)) issue(q);
  }
  float acc[Cf::MT][Cf::R];
#pragma unroll
  for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
    for (int j = 0; j < Cf::R; ++j) acc[i][j] = 0.f;
  int q = 0;
#pragma unroll 1
  for (int task_i = 0; task_i < ntm; ++task_i) {
  float serial_previous[Cf::MT][Cf::R];
  if (g.serial_bf16_k) {
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
      for (int j = 0; j < Cf::R; ++j) serial_previous[i][j] = 0.f;
  }
  unsigned claim_ticket = 0u;
  if (claim) {
    if (task_id(task_i) < 0) break;                                // CTA-uniform (one slot value)
#if POPS4B_CLAIM_ASYNC
    // The ticket for sequence task_i + 3 is requested now and published after this task's k steps, so the atomic's
    // latency overlaps the main loop. A CTA's tickets increase, so an exhausted slot is followed by exhausted ones.
    if (tid == 0) claim_ticket = atomicAdd(g.claim, 1u);
#else
    if (tid == 0) {
#if POPS4B_CLAIM_AHEAD == 1
      const int cur = tma_slot[task_i & 7];
      tma_slot[(task_i + 1) & 7] = cur >= 0 ? claim_next() : -1;    // the next task's ticket, before its stages
#else
      const int prev = tma_slot[(task_i + 2) & 7];
      tma_slot[(task_i + 3) & 7] = prev >= 0 ? claim_next() : -1;  // readers of this sequence are >= 1 behind
#endif
    }
#endif
  }
  if constexpr (has_prow<EPI>::value) {
    int m0, n0;
    where(task_i, m0, n0);
    const int ncol = g.N - n0 < BN ? g.N - n0 : BN;
    for (int r = tid; r < BM; r += 256)
      if (m0 + r < g.M) epi.prow((unsigned)((long long)(m0 + r) * g.ldc + n0), (unsigned)ncol);
  }
  if constexpr (has_pre<EPI>::value) {
    int m0, n0;
    where(task_i, m0, n0);
    if ((lane & 3) == 0) {
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) {
        const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < Cf::WTN / 8; j += 4) {
          const int col = n0 + wgn * Cf::WTN + 8 * j;
#pragma unroll
          for (int h = 0; h < 2; ++h)
            if (row + 8 * h < g.M && col < g.N) epi.pre((unsigned)((long long)(row + 8 * h) * g.ldc + col));
        }
      }
    }
  }
#pragma unroll 1
  for (int kk = 0; kk < nk; ++kk, ++q) {
    const int st = q % NST;
    bar_wait(bar0 + 8 * st, (uses0[st] + (unsigned)(q / NST)) & 1u);
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
    wg_fence();
    const unsigned sA = base + st * Cf::STAGE, sB = sA + Cf::A_BYTES;
#pragma unroll
    for (int k16 = 0; k16 < 4; ++k16) {
      const uint64_t db = BKC ? dk(sB + wgn * Cf::WTN * 128 + k16 * 32) : dmn(sB + (wgn * Cf::WTN / 64) * 8192 + k16 * 2048);
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) {
        const int r0 = wgm * Cf::WTM + i * 64;
        const uint64_t da = AK ? dk(sA + r0 * 128 + k16 * 32) : dmn(sA + (r0 / 64) * 8192 + k16 * 2048);
        const bool add = g.serial_bf16_k ? ((kk * 64 + k16 * 16) % g.serial_bf16_k != 0)
                                         : (kk | k16) != 0;
        s101h::MmeWg<Cf::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, add);
      }
      if (g.serial_bf16_k && (kk * 64 + (k16 + 1) * 16) < g.K
          && (kk * 64 + (k16 + 1) * 16) % g.serial_bf16_k == 0) {
        // Drain before reading registers. The next k16 starts from zero;
        // previous BF16 is added only after a complete independent partial.
        wg_commit();
        wg_wait<0>();
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int j = 0; j < Cf::R; ++j)
            serial_previous[i][j] = __bfloat162float(__float2bfloat16_rn(acc[i][j] + serial_previous[i][j]));
      }
    }
    wg_commit();
    wg_wait<1>();                                                  // step q-1 complete: its stage may be refilled
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
#if POPS4B_LAST_ARRIVER
    if (q >= 1 && (claim || q - 1 + NST < Q) && lane == 0) {
      const int sf = (q - 1) % NST;
      // cumulative count per stage (no reset): the 8th arrival of this use of the stage refills it
      if (atomicAdd(&tma_free[sf], 1u) == 8u * (unsigned)((q - 1) / NST) + 7u && step_valid(q - 1 + NST))
        issue(q - 1 + NST);
    }
#else
    __syncthreads();
    if (tid == 0 && q >= 1 && step_valid(q - 1 + NST) && (!RING || q - 1 + NST < (task_i + 1) * nk))
      issue(q - 1 + NST);
#endif
  }                                                                // k steps of task task_i
#if POPS4B_CLAIM_ASYNC
      if (claim && tid == 0)
        tma_slot[(task_i + 3) & 7] = claim_ticket < (unsigned)ntask ? (int)claim_ticket : -1;
#endif
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
      if (g.serial_bf16_k) {
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int j = 0; j < Cf::R; ++j) acc[i][j] += serial_previous[i][j];
      }
#if POPS4B_TRACE
      if (task_i == 0) POPS4B_T(tcall_, 2);
#endif
      int m0, n0;
      where(task_i, m0, n0);
      const bool masked = !PAIR || m0 + BM > g.M || n0 + BN > g.N;
      if constexpr (SPLIT > 1 || RAW) {
        float* cf = g.CF + (long long)slice_of_task(task_i) * g.M * g.ldc;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j) {
            const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int r = row + 8 * h;
              if (!masked || (r < g.M && col < g.N))
                *reinterpret_cast<float2*>(cf + (long long)r * g.ldc + col) =
                    make_float2(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
            }
          }
        }
      } else if constexpr (STAGED) {
        static_assert(PAIR && BN % 8 == 0, "staged epilogue needs 8-column chunks");
        // P9-E HALF: the tile is staged and completed in two column halves (BN / 2 wide), so a 128-column tile's
        // staging fits next to its operand ring; every element, its functor call and its store are unchanged.
        constexpr int SBN = HALF ? BN / 2 : BN;
        static_assert(!HALF || (!RING && SBN % 64 == 0), "half staging: unaliased, 64-column halves");
        using St = Staging<BM, SBN>;
        bf16* st = reinterpret_cast<bf16*>(smem) + ((base - sa(smem)) + (RING ? 0 : NST * Cf::STAGE)) / 2;
        float rs_m = -INFINITY, rs_s = 0.f;                          // row softmax partial (rowpart)
#pragma unroll 1
        for (int hp = 0; hp < (HALF ? 2 : 1); ++hp) {
        const int c_lo = hp * SBN;
        __syncthreads();                                             // the previous task's staged reads are done
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int rl = wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j) {
            const int cl = wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
            if (!HALF || (cl >= c_lo && cl < c_lo + SBN)) {
#pragma unroll
              for (int h = 0; h < 2; ++h)
                *reinterpret_cast<uint32_t*>(st + (rl + 8 * h) * St::LDS + (cl - c_lo)) =
                    s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
            }
          }
        }
        __syncthreads();
#if POPS4B_TRACE
        if (task_i == 0 && hp == 0) POPS4B_T(tcall_, 3);
#endif
        constexpr int CH = SBN / 8;
        constexpr bool CS = has_vec8c<EPI>::value;
        static_assert(!CS || (256 % CH == 0 && BM % 64 == 0), "column sums: fixed column group per thread");
        constexpr int NSL = CS ? BM / 64 : 1;
        float colacc[NSL][8];
        if constexpr (CS) {
#pragma unroll
          for (int sl = 0; sl < NSL; ++sl)
#pragma unroll
            for (int e = 0; e < 8; ++e) colacc[sl][e] = 0.f;
        }
#if POPS4B_EPI_UNROLL == 2
#pragma unroll 2
#else
#pragma unroll kEpiUnroll
#endif
        for (int c = tid; c < BM * CH; c += 256) {
          const int r = c / CH, cc = (c % CH) * 8, gr = m0 + r, gc = n0 + c_lo + cc;
          if (gr < g.M && gc < g.N) {
            const uint4 v = *reinterpret_cast<const uint4*>(st + r * St::LDS + cc);
            const unsigned i = (unsigned)((long long)gr * g.ldc + gc);
            if constexpr (!std::is_same<EPI, NoEpi>::value && !ROWS) {
              float x[8];
              const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                const __nv_bfloat162 pr = *reinterpret_cast<const __nv_bfloat162*>(&w[e]);
                x[2 * e] = __bfloat162float(pr.x);
                x[2 * e + 1] = __bfloat162float(pr.y);
              }
              if constexpr (CS) {
                float cs[8];
                epi.vec8c(i, x, cs);
                const int sl = r / 64;
#pragma unroll
                for (int q = 0; q < NSL; ++q)
                  if (q == sl)
#pragma unroll
                    for (int e = 0; e < 8; ++e) colacc[q][e] += cs[e];
              } else {
                epi.vec8(i, x);
              }
            }
            if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + i) = v;
          }
        }
#if POPS4B_TRACE
        if (task_i == 0) POPS4B_T(tcall_, 4);
#endif
        if constexpr (CS) {
          // slab partials: red[sl][tid][8] in the staging tile, then columns summed over the slab's threads in order
          float* red = reinterpret_cast<float*>(st);
          __syncthreads();                                           // staged tile reads are done
#pragma unroll
          for (int sl = 0; sl < NSL; ++sl)
#pragma unroll
            for (int e = 0; e < 8; ++e) red[(sl * 256 + tid) * 8 + e] = colacc[sl][e];
          __syncthreads();
          for (int o = tid; o < NSL * SBN; o += 256) {
            const int sl = o / SBN, col = o % SBN, grp = col / 8, e = col % 8, gc = n0 + c_lo + col;
            if (gc < g.N && m0 + 64 * sl < g.M) {
              float s0 = 0.f;
#pragma unroll 4
              for (int t = grp; t < 256; t += CH) s0 += red[(sl * 256 + t) * 8 + e];
              g.colpart[(long long)((m0 >> 6) + sl) * g.N + gc] = s0;
            }
          }
          __syncthreads();                                           // reduction buffer (staging tile) is free
        }
        if (BM == 128 && g.rowpart) {
          // two threads per row, SBN / 2 columns each, read from the staged (rounded) tile
          constexpr int HW = SBN / 2;
          static_assert(HW % 8 == 0, "row partials: 8-column chunks");
          const int r = tid >> 1, c0 = n0 + c_lo + (tid & 1) * HW;
          const bf16* rp = st + r * St::LDS + (tid & 1) * HW;
          float mloc = -INFINITY, sloc = 0.f;
#pragma unroll 2
          for (int e8 = 0; e8 < HW / 8; ++e8) {
            const uint4 v = *reinterpret_cast<const uint4*>(rp + 8 * e8);
            const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int e = 0; e < 8; ++e)
              if (c0 + 8 * e8 + e < g.rowvalid)
                mloc = fmaxf(mloc, __uint_as_float((e & 1) ? (w[e >> 1] & 0xffff0000u) : (w[e >> 1] << 16)));
          }
          if (mloc != -INFINITY) {
#pragma unroll 2
            for (int e8 = 0; e8 < HW / 8; ++e8) {
              const uint4 v = *reinterpret_cast<const uint4*>(rp + 8 * e8);
              const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
              for (int e = 0; e < 8; ++e)
                if (c0 + 8 * e8 + e < g.rowvalid)
                  sloc += expf(__uint_as_float((e & 1) ? (w[e >> 1] & 0xffff0000u) : (w[e >> 1] << 16)) - mloc);
            }
          }
          row_merge(rs_m, rs_s, mloc, sloc);
        }
        }                                                            // column halves
        if (BM == 128 && g.rowpart) {
          const float m2 = __shfl_xor_sync(0xffffffffu, rs_m, 1), s2 = __shfl_xor_sync(0xffffffffu, rs_s, 1);
          const int r = tid >> 1;
          if ((tid & 1) == 0 && m0 + r < g.M) {
            float m = rs_m, s = rs_s;
            row_merge(m, s, m2, s2);
            g.rowpart[(long long)(m0 + r) * g.rowtiles + n0 / BN] = make_float2(m, s);
          }
        }
        if constexpr (RING) {
          if constexpr (ROWS) epi.rows(st, (unsigned)St::LDS, (unsigned)m0, (unsigned)(g.M - m0 < BM ? g.M - m0 : BM));
          __syncthreads();                                           // staged reads done: the ring may be refilled
          if (tid == 0 && task_i + 1 < ntm) {
            proxy_fence();
            for (int s2 = 0; s2 < (nk < NST ? nk : NST); ++s2)
              if (step_valid((task_i + 1) * nk + s2)) issue((task_i + 1) * nk + s2);
          }
        }
      } else {
      if (WM == 2 && g.rowpart) {
        // LATCH-MOESUP register-direct row softmax partials: with WM == 2 a row's BN columns sit in one quad of one
        // warpgroup. Each thread rounds its values exactly as the stores do, takes the max then the exp-sum in
        // ascending column order, then the quad merges over lane xor 1 and xor 2 (commutative: lanes agree).
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float m = -INFINITY, s = 0.f;
#pragma unroll
            for (int j = 0; j < Cf::WTN / 8; ++j) {
              const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
              if (col < g.rowvalid) m = fmaxf(m, __bfloat162float(__float2bfloat16_rn(acc[i][4 * j + 2 * h])));
              if (col + 1 < g.rowvalid) m = fmaxf(m, __bfloat162float(__float2bfloat16_rn(acc[i][4 * j + 2 * h + 1])));
            }
            if (m != -INFINITY) {
#pragma unroll
              for (int j = 0; j < Cf::WTN / 8; ++j) {
                const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
                if (col < g.rowvalid) s += expf(__bfloat162float(__float2bfloat16_rn(acc[i][4 * j + 2 * h])) - m);
                if (col + 1 < g.rowvalid) s += expf(__bfloat162float(__float2bfloat16_rn(acc[i][4 * j + 2 * h + 1])) - m);
              }
            }
#pragma unroll
            for (int d = 1; d <= 2; d <<= 1) {
              const float m2 = __shfl_xor_sync(0xffffffffu, m, d), s2 = __shfl_xor_sync(0xffffffffu, s, d);
              row_merge(m, s, m2, s2);
            }
            if ((lane & 3) == 0 && row + 8 * h < g.M)
              g.rowpart[(long long)(row + 8 * h) * g.rowtiles + n0 / BN] = make_float2(m, s);
          }
        }
      }
      if constexpr (!std::is_same<EPI, NoEpi>::value) {
        static_assert(PAIR, "pair epilogue needs even N");
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j) {
            const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int r = row + 8 * h;
              if (r < g.M && col < g.N) {
                const uint32_t packed = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
                epi((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x), __bfloat162float(pair.y));
              }
            }
          }
        }
      }
      if constexpr (!STORE) {
      } else if constexpr (VEC) {
        // 16-byte stores (N % 8 == 0): every chunk is wholly present or wholly absent.
        const int t = lane & 3;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int jj = 0; jj < Cf::WTN / 32; ++jj) {
            const int col = n0 + wgn * Cf::WTN + 8 * (4 * jj + t);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int j0 = 4 * jj;
              const uint4 v = quad_chunk(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
              const int r = row + 8 * h;
              if (!masked || (r < g.M && col < g.N))
                *reinterpret_cast<uint4*>(g.C + (long long)r * g.ldc + col) = v;
            }
          }
        }
      } else
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) {
        const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
        for (int j = 0; j < Cf::WTN / 8; ++j) {
          const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int r = row + 8 * h;
            const uint32_t v = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
            if (!masked) {
              *reinterpret_cast<uint32_t*>(g.C + (long long)r * g.ldc + col) = v;
            } else if (r < g.M && col < g.N) {
              if (PAIR) {
                *reinterpret_cast<uint32_t*>(g.C + (long long)r * g.ldc + col) = v;
              } else {
                unsigned short* c = reinterpret_cast<unsigned short*>(g.C + (long long)r * g.ldc + col);
                c[0] = (unsigned short)(v & 0xffffu);
                if (col + 1 < g.N) c[1] = (unsigned short)(v >> 16);
              }
            }
          }
        }
      }
      }                                                            // SPLIT == 1 (the next task's first k16 overwrites acc)
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
#if POPS4B_TRACE
  POPS4B_T(tcall_, 5);
#endif
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      if (POPS4B_PERSIST) tma_uses[st] = uses0[st] + (unsigned)((q - st + NST - 1) / NST);  // q = steps consumed
      else bar_inval(bar0 + 8 * st);
    }
    proxy_fence();
  }
}

// Entry-resident form (inlined into the kernel: wgmma inside called functions is serialized by ptxas, C7510) and a
// called form for comparison.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false, class EPI = NoEpi,
          bool STORE = true, int SPLIT = 1, bool STAGED = false, bool ROWS = false, bool ALIAS = false,
          bool HALF = false, bool RAW = false>
DEV void tma_phase_inline(const TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0,
                          const EPI epi = EPI{}) {
  tma_body<BM, BN, WM, AK, BKC, NST, PAIR, VEC, EPI, STORE, SPLIT, STAGED, ROWS, ALIAS, HALF, RAW>(g, smem, lo, hi, den,
                                                                                              rot, epi);
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false>
__device__ __noinline__ void tma_phase(const TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  tma_body<BM, BN, WM, AK, BKC, NST, PAIR, VEC>(g, smem, lo, hi, den, rot);
}

}  // namespace pops4b
