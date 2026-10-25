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
// Arithmetic: identical to pops4b_wg_p7a.cuh (FP32 wgmma accumulation, ascending k16, one cvt.rn.bf16x2 rounding).
#pragma once
#include "pops4b_wg_p7a.cuh"

#ifndef P7_RING_RELEASE
#define P7_RING_RELEASE 0   // P7-A: per-stage empty mbarriers replace the CTA-wide barrier of every k step
#endif
#ifndef P7_ACC_REZERO
#define P7_ACC_REZERO 0   // P7-A: accumulators re-zeroed at every tile start (dead, hence reusable, during epilogues)
#endif
#ifndef P7_DIAG_MODE
#define P7_DIAG_MODE 0   // P7-A timing diagnostics only (wrong results): 1 no epilogue/stores, 2 no wgmma, 3 no TMA loads
#endif

namespace p7a4b {

__shared__ __align__(8) unsigned long long tma_bars[8];
__shared__ __align__(8) unsigned long long p7_empty_bars[8];
DEV void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }

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
  unsigned* dyn;          // P7-A dynamic tile claiming: zero-initialized claim counter (nullptr: static slices)
};
__shared__ int p7_dyn_ids[16];   // P7-A: claimed tile of task i at [i % 16] (-1: no tile left)

// Quad transpose of packed BF16 pairs: lane t of a quad holds x[t'] = columns 8 (4 jj + t') + 2 t + {0, 1}; afterwards
// it holds y[t'] = columns 8 (4 jj + t) + 2 t' + {0, 1}, i.e. one 16-byte chunk (8 consecutive columns). Data movement
// only (shfl.xor within the quad); values are unchanged.
#ifndef P7_QUAD_SEL
#define P7_QUAD_SEL 0   // P7-A: select-only quad transpose (two xor rounds, no branches); same data movement
#endif
DEV uint4 quad_chunk(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, int t) {
#if P7_QUAD_SEL
  // 4 x 4 transpose within the quad by two xor-butterfly rounds (masks 1, 2): in round m a lane keeps the element of
  // each pair {k, k | m} whose bit m matches its own and receives the other from lane t ^ m. Afterwards lane t holds
  // y[t'] = x[t] of lane t' (exactly the result of the shuffle/pick form below), with selects only.
  uint32_t a0 = x0, a1 = x1, a2 = x2, a3 = x3;
  {
    const bool h = (t & 1) != 0;
    const uint32_t s0 = h ? a0 : a1, s2 = h ? a2 : a3;
    const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 1), r2 = __shfl_xor_sync(0xffffffffu, s2, 1);
    a0 = h ? r0 : a0; a1 = h ? a1 : r0; a2 = h ? r2 : a2; a3 = h ? a3 : r2;
  }
  {
    const bool h = (t & 2) != 0;
    const uint32_t s0 = h ? a0 : a2, s1 = h ? a1 : a3;
    const uint32_t r0 = __shfl_xor_sync(0xffffffffu, s0, 2), r1 = __shfl_xor_sync(0xffffffffu, s1, 2);
    a0 = h ? r0 : a0; a2 = h ? a2 : r0; a1 = h ? r1 : a1; a3 = h ? a3 : r1;
  }
  return make_uint4(a0, a1, a2, a3);
#endif
  auto pick = [&](int i) { return i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3; };
  const uint32_t r0 = pick(t);                                     // from lane t ^ 0
  const uint32_t r1 = __shfl_xor_sync(0xffffffffu, pick(t ^ 1), 1);
  const uint32_t r2 = __shfl_xor_sync(0xffffffffu, pick(t ^ 2), 2);
  const uint32_t r3 = __shfl_xor_sync(0xffffffffu, pick(t ^ 3), 3);
  auto from = [&](int lane) { const int sft = t ^ lane; return sft == 0 ? r0 : sft == 1 ? r1 : sft == 2 ? r2 : r3; };
  return make_uint4(from(0), from(1), from(2), from(3));
}

// P6-A: register-direct pair epilogue. Epi::kPair functors receive (flat output index of the first element, the two
// BF16-rounded values as float) for every stored column pair; STORE=false omits the product store when only the
// epilogue reads it. NoPairEpi keeps the P-OPS4-B body unchanged.
struct NoPairEpi {
  static constexpr bool kPair = false;
  static constexpr bool kPair2 = false;
  static constexpr int kColSum = 0;
  DEV void operator()(unsigned, float, float) const {}
};

// P7-A staged epilogue detection: pair functors declaring kStaged = true take the staged, coalesced traversal.
template <class E, class = void> struct staged_epi { static constexpr bool value = false; };
template <class E> struct staged_epi<E, decltype((void)E::kStaged, void())> { static constexpr bool value = E::kStaged; };

// P7-A vector epilogue detection: functors declaring kVec = 8 provide loadc / loadv / applyv over 8 consecutive
// columns of one row (16-byte operand loads and result stores; per-column operands loaded once per tile).
template <class E, class = void> struct vec_epi { static constexpr bool value = false; };
template <class E> struct vec_epi<E, decltype((void)E::kVec, void())> { static constexpr bool value = E::kVec == 8; };
// P7-A early vector epilogue loads: functors declaring kVecEarly = true have every operand row of a warp's tile
// share loaded (Epi::loadv) before the tile's main loop, so the loads overlap the matrix work.
template <class E, class = void> struct vec_early { static constexpr bool value = false; };
template <class E> struct vec_early<E, decltype((void)E::kVecEarly, void())> { static constexpr bool value = E::kVecEarly; };
// P7-A pipelined vector epilogues: functors declaring kVecPipe = true have tile t's epilogue batches executed
// between the warpgroup-MMA issue and wait of tile t + 1's main-loop steps (same warps; accumulators of tile t are
// already in the shared staging tile, so the epilogue never touches the in-flight accumulator registers).
template <class E, class = void> struct vec_pipe { static constexpr bool value = false; };
template <class E> struct vec_pipe<E, decltype((void)E::kVecPipe, void())> { static constexpr bool value = E::kVecPipe; };
template <class E, bool V> struct vec_types { struct Lv {}; };
template <class E> struct vec_types<E, true> { using Lv = typename E::Lv; };

// P7-A bulk-store detection: staged pair functors with kSt stored members write them through apply_st into shared
// tiles that leave by asynchronous bulk copies (cp.async.bulk, TMA engine) while the next tile's main loop runs.
template <class E, class = void> struct bulk_epi { static constexpr bool value = false; static constexpr int count = 0; };
template <class E> struct bulk_epi<E, decltype((void)E::kSt, void())> {
  static constexpr bool value = true;
  static constexpr int count = E::kSt;
};
__shared__ __align__(8) unsigned long long p7_side_bars[2];
template <class E, class = void> struct side_epi { static constexpr bool value = false; static constexpr int count = 0; };
template <class E> struct side_epi<E, decltype((void)E::kLdSt, void())> {
  static constexpr bool value = true;
  static constexpr int count = E::kLdSt;
};
template <class E> __host__ __device__ constexpr int st_total_bytes() {
  if constexpr (bulk_epi<E>::value) {
    int t = 0;
    for (int k = 0; k < E::kSt; ++k) t += E::st_alias(k) < 0 ? E::st_bytes(k) : 0;   // aliased tiles: no storage
    return t;
  } else {
    return 0;
  }
}
template <class E> __host__ __device__ constexpr int ld_total_bytes() {
  if constexpr (side_epi<E>::value) {
    int t = 0;
    for (int k = 0; k < E::kLdSt; ++k) t += E::ld_bytes(k);
    return t;
  } else {
    return 0;
  }
}
DEV void bulk_load(void* sdst, const void* gsrc, unsigned bytes, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(sa(sdst)), "l"(gsrc), "r"(bytes), "r"(bar) : "memory");
}
DEV void bulk_row(void* gdst, const void* ssrc, unsigned bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               ::"l"(gdst), "r"(sa(ssrc)), "r"(bytes) : "memory");
}

// P7-A staged epilogue shared bytes after the stage ring + 2048 B slack + column-sum slabs (pops4b layout).
template <int BM, int BN, int WM, int NCS>
__host__ __device__ constexpr int staged_cs_bytes() { return 2 * 4 * (NCS > 0 ? NCS : 1) * Cfg<BM, BN, WM>::WTN * 4; }
template <int BM, int BN, int WM>
__host__ __device__ constexpr int staged_tile_bytes() { return 2 * Cfg<BM, BN, WM>::WTM * (Cfg<BM, BN, WM>::WTN + 8) * 2; }

// P7-A independent-warpgroup detection: staged pair functors declaring kIndep = true run each output tile on one
// warpgroup with its own stage ring (tma_body_indep).
template <class E, class = void> struct indep_epi { static constexpr bool value = false; };
template <class E> struct indep_epi<E, decltype((void)E::kIndep, void())> { static constexpr bool value = E::kIndep; };

// P7-A staged epilogue of one warpgroup's WTM x WTN accumulator tile (MT 64-row slabs), see tma_body.
// st: this warpgroup's padded BF16 sub-tile; cs_smem: its [4 warps][NCS][WTN] column-partial scratch; bar: its
// 128-thread named barrier. Same per-element functor protocol and fixed partial order as the cooperative path.
template <int WTM, int WTN, bool STORE, class Epi>
DEV void staged_tile(const TArgs& g, const Epi& epi, float (&acc)[WTM / 64][WTN / 2], bf16* st, float* cs_smem,
                     int rbase, int cbase, bool masked, int bar, int warp, int lane) {
  constexpr int MT = WTM / 64, NCS = Epi::kColSum, LDS = WTN + 8, PR = WTN / 2, QN = (PR + 31) / 32;
  constexpr int RW = WTM / 4, RB = RW >= 8 ? 8 : RW;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < WTN / 8; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *reinterpret_cast<uint32_t*>(st + (i * 64 + warp * 16 + (lane >> 2) + 8 * h) * LDS + 8 * j + 2 * (lane & 3)) =
            s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
  asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
  float colacc[MT][QN][NCS > 0 ? NCS : 1][2];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int q = 0; q < QN; ++q)
#pragma unroll
      for (int k = 0; k < (NCS > 0 ? NCS : 1); ++k) colacc[i][q][k][0] = colacc[i][q][k][1] = 0.0f;
#pragma unroll 1
  for (int k0 = 0; k0 < RW; k0 += RB) {
    typename Epi::Ld ld[RB][QN];
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int r = rbase + warp + 4 * (k0 + u);
#pragma unroll
      for (int q = 0; q < QN; ++q) {
        const int p = lane + 32 * q, col = cbase + 2 * p;
        if ((PR % 32 == 0 || p < PR) && (!masked || (r < g.M && col < g.N)))
          ld[u][q] = epi.load((unsigned)((long long)r * g.ldc + col));
      }
    }
#pragma unroll
    for (int u = 0; u < RB; ++u) {
      const int rl = warp + 4 * (k0 + u), r = rbase + rl;
#pragma unroll
      for (int q = 0; q < QN; ++q) {
        const int p = lane + 32 * q, col = cbase + 2 * p;
        if ((PR % 32 == 0 || p < PR) && (!masked || (r < g.M && col < g.N))) {
          const uint32_t packed = *reinterpret_cast<const uint32_t*>(st + rl * LDS + 2 * p);
          const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
          if constexpr (STORE) *reinterpret_cast<uint32_t*>(g.C + (long long)r * g.ldc + col) = packed;
          if constexpr (NCS > 0) {
            float cs[NCS * 2];
            epi.apply((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x), __bfloat162float(pair.y),
                      ld[u][q], cs);
            const int sl = MT == 1 ? 0 : (rl >= 64 ? 1 : 0);
#pragma unroll
            for (int k = 0; k < NCS; ++k)
#pragma unroll
              for (int i = 0; i < MT; ++i)
                if (i == sl) {
                  colacc[i][q][k][0] += cs[2 * k];
                  colacc[i][q][k][1] += cs[2 * k + 1];
                }
          } else {
            epi.apply((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x), __bfloat162float(pair.y),
                      ld[u][q]);
          }
        }
      }
    }
  }
  if constexpr (NCS > 0) {
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int q = 0; q < QN; ++q) {
        const int p = lane + 32 * q;
        if (PR % 32 == 0 || p < PR)
#pragma unroll
          for (int k = 0; k < NCS; ++k) {
            cs_smem[(warp * NCS + k) * WTN + 2 * p] = colacc[i][q][k][0];
            cs_smem[(warp * NCS + k) * WTN + 2 * p + 1] = colacc[i][q][k][1];
          }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
      const int slab = (rbase + i * 64) / 64;
      if (warp == 0 && rbase + i * 64 < g.M) {
#pragma unroll 1
        for (int c = lane; c < WTN; c += 32) {
          const int col = cbase + c;
          if (col < g.N) {
#pragma unroll
            for (int k = 0; k < NCS; ++k) {
              const float s0 = ((cs_smem[(0 * NCS + k) * WTN + c] + cs_smem[(1 * NCS + k) * WTN + c])
                                + cs_smem[(2 * NCS + k) * WTN + c]) + cs_smem[(3 * NCS + k) * WTN + c];
              epi.colsum(k)[(long long)slab * g.N + col] = s0;
            }
          }
        }
      }
      asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
    }
  } else {
    asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");   // staged sub-tile reads done before reuse
  }
}

// Stage ring depth of the independent-warpgroup body that fits two rings plus staging in the opt-in limit.
#ifndef P7_INDEP_SMEM
#define P7_INDEP_SMEM 232448   // P7-A: shared bytes the two rings + staging may use (program carve-out rule)
#endif
template <int BM, int BN, int NCS>
__host__ __device__ constexpr int indep_stages() {
  return (2 * 4 * (BM + BN) * 128 + 1024 + 2048 + 2 * 4 * (NCS > 0 ? NCS : 1) * BN * 4 + 2 * BM * (BN + 8) * 2
          <= P7_INDEP_SMEM) ? 4 : (2 * 3 * (BM + BN) * 128 + 1024 + 2048 + 2 * 4 * (NCS > 0 ? NCS : 1) * BN * 4
                             + 2 * BM * (BN + 8) * 2 <= P7_INDEP_SMEM) ? 3 : 2;
}

// P7-A independent warpgroups. Each warpgroup computes whole BM x BN output tiles (tasks w = 2 me + wg of the CTA
// slice) with its own NST-stage TMA ring and mbarriers, waits only on its own 128-thread named barrier, and runs the
// staged epilogue of its tile while the other warpgroup's wgmma main loop proceeds. Same tensor maps, operand
// boxes, k order and per-element epilogue arithmetic as tma_body (one FP32 accumulator per element, ascending k16,
// one BF16 rounding); only the assignment of tiles to warpgroups and the synchronization change.
template <int BM, int BN, bool AK, bool BKC, int NST, bool PAIR, class Epi, bool STORE>
DEV void tma_body_indep(const TArgs& g, float* smem, int lo, int hi, int den, int rot, const Epi& epi) {
  constexpr int MT = BM / 64, WTN = BN, R = BN / 2, A_BYTES = BM * 128, STAGE = (BM + BN) * 128;
  constexpr int NCS = Epi::kColSum;
  static_assert(NST >= 2 && NST <= 4 && BM % 64 == 0 && BM <= 128 && BN % 64 == 0 && BN <= 256, "indep tile");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn;
  const int vme = 2 * s.me + wg, vcount = 2 * s.count;
  const int ntm = vme < ntask ? (ntask - vme + vcount - 1) / vcount : 0;
  const int nk = (g.K + 63) / 64, Q = ntm * nk;
  const unsigned ring0 = (sa(smem) + 1023u) & ~1023u;
  const unsigned base = ring0 + wg * NST * STAGE, bar0 = sa(&tma_bars[0]) + 8 * NST * wg;
  char* const after = reinterpret_cast<char*>(smem) + 2 * NST * STAGE + 2048;
  float* const cs_smem = reinterpret_cast<float*>(after) + wg * 4 * (NCS > 0 ? NCS : 1) * WTN;
  bf16* const st = reinterpret_cast<bf16*>(after + 2 * 4 * (NCS > 0 ? NCS : 1) * WTN * 4) + wg * BM * (WTN + 8);
  const int bar = 12 + wg;                                         // named barrier ids unused elsewhere
  auto where = [&](int i, int& m0, int& n0) {
    const int task = vme + i * vcount;
    m0 = (task / tn) * BM;
    n0 = (task % tn) * BN;
  };
  auto issue = [&](int q) {
    int m0, n0;
    where(q / nk, m0, n0);
    const int kk = (q % nk) * 64, st_ = q % NST;
    const unsigned dA = base + st_ * STAGE, dB = dA + A_BYTES, b = bar0 + 8 * st_;
    if constexpr (P7_DIAG_MODE == 3) {
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
      return;
    }
    bar_expect(b, STAGE);
    if (AK) tma2d(dA, g.tA, kk, m0, b);
    else {
#pragma unroll
      for (int c = 0; c < BM / 64; ++c) tma2d(dA + c * 8192, g.tA, m0 + 64 * c, kk, b);
    }
    if (BKC) tma2d(dB, g.tB, kk, n0, b);
    else {
#pragma unroll
      for (int c = 0; c < BN / 64; ++c) tma2d(dB + c * 8192, g.tB, n0 + 64 * c, kk, b);
    }
  };
  __syncthreads();                                                 // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int b = 0; b < 2 * NST; ++b) bar_init(sa(&tma_bars[0]) + 8 * b, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    proxy_fence();
  }
  __syncthreads();
  if (wt == 0) {
    const int pre = Q < NST ? Q : NST;
    for (int q = 0; q < pre; ++q) issue(q);
  }
  float acc[MT][R];
  int q = 0;
#pragma unroll 1
  for (int task_i = 0; task_i < ntm; ++task_i) {
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st_ = q % NST;
      bar_wait(bar0 + 8 * st_, (unsigned)(q / NST) & 1u);
#pragma unroll
      for (int i = 0; i < MT; ++i) keep(acc[i]);
      wg_fence();
      const unsigned sA = base + st_ * STAGE, sB = sA + A_BYTES;
#pragma unroll
      for (int k16 = 0; k16 < 4; ++k16) {
        const uint64_t db = BKC ? dk(sB + k16 * 32) : dmn(sB + k16 * 2048);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          const uint64_t da = AK ? dk(sA + i * 64 * 128 + k16 * 32) : dmn(sA + i * 8192 + k16 * 2048);
          if constexpr (P7_DIAG_MODE != 2) s101h::MmeWg<WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (kk | k16) != 0);
        }
      }
      wg_commit();
      wg_wait<1>();                                                // step q-1 complete: its stage may be refilled
#pragma unroll
      for (int i = 0; i < MT; ++i) keep(acc[i]);
      asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");     // the warpgroup finished reading stage q-1
      if (wt == 0 && q >= 1 && q - 1 + NST < Q) issue(q - 1 + NST);
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i) keep(acc[i]);
    int m0, n0;
    where(task_i, m0, n0);
    const bool masked = !PAIR || m0 + BM > g.M || n0 + BN > g.N;
    staged_tile<BM, WTN, STORE>(g, epi, acc, st, cs_smem, m0, n0, masked, bar, warp, lane);
  }
  __syncthreads();                                                 // every wgmma read of both rings is complete
  if (tid == 0) {
#pragma unroll
    for (int b = 0; b < 2 * NST; ++b) bar_inval(sa(&tma_bars[0]) + 8 * b);
    proxy_fence();
  }
}

template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false, class Epi = NoPairEpi,
          bool STORE = true, bool DYN = false>
DEV void tma_body(const TArgs& g, float* smem, int lo, int hi, int den, int rot, const Epi& epi = Epi{}) {
  using Cf = Cfg<BM, BN, WM>;
  static_assert(NST >= 2 && NST <= 8, "stages");
  static_assert(!AK || BM <= 256, "TMA box rows");
  static_assert(!BKC || BN <= 256, "TMA box rows");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn;
  // P7-A DYN: every CTA of the slice claims tiles from g.dyn in order (thread 0 claims a tile when it issues the
  // tile's first stage); a CTA that finished earlier work of the phase claims more tiles. Each tile is computed
  // exactly as in the static schedule, so results do not depend on which CTA computes it.
  const int ntm = DYN ? 0x7fffffff : (s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0);
  const int nk = (g.K + 63) / 64, Q = DYN ? 0x7fffffff : ntm * nk;
  if (Q == 0 || ntask == 0) return;                                // CTA-uniform
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&tma_bars[0]);
  auto where = [&](int i, int& m0, int& n0) {
    const int task = DYN ? p7_dyn_ids[i & 15] : s.me + i * s.count;
    m0 = (task / tn) * BM;
    n0 = (task % tn) * BN;
  };
  bool dyn_done = false;                                           // thread 0: no tile left to claim
  auto issue = [&](int q) {
    const int kk = (q % nk) * 64, st = q % NST;
    const unsigned dA = base + st * Cf::STAGE, dB = dA + Cf::A_BYTES, bar = bar0 + 8 * st;
    if constexpr (DYN) {
      if (dyn_done) return;
      if (q % nk == 0) {
        const unsigned c = atomicAdd(g.dyn, 1u);
        p7_dyn_ids[(q / nk) & 15] = c < (unsigned)ntask ? (int)c : -1;
        if (c >= (unsigned)ntask) {
          dyn_done = true;
          bar_arrive(bar);                                         // waiters of this stage pass and read -1
          return;
        }
      }
    }
    int m0, n0;
    where(q / nk, m0, n0);
    if constexpr (P7_DIAG_MODE == 3) {
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
      return;
    }
    bar_expect(bar, Cf::STAGE);
    if (AK) tma2d(dA, g.tA, kk, m0, bar);
    else {
#pragma unroll
      for (int c = 0; c < BM / 64; ++c) tma2d(dA + c * 8192, g.tA, m0 + 64 * c, kk, bar);
    }
    if (BKC) tma2d(dB, g.tB, kk, n0, bar);
    else {
#pragma unroll
      for (int c = 0; c < BN / 64; ++c) tma2d(dB + c * 8192, g.tB, n0 + 64 * c, kk, bar);
    }
  };
  constexpr bool SIDE = side_epi<Epi>::value && staged_epi<Epi>::value;
  constexpr int NSD = side_epi<Epi>::count, SLDS = Cf::WTN + 8;
  // P7-A vector epilogue (kVec): 8 columns per lane, Cf::WTN / 8 lanes per row (a power of two dividing 32).
  constexpr bool VECB = vec_epi<Epi>::value && !SIDE && !bulk_epi<Epi>::value &&
                        (Cf::WTN == 64 || Cf::WTN == 128 || Cf::WTN == 256);
  constexpr bool EARLY = VECB && vec_early<Epi>::value;
  static_assert(!(DYN && (SIDE || EARLY || P7_RING_RELEASE)), "dynamic tiles: static-schedule features only");
  constexpr bool PIPE = VECB && vec_pipe<Epi>::value && !EARLY && !P7_RING_RELEASE && P7_DIAG_MODE == 0;
  constexpr int VLPR = VECB ? Cf::WTN / 8 : 8, VRPI = 32 / VLPR, VNIT = VECB ? Cf::WTM / (4 * VRPI) : 1;
  typename vec_types<Epi, VECB>::Lv early[EARLY ? VNIT : 1];
  __syncthreads();                                                 // earlier generic users of smem are done
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) bar_init(bar0 + 8 * st, 1);
    if constexpr (P7_RING_RELEASE) {
#pragma unroll
      for (int st = 0; st < NST; ++st) bar_init(sa(&p7_empty_bars[st]), NTHREADS);
    }
    if constexpr (SIDE) {
      bar_init(sa(&p7_side_bars[0]), 1);
      bar_init(sa(&p7_side_bars[1]), 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    proxy_fence();                                                 // generic global/shared writes -> TMA
  }
  __syncthreads();
  if (tid == 0) {
    const int pre = Q < NST ? Q : NST;
    for (int q = 0; q < pre; ++q) issue(q);
  }
  // P7-A side-input tiles: after the staged product, column-sum and bulk-store tiles (per warpgroup halves).
  char* const side0 = reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048 +
                      staged_cs_bytes<BM, BN, WM, Epi::kColSum>() + staged_tile_bytes<BM, BN, WM>() +
                      2 * Cf::WTM * SLDS * st_total_bytes<Epi>();
  auto side_buf = [&](int k) -> char* {
    char* q_ = side0;
    if constexpr (SIDE) {
      for (int k2 = 0; k2 < k; ++k2) q_ += 2 * Cf::WTM * SLDS * Epi::ld_bytes(k2);
      return q_ + wg * Cf::WTM * SLDS * Epi::ld_bytes(k);
    } else {
      return q_;
    }
  };
  unsigned side_phase = 0;
  float acc[Cf::MT][Cf::R];
#pragma unroll
  for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
    for (int j = 0; j < Cf::R; ++j) acc[i][j] = 0.f;
  int q = 0;
  if constexpr (PIPE) {
    constexpr int NCS = Epi::kColSum, LDS = Cf::WTN + 8, LPR = Cf::WTN / 8, RPI = 32 / LPR;
    constexpr int NIT = Cf::WTM / (4 * RPI), RB = NIT >= Epi::kVecRB ? Epi::kVecRB : NIT, NBT = NIT / RB;
    static_assert(NIT * 4 * RPI == Cf::WTM && NIT % RB == 0, "vector epilogue strips");
    float* const cs_smem = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048)
                           + wg * 4 * (NCS > 0 ? NCS : 1) * Cf::WTN;
    bf16* const stg = reinterpret_cast<bf16*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048 +
                                              staged_cs_bytes<BM, BN, WM, NCS>()) + wg * Cf::WTM * LDS;
    const int lr = lane / LPR, lc = (lane % LPR) * 8;
    int pm0 = 0, pn0 = 0;
    bool pmask = false, pend = false;
    typename Epi::Lc lcv;
    float colacc[Cf::MT][NCS > 0 ? NCS : 1][8];
    // One batch (RB strips) of the pending tile's vector epilogue: identical traversal/arithmetic to VECB.
    auto batch = [&](int b) {
      const int rbase = pm0 + wgm * Cf::WTM, col = pn0 + wgn * Cf::WTN + lc;
      const bool col_ok = !pmask || col < g.N;
      typename Epi::Lv ld[RB];
#pragma unroll
      for (int u = 0; u < RB; ++u) {
        const int rl = ((b * RB + u) * 4 + warp) * RPI + lr, r = rbase + rl;
        if (col_ok && (!pmask || r < g.M)) ld[u] = epi.loadv((unsigned)((long long)r * g.ldc + col));
      }
#pragma unroll
      for (int u = 0; u < RB; ++u) {
        const int rl = ((b * RB + u) * 4 + warp) * RPI + lr, r = rbase + rl;
        if (col_ok && (!pmask || r < g.M)) {
          const uint4 packed = *reinterpret_cast<const uint4*>(stg + rl * LDS + lc);
          if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + (long long)r * g.ldc + col) = packed;
          float gv[8];
          {
            const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              gv[2 * e] = __bfloat162float(p2[e].x);
              gv[2 * e + 1] = __bfloat162float(p2[e].y);
            }
          }
          const unsigned i0 = (unsigned)((long long)r * g.ldc + col);
          if constexpr (NCS > 0) {
            float cs[NCS * 8];
            epi.applyv(i0, gv, ld[u], lcv, cs);
            const int sl = Cf::MT == 1 ? 0 : (rl >= 64 ? 1 : 0);
#pragma unroll
            for (int i = 0; i < Cf::MT; ++i)
              if (i == sl)
#pragma unroll
                for (int k = 0; k < NCS; ++k)
#pragma unroll
                  for (int e = 0; e < 8; ++e) colacc[i][k][e] += cs[8 * k + e];
          } else {
            epi.applyv(i0, gv, ld[u], lcv);
          }
        }
      }
    };
    // Column partials of the pending tile, then the staging tile is free (warpgroup barrier in both paths).
    auto finish = [&]() {
      if constexpr (NCS > 0) {
        const int rbase = pm0 + wgm * Cf::WTM;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
#pragma unroll
          for (int k = 0; k < NCS; ++k)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              float v = colacc[i][k][e];
#pragma unroll
              for (int o = LPR; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
              if (lr == 0) cs_smem[(warp * NCS + k) * Cf::WTN + lc + e] = v;
            }
          asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
          const int slab = (rbase + i * 64) / 64;
          if (warp == 0 && rbase + i * 64 < g.M) {
#pragma unroll 1
            for (int c = lane; c < Cf::WTN; c += 32) {
              const int cc = pn0 + wgn * Cf::WTN + c;
              if (cc < g.N) {
#pragma unroll
                for (int k = 0; k < NCS; ++k) {
                  const float s0 = ((cs_smem[(0 * NCS + k) * Cf::WTN + c] + cs_smem[(1 * NCS + k) * Cf::WTN + c])
                                    + cs_smem[(2 * NCS + k) * Cf::WTN + c]) + cs_smem[(3 * NCS + k) * Cf::WTN + c];
                  epi.colsum(k)[(long long)slab * g.N + cc] = s0;
                }
              }
            }
          }
          asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
        }
      } else {
        asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
      }
      pend = false;
    };
#pragma unroll 1
    for (int task_i = 0; task_i < ntm; ++task_i) {
      if constexpr (DYN) {
        bar_wait(bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u);
        if (p7_dyn_ids[task_i & 15] < 0) break;
      }
      int done_b = 0;
      if constexpr (P7_ACC_REZERO) {
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int j = 0; j < Cf::R; ++j) acc[i][j] = 0.f;
      }
#pragma unroll 1
      for (int kk = 0; kk < nk; ++kk, ++q) {
        const int st = q % NST;
        bar_wait(bar0 + 8 * st, (unsigned)(q / NST) & 1u);
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
            s101h::MmeWg<Cf::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (kk | k16) != 0);
          }
        }
        wg_commit();
        if (pend && done_b < NBT) {                                // previous tile's epilogue under this step's MMA
          batch(done_b);
          ++done_b;
        }
        wg_wait<1>();
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
        __syncthreads();
        if (tid == 0 && q >= 1 && q - 1 + NST < Q) issue(q - 1 + NST);
      }
      if (pend) {
#pragma unroll 1
        for (; done_b < NBT; ++done_b) batch(done_b);
        finish();
      }
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
      int m0, n0;
      where(task_i, m0, n0);
      pm0 = m0;
      pn0 = n0;
      pmask = !PAIR || m0 + BM > g.M || n0 + BN > g.N;
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int j = 0; j < Cf::WTN / 8; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h)
            *reinterpret_cast<uint32_t*>(stg + (i * 64 + warp * 16 + (lane >> 2) + 8 * h) * LDS + 8 * j + 2 * (lane & 3)) =
                s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
      asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
      {
        const int col = pn0 + wgn * Cf::WTN + lc;
        if (!pmask || col < g.N) lcv = epi.loadc((unsigned)col);
      }
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
        for (int k = 0; k < (NCS > 0 ? NCS : 1); ++k)
#pragma unroll
          for (int e = 0; e < 8; ++e) colacc[i][k][e] = 0.0f;
      pend = true;
    }
    if (pend) {
#pragma unroll 1
      for (int b = 0; b < NBT; ++b) batch(b);
      finish();
    }
  }
#pragma unroll 1
  for (int task_i = 0; !PIPE && task_i < ntm; ++task_i) {
  if constexpr (DYN) {
    // The tile of this task is known once its first stage completes (or was released without data: -1).
    bar_wait(bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u);
    if (p7_dyn_ids[task_i & 15] < 0) break;
  }
  bool side_now = false;
  if constexpr (P7_ACC_REZERO) {
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
      for (int j = 0; j < Cf::R; ++j) acc[i][j] = 0.f;
  }
  if constexpr (EARLY) {
    // P7-A: this tile's epilogue operand rows, loaded before its main loop (values read before the epilogue
    // exactly as at epilogue time: no member of the phase writes an epilogue operand).
    int em0, en0;
    where(task_i, em0, en0);
    const bool emasked = !PAIR || em0 + BM > g.M || en0 + BN > g.N;
    const int ecol = en0 + wgn * Cf::WTN + (lane % VLPR) * 8;
#pragma unroll
    for (int u = 0; u < VNIT; ++u) {
      const int r = em0 + wgm * Cf::WTM + (u * 4 + warp) * VRPI + lane / VLPR;
      if (!emasked || (ecol < g.N && r < g.M)) early[u] = epi.loadv((unsigned)((long long)r * g.ldc + ecol));
    }
  }
  if constexpr (SIDE) {
    // P7-A: this tile's identity-indexed epilogue side inputs leave global memory by bulk copies now and land in
    // shared tiles while the main loop runs (the previous tile's epilogue has finished reading them).
    int sm0, sn0;
    where(task_i, sm0, sn0);
    side_now = !(sm0 + BM > g.M || sn0 + BN > g.N);
    if (side_now) {
      const int wt = tid & 127;
      const unsigned sb = sa(&p7_side_bars[wg]);
      async_fence();
      if (wt == 0) bar_expect(sb, Cf::WTM * Cf::WTN * ld_total_bytes<Epi>());
      asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
      if (wt < Cf::WTM) {
        // Aliased store tiles: this thread's previous bulk stores of the row have finished reading it.
        if constexpr (bulk_epi<Epi>::value) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        const long long row = (long long)(sm0 + wgm * Cf::WTM + wt) * g.ldc + sn0 + wgn * Cf::WTN;
#pragma unroll
        for (int k = 0; k < NSD; ++k)
          bulk_load(side_buf(k) + wt * SLDS * Epi::ld_bytes(k), epi.ld_ptr(k) + row * Epi::ld_bytes(k),
                    Cf::WTN * Epi::ld_bytes(k), sb);
      }
    }
  }
#pragma unroll 1
  for (int kk = 0; kk < nk; ++kk, ++q) {
    const int st = q % NST;
    bar_wait(bar0 + 8 * st, (unsigned)(q / NST) & 1u);
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
        if constexpr (P7_DIAG_MODE != 2) s101h::MmeWg<Cf::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (kk | k16) != 0);
      }
    }
    wg_commit();
    wg_wait<1>();                                                  // step q-1 complete: its stage may be refilled
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
    if constexpr (P7_RING_RELEASE) {
      // This thread's reads of stage q-1 are complete; only the refilling thread waits for every release.
      if (q >= 1) bar_arrive(sa(&p7_empty_bars[(q - 1) % NST]));
      if (tid == 0 && q >= 1 && q - 1 + NST < Q) {
        bar_wait(sa(&p7_empty_bars[(q - 1) % NST]), (unsigned)((q - 1) / NST) & 1u);
        issue(q - 1 + NST);
      }
    } else {
      __syncthreads();
      if (tid == 0 && q >= 1 && q - 1 + NST < Q) issue(q - 1 + NST);
    }
  }                                                                // k steps of task task_i
      wg_wait<0>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
      int m0, n0;
      where(task_i, m0, n0);
      const bool masked = !PAIR || m0 + BM > g.M || n0 + BN > g.N;
      if constexpr (P7_DIAG_MODE == 1) {
      } else if constexpr (VECB) {
        // P7-A vector staged epilogue. The warpgroup's rounded BF16 accumulator pairs go to the padded shared
        // sub-tile (as the staged epilogue); then lane l of a warp owns columns 8 (l % LPR) .. +7 of row l / LPR of
        // each RPI-row strip: one 16-byte shared read and one 16-byte product store per 8 outputs, 16-byte operand
        // loads (Epi::loadv), per-column operands loaded once per tile (Epi::loadc), 16-byte member stores
        // (Epi::applyv). Per-element arithmetic is unchanged. Column partials: per lane over its rows, xor shuffles
        // over the lanes of one column chunk, then the four warps in fixed order (FP32 reassociation).
        static_assert(PAIR, "vector epilogues need an even N");
        constexpr int NCS = Epi::kColSum, LDS = Cf::WTN + 8, LPR = Cf::WTN / 8, RPI = 32 / LPR;
        constexpr int NIT = Cf::WTM / (4 * RPI), RB = EARLY ? NIT : (NIT >= Epi::kVecRB ? Epi::kVecRB : NIT);
        static_assert(NIT * 4 * RPI == Cf::WTM && NIT % RB == 0, "vector epilogue strips");
        float* const cs_smem = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048)
                               + wg * 4 * (NCS > 0 ? NCS : 1) * Cf::WTN;
        bf16* const st = reinterpret_cast<bf16*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048 +
                                                 staged_cs_bytes<BM, BN, WM, NCS>()) + wg * Cf::WTM * LDS;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
              *reinterpret_cast<uint32_t*>(st + (i * 64 + warp * 16 + (lane >> 2) + 8 * h) * LDS + 8 * j + 2 * (lane & 3)) =
                  s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
        asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
        const int rbase = m0 + wgm * Cf::WTM, lr = lane / LPR, lc = (lane % LPR) * 8;
        const int col = n0 + wgn * Cf::WTN + lc;
        const bool col_ok = !masked || col < g.N;
        typename Epi::Lc lcv;
        if (col_ok) lcv = epi.loadc((unsigned)col);
        float colacc[Cf::MT][NCS > 0 ? NCS : 1][8];
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int k = 0; k < (NCS > 0 ? NCS : 1); ++k)
#pragma unroll
            for (int e = 0; e < 8; ++e) colacc[i][k][e] = 0.0f;
#pragma unroll 1
        for (int it0 = 0; it0 < NIT; it0 += RB) {
          typename Epi::Lv ld[EARLY ? 1 : RB];
          if constexpr (!EARLY) {
#pragma unroll
            for (int u = 0; u < RB; ++u) {
              const int rl = ((it0 + u) * 4 + warp) * RPI + lr, r = rbase + rl;
              if (col_ok && (!masked || r < g.M)) ld[u] = epi.loadv((unsigned)((long long)r * g.ldc + col));
            }
          }
#pragma unroll
          for (int u = 0; u < RB; ++u) {
            const int rl = ((it0 + u) * 4 + warp) * RPI + lr, r = rbase + rl;
            if (col_ok && (!masked || r < g.M)) {
              const uint4 packed = *reinterpret_cast<const uint4*>(st + rl * LDS + lc);
              if constexpr (STORE) *reinterpret_cast<uint4*>(g.C + (long long)r * g.ldc + col) = packed;
              float gv[8];
              {
                const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                  gv[2 * e] = __bfloat162float(p2[e].x);
                  gv[2 * e + 1] = __bfloat162float(p2[e].y);
                }
              }
              const unsigned i0 = (unsigned)((long long)r * g.ldc + col);
              if constexpr (NCS > 0) {
                float cs[NCS * 8];
                if constexpr (EARLY) epi.applyv(i0, gv, early[u], lcv, cs);
                else epi.applyv(i0, gv, ld[u], lcv, cs);
                const int sl = Cf::MT == 1 ? 0 : (rl >= 64 ? 1 : 0);
#pragma unroll
                for (int i = 0; i < Cf::MT; ++i)
                  if (i == sl)
#pragma unroll
                    for (int k = 0; k < NCS; ++k)
#pragma unroll
                      for (int e = 0; e < 8; ++e) colacc[i][k][e] += cs[8 * k + e];
              } else {
                if constexpr (EARLY) epi.applyv(i0, gv, early[u], lcv);
                else epi.applyv(i0, gv, ld[u], lcv);
              }
            }
          }
        }
        if constexpr (NCS > 0) {
#pragma unroll
          for (int i = 0; i < Cf::MT; ++i) {
#pragma unroll
            for (int k = 0; k < NCS; ++k)
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                float v = colacc[i][k][e];
#pragma unroll
                for (int o = LPR; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if (lr == 0) cs_smem[(warp * NCS + k) * Cf::WTN + lc + e] = v;
              }
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
            const int slab = (rbase + i * 64) / 64;
            if (warp == 0 && rbase + i * 64 < g.M) {
#pragma unroll 1
              for (int c = lane; c < Cf::WTN; c += 32) {
                const int cc = n0 + wgn * Cf::WTN + c;
                if (cc < g.N) {
#pragma unroll
                  for (int k = 0; k < NCS; ++k) {
                    const float s0 = ((cs_smem[(0 * NCS + k) * Cf::WTN + c] + cs_smem[(1 * NCS + k) * Cf::WTN + c])
                                      + cs_smem[(2 * NCS + k) * Cf::WTN + c]) + cs_smem[(3 * NCS + k) * Cf::WTN + c];
                    epi.colsum(k)[(long long)slab * g.N + cc] = s0;
                  }
                }
              }
            }
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
          }
        } else {
          asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");   // staged sub-tile reads done
        }
      } else if constexpr (staged_epi<Epi>::value) {
        // P7-A staged epilogue. The warpgroup writes its rounded BF16 accumulator pairs (exactly the values the
        // register-direct path passes to the functor) into a padded shared sub-tile, then traverses it row-major:
        // lane -> column pair, warp -> rows warp, warp + 4, ... Side-operand loads, member stores and the product
        // store of a warp touch contiguous row segments (coalesced) instead of 8 rows x 16 bytes per instruction.
        // Per-element arithmetic is unchanged; column partials per 64-row slab are summed per lane over its rows,
        // then over the four warps (fixed order) - an FP32 reassociation of the register-direct partials.
        static_assert(PAIR, "staged epilogues need an even N");
        constexpr int NCS = Epi::kColSum, LDS = Cf::WTN + 8, PR = Cf::WTN / 2, QN = (PR + 31) / 32;
        constexpr int RW = Cf::WTM / 4, RB = RW >= 8 ? 8 : RW;   // rows per warp; loads of RB rows in flight
        float* const cs_smem = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048)
                               + wg * 4 * (NCS > 0 ? NCS : 1) * Cf::WTN;
        bf16* const st = reinterpret_cast<bf16*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048 +
                                                 staged_cs_bytes<BM, BN, WM, NCS>()) + wg * Cf::WTM * LDS;
        constexpr bool BULK = bulk_epi<Epi>::value;
        constexpr int NSK = bulk_epi<Epi>::count;
        char* sk[NSK > 0 ? NSK : 1];
        {
          char* q_ = reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048 + staged_cs_bytes<BM, BN, WM, NCS>() +
                     staged_tile_bytes<BM, BN, WM>();
#pragma unroll
          for (int k = 0; k < (NSK > 0 ? NSK : 1); ++k) {
            if constexpr (BULK) {
              if (Epi::st_alias(k) >= 0) {
                sk[k] = side_buf(Epi::st_alias(k));              // same thread reads then writes each element
              } else {
                sk[k] = q_ + wg * Cf::WTM * LDS * Epi::st_bytes(k);
                q_ += 2 * Cf::WTM * LDS * Epi::st_bytes(k);
              }
            } else {
              sk[k] = nullptr;
            }
          }
        }
        const bool bulk = BULK && !masked;
        const int wt = tid & 127;
        char* sside[NSD > 0 ? NSD : 1];
#pragma unroll
        for (int k = 0; k < (NSD > 0 ? NSD : 1); ++k) sside[k] = side_buf(k);
        if constexpr (SIDE) {
          if (side_now) {
            bar_wait(sa(&p7_side_bars[wg]), side_phase);
            side_phase ^= 1u;
          }
        }
        if constexpr (BULK) {
          // The previous tile's bulk copies have finished reading the staged product and member tiles.
          if (wt < Cf::WTM) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
          asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
        }
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
              *reinterpret_cast<uint32_t*>(st + (i * 64 + warp * 16 + (lane >> 2) + 8 * h) * LDS + 8 * j + 2 * (lane & 3)) =
                  s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
        asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
        float colacc[Cf::MT][QN][NCS > 0 ? NCS : 1][2];
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
          for (int q = 0; q < QN; ++q)
#pragma unroll
            for (int k = 0; k < (NCS > 0 ? NCS : 1); ++k) colacc[i][q][k][0] = colacc[i][q][k][1] = 0.0f;
        const int rbase = m0 + wgm * Cf::WTM, cbase = n0 + wgn * Cf::WTN;
#pragma unroll 1
        for (int k0 = 0; k0 < RW; k0 += RB) {
          typename Epi::Ld ld[RB][QN];
#pragma unroll
          for (int u = 0; u < RB; ++u) {
            const int r = rbase + warp + 4 * (k0 + u);
#pragma unroll
            for (int q = 0; q < QN; ++q) {
              const int p = lane + 32 * q, col = cbase + 2 * p;
              if ((PR % 32 == 0 || p < PR) && (!masked || (r < g.M && col < g.N))) {
                if constexpr (SIDE) {
                  if (side_now)
                    ld[u][q] = epi.load_st((unsigned)((long long)r * g.ldc + col),
                                           (unsigned)((warp + 4 * (k0 + u)) * LDS + 2 * p), sside);
                  else
                    ld[u][q] = epi.load((unsigned)((long long)r * g.ldc + col));
                } else {
                  ld[u][q] = epi.load((unsigned)((long long)r * g.ldc + col));
                }
              }
            }
          }
#pragma unroll
          for (int u = 0; u < RB; ++u) {
            const int rl = warp + 4 * (k0 + u), r = rbase + rl;
#pragma unroll
            for (int q = 0; q < QN; ++q) {
              const int p = lane + 32 * q, col = cbase + 2 * p;
              if ((PR % 32 == 0 || p < PR) && (!masked || (r < g.M && col < g.N))) {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(st + rl * LDS + 2 * p);
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
                if constexpr (STORE) {
                  if (!bulk) *reinterpret_cast<uint32_t*>(g.C + (long long)r * g.ldc + col) = packed;
                }
                const unsigned i0 = (unsigned)((long long)r * g.ldc + col), loc = (unsigned)(rl * LDS + 2 * p);
                if constexpr (NCS > 0) {
                  float cs[NCS * 2];
                  if constexpr (BULK) {
                    if (bulk) epi.apply_st(i0, loc, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q], cs, sk);
                    else epi.apply(i0, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q], cs);
                  } else {
                    epi.apply(i0, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q], cs);
                  }
                  const int sl = Cf::MT == 1 ? 0 : (rl >= 64 ? 1 : 0);
#pragma unroll
                  for (int k = 0; k < NCS; ++k)
#pragma unroll
                    for (int i = 0; i < Cf::MT; ++i)
                      if (i == sl) {
                        colacc[i][q][k][0] += cs[2 * k];
                        colacc[i][q][k][1] += cs[2 * k + 1];
                      }
                } else {
                  if constexpr (BULK) {
                    if (bulk) epi.apply_st(i0, loc, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q], sk);
                    else epi.apply(i0, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q]);
                  } else {
                    epi.apply(i0, __bfloat162float(pair.x), __bfloat162float(pair.y), ld[u][q]);
                  }
                }
              }
            }
          }
        }
        if constexpr (BULK) {
          if (bulk) {
            async_fence();                                         // generic smem writes -> async proxy reads
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
            if (wt < Cf::WTM) {
              const long long row = (long long)(rbase + wt) * g.ldc + cbase;
              if constexpr (STORE) bulk_row(g.C + row, st + wt * LDS, Cf::WTN * 2);
#pragma unroll
              for (int k = 0; k < NSK; ++k)
                bulk_row(epi.st_ptr(k) + row * Epi::st_bytes(k), sk[k] + wt * LDS * Epi::st_bytes(k),
                         Cf::WTN * Epi::st_bytes(k));
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
          }
        }
        if constexpr (NCS > 0) {
#pragma unroll
          for (int i = 0; i < Cf::MT; ++i) {
#pragma unroll
            for (int q = 0; q < QN; ++q) {
              const int p = lane + 32 * q;
              if (PR % 32 == 0 || p < PR)
#pragma unroll
                for (int k = 0; k < NCS; ++k) {
                  cs_smem[(warp * NCS + k) * Cf::WTN + 2 * p] = colacc[i][q][k][0];
                  cs_smem[(warp * NCS + k) * Cf::WTN + 2 * p + 1] = colacc[i][q][k][1];
                }
            }
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
            const int slab = (rbase + i * 64) / 64;
            if (warp == 0 && rbase + i * 64 < g.M) {
#pragma unroll 1
              for (int c = lane; c < Cf::WTN; c += 32) {
                const int col = cbase + c;
                if (col < g.N) {
#pragma unroll
                  for (int k = 0; k < NCS; ++k) {
                    const float s0 = ((cs_smem[(0 * NCS + k) * Cf::WTN + c] + cs_smem[(1 * NCS + k) * Cf::WTN + c])
                                      + cs_smem[(2 * NCS + k) * Cf::WTN + c]) + cs_smem[(3 * NCS + k) * Cf::WTN + c];
                    epi.colsum(k)[(long long)slab * g.N + col] = s0;
                  }
                }
              }
            }
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
          }
        } else {
          asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");   // staged sub-tile reads done
        }
      } else if constexpr (Epi::kPair2) {
        // P6-A: loads of a chunk of pairs are issued before any of their stores (Epi::load / Epi::apply), so
        // global-memory latency overlaps across pairs instead of serializing behind possibly-aliasing stores.
        // Epi::kColSum > 0: apply also returns kColSum per-element values whose column sums over each 64-row
        // slab are written to Epi::colsum(k)[slab][col] (fixed order: thread rows h, lane shuffles 4/8/16, warps
        // 0..3 of the warpgroup), replacing a separate column-reduction pass over a stored copy.
        static_assert(PAIR, "pair epilogues need an even N");
        constexpr int JC = Cf::WTN / 8 >= 4 ? 4 : Cf::WTN / 8;
        constexpr int NCS = Epi::kColSum;
        float* const cs_smem = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + Cf::STAGE * NST + 2048)
                               + wg * 4 * (NCS > 0 ? NCS : 1) * Cf::WTN;
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int j0 = 0; j0 < Cf::WTN / 8; j0 += JC) {
            typename Epi::Ld ld[JC][2];
#pragma unroll
            for (int jj = 0; jj < JC; ++jj) {
              const int col = n0 + wgn * Cf::WTN + 8 * (j0 + jj) + 2 * (lane & 3);
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const int r = row + 8 * h;
                if (!masked || (r < g.M && col < g.N)) ld[jj][h] = epi.load((unsigned)((long long)r * g.ldc + col));
              }
            }
            float colpart[JC][NCS > 0 ? NCS : 1][2];
#pragma unroll
            for (int jj = 0; jj < JC; ++jj)
#pragma unroll
              for (int k = 0; k < (NCS > 0 ? NCS : 1); ++k) colpart[jj][k][0] = colpart[jj][k][1] = 0.0f;
#pragma unroll
            for (int jj = 0; jj < JC; ++jj) {
              const int col = n0 + wgn * Cf::WTN + 8 * (j0 + jj) + 2 * (lane & 3);
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const int r = row + 8 * h;
                if (!masked || (r < g.M && col < g.N)) {
                  const uint32_t packed = s101h::bpk(acc[i][4 * (j0 + jj) + 2 * h], acc[i][4 * (j0 + jj) + 2 * h + 1]);
                  const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
                  if constexpr (NCS > 0) {
                    float cs[NCS * 2];
                    epi.apply((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x),
                              __bfloat162float(pair.y), ld[jj][h], cs);
#pragma unroll
                    for (int k = 0; k < NCS; ++k) {
                      colpart[jj][k][0] += cs[2 * k];
                      colpart[jj][k][1] += cs[2 * k + 1];
                    }
                  } else {
                    epi.apply((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x),
                              __bfloat162float(pair.y), ld[jj][h]);
                  }
                }
              }
            }
            if constexpr (NCS > 0) {
#pragma unroll
              for (int jj = 0; jj < JC; ++jj)
#pragma unroll
                for (int k = 0; k < NCS; ++k)
#pragma unroll
                  for (int e = 0; e < 2; ++e) {
                    float v = colpart[jj][k][e];
                    v += __shfl_xor_sync(0xffffffffu, v, 4);
                    v += __shfl_xor_sync(0xffffffffu, v, 8);
                    v += __shfl_xor_sync(0xffffffffu, v, 16);
                    if ((lane >> 2) == 0)
                      cs_smem[(warp * NCS + k) * Cf::WTN + 8 * (j0 + jj) + 2 * (lane & 3) + e] = v;
                  }
            }
          }
          if constexpr (NCS > 0) {
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
            const int slab = (m0 + wgm * Cf::WTM + i * 64) / 64;
            if (warp == 0 && m0 + wgm * Cf::WTM + i * 64 < g.M) {
#pragma unroll 1
              for (int c = lane; c < Cf::WTN; c += 32) {
                const int col = n0 + wgn * Cf::WTN + c;
                if (col < g.N) {
#pragma unroll
                  for (int k = 0; k < NCS; ++k) {
                    const float s0 = ((cs_smem[(0 * NCS + k) * Cf::WTN + c] + cs_smem[(1 * NCS + k) * Cf::WTN + c])
                                      + cs_smem[(2 * NCS + k) * Cf::WTN + c]) + cs_smem[(3 * NCS + k) * Cf::WTN + c];
                    epi.colsum(k)[(long long)slab * g.N + col] = s0;
                  }
                }
              }
            }
            asm volatile("bar.sync %0, 128;" ::"r"(8 + wg) : "memory");
          }
        }
      } else if constexpr (Epi::kPair) {
        static_assert(PAIR, "pair epilogues need an even N");
#pragma unroll
        for (int i = 0; i < Cf::MT; ++i) {
          const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
          for (int j = 0; j < Cf::WTN / 8; ++j) {
            const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int r = row + 8 * h;
              if (!masked || (r < g.M && col < g.N)) {
                const uint32_t packed = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
                epi((unsigned)((long long)r * g.ldc + col), __bfloat162float(pair.x), __bfloat162float(pair.y));
              }
            }
          }
        }
      }
      if constexpr (!STORE || staged_epi<Epi>::value || VECB || P7_DIAG_MODE == 1) {
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
  }
  if constexpr (SIDE) {
    __syncthreads();
    if (tid == 0) {
      bar_inval(sa(&p7_side_bars[0]));
      bar_inval(sa(&p7_side_bars[1]));
    }
  }
  if constexpr (bulk_epi<Epi>::value) {
    // Bulk stores complete (writes, not only reads) and ordered before the phase's grid barrier.
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  __syncthreads();                                                 // every wgmma read of the stages is complete
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) bar_inval(bar0 + 8 * st);
    if constexpr (P7_RING_RELEASE) {
#pragma unroll
      for (int st = 0; st < NST; ++st) bar_inval(sa(&p7_empty_bars[st]));
    }
    proxy_fence();
  }
}

// P6-A split-K partial products: task t = (tile, slice) with slice = t % KS covers k steps [slice * nk, (slice + 1) *
// nk) of K / 64 = KS * nk steps; the FP32 accumulator (no rounding) is stored to P[slice][row][col] (row-major M x N
// per slice). A separate fixed-order sum over slices rounds once to BF16. Same main loop and staging as tma_body.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int KS>
DEV void tma_split_body(const TArgs& g, float* P, float* smem, int lo, int hi, int den, int rot) {
  using Cf = Cfg<BM, BN, WM>;
  static_assert(NST >= 2 && NST <= 8, "stages");
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn * KS;
  const int ntm = s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0;
  const int nk = (g.K / 64) / KS, Q = ntm * nk;
  if (Q == 0) return;                                              // CTA-uniform
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  const unsigned base = (sa(smem) + 1023u) & ~1023u, bar0 = sa(&tma_bars[0]);
  auto where = [&](int i, int& m0, int& n0, int& sl) {
    const int task = s.me + i * s.count, tile = task / KS;
    sl = task % KS;
    m0 = (tile / tn) * BM;
    n0 = (tile % tn) * BN;
  };
  auto issue = [&](int q) {
    int m0, n0, sl;
    where(q / nk, m0, n0, sl);
    const int kk = (sl * nk + q % nk) * 64, st = q % NST;
    const unsigned dA = base + st * Cf::STAGE, dB = dA + Cf::A_BYTES, bar = bar0 + 8 * st;
    if constexpr (P7_DIAG_MODE == 3) {
      asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
      return;
    }
    bar_expect(bar, Cf::STAGE);
    if (AK) tma2d(dA, g.tA, kk, m0, bar);
    else {
#pragma unroll
      for (int c = 0; c < BM / 64; ++c) tma2d(dA + c * 8192, g.tA, m0 + 64 * c, kk, bar);
    }
    if (BKC) tma2d(dB, g.tB, kk, n0, bar);
    else {
#pragma unroll
      for (int c = 0; c < BN / 64; ++c) tma2d(dB + c * 8192, g.tB, n0 + 64 * c, kk, bar);
    }
  };
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) bar_init(bar0 + 8 * st, 1);
    if constexpr (P7_RING_RELEASE) {
#pragma unroll
      for (int st = 0; st < NST; ++st) bar_init(sa(&p7_empty_bars[st]), NTHREADS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    proxy_fence();
  }
  __syncthreads();
  if (tid == 0) {
    const int pre = Q < NST ? Q : NST;
    for (int q = 0; q < pre; ++q) issue(q);
  }
  float acc[Cf::MT][Cf::R];
  int q = 0;
#pragma unroll 1
  for (int task_i = 0; task_i < ntm; ++task_i) {
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i)
#pragma unroll
      for (int j = 0; j < Cf::R; ++j) acc[i][j] = 0.f;
#pragma unroll 1
    for (int kk = 0; kk < nk; ++kk, ++q) {
      const int st = q % NST;
      bar_wait(bar0 + 8 * st, (unsigned)(q / NST) & 1u);
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
          if constexpr (P7_DIAG_MODE != 2) s101h::MmeWg<Cf::WTN>::template run<AK ? 0 : 1, BKC ? 0 : 1>(acc[i], da, db, (kk | k16) != 0);
        }
      }
      wg_commit();
      wg_wait<1>();
#pragma unroll
      for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
      if constexpr (P7_RING_RELEASE) {
        if (q >= 1) bar_arrive(sa(&p7_empty_bars[(q - 1) % NST]));
        if (tid == 0 && q >= 1 && q - 1 + NST < Q) {
          bar_wait(sa(&p7_empty_bars[(q - 1) % NST]), (unsigned)((q - 1) / NST) & 1u);
          issue(q - 1 + NST);
        }
      } else {
        __syncthreads();
        if (tid == 0 && q >= 1 && q - 1 + NST < Q) issue(q - 1 + NST);
      }
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) keep(acc[i]);
    int m0, n0, sl;
    where(task_i, m0, n0, sl);
    float* Ps = P + (long long)sl * g.M * g.N;
    if constexpr (P7_DIAG_MODE == 1) continue;
#pragma unroll
    for (int i = 0; i < Cf::MT; ++i) {
      const int row = m0 + wgm * Cf::WTM + i * 64 + warp * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < Cf::WTN / 8; ++j) {
        const int col = n0 + wgn * Cf::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = row + 8 * h;
          if (r < g.M && col < g.N)
            *reinterpret_cast<float2*>(Ps + (long long)r * g.N + col) =
                make_float2(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
        }
      }
    }
  }
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) bar_inval(bar0 + 8 * st);
    if constexpr (P7_RING_RELEASE) {
#pragma unroll
      for (int st = 0; st < NST; ++st) bar_inval(sa(&p7_empty_bars[st]));
    }
    proxy_fence();
  }
}

template <int BM, int BN, int WM, bool AK, bool BKC, int NST, int KS>
DEV void tma_split_phase_inline(const TArgs g, float* P, float* smem, int lo = 0, int hi = 1, int den = 1,
                                int rot = 0) {
  tma_split_body<BM, BN, WM, AK, BKC, NST, KS>(g, P, smem, lo, hi, den, rot);
}

// Entry-resident form (inlined into the kernel: wgmma inside called functions is serialized by ptxas, C7510) and a
// called form for comparison.
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false, class Epi = NoPairEpi,
          bool STORE = true, bool DYN = false>
DEV void tma_phase_inline(const TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0,
                          const Epi epi = Epi{}) {
  if constexpr (indep_epi<Epi>::value && staged_epi<Epi>::value) {
    tma_body_indep<BM, BN, AK, BKC, indep_stages<BM, BN, Epi::kColSum>(), PAIR, Epi, STORE>(g, smem, lo, hi, den, rot,
                                                                                             epi);
  } else {
    tma_body<BM, BN, WM, AK, BKC, NST, PAIR, VEC, Epi, STORE, DYN>(g, smem, lo, hi, den, rot, epi);
  }
}
template <int BM, int BN, int WM, bool AK, bool BKC, int NST, bool PAIR, bool VEC = false>
__device__ __noinline__ void tma_phase(const TArgs g, float* smem, int lo = 0, int hi = 1, int den = 1, int rot = 0) {
  tma_body<BM, BN, WM, AK, BKC, NST, PAIR, VEC>(g, smem, lo, hi, den, rot);
}

}  // namespace pops4b
