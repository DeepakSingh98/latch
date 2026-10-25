// LATCH-STEP19-SPEED: generated BF16 product epilogues over a bounded active extent (new code in this lane).
//
// btile2e is pops2b::btile2 (P-OPS2-B) without operand prologues, staged through the existing masked-edge operand
// loader (popsc::bstage_edge, P-OPS4-B/P7-B) whenever a tile crosses the runtime M or K extent. Per element the
// arithmetic is unchanged: one FP32 accumulator from +0, one mma.sync.m16n8k16 per k16 block in ascending k (absent
// k terms are exact +0 products), one cvt.rn.bf16x2 rounding. The generated tile epilogue receives the number of
// existing rows of its tile, so neither its operand prefetch nor its expressions or stores touch inactive rows.
// Separate header: static programs never include it, so their sources and header identities are unchanged.
#pragma once
#include "pops2b_mm.cuh"

namespace latch_dyn {

using s101h::bf16;

// pops2b::epi_prefetch restricted to the first `rows` rows of the tile.
template <class T, int BM, int BN>
DEV void epi_prefetch_rows(const T* __restrict__ src, unsigned ldc, T* __restrict__ dst, int rows) {
  constexpr int V = 16 / (int)sizeof(T), CPR = BN / V, CH = BM * CPR;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if ((CH % NTHREADS == 0 || c < CH) && c / CPR < rows) {
      const int r = c / CPR, cc = (c % CPR) * V;
      s101h::cpa16(reinterpret_cast<float*>(dst + r * BN + cc), reinterpret_cast<const float*>(src + (size_t)r * ldc + cc));
    }
  }
}

// No-epilogue functor with the row-bounded interface.
struct NoEpiRows {
  static constexpr bool kTile = false;
  template <int BM, int BN> __host__ __device__ static constexpr int ebytes() { return 0; }
  template <int BM, int BN> DEV void prefetch_rows(char*, unsigned, unsigned, int) const {}
  template <int BM, int BN> DEV void tile_rows(const float*, int, const char*, unsigned, unsigned, int) const {}
};

// pops2b::sstage with extents: source element (r, kk) exists for r0 + r < rlim and k + kk < klim; absent elements
// stage as +0 without global loads (element-granular partial chunks, as popsc::bstage_edge).
template <class T, int R, bool RK, int KS>
DEV void sstage_edge(const T* __restrict__ X, long long sr, long long sk, int r0, int k, T* __restrict__ S,
                     int rlim, int klim) {
  constexpr int V = 16 / (int)sizeof(T);
  constexpr int CH = R * KS / V;
#pragma unroll
  for (int c0 = 0; c0 < CH; c0 += NTHREADS) {
    const int c = c0 + threadIdx.x;
    if (CH % NTHREADS == 0 || c < CH) {
      if (RK) {
        const int r = c / (KS / V), kc = (c % (KS / V)) * V;
        const int left = klim - (k + kc), e = (r0 + r < rlim) ? (left < 0 ? 0 : (left > V ? V : left)) : 0;
        T* dst = S + pops2b::Src<T, R, RK, KS>::pos(r, kc);
        const unsigned sh = (unsigned)__cvta_generic_to_shared(dst);
        const T* src = e ? X + (long long)(r0 + r) * sr + (k + kc) : X;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sh), "l"(src), "r"(e * (int)sizeof(T)) : "memory");
      } else {
        const int kr = c / (R / V), rc = (c % (R / V)) * V;
        const int left = rlim - (r0 + rc), e = (k + kr < klim) ? (left < 0 ? 0 : (left > V ? V : left)) : 0;
        T* dst = S + pops2b::Src<T, R, RK, KS>::pos(rc, kr);
        const unsigned sh = (unsigned)__cvta_generic_to_shared(dst);
        const T* src = e ? X + (long long)(k + kr) * sk + r0 + rc : X;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sh), "l"(src), "r"(e * (int)sizeof(T)) : "memory");
      }
    }
  }
}

// No operand prologue (edge interface).
struct NoProE {
  static constexpr bool kStaged = false;
  template <int R, bool RK, int KS> __host__ __device__ static constexpr int bytes() { return 0; }
  template <int R, bool RK, int KS> DEV void issue_e(char*, long long, long long, long long, int, int, int, int) const {}
  template <int R, bool RK, int KS>
  DEV void convert_e(const char*, bf16*, bf16*, long long, long long, long long, int, int, int, int) const {}
};

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, class ProA = NoProE, class ProB = NoProE>
struct EdgeTile {
  using L = popsc::BLay<BM, BN, AK, BKC, KS>;
  static constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8, FR = MT * NT * 4;
  static_assert(WM * WN == NTHREADS / 32 && WTM % 16 == 0 && WTN % 8 == 0 && (NT == 1 || NT % 2 == 0), "warp tile");
  // Stage = [A bf16][B bf16][A prologue sources][B prologue sources] (pops2b::btile2 layout).
  static constexpr int AB = (L::STAGE * 2 + 15) / 16 * 16;
  static constexpr int SA = ProA::template bytes<BM, AK, KS>(), SB = ProB::template bytes<BN, BKC, KS>();
  static constexpr int STAGE_BYTES = AB + SA + SB;
};

// Operand prologue bindings of one product (offsets of the operand views; optional materialized outputs).
struct ProArgs {
  long long aoff, boff;
  bf16* aout;
  bf16* bout;
};

// Mainloop over K steps [k0, k0 + nk*KS) of one tile (absent M/N/K elements staged as +0 without loads).
// Prologue operands stage their FP32/BF16 sources and convert them into the stage's BF16 slot after the stage
// barrier (pops2b::btile2); absent elements convert to +0 and are never materialized. A materialized operand
// k-chunk (global step kstep) is written by the tile whose column (row) index equals kstep modulo the count.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, class ProA = NoProE, class ProB = NoProE>
DEV void edge_mainloop(const popsc::BArgs& g, int m0, int n0, int k0, int nk, bool full, bf16* S,
                       float (&acc)[EdgeTile<BM, BN, WM, WN, AK, BKC, KS>::MT][EdgeTile<BM, BN, WM, WN, AK, BKC, KS>::NT][4],
                       const ProA& pa = ProA{}, const ProB& pb = ProB{}, const ProArgs& pr = ProArgs{0, 0, nullptr, nullptr}) {
  using T = EdgeTile<BM, BN, WM, WN, AK, BKC, KS, ProA, ProB>;
  using L = typename T::L;
  constexpr int MT = T::MT, NT = T::NT;
  constexpr bool PA = ProA::kStaged, PB = ProB::kStaged;
  static_assert(NST >= 2, "stage ring");
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int wm0 = (w / WN) * T::WTM, wn0 = (w % WN) * T::WTN;
  const int ncol = (g.N + BN - 1) / BN, tcol = n0 / BN, nrow = (g.M + BM - 1) / BM, trow = m0 / BM, kstep0 = k0 / KS;
  char* const base = reinterpret_cast<char*>(S);
  auto issue = [&](int kk, int st) {
    char* stage = base + st * T::STAGE_BYTES;
    bf16* As = reinterpret_cast<bf16*>(stage);
    if constexpr (PA) pa.template issue_e<BM, AK, KS>(stage + T::AB, pr.aoff, g.sam, g.sak, m0, k0 + kk, g.M, g.K);
    else if (full) popsc::bstage<BM, AK, KS>(g.A, g.sam, g.sak, m0, k0 + kk, As);
    else popsc::bstage_edge<BM, AK, KS>(g.A, g.sam, g.sak, m0, k0 + kk, As, g.M, g.K);
    if constexpr (PB) pb.template issue_e<BN, BKC, KS>(stage + T::AB + T::SA, pr.boff, g.sbn, g.sbk, n0, k0 + kk, g.N, g.K);
    else if (full) popsc::bstage<BN, BKC, KS>(g.B, g.sbn, g.sbk, n0, k0 + kk, As + L::A_EL);
    else popsc::bstage_edge<BN, BKC, KS>(g.B, g.sbn, g.sbk, n0, k0 + kk, As + L::A_EL, g.N, g.K);
  };
#pragma unroll
  for (int s = 0; s < NST - 1; ++s) {
    if (s < nk) issue(s * KS, s);
    s101h::cpa_commit();
  }
#pragma unroll 1
  for (int it = 0; it < nk; ++it) {
    s101h::cpa_wait<NST - 2>();
    __syncthreads();                                               // stage it landed; stage it-1 fully consumed
    if constexpr (PA || PB) {
      char* stage = base + (it % NST) * T::STAGE_BYTES;
      bf16* As = reinterpret_cast<bf16*>(stage);
      if constexpr (PA) pa.template convert_e<BM, AK, KS>(stage + T::AB, As,
          (pr.aout != nullptr && (kstep0 + it) % ncol == tcol) ? pr.aout : nullptr,
          pr.aoff, g.sam, g.sak, m0, k0 + it * KS, g.M, g.K);
      if constexpr (PB) pb.template convert_e<BN, BKC, KS>(stage + T::AB + T::SA, As + L::A_EL,
          (pr.bout != nullptr && (kstep0 + it) % nrow == trow) ? pr.bout : nullptr,
          pr.boff, g.sbn, g.sbk, n0, k0 + it * KS, g.N, g.K);
      __syncthreads();                                             // converted BF16 operands visible
    }
    if (it + NST - 1 < nk) issue((it + NST - 1) * KS, (it + NST - 1) % NST);
    s101h::cpa_commit();
    const bf16* As = reinterpret_cast<const bf16*>(base + (it % NST) * T::STAGE_BYTES);
    const bf16* Bs = As + L::A_EL;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t af[MT][4];
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (AK) s101h::bldsm4(af[i], As + (wm0 + 16 * i + (lane & 15)) * (KS + 8) + k16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[i], As + (k16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + wm0 + 16 * i +
                                       (((lane >> 3) & 1) << 3));
      }
      if constexpr (NT == 1) {
        uint32_t bb[2];
        if (BKC) popsc::bldsm2(bb, Bs + (wn0 + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
        else popsc::bldsm2t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0);
#pragma unroll
        for (int i = 0; i < MT; ++i) s101h::bmma(acc[i][0], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
      } else {
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
          uint32_t bb[4];
          if (BKC) s101h::bldsm4(bb, Bs + (wn0 + 8 * (j + (lane >> 4)) + (lane & 7)) * (KS + 8) + k16 + ((lane >> 3) & 1) * 8);
          else s101h::bldsm4t(bb, Bs + (k16 + (lane & 7) + ((lane >> 3) & 1) * 8) * (BN + 8) + wn0 + 8 * (j + (lane >> 4)));
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            s101h::bmma(acc[i][j], af[i][0], af[i][1], af[i][2], af[i][3], bb[0], bb[1]);
            s101h::bmma(acc[i][j + 1], af[i][0], af[i][1], af[i][2], af[i][3], bb[2], bb[3]);
          }
        }
      }
    }
  }
  s101h::cpa_wait<0>();
}

// One BF16 rounding of the accumulators; masked stores; row-bounded generated tile epilogue.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, bool PAIR, class Epi,
          class ProA = NoProE, class ProB = NoProE>
DEV void edge_finish(const popsc::BArgs& g, int m0, int n0, bf16* S, const Epi& epi,
                     float (&acc)[EdgeTile<BM, BN, WM, WN, AK, BKC, KS>::MT][EdgeTile<BM, BN, WM, WN, AK, BKC, KS>::NT][4]) {
  using T = EdgeTile<BM, BN, WM, WN, AK, BKC, KS, ProA, ProB>;
  constexpr int MT = T::MT, NT = T::NT;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int wm0 = (w / WN) * T::WTM, wn0 = (w % WN) * T::WTN;
  const int rows = g.M - m0 < BM ? g.M - m0 : BM;
  char* const E = reinterpret_cast<char*>(S) + NST * T::STAGE_BYTES;  // epilogue operand prefetch region
  const unsigned ebase = (unsigned)((long long)m0 * g.ldc + n0);
  bf16* C = g.C;
  constexpr int LDT = BN + 4;                                      // FP32 tile staging (row-major, padded)
  float* Ts = reinterpret_cast<float*>(S);
  static_assert(!Epi::kTile || BM * LDT * 4 <= NST * T::STAGE_BYTES, "tile staging fits the stage ring");
  if constexpr (Epi::kTile) __syncthreads();                       // every warp finished reading the stage ring
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int m = m0 + wm0 + 16 * i + gq, n = n0 + wn0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = m + 8 * h;
        const uint32_t packed = s101h::bpk(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        if constexpr (STORE) {
          if (mm < g.M && n < g.N) {
            if (PAIR) {
              *reinterpret_cast<uint32_t*>(C + (long long)mm * g.ldc + n) = packed;
            } else {
              unsigned short* c = reinterpret_cast<unsigned short*>(C + (long long)mm * g.ldc + n);
              c[0] = (unsigned short)(packed & 0xffffu);
              if (n + 1 < g.N) c[1] = (unsigned short)(packed >> 16);
            }
          }
        }
        if constexpr (Epi::kTile) {
          const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&packed);
          *reinterpret_cast<float2*>(Ts + (mm - m0) * LDT + (n - n0)) =
              make_float2(__bfloat162float(pair.x), __bfloat162float(pair.y));
        }
      }
    }
  if constexpr (Epi::kTile) {
    __syncthreads();
    epi.template tile_rows<BM, BN>(Ts, LDT, E, ebase, (unsigned)g.ldc, rows);
  }
  __syncthreads();                                                 // every fragment/tile read done before smem reuse
}

template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, bool PAIR, class Epi,
          class ProA, class ProB>
DEV void btile2e(const popsc::BArgs& g, int m0, int n0, bf16* S, const Epi& epi, const ProA& pa, const ProB& pb,
                 const ProArgs& pr) {
  using T = EdgeTile<BM, BN, WM, WN, AK, BKC, KS, ProA, ProB>;
  float acc[T::MT][T::NT][4];
#pragma unroll
  for (int i = 0; i < T::MT; ++i)
#pragma unroll
    for (int j = 0; j < T::NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
  const bool full = g.K % KS == 0 && m0 + BM <= g.M && n0 + BN <= g.N;
  const int rows = g.M - m0 < BM ? g.M - m0 : BM;
  if constexpr (Epi::kTile)
    epi.template prefetch_rows<BM, BN>(reinterpret_cast<char*>(S) + NST * T::STAGE_BYTES,
                                       (unsigned)((long long)m0 * g.ldc + n0), (unsigned)g.ldc, rows);
  edge_mainloop<BM, BN, WM, WN, AK, BKC, KS, NST, ProA, ProB>(g, m0, n0, 0, (g.K + KS - 1) / KS, full, S, acc, pa, pb, pr);
  edge_finish<BM, BN, WM, WN, AK, BKC, KS, NST, STORE, PAIR, Epi, ProA, ProB>(g, m0, n0, S, epi, acc);
}

// Whole-grid or CTA-slice phase over ceil(M/BM) x N/BN tiles (N and the tile plan are capacity-static).
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, bool STORE, bool PAIR, class Epi,
          class ProA, class ProB>
__device__ __noinline__ void bmm_phase2e(const popsc::BArgs g, float* smem, const Epi epi, const ProA pa, const ProB pb,
                                         const ProArgs pr, int lo = 0, int hi = 1,
                                         int den = 1, int rot = 0) {
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, ntask = tm * tn;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int j = task % tn, r = task / tn;
    btile2e<BM, BN, WM, WN, AK, BKC, KS, NST, STORE, PAIR, Epi, ProA, ProB>(g, r * BM, j * BN, S, epi, pa, pb, pr);
  }
}

// Split-K over a static K with runtime M: tile x slice tasks. Each slice writes its FP32 accumulator fragments to
// private partial storage; the last arriving slice of a tile (arrival counter, restored to zero) sums the SPLIT
// partials in ascending slice order, rounds once to BF16 and runs the store/epilogue (the popsc::bmm_phase_sk
// protocol, P8-C, with masked edges and row-bounded epilogues). Deterministic for a given SPLIT.
template <int BM, int BN, int WM, int WN, bool AK, bool BKC, int KS, int NST, int SPLIT, bool STORE, bool PAIR, class Epi,
          class ProA, class ProB>
__device__ __noinline__ void bmm_phase2e_sk(const popsc::BArgs g, float* smem, const Epi epi, const ProA pa, const ProB pb,
                                            const ProArgs pr, int lo, int hi, int den,
                                            int rot, float* __restrict__ part, unsigned* cnt) {
  const popsc::Slice s = popsc::slice_of(lo, hi, den, rot);
  if ((int)blockIdx.x < s.first || (int)blockIdx.x >= s.first + s.count) return;
  using T = EdgeTile<BM, BN, WM, WN, AK, BKC, KS, ProA, ProB>;
  __shared__ unsigned latch_sk_last;
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, tiles = tm * tn, ntask = tiles * SPLIT;
  const int kslice = g.K / SPLIT, nk = kslice / KS;
  bf16* S = reinterpret_cast<bf16*>(smem);
  for (int task = s.me; task < ntask; task += s.count) {
    const int tile = task % tiles, slice = task / tiles;
    const int m0 = (tile / tn) * BM, n0 = (tile % tn) * BN;
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
      for (int j = 0; j < T::NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    const bool full = m0 + BM <= g.M && n0 + BN <= g.N;
    edge_mainloop<BM, BN, WM, WN, AK, BKC, KS, NST, ProA, ProB>(g, m0, n0, slice * kslice, nk, full, S, acc, pa, pb, pr);
    float* mine = part + ((long long)tile * SPLIT + slice) * (BM * BN) + threadIdx.x * T::FR;
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
      for (int j = 0; j < T::NT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) mine[(i * T::NT + j) * 4 + q] = acc[i][j][q];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      const unsigned prev = atomicAdd(cnt + tile, 1u);
      latch_sk_last = prev == (unsigned)(SPLIT - 1);
      if (latch_sk_last) cnt[tile] = 0u;                           // restored for the next use in this launch
    }
    __syncthreads();
    if (latch_sk_last) {
      __threadfence();
      const float* base = part + (long long)tile * SPLIT * (BM * BN) + threadIdx.x * T::FR;
#pragma unroll
      for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            float sum = __ldcg(base + (i * T::NT + j) * 4 + q);
#pragma unroll
            for (int sl = 1; sl < SPLIT; ++sl) sum += __ldcg(base + (long long)sl * (BM * BN) + (i * T::NT + j) * 4 + q);
            acc[i][j][q] = sum;
          }
      const int rows = g.M - m0 < BM ? g.M - m0 : BM;
      if constexpr (Epi::kTile) {
        epi.template prefetch_rows<BM, BN>(reinterpret_cast<char*>(S) + NST * T::STAGE_BYTES,
                                           (unsigned)((long long)m0 * g.ldc + n0), (unsigned)g.ldc, rows);
        s101h::cpa_commit();
        s101h::cpa_wait<0>();                                      // epilogue operands landed (edge_finish syncs)
      }
      edge_finish<BM, BN, WM, WN, AK, BKC, KS, NST, STORE, PAIR, Epi, ProA, ProB>(g, m0, n0, S, epi, acc);
    }
    __syncthreads();
  }
}

}  // namespace latch_dyn
