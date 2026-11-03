// Vocabulary-head streaming: logits product with the row statistics in its epilogue.
//
// Replaces the vocabulary logits product (pops4b_tma, [R, Vp] = X[R, K] W[Vp, K]^T) followed by the separate
// statistics pass (vocab_head.py _emit_stats_online, one warp per row re-reading the [R, Vp] logits).
//
// Schedule. One CTA owns a 128-row block of X for the whole vocabulary: its rows stay resident in shared memory
// (nk 64-wide k blocks, TMA, 128-byte swizzle) and only W streams, through an NST-stage ring of [128 x 64] TMA
// stages in the CTA's single stage sequence s = n * nk + kb (tile n = 128 vocabulary columns, ascending). The two
// warpgroups take alternate tiles (warpgroup n & 1), each computing the full 128 x 128 tile; their main loops run in
// tile order (a warpgroup starts tile n once every stage of tile n - 1 has landed: an mbarrier turn handed over by
// the other warpgroup), so one warpgroup's epilogue overlaps the other's main loop and no stage barrier phase can be
// observed early. The consumer of stage s refills its
// slot with stage s + NST once its MMAs have completed.
//
// Arithmetic (bitwise the replaced phases):
// * logits: m64n128k16 BF16 wgmma, FP32 accumulation from +0 over ascending k16 steps of the same zero-filled
//   64-wide k blocks as the pops4b_tma body (identical per-element instruction sequence), one cvt.rn.bf16x2
//   rounding; the 16-byte stores of its VEC epilogue (rows >= M masked).
// * statistics: the online warp statistics' exact per-lane sequence. That pass gives lane l the 8-column chunks
//   q = l, l + 32, ... (q < V / 8) in ascending order, then lane 0 the partial chunk V / 8 element by element, then
//   merges the 32 lanes with an xor butterfly. Tile n holds chunks 16 n .. 16 n + 15, i.e. lanes 16 (n & 1) + c, so
//   the warpgroup of parity n & 1 owns those lanes' states for every row and visits their chunks in ascending q. The
//   chunk update, the tail's element updates and the butterfly are the same expressions on the same rounded values.
#pragma once
#include "pops4b_tma.cuh"

namespace latch_vocab_stream {

using s101h::bf16;

// [0, NST): ring stages; [8]: resident rows; [9 + w]: warpgroup w may start its next main loop
__shared__ __align__(8) unsigned long long vs_bars[11];

struct FArgs {
  const void* tX;     // CUtensorMap of X: dims {K, M}, box {64, 128}
  const void* tW;     // CUtensorMap of W (padded rows): dims {K, N}, box {64, 128}
  bf16* C;            // logits [M, ldc]
  long long ldc;
  float* stats;       // [M, 2]: row max, log of the shifted exponential sum over the first `vocab` columns
  int M, N, K;
  int vocab;
  const void* tC = nullptr;   // TSE only: CUtensorMap of the logits, dims {N, M}, box {64, 64}
};

__device__ __forceinline__ void wg_bar(int wg) { asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory"); }
__device__ __forceinline__ void bar_arrive(unsigned b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory");
}

__device__ __forceinline__ void unpack8(const uint4 raw, float* v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
}

// _emit_stats_online, full chunk.
template <bool FAST = false>
__device__ __forceinline__ void chunk_update(float& m, float& s, const float* v) {
  float mc = v[0];
#pragma unroll
  for (int e = 1; e < 8; ++e) mc = fmaxf(mc, v[e]);
  if (mc > m) { s = s * (FAST ? __expf(m - mc) : expf(m - mc)); m = mc; }   // branch-free select measured equal
#pragma unroll
  for (int e = 0; e < 8; ++e) s += FAST ? __expf(v[e] - m) : expf(v[e] - m);
}

// Stage s of the W stream (tile s / NK, k block s % NK) into ring slot s % NST.
template <int NST, int NK>
__device__ __forceinline__ void issue(const FArgs& g, unsigned ring, unsigned bar0, int s, int rot = 0, int tn = 1) {
  const int st = s % NST;
  const unsigned bar = bar0 + 8 * st;
  pops4b::bar_expect(bar, 128u * 128u);
  pops4b::tma2d(ring + st * 128u * 128u, g.tW, (s % NK) * 64, (rot ? (s / NK + rot) % tn : s / NK) * 128, bar);
}

// NK = ceil(K / 64) <= 8 k blocks (compile-time: no runtime divisions near the MMAs).
// DIAG (timing diagnostics only; results wrong): bit 1 no statistics updates, bit 2 no logits stores, bit 4 each CTA
// streams the tiles rotated by an even CTA-dependent offset (no shared-W hot spot), bit 8 fast __expf in the updates, bit 16 a small stagger (0-6 tiles) instead of bit 4's rotation.
// SS: the per-lane statistics states live in shared memory ([warpgroup][jj][128 rows][4 t] float2 after the tail
// chunks) instead of 32 registers per thread across the tile loop; same values and update order.
// STATS = false (statistics-free form) is the bare streamed logits product (no statistics work,
// no statistics output); the original online statistics pass then runs as its own phase.
// ROTG > 0 (only with STATS = false: tile order is then free): CTA groups mb % ROTG start their tile
// sweep at tn * (mb % ROTG) / ROTG, so ROTG groups share each W tile in L2 instead of every CTA at once.
// TSE (only with STATS = false): the epilogue stages each warpgroup's rounded 64 x 128 halves in a
// per-warpgroup 16 KB 128-byte-swizzled shared buffer (one 4-byte store per BF16 pair, no shuffles) and leaves
// memory by two TMA bulk tensor stores per half (rows >= M clipped). Same values, same rounding.
// TSE == 2 stages one 64 x 64 quarter (8 KB per warpgroup) per round, four rounds per tile.
template <int NST, int NK, int DIAG = 0, bool SS = false, bool STATS = true, int ROTG = 0, int TSE = 0>
__device__ __forceinline__ void forward(const FArgs& g, float* smem) {
  static_assert(!TSE || !STATS, "staged TMA-store epilogue only without the statistics");
  static_assert(ROTG == 0 || !STATS, "rotated tile order only without the ordered statistics");
  constexpr int BM = 128, BN = 128;
  constexpr unsigned XB = BM * 128, WB = BN * 128;                 // bytes of one 64-wide k block
  static_assert(NST >= 2 && NST <= 8 && NK >= 1 && NK <= 8, "stages / k blocks");
  const int tid = threadIdx.x, wg = tid >> 7, wt = tid & 127, lane = tid & 31, warp = (tid >> 5) & 3, t = lane & 3;
  constexpr int nk = NK;
  const int tn = g.N / BN, tm = (g.M + BM - 1) / BM, Q = tn * nk;
  const int full = g.vocab >> 3;
  const bool tail = (g.vocab & 7) != 0;
  const unsigned raw0 = pops4b::sa(smem);
  const unsigned base = (raw0 + 1023u) & ~1023u, xs = base, ring = base + 8u * XB;
  const unsigned bar0 = pops4b::sa(&vs_bars[0]), xbar = bar0 + 64u, tbar = bar0 + 72u;
  uint4* tailbuf = reinterpret_cast<uint4*>(reinterpret_cast<char*>(smem) + (ring + NST * WB - raw0));
  float2* table = reinterpret_cast<float2*>(reinterpret_cast<char*>(smem) + (base - raw0));
  float2* states = reinterpret_cast<float2*>(reinterpret_cast<char*>(smem) + (ring + NST * WB + 2048u - raw0)) + wg * 2048;
  // state (i, h, jj) of this thread: row r = 64 i + 16 warp + lane / 4 + 8 h, at states[(jj * 128 + r) * 4 + t]
  auto sidx = [&](int si) __attribute__((always_inline)) {
    return ((si & 3) * 128 + (si >> 3) * 64 + warp * 16 + (lane >> 2) + 8 * ((si >> 2) & 1)) * 4 + t;
  };
  __syncthreads();                                                 // earlier generic users of smem are done
#pragma unroll 1
  for (int mb = blockIdx.x; mb < tm; mb += gridDim.x) {
    const int m0 = mb * BM;
    const int rot = (DIAG & 4) ? ((mb * 2 * 37) % tn) & ~1 : (DIAG & 16) ? 2 * (mb & 3)
                  : ROTG ? (int)((long long)(mb % (ROTG > 0 ? ROTG : 1)) * tn / (ROTG > 0 ? ROTG : 1)) : 0;
    if (tid == 0) {
#pragma unroll
      for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
      pops4b::bar_init(xbar, 1);
      pops4b::bar_init(tbar, 1);
      pops4b::bar_init(tbar + 8, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      pops4b::proxy_fence();                                       // generic shared writes -> TMA
    }
    __syncthreads();
    if (tid == 0) {
      pops4b::bar_expect(xbar, nk * XB);
      for (int kb = 0; kb < nk; ++kb) pops4b::tma2d(xs + kb * XB, g.tX, kb * 64, m0, xbar);
      for (int s = 0; s < NST && s < Q; ++s) issue<NST, NK>(g, ring, bar0, s, rot, tn);
    }
    float sm[16], ss[16];                                          // (i, h, jj): rows 64i + 16warp + lane/4 + 8h
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      sm[i] = -INFINITY;
      ss[i] = 0.0f;
      if (SS) states[sidx(i)] = make_float2(-INFINITY, 0.0f);
    }
    float acc[2][64];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 64; ++j) acc[i][j] = 0.f;
    pops4b::bar_wait(xbar, 0u);
#pragma unroll 1
    for (int n = wg; n < tn; n += 2) {
      // Turn k of warpgroup 1 (tile 2k + 1) / warpgroup 0 (tile 2k + 2): completion k of its turn barrier.
      if (n >= 1) pops4b::bar_wait(tbar + 8 * wg, (unsigned)((n >> 1) - (wg == 0 ? 1 : 0)) & 1u);
#pragma unroll 1
      for (int kb = 0; kb < nk; ++kb) {
        const int s = n * nk + kb, st = s % NST;
        pops4b::bar_wait(bar0 + 8 * st, (unsigned)(s / NST) & 1u);
        pops4b::keep(acc[0]);
        pops4b::keep(acc[1]);
        pops4b::wg_fence();
        const unsigned sW = ring + st * WB, sX = xs + kb * XB;
#pragma unroll
        for (int k16 = 0; k16 < 4; ++k16) {
          const uint64_t db = pops4b::dk(sW + k16 * 32);
#pragma unroll
          for (int i = 0; i < 2; ++i)
            s101h::MmeWg<128>::template run<0, 0>(acc[i], pops4b::dk(sX + i * 64 * 128 + k16 * 32), db,
                                                    (kb | k16) != 0);
        }
        pops4b::wg_commit();
        if (kb == nk - 1) {                                        // every stage of tile n has landed
          wg_bar(wg);
          if (wt == 0 && n + 1 < tn) bar_arrive(tbar + 8 * ((n + 1) & 1));
        }
        pops4b::wg_wait<1>();                                      // stage s - 1's MMAs are complete
        pops4b::keep(acc[0]);
        pops4b::keep(acc[1]);
        if (kb >= 1) {
          wg_bar(wg);
          if (wt == 0 && s - 1 + NST < Q) issue<NST, NK>(g, ring, bar0, s - 1 + NST, rot, tn);
        }
      }
      pops4b::wg_wait<0>();
      pops4b::keep(acc[0]);
      pops4b::keep(acc[1]);
      wg_bar(wg);
      if (wt == 0 && n * nk + nk - 1 + NST < Q) issue<NST, NK>(g, ring, bar0, n * nk + nk - 1 + NST, rot, tn);
      // Epilogue: the VEC body's 16-byte chunks (rounded and stored first, so the accumulators are dead before the
      // statistics); each chunk is also one statistics chunk of its row.
      const int n0 = (((DIAG & 20) || ROTG) ? (n + rot) % tn : n) * BN;
      if constexpr (TSE == 2) {
        const unsigned stg = ring + NST * WB + (unsigned)wg * 8192u;
        char* const stg_p = reinterpret_cast<char*>(smem) + (stg - raw0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            if (wt == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // staging free again
            wg_bar(wg);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const unsigned r = warp * 16 + (lane >> 2) + 8 * h;
#pragma unroll
              for (int jj = 0; jj < 8; ++jj) {
                const int j = 8 * c + jj;
                const unsigned off = r * 128u + ((((unsigned)jj) ^ (r & 7u)) << 4) + 4u * t;
                *reinterpret_cast<uint32_t*>(stg_p + off) = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
              }
            }
            pops4b::async_fence();
            wg_bar(wg);
            if (wt == 0 && !(DIAG & 2)) {
              asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                           ::"l"(g.tC), "r"(n0 + 64 * c), "r"(m0 + 64 * i), "r"(stg) : "memory");
              asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            }
          }
        }
      } else if constexpr (TSE == 1) {
        const unsigned stg = ring + NST * WB + (unsigned)wg * 16384u;
        char* const stg_p = reinterpret_cast<char*>(smem) + (stg - raw0);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          if (wt == 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");   // staging free again
          wg_bar(wg);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const unsigned r = warp * 16 + (lane >> 2) + 8 * h;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
              const unsigned off = (unsigned)(j >> 3) * 8192u + r * 128u + ((((unsigned)j & 7u) ^ (r & 7u)) << 4) + 4u * t;
              *reinterpret_cast<uint32_t*>(stg_p + off) = s101h::bpk(acc[i][4 * j + 2 * h], acc[i][4 * j + 2 * h + 1]);
            }
          }
          pops4b::async_fence();                                   // generic smem writes -> TMA (async proxy)
          wg_bar(wg);
          if (wt == 0 && !(DIAG & 2)) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
              asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
                           ::"l"(g.tC), "r"(n0 + 64 * c), "r"(m0 + 64 * i), "r"(stg + 8192u * c) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
          }
        }
      } else {
      uint4 pk[16];                                                // [(i * 2 + h) * 4 + jj]
#pragma unroll
      for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int gr = m0 + i * 64 + warp * 16 + (lane >> 2) + 8 * h;
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            const int j0 = 4 * jj;
            const uint4 v = pops4b::quad_chunk(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                               s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), t);
            pk[(i * 2 + h) * 4 + jj] = v;
            if (!(DIAG & 2) && gr < g.M) *reinterpret_cast<uint4*>(g.C + (long long)gr * g.ldc + n0 + 8 * (4 * jj + t)) = v;
          }
        }
      }
      if (STATS && !(DIAG & 1)) {
        if (n * 16 + 12 + t < full) {                              // every chunk of this thread wholly valid
#pragma unroll
          for (int si = 0; si < 16; ++si) {
            float x[8];
            unpack8(pk[si], x);
            if (SS) {
              float2 p = states[sidx(si)];
              chunk_update<(DIAG & 8) != 0>(p.x, p.y, x);
              states[sidx(si)] = p;
            } else {
              chunk_update<(DIAG & 8) != 0>(sm[si], ss[si], x);
            }
          }
        } else {
#pragma unroll
          for (int si = 0; si < 16; ++si) {
            const int q = n * 16 + 4 * (si & 3) + t;
            if (q < full) {
              float x[8];
              unpack8(pk[si], x);
              if (SS) {
                float2 p = states[sidx(si)];
                chunk_update<(DIAG & 8) != 0>(p.x, p.y, x);
                states[sidx(si)] = p;
              } else {
                chunk_update<(DIAG & 8) != 0>(sm[si], ss[si], x);
              }
            } else if (tail && q == full) {
              tailbuf[(si >> 3) * 64 + warp * 16 + (lane >> 2) + 8 * ((si >> 2) & 1)] = pk[si];
            }
          }
        }
      }
      }  // !TSE
    }
    if (TSE && wt == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");   // staged stores complete
    __syncthreads();                                               // every tile, stage and tail chunk is settled
    if constexpr (STATS) {
    if (SS) {
#pragma unroll
      for (int si = 0; si < 16; ++si) {
        const float2 p = states[sidx(si)];
        sm[si] = p.x;
        ss[si] = p.y;
      }
    }
    if (tail && wg == 0 && t == 0) {
      // lane 0 (jj == 0, t == 0) then takes the partial chunk V / 8 element by element (the tail loop).
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = i * 64 + warp * 16 + (lane >> 2) + 8 * h, si = (i * 2 + h) * 4;
          float v[8];
          unpack8(tailbuf[r], v);
          float m = sm[si], s = ss[si];
#pragma unroll
          for (int e = 0; e < 8; ++e) if (8 * full + e < g.vocab) {
            if (v[e] > m) { s = s * expf(m - v[e]); m = v[e]; }
            s += expf(v[e] - m);
          }
          sm[si] = m;
          ss[si] = s;
        }
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int r = i * 64 + warp * 16 + (lane >> 2) + 8 * h, si = (i * 2 + h) * 4 + jj;
          table[r * 32 + 16 * wg + 4 * jj + t] = make_float2(sm[si], ss[si]);
        }
    __syncthreads();
#pragma unroll 1
    for (int r = tid >> 5; r < BM; r += 8) {
      const float2 p = table[r * 32 + lane];
      float m = p.x, s = p.y;
#pragma unroll
      for (int delta = 16; delta > 0; delta /= 2) {
        const float m2 = __shfl_xor_sync(0xffffffffu, m, delta), s2 = __shfl_xor_sync(0xffffffffu, s, delta);
        const float mm = fmaxf(m, m2);
        s = (m == -INFINITY ? 0.0f : s * expf(m - mm)) + (m2 == -INFINITY ? 0.0f : s2 * expf(m2 - mm));
        m = mm;
      }
      if (lane == 0 && m0 + r < g.M) { g.stats[2 * (m0 + r)] = m; g.stats[2 * (m0 + r) + 1] = logf(s); }
    }
    }  // STATS
    __syncthreads();                                               // table reads done; barriers idle
    if (tid == 0) {
#pragma unroll
      for (int st = 0; st < NST; ++st) pops4b::bar_inval(bar0 + 8 * st);
      pops4b::bar_inval(xbar);
      pops4b::bar_inval(tbar);
      pops4b::bar_inval(tbar + 8);
      pops4b::proxy_fence();
    }
  }
  __syncthreads();
}

}  // namespace latch_vocab_stream
