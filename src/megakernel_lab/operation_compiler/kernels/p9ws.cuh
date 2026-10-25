// P9-B generalization of P8-A's warp-specialized BF16 product body (kernels/p8ws.cuh @ 25f1d13b3): per-item tile
// width BN in {64, 96, 128} (wave-quantization choice of the compiler), padded staging rows instead of the XOR chunk
// swizzle (valid for any BN), a stage ring sized for the widest member. Everything else is P8-A's design.
//
// Original P8-A header follows.
//
//
// One call executes a group of mutually independent products (the serial members of one phase, or a single product)
// as one continuous pipeline; the stage ring, staging buffers and barrier phases carry across products, so the next
// product's operand loads and MMAs overlap the previous product's last epilogue.
//
// Roles inside a call (CTA of 256 threads):
//   warpgroup 0 (threads 0-127)  MMA: each BM x 128 tile (MT = BM / 64 slabs of wgmma m64n128k16), one batch of 4 k16
//                                 steps per 64-k stage, one batch in flight; stages the rounded BF16 tile in shared
//                                 memory for the epilogue warps, then continues with its next tile.
//   MODE 0: warp 4 lane 0         TMA producer over the flattened (product, task, k) sequence; waits only on "empty"
//                                 barriers (one arrival by the MMA warpgroup per consumed stage). Epilogue: warps 5-7.
//   MODE 1: (no producer warp)    the MMA warpgroup refills the stage of step q-1 itself after the step's wgmma completed
//                                 in its four warps (128-thread named barrier). Epilogue: warps 4-7.
//   epilogue warps                wait for a staged tile ("sfull"), apply the generated register epilogue (same functor
//                                 contract as p5c::body: ld(e, i0) prefetch, epi(e, i0, gv) on 8 contiguous outputs at flat
//                                 offset i0, gv = rounded BF16 widened) and the output store, then free the buffer
//                                 ("sfree"). Two staging buffers: MMA runs up to one tile ahead of the epilogue.
// Measured standalone (P8-A gpu-03/04, o-l-bf16 shapes): one MMA warpgroup with a dedicated producer reaches 600-690
// TF/s where the 256-thread both-warpgroups-compute body with a CTA barrier per k step reaches 460-550.
//
// Arithmetic: identical per element to p5c::body with WM = 2 (each 64-row slab is one m64n128k16 wgmma chain over
// ascending k16 steps from scale-d = 0, FP32 accumulation, one cvt.rn.bf16x2 rounding, identical quad transpose and
// widening); only the thread that performs the epilogue and the store differs.
// Aligned extents only (M % BM == N % 128 == K % 64 == 0); no split-K, no multicast.
#pragma once
#include "p5c_wg.cuh"
#include "p8ws_region.cuh"


// Staging buffers (1 or 2): the MMA warpgroup may run this many tiles ahead of the epilogue units.
#ifndef P8WS_NSB
#define P8WS_NSB 2
#endif
// Staging: stmatrix.x4 (1) or shuffle-transposed 16-byte stores (0); one arrival per MMA warp (1) or per thread (0).
#ifndef P8WS_STMATRIX
#define P8WS_STMATRIX 1
#endif
#ifndef P8WS_WARP_ARRIVE
#define P8WS_WARP_ARRIVE 1
#endif

namespace p9ws {
using p8ws_region::arrive_if;
using p8ws_region::arrive;
using p8ws_region::st_shared16;
using p8ws_region::ld_shared16;
using p8ws_region::trace;

using p5c::Args;
using pops4b::sa;

__shared__ __align__(8) unsigned long long w9_full[8], w9_empty[8], w9_sfull[2], w9_sfree[2];

constexpr int BNMAX = 128;
constexpr int SBUF = 128 * BNMAX * 2;          // one staging buffer (bytes): 128 rows of the widest tile
// Staging layout: row stride BN * 2 bytes, 16-byte chunk k of a row stored at slot swz(k, row): an XOR inside groups
// of eight chunks (and inside the trailing group of four for BN = 96), so stmatrix rows spread over banks without
// padding (keeps the program's shared memory within the 196 KiB carve-out of P8-A's body).
template <int BN> DEV int swz(int k, int row) {
  if constexpr (BN == 96) return k < 8 ? (k ^ ((row & 3) << 1)) : (8 + ((k - 8) ^ (row & 3)));
  else if constexpr (BN == 64) return k ^ ((row & 3) << 1);
  else return k ^ ((row & 3) << 2);
}
// Shared bytes of one call: NST stages of (128 + 128) x 64 BF16 (the largest member tile), two 128 x 128 BF16 staging
// buffers, 1 KiB alignment.
template <int NST> constexpr int shared_bytes() { return NST * (128 + BNMAX) * 128 + P8WS_NSB * SBUF + 1024; }

// One product of a group: operand maps/pointers, its generated epilogue functors, and its CTA rotation.
template <int BM_, int BN_, bool AK_, bool BKC_, int OUT_, bool STORE_, class Epi_, class Ld_, int MC_ = 1,
          bool BAL_ = false>
struct Item {
  static constexpr int BM = BM_, BN = BN_, OUT = OUT_, MT = BM_ / 64, MC = MC_;
  static constexpr bool BAL = BAL_;   // balanced group schedule (compile-time: no rotation share code)
  static_assert(MC_ == 1 || (MC_ == 2 && BM_ == 128 && AK_), "multicast: K-major 128-row A, two halves");
  static_assert(BN_ == 64 || BN_ == 96 || BN_ == 128, "p9ws tile columns");
  static_assert(BKC_ || BN_ % 64 == 0, "N-major B operands need 64-column TMA chunks");
  static constexpr bool AK = AK_, BKC = BKC_, STORE = STORE_;
  using Epi = Epi_;
  using Ld = Ld_;
  Args g;
  Epi epi;
  Ld ld;
  int rot;
  // P9-B balanced group schedule (optional): sch = [sgrid][scount + 1] per-CTA prefix offsets, then the tile lists;
  // this item is member sidx. Used only on the planned grid with the whole-grid slice.
  const unsigned short* sch = nullptr;
  int sidx = 0, scount = 0, sgrid = 0;
};
template <int BM, int BN_, bool AK, bool BKC, int OUT, bool STORE, int MC = 1, class Epi, class Ld>
DEV Item<BM, BN_, AK, BKC, OUT, STORE, Epi, Ld, MC> item(const Args& g, Epi epi, Ld ld, int rot) {
  static_assert(BM == 64 || BM == 128, "p9ws tile rows");
  return Item<BM, BN_, AK, BKC, OUT, STORE, Epi, Ld, MC>{g, epi, ld, rot};
}
// Balanced member: this CTA's tiles come from the group's compile-time table (sch, member sidx of scount, planned
// grid sgrid); on another grid size the tiles are dealt round-robin over the whole grid.
template <int BM, int BN_, bool AK, bool BKC, int OUT, bool STORE, class Epi, class Ld>
DEV Item<BM, BN_, AK, BKC, OUT, STORE, Epi, Ld, 1, true> bitem(const Args& g, Epi epi, Ld ld,
                                                               const unsigned short* sch, int sidx, int scount,
                                                               int sgrid) {
  static_assert(BM == 64 || BM == 128, "p9ws tile rows");
  return Item<BM, BN_, AK, BKC, OUT, STORE, Epi, Ld, 1, true>{g, epi, ld, 0, sch, sidx, scount, sgrid};
}

// This CTA's share of one item: tasks first + i * count (i < ntm), nk 64-k steps each.
// MC = 2: the 2-CTA cluster (even rotation, whole even slice) owns units (M tile, N-tile pair); rank r takes N tile
// 2j + r, both CTAs of a pair walk the same units.
struct Share {
  int me, count, tn, ntm, nk, mc, rank;
  const unsigned short* tl;   // balanced schedule: this CTA's tile list of the item (nullptr: rotation share)
};
template <class I> DEV Share share_of(const I& it, int lo, int hi, int den) {
  const int tm = it.g.M / I::BM, tn = it.g.N / I::BN;
  if constexpr (I::BAL) {
    const int G = gridDim.x, me = blockIdx.x;
    if (G == it.sgrid) {
      const unsigned short* row = it.sch + me * (it.scount + 1) + it.sidx;
      const int b = row[0], e = row[1];
      return {0, 0, tn, e - b, it.g.K / 64, 1, 0, it.sch + it.sgrid * (it.scount + 1) + b};
    }
    const int ntask = tm * tn;
    return {me, G, tn, me < ntask ? (ntask - me + G - 1) / G : 0, it.g.K / 64, 1, 0, nullptr};
  }
  const popsc::Slice s = popsc::slice_of(lo, hi, den, it.rot);
  if constexpr (I::MC == 2) {
    const int pair = s.me >> 1, pairs = s.count >> 1, units = tm * (tn / 2);
    return {pair, pairs, tn, pair < units ? (units - pair + pairs - 1) / pairs : 0, it.g.K / 64, 2, s.me & 1, nullptr};
  }
  const int ntask = tm * tn;
  return {s.me, s.count, tn, s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0, it.g.K / 64, 1, 0, nullptr};
}
template <bool BAL = false> DEV void where(const Share& s, int bm, int bn, int i, int& m0, int& n0) {
  const int u = (BAL && s.tl) ? (int)s.tl[i] : s.me + i * s.count;
  if (s.mc == 2) {
    m0 = (u / (s.tn / 2)) * bm;
    n0 = (2 * (u % (s.tn / 2)) + s.rank) * bn;
  } else {
    m0 = (u / s.tn) * bm;
    n0 = (u % s.tn) * bn;
  }
}

struct Ctx {
  unsigned base, stg0, full0, empty0, sfull0, sfree0;
  int tid, lane, warp, qtotal;
};
constexpr int STAGE = (128 + BNMAX) * 128;   // ring slot size (largest member tile)

// Issue the TMA loads of local step qq of item `it` (predicated on `leader`; every thread executes the instruction).
template <class I> DEV void issue(const I& it, const Share& s, bool leader, int qq, unsigned slot, unsigned bar) {
  int m0, n0;
  where<I::BAL>(s, I::BM, I::BN, (leader ? qq : 0) / s.nk, m0, n0);
  if constexpr (I::MC == 2) p5c::stage_issue_mc<I::BM, I::BN, I::AK, I::BKC>(it.g, leader, (unsigned)s.rank, m0, n0, (qq % s.nk) * 64, slot, bar);
  else p5c::stage_issue<I::BM, I::BN, I::AK, I::BKC>(it.g, leader, m0, n0, (qq % s.nk) * 64, slot, bar);
}

// MODE 1 refill: global step gq belongs to exactly one item (offsets in item order); each item's issue is predicated.
template <int NST, class... Items>
DEV void refill_global(const Ctx& c, int gq, int lo, int hi, int den, const Items&... items) {
  int off = 0;
  const bool live = c.tid == 0 && gq < c.qtotal;
  auto one = [&](const auto& it) {
    const Share s = share_of(it, lo, hi, den);
    const int qi = s.ntm * s.nk;
    const bool mine = live && gq >= off && gq < off + qi;
    issue(it, s, mine, mine ? gq - off : 0, c.base + (gq % NST) * STAGE, c.full0 + 8 * (gq % NST));
    off += qi;
  };
  (one(items), ...);
}

template <class... Items> DEV void help_before(const Ctx& c, int t, int lo, int hi, int den, const Items&... items);

template <int MODE, int NST, bool HELP, class I, class... Items>
DEV void mma_item(const Ctx& c, const I& it, int lo, int hi, int den, int& q, int& t, const Items&... all) {
  constexpr int MT = I::MT, BN = I::BN, R = BN / 2, A_BYTES = I::BM * 128;
  const Share s = share_of(it, lo, hi, den);
  const int w = c.warp & 3, tq = c.lane & 3;
  auto release = [&](int done) {
    if constexpr (MODE == 1) {
      asm volatile("bar.sync 8, 128;" ::: "memory");
      refill_global<NST>(c, done + NST, lo, hi, den, all...);
    } else {
      if constexpr (I::MC == 2) {                                  // this CTA's and the peer's copy of the stage
        const unsigned eb = c.empty0 + 8 * (done % NST);
        p5c::release_pair_if(c.tid == 0, eb, p5c::peer_addr(eb, (unsigned)s.rank ^ 1u));
      } else {
        arrive_if(c.tid == 0, c.empty0 + 8 * (done % NST));
      }
    }
  };
  float acc[MT][R];
#pragma unroll 1
  for (int task = 0; task < s.ntm; ++task, ++t) {
    // HELP: before computing tile t, while tile t-2 still occupies t's staging buffer, this warp finishes epilogue units
    // of already staged tiles (no accumulator is live here), so an epilogue-bound group is drained by all warps.
    if constexpr (HELP) help_before(c, t, lo, hi, den, all...);
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(2, t);
#endif
    p5c::stage_batch<MT, BN, I::AK, I::BKC>(acc, c.full0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
                                            c.base + (q % NST) * STAGE, c.base + (q % NST) * STAGE + A_BYTES, 0, 0, true);
    ++q;
#pragma unroll 1
    for (int kk = 1; kk < s.nk; ++kk, ++q) {
      p5c::stage_batch<MT, BN, I::AK, I::BKC>(acc, c.full0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
                                              c.base + (q % NST) * STAGE, c.base + (q % NST) * STAGE + A_BYTES, 0, 0, false);
      pops4b::wg_wait<1>();
#pragma unroll
      for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
      release(q - 1);                                              // step q-1's stage consumed
    }
    pops4b::wg_wait<0>();
#pragma unroll
    for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
    release(q - 1);
    // Stage the rounded tile: chunk (row, k) of 8 BF16 columns at row * 256 + ((k ^ ((row & 3) << 2)) << 4).
    const int b = t % P8WS_NSB;
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(3, t);
#endif
    if (t >= P8WS_NSB) p5c::wait_parity(c.sfree0 + 8 * b, (unsigned)((t / P8WS_NSB) - 1) & 1u);
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(4, t);
#endif
    const unsigned buf = c.stg0 + (unsigned)b * SBUF;
#if P8WS_STMATRIX
    // stmatrix.x4: the wgmma accumulator fragment of an n8 block (rows g / g + 8, columns 2t, 2t + 1) is the m8n8
    // fragment layout; four 8 x 8 BF16 matrices (rows 0-7 / 8-15 of n8 blocks 2p and 2p + 1) per instruction. Lane l
    // addresses row (l % 8) of matrix l / 8. Same cvt.rn.bf16x2 values and swizzled chunk positions as below.
    {
      const int mi = c.lane >> 3, r8 = c.lane & 7;
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int p = 0; p < BN / 16; ++p) {
          const int row = i * 64 + w * 16 + r8 + 8 * (mi & 1), k = 2 * p + (mi >> 1);
          const unsigned addr = buf + row * (BN * 2) + (swz<BN>(k, row) << 4);
          const uint32_t r0 = s101h::bpk(acc[i][8 * p + 0], acc[i][8 * p + 1]), r1 = s101h::bpk(acc[i][8 * p + 2], acc[i][8 * p + 3]);
          const uint32_t r2 = s101h::bpk(acc[i][8 * p + 4], acc[i][8 * p + 5]), r3 = s101h::bpk(acc[i][8 * p + 6], acc[i][8 * p + 7]);
          asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
                       ::"r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
        }
    }
#else
    constexpr int JJ = BN / 32;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int cc = 0; cc < JJ * 2; ++cc) {
        const int jj = cc / 2, h = cc % 2, j0 = 4 * jj;
        const uint4 v = p5c::quad_chunk2(s101h::bpk(acc[i][4 * j0 + 2 * h], acc[i][4 * j0 + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 1) + 2 * h], acc[i][4 * (j0 + 1) + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 2) + 2 * h], acc[i][4 * (j0 + 2) + 2 * h + 1]),
                                         s101h::bpk(acc[i][4 * (j0 + 3) + 2 * h], acc[i][4 * (j0 + 3) + 2 * h + 1]), tq);
        const int row = i * 64 + w * 16 + (c.lane >> 2) + 8 * h, k = 4 * jj + tq;
        st_shared16(buf + row * (BN * 2) + (swz<BN>(k, row) << 4), v);
      }
#endif
#if P8WS_WARP_ARRIVE
    __syncwarp();                                                  // this warp's staged chunks precede its arrival
    arrive_if(c.lane == 0, c.sfull0 + 8 * b);
#else
    arrive(c.sfull0 + 8 * b);                                      // release: this thread's staged chunks
#endif
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(5, t);
#endif
  }
}

template <int NST, class I>
DEV void produce_item(const Ctx& c, const I& it, int lo, int hi, int den, int& q) {
  const Share s = share_of(it, lo, hi, den);
  const int qi = s.ntm * s.nk;
#pragma unroll 1
  for (int local = 0; local < qi; ++local, ++q) {
    const int st = q % NST;
    if (q >= NST) p5c::wait_parity(c.empty0 + 8 * st, (unsigned)((q / NST) - 1) & 1u);
#ifdef P8WS_TRACE
    if (local % s.nk == 0) trace(6, q / s.nk);
#endif
    issue(it, s, true, local, c.base + st * STAGE, c.full0 + 8 * st);
  }
}

__shared__ int w9_next;   // next epilogue unit (tile * 8 + unit) to claim, in order

// One epilogue unit: 1/8 of item tile `task` (CHUNKS / 8 chunks of 8 contiguous outputs), processed by one warp. Lane l
// handles chunks unit * U + l + 32 e (e < U / 32): consecutive lanes take consecutive 16-byte chunks of a row.
template <class I>
DEV void epilogue_unit(const Ctx& c, const I& it, const Share& s, int task, int unit, unsigned buf) {
  constexpr int CPR = I::BN / 8, CHUNKS = I::BM * CPR, U = CHUNKS / 8, E = U / 32;
  static_assert(E >= 1 && E <= 8, "epilogue unit");
  int m0, n0;
  where<I::BAL>(s, I::BM, I::BN, task, m0, n0);
  auto& epi = it.epi;
  auto& ld = it.ld;
  auto offset = [&](int k) { return (long long)(m0 + k / CPR) * it.g.ldc + n0 + 8 * (k % CPR); };
  const int k0 = unit * U + c.lane;
  if constexpr (!p5c::IsNoEpi<typename I::Epi>::value) {
#pragma unroll
    for (int e = 0; e < E; ++e) ld(e, (unsigned)offset(k0 + 32 * e));
  }
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int k = k0 + 32 * e, row = k / CPR, col = k % CPR;
    const uint4 v = ld_shared16(buf + row * (I::BN * 2) + (swz<I::BN>(col, row) << 4));
    const long long off = offset(k);
    if constexpr (!p5c::IsNoEpi<typename I::Epi>::value) {
      const float gv[8] = {__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                           __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u),
                           __uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                           __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u)};
      epi(e, (unsigned)off, gv);
    }
    if constexpr (!I::STORE) {
    } else if constexpr (I::OUT == 0) {
      *reinterpret_cast<uint4*>(reinterpret_cast<s101h::bf16*>(it.g.C) + off) = v;
    } else if constexpr (I::OUT == 2) {
      float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(it.g.C) + off);
      const float4 a0 = dst[0], a1 = dst[1];
      dst[0] = make_float4(a0.x + __uint_as_float(v.x << 16), a0.y + __uint_as_float(v.x & 0xffff0000u),
                           a0.z + __uint_as_float(v.y << 16), a0.w + __uint_as_float(v.y & 0xffff0000u));
      dst[1] = make_float4(a1.x + __uint_as_float(v.z << 16), a1.y + __uint_as_float(v.z & 0xffff0000u),
                           a1.z + __uint_as_float(v.w << 16), a1.w + __uint_as_float(v.w & 0xffff0000u));
    } else {
      float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(it.g.C) + off);
      dst[0] = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                           __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
      dst[1] = make_float4(__uint_as_float(v.z << 16), __uint_as_float(v.z & 0xffff0000u),
                           __uint_as_float(v.w << 16), __uint_as_float(v.w & 0xffff0000u));
    }
  }
}

// Epilogue worker (one warp): claim units in order until none remain. Every warp joins once its own role is done (the
// epilogue warps at once, the producer warp after its last load, the MMA warps after staging their last tile), so the
// group's final tiles are finished by up to eight warps. Results are elementwise: the claiming order changes no value.
template <class... Items>
DEV void epilogue_worker(const Ctx& c, int lo, int hi, int den, int tiles, const Items&... items) {
#pragma unroll 1
  while (true) {
    int g = 0;
    if (c.lane == 0) g = atomicAdd(&w9_next, 1);
    g = __shfl_sync(0xffffffffu, g, 0);
    if (g >= 8 * tiles) break;                                     // warp-uniform
    const int t = g >> 3, unit = g & 7, b = t % P8WS_NSB;
#ifdef P8WS_TRACE
    if (c.lane == 0) trace(7, (unsigned)g | ((unsigned)c.warp << 9));
#endif
    p5c::wait_parity(c.sfull0 + 8 * b, (unsigned)(t / P8WS_NSB) & 1u);
#ifdef P8WS_TRACE
    if (c.lane == 0) trace(8, (unsigned)g | ((unsigned)c.warp << 9));
#endif
    const unsigned buf = c.stg0 + (unsigned)b * SBUF;
    int first = 0;
    auto one = [&](const auto& it) {
      const Share s = share_of(it, lo, hi, den);
      if (t >= first && t < first + s.ntm) epilogue_unit(c, it, s, t - first, unit, buf);
      first += s.ntm;
    };
    (one(items), ...);
    __syncwarp();
    if (c.lane == 0) arrive(c.sfree0 + 8 * b);                     // release: this warp's unit of the buffer
#ifdef P8WS_TRACE
    if (c.lane == 0) trace(9, (unsigned)g | ((unsigned)c.warp << 9));
#endif
  }
}

// Claim the next epilogue unit only if it belongs to a tile below `limit` (CAS on the in-order queue; lane 0, broadcast).
DEV int claim_below(int limit, int lane) {
  int g = -1;
  if (lane == 0) {
    int cur = *reinterpret_cast<volatile int*>(&w9_next);
    while (cur < 8 * limit) {
      const int prev = atomicCAS(&w9_next, cur, cur + 1);
      if (prev == cur) { g = cur; break; }
      cur = prev;
    }
  }
  return __shfl_sync(0xffffffffu, g, 0);
}

template <class... Items>
DEV void help_before(const Ctx& c, int t, int lo, int hi, int den, const Items&... items) {
  if (t < P8WS_NSB) return;
  const int b = t % P8WS_NSB;
  const unsigned parity = (unsigned)((t / P8WS_NSB) - 1) & 1u;
#pragma unroll 1
  while (!__shfl_sync(0xffffffffu, (int)pops4b::bar_try(c.sfree0 + 8 * b, parity), 0)) {
    const int g = claim_below(t, c.lane);                          // tiles < t are staged (this warpgroup staged them)
    if (g < 0) continue;                                           // nothing claimable: keep polling the buffer
    const int u = g >> 3, unit = g & 7, ub = u % P8WS_NSB;
    p5c::wait_parity(c.sfull0 + 8 * ub, (unsigned)(u / P8WS_NSB) & 1u);
    const unsigned buf = c.stg0 + (unsigned)ub * SBUF;
    int first = 0;
    auto one = [&](const auto& it) {
      const Share s = share_of(it, lo, hi, den);
      if (u >= first && u < first + s.ntm) epilogue_unit(c, it, s, u - first, unit, buf);
      first += s.ntm;
    };
    (one(items), ...);
    __syncwarp();
    if (c.lane == 0) arrive(c.sfree0 + 8 * ub);
  }
}

template <class... Items> DEV int tiles_hint(int lo, int hi, int den, const Items&... items) {
  int n = 0;
  auto add = [&](const auto& it) { n += share_of(it, lo, hi, den).ntm; };
  (add(items), ...);
  return n;
}

template <int MODE, int NST, bool HELP = true, class... Items>
DEV void group(float* smem, int lo, int hi, int den, const Items&... items) {
  constexpr int GMC = ((Items::MC == 2) || ...) ? 2 : 1;
  static_assert(((Items::MC == GMC) && ...), "every member of a multicast group is multicast");
  static_assert(NST >= 2 && NST <= 8, "p8ws stages");
  const popsc::Slice s0 = popsc::slice_of(lo, hi, den, 0);
  if ((int)blockIdx.x < s0.first || (int)blockIdx.x >= s0.first + s0.count) return;
  int qtotal = 0;
  auto count = [&](const auto& it) { const Share s = share_of(it, lo, hi, den); qtotal += s.ntm * s.nk; };
  (count(items), ...);
  if (qtotal == 0) return;                                         // CTA-uniform
  Ctx c;
  c.tid = threadIdx.x;
  c.lane = c.tid & 31;
  c.warp = __shfl_sync(0xffffffffu, c.tid >> 5, 0);
  c.qtotal = qtotal;
  const int wg = __shfl_sync(0xffffffffu, c.tid >> 7, 0);
  c.base = (sa(smem) + 1023u) & ~1023u;
  c.stg0 = c.base + NST * STAGE;
  c.full0 = sa(&w9_full[0]); c.empty0 = sa(&w9_empty[0]); c.sfull0 = sa(&w9_sfull[0]); c.sfree0 = sa(&w9_sfree[0]);
  __syncthreads();                                                 // earlier generic users of smem are done
  if (c.tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_init(c.full0 + 8 * st, 1);
      pops4b::bar_init(c.empty0 + 8 * st, GMC);
    }
#pragma unroll
    for (int b = 0; b < P8WS_NSB; ++b) {
      pops4b::bar_init(c.sfull0 + 8 * b, P8WS_WARP_ARRIVE ? 4 : 128);
      pops4b::bar_init(c.sfree0 + 8 * b, 8);                       // one arrival per epilogue unit
    }
    w9_next = 0;
#ifdef P8WS_TRACE
    if (blockIdx.x < 4) p8ws_region::p8ws_trace_group[blockIdx.x] += 1u;
    trace(1, (unsigned)tiles_hint(lo, hi, den, items...));
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
  }
  __syncthreads();
  if constexpr (GMC == 2) p5c::cluster_sync();                     // peer barriers exist before any multicast
  int tiles = 0;
  auto count_tiles = [&](const auto& it) { tiles += share_of(it, lo, hi, den).ntm; };
  (count_tiles(items), ...);
  if (wg == 0) {
    if constexpr (MODE == 1) {
#pragma unroll 1
      for (int gq = 0; gq < NST; ++gq) refill_global<NST>(c, gq, lo, hi, den, items...);
    }
    int q = 0, t = 0;
    (mma_item<MODE, NST, HELP>(c, items, lo, hi, den, q, t, items...), ...);
  } else if (MODE == 0 && c.warp == 4) {
    if (c.lane == 0) {
      int q = 0;
      (produce_item<NST>(c, items, lo, hi, den, q), ...);
    }
    __syncwarp();
  }
  epilogue_worker(c, lo, hi, den, tiles, items...);
  __syncthreads();                                                 // all roles done: stages and staging reusable
  if constexpr (GMC == 2) p5c::cluster_sync();                     // peer's remote arrivals and multicasts landed
#ifdef P8WS_TRACE
  if (c.tid == 0) trace(10, (unsigned)tiles);
#endif
  if (c.tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_inval(c.full0 + 8 * st);
      pops4b::bar_inval(c.empty0 + 8 * st);
    }
#pragma unroll
    for (int b = 0; b < P8WS_NSB; ++b) {
      pops4b::bar_inval(c.sfull0 + 8 * b);
      pops4b::bar_inval(c.sfree0 + 8 * b);
    }
    pops4b::proxy_fence();
  }
}

// Single-product form (MODE 0 default), argument order of p5c::phase_epi plus the tile width.
template <int MODE, int BM, int BN_, bool AK, bool BKC, int NST, int OUT, bool STORE, class Epi, class Ld, int MC = 1>
DEV void phase_mode(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  group<MODE, NST>(smem, lo, hi, den, item<BM, BN_, AK, BKC, OUT, STORE, MC>(g, epi, ld, rot));
}
template <int MODE, int BM, int BN_, bool AK, bool BKC, int NST, int OUT, bool STORE, int MC, class Epi, class Ld>
DEV void phase_mc(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  group<MODE, NST>(smem, lo, hi, den, item<BM, BN_, AK, BKC, OUT, STORE, MC>(g, epi, ld, rot));
}

}  // namespace p9ws
