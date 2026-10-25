// P8-A warp-specialized BF16 product body for the 256-thread cooperative program (new code in this lane).
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

namespace p8ws_region {

using p5c::Args;
using pops4b::sa;

__shared__ __align__(8) unsigned long long ws_full[8], ws_empty[8], ws_sfull[2], ws_sfree[2];

DEV void arrive_if(bool p, unsigned bar) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.u32 q, %0, 0;\n@q mbarrier.arrive.shared::cta.b64 _, [%1];\n}\n"
               ::"r"((unsigned)p), "r"(bar) : "memory");
}
DEV void arrive(unsigned bar) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory"); }

// Diagnostic trace (P8WS_TRACE builds only): CTAs 0-3 record (globaltimer, code) events; code = kind << 24 |
// group << 12 | value. Read back through the host entry p8ws_trace_read after a launch.
#ifdef P8WS_TRACE
#define P8WS_TRACE_N 32768
__device__ unsigned long long p8ws_trace_buf[4][P8WS_TRACE_N][2];
__device__ unsigned p8ws_trace_count[4];
__device__ unsigned p8ws_trace_group[4];
DEV void trace(unsigned kind, unsigned value) {
  if (blockIdx.x < 4) {
    const unsigned i = atomicAdd(&p8ws_trace_count[blockIdx.x], 1u);
    if (i < P8WS_TRACE_N) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
      p8ws_trace_buf[blockIdx.x][i][0] = t;
      p8ws_trace_buf[blockIdx.x][i][1] = ((unsigned long long)kind << 24) |
                                        ((unsigned long long)(p8ws_trace_group[blockIdx.x] & 0xfffu) << 12) | (value & 0xfffu);
    }
  }
}
extern "C" int p8ws_trace_read(void* dst, void* counts) {
  cudaError_t e = cudaMemcpyFromSymbol(dst, p8ws_trace_buf, sizeof(p8ws_trace_buf));
  if (e == cudaSuccess) e = cudaMemcpyFromSymbol(counts, p8ws_trace_count, sizeof(p8ws_trace_count));
  return (int)e;
}
extern "C" int p8ws_trace_reset() {
  unsigned z[4] = {0u, 0u, 0u, 0u};
  cudaError_t e = cudaMemcpyToSymbol(p8ws_trace_count, z, sizeof(z));
  if (e == cudaSuccess) e = cudaMemcpyToSymbol(p8ws_trace_group, z, sizeof(z));
  return (int)e;
}
#else
DEV void trace(unsigned, unsigned) {}
#endif
DEV void st_shared16(unsigned addr, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
DEV uint4 ld_shared16(unsigned addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr) : "memory");
  return v;
}

constexpr int BN = 128;
// Shared bytes of one call: NST stages of (128 + 128) x 64 BF16 (the largest member tile), two 128 x 128 BF16 staging
// buffers, 1 KiB alignment.
template <int NST> constexpr int shared_bytes() { return NST * (128 + BN) * 128 + P8WS_NSB * 128 * BN * 2 + 1024; }

// One product of a group: operand maps/pointers, its generated epilogue functors, and its CTA rotation.
template <int BM_, bool AK_, bool BKC_, int OUT_, bool STORE_, class Epi_, class Ld_>
struct Item {
  static constexpr int BM = BM_, OUT = OUT_, MT = BM_ / 64;
  static constexpr bool AK = AK_, BKC = BKC_, STORE = STORE_;
  using Epi = Epi_;
  using Ld = Ld_;
  Args g;
  Epi epi;
  Ld ld;
  int rot;
};
template <int BM, bool AK, bool BKC, int OUT, bool STORE, class Epi, class Ld>
DEV Item<BM, AK, BKC, OUT, STORE, Epi, Ld> item(const Args& g, Epi epi, Ld ld, int rot) {
  static_assert(BM == 64 || BM == 128, "p8ws tile rows");
  return Item<BM, AK, BKC, OUT, STORE, Epi, Ld>{g, epi, ld, rot};
}

// This CTA's share of one item: tasks first + i * count (i < ntm), nk 64-k steps each.
struct Share {
  int me, count, tn, ntm, nk;
};
template <class I> DEV Share share_of(const I& it, int lo, int hi, int den) {
  const popsc::Slice s = popsc::slice_of(lo, hi, den, it.rot);
  const int tm = it.g.M / I::BM, tn = it.g.N / BN, ntask = tm * tn;
  return {s.me, s.count, tn, s.me < ntask ? (ntask - s.me + s.count - 1) / s.count : 0, it.g.K / 64};
}
DEV void where(const Share& s, int bm, int i, int& m0, int& n0) {
  const int u = s.me + i * s.count;
  m0 = (u / s.tn) * bm;
  n0 = (u % s.tn) * BN;
}

struct Ctx {
  unsigned base, stg0, full0, empty0, sfull0, sfree0;
  int tid, lane, warp, qtotal;
};
constexpr int STAGE = (128 + BN) * 128;   // ring slot size (largest member tile)

// Issue the TMA loads of local step qq of item `it` (predicated on `leader`; every thread executes the instruction).
template <class I> DEV void issue(const I& it, const Share& s, bool leader, int qq, unsigned slot, unsigned bar) {
  int m0, n0;
  where(s, I::BM, (leader ? qq : 0) / s.nk, m0, n0);
  p5c::stage_issue<I::BM, BN, I::AK, I::BKC>(it.g, leader, m0, n0, (qq % s.nk) * 64, slot, bar);
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
  constexpr int MT = I::MT, R = BN / 2, A_BYTES = I::BM * 128;
  const Share s = share_of(it, lo, hi, den);
  const int w = c.warp & 3, tq = c.lane & 3;
  auto release = [&](int done) {
    if constexpr (MODE == 1) {
      asm volatile("bar.sync 8, 128;" ::: "memory");
      refill_global<NST>(c, done + NST, lo, hi, den, all...);
    } else {
      arrive_if(c.tid == 0, c.empty0 + 8 * (done % NST));
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
    const unsigned buf = c.stg0 + (unsigned)b * (128 * BN * 2);
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
          const unsigned addr = buf + row * (BN * 2) + ((k ^ ((row & 3) << 2)) << 4);
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
        st_shared16(buf + row * (BN * 2) + ((k ^ ((row & 3) << 2)) << 4), v);
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

__shared__ int ws_next;   // next epilogue unit (tile * 8 + unit) to claim, in order

// One epilogue unit: 1/8 of item tile `task` (CHUNKS / 8 chunks of 8 contiguous outputs), processed by one warp. Lane l
// handles chunks unit * U + l + 32 e (e < U / 32): consecutive lanes take consecutive 16-byte chunks of a row.
template <class I>
DEV void epilogue_unit(const Ctx& c, const I& it, const Share& s, int task, int unit, unsigned buf) {
  constexpr int CPR = BN / 8, CHUNKS = I::BM * CPR, U = CHUNKS / 8, E = U / 32;
  static_assert(E >= 1 && E <= 8, "epilogue unit");
  int m0, n0;
  where(s, I::BM, task, m0, n0);
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
    const uint4 v = ld_shared16(buf + row * (BN * 2) + ((col ^ ((row & 3) << 2)) << 4));
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
    if (c.lane == 0) g = atomicAdd(&ws_next, 1);
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
    const unsigned buf = c.stg0 + (unsigned)b * (128 * BN * 2);
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
    int cur = *reinterpret_cast<volatile int*>(&ws_next);
    while (cur < 8 * limit) {
      const int prev = atomicCAS(&ws_next, cur, cur + 1);
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
    const unsigned buf = c.stg0 + (unsigned)ub * (128 * BN * 2);
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
  c.full0 = sa(&ws_full[0]); c.empty0 = sa(&ws_empty[0]); c.sfull0 = sa(&ws_sfull[0]); c.sfree0 = sa(&ws_sfree[0]);
  __syncthreads();                                                 // earlier generic users of smem are done
  if (c.tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) {
      pops4b::bar_init(c.full0 + 8 * st, 1);
      pops4b::bar_init(c.empty0 + 8 * st, 1);
    }
#pragma unroll
    for (int b = 0; b < P8WS_NSB; ++b) {
      pops4b::bar_init(c.sfull0 + 8 * b, P8WS_WARP_ARRIVE ? 4 : 128);
      pops4b::bar_init(c.sfree0 + 8 * b, 8);                       // one arrival per epilogue unit
    }
    ws_next = 0;
#ifdef P8WS_TRACE
    if (blockIdx.x < 4) p8ws_trace_group[blockIdx.x] += 1u;
    trace(1, (unsigned)tiles_hint(lo, hi, den, items...));
#endif
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();                                         // generic global/shared writes -> TMA
  }
  __syncthreads();
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

// Single-product forms (MODE 0 default), argument order of p5c::phase_epi.
template <int BM, int BN_, bool AK, bool BKC, int NST, int OUT, bool STORE, class Epi, class Ld>
DEV void phase(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  static_assert(BN_ == BN, "p8ws tile columns");
  group<0, NST>(smem, lo, hi, den, item<BM, AK, BKC, OUT, STORE>(g, epi, ld, rot));
}
template <int MODE, int BM, int BN_, bool AK, bool BKC, int NST, int OUT, bool STORE, class Epi, class Ld>
DEV void phase_mode(const Args g, float* smem, int lo, int hi, int den, int rot, Epi epi, Ld ld) {
  static_assert(BN_ == BN, "p8ws tile columns");
  group<MODE, NST>(smem, lo, hi, den, item<BM, AK, BKC, OUT, STORE>(g, epi, ld, rot));
}
// Back-compat for the epilogue microbenchmark: body<BM, BN, AK, BKC, NST, OUT, STORE, Epi, Ld, MODE>.
template <int BM, int BN_, bool AK, bool BKC, int NST, int OUT, bool STORE, class Epi = p5c::NoEpi, class Ld = p5c::NoLd,
          int MODE = 0>
DEV void body(const Args& g, float* smem, int lo, int hi, int den, int rot, Epi epi = Epi(), Ld ld = Ld()) {
  group<MODE, NST>(smem, lo, hi, den, item<BM, AK, BKC, OUT, STORE>(g, epi, ld, rot));
}

}  // namespace p8ws_region
