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

namespace p8ws {

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

// P9-A: L2 prefetch of epilogue operand rows (cp.async.bulk.prefetch.L2), issued by idle epilogue warps ahead of
// the tile's epilogue; no value or order changes.
#ifndef P9_EPI_PREFETCH
#define P9_EPI_PREFETCH 0
#endif
DEV void pf_l2(const void* p, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
struct NoPf {
  DEV void operator()(unsigned) const {}
};
template <class T> struct IsNoPf { static constexpr bool value = false; };
template <> struct IsNoPf<NoPf> { static constexpr bool value = true; };

constexpr int BN = 128;
// Shared bytes of one call: NST stages of (128 + 128) x 64 BF16 (the largest member tile), two 128 x 128 BF16 staging
// buffers, 1 KiB alignment.
template <int NST> constexpr int shared_bytes() { return NST * (128 + BN) * 128 + P8WS_NSB * 128 * BN * 2 + 1024; }

// One product of a group: operand maps/pointers, its generated epilogue functors, and its CTA rotation.
// PK_ > 0 (release step 3 repair): the item's K loop is a sequence of wgmma chains of at most PK_ 64-k stages, each
// started from scale-d = 0; their FP32 results are summed in chunk order with round-to-nearest FP32 additions (see
// mma_item). PK_ = 0 is the single-chain body, unchanged.
template <int BM_, bool AK_, bool BKC_, int OUT_, bool STORE_, class Epi_, class Ld_, class Pf_ = NoPf, int PK_ = 0>
struct Item {
  static constexpr int BM = BM_, OUT = OUT_, MT = BM_ / 64, PK = PK_;
  static constexpr bool AK = AK_, BKC = BKC_, STORE = STORE_;
  using Epi = Epi_;
  using Ld = Ld_;
  using Pf = Pf_;
  Args g;
  Epi epi;
  Ld ld;
  int rot;
  Pf pf;
  int wt;                                                          // P9-A interleave weight (lower: earlier tiles)
};
template <int BM, bool AK, bool BKC, int OUT, bool STORE, class Epi, class Ld>
DEV Item<BM, AK, BKC, OUT, STORE, Epi, Ld> item(const Args& g, Epi epi, Ld ld, int rot) {
  static_assert(BM == 64 || BM == 128, "p8ws tile rows");
  return Item<BM, AK, BKC, OUT, STORE, Epi, Ld>{g, epi, ld, rot, NoPf(), 100};
}
template <int BM, bool AK, bool BKC, int OUT, bool STORE, class Epi, class Ld>
DEV Item<BM, AK, BKC, OUT, STORE, Epi, Ld> item_w(const Args& g, Epi epi, Ld ld, int wt, int rot) {
  static_assert(BM == 64 || BM == 128, "p8ws tile rows");
  return Item<BM, AK, BKC, OUT, STORE, Epi, Ld>{g, epi, ld, rot, NoPf(), wt};
}
template <int BM, bool AK, bool BKC, int OUT, bool STORE, class Epi, class Ld, class Pf>
DEV Item<BM, AK, BKC, OUT, STORE, Epi, Ld, Pf> item_pf(const Args& g, Epi epi, Ld ld, Pf pf, int rot) {
  static_assert(BM == 64 || BM == 128, "p8ws tile rows");
  return Item<BM, AK, BKC, OUT, STORE, Epi, Ld, Pf>{g, epi, ld, rot, pf, 100};
}
// Release step 3 repair: chunked accumulation of a long K loop (PK 64-k stages per wgmma chain).
template <int BM, bool AK, bool BKC, int OUT, bool STORE, int PK, class Epi, class Ld>
DEV Item<BM, AK, BKC, OUT, STORE, Epi, Ld, NoPf, PK> item_k(const Args& g, Epi epi, Ld ld, int rot) {
  static_assert(BM == 64 || BM == 128, "p8ws tile rows");
  static_assert(PK > 0, "chunk length in 64-k stages");
  return Item<BM, AK, BKC, OUT, STORE, Epi, Ld, NoPf, PK>{g, epi, ld, rot, NoPf(), 100};
}

// This CTA's share of one item: tasks first + i * count (i < ntm), nk 64-k steps each.
struct Share {
  int me, count, tn, ntm, nk;
};
// LATCH-FFN-RETAIN (OUT == 3): raw FP32 items (the unrounded FP32 product of BF16-exact operands, p5c OUT == 3) stage
// their tile in FP32 across both staging buffers (BM x BN x 4 bytes), so a group of raw items runs a one-slot ring;
// raw and rounded items never share a group. The epilogue receives the FP32 accumulators themselves (the values
// p5c::phase_epi<..., OUT = 3> passes), per element the same wgmma chain: bitwise equal results.
template <class... Items> __host__ __device__ constexpr bool all_raw() { return ((Items::OUT == 3) && ...); }
template <class... Items> __host__ __device__ constexpr bool any_raw() { return ((Items::OUT == 3) || ...); }
template <class... Items> __host__ __device__ constexpr int ring() { return all_raw<Items...>() ? 1 : P8WS_NSB; }
DEV void st_shared8(unsigned addr, float a, float b) {
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}
// FP32 staging: 16-byte chunk k (4 columns) of row `row` at row * BN * 4 + ((k ^ (row & 7)) << 4).
DEV unsigned raw_chunk(unsigned buf, int row, int k) { return buf + (unsigned)(row * (BN * 4)) + (unsigned)((k ^ (row & 7)) << 4); }

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
DEV void mma_item(const Ctx& c, const I& it, int lo, int hi, int den, int& q, int& t, int k0, int k1,
                  const Items&... all) {
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
  for (int task = k0; task < (k1 < s.ntm ? k1 : s.ntm); ++task, ++t) {
    // HELP: before computing tile t, while tile t-2 still occupies t's staging buffer, this warp finishes epilogue units
    // of already staged tiles (no accumulator is live here), so an epilogue-bound group is drained by all warps.
    if constexpr (HELP) help_before(c, t, lo, hi, den, all...);
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(2, t);
#endif
    if constexpr (I::PK == 0) {
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
    } else {
      // Release step 3 repair, chunked accumulation: one tensor-core chain over thousands of k16 steps loses accuracy
      // (each accumulate is truncated relative to the growing running sum), so every PK stages the chain restarts from
      // scale-d = 0 and its FP32 result is added to a running FP32 sum with round-to-nearest additions:
      // sum = ((c0 + c1) + c2) + ... in chunk order (deterministic; independent of timing and CTA placement).
      // The running sum lives in the staging buffers (thread-major float4 slots, conflict-free), which hold no staged
      // tile while this tile computes once the epilogues of the previous NSB tiles have released them.
      static_assert(I::BM * 128 * 4 <= P8WS_NSB * 128 * BN * 2, "chunk sum needs the FP32 tile in the staging buffers");
      constexpr int R4 = R / 4;
#pragma unroll 1
      for (int c0 = 0; c0 < s.nk; c0 += I::PK) {
        const int ce = c0 + I::PK < s.nk ? c0 + I::PK : s.nk;
        p5c::stage_batch<MT, BN, I::AK, I::BKC>(acc, c.full0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
                                                c.base + (q % NST) * STAGE, c.base + (q % NST) * STAGE + A_BYTES, 0, 0, true);
        ++q;
#pragma unroll 1
        for (int kk = c0 + 1; kk < ce; ++kk, ++q) {
          p5c::stage_batch<MT, BN, I::AK, I::BKC>(acc, c.full0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
                                                  c.base + (q % NST) * STAGE, c.base + (q % NST) * STAGE + A_BYTES, 0, 0, false);
          pops4b::wg_wait<1>();
#pragma unroll
          for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
          release(q - 1);
        }
        pops4b::wg_wait<0>();
#pragma unroll
        for (int i = 0; i < MT; ++i) pops4b::keep(acc[i]);
        release(q - 1);
        if (c0 == 0) {
          // The previous NSB tiles' epilogue units have finished reading their staged values.
#pragma unroll 1
          for (int u = t - P8WS_NSB; u < t; ++u)
            if (u >= 0) p5c::wait_parity(c.sfree0 + 8 * (u % P8WS_NSB), (unsigned)(u / P8WS_NSB) & 1u);
        }
        // Branch-free: the first chunk stores its result; later chunks add theirs to the running sum. After the last
        // chunk acc holds the complete sum (stores of earlier chunks are overwritten by the next chain's scale-d = 0).
        const bool head = c0 == 0;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
          for (int r = 0; r < R4; ++r) {
            const unsigned addr = c.stg0 + (unsigned)(((i * R4 + r) * 128 + c.tid) * 16);
            float p0, p1, p2, p3;
            asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(p0), "=f"(p1), "=f"(p2), "=f"(p3) : "r"(addr) : "memory");
            acc[i][4 * r + 0] = head ? acc[i][4 * r + 0] : __fadd_rn(p0, acc[i][4 * r + 0]);
            acc[i][4 * r + 1] = head ? acc[i][4 * r + 1] : __fadd_rn(p1, acc[i][4 * r + 1]);
            acc[i][4 * r + 2] = head ? acc[i][4 * r + 2] : __fadd_rn(p2, acc[i][4 * r + 2]);
            acc[i][4 * r + 3] = head ? acc[i][4 * r + 3] : __fadd_rn(p3, acc[i][4 * r + 3]);
            asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "f"(acc[i][4 * r + 0]), "f"(acc[i][4 * r + 1]),
                         "f"(acc[i][4 * r + 2]), "f"(acc[i][4 * r + 3]) : "memory");
          }
      }
      asm volatile("bar.sync 9, 128;" ::: "memory");             // every running-sum read precedes the staging writes
    }
    // Stage the rounded tile: chunk (row, k) of 8 BF16 columns at row * 256 + ((k ^ ((row & 3) << 2)) << 4).
    constexpr int NR = ring<Items...>();
    const int b = t % NR;
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(3, t);
#endif
    if (t >= NR) p5c::wait_parity(c.sfree0 + 8 * b, (unsigned)((t / NR) - 1) & 1u);
#ifdef P8WS_TRACE
    if (c.tid == 0) trace(4, t);
#endif
    const unsigned buf = c.stg0 + (unsigned)b * (128 * BN * 2);
    if constexpr (I::OUT == 3) {
      // Raw FP32 tile: accumulator pairs (row g / g + 8 of n8 block q, columns 2t, 2t + 1) in place, unrounded.
      static_assert(I::PK == 0, "raw FP32 items have no chunked accumulation");
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q8 = 0; q8 < BN / 8; ++q8) {
          const int row = i * 64 + w * 16 + (c.lane >> 2), col = 8 * q8 + 2 * tq;
          st_shared8(raw_chunk(buf, row, col >> 2) + ((col & 3) << 2), acc[i][4 * q8], acc[i][4 * q8 + 1]);
          st_shared8(raw_chunk(buf, row + 8, col >> 2) + ((col & 3) << 2), acc[i][4 * q8 + 2], acc[i][4 * q8 + 3]);
        }
    } else {
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
    }
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

// P9-A: L2 prefetch of the operand boxes of local step qq (TMA tensor prefetch; no shared memory, no value change).
#ifndef P9_OPF
#define P9_OPF 0
#endif
DEV void tma_pf2d(const void* map, int c0, int c1) {
  asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global.tile [%0, {%1, %2}];" ::"l"(map), "r"(c0), "r"(c1) : "memory");
}
template <class I> DEV void prefetch_step(const I& it, const Share& s, int qq) {
  int m0, n0;
  where(s, I::BM, qq / s.nk, m0, n0);
  const int kk = (qq % s.nk) * 64;
  if (it.g.vi_c) return;   // LATCH-STEP10-SPEED: im2col operands are not prefetched (no 2-D tile coordinates)
  if (I::AK) tma_pf2d(it.g.tA, kk, m0);
  else {
#pragma unroll
    for (int cc = 0; cc < I::BM / 64; ++cc) tma_pf2d(it.g.tA, m0 + 64 * cc, kk);
  }
  if (it.g.vb_c) return;   // step 10: tap-shifted operands are not prefetched (their slabs are re-read anyway)
  if (I::BKC) tma_pf2d(it.g.tB, kk, n0);
  else {
#pragma unroll
    for (int cc = 0; cc < BN / 64; ++cc) tma_pf2d(it.g.tB, n0 + 64 * cc, kk);
  }
}

template <int NST, class I>
DEV void produce_item(const Ctx& c, const I& it, int lo, int hi, int den, int& q, int k0 = 0, int k1 = 1 << 30) {
  const Share s = share_of(it, lo, hi, den);
  const int qi = (k1 < s.ntm ? k1 : s.ntm) * s.nk;
#if P9_OPF
  for (int pq = k0 * s.nk; pq < qi && pq < k0 * s.nk + NST + P9_OPF; ++pq)
    if (pq >= k0 * s.nk + NST) prefetch_step(it, s, pq);            // the first boxes beyond the ring
#endif
#pragma unroll 1
  for (int local = k0 * s.nk; local < qi; ++local, ++q) {
#if P9_OPF
    if (local + NST + P9_OPF < qi) prefetch_step(it, s, local + NST + P9_OPF);
#endif
    const int st = q % NST;
    if (q >= NST) p5c::wait_parity(c.empty0 + 8 * st, (unsigned)((q / NST) - 1) & 1u);
#ifdef P8WS_TRACE
    if (local % s.nk == 0) trace(6, q / s.nk);
#endif
    issue(it, s, true, local, c.base + st * STAGE, c.full0 + 8 * st);
  }
}

__shared__ int ws_next;   // next epilogue unit (tile * 8 + unit) to claim, in order
__shared__ int ws_il;                                              // P9-A: 1 when ws_order holds this group's tile order
__shared__ unsigned short ws_order[64];                            // tile t -> item << 8 | task
__shared__ int ws_side;                                            // P9-A: next local side-work chunk to claim

// P9-A side work: independent memory-bound chunks (e.g. casts whose readers are later phases) that epilogue warps run
// while their next staged unit is not ready; whatever remains is drained by all warps after the group. Chunk k of this
// CTA is global chunk blockIdx.x + k * gridDim.x. Every chunk is evaluated exactly once with its own fixed arithmetic.
#ifndef P9_PF_AHEAD
#define P9_PF_AHEAD P8WS_NU   // epilogue L2 prefetch distance in units (P8WS_NU: same unit of the next tile)
#endif
#ifndef P8WS_NU
#define P8WS_NU 8   // epilogue units per tile (P9-A: 16 halves each unit's unrolled code)
#endif
#ifndef P9_EARLY_LD
#define P9_EARLY_LD 0
#endif
struct NoSide {
  static constexpr bool active = false;
  DEV void operator()(unsigned) const {}
};
template <class F> struct SideCtx {
  static constexpr bool active = true;
  const F* f;
  int nloc;
  DEV void operator()(unsigned k) const { (*f)(k); }
};

// Item and task of tile t: the interleaved order when active, else items in sequence.
template <class... Items> DEV void tile_of(int t, int lo, int hi, int den, int& item, int& task, const Items&... items) {
  if (ws_il) {
    const unsigned v = ws_order[t];
    item = (int)(v >> 8);
    task = (int)(v & 255u);
    return;
  }
  int first = 0, idx = 0;
  item = -1;
  auto one = [&](const auto& it) {
    const int n = share_of(it, lo, hi, den).ntm;
    if (item < 0 && t < first + n) { item = idx; task = t - first; }
    first += n;
    ++idx;
  };
  (one(items), ...);
}

// One epilogue unit: 1/8 of item tile `task` (CHUNKS / 8 chunks of 8 contiguous outputs), processed by one warp. Lane l
// handles chunks unit * U + l + 32 e (e < U / 32): consecutive lanes take consecutive 16-byte chunks of a row.
// P9-A: the epilogue operand loads of one unit (the `ld` functor), issued before its tile is staged.
template <class I>
DEV void epilogue_load(const Ctx& c, const I& it, const Share& s, int task, int unit) {
  if constexpr (!p5c::IsNoEpi<typename I::Epi>::value) {
    constexpr int CPR = BN / 8, CHUNKS = I::BM * CPR, U = CHUNKS / P8WS_NU, E = U / 32;
    int m0, n0;
    where(s, I::BM, task, m0, n0);
    const int k0 = unit * U + c.lane;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const int k = k0 + 32 * e;
      it.ld(e, (unsigned)((long long)(m0 + k / CPR) * it.g.ldc + n0 + 8 * (k % CPR)));
    }
  }
}

template <class I>
DEV void epilogue_unit(const Ctx& c, const I& it, const Share& s, int task, int unit, unsigned buf, bool do_load = true) {
  constexpr int CPR = BN / 8, CHUNKS = I::BM * CPR, U = CHUNKS / P8WS_NU, E = U / 32;
  static_assert(E >= 1 && E <= 8, "epilogue unit");
  int m0, n0;
  where(s, I::BM, task, m0, n0);
  auto& epi = it.epi;
  auto& ld = it.ld;
  auto offset = [&](int k) { return (long long)(m0 + k / CPR) * it.g.ldc + n0 + 8 * (k % CPR); };
  const int k0 = unit * U + c.lane;
  if constexpr (!p5c::IsNoEpi<typename I::Epi>::value) {
    if (do_load) {
#pragma unroll
      for (int e = 0; e < E; ++e) ld(e, (unsigned)offset(k0 + 32 * e));
    }
  }
#pragma unroll
  for (int e = 0; e < E; ++e) {
    const int k = k0 + 32 * e, row = k / CPR, col = k % CPR;
    if constexpr (I::OUT == 3) {
      const uint4 lo = ld_shared16(raw_chunk(buf, row, 2 * col)), hi = ld_shared16(raw_chunk(buf, row, 2 * col + 1));
      const float gv[8] = {__uint_as_float(lo.x), __uint_as_float(lo.y), __uint_as_float(lo.z), __uint_as_float(lo.w),
                           __uint_as_float(hi.x), __uint_as_float(hi.y), __uint_as_float(hi.z), __uint_as_float(hi.w)};
      const long long off = offset(k);
      if constexpr (!p5c::IsNoEpi<typename I::Epi>::value) epi(e, (unsigned)off, gv);
      if constexpr (I::STORE) {
        float4* dst = reinterpret_cast<float4*>(reinterpret_cast<float*>(it.g.C) + off);
        dst[0] = make_float4(gv[0], gv[1], gv[2], gv[3]);
        dst[1] = make_float4(gv[4], gv[5], gv[6], gv[7]);
      }
      continue;
    }
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

// Rows of epilogue unit `unit` of item tile `task`: lane r < BM / 8 prefetches row r of every epilogue operand.
template <class I> DEV void prefetch_unit(const Ctx& c, const I& it, const Share& s, int task, int unit) {
  if constexpr (!IsNoPf<typename I::Pf>::value) {
    constexpr int ROWS = I::BM / P8WS_NU;
    if (c.lane < ROWS) {
      int m0, n0;
      where(s, I::BM, task, m0, n0);
      it.pf((unsigned)((long long)(m0 + unit * ROWS + c.lane) * it.g.ldc + n0));
    }
  }
}
template <class... Items> DEV void prefetch_g(const Ctx& c, int g, int tiles, int lo, int hi, int den, const Items&... items) {
#if P9_EPI_PREFETCH
  if (g >= P8WS_NU * tiles) return;
  const int t = g / P8WS_NU, unit = g % P8WS_NU;
  int ti, tk, idx = 0;
  tile_of(t, lo, hi, den, ti, tk, items...);
  auto one = [&](const auto& it) {
    if (idx++ == ti) prefetch_unit(c, it, share_of(it, lo, hi, den), tk, unit);
  };
  (one(items), ...);
#endif
}

// Epilogue worker (one warp): claim units in order until none remain. Every warp joins once its own role is done (the
// epilogue warps at once, the producer warp after its last load, the MMA warps after staging their last tile), so the
// group's final tiles are finished by up to eight warps. Results are elementwise: the claiming order changes no value.
template <class SC, class... Items>
DEV void epilogue_worker(const Ctx& c, const SC& side, int lo, int hi, int den, int tiles, const Items&... items) {
  bool side_open = SC::active;
#if P9_EPI_PREFETCH
  if (c.warp >= 5) {                                               // the first tile's units, before any is staged
#pragma unroll 1
    for (int u = c.warp - 5; u < P9_PF_AHEAD; u += 3) prefetch_g(c, u, tiles, lo, hi, den, items...);
  }
#endif
#pragma unroll 1
  while (true) {
    int g = 0;
    if (c.lane == 0) g = atomicAdd(&ws_next, 1);
    g = __shfl_sync(0xffffffffu, g, 0);
    if (g >= P8WS_NU * tiles) break;                               // warp-uniform
    prefetch_g(c, g + P9_PF_AHEAD, tiles, lo, hi, den, items...);  // P9-A: the unit P9_PF_AHEAD claims ahead
    constexpr int NR = ring<Items...>();
    const int t = g / P8WS_NU, unit = g % P8WS_NU, b = t % NR;
#ifdef P8WS_TRACE
    if (c.lane == 0) trace(7, (unsigned)g | ((unsigned)c.warp << 9));
#endif
    int ti, tk, idx = 0;
    tile_of(t, lo, hi, den, ti, tk, items...);
#if P9_EARLY_LD
    {
      int li = 0;
      auto load = [&](const auto& it) { if (li++ == ti) epilogue_load(c, it, share_of(it, lo, hi, den), tk, unit); };
      (load(items), ...);
    }
#endif
    if constexpr (SC::active) {
      // P9-A: while this unit's tile is not staged yet, run side-work chunks (one claim per chunk).
#pragma unroll 1
      while (side_open && !__shfl_sync(0xffffffffu, (int)pops4b::bar_try(c.sfull0 + 8 * b, (unsigned)(t / NR) & 1u), 0)) {
        int j = 0;
        if (c.lane == 0) j = atomicAdd(&ws_side, 1);
        j = __shfl_sync(0xffffffffu, j, 0);
        if (j >= side.nloc) { side_open = false; break; }
        side(blockIdx.x + (unsigned)j * gridDim.x);
      }
    }
    p5c::wait_parity(c.sfull0 + 8 * b, (unsigned)(t / NR) & 1u);
#ifdef P8WS_TRACE
    if (c.lane == 0) trace(8, (unsigned)g | ((unsigned)c.warp << 9));
#endif
    const unsigned buf = c.stg0 + (unsigned)b * (128 * BN * 2);
    auto one = [&](const auto& it) {
      if (idx++ == ti) epilogue_unit(c, it, share_of(it, lo, hi, den), tk, unit, buf, !P9_EARLY_LD);
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
    while (cur < P8WS_NU * limit) {
      const int prev = atomicCAS(&ws_next, cur, cur + 1);
      if (prev == cur) { g = cur; break; }
      cur = prev;
    }
  }
  return __shfl_sync(0xffffffffu, g, 0);
}

template <class... Items>
DEV void help_before(const Ctx& c, int t, int lo, int hi, int den, const Items&... items) {
  constexpr int NR = ring<Items...>();
  if (t < NR) return;
  const int b = t % NR;
  const unsigned parity = (unsigned)((t / NR) - 1) & 1u;
#pragma unroll 1
  while (!__shfl_sync(0xffffffffu, (int)pops4b::bar_try(c.sfree0 + 8 * b, parity), 0)) {
    const int g = claim_below(t, c.lane);                          // tiles < t are staged (this warpgroup staged them)
    if (g < 0) continue;                                           // nothing claimable: keep polling the buffer
    const int u = g / P8WS_NU, unit = g % P8WS_NU, ub = u % NR;
    p5c::wait_parity(c.sfull0 + 8 * ub, (unsigned)(u / NR) & 1u);
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

template <int MODE, int NST, bool HELP, bool IL, class SC, class... Items>
DEV void group_impl(float* smem, const SC& side, int lo, int hi, int den, const Items&... items) {
  static_assert(NST >= 2 && NST <= 8, "p8ws stages");
  static_assert(all_raw<Items...>() || !any_raw<Items...>(), "raw FP32 and rounded items never share a group");
  static_assert(!any_raw<Items...>() || (P8WS_NSB == 2 && ((Items::BM * BN * 4 <= 2 * 128 * BN * 2) && ...)),
                "raw FP32 tiles occupy both staging buffers");
  static_assert(!(IL && (HELP || MODE != 0)), "interleaved tile order requires MODE 0 without HELP");
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
      pops4b::bar_init(c.sfree0 + 8 * b, P8WS_NU);                 // one arrival per epilogue unit
    }
    ws_next = 0;
    ws_il = 0;
    if constexpr (IL) {
      // P9-A: tile k of item i at key (2k + 1) wt_i / (2 n_i) (heavier epilogues carry lower weights, so their tiles
      // spread over the earlier part of the sequence); ties by item. Order only: every tile computes the same values.
      int n[8], w[8], ni = 0, total = 0;
      auto add = [&](const auto& it) { if (ni < 8) { n[ni] = share_of(it, lo, hi, den).ntm; w[ni] = it.wt; total += n[ni]; } ++ni; };
      (add(items), ...);
      if (ni <= 8 && total <= 64) {
        int len = 0;
        for (int i = 0; i < ni; ++i)
          for (int k = 0; k < n[i]; ++k) {
            const long long num = (2LL * k + 1) * w[i], dd = 2LL * n[i];
            int pos = len;
            while (pos > 0) {
              const int pi = ws_order[pos - 1] >> 8, pk = ws_order[pos - 1] & 255;
              const long long pnum = (2LL * pk + 1) * w[pi], pden = 2LL * n[pi];
              if (pnum * dd > num * pden || (pnum * dd == num * pden && pi > i)) { ws_order[pos] = ws_order[pos - 1]; --pos; }
              else break;
            }
            ws_order[pos] = (unsigned short)((i << 8) | k);
            ++len;
          }
        ws_il = 1;
      }
    }
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
    if (IL && ws_il) {
#pragma unroll 1
      for (int u = 0; u < tiles; ++u) {
        int ti, tk, idx = 0;
        tile_of(u, lo, hi, den, ti, tk, items...);
        ti = __shfl_sync(0xffffffffu, ti, 0);                     // warp-uniform dispatch around the wgmma loop
        tk = __shfl_sync(0xffffffffu, tk, 0);
        auto one = [&](const auto& it) { if (idx++ == ti) mma_item<MODE, NST, HELP>(c, it, lo, hi, den, q, t, tk, tk + 1, items...); };
        (one(items), ...);
      }
    } else {
      (mma_item<MODE, NST, HELP>(c, items, lo, hi, den, q, t, 0, 1 << 30, items...), ...);
    }
  } else if (MODE == 0 && c.warp == 4) {
    if (c.lane == 0) {
      int q = 0;
      if (IL && ws_il) {
#pragma unroll 1
        for (int u = 0; u < tiles; ++u) {
          int ti, tk, idx = 0;
          tile_of(u, lo, hi, den, ti, tk, items...);
          auto one = [&](const auto& it) { if (idx++ == ti) produce_item<NST>(c, it, lo, hi, den, q, tk, tk + 1); };
          (one(items), ...);
        }
      } else {
        (produce_item<NST>(c, items, lo, hi, den, q), ...);
      }
    }
    __syncwarp();
  }
  epilogue_worker(c, side, lo, hi, den, tiles, items...);
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

template <int MODE, int NST, bool HELP = true, bool IL = false, class... Items>
DEV void group(float* smem, int lo, int hi, int den, const Items&... items) {
  group_impl<MODE, NST, HELP, IL>(smem, NoSide{}, lo, hi, den, items...);
}

// P9-A: the group with side work of `nside` chunks (see SideCtx). Every CTA of the grid runs its own chunks, also a CTA
// that has no product tile in this group.
template <int MODE, int NST, bool HELP = true, bool IL = false, class F, class... Items>
DEV void group_s(float* smem, const F& f, unsigned nside, int lo, int hi, int den, const Items&... items) {
  const int nloc = blockIdx.x < nside ? (int)((nside - 1u - blockIdx.x) / gridDim.x + 1u) : 0;
  __syncthreads();                                                 // earlier users of ws_side are done
  if (threadIdx.x == 0) ws_side = 0;
  __syncthreads();
  group_impl<MODE, NST, HELP, IL>(smem, SideCtx<F>{&f, nloc}, lo, hi, den, items...);
  __syncthreads();                                                 // group roles done (or no tile here)
  const int lane = threadIdx.x & 31;
#pragma unroll 1
  while (true) {
    int j = 0;
    if (lane == 0) j = atomicAdd(&ws_side, 1);
    j = __shfl_sync(0xffffffffu, j, 0);
    if (j >= nloc) break;
    f(blockIdx.x + (unsigned)j * gridDim.x);
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

}  // namespace p8ws
