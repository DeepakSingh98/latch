// LATCH-STEP-06: deferred shared-weight gradient products of a saved-tape recurrent update (ops/loop_defer_grads.py).
//
// For every deferred weight W (M x N, FP32 running gradient G) and every output tile of 128 x 128, one CTA loads the
// tile of G once, then for execution steps e = 0..count-1 (iteration r = count-1-e) forms the BF16 x BF16 -> FP32
// wgmma k-sequence over that iteration's T rows of A_r^T (saved tape [T, M]) and B_r (reverse-tape cotangent [T, N])
// from +0, rounds it once to BF16 (cvt.rn.bf16x2, as the p5c epilogue), widens it exactly and adds it in FP32 to the
// tile, and finally stores the tile. Per element this is the operation sequence of the reverse body's per-step
// read-modify-write epilogue (G = G + float(bf16_rn(acc_r)), r = count-1 .. 0): the results are bitwise equal.
//
// The stage ring, k16 association (p5c::stage_batch, ascending k from +0) and TMA boxes are those of the p5c body for
// M-major A and N-major B (AK = BKC = false) with 128 x 128 tiles (two warpgroups of 64 x 128). The tape slots of
// consecutive iterations are contiguous rows, so one tensor map per operand spans [max_steps * T] rows.
#pragma once
#include "p5c_wg.cuh"

namespace step06 {

struct DEnt {
  int map_a, map_b;       // tensor-map indices (128-byte maps in the program's table)
  long long g_word;       // arena word offset of the FP32 gradient [M, N]
  int M, N;
  int tile0;              // first tile of this entry in the concatenated tile list (row-major 128 x 128 tiles)
  int zero;               // 1: the running gradient starts from +0 (its terminal +0 fill is elided); stored for count 0 too
  int opt;                // >= 0: elementwise optimizer update applied with the final gradient (Opt::apply case)
  int p_arg, m_arg;       // loop arguments of the parameter and its optimizer state (opt >= 0)
};
struct NoOpt {
  DEV static void apply(int, float, float, float, float&, float&) {}
};

template <int NE>
DEV int entry_of(const DEnt* E, int u) {
  int e = 0;
#pragma unroll
  for (int i = 1; i < NE; ++i) e = u >= E[i].tile0 ? i : e;
  return e;
}

// E: NE entries (compile-time table), total tiles `total`, T rows per iteration, `count` executed iterations.
template <int NE, int NST, class Opt = NoOpt>
DEV void dgrad(const DEnt* E, int total, int T, const char* maps, float* arena, int count, float* smem,
               float** args = nullptr) {
  using Cf = pops4b::Cfg<128, 128, 2>;
  static_assert(Cf::MT == 1 && Cf::WTN == 128 && Cf::WTM == 64, "two 64 x 128 warpgroup slabs");
  const int units = (int)gridDim.x, unit = (int)blockIdx.x;
  const int ntile = unit < total ? (total - unit + units - 1) / units : 0;
  const int nseg = count > 0 ? count : 0;
  const int nk = T / 64, ntask = ntile * nseg, Q = ntask * nk;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, t4 = lane & 3;
  if (nseg == 0) {
    // count 0: no reverse contribution; +0-initialized gradients still receive their +0 values.
    for (int i = 0; i < ntile; ++i) {
      const int u = unit + i * units, e = entry_of<NE>(E, u);
      if (!E[e].zero) continue;
      const int local = u - E[e].tile0, tn = E[e].N / 128, N = E[e].N;
      const int m0 = (local / tn) * 128, n0 = (local % tn) * 128;
      float* G = arena + E[e].g_word;
      for (int c = tid; c < 128 * 32; c += 256) {
        const int row = c >> 5, col = (c & 31) * 4;
        const long long off = (long long)(m0 + row) * N + n0 + col;
        *reinterpret_cast<float4*>(G + off) = make_float4(0.f, 0.f, 0.f, 0.f);
        if (E[e].opt >= 0) {
          float* P = args[E[e].p_arg];
          float* Mm = args[E[e].m_arg];
#pragma unroll
          for (int q4 = 0; q4 < 4; ++q4) {
            float pn, mn;
            Opt::apply(E[e].opt, 0.f, P[off + q4], Mm[off + q4], pn, mn);
            P[off + q4] = pn;
            Mm[off + q4] = mn;
          }
        }
      }
    }
    return;
  }
  if (Q == 0) return;                                                // CTA-uniform
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u, bar0 = pops4b::sa(&pops4b::tma_bars[0]);
  const bool leader = tid == 0;
  // (tile, segment) of task i: tile u = unit + (i / nseg) * units, segment s = i % nseg -> iteration count-1-s.
  auto locate = [&](int task, int& e, int& m0, int& n0, int& row0) {
    const int u = unit + (task / nseg) * units, s = task % nseg;
    e = entry_of<NE>(E, u);
    const int local = u - E[e].tile0, tn = E[e].N / 128;
    m0 = (local / tn) * 128;
    n0 = (local % tn) * 128;
    row0 = (count - 1 - s) * T;
  };
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();
  }
  __syncthreads();
#define S06_ISSUE(QQ)                                                                                              \
  { const int qq_ = (QQ); int e_, m_, n_, r_; locate(qq_ / nk, e_, m_, n_, r_);                                     \
    p5c::Args a_{}; a_.tA = maps + 128ll * E[e_].map_a; a_.tB = maps + 128ll * E[e_].map_b;                         \
    p5c::stage_issue<128, 128, false, false>(a_, leader, m_, n_, r_ + (qq_ % nk) * 64,                              \
                                             base + (qq_ % NST) * Cf::STAGE, bar0 + 8 * (qq_ % NST)); }
  {
    const int pre = Q < NST ? Q : NST;
#pragma unroll 1
    for (int q = 0; q < pre; ++q) S06_ISSUE(q)
  }
  float acc[Cf::MT][Cf::R];
  float gr[Cf::R];                                                   // running gradient of this thread's elements
  int q = 0;
#pragma unroll 1
  for (int task = 0; task < ntask; ++task) {
    int e, m0, n0, row0;
    locate(task, e, m0, n0, row0);
    const int s = task % nseg, N = E[e].N;
    float* G = arena + E[e].g_word;
    const int rr = wg * 64 + warp * 16 + (lane >> 2);
    if (s == 0 && E[e].zero) {
#pragma unroll
      for (int j = 0; j < Cf::R; ++j) gr[j] = 0.0f;
    } else if (s == 0) {
#pragma unroll
      for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
        const int col = n0 + nb * 8 + 2 * t4;
        const float2 a = *reinterpret_cast<const float2*>(G + (long long)(m0 + rr) * N + col);
        const float2 b = *reinterpret_cast<const float2*>(G + (long long)(m0 + rr + 8) * N + col);
        gr[4 * nb] = a.x; gr[4 * nb + 1] = a.y; gr[4 * nb + 2] = b.x; gr[4 * nb + 3] = b.y;
      }
    }
    p5c::stage_batch<Cf::MT, Cf::WTN, false, false>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
        base + (q % NST) * Cf::STAGE, base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wg * Cf::WTM, 0, true);
    ++q;
#pragma unroll 1
    for (int kk = 1; kk < nk; ++kk, ++q) {
      p5c::stage_batch<Cf::MT, Cf::WTN, false, false>(acc, bar0 + 8 * (q % NST), (unsigned)(q / NST) & 1u,
          base + (q % NST) * Cf::STAGE, base + (q % NST) * Cf::STAGE + Cf::A_BYTES, wg * Cf::WTM, 0, false);
      pops4b::wg_wait<1>();
      pops4b::keep(acc[0]);
      __syncthreads();
      if (q - 1 + NST < Q) S06_ISSUE(q - 1 + NST)
    }
    pops4b::wg_wait<0>();
    pops4b::keep(acc[0]);
    __syncthreads();
    if (q - 1 + NST < Q) S06_ISSUE(q - 1 + NST)
    // G = G + float(bf16_rn(acc)) per element: the p5c OUT = 2 epilogue arithmetic, in registers.
#pragma unroll
    for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
      const uint32_t u0 = s101h::bpk(acc[0][4 * nb], acc[0][4 * nb + 1]);
      const uint32_t u1 = s101h::bpk(acc[0][4 * nb + 2], acc[0][4 * nb + 3]);
      gr[4 * nb] = gr[4 * nb] + __uint_as_float(u0 << 16);
      gr[4 * nb + 1] = gr[4 * nb + 1] + __uint_as_float(u0 & 0xffff0000u);
      gr[4 * nb + 2] = gr[4 * nb + 2] + __uint_as_float(u1 << 16);
      gr[4 * nb + 3] = gr[4 * nb + 3] + __uint_as_float(u1 & 0xffff0000u);
    }
    if (s == nseg - 1) {
#pragma unroll
      for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
        const int col = n0 + nb * 8 + 2 * t4;
        *reinterpret_cast<float2*>(G + (long long)(m0 + rr) * N + col) = make_float2(gr[4 * nb], gr[4 * nb + 1]);
        *reinterpret_cast<float2*>(G + (long long)(m0 + rr + 8) * N + col) = make_float2(gr[4 * nb + 2], gr[4 * nb + 3]);
      }
      if (E[e].opt >= 0) {
        // The optimizer's elementwise update of this tile with the final gradient values (same expressions and
        // operand order as the generated optimizer phase it replaces; parameter and state updated in place).
        float* P = args[E[e].p_arg];
        float* Mm = args[E[e].m_arg];
#pragma unroll 4
        for (int nb = 0; nb < Cf::WTN / 8; ++nb) {
          const int col = n0 + nb * 8 + 2 * t4;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const long long off = (long long)(m0 + rr + 8 * h) * N + col;
            const float2 pv = *reinterpret_cast<const float2*>(P + off), mv = *reinterpret_cast<const float2*>(Mm + off);
            float p0, m0v, p1, m1v;
            Opt::apply(E[e].opt, gr[4 * nb + 2 * h], pv.x, mv.x, p0, m0v);
            Opt::apply(E[e].opt, gr[4 * nb + 2 * h + 1], pv.y, mv.y, p1, m1v);
            *reinterpret_cast<float2*>(P + off) = make_float2(p0, p1);
            *reinterpret_cast<float2*>(Mm + off) = make_float2(m0v, m1v);
          }
        }
      }
    }
  }
#undef S06_ISSUE
  __syncthreads();
  if (tid == 0) {
#pragma unroll
    for (int st = 0; st < NST; ++st) pops4b::bar_inval(bar0 + 8 * st);
    pops4b::proxy_fence();
  }
}

}  // namespace step06

// ---------------------------------------------------------------------------------------------------------------
// MODE 'tape': the reverse body stored each step's BF16 weight-gradient product P_r = bf16_rn(acc_r) in slot r of a
// reverse tape; here G = G + float(P_r) for r = count-1 .. 0 (execution order), element by element: the operation
// sequence of the in-body FP32 read-modify-write epilogues. Optionally the elementwise optimizer update follows.
namespace step06 {
struct DSum {
  long long g_word;       // arena word offset of the FP32 gradient
  long long p_word;       // arena word offset of slot 0 of the BF16 per-iteration products
  long long slot_words;   // words between consecutive iteration slots
  long long quads;        // elements / 4
  int zero, opt, p_arg, m_arg;
};

// 8 elements per thread and pass with every load (products, running gradient, parameter, momentum) issued before
// the first add: more bytes in flight per thread for this DRAM-bound stage. Per element the same adds in the same
// order (r = count-1 .. 0) and the same optimizer expressions.
template <class Opt>
DEV void dsum8(const DSum& d, int n, float** args, float* arena, long long tid, long long nt, long long o0 = 0,
               long long o1 = -1) {
  float4* G = reinterpret_cast<float4*>(arena + d.g_word);
  const uint4* P = reinterpret_cast<const uint4*>(arena + d.p_word);
  const long long slot = d.slot_words / 4, octs = o1 >= 0 ? o1 : d.quads / 2;     // uint4 units
  float4* Pp = d.opt >= 0 ? reinterpret_cast<float4*>(args[d.p_arg]) : nullptr;
  float4* Mm = d.opt >= 0 ? reinterpret_cast<float4*>(args[d.m_arg]) : nullptr;
  auto add8 = [](float4& a, float4& b, const uint4 v) {
    a.x = a.x + __uint_as_float(v.x << 16);
    a.y = a.y + __uint_as_float(v.x & 0xffff0000u);
    a.z = a.z + __uint_as_float(v.y << 16);
    a.w = a.w + __uint_as_float(v.y & 0xffff0000u);
    b.x = b.x + __uint_as_float(v.z << 16);
    b.y = b.y + __uint_as_float(v.z & 0xffff0000u);
    b.z = b.z + __uint_as_float(v.w << 16);
    b.w = b.w + __uint_as_float(v.w & 0xffff0000u);
  };
  // Two octets per thread and pass (q, q + nt): every load of both is issued before the first add.
  auto finish = [&](long long q, float4 g0, float4 g1, float4 p0, float4 p1, float4 m0, float4 m1) {
    G[2 * q] = g0;
    G[2 * q + 1] = g1;
    if (Pp) {
      float4 a, b, c, e;
      Opt::apply(d.opt, g0.x, p0.x, m0.x, a.x, b.x);
      Opt::apply(d.opt, g0.y, p0.y, m0.y, a.y, b.y);
      Opt::apply(d.opt, g0.z, p0.z, m0.z, a.z, b.z);
      Opt::apply(d.opt, g0.w, p0.w, m0.w, a.w, b.w);
      Opt::apply(d.opt, g1.x, p1.x, m1.x, c.x, e.x);
      Opt::apply(d.opt, g1.y, p1.y, m1.y, c.y, e.y);
      Opt::apply(d.opt, g1.z, p1.z, m1.z, c.z, e.z);
      Opt::apply(d.opt, g1.w, p1.w, m1.w, c.w, e.w);
      Pp[2 * q] = a;
      Pp[2 * q + 1] = c;
      Mm[2 * q] = b;
      Mm[2 * q + 1] = e;
    }
  };
#pragma unroll 1
  for (long long q = o0 + tid; q < octs; q += 2 * nt) {
    const long long u = q + nt;
    const bool two = u < octs;                                     // thread-uniform per pass except at the tail
    const long long uu = two ? u : q;
    uint4 w[8], x[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) if (s < n) {
      w[s] = __ldcs(P + (long long)(n - 1 - s) * slot + q);
      x[s] = __ldcs(P + (long long)(n - 1 - s) * slot + uu);
    }
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 g0 = z, g1 = z, p0 = z, p1 = z, m0 = z, m1 = z, h0 = z, h1 = z, r0 = z, r1 = z, k0 = z, k1 = z;
    if (!d.zero) { g0 = G[2 * q]; g1 = G[2 * q + 1]; h0 = G[2 * uu]; h1 = G[2 * uu + 1]; }
    if (Pp) {
      p0 = Pp[2 * q]; p1 = Pp[2 * q + 1]; m0 = Mm[2 * q]; m1 = Mm[2 * q + 1];
      r0 = Pp[2 * uu]; r1 = Pp[2 * uu + 1]; k0 = Mm[2 * uu]; k1 = Mm[2 * uu + 1];
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (s >= n) break;
      add8(g0, g1, w[s]);
      add8(h0, h1, x[s]);
    }
#pragma unroll 1
    for (int s = 8; s < n; ++s) {
      add8(g0, g1, __ldcs(P + (long long)(n - 1 - s) * slot + q));
      add8(h0, h1, __ldcs(P + (long long)(n - 1 - s) * slot + uu));
    }
    finish(q, g0, g1, p0, p1, m0, m1);
    if (two) finish(u, h0, h1, r0, r1, k0, k1);
  }
}

template <int NE, class Opt = NoOpt>
DEV void dsum(const DSum* E, int count, float** args, float* arena) {
  const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x, nt = (long long)gridDim.x * blockDim.x;
  const int n = count > 0 ? count : 0;
#pragma unroll 1
  for (int e = 0; e < NE; ++e) {
    const DSum d = E[e];
#ifdef STEP06_DSUM8
    if ((d.quads & 1) == 0 && (d.p_word & 3) == 0 && (d.slot_words & 3) == 0 && (d.g_word & 3) == 0) {
      dsum8<Opt>(d, n, args, arena, tid, nt);
      continue;
    }
#endif
    float4* G = reinterpret_cast<float4*>(arena + d.g_word);
    const uint2* P = reinterpret_cast<const uint2*>(arena + d.p_word);
    const long long slot = d.slot_words / 2;                        // uint2 units per slot
#pragma unroll 1
    for (long long q = tid; q < d.quads; q += nt) {
      float4 g = d.zero ? make_float4(0.f, 0.f, 0.f, 0.f) : G[q];
      uint2 w[8];
#pragma unroll
      for (int s = 0; s < 8; ++s) if (s < n) w[s] = __ldcs(P + (long long)(n - 1 - s) * slot + q);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        if (s >= n) break;
        g.x = g.x + __uint_as_float(w[s].x << 16);
        g.y = g.y + __uint_as_float(w[s].x & 0xffff0000u);
        g.z = g.z + __uint_as_float(w[s].y << 16);
        g.w = g.w + __uint_as_float(w[s].y & 0xffff0000u);
      }
#pragma unroll 1
      for (int s = 8; s < n; ++s) {                                // counts above 8: same order, one slot at a time
        const uint2 v = __ldcs(P + (long long)(n - 1 - s) * slot + q);
        g.x = g.x + __uint_as_float(v.x << 16);
        g.y = g.y + __uint_as_float(v.x & 0xffff0000u);
        g.z = g.z + __uint_as_float(v.y << 16);
        g.w = g.w + __uint_as_float(v.y & 0xffff0000u);
      }
      G[q] = g;
      if (d.opt >= 0) {
        float4* Pp = reinterpret_cast<float4*>(args[d.p_arg]);
        float4* Mm = reinterpret_cast<float4*>(args[d.m_arg]);
        const float4 pv = Pp[q], mv = Mm[q];
        float4 pn, mn;
        Opt::apply(d.opt, g.x, pv.x, mv.x, pn.x, mn.x);
        Opt::apply(d.opt, g.y, pv.y, mv.y, pn.y, mn.y);
        Opt::apply(d.opt, g.z, pv.z, mv.z, pn.z, mn.z);
        Opt::apply(d.opt, g.w, pv.w, mv.w, pn.w, mn.w);
        Pp[q] = pn;
        Mm[q] = mn;
      }
    }
  }
}
// LATCH-STEP-06 (loop_dgrad.BG): the ordered sum split into chunks of whole quads. During the final reverse step a
// CTA that waits at a grid barrier claims chunks whose weight's last product completed in an earlier level; block C
// drains the rest. Every chunk is summed exactly once, element by element in the same order (r = count-1 .. 0),
// with the same optimizer expressions: results are those of dsum.
struct BgChunk { int entry; int q0, q1; };     // quads [q0, q1) of entry
__device__ unsigned step06_bg_next;              // claimed chunks (reset at kernel entry)

template <class Opt>
DEV void chunk_sum(const DSum& d, int n, float** args, float* arena, int q0, int q1) {
  if ((d.quads & 1) == 0 && (q0 & 1) == 0 && (q1 & 1) == 0 && (d.p_word & 3) == 0 && (d.slot_words & 3) == 0
      && (d.g_word & 3) == 0) {
    // Octets [q0 / 2, q1 / 2) with the 8-wide pass (one CTA; same per-element arithmetic).
    dsum8<Opt>(d, n, args, arena, (long long)threadIdx.x, (long long)blockDim.x, q0 / 2, q1 / 2);
    return;
  }
  float4* G = reinterpret_cast<float4*>(arena + d.g_word);
  const uint2* P = reinterpret_cast<const uint2*>(arena + d.p_word);
  const long long slot = d.slot_words / 2;
  float4* Pp = d.opt >= 0 ? reinterpret_cast<float4*>(args[d.p_arg]) : nullptr;
  float4* Mm = d.opt >= 0 ? reinterpret_cast<float4*>(args[d.m_arg]) : nullptr;
#pragma unroll 1
  for (int q = q0 + (int)threadIdx.x; q < q1; q += (int)blockDim.x) {
    uint2 w[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) if (s < n) w[s] = __ldcs(P + (long long)(n - 1 - s) * slot + q);
    float4 g = d.zero ? make_float4(0.f, 0.f, 0.f, 0.f) : G[q];
    float4 pv = make_float4(0.f, 0.f, 0.f, 0.f), mv = pv;
    if (Pp) { pv = Pp[q]; mv = Mm[q]; }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      if (s >= n) break;
      g.x = g.x + __uint_as_float(w[s].x << 16);
      g.y = g.y + __uint_as_float(w[s].x & 0xffff0000u);
      g.z = g.z + __uint_as_float(w[s].y << 16);
      g.w = g.w + __uint_as_float(w[s].y & 0xffff0000u);
    }
#pragma unroll 1
    for (int s = 8; s < n; ++s) {
      const uint2 v = __ldcs(P + (long long)(n - 1 - s) * slot + q);
      g.x = g.x + __uint_as_float(v.x << 16);
      g.y = g.y + __uint_as_float(v.x & 0xffff0000u);
      g.z = g.z + __uint_as_float(v.y << 16);
      g.w = g.w + __uint_as_float(v.y & 0xffff0000u);
    }
    G[q] = g;
    if (Pp) {
      float4 pn, mn;
      Opt::apply(d.opt, g.x, pv.x, mv.x, pn.x, mn.x);
      Opt::apply(d.opt, g.y, pv.y, mv.y, pn.y, mn.y);
      Opt::apply(d.opt, g.z, pv.z, mv.z, pn.z, mn.z);
      Opt::apply(d.opt, g.w, pv.w, mv.w, pn.w, mn.w);
      Pp[q] = pn;
      Mm[q] = mn;
    }
  }
}

// Claims while the barrier is still closed: thread 0 decides for the CTA (uniform), the CTA sums the chunk.
#ifndef STEP06_BG_INLINE
#define STEP06_BG_WAIT_ATTR __device__ __noinline__
#else
#define STEP06_BG_WAIT_ATTR __device__ __forceinline__   // no call inside the runtime loops (no ABI spills)
#endif
template <class Opt>
STEP06_BG_WAIT_ATTR void bg_wait(const unsigned long long* counter, unsigned long long target, int ready,
                                 const DSum* E, const BgChunk* C, int count, float** args, float* arena) {
  __shared__ int job_sh;
  const int n = count > 0 ? count : 0;
  for (;;) {
    if (threadIdx.x == 0) {
      int job = -1;
      if (s101h::ld_acq_u64(counter) < target) {
        const unsigned cur = *reinterpret_cast<volatile unsigned*>(&step06_bg_next);
        if (cur < (unsigned)ready) job = atomicCAS(&step06_bg_next, cur, cur + 1u) == cur ? (int)cur : -2;
      }
      job_sh = job;
    }
    __syncthreads();
    const int job = job_sh;
    __syncthreads();
    if (job == -1) return;
    if (job >= 0) {
      const BgChunk c = C[job];
      chunk_sum<Opt>(E[c.entry], n, args, arena, c.q0, c.q1);
    }
  }
}

// Block C: every chunk not yet claimed.
template <class Opt>
__device__ __noinline__ void bg_drain(const DSum* E, const BgChunk* C, int nch, int count, float** args, float* arena) {
  __shared__ int job_sh;
  const int n = count > 0 ? count : 0;
  for (;;) {
    if (threadIdx.x == 0) job_sh = (int)atomicAdd(&step06_bg_next, 1u);
    __syncthreads();
    const int job = job_sh;
    __syncthreads();
    if (job >= nch) return;
    const BgChunk c = C[job];
    chunk_sum<Opt>(E[c.entry], n, args, arena, c.q0, c.q1);
  }
}

// Out-of-line entry (no warpgroup MMA inside): the ordered sum keeps its own register allocation instead of
// adding to the single kernel function's pressure.
template <int NE, class Opt = NoOpt>
__device__ __noinline__ void dsum_out(const DSum* E, int count, float** args, float* arena) {
  dsum<NE, Opt>(E, count, args, arena);
}
}  // namespace step06
