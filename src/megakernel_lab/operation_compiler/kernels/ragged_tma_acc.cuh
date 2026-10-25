// LATCH-SELECTIVE-10 (ops.accumulation_rules ``accfuse``): latch_ragged::touter with an accumulating epilogue.
// Segment e's result tile is rounded to the BF16 pairs touter stores (one cvt.rn.bf16x2 per pair), exactly widened,
// masked by *flags[e] (+0 otherwise, as where(flag, g, zeros_like(g))) and added to acc_in[e] (FP32 [M, N],
// ldc = N); the sum is stored to acc_out[e], which may be acc_in[e]: each element is read, then written, by one
// thread. Tasks, k order, accumulation from scale-d = 0, zeroed segment tails and rounding are those of touter
// (kernels/ragged_tma.cuh); an empty segment adds the +0 (or masked +0) results touter would have stored.
#pragma once
#include "ragged_tma.cuh"

namespace latch_ragged {

template<int E, int BM, int BN, int WM, int NST>
DEV void touter_acc(const void* tA, const void* tB, const popsc::BArgs& g, const int64_t* offsets, int rows, float* smem,
                    int rot, const bool* const* flags, const float* const* acc_in, float* const* acc_out) {
  using C = pops4b::Cfg<BM, BN, WM>;
  constexpr int STAGE = C::A_BYTES + C::B_BYTES;
  static_assert(NST >= 2 && NST <= 8, "stages");
  if (offsets[0] != 0 || offsets[E] != rows) __trap();
  const unsigned base = (pops4b::sa(smem) + 1023u) & ~1023u;
  int* pre = reinterpret_cast<int*>(reinterpret_cast<char*>(smem) + (base - pops4b::sa(smem)) + NST * STAGE);
  int* off = pre + (E + 1);
  const int tid = threadIdx.x, tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN;
  __syncthreads();                                                 // earlier users of shared memory are done
  if (tid < E) {
    int lo, hi; bounds(offsets, tid, rows, lo, hi);
    off[tid] = lo;
    if (tid == E - 1) off[E] = hi;
  }
  __syncthreads();
  const int G = (int)gridDim.x, T = E * tm * tn;
  const int me = (int)(((unsigned)blockIdx.x + (unsigned)rot) % (unsigned)G);
  const int ntask = me < T ? (T - me + G - 1) / G : 0;
  if (ntask == 0) return;                                          // CTA-uniform; no barrier was used
  const int wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
  const int wgm = WM == 2 ? wg : 0, wgn = WM == 2 ? 0 : wg;
  struct Task { int e, lo, K, m0, n0, nk; };
  auto task = [&](int i, Task& t) {
    const int id = me + i * G, e = id / (tm * tn), local = id % (tm * tn);
    t.e = e; t.lo = off[e]; t.K = off[e + 1] - off[e]; t.nk = (t.K + 63) / 64;
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
  int li = 0, lk = 0, lq = 0;
  task(0, lt);
  auto issue = [&]() {                                             // thread 0 only
#pragma unroll 1
    while (li < ntask && lk >= lt.nk) { ++li; lk = 0; if (li < ntask) task(li, lt); }
    if (li >= ntask) return;
    const int kk = lk * 64, st = lq % NST;
    const unsigned dA = base + st * STAGE, dB = dA + C::A_BYTES, bar = bar0 + 8 * st;
    pops4b::bar_expect(bar, STAGE);
#pragma unroll
    for (int c = 0; c < BM / 64; ++c) pops4b::tma2d(dA + c * 8192, tA, lt.m0 + 64 * c, lt.lo + kk, bar);
#pragma unroll
    for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tB, lt.n0 + 64 * c, lt.lo + kk, bar);
    ++lk; ++lq;
  };
  if (tid == 0) {
#pragma unroll 1
    for (int s = 0; s < NST - 1; ++s) issue();
  }
  float acc[C::MT][C::R];
  int q = 0;
#pragma unroll 1
  for (int i = 0; i < ntask; ++i) {
    Task t;
    task(i, t);
#pragma unroll
    for (int a = 0; a < C::MT; ++a)
#pragma unroll
      for (int j = 0; j < C::R; ++j) acc[a][j] = 0.f;              // an empty segment adds these
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
        pops4b::async_fence();                                     // generic zero stores -> wgmma (async proxy)
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
    const bool fl = *flags[t.e];
    const float* ain = acc_in[t.e];
    float* aout = acc_out[t.e];
#pragma unroll
    for (int a = 0; a < C::MT; ++a) {
      const int row = t.m0 + wgm * C::WTM + a * 64 + warp * 16 + (lane >> 2);
#pragma unroll
      for (int j = 0; j < C::WTN / 8; ++j) {
        const int col = t.n0 + wgn * C::WTN + 8 * j + 2 * (lane & 3);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = row + 8 * h;
          if (r < g.M && col < g.N) {
            const uint32_t p = s101h::bpk(acc[a][4 * j + 2 * h], acc[a][4 * j + 2 * h + 1]);
            const float w0 = fl ? __uint_as_float(p << 16) : 0.0f;
            const float w1 = fl ? __uint_as_float(p & 0xffff0000u) : 0.0f;
            const long long at = (long long)r * g.ldc + col;
            const float2 old = *reinterpret_cast<const float2*>(ain + at);
            *reinterpret_cast<float2*>(aout + at) = make_float2(old.x + w0, old.y + w1);
          }
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
}

}  // namespace latch_ragged
