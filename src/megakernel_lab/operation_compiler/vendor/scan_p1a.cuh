// LATCH-OPS selective-scan primitive: accepted P1-A scan body adapted to an operation interface (C-OPS2-C).
//
// Source: P1-A frozen candidate p1a-freeze-v1 (tag p1a-final, commit 96f20b9f4, /private/tmp/megakernel-p1a
// research/p1/P1-A/ssm/gen/device): scan2.cuh sha256 23e8e5ce52ce3d21bda91ac54c8edad3641740b5fb6daef72674507ac71cb43c,
// scan.cuh 2470be09c84e579b1756913d8be4501a150b469ae2cc206cdbbd55fca9d5fb91, common.cuh
// 10b4ac62d8a1fc5352dee9ab25308dafd9fbdf4669182e07be7500420c7eeb2a. Lineage: S108F -> S109D v2 -> S110A -> P1-A.
// Kept unchanged: the warp-task decomposition (sequence b, 32-channel group, time chunk), chunk/carry-piece rules,
// per-warp double-buffered cp.async staging, 8-step checkpoints, register recurrences with MUFU ex2 (decay via the
// exponent, resets as dl = +inf), the backward 8-step sub-chunk recomputation with G = 2 state groups, the
// recursive-halving reduce-scatter of dB/dC into per-32-channel partials summed in group order, and the vector-gradient
// partial rows summed in row order. Options fixed to the frozen candidate: P1A_NOSG on (sigma(q) recomputed),
// S109D_FLAGS 0 (carry phases end at a grid barrier issued by the generated caller), S109D_PEX 0, S109D_BPRO 0,
// S109D_CTRIM 2, DLR/DLS off.
// Changed for the operation interface: operands are separate BF16 row pointers with (batch, step) element strides
// (u, z, delta, B, C, dout) instead of one packed z [M][3E+2N]; outputs are separate tensors (out, du, dz, ddelta, dB,
// dC [M][*]); the operation takes A (= -exp(a_log) in the s99k model) and returns dA, so lam = -A replaces exp(a_log)
// and the vector partial carries +dA instead of -dA*exp(a_log); chunk geometry is computed on the device from
// gridDim.x (the generated grid is chosen at launch); warps per CTA and per-warp shared bytes come from the
// generator's staging footprint for a 256-thread CTA. Arithmetic order within each task is unchanged.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace latch_scan_p1a {

#define LSP_DEV __device__ __forceinline__
#define LSP_NI __device__ __noinline__
// P-OPS4-G contraction policy. The operation compiler builds with --fmad=false; P1-A was built with nvcc's default
// contraction. LSP_CONTRACT restores single-rounding multiply-adds at the body's a*b+c sites explicitly (the
// LSP_FMA2_SECOND form fuses the second product of a*b+c*d instead of the first). Without it every macro expands to
// the original uncontracted expression, operation for operation.
#ifdef LSP_CONTRACT
#define LSP_FMA(a, b, c) __fmaf_rn((a), (b), (c))
#ifdef LSP_FMA2_SECOND
#define LSP_FMA2(a, b, c, d) __fmaf_rn((c), (d), (a) * (b))
#else
#define LSP_FMA2(a, b, c, d) __fmaf_rn((a), (b), (c) * (d))
#endif
#define LSP_FMS(a, b, c) __fmaf_rn(-(a), (b), (c))
#else
#define LSP_FMA(a, b, c) ((a) * (b) + (c))
#define LSP_FMA2(a, b, c, d) ((a) * (b) + (c) * (d))
#define LSP_FMS(a, b, c) ((c) - (a) * (b))
#endif

constexpr int CK = 8;          // checkpoint interval = backward sub-chunk
constexpr int FB = 16;         // forward staging block steps
constexpr float LOG2E = 1.4426950408889634f;

struct Args {
  const __nv_bfloat16 *u, *z, *dl, *Bm, *Cm, *dout;
  long long u_b, u_t, z_b, z_t, dl_b, dl_t, B_b, B_t, C_b, C_t, do_b, do_t;
  const unsigned char* rst;            // bool [B][T] contiguous, or null (no resets)
  const float *A, *D, *bias;           // A [E][N] contiguous, D / bias [E]
  __nv_bfloat16* out;                  // forward [M][E]
  float* ckpt;                         // [B][NCK][N][E]
  __nv_bfloat16 *du, *dz, *ddl, *dB, *dC;   // backward [M][E] x3, [M][N] x2
  float *dA, *dD, *dbias;              // [E][N], [E], [E]
  float* carry;                        // L | S fragments, sized for ntmax chunks
  float *krp, *vp;                     // [E/32][2N][M], [B ntmax][E N + 2E]
  int B, T, E, M, NCK, ntmax;
  // P6-E output placement: row strides (elements) of du | dz | ddelta and of dB | dC (E and N when unplaced).
  long long o_t, bc_t;
  int canon0;
  // active warps per CTA / shared bytes per warp: forward (carry = main staging), backward main, backward carry
  int wf, bytes_f, wbm, bytes_bm, wbc, bytes_bc;
};

LSP_DEV float bfu(const unsigned char* p) { return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p)); }
LSP_DEV float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
LSP_DEV float rcpa(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
LSP_DEV float lg2f(float x) { float y; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
LSP_DEV float fsig2(float x) { return rcpa(1.f + ex2(-x * LOG2E)); }
LSP_DEV float fsp2(float x) {
  const float e = __expf(fminf(x, 20.f));
  const float ser = e * LSP_FMS(e, LSP_FMS(e, 0.33333334f, 0.5f), 1.f);
  const float lg = lg2f(1.f + e) * 6.9314718e-1f;
  return x > 20.f ? x : (e < 1e-2f ? ser : lg);
}
// lam = -A; donor lam2_of(a_log) = fminf(-exp(a_log) * LOG2E, -FLT_MIN): the clamp keeps reset (dl = +inf) exactly 0.
LSP_DEV float lam2_of_A(float a) { return fminf(a * LOG2E, -1.17549435e-38f); }
LSP_DEV unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
LSP_DEV void cp16(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
LSP_DEV void cp8(unsigned dst, const void* src, int bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::"r"(dst), "l"(src), "r"(bytes) : "memory");
}
LSP_DEV void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int K> LSP_DEV void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(K) : "memory"); }
// P6-E: wait until at most k (0..3, warp-uniform) committed groups are pending
LSP_DEV void cp_wait_n(int k) {
  if (k >= 3) cp_wait<3>(); else if (k == 2) cp_wait<2>(); else if (k == 1) cp_wait<1>(); else cp_wait<0>();
}
// P6-E forward staging ring: LSP_FNB buffers of FB steps per warp (2 = the donor double buffer); block jb + FNB - 1 is
// issued while block jb is consumed. Same copies to the same per-block layout; only how early they are issued.
#ifndef LSP_FNB
#define LSP_FNB 2
#endif
// P6-E backward-main staging ring: LSP_BNB buffers of CK steps (2 = donor); sub-chunk sc - (BNB - 1) is issued while
// sub-chunk sc (reverse order) is consumed.
#ifndef LSP_BNB
#define LSP_BNB 2
#endif
LSP_DEV int cdiv(int a, int b) { return (a + b - 1) / b; }
// P6-E: with Args.canon0 the BF16 gradient outputs store -0 as +0, which is what the reference's assembly of the
// outputs into one fused gradient (exact additions of +0 partial sums) produces; every other value is unchanged.
LSP_DEV __nv_bfloat16 lsp_c0(__nv_bfloat16 x, int c) {
  return (c && __bfloat16_as_ushort(x) == (unsigned short)0x8000u) ? __ushort_as_bfloat16((unsigned short)0) : x;
}
LSP_DEV __nv_bfloat162 lsp_c02(__nv_bfloat162 x, int c) { return __halves2bfloat162(lsp_c0(x.x, c), lsp_c0(x.y, c)); }

// Donor model.chunk_rule / carry_pieces evaluated on the device (W = this sub-phase's warps, grid = gridDim.x).
struct Geo { int nt, TCh, NP, PL; };
LSP_DEV Geo chunks(const Args& a, int W_main, int W_carry, bool fwd) {
  Geo g;
  const int lanes32 = a.B * (a.E / 32), grid = (int)gridDim.x;
  int want = max(1, (W_main * grid) / lanes32);
#ifdef LSP_NT_MIN
  want = max(want, LSP_NT_MIN);   // P6-E experiment: floor on time chunks (same arithmetic per chunk)
#endif
#ifdef LSP_NT_MIN_BWD
  if (!fwd) want = max(want, LSP_NT_MIN_BWD);   // P6-E: backward-only floor (backward chunks are independent)
#endif
  want = max(1, min(min(want, a.T / FB), a.ntmax));
  g.TCh = cdiv(cdiv(a.T, want), FB) * FB;
  g.nt = cdiv(a.T, g.TCh);
  if (g.nt <= 1) { g.NP = max(1, g.nt - 1); g.PL = g.TCh; return g; }
  const int span = fwd ? (g.nt - 1) * g.TCh : a.T - g.TCh;
  const int lo = g.nt - 1, hi = 2 * (g.nt - 1);
  int NP = min(hi, max(lo, (W_carry * grid) / lanes32));
  g.PL = cdiv(max(cdiv(span, NP), cdiv(g.TCh, 2)), CK) * CK;
  g.NP = cdiv(span, g.PL);
  return g;
}

// Per-warp step staging (donor Stage2). Buffer: [H: N x 128 B checkpoint (backward main)] [SB steps x (NR x 64 B rows,
// 4N B B|C)] [SB reset bytes]. Rows: 0 u, 1 z, 2 delta, 3 dout. RM: staged rows; KRM: staged B|C halves (1 B, 2 C, 3).
template <int N, int NR, int SB, bool HST, int RM = (1 << NR) - 1, int KRM = 3>
struct Stage {
  static_assert(RM == (1 << NR) - 1 || RM == 5 || RM == 14, "row mask");
  static_assert(N % 8 == 0, "B|C pieces are 16-byte halves");
  static constexpr int NRL = RM == 5 ? 2 : (RM == 14 ? 3 : NR);
  static constexpr int KRC = KRM == 3 ? N / 4 : N / 8;
  static constexpr int KR0 = KRM == 2 ? N / 8 : 0;
  LSP_DEV static int row_of(int i) { return RM == 5 ? 2 * i : (RM == 14 ? i + 1 : i); }
  static constexpr int HB = HST ? N * 128 : 0;
  static constexpr int KRO = NR * 64;
  static constexpr int STEP = KRO + 4 * N;
  static constexpr int BYTES = HB + SB * STEP + SB;
  static constexpr int BUF = (BYTES + 15) / 16 * 16;
  // P-OPS4-G staging roles. Every lane owns one fixed 16-byte piece of a step (an operand row quarter, or a B|C
  // piece) for the whole task: its source pointer, step stride and shared offset are set once here, in registers.
  // A staging block is then a short unrolled, branch-free run of cp.async, one per INC steps, instead of a strided
  // loop that re-derived (step, piece) by division, re-selected the operand and diverged between row and B|C pieces
  // on every iteration (~1/3 of the main passes' stall samples, gpu-04 SASS). The same bytes land at the same shared
  // addresses, so the arithmetic that consumes them is unchanged.
  static constexpr int RP = NRL * 4;           // row pieces per step
  static constexpr int PP = RP + KRC;          // all pieces per step (<= 24)
  static constexpr int INC = 32 / PP;          // steps covered per issue round (1 or 2)
  static constexpr int ACT = INC * PP;         // lanes with a staging role
  static constexpr int ITER = SB / INC;
  static_assert(PP <= 32 && SB % INC == 0, "staging roles");
  const unsigned char* src;                    // this lane's piece at step 0 of this task's sequence
  long long stride;                            // its step stride in bytes
  int off, s0;                                 // shared offset within a step, first step of the lane
  const unsigned char* rst;
  const float* hc;
  const void* any;                             // valid address for zero-byte copies
  long long rowb;
  int T, E, NCK, b, e0;
  LSP_DEV Stage(const Args& a, int b_, int e0_)
      : rst(a.rst), hc(a.ckpt), any(a.u), rowb((long long)b_ * a.T), T(a.T), E(a.E), NCK(a.NCK), b(b_), e0(e0_) {
    const int lane = threadIdx.x & 31, p = lane % PP;
    s0 = lane / PP;
    if (p < RP) {
      const int r = row_of(p >> 2), q = p & 3;
      const __nv_bfloat16* base = r == 0 ? a.u + b_ * a.u_b : r == 1 ? a.z + b_ * a.z_b
                                : r == 2 ? a.dl + b_ * a.dl_b : a.dout + b_ * a.do_b;
      const long long st = r == 0 ? a.u_t : r == 1 ? a.z_t : r == 2 ? a.dl_t : a.do_t;
      src = reinterpret_cast<const unsigned char*>(base + e0_ + 8 * q);
      stride = 2 * st;
      off = r * 64 + 16 * q;
    } else {
      const int q = p - RP + KR0;
      src = q < N / 8 ? reinterpret_cast<const unsigned char*>(a.Bm + b_ * a.B_b + 8 * q)
                      : reinterpret_cast<const unsigned char*>(a.Cm + b_ * a.C_b + 8 * (q - N / 8));
      stride = 2 * (q < N / 8 ? a.B_t : a.C_t);
      off = KRO + 16 * q;
    }
  }
  LSP_DEV void issue(unsigned dst, int t0) const {
    const int lane = threadIdx.x & 31;
    if (HST) {
      const float* hp = hc + ((long long)(b * NCK + t0 / CK) * N) * E + e0;
#pragma unroll
      for (int idx = lane; idx < N * 8; idx += 32) cp16(dst + idx * 16, hp + (long long)(idx >> 3) * E + 4 * (idx & 7), 16);
    }
    if (lane < ACT) {
#pragma unroll
      for (int k = 0; k < ITER; ++k) {
        const int s = s0 + k * INC, t = t0 + s;
        const bool ok = t < T;
        cp16(dst + HB + s * STEP + off, src + (ok ? t : 0) * stride, ok ? 16 : 0);
      }
    }
    for (int q = lane; q < SB / 8; q += 32) {
      const int t = t0 + 8 * q;
      const bool ok = t < T && rst != nullptr;
      cp8(dst + HB + SB * STEP + 8 * q, ok ? (const void*)(rst + rowb + t) : any, ok ? 8 : 0);
    }
    cp_commit();
  }
  LSP_DEV static void widen(const unsigned char* buf, float* kr) {
    const int lane = threadIdx.x & 31;
    constexpr int V8 = SB * KRC;
    for (int idx = lane; idx < V8; idx += 32) {
      const int s = idx / KRC, q = idx % KRC + KR0;
      const uint4 u = *reinterpret_cast<const uint4*>(buf + HB + s * STEP + KRO + 16 * q);
      float4 lo, hi;
      lo.x = __uint_as_float(u.x << 16); lo.y = __uint_as_float(u.x & 0xffff0000u);
      lo.z = __uint_as_float(u.y << 16); lo.w = __uint_as_float(u.y & 0xffff0000u);
      hi.x = __uint_as_float(u.z << 16); hi.y = __uint_as_float(u.z & 0xffff0000u);
      hi.z = __uint_as_float(u.w << 16); hi.w = __uint_as_float(u.w & 0xffff0000u);
      float4* d = reinterpret_cast<float4*>(kr + s * 2 * N + 8 * q);
      d[0] = lo;
      d[1] = hi;
    }
  }
  LSP_DEV static void widen_pairs(const unsigned char* buf, float* kr) {
    const int lane = threadIdx.x & 31;
    constexpr int NP = SB * N / 2;
    for (int idx = lane; idx < NP; idx += 32) {
      const int s = idx / (N / 2), q = idx % (N / 2);
      const unsigned char* src = buf + HB + s * STEP + KRO;
      const unsigned uk = *reinterpret_cast<const unsigned*>(src + 4 * q);
      const unsigned ur = *reinterpret_cast<const unsigned*>(src + 2 * N + 4 * q);
      float4 o;
      o.x = __uint_as_float(uk << 16); o.z = __uint_as_float(uk & 0xffff0000u);
      o.y = __uint_as_float(ur << 16); o.w = __uint_as_float(ur & 0xffff0000u);
      reinterpret_cast<float4*>(kr)[s * (N / 2) + q] = o;
    }
  }
};

// ------------------------------------------------------------------------------------------------ forward
// CARRY: chunk-local end states / decay sums of the carried span (fragments, <= 3 per chunk); main: chunk start state
// from the earlier fragments, outputs and checkpoints. The caller issues grid.sync() between the two.
template <int N, bool CARRY>
LSP_NI void forward(const Args& args, unsigned char* smem) {
  const Args a = args;   // P-OPS4-G: by-value copy; fields are promoted to registers
  using S = Stage<N, 3, FB, false, 7, 3>;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  if (w >= a.wf) return;
  const Geo g = chunks(a, a.wf, a.wf, true);
  const int EG = a.E / 32, E = a.E;
  float* Lc = a.carry;
  float* Sc = a.carry + 3LL * a.B * g.nt * N * E;
  unsigned char* wbase = smem + w * a.bytes_f;
  float* krf = reinterpret_cast<float*>(wbase + LSP_FNB * S::BUF);
  const unsigned sb0 = smem_addr(wbase);
  const int nw = a.wf * gridDim.x;
  const int ntask = a.B * EG * (CARRY ? g.NP : g.nt);
  if (CARRY && g.nt <= 1) return;
  for (int task = w * gridDim.x + blockIdx.x; task < ntask; task += nw) {
    const int grp = task % EG, rest = task / EG, b = rest % a.B, j = rest / a.B;
    const int e0 = grp * 32, e = e0 + lane;
    const int NP = g.NP, PL = g.PL;
    S st(a, b, e0);
    float lam2[N], h[N];
#pragma unroll
    for (int n = 0; n < N; ++n) { lam2[n] = lam2_of_A(a.A[e * N + n]); h[n] = 0.f; }
    const float db = a.bias[e], Dk = a.D[e];
    if (!CARRY) {
      st.issue(sb0, j * g.TCh);
      constexpr int UJ = N <= 16 ? 2 : 1;
      float suf = 0.f;
#pragma unroll UJ
      for (int jj = j - 1; jj >= 0; --jj) {
        const int p0 = jj * g.TCh / PL, p1 = min(NP - 1, ((jj + 1) * g.TCh - 1) / PL);
        const long long fb = (long long)(b * g.nt + jj) * 3;
#pragma unroll
        for (int f = 2; f >= 0; --f)
          if (p0 + f <= p1) {
            const float Sf = Sc[(fb + f) * E + e];
            const float* Lp = Lc + (fb + f) * N * E + e;
#pragma unroll
            for (int n = 0; n < N; ++n) h[n] = LSP_FMA(ex2(suf * lam2[n]), Lp[(long long)n * E], h[n]);
            suf += Sf;
          }
      }
    }
    float Ssum = 0.f;
    auto segment = [&](const int ta, const int tb, const bool issued) {
    const int nblk = (tb - ta + FB - 1) / FB;
    if (nblk > 0 && !issued) st.issue(sb0, ta);
    constexpr int NB = LSP_FNB;
#pragma unroll
    for (int k = 1; k < NB - 1; ++k)
      if (k < nblk) st.issue(sb0 + k * S::BUF, ta + k * FB);
    for (int jb = 0; jb < nblk; ++jb) {
      if (NB == 2) {
        if (jb + 1 < nblk) { st.issue(sb0 + ((jb + 1) & 1) * S::BUF, ta + (jb + 1) * FB); cp_wait<1>(); }
        else cp_wait<0>();
      } else {
        if (jb + NB - 1 < nblk) st.issue(sb0 + ((jb + NB - 1) % NB) * S::BUF, ta + (jb + NB - 1) * FB);
        cp_wait_n(min(NB - 1, nblk - 1 - jb));
      }
      __syncwarp();
      const unsigned char* bs = wbase + (NB == 2 ? (jb & 1) : jb % NB) * S::BUF;
      S::widen(bs, krf);
      __syncwarp();
      const int t0 = ta + jb * FB, len = min(FB, tb - t0);
      auto step = [&](const int s) {
        const int t = t0 + s;
        const unsigned char* sp = bs + s * S::STEP;
        if (!CARRY && (s % CK) == 0) {
          float* hp = a.ckpt + ((long long)(b * a.NCK + t / CK) * N) * E + e;
#pragma unroll
          for (int n = 0; n < N; ++n) __stcs(hp + (long long)n * E, h[n]);
        }
        const float v = bfu(sp + 2 * lane), dl = fsp2(bfu(sp + 128 + 2 * lane) + db), dlv = dl * v;
        const bool reset = bs[FB * S::STEP + s] != 0;
        const float dle = reset ? INFINITY : dl;
        if (CARRY) Ssum = reset ? INFINITY : Ssum + dl;
        const float4* k4 = reinterpret_cast<const float4*>(krf + s * 2 * N);
        const float4* r4 = k4 + N / 4;
        float y0 = 0.f, y1 = 0.f;
#pragma unroll
        for (int q = 0; q < N / 4; ++q) {
          const float4 kf = k4[q];
          const int n = 4 * q;
          h[n] = LSP_FMA2(ex2(dle * lam2[n]), h[n], dlv, kf.x);
          h[n + 1] = LSP_FMA2(ex2(dle * lam2[n + 1]), h[n + 1], dlv, kf.y);
          h[n + 2] = LSP_FMA2(ex2(dle * lam2[n + 2]), h[n + 2], dlv, kf.z);
          h[n + 3] = LSP_FMA2(ex2(dle * lam2[n + 3]), h[n + 3], dlv, kf.w);
          if (!CARRY) {
            const float4 rf = r4[q];
            y0 = LSP_FMA(rf.x, h[n], y0);
            y1 = LSP_FMA(rf.y, h[n + 1], y1);
            y0 = LSP_FMA(rf.z, h[n + 2], y0);
            y1 = LSP_FMA(rf.w, h[n + 3], y1);
          }
        }
        if (!CARRY) {
          const float q = bfu(sp + 64 + 2 * lane);
          const __nv_bfloat16 ov = __float2bfloat16_rn(LSP_FMA(Dk, v, y0 + y1) * q * fsig2(q));
          reinterpret_cast<__nv_bfloat16*>(const_cast<unsigned char*>(sp) + 64)[lane] = ov;
        }
      };
      if (len == FB) {
#ifndef LSP_FUNROLL
#define LSP_FUNROLL 4
#endif
#define LSP_PRAGMA_(x) _Pragma(#x)
#define LSP_UNROLL_(n) LSP_PRAGMA_(unroll n)
        LSP_UNROLL_(LSP_FUNROLL)
        for (int s = 0; s < FB; ++s) step(s);
      } else {
#pragma unroll 1
        for (int s = 0; s < len; ++s) step(s);
      }
      __syncwarp();
      if (!CARRY)
        for (int idx = lane; idx < len * 4; idx += 32) {
          const int s = idx >> 2, q = idx & 3;
          *reinterpret_cast<uint4*>(a.out + (st.rowb + t0 + s) * E + e0 + 8 * q) =
              *reinterpret_cast<const uint4*>(bs + s * S::STEP + 64 + 16 * q);
        }
      __syncwarp();
    }
    };
    if (!CARRY) {
      segment(j * g.TCh, min(a.T, (j + 1) * g.TCh), true);
    } else {
      const int pend = min(min(a.T, (g.nt - 1) * g.TCh), (j + 1) * PL);
      for (int ta = j * PL; ta < pend;) {
        const int jc = ta / g.TCh, tb = min(pend, (jc + 1) * g.TCh);
#pragma unroll
        for (int n = 0; n < N; ++n) h[n] = 0.f;
        Ssum = 0.f;
        segment(ta, tb, false);
        const long long fr = (long long)(b * g.nt + jc) * 3 + (j - jc * g.TCh / PL);
        float* Lp = Lc + fr * N * E + e;
#pragma unroll
        for (int n = 0; n < N; ++n) Lp[(long long)n * E] = h[n];
        Sc[fr * E + e] = Ssum;
        ta = tb;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ backward
template <int V, int H> LSP_DEV void rs_level_v(float (&v)[V], int lane) {
  const bool upper = (lane & H) != 0;
#pragma unroll
  for (int i = 0; i < H; ++i) {
    const float send = upper ? v[i] : v[i + H];
    const float kept = upper ? v[i + H] : v[i];
    v[i] = kept + __shfl_xor_sync(0xffffffffu, send, H);
  }
}
template <int V> LSP_DEV float rsv(float (&v)[V]) {
  const int lane = threadIdx.x & 31;
  if constexpr (V == 32) rs_level_v<V, 16>(v, lane);
  if constexpr (V >= 16) rs_level_v<V, 8>(v, lane);
  rs_level_v<V, 4>(v, lane);
  rs_level_v<V, 2>(v, lane);
  rs_level_v<V, 1>(v, lane);
  float r = v[0];
#pragma unroll
  for (int d = V; d < 32; d <<= 1) r += __shfl_xor_sync(0xffffffffu, r, d);
  return r;
}

// CARRY: per carried piece, the adjoint leaving each part start and the part decay sum; main: adjoint entering the
// chunk end from later fragments, 8-step sub-chunks in reverse from the checkpoints, outputs du | dz | ddelta, dB | dC
// partials and the task's vector-gradient partial row. Chunk geometry: donor rules with the backward main/carry warps.
template <int N, bool CARRY>
LSP_NI void backward(const Args& args, unsigned char* smem) {
  const Args a = args;   // P-OPS4-G: by-value copy; fields are promoted to registers
  // P-OPS4-G: LSP_BG states per backward-main group (default 2 = P1-A). Larger groups add independent recurrence chains
  // per warp (more registers, same arithmetic order: per-step sums still run over states in index order and dB|dC are
  // reduce-scattered per state pair exactly as with G = 2).
#ifndef LSP_BG
#define LSP_BG 2
#endif
  constexpr int G = LSP_BG, K = CK * 2;
  static_assert(G % 2 == 0 && N % G == 0, "backward state groups are whole state pairs");
  using S = Stage<N, 4, CK, !CARRY, CARRY ? 14 : 15, CARRY ? 2 : 3>;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int warps = CARRY ? a.wbc : a.wbm;
  if (w >= warps) return;
  const Geo g = chunks(a, a.wbm, a.wbc, false);
  if (CARRY && g.nt <= 1) return;
  const int E = a.E, EG = E / 32;
  float* Lb = a.carry;
  float* Sb = a.carry + 3LL * a.B * g.nt * N * E;
  unsigned char* wbase = smem + w * (CARRY ? a.bytes_bc : a.bytes_bm);
  float* krf = reinterpret_cast<float*>(wbase + (CARRY ? 2 : LSP_BNB) * S::BUF);
  const unsigned sb0 = smem_addr(wbase);
  const int nw = warps * gridDim.x;
  const int ntask = a.B * EG * (CARRY ? g.NP : g.nt);
  for (int task = w * gridDim.x + blockIdx.x; task < ntask; task += nw) {
    const int grp = task % EG, rest = task / EG, b = rest % a.B, j = rest / a.B;
    const int e0 = grp * 32, e = e0 + lane;
    const int NP = g.NP, PL = g.PL;
    const int ta = j * g.TCh, tb = min(a.T, ta + g.TCh);
    S st(a, b, e0);
    const float db = a.bias[e], Dk = a.D[e];
    const float* al = a.A + e * N;
    const int nsc = (tb - ta) / CK;
    if (CARRY) {
      float R[N], lam2[N];
#pragma unroll
      for (int n = 0; n < N; ++n) lam2[n] = lam2_of_A(al[n]);
      const int pend = min(a.T, g.TCh + (j + 1) * PL);
      for (int pa = g.TCh + j * PL; pa < pend;) {
      const int jc = pa / g.TCh, pb = min(pend, (jc + 1) * g.TCh), ta = pa, nsc = (pb - pa) / CK;
#pragma unroll
      for (int n = 0; n < N; ++n) R[n] = 0.f;
      float Ssum = 0.f;
      st.issue(sb0 + ((nsc - 1) & 1) * S::BUF, ta + (nsc - 1) * CK);
      for (int sc = nsc - 1; sc >= 0; --sc) {
        if (sc > 0) { st.issue(sb0 + ((sc - 1) & 1) * S::BUF, ta + (sc - 1) * CK); cp_wait<1>(); }
        else cp_wait<0>();
        __syncwarp();
        const unsigned char* bs = wbase + (sc & 1) * S::BUF;
        S::widen(bs, krf);
        __syncwarp();
#pragma unroll
        for (int i = CK - 1; i >= 0; --i) {
          const unsigned char* sp = bs + S::HB + i * S::STEP;
          const float q = bfu(sp + 64 + 2 * lane);
          const float dl = fsp2(bfu(sp + 128 + 2 * lane) + db);
          const float ds = bfu(sp + 192 + 2 * lane) * q * fsig2(q);
          const bool reset = bs[S::HB + CK * S::STEP + i] != 0;
          const float dle = reset ? INFINITY : dl;
          Ssum = reset ? INFINITY : Ssum + dl;
          const float4* r4 = reinterpret_cast<const float4*>(krf + i * 2 * N + N);
#pragma unroll
          for (int qq = 0; qq < N / 4; ++qq) {
            const float4 f = r4[qq];
            const int n = 4 * qq;
            R[n] = ex2(dle * lam2[n]) * LSP_FMA(f.x, ds, R[n]);
            R[n + 1] = ex2(dle * lam2[n + 1]) * LSP_FMA(f.y, ds, R[n + 1]);
            R[n + 2] = ex2(dle * lam2[n + 2]) * LSP_FMA(f.z, ds, R[n + 2]);
            R[n + 3] = ex2(dle * lam2[n + 3]) * LSP_FMA(f.w, ds, R[n + 3]);
          }
        }
        __syncwarp();
      }
      const long long fr = (long long)(b * g.nt + jc) * 3 + (j - (jc - 1) * g.TCh / PL);
      float* Lp = Lb + fr * N * E + e;
#pragma unroll
      for (int n = 0; n < N; ++n) Lp[(long long)n * E] = R[n];
      Sb[fr * E + e] = Ssum;
      pa = pb;
      }
      continue;
    }
    float* RS = krf + CK * 2 * N;                  // [N][32] per-warp adjoint carries
    float* dAs = RS + N * 32;                      // [N][32] dA accumulators
    constexpr bool LAMS = N <= 16;                 // [N][32] lam = -A per lane cached in shared memory (N <= 16)
    float* LAM = dAs + N * 32;
    {
      float R[N], lam2[N];
#pragma unroll
      for (int n = 0; n < N; ++n) { R[n] = 0.f; lam2[n] = lam2_of_A(al[n]); }
      for (int jj = g.nt - 1; jj > j; --jj) {
        const int p0 = (jj - 1) * g.TCh / PL, p1 = min(NP - 1, (min(a.T, (jj + 1) * g.TCh) - g.TCh - 1) / PL);
        for (int pp = p1; pp >= p0; --pp) {
          const long long fr = (long long)(b * g.nt + jj) * 3 + (pp - p0);
          const float Sj = Sb[fr * E + e];
          const float* Lp = Lb + fr * N * E + e;
#pragma unroll
          for (int n = 0; n < N; ++n) R[n] = LSP_FMA((Sj == INFINITY ? 0.f : ex2(Sj * lam2[n])), R[n], Lp[(long long)n * E]);
        }
      }
#pragma unroll
      for (int n = 0; n < N; ++n) {
        RS[n * 32 + lane] = R[n];
        dAs[n * 32 + lane] = 0.f;
        if (LAMS) LAM[n * 32 + lane] = -al[n];
      }
    }
    float sdt = 0.f, sD = 0.f;
    const int oi0 = lane / 12, ooff = ((lane % 12) >> 2) * 64 + 16 * (lane & 3);
    __nv_bfloat16* const obase = ((lane % 12) >> 2 == 0 ? a.du : ((lane % 12) >> 2 == 1 ? a.dz : a.ddl)) + e0 + 8 * (lane & 3);
    constexpr int BNB = LSP_BNB;
    if (BNB == 2) {
      st.issue(sb0 + ((nsc - 1) & 1) * S::BUF, ta + (nsc - 1) * CK);
    } else {
#pragma unroll
      for (int k = 0; k < BNB - 1; ++k)
        if (nsc - 1 - k >= 0) st.issue(sb0 + ((nsc - 1 - k) % BNB) * S::BUF, ta + (nsc - 1 - k) * CK);
    }
    for (int sc = nsc - 1; sc >= 0; --sc) {
      if (BNB == 2) {
        if (sc > 0) { st.issue(sb0 + ((sc - 1) & 1) * S::BUF, ta + (sc - 1) * CK); cp_wait<1>(); }
        else cp_wait<0>();
      } else {
        if (sc - (BNB - 1) >= 0) st.issue(sb0 + ((sc - (BNB - 1)) % BNB) * S::BUF, ta + (sc - (BNB - 1)) * CK);
        cp_wait_n(min(BNB - 1, sc));
      }
      __syncwarp();
      const unsigned char* bs = wbase + (BNB == 2 ? (sc & 1) : sc % BNB) * S::BUF;
      S::widen_pairs(bs, krf);
      __syncwarp();
      const float* hst_s = reinterpret_cast<const float*>(bs);
      const int ts0 = ta + sc * CK;
      float dl[CK], dle[CK], dlv[CK], ds[CK], s_[CK], xk[CK], la[CK];
#pragma unroll
      for (int i = 0; i < CK; ++i) {
        const unsigned char* sp = bs + S::HB + i * S::STEP;
        const float q = bfu(sp + 64 + 2 * lane), v = bfu(sp + 2 * lane);
        dl[i] = fsp2(bfu(sp + 128 + 2 * lane) + db);
        dle[i] = bs[S::HB + CK * S::STEP + i] ? INFINITY : dl[i];
        dlv[i] = dl[i] * v;
        ds[i] = bfu(sp + 192 + 2 * lane) * q * fsig2(q);
        s_[i] = Dk * v;
        xk[i] = 0.f;
        la[i] = 0.f;
      }
#pragma unroll 1
      for (int ng = 0; ng < N; ng += G) {
        float av[CK][G], hh[CK][G], hst[G], lam[G], lam2[G], R[G], dA[G];
#pragma unroll
        for (int u = 0; u < G; ++u) {
          hst[u] = hst_s[(ng + u) * 32 + lane];
          lam[u] = LAMS ? LAM[(ng + u) * 32 + lane] : -al[ng + u];
          lam2[u] = fminf(-lam[u] * LOG2E, -1.17549435e-38f);
          R[u] = RS[(ng + u) * 32 + lane];
          dA[u] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < CK; ++i) {
          float kk[G], rr[G];
#pragma unroll
          for (int p2 = 0; p2 < G / 2; ++p2) {
            const float4 f = reinterpret_cast<const float4*>(krf)[i * (N / 2) + ng / 2 + p2];
            kk[2 * p2] = f.x; rr[2 * p2] = f.y; kk[2 * p2 + 1] = f.z; rr[2 * p2 + 1] = f.w;
          }
#pragma unroll
          for (int u = 0; u < G; ++u) {
            av[i][u] = ex2(dle[i] * lam2[u]);
            hh[i][u] = LSP_FMA2(av[i][u], (i == 0 ? hst[u] : hh[i - 1][u]), dlv[i], kk[u]);
            s_[i] = LSP_FMA(rr[u], hh[i][u], s_[i]);
          }
        }
#pragma unroll
        for (int i = CK - 1; i >= 0; --i) {
          float kk[G], rr[G];
#pragma unroll
          for (int p2 = 0; p2 < G / 2; ++p2) {
            const float4 f = reinterpret_cast<const float4*>(krf)[i * (N / 2) + ng / 2 + p2];
            kk[2 * p2] = f.x; rr[2 * p2] = f.y; kk[2 * p2 + 1] = f.z; rr[2 * p2 + 1] = f.w;
          }
#pragma unroll
          for (int u = 0; u < G; ++u) {
            const float x = LSP_FMA(rr[u], ds[i], R[u]);
            const float Rn = av[i][u] * x;
            const float ada = Rn * (i == 0 ? hst[u] : hh[i - 1][u]);
            la[i] = LSP_FMA(lam[u], ada, la[i]);
            dA[u] = LSP_FMA(ada, dl[i], dA[u]);
            xk[i] = LSP_FMA(x, kk[u], xk[i]);
            av[i][u] = x * dlv[i];             // dB contribution
            hh[i][u] = hh[i][u] * ds[i];       // dC contribution
            R[u] = Rn;
          }
        }
#pragma unroll
        for (int u = 0; u < G; ++u) {
          RS[(ng + u) * 32 + lane] = R[u];
          dAs[(ng + u) * 32 + lane] += dA[u];
        }
#pragma unroll
        for (int p2 = 0; p2 < G / 2; ++p2) {
          float red[2 * K];
#pragma unroll
          for (int i = 0; i < CK; ++i)
#pragma unroll
            for (int u = 0; u < 2; ++u) { red[i * 2 + u] = av[i][2 * p2 + u]; red[K + i * 2 + u] = hh[i][2 * p2 + u]; }
          const float sum = rsv<2 * K>(red);
          const int idx = lane & (2 * K - 1), ii = (idx % K) / 2, uu = idx % 2, which = idx / K;
          if (2 * K == 32 || lane < 2 * K)
            a.krp[((long long)(e0 / 32) * 2 * N + which * N + ng + 2 * p2 + uu) * a.M + st.rowb + ts0 + ii] = sum;
        }
      }
#pragma unroll
      for (int i = 0; i < CK; ++i) {
        const unsigned char* sp = bs + S::HB + i * S::STEP;
        const float q = bfu(sp + 64 + 2 * lane), dO = bfu(sp + 192 + 2 * lane), v = bfu(sp + 2 * lane);
        const float pre = bfu(sp + 128 + 2 * lane) + db;
        const float sgi = fsig2(q);
        const float dq = dO * s_[i] * sgi * LSP_FMA(q, 1.f - sgi, 1.f);
        const float ddl = LSP_FMS(-v, xk[i], -la[i]);
        const float dp = ddl * (pre > 20.f ? 1.f : fsig2(pre));
        sdt += dp;
        sD = LSP_FMA(ds[i], v, sD);
        const __nv_bfloat16 z0 = lsp_c0(__float2bfloat16_rn(LSP_FMA2(dl[i], xk[i], ds[i], Dk)), a.canon0), z1 = lsp_c0(__float2bfloat16_rn(dq), a.canon0),
                            z2 = lsp_c0(__float2bfloat16_rn(dp), a.canon0);
        __nv_bfloat16* row = reinterpret_cast<__nv_bfloat16*>(const_cast<unsigned char*>(sp));
        row[lane] = z0;
        row[32 + lane] = z1;
        row[64 + lane] = z2;
      }
      __syncwarp();
      if (lane < 24) {                                         // fixed piece per lane: output r (du|dz|ddelta), quarter q
#pragma unroll
        for (int k = 0; k < CK / 2; ++k) {
          const int i = oi0 + 2 * k;
          const uint4 val = *reinterpret_cast<const uint4*>(bs + S::HB + i * S::STEP + ooff);
          *reinterpret_cast<uint4*>(obase + (st.rowb + ts0 + i) * a.o_t) = val;
        }
      }
      __syncwarp();
    }
    float* out = a.vp + ((long long)b * g.nt + j) * (E * N + 2 * E);
    for (int n = 0; n < N; ++n) out[(long long)n * E + e] = dAs[n * 32 + lane];
    out[E * N + e] = sdt;
    out[E * N + E + e] = sD;
    __syncwarp();
  }
}

// dB | dC[m][n] = bf16(sum over 32-channel groups in group order) (donor phase_kr2 kr_rows<1>), then the vector partial
// rows (B x backward chunks) summed in row order into dA [E][N], dbias, dD (donor tiled path for >= 64 rows).
template <int N>
LSP_NI void reduce(const Args& args, unsigned char* smem) {
  const Args a = args;   // P-OPS4-G: by-value copy; fields are promoted to registers
  const Geo g = chunks(a, a.wbm, a.wbc, false);
  const int M = a.M, N2 = 2 * N, EG = a.E / 32, E = a.E;
  const int MR = M / 4;
  const long long items = (long long)N * MR;
  const int stride = gridDim.x * blockDim.x;
  for (long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x; it < items; it += stride) {
    const int m = 4 * (int)(it % MR), n2 = 2 * (int)(it / MR);
    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0;
#pragma unroll 8
    for (int gg = 0; gg < EG; ++gg) {
      const float* r0 = a.krp + ((long long)gg * N2 + n2) * M + m;
      const float4 x0 = *reinterpret_cast<const float4*>(r0);
      const float4 x1 = *reinterpret_cast<const float4*>(r0 + M);
      a0.x += x0.x; a0.y += x0.y; a0.z += x0.z; a0.w += x0.w;
      a1.x += x1.x; a1.y += x1.y; a1.z += x1.z; a1.w += x1.w;
    }
    const long long bt = a.bc_t;
    __nv_bfloat16* d = n2 < N ? a.dB + (long long)m * bt + n2 : a.dC + (long long)m * bt + (n2 - N);
    *reinterpret_cast<__nv_bfloat162*>(d) = lsp_c02(__floats2bfloat162_rn(a0.x, a1.x), a.canon0);
    *reinterpret_cast<__nv_bfloat162*>(d + bt) = lsp_c02(__floats2bfloat162_rn(a0.y, a1.y), a.canon0);
    *reinterpret_cast<__nv_bfloat162*>(d + 2 * bt) = lsp_c02(__floats2bfloat162_rn(a0.z, a1.z), a.canon0);
    *reinterpret_cast<__nv_bfloat162*>(d + 3 * bt) = lsp_c02(__floats2bfloat162_rn(a0.w, a1.w), a.canon0);
  }
  const int R = a.B * g.nt, C = E * N + 2 * E, EN = E * N;
  const int NWARP = blockDim.x >> 5;
  auto store = [&](int c, float t) {
    if (c < EN) a.dA[(c % E) * N + c / E] = t;
    else if (c < EN + E) a.dbias[c - EN] = t;
    else a.dD[c - EN - E] = t;
  };
  if (R >= 64) {
    float* part = reinterpret_cast<float*>(smem);            // [NWARP][32]
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int tiles = (C + 31) / 32;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int c = tile * 32 + lane;
      float s = 0.f;
      if (c < C)
#pragma unroll 4
        for (int r = w; r < R; r += NWARP) s += a.vp[(long long)r * C + c];
      part[w * 32 + lane] = s;
      __syncthreads();
      if (w == 0 && c < C) {
        float t = 0.f;
        for (int k = 0; k < NWARP; ++k) t += part[k * 32 + lane];
        store(c, t);
      }
      __syncthreads();
    }
  } else {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += stride) {
      float s = 0.f;
#pragma unroll 8
      for (int r = 0; r < R; ++r) s += a.vp[(long long)r * C + c];
      store(c, s);
    }
  }
}

#undef LSP_DEV
#undef LSP_NI
#undef LSP_FMA
#undef LSP_FMA2
#undef LSP_FMS
}  // namespace latch_scan_p1a
