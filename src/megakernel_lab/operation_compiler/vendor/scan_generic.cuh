// LATCH-OPS selective-scan primitive: generic device implementation (C-OPS2-C, written for this lane).
//
// Any FP32/BF16 operands with arbitrary positive strides, any state count N (template), any channel count. One warp
// task = (sequence b, 32-channel group); a lane owns one channel and walks time sequentially with its N states in
// registers (the S108F "channel lanes" decomposition), without time chunks or staging. Accurate expf/log1pf.
// Backward: per 8-step block from the forward checkpoint, recompute the block's states, walk it in reverse; dB/dC are
// butterfly-summed over the warp's lanes into per-group partials [EG][2N][M] and summed in group order by reduce(), as
// are the per-sequence vector partials. Semantics, checkpoint layout and partial layouts match scan_p1a.cuh; this
// path is the conservative reference / fallback, not a performance path.
#pragma once
#include <cuda_bf16.h>
#include <stdint.h>

namespace latch_scan_generic {

#define LSG_DEV __device__ __forceinline__

LSG_DEV float ld(const float* p) { return *p; }
LSG_DEV float ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
LSG_DEV void st(float* p, float v) { *p = v; }
LSG_DEV void st(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

template <class T>
struct Args {
  const T *u, *dl, *z, *Bm, *Cm, *dout;
  long long u_b, u_t, u_e, dl_b, dl_t, dl_e, z_b, z_t, z_e, B_b, B_t, B_n, C_b, C_t, C_n, do_b, do_t, do_e;
  const float *A, *D, *bias;
  long long A_e, A_n, D_e, bias_e;
  const bool* rst;
  long long r_b, r_t;
  int softplus;
  T* out;                 // [M][E]
  float* ckpt;            // [B][NCK][N][E]
  T *du, *dz, *ddl, *dB, *dC;
  float *dA, *dD, *dbias;
  float *krp, *vp;        // [EG][2N][M], [B][E N + 2E]
  int B, T_, E, M, NCK;
};

LSG_DEV float softplus_(float x) { return x > 20.f ? x : log1pf(expf(x)); }
LSG_DEV float sigmoid_(float x) { return 1.f / (1.f + expf(-x)); }
LSG_DEV float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
  return v;
}

template <class T, int N>
__device__ __noinline__ void forward(const Args<T>& a) {
  const int lane = threadIdx.x & 31;
  const int warp = (int)(threadIdx.x >> 5) * gridDim.x + blockIdx.x, nwarps = (blockDim.x >> 5) * gridDim.x;
  const int EG = (a.E + 31) / 32;
  for (int task = warp; task < a.B * EG; task += nwarps) {
    const int b = task / EG, e = (task % EG) * 32 + lane;
    const bool live = e < a.E;
    const int ec = live ? e : 0;
    float Ar[N], h[N];
#pragma unroll
    for (int n = 0; n < N; ++n) { Ar[n] = a.A[ec * a.A_e + n * a.A_n]; h[n] = 0.f; }
    const float Dv = a.D[ec * a.D_e], bias = a.bias[ec * a.bias_e];
    for (int t = 0; t < a.T_; ++t) {
      if ((t & 7) == 0 && live)
#pragma unroll
        for (int n = 0; n < N; ++n) a.ckpt[((long long)(b * a.NCK + t / 8) * N + n) * a.E + e] = h[n];
      const float u = ld(a.u + b * a.u_b + t * a.u_t + ec * a.u_e);
      const float pre = ld(a.dl + b * a.dl_b + t * a.dl_t + ec * a.dl_e) + bias;
      const float zz = ld(a.z + b * a.z_b + t * a.z_t + ec * a.z_e);
      const float dl = a.softplus ? softplus_(pre) : pre;
      const bool rs = a.rst != nullptr && a.rst[b * a.r_b + t * a.r_t];
      float y = 0.f;
#pragma unroll
      for (int n = 0; n < N; ++n) {
        const float Bn = ld(a.Bm + b * a.B_b + t * a.B_t + n * a.B_n), Cn = ld(a.Cm + b * a.C_b + t * a.C_t + n * a.C_n);
        const float al = rs ? 0.f : expf(dl * Ar[n]);
        h[n] = al * h[n] + dl * u * Bn;
        y += Cn * h[n];
      }
      y += Dv * u;
      if (live) st(a.out + ((long long)b * a.T_ + t) * a.E + e, y * zz * sigmoid_(zz));
    }
  }
}

template <class T, int N>
__device__ __noinline__ void backward(const Args<T>& a) {
  const int lane = threadIdx.x & 31;
  const int warp = (int)(threadIdx.x >> 5) * gridDim.x + blockIdx.x, nwarps = (blockDim.x >> 5) * gridDim.x;
  const int EG = (a.E + 31) / 32;
  for (int task = warp; task < a.B * EG; task += nwarps) {
    const int b = task / EG, grp = task % EG, e = grp * 32 + lane;
    const bool live = e < a.E;
    const int ec = live ? e : 0;
    float Ar[N], R[N], dA[N];
#pragma unroll
    for (int n = 0; n < N; ++n) { Ar[n] = a.A[ec * a.A_e + n * a.A_n]; R[n] = 0.f; dA[n] = 0.f; }
    const float Dv = a.D[ec * a.D_e], bias = a.bias[ec * a.bias_e];
    float sD = 0.f, sbias = 0.f;
    for (int blk = a.NCK - 1; blk >= 0; --blk) {
      const int t0 = blk * 8, t1 = min(a.T_, t0 + 8);
      float hs[9][N];
#pragma unroll
      for (int n = 0; n < N; ++n) hs[0][n] = live ? a.ckpt[((long long)(b * a.NCK + blk) * N + n) * a.E + e] : 0.f;
      for (int t = t0; t < t1; ++t) {
        const int k = t - t0;
        const float u = ld(a.u + b * a.u_b + t * a.u_t + ec * a.u_e);
        const float pre = ld(a.dl + b * a.dl_b + t * a.dl_t + ec * a.dl_e) + bias;
        const float dl = a.softplus ? softplus_(pre) : pre;
        const bool rs = a.rst != nullptr && a.rst[b * a.r_b + t * a.r_t];
#pragma unroll
        for (int n = 0; n < N; ++n) {
          const float Bn = ld(a.Bm + b * a.B_b + t * a.B_t + n * a.B_n);
          hs[k + 1][n] = (rs ? 0.f : expf(dl * Ar[n])) * hs[k][n] + dl * u * Bn;
        }
      }
      for (int t = t1 - 1; t >= t0; --t) {
        const int k = t - t0;
        const long long m = (long long)b * a.T_ + t;
        const float u = live ? ld(a.u + b * a.u_b + t * a.u_t + ec * a.u_e) : 0.f;
        const float pre = ld(a.dl + b * a.dl_b + t * a.dl_t + ec * a.dl_e) + bias;
        const float zz = ld(a.z + b * a.z_b + t * a.z_t + ec * a.z_e);
        const float dO = live ? ld(a.dout + b * a.do_b + t * a.do_t + ec * a.do_e) : 0.f;
        const float dl = a.softplus ? softplus_(pre) : pre;
        const bool rs = a.rst != nullptr && a.rst[b * a.r_b + t * a.r_t];
        float s = Dv * u;
#pragma unroll
        for (int n = 0; n < N; ++n) s += ld(a.Cm + b * a.C_b + t * a.C_t + n * a.C_n) * hs[k + 1][n];
        const float sg = sigmoid_(zz);
        const float ds = dO * zz * sg;
        const float dz = dO * s * sg * (1.f + zz * (1.f - sg));
        sD += ds * u;
        float du = ds * Dv, ddl = 0.f;
#pragma unroll
        for (int n = 0; n < N; ++n) {
          const float Bn = ld(a.Bm + b * a.B_b + t * a.B_t + n * a.B_n), Cn = ld(a.Cm + b * a.C_b + t * a.C_t + n * a.C_n);
          const float dh = R[n] + Cn * ds;
          const float al = rs ? 0.f : expf(dl * Ar[n]);
          const float ada = dh * al * hs[k][n];
          ddl += ada * Ar[n] + dh * u * Bn;
          du += dh * dl * Bn;
          dA[n] += ada * dl;
          const float pB = warp_sum(dh * dl * u), pC = warp_sum(hs[k + 1][n] * ds);
          if (lane == 0) {
            a.krp[((long long)grp * 2 * N + n) * a.M + m] = pB;
            a.krp[((long long)grp * 2 * N + N + n) * a.M + m] = pC;
          }
          R[n] = al * dh;
        }
        const float dp = a.softplus ? ddl * (pre > 20.f ? 1.f : sigmoid_(pre)) : ddl;
        sbias += dp;
        if (live) {
          st(a.du + m * a.E + e, du);
          st(a.dz + m * a.E + e, dz);
          st(a.ddl + m * a.E + e, dp);
        }
      }
    }
    if (live) {
      float* out = a.vp + (long long)b * (a.E * N + 2 * a.E);
#pragma unroll
      for (int n = 0; n < N; ++n) out[(long long)n * a.E + e] = dA[n];
      out[a.E * N + e] = sbias;
      out[a.E * N + a.E + e] = sD;
    }
  }
}

template <class T, int N>
__device__ __noinline__ void reduce(const Args<T>& a) {
  const int EG = (a.E + 31) / 32, stride = gridDim.x * blockDim.x;
  const long long items = (long long)a.M * 2 * N;
  for (long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x; it < items; it += stride) {
    const long long m = it % a.M;
    const int n2 = (int)(it / a.M);
    float s = 0.f;
    for (int g = 0; g < EG; ++g) s += a.krp[((long long)g * 2 * N + n2) * a.M + m];
    st(n2 < N ? a.dB + m * N + n2 : a.dC + m * N + (n2 - N), s);
  }
  const int C = a.E * N + 2 * a.E, EN = a.E * N;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < C; c += stride) {
    float s = 0.f;
    for (int r = 0; r < a.B; ++r) s += a.vp[(long long)r * C + c];
    if (c < EN) a.dA[(c % a.E) * N + c / a.E] = s;
    else if (c < EN + a.E) a.dbias[c - EN] = s;
    else a.dD[c - EN - a.E] = s;
  }
}

#undef LSG_DEV
}  // namespace latch_scan_generic
