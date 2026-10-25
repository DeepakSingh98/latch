// P-OPS-D fused FP32 attention work items for the LATCH-OPS cooperative program.
//
// Design lineage (attribution): the tile decomposition follows Latch v0.6.0 S101H
// "v2" FP32 SIMT attention (src/megakernel_releases/s103a/_frozen/d/vendor/s101h/
// native/lib/attn.cuh @ tag latch-v0.6.0: 64x64 tiles, 256 threads, tx=tid&15,
// ty=tid>>4 4x4 register blocks, half-warp row reductions, online softmax with
// LSE, fixed-order no-atomic backward), itself after FlashAttention-2 (Dao, 2023).
// No source is copied verbatim. Changes and restrictions that are NOT inherited:
//   * head width HD in {32, 64} (template), context T in [1, 4096] (ragged tiles
//     are zero-filled and excluded);
//   * a general additive mask tensor (any broadcast strides) instead of a
//     hard-coded causal predicate; optional multiply/divide score scaling with the
//     captured constant, rounded exactly as the captured graph does:
//     x = fl(fl(dot) op c) + mask;
//   * tile skipping only by a runtime proof that every skipped probability is an
//     exact FP32 zero in the reference (see skip_proof);
//   * deterministic dQ by a separate query-block pass (recomputation), no
//     partial-tile buffer, no atomics;
//   * delta = rowsum(dO * O) in a separate fixed-order phase.
// Arithmetic: FP32 fmaf in ascending index order; expf/logf (no fast math, the
// program is compiled with --fmad=false); no TF32, no tensor cores. Requires the
// program's fixed 256-thread CTA. All loads are ordinary coherent loads: the
// arena is reused across phases of the same launch, so no read-only (nc) path.
#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace pattn {

#define PATTN_DEV __device__ __forceinline__
constexpr int NT = 256;

// Additive mask, read with broadcast strides relative to (head-batch, query, key).
struct Mask {
  const float* p;
  long long sb, si, sj;
};

// scale_mode: 0 none, 1 x*scale, 2 x/scale (exactly the captured scalar op).
struct Attn {
  const float* q; long long q_b, q_r;   // [BH][T][HD], unit inner stride
  const float* k; long long k_b, k_r;
  const float* v; long long v_b, v_r;
  float* o;                             // forward output, contiguous [BH][T][HD]
  const float* oin; long long o_b, o_r; // backward input O (same tensor)
  float* lse;                           // [BH][T]
  const float* dout; long long d_b, d_r;
  float* delta;                         // [BH][T]
  float *dq, *dk, *dv;                  // contiguous [BH][T][HD]
  float* qn;                            // [BH][T] query row norms
  float* kmax;                          // [BH][NK] maximum key row norm per key block
  float* mmax;                          // [MB][NQ][NK] mask tile maximum (NaN if any NaN)
  float* skip;                          // [BH][NQ][NK] 1 = proven exact-zero tile (backward)
  Mask mask; int has_mask, mask_batched;
  int bh, t;
  // Release step 09 (defaults keep the original semantics): the mask batch of head batch b is b / m_heads
  // (per-example masks shared by every head); safe_rows gives a row with no allowed key (softmax denominator 0)
  // a zero output and lse = +inf, so the backward's exp(x - lse) is exactly 0 (PyTorch boolean-mask softmax).
  int m_heads = 1;
  int safe_rows = 0;
  float scale; int scale_mode;
  float bound_scale;                    // >= |d(score)/d(dot)|: |c| or 1/|c| (rounded up) or 1
};

PATTN_DEV int nblk(const Attn& a) { return (a.t + 63) / 64; }
PATTN_DEV int sw(int row, int group) { return ((group ^ ((row >> 2) & 15)) << 2); }

PATTN_DEV float scaled(const Attn& a, float dot) {
  return a.scale_mode == 1 ? dot * a.scale : (a.scale_mode == 2 ? dot / a.scale : dot);
}
PATTN_DEV float mask_at(const Attn& a, int b, int i, int j) {
  return a.mask.p[(long long)b * a.mask.sb + (long long)i * a.mask.si + (long long)j * a.mask.sj];
}
PATTN_DEV float mask_tile_max(const Attn& a, int b, int qb, int kb) {
  if (!a.has_mask) return 0.f;
  const int n = nblk(a);
  return a.mmax[((long long)(a.mask_batched ? b : 0) * n + qb) * n + kb];
}

PATTN_DEV float half_max(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
PATTN_DEV float half_add(float v) {
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- 64 x HD tiles staged through registers (NV float4 per thread, unrolled).
template <int HD>
PATTN_DEV void ld_regs(float4 (&r)[HD / 16], const float* src, long long row_stride, int valid) {
  constexpr int G = HD / 4;
#pragma unroll
  for (int u = 0; u < HD / 16; ++u) {
    const int idx = threadIdx.x + NT * u, i = idx / G, c = idx % G;
    r[u] = i < valid ? *reinterpret_cast<const float4*>(src + i * row_stride + 4 * c)
                     : make_float4(0.f, 0.f, 0.f, 0.f);
  }
}
// Transposed, swizzled: dst[k*64 + sw(k, i/4) + i%4] = X[i][k].
template <int HD>
PATTN_DEV void st_t(float* dst, const float4 (&r)[HD / 16]) {
  constexpr int G = HD / 4;
#pragma unroll
  for (int u = 0; u < HD / 16; ++u) {
    const int idx = threadIdx.x + NT * u, i = idx / G, k = 4 * (idx % G);
    dst[(k + 0) * 64 + sw(k + 0, i >> 2) + (i & 3)] = r[u].x;
    dst[(k + 1) * 64 + sw(k + 1, i >> 2) + (i & 3)] = r[u].y;
    dst[(k + 2) * 64 + sw(k + 2, i >> 2) + (i & 3)] = r[u].z;
    dst[(k + 3) * 64 + sw(k + 3, i >> 2) + (i & 3)] = r[u].w;
  }
}
// Row-major 64 x HD.
template <int HD>
PATTN_DEV void st_r(float* dst, const float4 (&r)[HD / 16]) {
  constexpr int G = HD / 4;
#pragma unroll
  for (int u = 0; u < HD / 16; ++u) {
    const int idx = threadIdx.x + NT * u;
    *reinterpret_cast<float4*>(dst + (idx / G) * HD + 4 * (idx % G)) = r[u];
  }
}
PATTN_DEV float4 get_t(const float* buf, int k, int group) {
  return *reinterpret_cast<const float4*>(buf + k * 64 + sw(k, group));
}

// acc[a][b] += sum_k A^T[k][4ga+a] * B^T[k][4gb+b], ascending k. The swizzle of
// rows 4*k4..4*k4+3 is one value, computed once per four rows.
template <int L>
PATTN_DEV void mm_tt(float (&acc)[4][4], const float* at, int ga, const float* bt, int gb) {
#pragma unroll 2
  for (int k4 = 0; k4 < L / 4; ++k4) {
    const float* pa = at + k4 * 256 + ((ga ^ (k4 & 15)) << 2);
    const float* pb = bt + k4 * 256 + ((gb ^ (k4 & 15)) << 2);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float4 x = *reinterpret_cast<const float4*>(pa + e * 64);
      const float4 y = *reinterpret_cast<const float4*>(pb + e * 64);
      const float xa[4] = {x.x, x.y, x.z, x.w}, yb[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int b = 0; b < 4; ++b) acc[a][b] = fmaf(xa[a], yb[b], acc[a][b]);
    }
  }
}
// acc[a][c] += sum_r X^T[r][4ga+a] * Y[r][C*gc+c] (X swizzled-transposed 64x64, Y row-major 64xHD).
template <int HD>
PATTN_DEV void mm_tr(float (&acc)[4][HD / 16], const float* xt, int ga, const float* y, int gc) {
  constexpr int C = HD / 16;
#pragma unroll 2
  for (int r4 = 0; r4 < 16; ++r4) {
    const float* px = xt + r4 * 256 + ((ga ^ r4) << 2);
    const float* py = y + r4 * 4 * HD + C * gc;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float4 x = *reinterpret_cast<const float4*>(px + e * 64);
      const float xa[4] = {x.x, x.y, x.z, x.w};
      float yc[C];
      if constexpr (C == 4) {
        const float4 w = *reinterpret_cast<const float4*>(py + e * HD);
        yc[0] = w.x; yc[1] = w.y; yc[2] = w.z; yc[3] = w.w;
      } else {
        const float2 w = *reinterpret_cast<const float2*>(py + e * HD);
        yc[0] = w.x; yc[1] = w.y;
      }
#pragma unroll
      for (int a = 0; a < 4; ++a)
#pragma unroll
        for (int c = 0; c < C; ++c) acc[a][c] = fmaf(xa[a], yc[c], acc[a][c]);
    }
  }
}
// Store p[a][b] (rows 4*gr+a, cols 4*gc+b) into a [col][row] swizzled buffer.
PATTN_DEV void put_t(float* buf, const float (&p)[4][4], int gr, int gc) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    const int col = 4 * gc + b;
    *reinterpret_cast<float4*>(buf + col * 64 + sw(col, gr)) = make_float4(p[0][b], p[1][b], p[2][b], p[3][b]);
  }
}
// Mask values of a 64x64 tile at (rows 4*gr+a, cols 4*gc+b), issued early (registers).
PATTN_DEV void mask_regs(const Attn& a, int b, int i0, int j0, int gr, int gc, bool rows_are_queries, float (&m)[4][4]) {
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int i = rows_are_queries ? i0 + 4 * gr + r : i0 + 4 * gc + c;
      const int j = rows_are_queries ? j0 + 4 * gc + c : j0 + 4 * gr + r;
      m[r][c] = (a.has_mask && i < a.t && j < a.t) ? a.mask.p[(long long)b * a.mask.sb + (long long)i * a.mask.si + (long long)j * a.mask.sj] : 0.f;
    }
}

// Exact-zero tile proof. For every valid query row r of the tile,
//   bound_scale*|q_r|*max_j|k_j|*(1+2^-10) + max(mask tile) < ref_r - 110,
// where ref_r is a lower bound of the row maximum actually used (running max or
// final LSE). Every skipped score x then satisfies x - ref_r < -110 and expf() is
// exactly +0 in FP32 here and in the captured reference (whose row max/LSE is
// >= ref_r): skipping adds exactly the reference's zeros. Norms and tile maxima
// come from the prepare phase. Any NaN, +inf or missing reference refuses.
// One thread per query row; returns the per-thread verdict (true for idle rows).
PATTN_DEV bool row_proof(const Attn& a, float qn, float kmax, float mmax, float ref) {
  const float bound = a.bound_scale * qn * kmax * 1.0009765625f + mmax;
  return isfinite(ref) && !isnan(bound) && bound < ref - 110.0f;
}

// ----------------------------------------------------------------- prepare
// Items [0, BH*NK): row norms of one 64-row block of K and Q; kmax for the block.
// Items [BH*NK, BH*NK + MB*NQ*NK): one mask tile maximum.
template <int HD>
__device__ __noinline__ void prepare_item(const Attn a, long long item, float* sm) {
  const int t = threadIdx.x, n = nblk(a);
  const long long norm_items = (long long)a.bh * n;
  if (item < norm_items) {
    const int b = (int)(item / n), blk = (int)(item % n), row = t >> 2, part = t & 3;
    const int i = blk * 64 + row;
    float sk = 0.f, sq = 0.f;
    if (i < a.t) {
      const float* kr = a.k + b * a.k_b + (long long)i * a.k_r + part * (HD / 4);
      const float* qr = a.q + b * a.q_b + (long long)i * a.q_r + part * (HD / 4);
#pragma unroll
      for (int e = 0; e < HD / 4; ++e) { const float x = kr[e]; sk = fmaf(x, x, sk); }
#pragma unroll
      for (int e = 0; e < HD / 4; ++e) { const float x = qr[e]; sq = fmaf(x, x, sq); }
    }
    sk += __shfl_xor_sync(0xffffffffu, sk, 1); sk += __shfl_xor_sync(0xffffffffu, sk, 2);
    sq += __shfl_xor_sync(0xffffffffu, sq, 1); sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    float kn = sqrtf(sk);
    if (part == 0 && i < a.t) a.qn[(long long)b * a.t + i] = sqrtf(sq);
    int nan_seen = isnan(kn);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kn = fmaxf(kn, __shfl_xor_sync(0xffffffffu, kn, o));
    if ((t & 31) == 0) sm[t >> 5] = kn;
    nan_seen = __syncthreads_or(nan_seen);
    if (t == 0) {
      float m = sm[0];
      for (int w = 1; w < NT / 32; ++w) m = fmaxf(m, sm[w]);
      a.kmax[(long long)b * n + blk] = nan_seen ? NAN : m;
    }
    __syncthreads();
    return;
  }
  const long long r = item - norm_items;
  const int mb = (int)(r / ((long long)n * n)), qb = (int)(r / n % n), kb = (int)(r % n);
  float mx = -INFINITY;
  int bad = 0;
#pragma unroll 4
  for (int e = 0; e < 16; ++e) {
    const int idx = t + NT * e, i = qb * 64 + idx / 64, j = kb * 64 + idx % 64;
    if (i < a.t && j < a.t) {
      const float m = a.mask.p[(long long)mb * a.mask.sb + (long long)i * a.mask.si + (long long)j * a.mask.sj];
      bad |= isnan(m);
      mx = m > mx ? m : mx;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, mx, o);
    mx = other > mx ? other : mx;
  }
  if ((t & 31) == 0) sm[t >> 5] = mx;
  bad = __syncthreads_or(bad);
  if (t == 0) {
    float m = sm[0];
    for (int w = 1; w < NT / 32; ++w) m = sm[w] > m ? sm[w] : m;
    a.mmax[((long long)mb * n + qb) * n + kb] = bad ? NAN : m;
  }
  __syncthreads();
}

// ----------------------------------------------------------------- forward
// Shared floats: Qt HD*64, Kt HD*64, V 64*HD, Pt 64*64, qn 64, mrow 64, kmx 64, mmx 64.
template <int HD>
__device__ __noinline__ void forward_item(const Attn a, int b, int qblk, float* sm) {
  constexpr int C = HD / 16;
  float* Qt = sm;
  float* Kt = Qt + HD * 64;
  float* Vs = Kt + HD * 64;
  float* Pt = Vs + 64 * HD;
  float* qn = Pt + 64 * 64;
  float* mrow = qn + 64;
  float* kmx = mrow + 64;
  float* mmx = kmx + 64;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4, nk = nblk(a);
  const int i0 = qblk * 64, qvalid = min(64, a.t - i0);
  {
    float4 r[HD / 16];
    ld_regs<HD>(r, a.q + b * a.q_b + (long long)i0 * a.q_r, a.q_r, qvalid);
    st_t<HD>(Qt, r);
  }
  if (t < 64) qn[t] = t < qvalid ? a.qn[(long long)b * a.t + i0 + t] : 0.f;
  const int mb = b / a.m_heads;  // mask batch of head batch b (release step 09; == b by default)
  if (t < nk) { kmx[t] = a.kmax[(long long)b * nk + t]; mmx[t] = mask_tile_max(a, mb, qblk, t); }
  float m[4], l[4], o[4][C];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < C; ++c) o[r][c] = 0.f;
  }
  float4 kr[HD / 16], vr[HD / 16];
  int have = 0;  // key block currently staged in kr/vr (-1: none)
  ld_regs<HD>(kr, a.k + b * a.k_b, a.k_r, min(64, a.t));
  ld_regs<HD>(vr, a.v + b * a.v_b, a.v_r, min(64, a.t));
  for (int kb = 0; kb < nk; ++kb) {
    const int j0 = kb * 64, kvalid = min(64, a.t - j0);
    if (tx == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) mrow[4 * ty + r] = m[r];
    }
    __syncthreads();  // mrow/qn/kmx visible; previous tile's Kt/Vs/Pt readers done
    const bool idle = t >= 64 || i0 + t >= a.t;
    if (__syncthreads_and(idle || row_proof(a, qn[t & 63], kmx[kb], mmx[kb], mrow[t & 63]))) continue;
    if (have != kb) {
      ld_regs<HD>(kr, a.k + b * a.k_b + (long long)j0 * a.k_r, a.k_r, kvalid);
      ld_regs<HD>(vr, a.v + b * a.v_b + (long long)j0 * a.v_r, a.v_r, kvalid);
    }
    st_t<HD>(Kt, kr);
    st_r<HD>(Vs, vr);
    float mk[4][4];
    mask_regs(a, mb, i0, j0, ty, tx, true, mk);
    __syncthreads();
    // Prefetch the next key block unless it is already proven negligible under
    // the current (smaller or equal) running maxima: then it will be skipped.
    have = -1;
    if (kb + 1 < nk) {
      const int nj = (kb + 1) * 64;
      if (!__syncthreads_and(idle || row_proof(a, qn[t & 63], kmx[kb + 1], mmx[kb + 1], mrow[t & 63]))) {
        ld_regs<HD>(kr, a.k + b * a.k_b + (long long)nj * a.k_r, a.k_r, min(64, a.t - nj));
        ld_regs<HD>(vr, a.v + b * a.v_b + (long long)nj * a.v_r, a.v_r, min(64, a.t - nj));
        have = kb + 1;
      }
    }
    float s[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) s[r][c] = 0.f;
    mm_tt<HD>(s, Qt, ty, Kt, tx);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float mx = -INFINITY;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int j = j0 + 4 * tx + c;
        float x = scaled(a, s[r][c]);
        if (a.has_mask) x = x + mk[r][c];
        if (j >= a.t) x = -INFINITY;
        s[r][c] = x;
        mx = fmaxf(mx, x);
      }
      mx = half_max(mx);
      const float mnew = fmaxf(m[r], mx);
      const float base = mnew == -INFINITY ? 0.f : mnew;
      const float alpha = expf(m[r] - base);
      float rs = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) { s[r][c] = expf(s[r][c] - base); rs += s[r][c]; }
      rs = half_add(rs);
      l[r] = l[r] * alpha + rs;
      m[r] = mnew;
#pragma unroll
      for (int c = 0; c < C; ++c) o[r][c] *= alpha;
    }
    put_t(Pt, s, ty, tx);
    __syncthreads();
    mm_tr<HD>(o, Pt, ty, Vs, tx);
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = i0 + 4 * ty + r;
    if (i >= a.t) continue;
    float* dst = a.o + ((long long)b * a.t + i) * HD + C * tx;
    const bool empty = a.safe_rows && l[r] == 0.f;  // every key forbidden: o[r] is exactly 0
    const float den = empty ? 1.f : l[r];
#pragma unroll
    for (int c = 0; c < C; ++c) dst[c] = o[r][c] / den;
    if (tx == 0) a.lse[(long long)b * a.t + i] = empty ? INFINITY : m[r] + logf(l[r]);
  }
  __syncthreads();
}

// ----------------------------------------------------------------- delta + backward skip table
// Item (b, qb): delta = rowsum(dO*O) in ascending e, and the exact-zero proof of
// every (qb, kb) tile against the final LSE.
template <int HD>
__device__ __noinline__ void delta_item(const Attn a, int b, int qb) {
  const int mb = b / a.m_heads;  // mask batch (release step 09; == b by default)
  const int t = threadIdx.x, n = nblk(a), i = qb * 64 + t;
  float lse = 0.f, qn = 0.f;
  if (t < 64 && i < a.t) {
    const float* x = a.dout + b * a.d_b + (long long)i * a.d_r;
    const float* y = a.oin + b * a.o_b + (long long)i * a.o_r;
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < HD; ++e) acc = fmaf(x[e], y[e], acc);
    a.delta[(long long)b * a.t + i] = acc;
    lse = a.lse[(long long)b * a.t + i];
    qn = a.qn[(long long)b * a.t + i];
  }
  const bool idle = t >= 64 || i >= a.t;
  for (int kb = 0; kb < n; ++kb) {
    const bool proof = idle || row_proof(a, qn, a.kmax[(long long)b * n + kb], mask_tile_max(a, mb, qb, kb), lse);
    const int all = __syncthreads_and(proof);
    if (t == 0) a.skip[((long long)b * n + qb) * n + kb] = all ? 1.f : 0.f;
  }
}

// ----------------------------------------------------------------- backward
// Key-block item: dK, dV for keys j0..j0+63 over all non-negligible query blocks.
// Shared floats: Kt, Vt, Qt, dOt (HD*64 each), Qr, dOr (64*HD each), Pb, Sb (64*64), lse, del, flags (64 each).
template <int HD>
__device__ __noinline__ void backward_kv_item(const Attn a, int b, int kblk, float* sm) {
  const int mb = b / a.m_heads;  // mask batch (release step 09; == b by default)
  constexpr int C = HD / 16;
  float* Kt = sm;
  float* Vt = Kt + HD * 64;
  float* Qt = Vt + HD * 64;
  float* dOt = Qt + HD * 64;
  float* Qr = dOt + HD * 64;
  float* dOr = Qr + 64 * HD;
  float* Pb = dOr + 64 * HD;
  float* Sb = Pb + 64 * 64;
  float* lse = Sb + 64 * 64;
  float* del = lse + 64;
  float* flag = del + 64;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4, n = nblk(a);
  const int j0 = kblk * 64, kvalid = min(64, a.t - j0);
  {
    float4 r[HD / 16];
    ld_regs<HD>(r, a.k + b * a.k_b + (long long)j0 * a.k_r, a.k_r, kvalid);
    st_t<HD>(Kt, r);
    ld_regs<HD>(r, a.v + b * a.v_b + (long long)j0 * a.v_r, a.v_r, kvalid);
    st_t<HD>(Vt, r);
  }
  if (t < n) flag[t] = a.skip[((long long)b * n + t) * n + kblk];
  __syncthreads();
  float dk[4][C], dv[4][C];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < C; ++c) { dk[r][c] = 0.f; dv[r][c] = 0.f; }
  int qb = 0;
  while (qb < n && flag[qb] != 0.f) ++qb;
  float4 qr[HD / 16], gr[HD / 16];
  float lr = 0.f, dr = 0.f;
  auto fetch = [&](int blk) {
    const int i0 = blk * 64, valid = min(64, a.t - i0);
    ld_regs<HD>(qr, a.q + b * a.q_b + (long long)i0 * a.q_r, a.q_r, valid);
    ld_regs<HD>(gr, a.dout + b * a.d_b + (long long)i0 * a.d_r, a.d_r, valid);
    if (t < 64) {
      lr = t < valid ? a.lse[(long long)b * a.t + i0 + t] : 0.f;
      dr = t < valid ? a.delta[(long long)b * a.t + i0 + t] : 0.f;
    }
  };
  if (qb < n) fetch(qb);
  while (qb < n) {
    const int i0 = qb * 64;
    st_t<HD>(Qt, qr); st_r<HD>(Qr, qr);
    st_t<HD>(dOt, gr); st_r<HD>(dOr, gr);
    if (t < 64) { lse[t] = lr; del[t] = dr; }
    float mk[4][4];
    mask_regs(a, mb, i0, j0, ty, tx, false, mk);
    __syncthreads();
    int next = qb + 1;
    while (next < n && flag[next] != 0.f) ++next;
    if (next < n) fetch(next);  // in flight during this tile's arithmetic
    // Rows of the register tile are keys (4ty+r), columns are queries (4tx+c).
    float p[4][4], ds[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) { p[r][c] = 0.f; ds[r][c] = 0.f; }
    mm_tt<HD>(p, Kt, ty, Qt, tx);    // dot(k_j, q_i): the forward's ascending-d fmaf chain
    mm_tt<HD>(ds, Vt, ty, dOt, tx);  // dP[i][j] = dot(dO_i, v_j)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = j0 + 4 * ty + r;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int il = 4 * tx + c, i = i0 + il;
        float x = scaled(a, p[r][c]);
        if (a.has_mask) x = x + mk[r][c];
        const float pr = (i < a.t && j < a.t) ? expf(x - lse[il]) : 0.f;
        const float g = pr * (ds[r][c] - del[il]);
        p[r][c] = pr;
        ds[r][c] = a.scale_mode == 1 ? g * a.scale : (a.scale_mode == 2 ? g / a.scale : g);
      }
    }
    put_t(Pb, p, ty, tx);   // Pb[i][j]
    put_t(Sb, ds, ty, tx);  // Sb[i][j]
    __syncthreads();
    mm_tr<HD>(dv, Pb, ty, dOr, tx);  // dV[j][e] += sum_i P[i][j] dO[i][e]
    mm_tr<HD>(dk, Sb, ty, Qr, tx);   // dK[j][d] += sum_i dS[i][j] Q[i][d]
    __syncthreads();
    qb = next;
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int j = j0 + 4 * ty + r;
    if (j >= a.t) continue;
    float* gk = a.dk + ((long long)b * a.t + j) * HD + C * tx;
    float* gv = a.dv + ((long long)b * a.t + j) * HD + C * tx;
#pragma unroll
    for (int c = 0; c < C; ++c) { gk[c] = dk[r][c]; gv[c] = dv[r][c]; }
  }
  __syncthreads();
}

// Query-block item: dQ for queries i0..i0+63 over all non-negligible key blocks.
// Shared floats: Qt, dOt, Kt, Vt (HD*64 each), Kr (64*HD), Sb (64*64), lse, del, flags (64 each).
template <int HD>
__device__ __noinline__ void backward_q_item(const Attn a, int b, int qblk, float* sm) {
  const int mb = b / a.m_heads;  // mask batch (release step 09; == b by default)
  constexpr int C = HD / 16;
  float* Qt = sm;
  float* dOt = Qt + HD * 64;
  float* Kt = dOt + HD * 64;
  float* Vt = Kt + HD * 64;
  float* Kr = Vt + HD * 64;
  float* Sb = Kr + 64 * HD;
  float* lse = Sb + 64 * 64;
  float* del = lse + 64;
  float* flag = del + 64;
  const int t = threadIdx.x, tx = t & 15, ty = t >> 4, n = nblk(a);
  const int i0 = qblk * 64, qvalid = min(64, a.t - i0);
  {
    float4 r[HD / 16];
    ld_regs<HD>(r, a.q + b * a.q_b + (long long)i0 * a.q_r, a.q_r, qvalid);
    st_t<HD>(Qt, r);
    ld_regs<HD>(r, a.dout + b * a.d_b + (long long)i0 * a.d_r, a.d_r, qvalid);
    st_t<HD>(dOt, r);
  }
  if (t < 64) {
    lse[t] = t < qvalid ? a.lse[(long long)b * a.t + i0 + t] : 0.f;
    del[t] = t < qvalid ? a.delta[(long long)b * a.t + i0 + t] : 0.f;
  }
  if (t < n) flag[t] = a.skip[((long long)b * n + qblk) * n + t];
  __syncthreads();
  float dq[4][C];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < C; ++c) dq[r][c] = 0.f;
  int kb = 0;
  while (kb < n && flag[kb] != 0.f) ++kb;
  float4 kr[HD / 16], vr[HD / 16];
  auto fetch = [&](int blk) {
    const int j0 = blk * 64, valid = min(64, a.t - j0);
    ld_regs<HD>(kr, a.k + b * a.k_b + (long long)j0 * a.k_r, a.k_r, valid);
    ld_regs<HD>(vr, a.v + b * a.v_b + (long long)j0 * a.v_r, a.v_r, valid);
  };
  if (kb < n) fetch(kb);
  while (kb < n) {
    const int j0 = kb * 64;
    st_t<HD>(Kt, kr); st_r<HD>(Kr, kr);
    st_t<HD>(Vt, vr);
    float mk[4][4];
    mask_regs(a, mb, i0, j0, ty, tx, true, mk);
    __syncthreads();
    int next = kb + 1;
    while (next < n && flag[next] != 0.f) ++next;
    if (next < n) fetch(next);
    float s[4][4], dp[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) { s[r][c] = 0.f; dp[r][c] = 0.f; }
    mm_tt<HD>(s, Qt, ty, Kt, tx);
    mm_tt<HD>(dp, dOt, ty, Vt, tx);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int il = 4 * ty + r, i = i0 + il;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int j = j0 + 4 * tx + c;
        float x = scaled(a, s[r][c]);
        if (a.has_mask) x = x + mk[r][c];
        const float pr = (i < a.t && j < a.t) ? expf(x - lse[il]) : 0.f;
        const float g = pr * (dp[r][c] - del[il]);
        s[r][c] = a.scale_mode == 1 ? g * a.scale : (a.scale_mode == 2 ? g / a.scale : g);
      }
    }
    put_t(Sb, s, ty, tx);  // Sb[j][i]
    __syncthreads();
    mm_tr<HD>(dq, Sb, ty, Kr, tx);  // dQ[i][d] += sum_j dS[i][j] K[j][d]
    __syncthreads();
    kb = next;
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = i0 + 4 * ty + r;
    if (i >= a.t) continue;
    float* g = a.dq + ((long long)b * a.t + i) * HD + C * tx;
#pragma unroll
    for (int c = 0; c < C; ++c) g[c] = dq[r][c];
  }
  __syncthreads();
}

template <int HD> constexpr int forward_floats() { return 3 * HD * 64 + 64 * 64 + 4 * 64; }
template <int HD> constexpr int backward_floats() { return 6 * HD * 64 + 2 * 64 * 64 + 3 * 64; }

}  // namespace pattn
