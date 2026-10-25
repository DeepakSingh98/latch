// P-OPS2-C fused FP32 attention, version 2 work items for the LATCH-OPS
// cooperative program (head width 64).
//
// Same semantics and exact-zero skipping proof as P-OPS-D v1 (attention_fp32.cuh,
// whose Attn descriptor, prepare/delta tables and row_proof are reused):
//   x = fl(fl(dot) op c) + mask, -inf for keys >= T; online softmax with row LSE;
//   P = exp(x - lse); dS = P * (dP - delta) [op c]; FP32 fmaf only, no fast math,
//   no tensor cores, no atomics; fixed-order and run-to-run deterministic.
// What changes is execution (P-OPS2-C):
//   * forward items are 128 query rows (4 rows x 8 keys per thread, 8-lane row
//     reductions) over 64-key tiles; S and P.V are 4x8 register blocks fed by
//     conflict-free XOR-swizzled row-major shared tiles (12 LDS.128 / 128 FMA);
//   * K/V/Q/dO tiles are copied global->shared with cp.async.cg (L2, coherent;
//     no register staging, no transposition), overlapped with the other product;
//   * mask tiles proven all-zero by the prepare phase are not loaded or added
//     (x + (+-0) == x up to the sign of a zero score, which no later operation
//     observes: max, subtraction from the row reference and exp);
//   * the backward key-block items (128 keys, 32-query steps) produce dK, dV and
//     store dS once to a [BH][Tp][Tp] scratch; a separate query-block pass forms
//     dQ = dS.K with one product per tile instead of recomputing S and dP
//     (7 instead of 9 tile products per attention);
//   * items are assigned to CTAs in a boustrophedon order of decreasing causal
//     cost (static LPT); correctness never depends on the order.
// Ascending-index fmaf chains: S/S^T dots ascend over the head dimension (forward
// and backward recomputation are bitwise identical), P.V ascends over keys,
// dV/dK ascend over queries, dQ ascends over keys.
#pragma once
#include "attention_fp32.cuh"

namespace pattn2 {

using pattn::Attn;
using pattn::NT;

#define PA2_DEV __device__ __forceinline__
// PA2_PIPELINE 1: inner products double-buffer their shared-memory operand fragments
// in registers (loads for chunk c+1 are issued before the FMAs of chunk c). Same
// arithmetic and order; 0 keeps the single-buffered loops (ablation).
#ifndef PA2_PIPELINE
#define PA2_PIPELINE 1
#endif
// Measured on H100 (P-OPS2-C gpu-05): double buffering helps the forward products
// but not the backward/dQ products, so the backward loops default to single buffering.
#ifndef PA2_PIPELINE_BWD
#define PA2_PIPELINE_BWD 0
#endif

// ---------------------------------------------------------------- async copies
PA2_DEV void cp16(float* dst, const float* src, bool valid) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
  // src-size 0 zero-fills the 16 bytes without reading the (still valid) source.
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src), "r"(valid ? 16 : 0));
}
PA2_DEV void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
PA2_DEV void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---------------------------------------------------------------- padded tiles
// Row-major tiles with padded row strides: 64-wide rows use 68 floats, 128-wide
// rows 132 floats (an odd number of 16-byte chunks). Threads whose rows are
// consecutive then hit distinct bank groups, and every address is a per-thread
// base plus a compile-time offset (no swizzle arithmetic in the inner loops).
constexpr int L64 = 68, L128 = 132;

// ROWS x 64 tile from global rows src + r*rs (unit inner stride, 16-byte aligned)
// into a dst tile of row stride LD; rows >= valid are zero-filled.
template <int ROWS, int LD>
PA2_DEV void load_rows(float* dst, const float* src, long long rs, int valid) {
  static_assert((ROWS * 16) % NT == 0, "tile copy must divide the CTA");
#pragma unroll
  for (int u = 0; u < ROWS * 16 / NT; ++u) {
    const int idx = threadIdx.x + NT * u, r = idx >> 4, c = idx & 15;
    const bool ok = r < valid;
    cp16(dst + r * LD + 4 * c, ok ? src + (long long)r * rs + 4 * c : src, ok);
  }
}
PA2_DEV float4 ld4(const float* p) { return *reinterpret_cast<const float4*>(p); }
PA2_DEV void st4(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }

// Score scaling with the captured scalar, rounded exactly as the captured graph
// (SMODE 0 none, 1 multiply, 2 divide) - a compile-time property of the group.
template <int SMODE>
PA2_DEV float fl_scale(float v, float c) {
  if constexpr (SMODE == 1) return v * c;
  else if constexpr (SMODE == 2) return v / c;
  else return v;
}
// Exact-zero mask tile flag (prepare phase) for 64x64 tile (qb, kb).
PA2_DEV bool mask_zero(const Attn& a, const float* mzero, int b, int qb, int kb) {
  const int n = pattn::nblk(a);
  if (!a.has_mask || qb >= n || kb >= n) return true;  // no valid entries outside the T x T range
  return mzero[((long long)(a.mask_batched ? b : 0) * n + qb) * n + kb] != 0.f;
}
PA2_DEV bool skip64(const Attn& a, int b, int qb, int kb) {
  const int n = pattn::nblk(a);
  return qb >= n || kb >= n || a.skip[((long long)b * n + qb) * n + kb] != 0.f;
}

// Boustrophedon (snake) position of this CTA in round r: items are listed in
// decreasing causal cost, rounds alternate direction (static LPT).
PA2_DEV long long snake(long long r) {
  return r * gridDim.x + ((r & 1) ? (long long)gridDim.x - 1 - blockIdx.x : (long long)blockIdx.x);
}

// ---------------------------------------------------------------- prepare (adds mask-zero flags)
// Items [0, BH*NK): row norms of a 64-row block of K and Q (same ascending fmaf
// chains as v1, loads vectorized) and kmax. Items [BH*NK, +MB*NQ*NK): mask tile
// max (NaN if any NaN) and the all-zero flag (1 iff every valid entry compares
// equal to 0); unit-stride aligned masks load float4, issued together.
PA2_DEV void norm_item(const Attn& a, long long item, float* sm) {
  const int t = threadIdx.x, n = pattn::nblk(a);
  const int b = (int)(item / n), blk = (int)(item % n), row = t >> 2, part = t & 3;
  const int i = blk * 64 + row;
  float sk = 0.f, sq = 0.f;
  if (i < a.t) {
    const float4* kr = reinterpret_cast<const float4*>(a.k + b * a.k_b + (long long)i * a.k_r + part * 16);
    const float4* qr = reinterpret_cast<const float4*>(a.q + b * a.q_b + (long long)i * a.q_r + part * 16);
    float4 kv[4], qv[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) { kv[e] = kr[e]; qv[e] = qr[e]; }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      sk = fmaf(kv[e].x, kv[e].x, sk); sk = fmaf(kv[e].y, kv[e].y, sk); sk = fmaf(kv[e].z, kv[e].z, sk); sk = fmaf(kv[e].w, kv[e].w, sk);
      sq = fmaf(qv[e].x, qv[e].x, sq); sq = fmaf(qv[e].y, qv[e].y, sq); sq = fmaf(qv[e].z, qv[e].z, sq); sq = fmaf(qv[e].w, qv[e].w, sq);
    }
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
}

__device__ __noinline__ void prepare_item(const Attn a, float* mzero, long long item, float* sm) {
  const int t = threadIdx.x, n = pattn::nblk(a);
  const long long norm_items = (long long)a.bh * n;
  if (item < norm_items) {
    norm_item(a, item, sm);
    return;
  }
  const long long r = item - norm_items;
  const int mb = (int)(r / ((long long)n * n)), qb = (int)(r / n % n), kb = (int)(r % n);
  float mx = -INFINITY;
  int bad = 0, nonzero = 0;
  const float* base = a.mask.p + (long long)mb * a.mask.sb;
  const bool vec = a.mask.sj == 1 && ((a.mask.si | a.mask.sb) & 3) == 0 && (a.t & 3) == 0 &&
                   ((reinterpret_cast<unsigned long long>(a.mask.p) & 15) == 0);
  if (vec) {
    float4 v[4];
    bool ok[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int idx = t + NT * e, i = qb * 64 + (idx >> 4), j = kb * 64 + 4 * (idx & 15);
      ok[e] = i < a.t && j < a.t;  // T % 4 == 0: a chunk is wholly valid or wholly outside
      v[e] = ok[e] ? *reinterpret_cast<const float4*>(base + (long long)i * a.mask.si + j) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      if (!ok[e]) continue;
      const float m4[4] = {v[e].x, v[e].y, v[e].z, v[e].w};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        bad |= isnan(m4[c]);
        nonzero |= !(m4[c] == 0.f);
        mx = m4[c] > mx ? m4[c] : mx;
      }
    }
  } else {
#pragma unroll 4
    for (int e = 0; e < 16; ++e) {
      const int idx = t + NT * e, i = qb * 64 + idx / 64, j = kb * 64 + idx % 64;
      if (i < a.t && j < a.t) {
        const float m = base[(long long)i * a.mask.si + (long long)j * a.mask.sj];
        bad |= isnan(m);
        nonzero |= !(m == 0.f);
        mx = m > mx ? m : mx;
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, mx, o);
    mx = other > mx ? other : mx;
  }
  if ((t & 31) == 0) sm[t >> 5] = mx;
  bad = __syncthreads_or(bad);
  nonzero = __syncthreads_or(nonzero);
  if (t == 0) {
    float m = sm[0];
    for (int w = 1; w < NT / 32; ++w) m = sm[w] > m ? sm[w] : m;
    const long long at = ((long long)mb * n + qb) * n + kb;
    a.mmax[at] = bad ? NAN : m;
    mzero[at] = (bad || nonzero) ? 0.f : 1.f;
  }
  __syncthreads();
}

// ---------------------------------------------------------------- warp-level prepare
// One warp per work item, no block barriers: items [0, BH*NK) are 64-row blocks
// (lane owns rows lane and lane+32: |k_row|, |q_row| as ascending fmaf chains;
// kmax = warp max, NaN if any norm is NaN); items [BH*NK, +MB*NQ*NK) are 64x64 mask
// tiles (max, NaN flag, all-zero flag). Norms feed only the exact-zero proof, whose
// 2^-10 slack covers any summation order.
__device__ __noinline__ void prepare_warp(const Attn a, float* mzero, long long item, bool with_mask_items) {
  const int lane = threadIdx.x & 31, n = pattn::nblk(a);
  const long long norm_items = (long long)a.bh * n;
  if (item < norm_items) {
    const int b = (int)(item / n), blk = (int)(item % n);
    float kn_max = 0.f;
    int nan_seen = 0;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int i = blk * 64 + lane + 32 * h;
      if (i >= a.t) continue;
      const float4* kr = reinterpret_cast<const float4*>(a.k + b * a.k_b + (long long)i * a.k_r);
      const float4* qr = reinterpret_cast<const float4*>(a.q + b * a.q_b + (long long)i * a.q_r);
      float sk = 0.f, sq = 0.f;
#pragma unroll 4
      for (int c = 0; c < 16; ++c) {
        const float4 kv = kr[c], qv = qr[c];
        sk = fmaf(kv.x, kv.x, sk); sk = fmaf(kv.y, kv.y, sk); sk = fmaf(kv.z, kv.z, sk); sk = fmaf(kv.w, kv.w, sk);
        sq = fmaf(qv.x, qv.x, sq); sq = fmaf(qv.y, qv.y, sq); sq = fmaf(qv.z, qv.z, sq); sq = fmaf(qv.w, qv.w, sq);
      }
      const float kn = sqrtf(sk);
      a.qn[(long long)b * a.t + i] = sqrtf(sq);
      nan_seen |= isnan(kn);
      kn_max = fmaxf(kn_max, kn);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) kn_max = fmaxf(kn_max, __shfl_xor_sync(0xffffffffu, kn_max, o));
    nan_seen = __any_sync(0xffffffffu, nan_seen);
    if (lane == 0) a.kmax[(long long)b * n + blk] = nan_seen ? NAN : kn_max;
    return;
  }
  if (!with_mask_items) return;
  const long long r = item - norm_items;
  const int mb = (int)(r / ((long long)n * n)), qb = (int)(r / n % n), kb = (int)(r % n);
  const float* base = a.mask.p + (long long)mb * a.mask.sb;
  float mx = -INFINITY;
  int bad = 0, nonzero = 0;
  const bool vec = a.mask.sj == 1 && ((a.mask.si | a.mask.sb) & 3) == 0 && (a.t & 3) == 0 &&
                   ((reinterpret_cast<unsigned long long>(a.mask.p) & 15) == 0);
  if (vec) {
    // Lane covers chunks (lane & 15) of rows 2*row4 + (lane >> 4), 32 rows per pass.
#pragma unroll 4
    for (int pass = 0; pass < 32; ++pass) {
      const int i = qb * 64 + 2 * pass + (lane >> 4), j = kb * 64 + 4 * (lane & 15);
      if (i < a.t && j < a.t) {
        const float4 v = *reinterpret_cast<const float4*>(base + (long long)i * a.mask.si + j);
        const float m4[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          bad |= isnan(m4[c]);
          nonzero |= !(m4[c] == 0.f);
          mx = m4[c] > mx ? m4[c] : mx;
        }
      }
    }
  } else {
    for (int e = lane; e < 64 * 64; e += 32) {
      const int i = qb * 64 + e / 64, j = kb * 64 + e % 64;
      if (i < a.t && j < a.t) {
        const float m = base[(long long)i * a.mask.si + (long long)j * a.mask.sj];
        bad |= isnan(m);
        nonzero |= !(m == 0.f);
        mx = m > mx ? m : mx;
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, mx, o);
    mx = other > mx ? other : mx;
  }
  bad = __any_sync(0xffffffffu, bad);
  nonzero = __any_sync(0xffffffffu, nonzero);
  if (lane == 0) {
    const long long at = ((long long)mb * n + qb) * n + kb;
    a.mmax[at] = bad ? NAN : mx;
    mzero[at] = (bad || nonzero) ? 0.f : 1.f;
  }
}

// ---------------------------------------------------------------- delta + backward skip table
// Item (b, 64-query block): delta = rowsum(dO*O) by the same ascending-e fmaf chain
// as v1 (vectorized loads), and the exact-zero proof of every (qb, kb) tile against
// the final LSE, voted per warp (rows 0..63 live in warps 0 and 1) with one barrier.
__device__ __noinline__ void delta_item(const Attn a, int b, int qb, float* sm) {
  const int t = threadIdx.x, n = pattn::nblk(a), i = qb * 64 + t, lane = t & 31, w = t >> 5;
  if (w < 2) {
    float lse = 0.f, qn = 0.f;
    const bool valid = i < a.t;
    if (valid) {
      const float4* x = reinterpret_cast<const float4*>(a.dout + b * a.d_b + (long long)i * a.d_r);
      const float4* y = reinterpret_cast<const float4*>(a.oin + b * a.o_b + (long long)i * a.o_r);
      float acc = 0.f;
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const float4 u = x[e], v = y[e];
        acc = fmaf(u.x, v.x, acc); acc = fmaf(u.y, v.y, acc); acc = fmaf(u.z, v.z, acc); acc = fmaf(u.w, v.w, acc);
      }
      a.delta[(long long)b * a.t + i] = acc;
      lse = a.lse[(long long)b * a.t + i];
      qn = a.qn[(long long)b * a.t + i];
    }
    for (int kb = 0; kb < n; ++kb) {
      const bool proof = !valid || pattn::row_proof(a, qn, a.kmax[(long long)b * n + kb], pattn::mask_tile_max(a, b, qb, kb), lse);
      const bool all = __all_sync(0xffffffffu, proof);
      if (lane == 0) sm[w * 64 + kb] = all ? 1.f : 0.f;
    }
  }
  __syncthreads();
  if (t < n) a.skip[((long long)b * n + qb) * n + t] = (sm[t] != 0.f && sm[64 + t] != 0.f) ? 1.f : 0.f;
  __syncthreads();
}

// ---------------------------------------------------------------- forward
// Item (b, 128-query block) over 64-key tiles.
//   S mapping: qg = t>>3, kg = t&7; rows qg+32a (a<4), keys kg+8u (u<8);
//              8-lane row reductions; running max/sum per row.
//   O mapping: qo = t>>3, ko = t&7; rows 4qo+a, columns 4ko..+3 and 32+4ko..+3;
//              per-tile rescale factors and the final sums come through shared memory.
// Shared floats: Q 128x68, K 64x68, V 64x64, P^T 64x132, qn/mrow/alpha/lfin 128 each,
// kmx 64, mmx/mzf 2x64 each.
constexpr int FWD_FLOATS = 128 * L64 + 64 * L64 + 64 * 64 + 64 * L128 + 4 * 128 + 64 + 4 * 64;

// dS / score scratch row stride: T rounded up to 128 (keys beyond T hold -inf scores / zero dS).
PA2_DEV long long ds_stride(const Attn& a) { return (long long)((a.t + 127) / 128) * 128; }

// STORE: also write every computed score x = fl(fl(dot) op c) + mask (-inf for keys
// >= T) to xs[b][i][j] (row stride ds_stride) for the stored-score backward.
template <int SMODE, bool STORE>
__device__ __noinline__ void forward_item(const Attn a, const float* mzero, float* xs, int b, int qblk, float* sm) {
  float* Qs = sm;
  float* Ks = Qs + 128 * L64;
  float* Vs = Ks + 64 * L64;
  float* Pt = Vs + 64 * 64;
  float* qn = Pt + 64 * L128;
  float* mrow = qn + 128;
  float* alph = mrow + 128;
  float* lfin = alph + 128;
  float* kmx = lfin + 128;
  float* mmx = kmx + 64;   // [2][64]: per 64-row half
  float* mzf = mmx + 128;  // [2][64]
  const int t = threadIdx.x, kg = t & 7, qg = t >> 3, nk = pattn::nblk(a);
  const int i0 = qblk * 128, qvalid = min(128, a.t - i0);
  const long long xstride = ds_stride(a);
  const float* kbase = a.k + b * a.k_b;
  const float* vbase = a.v + b * a.v_b;
  load_rows<128, L64>(Qs, a.q + b * a.q_b + (long long)i0 * a.q_r, a.q_r, qvalid);
  cp_commit();
  if (t < 128) qn[t] = t < qvalid ? a.qn[(long long)b * a.t + i0 + t] : 0.f;
  if (t < nk) {
    kmx[t] = a.kmax[(long long)b * nk + t];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int qb = 2 * qblk + h;
      mmx[h * 64 + t] = qb < nk ? pattn::mask_tile_max(a, b, qb, t) : -INFINITY;
      mzf[h * 64 + t] = (qb >= nk || mask_zero(a, mzero, b, qb, t)) ? 1.f : 0.f;
    }
  }
  float m[4], l[4], o[4][8];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    m[r] = -INFINITY; l[r] = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) o[r][c] = 0.f;
  }
  load_rows<64, L64>(Ks, kbase, a.k_r, min(64, a.t));
  cp_commit();
  load_rows<64, 64>(Vs, vbase, a.v_r, min(64, a.t));
  cp_commit();
  int kload = 0, vload = 0;
  const bool idle = t >= 128 || i0 + t >= a.t;
  const float* qrow = Qs + qg * L64;  // S-mapping rows qg + 32a
  const float* krow = Ks + kg * L64;  // S-mapping keys kg + 8u
  for (int kb = 0; kb < nk; ++kb) {
    const int j0 = kb * 64;
    if (kg == 0) {
#pragma unroll
      for (int r = 0; r < 4; ++r) mrow[qg + 32 * r] = m[r];
    }
    __syncthreads();  // mrow/qn/kmx visible; previous tile's readers of Ks/Vs/Pt/alph done
    if (__syncthreads_and(idle || pattn::row_proof(a, qn[t & 127], kmx[kb], mmx[((t & 127) >> 6) * 64 + kb], mrow[t & 127])))
      continue;
    if (kload != kb || vload != kb) {
      cp_wait<0>();
      __syncthreads();
      load_rows<64, L64>(Ks, kbase + (long long)j0 * a.k_r, a.k_r, min(64, a.t - j0));
      cp_commit();
      load_rows<64, 64>(Vs, vbase + (long long)j0 * a.v_r, a.v_r, min(64, a.t - j0));
      cp_commit();
      kload = vload = kb;
    }
    // Rows qg+32a lie in 64-row half a>>1; mask values are loaded only for tiles
    // not proven all-zero, before the product.
    const bool use0 = mzf[kb] == 0.f, use1 = mzf[64 + kb] == 0.f;
    cp_wait<1>();  // Q (first tile) and K(kb) complete; V(kb) may be in flight
    __syncthreads();
    float s[4][8];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int u = 0; u < 8; ++u) s[r][u] = 0.f;
    {
      float4 qv[2][4], kv[2][8];
      auto lds = [&](float4 (&q)[4], float4 (&k)[8], int c) {
#pragma unroll
        for (int r = 0; r < 4; ++r) q[r] = ld4(qrow + r * 32 * L64 + 4 * c);
#pragma unroll
        for (int u = 0; u < 8; ++u) k[u] = ld4(krow + u * 8 * L64 + 4 * c);
      };
      auto fma = [&](const float4 (&q)[4], const float4 (&k)[8]) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float qa[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
#pragma unroll
          for (int e = 0; e < 4; ++e)
#pragma unroll
            for (int u = 0; u < 8; ++u) {
              const float ka[4] = {k[u].x, k[u].y, k[u].z, k[u].w};
              s[r][u] = fmaf(qa[e], ka[e], s[r][u]);
            }
        }
      };
#if PA2_PIPELINE
      lds(qv[0], kv[0], 0);
#pragma unroll 1
      for (int c = 0; c < 16; c += 2) {
        lds(qv[1], kv[1], c + 1);
        fma(qv[0], kv[0]);
        if (c + 2 < 16) lds(qv[0], kv[0], c + 2);
        fma(qv[1], kv[1]);
      }
#else
#pragma unroll 4
      for (int c = 0; c < 16; ++c) { lds(qv[0], kv[0], c); fma(qv[0], kv[0]); }
#endif
    }
    __syncthreads();  // every thread is done reading Ks
    if (kb + 1 < nk) {
      const int nj = j0 + 64;
      load_rows<64, L64>(Ks, kbase + (long long)nj * a.k_r, a.k_r, min(64, a.t - nj));
      kload = kb + 1;
    }
    cp_commit();
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      float mx = -INFINITY;
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int j = j0 + kg + 8 * u;
        float x = fl_scale<SMODE>(s[r][u], a.scale);
        if (r < 2 ? use0 : use1) {  // only tiles not proven all-zero; loaded on demand
          const int i = i0 + qg + 32 * r;
          x = x + ((i < a.t && j < a.t) ? pattn::mask_at(a, b, i, j) : 0.f);
        }
        if (j >= a.t) x = -INFINITY;
        s[r][u] = x;
        mx = fmaxf(mx, x);
        if constexpr (STORE) {
          const int i = i0 + qg + 32 * r;
          if (i < a.t) xs[((long long)b * xstride + i) * xstride + j] = x;
        }
      }
#pragma unroll
      for (int sh = 1; sh < 8; sh <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, sh));
      const float mnew = fmaxf(m[r], mx);
      const float base = mnew == -INFINITY ? 0.f : mnew;
      const float alpha = expf(m[r] - base);
      float rs = 0.f;
#pragma unroll
      for (int u = 0; u < 8; ++u) { s[r][u] = expf(s[r][u] - base); rs += s[r][u]; }
#pragma unroll
      for (int sh = 1; sh < 8; sh <<= 1) rs += __shfl_xor_sync(0xffffffffu, rs, sh);
      l[r] = l[r] * alpha + rs;
      m[r] = mnew;
      if (kg == 0) alph[qg + 32 * r] = alpha;
#pragma unroll
      for (int u = 0; u < 8; ++u) Pt[(kg + 8 * u) * L128 + qg + 32 * r] = s[r][u];
    }
    cp_wait<1>();  // V(kb) complete; K(kb+1) may be in flight
    __syncthreads();
    {
      const float4 al = ld4(alph + 4 * qg);
      const float af[4] = {al.x, al.y, al.z, al.w};
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) o[r][c] *= af[r];
    }
    const float* prow = Pt + 4 * qg;  // O mapping: qo = qg, ko = kg
    const float* vrow = Vs + 4 * kg;
    {
      auto fma = [&](const float4& p, const float4& v0, const float4& v1) {
        const float pa[4] = {p.x, p.y, p.z, p.w};
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          o[r][0] = fmaf(pa[r], v0.x, o[r][0]); o[r][1] = fmaf(pa[r], v0.y, o[r][1]);
          o[r][2] = fmaf(pa[r], v0.z, o[r][2]); o[r][3] = fmaf(pa[r], v0.w, o[r][3]);
          o[r][4] = fmaf(pa[r], v1.x, o[r][4]); o[r][5] = fmaf(pa[r], v1.y, o[r][5]);
          o[r][6] = fmaf(pa[r], v1.z, o[r][6]); o[r][7] = fmaf(pa[r], v1.w, o[r][7]);
        }
      };
#if PA2_PIPELINE
      // Four keys per stage, two stages in flight.
      float4 pa[2][4], va[2][4], vb[2][4];
      auto lds = [&](float4 (&p)[4], float4 (&v0)[4], float4 (&v1)[4], int k0) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          p[k] = ld4(prow + (k0 + k) * L128);
          v0[k] = ld4(vrow + (k0 + k) * 64);
          v1[k] = ld4(vrow + (k0 + k) * 64 + 32);
        }
      };
      lds(pa[0], va[0], vb[0], 0);
#pragma unroll 1
      for (int k0 = 0; k0 < 64; k0 += 8) {
        lds(pa[1], va[1], vb[1], k0 + 4);
#pragma unroll
        for (int k = 0; k < 4; ++k) fma(pa[0][k], va[0][k], vb[0][k]);
        if (k0 + 8 < 64) lds(pa[0], va[0], vb[0], k0 + 8);
#pragma unroll
        for (int k = 0; k < 4; ++k) fma(pa[1][k], va[1][k], vb[1][k]);
      }
#else
#pragma unroll 8
      for (int k = 0; k < 64; ++k)
        fma(ld4(prow + k * L128), ld4(vrow + k * 64), ld4(vrow + k * 64 + 32));
#endif
    }
    __syncthreads();  // every thread is done reading Vs and Pt
    if (kb + 1 < nk) {
      const int nj = j0 + 64;
      load_rows<64, 64>(Vs, vbase + (long long)nj * a.v_r, a.v_r, min(64, a.t - nj));
      vload = kb + 1;
    }
    cp_commit();
  }
  cp_wait<0>();
  if (kg == 0) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = i0 + qg + 32 * r;
      lfin[qg + 32 * r] = l[r];
      if (i < a.t) a.lse[(long long)b * a.t + i] = m[r] + logf(l[r]);
    }
  }
  __syncthreads();
  {
    const float4 lv = ld4(lfin + 4 * qg);
    const float lf[4] = {lv.x, lv.y, lv.z, lv.w};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int i = i0 + 4 * qg + r;
      if (i >= a.t) continue;
      float* dst = a.o + ((long long)b * a.t + i) * 64;
      st4(dst + 4 * kg, make_float4(o[r][0] / lf[r], o[r][1] / lf[r], o[r][2] / lf[r], o[r][3] / lf[r]));
      st4(dst + 32 + 4 * kg, make_float4(o[r][4] / lf[r], o[r][5] / lf[r], o[r][6] / lf[r], o[r][7] / lf[r]));
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------- backward key-block items
// Item (b, 128-key block), 32-query steps.
//   Mapping A (S^T, dP^T, P, dS): kA = t>>3 keys kA+32r, qA = t&7 queries qA+8c.
//   Mapping B (dV, dK): warp w, lane l: kB = 4(w>>1)+(l>>3) keys 8kB..+7,
//                       eB = 8(w&1)+(l&7) columns 4eB..+3.
// The next step's Q/dO rows are prefetched into registers during the current step.
// Shared floats: K, V 128x68; Q, dO 32x68; lse, delta 32; P, dS 32x132 (row = query);
// 128 step flags.
constexpr int KV_FLOATS = 2 * 128 * L64 + 2 * 32 * L64 + 64 + 2 * 32 * L128 + 128;


template <int SMODE>
__device__ __noinline__ void backward_kv_item(const Attn a, float* dsg, const float* mzero, int b, int kblk, float* sm) {
  float* Ks = sm;
  float* Vs = Ks + 128 * L64;
  float* Qs = Vs + 128 * L64;
  float* Ds = Qs + 32 * L64;
  float* lse = Ds + 32 * L64;
  float* del = lse + 32;
  float* Pb = del + 32;
  float* Sb = Pb + 32 * L128;
  float* flag = Sb + 32 * L128;
  const int t = threadIdx.x, steps = (a.t + 31) / 32;
  const int kA = t >> 3, qA = t & 7;
  const int w = t >> 5, ln = t & 31, kB = 4 * (w >> 1) + (ln >> 3), eB = 8 * (w & 1) + (ln & 7);
  const int j0 = kblk * 128, kvalid = min(128, a.t - j0);
  const long long dstride = ds_stride(a);
  load_rows<128, L64>(Ks, a.k + b * a.k_b + (long long)j0 * a.k_r, a.k_r, kvalid);
  load_rows<128, L64>(Vs, a.v + b * a.v_b + (long long)j0 * a.v_r, a.v_r, kvalid);
  cp_commit();
  // Step s (queries 32s..32s+31) is skipped iff both covering 64x64 tiles are proven zero.
  if (t < 128) flag[t] = (t < steps && !(skip64(a, b, t >> 1, 2 * kblk) && skip64(a, b, t >> 1, 2 * kblk + 1))) ? 1.f : 0.f;
  __syncthreads();
  // Register prefetch of one step: Q, dO chunks (r = idx>>4, c = idx&15, idx = t + 256u) and lse/delta.
  float4 pq[2], pd[2];
  float pl = 0.f, pde = 0.f;
  auto fetch = [&](int s) {
    const int i0 = s * 32, valid = min(32, a.t - i0);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + NT * u, r = idx >> 4, c = idx & 15;
      const bool ok = r < valid;
      pq[u] = ok ? ld4(a.q + b * a.q_b + (long long)(i0 + r) * a.q_r + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
      pd[u] = ok ? ld4(a.dout + b * a.d_b + (long long)(i0 + r) * a.d_r + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (t < 32) {
      pl = t < valid ? a.lse[(long long)b * a.t + i0 + t] : 0.f;
      pde = t < valid ? a.delta[(long long)b * a.t + i0 + t] : 0.f;
    }
  };
  float dk[8][4], dv[8][4];
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int c = 0; c < 4; ++c) { dk[r][c] = 0.f; dv[r][c] = 0.f; }
  int s = 0;
  while (s < steps && flag[s] == 0.f) ++s;
  if (s < steps) fetch(s);
  const int halfA = kA >> 4;  // 64-key half of mapping-A keys (kA+32r: half = r>>1) handled per r below
  (void)halfA;
  const float* krow = Ks + kA * L64;  // keys kA + 32r
  const float* vrow = Vs + kA * L64;
  const float* qrow = Qs + qA * L64;  // queries qA + 8c
  const float* grow = Ds + qA * L64;
  while (s < steps) {
    const int i0 = s * 32;
    __syncthreads();  // previous step's readers of Qs/Ds/Pb/Sb done
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + NT * u, r = idx >> 4, c = idx & 15;
      st4(Qs + r * L64 + 4 * c, pq[u]);
      st4(Ds + r * L64 + 4 * c, pd[u]);
    }
    if (t < 32) { lse[t] = pl; del[t] = pde; }
    int next = s + 1;
    while (next < steps && flag[next] == 0.f) ++next;
    if (next < steps) fetch(next);  // in flight during this step's arithmetic
    const bool use0 = !mask_zero(a, mzero, b, s >> 1, 2 * kblk), use1 = !mask_zero(a, mzero, b, s >> 1, 2 * kblk + 1);
    float mk[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const bool use = r < 2 ? use0 : use1;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int i = i0 + qA + 8 * c, j = j0 + kA + 32 * r;
        mk[r][c] = (use && i < a.t && j < a.t) ? pattn::mask_at(a, b, i, j) : 0.f;
      }
    }
    cp_wait<0>();  // K/V (first step)
    __syncthreads();
    float x[4][4], g[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) { x[r][c] = 0.f; g[r][c] = 0.f; }
#pragma unroll 4
    for (int c4 = 0; c4 < 16; ++c4) {
      float4 kv[4], qv[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) kv[r] = ld4(krow + r * 32 * L64 + 4 * c4);
#pragma unroll
      for (int c = 0; c < 4; ++c) qv[c] = ld4(qrow + c * 8 * L64 + 4 * c4);
#pragma unroll
      for (int e = 0; e < 4; ++e)
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float ka[4] = {kv[r].x, kv[r].y, kv[r].z, kv[r].w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const float qa[4] = {qv[c].x, qv[c].y, qv[c].z, qv[c].w};
            x[r][c] = fmaf(qa[e], ka[e], x[r][c]);  // dot(q_i, k_j): the forward's ascending-d chain
          }
        }
    }
    {
      float4 vv[2][4], gv[2][4];
      auto lds = [&](float4 (&v)[4], float4 (&gg)[4], int c4) {
#pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = ld4(vrow + r * 32 * L64 + 4 * c4);
#pragma unroll
        for (int c = 0; c < 4; ++c) gg[c] = ld4(grow + c * 8 * L64 + 4 * c4);
      };
      auto fma = [&](const float4 (&v)[4], const float4 (&gg)[4]) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float va[4] = {v[r].x, v[r].y, v[r].z, v[r].w};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const float ga[4] = {gg[c].x, gg[c].y, gg[c].z, gg[c].w};
              g[r][c] = fmaf(ga[e], va[e], g[r][c]);  // dP[i][j] = dot(dO_i, v_j)
            }
          }
      };
#if PA2_PIPELINE && PA2_PIPELINE_BWD
      lds(vv[0], gv[0], 0);
#pragma unroll 1
      for (int c4 = 0; c4 < 16; c4 += 2) {
        lds(vv[1], gv[1], c4 + 1);
        fma(vv[0], gv[0]);
        if (c4 + 2 < 16) lds(vv[0], gv[0], c4 + 2);
        fma(vv[1], gv[1]);
      }
#else
#pragma unroll 4
      for (int c4 = 0; c4 < 16; ++c4) { lds(vv[0], gv[0], c4); fma(vv[0], gv[0]); }
#endif
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = j0 + kA + 32 * r;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int il = qA + 8 * c, i = i0 + il;
        float v = fl_scale<SMODE>(x[r][c], a.scale);
        if (r < 2 ? use0 : use1) v = v + mk[r][c];
        const float pr = (i < a.t && j < a.t) ? expf(v - lse[il]) : 0.f;
        Pb[il * L128 + kA + 32 * r] = pr;
        Sb[il * L128 + kA + 32 * r] = fl_scale<SMODE>(pr * (g[r][c] - del[il]), a.scale);
      }
    }
    __syncthreads();
    {
      const float* pr = Pb + 8 * kB;
      const float* sr = Sb + 8 * kB;
      const float* gr = Ds + 4 * eB;
      const float* qr = Qs + 4 * eB;
      float4 op[2][6];  // p0, p1, d0, d1, dO, Q of one query row
      auto lds = [&](float4 (&x)[6], int q) {
        x[0] = ld4(pr + q * L128); x[1] = ld4(pr + q * L128 + 4);
        x[2] = ld4(sr + q * L128); x[3] = ld4(sr + q * L128 + 4);
        x[4] = ld4(gr + q * L64); x[5] = ld4(qr + q * L64);
      };
      auto fma = [&](const float4 (&x)[6]) {
        const float pa[8] = {x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w};
        const float sa[8] = {x[2].x, x[2].y, x[2].z, x[2].w, x[3].x, x[3].y, x[3].z, x[3].w};
        const float ga[4] = {x[4].x, x[4].y, x[4].z, x[4].w}, qa[4] = {x[5].x, x[5].y, x[5].z, x[5].w};
#pragma unroll
        for (int r = 0; r < 8; ++r)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            dv[r][c] = fmaf(pa[r], ga[c], dv[r][c]);
            dk[r][c] = fmaf(sa[r], qa[c], dk[r][c]);
          }
      };
#if PA2_PIPELINE && PA2_PIPELINE_BWD
      lds(op[0], 0);
#pragma unroll 2
      for (int q = 0; q < 32; q += 2) {
        lds(op[1], q + 1);
        fma(op[0]);
        if (q + 2 < 32) lds(op[0], q + 2);
        fma(op[1]);
      }
#else
#pragma unroll 4
      for (int q = 0; q < 32; ++q) { lds(op[0], q); fma(op[0]); }
#endif
    }
    // dS rows of this step to the scratch (queries < T; all 128 keys, zeros beyond T).
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + NT * u, r = idx >> 5, c = idx & 31, i = i0 + r;
      if (i < a.t) st4(dsg + ((long long)b * dstride + i) * dstride + j0 + 4 * c, ld4(Sb + r * L128 + 4 * c));
    }
    s = next;
  }
  cp_wait<0>();
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int j = j0 + 8 * kB + r;
    if (j >= a.t) continue;
    st4(a.dk + ((long long)b * a.t + j) * 64 + 4 * eB, make_float4(dk[r][0], dk[r][1], dk[r][2], dk[r][3]));
    st4(a.dv + ((long long)b * a.t + j) * 64 + 4 * eB, make_float4(dv[r][0], dv[r][1], dv[r][2], dv[r][3]));
  }
  __syncthreads();
}

// ---------------------------------------------------------------- stored-score backward
// Per 32-query step s and 64-key block kb: "negligible" iff every valid row of the step
// satisfies the exact-zero proof against its final LSE (same row_proof inputs as the
// forward). Evaluated by one warp (lane = row); key/value items and dQ items evaluate
// it identically, so they agree on which dS entries exist.
// Vote of one step's 32 rows (lane = row) for one 64-key block with preloaded,
// warp-uniform kmax / mask-tile max (a step lies inside one 64-row mask block).
PA2_DEV bool step_vote(const Attn& a, bool valid, float qn, float lse, float kmax, float mmax) {
  return __all_sync(0xffffffffu, !valid || pattn::row_proof(a, qn, kmax, mmax, lse));
}

// Item (b, 128-key block), 32-query steps, reading the forward's stored scores x:
//   P = exp(x - lse) (exactly 0 for negligible halves, whose x the forward may not
//   have written: a tile skipped by the forward proof is negligible here because
//   the final LSE >= the forward's running maximum), delta = rowsum(dO*O) (same
//   ascending-e chain as v1, per step), dP^T = V.dO^T, dS = P*(dP-delta) op c,
//   dV += P^T.dO, dK += dS^T.Q; dS goes to the scratch for the dQ pass.
// Mappings as backward_kv_item. Shared floats: V 128x68; Q, dO, O 32x68; lse, delta
// 32; P, dS (x staging) 32x132; step flags 2x128.
constexpr int KVS_FLOATS = 128 * L64 + 3 * 32 * L64 + 64 + 2 * 32 * L128 + 256;

template <int SMODE>
__device__ __noinline__ void backward_kv_stored_item(const Attn a, const float* xs, float* dsg, int b, int kblk, float* sm) {
  float* Vs = sm;
  float* Qs = Vs + 128 * L64;
  float* Ds = Qs + 32 * L64;
  float* Os = Ds + 32 * L64;
  float* lse = Os + 32 * L64;
  float* del = lse + 32;
  float* Pb = del + 32;
  float* Sb = Pb + 32 * L128;
  float* flag = Sb + 32 * L128;  // [2][128]: half h of step s negligible
  const int t = threadIdx.x, steps = (a.t + 31) / 32;
  const int kA = t >> 3, qA = t & 7;
  const int w = t >> 5, ln = t & 31, kB = 4 * (w >> 1) + (ln >> 3), eB = 8 * (w & 1) + (ln & 7);
  const int j0 = kblk * 128, kvalid = min(128, a.t - j0);
  const long long stride = ds_stride(a);
  load_rows<128, L64>(Vs, a.v + b * a.v_b + (long long)j0 * a.v_r, a.v_r, kvalid);
  cp_commit();
  {
    // Each warp votes its steps w, w+8, ...: all inputs loaded first (one latency).
    const int n = pattn::nblk(a), kb0 = 2 * kblk, kb1 = 2 * kblk + 1;
    const float km0 = kb0 < n ? a.kmax[(long long)b * n + kb0] : 0.f, km1 = kb1 < n ? a.kmax[(long long)b * n + kb1] : 0.f;
    float rl[16], rq[16], m0[16], m1[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const int s = w + 8 * k, i = 32 * s + ln;
      const bool ok = s < steps && i < a.t;
      rl[k] = ok ? a.lse[(long long)b * a.t + i] : 0.f;
      rq[k] = ok ? a.qn[(long long)b * a.t + i] : 0.f;
      m0[k] = (s < steps && kb0 < n) ? pattn::mask_tile_max(a, b, s >> 1, kb0) : 0.f;
      m1[k] = (s < steps && kb1 < n) ? pattn::mask_tile_max(a, b, s >> 1, kb1) : 0.f;
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const int s = w + 8 * k, i = 32 * s + ln;
      if (s >= steps) break;
      const bool v = i < a.t;
      const bool n0 = kb0 >= n || step_vote(a, v, rq[k], rl[k], km0, m0[k]);
      const bool n1 = kb1 >= n || step_vote(a, v, rq[k], rl[k], km1, m1[k]);
      if (ln == 0) { flag[s] = n0 ? 1.f : 0.f; flag[128 + s] = n1 ? 1.f : 0.f; }
    }
  }
  __syncthreads();
  auto needed = [&](int s) { return !(flag[s] != 0.f && flag[128 + s] != 0.f); };
  // Register prefetch of one step: Q, dO, O chunks (2 each), stored scores (4 float4), lse.
  float4 pq[2], pd[2], po[2], px[4];
  float pl = 0.f;
  auto fetch = [&](int s) {
    const int i0 = s * 32, valid = min(32, a.t - i0);
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + NT * u, r = idx >> 4, c = idx & 15;
      const bool ok = r < valid;
      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      pq[u] = ok ? ld4(a.q + b * a.q_b + (long long)(i0 + r) * a.q_r + 4 * c) : z;
      pd[u] = ok ? ld4(a.dout + b * a.d_b + (long long)(i0 + r) * a.d_r + 4 * c) : z;
      po[u] = ok ? ld4(a.oin + b * a.o_b + (long long)(i0 + r) * a.o_r + 4 * c) : z;
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + NT * u, r = idx >> 5, c = idx & 31, h = c >> 4;
      const bool ok = r < valid && flag[h * 128 + s] == 0.f;  // only scores the forward wrote
      px[u] = ok ? ld4(xs + ((long long)b * stride + i0 + r) * stride + j0 + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    if (t < 32) pl = t < valid ? a.lse[(long long)b * a.t + i0 + t] : 0.f;
  };
  float dk[8][4], dv[8][4];
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int c = 0; c < 4; ++c) { dk[r][c] = 0.f; dv[r][c] = 0.f; }
  int s = 0;
  while (s < steps && !needed(s)) ++s;
  if (s < steps) fetch(s);
  const float* vrow = Vs + kA * L64;  // keys kA + 32r
  const float* grow = Ds + qA * L64;  // queries qA + 8c
  while (s < steps) {
    const int i0 = s * 32;
    __syncthreads();  // previous step's readers of Qs/Ds/Os/Pb/Sb done
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int idx = t + NT * u, r = idx >> 4, c = idx & 15;
      st4(Qs + r * L64 + 4 * c, pq[u]);
      st4(Ds + r * L64 + 4 * c, pd[u]);
      st4(Os + r * L64 + 4 * c, po[u]);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + NT * u, r = idx >> 5, c = idx & 31;
      st4(Sb + r * L128 + 4 * c, px[u]);
    }
    if (t < 32) lse[t] = pl;
    const bool neg0 = flag[s] != 0.f, neg1 = flag[128 + s] != 0.f;
    int next = s + 1;
    while (next < steps && !needed(next)) ++next;
    if (next < steps) fetch(next);  // in flight during this step's arithmetic
    cp_wait<0>();  // V (first step)
    __syncthreads();
    {  // delta = rowsum(dO*O): 8 threads per row, ascending within 8-element parts, fixed tree
      const int row = t >> 3, part = t & 7;
      const float4 x0 = ld4(Ds + row * L64 + 8 * part), x1 = ld4(Ds + row * L64 + 8 * part + 4);
      const float4 y0 = ld4(Os + row * L64 + 8 * part), y1 = ld4(Os + row * L64 + 8 * part + 4);
      float acc = 0.f;
      acc = fmaf(x0.x, y0.x, acc); acc = fmaf(x0.y, y0.y, acc); acc = fmaf(x0.z, y0.z, acc); acc = fmaf(x0.w, y0.w, acc);
      acc = fmaf(x1.x, y1.x, acc); acc = fmaf(x1.y, y1.y, acc); acc = fmaf(x1.z, y1.z, acc); acc = fmaf(x1.w, y1.w, acc);
#pragma unroll
      for (int o = 1; o < 8; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
      if (part == 0) del[row] = acc;
    }
    float g[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) g[r][c] = 0.f;
    {
      float4 vv[2][4], gv[2][4];
      auto lds = [&](float4 (&v)[4], float4 (&gg)[4], int c4) {
#pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = ld4(vrow + r * 32 * L64 + 4 * c4);
#pragma unroll
        for (int c = 0; c < 4; ++c) gg[c] = ld4(grow + c * 8 * L64 + 4 * c4);
      };
      auto fma = [&](const float4 (&v)[4], const float4 (&gg)[4]) {
#pragma unroll
        for (int e = 0; e < 4; ++e)
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float va[4] = {v[r].x, v[r].y, v[r].z, v[r].w};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const float ga[4] = {gg[c].x, gg[c].y, gg[c].z, gg[c].w};
              g[r][c] = fmaf(ga[e], va[e], g[r][c]);  // dP[i][j] = dot(dO_i, v_j)
            }
          }
      };
#if PA2_PIPELINE && PA2_PIPELINE_BWD
      lds(vv[0], gv[0], 0);
#pragma unroll 1
      for (int c4 = 0; c4 < 16; c4 += 2) {
        lds(vv[1], gv[1], c4 + 1);
        fma(vv[0], gv[0]);
        if (c4 + 2 < 16) lds(vv[0], gv[0], c4 + 2);
        fma(vv[1], gv[1]);
      }
#else
#pragma unroll 4
      for (int c4 = 0; c4 < 16; ++c4) { lds(vv[0], gv[0], c4); fma(vv[0], gv[0]); }
#endif
    }
    float xr[4][4];
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) xr[r][c] = Sb[(qA + 8 * c) * L128 + kA + 32 * r];
    __syncthreads();  // delta visible; every thread has read its staged scores
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int j = j0 + kA + 32 * r;
      const bool neg = r < 2 ? neg0 : neg1;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int il = qA + 8 * c, i = i0 + il;
        const float pr = (!neg && i < a.t && j < a.t) ? expf(xr[r][c] - lse[il]) : 0.f;
        Pb[il * L128 + kA + 32 * r] = pr;
        Sb[il * L128 + kA + 32 * r] = fl_scale<SMODE>(pr * (g[r][c] - del[il]), a.scale);
      }
    }
    __syncthreads();
    {
      const float* pr = Pb + 8 * kB;
      const float* sr = Sb + 8 * kB;
      const float* gr = Ds + 4 * eB;
      const float* qr = Qs + 4 * eB;
      float4 op[2][6];  // p0, p1, d0, d1, dO, Q of one query row
      auto lds = [&](float4 (&x)[6], int q) {
        x[0] = ld4(pr + q * L128); x[1] = ld4(pr + q * L128 + 4);
        x[2] = ld4(sr + q * L128); x[3] = ld4(sr + q * L128 + 4);
        x[4] = ld4(gr + q * L64); x[5] = ld4(qr + q * L64);
      };
      auto fma = [&](const float4 (&x)[6]) {
        const float pa[8] = {x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w};
        const float sa[8] = {x[2].x, x[2].y, x[2].z, x[2].w, x[3].x, x[3].y, x[3].z, x[3].w};
        const float ga[4] = {x[4].x, x[4].y, x[4].z, x[4].w}, qa[4] = {x[5].x, x[5].y, x[5].z, x[5].w};
#pragma unroll
        for (int r = 0; r < 8; ++r)
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            dv[r][c] = fmaf(pa[r], ga[c], dv[r][c]);
            dk[r][c] = fmaf(sa[r], qa[c], dk[r][c]);
          }
      };
#if PA2_PIPELINE && PA2_PIPELINE_BWD
      lds(op[0], 0);
#pragma unroll 2
      for (int q = 0; q < 32; q += 2) {
        lds(op[1], q + 1);
        fma(op[0]);
        if (q + 2 < 32) lds(op[0], q + 2);
        fma(op[1]);
      }
#else
#pragma unroll 4
      for (int q = 0; q < 32; ++q) { lds(op[0], q); fma(op[0]); }
#endif
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int idx = t + NT * u, r = idx >> 5, c = idx & 31, i = i0 + r;
      if (i < a.t) st4(dsg + ((long long)b * stride + i) * stride + j0 + 4 * c, ld4(Sb + r * L128 + 4 * c));
    }
    s = next;
  }
  cp_wait<0>();
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int j = j0 + 8 * kB + r;
    if (j >= a.t) continue;
    st4(a.dk + ((long long)b * a.t + j) * 64 + 4 * eB, make_float4(dk[r][0], dk[r][1], dk[r][2], dk[r][3]));
    st4(a.dv + ((long long)b * a.t + j) * 64 + 4 * eB, make_float4(dv[r][0], dv[r][1], dv[r][2], dv[r][3]));
  }
  __syncthreads();
}

// ---------------------------------------------------------------- dQ items
// Item (b, 128-query block): dQ = sum over key chunks of dS[q][k] K[k][d], ascending k.
// Threads: qg = t>>3 rows qg+32a, kg = t&7 columns 4kg..+3, 32+4kg..+3.
// Shared floats: two stages of dS 128x68 and K 64x64; 2x64 chunk flags.
constexpr int DQ_FLOATS = 2 * (128 * L64 + 64 * 64) + 4 * 64 + 3 * 64;

// STEPS: dS entries exist per (32-query step, 128-key block) as decided by
// step_negligible (stored-score backward); otherwise per 64x64 delta skip table.
template <bool STEPS>
__device__ __noinline__ void dq_item(const Attn a, const float* dsg, int b, int qblk, float* sm) {
  constexpr int STAGE = 128 * L64 + 64 * 64;
  float* flag = sm + 2 * STAGE;  // STEPS: [4 steps][64 kb64] written; else [2 halves][64]
  const int t = threadIdx.x, kg = t & 7, qg = t >> 3, n = pattn::nblk(a);
  const int i0 = qblk * 128, qvalid = min(128, a.t - i0);
  const long long dstride = ds_stride(a);
  if constexpr (STEPS) {
    // Warp w < 4 owns step 4*qblk + w. A step's dS row block for 128-key block kc/2 was
    // written iff not both 64-key halves are negligible; 64-key chunk kc is read iff the
    // block was written and kc itself is not negligible (else it holds exact zeros).
    // kmax and mask-tile maxima are staged in shared memory first (one latency).
    float* kmx = flag + 4 * 64;  // [64]
    float* mmx = kmx + 64;       // [2][64]
    if (t < 64 && t < n) kmx[t] = a.kmax[(long long)b * n + t];
    if (t < 128) {
      const int h = t >> 6, kc = t & 63, qb = 2 * qblk + h;
      if (kc < n) mmx[t] = qb < n ? pattn::mask_tile_max(a, b, qb, kc) : 0.f;
    }
    __syncthreads();
    const int w = t >> 5, ln = t & 31, s = 4 * qblk + w, i = 32 * s + ln;
    if (w < 4) {
      const bool v = i < a.t;
      const float lse = v ? a.lse[(long long)b * a.t + i] : 0.f, qn = v ? a.qn[(long long)b * a.t + i] : 0.f;
      for (int kc = 0; kc < n; ++kc) {
        const bool neg = step_vote(a, v, qn, lse, kmx[kc], mmx[(w >> 1) * 64 + kc]);
        if (ln == 0) flag[w * 64 + kc] = neg ? 1.f : 0.f;
      }
    }
  } else if (t < 2 * 64) {
    const int h = t >> 6, kc = t & 63;
    flag[t] = skip64(a, b, 2 * qblk + h, kc) ? 1.f : 0.f;
  }
  __syncthreads();
  auto zero_rows = [&](int r, int kc) {  // true: dS rows r of chunk kc are exact zeros / unwritten
    if constexpr (STEPS) return flag[(r >> 5) * 64 + kc] != 0.f;
    else return flag[(r >> 6) * 64 + kc] != 0.f;
  };
  auto needed = [&](int kc) {
    if (kc >= n) return false;
    if constexpr (STEPS) {
      return !(flag[kc] != 0.f && flag[64 + kc] != 0.f && flag[128 + kc] != 0.f && flag[192 + kc] != 0.f);
    } else {
      return !(flag[kc] != 0.f && flag[64 + kc] != 0.f);
    }
  };
  auto issue = [&](int kc, float* st) {
    const int j0 = kc * 64;
#pragma unroll
    for (int u = 0; u < 128 * 16 / NT; ++u) {
      const int idx = t + NT * u, r = idx >> 4, c = idx & 15;
      const bool ok = r < qvalid && !zero_rows(r, kc);
      const float* src = dsg + ((long long)b * dstride + i0 + (ok ? r : 0)) * dstride + j0 + 4 * c;
      cp16(st + r * L64 + 4 * c, src, ok);
    }
    load_rows<64, 64>(st + 128 * L64, a.k + b * a.k_b + (long long)j0 * a.k_r, a.k_r, min(64, a.t - j0));
    cp_commit();
  };
  float acc[4][8];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[r][c] = 0.f;
  int kc = 0;
  while (kc < n && !needed(kc)) ++kc;
  int buf = 0;
  if (kc < n) issue(kc, sm);
  while (kc < n) {
    const float* Dsb = sm + buf * STAGE;
    const float* Kk = Dsb + 128 * L64;
    cp_wait<0>();
    __syncthreads();
    int next = kc + 1;
    while (next < n && !needed(next)) ++next;
    if (next < n) issue(next, sm + (buf ^ 1) * STAGE);
    const float* drow = Dsb + qg * L64;  // rows qg + 32a
    const float* krow = Kk + 4 * kg;
    {
      float4 dv[2][4], kk[2][8];
      auto lds = [&](float4 (&d)[4], float4 (&k)[8], int c4) {
#pragma unroll
        for (int r = 0; r < 4; ++r) d[r] = ld4(drow + r * 32 * L64 + 4 * c4);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          k[2 * e] = ld4(krow + (4 * c4 + e) * 64);
          k[2 * e + 1] = ld4(krow + (4 * c4 + e) * 64 + 32);
        }
      };
      auto fma = [&](const float4 (&d)[4], const float4 (&k)[8]) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float4 k0 = k[2 * e], k1 = k[2 * e + 1];
#pragma unroll
          for (int r = 0; r < 4; ++r) {
            const float da[4] = {d[r].x, d[r].y, d[r].z, d[r].w};
            acc[r][0] = fmaf(da[e], k0.x, acc[r][0]); acc[r][1] = fmaf(da[e], k0.y, acc[r][1]);
            acc[r][2] = fmaf(da[e], k0.z, acc[r][2]); acc[r][3] = fmaf(da[e], k0.w, acc[r][3]);
            acc[r][4] = fmaf(da[e], k1.x, acc[r][4]); acc[r][5] = fmaf(da[e], k1.y, acc[r][5]);
            acc[r][6] = fmaf(da[e], k1.z, acc[r][6]); acc[r][7] = fmaf(da[e], k1.w, acc[r][7]);
          }
        }
      };
#if PA2_PIPELINE && PA2_PIPELINE_BWD
      lds(dv[0], kk[0], 0);
#pragma unroll 1
      for (int c4 = 0; c4 < 16; c4 += 2) {
        lds(dv[1], kk[1], c4 + 1);
        fma(dv[0], kk[0]);
        if (c4 + 2 < 16) lds(dv[0], kk[0], c4 + 2);
        fma(dv[1], kk[1]);
      }
#else
#pragma unroll 4
      for (int c4 = 0; c4 < 16; ++c4) { lds(dv[0], kk[0], c4); fma(dv[0], kk[0]); }
#endif
    }
    __syncthreads();  // done reading this stage before it is refilled
    buf ^= 1;
    kc = next;
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int i = i0 + qg + 32 * r;
    if (i >= a.t) continue;
    float* dst = a.dq + ((long long)b * a.t + i) * 64;
    st4(dst + 4 * kg, make_float4(acc[r][0], acc[r][1], acc[r][2], acc[r][3]));
    st4(dst + 32 + 4 * kg, make_float4(acc[r][4], acc[r][5], acc[r][6], acc[r][7]));
  }
  __syncthreads();
}

}  // namespace pattn2
