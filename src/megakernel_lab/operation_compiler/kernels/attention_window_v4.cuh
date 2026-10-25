// P9-C v4 phases of the rectangular masked mixed attention (new code; included by attention_window.cuh inside
// namespace pwin). The arithmetic is that of the v3 phases above: the same tiles, in the same order, with the same
// rounding points and the same FP32 association, so results are bitwise identical (checked by the P9-C bench and the
// native compare task). Only the instruction stream changes:
//
//  1. Operand rows are loaded in register batches (every load of a batch is issued before its shared stores), row
//     norms read 16-byte shared vectors, and the CTA's next item is prefetched to L2 while the current item computes.
//  2. Fast items. An item is fast when every mask-table block of the item is all +-0 or binary with value
//     v <= -2^100, every query/key row norm is <= 2^43 (so |s| <= 2^86) and the scale is a Markstein-safe divisor.
//     Fast items form x = s / c with the unguarded Markstein quotient and apply binary blocks by selection:
//     fl(x' + v) = v whenever |x'| < 2^103. The quotient is RN(s / c) except possibly when 0 < |s| < 2^-90, where it
//     is still below 2^-89 in magnitude. When the row maximum m satisfies |m| >= 2^-60, such an x cannot be the
//     maximum (that would force m <= 2^-89 with m >= x > -2^-89) and fl(x - m) = -m for the exact and the computed x
//     alike (|x| is below a quarter ulp of m), so m, l, P and every result are unchanged. Forward row blocks whose
//     maxima fail the test are recomputed with the guarded v3 arithmetic; backward items use the guarded arithmetic
//     unless every stored row maximum passes it.
//  3. P = e / l: e in [0, 1] and l in [1, N] for fast items (the maximal element contributes 1), so the Markstein
//     quotient is exact unless 0 < e < 2^-90 somewhere in the warp's tile, in which case the tile uses IEEE division
//     (both are the correctly rounded quotient, as in v3).
//  4. Backward row statistics are interleaved {m, l, RN(1/l), delta} so the transposed sweep reads 16-byte vectors.
#pragma once

#ifndef PWIN_V4
#define PWIN_V4 1
#endif

// Round a tile to BF16 and widen back (same values as rbf on every element): one packed conversion per pair
// (cvt.rn.bf16x2.f32) instead of one conversion per element.
PWIN_DEV void rbf_tile(float (&s)[8][4]) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      const uint32_t w = pack(s[n][e], s[n][e + 1]);
      s[n][e] = __uint_as_float(w << 16);
      s[n][e + 1] = __uint_as_float(w & 0xffff0000u);
    }
}

PWIN_DEV bool vec_bf16_op(const Op& op) {
  return !op.fp32 && op.sd == 1 && (op.sr % 8) == 0 && (op.sz % 8) == 0 &&
         ((reinterpret_cast<uintptr_t>(op.p) & 15) == 0);
}
PWIN_DEV bool vec_fp32_op(const Op& op) {
  return op.fp32 && op.sd == 1 && (op.sr % 4) == 0 && (op.sz % 4) == 0 &&
         ((reinterpret_cast<uintptr_t>(op.p) & 15) == 0);
}

// Rows [0, rows) of operand z into shared [rows][LD] (same values as load_rows), loads batched ahead of stores.
// rows % 64 == 0 (admission): thread t owns 16-byte chunk t % 4 of rows t / 4 + 64 u.
PWIN_DEV void load_rows4(bf16* dst, const Op& op, int z, int rows) {
  const int t = (int)threadIdx.x, r0 = t >> 2, c = 8 * (t & 3);
  bf16* d0 = dst + r0 * LD + c;
  if (vec_bf16_op(op) && rows % 64 == 0) {
    const bf16* s0 = op.p + (long long)z * op.sz + (long long)r0 * op.sr + c;
    const long long step = 64LL * op.sr;
    for (int u0 = 0; u0 < rows / 64; u0 += 8) {
      uint4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (u0 + u < rows / 64) v[u] = __ldg(reinterpret_cast<const uint4*>(s0 + (u0 + u) * step));
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (u0 + u < rows / 64) *reinterpret_cast<uint4*>(d0 + (u0 + u) * 64 * LD) = v[u];
    }
  } else if (vec_fp32_op(op) && rows % 64 == 0) {
    const float* s0 = reinterpret_cast<const float*>(op.p) + (long long)z * op.sz + (long long)r0 * op.sr + c;
    const long long step = 64LL * op.sr;
    for (int u0 = 0; u0 < rows / 64; u0 += 4) {
      float4 v[4][2];
#pragma unroll
      for (int u = 0; u < 4; ++u)
        if (u0 + u < rows / 64) {
          v[u][0] = __ldg(reinterpret_cast<const float4*>(s0 + (u0 + u) * step));
          v[u][1] = __ldg(reinterpret_cast<const float4*>(s0 + (u0 + u) * step + 4));
        }
#pragma unroll
      for (int u = 0; u < 4; ++u)
        if (u0 + u < rows / 64)
          *reinterpret_cast<uint4*>(d0 + (u0 + u) * 64 * LD) =
              make_uint4(pack(v[u][0].x, v[u][0].y), pack(v[u][0].z, v[u][0].w),
                         pack(v[u][1].x, v[u][1].y), pack(v[u][1].z, v[u][1].w));
    }
  } else {
    load_rows(dst, op, z, 0, rows, rows);
  }
}

// L2 prefetch of operand z's rows: dense rows (sr == HD) are one contiguous range, prefetched by 128-byte lines.
#ifndef PWIN_PREFETCH
#define PWIN_PREFETCH 0
#endif
PWIN_DEV void prefetch_rows(const Op& op, int z, int rows) {
  if (!PWIN_PREFETCH || op.sd != 1 || op.sr != HD) return;
  const int esz = op.fp32 ? 4 : 2;
  const char* p = reinterpret_cast<const char*>(op.p) + (long long)z * op.sz * esz;
  const int lines = (rows * HD * esz + 127) / 128;
  for (int i = (int)threadIdx.x; i < lines; i += NT) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + 128LL * i));
}

// row_norms with 16-byte shared reads (same order: d ascending). Returns whether every norm is <= 2^43.
PWIN_DEV bool row_norms4(const bf16* s, int rows, float* out) {
  bool ok = true;
  for (int r = (int)threadIdx.x; r < rows; r += NT) {
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < HD / 8; ++c) {
      const uint4 w = *reinterpret_cast<const uint4*>(s + r * LD + 8 * c);
      const unsigned u[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float x0 = __uint_as_float(u[k] << 16), x1 = __uint_as_float(u[k] & 0xffff0000u);
        acc = __fadd_rn(acc, __fmul_rn(x0, x0));
        acc = __fadd_rn(acc, __fmul_rn(x1, x1));
      }
    }
    const float nr = sqrtf(acc);
    out[r] = nr;
    ok = ok && nr <= 8.796093022208e12f;   // 2^43
  }
  return ok;
}

// v4 item layout: tiles as item_smem, then st[M] (float4 {m, l, RN(1/l), delta}), qn[M], kn[N], kmax[N/64].
struct Smem4 {
  bf16 *Qs, *Ks, *Vs, *Ds;
  float4* st;
  float *qn, *kn, *kmax;
  const long long* tb;
  bool fast;
};
__host__ __device__ constexpr long long item_shared_bytes4(int M, int N) {
  return (2LL * (2 * M + 2 * N) * LD + 16LL * M + 4LL * (M + N + N / KB) + 15) / 16 * 16;
}
PWIN_DEV Smem4 item_smem4(const Args& a, float* smem) {
  Smem4 s;
  s.Qs = reinterpret_cast<bf16*>(smem);
  s.Ks = s.Qs + a.M * LD;
  s.Vs = s.Ks + a.N * LD;
  s.Ds = s.Vs + a.N * LD;
  s.st = reinterpret_cast<float4*>(s.Ds + a.M * LD);   // 16-byte aligned: tiles are multiples of 16 rows * 80 bytes
  s.qn = reinterpret_cast<float*>(s.st + a.M);
  s.kn = s.qn + a.M;
  s.kmax = s.kn + a.N;
  s.tb = nullptr;
  s.fast = false;
  return s;
}
PWIN_DEV const long long* tab_entry4(const Args& a, const Smem4& s, int rb, int kb) {
  return s.tb + TE * (rb * (a.N / KB) + kb);
}

// Loads item z; returns whether it is a fast item (block-uniform). st: load row statistics (backward).
PWIN_DEV void load_src_rows(const Args& a, bf16* dst, int z, int slot, int rows, int roff, bool rope);
PWIN_DEV void load_dout_rows(const Args& a, bf16* dst, int z, int rows);
template <bool ABS>
PWIN_DEV bool load_item4(const Args& a, Smem4& s, int z, bool v, bool d, bool st) {
  s.tb = a.table ? a.table + TE * (((long long)(z / a.g1) * a.tb0 + (long long)(z % a.g1) * a.tb1) *
                                   (a.M / 16) * (a.N / KB))
                 : nullptr;
  if (ABS) {
    load_src_rows(a, s.Qs, z, 0, a.M, 0, true);
    load_src_rows(a, s.Ks, z, 1, a.N, a.M - a.N, true);
    if (v) load_src_rows(a, s.Vs, z, 2, a.N, a.M - a.N, false);
    if (d) load_dout_rows(a, s.Ds, z, a.M);
  } else {
    load_rows4(s.Qs, a.q, z, a.M);
    load_rows4(s.Ks, a.k, z, a.N);
    if (v) load_rows4(s.Vs, a.v, z, a.N);
    if (d) load_rows4(s.Ds, a.dout, z, a.M);
  }
  bool ok = a.scale_mode != 2 || mk_divisor(a.scale);
  if (st) {
    for (int r = (int)threadIdx.x; r < a.M; r += NT) {
      const float2 x = __ldcg(reinterpret_cast<const float2*>(a.stats) + (long long)z * a.M + r);
      s.st[r] = make_float4(x.x, x.y, __frcp_rn(x.y), 0.f);
      ok = ok && fabsf(x.x) >= 8.673617379884035e-19f && fabsf(x.x) < INFINITY;   // 2^-60 <= |m| < inf
    }
  }
  if (a.table) {
    const int entries = (a.M / 16) * (a.N / KB);
    for (int e = (int)threadIdx.x; e < entries; e += NT) {
      const long long* t = s.tb + TE * e;
      ok = ok && (tab_zero(t) || (tab_binary(t) && tab_value(t) <= -1.2676506002282294e30f));   // v <= -2^100
    }
  } else if (a.mask) {
    ok = false;
  }
  __syncthreads();
  ok = row_norms4(s.Qs, a.M, s.qn) && ok;
  ok = row_norms4(s.Ks, a.N, s.kn) && ok;
  __syncthreads();
  block_max(s.kn, a.N, s.kmax);
  s.fast = __syncthreads_and(ok ? 1 : 0) != 0;
  return s.fast;
}

// Fast score finish (fast items only): rbf, unguarded Markstein scale, binary blocks by selection. zero / v: the
// block's all-zero flag and binary value (row-block context); the row bitmaps load before the arithmetic.
PWIN_DEV void finish_fast(const Args& a, float (&s)[8][4], const long long* t, int zero, float v) {
  const int lane = threadIdx.x & 31;
  const bool masked = a.mask && !zero;
  unsigned long long b0 = ~0ull, b1 = ~0ull;
  if (masked) {
    b0 = (unsigned long long)__ldg(t + 4 + (lane >> 2));
    b1 = (unsigned long long)__ldg(t + 12 + (lane >> 2));
  }
  rbf_tile(s);
  if (a.scale_mode == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[n][e] = __fmul_rn(s[n][e], a.scale);
  } else if (a.scale_mode == 2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[n][e] = mk_div(s[n][e], a.scale, a.rscale);
  }
  if (!masked) return;
  b0 >>= 2 * (lane & 3);
  b1 >>= 2 * (lane & 3);
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = 8 * n + (e & 1);
      if (!((((e >> 1) ? b1 : b0) >> j) & 1ull)) s[n][e] = v;
    }
}
// Transposed fast finish (16 keys from kj0 x 64 query rows from ri0), as finish_scores_t.
PWIN_DEV void finish_fast_t(const Args& a, float (&s)[8][4], int kj0, const long long* const (&t4)[4]) {
  const int lane = threadIdx.x & 31;
  rbf_tile(s);
  if (a.scale_mode == 1) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[n][e] = __fmul_rn(s[n][e], a.scale);
  } else if (a.scale_mode == 2) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) s[n][e] = mk_div(s[n][e], a.scale, a.rscale);
  }
  if (!a.mask) return;
  const int jl0 = (kj0 % KB) + (lane >> 2);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const long long* t = t4[q];
    if (tab_zero(t)) continue;
    const float v = tab_value(t);
#pragma unroll
    for (int n = 2 * q; n < 2 * q + 2; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jl = jl0 + 8 * (e >> 1);
        const int il = (8 * n + 2 * (lane & 3) + (e & 1)) & 15;
        if (!((tab_bits(t, il) >> jl) & 1ull)) s[n][e] = v;
      }
  }
}

// Tile of e / b (per-row divisors b[0], b[1] in [1, N]; e in [0, 1]): Markstein unless some 0 < e < 2^-90 in the
// warp's tile, then IEEE (both correctly rounded).
PWIN_DEV void tile_div_probs(float (&v)[8][4], const float (&b)[2], const float (&y)[2]) {
  bool tiny = false;
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) tiny = tiny || (__float_as_uint(v[n][e]) - 1u) < 0x12800000u - 1u;   // (0, 2^-90)
  if (!__any_sync(0xffffffffu, tiny)) {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[n][e] = mk_div(v[n][e], b[e >> 1], y[e >> 1]);
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[n][e] = __fdiv_rn(v[n][e], b[e >> 1]);
  }
}


// ---------------------------------------------------------------------------- P9-C absorbed operands (ops/window_fusion.py)
// Item z = (b * H + h) * nb + n. Projection src [B][L][3][H][HD] (BF16), dO head-merged [B][L][H][HD] (BF16), rotation
// tables [L][HD/2] (FP32). Values equal those of the replaced graph: the rotation is fl(x1 c - x2 s) | fl(x2 c + x1 s)
// on the exact FP32 widening of the BF16 projection, rounded once to BF16 (the absorbed operand cast).
PWIN_DEV void item_coords(const Args& a, int z, int& b, int& h, int& n) {
  n = z % a.gnb;
  const int bh = z / a.gnb;
  h = bh % a.gH;
  b = bh / a.gH;
}
// Rows [0, rows) of item z from projection slot (0 q, 1 k, 2 v); row r is position n W + r + roff, zero outside [0, L).
// rows % 128 == 0 (lowering admission).
PWIN_DEV void load_src_rows(const Args& a, bf16* dst, int z, int slot, int rows, int roff, bool rope) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  const long long hdt = (long long)a.gH * HD, pstride = 3 * hdt;
  const bf16* base = a.src + (long long)b * a.gL * pstride + (long long)slot * hdt + (long long)h * HD;
  const int t0 = n * a.gW + roff;
  const int t = (int)threadIdx.x;
  if (!rope) {
    const int r0 = t >> 2, c = 8 * (t & 3);
    for (int u0 = 0; u0 < rows / 64; u0 += 8) {
      uint4 v[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const int pos = t0 + r0 + 64 * (u0 + u);
        v[u] = (u0 + u < rows / 64 && pos >= 0 && pos < a.gL)
                   ? __ldg(reinterpret_cast<const uint4*>(base + (long long)pos * pstride + c))
                   : make_uint4(0u, 0u, 0u, 0u);
      }
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (u0 + u < rows / 64) *reinterpret_cast<uint4*>(dst + (r0 + 64 * (u0 + u)) * LD + c) = v[u];
    }
    return;
  }
  constexpr int H2 = HD / 2;
  const int r0 = t >> 1, c = 8 * (t & 1);
  for (int u0 = 0; u0 < rows / 128; u0 += 2) {
    uint4 x1[2], x2[2];
    float4 cs[2][2], sn[2][2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int pos = t0 + r0 + 128 * (u0 + u);
      if (u0 + u < rows / 128 && pos >= 0 && pos < a.gL) {
        const bf16* row = base + (long long)pos * pstride;
        x1[u] = __ldg(reinterpret_cast<const uint4*>(row + c));
        x2[u] = __ldg(reinterpret_cast<const uint4*>(row + H2 + c));
        const float* cr = a.rcos + (long long)pos * H2 + c;
        const float* sr = a.rsin + (long long)pos * H2 + c;
        cs[u][0] = __ldg(reinterpret_cast<const float4*>(cr));
        cs[u][1] = __ldg(reinterpret_cast<const float4*>(cr + 4));
        sn[u][0] = __ldg(reinterpret_cast<const float4*>(sr));
        sn[u][1] = __ldg(reinterpret_cast<const float4*>(sr + 4));
      } else {
        x1[u] = x2[u] = make_uint4(0u, 0u, 0u, 0u);
        cs[u][0] = cs[u][1] = sn[u][0] = sn[u][1] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      if (u0 + u >= rows / 128) continue;
      const unsigned w1[4] = {x1[u].x, x1[u].y, x1[u].z, x1[u].w}, w2[4] = {x2[u].x, x2[u].y, x2[u].z, x2[u].w};
      const float cc[8] = {cs[u][0].x, cs[u][0].y, cs[u][0].z, cs[u][0].w, cs[u][1].x, cs[u][1].y, cs[u][1].z, cs[u][1].w};
      const float ss[8] = {sn[u][0].x, sn[u][0].y, sn[u][0].z, sn[u][0].w, sn[u][1].x, sn[u][1].y, sn[u][1].z, sn[u][1].w};
      uint32_t o1[4], o2[4];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float y1[2], y2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const float a1 = __uint_as_float(e ? (w1[k] & 0xffff0000u) : (w1[k] << 16));
          const float a2 = __uint_as_float(e ? (w2[k] & 0xffff0000u) : (w2[k] << 16));
          const float cv = cc[2 * k + e], sv = ss[2 * k + e];
          y1[e] = __fsub_rn(__fmul_rn(a1, cv), __fmul_rn(a2, sv));
          y2[e] = __fadd_rn(__fmul_rn(a2, cv), __fmul_rn(a1, sv));
        }
        o1[k] = pack(y1[0], y1[1]);
        o2[k] = pack(y2[0], y2[1]);
      }
      bf16* drow = dst + (r0 + 128 * (u0 + u)) * LD;
      *reinterpret_cast<uint4*>(drow + c) = make_uint4(o1[0], o1[1], o1[2], o1[3]);
      *reinterpret_cast<uint4*>(drow + H2 + c) = make_uint4(o2[0], o2[1], o2[2], o2[3]);
    }
  }
}
// dO rows of item z from the head-merged BF16 dsrc.
PWIN_DEV void load_dout_rows(const Args& a, bf16* dst, int z, int rows) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  const long long hdt = (long long)a.gH * HD;
  const bf16* base = a.dsrc + ((long long)b * a.gL + (long long)n * a.gW) * hdt + (long long)h * HD;
  const int t = (int)threadIdx.x, r0 = t >> 2, c = 8 * (t & 3);
  for (int u0 = 0; u0 < rows / 64; u0 += 8) {
    uint4 v[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (u0 + u < rows / 64) v[u] = __ldg(reinterpret_cast<const uint4*>(base + (long long)(r0 + 64 * (u0 + u)) * hdt + c));
#pragma unroll
    for (int u = 0; u < 8; ++u)
      if (u0 + u < rows / 64) *reinterpret_cast<uint4*>(dst + (r0 + 64 * (u0 + u)) * LD + c) = v[u];
  }
}
// O rows (row0 .. row0 + 15) of item z, head-merged BF16.
PWIN_DEV void store_o_merged(const Args& a, const float (&acc)[HD / 8][4], int z, int row0) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
  const long long hdt = (long long)a.gH * HD;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int row = row0 + (lane >> 2) + 8 * hh;
    bf16* dst = a.o_m + ((long long)b * a.gL + (long long)n * a.gW + row) * hdt + (long long)h * HD;
#pragma unroll
    for (int nt = 0; nt < HD / 8; ++nt)
      *reinterpret_cast<uint32_t*>(dst + 8 * nt + c) = pack(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
  }
}
// dQ rows of item z: rbf(dQ), the captured rotation derivative (first fl(g2 s + g1 c), second fl(g2 c + (-g1) s)),
// canonical zero (fl(x + 0), as the graph's zero-padded sums), rounded to BF16 into dq_o [B][H][L][HD].
PWIN_DEV void store_dq_rope(const Args& a, const float (&acc)[HD / 8][4], int z, int row0) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
  constexpr int H2 = HD / 2;
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    const int row = row0 + (lane >> 2) + 8 * hh, pos = n * a.gW + row;
    bf16* dst = a.dq_o + ((long long)z * a.gW + row) * HD;
#pragma unroll
    for (int nt = 0; nt < H2 / 8; ++nt) {
      const float2 cv = __ldg(reinterpret_cast<const float2*>(a.rcos + (long long)pos * H2 + 8 * nt + c));
      const float2 sv = __ldg(reinterpret_cast<const float2*>(a.rsin + (long long)pos * H2 + 8 * nt + c));
      float f[2], g[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float g1 = rbf(acc[nt][2 * hh + e]), g2 = rbf(acc[nt + H2 / 8][2 * hh + e]);
        const float co = e ? cv.y : cv.x, si = e ? sv.y : sv.x;
        f[e] = __fadd_rn(__fadd_rn(__fmul_rn(g2, si), __fmul_rn(g1, co)), 0.f);
        g[e] = __fadd_rn(__fadd_rn(__fmul_rn(g2, co), __fmul_rn(-g1, si)), 0.f);
      }
      *reinterpret_cast<uint32_t*>(dst + 8 * nt + c) = pack(f[0], f[1]);
      *reinterpret_cast<uint32_t*>(dst + H2 + 8 * nt + c) = pack(g[0], g[1]);
    }
  }
}
// dK / dV rows (keys k0 .. k0 + 15) of item z: previous-block keys (k0 < W) go to the previous item's slot of the
// shifted buffer (dropped for n = 0), current-block keys to this item's slot of the current buffer; BF16 results.
PWIN_DEV void store_kv_split(const Args& a, bf16* cur, bf16* prv, const float (&acc)[HD / 8][4], int z, int k0) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  bf16* dst;
  int r0;
  if (k0 < a.gW) {
    if (n == 0) return;
    dst = prv + (long long)(z - 1) * a.gW * HD;
    r0 = k0;
  } else {
    dst = cur + (long long)z * a.gW * HD;
    r0 = k0 - a.gW;
  }
  const int lane = threadIdx.x & 31, c = 2 * (lane & 3);
#pragma unroll
  for (int hh = 0; hh < 2; ++hh) {
    bf16* d = dst + (long long)(r0 + (lane >> 2) + 8 * hh) * HD;
#pragma unroll
    for (int nt = 0; nt < HD / 8; ++nt)
      *reinterpret_cast<uint32_t*>(d + 8 * nt + c) = pack(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
  }
}
// The last block's shifted slot has no next block: zero (slice_backward padding).
PWIN_DEV void zero_last_shifted(const Args& a, int z) {
  int b, h, n;
  item_coords(a, z, b, h, n);
  if (n != a.gnb - 1) return;
  const long long base = (long long)z * a.gW * HD;
  for (int i = (int)threadIdx.x; i < a.gW * HD / 8; i += NT) {
    reinterpret_cast<uint4*>(a.dkp_o + base)[i] = make_uint4(0u, 0u, 0u, 0u);
    reinterpret_cast<uint4*>(a.dvp_o + base)[i] = make_uint4(0u, 0u, 0u, 0u);
  }
}

// ---------------------------------------------------------------------------------------------- forward
// Row-block context: the mask-table header and key-norm maximum of key block kb are held by lane kb (nkb <= 8), the
// thread's two query-row norms in registers; tiles read them by shuffle instead of dependent global/shared loads.
struct Rb {
  float tm, vv, km, qn0, qn1;
  int zr;
  const long long* t0;
};
PWIN_DEV Rb rb_ctx(const Args& a, const Smem4& s, int rb) {
  Rb c;
  const int lane = threadIdx.x & 31, nkb = a.N / KB;
  c.t0 = a.table ? tab_entry4(a, s, rb, 0) : nullptr;
  c.tm = 0.f;
  c.vv = 0.f;
  c.km = 0.f;
  c.zr = 1;
  if (lane < nkb) {
    c.km = s.kmax[lane];
    if (a.table) {
      const long long* t = c.t0 + TE * lane;
      c.tm = __int_as_float((int)(__ldg(t) & 0xffffffffLL));
      c.zr = __ldg(t + 1) != 0;
      c.vv = __int_as_float((int)(__ldg(t + 3) & 0xffffffffLL));
    }
  }
  c.qn0 = s.qn[16 * rb + (lane >> 2)];
  c.qn1 = s.qn[16 * rb + (lane >> 2) + 8];
  return c;
}
// Class of key block kb (0 compute, 2 skip candidate) and its skip test against row maxima m (warp-uniform).
PWIN_DEV int class_of(const Args& a, const Rb& c, int kb, float& tmax) {
  tmax = __shfl_sync(0xffffffffu, c.tm, kb);
  return a.table && tmax < -1e30f ? 2 : 0;
}
PWIN_DEV bool skip_of(const Args& a, const Rb& c, int kb, float tmax, const float (&m)[2]) {
  return skip_rows(a, tmax, c.qn0, c.qn1, __shfl_sync(0xffffffffu, c.km, kb), m);
}
template <bool FAST>
PWIN_DEV void scores4(const Args& a, const Smem4& s, const Rb& c, float (&x)[8][4], int z, int r0, int kb) {
  const long long* t = a.table ? c.t0 + TE * kb : nullptr;
  const int zero = __shfl_sync(0xffffffffu, c.zr, kb);
  const float v = __shfl_sync(0xffffffffu, c.vv, kb);
  tile_nt(x, s.Qs, r0, s.Ks, kb * KB);
  if (FAST) finish_fast(a, x, t, zero, v);
  else finish_scores(a, x, z, r0, kb * KB, t);
}
// P tile from (m, l); false when skipped.
template <bool FAST>
PWIN_DEV bool probs4(const Args& a, const Smem4& s, const Rb& c, float (&p)[8][4], int z, int r0, int kb,
                     const float (&m)[2], const float (&l)[2]) {
  float tmax;
  const int cls = class_of(a, c, kb, tmax);
  if (cls == 2 && skip_of(a, c, kb, tmax, m)) return false;
  scores4<FAST>(a, s, c, p, z, r0, kb);
  const float rl[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) p[n][e] = expf(__fsub_rn(p[n][e], m[e >> 1]));
  if (FAST) tile_div_probs(p, l, rl);
  else tile_div_rows(p, l, rl);
  return true;
}
PWIN_DEV bool rows_ok(const float (&m)[2]) {
  const bool ok = fabsf(m[0]) >= 8.673617379884035e-19f && fabsf(m[0]) < INFINITY &&
                  fabsf(m[1]) >= 8.673617379884035e-19f && fabsf(m[1]) < INFINITY;
  return __all_sync(0xffffffffu, ok);
}
// One 16-row block (v3 fwd_phase sweeps). FAST: returns false (nothing stored) when a row maximum fails the test.
template <bool FAST, bool ABS>
PWIN_DEV bool fwd_rows4(const Args& a, const Smem4& s, int z, int r0) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, rb = r0 / 16;
  const Rb c = rb_ctx(a, s, rb);
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  for (int pass = 0; pass < 2; ++pass) {
    for (int kb = 0; kb < nkb; ++kb) {
      float tmax;
      const int cls = class_of(a, c, kb, tmax);
      if ((cls == 2) != (pass == 1)) continue;
      if (cls == 2 && skip_of(a, c, kb, tmax, m)) continue;
      float x[8][4];
      scores4<FAST>(a, s, c, x, z, r0, kb);
      online(x, m, l);
    }
  }
  if (FAST && !rows_ok(m)) return false;
  quad_sum2(l);
  float o[HD / 8][4];
  zero_acc(o);
  for (int kb = 0; kb < nkb; ++kb) {
    float p[8][4];
    if (!probs4<FAST>(a, s, c, p, z, r0, kb, m, l)) continue;
    tile_pv(o, p, s.Vs, kb * KB);
  }
  if (ABS) store_o_merged(a, o, z, r0);
  else store_rows(a.out, o, z, r0, a.M);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      reinterpret_cast<float2*>(a.stats)[(long long)z * a.M + r0 + (lane >> 2) + 8 * h] = make_float2(m[h], l[h]);
  }
  return true;
}


// Fast-item row block with software-pipelined tiles (PWIN_PIPE): the next computed tile's products are issued
// before the current tile's elementwise work. Same tiles, order, arithmetic and results as fwd_rows4<true>.
#ifndef PWIN_PIPE
#define PWIN_PIPE 0   // measured slower (P9-C gpu-09: 0.213 vs 0.172 ms, spills)
#endif
PWIN_DEV void fwd_tile1(const Args& a, const Rb& c, float (&x)[8][4], int kb, float (&m)[2], float (&l)[2]) {
  const long long* t = a.table ? c.t0 + TE * kb : nullptr;
  finish_fast(a, x, t, __shfl_sync(0xffffffffu, c.zr, kb), __shfl_sync(0xffffffffu, c.vv, kb));
  online(x, m, l);
}
PWIN_DEV void fwd_tile2(const Args& a, const Smem4& s, const Rb& c, float (&x)[8][4], int kb, const float (&m)[2],
                        const float (&l)[2], const float (&rl)[2], float (&o)[HD / 8][4]) {
  const long long* t = a.table ? c.t0 + TE * kb : nullptr;
  finish_fast(a, x, t, __shfl_sync(0xffffffffu, c.zr, kb), __shfl_sync(0xffffffffu, c.vv, kb));
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) x[n][e] = expf(__fsub_rn(x[n][e], m[e >> 1]));
  tile_div_probs(x, l, rl);
  tile_pv(o, x, s.Vs, kb * KB);
}
PWIN_DEV int next_bit(unsigned& list) {
  if (!list) return -1;
  const int k = __ffs(list) - 1;
  list &= list - 1;
  return k;
}
template <bool ABS>
PWIN_DEV bool fwd_rows4_pipe(const Args& a, const Smem4& s, int z, int r0) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, rb = r0 / 16;
  const Rb c = rb_ctx(a, s, rb);
  const unsigned all = nkb >= 32 ? 0xffffffffu : ((1u << nkb) - 1u);
  const unsigned cand = __ballot_sync(0xffffffffu, lane < nkb && a.table && c.tm < -1e30f) & all;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  float xa[8][4], xb[8][4];
  // Sweep 1, pass 0: class-0 blocks ascending (pipelined).
  {
    unsigned list = all & ~cand;
    int ka = next_bit(list);
    if (ka >= 0) tile_nt(xa, s.Qs, r0, s.Ks, ka * KB);
    while (ka >= 0) {
      const int kb = next_bit(list);
      if (kb >= 0) tile_nt(xb, s.Qs, r0, s.Ks, kb * KB);
      fwd_tile1(a, c, xa, ka, m, l);
      if (kb < 0) break;
      ka = next_bit(list);
      if (ka >= 0) tile_nt(xa, s.Qs, r0, s.Ks, ka * KB);
      fwd_tile1(a, c, xb, kb, m, l);
    }
  }
  // Pass 1: skip candidates ascending against the running maxima.
  {
    unsigned list = cand;
    for (int kb = next_bit(list); kb >= 0; kb = next_bit(list)) {
      float tmax;
      class_of(a, c, kb, tmax);
      if (skip_of(a, c, kb, tmax, m)) continue;
      tile_nt(xa, s.Qs, r0, s.Ks, kb * KB);
      fwd_tile1(a, c, xa, kb, m, l);
    }
  }
  if (!rows_ok(m)) return false;
  quad_sum2(l);
  // Sweep 2: the computed set (class 0, and candidates not skipped against the final maxima), ascending.
  unsigned list2 = all & ~cand;
  {
    unsigned list = cand;
    for (int kb = next_bit(list); kb >= 0; kb = next_bit(list)) {
      float tmax;
      class_of(a, c, kb, tmax);
      if (!skip_of(a, c, kb, tmax, m)) list2 |= 1u << kb;
    }
  }
  const float rl[2] = {__frcp_rn(l[0]), __frcp_rn(l[1])};
  float o[HD / 8][4];
  zero_acc(o);
  {
    int ka = next_bit(list2);
    if (ka >= 0) tile_nt(xa, s.Qs, r0, s.Ks, ka * KB);
    while (ka >= 0) {
      const int kb = next_bit(list2);
      if (kb >= 0) tile_nt(xb, s.Qs, r0, s.Ks, kb * KB);
      fwd_tile2(a, s, c, xa, ka, m, l, rl, o);
      if (kb < 0) break;
      ka = next_bit(list2);
      if (ka >= 0) tile_nt(xa, s.Qs, r0, s.Ks, ka * KB);
      fwd_tile2(a, s, c, xb, kb, m, l, rl, o);
    }
  }
  if (ABS) store_o_merged(a, o, z, r0);
  else store_rows(a.out, o, z, r0, a.M);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h)
      reinterpret_cast<float2*>(a.stats)[(long long)z * a.M + r0 + (lane >> 2) + 8 * h] = make_float2(m[h], l[h]);
  }
  return true;
}

template <bool ABS>
PWIN_DEV void fwd_phase_v4(const Args& a, float* smem) {
  Smem4 s = item_smem4(a, smem);
  const int warp = threadIdx.x >> 5;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    const bool fast = load_item4<ABS>(a, s, z, true, false, false);
    const int zn = z + (int)gridDim.x;
    if (zn < a.Z && !ABS) {
      prefetch_rows(a.q, zn, a.M);
      prefetch_rows(a.k, zn, a.N);
      prefetch_rows(a.v, zn, a.N);
    }
    for (int rb = warp; rb < a.M / 16; rb += NW) {
      if (!(fast && (PWIN_PIPE ? fwd_rows4_pipe<ABS>(a, s, z, 16 * rb) : fwd_rows4<true, ABS>(a, s, z, 16 * rb))))
        fwd_rows4<false, ABS>(a, s, z, 16 * rb);
    }
  }
}

// --------------------------------------------------------------------------------------------- backward
PWIN_DEV void stats4_of(const Smem4& s, int r0, float (&m)[2], float (&l)[2]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const float4 v = s.st[r0 + (lane >> 2) + 8 * h];
    m[h] = v.x;
    l[h] = v.y;
  }
}
template <bool FAST>
PWIN_DEV void delta_rows4(const Args& a, Smem4& s, int z, int rb) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, r0 = 16 * rb;
  float m[2], l[2];
  stats4_of(s, r0, m, l);
  const Rb c = rb_ctx(a, s, rb);
  float sum[2] = {0.f, 0.f};
  for (int kb = 0; kb < nkb; ++kb) {
    float p[8][4], dp[8][4];
    if (!probs4<FAST>(a, s, c, p, z, r0, kb, m, l)) continue;
    tile_nt(dp, s.Ds, r0, s.Vs, kb * KB);
    rbf_tile(dp);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      sum[0] = __fadd_rn(__fadd_rn(sum[0], __fmul_rn(p[n][0], dp[n][0])), __fmul_rn(p[n][1], dp[n][1]));
      sum[1] = __fadd_rn(__fadd_rn(sum[1], __fmul_rn(p[n][2], dp[n][2])), __fmul_rn(p[n][3], dp[n][3]));
    }
  }
  quad_sum2(sum);
  if ((lane & 3) == 0) {
#pragma unroll
    for (int h = 0; h < 2; ++h) s.st[r0 + (lane >> 2) + 8 * h].w = sum[h];
  }
}
template <bool FAST, bool ABS>
PWIN_DEV void dq_rows4(const Args& a, const Smem4& s, const Res& out, int z, int rb) {
  const int lane = threadIdx.x & 31, nkb = a.N / KB, r0 = 16 * rb;
  float m[2], l[2], dl[2];
  stats4_of(s, r0, m, l);
#pragma unroll
  for (int h = 0; h < 2; ++h) dl[h] = s.st[r0 + (lane >> 2) + 8 * h].w;
  const Rb c = rb_ctx(a, s, rb);
  float dq[HD / 8][4];
  zero_acc(dq);
  for (int kb = 0; kb < nkb; ++kb) {
    float p[8][4], dp[8][4];
    if (!probs4<FAST>(a, s, c, p, z, r0, kb, m, l)) continue;
    tile_nt(dp, s.Ds, r0, s.Vs, kb * KB);
    rbf_tile(dp);
#pragma unroll
    for (int n = 0; n < 8; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(dp[n][e], dl[e >> 1]));
    tile_scale(a, dp);
    rbf_tile(dp);
    tile_pv(dq, dp, s.Ks, kb * KB);
  }
  if (ABS) store_dq_rope(a, dq, z, r0);
  else store_rows(out, dq, z, r0, a.M);
}
template <bool FAST, bool ABS>
PWIN_DEV void dkv_keys4(const Args& a, const Smem4& s, const Res& dk_out, const Res& dv_out, int z, int kb16) {
  const int lane = threadIdx.x & 31, k0 = 16 * kb16;
  float kmax = 0.f;
  for (int r = 0; r < 16; ++r) kmax = fmaxf(kmax, s.kn[k0 + r]);
  float dk[HD / 8][4], dv[HD / 8][4];
  zero_acc(dk);
  zero_acc(dv);
  for (int i0 = 0; i0 < a.M; i0 += 64) {
    const long long* t4[4] = {nullptr, nullptr, nullptr, nullptr};
    if (a.table) {
      float tmax = -INFINITY;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        t4[q] = tab_entry4(a, s, i0 / 16 + q, k0 / KB);
        tmax = fmaxf(tmax, tab_max(t4[q]));
      }
      if (tmax < -1e30f) {
        bool ok = true;
        for (int c = lane; c < 64; c += 32) {
          const float bound = __fmul_rn(__fmul_rn(scale_abs(a), __fmul_rn(s.qn[i0 + c], kmax)), SLACK);
          ok = ok && (tmax + bound < s.st[i0 + c].x - MARGIN);
        }
        if (__all_sync(0xffffffffu, ok)) continue;
      }
    }
    float p[8][4], dp[8][4];
    tile_nt(p, s.Ks, k0, s.Qs, i0);           // S^T: keys x 64 query rows
    if (FAST) finish_fast_t(a, p, k0, t4);
    else finish_scores_t(a, p, z, k0, i0, t4);
    tile_nt(dp, s.Vs, k0, s.Ds, i0);          // dP^T = V dO^T
    // Row statistics are read where used (scalar/16-byte shared loads) so they are not live across the tile.
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int i = i0 + 8 * n + 2 * (lane & 3);
      const float m0 = s.st[i].x, m1 = s.st[i + 1].x;
#pragma unroll
      for (int e = 0; e < 4; ++e) p[n][e] = expf(__fsub_rn(p[n][e], (e & 1) ? m1 : m0));
    }
    // P = e / l with a per-column divisor: Markstein when exact for the whole warp tile, else IEEE (both RN).
    {
      bool ok = true;
      if (FAST) {
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
          for (int e = 0; e < 4; ++e) ok = ok && !((__float_as_uint(p[n][e]) - 1u) < 0x12800000u - 1u);
      } else {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const int i = i0 + 8 * n + 2 * (lane & 3);
          const float l0 = s.st[i].y, l1 = s.st[i + 1].y;
#pragma unroll
          for (int e = 0; e < 4; ++e) ok = ok && mk_safe(p[n][e]) && mk_divisor((e & 1) ? l1 : l0);
        }
      }
      if (__all_sync(0xffffffffu, ok)) {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const int i = i0 + 8 * n + 2 * (lane & 3);
          const float4 A = s.st[i], B = s.st[i + 1];
#pragma unroll
          for (int e = 0; e < 4; ++e) p[n][e] = mk_div(p[n][e], (e & 1) ? B.y : A.y, (e & 1) ? B.z : A.z);
        }
      } else {
#pragma unroll
        for (int n = 0; n < 8; ++n) {
          const int i = i0 + 8 * n + 2 * (lane & 3);
          const float l0 = s.st[i].y, l1 = s.st[i + 1].y;
#pragma unroll
          for (int e = 0; e < 4; ++e) p[n][e] = __fdiv_rn(p[n][e], (e & 1) ? l1 : l0);
        }
      }
    }
    rbf_tile(dp);
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      const int i = i0 + 8 * n + 2 * (lane & 3);
      const float d0 = s.st[i].w, d1 = s.st[i + 1].w;
#pragma unroll
      for (int e = 0; e < 4; ++e) dp[n][e] = __fmul_rn(p[n][e], __fsub_rn(dp[n][e], (e & 1) ? d1 : d0));
    }
    tile_scale(a, dp);
    rbf_tile(dp);
    tile_pv(dv, p, s.Ds, i0);                 // dV += bf16(P)^T dO
    tile_pv(dk, dp, s.Qs, i0);                // dK += dS^T q
  }
  if (ABS) {
    store_kv_split(a, a.dkc_o, a.dkp_o, dk, z, k0);
    store_kv_split(a, a.dvc_o, a.dvp_o, dv, z, k0);
  } else {
    store_rows(dk_out, dk, z, k0, a.N);
    store_rows(dv_out, dv, z, k0, a.N);
  }
}

template <bool ABS>
PWIN_DEV void bwd_phase_v4(const Args& a, const Res& dq_out, const Res& dk_out, const Res& dv_out, float* smem) {
  Smem4 s = item_smem4(a, smem);
  const int warp = threadIdx.x >> 5;
  for (int z = (int)blockIdx.x; z < a.Z; z += (int)gridDim.x) {
    if (z != (int)blockIdx.x) __syncthreads();
    const bool fast = load_item4<ABS>(a, s, z, true, true, true);
    const int zn = z + (int)gridDim.x;
    if (zn < a.Z && !ABS) {
      prefetch_rows(a.q, zn, a.M);
      prefetch_rows(a.k, zn, a.N);
      prefetch_rows(a.v, zn, a.N);
      prefetch_rows(a.dout, zn, a.M);
    }
    if (ABS) zero_last_shifted(a, z);
    for (int rb = warp; rb < a.M / 16; rb += NW) {
      if (fast) delta_rows4<true>(a, s, z, rb);
      else delta_rows4<false>(a, s, z, rb);
    }
    __syncthreads();
    const int nrb = a.M / 16, nkb16 = a.N / 16;
    for (int w = warp; w < nrb + nkb16; w += NW) {
      if (w < nrb) {
        if (fast) dq_rows4<true, ABS>(a, s, dq_out, z, w);
        else dq_rows4<false, ABS>(a, s, dq_out, z, w);
      } else {
        if (fast) dkv_keys4<true, ABS>(a, s, dk_out, dv_out, z, w - nrb);
        else dkv_keys4<false, ABS>(a, s, dk_out, dv_out, z, w - nrb);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------ dispatch
PWIN_DEV void fwd_phase(const Args& a, float* smem) {
  if (PWIN_V4) fwd_phase_v4<false>(a, smem);
  else fwd_phase_v3(a, smem);
}
PWIN_DEV void bwd_phase(const Args& a, const Res& dq_out, const Res& dk_out, const Res& dv_out, float* smem) {
  if (PWIN_V4) bwd_phase_v4<false>(a, dq_out, dk_out, dv_out, smem);
  else bwd_phase_v3(a, dq_out, dk_out, dv_out, smem);
}
// Absorbed operands (mattn_fwd2 / mattn_bwd2): separate instantiations so neither path carries the other's code.
PWIN_DEV void fwd2_phase(const Args& a, float* smem) { fwd_phase_v4<true>(a, smem); }
PWIN_DEV void bwd2_phase(const Args& a, float* smem) { bwd_phase_v4<true>(a, Res{}, Res{}, Res{}, smem); }
__host__ __device__ constexpr long long phase_shared_bytes(int M, int N) {
  return PWIN_V4 ? (item_shared_bytes4(M, N) > item_shared_bytes(M, N) ? item_shared_bytes4(M, N)
                                                                        : item_shared_bytes(M, N))
                 : item_shared_bytes(M, N);
}

// mattn_dsrc: projection gradient [B][L][3][H][HD] from dq (final query-slot values), dkc + dkp and dvc + dvp
// (FP32 sums of BF16 values; the key sum through the rotation derivative), canonical zeros, rounded to BF16.
// Task = (b, position, head, half-chunk c): 8 rotation pairs (d = c .. c + 7 and HD/2 + c ..).
PWIN_DEV void dsrc_phase(const Args& a) {
  constexpr int H2 = HD / 2;
  const long long hdt = (long long)a.gH * HD;
  const long long tasks = (long long)a.gB * a.gL * a.gH * 2;
  for (long long task = (long long)blockIdx.x * NT + threadIdx.x; task < tasks; task += (long long)gridDim.x * NT) {
    const int c = 8 * (int)(task & 1);
    const long long rest = task >> 1;
    const int h = (int)(rest % a.gH);
    const long long bp = rest / a.gH;                // b * L + position
    const int pos = (int)(bp % a.gL), b = (int)(bp / a.gL);
    const long long g = (((long long)b * a.gH + h) * a.gL + pos) * HD;   // [B][H][L][HD] element of (b, h, pos, 0)
    const uint4 q1 = __ldg(reinterpret_cast<const uint4*>(a.dq_o + g + c));
    const uint4 q2 = __ldg(reinterpret_cast<const uint4*>(a.dq_o + g + H2 + c));
    const uint4 kc1 = __ldg(reinterpret_cast<const uint4*>(a.dkc_o + g + c));
    const uint4 kc2 = __ldg(reinterpret_cast<const uint4*>(a.dkc_o + g + H2 + c));
    const uint4 kp1 = __ldg(reinterpret_cast<const uint4*>(a.dkp_o + g + c));
    const uint4 kp2 = __ldg(reinterpret_cast<const uint4*>(a.dkp_o + g + H2 + c));
    const uint4 vc1 = __ldg(reinterpret_cast<const uint4*>(a.dvc_o + g + c));
    const uint4 vc2 = __ldg(reinterpret_cast<const uint4*>(a.dvc_o + g + H2 + c));
    const uint4 vp1 = __ldg(reinterpret_cast<const uint4*>(a.dvp_o + g + c));
    const uint4 vp2 = __ldg(reinterpret_cast<const uint4*>(a.dvp_o + g + H2 + c));
    const float* cr = a.rcos + (long long)pos * H2 + c;
    const float* sr = a.rsin + (long long)pos * H2 + c;
    const float4 c0 = __ldg(reinterpret_cast<const float4*>(cr)), c1 = __ldg(reinterpret_cast<const float4*>(cr + 4));
    const float4 s0 = __ldg(reinterpret_cast<const float4*>(sr)), s1 = __ldg(reinterpret_cast<const float4*>(sr + 4));
    const float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    const float ss[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    const unsigned wkc1[4] = {kc1.x, kc1.y, kc1.z, kc1.w}, wkc2[4] = {kc2.x, kc2.y, kc2.z, kc2.w};
    const unsigned wkp1[4] = {kp1.x, kp1.y, kp1.z, kp1.w}, wkp2[4] = {kp2.x, kp2.y, kp2.z, kp2.w};
    const unsigned wvc1[4] = {vc1.x, vc1.y, vc1.z, vc1.w}, wvc2[4] = {vc2.x, vc2.y, vc2.z, vc2.w};
    const unsigned wvp1[4] = {vp1.x, vp1.y, vp1.z, vp1.w}, wvp2[4] = {vp2.x, vp2.y, vp2.z, vp2.w};
    uint32_t k1[4], k2[4], v1[4], v2[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float fk[2], sk[2], fv[2], sv2[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        auto wide = [e](unsigned w) { return __uint_as_float(e ? (w & 0xffff0000u) : (w << 16)); };
        const float g1 = __fadd_rn(wide(wkc1[k]), wide(wkp1[k]));     // dk first half
        const float g2 = __fadd_rn(wide(wkc2[k]), wide(wkp2[k]));     // dk second half
        const float co = cc[2 * k + e], si = ss[2 * k + e];
        fk[e] = __fadd_rn(__fadd_rn(__fmul_rn(g2, si), __fmul_rn(g1, co)), 0.f);
        sk[e] = __fadd_rn(__fadd_rn(__fmul_rn(g2, co), __fmul_rn(-g1, si)), 0.f);
        fv[e] = __fadd_rn(__fadd_rn(wide(wvc1[k]), wide(wvp1[k])), 0.f);
        sv2[e] = __fadd_rn(__fadd_rn(wide(wvc2[k]), wide(wvp2[k])), 0.f);
      }
      k1[k] = pack(fk[0], fk[1]);
      k2[k] = pack(sk[0], sk[1]);
      v1[k] = pack(fv[0], fv[1]);
      v2[k] = pack(sv2[0], sv2[1]);
    }
    bf16* out = a.proj_grad + bp * 3 * hdt + (long long)h * HD;
    *reinterpret_cast<uint4*>(out + c) = q1;
    *reinterpret_cast<uint4*>(out + H2 + c) = q2;
    *reinterpret_cast<uint4*>(out + hdt + c) = make_uint4(k1[0], k1[1], k1[2], k1[3]);
    *reinterpret_cast<uint4*>(out + hdt + H2 + c) = make_uint4(k2[0], k2[1], k2[2], k2[3]);
    *reinterpret_cast<uint4*>(out + 2 * hdt + c) = make_uint4(v1[0], v1[1], v1[2], v1[3]);
    *reinterpret_cast<uint4*>(out + 2 * hdt + H2 + c) = make_uint4(v2[0], v2[1], v2[2], v2[3]);
  }
}
