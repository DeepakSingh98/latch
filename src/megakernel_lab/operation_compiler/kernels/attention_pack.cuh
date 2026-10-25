// LATCH-TINYMOE-ATTN operand pack for P-OPS4-D flat-long attention (ops.attention_rules ``pack``).
//
// Writes the padded contiguous BF16 operands [B][H][T][64] that the captured graph materializes before an explicit
// BF16 SDPA (optional FP32 -> BF16 round-to-nearest cast, grouped-query head repetition h -> h / G, zero channels
// [W, 64)), reading each operand from its pre-layout source through (batch, head, row) element strides with unit
// channel stride; and the flat body's per-tile key-norm proof table with prepare_unit's exact lane mapping and
// arithmetic on the same BF16 values. Optionally the constant mask's proof tables (prepare_unit's mask units,
// identical values: maxima and all-zero flags are exact).
//
// Code size matters more than unrolling here: these bodies run once per layer from a cold instruction cache, so
// they are short loops (grid-stride 16-byte chunks for the copy; one warp per tile for the tables, partially
// unrolled). Sources must be 16-byte (BF16) / 32-byte (FP32) aligned per 8-channel chunk (checked at emission).
// Include after attention_long_flat.cuh.
#pragma once

namespace pattnpack {

struct Src {
  const void* p;
  long long sb, sh, sr;  // element strides of (batch, source head, row); unit stride along channels
  int fp32;              // 1: FP32 source rounded to BF16 (__float2bfloat16_rn); 0: BF16 source copied
  int group;             // query heads per source head (grouped-query repetition)
  int vec;               // alignment proof (always 1: emission refuses otherwise)
  int width;             // real channels (multiple of 8, <= 64); channels [width, 64) are written as +0
  // Half-rotation rotary embedding of a BF16 source (half > 0): y = x * cos + rot(x) * sin in FP32 with the
  // captured elementwise expression, rot(x) = cat(-x[half:], x[:half]) in BF16, then rounded to BF16.
  const float* cos;
  const float* sin;
  long long crow;        // row (position) stride of the cos/sin tables; unit channel stride
  int half;
};

__device__ __forceinline__ float bf_at(const uint4& w, int e) {
  const uint32_t word = e < 2 ? w.x : e < 4 ? w.y : e < 6 ? w.z : w.w;
  return (e & 1) ? pflatlong::bf_hi(word) : pflatlong::bf_lo(word);
}

__device__ __forceinline__ uint32_t pack_word(float lo, float hi);

// Rotary chunk: channels [c0, c0 + 8) of row ``row`` from the BF16 row at ``x`` (c0 + 8 <= width).
__device__ __forceinline__ uint4 rope_chunk(const Src& s, const __nv_bfloat16* x, unsigned row, int c0) {
  const int partner = c0 < s.half ? c0 + s.half : c0 - s.half;
  const uint4 xv = __ldcg(reinterpret_cast<const uint4*>(x + c0));
  const uint4 pv = __ldcg(reinterpret_cast<const uint4*>(x + partner));
  const float* cp = s.cos + (long long)row * s.crow + c0;
  const float* sp = s.sin + (long long)row * s.crow + c0;
  const float4 ca = __ldcg(reinterpret_cast<const float4*>(cp)), cb = __ldcg(reinterpret_cast<const float4*>(cp) + 1);
  const float4 sa = __ldcg(reinterpret_cast<const float4*>(sp)), sb = __ldcg(reinterpret_cast<const float4*>(sp) + 1);
  const float cv[8] = {ca.x, ca.y, ca.z, ca.w, cb.x, cb.y, cb.z, cb.w};
  const float sv[8] = {sa.x, sa.y, sa.z, sa.w, sb.x, sb.y, sb.z, sb.w};
  float y[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float p = bf_at(pv, e);
    const __nv_bfloat16 r = c0 < s.half ? __float2bfloat16_rn((-p)) : __float2bfloat16_rn(p);
    // The captured elementwise body's expression (mul, mul, add with alpha 1), same operand order.
    const float l18 = (bf_at(xv, e) * cv[e]);
    const float l21 = (__bfloat162float(r) * sv[e]);
    y[e] = (l18 + (l21 * 1.0f));
  }
  return make_uint4(pack_word(y[0], y[1]), pack_word(y[2], y[3]), pack_word(y[4], y[5]), pack_word(y[6], y[7]));
}

struct Args {
  Src src[3];                // q, k, v
  __nv_bfloat16* out[3];     // contiguous [B][H][T][64]
  float* kn;                 // [B*H][T/64] key-norm table, or nullptr
  int heads, bh, t;          // query heads, B*H, context
};

__device__ __forceinline__ uint32_t pack_word(float lo, float hi) {
  const __nv_bfloat16 a = __float2bfloat16_rn(lo), b = __float2bfloat16_rn(hi);
  return (uint32_t)__bfloat16_as_ushort(a) | ((uint32_t)__bfloat16_as_ushort(b) << 16);
}

// Eight channels [8 * sub, 8 * sub + 8) of query-head row (bh, row) of one operand as BF16x2 words.
template <int HEADS>
__device__ __forceinline__ uint4 chunk(const Src& s, unsigned bh, unsigned row, unsigned sub) {
  uint4 r = make_uint4(0u, 0u, 0u, 0u);
  if (8 * (int)sub >= s.width) return r;
  if (s.half) {
    const long long base = (long long)(bh / HEADS) * s.sb + (long long)((bh % HEADS) / (unsigned)s.group) * s.sh
                           + (long long)row * s.sr;
    return rope_chunk(s, static_cast<const __nv_bfloat16*>(s.p) + base, row, 8 * (int)sub);
  }
  const long long e = (long long)(bh / HEADS) * s.sb + (long long)((bh % HEADS) / (unsigned)s.group) * s.sh
                      + (long long)row * s.sr + 8 * sub;
  if (s.fp32) {
    const float4* p = reinterpret_cast<const float4*>(static_cast<const float*>(s.p) + e);
    const float4 x = __ldcg(p), y = __ldcg(p + 1);
    r.x = pack_word(x.x, x.y); r.y = pack_word(x.z, x.w); r.z = pack_word(y.x, y.y); r.w = pack_word(y.z, y.w);
  } else {
    r = __ldcg(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(s.p) + e));
  }
  return r;
}

// Padded operands: one 16-byte output chunk per iteration over every thread of the grid (constant shapes, so
// the index arithmetic is strength-reduced 32-bit work).
template <int BH, int T, int HEADS>
__device__ __forceinline__ void pack_copy(const Args& a, unsigned tid, unsigned threads) {
  constexpr unsigned PER = (unsigned)BH * T * 8u, TOTAL = 3u * PER, STEP = 4;
  // STEP chunks per iteration: every load is issued before any store (stores may alias nothing they read, but
  // the compiler cannot prove it), so one iteration costs one memory round trip.
#pragma unroll 1
  for (unsigned c0 = tid; c0 < TOTAL; c0 += STEP * threads) {
    uint4 w[STEP];
#pragma unroll
    for (int j = 0; j < STEP; ++j) {
      const unsigned c = c0 + j * threads;
      const unsigned which = c / PER, rem = c - which * PER;
      w[j] = c < TOTAL ? chunk<HEADS>(a.src[which < 3 ? which : 0], rem / (T * 8u), (rem >> 3) % T, rem & 7u)
                       : make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int j = 0; j < STEP; ++j) {
      const unsigned c = c0 + j * threads;
      if (c < TOTAL) {
        const unsigned which = c / PER;
        reinterpret_cast<uint4*>(a.out[which])[c - which * PER] = w[j];
      }
    }
  }
}

// Key-norm table unit (bh, kt): prepare_unit's lanes (8 lanes x 16 bytes per row, 4 rows per step), per-row FMA
// order and xor-shuffle sum, fmaxf over rows in increasing step order, then warp_max and the 1 + 1/256 margin.
template <int T, int HEADS>
__device__ __forceinline__ void key_norm_unit(const Args& a, unsigned u) {
  const unsigned lane = threadIdx.x & 31, nt = T / 64;
  const unsigned bh = u / nt, kt = u % nt, sub = lane & 7, rsub = lane >> 3;
  float best = 0.f;
#pragma unroll 8
  for (int i = 0; i < 16; ++i) {
    const uint4 w = chunk<HEADS>(a.src[1], bh, kt * 64 + 4 * i + rsub, sub);
    const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      acc = fmaf(pflatlong::bf_lo(ws[e]), pflatlong::bf_lo(ws[e]), acc);
      acc = fmaf(pflatlong::bf_hi(ws[e]), pflatlong::bf_hi(ws[e]), acc);
    }
#pragma unroll
    for (int o = 1; o < 8; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    best = fmaxf(best, sqrtf(acc));
  }
  best = pflatlong::warp_max(best);
  if (lane == 0) a.kn[u] = best * (1.0f + 1.0f / 256.0f);
}

// prepare_unit's mask unit (mb, qt, kt) with the same lanes (16 lanes x 16 bytes per row, 2 rows per step):
// tile maximum, per-row maxima and the all-zero flag (maxima and the flag are exact, so the loop shape is free).
__device__ __forceinline__ void mask_unit(const pflatlong::Args& m, long long u) {
  const int lane = threadIdx.x & 31, nt = m.t >> 6;
  const int mb = (int)(u / ((long long)nt * nt)), rem = (int)(u % ((long long)nt * nt));
  const int qt = rem / nt, kt = rem % nt, sub = lane & 15, rsub = lane >> 4;
  const float* mbase = m.mask + (long long)mb * m.m_b;
  float tmax = -INFINITY;
  bool zero = true;
#pragma unroll 4
  for (int i = 0; i < 32; ++i) {
    const int row = qt * 64 + 2 * i + rsub;
    const float* rp = mbase + (long long)row * m.m_i + (long long)(kt * 64) * m.m_j;
    const float4 w = make_float4(__ldcg(rp + (long long)(4 * sub) * m.m_j), __ldcg(rp + (long long)(4 * sub + 1) * m.m_j),
                                 __ldcg(rp + (long long)(4 * sub + 2) * m.m_j), __ldcg(rp + (long long)(4 * sub + 3) * m.m_j));
    float rmax = fmaxf(fmaxf(w.x, w.y), fmaxf(w.z, w.w));
    zero = zero && w.x == 0.f && w.y == 0.f && w.z == 0.f && w.w == 0.f;
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) rmax = fmaxf(rmax, __shfl_xor_sync(0xffffffffu, rmax, o));
    if (sub == 0) m.rm[((long long)mb * nt + kt) * m.t + row] = rmax;
    tmax = fmaxf(tmax, rmax);
  }
  tmax = pflatlong::warp_max(tmax);
  zero = __all_sync(0xffffffffu, zero);
  if (lane == 0) {
    m.mmax[u] = tmax;
    m.mzero[u] = zero ? 1.f : 0.f;
  }
}

}  // namespace pattnpack
