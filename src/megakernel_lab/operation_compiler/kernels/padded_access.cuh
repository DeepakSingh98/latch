// Step 02 (latch-step-02): grouped access to BF16 values stored with a padded row pitch.
//
// A logical row-major [rows, D] BF16 tensor is stored with rows P elements apart (P a multiple of 8, so every row
// starts on 16 bytes). Vectorized elementwise loops handle 8 consecutive logical elements b .. b + 7. When they lie
// in one row (column c = b % D, c + 8 <= D) they are 8 contiguous stored elements at a 2-byte aligned address whose
// offset within 16 bytes is uniform across a warp (consecutive groups advance c by 8); they are loaded with the two
// covering aligned 16-byte words and shifted, and stored with the widest aligned pieces. A group crossing a row end
// uses per-element addressing. Data movement only: the same bits are read and written.
#pragma once
#include <cstdint>

namespace padacc {

__device__ __forceinline__ unsigned pick4(unsigned q, unsigned a, unsigned b, unsigned c, unsigned d) {
  return (q & 2u) ? ((q & 1u) ? d : c) : ((q & 1u) ? b : a);
}

// 8 BF16 bit patterns (word k holds elements 2k, 2k + 1 in its low, high halves) of logical elements b .. b + 7.
template <unsigned D, unsigned P>
__device__ __forceinline__ uint4 gather8(const void* base, unsigned b) {
  const unsigned w = b / D, c = b - w * D;
  const unsigned short* row = reinterpret_cast<const unsigned short*>(base) + (unsigned long long)w * P;
  if (c + 8u <= D) {
    const unsigned long long a = reinterpret_cast<unsigned long long>(row + c);
    const unsigned off = (unsigned)(a & 15u);
    const uint4* q = reinterpret_cast<const uint4*>(a - off);
    const uint4 lo = q[0];
    if (off == 0u) return lo;
    const uint4 hi = q[1];
    const unsigned s = off >> 2, r = (off & 3u) * 8u;   // whole-word offset, remaining byte shift (0 or 16 bits)
    const unsigned w0 = pick4(s, lo.x, lo.y, lo.z, lo.w), w1 = pick4(s, lo.y, lo.z, lo.w, hi.x);
    const unsigned w2 = pick4(s, lo.z, lo.w, hi.x, hi.y), w3 = pick4(s, lo.w, hi.x, hi.y, hi.z);
    const unsigned w4 = pick4(s, hi.x, hi.y, hi.z, hi.w);
    return make_uint4(__funnelshift_r(w0, w1, r), __funnelshift_r(w1, w2, r), __funnelshift_r(w2, w3, r),
                      __funnelshift_r(w3, w4, r));
  }
  unsigned short e[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned col = c + (unsigned)k;
    e[k] = col < D ? row[col] : row[P + col - D];
  }
  return make_uint4(e[0] | ((unsigned)e[1] << 16), e[2] | ((unsigned)e[3] << 16), e[4] | ((unsigned)e[5] << 16),
                    e[6] | ((unsigned)e[7] << 16));
}

// Store 8 BF16 bit patterns (layout as gather8) to logical elements b .. b + 7.
template <unsigned D, unsigned P>
__device__ __forceinline__ void scatter8(void* base, unsigned b, uint4 v) {
  const unsigned w = b / D, c = b - w * D;
  unsigned short* row = reinterpret_cast<unsigned short*>(base) + (unsigned long long)w * P;
  if (c + 8u <= D) {
    unsigned short* p = row + c;
    const unsigned off = (unsigned)(reinterpret_cast<unsigned long long>(p) & 15u);
    if (off == 0u) {
      *reinterpret_cast<uint4*>(p) = v;
    } else if ((off & 7u) == 0u) {
      reinterpret_cast<uint2*>(p)[0] = make_uint2(v.x, v.y);
      reinterpret_cast<uint2*>(p)[1] = make_uint2(v.z, v.w);
    } else if ((off & 3u) == 0u) {
      unsigned* q = reinterpret_cast<unsigned*>(p);
      q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
    } else {
      // 2-byte aligned: first and last element alone, the 6 middle elements as three aligned words.
      p[0] = (unsigned short)(v.x & 0xffffu);
      unsigned* q = reinterpret_cast<unsigned*>(p + 1);
      q[0] = __funnelshift_r(v.x, v.y, 16);
      q[1] = __funnelshift_r(v.y, v.z, 16);
      q[2] = __funnelshift_r(v.z, v.w, 16);
      p[7] = (unsigned short)(v.w >> 16);
    }
    return;
  }
  const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const unsigned col = c + (unsigned)k;
    const unsigned short bits = (unsigned short)((k & 1) ? (words[k >> 1] >> 16) : (words[k >> 1] & 0xffffu));
    if (col < D) row[col] = bits; else row[P + col - D] = bits;
  }
}

}  // namespace padacc
