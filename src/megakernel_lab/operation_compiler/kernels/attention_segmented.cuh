// Packed segmented extension of P-OPS4-D/P9-A v7 BF16 WGMMA attention.
// Arithmetic and stage pipeline adapted from attention_long_flat.cuh's
// forward_item6/dq_item6/kv_item6; attribution and original bodies are retained.
// Only device-resident offsets select work. Operand rows are never padded per
// segment: cp.async zero-fills tile tails in shared memory, and stores are
// predicated. Statistics use [heads,total_rows]; no max-sequence allocations.
#pragma once
#ifndef PFLATLONG_WGMMA
#define PFLATLONG_WGMMA 1
#endif
#include "attention_long_flat.cuh"

namespace latch_segmented {
using namespace pflatlong;
struct Args : pflatlong::Args {
  int rows, groups;
  int64_t* work;
  // Tile readiness (ops.tile_ready): optional per-item row publication of this phase's row results (forward O, dK/dV,
  // dQ): one counter per tr_rb-row block of the packed rows, rows x 32 elements per (row, head) item. Null: none.
  unsigned* tr_cnt = nullptr;
  // Fused backward (VAR 8 dK/dV items): FP32 dQ accumulator of the single-pass backward. Null: none.
  float* dqa = nullptr;
  int tr_rb = 0;
  int tr_delay = 0;       // nanoseconds to wait before each publication (0: none; values unchanged)
  // Claimed items (ops.attention_claim, opt-in LATCH_SEG_ATTN_CLAIM): width-32 items are claimed from this
  // entry-zeroed per-phase ticket instead of the static snake order. Items are independent and write disjoint rows,
  // so only the warpgroup that computes an item changes (same items, same arithmetic, same work counters).
  // Null: static order.
#if defined(LATCH_SEG_ATTN_CLAIM)
  unsigned* tkt_claim = nullptr;
  int tkt_ahead = 1;      // claim the next ticket before computing the current item (hides the atomic latency)
  int tkt_rank = 0;       // tickets in rank-major order: every segment's largest tiles first (same items)
#endif
};
#if defined(LATCH_SEG_ATTN_CLAIM)
// One ticket from a per-phase claim counter (a single global atomic; no generic-address CAS loop).
PFLATLONG_DEV unsigned tkt_ticket(unsigned* c) {
  unsigned v;
  asm volatile("{\n.reg .u64 g;\ncvta.to.global.u64 g, %1;\natom.relaxed.gpu.global.add.u32 %0, [g], 1;\n}\n"
               : "=r"(v) : "l"(c) : "memory");
  return v;
}
#endif
// Release the rows an item just stored (one thread, after the warpgroup barrier that follows the stores).
PFLATLONG_DEV void tr_publish(unsigned* cnt, int rb, long long r0, int rows, int cols) {
  __threadfence();
  const long long r1 = r0 + rows;
#pragma unroll 1
  for (long long j = r0 / rb; j * rb < r1; ++j) {
    const long long lo = r0 > j * rb ? r0 : j * rb, hi = r1 < (j + 1) * rb ? r1 : (j + 1) * rb;
    atomicAdd(cnt + j, (unsigned)((hi - lo) * cols));
  }
}
PFLATLONG_DEV void add_count(int64_t* p, unsigned long long value) {
  atomicAdd(reinterpret_cast<unsigned long long*>(p), value);
}
PFLATLONG_DEV unsigned long long low_bits(int n) {
  return n >= 64 ? ~0ull : (n <= 0 ? 0ull : (1ull << n) - 1ull);
}
// cp.async's source-size operand zero-fills the shared tile without reading a
// later segment. The source for an invalid row is clamped to this valid tile.
PFLATLONG_DEV void cp16_tail(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::
      "r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
PFLATLONG_DEV void load_rows_tail(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                long long stride, int tid, int valid_rows) {
#pragma unroll
  for (int k=0; k<4; ++k) {
    const int idx=tid+k*128, r=idx>>3, c=idx&7;
    cp16_tail(dst+r*LD+8*c, src+(long long)(r<valid_rows?r:0)*stride+8*c, r<valid_rows);
  }
}
PFLATLONG_DEV void load_sw_tail(char* dst, const __nv_bfloat16* src,
                              long long stride, int tid, int valid_rows) {
#pragma unroll
  for (int k=0; k<4; ++k) {
    const int idx=tid+k*128, r=idx>>3, c=idx&7;
    cp16_tail(dst+r*128+((c^(r&7))<<4), src+(long long)(r<valid_rows?r:0)*stride+8*c, r<valid_rows);
  }
}
PFLATLONG_DEV void causal_scores(float (&s)[8][4], int length, int row_lo,
                                int row_hi, int key0, int lane, bool unused) {
#pragma unroll
  for (int n=0; n<8; ++n) {
    const int key=key0+8*n+2*(lane&3);
#pragma unroll
    for (int e=0; e<4; ++e) {
      const int row=e<2?row_lo:row_hi, col=key+(e&1);
      if (row>=length || col>=length || col>row) s[n][e]=-INFINITY;
    }
  }
}
// Runtime offsets are checked before any tensor access; zero-length segments
// contribute zero jobs. The prefix has just G+1 integers, regardless of lengths.
template <typename Index>
PFLATLONG_DEV void prepare(const Index* offsets, int64_t* prefix, int64_t* work,
                          int groups, int heads, int rows, int bound, int columns) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<groups*heads*columns; i+=gridDim.x*blockDim.x) work[i]=0;
  if (blockIdx.x==0 && threadIdx.x==0) {
    if (offsets[0]!=0 || offsets[groups]!=rows) __trap();
    prefix[0]=0;
    for (int g=0; g<groups; ++g) {
      const int64_t lo=offsets[g], hi=offsets[g+1];
      if (lo<0 || hi<lo || hi>rows || hi-lo>bound) __trap();
      prefix[g+1]=prefix[g]+((hi-lo+63)/64)*heads;
    }
  }
}

// The v6/v7 FP32 softmax and score-gradient recipe is copied here only to bind
// exp2f explicitly. Co-composed dense attention may define PFLATLONG_FTZ; that
// other operation's numerical policy must never change this primitive's values.
// WGMMA products, operand formats, packing and pipeline helpers remain reused.
template <bool MASKED, bool POW2>
PFLATLONG_DEV void softmax_tile(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                             float (&o)[8][4], const float* mlo, const float* mhi, long long col0, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    if (!FOLD) {
      const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = scaled(a, s[n][e]);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
        s[n][e] = x;
      }
    }
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  if (FOLD) {  // max of the scaled scores = scale * max of the products (positive power of two: exact)
    x0 *= a.scale;
    x1 *= a.scale;
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  // A finite additive mask may use -FLT_MAX. In a leading fully masked row/tile,
  // multiplying that maximum by log2(e) overflows even though score - maximum
  // is finite. Subtract first only in that exceptional case; ordinary online
  // softmax retains its existing fused arithmetic and rounding.
  const bool shift0 = isinf(ml0) && isfinite(n0), shift1 = isinf(ml1) && isfinite(n1);
  const float c0 = m0 == -INFINITY ? 0.f : exp2f(shift0 ? (m0 - n0) * L2E : fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : exp2f(shift1 ? (m1 - n1) * L2E : fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = FOLD ? sc2<true>(a) : L2E;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = exp2f(shift0 ? ((FOLD ? scaled(a, s[n][0]) : s[n][0]) - n0) * L2E : fmaf(s[n][0], k, -ml0));
    s[n][1] = exp2f(shift0 ? ((FOLD ? scaled(a, s[n][1]) : s[n][1]) - n0) * L2E : fmaf(s[n][1], k, -ml0));
    s[n][2] = exp2f(shift1 ? ((FOLD ? scaled(a, s[n][2]) : s[n][2]) - n1) * L2E : fmaf(s[n][2], k, -ml1));
    s[n][3] = exp2f(shift1 ? ((FOLD ? scaled(a, s[n][3]) : s[n][3]) - n1) * L2E : fmaf(s[n][3], k, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
    o[n][0] *= c0;
    o[n][1] *= c0;
    o[n][2] *= c1;
    o[n][3] *= c1;
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

template <bool MASKED, bool POW2>
PFLATLONG_DEV void ds_rows(const Args& a, float (&s)[8][4], const float (&dp)[8][4], const float* mlo, const float* mhi,
                        long long col0, float s_lo, float s_hi, float d_lo, float d_hi, int lane) {
  constexpr bool FOLD = POW2 && !MASKED;
  const float k = FOLD ? sc2<true>(a) : L2E;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const long long col = col0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float x = s[n][e];
      if (!FOLD) {
        x = scaled(a, x);
        if (MASKED) x += __ldg((e < 2 ? mlo : mhi) + (col + (e & 1)) * a.m_j);
      }
      const float p = exp2f(fmaf(x, k, -(e < 2 ? s_lo : s_hi)));
      s[n][e] = grad_scale(a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
    }
  }
}

template <int NS, bool POW2>
PFLATLONG_DEV void forward_item(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  if (tid == 0) { add_count(a.work, min(64, t-row0)); add_count(a.work+1, 1); }
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_tail(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid, a.t - row0);
  cp_commit();
  cp_wait0();
  wg_bar(wg);
  const unsigned long long mine = low_bits(qb64 + 1);

  if (lane == 0) {
    flags[w4] = mine;

  }
  uint32_t qa[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  wg_bar(wg);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  P9AX(3, P9NOW());
  P9AX(4, (unsigned long long)__popcll(group));
  const uint64_t ring_desc = sw_desc(ring, 1, 64);  // K-major and transposed descriptors share these fields
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_tail(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid, a.t - tile * BT);
      load_sw_tail(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid, a.t - tile * BT);
    }
  };
  float o[8][4], sc[8][4], sn[8][4];
  zero8(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  cp_wait1();
  proxy_fence();
  wg_bar(wg);
  if (cur < 64) {
    wg_fence();
    wg_nt6(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
    pin(qa);
  }
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait0();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const int ns = stage == 2 ? 0 : stage + 1;
    if (nxt < 64) {
      wg_fence();
      wg_nt6(sn, qa, desc_add(ring_desc, ns * ST4));
      wg_commit();
    }
    causal_scores(sc, a.t, r_lo, r_hi, cur * BT, lane, false);
    softmax_tile<false, false>(a, sc, m0, m1, l0, l1, o, nullptr, nullptr, 0, lane);
    if (tid == 0) add_count(a.work + 2, 1);
    uint32_t pa[4][NS][4];
    to_a<NS, 4, 8>(pa, sc);
    wg_fence();
    wg_pv6<NS>(o, pa, desc_add(ring_desc, stage * ST4 + SWT));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    if (nxt < 64) {
#pragma unroll
      for (int n = 0; n < 8; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[n][e] = sn[n][e];
    }
    cur = nxt;
    nxt = after;
    stage = ns;
  }
  cp_wait0();
  P9AX(5, P9NOW());
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;  // no allowed key: o is 0
  if (empty0) l0 = 1.f;
  if (empty1) l1 = 1.f;
  __nv_bfloat16* og = a.o + head_offset(a.lo, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    const float y0 = o[n][0] / l0, y1 = o[n][1] / l0, y2 = o[n][2] / l1, y3 = o[n][3] / l1;
    if (r_lo < a.t) store_bf16x2(og + (long long)r_lo * a.lo.sr + d, y0, y1);
    if (r_hi < a.t) store_bf16x2(og + (long long)r_hi * a.lo.sr + d, y2, y3);
  }
  if ((lane & 3) == 0) {
    if (r_lo < a.t) a.lse2[r_lo] = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    if (r_hi < a.t) a.lse2[r_hi] = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void dq_item(const Args& a, int bh, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(ring + 3 * ST4);
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* ds = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, qt = qb64;
  if (tid == 0) { add_count(a.work, min(64, t-row0)); add_count(a.work+1, 1); }
  const __nv_bfloat16* kg = a.k + head_offset(a.lk, bh, a.heads);
  const __nv_bfloat16* vg = a.v + head_offset(a.lv, bh, a.heads);
  load_rows_tail(qs, a.q + head_offset(a.lq, bh, a.heads) + (long long)row0 * a.lq.sr, a.lq.sr, tid, a.t - row0);
  load_rows_tail(ds, a.dout + head_offset(a.ld, bh, a.heads) + (long long)row0 * a.ld.sr, a.ld.sr, tid, a.t - row0);
  cp_commit();
  const unsigned long long mine = low_bits(qb64 + 1);

  if (lane == 0) flags[w4] = mine;
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1;
    const __nv_bfloat16* dp = a.dout + head_offset(a.ld, bh, a.heads) + (long long)r * a.ld.sr + 32 * h;
    float acc = 0.f;
    const __nv_bfloat16* op = a.o + head_offset(a.lo, bh, a.heads) + (long long)r * a.lo.sr + 32 * h;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = r < a.t ? __ldcg(reinterpret_cast<const uint4*>(dp) + c) : make_uint4(0,0,0,0);
      const uint4 z = r < a.t ? __ldcg(reinterpret_cast<const uint4*>(op) + c) : make_uint4(0,0,0,0);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }

    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    if (h == 0 && r < a.t) a.delta[r] = dl;
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float s_lo = r_lo < a.t ? a.lse2[r_lo] : INFINITY, s_hi = r_hi < a.t ? a.lse2[r_hi] : INFINITY;
  cp_wait0();
  wg_bar(wg);
  uint32_t qa[4][4], da[4][4];
  a_frags(qa, qs, 16 * w4, lane);
  a_frags(da, ds, 16 * w4, lane);
  const unsigned long long group = flags[0] | flags[1] | flags[2] | flags[3];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  auto issue = [&](int tile, int stage) {
    if (tile < 64) {
      load_sw_tail(ring + stage * ST4, kg + (long long)(tile * BT) * a.lk.sr, a.lk.sr, tid, a.t - tile * BT);
      load_sw_tail(ring + stage * ST4 + SWT, vg + (long long)(tile * BT) * a.lv.sr, a.lv.sr, tid, a.t - tile * BT);
    }
  };
  float dq[8][4], s[8][4], dp[8][4];
  zero8(dq);
  int cur = next_bit(group, 0), nxt = next_bit(group, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(group, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t kd = desc_add(ring_desc, stage * ST4), vd = desc_add(kd, SWT);
    wg_fence();
    wg_nt6(s, qa, kd);
    wg_nt6(dp, da, vd);
    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(qa);
    pin(da);
    causal_scores(s, a.t, r_lo, r_hi, cur * BT, lane, false);
    ds_rows<false, false>(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    if (tid == 0) add_count(a.work + 2, 1);
    uint32_t sa[4][NS][4];
    to_a<NS, 4, 8>(sa, s);
    wg_fence();
    wg_pv6<NS>(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* qg = a.dq + head_offset(a.ldq, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    if (r_lo < a.t) store_bf16x2(qg + (long long)r_lo * a.ldq.sr + d, dq[n][0], dq[n][1]);
    if (r_hi < a.t) store_bf16x2(qg + (long long)r_hi * a.ldq.sr + d, dq[n][2], dq[n][3]);
  }
  wg_bar(wg);
}

template <int NS, bool POW2>
PFLATLONG_DEV void kv_item(const Args& a, int bh, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + 3 * ST4);
  unsigned long long* flags = reinterpret_cast<unsigned long long*>(sv + 3 * 2 * BT);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + ST4);
  __nv_bfloat16* vs = reinterpret_cast<__nv_bfloat16*>(ring + 2 * ST4);
  const int t = a.t, key0 = kb64 * 64, kt = kb64;
  if (tid == 0) { add_count(a.work+3, min(64,t-key0)); add_count(a.work+4,1); }
  const __nv_bfloat16* qg = a.q + head_offset(a.lq, bh, a.heads);
  const __nv_bfloat16* dg = a.dout + head_offset(a.ld, bh, a.heads);
  load_rows_tail(ks, a.k + head_offset(a.lk, bh, a.heads) + (long long)key0 * a.lk.sr, a.lk.sr, tid, a.t - key0);
  load_rows_tail(vs, a.v + head_offset(a.lv, bh, a.heads) + (long long)key0 * a.lv.sr, a.lv.sr, tid, a.t - key0);
  cp_commit();
  if (w4 == 0) {
    const unsigned long long lv = low_bits((a.t+63)/64) & ~low_bits(kb64);
    const unsigned long long mx = 0;
    if (lane == 0) {
      flags[0] = lv;
      flags[1] = mx;
    }
  }
  cp_wait0();
  wg_bar(wg);
  uint32_t ka[4][4], va[4][4];
  a_frags(ka, ks, 16 * w4, lane);
  a_frags(va, vs, 16 * w4, lane);
  const unsigned long long mine = flags[0], mixed = flags[1];
  wg_bar(wg);
  const uint64_t ring_desc = sw_desc(ring, 1, 64);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[8][4], dv[8][4], s[8][4], dp[8][4];
  zero8(dk);
  zero8(dv);
  auto issue = [&](int qt, int stg) {
    if (qt >= 64) return;
    load_sw_tail(ring + stg * ST4, qg + (long long)(qt * BT) * a.lq.sr, a.lq.sr, tid, a.t - qt * BT);
    load_sw_tail(ring + stg * ST4 + SWT, dg + (long long)(qt * BT) * a.ld.sr, a.ld.sr, tid, a.t - qt * BT);
    float* d = sv + stg * 2 * BT;
    if (tid < BT) {
      const int row = qt * BT + tid;
      d[tid] = row < a.t ? a.lse2[row] : INFINITY;
      d[BT + tid] = row < a.t ? a.delta[row] : 0.f;
    }
  };
  int cur = next_bit(mine, 0), nxt = next_bit(mine, cur + 1);
  issue(cur, 0);
  cp_commit();
  issue(nxt, 1);
  cp_commit();
  int stage = 0;
  while (cur < 64) {
    const int after = next_bit(mine, nxt + 1);
    cp_wait1();
    proxy_fence();
    wg_bar(wg);
    issue(after, (stage + 2) % 3);
    cp_commit();
    const uint64_t qd = desc_add(ring_desc, stage * ST4), dd = desc_add(qd, SWT);
    const float* sl = sv + stage * 2 * BT;
    wg_fence();
    wg_nt6(s, ka, qd);
    wg_nt6(dp, va, dd);

    wg_commit();
    wg_wait<0>();
    pin(s);
    pin(dp);
    pin(ka);
    pin(va);
    {
      // Branch-free body: one loop for masked and unmasked tiles (selects, predicated mask loads). Every variant that
      // branched once per tile around code updating both wgmma accumulators produced wrong dK/dV and clobbered dQ
      // (gpu-10..13, kv_tile6 and a per-tile-branch inline body); this form matches v5 bit for bit when POW2 is false.
      const float* dl = sl + BT;
      const bool mx = (mixed >> cur) & 1ull;
      const bool fold = POW2 && !mx;
      const float kf = fold ? sc2<true>(a) : L2E;
#pragma unroll
      for (int n = 0; n < 8; ++n) {
        const int j = 8 * n + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = fold ? s[n][e] : scaled(a, s[n][e]);
          const int query_row = cur * BT + jj, key_row = e < 2 ? k_lo : k_hi;
          if (query_row >= a.t || key_row >= a.t || key_row > query_row) x = -INFINITY;
          const float p = exp2f(fmaf(x, kf, -sl[jj]));
          s[n][e] = p;
          dp[n][e] = grad_scale(a, p * (dp[n][e] - dl[jj]));
        }
      }
    }
    if (tid == 0) add_count(a.work + 5, 1);
    uint32_t pa[4][NS][4], sa[4][NS][4];
    to_a<NS, 4, 8>(pa, s);
    to_a<NS, 4, 8>(sa, dp);
    wg_fence();
    wg_pv6<NS>(dv, pa, dd);
    wg_pv6<NS>(dk, sa, qd);

    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    stage = stage == 2 ? 0 : stage + 1;
    cur = nxt;
    nxt = after;
  }
  cp_wait0();
  __nv_bfloat16* kgo = a.dk + head_offset(a.ldk, bh, a.heads);
  __nv_bfloat16* vgo = a.dv + head_offset(a.ldv, bh, a.heads);
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int d = 8 * n + 2 * (lane & 3);
    if (k_lo < a.t) store_bf16x2(kgo + (long long)k_lo * a.ldk.sr + d, dk[n][0], dk[n][1]);
    if (k_hi < a.t) store_bf16x2(kgo + (long long)k_hi * a.ldk.sr + d, dk[n][2], dk[n][3]);
    if (k_lo < a.t) store_bf16x2(vgo + (long long)k_lo * a.ldv.sr + d, dv[n][0], dv[n][1]);
    if (k_hi < a.t) store_bf16x2(vgo + (long long)k_hi * a.ldv.sr + d, dv[n][2], dv[n][3]);
  }
  wg_bar(wg);
}

// =====================================================================================================================
// Width-32 bodies. A head of width 30 (for example) travels in a 32-wide packed head (two exact zero
// channels). The width-64 items above zero-pad that head again to 64, doubling every tensor-core product and every
// operand byte. These items keep the same packed-segment arithmetic: the FP32 softmax / score-gradient recipe below is
// operation for operation that of softmax_tile<false,false> / ds_rows<false,false> / the kv body, BF16 WGMMA operands,
// FP32 accumulation from +0 and exp2f. They contract over 32 channels instead: Q K^T, dO V^T, K Q^T and V dO^T issue
// two k16 steps of m64n64k16 against 64-byte-swizzled K-major tiles; P V, dS K, P^T dO and dS^T Q issue m64n32k16
// against the same tiles read MN-major. In the width-64 body channels 32..63 only add exact zero products, so
// agreement with it is expected bit for bit; that agreement is measured by the probe, never assumed.
// A ring stage is two 64 x 32 BF16 tiles (8 KB), so each warpgroup keeps five stages and loads three tiles ahead.
// VAR 0 keeps the v7 per-tile schedule (score products of the next tile overlap only the current FP32 work).
// VAR 1 also leaves each tile's output/gradient products in flight across the next tile's FP32 work
// (FlashAttention-3 style intra-warpgroup pipelining): two WGMMA groups per tile, each waited exactly once.
// Diagonal-only causal masking (forward/dQ) and asynchronous statistics staging (dK/dV) change no stored value: only
// rows past the segment end, whose outputs are never stored, see different intermediate scores.
// Every lambda of the width-32 bodies is force-inlined. Large full-model units otherwise keep some as separate
// functions: there the scale mode no longer folds (division slow-path calls) and ptxas fences or serializes the
// warpgroup MMAs (C7517/C7520) and spills - observed for one dK/dV copy of a full build, never in the probes.
#define LATCH_SEG_INLINE __attribute__((always_inline))
constexpr int W32 = 32;
constexpr int LD32 = 40;          // row-major A-operand staging stride in BF16 (80 B rows): conflict-free ldmatrix
constexpr int SWT32 = 64 * 64;    // bytes of one 64-row x 32-BF16 tile with 64-byte swizzle
constexpr int ST32 = 2 * SWT32;   // one ring stage: two tiles
constexpr int NST32 = 5;          // ring stages; row-major staging borrows stages 3..4 before their first fill
static_assert(1024 + NST32 * ST32 + NST32 * 2 * BT * 4 <= WGB, "width-32 ring and statistics exceed the region");
static_assert(2 * 64 * LD32 * 2 <= 2 * ST32, "row-major staging exceeds the borrowed ring stages");

// Segmented attention always lowers scale_mode 1 (x * scale). The shared scaled()/grad_scale() also carry a
// scale_mode 2 division; where a compile context fails to fold the mode, that dead path stays as calls to the
// division slow-path subroutine, which serialize every warpgroup MMA of the function (C7509/C7520) and force spills
// (observed in one full-model unit). The width-32 bodies multiply directly: the value of the mode-1 path, no division.
PFLATLONG_DEV float sc32(const Args& a, float x) { return x * a.scale; }
// VAR 6 exponential policy: the bare MUFU ex2 (ex2.approx.ftz), the exponential Latch's audited dense v7 items use
// under PFLATLONG_FTZ. It equals exp2f for every argument except [-150, -126), where exp2f returns a subnormal
// probability (< 2^-126) and ex2.approx.ftz returns 0. VAR 5 and earlier keep exp2f (bitwise ns05).
template <bool FTZ>
PFLATLONG_DEV float seg_ex2(float x) {
  if constexpr (FTZ) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
  } else {
    return exp2f(x);
  }
}
// ds_rows<false, false> with the direct product (k = log2(e): no folded scale, no mask table).
template <bool FTZ = false>
PFLATLONG_DEV void ds_rows32(const Args& a, float (&s)[8][4], const float (&dp)[8][4], const float*, const float*,
                             long long, float s_lo, float s_hi, float d_lo, float d_hi, int) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float x = sc32(a, s[n][e]);
      const float p = seg_ex2<FTZ>(fmaf(x, L2E, -(e < 2 ? s_lo : s_hi)));
      s[n][e] = sc32(a, p * (dp[n][e] - (e < 2 ? d_lo : d_hi)));
    }
}
template <int N>
PFLATLONG_DEV void cp_wait_n() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
// ptxas serializes every warpgroup MMA of a function that contains an external call (C7509) or divergent code it must
// fence (C7518). The width-32 items therefore avoid both inside their item loop: counters are one predicated global
// reduction (atomicAdd on a generic 64-bit pointer emits a shared/global test and a CAS spin loop), quotients use
// the inline correctly rounded sequence below (x / y calls the division slow-path subroutine), masks are selects.
// Executed-work counting is unchanged: the same increments at the same executed items and tiles.
PFLATLONG_DEV void count_if(bool pred, int64_t* p, unsigned long long value) {
  asm volatile("{\n.reg .pred q;\n.reg .u64 g;\nsetp.ne.b32 q, %2, 0;\ncvta.to.global.u64 g, %0;\n"
               "@q red.relaxed.gpu.global.add.u64 [g], %1;\n}\n" :: "l"(p), "l"(value), "r"((int)pred) : "memory");
}
// IEEE round-to-nearest x / y for a row sum y >= 1 (the row maximum contributes exp2(0) = 1, an empty row uses 1):
// the reciprocal-refinement quotient of CUDA's own div.rn fast path, which is the correctly rounded quotient whenever
// its range check passes; that holds for finite y in [1, 2^24] and |x| >= 2^-100. Smaller |x| is scaled by 2^64
// (exact) and the quotient scaled back (exact unless the quotient is subnormal); zero keeps its sign as x / y does.
// Agreement with the width-64 body's x / y is measured bit for bit by the probes.
PFLATLONG_DEV float div_row(float x, float y) {
  const bool tiny = fabsf(x) < 0x1p-100f;
  const float xs = tiny ? x * 0x1p64f : x;
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y));
  r = fmaf(fmaf(-y, r, 1.f), r, r);
  const float q0 = xs * r;
  const float q = fmaf(fmaf(-y, q0, xs), r, q0);
  return x == 0.f ? x : (tiny ? q * 0x1p-64f : q);
}
// Predicated global stores (predicate inside PTX, no branch). The BF16 pair conversion is store_bf16x2's.
PFLATLONG_DEV void st_bf16x2_if(bool pred, __nv_bfloat16* ptr, float x0, float x1) {
  const uint32_t v = pack(x0, x1);
  asm volatile("{\n.reg .pred q;\n.reg .u64 g;\nsetp.ne.b32 q, %2, 0;\ncvta.to.global.u64 g, %0;\n"
               "@q st.global.b32 [g], %1;\n}\n" :: "l"(ptr), "r"(v), "r"((int)pred) : "memory");
}
PFLATLONG_DEV void st_f32_if(bool pred, float* ptr, float x) {
  asm volatile("{\n.reg .pred q;\n.reg .u64 g;\nsetp.ne.b32 q, %2, 0;\ncvta.to.global.u64 g, %0;\n"
               "@q st.global.f32 [g], %1;\n}\n" :: "l"(ptr), "f"(x), "r"((int)pred) : "memory");
}
// Diagonal causal mask as selects. Off-diagonal key tiles of a query tile hold only keys < row0 < length, and on the
// diagonal tile key > row covers every key past the segment end for every stored row, so stored rows see exactly
// causal_scores' result; rows past the segment end (never stored, their statistics are +inf) may differ.
PFLATLONG_DEV void diag_mask(float (&s)[8][4], bool diag, int r_lo, int r_hi, int key0, int lane) {
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    const int key = key0 + 8 * n + 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int row = e < 2 ? r_lo : r_hi, col = key + (e & 1);
      s[n][e] = diag && col > row ? -INFINITY : s[n][e];
    }
  }
}
// 64-byte swizzle (Swizzle<2,4,3>): the 16-byte chunk index of a 64-byte row is XORed with bits 1..2 of the row.
PFLATLONG_DEV int sw64(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }
PFLATLONG_DEV void load_sw32_tail(char* dst, const __nv_bfloat16* src, long long stride, int tid, int valid_rows) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = tid + k * 128, r = idx >> 2, c = idx & 3;
    cp16_tail(dst + sw64(r, c), src + (long long)(r < valid_rows ? r : 0) * stride + 8 * c, r < valid_rows);
  }
}
PFLATLONG_DEV void load_rows32_tail(__nv_bfloat16* dst, const __nv_bfloat16* src, long long stride, int tid,
                                    int valid_rows) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int idx = tid + k * 128, r = idx >> 2, c = idx & 3;
    cp16_tail(dst + r * LD32 + 8 * c, src + (long long)(r < valid_rows ? r : 0) * stride + 8 * c, r < valid_rows);
  }
}
// Four-byte asynchronous copy with zero fill (statistics rows of a tile; their packed origin is not 16-B aligned).
PFLATLONG_DEV void cp4_tail(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::
      "r"(smem_u32(dst)), "l"(src), "r"(valid ? 4 : 0));
}
// A fragments (two k16 steps over 32 channels) of shared rows [row0, row0 + 16).
PFLATLONG_DEV void a_frags32(uint32_t (&a)[2][4], const __nv_bfloat16* s, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
    ldsm_x4(a[ks], s + (row0 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD32 + 16 * ks + (lane >> 4) * 8);
}
// 64-byte swizzle descriptor. SBO = 512 B between 8-row core groups (rows of 64 B) for both readings: K-major
// (rows = n, k16 step ks starts 32 B further along the row) and MN-major (rows = k, k16 step kk starts 1024 B
// further). The strided-dimension extent is exactly one 32-element swizzle atom, so LBO is unused.
PFLATLONG_DEV uint64_t sw32_desc(const void* p) {
  uint64_t d = (uint64_t)((smem_u32(p) & 0x3FFFF) >> 4);
  d |= (uint64_t)1 << 16;
  d |= (uint64_t)32 << 32;
  d |= (uint64_t)2 << 62;
  return d;
}
// LATCH-COLD-PREP renewal (see step17_intervals._RENEW_A): issue loop-invariant register-A products from a copy
// (fragment XOR a runtime zero ptxas cannot fold), so a small compile context cannot recycle those registers.
PFLATLONG_DEV void renew32(uint32_t (&f)[2][4], const uint32_t (&a)[2][4]) {
  unsigned lane, zero;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(zero) : "r"(lane));
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) f[k][e] = a[k][e] ^ zero;
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}
template <int TNSP>
PFLATLONG_DEV void wgmma32(float (&d)[4][4], const uint32_t (&a)[4], uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %20, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %21, p, 1, 1, "
      "%22;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(scale_d), "l"(b), "n"(TNSP));
}
// acc (64 x 64) = A (64 x 32, registers) * B where B[k][n] = tile[n][k] (K-major 64-byte-swizzled tile).
PFLATLONG_DEV void wg_nt32(float (&acc)[8][4], const uint32_t (&a_in)[2][4], uint64_t d0) {
  uint32_t a[2][4];
  renew32(a, a_in);
  wgmma64<0>(acc, a[0], d0, 0);
  wgmma64<0>(acc, a[1], desc_add(d0, 32), 1);
}
// acc (64 x 32) += P (64 x 64 keys/queries, registers) * B where B[k][n] = tile[k][n] (the tile read MN-major).
PFLATLONG_DEV void wg_pv32(float (&acc)[4][4], const uint32_t (&p)[4][1][4], uint64_t d0) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) wgmma32<1>(acc, p[kk][0], desc_add(d0, 1024 * kk), 1);
}
using pflatlong::pin;  // the width-32 overloads below extend, never hide, the shared register pins
PFLATLONG_DEV void pin(float (&x)[4][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+f"(x[n][e])::"memory");
}
PFLATLONG_DEV void pin(uint32_t (&x)[2][4]) {
#pragma unroll
  for (int k = 0; k < 2; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) asm volatile("" : "+r"(x[k][e])::"memory");
}
PFLATLONG_DEV void zero4(float (&x)[4][4]) {
#pragma unroll
  for (int n = 0; n < 4; ++n) x[n][0] = x[n][1] = x[n][2] = x[n][3] = 0.f;
}
PFLATLONG_DEV void copy8(float (&d)[8][4], const float (&s)[8][4]) {
#pragma unroll
  for (int n = 0; n < 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) d[n][e] = s[n][e];
}
// softmax_tile<false, false> without the output rescale: identical scaled scores, row maxima, correction factors,
// exponentials and row sums (same operations in the same order). The caller applies o *= c afterwards.
PFLATLONG_DEV void stats32(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                           float& c0, float& c1) {
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = sc32(a, s[n][e]);
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  const bool shift0 = isinf(ml0) && isfinite(n0), shift1 = isinf(ml1) && isfinite(n1);
  c0 = m0 == -INFINITY ? 0.f : exp2f(shift0 ? (m0 - n0) * L2E : fmaf(m0, L2E, -ml0));
  c1 = m1 == -INFINITY ? 0.f : exp2f(shift1 ? (m1 - n1) * L2E : fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = L2E;
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = exp2f(shift0 ? (s[n][0] - n0) * L2E : fmaf(s[n][0], k, -ml0));
    s[n][1] = exp2f(shift0 ? (s[n][1] - n0) * L2E : fmaf(s[n][1], k, -ml0));
    s[n][2] = exp2f(shift1 ? (s[n][2] - n1) * L2E : fmaf(s[n][2], k, -ml1));
    s[n][3] = exp2f(shift1 ? (s[n][3] - n1) * L2E : fmaf(s[n][3], k, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}
PFLATLONG_DEV void rescale32(float (&o)[4][4], float c0, float c1) {
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    o[n][0] *= c0;
    o[n][1] *= c0;
    o[n][2] *= c1;
    o[n][3] *= c1;
  }
}

template <int VAR>
PFLATLONG_DEV void forward32(const Args& a, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, n = qb64 + 1;
  const long long sr = a.lq.sr;
  count_if(tid == 0, a.work, min(64, t-row0));
  count_if(tid == 0, a.work + 1, 1);
  load_rows32_tail(qs, a.q + (long long)row0 * sr, sr, tid, t - row0);
  cp_commit();
  // One cp.async group per call. Past the last tile the copy is a zero fill of stage (tile % NST32), which last held
  // tile tile - NST32 <= j - 2 (complete); the source stays inside the last tile. No branch, uniform group count.
  auto issue = [&](int tile) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + (tile % NST32) * ST32;
    load_sw32_tail(st, a.k + (long long)(tl * BT) * sr, sr, tid, valid);
    load_sw32_tail(st + SWT32, a.v + (long long)(tl * BT) * sr, sr, tid, valid);
    cp_commit();
  };
  issue(0);
  issue(1);
  issue(2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qa[2][4];
  a_frags32(qa, qs, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);  // tile 0 is visible to the tensor cores and every warp holds its Q fragments (stage 3 may refill)
  const uint64_t ring_desc = sw32_desc(ring);
  float o[4][4], sc[8][4], sn[8][4];
  uint32_t pa[4][1][4];
  zero4(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  // Score products of tile j + 1 are issued during tile j. Past the last tile they repeat tile j (discarded), so
  // every warpgroup MMA sits outside data-dependent control flow.
  if (VAR == 0) {
    wg_fence();
    wg_nt32(sc, qa, ring_desc);
    wg_commit();
    wg_wait<0>();
    pin(sc);
  } else {
    wg_fence();
    wg_nt32(sn, qa, ring_desc);
    wg_commit();
    wg_commit();  // empty group standing for the (absent) output products of tile -1
  }
  pin(qa);
  for (int j = 0; j < n; ++j) {
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage (j + 3) % NST32 last held tile j - 2, whose products are complete
    issue(j + 3);
    const uint64_t next = desc_add(ring_desc, ((j + 1 < n ? j + 1 : j) % NST32) * ST32);
    const uint64_t vdesc = desc_add(ring_desc, (j % NST32) * ST32 + SWT32);
    float c0, c1;
    if (VAR == 0) {
      wg_fence();
      wg_nt32(sn, qa, next);
      wg_commit();
      diag_mask(sc, j == qb64, r_lo, r_hi, j * BT, lane);
      stats32(a, sc, m0, m1, l0, l1, c0, c1);
      rescale32(o, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(pa, sc);
      wg_fence();
      wg_pv32(o, pa, vdesc);
      wg_commit();
      wg_wait<0>();
      pin(o);
      pin(sn);
      pin(pa);
      pin(qa);
      copy8(sc, sn);
    } else {
      wg_wait<1>();  // pending [S(j), PV(j-1)] -> S(j) complete
      pin(sn);
      copy8(sc, sn);
      wg_fence();
      wg_nt32(sn, qa, next);
      wg_commit();   // pending [PV(j-1), S(j+1)]
      pin(qa);
      diag_mask(sc, j == qb64, r_lo, r_hi, j * BT, lane);
      stats32(a, sc, m0, m1, l0, l1, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      wg_wait<1>();  // PV(j-1) complete: o and pa may be touched
      pin(o);
      pin(pa);
      rescale32(o, c0, c1);
      to_a<1, 4, 8>(pa, sc);
      wg_fence();
      wg_pv32(o, pa, vdesc);
      wg_commit();   // pending [S(j+1), PV(j)]
    }
  }
  if (VAR != 0) {
    wg_wait<0>();
    pin(o);
    pin(pa);
    pin(sn);
  }
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;  // no allowed key: o is 0
  if (empty0) l0 = 1.f;
  if (empty1) l1 = 1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    const float y0 = div_row(o[c][0], l0), y1 = div_row(o[c][1], l0);
    const float y2 = div_row(o[c][2], l1), y3 = div_row(o[c][3], l1);
    st_bf16x2_if(r_lo < t, a.o + (long long)r_lo * a.lo.sr + d, y0, y1);
    st_bf16x2_if(r_hi < t, a.o + (long long)r_hi * a.lo.sr + d, y2, y3);
  }
  const float lse_lo = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
  const float lse_hi = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  st_f32_if((lane & 3) == 0 && r_lo < t, a.lse2 + r_lo, lse_lo);
  st_f32_if((lane & 3) == 0 && r_hi < t, a.lse2 + r_hi, lse_hi);
  wg_bar(wg);
}

template <int VAR>
PFLATLONG_DEV void dq32(const Args& a, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  __nv_bfloat16* ds = qs + 64 * LD32;
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, n = qb64 + 1;
  const long long sr = a.lq.sr;
  count_if(tid == 0, a.work, min(64, t-row0));
  count_if(tid == 0, a.work + 1, 1);
  load_rows32_tail(qs, a.q + (long long)row0 * sr, sr, tid, t - row0);
  load_rows32_tail(ds, a.dout + (long long)row0 * sr, sr, tid, t - row0);
  cp_commit();
  auto issue = [&](int tile) LATCH_SEG_INLINE {  // branch-free as in forward32
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + (tile % NST32) * ST32;
    load_sw32_tail(st, a.k + (long long)(tl * BT) * sr, sr, tid, valid);
    load_sw32_tail(st + SWT32, a.v + (long long)(tl * BT) * sr, sr, tid, valid);
    cp_commit();
  };
  issue(0);
  issue(1);
  issue(2);
  float dl;
  {
    // Row r = wrow + lane / 2: the even lane sums its 32 channels in the width-64 body's order, the odd lane holds
    // the zero contribution of channels 32..63 (exactly +0, as there), and the pair is combined as there.
    // Every lane loads a valid row (clamped) without branching; only the owning even lane keeps its sum.
    const int r = wrow + (lane >> 1), h = lane & 1, rc = r < t ? r : 0;
    const bool own = h == 0 && r < t;
    const __nv_bfloat16* dp = a.dout + (long long)rc * sr;
    const __nv_bfloat16* op = a.o + (long long)rc * sr;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
    acc = own ? acc : 0.f;
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    st_f32_if(own, a.delta + r, dl);
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float l2lo = a.lse2[r_lo < t ? r_lo : 0], l2hi = a.lse2[r_hi < t ? r_hi : 0];
  const float s_lo = r_lo < t ? l2lo : INFINITY, s_hi = r_hi < t ? l2hi : INFINITY;
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qa[2][4], da[2][4];
  a_frags32(qa, qs, 16 * w4, lane);
  a_frags32(da, ds, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  float dq[4][4], s[8][4], dp[8][4], sn[8][4], dn[8][4];
  uint32_t sa[4][1][4];
  zero4(dq);
  if (VAR != 0) {
    wg_fence();
    wg_nt32(sn, qa, ring_desc);
    wg_nt32(dn, da, desc_add(ring_desc, SWT32));
    wg_commit();
    wg_commit();  // empty group standing for the (absent) dQ products of tile -1
    pin(qa);
    pin(da);
  }
  for (int j = 0; j < n; ++j) {
    const uint64_t kd = desc_add(ring_desc, (j % NST32) * ST32);
    if (VAR == 0) {
      cp_wait_n<2>();
      proxy_fence();
      wg_bar(wg);  // tile j visible; stage (j + 3) % NST32 last held tile j - 2
      issue(j + 3);
      wg_fence();
      wg_nt32(s, qa, kd);
      wg_nt32(dp, da, desc_add(kd, SWT32));
      wg_commit();
      wg_wait<0>();
      pin(s);
      pin(dp);
      pin(qa);
      pin(da);
    } else {
      cp_wait_n<1>();
      proxy_fence();
      wg_bar(wg);  // tile j + 1 visible; stage (j + 3) % NST32 last held tile j - 2, its dQ products complete
      issue(j + 3);
      wg_wait<1>();  // pending [N(j), dQ(j-1)] -> N(j) complete
      pin(sn);
      pin(dn);
      copy8(s, sn);
      copy8(dp, dn);
      const uint64_t nd = desc_add(ring_desc, ((j + 1 < n ? j + 1 : j) % NST32) * ST32);
      wg_fence();
      wg_nt32(sn, qa, nd);
      wg_nt32(dn, da, desc_add(nd, SWT32));
      wg_commit();   // pending [dQ(j-1), N(j+1)]
      pin(qa);
      pin(da);
    }
    diag_mask(s, j == qb64, r_lo, r_hi, j * BT, lane);
    ds_rows32(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    count_if(tid == 0, a.work + 2, 1);
    if (VAR != 0) {
      wg_wait<1>();  // dQ(j-1) complete: its A fragments may be replaced
      pin(sa);
      pin(dq);
    }
    to_a<1, 4, 8>(sa, s);
    wg_fence();
    wg_pv32(dq, sa, kd);
    wg_commit();
    if (VAR == 0) {
      wg_wait<0>();
      pin(dq);
      pin(sa);
    }
  }
  if (VAR != 0) {
    wg_wait<0>();
    pin(dq);
    pin(sa);
    pin(sn);
    pin(dn);
  }
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(r_lo < t, a.dq + (long long)r_lo * a.ldq.sr + d, dq[c][0], dq[c][1]);
    st_bf16x2_if(r_hi < t, a.dq + (long long)r_hi * a.ldq.sr + d, dq[c][2], dq[c][3]);
  }
  wg_bar(wg);
}

template <int VAR>
PFLATLONG_DEV void kv32(const Args& a, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + NST32 * ST32);  // per stage: lse2[64], delta[64]
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  __nv_bfloat16* vs = ks + 64 * LD32;
  const int t = a.t, key0 = kb64 * 64, n = (t + 63) / 64 - kb64;  // query tiles kb64 .. last
  const long long sr = a.lq.sr;
  count_if(tid == 0, a.work + 3, min(64, t-key0));
  count_if(tid == 0, a.work + 4, 1);
  load_rows32_tail(ks, a.k + (long long)key0 * sr, sr, tid, t - key0);
  load_rows32_tail(vs, a.v + (long long)key0 * sr, sr, tid, t - key0);
  cp_commit();
  auto issue = [&](int i) LATCH_SEG_INLINE {  // branch-free as in forward32 (past the last tile: zero fill of a completed stage)
    const int il = i < n ? i : n - 1, stg = i % NST32, row = (kb64 + il) * BT, r = tid & 63;
    const int valid = i < n ? t - row : 0;
    char* st = ring + stg * ST32;
    load_sw32_tail(st, a.q + (long long)row * sr, sr, tid, valid);
    load_sw32_tail(st + SWT32, a.dout + (long long)row * sr, sr, tid, valid);
    // lse2 rows then delta rows of the tile. Rows past the segment end stay zero; every such score is masked to
    // -inf before its exponential, as in the width-64 body (which staged +inf / 0 there).
    const float* stat = (tid < 64 ? a.lse2 : a.delta) + row;
    cp4_tail(sv + stg * 2 * BT + tid, stat + (r < valid ? r : 0), r < valid);
    cp_commit();
  };
  issue(0);
  issue(1);
  issue(2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t ka[2][4], va[2][4];
  a_frags32(ka, ks, 16 * w4, lane);
  a_frags32(va, vs, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[4][4], dv[4][4], s[8][4], dp[8][4], sn[8][4], dn[8][4];
  uint32_t pa[4][1][4], sa[4][1][4];
  zero4(dk);
  zero4(dv);
  if (VAR != 0) {
    wg_fence();
    wg_nt32(sn, ka, ring_desc);
    wg_nt32(dn, va, desc_add(ring_desc, SWT32));
    wg_commit();
    wg_commit();
    pin(ka);
    pin(va);
  }
  for (int i = 0; i < n; ++i) {
    const int stg = i % NST32, qt = kb64 + i;
    const uint64_t qd = desc_add(ring_desc, stg * ST32), dd = desc_add(qd, SWT32);
    if (VAR == 0) {
      cp_wait_n<2>();
      proxy_fence();
      wg_bar(wg);
      issue(i + 3);
      wg_fence();
      wg_nt32(s, ka, qd);
      wg_nt32(dp, va, dd);
      wg_commit();
      wg_wait<0>();
      pin(s);
      pin(dp);
      pin(ka);
      pin(va);
    } else {
      cp_wait_n<1>();
      proxy_fence();
      wg_bar(wg);
      issue(i + 3);
      wg_wait<1>();
      pin(sn);
      pin(dn);
      copy8(s, sn);
      copy8(dp, dn);
      const uint64_t nq = desc_add(ring_desc, ((i + 1 < n ? i + 1 : i) % NST32) * ST32);
      wg_fence();
      wg_nt32(sn, ka, nq);
      wg_nt32(dn, va, desc_add(nq, SWT32));
      wg_commit();
      pin(ka);
      pin(va);
    }
    {
      // The width-64 kv body's branch-free element recipe (POW2 false: every tile scales and masks per element).
      const float* sl = sv + stg * 2 * BT;
      const float* dl = sl + BT;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int j = 8 * c + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = sc32(a, s[c][e]);
          const int query_row = qt * BT + jj, key_row = e < 2 ? k_lo : k_hi;
          if (query_row >= t || key_row >= t || key_row > query_row) x = -INFINITY;
          const float p = exp2f(fmaf(x, L2E, -sl[jj]));
          s[c][e] = p;
          dp[c][e] = sc32(a, p * (dp[c][e] - dl[jj]));
        }
      }
    }
    count_if(tid == 0, a.work + 5, 1);
    if (VAR != 0) {
      wg_wait<1>();  // dV/dK(i-1) complete: their A fragments may be replaced
      pin(pa);
      pin(sa);
      pin(dv);
      pin(dk);
    }
    to_a<1, 4, 8>(pa, s);
    to_a<1, 4, 8>(sa, dp);
    wg_fence();
    wg_pv32(dv, pa, dd);
    wg_pv32(dk, sa, qd);
    wg_commit();
    if (VAR == 0) {
      wg_wait<0>();
      pin(dv);
      pin(dk);
      pin(pa);
      pin(sa);
    }
  }
  if (VAR != 0) {
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    pin(sn);
    pin(dn);
  }
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(k_lo < t, a.dk + (long long)k_lo * a.ldk.sr + d, dk[c][0], dk[c][1]);
    st_bf16x2_if(k_hi < t, a.dk + (long long)k_hi * a.ldk.sr + d, dk[c][2], dk[c][3]);
    st_bf16x2_if(k_lo < t, a.dv + (long long)k_lo * a.ldv.sr + d, dv[c][0], dv[c][1]);
    st_bf16x2_if(k_hi < t, a.dv + (long long)k_hi * a.ldv.sr + d, dv[c][2], dv[c][3]);
  }
  wg_bar(wg);
}

// ---------------------------------------------------------------------------------------------------------- VAR 2
// Lean width-32 schedule. The v0/v1 loops spend most issue slots outside the exponential: a diagonal mask evaluated
// on every tile, the softmax overflow alternative (shift) evaluated per element although it can only be selected when
// a running row maximum exceeds FLT_MAX / log2(e), and per-tile address/modulo arithmetic. VAR 2 keeps every value:
// - the diagonal key tile (forward/dQ) and the first/last query tiles (dK/dV) are peeled into masked steps; every
//   other tile provably needs no mask (keys < row0 <= every query row < length; rows/keys inside the segment);
// - the shift alternative is taken per warp when any row of the warp needs it (__any_sync), else the plain
//   fmaf/exp2f path, which is exactly what the per-element select chose for those rows;
// - stage indices advance incrementally and each thread's two copy chunks keep precomputed offsets;
// - dQ and dK/dV issue the next tile's score products before the current tile's FP32 work (as the forward does).
// Each step waits for all of its warpgroup MMAs before the next step: no MMA crosses a step or loop boundary.
template <bool B> struct Flag { static constexpr bool value = B; };

template <bool FTZ = false>
PFLATLONG_DEV void stats32_lean(const Args& a, float (&s)[8][4], float& m0, float& m1, float& l0, float& l1,
                                float& c0, float& c1) {
  float x0 = -INFINITY, x1 = -INFINITY;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
#pragma unroll
    for (int e = 0; e < 4; ++e) s[n][e] = sc32(a, s[n][e]);
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, off));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, off));
  }
  const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
  const float ml0 = n0 == -INFINITY ? 0.f : n0 * L2E, ml1 = n1 == -INFINITY ? 0.f : n1 * L2E;
  const bool shift0 = isinf(ml0) && isfinite(n0), shift1 = isinf(ml1) && isfinite(n1);
  c0 = m0 == -INFINITY ? 0.f : seg_ex2<FTZ>(shift0 ? (m0 - n0) * L2E : fmaf(m0, L2E, -ml0));
  c1 = m1 == -INFINITY ? 0.f : seg_ex2<FTZ>(shift1 ? (m1 - n1) * L2E : fmaf(m1, L2E, -ml1));
  m0 = n0;
  m1 = n1;
  const float k = L2E;
  float sum0 = 0.f, sum1 = 0.f;
  if (__any_sync(0xffffffffu, shift0 || shift1)) {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = seg_ex2<FTZ>(shift0 ? (s[n][0] - n0) * L2E : fmaf(s[n][0], k, -ml0));
      s[n][1] = seg_ex2<FTZ>(shift0 ? (s[n][1] - n0) * L2E : fmaf(s[n][1], k, -ml0));
      s[n][2] = seg_ex2<FTZ>(shift1 ? (s[n][2] - n1) * L2E : fmaf(s[n][2], k, -ml1));
      s[n][3] = seg_ex2<FTZ>(shift1 ? (s[n][3] - n1) * L2E : fmaf(s[n][3], k, -ml1));
      sum0 += s[n][0] + s[n][1];
      sum1 += s[n][2] + s[n][3];
    }
  } else {
#pragma unroll
    for (int n = 0; n < 8; ++n) {
      s[n][0] = seg_ex2<FTZ>(fmaf(s[n][0], k, -ml0));
      s[n][1] = seg_ex2<FTZ>(fmaf(s[n][1], k, -ml0));
      s[n][2] = seg_ex2<FTZ>(fmaf(s[n][2], k, -ml1));
      s[n][3] = seg_ex2<FTZ>(fmaf(s[n][3], k, -ml1));
      sum0 += s[n][0] + s[n][1];
      sum1 += s[n][2] + s[n][3];
    }
  }
  l0 = fmaf(l0, c0, sum0);
  l1 = fmaf(l1, c1, sum1);
}

// One thread's two 16-byte chunks of a 64 x 32 tile (rows r and r + 32, chunk c), precomputed once per item.
struct Chunks32 {
  int row, d0, d1, s0, s1;  // tile row of chunk 0, swizzled destinations, row-major destinations (LD32 staging)
  long long o0, o1, oc;     // source element offsets of chunk 0/1 and of row 0 (clamp target)
  PFLATLONG_DEV Chunks32(int tid, long long sr) {
    row = tid >> 2;
    const int c = tid & 3;
    d0 = sw64(row, c);
    d1 = sw64(row + 32, c);
    s0 = row * LD32 + 8 * c;
    s1 = (row + 32) * LD32 + 8 * c;
    oc = 8 * c;
    o0 = (long long)row * sr + oc;
    o1 = o0 + 32 * sr;
  }
  PFLATLONG_DEV void sw(char* dst, const __nv_bfloat16* tile, int valid) const {
    cp16_tail(dst + d0, tile + (row < valid ? o0 : oc), row < valid);
    cp16_tail(dst + d1, tile + (row + 32 < valid ? o1 : oc), row + 32 < valid);
  }
  PFLATLONG_DEV void rows(__nv_bfloat16* dst, const __nv_bfloat16* tile, int valid) const {
    cp16_tail(dst + s0, tile + (row < valid ? o0 : oc), row < valid);
    cp16_tail(dst + s1, tile + (row + 32 < valid ? o1 : oc), row + 32 < valid);
  }
};
PFLATLONG_DEV int stage_add(int s, int k) { return s + k >= NST32 ? s + k - NST32 : s + k; }

PFLATLONG_DEV void forward32_v2(const Args& a, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, n = qb64 + 1;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t-row0));
  count_if(tid == 0, a.work + 1, 1);
  ch.rows(qs, a.q + (long long)row0 * sr, t - row0);
  cp_commit();
  // Past the last tile: zero fill of a completed stage, source inside the last tile (as forward32).
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qa[2][4];
  a_frags32(qa, qs, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  float o[4][4], sc[8][4], sn[8][4];
  uint32_t pa[4][1][4];
  zero4(o);
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  wg_fence();
  wg_nt32(sc, qa, ring_desc);
  wg_commit();
  wg_wait<0>();
  pin(sc);
  pin(qa);
  int stage = 0;  // stage of tile j
  auto step = [&](int j, auto diag) LATCH_SEG_INLINE {
    constexpr bool DIAG = decltype(diag)::value;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (complete)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    wg_fence();
    wg_nt32(sn, qa, desc_add(ring_desc, (DIAG ? stage : next) * ST32));  // the diagonal is last: repeat, discard
    wg_commit();
    if (DIAG) diag_mask(sc, true, r_lo, r_hi, j * BT, lane);
    float c0, c1;
    stats32_lean(a, sc, m0, m1, l0, l1, c0, c1);
    rescale32(o, c0, c1);
    count_if(tid == 0, a.work + 2, 1);
    to_a<1, 4, 8>(pa, sc);
    wg_fence();
    wg_pv32(o, pa, desc_add(ring_desc, stage * ST32 + SWT32));
    wg_commit();
    wg_wait<0>();
    pin(o);
    pin(sn);
    pin(pa);
    pin(qa);
    copy8(sc, sn);
    stage = next;
  };
  for (int j = 0; j < n - 1; ++j) step(j, Flag<false>{});
  step(n - 1, Flag<true>{});
  cp_wait0();
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;
  if (empty0) l0 = 1.f;
  if (empty1) l1 = 1.f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    const float y0 = div_row(o[c][0], l0), y1 = div_row(o[c][1], l0);
    const float y2 = div_row(o[c][2], l1), y3 = div_row(o[c][3], l1);
    st_bf16x2_if(r_lo < t, a.o + (long long)r_lo * a.lo.sr + d, y0, y1);
    st_bf16x2_if(r_hi < t, a.o + (long long)r_hi * a.lo.sr + d, y2, y3);
  }
  const float lse_lo = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
  const float lse_hi = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
  st_f32_if((lane & 3) == 0 && r_lo < t, a.lse2 + r_lo, lse_lo);
  st_f32_if((lane & 3) == 0 && r_hi < t, a.lse2 + r_hi, lse_hi);
  wg_bar(wg);
}

PFLATLONG_DEV void dq32_v2(const Args& a, int qb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  __nv_bfloat16* ds = qs + 64 * LD32;
  const int t = a.t, row0 = qb64 * 64, wrow = row0 + 16 * w4, n = qb64 + 1;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t-row0));
  count_if(tid == 0, a.work + 1, 1);
  ch.rows(qs, a.q + (long long)row0 * sr, t - row0);
  ch.rows(ds, a.dout + (long long)row0 * sr, t - row0);
  cp_commit();
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  float dl;
  {
    const int r = wrow + (lane >> 1), h = lane & 1, rc = r < t ? r : 0;
    const bool own = h == 0 && r < t;
    const __nv_bfloat16* dp = a.dout + (long long)rc * sr;
    const __nv_bfloat16* op = a.o + (long long)rc * sr;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
    acc = own ? acc : 0.f;
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    dl = h ? other + acc : acc + other;
    st_f32_if(own, a.delta + r, dl);
  }
  const int r_lo = wrow + (lane >> 2), r_hi = r_lo + 8;
  const float d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
  const float d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  const float l2lo = a.lse2[r_lo < t ? r_lo : 0], l2hi = a.lse2[r_hi < t ? r_hi : 0];
  const float s_lo = r_lo < t ? l2lo : INFINITY, s_hi = r_hi < t ? l2hi : INFINITY;
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qa[2][4], da[2][4];
  a_frags32(qa, qs, 16 * w4, lane);
  a_frags32(da, ds, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  float dq[4][4], s[8][4], dp[8][4], sn[8][4], dn[8][4];
  uint32_t sa[4][1][4];
  zero4(dq);
  wg_fence();
  wg_nt32(s, qa, ring_desc);
  wg_nt32(dp, da, desc_add(ring_desc, SWT32));
  wg_commit();
  wg_wait<0>();
  pin(s);
  pin(dp);
  pin(qa);
  pin(da);
  int stage = 0;
  auto step = [&](int j, auto diag) LATCH_SEG_INLINE {
    constexpr bool DIAG = decltype(diag)::value;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (complete)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    const uint64_t kd = desc_add(ring_desc, stage * ST32), nd = desc_add(ring_desc, (DIAG ? stage : next) * ST32);
    wg_fence();
    wg_nt32(sn, qa, nd);
    wg_nt32(dn, da, desc_add(nd, SWT32));
    wg_commit();
    if (DIAG) diag_mask(s, true, r_lo, r_hi, j * BT, lane);
    ds_rows32(a, s, dp, nullptr, nullptr, 0, s_lo, s_hi, d_lo, d_hi, lane);
    count_if(tid == 0, a.work + 2, 1);
    to_a<1, 4, 8>(sa, s);
    wg_fence();
    wg_pv32(dq, sa, kd);
    wg_commit();
    wg_wait<0>();
    pin(dq);
    pin(sa);
    pin(sn);
    pin(dn);
    pin(qa);
    pin(da);
    copy8(s, sn);
    copy8(dp, dn);
    stage = next;
  };
  for (int j = 0; j < n - 1; ++j) step(j, Flag<false>{});
  step(n - 1, Flag<true>{});
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(r_lo < t, a.dq + (long long)r_lo * a.ldq.sr + d, dq[c][0], dq[c][1]);
    st_bf16x2_if(r_hi < t, a.dq + (long long)r_hi * a.ldq.sr + d, dq[c][2], dq[c][3]);
  }
  wg_bar(wg);
}

PFLATLONG_DEV void kv32_v2(const Args& a, int kb64, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + NST32 * ST32);
  __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(ring + 3 * ST32);
  __nv_bfloat16* vs = ks + 64 * LD32;
  const int t = a.t, key0 = kb64 * 64, n = (t + 63) / 64 - kb64;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  const int sr_row = tid & 63;
  const float* stat_base = tid < 64 ? a.lse2 : a.delta;
  count_if(tid == 0, a.work + 3, min(64, t-key0));
  count_if(tid == 0, a.work + 4, 1);
  ch.rows(ks, a.k + (long long)key0 * sr, t - key0);
  ch.rows(vs, a.v + (long long)key0 * sr, t - key0);
  cp_commit();
  auto issue = [&](int i, int stg) LATCH_SEG_INLINE {
    const int il = i < n ? i : n - 1, row = (kb64 + il) * BT, valid = i < n ? t - row : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.q + row * sr, valid);
    ch.sw(st + SWT32, a.dout + row * sr, valid);
    cp4_tail(sv + stg * 2 * BT + tid, stat_base + row + (sr_row < valid ? sr_row : 0), sr_row < valid);
    cp_commit();
  };
  (void)tstride;
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t ka[2][4], va[2][4];
  a_frags32(ka, ks, 16 * w4, lane);
  a_frags32(va, vs, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  const int k_lo = key0 + 16 * w4 + (lane >> 2), k_hi = k_lo + 8;
  float dk[4][4], dv[4][4], s[8][4], dp[8][4], sn[8][4], dn[8][4];
  uint32_t pa[4][1][4], sa[4][1][4];
  zero4(dk);
  zero4(dv);
  wg_fence();
  wg_nt32(s, ka, ring_desc);
  wg_nt32(dp, va, desc_add(ring_desc, SWT32));
  wg_commit();
  wg_wait<0>();
  pin(s);
  pin(dp);
  pin(ka);
  pin(va);
  int stage = 0;
  auto step = [&](int i, auto masked) LATCH_SEG_INLINE {
    constexpr bool MASK = decltype(masked)::value;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);
    const int next = stage_add(stage, 1), qt = kb64 + i;
    issue(i + 3, stage_add(stage, 3));
    const uint64_t qd = desc_add(ring_desc, stage * ST32), dd = desc_add(qd, SWT32);
    const uint64_t nq = desc_add(ring_desc, (i + 1 < n ? next : stage) * ST32);
    wg_fence();
    wg_nt32(sn, ka, nq);
    wg_nt32(dn, va, desc_add(nq, SWT32));
    wg_commit();
    {
      // The width-64 kv body's element recipe; masked steps are its branch-free per-element mask verbatim.
      const float* sl = sv + stage * 2 * BT;
      const float* dl = sl + BT;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int j = 8 * c + 2 * (lane & 3);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int jj = j + (e & 1);
          float x = sc32(a, s[c][e]);
          if (MASK) {
            const int query_row = qt * BT + jj, key_row = e < 2 ? k_lo : k_hi;
            if (query_row >= t || key_row >= t || key_row > query_row) x = -INFINITY;
          }
          const float p = exp2f(fmaf(x, L2E, -sl[jj]));
          s[c][e] = p;
          dp[c][e] = sc32(a, p * (dp[c][e] - dl[jj]));
        }
      }
    }
    count_if(tid == 0, a.work + 5, 1);
    to_a<1, 4, 8>(pa, s);
    to_a<1, 4, 8>(sa, dp);
    wg_fence();
    wg_pv32(dv, pa, dd);
    wg_pv32(dk, sa, qd);
    wg_commit();
    wg_wait<0>();
    pin(dv);
    pin(dk);
    pin(pa);
    pin(sa);
    pin(sn);
    pin(dn);
    pin(ka);
    pin(va);
    copy8(s, sn);
    copy8(dp, dn);
    stage = next;
  };
  // Query tile kb64 holds the diagonal (key > query) and, when it is the last, the row tail; the last query tile
  // holds the row tail. Every other query tile lies strictly after this key tile and inside the segment.
  // (An alternating-buffer unroll without the copies measured slower on the H100 probes; the copies stay.)
  step(0, Flag<true>{});
  for (int i = 1; i < n - 1; ++i) step(i, Flag<false>{});
  if (n > 1) step(n - 1, Flag<true>{});
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(k_lo < t, a.dk + (long long)k_lo * a.ldk.sr + d, dk[c][0], dk[c][1]);
    st_bf16x2_if(k_hi < t, a.dk + (long long)k_hi * a.ldk.sr + d, dk[c][2], dk[c][3]);
    st_bf16x2_if(k_lo < t, a.dv + (long long)k_lo * a.ldv.sr + d, dv[c][0], dv[c][1]);
    st_bf16x2_if(k_hi < t, a.dv + (long long)k_hi * a.ldv.sr + d, dv[c][2], dv[c][3]);
  }
  wg_bar(wg);
}

// Each worker binary-searches a compact prefix of actual live jobs. No fixed
// max-length slot is submitted to a tensor-core item, even for an empty segment.
// W selects the packed head width (64: the v7 body; 32: the width-32 body with schedule VAR 0/1/2).
// ---------------------------------------------------------------------------------------------------------- VAR 4
// 128-row width-32 items for forward and dQ: a warpgroup owns query tiles 2p (rows A) and 2p + 1 (rows B, absent
// past the segment end) of one (segment, head). Each 64-key tile is loaded once for both row blocks, one barrier and
// one copy issue serve 128 rows, and the two blocks' independent chains overlap: block B's FP32 work runs while block
// A's output/gradient products execute, and the next tile's score products are issued during the step. Per row block
// every operation is that of VAR 2 (same products, same FP32 recipe, same order over key tiles), so stored values
// are unchanged; only rows past the segment end differ. Every step ends with one wait for all of its warpgroup MMAs.
// dK/dV keep the VAR 2 64-key items (a two-block dK/dV would need about 290 registers per thread).
// Row-major A-operand staging uses the 64-byte swizzle (4 KB per tile): four tiles fit in borrowed stages 3..4.
template <bool USE, bool DIAG> struct Rows { static constexpr bool use = USE, diag = DIAG; };

PFLATLONG_DEV void a_frags_sw32(uint32_t (&a)[2][4], const char* tile, int row0, int lane) {
#pragma unroll
  for (int ks = 0; ks < 2; ++ks)
    ldsm_x4(a[ks], tile + sw64(row0 + (lane & 7) + ((lane >> 3) & 1) * 8, 2 * ks + (lane >> 4)));
}

// Both prefixes of real work units per (segment, head): 64-row tiles in [0, G], 128-row pairs in [G + 1, 2G + 1].
template <typename Index>
PFLATLONG_DEV void prepare2(const Index* offsets, int64_t* prefix, int64_t* work,
                            int groups, int heads, int rows, int bound, int columns) {
  for (int i=blockIdx.x*blockDim.x+threadIdx.x; i<groups*heads*columns; i+=gridDim.x*blockDim.x) work[i]=0;
  // One warp scans 32 segments per round (shuffle inclusive scan + carry): the same integer prefix sums as the
  // width-64 prepare's sequential loop, without its chain of dependent loads. The same refusals trap.
  if (blockIdx.x==0 && threadIdx.x<32) {
    const int lane=threadIdx.x;
    if (offsets[0]!=0 || offsets[groups]!=rows) __trap();
    if (lane==0) { prefix[0]=0; prefix[groups+1]=0; }
    long long carry64=0, carry128=0;
    for (int base=0; base<groups; base+=32) {
      const int g=base+lane;
      const bool ok=g<groups;
      const long long lo=ok ? (long long)offsets[g] : 0, hi=ok ? (long long)offsets[g+1] : 0;
      if (ok && (lo<0 || hi<lo || hi>rows || hi-lo>bound)) __trap();
      long long s64=ok ? ((hi-lo+63)/64)*heads : 0, s128=ok ? ((hi-lo+127)/128)*heads : 0;
#pragma unroll
      for (int off=1; off<32; off<<=1) {
        const long long a64=__shfl_up_sync(0xffffffffu, s64, off), a128=__shfl_up_sync(0xffffffffu, s128, off);
        if (lane>=off) { s64+=a64; s128+=a128; }
      }
      if (ok) { prefix[g+1]=carry64+s64; prefix[groups+2+g]=carry128+s128; }
      carry64+=__shfl_sync(0xffffffffu, s64, 31);
      carry128+=__shfl_sync(0xffffffffu, s128, 31);
    }
  }
}

template <bool FTZ>
PFLATLONG_DEV void forward32_v3(const Args& a, int pair, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  char* qsA = ring + 3 * ST32;
  char* qsB = qsA + SWT32;
  const int t = a.t, ntiles = (t + 63) / 64, tA = 2 * pair, tB = tA + 1;
  const bool hasB = tB < ntiles;
  const int n = hasB ? tB + 1 : tA + 1, rowA = tA * 64, rowB = tB * 64;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t - rowA));
  count_if(tid == 0 && hasB, a.work, min(64, t - rowB));
  count_if(tid == 0, a.work + 1, hasB ? 2 : 1);
  ch.sw(qsA, a.q + (long long)rowA * sr, t - rowA);
  ch.sw(qsB, a.q + (long long)(hasB ? rowB : rowA) * sr, hasB ? t - rowB : 0);
  cp_commit();
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qaA[2][4], qaB[2][4];
  a_frags_sw32(qaA, qsA, 16 * w4, lane);
  a_frags_sw32(qaB, qsB, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);  // tile 0 visible; every warp holds its Q fragments (stage 3 may refill)
  const uint64_t ring_desc = sw32_desc(ring);
  float oA[4][4], oB[4][4], sA[8][4], sB[8][4];
  uint32_t paA[4][1][4], paB[4][1][4];
  zero4(oA);
  zero4(oB);
  float mA0 = -INFINITY, mA1 = -INFINITY, lA0 = 0.f, lA1 = 0.f, mB0 = -INFINITY, mB1 = -INFINITY, lB0 = 0.f, lB1 = 0.f;
  const int rA_lo = rowA + 16 * w4 + (lane >> 2), rA_hi = rA_lo + 8, rB_lo = rowB + 16 * w4 + (lane >> 2), rB_hi = rB_lo + 8;
  wg_fence();
  wg_nt32(sA, qaA, ring_desc);
  wg_nt32(sB, qaB, ring_desc);
  wg_commit();
  wg_wait<0>();
  pin(sA);
  pin(sB);
  pin(qaA);
  pin(qaB);
  int stage = 0;
  auto step = [&](int j, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UA = decltype(ra)::use, DA = decltype(ra)::diag, UB = decltype(rb)::use, DB = decltype(rb)::diag;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (complete)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    const uint64_t vd = desc_add(ring_desc, stage * ST32 + SWT32);
    const uint64_t nd = desc_add(ring_desc, (j + 1 < n ? next : stage) * ST32);  // past the last tile: discarded
    float c0, c1;
    if (UA) {
      if (DA) diag_mask(sA, true, rA_lo, rA_hi, j * BT, lane);
      stats32_lean<FTZ>(a, sA, mA0, mA1, lA0, lA1, c0, c1);
      rescale32(oA, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(paA, sA);
      wg_fence();
      wg_pv32(oA, paA, vd);
    }
    wg_fence();
    wg_nt32(sA, qaA, nd);
    wg_commit();
    if (UB) {
      if (DB) diag_mask(sB, true, rB_lo, rB_hi, j * BT, lane);
      stats32_lean<FTZ>(a, sB, mB0, mB1, lB0, lB1, c0, c1);
      rescale32(oB, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(paB, sB);
      wg_fence();
      wg_pv32(oB, paB, vd);
    }
    wg_fence();
    wg_nt32(sB, qaB, nd);
    wg_commit();
    wg_wait<0>();
    pin(oA);
    pin(oB);
    pin(paA);
    pin(paB);
    pin(sA);
    pin(sB);
    pin(qaA);
    pin(qaB);
    stage = next;
  };
  if (hasB) {
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<true, false>{});
    step(tA, Rows<true, true>{}, Rows<true, false>{});
    step(tB, Rows<false, false>{}, Rows<true, true>{});
  } else {  // block B lies past the segment end: no products, FP32 work or counts for it
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<false, false>{});
    step(tA, Rows<true, true>{}, Rows<false, false>{});
  }
  cp_wait0();
  auto store = [&](float (&o)[4][4], float m0, float m1, float l0, float l1, int r_lo, int r_hi) LATCH_SEG_INLINE {
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, off);
      l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;
    l0 = empty0 ? 1.f : l0;
    l1 = empty1 ? 1.f : l1;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = 8 * c + 2 * (lane & 3);
      const float y0 = div_row(o[c][0], l0), y1 = div_row(o[c][1], l0);
      const float y2 = div_row(o[c][2], l1), y3 = div_row(o[c][3], l1);
      st_bf16x2_if(r_lo < t, a.o + (long long)r_lo * a.lo.sr + d, y0, y1);
      st_bf16x2_if(r_hi < t, a.o + (long long)r_hi * a.lo.sr + d, y2, y3);
    }
    const float lse_lo = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * L2E) + log2f(l0);
    const float lse_hi = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * L2E) + log2f(l1);
    st_f32_if((lane & 3) == 0 && r_lo < t, a.lse2 + r_lo, lse_lo);
    st_f32_if((lane & 3) == 0 && r_hi < t, a.lse2 + r_hi, lse_hi);
  };
  store(oA, mA0, mA1, lA0, lA1, rA_lo, rA_hi);
  store(oB, mB0, mB1, lB0, lB1, rB_lo, rB_hi);
  wg_bar(wg);
}

template <bool FTZ>
PFLATLONG_DEV void dq32_v3(const Args& a, int pair, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  char* qsA = ring + 3 * ST32;
  char* dsA = qsA + SWT32;
  char* qsB = dsA + SWT32;
  char* dsB = qsB + SWT32;
  const int t = a.t, ntiles = (t + 63) / 64, tA = 2 * pair, tB = tA + 1;
  const bool hasB = tB < ntiles;
  const int n = hasB ? tB + 1 : tA + 1, rowA = tA * 64, rowB = tB * 64;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t - rowA));
  count_if(tid == 0 && hasB, a.work, min(64, t - rowB));
  count_if(tid == 0, a.work + 1, hasB ? 2 : 1);
  const int rowBc = hasB ? rowB : rowA, validB = hasB ? t - rowB : 0;
  ch.sw(qsA, a.q + (long long)rowA * sr, t - rowA);
  ch.sw(dsA, a.dout + (long long)rowA * sr, t - rowA);
  ch.sw(qsB, a.q + (long long)rowBc * sr, validB);
  ch.sw(dsB, a.dout + (long long)rowBc * sr, validB);
  cp_commit();
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  // Row statistics of both blocks: delta = rowsum(dO * O) in the width-64 body's order (see dq32), lse2 per row.
  auto delta = [&](int wrow, float& d_lo, float& d_hi) LATCH_SEG_INLINE {
    const int r = wrow + (lane >> 1), h = lane & 1, rc = r < t ? r : 0;
    const bool own = h == 0 && r < t;
    const __nv_bfloat16* dp = a.dout + (long long)rc * sr;
    const __nv_bfloat16* op = a.o + (long long)rc * sr;
    float acc = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 w = __ldcg(reinterpret_cast<const uint4*>(dp) + c);
      const uint4 z = __ldcg(reinterpret_cast<const uint4*>(op) + c);
      acc = fmaf(bf_lo(w.x), bf_lo(z.x), acc); acc = fmaf(bf_hi(w.x), bf_hi(z.x), acc);
      acc = fmaf(bf_lo(w.y), bf_lo(z.y), acc); acc = fmaf(bf_hi(w.y), bf_hi(z.y), acc);
      acc = fmaf(bf_lo(w.z), bf_lo(z.z), acc); acc = fmaf(bf_hi(w.z), bf_hi(z.z), acc);
      acc = fmaf(bf_lo(w.w), bf_lo(z.w), acc); acc = fmaf(bf_hi(w.w), bf_hi(z.w), acc);
    }
    acc = own ? acc : 0.f;
    const float other = __shfl_xor_sync(0xffffffffu, acc, 1);
    const float dl = h ? other + acc : acc + other;
    st_f32_if(own, a.delta + r, dl);
    d_lo = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2));
    d_hi = __shfl_sync(0xffffffffu, dl, 2 * (lane >> 2) + 16);
  };
  float dA_lo, dA_hi, dB_lo, dB_hi;
  delta(rowA + 16 * w4, dA_lo, dA_hi);
  delta(rowB + 16 * w4, dB_lo, dB_hi);
  const int rA_lo = rowA + 16 * w4 + (lane >> 2), rA_hi = rA_lo + 8, rB_lo = rowB + 16 * w4 + (lane >> 2), rB_hi = rB_lo + 8;
  auto lse = [&](int r) LATCH_SEG_INLINE { const float v = a.lse2[r < t ? r : 0]; return r < t ? v : INFINITY; };
  const float sA_lo = lse(rA_lo), sA_hi = lse(rA_hi), sB_lo = lse(rB_lo), sB_hi = lse(rB_hi);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t qaA[2][4], daA[2][4], qaB[2][4], daB[2][4];
  a_frags_sw32(qaA, qsA, 16 * w4, lane);
  a_frags_sw32(daA, dsA, 16 * w4, lane);
  a_frags_sw32(qaB, qsB, 16 * w4, lane);
  a_frags_sw32(daB, dsB, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  float dqA[4][4], dqB[4][4], s[8][4], dp[8][4];
  uint32_t saA[4][1][4], saB[4][1][4];
  zero4(dqA);
  zero4(dqB);
  wg_fence();
  wg_nt32(s, qaA, ring_desc);
  wg_nt32(dp, daA, desc_add(ring_desc, SWT32));
  wg_commit();
  wg_wait<0>();
  pin(s);
  pin(dp);
  pin(qaA);
  pin(daA);
  int stage = 0;
  auto step = [&](int j, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UA = decltype(ra)::use, DA = decltype(ra)::diag, UB = decltype(rb)::use, DB = decltype(rb)::diag;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (complete)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    const uint64_t kd = desc_add(ring_desc, stage * ST32), nd = desc_add(ring_desc, (j + 1 < n ? next : stage) * ST32);
    if (UA) {  // (s, dp) hold block A's products of tile j
      if (DA) diag_mask(s, true, rA_lo, rA_hi, j * BT, lane);
      ds_rows32<FTZ>(a, s, dp, nullptr, nullptr, 0, sA_lo, sA_hi, dA_lo, dA_hi, lane);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(saA, s);
    }
    if (UB) {
      wg_fence();
      wg_nt32(s, qaB, kd);
      wg_nt32(dp, daB, desc_add(kd, SWT32));
      wg_commit();
    }
    if (UA) {
      wg_fence();
      wg_pv32(dqA, saA, kd);
      wg_commit();
    }
    if (UB) {
      if (UA) wg_wait<1>(); else wg_wait<0>();  // block B's products complete (A's dQ products may run)
      pin(s);
      pin(dp);
      pin(qaB);
      pin(daB);
      if (DB) diag_mask(s, true, rB_lo, rB_hi, j * BT, lane);
      ds_rows32<FTZ>(a, s, dp, nullptr, nullptr, 0, sB_lo, sB_hi, dB_lo, dB_hi, lane);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(saB, s);
      wg_fence();
      wg_pv32(dqB, saB, kd);
      wg_commit();
    }
    wg_fence();
    wg_nt32(s, qaA, nd);  // block A's products of tile j + 1 (discarded when unused)
    wg_nt32(dp, daA, desc_add(nd, SWT32));
    wg_commit();
    wg_wait<0>();
    pin(dqA);
    pin(dqB);
    pin(saA);
    pin(saB);
    pin(s);
    pin(dp);
    pin(qaA);
    pin(daA);
    pin(qaB);
    pin(daB);
    stage = next;
  };
  if (hasB) {
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<true, false>{});
    step(tA, Rows<true, true>{}, Rows<true, false>{});
    step(tB, Rows<false, false>{}, Rows<true, true>{});
  } else {  // block B lies past the segment end: no products, FP32 work or counts for it
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<false, false>{});
    step(tA, Rows<true, true>{}, Rows<false, false>{});
  }
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(rA_lo < t, a.dq + (long long)rA_lo * a.ldq.sr + d, dqA[c][0], dqA[c][1]);
    st_bf16x2_if(rA_hi < t, a.dq + (long long)rA_hi * a.ldq.sr + d, dqA[c][2], dqA[c][3]);
    st_bf16x2_if(rB_lo < t, a.dq + (long long)rB_lo * a.ldq.sr + d, dqB[c][0], dqB[c][1]);
    st_bf16x2_if(rB_hi < t, a.dq + (long long)rB_hi * a.ldq.sr + d, dqB[c][2], dqB[c][3]);
  }
  wg_bar(wg);
}

// ---------------------------------------------------------------------------------------------------------- VAR 5
// 128-key dK/dV items (VAR 5 = VAR 4 forward/dQ + these): a warpgroup owns key tiles 2p (keys A) and 2p + 1 (keys B,
// absent past the segment end). Each query/dO tile and its statistics are loaded once for both key blocks; block B's
// FP32 work runs while block A's dV/dK products execute; the next tile's block-A score products are issued during the
// step. Per key block every operation is that of kv32_v2 (same products in the same query-tile order, the width-64
// element recipe, masks on the same tiles), so stored values are unchanged. One (s, dp) score pair and one (pa, sa)
// fragment pair serve both blocks in turn, keeping about 190 live registers.
template <bool FTZ>
PFLATLONG_DEV void kv32_v3(const Args& a, int kpair, void* smem_v) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB;
  float* sv = reinterpret_cast<float*>(ring + NST32 * ST32);
  char* ksA = ring + 3 * ST32;
  char* vsA = ksA + SWT32;
  char* ksB = vsA + SWT32;
  char* vsB = ksB + SWT32;
  const int t = a.t, ntiles = (t + 63) / 64, kA = 2 * kpair, kB = kA + 1;
  const bool hasB = kB < ntiles;
  const int n = ntiles - kA, keyA = kA * 64, keyB = kB * 64;  // query tiles kA .. last
  const long long sr = a.lq.sr;
  const Chunks32 ch(tid, sr);
  const int sr_row = tid & 63;
  const float* stat_base = tid < 64 ? a.lse2 : a.delta;
  count_if(tid == 0, a.work + 3, min(64, t - keyA));
  count_if(tid == 0 && hasB, a.work + 3, min(64, t - keyB));
  count_if(tid == 0, a.work + 4, hasB ? 2 : 1);
  const int keyBc = hasB ? keyB : keyA, validB = hasB ? t - keyB : 0;
  ch.sw(ksA, a.k + (long long)keyA * sr, t - keyA);
  ch.sw(vsA, a.v + (long long)keyA * sr, t - keyA);
  ch.sw(ksB, a.k + (long long)keyBc * sr, validB);
  ch.sw(vsB, a.v + (long long)keyBc * sr, validB);
  cp_commit();
  auto issue = [&](int i, int stg) LATCH_SEG_INLINE {
    const int il = i < n ? i : n - 1, row = (kA + il) * BT, valid = i < n ? t - row : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.q + row * sr, valid);
    ch.sw(st + SWT32, a.dout + row * sr, valid);
    cp4_tail(sv + stg * 2 * BT + tid, stat_base + row + (sr_row < valid ? sr_row : 0), sr_row < valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<3>();
  wg_bar(wg);
  uint32_t kaA[2][4], vaA[2][4], kaB[2][4], vaB[2][4];
  a_frags_sw32(kaA, ksA, 16 * w4, lane);
  a_frags_sw32(vaA, vsA, 16 * w4, lane);
  a_frags_sw32(kaB, ksB, 16 * w4, lane);
  a_frags_sw32(vaB, vsB, 16 * w4, lane);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);
  const uint64_t ring_desc = sw32_desc(ring);
  const int kA_lo = keyA + 16 * w4 + (lane >> 2), kA_hi = kA_lo + 8, kB_lo = keyB + 16 * w4 + (lane >> 2), kB_hi = kB_lo + 8;
  float dkA[4][4], dvA[4][4], dkB[4][4], dvB[4][4], s[8][4], dp[8][4];
  uint32_t pa[4][1][4], sa[4][1][4];
  zero4(dkA);
  zero4(dvA);
  zero4(dkB);
  zero4(dvB);
  wg_fence();
  wg_nt32(s, kaA, ring_desc);
  wg_nt32(dp, vaA, desc_add(ring_desc, SWT32));
  wg_commit();
  wg_wait<0>();
  pin(s);
  pin(dp);
  pin(kaA);
  pin(vaA);
  int stage = 0;
  // The width-64 kv body's element recipe for one key block; MASK applies its per-element mask verbatim.
  auto elements = [&](auto masked, int qt, int k_lo, int k_hi, const float* sl) LATCH_SEG_INLINE {
    constexpr bool MASK = decltype(masked)::value;
    const float* dl = sl + BT;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int j = 8 * c + 2 * (lane & 3);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int jj = j + (e & 1);
        float x = sc32(a, s[c][e]);
        if (MASK) {
          const int query_row = qt * BT + jj, key_row = e < 2 ? k_lo : k_hi;
          if (query_row >= t || key_row >= t || key_row > query_row) x = -INFINITY;
        }
        const float p = seg_ex2<FTZ>(fmaf(x, L2E, -sl[jj]));
        s[c][e] = p;
        dp[c][e] = sc32(a, p * (dp[c][e] - dl[jj]));
      }
    }
    count_if(tid == 0, a.work + 5, 1);
  };
  auto step = [&](int i, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UA = decltype(ra)::use, MA = decltype(ra)::diag, UB = decltype(rb)::use, MB = decltype(rb)::diag;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // query tile i + 1 visible; stage of tile i + 3 last held tile i - 2 (complete)
    const int next = stage_add(stage, 1), qt = kA + i;
    issue(i + 3, stage_add(stage, 3));
    const uint64_t qd = desc_add(ring_desc, stage * ST32), dd = desc_add(qd, SWT32);
    const uint64_t nq = desc_add(ring_desc, (i + 1 < n ? next : stage) * ST32);
    const float* sl = sv + stage * 2 * BT;
    if (UA) {  // (s, dp) hold block A's products of query tile i
      elements(Flag<MA>{}, qt, kA_lo, kA_hi, sl);
      to_a<1, 4, 8>(pa, s);
      to_a<1, 4, 8>(sa, dp);
    }
    if (UB) {
      wg_fence();
      wg_nt32(s, kaB, qd);
      wg_nt32(dp, vaB, dd);
      wg_commit();
    }
    if (UA) {
      wg_fence();
      wg_pv32(dvA, pa, dd);
      wg_pv32(dkA, sa, qd);
      wg_commit();
    }
    if (UB) {
      if (UA) wg_wait<1>(); else wg_wait<0>();  // block B's products complete (A's dV/dK products may run)
      pin(s);
      pin(dp);
      pin(kaB);
      pin(vaB);
      elements(Flag<MB>{}, qt, kB_lo, kB_hi, sl);
      if (UA) {
        wg_wait<0>();  // A's products have read pa / sa
        pin(pa);
        pin(sa);
        pin(dvA);
        pin(dkA);
      }
      to_a<1, 4, 8>(pa, s);
      to_a<1, 4, 8>(sa, dp);
      wg_fence();
      wg_pv32(dvB, pa, dd);
      wg_pv32(dkB, sa, qd);
      wg_commit();
    }
    wg_fence();
    wg_nt32(s, kaA, nq);  // block A's products of query tile i + 1 (discarded past the last tile)
    wg_nt32(dp, vaA, desc_add(nq, SWT32));
    wg_commit();
    wg_wait<0>();
    pin(dvA);
    pin(dkA);
    pin(dvB);
    pin(dkB);
    pin(pa);
    pin(sa);
    pin(s);
    pin(dp);
    pin(kaA);
    pin(vaA);
    pin(kaB);
    pin(vaB);
    stage = next;
  };
  // Query tile kA: diagonal of block A (and the row tail when it is the last); block B's keys all follow it.
  // Query tile kB: diagonal of block B; block A unmasked unless kB is the last (row tail). Later tiles need masks
  // only as the last (row tail). Every other (block, tile) lies strictly after the keys and inside the segment.
  step(0, Rows<true, true>{}, Rows<false, false>{});
  if (n > 1) {
    if (n == 2) {
      step(1, Rows<true, true>{}, Rows<true, true>{});
    } else {
      step(1, Rows<true, false>{}, Rows<true, true>{});
      for (int i = 2; i < n - 1; ++i) step(i, Rows<true, false>{}, Rows<true, false>{});
      step(n - 1, Rows<true, true>{}, Rows<true, true>{});
    }
  }
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(kA_lo < t, a.dk + (long long)kA_lo * a.ldk.sr + d, dkA[c][0], dkA[c][1]);
    st_bf16x2_if(kA_hi < t, a.dk + (long long)kA_hi * a.ldk.sr + d, dkA[c][2], dkA[c][3]);
    st_bf16x2_if(kA_lo < t, a.dv + (long long)kA_lo * a.ldv.sr + d, dvA[c][0], dvA[c][1]);
    st_bf16x2_if(kA_hi < t, a.dv + (long long)kA_hi * a.ldv.sr + d, dvA[c][2], dvA[c][3]);
    st_bf16x2_if(kB_lo < t, a.dk + (long long)kB_lo * a.ldk.sr + d, dkB[c][0], dkB[c][1]);
    st_bf16x2_if(kB_hi < t, a.dk + (long long)kB_hi * a.ldk.sr + d, dkB[c][2], dkB[c][3]);
    st_bf16x2_if(kB_lo < t, a.dv + (long long)kB_lo * a.ldv.sr + d, dvB[c][0], dvB[c][1]);
    st_bf16x2_if(kB_hi < t, a.dv + (long long)kB_hi * a.ldv.sr + d, dvB[c][2], dvB[c][3]);
  }
  wg_bar(wg);
}

// ---------------------------------------------------------------------------------------------------------- VAR 8
// Single-pass width-32 backward (segmented_attention 'fused_backward'). The dQ pass is gone:
// each 128-key dK/dV item (VAR 5's item set and order) also forms dS = P o (dP - delta) for its query tiles, stages
// it in shared memory as BF16 and adds dS K into an FP32 [rows, heads, 32] dQ accumulator by global reductions.
// delta = rowsum(dO o O) and the accumulator zeroing run first (prepare phase); a final elementwise phase writes
// dQ = scale * accumulator in BF16. Exponentials: one MUFU ex2.approx.ftz per score element in the backward instead
// of two; the exponent folds the softmax scale (fmaf(s, scale * log2 e, -lse2)) and the scale multiplies dK/dQ once
// at the end, so per element: FFMA, MUFU, FADD, FMUL (+ two BF16 packs). Same mathematics as VAR 6; not bitwise
// (folded scale, regrouped and order-nondeterministic dQ sums).
// Warpgroup region WGB8 (72 KB; VAR 8 phases declare 1 KB + 2 x 72 KB of shared memory): ring of 4 stages (Q tile,
// dO tile: 8 KB) [0, 32K); statistics 4 x 512 B at 32K; resident K tiles (keys A, B; 64-byte swizzle) at 34K and V
// tiles at 42K (both read by wgmma from shared memory: no operand registers); dS^T staging (64 keys x 64 queries,
// 128-byte swizzle) for key block A at 50K and B at 58K; run32's prefix cache / claim slots / rank table at 66K.
// KX 6 adds two 8 KB FP32 dQ staging tiles at 66K (asynchronous bulk reductions). The forward items use [0, 48K)
// of the same region (5-stage K/V ring + resident Q tiles).
constexpr int WGB8 = 88 * 1024, POFF8 = 82 * 1024, N8_XST = 66 * 1024;
constexpr int N8_STG = 4, N8_STATS = 32 * 1024, N8_KRES = 34 * 1024, N8_VRES = 42 * 1024, N8_DS = 50 * 1024;
static_assert(N8_STG * ST32 <= N8_STATS && N8_STATS + N8_STG * 2 * BT * 4 <= N8_KRES && N8_KRES + 2 * SWT32 <= N8_VRES &&
              N8_VRES + 2 * SWT32 <= N8_DS && N8_DS + 2 * 64 * 128 <= N8_XST && N8_XST + 2 * 8192 <= POFF8 && POFF8 + 2 * 8 * (256 + 1) + 8 + 33 * 8 <= WGB8,
              "VAR 8 warpgroup layout");
// The warpgroup index broadcast from lane 0 (CUTLASS canonical_warp_group_idx): the compiler then keeps shared
// addresses and wgmma descriptors in uniform registers instead of one R2UR per descriptor use.
PFLATLONG_DEV int wg_uniform() { return __shfl_sync(0xffffffffu, (int)(threadIdx.x >> 7), 0); }
// VAR 8 descriptors as 32-bit low words with constant high words: offsets are 32-bit adds (no carry), and the high
// word is a compile-time constant ptxas can keep in a uniform register (fewer IADD3/R2UR per wgmma).
constexpr uint32_t HI64 = 0x80000020u;   // 64-byte swizzle, SBO 512 B
constexpr uint32_t HI128 = 0x40000040u;  // 128-byte swizzle, SBO 1024 B
PFLATLONG_DEV uint32_t dlo(const void* p) { return ((smem_u32(p) & 0x3FFFFu) >> 4) | (1u << 16); }
PFLATLONG_DEV uint32_t dadd(uint32_t lo, uint32_t bytes) { return lo + (bytes >> 4); }
#define N8_ACC64 "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, " \
                 "%22, %23, %24, %25, %26, %27, %28, %29, %30, %31}"
#define N8_OUT64(M) M(d[0][0]), M(d[0][1]), M(d[0][2]), M(d[0][3]), M(d[1][0]), M(d[1][1]), M(d[1][2]), M(d[1][3]), \
    M(d[2][0]), M(d[2][1]), M(d[2][2]), M(d[2][3]), M(d[3][0]), M(d[3][1]), M(d[3][2]), M(d[3][3]), M(d[4][0]), \
    M(d[4][1]), M(d[4][2]), M(d[4][3]), M(d[5][0]), M(d[5][1]), M(d[5][2]), M(d[5][3]), M(d[6][0]), M(d[6][1]), \
    M(d[6][2]), M(d[6][3]), M(d[7][0]), M(d[7][1]), M(d[7][2]), M(d[7][3])
#define N8_OUT32(M) M(d[0][0]), M(d[0][1]), M(d[0][2]), M(d[0][3]), M(d[1][0]), M(d[1][1]), M(d[1][2]), M(d[1][3]), \
    M(d[2][0]), M(d[2][1]), M(d[2][2]), M(d[2][3]), M(d[3][0]), M(d[3][1]), M(d[3][2]), M(d[3][3])
#define N8_W(x) "=f"(x)
#define N8_RW(x) "+f"(x)
// 64 x 64 scores from two K-major 64-byte-swizzled 64 x 32 tiles (A rows = M, B rows = N): two k16 steps, the first
// writes the accumulators without reading them.
template <uint32_t AH = HI64, uint32_t BH = HI64>
PFLATLONG_DEV void n8_ss64(float (&d)[8][4], uint32_t a, uint32_t b) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 da, db;\nsetp.ne.b32 p, %36, 0;\nmov.b64 da, {%32, %34};\n"
               "mov.b64 db, {%33, %35};\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " N8_ACC64 ", da, db, p, 1, 1, 0, 0;\n}\n"
               : N8_OUT64(N8_W) : "r"(a), "r"(b), "r"(AH), "r"(BH), "r"(0));
  asm volatile("{\n.reg .pred p;\n.reg .b64 da, db;\nsetp.ne.b32 p, %36, 0;\nmov.b64 da, {%32, %34};\n"
               "mov.b64 db, {%33, %35};\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 " N8_ACC64 ", da, db, p, 1, 1, 0, 0;\n}\n"
               : N8_OUT64(N8_RW) : "r"(a + 2u), "r"(b + 2u), "r"(AH), "r"(BH), "r"(1));
}
// m64n32k16 accumulate, both operands in shared memory (TA / TB: transposed readings).
template <int TA, int TB, uint32_t AH, uint32_t BH>
PFLATLONG_DEV void n8_ss32(float (&d)[4][4], uint32_t a, uint32_t b, int scale_d) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 da, db;\nsetp.ne.b32 p, %18, 0;\nmov.b64 da, {%16, %19};\nmov.b64 db, {%17, %20};\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
               "%11, %12, %13, %14, %15}, da, db, p, 1, 1, %21, %22;\n}\n"
               : N8_OUT32(N8_RW) : "r"(a), "r"(b), "r"(scale_d), "r"(AH), "r"(BH), "n"(TA), "n"(TB));
}
// m64n32k16 accumulate, A from registers, B (64-byte swizzle) read MN-major.
PFLATLONG_DEV void n8_rs32(float (&d)[4][4], const uint32_t (&a)[4], uint32_t b) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 db;\nsetp.ne.b32 p, %22, 0;\nmov.b64 db, {%20, %21};\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, "
               "%11, %12, %13, %14, %15}, {%16, %17, %18, %19}, db, p, 1, 1, 1;\n}\n"
               : N8_OUT32(N8_RW) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b), "r"(HI64), "r"(1));
}
// acc (64 x 32) += P (64 x 64, register fragments) * B (64 x 32 tile read MN-major): four k16 steps of 1 KB.
PFLATLONG_DEV void n8_pv32(float (&acc)[4][4], const uint32_t (&p)[4][1][4], uint32_t b) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) n8_rs32(acc, p[kk][0], b + 64u * kk);
}

PFLATLONG_DEV int stage4(int s, int k) { return (s + k) & 3; }
// m64n64k16 with both operands in shared memory (no transposes).
PFLATLONG_DEV void wgmma64ss(float (&d)[8][4], uint64_t a, uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
      "%23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3]), "+f"(d[4][0]), "+f"(d[4][1]), "+f"(d[4][2]), "+f"(d[4][3]), "+f"(d[5][0]),
        "+f"(d[5][1]), "+f"(d[5][2]), "+f"(d[5][3]), "+f"(d[6][0]), "+f"(d[6][1]), "+f"(d[6][2]), "+f"(d[6][3]),
        "+f"(d[7][0]), "+f"(d[7][1]), "+f"(d[7][2]), "+f"(d[7][3])
      : "l"(a), "l"(b), "r"(scale_d));
}
// m64n32k16 with both operands in shared memory; TA/TB select the transposed (MN-major) reading.
template <int TA, int TB>
PFLATLONG_DEV void wgmma32ss(float (&d)[4][4], uint64_t a, uint64_t b, int scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, p, 1, 1, %19, %20;\n}\n"
      : "+f"(d[0][0]), "+f"(d[0][1]), "+f"(d[0][2]), "+f"(d[0][3]), "+f"(d[1][0]), "+f"(d[1][1]), "+f"(d[1][2]),
        "+f"(d[1][3]), "+f"(d[2][0]), "+f"(d[2][1]), "+f"(d[2][2]), "+f"(d[2][3]), "+f"(d[3][0]), "+f"(d[3][1]),
        "+f"(d[3][2]), "+f"(d[3][3])
      : "l"(a), "l"(b), "r"(scale_d), "n"(TA), "n"(TB));
}
// acc (64 x 64) = A (64 x 32 tile, K-major 64-byte swizzle) * B^T (64 x 32 tile, K-major 64-byte swizzle).
PFLATLONG_DEV void wg_ss64(float (&acc)[8][4], uint64_t a, uint64_t b) {
  wgmma64ss(acc, a, b, 0);
  wgmma64ss(acc, desc_add(a, 32), desc_add(b, 32), 1);
}
// 128-byte swizzle descriptor of the dS^T staging tile (rows of 64 BF16): SBO 1024 B between 8-row groups.
PFLATLONG_DEV uint64_t ds_desc(const void* p) { return sw_desc(p, 1, 64); }
// dK (64 keys x 32) += dS^T (keys x 64 queries, staging tile K-major) * Q (64 queries x 32, ring tile read MN-major).
PFLATLONG_DEV void wg_dk8(float (&acc)[4][4], uint32_t ds, uint32_t q) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) n8_ss32<0, 1, HI128, HI64>(acc, ds + 2u * kk, q + 64u * kk, 1);
}
// dQ (64 queries x 32) (+)= dS (queries x 64 keys, staging tile read MN-major) * K (64 keys x 32, resident tile MN-major).
PFLATLONG_DEV void wg_dq8(float (&acc)[4][4], uint32_t ds, uint32_t k, int first) {
#pragma unroll
  for (int kk = 0; kk < 4; ++kk) n8_ss32<1, 1, HI128, HI64>(acc, ds + 128u * kk, k + 64u * kk, first && kk == 0 ? 0 : 1);
}
// Two FP32 global reductions of adjacent channels (sm_90 vector red; no return value, no CAS loop).
PFLATLONG_DEV void red2_if(bool pred, float* ptr, float x0, float x1) {
  asm volatile("{\n.reg .pred q;\n.reg .u64 g;\nsetp.ne.b32 q, %3, 0;\ncvta.to.global.u64 g, %0;\n"
               "@q red.relaxed.gpu.global.add.v2.f32 [g], {%1, %2};\n}\n"
               :: "l"(ptr), "f"(x0), "f"(x1), "r"((int)pred) : "memory");
}
// Asynchronous bulk FP32 reduction of a contiguous shared-memory block into global memory (sm_90 bulk async copy).
PFLATLONG_DEV void bulk_red_f32(float* gdst, const void* ssrc, uint32_t bytes) {
  asm volatile("{\n.reg .u64 g;\ncvta.to.global.u64 g, %0;\n"
               "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [g], [%1], %2;\n}\n"
               :: "l"(gdst), "r"(smem_u32(ssrc)), "r"(bytes) : "memory");
}
PFLATLONG_DEV void bulk_commit() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
template <int N>
PFLATLONG_DEV void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory"); }
PFLATLONG_DEV void bulk_wait_all() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }
PFLATLONG_DEV void stsm_x4(uint32_t addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
               :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
// BF16 dS^T fragment (rows = keys of this warp, columns = 64 queries) into the 128-byte-swizzled staging tile with
// four stmatrix.x4 (matrices: rows 0-7 / 8-15 of column groups 2cc and 2cc + 1). off[cc] = this lane's row address.
PFLATLONG_DEV void store_ds8(const uint32_t (&off)[4], const float (&x)[8][4]) {
#pragma unroll
  for (int cc = 0; cc < 4; ++cc)
    stsm_x4(off[cc], pack(x[2 * cc][0], x[2 * cc][1]), pack(x[2 * cc][2], x[2 * cc][3]),
            pack(x[2 * cc + 1][0], x[2 * cc + 1][1]), pack(x[2 * cc + 1][2], x[2 * cc + 1][3]));
}

// KX 0: each step's dQ partial is reduced at the end of the step; KX 1: it is reduced one step later, right after
// the next step's barrier, so the shared-memory proxy fence (a CTA memory barrier) does not wait for its reductions.
template <int KX = 0>
PFLATLONG_DEV void kv32_v8(const Args& a, int kpair, void* smem_v) {
  const int lane = threadIdx.x & 31, wg = wg_uniform(), w4 = (threadIdx.x >> 5) & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB8;
  float* sv = reinterpret_cast<float*>(ring + N8_STATS);
  char* kres = ring + N8_KRES;
  char* vres = ring + N8_VRES;
  char* dsb = ring + N8_DS;
  const int t = a.t, ntiles = (t + 63) / 64, kA = 2 * kpair, kB = kA + 1;
  const bool hasB = kB < ntiles;
  const int n = ntiles - kA, keyA = kA * 64, keyB = kB * 64;
  const long long sr = a.lq.sr;
  const Chunks32 ch(tid, sr);
  const int sr_row = tid & 63;
  const float* stat_base = tid < 64 ? a.lse2 : a.delta;
  const float c2 = a.scale * L2E;
  count_if(tid == 0, a.work + 3, min(64, t - keyA));
  count_if(tid == 0 && hasB, a.work + 3, min(64, t - keyB));
  count_if(tid == 0, a.work + 4, hasB ? 2 : 1);
  const int keyBc = hasB ? keyB : keyA, validB = hasB ? t - keyB : 0;
  ch.sw(kres, a.k + (long long)keyA * sr, t - keyA);
  ch.sw(kres + SWT32, a.k + (long long)keyBc * sr, validB);
  ch.sw(vres, a.v + (long long)keyA * sr, t - keyA);
  ch.sw(vres + SWT32, a.v + (long long)keyBc * sr, validB);
  cp_commit();
  auto issue = [&](int i, int stg) LATCH_SEG_INLINE {
    const int il = i < n ? i : n - 1, row = (kA + il) * BT, valid = i < n ? t - row : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.q + row * sr, valid);
    ch.sw(st + SWT32, a.dout + row * sr, valid);
    // KX 9: column j of a tile is stored at ((j >> 1) & 3) * 16 + (j >> 3) * 2 + (j & 1), so each thread's 16 columns
    // (8 c + 2 (lane & 3) + {0, 1}) are contiguous: four 16-byte loads per statistic instead of eight 8-byte loads.
    const int sdst = KX == 9 ? (tid & 64) + ((sr_row >> 1) & 3) * 16 + (sr_row >> 3) * 2 + (sr_row & 1) : tid;
    cp4_tail(sv + stg * 2 * BT + sdst, stat_base + row + (sr_row < valid ? sr_row : 0), sr_row < valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);  // K, V tiles and query tile 0 visible
  const uint32_t ring_desc = dlo(ring), kdA = dlo(kres), kdB = kdA + SWT32 / 16, dsd = dlo(dsb);
  const uint32_t vdA = dlo(vres), vdB = vdA + SWT32 / 16, dsdB = dsd + 64 * 128 / 16;
  uint32_t dsoff[4], dsoffB[4];
  {
    const int m = lane >> 3, rr = lane & 7, row = 16 * w4 + 8 * (m & 1) + rr;
#pragma unroll
    for (int cc = 0; cc < 4; ++cc) {
      dsoff[cc] = smem_u32(dsb + row * 128 + (((2 * cc + (m >> 1)) ^ rr) << 4));
      dsoffB[cc] = dsoff[cc] + 64 * 128;
    }
  }
  const int kA_lo = keyA + 16 * w4 + (lane >> 2), kA_hi = kA_lo + 8, kB_lo = keyB + 16 * w4 + (lane >> 2), kB_hi = kB_lo + 8;
  float dkA[4][4], dvA[4][4], dkB[4][4], dvB[4][4], dq[4][4], s[8][4], dp[8][4];
  uint32_t pa[4][1][4];
  zero4(dkA);
  zero4(dvA);
  zero4(dkB);
  zero4(dvB);
  zero4(dq);
  wg_fence();
  n8_ss64(s, kdA, ring_desc);
  n8_ss64(dp, vdA, ring_desc + SWT32 / 16);
  wg_commit();
  wg_wait<0>();
  pin(s);
  pin(dp);
  int stage = 0;
  // P = ex2(s * scale * log2e - lse2), dS = P o (dP - delta) (unscaled); MASK applies the width-64 element mask.
  auto elements = [&](auto masked, int qt, int k_lo, int k_hi, const float* sl) LATCH_SEG_INLINE {
    constexpr bool MASK = decltype(masked)::value;
    float4 Lq[4], Dq[4];
    if (KX == 9) {
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        Lq[k] = *reinterpret_cast<const float4*>(sl + (lane & 3) * 16 + 4 * k);
        Dq[k] = *reinterpret_cast<const float4*>(sl + BT + (lane & 3) * 16 + 4 * k);
      }
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const int j = 8 * c + 2 * (lane & 3);
      float2 L, D;
      if (KX == 9) {
        L = (c & 1) ? make_float2(Lq[c >> 1].z, Lq[c >> 1].w) : make_float2(Lq[c >> 1].x, Lq[c >> 1].y);
        D = (c & 1) ? make_float2(Dq[c >> 1].z, Dq[c >> 1].w) : make_float2(Dq[c >> 1].x, Dq[c >> 1].y);
      } else {
        L = *reinterpret_cast<const float2*>(sl + j);
        D = *reinterpret_cast<const float2*>(sl + BT + j);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float x = s[c][e];
        if (MASK) {
          const int query_row = qt * BT + j + (e & 1), key_row = e < 2 ? k_lo : k_hi;
          if (query_row >= t || key_row >= t || key_row > query_row) x = -INFINITY;
        }
        const float p = seg_ex2<true>(fmaf(x, c2, -((e & 1) ? L.y : L.x)));
        s[c][e] = p;
        dp[c][e] = p * (dp[c][e] - ((e & 1) ? D.y : D.x));
      }
    }
    count_if(tid == 0, a.work + 5, 1);
  };
  float dqp[4][4];
  int qtp = -1;
  // KX 6: FP32 dQ partial tiles staged in shared memory (rows x 32, row-major) and added to the [heads, rows, 32]
  // accumulator by one asynchronous bulk reduction per step (issued after the next step's barrier).
  float* xst = reinterpret_cast<float*>(ring + N8_XST);
  int xq = -1, xb = 0;  // staged query tile (or -1) and its buffer
  auto stage_dq = [&](int buf) LATCH_SEG_INLINE {
    float* x = xst + buf * 2048;
    const int r_lo = 16 * w4 + (lane >> 2), r_hi = r_lo + 8;
    if (KX == 10) {
      // In store k a lane writes column group c = k ^ (row & 3): the eight rows of one store hit four different bank
      // groups (2-way instead of 8-way conflicts). Values are selected from registers (no local memory).
      const int rx = (lane >> 2) & 3;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int c = k ^ rx, d = 8 * c + 2 * (lane & 3);
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[e] = c == 0 ? dq[0][e] : c == 1 ? dq[1][e] : c == 2 ? dq[2][e] : dq[3][e];
        *reinterpret_cast<float2*>(x + r_lo * 32 + d) = make_float2(v[0], v[1]);
        *reinterpret_cast<float2*>(x + r_hi * 32 + d) = make_float2(v[2], v[3]);
      }
    } else {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int d = 8 * c + 2 * (lane & 3);
        *reinterpret_cast<float2*>(x + r_lo * 32 + d) = make_float2(dq[c][0], dq[c][1]);
        *reinterpret_cast<float2*>(x + r_hi * 32 + d) = make_float2(dq[c][2], dq[c][3]);
      }
    }
  };
  auto issue_dq = [&]() LATCH_SEG_INLINE {  // after a barrier that follows the staging writes and a proxy fence
    if (tid == 0 && xq >= 0) {
      const int rows = min(64, t - xq * BT);
      if (KX != 8) bulk_red_f32(a.dqa + (long long)xq * BT * W32, xst + xb * 2048, (uint32_t)rows * 128u);
      bulk_commit();
      bulk_wait_read<1>();  // the other buffer's reduction has read its tile
    }
  };
  auto reduce_dq = [&](int qt, const float (&x)[4][4]) LATCH_SEG_INLINE {
    if (KX == 4) return;  // timing diagnostic only: no dQ reductions (dQ wrong)
    const int q_lo = qt * BT + 16 * w4 + (lane >> 2), q_hi = q_lo + 8;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = 8 * c + 2 * (lane & 3);
      red2_if(q_lo < t, a.dqa + (long long)q_lo * sr + d, x[c][0], x[c][1]);
      red2_if(q_hi < t, a.dqa + (long long)q_hi * sr + d, x[c][2], x[c][3]);
    }
  };
  auto step = [&](int i, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UB = decltype(rb)::use, MA = decltype(ra)::diag, MB = decltype(rb)::diag;
    const int next = stage4(stage, 1), qt = kA + i;
    const uint32_t qd = ring_desc + stage * (ST32 / 16), dd = qd + SWT32 / 16;
    const uint32_t nq = ring_desc + (i + 1 < n ? next : stage) * (ST32 / 16);
    const float* sl = sv + stage * 2 * BT;
    elements(Flag<MA>{}, qt, kA_lo, kA_hi, sl);   // (s, dp) held block A's products of query tile i
    to_a<1, 4, 8>(pa, s);
    store_ds8(dsoff, dp);   // block A's previous products (step i - 1) completed before this step began
    cp_wait_n<1>();         // query tile i + 1 landed (tile i + 2 may still be in flight)
    proxy_fence();
    wg_bar(wg);  // dS_A^T and tile i + 1 visible; every warp finished step i - 1, so tile i - 1's stage is free
    issue(i + 3, stage4(stage, 3));
    if (KX == 1 && qtp >= 0) reduce_dq(qtp, dqp);
    if (KX == 6 || KX == 8 || KX == 9 || KX == 10) issue_dq();
    if (UB) {
      wg_fence();
      n8_ss64(s, kdB, qd);
      n8_ss64(dp, vdB, dd);
      wg_commit();
    }
    wg_fence();
    n8_pv32(dvA, pa, dd);
    wg_dk8(dkA, dsd, qd);
    wg_dq8(dq, dsd, kdA, 1);
    wg_commit();
    if (UB) {
      wg_wait<1>();  // block B's products complete (A's dV/dK/dQ products may run)
      pin(s);
      pin(dp);
      elements(Flag<MB>{}, qt, kB_lo, kB_hi, sl);
      wg_wait<0>();  // A's products have read pa
      pin(pa);
      pin(dvA);
      pin(dkA);
      pin(dq);
      to_a<1, 4, 8>(pa, s);
      store_ds8(dsoffB, dp);
      proxy_fence();
      wg_bar(wg);
      wg_fence();
      n8_pv32(dvB, pa, dd);
      wg_dk8(dkB, dsdB, qd);
      wg_dq8(dq, dsdB, kdB, 0);
      wg_commit();
    }
    wg_fence();
    n8_ss64(s, kdA, nq);  // block A's products of query tile i + 1 (discarded past the last tile)
    n8_ss64(dp, vdA, nq + SWT32 / 16);
    wg_commit();
    wg_wait<0>();
    pin(dvA);
    pin(dkA);
    pin(dvB);
    pin(dkB);
    pin(dq);
    pin(pa);
    pin(s);
    pin(dp);
    if (KX == 1) {
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int e = 0; e < 4; ++e) dqp[c][e] = dq[c][e];
      qtp = qt;
    } else if (KX == 6 || KX == 8 || KX == 9 || KX == 10) {
      if (!UB) wg_bar(wg);  // the issuing thread's read wait (this step) precedes every overwrite of the other buffer
      xb = xq >= 0 ? xb ^ 1 : 0;
      stage_dq(xb);
      xq = qt;
    } else {
      reduce_dq(qt, dq);
    }
    stage = next;
  };
  // KX 2: rotated step. The next tile's score products are issued before key block B's dV/dK/dQ products, and the
  // latter stay in flight into the next step: they overlap block A's element work, and are waited for (with the
  // reduction of the finished dQ partial) just before block A's dS is staged. Same products, same order per block.
  auto step2 = [&](int i, auto ra, auto rb, auto pb) LATCH_SEG_INLINE {
    constexpr bool UB = decltype(rb)::use, MA = decltype(ra)::diag, MB = decltype(rb)::diag;
    constexpr bool PB = decltype(pb)::value;  // the previous step left block B's products in flight
    const int next = stage4(stage, 1), qt = kA + i;
    const uint32_t qd = ring_desc + stage * (ST32 / 16), dd = qd + SWT32 / 16;
    const uint32_t nq = ring_desc + (i + 1 < n ? next : stage) * (ST32 / 16);
    const float* sl = sv + stage * 2 * BT;
    elements(Flag<MA>{}, qt, kA_lo, kA_hi, sl);
    if (PB) {
      wg_wait<0>();  // previous block B products: pa, dS_B staging and the dQ partial are free / complete
      pin(pa);
      pin(dvB);
      pin(dkB);
      pin(dq);
    }
    to_a<1, 4, 8>(pa, s);
    store_ds8(dsoff, dp);
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);
    issue(i + 3, stage4(stage, 3));
    if (i > 0) reduce_dq(qt - 1, dq);  // previous step's partial, before this step's first dQ product overwrites it
    if (UB) {
      wg_fence();
      n8_ss64(s, kdB, qd);
      n8_ss64(dp, vdB, dd);
      wg_commit();
    }
    wg_fence();
    n8_pv32(dvA, pa, dd);
    wg_dk8(dkA, dsd, qd);
    wg_dq8(dq, dsd, kdA, 1);
    wg_commit();
    if (UB) {
      wg_wait<1>();
      pin(s);
      pin(dp);
      elements(Flag<MB>{}, qt, kB_lo, kB_hi, sl);
      wg_wait<0>();
      pin(pa);
      pin(dvA);
      pin(dkA);
      pin(dq);
      to_a<1, 4, 8>(pa, s);
      store_ds8(dsoffB, dp);
      proxy_fence();
      wg_bar(wg);
      wg_fence();
      n8_ss64(s, kdA, nq);  // block A's products of query tile i + 1 first
      n8_ss64(dp, vdA, nq + SWT32 / 16);
      wg_commit();
      wg_fence();
      n8_pv32(dvB, pa, dd);
      wg_dk8(dkB, dsdB, qd);
      wg_dq8(dq, dsdB, kdB, 0);
      wg_commit();
      wg_wait<1>();  // next scores complete; block B's products continue into the next step
      pin(s);
      pin(dp);
    } else {
      wg_fence();
      n8_ss64(s, kdA, nq);
      n8_ss64(dp, vdA, nq + SWT32 / 16);
      wg_commit();
      wg_wait<0>();
      pin(dvA);
      pin(dkA);
      pin(dq);
      pin(pa);
      pin(s);
      pin(dp);
    }
    stage = next;
  };
  if constexpr (KX == 2) {
    step2(0, Rows<true, true>{}, Rows<false, false>{}, Flag<false>{});
    bool tailb = false;
    if (n > 1) {
      if (n == 2) {
        step2(1, Rows<true, true>{}, Rows<true, true>{}, Flag<false>{});
      } else {
        step2(1, Rows<true, false>{}, Rows<true, true>{}, Flag<false>{});
        for (int i = 2; i < n - 1; ++i) step2(i, Rows<true, false>{}, Rows<true, false>{}, Flag<true>{});
        step2(n - 1, Rows<true, true>{}, Rows<true, true>{}, Flag<true>{});
      }
      tailb = true;
    }
    if (tailb) {
      wg_wait<0>();
      pin(dvB);
      pin(dkB);
      pin(dq);
      pin(pa);
    }
    reduce_dq(kA + n - 1, dq);
  } else {
  step(0, Rows<true, true>{}, Rows<false, false>{});
  if (n > 1) {
    if (n == 2) {
      step(1, Rows<true, true>{}, Rows<true, true>{});
    } else {
      step(1, Rows<true, false>{}, Rows<true, true>{});
      for (int i = 2; i < n - 1; ++i) step(i, Rows<true, false>{}, Rows<true, false>{});
      step(n - 1, Rows<true, true>{}, Rows<true, true>{});
    }
  }
  }
  if (KX == 1) reduce_dq(qtp, dqp);
  if (KX == 6 || KX == 8 || KX == 9 || KX == 10) {
    proxy_fence();
    wg_bar(wg);
    issue_dq();
    if (tid == 0) {
      bulk_wait_all();
      asm volatile("fence.proxy.async.global;\n" ::: "memory");
    }
  }
  cp_wait0();
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int d = 8 * c + 2 * (lane & 3);
    st_bf16x2_if(kA_lo < t, a.dk + (long long)kA_lo * a.ldk.sr + d, dkA[c][0] * a.scale, dkA[c][1] * a.scale);
    st_bf16x2_if(kA_hi < t, a.dk + (long long)kA_hi * a.ldk.sr + d, dkA[c][2] * a.scale, dkA[c][3] * a.scale);
    st_bf16x2_if(kA_lo < t, a.dv + (long long)kA_lo * a.ldv.sr + d, dvA[c][0], dvA[c][1]);
    st_bf16x2_if(kA_hi < t, a.dv + (long long)kA_hi * a.ldv.sr + d, dvA[c][2], dvA[c][3]);
    st_bf16x2_if(kB_lo < t, a.dk + (long long)kB_lo * a.ldk.sr + d, dkB[c][0] * a.scale, dkB[c][1] * a.scale);
    st_bf16x2_if(kB_hi < t, a.dk + (long long)kB_hi * a.ldk.sr + d, dkB[c][2] * a.scale, dkB[c][3] * a.scale);
    st_bf16x2_if(kB_lo < t, a.dv + (long long)kB_lo * a.ldv.sr + d, dvB[c][0], dvB[c][1]);
    st_bf16x2_if(kB_hi < t, a.dv + (long long)kB_hi * a.ldv.sr + d, dvB[c][2], dvB[c][3]);
  }
  wg_bar(wg);
}

// VAR 8 two-group forward: VAR 6's 128-row items, ring and statistics recipe with
// (1) the softmax scale folded into the exponent (row maxima of the raw scores; p = ex2(fmaf(s, scale log2 e, -m)));
// (2) no overflow alternative (|m scale log2 e| is far below FLT_MAX for BF16 operands of width 32);
// (3) a two-group software pipeline: block A's softmax runs while block B's products execute and vice versa
//     (commit order A, B; each block waits for all but the newest group). Same mathematics; not bitwise to VAR 6.
// FX 0: two-group pipeline; 1: every block section ends with a full wait (no cross-block overlap);
// 2: two-group pipeline with operand fences after each commit (CUTLASS warpgroup_fence_operand placement).
template <bool HB, int FX = 0>
PFLATLONG_DEV void forward32_v8(const Args& a, int pair, void* smem_v) {
  const int lane = threadIdx.x & 31, wg = wg_uniform(), w4 = (threadIdx.x >> 5) & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB8;   // 5-stage K/V ring [0, 40K); resident Q tiles A, B at 40K, 44K
  char* qsA = ring + 5 * ST32;
  char* qsB = qsA + SWT32;
  const int t = a.t, tA = 2 * pair, tB = tA + 1;
  const int n = HB ? tB + 1 : tA + 1, rowA = tA * 64, rowB = tB * 64;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const float c2 = a.scale * L2E;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t - rowA));
  count_if(tid == 0 && HB, a.work, min(64, t - rowB));
  count_if(tid == 0, a.work + 1, HB ? 2 : 1);
  ch.sw(qsA, a.q + (long long)rowA * sr, t - rowA);
  ch.sw(qsB, a.q + (long long)(HB ? rowB : rowA) * sr, HB ? t - rowB : 0);
  cp_commit();
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);  // Q tiles and tile 0 visible
  const uint32_t ring_desc = dlo(ring), qdA = dlo(qsA), qdB = qdA + SWT32 / 16;
  float oA[4][4], oB[4][4], sA[8][4], sB[8][4];
  uint32_t paA[4][1][4], paB[4][1][4];
  zero4(oA);
  zero4(oB);
  float mA0 = -INFINITY, mA1 = -INFINITY, lA0 = 0.f, lA1 = 0.f, mB0 = -INFINITY, mB1 = -INFINITY, lB0 = 0.f, lB1 = 0.f;
  const int rA_lo = rowA + 16 * w4 + (lane >> 2), rA_hi = rA_lo + 8, rB_lo = rowB + 16 * w4 + (lane >> 2), rB_hi = rB_lo + 8;
  wg_fence();
  n8_ss64(sA, qdA, ring_desc);
  wg_commit();
  if (HB) {
    wg_fence();
    n8_ss64(sB, qdB, ring_desc);
    wg_commit();
  }
  // Online softmax of one block's 64 x 64 raw scores: new row maxima, correction factors, probabilities, sums.
  auto softmax = [&](float (&s)[8][4], float& m0, float& m1, float& l0, float& l1, float& k0, float& k1) LATCH_SEG_INLINE {
    float x0 = fmaxf(s[0][0], s[0][1]), x1 = fmaxf(s[0][2], s[0][3]);
#pragma unroll
    for (int c = 1; c < 8; ++c) {
      x0 = fmaxf(x0, fmaxf(s[c][0], s[c][1]));
      x1 = fmaxf(x1, fmaxf(s[c][2], s[c][3]));
    }
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
    x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
    x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
    const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
    const float ml0 = n0 == -INFINITY ? 0.f : n0 * c2, ml1 = n1 == -INFINITY ? 0.f : n1 * c2;
    k0 = seg_ex2<true>(fmaf(m0, c2, -ml0));
    k1 = seg_ex2<true>(fmaf(m1, c2, -ml1));
    m0 = n0;
    m1 = n1;
    float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      s[c][0] = seg_ex2<true>(fmaf(s[c][0], c2, -ml0));
      s[c][1] = seg_ex2<true>(fmaf(s[c][1], c2, -ml0));
      s[c][2] = seg_ex2<true>(fmaf(s[c][2], c2, -ml1));
      s[c][3] = seg_ex2<true>(fmaf(s[c][3], c2, -ml1));
      sum0 += s[c][0] + s[c][1];
      sum1 += s[c][2] + s[c][3];
    }
    l0 = fmaf(l0, k0, sum0);
    l1 = fmaf(l1, k1, sum1);
  };
  int stage = 0;
  auto step = [&](int j, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UA = decltype(ra)::use, DA = decltype(ra)::diag, UB = decltype(rb)::use, DB = decltype(rb)::diag;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (its products completed in step j - 1)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    const uint32_t vd = ring_desc + (stage * ST32 + SWT32) / 16;
    const uint32_t nd = ring_desc + (j + 1 < n ? next : stage) * (ST32 / 16);  // past the last tile: discarded
    // block A: its scores of tile j and its previous output products are complete; block B's may still run
    if (HB) wg_wait<1>(); else wg_wait<0>();
    pin(sA);
    pin(oA);
    pin(paA);
    if (UA) {
      if (DA && FX != 6) diag_mask(sA, true, rA_lo, rA_hi, j * BT, lane);
      float k0, k1;
      softmax(sA, mA0, mA1, lA0, lA1, k0, k1);
      if (FX != 5) rescale32(oA, k0, k1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(paA, sA);
      if (FX != 7) {
        pin(oA);
        pin(paA);
      }
      wg_fence();
      if (FX == 3) wg_pv32(oA, paA, (uint64_t)vd | ((uint64_t)HI64 << 32)); else n8_pv32(oA, paA, vd);
      if (FX == 4) wg_ss64(sA, (uint64_t)qdA | ((uint64_t)HI64 << 32), (uint64_t)nd | ((uint64_t)HI64 << 32));
      else n8_ss64(sA, qdA, nd);
    }
    wg_commit();
    if (FX == 2) {
      pin(oA);
      pin(sA);
      pin(paA);
    }
    if (FX == 1) {
      wg_wait<0>();
      pin(oA);
      pin(sA);
      pin(paA);
    }
    if (HB) {
      wg_wait<1>();
      pin(sB);
      pin(oB);
      pin(paB);
      if (UB) {
        if (DB && FX != 6) diag_mask(sB, true, rB_lo, rB_hi, j * BT, lane);
        float k0, k1;
        softmax(sB, mB0, mB1, lB0, lB1, k0, k1);
        if (FX != 5) rescale32(oB, k0, k1);
        count_if(tid == 0, a.work + 2, 1);
        to_a<1, 4, 8>(paB, sB);
        if (FX != 7) {
          pin(oB);
          pin(paB);
        }
        wg_fence();
        if (FX == 3) wg_pv32(oB, paB, (uint64_t)vd | ((uint64_t)HI64 << 32)); else n8_pv32(oB, paB, vd);
        if (FX == 4) wg_ss64(sB, (uint64_t)qdB | ((uint64_t)HI64 << 32), (uint64_t)nd | ((uint64_t)HI64 << 32));
        else n8_ss64(sB, qdB, nd);
      }
      wg_commit();
      if (FX == 2) {
        pin(oB);
        pin(sB);
        pin(paB);
      }
      if (FX == 1) {
        wg_wait<0>();
        pin(oB);
        pin(sB);
        pin(paB);
      }
    }
    stage = next;
  };
  if (HB) {
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<true, false>{});
    step(tA, Rows<true, true>{}, Rows<true, false>{});
    step(tB, Rows<false, false>{}, Rows<true, true>{});
  } else {
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<false, false>{});
    step(tA, Rows<true, true>{}, Rows<false, false>{});
  }
  wg_wait<0>();
  pin(oA);
  pin(oB);
  pin(sA);
  pin(sB);
  cp_wait0();
  auto store = [&](float (&o)[4][4], float m0, float m1, float l0, float l1, int r_lo, int r_hi) LATCH_SEG_INLINE {
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, off);
      l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;
    l0 = empty0 ? 1.f : l0;
    l1 = empty1 ? 1.f : l1;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = 8 * c + 2 * (lane & 3);
      const float y0 = div_row(o[c][0], l0), y1 = div_row(o[c][1], l0);
      const float y2 = div_row(o[c][2], l1), y3 = div_row(o[c][3], l1);
      st_bf16x2_if(r_lo < t, a.o + (long long)r_lo * a.lo.sr + d, y0, y1);
      st_bf16x2_if(r_hi < t, a.o + (long long)r_hi * a.lo.sr + d, y2, y3);
    }
    const float lse_lo = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * c2) + log2f(l0);
    const float lse_hi = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * c2) + log2f(l1);
    st_f32_if((lane & 3) == 0 && r_lo < t, a.lse2 + r_lo, lse_lo);
    st_f32_if((lane & 3) == 0 && r_hi < t, a.lse2 + r_hi, lse_hi);
  };
  store(oA, mA0, mA1, lA0, lA1, rA_lo, rA_hi);
  if (HB) store(oB, mB0, mB1, lB0, lB1, rB_lo, rB_hi);
  wg_bar(wg);
}

// Folded-scale online softmax statistics of one 64 x 64 block of raw scores (FX 8): row maxima of the raw dots,
// correction factors, probabilities and per-thread row sums; the caller rescales its output.
template <bool LAZY = false>
PFLATLONG_DEV bool stats9(float (&s)[8][4], float& m0, float& m1, float& l0, float& l1, float& c0, float& c1, float c2) {
  float x0 = fmaxf(s[0][0], s[0][1]), x1 = fmaxf(s[0][2], s[0][3]);
#pragma unroll
  for (int n = 1; n < 8; ++n) {
    x0 = fmaxf(x0, fmaxf(s[n][0], s[n][1]));
    x1 = fmaxf(x1, fmaxf(s[n][2], s[n][3]));
  }
  x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 1));
  x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 1));
  x0 = fmaxf(x0, __shfl_xor_sync(0xffffffffu, x0, 2));
  x1 = fmaxf(x1, __shfl_xor_sync(0xffffffffu, x1, 2));
  bool move = true;
  if (LAZY) move = __any_sync(0xffffffffu, !((x0 - m0) * c2 <= 8.f) || !((x1 - m1) * c2 <= 8.f));
  float ml0, ml1;
  if (move) {
    const float n0 = fmaxf(m0, x0), n1 = fmaxf(m1, x1);
    ml0 = n0 == -INFINITY ? 0.f : n0 * c2;
    ml1 = n1 == -INFINITY ? 0.f : n1 * c2;
    c0 = seg_ex2<true>(fmaf(m0, c2, -ml0));
    c1 = seg_ex2<true>(fmaf(m1, c2, -ml1));
    m0 = n0;
    m1 = n1;
  } else {
    ml0 = m0 * c2;
    ml1 = m1 * c2;
    c0 = c1 = 1.f;
  }
  float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
  for (int n = 0; n < 8; ++n) {
    s[n][0] = seg_ex2<true>(fmaf(s[n][0], c2, -ml0));
    s[n][1] = seg_ex2<true>(fmaf(s[n][1], c2, -ml0));
    s[n][2] = seg_ex2<true>(fmaf(s[n][2], c2, -ml1));
    s[n][3] = seg_ex2<true>(fmaf(s[n][3], c2, -ml1));
    sum0 += s[n][0] + s[n][1];
    sum1 += s[n][2] + s[n][3];
  }
  l0 = move ? fmaf(l0, c0, sum0) : l0 + sum0;
  l1 = move ? fmaf(l1, c1, sum1) : l1 + sum1;
  return move;
}
// Folded-scale forward (FX 8): VAR 6's forward schedule (which ptxas keeps asynchronous) with a folded-scale softmax:
// raw-score row maxima, p = ex2(fmaf(s, scale log2 e, -m scale log2 e)), no overflow alternative. Same mathematics.
// QF 0 (FX 8): Q register fragments as VAR 6; QF 1 (FX 9): Q tiles stay in shared memory and wgmma reads them (no
// fragment renewal); QF 2 (FX 10): QF 1 with a conditional rescale (the running maximum moves only when a row's
// maximum grows by more than 2^8 in the exponent; exact softmax, probabilities stay below 256).
template <int QF>
PFLATLONG_DEV void forward32_v9(const Args& a, int pair, void* smem_v) {
  constexpr bool FTZ = true;
  const float c2 = a.scale * L2E;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wg = warp >> 2, w4 = warp & 3, tid = threadIdx.x & 127;
  char* ring = align1024(smem_v) + wg * WGB8;
  char* qsA = ring + (QF != 0 ? 5 : 3) * ST32;
  char* qsB = qsA + SWT32;
  const int t = a.t, ntiles = (t + 63) / 64, tA = 2 * pair, tB = tA + 1;
  const bool hasB = tB < ntiles;
  const int n = hasB ? tB + 1 : tA + 1, rowA = tA * 64, rowB = tB * 64;
  const long long sr = a.lq.sr, tstride = 64 * sr;
  const Chunks32 ch(tid, sr);
  count_if(tid == 0, a.work, min(64, t - rowA));
  count_if(tid == 0 && hasB, a.work, min(64, t - rowB));
  count_if(tid == 0, a.work + 1, hasB ? 2 : 1);
  ch.sw(qsA, a.q + (long long)rowA * sr, t - rowA);
  ch.sw(qsB, a.q + (long long)(hasB ? rowB : rowA) * sr, hasB ? t - rowB : 0);
  cp_commit();
  auto issue = [&](int tile, int stg) LATCH_SEG_INLINE {
    const int tl = tile < n ? tile : n - 1, valid = tile < n ? t - tl * BT : 0;
    char* st = ring + stg * ST32;
    ch.sw(st, a.k + tl * tstride, valid);
    ch.sw(st + SWT32, a.v + tl * tstride, valid);
    cp_commit();
  };
  issue(0, 0);
  issue(1, 1);
  issue(2, 2);
  uint32_t qaA[2][4], qaB[2][4];
  if (!QF) {
    cp_wait_n<3>();
    wg_bar(wg);
    a_frags_sw32(qaA, qsA, 16 * w4, lane);
    a_frags_sw32(qaB, qsB, 16 * w4, lane);
  }
  cp_wait_n<2>();
  proxy_fence();
  wg_bar(wg);  // tile 0 visible; every warp holds its Q fragments (stage 3 may refill)
  const uint64_t ring_desc = sw32_desc(ring);
  const uint32_t rlo = dlo(ring), qlA = dlo(qsA), qlB = qlA + SWT32 / 16;
  float oA[4][4], oB[4][4], sA[8][4], sB[8][4];
  uint32_t paA[4][1][4], paB[4][1][4];
  zero4(oA);
  zero4(oB);
  float mA0 = -INFINITY, mA1 = -INFINITY, lA0 = 0.f, lA1 = 0.f, mB0 = -INFINITY, mB1 = -INFINITY, lB0 = 0.f, lB1 = 0.f;
  const int rA_lo = rowA + 16 * w4 + (lane >> 2), rA_hi = rA_lo + 8, rB_lo = rowB + 16 * w4 + (lane >> 2), rB_hi = rB_lo + 8;
  wg_fence();
  if (QF) {
    n8_ss64(sA, qlA, rlo);
    n8_ss64(sB, qlB, rlo);
  } else {
    wg_nt32(sA, qaA, ring_desc);
    wg_nt32(sB, qaB, ring_desc);
  }
  wg_commit();
  wg_wait<0>();
  pin(sA);
  pin(sB);
  if (!QF) {
    if (!QF) pin(qaA);
    if (!QF) pin(qaB);
  }
  int stage = 0;
  auto step = [&](int j, auto ra, auto rb) LATCH_SEG_INLINE {
    constexpr bool UA = decltype(ra)::use, DA = decltype(ra)::diag, UB = decltype(rb)::use, DB = decltype(rb)::diag;
    cp_wait_n<1>();
    proxy_fence();
    wg_bar(wg);  // tile j + 1 visible; stage of tile j + 3 last held tile j - 2 (complete)
    const int next = stage_add(stage, 1);
    issue(j + 3, stage_add(stage, 3));
    const uint64_t vd = desc_add(ring_desc, stage * ST32 + SWT32);
    const uint64_t nd = desc_add(ring_desc, (j + 1 < n ? next : stage) * ST32);
    const uint32_t nlo = rlo + (j + 1 < n ? next : stage) * (ST32 / 16);  // past the last tile: discarded
    float c0, c1;
    if (UA) {
      if (DA) diag_mask(sA, true, rA_lo, rA_hi, j * BT, lane);
      if (stats9<(QF == 2)>(sA, mA0, mA1, lA0, lA1, c0, c1, c2)) rescale32(oA, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(paA, sA);
      wg_fence();
      wg_pv32(oA, paA, vd);
    }
    wg_fence();
    if (QF) n8_ss64(sA, qlA, nlo); else wg_nt32(sA, qaA, nd);
    wg_commit();
    if (QF == 4) {
      wg_wait<1>();  // block B's previous output products and scores complete (block A's new group may run)
      pin(oB);
      pin(paB);
      pin(sB);
    }
    if (UB) {
      if (DB) diag_mask(sB, true, rB_lo, rB_hi, j * BT, lane);
      if (stats9<(QF == 2)>(sB, mB0, mB1, lB0, lB1, c0, c1, c2)) rescale32(oB, c0, c1);
      count_if(tid == 0, a.work + 2, 1);
      to_a<1, 4, 8>(paB, sB);
      wg_fence();
      wg_pv32(oB, paB, vd);
    }
    wg_fence();
    if (QF) n8_ss64(sB, qlB, nlo); else wg_nt32(sB, qaB, nd);
    wg_commit();
    if (QF == 4) {
      wg_wait<1>();  // block A's output and next scores complete; block B's continue into the next step
      pin(oA);
      pin(paA);
      pin(sA);
    } else {
      wg_wait<0>();
      pin(oA);
      pin(oB);
      pin(paA);
      pin(paB);
      pin(sA);
      pin(sB);
      if (!QF) pin(qaA);
      if (!QF) pin(qaB);
    }
    stage = next;
  };
  if (hasB) {
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<true, false>{});
    step(tA, Rows<true, true>{}, Rows<true, false>{});
    step(tB, Rows<false, false>{}, Rows<true, true>{});
  } else {  // block B lies past the segment end: no products, FP32 work or counts for it
    for (int j = 0; j < tA; ++j) step(j, Rows<true, false>{}, Rows<false, false>{});
    step(tA, Rows<true, true>{}, Rows<false, false>{});
  }
  if (QF == 4) {
    wg_wait<0>();
    pin(oA);
    pin(oB);
    pin(sB);
    pin(paB);
  }
  cp_wait0();
  auto store = [&](float (&o)[4][4], float m0, float m1, float l0, float l1, int r_lo, int r_hi) LATCH_SEG_INLINE {
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      l0 += __shfl_xor_sync(0xffffffffu, l0, off);
      l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const bool empty0 = a.safe_rows && l0 == 0.f, empty1 = a.safe_rows && l1 == 0.f;
    l0 = empty0 ? 1.f : l0;
    l1 = empty1 ? 1.f : l1;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = 8 * c + 2 * (lane & 3);
      const float y0 = div_row(o[c][0], l0), y1 = div_row(o[c][1], l0);
      const float y2 = div_row(o[c][2], l1), y3 = div_row(o[c][3], l1);
      st_bf16x2_if(r_lo < t, a.o + (long long)r_lo * a.lo.sr + d, y0, y1);
      st_bf16x2_if(r_hi < t, a.o + (long long)r_hi * a.lo.sr + d, y2, y3);
    }
    const float lse_lo = empty0 ? INFINITY : (m0 == -INFINITY ? 0.f : m0 * c2) + log2f(l0);
    const float lse_hi = empty1 ? INFINITY : (m1 == -INFINITY ? 0.f : m1 * c2) + log2f(l1);
    st_f32_if((lane & 3) == 0 && r_lo < t, a.lse2 + r_lo, lse_lo);
    st_f32_if((lane & 3) == 0 && r_hi < t, a.lse2 + r_hi, lse_hi);
  };
  store(oA, mA0, mA1, lA0, lA1, rA_lo, rA_hi);
  store(oB, mB0, mB1, lB0, lB1, rB_lo, rB_hi);
  wg_bar(wg);
}

// VAR 8 backward prepare (all CTAs): prefix2 + work zeroing, then per (row, head) delta = rowsum(dO o O) in the
// [heads, rows] statistics layout and a zeroed FP32 dQ accumulator row (32 channels). Warp-cooperative: four lanes
// per (row, head) (16 B of dO and of O each, coalesced 512 B per warp access), four groups of 8 items in flight.
template <typename Index>
PFLATLONG_DEV void fused_bprep(const Index* offsets, int64_t* prefix, int64_t* work, int groups, int heads, int rows,
                               int bound, const __nv_bfloat16* dout, const __nv_bfloat16* o, float* delta, float* dqa) {
  prepare2(offsets, prefix, work, groups, heads, rows, bound, 6);
  constexpr int U = 4;
  const int lane = threadIdx.x & 31, c = lane & 3;
  const long long total = (long long)rows * heads;
  const long long gw = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = ((long long)gridDim.x * blockDim.x) >> 5;
  for (long long base = gw * 8 * U; base < total; base += nw * 8 * U) {
    uint4 w[U], z[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long i = base + 8 * u + (lane >> 2);
      const bool ok = i < total;
      w[u] = ok ? __ldcs(reinterpret_cast<const uint4*>(dout + i * W32) + c) : make_uint4(0, 0, 0, 0);
      z[u] = ok ? __ldcs(reinterpret_cast<const uint4*>(o + i * W32) + c) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long i = base + 8 * u + (lane >> 2);
      float acc = 0.f;
      acc = fmaf(bf_lo(w[u].x), bf_lo(z[u].x), acc); acc = fmaf(bf_hi(w[u].x), bf_hi(z[u].x), acc);
      acc = fmaf(bf_lo(w[u].y), bf_lo(z[u].y), acc); acc = fmaf(bf_hi(w[u].y), bf_hi(z[u].y), acc);
      acc = fmaf(bf_lo(w[u].z), bf_lo(z[u].z), acc); acc = fmaf(bf_hi(w[u].z), bf_hi(z[u].z), acc);
      acc = fmaf(bf_lo(w[u].w), bf_lo(z[u].w), acc); acc = fmaf(bf_hi(w[u].w), bf_hi(z[u].w), acc);
      acc += __shfl_xor_sync(0xffffffffu, acc, 1);
      acc += __shfl_xor_sync(0xffffffffu, acc, 2);
      if (i < total) {
        if (c == 0) delta[(long long)((int)i % heads) * rows + (int)i / heads] = acc;
        float4* zp = reinterpret_cast<float4*>(dqa + i * W32 + 8 * c);
        zp[0] = make_float4(0.f, 0.f, 0.f, 0.f);
        zp[1] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }
}
// KX 6..11: dQ (BF16, [rows, heads, 32]) = scale * accumulator ([heads, rows, 32]); eight channels per thread step.
PFLATLONG_DEV void fused_dqcvt_hr(const float* dqa, __nv_bfloat16* dq, int rows, int heads, float scale) {
  constexpr int U = 4;
  const long long eighths = (long long)rows * heads * 4, nt = (long long)gridDim.x * blockDim.x;
  for (long long base = (long long)blockIdx.x * blockDim.x + threadIdx.x; base < eighths; base += nt * U) {
    float4 x[U], y[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long i = base + nt * u;
      const bool ok = i < eighths;
      const int c8 = (int)(i & 3), rh = (int)(i >> 2), r = rh / heads, h = rh - r * heads;
      const float4* src = reinterpret_cast<const float4*>(dqa + ((long long)h * rows + r) * W32 + 8 * c8);
      x[u] = ok ? __ldcs(src) : make_float4(0.f, 0.f, 0.f, 0.f);
      y[u] = ok ? __ldcs(src + 1) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long long i = base + nt * u;
      if (i < eighths) {
        uint4 out;
        out.x = pack(x[u].x * scale, x[u].y * scale);
        out.y = pack(x[u].z * scale, x[u].w * scale);
        out.z = pack(y[u].x * scale, y[u].y * scale);
        out.w = pack(y[u].z * scale, y[u].w * scale);
        reinterpret_cast<uint4*>(dq)[i] = out;
      }
    }
  }
}
// Width 32: each warpgroup first copies the G+1 prefix and offsets into its own shared region (beyond the ring and
// statistics), so the per-item segment search reads shared memory instead of chaining dependent global loads. More
// than PREFIX_CACHE segments use the global arrays exactly as the width-64 path does.
constexpr int PREFIX_OFF = 44 * 1024, PREFIX_CACHE = 256;
static_assert(PREFIX_OFF >= NST32 * ST32 + NST32 * 2 * BT * 4 && PREFIX_OFF + 2 * 8 * (PREFIX_CACHE + 1) <= WGB,
              "prefix cache overlaps the width-32 ring or leaves the warpgroup region");
static_assert(PREFIX_OFF + 2 * 8 * (PREFIX_CACHE + 1) + 2 * 4 <= WGB,
              "claim slots leave the warpgroup region");
template <int Phase, int VARX, typename Index>
PFLATLONG_DEV void run32(Args base, const Index* offsets, const int64_t* prefix, void* smem) {
  constexpr int VAR = VARX & 15, FX = VARX >> 4;  // FX selects a VAR 8 forward or backward form
  constexpr int RB = VAR == 8 ? WGB8 : WGB, ROFF = VAR == 8 ? POFF8 : PREFIX_OFF;
  const int wg = threadIdx.x >> 7, tid = threadIdx.x & 127;
  long long* sp = reinterpret_cast<long long*>(align1024(smem) + wg * RB + ROFF);
  long long* so = sp + PREFIX_CACHE + 1;
  const int groups = base.groups;
  const bool cached = groups <= PREFIX_CACHE;
  // VAR 4: forward and dQ enumerate 128-row pairs (prefix2's second half); dK/dV keep 64-key tiles.
  // VAR 5: dK/dV also enumerate 128-key pairs.
  if constexpr ((VAR == 4 && Phase != 2) || VAR >= 5) prefix += groups + 1;
  // Shared slots are read only after this warpgroup barrier; the previous phase's last item ended with wg_bar.
#pragma unroll
  for (int k = 0; k < (PREFIX_CACHE + 128) / 128; ++k) {
    const int i = tid + 128 * k, src = i <= groups ? i : groups;
    const long long pv = prefix[src], ov = (long long)offsets[src];
    if (cached && i <= groups) { sp[i] = pv; so[i] = ov; }
  }
  // Claimed items: two shared ticket slots after the prefix cache; the slot for round r + 1 is written
  // while round r computes, and every item body ends with wg_bar, so a slot is rewritten only after all its readers.
#if defined(LATCH_SEG_ATTN_CLAIM)
  unsigned* tkt_slot = reinterpret_cast<unsigned*>(so + PREFIX_CACHE + 1);
  // Rank-major tickets (tkt_rank): rank r holds tile rank r (the existing largest-first order inside a segment) of
  // every segment with more than r tiles, segments ascending, heads ascending. tkt_cum[r] counts the items of ranks
  // below r. Ticket t maps to the segment-major index prefix[g] + r * heads + h of the same item, so the set of
  // items is unchanged; only their claim order differs.
  long long* tkt_cum = reinterpret_cast<long long*>(tkt_slot + 2);
  constexpr int TKT_RANKS = (RB - ROFF - 2 * 8 * (PREFIX_CACHE + 1) - 8) / 8 - 1;
  const bool tkt_claimed = base.tkt_claim != nullptr;
  int tkt_rmax = 0;
  // KX 7..11: the dK/dV/dQ items take tickets in segment-major order (each segment's largest items first), so
  // concurrently running items share segments and their FP32 dQ accumulator rows stay resident in L2.
  if (tkt_claimed && base.tkt_rank && cached && !(VAR == 8 && Phase == 2 && (FX == 7 || FX == 8 || FX == 9 || FX == 10 || FX == 11))) {
    wg_bar(wg);   // the prefix cache is complete
    for (int g = 0; g < groups; ++g) {
      const int n = (int)((sp[g + 1] - sp[g]) / base.heads);
      tkt_rmax = n > tkt_rmax ? n : tkt_rmax;
    }
    if (tkt_rmax <= TKT_RANKS) {
      for (int r = tid; r < tkt_rmax; r += 128) {
        int c = 0;
        for (int g = 0; g < groups; ++g) c += (sp[g + 1] - sp[g]) > (long long)r * base.heads;
        tkt_cum[r + 1] = (long long)c * base.heads;
      }
      wg_bar(wg);
      if (tid == 0) { tkt_cum[0] = 0; for (int r = 0; r < tkt_rmax; ++r) tkt_cum[r + 1] += tkt_cum[r]; }
    } else {
      tkt_rmax = 0;    // more ranks than the shared table holds: ascending tickets
    }
  }
  auto tkt_next = [&]() -> unsigned {
    const unsigned t = tkt_ticket(base.tkt_claim);
    if constexpr (VAR == 8 && Phase == 2 && FX == 11) {
      // KX 11: chunks of 16 segments in order; inside a chunk, rank-major (every segment's largest items
      // first). Concurrent items stay within ~one chunk (L2-resident dQ accumulator rows); the tail is rank-major.
      if (cached && (long long)t < prefix[groups]) {
        constexpr int CH = 16;
        long long rem = t;
        int g0 = 0;
        for (;;) {
          const int g1 = g0 + CH < groups ? g0 + CH : groups;
          const long long tot = sp[g1] - sp[g0];
          if (rem < tot || g1 >= groups) break;
          rem -= tot;
          g0 = g1;
        }
        const int g1 = g0 + CH < groups ? g0 + CH : groups;
        for (int r = 0; r < 4096; ++r) {
          int cnt = 0;
          for (int g = g0; g < g1; ++g) cnt += (sp[g + 1] - sp[g]) > (long long)r * base.heads;
          const long long c = (long long)cnt * base.heads;
          if (rem < c) {
            long long j = rem / base.heads;
            for (int g = g0; g < g1; ++g)
              if ((sp[g + 1] - sp[g]) > (long long)r * base.heads) {
                if (j == 0) return (unsigned)(sp[g] + (long long)r * base.heads + rem % base.heads);
                --j;
              }
          }
          rem -= c;
        }
      }
      return t;
    }
    if (!tkt_rmax || (long long)t >= prefix[groups]) return t;
    int lo = 0, hi = tkt_rmax - 1;             // the rank r with cum[r] <= t < cum[r + 1]
    while (lo < hi) { const int mid = (lo + hi + 1) / 2; if (tkt_cum[mid] <= (long long)t) lo = mid; else hi = mid - 1; }
    const long long rem = (long long)t - tkt_cum[lo];
    long long j = rem / base.heads;
    int g = 0;
    for (;; ++g) if ((sp[g + 1] - sp[g]) > (long long)lo * base.heads) { if (j == 0) break; --j; }
    return (unsigned)(sp[g] + (long long)lo * base.heads + rem % base.heads);
  };
  if constexpr (VAR == 8 && ((Phase == 2 && FX == 3) || (Phase == 0 && FX == 11))) {  // diagnostic: one warpgroup
    if (wg == 1) return;
  }
  if (tkt_claimed && tid == 0) tkt_slot[0] = tkt_next();
#endif
  wg_bar(wg);
  const int64_t total = prefix[groups];
  for (long long round=0;; ++round) {
#if defined(LATCH_SEG_ATTN_CLAIM)
    long long idx;
    if (tkt_claimed) {
      if (base.tkt_ahead) {
        idx = (long long)tkt_slot[round & 1];
        if (tid == 0 && idx < total) tkt_slot[(round + 1) & 1] = tkt_next();
      } else {
        if (tid == 0) tkt_slot[round & 1] = tkt_next();
        wg_bar(wg);
        idx = (long long)tkt_slot[round & 1];
      }
    } else {
      idx=snake2(round);
    }
#else
    const long long idx=snake2(round);
#endif
    if (idx>=total) break;
    int lo=0, hi=groups;
    while (lo<hi) {
      const int mid=(lo+hi)/2;
      const long long next = cached ? sp[mid+1] : prefix[mid+1];
      if (next<=idx) lo=mid+1; else hi=mid;
    }
    const int group=lo;
    const long long p0 = cached ? sp[group] : prefix[group], p1 = cached ? sp[group+1] : prefix[group+1];
    const long long b0 = cached ? so[group] : (long long)offsets[group];
    const long long b1 = cached ? so[group+1] : (long long)offsets[group+1];
    int begin=(int)b0, length=(int)(b1-b0);
    int local=(int)(idx-p0);
    if constexpr (VAR==8) {  // broadcast so ptxas sees warp-uniform item control (uniform descriptors)
      begin=__shfl_sync(0xffffffffu, begin, 0);
      length=__shfl_sync(0xffffffffu, length, 0);
      local=__shfl_sync(0xffffffffu, local, 0);
    }
    const int head=local%base.heads;
    // VAR 2: forward/dQ query tiles run largest first within a segment (query tile q has q + 1 key tiles), so the
    // phase tail holds the shortest items; dK/dV key tiles already run largest first. Items are independent: only
    // their order changes, never their arithmetic. VAR 3 keeps the width-64 order.
    int tile=((VAR==2 || VAR>=4) && Phase!=2) ? (int)((p1-p0)/base.heads) - 1 - local/base.heads
                                              : local/base.heads;
    if constexpr (VAR==8) tile=__shfl_sync(0xffffffffu, tile, 0);
    Args a=base;
    a.t=length; a.bh=1; a.safe_rows=1;
    const long long operand=(long long)begin*base.heads*W32+head*W32;
    const long long stat=(long long)head*base.rows+begin;
    a.q+=operand; a.k+=operand; a.v+=operand; a.o+=operand; a.lse2+=stat;
    a.lq=a.lk=a.lv=a.lo=a.ld=a.ldq=a.ldk=a.ldv={0,0,(long long)base.heads*W32};
    a.work+=(group*base.heads+head)*(Phase==0?3:6);
    if (Phase!=0) { a.dout+=operand; a.delta+=stat; a.dq+=operand; a.dk+=operand; a.dv+=operand; }
    if (a.dqa != nullptr) a.dqa += (VAR == 8 && Phase == 2 && (FX == 6 || FX == 7 || FX == 8 || FX == 9 || FX == 10 || FX == 11)) ? ((long long)head*base.rows + begin)*W32
                                                                      : operand;
    if constexpr (VAR==8) {  // VAR 6 forward items, single-pass dK/dV/dQ items (no dQ item phase)
      if constexpr (Phase==0) {
        if constexpr (FX>=8 && FX<=10) forward32_v9<FX - 8>(a,tile,smem);
        else if constexpr (FX==11) forward32_v9<1>(a,tile,smem);   // diagnostic: one working warpgroup
        else if constexpr (FX==12) forward32_v9<4>(a,tile,smem);
        else if ((2 * tile + 1) * 64 < length) forward32_v8<true, FX>(a,tile,smem); else forward32_v8<false, FX>(a,tile,smem);
      }
      else if constexpr (Phase==2) kv32_v8<FX == 3 ? 0 : FX == 7 ? 6 : FX == 11 ? 10 : FX>(a,tile,smem);  // FX 8: KX 7 without the bulk reductions  // FX 4: timing diagnostic
      else __trap();
    } else if constexpr (VAR>=5) {  // VAR 6: VAR 5 with the FTZ exponential
      if constexpr (Phase==0) forward32_v3<VAR==6>(a,tile,smem);
      else if constexpr (Phase==1) dq32_v3<VAR==6>(a,tile,smem);
      else kv32_v3<VAR==6>(a,tile,smem);
    } else if constexpr (VAR==4) {
      if constexpr (Phase==0) forward32_v3<false>(a,tile,smem);
      else if constexpr (Phase==1) dq32_v3<false>(a,tile,smem);
      else kv32_v2(a,tile,smem);
    } else if constexpr (VAR>=2) {
      if constexpr (Phase==0) forward32_v2(a,tile,smem);
      else if constexpr (Phase==1) dq32_v2(a,tile,smem);
      else kv32_v2(a,tile,smem);
    } else {
      if constexpr (Phase==0) forward32<VAR>(a,tile,smem);
      else if constexpr (Phase==1) dq32<VAR>(a,tile,smem);
      else kv32<VAR>(a,tile,smem);
    }
    if (base.tr_cnt != nullptr) {
      // Tile readiness: this item's rows of its head are stored by this warpgroup; publish rows x W32 elements.
      constexpr int RT = ((VAR == 4 && Phase != 2) || VAR >= 5) ? 128 : 64;
      wg_bar(wg);
      if (tid == 0) {
        const int rows_ = length - tile * RT < RT ? length - tile * RT : RT;
        if (base.tr_delay) __nanosleep((unsigned)base.tr_delay);
        if (rows_ > 0) tr_publish(base.tr_cnt, base.tr_rb, (long long)begin + (long long)tile * RT, rows_, W32);
      }
    }
  }
}

template <int Phase, int W, int VAR, typename Index>
PFLATLONG_DEV void run(Args base, const Index* offsets, const int64_t* prefix, void* smem) {
  static_assert(W == 32 || W == 64, "packed segmented attention supports head widths 32 and 64");
  if constexpr (W == 32) {
    run32<Phase, VAR>(base, offsets, prefix, smem);
  } else {
  if (base.tr_cnt != nullptr) __trap();     // tile-ready row publication exists only for the width-32 bodies
  const int64_t total=prefix[base.groups];
  for (long long round=0;; ++round) {
    const long long idx=snake2(round);
    if (idx>=total) break;
    int lo=0, hi=base.groups;
    while (lo<hi) { const int mid=(lo+hi)/2; if (prefix[mid+1]<=idx) lo=mid+1; else hi=mid; }
    const int group=lo, begin=(int)offsets[group], length=(int)(offsets[group+1]-offsets[group]);
    const int local=(int)(idx-prefix[group]), head=local%base.heads, tile=local/base.heads;
    Args a=base;
    a.t=length; a.bh=1; a.safe_rows=1;
    const long long operand=(long long)begin*base.heads*64+head*64;
    const long long stat=(long long)head*base.rows+begin;
    a.q+=operand; a.k+=operand; a.v+=operand; a.o+=operand; a.lse2+=stat;
    a.lq=a.lk=a.lv=a.lo=a.ld=a.ldq=a.ldk=a.ldv={0,0,(long long)base.heads*64};
    a.work+=(group*base.heads+head)*(Phase==0?3:6);
    if (Phase==0) forward_item<1,false>(a,0,tile,smem);
    else {
      a.dout+=operand; a.delta+=stat; a.dq+=operand; a.dk+=operand; a.dv+=operand;
      if (Phase==1) dq_item<1,false>(a,0,tile,smem);
      else kv_item<1,false>(a,0,tile,smem);
    }
  }  }  // W == 64
}
template <int W=64, int VAR=0, typename Index> PFLATLONG_DEV void run_forward(Args a,const Index* offsets,const int64_t* prefix,void* smem) { run<0,W,VAR>(a,offsets,prefix,smem); }
template <int W=64, int VAR=0, typename Index> PFLATLONG_DEV void run_dq(Args a,const Index* offsets,const int64_t* prefix,void* smem) { run<1,W,VAR>(a,offsets,prefix,smem); }
template <int W=64, int VAR=0, typename Index> PFLATLONG_DEV void run_kv(Args a,const Index* offsets,const int64_t* prefix,void* smem) { run<2,W,VAR>(a,offsets,prefix,smem); }
} // namespace latch_segmented
