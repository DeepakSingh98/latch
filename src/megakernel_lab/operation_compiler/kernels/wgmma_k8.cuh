// LATCH-MOE-EMA-FAST: warpgroup execution of the qualified mma.sync m16n8k8 readout arithmetic.
//
// Every k8 group that latch_mma_k8::tile issues as one mma.sync.m16n8k8 into an output's FP32 accumulator is
// issued here as one wgmma.mma_async.m64nNk16 whose register A operand holds that k8 group in one half and exact
// +0 in the other half ({lo,0} or {0,hi}); B is the full k16 shared-memory window. On H100 this instruction pair
// was measured bitwise identical to the m16n8k8 chain (signed zeros included) on 540M outputs over ten
// adversarial operand distributions, and for the wide m64n256 shape (LATCH-MOE-EMA-FAST receipts
// a01-k8-probe / a06-k8-probe-v2, scripts/moe_ema_fast/probes). Everything else is the reference contract:
// ascending (virtual) K from an FP32 +0 accumulator per partition, the reference's skipped k8 halves and k16
// blocks at edges (never executed with zero operands), initial-residue virtual-K staging, BF16
// RN(partial + widened previous) at bound partitions, BF16-rounded/widened serial weight partials summed in
// ascending order without an added initial zero, and one cvt.rn rounding at the store. Only the instruction
// issue, tile/warpgroup schedule and shared layouts change. A is staged as in mma_k8.cuh (popsc padded rows,
// ldmatrix); B uses the pops4b 128-byte swizzle layouts and descriptors. NaN/Inf operands are outside the
// measured scope (an Inf in the zero half's B window would meet a 0 product).
#pragma once
#include "mma_k8.cuh"
#include "pops4b_wg.cuh"
#include "pops4b_tma.cuh"

namespace latch_wgk8 {
using s101h::bf16;
using popsc::BArgs;

template <int TB> DEV void rs256(float (&d)[128], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                int sd = 1) {
  // sd = 0: D = A*B (wgmma scale-d false). Measured bitwise equal to accumulating that k8 group into an FP32 +0
  // (probes/k8_scale0.cu receipt a13); used for the first group of a partition instead of zeroing registers.
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %134, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, {%128,%129,%130,%131}, %132, p, 1, 1, %133;\n}\n"
               : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "n"(TB), "r"(sd));
}
template <int TB> DEV void rs128(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                int sd = 1) {
  // sd = 0: D = A*B (wgmma scale-d false). Measured bitwise equal to accumulating that k8 group into an FP32 +0
  // (probes/k8_scale0.cu receipt a13); used for the first group of a partition instead of zeroing registers.
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %70, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, p, 1, 1, %69;\n}\n"
               : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "n"(TB), "r"(sd));
}

template <int BN> struct Acc;
template <> struct Acc<256> {
  template <int TB> DEV static void run(float (&d)[128], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                       int sd = 1) {
    rs256<TB>(d, a0, a1, a2, a3, db, sd);
  }
};
template <> struct Acc<128> {
  template <int TB> DEV static void run(float (&d)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                      int sd = 1) {
    rs128<TB>(d, a0, a1, a2, a3, db, sd);
  }
};

template <int TB> DEV void rs192(float (&d)[96], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                int sd = 1) {
  // LATCH-MOE-EMA-FAST: the 192-column form of rs256/rs128 (same instruction family, scale-d operand).
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %102, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n192k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95}, {%96,%97,%98,%99}, %100, p, 1, 1, %101;\n}\n"
               : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db), "n"(TB), "r"(sd));
}
template <> struct Acc<192> {
  template <int TB> DEV static void run(float (&d)[96], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db,
                                      int sd = 1) {
    rs192<TB>(d, a0, a1, a2, a3, db, sd);
  }
};

// Initial-residue virtual-K staging into the swizzled B layouts (scalar: the shifted source is not vector aligned).
template <int R, bool RK>
DEV void stage_residue_sw(const bf16* X, long long sr, long long sk, int r0, int kk, bf16* S, int rlim, int klim,
                          int cut, int padding) {
  for (int index = threadIdx.x; index < R * 64; index += NTHREADS) {
    const int r = RK ? index / 64 : index % R;
    const int kc = RK ? index % 64 : index / R;
    const int virtual_k = kk + kc;
    const bool hole = virtual_k >= cut && virtual_k < cut + padding;
    const int source_k = virtual_k < cut ? virtual_k : virtual_k - padding;
    const int element = RK ? (r * 128 + (((kc >> 3) ^ (r & 7)) << 4)) / 2 + (kc & 7)
                           : ((r >> 6) * 8192 + kc * 128 + ((((r & 63) >> 3) ^ (kc & 7)) << 4)) / 2 + (r & 7);
    S[element] = (!hole && r0 + r < rlim && source_k >= 0 && source_k < klim)
      ? X[(long long)(r0 + r) * sr + (long long)source_k * sk]
      : __float2bfloat16_rn(0.f);
  }
}

// EPI: 0 plain, 1 bound input-VJP partitions (packed BF16 subtotals), 2 serial weight partials (FP32 total).
// K is the static logical contraction length: every stage's issue pattern is then compile-time, so no wgmma
// sits in a runtime branch (ptxas serializes all warpgroup MMAs of a function containing one that might diverge).
#ifndef LATCH_WGK8_DIAG
#define LATCH_WGK8_DIAG 0   // diagnostics only (wrong results): 1 no refill loads, 2 no wgmma, 3 no output stores
#endif
template <int BN, bool AK, bool BKC, int EPI, int PARTITION, bool INITIAL_RESIDUE, int K>
struct Tile {
  static constexpr int BM = 128, KS = 64, NST = 4, R = BN / 2;
  static constexpr int A_BYTES = AK ? BM * (KS + 8) * 2 : KS * (BM + 8) * 2;
  static constexpr int B_BYTES = BN * 128;
  static constexpr int STAGE = A_BYTES + B_BYTES;
  static constexpr int SHARED = NST * STAGE + 1024;
  static constexpr int TB = BKC ? 0 : 1;
  // Initial residue exactly as latch_mma_k8::tile (virtual zeros after the final partition's short k8 group).
  static constexpr int TAIL_START = INITIAL_RESIDUE ? ((K - 1) / PARTITION) * PARTITION : 0;
  static constexpr int SHORT_GROUP = INITIAL_RESIDUE ? (K - TAIL_START) % 8 : 0;
  static constexpr int CUT = INITIAL_RESIDUE ? TAIL_START + SHORT_GROUP : K;
  static constexpr int PAD = SHORT_GROUP == 0 ? 0 : 8 - SHORT_GROUP;
  static constexpr int EK = K + PAD, NK = (EK + KS - 1) / KS;
  static constexpr bool GUARD = NK * KS > EK;
  static constexpr int NFULL = GUARD ? NK - 1 : NK;
  static_assert(K > 0 && A_BYTES % 1024 == 0 && B_BYTES % 1024 == 0, "1024-byte aligned swizzle stages");
  static_assert(EPI == 0 || (PARTITION > 0 && PARTITION % 16 == 0), "partition boundaries at whole k16 blocks");
  static_assert(!INITIAL_RESIDUE || EPI == 1, "initial residue belongs to the bound input-VJP partitions");

  // Runs the whole K loop of one output tile. On return acc holds the final partition's accumulator; for
  // EPI 1 previous holds the packed BF16 subtotal, for EPI 2 total holds the ordered sum of all earlier
  // partitions (first: none yet).
  DEV static void run(const BArgs& g, int b, int m0, int n0, char* smem_raw, float (&acc)[R],
                      uint32_t (&previous)[EPI == 1 ? R / 2 : 1], float (&total)[EPI == 2 ? R : 1], bool& first) {
    const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
    const unsigned base0 = pops4b::sa(smem_raw), base = (base0 + 1023u) & ~1023u;
    char* sbase = smem_raw + (base - base0);
    const bf16* A = g.A + (long long)b * g.bsa;
    const bf16* B = g.B + (long long)b * g.bsb;
    const int row0 = wg * 64 + 16 * warp;
#pragma unroll
    for (int i = 0; i < R; ++i) acc[i] = 0.f;
    int fresh = 0;
    auto load = [&](int kk, int st) {
      bf16* As = reinterpret_cast<bf16*>(sbase + st * STAGE);
      if constexpr (INITIAL_RESIDUE && PAD > 0) {
        if (kk + KS > CUT) {
          latch_mma_k8::bstage_initial_residue<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As, g.M, K, CUT, PAD);
          stage_residue_sw<BN, BKC>(B, g.sbn, g.sbk, n0, kk, reinterpret_cast<bf16*>(sbase + st * STAGE + A_BYTES),
                                    g.N, K, CUT, PAD);
          return;
        }
      }
      popsc::bstage_edge<BM, AK, KS>(A, g.sam, g.sak, m0, kk, As, g.M, K);
      pops4b::stage<BN, BKC, true>(B, g.sbn, g.sbk, n0, kk, base + st * STAGE + A_BYTES, g.N, K);
    };
    auto boundary = [&]() {
      pops4b::wg_commit();
      pops4b::wg_wait<0>();
      pops4b::keep(acc);
      if constexpr (EPI == 1) {
#pragma unroll
        for (int j = 0; j < R / 4; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t w = previous[2 * j + h];
            const float p0 = __uint_as_float(w << 16), p1 = __uint_as_float(w & 0xffff0000u);
            previous[2 * j + h] = s101h::bpk(__fadd_rn(acc[4 * j + 2 * h], p0), __fadd_rn(acc[4 * j + 2 * h + 1], p1));
          }
        fresh = 1;
      } else if constexpr (EPI == 2) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
          total[i] = first ? x : __fadd_rn(total[i], x);
        }
        first = false;
      }
      fresh = 1;                                                     // next k8 group starts the partition from scale-d 0
      pops4b::wg_fence();
    };
    // ITC < 0: a full (unguarded) stage at runtime index it. ITC >= 0: the single guarded stage, index ITC.
    auto body = [&](auto itc, int it, uint32_t (&af)[4][4]) {
      constexpr int ITC = decltype(itc)::value;
      s101h::cpa_wait<NST - 2>();
      pops4b::async_fence();
      __syncthreads();                                             // stage it visible to every warp and wgmma
      const bf16* As = reinterpret_cast<const bf16*>(sbase + (it % NST) * STAGE);
      const unsigned Bs = base + (it % NST) * STAGE + A_BYTES;
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) {
        if (AK) s101h::bldsm4(af[kq], As + (row0 + (lane & 15)) * (KS + 8) + kq * 16 + (lane >> 4) * 8);
        else s101h::bldsm4t(af[kq], As + (kq * 16 + (lane & 7) + ((lane >> 4) << 3)) * (BM + 8) + row0 +
                                       (((lane >> 3) & 1) << 3));
      }
      pops4b::wg_fence();
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) {
        const int k16 = kq * 16;
        const uint64_t db = BKC ? pops4b::dk(Bs + kq * 32) : pops4b::dmn(Bs + kq * 2048);
        if constexpr (LATCH_WGK8_DIAG == 2) {
        } else if constexpr (ITC < 0) {
          { Acc<BN>::template run<TB>(acc, af[kq][0], af[kq][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          Acc<BN>::template run<TB>(acc, 0u, 0u, af[kq][2], af[kq][3], db);
        } else {
          // The reference's guarded stage: skip k16 blocks at/after effective K and absent second k8 halves.
          if constexpr (ITC * KS + 0 < EK) if (kq == 0) { Acc<BN>::template run<TB>(acc, af[0][0], af[0][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 0 + 8 < EK) if (kq == 0) Acc<BN>::template run<TB>(acc, 0u, 0u, af[0][2], af[0][3], db);
          if constexpr (ITC * KS + 16 < EK) if (kq == 1) { Acc<BN>::template run<TB>(acc, af[1][0], af[1][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 16 + 8 < EK) if (kq == 1) Acc<BN>::template run<TB>(acc, 0u, 0u, af[1][2], af[1][3], db);
          if constexpr (ITC * KS + 32 < EK) if (kq == 2) { Acc<BN>::template run<TB>(acc, af[2][0], af[2][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 32 + 8 < EK) if (kq == 2) Acc<BN>::template run<TB>(acc, 0u, 0u, af[2][2], af[2][3], db);
          if constexpr (ITC * KS + 48 < EK) if (kq == 3) { Acc<BN>::template run<TB>(acc, af[3][0], af[3][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 48 + 8 < EK) if (kq == 3) Acc<BN>::template run<TB>(acc, 0u, 0u, af[3][2], af[3][3], db);
        }
        if constexpr (EPI != 0) {
          const int end = it * KS + k16 + 16;
          if (end < EK && end % PARTITION == 0) boundary();
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();                                          // group it-1 done: its A registers and stage free
      pops4b::keep(acc);
      __syncthreads();
      if (LATCH_WGK8_DIAG != 1 && it + NST - 1 < NK) load((it + NST - 1) * KS, (it + NST - 1) % NST);
      s101h::cpa_commit();
    };
#pragma unroll
    for (int s = 0; s < NST - 1; ++s) {
      if (s < NK) load(s * KS, s);
      s101h::cpa_commit();
    }
    uint32_t af0[4][4], af1[4][4];
    using Full = std::integral_constant<int, -1>;
#pragma unroll 1
    for (int it = 0; it + 1 < NFULL; it += 2) {
      body(Full{}, it, af0);
      body(Full{}, it + 1, af1);
    }
    if constexpr (NFULL % 2 == 1) body(Full{}, NFULL - 1, af0);
    if constexpr (GUARD) {
      if constexpr ((NK - 1) % 2 == 0) body(std::integral_constant<int, NK - 1>{}, NK - 1, af0);
      else body(std::integral_constant<int, NK - 1>{}, NK - 1, af1);
    }
    pops4b::wg_wait<0>();
    pops4b::keep(acc);
    s101h::cpa_wait<0>();
    __syncthreads();                                                 // every stage read before the next tile's staging
  }
};

// ------------------------------------------------------------------------------------------------------------
// TMA-fed form. Thread 0 issues each stage as cp.async.bulk.tensor loads completing a stage mbarrier
// (pops4b_tma.cuh helpers, 128-byte swizzle, zero OOB fill), so operand traffic costs one instruction per box
// instead of one cp.async per 16 bytes and thread. Warps ldmatrix their A fragments from the swizzled A stage
// (K-major rows like pops4b A, or 64-wide M-major chunks). Initial-residue stages wholly after the hole load at
// the shifted source coordinate; the one stage containing the cut is filled element-wise by all threads
// (generic stores, proxy fence, plain arrive). Same per-output arithmetic and issue pattern as Tile.
DEV void ldsm4_s(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEV void ldsm4t_s(uint32_t (&r)[4], unsigned a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
DEV void bar_arrive(unsigned b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }

#ifndef LATCH_WGK8_TMA_SHIFT
#define LATCH_WGK8_TMA_SHIFT 0   // 1 (shifted K-major TMA coordinates) traps on H100: a17
#endif
// RAW (LATCH-MOE-EMA-100): stages wholly after the initial-residue hole read A through an unswizzled raw box that
// starts 8 elements before the stage's shifted source window (a 16-byte aligned TMA coordinate, 72 columns, zero
// OOB fill), and the register-A fragments take each k8 group's elements from that raw tile with the fixed shift
// 8 - PAD (two aligned 32-bit shared loads and one byte permute per register when the shift is odd). B (MN-major)
// takes the shifted source row coordinate directly. Every A/B value and zero is the one the element-wise staging
// stored; only how a stage reaches shared memory and registers changes.
template <int BN, bool AK, bool BKC, int EPI, int PARTITION, bool INITIAL_RESIDUE, int K, bool RAW = false>
struct TileT {
  using T = Tile<BN, AK, BKC, EPI, PARTITION, INITIAL_RESIDUE, K>;   // static contract values
  static constexpr int BM = 128, KS = 64, NST = 4, R = BN / 2;
  static constexpr bool RAWA = RAW && INITIAL_RESIDUE && T::PAD > 0 && AK && !BKC;
  static constexpr int RAW_PITCH = 72, RAW_SHIFT = 8 - T::PAD;
  static constexpr int A_TX = BM * 128, A_RAW_TX = BM * RAW_PITCH * 2;
  static constexpr int A_BYTES = RAWA ? A_RAW_TX : A_TX, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
  static constexpr int SHARED = NST * STAGE + 1024;
  static constexpr int TB = BKC ? 0 : 1;
  static constexpr int CUT = T::CUT, PAD = T::PAD, EK = T::EK, NK = T::NK, NFULL = T::NFULL;
  static constexpr bool GUARD = T::GUARD;
  static_assert(!INITIAL_RESIDUE || AK, "initial residue: K-major A (the input VJP)");
  static_assert(NST <= 8, "pops4b stage barriers");
  static_assert(A_BYTES % 1024 == 0, "1024-byte aligned B stage after A");

  DEV static bool hole(int it) {
    // LATCH_WGK8_TMA_SHIFT 0: every stage after the cut is filled element-wise (no shifted TMA coordinates),
    // except with RAW, where only stages overlapping the hole are.
    if constexpr (INITIAL_RESIDUE && PAD > 0)
      return it * KS + KS > CUT && (LATCH_WGK8_TMA_SHIFT == 0 && !RAWA || it * KS < CUT + PAD);
    return false;
  }
  DEV static bool raw(int it) {
    if constexpr (RAWA) return it * KS >= CUT + PAD;
    return false;
  }

  DEV static void run(const BArgs& g, const void* tA, const void* tB, int m0, int n0, char* smem_raw,
                      float (&acc)[R], uint32_t (&previous)[EPI == 1 ? R / 2 : 1], float (&total)[EPI == 2 ? R : 1],
                      bool& first, unsigned* uses, const void* tAraw = nullptr) {
    const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
    const unsigned base0 = pops4b::sa(smem_raw), base = (base0 + 1023u) & ~1023u;
    char* sbase = smem_raw + (base - base0);
    const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
    const int row0 = wg * 64 + 16 * warp;
#pragma unroll
    for (int i = 0; i < R; ++i) acc[i] = 0.f;
    int fresh = 0;
    auto issue = [&](int it) {                                       // thread 0 only; not for a hole stage
      const int st = it % NST;
      const unsigned dA = base + st * STAGE, dB = dA + A_BYTES, bar = bar0 + 8 * st;
      int kk = it * KS;
      if constexpr (INITIAL_RESIDUE && PAD > 0) {
        if (kk >= CUT + PAD) kk -= PAD;                              // wholly after the hole: shifted source
      }
      const bool rw = raw(it);
      pops4b::bar_expect(bar, (rw ? A_RAW_TX : A_TX) + B_BYTES);
      if constexpr (AK) {
        if (rw) pops4b::tma2d(dA, tAraw, it * KS - 8, m0, bar);      // aligned raw window (source kk - RAW_SHIFT)
        else pops4b::tma2d(dA, tA, kk, m0, bar);
      } else {
#pragma unroll
        for (int c = 0; c < BM / 64; ++c) pops4b::tma2d(dA + c * 8192, tA, m0 + 64 * c, kk, bar);
      }
      if constexpr (BKC) pops4b::tma2d(dB, tB, kk, n0, bar);
      else {
#pragma unroll
        for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tB, n0 + 64 * c, kk, bar);
      }
    };
    auto fill_hole = [&](int it) {                                   // every thread
      const int st = it % NST;
      char* p = sbase + st * STAGE;
      stage_residue_sw<BM, true>(g.A, g.sam, g.sak, m0, it * KS, reinterpret_cast<bf16*>(p), g.M, K, CUT, PAD);
      stage_residue_sw<BN, BKC>(g.B, g.sbn, g.sbk, n0, it * KS, reinterpret_cast<bf16*>(p + A_BYTES), g.N, K, CUT, PAD);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncthreads();
      if (tid == 0) bar_arrive(bar0 + 8 * st);
    };
    auto provide = [&](int it) {                                     // called by every thread, uniform
      if (hole(it)) fill_hole(it);
      else if (tid == 0) issue(it);
    };
    auto boundary = [&]() {
      pops4b::wg_commit();
      pops4b::wg_wait<0>();
      pops4b::keep(acc);
      if constexpr (EPI == 1) {
#pragma unroll
        for (int j = 0; j < R / 4; ++j)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t w = previous[2 * j + h];
            const float p0 = __uint_as_float(w << 16), p1 = __uint_as_float(w & 0xffff0000u);
            previous[2 * j + h] = s101h::bpk(__fadd_rn(acc[4 * j + 2 * h], p0), __fadd_rn(acc[4 * j + 2 * h + 1], p1));
          }
        fresh = 1;
      } else if constexpr (EPI == 2) {
#pragma unroll
        for (int i = 0; i < R; ++i) {
          const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
          total[i] = first ? x : __fadd_rn(total[i], x);
        }
        first = false;
      }
      fresh = 1;                                                     // next k8 group starts the partition from scale-d 0
      pops4b::wg_fence();
    };
    auto body = [&](auto itc, int it, uint32_t (&af)[4][4]) {
      constexpr int ITC = decltype(itc)::value;
      const int st = it % NST;
      pops4b::bar_wait(bar0 + 8 * st, uses[st] & 1u);
      uses[st] += 1u;
      const unsigned sA = base + st * STAGE, Bs = sA + A_BYTES;
      const bool rw = raw(it);
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) {
        if constexpr (AK) {
          if (rw) {
            // ldmatrix.x4 fragment order: register i holds rows +8*(i&1), columns +8*(i>>1) of the 16x16 block.
            const char* raw_tile = sbase + st * STAGE;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int row = row0 + (lane >> 2) + 8 * (i & 1);
              const int col = kq * 16 + 8 * (i >> 1) + 2 * (lane & 3) + RAW_SHIFT;
              const char* line = raw_tile + row * (RAW_PITCH * 2);
              if constexpr (RAW_SHIFT & 1) {
                const uint32_t w0 = *reinterpret_cast<const uint32_t*>(line + (col - 1) * 2);
                const uint32_t w1 = *reinterpret_cast<const uint32_t*>(line + (col + 1) * 2);
                af[kq][i] = __byte_perm(w0, w1, 0x5432);
              } else {
                af[kq][i] = *reinterpret_cast<const uint32_t*>(line + col * 2);
              }
            }
          } else {
            const int r = row0 + (lane & 15), chunk = kq * 2 + (lane >> 4);
            ldsm4_s(af[kq], sA + r * 128 + ((chunk ^ (r & 7)) << 4));
          }
        } else {
          const int k = kq * 16 + (lane & 7) + ((lane >> 4) << 3), m = row0 + (((lane >> 3) & 1) << 3);
          ldsm4t_s(af[kq], sA + (m >> 6) * 8192 + k * 128 + ((((m & 63) >> 3) ^ (k & 7)) << 4));
        }
      }
      pops4b::wg_fence();
#pragma unroll
      for (int kq = 0; kq < 4; ++kq) {
        const int k16 = kq * 16;
        const uint64_t db = BKC ? pops4b::dk(Bs + kq * 32) : pops4b::dmn(Bs + kq * 2048);
        if constexpr (LATCH_WGK8_DIAG == 2) {
        } else if constexpr (ITC < 0) {
          { Acc<BN>::template run<TB>(acc, af[kq][0], af[kq][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          Acc<BN>::template run<TB>(acc, 0u, 0u, af[kq][2], af[kq][3], db);
        } else {
          if constexpr (ITC * KS + 0 < EK) if (kq == 0) { Acc<BN>::template run<TB>(acc, af[0][0], af[0][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 0 + 8 < EK) if (kq == 0) Acc<BN>::template run<TB>(acc, 0u, 0u, af[0][2], af[0][3], db);
          if constexpr (ITC * KS + 16 < EK) if (kq == 1) { Acc<BN>::template run<TB>(acc, af[1][0], af[1][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 16 + 8 < EK) if (kq == 1) Acc<BN>::template run<TB>(acc, 0u, 0u, af[1][2], af[1][3], db);
          if constexpr (ITC * KS + 32 < EK) if (kq == 2) { Acc<BN>::template run<TB>(acc, af[2][0], af[2][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 32 + 8 < EK) if (kq == 2) Acc<BN>::template run<TB>(acc, 0u, 0u, af[2][2], af[2][3], db);
          if constexpr (ITC * KS + 48 < EK) if (kq == 3) { Acc<BN>::template run<TB>(acc, af[3][0], af[3][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
          if constexpr (ITC * KS + 48 + 8 < EK) if (kq == 3) Acc<BN>::template run<TB>(acc, 0u, 0u, af[3][2], af[3][3], db);
        }
        if constexpr (EPI != 0) {
          const int end = it * KS + k16 + 16;
          if (end < EK && end % PARTITION == 0) boundary();
        }
      }
      pops4b::wg_commit();
      pops4b::wg_wait<1>();                                          // group it-1 done: its A registers and stage free
      pops4b::keep(acc);
      __syncthreads();                                               // every warp finished reading stage it-1
      if (LATCH_WGK8_DIAG != 1 && it + NST - 1 < NK) provide(it + NST - 1);
    };
#pragma unroll
    for (int s = 0; s < NST - 1; ++s)
      if (s < NK) provide(s);
    uint32_t af0[4][4], af1[4][4];
    using Full = std::integral_constant<int, -1>;
#pragma unroll 1
    for (int it = 0; it + 1 < NFULL; it += 2) {
      body(Full{}, it, af0);
      body(Full{}, it + 1, af1);
    }
    if constexpr (NFULL % 2 == 1) body(Full{}, NFULL - 1, af0);
    if constexpr (GUARD) {
      if constexpr ((NK - 1) % 2 == 0) body(std::integral_constant<int, NK - 1>{}, NK - 1, af0);
      else body(std::integral_constant<int, NK - 1>{}, NK - 1, af1);
    }
    pops4b::wg_wait<0>();
    pops4b::keep(acc);
    __syncthreads();                                                 // ring free for the staged epilogue
  }
};

DEV void tma_bars_init() {
  __syncthreads();                                                   // earlier generic users of shared memory done
  if (threadIdx.x == 0) {
    const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
#pragma unroll
    for (int st = 0; st < 4; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    pops4b::proxy_fence();
  }
  __syncthreads();
}
DEV void tma_bars_release() {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
#pragma unroll
    for (int st = 0; st < 4; ++st) pops4b::bar_inval(bar0 + 8 * st);
  }
  __syncthreads();
}


// Bound-partition / plain products stored as BF16 (or exactly widened FP32), exactly as latch_mma_k8::tile does.
template <int BN, bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE, int K, bool ROUNDED_F32, bool TMA = false,
          bool RAW = false>
DEV void product_tile(const BArgs& g, int b, int m0, int n0, char* smem, const void* tA = nullptr,
                      const void* tB = nullptr, unsigned* uses = nullptr, const void* tAraw = nullptr) {
  constexpr int EPI = PARTITION > 0 ? 1 : 0;
  using T0 = Tile<BN, AK, BKC, EPI, PARTITION, INITIAL_RESIDUE, K>;
  using T1 = TileT<BN, AK, BKC, EPI, PARTITION, INITIAL_RESIDUE, K, RAW>;
  using T = std::conditional_t<TMA, T1, T0>;
  float acc[T::R];
  uint32_t previous[EPI == 1 ? T::R / 2 : 1] = {};
  float total[1];
  bool first = true;
  if constexpr (TMA) T1::run(g, tA, tB, m0, n0, smem, acc, previous, total, first, uses, tAraw);
  else T0::run(g, b, m0, n0, smem, acc, previous, total, first);
  if constexpr (EPI == 1) {
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t w = previous[2 * j + (q >> 1)];
        acc[4 * j + q] = __fadd_rn(acc[4 * j + q], __uint_as_float((q & 1) ? (w & 0xffff0000u) : (w << 16)));
      }
  }
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  const int mrow = m0 + wg * 64 + 16 * warp + gq;
  if constexpr (LATCH_WGK8_DIAG == 3) {
    if (acc[0] == 1234.5f) g.C[0] = __float2bfloat16_rn(acc[1]);   // keep the accumulators live
    return;
  }
#pragma unroll
  for (int j = 0; j < T::R / 4; ++j) {
    const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int mm = mrow + 8 * h;
      if (mm >= g.M || n >= g.N) continue;
      if constexpr (ROUNDED_F32) {
        float* Cf = reinterpret_cast<float*>(g.C) + (long long)b * g.bsc;
        const float x = __bfloat162float(__float2bfloat16_rn(acc[4 * j + 2 * h]));
        const float y = __bfloat162float(__float2bfloat16_rn(acc[4 * j + 2 * h + 1]));
        if (n + 1 < g.N && g.ldc % 2 == 0)
          *reinterpret_cast<float2*>(Cf + (long long)mm * g.ldc + n) = make_float2(x, y);
        else {
          Cf[(long long)mm * g.ldc + n] = x;
          if (n + 1 < g.N) Cf[(long long)mm * g.ldc + n + 1] = y;
        }
      }
    }
  }
  if constexpr (!ROUNDED_F32) {
    // The same cvt.rn BF16 values staged through the drained operand ring (Tile::run ends with every stage
    // consumed and no copy in flight), then written as whole contiguous row segments: 32-bit words where the
    // row segment is word aligned, BF16 elements otherwise. Store layout only; the bytes stored are unchanged.
    constexpr int PITCH = BN + 8;
    static_assert(128 * PITCH * 2 + 1024 <= T::SHARED, "staged tile fits the operand ring");
    const unsigned base0 = pops4b::sa(smem), base = (base0 + 1023u) & ~1023u;
    bf16* stile = reinterpret_cast<bf16*>(smem + (base - base0));
    const int lrow = wg * 64 + 16 * warp + gq;
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *reinterpret_cast<uint32_t*>(stile + (lrow + 8 * h) * PITCH + 8 * j + 2 * tq) =
            s101h::bpk(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1]);
    __syncthreads();
    bf16* C = g.C + (long long)b * g.bsc;
    const int rows = g.M - m0 < 128 ? g.M - m0 : 128, cols = g.N - n0 < BN ? g.N - n0 : BN;
    for (int r = tid >> 5; r < rows; r += 8) {
      bf16* dst = C + (long long)(m0 + r) * g.ldc + n0;
      const bf16* src = stile + r * PITCH;
      if (cols == BN && (reinterpret_cast<unsigned long long>(dst) & 3ull) == 0ull) {
#pragma unroll
        for (int w = lane; w < BN / 2; w += 32)
          reinterpret_cast<uint32_t*>(dst)[w] = reinterpret_cast<const uint32_t*>(src)[w];
      } else {
        for (int c = lane; c < cols; c += 32) dst[c] = src[c];
      }
    }
    __syncthreads();                                                 // staged tile read before the next tile's staging
  }
}

#ifndef LATCH_WGK8_PARTITION_BN
#define LATCH_WGK8_PARTITION_BN 256
#endif
// Drop-in for latch_mma_k8::phase_wide (grouped raster and per-output arithmetic unchanged); the fifth template
// argument is the static contraction length K (g.K must equal it). Tiles are 128x256 (bound partitions:
// LATCH_WGK8_PARTITION_BN). Inlined into the region: a warpgroup-MMA pipeline must not cross a call (C7510).
template <bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE, int K, bool ROUNDED_F32 = false,
          bool PARTITION_WIDE = true>
DEV void phase_wide(const BArgs g, float* smem, int rot = 0) {
  if (g.K != K) __trap();
  constexpr int BM = 128, BN = PARTITION > 0 ? LATCH_WGK8_PARTITION_BN : 256, GROUP = 8;
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, rot);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, per = tm * tn, ntask = g.nb * per;
  for (int task = slice.me; task < ntask; task += slice.count) {
    const int b = task / per, t = task % per, span = GROUP * tn, first = (t / span) * GROUP;
    const int size = tm - first < GROUP ? tm - first : GROUP, local = t % span;
    const int mi = first + local % size, nj = local / size;
    product_tile<BN, AK, BKC, PARTITION, INITIAL_RESIDUE, K, ROUNDED_F32>(g, b, mi * BM, nj * BN,
                                                                      reinterpret_cast<char*>(smem));
  }
}

// Drop-in for latch_mma_k8::phase_serial_sum: per 128x128 output tile, consecutive CHUNK-token k8 products from +0,
// each BF16-rounded and widened, summed in ascending chunk order into an FP32 total (no added initial zero).
template <bool AK, bool BKC, int CHUNK, int K>
DEV void phase_serial_sum(const BArgs g, float* smem) {
  constexpr int BM = 128, BN = 128, GROUP = 8;
  using T = Tile<BN, AK, BKC, 2, CHUNK, false, K>;
  if (g.K != K) __trap();
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, 0);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, span = GROUP * tn;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  for (int task = slice.me; task < tm * tn; task += slice.count) {
    const int first_m = (task / span) * GROUP, size = tm - first_m < GROUP ? tm - first_m : GROUP;
    const int local = task % span, m0 = (first_m + local % size) * BM, n0 = (local / size) * BN;
    float acc[T::R], total[T::R];
    uint32_t previous[1];
    bool first = true;
    T::run(g, 0, m0, n0, reinterpret_cast<char*>(smem), acc, previous, total, first);
#pragma unroll
    for (int i = 0; i < T::R; ++i) {                                 // final chunk
      const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
      total[i] = first ? x : __fadd_rn(total[i], x);
    }
    float* out = reinterpret_cast<float*>(g.C);
    const int mrow = m0 + wg * 64 + 16 * warp + gq;
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j) {
      const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = mrow + 8 * h;
        if (mm >= g.M || n >= g.N) continue;
        const long long where = (long long)mm * g.ldc + n;
        if (n + 1 < g.N && g.ldc % 2 == 0)
          *reinterpret_cast<float2*>(out + where) = make_float2(total[4 * j + 2 * h], total[4 * j + 2 * h + 1]);
        else {
          out[where] = total[4 * j + 2 * h];
          if (n + 1 < g.N) out[where + 1] = total[4 * j + 2 * h + 1];
        }
      }
    }
  }
}

// TMA-fed drop-ins (tensor maps of the A / B operand views: K-major dims {K, rows}, box {64, 128 | BN};
// M/N-major dims {rows, K}, box {64, 64}; BF16, 128-byte swizzle, zero OOB fill).
template <bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE, int K, bool ROUNDED_F32 = false,
          bool PARTITION_WIDE = true, bool RAW = false>
DEV void phase_wide_tma(const BArgs g, const void* tA, const void* tB, float* smem, int rot = 0,
                        const void* tAraw = nullptr) {
  if (g.K != K || g.nb != 1) __trap();
  constexpr int BM = 128, BN = PARTITION > 0 ? LATCH_WGK8_PARTITION_BN : 256, GROUP = 8;
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, rot);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, per = tm * tn;
  tma_bars_init();
  unsigned uses[4] = {0u, 0u, 0u, 0u};
  for (int task = slice.me; task < per; task += slice.count) {
    const int span = GROUP * tn, first = (task / span) * GROUP;
    const int size = tm - first < GROUP ? tm - first : GROUP, local = task % span;
    const int mi = first + local % size, nj = local / size;
    product_tile<BN, AK, BKC, PARTITION, INITIAL_RESIDUE, K, ROUNDED_F32, true, RAW>(
        g, 0, mi * BM, nj * BN, reinterpret_cast<char*>(smem), tA, tB, uses, tAraw);
  }
  tma_bars_release();
}

template <bool AK, bool BKC, int CHUNK, int K>
DEV void phase_serial_sum_tma(const BArgs g, const void* tA, const void* tB, float* smem) {
  constexpr int BM = 128, BN = 128, GROUP = 8;
  using T = TileT<BN, AK, BKC, 2, CHUNK, false, K>;
  if (g.K != K) __trap();
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, 0);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, span = GROUP * tn;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  tma_bars_init();
  unsigned uses[4] = {0u, 0u, 0u, 0u};
  for (int task = slice.me; task < tm * tn; task += slice.count) {
    const int first_m = (task / span) * GROUP, size = tm - first_m < GROUP ? tm - first_m : GROUP;
    const int local = task % span, m0 = (first_m + local % size) * BM, n0 = (local / size) * BN;
    float acc[T::R], total[T::R];
    uint32_t previous[1];
    bool first = true;
    T::run(g, tA, tB, m0, n0, reinterpret_cast<char*>(smem), acc, previous, total, first, uses);
#pragma unroll
    for (int i = 0; i < T::R; ++i) {                                 // final chunk
      const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
      total[i] = first ? x : __fadd_rn(total[i], x);
    }
    float* out = reinterpret_cast<float*>(g.C);
    const int mrow = m0 + wg * 64 + 16 * warp + gq;
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j) {
      const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = mrow + 8 * h;
        if (mm >= g.M || n >= g.N) continue;
        const long long where = (long long)mm * g.ldc + n;
        if (n + 1 < g.N && g.ldc % 2 == 0)
          *reinterpret_cast<float2*>(out + where) = make_float2(total[4 * j + 2 * h], total[4 * j + 2 * h + 1]);
        else {
          out[where] = total[4 * j + 2 * h];
          if (n + 1 < g.N) out[where + 1] = total[4 * j + 2 * h + 1];
        }
      }
    }
  }
  tma_bars_release();
}

// LATCH-MOE-EMA-100: a flat FP32 accumulation chain that starts from a stored total and continues past this
// product's partials: total = init; total = RN(total + widen(RN_bf16(partial_c))) for the CHUNK-token partials in
// ascending order (the same per-partial arithmetic as phase_serial_sum_tma); then total = RN(total + extra_e[i]) for
// the NEXTRA stored FP32 addends in their listed order; one FP32 store. Each output element's additions keep the
// chain's exact left-to-right order; only where the running value lives (registers) changes.
template <bool AK, bool BKC, int CHUNK, int K, int NEXTRA>
DEV void phase_serial_chain_tma(const BArgs g, const void* tA, const void* tB, float* smem, const float* init,
                                const float* const (&extra)[NEXTRA > 0 ? NEXTRA : 1]) {
  constexpr int BM = 128, BN = 128, GROUP = 8;
  using T = TileT<BN, AK, BKC, 2, CHUNK, false, K>;
  if (g.K != K) __trap();
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, 0);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, span = GROUP * tn;
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  tma_bars_init();
  unsigned uses[4] = {0u, 0u, 0u, 0u};
  for (int task = slice.me; task < tm * tn; task += slice.count) {
    const int first_m = (task / span) * GROUP, size = tm - first_m < GROUP ? tm - first_m : GROUP;
    const int local = task % span, m0 = (first_m + local % size) * BM, n0 = (local / size) * BN;
    const int mrow = m0 + wg * 64 + 16 * warp + gq;
    float acc[T::R], total[T::R];
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j) {
      const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = mrow + 8 * h;
        const bool row = mm < g.M;
        const long long where = (long long)mm * g.ldc + n;
        total[4 * j + 2 * h] = row && n < g.N ? init[where] : 0.f;
        total[4 * j + 2 * h + 1] = row && n + 1 < g.N ? init[where + 1] : 0.f;
      }
    }
    uint32_t previous[1];
    bool first = false;                                              // the chain's running value exists already
    T::run(g, tA, tB, m0, n0, reinterpret_cast<char*>(smem), acc, previous, total, first, uses);
#pragma unroll
    for (int i = 0; i < T::R; ++i)                                   // final chunk
      total[i] = __fadd_rn(total[i], __bfloat162float(__float2bfloat16_rn(acc[i])));
    float* out = reinterpret_cast<float*>(g.C);
#pragma unroll
    for (int j = 0; j < T::R / 4; ++j) {
      const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = mrow + 8 * h;
        if (mm >= g.M || n >= g.N) continue;
        const long long where = (long long)mm * g.ldc + n;
        const bool two = n + 1 < g.N;
        float t0 = total[4 * j + 2 * h], t1 = total[4 * j + 2 * h + 1];
#pragma unroll 4
        for (int e = 0; e < NEXTRA; ++e) {
          t0 = __fadd_rn(t0, __ldcs(extra[e] + where));
          if (two) t1 = __fadd_rn(t1, __ldcs(extra[e] + where + 1));
        }
        if (two && g.ldc % 2 == 0)
          *reinterpret_cast<float2*>(out + where) = make_float2(t0, t1);
        else {
          out[where] = t0;
          if (two) out[where + 1] = t1;
        }
      }
    }
  }
  tma_bars_release();
}

// ------------------------------------------------------------------------------------------------------------
// Cross-tile TMA pipeline. Thread 0 keeps NSTX stages in flight over the CTA's whole task sequence (global step
// q = task_i * NK + it), so a tile's first stages load during the previous tile's tail and epilogue. Every tile
// and every bound partition starts with wgmma scale-d = 0 (probe a13: identical to accumulating into FP32 +0),
// so no non-wgmma instruction defines an accumulator inside the pipeline. EPI 0/1 products stage their BF16
// tile in a dedicated region after the ring; EPI 2 (serial weight partials) stores FP32 totals from registers.
// Per-output arithmetic, skip pattern, residue handling and rounding are those of Tile / TileT.
template <int BN, bool AK, bool BKC, int EPI, int PARTITION, bool INITIAL_RESIDUE, int K, bool ROUNDED_F32, int NSTX>
struct Pipe {
  using T = Tile<BN, AK, BKC, EPI, PARTITION, INITIAL_RESIDUE, K>;
  static constexpr int BM = 128, KS = 64, R = BN / 2;
  static constexpr int A_BYTES = BM * 128, B_BYTES = BN * 128, STAGE = A_BYTES + B_BYTES;
  static constexpr int PITCH = BN + 8;
  static constexpr int STAGING = (EPI == 2 || ROUNDED_F32) ? 0 : ((BM * PITCH * 2 + 1023) / 1024) * 1024;
  static constexpr int SHARED = NSTX * STAGE + STAGING + 1024;
  static constexpr int TB = BKC ? 0 : 1;
  static constexpr int CUT = T::CUT, PAD = T::PAD, EK = T::EK, NK = T::NK, NFULL = T::NFULL;
  static constexpr bool GUARD = T::GUARD;
  static_assert(!INITIAL_RESIDUE || AK, "initial residue: K-major A (the input VJP)");
  static_assert(NSTX >= 2 && NSTX <= 8, "pops4b stage barriers");
  static_assert(EPI == 0 || (PARTITION > 0 && PARTITION % 16 == 0), "partition boundaries at whole k16 blocks");

  DEV static bool hole(int it) {
    if constexpr (INITIAL_RESIDUE && PAD > 0)
      return it * KS + KS > CUT && (LATCH_WGK8_TMA_SHIFT == 0 || it * KS < CUT + PAD);
    return false;
  }

  // where(task_i, m0, n0): the task's output tile; ntm: this CTA's task count.
  template <class Where, class Finish>
  DEV static void run(const BArgs& g, const void* tA, const void* tB, char* smem_raw, int ntm, Where where,
                      Finish finish) {
    const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3;
    const unsigned base0 = pops4b::sa(smem_raw), base = (base0 + 1023u) & ~1023u;
    char* sbase = smem_raw + (base - base0);
    const unsigned bar0 = pops4b::sa(&pops4b::tma_bars[0]);
    const int row0 = wg * 64 + 16 * warp;
    const int Q = ntm * NK;
    if (Q == 0) return;
    if (tid == 0) {
#pragma unroll
      for (int st = 0; st < NSTX; ++st) pops4b::bar_init(bar0 + 8 * st, 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      pops4b::proxy_fence();
    }
    __syncthreads();
    auto issue = [&](int q) {                                        // thread 0 only; not for a hole stage
      const int st = q % NSTX, it = q % NK;
      int m0, n0;
      where(q / NK, m0, n0);
      const unsigned dA = base + st * STAGE, dB = dA + A_BYTES, bar = bar0 + 8 * st;
      int kk = it * KS;
      if constexpr (INITIAL_RESIDUE && PAD > 0) {
        if (kk >= CUT + PAD) kk -= PAD;
      }
      pops4b::bar_expect(bar, STAGE);
      if constexpr (AK) pops4b::tma2d(dA, tA, kk, m0, bar);
      else {
#pragma unroll
        for (int c = 0; c < BM / 64; ++c) pops4b::tma2d(dA + c * 8192, tA, m0 + 64 * c, kk, bar);
      }
      if constexpr (BKC) pops4b::tma2d(dB, tB, kk, n0, bar);
      else {
#pragma unroll
        for (int c = 0; c < BN / 64; ++c) pops4b::tma2d(dB + c * 8192, tB, n0 + 64 * c, kk, bar);
      }
    };
    auto provide = [&](int q) {                                      // every thread, uniform
      const int it = q % NK;
      if (hole(it)) {
        int m0, n0;
        where(q / NK, m0, n0);
        char* p = sbase + (q % NSTX) * STAGE;
        stage_residue_sw<BM, true>(g.A, g.sam, g.sak, m0, it * KS, reinterpret_cast<bf16*>(p), g.M, K, CUT, PAD);
        stage_residue_sw<BN, BKC>(g.B, g.sbn, g.sbk, n0, it * KS, reinterpret_cast<bf16*>(p + A_BYTES), g.N, K, CUT, PAD);
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        if (tid == 0) bar_arrive(bar0 + 8 * (q % NSTX));
      } else if (tid == 0) {
        issue(q);
      }
    };
#pragma unroll 1
    for (int s = 0; s < NSTX - 1; ++s)
      if (s < Q) provide(s);
    uint32_t af0[4][4], af1[4][4];
#pragma unroll 1
    for (int task_i = 0; task_i < ntm; ++task_i) {
      float acc[R];
      uint32_t previous[EPI == 1 ? R / 2 : 1];
      float total[EPI == 2 ? R : 1];
      if constexpr (EPI == 1) {
#pragma unroll
        for (int i = 0; i < R / 2; ++i) previous[i] = 0u;
      }
      bool first = true;
      int fresh = 1;                                                 // tile start: scale-d 0
      auto boundary = [&]() {
        pops4b::wg_commit();
        pops4b::wg_wait<0>();
        pops4b::keep(acc);
        if constexpr (EPI == 1) {
#pragma unroll
          for (int j = 0; j < R / 4; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t w = previous[2 * j + h];
              const float p0 = __uint_as_float(w << 16), p1 = __uint_as_float(w & 0xffff0000u);
              previous[2 * j + h] = s101h::bpk(__fadd_rn(acc[4 * j + 2 * h], p0), __fadd_rn(acc[4 * j + 2 * h + 1], p1));
            }
        } else if constexpr (EPI == 2) {
#pragma unroll
          for (int i = 0; i < R; ++i) {
            const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
            total[i] = first ? x : __fadd_rn(total[i], x);
          }
          first = false;
        }
        fresh = 1;
        pops4b::wg_fence();
      };
      auto body = [&](auto itc, int it, uint32_t (&af)[4][4]) {
        constexpr int ITC = decltype(itc)::value;
        const int q = task_i * NK + it, st = q % NSTX;
        pops4b::bar_wait(bar0 + 8 * st, (unsigned)(q / NSTX) & 1u);
        const unsigned sA = base + st * STAGE, Bs = sA + A_BYTES;
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
          if constexpr (AK) {
            const int r = row0 + (lane & 15), chunk = kq * 2 + (lane >> 4);
            ldsm4_s(af[kq], sA + r * 128 + ((chunk ^ (r & 7)) << 4));
          } else {
            const int k = kq * 16 + (lane & 7) + ((lane >> 4) << 3), m = row0 + (((lane >> 3) & 1) << 3);
            ldsm4t_s(af[kq], sA + (m >> 6) * 8192 + k * 128 + ((((m & 63) >> 3) ^ (k & 7)) << 4));
          }
        }
        pops4b::wg_fence();
#pragma unroll
        for (int kq = 0; kq < 4; ++kq) {
          const int k16 = kq * 16;
          const uint64_t db = BKC ? pops4b::dk(Bs + kq * 32) : pops4b::dmn(Bs + kq * 2048);
          if constexpr (ITC < 0) {
            { Acc<BN>::template run<TB>(acc, af[kq][0], af[kq][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
            Acc<BN>::template run<TB>(acc, 0u, 0u, af[kq][2], af[kq][3], db);
          } else {
            if constexpr (ITC * KS + 0 < EK) if (kq == 0) { Acc<BN>::template run<TB>(acc, af[0][0], af[0][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
            if constexpr (ITC * KS + 0 + 8 < EK) if (kq == 0) Acc<BN>::template run<TB>(acc, 0u, 0u, af[0][2], af[0][3], db);
            if constexpr (ITC * KS + 16 < EK) if (kq == 1) { Acc<BN>::template run<TB>(acc, af[1][0], af[1][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
            if constexpr (ITC * KS + 16 + 8 < EK) if (kq == 1) Acc<BN>::template run<TB>(acc, 0u, 0u, af[1][2], af[1][3], db);
            if constexpr (ITC * KS + 32 < EK) if (kq == 2) { Acc<BN>::template run<TB>(acc, af[2][0], af[2][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
            if constexpr (ITC * KS + 32 + 8 < EK) if (kq == 2) Acc<BN>::template run<TB>(acc, 0u, 0u, af[2][2], af[2][3], db);
            if constexpr (ITC * KS + 48 < EK) if (kq == 3) { Acc<BN>::template run<TB>(acc, af[3][0], af[3][1], 0u, 0u, db, fresh ? 0 : 1); fresh = 0; }
            if constexpr (ITC * KS + 48 + 8 < EK) if (kq == 3) Acc<BN>::template run<TB>(acc, 0u, 0u, af[3][2], af[3][3], db);
          }
          if constexpr (EPI != 0) {
            const int end = it * KS + k16 + 16;
            if (end < EK && end % PARTITION == 0) boundary();
          }
        }
        pops4b::wg_commit();
        pops4b::wg_wait<1>();                                        // step q-1 complete: its registers and stage free
        pops4b::keep(acc);
        __syncthreads();                                             // every warp finished reading step q-1's stage
        if (q + NSTX - 1 < Q) provide(q + NSTX - 1);
      };
      using Full = std::integral_constant<int, -1>;
#pragma unroll 1
      for (int it = 0; it + 1 < NFULL; it += 2) {
        body(Full{}, it, af0);
        body(Full{}, it + 1, af1);
      }
      if constexpr (NFULL % 2 == 1) body(Full{}, NFULL - 1, af0);
      if constexpr (GUARD) {
        if constexpr ((NK - 1) % 2 == 0) body(std::integral_constant<int, NK - 1>{}, NK - 1, af0);
        else body(std::integral_constant<int, NK - 1>{}, NK - 1, af1);
      }
      pops4b::wg_wait<0>();
      pops4b::keep(acc);
      int m0, n0;
      where(task_i, m0, n0);
      finish(m0, n0, acc, previous, total, first, sbase + NSTX * STAGE);
    }
    __syncthreads();
    if (tid == 0) {
#pragma unroll
      for (int st = 0; st < NSTX; ++st) pops4b::bar_inval(bar0 + 8 * st);
    }
    __syncthreads();
  }
};

#ifndef LATCH_WGK8_PIPE_STAGES
#define LATCH_WGK8_PIPE_STAGES 3
#endif
// Cross-tile pipelined drop-in for phase_wide_tma (same arguments and arithmetic).
template <bool AK, bool BKC, int PARTITION, bool INITIAL_RESIDUE, int K, bool ROUNDED_F32 = false,
          bool PARTITION_WIDE = true>
DEV void phase_wide_tma2(const BArgs g, const void* tA, const void* tB, float* smem, int rot = 0) {
  if (g.K != K || g.nb != 1) __trap();
  constexpr int BM = 128, BN = PARTITION > 0 ? LATCH_WGK8_PARTITION_BN : 256, GROUP = 8;
  constexpr int EPI = PARTITION > 0 ? 1 : 0;
  using P = Pipe<BN, AK, BKC, EPI, PARTITION, INITIAL_RESIDUE, K, ROUNDED_F32, LATCH_WGK8_PIPE_STAGES>;
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, rot);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, per = tm * tn;
  const int ntm = slice.me < per ? (per - slice.me + slice.count - 1) / slice.count : 0;
  auto where = [&](int task_i, int& m0, int& n0) {
    const int task = slice.me + task_i * slice.count;
    const int span = GROUP * tn, first = (task / span) * GROUP;
    const int size = tm - first < GROUP ? tm - first : GROUP, local = task % span;
    m0 = (first + local % size) * BM;
    n0 = (local / size) * BN;
  };
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  auto finish = [&](int m0, int n0, float (&acc)[P::R], uint32_t (&previous)[EPI == 1 ? P::R / 2 : 1],
                    float (&)[1], bool, char* staging) {
    if constexpr (EPI == 1) {
#pragma unroll
      for (int j = 0; j < P::R / 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint32_t w = previous[2 * j + (q >> 1)];
          acc[4 * j + q] = __fadd_rn(acc[4 * j + q], __uint_as_float((q & 1) ? (w & 0xffff0000u) : (w << 16)));
        }
    }
    const int mrow = m0 + wg * 64 + 16 * warp + gq;
    if constexpr (ROUNDED_F32) {
#pragma unroll
      for (int j = 0; j < P::R / 4; ++j) {
        const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int mm = mrow + 8 * h;
          if (mm >= g.M || n >= g.N) continue;
          float* Cf = reinterpret_cast<float*>(g.C);
          const float x = __bfloat162float(__float2bfloat16_rn(acc[4 * j + 2 * h]));
          const float y = __bfloat162float(__float2bfloat16_rn(acc[4 * j + 2 * h + 1]));
          if (n + 1 < g.N && g.ldc % 2 == 0)
            *reinterpret_cast<float2*>(Cf + (long long)mm * g.ldc + n) = make_float2(x, y);
          else {
            Cf[(long long)mm * g.ldc + n] = x;
            if (n + 1 < g.N) Cf[(long long)mm * g.ldc + n + 1] = y;
          }
        }
      }
    } else {
      bf16* stile = reinterpret_cast<bf16*>(staging);
      const int lrow = wg * 64 + 16 * warp + gq;
#pragma unroll
      for (int j = 0; j < P::R / 4; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
          *reinterpret_cast<uint32_t*>(stile + (lrow + 8 * h) * P::PITCH + 8 * j + 2 * tq) =
              s101h::bpk(acc[4 * j + 2 * h], acc[4 * j + 2 * h + 1]);
      __syncthreads();
      const int rows = g.M - m0 < BM ? g.M - m0 : BM, cols = g.N - n0 < BN ? g.N - n0 : BN;
      for (int r = tid >> 5; r < rows; r += 8) {
        bf16* dst = g.C + (long long)(m0 + r) * g.ldc + n0;
        const bf16* src = stile + r * P::PITCH;
        if (cols == BN && (reinterpret_cast<unsigned long long>(dst) & 3ull) == 0ull) {
#pragma unroll
          for (int w = lane; w < BN / 2; w += 32)
            reinterpret_cast<uint32_t*>(dst)[w] = reinterpret_cast<const uint32_t*>(src)[w];
        } else {
          for (int c = lane; c < cols; c += 32) dst[c] = src[c];
        }
      }
      __syncthreads();                                               // staging read before the next tile reuses it
    }
  };
  P::run(g, tA, tB, reinterpret_cast<char*>(smem), ntm, where, finish);
}

// Cross-tile pipelined drop-in for phase_serial_sum_tma.
template <bool AK, bool BKC, int CHUNK, int K>
DEV void phase_serial_sum_tma2(const BArgs g, const void* tA, const void* tB, float* smem) {
  constexpr int BM = 128, BN = 128, GROUP = 8;
  using P = Pipe<BN, AK, BKC, 2, CHUNK, false, K, false, 4>;
  if (g.K != K) __trap();
  const popsc::Slice slice = popsc::slice_of(0, 1, 1, 0);
  const int tm = (g.M + BM - 1) / BM, tn = (g.N + BN - 1) / BN, span = GROUP * tn, per = tm * tn;
  const int ntm = slice.me < per ? (per - slice.me + slice.count - 1) / slice.count : 0;
  auto where = [&](int task_i, int& m0, int& n0) {
    const int task = slice.me + task_i * slice.count;
    const int first_m = (task / span) * GROUP, size = tm - first_m < GROUP ? tm - first_m : GROUP;
    const int local = task % span;
    m0 = (first_m + local % size) * BM;
    n0 = (local / size) * BN;
  };
  const int tid = threadIdx.x, wg = tid >> 7, lane = tid & 31, warp = (tid >> 5) & 3, gq = lane >> 2, tq = lane & 3;
  auto finish = [&](int m0, int n0, float (&acc)[P::R], uint32_t (&)[1], float (&total)[P::R], bool first, char*) {
#pragma unroll
    for (int i = 0; i < P::R; ++i) {                                 // final chunk
      const float x = __bfloat162float(__float2bfloat16_rn(acc[i]));
      total[i] = first ? x : __fadd_rn(total[i], x);
    }
    float* out = reinterpret_cast<float*>(g.C);
    const int mrow = m0 + wg * 64 + 16 * warp + gq;
#pragma unroll
    for (int j = 0; j < P::R / 4; ++j) {
      const int n = n0 + 8 * j + 2 * tq;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mm = mrow + 8 * h;
        if (mm >= g.M || n >= g.N) continue;
        const long long where_ = (long long)mm * g.ldc + n;
        if (n + 1 < g.N && g.ldc % 2 == 0)
          *reinterpret_cast<float2*>(out + where_) = make_float2(total[4 * j + 2 * h], total[4 * j + 2 * h + 1]);
        else {
          out[where_] = total[4 * j + 2 * h];
          if (n + 1 < g.N) out[where_ + 1] = total[4 * j + 2 * h + 1];
        }
      }
    }
  };
  P::run(g, tA, tB, reinterpret_cast<char*>(smem), ntm, where, finish);
}
}  // namespace latch_wgk8
