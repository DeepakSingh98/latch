// [S109A component copy of the pinned donor header: identical code unless a timing-only diagnostic define
// S109A_DIAG_NOSTORE / S109A_DIAG_NOSIDE is set; then, while an engine epilogue runs (s109a_diag_epi), the helpers skip
// global stores / return a constant instead of loading the side input. Timing-only; results are invalid.]
// S101H GEMM epilogues. Each functor receives (batch b, row m, column n, 4 consecutive columns v) of a tile.
// Chain epilogues map (b = document, m = row within chunk c) to the global row b*S + c*C + m.
#pragma once
#include "core.cuh"

namespace s101h {

#if defined(S109A_DIAG_NOSTORE) || defined(S109A_DIAG_NOSIDE)
__shared__ int s109a_diag_epi;
#endif
#ifdef S109A_DIAG_NOSTORE
#define S109A_DIAG_ST(stmt) do { if (!s109a_diag_epi) { stmt; } } while (0)
#else
#define S109A_DIAG_ST(stmt) stmt
#endif
#ifdef S109A_DIAG_NOSIDE
#define S109A_DIAG_LD(p, expr, alt) (s109a_diag_epi ? (alt) : (expr))
#else
#define S109A_DIAG_LD(p, expr, alt) (expr)
#endif
DEV void st4(float* p, float4 v) { S109A_DIAG_ST(*reinterpret_cast<float4*>(p) = v); }
DEV float4 ld4(const float* p) { return S109A_DIAG_LD(p, *reinterpret_cast<const float4*>(p), make_float4(0.5f, 0.5f, 0.5f, 0.5f)); }

struct EpiStore {                      // C[b*bs + m*ld + n] = v
  float* C; long long ld, bs;
  DEV void operator()(int b, int m, int n, float4 v) const { st4(C + (long long)b * bs + (long long)m * ld + n, v); }
};

struct EpiScaleStore {                 // C = s * v
  float* C; long long ld, bs; float s;
  DEV void operator()(int b, int m, int n, float4 v) const {
    st4(C + (long long)b * bs + (long long)m * ld + n, make_float4(s * v.x, s * v.y, s * v.z, s * v.w));
  }
};

struct EpiResid {                      // Y = X + v (row-major ld)
  const float* X; float* Y; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const float4 x = ld4(X + (long long)m * ld + n);
    st4(Y + (long long)m * ld + n, make_float4(x.x + v.x, x.y + v.y, x.z + v.z, x.w + v.w));
  }
};

DEV float silu_(float z) { const float s = sigmoidf_(z); return z * s; }
DEV float silu1_(float z) { const float s = sigmoidf_(z); return s * (1.0f + z * (1.0f - s)); }

struct EpiSilu2 {                      // HP = v, HA = silu(v)
  float* HP; float* HA; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    st4(HP + o, v);
    st4(HA + o, make_float4(silu_(v.x), silu_(v.y), silu_(v.z), silu_(v.w)));
  }
};

struct EpiDSilu {                      // dHP = v * silu'(HP)
  const float* HP; float* DH; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    const float4 z = ld4(HP + o);
    st4(DH + o, make_float4(v.x * silu1_(z.x), v.y * silu1_(z.y), v.z * silu1_(z.z), v.w * silu1_(z.w)));
  }
};

// qkv projection epilogue: RoPE on q and k (interleaved pairs), scatter into [D][H][S][dh]
struct EpiQKV {
  float* Q; float* K; float* V; const float* cosT; const float* sinT; int S, H, dh, d;
  DEV void operator()(int, int m, int n, float4 v) const {
    const int which = n / d, rem = n % d, h = rem / dh, j = rem % dh;
    const int b = m / S, s = m % S;
    float* dst = (which == 0 ? Q : which == 1 ? K : V) + (((long long)b * H + h) * S + s) * dh + j;
    if (which < 2) {
      const int p0 = j >> 1;                             // pairs (j, j+1) and (j+2, j+3)
      const float c0 = cosT[s * (dh / 2) + p0], s0 = sinT[s * (dh / 2) + p0];
      const float c1 = cosT[s * (dh / 2) + p0 + 1], s1 = sinT[s * (dh / 2) + p0 + 1];
      v = make_float4(v.x * c0 - v.y * s0, v.x * s0 + v.y * c0, v.z * c1 - v.w * s1, v.z * s1 + v.w * c1);
    }
    st4(dst, v);
  }
};

// ------------------------------------------------------------------------------------------ tail chain
struct ChainRows {
  int S, C, c;
  DEV long long row(int b, int m) const { return (long long)b * S + (long long)c * C + m; }
};

struct EpiChainZ {                     // zf = v; sf = sigmoid; af = zf*sf; sp = silu'
  ChainRows R; float* ZF; float* SF; float* AF; float* SP; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long o = R.row(b, m) * ld + n;
    float z[4] = {v.x, v.y, v.z, v.w}, s[4], a[4], p[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { s[i] = sigmoidf_(z[i]); a[i] = z[i] * s[i]; p[i] = s[i] * (1.0f + z[i] * (1.0f - s[i])); }
    st4(ZF + o, v);
    st4(SF + o, make_float4(s[0], s[1], s[2], s[3]));
    st4(AF + o, make_float4(a[0], a[1], a[2], a[3]));
    st4(SP + o, make_float4(p[0], p[1], p[2], p[3]));
  }
};

struct EpiChainR {                     // r = u + (ms + v) * inv_ts   (ms == nullptr: dense tail, r = u + v * inv_ts)
  ChainRows R; const float* U; const float* MS; float* Rout; float inv_ts; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long o = R.row(b, m) * ld + n;
    const float4 u = ld4(U + o);
    float4 s = v;
    if (MS) { const float4 q = ld4(MS + o); s = make_float4(q.x + v.x, q.y + v.y, q.z + v.z, q.w + v.w); }
    st4(Rout + o, make_float4(u.x + s.x * inv_ts, u.y + s.y * inv_ts, u.z + s.z * inv_ts, u.w + s.w * inv_ts));
  }
};

struct EpiChainStore {                 // chain rows, row-major ld
  ChainRows R; float* C; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const { st4(C + R.row(b, m) * ld + n, v); }
};

struct EpiChainScaled2 {               // A = v; B = v * g[n]   (do/drh and dot/drht)
  ChainRows R; float* A; float* B; const float* g; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long o = R.row(b, m) * ld + n;
    const float4 gv = ld4(g + n);
    st4(A + o, v);
    st4(B + o, make_float4(v.x * gv.x, v.y * gv.y, v.z * gv.z, v.w * gv.w));
  }
};

// inner gradient of a fast matrix, stored per (doc, chunk), and the write W_{c+1} = W_c - eta * g if pub
struct EpiWrite {                      // bsg: per-document stride of Gc, bsw: of the version arrays
  float* Gc; const float* Wc; float* Wn; long long ld, bsg, bsw; float eta; int pub;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long e = (long long)m * ld + n;
    st4(Gc + (long long)b * bsg + e, v);
    const float4 w = ld4(Wc + (long long)b * bsw + e);
    st4(Wn + (long long)b * bsw + e, pub ? make_float4(w.x - eta * v.x, w.y - eta * v.y, w.z - eta * v.z, w.w - eta * v.w) : w);
  }
};

struct EpiDaf {                        // daf = v; dzf = v * sp; dzfn = -eta * dzf
  ChainRows R; const float* SP; float* DAF; float* DZF; float* DZFN; float eta; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long o = R.row(b, m) * ld + n;
    const float4 sp = ld4(SP + o);
    const float4 dz = make_float4(v.x * sp.x, v.y * sp.y, v.z * sp.z, v.w * sp.w);
    st4(DAF + o, v);
    st4(DZF + o, dz);
    st4(DZFN + o, make_float4(-eta * dz.x, -eta * dz.y, -eta * dz.z, -eta * dz.w));
  }
};

struct EpiZt {                         // zt = v; at = v * sp
  ChainRows R; const float* SP; float* ZT; float* AT; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long o = R.row(b, m) * ld + n;
    const float4 sp = ld4(SP + o);
    st4(ZT + o, v);
    st4(AT + o, make_float4(v.x * sp.x, v.y * sp.y, v.z * sp.z, v.w * sp.w));
  }
};

struct EpiChainScale {                 // C = v * s   (rt = mt * inv_ts)
  ChainRows R; float* C; float s; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    st4(C + R.row(b, m) * ld + n, make_float4(v.x * s, v.y * s, v.z * s, v.w * s));
  }
};

// lambda_c = wo * g_c + lambda_{c+1} - eta * v   (per document; wo = n_c / N for this document's chunk)
struct EpiLambda {                     // bsg: per-document stride of Gc, bsl: of the lambda arrays
  const float* Gc; const float* Lnext; float* Lc; const float* wo; long long ld, bsg, bsl; float eta;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long e = (long long)m * ld + n, o = (long long)b * bsl + e;
    const float4 g = ld4(Gc + (long long)b * bsg + e), l = ld4(Lnext + o);
    const float w = wo[b];
    st4(Lc + o, make_float4(w * g.x + l.x - eta * v.x, w * g.y + l.y - eta * v.y, w * g.z + l.z - eta * v.z,
                            w * g.w + l.w - eta * v.w));
  }
};

// dzt = v * sp + daf * silu''(zf) * zt (stored); comb_dzf = wo * dzf - eta * dzt
struct EpiDzt {
  ChainRows R; const float* SP; const float* DAF; const float* ZF; const float* SF; const float* ZT; const float* DZF;
  const float* worow; float* DZT; float* CDZF; float eta; int ld;
  DEV void operator()(int b, int m, int n, float4 v) const {
    const long long row = R.row(b, m), o = row * ld + n;
    const float4 sp = ld4(SP + o), da = ld4(DAF + o), z = ld4(ZF + o), s = ld4(SF + o), zt = ld4(ZT + o), dz = ld4(DZF + o);
    const float wo = worow[row];
    float vv[4] = {v.x, v.y, v.z, v.w}, spv[4] = {sp.x, sp.y, sp.z, sp.w}, dav[4] = {da.x, da.y, da.z, da.w};
    float zv[4] = {z.x, z.y, z.z, z.w}, sv[4] = {s.x, s.y, s.z, s.w}, ztv[4] = {zt.x, zt.y, zt.z, zt.w};
    float dzv[4] = {dz.x, dz.y, dz.z, dz.w}, out[4], t[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float spp = sv[i] * (1.0f - sv[i]) * (2.0f + zv[i] * (1.0f - 2.0f * sv[i]));
#if defined(S102G_FX_MIXED)
      t[i] = vv[i] * spv[i] + 0.0f * (dav[i] * spp * ztv[i]);      // semantic fault: mixed second-order term dropped
#else
      t[i] = vv[i] * spv[i] + dav[i] * spp * ztv[i];
#endif
      out[i] = wo * dzv[i] - eta * t[i];
    }
    st4(DZT + o, make_float4(t[0], t[1], t[2], t[3]));
    st4(CDZF + o, make_float4(out[0], out[1], out[2], out[3]));
  }
};

struct EpiCombDzs {                    // comb_dzs = v * silu'(zs)
  float* CDZS; const float* ZS; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    const float4 z = ld4(ZS + o);
    st4(CDZS + o, make_float4(v.x * silu1_(z.x), v.y * silu1_(z.y), v.z * silu1_(z.z), v.w * silu1_(z.w)));
  }
};

struct EpiAddInto {                    // C = A + v (e.g. tied embedding: lookup + head)
  const float* A; float* C; long long ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const float4 a = ld4(A + (long long)m * ld + n);
    st4(C + (long long)m * ld + n, make_float4(a.x + v.x, a.y + v.y, a.z + v.z, a.w + v.w));
  }
};

}  // namespace s101h

namespace s101h {
// ------------------------------------------------------------------------------------------ Muon NS5 (BF16 rounding)
// upstream: A = X @ X.mT;  B = b * A + c * A @ A  (python: (c * A) @ A);  X = a * X + B @ X   (every op rounds to BF16)
struct EpiNsA {                        // A = bf16(acc); CA = bf16(c * A)
  float* A; float* CA; float c;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * 256 + n;
    const float4 a = make_float4(bf16r(v.x), bf16r(v.y), bf16r(v.z), bf16r(v.w));
    st4(A + o, a);
    st4(CA + o, make_float4(bf16r(c * a.x), bf16r(c * a.y), bf16r(c * a.z), bf16r(c * a.w)));
  }
};
struct EpiNsB {                        // Bm = bf16(bf16(b * A) + bf16(acc))   (acc = CA @ A)
  float* Bm; const float* A; float b;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * 256 + n;
    const float4 a = ld4(A + o);
    st4(Bm + o, make_float4(bf16r(bf16r(b * a.x) + bf16r(v.x)), bf16r(bf16r(b * a.y) + bf16r(v.y)),
                            bf16r(bf16r(b * a.z) + bf16r(v.z)), bf16r(bf16r(b * a.w) + bf16r(v.w))));
  }
};
struct EpiNsX {                        // X' = bf16(bf16(a * X) + bf16(acc))   (acc = Bm @ X)
  float* Xo; const float* Xi; float a; int ld;
  DEV void operator()(int, int m, int n, float4 v) const {
    const long long o = (long long)m * ld + n;
    const float4 x = ld4(Xi + o);
    st4(Xo + o, make_float4(bf16r(bf16r(a * x.x) + bf16r(v.x)), bf16r(bf16r(a * x.y) + bf16r(v.y)),
                            bf16r(bf16r(a * x.z) + bf16r(v.z)), bf16r(bf16r(a * x.w) + bf16r(v.w))));
  }
};
}  // namespace s101h
