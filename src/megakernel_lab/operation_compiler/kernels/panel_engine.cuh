// P8-D panel engine: device primitives for cluster-resident row-panel execution of a bounded sampling loop
// (new code in this lane; generated programs include it, see ops/sampling_panels.py).
//
// A row group of R rows lives on one thread-block cluster of C CTAs for every loop iteration. Each CTA owns a
// column slice (whole 16-column units) of every row-indexed value; elementwise work runs on owned slices in shared
// memory. A matrix product out[R, N] = panel[R, K] @ B[K, N] (B loop-invariant BF16) needs the full panel: every
// CTA writes its owned K-slice of the BF16 operand into its own panel buffer, pushes it to each peer with one bulk
// DSMEM copy (cp.async.bulk shared::cluster) that completes a transaction count on the peer's mbarrier, and waits
// on its own mbarrier for the peers' bytes. Two panel buffers alternate between consecutive products, so a buffer
// is rewritten only after every peer has consumed it (a peer's next slice for that buffer is causally after the
// consumer's use; see ops/sampling_panels.py). The CTA then computes its owned output features over the full K with
// mma.sync m16n8k16 (A = streamed weight chunk, swap-AB; B = panel), FP32 accumulation, one BF16 rounding.
//
// Weight operand slices stream from global/L2 through a STAGES-deep cp.async ring of K-chunks. The chunk sequence of
// one iteration is static, so prefetch continues across product and iteration boundaries.
//
// Panel layout (BF16, R rows): element (r, k) at byte ((k >> 3) * R + r) * 16 + (k & 7) * 2. A CTA's owned units
// [u0, u1) occupy bytes [u0 * 32 R, u1 * 32 R): one contiguous bulk copy per peer; 8x8 core matrices are 128 bytes.
//
// Arithmetic contract of a product: each output element accumulates BF16 x BF16 products in FP32 (one mma per k16
// block; with S > 1 k-splits, S partial chains summed in fixed order s = 0..S-1), then rounds once to BF16 (RN).
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace panel {

typedef __nv_bfloat16 bf16;
#define PDEV __device__ __forceinline__

PDEV uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
PDEV uint32_t cluster_rank() { uint32_t r; asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r; }
PDEV uint32_t map_peer(uint32_t addr, uint32_t rank) {
  uint32_t out; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank)); return out;
}
PDEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
PDEV void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(saddr(bar)), "r"(count) : "memory");
}
PDEV void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
PDEV void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" :: "r"(saddr(bar)), "r"(bytes) : "memory");
}
PDEV bool mbar_try(uint32_t a, uint32_t parity) {
  uint32_t done;
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
      "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
  return done != 0;
}
// Non-suspending poll (test_wait): for short waits inside compute loops, where try_wait's suspend/wake costs more
// than the wait itself.
PDEV void mbar_spin(uint64_t* bar, uint32_t parity) {
  const uint32_t a = saddr(bar);
  for (long long spin = 0;; ++spin) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) return;
    if (spin > (1LL << 28)) __trap();
  }
}
// Bounded wait: a protocol error traps (fails the launch) instead of hanging the device.
PDEV void mbar_wait(uint64_t* bar, uint32_t parity) {
  const uint32_t a = saddr(bar);
  for (long long spin = 0; !mbar_try(a, parity); ++spin)
    if (spin > (1LL << 26)) __trap();
}
PDEV void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// Bulk copy from this CTA's shared memory to a (peer) CTA's shared memory; completes `bytes` on the peer's mbarrier.
PDEV void bulk_to_peer(uint32_t dst_cluster, uint32_t src_cta, uint32_t bytes, uint32_t bar_cluster) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst_cluster), "r"(src_cta), "r"(bytes), "r"(bar_cluster) : "memory");
}
PDEV void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
}
PDEV void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> PDEV void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
PDEV void ldsm4(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
PDEV void ldsm4t(uint32_t (&r)[4], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
PDEV void ldsm2(uint32_t (&r)[2], uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
PDEV void mma16816(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
PDEV float bf16_round(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Byte offset of element (r, k) in an R-row panel.
PDEV uint32_t panel_off(int R, int r, int k) { return ((uint32_t)((k >> 3) * R + r) << 4) + ((uint32_t)(k & 7) << 1); }

// Store four consecutive BF16 values (k % 4 == 0) of row r into a panel.
PDEV void panel_store4(unsigned char* panel, int R, int r, int k, float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  uint2 v; v.x = *reinterpret_cast<uint32_t*>(&lo); v.y = *reinterpret_cast<uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(panel + panel_off(R, r, k)) = v;
}
PDEV void bf16_store4(bf16* p, float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  uint2 v; v.x = *reinterpret_cast<uint32_t*>(&lo); v.y = *reinterpret_cast<uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(p) = v;
}
PDEV float4 bf16_load4(const bf16* p) {
  const uint2 v = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&v.x), hi = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
  return make_float4(__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi));
}

// ---------------------------------------------------------------- packed product operands
// Once per update (before the loop) every product's loop-invariant operand B[K, N] is re-laid out for its consumers:
// rank c (owned features [n_lo, n_lo + w)) gets a contiguous slice of Kpad / KC chunks, chunk q holding A(m, k) =
// B(q KC + k, n_lo + m) as w rows of KC = 64 BF16 (128 bytes) with the 16-byte column index XOR-swizzled by m & 7
// (conflict-free ldmatrix without padding). Rank c's slice starts at element n_lo * Kpad of the product's block, so a
// chunk is ONE contiguous range: one bulk copy per chunk. k >= K is zero-filled (the panel is zero there too).
PDEV uint32_t swz(int m, int k) {  // byte offset of A(m, k) inside a chunk (KC = 64)
  return (uint32_t)(m * 128 + ((((k >> 3) ^ (m & 7)) & 7) << 4) + ((k & 7) << 1));
}

// One 16-byte vector v of a product's pack (P9-D fused pack: every product shares one grid-strided loop).
PDEV void pack_vector(bf16* __restrict__ dst, const bf16* __restrict__ src, int kmajor, long long stride, int K,
                      int Kpad, int N, int C, long long v) {
  const int u = N >> 4, k8s = Kpad >> 3;
  int n, k8;
  if (kmajor) { n = (int)(v / k8s); k8 = (int)(v - (long long)n * k8s); }
  else { k8 = (int)(v / N); n = (int)(v - (long long)k8 * N); }
  const int t = n >> 4;
  int c = (C * (t + 1) + u - 1) / u - 1;
  c = c < C - 1 ? c : C - 1;
  const int lo = 16 * ((u * c) / C), w = 16 * ((u * (c + 1)) / C) - lo;
  const int m = n - lo, q = k8 >> 3, kk8 = k8 & 7, k = k8 << 3;
  uint4 out = make_uint4(0, 0, 0, 0);
  if (k < K) {
    if (kmajor) {
      out = *reinterpret_cast<const uint4*>(src + (long long)n * stride + k);
    } else {
      bf16 e[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) e[j] = src[(long long)(k + j) * stride + n];
      out = *reinterpret_cast<const uint4*>(e);
    }
  }
  bf16* base = dst + (long long)lo * Kpad + (long long)q * w * 64 + m * 64;
  *reinterpret_cast<uint4*>(base + ((kk8 ^ (m & 7)) << 3)) = out;
}

// Grid-strided pack of one product operand (all ranks): B(k, n) = src[k + n * stride] (kmajor) or src[k * stride + n].
__device__ __noinline__ void pack_operand(bf16* __restrict__ dst, const bf16* __restrict__ src, int kmajor,
                                          long long stride, int K, int Kpad, int N, int C) {
  const int u = N >> 4, k8s = Kpad >> 3;
  const long long vectors = (long long)N * k8s;
  for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < vectors; v += (long long)gridDim.x * blockDim.x) {
    // k-major sources: consecutive threads walk k (contiguous); n-major sources: consecutive threads walk n.
    int n, k8;
    if (kmajor) { n = (int)(v / k8s); k8 = (int)(v - (long long)n * k8s); }
    else { k8 = (int)(v / N); n = (int)(v - (long long)k8 * N); }
    const int t = n >> 4;
    int c = (C * (t + 1) + u - 1) / u - 1;          // owner: max c with floor(u c / C) <= t
    c = c < C - 1 ? c : C - 1;
    const int lo = 16 * ((u * c) / C), w = 16 * ((u * (c + 1)) / C) - lo;
    const int m = n - lo, q = k8 >> 3, kk8 = k8 & 7, k = k8 << 3;
    uint4 out = make_uint4(0, 0, 0, 0);
    if (k < K) {
      if (kmajor) {
        out = *reinterpret_cast<const uint4*>(src + (long long)n * stride + k);
      } else {
        bf16 e[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = src[(long long)(k + j) * stride + n];
        out = *reinterpret_cast<const uint4*>(e);
      }
    }
    bf16* base = dst + (long long)lo * Kpad + (long long)q * w * 64 + m * 64;
    *reinterpret_cast<uint4*>(base + ((kk8 ^ (m & 7)) << 3)) = out;
  }
}

// This CTA's packed slice of one product.
struct Prod {
  const bf16* pack;   // first chunk of this rank's slice
  int width;          // owned output features (multiple of 16; may be 0)
  int nchunks;        // Kpad / 64
  int chunk0;         // first chunk index within one iteration
  int pad;
};

PDEV void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(saddr(bar)) : "memory");
}
PDEV void bulk_from_global(uint32_t dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(saddr(bar)) : "memory");
}

// Weight ring: STAGES slots of SLOT bytes; full[s] (1 arrival + transaction bytes) and empty[s] (8 warp arrivals).
struct Ring {
  unsigned char* slots;
  uint64_t* full;
  uint64_t* empty;
  const unsigned char* owner;   // product of each chunk of one iteration
};
// The producer refills the slot of chunk g - LAG while chunk g is consumed (LAG = 2: the slowest warp of chunk g - 2
// has normally released it, so the producing thread rarely stalls its own warp). STAGES - LAG chunks stay in flight.
constexpr int LAG = 2;

// Issue global chunk index `ahead` (all iterations) into its slot. Producer: one thread.
template <int STAGES, int SLOT>
PDEV void issue(const Ring& ring, const Prod* prods, int nprods, int nchunks_iter, int ahead) {
  const int gi = ahead % nchunks_iter;
  const Prod& P = prods[ring.owner[gi]];
  const int slot = ahead % STAGES;
  const uint32_t bytes = (uint32_t)(P.width * 128);
#ifdef PANEL_DIAG_SKIP_WEIGHTS
  mbar_arrive(&ring.full[slot]);
#else
  if (bytes) {
    mbar_arrive_expect_tx(&ring.full[slot], bytes);
    bulk_from_global(saddr(ring.slots + slot * SLOT), P.pack + (long long)(gi - P.chunk0) * P.width * 64, bytes,
                     &ring.full[slot]);
  } else {
    mbar_arrive(&ring.full[slot]);
  }
#endif
}

// Product p: out[r, m] (owned features m < width, rows r < R) from the gathered panel over the full (padded) K.
// Warp w: m-tiles {wm + i WM} and k16 steps {s + j S} of every chunk, wm = w % WM, s = w / WM (WM * S <= 8 warps
// compute; all 8 release ring slots). Partial chains of the S k-phases are summed in order s = 0..S-1.
template <int R, int STAGES, int SLOT, int MTW, int WM, int S>
__device__ __noinline__ void product(const Prod* prods, int nprods, int p, int nchunks_iter, int& g, int total,
                                     const Ring& ring, const unsigned char* panel_buf, bf16* out, int ldo,
                                     float* scratch) {
  constexpr int NT = R / 8;
  const Prod& P = prods[p];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wm = warp % WM, s = warp / WM;
  const bool active = s < S;
  const int mt = P.width >> 4;
  float acc[MTW][NT][4];
#pragma unroll
  for (int i = 0; i < MTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.f;
  const uint32_t pbase = saddr(panel_buf);
  // Lane-constant parts of the ldmatrix addresses.
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_k8 = lane >> 4;
  const int b_row = lane & 7, b_k8 = (lane >> 3) & 1;
  const int nch = P.nchunks;
  for (int c = 0; c < nch; ++c, ++g) {
    const int slot = g % STAGES;
    if (threadIdx.x == 0) {
      const int ahead = g + STAGES - LAG;
      if (ahead < total) {
        if (g >= LAG) mbar_wait(&ring.empty[(g - LAG) % STAGES], (uint32_t)(((g - LAG) / STAGES) & 1));
        issue<STAGES, SLOT>(ring, prods, nprods, nchunks_iter, ahead);
      }
    }
    mbar_wait(&ring.full[slot], (uint32_t)((g / STAGES) & 1));
#ifndef PANEL_DIAG_SKIP_MMA
    if (active) {
      const uint32_t sbase = saddr(ring.slots + slot * SLOT);
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        if (qq % S != s) continue;
        const int kq = c * 4 + qq;
        uint32_t b[NT][2];
#pragma unroll
        for (int n = 0; n < NT; ++n) ldsm2(b[n], pbase + panel_off(R, n * 8 + b_row, kq * 16 + b_k8 * 8));
#pragma unroll
        for (int i = 0; i < MTW; ++i) {
          const int mi = wm + i * WM;
          if (mi < mt) {
            uint32_t a[4];
            const int m = mi * 16 + a_row;
            ldsm4(a, sbase + swz(m, qq * 16 + a_k8 * 8));
#pragma unroll
            for (int n = 0; n < NT; ++n) mma16816(acc[i][n], a, b[n]);
          }
        }
      }
    }
#endif
    __syncwarp();
    if (lane == 0) mbar_arrive(&ring.empty[slot]);
  }
  const int gq = lane >> 2, tq = lane & 3;
  if (S > 1) {
    if (active && s > 0) {
      float* sp = scratch + (s - 1) * (R * ldo);
#pragma unroll
      for (int i = 0; i < MTW; ++i) {
        const int mi = wm + i * WM;
        if (mi >= mt) continue;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          const int m = mi * 16 + gq, r = n * 8 + 2 * tq;
          sp[r * ldo + m] = acc[i][n][0];
          sp[(r + 1) * ldo + m] = acc[i][n][1];
          sp[r * ldo + m + 8] = acc[i][n][2];
          sp[(r + 1) * ldo + m + 8] = acc[i][n][3];
        }
      }
    }
    __syncthreads();
  }
  if (!active || s != 0) return;
#pragma unroll
  for (int i = 0; i < MTW; ++i) {
    const int mi = wm + i * WM;
    if (mi >= mt) continue;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int m = mi * 16 + gq, r = n * 8 + 2 * tq;
      float v0 = acc[i][n][0], v1 = acc[i][n][1], v2 = acc[i][n][2], v3 = acc[i][n][3];
#pragma unroll
      for (int t = 1; t < S; ++t) {
        const float* sp = scratch + (t - 1) * (R * ldo);
        v0 += sp[r * ldo + m]; v1 += sp[(r + 1) * ldo + m]; v2 += sp[r * ldo + m + 8]; v3 += sp[(r + 1) * ldo + m + 8];
      }
      out[r * ldo + m] = __float2bfloat16_rn(v0);
      out[(r + 1) * ldo + m] = __float2bfloat16_rn(v1);
      out[r * ldo + m + 8] = __float2bfloat16_rn(v2);
      out[(r + 1) * ldo + m + 8] = __float2bfloat16_rn(v3);
    }
  }
}

// Push this CTA's owned panel units [u0, u1) to every peer of the cluster (after __syncthreads()).
PDEV void push_slice(unsigned char* panel_buf, int R, int u0, int u1, int C, int rank, uint64_t* bar) {
#ifdef PANEL_DIAG_SKIP_EXCHANGE
  return;
#endif
  // One copy per peer, issued in parallel by threads 1..C-1 of warp 0.
  const int d = threadIdx.x;
  if (d >= 1 && d < C) {
    fence_proxy_async();
    const uint32_t off = (uint32_t)(u0 * 32 * R), bytes = (uint32_t)((u1 - u0) * 32 * R);
    const uint32_t src = saddr(panel_buf) + off, b = saddr(bar);
    if (bytes) {
      const uint32_t peer = (uint32_t)((rank + d) % C);
      bulk_to_peer(map_peer(src, peer), src, bytes, map_peer(b, peer));
    }
  }
}


// ---------------------------------------------------------------- slice mode (one bulk copy per product operand)
// A product's packed operand slice for this rank (w rows x Kpad, swizzled 64-deep chunks back to back) is contiguous
// in the pack, so it moves with ONE bulk copy into one of two slice slots, issued a whole hop before its product.
// The product then runs over the full K with no per-chunk synchronization.
PDEV void load_slice(unsigned char* slot, uint64_t* full, const bf16* src, uint32_t bytes) {
  // caller: one thread
#ifdef PANEL_DIAG_SKIP_WEIGHTS
  mbar_arrive(full);
#else
  if (bytes) {
    mbar_arrive_expect_tx(full, bytes);
    bulk_from_global(saddr(slot), src, bytes, full);
  } else {
    mbar_arrive(full);
  }
#endif
}

// Remote arrival on the same-offset mbarrier of a peer CTA (cluster scope, release).
PDEV void arrive_peer(uint64_t* bar, uint32_t peer) {
  const uint32_t remote = map_peer(saddr(bar), peer);
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
PDEV void mbar_wait_cluster(uint64_t* bar, uint32_t parity) {
  const uint32_t a = saddr(bar);
  for (long long spin = 0;; ++spin) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
    if (done) break;
    if (spin > (1LL << 26)) __trap();
  }
}

// Single panel buffer flow control: after its product's MMA finished reading the panel, a CTA tells every peer
// (threads 1..C-1, one peer each) that its buffer may receive the next product's slices.
PDEV void signal_consumed(uint64_t* free_bar, int C, int rank) {
  const int d = threadIdx.x;
  if (d >= 1 && d < C) arrive_peer(free_bar, (uint32_t)((rank + d) % C));
}

// Push after the peers consumed the previous product (free phase `phase` complete), one peer per thread 1..C-1.
PDEV void push_slice_flow(unsigned char* panel_buf, int R, int u0, int u1, int C, int rank, uint64_t* bar,
                          uint64_t* free_bar, int wait_free, uint32_t free_parity) {
#ifdef PANEL_DIAG_SKIP_EXCHANGE
  return;
#endif
  const int d = threadIdx.x;
  if (d >= 1 && d < C) {
    const uint32_t peer = (uint32_t)((rank + d) % C);
    // The peer's buffer is free once that peer consumed the previous product; its consumed signal arrives on OUR
    // free barrier (count C-1), so every pushing thread waits for all peers (one phase per product).
    if (wait_free) mbar_wait_cluster(free_bar, free_parity);
    fence_proxy_async();
    const uint32_t off = (uint32_t)(u0 * 32 * R), bytes = (uint32_t)((u1 - u0) * 32 * R);
    const uint32_t src = saddr(panel_buf) + off;
    if (bytes) bulk_to_peer(map_peer(src, peer), src, bytes, map_peer(saddr(bar), peer));
  }
}

// Product over a resident slice: out[r, m] = sum_k panel[r, k] A(m, k), A in 64-deep swizzled chunks of w rows.
// Warp w: m-tiles {wm + i WM}, k16 steps {s + j S} over the whole K (independent accumulators for ILP).
// With S > 1 the k-phase partials go through the product's own slot once every warp finished reading it.
template <int R, int MTW, int WM, int S>
__device__ __noinline__ void product_slice(unsigned char* slot, int width, int nchunks,
                                           const unsigned char* panel_buf, bf16* out, int ldo) {
  float* scratch = reinterpret_cast<float*>(slot);
  constexpr int NT = R / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wm = warp % WM, s = warp / WM;
  const bool active = s < S;
  const int mt = width >> 4;
  float acc[MTW][NT][4];
#pragma unroll
  for (int i = 0; i < MTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.f;
  const uint32_t pbase = saddr(panel_buf), sbase = saddr(slot);
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_k8 = lane >> 4;
  const int b_row = lane & 7, b_k8 = (lane >> 3) & 1;
  const uint32_t chunk_bytes = (uint32_t)(width * 128);
#ifndef PANEL_DIAG_SKIP_MMA
  if (active) {
    const int kq_total = nchunks * 4;
    for (int kq = s; kq < kq_total; kq += S) {
      const int c = kq >> 2, qq = kq & 3;
      uint32_t b[NT][2];
#pragma unroll
      for (int n = 0; n < NT; ++n) ldsm2(b[n], pbase + panel_off(R, n * 8 + b_row, kq * 16 + b_k8 * 8));
      const uint32_t cbase = sbase + (uint32_t)c * chunk_bytes;
#pragma unroll
      for (int i = 0; i < MTW; ++i) {
        const int mi = wm + i * WM;
        if (mi < mt) {
          uint32_t a[4];
          ldsm4(a, cbase + swz(mi * 16 + a_row, qq * 16 + a_k8 * 8));
#pragma unroll
          for (int n = 0; n < NT; ++n) mma16816(acc[i][n], a, b[n]);
        }
      }
    }
  }
#endif
  const int gq = lane >> 2, tq = lane & 3;
  if (S > 1) {
    __syncthreads();   // every warp finished reading the slot: it becomes the partial-sum scratch
    if (active && s > 0) {
      float* sp = scratch + (s - 1) * (R * ldo);
#pragma unroll
      for (int i = 0; i < MTW; ++i) {
        const int mi = wm + i * WM;
        if (mi >= mt) continue;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          const int m = mi * 16 + gq, r = n * 8 + 2 * tq;
          sp[r * ldo + m] = acc[i][n][0];
          sp[(r + 1) * ldo + m] = acc[i][n][1];
          sp[r * ldo + m + 8] = acc[i][n][2];
          sp[(r + 1) * ldo + m + 8] = acc[i][n][3];
        }
      }
    }
    __syncthreads();
  }
  if (!active || s != 0) return;
#pragma unroll
  for (int i = 0; i < MTW; ++i) {
    const int mi = wm + i * WM;
    if (mi >= mt) continue;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int m = mi * 16 + gq, r = n * 8 + 2 * tq;
      float v0 = acc[i][n][0], v1 = acc[i][n][1], v2 = acc[i][n][2], v3 = acc[i][n][3];
#pragma unroll
      for (int t = 1; t < S; ++t) {
        const float* sp = scratch + (t - 1) * (R * ldo);
        v0 += sp[r * ldo + m]; v1 += sp[(r + 1) * ldo + m]; v2 += sp[r * ldo + m + 8]; v3 += sp[(r + 1) * ldo + m + 8];
      }
      out[r * ldo + m] = __float2bfloat16_rn(v0);
      out[(r + 1) * ldo + m] = __float2bfloat16_rn(v1);
      out[r * ldo + m + 8] = __float2bfloat16_rn(v2);
      out[(r + 1) * ldo + m + 8] = __float2bfloat16_rn(v3);
    }
  }
}


PDEV void signal_consumed(uint64_t* free_bar, int C, int rank);
// Fused form: every active warp leaves its FP32 k-phase partials in the (consumed) slot, [S][R][ldo]; the following
// elementwise segment sums them in order s = 0..S-1 and rounds once to BF16 (same value as product_slice's output).
// Per-source readiness (OVERLAP): unit t of the panel (one k16 step) is owned by rank c(t); the step waits only for
// that peer's bulk copy (mbarrier from[c], one per source), the CTA's own units come first (rotation by its first
// unit), so the MMA starts on local data while peer slices are still arriving.
template <int R, int MTW, int WM, int S, int OVERLAP>
__device__ __noinline__ void product_slice_partials(unsigned char* slot, int width, int nchunks,
                                                    const unsigned char* panel_buf, int ldo, uint64_t* from,
                                                    uint32_t parity, int units, int rank, int C,
                                                    uint64_t* consumed) {
  constexpr int NT = R / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int wm = warp % WM, s = warp / WM;
  const bool active = s < S;
  const int mt = width >> 4;
  float acc[MTW][NT][4];
#pragma unroll
  for (int i = 0; i < MTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.f;
  const uint32_t pbase = saddr(panel_buf), sbase = saddr(slot);
  const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_k8 = lane >> 4;
  const int b_row = lane & 7, b_k8 = (lane >> 3) & 1;
  const uint32_t chunk_bytes = (uint32_t)(width * 128);
#ifndef PANEL_DIAG_SKIP_MMA
  if (active) {
    // Two-stage software pipeline: the fragments of step kq + S are loaded before the MMAs of step kq issue
    // (the ldmatrix/mma asm statements are volatile, so the compiler keeps this order).
    const int kq_total = nchunks * 4;
    const int rot = OVERLAP ? (units * rank) / C : 0;
    uint32_t waited = 0u;
    uint32_t b[2][NT][2], a[2][MTW][4];
    auto load = [&](int buf, int step) {
      int kq = step + rot;
      if (kq >= kq_total) kq -= kq_total;
      if (OVERLAP && kq < units) {
        int owner = (C * (kq + 1) + units - 1) / units - 1;
        owner = owner < C - 1 ? owner : C - 1;
        if (owner != rank && !((waited >> owner) & 1u)) { mbar_spin(from + owner, parity); waited |= 1u << owner; }
      }
      const int c = kq >> 2, qq = kq & 3;
#pragma unroll
      for (int n = 0; n < NT; ++n) ldsm2(b[buf][n], pbase + panel_off(R, n * 8 + b_row, kq * 16 + b_k8 * 8));
      const uint32_t cbase = sbase + (uint32_t)c * chunk_bytes;
#pragma unroll
      for (int i = 0; i < MTW; ++i) {
        const int mi = wm + i * WM;
        if (mi < mt) ldsm4(a[buf][i], cbase + swz(mi * 16 + a_row, qq * 16 + a_k8 * 8));
      }
    };
    auto compute = [&](int buf) {
#pragma unroll
      for (int i = 0; i < MTW; ++i) {
        const int mi = wm + i * WM;
        if (mi < mt) {
#pragma unroll
          for (int n = 0; n < NT; ++n) mma16816(acc[i][n], a[buf][i], b[buf][n]);
        }
      }
    };
    int kq = s;
    if (kq < kq_total) load(0, kq);
    for (; kq < kq_total; kq += 2 * S) {
      if (kq + S < kq_total) load(1, kq + S);
      compute(0);
      if (kq + S >= kq_total) break;
      if (kq + 2 * S < kq_total) load(0, kq + 2 * S);
      compute(1);
    }
  }
#endif
  __syncthreads();   // every warp finished reading the slot and the panel
  if (consumed) signal_consumed(consumed, C, rank);   // peers may now deliver the next panel
  if (!active) return;
  float* sp = reinterpret_cast<float*>(slot) + s * (R * ldo);
  const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int i = 0; i < MTW; ++i) {
    const int mi = wm + i * WM;
    if (mi >= mt) continue;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int m = mi * 16 + gq, r = n * 8 + 2 * tq;
      sp[r * ldo + m] = acc[i][n][0];
      sp[(r + 1) * ldo + m] = acc[i][n][1];
      sp[r * ldo + m + 8] = acc[i][n][2];
      sp[(r + 1) * ldo + m + 8] = acc[i][n][3];
    }
  }
}


// ---------------------------------------------------------------- warpgroup MMA product (P9-D)
// out^T[m, r] = sum_k A(m, k) panel(r, k): A = this rank's packed operand chunks (w rows x 64 k, 128-byte swizzle,
// K-major; m64 tiles read rows >= w as padding whose results are discarded), B = the panel (K-major core matrices of
// 8 rows x 16 bytes: rows adjacent at 16 B, 8-row groups at 128 B, k8 groups at R x 16 B). Two warpgroups take the
// two halves of the k16 steps (k-split S = 2); each accumulates m64nRk16 BF16 products in FP32 from +0, then writes
// its FP32 partial [R][ldo] to the consumed slot, exactly like product_slice_partials (fused reduction in the next
// segment, fixed order s = 0, 1). Tensor-core association inside a k16 step differs from mma.sync (declared).
template <int N> PDEV void wg_mma(float (&d)[N / 2], uint64_t da, uint64_t db);   // emitted per R by the generator
PDEV void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
PDEV void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
PDEV void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
PDEV uint64_t wg_desc_sw128(uint32_t a) {   // K-major, 128-byte swizzle, SBO 1024 (8 rows x 128 B)
  return (uint64_t)((a & 0x3FFFFu) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}
template <int LBO, int SBO> PDEV uint64_t wg_desc_plain(uint32_t a) {   // K-major, no swizzle (core matrices)
  return (uint64_t)((a & 0x3FFFFu) >> 4) | ((uint64_t)(LBO >> 4) << 16) | ((uint64_t)(SBO >> 4) << 32);
}

PDEV void signal_consumed(uint64_t* free_bar, int C, int rank);
// One product's whole k loop for one warpgroup as ONE asm block (fence, NC x 2 wgmma with descriptor arithmetic in
// PTX, commit, wait): emitted per (R, NC) by the generator. Warpgroup w takes k16 steps 2w, 2w + 1 of every 64-deep
// chunk, so both warpgroups issue the same compile-time count; A advances 32 B inside a chunk and `jump` units
// (chunk_bytes - 32 B) across chunks, B advances one k16 (R x 32 B) inside and three across.
template <int R, int NC> PDEV void wg_steps(float (&d)[R / 2], uint64_t da, uint64_t db, uint64_t jump);

template <int R, int NC>
__device__ __noinline__ void product_slice_wg(unsigned char* slot, int width, const unsigned char* panel_buf, int ldo,
                                              uint64_t* consumed, int rank, int C) {
  const int wg = threadIdx.x >> 7, w = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31;
  float d[R / 2];
#pragma unroll
  for (int j = 0; j < R / 2; ++j) d[j] = 0.f;
  const uint32_t pbase = saddr(panel_buf), sbase = saddr(slot);
  const uint32_t chunk_bytes = (uint32_t)(width * 128);
  const uint64_t da = wg_desc_sw128(sbase + (uint32_t)(2 * wg) * 32u);
  const uint64_t db = wg_desc_plain<R * 16, 128>(pbase + (uint32_t)(2 * wg * R * 32));
  wg_steps<R, NC>(d, da, db, (uint64_t)((chunk_bytes - 32u) >> 4));
  __syncthreads();   // both warpgroups finished reading the slot and the panel
  if (consumed) signal_consumed(consumed, C, rank);
  float* sp = reinterpret_cast<float*>(slot) + wg * (R * ldo);
  const int g = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int j = 0; j < R / 2; ++j) {
    const int m = 16 * w + g + 8 * ((j >> 1) & 1), r = 8 * (j >> 2) + 2 * tq + (j & 1);
    if (m < width) sp[r * ldo + m] = d[j];
  }
}

// ---------------------------------------------------------------- multicast exchange
// The owned slice is staged in L2 (global, same panel layout per cluster) and delivered to every CTA of the cluster
// by ONE bulk copy with .multicast::cluster, completing the transaction count on each destination's same-offset
// mbarrier. Measured (p8_d/probe/exchange.cu, H100): 2.5 KB slices 1.05 us/round vs 1.86 us for 15 DSMEM pushes.
PDEV void multicast_slice(unsigned char* panel_buf, const unsigned char* stage, int R, int u0, int u1, int C,
                          uint64_t* bar, uint64_t* free_bar, int wait_free, uint32_t free_parity) {
#ifdef PANEL_DIAG_SKIP_EXCHANGE
  return;
#endif
  if (threadIdx.x == 0) {
    const uint32_t off = (uint32_t)(u0 * 32 * R), bytes = (uint32_t)((u1 - u0) * 32 * R);
    if (wait_free) mbar_wait_cluster(free_bar, free_parity);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    if (bytes) {
      const uint16_t mask = (uint16_t)((1u << C) - 1u);
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
                   :: "r"(saddr(panel_buf) + off), "l"(stage + off), "r"(bytes), "r"(saddr(bar)), "h"(mask) : "memory");
    }
  }
}

// Global staging store of four BF16 values (same layout as the panel).
PDEV void stage_store4(unsigned char* stage, int R, int r, int k, float a, float b, float c, float d) {
  __nv_bfloat162 lo = __floats2bfloat162_rn(a, b), hi = __floats2bfloat162_rn(c, d);
  uint2 v; v.x = *reinterpret_cast<uint32_t*>(&lo); v.y = *reinterpret_cast<uint32_t*>(&hi);
  *reinterpret_cast<uint2*>(stage + panel_off(R, r, k)) = v;
}

}  // namespace panel
