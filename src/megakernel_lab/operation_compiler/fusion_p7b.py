# Integrated from frozen donor f6f60bd1325cae4ded5fba17083feb3b85a791b0; private scheduling/resource protocol.
# P-OPS2-A/B/C integration: pins 4dcee751fab59f5ecea67ebada3167ca93fba841,
# d8f8c0d981354e3b81ef4c8671ae6cc2481a3af9, e7fe4b165f186fa1e86df3c49df6bbfb15e213f2.
# P-OPS2-B adaptation from d8f8c0d981354e3b81ef4c8671ae6cc2481a3af9; D matrix-engine override preserved.
"""Dependency-scheduled elementwise fusion and state commit for LATCH-OPS (P-OPS-A).

Adapted from P-OPS-A e4df765e30b7abd6bfcc3110703fa2f4b4cf2947. The integrated
``lower(..., policy="auto")`` entry point selects this schedule for flat graphs.
It accepts Stage 4 operation IR (including general optimizer state and counters)
and emits one cooperative CUDA program. Its central effects scheduler owns
fusion, materialization and placement:

* Cheap pure elementwise producers are inlined into their consumers as the
  identical per-operation expression at the consumer's index. A BF16 result is
  still rounded (`__float2bfloat16_rn`) and widened exactly as a load of the
  stored BF16 value would be, so eliminating an intermediate buffer preserves
  the required rounding bit for bit.
* Remaining elementwise operations are placed in the earliest dependency-legal
  grid phase (ASAP list scheduling), across dtypes and shapes. Several
  independent elementwise loops, and at most one non-matrix heavy operation,
  may share one phase. A value consumed at the same flat index in the same
  phase loop stays in a register; it is stored only if an output, a state
  write, a heavy operation or a later phase reads it.
* A state write whose source is a same-shape elementwise value is committed in
  place when every other read of the old state precedes it (or reads the same
  index in the same loop, before the store). The returned new-state output then
  views the committed input storage. Other writes keep the final commit copy.
* Elementwise loops issue all loads of an unrolled group before any store and
  use 32-bit indices with merged contiguous dimensions.

The unrolled loads-before-arithmetic loop and in-place state update follow the
older Latch S102G `opt_flat` optimizer pattern (private Latch, tag latch-v0.6.0,
commit 0987d803f7b8be0d41cc653efd928e6581a97026,
src/megakernel_releases/s108/_current/ordinary/native/lib/optim.cuh). That code
is not copied: it is a fixed AdamW/Muon kernel over a segment table, while this
emitter generates loops from captured ATen elementwise operations.

``coschedule_matrix=True`` groups independent matrices once. C matrix specs use
the shared C group emitter; pinned bodies use their existing vbase argument.
The shared emitters also supply D attention, reductions and ordered split-K.
Eligibility, arithmetic changes and fallback decisions are recorded in the
manifest; the effects scheduler does not run another donor's fusion pass.
"""
from __future__ import annotations

from . import attention_lower

import hashlib
from types import SimpleNamespace
import json
import math
import re

from .vendor import plan as arena_plan
from . import arena_assign  # LATCH-BUILD-600: same offsets as arena_plan.assign, without its quadratic scan

UNROLL_MAX = 8
P6B_TICK_BASE, P6B_TICK_STRIDE = 4096, 528  # P6-B per-block arrival ticks (diagnostic; <= 4 x 132 blocks)
P6B_BATCH_LIVE = 64                   # P6-B load-batched loops: loop values held per batch
RESIDENT_THREADS = 128 * 256          # default H100 launch cap x CTA width
TRANSCENDENTAL = {"erf", "sin", "cos", "tanh", "exp", "expm1", "log1p", "reciprocal", "log", "sqrt", "rsqrt", "sigmoid", "silu", "silu_backward",
                  "pow", "div", "sigmoid_backward", "tanh_backward"}
INLINE_TOTAL_LIMIT = 12                # recomputed operations per inlined value use
INLINE_WORK_LIMIT = 24                 # summed over all uses
P7B_REMAT_MIN = 1 << 16                # P7-B rematerialization applies to values of at least this many elements
P7B_VEC8 = (
    "#ifndef LATCH_P7B_VEC8_DEFINED\n#define LATCH_P7B_VEC8_DEFINED\n"
    "// P7-B 16-byte accesses of 8 consecutive elements (callers pass 32-byte / 16-byte aligned FP32 / BF16 addresses)\n"
    "__device__ __forceinline__ void latch_p7b_ld8(const float* p, float* v) {\n"
    "  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);\n"
    "  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;\n}\n"
    "__device__ __forceinline__ void latch_p7b_ld8(const __nv_bfloat16* p, __nv_bfloat16* v) {\n"
    "  const uint4 q = *reinterpret_cast<const uint4*>(p);\n"
    "  const unsigned w[4] = {q.x, q.y, q.z, q.w};\n"
    "#pragma unroll\n  for (int k = 0; k < 4; ++k) {\n"
    "    v[2 * k] = __ushort_as_bfloat16((unsigned short)(w[k] & 0xffffu));\n"
    "    v[2 * k + 1] = __ushort_as_bfloat16((unsigned short)(w[k] >> 16));\n  }\n}\n"
    "__device__ __forceinline__ void latch_p7b_st8(float* p, const float* v) {\n"
    "  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);\n"
    "  *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);\n}\n"
    "__device__ __forceinline__ void latch_p7b_st8(__nv_bfloat16* p, const __nv_bfloat16* v) {\n"
    "  unsigned w[4];\n"
    "#pragma unroll\n  for (int k = 0; k < 4; ++k)\n"
    "    w[k] = (unsigned)__bfloat16_as_ushort(v[2 * k]) | ((unsigned)__bfloat16_as_ushort(v[2 * k + 1]) << 16);\n"
    "  *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);\n}\n#endif\n")
P7B_CONE_COST = 48                     # P7-B cone rematerialization: recomputed cost bound per value (inline units)
P7B_MAX_PARTS = 4                      # P7-B segment units per segment (rows split over CTAs, last-arriver combine)
P7B_CLAIM_HELPER = (
    "#ifndef LATCH_P7B_CLAIM_DEFINED\n#define LATCH_P7B_CLAIM_DEFINED\n"
    "// P7-B: block-uniform claim of the next item (counter zeroed at launch); n when every item is taken.\n"
    "__device__ __forceinline__ unsigned latch_p7b_claim(unsigned* ctr, unsigned n) {\n"
    "  __shared__ unsigned latch_p7b_slot;\n  __syncthreads();\n"
    "  if (threadIdx.x == 0) latch_p7b_slot = atomicAdd(ctr, 1u);\n  __syncthreads();\n"
    "  const unsigned v = latch_p7b_slot;\n  return v < n ? v : n;\n}\n#endif\n")
P7B_COMPANION_KINDS = ("reduction", "embedding_gradient", "embedding_gradient_rank", "attention_mixed_bias")
P7B_SPLIT_GRID = 264                   # P7-B split generic reductions: resident CTAs the parts are sized for
P7B_SPLIT_MIN_EXTENT = 2048            # P7-B split generic reductions: minimum reduced extent
P7B_SPLIT_MIN_ROWS = 512               # P7-B split generic reductions: minimum reduced elements per part
P7B_PRIORITY_MARGIN = 8                # a product joins a full matrix phase when this much more critical
LIVE_VALUE_CAP = 64                    # unroll x loop values held between loads and stores
VECTOR_MIN_ELEMENTS = 65536            # x4 vector loops only where each thread gets several groups
VECTOR_HELPERS = r"""
#ifndef LATCH_OPS_VECTOR4
#define LATCH_OPS_VECTOR4
#include <cuda_bf16.h>
__device__ __forceinline__ void latch_ld4(const float* p, float (&v)[4]) {
  const float4 t = *reinterpret_cast<const float4*>(p); v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
}
__device__ __forceinline__ void latch_ld4(const __nv_bfloat16* p, __nv_bfloat16 (&v)[4]) {
  const uint2 t = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&t.x);
  const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&t.y);
  v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
}
__device__ __forceinline__ void latch_st4(float* p, float a, float b, float c, float d) {
  *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}
__device__ __forceinline__ void latch_st4(__nv_bfloat16* p, __nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d) {
  __nv_bfloat162 x, y; x.x = a; x.y = b; y.x = c; y.y = d;
  uint2 t; t.x = *reinterpret_cast<const unsigned*>(&x); t.y = *reinterpret_cast<const unsigned*>(&y);
  *reinterpret_cast<uint2*>(p) = t;
}
#endif
"""
GRID_MODEL = 128                       # default launch cap used by the placement cost model
BYTES_PER_US = 2.6e6                   # modeled achievable HBM bytes/us (H100), whole grid
_TOKEN = re.compile(r"@@LOAD(\d+)@@")
P6B_CHUNK_U = 4      # P6-B chunked multi-tensor loops: quads per thread per chunk
P8E_SPLIT_GRID = 264                   # P8-E split generic products: resident CTAs the K parts are sized for
P6B_MULTI_MIN = 3   # components of one element functor type merged into a multi-tensor loop
# P6-B: inline IEEE division for large element loops / pair epilogues (same a / b as the outlined latch_fdiv).
P6B_INLINE_DIV_MIN = 65536
P6B_INLINE_DIV = ("#ifndef LATCH_P6B_FDIV_DEFINED\n#define LATCH_P6B_FDIV_DEFINED\n"
                  "__device__ __forceinline__ float latch_fdiv_inline(float a, float b) { return a / b; }\n#endif\n")
# P6-B indexed selections lowered as per-element expressions (fusion option p6b_index_elementwise).
P6B_SELECTIONS = {"cat", "arange", "index", "embedding"}
P6B_ROW_HELPER = (
    "#ifndef LATCH_P6B_ROW_DEFINED\n#define LATCH_P6B_ROW_DEFINED\n"
    "// Checked gather row: an out-of-range index aborts the kernel (as the assert of the copy kernels does).\n"
    "__device__ __forceinline__ unsigned latch_p6b_row(long long r, long long n) {\n"
    "  if (r < 0 || r >= n) __trap();\n  return static_cast<unsigned>(r);\n}\n#endif\n")
SMALL_REDUCTION_EXTENT = 32            # one warp of the frozen 256-thread CTA tree
SMALL_REDUCTION_OUTPUTS = 4096


def _butterfly_lane0(leaves):
    """Lane 0 of `v += shfl_xor(v, o)` for o = 16..1, lanes >= len(leaves) holding +0.

    Valid when no operand can be -0 (x + 0 == x then); the second-stage butterfly
    over warp partials of a <= 32-element reduction adds only such zeros.
    """
    lanes = list(leaves) + [None] * (32 - len(leaves))
    for offset in (16, 8, 4, 2, 1):
        lanes = [lanes[l] if lanes[l ^ offset] is None else lanes[l ^ offset] if lanes[l] is None
                 else f"({lanes[l]} + {lanes[l ^ offset]})" for l in range(32)]
    return lanes[0]


SPLIT_MIN_EXTENT = 16384     # C-OPS2-C split reductions: only long reductions (no integrated baseline case)
SPLIT_MAX_RESULTS = 4096     # ... with few outputs, where one CTA per output leaves the grid idle


def _split_reduction_code(*, out, pack, read, extent, results, column, rounded, scale, out_words):
    """Two-pass fixed-order FP32 reduction for long extents with few results (C-OPS2-C 0b1ee0296, copied unchanged).

    Pass 1 sums S contiguous slices of the reduced index per result into partials [S][R];
    column mode (kept innermost axis unit-stride) gives each thread one (slice, result) with
    consecutive results on consecutive lanes; row mode gives each (slice, result) to a warp
    with lane-strided reads and a butterfly sum. A grid barrier precedes pass 2, which sums
    each result's partials in slice order. Deterministic, but a different summation order
    from the one-CTA-per-result tree (the reference policy keeps that tree).
    """
    if column:
        slices = max(1, min(-(-32768 // results), -(-extent // 64)))
    else:
        slices = max(1, min(-(-1024 // results), -(-extent // 256)))
    length = -(-extent // slices)
    slices = -(-extent // length)
    part0 = -(-out_words // 64) * 64
    items = slices * results
    if column:
        first = f"""for (long long it = (long long)blockIdx.x * blockDim.x + threadIdx.x; it < {items}LL; it += (long long)gridDim.x * blockDim.x) {{
  const long long o = it % {results}LL, r0 = (it / {results}LL) * {length}LL;
  const long long r1 = r0 + {length}LL < {extent}LL ? r0 + {length}LL : {extent}LL;
  float acc = 0.0f;
  for (long long r = r0; r < r1; ++r) acc += {read};
  part[it] = acc;
}}"""
    else:
        first = f"""for (long long it = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; it < {items}LL; it += ((long long)gridDim.x * blockDim.x) >> 5) {{
  const int lane = threadIdx.x % 32;
  const long long o = it % {results}LL, r0 = (it / {results}LL) * {length}LL;
  const long long r1 = r0 + {length}LL < {extent}LL ? r0 + {length}LL : {extent}LL;
  float acc = 0.0f;
  for (long long r = r0 + lane; r < r1; r += 32) acc += {read};
  acc = s101h::warp_sum(acc);
  if (lane == 0) part[it] = acc;
}}"""
    code = f"""{{ // split reduction: {results} results x {extent} elements, {slices} slices ({'column' if column else 'row'} mode)
float* part = reinterpret_cast<float*>({pack}) + {part0}LL;
{first}
grid.sync();
for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < {results}LL; o += (long long)gridDim.x * blockDim.x) {{
  float acc = 0.0f;
  for (long long s = 0; s < {slices}LL; ++s) acc += part[s * {results}LL + o];
  {out}[o] = {rounded('acc' + scale)};
}}
}}"""
    work = -(-items // (256 if column else 8))
    extra = {"engine": "split_two_pass_" + ("column" if column else "row"), "exclusive_phase": True,
             "internal_grid_barriers": 1, "work_items": max(1, work), "slices": slices,
             "arithmetic": "FP32 sequential slice sums, slice-ordered combination"}
    return code, extra, part0 + items


def _u(n):
    return f"{int(n)}u"


def _index32(layout, out_shape, idx):
    """32-bit flat index into `layout` for flat index `idx` of `out_shape`.

    Semantically identical to lower._index (broadcast, strides), with singleton
    dimensions dropped and contiguous/broadcast runs merged so a same-shape
    contiguous read is simply `idx`.
    """
    from .lower import LoweringError
    if len(layout.shape) > len(out_shape):
        raise LoweringError(f"cannot broadcast {layout.shape} to {out_shape}")
    padded = (1,) * (len(out_shape) - len(layout.shape)) + tuple(layout.shape)
    strides = (0,) * (len(out_shape) - len(layout.shape)) + tuple(layout.strides)
    dims = []
    for a, b, s in zip(padded, out_shape, strides):
        if a not in (1, b):
            raise LoweringError(f"cannot broadcast {layout.shape} to {out_shape}")
        if b == 1:
            continue
        dims.append([b, 0 if a == 1 else s])
    merged = []                       # inner first
    for size, stride in reversed(dims):
        if merged and stride == merged[-1][1] * merged[-1][0]:
            merged[-1] = [merged[-1][0] * size, merged[-1][1]]
        else:
            merged.append([size, stride])
    merged.reverse()                  # outer first
    terms, tail = [], math.prod(size for size, _ in merged)
    for k, (size, stride) in enumerate(merged):
        tail //= size
        if stride == 0:
            continue
        coord = idx if tail == 1 else f"({idx} / {_u(tail)})"
        if k != 0:
            coord = f"({coord} % {_u(size)})"
        terms.append(coord if stride == 1 else f"{coord} * {_u(stride)}")
    return " + ".join(terms) or "0u"


class _Node:
    __slots__ = ("name", "kind", "shape", "dtype", "render", "code", "extra", "loads", "forced",
                 "base", "target", "own_cost", "order", "direct")

    def __init__(self, **kw):
        for key in self.__slots__:
            setattr(self, key, kw.get(key))


def _matrix_us(k):
    """Modeled 64x64 pinned BF16/FP32 tile time (us) for reduction length k (H100 fit)."""
    return 2.0 + 0.0137 * k


def _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                    forced, numel, idle_heavy=None, deferred=None, preload=None, fixed=None, grid_model=None):
    """Move elementwise register chains later into phases with idle matrix blocks.

    A chain (items joined by same-loop register reads) may occupy any phase from
    its ASAP phase to just before its first external reader (or the last phase).
    Matrix-only phases using at most half the modeled grid leave idle blocks; a
    loop placed there runs only on those blocks and is hidden up to the modeled
    matrix time. Moves are later-only and re-validated by _check_schedule.
    """
    size = {"float32": 4, "bfloat16": 2, "int64": 8, "bool": 1}
    GRID_MODEL = grid_model or globals()["GRID_MODEL"]   # P7-B: the resident grid when given
    names = [n.name for n in items]
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in phase_of:
                continue
            reg = (shape is not None and register_load(node, root, layout, shape, idx)
                   and phase_of[root] == phase_of[node.name])
            readers.setdefault(root, []).append((node.name, reg))
            if reg:
                parent[find(node.name)] = find(root)
    chains = {}
    for node in items:
        if node.kind == "elementwise":
            chains.setdefault(find(node.name), []).append(node.name)
    if any(nodes[r].kind != "elementwise" for members in chains.values() for r in members):
        return

    def heavy_profile(p):
        heavy = [nodes[n] for n in phases[p]["heavy"]]
        if not heavy:
            return 0.0, 0
        main = [h for h in heavy if h.name not in (deferred or {})]
        if idle_heavy and main and all(h.kind in idle_heavy and h.extra.get("work_items") for h in main):
            # P-OPS2-B: item-per-block heavy phases (items on blocks [0, n)) leave the other blocks idle.
            n = sum(int(h.extra["work_items"]) for h in main)
            if 2 * n > GRID_MODEL:
                return 0.0, 0
            return max(idle_heavy[h.kind](h) for h in main), GRID_MODEL - n
        if any(h.kind != "gemm" for h in heavy):
            return 0.0, 0
        tasks = sum(int(h.extra.get("tiles", 0)) for h in heavy)
        if 2 * tasks > GRID_MODEL:
            return 0.0, 0
        if grid_model:
            # P7-B: warpgroup products carry their own modeled time (plan estimate on the resident grid)
            return max(float((h.extra.get("matrix_plan") or {}).get("estimate_us") or _matrix_us(int(h.extra.get("k", 0))))
                       for h in heavy), GRID_MODEL - tasks
        return max(_matrix_us(int(h.extra.get("k", 0))) for h in heavy), GRID_MODEL - tasks

    def work_us(members):
        total = 0.0
        for name in members:
            node = nodes[name]
            count = numel(node.shape)
            for root, layout, shape, idx in loads_of[name]:
                if root in members and shape is not None and register_load(node, root, layout, shape, idx):
                    continue
                total += count * size.get(dtypes.get(root, "float32"), 4)
            if name in forced or name in inplace_source or any(not reg for r, reg in readers.get(name, ())
                                                               if r not in members):
                total += count * size.get(node.dtype, 4)
        return total / BYTES_PER_US
    busy = {p: heavy_profile(p) for p in range(len(phases))}
    load = {p: float((preload or {}).get(p, 0.0)) for p in range(len(phases))}
    work = {key: work_us(members) for key, members in chains.items()}
    for key, members in chains.items():
        p = phase_of[members[0]]
        us, idle = busy[p]
        if idle:
            load[p] += work[key] * GRID_MODEL / idle

    def extra(p, w):
        us, idle = busy[p]
        if not idle:
            return w, 0.0
        t = w * GRID_MODEL / idle
        return max(0.0, load[p] + t - us) - max(0.0, load[p] - us), t
    order = {n: k for k, n in enumerate(names)}
    commit_phase = {t: phase_of[s] for t, s in in_place.items()}
    for key in sorted(chains, key=lambda k: -max(order[m] for m in chains[k])):
        members = set(chains[key])
        if fixed and members & fixed:
            continue                    # P-OPS3-C: row-attached members stay with their reduction
        current = phase_of[chains[key][0]]
        hi = len(phases) - 1
        for m in members:
            for r, reg in readers.get(m, ()):
                if r not in members:
                    hi = min(hi, phase_of[r] - 1)
            for root, layout, shape, idx in loads_of[m]:
                if root in commit_phase and in_place[root] not in members:
                    hi = min(hi, commit_phase[root] - 1)
        if hi <= current:
            continue
        w = work[key]
        us, idle = busy[current]
        if idle:
            load[current] -= w * GRID_MODEL / idle
        best, best_cost = current, extra(current, w)[0]
        for p in range(current + 1, hi + 1):
            cost = extra(p, w)[0]
            if cost < best_cost - 0.05:
                best, best_cost = p, cost
        load[best] += extra(best, w)[1]
        if best != current:
            n_elements = numel(nodes[chains[key][0]].shape)
            loop = phases[current]["loops"][n_elements]
            for m in chains[key]:
                loop.remove(m)
                phase_of[m] = best
            if not loop:
                del phases[current]["loops"][n_elements]
            phases[best]["loops"].setdefault(n_elements, []).extend(chains[key])
            # Keep operation order inside the destination loop (register producers first).
            phases[best]["loops"][n_elements].sort(key=lambda n: order[n])
    empty = [p for p, ph in enumerate(phases) if not ph["heavy"] and not ph["loops"]]
    if empty:
        remap, kept = {}, []
        for p, ph in enumerate(phases):
            if ph["heavy"] or ph["loops"]:
                remap[p] = len(kept)
                kept.append(ph)
        phases[:] = kept
        for n in phase_of:
            phase_of[n] = remap[phase_of[n]]


def _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, error, epilogue_of=None,
                    permuted=None, row_of=None, row_chain=None):
    """Independent re-check of the schedule (the scheduler is not trusted).

    Every read is of a graph input, a value from an earlier phase that has
    storage, or a same-loop register at the identical flat index. Each in-place
    target is read only before its committing phase, or at the committing
    loop's own index (all loads of an unrolled group precede its stores).
    """
    epilogue_of = epilogue_of or {}
    row_of = row_of or {}

    def loop_key(name):
        node = nodes[name]
        if node.kind != "elementwise":
            return None
        key = (phase_of[name], math.prod(node.shape))
        if name in row_of:
            key = key + ("row", row_of[name])
        return key + (epilogue_of[name],) if name in epilogue_of else key
    commit_phase = {target: phase_of[source] for target, source in in_place.items()}
    for node in items:
        p = phase_of[node.name]
        for root, layout, shape, idx in loads_of[node.name]:
            if root in commit_phase:
                source = in_place[root]
                q = commit_phase[root]
                if p > q or (p == q and not (node.kind == "elementwise" and loop_key(node.name) == loop_key(source)
                                              and idx == "i" and layout.contiguous and layout.offset == 0)):
                    raise error(f"schedule check: {node.name} reads old state {root} after/alongside its in-place commit")
                continue
            if root not in phase_of:
                continue
            q = phase_of[root]
            if row_of.get(node.name) == root and q == p:
                continue                    # row-attached member reads its warp's statistic from a register
            if row_chain and q == p:
                engine = node.name if node.name in row_chain else row_of.get(node.name)
                pred = row_chain.get(engine)
                if pred is not None and (root == pred or row_of.get(root) == pred) and (
                        nodes[root].kind != "elementwise" or root in stored):
                    continue                # P6-B: chained row engine (same rows -> same warp; block barrier between)
            if row_of.get(root) == node.name and q == p:
                continue                    # row engine evaluates its own pre-member producer in the leaf
            if row_of.get(node.name) is not None and row_of.get(root) == row_of.get(node.name) and q == p:
                continue                    # members of one row engine share per-row values in registers
            if epilogue_of.get(node.name) == root and q == p and (
                    node.name in (permuted or {}) or (
                        idx == "i" and shape is not None and layout.contiguous and layout.offset == 0
                        and math.prod(shape) == layout.numel == math.prod(node.shape))):
                continue                    # matrix epilogue reads the product's rounded result in registers
            if q < p:
                if nodes[root].kind == "elementwise" and root not in stored:
                    raise error(f"schedule check: {node.name} reads {root} from memory but it has no storage")
                continue
            if not (q == p and shape is not None and register_load(node, root, layout, shape, idx)
                    and loop_key(root) == loop_key(node.name)):
                raise error(f"schedule check: {node.name} (phase {p}) reads {root} produced in phase {q}")


@attention_lower.retry_lowering
def _materialized_attention_refusals(group, values, arch, bodies):
    """Products of the materialized attention that the selected FP32 bodies would refuse (P-OPS4-C).

    Shapes come from Q [b, Tq, D], K^T [b, D, Tk] and V [b, Tk, Dv]; materialized scores, probabilities and
    their gradients are contiguous [b, Tq, Tk] values and Q/K/V are read as dense head-major views, which is
    the most favourable layout the materialized schedule can produce. Uses only shapes and dtypes."""
    from .lower import TensorLayout, _contiguous, _field
    from .matrix_select import plan
    try:
        q = [int(x) for x in _field(values[group["q"]], "shape")]
        kt = [int(x) for x in _field(values[group["kt"]], "shape")]
        v = [int(x) for x in _field(values[group["v"]], "shape")]
    except (KeyError, TypeError, ValueError):
        return ["unknown attention operand shapes"]
    if len(q) != 3 or len(kt) != 3 or len(v) != 3:
        return ["attention operands are not rank-3"]
    b, tq, d = q
    tk, dv = kt[-1], v[-1]

    def dense(shape):
        return TensorLayout(tuple(shape), tuple(_contiguous(shape)), "x")

    def transposed(shape):  # [b, r, c] view of a dense [b, c, r] value
        bb, r, c = shape
        return TensorLayout((bb, r, c), (r * c, 1, r), "x")

    products = {
        "scores": (dense([b, tq, d]), transposed([b, d, tk]), (b, tq, tk)),
        "output": (dense([b, tq, tk]), dense([b, tk, dv]), (b, tq, dv)),
        "probability_gradient": (dense([b, tq, dv]), transposed([b, dv, tk]), (b, tq, tk)),
        "value_gradient": (transposed([b, tk, tq]), dense([b, tq, dv]), (b, tk, dv)),
        "query_gradient": (dense([b, tq, tk]), dense([b, tk, d]), (b, tq, d)),
        "key_gradient": (transposed([b, d, tq]), dense([b, tq, tk]), (b, d, tk)),
    }
    from .matrix_codegen_p7b import select as small_select
    refused = []
    for label, (la, lb, shape) in products.items():
        # Same order as the product lowering: P-OPS-C exact small tiles, then the P-OPS2-A selected bodies.
        small, small_reason = small_select("a", "b", "c", la, lb, shape, "float32", arch=arch)
        if small is not None:
            continue
        selected, reason = plan("a", "b", "c", la, lb, shape, ident=str, arch=arch, bodies=bodies)
        if selected is None:
            refused.append(f"{label} {list(shape)} x K={la.shape[-1]}: {small_reason}; {reason}")
    return refused


def lower_fused(graph, *, retain_all=False, coschedule_matrix=True, place_by_slack=True,
                matrix_policy="auto", split_k=True, exact_reductions=True, fuse_attention=True,
                ordered_index_gradients=True, target_arch="sm_90", pops2_policy=True, matrix_engine=None,
                matrix_bodies=None, attention_min_keys=None, warp_rows=None,
                matrix_runtime_layout=False, matrix_residency=1, pops4c=True, pops2b=True, pops2b_prefetch=False,
                pops2b_epilogue=True, pops2b_permuted=False, pops2b_casts=False, pops2b_idle=True,
                pops2b_plans=False, pops2b_defer=False, pops2b_idle_fill=None, pops2b_epi_cost=None,
                pops2b_epi_stream_gate=True, pops2b_defer_epilogue=True, pops2b_barrier=True,
                attention_schedule=True, attention_store_scores=True, attention_debug_repeat=False,
                attention_pipeline=True, pops5d=True, pops5d_defines=None, pops5d_absorb=True,
                pops5d_bias_phase=True, pops5d_trace=False, pops5d_store_probs=True,
                pops3c=True, pops3c_attach=True, pops3c_share=True, pops3c_column=True, pops3c_row=True,
                pops3c_pre=False, pops3c_col_div=2, pops3c_attach_shared=False, pops3c_outline=True,
                pops3c_factor=True, pops3c_pow1=True, pops3c_outline_ops=True, pops3c_overlap=False,
                pops3c_outline_headers=False, pops3c_col_group=8,
                split_reductions=True, pops4a="auto", pops4a_cols=2, pops4a_unroll=8, pops4a_chunk=8192,
                pops4a_items=128, pops4a_last=False, pops4a_outline="inline", pops4a_merge=True, pops4a_prefetch=0,
                pops4a_stage=False, pops4a_stage_bytes=65536, pops4a_vec=1, pops4a_vec_full=True,
                pops4a_floor=True, pops4a_over_exact="large", pops4a_last_column=False, scan_engine="auto",
                p6b_scalar_chain=True, p6b_trap_asserts=True, p6b_column_tree=True,
                p6b_index_elementwise=True, p6b_epilogue_diagnostics=False, p6b_epilogue_full_grid_cost=None,
                p6b_loop_balance=False, p6b_row_registers=False, p6b_row_chain=False, p6b_reduction_share=False,
                p6b_splitk=False, p6b_wg_epilogue=False, p6b_cse=False, p6b_rotate_exclude=False,
                p6b_scalar_store=False, p6b_tree_merge=False, p6b_node_cse=False, p6b_inline_div=False,
                p6b_multi_tensor_loops=False, p6b_rotate_grid=False, p6b_loop_batch=False, p6b_block_ticks=False,
                p6b_wg_split=False, p6b_split_grid=None, p6b_tree_spread=False, p6b_epilogue_prefetch=False,
                p6b_tree_quads=False, p6b_wg_split_edge=False, p6b_index32=False,
                p6b_split_objective='model', p6b_loop_batch_unrolled=False,
                p6b_generic_unroll=False, p6b_loop_batch_heavy=0, p6b_chunk_multi_loops=False, p8e_small_multi=False,
                p8e_generic_split=False, p8e_chunk_small=False, p8e_fast_tanh=False,
                p7b_segment=False, p7b_remat=0, p7b_priority=False, p7b_companions=False,
                p7b_staged_epilogue=False, p7b_segment_parts=1, p7b_tma_arrive=False, p7b_segment_index=False,
                p7b_segment_pair=False, p7b_tma_persist=False, p7b_dynamic_items=False, p7b_segment_vec=False,
                p7b_cone=False, p7b_attention_hosts=False, p7b_list_schedule=False, p7b_attention_products=False,
                p7b_inline_bytes=False, p7b_claim_async=False, p7b_axis_order=False, p7b_split_generic=False, p7b_row_epilogue=False,
                p7b_heavy_batch=0, p7b_slack_grid=0, p7b_segment_affine=False,
                p7b_epi_affine=False, p7b_pass_prefetch=False, p9e_epi_colsum=False, p9e_remat_stored=0, p9e_remat_rowstat=False, p9e_attention_last=False, p9e_defer_attention=0, p9e_split_alone=0, p9e_defer_heavy=False, p9e_split_colsum=0, p9e_epi_unroll=0, p9e_ksum=False,
                p7b_wave_group=0, p7b_small_epilogue=0,
                p7b_attention_bump=False, p7b_defer_reductions=False,
                p7b_last_small=0, p7b_cone_recheck=False, p7b_chunk_live=0,
                p7b_inline_pure=False):
    # P-OPS5-D: fused mixed-precision short attention (ops/mixed_attention.py); pops5d=False reproduces the
    # previous lowering byte for byte. pops5d_store_probs (default) saves the forward's FP32 probabilities in a
    # fragment-native layout instead of recomputing them in the backward (memory for time; measured gpu-11).
    # A selects actual FP32 products; B schedules explicit BF16 products. D's
    # engine dictionary remains available for staged accumulation/layout paths.
    matrix_engine = ("auto" if pops2_policy else "popsb") if matrix_engine is None else matrix_engine
    matrix_bodies = (("simt", "popsb") if pops2_policy else None) if matrix_bodies is None else tuple(matrix_bodies)
    attention_override = attention_min_keys is not None
    from .lower import (GeneratedProgram, LoweringError, TensorLayout, VENDOR, _header_path, _BYTES, _CTYPES,
                        _contiguous, _dtype, _field, _flatten_refs, _float, _double, _ident, _numel,
                        _ref, _reshape_strides, _translation_unit, slice_view, slice_backward_index)

    raw_values = _field(graph, "values")
    values = raw_values if isinstance(raw_values, dict) else {_field(v, "name"): v for v in raw_values}
    has_bf16 = any(_dtype(value) == "bfloat16" for value in values.values())
    # The A short-context choice was qualified for all-FP32 operation graphs;
    # B's BF16 graph qualification retains fused attention at short contexts.
    if not attention_override:
        attention_min_keys = 512 if pops2_policy and not has_bf16 else None
    warp_rows = bool(pops2_policy and not has_bf16) if warp_rows is None else bool(warp_rows)
    attention_policy = {"explicit_override": attention_override, "has_bfloat16": has_bf16,
                        "materialize_at_most_keys": attention_min_keys,
                        "reason": ("explicit attention_min_keys override" if attention_override else
                                   "A short-context materialization for FP32-only graphs" if not has_bf16 and pops2_policy else
                                   "preserve fused attention for explicit BF16 projection graphs" if has_bf16 else
                                   "A policy disabled")}
    names = set(values)
    inputs = list(_field(graph, "inputs", _field(graph, "input_names", [])))
    structured_outputs = _field(graph, "outputs", {})
    if isinstance(structured_outputs, dict):
        outputs = [structured_outputs["loss"]]
        parameter_names = _field(graph, "metadata", {}).get("parameter_names", tuple(structured_outputs.get("grads", structured_outputs["params"])))
        for kind in (("grads",) if "grads" in structured_outputs else ()) + ("params",):  # step 15: consumed gradients
            outputs.extend(structured_outputs[kind][name] for name in parameter_names)
        optimizer_group = "optimizer_state" if "optimizer_state" in structured_outputs else "momentum"
        optimizer_names = _field(graph, "metadata", {}).get("optimizer_state_names", parameter_names)
        outputs.extend(structured_outputs[optimizer_group][name] for name in optimizer_names)
        outputs.extend(structured_outputs.get("state", {})[name]
                       for name in _field(graph, "metadata", {}).get("state_names", ()))
    else:
        outputs = list(structured_outputs)
    inputs = [_ref(x, names) or x for x in inputs]
    outputs = [_ref(x, names) or x for x in outputs]
    operations = _field(graph, "operations", _field(graph, "nodes", []))
    p9e_colsum_records = []
    if p9e_split_colsum and not retain_all and target_arch == "sm_90":
        # P9-E (amendment A8): a full column sum of a rank-3 [S, T, D] FP32 value over its two leading axes
        # (D <= p9e_split_colsum) is evaluated as per-segment column sums over T ([S, 1, D], FP32) followed by
        # their sum over S. The per-segment sums are middle-axis column reductions, which the P7-B segment engine
        # evaluates from registers when it produces the summed value (no re-read of the [S, T, D] source).
        # Same terms, FP32 accumulation; only the association order of the additions changes.
        from dataclasses import replace as _dc_replace
        from .ir import Operation as _IrOperation, Ref as _IrRef, Value as _IrValue
        values = dict(values)
        rewritten_ops = []
        for op_ in operations:
            target_ = str(_field(op_, "target", _field(op_, "op"))).removeprefix("torch.ops.")
            args_ = list(_field(op_, "args", []))
            kwargs_ = dict(_field(op_, "kwargs", {}))
            out_ = op_.name if isinstance(op_, _IrOperation) and tuple(op_.outputs) == (op_.name,) else None
            src_ = _ref(args_[0], names) if args_ else None
            ok_ = (out_ is not None and target_ == "aten.sum.dim_IntList" and src_ in values and not kwargs_.get("dtype")
                   and len(tuple(_field(values[src_], "shape"))) == 3 and len(args_) >= 2)
            if ok_:
                shp_ = tuple(int(x) for x in _field(values[src_], "shape"))
                dims_ = args_[1]
                dims_ = sorted({int(d) % 3 for d in ([dims_] if isinstance(dims_, int) else dims_)}) if dims_ else []
                keep_ = bool(args_[2] if len(args_) > 2 else kwargs_.get("keepdim", False))
                ok_ = (dims_ == [0, 1] and _dtype(values[src_]) == "float32" and _dtype(values[out_]) == "float32"
                       and shp_[0] >= 2 and shp_[1] >= 2 and 32 <= shp_[2] <= int(p9e_split_colsum)
                       and len(args_) <= 3 and not set(kwargs_) - {"keepdim"}
                       and _field(values[src_], "alias_of") is None)   # views (e.g. attention gradients) keep their engines
            if not ok_:
                rewritten_ops.append(op_)
                continue
            mid_ = f"{out_}_p9cs"
            if mid_ in values or mid_ in names:
                raise LoweringError(f"P9-E column-sum split name collision: {mid_}")
            S_, T_, D_ = shp_
            values[mid_] = _IrValue(mid_, (S_, 1, D_), "float32", (D_, D_, 1), producer=mid_)
            names.add(mid_)
            deps_ = tuple(_field(op_, "dependencies", ()) or ())
            rewritten_ops.append(_IrOperation(mid_, "aten.sum.dim_IntList", (args_[0], [1], True), {}, (src_,), (mid_,),
                                              deps_, (src_,), (mid_,), reduction_axes=(1,)))
            rewritten_ops.append(_dc_replace(op_, args=(_IrRef(mid_), [0, 1], keep_), kwargs={}, inputs=(mid_,),
                                             dependencies=(mid_,), reads=(mid_,), reduction_axes=(0, 1)))
            p9e_colsum_records.append({"sum": out_, "source": src_, "segments": S_, "segment_rows": T_, "width": D_,
                                       "segment_sums": mid_})
        operations = rewritten_ops
    writes = dict(_field(graph, "state_writes", {}) or {})
    if not writes and isinstance(structured_outputs, dict):
        for name in inputs:
            value = values[name]
            role = _field(value, "role")
            if role in ("parameter", "momentum", "optimizer_state", "state"):
                kind = "params" if role == "parameter" else role
                writes[name] = structured_outputs[kind][_field(value, "parameter")]
    if isinstance(writes, list):
        writes = dict(writes)
    writes = {(_ref(k, names) or k): (_ref(v, names) or v) for k, v in writes.items()}
    p9e_ksum_records = {"rewritten": [], "candidates": []}
    if p9e_ksum and not retain_all and target_arch == "sm_90":
        # P9-E (candidate amendment A9): a BF16 sum of BF16 products with equal [M, N] (through views), each product
        # and intermediate used once, becomes one K-segmented product (latch_ops.mm_ksum): sum_s A_s B_s accumulated
        # in FP32 over the segments' K steps in segment order and rounded once, instead of one rounding per product
        # and per add. The sum's result keeps its name and shape (a view of the segmented product).
        from dataclasses import replace as _dc_replace
        from .ir import Operation as _IrOperation, Ref as _IrRef, Value as _IrValue
        values = dict(values)
        producer_ = {o.name: o for o in operations if isinstance(o, _IrOperation)}
        users_ = {}
        for o in operations:
            for r_ in set(_flatten_refs((list(_field(o, "args", [])), dict(_field(o, "kwargs", {}))), names)):
                users_[r_] = users_.get(r_, 0) + 1
        protected_ = set(outputs) | set(writes.values()) | set(writes.keys()) | set(inputs)
        view_t_ = {"aten.view.default", "aten._unsafe_view.default", "aten.reshape.default"}

        ks_cast_ = [False, None]   # [refuse casts, last BF16 -> FP32 cast template]
        ks_root_ = [None]

        def ks_trace(ref, depth=0):
            o = producer_.get(ref)
            if o is None or ref in protected_ or users_.get(ref, 0) != 1:
                return None
            t_ = str(o.target).removeprefix("torch.ops.")
            if t_ in view_t_ and o.args:
                sub = ks_trace(_ref(o.args[0], names), depth)
                return None if sub is None else [(m_, ch + [ref]) for m_, ch in sub]
            if t_ == "aten.mm.default" and _dtype(values[ref]) == "bfloat16":
                return [(o, [ref])]
            if (t_ == "aten._to_copy.default" and o.args and _dtype(values[ref]) == "float32"
                    and _ref(o.args[0], names) is not None and _dtype(values[_ref(o.args[0], names)]) == "bfloat16"
                    and set(o.kwargs) <= {"dtype", "layout", "device", "pin_memory", "memory_format", "non_blocking"}
                    and str(o.kwargs.get("memory_format", "preserve")).find("channels") < 0 and not ks_cast_[0]):
                ks_cast_[1] = o
                sub = ks_trace(_ref(o.args[0], names), depth)
                return None if sub is None else [(m_, ch + [ref]) for m_, ch in sub]
            if (t_ == "aten.add.Tensor" and depth < 3 and len(o.args) == 2 and not o.kwargs
                    and _dtype(values[ref]) == _dtype(values[ks_root_[0]])):
                a_ = ks_trace(_ref(o.args[0], names), depth + 1)
                b_ = ks_trace(_ref(o.args[1], names), depth + 1)
                if a_ is None or b_ is None:
                    return None
                return [(m_, ch + [ref]) for m_, ch in a_ + b_]
            return None
        consumers_ = {}
        for o in operations:
            if isinstance(o, _IrOperation):
                for r_ in set(_flatten_refs((list(_field(o, "args", [])), dict(_field(o, "kwargs", {}))), names)):
                    consumers_.setdefault(r_, []).append(o)
        chains_ = []
        for o in operations:
            if (isinstance(o, _IrOperation) and str(o.target).endswith("aten.mm.default") and len(chains_) < 24
                    and o.name in ("mm_51", "mm_53", "mm_55", "mm_57", "mm_59", "mm_44", "mm_46")):
                ch_, cur_ = [], o.name
                for _ in range(4):
                    cs_ = consumers_.get(cur_, [])
                    ch_.append([(str(c_.target).removeprefix("torch.ops."), c_.name, _dtype(values[c_.name]),
                                 str(c_.args)[:120], str(c_.kwargs)[:60]) for c_ in cs_][:3])
                    if len(cs_) != 1:
                        break
                    cur_ = cs_[0].name
                chains_.append({"mm": o.name, "dtype": _dtype(values[o.name]), "chain": ch_})
        p9e_ksum_records["mm_consumers"] = chains_
        dropped_, replaced_ = set(), {}
        why_ = {}
        p9e_ksum_records["why"] = why_
        for o in reversed(operations):
            if not isinstance(o, _IrOperation) or o.name in dropped_:
                continue
            t_ = str(o.target).removeprefix("torch.ops.")
            if t_ == "aten.add.Tensor":
                why_.setdefault("adds", []).append([o.name, len(o.args), str(o.kwargs)[:40], _dtype(values[o.name]),
                                                    [type(a).__name__ for a in o.args]]) if len(why_.get("adds", [])) < 12 else None
            if (t_ != "aten.add.Tensor" or len(o.args) != 2 or o.kwargs
                    or _dtype(values[o.name]) not in ("bfloat16", "float32")):
                continue
            ra_, rb_ = _ref(o.args[0], names), _ref(o.args[1], names)
            if ra_ is None or rb_ is None:
                continue
            ks_root_[0] = o.name
            ks_cast_[0] = _dtype(values[o.name]) == "bfloat16"   # FP32 sums: every term through a BF16 -> FP32 cast
            ks_cast_[1] = None
            ta_, tb_ = ks_trace(ra_, 1), ks_trace(rb_, 1)
            cast_tpl_ = ks_cast_[1]
            if ta_ is not None and tb_ is not None and _dtype(values[o.name]) == "float32":
                n_casts_ = sum(1 for m_, ch in ta_ + tb_ for x_ in ch
                               if str(producer_[x_].target).removeprefix("torch.ops.") == "aten._to_copy.default")
                if n_casts_ != len(ta_ + tb_) or cast_tpl_ is None:
                    ta_ = None
            rec_ = {"sum": o.name, "a": ra_, "b": rb_, "a_terms": None if ta_ is None else [m_.name for m_, _ in ta_],
                    "b_terms": None if tb_ is None else [m_.name for m_, _ in tb_]}
            if ta_ is None or tb_ is None:
                if ((ta_ is None) != (tb_ is None) and len(p9e_ksum_records["candidates"]) < 40):
                    p9e_ksum_records["candidates"].append(rec_)
                continue
            terms_ = ta_ + tb_
            shapes_ = {tuple(_field(values[m_.name], "shape")) for m_, _ in terms_}
            ks_ = [tuple(_field(values[_ref(m_.args[0], names)], "shape"))[-1] for m_, _ in terms_]
            if not (2 <= len(terms_) <= 3 and len(shapes_) == 1 and len(set(ks_)) == 1 and ks_[0] % 64 == 0
                    and _numel(next(iter(shapes_))) == _numel(tuple(_field(values[o.name], "shape")))):
                rec_["refused"] = "terms/shapes"
                p9e_ksum_records["candidates"].append(rec_)
                continue
            M_, N_ = next(iter(shapes_))
            # TMA warpgroup products only (the lowering requires it), and only while the fused product still fills
            # the grid with 128 x 64 tiles: fusing divides the task count by the segment count, which lengthens the
            # critical path of under-filled products (measured: DiT-S 0.982 -> 1.003, gpu-52a).
            if (M_ < 1024 or N_ % 64 or ks_[0] < 128
                    or -(-M_ // 128) * -(-N_ // 64) < int(p6b_split_grid or 264)):
                rec_["refused"] = "small product"
                p9e_ksum_records["candidates"].append(rec_)
                continue
            ks_name_ = f"{o.name}_p9ks"
            if ks_name_ in values or ks_name_ in names:
                raise LoweringError(f"P9-E K-segment name collision: {ks_name_}")
            values[ks_name_] = _IrValue(ks_name_, (M_, N_), "bfloat16", (N_, 1), producer=ks_name_)
            names.add(ks_name_)
            ks_args_ = tuple(x for m_, _ in terms_ for x in (m_.args[0], m_.args[1]))
            ks_reads_ = tuple(dict.fromkeys(_ref(x, names) for x in ks_args_))
            ks_op_ = _IrOperation(ks_name_, "latch_ops.mm_ksum", ks_args_, {}, ks_reads_, (ks_name_,), (), ks_reads_,
                                  (ks_name_,))
            out_shape_ = list(_field(values[o.name], "shape"))
            if _dtype(values[o.name]) == "bfloat16":
                view_op_ = _IrOperation(o.name, "aten.view.default", (_IrRef(ks_name_), out_shape_),
                                        {}, (ks_name_,), (o.name,), (ks_name_,), (ks_name_,), (), synchronization="view")
                values[o.name] = _dc_replace(values[o.name], alias_of=ks_name_)
                replaced_[o.name] = (ks_op_, view_op_)
            else:
                ksv_ = f"{ks_name_}_v"
                values[ksv_] = _IrValue(ksv_, tuple(out_shape_), "bfloat16",
                                        tuple(int(x) for x in _contiguous(tuple(out_shape_))), producer=ksv_,
                                        alias_of=ks_name_)
                names.add(ksv_)
                view_op_ = _IrOperation(ksv_, "aten.view.default", (_IrRef(ks_name_), out_shape_), {}, (ks_name_,),
                                        (ksv_,), (ks_name_,), (ks_name_,), (), synchronization="view")
                cast_op_ = _IrOperation(o.name, "aten._to_copy.default", (_IrRef(ksv_),), dict(cast_tpl_.kwargs),
                                        (ksv_,), (o.name,), (ksv_,), (ksv_,), (o.name,))
                replaced_[o.name] = (ks_op_, view_op_, cast_op_)
            for m_, ch in terms_:
                dropped_.update(ch)
            rec_["terms"] = [m_.name for m_, _ in terms_]
            p9e_ksum_records["rewritten"].append(rec_)
        if replaced_:
            new_ops_ = []
            for o in operations:
                name_ = getattr(o, "name", None)
                if name_ in replaced_:
                    new_ops_.extend(replaced_[name_])
                elif name_ in dropped_:
                    continue
                else:
                    new_ops_.append(o)
            operations = new_ops_
    layouts = {}
    for name in inputs:
        v = values[name]
        shape = tuple(int(x) for x in _field(v, "shape"))
        strides = tuple(_field(v, "strides", _field(v, "stride", _contiguous(shape))))
        layouts[name] = TensorLayout(shape, strides, name)
        if any(d <= 0 for d in shape) or any(s < 0 for s in strides):
            raise LoweringError(f"empty/negative-stride input {name} unsupported")
    dtypes = {name: _dtype(value) for name, value in values.items()}
    for name, value in values.items():
        if dtypes[name] not in _CTYPES:
            raise LoweringError(f"{name}: dtype {dtypes[name]} has no qualified storage representation")
        if _field(value, "role") == "constant":
            raise LoweringError(f"{name}: captured tensor constants need immutable runtime binding; pass this tensor as an explicit input")
        _ident(name)

    def output_name(op):
        result = _field(op, "output", _field(op, "name"))
        if result is None:
            outs = _field(op, "outputs")
            if len(outs) != 1:
                raise LoweringError("multi-result ATen operation must be decomposed")
            result = outs[0]
        return _ref(result, names) or result

    undefined = {output_name(op) for op in operations
                 if str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0] == "empty_like"}
    if undefined.intersection(outputs) or undefined.intersection(writes.values()):
        raise LoweringError("uninitialized empty_like tensor cannot be returned or committed")
    for op in operations:
        target = str(_field(op, "target", _field(op, "op")))
        base = target.removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        args = list(_field(op, "args", []))
        kwargs = dict(_field(op, "kwargs", {}))
        used = undefined.intersection(_flatten_refs((args, kwargs), names))
        if used and not (base == "fill" and args and used == {_ref(args[0], names)}
                         and not undefined.intersection(_flatten_refs((args[1:], kwargs), names))):
            raise LoweringError(f"{target}: reads undefined empty_like value(s) {sorted(used)}; only functional fill is admitted")

    nodes = {}            # root name -> _Node, in operation order
    outline_ops = pops3c and pops3c_outline and pops3c_outline_ops and not retain_all
    factored_values = {}  # P-OPS3-C: broadcast value -> its once-per-source-element node
    protected_names = set(outputs) | set(writes.values())
    consumer_bases = {}   # value -> base targets of the operations reading it (factoring keeps views pointwise)
    for _op_ in operations:
        _base_ = str(_field(_op_, "target", _field(_op_, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        for _name_ in _flatten_refs((list(_field(_op_, "args", [])), dict(_field(_op_, "kwargs", {}))), names):
            consumer_bases.setdefault(_name_, set()).add(_base_)
    views = []
    helper_parts = []
    extra_headers = []
    matrix_decisions = []
    matrix_operands = []
    token_loads = []      # (ref, shape, idx) per heavy-code placeholder

    def placeholder_read(arg, shape, idx="i", register_values=None):
        ref = _ref(arg, names)
        if ref is None:
            if isinstance(arg, (int, float, bool)):
                return _float(arg)
            raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
        token_loads.append(("index", ref, tuple(shape), idx))
        return f"@@LOAD{len(token_loads) - 1}@@"

    def storage_read(ref, position):
        token_loads.append(("storage", ref, None, position))
        return f"@@LOAD{len(token_loads) - 1}@@"

    p4a_state = {"count": 0, "producer": None, "decisions": [], "sibling_tokens": set()}
    p6b_records = {}                      # P6-B rule decisions (manifest "p6b")
    p7b_reds = {}                         # P7-B segment-engine reduction candidates (name -> leaf metadata)
    p7b_groups, p7b_member_of = {}, {}    # P7-B segment groups (group node -> info), member -> group node
    p7b_records = {}                      # P7-B rule decisions (manifest "p7b")

    p7b_base_memo = {}

    def p7b_base_bytes(refs, n, depth=0):
        """Bytes per element of full-size non-elementwise roots (products, reductions, inputs) reached through
        elementwise chains from `refs` (broadcast operands free). Conservative (large) beyond depth 24."""
        if depth > 24:
            return 1 << 20
        total, seen = 0, set()
        for ref in refs:
            root = layouts[ref].root
            if root in seen:
                continue
            seen.add(root)
            if layouts[ref].numel < n and layouts[root].numel < n:
                continue
            node_ = nodes.get(root)
            if node_ is not None and node_.kind == "elementwise" and node_.loads is not None:
                if root not in p7b_base_memo:
                    p7b_base_memo[root] = p7b_base_bytes(node_.loads, _numel(node_.shape), depth + 1)
                total += p7b_base_memo[root]
            else:
                total += _BYTES.get(dtypes.get(root, "float32"), 4)
        return total

    def p7b_reduction_kind(src, dims, results, extent):
        """'row' (innermost axis, unit stride, >= 64 rows), 'colT' (middle axis of a rank-3 source with a
        unit-stride innermost axis of >= 32), else None. Shape/stride/axis rule only."""
        n = len(src.shape)
        if extent < 2 or results < 2 or max(src.numel, 1 + sum((d - 1) * st for d, st in zip(src.shape, src.strides))) >= (1 << 31):
            return None
        if dims == [n - 1] and src.strides[-1] == 1 and 32 <= extent <= 4096 and results >= 64:
            return "row"
        if n == 3 and dims == [1] and src.strides[2] == 1 and 32 <= src.shape[2] <= 4096 and src.shape[1] >= 2:
            return "colT"
        return None

    def p6b_index32_code(code):
        """P6-B: 32-bit unsigned index arithmetic for a per-thread tree whose program addresses fewer than 2^31
        elements per tensor (every index term is a non-negative product/quotient/remainder below the tensor size).
        Leaves the code unchanged when any negative 64-bit literal appears."""
        if re.search(r"-\s*\d+LL\b", code) or max(l.numel for l in layouts.values()) >= (1 << 31):
            return code
        p6b_records["index32"] = p6b_records.get("index32", 0) + 1   # emitted items (diagnostic count)
        code = re.sub(r"\b(\d+)LL\b", r"\1u", code)
        return code.replace("(long long)", "(unsigned)").replace("long long ", "unsigned ")

    def p6b_tree_group(op, name, source, src, dims, extent, results, refs, code):
        """P6-B: same-geometry coalesced column trees whose operands exist before the group's first reduction share
        one item: each thread evaluates up to four trees (shared operand loads), results are views of one root."""
        if "tree_index" not in p6b_records:
            p6b_records["tree_index"] = {output_name(o): k for k, o in enumerate(operations)}
        index = p6b_records["tree_index"]
        if dtypes[name] != "float32" or layouts[name].numel != results:
            return False
        key = (tuple(src.shape), tuple(src.strides), tuple(dims), extent, results)
        group = p6b_records.get("_tree_open")
        marker = "/*P6B_BODY*/\n"                                 # quad trees: per-iteration header lines
        body = (code.split(marker, 1)[1] if marker in code else code.split("{\n", 1)[1]).rsplit("}", 1)[0]
        shape = layouts[name].shape
        def read_since(group):
            """Does any node created after the group started read the group's results?"""
            root_ = group["root"]
            started = False
            for node_name, node_ in nodes.items():
                if node_name == root_:
                    started = True
                    continue
                if not started:
                    continue
                if node_.kind == "elementwise":
                    refs_ = list(node_.loads or ())
                else:
                    refs_ = [token_loads[t][1] for t in (node_.loads or ())] + list(node_.direct or ())
                if any(r_ in layouts and layouts[r_].root == root_ for r_ in refs_):
                    return True
            return False
        if (group is not None and group["key"] == key and len(group["names"]) < 4
                and (index.get(source, -1) < group["index"] or not read_since(group))):
            if index.get(source, -1) >= group["index"]:
                # Move the merged item to this (latest) member's position: still topological, since nothing
                # created in between reads the group's results.
                del nodes[group["root"]]
                group["moved"] = True
            k = len(group["names"])
            root = group["root"]
            layouts[root] = TensorLayout(((k + 1) * results,), (1,), root)
            layouts[name] = TensorLayout(shape, _contiguous(shape), root, k * results)
            views.append({"name": name, "target": "p6b_tree_result", "source": root})
            group["names"].append(name)
            group["bodies"].append(re.sub(r"\bacc(\b|_w)", lambda m_, k=k: f"acc{k}{m_.group(1)}", body))
            group["refs"] += list(refs)
        else:
            root = f"{name}_p6ct"
            if root in names or root in layouts:
                raise LoweringError(f"P6-B tree group root collision: {root}")
            layouts[root] = TensorLayout((results,), (1,), root)
            dtypes[root] = "float32"
            layouts[name] = TensorLayout(shape, _contiguous(shape), root, 0)
            views.append({"name": name, "target": "p6b_tree_result", "source": root})
            group = {"key": key, "root": root, "names": [name], "index": index.get(name, len(index)), "op": op,
                     "refs": list(refs), "bodies": [re.sub(r"\bacc(\b|_w)", lambda m_: f"acc0{m_.group(1)}", body)]}
            p6b_records["_tree_open"] = group
        loop = code.split(marker, 1)[0] if marker in code else code.split("{\n", 1)[0] + "{\n"
        merged = loop + "".join(group["bodies"]) + "}"
        if p6b_tree_spread and len(group["bodies"]) > 1:
            # P6-B: one thread per (member, result) instead of one per result: the merged trees spread over
            # members x results threads (warp-uniform member branch when results % 32 == 0); same per-tree order.
            k_ = len(group["bodies"])
            branches = " else ".join(f"if (latch_m == {j}) {{\n{b}}}" for j, b in enumerate(group["bodies"]))
            merged = (f"for (long long latch_t = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
                      f"latch_t < {k_ * results}LL; latch_t += (long long)gridDim.x * blockDim.x) {{\n"
                      f"  const long long o = latch_t % {results}LL; const int latch_m = (int)(latch_t / {results}LL);\n"
                      f"  {branches}\n}}")
        previous = nodes.get(group["root"])
        add_heavy("reduction", group["op"], group["root"], merged, group["refs"],
                  engine="exact_cta_tree_per_thread_column", reduction_elements=extent, merged=len(group["names"]),
                  **({"p6b_index32": True} if p6b_index32 else {}))
        if previous is not None:
            nodes[group["root"]].order = previous.order
        p6b_records.setdefault("tree_groups", {})[group["root"]] = list(group["names"])
        return True

    def p6b_selection_admitted(base, target, args, name, shape):
        """P6-B: indexed selections evaluated as per-element expressions (no phase of their own): cat of same-dtype
        operands, FP32 arange, single-index gathers (index.Tensor / embedding) of FP32/BF16 sources. Each result
        element is one operand element (or the arange formula), so values are bit-identical to the copy kernels."""
        if base not in P6B_SELECTIONS or dtypes[name] not in ("float32", "bfloat16"):
            return False
        if base == "cat":
            refs = [_ref(a, names) for a in args[0]]
            return all(r is not None and dtypes[r] == dtypes[name] for r in refs)
        if base == "arange":
            return target in ("arange.default", "arange.start", "arange.start_step") and dtypes[name] == "float32"
        if base == "index":
            indices = list(args[1]) if len(args) > 1 else []
            return (target == "index.Tensor" and len(indices) == 1 and _ref(indices[0], names) is not None
                    and dtypes[_ref(indices[0], names)] == "int64" and _ref(args[0], names) is not None
                    and dtypes[_ref(args[0], names)] == dtypes[name])
        if base == "embedding":
            weight, ids = _ref(args[0], names), _ref(args[1], names)
            return (weight is not None and ids is not None and dtypes[ids] == "int64" and dtypes[weight] == dtypes[name]
                    and len(layouts[weight].shape) == 2
                    and not bool(args[3] if len(args) > 3 else False) and not bool(args[4] if len(args) > 4 else False))
        return False

    def p6b_selection_render(base, target, args, name, shape, idx, rd_ref):
        i = idx if re.fullmatch(r"\w+", idx) else f"({idx})"
        if base == "cat":
            dim = int(args[1] if len(args) > 1 else 0) % len(shape)
            tail, size = _numel(shape[dim + 1:]), shape[dim]
            c = f"(({i} / {_u(tail)}) % {_u(size)})" if tail != 1 else f"({i} % {_u(size)})"
            outer = f"({i} / {_u(size * tail)})"
            inner = f"({i} % {_u(tail)})" if tail != 1 else "0u"
            branches, start = [], 0
            for arg in args[0]:
                ref = _ref(arg, names)
                op_shape = tuple(int(x) for x in layouts[ref].shape)
                if len(op_shape) != len(shape) or any(op_shape[d] != shape[d] for d in range(len(shape)) if d != dim):
                    raise LoweringError("cat operand shapes disagree outside the concatenated dimension")
                extent = op_shape[dim]
                position = f"({outer} * {_u(extent * tail)} + ({c} - {_u(start)}) * {_u(tail)} + {inner})"
                branches.append((start + extent, rd_ref(arg, position, op_shape)))
                start += extent
            if start != size:
                raise LoweringError("cat result extent mismatch")
            expression = branches[-1][1]
            for end, value in reversed(branches[:-1]):
                expression = f"({c} < {_u(end)} ? {value} : {expression})"
            return expression
        if base == "arange":
            numbers = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
            start, step = ((0, 1) if target == "arange.default" else
                           (numbers[0], 1) if target == "arange.start" else (numbers[0], numbers[2]))
            # Same formula as the arange kernel: start + step * i in double, one rounding to FP32.
            return f"static_cast<float>((double){float(start)!r} + (double){float(step)!r} * (double){i})"
        if base in ("index", "embedding"):
            if base == "index":
                source, ids_arg = args[0], list(args[1])[0]
            else:
                source, ids_arg = args[0], args[1]
            src_shape = tuple(int(x) for x in layouts[_ref(source, names)].shape)
            ids_shape = tuple(int(x) for x in layouts[_ref(ids_arg, names)].shape)
            rest = _numel(src_shape[1:])
            if tuple(shape) != (*ids_shape, *src_shape[1:]):
                raise LoweringError(f"{target}: gather result shape mismatch")
            row = f"latch_p6b_row({rd_ref(ids_arg, f'({i} / {_u(rest)})' if rest != 1 else i, ids_shape)}, {src_shape[0]}LL)"
            position = f"({row} * {_u(rest)} + {i} % {_u(rest)})" if rest != 1 else row
            if P6B_ROW_HELPER not in helper_parts:
                helper_parts.append(P6B_ROW_HELPER)
            return rd_ref(source, position, src_shape)
        raise LoweringError(f"unsupported selection {target}")

    def p4a_reduction(op, name, out, source, src, dims, kept, extent, results, refs, rounded, scale):
        """P-OPS4-A hierarchical leading-axis reduction (ops/pops4a_rules.py); False leaves the existing rules."""
        from . import pops4a_rules as R
        from . import norm_rules
        over_exact = (pops4a_over_exact is True
                      or (pops4a_over_exact == "large" and extent * results >= R.OVER_EXACT_MIN_ELEMENTS))
        exact_column = (pops3c and pops3c_column and not over_exact
                        and norm_rules.column_eligible(src.shape, src.strides, dims, results, extent))
        rotate = 37 * p4a_state["count"]
        # P7-B p7b_last_small: small reductions finalize in their own item (last arriver), one phase instead of two
        last_ = pops4a_last or bool(p7b_last_small and extent * max(1, results) <= p7b_last_small)
        part = f"{name}_p4a"
        if part in names or part in layouts:
            raise LoweringError(f"P-OPS4-A partial name collision: {part}")
        if "outline" not in p4a_state:
            p4a_state["bf16"] = any(str(_field(v, "dtype", "float32")) == "bfloat16" for v in values.values())
            # Spec bodies only in programs without BF16 values (no warpgroup MMA callers of outlined code).
            p4a_state["outline"] = bool(pops4a_outline) and not p4a_state["bf16"]
            p4a_state["index"] = {output_name(o): k for k, o in enumerate(operations)}
        if pops4a == "auto" and p4a_state["bf16"]:
            # Automatic scope: FP32 programs. BF16 programs keep their qualified exact (bitwise) engines.
            return False
        if p4a_state["outline"] and not pops4a_last and dtypes[name] == "float32":
            return p4a_outlined(op, name, out, source, src, dims, kept, extent, results, refs, scale, rotate, part,
                                exact_column)
        if not kept and extent >= 32:
            if not R.dense_permutation(src.shape, src.strides, src.offset):
                return False
            position = "s"
            if src.contiguous:
                # Read a permuted dense operand of an elementwise source in its own storage order.
                if p4a_state["producer"] is None:
                    p4a_state["producer"] = {output_name(o): o for o in operations}
                producer = p4a_state["producer"].get(source)
                if producer is not None:
                    for ref in _flatten_refs(list(_field(producer, "args", [])), names):
                        lay = layouts.get(ref)
                        if (lay is not None and tuple(lay.shape) == tuple(src.shape) and not lay.contiguous
                                and R.dense_permutation(lay.shape, lay.strides, lay.offset)):
                            position = R.storage_order_position(lay.shape, lay.strides, "s")
                            break
            leaf = storage_read(source, position)
            plan = R.full_plan(extent, chunk=pops4a_chunk)
            p4a_state["count"] += 1
            if plan["mode"] == "direct":
                code = R.direct(out=out, extent=extent, leaf=leaf, rounded=rounded, scale=scale, rotate=rotate)
                add_heavy("pops4a", op, name, code, refs, pops4a=True, engine="pops4a_direct",
                          reduction_elements=extent, work_items=1)
                p4a_state["decisions"].append({"name": name, "mode": "direct", "extent": extent,
                                               "storage_order": position != "s" and "permuted" or "identity"})
                return True
            if last_:
                code = R.full_last(out=out, extent=extent, leaf=leaf, plan=plan, rotate=rotate, rounded=rounded,
                                   scale=scale, unroll=pops4a_unroll)
                return p4a_single(op, name, code, refs, plan, extent, "full", counters=1, partials=plan["chunks"],
                                  storage="identity" if position == "s" else "permuted")
            code = R.full_partials(part=_ident(part), extent=extent, leaf=leaf, plan=plan, rotate=rotate,
                                   unroll=pops4a_unroll)
            final = R.full_finalize(part=_ident(part), out=out, plan=plan, rounded=rounded, scale=scale,
                                    rotate=rotate)
            words = plan["chunks"]
        elif R.column_eligible(src.shape, src.strides, dims, kept, extent, results, exact_column):
            cols = max(1, min(int(pops4a_cols), results // 32))
            plan = R.column_plan(extent, results, cols=cols, target_items=pops4a_items)
            leaves = [storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pr", f"o{j}"))
                      for j in range(cols)]
            p4a_state["sibling_tokens"].update(int(_TOKEN.search(leaf).group(1)) for leaf in leaves[1:])
            p4a_state["count"] += 1
            if last_:
                code = R.column_last(out=out, extent=extent, results=results, leaves=leaves, plan=plan,
                                     rotate=rotate, rounded=rounded, scale=scale, unroll=pops4a_unroll)
                return p4a_single(op, name, code, refs, plan, extent, "column", counters=plan["blocks"],
                                  partials=plan["tiles"] * results, results=results)
            code = R.column_partials(part=_ident(part), extent=extent, results=results, leaves=leaves, plan=plan,
                                     rotate=rotate, unroll=pops4a_unroll)
            final = R.column_finalize(part=_ident(part), out=out, results=results, plan=plan, rounded=rounded,
                                      scale=scale, rotate=rotate)
            words = plan["tiles"] * results
            position = None
        else:
            return False
        layouts[part] = TensorLayout((words,), (1,), part)
        dtypes[part] = "float32"
        if kept:   # P9-E: what the epilogue column-sum rule needs to re-plan this item (p9e_epi_colsum)
            p4a_state.setdefault("colinfo", {})[part] = {
                "leaves": list(leaves), "E": extent, "N": results, "final": name, "out": out, "plan": dict(plan),
                "rounded": rounded, "scale": scale, "rotate": rotate}
        add_heavy("pops4a", op, part, code, refs, pops4a=True, engine="pops4a_" + ("column" if kept else "full"),
                  reduction_elements=extent, work_items=plan["items"],
                  arithmetic="FP32 fixed-order tile/chunk partials, warp-order CTA combination, tile-order finalize")
        add_heavy("pops4a", op, name, final, [part], pops4a=True, engine="pops4a_finalize",
                  reduction_elements=extent)
        p4a_state["decisions"].append({"name": name, "mode": "column" if kept else "full", "extent": extent,
                                       "results": results, "partials": words,
                                       **({"storage_order": "identity" if position == "s" else "permuted"}
                                          if not kept else {"tiles": plan["tiles"], "rows": plan["rows"]})})
        return True

    def p4a_outlined(op, name, out, source, src, dims, kept, extent, results, refs, scale, rotate, part,
                     exact_column):
        """Outlined P-OPS4-A items (pops4a_rules.OUTLINED_HEADER); the code is generated at emission from the
        recorded spec once inlining is known. Same-shape column reductions whose operands exist before the first
        one's operation share one item (K accumulator sets; each reads the common operand once)."""
        from . import pops4a_rules as R
        from . import norm_rules
        div = f"{_float(extent)}" if scale else "0.0f"
        if not kept and extent > 32:
            if not R.dense_permutation(src.shape, src.strides, src.offset):
                return False
            position = "s"
            if extent <= R.DIRECT_MAX:
                # Direct items replay the frozen generic CTA tree exactly (same logical order and two-stage
                # butterfly): bitwise identical results, only the placement changes.
                position = norm_rules.position(src.shape, src.strides, dims, [], "s", "0u")
            elif src.contiguous:
                producer = p4a_producer().get(source)
                if producer is not None:
                    for ref in _flatten_refs(list(_field(producer, "args", [])), names):
                        lay = layouts.get(ref)
                        if (lay is not None and tuple(lay.shape) == tuple(src.shape) and not lay.contiguous
                                and R.dense_permutation(lay.shape, lay.strides, lay.offset)):
                            position = R.storage_order_position(lay.shape, lay.strides, "s")
                            break
            leaf = storage_read(source, position)
            plan = R.full_plan(extent, chunk=pops4a_chunk)
            p4a_state["count"] += 1
            record = {"name": name, "extent": extent, "outlined": True,
                      "storage_order": "identity" if position == "s" else "permuted"}
            if plan["mode"] == "direct":
                add_heavy("pops4a", op, name, leaf, refs, pops4a=True, engine="pops4a_direct",
                          reduction_elements=extent, work_items=1,
                          p4a_spec={"mode": "direct", "leaves": [leaf], "S": extent, "rot": rotate, "out": out,
                                    "div": div})
                p4a_state["decisions"].append({**record, "mode": "direct", "exact": True,
                                               "storage_order": "logical (frozen tree)"})
                return True
            layouts[part] = TensorLayout((plan["chunks"],), (1,), part)
            dtypes[part] = "float32"
            add_heavy("pops4a", op, part, leaf, refs, pops4a=True, engine="pops4a_full",
                      reduction_elements=extent, work_items=plan["items"],
                      p4a_spec={"mode": "full", "leaves": [leaf], "S": extent, "CH": plan["chunk"],
                                "NC": plan["chunks"], "rot": rotate, "part": _ident(part)})
            add_heavy("pops4a", op, name, f"latch_p4a_fullfin({_ident(part)}, {out}, {plan['chunks']}u, "
                                          f"{rotate % 4096}u, {div});", [part], pops4a=True,
                      engine="pops4a_finalize", reduction_elements=extent, p4a_spec={"mode": "finalize"})
            p4a_state["decisions"].append({**record, "mode": "full", "chunks": plan["chunks"]})
            return True
        if not R.column_eligible(src.shape, src.strides, dims, kept, extent, results, exact_column):
            return False
        cols = max(1, min(int(pops4a_cols), results // 32))
        plan = R.column_plan(extent, results, cols=cols, target_items=pops4a_items)
        vec = int(pops4a_vec or 0) if results % 4 == 0 and results >= 128 else 0
        if vec:
            # Vector tiling decides tiles/rows (partials size); the scalar fallback keeps the same tiles.
            vplan = R.column_plan(extent, results, cols=4 * vec, target_items=pops4a_items, floor=pops4a_floor)
            plan = dict(plan, tiles=vplan["tiles"], rows=vplan["rows"], vblocks=vplan["blocks"],
                        blocks=-(-results // (32 * cols)))
            plan["items"] = plan["tiles"] * plan["blocks"]
        leaf = storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pr", "o"))
        words = plan["tiles"] * results
        key = (tuple(src.shape), tuple(dims), extent, results, cols, vec)
        index = p4a_state["index"]
        if pops4a_last_column:
            return p4a_last_column(op, name, source, refs, leaf, plan, cols, vec, extent, results, div, rotate, key)
        group = p4a_state.get("open_column")
        if (pops4a_merge and group is not None and group["key"] == key and len(group["leaves"]) < 4
                and index.get(source, -1) < group["index"]):
            # Merge into the open item: one more accumulator set and partial block in its storage root.
            k = len(group["leaves"])
            group["leaves"].append(leaf)
            group["parts"].append(_ident(part))
            root = group["root"]
            layouts[root] = TensorLayout(((k + 1) * words,), (1,), root)
            layouts[part] = TensorLayout((words,), (1,), root, k * words)
            dtypes[part] = "float32"
            views.append({"name": part, "target": "pops4a_merged_partial", "source": root})
            old = nodes[root]
            spec = dict(old.extra["p4a_spec"], leaves=list(group["leaves"]), parts=list(group["parts"]))
            add_heavy("pops4a", group["op"], root, " ".join(group["leaves"]), group["refs"] + list(refs),
                      **{**old.extra, "p4a_spec": spec, "merged": k + 1})
            nodes[root].order = old.order
            group["refs"] = group["refs"] + list(refs)
            p4a_state["decisions"].append({"name": name, "mode": "column", "outlined": True, "extent": extent,
                                           "results": results, "merged_into": root})
        else:
            p4a_state["count"] += 1
            layouts[part] = TensorLayout((words,), (1,), part)
            dtypes[part] = "float32"
            spec = {"mode": "column", "leaves": [leaf], "parts": [_ident(part)], "E": extent, "N": results,
                    "T": plan["tiles"], "RT": plan["rows"], "B": plan["blocks"], "C": cols, "rot": rotate,
                    **({"VC": vec, "VB": plan["vblocks"]} if vec else {})}
            add_heavy("pops4a", op, part, leaf, refs, pops4a=True, engine="pops4a_column",
                      reduction_elements=extent, work_items=plan["items"], p4a_spec=spec)
            p4a_state["open_column"] = {"key": key, "leaves": [leaf], "parts": [_ident(part)], "root": part,
                                        "index": index.get(name, len(index)), "op": op, "refs": list(refs)}
            p4a_state["decisions"].append({"name": name, "mode": "column", "outlined": True, "extent": extent,
                                           "results": results, "tiles": plan["tiles"], "rows": plan["rows"],
                                           "cols": cols})
        add_heavy("pops4a", op, name, f"latch_p4a_colfin({_ident(part)}, {out}, {results}u, {plan['tiles']}u, "
                                      f"{rotate % 4096}u, {div});", [part], pops4a=True, engine="pops4a_finalize",
                  reduction_elements=extent, p4a_spec={"mode": "finalize"})
        return True

    def p4a_last_column(op, name, source, refs, leaf, plan, cols, vec, extent, results, div, rotate, key):
        """Single-item column reduction whose last arriving CTA combines the tile partials (results are views of
        one item-owned root; merging as for the two-phase form)."""
        a64 = lambda n: -(-n // 64) * 64
        shape = layouts[name].shape
        index = p4a_state["index"]
        group = p4a_state.get("open_last")
        key = key + (div,)
        if (pops4a_merge and group is not None and group["key"] == key and len(group["leaves"]) < 4
                and index.get(source, -1) < group["index"]):
            k = len(group["leaves"])
            root = group["root"]
            group["leaves"].append(leaf)
            group["outs"].append(_ident(name))
            layouts[root] = TensorLayout(((k + 1) * results,), (1,), root)
            layouts[name] = TensorLayout(shape, _contiguous(shape), root, k * results)
            views.append({"name": name, "target": "pops4a_last_result", "source": root})
            old = nodes[root]
            spec = dict(old.extra["p4a_spec"], leaves=list(group["leaves"]), outs=list(group["outs"]))
            split = dict(old.extra["p2_split"], partial_words=a64((k + 1) * plan["tiles"] * results))
            group["refs"] = group["refs"] + list(refs)
            add_heavy("pops4a", group["op"], root, " ".join(group["leaves"]), group["refs"],
                      **{**old.extra, "p4a_spec": spec, "p2_split": split, "merged": k + 1})
            nodes[root].order = old.order
            p4a_state["decisions"].append({"name": name, "mode": "column_last", "extent": extent,
                                           "results": results, "merged_into": root})
            return True
        p4a_state["count"] += 1
        root = f"{name}_p4l"
        if root in names or root in layouts:
            raise LoweringError(f"P-OPS4-A result root collision: {root}")
        layouts[root] = TensorLayout((results,), (1,), root)
        dtypes[root] = "float32"
        layouts[name] = TensorLayout(shape, _contiguous(shape), root, 0)
        views.append({"name": name, "target": "pops4a_last_result", "source": root})
        blocks = max(plan["blocks"], plan.get("vblocks", 0))
        spec = {"mode": "column_last", "leaves": [leaf], "outs": [_ident(name)], "E": extent, "N": results,
                "T": plan["tiles"], "RT": plan["rows"], "B": plan["blocks"], "C": cols, "rot": rotate, "div": div,
                **({"VC": vec, "VB": plan["vblocks"]} if vec else {})}
        add_heavy("pops4a", op, root, leaf, refs, pops4a=True, engine="pops4a_column_last",
                  reduction_elements=extent, work_items=plan["items"], p4a_spec=spec,
                  p2_split={"counter_words": a64(blocks), "partial_words": a64(plan["tiles"] * results)})
        p4a_state["open_last"] = {"key": key, "leaves": [leaf], "outs": [_ident(name)], "root": root,
                                  "index": index.get(name, len(index)), "op": op, "refs": list(refs)}
        p4a_state["decisions"].append({"name": name, "mode": "column_last", "extent": extent, "results": results,
                                       "tiles": plan["tiles"], "rows": plan["rows"], "vector": bool(vec)})
        return True

    def p4a_producer():
        if p4a_state["producer"] is None:
            p4a_state["producer"] = {output_name(o): o for o in operations}
        return p4a_state["producer"]

    def p4a_single(op, name, code, refs, plan, extent, mode, *, counters, partials, **record):
        """One P-OPS4-A item whose last arriving CTA combines the partials (split-K scratch and counters)."""
        a64 = lambda n: -(-n // 64) * 64
        add_heavy("pops4a", op, name, code, refs, pops4a=True, engine=f"pops4a_{mode}_last",
                  reduction_elements=extent, work_items=plan["items"],
                  p2_split={"counter_words": a64(counters), "partial_words": a64(partials)},
                  arithmetic="FP32 fixed-order tile/chunk partials, warp-order CTA combination, "
                             "last-arriver tile-order combination")
        p4a_state["decisions"].append({"name": name, "mode": mode + "_last", "extent": extent,
                                       "partials": partials, **record,
                                       **({"tiles": plan["tiles"], "rows": plan["rows"], "cols": plan["cols"]}
                                          if mode == "column" else {"chunks": plan["chunks"]})})
        return True

    def add_heavy(kind, op, name, code, refs, **extra):
        loads = [int(m) for m in _TOKEN.findall(code)]
        direct = [ref for ref in dict.fromkeys(refs) if re.search(r"\b" + re.escape(_ident(ref)) + r"\b", code)]
        nodes[name] = _Node(name=name, kind=kind, code=code, extra=extra, loads=loads, direct=direct,
                            target=str(_field(op, "target", _field(op, "op"))), order=len(nodes))

    attention_decisions, index_decisions = [], []
    indexed_occurrence_cache = {}
    attention_groups, attention_refusals = attention_lower.plan(
        operations, values, names, ref=_ref, field=_field, output_name=output_name,
        protected=set(outputs) | set(writes.values()), exclude=attention_lower.excluded_groups()
    ) if fuse_attention else ([], [])
    pops4c_attention = []
    if attention_min_keys and attention_groups:
        # P-OPS2-A: for short key ranges the materialized scores are small and the selected FP32 product
        # bodies plus softmax phases are faster than the fused attention phases (measured on H100).
        # P-OPS4-C: that calibration assumed every materialized product reaches a selected FP32 body. When
        # any would be refused (e.g. odd key/query extents: unaligned leading dimensions and tile tails send
        # it to the generic 16x16 SIMT body), the existing fused attention body is kept instead.
        kept = []
        for group in attention_groups:
            keys = int(_field(values[group["kt"]], "shape")[-1]) if group.get("kt") in values else None
            if keys is not None and keys <= attention_min_keys:
                refused = (_materialized_attention_refusals(group, values, target_arch, matrix_bodies)
                           if pops4c and not attention_override and matrix_engine == "auto" else [])
                if refused:
                    pops4c_attention.append({"group": group.get("label"), "keys": keys,
                                             "decision": "fused attention kept", "refused_products": refused})
                    kept.append(group)
                    continue
                attention_refusals.append({"group": group.get("label"), "reason":
                    f"{keys} keys <= {attention_min_keys}: product bodies + softmax selected over fused attention"})
            else:
                kept.append(group)
        attention_groups = kept
    for group in attention_groups:
        group["v2_enabled"] = attention_schedule if attention_schedule == "force" else bool(attention_schedule)
        group["store_scores"] = bool(attention_store_scores)
        group["debug_repeat"] = bool(attention_debug_repeat)
    attention_lower.share_mask_tables(attention_groups, None, values, _field)
    attention_members = {name: group for group in attention_groups for name in group["skip"]}
    mixed_plans = {}
    if pops5d and target_arch == "sm_90":
        from . import mixed_attention
        static = {}
        for value_name, value in values.items():
            value_shape = tuple(int(x) for x in _field(value, "shape"))
            static[value_name] = (value_shape, tuple(int(x) for x in _field(
                value, "strides", _field(value, "stride", _contiguous(value_shape)))))
        mixed_plans, mixed_refused = mixed_attention.admit(
            (operations, values, names, _ref, _field, output_name), static.__getitem__,
            set(outputs) | set(writes.values()), exclude=attention_lower.excluded_groups(), absorb=pops5d_absorb)
        attention_refusals.extend(mixed_refused)
        for label, plan in list(mixed_plans.items()):
            if plan["removed"] & set(attention_members):
                del mixed_plans[label]
                attention_refusals.append({"group": label, "reason": "mixed attention overlaps a fused group"})
    mixed_members = {name: plan for plan in mixed_plans.values() for name in plan["removed"]}

    def mixed_install(plan, name):
        """P-OPS5-D: install the forward (output materialization) or backward (first gradient) phase."""
        from . import mixed_attention
        group = plan["group"]
        if name not in (plan["anchor_forward"], plan["anchor_backward"]):
            return
        which = "forward" if name == plan["anchor_forward"] else "backward"

        def lowered(value_name):
            if value_name in layouts:
                return (layouts[value_name].shape, layouts[value_name].strides)
            return static[value_name]
        try:
            fresh = mixed_attention.refresh(plan, (operations, values, names, _ref, _field, output_name), lowered,
                                            set(outputs) | set(writes.values()))
        except mixed_attention.Refused as exc:
            raise attention_lower.AttentionRefused(group["label"], f"mixed attention at lowering: {exc}")
        used = ("q", "k", "v") if which == "forward" else ("q", "k", "v", "do")
        if any(fresh["strides"][f"in:{k}"] != plan["strides"][f"in:{k}"] for k in used):
            raise attention_lower.AttentionRefused(group["label"], "lowered operand layouts differ from admission")
        for key in used:
            if fresh["sources"][key] not in layouts or (key in fresh["biases"] and fresh["biases"][key] not in layouts):
                raise attention_lower.AttentionRefused(group["label"], f"{key}: operand not defined at installation")
        kind, code, extra = mixed_attention.emit(fresh, _ident, dtypes.__getitem__, which=which,
                                                 bias_phase=pops5d_bias_phase,
                                                 trace=bool(pops5d_trace and (pops5d_defines or {}).get("PMIX_TRACE")),
                                                 store_probs=bool(pops5d_store_probs))
        for value_name, spec in extra["outputs"].items():
            if spec is None:
                value_shape = tuple(int(x) for x in _field(values[value_name], "shape"))
                value_dtype = _dtype(values[value_name])
            else:
                value_shape, value_dtype = tuple(spec[0]), spec[1]
                if value_name in names or value_name in layouts:
                    raise attention_lower.AttentionRefused(group["label"], f"synthetic name collision {value_name}")
            layouts[value_name] = TensorLayout(value_shape, _contiguous(value_shape), value_name)
            dtypes[value_name] = value_dtype
        defines_ = dict(pops5d_defines or {})
        if not mixed_attention.LONG_V2:
            defines_.setdefault("PMIX_LONG_V2", 0)   # P9-E: the header follows the planner's shared-memory choice
        if not mixed_attention.LONG_V3:
            defines_.setdefault("PMIX_LONG_V3", 0)   # P9-E: no halves scratch is planned without v3
        if mixed_attention.LONG_FWD_V2:
            defines_.setdefault("PMIX_LONG_FWD_V2", 1)   # P9-E: planner reserved the v2 forward's shared memory
        if mixed_attention.FA_DELTA:
            defines_.setdefault("PMIX_FA_DELTA", 1)      # P9-E amendment A4: the backward reads the forward output
        include = "".join(f"#define {key} {value}\n" for key, value in sorted(defines_.items()))
        include += f'#include "{mixed_attention.HEADER}"\n'
        if include not in helper_parts:
            helper_parts.append(include)
        if mixed_attention.HEADER not in extra_headers:
            extra_headers.append(mixed_attention.HEADER)
        reads = [fresh["sources"][key] for key in used] + [fresh["biases"][k] for k in used if k in fresh["biases"]]
        if which == "backward":
            reads.append(mixed_attention.stats_name(group))
        reads += extra["aux_reads"]
        bias = extra.pop("bias_phase", None)
        metadata = {key: value for key, value in extra.items() if key not in ("outputs", "aux_reads")}
        metadata.update(group=group["label"], attention_output=group["o"], source_commit="P-OPS5-D",
                        header=mixed_attention.HEADER, threads=256)
        attention_lower.add_fused_nodes(kind, code, metadata, list(dict.fromkeys(reads)), list(extra["outputs"]),
            layouts=layouts, dtypes=dtypes, make_layout=TensorLayout, add_heavy=add_heavy, add_view=attention_view,
            reserved_names=names)
        if bias is not None:
            # P-OPS5-D: the bias-gradient sums of the absorbed reductions run as their own small node after the
            # backward phase (fixed tree order over the per-item partials; no atomics).
            bias_code, bias_extra = bias
            for value_name in bias_extra["outputs"]:
                value_shape = tuple(int(x) for x in _field(values[value_name], "shape"))
                layouts[value_name] = TensorLayout(value_shape, _contiguous(value_shape), value_name)
                dtypes[value_name] = _dtype(values[value_name])
            bias_meta = {key: value for key, value in bias_extra.items() if key not in ("outputs", "aux_reads")}
            bias_meta.update(group=group["label"], attention_output=group["o"], source_commit="P-OPS5-D",
                             header=mixed_attention.HEADER, threads=256)
            attention_lower.add_fused_nodes("attention_mixed_bias", bias_code, bias_meta, list(bias_extra["aux_reads"]),
                list(bias_extra["outputs"]), layouts=layouts, dtypes=dtypes, make_layout=TensorLayout,
                add_heavy=add_heavy, add_view=attention_view, reserved_names=names)
        if which == "forward":
            attention_decisions.append({"group": group["label"], "decision": "mixed_bf16_short_attention", "engine": "long" if mixed_attention.is_long(fresh["t"]) else "short",
                "head_width": fresh["hd"], "sequence": fresh["t"], "head_batch": fresh["bh"], "heads": fresh["heads"],
                "scale": group["scale"], "scale_mode": group["scale_mode"], "sources": fresh["sources"],
                "biases": fresh["biases"], "outputs": fresh["finals"], "kinds": {k: d["kind"] for k, d in fresh["descs"].items()},
                "strides": {k: list(v) for k, v in fresh["strides"].items()},
                "absorbed_reductions": {k: list(v) for k, v in fresh.get("reduce", {}).items()},
                "bias_strides": {k: list(v) for k, v in fresh["bias_strides"].items()},
                "removed": sorted(fresh["removed"]), "source_commit": "P-OPS5-D"})

    def attention_define(name, shape, root=None, strides=None):
        layouts[name] = TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), root or name)
        dtypes[name] = "float32"

    def attention_view(name, target, source):
        views.append({"name": name, "target": target, "source": source})

    def attention_phase(kind, code, metadata, reads, phase_writes):
        include = '#include "attention_fp32.cuh"\n'
        if include not in helper_parts:
            helper_parts.append(include)
        if attention_lower.HEADER not in extra_headers:
            extra_headers.append(attention_lower.HEADER)
        if metadata.get("header") == attention_lower.HEADER_V2:
            include_v2 = ('' if attention_pipeline else '#define PA2_PIPELINE 0\n') + '#include "attention_fp32_v2.cuh"\n'
            if include_v2 not in helper_parts:
                helper_parts.append(include_v2)
            if attention_lower.HEADER_V2 not in extra_headers:
                extra_headers.append(attention_lower.HEADER_V2)
        attention_lower.add_fused_nodes(kind, code, metadata, reads, phase_writes,
            layouts=layouts, dtypes=dtypes, make_layout=TensorLayout, add_heavy=add_heavy,
            add_view=attention_view, reserved_names=names)

    scan_decisions = []
    for op in operations:
        target = str(_field(op, "target", _field(op, "op")))
        target = target.removeprefix("torch.ops.").removeprefix("aten.")
        base = target.split(".")[0]
        name = output_name(op)
        args = list(_field(op, "args", []))
        kwargs = dict(_field(op, "kwargs", {}))
        refs = _flatten_refs((args, kwargs), names)
        shape = tuple(int(x) for x in _field(values[name], "shape"))
        if any(d <= 0 for d in shape):
            raise LoweringError(f"{name}: empty tensors are not admitted")
        if name in undefined:
            continue
        if name in mixed_members:
            mixed_install(mixed_members[name], name)
            continue
        if name in attention_members:
            attention_lower.emit_member(attention_members[name], name, layouts, dtypes, values,
                field=_field, ident=_ident, define=attention_define, add_phase=attention_phase,
                add_view=attention_view, record=attention_decisions.append)
            continue
        if base == "fill":
            if len(args) != 2 or _ref(args[0], names) is None:
                raise LoweringError("functional fill requires a tensor template and one fill value")
            template = values[_ref(args[0], names)]
            if tuple(_field(template, "shape")) != shape or _dtype(template) != dtypes[name]:
                raise LoweringError("functional fill must preserve template shape/dtype")
            refs = _flatten_refs((args[1:], kwargs), names)
        if base in {"slice", "select"}:
            from .extra_ops import view as extra_view
            source = _ref(args[0], names)
            src = layouts[source]
            strides, offset = extra_view(base, args, src, shape)
            layouts[name] = TensorLayout(shape, strides, src.root, offset)
            views.append({"name": name, "target": target, "source": source})
            continue
        if base in {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand", "squeeze", "unsqueeze", "slice", "select", "as_strided"}:
            source = _ref(args[0], names)
            src = layouts[source]
            strides = list(src.strides)
            offset = src.offset
            if base == "as_strided":
                # Functionalized split/chunk views. as_strided addresses the
                # source's storage absolutely; admit it only on an offset-0,
                # contiguous source whose storage is that root's own allocation.
                root_value = values.get(src.root)
                if (src.offset or not src.contiguous or src.numel != layouts[src.root].numel
                        or (root_value is not None and int(_field(root_value, "storage_offset", 0) or 0))):
                    raise LoweringError(f"{name}: as_strided requires a dense offset-0 source allocation")
                strides = [int(v) for v in args[2]]
                offset = int(args[3]) if len(args) > 3 and args[3] is not None else 0
                extent = offset + sum((d - 1) * s for d, s in zip(shape, strides))
                if any(s < 0 for s in strides) or extent >= src.numel:
                    raise LoweringError(f"{name}: as_strided view exceeds its source allocation")
            elif base in {"slice", "select"}:
                # C-OPS2-B: strided subviews. Offsets are in elements of the root.
                dim = int(args[1] if len(args) > 1 else 0) % len(src.shape)
                if base == "slice":
                    extent = src.shape[dim]
                    start = args[2] if len(args) > 2 and args[2] is not None else 0
                    step = int(args[4]) if len(args) > 4 else 1
                    start = max(0, min(extent, int(start) + extent if int(start) < 0 else int(start)))
                    if step <= 0:
                        raise LoweringError("slice requires a positive step")
                    offset += start * src.strides[dim]
                    strides[dim] = src.strides[dim] * step
                else:
                    index = int(args[2]) % src.shape[dim]
                    offset += index * src.strides[dim]
                    del strides[dim]
            elif base == "t":
                if len(src.shape) > 2:
                    raise LoweringError("t only supports rank <= 2")
                strides = list(reversed(strides))
            elif base == "transpose":
                a, b = int(args[1]) % len(strides), int(args[2]) % len(strides)
                strides[a], strides[b] = strides[b], strides[a]
            elif base == "permute":
                strides = [src.strides[int(d)] for d in args[1]]
            elif base == "expand":
                pad_shape = (1,) * (len(shape) - len(src.shape)) + src.shape
                pad_stride = (0,) * (len(shape) - len(src.shape)) + src.strides
                strides = []
                for old, new, stride in zip(pad_shape, shape, pad_stride):
                    if old not in (1, new):
                        raise LoweringError("invalid expansion")
                    strides.append(0 if old == 1 and new != 1 else stride)
            elif base == "unsqueeze":
                dim = int(args[1]) % (len(src.shape) + 1)
                strides.insert(dim, src.strides[dim] * src.shape[dim] if dim < len(src.shape) else 1)
            elif base == "squeeze":
                if len(args) == 1:
                    strides = [s for d, s in zip(src.shape, src.strides) if d != 1]
                else:
                    dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
                    dims = [int(d) % len(src.shape) for d in dims]
                    strides = [s for k, (d, s) in enumerate(zip(src.shape, src.strides)) if k not in dims or d != 1]
            elif base in {"view", "_unsafe_view", "reshape"}:
                if src.numel != _numel(shape):
                    raise LoweringError("reshape changes numel")
                strides = (_contiguous(shape) if src.contiguous else
                           _reshape_strides(src.shape, src.strides, shape))
                if strides is None:
                    raise LoweringError(f"{name}: reshape crosses noncontiguous stride subspaces; "
                                        f"copy decomposition required: {src.shape}/{src.strides} -> {shape}")
            if len(strides) != len(shape):
                raise LoweringError(f"{name}: invalid view rank")
            layouts[name] = TensorLayout(shape, tuple(strides), src.root, offset)
            views.append({"name": name, "target": target, "source": source})
            continue
        layouts[name] = TensorLayout(shape, _contiguous(shape), name)
        out = _ident(name)
        if target in {"latch_ops.philox_mask.default", "latch_ops.philox_advance.default"}:
            from .random_cuda import CUDA_HELPERS
            if CUDA_HELPERS not in helper_parts:
                helper_parts.append(CUDA_HELPERS)
            state = _ref(args[0], names)
            if dtypes[state] != "int64" or layouts[state].shape != (4,) or not layouts[state].contiguous:
                raise LoweringError("Philox state requires contiguous int64[4]")
            if target.endswith("philox_mask.default"):
                if dtypes[name] != "float32" or list(shape) != list(args[1]):
                    raise LoweringError("Philox mask requires FP32 declared shape")
                code = f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {_numel(shape)}LL; i += (long long)gridDim.x * blockDim.x) {out}[i] = latch_philox_mask({_ident(state)}, i, {int(args[2])}ULL, {_float(args[3])});"
            else:
                if dtypes[name] != "int64" or shape != (4,):
                    raise LoweringError("Philox advance requires int64[4] output")
                code = f"if (blockIdx.x == 0 && threadIdx.x < 4) {out}[threadIdx.x] = latch_philox_advance_word({_ident(state)}, {int(args[1])}ULL, threadIdx.x);"
            add_heavy("random_state" if "advance" in target else "random_mask", op, name, code, refs)
            continue
        if target == "latch_ops.mm_ksum":
            # P9-E: K-segmented BF16 product (see the p9e_ksum rewrite): every segment must select the same TMA
            # warpgroup spec (shape, strides, operand majors); the segments' maps are passed to one product.
            from .matrix_codegen_p7b import select
            from .matrix_rules_p7b import HEADER, emit_group, spec_headers
            pairs_ = [(_ref(args[2 * j_], names), _ref(args[2 * j_ + 1], names)) for j_ in range(len(args) // 2)]
            out_ = _ident(name)
            specs_ = []
            for a_, b_ in pairs_:
                sp_, why_ = select(a_, b_, out_, layouts[a_], layouts[b_], shape, dtypes[name], arch=target_arch)
                if sp_ is None or not sp_.get("tma"):
                    raise LoweringError(f"P9-E K-segmented product {name}: segment {a_} x {b_} not a TMA product ({why_})")
                specs_.append(sp_)
            same_ = lambda x, y: {k_: v_ for k_, v_ in x.items() if k_ not in ("a", "b", "tma")} == \
                                 {k_: v_ for k_, v_ in y.items() if k_ not in ("a", "b", "tma")}
            if not all(same_(specs_[0], y_) for y_ in specs_[1:]):
                raise LoweringError(f"P9-E K-segmented product {name}: segments disagree in shape or layout")
            spec = dict(specs_[0], k=specs_[0]["k"] * len(specs_), kseg=specs_[0]["k"],
                        ksegs=[{"a": y_["a"], "b": y_["b"], "tma": y_["tma"]} for y_ in specs_[1:]])
            for header in ("tgemm.cuh", "bgemm.cuh", HEADER) + tuple(spec_headers(spec)):
                if header not in extra_headers:
                    extra_headers.append(header)
                if header not in ("tgemm.cuh", "bgemm.cuh") and f'#include "{header}"' not in helper_parts:
                    helper_parts.append(f'#include "{header}"')
            if "#define POPS4B_KSEG 1" not in helper_parts:
                helper_parts.insert(0, "#define POPS4B_KSEG 1")
            code, extra = emit_group([spec], ident=_ident)
            add_heavy("gemm", op, name, code, refs, k=spec["k"], matrix_spec=spec, **extra)
            matrix_decisions.append({"output": name, "selected": True, "reason": f"P9-E K-segmented product ({len(specs_)} segments)"})
            continue
        from .transformer_lower import emit_operation, ordered_embedding_backward
        if ordered_index_gradients and base == "embedding_dense_backward":
            # LATCH-STEP-04: an explicitly selected reusable indexed engine (indexed_compose).
            from .indexed_compose import legacy_engine
            engine = legacy_engine(args, kwargs, name, shape, layouts, dtypes, names, cache=indexed_occurrence_cache)
            if engine is not None:
                if engine["helpers"] and engine["helpers"] not in helper_parts:
                    helper_parts.append(engine["helpers"])
                for value, value_shape, dtype in engine["auxiliary"]:
                    layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
                    dtypes[value] = dtype
                for phase in engine["phases"]:
                    extra = {key: value for key, value in phase.items() if key not in {"kind", "writes", "reads", "code"}}
                    add_heavy(phase["kind"], op, phase["writes"][0], phase["code"], phase["reads"], **extra)
                index_decisions.append({"output": name, **{key: value for key, value in engine.items()
                                        if key not in {"auxiliary", "phases", "helpers"}}})
                continue
            indexed = ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names)
            if indexed is not None:
                for value, value_shape, dtype in indexed["auxiliary"]:
                    layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
                    dtypes[value] = dtype
                for kind, phase_writes, code, extra in indexed["phases"]:
                    if len(phase_writes) > 1:
                        attention_lower.add_fused_nodes(kind, code,
                            dict(extra, group=name, atomic_output=name, source_commit=indexed["source_commit"]),
                            [_ref(args[1], names)], phase_writes, layouts=layouts, dtypes=dtypes,
                            make_layout=TensorLayout, add_heavy=add_heavy, add_view=attention_view,
                            reserved_names=names)
                    else:
                        add_heavy(kind, op, phase_writes[0], code,
                                  [_ref(args[0], names), *indexed["internal"]], **extra)
                index_decisions.append({"output": name, "decision": "ordered_occurrence_list",
                                        "source_commit": indexed["source_commit"],
                                        "accumulation_order": indexed["accumulation_order"]})
                continue
        indexed_elementwise = (p6b_index_elementwise and not retain_all
                               and p6b_selection_admitted(base, target, args, name, shape))
        emitted = None if indexed_elementwise else emit_operation(
            target, args, kwargs, name, shape, layouts, dtypes, names, placeholder_read,
            warp_rows="pops2a" if warp_rows else False, inputs=tuple(inputs))
        if emitted is not None:
            kind, code, extra, helper = emitted
            if helper and helper not in helper_parts:
                helper_parts.append(helper)
            add_heavy(kind, op, name, code, refs, **extra)
            continue
        if base in {"mm", "matmul", "bmm", "addmm"}:
            aa, bb = (args[0], args[1]) if base != "addmm" else (args[1], args[2])
            a, b = _ref(aa, names), _ref(bb, names)
            la, lb = layouts[a], layouts[b]
            # P-OPS4-C: operand layouts of every product, for refusal accounting (manifest only).
            matrix_operands.append({'output': name, 'dtype': dtypes[name], 'shape': list(shape),
                                    **{key: {'value': value, 'shape': list(layout.shape), 'strides': list(layout.strides),
                                             'offset': layout.offset, 'root': layout.root, 'dtype': dtypes[value],
                                             'root_numel': layouts[layout.root].numel if layout.root in layouts else None}
                                       for key, value, layout in (('a', a, la), ('b', b, lb))}})
            if matrix_policy != "frozen" and base != "addmm" and dtypes[a] == dtypes[b] == dtypes[name]:
                from .matrix_codegen_p7b import select
                from .matrix_rules_p7b import HEADER, emit_group
                spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch)
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                    from .matrix_rules_p7b import spec_headers
                    for header in spec_headers(spec):
                        if header not in extra_headers:
                            extra_headers.append(header)
                        if f'#include "{header}"' not in helper_parts:
                            helper_parts.append(f'#include "{header}"')
                    from .matrix_rules_p7b import split_choice
                    split = None
                    if (p6b_wg_split_edge and p6b_wg_split and p6b_splitk and not retain_all and spec.get("edge")
                            and not spec.get("wg") and spec["batch"] == 1 and spec["k"] >= 2048 and spec["n"] % 2 == 0):
                        # P6-B: long-K masked-edge products (small M or N) as split-K TMA warpgroup tasks: TMA zero-fills
                        # the absent rows, slice stores are masked.
                        from .matrix_rules_p7b import tma_allowed, wg_split_choice
                        if tma_allowed():
                            edge_spec = dict(spec, wg=True, pair=True, tma={"a": (spec["a"], 0), "b": (spec["b"], 0)})
                            split = wg_split_choice(edge_spec, grid=p6b_split_grid, edge=True,
                                                    objective=p6b_split_objective)
                            if split:
                                spec = edge_spec
                                for header in spec_headers(spec):
                                    if header not in extra_headers:
                                        extra_headers.append(header)
                                    if f'#include "{header}"' not in helper_parts:
                                        helper_parts.append(f'#include "{header}"')
                    if p6b_wg_split and p6b_splitk and not retain_all and spec.get("tma") and not split:
                        # P6-B: split-K on the TMA warpgroup body (FP32 slice results), before the mma.sync slices.
                        from .matrix_rules_p7b import tma_allowed, wg_split_choice
                        if tma_allowed():
                            split = wg_split_choice(spec, grid=p6b_split_grid, objective=p6b_split_objective)
                    if p6b_splitk and not retain_all and not split:
                        # Long-K small-output products: split-K slices on the mma.sync body take precedence over the
                        # warpgroup body (whose large tiles leave most of the grid idle there).
                        plain_spec = ({k_: v_ for k_, v_ in spec.items() if k_ not in ("wg", "tma", "pair")}
                                      if spec.get("wg") and not spec.get("edge") else spec)
                        split = split_choice(plain_spec)
                        if split:
                            spec = plain_spec
                    if split:
                        # P6-B deterministic split-K: the product writes FP32 K-slice results [split][m][n]; the BF16
                        # product value is a per-element fixed-order slice sum rounded once (an ordinary elementwise
                        # value, evaluated where it is consumed).
                        part = f"{name}_p6sk"
                        if part in names or part in layouts:
                            raise LoweringError(f"split-K partial name collision: {part}")
                        mn = spec["m"] * spec["n"]
                        layouts[part] = TensorLayout((split * mn,), (1,), part)
                        dtypes[part] = "float32"
                        names.add(part)
                        split_spec = dict(spec, out=_ident(part), split=split)
                        code, extra = emit_group([split_spec], ident=_ident)
                        add_heavy('gemm', op, part, code, refs, k=la.shape[-1] // split, matrix_spec=split_spec, **extra)
                        matrix_decisions[-1].update(split_k=split, partial=part)
                        p6b_records.setdefault("split_k", []).append({"product": name, "split": split,
                                                                      "shape": [spec["m"], spec["n"], spec["k"]]})

                        def split_render(idx, rd_ref, part=part, split=split, mn=mn):
                            i = idx if re.fullmatch(r"\w+", idx) else f"({idx})"
                            terms = [rd_ref({"ref": part}, i if s_ == 0 else f"({_u(s_ * mn)} + {i})", (split * mn,))
                                     for s_ in range(split)]
                            total = terms[0]
                            for term in terms[1:]:
                                total = f"({total} + {term})"
                            return total
                        nodes[name] = _Node(name=name, kind="elementwise", shape=shape, dtype=dtypes[name],
                                            render=split_render, base="p6b_split_sum", target="p6b.split_k_sum",
                                            own_cost=1, loads=[part] * split, order=len(nodes))
                        continue
                    code, extra = emit_group([spec], ident=_ident)
                    add_heavy('gemm', op, name, code, refs, k=la.shape[-1], matrix_spec=spec, **extra)
                    continue
            if all(dtypes[v] == "bfloat16" for v in (a, b, name)):
                if base == "addmm":
                    raise LoweringError("BF16 addmm must be decomposed into mm and pointwise add")
                from .mixed_gemm import CUDA_HELPERS, HEADERS, emit
                if CUDA_HELPERS not in helper_parts:
                    helper_parts.append(CUDA_HELPERS)
                extra_headers.extend(h for h in HEADERS if h not in extra_headers)
                code, extra = emit(a, b, out, la, lb, shape, split_grid=P8E_SPLIT_GRID if p8e_generic_split else 0)
                add_heavy("gemm", op, name, code, refs, k=la.shape[-1], **extra)
                continue
            if any(dtypes[v] != "float32" for v in (a, b, name)):
                raise LoweringError("matrix products require matching FP32 or explicit BF16 operands/results")
            if len(la.shape) not in (2, 3) or len(lb.shape) != len(la.shape):
                raise LoweringError(f"{name}: mm/bmm requires equal rank-2 or rank-3 operands")
            m, k = la.shape[-2:]
            kb, n = lb.shape[-2:]
            batched = len(la.shape) == 3
            batch = la.shape[0] if batched else 1
            if k != kb or (batched and lb.shape[0] != batch) or shape != ((batch, m, n) if batched else (m, n)):
                raise LoweringError(f"{name}: matrix shape mismatch {la.shape} @ {lb.shape} -> {shape}")
            if matrix_policy != 'frozen' and base != 'addmm' and matrix_engine == 'auto':
                # P-OPS2-A: body/tile/pipeline/split selection; split scratch bound after scheduling.
                from .matrix_select import plan as select_plan
                plan, reason = select_plan(a, b, name, la, lb, shape, ident=_ident, arch=target_arch, split=split_k,
                                           bodies=matrix_bodies, runtime_layout=matrix_runtime_layout,
                                           residency=min(matrix_residency, 2))
                matrix_decisions.append({'output': name, 'selected': plan is not None,
                                         'candidate': 'pops2_fp32', 'reason': reason,
                                         **({'plan': plan['metadata']['matrix_plan']} if plan else {})})
                if plan is not None:
                    if plan['helpers'] not in helper_parts:
                        helper_parts.append(plan['helpers'])
                    extra_headers.extend(h for h in plan['headers'] if h not in extra_headers)
                    add_heavy('gemm', op, name, plan['code'], plan['operand_roots'], **plan['metadata'])
                    continue
            if matrix_policy != 'frozen' and base != 'addmm' and (matrix_engine == 'popsb' or isinstance(matrix_engine, dict)):
                from .matrix_codegen_p7b import fp32_async
                plan, reason = fp32_async(a, b, name, la, lb, shape, ident=_ident,
                    arch=target_arch, split=split_k, occupied_names=names | set(layouts), engine=matrix_engine if isinstance(matrix_engine, dict) else None)
                matrix_decisions.append({'output': name, 'selected': plan is not None,
                                         'candidate': 'popsb_cpasync', 'reason': reason})
                if plan is not None:
                    helper_parts.append(plan['helpers'])
                    extra_headers.extend(h for h in plan['headers'] if h not in extra_headers)
                    scratch_refs = []
                    for scratch in plan['scratch']:
                        key = scratch['name']
                        layouts[key] = TensorLayout(tuple(scratch['shape']), (1,), key)
                        dtypes[key] = scratch['dtype']
                        scratch_refs.append(key)
                    if plan['counter_init'] is not None:
                        initialization = plan['counter_init']
                        add_heavy('matrix_counter_init', op, initialization['writes'][0],
                                  initialization['code'], [],
                                  work_items=(initialization['counter_words'] + 255) // 256,
                                  scratch_initialization=plan['scratch'])
                        # Conservative ordering edges keep private partials near
                        # the product, instead of allocating every product's
                        # scratch from phase zero. No operand values are read.
                        nodes[initialization['writes'][0]].direct = list(plan['operand_roots'])
                    add_heavy('gemm', op, name, plan['code'],
                              [*plan['operand_roots'], *scratch_refs], **plan['metadata'])
                    continue
            sam, sak = la.strides[-2:]
            sbk, sbn = lb.strides[-2:]
            bsa, bsb = (la.strides[0], lb.strides[0]) if batched else (0, 0)
            vector_safe = ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)
                           and all(s % 4 == 0 for s in (sam if sak == 1 else sak,
                                                       sbn if sbk == 1 else sbk, bsa, bsb,
                                                       la.offset, lb.offset)))
            tiled = not (m % 64 or n % 64 or k % 16) and vector_safe
            if tiled and split_k and target_arch == 'sm_90':
                from .matrix_codegen_p7b import split_k as split_codegen
                split = split_codegen(a, b, name, la, lb, ident=_ident)
                if split is not None and (split['part'] in layouts or split['part'] in names):
                    matrix_decisions.append({'output': name, 'selected': False,
                        'reason': 'split-K scratch name collision; existing matrix body retained'})
                    split = None
                if split is not None:
                    part = split['part']
                    layouts[part] = TensorLayout((split['elements'],), (1,), part)
                    dtypes[part] = 'float32'
                    add_heavy('gemm', op, part, split['code'], refs, **split['metadata'])
                    add_heavy('splitk_reduce', op, name, split['reduce'], [part], reduction_elements=split['metadata']['ksplit'])
                    matrix_decisions.append({'output': name, 'selected': True, 'reason': 'underfilled FP32 ordered split-K', 'ksplit': split['metadata']['ksplit']})
                    continue
            if tiled:
                code = f'''{{
  s101h::GemmArgs<1> g{{}};
  g.p[0] = {{{_ident(a)}, {sam}LL, {sak}LL, {bsa}LL, 0, {_ident(b)}, {sbk}LL, {sbn}LL, {bsb}LL, 0, {k}}};
  g.M = {m}; g.N = {n}; g.nb = {batch}; g.nb2 = 1; g.ksplit = 1;
  s101h::gemm_phase<64,64,{str(sak == 1).lower()},{str(sbk == 1).lower()},1>(g, s101h::EpiStore{{{out}, {n}LL, {m*n}LL}}, smem);
}}'''
                tiles = batch * (m // 64) * (n // 64)
            else:
                mt, nt = (m + 15) // 16, (n + 15) // 16
                tiles = batch * mt * nt
                code = f'''for (long long tile = blockIdx.x; tile < {tiles}LL; tile += gridDim.x) {{
  const long long batch = tile / {mt*nt}LL;
  const int row = (tile / {nt}LL % {mt}LL) * 16 + threadIdx.x / 16;
  const int col = (tile % {nt}LL) * 16 + threadIdx.x % 16;
  float acc = 0.0f;
  for (int kk = 0; kk < {k}; kk += 16) {{
    const int ak = kk + threadIdx.x % 16, bk = kk + threadIdx.x / 16;
    smem[threadIdx.x] = row < {m} && ak < {k} ? {_ident(a)}[batch * {bsa}LL + row * {sam}LL + ak * {sak}LL] : 0.0f;
    smem[256 + threadIdx.x] = bk < {k} && col < {n} ? {_ident(b)}[batch * {bsb}LL + bk * {sbk}LL + col * {sbn}LL] : 0.0f;
    __syncthreads();
    for (int z = 0; z < 16 && kk + z < {k}; ++z)
      acc = fmaf(smem[(threadIdx.x / 16) * 16 + z], smem[256 + z * 16 + threadIdx.x % 16], acc);
    __syncthreads();
  }}
  if (row < {m} && col < {n}) {out}[batch * {m*n}LL + row * {n}LL + col] = acc;
}}'''
            if base == "addmm":
                raise LoweringError("addmm must be decomposed into mm and pointwise add before lowering")
            add_heavy("gemm", op, name, code, refs, tiles=tiles, k=k,
                      engine="pinned_simt_64" if tiled else "generic_simt_16", batch=batch)
            continue
        if base in {"sum", "mean"}:
            source = _ref(args[0], names)
            src = layouts[source]
            integer_sum = base == "sum" and dtypes[source] in {"bool", "int64"} and dtypes[name] == "int64"
            if not integer_sum and (dtypes[source] not in {"float32", "bfloat16"} or dtypes[name] not in {"float32", "bfloat16"}):
                raise LoweringError("reductions require FP32/BF16 operands/results with FP32 accumulation")
            dims = args[1] if len(args) > 1 else kwargs.get("dim")
            dims = list(range(len(src.shape))) if dims is None or dims == [] or dims == () else ([dims] if isinstance(dims, int) else list(dims))
            dims = sorted({int(d) % len(src.shape) for d in dims})
            keep = bool(args[2] if len(args) > 2 else kwargs.get("keepdim", False))
            kept = [i for i in range(len(src.shape)) if i not in dims]
            expected = tuple(1 if i in dims else d for i, d in enumerate(src.shape)) if keep else tuple(src.shape[i] for i in kept)
            if expected != shape:
                raise LoweringError(f"{name}: reduction shape mismatch {expected} != {shape}")
            reduction_shape = tuple(src.shape[i] for i in dims)
            kept_shape = tuple(src.shape[i] for i in kept)
            if p7b_axis_order and kept and dims != list(range(len(dims))):
                # P7-B: when every kept axis has a smaller stride than every reduced axis, view the source with
                # the kept axes trailing (each set keeps its relative order: the reduced flat index, hence every
                # result's element order, and the result order are unchanged), so leading-axis column engines
                # apply (e.g. a channel bias gradient of a channels-last NCHW view).
                big = [src.strides[a] for a in dims if src.shape[a] > 1]
                small = [src.strides[a] for a in kept if src.shape[a] > 1]
                if big and small and min(big) > max(small):
                    order = list(dims) + list(kept)
                    src = TensorLayout(tuple(src.shape[a] for a in order), tuple(src.strides[a] for a in order),
                                       src.root, src.offset)
                    p7b_records.setdefault("axis_order", []).append(name)
                    dims, kept = list(range(len(dims))), list(range(len(dims), len(order)))
            terms = []
            for sequence, subshape, index in [(dims, reduction_shape, "r"), (kept, kept_shape, "o")]:
                for j, axis in enumerate(sequence):
                    terms.append(f"(({index} / {_numel(subshape[j+1:])}LL) % {src.shape[axis]}LL) * {src.strides[axis]}LL")
            pos = " + ".join(terms) or "0"
            scale = f" / {_float(_numel(reduction_shape))}" if base == "mean" else ""
            rounded_result = (lambda e: f"__float2bfloat16_rn({e})") if dtypes[name] == "bfloat16" else (lambda e: e)
            extent, results = _numel(reduction_shape), _numel(shape)
            if p7b_segment and not integer_sum and not retain_all and target_arch == "sm_90" and exact_reductions:
                # P7-B: candidate for a segment engine (row reductions over the innermost axis, middle-axis column
                # reductions of [S, T, D]); its leaf position is recorded in the engine's own variables (g, c) /
                # (t, o). Candidates keep a single-node fallback engine (no P-OPS4-A split, no merged trees).
                p7b_kind = p7b_reduction_kind(src, dims, results, extent)
                if p7b_kind is not None:
                    from . import norm_rules as _nr
                    p7b_reds[name] = {
                        "kind": p7b_kind, "source": source, "extent": extent, "results": results,
                        "scale": scale, "dtype": dtypes[name],
                        "pos": (("i" if p7b_segment_index else f"(g * {_u(src.shape[-1])} + c)")
                                if tuple(src.strides) == tuple(_contiguous(src.shape)) else
                                _nr.position(src.shape, src.strides, dims, kept, "c", "g") if p7b_kind == "row" else
                                _nr.position(src.shape, src.strides, dims, kept, "t", "o")),
                        "segments": src.shape[0] if p7b_kind == "colT" else None,
                        "segment_rows": src.shape[1] if p7b_kind == "colT" else None,
                        "row_extent": extent if p7b_kind == "row" else src.shape[2]}
            if (pops4a and not integer_sum and not retain_all and target_arch == "sm_90" and exact_reductions
                    and name not in p7b_reds):
                emitted = p4a_reduction(op, name, out, source, src, dims, kept, extent, results, refs,
                                        rounded_result, scale)
                if emitted:
                    continue
            if (split_reductions and not integer_sum and extent >= SPLIT_MIN_EXTENT and results <= SPLIT_MAX_RESULTS
                    and name not in p7b_reds):
                root = f"{name}_split_pack"
                if root in names or root in layouts:
                    raise LoweringError(f"split reduction scratch name collision: {root}")
                column = bool(kept) and src.strides[kept[-1]] == 1 and src.shape[kept[-1]] >= 32
                code, extra, part_words = _split_reduction_code(
                    out=_ident(name), pack=_ident(root), read=storage_read(source, pos), extent=extent,
                    results=results, column=column, rounded=rounded_result, scale=scale,
                    out_words=-(-results * _BYTES[dtypes[name]] // 4))
                layouts[root] = TensorLayout((part_words,), (1,), root)
                dtypes[root] = "float32"
                layouts[name] = TensorLayout(shape, _contiguous(shape), root, 0)
                views.append({"name": name, "target": target + ":split_result", "source": root})
                add_heavy("reduction", op, root, code, refs, reduction_elements=extent, **extra)
                continue
            if integer_sum:
                # Stage4's exact count/index reduction: retain integer bits all
                # the way through the tree, including values beyond 2**24.
                code = f'''for (long long o = blockIdx.x; o < {_numel(shape)}LL; o += gridDim.x) {{
  unsigned long long acc = 0;
  for (long long r = threadIdx.x; r < {_numel(reduction_shape)}LL; r += blockDim.x)
    acc += static_cast<unsigned long long>({storage_read(source, pos)});
  for (int delta = 16; delta > 0; delta /= 2) acc += __shfl_down_sync(0xffffffffU, acc, delta);
  unsigned long long* partials = reinterpret_cast<unsigned long long*>(smem);
  if ((threadIdx.x & 31) == 0) partials[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? partials[threadIdx.x] : 0ULL;
    for (int delta = 16; delta > 0; delta /= 2) acc += __shfl_down_sync(0xffffffffU, acc, delta);
    if (threadIdx.x == 0) {out}[o] = static_cast<int64_t>(acc);
  }}
  __syncthreads();
}}'''
                add_heavy("reduction", op, name, code, refs, reduction_elements=_numel(reduction_shape),
                          arithmetic="integer modulo2**64 sum; no FP32 conversion")
                continue
            # The CTA tree below is the frozen fixed order. A token stands for the
            # (possibly fused-producer) source element so materialization stays a
            # scheduling decision; the summed values and their order are unchanged.
            code = f'''for (long long o = blockIdx.x; o < {_numel(shape)}LL; o += gridDim.x) {{
  float acc = 0.0f;
  for (long long r = threadIdx.x; r < {_numel(reduction_shape)}LL; r += blockDim.x)
    acc += {storage_read(source, pos)};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {out}[o] = {rounded_result('acc' + scale)};
  }}
  __syncthreads();
}}'''
            extent, results = _numel(reduction_shape), _numel(shape)
            if pops3c and exact_reductions:
                # P-OPS3-C: same frozen CTA tree, warp-per-row / coalesced column work assignment.
                from . import norm_rules
                if pops3c_row and norm_rules.row_eligible(src.shape, src.strides, dims, results, extent):
                    leaf = storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pcol", "prow"))
                    code = norm_rules.row_code(out=out, rows=results, extent=extent, leaf=leaf,
                                               rounded=rounded_result, scale=scale)
                    add_heavy("reduction", op, name, code, refs, engine="pops3c_row", reduction_elements=extent,
                              row_rows=results, row_extent=extent, row_scale=scale)
                    continue
                if pops3c_column and norm_rules.column_eligible(src.shape, src.strides, dims, results, extent):
                    leaf = storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pr", "o"))
                    code = norm_rules.column_code(out=out, columns=results, extent=extent, leaf=leaf,
                                                  rounded=rounded_result, scale=scale, group=pops3c_col_group)
                    add_heavy("reduction", op, name, code, refs, engine="pops3c_col", reduction_elements=extent,
                              column_groups=-(-results // pops3c_col_group))
                    continue
            if exact_reductions:
                from .codegen_rules import exact_reduction
                # P6-B: thread-per-result replay when the kept innermost axis is unit-stride (coalesced leaves).
                coalesced = (p6b_column_tree and bool(kept) and src.strides[kept[-1]] == 1
                             and src.shape[kept[-1]] >= 32)
                leaf_reads = []

                def leaf_read(position):
                    leaf_reads.append(storage_read(source, position))
                    return leaf_reads[-1]
                selected = exact_reduction(out=out, position=pos, extent=extent, results=results,
                                           read=leaf_read, rounded=rounded_result, scale=scale,
                                           coalesced_threads=coalesced, quad_threads=p6b_tree_quads)
                if (p6b_index32 and selected is not None
                        and selected[1].get('engine') == 'exact_cta_tree_per_thread_column'):
                    # applied to the expanded code (leaf tokens included) when the item is emitted
                    selected = (selected[0], dict(selected[1], p6b_index32=True))
                if selected is not None and selected[1].get('engine') == 'exact_cta_tree_per_thread_column':
                    # Every source element is one leaf of one result: count one use (inlining decisions as before).
                    p4a_state["sibling_tokens"].update(int(_TOKEN.search(leaf).group(1)) for leaf in leaf_reads[1:])
                    if p6b_tree_merge and name not in p7b_reds and p6b_tree_group(op, name, source, src, dims, extent, results, refs,
                                                         selected[0]):
                        continue
                if selected is not None:
                    code, extra = selected
                    add_heavy('reduction', op, name, code, refs, **extra)
                    continue
            if (p7b_split_generic and not integer_sum and extent >= P7B_SPLIT_MIN_EXTENT
                    and 2 * results <= P7B_SPLIT_GRID):
                # P7-B: few results over a long extent: each result's extent in P contiguous parts over the grid
                # (one CTA tree per part), the last arriving part of a result sums the P partials in part order.
                P = max(2, min(extent // P7B_SPLIT_MIN_ROWS, P7B_SPLIT_GRID // results))
                chunk = -(-extent // P)
                P = -(-extent // chunk)
                leaf = storage_read(source, pos.replace("LL", "u"))
                value = rounded_result("(latch_v" + scale + ")")
                code = (f"{{ // P7-B split reduction: {results} results x {extent} in {P} parts\n"
                        f"  unsigned* latch_cnt = /*P2_CNT*/;\n  float* latch_part = /*P2_PART*/;\n"
                        f"  __shared__ bool latch_last;\n"
                        f"  for (unsigned latch_u = blockIdx.x; latch_u < {_u(results * P)}; latch_u += gridDim.x) {{\n"
                        f"    const unsigned o = latch_u / {P}u, latch_p = latch_u % {P}u;\n"
                        f"    const unsigned latch_r1 = min({_u(extent)}, (latch_p + 1u) * {_u(chunk)});\n"
                        f"    float acc = 0.0f;\n#pragma unroll 8\n"
                        f"    for (unsigned r = latch_p * {_u(chunk)} + threadIdx.x; r < latch_r1; r += blockDim.x)\n"
                        f"      acc += {leaf};\n"
                        f"    acc = s101h::warp_sum(acc);\n"
                        f"    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;\n    __syncthreads();\n"
                        f"    if (threadIdx.x == 0u) {{\n      float latch_v = smem[0];\n"
                        f"      for (unsigned k = 1; k < 8u; ++k) latch_v += smem[k];\n"
                        f"      latch_part[latch_u] = latch_v;\n      __threadfence();\n"
                        f"      const unsigned old = atomicAdd(latch_cnt + o, 1u);\n"
                        f"      latch_last = old == {P - 1}u;\n      if (latch_last) atomicExch(latch_cnt + o, 0u);\n    }}\n"
                        f"    __syncthreads();\n"
                        f"    if (latch_last && threadIdx.x == 0u) {{\n      __threadfence();\n"
                        f"      float latch_v = __ldcg(latch_part + o * {P}u);\n"
                        f"      for (unsigned q = 1; q < {P}u; ++q) latch_v += __ldcg(latch_part + o * {P}u + q);\n"
                        f"      {out}[o] = {value};\n    }}\n    __syncthreads();\n  }}\n}}")
                a64 = lambda n_: -(-n_ // 64) * 64
                p7b_records.setdefault("split_generic", []).append({"name": name, "results": results,
                                                                     "extent": extent, "parts": P})
                add_heavy("reduction", op, name, code, refs, reduction_elements=extent, work_items=results * P,
                          p2_split={"counter_words": a64(results), "partial_words": a64(results * P)})
                continue
            if p6b_generic_unroll and code.count("\n  for (long long r = threadIdx.x;") == 1:
                # P6-B: the generic frozen CTA tree with its row loop unrolled (each thread's loads issue ahead of
                # its in-order accumulation; same additions in the same order) and 32-bit indices.
                code = code.replace("\n  for (long long r = threadIdx.x;", "\n#pragma unroll 8\n  for (long long r = threadIdx.x;")
                add_heavy("reduction", op, name, code, refs, reduction_elements=extent, p6b_index32=True)
                continue
            add_heavy("reduction", op, name, code, refs, reduction_elements=extent)
            continue

        if (pops3c and pops3c_factor and not retain_all and name not in protected_names
                and dtypes[name] in ("float32", "bfloat16")):
            # P-OPS3-C: a pointwise operation of a single broadcast (expanded) operand is evaluated once per
            # source element and itself viewed as the same expansion (identical per-element arithmetic).
            from .norm_rules import FACTORABLE
            tensor_args = [k for k, a in enumerate(args) if _ref(a, names) is not None]
            if (base in FACTORABLE and len(tensor_args) == 1 and consumer_bases.get(name, set()) <= FACTORABLE
                    and not any(_ref(v, names) for v in kwargs.values())):
                ref = _ref(args[tensor_args[0]], names)
                view = next((v for v in reversed(views) if v["name"] == ref), None)
                if view is not None and "expand" in str(view["target"]) and layouts[ref].shape == shape:
                    source_layout = layouts[view["source"]]
                    compact = (1,) * (len(shape) - len(source_layout.shape)) + tuple(source_layout.shape)
                    if (_numel(compact) * 4 <= _numel(shape) and len(compact) == len(shape)
                            and all(c in (1, d) for c, d in zip(compact, shape))):
                        factored = name + "__p3row"
                        compact_strides = _contiguous(compact)
                        layouts[factored] = TensorLayout(compact, compact_strides, factored)
                        dtypes[factored] = dtypes[name]
                        layouts[name] = TensorLayout(shape, tuple(0 if c == 1 and d != 1 else st for c, d, st in
                                                                  zip(compact, shape, compact_strides)), factored, 0)
                        views.append({"name": name, "target": "pops3c_factored_expand", "source": factored})
                        factored_values[name] = factored
                        names.add(factored)
                        original = args[tensor_args[0]]
                        args = list(args)
                        args[tensor_args[0]] = ({"ref": view["source"]} if isinstance(original, dict)
                                                else type(original)(view["source"]))
                        name, shape = factored, compact

        from . import extra_ops
        if extra_ops.uses_expression(base, kwargs, dtypes[name], target):
            for helper in (extra_ops.EXTRA_HELPERS, extra_ops.FIELD_HELPERS):
                if helper not in helper_parts:
                    helper_parts.append(helper)

        def make_render(base=base, args=args, kwargs=kwargs, name=name, target=target, shape=shape,
                        indexed_elementwise=indexed_elementwise):
            def render(idx, rd_ref):
                rd = lambda i: rd_ref(args[i], idx)
                # The B protocol has already admitted this operation and skipped
                # its heavy emitter. Preserve that typed expression before the
                # generic union (whose arange rule intentionally handles int64).
                if indexed_elementwise:
                    return p6b_selection_render(base, target, args, name, shape, idx, rd_ref)
                extra = extra_ops.expression(base, target, args, kwargs, name, dtypes, names, rd, idx, shape,
                                             input_names=tuple(inputs))
                if extra is not None:
                    return extra
                from .expression_rules import standard_expression
                extra = standard_expression(base, target, args, kwargs, dtypes, names, rd, layouts=layouts, name=name)
                if extra is not None:
                    return extra
                if base == "slice_backward":
                    grad = _ref(args[0], names)
                    grad_shape = tuple(int(x) for x in _field(values[grad], "shape"))
                    condition, position = slice_backward_index(args, shape, grad_shape, idx, unsigned=True)
                    return f"({condition} ? {rd_ref(args[0], position, grad_shape)} : 0.0f)"
                if base == "scalar_tensor":
                    if shape != () or len(args) != 1 or not isinstance(args[0], (int, float, bool)):
                        raise LoweringError("scalar_tensor requires one static numeric scalar and scalar output")
                    if dtypes[name] == "int64":
                        if not isinstance(args[0], (int, bool)):
                            raise LoweringError("int64 scalar_tensor requires an integer literal")
                        return f"static_cast<int64_t>({int(args[0]) % (1 << 64)}ULL)"
                    if dtypes[name] == "bool":
                        return str(bool(args[0])).lower()
                    return _double(args[0]) if dtypes[name] == "float64" else _float(args[0])
                if base == "where":
                    if len(args) != 3 or dtypes[_ref(args[0], names)] != "bool":
                        raise LoweringError("where requires an explicit boolean condition and two tensor operands")
                    if any(_ref(arg, names) is not None and dtypes[_ref(arg, names)] != dtypes[name] for arg in args[1:]):
                        raise LoweringError("where requires matching value/result dtypes")
                    def selection(index):
                        if dtypes[name] == "float64" and _ref(args[index], names) is None:
                            return _double(args[index])
                        if _ref(args[index], names) is None and dtypes[name] == "int64":
                            if type(args[index]) is not int:
                                raise LoweringError("int64 where requires integer scalar values")
                            return f"static_cast<int64_t>({args[index] % (1 << 64)}ULL)"
                        return rd(index)
                    return f"({rd(0)} ? {selection(1)} : {selection(2)})"
                if dtypes[name] not in {"float32", "bfloat16"}:
                    if base == "_to_copy" and dtypes[_ref(args[0], names)] in {"int64", "bool"}:
                        return f"static_cast<{_CTYPES[dtypes[name]]}>({rd(0)})"
                    if base in {"zeros_like", "ones_like", "zeros", "ones", "new_zeros", "new_ones", "full", "full_like", "new_full", "fill"}:
                        if base == "fill":
                            value = args[1]
                        elif base in {"full", "full_like", "new_full"}:
                            value = args[2] if base == "new_full" else args[1]
                        else:
                            value = 1 if "ones" in base else 0
                        if dtypes[name] == "float64":
                            return _double(value)
                        if not isinstance(value, (int, bool)):
                            raise LoweringError("integer/bool fills require integer scalar literals")
                        return (f"static_cast<int64_t>({int(value) % (1 << 64)}ULL)" if dtypes[name] == "int64"
                                else str(bool(value)).lower())
                    if base in {"clone", "contiguous", "lift_fresh_copy"} and dtypes[_ref(args[0], names)] == dtypes[name]:
                        return rd(0)
                    if dtypes[name] == "bool":
                        if base in {"logical_not", "bitwise_not"} and dtypes[_ref(args[0], names)] == "bool":
                            return f"(!{rd(0)})"
                        if base in {"eq", "ne", "lt", "le", "gt", "ge"}:
                            symbol = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}[base]
                            # Preserve int64 comparison precision instead of routing
                            # integer literals through the FP32 scalar renderer.
                            def comparison_arg(index):
                                ref = _ref(args[index], names)
                                if ref is None and type(args[index]) is int:
                                    return f"static_cast<int64_t>({args[index] % (1 << 64)}ULL)"
                                return rd(index)
                            return f"({comparison_arg(0)} {symbol} {comparison_arg(1)})"
                        if base in {"logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor"}:
                            if any(dtypes[_ref(arg, names)] != "bool" for arg in args):
                                raise LoweringError("boolean combination requires bool tensor operands")
                            symbol = {"logical_and": "&&", "logical_or": "||", "logical_xor": "!=", "bitwise_and": "&&", "bitwise_or": "||", "bitwise_xor": "!="}[base]
                            return f"({rd(0)} {symbol} {rd(1)})"
                    if dtypes[name] == "int64" and base in {"add", "sub", "mul"}:
                        def integer_arg(index):
                            ref = _ref(args[index], names)
                            if ref is None:
                                if type(args[index]) is not int:
                                    raise LoweringError("integer operations require integer scalar literals")
                                return f"{args[index] % (1 << 64)}ULL"
                            if dtypes[ref] != "int64":
                                raise LoweringError("implicit mixed integer arithmetic is unsupported")
                            return f"static_cast<uint64_t>({rd(index)})"
                        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                        if type(alpha) is not int:
                            raise LoweringError("integer alpha must be an integer")
                        op = {"add": "+", "sub": "-", "mul": "*"}[base]
                        right = integer_arg(1)
                        if base in {"add", "sub"}:
                            right = f"({right} * {alpha % (1 << 64)}ULL)"
                        # Unsigned arithmetic avoids signed-overflow UB. CUDA's
                        # two's-complement conversion matches torch.int64 wrapping.
                        return f"static_cast<int64_t>({integer_arg(0)} {op} {right})"
                    raise LoweringError(f"integer/bool arithmetic needs an explicit rule: {target}")
                if base == "masked_fill":
                    if target != "masked_fill.Scalar" or dtypes[_ref(args[1], names)] != "bool":
                        raise LoweringError("masked_fill admits a bool mask and a static scalar value")
                    return f"({rd(1)} ? {_float(args[2])} : {rd(0)})"
                if base == "rsub":
                    # torch.rsub(input, other, alpha) = other - alpha * input
                    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                    return f"({rd(1)} - ({rd(0)} * {_float(alpha)}))"
                if base in {"add", "sub"}:
                    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                    return f"({rd(0)} {'+' if base == 'add' else '-'} ({rd(1)} * {_float(alpha)}))"
                if base in {"mul", "div"}:
                    if base == "div" and kwargs.get("rounding_mode") is not None:
                        raise LoweringError("rounded division unsupported")
                    if base == "div" and outline_ops and _ref(args[1], names) is not None:
                        return f"latch_fdiv({rd(0)}, {rd(1)})"
                    return f"({rd(0)} {'*' if base == 'mul' else '/'} {rd(1)})"
                if base == "pow":
                    if args[1] == 2:
                        return f"({rd(0)} * {rd(0)})"
                    if args[1] == 0.5 and target == "pow.Tensor_Scalar":
                        return f"sqrtf({rd(0)})"
                    if pops3c and pops3c_pow1 and isinstance(args[1], (int, float)) and float(args[1]) == 1.0:
                        # P-OPS3-C: libdevice powf(x, 1.0f) == x for every float bit pattern (exhaustive H100 check,
                        # pops3c_jobs.pow_identity_check); NaN payloads are not values the program can observe.
                        return f"({rd(0)})"
                    return f"powf({rd(0)}, {rd(1)})"
                if base in {"erf", "sin", "cos"}:
                    return f"{base}f({rd(0)})"
                if base in {"clamp", "clamp_min", "clamp_max"}:
                    low = args[1] if base in {"clamp", "clamp_min"} and len(args) > 1 else kwargs.get("min")
                    high = (args[2] if len(args) > 2 else kwargs.get("max")) if base == "clamp" else (args[1] if base == "clamp_max" else None)
                    if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in (low, high)):
                        raise LoweringError("clamp admits static scalar bounds only")
                    value = rd(0)
                    # NaN propagates as in torch: comparisons with NaN are false.
                    if low is not None:
                        value = f"({value} < {_float(low)} ? {_float(low)} : {value})"
                    if high is not None:
                        value = f"({value} > {_float(high)} ? {_float(high)} : {value})"
                    return value
                if base in {"minimum", "maximum"}:
                    symbol = "<" if base == "minimum" else ">"
                    return f"(({rd(0)} != {rd(0)}) || ({rd(1)} != {rd(1)}) ? __int_as_float(0x7fc00000) : ({rd(0)} {symbol} {rd(1)} ? {rd(0)} : {rd(1)}))"
                if base == "neg":
                    return f"(-{rd(0)})"
                if base == "reciprocal":
                    if outline_ops:
                        return f"latch_fdiv(1.0f, {rd(0)})"
                    return f"(1.0f / {rd(0)})"
                if base in {"tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs"}:
                    fn = {"abs": "fabsf"}.get(base, base + "f")
                    return f"{fn}({rd(0)})"
                if base in {"sigmoid", "silu"} and outline_ops:
                    return f"latch_{base}({rd(0)})"
                if base == "silu_backward" and outline_ops:
                    return f"latch_silu_backward({rd(0)}, {rd(1)})"
                if base == "sigmoid":
                    return f"(1.0f / (1.0f + expf(-{rd(0)})))"
                if base == "silu":
                    return f"({rd(0)} / (1.0f + expf(-{rd(0)})))"
                if base == "silu_backward":
                    sigmoid = f"(1.0f / (1.0f + expf(-{rd(1)})))"
                    return f"({rd(0)} * {sigmoid} * (1.0f + {rd(1)} * (1.0f - {sigmoid})))"
                if base == "tanh_backward":
                    return f"({rd(0)} * (1.0f - {rd(1)} * {rd(1)}))"
                if base == "sigmoid_backward":
                    return f"({rd(0)} * {rd(1)} * (1.0f - {rd(1)}))"
                if base == "relu":
                    return f"fmaxf({rd(0)}, 0.0f)"
                if base == "threshold_backward":
                    return f"({rd(1)} <= {rd(2)} ? 0.0f : {rd(0)})"
                if base in {"clone", "contiguous", "_to_copy", "lift_fresh_copy"}:
                    return rd(0)
                if base == "fill":
                    return rd(1)
                if base in {"ones_like", "zeros_like"}:
                    return "1.0f" if base == "ones_like" else "0.0f"
                if base in {"full", "full_like", "new_full"}:
                    return _float(args[2] if base == "new_full" else args[1])
                if base in {"zeros", "new_zeros", "ones", "new_ones"}:
                    return "1.0f" if "ones" in base else "0.0f"
                raise LoweringError(f"unsupported operation {target} at {name}")
            return render

        render = make_render()
        # Validate and count operand loads once (also rejects unsupported targets early).
        probe_loads = []

        def probe(arg, idx, shape=None):
            ref = _ref(arg, names)
            if ref is None:
                if isinstance(arg, (int, float, bool)):
                    return _float(arg)
                raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
            probe_loads.append(ref)
            return f"v_{ref}[0]"
        render("i", probe)
        own = 4 if base in TRANSCENDENTAL else (0 if not probe_loads else 1)
        if p6b_node_cse and not retain_all and name not in protected_names and name not in factored_values:
            # P6-B: an elementwise value with the identical per-element expression over the identical operands
            # (same roots, offsets and index maps) as an earlier one is that value: alias it instead of recomputing
            # and storing a copy.
            def signature_ref(arg, idx, shape_=None):
                ref = _ref(arg, names)
                if ref is None:
                    return _float(arg) if isinstance(arg, (int, float, bool)) else repr(arg)
                lay = layouts[ref]
                return (f"<{lay.root}+{lay.offset}:{_index32(lay, shape if shape_ is None else shape_, idx)}"
                        f":{dtypes[ref]}>")
            try:
                signature = (render("i", signature_ref), tuple(shape), dtypes[name])
            except Exception:
                signature = None
            first = p6b_records.setdefault("_cse_seen", {}).get(signature) if signature else None
            if (first is not None and p7b_remat and _numel(shape) >= P7B_REMAT_MIN
                    and p7b_base_bytes(probe_loads, _numel(shape)) < _BYTES[dtypes[name]]):
                # P7-B: no alias when recomputing the value reads fewer bytes than keeping the first copy stored
                # (an alias across the forward/backward boundary would keep an FP32 activation function value live).
                p7b_records.setdefault("cse_kept_separate", []).append(name)
                first = None
            if first is not None and first in nodes and layouts[first].contiguous and layouts[first].offset == 0:
                layouts[name] = TensorLayout(shape, _contiguous(shape), layouts[first].root, 0)
                views.append({"name": name, "target": "p6b_value_cse", "source": first})
                p6b_records.setdefault("value_cse", []).append({"value": name, "same_as": first})
                continue
            if signature:
                p6b_records["_cse_seen"][signature] = name
        nodes[name] = _Node(name=name, kind="elementwise", shape=shape, dtype=dtypes[name], render=render,
                            base=base, target=str(_field(op, "target", _field(op, "op"))),
                            own_cost=own, loads=probe_loads, order=len(nodes))

    # ------------------------------------------------------------------ writes
    for target, source in writes.items():
        if target not in inputs:
            raise LoweringError(f"state write target {target} is not an input")
        if layouts[target].shape != layouts[source].shape or dtypes[target] != dtypes[source] or not layouts[target].contiguous:
            raise LoweringError(f"state write {target} needs equal shapes and contiguous destination")
    for target, source in writes.items():
        if layouts[source].root in writes and layouts[source].root != target:
            raise LoweringError(f"state output {source} aliases another mutable input; version-preserving copy required")
    for name in outputs:
        if layouts[name].root in writes:
            raise LoweringError(f"output {name} aliases old mutable input; version-preserving copy required")
    root_of = lambda ref: layouts[ref].root
    sources = {}
    for target, source in writes.items():
        sources.setdefault(root_of(source), []).append(target)

    # ------------------------------------------------------------------ materialization
    forced = set()
    for node in nodes.values():
        if node.kind != "elementwise":
            for ref in node.direct:
                forced.add(root_of(ref))
    for name in outputs + list(writes.values()):
        forced.add(root_of(name))
    uses = {}
    for node in nodes.values():
        if node.kind == "elementwise":
            for ref in node.loads:
                uses[root_of(ref)] = uses.get(root_of(ref), 0) + 1
    for k, (kind, ref, _, _) in enumerate(token_loads):
        if k in p4a_state["sibling_tokens"]:
            continue    # P-OPS4-A: per-column copies of one leaf read each element once
        uses[root_of(ref)] = uses.get(root_of(ref), 0) + 1
    inline, total_cost = set(), {}

    def p7b_input_bytes(node, seen=None):
        """Bytes per element read from memory by `node`'s expression through already-inlined producers (full-size
        operands only; broadcast operands are cached and counted as free)."""
        seen = set() if seen is None else seen
        total = 0
        n = _numel(node.shape)
        for ref in node.loads:
            root = root_of(ref)
            if root in seen:
                continue
            seen.add(root)
            if root in inline:
                total += p7b_input_bytes(nodes[root], seen)
            elif layouts[ref].numel >= n:
                total += _BYTES.get(dtypes[root], 4)
        return total
    def p9e_stored_leaves(node):
        """P9-E: the full-size stored roots `node` reads through already-inlined producers (row statistics and
        other values whose storage is smaller than `node` are free); None when any of them is not FP32."""
        n, seen, out, stack = _numel(node.shape), set(), set(), list(node.loads)
        while stack:
            root = root_of(stack.pop())
            if root in seen:
                continue
            seen.add(root)
            if root in inline:
                stack.extend(nodes[root].loads)
                continue
            shape_ = nodes[root].shape if root in nodes else (layouts[root].shape if root in layouts else None)
            if shape_ is None and root in values:   # e.g. a row reduction lowered after this decision
                shape_ = tuple(int(x) for x in _field(values[root], "shape"))
                if _numel(shape_) < n:
                    if node.shape and _numel(shape_) * int(node.shape[-1]) == n:
                        p9e_rowstat.add(node.name)
                    continue
            if shape_ is None or root not in dtypes:
                p9e_leaf_why[node.name] = f"unknown root {root}"
                return None
            if _numel(shape_) < n:
                if node.shape and _numel(shape_) * int(node.shape[-1]) == n:
                    p9e_rowstat.add(node.name)    # a per-row statistic (e.g. mean / rstd of a normalization)
                continue
            if dtypes[root] != "float32":
                p9e_leaf_why[node.name] = f"{root} {dtypes[root]} {list(shape_)}"
                return None
            out.add(root)
        return out
    p9e_leaf_why, p9e_rowstat = {}, set()
    column_only = set()
    if pops3c:
        # P-OPS3-C: a value whose only consumer is a column engine is recomputed in the engine's leaf (a stored
        # loop would add a phase between the producer and the weight-gradient reduction).
        consumers = {}
        for node in nodes.values():
            if node.kind == "elementwise":
                for ref in node.loads:
                    consumers.setdefault(root_of(ref), set()).add(node.name)
            else:
                for k in node.loads:
                    consumers.setdefault(root_of(token_loads[k][1]), set()).add(node.name)
        column_only = {name for name, readers_ in consumers.items()
                       if readers_ and all(nodes[r].kind != "elementwise" and nodes[r].extra.get("engine") == "pops3c_col"
                                           for r in readers_)}
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        cost = node.own_cost + sum(total_cost[root_of(ref)] for ref in node.loads if root_of(ref) in inline)
        total_cost[node.name] = cost
        gather_ids = p6b_index_elementwise and node.base in ("index", "embedding")   # int64 row ids, checked
        if (p9e_remat_stored and node.dtype == "float32" and _numel(node.shape) >= P7B_REMAT_MIN
                and len(p7b_records.get("remat_stored_seen", ())) < 64
                and all(dtypes[root_of(r)] == "float32" for r in node.loads if layouts[r].numel >= _numel(node.shape))):
            p7b_records.setdefault("remat_stored_seen", []).append(
                {"name": node.name, "forced": node.name in forced, "uses": uses.get(node.name, 0), "cost": cost,
                 "leaves": sorted(p9e_stored_leaves(node)) if p9e_stored_leaves(node) is not None else p9e_leaf_why.get(node.name),
                 "bytes": p7b_input_bytes(node), "inline_inputs": [root_of(r) for r in node.loads if root_of(r) in inline]})
        if (retain_all or node.name in forced or node.dtype not in ("float32", "bfloat16")
                or any(dtypes[root_of(ref)] not in ("float32", "bfloat16")
                       and not (gather_ids and dtypes[root_of(ref)] == "int64") for ref in node.loads)):
            continue
        n_uses = uses.get(node.name, 0)
        if p6b_scalar_store and _numel(node.shape) == 1 and cost >= 4:
            # P6-B: a transcendental scalar (e.g. AdamW bias corrections of a step counter) is computed once and
            # stored; inlined, it would be recomputed for every element of each consumer loop.
            continue
        if (p9e_remat_stored and n_uses and cost <= int(p9e_remat_stored)
                and cost * n_uses <= 4 * int(p9e_remat_stored) and node.dtype == "float32"
                and _numel(node.shape) >= P7B_REMAT_MIN and p9e_stored_leaves(node) is not None
                and sum(_BYTES[dtypes[r]] for r in p9e_stored_leaves(node)) <= _BYTES[node.dtype]
                and (not p9e_remat_rowstat or node.name in p9e_rowstat)):
            # P9-E: an FP32 value computed from FP32 values that stay stored anyway (e.g. a normalized activation
            # from its stored input and per-row statistics) is recomputed at its uses instead of being stored:
            # the uses read the same bytes per element and the store disappears. Same expression, same value.
            # (Checked before the P7-B byte rule, whose model counts broadcast row statistics as full reads.)
            inline.add(node.name)
            p7b_records.setdefault("remat_stored", []).append(node.name)
            continue
        if (p7b_inline_bytes and n_uses >= 2 and _numel(node.shape) >= P7B_REMAT_MIN
                and p7b_input_bytes(node) * n_uses > _BYTES[node.dtype] * (n_uses + 1)):
            # P7-B: recomputing this value in each of its uses would read more bytes than storing it once and reading
            # it back; keep it a stored value (same expression, evaluated once).
            p7b_records.setdefault("kept_stored", []).append(node.name)
            continue
        if n_uses and cost <= INLINE_TOTAL_LIMIT and cost * n_uses <= INLINE_WORK_LIMIT:
            inline.add(node.name)
        elif (p7b_remat and n_uses == 1 and cost <= int(p7b_remat) and _numel(node.shape) >= P7B_REMAT_MIN
              and p7b_input_bytes(node) < _BYTES[node.dtype]):
            # P7-B: a single-use value is recomputed at its consumer when the recomputation reads fewer bytes per
            # element than storing and re-reading it (e.g. a backward-only derivative of a forward activation would
            # otherwise be evaluated ASAP in the forward and kept in FP32 until the backward). Same expression.
            inline.add(node.name)
            p7b_records.setdefault("remat", []).append(node.name)
        elif node.name in column_only and cost <= 2 * INLINE_TOTAL_LIMIT:
            inline.add(node.name)

    if p7b_cone and not retain_all:
        # P7-B cone rematerialization: a large value whose every consumer is elementwise is recomputed in its consumers
        # (same expression) when that reads fewer bytes per element than storing and re-reading it (e.g. an
        # activation derivative needed only by the backward is otherwise computed in the forward and kept in FP32).
        consumers_of = {}
        for node in nodes.values():
            if node.kind == "elementwise":
                for ref in node.loads:
                    consumers_of.setdefault(root_of(ref), []).append(node)
            else:
                for k_ in node.loads:
                    consumers_of.setdefault(root_of(token_loads[k_][1]), []).append(node)
        changed = True
        while changed:
            changed = False
            for node in nodes.values():
                if (node.kind != "elementwise" or node.name in inline or node.name in forced
                        or node.dtype not in ("float32", "bfloat16") or _numel(node.shape) < P7B_REMAT_MIN):
                    continue
                cons = consumers_of.get(node.name, [])
                if not cons:
                    continue
                if int(p7b_cone) >= 2:
                    # mode 2: any non-product consumer (reduction leaves evaluate the expression once per element);
                    # recompute bytes x uses must beat one store plus a reload per use
                    if any(c.kind == "gemm" or (c.kind != "elementwise" and c.name not in p7b_reds) for c in cons):
                        continue
                elif any(c.kind != "elementwise" for c in cons):
                    continue
                cost = node.own_cost + sum(total_cost.get(root_of(ref), 0) for ref in node.loads
                                           if root_of(ref) in inline)
                n_cons = len({c.name for c in cons})
                if cost > P7B_CONE_COST:
                    continue
                if int(p7b_cone) >= 2:
                    if p7b_input_bytes(node) * n_cons >= _BYTES[node.dtype] * (1 + n_cons):
                        continue
                elif p7b_input_bytes(node) >= _BYTES[node.dtype]:
                    continue
                inline.add(node.name)
                total_cost[node.name] = cost
                p7b_records.setdefault("cone", []).append(node.name)
                changed = True
        coned = set(p7b_records.get("cone", ()))
        for node in nodes.values():
            if node.kind == "elementwise":
                total_cost[node.name] = node.own_cost + sum(total_cost.get(root_of(ref), 0) for ref in node.loads
                                                            if root_of(ref) in inline)
                if (p7b_cone_recheck and node.name in inline and node.name not in coned
                        and any(c.kind != "elementwise" for c in consumers_of.get(node.name, ()))
                        and (total_cost[node.name] > INLINE_TOTAL_LIMIT
                             or total_cost[node.name] * uses.get(node.name, 0) > INLINE_WORK_LIMIT)):
                    # P7-B: a value inlined before cone rematerialization grew past the inline bounds (its operands are
                    # now recomputed): store it once instead of re-evaluating the whole cone in every consumer.
                    inline.discard(node.name)
                    total_cost[node.name] = node.own_cost
                    p7b_records.setdefault("cone_restored", []).append(node.name)

    def ctype_round(expression, dtype):
        return f"__float2bfloat16_rn({expression})" if dtype == "bfloat16" else expression

    def expand(ref, shape, idx, loader):
        """Float expression for element idx of `ref` viewed at `shape`."""
        layout = layouts[ref]
        root = layout.root
        if root in inline:
            node = nodes[root]
            position = _index32(layout, shape, idx)
            if layout.offset:
                position = f"{_u(layout.offset)} + {position}"
            position = position if re.fullmatch(r"\w+", position) else f"({position})"
            inner = node.render(position, lambda arg, i2, s=None: operand(arg, node.shape if s is None else s, i2, loader))
            if node.dtype == "bfloat16":
                return f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
            return inner
        return loader(ref, shape, idx)

    def operand(arg, shape, idx, loader):
        ref = _ref(arg, names)
        if ref is None:
            if isinstance(arg, (int, float, bool)):
                return _float(arg)
            raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
        return expand(ref, shape, idx, loader)

    def memory_loads(node):
        """(root, numel(root), idx, consumer numel) for every memory load, through inlining."""
        result = []

        def loader(ref, shape, idx):
            result.append((root_of(ref), layouts[ref], tuple(shape) if shape is not None else None, idx))
            return f"v_{ref}[0]"
        if node.kind == "elementwise":
            node.render("i", lambda arg, i2, s=None: operand(arg, node.shape if s is None else s, i2, loader))
        else:
            for k in node.loads:
                kind, ref, shape, idx = token_loads[k]
                if kind == "index":
                    expand(ref, shape, idx, loader)
                else:
                    expand_storage(ref, idx, loader)
            for ref in node.direct:
                result.append((root_of(ref), layouts[ref], None, None))
        return result

    def expand_storage(ref, position, loader):
        layout = layouts[ref]
        root = layout.root
        if root in inline:
            node = nodes[root]
            flat = f"({_u(layout.offset)} + (unsigned)({position}))" if layout.offset else f"(unsigned)({position})"
            inner = node.render(flat, lambda arg, i2, s=None: operand(arg, node.shape if s is None else s, i2, loader))
            if node.dtype == "bfloat16":
                return f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
            return inner
        return loader(ref, None, position)

    # ------------------------------------------------------------------ scheduling
    items = [node for node in nodes.values() if node.kind != "elementwise" or node.name not in inline]
    loads_of = {node.name: memory_loads(node) for node in items}
    from .pops2b_rules_p7b import IDLE_HEAVY_KINDS
    # P-OPS2-B rules belong to programs with exact-order BF16 products; others keep the base schedule.
    pops2b = pops2b and any(node.kind == "gemm" and (node.extra.get("matrix_spec") or {}).get("dtype") == "bfloat16"
                            for node in nodes.values())
    absorbed = {}                         # BF16 cast -> (product, operand) materialized in operand staging
    if pops2b and pops2b_casts and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7b import select_absorbed_casts
        absorbed = select_absorbed_casts(items=items, loads_of=loads_of, nodes=nodes, layouts=layouts, dtypes=dtypes,
                                         protected={root_of(n) for n in outputs + list(writes.values())},
                                         mutable=set(writes), cost_of=lambda n: total_cost.get(n, 0))
        if absorbed:
            items = [node for node in items if node.name not in absorbed]
            for cast, decision in absorbed.items():
                cast_loads = loads_of.pop(cast)
                for product, _ in decision["readers"]:
                    loads_of[product] = [entry for entry in loads_of[product] if entry[0] != cast] + cast_loads

    def register_load(node, root, layout, shape, idx):
        """A same-loop register read: same flat index over the same element count."""
        # P6-B: a one-element loop has one index; a scalar read through an inlined scalar arrives as "0u".
        same_index = idx == "i" or (p6b_scalar_chain and idx in ("0u", "0") and _numel(node.shape) == 1)
        return (node.kind == "elementwise" and same_index and root in nodes and nodes[root].kind == "elementwise"
                and root not in inline and root not in absorbed and layout.root == root and layout.contiguous
                and layout.offset == 0 and _numel(nodes[root].shape) == _numel(node.shape) == _numel(shape))

    # P-OPS3-C: members attached to a row reduction form their own register group (evaluated by the owning warp).
    attached_to, row_members, row_pre = {}, {}, {}
    _same_loop_read = register_load

    def register_load(node, root, layout, shape, idx):
        if attached_to.get(node.name) != attached_to.get(root):
            return False
        return _same_loop_read(node, root, layout, shape, idx)

    readers = {}                          # root -> item names reading it from memory or register
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, []).append(node.name)
    order = {node.name: k for k, node in enumerate(items)}
    in_place = {}
    if not retain_all:
        for source_root, targets in sources.items():
            if len(targets) != 1 or source_root not in nodes or source_root in inline:
                continue
            target = targets[0]
            node = nodes[source_root]
            if (node.kind != "elementwise" or source_root in inputs or layouts[target].shape != node.shape
                    or not layouts[target].contiguous or dtypes[target] != node.dtype
                    or layouts[writes[target]].root != source_root or writes[target] != source_root):
                continue
            if any(order[r] > order[source_root] for r in readers.get(target, ()) if r != source_root):
                continue
            # Old state is only read by earlier items, and the source itself reads it at index i.
            ok = all(idx == "i" and lay.contiguous and lay.offset == 0 and _numel(lay.shape) == _numel(node.shape)
                     for root, lay, shape, idx in loads_of[source_root] if root == target)
            if ok:
                in_place[target] = source_root
    inplace_source = {source: target for target, source in in_place.items()}

    # ------------------------------------------------------------------ P7-B segment groups
    # A segment group is evaluated by one engine pass: a warp owns one row g of an [R, D] iteration space (lane l
    # holds columns c = l + 32 j in registers); elementwise members are computed per element, row reductions with
    # one warp butterfly, middle-axis column reductions of [S, T, D] by per-thread accumulation over the rows of
    # a segment s (one CTA per segment) and a fixed-order combination of the eight warp partials. Membership is
    # derived from shapes, layouts and data dependencies only: two candidates join when one reads the other at its
    # own element (full [R, D] values) or at its own row ([R] values, row reductions), checked numerically on
    # sampled (g, c); groups never span a non-candidate heavy item and are convex. Arithmetic: per-operation
    # expressions unchanged; row/column sums are FP32 reassociations (fixed order, deterministic).
    def p7b_eval(text, env):
        expression = re.sub(r"\(unsigned\)", "", text)
        expression = re.sub(r"(?<![\w.])(\d+)(?:u|LL|ULL)\b", r"\1", expression).replace("/", "//")
        if re.search(r"[^\w\s()+*%/-]", expression):
            return None
        try:
            return int(eval(expression, {"__builtins__": {}}, dict(env)))
        except Exception:
            return None

    def p7b_samples(R, D, T):
        rows = {0, 1, R // 2, R - 1}
        if T and T < R:
            rows |= {T - 1, T, min(R - 1, 2 * T + 1), R - T}
        envs = []
        for g in sorted(r for r in rows if 0 <= r < R):
            for c in sorted({0, 1, D // 2, D - 1}):
                s_, t_ = (g // T, g % T) if T else (0, g)
                envs.append({"g": g, "c": c, "i": g * D + c, "s": s_, "t": t_, "o": s_ * D + c})
        return envs

    def p7b_address(layout, shape, idx, env):
        text = _index32(layout, shape, idx) if shape is not None else str(idx)
        value = p7b_eval(text, env)
        return None if value is None else layout.offset + value

    def p7b_local(root, layout, shape, idx, R, D, T, full):
        """Does this read of `root` hit the consumer's own element (full) / own row (row-shaped) everywhere?"""
        for env in p7b_samples(R, D, T):
            want = env["g"] * D + env["c"] if full else env["g"]
            if p7b_address(layout, shape, idx, env) != want:
                return False
        return True

    def p7b_render_loads(name, space):
        result = []

        def loader(ref, shape, idx):
            result.append((root_of(ref), layouts[ref], tuple(shape) if shape is not None else None, idx))
            return f"v_{ref}[0]"
        node = nodes[name]
        if node.kind == "elementwise":
            index = "i" if _numel(node.shape) == space[0] * space[1] else "g"
            node.render(index, lambda arg, i2, s=None: operand(arg, node.shape if s is None else s, i2, loader))
        else:
            red = p7b_reds[name]
            expand_storage(red["source"], red["pos"], loader)
        return result

    if p7b_segment and not retain_all and target_arch == "sm_90" and p7b_reds:
        space_of = {}
        for node in items:
            if node.kind == "reduction" and node.name in p7b_reds and node.name not in inplace_source:
                red = p7b_reds[node.name]
                rows_ = red["results"] if red["kind"] == "row" else red["segments"] * red["segment_rows"]
                space_of[node.name] = (rows_, red["row_extent"])
        spaces = set(space_of.values())
        protected_roots = {root_of(n) for n in list(writes)}
        for node in items:
            if (node.kind == "elementwise" and node.name not in inplace_source and node.name not in protected_roots
                    and node.dtype in ("float32", "bfloat16") and layouts[node.name].root == node.name):
                n_ = _numel(node.shape)
                fits = [sp for sp in spaces if n_ in (sp[0] * sp[1], sp[0])]
                if len(fits) == 1:
                    space_of[node.name] = fits[0]
        loads7 = {}
        for name in list(space_of):
            try:
                loads7[name] = p7b_render_loads(name, space_of[name])
            except Exception:
                del space_of[name]
        # barrier depth: non-candidate heavy items on the longest path (groups never span one)
        depth = {}
        for node in items:
            d_ = 0
            for root, _, _, _ in loads_of[node.name]:
                if root in depth:
                    step = 0 if (root in space_of or nodes[root].kind == "elementwise") else 1
                    d_ = max(d_, depth[root] + step)
            depth[node.name] = d_

        def seg_rows(name):
            red = p7b_reds.get(name)
            return red["segment_rows"] if red and red["kind"] == "colT" else None

        def is_full(name):
            return nodes[name].kind == "elementwise" and _numel(nodes[name].shape) == space_of[name][0] * space_of[name][1]

        def colt(name):
            return name in p7b_reds and p7b_reds[name]["kind"] == "colT"
        parent7 = {n: n for n in space_of}

        def find7(x):
            while parent7[x] != x:
                parent7[x] = parent7[parent7[x]]
                x = parent7[x]
            return x
        for x in space_of:
            R_, D_ = space_of[x]
            for root, layout, shape, idx in loads7[x]:
                if (root == x or root not in space_of or space_of[root] != (R_, D_) or depth.get(root) != depth.get(x)
                        or colt(root)):
                    continue
                if nodes[x].kind == "elementwise" and not is_full(x) and is_full(root):
                    continue
                if p7b_local(root, layout, shape, idx, R_, D_, seg_rows(x), is_full(root)):
                    parent7[find7(x)] = find7(root)
        # Siblings: candidates of one space and depth reading the same non-candidate value at their own element
        # (e.g. several reductions of one product's result) are evaluated in the same pass.
        shared_input = {}
        for x in space_of:
            R_, D_ = space_of[x]
            if nodes[x].kind == "elementwise" and not is_full(x):
                continue
            for root, layout, shape, idx in loads7[x]:
                if root in space_of or root == x or layouts[root].numel != R_ * D_:
                    continue
                if p7b_local(root, layout, shape, idx, R_, D_, seg_rows(x), True):
                    shared_input.setdefault((root, R_, D_, depth.get(x)), []).append(x)
        for group_ in shared_input.values():
            for x in group_[1:]:
                parent7[find7(x)] = find7(group_[0])
        comps = {}
        for x in space_of:
            comps.setdefault(find7(x), set()).add(x)
        item_order = {node.name: k for k, node in enumerate(items)}
        final = []
        for members in comps.values():
            members = set(members)
            while members:
                changed = False
                ts = {seg_rows(m) for m in members if colt(m)}
                if len(ts) > 1:
                    keep_t = min(ts)
                    members -= {m for m in members if colt(m) and seg_rows(m) != keep_t}
                    continue
                T_ = next(iter(ts)) if ts else None
                R_, D_ = space_of[next(iter(members))]
                if T_ and R_ % T_:
                    members = set()
                    break
                # every intra-group read is local and never of a column result
                for x in sorted(members, key=item_order.get):
                    bad = False
                    for root, layout, shape, idx in loads7[x]:
                        if root == x or root not in members:
                            continue
                        if colt(root) or (nodes[x].kind == "elementwise" and not is_full(x) and is_full(root)) or \
                                not p7b_local(root, layout, shape, idx, R_, D_, T_, is_full(root)):
                            bad = True
                            break
                    if bad:
                        members.discard(x)
                        changed = True
                        break
                if changed:
                    continue
                # convexity: no path member -> non-member(s) -> member
                reached, frontier, seen7 = set(), [], set()
                for m in members:
                    for r in readers.get(m, ()):
                        if r not in members and r not in seen7:
                            seen7.add(r)
                            frontier.append(r)
                while frontier:
                    r = frontier.pop()
                    for r2 in readers.get(r, ()):
                        if r2 in members:
                            reached.add(r2)
                        elif r2 not in seen7:
                            seen7.add(r2)
                            frontier.append(r2)
                if reached:
                    members.discard(max(reached, key=item_order.get))
                    continue
                break
            if any(nodes[m].kind != "elementwise" for m in members):
                final.append(sorted(members, key=item_order.get))
        final.sort(key=lambda ms: item_order[ms[0]])
        for k_, members in enumerate(final):
            gname = f"p7bseg_{k_}"
            if gname in nodes or gname in layouts:
                raise LoweringError(f"P7-B segment name collision: {gname}")
            R_, D_ = space_of[members[0]]
            ts = {seg_rows(m) for m in members if colt(m)}
            T_ = next(iter(ts)) if ts else None
            member_set = set(members)
            external = [entry for m in members for entry in loads7[m] if entry[0] not in member_set]
            ncols = sum(1 for m in members if colt(m))
            nodes[gname] = _Node(name=gname, kind="reduction", shape=(1,), dtype="float32", code="",
                                 extra={"engine": "p7b_segment", "reduction_elements": D_,
                                        "shared_bytes": 4 * 8 * D_ * ncols if ncols else 0},
                                 loads=[], direct=[], target="p7b.segment", order=len(nodes))
            loads_of[gname] = external
            parts_ = 1
            if T_ and ncols and p7b_segment_parts != 1:
                resident = 132 * max(1, int(matrix_residency))
                parts_ = (int(p7b_segment_parts) if p7b_segment_parts else
                          max(1, min(P7B_MAX_PARTS, round(resident / (R_ // T_)), T_ // 8)))
            p7b_groups[gname] = {"members": members, "rows": R_, "extent": D_, "segment_rows": T_,
                                 "segments": R_ // T_ if T_ else None, "parts": parts_}
            if parts_ > 1:
                a64 = lambda n_: -(-n_ // 64) * 64
                nodes[gname].extra["p2_split"] = {"counter_words": a64(R_ // T_),
                                                  "partial_words": a64(ncols * (R_ // T_) * parts_ * D_)}
            for m in members:
                p7b_member_of[m] = gname
                attached_to[m] = gname
                if nodes[m].kind != "elementwise":
                    nodes[m].extra = dict(nodes[m].extra or {}, engine="p7b_member")
            p7b_records.setdefault("segments", []).append(
                {"group": gname, "rows": R_, "extent": D_, "segment_rows": T_, "members": members,
                 "row_reductions": [m for m in members if m in p7b_reds and not colt(m)],
                 "column_reductions": [m for m in members if colt(m)]})
        if p7b_groups:
            last_of = {g: max(item_order[m] for m in info["members"]) for g, info in p7b_groups.items()}
            pending = {g: set(info["members"]) for g, info in p7b_groups.items()}
            blocked = set(p7b_member_of)
            new_items, waiting = [], []

            def free(node):
                return not any(root in blocked for root, _, _, _ in loads_of[node.name])

            def place(node):
                # A group node is placed once every external producer is placed; its members then unblock.
                new_items.append(node)
                blocked.discard(node.name)
                if node.name in pending:
                    blocked.difference_update(pending[node.name])
                progress = True
                while progress:
                    progress = False
                    for w in list(waiting):
                        if free(w):
                            waiting.remove(w)
                            new_items.append(w)
                            blocked.discard(w.name)
                            if w.name in pending:
                                blocked.difference_update(pending[w.name])
                            progress = True
            for node in items:
                if node.name in p7b_member_of:
                    g = p7b_member_of[node.name]
                    if item_order[node.name] == last_of[g]:
                        if free(nodes[g]):
                            place(nodes[g])
                        else:
                            waiting.append(nodes[g])
                            blocked.add(g)
                    continue
                if free(node):
                    place(node)
                else:
                    waiting.append(node)
                    blocked.add(node.name)
            if waiting:
                raise LoweringError("internal: P7-B segment ordering left items waiting: "
                                    + ", ".join(w.name for w in waiting[:4]))
            items = new_items
            order = {node.name: k for k, node in enumerate(items)}
            for g, info in p7b_groups.items():
                for m in info["members"]:
                    order[m] = order[g]
            readers = {}
            for node in items:
                for root, _, _, _ in loads_of[node.name]:
                    readers.setdefault(root, []).append(node.name)

    phase_of, phases = {}, []             # phase: {"heavy": [names], "loops": {numel: [names]}}

    def new_phase():
        phases.append({"heavy": [], "loops": {}})
        return len(phases) - 1

    def p4a_item(h):
        return bool((h.extra or {}).get("pops4a"))

    def p4a_companion(h):
        # P-OPS4-A items join phases of products and ordinary reductions (sequential per block after them).
        # P7-B (p7b_attention_hosts): fused attention items host them too (they run after the attention item).
        return (p4a_item(h) or h.kind == "gemm" or (h.kind == "reduction" and not (h.extra or {}).get("exclusive_phase"))
                or (p7b_attention_hosts and h.kind.startswith("attention_mixed")
                    and not (h.extra or {}).get("exclusive_phase")))

    P6B_SHAREABLE = {"pops3c_row", "pops3c_col", "exact_cta_tree_per_thread_column", "exact_cta_tree_per_thread",
                     "exact_cta_tree_per_warp", None}

    def p6b_shareable(h):
        # Level 1: whole-grid reduction engines; level 2 adds column engines and ordered embedding gradients.
        extra = h.extra or {}
        kinds = ("reduction", "embedding_gradient") if p6b_reduction_share == 2 else ("reduction",)
        engines = P6B_SHAREABLE if p6b_reduction_share == 2 else P6B_SHAREABLE - {"pops3c_col"}
        return (h.kind in kinds and extra.get("engine") in engines
                and not extra.get("exclusive_phase") and not extra.get("pops4a"))

    def p6b_row_col(node, heavy):
        # A row and a column engine in one phase split the grid (P-OPS3-C block ranges); not a sequential pair.
        engines = {(h.extra or {}).get("engine") for h in heavy}
        mine = (node.extra or {}).get("engine")
        return (mine == "pops3c_row" and "pops3c_col" in engines) or (mine == "pops3c_col" and "pops3c_row" in engines)

    def p7b_companion(h):
        """P7-B: a grid-stride heavy item that may run after a phase's matrix tasks (as P-OPS4-A items do)."""
        extra_ = h.extra or {}
        if not p7b_companions or h.kind not in P7B_COMPANION_KINDS or extra_.get("exclusive_phase"):
            return False
        if extra_.get("engine") == "pops3c_row" or p4a_item(h):
            return False
        if h.name in row_chain or h.name in row_chain.values():
            return False
        if h.name in phase_of and phases[phase_of[h.name]].get("rows", {}).get(h.name):
            return False
        return True

    def p7b_attention_pair(phase, node):
        """P7-B: a fused attention item and at most two lower-height products share a phase (attention first)."""
        if not (p7b_attention_products and p7b_priority):
            return False
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        spec_ = lambda h: h.kind == "gemm" and bool((h.extra or {}).get("matrix_spec"))
        att = [h for h in heavy + [node] if h.kind.startswith("attention_mixed") and h.kind != "attention_mixed_bias"]
        gem = [h for h in heavy + [node] if spec_(h)]
        rest = [h for h in heavy + [node] if h not in att and h not in gem and not p4a_item(h)]
        if len(att) != 1 or not (gem or rest) or len(gem) > 3 or any(not p7b_companion(h) for h in rest):
            return False
        top = p7b_height.get(att[0].name, 1)
        return all(p7b_height.get(h.name, 1) + P7B_PRIORITY_MARGIN < top for h in gem + rest)

    def gemm_or_one_other(phase, node):
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        if p7b_attention_pair(phase, node):
            return True
        if p7b_companions:
            spec_ = lambda h: h.kind == "gemm" and bool((h.extra or {}).get("matrix_spec"))
            gemms_ = [h for h in heavy if spec_(h)]
            rest_ = [h for h in heavy if not spec_(h) and not p4a_item(h)]
            if spec_(node) and rest_ and all(p7b_companion(h) for h in rest_) and all(
                    spec_(h) for h in heavy if h.kind == "gemm"):
                wave_ = (p7b_wave_group and sum(int(h.extra.get('tiles', 0)) for h in gemms_)
                         + int(node.extra.get('tiles', 0)) <= int(p7b_wave_group))
                if len(gemms_) < 3 or wave_ or (p7b_priority and gemms_ and p7b_height.get(node.name, 1) >
                                       min(p7b_height.get(h.name, 1) for h in gemms_) + P7B_PRIORITY_MARGIN):
                    return True
            if p7b_companion(node) and gemms_ and all(spec_(h) or p4a_item(h) or p7b_companion(h) for h in heavy):
                return True
        if (p7b_priority and node.kind == "gemm" and (node.extra or {}).get("matrix_spec") and heavy
                and all(h.kind == "gemm" and (h.extra or {}).get("matrix_spec") or p4a_item(h) for h in heavy)):
            gemms_ = [h for h in heavy if h.kind == "gemm"]
            if (p7b_wave_group and sum(int(h.extra.get('tiles', 0)) for h in gemms_)
                    + int(node.extra.get('tiles', 0)) <= int(p7b_wave_group)):
                return True               # P7-B: products whose tiles fit one resident wave share the phase
            if len(gemms_) >= 3 and p7b_height.get(node.name, 1) > min(
                    p7b_height.get(h.name, 1) for h in gemms_) + P7B_PRIORITY_MARGIN:
                return True               # P7-B: a critical product joins a full matrix phase (serial group)
        if (p6b_reduction_share and p6b_shareable(node)
                and all(p6b_shareable(h) or p4a_item(h) for h in heavy) and not p6b_row_col(node, heavy)
                and (p6b_reduction_share == 2 or not any((h.extra or {}).get("engine") == "pops3c_col" for h in heavy))):
            # P6-B: independent whole-grid reductions run as sequential items of one phase (block barrier between
            # items instead of a grid barrier).
            return True
        if p4a_item(node):
            return all(p4a_companion(h) for h in heavy)
        if any(p4a_item(h) for h in heavy) and not p4a_companion(node):
            return False
        heavy = [h for h in heavy if not p4a_item(h)]
        if not heavy:
            return True
        if pops3c and pops3c_share:
            engines = {"pops3c_row", "pops3c_col"}
            mine = (node.extra or {}).get("engine")
            if (mine in engines and len(heavy) == 1 and (heavy[0].extra or {}).get("engine") in engines
                    and heavy[0].extra["engine"] != mine):
                return True
        if not coschedule_matrix:
            return False
        if node.kind == 'matrix_counter_init':
            return all(h.kind == 'matrix_counter_init' for h in heavy)
        if (node.extra or {}).get('exclusive_phase') or any((h.extra or {}).get('exclusive_phase') for h in heavy):
            return False  # C-OPS2-C: items with internal grid barriers run alone in their phase
        if node.extra and node.extra.get('matrix_spec'):
            # P7-B p7b_wave_group: more than three products share a phase while their tiles together fit one wave of
            # the resident grid (the serial group's rotated starts then run them side by side).
            wave = (p7b_wave_group and sum(int(h.extra.get('tiles', 0)) for h in heavy)
                    + int(node.extra.get('tiles', 0)) <= int(p7b_wave_group))
            return (len(heavy) < 3 or wave) and all(h.kind == 'gemm' and h.extra.get('matrix_spec') for h in heavy)
        if any(h.extra.get('matrix_spec') for h in heavy):
            return False
        if node.extra and node.extra.get('engine') in ('popsb_cpasync', 'pops2_fp32'):
            return len(heavy) < 3 and all(h.kind == 'gemm' and h.extra.get('engine') == node.extra['engine']
                                         for h in heavy)
        if any(h.extra.get('engine') in ('popsb_cpasync', 'pops2_fp32') for h in heavy):
            return False
        pinned = lambda h: h.kind != "gemm" or str(h.extra.get("engine", "")).startswith("pinned")
        if not pinned(node) or not all(pinned(h) for h in heavy):
            return False
        others = [h for h in heavy if h.kind != "gemm"]
        if node.kind == "gemm":
            return all(h.kind == "gemm" for h in heavy) or len(others) <= 1
        return not others and all(h.kind == "gemm" for h in heavy)

    def shares_column(owner):
        return any(nodes[n].extra.get("engine") == "pops3c_col" for n in phases[phase_of[owner]]["heavy"])

    row_chain = {}                        # P6-B: chained row engine -> predecessor row engine (same phase)

    def p6b_eval_index(text, env):
        """Value of a generated 32-bit index expression for integer variable values (None if not evaluable)."""
        expression = re.sub(r"\(unsigned\)", "", text)
        expression = re.sub(r"(?<![\w.])(\d+)(?:u|LL|ULL)\b", r"\1", expression).replace("/", "//")
        if re.search(r"[^\w\s()+*%/-]", expression):
            return None
        try:
            return int(eval(expression, {"__builtins__": {}}, dict(env)))
        except Exception:
            return None

    def p6b_row_written(owner, root, address, row):
        """Is storage `address` of `root` written for `row` by the row engine `owner`'s warp (statistic or member)?"""
        n = layouts[root].numel if root in layouts else None
        rows_ = nodes[owner].extra["row_rows"]
        if root == owner:
            return address == row
        if attached_to.get(root) != owner or root not in stored_hint:
            return False
        if n == rows_:
            return address == row
        extent_ = nodes[owner].extra["row_extent"]
        return n == rows_ * extent_ and row * extent_ <= address < (row + 1) * extent_

    def p6b_row_local(owner, root, layout, shape, idx, rows_, extent_, variables):
        """Numeric check over sampled (row, column): the load reads what `owner`'s warp wrote for the same row."""
        text = _index32(layout, shape, idx) if shape is not None else idx
        for r in sorted({0, 1, rows_ // 2, rows_ - 1}):
            for c in sorted({0, 1, extent_ // 2, extent_ - 1}):
                env = {name: value(r, c) for name, value in variables.items()}
                value_ = p6b_eval_index(text, env)
                if value_ is None or not p6b_row_written(owner, root, layout.offset + value_, r):
                    return False
        return True

    stored_hint = set()                   # members whose storage is certain when chaining (read from memory later)

    def p6b_chain_owner(node, q):
        """A one-warp-per-row engine whose phase-q reads are all row-local reads of one row engine of phase q."""
        rows_, extent_ = node.extra["row_rows"], node.extra["row_extent"]
        candidates = [n for n in phases[q]["heavy"] if (nodes[n].extra or {}).get("engine") == "pops3c_row"
                      and nodes[n].extra["row_rows"] == rows_]
        chosen = None
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in phase_of or phase_of[root] != q:
                continue
            owner = root if root in candidates else attached_to.get(root)
            if owner not in candidates or (chosen is not None and owner != chosen):
                return None
            chosen = owner
            if root != owner:
                stored_hint.add(root)
            if not p6b_row_local(owner, root, layout, shape, idx, rows_, extent_,
                                 {"prow": lambda r, c: r, "pcol": lambda r, c: c}):
                stored_hint.discard(root)
                return None
        return chosen

    def row_attach(node):
        """Attach `node` to a row engine's warp (consumer of its statistic, or same-row sibling); phase or None."""
        from . import norm_rules
        n = _numel(node.shape)
        if node.dtype not in ("float32", "bfloat16") or node.name in inplace_source or node.name in absorbed:
            return None
        loads, roots = [], set()
        for root, layout, shape, idx in loads_of[node.name]:
            row = nodes.get(root)
            statistic = None
            if (row is not None and (row.extra or {}).get("engine") == "pops3c_row" and root in phase_of
                    and shape is not None):
                at = norm_rules.normalized_index(_index32(layout, shape, idx))
                rows_, extent_ = row.extra["row_rows"], row.extra["row_extent"]
                if ((n == rows_ * extent_ and at == norm_rules.row_index_text(extent_))
                        or (pops3c_factor and n == rows_ and at == "i")):
                    statistic = root
            register = (attached_to[root] if root in attached_to and root not in p7b_member_of and shape is not None
                        and _same_loop_read(node, root, layout, shape, idx) else None)
            if (register is None and root in attached_to and root not in p7b_member_of and shape is not None
                    and pops3c_factor
                    and _numel(nodes[root].shape) == nodes[attached_to[root]].extra["row_rows"]):
                owner_ = nodes[attached_to[root]]
                if (n == owner_.extra["row_rows"] * owner_.extra["row_extent"]
                        and norm_rules.normalized_index(_index32(layout, shape, idx))
                        == norm_rules.row_index_text(owner_.extra["row_extent"])):
                    register = attached_to[root]   # per-row member, held by every thread of the group
            chained_ok = set()
            if p6b_row_chain and row_chain and root in phase_of:
                # P6-B: a chained engine's members may read its predecessor's row values in the same phase.
                pred = root if root in row_chain.values() else attached_to.get(root)
                for c_, p_ in row_chain.items():
                    if p_ == pred and pred is not None and phase_of[c_] == phase_of[root]:
                        rr, ee = nodes[c_].extra["row_rows"], nodes[c_].extra["row_extent"]
                        per_row = n == rr
                        if per_row or n == rr * ee:
                            hinted = root != pred and root not in stored_hint
                            if hinted:
                                stored_hint.add(root)
                            if p6b_row_local(pred, root, layout, shape, idx, rr, ee,
                                             {"i": (lambda r, c: r) if per_row else (lambda r, c, e=ee: r * e + c)}):
                                chained_ok.add(c_)
                            elif hinted:
                                stored_hint.discard(root)
            loads.append((statistic, register, phase_of[root] + 1 if root in phase_of else 0, chained_ok))
            if shape is not None and layout.numel == n:
                roots.add(root)
        owner = None
        # The latest row engine whose warp can evaluate `node`: every other load is ready before its phase.
        for candidate in sorted({x for s_, r_, _, _ in loads for x in (s_, r_) if x} | {x for *_, ok in loads for x in ok},
                                key=lambda r: -phase_of[r]):
            if all(s_ == candidate or r_ == candidate or ready <= phase_of[candidate] or candidate in ok
                   for s_, r_, ready, ok in loads):
                owner = candidate
                break
        row_member = owner is not None and n == nodes[owner].extra["row_rows"]
        if owner is not None and not row_member and not pops3c_attach_shared and shares_column(owner):
            return None
        if owner is None:
            # Same-row sibling: ready exactly in a row engine's phase and sharing a full-size operand with its leaf.
            earliest_other = max([ready for _, _, ready, _ in loads] + [0])
            for name in phases[earliest_other]["heavy"] if earliest_other < len(phases) else ():
                row = nodes[name]
                if ((row.extra or {}).get("engine") == "pops3c_row" and row.extra["row_rows"] * row.extra["row_extent"] == n
                        and roots & {r for r, _, _, _ in loads_of[name]}):
                    owner = name
                    break
            if owner is None or (not pops3c_attach_shared and shares_column(owner)):
                return None
        p = phase_of[owner]
        phase_of[node.name] = p
        attached_to[node.name] = owner
        row_members.setdefault(owner, []).append(node.name)
        phases[p].setdefault("rows", {})[owner] = row_members[owner]
        return p

    def row_pre_members(node, earliest):
        """P-OPS3-C: same-row producers that alone delay a row engine by one phase are evaluated by its warps."""
        n = node.extra["row_rows"] * node.extra["row_extent"]
        binding = {root for root, _, _, _ in loads_of[node.name] if root in phase_of and phase_of[root] + 1 == earliest}
        chosen = []
        for root in binding:
            producer = nodes.get(root)
            if (producer is None or producer.kind != "elementwise" or root in attached_to or root in inplace_source
                    or root in absorbed or producer.dtype not in ("float32", "bfloat16") or _numel(producer.shape) != n
                    or root not in phases[earliest - 1]["loops"].get(n, ()) or not layouts[root].contiguous
                    or layouts[root].offset):
                return []
            # The producer's own operands come from earlier phases; nothing else in its phase reads it.
            if any(r in phase_of and phase_of[r] >= earliest - 1 for r, _, _, _ in loads_of[root]):
                return []
            if any(r != node.name and phase_of.get(r) == earliest - 1 for r in readers.get(root, ())):
                return []
            # The engine reads it at its own flat element (identity), directly or through inlined producers.
            for r, layout, shape, idx in loads_of[node.name]:
                if r == root and not (layout.contiguous and layout.offset == 0 and layout.numel == n
                                      and (shape is None or _numel(shape) == n)):
                    return []
            chosen.append(root)
        return sorted(chosen, key=lambda r: order[r])

    # P7-B critical-path priority: height = longest chain of items to the end of the update (a group counts once).
    p7b_height = {}
    if p7b_priority and not retain_all:
        item_names = {node.name for node in items}
        for node in items:
            p7b_height.setdefault(node.name, 1)
        for node in reversed(items):
            for root, _, _, _ in loads_of[node.name]:
                producer = p7b_member_of.get(root, root)
                if producer in item_names and producer != node.name:
                    p7b_height[producer] = max(p7b_height.get(producer, 1), p7b_height[node.name] + 1)

    if p7b_list_schedule and p7b_height and not retain_all:
        # P7-B list scheduling: a topological order that takes the most critical ready item first (height, then the
        # original order), so critical items claim their earliest phases before off-critical work fills around them.
        import heapq
        position = {node.name: k for k, node in enumerate(items)}
        by_name = {node.name: node for node in items}
        preds = {node.name: set() for node in items}
        succs = {node.name: set() for node in items}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                producer = p7b_member_of.get(root, root)
                if producer in by_name and producer != node.name:
                    preds[node.name].add(producer)
                    succs[producer].add(node.name)
            if node.name in inplace_source:
                # an in-place committer follows every earlier reader of the old state
                target = inplace_source[node.name]
                for reader in readers.get(target, ()):
                    if reader in by_name and reader != node.name and position[reader] < position[node.name]:
                        preds[node.name].add(reader)
                        succs[reader].add(node.name)
        pending = {n: len(p_) for n, p_ in preds.items()}
        ready = [(-p7b_height.get(n, 1), position[n], n) for n, c in pending.items() if c == 0]
        heapq.heapify(ready)
        ordered = []
        while ready:
            _, _, n = heapq.heappop(ready)
            ordered.append(by_name[n])
            for m_ in succs[n]:
                pending[m_] -= 1
                if pending[m_] == 0:
                    heapq.heappush(ready, (-p7b_height.get(m_, 1), position[m_], m_))
        if len(ordered) == len(items):
            items = ordered
            order = {node.name: k for k, node in enumerate(items)}
            for g_, info_ in p7b_groups.items():
                for m_ in info_["members"]:
                    order[m_] = order[g_]
            p7b_records["list_schedule"] = True

    def p7b_placed_readers(name):
        names_ = [name, *p7b_groups.get(name, {}).get("members", ())]
        return [phase_of[r] for n_ in names_ for r in readers.get(n_, ()) if r in phase_of and r not in names_]

    def p7b_movable(h):
        extra_ = h.extra or {}
        if h.name in row_chain or h.name in row_chain.values() or phases[phase_of[h.name]].get("rows", {}).get(h.name):
            return False
        if extra_.get("exclusive_phase") or extra_.get("engine") == "pops3c_row" or p4a_item(h):
            return False
        if any(root in in_place for root, _, _, _ in loads_of[h.name]):
            return False                  # reads mutable state committed in place later
        return h.kind in ("gemm", "reduction", "embedding_gradient")

    def p7b_bump(phase, node):
        """Place critical product `node` in `phase` by moving lower-height heavy items to later phases."""
        mine = p7b_height.get(node.name, 1)
        heavy_ = [nodes[n] for n in phases[phase]["heavy"]]
        if node.kind != "gemm":
            # P7-B: a critical non-product item (e.g. fused attention) moves every lower-height heavy item it cannot
            # share the phase with (P-OPS4-A items stay when it hosts them).
            if not node.kind.startswith("attention_mixed") or not (p7b_attention_hosts or p7b_attention_bump):
                return False
            # p7b_attention_bump: every lower-height item moves (P-OPS4-A items too), so the attention phase holds the
            # attention item alone and the moved items ride later product phases as companions.
            blockers = list(heavy_) if p7b_attention_bump else [h for h in heavy_ if not p4a_item(h)]
        else:
            blockers = [h for h in heavy_ if not p4a_item(h) and not (h.kind == "gemm" and (h.extra or {}).get("matrix_spec"))]
            gemms_ = [h for h in heavy_ if h.kind == "gemm" and (h.extra or {}).get("matrix_spec")]
            if not (node.extra or {}).get("matrix_spec") or any(not (h.extra or {}).get("matrix_spec") for h in heavy_
                                                               if h.kind == "gemm"):
                return False
            if len(gemms_) >= 3:
                gemms_.sort(key=lambda h: p7b_height.get(h.name, 1))
                blockers += gemms_[:len(gemms_) - 2]
        if not blockers or any(p7b_height.get(h.name, 1) >= mine or not p7b_movable(h) for h in blockers):
            p7b_records.setdefault("_bump_refused", []).append(
                [node.name, phase, [(h.name, p7b_height.get(h.name, 1), p7b_movable(h)) for h in blockers]])
            return False
        targets = {}
        for h in blockers:
            phases[phase]["heavy"].remove(h.name)
        for h in blockers:
            limit = min(p7b_placed_readers(h.name), default=None)
            q = phase + 1
            while True:
                if limit is not None and q >= limit:
                    q = None
                    break
                while q >= len(phases):
                    new_phase()
                if gemm_or_one_other(q, h):
                    break
                q += 1
            if q is None:
                p7b_records.setdefault("_bump_refused", []).append([node.name, phase, h.name, "readers", limit])
                for h2 in blockers:   # undo
                    if h2.name in targets:
                        phases[targets[h2.name]]["heavy"].remove(h2.name)
                    phases[phase]["heavy"].append(h2.name)
                return False
            phases[q]["heavy"].append(h.name)
            targets[h.name] = q
        if not gemm_or_one_other(phase, node):
            for h in blockers:
                phases[targets[h.name]]["heavy"].remove(h.name)
                phases[phase]["heavy"].append(h.name)
            return False
        for h in blockers:
            phase_of[h.name] = targets[h.name]
            for member_ in p7b_groups.get(h.name, {}).get("members", ()):
                phase_of[member_] = targets[h.name]
        p7b_records.setdefault("priority_moves", []).append(
            {"for": node.name, "phase": phase, "moved": {h.name: targets[h.name] for h in blockers}})
        return True

    def p7b_compact():
        """P7-B: after epilogue attachment (a consumer of an epilogue value was placed when that value was still its
        own loop), pull each matrix product to the earliest phase its operands (and its epilogue members' operands)
        allow, then drop emptied phases."""
        absorbed_products = {product for decision in absorbed.values() for product, _ in decision["readers"]}
        # p7b_attention_bump: compact the most critical items first (an off-critical reduction must not take the
        # earliest slot of a critical attention item)
        order_ = (sorted(items, key=lambda n_: -p7b_height.get(n_.name, 1)) if p7b_attention_bump else items)
        for _ in range(4):
            changed = False
            for node in order_:
                spec_ = (node.extra or {}).get("matrix_spec") if node.kind == "gemm" else None
                if node.name in absorbed_products or node.name in deferred or node.kind == "elementwise":
                    continue
                if not spec_ and (node.name in row_chain or node.name in row_chain.values()
                                  or phases[phase_of[node.name]].get("rows", {}).get(node.name)
                                  or (node.extra or {}).get("engine") == "pops3c_row"):
                    continue
                cur = phase_of[node.name]
                members_ = list(epilogues.get(node.name, ()))
                if any(m in inplace_source for m in members_):
                    continue
                own = set(members_) | {node.name}
                earliest = 0
                for name_ in [node.name, *members_]:
                    for root, _, _, _ in loads_of[name_]:
                        if root not in own and root in phase_of:
                            earliest = max(earliest, phase_of[root] + 1)
                for q in range(earliest, cur):
                    if gemm_or_one_other(q, node):
                        phases[cur]["heavy"].remove(node.name)
                        phases[q]["heavy"].append(node.name)
                        phase_of[node.name] = q
                        for member_ in p7b_groups.get(node.name, {}).get("members", ()):
                            phase_of[member_] = q
                        if members_:
                            del phases[cur]["epilogue"][node.name]
                            phases[q].setdefault("epilogue", {})[node.name] = members_
                            for m in members_:
                                phase_of[m] = q
                        p7b_records.setdefault("compacted", []).append([node.name, cur, q])
                        changed = True
                        break
            if not changed:
                break
        keep = [k for k, ph in enumerate(phases) if ph["heavy"] or ph["loops"] or ph.get("epilogue") or ph.get("rows")]
        if len(keep) != len(phases):
            remap = {k: j for j, k in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for name_ in phase_of:
                phase_of[name_] = remap[phase_of[name_]]

    for node in items:
        if pops3c and pops3c_attach and not retain_all and node.kind == "elementwise" and row_attach(node) is not None:
            continue
        earliest = 0
        same_index_roots = set()
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in phase_of:
                continue                    # graph input
            if shape is not None and register_load(node, root, layout, shape, idx):
                earliest = max(earliest, phase_of[root])
                same_index_roots.add(root)
            else:
                earliest = max(earliest, phase_of[root] + 1)
        if node.name in inplace_source:
            target = inplace_source[node.name]
            for reader in readers.get(target, ()):
                if reader == node.name:
                    continue
                lay_ok = all(idx == "i" and lay.contiguous and lay.offset == 0 and nodes[reader].kind == "elementwise"
                             and _numel(nodes[reader].shape) == _numel(node.shape)
                             for root, lay, shape, idx in loads_of[reader] if root == target)
                earliest = max(earliest, phase_of[reader] if lay_ok else phase_of[reader] + 1)
        if node.kind == "elementwise":
            numel = _numel(node.shape)
            # Register producers must share the loop: same phase and element count.
            phase = earliest
            while phase >= len(phases):
                new_phase()
            phases[phase]["loops"].setdefault(numel, []).append(node.name)
            phase_of[node.name] = phase
        else:
            pre = []
            if (pops3c and pops3c_pre and pops3c_attach and not retain_all and earliest > 0
                    and node.extra.get("engine") == "pops3c_row"):
                pre = row_pre_members(node, earliest)
                if pre:
                    earliest -= 1
            chained = None
            if (p6b_row_chain and not pre and not retain_all and earliest > 0
                    and node.extra.get("engine") == "pops3c_row" and node.name not in row_pre):
                chained = p6b_chain_owner(node, earliest - 1)
                # Sequential items of one phase: no products (tile offsets) or exclusive items in the target phase.
                if chained is not None and not any(nodes[h].kind == "gemm" or (nodes[h].extra or {}).get("exclusive_phase")
                                                   for h in phases[earliest - 1]["heavy"]):
                    earliest -= 1
                else:
                    chained = None
            phase = earliest
            while True:
                while phase >= len(phases):
                    new_phase()
                if (chained is not None and phase == earliest) or gemm_or_one_other(phase, node):
                    break
                if p7b_priority and (node.kind == "gemm" or p7b_attention_hosts or p7b_attention_bump) and p7b_bump(phase, node):
                    break
                phase += 1
            phases[phase]["heavy"].append(node.name)
            phase_of[node.name] = phase
            if p7b_priority and node.kind == "gemm":
                p7b_records.setdefault("_gemm_place", {})[node.name] = (earliest, phase, p7b_height.get(node.name))
            for member_ in p7b_groups.get(node.name, {}).get("members", ()):
                phase_of[member_] = phase   # P7-B: segment members are evaluated by the group's engine
            if (p6b_row_chain and chained is None and node.extra.get("engine") == "pops3c_row"
                    and not any(nodes[h].extra.get("engine") == "pops3c_col" for h in phases[phase]["heavy"])):
                # An independent same-rows engine earlier in this phase: same warps, block barrier between items, so
                # this engine's members may read its row values (as for a chained engine).
                shared = [h for h in phases[phase]["heavy"][:-1] if (nodes[h].extra or {}).get("engine") == "pops3c_row"
                          and nodes[h].extra["row_rows"] == node.extra["row_rows"]]
                if shared:
                    chained, earliest = shared[-1], phase
            if chained is not None and phase == earliest:
                row_chain[node.name] = chained
                p6b_records.setdefault("row_chain", []).append({"engine": node.name, "after": chained,
                                                                 "phase": phase})
            if pre and phase == earliest:
                for producer in pre:
                    n_elements = _numel(nodes[producer].shape)
                    loop = phases[phase]["loops"][n_elements]
                    loop.remove(producer)
                    if not loop:
                        del phases[phase]["loops"][n_elements]
                    attached_to[producer] = node.name
                    row_members.setdefault(node.name, []).append(producer)
                    row_pre.setdefault(node.name, set()).add(producer)
                phases[phase].setdefault("rows", {})[node.name] = row_members[node.name]
            for cast, decision in absorbed.items():
                if decision["mode"] == "materialize" and decision["readers"][0][0] == node.name:
                    phase_of[cast] = phase
    commit = len(phases)
    deferred, preload = {}, {}
    if pops2b and pops2b_defer and pops2b_idle and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7b import defer_products, idle_heavy_model
        deferred, preload = defer_products(items=items, loads_of=loads_of, phase_of=phase_of, phases=phases,
                                           nodes=nodes, idle_model=idle_heavy_model(), numel_of=_numel,
                                           inplace_source=inplace_source, dtypes=dtypes)
        commit = len(phases)
    if p9e_defer_attention and not retain_all and target_arch == "sm_90":
        # P9-E: a split-K sink product (every transitive reader elementwise: slice sum, optimizer chain) moves into the
        # next attention-backward phase, on the whole grid before the attention item (p9e_attention_last), whose
        # dynamically claimed units then absorb the product's static tail. Values and arithmetic are unchanged;
        # only the phase changes, and its reader loops move after it as in defer_products.
        from . import matrix_rules_p7b as R7
        readers_ = {}
        for node_ in items:
            for root_, _, _, _ in loads_of[node_.name]:
                readers_.setdefault(root_, set()).add(node_.name)
        order_ = {node_.name: k_ for k_, node_ in enumerate(items)}
        grid_ = R7.TARGET_GRID
        for node_ in sorted(items, key=lambda n_: phase_of[n_.name]):
            spec_ = (node_.extra or {}).get("matrix_spec") if node_.kind == "gemm" else None
            if not spec_ or not spec_.get("split") or node_.name in deferred:
                continue
            start_ = phase_of[node_.name]
            target_ = next((q_ for q_ in range(start_ + 1, min(len(phases), start_ + 1 + int(p9e_defer_attention)))
                            if any(nodes[h_].kind == "attention_mixed_backward" for h_ in phases[q_]["heavy"])), None)
            # Elementwise readers at or before the target follow the product to target + 1; any other reader
            # (e.g. a gradient-norm reduction) must already lie after target + 1.
            closure_, frontier_, ok_ = set(), list(readers_.get(node_.name, ())), bool(readers_.get(node_.name))
            heavy_moves_, blocker_ = [], None
            while frontier_ and ok_ and target_ is not None:
                r_ = frontier_.pop()
                if r_ in closure_ or r_ in heavy_moves_:
                    continue
                if nodes[r_].kind != "elementwise" or phase_of[r_] > target_ + 1:
                    if phase_of[r_] <= target_ + 1:
                        # P9-E: a heavy reduction reader (e.g. a gradient-norm partial) may follow to target + 1
                        # when every one of its readers already lies after target + 1.
                        if (p9e_defer_heavy and nodes[r_].kind != "elementwise" and r_ in phases[phase_of[r_]]["heavy"]
                                and all(phase_of.get(x_, 10 ** 9) > target_ + 2 for x_ in readers_.get(r_, ()))):
                            heavy_moves_.append(r_)
                        else:
                            ok_ = False
                            blocker_ = {"reader": r_, "kind": nodes[r_].kind, "phase": phase_of[r_],
                                        "readers_phases": sorted({phase_of.get(x_, -1) for x_ in readers_.get(r_, ())})}
                    continue
                closure_.add(r_)
                frontier_.extend(readers_.get(r_, ()))
            if ok_ and target_ is not None and any(
                    r_ not in phases[phase_of[r_]]["loops"].get(_numel(nodes[r_].shape), ())
                    for r_ in closure_ if phase_of[r_] <= target_):
                ok_ = False      # a reader placed outside the phase loops (rows, epilogue) cannot follow
            if not ok_ or target_ is None:
                p7b_records.setdefault("defer_attention_skipped", []).append(
                    {"product": node_.name, "closure_ok": ok_, "start": start_, "blocker": blocker_})
                continue
            phases[start_]["heavy"].remove(node_.name)
            phases[target_]["heavy"].insert(0, node_.name)
            phase_of[node_.name] = target_
            while target_ + 2 >= len(phases):
                phases.append({"heavy": [], "loops": {}})
            for r_ in heavy_moves_:   # after the reader loops that follow the product to target + 1
                phases[phase_of[r_]]["heavy"].remove(r_)
                phases[target_ + 2]["heavy"].append(r_)
                phase_of[r_] = target_ + 2
            for r_ in sorted(closure_, key=order_.get):
                if phase_of[r_] <= target_:
                    numel_ = _numel(nodes[r_].shape)
                    loop_ = phases[phase_of[r_]]["loops"][numel_]
                    loop_.remove(r_)
                    if not loop_:
                        del phases[phase_of[r_]]["loops"][numel_]
                    phases[target_ + 1]["loops"].setdefault(numel_, []).append(r_)
                    phases[target_ + 1]["loops"][numel_].sort(key=order_.get)
                    phase_of[r_] = target_ + 1
            deferred[node_.name] = {"phase": target_, "slice": [0, grid_, grid_], "plan_grid": grid_,
                                    "estimate_us": 0.0}
            p7b_records.setdefault("defer_attention", []).append({"product": node_.name, "from": start_,
                                                                  "to": target_})
        keep_ = [k_ for k_, ph_ in enumerate(phases) if ph_["heavy"] or ph_["loops"] or ph_.get("epilogue")]
        if len(keep_) != len(phases):
            remap_ = {old_: new_ for new_, old_ in enumerate(keep_)}
            phases[:] = [phases[k_] for k_ in keep_]
            for name_ in phase_of:
                phase_of[name_] = remap_[phase_of[name_]]
            for info_ in deferred.values():
                info_["phase"] = remap_[info_["phase"]]
        commit = len(phases)
    if place_by_slack and not retain_all:
        idle_heavy = None
        if pops2b and pops2b_idle and target_arch == "sm_90":
            from .pops2b_rules_p7b import idle_heavy_model
            idle_heavy = idle_heavy_model(pops2b_idle_fill)
        _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                        forced, _numel, idle_heavy=idle_heavy, deferred=deferred, preload=preload,
                        fixed=set(attached_to), grid_model=p7b_slack_grid or None)
        commit = len(phases)
    epilogue_of, epilogues, permuted_members = {}, {}, {}
    if pops2b and pops2b_epilogue and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7b import attach_epilogues
        epilogue_of, epilogues, permuted_members = attach_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            dtypes=dtypes, in_place=in_place, inplace_source=inplace_source, register_load=register_load,
            numel_of=_numel, permuted_copies=pops2b_permuted, cost_of=lambda n: total_cost.get(n, 0),
            cost_limit=pops2b_epi_cost, stream_gate=pops2b_epi_stream_gate,
            background=set(deferred) if pops2b_defer_epilogue else set(), fixed=set(attached_to),
            diagnostics=p6b_records.setdefault("epilogue_rejections", []) if p6b_epilogue_diagnostics else None,
            full_grid_cost=p6b_epilogue_full_grid_cost, wg_epilogue=p6b_wg_epilogue,
            small_elements=p7b_small_epilogue)
        commit = len(phases)
    if p7b_priority and not retain_all:
        p7b_compact()
        commit = len(phases)

    def p7b_defer_reduction_phases():
        """P7-B: a phase holding only P-OPS4-A reduction items (+ the scalar loops reading them) is dissolved when every
        item can join a later product phase (as a companion, before its first reader) and every loop can follow its
        producers into an existing later phase before its own readers; emptied phases are dropped."""
        def first_reader(name, own=()):
            return min((phase_of[r] for r in readers.get(name, ()) if r in phase_of and r != name and r not in own),
                       default=len(phases))
        moved_phases = []
        for k in range(len(phases)):
            ph = phases[k]
            own_loops = {m for ms in ph["loops"].values() for m in ms}
            heavy = list(ph["heavy"])
            if (not heavy or not all(p4a_item(nodes[h]) for h in heavy) or ph.get("epilogue") or ph.get("rows")
                    or any(n in inplace_source for ms in ph["loops"].values() for n in ms)):
                continue
            def product_phase(q, h):
                # a later phase with products whose items all accept P-OPS4-A companions (the attention bias item is
                # itself a column-reduction item running beside products)
                return (any(nodes[x].kind == "gemm" and (nodes[x].extra or {}).get("matrix_spec") for x in phases[q]["heavy"])
                        and all(p4a_companion(nodes[x]) or nodes[x].kind == "attention_mixed_bias"
                                for x in phases[q]["heavy"]))
            plan = {}
            for h in heavy:
                lim = first_reader(h, own_loops)
                target = next((q for q in range(k + 1, min(lim, len(phases))) if product_phase(q, h)), None)
                if target is None and lim < len(phases):
                    # the first reader is this item's P-OPS4-A finalize in a later phase: move both (partial into the
                    # first product phase, finalize into a later product phase before its own readers)
                    fin = [r for r in readers.get(h, ()) if r in phase_of and phase_of[r] == lim and p4a_item(nodes[r])
                           and r not in plan]
                    if len(fin) == 1:
                        r = fin[0]
                        for q in range(k + 1, len(phases)):
                            if not product_phase(q, h):
                                continue
                            q2 = next((q2 for q2 in range(max(q + 1, lim), min(first_reader(r), len(phases)))
                                       if q2 == lim or product_phase(q2, nodes[r].name)), None)
                            if q2 is not None:
                                target = q
                                if q2 != lim:
                                    plan[r] = q2
                                break
                if target is None:
                    break
                plan[h] = target
            if not all(h in plan for h in heavy):
                continue
            loop_plan, ok = {}, True
            for numel_, members_ in ph["loops"].items():
                for m in members_:
                    need = k + 1
                    for root, _, _, _ in loads_of[m]:
                        if root in plan:
                            need = max(need, plan[root] + 1)
                        elif root in members_ and root != m:
                            need = max(need, loop_plan.get(root, need))
                    if need >= first_reader(m) or need >= len(phases):
                        ok = False
                        break
                    loop_plan[m] = (numel_, need)
                if not ok:
                    break
            if not ok:
                continue
            # register-chained loop members must stay in one loop: give each loop group the latest member target
            for numel_, members_ in ph["loops"].items():
                q_ = max(loop_plan[m][1] for m in members_)
                if any(q_ >= first_reader(m) for m in members_):
                    ok = False
                for m in members_:
                    loop_plan[m] = (numel_, q_)
            if not ok:
                continue
            for h, q in plan.items():
                phases[phase_of[h]]["heavy"].remove(h)
                phases[q]["heavy"].append(h)
                phase_of[h] = q
            for m, (numel_, q) in loop_plan.items():
                ph["loops"][numel_].remove(m)
                if not ph["loops"][numel_]:
                    del ph["loops"][numel_]
                phases[q]["loops"].setdefault(numel_, []).append(m)
                phase_of[m] = q
            moved_phases.append(k)
        if moved_phases:
            p7b_records.setdefault("deferred_reduction_phases", []).extend(moved_phases)
            keep = [k for k, ph in enumerate(phases) if ph["heavy"] or ph["loops"] or ph.get("epilogue") or ph.get("rows")]
            remap = {k: j for j, k in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for name_ in phase_of:
                phase_of[name_] = remap[phase_of[name_]]
    if p7b_defer_reductions and not retain_all:
        p7b_defer_reduction_phases()
        if p7b_priority:
            p7b_compact()                # a critical item may now take the freed slot
        commit = len(phases)
    p7b_rowepi = {}                       # P7-B: product -> row-only segment group run as its whole-row epilogue
    if p7b_row_epilogue and p7b_groups and not retain_all and target_arch == "sm_90":
        from .matrix_rules_p7b import row_plan
        all_readers = {}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                all_readers.setdefault(root, set()).add(node.name)
        for gname, info in p7b_groups.items():
            if info.get("segment_rows"):
                continue                  # column (segment) reductions span tiles
            R_, D_, members = info["rows"], info["extent"], info["members"]
            mset, q = set(members), phase_of[gname]
            reject = lambda why: p7b_records.setdefault("row_epilogue_rejections", []).append([gname, why])
            if gname not in phases[q]["heavy"]:
                reject("not heavy in its phase")
                continue
            products = {}
            for m in members:
                for root, lay, shape, idx in p7b_render_loads(m, (R_, D_)):
                    nd = nodes.get(root)
                    if root in mset or nd is None or nd.kind != "gemm":
                        continue
                    # the read hits the member's own element (g, c) everywhere (sampled numerically)
                    ident_ = shape is not None and p7b_local(root, lay, shape, idx, R_, D_, None, True)
                    products[root] = products.get(root, True) and ident_
            if not products:
                reject("no product operand")
                continue
            P_ = max(products, key=lambda n_: phase_of[n_])   # the latest product; earlier ones are plain operands
            if not products[P_]:
                reject(f"product {P_} read non-locally")
                continue
            spec_ = (nodes[P_].extra or {}).get("matrix_spec")
            p = phase_of[P_]
            if (not spec_ or P_ in deferred or any(P_ == pr for d_ in absorbed.values() for pr, _ in d_['readers']) or phases[p]["heavy"] != [P_] or p >= q
                    or phases[p].get("epilogue", {}).get(P_) or spec_["m"] != R_ or spec_["n"] != D_
                    or row_plan(spec_) is None or P_ in p7b_rowepi):
                reject(f"product {P_} phase {p}/{q} heavy {phases[p]['heavy']} plan {row_plan(spec_) if spec_ else None}")
                continue
            if any(phase_of.get(root, -1) >= p for m in members for root, _, _, _ in loads_of[m]
                   if root not in mset and root != P_):
                reject("operand not ready before the product's phase")
                continue                  # every other operand exists before the product's phase
            if any(p <= phase_of[r] <= q for m in members for r in all_readers.get(m, ()) if r not in mset):
                reject("member read in the phases between")
                continue                  # member values are read only in later phases (from memory)
            phases[q]["heavy"].remove(gname)
            for name_ in [gname, *members]:
                phase_of[name_] = p
                epilogue_of[name_] = P_
            loads_of[gname] = [e for e in loads_of[gname] if e[0] != P_]
            p7b_rowepi[P_] = gname
            p7b_records.setdefault("row_epilogues", []).append({"product": P_, "group": gname, "from": q, "to": p})
        keep = [k for k, ph in enumerate(phases) if ph["heavy"] or ph["loops"] or ph.get("epilogue") or ph.get("rows")]
        if len(keep) != len(phases):
            remap = {k: j for j, k in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for name_ in phase_of:
                phase_of[name_] = remap[phase_of[name_]]
        commit = len(phases)


    # P9-E: epilogue column sums (port of the P6-A/P7-A rule to this schedule): a full-column FP32 reduction of a
    # value computed in a staged TMA epilogue is accumulated there in 64-row slab partials; the column item leaves
    # its phase, the finalize sums ceil(M / 64) slabs (fixed order), and the summed value is no longer read back.
    colsum_of = {}                        # product -> [(summed epilogue member, partial ident)]
    if p9e_epi_colsum:
        p7b_records["epi_colsum_guard"] = {"retain_all": bool(retain_all), "epilogues": len(epilogue_of),
                                           "staged": bool(p7b_staged_epilogue),
                                           "columns": sum(1 for n in items if ((nodes[n.name].extra or {}).get("p4a_spec")
                                                                                or {}).get("mode") == "column")}
    if p9e_epi_colsum and not retain_all and epilogue_of and p7b_staged_epilogue:
        from .fusion_p7a import _row_index_exact
        from . import pops4a_rules as R4
        colinfo = p4a_state.get("colinfo", {})
        for part_name in [n.name for n in items]:
            pnode = nodes[part_name]
            info = colinfo.get(part_name)
            if info is None or (pnode.extra or {}).get("engine") != "pops4a_column":
                continue
            def _rej(why, part_name=part_name):
                p7b_records.setdefault("epi_colsum_reject", []).append({"partial": part_name, "why": why})
            if part_name in deferred or info["final"] not in nodes:
                _rej(f"deferred {part_name in deferred} final {info['final'] in nodes}")
                continue
            toks = _TOKEN.findall(info["leaves"][0])
            if len(toks) != 1 or info["leaves"][0].strip() != f"@@LOAD{toks[0]}@@":
                _rej("leaf " + str(info["leaves"][0])[:120])
                continue
            kind_, ref_, shape_, pos_ = token_loads[int(toks[0])]
            summed = root_of(ref_)
            product = epilogue_of.get(summed)
            pspec = (nodes[product].extra or {}).get("matrix_spec") if product else None
            lay_ = layouts[ref_]
            E_, N_ = info["E"], info["N"]
            if (kind_ != "storage" or not pspec or not pspec.get("wg") or not pspec.get("tma") or pspec.get("split")
                    or pspec["m"] != E_ or pspec["n"] != N_ or not pspec.get("pair") or pspec["n"] % 8
                    or pspec["batch"] != 1 or lay_.offset or not lay_.contiguous
                    or lay_.numel != E_ * N_ or nodes[summed].dtype != "float32"
                    or product in colsum_of or summed in permuted_members):
                _rej({"kind": kind_, "summed": summed, "product": product,
                      "pspec": {k_: pspec.get(k_) for k_ in ("wg", "tma", "split", "m", "n", "pair", "batch")} if pspec else None,
                      "E": E_, "N": N_, "offset": lay_.offset, "contiguous": lay_.contiguous,
                      "numel": lay_.numel, "dtype": nodes[summed].dtype if summed in nodes else None,
                      "permuted": summed in permuted_members, "taken": product in colsum_of})
                continue
            text_ = re.sub(r"\bo0?\b", "pcol", re.sub(r"\bpr\b", "prow", str(pos_)))
            if not _row_index_exact(text_, E_, N_, element=True):
                _rej("index " + text_[:160])
                continue
            slabs = -(-E_ // 64)
            layouts[part_name] = TensorLayout((slabs * N_,), (1,), part_name)
            nodes[info["final"]].code = R4.column_finalize(
                part=_ident(part_name), out=info["out"], results=N_, plan=dict(info["plan"], tiles=slabs),
                rounded=info["rounded"], scale=info["scale"], rotate=info["rotate"])
            phases[phase_of[part_name]]["heavy"].remove(part_name)
            phase_of[part_name] = phase_of[product]
            items = [n for n in items if n.name != part_name]
            loads_of.pop(part_name, None)
            pnode.extra["epilogue_colsum_of"] = product
            colsum_of.setdefault(product, []).append((summed, _ident(part_name)))
            p7b_records.setdefault("epi_colsum", []).append({"product": product, "summed": summed,
                                                             "partial": part_name, "slabs": slabs})
        if colsum_of:
            keep = [k for k, ph in enumerate(phases) if ph["heavy"] or ph["loops"] or ph.get("epilogue") or ph.get("rows")]
            if len(keep) != len(phases):
                remap = {k: j for j, k in enumerate(keep)}
                phases[:] = [phases[k] for k in keep]
                for name_ in phase_of:
                    phase_of[name_] = remap[phase_of[name_]]
            commit = len(phases)

    if p9e_split_alone and not retain_all:
        # P9-E: a split-K product sharing a phase with an unsplit product of a large output (>= p9e_split_alone
        # elements; its epilogue streams the output) runs in its own phase right after instead: both are operand-
        # bandwidth-bound and contend when co-scheduled. Its readers already lie in later phases.
        q_ = 0
        while q_ < len(phases):
            heavy_ = phases[q_]["heavy"]
            splits_ = [h_ for h_ in heavy_ if nodes[h_].kind == "gemm" and h_ not in deferred
                       and ((nodes[h_].extra or {}).get("matrix_spec") or {}).get("split")]
            big_ = [h_ for h_ in heavy_ if nodes[h_].kind == "gemm" and h_ not in splits_
                    and ((nodes[h_].extra or {}).get("matrix_spec") or {}).get("m", 0)
                    * ((nodes[h_].extra or {}).get("matrix_spec") or {}).get("n", 0) >= int(p9e_split_alone)]
            if splits_ and big_ and len(splits_) < len(heavy_):
                for h_ in splits_:
                    heavy_.remove(h_)
                phases.insert(q_ + 1, {"heavy": list(splits_), "loops": {}})
                for name_ in phase_of:
                    if phase_of[name_] > q_:
                        phase_of[name_] += 1
                for h_ in splits_:
                    phase_of[h_] = q_ + 1
                for info_ in deferred.values():
                    if info_["phase"] > q_:
                        info_["phase"] += 1
                preload = {(k_ + 1 if k_ > q_ else k_): v_ for k_, v_ in preload.items()}
                p7b_records.setdefault("split_alone", []).append({"phase": q_, "products": list(splits_),
                                                                  "with": list(big_)})
                q_ += 2
                continue
            q_ += 1
        commit = len(phases)

    # ------------------------------------------------------------------ storage decisions
    stored = set()                        # elementwise roots needing arena storage
    register_values = {}                  # name -> set of consumers reading it via register
    memory_readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in nodes:
                continue
            if (shape is not None and register_load(node, root, layout, shape, idx)
                    and phase_of[root] == phase_of[node.name]):
                register_values.setdefault(root, set()).add(node.name)
            else:
                memory_readers.setdefault(root, []).append(phase_of[node.name])
    for node in items:
        if node.kind != "elementwise":
            continue
        if (retain_all or node.name in memory_readers or node.name in forced
                or node.name in inplace_source or (row_chain and node.name in stored_hint)):
            stored.add(node.name)
    stored.update(cast for cast, decision in absorbed.items() if decision["mode"] == "materialize")
    for info_ in p7b_groups.values():
        for member_ in info_["members"]:
            if nodes[member_].kind == "elementwise" and (member_ in memory_readers or member_ in forced):
                stored.add(member_)
    elided_products = set()
    for product, members in epilogues.items():
        product_readers = [(node.name, layout, shape, idx) for node in items
                           for root, layout, shape, idx in loads_of[node.name] if root == product]
        size = _numel(layouts[product].shape)
        if product not in forced and all(
                epilogue_of.get(reader) == product and (reader in permuted_members or (
                    idx == "i" and shape is not None and _numel(shape) == size
                    and layout.numel == size and layout.contiguous and layout.offset == 0))
                for reader, layout, shape, idx in product_readers):
            elided_products.add(product)
    row_check = {"row_of": attached_to} if attached_to else {}
    if row_chain:
        row_check = dict(row_check, row_chain=row_chain, row_of=attached_to)
    if epilogue_of:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError,
                        epilogue_of=epilogue_of, permuted=permuted_members, **row_check)
    else:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError, **row_check)
    storage_roots = [n for n in nodes if (nodes[n].kind != "elementwise" or n in stored) and n not in inplace_source
                     and n not in elided_products and n not in p7b_groups]
    if retain_all:
        storage_roots = [n for n in nodes if n not in inplace_source]
        if inline:
            raise LoweringError("internal: retain_all must not inline")
    pre_layout_root = {name: layout.root for name, layout in layouts.items()}
    # In-place sources live in their committed input storage.
    for source, target in inplace_source.items():
        layouts[source] = TensorLayout(layouts[source].shape, layouts[source].strides, target, 0)
    for view in views:
        source = view["source"]
        layout = layouts[view["name"]]
        if layouts[source].root != layout.root:
            layouts[view["name"]] = TensorLayout(layout.shape, layout.strides, layouts[source].root, layout.offset)
    # Recompute view roots for chains (views of views of an in-place source).
    changed = True
    while changed:
        changed = False
        for view in views:
            layout = layouts[view["name"]]
            root = layouts[view["source"]].root
            if layout.root != root:
                layouts[view["name"]] = TensorLayout(layout.shape, layout.strides, root, layout.offset)
                changed = True

    sizes = {name: (layouts[name].numel * _BYTES[dtypes[name]] + 3) // 4 for name in storage_roots}
    classes = {name: ("live", phase_of.get(name, 0), phase_of.get(name, 0)) for name in storage_roots}
    for name, phases_read in memory_readers.items():
        if name in classes:
            _, first, last = classes[name]
            classes[name] = ("live", first, max([last] + phases_read))
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            if root in classes:
                _, first, last = classes[root]
                classes[root] = ("live", first, max(last, phase_of[node.name]))
    for name in outputs + list(writes.values()):
        root = layouts[name].root
        if root in classes:
            _, first, _ = classes[root]
            classes[root] = ("live", first, commit)
    if retain_all:
        classes = {name: ("live", phase_of.get(name, 0), commit) for name in storage_roots}
    offsets, arena_words, _ = arena_assign.assign(storage_roots, sizes, classes, commit)
    bar_offset = None
    if pops2b and pops2b_barrier and not retain_all and target_arch == "sm_90":
        # P-OPS2-B: one 8-byte-aligned arrival counter word after every planned value (never aliased).
        bar_offset = (arena_words + 1) // 2 * 2
        arena_words = bar_offset + 2
    errors = arena_plan.check(storage_roots, sizes, classes, offsets, arena_words)
    if errors:
        raise LoweringError("invalid arena: " + "; ".join(errors))

    # ------------------------------------------------------------------ emission
    has_storage = set(storage_roots) | set(inputs)
    declared = {}
    declarations = []
    local_declarations = {}
    local_pointers = any(node.extra and node.extra.get('engine') in ('popsb_cpasync', 'pops2_fp32')
                         for node in nodes.values())
    # P-OPS2-A split-K scratch beyond the planned arena: counters (zeroed in the prologue, restored to zero
    # by every last arriver) then per-phase partials. Offsets restart in every phase.
    p2_base, p2_counter_words, p2_partial_words = arena_plan.a64(arena_words), 0, 0
    p2_first_split_phase = None
    for name, layout in layouts.items():
        if layout.root not in has_storage:
            continue
        pointer = f"args[{inputs.index(layout.root)}]" if layout.root in inputs else f"arena + {offsets[layout.root]}LL"
        if dtypes[name] != "float32":
            pointer = f"reinterpret_cast<{_CTYPES[dtypes[name]]}*>({pointer})"
        declaration = f"{_CTYPES[dtypes[name]]}* {_ident(name)} = {pointer}" + (f" + {layout.offset}LL" if layout.offset else "") + ";"
        declarations.append(declaration)
        local_declarations[_ident(name)] = declaration.replace('args[', 'latch_pa[').replace('arena +', 'latch_pr +')
        declared[name] = True

    def scoped(code):
        if not local_pointers:
            return code
        used = sorted(set(re.findall(r"\bv_\w+\b", code)) & set(local_declarations))
        return ("float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n"
                + "\n".join(local_declarations[name] for name in used) + "\n" + code)

    if local_pointers:
        declarations = []
        helper_parts.append("#ifndef LATCH_OPQ_DEFINED\n#define LATCH_OPQ_DEFINED\n"
                            "template <class T> __device__ __forceinline__ T* latch_opq(T* p) { asm volatile(\"\" : \"+l\"(p)); return p; }\n#endif")

    def memory_loader(ref, shape, idx):
        if ref not in declared:
            raise LoweringError(f"internal: {ref} read from memory without storage")
        expression = f"{_ident(ref)}[{_index32(layouts[ref], shape, idx)}]" if shape is not None else f"{_ident(ref)}[{idx}]"
        return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression

    def heavy_code(node):
        loader = memory_loader
        if node.name in row_pre:
            # P-OPS3-C: the row engine's leaf evaluates its pre-member producers' expressions (stored by its warps).
            def loader(ref, shape, idx, owned=row_pre[node.name]):
                root = layouts[ref].root
                if root not in owned:
                    return memory_loader(ref, shape, idx)
                index = f"(unsigned)({idx})" if shape is None else _index32(layouts[ref], shape, idx)
                producer = nodes[root]
                inner = producer.render(index, lambda arg, i2: operand(arg, producer.shape, i2, memory_loader))
                return (f"__bfloat162float({ctype_round(inner, 'bfloat16')})" if producer.dtype == "bfloat16"
                        else inner)

        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            if kind == "index":
                return expand(ref, shape, idx, loader)
            return expand_storage(ref, idx, loader)
        code = _TOKEN.sub(substitute, node.code)
        if getattr(node, "extra", {}).get("p6b_index32"):   # outlined P-OPS4-A leaves are plain namespaces
            code = p6b_index32_code(code)
        return code

    vector_helpers_needed = []

    def p6b_loop_components(members, k):
        """Independent register chains of one elementwise loop (members reading each other's registers stay
        together, in their original order). Separate loops of one phase cannot read each other: same-phase reads
        are register reads (checked by _check_schedule)."""
        member_set = set(members)
        parent = {m: m for m in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for m in members:
            for root, layout, shape, idx in loads_of.get(m, ()):
                if (root in member_set and shape is not None and register_load(nodes[m], root, layout, shape, idx)
                        and phase_of.get(root) == k):
                    parent[find(m)] = find(root)
        groups = {}
        for m in members:
            groups.setdefault(find(m), []).append(m)
        return list(groups.values())

    def p6b_element_functor(numel, members):
        """Element functor (unsigned i) computing and storing one loop component at flat index i, with the same
        per-element expressions as loop_code (registers for same-index member reads, memory otherwise)."""
        member_set = set(members)

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            same_index = idx == "i" or (p6b_scalar_chain and numel == 1 and idx in ("0u", "0"))
            identity = (same_index and shape is not None and _numel(shape) == numel
                        and layouts[ref].numel == numel and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin in member_set and _numel(nodes[origin].shape) == numel:
                value = f"r_{origin}"
                return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
            return memory_loader(ref, shape, idx)
        lines = []
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
            lines.append(f"const {_CTYPES[node.dtype]} r_{name} = {ctype_round(expression, node.dtype)};")
        if p6b_cse:
            from .norm_rules import cse_lines
            lines = cse_lines(lines, "cse_m")
        if p6b_inline_div and numel >= P6B_INLINE_DIV_MIN:
            lines = [line.replace("latch_fdiv(", "latch_fdiv_inline(") for line in lines]
            if P6B_INLINE_DIV not in helper_parts:
                helper_parts.append(P6B_INLINE_DIV)
        lines += [f"{_ident(n)}[i] = r_{n};" for n in members if n in stored]
        return canonical_functor("void", "unsigned i", "\n    ".join(lines))

    def p6b_chunk_functor(numel, members, force_vector=False):
        """P6-B: vector element functor f(base, gs, nq) evaluating quads base + u gs (u < nq <= 4) of one loop
        component with every quad's loads/compute before its stores (the loop's own vector body, u = 4, bounds
        check replaced by the caller's quad count). Returns (construction, type name) or None."""
        # P7-B p7b_chunk_live: quads per thread per chunk from the functor's live values (casts 16, AdamW 4)
        U = (max(P6B_CHUNK_U, min(16, int(p7b_chunk_live) // (4 * max(1, len(members))))) if p7b_chunk_live
             else P6B_CHUNK_U)
        piece = loop_code(numel, members, 0, pieces="chunk", force_u=U, force_vector=force_vector)
        if not piece["vector"]:
            return None
        check = f"if (q < {_u(piece['count'])}) {{"
        body, store = piece["body"], piece["store"]
        if check not in body or (store and check not in store):
            return None
        body = body.replace(check, "if ((unsigned)u < latch_nq) {")
        store = store.replace(check, "if ((unsigned)u < latch_nq) {")
        text = (f"{piece['decls']}\n#pragma unroll\n    for (int u = 0; u < {U}; ++u) {{\n" + body + "    }\n"
                + store)
        if _u(piece["count"]) in text:
            return None                   # a size-dependent expression: keep the component's own loop
        functor = canonical_functor("void", "unsigned base, unsigned gs, unsigned latch_nq", text)
        return functor, functor.split("{", 1)[0], U

    def p6b_chunk_multi(entries):
        """P6-B: loop components sharing one vector element functor type become one chunked loop: chunks of
        P6B_CHUNK_U x 256 quads of one component each, assigned round-robin to blocks; per chunk the component
        is found once (block-uniform) and each thread evaluates its quads with one functor call. Same per-element
        expressions and roundings as the separate loops; components of one phase are independent."""
        typed, leftover = {}, []
        small = []
        for numel, members in entries:
            made = p6b_chunk_functor(numel, members) if numel >= VECTOR_MIN_ELEMENTS and numel % 4 == 0 else None
            if made is None:
                if p8e_chunk_small and 4 <= numel < VECTOR_MIN_ELEMENTS and numel % 4 == 0:
                    small.append((numel, members))
                else:
                    leftover.append((numel, members))
                continue
            typed.setdefault((made[1], made[2]), []).append((numel, members, made[0]))
        for numel, members in small:
            # P8-E: a sub-vector component (e.g. a bias's optimizer update) joins an existing chunked group of the
            # same vector element functor as one more (partial) chunk instead of a separate batched pass.
            made = p6b_chunk_functor(numel, members, force_vector=True)
            if made is None or (made[1], made[2]) not in typed or len(typed[(made[1], made[2])]) < P6B_MULTI_MIN:
                leftover.append((numel, members))
                continue
            typed[(made[1], made[2])].append((numel, members, made[0]))
        parts = []
        for (type_name, U), group in typed.items():
            chunk = U * 256
            if len(group) < P6B_MULTI_MIN:
                leftover.extend((numel, members) for numel, members, _ in group)
                continue
            ends, total = [], 0
            for numel, _, _ in group:
                total += -(-(numel // 4) // chunk)
                ends.append(total)
            fill = "".join(f"    latch_mt_f[{j}] = {functor};\n" for j, (_, _, functor) in enumerate(group))
            fill += "".join(f"    latch_mt_end[{j}] = {e}u; latch_mt_nq[{j}] = {_u(numel // 4)};\n"
                            for j, (e, (numel, _, _)) in enumerate(zip(ends, group)))
            parts.append(
                f"{{ // P6-B chunked multi-tensor loop: {len(group)} components, {total} chunks of {chunk} quads\n"
                f"  {type_name}* latch_mt_f = reinterpret_cast<{type_name}*>(smem);\n"
                f"  unsigned* latch_mt_end = reinterpret_cast<unsigned*>(latch_mt_f + {len(group)});\n"
                f"  unsigned* latch_mt_nq = latch_mt_end + {len(group)};\n"
                f"  __syncthreads();\n  if (threadIdx.x == 0) {{\n{fill}  }}\n  __syncthreads();\n"
                f"  for (unsigned c = blockIdx.x; c < {_u(total)}; c += gridDim.x) {{\n"
                f"    unsigned lo = 0u, hi = {len(group) - 1}u;\n"
                f"    while (lo < hi) {{ const unsigned mid = (lo + hi) >> 1; "
                f"if (c < latch_mt_end[mid]) hi = mid; else lo = mid + 1u; }}\n"
                f"    const unsigned q0 = (c - (lo ? latch_mt_end[lo - 1u] : 0u)) * {chunk}u;\n"
                f"    const unsigned rem = latch_mt_nq[lo] - q0;  // quads left in the component (blockDim = 256)\n"
                f"    const unsigned nq = rem > threadIdx.x ? min({U}u, (rem - threadIdx.x + 255u) / 256u) : 0u;\n"
                f"    const {type_name} f = latch_mt_f[lo];\n"
                f"    f(q0 + threadIdx.x, 256u, nq);\n"
                f"  }}\n  __syncthreads();\n}}")
            p6b_records.setdefault("chunk_multi", []).append({"components": len(group), "chunks": total})
        return parts, leftover

    def p6b_multi_tensor(entries, exclude, allow_scalar=False):
        """One grid-stride loop over the concatenated elements of loop components sharing one element functor type.
        entries: [(numel, members)]. Returns (code, leftover entries). P8-E allow_scalar: one-element components
        (per-tensor scalar chains) take part too."""
        typed = {}
        for numel, members in entries:
            if numel < (1 if allow_scalar else 2):
                typed.setdefault(None, []).append((numel, members, None))
                continue
            functor = p6b_element_functor(numel, members)
            typed.setdefault(functor.split("{", 1)[0], []).append((numel, members, functor))
        parts, leftover = [], []
        for type_name, group in typed.items():
            if type_name is None or len(group) < P6B_MULTI_MIN:
                leftover.extend((numel, members) for numel, members, _ in group)
                continue
            ends, total = [], 0
            for numel, _, _ in group:
                total += numel
                ends.append(total)
            if total >= (1 << 31):
                leftover.extend((numel, members) for numel, members, _ in group)
                continue
            table = ", ".join(functor for _, _, functor in group)
            if exclude:
                head = (f"  const unsigned skip = 2u * {exclude}u <= gridDim.x ? {exclude}u : 0u;\n"
                        f"  const unsigned vb = (blockIdx.x + gridDim.x - skip) % gridDim.x;\n"
                        f"  const unsigned gs = (gridDim.x - skip) * blockDim.x;\n"
                        f"  if (vb < gridDim.x - skip)\n")
                start = "vb"
            else:
                head = "  const unsigned gs = gridDim.x * blockDim.x;\n"
                start = "blockIdx.x"
            fill = "".join(f"    latch_mt_f[{j}] = {functor};\n" for j, (_, _, functor) in enumerate(group))
            fill += "".join(f"    latch_mt_end[{j}] = {e}u;\n" for j, e in enumerate(ends))
            parts.append(
                f"{{ // P6-B multi-tensor loop: {len(group)} components, {total} elements\n"
                f"  // tensor table in shared memory (one fill per block; same element functors as separate loops)\n"
                f"  {type_name}* latch_mt_f = reinterpret_cast<{type_name}*>(smem);\n"
                f"  unsigned* latch_mt_end = reinterpret_cast<unsigned*>(latch_mt_f + {len(group)});\n"
                f"  __syncthreads();\n  if (threadIdx.x == 0) {{\n{fill}  }}\n  __syncthreads();\n"
                + "  {\n" + head +
                f"  for (unsigned g = {start} * blockDim.x + threadIdx.x; g < {_u(total)}; g += gs) {{\n"
                f"    unsigned lo = 0u, hi = {len(group) - 1}u;\n"
                f"    while (lo < hi) {{ const unsigned mid = (lo + hi) >> 1; "
                f"if (g < latch_mt_end[mid]) hi = mid; else lo = mid + 1u; }}\n"
                f"    const unsigned i = g - (lo ? latch_mt_end[lo - 1u] : 0u);\n"
                f"    latch_mt_f[lo](i);\n"
                f"  }}\n  }}\n  __syncthreads();\n}}")
            p6b_records.setdefault("multi_tensor", []).append({"components": len(group), "elements": total})
        return parts, leftover

    def loop_code(numel, members, exclude, rotate=0, pieces=False, force_u=None, force_vector=False):
        if numel >= (1 << 31):
            raise LoweringError("fused elementwise loops require fewer than 2^31 elements")
        loop_regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        vector = floating and numel % 4 == 0 and (numel >= VECTOR_MIN_ELEMENTS or force_vector)
        vector_refs = {}

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            # The operand's own element count must equal the loop's: a broadcast
            # operand (e.g. a [512] weight in a [256,512] loop) is never an identity read.
            same_index = idx == "i" or (p6b_scalar_chain and numel == 1 and idx in ("0u", "0"))
            identity = (same_index and shape is not None and _numel(shape) == numel
                        and layouts[ref].numel == numel and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin in member_set and _numel(nodes[origin].shape) == numel:
                expression = f"{loop_regs[origin]}[v]"
                return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
            if vector and identity and ref in declared and dtypes[ref] in ("float32", "bfloat16"):
                vector_refs.setdefault(ref, f"in_{len(vector_refs)}")
                expression = f"{vector_refs[ref]}[j]"
                return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        compute, stored_members = [], [n for n in members if n in stored]
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
            if outline_ops or p6b_cse:
                from .norm_rules import PURE_CALLS, PURE_CALLS_P6B, cse_calls
                prelude, expression = cse_calls(expression, f"c_{name}", PURE_CALLS_P6B if p6b_cse else PURE_CALLS)
                compute.extend(prelude)
            compute.append(f"{loop_regs[name]}[v] = {ctype_round(expression, node.dtype)};")
        if p6b_cse:
            from .norm_rules import cse_lines
            compute = cse_lines(compute, "cse_l")
        if p6b_inline_div and numel >= P6B_INLINE_DIV_MIN:
            compute = [line.replace("latch_fdiv(", "latch_fdiv_inline(") for line in compute]
            if P6B_INLINE_DIV not in helper_parts:
                helper_parts.append(P6B_INLINE_DIV)
        lanes = 4 if vector else 1
        count = numel // lanes
        u = 1
        per_thread = -(-count // RESIDENT_THREADS)
        cap = max(1, LIVE_VALUE_CAP // max(1, lanes * len(members)))
        while u * 2 <= min(per_thread, UNROLL_MAX, cap):
            u *= 2
        if pieces == "all":
            u = 1                         # P6-B: batched loops overlap across loops instead of unrolling one
        if force_u:
            u = int(force_u)
        declarations_local = " ".join(f"{_CTYPES[nodes[n].dtype]} {loop_regs[n]}[{u * lanes}];" for n in members)
        if exclude:
            # Blocks [0, exclude) run the phase's matrix tasks; when they use at
            # most half the grid, the loop is distributed over the idle blocks.
            head = (f"  const unsigned skip = 2u * {exclude}u <= gridDim.x ? {exclude}u : 0u;\n"
                    f"  const unsigned vb = (blockIdx.x + gridDim.x - skip) % gridDim.x;\n"
                    f"  const unsigned gs = (gridDim.x - skip) * blockDim.x;\n"
                    f"  if (vb < gridDim.x - skip)\n")
            start_block = "vb"
            if rotate:
                # P6-B: rotated start within the participating blocks.
                start_block = (f"((vb + (gridDim.x - skip) - ({_u(rotate)} % (gridDim.x - skip))) "
                               f"% (gridDim.x - skip))")
        elif rotate:
            # P6-B: this loop starts at block `rotate` (cumulative work of the phase's earlier loops), so loops of
            # one phase spread over the grid instead of all starting at block 0.
            head = "  const unsigned gs = gridDim.x * blockDim.x;\n"
            start_block = f"((blockIdx.x + gridDim.x - ({_u(rotate)} % gridDim.x)) % gridDim.x)"
        else:
            head = "  const unsigned gs = gridDim.x * blockDim.x;\n"
            start_block = "blockIdx.x"
        if vector:
            if not vector_helpers_needed:
                vector_helpers_needed.append(True)
            loads = "".join(f"        {_CTYPES[dtypes[ref]]} {reg}[4]; latch_ld4({_ident(ref)} + 4u * q, {reg});\n"
                            for ref, reg in vector_refs.items())
            body = (f"      const unsigned q = base + (unsigned)u * gs;\n      if (q < {_u(count)}) {{\n{loads}"
                    f"#pragma unroll\n        for (int j = 0; j < 4; ++j) {{\n"
                    f"          const unsigned i = 4u * q + (unsigned)j; const int v = u * 4 + j;\n          "
                    + "\n          ".join(compute) + "\n        }\n      }\n")
            stores = "".join(f"        latch_st4({_ident(n)} + 4u * q, {loop_regs[n]}[u*4], {loop_regs[n]}[u*4+1], "
                             f"{loop_regs[n]}[u*4+2], {loop_regs[n]}[u*4+3]);\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned q = base + (unsigned)u * gs;\n"
                           f"      if (q < {_u(count)}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        else:
            body = (f"      const unsigned i = base + (unsigned)u * gs; const int v = u;\n"
                    f"      if (i < {_u(count)}) {{\n        " + "\n        ".join(compute) + "\n      }\n")
            stores = "".join(f"        {_ident(n)}[i] = {loop_regs[n]}[u];\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned i = base + (unsigned)u * gs;\n"
                           f"      if (i < {_u(count)}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        if pieces == "chunk":
            # P6-B: parts for a chunked multi-tensor element functor (p6b_chunk_multi).
            return {"numel": numel, "members": members, "vector": vector, "count": count, "u": u,
                    "decls": declarations_local, "body": body, "store": store_block}
        if pieces and u == 1:
            # P6-B: parts for a load-batched loop group (p6b_batch_code).
            return {"numel": numel, "members": members, "vector": vector, "head": head, "start": start_block,
                    "count": count, "decls": declarations_local, "body": body, "store": store_block,
                    "live": lanes * len(members)}
        return (f"{{ // {numel} elements{' (x4 vector)' if vector else ''}: {', '.join(members)}\n" + head +
                f"  for (unsigned base = {start_block} * blockDim.x + threadIdx.x; base < {_u(count)}; base += {u}u * gs) {{\n"
                f"    {declarations_local}\n#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n" + body + "    }\n"
                + store_block + "  }\n}")

    def loop_code_unrolled(numel, members):
        """Would the loop unroll (more than one iteration per resident thread)?"""
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        lanes = 4 if floating and numel % 4 == 0 and numel >= VECTOR_MIN_ELEMENTS else 1
        return -(-(numel // lanes) // RESIDENT_THREADS) > 1

    def p7b_prefetch_code(group):
        """P7-B: L2 prefetch of a batched pass's identity operands at this thread's own elements (one prefetch per
        128-byte line: lanes whose element offset is a multiple of the line's element count)."""
        lines = []
        for j, g in enumerate(group):
            vec_refs = re.findall(r"latch_ld4\((v_\w+) \+ 4u \* q,", g["body"])
            scalar_refs = [] if g["vector"] else re.findall(r"\b(v_\w+)\[i\]", g["body"])
            stmts = []
            for ref in dict.fromkeys(vec_refs + scalar_refs):
                dtype = ident_dtype.get(ref, "float32")
                elems = (4 if g["vector"] else 1) * _BYTES.get(dtype, 4)
                per_line = max(1, 128 // elems)
                stmts.append(f"if ((latch_q{j} % {per_line}u) == 0u) asm volatile(\"prefetch.global.L2 [%0];\" :: "
                             f"\"l\"({ref} + {4 if g['vector'] else 1}u * latch_q{j}));")
            if stmts:
                lines.append((j, g, stmts))
        if not lines:
            return ""
        starts = "".join(f"  const unsigned latch_p{j} = {g['start']} * blockDim.x + threadIdx.x;\n" for j, g, _ in lines)
        cond = " || ".join(f"latch_p{j} + d < {_u(g['count'])}" for j, g, _ in lines)
        body = "".join(f"    {{ const unsigned latch_q{j} = latch_p{j} + d; if (latch_q{j} < {_u(g['count'])}) {{ "
                       + " ".join(st) + " } }\n" for j, g, st in lines)
        return (f"  {{ // P7-B: L2 prefetch of the next pass's operands\n{starts}"
                f"  for (unsigned d = 0u; {cond}; d += gs) {{\n{body}  }}\n  }}\n")

    def p6b_batch_code(group, prefetch=None):
        """P6-B: one grid-stride pass over several same-phase loops whose threads run at most a few iterations.
        Every loop's loads/compute precede every loop's stores, so a thread's independent loads overlap instead of
        each loop paying its own memory round trip. Same element expressions and roundings as separate loops;
        same-phase loops never read each other's outputs (they would race across threads otherwise)."""
        pre = p7b_prefetch_code(prefetch) if prefetch and prefetch[0]["head"] == group[0]["head"] else ""
        if pre and "if (vb <" in group[0]["head"]:
            pre = ""                          # excluded-block heads guard the loop statement only
        if len(group) == 1:
            g = group[0]
            return (f"{{ // {g['numel']} elements{' (x4 vector)' if g['vector'] else ''}: {', '.join(g['members'])}\n"
                    + g["head"] + pre + f"  for (unsigned base = {g['start']} * blockDim.x + threadIdx.x; base < "
                    f"{_u(g['count'])}; base += 1u * gs) {{\n    {g['decls']}\n#pragma unroll\n    for (int u = 0; u < 1;"
                    f" ++u) {{\n" + g["body"] + "    }\n" + g["store"] + "  }\n}")
        starts = "".join(f"  const unsigned latch_s{j} = {g['start']} * blockDim.x + threadIdx.x;\n"
                         for j, g in enumerate(group))
        cond = " || ".join(f"latch_s{j} + d < {_u(g['count'])}" for j, g in enumerate(group))
        decls = "\n    ".join(g["decls"] for g in group)
        bodies = "".join(f"    {{ const unsigned base = latch_s{j} + d; const int u = 0;\n{g['body']}    }}\n"
                         for j, g in enumerate(group))
        stores = "".join(f"    {{ const unsigned base = latch_s{j} + d;\n{g['store']}    }}\n"
                         for j, g in enumerate(group) if g["store"])
        label = "; ".join(f"{g['numel']}{' x4' if g['vector'] else ''}: {', '.join(g['members'])}" for g in group)
        p6b_records.setdefault("loop_batch", []).append([g["numel"] for g in group])
        return (f"{{ // P6-B batched loops ({len(group)}): {label}\n" + group[0]["head"] + "  {\n" + pre + starts +
                f"  for (unsigned d = 0u; {cond}; d += gs) {{\n    {decls}\n" + bodies + stores + "  }\n  }\n}")

    def row_members_code(owner, members):
        """P-OPS3-C: attached members evaluated by the owning warp group (row-shaped ones once per row)."""
        node = nodes[owner]
        per_row = [m for m in members if _numel(nodes[m].shape) == node.extra["row_rows"]]
        members = [m for m in members if m not in per_row]
        head = ""
        if per_row:
            value = ctype_round("second" + node.extra["row_scale"], dtypes[owner])
            statistic = f"__bfloat162float({value})" if dtypes[owner] == "bfloat16" else value
            row_regs = {m: f"q_{m}" for m in per_row}

            def row_loader(ref, shape, idx):
                origin = layouts[ref].root
                if origin == owner:
                    return "latch_rowq"
                if (origin in row_regs and idx == "g" and shape is not None and layouts[ref].contiguous
                        and layouts[ref].offset == 0 and layouts[ref].numel == _numel(shape)):
                    reg = row_regs[origin]
                    return f"__bfloat162float({reg})" if dtypes[origin] == "bfloat16" else reg
                return memory_loader(ref, shape, idx)
            lines = []
            for m in per_row:
                member = nodes[m]
                expression = member.render("g", lambda arg, i2, member=member: operand(arg, member.shape, i2, row_loader))
                lines.append(f"const {_CTYPES[member.dtype]} {row_regs[m]} = {ctype_round(expression, member.dtype)};")
            stores_ = [f"{_ident(m)}[g] = {row_regs[m]};" for m in per_row if m in stored]
            shared_use = any(layouts[ref].root in row_regs for m in members
                             for root, layout, shape, idx in loads_of[m] for ref in [root] if root in row_regs)
            if shared_use:
                # Element members read these values: every thread of the group evaluates them; stored once.
                head = (f"        // per row (every thread of the group; stored once): {', '.join(per_row)}\n"
                        f"        const float latch_rowq = {statistic};\n"
                        + "".join(f"        {l}\n" for l in lines)
                        + (f"        if (latch_w0 == 0u && lane == 0u) {{ " + " ".join(stores_) + " }\n"
                           if stores_ else ""))
            else:
                head = (f"        if (latch_w0 == 0u && lane == 0u) {{  // per row: {', '.join(per_row)}\n"
                        f"          const float latch_rowq = {statistic};\n          "
                        + "\n          ".join(lines + stores_) + "\n        }\n")
        if not members:
            return head
        from . import norm_rules
        row_regs_all = {m: f"q_{m}" for m in per_row}
        extent, rows = node.extra["row_extent"], node.extra["row_rows"]
        numel = rows * extent
        regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        value = ctype_round("second" + node.extra["row_scale"], dtypes[owner])
        statistic = f"__bfloat162float({value})" if dtypes[owner] == "bfloat16" else value

        def loader(ref, shape, idx):
            origin = layouts[ref].root
            if origin == owner:
                return "latch_rowv"
            identity = (idx == "i" and shape is not None and _numel(shape) == numel
                        and layouts[ref].numel == numel and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin in member_set:
                value = f"{regs[origin]}[e]"
                return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
            if origin in row_regs_all and shape is not None and norm_rules.normalized_index(
                    _index32(layouts[ref], shape, idx)) == norm_rules.row_index_text(extent):
                reg = row_regs_all[origin]
                return f"__bfloat162float({reg})" if dtypes[origin] == "bfloat16" else reg
            return memory_loader(ref, shape, idx)
        leaves = -(-extent // 256)
        compute, stores = [], []
        row_of_i = re.compile(r"\(i / " + str(extent) + r"u\)")
        for name in members:
            member = nodes[name]
            expression = member.render("i", lambda arg, i2, member=member: operand(arg, member.shape, i2, loader))
            # i / extent == g for every element of the warp's row: row-uniform loads and arithmetic become invariant.
            expression = row_of_i.sub("g", expression)
            if outline_ops:
                from .norm_rules import cse_calls
                prelude, expression = cse_calls(expression, f"c_{name}")
                compute.extend(prelude)
            compute.append(f"{regs[name]}[e] = {ctype_round(expression, member.dtype)};")
            if name in stored:
                stores.append(f"{_ident(name)}[i] = {regs[name]}[e];")
        # Every load of the warp's elements precedes every store (no aliasing reloads of row-uniform operands).
        arrays = " ".join(f"{_CTYPES[nodes[n].dtype]} {regs[n]}[latch_nw * {leaves}u];" for n in members)

        def element_loop(body):
            return (f"#pragma unroll\n        for (unsigned v = 0; v < latch_nw; ++v) {{\n#pragma unroll\n"
                    f"          for (unsigned j = 0; j < {leaves}u; ++j) {{\n"
                    f"            const unsigned c = 32u * (latch_w0 + v) + lane + 256u * j, e = v * {leaves}u + j;\n"
                    f"            if (c < {_u(extent)}) {{\n              const unsigned i = g * {_u(extent)} + c;\n              "
                    + "\n              ".join(body) + "\n            }\n          }\n        }\n")
        return (head + f"        const float latch_rowv = {statistic};  // attached: {', '.join(members)}\n        {arrays}\n"
                + element_loop(compute) + (element_loop(stores) if stores else ""))

    def p7b_row_index(text, D):
        """Inside a segment row, i = g * D + c with c < D: i / D == g, i % D == c and i / (k D) == g / k (row-uniform,
        evaluated once per row instead of per element). Exact integer identities."""
        text = text.replace("(unsigned)(i)", "i")
        previous = None
        while previous != text:            # "(i)" not preceded by an identifier (never a call argument)
            previous, text = text, re.sub(r"(?<![\w\]])\(i\)", "i", text)
        text = re.sub(r"\(i / " + str(D) + r"u\)", "g", text)
        text = re.sub(r"\(i % " + str(D) + r"u\)", "c", text)

        def quotient(match):
            x = int(match.group(1))
            return f"(g / {x // D}u)" if x % D == 0 and x > D else match.group(0)
        text = re.sub(r"\(i / (\d+)u\)", quotient, text)
        # ((E * D + c) / D) == E and ((E * D + c) % D) == c for row-uniform E (c < D)
        uniform = r"(\(g / \d+u\)|g|s|t)"
        text = re.sub(r"\(\(" + uniform + r" \* " + str(D) + r"u \+ c\) / " + str(D) + r"u\)", r"\1", text)
        text = re.sub(r"\(\(" + uniform + r" \* " + str(D) + r"u \+ c\) % " + str(D) + r"u\)", "c", text)
        return text

    def p7b_segment_code(gname, tile_product=None):
        """P7-B segment engine for one group (see the group formation rule). tile_product: row-only group run as
        that product's whole-row epilogue (returns the functor; the product value is read from the staged tile)."""
        info = p7b_groups[gname]
        members, R, D, T = info["members"], info["rows"], info["extent"], info["segment_rows"]
        pair_ = bool(p7b_segment_pair) and (p7b_segment_pair != "rows" or not T)   # "rows": row-only segments
        vec = bool(p7b_segment_vec) and D % 4 == 0
        # vec: lane l holds columns 128 jj + 4 l + e (4 contiguous per chunk, 16-byte operand loads); else l + 32 j.
        J = 4 * -(-D // 128) if vec else -(-D // 32)
        cexpr = "128u * (j >> 2) + 4u * lane + (j & 3u)" if vec else "lane + 32u * j"
        vrefs = {}
        member_set = set(members)
        full = {m for m in members if nodes[m].kind == "elementwise" and _numel(nodes[m].shape) == R * D}
        cols = [m for m in members if m in p7b_reds and p7b_reds[m]["kind"] == "colT"]
        samples = p7b_samples(R, D, T)

        def reg(root):
            value = f"r_{root}[j]" if root in full else f"q_{root}"
            return f"__bfloat162float({value})" if dtypes[root] == "bfloat16" else value

        def loader(ref, shape, idx):
            root = layouts[ref].root
            if tile_product is not None and pre_layout_root.get(ref, root) == tile_product:
                if not all(p7b_address(layouts[ref], shape, idx, env) == env["g"] * D + env["c"] for env in samples):
                    raise LoweringError(f"internal: P7-B row epilogue {gname} reads {tile_product} non-locally")
                return "__bfloat162float(latch_st[latch_r * latch_lds + c])"
            if root in member_set:
                want_full = root in full
                if root in cols or not all(p7b_address(layouts[ref], shape, idx, env) ==
                                           (env["g"] * D + env["c"] if want_full else env["g"]) for env in samples):
                    raise LoweringError(f"internal: P7-B segment {gname} reads {root} non-locally")
                return reg(root)
            lay = layouts[ref]
            if (vec and ref in declared and dtypes[ref] in ("float32", "bfloat16") and lay.offset % 4 == 0
                    and all(p7b_address(lay, shape, idx, env) == env["g"] * D + env["c"] for env in samples)):
                reg_ = vrefs.setdefault(ref, f"in_{len(vrefs)}")
                return f"__bfloat162float({reg_}[e])" if dtypes[ref] == "bfloat16" else f"{reg_}[e]"
            if (vec and p7b_segment_affine and ref in declared and dtypes[ref] in ("float32", "bfloat16")
                    and lay.offset % 4 == 0 and shape is not None):
                # P7-B: row-affine operand: address = base(g) + c with base(g) % 4 == 0 for every sampled row and
                # every column (checked numerically), e.g. per-segment modulation rows and bias vectors -> one
                # 16-byte load per 4 columns from base(g) + c0 (same elements).
                text = _index32(lay, shape, idx)
                if not set(re.findall(r"(?<![\w.])[A-Za-z_]\w*", re.sub(r"\(unsigned\)", "", text))) - {"g", "c", "i"}:
                    py = re.sub(r"\(unsigned\)", "", text)
                    py = re.sub(r"(?<![\w.])(\d+)(?:u|LL|ULL)\b", r"\1", py).replace("/", "//")
                    try:
                        f_ = eval("lambda g, c, i: " + py, {"__builtins__": {}})
                    except Exception:
                        f_ = None
                    rows_ = sorted({env["g"] for env in samples} | set(range(min(R, 128)))
                                   | set(range(0, R, max(1, R // 64))))
                    ok = f_ is not None
                    for g_ in rows_ if ok else ():
                        base_v = f_(g_, 0, g_ * D)
                        if base_v % 4 or any(f_(g_, c_, g_ * D + c_) != base_v + c_ for c_ in range(D)):
                            ok = False
                            break
                    if ok:
                        base = re.sub(r"\bc\b", "0u", re.sub(r"\bi\b", f"(g * {_u(D)})", text))
                        reg_ = vrefs.setdefault((ref, base), f"in_{len(vrefs)}")
                        return f"__bfloat162float({reg_}[e])" if dtypes[ref] == "bfloat16" else f"{reg_}[e]"
            return memory_loader(ref, shape, idx)

        def elements(body):
            o_line = f"const unsigned o = s * {_u(D)} + c; " if T else ""
            if vec:
                loads = "".join(
                    f"          {_CTYPES[dtypes[key]]} {reg_}[4]; latch_ld4({_ident(key)} + g * {_u(D)} + c0, {reg_});\n"
                    if isinstance(key, str) else
                    f"          {_CTYPES[dtypes[key[0]]]} {reg_}[4]; latch_ld4({_ident(key[0])} + ({key[1]}) + c0, {reg_});\n"
                    for key, reg_ in vrefs.items())
                vrefs.clear()
                if loads and not vector_helpers_needed:
                    vector_helpers_needed.append(True)
                return (f"#pragma unroll\n      for (unsigned jj = 0; jj < {J // 4}u; ++jj) {{\n"
                        f"        const unsigned c0 = 128u * jj + 4u * lane;\n        if (c0 < {_u(D)}) {{\n" + loads +
                        f"#pragma unroll\n          for (unsigned e = 0; e < 4u; ++e) {{\n"
                        f"          const unsigned c = c0 + e, j = jj * 4u + e;\n"
                        f"          const unsigned i = g * {_u(D)} + c; {o_line}(void)i;\n          "
                        + "\n          ".join(body) + "\n          }\n        }\n      }")
            return (f"#pragma unroll\n      for (unsigned j = 0; j < {J}u; ++j) {{\n"
                    f"        const unsigned c = lane + 32u * j;\n        if (c < {_u(D)}) {{\n"
                    f"          const unsigned i = g * {_u(D)} + c; {o_line}(void)i;\n          "
                    + "\n          ".join(body) + "\n        }\n      }")
        lines, stores = [], []
        for m in members:
            node = nodes[m]
            ctype = _CTYPES[dtypes[m]]
            if m in full:
                expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
                if outline_ops or p6b_cse:
                    from .norm_rules import PURE_CALLS, PURE_CALLS_P6B, cse_calls
                    prelude, expression = cse_calls(expression, f"c_{m}", PURE_CALLS_P6B if p6b_cse else PURE_CALLS)
                else:
                    prelude = []
                lines.append(f"{ctype} r_{m}[{J}];")
                lines.append(elements(prelude + [f"r_{m}[j] = {ctype_round(expression, node.dtype)};"]))
                if m in stored and vec:
                    if not vector_helpers_needed:
                        vector_helpers_needed.append(True)
                    stores.append(f"#pragma unroll\n      for (unsigned jj = 0; jj < {J // 4}u; ++jj) {{\n"
                                  f"        const unsigned c0 = 128u * jj + 4u * lane;\n"
                                  f"        if (c0 < {_u(D)}) latch_st4({_ident(m)} + g * {_u(D)} + c0, r_{m}[4u * jj], "
                                  f"r_{m}[4u * jj + 1u], r_{m}[4u * jj + 2u], r_{m}[4u * jj + 3u]);\n      }}")
                elif m in stored:
                    stores.append(elements([f"{_ident(m)}[i] = r_{m}[j];"]))
            elif node.kind == "elementwise":
                expression = node.render("g", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
                lines.append(f"const {ctype} q_{m} = {ctype_round(expression, node.dtype)};")
                if m in stored:
                    stores.append(f"if (lane == 0u) {_ident(m)}[g] = q_{m};")
            else:
                red = p7b_reds[m]
                leaf = expand_storage(red["source"], red["pos"], loader)
                if red["kind"] == "row":
                    total = f"(a_{m}{red['scale']})"
                    lines.append(f"float a_{m} = 0.0f;")
                    lines.append(elements([f"a_{m} += {leaf};"]))
                    lines.append(f"a_{m} = s101h::warp_sum(a_{m});")
                    lines.append(f"const {ctype} q_{m} = {ctype_round(total, red['dtype'])};")
                    stores.append(f"if (lane == 0u) {_ident(m)}[g] = q_{m};")
                else:
                    lines.append(elements([f"ca_{m}[j] += {leaf};"]))
        if p6b_inline_div:
            lines = [line.replace("latch_fdiv(", "latch_fdiv_inline(") for line in lines]
            if P6B_INLINE_DIV not in helper_parts:
                helper_parts.append(P6B_INLINE_DIV)
        if p7b_segment_index:
            lines = [p7b_row_index(line, D) for line in lines]
            stores = [p7b_row_index(line, D) for line in stores]
        body = "\n      ".join(lines + stores)
        if pair_:
            # P7-B: two rows per warp iteration (row g and the warp's next row g1, clamped to a valid row when absent):
            # both rows' loads/compute precede both rows' stores; the second row's column-sum terms are masked
            # (+0.0f) and its stores skipped when it is absent.
            def second(text):
                text = re.sub(r"\bg\b", "g1", text)
                text = re.sub(r"\bt\b", "t1", text)
                text = re.sub(r"\b([rqa])_", r"\g<1>1_", text)
                text = re.sub(r"\bc_(\w)", r"c1_\1", text)
                return re.sub(r"(ca_\w+\[j\]) \+= (.*);", r"\1 += latch_b ? (\2) : 0.0f;", text)
            body = "\n      ".join(lines + [second(line) for line in lines] + stores
                                     + ["if (latch_b) {"] + [second(line) for line in stores] + ["}"])
        label = f"// P7-B segment engine {gname}: {R} rows x {D}" + (f", {R // T} segments x {T} rows" if T else "")
        label += f"; members {', '.join(members)}"
        if tile_product is not None:
            if T or pair_:
                raise LoweringError("internal: P7-B row epilogues are row-only, one row per warp iteration")
            rows_body = ("const unsigned lane = threadIdx.x & 31u, latch_w = threadIdx.x >> 5;\n"
                         "    for (unsigned latch_r = latch_w; latch_r < latch_mr; latch_r += 8u) {\n"
                         f"      const unsigned g = latch_m0 + latch_r;\n      {body}\n    }}")
            return canonical_functor("void", "const __nv_bfloat16* latch_st, unsigned latch_lds, unsigned latch_m0, "
                                     "unsigned latch_mr", rows_body, method="rows")
        if not T:
            if pair_:
                return (f"{{ {label}\n  const unsigned lane = threadIdx.x & 31u, latch_w = threadIdx.x >> 5;\n"
                        f"  const unsigned latch_step = gridDim.x * 8u;\n"
                        f"  for (unsigned g = blockIdx.x * 8u + latch_w; g < {_u(R)}; g += 2u * latch_step) {{\n"
                        f"      const bool latch_b = g + latch_step < {_u(R)}; const unsigned g1 = latch_b ? g + latch_step : g;\n"
                        f"      {body}\n  }}\n}}")
            return (f"{{ {label}\n  const unsigned lane = threadIdx.x & 31u, latch_w = threadIdx.x >> 5;\n"
                    f"  for (unsigned g = blockIdx.x * 8u + latch_w; g < {_u(R)}; g += gridDim.x * 8u) {{\n"
                    f"      {body}\n  }}\n}}")
        S = R // T
        P = int(info.get("parts", 1))
        accs = " ".join(f"float ca_{m}[{J}];" for m in cols)
        zero = "".join(f"ca_{m}[j] = 0.0f; " for m in cols)
        put = "".join(f"smem[{k * 8}u * {_u(D)} + latch_w * {_u(D)} + c] = ca_{m}[j]; " for k, m in enumerate(cols))
        combine = []
        for k, m in enumerate(cols):
            red = p7b_reds[m]
            terms = " + ".join(f"smem[{(k * 8 + w) * D}u + c]" for w in range(1, 8))
            value = ctype_round(f"((smem[{k * 8 * D}u + c] + {terms}){red['scale']})" if False else
                                f"(latch_v{red['scale']})", red["dtype"])
            combine.append(f"    for (unsigned c = threadIdx.x; c < {_u(D)}; c += blockDim.x) {{\n"
                           f"      float latch_v = smem[{k * 8 * D}u + c];\n"
                           + "".join(f"      latch_v += smem[{(k * 8 + w) * D}u + c];\n" for w in range(1, 8))
                           + f"      {_ident(m)}[s * {_u(D)} + c] = {value};\n    }}\n")
        if P > 1:
            # Units (segment, part) split each segment's rows over P CTAs; each unit stores its fixed-order warp
            # combination as a partial, the last arriving unit of a segment sums the P partials in part order.
            TPp = -(-T // P)
            label += f"; {P} parts per segment (last-arriver part-order combination)"
            part_stores, finals = [], []
            for k, m in enumerate(cols):
                red = p7b_reds[m]
                part_stores.append(
                    f"    for (unsigned c = threadIdx.x; c < {_u(D)}; c += blockDim.x) {{\n"
                    f"      float latch_v = smem[{k * 8 * D}u + c];\n"
                    + "".join(f"      latch_v += smem[{(k * 8 + w) * D}u + c];\n" for w in range(1, 8))
                    + f"      latch_part[(({k}u * {_u(S)} + s) * {P}u + p) * {_u(D)} + c] = latch_v;\n    }}\n")
                value = ctype_round(f"(latch_v{red['scale']})", red["dtype"])
                finals.append(
                    f"      for (unsigned c = threadIdx.x; c < {_u(D)}; c += blockDim.x) {{\n"
                    f"        float latch_v = latch_part[(({k}u * {_u(S)} + s) * {P}u) * {_u(D)} + c];\n"
                    + "".join(f"        latch_v += latch_part[(({k}u * {_u(S)} + s) * {P}u + {q}u) * {_u(D)} + c];\n"
                              for q in range(1, P))
                    + f"        {_ident(m)}[s * {_u(D)} + c] = {value};\n      }}\n")
            return (f"{{ {label}\n  const unsigned lane = threadIdx.x & 31u, latch_w = threadIdx.x >> 5;\n"
                    f"  unsigned* latch_cnt = /*P2_CNT*/;\n  float* latch_part = /*P2_PART*/;\n"
                    f"  __shared__ bool latch_last;\n"
                    f"  for (unsigned latch_u = blockIdx.x; latch_u < {_u(S * P)}; latch_u += gridDim.x) {{\n"
                    f"    const unsigned s = latch_u / {P}u, p = latch_u % {P}u;\n"
                    f"    const unsigned latch_t1 = min({_u(T)}, (p + 1u) * {_u(TPp)});\n"
                    f"    {accs}\n#pragma unroll\n    for (unsigned j = 0; j < {J}u; ++j) {{ {zero}}}\n"
                    + (f"    for (unsigned t = p * {_u(TPp)} + latch_w; t < latch_t1; t += 16u) {{\n"
                       f"      const bool latch_b = t + 8u < latch_t1; const unsigned t1 = latch_b ? t + 8u : t;\n"
                       f"      const unsigned g = s * {_u(T)} + t, g1 = s * {_u(T)} + t1;\n      {body}\n    }}\n"
                       if pair_ else
                       f"    for (unsigned t = p * {_u(TPp)} + latch_w; t < latch_t1; t += 8u) {{\n"
                       f"      const unsigned g = s * {_u(T)} + t;\n      {body}\n    }}\n") +
                    f"#pragma unroll\n    for (unsigned j = 0; j < {J}u; ++j) {{\n      const unsigned c = {cexpr};\n"
                    f"      if (c < {_u(D)}) {{ {put}}}\n    }}\n    __syncthreads();\n" + "".join(part_stores)
                    + "    __threadfence();\n    __syncthreads();\n"
                    f"    if (threadIdx.x == 0u) {{\n      const unsigned old = atomicAdd(latch_cnt + s, 1u);\n"
                    f"      latch_last = old == {P - 1}u;\n      if (latch_last) atomicExch(latch_cnt + s, 0u);\n    }}\n"
                    "    __syncthreads();\n    if (latch_last) {\n      __threadfence();\n" + "".join(finals)
                    + "    }\n    __syncthreads();\n  }\n}")
        return (f"{{ {label}\n  const unsigned lane = threadIdx.x & 31u, latch_w = threadIdx.x >> 5;\n"
                f"  for (unsigned s = blockIdx.x; s < {_u(S)}; s += gridDim.x) {{\n"
                f"    {accs}\n#pragma unroll\n    for (unsigned j = 0; j < {J}u; ++j) {{ {zero}}}\n"
                + (f"    for (unsigned t = latch_w; t < {_u(T)}; t += 16u) {{\n"
                   f"      const bool latch_b = t + 8u < {_u(T)}; const unsigned t1 = latch_b ? t + 8u : t;\n"
                   f"      const unsigned g = s * {_u(T)} + t, g1 = s * {_u(T)} + t1;\n      {body}\n    }}\n"
                   if pair_ else
                   f"    for (unsigned t = latch_w; t < {_u(T)}; t += 8u) {{\n      const unsigned g = s * {_u(T)} + t;\n"
                   f"      {body}\n    }}\n") +
                f"#pragma unroll\n    for (unsigned j = 0; j < {J}u; ++j) {{\n      const unsigned c = {cexpr};\n"
                f"      if (c < {_u(D)}) {{ {put}}}\n    }}\n    __syncthreads();\n" + "".join(combine)
                + "    __syncthreads();\n  }\n}")

    ident_dtype = {_ident(name): dtypes[name] for name in layouts}
    _STAGEABLE = ("float32", "bfloat16")
    functor_types = {}

    def _canonical_names(text, names):
        for k, name in enumerate(names):
            text = re.sub(r"\b" + re.escape(name) + r"\b", f"m{k}", text)
        for k, local in enumerate(dict.fromkeys(re.findall(r"\b[ejr]_\w+\b", text))):
            text = re.sub(r"\b" + re.escape(local) + r"\b", f"t{k}", text)
        return text

    def canonical_functor(result, parameters, body, method="operator()", pre=None):
        """Generated functor as a struct whose pointers are members, so identical per-layer code shares one type
        (and one matrix body instantiation). Returns the construction expression for the call site.
        pre: optional body of `void pre(unsigned i) const` (P6-B epilogue operand prefetch), same members."""
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", body + (pre or ""))))
        canonical = _canonical_names(body, names)
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        pre_method = (f"  __device__ __forceinline__ void pre(unsigned i) const {{\n    {_canonical_names(pre, names)}\n  }}\n"
                      if pre else "")
        key = (result, parameters, fields, canonical, method, pre_method)
        if key not in functor_types:
            type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            functor_types[key] = type_name
            extra = ""
            if method == "tile":
                extra = ("  static constexpr bool kTile = true;\n"
                         "  __device__ __forceinline__ void operator()(unsigned, float, float) const {}\n")
            helper_parts.append(f"struct {type_name} {{\n  {fields}\n{extra}{pre_method}  __device__ __forceinline__ {result} "
                                f"{method}({parameters}) const {{\n    {canonical}\n  }}\n}};")
        return f"{functor_types[key]}{{{', '.join(names)}}}"

    def p4a_emit(name, spec):
        """Outlined P-OPS4-A call: canonical leaf functor (pointer members) + shared template instantiation."""
        if spec["mode"] == "finalize":
            code = nodes[name].code
        elif spec["mode"] == "column_last":
            return p4a_last_call(name, spec)
        else:
            from .pops4a_rules import OUTLINED_HEADER, INLINE_HEADER
            header = INLINE_HEADER if pops4a_outline == "inline" else OUTLINED_HEADER
            if header not in helper_parts:
                helper_parts.append(header)
            leaves = [heavy_code(SimpleNamespace(name=name, code=leaf)) for leaf in spec["leaves"]]
            if spec["mode"] == "column":
                body = " ".join(f"v[{k}] = {leaf};" for k, leaf in enumerate(leaves))
                functor = canonical_functor("void", "unsigned pr, unsigned o, float* v", body)
                parts = spec["parts"] + ["nullptr"] * (4 - len(spec["parts"]))
                staged = p4a_staged_call(name, spec, parts) if pops4a_stage else None
                if staged is not None:
                    return staged
                vector = p4a_vector_call(name, spec, parts) if spec.get("VC") else None
                if vector is not None:
                    return vector
                distance = int(pops4a_prefetch or 0)
                if distance:
                    from .pops4a_rules import PF_HEADER
                    if PF_HEADER not in helper_parts:
                        helper_parts.append(PF_HEADER)

                    def pf_loader(ref, shape, idx):
                        expression = memory_loader(ref, shape, idx)
                        return f"latch_pf(&({expression}))" if dtypes[ref] == "float32" else "0.0f"

                    def pf_substitute(match):
                        kind, ref, shape, idx = token_loads[int(match.group(1))]
                        return (expand(ref, shape, idx, pf_loader) if kind == "index"
                                else expand_storage(ref, idx, pf_loader))
                    pf_body = " ".join(f"(void)({_TOKEN.sub(pf_substitute, leaf)});" for leaf in spec["leaves"])
                    pf = canonical_functor("void", "unsigned pr, unsigned o", pf_body)
                else:
                    pf = "latch_p4a_nopf{}"
                code = (f"latch_p4a_column<{functor.split('{')[0]}, {pf.split('{')[0]}, {len(leaves)}, {spec['C']}, "
                        f"{distance}>({functor}, {pf}, "
                        f"{', '.join(parts)}, {spec['E']}u, {spec['N']}u, {spec['T']}u, {spec['RT']}u, {spec['B']}u, "
                        f"{spec['rot'] % 4096}u, smem);")
            else:
                vector = (p4a_vector_full_call(name, spec)
                          if spec["mode"] == "full" and pops4a_vec and pops4a_vec_full and spec["S"] % 4 == 0 and spec["CH"] % 1024 == 0
                          else None)
                if vector is not None:
                    return vector
                functor = canonical_functor("float", "unsigned s", f"return {leaves[0]};")
                if spec["mode"] == "full":
                    code = (f"latch_p4a_full({functor}, {spec['part']}, {spec['S']}u, {spec['CH']}u, {spec['NC']}u, "
                            f"{spec['rot'] % 4096}u, smem);")
                else:
                    code = (f"if (blockIdx.x == {spec['rot'] % 4096}u % gridDim.x) {{\n"
                            f"  const float latch_t = latch_p4a_direct({functor}, {spec['S']}u, smem);\n"
                            f"  if (threadIdx.x == 0u) {spec['out']}[0] = "
                            f"{'latch_t / ' + spec['div'] if spec['div'] != '0.0f' else 'latch_t'};\n}}")
        return code.replace("latch_p4a_", "latch_p4i_") if pops4a_outline == "inline" else code

    def p4a_last_call(name, spec):
        from . import pops4a_rules as R
        if R.LAST_HEADER not in helper_parts:
            helper_parts.append(R.LAST_HEADER)
        outs = spec["outs"] + ["nullptr"] * (4 - len(spec["outs"]))
        tail = (f"{', '.join(outs)}, {spec['div']}, /*P2_CNT*/, /*P2_PART*/, {spec['E']}u, {spec['N']}u, "
                f"{spec['T']}u, {spec['RT']}u")
        functor = p4a_vector_functor(spec) if spec.get("VC") else None
        if functor is not None:
            texts = len(spec["leaves"])
            nodes[name].extra["shared_bytes"] = texts * spec["VC"] * 4 * 256 * 4
            return (f"latch_p4l_vector<{functor.split('{')[0]}, {texts}, {spec['VC']}>({functor}, {tail}, "
                    f"{spec['VB']}u, {spec['rot'] % 4096}u, smem);")
        leaves = [heavy_code(SimpleNamespace(name=name, code=leaf)) for leaf in spec["leaves"]]
        functor = canonical_functor("void", "unsigned pr, unsigned o, float* v",
                                    " ".join(f"v[{k}] = {leaf};" for k, leaf in enumerate(leaves)))
        return (f"latch_p4l_scalar<{functor.split('{')[0]}, {len(leaves)}, {spec['C']}>({functor}, {tail}, "
                f"{spec['B']}u, {spec['rot'] % 4096}u, smem);")

    def p4a_vector_functor(spec):
        """Canonical float4 leaf functor, or None when a load is not proven contiguous/aligned or broadcast."""
        from . import pops4a_rules as R
        records = []

        def loader(ref, shape, idx):
            records.append((ref, memory_loader(ref, shape, idx)))
            return f"@@Q{len(records) - 1}@@"

        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            return expand(ref, shape, idx, loader) if kind == "index" else expand_storage(ref, idx, loader)
        texts = [_TOKEN.sub(substitute, leaf) for leaf in spec["leaves"]]
        if any(re.search(r"\bo\b", re.sub(r"@@Q\d+@@", "", t)) for t in texts):
            return None
        declarations, components = [], {}
        for n, (ref, expression) in enumerate(records):
            match = re.fullmatch(r"(v_\w+)\[(.*)\]", expression)
            if dtypes[ref] != "float32" or not match:
                return None
            kind = R.classify_load(match.group(2), spec["E"], spec["N"])
            layout = layouts[ref]
            base = layout.offset + (0 if layout.root in inputs else offsets.get(layout.root, 1))
            if kind == "vec" and base % 4 == 0:
                declarations.append(f"const float4 q{n} = *reinterpret_cast<const float4*>(&{expression});")
                components[n] = [f"q{n}.{c}" for c in "xyzw"]
            elif kind == "bcast":
                declarations.append(f"const float q{n} = {expression};")
                components[n] = [f"q{n}"] * 4
            else:
                return None
        placeholder = re.compile(r"@@Q(\d+)@@")
        lanes = [f"v[{k}][{e}] = " + placeholder.sub(lambda m, e=e: components[int(m.group(1))][e], t) + ";"
                 for k, t in enumerate(texts) for e in range(4)]
        return canonical_functor("void", "unsigned pr, unsigned o, float (*v)[4]",
                                 " ".join(declarations) + " " + " ".join(lanes))

    def p4a_vector_call(name, spec, parts):
        """Vectorized column item (pops4a_rules.VECTOR_HEADER) when every leaf load is proven unit-stride and
        16-byte aligned in the column index, or independent of it; None keeps the scalar body."""
        from . import pops4a_rules as R
        records = []

        def loader(ref, shape, idx):
            records.append((ref, memory_loader(ref, shape, idx)))
            return f"@@Q{len(records) - 1}@@"

        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            return expand(ref, shape, idx, loader) if kind == "index" else expand_storage(ref, idx, loader)
        texts = [_TOKEN.sub(substitute, leaf) for leaf in spec["leaves"]]
        if any(re.search(r"\bo\b", re.sub(r"@@Q\d+@@", "", t)) for t in texts):
            return None
        declarations, components = [], {}
        for n, (ref, expression) in enumerate(records):
            match = re.fullmatch(r"(v_\w+)\[(.*)\]", expression)
            if dtypes[ref] != "float32" or not match:
                return None
            kind = R.classify_load(match.group(2), spec["E"], spec["N"])
            layout = layouts[ref]
            base = layout.offset + (0 if layout.root in inputs else offsets.get(layout.root, 1))
            if kind == "vec" and base % 4 == 0:
                declarations.append(f"const float4 q{n} = *reinterpret_cast<const float4*>(&{expression});")
                components[n] = [f"q{n}.{c}" for c in "xyzw"]
            elif kind == "bcast":
                declarations.append(f"const float q{n} = {expression};")
                components[n] = [f"q{n}"] * 4
            else:
                return None
        placeholder = re.compile(r"@@Q(\d+)@@")
        lanes = []
        for k, t in enumerate(texts):
            for e in range(4):
                value = placeholder.sub(lambda m, e=e: components[int(m.group(1))][e], t)
                lanes.append(f"v[{k}][{e}] = {value};")
        body = " ".join(declarations) + " " + " ".join(lanes)
        functor = canonical_functor("void", "unsigned pr, unsigned o, float (*v)[4]", body)
        from .pops4a_rules import VECTOR_HEADER
        if VECTOR_HEADER not in helper_parts:
            helper_parts.append(VECTOR_HEADER)
        nodes[name].extra["shared_bytes"] = len(texts) * spec["VC"] * 4 * 256 * 4
        nodes[name].extra["vectorized"] = True
        return (f"latch_p4v_column<{functor.split('{')[0]}, {len(texts)}, {spec['VC']}>({functor}, "
                f"{', '.join(parts)}, {spec['E']}u, {spec['N']}u, {spec['T']}u, {spec['RT']}u, {spec['VB']}u, "
                f"{spec['rot'] % 4096}u, smem);")

    def p4a_vector_full_call(name, spec):
        """Vectorized full-reduction chunks (pops4a_rules.VECTOR_FULL_HEADER): proven unit-stride, 16-byte aligned
        storage-order loads; row-independent loads stay scalar. None keeps the scalar body."""
        from . import pops4a_rules as R
        records = []

        def loader(ref, shape, idx):
            records.append((ref, memory_loader(ref, shape, idx)))
            return f"@@Q{len(records) - 1}@@"

        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            return expand(ref, shape, idx, loader) if kind == "index" else expand_storage(ref, idx, loader)
        text = _TOKEN.sub(substitute, spec["leaves"][0])
        if re.search(r"\bs\b", re.sub(r"@@Q\d+@@", "", text)):
            return None
        declarations, components = [], {}
        for n, (ref, expression) in enumerate(records):
            match = re.fullmatch(r"(v_\w+)\[(.*)\]", expression)
            if dtypes[ref] != "float32" or not match:
                return None
            kind = R.classify_full_load(match.group(2), spec["S"])
            layout = layouts[ref]
            base = layout.offset + (0 if layout.root in inputs else offsets.get(layout.root, 1))
            if kind == "vec" and base % 4 == 0:
                declarations.append(f"const float4 q{n} = *reinterpret_cast<const float4*>(&{expression});")
                components[n] = [f"q{n}.{c}" for c in "xyzw"]
            elif kind == "bcast":
                declarations.append(f"const float q{n} = {expression};")
                components[n] = [f"q{n}"] * 4
            else:
                return None
        placeholder = re.compile(r"@@Q(\d+)@@")
        lanes = [f"v[{e}] = " + placeholder.sub(lambda m, e=e: components[int(m.group(1))][e], text) + ";"
                 for e in range(4)]
        functor = canonical_functor("void", "unsigned s, float* v", " ".join(declarations) + " " + " ".join(lanes))
        from .pops4a_rules import VECTOR_FULL_HEADER
        if VECTOR_FULL_HEADER not in helper_parts:
            helper_parts.append(VECTOR_FULL_HEADER)
        nodes[name].extra["vectorized"] = True
        return (f"latch_p4v_full({functor}, {spec['part']}, {spec['S']}u, {spec['CH']}u, {spec['NC']}u, "
                f"{spec['rot'] % 4096}u, smem);")

    def p4a_staged_call(name, spec, parts):
        """cp.async-staged column item (pops4a_rules.STAGED_HEADER); None when an operand is not FP32."""
        from .pops4a_rules import STAGED_HEADER, stage_depth
        cols = spec["C"]
        stride = cols * 32
        refs_seen = []

        def render(loader_of):
            counter = [0]

            def loader(ref, shape, idx):
                refs_seen.append(ref)
                q = counter[0]
                counter[0] += 1
                return loader_of(q, ref, shape, idx)

            def substitute(match):
                kind, ref, shape, idx = token_loads[int(match.group(1))]
                return expand(ref, shape, idx, loader) if kind == "index" else expand_storage(ref, idx, loader)
            texts = [_TOKEN.sub(substitute, leaf) for leaf in spec["leaves"]]
            return texts, counter[0]
        issue_texts, loads = render(lambda q, ref, shape, idx:
                                    f"latch_cpa(slot + {q * stride}u, &({memory_loader(ref, shape, idx)}))")
        if any(dtypes[ref] != "float32" for ref in refs_seen) or loads == 0:
            return None
        compute_texts, loads2 = render(lambda q, ref, shape, idx: f"slot[{q * stride}u]")
        assert loads2 == loads
        if STAGED_HEADER not in helper_parts:
            helper_parts.append(STAGED_HEADER)
        depth = stage_depth(loads, cols, pops4a_stage_bytes)
        issue = canonical_functor("void", "unsigned pr, unsigned o, float* slot",
                                  " ".join(f"(void)({t});" for t in issue_texts))
        leaf = canonical_functor("void", "const float* slot, float* v",
                                 " ".join(f"v[{k}] = {t};" for k, t in enumerate(compute_texts)))
        nodes[name].extra["shared_bytes"] = max(8 * depth * loads * stride * 4, len(compute_texts) * cols * 256 * 4)
        nodes[name].extra["stage"] = {"depth": depth, "loads": loads}
        return (f"latch_p4s_column<{issue.split('{')[0]}, {leaf.split('{')[0]}, {len(compute_texts)}, {cols}, "
                f"{depth}, {loads}>({issue}, {leaf}, {', '.join(parts)}, {spec['E']}u, {spec['N']}u, {spec['T']}u, "
                f"{spec['RT']}u, {spec['B']}u, {spec['rot'] % 4096}u, smem);")

    def tile_functor(body, prefetch, ebytes):
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", body + "\n" + prefetch)))
        body_c, prefetch_c = _canonical_names(body, names), _canonical_names(prefetch, names)
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        key = ("tile", fields, body_c, prefetch_c, ebytes)
        if key not in functor_types:
            type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            functor_types[key] = type_name
            helper_parts.append(
                f"struct {type_name} {{\n  {fields}\n  static constexpr bool kTile = true;\n"
                f"  template <int BM, int BN> __host__ __device__ static constexpr int ebytes() {{ return {ebytes}; }}\n"
                "  __device__ __forceinline__ void operator()(unsigned, float, float) const {}\n"
                f"  template <int BM, int BN> __device__ __forceinline__ void prefetch(char* E, unsigned base, unsigned ldc) const {{\n"
                f"    {prefetch_c}\n  }}\n"
                f"  template <int BM, int BN> __device__ __forceinline__ void tile(const float* T, int ldt, const char* E, "
                f"unsigned base, unsigned ldc) const {{\n    {body_c}\n  }}\n}};")
        return f"{functor_types[key]}{{{', '.join(names)}}}"

    def pair_epilogue_code(product, members):
        """P6-B: register-direct pair epilogue (pops4b::tma_body EPI) for a warpgroup-MMA product.

        epi(i, g0, g1): elements i and i + 1 of the row-major product (an even column pair); g0/g1 are the rounded
        and widened product values. Per element: the identical per-operation expressions of the members, stores of
        the stored members (permuted members at their bijective position)."""
        spec = nodes[product].extra["matrix_spec"]
        size = spec["m"] * spec["n"]
        member_set = set(members)
        compute = []
        prefetch = []                     # P6-B: identity-indexed operands read from memory (epilogue prefetch)
        for j in (0, 1):
            here = "i" if j == 0 else "(i + 1u)"

            def loader(ref, shape, idx, j=j, here=here):
                from .norm_rules import normalized_index
                origin = pre_layout_root.get(ref, layouts[ref].root)
                identity = (normalized_index(idx) == normalized_index(here) and shape is not None
                            and _numel(shape) == size and layouts[ref].numel == size
                            and layouts[ref].contiguous and layouts[ref].offset == 0)
                if identity and origin == product:
                    return f"g{j}"
                if identity and origin in member_set and origin not in permuted_members:
                    value = f"r_{origin}_{j}"
                    return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
                if identity and j == 0 and ref in declared and _ident(ref) not in prefetch:
                    prefetch.append(_ident(ref))
                return memory_loader(ref, shape, idx)
            for name in members:
                node = nodes[name]
                if name in permuted_members:
                    strides = _contiguous(node.shape)
                    terms = [f"(({here} / {_u(stride)}) % {_u(extent)}) * {_u(cstride)}"
                             for (stride, extent), cstride in zip(permuted_members[name], strides) if extent != 1]
                    index = f"j_{name}_{j}"
                    expression = node.render(index, lambda arg, i2, s=None, node=node, j=j:
                                             operand(arg, node.shape if s is None else s, i2,
                                                     lambda ref, shape, idx, j=j: f"g{j}"))
                    compute.append(f"const unsigned {index} = {' + '.join(terms) or '0u'};")
                    compute.append(f"const {_CTYPES[node.dtype]} r_{name}_{j} = {ctype_round(expression, node.dtype)};")
                    if name in stored:
                        compute.append(f"{_ident(name)}[{index}] = r_{name}_{j};")
                    continue
                expression = node.render(here, lambda arg, i2, s=None, node=node, loader=loader:
                                         operand(arg, node.shape if s is None else s, i2, loader))
                if p6b_cse:
                    from .norm_rules import PURE_CALLS_P6B, cse_calls
                    prelude, expression = cse_calls(expression, f"c_{name}_{j}", PURE_CALLS_P6B)
                    compute.extend(prelude)
                compute.append(f"const {_CTYPES[node.dtype]} r_{name}_{j} = {ctype_round(expression, node.dtype)};")
                if name in stored:
                    compute.append(f"{_ident(name)}[{here}] = r_{name}_{j};")
        if p6b_cse:
            from .norm_rules import cse_lines
            compute = cse_lines(compute, "cse_p")
        if p6b_inline_div and size >= P6B_INLINE_DIV_MIN:
            compute = [line.replace("latch_fdiv(", "latch_fdiv_inline(") for line in compute]
            if P6B_INLINE_DIV not in helper_parts:
                helper_parts.append(P6B_INLINE_DIV)
        pre = None
        if p6b_epilogue_prefetch and prefetch:
            pre = " ".join(f'asm volatile("prefetch.global.L2 [%0];" :: "l"({ident} + i));' for ident in prefetch)
        functor = canonical_functor("void", "unsigned i, float g0, float g1", "\n    ".join(compute), pre=pre)
        return {"code": functor, "store": product not in elided_products, "members": list(members), "pair": True}

    def p7b_vec_epilogue_code(product, members):
        """P7-B staged epilogue functor: vec8(i, g) evaluates elements i..i+7 of one row (8-column chunk of the staged
        BF16-rounded tile; g are the rounded, widened values). Identity-indexed operands are read and stored members
        written with 16-byte vector accesses; other operands per element. Same per-operation expressions as the
        pair epilogue. Returns None when a member is not elementwise-identity (permuted members)."""
        spec = nodes[product].extra["matrix_spec"]
        size = spec["m"] * spec["n"]
        if spec["n"] % 8 or any(m in permuted_members for m in members):
            return None
        member_set = set(members)
        vec_in, compute, prefetch = {}, [], []
        from .norm_rules import normalized_index
        affine_memo = {}

        def affine(ref, shape):
            """P7-B: row-affine side input: address f(i) with f(i + e) = f(i) + e (e < 8) and f(i) % 8 == 0 for every
            8-column chunk of densely sampled rows (numeric check) -> one 8-element vector load at f(i)."""
            if ref in affine_memo:
                return affine_memo[ref]
            result = None
            if (p7b_epi_affine and ref in declared and dtypes[ref] in ("float32", "bfloat16") and shape is not None
                    and layouts[ref].offset % 8 == 0):
                text = _index32(layouts[ref], shape, "i")
                py = re.sub(r"\(unsigned\)", "", text)
                py = re.sub(r"(?<![\w.])(\d+)(?:u|LL|ULL)\b", r"\1", py).replace("/", "//")
                if not set(re.findall(r"(?<![\w.])[A-Za-z_]\w*", py)) - {"i"}:
                    try:
                        f_ = eval("lambda i: " + py, {"__builtins__": {}})
                    except Exception:
                        f_ = None
                    M_, N_ = spec["m"], spec["n"]
                    rows_ = sorted({0, 1, M_ // 2, M_ - 1} | set(range(min(M_, 64))) | set(range(0, M_, max(1, M_ // 64))))
                    ok = f_ is not None
                    for r_ in rows_ if ok else ():
                        for c_ in range(0, N_, 8):
                            i_ = r_ * N_ + c_
                            b_ = f_(i_)
                            if b_ % 8 or any(f_(i_ + e_) != b_ + e_ for e_ in range(8)):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        result = text
            affine_memo[ref] = result
            return result
        for e in range(8):
            here = "i" if e == 0 else f"(i + {e}u)"

            def loader(ref, shape, idx, e=e, here=here):
                origin = pre_layout_root.get(ref, layouts[ref].root)
                identity = (normalized_index(idx) == normalized_index(here) and shape is not None
                            and _numel(shape) == size and layouts[ref].numel == size
                            and layouts[ref].contiguous and layouts[ref].offset == 0)
                if identity and origin == product:
                    return f"g[{e}]"
                if identity and origin in member_set:
                    value = f"r_{origin}[{e}]"
                    return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
                if identity and ref in declared and dtypes[ref] in ("float32", "bfloat16"):
                    if ref not in vec_in:
                        vec_in[ref] = f"in_{len(vec_in)}"
                        if p6b_epilogue_prefetch:
                            prefetch.append(_ident(ref))
                    value = f"{vec_in[ref]}[{e}]"
                    return f"__bfloat162float({value})" if dtypes[ref] == "bfloat16" else value
                if normalized_index(idx) == normalized_index(here):
                    base = affine(ref, shape)
                    if base is not None:
                        key = (ref, base)
                        if key not in vec_in:
                            vec_in[key] = f"in_{len(vec_in)}"
                        value = f"{vec_in[key]}[{e}]"
                        return f"__bfloat162float({value})" if dtypes[ref] == "bfloat16" else value
                return memory_loader(ref, shape, idx)
            for name in members:
                node = nodes[name]
                expression = node.render(here, lambda arg, i2, s=None, node=node, loader=loader:
                                         operand(arg, node.shape if s is None else s, i2, loader))
                if p6b_cse:
                    from .norm_rules import PURE_CALLS_P6B, cse_calls
                    prelude, expression = cse_calls(expression, f"c_{name}_{e}", PURE_CALLS_P6B)
                    compute.extend(prelude)
                compute.append(f"r_{name}[{e}] = {ctype_round(expression, node.dtype)};")
        if p6b_cse:
            from .norm_rules import cse_lines
            compute = cse_lines(compute, "cse_p")
        if p6b_inline_div and size >= P6B_INLINE_DIV_MIN:
            compute = [line.replace("latch_fdiv(", "latch_fdiv_inline(") for line in compute]
            if P6B_INLINE_DIV not in helper_parts:
                helper_parts.append(P6B_INLINE_DIV)
        if P7B_VEC8 not in helper_parts:
            helper_parts.insert(0, P7B_VEC8)
        loads = [f"{_CTYPES[dtypes[ref]]} {reg}[8]; latch_p7b_ld8({_ident(ref)} + i, {reg});"
                 if isinstance(ref, str) else
                 f"{_CTYPES[dtypes[ref[0]]]} {reg}[8]; latch_p7b_ld8({_ident(ref[0])} + ({ref[1]}), {reg});"
                 for ref, reg in vec_in.items()]
        decls = [f"{_CTYPES[nodes[n].dtype]} r_{n}[8];" for n in members]
        stores = [f"latch_p7b_st8({_ident(n)} + i, r_{n});" for n in members if n in stored]
        body = "\n    ".join(loads + decls + compute + stores)
        pre = None
        if prefetch:
            pre = " ".join(f'asm volatile("prefetch.global.L2 [%0];" :: "l"({ident} + i));' for ident in prefetch)
        sums = colsum_of.get(product, ())
        if sums:
            if len(sums) != 1 or sums[0][0] not in member_set:
                raise LoweringError(f"internal: P9-E epilogue column sum of {product} needs one summed member")
            body = body + "\n    " + " ".join(f"cs[{e}] = (float)(r_{sums[0][0]}[{e}]);" for e in range(8))
            functor = canonical_functor("void", "unsigned i, const float* g, float* cs", body, method="vec8c", pre=pre)
            return {"code": functor, "store": product not in elided_products, "members": list(members), "pair": True,
                    "staged": True, "colpart": sums[0][1]}
        functor = canonical_functor("void", "unsigned i, const float* g", body, method="vec8", pre=pre)
        return {"code": functor, "store": product not in elided_products, "members": list(members), "pair": True,
                "staged": True}

    def epilogue_code(product, members):
        """Generated matrix epilogue over the product tile staged in shared memory.

        Per element: identical per-operation expressions on the rounded BF16 product value. Threads walk the tile
        in row-major groups of four contiguous outputs; identity-indexed operands use 16-byte loads, all loads of
        an unrolled group are issued before any store (A's loop discipline).
        """
        spec = nodes[product].extra["matrix_spec"]
        size = spec["m"] * spec["n"]
        member_set = set(members)
        vector_refs = {}

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            identity = (idx == "i" and shape is not None and _numel(shape) == size and layouts[ref].numel == size
                        and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin == product:
                return "gv[u][j]"
            if identity and origin in member_set and origin not in permuted_members:
                expression = f"r_{origin}[u][j]"
                return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
            if identity and dtypes[ref] in ("float32", "bfloat16"):
                vector_refs.setdefault(ref, f"in_{len(vector_refs)}")
                expression = f"{vector_refs[ref]}[u][j]"
                return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        compute, scattered = [], []
        for name in members:
            node = nodes[name]
            if name in permuted_members:
                strides = _contiguous(node.shape)
                terms = [f"((i / {_u(stride)}) % {_u(extent)}) * {_u(cstride)}"
                         for (stride, extent), cstride in zip(permuted_members[name], strides) if extent != 1]
                index = f"j_{name}"
                expression = node.render(index, lambda arg, i2, node=node: operand(arg, node.shape, i2,
                                                                                  lambda ref, shape, idx: "gv[u][j]"))
                compute.append(f"const unsigned {index} = {' + '.join(terms) or '0u'};")
                compute.append(f"r_{name}[u][j] = {ctype_round(expression, node.dtype)};")
                if name in stored:
                    scattered.append(f"{_ident(name)}[{index}] = r_{name}[u][j];")
                continue
            expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
            compute.append(f"r_{name}[u][j] = {ctype_round(expression, node.dtype)};")
        if scattered:
            # Scattered (permuted) stores happen with their computation; nothing else reads them in this phase.
            compute.extend(scattered)
        stored_members = [n for n in members if n in stored and n not in permuted_members]
        live = 4 * (len(vector_refs) + len(members) + 1)
        unroll = 1
        while unroll * 2 <= 4 and unroll * 2 * live <= LIVE_VALUE_CAP:
            unroll *= 2
        declarations = " ".join([f"float gv[{unroll}][4];"] +
                                [f"{_CTYPES[dtypes[ref]]} {reg}[{unroll}][4];" for ref, reg in vector_refs.items()] +
                                [f"{_CTYPES[nodes[n].dtype]} r_{n}[{unroll}][4];" for n in members])
        # Identity-indexed epilogue operands were prefetched into E (row-major [BM][BN] per operand) by cp.async at
        # tile start, overlapping the mainloop; the epilogue reads them from shared memory.
        offsets, running = {}, "0"
        for ref in vector_refs:
            offsets[ref] = running
            running = f"{running} + pops2b::epi_bytes<{_CTYPES[dtypes[ref]]}, BM, BN>()"
        loads = "".join(f"        latch_ld4(reinterpret_cast<const {_CTYPES[dtypes[ref]]}*>(E + {offsets[ref]}) + r * BN + c, {reg}[u]);\n"
                        for ref, reg in vector_refs.items())
        stores = "".join(f"        latch_st4({_ident(n)} + f[u], r_{n}[u][0], r_{n}[u][1], r_{n}[u][2], r_{n}[u][3]);\n"
                         for n in stored_members)
        if VECTOR_HELPERS not in helper_parts:
            helper_parts.append(VECTOR_HELPERS)          # guarded; must precede the functor definitions
        prefetch = "\n    ".join(f"pops2b::epi_prefetch<{_CTYPES[dtypes[ref]]}, BM, BN>({_ident(ref)} + base, ldc, "
                                   f"reinterpret_cast<{_CTYPES[dtypes[ref]]}*>(E + {offsets[ref]}));"
                                   for ref in vector_refs)
        ebytes = running
        body = (f"constexpr int rows = BM, cols = BN;\n    const int cg = cols >> 2, groups = rows * cg;\n"
                f"    for (int q0 = threadIdx.x; q0 < groups; q0 += {unroll} * (int)blockDim.x) {{\n"
                f"      {declarations} unsigned f[{unroll}]; bool ok[{unroll}];\n"
                f"#pragma unroll\n      for (int u = 0; u < {unroll}; ++u) {{\n"
                f"        const int q = q0 + u * (int)blockDim.x; ok[u] = q < groups;\n"
                f"        if (!ok[u]) continue;\n"
                f"        const int r = q / cg, c = (q - r * cg) << 2;\n"
                f"        f[u] = base + (unsigned)r * ldc + (unsigned)c;\n"
                f"        const float4 t = *reinterpret_cast<const float4*>(T + r * ldt + c);\n"
                f"        gv[u][0] = t.x; gv[u][1] = t.y; gv[u][2] = t.z; gv[u][3] = t.w;\n{loads}"
                f"      }}\n"
                f"#pragma unroll\n      for (int u = 0; u < {unroll}; ++u) {{\n"
                f"        if (!ok[u]) continue;\n"
                f"#pragma unroll\n        for (int j = 0; j < 4; ++j) {{\n"
                f"          const unsigned i = f[u] + (unsigned)j;\n          " + "\n          ".join(compute) +
                f"\n        }}\n      }}\n"
                f"#pragma unroll\n      for (int u = 0; u < {unroll}; ++u) {{\n"
                f"        if (!ok[u]) continue;\n{stores}      }}\n    }}")
        code = tile_functor(body, prefetch, ebytes)
        written = sum(_BYTES[nodes[name].dtype] for name in members if name in stored)
        bytes_read = sum(_BYTES[dtypes[ref]] for ref in vector_refs)
        return {"code": code, "store": product not in elided_products, "members": list(members),
                "bytes_per_element": bytes_read + written, "prefetch_sizes": [_BYTES[dtypes[ref]] for ref in vector_refs]}

    def prologue_code(cast, product, which):
        """Staged operand prologue: identity-indexed sources of the cast are copied by cp.async into their own
        shared-memory rings; the cast's identical FP32 expression is evaluated from them (broadcast operands read
        directly) and rounded with cvt.rn.bf16x2 into the stage's BF16 operand slot."""
        node = nodes[cast]
        size = _numel(node.shape)
        spec = nodes[product].extra["matrix_spec"]
        staged = {}

        def loader(ref, shape, idx):
            identity = (idx == "i" and shape is not None and _numel(shape) == size and layouts[ref].numel == size
                        and layouts[ref].contiguous and layouts[ref].offset == 0 and dtypes[ref] in _STAGEABLE)
            if identity:
                k = staged.setdefault(ref, len(staged))
                expression = f"s{k}[q{k}]"
                return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
        types = {ref: _CTYPES[dtypes[ref]] for ref in staged}
        geo = lambda ref: f"pops2b::Src<{types[ref]}, R, RK, KS>"
        offsets, running = {}, "0"
        for ref in staged:
            offsets[ref] = running
            running = f"{running} + {geo(ref)}::BYTES"
        byte_sum = " + ".join(f"{geo(ref)}::BYTES" for ref in staged) or "0"
        issue = "\n    ".join(f"pops2b::sstage<{types[ref]}, R, RK, KS>({_ident(ref)} + off, sr, sk, r0, k, "
                              f"reinterpret_cast<{types[ref]}*>(src + {offsets[ref]}));" for ref in staged)
        pointers = " ".join(f"const {types[ref]}* s{k} = reinterpret_cast<const {types[ref]}*>(src + {offsets[ref]});"
                            for ref, k in staged.items())
        positions = " ".join(f"const int q{k} = {geo(ref)}::pos(er, ek);" for ref, k in staged.items())
        body_issue = issue
        body_convert = (f"{pointers}\n"
                        "    const long long step = RK ? sk : sr;\n"
                        "    for (int c = threadIdx.x; c < R * KS / 2; c += NTHREADS) {\n"
                        "      const int r = RK ? c / (KS / 2) : (c % (R / 2)) * 2, kk = RK ? (c % (KS / 2)) * 2 : c / (R / 2);\n"
                        "      float v[2]; unsigned at[2];\n"
                        "#pragma unroll\n      for (int e = 0; e < 2; ++e) {\n"
                        "        const int er = RK ? r : r + e, ek = RK ? kk + e : kk;\n"
                        "        const unsigned i = (unsigned)(off + (long long)(r0 + er) * sr + (long long)(k + ek) * sk);\n"
                        f"        {positions}\n"
                        f"        at[e] = i; v[e] = {expression};\n"
                        "      }\n"
                        "      const uint32_t packed = s101h::bpk(v[0], v[1]);\n"
                        "      pops2b::put_pair<R, RK, KS>(dst, r, kk, packed);\n"
                        "      if (out != nullptr) {\n"
                        "        if (step == 1) *reinterpret_cast<uint32_t*>(out + at[0]) = packed;\n"
                        "        else { const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&packed); out[at[0]] = h[0]; out[at[1]] = h[1]; }\n"
                        "      }\n    }")
        code = staged_functor(byte_sum, body_issue, body_convert)
        materialize = absorbed[cast]["mode"] == "materialize"
        return {"code": code, "offset": layouts[spec[which]].offset,
                "out": _ident(cast) if materialize else "nullptr", "cast": cast, "mode": absorbed[cast]["mode"],
                "sources": [dtypes[ref] for ref in staged]}

    def staged_functor(byte_sum, body_issue, body_convert):
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", body_issue + "\n" + body_convert)))
        rename = lambda text: _canonical_names(text, names)
        issue, convert = rename(body_issue), rename(body_convert)
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        key = ("staged", fields, byte_sum, issue, convert)
        if key not in functor_types:
            type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            functor_types[key] = type_name
            helper_parts.append(
                f"struct {type_name} {{\n  {fields}\n  static constexpr bool kStaged = true;\n"
                f"  template <int R, bool RK, int KS> __host__ __device__ static constexpr int bytes() {{ return {byte_sum}; }}\n"
                f"  template <int R, bool RK, int KS> __device__ __forceinline__ void issue(char* src, long long off, "
                f"long long sr, long long sk, int r0, int k) const {{\n    {issue}\n  }}\n"
                f"  template <int R, bool RK, int KS> __device__ __forceinline__ void convert(const char* src, "
                f"__nv_bfloat16* dst, __nv_bfloat16* out, long long off, long long sr, long long sk, int r0, int k) const {{\n"
                f"    {convert}\n  }}\n}};")
        return f"{functor_types[key]}{{{', '.join(names)}}}"

    absorbed_by = {}
    for cast, decision in absorbed.items():
        for product, which in decision["readers"]:
            absorbed_by.setdefault(product, {})[which] = cast

    p7b_claims = [0]

    p7b_dyn = [0]

    def p7b_dynamic(code):
        """P7-B dynamic items: static item loops (rotated round robin over the grid, segment units, one-CTA items) claim
        their items from a per-launch zeroed counter, so CTAs that finish their other work early take more items.
        Same item bodies, same per-item arithmetic."""
        if not p7b_dynamic_items:
            return code

        def counter():
            p7b_dyn[0] += 1
            return f"latch_dc{p7b_dyn[0]}"

        def item_loop(match):
            var, bound = match.group(1), match.group(2)
            c = counter()
            return (f"unsigned* {c} = /*P7B_CLAIM*/;\n  for (unsigned {var} = latch_p7b_claim({c}, {bound}); {var} < {bound}; "
                    f"{var} = latch_p7b_claim({c}, {bound}))")
        code = re.sub(r"for \(unsigned (it) = \(blockIdx\.x \+ \d+u\) % gridDim\.x; it < (\d+u); it \+= gridDim\.x\)",
                      item_loop, code)
        code = re.sub(r"for \(unsigned (latch_u|s) = blockIdx\.x; (?:latch_u|s) < (\d+u); (?:latch_u|s) \+= gridDim\.x\)",
                      item_loop, code)
        code = re.sub(r"if \(blockIdx\.x == \d+u % gridDim\.x\)",
                      lambda m: "if (latch_p7b_claim(/*P7B_CLAIM*/, 1u) == 0u)", code)
        if P7B_CLAIM_HELPER not in helper_parts:
            helper_parts.insert(0, P7B_CLAIM_HELPER)
        return code

    def p7b_number_claims(code, k):
        """P7-B: unique claim counters (zeroed with the split-K counters at entry; never reused by another phase)."""
        while "/*P7B_CLAIM*/" in code:
            code = code.replace("/*P7B_CLAIM*/", f"/*P7B_CLAIM_{p7b_claims[0]}*/", 1)
            p7b_claims[0] += 1
        return code

    phase_source, phase_records = [], []
    for k, phase in enumerate(phases):
        parts, gemm_offset, rotate = [], 0, 0
        matrix_group = None
        # P-OPS4-A items run after the phase's other heavy items; the existing emission sees only the others.
        p4a_here = [n for n in phase["heavy"] if p4a_item(nodes[n])]
        full_heavy = phase["heavy"]
        phase = dict(phase, heavy=[n for n in full_heavy if n not in p4a_here])
        p7b_after, p7b_before = [], []
        if p7b_companions or p7b_attention_products:
            # P7-B companions run after the phase's matrix tasks (block barrier between items), like P-OPS4-A items;
            # a hosting fused attention item runs before them.
            specs_here = [n for n in phase['heavy'] if nodes[n].extra.get('matrix_spec')]
            if specs_here and len(specs_here) < len(phase['heavy']):
                p7b_after = [n for n in phase['heavy'] if n not in specs_here]
                p7b_before = ([n for n in p7b_after if nodes[n].kind.startswith("attention_mixed")
                               and nodes[n].kind != "attention_mixed_bias"] if p7b_attention_products else [])
                p7b_after = [n for n in p7b_after if n not in p7b_before]
                phase = dict(phase, heavy=specs_here)
        group_specs = [nodes[name].extra.get('matrix_spec') for name in phase['heavy']]
        phase_epilogues = phase.get('epilogue', {})
        if len(phase['heavy']) == 1 and phase['heavy'][0] in p7b_rowepi:
            from .matrix_rules_p7b import _code as matrix_code, row_plan
            P_ = phase['heavy'][0]
            spec_ = nodes[P_].extra['matrix_spec']
            plan_ = row_plan(spec_)
            epi_ = {"code": p7b_segment_code(p7b_rowepi[P_], tile_product=P_), "store": True, "rows": True,
                    "staged": True}
            parts.append(f"// gemm + P7-B row epilogue {p7b_rowepi[P_]}: {P_}\n" +
                         matrix_code(spec_, plan_, _ident, 0, 1, 1, 0, epi_))
            gemm_offset = plan_['tiles']
        elif group_specs and all(group_specs) and (phase_epilogues or pops2b_plans
                                                 or any(n in absorbed_by for n in phase['heavy'])):
            from .pops2b_rules_p7b import HEADER as EPI_HEADER, emit_group as emit_epilogue_group
            if EPI_HEADER not in extra_headers:
                extra_headers.append(EPI_HEADER)
                helper_parts.append(f'#include "{EPI_HEADER}"')
            extras = []
            for name in phase['heavy']:
                extra = {}
                if name in phase_epilogues:
                    staged_ = (p7b_vec_epilogue_code(name, phase_epilogues[name])
                               if p7b_staged_epilogue and nodes[name].extra['matrix_spec'].get('tma')
                               and nodes[name].extra['matrix_spec'].get('wg') else None)
                    if name in colsum_of and staged_ is None:
                        raise LoweringError(f"internal: P9-E epilogue column sum of {name} lost its staged epilogue")
                    extra['epi'] = (staged_ if staged_ is not None else
                                    pair_epilogue_code(name, phase_epilogues[name])
                                    if nodes[name].extra['matrix_spec'].get('wg') else
                                    epilogue_code(name, phase_epilogues[name]))
                for which, cast in absorbed_by.get(name, {}).items():
                    extra['p' + which] = prologue_code(cast, name, which)
                extras.append(extra or None)
            code, matrix_group = emit_epilogue_group(group_specs, extras, ident=_ident, plans=pops2b_plans)
            code = p7b_number_claims(code, k)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
            for definition in matrix_group.pop('p7b_tables', ()):
                if definition not in helper_parts:
                    helper_parts.append(definition)
        elif group_specs and all(group_specs):
            from .matrix_rules_p7b import emit_group
            code, matrix_group = emit_group(group_specs, ident=_ident)
            code = p7b_number_claims(code, k)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
            for definition in matrix_group.pop('p7b_tables', ()):
                if definition not in helper_parts:
                    helper_parts.append(definition)
        else:
            engines = {nodes[n].extra.get("engine"): n for n in phase["heavy"]}
            for position, name in enumerate(phase["heavy"]):
                node = nodes[name]
                code = p7b_segment_code(name) if name in p7b_groups else heavy_code(node)
                if node.kind not in ("gemm",) and not node.kind.startswith("attention"):
                    code = p7b_number_claims(p7b_dynamic(code), k)
                engine = node.extra.get("engine")
                if engine == "pops3c_row":
                    from .norm_rules import GRID_CAP, warps_per_row
                    members = phase.get("rows", {}).get(name, [])
                    code = code.replace("/*ROW_MEMBERS*/", row_members_code(name, members) if members else "")
                    blocks = GRID_CAP
                    if "pops3c_col" in engines and not pops3c_overlap:
                        groups = nodes[engines["pops3c_col"]].extra["column_groups"]
                        code = code.replace("/*ROW_LO*/0u", f"min({groups}u, gridDim.x / {pops3c_col_div}u)")
                        blocks = GRID_CAP - min(groups, GRID_CAP // pops3c_col_div)
                    wr = warps_per_row(node.extra["row_rows"], blocks)
                    code = code.replace("/*ROW_WR*/8u", f"{wr}u").replace("/*ROW_W*/", str(wr))
                    if p6b_row_registers and wr == 1:
                        from .norm_rules import single_warp_rows
                        code = single_warp_rows(code)
                elif engine == "pops3c_col" and "pops3c_row" in engines and not pops3c_overlap:
                    groups = node.extra["column_groups"]
                    code = code.replace(f"/*COL_SPAN*/min({groups}u, gridDim.x)",
                                        f"min({groups}u, gridDim.x / {pops3c_col_div}u)")
                if name in deferred:
                    info = deferred[name]
                    spec = node.extra["matrix_spec"]
                    if name in phase.get("epilogue", {}):
                        from .pops2b_rules_p7b import HEADER as EPI_HEADER, _code as epi_matrix_code, tile_plan
                        if EPI_HEADER not in extra_headers:
                            extra_headers.append(EPI_HEADER)
                            helper_parts.append(f'#include "{EPI_HEADER}"')
                        extra = {"epi": epilogue_code(name, phase["epilogue"][name])}
                        code = epi_matrix_code(spec, tile_plan(spec, info["plan_grid"], extra), _ident,
                                               *info["slice"], 0, extra)
                    else:
                        from .matrix_rules_p7b import _code as matrix_code, _plan as matrix_plan
                        code = matrix_code(spec, matrix_plan(spec, info["plan_grid"]), _ident, *info["slice"], 0)
                if position:
                    parts.append("__syncthreads();  // shared staging reuse between co-scheduled heavy items")
                split_need = node.extra.get('p2_split') if node.extra else None
                if split_need:
                    if '/*P2_CNT*/' not in code or '/*P2_PART*/' not in code:
                        raise LoweringError("internal: split product without scratch placeholders")
                    p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                    p2_part_cursor = phase.setdefault('_p2_part', 0)
                    code = code.replace('/*P2_CNT*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + p2_cnt_cursor}LL)')
                    code = code.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {p2_part_cursor}LL)')
                    phase['_p2_cnt'] = p2_cnt_cursor + split_need['counter_words']
                    phase['_p2_part'] = p2_part_cursor + split_need['partial_words']
                    p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
                    p2_partial_words = max(p2_partial_words, phase['_p2_part'])
                    if p2_first_split_phase is None:
                        p2_first_split_phase = k
                if '/*B_VBASE*/' in code:
                    code = code.replace('/*B_VBASE*/', str(gemm_offset))
                elif node.kind == "gemm" and len(phase["heavy"]) > 1 and gemm_offset:
                    if code.count("smem);") != 1:
                        raise LoweringError("internal: co-scheduled matrix phase needs one pinned body call")
                    code = code.replace("smem);", f"smem, {gemm_offset});")
                if node.kind == "gemm" and name not in deferred:
                    gemm_offset += int(node.extra.get("tiles", 0))
                parts.append(f"// {node.kind}: {name}\n{{\n{code}\n}}")
        if p7b_before:
            def hosted_code(n):
                # P9-E: a hosting attention item's split scratch placeholders get this phase's cursor like any item
                code_ = heavy_code(nodes[n])
                need_ = nodes[n].extra.get('p2_split') if nodes[n].extra else None
                if need_:
                    c_ = phase.setdefault('_p2_cnt', 0)
                    q_ = phase.setdefault('_p2_part', 0)
                    code_ = code_.replace('/*P2_CNT*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + c_}LL)')
                    code_ = code_.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {q_}LL)')
                    phase['_p2_cnt'] = c_ + need_['counter_words']
                    phase['_p2_part'] = q_ + need_['partial_words']
                    nonlocal_max['cnt'] = max(nonlocal_max['cnt'], phase['_p2_cnt'])
                    nonlocal_max['part'] = max(nonlocal_max['part'], phase['_p2_part'])
                    nonlocal_max['first'] = k if nonlocal_max['first'] is None else nonlocal_max['first']
                return code_
            nonlocal_max = {'cnt': 0, 'part': 0, 'first': None}
            first = [f"// {nodes[n].kind} (before the matrix tasks): {n}\n{{\n{hosted_code(n)}\n}}" for n in p7b_before]
            p2_counter_words = max(p2_counter_words, nonlocal_max['cnt'])
            p2_partial_words = max(p2_partial_words, nonlocal_max['part'])
            if nonlocal_max['first'] is not None and p2_first_split_phase is None:
                p2_first_split_phase = nonlocal_max['first']
            if p9e_attention_last:
                # P9-E: the statically distributed matrix tasks first, then the attention item whose units are
                # claimed dynamically, so CTAs that finish their products early take more attention units.
                parts += ["__syncthreads();  // P9-E hosted attention after the matrix tasks"] + first
            else:
                parts[:0] = first + ["__syncthreads();  // P7-B matrix tasks after the hosting attention item"]
        for name in p7b_after:
            node = nodes[name]
            code = p7b_segment_code(name) if name in p7b_groups else heavy_code(node)
            if not node.kind.startswith("attention"):
                code = p7b_number_claims(p7b_dynamic(code), k)
            split_need = node.extra.get('p2_split') if node.extra else None
            if split_need:
                p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                p2_part_cursor = phase.setdefault('_p2_part', 0)
                code = code.replace('/*P2_CNT*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + p2_cnt_cursor}LL)')
                code = code.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {p2_part_cursor}LL)')
                phase['_p2_cnt'] = p2_cnt_cursor + split_need['counter_words']
                phase['_p2_part'] = p2_part_cursor + split_need['partial_words']
                p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
                p2_partial_words = max(p2_partial_words, phase['_p2_part'])
                if p2_first_split_phase is None:
                    p2_first_split_phase = k
            parts.append("__syncthreads();  // P7-B companion after the matrix tasks")
            parts.append(f"// {node.kind} (companion): {name}\n{{\n{code}\n}}")
        matrix_only = phase["heavy"] and all(nodes[n].kind == "gemm" for n in phase["heavy"]) and not p7b_after
        exclude = gemm_offset if matrix_only else 0
        main_heavy = [n for n in phase["heavy"] if n not in deferred]
        if (pops2b and pops2b_idle and target_arch == "sm_90" and main_heavy and not matrix_only
                and all(nodes[n].kind in IDLE_HEAVY_KINDS and nodes[n].extra.get("work_items") for n in main_heavy)):
            exclude = sum(int(nodes[n].extra["work_items"]) for n in main_heavy)
        for name in p4a_here:
            if parts:
                parts.append("__syncthreads();  // shared memory reuse before a P-OPS4-A item")
            spec = nodes[name].extra.get('p4a_spec')
            code = p4a_emit(name, spec) if spec else heavy_code(nodes[name])
            code = p7b_number_claims(p7b_dynamic(code), k)
            split_need = nodes[name].extra.get('p2_split')
            if split_need:
                # Last-arriver items: arrival counters and partials in this phase's split scratch (as split-K).
                p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                p2_part_cursor = phase.setdefault('_p2_part', 0)
                code = code.replace('/*P2_CNT*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + p2_cnt_cursor}LL)')
                code = code.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {p2_part_cursor}LL)')
                phase['_p2_cnt'] = p2_cnt_cursor + split_need['counter_words']
                phase['_p2_part'] = p2_part_cursor + split_need['partial_words']
                p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
                p2_partial_words = max(p2_partial_words, phase['_p2_part'])
                if p2_first_split_phase is None:
                    p2_first_split_phase = k
            parts.append(f"// pops4a: {name}\n{{\n{code}\n}}")
        rotation = 0
        entries = []
        for numel, members in phase["loops"].items():
            split = (p6b_loop_balance and (not exclude or p6b_rotate_exclude)) or p6b_multi_tensor_loops
            groups = p6b_loop_components(members, k) if split else [members]
            entries.extend((numel, group) for group in groups)
        if p6b_chunk_multi_loops and not exclude:
            multi, entries = p6b_chunk_multi(entries)
            parts.extend(multi)
        if p8e_small_multi and not p6b_multi_tensor_loops:
            # P8-E: components below the vector threshold (scalar step/bias-correction chains, small bias tensors)
            # sharing one element functor become one table-driven grid-stride loop instead of straight-line batched
            # blocks (instruction-fetch bound); same per-element expressions and roundings.
            # Measured (P8-E gpu-03): merging sub-vector tensors (256..4096 elements) adds table/barrier overhead to
            # many phases (+59 us); only one-element scalar chains are merged.
            small = [(numel, group) for numel, group in entries if numel == 1]
            if len(small) >= P6B_MULTI_MIN:
                multi, rest = p6b_multi_tensor(small, exclude, allow_scalar=True)
                parts.extend(multi)
                taken = {id(g) for _, g in small} - {id(g) for _, g in rest}
                entries = [(numel, group) for numel, group in entries if id(group) not in taken]
        if p6b_multi_tensor_loops:
            multi, entries = p6b_multi_tensor(entries, exclude)
            parts.extend(multi)
            if not p6b_loop_balance:
                # Components not taken by a multi-tensor loop stay together in their original loop.
                rejoined = {}
                for numel, group in entries:
                    rejoined.setdefault(numel, []).extend(group)
                order_ = {n: j for j, n in enumerate(m for ms in phase["loops"].values() for m in ms)}
                entries = [(numel, sorted(group, key=order_.get)) for numel, group in rejoined.items()]
        batch, heavy_batch = [], []

        pending = []                      # P7-B pass prefetch: groups rendered after the phase's loops are known

        def flush_batch():
            if batch:
                if p7b_pass_prefetch:
                    pending.append((len(parts), list(batch)))
                    parts.append(None)
                else:
                    parts.append(p6b_batch_code(list(batch)))
                batch.clear()

        def flush_heavy():
            if heavy_batch:
                if p7b_pass_prefetch:
                    pending.append((len(parts), list(heavy_batch)))
                    parts.append(None)
                else:
                    parts.append(p6b_batch_code(list(heavy_batch)))
                heavy_batch.clear()
        for numel, group in entries:
            if True:
                rotate_ = rotation if p6b_loop_balance and (not exclude or p6b_rotate_exclude) else 0
                if p6b_loop_batch and (p6b_loop_batch in (True, "light") or len(group) <= int(p6b_loop_batch)):
                    # p6b_loop_batch: True batches every single-pass loop; an integer only loops of at most that many
                    # values (copy/cast loops); "light" loops without divisions, square roots or transcendentals
                    # (computation-heavy loops keep their own pass).
                    piece = loop_code(numel, group, exclude, rotate=rotate_,
                                      pieces="all" if p6b_loop_batch_unrolled else True)
                    heavy_body = isinstance(piece, dict) and p6b_loop_batch == "light" and re.search(
                        r"fdiv|sqrt|exp|log|tanh|erf|pow", piece["body"])
                    if (heavy_body and p7b_heavy_batch and loop_code_unrolled(numel, group)
                            and isinstance(piece, dict)):
                        # P7-B: large computation-heavy (memory-bound) loops batched in pairs under their own
                        # live-value budget, so a thread keeps several tensors' loads in flight (same expressions).
                        if heavy_batch and sum(b["live"] for b in heavy_batch) + piece["live"] > int(p7b_heavy_batch):
                            flush_heavy()
                        heavy_batch.append(piece)
                    elif heavy_body and p6b_loop_batch_heavy and not loop_code_unrolled(numel, group):
                        # P6-B: computation-heavy single-pass loops batched in smaller groups (live-value budget
                        # p6b_loop_batch_heavy) so a thread overlaps their memory round trips.
                        if heavy_batch and sum(b["live"] for b in heavy_batch) + piece["live"] > int(p6b_loop_batch_heavy):
                            flush_heavy()
                        heavy_batch.append(piece)
                    elif heavy_body:
                        flush_batch()
                        parts.append(loop_code(numel, group, exclude, rotate=rotate_))
                    elif isinstance(piece, dict):
                        if batch and sum(b["live"] for b in batch) + piece["live"] > P6B_BATCH_LIVE:
                            flush_batch()
                        batch.append(piece)
                    else:
                        parts.append(piece)
                else:
                    parts.append(loop_code(numel, group, exclude, rotate=rotate_))
                lanes = 4 if (numel % 4 == 0 and numel >= VECTOR_MIN_ELEMENTS) else 1
                # blocks of the 128-CTA grid; P6-B p6b_rotate_grid: of the resident grid (132 SMs x residency)
                rotation = (rotation + -(-(numel // lanes) // 256)) % (132 * matrix_residency if p6b_rotate_grid
                                                                        else 128)
        flush_batch()
        flush_heavy()
        for k_, (slot, group_) in enumerate(pending):
            parts[slot] = p6b_batch_code(group_, prefetch=pending[k_ + 1][1] if k_ + 1 < len(pending) else None)
        phase = dict(phase, heavy=full_heavy)
        heavy_nodes = [nodes[n] for n in phase["heavy"]]
        loop_names = [n for members in phase["loops"].values() for n in members]
        loop_names += [n for members in phase.get("rows", {}).values() for n in members]
        epilogue_names = [n for members in phase.get("epilogue", {}).values() for n in members]
        epilogue_names += [cast for name in phase["heavy"] for cast in absorbed_by.get(name, {}).values()
                           if absorbed[cast]["mode"] == "materialize"]
        kind = (next((h.kind for h in heavy_nodes if h.kind not in ("gemm", "pops4a")), None)
                or ("gemm" if any(h.kind == "gemm" for h in heavy_nodes) else None)
                or (heavy_nodes[0].kind if heavy_nodes else "elementwise"))
        all_names = [*phase["heavy"], *epilogue_names, *loop_names,
                     *[m for h in phase["heavy"] for m in p7b_groups.get(h, {}).get("members", ())]]
        reads = []
        for name in all_names:
            for root, layout, shape, idx in loads_of.get(name, ()):
                if epilogue_of.get(name) == root or attached_to.get(name) == root:
                    continue
                if not (shape is not None and register_load(nodes[name], root, layout, shape, idx)
                        and phase_of.get(root) == k):
                    reads.append(root)
        record = {"kind": kind, "names": all_names,
                  "targets": [nodes[n].target for n in all_names],
                  "reads": list(dict.fromkeys(reads)),
                  "writes": [n for n in all_names if (nodes[n].kind != "elementwise" or n in stored)
                             and n not in elided_products and n not in p7b_groups],
                  "items": [{"kind": h.kind, "name": h.name,
                             **{key: h.extra[key] for key in ("tiles", "work_items", "group", "source_commit",
                                                             "packed_outputs", "packed_elements", "phase_writes")
                                if key in h.extra}}
                            for h in heavy_nodes]
                           + [{"kind": "epilogue", "product": product, "names": list(members),
                               "product_store_elided": product in elided_products}
                              for product, members in phase.get("epilogue", {}).items()]
                           + [{"kind": "operand_cast", "name": cast, "product": name, "operand": which,
                               "mode": absorbed[cast]["mode"]}
                              for name in phase["heavy"] for which, cast in absorbed_by.get(name, {}).items()]
                           + [{"kind": "elementwise", "elements": numel, "names": members}
                              for numel, members in phase["loops"].items()]
                           + [{"kind": "row_members", "reduction": owner, "names": list(members)}
                              for owner, members in phase.get("rows", {}).items()]}
        if not heavy_nodes:
            shapes = [nodes[n].shape for n in loop_names]
            record["shape"] = list(max(shapes, key=_numel))
            record["dtype"] = nodes[loop_names[0]].dtype
        for h in heavy_nodes:
            for key in ("tiles", "reduction_elements", "engine", "batch", "arithmetic", "work_items",
                        "selection_grid", "requires_full_grid", "matrix_plan", "donor_commit"):
                if key in h.extra and key not in record:
                    record[key] = h.extra[key]
        if sum(1 for h in heavy_nodes if h.kind == "gemm") > 1:
            record["tiles"] = sum(int(h.extra.get("tiles", 0)) for h in heavy_nodes if h.kind == "gemm")
        if any("shared_bytes" in h.extra for h in heavy_nodes):
            record["shared_bytes"] = max(h.extra.get("shared_bytes", 0) for h in heavy_nodes)
        if matrix_group is not None:
            record.update(matrix_group)
        phase_records.append(record)
        # P6-B diagnostic: each block's arrival time at the phase's closing barrier (profile launches only).
        arrival = (f"\nif (ticks) {{ __syncthreads(); if (threadIdx.x == 0) ticks[{P6B_TICK_BASE + k * P6B_TICK_STRIDE}u"
                   f" + blockIdx.x] = stamp(); }}") if p6b_block_ticks else ""
        phase_source.append(f"// Phase {k}: {', '.join(all_names)}\n{{\n" + scoped("\n".join(parts))
                            + f"\n}}{arrival}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{k+1}] = stamp();")
    prefetches = []
    if pops2b and pops2b_prefetch and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7b import prefetch_plan
        for k, chunks in prefetch_plan(phase_records, inputs, layouts, dtypes, _BYTES).items():
            code = ["{ // P-OPS2-B: bulk L2 prefetch of phase %d's external read-only operands" % (k + 1),
                    "  const unsigned latch_t = blockIdx.x * blockDim.x + threadIdx.x, latch_n = gridDim.x * blockDim.x;"]
            for root, nbytes, chunk in chunks:
                count = -(-nbytes // chunk)
                last = nbytes - (count - 1) * chunk
                code.append(f"  for (unsigned c = latch_t; c < {count}u; c += latch_n) {{\n"
                            f"    const char* p = reinterpret_cast<const char*>(args[{inputs.index(root)}]) + (size_t)c * {chunk}u;\n"
                            f"    const unsigned bytes = c + 1u == {count}u ? {last}u : {chunk}u;\n"
                            f"    asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(p), \"r\"(bytes) : \"memory\");\n  }}")
            code.append("}")
            header = f"// Phase {k}: "
            assert phase_source[k].startswith(header)
            first_brace = phase_source[k].index("\n{\n") + 3
            phase_source[k] = phase_source[k][:first_brace] + "\n".join(code) + "\n" + phase_source[k][first_brace:]
            prefetches.append({"phase": k, "for_phase": k + 1, "operands": [[r, b] for r, b, _ in chunks]})
    commit_code = []
    for target, source in writes.items():
        if target in in_place:
            continue
        n = layouts[target].numel
        value = memory_loader(source, layouts[target].shape, "i")
        value = ctype_round(value, dtypes[target])
        commit_code.append(f"for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {_u(n)}; i += gridDim.x * blockDim.x) "
                           f"{_ident(target)}[i] = {value};")
    tail = ("\n// All old-state reads have completed before any remaining state mutation.\n{\n" + scoped("\n".join(commit_code)) + "\n}"
            + "\n  grid.sync();\n" if commit_code else
            "\n// Every state write was committed in place after its last old-state read.\n  if (ticks) grid.sync();\n")
    if p7b_claims[0]:
        claim_base = p2_base + p2_counter_words
        for n_ in range(p7b_claims[0]):
            pointer = f"(reinterpret_cast<unsigned*>(arena + {claim_base + n_}LL))"
            phase_source = [code.replace(f"/*P7B_CLAIM_{n_}*/", pointer) for code in phase_source]
        p2_counter_words += -(-p7b_claims[0] // 64) * 64
        if p2_first_split_phase is None:
            p2_first_split_phase = 0
        p7b_records["claim_counters"] = p7b_claims[0]
    p2_prologue = ""
    if p2_counter_words:
        p2_partbase = f"{p2_base + p2_counter_words}LL"
        phase_source = [code.replace('/*P2_PARTBASE*/', p2_partbase) for code in phase_source]
        p2_prologue = (f"  // P-OPS2-A split-K arrival counters start at zero; every last arriver restores zero.\n"
                       f"  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {_u(p2_counter_words)}; "
                       f"i += gridDim.x * blockDim.x) reinterpret_cast<unsigned*>(arena + {p2_base}LL)[i] = 0u;\n"
                       + ("  grid.sync();\n" if p2_first_split_phase == 0 and bar_offset is None else ""))
        arena_words = p2_base + p2_counter_words + p2_partial_words
    barrier_setup = ""
    if bar_offset is not None:
        # Release/acquire arena-counter barrier between phases (E's S102G/bar2 adaptation, loop_pipeline); one
        # cooperative grid sync at entry zeroes the counter, so independent runtimes stay separate.
        from .loop_pipeline import COUNTER_BARRIER
        if COUNTER_BARRIER not in helper_parts:
            helper_parts.append(COUNTER_BARRIER)
        barrier_setup = (f"\n  unsigned long long* latch_bar = reinterpret_cast<unsigned long long*>(arena + {bar_offset}LL);"
                         "\n  latch_loop_bar_begin(latch_bar);")
        phase_source = [code.replace("\ngrid.sync();\n", "\nlatch_loop_barrier(latch_bar);\n") for code in phase_source]
    kernel_body = ("  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n" + "\n".join(declarations)
                   + p2_prologue + barrier_setup
                   + "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n" + "\n".join(phase_source)
                   + tail + f"  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{commit+1}] = stamp();\n")
    if vector_helpers_needed:
        if p6b_chunk_multi_loops or p7b_rowepi:
            helper_parts.insert(0, VECTOR_HELPERS)   # P6-B / P7-B: before element / row-epilogue functors
        else:
            helper_parts.append(VECTOR_HELPERS)
    # Canonical B functors may recur in independently lowered branch/loop
    # helper bundles. Guard each definition, not the complete bundle string.
    helper_parts = [
        (f"#ifndef {match.group(1).upper()}_DEFINED\n#define {match.group(1).upper()}_DEFINED\n"
         + part + "\n#endif\n")
        if (match := re.match(r"struct (latch_fn_[0-9a-f]+) ", part)) else part
        for part in helper_parts]
    if pops3c and pops3c_outline and not retain_all:
        # P-OPS3-C: one out-of-line copy of each libdevice transcendental per program instead of an inlined copy at
        # every unrolled use; same function, identical results, smaller per-phase instruction footprint.
        from .norm_rules import OUTLINED, outline
        kernel_body, used = outline(kernel_body)
        include = '#include "attention_fp32.cuh"\n'
        if pops3c_outline_headers and include in helper_parts:
            # Same out-of-line copies for the vendored attention body's libdevice calls (scoped macro; identical
            # functions and results).
            position = helper_parts.index(include)
            helper_parts[position] = (OUTLINED + "#define expf latch_expf\n#define logf latch_logf\n" + include
                                      + "#undef expf\n#undef logf\n")
        elif used or any(re.search(r"\blatch_(fdiv|sigmoid|silu|expf|logf|powf)\w*\(", part) for part in helper_parts):
            # Definitions precede every helper (generated functor structs may call them too).
            helper_parts.insert(0, OUTLINED)
    helpers = "\n".join(helper_parts)
    if p7b_inline_pure:
        # P7-B: the elementwise sigmoid/SiLU helpers inline into their (epilogue) callers instead of a call per element
        for fn in ("latch_sigmoid", "latch_silu", "latch_silu_backward"):
            helpers = helpers.replace(f"__device__ __noinline__ float {fn}(", f"__device__ __forceinline__ float {fn}(")
    if p7b_tma_arrive and "pops4b_tma.cuh" in extra_headers:
        # P7-B: TMA body stages refilled by the last warp to release them (no CTA barrier per k step).
        helpers = "#define POPS4B_LAST_ARRIVER 1\n" + helpers
        p7b_records["tma_last_arriver"] = True
    if p7b_claim_async and "pops4b_tma.cuh" in extra_headers:
        helpers = "#define POPS4B_CLAIM_ASYNC 1\n" + helpers
        p7b_records["tma_claim_async"] = True
    if p9e_epi_unroll and "pops4b_tma.cuh" in extra_headers:
        helpers = f"#define POPS4B_EPI_UNROLL {int(p9e_epi_unroll)}\n" + helpers   # P9-E staged-epilogue unroll
        p7b_records["tma_epilogue_unroll"] = int(p9e_epi_unroll)
    if p7b_tma_persist and "pops4b_tma.cuh" in extra_headers:
        # P7-B: stage mbarriers initialized once per launch; phase parity carried across product calls.
        helpers = "#define POPS4B_PERSIST 1\n" + helpers
        kernel_body = kernel_body.replace("\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n",
                                          "\n  pops4b::tma_persist_init();\n"
                                          "  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n", 1)
        p7b_records["tma_persistent_barriers"] = True
    if p6b_trap_asserts and any(h in extra_headers for h in ("pops4b_wg.cuh", "pops4b_tma.cuh")):
        # P6-B: device assert() lowers to an __assertfail call, and a call anywhere in the entry function serializes
        # every warpgroup MMA (ptxas C7510, P-OPS5-C). In programs with wgmma bodies the generated index checks trap
        # directly instead (same abort on an out-of-range index, without the message).
        kernel_body, trapped = re.subn(r"\bassert\(([^;]*)\);", r"if (!(\1)) __trap();", kernel_body)
        p6b_records["trapped_asserts"] = trapped
    shared_bytes = max([17408] + [p.get("shared_bytes", 0) for p in phase_records])
    from .resources import select_program_resources
    resource_policy = select_program_resources(shared_bytes, phases=phase_records, arch=target_arch,
                                               pops2_two_cta=matrix_residency >= 2, residency=matrix_residency)
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    if p8e_fast_tanh and "tanhf(" in source:
        # P8-E declared precision option (SELECTION amendment A1): hardware tanh.approx.f32 for generated tanh.
        helper = ("__device__ __forceinline__ float latch_tanh_approx(float x) {\n"
                  "  float y; asm(\"tanh.approx.f32 %0, %1;\" : \"=f\"(y) : \"f\"(x)); return y;\n}\n")
        uses = source.count("tanhf(")
        source = source.replace("tanhf(", "latch_tanh_approx(")
        marker = "namespace cg = cooperative_groups;\n"
        if marker not in source:
            raise LoweringError("fast tanh: translation-unit prologue not found")
        source = source.replace(marker, marker + helper, 1)
        p7b_records["fast_tanh"] = {"replaced_calls": uses, "instruction": "tanh.approx.f32"}
    from .matrix_rules_p7b import collect_tensor_maps
    source, tensor_maps = collect_tensor_maps(source, len(inputs))
    header_hashes = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
                     for name in ("core.cuh", "gemm.cuh", "epi.cuh", "plan.py", *extra_headers)}
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    aliases = {name: {**layout.to_dict(), "dtype": dtypes[name]} for name, layout in layouts.items()
               if layout.root in has_storage}
    elementwise_names = [n for n in nodes if nodes[n].kind == "elementwise"]
    manifest = {
        "attention_selection_policy": attention_policy,
        "counter_barrier": {"enabled": bar_offset is not None, "arena_word_offset": bar_offset,
                            "initialization": "per-launch cooperative entry" if bar_offset is not None else None},
        "matrix_scratch": {"counter_word_offset": p2_base if p2_counter_words else None,
                           "counter_words": p2_counter_words, "partial_words": p2_partial_words},
        "format": "latch-ops-cuda-v1", "schedule": "fused-v1" + ("+matrix-coschedule" if coschedule_matrix else ""),
        "source_hash": source_hash, "header_hashes": header_hashes,
        "inputs": inputs, "outputs": outputs, "structured_outputs": structured_outputs,
        "input_roles": {name: {"role": _field(values[name], "role", "input"), "parameter": _field(values[name], "parameter")} for name in inputs},
        "state_writes": writes, "in_place_commits": in_place,
        "copied_commits": {t: s for t, s in writes.items() if t not in in_place},
        "layouts": aliases,
        "offsets": offsets, "arena_words": arena_words, "arena_bytes": 4 * arena_words,
        "lifetimes": classes, "retained_all": retain_all, "views": views,
        "undefined_metadata_only": sorted(undefined),
        "materialization": {"elementwise": len(elementwise_names), "inlined": sorted(inline),
                            "register_only": sorted(n for n in elementwise_names if n not in inline and n not in stored),
                            "stored": len([n for n in elementwise_names if n in stored])},
        "phases": phase_records,
        "matrix_decisions": matrix_decisions,
        "pops4c": {"enabled": bool(pops4c), "attention": pops4c_attention},
        "matrix_operands": [{**row, 'engine': next((nodes[key].extra.get('engine') for key in (row['output'], row['output'] + '__splitk')
                                                    if key in nodes and nodes[key].extra), None)} for row in matrix_operands],
        **({"pops2b": {"matrix_epilogues": {g: list(m) for g, m in epilogues.items()},
                       "elided_product_stores": sorted(elided_products),
                       "operand_casts": absorbed, "l2_prefetch": prefetches, "deferred_products": deferred,
                       "rules": "attach_epilogues: identical per-operation expressions on rounded BF16 results; "
                                "select_absorbed_casts: BF16 casts materialized by their first product's staging"}}
           if pops2b and (epilogues or absorbed or deferred) else {}),
        **({"pops3c": {"row_engines": sorted(n for n in nodes if nodes[n].extra and nodes[n].extra.get("engine") == "pops3c_row"),
                       "column_engines": sorted(n for n in nodes if nodes[n].extra and nodes[n].extra.get("engine") == "pops3c_col"),
                       "attached": row_members,
                       "shared_phases": sum(1 for ph in phases if {"pops3c_row", "pops3c_col"} <= {
                           nodes[n].extra.get("engine") for n in ph["heavy"]}),
                       "rules": "frozen CTA reduction tree replayed by warp-per-row / lanes-over-columns engines; "
                                "row-statistic consumers and same-row siblings evaluated by the owning warp"}}
           if pops3c and any(nodes[n].extra and str(nodes[n].extra.get("engine", "")).startswith("pops3c") for n in nodes) else {}),
        "attention_groups": attention_decisions, "attention_refusals": attention_refusals,
        "ordered_index_gradients": index_decisions,
        **({"pops4a": {"reductions": p4a_state["decisions"],
                       "phases_with_items": sum(1 for ph in phases if any(p4a_item(nodes[n]) for n in ph["heavy"]))}}
           if p4a_state["decisions"] else {}),
        **({"selective_scans": scan_decisions} if scan_decisions else {}),
        **({"p6b": {k: v for k, v in p6b_records.items() if not k.startswith("_") and k != "tree_index"}}
           if p6b_records else {}),
        **({"p7b": dict(p7b_records, candidates=sorted(p7b_reds))} if p7b_segment else {}),
        **({"p9e_split_colsum": p9e_colsum_records} if p9e_colsum_records else {}),
        **({"p9e_ksum": p9e_ksum_records} if p9e_ksum else {}),
        "compiler_policy": "integrated_v1", "target_arch": target_arch,
        "phase_count": commit + 1, "barrier_count": commit + (1 if commit_code else 0),
        **({"tick_words": P6B_TICK_BASE + (commit + 1) * P6B_TICK_STRIDE,
            "p6b_block_ticks": {"base": P6B_TICK_BASE, "stride": P6B_TICK_STRIDE, "phases": commit}}
           if p6b_block_ticks else {}),
        "max_gemm_tiles": max([p.get("tiles", 0) for p in phase_records] or [1]),
        "shared_bytes": shared_bytes,
        **({"tensor_maps": tensor_maps} if tensor_maps else {}),
        **({"resource_policy": resource_policy} if resource_policy['has_popsb'] or resource_policy.get('pops2_two_cta') else {}),
        "numerical_policy": ("Explicit BF16 casts/operands/results; BF16 MMA with FP32 accumulation; FP32 master state; fixed reduction order; no TF32/data atomics"
                             if "bfloat16" in dtypes.values() else
                             "FP32 SIMT fmaf; admitted ordered split-K; separately rounded pointwise arithmetic; exact reduction tree; no TF32/data atomics"),
        "fusion_policy": "identical per-operation expressions; BF16 rounding retained when a buffer is elided; inline limit "
                         f"{INLINE_TOTAL_LIMIT}/{INLINE_WORK_LIMIT}; loads before stores; unroll <= {UNROLL_MAX}",
        "execution": f"one cooperative CUDA launch, 256 threads/CTA, {shared_bytes} shared bytes/CTA; occupancy admitted; in-place or final state commit",
        "memory_plan": "pinned plan.assign + independent plan.check; inclusive phase intervals and alias roots; elided values have no storage",
    }
    return GeneratedProgram(source, manifest, kernel_body, helpers)
