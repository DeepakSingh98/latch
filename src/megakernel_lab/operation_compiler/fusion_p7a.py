# Adapted from frozen P7-A f36b4737cee732e6915d11e233bcc068126b3e64; private scheduling/resource protocol.
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
P6A_PIPELINE_ITERS = 4                 # P6-A: planned grid-stride iterations per pipelined loop
P6A_PIPELINE_LIVE = 96                 # P6-A: values live per pipelined stage (two stages overlap)
RESIDENT_THREADS = 128 * 256          # default H100 launch cap x CTA width
P7_FLAT_MIN_LOADS = 2                 # P7-A: mean identity loads per flattened item
TRANSCENDENTAL = {"erf", "sin", "cos", "tanh", "exp", "expm1", "log1p", "reciprocal", "log", "sqrt", "rsqrt", "sigmoid", "silu", "silu_backward",
                  "pow", "div", "sigmoid_backward", "tanh_backward"}
EXPENSIVE_FUNCTIONS = {"erf", "sin", "cos", "tanh", "exp", "expm1", "log1p", "log", "sigmoid", "silu", "silu_backward",
                       "gelu", "gelu_backward", "pow"}   # P6-A leaf-materialization rule
INLINE_TOTAL_LIMIT = 12                # recomputed operations per inlined value use
INLINE_WORK_LIMIT = 24                 # summed over all uses
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
P7_VEC_HELPERS = r"""
#ifndef LATCH_P7_VEC8
#define LATCH_P7_VEC8
#include <cuda_bf16.h>
// P7-A vector epilogue helpers: 8 consecutive elements (16-byte aligned) as FP32 values.
__device__ __forceinline__ void latch_vld8(const float* p, float* v) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ void latch_vld8(const __nv_bfloat16* p, float* v) {
  const uint4 t = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&t);
#pragma unroll
  for (int e = 0; e < 4; ++e) { v[2 * e] = __bfloat162float(h[e].x); v[2 * e + 1] = __bfloat162float(h[e].y); }
}
__device__ __forceinline__ float latch_p7_bf(const __nv_bfloat162* v, int e) {
  return __bfloat162float((e & 1) ? v[e >> 1].y : v[e >> 1].x);
}
__device__ __forceinline__ void latch_vst8(float* p, const float* v) {
  reinterpret_cast<float4*>(p)[0] = make_float4(v[0], v[1], v[2], v[3]);
  reinterpret_cast<float4*>(p)[1] = make_float4(v[4], v[5], v[6], v[7]);
}
__device__ __forceinline__ void latch_vst8(__nv_bfloat16* p, const __nv_bfloat16* v) {
  uint4 t; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&t);
#pragma unroll
  for (int e = 0; e < 4; ++e) { h[e].x = v[2 * e]; h[e].y = v[2 * e + 1]; }
  *reinterpret_cast<uint4*>(p) = t;
}
#endif
"""
GRID_MODEL = 128                       # default launch cap used by the placement cost model
BYTES_PER_US = 2.6e6                   # modeled achievable HBM bytes/us (H100), whole grid
_TOKEN = re.compile(r"@@LOAD(\d+)@@")
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


_HOIST_ALLOWED = re.compile(r"^[\s\w.()+\-*/<>=?:!,\[\]]*$")
_HOIST_PURE = {"sqrtf", "rsqrtf", "expf", "expm1f", "logf", "log1pf", "erff", "tanhf", "fabsf", "fminf", "fmaxf",
               "powf", "latch_fdiv", "latch_expf", "latch_logf", "latch_powf", "__bfloat162float",
               "__float2bfloat16_rn", "min", "max"}


def _hoist_invariants(expressions, written, prefix):
    """P6-A: loop-invariant subexpressions of per-element expressions (pure math of literals and one-element `[0u]`
    loads of arrays the loop does not write) are evaluated once per thread. Returns (declarations, expressions).
    Values are unchanged: the same IEEE operations (--fmad=false) on the same operands, evaluated once."""
    declarations, cache = [], {}

    def hoistable(text):
        if "[0u]" not in text or not _HOIST_ALLOWED.match(text):
            return False
        stripped = re.sub(r"\bv_\w+\[0u\]", "S", text)
        stripped = re.sub(r"(?<![\w.])\d+\.?\d*(?:e[+-]?\d+)?[fuUL]*", "N", stripped)
        if "[" in stripped:
            return False
        for name in re.findall(r"\bv_\w+(?=\[0u\])", text):
            if name in written:
                return False
        return all(token in ("S", "N") or token in _HOIST_PURE for token in re.findall(r"[A-Za-z_]\w*", stripped))

    def rewrite(text):
        out, k = [], 0
        while k < len(text):
            if text[k] != "(":
                out.append(text[k])
                k += 1
                continue
            # The group: optional callee identifier already emitted, then a balanced parenthesis group.
            depth, j = 0, k
            while j < len(text):
                depth += text[j] == "("
                depth -= text[j] == ")"
                if depth == 0:
                    break
                j += 1
            group = text[k:j + 1]
            callee = re.search(r"([A-Za-z_]\w*)$", "".join(out))
            whole = (callee.group(1) + group) if callee else group
            if hoistable(whole):
                if whole not in cache:
                    cache[whole] = f"{prefix}{len(cache)}"
                    declarations.append(f"const float {cache[whole]} = {whole};")
                if callee:
                    del out[len(out) - len(callee.group(1)):]
                out.append(cache[whole])
            else:
                out.append("(" + rewrite(group[1:-1]) + ")")
            k = j + 1
        return "".join(out)
    return declarations, [rewrite(e) for e in expressions]


def _row_index_exact(text, rows, extent, element=False):
    """P6-A: True when the 32-bit index expression `text` over (prow, pcol) equals prow for every element of a
    [rows, extent] row engine (numeric proof over the whole index space; no symbolic reasoning)."""
    import numpy as np
    expr = text.replace("(unsigned)", "")
    expr = re.sub(r"(\d+)u\b", r"\1", expr).replace("/", "//")
    if re.search(r"[^\sprowcl0-9()+\-*/%]", expr) or re.search(r"[a-z]+", re.sub(r"\b(prow|pcol)\b", "", expr)):
        return False
    prow = np.arange(rows, dtype=np.int64)[:, None]
    pcol = np.arange(extent, dtype=np.int64)[None, :]
    try:
        value = eval(expr, {"__builtins__": {}}, {"prow": prow, "pcol": pcol})
    except Exception:
        return False
    expected = prow * extent + pcol if element else prow
    return bool(np.array_equal(np.broadcast_to(value, (rows, extent)), np.broadcast_to(expected, (rows, extent))))


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
                    forced, numel, idle_heavy=None, deferred=None, preload=None, fixed=None):
    """Move elementwise register chains later into phases with idle matrix blocks.

    A chain (items joined by same-loop register reads) may occupy any phase from
    its ASAP phase to just before its first external reader (or the last phase).
    Matrix-only phases using at most half the modeled grid leave idle blocks; a
    loop placed there runs only on those blocks and is hidden up to the modeled
    matrix time. Moves are later-only and re-validated by _check_schedule.
    """
    size = {"float32": 4, "bfloat16": 2, "int64": 8, "bool": 1}
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
                    permuted=None, row_of=None, row_before=None):
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
            if q == p and (root in (row_before or {}).get(node.name, ())
                           or row_of.get(root) in (row_before or {}).get(node.name, ())):
                continue                    # P6-A: row engine reads an earlier same-phase engine's stored values
            if q == p and (root in (row_before or {}).get(row_of.get(node.name), ())
                           or row_of.get(root) in (row_before or {}).get(row_of.get(node.name), ())):
                continue                    # P6-A: members of a row engine read earlier same-phase stored values
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
    from .matrix_codegen_p7a import select as small_select
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
                pops4a_floor=True, pops4a_over_exact="large", pops4a_last_column=False, pops4a_bf16=True, scan_engine="auto",
                p6a_rowchain=True, p6a_compact=True, p6a_leaf_materialize=True, p6a_hoist=True, p6a_split_loops=True, p6a_inline_traffic=True, p6a_row_batch=4, p6a_wg_epilogue=True, p6a_split_cap=128, p6a_full_grid=True, p6a_loop_ranges=False, p6a_bf16_vector=True, p6a_row_fuse=False, p6a_no_calls=True, p6a_epi_colsum=True, p6a_cat_elementwise=True, p6a_item_ticks=False, p6a_pipeline=False, p6a_two_cta=False, p6a_rotate=False,
                p7_rowreg=False, p7_staged_epi=False, p7_row_columns=False, p7_row_virtual=False,
                p7_flat_loops=False, p7_wg_indep=False, p7_diag=None,
                p7_staged_colsum=False, p7_const_args=False,
                p7_bulk_store=False, p7_bulk_load=False,
                p7_scalar_loads=False, p7_flat_table=0,
                p7_outline=False, p7_row_rb_cap=8,
                p7_pair_stores=False, p7_ring_release=False,
                p7_indep_smem=None, p7_scalar_block=False, p7_vec_epi=False, p7_vec_rb=4, p7_row_prefetch=False,
                p7_scalar_pin=False, p7_quad_sel=False, p7_vec_early=False, p7_col_tiles=None, p7_vec_raw=False,
                p7_vec_pipe=False, p7_bcast_traffic=False, p7_acc_rezero=False, p7_trans_store=0,
                p7_rowstat_remat=False, p7_scalar_par=False):
    # P-OPS5-D: fused mixed-precision short attention (ops/mixed_attention.py); pops5d=False reproduces the
    # previous lowering byte for byte. pops5d_store_probs (default) saves the forward's FP32 probabilities in a
    # fragment-native layout instead of recomputing them in the backward (memory for time; measured gpu-11).
    # P-OPS4-A (intaken by P6-A from 28f47a10f): P-OPS3-C norm rules, C-OPS2-C split reductions and the
    # hierarchical leading-axis reduction rule.
    # A selects actual FP32 products; B schedules explicit BF16 products. D's
    # engine dictionary remains available for staged accumulation/layout paths.
    matrix_engine = ("auto" if pops2_policy else "popsb") if matrix_engine is None else matrix_engine
    matrix_bodies = (("simt", "popsb") if pops2_policy else None) if matrix_bodies is None else tuple(matrix_bodies)
    attention_override = attention_min_keys is not None
    from .lower import (GeneratedProgram, LoweringError, TensorLayout, VENDOR, _header_path, _BYTES, _CTYPES,
                        _contiguous, _dtype, _field, _flatten_refs, _float, _double, _ident, _numel,
                        _ref, _reshape_strides, _translation_unit, slice_view, slice_backward_index)
    try:
        from . import scan_lower
    except ValueError:      # P6-A: see lower._scan_targets (local torch < 2.5 only)
        scan_lower = None

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
    # P6-A: programs with TMA warpgroup-MMA products contain no out-of-line call (ptxas serializes every wgmma of a
    # kernel that calls a function, C7510): no outlined math, force-inlined matrix phase functions.
    from . import matrix_rules_p7a as _mr
    no_calls = bool(p6a_no_calls and _mr.WGMMA and _mr.WGMMA_ALIGNED and _mr.WG_TMA and not retain_all)
    if no_calls:
        pops3c_outline = False
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
    p7_row_records = []
    p7_flat_records = []
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
    split_finalizers = set()        # P6-A split-K finalize nodes (never inlined: each use would re-sum the slices)

    def p4a_reduction(op, name, out, source, src, dims, kept, extent, results, refs, rounded, scale):
        """P-OPS4-A hierarchical leading-axis reduction (ops/pops4a_rules.py); False leaves the existing rules."""
        from . import pops4a_rules_p7a as R
        from . import norm_rules_p7a as norm_rules
        over_exact = (pops4a_over_exact is True
                      or (pops4a_over_exact == "large" and extent * results >= R.OVER_EXACT_MIN_ELEMENTS))
        exact_column = (pops3c and pops3c_column and not over_exact
                        and norm_rules.column_eligible(src.shape, src.strides, dims, results, extent))
        rotate = 37 * p4a_state["count"]
        part = f"{name}_p4a"
        if part in names or part in layouts:
            raise LoweringError(f"P-OPS4-A partial name collision: {part}")
        if "outline" not in p4a_state:
            p4a_state["bf16"] = any(str(_field(v, "dtype", "float32")) == "bfloat16" for v in values.values())
            # Spec bodies only in programs without BF16 values (no warpgroup MMA callers of outlined code).
            # P6-A: force-inlined spec bodies (pops4a_outline="inline") emit no outlined call, so BF16 programs
            # take the same selected bodies when pops4a_bf16 admits them.
            p4a_state["outline"] = bool(pops4a_outline) and (not p4a_state["bf16"]
                                                            or (pops4a_bf16 and pops4a_outline == "inline"))
            p4a_state["index"] = {output_name(o): k for k, o in enumerate(operations)}
        if pops4a == "auto" and p4a_state["bf16"] and not pops4a_bf16:
            # Automatic scope before P6-A: FP32 programs. pops4a_bf16=False keeps BF16 programs on their exact
            # (bitwise) engines; the P6-A default admits their FP32 reductions (FP32 reassociation only).
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
            if pops4a_last:
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
            if pops4a_last:
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
        from . import pops4a_rules_p7a as R
        from . import norm_rules_p7a as norm_rules
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
            vblocks_ = -(-results // (32 * 4 * vec))
            # P7-A: p7_col_tiles row tiles per column reduction (e.g. one per CTA so a row engine folding the
            # partials uses the whole grid); None keeps the P-OPS4-A item target.
            vplan = R.column_plan(extent, results, cols=4 * vec,
                                  target_items=(int(p7_col_tiles) * vblocks_ if p7_col_tiles else pops4a_items),
                                  floor=pops4a_floor)
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
                                                 bias_phase=pops5d_bias_phase, single_only=p6a_two_cta,
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
        if p6a_two_cta:
            # P6-A two-CTA programs: single-item attention schedules (their shared bytes fit two CTAs per SM; the
            # co-resident CTA supplies the overlap the producer/pipelined schedules provided).
            defines_.update(PMIX_PIPELINE=0, PMIX_PRODUCER=0)
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
        if scan_lower is not None and target in scan_lower.TARGETS:
            # One multi-result operation: packed typed result views + one exclusive heavy item.
            try:
                emission = scan_lower.emit(op, values=values, layouts=layouts, dtypes=dtypes, names=names,
                                           ref=_ref, field=_field, ident=_ident, engine=scan_engine)
            except scan_lower.ScanRefused as exc:
                raise LoweringError(f"{target}: {exc}") from exc
            layouts[emission.root] = TensorLayout((emission.root_words,), (1,), emission.root)
            dtypes[emission.root] = "float32"
            for result, result_shape, result_dtype, offset in emission.outputs:
                layouts[result] = TensorLayout(result_shape, _contiguous(result_shape), emission.root, offset)
                dtypes[result] = result_dtype
                views.append({"name": result, "target": target + ":packed_result", "source": emission.root})
            include = f'#include "{emission.header}"'
            if include not in helper_parts:
                helper_parts.append(include)
            if emission.header not in extra_headers:
                extra_headers.append(emission.header)
            add_heavy(emission.kind, op, emission.root, emission.code, emission.reads, **emission.metadata)
            scan_decisions.append(emission.decision)
            continue
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
        if base == "slice":
            layouts[name] = slice_view(layouts[_ref(args[0], names)], args, shape)
            views.append({"name": name, "target": target, "source": _ref(args[0], names)})
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
        if (p6a_cat_elementwise and base == "cat" and dtypes[name] in ("float32", "bfloat16") and len(shape) >= 1
                and isinstance(args[0], (list, tuple)) and all(_ref(a_, names) is not None for a_ in args[0])
                and all(dtypes[_ref(a_, names)] == dtypes[name] for a_ in args[0])):
            # P6-A: concatenation as an elementwise value (each element reads exactly one operand element), so it
            # fuses into its consumers (e.g. a row engine's leaf) instead of a separate copy phase.
            dim_ = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % len(shape)
            inner_, extent_ = _numel(shape[dim_ + 1:]), shape[dim_]
            pieces, start_ = [], 0
            for a_ in args[0]:
                ref_ = _ref(a_, names)
                sshape = tuple(layouts[ref_].shape)
                if len(sshape) != len(shape) or any(sshape[d] != shape[d] for d in range(len(shape)) if d != dim_):
                    raise LoweringError("cat operand shapes disagree outside the concatenated dimension")
                pieces.append((ref_, start_, sshape))
                start_ += sshape[dim_]
            if start_ != extent_:
                raise LoweringError("cat result extent mismatch")

            def cat_render(index, cb, pieces=pieces, inner_=inner_, extent_=extent_, dim_=dim_):
                c_ = f"(({index} / {inner_}u) % {extent_}u)"
                outer_ = f"({index} / {inner_ * extent_}u)"
                in_ = f"({index} % {inner_}u)"
                values = []
                for ref_, st_, sshape in pieces:
                    # Piece-local coordinate clamped into the piece: consumers may evaluate every piece's loads
                    # (e.g. hoisted pair-epilogue loads), so an unselected piece must still read in bounds.
                    len_ = sshape[dim_]
                    local = f"({c_} - {st_}u)" if st_ else c_
                    if st_:
                        local = f"({c_} < {st_}u ? 0u : {local})"
                    if st_ + len_ < extent_:
                        local = f"({c_} >= {st_ + len_}u ? {len_ - 1}u : {local})"
                    flat = f"({outer_} * {len_ * inner_}u + {local} * {inner_}u + {in_})"
                    values.append((st_ + sshape[dim_], cb({"ref": ref_}, flat, sshape)))
                expression = values[-1][1]
                for end_, value in reversed(values[:-1]):
                    expression = f"({c_} < {end_}u ? {value} : {expression})"
                return expression
            nodes[name] = _Node(name=name, kind="elementwise", shape=shape, dtype=dtypes[name], render=cat_render,
                                base="cat", target=target, own_cost=1, loads=[p_[0] for p_ in pieces],
                                order=len(nodes))
            continue
        emitted = emit_operation(target, args, kwargs, name, shape, layouts, dtypes, names, placeholder_read,
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
                from .matrix_codegen_p7a import select
                from .matrix_rules_p7a import HEADER, emit_group
                spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch)
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                    from .matrix_rules_p7a import spec_headers
                    for header in spec_headers(spec):
                        if header not in extra_headers:
                            extra_headers.append(header)
                        if f'#include "{header}"' not in helper_parts:
                            helper_parts.append(f'#include "{header}"')
                    if spec.get('splitk'):
                        # P6-A split-K: FP32 K-slice partials (one heavy product item over slices) and an elementwise
                        # finalize named as the product (slice-order FP32 sum, one BF16 rounding); downstream
                        # elementwise consumers join the finalize's loop.
                        ks, mm_, nn_ = spec['splitk'], spec['m'], spec['n']
                        part = f"{name}__skp"
                        if part in names or part in layouts:
                            raise LoweringError(f"split-K partial name collision: {part}")
                        layouts[part] = TensorLayout((ks * mm_ * nn_,), (1,), part)
                        dtypes[part] = "float32"
                        spec = dict(spec, out=_ident(part))
                        code, extra = emit_group([spec], ident=_ident)
                        add_heavy('gemm', op, part, code, refs, k=spec['k'], matrix_spec=spec, **extra)
                        size_ = mm_ * nn_

                        def split_render(index, cb, part=part, ks=ks, size_=size_):
                            terms = [cb({"ref": part}, f"({s_ * size_}u + {index})" if s_ else f"{index}",
                                        (ks * size_,)) for s_ in range(ks)]
                            total = terms[0]
                            for term in terms[1:]:
                                total = f"({total} + {term})"
                            return total
                        nodes[name] = _Node(name=name, kind="elementwise", shape=shape, dtype=dtypes[name],
                                            render=split_render, base="splitk_sum", target="p6a.splitk_sum",
                                            own_cost=ks, loads=[part], order=len(nodes))
                        split_finalizers.add(name)
                        matrix_decisions[-1].update(splitk=ks)
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
                code, extra = emit(a, b, out, la, lb, shape)
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
                                           residency=matrix_residency)
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
                from .matrix_codegen_p7a import fp32_async
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
                from .matrix_codegen_p7a import split_k as split_codegen
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
            terms = []
            for sequence, subshape, index in [(dims, reduction_shape, "r"), (kept, kept_shape, "o")]:
                for j, axis in enumerate(sequence):
                    terms.append(f"(({index} / {_numel(subshape[j+1:])}LL) % {src.shape[axis]}LL) * {src.strides[axis]}LL")
            pos = " + ".join(terms) or "0"
            scale = f" / {_float(_numel(reduction_shape))}" if base == "mean" else ""
            rounded_result = (lambda e: f"__float2bfloat16_rn({e})") if dtypes[name] == "bfloat16" else (lambda e: e)
            extent, results = _numel(reduction_shape), _numel(shape)
            if pops4a and not integer_sum and not retain_all and target_arch == "sm_90" and exact_reductions:
                emitted = p4a_reduction(op, name, out, source, src, dims, kept, extent, results, refs,
                                        rounded_result, scale)
                if emitted:
                    continue
            if (split_reductions and not integer_sum and extent >= SPLIT_MIN_EXTENT and results <= SPLIT_MAX_RESULTS):
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
                from . import norm_rules_p7a as norm_rules
                if pops3c_row and norm_rules.row_eligible(src.shape, src.strides, dims, results, extent):
                    leaf = storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pcol", "prow"))
                    code = norm_rules.row_code(out=out, rows=results, extent=extent, leaf=leaf,
                                               rounded=rounded_result, scale=scale)
                    add_heavy("reduction", op, name, code, refs, engine="pops3c_row", reduction_elements=extent,
                              row_rows=results, row_extent=extent, row_scale=scale, row_leaf=leaf,
                              row_store=f"{out}[g] = {rounded_result('second' + scale)};")
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
                selected = exact_reduction(out=out, position=pos, extent=extent, results=results,
                                           read=lambda position: storage_read(source, position),
                                           rounded=rounded_result, scale=scale)
                if selected is not None:
                    code, extra = selected
                    add_heavy('reduction', op, name, code, refs, **extra)
                    continue
            add_heavy("reduction", op, name, code, refs, reduction_elements=extent)
            continue

        if (pops3c and pops3c_factor and not retain_all and name not in protected_names
                and dtypes[name] in ("float32", "bfloat16")):
            # P-OPS3-C: a pointwise operation of a single broadcast (expanded) operand is evaluated once per
            # source element and itself viewed as the same expansion (identical per-element arithmetic).
            from .norm_rules_p7a import FACTORABLE
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

        def make_render(base=base, args=args, kwargs=kwargs, name=name, target=target, shape=shape):
            def render(idx, rd_ref):
                rd = lambda i: rd_ref(args[i], idx)
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
                    condition, position = slice_backward_index(args, shape, grad_shape, idx, unsigned=True, clamp=True)
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
    # P6-A: a transcendental-bearing value read by a P-OPS4-A partial leaf is stored once (FP32/BF16 as declared) and
    # streamed by the partial (vector loads) instead of being re-evaluated per element inside the scalar leaf.
    p4a_leaf_roots = set()
    if p6a_leaf_materialize:
        for node in nodes.values():
            if node.kind != "elementwise" and (node.extra or {}).get("pops4a"):
                for k in node.loads or ():
                    p4a_leaf_roots.add(root_of(token_loads[k][1]))
    transcendental, full_loads = {}, {}
    p7_trans_stored = set()
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        # Expensive libdevice functions only (sqrt/rsqrt/reciprocal are single fast instructions or short sequences).
        transcendental[node.name] = (node.base in EXPENSIVE_FUNCTIONS
                                     or any(transcendental.get(root_of(ref), False)
                                            for ref in node.loads if root_of(ref) in inline))
        cost = node.own_cost + sum(total_cost[root_of(ref)] for ref in node.loads if root_of(ref) in inline)
        total_cost[node.name] = cost
        if node.name in p4a_leaf_roots and transcendental[node.name]:
            continue
        if node.name in split_finalizers:
            continue
        if p7_trans_store and transcendental.get(node.name) and cost >= p7_trans_store and _numel(node.shape) > 1:
            # P7-A: an expensive transcendental closure (e.g. an activation derivative of a product output) is
            # materialized where its inputs are produced (an epilogue member of that product) and read by its
            # consumers, instead of being re-evaluated inside a later phase's epilogue. Same expression.
            total_cost[node.name] = cost
            p7_trans_stored.add(node.name)
            continue
        if p7_trans_store and any(root_of(ref) in p7_trans_stored for ref in node.loads) and _numel(node.shape) > 1:
            # Its direct full-size consumer stays a value of its own (an epilogue member of the consuming product),
            # so column sums over it keep fusing into that product's epilogue.
            continue
        # P6-A: full-size tensors read through this value's inline closure. Inlining costs uses x loads reads;
        # storing costs one write plus one read per use.
        own_numel = _numel(node.shape)
        closure = set()
        for ref in node.loads:
            r = root_of(ref)
            distinct = 1
            for d_, s_ in zip(layouts[ref].shape, layouts[ref].strides):
                distinct *= d_ if s_ != 0 else 1
            # P7-A p7_rowstat_remat: a per-row statistic read at every element (last-dimension stride 0) streams
            # one value per row; it is not a full-size read of the inline closure.
            lay_ = layouts[ref]
            row_bcast = bool(p7_rowstat_remat and lay_.shape and (
                (lay_.strides[-1] == 0 and lay_.shape[-1] > 1)
                or (lay_.shape[-1] == 1 and _numel(lay_.shape) < own_numel)))
            if r in inline:
                if (not p7_bcast_traffic or distinct == own_numel) and not row_bcast:
                    closure |= full_loads.get(r, set())
            elif row_bcast:
                pass
            elif _numel(layouts[ref].shape) == own_numel and own_numel > 1:
                # P7-A p7_bcast_traffic: a broadcast view (stride-0 dimensions, e.g. a per-row statistic read at
                # every element) streams only its distinct elements, so it is not a full-size read.
                if not p7_bcast_traffic or distinct == own_numel:
                    closure.add(r)
        full_loads[node.name] = closure
        if (p6a_inline_traffic and node.name not in column_only and own_numel > 1
                and 2 * uses.get(node.name, 0) * len(closure) > 3 * (1 + uses.get(node.name, 0))):
            continue            # recomputation would read >1.5x the traffic of one store plus a read per use
        if (retain_all or node.name in forced or node.dtype not in ("float32", "bfloat16")
                or any(dtypes[root_of(ref)] not in ("float32", "bfloat16") for ref in node.loads)):
            continue
        n_uses = uses.get(node.name, 0)
        # P6-A: recomputation that saves memory traffic (fewer full-size reads than a store + reads) may cost up to
        # twice the operation limits.
        scale_ = 2 if (p6a_inline_traffic and n_uses * len(full_loads[node.name]) < 1 + n_uses) else 1
        if n_uses and cost <= scale_ * INLINE_TOTAL_LIMIT and cost * n_uses <= scale_ * INLINE_WORK_LIMIT:
            inline.add(node.name)
        elif node.name in column_only and cost <= 2 * INLINE_TOTAL_LIMIT:
            inline.add(node.name)

    def ctype_round(expression, dtype):
        return f"__float2bfloat16_rn({expression})" if dtype == "bfloat16" else expression

    p7_hook = [None]        # P7-A: register substitution of inline producers inside row-engine rendering

    def expand(ref, shape, idx, loader):
        """Float expression for element idx of `ref` viewed at `shape`."""
        if p7_hook[0] is not None:
            hooked = p7_hook[0](ref, shape, idx, "index")
            if hooked is not None:
                return hooked
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
        if p7_hook[0] is not None:
            hooked = p7_hook[0](ref, None, position, "storage")
            if hooked is not None:
                return hooked
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
    from .pops2b_rules_p7a import IDLE_HEAVY_KINDS
    # P-OPS2-B rules belong to programs with exact-order BF16 products; others keep the base schedule.
    pops2b = pops2b and any(node.kind == "gemm" and (node.extra.get("matrix_spec") or {}).get("dtype") == "bfloat16"
                            for node in nodes.values())
    absorbed = {}                         # BF16 cast -> (product, operand) materialized in operand staging
    if pops2b and pops2b_casts and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7a import select_absorbed_casts
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
        # P6-A: a one-element loop reads one-element values at the constant index "0u" (same element, same thread).
        same = idx == "i" or (idx == "0u" and _numel(node.shape) == 1)
        return (node.kind == "elementwise" and same and root in nodes and nodes[root].kind == "elementwise"
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

    phase_of, phases = {}, []             # phase: {"heavy": [names], "loops": {numel: [names]}}

    def new_phase():
        phases.append({"heavy": [], "loops": {}})
        return len(phases) - 1

    def p4a_item(h):
        # P6-A: the fused attention's fixed-tree bias sum is also a trailing item (it only reads the backward's
        # partials, written in an earlier phase), so it can follow the next product phase's work.
        return bool((h.extra or {}).get("pops4a")) or (p6a_compact and h.kind == "attention_mixed_bias")

    def p4a_companion(h):
        # P-OPS4-A items join phases of products and ordinary reductions (sequential per block after them).
        return (p4a_item(h) or h.kind == "gemm" or (h.kind == "reduction" and not (h.extra or {}).get("exclusive_phase"))
                # P6-A: fused attention work items (whole-grid item loops) also run before P-OPS4-A items
                or (p6a_compact and str(h.kind).startswith("attention_mixed")))

    def gemm_or_one_other(phase, node):
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        if p4a_item(node):
            return all(p4a_companion(h) for h in heavy)
        if any(p4a_item(h) for h in heavy) and not p4a_companion(node):
            return False
        heavy = [h for h in heavy if not p4a_item(h)]
        if not heavy:
            return True
        if p6a_rowchain and pops3c and (node.extra or {}).get("engine") == "pops3c_row" and len(heavy) <= 2 and all(
                (h.extra or {}).get("engine") == "pops3c_row"
                and (h.extra["row_rows"], h.extra["row_extent"]) == (node.extra["row_rows"], node.extra["row_extent"])
                for h in heavy):
            return True     # P6-A: same-shape row engines run back to back on the identical row-to-warp mapping
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
            return len(heavy) < 3 and all(h.kind == 'gemm' and h.extra.get('matrix_spec') for h in heavy)
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

    def row_attach(node):
        """Attach `node` to a row engine's warp (consumer of its statistic, or same-row sibling); phase or None."""
        from . import norm_rules_p7a as norm_rules
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
            register = (attached_to[root] if root in attached_to and shape is not None
                        and _same_loop_read(node, root, layout, shape, idx) else None)
            if (register is None and root in attached_to and shape is not None and pops3c_factor
                    and _numel(nodes[root].shape) == nodes[attached_to[root]].extra["row_rows"]):
                owner_ = nodes[attached_to[root]]
                if (n == owner_.extra["row_rows"] * owner_.extra["row_extent"]
                        and norm_rules.normalized_index(_index32(layout, shape, idx))
                        == norm_rules.row_index_text(owner_.extra["row_extent"])):
                    register = attached_to[root]   # per-row member, held by every thread of the group
            loads.append((statistic, register, phase_of[root] + 1 if root in phase_of else 0, root))
            if shape is not None and layout.numel == n:
                roots.add(root)
        owner = None
        # The latest row engine whose warp can evaluate `node`: every other load is ready before its phase.
        # P6-A: statistics of row engines emitted earlier in the candidate's phase are stored before its barrier.
        for candidate in sorted({x for s_, r_, _, _ in loads for x in (s_, r_) if x}, key=lambda r: -phase_of[r]):
            if all(s_ == candidate or r_ == candidate or ready <= phase_of[candidate]
                   or root_ in row_before.get(candidate, ())
                   or attached_to.get(root_) in row_before.get(candidate, ()) for s_, r_, ready, root_ in loads):
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

    row_chain, row_chain_sources, row_before = {}, set(), {}

    def row_chain_source(node, earliest):
        """P6-A: the one same-shape row engine that alone delays `node` (a row engine) by one phase, read only at
        each row's own statistic; None otherwise."""
        rows_, extent_ = node.extra["row_rows"], node.extra["row_extent"]
        source, others = None, 0
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in phase_of:
                continue
            producer = nodes.get(root)
            owner_ = attached_to.get(root)
            engine_ = root if (producer is not None and (producer.extra or {}).get("engine") == "pops3c_row") else owner_
            if (engine_ is not None and (nodes[engine_].extra or {}).get("engine") == "pops3c_row"
                    and phase_of[root] + 1 == earliest and phase_of[engine_] == phase_of[root]
                    and (nodes[engine_].extra["row_rows"], nodes[engine_].extra["row_extent"]) == (rows_, extent_)):
                # The engine's statistic at the row, or one of its row members at this very element.
                text = _index32(layout, shape, idx) if shape is not None else str(idx)
                member_ = engine_ != root
                if member_ and _numel(nodes[root].shape) != rows_ * extent_:
                    return None
                if layout.offset or not _row_index_exact(text, rows_, extent_, element=member_):
                    return None
                if source not in (None, engine_):
                    return None
                source = engine_
            else:
                others = max(others, phase_of[root] + (0 if shape is not None and register_load(node, root, layout, shape, idx) else 1))
        if source is None or others > phase_of[source]:
            return None
        return source

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
            chain_from = None
            if (p6a_rowchain and pops3c and not retain_all and earliest > 0
                    and node.extra.get("engine") == "pops3c_row"):
                chain_from = row_chain_source(node, earliest)
                if chain_from is not None:
                    earliest = phase_of[chain_from]
            pre = []
            if (pops3c and pops3c_pre and pops3c_attach and not retain_all and earliest > 0
                    and node.extra.get("engine") == "pops3c_row"):
                pre = row_pre_members(node, earliest)
                if pre:
                    earliest -= 1
            phase = earliest
            while True:
                while phase >= len(phases):
                    new_phase()
                if gemm_or_one_other(phase, node):
                    break
                phase += 1
            phases[phase]["heavy"].append(node.name)
            phase_of[node.name] = phase
            if chain_from is not None:
                if phase == phase_of[chain_from]:
                    row_chain[node.name] = chain_from
                    row_chain_sources.add(chain_from)
                elif phase <= phase_of[chain_from]:
                    raise LoweringError("internal: chained row engine placed before its statistic")
            if p6a_rowchain and node.extra.get("engine") == "pops3c_row":
                peers = {h for h in phases[phase]["heavy"]
                         if h != node.name and (nodes[h].extra or {}).get("engine") == "pops3c_row"}
                if peers:
                    row_before[node.name] = peers
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
        from .pops2b_rules_p7a import defer_products, idle_heavy_model
        deferred, preload = defer_products(items=items, loads_of=loads_of, phase_of=phase_of, phases=phases,
                                           nodes=nodes, idle_model=idle_heavy_model(), numel_of=_numel,
                                           inplace_source=inplace_source, dtypes=dtypes)
        commit = len(phases)
    if place_by_slack and not retain_all:
        idle_heavy = None
        if pops2b and pops2b_idle and target_arch == "sm_90":
            from .pops2b_rules_p7a import idle_heavy_model
            idle_heavy = idle_heavy_model(pops2b_idle_fill)
        _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                        forced, _numel, idle_heavy=idle_heavy, deferred=deferred, preload=preload,
                        fixed=set(attached_to))
        commit = len(phases)
    epilogue_of, epilogues, permuted_members = {}, {}, {}
    if pops2b and pops2b_epilogue and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7a import attach_epilogues
        epilogue_of, epilogues, permuted_members = attach_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            dtypes=dtypes, in_place=in_place, inplace_source=inplace_source, register_load=register_load,
            numel_of=_numel, permuted_copies=pops2b_permuted, cost_of=lambda n: total_cost.get(n, 0),
            cost_limit=pops2b_epi_cost, stream_gate=pops2b_epi_stream_gate,
            background=set(deferred) if pops2b_defer_epilogue else set(), fixed=set(attached_to),
            wg_epilogue=p6a_wg_epilogue)
        commit = len(phases)


    if p6a_compact and not retain_all:
        # P6-A: epilogue attachment and deferral run after ASAP placement, so values can now be ready earlier than
        # when their readers were placed. Pull self-contained heavy items (no epilogue/row members, no in-place
        # state, no deferred/absorbed roles) into the earliest compatible phase; later readers only.
        absorbed_readers = {product for decision in absorbed.values() for product, _ in decision["readers"]}
        moved_items = 0
        for p_ in range(len(phases)):
            for name in list(phases[p_]["heavy"]):
                node = nodes[name]
                if name not in phases[p_]["heavy"]:
                    continue                    # moved with its row-engine group
                if (name in deferred or name in absorbed_readers or name in row_before
                        or name in inplace_source or (node.extra or {}).get("exclusive_phase")
                        or (node.extra or {}).get("p2_split")):
                    continue
                # A row engine moves together with the same-phase row engines emitted after it (row_before peers).
                engines = [name] + [h for h in phases[p_]["heavy"] if name in row_before.get(h, ())]
                if any(e in deferred or e in inplace_source for e in engines):
                    continue
                epi_members = list(phases[p_].get("epilogue", {}).get(name, ()))
                row_members_ = [m for e in engines for m in phases[p_].get("rows", {}).get(e, ())]
                group = {*engines, *epi_members, *row_members_}
                ready = 0
                for member in group:
                    if member != name and (member in inplace_source or member in row_pre.get(name, ())):
                        ready = p_
                    for root, layout, shape, idx in loads_of[member]:
                        # Old state of an in-place target (committed in a later phase) may be read earlier.
                        if root in group:
                            continue            # owner result / fellow members: registers or same-item order
                        elif root in phase_of:
                            ready = max(ready, phase_of[root] + 1)
                if ready >= p_:
                    continue
                for q_ in range(ready, p_):
                    placed = []
                    for e in engines:
                        if not gemm_or_one_other(q_, nodes[e]):
                            break
                        phases[q_]["heavy"].append(e)
                        placed.append(e)
                    if len(placed) != len(engines):
                        for e in placed:
                            phases[q_]["heavy"].remove(e)
                        continue
                    for e in engines:
                        phases[p_]["heavy"].remove(e)
                        if e in phases[p_].get("rows", {}):
                            phases[q_].setdefault("rows", {})[e] = phases[p_]["rows"].pop(e)
                        peers = {h for h in phases[q_]["heavy"][:phases[q_]["heavy"].index(e)]
                                 if (nodes[h].extra or {}).get("engine") == "pops3c_row"}
                        if (nodes[e].extra or {}).get("engine") == "pops3c_row" and peers:
                            row_before[e] = peers
                    if epi_members:
                        phases[q_].setdefault("epilogue", {})[name] = phases[p_]["epilogue"].pop(name)
                    for member in group:
                        phase_of[member] = q_
                    moved_items += 1
                    break
            if (not phases[p_]["heavy"] and phases[p_]["loops"] and not phases[p_].get("epilogue")
                    and not phases[p_].get("rows")):
                # Only loops remain: move every loop to its ready phase when all can leave (the phase disappears).
                targets = {}
                for numel_, members_ in phases[p_]["loops"].items():
                    group = set(members_)
                    if any(m in inplace_source or m in attached_to or m in epilogue_of for m in group):
                        targets = None
                        break
                    ready = 0
                    for m in group:
                        for root, layout, shape, idx in loads_of[m]:
                            if root in group:
                                continue
                            if root in phase_of:
                                ready = max(ready, phase_of[root] + 1)
                    if ready >= p_:
                        targets = None
                        break
                    targets[numel_] = ready
                if targets:
                    for numel_, q_ in targets.items():
                        members_ = phases[p_]["loops"][numel_]
                        phases[q_]["loops"].setdefault(numel_, []).extend(members_)
                        phases[q_]["loops"][numel_].sort(key=lambda n: order[n])
                        for m in members_:
                            phase_of[m] = q_
                    phases[p_]["loops"] = {}
                    moved_items += 1
        if moved_items:
            empty = [k for k, ph in enumerate(phases)
                     if not ph["heavy"] and not ph["loops"] and not ph.get("epilogue") and not ph.get("rows")]
            if empty:
                remap, kept = {}, []
                for k, ph in enumerate(phases):
                    if k not in empty:
                        remap[k] = len(kept)
                        kept.append(ph)
                phases[:] = kept
                for n in phase_of:
                    phase_of[n] = remap[phase_of[n]]
            commit = len(phases)

    colsum_of = {}                        # P6-A: product -> [(summed epilogue member, partial ident)]
    if p6a_epi_colsum and not retain_all and epilogue_of:
        for part_name in [n.name for n in items]:
            pnode = nodes[part_name]
            spec = (pnode.extra or {}).get("p4a_spec")
            if (not spec or spec.get("mode") != "column" or len(spec["leaves"]) != 1 or pnode.extra.get("merged")
                    or part_name in deferred):
                continue
            toks = _TOKEN.findall(spec["leaves"][0])
            if len(toks) != 1:
                continue
            kind_, ref_, shape_, pos_ = token_loads[int(toks[0])]
            summed = root_of(ref_)
            product = epilogue_of.get(summed)
            pspec = (nodes[product].extra or {}).get("matrix_spec") if product else None
            lay_ = layouts[ref_]
            if (kind_ != "storage" or not pspec or not pspec.get("wg") or not pspec.get("tma") or pspec.get("splitk")
                    or pspec["m"] != spec["E"] or pspec["n"] != spec["N"] or not pspec.get("pair")
                    or lay_.offset or not lay_.contiguous or lay_.numel != spec["E"] * spec["N"]
                    or nodes[summed].dtype != "float32"):
                continue
            text_ = re.sub(r"\bo\b", "pcol", re.sub(r"\bpr\b", "prow", str(pos_)))
            if not _row_index_exact(text_, spec["E"], spec["N"], element=True):
                continue
            finals = [n for n in nodes if (nodes[n].code or "").startswith(f"latch_p4a_colfin({_ident(part_name)}, ")]
            if len(finals) != 1:
                continue
            slabs = -(-spec["E"] // 64)
            layouts[part_name] = TensorLayout((slabs * spec["N"],), (1,), part_name)
            nodes[finals[0]].code = re.sub(r"^(latch_p4a_colfin\(\w+, \w+, \d+u, )\d+u", rf"\g<1>{slabs}u",
                                           nodes[finals[0]].code)
            phases[phase_of[part_name]]["heavy"].remove(part_name)
            phase_of[part_name] = phase_of[product]
            items = [n for n in items if n.name != part_name]
            loads_of.pop(part_name, None)
            pnode.extra["epilogue_colsum_of"] = product
            colsum_of.setdefault(product, []).append((summed, _ident(part_name)))
        if colsum_of:
            empty = [k for k, ph in enumerate(phases)
                     if not ph["heavy"] and not ph["loops"] and not ph.get("epilogue") and not ph.get("rows")]
            if empty:
                remap, kept = {}, []
                for k, ph in enumerate(phases):
                    if k not in empty:
                        remap[k] = len(kept)
                        kept.append(ph)
                phases[:] = kept
                for n in phase_of:
                    phase_of[n] = remap[phase_of[n]]
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
                or node.name in inplace_source):
            stored.add(node.name)
    stored.update(cast for cast, decision in absorbed.items() if decision["mode"] == "materialize")
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
    if row_before:
        row_check["row_before"] = row_before
    if epilogue_of:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError,
                        epilogue_of=epilogue_of, permuted=permuted_members, **row_check)
    else:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError, **row_check)
    storage_roots = [n for n in nodes if (nodes[n].kind != "elementwise" or n in stored) and n not in inplace_source
                     and n not in elided_products]
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
    scalar_roots = ([r_ for r_ in storage_roots if sizes[r_] <= 2 and layouts[r_].numel == 1]
                    if p7_scalar_block and not retain_all else [])
    offsets, arena_words, _ = arena_assign.assign([r_ for r_ in storage_roots if r_ not in scalar_roots], sizes,
                                                {k_: v_ for k_, v_ in classes.items() if k_ not in scalar_roots},
                                                commit)
    if scalar_roots:
        # P7-A: one-element values (e.g. per-tensor norm partials read by one scalar chain) live in one contiguous,
        # never-reused block after the planned arena (16-byte slots), instead of scattered reuse gaps.
        base_ = (arena_words + 63) // 64 * 64
        for k_, r_ in enumerate(scalar_roots):
            offsets[r_] = base_ + 64 * k_              # the arena's 64-word allocation unit (plan.check rule)
        arena_words = base_ + 64 * len(scalar_roots)
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
    aligned16 = {}
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
        # P7-A: 16-byte alignment of the value pointer (inputs are checked 16-byte aligned by the Runtime; the
        # arena base is a CUDA allocation; arena offsets are in 4-byte words).
        aligned16[_ident(name)] = ((layout.root in inputs or (4 * offsets[layout.root]) % 16 == 0)
                                   and (layout.offset * _BYTES[dtypes[name]]) % 16 == 0)

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
        return _TOKEN.sub(substitute, node.code)

    vector_helpers_needed = []

    def split_loop(numel, members):
        """P6-A: a loop whose members form independent register-connected components (e.g. one AdamW chain per
        parameter tensor of equal size) is emitted as consecutive loops over packed components, keeping the
        values live between a group's loads and stores within LIVE_VALUE_CAP (no register spills)."""
        if not p6a_split_loops or len(members) <= 1:
            return [members]
        member_set = set(members)
        parent = {m: m for m in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        refs = {m: set() for m in members}
        for m in members:
            for root, layout, shape, idx in loads_of[m]:
                if root in member_set:
                    parent[find(m)] = find(root)
                else:
                    refs[m].add(root)
        groups = {}
        for m in members:
            groups.setdefault(find(m), []).append(m)
        if len(groups) <= 1:
            return [members]
        if numel == 1 or (p6a_rotate and numel < RESIDENT_THREADS):
            # One loop per independent component; loops smaller than the resident grid then run concurrently on
            # different thread ranges (one-element chains on warps, small loops rotated) instead of in sequence.
            return [sorted(g, key=lambda n: order[n]) for g in sorted(groups.values(), key=lambda g: order[g[0]])]
        piped = p6a_pipeline and numel % 4 == 0 and all(nodes[m].dtype in ("float32", "bfloat16") for m in members)
        lanes = 4 if numel % 4 == 0 and (numel >= VECTOR_MIN_ELEMENTS or piped) else 1
        cap_ = min(p6a_split_cap, P6A_PIPELINE_LIVE) if piped else p6a_split_cap
        packed, current, current_refs = [], [], set()
        for comp in sorted(groups.values(), key=lambda g: order[g[0]]):
            comp_refs = set().union(*(refs[m] for m in comp))
            live = lanes * (len(current) + len(comp) + len(current_refs | comp_refs))
            if current and live > cap_:
                packed.append(current)
                current, current_refs = [], set()
            current = current + comp
            current_refs |= comp_refs
        if current:
            packed.append(current)
        return [sorted(g, key=lambda n: order[n]) for g in packed]

    def loop_code(numel, members, exclude, slot=0, share=None, pipeline=False, rot=0):
        if numel >= (1 << 31):
            raise LoweringError("fused elementwise loops require fewer than 2^31 elements")
        loop_regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        vector = floating and numel % 4 == 0 and (numel >= VECTOR_MIN_ELEMENTS or pipeline)
        vector_refs = {}

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            # The operand's own element count must equal the loop's: a broadcast
            # operand (e.g. a [512] weight in a [256,512] loop) is never an identity read.
            identity = ((idx == "i" or (idx == "0u" and numel == 1)) and shape is not None and _numel(shape) == numel
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
            if outline_ops:
                from .norm_rules_p7a import cse_calls
                prelude, expression = cse_calls(expression, f"c_{name}")
                compute.extend(prelude)
            compute.append(f"{loop_regs[name]}[v] = {ctype_round(expression, node.dtype)};")
        hoisted = []
        if p6a_hoist and numel > 1:
            written = {_ident(n) for n in members} | {_ident(t) for t, src in in_place.items() if src in member_set}
            tag = hashlib.sha256(",".join(members).encode()).hexdigest()[:8]
            hoisted, compute = _hoist_invariants(compute, written, f"h_{tag}_")
        p7_par_prep = ""
        if p7_scalar_loads and numel == 1:
            # P7-A: a one-element chain issues every read-only scalar load before its first dependent operation
            # (one memory round trip instead of one per link of the chain); values and operations are unchanged.
            written = {_ident(n) for n in members} | {_ident(t) for t, src in in_place.items() if src in member_set}
            scalars = []
            for line_ in compute:
                for m_ in re.finditer(r"\b(v_\w+)\[0u\]", line_):
                    if m_.group(1) not in written and m_.group(0) not in scalars:
                        scalars.append(m_.group(0))
            if len(scalars) >= 4:
                names_ = {sc: f"s7_{k_}" for k_, sc in enumerate(scalars)}
                decls = [f"const {_CTYPES[ident_dtype[sc.split('[')[0]]]} {names_[sc]} = {sc};" for sc in scalars]
                if p7_scalar_pin:
                    # P7-A: FP32 scalar loads as volatile asm, so neither NVVM nor ptxas sinks a load next to its
                    # first use: with IEEE sqrt/div slow-path branches between uses, sunk loads issue one at a time
                    # (one L2 round trip per link). The loaded values are unchanged.
                    decls = [(f"float {names_[sc]}; asm volatile(\"ld.global.f32 %0, [%1];\" : \"=f\"({names_[sc]}) "
                              f": \"l\"({sc.split('[')[0]}));")
                             if ident_dtype[sc.split('[')[0]] == "float32" else d_
                             for sc, d_ in zip(scalars, decls)]
                compute = decls + [re.sub(r"\b(v_\w+)\[0u\]", lambda m_: names_.get(m_.group(0), m_.group(0)), l_)
                                   for l_ in compute]
                if p7_scalar_par and not (exclude or share or rot or slot or vector or pipeline):
                    # P7-A: unary transcendental calls of one loaded scalar each (e.g. per-tensor norms) are
                    # evaluated by separate threads of the CTA that runs the chain, before it; the chain reads
                    # the identical values from shared memory instead of issuing the calls (and their IEEE
                    # slow-path branches) one after another.
                    calls = []
                    for l_ in compute:
                        for m_ in re.finditer(r"\b(sqrtf|rsqrtf|expf|logf|erff)\((s7_\d+)\)", l_):
                            if m_.group(0) not in calls:
                                calls.append(m_.group(0))
                    if len(calls) >= 8:
                        inv = {v_: k_ for k_, v_ in names_.items()}
                        cases = []
                        for j_, c_ in enumerate(calls):
                            fn_, sv_ = re.fullmatch(r"(\w+)\((s7_\d+)\)", c_).groups()
                            cases.append(f"case {j_}u: p7_par[{j_}] = {fn_}({inv[sv_]}); break;")
                        par_index = {c_: j_ for j_, c_ in enumerate(calls)}
                        compute = [re.sub(r"\b(?:sqrtf|rsqrtf|expf|logf|erff)\(s7_\d+\)",
                                          lambda m_: f"p7_par[{par_index[m_.group(0)]}]", l_) for l_ in compute]
                        p7_par_prep = (f"  if (blockIdx.x == 0) {{\n    __syncthreads();\n"
                                       f"    for (unsigned k = threadIdx.x; k < {len(calls)}u; k += blockDim.x) {{\n"
                                       f"      switch (k) {{ " + " ".join(cases) + " default: break; }\n    }\n"
                                       f"    __syncthreads();\n  }}\n")
        lanes = 4 if vector else 1
        count = numel // lanes
        if pipeline:
            # P6-A: the pieces of a x4 vector loop for pipelined_loops (identity loads issued one stage ahead).
            return {"numel": numel, "members": members, "count": count, "hoisted": hoisted, "compute": compute,
                    "refs": [(ref, reg, _CTYPES[dtypes[ref]]) for ref, reg in vector_refs.items()],
                    "regs": [(_CTYPES[nodes[n].dtype], loop_regs[n]) for n in members],
                    "stores": [(_ident(n), loop_regs[n]) for n in stored_members]}
        u = 1
        threads_ = RESIDENT_THREADS if not share else max(256, RESIDENT_THREADS * (share[1] - share[0]) // share[2])
        per_thread = -(-count // threads_)
        cap = max(1, LIVE_VALUE_CAP // max(1, lanes * len(members)))
        while u * 2 <= min(per_thread, UNROLL_MAX, cap):
            u *= 2
        declarations_local = " ".join(f"{_CTYPES[nodes[n].dtype]} {loop_regs[n]}[{u * lanes}];" for n in members)
        if exclude:
            # Blocks [0, exclude) run the phase's matrix tasks; when they use at
            # most half the grid, the loop is distributed over the idle blocks.
            head = (f"  const unsigned skip = 2u * {exclude}u <= gridDim.x ? {exclude}u : 0u;\n"
                    f"  const unsigned vb = (blockIdx.x + gridDim.x - skip) % gridDim.x;\n"
                    f"  const unsigned gs = (gridDim.x - skip) * blockDim.x;\n"
                    f"  if (vb < gridDim.x - skip)\n")
            start_block = "vb"
        elif share:
            # P6-A: independent loops of one phase run concurrently on proportional CTA ranges.
            a_, b_, d_ = share
            head = (f"  const unsigned clo = (gridDim.x * {a_}u) / {d_}u;\n"
                    f"  const unsigned chi = max(clo + 1u, (gridDim.x * {b_}u) / {d_}u);\n"
                    f"  const unsigned vb = blockIdx.x - clo;\n"
                    f"  const unsigned gs = (chi - clo) * blockDim.x;\n"
                    f"  if (blockIdx.x >= clo && blockIdx.x < chi)\n")
            start_block = "vb"
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
        head = "".join(f"  {d}\n" for d in hoisted) + head
        if p7_par_prep:
            head = "  float* const p7_par = smem;\n" + p7_par_prep + head
        start = f"{start_block} * blockDim.x + threadIdx.x" + (f" - {slot}u" if slot else "")
        if rot and not slot and start_block == "blockIdx.x":
            # P6-A: rotated start (a bijection on [0, gs) for any launched grid), so small loops begin on
            # different warps/CTAs.
            start = f"(blockIdx.x * blockDim.x + threadIdx.x + gs - {rot}u % gs) % gs"
        return (f"{{ // {numel} elements{' (x4 vector)' if vector else ''}: {', '.join(members)}\n" + head +
                f"  for (unsigned base = {start}; base < {_u(count)}; base += {u}u * gs) {{\n"
                f"    {declarations_local}\n#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n" + body + "    }\n"
                + store_block + "  }\n" + ("  if (blockIdx.x == 0) __syncthreads();  // shared values read\n"
                                         if p7_par_prep else "") + "}")

    def p7_components(members):
        """Register-connected components of a loop's members (as split_loop), in member order."""
        member_set = set(members)
        parent = {m: m for m in members}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for m in members:
            for root, layout, shape, idx in loads_of[m]:
                if root in member_set:
                    parent[find(m)] = find(root)
        groups = {}
        for m in members:
            groups.setdefault(find(m), []).append(m)
        return [sorted(g, key=lambda n: order[n]) for g in sorted(groups.values(), key=lambda g: order[g[0]])]

    def p7_flat_loops_code(parts_):
        """P7-A flattened loops: the independent x4 vector loops of one phase (one segment per register-connected
        component) form one index space walked grid-stride by every thread; a balanced constant binary decision
        selects the segment of an item. Loop-invariant scalars (P6-A hoists, e.g. per-tensor optimizer bias
        corrections) are evaluated once per CTA into shared memory instead of by every thread before every
        loop. Per-element arithmetic, loads and stores are the unchanged loop bodies."""
        pieces = []
        for numel, members in parts_:
            for comp in p7_components(members):
                piece = loop_code(numel, comp, 0, pipeline=True)
                pieces.append(piece)
        # Rule scope (gpu-05): one flattened item per iteration helps loops with several identity loads per item
        # (optimizer-like chains); single-load copy/cast loops keep their unrolled grid-stride form.
        items_ = sum(pc["count"] for pc in pieces)
        if items_ == 0 or sum(len(pc["refs"]) * pc["count"] for pc in pieces) < P7_FLAT_MIN_LOADS * items_:
            return None
        hoists = []
        for piece in pieces:
            renamed = []
            for line in piece["hoisted"]:
                m_ = re.fullmatch(r"const float (h_\w+) = (.*);", line.strip())
                if not m_:
                    return None
                renamed.append((m_.group(1), m_.group(2)))
            piece["p7_hoists"] = []
            for name_, expr_ in renamed:
                piece["p7_hoists"].append((name_, len(hoists)))
                hoists.append(expr_)
        if 4 * len(hoists) > 16384:
            return None
        table_text = ""
        if p7_flat_table:
            table_text, pieces = p7_flat_table_code(pieces, int(p7_flat_table), len(hoists))
        bounds, total = [], 0
        for piece in pieces:
            bounds.append(total)
            total += piece["count"]
        lines = [f"{{ // P7-A flattened loops: {len(pieces)} segments, {total} x4 items, {len(hoists)} CTA-shared invariants"]
        if hoists:
            lines.append("  __syncthreads();  // shared memory reuse")
            lines.append("  switch (threadIdx.x) {")
            for t_ in range(min(256, len(hoists))):
                stmts = " ".join(f"smem[{k_}] = {hoists[k_]};" for k_ in range(t_, len(hoists), 256))
                lines.append(f"    case {t_}: {stmts} break;")
            lines.append("    default: break;\n  }\n  __syncthreads();")

        def body(k_):
            piece = pieces[k_]
            out_ = [f"const unsigned q = f - {bounds[k_]}u;"]
            out_ += [f"const float {name_} = smem[{slot_}];" for name_, slot_ in piece["p7_hoists"]]
            out_ += [f"{ct} {reg}[4]; latch_ld4({_ident(ref)} + 4u * q, {reg});" for ref, reg, ct in piece["refs"]]
            out_ += [f"{ct} {reg}[4];" for ct, reg in piece["regs"]]
            out_.append("#pragma unroll\n      for (int j = 0; j < 4; ++j) {\n        const unsigned i = 4u * q + (unsigned)j; "
                        "const int v = j;\n        " + "\n        ".join(piece["compute"]) + "\n      }")
            out_ += [f"latch_st4({ident} + 4u * q, {reg}[0], {reg}[1], {reg}[2], {reg}[3]);"
                     for ident, reg in piece["stores"]]
            return "{\n      " + "\n      ".join(out_) + "\n    }"

        def tree(lo_, hi_, depth):
            if hi_ - lo_ == 1:
                return body(lo_)
            mid = (lo_ + hi_) // 2
            pad = "  " * depth
            return (f"if (f < {bounds[mid]}u) {{ {tree(lo_, mid, depth + 1)} }}\n    {pad}else {{ {tree(mid, hi_, depth + 1)} }}")
        if table_text:
            lines.append(table_text)
        if pieces:
            lines.append("  {\n  const unsigned gs = gridDim.x * blockDim.x;")
            lines.append(f"  for (unsigned f = blockIdx.x * blockDim.x + threadIdx.x; f < {total}u; f += gs) {{")
            lines.append("    " + tree(0, len(pieces), 0))
            lines.append("  }\n  }")
        lines.append("}")
        return "\n".join(lines), len(hoists)

    def p7_flat_table_code(pieces, unroll, n_inv):
        """P7-A table-driven flattened loops: segments whose loop bodies are identical up to the identity of their
        operand/result pointers and invariant values (canonical text; e.g. one optimizer chain per parameter tensor)
        share one body. A per-segment table (pointers, invariants, item bases) is written to shared memory once
        per CTA; each thread then keeps `unroll` items' loads in flight (all loads issued before any compute).
        Per-element arithmetic and loads/stores are those of the unchanged bodies. Returns (code, other pieces)."""
        def canon(piece):
            text = "\n".join(piece["compute"])
            for k_, (ct_, reg_) in enumerate(piece["regs"]):
                text = re.sub(r"\b" + re.escape(reg_) + r"\b", f"R{k_}", text)
            for k_, (name_, _) in enumerate(piece["p7_hoists"]):
                text = re.sub(r"\b" + re.escape(name_) + r"\b", f"H{k_}", text)
            regs_ = {reg_: f"R{k_}" for k_, (ct_, reg_) in enumerate(piece["regs"])}
            stores_ = tuple(regs_[reg_] for _, reg_ in piece["stores"])
            others = tuple(sorted(set(re.findall(r"\bv_\w+\b", text))))
            return (text, tuple(ct_ for _, _, ct_ in piece["refs"]), tuple(reg_ for _, reg_, _ in piece["refs"]),
                    tuple(ct_ for ct_, _ in piece["regs"]), stores_, others, len(piece["p7_hoists"]))
        groups = {}
        for piece in pieces:
            groups.setdefault(canon(piece), []).append(piece)
        key, members_ = max(groups.items(), key=lambda kv: sum(pc["count"] for pc in kv[1]))
        if len(members_) < 4 or len(members_) > 256:
            return "", pieces
        text, ref_ct, ref_regs, reg_ct, stores_, others, nh = key
        S, NL, NS = len(members_), len(ref_ct), len(stores_)
        NP = NL + NS
        total = sum(pc["count"] for pc in members_)
        # shared layout after the CTA invariants: pointers (u64, 8-byte aligned), invariants, bases
        off_p = -(-4 * n_inv // 8) * 8
        off_h = off_p + 8 * S * NP
        off_b = off_h + 4 * S * max(nh, 1)
        if off_b + 4 * (S + 1) > 16384:
            return "", pieces
        lines = [f"  {{ // P7-A table-driven flattened loops: {S} segments share one body, {total} x4 items, "
                 f"{unroll} items per thread in flight",
                 f"  unsigned long long* p7_tp = reinterpret_cast<unsigned long long*>(reinterpret_cast<char*>(smem) + {off_p});",
                 f"  float* p7_th = reinterpret_cast<float*>(reinterpret_cast<char*>(smem) + {off_h});",
                 f"  unsigned* p7_tb = reinterpret_cast<unsigned*>(reinterpret_cast<char*>(smem) + {off_b});",
                 "  __syncthreads();",
                 "  switch (threadIdx.x) {"]
        base_ = 0
        for s_, pc in enumerate(members_):
            stmts = []
            for j_, (ref, _, _) in enumerate(pc["refs"]):
                stmts.append(f"p7_tp[{s_ * NP + j_}] = (unsigned long long)({_ident(ref)});")
            for j_, (ident, _) in enumerate(pc["stores"]):
                stmts.append(f"p7_tp[{s_ * NP + NL + j_}] = (unsigned long long)({ident});")
            for k_, line in enumerate(pc["hoisted"]):
                expr_ = re.fullmatch(r"const float h_\w+ = (.*);", line.strip()).group(1)
                stmts.append(f"p7_th[{s_ * max(nh, 1) + k_}] = {expr_};")
            stmts.append(f"p7_tb[{s_}] = {base_}u;")
            base_ += pc["count"]
            lines.append(f"    case {s_}: {' '.join(stmts)} break;")
        lines.append(f"    default: break;\n  }}\n  if (threadIdx.x == 0) p7_tb[{S}] = {total}u;\n  __syncthreads();")
        U = unroll
        loads = []
        for j_, ct_ in enumerate(ref_ct):
            loads.append(f"latch_ld4(reinterpret_cast<const {ct_}*>(p7_tp[s_[u] * {NP}u + {j_}u]) + 4u * q_[u], "
                         f"{ref_regs[j_]}[u]);")
        decl_in = " ".join(f"{ct_} {ref_regs[j_]}[{U}][4];" for j_, ct_ in enumerate(ref_ct))
        body_text = text
        for j_ in range(NL):
            body_text = re.sub(r"\b" + re.escape(ref_regs[j_]) + r"\[j\]", f"{ref_regs[j_]}[u][j]", body_text)
        stores_text = " ".join(f"latch_st4(reinterpret_cast<{reg_ct[int(r_[1:])]}*>(p7_tp[s_[u] * {NP}u + {NL + j_}u]) + 4u * q_[u], "
                               f"{r_}[0], {r_}[1], {r_}[2], {r_}[3]);" for j_, r_ in enumerate(stores_))
        hdecl = " ".join(f"const float H{k_} = p7_th[s_[u] * {max(nh, 1)}u + {k_}u];" for k_ in range(nh))
        rdecl = " ".join(f"{ct_} R{k_}[4];" for k_, ct_ in enumerate(reg_ct))
        lines += [
            "  const unsigned gs = gridDim.x * blockDim.x;",
            f"  for (unsigned f0 = blockIdx.x * blockDim.x + threadIdx.x; f0 < {total}u; f0 += {U}u * gs) {{",
            f"    unsigned s_[{U}], q_[{U}]; bool ok_[{U}]; {decl_in}",
            f"#pragma unroll\n    for (int u = 0; u < {U}; ++u) {{",
            f"      const unsigned f = f0 + (unsigned)u * gs; ok_[u] = f < {total}u; s_[u] = 0u; q_[u] = 0u;",
            "      if (ok_[u]) {",
            f"        unsigned lo = 0u, hi = {S}u;",
            "        while (hi - lo > 1u) { const unsigned mid = (lo + hi) >> 1; if (f >= p7_tb[mid]) lo = mid; else hi = mid; }",
            "        s_[u] = lo; q_[u] = f - p7_tb[lo];",
            "        " + " ".join(loads),
            "      }",
            "    }",
            f"#pragma unroll\n    for (int u = 0; u < {U}; ++u) {{",
            "      if (!ok_[u]) continue;",
            f"      {hdecl} {rdecl}",
            "#pragma unroll\n      for (int j = 0; j < 4; ++j) {\n        const unsigned i = 4u * q_[u] + (unsigned)j; const int v = j; (void)i;\n        "
            + body_text.replace("\n", "\n        ") + "\n      }",
            f"      {stores_text}",
            "    }",
            "  }",
            "  }"]
        rest = [pc for pc in pieces if not any(pc is m_ for m_ in members_)]
        return "\n".join(lines), rest


    def pipelined_loops(pieces):
        """P6-A: the independent x4 vector loops of one phase as one software-pipelined sequence of stages (one
        planned grid-stride iteration of one loop each): the identity operand loads of stage s+1 are issued before
        stage s computes and stores, so a thread's DRAM round trips overlap instead of serializing loop after loop.
        Same per-element arithmetic and stores; iterations beyond the planned resident grid run in ordinary
        remainder loops (correct for any launched grid)."""
        stages, cursor_ = [], 0
        for k_, piece in enumerate(pieces):
            planned = min(P6A_PIPELINE_ITERS, -(-piece["count"] // RESIDENT_THREADS))
            piece["planned"], piece["base"] = planned, 0
            if planned == 1 and piece["count"] < RESIDENT_THREADS:
                # Loops smaller than the resident grid start on different warps (thread t0 handles t0 - base).
                width = -(-piece["count"] // 32) * 32
                if cursor_ + width > RESIDENT_THREADS:
                    cursor_ = 0
                piece["base"], cursor_ = cursor_, cursor_ + width
            stages += [(k_, it) for it in range(planned)]
        lines = [f"{{ // P6-A pipelined loops ({len(stages)} stages): "
                 + "; ".join(f"{p_['numel']} x4: {', '.join(p_['members'])}" for p_ in pieces)]
        for piece in pieces:
            lines += [f"  {d}" for d in piece["hoisted"]]
        lines.append("  const unsigned gs = gridDim.x * blockDim.x, t0 = blockIdx.x * blockDim.x + threadIdx.x;")

        def rename(text, s_):
            return re.sub(r"\bin_(\d+)\b", rf"p{s_}_in_\1", text)

        def load(s_):
            k_, it = stages[s_]
            piece = pieces[k_]
            out_ = [f"  const unsigned q{s_} = t0 + {it}u * gs;" if not piece["base"] else
                    f"  const unsigned q{s_} = t0 - {piece['base']}u;  // wraps above the count below base"]
            out_ += [f"  {ct} p{s_}_{reg}[4];" for _, reg, ct in piece["refs"]]
            if piece["refs"]:
                out_.append(f"  if (q{s_} < {_u(piece['count'])}) {{ "
                            + " ".join(f"latch_ld4({_ident(ref)} + 4u * q{s_}, p{s_}_{reg});"
                                       for ref, reg, _ in piece["refs"]) + " }")
            return out_

        def body(piece, q, s_=None):
            decl = " ".join(f"{ct} {reg}[4];" for ct, reg in piece["regs"])
            compute = [rename(c, s_) if s_ is not None else c for c in piece["compute"]]
            stores = "".join(f"      latch_st4({ptr} + 4u * {q}, {reg}[0], {reg}[1], {reg}[2], {reg}[3]);\n"
                             for ptr, reg in piece["stores"])
            return (f"    {decl}\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {{\n"
                    f"      const unsigned i = 4u * {q} + (unsigned)j; const int v = j;\n      "
                    + "\n      ".join(compute) + f"\n    }}\n{stores}")
        if stages:
            lines += load(0)
        for s_, (k_, it) in enumerate(stages):
            if s_ + 1 < len(stages):
                lines += load(s_ + 1)
            lines.append(f"  if (q{s_} < {_u(pieces[k_]['count'])}) {{\n" + body(pieces[k_], f"q{s_}", s_) + "  }")
        for piece in pieces:
            loads = "".join(f"    {ct} {reg}[4]; latch_ld4({_ident(ref)} + 4u * q, {reg});\n"
                            for ref, reg, ct in piece["refs"])
            start_ = (f"t0 + {piece['planned']}u * gs" if not piece["base"] else
                      f"t0 + (gs > {piece['base']}u ? gs - {piece['base']}u : 0u)")
            lines.append(f"  for (unsigned q = {start_}; q < {_u(piece['count'])}; q += gs) {{\n"
                         + loads + body(piece, "q") + "  }")
        lines.append("}")
        return "\n".join(lines)

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
        from . import norm_rules_p7a as norm_rules
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
                from .norm_rules_p7a import cse_calls
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

    ident_dtype = {_ident(name): dtypes[name] for name in layouts}
    _CTYPES_STAGE = ("float32", "bfloat16")

    p7_outlined = {}

    def p7_outline_block(text, prefix):
        """P7-A: a phase block whose code is identical across layers up to its pointer operands becomes one shared
        __noinline__ device function (pointer parameters, shared memory passed in); every phase calls it. One code
        copy stays hot in the instruction caches instead of one cold copy per layer. Only wgmma-free bodies are
        outlined (warpgroup-MMA main loops must stay in the kernel entry, see wgmma serialization)."""
        if "wgmma" in text or "tma_" in text or "/*" in text.replace("/*ROW_LO*/0u", ""):
            return text
        idents = list(dict.fromkeys(re.findall(r"\bv_\w+\b", text)))
        if any(ident_dtype.get(x) is None for x in idents):
            return text
        body = text.replace("/*ROW_LO*/0u", "0u")
        for k_, x in enumerate(idents):
            body = re.sub(r"\b" + re.escape(x) + r"\b", f"p7a{k_}", body)
        # Local register names derived from value names (x_/k_/y_/r_/q_/c_ prefixes, s_v_/p7_a_v_ statistics) are
        # canonicalized too, so identical per-layer structures share one function.
        locals_ = list(dict.fromkeys(re.findall(r"\b(?:[xkyrqc]_\w+|s_v_\w+|p7_a_v_\w+)\b", body)))
        for k_, x in enumerate(locals_):
            body = re.sub(r"\b" + re.escape(x) + r"\b", f"p7l{k_}", body)
        key = (prefix, body, tuple(ident_dtype[x] for x in idents))
        if key not in p7_outlined:
            fname = f"{prefix}_{hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]}"
            params = ", ".join(f"{_CTYPES[ident_dtype[x]]}* __restrict__ p7a{k_}" if False else
                               f"{_CTYPES[ident_dtype[x]]}* p7a{k_}" for k_, x in enumerate(idents))
            helper_parts.append(f"__device__ __noinline__ void {fname}({params}{', ' if params else ''}float* smem) {{\n"
                                f"{body}\n}}")
            p7_outlined[key] = fname
        return f"{p7_outlined[key]}({', '.join(idents)}{', ' if idents else ''}smem);"

    def p7_row_group(phase, p4a_items=()):
        """P7-A register row engine (ops/p7_rows.py) for the phase's row engines; (None, None) when not admitted."""
        from . import p7_rows
        group_names = [n for n in phase["heavy"] if (nodes[n].extra or {}).get("engine") == "pops3c_row"]
        if not group_names or any(n in deferred for n in group_names):
            return None, None, ()
        shapes = {(nodes[n].extra["row_rows"], nodes[n].extra["row_extent"]) for n in group_names}
        if len(shapes) != 1 or not all("row_leaf" in nodes[n].extra for n in group_names):
            return None, None, ()
        rows, extent = shapes.pop()
        if not p7_rows.eligible(rows, extent):
            return None, None, ()
        stats, outs, written = {}, {}, set()
        for n in group_names:
            m_ = re.fullmatch(r"(\w+)\[g\] = (.*);", nodes[n].extra["row_store"].strip())
            if not m_ or dtypes[n] not in ("float32", "bfloat16"):
                return None, None, ()
            outs[n] = (m_.group(1), m_.group(2))
            stats[m_.group(1)] = "s_" + m_.group(1)
            written.add(m_.group(1))
            for member in phase.get("rows", {}).get(n, []):
                written.add(_ident(member))
                if dtypes[member] not in ("float32", "bfloat16"):
                    return None, None, ()
        numel = rows * extent
        row_of_i = re.compile(r"\(i / " + str(extent) + r"u\)")
        classify = p7_rows.Classifier(rows, extent)
        pre_owned = any(n in row_pre for n in group_names)

        def identity_root(ref, shape, idx, kind):
            """Inline producer root read at the group's own element (proven), else None."""
            layout = layouts.get(ref)
            if layout is None or layout.root not in inline or layout.offset or not layout.contiguous:
                return None
            if nodes[layout.root].dtype not in ("float32", "bfloat16"):
                return None
            text = f"(unsigned)({idx})" if kind == "storage" else (
                _index32(layout, shape, idx) if shape is not None else None)
            return layout.root if text is not None and classify(text) == "element" else None

        def build(virtual, counts):
            group = p7_rows.Group(rows, extent, ident_dtype, written, stats)
            group.classify = classify

            def hit(ref, shape, idx, kind):
                root = identity_root(ref, shape, idx, kind)
                if root is None:
                    return None
                counts[root] = counts.get(root, 0) + 1
                return f"y_{root}[r][e]" if root in virtual else None
            p7_hook[0] = hit
            try:
                return build_body(group, virtual, counts, hit)
            finally:
                p7_hook[0] = None

        def build_body(group, virtual, counts, hit):

            def render_tokens(n, text):
                if pre_owned:
                    return heavy_code(SimpleNamespace(name=n, code=text))

                def substitute(match):
                    kind, ref, shape, idx = token_loads[int(match.group(1))]
                    value = hit(ref, shape, idx, kind)
                    if value is not None:
                        return value
                    return (expand(ref, shape, idx, memory_loader) if kind == "index"
                            else expand_storage(ref, idx, memory_loader))
                return _TOKEN.sub(substitute, text)

            def p7_operand(arg, shape, idx, loader):
                ref = _ref(arg, names)
                if ref is not None and shape is not None:
                    value = hit(ref, shape, idx, "index")
                    if value is not None:
                        return value
                return operand(arg, shape, idx, loader)
            entries = []
            for n in group_names:
                out_ident, rhs = outs[n]
                leaf = group.regify(render_tokens(n, nodes[n].extra["row_leaf"]))
                sreg = f"s_{out_ident}[r]"
                stat_value = f"__bfloat162float({sreg})" if dtypes[n] == "bfloat16" else sreg
                members = phase.get("rows", {}).get(n, [])
                per_row = [m for m in members if _numel(nodes[m].shape) == rows]
                elem = [m for m in members if m not in per_row]
                if any(_numel(nodes[m].shape) != numel for m in elem):
                    return None

                def loader(ref, shape, idx, owner=n, elem=set(elem), per_row=set(per_row), stat_value=stat_value):
                    origin = layouts[ref].root
                    if origin == owner:
                        return stat_value
                    plain = (layouts[ref].contiguous and layouts[ref].offset == 0 and shape is not None)
                    if (plain and origin in elem and idx == "i" and _numel(shape) == numel
                            and layouts[ref].numel == numel):
                        value = f"r_{origin}[e]"
                        return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
                    if plain and origin in per_row and idx == "g" and layouts[ref].numel == _numel(shape):
                        value = f"q_{origin}"
                        return f"__bfloat162float({value})" if dtypes[origin] == "bfloat16" else value
                    return memory_loader(ref, shape, idx)
                member_entries = []
                for m in per_row + elem:
                    member = nodes[m]
                    index = "g" if m in per_row else "i"
                    use = p7_operand if m not in per_row else operand
                    expression = member.render(index, lambda arg, i2, member=member, loader=loader, use=use:
                                               use(arg, member.shape, i2, loader))
                    expression = row_of_i.sub("g", expression)
                    if outline_ops:
                        from .norm_rules_p7a import cse_calls
                        prelude, expression = cse_calls(expression, f"c_{m}")
                        if prelude:
                            return None          # keep the engine simple: outlined-call CSE stays on P-OPS3-C
                    expression = group.regify(ctype_round(expression, member.dtype))
                    if _ident(m) != f"v_{m}":
                        return None
                    member_entries.append((_ident(m), _CTYPES[member.dtype], expression, m in stored, m in per_row))
                entries.append((out_ident, _CTYPES[dtypes[n]], leaf, rhs, member_entries))
            # Column-partial items of the phase over the same rows (normalization weight/bias gradients) fold in.
            columns, folded, tiles = [], [], None
            for item in (p4a_items if p7_row_columns else ()):
                spec = (nodes[item].extra or {}).get("p4a_spec") or {}
                if spec.get("mode") != "column" or spec.get("E") != rows or spec.get("N") != extent:
                    continue
                if tiles is not None and (spec["T"], spec["RT"]) != tiles:
                    continue
                if sum(len(c[0]) for c in columns) + len(spec["leaves"]) > 4:
                    continue
                leaves = [group.regify(render_tokens(item, leaf)) for leaf in spec["leaves"]]
                tiles = (spec["T"], spec["RT"])
                columns.append((leaves, list(spec["parts"])))
                folded.append(item)
            # Virtual element registers: each admitted inline producer evaluated once per element.
            virtual_texts = []
            for k_, root in enumerate(virtual):
                earlier = set(virtual[:k_])
                p7_hook[0] = lambda ref, shape, idx, kind, earlier=earlier: (
                    f"y_{identity_root(ref, shape, idx, kind)}[r][e]"
                    if identity_root(ref, shape, idx, kind) in earlier else None)
                producer = nodes[root]
                inner = producer.render("i", lambda arg, i2, s=None, producer=producer:
                                        operand(arg, producer.shape if s is None else s, i2, memory_loader))
                p7_hook[0] = hit
                if producer.dtype == "bfloat16":
                    inner = f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
                virtual_texts.append((f"y_{root}", group.regify(inner)))
            return group, entries, columns, folded, tiles, virtual_texts
        counts = {}
        built = build((), counts)
        if built is None:
            return None, None, ()
        virtual = []
        if p7_row_virtual:
            for root in sorted((r_ for r_, k in counts.items() if k >= 2), key=lambda r_: order.get(r_, 0)):
                producer = nodes[root]
                try:
                    text_ = producer.render("i", lambda arg, i2, s=None, producer=producer:
                                            operand(arg, producer.shape if s is None else s, i2, memory_loader))
                except LoweringError:
                    continue                     # reads a register-only value of the group
                # Only values available before the phase: nothing the group itself stores (statistics/members).
                if not any(re.search(r"\b" + re.escape(w_) + r"\b", text_) for w_ in written):
                    virtual.append(root)
        if virtual:
            built = build(virtual, {})
            if built is None:
                return None, None, ()
        group, entries, columns, folded, tiles, virtual_texts = built
        ncol = sum(len(c[0]) for c in columns)
        rb = p7_rows.choose_rb(rows, group.row_registers() + len(virtual_texts) * (extent // 32),
                               RESIDENT_THREADS // 32)
        if tiles:
            rb = max(1, min(rb, -(-tiles[1] // 8)))
            nodes[group_names[0]].extra["shared_bytes"] = max(nodes[group_names[0]].extra.get("shared_bytes", 0),
                                                              ncol * (extent // 32) * 256 * 4)
        if VECTOR_HELPERS not in helper_parts:
            helper_parts.append(VECTOR_HELPERS)
        rb = min(rb, max(1, int(p7_row_rb_cap)))
        text = p7_rows.code(group, entries, rb=rb, lo="/*ROW_LO*/0u", tiles=tiles, columns=columns,
                            virtuals=virtual_texts, prefetch=p7_row_prefetch)
        engines_here = {(nodes[n].extra or {}).get("engine") for n in phase["heavy"]}
        if p7_outline and "pops3c_col" not in engines_here:
            text = p7_outline_block(text, "p7_row")
        p7_row_records.append({"rows": rows, "extent": extent, "statistics": [o for o, _ in outs.values()],
                               "rb": rb, "element_operands": sorted(group.element),
                               "column_operands": sorted(group.column), "folded_columns": folded,
                               "tiles": list(tiles) if tiles else None,
                               "virtual": [reg for reg, _ in virtual_texts]})
        return group_names, text, folded

    _STAGEABLE = ("float32", "bfloat16")
    functor_types = {}

    def _canonical_names(text, names):
        for k, name in enumerate(names):
            text = re.sub(r"\b" + re.escape(name) + r"\b", f"m{k}", text)
        for k, local in enumerate(dict.fromkeys(re.findall(r"\b[ejr]_\w+\b", text))):
            text = re.sub(r"\b" + re.escape(local) + r"\b", f"t{k}", text)
        return text

    def canonical_functor(result, parameters, body, method="operator()"):
        """Generated functor as a struct whose pointers are members, so identical per-layer code shares one type
        (and one matrix body instantiation). Returns the construction expression for the call site."""
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", body)))
        canonical = _canonical_names(body, names)
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        key = (result, parameters, fields, canonical, method)
        if key not in functor_types:
            type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            functor_types[key] = type_name
            extra = ""
            if method == "pair":
                extra = "  static constexpr bool kPair = true;\n"
                method = "operator()"
            if method == "tile":
                extra = ("  static constexpr bool kTile = true;\n"
                         "  __device__ __forceinline__ void operator()(unsigned, float, float) const {}\n")
            helper_parts.append(f"struct {type_name} {{\n  {fields}\n{extra}  __device__ __forceinline__ {result} "
                                f"{method}({parameters}) const {{\n    {canonical}\n  }}\n}};")
        return f"{functor_types[key]}{{{', '.join(names)}}}"

    def p4a_emit(name, spec):
        """Outlined P-OPS4-A call: canonical leaf functor (pointer members) + shared template instantiation."""
        if spec["mode"] == "finalize":
            code = nodes[name].code
        elif spec["mode"] == "column_last":
            return p4a_last_call(name, spec)
        else:
            from .pops4a_rules_p7a import OUTLINED_HEADER, INLINE_HEADER
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
                    from .pops4a_rules_p7a import PF_HEADER
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
        from . import pops4a_rules_p7a as R
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
        from . import pops4a_rules_p7a as R
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

    def p4a_vector_load(n, ref, expression, classify):
        """One leaf load of a vector P-OPS4-A body: FP32 float4, (P6-A) BF16 4-element 8-byte loads widened exactly
        (bits to the upper half), or a broadcast scalar; None when unproven."""
        fp32 = re.fullmatch(r"(v_\w+)\[(.*)\]", expression)
        bf16 = re.fullmatch(r"__bfloat162float\((v_\w+)\[(.*)\]\)", expression)
        if dtypes[ref] == "float32" and fp32:
            array, index = fp32.groups()
        elif dtypes[ref] == "bfloat16" and bf16 and p6a_bf16_vector:
            array, index = bf16.groups()
        else:
            return None
        kind = classify(index)
        layout = layouts[ref]
        base = layout.offset + (0 if layout.root in inputs else offsets.get(layout.root, 1))
        if kind == "bcast":
            return f"const float q{n} = {expression};", [f"q{n}"] * 4
        if kind != "vec" or base % 4:
            return None
        if dtypes[ref] == "float32":
            return (f"const float4 q{n} = *reinterpret_cast<const float4*>(&{expression});",
                    [f"q{n}.{c}" for c in "xyzw"])
        if layout.root not in inputs and offsets.get(layout.root, 1) % 2:
            return None                       # 8-byte alignment of the BF16 storage (arena words are 4 bytes)
        return (f"const uint2 q{n} = *reinterpret_cast<const uint2*>(&{array}[{index}]);",
                [f"__uint_as_float(q{n}.x << 16)", f"__uint_as_float(q{n}.x & 0xffff0000u)",
                 f"__uint_as_float(q{n}.y << 16)", f"__uint_as_float(q{n}.y & 0xffff0000u)"])

    def p4a_vector_call(name, spec, parts):
        """Vectorized column item (pops4a_rules.VECTOR_HEADER) when every leaf load is proven unit-stride and
        16-byte aligned in the column index, or independent of it; None keeps the scalar body."""
        from . import pops4a_rules_p7a as R
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
            decl = p4a_vector_load(n, ref, expression, lambda text: R.classify_load(text, spec["E"], spec["N"]))
            if decl is None:
                return None
            declarations.append(decl[0])
            components[n] = decl[1]
        placeholder = re.compile(r"@@Q(\d+)@@")
        lanes = []
        for k, t in enumerate(texts):
            for e in range(4):
                value = placeholder.sub(lambda m, e=e: components[int(m.group(1))][e], t)
                lanes.append(f"v[{k}][{e}] = {value};")
        body = " ".join(declarations) + " " + " ".join(lanes)
        functor = canonical_functor("void", "unsigned pr, unsigned o, float (*v)[4]", body)
        from .pops4a_rules_p7a import VECTOR_HEADER
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
        from . import pops4a_rules_p7a as R
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
            decl = p4a_vector_load(n, ref, expression, lambda text: R.classify_full_load(text, spec["S"]))
            if decl is None:
                return None
            declarations.append(decl[0])
            components[n] = decl[1]
        placeholder = re.compile(r"@@Q(\d+)@@")
        lanes = [f"v[{e}] = " + placeholder.sub(lambda m, e=e: components[int(m.group(1))][e], text) + ";"
                 for e in range(4)]
        functor = canonical_functor("void", "unsigned s, float* v", " ".join(declarations) + " " + " ".join(lanes))
        from .pops4a_rules_p7a import VECTOR_FULL_HEADER
        if VECTOR_FULL_HEADER not in helper_parts:
            helper_parts.append(VECTOR_FULL_HEADER)
        nodes[name].extra["vectorized"] = True
        return (f"latch_p4v_full({functor}, {spec['part']}, {spec['S']}u, {spec['CH']}u, {spec['NC']}u, "
                f"{spec['rot'] % 4096}u, smem);")

    def p4a_staged_call(name, spec, parts):
        """cp.async-staged column item (pops4a_rules.STAGED_HEADER); None when an operand is not FP32."""
        from .pops4a_rules_p7a import STAGED_HEADER, stage_depth
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

    def epilogue_pair_code(product, members, colsums=()):
        st_members = []                     # P7-A: stored members in order (bulk-store staging slots)
        """P6-A register-direct epilogue for warpgroup-MMA products: per output pair (flat index i0, i0 + 1) the
        identical per-operation member expressions on the rounded BF16 product values; operands load directly."""
        spec = nodes[product].extra["matrix_spec"]
        size = spec["m"] * spec["n"]
        member_set = set(members)
        if permuted_members and any(m in permuted_members for m in members):
            raise LoweringError("internal: permuted members are not admitted on pair epilogues")

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            identity = (idx == "i" and shape is not None and _numel(shape) == size and layouts[ref].numel == size
                        and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin == product:
                return "gv"
            if identity and origin in member_set:
                expression = f"r_{origin}"
                return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        loads_seen = {}

        def batched_loader(ref, shape, idx):
            text = loader(ref, shape, idx)
            if text == "gv" or text.startswith("r_") or text.startswith("__bfloat162float(r_"):
                return text
            if text not in loads_seen:
                loads_seen[text] = f"q{len(loads_seen)}"
            return f"@@L{loads_seen[text]}@@"
        lines = []
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, batched_loader))
            lines.append(f"const {_CTYPES[node.dtype]} r_{name} = {ctype_round(expression, node.dtype)};")
            if name in stored:
                st_index = len(st_members)
                st_members.append(name)
                lines.append(f"@@ST{st_index}@@{_ident(name)}[i] = r_{name};")
        # P6-A load/apply protocol: every operand load of a pair is issued in load(); apply() computes and stores.
        load_lines = [f"L.{field}[e] = {text};" for text, field in loads_seen.items()]
        member_lines = list(lines)
        lines += [f"cs[{2 * k} + e] = (float)(r_{summed});" for k, (summed, _) in enumerate(colsums)]
        apply_lines = [re.sub(r"@@L(q\d+)@@", lambda m: f"L.{m.group(1)}[e]", line) for line in lines]
        load_body = ("Ld L;\n#pragma unroll\n    for (int e = 0; e < 2; ++e) {\n      const unsigned i = i0 + (unsigned)e;\n      "
                     + "\n      ".join(load_lines) + "\n    }\n    return L;")
        apply_body = ("#pragma unroll\n    for (int e = 0; e < 2; ++e) {\n      const unsigned i = i0 + (unsigned)e; "
                      "const float gv = e ? g1 : g0;\n      " + "\n      ".join(apply_lines) + "\n    }")
        if p7_pair_stores and st_members:
            # P7-A: stored members leave as one packed pair per column pair (bf16x2 / float2; even columns, so the
            # pair is 4- / 8-byte aligned) instead of two scalar stores; the stored values are unchanged.
            pair_decl = " ".join(f"{_CTYPES[nodes[n].dtype]} e_st{k_}[2];" for k_, n in enumerate(st_members))
            pair_lines = [re.sub(r"@@ST(\d+)@@\w+\[i\] = (r_\w+);",
                                 lambda m_: f"e_st{int(m_.group(1))}[e] = {m_.group(2)};", l_)
                          for l_ in apply_lines]
            pair_out = []
            for k_, n in enumerate(st_members):
                if dtypes[n] == "bfloat16":
                    pair_out.append(f"{{ __nv_bfloat162 e_t2; e_t2.x = e_st{k_}[0]; e_t2.y = e_st{k_}[1]; "
                                    f"*reinterpret_cast<__nv_bfloat162*>({_ident(n)} + i0) = e_t2; }}")
                else:
                    pair_out.append(f"*reinterpret_cast<float2*>({_ident(n)} + i0) = make_float2(e_st{k_}[0], e_st{k_}[1]);")
            apply_body = (pair_decl + "\n#pragma unroll\n    for (int e = 0; e < 2; ++e) {\n      const unsigned i = i0 + (unsigned)e; "
                          "const float gv = e ? g1 : g0;\n      " + "\n      ".join(pair_lines) + "\n    }\n    "
                          + " ".join(pair_out))
        # P7-A bulk stores: apply_st writes stored members into staged shared tiles (local offset loc + e).
        st_body = re.sub(r"@@ST(\d+)@@\w+\[i\] = (r_\w+);",
                         lambda m_: f"reinterpret_cast<{_CTYPES[nodes[st_members[int(m_.group(1))]].dtype]}*>"
                                    f"(sk[{m_.group(1)}])[loc + e] = {m_.group(2)};", apply_body)
        apply_body = re.sub(r"@@ST\d+@@", "", apply_body)
        st_bytes = [_BYTES[nodes[n].dtype] for n in st_members]
        # P7-A bulk loads: identity-indexed side inputs read from shared tiles prefetched by bulk copies.
        side, side_lines = [], []
        for text, field in loads_seen.items():
            m_ = re.fullmatch(r"(__bfloat162float\()?(v_\w+)\[i\](\))?", text)
            if m_ and bool(m_.group(1)) == bool(m_.group(3)) and ident_dtype.get(m_.group(2)) in _CTYPES_STAGE:
                k_ = len(side)
                side.append(m_.group(2))
                ct_ = _CTYPES[ident_dtype[m_.group(2)]]
                value = f"reinterpret_cast<const {ct_}*>(sl[{k_}])[loc + e]"
                side_lines.append(f"L.{field}[e] = {'__bfloat162float(' + value + ')' if m_.group(1) else value};")
            else:
                side_lines.append(f"L.{field}[e] = {text};")
        ld_body = ("Ld L;\n#pragma unroll\n    for (int e = 0; e < 2; ++e) {\n      const unsigned i = i0 + (unsigned)e;\n"
                   "      (void)i;\n      " + "\n      ".join(side_lines) + "\n    }\n    return L;")
        use_side = p7_bulk_load and bool(side)
        # Store tiles alias same-width side tiles: each staged element is read (load_st, batched before apply) and
        # written (apply_st) by the same thread at the same tile offset, so one shared tile serves both.
        st_alias, used_ = [], set()
        for n in st_members:
            j_ = next((j for j, x in enumerate(side) if use_side and p7_bulk_store and j not in used_
                       and _BYTES[ident_dtype[x]] == _BYTES[nodes[n].dtype]), -1)
            if j_ >= 0:
                used_.add(j_)
            st_alias.append(j_)
        vec = None
        if p7_vec_epi and spec["n"] % 8 == 0 and not use_side and not p7_bulk_store:
            vec = p7_vec_bodies(spec, loads_seen, member_lines, st_members, colsums)
        code = pair2_functor(load_body, apply_body, len(loads_seen), [ident for _, ident in colsums],
                             st_body=st_body if p7_bulk_store else None, st_bytes=st_bytes,
                             st_targets=[_ident(n) for n in st_members],
                             ld_body=ld_body if use_side else None, ld_targets=side if use_side else (),
                             ld_bytes=[_BYTES[ident_dtype[x]] for x in side] if use_side else (),
                             st_alias=st_alias, vec=vec)
        return {"code": code, "store": product not in elided_products, "members": list(members),
                "bytes_per_element": 0, "prefetch_sizes": [], "colsum": len(colsums),
                "st_bytes": [b_ for b_, a_ in zip(st_bytes, st_alias) if a_ < 0] if p7_bulk_store else [],
                "ld_bytes": [_BYTES[ident_dtype[x]] for x in side] if use_side else [], "vec": vec is not None}

    p7_vec_classifiers = {}

    def p7_vec_bodies(spec, loads_seen, member_lines, st_members, colsums):
        """P7-A vector epilogue bodies over 8 consecutive output columns of one row (i0 = flat index of the first,
        a multiple of 8; N % 8 == 0). Each operand load is classified by a numeric proof over the whole [M, N]
        output index space (p7_rows.Classifier): element-identity and per-column operands of 16-byte aligned
        FP32/BF16 values use 16-byte vector loads (per-column ones once per tile, loadc), per-row operands one load
        broadcast to the 8 elements, anything else the unchanged per-element load. Stored members leave by 16-byte
        stores when aligned. Member expressions are unchanged."""
        from . import p7_rows
        m_, n_ = spec["m"], spec["n"]
        if m_ * n_ >= 2 ** 31:
            return None
        if (m_, n_) not in p7_vec_classifiers:
            p7_vec_classifiers[(m_, n_)] = p7_rows.Classifier(m_, n_)
        classify = p7_vec_classifiers[(m_, n_)]
        lv, lc, lv_lines, lc_lines, raw = [], [], [], [], set()
        for text, field in loads_seen.items():
            mt = re.fullmatch(r"(__bfloat162float\()?(v_\w+)\[(.+)\](\))?", text)
            kind, ident, idx = "scalar", None, None
            if mt and bool(mt.group(1)) == bool(mt.group(4)):
                ident, idx = mt.group(2), mt.group(3)
                c_ = classify(idx)
                vec_ok = aligned16.get(ident) and ident_dtype.get(ident) in ("float32", "bfloat16")
                if c_ == "element" and vec_ok:
                    kind = "element"
                elif c_ == "col" and vec_ok:
                    kind = "col"
                elif c_ == "row" or classify.row_only(idx):
                    kind = "row"
            if kind == "col":
                lc.append(field)
                lc_lines.append(f"{{ const unsigned i = c0; latch_vld8({ident} + ({idx}), C.{field}); }}")
            elif kind == "element" and p7_vec_raw and ident_dtype.get(ident) == "bfloat16":
                # P7-A: BF16 element operands stay packed (4 registers per 8 values) until applyv widens them.
                lv.append(field)
                raw.add(field)
                lv_lines.append(f"*reinterpret_cast<uint4*>(L.{field}) = *reinterpret_cast<const uint4*>({ident} + i0);")
            elif kind == "element":
                lv.append(field)
                lv_lines.append(f"latch_vld8({ident} + i0, L.{field});")
            elif kind == "row":
                lv.append(field)
                lv_lines.append(f"{{ const unsigned i = i0; const float t_ = {text};\n#pragma unroll\n"
                                f"      for (int e = 0; e < 8; ++e) L.{field}[e] = t_; }}")
            else:
                lv.append(field)
                lv_lines.append(f"\n#pragma unroll\n    for (int e = 0; e < 8; ++e) {{ const unsigned i = i0 + (unsigned)e; "
                                f"L.{field}[e] = {text}; }}")
        apply_lines = []
        for line in member_lines:
            line = re.sub(r"@@L(q\d+)@@", lambda m2: (f"latch_p7_bf(L.{m2.group(1)}, e)" if m2.group(1) in raw else
                                                     f"{'C' if m2.group(1) in lc else 'L'}.{m2.group(1)}[e]"), line)
            line = re.sub(r"@@ST(\d+)@@\w+\[i\] = (r_\w+);", lambda m2: f"e_st{m2.group(1)}[e] = {m2.group(2)};", line)
            apply_lines.append(line)
        apply_lines += [f"cs[{8 * k} + e] = (float)(r_{summed});" for k, (summed, _) in enumerate(colsums)]
        decl = " ".join(f"{_CTYPES[nodes[n].dtype]} e_st{k}[8];" for k, n in enumerate(st_members))
        outs = []
        for k, n in enumerate(st_members):
            if aligned16.get(_ident(n)) and nodes[n].dtype in ("float32", "bfloat16"):
                outs.append(f"latch_vst8({_ident(n)} + i0, e_st{k});")
            else:
                outs.append(f"\n#pragma unroll\n    for (int e = 0; e < 8; ++e) {_ident(n)}[i0 + (unsigned)e] = e_st{k}[e];")
        applyv = (decl + "\n#pragma unroll\n    for (int e = 0; e < 8; ++e) {\n      const unsigned i = i0 + (unsigned)e; "
                  "const float gv = g[e]; (void)i;\n      " + "\n      ".join(apply_lines) + "\n    }\n    " + " ".join(outs))
        return {"lv": lv, "lc": lc, "loadv": "Lv L;\n    " + "\n    ".join(lv_lines) + "\n    return L;",
                "loadc": "Lc C;\n    " + ("\n    ".join(lc_lines) if lc_lines else "C.unused[0] = 0.0f;") + "\n    return C;",
                "applyv": applyv, "rb": int(p7_vec_rb), "raw": sorted(raw)}

    def pair2_functor(load_body, apply_body, n_loads, colsum_ptrs=(), st_body=None, st_bytes=(), st_targets=(),
                      ld_body=None, ld_targets=(), ld_bytes=(), st_alias=(), vec=None):
        colsum_text = " ".join(f"{p_}" for p_ in colsum_ptrs)
        vec_text = "" if vec is None else "\n".join((vec["loadv"], vec["loadc"], vec["applyv"]))
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", load_body + "\n" + apply_body + "\n" + colsum_text
                                              + "\n" + vec_text)))
        load_c, apply_c = _canonical_names(load_body, names), _canonical_names(apply_body, names)
        st_c = _canonical_names(st_body, names) if st_body is not None else None
        st_fields = [_canonical_names(f_, names) for f_ in st_targets]
        ld_c = _canonical_names(ld_body, names) if ld_body is not None else None
        side_fields = [_canonical_names(f_, names) for f_ in ld_targets]
        colsum_c = [_canonical_names(p_, names) for p_ in colsum_ptrs]
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        ld_fields = " ".join(f"float q{k}[2];" for k in range(n_loads)) or "float unused[1];"
        cs_param = ", float* cs" if colsum_ptrs else ""
        cs_method = ("  __device__ __forceinline__ float* colsum(int k) const {\n    "
                     + " ".join(f"if (k == {k}) return {c};" for k, c in enumerate(colsum_c)) + " return nullptr;\n  }\n"
                     if colsum_ptrs else "")
        vec_c = None
        if vec is not None:
            vec_c = {k_: _canonical_names(vec[k_], names) for k_ in ("loadv", "loadc", "applyv")}
            if P7_VEC_HELPERS not in helper_parts:
                helper_parts.append(P7_VEC_HELPERS)
        key = ("pair2", fields, load_c, apply_c, n_loads, tuple(colsum_c), st_c, tuple(st_bytes), ld_c,
               tuple(ld_bytes), tuple(st_alias), None if vec is None else (tuple(sorted(vec_c.items())), tuple(vec["lv"]), tuple(vec["lc"]), vec["rb"], bool(p7_vec_early), tuple(vec.get("raw", ())),
               bool(p7_vec_pipe)))
        if key not in functor_types:
            type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            functor_types[key] = type_name
            helper_parts.append(
                f"struct {type_name} {{\n  {fields}\n  static constexpr bool kPair = false;\n"
                f"  static constexpr bool kPair2 = true;\n  static constexpr int kColSum = {len(colsum_ptrs)};\n"
                + ("  static constexpr bool kStaged = true;\n"
                   if p7_staged_epi and (p7_staged_colsum or not colsum_ptrs) else "")
                + ("  static constexpr bool kIndep = true;\n"
                   if p7_staged_epi and p7_wg_indep and (p7_staged_colsum or not colsum_ptrs) else "") +
                f"  struct Ld {{ {ld_fields} }};\n"
                f"  __device__ __forceinline__ Ld load(unsigned i0) const {{\n    {load_c}\n  }}\n"
                + ("" if vec_c is None else
                   f"  static constexpr int kVec = 8;\n  static constexpr int kVecRB = {vec['rb']};\n"
                   + ("  static constexpr bool kVecEarly = true;\n" if p7_vec_early else "")
                   + ("  static constexpr bool kVecPipe = true;\n" if p7_vec_pipe else "") +
                   f"  struct Lv {{ {' '.join((f'__nv_bfloat162 {f_}[4];' if f_ in vec.get('raw', ()) else f'float {f_}[8];') for f_ in vec['lv']) or 'float unused[1];'} }};\n"
                   f"  struct Lc {{ {' '.join(f'float {f_}[8];' for f_ in vec['lc']) or 'float unused[1];'} }};\n"
                   f"  __device__ __forceinline__ Lc loadc(unsigned c0) const {{\n    {vec_c['loadc']}\n  }}\n"
                   f"  __device__ __forceinline__ Lv loadv(unsigned i0) const {{\n    {vec_c['loadv']}\n  }}\n"
                   f"  __device__ __forceinline__ void applyv(unsigned i0, const float* g, const Lv& L, const Lc& C"
                   f"{cs_param}) const {{\n    {vec_c['applyv']}\n  }}\n") +
                f"  __device__ __forceinline__ void apply(unsigned i0, float g0, float g1, const Ld& L{cs_param}) const {{\n"
                f"    {apply_c}\n  }}\n{cs_method}"
                + ("" if st_c is None else
                   f"  static constexpr int kSt = {len(st_bytes)};\n"
                   f"  __host__ __device__ static constexpr int st_bytes(int k) {{ return "
                   + " ".join(f"k == {k_} ? {b_} :" for k_, b_ in enumerate(st_bytes)) + " 0; }\n"
                   + f"  __host__ __device__ static constexpr int st_alias(int k) {{ return "
                   + " ".join(f"k == {k_} ? {a_} :" for k_, a_ in enumerate(st_alias or [-1] * len(st_bytes)))
                   + " -1; }\n"
                   f"  __device__ __forceinline__ void apply_st(unsigned i0, unsigned loc, float g0, float g1, "
                   f"const Ld& L{cs_param}, char* const* sk) const {{\n    {st_c}\n  }}\n"
                   f"  __device__ __forceinline__ char* st_ptr(int k) const {{\n    "
                   + " ".join(f"if (k == {k_}) return reinterpret_cast<char*>({f_});" for k_, f_ in enumerate(st_fields))
                   + " return nullptr;\n  }\n")
                + ("" if ld_c is None else
                   f"  static constexpr int kLdSt = {len(ld_bytes)};\n"
                   f"  __host__ __device__ static constexpr int ld_bytes(int k) {{ return "
                   + " ".join(f"k == {k_} ? {b_} :" for k_, b_ in enumerate(ld_bytes)) + " 0; }\n"
                   f"  __device__ __forceinline__ Ld load_st(unsigned i0, unsigned loc, char* const* sl) const {{\n"
                   f"    {ld_c}\n  }}\n"
                   f"  __device__ __forceinline__ const char* ld_ptr(int k) const {{\n    "
                   + " ".join(f"if (k == {k_}) return reinterpret_cast<const char*>({f_});" for k_, f_ in enumerate(side_fields))
                   + " return nullptr;\n  }\n")
                + "};")
        return f"{functor_types[key]}{{{', '.join(names)}}}"

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

    phase_source, phase_records = [], []
    item_ticks, item_tick_base = [], [len(phases) + 2]
    for k, phase in enumerate(phases):
        parts, gemm_offset, rotate = [], 0, 0
        matrix_group = None
        # P-OPS4-A items run after the phase's other heavy items; the existing emission sees only the others.
        p4a_here = [n for n in phase["heavy"] if p4a_item(nodes[n])]
        p7_folded_names = set()
        full_heavy = phase["heavy"]
        phase = dict(phase, heavy=[n for n in full_heavy if n not in p4a_here])
        group_specs = [nodes[name].extra.get('matrix_spec') for name in phase['heavy']]
        phase_epilogues = phase.get('epilogue', {})
        if group_specs and all(group_specs) and (phase_epilogues or pops2b_plans
                                                 or any(n in absorbed_by for n in phase['heavy'])):
            from .pops2b_rules_p7a import HEADER as EPI_HEADER, emit_group as emit_epilogue_group
            if EPI_HEADER not in extra_headers:
                extra_headers.append(EPI_HEADER)
                helper_parts.append(f'#include "{EPI_HEADER}"')
            extras = []
            for name in phase['heavy']:
                extra = {}
                if name in phase_epilogues:
                    extra['epi'] = (epilogue_pair_code(name, phase_epilogues[name], colsum_of.get(name, ()))
                                    if nodes[name].extra['matrix_spec'].get('wg') else
                                    epilogue_code(name, phase_epilogues[name]))
                for which, cast in absorbed_by.get(name, {}).items():
                    extra['p' + which] = prologue_code(cast, name, which)
                extras.append(extra or None)
            code, matrix_group = emit_epilogue_group(group_specs, extras, ident=_ident, plans=pops2b_plans)
            for member, extra_ in zip(matrix_group.get('members', []), extras):
                ncs = ((extra_ or {}).get('epi') or {}).get('colsum', 0)
                if (((p7_staged_epi and (p7_staged_colsum or not ncs)) or ((extra_ or {}).get('epi') or {}).get('vec'))
                        and (extra_ or {}).get('epi')
                        and 'pair2' in str(((extra_ or {}).get('epi') or {}).get('kind', 'pair2'))):
                    # P7-A staged epilogue: column-sum slabs plus the padded BF16 warpgroup sub-tiles.
                    mp = member['matrix_plan']
                    if mp.get('wg'):
                        wtn = mp['bn'] // (2 // mp['wm'])
                        wtm = mp['bm'] // mp['wm']
                        need = ((mp['bm'] + mp['bn']) * 128 * mp['nst'] + 2048 + 2 * 4 * max(ncs, 1) * wtn * 4
                                + 2 * wtm * (wtn + 8) * 2
                                + sum(2 * wtm * (wtn + 8) * b_ for b_ in (extra_['epi'].get('st_bytes') or []))
                                + sum(2 * wtm * (wtn + 8) * b_ for b_ in (extra_['epi'].get('ld_bytes') or [])))
                        if p7_wg_indep:
                            # P7-A independent warpgroups: two rings (indep_stages) + per-warpgroup BM x BN staging.
                            bm_, bn_ = mp['bm'], mp['bn']
                            tail_ = 2048 + 2 * 4 * max(ncs, 1) * bn_ * 4 + 2 * bm_ * (bn_ + 8) * 2
                            cap_ = int(p7_indep_smem or 232448)
                            nsti = next((n_ for n_ in (4, 3) if 2 * n_ * (bm_ + bn_) * 128 + 1024 + tail_ <= cap_), 2)
                            need = 2 * nsti * (bm_ + bn_) * 128 + 1024 + tail_
                        matrix_group['shared_bytes'] = max(matrix_group.get('shared_bytes', 0), need)
                if ncs:
                    # P6-A epilogue column partials: slab scratch after the stage ring (pops4b_tma_p7a.cuh layout).
                    mp = member['matrix_plan']
                    wtn = mp['bn'] // (2 // mp['wm'])
                    need = (mp['bm'] + mp['bn']) * 128 * mp['nst'] + 2048 + 2 * 4 * ncs * wtn * 4
                    matrix_group['shared_bytes'] = max(matrix_group.get('shared_bytes', 0), need)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        elif group_specs and all(group_specs):
            from .matrix_rules_p7a import emit_group
            code, matrix_group = emit_group(group_specs, ident=_ident)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        else:
            engines = {nodes[n].extra.get("engine"): n for n in phase["heavy"]}
            # P6-A: independent same-shape row engines of the phase share one fused pass.
            row_group = [n for n in phase["heavy"] if (nodes[n].extra or {}).get("engine") == "pops3c_row"]
            fused_rows, fused_kind = None, None
            if (p6a_row_fuse and len(row_group) >= 2 and not any(n in row_pre for n in row_group)
                    and len({(nodes[n].extra["row_rows"], nodes[n].extra["row_extent"]) for n in row_group}) == 1
                    and all("row_leaf" in nodes[n].extra for n in row_group)):
                from .norm_rules_p7a import GRID_CAP as _gc, warps_per_row as _wpr
                dependent = any(root in row_group[:i] or attached_to.get(root) in row_group[:i]
                                for i, n in enumerate(row_group) for root, _, _, _ in loads_of[n])
                if not dependent:
                    fused_rows, fused_kind = row_group, "multi"
                elif "pops3c_col" not in engines and _wpr(nodes[row_group[0]].extra["row_rows"], _gc) == 1:
                    fused_rows, fused_kind = row_group, "chain"
            p7_group, p7_text, p7_folded = (p7_row_group(phase, p4a_here) if p7_rowreg and not retain_all
                                            and not fused_rows else (None, None, ()))
            p7_folded_names = set(p7_folded)
            for position, name in enumerate(phase["heavy"]):
                node = nodes[name]
                if fused_rows and name in fused_rows[1:]:
                    continue
                if p7_group and name in p7_group[1:]:
                    continue
                if p7_group and name == p7_group[0]:
                    code = p7_text
                elif fused_rows and name == fused_rows[0]:
                    from .norm_rules_p7a import row_code_multi, row_code_chain
                    first = nodes[name].extra
                    multi = (row_code_multi if fused_kind == "multi" else row_code_chain)(
                        rows=first["row_rows"], extent=first["row_extent"],
                        entries=[(nodes[n].extra["row_leaf"], nodes[n].extra["row_store"]) for n in fused_rows])
                    code = heavy_code(SimpleNamespace(name=name, code=multi))
                    for k_, n in enumerate(fused_rows):
                        members = phase.get("rows", {}).get(n, [])
                        code = code.replace(f"/*ROW_MEMBERS_{k_}*/", row_members_code(n, members) if members else "")
                else:
                    code = heavy_code(node)
                engine = node.extra.get("engine")
                if engine == "pops3c_row":
                    from .norm_rules_p7a import GRID_CAP, warps_per_row
                    members = phase.get("rows", {}).get(name, [])
                    code = code.replace("/*ROW_MEMBERS*/", row_members_code(name, members) if members else "")
                    blocks = GRID_CAP
                    if "pops3c_col" in engines and not pops3c_overlap:
                        groups = nodes[engines["pops3c_col"]].extra["column_groups"]
                        code = code.replace("/*ROW_LO*/0u", f"min({groups}u, gridDim.x / {pops3c_col_div}u)")
                        blocks = GRID_CAP - min(groups, GRID_CAP // pops3c_col_div)
                    wr = warps_per_row(node.extra["row_rows"], blocks)
                    code = code.replace("/*ROW_WR*/8u", f"{wr}u").replace("/*ROW_W*/", str(wr))
                    rbn = 1
                    if p6a_row_batch and p6a_row_batch > 1:
                        iterations = -(-node.extra["row_rows"] // (blocks * (8 // wr)))
                        rbn = max(1, min(int(p6a_row_batch), iterations))
                    code = code.replace("/*ROW_RB*/1u", f"{rbn}u")
                elif engine == "pops3c_col" and "pops3c_row" in engines and not pops3c_overlap:
                    groups = node.extra["column_groups"]
                    code = code.replace(f"/*COL_SPAN*/min({groups}u, gridDim.x)",
                                        f"min({groups}u, gridDim.x / {pops3c_col_div}u)")
                if name in deferred:
                    info = deferred[name]
                    spec = node.extra["matrix_spec"]
                    if name in phase.get("epilogue", {}):
                        from .pops2b_rules_p7a import HEADER as EPI_HEADER, _code as epi_matrix_code, tile_plan
                        if EPI_HEADER not in extra_headers:
                            extra_headers.append(EPI_HEADER)
                            helper_parts.append(f'#include "{EPI_HEADER}"')
                        extra = {"epi": epilogue_code(name, phase["epilogue"][name])}
                        code = epi_matrix_code(spec, tile_plan(spec, info["plan_grid"], extra), _ident,
                                               *info["slice"], 0, extra)
                    else:
                        from .matrix_rules_p7a import _code as matrix_code, _plan as matrix_plan
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
        matrix_only = phase["heavy"] and all(nodes[n].kind == "gemm" for n in phase["heavy"])
        exclude = gemm_offset if matrix_only else 0
        main_heavy = [n for n in phase["heavy"] if n not in deferred]
        if (pops2b and pops2b_idle and target_arch == "sm_90" and main_heavy and not matrix_only
                and all(nodes[n].kind in IDLE_HEAVY_KINDS and nodes[n].extra.get("work_items") for n in main_heavy)):
            exclude = sum(int(nodes[n].extra["work_items"]) for n in main_heavy)
        for name in p4a_here:
            if name in p7_folded_names:
                continue                     # P7-A: folded into the phase's register row engine
            if parts:
                parts.append("__syncthreads();  // shared memory reuse before a P-OPS4-A item")
            spec = nodes[name].extra.get('p4a_spec')
            code = p4a_emit(name, spec) if spec else heavy_code(nodes[name])
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
        loop_parts = [(numel, part_members) for numel, members in phase["loops"].items()
                      for part_members in split_loop(numel, members)]
        weights = [numel * len(pm) if numel > 1 else 0 for numel, pm in loop_parts]
        shared_ranges = p6a_loop_ranges and not exclude and sum(1 for w in weights if w) >= 2
        total_w, cursor = sum(weights), 0
        scale_w = max(1, total_w // 4096)            # integer weights in 32-bit range
        pipelined, rot_cursor = [], 0
        if p7_flat_loops and not exclude and not retain_all and not shared_ranges:
            flat_ = [(numel, pm) for numel, pm in loop_parts
                     if numel > 1 and numel % 4 == 0 and all(nodes[n].dtype in ("float32", "bfloat16") for n in pm)]
            if len(flat_) >= 2 and sum(numel // 4 for numel, _ in flat_) >= 4096:
                built_ = p7_flat_loops_code(flat_)
                if built_ is not None:
                    vector_helpers_needed.append(True)
                    parts.append(built_[0])
                    p7_flat_records.append({"phase": k, "segments": built_[0].count("const unsigned q = f -"),
                                            "invariants": built_[1]})
                    loop_parts = [lp for lp in loop_parts if lp not in flat_]
        for k_, (numel, part_members) in enumerate(loop_parts):
            # P6-A: independent one-element chains run on different warps (lane 0 of warp k) concurrently.
            slot = (32 * k_) % RESIDENT_THREADS if (numel == 1 and p6a_split_loops) else 0
            share = None
            if shared_ranges and weights[k_]:
                w_ = max(1, weights[k_] // scale_w)
                share = (cursor, cursor + w_, max(1, total_w // scale_w + len(loop_parts)))
                cursor += w_
            if (p6a_pipeline and not exclude and not share and not slot and numel > 1 and numel % 4 == 0
                    and all(nodes[n].dtype in ("float32", "bfloat16") for n in part_members)):
                pipelined.append(loop_code(numel, part_members, exclude, pipeline=True))
                continue
            rot = 0
            if p6a_rotate and not exclude and not share and not slot and 1 < numel < RESIDENT_THREADS:
                vec_ = (all(nodes[n].dtype in ("float32", "bfloat16") for n in part_members) and numel % 4 == 0
                        and numel >= VECTOR_MIN_ELEMENTS)
                width = -(-(numel // 4 if vec_ else numel) // 32) * 32
                if rot_cursor + width > RESIDENT_THREADS:
                    rot_cursor = 0
                rot, rot_cursor = rot_cursor, rot_cursor + width
            parts.append(loop_code(numel, part_members, exclude, slot=slot, share=share, rot=rot))
        if pipelined:
            if VECTOR_HELPERS not in helper_parts:
                vector_helpers_needed.append(True)
            parts.append(pipelined_loops(pipelined))
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
        all_names = [*phase["heavy"], *epilogue_names, *loop_names]
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
                             and n not in elided_products],
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
        if p6a_item_ticks and len(parts) > 1:
            # P6-A diagnostic: CTA0 clock after each item of the phase (profile runs only; ticks is null otherwise).
            stamped = []
            for k_, part_ in enumerate(parts):
                stamped.append(part_)
                if k_ + 1 < len(parts) and not part_.startswith("__syncthreads"):
                    slot_ = item_tick_base[0] + len(item_ticks)
                    label_ = part_.split("\n", 1)[0][:90]
                    item_ticks.append({"phase": k, "tick": slot_, "label": label_})
                    stamped.append(f"if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{slot_}] = stamp();")
            parts = stamped
        phase_source.append(f"// Phase {k}: {', '.join(all_names)}\n{{\n" + scoped("\n".join(parts))
                            + f"\n}}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{k+1}] = stamp();")
    prefetches = []
    if pops2b and pops2b_prefetch and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p7a import prefetch_plan
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
    p2_prologue = ""
    if p2_counter_words:
        p2_partbase = f"{p2_base + p2_counter_words}LL"
        phase_source = [code.replace('/*P2_PARTBASE*/', p2_partbase) for code in phase_source]
        p2_prologue = (f"  // P-OPS2-A split-K arrival counters start at zero; every last arriver restores zero.\n"
                       f"  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {_u(p2_counter_words)}; "
                       f"i += gridDim.x * blockDim.x) reinterpret_cast<unsigned*>(arena + {p2_base}LL)[i] = 0u;\n"
                       + ("  grid.sync();\n" if p2_first_split_phase == 0 and bar_offset is None else ""))
        arena_words = p2_base + p2_counter_words + p2_partial_words
    dyn_prologue = ""
    dyn_count = sum(code.count('/*P7_DYN*/nullptr') for code in phase_source)
    if dyn_count:
        # P7-A dynamic tile claims: one zeroed counter word per dynamic product instance (after all other scratch).
        dyn_base = arena_plan.a64(arena_words)
        cursor = [0]

        def take(_m):
            cursor[0] += 1
            return f"reinterpret_cast<unsigned*>(arena + {dyn_base + cursor[0] - 1}LL)"
        phase_source = [re.sub(r"/\*P7_DYN\*/nullptr", take, code) for code in phase_source]
        arena_words = dyn_base + dyn_count
        dyn_prologue = (f"  // P7-A dynamic tile claim counters start at zero in every launch.\n"
                        f"  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {_u(dyn_count)}; "
                        f"i += gridDim.x * blockDim.x) reinterpret_cast<unsigned*>(arena + {dyn_base}LL)[i] = 0u;\n"
                        + ("  grid.sync();\n" if bar_offset is None else ""))
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
                   + p2_prologue + dyn_prologue + barrier_setup
                   + "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n" + "\n".join(phase_source)
                   + tail + f"  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{commit+1}] = stamp();\n")
    if vector_helpers_needed:
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
        from .norm_rules_p7a import OUTLINED, outline
        kernel_body, used = outline(kernel_body)
        include = '#include "attention_fp32.cuh"\n'
        if pops3c_outline_headers and include in helper_parts:
            # Same out-of-line copies for the vendored attention body's libdevice calls (scoped macro; identical
            # functions and results).
            position = helper_parts.index(include)
            helper_parts[position] = (OUTLINED + "#define expf latch_expf\n#define logf latch_logf\n" + include
                                      + "#undef expf\n#undef logf\n")
        elif used:
            helper_parts.append(OUTLINED)
    if no_calls:
        helper_parts.insert(0, "#define POPSC_PHASE_ATTR __device__ __forceinline__")
        # Index checks abort with a trap instruction instead of the assert system call (same abort condition).
        kernel_body = re.sub(r"(?<![\w.])assert\((.*)\);", r"if (!(\1)) __trap();", kernel_body)
    if p7_outline:
        # P7-A: the fused attention phase functions (one template instantiation per head shape) are called through
        # one __noinline__ wrapper each instead of being inlined at every layer (mma.sync, named barriers and
        # cp.async only; no warpgroup MMA).
        attention_calls = sorted(set(re.findall(r"pmix::(forward_phase|backward_phase|bias_phase)<([^>]*)>\(", kernel_body)))
        for fn_, targs in attention_calls:
            wname = f"p7_{fn_}_" + re.sub(r"\W+", "_", targs).strip("_")
            if fn_ == "bias_phase":
                helper_parts.append(f"__device__ __noinline__ void {wname}(const pmix::Args A, float* smem) "
                                    f"{{ pmix::{fn_}<{targs}>(A, smem); }}")
            else:
                helper_parts.append(f"__device__ __noinline__ void {wname}(const pmix::Args A, long long items, "
                                    f"float* smem) {{ pmix::{fn_}<{targs}>(A, items, smem); }}")
            kernel_body = kernel_body.replace(f"pmix::{fn_}<{targs}>(", f"{wname}(")
    if p7_ring_release:
        helper_parts.insert(0, "#define P7_RING_RELEASE 1")
    if p7_quad_sel:
        helper_parts.insert(0, "#define P7_QUAD_SEL 1")
    if p7_acc_rezero:
        helper_parts.insert(0, "#define P7_ACC_REZERO 1")
    if p7_indep_smem:
        helper_parts.insert(0, f"#define P7_INDEP_SMEM {int(p7_indep_smem)}")
    if p7_diag in ("no_epi", "no_mma", "no_load"):
        # P7-A DIAGNOSTIC ONLY (wrong results): decompose warpgroup-MMA product phases (kernels/pops4b_tma_p7a.cuh).
        helper_parts.insert(0, f"#define P7_DIAG_MODE {dict(no_epi=1, no_mma=2, no_load=3)[p7_diag]}")
    if p7_diag == "cheap_erf":
        # P7-A DIAGNOSTIC ONLY (wrong results): erff/expf replaced by one multiply, to attribute epilogue math time.
        helper_parts.insert(0, "__device__ __forceinline__ float p7_diag_f(float x) { return 0.5f * x; }")
        kernel_body = re.sub(r"(?<![\w.])(erff|expf)\(", "p7_diag_f(", kernel_body)
        helper_parts = [re.sub(r"(?<![\w.])(erff|expf)\(", "p7_diag_f(", h_) if "p7_diag_f(float" not in h_ else h_
                        for h_ in helper_parts]
    helpers = "\n".join(helper_parts)
    shared_bytes = max([17408] + [p.get("shared_bytes", 0) for p in phase_records])
    from .resources import select_program_resources
    from .resources import TWO_CTA_SHARED_MAX
    resource_policy = select_program_resources(shared_bytes, phases=phase_records, arch=target_arch,
                                               pops2_two_cta=matrix_residency == 2 or p6a_two_cta,
                                               **({"two_cta_limit": TWO_CTA_SHARED_MAX} if p6a_two_cta else {}))
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    from .matrix_rules_p7a import collect_tensor_maps
    source, tensor_maps = collect_tensor_maps(source, len(inputs))
    if p7_const_args:
        # P7-A binding table in constant memory (S108H constant-offset mechanism): every bound pointer is a uniform,
        # rematerializable constant-bank load at its use instead of a global args[] load at kernel entry kept live
        # (or spilled) across every phase. The Runtime writes the table once (latch_bind_cargs) before launches.
        n_cargs = len(inputs) + 1
        table = (f"__constant__ unsigned long long latch_cargs[{n_cargs}];\n"
                 f"extern \"C\" int latch_bind_cargs(const unsigned long long* host, int n) {{\n"
                 f"  static int bound = 0; static unsigned long long first[{n_cargs}];\n"
                 f"  if (n < 1 || n > {n_cargs}) return (int)cudaErrorInvalidValue;\n"
                 "  if (bound) {  // one bound Runtime per loaded library: a different binding is refused\n"
                 "    for (int k = 0; k < n; ++k) if (first[k] != host[k]) return (int)cudaErrorInvalidValue;\n"
                 "    return 0;\n  }\n"
                 "  cudaError_t e = cudaMemcpyToSymbol(latch_cargs, host, sizeof(unsigned long long) * n);\n"
                 "  if (e != cudaSuccess) return (int)e;\n"
                 "  for (int k = 0; k < n; ++k) first[k] = host[k];\n  bound = 1;\n  return 0;\n}\n")
        marker = 'extern "C" __global__ '
        if source.count(marker) != 1:
            raise LoweringError("p7_const_args: one kernel entry expected")
        head, tail = source.split(marker)
        end = '\nextern "C" int latch_query('
        if tail.count(end) != 1:
            raise LoweringError("p7_const_args: kernel end not found")
        body_, rest_ = tail.split(end)
        body_ = re.sub(r"(?<![\w.])args\[(\d+)\]", r"((float*)latch_cargs[\1])", body_)
        source = head + table + marker + body_ + end + rest_
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
        "compiler_policy": "integrated_v1", "target_arch": target_arch,
        "phase_count": commit + 1, "barrier_count": commit + (1 if commit_code else 0),
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
    if p6a_full_grid:
        # P6-A: every phase is written against gridDim; launch one cooperative CTA per SM (all 132 on H100).
        manifest["launch_full_device"] = True
    if p7_row_records:
        manifest["p7_row_engines"] = p7_row_records
    if p7_flat_records:
        manifest["p7_flat_loops"] = p7_flat_records
    if p7_const_args:
        manifest["p7_const_args"] = len(inputs) + 1
    if item_ticks:
        manifest["item_ticks"] = item_ticks
        manifest["tick_words"] = item_tick_base[0] + len(item_ticks)
    return GeneratedProgram(source, manifest, kernel_body, helpers)
