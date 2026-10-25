# Integrated from frozen donor 5f78a0ef1d044fccb5813bea5329b77901bdcab8; private scheduling/resource protocol.
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
RESIDENT_THREADS = 128 * 256          # default H100 launch cap x CTA width
TRANSCENDENTAL = {"tanh", "exp", "expm1", "log1p", "reciprocal", "log", "sqrt", "rsqrt", "sigmoid", "silu", "silu_backward",
                  "pow", "div", "sigmoid_backward", "tanh_backward"}
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
    """Two-pass fixed-order FP32 reduction for long extents with few results (C-OPS2-C).

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
                    permuted=None, row_of=None):
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
def lower_fused(graph, *, retain_all=False, coschedule_matrix=True, place_by_slack=True,
                matrix_policy="auto", split_k=True, exact_reductions=True, fuse_attention=True,
                ordered_index_gradients=True, target_arch="sm_90", pops2_policy=True, matrix_engine=None,
                matrix_bodies=None, attention_min_keys=None, warp_rows=None,
                matrix_runtime_layout=False, matrix_residency=1, pops2b=True, pops2b_prefetch=False,
                pops2b_epilogue=True, pops2b_permuted=False, pops2b_casts=False, pops2b_idle=True,
                pops2b_plans=False, pops2b_defer=False, pops2b_idle_fill=None, pops2b_epi_cost=None,
                pops2b_epi_stream_gate=True, pops2b_defer_epilogue=True, pops2b_barrier=True,
                attention_schedule=True, attention_store_scores=True, attention_debug_repeat=False,
                attention_pipeline=True, scan_engine="auto",
                pops3c=False, pops3c_attach=True, pops3c_share=True, pops3c_column=True, pops3c_row=True,
                pops3c_pre=False, pops3c_col_div=2, pops3c_attach_shared=False, pops3c_outline=True,
                pops3c_factor=True, pops3c_pow1=True, pops3c_outline_ops=True, pops3c_overlap=False,
                pops3c_outline_headers=False, pops3c_col_group=8,
                split_reductions=True, pops4a=False, pops4a_cols=2, pops4a_unroll=8, pops4a_chunk=4096,
                pops4a_items=256, pops4a_last=False, pops4a_outline="inline", pops4a_merge=True, pops4a_prefetch=0,
                pops4a_stage=False, pops4a_stage_bytes=65536, pops5e_store=False, pops5e_store_min=65536, pops5e_stream=False, pops5e_inline=4,
                pops5e_plan=None, pops5e_full=False, pops5e_budget=None,
                pops5e_min_elements=None, scan_options=None, p6e_trap=False, p6e_assembly=False, p6e_epilogue=False,
                p6e_split=False, p6e_licm=False, p6e_gather=False,
                p6e_fast_act=False, p6e_scan_place=False, p6e_loop_split=0, p6e_index=False, p6e_early_ld=False, p6e_last_issue=False, p6e_scalar_chain=False, p6e_quad=False, p6e_softmax_rows=0, p6e_emb_pipe=False, p6e_scatter_vec=False, p6e_fin_batch=0, p6e_live_cap=0, p6e_ld_first=False, p6e_defer=0,
                p6e_inline_div=False):
    from . import pops2b_rules_p6e as _p2b
    _p2b.P6E_TMA_EPILOGUE = bool(p6e_epilogue)       # P6-E register epilogues on p6e TMA products (per lowering)
    from . import matrix_rules_p6e as _mr
    _mr.WG_SPLIT = bool(p6e_split)                    # P6-E split-K candidates for p6e TMA products (per lowering)
    from . import transformer_lower as _tl
    _tl.P6E_VEC_GATHER = bool(p6e_gather)             # P6-E 16-byte embedding row gathers (per lowering)
    from . import pops5e_rules as _p5r
    _p5r.FIN_BATCH = int(p6e_fin_batch or 0)          # P6-E batched block-partial loads in column finalizes
    _tl.P6E_SCATTER_VEC = bool(p6e_scatter_vec)        # P6-E vectorized last-axis scalar scatter
    _tl.P6E_EMB_PIPE = bool(p6e_emb_pipe)              # P6-E pipelined embedding-gradient occurrence batches
    _tl.P6E_SOFTMAX_ROWS = int(p6e_softmax_rows or 0)  # P6-E rows per warp iteration of warp-row softmax
    from . import lower as _lw
    _lw.P6E_IDENTITY_INDEX = bool(p6e_index)          # P6-E identity index for contiguous same-shape layouts
    # A selects actual FP32 products; B schedules explicit BF16 products. D's
    # engine dictionary remains available for staged accumulation/layout paths.
    matrix_engine = ("auto" if pops2_policy else "popsb") if matrix_engine is None else matrix_engine
    matrix_bodies = (("simt", "popsb") if pops2_policy else None) if matrix_bodies is None else tuple(matrix_bodies)
    attention_override = attention_min_keys is not None
    from .lower import (GeneratedProgram, LoweringError, TensorLayout, VENDOR, _header_path, _BYTES, _CTYPES,
                        _contiguous, _dtype, _field, _flatten_refs, _float, _double, _ident, _numel,
                        _ref, _reshape_strides, _translation_unit, slice_view, slice_backward_index)
    from . import scan_lower

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
    p6e_scan_plans, p6e_skip, p6e_assembly_views = {}, set(), {}
    if p6e_scan_place:
        from .p6e_rules import scan_placements
        from .scan import BWD_TARGET as _BWD
        p6e_scan_plans = scan_placements(operations, values, names=names, ref=lambda a: _ref(a, names), field=_field,
                                         output_name=output_name, flatten_refs=_flatten_refs,
                                         bwd_target=_BWD.removeprefix("torch.ops.").removeprefix("aten."))
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
    ew_args = {}          # P6-E: elementwise root name -> (base, args, kwargs) for structural rules
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

    P5E_VIEWS = {"view", "_unsafe_view", "reshape", "squeeze", "unsqueeze", "alias", "detach", "expand", "t",
                 "transpose", "permute"}

    def p5e_space(src_shape, src_strides, dims):
        """(mode, rows, extent) of a reduction over a contiguous [rows, extent] space, else None."""
        n = len(src_shape)
        if tuple(src_strides) != tuple(_contiguous(tuple(src_shape))) or not dims:
            return None
        k = len(dims)
        if k == n:
            if not pops5e_full or n < 2 or _numel(src_shape) < (1 << 18):
                return None
            # full reduction of a dense operand: per-row partials over [shape[0], rest], ordered finalize
            # full reduction of a dense operand: per-row partials over a [rows, ext] view (any fixed order is
            # admitted only through the checks), ordered finalize; prefer >= 2048 rows of 512..4096 elements
            total = _numel(src_shape)
            mode, rows, ext = "full", int(src_shape[0]), _numel(src_shape[1:])
            for cand in range(512, 4097, 4):
                if total % cand == 0 and total // cand >= 2048:
                    rows, ext = total // cand, cand
                    break
        elif list(dims) == list(range(n - k, n)):
            mode, rows, ext = "row", _numel(src_shape[:n - k]), _numel(src_shape[n - k:])
        elif list(dims) == list(range(k)):
            mode, rows, ext = "column", _numel(src_shape[:k]), _numel(src_shape[k:])
        else:
            return None
        from .pops5e_rules import MIN_ROWS, MIN_ELEMENTS
        floor = MIN_ELEMENTS if pops5e_min_elements is None else pops5e_min_elements   # override: checks only
        if mode == "full":
            floor = min(floor, 1 << 18)          # staged full reductions pay off from 256K elements
        if rows < min(MIN_ROWS, floor or 1) or ext < 32 or rows * ext >= 2**31 or rows * ext < floor:
            return None
        return mode, rows, ext

    def p5e_pairs():
        """Row/column reductions over the same space with no reader of the first before the second."""
        cands = []
        for k, op in enumerate(operations):
            target = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.")
            if target.split(".")[0] not in {"sum", "mean"}:
                continue
            args = list(_field(op, "args", []))
            source = _ref(args[0], names)
            value = values.get(source)
            if value is None or _dtype(value) not in ("float32", "bfloat16"):
                continue
            shape_ = tuple(int(x) for x in _field(value, "shape"))
            strides_ = tuple(int(x) for x in _field(value, "strides"))
            kwargs = dict(_field(op, "kwargs", {}))
            d = args[1] if len(args) > 1 else kwargs.get("dim")
            if d is None or d == [] or d == ():
                continue
            d = sorted({int(x) % len(shape_) for x in ([d] if isinstance(d, int) else d)})
            space = p5e_space(shape_, strides_, d)
            if space and space[0] != "full":
                cands.append((k, output_name(op), space))
        readers = {}
        for k, op in enumerate(operations):
            for ref in _flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names):
                readers.setdefault(ref, []).append(k)
        pairs, used = {}, set()
        for a in range(len(cands)):
            ka, na, sa = cands[a]
            if na in used:
                continue
            for b in range(a + 1, len(cands)):
                kb, nb, sb = cands[b]
                if nb in used or sa[1:] != sb[1:] or sa[0] == sb[0]:
                    continue
                dependents, ok = {na}, True
                for kk in range(ka + 1, kb):
                    o = operations[kk]
                    refs_ = _flatten_refs((list(_field(o, "args", [])), dict(_field(o, "kwargs", {}))), names)
                    if dependents.intersection(refs_):
                        base_ = str(_field(o, "target", _field(o, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
                        if base_ not in P5E_VIEWS:
                            ok = False
                            break
                        dependents.add(output_name(o))
                if not ok:
                    break
                pairs[na] = nb
                pairs[nb] = na
                used.update((na, nb))
                break
        return pairs

    def p5e_reduction(op, name, source, src, dims, kept, extent, results, refs, rounded, scale):
        """P-OPS5-E staged dense reduction (ops/pops5e_rules.py); False leaves the existing rules."""
        from . import pops5e_rules as P5
        space = p5e_space(src.shape, src.strides, dims)
        if not space or src.offset:
            if name in p5e["pairs"]:
                raise LoweringError(f"internal: paired reduction {name} lost its dense space")
            return False
        mode, rows, ext = space
        leaf = storage_read(source, P5.position(ext))
        entry = dict(op=op, name=name, mode=mode, leaf=leaf, refs=list(refs), rounded=rounded, scale=scale,
                     source=source)
        if mode == "full":
            root = name + "_p5e"
            if root in names or root in layouts:
                raise LoweringError(f"P-OPS5-E scratch name collision: {root}")
            layouts[root] = TensorLayout((rows,), (1,), root)
            dtypes[root] = "float32"
            spec = dict(mode="row", rows=rows, extent=ext, row_out=_ident(root), col_part=None, row_scale="",
                        row_rounded=None, row_token=leaf, col_token=None)
            add_heavy("reduction", op, root, "/*P5E_ROW*/" + leaf + "/*P5E_COL*//*P5E_END*/", list(refs),
                      engine="pops5e_full", reduction_elements=rows * ext, p5e_spec=spec, p5e_share=True,
                      arithmetic="per-row frozen CTA-tree partials; row-order finalize")
            final = P5.full_finalize(part=_ident(root), out=_ident(name), rows=rows, rounded=rounded, scale=scale,
                                     block=37 * len(p5e["decisions"]) + 11)
            add_heavy("pops4a", op, name, final, [root], pops4a=True, engine="pops5e_finalize",
                      reduction_elements=rows * ext)
            p5e["decisions"].append({"root": root, "mode": "full", "rows": rows, "extent": ext, "row": None,
                                     "column": None, "full": name})
            return True
        partner = p5e["pairs"].get(name)
        if partner is None and pops5e_stream == "fused":
            return False      # 'fused': only row/column pairs sharing one operand pass
        if partner is not None and partner not in p5e["pending"]:
            p5e["pending"][name] = entry
            p5e_waiting[name] = []
            p5e_blocked[name] = name
            return True
        members = [entry] + ([p5e["pending"].pop(partner)] if partner is not None else [])
        if partner is not None:
            released = p5e_waiting.pop(partner, [])
            for key in [k for k, v in p5e_blocked.items() if v == partner]:
                del p5e_blocked[key]
            p5e_queue[0:0] = released      # views of the deferred result, now that its layout exists
        row = next((m for m in members if m["mode"] == "row"), None)
        col = next((m for m in members if m["mode"] == "column"), None)
        root = (row or col)["name"] + "_p5e"
        if root in names or root in layouts:
            raise LoweringError(f"P-OPS5-E scratch name collision: {root}")
        words = (rows if row else 0) + (P5.GRID_MAX * ext if col else 0)
        layouts[root] = TensorLayout((words,), (1,), root)
        dtypes[root] = "float32"
        part = None
        if row:
            layouts[row["name"]] = TensorLayout(tuple(int(x) for x in _field(values[row["name"]], "shape")),
                                                _contiguous(tuple(int(x) for x in _field(values[row["name"]], "shape"))),
                                                root, 0)
            dtypes[row["name"]] = "float32" if dtypes[row["name"]] == "float32" else dtypes[row["name"]]
            if dtypes[row["name"]] != "float32":
                raise LoweringError("P-OPS5-E row results are FP32")
            views.append({"name": row["name"], "target": "pops5e_row_result", "source": root})
        if col:
            part = col["name"] + "_p5e_part"
            layouts[part] = TensorLayout((P5.GRID_MAX * ext,), (1,), root, rows if row else 0)
            dtypes[part] = "float32"
            views.append({"name": part, "target": "pops5e_column_partials", "source": root})
        spec = dict(mode="fused" if row and col else mode, rows=rows, extent=ext,
                    row_out=_ident(row["name"]) if row else None, col_part=_ident(part) if col else None,
                    row_scale=row["scale"] if row else "", row_rounded=row["rounded"] if row else None,
                    row_token=row["leaf"] if row else None, col_token=col["leaf"] if col else None)
        code = "/*P5E_ROW*/" + (row["leaf"] if row else "") + "/*P5E_COL*/" + (col["leaf"] if col else "") + "/*P5E_END*/"
        all_refs = list(dict.fromkeys((row["refs"] if row else []) + (col["refs"] if col else [])))
        row_extra = dict(row_rows=rows, row_extent=ext, row_scale=row["scale"]) if row else {}
        add_heavy("reduction", (row or col)["op"], root, code, all_refs, engine="pops5e_" + spec["mode"],
                  reduction_elements=rows * ext, p5e_spec=spec, **row_extra,
                  arithmetic="rows: frozen CTA-tree order; columns: per-block row-order partials, block-order finalize")
        if col:
            final = P5.finalize(part=_ident(part), out=_ident(col["name"]), extent=ext, rounded=col["rounded"],
                                scale=col["scale"])
            add_heavy("pops4a", col["op"], col["name"], final, [part], pops4a=True, engine="pops5e_finalize",
                      reduction_elements=rows)
        p5e["decisions"].append({"root": root, "mode": spec["mode"], "rows": rows, "extent": ext,
                                 "row": row["name"] if row else None, "column": col["name"] if col else None})
        return True

    def p4a_reduction(op, name, out, source, src, dims, kept, extent, results, refs, rounded, scale):
        """P-OPS4-A hierarchical leading-axis reduction (ops/pops4a_rules.py); False leaves the existing rules."""
        from . import pops4a_rules as R
        from . import norm_rules
        exact_column = (pops3c and pops3c_column
                        and norm_rules.column_eligible(src.shape, src.strides, dims, results, extent))
        rotate = 37 * p4a_state["count"]
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
        leaf = storage_read(source, norm_rules.position(src.shape, src.strides, dims, kept, "pr", "o"))
        words = plan["tiles"] * results
        key = (tuple(src.shape), tuple(dims), extent, results, cols)
        index = p4a_state["index"]
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
                    "T": plan["tiles"], "RT": plan["rows"], "B": plan["blocks"], "C": cols, "rot": rotate}
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
        protected=set(outputs) | set(writes.values()), exclude=attention_lower.excluded_groups(), masks=True
    ) if fuse_attention else ([], [])
    if attention_min_keys and attention_groups:
        # P-OPS2-A: for short key ranges the materialized scores are small and the selected FP32 product
        # bodies plus softmax phases are faster than the fused attention phases (measured on H100).
        kept = []
        for group in attention_groups:
            keys = int(_field(values[group["kt"]], "shape")[-1]) if group.get("kt") in values else None
            if keys is not None and keys <= attention_min_keys:
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
    p5e = {"pairs": {}, "pending": {}, "decisions": [], "helpers": False}
    if pops5e_stream and not retain_all and target_arch == "sm_90" and exact_reductions:
        p5e["pairs"] = p5e_pairs()
    p5e_queue = list(operations)
    p5e_waiting = {}                       # deferred reduction name -> ops reading it (pure views), in order
    p5e_blocked = {}                       # value name -> deferred reduction it waits for
    while p5e_queue:
        op = p5e_queue.pop(0)
        if p5e_waiting:
            hit = next((p5e_blocked[r] for r in _flatten_refs((list(_field(op, "args", [])),
                        dict(_field(op, "kwargs", {}))), names) if r in p5e_blocked), None)
            if hit is not None:
                p5e_waiting[hit].append(op)
                p5e_blocked[output_name(op)] = hit
                continue
        target = str(_field(op, "target", _field(op, "op")))
        target = target.removeprefix("torch.ops.").removeprefix("aten.")
        if target in scan_lower.TARGETS:
            # One multi-result operation: packed typed result views + one exclusive heavy item.
            try:
                plan_ = p6e_scan_plans.get(output_name(op))
                emission = scan_lower.emit(op, values=values, layouts=layouts, dtypes=dtypes, names=names,
                                           ref=_ref, field=_field, ident=_ident, engine=scan_engine,
                                           options=scan_options, placement=plan_)
            except scan_lower.ScanRefused as exc:
                raise LoweringError(f"{target}: {exc}") from exc
            layouts[emission.root] = TensorLayout((emission.root_words,), (1,), emission.root)
            dtypes[emission.root] = "float32"
            placed_ = getattr(emission, "placed", {}) or {}
            for result, result_shape, result_dtype, offset in emission.outputs:
                strides_ = placed_.get(result) or _contiguous(result_shape)
                layouts[result] = TensorLayout(result_shape, tuple(strides_), emission.root, offset)
                dtypes[result] = result_dtype
                views.append({"name": result, "target": target + ":packed_result", "source": emission.root})
            if getattr(emission, "assembly", None):
                xname, xshape = emission.assembly
                p6e_skip.update(n for n in plan_["skip"] if n != xname)
                p6e_assembly_views[xname] = (emission.root, xshape)
            include = emission.include
            if include not in helper_parts:
                helper_parts.append(include)
            if emission.header not in extra_headers:
                extra_headers.append(emission.header)
            add_heavy(emission.kind, op, emission.root, emission.code, emission.reads, **emission.metadata)
            scan_decisions.append(emission.decision)
            continue
        if p6e_skip and output_name(op) in p6e_skip:
            continue                       # P6-E: assembly term written in place by its backward scan
        if output_name(op) in p6e_assembly_views:
            xname = output_name(op)
            xroot, xshape = p6e_assembly_views[xname]
            layouts[xname] = TensorLayout(tuple(xshape), _contiguous(tuple(xshape)), xroot, 0)
            dtypes[xname] = "bfloat16"
            views.append({"name": xname, "target": "p6e_scan_assembly", "source": xroot})
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
        if base in {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand", "squeeze", "unsqueeze", "as_strided"}:
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
            if matrix_policy != "frozen" and base != "addmm" and dtypes[a] == dtypes[b] == dtypes[name]:
                from .matrix_codegen_p6e import select
                from .matrix_rules_p6e import HEADER, emit_group
                spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch)
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                    from .matrix_rules_p6e import spec_headers
                    for header in spec_headers(spec):
                        if header not in extra_headers:
                            extra_headers.append(header)
                        if f'#include "{header}"' not in helper_parts:
                            helper_parts.append(f'#include "{header}"')
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
                from .matrix_codegen_p6e import fp32_async
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
                from .matrix_codegen_p6e import split_k as split_codegen
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
            if pops5e_stream and not integer_sum and not retain_all and target_arch == "sm_90" and exact_reductions:
                if p5e_reduction(op, name, source, src, dims, kept, extent, results, refs, rounded_result, scale):
                    continue
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
                    if pops3c and pops3c_pow1 and isinstance(args[1], (int, float)) and float(args[1]) == 1.0:
                        # P-OPS3-C: libdevice powf(x, 1.0f) == x for every float bit pattern (exhaustive H100 check,
                        # pops3c_jobs.pow_identity_check); NaN payloads are not values the program can observe.
                        return f"({rd(0)})"
                    return f"powf({rd(0)}, {rd(1)})"
                if base == "neg":
                    return f"(-{rd(0)})"
                if base == "reciprocal":
                    if outline_ops:
                        return f"latch_fdiv(1.0f, {rd(0)})"
                    return f"(1.0f / {rd(0)})"
                if base in {"tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs"}:
                    fn = {"abs": "fabsf"}.get(base, base + "f")
                    return f"{fn}({rd(0)})"
                if p6e_fast_act and dtypes[name] == "bfloat16" and base in {"sigmoid", "silu", "silu_backward"}:
                    from .p6e_rules import FAST_ACT
                    if FAST_ACT not in helper_parts:
                        helper_parts.append(FAST_ACT)
                    if base == "silu_backward":
                        return f"latch_fsilu_bwd({rd(0)}, {rd(1)})"
                    return f"latch_f{'sig' if base == 'sigmoid' else 'silu'}({rd(0)})"
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
        ew_args[name] = (base, args, kwargs)

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
    p5e_only = set()
    if pops5e_stream and pops5e_inline:
        # P-OPS5-E: a value consumed only by staged dense reductions (directly or through other elementwise values)
        # is recomputed in the engine's leaf from staged operands; storing it adds a full write+read pass.
        readers_ = {}
        for node in nodes.values():
            refs_ = node.loads if node.kind == "elementwise" else [token_loads[k][1] for k in node.loads]
            for ref in refs_:
                readers_.setdefault(root_of(ref), set()).add(node.name)
        for name in reversed([n for n in nodes if nodes[n].kind == "elementwise"]):
            rs = readers_.get(name, ())
            if rs and all((nodes[r].kind != "elementwise" and str(nodes[r].extra.get("engine", "")).startswith("pops5e_")
                           and nodes[r].kind == "reduction") or (nodes[r].kind == "elementwise" and r in p5e_only)
                          for r in rs):
                p5e_only.add(name)
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        cost = node.own_cost + sum(total_cost[root_of(ref)] for ref in node.loads if root_of(ref) in inline)
        total_cost[node.name] = cost
        if (retain_all or node.name in forced or node.dtype not in ("float32", "bfloat16")
                or any(dtypes[root_of(ref)] not in ("float32", "bfloat16") for ref in node.loads)):
            continue
        n_uses = uses.get(node.name, 0)
        if n_uses and cost <= INLINE_TOTAL_LIMIT and cost * n_uses <= INLINE_WORK_LIMIT:
            inline.add(node.name)
        elif node.name in column_only and cost <= 2 * INLINE_TOTAL_LIMIT:
            inline.add(node.name)
        elif node.name in p5e_only and cost <= pops5e_inline * INLINE_TOTAL_LIMIT:
            inline.add(node.name)

    pops5e_stored = []
    if pops5e_store and not retain_all:
        # P-OPS5-E memory-aware materialization of reduction operand chains: the operation-count limits above
        # inline long chains of cheap operations (e.g. a residual sum re-adding every earlier block's product)
        # even when each consumer then re-reads every tensor of the chain. Where a large value feeds a
        # reduction (directly or through inlined values) and its effective recomputation count times the bytes
        # its inlined expression reads exceeds computing it once, storing it and reading it back, it is stored.
        def nbytes(ref):
            root = root_of(ref)      # a broadcast operand costs its own (small) storage, not the expanded view
            return _numel(layouts[root].shape) * _BYTES[dtypes[root]]
        readers = {}
        for node in nodes.values():
            refs_ = (node.loads if node.kind == "elementwise"
                     else [token_loads[k][1] for k in node.loads if k not in p4a_state["sibling_tokens"]])
            for ref in refs_:
                readers.setdefault(root_of(ref), []).append(node.name)
        order = [n for n in nodes if nodes[n].kind == "elementwise"]
        feeds_reduction = {}
        for name in reversed(order):
            feeds_reduction[name] = any(
                nodes[r].kind in ("reduction", "pops4a") or (r in inline and feeds_reduction.get(r, False))
                for r in readers.get(name, ()))
        eff = {}
        for name in reversed(order):
            eff[name] = sum((eff.get(r, 1) if r in inline and nodes[r].kind == "elementwise" else 1)
                            for r in readers.get(name, ()))
        chain = {}
        for name in order:
            node = nodes[name]
            roots_ = {}
            for ref in node.loads:
                root = root_of(ref)
                if root in inline and root in chain:
                    roots_.update(chain[root])
                else:
                    roots_[root] = nbytes(ref)
            chain[name] = roots_
            if name not in inline or not feeds_reduction.get(name) or _numel(node.shape) < pops5e_store_min:
                continue
            read = sum(roots_.values())
            own = _numel(node.shape) * _BYTES[node.dtype]
            if eff.get(name, 0) >= 2 and eff[name] * read > read + own * (1 + eff[name]):
                inline.discard(name)
                chain[name] = {name: own}
                pops5e_stored.append({"name": name, "effective_uses": eff[name], "chain_bytes": read,
                                      "chain_roots": len(roots_)})

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
    from .pops2b_rules_p6e import IDLE_HEAVY_KINDS
    # P-OPS2-B rules belong to programs with exact-order BF16 products; others keep the base schedule.
    pops2b = pops2b and any(node.kind == "gemm" and (node.extra.get("matrix_spec") or {}).get("dtype") == "bfloat16"
                            for node in nodes.values())
    absorbed = {}                         # BF16 cast -> (product, operand) materialized in operand staging
    if pops2b and pops2b_casts and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p6e import select_absorbed_casts
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
        return (node.kind == "elementwise" and idx == "i" and root in nodes and nodes[root].kind == "elementwise"
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
        return bool((h.extra or {}).get("pops4a"))

    def p4a_companion(h):
        # P-OPS4-A items join phases of products and ordinary reductions (sequential per block after them).
        return p4a_item(h) or h.kind == "gemm" or (h.kind == "reduction" and not (h.extra or {}).get("exclusive_phase"))

    def gemm_or_one_other(phase, node):
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        if (node.extra or {}).get("p5e_share"):
            # P-OPS5-E: staged full reductions run back to back in one phase (each block streams its tiles of
            # one item, then the next); they join only phases of other such items and finalize-style items.
            return all((h.extra or {}).get("p5e_share") or p4a_item(h) for h in heavy)
        if any((h.extra or {}).get("p5e_share") for h in heavy) and not p4a_item(node):
            return False
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
            return False  # items with internal grid barriers run alone in their phase
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
        from . import norm_rules
        n = _numel(node.shape)
        if node.dtype not in ("float32", "bfloat16") or node.name in inplace_source or node.name in absorbed:
            return None
        loads, roots = [], set()
        for root, layout, shape, idx in loads_of[node.name]:
            row = nodes.get(root)
            statistic = None
            if (row is not None and (row.extra or {}).get("engine") in ("pops3c_row", "pops5e_row", "pops5e_fused") and root in phase_of
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
            loads.append((statistic, register, phase_of[root] + 1 if root in phase_of else 0))
            if shape is not None and layout.numel == n:
                roots.add(root)
        owner = None
        # The latest row engine whose warp can evaluate `node`: every other load is ready before its phase.
        for candidate in sorted({x for s_, r_, _ in loads for x in (s_, r_) if x}, key=lambda r: -phase_of[r]):
            if all(s_ == candidate or r_ == candidate or ready <= phase_of[candidate] for s_, r_, ready in loads):
                owner = candidate
                break
        row_member = owner is not None and n == nodes[owner].extra["row_rows"]
        if owner is not None and not row_member and not pops3c_attach_shared and shares_column(owner):
            return None
        if owner is None:
            # Same-row sibling: ready exactly in a row engine's phase and sharing a full-size operand with its leaf.
            earliest_other = max([ready for _, _, ready in loads] + [0])
            for name in phases[earliest_other]["heavy"] if earliest_other < len(phases) else ():
                row = nodes[name]
                if ((row.extra or {}).get("engine") in ("pops3c_row", "pops5e_row", "pops5e_fused") and row.extra["row_rows"] * row.extra["row_extent"] == n
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
            phase = earliest
            while True:
                while phase >= len(phases):
                    new_phase()
                if gemm_or_one_other(phase, node):
                    break
                phase += 1
            phases[phase]["heavy"].append(node.name)
            phase_of[node.name] = phase
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
        from .pops2b_rules_p6e import defer_products, idle_heavy_model
        deferred, preload = defer_products(items=items, loads_of=loads_of, phase_of=phase_of, phases=phases,
                                           nodes=nodes, idle_model=idle_heavy_model(), numel_of=_numel,
                                           inplace_source=inplace_source, dtypes=dtypes)
        commit = len(phases)
    if place_by_slack and not retain_all:
        idle_heavy = None
        if pops2b and pops2b_idle and target_arch == "sm_90":
            from .pops2b_rules_p6e import idle_heavy_model
            idle_heavy = idle_heavy_model(pops2b_idle_fill)
        _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                        forced, _numel, idle_heavy=idle_heavy, deferred=deferred, preload=preload,
                        fixed=set(attached_to))
        commit = len(phases)
    epilogue_of, epilogues, permuted_members = {}, {}, {}
    if pops2b and pops2b_epilogue and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p6e import attach_epilogues
        epilogue_of, epilogues, permuted_members = attach_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            dtypes=dtypes, in_place=in_place, inplace_source=inplace_source, register_load=register_load,
            numel_of=_numel, permuted_copies=pops2b_permuted, cost_of=lambda n: total_cost.get(n, 0),
            cost_limit=pops2b_epi_cost, stream_gate=pops2b_epi_stream_gate,
            background=set(deferred) if pops2b_defer_epilogue else set(), fixed=set(attached_to))
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
                inner = producer.render(index, lambda arg, i2, s=None: operand(arg, producer.shape if s is None else s, i2, memory_loader))
                return (f"__bfloat162float({ctype_round(inner, 'bfloat16')})" if producer.dtype == "bfloat16"
                        else inner)

        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            if kind == "index":
                return expand(ref, shape, idx, loader)
            return expand_storage(ref, idx, loader)
        text = _TOKEN.sub(substitute, node.code)
        if (getattr(node, "extra", None) or {}).get("p5e_spec"):
            text = p5e_emit(node, text)
        return text

    def p5e_emit(node, text):
        from . import pops5e_rules as P5
        spec = node.extra["p5e_spec"]
        row_leaf = text[text.index("/*P5E_ROW*/") + len("/*P5E_ROW*/"):text.index("/*P5E_COL*/")] or None
        col_leaf = text[text.index("/*P5E_COL*/") + len("/*P5E_COL*/"):text.index("/*P5E_END*/")] or None
        ident_of = {_ident(n): n for n in layouts if n in declared}
        canon = P5._normalize(P5.position(spec["extent"]))
        staged, row_bytes = [], 0
        for ident, indices in P5.reads([x for x in (row_leaf, col_leaf) if x]).items():
            name_ = ident_of.get(ident)
            if name_ is None or any(i != canon for i in indices):
                continue
            lay = layouts[name_]
            b = _BYTES[dtypes[name_]]
            if (not lay.contiguous or lay.numel != spec["rows"] * spec["extent"] or (spec["extent"] * b) % 16
                    or dtypes[name_] not in ("float32", "bfloat16")):
                continue
            staged.append((ident, _CTYPES[dtypes[name_]], b))
            row_bytes += spec["extent"] * b
        mode_ = "flight3" if row_leaf and col_leaf else "flight"
        prefer_ = pops5e_plan
        if isinstance(pops5e_plan, dict) and ("fused" in pops5e_plan or "row" in pops5e_plan):
            # P6-E: separate forced plans for fused row+column passes and row-only passes (None keeps the default)
            prefer_ = pops5e_plan.get("fused" if mode_ == "flight3" else "row")
        plan = P5.plan(spec["rows"], spec["extent"], row_bytes,
                       prefer=prefer_ or mode_,
                       budget=pops5e_budget or P5.STAGE_BUDGET)
        if plan is None:
            staged, plan = [], {"tile": 8, "stages": 0}
        code, shared = P5.emit(rows=spec["rows"], extent=spec["extent"], row_out=spec["row_out"], row_leaf=row_leaf,
                               col_part=spec["col_part"], col_leaf=col_leaf,
                               rounded=spec["row_rounded"] or (lambda e: e), scale=spec["row_scale"],
                               staged=staged, idents_bytes=None, tile=plan["tile"], stages=plan["stages"],
                               ctypes={_ident(n): _CTYPES[dtypes[n]] for n in layouts if n in declared})
        if shared:
            node.extra["shared_bytes"] = max(int(node.extra.get("shared_bytes", 0)), shared)
        node.extra["p5e_staged"] = [i for i, _, _ in staged]
        node.extra["p5e_names"] = {i: f"latch_s{k}" for k, (i, _, _) in enumerate(staged)} if plan["stages"] else {}
        node.extra["p5e_plan"] = plan
        if not p5e["helpers"]:
            helper_parts.append(P5.HELPERS)
            p5e["helpers"] = True
        return code

    vector_helpers_needed = []
    p6e_assemblies = []

    def loop_code(numel, members, exclude):
        if numel >= (1 << 31):
            raise LoweringError("fused elementwise loops require fewer than 2^31 elements")
        loop_regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        vector = floating and numel % 4 == 0 and numel >= VECTOR_MIN_ELEMENTS
        vector_refs = {}

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            # The operand's own element count must equal the loop's: a broadcast
            # operand (e.g. a [512] weight in a [256,512] loop) is never an identity read.
            identity = (idx == "i" and shape is not None and _numel(shape) == numel
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
        if p6e_assembly and len(members) == 1 and stored_members == list(members):
            from .p6e_rules import assembly_pattern, assembly_code, ASSEMBLY_HELPERS
            pattern = assembly_pattern(members[0], nodes=nodes, ew_args=ew_args, inline=inline, layouts=layouts,
                                       dtypes=dtypes, declared=declared, ref=lambda a: _ref(a, names))
            if pattern is not None:
                p6e_assemblies.append({"name": members[0], **{k: v for k, v in pattern.items() if k != "segments"},
                                       "segments": [[a, b, src] for a, b, src in pattern["segments"]]})
                if ASSEMBLY_HELPERS not in helper_parts:
                    helper_parts.append(ASSEMBLY_HELPERS)
                return assembly_code(pattern, _ident(members[0]), exclude=exclude, ident=_ident)
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
            if outline_ops:
                from .norm_rules import cse_calls
                prelude, expression = cse_calls(expression, f"c_{name}")
                compute.extend(prelude)
            compute.append(f"{loop_regs[name]}[v] = {ctype_round(expression, node.dtype)};")
        if p6e_inline_div:
            # P6-E: the same IEEE division as latch_fdiv (a / b), force-inlined in fused elementwise loops: no call
            # (and no caller-saved spills around it) in loops with many live operands; bitwise identical.
            from .p6e_rules import INLINE_DIV
            if INLINE_DIV not in helper_parts:
                helper_parts.append(INLINE_DIV)
            compute = [re.sub(r"(?<![\w:.])latch_fdiv\(", "latch_fdiv_i(", c) for c in compute]
        invariant = ""
        # p6e_licm: True hoists in every loop; an int n hoists only in loops of at most n members (one optimizer
        # group's few invariants stay live; many groups' invariants together spill).
        if p6e_licm and (p6e_licm is True or len(members) <= int(p6e_licm)):
            from .p6e_rules import licm
            hoisted, compute = licm(compute, "latch_li")
            invariant = "".join(f"  {line}\n" for line in hoisted)
        lanes = 4 if vector else 1
        count = numel // lanes
        u = 1
        per_thread = -(-count // RESIDENT_THREADS)
        cap = max(1, (int(p6e_live_cap) or LIVE_VALUE_CAP) // max(1, lanes * len(members)))  # P6-E p6e_live_cap
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
        else:
            head = "  const unsigned gs = gridDim.x * blockDim.x;\n"
            start_block = "blockIdx.x"
        if vector:
            if not vector_helpers_needed:
                vector_helpers_needed.append(True)
            loads = "".join(f"        {_CTYPES[dtypes[ref]]} {reg}[4]; latch_ld4({_ident(ref)} + 4u * q, {reg});\n"
                            for ref, reg in vector_refs.items())
            p6e_preload = ""
            if p6e_ld_first and vector_refs:
                # P6-E: every unit's operand loads are issued before any unit computes (out-of-line helper calls
                # in the compute otherwise keep later units' loads behind them); same loads, same values.
                p6e_preload = ("".join(f"    {_CTYPES[dtypes[ref]]} {reg}_a[{u}][4];\n" for ref, reg in vector_refs.items())
                           + f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned q = base + (unsigned)u * gs;\n"
                           + f"      if (q < {_u(count)}) {{\n"
                           + "".join(f"        latch_ld4({_ident(ref)} + 4u * q, {reg}_a[u]);\n" for ref, reg in vector_refs.items())
                           + "      }\n    }\n")
                loads = "".join(f"        {_CTYPES[dtypes[ref]]} (&{reg})[4] = {reg}_a[u];\n" for ref, reg in vector_refs.items())
            body = (f"      const unsigned q = base + (unsigned)u * gs;\n      if (q < {_u(count)}) {{\n{loads}"
                    f"#pragma unroll\n        for (int j = 0; j < 4; ++j) {{\n"
                    f"          const unsigned i = 4u * q + (unsigned)j; const int v = u * 4 + j;\n          "
                    + "\n          ".join(compute) + "\n        }\n      }\n")
            stores = "".join(f"        latch_st4({_ident(n)} + 4u * q, {loop_regs[n]}[u*4], {loop_regs[n]}[u*4+1], "
                             f"{loop_regs[n]}[u*4+2], {loop_regs[n]}[u*4+3]);\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned q = base + (unsigned)u * gs;\n"
                           f"      if (q < {_u(count)}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        else:
            p6e_preload = ""
            body = (f"      const unsigned i = base + (unsigned)u * gs; const int v = u;\n"
                    f"      if (i < {_u(count)}) {{\n        " + "\n        ".join(compute) + "\n      }\n")
            stores = "".join(f"        {_ident(n)}[i] = {loop_regs[n]}[u];\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned i = base + (unsigned)u * gs;\n"
                           f"      if (i < {_u(count)}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        return (f"{{ // {numel} elements{' (x4 vector)' if vector else ''}: {', '.join(members)}\n" + invariant + head +
                f"  for (unsigned base = {start_block} * blockDim.x + threadIdx.x; base < {_u(count)}; base += {u}u * gs) {{\n"
                f"    {declarations_local}\n" + (p6e_preload if vector else "") + f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n" + body + "    }\n"
                + store_block + "  }\n}")

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
                expression = member.render("g", lambda arg, i2, s=None, member=member: operand(arg, member.shape if s is None else s, i2, row_loader))
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
            expression = member.render("i", lambda arg, i2, s=None, member=member: operand(arg, member.shape if s is None else s, i2, loader))
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

    ident_dtype = {_ident(name): dtypes[name] for name in layouts}
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
    def p6e_place_split(code, matrix_group, phase, k):
        """P6-E split-K products: per-member arrival counters and FP32 partial tiles in this phase's split scratch
        (the P-OPS2-A region: counters zeroed at entry and restored to zero by every last arriver)."""
        nonlocal p2_counter_words, p2_partial_words, p2_first_split_phase
        for member in matrix_group.get('members', []):
            plan_ = member.get('matrix_plan') or {}
            split = int(plan_.get('split', 1))
            if split <= 1:
                continue
            if '/*P2_CNT*/' not in code or '/*P2_PART*/' not in code:
                raise LoweringError("internal: split product without scratch placeholders")
            cnt_cursor = phase.setdefault('_p2_cnt', 0)
            part_cursor = phase.setdefault('_p2_part', 0)
            code = code.replace('/*P2_CNT*/', f'(arena + {p2_base + cnt_cursor}LL)', 1)
            code = code.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {part_cursor}LL)', 1)
            phase['_p2_cnt'] = cnt_cursor + arena_plan.a64(plan_['out_tiles'])
            phase['_p2_part'] = part_cursor + arena_plan.a64(plan_['out_tiles'] * split * plan_['bm'] * plan_['bn'])
            p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
            p2_partial_words = max(p2_partial_words, phase['_p2_part'])
            if p2_first_split_phase is None:
                p2_first_split_phase = k
        return code

    def p6e_epilogue_code(product, members):
        """P6-E register epilogue of a p6e TMA product (after P-OPS5-C p5c epilogue_code): 8 contiguous outputs (flat
        i0..i0+7, ldc == N) per call. `latch_ld(e, i0)` prefetches every identity-indexed operand of chunk slot e as
        raw 16-byte vectors before any chunk is computed; `latch_epi(e, i0, gv)` evaluates each member's identical
        per-operation expression on gv[j], the product's rounded BF16 value widened to FP32 (exactly a load of the
        stored product), and stores stored members 8-wide."""
        spec = nodes[product].extra["matrix_spec"]
        size = spec["m"] * spec["n"]
        member_set = set(members)
        if member_set & set(permuted_members):
            raise LoweringError("internal: permuted epilogue members on a p6e TMA product")
        vector_refs = {}

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            identity = (idx == "i" and shape is not None and _numel(shape) == size and layouts[ref].numel == size
                        and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin == product:
                return "gv[j]"
            if identity and origin in member_set:
                expression = f"r_{origin}[j]"
                return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
            if identity and ref in declared and dtypes[ref] in ("float32", "bfloat16"):
                reg = vector_refs.setdefault(ref, f"p6in{len(vector_refs)}")
                if dtypes[ref] == "bfloat16":
                    return f"__bfloat162float(p6e::upkb({reg}[e], j))"
                return f"p6e::upk({reg}[e], j)"
            return memory_loader(ref, shape, idx)
        compute = []
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, s=None, node=node: operand(arg, node.shape if s is None else s, i2, loader))
            if outline_ops:
                from .norm_rules import cse_calls
                prelude_, expression = cse_calls(expression, f"c_{name}")
                compute.extend(prelude_)
            compute.append(f"r_{name}[j] = {ctype_round(expression, node.dtype)};")
        prelude = " ".join(f"uint4 {reg}[{16 if p6e_early_ld >= 2 else 8}]{'[2]' if dtypes[ref] == 'float32' else ''};"
                           for ref, reg in vector_refs.items())
        loads = " ".join((f"{reg}[e] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0);" if dtypes[ref] == "bfloat16"
                          else f"{reg}[e][0] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0); "
                               f"{reg}[e][1] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0 + 4);")
                         for ref, reg in vector_refs.items())
        declarations = " ".join(f"{_CTYPES[nodes[n].dtype]} r_{n}[8];" for n in members)
        stores = " ".join(f"p6e::st8({_ident(n)} + i0, r_{n});" for n in members if n in stored)
        run = ("[&](int e, unsigned i0, const float (&gv)[8]) {\n"
               f"      {declarations}\n"
               "#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n"
               "        const unsigned i = i0 + (unsigned)j;\n        " + "\n        ".join(compute) +
               f"\n      }}\n      {stores}\n    }}")
        ld = f"[&](int e, unsigned i0) {{ {loads} }}"
        return {"prelude": prelude, "run": run, "ld": ld, "store": product not in elided_products,
                "members": list(members)}

    for k, phase in enumerate(phases):
        parts, gemm_offset, rotate = [], 0, 0
        matrix_group = None
        # P-OPS4-A items run after the phase's other heavy items; the existing emission sees only the others.
        p4a_here = [n for n in phase["heavy"] if p4a_item(nodes[n])]
        full_heavy = phase["heavy"]
        phase = dict(phase, heavy=[n for n in full_heavy if n not in p4a_here])
        group_specs = [nodes[name].extra.get('matrix_spec') for name in phase['heavy']]
        phase_epilogues = phase.get('epilogue', {})
        if (group_specs and all(group_specs) and phase_epilogues and not pops2b_plans
                and all(nodes[n].extra['matrix_spec'].get('tma') for n in phase_epilogues)
                and not any(n in absorbed_by for n in phase['heavy'])):
            from .matrix_rules_p6e import emit_group
            specs_ = []
            for name, spec in zip(phase['heavy'], group_specs):
                if name in phase_epilogues:
                    epi = p6e_epilogue_code(name, phase_epilogues[name])
                    spec = dict(spec, epi=epi, out=spec['out'] if epi['store'] else 'nullptr')
                specs_.append(spec)
            code, matrix_group = emit_group(specs_, ident=_ident)
            code = p6e_place_split(code, matrix_group, phase, k)
            for name, member in zip(phase['heavy'], matrix_group.get('members', [])):
                if name in phase_epilogues:
                    member.update(engine=member['engine'] + '_epi', epilogue=list(phase_epilogues[name]))
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        elif group_specs and all(group_specs) and (phase_epilogues or pops2b_plans
                                                 or any(n in absorbed_by for n in phase['heavy'])):
            from .pops2b_rules_p6e import HEADER as EPI_HEADER, emit_group as emit_epilogue_group
            if EPI_HEADER not in extra_headers:
                extra_headers.append(EPI_HEADER)
                helper_parts.append(f'#include "{EPI_HEADER}"')
            extras = []
            for name in phase['heavy']:
                extra = {}
                if name in phase_epilogues:
                    extra['epi'] = epilogue_code(name, phase_epilogues[name])
                for which, cast in absorbed_by.get(name, {}).items():
                    extra['p' + which] = prologue_code(cast, name, which)
                extras.append(extra or None)
            code, matrix_group = emit_epilogue_group(group_specs, extras, ident=_ident, plans=pops2b_plans)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        elif group_specs and all(group_specs):
            from .matrix_rules_p6e import emit_group
            code, matrix_group = emit_group(group_specs, ident=_ident)
            code = p6e_place_split(code, matrix_group, phase, k)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        else:
            engines = {nodes[n].extra.get("engine"): n for n in phase["heavy"]}
            for position, name in enumerate(phase["heavy"]):
                node = nodes[name]
                code = heavy_code(node)
                engine = node.extra.get("engine")
                if engine in ("pops5e_row", "pops5e_fused"):
                    members = phase.get("rows", {}).get(name, [])
                    member_code = row_members_code(name, members) if members else ""
                    staged_names = node.extra.get("p5e_names") or {}
                    for ident, sname in staged_names.items():
                        # attached members read staged operands from the shared tile of the current row
                        # (element c of row latch_lr); writes keep their global targets.
                        member_code = re.sub(r"\b" + re.escape(ident) + r"\[i\](?!\s*=[^=])",
                                             f"{sname}[latch_lr * {node.extra['row_extent']}u + c]", member_code)
                    code = code.replace("/*ROW_MEMBERS*/", member_code)
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
                elif engine == "pops3c_col" and "pops3c_row" in engines and not pops3c_overlap:
                    groups = node.extra["column_groups"]
                    code = code.replace(f"/*COL_SPAN*/min({groups}u, gridDim.x)",
                                        f"min({groups}u, gridDim.x / {pops3c_col_div}u)")
                if name in deferred:
                    info = deferred[name]
                    spec = node.extra["matrix_spec"]
                    if name in phase.get("epilogue", {}):
                        from .pops2b_rules_p6e import HEADER as EPI_HEADER, _code as epi_matrix_code, tile_plan
                        if EPI_HEADER not in extra_headers:
                            extra_headers.append(EPI_HEADER)
                            helper_parts.append(f'#include "{EPI_HEADER}"')
                        extra = {"epi": epilogue_code(name, phase["epilogue"][name])}
                        code = epi_matrix_code(spec, tile_plan(spec, info["plan_grid"], extra), _ident,
                                               *info["slice"], 0, extra)
                    else:
                        from .matrix_rules_p6e import _code as matrix_code, _plan as matrix_plan
                        code = matrix_code(spec, matrix_plan(spec, info["plan_grid"]), _ident, *info["slice"], 0)
                if node.kind == "gemm" and '/*P2_CNT*/' in code and node.extra and node.extra.get('members'):
                    code = p6e_place_split(code, node.extra, phase, k)
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
        for numel, members in phase["loops"].items():
            if p6e_loop_split and len(members) > p6e_loop_split:
                # P6-E: independent register-connected components as separate loops (same per-element code; fewer
                # live operands per thread, so deeper unrolling and no spills in very wide fused loops).
                parent_ = {m: m for m in members}

                def find_(x):
                    while parent_[x] != x:
                        parent_[x] = parent_[parent_[x]]
                        x = parent_[x]
                    return x
                mset = set(members)
                for m in members:
                    for root, layout, shape, idx in loads_of[m]:
                        if root in mset and shape is not None and register_load(nodes[m], root, layout, shape, idx):
                            parent_[find_(m)] = find_(root)
                # a component may not read (old) state that another component commits in place in this loop
                crossing = any(root in in_place and in_place[root] in mset and find_(in_place[root]) != find_(m)
                               for m in members for root, _, _, _ in loads_of[m])
                comps = {}
                for m in members:
                    comps.setdefault(find_(m), []).append(m)
                if not crossing and len(comps) > 1:
                    for comp in comps.values():
                        parts.append(loop_code(numel, comp, exclude))
                    continue
            parts.append(loop_code(numel, members, exclude))
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
        phase_source.append(f"// Phase {k}: {', '.join(all_names)}\n{{\n" + scoped("\n".join(parts))
                            + f"\n}}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{k+1}] = stamp();")
    if p6e_scalar_chain:
        # P6-E: consecutive phases whose only work is one-element elementwise loops run entirely on global thread 0
        # (loop start blockIdx.x * blockDim.x + threadIdx.x < 1); the barrier between two such phases orders thread 0
        # after itself only, so it is dropped (profile stamps kept). Same statements, same order: bitwise.
        def _scalar_only(rec):
            return (rec["kind"] == "elementwise" and rec["items"]
                    and all(it.get("kind") == "elementwise" and it.get("elements") == 1 for it in rec["items"]))
        merged = 0
        for k in range(1, len(phase_records)):
            if _scalar_only(phase_records[k - 1]) and _scalar_only(phase_records[k]):
                head, sep, rest = phase_source[k - 1].rpartition("\n}\ngrid.sync();\n")
                assert sep, "phase barrier"
                phase_source[k - 1] = head + "\n}\n" + rest
                merged += 1
        p6e_scalar_merged = merged
    prefetches = []
    if pops2b and pops2b_prefetch and not retain_all and target_arch == "sm_90":
        from .pops2b_rules_p6e import prefetch_plan
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
        elif used:
            helper_parts.append(OUTLINED)
    if p6e_early_ld:
        helper_parts.insert(0, f"#define P6E_EARLY_LD {int(p6e_early_ld)}   // P6-E: early epilogue operand loads (kernels/p6e_tma.cuh)")
    if p6e_defer:
        helper_parts.insert(0, f"#define P6E_DEFER {int(p6e_defer)}   // P6-E: deferred epilogue chunks per k step (kernels/p6e_tma.cuh)")
    if p6e_quad:
        helper_parts.insert(0, "#define P6E_QUAD 1   // P6-E: branch-free epilogue quad transpose, warp-0 TMA issue (kernels/p6e_tma.cuh)")
    if p6e_last_issue:
        helper_parts.insert(0, "#define P6E_LAST_ISSUE 1   // P6-E: stage refill by the releasing warpgroup (kernels/p6e_tma.cuh)")
    helpers = "\n".join(helper_parts)
    shared_bytes = max([17408] + [p.get("shared_bytes", 0) for p in phase_records])
    from .resources import select_program_resources
    resource_policy = select_program_resources(shared_bytes, phases=phase_records, arch=target_arch,
                                               pops2_two_cta=matrix_residency == 2)
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    from .matrix_rules_p6e import collect_tensor_maps
    source, tensor_maps = collect_tensor_maps(source, len(inputs))
    index_checks = None
    if p6e_trap and any(name in ("pops4b_wg.cuh", "pops4b_tma.cuh") for name in extra_headers):
        from .p6e_rules import TRAP_ASSERT
        entry = 'extern "C" __global__'
        if source.count(entry) != 1:
            raise LoweringError("expected exactly one kernel entry")
        source = source.replace(entry, TRAP_ASSERT + entry, 1)
        index_checks = "trap (assert overridden for wgmma programs)"
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
        **({"selective_scans": scan_decisions} if scan_decisions else {}),
        **({"pops5e_store": pops5e_stored} if pops5e_stored else {}),
        **({"pops5e_stream": [dict(d, **{k: nodes[d["root"]].extra.get(k) for k in ("p5e_staged", "p5e_plan")})
                               for d in p5e["decisions"]]} if p5e["decisions"] else {}),
        **({"pops4a": {"reductions": p4a_state["decisions"],
                       "phases_with_items": sum(1 for ph in phases if any(p4a_item(nodes[n]) for n in ph["heavy"]))}}
           if p4a_state["decisions"] else {}),
        "compiler_policy": "integrated_v1", "target_arch": target_arch,
        "phase_count": commit + 1, "barrier_count": commit + (1 if commit_code else 0),
        "max_gemm_tiles": max([p.get("tiles", 0) for p in phase_records] or [1]),
        "shared_bytes": shared_bytes,
        **({"tensor_maps": tensor_maps} if tensor_maps else {}),
        **({"index_checks": index_checks} if index_checks else {}),
        **({"p6e_assemblies": p6e_assemblies} if p6e_assemblies else {}),
        **({"p6e_scalar_barriers_removed": p6e_scalar_merged} if p6e_scalar_chain else {}),
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
