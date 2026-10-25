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
import json
import math
import re

from .vendor import plan as arena_plan

UNROLL_MAX = 8
RESIDENT_THREADS = 128 * 256          # default H100 launch cap x CTA width
TRANSCENDENTAL = {"tanh", "exp", "expm1", "log1p", "reciprocal", "log", "sqrt", "rsqrt", "sigmoid", "silu", "silu_backward",
                  "pow", "div", "sigmoid_backward", "tanh_backward"}
INLINE_TOTAL_LIMIT = 12                # recomputed operations per inlined value use
INLINE_WORK_LIMIT = 24                 # summed over all uses
LIVE_VALUE_CAP = 64                    # unroll x loop values held between loads and stores
# P9-D loop partition: in a phase made only of elementwise loops, the independent dependency components of every
# loop group (e.g. one optimizer chain per parameter tensor) run concurrently on disjoint CTA ranges sized by their
# element counts (resolved from gridDim.x at run time), instead of every CTA walking every chain in one register-
# heavy loop. Same expressions, same element ownership semantics (each element once), so results are unchanged.
# Measured (P9-D gpu-05): slower on the SD AdamW phase (218 -> 299 us; one CTA per SM, long dependent chains) and
# its first form raced on in-place commits read by two components (fixed below); kept as an explicit option.
LOOP_PARTITION = False
LOOP_PARTITION_MIN_ELEMENTS = 1 << 20
# P9-D loop split: each dependency component of a large multi-tensor loop group is its own grid-wide loop
# (fewer live values per thread than one joint loop over every member); option, measured before any default.
LOOP_SPLIT = False
# P9-D scalar hoist: a one-element elementwise value of real cost that is broadcast into large loops is stored once
# (computed in its own earlier phase slot) instead of being re-evaluated per element of every consuming loop
# (e.g. AdamW bias corrections -expm1(step log beta), their square roots and reciprocals). Same expression and
# rounding: the stored FP32 value is exactly what the inlined expression evaluated.
SCALAR_HOIST = False      # measured (gpu-06): 5.81 vs 5.72 ms without; kept as an option
# P9-D reduction stacking: reduction heavies (CTA trees, multi-CTA full sums) are self-contained grid-strided blocks;
# several may run back to back in one non-matrix phase (a CTA barrier between them). A phase holding only such
# reductions sinks, within each reduction's slack (before its first reader and any in-place commit of what it
# reads), into the next phase that already holds stackable reductions; emptied phases lose their grid barrier.
REDUCTION_STACK = True
# LATCH-STEP-04: zero-gradient rows' optimizer update inside the indexed phase (row-activity flags, active-row
# final loop). Correct (gated) but measured slower on H100 (embedding 32.8 -> 35.8 us): the streamed state update
# did not overlap the indexed phase's latency-bound work. Kept as an explicit option.
INDEXED_OPTIMIZER_SPLIT = False
# LATCH-STEP-04: forward embedding gathers as inlinable elementwise values. Measured (H100, build-21):
# expert 147.2 -> 137.9 us, embedding 32.9 -> 33.7 us (three consumers recompute the gather). Option, off.
INLINE_EMBEDDING_GATHER = False
# LATCH-STEP-04: pow(x, 1) emitted as x (MSE backward emits it per element; bitwise identical, see the rule).
POW_ONE_IDENTITY = True
# LATCH-STEP-04: same-phase elementwise loops whose extent equals a multi-CTA full reduction's are evaluated
# inside the reduction's chunk loop at i = r (each r in [0, extent) is visited once): one pass over the data,
# the reduction order and every elementwise value unchanged.
FUSE_FULL_REDUCTION_LOOPS = False   # measured: embedding 30.7 -> 32.1 us, expert neutral
# LATCH-STEP-04: the owner indexed engine reads gradient elements through storage tokens, so an elementwise
# producer (e.g. a gate x output-cotangent product) may be inlined instead of materialized and re-read. The
# engine's alternative read sites (row classes, alignment paths) count as one use: each element is read once.
# None: per engine (owner_v1 measured slower with tokens: embedding indexed phase 11.2 -> 31.2 us, inlined scalar
# reads in its serial lane loops; atomic_v1 reads one vector per thread). True/False force it for A/B builds.
INDEXED_PRODUCER_TOKENS = None
ATOMIC_TOKEN_ENGINES = ("atomic_v1",)
# LATCH-STEP-04 (rule from P-OPS5-C/P6-E): generated index checks abort through __trap() instead of assert(), whose
# __assertfail system call is a function call inside the kernel (ABI register saves around every unrolled check in
# the gather and indexed scan loops). Same abort-on-violation (visible launch failure), no message text.
TRAP_INDEX_CHECKS = True
# LATCH-STEP-04: unrolled leaf reads of one exact reduction over a non-broadcast source count as one inlining use.
REDUCTION_TOKEN_GROUPS = True
# LATCH-STEP-04: an indexed body that declares `prezero` (atomic_v1) has its output zeroed at the end of the previous
# phase (output storage live from there), removing its own zero stage and one internal grid barrier.
INDEXED_PREZERO = True
# LATCH-STEP-04: programs with an indexed engine declare a grid preference: the largest power of two within the
# measured cooperative occupancy (at most two CTAs per SM). Programs that compile to one CTA per SM keep 128.
INDEXED_GRID_POW2 = True
# LATCH-STEP-04: an indexed-engine body may share a phase with reductions and other indexed bodies (each heavy item
# runs on every CTA in sequence, separate split-counter slots), instead of always opening a new phase.
INDEXED_COSCHEDULE = True
# LATCH-STEP-04: programs with an indexed engine use doubled producer-inlining limits (24 operations per use, 48
# summed): gate/cotangent row products then feed indexed accumulation and reductions without materialization.
# Measured: expert 63.8 -> 52.4 us (with gather selection), embedding neutral. Other programs keep 12/24.
INDEXED_INLINE_LIMITS = (24, 48)
# LATCH-STEP-04: an elementwise value whose only same-phase dependency is an identity read of an exact reduction's
# results (other operands from earlier phases) is evaluated by the thread that writes each result, in the
# reduction's phase, instead of opening the next phase. Same expressions; its readers stay in later phases.
REDUCTION_EPILOGUES = False   # measured: expert 63.9 -> 67.2 us (one fewer phase, slower phases); option
REDUCTION_STACK_MAX = 8
SCALAR_HOIST_MIN_COST = 2
SCALAR_HOIST_MIN_CONSUMER = 4096
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
                    forced, numel, idle_heavy=None, deferred=None, preload=None, key_of=None):
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
            n_elements = key_of(chains[key][0]) if key_of else numel(nodes[chains[key][0]].shape)
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
                    permuted=None):
    """Independent re-check of the schedule (the scheduler is not trusted).

    Every read is of a graph input, a value from an earlier phase that has
    storage, or a same-loop register at the identical flat index. Each in-place
    target is read only before its committing phase, or at the committing
    loop's own index (all loads of an unrolled group precede its stores).
    """
    epilogue_of = epilogue_of or {}

    def loop_key(name):
        node = nodes[name]
        if node.kind != "elementwise":
            return None
        key = (phase_of[name], math.prod(node.shape))
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


# C-OPS4-B: fixed per-grid reassociation for long scalar reductions.
FULL_REDUCTION_MIN = 64 * 256
# LATCH-DENSE-EXEC: offer the P6-D exact row/column reduction engines (bitwise replays of the frozen tree).
REFERENCE_P6D_REDUCTIONS = True


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
                attention_pipeline=True, loop_partition=None, scalar_hoist=None, reduction_stack=None,
                loop_split=None, dynamic_batch=None):
    loop_split = LOOP_SPLIT if loop_split is None else bool(loop_split)
    loop_partition = LOOP_PARTITION if loop_partition is None else bool(loop_partition)
    reduction_stack = REDUCTION_STACK if reduction_stack is None else bool(reduction_stack)
    scalar_hoist = SCALAR_HOIST if scalar_hoist is None else bool(scalar_hoist)
    # A selects actual FP32 products; B schedules explicit BF16 products. D's
    # engine dictionary remains available for staged accumulation/layout paths.
    matrix_engine = ("auto" if pops2_policy else "popsb") if matrix_engine is None else matrix_engine
    matrix_bodies = (("simt", "popsb") if pops2_policy else None) if matrix_bodies is None else tuple(matrix_bodies)
    attention_override = attention_min_keys is not None
    from .lower import (GeneratedProgram, LoweringError, TensorLayout, VENDOR, _header_path, _BYTES, _CTYPES,
                        _contiguous, _dtype, _field, _flatten_refs, _float, _double, _ident, _numel,
                        _ref, _reshape_strides, _translation_unit, _lerp_scalar_expression)

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
    # LATCH-STEP19-SPEED: a bounded dynamic batch declaration (capacity storage, device
    # active extent). None for every static program, whose emission is unchanged.
    dynamic = None
    if dynamic_batch is not None:
        from .dynamic_flat import FlatActive
        dynamic = FlatActive(dynamic_batch, values, retain_all=retain_all)
        loop_partition = False
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
    views = []
    helper_parts = []
    if dynamic is not None:
        # Dynamic bodies and generated functors (kernels/dynamic_flat.cuh) precede every functor definition.
        for header in ("bgemm.cuh", "popsc_mm.cuh", "pops2b_mm.cuh", "dynamic_flat.cuh"):
            helper_parts.append(f'#include "{header}"')
    extra_headers = []
    if dynamic is not None:
        extra_headers.extend(["tgemm.cuh", "bgemm.cuh", "popsc_mm.cuh", "pops2b_mm.cuh", "dynamic_flat.cuh"])
    matrix_decisions = []
    pops2_layouts = {}
    token_loads = []      # (ref, shape, idx) per heavy-code placeholder

    def placeholder_read(arg, shape, idx="i", register_values=None):
        ref = _ref(arg, names)
        if ref is None:
            if (isinstance(arg, (int, float, bool)) or (dynamic is not None and isinstance(arg, dict) and 'dynamic_batch_scalar' in arg)):
                return _float(arg)
            raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
        token_loads.append(("index", ref, tuple(shape), idx))
        return f"@@LOAD{len(token_loads) - 1}@@"

    token_group = {}      # token index -> group: alternative code sites of one runtime read count as one use

    def storage_read(ref, position, group=None):
        token_loads.append(("storage", ref, None, position))
        if group is not None:
            token_group[len(token_loads) - 1] = group
        return f"@@LOAD{len(token_loads) - 1}@@"

    def storage_read4(ref, position, group=None):
        """float4 of elements position .. position+3 (16-byte aligned when materialized)."""
        token_loads.append(("storage4", ref, None, position))
        if group is not None:
            token_group[len(token_loads) - 1] = group
        return f"@@LOAD{len(token_loads) - 1}@@"

    def storage_addr(ref):
        """Address bits of a materialized value for alignment tests; 0 when the value is inlined."""
        token_loads.append(("storage_addr", ref, None, "0"))
        return f"@@LOAD{len(token_loads) - 1}@@"

    def add_heavy(kind, op, name, code, refs, **extra):
        loads = [int(m) for m in _TOKEN.findall(code)]
        direct = [ref for ref in dict.fromkeys(refs) if re.search(r"\b" + re.escape(_ident(ref)) + r"\b", code)]
        nodes[name] = _Node(name=name, kind=kind, code=code, extra=extra, loads=loads, direct=direct,
                            target=str(_field(op, "target", _field(op, "op"))), order=len(nodes))

    attention_decisions, index_decisions = [], []
    indexed_occurrence_cache = {}
    # LATCH-STEP-04: clip-norm terms sum(g * g) of embedding gradients fused into the indexed phase.
    from .indexed_compose import plan_norms as _plan_indexed_norms
    indexed_norms = (_plan_indexed_norms(operations, names, field=_field, output_name=output_name, ref=_ref,
                                         flatten_refs=_flatten_refs) if ordered_index_gradients else {})
    indexed_skip = set()
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
        if name in undefined or name in indexed_skip:
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
                if dynamic is not None and dynamic.active(name):
                    offset = dynamic.philox_offset(args[2]) or f"{int(args[2])}ULL"
                    code = f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long){dynamic.count(_numel(shape))}; i += (long long)gridDim.x * blockDim.x) {out}[i] = latch_philox_mask({_ident(state)}, i, {offset}, {_float(args[3])});"
                else:
                    code = f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {_numel(shape)}LL; i += (long long)gridDim.x * blockDim.x) {out}[i] = latch_philox_mask({_ident(state)}, i, {int(args[2])}ULL, {_float(args[3])});"
            else:
                if dtypes[name] != "int64" or shape != (4,):
                    raise LoweringError("Philox advance requires int64[4] output")
                advance = ((dynamic.philox_offset(args[1]) if dynamic is not None else None)
                           or f"{int(args[1])}ULL")
                code = f"if (blockIdx.x == 0 && threadIdx.x < 4) {out}[threadIdx.x] = latch_philox_advance_word({_ident(state)}, {advance}, threadIdx.x);"
            add_heavy("random_state" if "advance" in target else "random_mask", op, name, code, refs)
            continue
        from .transformer_lower import emit_operation, ordered_embedding_backward
        if ordered_index_gradients and base == "embedding_dense_backward":
            # LATCH-STEP-04: an explicitly selected reusable indexed engine (indexed_compose).
            from .indexed_compose import legacy_engine
            planned = indexed_norms.get(name)
            row_flags = (name + "__rowactive") if planned and INDEXED_OPTIMIZER_SPLIT else None
            if row_flags is not None:
                if row_flags in names or row_flags in layouts:
                    row_flags = None
                else:
                    layouts[row_flags] = TensorLayout((int(args[2]),), (1,), row_flags)
                    dtypes[row_flags] = "float32"
            grad_ref = _ref(args[0], names)
            read_grad = None
            from .indexed_reduction import requested_policy as _indexed_requested
            tokens = (INDEXED_PRODUCER_TOKENS if INDEXED_PRODUCER_TOKENS is not None
                      else _indexed_requested() in ATOMIC_TOKEN_ENGINES)
            if tokens and grad_ref is not None and dtypes.get(grad_ref) == "float32" and not retain_all:
                def read_grad(position, lanes, grad_ref=grad_ref, group=name):
                    if lanes == 0:
                        return storage_addr(grad_ref)
                    if lanes == 4:
                        return storage_read4(grad_ref, position, group)
                    return storage_read(grad_ref, position, group)
            engine = legacy_engine(args, kwargs, name, shape, layouts, dtypes, names, cache=indexed_occurrence_cache,
                                   norm=planned[1] if planned else None, flags=row_flags, read=read_grad)
            if engine is not None:
                if engine["helpers"] and engine["helpers"] not in helper_parts:
                    helper_parts.append(engine["helpers"])
                for value, value_shape, dtype in engine["auxiliary"]:
                    layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
                    dtypes[value] = dtype
                for phase in engine["phases"]:
                    extra = {key: value for key, value in phase.items() if key not in {"kind", "writes", "reads", "code"}}
                    if phase.get("fused_norm"):
                        norm_shape = tuple(int(x) for x in _field(values[planned[1]], "shape"))
                        layouts[planned[1]] = TensorLayout(norm_shape, _contiguous(norm_shape), planned[1])
                        attention_lower.add_fused_nodes(phase["kind"], phase["code"],
                            dict(extra, group=name, atomic_output=name), phase["reads"], phase["writes"],
                            layouts=layouts, dtypes=dtypes, make_layout=TensorLayout, add_heavy=add_heavy,
                            add_view=attention_view, reserved_names=names)
                        indexed_skip.update(planned)
                    else:
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
        if INLINE_EMBEDDING_GATHER and base == "embedding" and ordered_index_gradients and not retain_all:
            # LATCH-STEP-04: the forward row gather as an elementwise value, so consumers (loss, cotangent, state
            # copy) may inline it instead of materializing it in its own phase. Index and table reads are
            # ordinary tracked loads (an [*, 1] index view and a [1, D] table-row view) for scheduling and
            # in-place safety; the element value is exactly table[ids[row], col].
            w_ref, i_ref = _ref(args[0], names), _ref(args[1], names)
            lw = layouts.get(w_ref) if w_ref else None
            li = layouts.get(i_ref) if i_ref else None
            iv, wv = name + "__gidx", name + "__grow"
            if (lw is not None and li is not None and dtypes[w_ref] == "float32" and dtypes[name] == "float32"
                    and dtypes[i_ref] == "int64" and len(lw.shape) == 2 and lw.strides[1] == 1
                    and shape == (*li.shape, lw.shape[1])
                    and not ({iv, wv} & (set(names) | set(layouts)))):
                width, row_stride = lw.shape[1], lw.strides[0]
                layouts[iv] = TensorLayout((*li.shape, 1), (*li.strides, 0), li.root, li.offset)
                dtypes[iv] = "int64"
                layouts[wv] = TensorLayout((1, width), (row_stride, 1), lw.root, lw.offset)
                dtypes[wv] = "float32"

                def gather_render(position, rd, iv=iv, wv=wv, row_stride=row_stride):
                    index = rd({"ref": iv}, position)
                    column = rd({"ref": wv}, position)
                    head = f"{_ident(wv)}["
                    if head not in column:
                        head = f"v_{wv}["
                    if head not in column:
                        raise LoweringError("internal: gathered row load has an unexpected form")
                    return column.replace(head, f"{head}(long long)({index}) * {row_stride}LL + ", 1)
                gather_loads = []

                def gather_probe(arg, idx):
                    gather_loads.append(_ref(arg, names))
                    return f"v_{_ref(arg, names)}[0]"
                gather_render("i", gather_probe)
                nodes[name] = _Node(name=name, kind="elementwise", shape=shape, dtype="float32", render=gather_render,
                                    base=base, target=str(_field(op, "target", _field(op, "op"))),
                                    own_cost=2, loads=gather_loads, order=len(nodes))
                continue
        emitted = emit_operation(target, args, kwargs, name, shape, layouts, dtypes, names, placeholder_read,
                                 warp_rows="pops2a" if warp_rows else False, inputs=tuple(inputs))
        if emitted is not None:
            kind, code, extra, helper = emitted
            if dynamic is not None and any(v in dynamic.axes for v in (name, *refs)):
                raise LoweringError(f"{name}: {kind} has no bounded active-extent contract in the flat schedule")
            if helper and helper not in helper_parts:
                helper_parts.append(helper)
            add_heavy(kind, op, name, code, refs, **extra)
            continue
        if base in {"mm", "matmul", "bmm", "addmm"}:
            aa, bb = (args[0], args[1]) if base != "addmm" else (args[1], args[2])
            a, b = _ref(aa, names), _ref(bb, names)
            la, lb = layouts[a], layouts[b]
            dynamic_product = dynamic is not None and any(v in dynamic.axes for v in (a, b, name))
            if dynamic_product and (matrix_policy == "frozen" or base == "addmm"
                                    or not dtypes[a] == dtypes[b] == dtypes[name]):
                raise LoweringError("bounded dynamic flat schedule requires decomposed same-dtype products")
            if matrix_policy != "frozen" and base != "addmm" and dtypes[a] == dtypes[b] == dtypes[name]:
                from .matrix_codegen import select
                from .matrix_rules import HEADER, emit_group
                spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch)
                if dynamic_product:
                    spec = dynamic.matrix_spec(spec, a, b, name)
                    reason += "; masked-edge body over the capacity plan with runtime active extent"
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
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
                    pops2_layouts[name] = plan['selection_contract']
                    if plan['helpers'] not in helper_parts:
                        helper_parts.append(plan['helpers'])
                    extra_headers.extend(h for h in plan['headers'] if h not in extra_headers)
                    add_heavy('gemm', op, name, plan['code'], plan['operand_roots'], **plan['metadata'])
                    continue
            if matrix_policy != 'frozen' and base != 'addmm' and (matrix_engine == 'popsb' or isinstance(matrix_engine, dict)):
                from .matrix_codegen import fp32_async
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
                from .matrix_codegen import split_k as split_codegen
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
            if dynamic is not None and (source in dynamic.axes or name in dynamic.axes):
                if integer_sum:
                    raise LoweringError("bounded dynamic flat schedule admits floating reductions")
                selected = dynamic.reduction(name=name, source=source, src=src, dims=dims, kept=kept, out=out,
                                             read=lambda position: storage_read(source, position),
                                             rounded=rounded_result, mean=base == "mean", out_shape=shape)
                if selected is not None:
                    code, extra = selected
                    add_heavy("reduction", op, name, code, refs, **extra)
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
            if results == 1 and extent >= FULL_REDUCTION_MIN and dtypes[name] == "float32" and exact_reductions:
                # C-OPS4-B (adapted from P6-C msum, staged path): a long single-output reduction runs on every
                # CTA. CTA b sums the contiguous chunk b with the CTA tree; the last CTA to arrive (ticket counter
                # in the split-K counter scratch, zeroed per region, restored to zero) combines the partials in
                # fixed CTA order. FP32 reassociation of the single-CTA order; deterministic for a grid size.
                read_r = storage_read(source, pos)
                code = f'''{{ const long long o = 0;
  __shared__ unsigned latch_full_last_;
  const long long chunk_ = ({extent}LL + gridDim.x - 1) / gridDim.x, lo_ = (long long)blockIdx.x * chunk_;
  const long long hi_ = lo_ + chunk_ < {extent}LL ? lo_ + chunk_ : {extent}LL;
  float acc = 0.0f;
/*FUSED_ROW_HOIST*/
  for (long long r = lo_ + threadIdx.x; r < hi_; r += blockDim.x) {{/*FUSED_ROW_BODY*/ acc += {read_r}; }}
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {{
      (/*P2_PART*/)[blockIdx.x] = acc;
      __threadfence();
      latch_full_last_ = atomicAdd(/*P2_CNT*/, 1u) == gridDim.x - 1u;
    }}
  }}
  __syncthreads();
  if (latch_full_last_) {{
    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += __ldcg((/*P2_PART*/) + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {{
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) {{ {out}[0] = acc{scale}; *(/*P2_CNT*/) = 0u; }}
    }}
  }}
  __syncthreads();
}}'''
                add_heavy("reduction", op, name, code, refs, reduction_elements=extent, engine="multi_cta_full",
                          p2_split={"counter_words": 1, "partial_words": 1024},
                          arithmetic="per-CTA contiguous chunk trees, then fixed-order combine of CTA partials")
                continue
            if exact_reductions:
                from .codegen_rules import exact_reduction
                # LATCH-STEP-04: a reduction over a non-broadcast source reads each source element once, so its
                # unrolled leaf sites count as one use for producer inlining (no recomputation).
                lay_ = layouts[source]
                leaf_group = (name if REDUCTION_TOKEN_GROUPS and all(st != 0 for st, ex in zip(lay_.strides, lay_.shape)
                                                                     if ex != 1) else None)
                if REFERENCE_P6D_REDUCTIONS:
                    # LATCH-DENSE-EXEC: the P6-D exact engines (warp per row for unit-stride extents up to 4096;
                    # coalesced column engine) replay this same frozen 256-thread tree bitwise, without one CTA and
                    # two block barriers per output. Row-consumer sites stay empty in this schedule.
                    from . import p6d_rules
                    selected = p6d_rules.reduction_engine(out=out, position=pos, extent=extent, results=results,
                                                          read=lambda position: storage_read(source, position, leaf_group),
                                                          rounded=rounded_result, scale=scale, src=layouts[source],
                                                          dims=dims, kept=kept)
                    if selected is not None:
                        code, extra = selected
                        extra = {k: v for k, v in extra.items() if k != 'p9'}
                        add_heavy('reduction', op, name, code.replace(p6d_rules.ROW_MARK, ''), refs, **extra)
                        continue
                selected = exact_reduction(out=out, position=pos, extent=extent, results=results,
                                           read=lambda position: storage_read(source, position, leaf_group),
                                           rounded=rounded_result, scale=scale)
                if selected is not None:
                    code, extra = selected
                    add_heavy('reduction', op, name, code, refs, **extra)
                    continue
            add_heavy("reduction", op, name, code, refs, reduction_elements=extent)
            continue

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
                if base == "lerp":
                    return _lerp_scalar_expression(target, args, kwargs, name, dtypes, names, rd,
                        lambda: rd_ref(kwargs["weight"], idx))
                if base in {"add", "sub"}:
                    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
                    return f"({rd(0)} {'+' if base == 'add' else '-'} ({rd(1)} * {_float(alpha)}))"
                if base in {"mul", "div"}:
                    if base == "div" and kwargs.get("rounding_mode") is not None:
                        raise LoweringError("rounded division unsupported")
                    return f"({rd(0)} {'*' if base == 'mul' else '/'} {rd(1)})"
                if base == "pow":
                    if args[1] == 2:
                        return f"({rd(0)} * {rd(0)})"
                    if (POW_ONE_IDENTITY and isinstance(args[1], (int, float)) and not isinstance(args[1], bool)
                            and float(args[1]) == 1.0):
                        # Rule from P-OPS3-C: libdevice powf(x, 1.0f) == x for every float bit pattern
                        # (exhaustive H100 check, pops3c_jobs.pow_identity_check); ATen pow(x, 1) is a copy.
                        return f"({rd(0)})"
                    return f"powf({rd(0)}, {rd(1)})"
                if base == "neg":
                    return f"(-{rd(0)})"
                if base == "reciprocal":
                    return f"(1.0f / {rd(0)})"
                if base in {"tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs"}:
                    fn = {"abs": "fabsf"}.get(base, base + "f")
                    return f"{fn}({rd(0)})"
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

        def probe(arg, idx):
            ref = _ref(arg, names)
            if ref is None:
                if (isinstance(arg, (int, float, bool)) or (dynamic is not None and isinstance(arg, dict) and 'dynamic_batch_scalar' in arg)):
                    return _float(arg)
                raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
            probe_loads.append(ref)
            return f"v_{ref}[0]"
        render("i", probe)
        if dynamic is not None:
            dynamic.check_elementwise(name, shape, probe_loads, layouts)
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
    counted_groups = set()
    for k, (kind, ref, _, _) in enumerate(token_loads):
        if kind == "storage_addr":
            continue
        if k in token_group:
            if (token_group[k], root_of(ref)) in counted_groups:
                continue
            counted_groups.add((token_group[k], root_of(ref)))
        uses[root_of(ref)] = uses.get(root_of(ref), 0) + 1
    inline, total_cost = set(), {}
    broadcast_to = {}
    if scalar_hoist:
        for node in nodes.values():
            if node.kind == "elementwise":
                for ref in node.loads:
                    root = root_of(ref)
                    broadcast_to[root] = max(broadcast_to.get(root, 0), _numel(node.shape))
    inline_total, inline_work = INLINE_TOTAL_LIMIT, INLINE_WORK_LIMIT
    if INDEXED_INLINE_LIMITS and index_decisions and not retain_all:
        inline_total = max(inline_total, INDEXED_INLINE_LIMITS[0])
        inline_work = max(inline_work, INDEXED_INLINE_LIMITS[1])
    hoisted = []
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        cost = node.own_cost + sum(total_cost[root_of(ref)] for ref in node.loads if root_of(ref) in inline)
        total_cost[node.name] = cost
        # LATCH-STEP-04: one-element chains over one-element operands (e.g. the clip coefficient's NaN test
        # and clamp) may be recomputed by consumers, including bool links; same expressions, no new phase.
        scalar_chain = (_numel(node.shape) == 1 and node.dtype in ("float32", "bfloat16", "bool")
                        and all(_numel(layouts[root_of(ref)].shape) == 1
                                and dtypes[root_of(ref)] in ("float32", "bfloat16", "bool") for ref in node.loads))
        gather_value = node.base == "embedding"   # LATCH-STEP-04: its int64 load is a row index, not a value
        if (retain_all or node.name in forced or ((node.dtype not in ("float32", "bfloat16")
                or any(dtypes[root_of(ref)] not in ("float32", "bfloat16") for ref in node.loads))
                and not scalar_chain and not gather_value)):
            continue
        if (scalar_hoist and _numel(node.shape) == 1 and cost >= SCALAR_HOIST_MIN_COST
                and broadcast_to.get(node.name, 0) >= SCALAR_HOIST_MIN_CONSUMER):
            hoisted.append(node.name)
            continue
        n_uses = uses.get(node.name, 0)
        if n_uses and ((cost <= inline_total and cost * n_uses <= inline_work)
                       or (scalar_chain and cost <= 6 * INLINE_TOTAL_LIMIT)):
            inline.add(node.name)

    def ctype_round(expression, dtype):
        return f"__float2bfloat16_rn({expression})" if dtype == "bfloat16" else expression

    # LATCH-STEP-04: while an elementwise loop group renders, inlined one-element nodes (index-invariant:
    # bias corrections, clip coefficient) are evaluated once per thread before the loop, same expression.
    hoist_stack = []

    def expand(ref, shape, idx, loader):
        """Float expression for element idx of `ref` viewed at `shape`."""
        layout = layouts[ref]
        root = layout.root
        if root in inline:
            node = nodes[root]
            hoist = hoist_stack[-1] if hoist_stack and _numel(node.shape) == 1 else None
            if hoist is not None and root in hoist:
                return hoist[root][0]
            position = _index32(layout, shape, idx)
            if layout.offset:
                position = f"{_u(layout.offset)} + {position}"
            position = position if re.fullmatch(r"\w+", position) else f"({position})"
            inner = node.render(position, lambda arg, i2: operand(arg, node.shape, i2, loader))
            if node.dtype == "bfloat16":
                inner = f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
            if hoist is not None:
                hoist[root] = (f"h_{_ident(root)}", inner)
                return hoist[root][0]
            return inner
        return loader(ref, shape, idx)

    def operand(arg, shape, idx, loader):
        ref = _ref(arg, names)
        if ref is None:
            if (isinstance(arg, (int, float, bool)) or (dynamic is not None and isinstance(arg, dict) and 'dynamic_batch_scalar' in arg)):
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
            node.render("i", lambda arg, i2: operand(arg, node.shape, i2, loader))
        else:
            for k in node.loads:
                kind, ref, shape, idx = token_loads[k]
                if kind == "index":
                    expand(ref, shape, idx, loader)
                elif kind == "storage4":
                    for j in range(4):
                        expand_storage(ref, f"({idx}) + {j}", loader)
                elif kind == "storage_addr":
                    if root_of(ref) not in inline:
                        expand_storage(ref, idx, loader)
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
            inner = node.render(flat, lambda arg, i2: operand(arg, node.shape, i2, loader))
            if node.dtype == "bfloat16":
                return f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
            return inner
        return loader(ref, None, position)

    # ------------------------------------------------------------------ scheduling
    items = [node for node in nodes.values() if node.kind != "elementwise" or node.name not in inline]
    loads_of = {node.name: memory_loads(node) for node in items}
    from .pops2b_rules import IDLE_HEAVY_KINDS
    # P-OPS2-B rules belong to programs with exact-order BF16 products; others keep the base schedule.
    pops2b = pops2b and any(node.kind == "gemm" and (node.extra.get("matrix_spec") or {}).get("dtype") == "bfloat16"
                            for node in nodes.values())
    absorbed = {}                         # BF16 cast -> (product, operand) materialized in operand staging
    if pops2b and pops2b_casts and not retain_all and target_arch == "sm_90":
        from .pops2b_rules import select_absorbed_casts
        absorbed = select_absorbed_casts(items=items, loads_of=loads_of, nodes=nodes, layouts=layouts, dtypes=dtypes,
                                         protected={root_of(n) for n in outputs + list(writes.values())},
                                         mutable=set(writes), cost_of=lambda n: total_cost.get(n, 0))
        if absorbed:
            items = [node for node in items if node.name not in absorbed]
            for cast, decision in absorbed.items():
                cast_loads = loads_of.pop(cast)
                for product, _ in decision["readers"]:
                    loads_of[product] = [entry for entry in loads_of[product] if entry[0] != cast] + cast_loads

    red_epi_of = {}                       # LATCH-STEP-04: reduction-epilogue value -> its reduction

    def register_load(node, root, layout, shape, idx):
        """A same-loop register read: same flat index over the same element count."""
        return (node.kind == "elementwise" and idx == "i" and root in nodes and nodes[root].kind == "elementwise"
                and root not in red_epi_of
                and root not in inline and root not in absorbed and layout.root == root and layout.contiguous
                and layout.offset == 0 and _numel(nodes[root].shape) == _numel(node.shape) == _numel(shape)
                and (dynamic is None or dynamic.active(root) == dynamic.active(node.name)))

    readers = {}                          # root -> item names reading it from memory or register
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, []).append(node.name)
    order = {node.name: k for k, node in enumerate(items)}
    in_place = {}
    scattered = set()
    if not retain_all:
        for source_root, targets in sources.items():
            if len(targets) != 1 or source_root not in nodes or source_root in inline:
                continue
            target = targets[0]
            node = nodes[source_root]
            # P9-D: an index_copy whose `self` operand is the committed state itself becomes a row scatter in place.
            scatter = node.extra.get("in_place_scatter") if node.kind == "index_copy" else None
            if (scatter and not retain_all and writes.get(target) == source_root
                    and root_of(node.extra["self_operand"]) == target and source_root not in inputs
                    and layouts[target].shape == layouts[source_root].shape and layouts[target].contiguous
                    and layouts[node.extra["self_operand"]].contiguous and layouts[node.extra["self_operand"]].offset == 0
                    and dtypes[target] == dtypes[source_root]
                    and not any(order[r] > order[source_root] for r in readers.get(target, ()) if r != source_root)):
                in_place[target] = source_root
                scattered.add(source_root)
                continue
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
    for name in scattered:
        # The scatter writes only replaced rows of the committed storage and no longer reads the old state.
        node = nodes[name]
        node.code = node.extra["in_place_scatter"]
        target = inplace_source[name]
        node.direct = [ref for ref in node.direct if root_of(ref) != target]
        loads_of[name] = [entry for entry in loads_of[name] if entry[0] != target]
        readers[target] = [r for r in readers.get(target, []) if r != name]

    phase_of, phases = {}, []             # phase: {"heavy": [names], "loops": {numel: [names]}}

    def new_phase():
        phases.append({"heavy": [], "loops": {}})
        return len(phases) - 1

    def stackable(h):
        return h.kind == "reduction"

    def reduction_epilogue_target(node, earliest):
        """The exact reduction R (phase earliest - 1) whose results this elementwise node reads at identity index,
        when every other operand comes from an earlier phase or the inputs; else None."""
        if node.name in inplace_source or node.dtype not in ("float32", "bfloat16"):
            return None
        numel = _numel(node.shape)
        target = None
        for root, layout, shape, idx in loads_of[node.name]:
            if root not in phase_of:
                continue
            q = phase_of[root]
            if q < earliest - 1:
                continue
            if q != earliest - 1 or (target is not None and root != target):
                return None
            rn = nodes.get(root)
            if (rn is None or rn.kind != "reduction" or "/*RED_EPILOGUE*/" not in rn.code
                    or not (idx == "i" and shape is not None and layout.root == root and layout.contiguous
                            and layout.offset == 0 and _numel(shape) == layout.numel == numel
                            and _numel(layouts[root].shape) == numel)):
                return None
            target = root
        return target

    def gemm_or_one_other(phase, node):
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        if not heavy:
            return True
        indexed = lambda h: bool(h.extra and h.extra.get("indexed_engine"))
        if (INDEXED_COSCHEDULE and (indexed(node) or stackable(node))
                and all(indexed(h) or stackable(h) for h in heavy) and any(map(indexed, heavy + [node]))
                and len(heavy) < REDUCTION_STACK_MAX):
            return True
        if (reduction_stack and stackable(node) and all(stackable(h) for h in heavy)
                and len(heavy) < REDUCTION_STACK_MAX):
            return True
        if not coschedule_matrix:
            return False
        if node.kind == 'matrix_counter_init':
            return all(h.kind == 'matrix_counter_init' for h in heavy)
        if dynamic is not None:
            # Bounded reductions (a few CTAs) may run on the CTAs a small BF16 matrix group leaves idle.
            joined = [*heavy, node]
            specs = [h for h in joined if h.extra and h.extra.get('matrix_spec')]
            others = [h for h in joined if not (h.extra and h.extra.get('matrix_spec'))]
            if specs and others and all(h.kind == 'gemm' for h in specs) and len(specs) <= 2 and all(
                    h.kind == 'reduction' and h.extra.get('idle_ok') for h in others) and len(others) <= REDUCTION_STACK_MAX:
                from .matrix_rules import emit_group as _group, TARGET_GRID
                tiles = _group([h.extra['matrix_spec'] for h in specs], ident=str)[1]['tiles']
                ctas = sum(int(h.extra.get('work_items', 1)) for h in others)
                from . import dynamic_flat as _df
                if _df.REDUCTIONS_AFTER_TILES and ctas <= _df.IDLE_REDUCTION_CTAS:
                    return True     # reductions run after their CTAs' matrix tiles (idle_heavy wraps the range)
                if (_df.HALF_GRID_PLAN and 2 * tiles > TARGET_GRID and len(specs) == 1
                        and 'plan_grid' not in specs[0].extra['matrix_spec']):
                    # One product filling the grid: its half-grid tile plan (fewer, larger tiles) leaves CTAs
                    # for the reductions, instead of a separate phase and grid barrier for them.
                    half = {**specs[0].extra['matrix_spec'], 'plan_grid': TARGET_GRID // 2}
                    half_tiles = _group([half], ident=str)[1]['tiles']
                    if 2 * half_tiles <= TARGET_GRID and half_tiles + ctas <= TARGET_GRID:
                        specs[0].extra['matrix_spec']['plan_grid'] = TARGET_GRID // 2
                        return True
                return 2 * tiles <= TARGET_GRID and tiles + ctas <= TARGET_GRID
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

    for node in items:
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
            numel = _numel(node.shape) if dynamic is None else dynamic.key(node.name, _numel(node.shape))
            red = (reduction_epilogue_target(node, earliest)
                   if dynamic is None and REDUCTION_EPILOGUES and not retain_all and earliest >= 1 else None)
            if red is not None:
                phase = phase_of[red]
                phases[phase].setdefault("red_epilogue", {}).setdefault(red, []).append(node.name)
                phase_of[node.name] = phase
                red_epi_of[node.name] = red
                continue
            # Register producers must share the loop: same phase and element count.
            phase = earliest
            while phase >= len(phases):
                new_phase()
            phases[phase]["loops"].setdefault(numel, []).append(node.name)
            phase_of[node.name] = phase
        else:
            phase = earliest
            while True:
                while phase >= len(phases):
                    new_phase()
                if gemm_or_one_other(phase, node):
                    break
                phase += 1
            phases[phase]["heavy"].append(node.name)
            phase_of[node.name] = phase
            for cast, decision in absorbed.items():
                if decision["mode"] == "materialize" and decision["readers"][0][0] == node.name:
                    phase_of[cast] = phase
    stacked_moves = []
    if reduction_stack and not retain_all:
        consumers = {}
        for item in items:
            for root, _, _, _ in loads_of[item.name]:
                consumers.setdefault(root, set()).add(item.name)
        commit_of = {target: phase_of[source] for target, source in in_place.items()}
        for p_ in range(len(phases)):
            heavy = [nodes[n] for n in phases[p_]["heavy"]]
            if not heavy or phases[p_]["loops"] or not all(stackable(h) for h in heavy):
                continue
            for h in list(heavy):
                if h.name in inplace_source:
                    continue
                hi = len(phases) - 1
                for r in consumers.get(h.name, ()):
                    hi = min(hi, phase_of[r] - 1)
                for root, _, _, _ in loads_of[h.name]:
                    if root in commit_of:
                        hi = min(hi, commit_of[root] - 1)
                for q in range(p_ + 1, hi + 1):
                    other = [nodes[n] for n in phases[q]["heavy"]]
                    if other and all(stackable(o) for o in other) and len(other) < REDUCTION_STACK_MAX:
                        phases[p_]["heavy"].remove(h.name)
                        phases[q]["heavy"].append(h.name)
                        phases[q]["heavy"].sort(key=lambda n: nodes[n].order)
                        phase_of[h.name] = q
                        stacked_moves.append({"item": h.name, "from": p_, "to": q})
                        break
        empty = [q for q, ph in enumerate(phases) if not ph["heavy"] and not ph["loops"]]
        if empty:
            remap, kept = {}, []
            for q, ph in enumerate(phases):
                if ph["heavy"] or ph["loops"]:
                    remap[q] = len(kept)
                    kept.append(ph)
            phases[:] = kept
            for n in phase_of:
                phase_of[n] = remap[phase_of[n]]
    commit = len(phases)
    deferred, preload = {}, {}
    if pops2b and pops2b_defer and pops2b_idle and not retain_all and target_arch == "sm_90":
        from .pops2b_rules import defer_products, idle_heavy_model
        deferred, preload = defer_products(items=items, loads_of=loads_of, phase_of=phase_of, phases=phases,
                                           nodes=nodes, idle_model=idle_heavy_model(), numel_of=_numel,
                                           inplace_source=inplace_source, dtypes=dtypes)
        commit = len(phases)
    if place_by_slack and not retain_all:
        idle_heavy = None
        if pops2b and pops2b_idle and target_arch == "sm_90":
            from .pops2b_rules import idle_heavy_model
            idle_heavy = idle_heavy_model(pops2b_idle_fill)
        _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                        forced, _numel, idle_heavy=idle_heavy, deferred=deferred, preload=preload,
                        **({"key_of": lambda n: dynamic.key(n, _numel(nodes[n].shape))} if dynamic is not None else {}))
        commit = len(phases)
    epilogue_of, epilogues, permuted_members = {}, {}, {}
    if pops2b and pops2b_epilogue and not retain_all and target_arch == "sm_90":
        from .pops2b_rules import attach_epilogues
        epilogue_of, epilogues, permuted_members = attach_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            dtypes=dtypes, in_place=in_place, inplace_source=inplace_source, register_load=register_load,
            numel_of=_numel, permuted_copies=pops2b_permuted and dynamic is None, cost_of=lambda n: total_cost.get(n, 0),
            cost_limit=pops2b_epi_cost,
            stream_gate=pops2b_epi_stream_gate and (dynamic is None or dynamic.epilogue_stream_gate),
            background=set(deferred) if pops2b_defer_epilogue else set(),
            **({"key_of": lambda n: dynamic.key(n, _numel(layouts[n].shape)),
                "tile_work_limit": dynamic.epilogue_tile_work} if dynamic is not None else {}))
        commit = len(phases)


    for e_name, r_name in red_epi_of.items():
        if (phase_of[e_name] != phase_of[r_name] or e_name not in phases[phase_of[r_name]].get("red_epilogue", {}).get(r_name, ())
                or any(e_name in members for ph in phases for members in ph["loops"].values())):
            raise LoweringError(f"internal: reduction epilogue {e_name} separated from {r_name}")
    reduction_epilogues = {}
    if dynamic is not None and not retain_all:
        from .dynamic_flat import attach_reduction_epilogues, compact_heavy
        compact_heavy(phases=phases, phase_of=phase_of, items=items, loads_of=loads_of, nodes=nodes,
                      epilogues=epilogues, in_place=in_place, inplace_source=inplace_source,
                      can_join=gemm_or_one_other, register_load=register_load)
        reduction_epilogues = attach_reduction_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            in_place=in_place, inplace_source=inplace_source, register_load=register_load,
            key_of=lambda n: dynamic.key(n, _numel(layouts[n].shape)), epilogue_of=epilogue_of)
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
    if red_epi_of:
        epilogue_of = {**epilogue_of, **red_epi_of}
    if epilogue_of:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError,
                        epilogue_of=epilogue_of, permuted=permuted_members)
    else:
        _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError)
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
    prezero_jobs, prezeroed = {}, set()
    if INDEXED_PREZERO and not retain_all:
        for node in items:
            pz = (node.extra or {}).get("prezero")
            k_pz = phase_of.get(node.name, 0)
            if (pz and k_pz >= 1 and node.name in classes
                    and all(v["value"] in layouts and layouts[v["value"]].root == node.name
                            and dtypes[v["value"]] == "float32" and layouts[v["value"]].numel == v["elements"]
                            and layouts[v["value"]].contiguous for v in pz["values"])):
                _, first, last = classes[node.name]
                classes[node.name] = ("live", min(first, k_pz - 1), last)
                prezero_jobs.setdefault(k_pz - 1, []).extend(pz["values"])
                prezeroed.add(node.name)
    if retain_all:
        classes = {name: ("live", phase_of.get(name, 0), commit) for name in storage_roots}
    offsets, arena_words, _ = arena_plan.assign(storage_roots, sizes, classes, commit)
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
        def substitute(match):
            kind, ref, shape, idx = token_loads[int(match.group(1))]
            if kind == "index":
                return expand(ref, shape, idx, memory_loader)
            if kind == "storage4":
                if root_of(ref) in inline:
                    return "make_float4(" + ", ".join(expand_storage(ref, f"({idx}) + {j}", memory_loader)
                                                      for j in range(4)) + ")"
                return f"(*reinterpret_cast<const float4*>(&{memory_loader(ref, None, idx)}))"
            if kind == "storage_addr":
                if root_of(ref) in inline:
                    return "0ULL"
                return f"(unsigned long long)(&{memory_loader(ref, None, idx)})"
            return expand_storage(ref, idx, memory_loader)
        code = _TOKEN.sub(substitute, node.code)
        if "/*RED_EPI*/" in code:
            code = code.replace("/*RED_EPI*/", reduction_epilogue_code(node.name))
        return code

    def reduction_epilogue_code(red):
        """Chain consuming reduction output o in the thread that produced it (dynamic_flat)."""
        members = reduction_epilogues.get(red, ())
        if not members:
            return ""
        member_set = set(members)

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            if origin in member_set and idx == "i" and layouts[ref].contiguous and layouts[ref].offset == 0:
                expression = f"r_{origin}"
                return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        lines = [f"{{ const unsigned i = (unsigned)o;  // epilogue of {red}: {', '.join(members)}"]
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
            lines.append(f"  const {_CTYPES[node.dtype]} r_{name} = {ctype_round(expression, node.dtype)};")
        lines += [f"  {_ident(name)}[i] = r_{name};" for name in members if name in stored]
        lines.append("}")
        return "\n".join(lines)

    vector_helpers_needed = []

    partitioned_phases = []

    def loop_components(members):
        """Members of one loop group split into dependency components (register reads of other members)."""
        member_set, parent, owner = set(members), {n: n for n in members}, {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for name in members:
            found = set()

            def probe(ref, shape, idx, found=found):
                origin = pre_layout_root.get(ref, layouts[ref].root)
                if origin in member_set:
                    found.add(origin)
                return "0.0f"
            node = nodes[name]
            node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, probe))
            for other in found:
                parent[find(other)] = find(name)
            # Old state committed in place by one member must not be read by another component's CTAs.
            for root, _, _, _ in loads_of.get(name, ()):
                if root in in_place or root in inplace_source:
                    anchor = owner.setdefault(root, name)
                    parent[find(anchor)] = find(name)
        groups = {}
        for name in members:
            groups.setdefault(find(name), []).append(name)
        return list(groups.values())

    def loop_code(numel, members, exclude, cta_range=None, row_guard=None):
        active_loop = dynamic is not None and dynamic.is_active_key(numel)
        if dynamic is not None:
            numel = dynamic.numel_of_key(numel)
        if numel >= (1 << 31):
            raise LoweringError("fused elementwise loops require fewer than 2^31 elements")
        loop_regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        vector = floating and numel % 4 == 0 and numel >= VECTOR_MIN_ELEMENTS and (
            not active_loop or (numel // dynamic.maximum) % 4 == 0)
        if row_guard is not None and row_guard["D"] % 4:
            vector = False
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
        hoisted = {}
        hoist_stack.append(hoisted)
        try:
            for name in members:
                node = nodes[name]
                expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
                compute.append(f"{loop_regs[name]}[v] = {ctype_round(expression, node.dtype)};")
        finally:
            hoist_stack.pop()
        hoist_code = "".join(f"  const auto {var} = {expr};\n" for var, expr in hoisted.values())
        # LATCH-STEP-04: rows already updated by the indexed phase (zero gradient rows) are skipped,
        # unless the clip norm is NaN (then every output is NaN, as in the unsplit update).
        vguard = sguard = ""
        if row_guard is not None:
            hoist_code += f"  const bool latch_rg_ = !({row_guard['norm']}[0] == {row_guard['norm']}[0]);\n"
            vguard = f" && (latch_rg_ || {row_guard['flags']}[(4u * q) / {row_guard['D']}u] != 0.0f)"
            sguard = f" && (latch_rg_ || {row_guard['flags']}[i / {row_guard['D']}u] != 0.0f)"
        lanes = 4 if vector else 1
        count = numel // lanes
        # Active loops: unroll and vector width follow capacity; the bound is the active count.
        bound = dynamic.count(numel, lanes) if active_loop else _u(count)
        u = 1
        per_thread = -(-count // RESIDENT_THREADS)
        cap = max(1, LIVE_VALUE_CAP // max(1, lanes * len(members)))
        while u * 2 <= min(per_thread, UNROLL_MAX, cap):
            u *= 2
        declarations_local = " ".join(f"{_CTYPES[nodes[n].dtype]} {loop_regs[n]}[{u * lanes}];" for n in members)
        if cta_range is not None:
            lo, hi, total = cta_range
            per_thread = -(-count * total // max(1, (hi - lo) * RESIDENT_THREADS))
            u = 1
            while u * 2 <= min(per_thread, UNROLL_MAX, cap):
                u *= 2
            declarations_local = " ".join(f"{_CTYPES[nodes[n].dtype]} {loop_regs[n]}[{u * lanes}];" for n in members)
            head = (f"  const unsigned long long g_ = gridDim.x;\n"
                    f"  const unsigned b0 = (unsigned)(({lo}ULL * g_) / {total}ULL), b1_ = (unsigned)(({hi}ULL * g_) / {total}ULL);\n"
                    f"  const unsigned b1 = b1_ > b0 ? b1_ : b0 + 1u;\n"
                    f"  const unsigned gs = (b1 - b0) * blockDim.x;\n"
                    f"  if (blockIdx.x >= b0 && blockIdx.x < b1)\n")
            start_block = "(blockIdx.x - b0)"
        elif exclude:
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
            body = (f"      const unsigned q = base + (unsigned)u * gs;\n      if (q < {bound}{vguard}) {{\n{loads}"
                    f"#pragma unroll\n        for (int j = 0; j < 4; ++j) {{\n"
                    f"          const unsigned i = 4u * q + (unsigned)j; const int v = u * 4 + j;\n          "
                    + "\n          ".join(compute) + "\n        }\n      }\n")
            stores = "".join(f"        latch_st4({_ident(n)} + 4u * q, {loop_regs[n]}[u*4], {loop_regs[n]}[u*4+1], "
                             f"{loop_regs[n]}[u*4+2], {loop_regs[n]}[u*4+3]);\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned q = base + (unsigned)u * gs;\n"
                           f"      if (q < {bound}{vguard}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        else:
            body = (f"      const unsigned i = base + (unsigned)u * gs; const int v = u;\n"
                    f"      if (i < {bound}{sguard}) {{\n        " + "\n        ".join(compute) + "\n      }\n")
            stores = "".join(f"        {_ident(n)}[i] = {loop_regs[n]}[u];\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned i = base + (unsigned)u * gs;\n"
                           f"      if (i < {bound}{sguard}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        return (f"{{ // {numel} elements{' (x4 vector)' if vector else ''}: {', '.join(members)}\n" + hoist_code + head +
                f"  for (unsigned base = {start_block} * blockDim.x + threadIdx.x; base < {bound}; base += {u}u * gs) {{\n"
                f"    {declarations_local}\n#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n" + body + "    }\n"
                + store_block + "  }\n}")

    def fused_row_code(numel, members, index="r"):
        """Scalar elementwise body at element i = <index> (a reduction's chunk loop or result index)."""
        regs = {name: f"f_{name}" for name in members}
        member_set = set(members)

        def loader(ref, shape, idx):
            origin = pre_layout_root.get(ref, layouts[ref].root)
            identity = (idx == "i" and shape is not None and _numel(shape) == numel
                        and layouts[ref].numel == numel and layouts[ref].contiguous and layouts[ref].offset == 0)
            if identity and origin in member_set and _numel(nodes[origin].shape) == numel:
                return f"__bfloat162float({regs[origin]})" if dtypes[origin] == "bfloat16" else regs[origin]
            return memory_loader(ref, shape, idx)
        hoisted, lines = {}, []
        hoist_stack.append(hoisted)
        try:
            for name in members:
                node = nodes[name]
                expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
                lines.append(f"const {_CTYPES[node.dtype]} {regs[name]} = {ctype_round(expression, node.dtype)};")
        finally:
            hoist_stack.pop()
        hoist = "".join(f"  const auto {var} = {expr};\n" for var, expr in hoisted.values())
        stores = [f"{_ident(n)}[i] = {regs[n]};" for n in members if n in stored]
        return hoist, "{ const unsigned i = (unsigned)" + index + "; " + " ".join(lines + stores) + " }"

    fused_row_report = []

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

    def tile_functor(body, prefetch, ebytes):
        names = list(dict.fromkeys(re.findall(r"\bv_\w+\b", body + "\n" + prefetch)))
        body_c, prefetch_c = _canonical_names(body, names), _canonical_names(prefetch, names)
        fields = " ".join(f"{_CTYPES[ident_dtype[name]]}* m{k};" for k, name in enumerate(names))
        if dynamic is not None:
            # Row-bounded interface of kernels/dynamic_flat.cuh: latch_rows existing rows of this tile. The
            # functor also carries the device active extent for batch-dependent scalars in its expressions.
            fields = fields + " int latch_active_batch;"
            key = ("tile_rows", fields, body_c, prefetch_c, ebytes)
            if key not in functor_types:
                type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
                functor_types[key] = type_name
                helper_parts.append(
                    f"struct {type_name} {{\n  {fields}\n  static constexpr bool kTile = true;\n"
                    f"  template <int BM, int BN> __host__ __device__ static constexpr int ebytes() {{ return {ebytes}; }}\n"
                    f"  template <int BM, int BN> __device__ __forceinline__ void prefetch_rows(char* E, unsigned base, unsigned ldc, int latch_rows) const {{\n"
                    f"    {prefetch_c}\n  }}\n"
                    f"  template <int BM, int BN> __device__ __forceinline__ void tile_rows(const float* T, int ldt, const char* E, "
                    f"unsigned base, unsigned ldc, int latch_rows) const {{\n    {body_c}\n  }}\n}};")
            return f"{functor_types[key]}{{{', '.join([*names, 'latch_active_batch'])}}}"
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
        if dynamic is not None:
            prefetch = "\n    ".join(f"latch_dyn::epi_prefetch_rows<{_CTYPES[dtypes[ref]]}, BM, BN>({_ident(ref)} + base, ldc, "
                                       f"reinterpret_cast<{_CTYPES[dtypes[ref]]}*>(E + {offsets[ref]}), latch_rows);"
                                       for ref in vector_refs)
        ebytes = running
        body = ((f"const int rows = latch_rows; constexpr int cols = BN;\n    const int cg = cols >> 2, groups = rows * cg;\n"
                 if dynamic is not None else
                 f"constexpr int rows = BM, cols = BN;\n    const int cg = cols >> 2, groups = rows * cg;\n")
                + f"    for (int q0 = threadIdx.x; q0 < groups; q0 += {unroll} * (int)blockDim.x) {{\n"
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
        if dynamic is not None:
            # Masked staging (absent rows/K of an active extent load nothing and stage +0).
            issue = "\n    ".join(f"latch_dyn::sstage_edge<{types[ref]}, R, RK, KS>({_ident(ref)} + off, sr, sk, r0, k, "
                                  f"reinterpret_cast<{types[ref]}*>(src + {offsets[ref]}), rlim, klim);" for ref in staged)
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
        if dynamic is not None:
            # Absent elements of an active extent convert to +0 (never evaluated) and are never materialized.
            body_convert = (f"{pointers}\n"
                            "    const long long step = RK ? sk : sr;\n"
                            "    for (int c = threadIdx.x; c < R * KS / 2; c += NTHREADS) {\n"
                            "      const int r = RK ? c / (KS / 2) : (c % (R / 2)) * 2, kk = RK ? (c % (KS / 2)) * 2 : c / (R / 2);\n"
                            "      float v[2]; unsigned at[2]; bool ok[2];\n"
                            "#pragma unroll\n      for (int e = 0; e < 2; ++e) {\n"
                            "        const int er = RK ? r : r + e, ek = RK ? kk + e : kk;\n"
                            "        ok[e] = r0 + er < rlim && k + ek < klim;\n"
                            "        const unsigned i = (unsigned)(off + (long long)(r0 + er) * sr + (long long)(k + ek) * sk);\n"
                            f"        {positions}\n"
                            f"        at[e] = i; v[e] = ok[e] ? ({expression}) : 0.0f;\n"
                            "      }\n"
                            "      const uint32_t packed = s101h::bpk(v[0], v[1]);\n"
                            "      pops2b::put_pair<R, RK, KS>(dst, r, kk, packed);\n"
                            "      if (out != nullptr) {\n"
                            "        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&packed);\n"
                            "        if (step == 1 && ok[0] && ok[1]) *reinterpret_cast<uint32_t*>(out + at[0]) = packed;\n"
                            "        else { if (ok[0]) out[at[0]] = h[0]; if (ok[1]) out[at[1]] = h[1]; }\n"
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
        if dynamic is not None:
            # Masked interface of kernels/dynamic_flat.cuh (rlim/klim: existing rows and K of the operand).
            fields = fields + " int latch_active_batch;"
            key = ("staged_edge", fields, byte_sum, issue, convert)
            if key not in functor_types:
                type_name = "latch_fn_" + hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
                functor_types[key] = type_name
                helper_parts.append(
                    f"struct {type_name} {{\n  {fields}\n  static constexpr bool kStaged = true;\n"
                    f"  template <int R, bool RK, int KS> __host__ __device__ static constexpr int bytes() {{ return {byte_sum}; }}\n"
                    f"  template <int R, bool RK, int KS> __device__ __forceinline__ void issue_e(char* src, long long off, "
                    f"long long sr, long long sk, int r0, int k, int rlim, int klim) const {{\n    {issue}\n  }}\n"
                    f"  template <int R, bool RK, int KS> __device__ __forceinline__ void convert_e(const char* src, "
                    f"__nv_bfloat16* dst, __nv_bfloat16* out, long long off, long long sr, long long sk, int r0, int k, "
                    f"int rlim, int klim) const {{\n    {convert}\n  }}\n}};")
            return f"{functor_types[key]}{{{', '.join([*names, 'latch_active_batch'])}}}"
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

    phase_source, phase_records, pops2_contracts, bf16_contracts = [], [], [], []
    # LATCH-STEP-04: inactive-row optimizer split for fused-norm indexed gradients (see indexed_owner).
    inactive_split, loop_guards, split_report = {}, {}, []
    for kE, phE in enumerate(phases):
        for hname in phE["heavy"]:
            hextra = nodes[hname].extra or {}
            flags_name = hextra.get("row_active")
            written = hextra.get("phase_writes") or []
            if not flags_name or len(written) != 3 or written[2] != flags_name:
                continue
            g_name, norm_name = written[0], written[1]
            groot = pre_layout_root.get(g_name, layouts[g_name].root)
            g_numel, D = layouts[g_name].numel, layouts[g_name].shape[-1]
            def reads_gradient(names_):
                # Element reads of the gradient itself (the packed norm scalar is shared by every table's update).
                return any(root == groot and layout.numel == g_numel for m in names_
                           for root, layout, _, _ in loads_of[m])
            found = [(kA, numel, members) for kA in range(kE + 1, len(phases))
                     for numel, members in phases[kA]["loops"].items()
                     if numel == g_numel and reads_gradient(members)]
            if len(found) != 1:
                split_report.append({"engine": hname, "split": False, "reason": f"{len(found)} candidate loops"})
                continue
            kA, numel, group = found[0]
            # Only the dependency component reading this gradient (another table's update may share the loop).
            comps = [c for c in loop_components(group) if reads_gradient(c)]
            if len(comps) != 1:
                split_report.append({"engine": hname, "split": False, "reason": f"{len(comps)} reading components"})
                continue
            members = comps[0]
            member_set, ok, why = set(members), True, []
            for m in members:
                for root, layout, shape, idx in loads_of[m]:
                    identity = (idx == "i" and shape is not None and _numel(shape) == numel and layout.numel == numel
                                and layout.contiguous and layout.offset == 0)
                    if root == groot:
                        if not (identity or layout.numel == 1):          # gradient element or the norm scalar
                            ok = False; why.append(f"{m}: non-identity gradient read")
                    elif root in member_set:
                        if not identity:
                            ok = False; why.append(f"{m}: non-identity member read {root}")
                    elif root in nodes and root in phase_of and phase_of[root] >= kE:
                        ok = False; why.append(f"{m}: reads {root} produced in phase {phase_of[root]} >= {kE}")
                    elif layout.numel != 1 and not identity:
                        ok = False; why.append(f"{m}: cross-row read of {root}")
            stored_here = [m for m in members if m in stored]
            # Early stores must target committed state storage, never arena storage with a later lifetime.
            if not all(m in inplace_source for m in stored_here):
                ok = False; why.append("stored members not all committed in place")
            written_roots = {inplace_source.get(m, m) for m in stored_here}
            for k2 in range(kE, kA + 1):
                for other in list(phases[k2]["heavy"]) + [n for ms in phases[k2]["loops"].values() for n in ms]:
                    if other == hname or other in member_set:
                        continue
                    if any(root in written_roots for root, _, _, _ in loads_of.get(other, ())):
                        ok = False; why.append(f"{other} in phase {k2} reads old committed state")
            if not ok or not stored_here:
                split_report.append({"engine": hname, "split": False, "reason": why or ["no stored members"]})
                continue
            split_report.append({"engine": hname, "split": True, "update_phase": kA, "members": list(members)})
            inactive_split[hname] = dict(D=D, numel=numel, members=members, groot=groot, stored=stored_here)
            loop_guards.setdefault((kA, numel), []).append(dict(D=D, flags=_ident(flags_name), norm=_ident(norm_name),
                                                                members=set(members), split=inactive_split[hname]))

    def row_update_code(sp, *, zero_gradient, element, vector, prefix, valid):
        """Unchanged member expressions for one element group (4 consecutive elements when `vector`).

        zero_gradient: gradient elements and the indexed norm read as +0.0 (finite clip coefficient).
        Returns (loads, declarations, computes, stores); loads are predicated on `valid` and use
        float4 for identity reads of declared FP32 tensors when `vector`.
        """
        numel, members, groot = sp["numel"], sp["members"], sp["groot"]
        member_set, lanes = set(members), (4 if vector else 1)
        preload, pre_lines = {}, []

        def make_loader(q):
            def loader(ref, shape, idx):
                lay = layouts[ref]
                origin = pre_layout_root.get(ref, lay.root)
                identity = (idx == "i" and shape is not None and _numel(shape) == numel and lay.numel == numel
                            and lay.contiguous and lay.offset == 0)
                if zero_gradient and origin == groot:
                    return "0.0f"
                if identity and origin in member_set:
                    expression = f"{prefix}u_{_ident(origin)}_{q}"
                    return f"__bfloat162float({expression})" if dtypes[origin] == "bfloat16" else expression
                if identity and ref in declared and dtypes[ref] == "float32":
                    if ref not in preload:
                        reg = f"{prefix}l_{len(preload)}"
                        preload[ref] = reg
                        pre_lines.append(
                            f"const float4 {reg} = {valid} ? *reinterpret_cast<const float4*>({_ident(ref)} + {element}) "
                            f": make_float4(0.0f, 0.0f, 0.0f, 0.0f);" if vector else
                            f"const float {reg} = {valid} ? {_ident(ref)}[{element}] : 0.0f;")
                    return f"{preload[ref]}.{'xyzw'[q]}" if vector else preload[ref]
                return memory_loader(ref, shape, idx)
            return loader
        computes = []
        for q in range(lanes):
            loader = make_loader(q)
            body = []
            for name in members:
                node = nodes[name]
                expression = node.render("i", lambda arg, i2, node=node, loader=loader: operand(arg, node.shape, i2, loader))
                body.append(f"{prefix}u_{_ident(name)}_{q} = {ctype_round(expression, node.dtype)};")
            computes.append(f"if ({valid}) {{ const unsigned i = ({element}) + {q}u; " + " ".join(body) + " }")
        decls = " ".join(f"{_CTYPES[nodes[n].dtype]} {prefix}u_{_ident(n)}_{q};" for n in members for q in range(lanes))
        stores = []
        for n in sp["stored"]:
            if vector:
                stores.append(f"if ({valid}) *reinterpret_cast<float4*>({_ident(n)} + {element}) = make_float4("
                              + ", ".join(f"{prefix}u_{_ident(n)}_{q}" for q in range(4)) + ");")
            else:
                stores.append(f"if ({valid}) {_ident(n)}[{element}] = {prefix}u_{_ident(n)}_0;")
        return pre_lines, decls, computes, stores

    def inactive_rows_code(sp):
        """CTA-wide update of this CTA's owned zero-gradient rows (engine variables: L, jc, P, cta, start_)."""
        D = sp["D"]
        vector = D % 4 == 0 and all(dtypes[n] == "float32" for n in sp["members"])
        G = D // 4 if vector else D
        U = 4
        heads, loads, decls, computes, stores = [], [], [], [], []
        for u in range(U):
            heads.append(f"const int t{u}_ = tb_ + {u * 256}; const int j{u}_ = t{u}_ / {G}; "
                         f"const bool ok{u}_ = t{u}_ < L * {G} && start_[j{u}_ + 1] == start_[j{u}_]; "
                         f"const unsigned e{u}_ = ok{u}_ ? (unsigned)(((long long)cta + (long long)P * (jc + j{u}_)) * {D}LL) + "
                         f"(unsigned)(t{u}_ % {G}) * {4 if vector else 1}u : 0u;")
            pre, dec, comp, sto = row_update_code(sp, zero_gradient=True, element=f"e{u}_", vector=vector,
                                                  prefix=f"z{u}", valid=f"ok{u}_")
            loads += pre; decls.append(dec); computes += comp; stores += sto
        body = "\n    ".join(heads + loads + decls + computes + stores)
        return ("{ // LATCH-STEP-04: optimizer update of this CTA's zero-gradient rows (g = 0, finite clip coefficient)\n"
                f"  for (int tb_ = threadIdx.x; tb_ < L * {G}; tb_ += {256 * U}) {{\n    {body}\n  }}\n}}")

    def active_rows_code(sp, guard):
        """Warp-per-row update of rows with summed occurrences (every row when the clip norm is NaN)."""
        D = sp["D"]
        vector = D % 4 == 0 and all(dtypes[n] == "float32" for n in sp["members"])
        G = D // 4 if vector else D
        pre, dec, comp, sto = row_update_code(sp, zero_gradient=False, element="e_", vector=vector,
                                              prefix="a", valid="true")
        rows = sp["numel"] // D
        body = "\n      ".join(pre + [dec] + comp + sto)
        return ("{ // LATCH-STEP-04: optimizer update of rows with summed occurrences (all rows if the clip norm is NaN)\n"
                f"  const bool latch_rg_ = !({guard['norm']}[0] == {guard['norm']}[0]);\n"
                "  const int lane_ = threadIdx.x & 31, warps_ = gridDim.x * (blockDim.x >> 5);\n"
                f"  for (int row_ = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); row_ < {rows}; row_ += warps_) {{\n"
                f"    if (!latch_rg_ && {guard['flags']}[row_] == 0.0f) continue;\n"
                f"    for (int c_ = lane_; c_ < {G}; c_ += 32) {{\n"
                f"      const unsigned e_ = (unsigned)row_ * {D}u + (unsigned)c_ * {4 if vector else 1}u;\n"
                f"      {body}\n    }}\n  }}\n}}")

    for k, phase in enumerate(phases):
        parts, gemm_offset, rotate = [], 0, 0
        split_bindings = {}
        matrix_group = None
        idle_reductions = []
        if dynamic is not None and any(nodes[n].extra.get('matrix_spec') for n in phase['heavy']):
            idle_reductions = [n for n in phase['heavy'] if not nodes[n].extra.get('matrix_spec')]
            phase = {**phase, 'heavy': [n for n in phase['heavy'] if n not in idle_reductions]}
        group_specs = [nodes[name].extra.get('matrix_spec') for name in phase['heavy']]
        phase_epilogues = phase.get('epilogue', {})
        if group_specs and all(group_specs) and (phase_epilogues or pops2b_plans
                                                 or any(n in absorbed_by for n in phase['heavy'])):
            from .pops2b_rules import HEADER as EPI_HEADER, emit_group as emit_epilogue_group
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
            from .matrix_rules import emit_group
            code, matrix_group = emit_group(group_specs, ident=_ident)
            parts.append(code)
            gemm_offset = matrix_group['tiles']
        else:
            fused_rows = {}
            if FUSE_FULL_REDUCTION_LOOPS:
                for name in phase["heavy"]:
                    node = nodes[name]
                    if not (node.extra and node.extra.get("engine") == "multi_cta_full"
                            and "/*FUSED_ROW_BODY*/" in node.code):
                        continue
                    extent = int(node.extra["reduction_elements"])
                    members = phase["loops"].get(extent)
                    if (not members or extent in phase.get("_fused_loops", ()) or loop_guards.get((k, extent))
                            or extent >= (1 << 31)
                            or not all(nodes[n].kind == "elementwise" and nodes[n].dtype in ("float32", "bfloat16")
                                       for n in members)):
                        continue
                    # No member may write storage the reduction reads (stored roots and in-place commit targets).
                    written = ({layouts[n].root for n in members if n in stored}
                               | {inplace_source[n] for n in members if n in inplace_source})
                    aliases = {_ident(x) for x in layouts if layouts[x].root in written} | {_ident(x) for x in written}
                    if aliases & set(re.findall(r"\bv_\w+\b", heavy_code(node))):
                        continue
                    fused_rows[name] = (extent, list(members))
                    phase.setdefault("_fused_loops", []).append(extent)
                    fused_row_report.append({"phase": k, "reduction": name, "elements": extent,
                                             "members": list(members)})
            for position, name in enumerate(phase["heavy"]):
                node = nodes[name]
                code = heavy_code(node)
                pz = (node.extra or {}).get("prezero")
                if pz:
                    if name in prezeroed:
                        head, rest = code.split(pz["begin"], 1)
                        code = head + rest.split(pz["end"], 1)[1]
                    code = code.replace(pz["begin"], "").replace(pz["end"], "")
                red_members = phase.get("red_epilogue", {}).get(name)
                if red_members:
                    if "/*RED_EPILOGUE*/" not in code:
                        raise LoweringError(f"internal: reduction epilogue site missing in {name}")
                    hoist_r, body_r = fused_row_code(_numel(nodes[red_members[0]].shape), list(red_members), index="o")
                    code = code.replace("/*RED_EPILOGUE*/", "{ " + hoist_r.replace("\n", " ") + body_r + " }")
                code = code.replace("/*RED_EPILOGUE*/", "")
                if name in fused_rows:
                    hoist, body = fused_row_code(*fused_rows[name])
                    code = code.replace("/*FUSED_ROW_HOIST*/", hoist).replace("/*FUSED_ROW_BODY*/", body)
                code = code.replace("/*FUSED_ROW_HOIST*/\n", "").replace("/*FUSED_ROW_BODY*/", "")
                if name in inactive_split:
                    code = code.replace("/*LATCH_INACTIVE_ROWS_HOOK*/", inactive_rows_code(inactive_split[name]))
                if name in deferred:
                    info = deferred[name]
                    spec = node.extra["matrix_spec"]
                    if name in phase.get("epilogue", {}):
                        from .pops2b_rules import HEADER as EPI_HEADER, _code as epi_matrix_code, tile_plan
                        if EPI_HEADER not in extra_headers:
                            extra_headers.append(EPI_HEADER)
                            helper_parts.append(f'#include "{EPI_HEADER}"')
                        extra = {"epi": epilogue_code(name, phase["epilogue"][name])}
                        code = epi_matrix_code(spec, tile_plan(spec, info["plan_grid"], extra), _ident,
                                               *info["slice"], 0, extra)
                    else:
                        from .matrix_rules import _code as matrix_code, _plan as matrix_plan
                        code = matrix_code(spec, matrix_plan(spec, info["plan_grid"]), _ident, *info["slice"], 0)
                if position:
                    parts.append("__syncthreads();  // shared staging reuse between co-scheduled heavy items")
                split_need = node.extra.get('p2_split') if node.extra else None
                if split_need:
                    if '/*P2_CNT*/' not in code or '/*P2_PART*/' not in code:
                        raise LoweringError("internal: split product without scratch placeholders")
                    p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                    p2_part_cursor = phase.setdefault('_p2_part', 0)
                    split_bindings[name] = {**split_need, 'counter_word_offset': p2_cnt_cursor,
                                            'partial_word_offset': p2_part_cursor}
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
        if dynamic is not None and matrix_group is not None and any(spec.get("active") for spec in group_specs):
            for header in ("pops2b_mm.cuh", "dynamic_flat.cuh"):
                if header not in extra_headers:
                    extra_headers.append(header)
                    helper_parts.append(f'#include "{header}"')
            # Split-K members: bind each product's counters/partials in the split scratch of this phase.
            for member in matrix_group["members"]:
                plan_ = member["matrix_plan"]
                if not plan_.get("split"):
                    continue
                tag = member["output"]
                need = {"counter_words": plan_["tiles"] // plan_["split"],
                        "partial_words": plan_["tiles"] * plan_["bm"] * plan_["bn"]}
                p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                p2_part_cursor = phase.setdefault('_p2_part', 0)
                split_bindings[tag] = {**need, 'counter_word_offset': p2_cnt_cursor, 'partial_word_offset': p2_part_cursor}
                parts = [part.replace(f'/*P2_CNT:{tag}*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + p2_cnt_cursor}LL)')
                             .replace(f'/*P2_PART:{tag}*/', f'(arena + /*P2_PARTBASE*/ + {p2_part_cursor}LL)')
                         for part in parts]
                phase['_p2_cnt'] = p2_cnt_cursor + need['counter_words']
                phase['_p2_part'] = p2_part_cursor + need['partial_words']
                p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
                p2_partial_words = max(p2_partial_words, phase['_p2_part'])
                if p2_first_split_phase is None:
                    p2_first_split_phase = k
                member["split_scratch"] = split_bindings[tag]
        matrix_only = phase["heavy"] and all(nodes[n].kind == "gemm" for n in phase["heavy"])
        exclude = gemm_offset if matrix_only else 0
        if idle_reductions:
            # Reductions take the CTAs after the matrix tasks ([gemm_offset, grid)); loops follow them.
            from .dynamic_flat import idle_heavy
            idle_items = []
            for n in idle_reductions:
                code = heavy_code(nodes[n])
                split_need = nodes[n].extra.get('p2_split')
                if split_need:
                    # The static schedule's split scratch protocol (counters zeroed at entry, restored by the
                    # last arriver; partials per phase), bound for a reduction placed on idle CTAs.
                    p2_cnt_cursor = phase.setdefault('_p2_cnt', 0)
                    p2_part_cursor = phase.setdefault('_p2_part', 0)
                    split_bindings[n] = {**split_need, 'counter_word_offset': p2_cnt_cursor,
                                         'partial_word_offset': p2_part_cursor}
                    code = code.replace('/*P2_CNT*/', f'reinterpret_cast<unsigned*>(arena + {p2_base + p2_cnt_cursor}LL)')
                    code = code.replace('/*P2_PART*/', f'(arena + /*P2_PARTBASE*/ + {p2_part_cursor}LL)')
                    phase['_p2_cnt'] = p2_cnt_cursor + split_need['counter_words']
                    phase['_p2_part'] = p2_part_cursor + split_need['partial_words']
                    p2_counter_words = max(p2_counter_words, phase['_p2_cnt'])
                    p2_partial_words = max(p2_partial_words, phase['_p2_part'])
                    if p2_first_split_phase is None:
                        p2_first_split_phase = k
                idle_items.append((n, code, int(nodes[n].extra.get('work_items', 1))))
            parts.append(idle_heavy(idle_items, gemm_offset))
            exclude = gemm_offset + sum(int(nodes[n].extra.get('work_items', 1)) for n in idle_reductions)
            phase = {**phase, 'heavy': [*phase['heavy'], *idle_reductions]}
        main_heavy = [n for n in phase["heavy"] if n not in deferred]
        if (pops2b and pops2b_idle and target_arch == "sm_90" and main_heavy and not matrix_only
                and all(nodes[n].kind in IDLE_HEAVY_KINDS and nodes[n].extra.get("work_items") for n in main_heavy)):
            exclude = sum(int(nodes[n].extra["work_items"]) for n in main_heavy)
        work = []
        if loop_partition and not exclude and not phase["heavy"]:
            for numel, members in phase["loops"].items():
                work += [(numel, comp) for comp in loop_components(members)]
            if len(work) < 2 or sum(n for n, _ in work) < LOOP_PARTITION_MIN_ELEMENTS:
                work = []
        if work:
            total, cum = sum(n for n, _ in work), 0
            for numel, comp in work:
                parts.append(loop_code(numel, comp, 0, cta_range=(cum, cum + numel, total)))
                cum += numel
            partitioned_phases.append({"phase": k, "items": len(work), "elements": total})
        else:
            for numel, members in phase["loops"].items():
                if numel in phase.get("_fused_loops", ()):
                    continue
                comps = loop_components(members) if (loop_split and len(members) > 8) else [members]
                guards = loop_guards.get((k, numel))
                if guards:
                    comps = loop_components(members)          # guarded components become their own loops
                for comp in comps:
                    guard = next((g for g in guards or () if g["members"] == set(comp)), None)
                    if guard is not None:
                        parts.append(active_rows_code(guard["split"], guard))
                    else:
                        parts.append(loop_code(numel, comp, exclude))
        for pz in prezero_jobs.get(k, ()):
            out_ = _ident(pz["value"])
            parts.append(f"{{ // LATCH-STEP-04: zero {pz['value']} for the next phase's indexed atomic accumulation\n"
                         f"  const long long tz_ = (long long)blockIdx.x * blockDim.x + threadIdx.x, gz_ = (long long)gridDim.x * blockDim.x;\n"
                         f"  if ((((unsigned long long)({out_})) & 15ULL) == 0ULL && {pz['elements']}LL % 4LL == 0LL) {{\n"
                         f"    float4* z_ = reinterpret_cast<float4*>({out_});\n"
                         f"    for (long long e = tz_; e < {pz['elements'] // 4}LL; e += gz_) z_[e] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
                         f"  }} else {{\n"
                         f"    for (long long e = tz_; e < {pz['elements']}LL; e += gz_) {out_}[e] = 0.0f;\n"
                         f"  }}\n}}")
        heavy_nodes = [nodes[n] for n in phase["heavy"]]
        loop_names = [n for members in phase["loops"].values() for n in members]
        epilogue_names = [n for members in phase.get("epilogue", {}).values() for n in members]
        epilogue_names += [n for members in phase.get("red_epilogue", {}).values() for n in members]
        epilogue_names += [cast for name in phase["heavy"] for cast in absorbed_by.get(name, {}).values()
                           if absorbed[cast]["mode"] == "materialize"]
        kind = (next((h.kind for h in heavy_nodes if h.kind != "gemm"), None)
                or (heavy_nodes[0].kind if heavy_nodes else "elementwise"))
        all_names = [*phase["heavy"], *epilogue_names, *loop_names]
        reads = []
        for name in all_names:
            for root, layout, shape, idx in loads_of.get(name, ()):
                if epilogue_of.get(name) == root:
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
                                if key in h.extra},
                             # LATCH-STEP-04: per-body reduction contract, so stacked reductions can each be proven.
                             **({key: h.extra[key] for key in ("engine", "arithmetic", "reduction_elements")
                                 if key in h.extra} if h.kind == "reduction" else {})}
                            for h in heavy_nodes]
                           + [{"kind": "epilogue", "product": product, "names": list(members),
                               "product_store_elided": product in elided_products}
                              for product, members in phase.get("epilogue", {}).items()]
                           + [{"kind": "operand_cast", "name": cast, "product": name, "operand": which,
                               "mode": absorbed[cast]["mode"]}
                              for name in phase["heavy"] for which, cast in absorbed_by.get(name, {}).items()]
                           + [{"kind": "elementwise", "elements": numel, "names": members}
                              for numel, members in phase["loops"].items()]}
        if any(h.extra.get("engine") == "pops2_fp32" for h in heavy_nodes):
            # Declared emitted-body/scratch evidence for compatible schedule
            # retention. Aggregate phase metadata describes only its first
            # heavy item; it cannot prove a co-scheduled product's contract.
            # These records do not participate in code generation or loading,
            # and are not serialized into the historical runtime manifest.
            pops2_contracts.append({"phase": k, "split_bindings": split_bindings,
                "items": [{"name": h.name, "operand_roots": list(h.direct), "layout": pops2_layouts[h.name], "body": {
                    key: h.extra[key] for key in ("engine", "body", "matrix_shape", "matrix_plan", "arithmetic",
                                                 "transposed_product", "requires_full_grid", "shared_bytes")}}
                          for h in heavy_nodes if h.extra.get("engine") == "pops2_fp32"]})
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
            # Keep the exact spec and member returned by the actual group
            # emitter. Private selection evidence never changes saved runtime
            # manifests or the generated calculation.
            bf16_contracts.extend({"name": name, "phase": k, "spec": dict(spec), "member": dict(member)}
                for name, spec, member in zip(phase["heavy"], group_specs, matrix_group["members"])
                if spec.get("dtype") == "bfloat16")
        phase_records.append(record)
        phase_source.append(f"// Phase {k}: {', '.join(all_names)}\n{{\n" + scoped("\n".join(parts))
                            + f"\n}}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{k+1}] = stamp();")
    prefetches = []
    if pops2b and pops2b_prefetch and not retain_all and target_arch == "sm_90":
        from .pops2b_rules import prefetch_plan
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
    active_prologue = ""
    if dynamic is not None:
        if dynamic.input not in inputs or dtypes[dynamic.input] != "int64":
            raise LoweringError("bounded dynamic batch scalar must be an int64 program input")
        # Every CTA reads the staged scalar once; an out-of-domain value exits uniformly before any barrier.
        active_prologue = (f"\n  const int latch_active_batch = (int)reinterpret_cast<const int64_t*>(args[{inputs.index(dynamic.input)}])[0];"
                           f"\n  if (latch_active_batch < 1 || latch_active_batch > {dynamic.maximum}) return;")
    kernel_body = ("  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n" + "\n".join(declarations)
                   + active_prologue + p2_prologue + barrier_setup
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
    helpers = "\n".join(helper_parts)
    shared_bytes = max([17408] + [p.get("shared_bytes", 0) for p in phase_records])
    from .resources import select_program_resources
    resource_policy = select_program_resources(shared_bytes, phases=phase_records, arch=target_arch,
                                               pops2_two_cta=matrix_residency == 2)
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    index_checks = "assert"
    if TRAP_INDEX_CHECKS and "assert(" in kernel_body and "#define assert(" not in source:
        from .p6e_rules import TRAP_ASSERT
        entry = 'extern "C" __global__'
        if source.count(entry) == 1:
            source = source.replace(entry, TRAP_ASSERT + entry, 1)
            index_checks = "trap (assert overridden)"
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
        **({"scalar_hoist": hoisted} if hoisted else {}),
        **({"reduction_stack": stacked_moves} if stacked_moves else {}),
        **({"loop_partition": {"phases": partitioned_phases, "rule": "P9-D independent loop components on "
                                                                    "disjoint CTA ranges sized by element count"}}
           if partitioned_phases else {}),
        "matrix_decisions": matrix_decisions,
        **({"pops2b": {"matrix_epilogues": {g: list(m) for g, m in epilogues.items()},
                       "elided_product_stores": sorted(elided_products),
                       "operand_casts": absorbed, "l2_prefetch": prefetches, "deferred_products": deferred,
                       "rules": "attach_epilogues: identical per-operation expressions on rounded BF16 results; "
                                "select_absorbed_casts: BF16 casts materialized by their first product's staging"}}
           if pops2b and (epilogues or absorbed or deferred) else {}),
        "attention_groups": attention_decisions, "attention_refusals": attention_refusals,
        "ordered_index_gradients": index_decisions,
        "indexed_optimizer_split": split_report,
        "fused_reduction_loops": fused_row_report, "index_checks": index_checks,
        "indexed_prezero": sorted(prezeroed),
        # grid barriers inside bodies (atomic_v1 zero/norm stages; a prezeroed body drops its zero stage)
        "internal_grid_barriers": sum(int((nodes[n].extra or {}).get("internal_grid_barriers", 0))
                                      - (1 if n in prezeroed else 0) for ph in phases for n in ph["heavy"]),
        "reduction_epilogues": {e: r for e, r in sorted(red_epi_of.items())},
        "compiler_policy": "integrated_v1", "target_arch": target_arch,
        "phase_count": commit + 1, "barrier_count": commit + (1 if commit_code else 0),
        "max_gemm_tiles": max([p.get("tiles", 0) for p in phase_records] or [1]),
        "shared_bytes": shared_bytes,
        **({"resource_policy": resource_policy} if resource_policy['has_popsb'] or resource_policy.get('pops2_two_cta') else {}),
        **({"grid_preference": "power_of_two_occupancy"}
           if INDEXED_GRID_POW2 and index_decisions and int(resource_policy.get('resident_blocks_per_sm', 1)) == 1
           and not any(p.get("kind") == "gemm" for p in phase_records) else {}),
        "numerical_policy": ("Explicit BF16 casts/operands/results; BF16 MMA with FP32 accumulation; FP32 master state; fixed reduction order; no TF32/data atomics"
                             if "bfloat16" in dtypes.values() else
                             "FP32 SIMT fmaf; admitted ordered split-K; separately rounded pointwise arithmetic; exact reduction tree; no TF32/data atomics"),
        "fusion_policy": "identical per-operation expressions; BF16 rounding retained when a buffer is elided; inline limit "
                         f"{inline_total}/{inline_work}; loads before stores; unroll <= {UNROLL_MAX}",
        "execution": f"one cooperative CUDA launch, 256 threads/CTA, {shared_bytes} shared bytes/CTA; occupancy admitted; in-place or final state commit",
        "memory_plan": "pinned plan.assign + independent plan.check; inclusive phase intervals and alias roots; elided values have no storage",
    }
    if dynamic is not None:
        dynamic.manifest(manifest, layouts, phase_records)
    program = GeneratedProgram(source, manifest, kernel_body, helpers)
    if pops2_contracts:
        program._selection_contracts = {"pops2": pops2_contracts, "matrix_scratch_base": p2_base}
    if bf16_contracts:
        cast_contracts, add_contracts = [], []
        for node in nodes.values():
            if (node.kind == "elementwise" and node.base == "add" and len(node.loads) == 2
                    and all(dtypes[operand] == "float32" for operand in [*node.loads, node.name])):
                add_contracts.append({"name": node.name, "inputs": list(node.loads),
                    "input_layouts": [{**layouts[operand].to_dict(), "dtype": dtypes[operand]} for operand in node.loads],
                    "output_layout": {**layouts[node.name].to_dict(), "dtype": node.dtype},
                    "expression": node.render("i", lambda arg, idx: "operand" + str(node.loads.index(_ref(arg, names))))})
            if (node.kind != "elementwise" or node.base != "_to_copy" or len(node.loads) != 1
                    or "bfloat16" not in (node.dtype, dtypes[node.loads[0]])):
                continue
            operand = node.loads[0]
            probe = "__bfloat162float(operand)" if dtypes[operand] == "bfloat16" else "operand"
            cast_contracts.append({"name": node.name, "input": operand,
                "input_layout": {**layouts[operand].to_dict(), "dtype": dtypes[operand]},
                "output_layout": {**layouts[node.name].to_dict(), "dtype": node.dtype},
                "expression": ctype_round(node.render("i", lambda arg, idx: probe), node.dtype)})
        program._selection_contracts = {**getattr(program, "_selection_contracts", {}),
                                       "bf16": bf16_contracts, "bf16_casts": cast_contracts, "bf16_adds": add_contracts}
    return program
