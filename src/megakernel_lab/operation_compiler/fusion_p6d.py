# Integrated from frozen donor 81a52fc02783e0c537dcf1a909083916f2deaa66; private scheduling/resource protocol.
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

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from . import attention_lower

import hashlib
import json
import math
import re

from .vendor import plan as arena_plan
from . import arena_assign  # LATCH-BUILD-600: same offsets as arena_plan.assign, without its quadratic scan

UNROLL_MAX = 8
P9_LIGHT_KINDS = ("embedding", "embedding_gradient_rank", "scatter_add", "attention_long_prepare", "gather", "log_softmax", "embedding_gradient")
P9_DIAG_EXTRA_SYNCS = 0          # P9-A timing diagnostic: extra grid barriers in phase 1
P9_LOCAL_PTRS = True             # P9-A: every phase re-derives its tensor pointers from opaque arena/args copies
P9_SREG_MACRO = True             # P9-A: volatile special-register reads everywhere after the vendor headers
P9_OPAQUE_SREG = True            # P9-A: per-phase opaque copies of threadIdx/blockIdx/blockDim/gridDim .x
P9_LOCAL_SKIP_GEMM = False       # P9-A: product phases keep kernel-scope pointers (other phases re-derive)
P9_LATE_LOOPS = True             # P9-A: lone loop-only phases join the next phase when all readers are later
P9_LATE_LOOPS_DEBUG = False
P9_LIGHT_LATE = True             # P9-A: lone light items join the latest light phase before their first reader
P9_DEFER_STATE_PRODUCTS = False  # P9-A: optimizer-state products before a reduction phase move to the next product phase
P9_DIAG_NO_STATE_STORES = False  # P9-A timing-only diagnostic: product epilogues skip optimizer-state stores (wrong results)
# LATCH-DENSE-EXEC: long single-output FP32 sums (cross-entropy mean, global-norm terms) use the existing multi-CTA
# full-reduction body (the common planner's multi_cta_full) instead of one CTA. False reproduces the one-CTA tree.
P6D_FULL_REDUCTION = True
FULL_SIMPLE = True         # LATCH-FFN-RETAIN: merged-stride leaf position + unrolled chunk loop (exact)
FULL_UNROLL = 8
# LATCH-DENSE-EXEC: FP32 products of exact BF16 -> FP32 operand widenings (e.g. the opmath decomposition of a BF16
# linear with bias) run on p5c from the BF16 sources with an unrounded FP32 result. False keeps the FP32 bodies.
P6D_EXACT_BF16_PRODUCTS = True
FULL_REDUCTION_MIN = 64 * 256          # as fusion_reference / the common planner (C-OPS4-B)
P8_STREAM_EPILOGUE = False    # P8-A diagnostic: epilogue operand loads/stores with evict-first (.cs) cache hints
P8_ROW_HOIST = True            # (LATCH-FFN-RETAIN: default on; exact) P8-A: row-consumer row constants / column-broadcast quads loaded once (exact)
P8_PERIODIC_PREFETCH = True    # P8-A: x[i % K] (K % 8 == 0) epilogue operands join the chunk prefetch (exact)
RESIDENT_THREADS = 128 * 256          # default H100 launch cap x CTA width
TRANSCENDENTAL = {"tanh", "exp", "expm1", "log1p", "reciprocal", "log", "sqrt", "rsqrt", "sigmoid", "silu", "silu_backward",
                  "pow", "div", "sigmoid_backward", "tanh_backward"}
INLINE_TOTAL_LIMIT = 12                # recomputed operations per inlined value use
INLINE_WORK_LIMIT = 24                 # summed over all uses
LIVE_VALUE_CAP = 64                    # unroll x loop values held between loads and stores
VECTOR_MIN_ELEMENTS = 65536            # x4 vector loops only where each thread gets several groups
def _balanced_parens(text):
    depth = 0
    for ch in text:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0


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
                    forced, numel):
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
    load = {p: 0.0 for p in range(len(phases))}
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


def _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, error, epilogue_of=None):
    """Independent re-check of the schedule (the scheduler is not trusted).

    Every read is of a graph input, a value from an earlier phase that has
    storage, or a same-loop register at the identical flat index. Each in-place
    target is read only before its committing phase, or at the committing
    loop's own index (all loads of an unrolled group precede its stores).
    """
    def loop_key(name):
        node = nodes[name]
        # Equal element counts do not imply equal CTA ownership: attached row
        # consumers and matrix epilogues execute inside their owning body's
        # domain, independently of a standalone global-stride loop.
        return (phase_of[name], math.prod(node.shape), (epilogue_of or {}).get(name)) if node.kind == "elementwise" else None
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
            if epilogue_of and epilogue_of.get(node.name) == root and q == p:
                continue                    # p5c matrix epilogue reads the product's rounded result in registers
            if q < p:
                if nodes[root].kind == "elementwise" and root not in stored:
                    raise error(f"schedule check: {node.name} reads {root} from memory but it has no storage")
                continue
            if not (q == p and shape is not None and register_load(node, root, layout, shape, idx)
                    and loop_key(root) == loop_key(node.name)):
                raise error(f"schedule check: {node.name} (phase {p}) reads {root} produced in phase {q}")


@attention_lower.retry_lowering
def p9_rowblock_mod():
    from . import p9_rowblock
    return p9_rowblock


def lower_fused(graph, *, retain_all=False, coschedule_matrix=True, place_by_slack=True,
                matrix_policy="auto", split_k=True, exact_reductions=True, fuse_attention=True,
                ordered_index_gradients=True, target_arch="sm_90", pops4d=True, pops4d_recipe=None,
                pops4d_min_sequence=None, pops4d_defines=None, pops4d_version=None, pops4d_diag=None,
                p8_precision=None):
    # P8-A declared precision policy: None/'strict' keeps the recipe; 'practical' evaluates the sigmoid family
    # (sigmoid, silu, silu_backward) with approximate exponential/division intrinsics and selects the BF16-probability
    # attention recipe. Explicit option only; never chosen automatically.
    if p8_precision not in (None, 'strict', 'practical'):
        raise ValueError('p8_precision must be strict or practical')
    practical = p8_precision == 'practical'
    if practical and pops4d_recipe is None:
        pops4d_recipe = 'bf16'
    # P-OPS4-D (78201354e, ported by P6-D): BF16-boundary training attention (ops/long_attention.py) for matched
    # groups with head width 64 and 256 <= T <= 4096; pops4d=False reproduces the P-OPS5-C lowering byte for byte.
    from .lower import (GeneratedProgram, LoweringError, TensorLayout, VENDOR, _header_path, _BYTES, _CTYPES,
                        _contiguous, _dtype, _field, _flatten_refs, _float, _double, _ident, _numel,
                        _ref, _reshape_strides, _translation_unit)

    raw_values = _field(graph, "values")
    values = raw_values if isinstance(raw_values, dict) else {_field(v, "name"): v for v in raw_values}
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

    # P-OPS5-C: readers of every value (direct references), for the exact product-conversion rule below.
    value_readers = {}
    for op in operations:
        for ref in dict.fromkeys(_flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)):
            value_readers.setdefault(ref, []).append(op)
    converted_products = {}   # FP32 conversion value -> BF16 product whose matrix phase writes it directly

    nodes = {}            # root name -> _Node, in operation order
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
    from . import p6d_rules
    powf_nodes = set()
    long_plans, long_static = {}, {}
    if pops4d and attention_groups and target_arch == "sm_90":
        from . import flat_long_attention as long_attention
        pops4d_recipe = long_attention.DEFAULT_RECIPE if pops4d_recipe is None else pops4d_recipe
        if pops4d_recipe not in long_attention.RECIPES:
            raise LoweringError(f"unknown P-OPS4-D attention recipe {pops4d_recipe!r}")
        for value_name, value in values.items():
            value_shape = tuple(int(x) for x in _field(value, "shape"))
            long_static[value_name] = TensorLayout(value_shape, tuple(
                int(x) for x in _field(value, "strides", _field(value, "stride", _contiguous(value_shape)))), value_name)
        long_plans, long_refused = long_attention.admit(
            attention_groups, (operations, values, names, _ref, _field, output_name), long_static,
            set(outputs) | set(writes.values()),
            min_sequence=long_attention.MIN_SEQUENCE if pops4d_min_sequence is None else int(pops4d_min_sequence))
        attention_refusals.extend(long_refused)
        attention_groups = [group for group in attention_groups if group["label"] not in long_plans]
    long_members = {name: plan for plan in long_plans.values() for name in plan["skip"]}
    long_mask_owner = {}  # (mask, head batch, context, mask strides) -> mask-statistics table names
    long_versions = set()
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
        attention_lower.add_fused_nodes(kind, code, metadata, reads, phase_writes,
            layouts=layouts, dtypes=dtypes, make_layout=TensorLayout, add_heavy=add_heavy,
            add_view=attention_view, reserved_names=names)

    def long_install(plan, name):
        """P-OPS4-D: install prepare+forward (at O's product) or dQ+dK/dV (at the first gradient product)."""
        from . import flat_long_attention as long_attention
        group = plan["group"]
        if name not in (plan["anchor_forward"], plan["anchor_backward"]):
            return
        forward = name == plan["anchor_forward"]
        used = ("q", "k", "v") if forward else ("q", "k", "v", "do")
        for source in (plan["sources"][key] for key in used):
            actual, static = layouts[source], long_static[source]
            if (actual.shape, actual.strides, actual.root, actual.offset) != (static.shape, static.strides, source, 0):
                raise attention_lower.AttentionRefused(group["label"], f"{source}: lowered BF16 source layout differs")
        mask = None
        if group.get("mask") is not None:
            from . import padding_masks
            mask = padding_masks.mask_strides(group, layouts, plan["bh"], plan["t"], attention_lower._mask_strides)
            if mask is None or dtypes[group["mask"]] != "float32":
                raise attention_lower.AttentionRefused(group["label"], "mask is not an FP32 broadcastable tensor")
        include = "".join(f"#define {key} {value}\n" for key, value in sorted((pops4d_defines or {}).items()))
        version = long_attention.DEFAULT_VERSION if pops4d_version is None else int(pops4d_version)
        if version in long_attention.BF16_ONLY_VERSIONS and long_attention.RECIPES[pops4d_recipe][0] != 1:
            version = 7  # stored dS / fixed-point dQ are BF16-recipe schedules; exact recipes keep v7's backward
        long_versions.add(version)
        include += long_attention.defines(pops4d_recipe, version) + f'#include "{long_attention.HEADER}"\n'
        if include not in helper_parts:
            helper_parts.append(include)
        if long_attention.HEADER not in extra_headers:
            extra_headers.append(long_attention.HEADER)
        if forward and mask is not None:
            key = (group["mask"], plan["bh"], plan["t"], tuple(mask), getattr(mask, "heads", 1))
            if key in long_mask_owner:
                plan["mask_tables"] = long_mask_owner[key]
            else:
                long_mask_owner[key] = {k: v for k, v in long_attention.aux_names(group).items()
                                        if k in ("mmax", "mzero", "rm")}
        # A matching layout alone does not prove that any phase writes the key-norm table.
        # Only a stored, plain flat WS product has the exact register-epilogue writer.
        from . import p5c_matrix
        k_node = nodes.get(plan["sources"]["k"])
        k_spec = (k_node.extra or {}).get("matrix_spec") if k_node is not None else None
        keynorm_epilogue = bool(k_spec and k_spec.get("engine") == "p5c" and not k_spec.get("epi")
                               and not k_spec.get("out_f32") and not k_spec.get("out_acc")
                               and p5c_matrix.P8WS_GROUP
                               and p5c_matrix.flat_ws_plan(k_spec, 128) is not None)
        for which in (("prepare", "forward") if forward else long_attention.backward_phases(version)):
            emitted = long_attention.emit(plan, _ident, mask, which=which, recipe=pops4d_recipe,
                mask_tables=plan.get("mask_tables"), diag=pops4d_diag, version=version,
                keynorm_epilogue=keynorm_epilogue)
            if emitted is None:
                continue                      # nothing to prepare (mask tables owned by an earlier group)
            kind, code, extra = emitted
            for value_name, spec in extra["outputs"].items():
                if spec is None:
                    value_shape, value_dtype = tuple(int(x) for x in _field(values[value_name], "shape")), "bfloat16"
                else:
                    value_shape, value_dtype = tuple(spec[0]), spec[1]
                    if value_name in names or value_name in layouts:
                        raise attention_lower.AttentionRefused(group["label"], f"synthetic name collision {value_name}")
                layouts[value_name] = TensorLayout(value_shape, _contiguous(value_shape), value_name)
                dtypes[value_name] = value_dtype
            metadata = {key: value for key, value in extra.items() if key not in ("outputs", "reads")}
            metadata.update(group=group["label"], attention_output=group["o"], source_commit="P-OPS4-D",
                            header=long_attention.HEADER, threads=256)
            attention_lower.add_fused_nodes(kind, code, metadata, extra["reads"], list(extra["outputs"]),
                layouts=layouts, dtypes=dtypes, make_layout=TensorLayout, add_heavy=add_heavy,
                add_view=attention_view, reserved_names=names)
        if forward:
            attention_decisions.append({"group": group["label"], "decision": "long_bf16_boundary_attention",
                "recipe": pops4d_recipe, "version": version, "head_width": plan["hd"], "sequence": plan["t"],
                "head_batch": plan["bh"], "heads": plan["heads"], "scale": group["scale"],
                "scale_mode": group["scale_mode"], "mask": group["mask"], "sources": plan["sources"],
                "outputs": plan["finals"], "removed": sorted(plan["removed"]), "source_commit": "P-OPS4-D"})

    producer_of = {output_name(op): op for op in operations}

    def exact_bf16_operand(layout):
        """(BF16 source, layout of the same view over it) when ``layout`` views an exact widening of a BF16 value."""
        widen = producer_of.get(layout.root)
        if widen is None or str(_field(widen, "target", "")).removeprefix("aten.").split(".")[0] != "_to_copy":
            return None
        wargs, wkwargs = list(_field(widen, "args", [])), dict(_field(widen, "kwargs", {}))
        source = _ref(wargs[0], names) if wargs else None
        if (source is None or dtypes.get(source) != "bfloat16" or dtypes.get(layout.root) != "float32"
                or set(wkwargs) - {"dtype", "layout", "device", "memory_format"}
                or str(wkwargs.get("memory_format", "torch.preserve_format")) not in ("torch.preserve_format", "None")):
            return None
        whole, src = layouts[layout.root], layouts[source]
        if (whole.shape != src.shape or not whole.contiguous or whole.offset
                or src.root in undefined or source in undefined):
            return None
        if src.contiguous:
            return source, TensorLayout(layout.shape, layout.strides, src.root, layout.offset + src.offset)
        # LATCH-RESEARCH-EDITS: a transposed/permuted BF16 source (shared weight read through .t()); the same rule
        # as the common planner (lower._view_through_source) keeps both candidates on the same P5C body.
        from .lower import EXACT_VIEW_SOURCES, _view_through_source
        through = _view_through_source(layout, whole, src) if EXACT_VIEW_SOURCES else None
        return None if through is None else (source, through)

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
        if name in converted_products:
            continue          # written by its product's FP32-converting matrix store (P-OPS5-C)
        if name in long_members:
            long_install(long_members[name], name)
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
        from . import p9_rowblock as _p9
        emitted = emit_operation(target, args, kwargs, name, shape, layouts, dtypes, names, placeholder_read,
                                 warp_rows=_p9.WARP_SOFTMAX, inputs=tuple(inputs))
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
            exact = (P6D_EXACT_BF16_PRODUCTS and matrix_policy != "frozen" and base in {"mm", "matmul"}
                     and target_arch == "sm_90" and dtypes[a] == dtypes[b] == dtypes[name] == "float32")
            sources = (exact_bf16_operand(la), exact_bf16_operand(lb)) if exact else (None, None)
            if all(sources):
                # LATCH-DENSE-EXEC: an FP32 product whose operands are (views of) exact BF16 -> FP32 widenings, e.g.
                # PyTorch's opmath decomposition of a BF16 addmm/linear with bias. Every product term is exact in FP32,
                # so the warpgroup-MMA body reading the BF16 sources with FP32 accumulation and an unrounded FP32 result
                # computes the same FP32 product in a different (fixed) accumulation order: the arithmetic class of the
                # strict exact3 attention products. The widenings themselves are no longer read by this product.
                from .matrix_codegen_p6d import select
                from .matrix_rules_p6d import HEADER, emit_group, spec_headers
                (sa, la_b), (sb, lb_b) = sources
                spec, reason = select(sa, sb, out, la_b, lb_b, shape, "bfloat16", arch=target_arch, engine_ok=True)
                if spec is not None and spec.get('engine') == 'p5c':
                    from . import p5c_matrix as _p5c_raw
                    spec = dict(spec, out_raw=True, ws_ok=_p5c_raw.P8WS_RAW)
                    matrix_decisions.append({'output': name, 'selected': True, 'operands': [sa, sb],
                        'reason': 'BF16-exact FP32 product on the warpgroup-MMA engine p5c (exact operand widenings; '
                                  'FP32 accumulation; unrounded FP32 result)'})
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                    for header in spec_headers(spec):
                        if header not in extra_headers:
                            extra_headers.append(header)
                        if f'#include "{header}"' not in helper_parts:
                            helper_parts.append(f'#include "{header}"')
                    code, extra = emit_group([spec], ident=_ident)
                    add_heavy('gemm', op, name, code, [sa, sb], k=la.shape[-1], matrix_spec=spec, **extra)
                    continue
                matrix_decisions.append({'output': name, 'selected': False,
                    'reason': 'BF16-exact FP32 product not admitted by p5c: ' + str(reason)})
            if matrix_policy != "frozen" and base != "addmm" and dtypes[a] == dtypes[b] == dtypes[name]:
                from .matrix_codegen_p6d import select
                from .matrix_rules_p6d import HEADER, emit_group, spec_headers
                spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch, engine_ok=True)
                from . import p5c_matrix
                if spec is not None and spec.get('engine') == 'p5c' and p5c_matrix.P5C_CONVERT:
                    # Exact rule: when the BF16 product's only reader is its BF16 -> FP32 conversion, the matrix
                    # phase stores float(bf16_rn(acc)) into the conversion's storage; the BF16 value has no storage.
                    readers = value_readers.get(name, [])
                    conv = output_name(readers[0]) if len(readers) == 1 else None
                    rtarget = str(_field(readers[0], "target", "")) if readers else ""
                    # Only where the FP32 value is materialized anyway (a graph output, e.g. a returned gradient,
                    # or a committed state source); otherwise consumers inline the cheap conversion of BF16 loads.
                    if (conv is not None and rtarget.removeprefix("aten.").split(".")[0] == "_to_copy"
                            and (conv in outputs or conv in set(writes.values()))
                            and dtypes.get(conv) == "float32" and name not in outputs
                            and name not in set(writes.values()) and conv not in undefined
                            and tuple(int(x) for x in _field(values[conv], "shape")) == shape):
                        converted_products[conv] = name
                        layouts[conv] = TensorLayout(shape, _contiguous(shape), conv)
                        spec = dict(spec, out=_ident(conv), out_f32=True, converted_from=name)
                        reason += '; FP32-converting store into its only reader ' + conv
                        name = conv
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
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
            if matrix_policy != 'frozen' and base != 'addmm':
                from .matrix_codegen_p6d import fp32_async
                plan, reason = fp32_async(a, b, name, la, lb, shape, ident=_ident,
                    arch=target_arch, split=split_k, occupied_names=names | set(layouts))
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
                from .matrix_codegen_p6d import split_k as split_codegen
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
            if (P6D_FULL_REDUCTION and results == 1 and extent >= FULL_REDUCTION_MIN
                    and dtypes[name] == "float32" and exact_reductions):
                # LATCH-DENSE-EXEC: the existing multi-CTA full reduction (C-OPS4-B, from P6-C msum; the common
                # planner's multi_cta_full body, same text and arithmetic) instead of one CTA summing every
                # element. CTA b sums contiguous chunk b with the CTA tree; the last CTA to arrive (ticket counter
                # in this emitter's full-reduction scratch, zeroed in phase 0 and restored to zero by the last
                # arriver) combines the partials in fixed CTA order. Deterministic for a grid size.
                # LATCH-FFN-RETAIN: when the reduced axes merge into one strided run (e.g. a contiguous tensor)
                # the leaf reads its element at the equal 32-bit position r * stride instead of re-deriving it
                # with 64-bit divisions per element, and each thread keeps FULL_UNROLL of its ascending leaves in
                # flight. Same elements, same per-thread order, same trees: bitwise equal.
                simple = p6d_rules._simple_position(src, dims, kept, "r", "o") if FULL_SIMPLE else None
                read_r = storage_read(source, simple if simple is not None else pos)
                unroll = f"#pragma unroll {FULL_UNROLL}\n  " if FULL_SIMPLE else ""
                code = f'''{{ const long long o = 0;
if (gridDim.x <= 1024) {{
  __shared__ unsigned msum_last_;
  const long long chunk_ = ({extent}LL + gridDim.x - 1) / gridDim.x, lo_ = (long long)blockIdx.x * chunk_;
  const long long hi_ = lo_ + chunk_ < {extent}LL ? lo_ + chunk_ : {extent}LL;
  float acc = 0.0f;
  {unroll}for (long long r = lo_ + threadIdx.x; r < hi_; r += blockDim.x)
    acc += {read_r};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {{
      (/*FULL_PART*/)[blockIdx.x] = acc;
      __threadfence();
      msum_last_ = atomicAdd(/*FULL_CNT*/, 1u) == gridDim.x - 1u;
    }}
  }}
  __syncthreads();
  if (msum_last_) {{
    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += __ldcg((/*FULL_PART*/) + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {{
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) {{ {out}[0] = {rounded_result('acc' + scale)}; *(/*FULL_CNT*/) = 0u; }}
    }}
  }}
  __syncthreads();
}}
}}'''
                add_heavy("reduction", op, name, code, refs, reduction_elements=extent, engine="multi_cta_full",
                          requires_full_grid=True, max_grid_blocks=1024,
                          arithmetic="per-CTA contiguous chunk trees, then fixed-order combine of CTA partials")
                continue
            if exact_reductions:
                selected = p6d_rules.reduction_engine(out=out, position=pos, extent=extent, results=results,
                                                      read=lambda position: storage_read(source, position),
                                                      rounded=rounded_result, scale=scale, src=src, dims=dims,
                                                      kept=kept)
                if selected is not None:
                    code, extra = selected
                    add_heavy('reduction', op, name, code, refs, **extra)
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
                    if p6d_rules.POW_ONE and isinstance(args[1], (int, float)) and args[1] == 1:
                        return rd(0)          # P6-D: IEEE pow(x, 1) = x (exhaustively checked for CUDA powf)
                    return f"powf({rd(0)}, {rd(1)})"
                if base == "neg":
                    return f"(-{rd(0)})"
                if base == "reciprocal":
                    return f"(1.0f / {rd(0)})"
                if base in {"tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs"}:
                    fn = {"abs": "fabsf"}.get(base, base + "f")
                    return f"{fn}({rd(0)})"
                if base == "sigmoid":
                    if practical:
                        return f"__fdividef(1.0f, 1.0f + __expf(-{rd(0)}))"
                    return f"(1.0f / (1.0f + expf(-{rd(0)})))"
                if base == "silu":
                    if practical:
                        return f"__fdividef({rd(0)}, 1.0f + __expf(-{rd(0)}))"
                    return f"({rd(0)} / (1.0f + expf(-{rd(0)})))"
                if base == "silu_backward":
                    sigmoid = (f"__fdividef(1.0f, 1.0f + __expf(-{rd(1)}))" if practical
                               else f"(1.0f / (1.0f + expf(-{rd(1)})))")
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
                if isinstance(arg, (int, float, bool)):
                    return _float(arg)
                raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
            probe_loads.append(ref)
            return f"v_{ref}[0]"
        render("i", probe)
        own = 4 if base in TRANSCENDENTAL else (0 if not probe_loads else 1)
        if base == "pow" and len(args) > 1 and p6d_rules.powf_exponent(args[1]):
            powf_nodes.add(name)
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
    for kind, ref, _, _ in token_loads:
        uses[root_of(ref)] = uses.get(root_of(ref), 0) + 1
    # P6-D: widest reader extent of each root (an expanded view counts its broadcast extent).
    reader_extent = {}
    for node in nodes.values():
        if node.kind == "elementwise":
            for ref in node.loads:
                reader_extent[root_of(ref)] = max(reader_extent.get(root_of(ref), 0), _numel(node.shape))
    for kind, ref, shape, _ in token_loads:
        extent = _numel(shape) if shape is not None else layouts[ref].numel
        reader_extent[root_of(ref)] = max(reader_extent.get(root_of(ref), 0), extent)
    # Transitive through elementwise readers (which are usually inlined): the widest extent this value reaches.
    for node in reversed(list(nodes.values())):
        if node.kind == "elementwise":
            for ref in node.loads:
                reader_extent[root_of(ref)] = max(reader_extent.get(root_of(ref), 0),
                                                  reader_extent.get(node.name, 0), _numel(node.shape))
    broadcast_materialized = []
    from . import p9_rowblock
    # Roots read only through the leaves of exact reductions carrying the P9 staged-engine metadata.
    token_readers = {}
    for node in nodes.values():
        if node.kind != "elementwise" and isinstance(getattr(node, "code", None), str):
            staged = node.kind == "reduction" and bool((node.extra or {}).get("p9"))
            for k in _TOKEN.findall(node.code):
                token_readers.setdefault(root_of(token_loads[int(k)][1]), set()).add(staged)
    elementwise_read = {root_of(ref) for node in nodes.values() if node.kind == "elementwise" for ref in node.loads}
    reduction_only = {r for r, kinds in token_readers.items()
                      if kinds == {True} and r not in elementwise_read and r not in forced}
    inline, total_cost = set(), {}
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        cost = node.own_cost + sum(total_cost[root_of(ref)] for ref in node.loads if root_of(ref) in inline)
        total_cost[node.name] = cost
        if (retain_all or node.name in forced or node.dtype not in ("float32", "bfloat16")
                or any(dtypes[root_of(ref)] not in ("float32", "bfloat16") for ref in node.loads)):
            continue
        n_uses = uses.get(node.name, 0)
        if (p6d_rules.BROADCAST_MATERIALIZE and node.name in powf_nodes
                and 2 * _numel(node.shape) <= reader_extent.get(node.name, 0)):
            broadcast_materialized.append(node.name)   # P6-D: store once, not per broadcast reader element
            continue
        if p6d_rules.SILU_SAVE and node.base == "sigmoid":
            broadcast_materialized.append(node.name)   # P6-D: store sigmoid(z) beside its input (same expression)
            continue
        if p9_rowblock.FANIN_STORE and node.dtype == "float32" and n_uses:
            # P9-A: a value combining >= 2 single-use FP32 elementwise operands of its own size is stored rather than
            # re-evaluated by each reader: readers load one tensor instead of all operands, and the operands (read only
            # here) stay in registers of the producing loop. Same expression, same rounding: bitwise equal.
            fanin = {root_of(ref) for ref in node.loads}
            single = [r for r in fanin if r in nodes and nodes[r].kind == "elementwise" and r not in inline
                      and uses.get(r, 0) == 1 and nodes[r].dtype == "float32"
                      and _numel(nodes[r].shape) == _numel(node.shape)]
            if len(single) >= 2:
                continue
        if n_uses and cost <= INLINE_TOTAL_LIMIT and cost * n_uses <= INLINE_WORK_LIMIT:
            inline.add(node.name)
        elif (p9_rowblock.LEAF_INLINE and n_uses and node.name in reduction_only
              and cost <= p9_rowblock.LEAF_INLINE_LIMIT):
            # P9-A: a value read only by exact row/column reduction leaves is evaluated inside those leaves instead of
            # being stored; the reduction can then run as soon as the value's own operands are ready (same expression).
            inline.add(node.name)

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
            inner = node.render(position, lambda arg, i2: operand(arg, node.shape, i2, loader))
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
            node.render("i", lambda arg, i2: operand(arg, node.shape, i2, loader))
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
            inner = node.render(flat, lambda arg, i2: operand(arg, node.shape, i2, loader))
            if node.dtype == "bfloat16":
                return f"__bfloat162float({ctype_round(inner, 'bfloat16')})"
            return inner
        return loader(ref, None, position)

    # ------------------------------------------------------------------ scheduling
    items = [node for node in nodes.values() if node.kind != "elementwise" or node.name not in inline]
    loads_of = {node.name: memory_loads(node) for node in items}

    def register_load(node, root, layout, shape, idx):
        """A same-loop register read: same flat index over the same element count."""
        return (node.kind == "elementwise" and idx == "i" and root in nodes and nodes[root].kind == "elementwise"
                and root not in inline and layout.root == root and layout.contiguous and layout.offset == 0
                and _numel(nodes[root].shape) == _numel(node.shape) == _numel(shape))

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

    def gemm_or_one_other(phase, node):
        heavy = [nodes[n] for n in phases[phase]["heavy"]]
        if not heavy:
            return True
        if not coschedule_matrix:
            return False
        p6d_engine = lambda h: h.kind == "reduction" and str((h.extra or {}).get("engine", "")).startswith("p6d_exact")
        if p6d_rules.MULTI_REDUCTION and p6d_engine(node):
            # P6-D: independent exact-engine reductions share a phase (sequential per CTA, one barrier fewer).
            return len(heavy) < p6d_rules.MULTI_REDUCTION and all(p6d_engine(h) for h in heavy)
        if node.kind == 'matrix_counter_init':
            return all(h.kind == 'matrix_counter_init' for h in heavy)
        if node.extra and node.extra.get('matrix_spec'):
            return (len(heavy) < p6d_rules.MATRIX_GROUP_MAX
                    and all(h.kind == 'gemm' and h.extra.get('matrix_spec') for h in heavy))
        if any(h.extra.get('matrix_spec') for h in heavy):
            return False
        if node.extra and node.extra.get('engine') == 'popsb_cpasync':
            return len(heavy) < 3 and all(h.kind == 'gemm' and h.extra.get('engine') == 'popsb_cpasync'
                                         for h in heavy)
        if any(h.extra.get('engine') == 'popsb_cpasync' for h in heavy):
            return False
        light = lambda h: h.kind in P9_LIGHT_KINDS
        if P9_LIGHT_KINDS and light(node):
            # P9-A: input-indexed grid-stride items without grid barriers share a phase (sequential per CTA).
            return len(heavy) < 6 and all(light(h) for h in heavy)
        pinned = lambda h: h.kind != "gemm" or str(h.extra.get("engine", "")).startswith("pinned")
        if not pinned(node) or not all(pinned(h) for h in heavy):
            return False
        others = [h for h in heavy if h.kind != "gemm"]
        if node.kind == "gemm":
            return all(h.kind == "gemm" for h in heavy) or len(others) <= 1
        return not others and all(h.kind == "gemm" for h in heavy)

    remat_anchors = dict(_field(graph, "metadata", {}).get("remat_anchors", {}) or {})
    for node in items:
        earliest = 0
        anchor = remat_anchors.get(node.name)
        if anchor is not None and anchor in layouts and layouts[anchor].root in phase_of:
            # A late-operand copy (ops/remat.py) runs no earlier than its consumer's
            # latest other input; otherwise ASAP would hoist it back to forward.
            earliest = phase_of[layouts[anchor].root]
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
            phase = earliest
            while True:
                while phase >= len(phases):
                    new_phase()
                if gemm_or_one_other(phase, node):
                    break
                phase += 1
            phases[phase]["heavy"].append(node.name)
            phase_of[node.name] = phase
    commit = len(phases)
    if place_by_slack and not retain_all:
        _place_by_slack(items, loads_of, phase_of, phases, register_load, nodes, dtypes, in_place, inplace_source,
                        forced, _numel)
        commit = len(phases)
    from . import p5c_matrix
    epilogue_of, epilogues = {}, {}
    if p5c_matrix.P5C_EPILOGUE and not retain_all:
        from .p5c_epilogue import attach as attach_epilogues
        epilogue_of, epilogues = attach_epilogues(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            in_place=in_place, inplace_source=inplace_source, register_load=register_load, numel=_numel,
            outputs=outputs, optimizer=p6d_rules.OPTIMIZER_EPILOGUE,
            # LATCH-FFN-RETAIN: explicit-cast programs (no declared precision policy) are retained by the P5C
            # protocol proof, which, like the long-attention protocol, does not restrict epilogue operands to
            # identity reads. Only policy-captured flat programs keep the identity-operand restriction of
            # p6d_retention. The rule itself is unchanged (state read strictly before its commit phase).
            early_state_reads=bool(long_plans) or not _field(graph, "metadata", {}).get("precision_policy"))
        if epilogues:
            # Phases emptied by the move disappear together with their grid barrier.
            keep = [k for k, phase in enumerate(phases) if phase["heavy"] or phase["loops"] or phase.get("epilogue")]
            renumber = {old: new for new, old in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for name in list(phase_of):
                phase_of[name] = renumber[phase_of[name]]
            commit = len(phases)
    row_consumers, row_owner = {}, {}
    if p6d_rules.ROW_CONSUMERS and not retain_all:
        row_consumers, row_owner = p6d_rules.attach_row_consumers(
            items=items, loads_of=loads_of, phase_of=phase_of, phases=phases, nodes=nodes, layouts=layouts,
            in_place=in_place, inplace_source=inplace_source, numel=_numel, index32=_index32)
    sunk = 0
    if p6d_rules.SINK_TINY and not retain_all:
        sunk = p6d_rules.sink_tiny_loops(items=items, loads_of=loads_of, phase_of=phase_of, phases=phases,
                                         in_place=in_place, inplace_source=inplace_source)
    if row_consumers or sunk:
        if True:
            keep = [k for k, phase in enumerate(phases) if phase["heavy"] or phase["loops"] or phase.get("epilogue")]
            renumber = {old: new for new, old in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for name in list(phase_of):
                phase_of[name] = renumber[phase_of[name]]
            commit = len(phases)


    # P9-A late light items: a light heavy item that is alone in its phase (no loops, no other heavy item) moves to the
    # latest phase before its first reader whose heavy items are all light, so its own phase and barrier disappear.
    if P9_LIGHT_KINDS and P9_LIGHT_LATE and not retain_all:
        light_ok = lambda k: (phases[k]["heavy"] and len(phases[k]["heavy"]) < 6
                              and all(nodes[h].kind in P9_LIGHT_KINDS for h in phases[k]["heavy"]))
        readers_at = {}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                readers_at.setdefault(root, []).append(phase_of[node.name])
        moved_any = False
        for p in range(len(phases)):
            ph = phases[p]
            if len(ph["heavy"]) != 1 or ph["loops"] or ph.get("epilogue") or nodes[ph["heavy"][0]].kind not in P9_LIGHT_KINDS:
                continue
            name = ph["heavy"][0]
            if name in outputs or name in inplace_source or name in set(writes.values()):
                continue
            latest = min(readers_at.get(name, [len(phases)])) - 1
            q = next((k for k in range(latest, p, -1) if light_ok(k)), None)
            if q is None:
                continue
            ph["heavy"].remove(name)
            phases[q]["heavy"].append(name)
            phase_of[name] = q
            moved_any = True
        if moved_any:
            keep = [k for k, phase in enumerate(phases) if phase["heavy"] or phase["loops"] or phase.get("epilogue")]
            renumber = {old: new for new, old in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for nm in list(phase_of):
                phase_of[nm] = renumber[phase_of[nm]]
            commit = len(phases)

    # P9-A late lone loops: a phase holding only elementwise loops moves them into the next phase when every reader of
    # every member runs in a later phase (the loops run after that phase's heavy items; the barrier disappears).
    if P9_LATE_LOOPS_DEBUG:
        for k_, ph_ in enumerate(phases):
            if any('add_194' in loop for loop in ph_['loops'].values()):
                print('P9 add_194 phase', k_, ph_['heavy'], {n: l[:4] for n, l in ph_['loops'].items()}, list((ph_.get('epilogue') or {}).keys()), file=__import__('sys').stderr)
    if P9_LATE_LOOPS and not retain_all:
        def readers_in_ok(m, q, same):
            # every reader of m in phase q is a member of a loop with m's element count (it will share m's loop)
            # LATCH-DENSE-EXEC: and reads m only as a same-index register (the contract _check_schedule enforces).
            # A broadcast read (e.g. a 0-d clip coefficient at index 0u through an inlined chain) is a memory load,
            # issued before m's store in a shared loop; such a reader keeps m's phase separate.
            return all(r in same and all(s is not None and register_load(nodes[r], root, lay, s, idx)
                                         for root, lay, s, idx in loads_of[r] if root == m)
                       for r in readers_of_name.get(m, ()) if phase_of.get(r) == q)
        readers_of_name = {}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                readers_of_name.setdefault(root, set()).add(node.name)
        readers_at = {}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                readers_at.setdefault(root, []).append(phase_of[node.name])
        commit_sources = set(in_place.values()) | set(writes.values())
        moved_any = False
        for p in range(len(phases) - 1):
            ph = phases[p]
            if ph["heavy"] or ph.get("epilogue") or not ph["loops"]:
                continue
            members = [m for loop in ph["loops"].values() for m in loop]
            nxt = phases[p + 1]
            same = {m for numel, loop in nxt["loops"].items() if numel in ph["loops"] for m in loop}
            if P9_LATE_LOOPS_DEBUG:
                print('P9 cand', p, members, {m: sorted((r, phase_of.get(r)) for r in readers_of_name.get(m, ())) for m in members},
                      sorted(same), file=__import__('sys').stderr)
            if any(r != p and r <= p + 1 and not readers_in_ok(m, p + 1, same) for m in members
                   for r in readers_at.get(m, ())):
                continue
            if P9_LATE_LOOPS_DEBUG:
                print('P9 late loop', p, members, file=__import__('sys').stderr)
            for numel, loop in ph["loops"].items():
                nxt["loops"][numel] = list(loop) + nxt["loops"].get(numel, [])   # producers first in a shared loop
                for m in loop:
                    phase_of[m] = p + 1
            ph["loops"] = {}
            moved_any = True
        if moved_any:
            keep = [k for k, phase in enumerate(phases) if phase["heavy"] or phase["loops"] or phase.get("epilogue")]
            renumber = {old: new for new, old in enumerate(keep)}
            phases[:] = [phases[k] for k in keep]
            for nm in list(phase_of):
                phase_of[nm] = renumber[phase_of[nm]]
            commit = len(phases)

    # P9-A deferred optimizer products: a product phase followed by a row-reduction phase hands its products whose
    # register epilogue updates optimizer state to the next product phase (appended last), so the state stream's
    # write-back drains during the following phases instead of the memory-bound reduction. Values are unchanged.
    deferred_products = []
    if P9_DEFER_STATE_PRODUCTS and not retain_all and epilogues:
        state_sources = set(writes.values())

        def is_matrix(k):
            heavy = phases[k]["heavy"]
            return bool(heavy) and all(nodes[h].kind == "gemm" and (nodes[h].extra or {}).get("matrix_spec") for h in heavy)

        def is_reduction(k):
            return any(nodes[h].kind == "reduction" for h in phases[k]["heavy"])

        readers_of = {}
        for node in items:
            for root, _, _, _ in loads_of[node.name]:
                readers_of.setdefault(root, set()).add(node.name)
        for p in range(len(phases) - 2):
            if not is_matrix(p) or not is_reduction(p + 1):
                continue
            q = next((k for k in range(p + 2, len(phases)) if is_matrix(k)), None)
            if q is None:
                continue
            for m in list(phases[p]["heavy"]):
                chain = (phases[p].get("epilogue") or {}).get(m)
                if not chain or not any(c in state_sources for c in chain) or len(phases[p]["heavy"]) == 1:
                    continue
                group = set(chain) | {m}
                if any(phase_of.get(r, 1 << 30) <= q for g in group for r in readers_of.get(g, ()) if r not in group):
                    continue
                phases[p]["heavy"].remove(m)
                phases[q]["heavy"].append(m)
                phases[q].setdefault("epilogue", {})[m] = phases[p]["epilogue"].pop(m)
                for g in group:
                    phase_of[g] = q
                deferred_products.append((m, p, q))
        # (the schedule check after the storage decisions validates every moved product and member)

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
            elif epilogue_of.get(node.name) == root:
                register_values.setdefault(root, set()).add(node.name)   # rounded product value in registers
            else:
                memory_readers.setdefault(root, []).append(phase_of[node.name])
    for node in items:
        if node.kind != "elementwise":
            continue
        if (retain_all or node.name in memory_readers or node.name in forced
                or node.name in inplace_source):
            stored.add(node.name)
    _check_schedule(items, loads_of, phase_of, register_load, stored, in_place, nodes, LoweringError,
                    **({'epilogue_of': {**epilogue_of, **row_owner}} if (epilogue_of or row_owner) else {}))
    # P9-A side casts: large FP32->BF16 casts of graph inputs computed in phase 0 move into earlier warp-specialized
    # product phases (before their first reader), where idle epilogue warps evaluate them as side work.
    side_casts = {}                       # host phase -> [cast names]
    if p9_rowblock.SIDE_CASTS and not retain_all and phases and phases[0]["loops"]:
        commit_phase = {t: phase_of[src] for t, src in in_place.items()}
        phase0_loaded = {root for members in phases[0]["loops"].values() for m in members
                         for root, _, _, _ in loads_of[m]} | {root for h in phases[0]["heavy"]
                                                               for root, _, _, _ in loads_of[h]}

        def host_ok(p):
            heavy = phases[p]["heavy"]
            return (heavy and all(nodes[h].kind == "gemm" and (nodes[h].extra or {}).get("matrix_spec") for h in heavy))

        def host_weight(p):
            return sum(nodes[h].extra["matrix_spec"]["m"] * nodes[h].extra["matrix_spec"]["n"]
                       * nodes[h].extra["matrix_spec"]["k"] for h in phases[p]["heavy"])

        load = {}
        candidates = []
        for numel, members in phases[0]["loops"].items():
            for m in members:
                node = nodes[m]
                loads = loads_of[m]
                if (node.kind != "elementwise" or dtypes[m] != "bfloat16" or m not in stored or m in phase0_loaded
                        or "_to_copy" not in str(node.target) or numel % 1024 or numel < p9_rowblock.SIDE_MIN
                        or len(loads) != 1 or m in outputs or m in inplace_source):
                    continue
                root, lay, shape, idx = loads[0]
                if (root in nodes or idx != "i" or dtypes.get(root) != "float32" or not lay.contiguous or lay.offset
                        or _numel(lay.shape) != numel or root not in inputs):
                    continue
                first = min(memory_readers.get(m, [len(phases)]))
                limit = min(first, commit_phase.get(root, len(phases)))
                hosts = [p for p in range(1, limit) if host_ok(p)][-p9_rowblock.SIDE_WINDOW:]
                if hosts:
                    candidates.append((first, m, root, numel, hosts))
        for first, m, root, numel, hosts in sorted(candidates):
            host = min(hosts, key=lambda p: ((load.get(p, 0) + numel) / host_weight(p), -p))
            load[host] = load.get(host, 0) + numel
            side_casts.setdefault(host, []).append(m)
            for key, members in list(phases[0]["loops"].items()):
                if m in members:
                    members.remove(m)
                    if not members:
                        del phases[0]["loops"][key]
            phase_of[m] = host
    # A p5c product read only by its epilogue chain needs no storage (its store is omitted).
    elided_products = {product for product in epilogues
                       if product not in memory_readers and product not in outputs
                       and product not in set(writes.values()) and product not in forced}
    storage_roots = [n for n in nodes if (nodes[n].kind != "elementwise" or n in stored) and n not in inplace_source
                     and n not in elided_products]
    if retain_all:
        storage_roots = [n for n in nodes if n not in inplace_source]
        if inline:
            raise LoweringError("internal: retain_all must not inline")
    pre_layout_root = {name: layout.root for name, layout in layouts.items()}
    selection_logical_layouts = {name: {**layout.to_dict(), "dtype": dtypes[name]}
                                 for name, layout in layouts.items()}
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
    errors = arena_plan.check(storage_roots, sizes, classes, offsets, arena_words)
    if errors:
        raise LoweringError("invalid arena: " + "; ".join(errors))
    selection_tensor_arena_words = arena_words
    # P9-A dead-line discard: a planned buffer whose last reader is phase k is invalidated in L2 (discard.global.L2,
    # no write-back) at the start of phase k + 1, when no buffer sharing its arena words is live in phase k + 1.
    discard_at = {}
    if p9_rowblock_mod().DISCARD and not retain_all:
        spans = {n: (offsets[n], offsets[n] + sizes[n]) for n in storage_roots}
        for name in storage_roots:
            _, first, last = classes[name]
            k = last + 1
            if last >= commit or k >= len(phases) or sizes[name] * 4 < p9_rowblock_mod().DISCARD_MIN:
                continue
            lo, hi = spans[name]
            if any(o != name and spans[o][0] < hi and lo < spans[o][1] and classes[o][1] <= k <= classes[o][2]
                   for o in storage_roots):
                continue
            a = (lo * 4 + 127) // 128 * 128
            b = hi * 4 // 128 * 128
            if b > a:
                discard_at.setdefault(k, []).append((name, a, (b - a) // 128))
    # P9-A: one scratch region after the planned arena holds the column partial sums of the staged row-block engine
    # (written and finalized inside one phase, so every such phase reuses it).
    from . import p9_rowblock
    p9_offset, p9_words = None, 0
    if p9_rowblock.ROWBLOCK and not retain_all:
        p9_words = p9_rowblock.scratch_words(phases, nodes)
        if p9_words:
            p9_offset = (arena_words + 63) // 64 * 64
            arena_words = p9_offset + p9_words
    from . import flat_long_attention as _fla
    p9_kn_offset, p9_kn_words = None, sum(v['size'] for v in _fla.KN_EPI_TARGETS.values())
    if p9_kn_words:
        p9_kn_offset = (arena_words + 63) // 64 * 64
        arena_words = p9_kn_offset + p9_kn_words
    p9_ctr_offset = None
    if p9_rowblock.ROWBLOCK and not retain_all and p9_rowblock.DEFER_FINALIZE:
        p9_ctr_offset = (arena_words + 63) // 64 * 64       # one arrival counter per deferred finalize (int32)
        arena_words = p9_ctr_offset + 256
    # LATCH-DENSE-EXEC: multi-CTA full reductions: one ticket counter each, then 1024 private partial words each
    # (counters zeroed in phase 0, restored to zero by each reduction's last arriver).
    full_reductions = [n for n in nodes if nodes[n].kind == "reduction"
                       and (nodes[n].extra or {}).get("engine") == "multi_cta_full"]
    full_ctr_offset, full_slot = None, {n: j for j, n in enumerate(full_reductions)}
    if full_reductions:
        full_ctr_offset = (arena_words + 63) // 64 * 64
        arena_words = full_ctr_offset + len(full_reductions) * (1 + 1024)

    # ------------------------------------------------------------------ emission
    has_storage = set(storage_roots) | set(inputs)
    declared = {}
    declarations = []
    local_declarations = {}
    local_pointers = P9_LOCAL_PTRS or any(node.extra and node.extra.get('engine') == 'popsb_cpasync'
                                          for node in nodes.values())
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

    if p9_offset is not None:
        declarations.append(f"float* v_p9part = arena + {p9_offset}LL;")
        local_declarations["v_p9part"] = f"float* v_p9part = latch_pr + {p9_offset}LL;"
    if p9_ctr_offset is not None:
        declarations.append(f"int* v_p9ctr = reinterpret_cast<int*>(arena + {p9_ctr_offset}LL);")
        local_declarations["v_p9ctr"] = f"int* v_p9ctr = reinterpret_cast<int*>(latch_pr + {p9_ctr_offset}LL);"
    if p9_kn_offset is not None:
        declarations.append(f"float* v_p9kn = arena + {p9_kn_offset}LL;")
        local_declarations["v_p9kn"] = f"float* v_p9kn = latch_pr + {p9_kn_offset}LL;"
    if full_ctr_offset is not None:
        full_part_offset = full_ctr_offset + len(full_reductions)
        declarations.append(f"unsigned* v_fullctr = reinterpret_cast<unsigned*>(arena + {full_ctr_offset}LL);")
        declarations.append(f"float* v_fullpart = arena + {full_part_offset}LL;")
        local_declarations["v_fullctr"] = f"unsigned* v_fullctr = reinterpret_cast<unsigned*>(latch_pr + {full_ctr_offset}LL);"
        local_declarations["v_fullpart"] = f"float* v_fullpart = latch_pr + {full_part_offset}LL;"

    def scoped(code, local=True):
        if not local_pointers or not local:
            return code
        used = sorted(set(re.findall(r"\bv_\w+\b", code)) & set(local_declarations))
        sregs = ""
        if P9_OPAQUE_SREG:
            for reg, var in (("threadIdx.x", "p9s_tx"), ("blockIdx.x", "p9s_bx"), ("blockDim.x", "p9s_nt"), ("gridDim.x", "p9s_nb")):
                if reg in code:
                    code = code.replace(reg, var)
                    sregs += f"unsigned {var} = {reg}; asm volatile(\"\" : \"+r\"({var}));\n"
        return ("float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n" + sregs
                + "\n".join(local_declarations[name] for name in used) + "\n" + code)

    if local_pointers:
        if not P9_LOCAL_SKIP_GEMM:
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
            return expand_storage(ref, idx, memory_loader)
        return _TOKEN.sub(substitute, node.code)

    vector_helpers_needed = []
    selection_epilogues = {}

    def epilogue_code(product, members):
        """p5c register epilogue over 8 contiguous outputs (flat i0..i0+7) of the product, in two lambdas.

        `ld(e, i0)` prefetches every identity-indexed operand of chunk slot e as raw 16-byte vectors (the body issues
        the loads of up to 8 chunks before computing any); `run(e, i0, gv)` evaluates each member's identical
        per-operation expression on gv[j], the product's rounded BF16 value widened to FP32 (exactly a load of the
        stored result), and performs 8-wide vector stores of stored members.
        """
        size = _numel(layouts[product].shape)
        member_set = set(members)
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
                reg = vector_refs.setdefault(ref, f"p5c_in{len(vector_refs)}_{product}")
                if dtypes[ref] == "bfloat16":
                    return f"__bfloat162float(p5c::upkb({reg}[e], j))"
                return f"p5c::upk({reg}[e], j)"
            if P8_PERIODIC_PREFETCH and ref in declared and dtypes[ref] in ("float32", "bfloat16") and shape is not None:
                # P8-A: a periodic broadcast read x[i % K] with K % 8 == 0 covers 8 contiguous elements per chunk
                # (i0 is a multiple of 8): prefetch them with the identity reads instead of 8 dependent scalar loads.
                flat = _index32(layouts[ref], shape, idx)
                # LATCH-FFN-RETAIN: the same index behind balanced redundant parentheses (broadcast view chains).
                stripped = flat
                while stripped.startswith("((") and stripped.endswith("))") and _balanced_parens(stripped[1:-1]):
                    stripped = stripped[1:-1]
                match = re.fullmatch(r"\(i % (\d+)u\)", stripped)
                if match and int(match.group(1)) % 8 == 0 and layouts[ref].contiguous and layouts[ref].offset == 0:
                    key = (ref, int(match.group(1)))
                    reg = vector_refs.setdefault(key, f"p5c_in{len(vector_refs)}_{product}")
                    if dtypes[ref] == "bfloat16":
                        return f"__bfloat162float(p5c::upkb({reg}[e], j))"
                    return f"p5c::upk({reg}[e], j)"
            return memory_loader(ref, shape, idx)
        compute, expressions = [], []
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
            compute.append(f"r_{name}[j] = {ctype_round(expression, node.dtype)};")
            expressions.append({"name": name, "dtype": node.dtype,
                                "expression": ctype_round(expression, node.dtype)})
        compute_before_rewrites = list(compute)
        if p5c_matrix.P5C_EXACT_REWRITES:
            # Exact: identical subexpressions computed once; 1/x as the correctly rounded reciprocal.
            from .p5c_epilogue import common_subexpressions, reciprocal
            temps, compute = common_subexpressions(compute, 'p5c_t_')
            compute = [reciprocal(line) for line in temps + compute]
        def base_of(key):
            return (key, "i0") if isinstance(key, str) else (key[0], f"(i0 % {key[1]}u)")
        prelude = " ".join(f"uint4 {reg}[8]{'[2]' if dtypes[base_of(ref)[0]] == 'float32' else ''};"
                           for ref, reg in vector_refs.items())
        loads = " ".join((f"{reg}[e] = *reinterpret_cast<const uint4*>({_ident(base_of(ref)[0])} + {base_of(ref)[1]});"
                          if dtypes[base_of(ref)[0]] == "bfloat16"
                          else f"{reg}[e][0] = *reinterpret_cast<const uint4*>({_ident(base_of(ref)[0])} + {base_of(ref)[1]}); "
                               f"{reg}[e][1] = *reinterpret_cast<const uint4*>({_ident(base_of(ref)[0])} + {base_of(ref)[1]} + 4);")
                         for ref, reg in vector_refs.items())
        if P8_STREAM_EPILOGUE:
            loads = re.sub(r"\*reinterpret_cast<const uint4\*>\(([^;]*?)\);", r"__ldcs(reinterpret_cast<const uint4*>(\1));", loads)
        declarations = " ".join(f"{_CTYPES[nodes[n].dtype]} r_{n}[8];" for n in members)
        state_sources = set(writes.values()) if P9_DIAG_NO_STATE_STORES else set()
        stores = " ".join(f"p5c::st8{'cs' if P8_STREAM_EPILOGUE else ''}({_ident(n)} + i0, r_{n});"
                          for n in members if n in stored and n not in state_sources)
        run = ("[&](int e, unsigned i0, const float (&gv)[8]) {\n"
               f"      {declarations}\n"
               "#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n"
               "        const unsigned i = i0 + (unsigned)j;\n        " + "\n        ".join(compute) +
               f"\n      }}\n      {stores}\n    }}")
        ld = f"[&](int e, unsigned i0) {{ {loads} }}"
        selection_epilogues[product] = {
            "members": list(members), "expressions": expressions,
            "compute_before_rewrites": compute_before_rewrites,
            "compute_after_rewrites": list(compute),
            "exact_rewrites": bool(p5c_matrix.P5C_EXACT_REWRITES),
            "store_names": [n for n in members if n in stored and n not in state_sources],
            "store_code": stores,
            "vector_reads": [{"name": base_of(ref)[0], "index": base_of(ref)[1], "register": reg,
                              "dtype": dtypes[base_of(ref)[0]]} for ref, reg in vector_refs.items()],
            "code": {"prelude": prelude, "run": run, "ld": ld},
        }
        return {"prelude": prelude, "run": run, "ld": ld}

    def loop_code(numel, members, exclude, span=None, cta=False):
        # cta=True (with span): P9-A engine 6 -- all 256 threads of the CTA evaluate the members over row `o`.
        # span=(extent,): P6-D row consumers -- one warp evaluates the members over row `o` (extent elements).
        if numel >= (1 << 31):
            raise LoweringError("fused elementwise loops require fewer than 2^31 elements")
        loop_regs = {name: f"r_{name}" for name in members}
        member_set = set(members)
        floating = all(nodes[n].dtype in ("float32", "bfloat16") for n in members)
        vector = floating and numel % 4 == 0 and (numel >= VECTOR_MIN_ELEMENTS or span is not None)
        if span is not None:
            vector = vector and span[0] % 4 == 0
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
            if (P8_ROW_HOIST and span is not None and shape is not None and ref in declared
                    and dtypes[ref] in ("float32", "bfloat16")):
                # P8-A (exact): inside warp o's row, x[i / E] is the row constant x[o] (one load per row), and in
                # 4-wide loops x[i % E] covers 4 contiguous elements at (4 q) % E (one vector load per quad).
                flat = _index32(layouts[ref], shape, idx)
                extent = span[0]
                if re.fullmatch(rf"\(*i / {extent}u\)*", flat):
                    reg = row_scalars.setdefault(ref, f"rs_{len(row_scalars)}")
                    return f"__bfloat162float({reg})" if dtypes[ref] == "bfloat16" else reg
                if (vector and extent % 4 == 0 and re.fullmatch(rf"\(*i % {extent}u\)*", flat)
                        and layouts[ref].contiguous and layouts[ref].offset == 0):
                    reg = column_refs.setdefault(ref, f"cw_{len(column_refs)}")
                    expression = f"{reg}[j]"
                    return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression
            return memory_loader(ref, shape, idx)
        row_scalars, column_refs = {}, {}
        compute, stored_members = [], [n for n in members if n in stored]
        for name in members:
            node = nodes[name]
            expression = node.render("i", lambda arg, i2, node=node: operand(arg, node.shape, i2, loader))
            compute.append(f"{loop_regs[name]}[v] = {ctype_round(expression, node.dtype)};")
        lanes = 4 if vector else 1
        count = numel // lanes
        u = 1
        per_thread = (-(-count // RESIDENT_THREADS) if span is None else
                      -(-(span[0] // lanes) // (256 if cta else 32)))
        cap = max(1, LIVE_VALUE_CAP // max(1, lanes * len(members)))
        if span is not None and not cta:
            cap = min(cap, p9_rowblock_mod().ROW_UNROLL)
        while u * 2 <= min(per_thread, UNROLL_MAX, cap):
            u *= 2
        declarations_local = " ".join(f"{_CTYPES[nodes[n].dtype]} {loop_regs[n]}[{u * lanes}];" for n in members)
        limit = _u(count)
        if span is not None:
            row = span[0] // lanes
            head = f"  const unsigned gs = {256 if cta else 32}u; const unsigned qend = (unsigned)o * {_u(row)} + {_u(row)};\n"
            head += "".join(f"  const {_CTYPES[dtypes[ref]]} {reg} = {_ident(ref)}[o];\n" for ref, reg in row_scalars.items())
            start_block = None
            limit = "qend"
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
            loads += "".join(f"        {_CTYPES[dtypes[ref]]} {reg}[4]; latch_ld4({_ident(ref)} + (4u * q) % {_u(span[0])}, {reg});\n"
                             for ref, reg in column_refs.items())
            body = (f"      const unsigned q = base + (unsigned)u * gs;\n      if (q < {limit}) {{\n{loads}"
                    f"#pragma unroll\n        for (int j = 0; j < 4; ++j) {{\n"
                    f"          const unsigned i = 4u * q + (unsigned)j; const int v = u * 4 + j;\n          "
                    + "\n          ".join(compute) + "\n        }\n      }\n")
            stores = "".join(f"        latch_st4({_ident(n)} + 4u * q, {loop_regs[n]}[u*4], {loop_regs[n]}[u*4+1], "
                             f"{loop_regs[n]}[u*4+2], {loop_regs[n]}[u*4+3]);\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned q = base + (unsigned)u * gs;\n"
                           f"      if (q < {limit}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        else:
            body = (f"      const unsigned i = base + (unsigned)u * gs; const int v = u;\n"
                    f"      if (i < {limit}) {{\n        " + "\n        ".join(compute) + "\n      }\n")
            stores = "".join(f"        {_ident(n)}[i] = {loop_regs[n]}[u];\n" for n in stored_members)
            store_block = (f"#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n      const unsigned i = base + (unsigned)u * gs;\n"
                           f"      if (i < {limit}) {{\n{stores}      }}\n    }}\n") if stored_members else ""
        first = (f"{start_block} * blockDim.x + threadIdx.x" if start_block is not None
                 else f"(unsigned)o * {_u(span[0] // lanes)} + (unsigned){'threadIdx.x' if cta else 'lane'}")
        return (f"{{ // {numel} elements{' (x4 vector)' if vector else ''}{' (row of warp o)' if span else ''}: {', '.join(members)}\n" + head +
                f"  for (unsigned base = {first}; base < {limit}; base += {u}u * gs) {{\n"
                f"    {declarations_local}\n#pragma unroll\n    for (int u = 0; u < {u}; ++u) {{\n" + body + "    }\n"
                + store_block + "  }\n}")

    def row_consumer_code(reduction):
        """P6-D: members attached to a warp-per-row reduction, evaluated by the row's warp after its store."""
        spec = row_consumers[reduction]
        code = "__syncwarp();  // the row's reduction value is stored by lane 0 before its consumers read it\n"
        if spec["scalars"]:
            code += loop_code(spec["rows"], spec["scalars"], 0, span=(1,)) + "\n"
        if spec["elements"]:
            code += loop_code(spec["rows"] * spec["extent"], spec["elements"], 0, span=(spec["extent"],)) + "\n"
        return code

    def side_cast_code(names):
        # P9-A: chunk k (1024 elements) of the concatenated casts; each element is __float2bfloat16_rn of its input,
        # exactly as in the phase-0 loop it left.
        if not vector_helpers_needed:
            vector_helpers_needed.append(True)
        lines, base = [], 0
        for n, m in enumerate(names):
            src = _ident(loads_of[m][0][0])
            chunks = _numel(nodes[m].shape) // 1024
            cond = f"if (k < {base + chunks}u) " if n < len(names) - 1 else ""
            lines.append(f"  {'else ' if n else ''}{cond}{{ s_ = {src}; d_ = {_ident(m)}; c_ = k - {base}u; }}")
            base += chunks
        code = ("auto p9side = [&](unsigned k) {  // P9-A side casts: " + ", ".join(names) + "\n"
                "  const unsigned lane = threadIdx.x & 31u;\n"
                "  const float* s_; __nv_bfloat16* d_; unsigned c_;\n" + "\n".join(lines) + "\n"
                "  float x_[8][4];\n"
                "#pragma unroll\n"
                "  for (int u = 0; u < 8; ++u) latch_ld4(s_ + 4u * (c_ * 256u + 32u * (unsigned)u + lane), x_[u]);\n"
                "#pragma unroll\n"
                "  for (int u = 0; u < 8; ++u) latch_st4(d_ + 4u * (c_ * 256u + 32u * (unsigned)u + lane), "
                "__float2bfloat16_rn(x_[u][0]), __float2bfloat16_rn(x_[u][1]), __float2bfloat16_rn(x_[u][2]), "
                "__float2bfloat16_rn(x_[u][3]));\n"
                "};")
        return code, base

    from .p5c_matrix import SharedSites
    shared_sites = SharedSites()
    attention_sites = p6d_rules.AttentionSites()
    phase_source, phase_records, selection_products = [], [], []
    selection_reductions, selection_reduction_phases = [], []
    pending_finalize = {}                 # P9-A: phase -> (deferred column finalize code, counter index)
    for k, phase in enumerate(phases):
        parts, gemm_offset, rotate = [], 0, 0
        matrix_group = None
        if k == 1 and P9_DIAG_EXTRA_SYNCS:
            parts.append("\n".join(["grid.sync();  // P9-A timing diagnostic: extra grid barrier"] * int(P9_DIAG_EXTRA_SYNCS)))
        if k == 0 and p9_kn_offset is not None:
            parts.append(f"for (unsigned i_ = blockIdx.x * blockDim.x + threadIdx.x; i_ < {p9_kn_words}u; i_ += gridDim.x * blockDim.x) "
                         "v_p9kn[i_] = 0.0f;  // P9-A: key-norm tables (atomic maxima of K product epilogues)")
            if any((nodes[h].extra or {}).get("matrix_spec", {}).get("out") in _fla.KN_EPI_TARGETS
                   for h in phase["heavy"]):
                parts.append("grid.sync();  // key-norm initialization precedes every CTA's first producer")
        if k == 0 and p9_ctr_offset is not None:
            parts.append("if (blockIdx.x == 0) v_p9ctr[threadIdx.x] = 0;  // P9-A: deferred-finalize arrival counters")
        if k == 0 and full_ctr_offset is not None:
            parts.append(f"if (blockIdx.x == 0) for (unsigned i_ = threadIdx.x; i_ < {len(full_reductions)}u; i_ += blockDim.x) "
                         "v_fullctr[i_] = 0u;  // LATCH-DENSE-EXEC: full-reduction ticket counters")
            if any(n in full_slot for n in phase["heavy"]):
                parts.append("grid.sync();  // counter initialization precedes this phase's full reductions")
        if discard_at.get(k):
            lines = []
            for name, start, count in discard_at[k]:
                lines.append(f"  for (unsigned l_ = blockIdx.x * blockDim.x + threadIdx.x; l_ < {count}u; l_ += gridDim.x * blockDim.x) "
                             f"asm volatile(\"discard.global.L2 [%0], 128;\" :: \"l\"(reinterpret_cast<char*>(arena) + {start}LL + 128LL * l_) : \"memory\");  // {name}")
            parts.append("{ // P9-A: dead buffers (last read in the previous phase) leave L2 without write-back\n" + "\n".join(lines) + "\n}")
        group_specs = [nodes[name].extra.get('matrix_spec') for name in phase['heavy']]
        if group_specs and all(group_specs) and phase.get('epilogue'):
            group_specs = [dict(spec, epi=epilogue_code(name, phase['epilogue'][name]),
                                out='nullptr' if name in elided_products else spec['out'])
                           if name in phase['epilogue'] else spec
                           for name, spec in zip(phase['heavy'], group_specs)]
        p9_record = None
        if (p9_rowblock.ROWBLOCK and not retain_all and (P9_LOCAL_PTRS or not local_pointers) and phase['heavy']
                and not (group_specs and all(group_specs))):
            ident_names = {_ident(n): n for n in layouts}
            member_names = set(phase['heavy']) | {m for r in phase['heavy'] if r in row_consumers
                                                  for m in row_consumers[r]['elements'] + row_consumers[r]['scalars']}

            def p9_dtype(ident):
                name = ident_names.get(ident)
                return dtypes.get(name) if name is not None and name in declared else None

            def p9_ok(ident, rows, extent):
                layout = layouts[ident_names[ident]]
                return layout.numel == rows * extent and layout.contiguous

            def p9_numel(ident):
                name = ident_names.get(ident)
                return layouts[name].numel if name is not None and name in declared else None

            p9_args = dict(ident_dtype=p9_dtype, ident_ok=p9_ok, written={_ident(n) for n in member_names},
                           elsize={'float32': 4, 'bfloat16': 2}, scratch=p9_offset is not None,
                           resolve=lambda text: heavy_code(SimpleNamespace(code=text)))
            p9_consumer = lambda n: row_consumer_code(n) if n in row_consumers else ''
            # Engine 6: (scalar members per warp row, element members over the whole CTA) of a row reduction.
            p9_consumer.parts = lambda n: ((loop_code(row_consumers[n]['rows'], row_consumers[n]['scalars'], 0, span=(1,))
                                            if row_consumers[n]['scalars'] else ''),
                                           (loop_code(row_consumers[n]['rows'] * row_consumers[n]['extent'],
                                                      row_consumers[n]['elements'], 0,
                                                      span=(row_consumers[n]['extent'],), cta=True)
                                            if row_consumers[n]['elements'] else '')) if n in row_consumers else ('', '')
            # Engine 2 (chunked, staged consumers) where the phase has a column reduction; engine 1 for row-only phases
            # (measured faster there, gpu-08/gpu-10 phase benches).
            has_cols = any(((nodes[n].extra or {}).get('p9') or {}).get('kind') == 'col' for n in phase['heavy'])
            nxt = phases[k + 1]["heavy"] if k + 1 < len(phases) else []
            defer_fin = bool(p9_ctr_offset is not None and has_cols and nxt and len(pending_finalize) < 256 and
                             all(nodes[h].kind == "gemm" and (nodes[h].extra or {}).get("matrix_spec") for h in nxt))
            if defer_fin:
                col_outs = {n for n in phase['heavy'] if ((nodes[n].extra or {}).get('p9') or {}).get('kind') == 'col'}
                early = set(nxt) | {m for h in nxt for m in (phases[k + 1].get('epilogue') or {}).get(h, ())}
                defer_fin = not any(root in col_outs for m in early for root, _, _, _ in loads_of[m])
            emitted = (p9_rowblock.emit_phase_v2([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                 **p9_args) if p9_rowblock.ENGINE == 2 and has_cols else
                       p9_rowblock.emit_phase_v3([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                 defer_finalize=defer_fin, **p9_args) if p9_rowblock.ENGINE == 3 else
                       (p9_rowblock.emit_phase_v4([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                  **p9_args) or
                        p9_rowblock.emit_phase_v3([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                  **p9_args)) if p9_rowblock.ENGINE == 4 else
                       p9_rowblock.emit_phase_v5([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                 **p9_args) if p9_rowblock.ENGINE == 5 else
                       p9_rowblock.emit_phase_v6([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                 **p9_args) if p9_rowblock.ENGINE == 6 else
                       p9_rowblock.emit_phase_v7([nodes[n] for n in phase['heavy']], p9_consumer, ident_numel=p9_numel,
                                                 **p9_args) if p9_rowblock.ENGINE == 7 else None)
            if emitted is None:
                emitted = p9_rowblock.emit_phase([nodes[n] for n in phase['heavy']], p9_consumer, **p9_args)
            if emitted is None:
                # LATCH-FFN-RETAIN: column reductions of any widths (direct engine), with the phase's warp-per-row
                # reductions emitted unchanged beside them.
                direct_cols = [n for n in phase['heavy'] if ((nodes[n].extra or {}).get('p9') or {}).get('kind') == 'col']
                direct_rest = [n for n in phase['heavy'] if n not in direct_cols]
                if (direct_cols and all(p9_rowblock.direct_inherited(nodes[n]) and n not in full_slot for n in direct_rest)):
                    inherited = []
                    for n in direct_rest:
                        code = heavy_code(nodes[n])
                        if p6d_rules.ROW_MARK in code:
                            code = code.replace(p6d_rules.ROW_MARK, row_consumer_code(n) if n in row_consumers else "")
                        inherited.append((n, code))
                    emitted = p9_rowblock.emit_phase_direct([nodes[n] for n in direct_cols], p9_consumer,
                                                            inherited=inherited, **p9_args)
            if emitted is not None:
                p9_code, p9_record = emitted
                if p9_record.get('p9_deferred_finalize'):
                    pending_finalize[k + 1] = (p9_record.pop('p9_deferred_finalize'), len(pending_finalize))
                parts.append(p9_code)
                if p9_rowblock.HELPERS not in helper_parts:
                    helper_parts.append(p9_rowblock.HELPERS)
        if p9_record is not None:
            pass
        elif group_specs and all(group_specs):
            from .matrix_rules_p6d import emit_group
            code, matrix_group = emit_group(group_specs, ident=_ident)
            if p5c_matrix.P5C_SHARED_SITES:
                code = shared_sites.route(code)
            if side_casts.get(k) and len(re.findall(r'p8ws::group<', code)) == 1:
                side, nside = side_cast_code(side_casts[k])
                code = re.sub(r'p8ws::group<([^>]*)>\(smem, ',
                              lambda mt: f'p8ws::group_s<{mt.group(1)}>(smem, p9side, {nside}u, ', code, count=1)
                parts.append('{\n' + side + '\n' + code + '\n}')
                matrix_group = dict(matrix_group, p9_side_casts=list(side_casts[k]), p9_side_chunks=nside)
            else:
                parts.append(code)
                for m in side_casts.get(k, ()):
                    parts.append(loop_code(_numel(nodes[m].shape), [m], 0))
            gemm_offset = matrix_group['tiles']
        else:
            for position, name in enumerate(phase["heavy"]):
                node = nodes[name]
                code = heavy_code(node)
                if name in full_slot:
                    j = full_slot[name]
                    code = (code.replace("/*FULL_CNT*/", f"(v_fullctr + {j})")
                            .replace("/*FULL_PART*/", f"(v_fullpart + {1024 * j}LL)"))
                if "/*FULL_CNT*/" in code or "/*FULL_PART*/" in code:
                    raise LoweringError(f"internal: full reduction {name} emitted outside its bound phase body")
                if p6d_rules.ROW_MARK in code:
                    code = code.replace(p6d_rules.ROW_MARK, row_consumer_code(name) if name in row_consumers else "")
                if position:
                    parts.append("__syncthreads();  // shared staging reuse between co-scheduled heavy items")
                if '/*B_VBASE*/' in code:
                    code = code.replace('/*B_VBASE*/', str(gemm_offset))
                elif node.kind == "gemm" and len(phase["heavy"]) > 1 and gemm_offset:
                    if code.count("smem);") != 1:
                        raise LoweringError("internal: co-scheduled matrix phase needs one pinned body call")
                    code = code.replace("smem);", f"smem, {gemm_offset});")
                if node.kind == "gemm":
                    gemm_offset += int(node.extra.get("tiles", 0))
                if p6d_rules.ATTENTION_SITES and node.kind.startswith("attention_long_"):
                    code = attention_sites.route(code)
                parts.append(f"// {node.kind}: {name}\n{{\n{code}\n}}")
        if k in pending_finalize:
            fin, d = pending_finalize[k]
            parts.append("{ // P9-A: column finalize of the previous reduction phase, after this CTA's product tiles\n"
                         + fin + "\n  __syncthreads();\n"
                         f"  if (threadIdx.x == 0) {{ __threadfence(); atomicAdd(v_p9ctr + {d}, 1); "
                         f"while (atomicAdd(v_p9ctr + {d}, 0) < (int)gridDim.x) __nanosleep(64); __threadfence(); }}\n"
                         "  __syncthreads();  // every column value is stored before this phase's loops read it\n}")
        matrix_only = phase["heavy"] and all(nodes[n].kind == "gemm" for n in phase["heavy"])
        exclude = gemm_offset if matrix_only else 0
        for numel, members in phase["loops"].items():
            member_set = set(members)
            independent = all(root not in member_set for m in members for root, _, _, _ in loads_of[m])
            if (p9_rowblock.SPLIT_STREAMS and independent and len(members) > 1
                    and numel >= p9_rowblock.SPLIT_STREAMS_MIN):
                # P9-A: independent members of a large loop stream one tensor at a time (same expressions).
                for m in members:
                    parts.append(loop_code(numel, [m], exclude))
                continue
            parts.append(loop_code(numel, members, exclude))
        heavy_nodes = [nodes[n] for n in phase["heavy"]]
        loop_names = [n for members in phase["loops"].values() for n in members]
        kind = (next((h.kind for h in heavy_nodes if h.kind != "gemm"), None)
                or (heavy_nodes[0].kind if heavy_nodes else "elementwise"))
        all_names = [*phase["heavy"], *loop_names, *side_casts.get(k, ())]
        reads = []
        for name in all_names:
            for root, layout, shape, idx in loads_of[name]:
                if not (shape is not None and register_load(nodes[name], root, layout, shape, idx)
                        and phase_of.get(root) == k):
                    reads.append(root)
        record = {"kind": kind, "names": all_names,
                  "targets": [nodes[n].target for n in all_names],
                  "reads": list(dict.fromkeys(reads)),
                  "writes": [n for n in all_names if nodes[n].kind != "elementwise" or n in stored],
                  "items": [{"kind": h.kind, "name": h.name,
                             **{key: h.extra[key] for key in ("tiles", "work_items", "group", "source_commit",
                                                             "packed_outputs", "packed_elements", "phase_writes")
                                if key in h.extra}}
                            for h in heavy_nodes]
                           + [{"kind": "elementwise", "elements": numel, "names": members}
                              for numel, members in phase["loops"].items()]}
        if not heavy_nodes:
            shapes = [nodes[n].shape for n in loop_names]
            record["shape"] = list(max(shapes, key=_numel))
            record["dtype"] = nodes[loop_names[0]].dtype
        for h in heavy_nodes:
            for key in ("tiles", "reduction_elements", "engine", "batch", "arithmetic", "work_items",
                        "selection_grid", "requires_full_grid", "max_grid_blocks", "matrix_plan", "donor_commit"):
                if key in h.extra and key not in record:
                    record[key] = h.extra[key]
        if sum(1 for h in heavy_nodes if h.kind == "gemm") > 1:
            record["tiles"] = sum(int(h.extra.get("tiles", 0)) for h in heavy_nodes if h.kind == "gemm")
        if any("shared_bytes" in h.extra for h in heavy_nodes):
            record["shared_bytes"] = max(h.extra.get("shared_bytes", 0) for h in heavy_nodes)
        if matrix_group is not None:
            record.update(matrix_group)
            selection_products.extend({
                "name": name, "phase": k, "base_spec": deepcopy(nodes[name].extra['matrix_spec']),
                "spec": deepcopy(spec), "member": deepcopy(member),
                "stored_product": name not in elided_products,
                "epilogue": deepcopy(selection_epilogues.get(name)),
            } for name, spec, member in zip(phase['heavy'], group_specs, matrix_group['members']))
        if p9_record is not None:
            record.update(p9_record)
        phase_records.append(record)
        diag = p5c_matrix.P5C_DIAG_CALL
        if diag and ((diag == 'all') or (diag == 'elementwise' and not heavy_nodes) or (diag == 'first' and k == 0)
                     or (diag == 'last' and k == len(phases) - 1)):
            fn = p5c_matrix.P5C_DIAG_FN
            if fn == 'inline_assert':
                parts.append("assert(smem != nullptr);  // diagnostic device assert (P5C_DIAG_CALL)")
            elif fn == 'inline_trap':
                parts.append("if (smem == nullptr) __trap();  // diagnostic trap check (P5C_DIAG_CALL)")
            else:
                parts.append(f"p5c::{fn}(smem);  // diagnostic out-of-line call (P5C_DIAG_CALL)")
        local_phase = not (P9_LOCAL_SKIP_GEMM and phase["heavy"] and all(nodes[n].kind == "gemm" for n in phase["heavy"]))
        phase_source.append(f"// Phase {k}: {', '.join(all_names)}\n{{\n" + scoped("\n".join(parts), local_phase)
                            + f"\n}}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{k+1}] = stamp();")
        reduction_names = [name for name in phase['heavy'] if nodes[name].kind == 'reduction']
        if reduction_names:
            selection_reductions.extend({
                "name": name, "phase": k, "target": nodes[name].target,
                "shape": list(layouts[name].shape), "dtype": dtypes[name],
                "extra": deepcopy(nodes[name].extra),
                "node_code_sha256": hashlib.sha256(nodes[name].code.encode()).hexdigest(),
                "storage_reads": [{"token": f"@@LOAD{index}@@", "token_index": index, "kind": kind,
                                   "source": ref, "shape": list(shape) if shape is not None else None,
                                   "position": position}
                                  for index in nodes[name].loads
                                  for kind, ref, shape, position in [token_loads[index]]],
            } for name in reduction_names)
            selection_reduction_phases.append({
                "phase": k, "names": reduction_names, "p9_record": deepcopy(p9_record),
                "emitted_code_sha256": hashlib.sha256("\n".join(parts).encode()).hexdigest(),
                "phase_source_sha256": hashlib.sha256(phase_source[-1].encode()).hexdigest(),
            })
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
    kernel_body = ("  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n" + "\n".join(declarations)
                   + shared_sites.declarations() + attention_sites.declarations()
                   + "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n" + "\n".join(phase_source)
                   + tail + f"  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{commit+1}] = stamp();\n"
                   + shared_sites.definitions() + attention_sites.definitions())
    if vector_helpers_needed:
        helper_parts.append(VECTOR_HELPERS)
    helpers = "\n".join(helper_parts)
    if P9_SREG_MACRO:
        # P9-A: every special-register read in header and phase code is a volatile read, so no derived index value is
        # common-subexpression-eliminated across phases (live ranges stay inside a phase).
        helpers = (r'''#ifndef P9_SREG_MACRO_DEFINED
#define P9_SREG_MACRO_DEFINED
struct p9_u3 { unsigned x, y, z; };
__device__ __forceinline__ p9_u3 p9_sr_tid() { p9_u3 r; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r.x)); r.y = 0u; r.z = 0u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_ctaid() { p9_u3 r; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r.x)); r.y = 0u; r.z = 0u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_ntid() { p9_u3 r; asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(r.x)); r.y = 1u; r.z = 1u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_nctaid() { p9_u3 r; asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r.x)); r.y = 1u; r.z = 1u; return r; }
#define threadIdx p9_sr_tid()
#define blockIdx p9_sr_ctaid()
#define blockDim p9_sr_ntid()
#define gridDim p9_sr_nctaid()
#endif
''' + helpers)
    shared_bytes = max([17408] + [p.get("shared_bytes", 0) for p in phase_records])
    from .resources import select_program_resources
    # P-OPS4-D: long attention items keep FP32 accumulators in registers; programs with the rule keep one CTA per SM.
    resource_policy = select_program_resources(shared_bytes, phases=phase_records, arch=target_arch,
                                               allow_two_cta=not long_plans)
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    header_hashes = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
                     for name in ("core.cuh", "gemm.cuh", "epi.cuh", "plan.py", *extra_headers)}
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    aliases = {name: {**layout.to_dict(), "dtype": dtypes[name]} for name, layout in layouts.items()
               if layout.root in has_storage}
    elementwise_names = [n for n in nodes if nodes[n].kind == "elementwise"]
    # LATCH-DENSE-EXEC: declared same-loop register reads (root -> consumers), by the final placement and the
    # predicate _check_schedule enforces. Selection uses it to tell register reads through views from memory reads.
    register_reads = {}
    for node in items:
        for root, layout, shape_, idx in loads_of[node.name]:
            if (root in phase_of and shape_ is not None and phase_of[root] == phase_of.get(node.name)
                    and register_load(node, root, layout, shape_, idx)):
                register_reads.setdefault(root, set()).add(node.name)
    manifest = {
        **({"requires_arch": "sm_90a"} if long_versions & set(range(4, 13)) else {}),
        "register_reads": {root: sorted(names_) for root, names_ in sorted(register_reads.items())},
        **({"pops4d": {"recipe": pops4d_recipe, "groups": sorted(long_plans), "versions": sorted(long_versions),
                       "one_cta": resource_policy["resident_blocks_per_sm"] == 1,
                       "defines": dict(pops4d_defines or {}), "source_commit": "78201354e"}} if long_plans else {}),
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
        "materialization": {"elementwise": len(elementwise_names), "inlined": sorted(inline), "p6d_broadcast_materialized": broadcast_materialized,
                            "register_only": sorted(n for n in elementwise_names if n not in inline and n not in stored),
                            "stored": len([n for n in elementwise_names if n in stored])},
        "phases": phase_records,
        **({"matrix_scratch": {"counter_word_offset": full_ctr_offset, "counter_words": len(full_reductions),
                               "partial_words": 1024 * len(full_reductions),
                               "owner": "multi_cta_full reductions (LATCH-DENSE-EXEC)"}} if full_ctr_offset is not None else {}),
        "matrix_decisions": matrix_decisions,
        "attention_groups": attention_decisions, "attention_refusals": attention_refusals,
        "ordered_index_gradients": index_decisions,
        "compiler_policy": "integrated_v1", "target_arch": target_arch,
        "phase_count": commit + 1, "barrier_count": commit + (1 if commit_code else 0),
        "max_gemm_tiles": max([p.get("tiles", 0) for p in phase_records] or [1]),
        "shared_bytes": shared_bytes,
        **({"resource_policy": resource_policy} if resource_policy['has_popsb'] else {}),
        "numerical_policy": ("Explicit BF16 casts/operands/results; BF16 MMA with FP32 accumulation; FP32 master state; fixed reduction order; no TF32/data atomics"
                             if "bfloat16" in dtypes.values() else
                             "FP32 SIMT fmaf; admitted ordered split-K; separately rounded pointwise arithmetic; exact reduction tree; no TF32/data atomics"),
        "fusion_policy": "identical per-operation expressions; BF16 rounding retained when a buffer is elided; inline limit "
                         f"{INLINE_TOTAL_LIMIT}/{INLINE_WORK_LIMIT}; loads before stores; unroll <= {UNROLL_MAX}",
        "execution": f"one cooperative CUDA launch, 256 threads/CTA, {shared_bytes} shared bytes/CTA; occupancy admitted; in-place or final state commit",
        "memory_plan": "pinned plan.assign + independent plan.check; inclusive phase intervals and alias roots; elided values have no storage",
    }
    program = GeneratedProgram(source, manifest, kernel_body, helpers)
    # Internal evidence only: this records the operands, scalar recipes and
    # bindings used above. It neither changes code generation nor travels in
    # the runtime manifest. The selection verifier decides which contracts it
    # understands; unsupported products/expressions remain unproved.
    operations_by_output = {output_name(op): op for op in operations}
    scalars = []
    for node in nodes.values():
        if node.kind != "elementwise":
            continue
        symbolic_loads = []

        def symbolic_operand(arg, idx):
            ref = _ref(arg, names)
            if ref is None:
                return _float(arg)
            symbolic_loads.append({"name": ref, "index": idx})
            expression = f"REF({ref})" if idx == "i" else f"REF({ref},{idx})"
            return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression

        expression = ctype_round(node.render("i", symbolic_operand), node.dtype)
        op = operations_by_output.get(node.name)
        scalars.append({"name": node.name, "target": node.target, "base": node.base,
                        "args": deepcopy(_field(op, "args", ())),
                        "kwargs": deepcopy(_field(op, "kwargs", {})),
                        "dtype": node.dtype, "shape": list(node.shape), "loads": list(node.loads),
                        "symbolic_loads": symbolic_loads, "expression": expression})
    program._selection_contracts = {"p6d": {
        "version": 1, "precision_policy": p8_precision or "strict",
        "products": selection_products, "scalars": scalars,
        "reductions": selection_reductions, "reduction_phases": selection_reduction_phases,
        "calculation_sources": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(p9_rowblock.__file__), Path(p6d_rules.__file__),
                                             Path(__file__).with_name("codegen_rules.py"))},
        "layouts": {name: {**layout.to_dict(), "dtype": dtypes[name]} for name, layout in layouts.items()},
        "logical_layouts": selection_logical_layouts,
        "phase_of": dict(phase_of),
        "loads_of": {name: [{"root": root, "layout": {**layout.to_dict(), "dtype": dtypes[root]},
                             "shape": list(shape) if shape is not None else None, "index": idx}
                            for root, layout, shape, idx in loads] for name, loads in loads_of.items()},
        "phases": deepcopy(phases), "stored": sorted(stored), "inline": sorted(inline),
        "storage_roots": list(storage_roots), "elided_products": sorted(elided_products),
        "in_place": dict(in_place), "state_writes": dict(writes), "epilogue_of": dict(epilogue_of),
        "copied_commits": {"mapping": {target: value for target, value in writes.items() if target not in in_place},
                           "code": list(commit_code),
                           "code_sha256": hashlib.sha256("\n".join(commit_code).encode()).hexdigest()},
        "row_owner": dict(row_owner), "side_casts": deepcopy(side_casts),
        "allocation": {"tensor_arena_words": selection_tensor_arena_words, "arena_words": arena_words,
                       "offsets": dict(offsets), "sizes": dict(sizes), "lifetimes": deepcopy(classes),
                       "p9_offset": p9_offset, "p9_words": p9_words,
                       "p9_kn_offset": p9_kn_offset, "p9_kn_words": p9_kn_words,
                       "p9_ctr_offset": p9_ctr_offset, "p9_ctr_words": 256 if p9_ctr_offset is not None else 0,
                       "pending_finalize": deepcopy(pending_finalize)},
        "resource_policy": deepcopy(resource_policy), "shared_bytes": shared_bytes,
        "settings": {"p9_diag_no_state_stores": P9_DIAG_NO_STATE_STORES,
                     "p9_rowblock_diag": list(p9_rowblock.DIAG),
                     "p9_rowblock_engine": p9_rowblock.ENGINE,
                     "p9_defer_finalize": p9_rowblock.DEFER_FINALIZE,
                     "p5c_diag_call": p5c_matrix.P5C_DIAG_CALL,
                     "p5c_shared_sites": p5c_matrix.P5C_SHARED_SITES},
    }}
    return program
