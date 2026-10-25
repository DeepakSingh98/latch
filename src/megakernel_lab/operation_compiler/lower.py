"""Operation IR -> an explicit schedule, checked arena, and one CUDA kernel.

Attributed primitive implementations cover matrices, reductions and attention.
All composition, backward and optimizer operations come from captured ATen IR;
attention fusion matches a captured VJP rather than a model backward schedule.
Views are storage aliases; dependencies are interpreted through their roots.
"""
from __future__ import annotations

# P9-B: float4 elementwise phases with at least this many float4 per launch issue four float4 indices per thread
# iteration (all loads first); None keeps the parent threshold (vec4u, > 65536 float4)
P9_VEC_UNROLL = None
# P9-B: float4 indices per thread iteration of unrolled float4 elementwise phases (parent: 4)
P9_VEC_UNROLL_FACTOR = 4
# P9-B: float4 elementwise phases with at least this many float4 stream through shared memory with bulk copies
# (kernels/p9_bulk.cuh); None keeps register loads
P9_BULK_MIN_FLOAT4 = None
# P9-B: order groupable p9ws products first within a stage (members are independent) so they merge into one group
P9_GROUP_ORDER = True
# P9-B: products of one stage use a compile-time longest-first tile table instead of rotation-dealt tiles
P9_STAGE_LPT = True
# P9-B: the tile shapes of a stage's table products (no generated epilogue) are chosen jointly, minimizing the modeled
# makespan of the stage's longest-first table (a lone long product with few tiles takes narrower tiles); bitwise.
P9_STAGE_TILES = True    # gpu-114: 6.410 vs 6.584 ms (3 rounds), bitwise 12/12
P9_STAGE_TABLE_MIN = 2   # P9-B: products of a stage that take the stage table (and the joint tile search)
P9_STAGE_TILES_192 = False  # P9-B: 128 x 192 tiles among the stage search candidates
P9_STAGE_TILES_GAIN = 0.99  # P9-B: a tile change must cut the stage's modeled makespan below this fraction
P9_QNORM_EPI = True      # gpu-136: 6.244 vs 6.298 ms, bitwise. P9-B: also the attention's q-norm block maxima from that epilogue (exact)
P9_DELTA_EPI = True      # gpu-131: 6.215 vs 6.272 ms, bitwise. P9-B: attention-backward deltas from the dO product's staged epilogue (bitwise)
P9_CHAIN = True          # gpu-123: 6.315 vs 6.393 ms, bitwise. P9-B: same-shape, same-configuration table products of a stage run as one p5c pipeline
P9_STAGE_TILES_EPI = False  # P9-B: also products with generated epilogues (128 x 128 / 128 x 256 / 256 x 128)
P9_MAP_PREFETCH = False    # P9-B: prefetch.tensormap of every product operand map at stage entry (thread 0)
P9_GEMM_REPEAT = None      # diagnostic only: 'loop' (same code twice) or 'dup' (two code copies) for product stages
# P9-B: forward products read once-per-launch transposed weight casts (K-major B operands); ops/transposed_casts.py
P9_TRANSPOSED_CASTS = False
# P9-B: common-subexpression elimination of the saved body and its tape (ops/tape_cse.py)
P9_TAPE_CSE = True
# LATCH-STEP-06: shared-weight gradient products leave the reverse loop body and run once per update after the last
# reverse step, in the original per-step order (ops/loop_defer_grads.py, kernels/step06_dgrad.cuh; bitwise).
STEP06_DEFER_GRADS = False  # compatibility diagnostic; public selection uses recurrent_gradients
# LATCH-STEP-06: side products (outputs read by no phase of their staged region) move to later, imbalanced levels
# (ops/loop_slack.py; scheduling only, bitwise).
STEP06_SLACK = False   # g-05: slack 6.210 vs tapefo 6.146 ms (bitwise; moved dW products cost as much in attention levels: dQ queue absorbs item imbalance)
# P9-B: Philox dropout masks are recomputed in the reverse body instead of saved (ops/rng_remat.py)
P9_RNG_REMAT = True
# P9-B: FP32 elementwise products whose factors are saved too are recomputed in the reverse body (ops/tape_remat.py).
# Default since gpu-80 (with tape_remat.LATE): 6.994 vs 7.179 ms, peak 2947 -> 2515 MiB, bitwise 12/12.
P9_TAPE_REMAT = True
# P9-B: products whose only reader is a BF16 -> FP32 conversion keep their BF16 output (consumers widen on read, same
# values) instead of storing the widened FP32 values; accumulating conversions keep the FP32 epilogue. Default since
# gpu-71: 7.276 vs 7.534 ms (products -273 us: half the store bytes), bitwise 12/12.
P9_KEEP_BF16 = True
# P9-B: elementwise phases whose FP32 operands are float4-addressable and whose results are FP32 or BF16 (casts
# included) use the float4 path; BF16 results are stored as four packed values (8 bytes) per float4 index.
P9_VEC_BF16 = True

import contextvars
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Any

from .vendor import plan as arena_plan
from .explain.collect import note as _explain_note  # LATCH-EXPLAIN: no-op unless a collector is active
from . import attention_lower
from . import norm_fusion as _norm_fusion  # LATCH-TINYMOE-FUSE: colreg scope for recognized weight-gradient sums

VENDOR = Path(__file__).parent / "vendor"


def _scan_targets():
    from .scan import TARGETS
    return TARGETS


def _header_path(name):
    """Resolve only attributed, checked-in implementation headers."""
    if Path(name).name != name:
        raise LoweringError(f"invalid generated header name {name!r}")
    for folder in (VENDOR, Path(__file__).parent / "kernels"):
        path = folder / name
        if path.is_file():
            return path
    raise LoweringError(f"missing generated header {name!r}")


class LoweringError(ValueError):
    pass


# LATCH-DENSE-EXEC: FP32 products of exact BF16 -> FP32 operand widenings run on p5c from the BF16 sources with an
# unrounded FP32 result in the common planner (as fusion_p6d.P6D_EXACT_BF16_PRODUCTS). False keeps the FP32 bodies.
COMPOSE_EXACT_BF16_PRODUCTS = True


# LATCH-RESEARCH-EDITS A/B switches (research arms only, never an API): False restores the contiguous-source-only
# rule; False keeps BF16-exact FP32 products the P5C engine refuses on the existing FP32 bodies.
EXACT_VIEW_SOURCES = True
RAW_PROVIDER_PRODUCTS = True

# LATCH-TRAIN-PRECISION: opt-in ``exact_bf16_products='tensor'`` (ModuleUpdate.build). Outside the compose planner
# (where COMPOSE_EXACT_BF16_PRODUCTS already applies), an FP32 product of exact BF16 -> FP32 operand widenings runs on
# the masked TMA wgmma body from the BF16 sources with FP32 accumulation and an unrounded FP32 result (result mode
# store_raw), instead of the strict FP32 SIMT bodies. None (default) leaves lowering unchanged.
import contextlib as _contextlib
import contextvars as _contextvars
_EXACT_BF16_PRODUCTS = _contextvars.ContextVar("latch_exact_bf16_products", default=None)


def validate_exact_bf16_products(value):
    if value not in (None, "tensor"):
        raise ValueError("exact_bf16_products must be None or 'tensor'")


@_contextlib.contextmanager
def exact_bf16_products_scope(value):
    validate_exact_bf16_products(value)
    token = _EXACT_BF16_PRODUCTS.set(value)
    try:
        yield
    finally:
        _EXACT_BF16_PRODUCTS.reset(token)


def _exact_bf16_operands(la, lb, operations, output_name, names, layouts, dtypes, undefined, producer=None):
    """((BF16 source, view layout over it), ...) for both operands when each views an exact BF16 -> FP32 widening.

    ``producer`` (LATCH-BUILD-600) is the caller's ``{output_name(op): op for op in operations}`` built once per
    lowering (the operation list and value names are fixed while products are lowered), not once per product."""
    if producer is None:
        producer = {output_name(op): op for op in operations}
    found = []
    for layout in (la, lb):
        widen = producer.get(layout.root)
        if widen is None or str(_field(widen, "target", "")).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0] != "_to_copy":
            return None
        wargs, wkwargs = list(_field(widen, "args", [])), dict(_field(widen, "kwargs", {}))
        source = _ref(wargs[0], names) if wargs else None
        if (source is None or dtypes.get(source) != "bfloat16" or dtypes.get(layout.root) != "float32"
                or set(wkwargs) - {"dtype", "layout", "device", "memory_format"}
                or str(wkwargs.get("memory_format", "torch.preserve_format")) not in ("torch.preserve_format", "None")):
            return None
        whole, src = layouts[layout.root], layouts[source]
        if (whole.shape != src.shape or not whole.contiguous or whole.offset
                or source in undefined or src.root in undefined):
            return None
        if src.contiguous:
            found.append((source, TensorLayout(layout.shape, layout.strides, src.root, layout.offset + src.offset)))
            continue
        # LATCH-RESEARCH-EDITS: the BF16 source may itself be a strided permutation view of its storage (a shared
        # weight read through .t(), for example). The widened FP32 copy is contiguous, so each operand stride is one
        # of its contiguous strides; the same view over the source's strides reads identical BF16 elements.
        through = _view_through_source(layout, whole, src) if EXACT_VIEW_SOURCES else None
        if through is None:
            return None
        found.append((source, through))
    return tuple(found)


def _view_through_source(layout, whole, src):
    """``layout`` (a view of the contiguous ``whole``) expressed over ``src``'s storage, or None.

    Admits exactly the views whose strides are a selection of ``whole``'s contiguous strides with matching
    extents (identity, transposes, permutations and unit dimensions) at offset zero; slices and broadcasts are
    refused. ``whole`` and ``src`` have the same shape, so the mapped multi-index is identical and the element
    read through ``src`` is the one the widening copied."""
    if layout.offset or whole.shape != src.shape:
        return None
    strides = []
    for size, stride in zip(layout.shape, layout.strides):
        if size == 1:
            strides.append(0)
            continue
        matches = [index for index, (extent, contiguous_stride) in enumerate(zip(whole.shape, whole.strides))
                   if extent == size and contiguous_stride == stride and extent != 1]
        if len(matches) != 1:
            return None
        strides.append(src.strides[matches[0]])
    return TensorLayout(tuple(layout.shape), tuple(strides), src.root, src.offset)


def _validate_flat_precision_policy(policy, options, module_overrides):
    """Check effective arithmetic before applying diagnostic module settings.

    Scheduling/association choices retain their existing admission. Approximate
    activation/attention math and reduced probability pieces require the same
    explicit practical declaration through either supported option spelling.
    """
    if policy != "strict":
        return
    from . import flat_long_attention, mixed_attention
    practical_only = [key for key in ("p8e_fast_tanh", "p9e_ksum", "p6e_fast_act")
                      if options.get(key)]
    practical_only += [key for key in ("PMIX_FAST_SOFTMAX", "PMIX_FA_FWD", "PMIX_FA_DELTA")
                       if (options.get("pops5d_defines") or {}).get(key)]
    if module_overrides.get("mixed_attention.FA_DELTA", mixed_attention.FA_DELTA):
        practical_only.append("mixed_attention.FA_DELTA")
    explicit_recipe = options.get("pops4d_recipe")
    default_recipe = module_overrides.get("flat_long_attention.DEFAULT_RECIPE",
                                          flat_long_attention.DEFAULT_RECIPE)
    if (explicit_recipe if explicit_recipe is not None else default_recipe) != "exact3":
        practical_only.append("pops4d_recipe" if explicit_recipe is not None
                              else "flat_long_attention.DEFAULT_RECIPE")
    # Changing the dictionary can otherwise rename a one-/two-piece recipe to
    # 'exact3' and evade a check of its name. Keep all three pieces plus FP32 O.
    recipes = module_overrides.get("flat_long_attention.RECIPES", flat_long_attention.RECIPES)
    exact = recipes.get("exact3") if isinstance(recipes, dict) else None
    if not isinstance(exact, (tuple, list)) or tuple(exact) != (3, True):
        practical_only.append("flat_long_attention.RECIPES[exact3]")
    if practical_only:
        raise LoweringError("arithmetic-changing options require precision_policy='practical': "
                            + ", ".join(practical_only))


def _field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _numel(shape):
    return math.prod(shape)


def _contiguous(shape):
    result, stride = [], 1
    for d in reversed(shape):
        result.append(stride)
        stride *= d
    return tuple(reversed(result))


def _reshape_strides(old_shape, old_strides, new_shape):
    """Infer an alias by splitting/merging contiguous stride subspaces.

    Semantic reference: PyTorch v2.9.1 TensorUtils.cpp computeStride:
    https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/TensorUtils.cpp
    Singleton-axis strides are immaterial. This computes from our actual storage
    layout, since materialized operations need not preserve the tracer's strides.
    Return None when the logical flattening cannot be represented by one view.
    """
    if _numel(old_shape) != _numel(new_shape):
        return None
    axes = [(size, stride) for size, stride in zip(old_shape, old_strides) if size != 1]
    if not axes:
        return _contiguous(new_shape)
    # Chunks are ordered innermost first; each has one physical element stride.
    chunks = []
    elements, inner_stride = axes[-1]
    for size, stride in reversed(axes[:-1]):
        if stride == elements * inner_stride:
            elements *= size
        else:
            chunks.append((elements, inner_stride))
            elements, inner_stride = size, stride
    chunks.append((elements, inner_stride))
    result = [0] * len(new_shape)
    axis = len(new_shape) - 1
    for elements, stride in chunks:
        covered = 1
        while axis >= 0 and (covered < elements or new_shape[axis] == 1):
            result[axis] = covered * stride
            covered *= new_shape[axis]
            axis -= 1
        if covered != elements:
            return None
    return tuple(result) if axis == -1 else None


def _float(value):
    if isinstance(value, dict) and "dynamic_batch_scalar" in value:
        from .dynamic_codegen import scalar
        return scalar(value)
    value = float(value)
    if not math.isfinite(value):
        raise LoweringError("nonfinite scalar constants are not admitted")
    token = repr(value)
    return token + "f"


def _double(value):
    """An exact C++ double literal; never round an admitted FP64 scalar to FP32."""
    value = float(value)
    if not math.isfinite(value):
        raise LoweringError("nonfinite scalar constants are not admitted")
    return value.hex()


def _lerp_scalar_expression(target, args, kwargs, name, dtypes, names, rd, weight_read=None):
    # ATen/native/Lerp.h: opmath FP32, abs(weight)<0.5 branch.
    # Each subtraction rounds separately; the final multiply/add
    # contracts in PyTorch's CUDA kernel. Preserve that with explicit
    # fmaf under our otherwise deliberately --fmad=false build.
    # Source: pytorch/pytorch v2.9.1 aten/src/ATen/native/Lerp.h and
    # aten/src/ATen/native/cuda/Lerp.cu (BSD-3). Tensor and scalar weights
    # share the same opmath law; no optimizer/model pattern is recognized.
    weight = args[2] if len(args) == 3 else kwargs.get("weight")
    if target == "lerp.Tensor":
        if (len(args) not in (2, 3) or set(kwargs) - {"weight"}
                or (len(args) == 3 and kwargs) or dtypes[name] != "float32"
                or any(dtypes.get(_ref(a, names)) != "float32" for a in [*args[:2], weight])):
            raise LoweringError("lerp.Tensor requires three FP32 tensor operands")
        w = rd(2) if len(args) == 3 else weight_read()
        difference = f"({rd(1)} - {rd(0)})"
        return (f"(fabsf({w}) < 0.5f ? fmaf({w}, {difference}, {rd(0)})"
                f" : fmaf(-{difference}, (1.0f - {w}), {rd(1)}))")
    if (target != "lerp.Scalar" or len(args) not in (2, 3)
            or set(kwargs) - {"weight"} or (len(args) == 3 and kwargs)
            or type(weight) not in (int, float) or not math.isfinite(weight)
            or dtypes[name] != "float32"
            or any(dtypes.get(_ref(a, names)) != "float32" for a in args[:2])):
        raise LoweringError("lerp.Scalar requires FP32 tensors and one finite scalar weight")
    try:
        weight = struct.unpack('f', struct.pack('f', weight))[0]
    except (OverflowError, struct.error) as exc:
        raise LoweringError("lerp weight must be representable in FP32") from exc
    if not math.isfinite(weight):
        raise LoweringError("lerp weight must be representable in FP32")
    difference = f"({rd(1)} - {rd(0)})"
    if abs(weight) < .5:
        return f"fmaf({_float(weight)}, {difference}, {rd(0)})"
    complement = struct.unpack('f', struct.pack('f', 1.0-weight))[0]
    return f"fmaf(-{difference}, {_float(complement)}, {rd(1)})"


def _ident(name):
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise LoweringError(f"invalid IR identifier: {name!r}")
    return "v_" + name


def _used_tensor_declarations(named, code):
    """Select whole pointer identifiers in declaration order with one code scan.

    Searching separately for every declaration re-parses the same interval
    thousands of times and thrashes the regex cache on large training graphs.
    Comments and strings remain conservative references, as in the old search.
    """
    used = set(re.findall(r"\bv_\w+\b", code))
    return [(name, line) for name, line in named if name in used]


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    root: str
    offset: int = 0

    @property
    def numel(self):
        return _numel(self.shape)

    @property
    def contiguous(self):
        return all(d == 1 or s == t for d, s, t in zip(self.shape, self.strides, _contiguous(self.shape)))

    def to_dict(self):
        return {"shape": list(self.shape), "strides": list(self.strides), "root": self.root, "offset": self.offset}


def cuda_module_state_policy(manifest):
    """Classify emitted writable CUDA module state, independent of schedule names.

    Older manifests have no policy field, so callers always derive this from
    their hash-bound headers and constant-binding flag instead of trusting a
    potentially stale ``required=False`` annotation.
    """
    state_headers = {"attention_long.cuh", "attention_p9.cuh"}
    headers = sorted(state_headers.intersection(manifest.get("header_hashes", {})))
    reasons = (["constant argument bindings"] if manifest.get("p7_const_args") else [])
    reasons += ["writable attention scratch/counters: " + name for name in headers]
    return {"version": 1, "requires_private_module": bool(reasons),
            "headers": headers, "reasons": reasons,
            "scope": "one verified CUDA module per Runtime; one Runtime is not concurrently reentrant"}


@dataclass
class GeneratedProgram:
    source: str
    manifest: dict[str, Any]
    kernel_body: str | None = None
    helpers: str = ""
    functions: str = ""

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        source = (directory / "program.cu").read_text()
        manifest = json.loads((directory / "manifest.json").read_text())
        digest = hashlib.sha256((source + json.dumps(manifest["header_hashes"], sort_keys=True)).encode()).hexdigest()
        if digest != manifest["source_hash"]:
            raise LoweringError("generated source does not match manifest hash")
        for name in manifest["header_hashes"]:
            if not name.endswith(".cuh"):
                continue
            if Path(name).name != name:
                raise LoweringError("vendored header name must be a basename")
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["header_hashes"][name]:
                raise LoweringError(f"vendored header {name} does not match manifest hash")
        from .square_sum_association import validate_manifest as validate_square_sum_manifest
        validate_square_sum_manifest(manifest, source)
        return cls(source, manifest)

    @property
    def cache_key(self):
        return self.manifest["source_hash"]

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.manifest["cuda_module_state"] = cuda_module_state_policy(self.manifest)
        (directory / "program.cu").write_text(self.source)
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
        for name in self.manifest["header_hashes"]:
            if not name.endswith(".cuh"):
                continue
            if Path(name).name != name:
                raise LoweringError("vendored header name must be a basename")
            if hashlib.sha256(_header_path(name).read_bytes()).hexdigest() != self.manifest["header_hashes"][name]:
                raise LoweringError(f"vendored header {name} changed after source generation")
            (directory / name).write_bytes(_header_path(name).read_bytes())
        return directory / "program.cu"


def _ref(arg, names):
    if isinstance(arg, dict) and set(arg) == {"ref"}:
        return arg["ref"]
    if isinstance(arg, str) and arg in names:
        return arg
    name = _field(arg, "name")
    return name if name in names else None


def _flatten_refs(arg, names):
    ref = _ref(arg, names)
    if ref is not None:
        return [ref]
    if isinstance(arg, (tuple, list)):
        return [n for a in arg for n in _flatten_refs(a, names)]
    if isinstance(arg, dict):
        return [n for a in arg.values() for n in _flatten_refs(a, names)]
    return []


# P6-C 'idx32' (opt-in through group rules): index arithmetic for programs whose index spaces fit in 32 bits.
# Adjacent dimensions that address memory contiguously are merged, and coordinates use unsigned 32-bit
# division/modulo instead of 64-bit (which non-power-of-two extents turn into long emulated sequences).
# Addresses (coordinate * stride) stay 64-bit; every address is identical, only its computation changes.
_INDEX_MODE = contextvars.ContextVar("latch_ops_index_mode", default=None)
# P6-C diagnostic (loop option phase_ticks): staged regions log each phase's latest CTA finish stamp.
_PHASE_EVENTS = contextvars.ContextVar("latch_ops_phase_events", default=False)
# P6-C lptx: per-stage level context ({stage: {"load": [128 us], "peers": [stage]}}) set by the pipelined loop
# lowering for its second pass; read by balanced_plans for the loop-body regions only.
_LPT_HINTS = contextvars.ContextVar("latch_ops_lpt_hints", default=None)
_U32_LIMIT = 2 ** 31
# LATCH-STEP10-SPEED: flat-program multi-block reduction over long leading rows (see the reduction emitter).
FLAT_COLSPLIT = True
# LATCH-MOESUP: the flat column split above also for non-convolution flat programs (e.g. normalization-parameter
# gradients over 65536 rows), the same declared FP32 reassociation (fixed row blocks, fixed-order combine).
FLAT_COLSPLIT_ALL = True
# Optional non-convolution column splits must not preempt an established
# schedule. A selection probe records eligible outputs while emitting their
# prior reduction bodies; programs which need common planning are re-emitted
# with the selected split bodies after the complete retention proof.
_FLAT_COLSPLIT_PROBE = contextvars.ContextVar("latch_flat_colsplit_probe", default=None)

# LATCH-MOESUP-ROW: warp-row groups that gather table rows through an index already sorted by an indexed engine
# visit their rows in that sorted order (ops/row_fusion.py; rows independent, values unchanged). Measured slower on
# the 25M MoE backward row group (plain update 58.40 -> 59.48 ms, `plain-r678`: the scattered row reads/writes cost
# more than the table's L2 reuse saves), so off by default.
GATHER_ORDER = False


def _gather_orders(phases, layouts):
    """{index root: (sorted keys value, positions)} from segmented-engine phases over identity int64 indices."""
    orders = {}
    for phase in phases:
        keys, ids = phase.get("sorted_keys"), phase.get("sorted_ids")
        if not keys or ids not in layouts or keys not in layouts:
            continue
        li = layouts[ids]
        if li.offset or not li.contiguous or li.root not in layouts or li.numel != layouts[li.root].numel:
            continue
        orders.setdefault(li.root, (keys, int(phase["sorted_positions"])))
    return orders


# LATCH-MOESUP-ROW: column-split blocks with more than 256 columns (or non-dividing widths) keep all of a thread's
# columns and 16 rows of leaves in flight (same per-column ascending order: bit-identical partials).
COLSPLIT_PIPE = True
# LATCH-MOESUP-ROW: 'lptr' scopes keep tensor pointers in the global address space (opaque offsets from the kernel's
# arena parameter instead of an opaque arena pointer, which made nearly every access generic LD/ST). Option: measured
# plain -0.5..-0.7 ms but auxiliary +0.3..+0.4 ms on the 25M MoE+sup update (whole-kernel ptxas coupling), so off.
LPTR_GLOBAL = False
FLAT_COALESCED_REDUCTION = True
# LATCH-STEP10-SPEED: consecutive intervals of tiny phases run on one CTA (see the dependency-schedule emitter).
CHAIN_SMALL = False                   # off: tiny ops emitted one CTA per row serialize on one CTA (gpu-09/11)
CHAIN_MAX_ELEMENTS = 4096             # largest operand/result a chained phase may touch (one CTA: latency-bound)
EW_RECIPROCAL = False                 # (neutral on block_dev, gpu-12) exact (1.0f / (x)) -> __frcp_rn(x) in compose elementwise loops
REDUCTION_INLINE = True               # segmented reductions only (the thread-per-result replay: 9 -> 71 us, gpu-12): evaluate a single-use FP32 source per leaf
SEGMENTED_COLSUM = False              # (off: 0.3168 -> 0.3041 ms on block_dev, gpu-15) conv programs: middle-dim reductions with a unit-stride kept tail, one segment per CTA
REDUCTION_INLINE_NESTED = True        # per-leaf expressions inline single-use FP32 producers (needs REMAT_FULL to apply)
PIXELS_EPILOGUE = True                # identity elementwise readers of a conv3_pixels transpose run in its store loop
GENERIC_SPLIT = True                  # long-K BF16 products on the generic/skinny bodies split K (existing split bodies) wherever split-K is admitted
VEC_MIXED_UNROLL = None               # conv programs: vector indices in flight per thread in mixed loops (None: 4)
VEC4_ARENA_VIEWS = True               # LATCH-TINYMOE-EGRAD: vec4 phases read float4-aligned arena-root offset views as float4
_CHAIN_KINDS = {"elementwise", "reduction", "log_softmax", "gather", "scatter", "gemm"}
_CHAIN_FORBIDDEN = ("grid.", "atomic", "p5c::", "p8ws::", "p9ws::", "pops4b::", "latch_ctr", "cluster", "VBASE",
                    "ecta_", "p9bulk", "cp.async", "chain_")


def _ctrace_phase_stamps():
    """LATCH-TRAIN-SYNC diagnostic: LATCH_DIAGNOSTIC_CTRACE=interval keeps only per-interval start/end stamps."""
    return os.environ.get("LATCH_DIAGNOSTIC_CTRACE") != "interval"


def _chain_small(phase, body, layouts):
    """A phase that may run on CTA 0 alone: tiny operands, grid-stride (or whole-grid popsc slice) work distribution
    and no cross-CTA communication."""
    if phase.get("kind") not in _CHAIN_KINDS or phase.get("requires_full_grid") or phase.get("counter_effects"):
        return False
    if any(token in body for token in _CHAIN_FORBIDDEN):
        return False
    if "popsc::" in body and body.count("popsc::") != body.count("smem, 0, 1, 1, "):
        return False
    sizes = [layouts[r].numel for r in list(phase.get("reads", ())) + list(phase.get("writes", ())) if r in layouts]
    return bool(sizes) and max(sizes) <= CHAIN_MAX_ELEMENTS


def _chain_restrict(code):
    """Run a grid-distributed phase body on CTA 0 alone (``chain_b_``/``chain_g_`` are 0/1 in the chained block)."""
    code = code.replace("smem, 0, 1, 1, ", "smem, 0, 1, 4096, ")        # popsc: a one-CTA slice of the grid
    return code.replace("blockIdx.x", "chain_b_").replace("gridDim.x", "chain_g_")


def _index32():
    return _INDEX_MODE.get() in ("u32", "u32id")


def _coord_terms(index, dims):
    """dims: (extent, stride, tail) outer to inner, unit/broadcast entries already removed, ``None`` marking a
    broadcast break. Returns 32-bit coordinate * 64-bit stride terms with contiguous neighbours merged."""
    groups, adjacent = [], False
    for item in dims:
        if item is None:
            adjacent = False
            continue
        extent, stride, tail = item
        if adjacent and groups[-1][1] == stride * extent:
            groups[-1] = [groups[-1][0] * extent, stride, tail]
        else:
            groups.append([extent, stride, tail])
        adjacent = True
    q = f"(unsigned)({index})"
    return [(f"(long long)({q} / {tail}u % {extent}u)" if tail != 1 else f"(long long)({q} % {extent}u)")
            + f" * {stride}LL" for extent, stride, tail in groups]


def _div32(index, divisor):
    """Exact quotient of a nonnegative index below 2**31 (idx32 mode), else the original 64-bit form."""
    return f"(long long)((unsigned)({index}) / {divisor}u)" if _index32() else f"{index} / {divisor}LL"


def _mod32(index, modulus):
    return f"(long long)((unsigned)({index}) % {modulus}u)" if _index32() else f"{index} % {modulus}LL"


def _index(layout, out_shape, index="i"):
    if len(layout.shape) > len(out_shape):
        raise LoweringError(f"cannot broadcast {layout.shape} to {out_shape}")
    padded = (1,) * (len(out_shape) - len(layout.shape)) + layout.shape
    strides = (0,) * (len(out_shape) - len(layout.shape)) + layout.strides
    if _index32() and _numel(out_shape) < _U32_LIMIT:
        dims = []
        for dim, (a, b, stride) in enumerate(zip(padded, out_shape, strides)):
            if a not in (1, b):
                raise LoweringError(f"cannot broadcast {layout.shape} to {out_shape}")
            if b == 1:
                continue
            dims.append(None if a == 1 or stride == 0 else (b, stride, _numel(out_shape[dim + 1:])))
        if (_INDEX_MODE.get() == "u32id" and dims and None not in dims
                and all(d[1] == d[2] for d in dims)):
            # P6-C idxid: the layout addresses its whole loop index space element for element (every generated
            # loop bounds its index by that extent), so the coordinate is the index itself. "(i)" stays
            # textually distinct from the "[i]" store targets that fusion passes match.
            return f"({index})"
        return " + ".join(_coord_terms(index, dims)) or "0"
    terms = []
    for dim, (a, b, stride) in enumerate(zip(padded, out_shape, strides)):
        if a not in (1, b):
            raise LoweringError(f"cannot broadcast {layout.shape} to {out_shape}")
        if a == 1 or stride == 0:
            continue
        tail = _numel(out_shape[dim + 1:])
        coord = f"(({index} / {tail}LL) % {b}LL)" if tail != 1 else f"({index} % {b}LL)"
        terms.append(f"{coord} * {stride}LL")
    return " + ".join(terms) or "0"


def _transposed_readers(name, readers, rank, names):
    """Release step 14: every reader is a view swapping exactly the last two dimensions of ``name``."""
    if not readers:
        return False
    for op in readers:
        base = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        args = list(_field(op, "args", []))
        if base != "transpose" or len(args) != 3 or _ref(args[0], names) != name:
            return False
        dims = {int(args[1]) % rank, int(args[2]) % rank}
        if dims != {rank - 2, rank - 1}:
            return False
    return True


def _dtype(value):
    dtype = str(_field(value, "dtype", "float32")).removeprefix("torch.")
    if dtype == "float64" and tuple(_field(value, "shape", ())) != ():
        raise LoweringError("FP64 storage is restricted to scalar coefficients")
    return dtype


def _slice_bounds(args, size):
    dim = int(args[1]) if len(args) > 1 else 0
    start = args[2] if len(args) > 2 and args[2] is not None else 0
    end = args[3] if len(args) > 3 and args[3] is not None else size
    step = int(args[4]) if len(args) > 4 else 1
    start, end = int(start), int(end)
    start = max(0, min(size, start + size if start < 0 else start))
    end = max(start, min(size, end + size if end < 0 else end))
    if step <= 0:
        raise LoweringError("slice requires a positive step")
    return dim, start, end, step


def slice_view(src, args, shape):
    """Shared strided-slice alias used by the scan and sampled schedules."""
    from .extra_ops import view
    strides, offset = view("slice", args, src, shape)
    return TensorLayout(tuple(shape), tuple(strides), src.root, offset)


def slice_backward_index(args, out_shape, grad_shape, idx, *, unsigned=False, clamp=False):
    """(condition, grad flat index) for aten.slice_backward at output flat index idx.

    slice_backward(grad, input_sizes, dim, start, end, step) places grad into
    zeros of input_sizes; the result element is grad's element when its
    coordinate along dim lies on the slice, else +0.0.
    """
    rank = len(out_shape)
    if tuple(int(x) for x in args[1]) != tuple(out_shape) or len(grad_shape) != rank:
        raise LoweringError("slice_backward shape metadata differs from its result")
    dim = int(args[2]) % rank
    _, start, end, step = _slice_bounds([None, dim, args[3], args[4], args[5] if len(args) > 5 else 1], out_shape[dim])
    if -(-(end - start) // step) != grad_shape[dim]:
        raise LoweringError("slice_backward gradient extent differs from its bounds")
    tail = _numel(out_shape[dim + 1:])
    size = out_shape[dim]
    sfx = "u" if unsigned else "LL"
    i = idx if re.fullmatch(r"\w+", idx) else f"({idx})"
    coord = f"(({i} / {tail}{sfx}) % {size}{sfx})"
    rel = f"({coord} - {start}{sfx})"
    condition = f"({coord} >= {start}{sfx} && {coord} < {end}{sfx}" + (f" && {rel} % {step}{sfx} == 0)" if step != 1 else ")")
    outer = f"({i} / {size * tail}{sfx})"
    inner = f"({i} % {tail}{sfx})" if tail != 1 else f"0{sfx}"
    gsize = grad_shape[dim]
    if clamp:
        # P6-A: off-slice coordinates read the nearest in-bounds element (their value is discarded), so the
        # gradient load stays in bounds when a consumer evaluates it unconditionally (hoisted epilogue loads).
        rel = f"({coord} < {start}{sfx} ? 0{sfx} : ({coord} >= {end}{sfx} ? {(gsize - 1) * step}{sfx} : {rel}))"
    position = f"({outer} * {gsize * tail}{sfx} + ({rel} / {step}{sfx}) * {tail}{sfx} + {inner})"
    return condition, position


_CTYPES = {"float32": "float", "bfloat16": "__nv_bfloat16", "int64": "int64_t", "bool": "bool", "float64": "double"}
_BYTES = {"float32": 4, "bfloat16": 2, "int64": 8, "bool": 1, "float64": 8}


_VIEW_BASES = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand", "squeeze", "unsqueeze", "slice", "select", "as_strided"}
# Elementwise rules emitted by lower() itself (not transformer_lower) with one store per output element.
_ELEMENTWISE_BASES = {"add", "sub", "mul", "div", "pow", "neg", "tanh", "exp", "log", "sqrt", "rsqrt", "abs", "lerp",
                      "sigmoid", "silu", "silu_backward", "tanh_backward", "sigmoid_backward", "relu",
                      "threshold_backward", "clone", "contiguous", "_to_copy", "fill", "ones_like", "zeros_like",
                      "full", "full_like", "new_full", "zeros", "new_zeros", "ones", "new_ones"}
_ELEMENTWISE_BASES.update({"slice_backward"})
_ELEMENTWISE_BASES.update({"masked_fill", "rsub", "erf", "sin", "cos", "clamp", "clamp_min", "clamp_max", "minimum", "maximum",
                          "expm1", "log1p", "reciprocal", "where", "scalar_tensor", "lift_fresh_copy",
                          "logical_not", "bitwise_not", "eq", "ne", "lt", "le", "gt", "ge",
                          "logical_and", "logical_or", "logical_xor", "bitwise_and", "bitwise_or", "bitwise_xor"})
from .extra_ops import ELEMENTWISE as _EXTRA_ELEMENTWISE  # noqa: E402  C-OPS4-B
_ELEMENTWISE_BASES.update(_EXTRA_ELEMENTWISE)


def _is_zero_constant(base, args):
    """A tensor-valued constant zero: its content is +0.0 regardless of inputs."""
    if base in {"zeros_like", "new_zeros", "zeros"}:
        return True
    value = {"full": 1, "full_like": 1, "new_full": 2, "fill": 1}.get(base)
    return value is not None and len(args) > value and type(args[value]) in (int, float) and args[value] == 0


def _plan_output_placement(operations, values, inputs, outputs, bindings, undefined, names, output_name,
                           elide_zero=False):
    """P-OPS-E region interface: bind region results to caller-owned storage.

    bindings[i] is None (caller copies output i), "store" (the producing operation
    writes the caller's storage directly) or "accumulate" (the producing operation
    writes caller = caller + value, the same single FP32 addition the caller's
    separate accumulation pass performed). A returned clone of a region-local
    material value is elided: the source itself is placed. Accumulation is only
    placed when nothing else in the region reads the value and its producer has a
    single-store emission rule. Zero contributions retain their explicit addition
    so signed-zero semantics are preserved. Everything else is left to the
    caller's copy. Adapted from P-OPS-E commit 578fa6ea14f12b04b85148c73cf6ef44415910e1.
    """
    if bindings is None:
        return {}, {}, set()
    def valid(binding):
        return binding in (None, "store", "accumulate") or (
            isinstance(binding, str) and binding.startswith("inplace:") and binding[8:].isdigit())
    if len(bindings) != len(outputs) or not all(valid(b) for b in bindings):
        raise LoweringError("output bindings must give None/store/accumulate/inplace:<input> for every region output")
    producer, readers = {}, {}

    def base_of(op):
        return str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]

    for op in operations:
        producer[output_name(op)] = op
        # P8-C: contiguous results of the multi-result window-attention operations are placeable outputs (the
        # phase writes them directly into caller storage instead of a trailing copy).
        target_ = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.")
        if target_ in ("latch_ops.mattn_fwd.default", "latch_ops.mattn_fwd2.default"):
            for result_ in list(_field(op, "outputs", None) or []):
                if (result_ in values and tuple(int(x) for x in _field(values[result_], "strides"))
                        == _contiguous(tuple(int(x) for x in _field(values[result_], "shape")))):
                    producer.setdefault(result_, op)
    def root_of(name):
        while name in producer and base_of(producer[name]) in _VIEW_BASES:
            name = _ref(list(_field(producer[name], "args", []))[0], names)
        return name
    for op in operations:
        args, kwargs = list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))
        refs = _flatten_refs((args, kwargs), names)
        if base_of(op) == "fill":
            refs = _flatten_refs((args[1:], kwargs), names)  # the template is metadata only
        for ref in refs:
            readers.setdefault(root_of(ref), set()).add(output_name(op))
    placement, elided, zero = {}, {}, set()
    for index, (name, mode) in enumerate(zip(outputs, bindings)):
        if mode is None or name in inputs or name in undefined or name not in producer:
            continue
        op = producer[name]
        base = base_of(op)
        if base in _VIEW_BASES or outputs.count(name) != 1:
            continue
        root, clone = name, None
        if base in {"clone", "contiguous"}:
            source = _ref(list(_field(op, "args", []))[0], names)
            if (source is not None and source in producer and source not in inputs and source not in undefined
                    and base_of(producer[source]) not in _VIEW_BASES and source not in outputs
                    and source not in placement
                    and tuple(_field(values[source], "shape")) == tuple(_field(values[name], "shape"))
                    and _dtype(values[source]) == _dtype(values[name])):
                root, clone = source, name
        if root in placement:
            continue
        if mode.startswith("inplace:"):
            if int(mode[8:]) >= len(inputs):
                raise LoweringError(f"in-place result {name} names a missing input")
            target = inputs[int(mode[8:])]
            if (clone is None and base not in _ELEMENTWISE_BASES) or (clone and base_of(producer[root]) not in _ELEMENTWISE_BASES):
                raise LoweringError(f"in-place result {name} must come from an elementwise rule")
            if (tuple(_field(values[target], "shape")) != tuple(_field(values[root], "shape"))
                    or _dtype(values[target]) != _dtype(values[root])):
                raise LoweringError(f"in-place result {name} must match its input {target}")
        if mode == "accumulate":
            if _dtype(values[root]) != "float32" or readers.get(root, set()) - ({clone} if clone else set()):
                continue
            rop = producer[root]
            rbase = base_of(rop)
            rargs = list(_field(rop, "args", []))
            if not (rbase in _ELEMENTWISE_BASES or rbase in {"sum", "mean"}
                      or (rbase == "mm" and len(_field(values[root], "shape")) == 2)):
                continue
        if elide_zero and mode == "accumulate" and _is_zero_producer(producer[root], base_of(producer[root])):
            # P7-C zelide: a known +0 contribution accumulated into caller storage is omitted (no pass at all).
            # caller + 0 equals caller except that -0 becomes +0, so only the sign of exactly-zero elements can differ.
            zero.add(root)
        placement[root] = (index, mode)
        if clone:
            elided[clone] = root
    return placement, elided, zero


def _is_zero_producer(op, base):
    args, kwargs = list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))
    if base in {"zeros_like", "zeros", "new_zeros"}:
        return True
    if base in {"full", "full_like", "new_full", "fill"}:
        value = args[1] if len(args) > 1 else kwargs.get("fill_value", kwargs.get("value"))
        return isinstance(value, (int, float)) and float(value) == 0.0
    return False


# LATCH-MOE-EMA-100 diagnostic/selection switch (build-time environment): separate serial weight sums and k8
# readout products into consecutive intervals.
K8_SPLIT_INTERVALS = os.environ.get("LATCH_OPS_K8_SPLIT_INTERVALS") == "1"
# f01 fused accumulation chains: opt-in (measured slower than stored partials + elementwise chain, receipt c50).
K8_SERIAL_CHAINS = os.environ.get("LATCH_OPS_K8_SERIAL_CHAINS", "0") == "1"


def _barrier_intervals(phases, layouts, dependency):
    """Assign phases to grid-barrier intervals.

    "phase" keeps one barrier after every phase. "dependency" (P-OPS-E) places
    each phase in the earliest interval after every interval that writes a root
    it reads (as-soon-as-possible levels over the RAW graph; roots resolve
    views). Values are single-assignment, so WAR/WAW hazards can only arise from
    arena reuse; arena lifetimes are therefore planned in interval units and
    storage reuse never crosses an elided barrier. Within an interval phases keep
    program order. Each phase's arithmetic is unchanged.
    """
    if not dependency:
        return [[index] for index in range(len(phases))]
    from .execution_plan import current_execution, static_intervals
    if current_execution() == "compose":
        return static_intervals(phases, layouts)
    level, writer = [], {}
    for index, phase in enumerate(phases):
        reads = {layouts[name].root for name in phase["reads"] if name in layouts}
        writes = {layouts[name].root for name in phase["writes"]}
        if writes & set(writer):
            raise LoweringError("dependency barrier schedule requires single-assignment phase outputs")
        level.append(1 + max((level[writer[root]] for root in reads if root in writer), default=-1))
        for root in writes:
            writer[root] = index
    intervals = [[] for _ in range(max(level, default=-1) + 1)]
    for index, k in enumerate(level):
        intervals[k].append(index)
    return intervals


def _sink_recomputed(phases, intervals, layouts, recomputed):
    """P7-D sink: a phase that writes only rematerialized values (the region's recomputed_in_reverse set) moves to
    the interval just before its earliest reader instead of its as-soon-as-possible interval, so recomputed storage
    is live only around its consumers. Readers never move and no interval is added; later phases sink first so
    recomputed chains move together. Each phase's arithmetic is unchanged."""
    from .execution_plan import current_execution, dependency_successors
    successors = dependency_successors(phases, layouts) if current_execution() == "compose" else None
    level = {i: k for k, members in enumerate(intervals) for i in members}
    readers = {}
    for index, phase in enumerate(phases):
        for name in phase["reads"]:
            if name in layouts:
                readers.setdefault(layouts[name].root, set()).add(index)
    moved = []
    for index in range(len(phases) - 1, -1, -1):
        phase = phases[index]
        writes = [name for name in phase["writes"]]
        if index not in level or not writes or not all(name in recomputed for name in writes):
            continue
        roots = {layouts[name].root for name in writes if name in layouts}
        later = [level[r] for root in roots for r in readers.get(root, ()) if r != index and r in level]
        if not later:
            continue
        target = min(later) - 1
        if successors is not None:
            # Data readers alone omit WAR/WAW, private mutations and global
            # fences. A delay preserves predecessors but must not cross any
            # complete-contract successor, including one moved earlier here.
            target = min(target, min((level[s] - 1 for s in successors[index]), default=target))
        if target > level[index]:
            moved.append({"phase": index, "from": level[index], "to": target})
            level[index] = target
    result = [[] for _ in intervals]
    for index, k in level.items():
        result[k].append(index)
    return [sorted(members) for members in result if members], moved


def _footprint(phase, layouts, grid=128):
    """CTAs a non-matrix phase occupies when its work starts at CTA 0 (P-OPS2-D)."""
    if phase.get("requires_full_grid"):
        return grid
    kind = phase["kind"]
    outputs = max((_numel(layouts[w].shape) for w in phase["writes"] if w in layouts), default=1)
    if kind in ("row_fused", "softmax", "log_softmax") and phase.get("work_items"):
        return min(grid, int(phase["work_items"]))
    if kind == "reduction":
        engine = phase.get("engine")
        if engine == "exact_cta_tree_per_warp":
            return min(grid, (outputs + 7) // 8)
        if engine == "exact_col_tile":
            return min(grid, max(1, outputs // 32))
        if engine == "exact_cta_tree_per_thread":
            return min(grid, (outputs + 255) // 256)
        return min(grid, outputs)
    if kind in ("softmax", "log_softmax"):
        return min(grid, outputs // max(1, int(phase.get("reduction_elements", 1))))
    if kind in ("elementwise", "random_mask", "matrix_scratch_init"):
        return min(grid, max(1, (outputs + 255) // 256))
    if kind == "random_state":
        return 1
    if kind == "gemm":
        return min(grid, int(phase.get("tiles", grid)))
    return grid


def _defer_accumulate(phases, intervals, layouts, accumulate, grid=128, policy="earliest", rotated=False):
    """Slack placement (P-OPS2-D) of P-OPS-B products whose only results are
    accumulated region outputs that nothing in the region reads (for example a
    shared weight's gradient contribution). Each moves to the earliest later
    dependency interval without matrix work whose largest CTA footprint plus
    the product's tasks fits the grid; its tasks then run on otherwise idle CTAs.
    Placement uses dependencies and resource estimates only; arithmetic is unchanged.
    """
    from .execution_plan import current_execution, dependency_successors
    successors = dependency_successors(phases, layouts) if current_execution() == "compose" else None
    root = lambda n: layouts[n].root if n in layouts else n
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase["reads"]:
            readers.setdefault(root(r), set()).add(k)
    intervals = [list(members) for members in intervals]
    level = {i: k for k, members in enumerate(intervals) for i in members}
    moved = []

    def estimate(phase):
        # Coarse duration (us): B's own makespan estimate for products, small
        # constants otherwise; used only to pick where slack work hides best.
        if phase["kind"] == "gemm":
            return float((phase.get("matrix_plan") or {}).get("estimate_us", 8.0))
        return {"row_fused": 5.0, "softmax": 3.0, "log_softmax": 3.0, "reduction": 2.5}.get(phase["kind"], 2.0)
    for i, phase in enumerate(phases):
        code = phase.get("code") or ""
        if phase["kind"] != "gemm" or not ("/*B_VBASE*/" in code or (rotated and "/*C_VBASE*/" in code)):
            continue
        scratch = {item["name"] for item in phase.get("private_scratch", [])}
        # P-OPS5-B: a BF16 product whose store was elided (only its epilogue reads it) is internal.
        internal = ({phase["names"][0]} if (phase.get("bf16_epilogue") or {}).get("product_stored") is False
                    else set())
        results = [w for w in phase["writes"] if w not in scratch and w not in internal]
        if not results or any(w not in accumulate for w in results):
            continue
        if any(k != i for w in phase["writes"] for k in readers.get(root(w), ())):
            continue
        tasks = int(phase.get("tiles", grid))
        start = level[i]
        # This is a delay only, so predecessors remain before the product.
        # Keep every complete-contract successor strictly after it; output
        # readers alone do not account for fences and internal mutations.
        stop = (min((level[s] for s in successors[i]), default=len(intervals))
                if successors is not None else len(intervals))
        if policy == "duration":
            # Later interval with enough idle CTAs where the product's estimated
            # duration extends the interval least (earliest on ties).
            cost, best = estimate(phase), None
            for target in range(start + 1, stop):
                members = intervals[target]
                if any(phases[j]["kind"] == "splitk_reduce" or (phases[j]["kind"] == "gemm"
                       and "/*B_VBASE*/" not in (phases[j].get("code") or "")) for j in members):
                    continue
                others = [j for j in members if phases[j]["kind"] != "gemm"]
                used = (max((_footprint(phases[j], layouts, grid) for j in others), default=0)
                        + sum(int(phases[j].get("tiles", grid)) for j in members if phases[j]["kind"] == "gemm"))
                if used + tasks > grid:
                    continue
                existing = max((estimate(phases[j]) for j in members), default=0.0)
                key = (max(0.0, cost - existing), target)
                if best is None or key < best[0]:
                    best = (key, target, used)
            if best is not None:
                _, target, used = best
                intervals[start].remove(i)
                intervals[target].append(i)
                level[i] = target
                moved.append({"phase": i, "results": results, "from": start, "to": target,
                              "tasks": tasks, "idle_ctas_before": grid - used, "policy": policy})
            continue
        for target in range(start + 1, stop):
            members = intervals[target]
            if any(phases[j]["kind"] in ("gemm", "splitk_reduce") for j in members):
                continue
            used = max((_footprint(phases[j], layouts, grid) for j in members), default=0)
            if used + tasks <= grid:
                intervals[start].remove(i)
                members.append(i)
                level[i] = target
                moved.append({"phase": i, "results": results, "from": start, "to": target,
                              "tasks": tasks, "idle_ctas_before": grid - used})
                break
    return [sorted(members) for members in intervals if members], moved


_LOWERING_DEPTH = contextvars.ContextVar("latch_lowering_depth", default=0)
# A validated practical request may retain a proved stricter implementation.
# Scope this exception to the exact internal probe graph, never a caller option
# or another nested graph. The public request still goes through bind().
_STRICT_RETENTION_GRAPH = contextvars.ContextVar("latch_strict_retention_graph", default=None)


def _strict_retention_scope(graph, effective_precision):
    from contextlib import contextmanager
    from .execution_plan import current_precision

    @contextmanager
    def scope():
        selected = graph if current_precision() == "practical" and effective_precision == "strict" else None
        token = _STRICT_RETENTION_GRAPH.set(selected)
        try:
            yield
        finally:
            _STRICT_RETENTION_GRAPH.reset(token)
    return scope()


# LATCH-STEP-11: uniform-conditional lowering policy. A whole-training conditional
# branch is a complete standalone flat program (no output bindings); the wrapper
# only inlines its body. branch_selection lets compose apply the top-level flat
# selection rules to it; alias_committed_state exposes outputs that each branch
# already committed into the same input binding as views of that binding instead
# of copying them again; vector_copies copies remaining dense 4-byte outputs as
# 16-byte words. All three are bitwise-neutral (measured, gpu-diag-01) and on by
# default; {"branch_selection": False, ...} reproduces the pre-STEP-11 region.
# skip_copies is a timing diagnostic only (wrong outputs).
CONDITIONAL_POLICY = contextvars.ContextVar("latch_conditional_policy", default=None)
_CONDITIONAL_DEFAULTS = {"branch_selection": True, "alias_committed_state": True, "vector_copies": True,
                         "skip_copies": False}
_CONDITIONAL_BRANCH = contextvars.ContextVar("latch_conditional_branch_depth", default=0)


def conditional_policy():
    return {**_CONDITIONAL_DEFAULTS, **(CONDITIONAL_POLICY.get() or {})}


# LATCH-LOOP-GAP: ordered loop-region selection. The ordered loop composer (recompute_boundaries and other
# ordered single-level loops, the outer level of nested loops) and the nested composer (inner regions) lower
# every region as a complete standalone flat program without output bindings, staging or region-body
# options; the composer copies its outputs afterwards. ordered_region_selection lets compose apply the
# top-level flat selection rules to such a region, exactly as STEP-11 branch_selection does for a uniform
# conditional branch, so the verified established static emitter is no longer excluded by depth alone. All
# legacy retention checks (view hazards, reductions, matrices, protocols, indexed engines) still apply and any
# refusal keeps common planning. Bitwise-neutral (measured, t-latch-loopgap-gpu-ab-01). Staged/pipelined
# regions are unaffected (their explicit region options keep common planning).
# {"ordered_region_selection": False} reproduces the previous region selection.
LOOP_REGION_POLICY = contextvars.ContextVar("latch_loop_region_policy", default=None)
_LOOP_REGION_DEFAULTS = {"ordered_region_selection": True}
_ORDERED_LOOP_REGION = contextvars.ContextVar("latch_ordered_loop_region_depth", default=0)


def loop_region_policy():
    return {**_LOOP_REGION_DEFAULTS, **(LOOP_REGION_POLICY.get() or {})}


def ordered_loop_regions():
    """Scope for a composer that lowers each region as a standalone flat program (one level below)."""
    from contextlib import contextmanager

    @contextmanager
    def scope():
        token = _ORDERED_LOOP_REGION.set(_LOWERING_DEPTH.get() + 1)
        try:
            yield
        finally:
            _ORDERED_LOOP_REGION.reset(token)
    return scope()


_PELEM_KINDS = {"elementwise", "elementwise_inline", "embedding"}
# LATCH-MOESUP: row gathers as virtual index maps under vprod (measured slower: indirect inline loads cannot use the
# 16-byte vector loops; off by default).
VIRTUAL_GATHERS = False
# LATCH-MOE-EMA-FAST: strided leaves evaluated ahead in multi-CTA full sums (same order); 1 keeps one per step.
FULL_SUM_BATCH = 8
# LATCH-MOESUP: phases evaluating vprod-inlined producers take the same vector / ILP elementwise rules as ordinary
# elementwise phases (data movement only; previously a scalar grid-stride loop).
INLINE_VECTOR = True
GATHER_HELPER = """#ifndef LATCH_GATHER_ROW
#define LATCH_GATHER_ROW
__device__ __forceinline__ long long latch_gather_row(long long r, long long n) { if (r < 0 || r >= n) __trap(); return r; }
#endif
"""


def _gather_admitted(base, op, values, names):
    """Dim-0 row gather of a contiguous-row FP32/BF16 table by an int64 index; embedding without padding/scale."""
    args = list(_field(op, "args", []))
    kwargs = dict(_field(op, "kwargs", {}))
    if base == "embedding":
        if len(args) > 2 and args[2] not in (-1, None) or kwargs.get("padding_idx", -1) not in (-1, None):
            return False
        if (len(args) > 3 and args[3]) or kwargs.get("scale_grad_by_freq") or (len(args) > 4 and args[4]) \
                or kwargs.get("sparse"):
            return False
        table, index = _ref(args[0], names), _ref(args[1], names)
    else:
        if len(args) < 3 or int(args[1]) != 0:
            return False
        table, index = _ref(args[0], names), _ref(args[2], names)
    if table is None or index is None or table not in values or index not in values:
        return False
    if _dtype(values[index]) != "int64" or _dtype(values[table]) not in ("float32", "bfloat16"):
        return False
    return len(tuple(_field(values[table], "shape"))) >= 2
# P9-C pred: inline ticketed full reductions (their CTA count only changes the FP32 association; declared).
_PRED_ENGINES = {"p6c_msum", "p9c_colsplit"}


def _record_indexed_numerical_policy(program):
    """Correct descriptive receipts after selection; never alter emitted code or admission."""
    def visit(value):
        if isinstance(value, list):
            return any([visit(item) for item in value])
        if not isinstance(value, dict):
            return False
        nested_atomic = any([visit(item) for item in value.values()])
        records = value.get("ordered_index_gradients", ())
        atomic = nested_atomic or (isinstance(records, (list, tuple)) and any(
            isinstance(row, dict) and (row.get("indexed_reduction_policy") == "atomic_v1"
            or row.get("arithmetic_policy") == "atomic_fp32_unordered_v1") for row in records))
        text = value.get("numerical_policy")
        if atomic and isinstance(text, str):
            text = text.replace("fixed reduction order", "fixed non-indexed reduction order")
            text = text.replace("fixed CTA reduction tree", "fixed non-indexed CTA reduction tree")
            text = text.replace("no TF32/data atomics", "no TF32")
            text = text.replace("no TF32/atomics on data", "no TF32")
            note = "indexed FP32 atomics use hardware arrival order and are not bitwise repeatable"
            if note not in text:
                text += "; " + note
            value["numerical_policy"] = text
        return atomic
    visit(program.manifest)
    return program


def _serialized_lowering(function):
    from contextlib import nullcontext
    from functools import wraps
    @wraps(function)
    def call(*args, **kwargs):
        # All selected global primitive settings are restored under this lock.
        # RLock permits region recursion; independent worker processes remain parallel.
        from .performance_schedule import _LOCK
        with _LOCK:
            from .long_attention import program_state, region_namespace
            def conditional(region):
                if hasattr(region, "true_branch"):
                    return True
                return any(conditional(child) for name in
                           ("init", "body", "body_saved", "body_vjp", "final", "train")
                           if (child := getattr(region, name, None)) is not None)
            state = (program_state(shared_masks=not conditional(args[0] if args else kwargs["graph"]))
                     if not _LOWERING_DEPTH.get() else nullcontext())
            token = _LOWERING_DEPTH.set(_LOWERING_DEPTH.get() + 1)
            # LATCH-STEP10-SPEED: this lane's rules apply to programs that went through channels-last propagation.
            graph_ = args[0] if args else kwargs.get("graph")
            conv_token = (_CONV_PROGRAM.set(bool((getattr(graph_, "metadata", None) or {}).get("channels_last")))
                          if _LOWERING_DEPTH.get() == 1 else None)
            try:
                from .execution_plan import execution_scope, requested_execution
                execution = kwargs.get("execution")
                if execution == "auto" or (execution is None and _LOWERING_DEPTH.get() == 1
                                           and requested_execution() is None):
                    # One public default evaluates common planning and the
                    # established implementation's compatibility proofs. Keep
                    # explicit legacy schedule diagnostics available unchanged.
                    legacy_control = (kwargs.get("schedule", "auto") != "auto"
                                      or bool(kwargs.get("schedule_options"))
                                      or kwargs.get("compact_codegen", "off") != "off"
                                      or kwargs.get("policy", "auto") != "auto"
                                      or bool(kwargs.get("retain_all")))
                    execution = "legacy" if legacy_control else "compose"
                with state, region_namespace(), execution_scope(execution, kwargs.get("precision_policy")):
                    program = function(*args, **kwargs)
                # State identity belongs to the captured update, independent
                # of which compatible emitter supplies its CUDA schedule.
                # Historical flat emitters predate this manifest field.
                metadata = _field(graph_, "metadata", {}) or {}
                if "optimizer_state_schema" in metadata:
                    from copy import deepcopy
                    program.manifest.setdefault("optimizer_state_schema",
                        deepcopy(metadata["optimizer_state_schema"]))
                if _LOWERING_DEPTH.get() == 1:
                    # STEP17 (opt-in, LATCH_STEP17_OUTLINE): post-lowering block outlining; unchanged otherwise.
                    from .step17_outline import maybe_outline
                    program = maybe_outline(program)
                    # The historical summary participates in retention proofs.
                    # Amend it only after every emitter/region has been selected.
                    program = _record_indexed_numerical_policy(program)
                return program
            finally:
                if conv_token is not None:
                    _CONV_PROGRAM.reset(conv_token)
                _LOWERING_DEPTH.reset(token)
    return call


# LATCH-STEP10-SPEED: true while lowering a program with pixel-major convolutions (capture metadata 'channels_last').
# The lane's general rules (reductions, vector loops, epilogue loads, reciprocal, reduction inlining) apply only
# then, so every other program lowers byte-identically to the step-14 compiler until those rules are measured on it.
_CONV_PROGRAM = contextvars.ContextVar("latch_conv_program", default=False)


def lane_rules_active():
    return _CONV_PROGRAM.get()


# LATCH-MOESUP: the exact LATCH-STEP10-SPEED vector rules (non-identity broadcast reads inside 16-byte loops,
# chunk-uniform index hoisting, branch-free exact division with exact fallback, FP32/view vector epilogue loads) are
# data movement / exact-arithmetic rewrites, not convolution-specific. They also apply while the standalone flat
# composed planner lowers. FLAT_EXACT_RULES = False restores convolution-only scope (A/B arms).
FLAT_EXACT_RULES = True


def flat_exact_rules_active():
    if _CONV_PROGRAM.get():
        return True
    if not FLAT_EXACT_RULES:
        return False
    from .compose import STEP14_FLAT_ACTIVE
    return bool(STEP14_FLAT_ACTIVE.get())


def _rewrite_compose_casts(graph, *, retain_all, policy, target_arch):
    """Exact storage rewrites for static, explicit-cast optimized programs.

    Captured precision policies retain their certified materialized cast
    boundaries. Dynamic extents carry per-value axes which these rewrites do
    not yet update. Diagnostic/reference modes retain the original values.
    This only scopes the three optional cast rewrites; vocabulary and existing
    fusion/indexed selection retain their independent admission rules.
    """
    metadata = _field(graph, "metadata", {}) or {}
    if (retain_all or policy != "auto" or target_arch != "sm_90"
            or metadata.get("dynamic_batch")):
        return graph
    from .bf16_retention import exact_projection_capture
    if exact_projection_capture(graph):
        # The retained projection protocols certify the captured cast graph,
        # including per-use VJPs. Keep that canonical graph for both named and
        # explicit spellings before comparing their implementation candidates.
        return graph
    from .gather_cast import rewrite as rewrite_gather_casts
    from .widen_cast import rewrite as rewrite_widen_casts
    if metadata.get("precision_policy"):
        # LATCH-MOE-EMA-FAST: under a captured precision policy only the two bit-exact storage commutations run:
        # cast(gather(T)) == gather(cast(T)) and op(widen(b), x) == op(b read by exact widening, x). Every stored
        # value and every rounding of the certified policy is unchanged; only which copy is materialized changes.
        from . import gather_cast as gather_cast_rule, widen_cast as widen_cast_rule
        if gather_cast_rule.UNDER_PRECISION_POLICY:
            graph, report = rewrite_gather_casts(graph)
            _explain_note("cast_rewrite", **{"pass": "gather_cast", "report": report, "precision_policy": True})
        if widen_cast_rule.UNDER_PRECISION_POLICY:
            graph, report = rewrite_widen_casts(graph)
            _explain_note("cast_rewrite", **{"pass": "widen_cast", "report": report, "precision_policy": True})
        return graph
    graph, report = rewrite_gather_casts(graph)
    _explain_note("cast_rewrite", **{"pass": "gather_cast", "report": report})
    graph, report = rewrite_widen_casts(graph)
    _explain_note("cast_rewrite", **{"pass": "widen_cast", "report": report})
    from .indexed_reduction import requested_policy as _requested_indexed
    if _requested_indexed() in ("segmented_v1", "segmented_tile_v1"):
        from .scaled_merge import rewrite as rewrite_scaled_merges
        graph, report = rewrite_scaled_merges(graph)
        _explain_note("cast_rewrite", **{"pass": "scaled_merge", "report": report})
    return graph

# LATCH-MOE4-PARITY: a region's remaining state/boundary commits are plain copies of values that already
# hold the committed dtype. Copy them as 16-byte words (four independent words in flight per thread) when both
# pointers are 16-byte aligned at run time; otherwise the original scalar loop runs. Same bytes are stored for
# every finite value (the scalar loop's BF16 widen/round-to-nearest round trip is the identity on non-NaN
# values), no arithmetic. False restores the scalar copy loops.
COMMIT_VECTOR = True
_COMMIT_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int64": 8, "int32": 4, "bool": 1, "uint8": 1,
                 "int8": 1, "float64": 8}


def _vector_commit(target, source, layouts, dtypes, virtual_text):
    lt, ls = layouts.get(target), layouts.get(source)
    size = _COMMIT_BYTES.get(dtypes.get(target))
    if (lt is None or ls is None or size is None or dtypes.get(source) != dtypes.get(target)
            or ls.root in virtual_text or not ls.contiguous or ls.offset or not lt.contiguous or lt.offset
            or tuple(ls.shape) != tuple(lt.shape) or lt.numel * size % 16 or lt.numel == 0):
        return None
    words = lt.numel * size // 16
    return (f"{{ const uint4* cs_ = reinterpret_cast<const uint4*>({_ident(source)}); "
            f"uint4* cd_ = reinterpret_cast<uint4*>({_ident(target)});\n"
            "  if (((reinterpret_cast<unsigned long long>(cs_) | reinterpret_cast<unsigned long long>(cd_)) & 15ULL) == 0ULL) {\n"
            "    const long long st_ = (long long)gridDim.x * blockDim.x;\n"
            "    long long w_ = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            f"    for (; w_ + 3 * st_ < {words}LL; w_ += 4 * st_) {{ const uint4 a_ = cs_[w_], b_ = cs_[w_ + st_], "
            "c_ = cs_[w_ + 2 * st_], d_ = cs_[w_ + 3 * st_]; cd_[w_] = a_; cd_[w_ + st_] = b_; cd_[w_ + 2 * st_] = c_; "
            "cd_[w_ + 3 * st_] = d_; }\n"
            f"    for (; w_ < {words}LL; w_ += st_) cd_[w_] = cs_[w_];\n"
            "  } else {\n/*SCALAR*/\n  }\n}")


@_serialized_lowering
def lower(graph, *, retain_all=False, fuse_elementwise=True, policy="auto", precision_policy=None, execution=None,
          compose_selection="auto", recurrent_gradients="inline", **options):
    """Select one schedule from graph semantics and the declared CUDA target.

    Auto is the normal entry point, including newly captured modules. Reference
    keeps the ordered materialized operation generator for numerical audits.
    Region output bindings select E's direct-output schedule, never a second
    fusion pass on top of A's flat dependency schedule.
    """
    if options.get("loop_schedule") == "vector_resident":
        # GRADLEV-LATCH: opt-in SPMD schedule for forward-only runtime loops (decode / online learning).
        if not (hasattr(graph, "max_steps") and hasattr(graph, "init") and hasattr(graph, "body")
                and hasattr(graph, "output") and not hasattr(graph, "body_vjp")):
            raise LoweringError("loop_schedule='vector_resident' requires a forward-only runtime loop")
        from .vres_loop import lower_vector_resident
        return lower_vector_resident(graph, **{k: v for k, v in options.items() if k != "loop_schedule"})
    if recurrent_gradients not in ("inline", "deferred"):
        raise LoweringError("recurrent_gradients must be inline or deferred")
    if recurrent_gradients == "deferred" and (
            not hasattr(graph, "body_vjp") or getattr(graph, "nested_operation", False)
            or getattr(getattr(graph, "body", None), "nested_operation", False)):
        raise LoweringError("deferred recurrent gradients require one bounded training loop")
    if policy not in ("auto", "reference"):
        raise LoweringError(f"unknown compiler policy {policy}")
    from .execution_plan import current_execution, current_precision, record_composed_regions
    compose = current_execution() == "compose"
    if "static_shared_reserve_bytes" in options:
        from .compose import static_shared_budget
        try:
            static_shared_budget(options["static_shared_reserve_bytes"])
        except ValueError as error:
            raise LoweringError(str(error)) from error
        if options["static_shared_reserve_bytes"]:
            if (not compose or policy != "auto" or compose_selection not in ("auto", "common")
                    or _LOWERING_DEPTH.get() != 1 or retain_all
                    or options.get("target_arch", "sm_90") != "sm_90"
                    or hasattr(graph, "max_steps") or hasattr(graph, "true_branch")
                    or getattr(graph, "nested_operation", False)
                    or _field(graph, "metadata", {}).get("dynamic_batch")
                    or any(key in options for key in ("output_bindings", "region_body", "staged"))):
                raise LoweringError("static shared reserve requires a standalone static common compose program")
        else:
            # Explicit zero is exactly the legacy option set and metadata.
            options.pop("static_shared_reserve_bytes")
    from . import indexed_segmented_outline
    try:
        indexed_segmented_outline.validate_lowering(graph, compose=compose,
            common=compose_selection in ("auto", "common"), depth=_LOWERING_DEPTH.get(),
            retain_all=retain_all, options=options)
    except indexed_segmented_outline.Refusal as error:
        raise LoweringError(str(error)) from error
    from .embedding_backward_contract import validate_local as validate_embedding_arithmetic
    if not compose and validate_embedding_arithmetic(graph):
        raise LoweringError("explicit embedding arithmetic requires execution='compose'")
    from . import torch_layer_norm, torch_layer_norm_backward
    if (torch_layer_norm.KEY in (_field(graph, "metadata", {}) or {})
            and (torch_layer_norm.find_plans(graph) or torch_layer_norm_backward.find_plans(graph))
            and not compose):
        raise LoweringError("explicit CUDA LayerNorm arithmetic requires execution='compose'")
    if (torch_layer_norm_backward.AFFINE_KEY in (_field(graph, "metadata", {}) or {})
            and torch_layer_norm_backward.find_affine_plans(graph) and not compose):
        raise LoweringError("explicit CUDA LayerNorm affine arithmetic requires execution='compose'")
    from .torch_logsoftmax import KEY as logsoftmax_key
    has_logsoftmax = any(str(_field(op,"target",_field(op,"op",""))).removeprefix("torch.ops.").removeprefix("aten.")
                        in {"_log_softmax.default","_log_softmax_backward_data.default"}
                        for op in _field(graph,"operations",_field(graph,"nodes",[])))
    if ((_field(graph,"metadata",{}) or {}).get(logsoftmax_key) not in (None,"legacy")
            and has_logsoftmax and not compose):
        raise LoweringError("explicit CUDA log-softmax arithmetic requires execution='compose'")
    from . import torch_scalar_div
    torch_scalar_div.validate(_field(graph, "metadata", {}) or {})
    from . import torch_column_sum
    if (torch_column_sum.validate(_field(graph, "metadata", {}) or {})
            and not compose and torch_column_sum.graph_has_selection(graph)):
        raise LoweringError("explicit CUDA column-sum arithmetic requires execution='compose'")
    if not compose and torch_scalar_div.graph_has_selection(graph):
        raise LoweringError("explicit CUDA scalar-division arithmetic requires execution='compose'")
    from . import torch_softmax_backward
    selected_softmax_backward = torch_softmax_backward.validate(_field(graph, "metadata", {}) or {})
    if selected_softmax_backward and not compose and any(
            str(_field(op, "target", _field(op, "op", ""))).removeprefix("torch.ops.").removeprefix("aten.")
            == torch_softmax_backward.TARGET
            for op in _field(graph, "operations", _field(graph, "nodes", []))):
        raise LoweringError("explicit CUDA softmax backward arithmetic requires execution='compose'")
    if compose_selection not in ("auto", "common") or (not compose and compose_selection != "auto"):
        raise LoweringError("compose_selection must be auto, or common with execution='compose'")
    # Release step 15: optional caller declaration that gradients are consumed in-program (separate pass).
    from .gradient_consumption import consume as _consume_gradients, record as _record_gradients
    graph, gradient_report = _consume_gradients(graph, options.pop("gradient_outputs", "returned"))
    from . import practical_bf16
    bf16_weight_chunk_k = options.pop("bf16_weight_chunk_k", None)
    practical_bf16.validate(current_precision(), bf16_weight_chunk_k)
    from . import square_sum_association as square_sums
    square_sum_option = options.pop("square_sum_association", None)
    square_sums.validate_option(current_precision(), square_sum_option)
    practical_flat = (_LOWERING_DEPTH.get() == 1 and not hasattr(graph, "max_steps")
                      and not hasattr(graph, "true_branch"))
    if practical_flat and graph is not _STRICT_RETENTION_GRAPH.get():
        graph = practical_bf16.bind(graph, current_precision(), bf16_weight_chunk_k)
    elif bf16_weight_chunk_k is not None:
        raise LoweringError("bf16_weight_chunk_k currently requires a flat training graph")
    if not practical_flat and square_sum_option is not None:
        raise LoweringError("square_sum_association currently requires a flat training graph")
    if practical_flat:
        graph, _ = square_sums.rewrite(graph, option=square_sum_option)
    weight_chunk_report = None
    original_options = dict(options)
    # Release step 3: large-vocabulary heads take padded execution storage and fused row
    # cross-entropy (ops/vocab_head.py) for every schedule; "off" reproduces the base program.
    vocabulary_mode = options.pop("vocabulary_head", "auto")
    vocabulary_report = None
    if _LOWERING_DEPTH.get() == 1 and not hasattr(graph, "max_steps") and not hasattr(graph, "true_branch"):
        if compose:
            # Match the explicit unpadded numerical contract before a generic
            # padded vocabulary rewrite. Distributed capture applies this same
            # shared rule before collectives separate the primal and VJP.
            from .exact_vocab import rewrite as rewrite_exact_vocab
            graph, _exact_vocab_report = rewrite_exact_vocab(graph)
        from .vocab_head import rewrite as rewrite_vocabulary
        graph, vocabulary_report = rewrite_vocabulary(graph, vocabulary_mode)
        graph, _ = practical_bf16.chunk_weights(graph)
        weight_chunk_report = practical_bf16.chunk_record(graph)
        original_options["vocabulary_head"] = "off"
        if compose:
            graph = _rewrite_compose_casts(graph, retain_all=retain_all, policy=policy,
                                          target_arch=options.get("target_arch", "sm_90"))
            # LATCH-MOESUP-ROW (ported by LATCH-MOE4-PARITY): K masked argmax rounds over short rows run as
            # one row-resident selection phase (bit-identical indices; ops/topk_rounds.py).
            from .topk_rounds import rewrite as rewrite_topk_rounds
            graph, _topk_rounds_report = rewrite_topk_rounds(graph)
            if current_precision() == "practical":
                from .row_scatter_mm import rewrite as rewrite_row_products
                graph, _row_products_report = rewrite_row_products(graph, gather_products=True)
    compose_matrix_provider = None
    if precision_policy not in (None, "strict", "practical"):
        raise LoweringError("precision_policy must be strict or practical")
    options = dict(options)
    requested_schedule = options.pop("schedule", "auto")
    schedule_options = options.pop("schedule_options", None)
    compact_codegen = options.pop("compact_codegen", "off")
    if compact_codegen not in ("off", "fold", "elementwise", "full_sum", "elementwise_full_sum",
                               "phase_pointers", "elementwise_full_sum_phase_pointers",
                               "mixed_phase_pointers", "elementwise_full_sum_mixed_phase_pointers", "elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        raise LoweringError("unknown compact_codegen compile-time preference")
    if compose and (requested_schedule != "auto" or schedule_options or compact_codegen == "fold"):
        raise LoweringError("compose execution selects bodies independently; legacy schedule/options/fold cannot override it")
    composed = (hasattr(graph, "max_steps") or getattr(graph, "nested_operation", False)
                or (hasattr(graph, "true_branch") and hasattr(graph, "false_branch")))
    body_sharing = compact_codegen in ("elementwise", "full_sum", "elementwise_full_sum",
                                      "phase_pointers", "elementwise_full_sum_phase_pointers",
                                      "mixed_phase_pointers", "elementwise_full_sum_mixed_phase_pointers", "elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items")
    if body_sharing and (not compose or composed or _LOWERING_DEPTH.get() != 1
            or _field(graph, "metadata", {}).get("dynamic_batch") or retain_all
            or any(key in options for key in ("output_bindings", "region_body", "staged", "barrier_schedule"))):
        raise LoweringError("elementwise code sharing requires a standalone static common compose program")
    nested_flat = _LOWERING_DEPTH.get() > 1 and not composed and not any(
        key in options for key in ("output_bindings", "region_body", "staged", "barrier_schedule"))
    if (composed or nested_flat) and precision_policy == "practical":
        raise LoweringError("composed-region precision must be declared during capture, before differentiation")
    if (composed or nested_flat) and requested_schedule not in ("auto", "reference"):
        raise LoweringError("flat performance schedule override requires a standalone region; "
                            "composed regions do not bind its tensor maps or constant arguments")
    if (composed or nested_flat) and schedule_options:
        raise LoweringError("flat performance schedule options are not admitted inside a composed region")
    flat_rules = {key: options.pop(key) for key in (
        "pops2_policy", "matrix_bodies", "attention_min_keys", "warp_rows",
        "matrix_runtime_layout", "matrix_residency") if key in options}
    # P-OPS2-B BF16 matrix epilogue/operand rules (flat fused schedule only); False reproduces the base.
    pops2b = options.pop("pops2b", True)
    pops2b_rules = {key: options.pop(key) for key in ("pops2b_prefetch", "pops2b_epilogue", "pops2b_permuted",
                                                      "pops2b_casts", "pops2b_idle", "pops2b_plans", "pops2b_defer",
                                                      "pops2b_idle_fill", "pops2b_epi_cost",
                                                      "pops2b_epi_stream_gate", "pops2b_defer_epilogue", "pops2b_barrier")
                    if key in options}
    arch = options.setdefault("target_arch", "sm_90")
    optimized = policy == "auto" and arch == "sm_90"
    reasons = []
    if not optimized:
        reasons.append("explicit reference policy" if policy == "reference" else
                       "donor scheduling/cost model qualified for sm_90 only")
        options.update(matrix_policy="frozen", split_k=False, exact_reductions=False,
                       fuse_attention=False, ordered_index_gradients=False)
    if retain_all:
        options["fuse_attention"] = False
        reasons.append("diagnostic retention preserves every materialized value")
    if not optimized:
        options["scan_engine"] = "generic"
        if precision_policy == "practical" or (schedule_options or {}).get("p8_precision") == "practical":
            raise LoweringError("practical flat arithmetic requires its admitted optimized lowering")
    # P9-D: loop_partition (fusion_reference flat loops) reaches every nested flat region; None keeps the default.
    loop_partition = options.pop("loop_partition", None)
    scalar_hoist = options.pop("scalar_hoist", None)
    reduction_stack = options.pop("reduction_stack", None)
    loop_split = options.pop("loop_split", None)
    child_options = {k: v for k, v in options.items() if k in {
        "target_arch", "matrix_policy", "split_k", "exact_reductions", "fuse_attention",
        "ordered_index_gradients", "matrix_engine", "preserve_full_reductions", "preserve_matrix_plans", "region_expression_inline", "row_fusion", "warp_softmax",
        "batched_reduction_loads", "defer_accumulate", "epilogue_fusion", "write_through", "split_counters",
        "attention_schedule", "attention_store_scores",
        "attention_debug_repeat", "attention_pipeline", "scan_engine", "scan_options"}}
    sampled_options = options.pop("sampled_options", None)
    if _field(graph, "metadata", {}).get("dynamic_batch"):
        # Capacity layouts still use the checked ordinary arena and dependency
        # planner. Active loop bounds have their own admitted body contracts;
        # static-only whole-program fusion must not specialize those bounds.
        if composed or nested_flat or requested_schedule != "auto" or schedule_options or flat_rules:
            raise LoweringError("bounded dynamic batch requires a standalone feedforward schedule")
        program = None
        from . import dynamic_flat
        if dynamic_flat.ENABLED and optimized and fuse_elementwise:
            # LATCH-STEP19-SPEED: the established flat scheduler over capacity layouts with
            # device active extents (fusion, placement, matrix co-scheduling, in-place commits).
            from .fusion_reference import lower_fused as _flat_lower
            from .matrix_rules import MatrixRuleError
            try:
                program = _flat_lower(graph, retain_all=retain_all, pops2b=pops2b,
                                      **{**dynamic_flat.flat_options(pops2b_rules), **options},
                                      dynamic_batch=_field(graph, "metadata", {})["dynamic_batch"])
                schedule = "bounded_dynamic_flat"
                reasons.append("device active extent inside the established flat dependency scheduler; "
                               "capacity storage; one binary and graph for the entire declared domain")
            except (LoweringError, MatrixRuleError) as error:
                program = None
                reasons.append("flat dynamic schedule refused (" + str(error)[:300] + "); ordered dynamic lowering retained")
        if program is None:
            program = _lower_serial(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise,
                **{**options, "barrier_schedule": "dependency", "fuse_attention": False,
                   "elide_stores": True,
                   "row_fusion": False, "epilogue_fusion": False, "write_through": False,
                   "copy_bypass": False, "group_rules": False, "partition_phases": False})
            schedule = "bounded_dynamic_dependency"
            reasons.append("device active extent; capacity storage; one binary and graph for the entire declared domain")
    elif getattr(graph, "nested_operation", False):
        from .nested_loop_lower import lower_nested
        program = lower_nested(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise,
                               policy=policy, **child_options,
                               **{k: options[k] for k in ('output_bindings', 'region_body', 'loop_counter')
                                  if k in options})
        schedule = "bounded_nested_operation"
    elif hasattr(graph, "max_steps") and hasattr(graph, "train") and hasattr(graph, "body"):
        # C-OPS4-B: nondifferentiated sampling loop + one differentiated training region.
        from .sampling_lower import lower_sampled
        program = lower_sampled(graph, options=sampled_options, retain_all=retain_all,
                                fuse_elementwise=fuse_elementwise, policy=policy, **child_options,
                                **({"loop_partition": loop_partition} if loop_partition is not None else {}),
                                **({"scalar_hoist": scalar_hoist} if scalar_hoist is not None else {}),
                                **({"reduction_stack": reduction_stack} if reduction_stack is not None else {}),
                                **({"loop_split": loop_split} if loop_split is not None else {}))
        schedule = "sampled_training"
    elif sampled_options is not None:
        raise LoweringError("sampled_options require a sampled-training graph")
    elif (hasattr(graph, "max_steps") and hasattr(graph, "init") and hasattr(graph, "body")
            and hasattr(graph, "output") and not hasattr(graph, "body_vjp")):
        from .inference_lower import lower_inference_loop
        program = lower_inference_loop(graph, retain_all=retain_all,
                    fuse_elementwise=fuse_elementwise, policy=policy, **child_options)
        schedule = "forward_runtime_loop"
    elif hasattr(graph, "max_steps") and hasattr(graph, "body_vjp"):
        from .loop_lower import lower_loop
        loop_options = dict(options.pop("loop_options", {}) or {})
        replay_passes = (optimized and not retain_all
            and getattr(graph, "storage_mode", None) == "save_body_tape"
            and not getattr(graph, "metadata", {}).get("sequence_specs")
            and not getattr(graph, "metadata", {}).get("nested_loop")
            and not hasattr(getattr(graph, "body_saved", None), "true_branch")
            and not hasattr(getattr(graph, "body_vjp", None), "true_branch"))
        if not replay_passes:
            reasons.append("saved-body rewrites require a plain optimized body with no diagnostic retention or sequence inputs")
        if P9_RNG_REMAT and replay_passes:
            # P9-B: recompute saved Philox masks in the reverse body from its RNG state (ops/rng_remat.py).
            from .loop import BoundedLoopIR as _BLIR
            from .rng_remat import remat_philox_masks
            _data, _report = remat_philox_masks(graph.to_dict())
            if _report.get("recomputed"):
                graph = _BLIR.from_dict(_data)
                reasons.append(f"recomputed {len(_report['recomputed'])} saved Philox masks in the reverse body")
        if P9_TAPE_REMAT and replay_passes:
            from .loop import BoundedLoopIR as _BLIR
            from .tape_remat import remat_tape_products
            _data, _report = remat_tape_products(graph.to_dict())
            if _report.get("recomputed"):
                graph = _BLIR.from_dict(_data)
                reasons.append(f"recomputed {len(_report['recomputed'])} saved elementwise products in the reverse body")
        if P9_TAPE_CSE and replay_passes:
            # P9-B: merge identical pure values of the saved body and drop duplicate tape entries.
            from .loop import BoundedLoopIR as _BLIR
            from .tape_cse import cse_saved_body
            _data, _report = cse_saved_body(graph.to_dict())
            if _report.get("merged"):
                graph = _BLIR.from_dict(_data)
                reasons.append(f"saved-body CSE merged {len(_report['merged'])} values, "
                               f"{len(_report.get('tape_dropped', []))} tape entries")
        if (recurrent_gradients == "deferred" and replay_passes and graph.max_steps > 0
                and loop_options.get("pipeline", "auto") is not False):
            from .loop import BoundedLoopIR as _BLIR
            from .loop_defer_grads import defer_weight_gradients
            _data, _report = defer_weight_gradients(graph.to_dict())
            if _report.get("deferred"):
                graph = _BLIR.from_dict(_data)
                reasons.append(f"deferred {len(_report['deferred'])} shared-weight gradient products to one "
                               "post-loop stage in the original per-step order")
        if P9_TRANSPOSED_CASTS and replay_passes and getattr(graph, "invariant_casts", None) is not None:
            # P9-B: K-major copies of the invariant weight casts that forward products read N-major.
            from .loop import BoundedLoopIR as _BLIR
            from .transposed_casts import transpose_forward_casts
            _data, _report = transpose_forward_casts(graph.to_dict())
            if _report.get("transposed"):
                graph = _BLIR.from_dict(_data)
                reasons.append(f"transposed invariant casts for {_report['products']} forward products")
        if optimized and not retain_all and "group_rules" not in loop_options:
            metadata = getattr(graph, "metadata", {})
            attention = metadata.get("attention_tape", {}).get("groups", [])
            if metadata.get("sequence_specs"):
                # C-OPS4-A found the V4 set racing on this region shape; P8-C isolated the race to "part".
                from .execution_policy import SEQUENCE_RULES
                declared = metadata.get("precision_policy") or {}
                if declared.get("attention") == "bfloat16" and declared.get("name") != "mixed_bf16_online":
                    loop_options["group_rules"] = SEQUENCE_RULES
                    reasons.append("declared mixed-BF16 sequence regions use the qualified nonpartitioned schedule; tensor maps remain outside the iteration body")
                else:
                    loop_options["group_rules"] = "idx32+idxid"
                    reasons.append("unqualified sequence arithmetic retains the conservative schedule; prior partitioning failures remain excluded")
            elif (graph.storage_mode == "save_body_tape" and attention
                    and not hasattr(graph.body_saved, "true_branch")):
                from .execution_policy import RECURRENT_RULES
                loop_options["group_rules"] = RECURRENT_RULES + (
                    "+sink" if metadata.get("tape_policy") == "storage_remat" else "")
                reasons.append("admitted saved-loop attention and matrices use the qualified recurrent schedule")
        region_overrides = loop_options.get("region_options", {})
        if not isinstance(region_overrides, dict):
            raise LoweringError("loop region_options must be a compiler-option mapping")
        loop_options.update(region_policy=policy, region_options={**child_options, **region_overrides})
        if compose and optimized and not retain_all:
            # Region tensor maps already have per-instance bindings. Body
            # eligibility need not depend on the presence of an attention group.
            common = "p5c+p5cw+p5ce+vec4+vec4u+idx32+idxid"
            loop_options.setdefault("group_rules", common)
            reasons.append("shared body planner; only self-contained elementwise phases admit CTA partitioning")
        if not optimized:
            loop_options["pipeline"] = False
        from .group_rules import features_of
        rule_features = features_of(loop_options.get("group_rules"))
        # P8-C: sequence loops admit tensor-map products in their once-per-update regions only; the loop composer
        # removes them from the per-iteration body regions (loop_pipeline.region_rules).
        token = (_INDEX_MODE.set("u32id" if "idxid" in rule_features else "u32") if "idx32" in rule_features else None)
        from .group_rules import FAST_ACTIVATION
        fast_token = FAST_ACTIVATION.set(
            (getattr(graph, "metadata", {}).get("precision_policy") or {}).get("activation") == "fast")
        events = _PHASE_EVENTS.set(True) if loop_options.get("phase_ticks") else None
        try:
            program = lower_loop(graph, retain_all=retain_all,
                                 fuse_elementwise=fuse_elementwise, options=loop_options)
        finally:
            FAST_ACTIVATION.reset(fast_token)
            if token is not None:
                _INDEX_MODE.reset(token)
            if events is not None:
                _PHASE_EVENTS.reset(events)
        schedule = program.manifest["loop_schedule_policy"]["selected"]
    elif hasattr(graph, "true_branch") and hasattr(graph, "false_branch"):
        program = lower_conditional(graph, retain_all=retain_all,
                    fuse_elementwise=fuse_elementwise, policy=policy, **child_options)
        schedule = "uniform_conditional"
    elif compose and optimized and not retain_all:
        from .compose import (flat_selection, lower_composed_flat, verify_legacy_long_protocol,
                              verify_legacy_matrices, verify_legacy_reductions, view_read_hazards)
        schedule = "composed_dependency"
        depth = _LOWERING_DEPTH.get()
        if conditional_policy()["branch_selection"] and depth > 1 and _CONDITIONAL_BRANCH.get() == depth:
            depth = 1  # STEP-11: a uniform-conditional branch is a standalone complete flat program.
        if loop_region_policy()["ordered_region_selection"] and depth > 1 and _ORDERED_LOOP_REGION.get() == depth:
            depth = 1  # LATCH-LOOP-GAP: an ordered loop region is a standalone complete flat program.
        program, compose_matrix_provider = lower_composed_flat(graph, fuse_elementwise=fuse_elementwise,
            selection_probe=compose_selection == "auto" and depth == 1, **options)
        selection = flat_selection(graph, program, mode=compose_selection, depth=depth,
                                   explicit_options=set(original_options) - {"target_arch", "compact_codegen", "vocabulary_head"},
                                   fuse_elementwise=fuse_elementwise)
        # LATCH-EXPLAIN: a later retention refusal replaces selection["reasons"]; keep the initial record for reports.
        _explain_note("selection_snapshot", stage="initial_flat_selection", selection=selection)
        if selection["selected"] == "legacy_static":
            if body_sharing:
                raise LoweringError("elementwise code sharing requires common_static execution; "
                                    "set compose_selection='common' to request it explicitly")
            from .execution_plan import execution_scope
            # A practical declaration permits an exact implementation. The
            # captured BF16 casts/activation contract is kept, and the actual
            # established bodies are checked before accepting this schedule.
            effective_precision = ("strict" if selection.get("bf16_retention_candidate")
                                   and not selection.get("p6d_matrix_retention_candidate") else precision_policy)
            # LATCH-STEP-04: carry the common program's indexed engine into the static emitter.
            from contextlib import nullcontext
            from .indexed_compose import engine_of
            from .indexed_reduction import policy as indexed_policy
            indexed_engine = engine_of(program.manifest)
            try:
                # Stay at the original top-level depth and under the existing
                # shared lock. A decorated recursive call would look like a
                # nested region and would not reproduce normal legacy lowering.
                with _strict_retention_scope(graph, effective_precision), execution_scope("legacy", effective_precision), (
                        indexed_policy(indexed_engine) if indexed_engine else nullcontext()):
                    legacy = lower.__wrapped__(graph, retain_all=retain_all,
                        fuse_elementwise=fuse_elementwise, policy=policy,
                        precision_policy=effective_precision, execution="legacy", **original_options)
                    if indexed_engine:
                        from .indexed_compose import choose_gather_inlining
                        legacy = choose_gather_inlining(legacy, lambda: lower.__wrapped__(graph, retain_all=retain_all,
                            fuse_elementwise=fuse_elementwise, policy=policy,
                            precision_policy=effective_precision, execution="legacy", **original_options))
            except LoweringError as error:
                selection["selected"] = "common_static"
                selection["reasons"] = ["established static emitter refused: " + str(error)]
            else:
              if engine_of(legacy.manifest) != indexed_engine:
                selection["selected"] = "common_static"
                selection["reasons"] = ["established static emitter did not admit the selected indexed engine"]
              else:
                hazards = view_read_hazards(graph, legacy.manifest)
                reductions, refusal = verify_legacy_reductions(program.manifest, legacy.manifest,
                    selection.get("reduction_retention_candidates", []))
                matrices = []
                if not refusal:
                    matrices, refusal = verify_legacy_matrices(program.manifest, legacy.manifest,
                        selection.get("matrix_retention_candidates", []),
                        contracts=getattr(legacy, "_selection_contracts", None),
                        common_contracts=getattr(program, "_selection_contracts", None))
                protocol = None
                if selection.get("protocol_retention_candidate") and not refusal:
                    protocol, refusal = verify_legacy_long_protocol(graph, program.manifest, legacy.manifest)
                    if (not refusal and selection["protocol_retention_candidate"] == "p6d_explicit_flat"):
                        from .compose import step13_would_upgrade
                        if step13_would_upgrade(compose_matrix_provider):
                            refusal = ("step-13 structural default upgrades products of this explicit flat program to "
                                       "admitted provider bodies the established emitter lacks")
                bf16 = None
                p6d_matrices = None
                if selection.get("p6d_matrix_retention_candidate") and not refusal:
                    from .p6d_retention import verify as verify_p6d_matrices
                    from .execution_plan import current_precision
                    p6d_matrices, refusal = verify_p6d_matrices(graph, program.manifest, legacy.manifest,
                        contracts=getattr(legacy, "_selection_contracts", None), policy=current_precision())
                elif selection.get("bf16_retention_candidate") and not refusal:
                    from .bf16_retention import verify_legacy_bf16
                    bf16, refusal = verify_legacy_bf16(graph, program.manifest, legacy.manifest,
                        contracts=getattr(legacy, "_selection_contracts", None))
                if any(hazards.values()):
                    selection["selected"] = "common_static"
                    selection["legacy_view_hazards"] = hazards
                    selection["reasons"] = ["established emitter has new-value view reads requiring common safety"]
                elif refusal:
                    selection["selected"] = "common_static"
                    selection["reasons"] = [refusal]
                else:
                    if reductions:
                        selection["retained_reductions"] = reductions
                        selection["max_grid_blocks"] = 1024
                        selection["reasons"] = ["no additional body or common-only safety requirement; established full reductions match engine, output, extent, arithmetic and scratch protocol"]
                    if protocol:
                        selection["retained_protocol"] = protocol
                        selection["reasons"] = ["established homogeneous protocol covers every admitted long-attention group and P5C body with the same declared arithmetic"
                            if protocol.get("attention_groups") else
                            "established P6D schedule covers every admitted P5C body of this attention-free explicit-cast program with the same declared arithmetic"]
                    if matrices:
                        selection["retained_matrices"] = matrices
                        selection["reasons"] = ["established static schedule covers every admitted POPS2 split-K body with matching calculation, arithmetic, operands and disjoint counter/partial bindings"]
                    if bf16:
                        from .execution_plan import current_precision
                        selection.update(retained_bf16=bf16, requested_precision_policy=current_precision(),
                                         effective_precision_policy="strict")
                        selection["reasons"] = ["established exact schedule preserves every captured BF16 product, operand view and cast rounding; practical precision permits this strict implementation"]
                    if p6d_matrices:
                        from .execution_plan import current_precision
                        selection.update(retained_p6d_matrices=p6d_matrices,
                            requested_precision_policy=current_precision(), effective_precision_policy=current_precision())
                        selection["reasons"] = ["established flat P6D schedule covers admitted matrix bodies, cast/view aliases, rounded epilogues, state stores and synchronization under the requested precision"]
                    from .compose import long_protocol_common_upgrade
                    upgraded = long_protocol_common_upgrade(compose_matrix_provider, legacy.manifest, protocol,
                        lambda: lower_composed_flat(graph, fuse_elementwise=fuse_elementwise, **options))
                    if upgraded is None:
                        legacy.manifest["execution_selection"] = selection
                        if vocabulary_report is not None:
                            legacy.manifest["vocabulary_head"] = vocabulary_report
                        execution_precision = _field(graph, "metadata", {}).get(practical_bf16.KEY)
                        if execution_precision is not None:
                            legacy.manifest[practical_bf16.KEY] = execution_precision
                        if weight_chunk_report is not None:
                            legacy.manifest[practical_bf16.CHUNKS] = weight_chunk_report
                        return _record_gradients(square_sums.annotate_program(graph, legacy), gradient_report)
                    program, compose_matrix_provider, rule = upgraded
                    selection = flat_selection(graph, program, mode=compose_selection, depth=depth,
                        explicit_options=set(original_options) - {"target_arch", "compact_codegen", "vocabulary_head"},
                        fuse_elementwise=fuse_elementwise)
                    selection.update(selected="common_static", step13_default_rule=rule,
                        performance_selection={"rule": "homogeneous_long_without_p5c_step13_products",
                            "established_source_hash": legacy.cache_key, "verified_protocol": protocol},
                        reasons=["verified long-attention protocol has no P5C products; retain existing admitted step-13 matrix upgrades"])
        if selection["selected"] == "common_static":
            deferred = program.manifest.get("deferred_column_splits", [])
            deferred_stats = program.manifest.get("deferred_vocabulary_stats", [])
            if deferred or deferred_stats:
                # This is a body upgrade of an already selected common
                # program, not a proof that two reduction trees are equal.
                prior_selection = selection
                if compose_matrix_provider is not None and compose_matrix_provider.inventory is not None:
                    del compose_matrix_provider.inventory[compose_matrix_provider.inventory_start:]
                program, compose_matrix_provider = lower_composed_flat(
                    graph, fuse_elementwise=fuse_elementwise, **options)
                selection = flat_selection(graph, program, mode=compose_selection, depth=depth,
                    explicit_options=set(original_options) - {"target_arch", "compact_codegen", "vocabulary_head"},
                    fuse_elementwise=fuse_elementwise)
                selection.update(selected="common_static", common_body_upgrade={
                    "column_splits": deferred, "prior_selection": prior_selection})
                if deferred_stats:
                    selection["common_body_upgrade"]["vocabulary_stats"] = deferred_stats
            # Release step 13: structural default for programs that already need the common planner.
            from .compose import step13_common_upgrade
            common_body_upgrade = selection.get("common_body_upgrade")
            upgraded = step13_common_upgrade(compose_matrix_provider, lambda: lower_composed_flat(
                graph, fuse_elementwise=fuse_elementwise, **options))
            if upgraded is not None:
                program, compose_matrix_provider, rule = upgraded
                selection = flat_selection(graph, program, mode=compose_selection, depth=_LOWERING_DEPTH.get(),
                                           explicit_options=set(original_options) - {"target_arch", "compact_codegen", "vocabulary_head"},
                                           fuse_elementwise=fuse_elementwise)
                selection.update(selected="common_static", step13_default_rule=rule)
                if common_body_upgrade is not None:
                    selection["common_body_upgrade"] = common_body_upgrade
        program.manifest["execution_selection"] = selection
    elif optimized and not retain_all and fuse_elementwise and not any(
            k in options for k in ("output_bindings", "region_body", "staged", "barrier_schedule")):
        has_nested_scan = nested_flat and any(
            str(_field(op, "target", _field(op, "op"))) in _scan_targets()
            for op in _field(graph, "operations", _field(graph, "nodes", ())))
        if has_nested_scan:
            if options.get("scan_engine", "auto") not in ("auto", "generic"):
                raise LoweringError("forced scan engine is not admitted inside a composed region; "
                                    "the generic scan has no flat tensor-map bindings")
            program = _lower_serial(graph, retain_all=False, fuse_elementwise=fuse_elementwise,
                                    **{**options, "scan_engine": "generic"})
            schedule = "nested_ordered_scan"
            reasons.append("nested scan uses ordered generic lowering; flat tensor-map bindings are not composed")
        else:
            from .fusion import lower_fused
            program = lower_fused(graph, retain_all=False, pops2b=pops2b,
                                  schedule="reference" if nested_flat else requested_schedule,
                                  schedule_options=schedule_options,
                                  precision_policy=precision_policy,
                                  compact_codegen=compact_codegen, **pops2b_rules,
                                  **flat_rules, **options,
                                  **({"loop_partition": loop_partition} if loop_partition is not None else {}),
                                  **({"scalar_hoist": scalar_hoist} if scalar_hoist is not None else {}),
                                  **({"reduction_stack": reduction_stack} if reduction_stack is not None else {}),
                                  **({"loop_split": loop_split} if loop_split is not None else {}))
            schedule = "fused_dependency"
            if nested_flat:
                reasons.append("nested flat region retains the generic dependency schedule; "
                               "flat tensor-map and constant-argument bindings are not composed")
    else:
        program = _lower_serial(graph, retain_all=retain_all,
                                fuse_elementwise=fuse_elementwise, **options)
        schedule = "staged_direct_output" if options.get("region_body") else "ordered_operations"
    if compact_codegen == "fold" and schedule != "fused_dependency":
        reasons.append("compact generation applies only to admitted flat dependency programs; unchanged lowering")
    if flat_rules and schedule != "fused_dependency":
        reasons.append("flat scheduling overrides do not apply to this ordered or region schedule")
    program.manifest.update(compiler_policy="integrated_v3" if policy == "auto" else "reference",
        target_arch=arch, compiler_selection={"policy": policy, "schedule": schedule,
            "target_arch": arch, "reasons": reasons or ["eligible operation graph and layouts"],
            "options": child_options, "flat_diagnostic_options": flat_rules,
            "dispatch": "graph, layout, dtype and target; no model names"})
    if compose:
        program.manifest.setdefault("execution_selection", {
            "version": 1, "requested": "compose", "mode": compose_selection,
            "selected": "common_static", "reasons": ["region or diagnostic retains its existing common lowering"]})
        if "execution_plan" not in program.manifest:
            record_composed_regions(program.manifest)
        program.manifest["execution_plan"]["capture_precision"] = _field(graph, "metadata", {}).get("precision_policy")
        if _LOWERING_DEPTH.get() == 1:
            from .p5c_matrix import finalize
            program = finalize(program)
            if compose_matrix_provider is not None:
                # P5C starts the descriptor table; the local fallback provider
                # appends its own maps after those bindings have been resolved.
                program = compose_matrix_provider.finalize(program)
            # LATCH-TINYMOE-XMAT: TMA-fed segmented products in programs without provided matrix products.
            from .ragged_experts_lower import finalize_maps as _ragged_finalize_maps
            program = _ragged_finalize_maps(program)
            if not composed:
                program.manifest["full_grid"] = True
    if vocabulary_report is not None:
        program.manifest["vocabulary_head"] = vocabulary_report
    execution_precision = _field(graph, "metadata", {}).get(practical_bf16.KEY)
    if execution_precision is not None:
        program.manifest[practical_bf16.KEY] = execution_precision
    if weight_chunk_report is not None:
        program.manifest[practical_bf16.CHUNKS] = weight_chunk_report
    if hasattr(graph, "body_vjp"):
        program.manifest["recurrent_gradients"] = recurrent_gradients
    program = _record_gradients(square_sums.annotate_program(graph, program), gradient_report)
    if compact_codegen in ("elementwise", "elementwise_full_sum", "elementwise_full_sum_phase_pointers",
                           "elementwise_full_sum_mixed_phase_pointers", "elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .elementwise_codegen import fold, Refusal
        if program.manifest.get("execution_selection", {}).get("selected") != "common_static":
            raise LoweringError("elementwise code sharing requires common_static execution selection")
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("elementwise code sharing refused: " + str(error)) from error
    if compact_codegen in ("full_sum", "elementwise_full_sum", "elementwise_full_sum_phase_pointers",
                           "elementwise_full_sum_mixed_phase_pointers", "elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .full_sum_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("full-sum code sharing refused: " + str(error)) from error
    if compact_codegen in ("phase_pointers", "elementwise_full_sum_phase_pointers"):
        from .phase_pointer_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("phase pointer localization refused: " + str(error)) from error
    if compact_codegen in ("mixed_phase_pointers", "elementwise_full_sum_mixed_phase_pointers", "elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .mixed_phase_pointer_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("mixed phase pointer localization refused: " + str(error)) from error
    if compact_codegen in ("elementwise_full_sum_resource_phase_pointers", "elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .resource_phase_pointer_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("resource phase pointer localization refused: " + str(error)) from error
    if compact_codegen in ("elementwise_full_sum_resource_pure_intervals", "elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .pure_interval_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("pure interval outlining refused: " + str(error)) from error
    if compact_codegen in ("elementwise_full_sum_resource_pure_runs", "elementwise_full_sum_resource_pure_runs_attention_items"):
        from .pure_run_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("pure run outlining refused: " + str(error)) from error
    if compact_codegen == "elementwise_full_sum_resource_pure_runs_attention_items":
        from .attention_item_codegen import fold, Refusal
        try:
            program = fold(program)
        except Refusal as error:
            raise LoweringError("attention-item outlining refused: " + str(error)) from error
    from . import update_schedule as _update_schedule
    if _update_schedule.current() is not None and _LOWERING_DEPTH.get() == 1:
        if program.manifest.get("barrier_schedule") != "dependency":
            raise LoweringError("update_schedule requires a dependency-scheduled flat compose program")
        if _update_schedule.prologue_tables():
            # LATCH-TINYMOE-PROLOGUE: table phases for runs of tiny prologue phases (ops.prologue_batch).
            from . import prologue_batch as _prologue_batch
            try:
                program = _prologue_batch.batch(program)
            except _prologue_batch.Refusal as error:
                raise LoweringError("prologue batching refused: " + str(error)) from error
        try:
            program = _update_schedule.rotate(program)
        except _update_schedule.Refusal as error:
            raise LoweringError("update schedule rotation refused: " + str(error)) from error
    try:
        program = indexed_segmented_outline.annotate(program)
    except indexed_segmented_outline.Refusal as error:
        raise LoweringError(str(error)) from error
    return program


@attention_lower.retry_lowering
def _lower_serial(graph, *, retain_all=False, fuse_elementwise=True, loop_options=None, output_bindings=None,
          region_body=False, barrier_schedule="phase", elide_stores=False, staged=False,
          partition_phases=False, matrix_policy="auto", split_k=True, exact_reductions=True,
          fuse_attention=True, ordered_index_gradients=True, target_arch="sm_90", matrix_engine=None,
          split_counters="phase", row_fusion=False, warp_softmax=False, batched_reduction_loads=False,
          defer_accumulate=False, matrix_grids=None, epilogue_fusion=False, write_through=False, defer_policy="earliest",
          copy_bypass=False, attention_schedule=True, attention_store_scores=True,
          attention_debug_repeat=False, attention_pipeline=True, group_rules=False,
          scan_engine="auto", scan_options=None, matrix_provider=None, preserve_full_reductions=False,
          preserve_matrix_plans=False, region_expression_inline=False):
    """Generate without CUDA or a GPU. Static FP32 tensor shapes are required.

    Accepted graph fields are values, operations (or nodes), inputs, outputs,
    and state_writes. ATen arguments encode tensor references as {"ref": name}.
    Each operation has target, args, kwargs, and one output/name.
    loop_options applies only to bounded loops; see loop_lower.LOOP_OPTIONS.
    """
    if hasattr(graph, "max_steps") and hasattr(graph, "body_vjp"):
        from .loop_lower import lower_loop
        return lower_loop(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise, options=loop_options)
    if loop_options:
        raise LoweringError("loop_options require a bounded loop graph")
    if hasattr(graph, "true_branch") and hasattr(graph, "false_branch"):
        return lower_conditional(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise)
    from .group_rules import features_of
    features = features_of(group_rules)
    from .execution_plan import current_execution, current_precision, record_plan, requires_grid_fence
    compose = current_execution() == "compose"
    from . import square_sum_association as square_sums
    balanced_square_sums = square_sums.selected_full_sums(graph)
    if type(preserve_full_reductions) is not bool:
        raise LoweringError("preserve_full_reductions must be boolean")
    if preserve_full_reductions and not (staged and region_body):
        raise LoweringError("preserving region full reductions requires staged region output bindings")
    if type(preserve_matrix_plans) is not bool:
        raise LoweringError("preserve_matrix_plans must be boolean")
    if preserve_matrix_plans and not (staged and region_body):
        raise LoweringError("preserving region matrix plans requires staged region output bindings")
    if type(region_expression_inline) is not bool:
        raise LoweringError("region_expression_inline must be boolean")
    if region_expression_inline and not (staged and region_body):
        raise LoweringError("region expression inlining requires a staged region")
    raw_values = _field(graph, "values")
    values = raw_values if isinstance(raw_values, dict) else {_field(v, "name"): v for v in raw_values}
    dynamic_declaration = _field(graph, "metadata", {}).get("dynamic_batch")
    if region_expression_inline and dynamic_declaration:
        raise LoweringError("region expression inlining currently requires static extents")
    dynamic = None
    if dynamic_declaration:
        from .dynamic_codegen import ActiveExtents
        dynamic = ActiveExtents(dynamic_declaration, values)
        # Static state-update phases keep the existing vector body. Dynamic
        # phases below use an extent-aware loop, never a maximum-sized vector.
        features = {"vec4"}
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
    exact_bf16_producer = None   # LATCH-BUILD-600: built on first use, see _exact_bf16_operands
    from .embedding_backward_contract import validate_local as validate_embedding_arithmetic
    embedding_arithmetic = validate_embedding_arithmetic(graph)
    if embedding_arithmetic and (not compose or not ordered_index_gradients or staged or region_body):
        raise LoweringError("explicit embedding arithmetic requires ordinary common indexed lowering")
    writes = _field(graph, "state_writes", {})
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
    roots = []
    virtual, virtual_text, virtual_reads = set(), {}, {}   # P8-C vprod (filled before the op loop)
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
    phases = []
    producers = {}
    views = []
    helper_parts = []
    extra_headers = []
    matrix_decisions = []
    scalar_division_decisions = {}
    pops2_layouts = {}
    if split_counters not in ("phase", "persistent"):
        raise LoweringError(f"unknown split-K counter placement {split_counters}")
    # P-OPS2-D: staged regions may place split-K arrival counters in a caller-owned
    # persistent area (zeroed once per launch, restored to zero by each last
    # arriver), removing the counter-zero producer phase before every product.
    persistent_counters = split_counters == "persistent" and bool(staged)
    counter_top = 0

    def read(arg, shape, idx="i", register_values=None):
        ref = _ref(arg, names)
        if ref is None:
            if isinstance(arg, (int, float, bool)) or (dynamic and isinstance(arg, dict) and "dynamic_batch_scalar" in arg):
                return _float(arg)
            raise LoweringError(f"unsupported scalar/tensor argument {arg!r}")
        layout = layouts[ref]
        if dynamic and dynamic.axes.get(ref, 0) != 0:
            raise LoweringError(f"{ref}: nonleading active view is admitted only by matrix/reduction addressing")
        if layout.root in virtual_text:
            # P8-C vprod: the producer is not stored; its rounded expression is evaluated at the element this
            # reader addresses (its flat index in the producer's contiguous result).
            flat = _index(layout, tuple(shape), idx)
            flat = f"({layout.offset}LL + {flat})" if layout.offset else f"({flat})"
            text = re.sub(r"\bi\b", flat, virtual_text[layout.root])
            return f"__bfloat162float({text})" if dtypes[layout.root] == "bfloat16" else text
        if register_values and ref in register_values and layout.shape == tuple(shape):
            expression = register_values[ref]
        elif (compose and register_values and layout.root in register_values
              and layout.shape == tuple(shape) and layout.contiguous and not layout.offset
              and layouts[layout.root].shape == tuple(shape)):
            # An identity view of an earlier value in this same elementwise
            # loop observes its new register value. Keeping a memory load here
            # lets vector emission prefetch it before the producer's store,
            # including when last-use placement aliases that store to state.
            expression = register_values[layout.root]
        else:
            expression = f"{_ident(ref)}[{_index(layout, tuple(shape), idx)}]"
        return f"__bfloat162float({expression})" if dtypes[ref] == "bfloat16" else expression

    def rounded(expression, dtype):
        return f"__float2bfloat16_rn({expression})" if dtype == "bfloat16" else expression

    def output_name(op):
        result = _field(op, "output", _field(op, "name"))
        if result is None:
            outs = _field(op, "outputs")
            if len(outs) != 1:
                raise LoweringError("multi-result ATen operation must be decomposed")
            result = outs[0]
        return _ref(result, names) or result

    # TINYMOE-SELECTIVE-CADENCE, group rule ``ustate`` (ops.update_schedule "streamed_ustate"): a committed
    # parameter/state value that is only an exact copy of its own old binding (a tensor rule returned the input
    # unchanged and capture materialized it with clone) commits nothing. Its write is dropped, its returned
    # output names the unchanged binding itself, and the now unread copy chain is removed. The binding's bytes
    # are left exactly as the copy would have left them; other values and their order are untouched.
    unchanged_record = None
    if "ustate" in features and not retain_all and not staged and not region_body:
        op_by_out_ = {output_name(op): op for op in operations}
        copy_bases_ = {"clone", "contiguous", "alias", "detach", "lift_fresh_copy"}
        replaced_, chain_values_ = {}, set()
        for target in list(writes):
            source, chain_ = writes[target], []
            while source in op_by_out_ and source not in chain_:
                op = op_by_out_[source]
                base = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
                args_, kwargs_ = list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))
                fmt = str(kwargs_.get("memory_format") or "contiguous_format").removeprefix("torch.")
                if (base not in copy_bases_ or len(args_) != 1 or set(kwargs_) - {"memory_format"}
                        or fmt not in ("contiguous_format", "preserve_format")):
                    break
                chain_.append(source)
                source = _ref(args_[0], names)
            if source == target and chain_:
                del writes[target]
                replaced_.update({value: target for value in chain_})
                chain_values_.update(chain_)
        if replaced_:
            outputs = [replaced_.get(name, name) for name in outputs]
            if isinstance(structured_outputs, dict):
                structured_outputs = {kind: ({k: replaced_.get(_ref(v, names) or v, v) for k, v in group.items()}
                                             if isinstance(group, dict) else replaced_.get(_ref(group, names) or group, group))
                                      for kind, group in structured_outputs.items()}
            consumers_ = {}
            for op in operations:
                for r in set(_flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)):
                    consumers_[r] = consumers_.get(r, 0) + 1
            kept_ = set(outputs) | set(writes.values()) | set(writes)
            dropped_, changed_ = set(), True
            while changed_:
                changed_ = False
                for op in operations:
                    value = output_name(op)
                    if value in dropped_ or value not in chain_values_ or value in kept_ or consumers_.get(value, 0):
                        continue
                    dropped_.add(value)
                    changed_ = True
                    for r in set(_flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)):
                        consumers_[r] -= 1
            operations = [op for op in operations if output_name(op) not in dropped_]
            targets_ = sorted(set(replaced_.values()))
            unchanged_record = {"writes": len(targets_), "removed_copies": len(dropped_),
                                "targets_sha256": hashlib.sha256(json.dumps(targets_).encode()).hexdigest()}

    # Step 10: tap-shifted convolution operands (ops/conv_ops.py TapsOperands); deferred, read in place by
    # admitted flat p5c products, materialized before any other reader.
    from . import conv_ops as _conv_ops
    conv_taps = _conv_ops.TapsOperands(enabled=bool(compose) and _LOWERING_DEPTH.get() == 1 and not staged
                                       and not region_body and barrier_schedule == "dependency" and not retain_all
                                       and _conv_ops.FORM != "patches")

    def materialize_taps(root, op, refs, args, kwargs):
        kind, code, extra, _ = _conv_ops.emit(conv_taps.targets.get(root, _conv_ops.TAPS_TARGET), args, kwargs, root,
                                              tuple(layouts[root].shape), layouts, dtypes, names)
        if root not in placement and root not in roots:
            roots.append(root)
        add_phase(kind, op, root, code, refs, **extra)

    def add_phase(kind, op, name, code, refs, **extra):
        if conv_taps.deferred:
            conv_taps.before_phase(refs, layouts, materialize_taps)
        phase = {"kind": kind, "names": [name], "targets": [str(_field(op, "target", _field(op, "op")))],
                 "reads": list(dict.fromkeys(refs)), "writes": [name], "code": code, **extra}
        phases.append(phase)
        producers[name] = len(phases) - 1

    # Functionalization can represent a fresh filled tensor as empty_like + fill.
    # The allocation's contents have no semantics. Admit that metadata-only use
    # narrowly, and omit its storage/view from the runtime so diagnostic clients
    # cannot accidentally compare uninitialized bytes as a numerical result.
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
    placement, elided, zero_contributions = _plan_output_placement(
        operations, values, inputs, outputs, output_bindings, undefined, names, output_name,
        elide_zero="zelide" in features and not retain_all)
    accumulate = {name for name, (_, mode) in placement.items() if mode == "accumulate"}

    attention_decisions, index_decisions = [], []
    indexed_occurrence_cache = {}
    # LATCH-DISPATCH-GRAD: index facts proved from operation semantics (ragged_order permutations, identity
    # copies, floor division by a static k); ordinary common composition only, never frozen/baseline policies.
    permutation_facts = {}
    if compose and ordered_index_gradients and not staged and not region_body:
        from . import permutation_gradients
        from .indexed_reduction import requested_policy as _requested_indexed_policy
        if permutation_gradients.ENABLED and _requested_indexed_policy() not in ("frozen_base", "baseline"):
            permutation_facts = permutation_gradients.analyze(operations, values, names)
    # P-OPS2-D write-through: a product read only through a view chain ending in
    # a contiguous copy may store its result directly in the copy's layout.
    write_plans, write_done = {}, set()
    # P-OPS2-D copy bypass: products may read a contiguous copy's source through
    # a factored layout, so they need not wait for the copy itself.
    copy_sources = {}
    if copy_bypass and not retain_all and not preserve_matrix_plans:
        for op in operations:
            target_name = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
            fmt = str(dict(_field(op, "kwargs", {})).get("memory_format") or "contiguous_format").removeprefix("torch.")
            if target_name in ("clone", "contiguous") and fmt == "contiguous_format":
                source_name = _ref(list(_field(op, "args", []))[0], names)
                if source_name is not None:
                    copy_sources[output_name(op)] = source_name
    # Factored write-through currently replans through the B engine. Keep the
    # ordinary product and explicit copy when its original plan must survive.
    if write_through and not retain_all and not preserve_matrix_plans:
        view_bases = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape",
                      "squeeze", "unsqueeze"}
        op_by_out = {output_name(op): op for op in operations}
        basename = lambda op: str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        consumer_map = {}
        for op in operations:
            for r in set(_flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)):
                consumer_map.setdefault(r, []).append(output_name(op))
        protected_values = set(outputs) | set(writes.values()) | set(placement) | set(writes)
        for op in operations:
            copy = output_name(op)
            if basename(op) not in ("clone", "contiguous") or copy in protected_values and copy not in placement:
                continue
            # The factored write-through epilogue stores, but does not accumulate.
            # Keep the explicit accumulating copy for caller-bound gradients;
            # otherwise every reverse iteration replaces the previous sum.
            if copy in accumulate:
                continue
            fmt = str(dict(_field(op, "kwargs", {})).get("memory_format") or "contiguous_format").removeprefix("torch.")
            if fmt != "contiguous_format":
                continue
            chain, source = [], _ref(list(_field(op, "args", []))[0], names)
            while source in op_by_out and basename(op_by_out[source]) in view_bases:
                chain.append(op_by_out[source])
                source = _ref(list(_field(op_by_out[source], "args", []))[0], names)
            if not chain or source not in op_by_out or basename(op_by_out[source]) != "mm":
                continue
            sequence = [source, *(output_name(v) for v in reversed(chain)), copy]
            if any(value in protected_values for value in sequence[:-1]) or source in accumulate:
                continue
            if all(consumer_map.get(sequence[k], []) == [sequence[k + 1]] for k in range(len(sequence) - 1)):
                write_plans[source] = {"copy": copy, "chain": list(reversed(chain))}
    # P-OPS4-E: halves of short-attention groups split across a saved-body loop tape
    # (ops/loop_attention.py). They take precedence over P-OPS-D matching of the same score product.
    # P7-C p5c: readers of every value, for the exact FP32-converting product store rule.
    value_readers = {}
    for op in operations:
        for ref in dict.fromkeys(_flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)):
            value_readers.setdefault(ref, []).append(op)
    standalone_text, standalone_refs = {}, {}     # LATCH-STEP10-SPEED: reductions over single-use elementwise values
    # LATCH-STEP10-SPEED: rules that reassociate a reduction or change the interval schedule apply only to programs that
    # went through capture-time channels-last propagation (pixel-major convolutions); other programs keep their
    # established numerics and schedules until those rules are measured on them.
    conv_program = _CONV_PROGRAM.get()
    inlined_reductions = []
    converted_products = {}
    loop_plans, loop_refusals = [], []
    if fuse_attention and target_arch == "sm_90" and not retain_all:
        from . import loop_attention
        loop_plans, loop_refusals = loop_attention.region_plans(
            operations, values, names, _field(graph, "metadata", {}), ref=_ref, field=_field,
            output_name=output_name, protected=set(outputs) | set(writes.values()), inputs=set(inputs),
            excluded=attention_lower.excluded_groups())
    loop_members = {name: plan for plan in loop_plans for name in plan["skip"]}
    attention_groups, attention_refusals = attention_lower.plan(
        operations, values, names, ref=_ref, field=_field, output_name=output_name,
        protected=set(outputs) | set(writes.values()),
        exclude=attention_lower.excluded_groups() + tuple(plan["label"] for plan in loop_plans), masks=True
    ) if fuse_attention else ([], [])
    attention_refusals = [*loop_refusals, *attention_refusals]
    composed_attention = None
    if compose and fuse_attention and target_arch == "sm_90" and not retain_all:
        from .compose_attention import ComposedAttention
        composed_attention = ComposedAttention(operations=operations, values=values, names=names,
            inputs=inputs, outputs=outputs, state_writes=writes, placement=placement,
            layouts=layouts, dtypes=dtypes, roots=roots, phases=phases, producers=producers,
            helper_parts=helper_parts, extra_headers=extra_headers, decisions=attention_decisions,
            ref=_ref, field=_field, output_name=output_name, groups=attention_groups,
            excluded_members=set(loop_members), refusals=attention_refusals,
            allow_short=not staged and not region_body and _LOWERING_DEPTH.get() == 1,
            tape=(_field(graph, "metadata", {}) or {}).get("composed_attention_tape"))
        attention_groups = composed_attention.remaining_groups
    for group in attention_groups:
        if group["skip"].intersection(accumulate):
            raise attention_lower.AttentionRefused(group["label"], "direct accumulated output requires ordinary operation lowering")
        group["v2_enabled"] = attention_schedule if attention_schedule == "force" else bool(attention_schedule)
        group["store_scores"] = bool(attention_store_scores)
        group["debug_repeat"] = bool(attention_debug_repeat)
    attention_lower.share_mask_tables(attention_groups, None, values, _field)
    attention_members = {name: group for group in attention_groups for name in group["skip"]}
    from . import torch_layer_norm, torch_layer_norm_backward
    layer_norm_plans = (torch_layer_norm.find_plans(graph)
                       if torch_layer_norm.KEY in (_field(graph, "metadata", {}) or {}) else [])
    layer_norm_backward_plans = (torch_layer_norm_backward.find_plans(graph)
                       if torch_layer_norm.KEY in (_field(graph, "metadata", {}) or {}) else [])
    layer_norm_affine_plans = (torch_layer_norm_backward.find_affine_plans(graph)
                       if torch_layer_norm_backward.AFFINE_KEY in (_field(graph, "metadata", {}) or {}) else [])
    if (layer_norm_plans or layer_norm_backward_plans or layer_norm_affine_plans) and (not compose or target_arch != "sm_90" or retain_all or staged or region_body):
        raise LoweringError("explicit CUDA LayerNorm arithmetic requires ordinary sm_90 common composition")
    layer_norm_output_names = {_field(operation, "name"): output_name(operation) for operation in operations}
    layer_norm_members = {layer_norm_output_names[member]: plan
                          for plan in layer_norm_plans for member in plan.members}
    layer_norm_backward_members = {layer_norm_output_names[member]: plan
                          for plan in layer_norm_backward_plans for member in plan.members}
    layer_norm_affine_members = {layer_norm_output_names[member]: plan
                          for plan in layer_norm_affine_plans for member in plan.members}
    if set(layer_norm_members).intersection(layer_norm_backward_members):
        raise LoweringError("explicit CUDA LayerNorm forward/backward members overlap")
    if (set(layer_norm_members) | set(layer_norm_backward_members)).intersection(layer_norm_affine_members):
        raise LoweringError("explicit CUDA LayerNorm affine members overlap another replacement")
    if (set(layer_norm_members) | set(layer_norm_backward_members) | set(layer_norm_affine_members)).intersection(accumulate):
        raise LoweringError("explicit CUDA LayerNorm does not admit direct accumulated outputs")
    # Step 02: BF16 matrix operands with misaligned rows get padded row pitches (matrix_adapters/padding.py).
    padded_operands = {}
    if matrix_provider is not None and target_arch == "sm_90" and not retain_all:
        from .matrix_adapters import padding as _padding
        padded_operands = _padding.plan(operations, values, names, ref=_ref, field=_field, output_name=output_name,
            flatten_refs=_flatten_refs, dtype_of=lambda n: dtypes.get(n),
            protected=set(outputs) | set(writes.values()) | set(writes) | set(placement) | set(elided)
                      | set(zero_contributions),
            absorbed=set(attention_members) | set(loop_members)
                     | set(getattr(composed_attention, "members", None) or ()) | set(layer_norm_members)
                     | set(layer_norm_backward_members) | set(layer_norm_affine_members),
            excluded=set(write_plans) | set(copy_sources) | set(copy_sources.values()) | set(accumulate),
            slice_views={operand for binding in ((_field(graph, "metadata", {}) or {})
                         .get("serial_bf16_readout", {}).get("products", {}).values())
                         if binding.get("role") == "weight_partial"
                         for operand in (binding["a"], binding["b"])})

    padded_writers = set()      # padded values actually written through a pitch-aware writer

    def _split_sum(op, name, out, part, split):
        # LATCH-TINYMOE-MATRIX (ops/matrix_tma_split.py): ascending FP32 slice sum rounded once to the product dtype,
        # as an ordinary same-shape elementwise value (its consumers' chain rules apply) or a splitk_reduce phase.
        from . import matrix_tma_split
        record = {"splits": split["splits"], "partial": part}
        if matrix_tma_split.ELEMENTWISE_SUM:
            reg = "r_" + name
            add_phase("elementwise", op, name, "", [part], shape=tuple(layouts[name].shape), dtype=dtypes[name],
                      registers={name: reg}, tma_split_sum=record,
                      body=[f"const {_CTYPES[dtypes[name]]} {reg} = {rounded(split['sum_expression'], dtypes[name])}; "
                            f"{_padded_store(name, out)} = {reg};"])
        else:
            add_phase("splitk_reduce", op, name, split["reduce"], [part], reduction_elements=split["splits"],
                      tma_split_sum=record)

    def _padded_store(name, out):
        if name not in padded_operands:
            return f"{out}[i]"
        padded_writers.add(name)
        if name in accumulate:
            raise LoweringError(f"{name}: padded matrix operand cannot be an accumulated output")
        wide = layouts[name].numel >= _U32_LIMIT          # 32-bit quotients only below 2**31 elements
        index = _padding.store_index(layouts[name].shape, padded_operands[name]["pitch"],
                                     (lambda i, d: f"{i} / {d}LL") if wide else _div32,
                                     (lambda i, d: f"{i} % {d}LL") if wide else _mod32)
        return f"{out}[{index}]"

    def attention_define(name, shape, root=None, strides=None):
        layouts[name] = TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), root or name)
        dtypes[name] = "float32"
        if root is None and name not in roots and name not in placement:
            roots.append(name)

    def attention_view(name, target, source):
        views.append({"name": name, "target": target, "source": source})

    def attention_phase(kind, code, metadata, reads, phase_writes):
        include = '#include "attention_fp32.cuh"\n'
        if include not in helper_parts:
            helper_parts.append(include)
        if attention_lower.HEADER not in extra_headers:
            extra_headers.append(attention_lower.HEADER)
        if metadata.get("header") == attention_lower.HEADER_V2:
            include_v2 = ("" if attention_pipeline else "#define PA2_PIPELINE 0\n") + '#include "attention_fp32_v2.cuh"\n'
            if include_v2 not in helper_parts:
                helper_parts.append(include_v2)
            if attention_lower.HEADER_V2 not in extra_headers:
                extra_headers.append(attention_lower.HEADER_V2)
        phases.append({"kind": kind, "names": list(phase_writes), "targets": ["p_ops_d." + kind],
                       "reads": list(dict.fromkeys(reads)), "writes": list(phase_writes),
                       "code": code, **metadata})
        for value in phase_writes:
            producers[value] = len(phases) - 1

    def loop_install(plan):
        """P-OPS4-E: one short-attention phase whose statistics are loop tape slots."""
        from . import loop_attention, short_attention
        label = plan["label"]
        for key in ("q", "k", "v") + (("do",) if plan["which"] == "backward" else ()):
            source = plan["sources"][key]
            actual = layouts[source]
            shape = tuple(int(x) for x in _field(values[source], "shape"))
            strides = tuple(int(x) for x in _field(values[source], "strides", _contiguous(shape)))
            if (actual.shape, actual.strides, actual.root, actual.offset) != (shape, strides, source, 0):
                raise attention_lower.AttentionRefused(label, f"{source}: lowered BF16 source layout differs")
        mask = None
        if plan["group"].get("mask") is not None:
            mask = attention_lower._mask_strides(layouts[plan["group"]["mask"]], plan["bh"], plan["t"], True)
            if mask is None or dtypes[plan["group"]["mask"]] != "float32":
                raise attention_lower.AttentionRefused(label, "mask is not an FP32 broadcastable tensor")
            # Only read-only external masks can share the once-per-update table.
            # Carry/state or computed masks may differ across body iterations.
            mask_root = layouts[plan["group"]["mask"]].root
            labels = (_field(graph, "metadata", {}) or {}).get("input_labels", [])
            input_labels = dict(zip(inputs, labels))
            immutable = str(input_labels.get(mask_root, "")).startswith(("input:", "constant:"))
            plan = dict(plan, immutable_mask=immutable)
        from . import long_attention as _p9u
        if (P9_DELTA_EPI and plan["which"] == "backward" and plan.get("kernel") == "long"
                and _p9u.uses_p9(plan, mask)):
            # P9-B: the plain 128 x 128 staged p5c product writing dO computes the delta table in its epilogue.
            from . import long_attention as _p9l, p5c_matrix as _p5d
            src_ = plan["sources"]["do"]
            j_ = producers.get(src_)
            sp_ = phases[j_].get("matrix_spec") if j_ is not None else None
            pl_ = phases[j_].get("matrix_plan") if j_ is not None else None
            sdo_ = plan["in_strides"]["do"]
            if (sp_ and pl_ and sp_.get("engine") == "p5c" and not sp_.get("epi") and not sp_.get("out_f32")
                    and not sp_.get("out_acc") and not sp_.get("out_raw") and not sp_.get("splitk") and _p5d.P5C_STAGED
                    and (pl_.get("bm"), pl_.get("bn")) == (128, 128) and not pl_.get("ws") and not pl_.get("pingpong")
                    and pl_.get("mc", 1) == 1 and sp_.get("out") == _ident(src_) and plan["hd"] == 64
                    and sp_["n"] == plan["heads"] * 64 and sp_["m"] * plan["heads"] == plan["bh"] * plan["t"]
                    and tuple(sdo_) == (plan["t"] * plan["heads"] * 64, 64, plan["heads"] * 64)
                    and plan["aux"].get("o32")):
                found_ = re.search(r"\(g, smem, ([^,]+), ([^,]+), ([^,]+), ([^,)]+)", phases[j_]["code"])
                if found_:
                    dl_ = {"o32": _ident(plan["aux"]["o32"]), "t": plan["t"], "heads": plan["heads"],
                           "slot": _p9l.mask_slot(plan["label"], "backward")}
                    if P9_QNORM_EPI and plan["t"] % 128 == 0:
                        dl_.update(q=_ident(plan["sources"]["q"]), lq=tuple(plan["in_strides"]["q"]))
                    sp_ = dict(sp_, p9_delta=dl_)
                    phases[j_]["matrix_spec"] = sp_
                    phases[j_]["code"] = _p5d.code(sp_, pl_, _ident, *found_.groups())
                    phases[j_]["reads"] = list(dict.fromkeys(phases[j_]["reads"] + [plan["aux"]["o32"]]
                                                             + ([plan["sources"]["q"]] if dl_.get("q") else [])))
                    plan = dict(plan, p9_delta_ready=True, p9_qnorm_ready=bool(dl_.get("q")))
        from .group_rules import FAST_ACTIVATION
        kind, code, extra = loop_attention.emit(plan, layouts, _ident, mask,
                                               fast_exp=FAST_ACTIVATION.get())
        if attention_debug_repeat:
            # Diagnostic only: the idempotent phase body runs twice (in-situ cost of one execution).
            code = "{\n" + code + "\n}\n__syncthreads();\n{\n" + code + "\n}"
        for value in plan["writes"]:
            value_shape = tuple(int(x) for x in _field(values[value], "shape"))
            layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
            if value not in placement:
                roots.append(value)
        headers = [short_attention.HEADER] + (["attention_long.cuh"] if plan.get("kernel") == "long" else [])
        if extra.get("engine") == "p9":
            from . import long_attention as _p9_long
            for macro in _p9_long.P9_DEFINES:
                name_, _, value_ = macro.partition("=")
                if f"#define {name_} {value_ or 1}\n" not in helper_parts:
                    helper_parts.append(f"#define {name_} {value_ or 1}\n")
            headers.append(_p9_long.P9_HEADER)
        for header in headers:
            include = f'#include "{header}"\n'
            if include not in helper_parts:
                helper_parts.append(include)
            if header not in extra_headers:
                extra_headers.append(header)
        if extra.get("helper") and extra["helper"] not in helper_parts:
            helper_parts.append(extra["helper"])        # P9-B attention wrapper (after its headers)
        phases.append({"kind": kind, "names": list(plan["writes"]), "targets": ["p_ops4_e." + kind],
                       "reads": list(dict.fromkeys(loop_attention.reads(plan))), "writes": list(plan["writes"]),
                       "code": code, "work_items": extra["work_items"], "shared_bytes": extra["shared_bytes"],
                       "group": label, "attention_output": plan["group"]["o"], "header": headers[-1],
                       "step06_attention": {"t": plan["t"], "bh": plan["bh"], "which": plan["which"],
                                            "engine": extra.get("engine")},
                       "threads": 256, "source_commit": loop_attention.SOURCE_COMMIT,
                       **({"forward_schedule": extra["forward_schedule"]} if "forward_schedule" in extra else {})})
        if compose and plan.get("kernel") == "long":
            # The attributed long/P9 helpers may contain hidden grid barriers
            # and reuse one module-wide dS buffer. Region-local tensor hazards
            # alone do not order independent bodies/tracks using that storage.
            phases[-1].update(global_fence=True, requires_full_grid=True,
                global_effects=[{"storage": "private_runtime_cuda_module", "header": headers[-1],
                                 "access": "read_write", "scope": "global-fenced attention body",
                                 "reason": "hidden barrier/counter protocol and shared dS scratch"}])
        for value in plan["writes"]:
            producers[value] = len(phases) - 1
        attention_decisions.append({"group": label, "decision": f"loop_{plan.get('kernel', 'short')}_bf16_attention_" + plan["which"],
            "head_width": plan["hd"], "sequence": plan["t"], "head_batch": plan["bh"], "heads": plan["heads"],
            "splits": plan["splits"], "scale": plan["group"]["scale"], "mask": plan["group"]["mask"],
            "sources": plan["sources"], "writes": plan["writes"], "tape_statistics": plan["aux"],
            "removed": sorted(plan["removed"]), "source_commit": loop_attention.SOURCE_COMMIT})

    scan_decisions = []
    # P8-C vprod: single-consumer elementwise/padding producers (casts, adds, slice/select_backward, cat, clones)
    # whose readers are all elementwise expressions are not stored; readers evaluate them inline (same operations
    # and roundings, so bit-identical values) and depend on the producer's own inputs instead.
    region_inline_costs = {}
    # LATCH-TINYMOE-FUSE rope: the same inline evaluation admitted only for recognized rotary attention-boundary chains.
    def rope_u32(value, value_shape):
        """Inline index maps of a recognized rope value use 32-bit arithmetic (index space below 2**31)."""
        return (_norm_fusion.rope_value(value) and value in virtual and _index32()
                and _numel(tuple(value_shape)) < _U32_LIMIT)
    rope_only = (_norm_fusion.rope_scope() and not ("vprod" in features or "vprodx" in features or region_expression_inline)
                 and compose and not staged and not region_body)
    if ("vprod" in features or "vprodx" in features or region_expression_inline or rope_only) and fuse_elementwise and not retain_all:
        def vroot(n):
            for _ in range(256):
                alias = _field(values[n], "alias_of") if n in values else None
                if alias is None:
                    return n
                n = alias
            return n

        def vbase(op):
            return str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        candidates_bases = {"add", "sub", "mul", "div", "neg", "rsub", "_to_copy", "clone", "contiguous",
                            "slice_backward", "select_backward", "cat"}
        index_bases = {"slice_backward", "select_backward", "cat"}
        # LATCH-MOESUP: dim-0 row gathers (embedding forward, index_select) are index maps whose readers can evaluate
        # table[index[row]][col] inline (data movement only; out-of-range indices trap as in the stored form).
        gather_bases = {"embedding", "index_select"} if VIRTUAL_GATHERS else set()
        candidates_bases |= gather_bases
        if region_expression_inline:
            # Reuse the existing mapped expression renderer with the donor's
            # bounded recomputation budget. This opt-in initially admits pure
            # FP32 arithmetic only; matrix inputs and published values remain
            # materialized. No reduction or matrix calculation is replanned.
            candidates_bases = {"add", "sub", "mul", "div", "neg", "rsub", "clone", "contiguous", "pow"}
        elif "vprodx" in features and "vprod" not in features:
            candidates_bases = set(index_bases)   # P8-C vprodx: only index-map producers are inlined
        consumer_bases = (_ELEMENTWISE_BASES - {"slice_backward"} - {"fill", "where", "masked_fill", "scalar_tensor"})
        if region_expression_inline:
            # These consumers use expression()/read() below, rather than an
            # opaque operation emitter which may require a storage pointer.
            consumer_bases = {"add", "sub", "mul", "div", "pow", "neg", "rsub", "clone", "contiguous",
                              "tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs", "reciprocal",
                              "sigmoid", "silu", "silu_backward", "tanh_backward", "sigmoid_backward",
                              "relu", "threshold_backward", "erf", "sin", "cos"}
        inplace_inputs = {inputs[int(mode[8:])] for _, (_, mode) in placement.items()
                          if mode.startswith("inplace:") and int(mode[8:]) < len(inputs)}
        protected_v = (set(outputs) | set(writes.values()) | set(writes) | set(placement) | set(accumulate)
                       | set(write_plans) | set(elided) | set(zero_contributions) | set(undefined)
                       | set(loop_members) | set(attention_members) | set(copy_sources.values()))
        protected_v |= {vroot(n) for n in list(protected_v) if n in values}
        op_by_name, consumers_v, direct_reads, direct_uses, use_counts = {}, {}, {}, {}, {}
        for op in operations:
            n_ = output_name(op)
            op_by_name[n_] = op
            if vbase(op) in _VIEW_BASES:
                continue
            uses_ = _flatten_refs((list(_field(op, "args", [])), dict(_field(op, "kwargs", {}))), names)
            # Bound renderer substitutions, not just references in the IR.
            # For example silu_backward renders its saved input three times:
            # once directly and once inside each of its two sigmoid factors.
            # These extra uses affect costing only; dependency sets stay equal.
            args_ = list(_field(op, "args", []))
            if region_expression_inline:
                base_ = vbase(op)
                multiplicity = {"silu": {0: 2}, "silu_backward": {1: 3},
                                "tanh_backward": {1: 2}, "sigmoid_backward": {1: 2}}.get(base_, {})
                if base_ == "pow" and len(args_) == 2:
                    if args_[1] == 2:
                        multiplicity = {0: 2}
                    elif "powx" in features and isinstance(args_[1], (int, float)) and float(args_[1]) == 3.0:
                        multiplicity = {0: 3}
                for index, count in multiplicity.items():
                    if index < len(args_):
                        uses_ += _flatten_refs(args_[index], names) * (count - 1)
            direct_uses[n_] = [vroot(r) for r in uses_]
            for r in direct_uses[n_]:
                use_counts[r] = use_counts.get(r, 0) + 1
            reads_ = set(uses_)
            direct_reads[n_] = {vroot(r) for r in reads_}
            for r in reads_:
                consumers_v.setdefault(vroot(r), set()).add(n_)
        for n_, op in op_by_name.items():
            b_ = vbase(op)
            if b_ not in candidates_bases or n_ in protected_v or vroot(n_) != n_ or n_ not in values:
                continue
            if rope_only and not _norm_fusion.rope_value(n_):
                continue
            if region_expression_inline:
                args_ = list(_field(op, "args", []))
                if (_dtype(values[n_]) != "float32" or (b_ == "pow" and
                        (len(args_) != 2 or type(args_[1]) not in (int, float) or args_[1] not in (1, 2)))):
                    continue
            if _dtype(values[n_]) not in ("float32", "bfloat16") or not _numel(tuple(_field(values[n_], "shape"))):
                continue
            srcs = direct_reads.get(n_, set())
            if b_ in gather_bases:
                if not _gather_admitted(b_, op, values, names):
                    continue
                srcs = {r for r in srcs if r in values and _dtype(values[r]) != "int64"}
            if any(_dtype(values[r]) not in ("float32", "bfloat16") for r in srcs if r in values):
                continue
            if region_expression_inline and any(_dtype(values[r]) != "float32" for r in srcs if r in values):
                continue
            if srcs & inplace_inputs or (b_ == "cat" and not isinstance(list(_field(op, "args", []))[0], (list, tuple))):
                continue
            virtual.add(n_)
        changed = True
        while changed:
            changed = False
            for n_ in sorted(virtual):
                cons = consumers_v.get(n_, set())
                ok = len(cons) > 0 if region_expression_inline else 0 < len(cons) <= 2
                if not region_expression_inline and ok and len(cons) == 2 and (direct_reads.get(n_, set()) & virtual or vbase(op_by_name[n_]) in index_bases):
                    ok = False   # duplicate only leaf casts/arithmetic, never chains or index maps
                    if (rope_only and _norm_fusion.rope_value(n_) and vbase(op_by_name[n_]) not in index_bases
                            and all(_norm_fusion.rope_value(r) for r in direct_reads.get(n_, set()) & virtual)):
                        # LATCH-TINYMOE-FUSE rope: a short rotary arithmetic/cast chain read by the two rotate-half
                        # slices is evaluated in both readers (same operations on the same loaded values).
                        ok = True
                if (ok and len(cons) == 2 and vbase(op_by_name[n_]) in gather_bases
                        and not direct_reads.get(n_, set()) & virtual):
                    ok = True    # LATCH-MOESUP: a row gather read twice re-reads its (small) table rows; no store
                for c in cons:
                    if not ok:
                        break
                    if c in virtual:
                        continue
                    if region_expression_inline and vbase(op_by_name[c]) in {"sum", "mean"}:
                        # A full contiguous FP32 reduction can evaluate the
                        # same source expression at each unchanged leaf index.
                        # Partial/view/dynamic reductions keep their buffers.
                        cargs = list(_field(op_by_name[c], "args", []))
                        shape_ = tuple(_field(values[n_], "shape"))
                        strides_ = tuple(_field(values[n_], "strides", _field(values[n_], "stride", _contiguous(shape_))))
                        if (exact_reductions and _dtype(values[c]) == "float32"
                                and _numel(tuple(_field(values[c], "shape"))) == 1
                                and _ref(cargs[0], names) == n_ and strides_ == _contiguous(shape_)
                                and not _field(values[n_], "storage_offset", 0)
                                and c not in set(loop_members) | set(attention_members) | set(write_plans) | set(elided)):
                            continue
                    if (vbase(op_by_name[c]) not in consumer_bases
                            or c in set(loop_members) | set(attention_members) | set(write_plans) | set(elided)):
                        ok = False
                if not ok:
                    virtual.discard(n_)
                    changed = True
            if region_expression_inline:
                costs = {}
                for n_, op in op_by_name.items():
                    if n_ not in virtual:
                        continue
                    cost = (4 if vbase(op) in {"div", "pow"} else 1) + sum(
                        costs.get(r, 0) for r in direct_uses.get(n_, ()))
                    if cost > 12 or cost * use_counts[n_] > 24:
                        virtual.remove(n_)
                        changed = True
                    else:
                        costs[n_] = cost
                region_inline_costs = costs

    def real_refs(rs):
        result = []
        for r in rs:
            root_ = layouts[r].root if r in layouts else r
            result.extend(virtual_reads[root_] if root_ in virtual_text else [r])
        return list(dict.fromkeys(result))
    from . import window_attention, window_fusion, ragged_experts_lower
    for op in operations:
        target = str(_field(op, "target", _field(op, "op")))
        target = target.removeprefix("torch.ops.").removeprefix("aten.")
        if target == _norm_fusion.RPACKB_TARGET:
            def rpackb_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            for phase in _norm_fusion.emit_rotary_unpack(op, values=values, layouts=layouts, dtypes=dtypes,
                    names=names, ref=_ref, field=_field, ident=_ident, define=rpackb_define):
                phases.append(phase)
                for value in phase["writes"]:
                    producers[value] = len(phases) - 1
            continue
        if target == _norm_fusion.RPACK_TARGET:
            # LATCH-TINYMOE-FUSE rpack: one generated row pass per recognized padded rotary/GQA attention operand.
            def rpack_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            for phase in _norm_fusion.emit_rotary_pack(op, values=values, layouts=layouts, dtypes=dtypes,
                    names=names, ref=_ref, field=_field, ident=_ident, define=rpack_define):
                phases.append(phase)
                for value in phase["writes"]:
                    producers[value] = len(phases) - 1
            continue
        if target == "latch_ops.permutation_gather.default":
            from . import row_permutation
            def permutation_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            for phase in row_permutation.emit(op, values=values, layouts=layouts, dtypes=dtypes,
                    names=names, ref=_ref, field=_field, ident=_ident, define=permutation_define):
                phases.append(phase)
                for value in phase["writes"]:
                    producers[value] = len(phases) - 1
            continue
        if target in ragged_experts_lower.TARGETS:
            def ragged_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            emitted = ragged_experts_lower.emit(op, values=values, layouts=layouts, dtypes=dtypes,
                names=names, ref=_ref, field=_field, ident=_ident, define=ragged_define, target_arch=target_arch,
                phases=phases, producers=producers, operations=operations)
            include = '#include "ragged_experts.cuh"\n'
            if include not in helper_parts:
                helper_parts.append(include)
            extra_headers.extend(h for h in ragged_experts_lower.HEADERS if h not in extra_headers)
            if any("latch_ragged::touter_acc<" in phase_.get("code", "") for phase_ in emitted):
                # LATCH-SELECTIVE-10 accfuse (ops.accumulation_rules): its own header, so rule-off sources are unchanged.
                if '#include "ragged_tma.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "ragged_tma.cuh"\n')
                if '#include "ragged_tma_acc.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "ragged_tma_acc.cuh"\n')
                for header_ in ("ragged_tma_acc.cuh",):
                    if header_ not in extra_headers:
                        extra_headers.append(header_)
            if any("latch_ragged::order_scan<" in phase_.get("code", "") for phase_ in emitted):
                # LATCH-SELECTIVE-10 rorder (ops.accumulation_rules): its own header, so rule-off sources are unchanged.
                if '#include "ragged_order_scan.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "ragged_order_scan.cuh"\n')
                if "ragged_order_scan.cuh" not in extra_headers:
                    extra_headers.append("ragged_order_scan.cuh")
            if any(phase_.get("tma_operands") for phase_ in emitted):
                if '#include "ragged_tma.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "ragged_tma.cuh"\n')
                extra_headers.extend(h for h in ragged_experts_lower.TMA_HEADERS if h not in extra_headers)
            if any(phase_.get("flat_claim") for phase_ in emitted):
                # LATCH-TRAIN-EXPERTS claimed products: their own header, so claim-off sources are unchanged.
                if ragged_experts_lower._feature("sepi") and '#define LATCH_CLAIM_SEPI 1\n' not in helper_parts:
                    helper_parts.append('#define LATCH_CLAIM_SEPI 1\n')
                if '#include "ragged_claim.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "ragged_claim.cuh"\n')
                if ragged_experts_lower.CLAIM_HEADER not in extra_headers:
                    extra_headers.append(ragged_experts_lower.CLAIM_HEADER)
            for phase in emitted:
                phases.append(phase)
                for value in phase["writes"]:
                    producers[value] = len(phases) - 1
            continue
        if target == "latch_ops.cat_cast.default":
            # LATCH-TINYMOE-EGRAD: concatenation and float cast in one pass (ops/cast_concat.py).
            from . import cast_concat

            def cat_cast_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            for phase in cast_concat.emit(op, values=values, layouts=layouts, dtypes=dtypes, names=names, ref=_ref,
                                          field=_field, ident=_ident, define=cat_cast_define, target_arch=target_arch):
                phases.append(phase)
                for value in phase["writes"]:
                    producers[value] = len(phases) - 1
            continue
        if target in window_attention.TARGETS or target in window_fusion.TARGETS:
            # P8-C: opaque rectangular mixed attention operations (ops/window_attention.py), one phase each.
            def window_define(value, value_shape, value_strides, value_dtype):
                layouts[value] = TensorLayout(tuple(value_shape), tuple(value_strides or _contiguous(value_shape)), value)
                dtypes[value] = value_dtype
                if value not in placement and value not in roots:
                    roots.append(value)
            emitter = window_attention.emit if target in window_attention.TARGETS else window_fusion.emit
            kind, code, metadata, reads, results = emitter(
                op, values=values, layouts=layouts, dtypes=dtypes, names=names, ref=_ref, field=_field,
                ident=_ident, define=window_define)
            result = results[0]
            include = f'#include "{window_attention.HEADER}"\n'
            if include not in helper_parts:
                helper_parts.append(include)
            for header_ in (window_attention.HEADER, *window_attention.INCLUDED_HEADERS):
                if header_ not in extra_headers:
                    extra_headers.append(header_)
            phases.append({"kind": kind, "names": list(results), "targets": [target], "reads": reads,
                           "writes": list(results), "code": code, "work_items": metadata["work_items"],
                           "shared_bytes": metadata["shared_bytes"], "threads": 256,
                           "source_commit": "P8-C window attention"})
            for value in results:
                producers[value] = len(phases) - 1
            attention_decisions.append({"group": result, "decision": "p8c_" + kind, **metadata})
            continue
        if target in _scan_targets():
            from . import scan_lower
            if staged or partition_phases or region_body or (barrier_schedule != "phase" and not compose):
                raise LoweringError(f"{target}: scan items carry internal grid barriers; staged/region schedules refuse them")
            try:
                emission = scan_lower.emit(op, values=values, layouts=layouts, dtypes=dtypes, names=names,
                                           ref=_ref, field=_field, ident=_ident, engine=scan_engine,
                                           options=scan_options)
            except scan_lower.ScanRefused as exc:
                raise LoweringError(f"{target}: {exc}") from exc
            layouts[emission.root] = TensorLayout((emission.root_words,), (1,), emission.root)
            dtypes[emission.root] = "float32"
            roots.append(emission.root)
            for result, result_shape, result_dtype, offset in emission.outputs:
                layouts[result] = TensorLayout(result_shape, _contiguous(result_shape), emission.root, offset)
                dtypes[result] = result_dtype
                views.append({"name": result, "target": target + ":packed_result", "source": emission.root})
            include = emission.include
            if include not in helper_parts:
                helper_parts.append(include)
            if emission.header not in extra_headers:
                extra_headers.append(emission.header)
            phases.append({"kind": emission.kind, "names": [emission.root], "targets": [target],
                           "reads": list(dict.fromkeys(emission.reads)), "writes": [emission.root],
                           "code": emission.code, **emission.metadata})
            producers[emission.root] = len(phases) - 1
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
        if name in layer_norm_affine_members:
            plan = layer_norm_affine_members[name]
            if name == layer_norm_output_names[plan.anchor]:
                sources = (plan.x, plan.dy, plan.mean, plan.rstd)
                for source in sources:
                    layout = layouts[source]
                    if not layout.contiguous or layout.root in virtual:
                        raise LoweringError("CUDA LayerNorm affine source lost its material layout: " + source)
                for result in plan.finals:
                    result_shape = tuple(_field(values[result], "shape"))
                    layouts[result] = TensorLayout(result_shape, _contiguous(result_shape), result)
                    if result not in placement and result not in roots:
                        roots.append(result)
                header = "torch_layer_norm_affine.cuh"
                include = '#include "' + header + '"\n'
                if include not in helper_parts:
                    helper_parts.append(include)
                if header not in extra_headers:
                    extra_headers.append(header)
                pointers = [f"{_ident(layouts[source].root)} + {layouts[source].offset}LL" for source in sources]
                pointers.extend(_ident(result) for result in plan.finals)
                code = f'''for (int column_tile = blockIdx.x; column_tile < {plan.width // 32}; column_tile += gridDim.x) {{
  torch_ln_affine::tile({', '.join(pointers)}, column_tile, smem);
}}'''
                split_record = None
                if torch_layer_norm_backward.AFFINE_SPLIT and plan.width == 768 and plan.rows == 16384:
                    # LATCH-MOE-EMA-FAST: the logical threads' serial chains on four CTAs per column tile, then
                    # the unchanged tree (torch_ln_affine::chains / combine).
                    chains_name = f"{plan.finals[0]}__affine_chains"
                    layouts[chains_name] = TensorLayout((2, plan.width // 32, 32, 32),
                                                        _contiguous((2, plan.width // 32, 32, 32)), chains_name)
                    dtypes[chains_name] = "float32"
                    roots.append(chains_name)
                    chains_code = f'''for (int item = blockIdx.x; item < {plan.width // 32 * 4}; item += gridDim.x) {{
  torch_ln_affine::chains({', '.join(pointers[:4])}, {_ident(chains_name)}, item);
}}'''
                    phases.append(dict(kind="reduction", names=[chains_name], targets=[target],
                        reads=list(sources), writes=[chains_name], code=chains_code,
                        engine="torch_cuda_layer_norm_gamma_beta_chains",
                        column_arithmetic=torch_layer_norm_backward.AFFINE_CONTRACT,
                        work_items=plan.width // 32 * 4, reduction_elements=plan.rows, shared_bytes=0, threads=256,
                        arithmetic="one logical thread's serial 8-row-per-256-row-tile chain per physical thread"))
                    producers[chains_name] = len(phases) - 1
                    code = f'''for (int column_tile = blockIdx.x; column_tile < {plan.width // 32}; column_tile += gridDim.x) {{
  torch_ln_affine::combine({_ident(chains_name)}, {', '.join(pointers[4:])}, column_tile, smem);
}}'''
                    split_record = dict(chains=chains_name, sources=list(sources),
                                        chain_engine="torch_cuda_layer_norm_gamma_beta_chains")
                phases.append(dict(kind="reduction", names=list(plan.finals), targets=[target],
                    reads=[split_record["chains"]] if split_record else list(sources), writes=list(plan.finals),
                    code=code, **({"layer_norm_affine_split": split_record} if split_record else {}),
                    engine="torch_cuda_layer_norm_gamma_beta_32x32",
                    column_arithmetic=torch_layer_norm_backward.AFFINE_CONTRACT,
                    work_items=plan.width // 32, reduction_elements=plan.rows,
                    shared_bytes=8448, threads=256, logical_threads=1024,
                    arithmetic="32x32 logical tile;8 contiguous rows per thread per256-row tile;final FMA;transpose XOR16,8,4,2,1",
                    layer_norm_affine_plan=plan.to_dict()))
                for result in plan.finals:
                    producers[result] = len(phases) - 1
            continue
        if name in layer_norm_backward_members:
            plan = layer_norm_backward_members[name]
            if name == layer_norm_output_names[plan.anchor]:
                sources = (plan.x, plan.dy, plan.gain, plan.mean, plan.rstd)
                for source in sources:
                    layout = layouts[source]
                    if (not layout.contiguous or layout.root in virtual
                            or (source in (plan.x, plan.dy, plan.gain) and layout.offset % 4)):
                        raise LoweringError("CUDA LayerNorm dX source lost its material vector layout: " + source)
                result = plan.output
                result_shape = tuple(_field(values[result], "shape"))
                layouts[result] = TensorLayout(result_shape, _contiguous(result_shape), result)
                if result not in placement and result not in roots:
                    roots.append(result)
                header = "torch_layer_norm_backward.cuh"
                include = '#include "' + header + '"\n'
                if include not in helper_parts:
                    helper_parts.append(include)
                if header not in extra_headers:
                    extra_headers.append(header)
                pointers = [f"{_ident(layouts[source].root)} + {layouts[source].offset}LL" for source in sources]
                pointers.append(_ident(result))
                dx_rows = 2 * torch_layer_norm.LN_ROWS_PER_HALF if torch_layer_norm.ROWS_PER_CTA == 2 else 1
                if torch_layer_norm.ROWS_PER_CTA == 2 and torch_layer_norm.LN_ROWS_PER_HALF > 1:
                    code = f'''for (long long row = {dx_rows}LL * blockIdx.x; row < {plan.rows}LL; row += {dx_rows}LL * gridDim.x) {{
  torch_ln_dx::rows<{torch_layer_norm.LN_ROWS_PER_HALF}>({', '.join(pointers)}, row, {plan.rows}LL, {plan.width}, smem);
  __syncthreads();
}}'''
                elif torch_layer_norm.ROWS_PER_CTA == 2:
                    code = f'''for (long long row = 2LL * blockIdx.x; row < {plan.rows}LL; row += 2LL * gridDim.x) {{
  torch_ln_dx::row2({', '.join(pointers)}, row, {plan.rows}LL, {plan.width}, smem);
  __syncthreads();
}}'''
                else:
                    code = f'''for (long long row = blockIdx.x; row < {plan.rows}LL; row += gridDim.x) {{
  torch_ln_dx::row({', '.join(pointers)}, row, {plan.width}, smem);
  __syncthreads();
}}'''
                phases.append(dict(kind="reduction", names=[result], targets=[target],
                    reads=list(sources), writes=[result], code=code,
                    engine="torch_cuda_layer_norm_dx_vector4",
                    row_arithmetic=torch_layer_norm_backward.ROW_ARITHMETIC,
                    work_items=(plan.rows + dx_rows - 1) // dx_rows,
                    reduction_elements=plan.width, shared_bytes=(48 if dx_rows > 2 else 32) * dx_rows, threads=256,
                    arithmetic="pinned vector4 two-statistic dX tree; measured shared-left explicit FP32 contraction",
                    layer_norm_backward_plan=plan.to_dict()))
                producers[result] = len(phases) - 1
            continue
        if name in layer_norm_members:
            plan = layer_norm_members[name]
            if name == layer_norm_output_names[plan.anchor]:
                sources = (plan.x, plan.gain, plan.bias)
                for source in sources:
                    layout = layouts[source]
                    if not layout.contiguous or layout.offset % 4 or layout.root in virtual:
                        raise LoweringError("CUDA LayerNorm source lost its aligned material layout: " + source)
                for result in plan.finals:
                    result_shape = tuple(_field(values[result], "shape"))
                    layouts[result] = TensorLayout(result_shape, _contiguous(result_shape), result)
                    if result not in placement and result not in roots:
                        roots.append(result)
                header = "torch_layer_norm.cuh"
                include = '#include "' + header + '"\n'
                if include not in helper_parts:
                    helper_parts.append(include)
                if header not in extra_headers:
                    extra_headers.append(header)
                pointers = [f"{_ident(layouts[source].root)} + {layouts[source].offset}LL" for source in sources]
                pointers += [_ident(result) for result in plan.finals]
                ln_rows = 2 * torch_layer_norm.LN_ROWS_PER_HALF if torch_layer_norm.ROWS_PER_CTA == 2 else 1
                if torch_layer_norm.ROWS_PER_CTA == 2 and torch_layer_norm.LN_ROWS_PER_HALF > 1:
                    code = f'''for (long long row = {ln_rows}LL * blockIdx.x; row < {plan.rows}LL; row += {ln_rows}LL * gridDim.x) {{
  torch_ln_vector4::torch_ln_rows<{torch_layer_norm.LN_ROWS_PER_HALF}>({', '.join(pointers)}, row, {plan.rows}LL, {plan.width}, {_float(plan.epsilon)}, smem);
  __syncthreads();
}}'''
                elif torch_layer_norm.ROWS_PER_CTA == 2:
                    code = f'''for (long long row = 2LL * blockIdx.x; row < {plan.rows}LL; row += 2LL * gridDim.x) {{
  torch_ln_vector4::torch_ln_row2({', '.join(pointers)}, row, {plan.rows}LL, {plan.width}, {_float(plan.epsilon)}, smem);
  __syncthreads();
}}'''
                else:
                    code = f'''for (long long row = blockIdx.x; row < {plan.rows}LL; row += gridDim.x) {{
  torch_ln_vector4::torch_ln_row({', '.join(pointers)}, row, {plan.width}, {_float(plan.epsilon)}, smem);
  __syncthreads();
}}'''
                phases.append(dict(kind="reduction", names=list(plan.finals), targets=[target],
                    reads=list(sources), writes=list(plan.finals), code=code,
                    engine="torch_cuda_layer_norm_vector4", row_arithmetic=torch_layer_norm.CONTRACT,
                    work_items=(plan.rows + ln_rows - 1) // ln_rows,
                    reduction_elements=plan.width, shared_bytes=48 * ln_rows, threads=256,
                    arithmetic="pinned vector4 Welford tree and explicit FP32 contraction",
                    layer_norm_plan=plan.to_dict()))
                for result in plan.finals:
                    producers[result] = len(phases) - 1
            continue
        if name in write_done:
            continue  # the producing product stored this copy directly
        if name in converted_products:
            continue  # P7-C: written by its BF16 product's FP32-converting wgmma store
        if composed_attention is not None and name in composed_attention.members:
            composed_attention.install(name)
            continue
        if name in loop_members:
            if name == loop_members[name]["anchor"]:
                loop_install(loop_members[name])
            continue
        if name in attention_members:
            attention_lower.emit_member(attention_members[name], name, layouts, dtypes, values,
                field=_field, ident=_ident, define=attention_define, add_phase=attention_phase,
                add_view=attention_view, record=attention_decisions.append)
            continue
        if name in elided:
            # Returned clone of a placed region value: an alias of the caller's storage.
            layouts[name] = TensorLayout(shape, _contiguous(shape), elided[name])
            views.append({"name": name, "target": target, "source": elided[name], "placement": "elided returned clone"})
            continue
        if name in zero_contributions:
            # Known zero accumulated into caller storage: no store at all.
            layouts[name] = TensorLayout(shape, _contiguous(shape), name)
            views.append({"name": name, "target": target, "source": None, "placement": "zero accumulation omitted"})
            continue
        if base == "fill":
            if len(args) != 2 or _ref(args[0], names) is None:
                raise LoweringError("functional fill requires a tensor template and one fill value")
            template = values[_ref(args[0], names)]
            if tuple(_field(template, "shape")) != shape or _dtype(template) != dtypes[name]:
                raise LoweringError("functional fill must preserve template shape/dtype")
            refs = _flatten_refs((args[1:], kwargs), names)
        if group_rules and base in {"zeros_like", "ones_like", "full_like", "new_zeros", "new_ones", "new_full"}:
            # P-OPS5-B: a constant fill reads only its template's static shape/dtype, never its data,
            # so it is not ordered after the template's producer (as functional fill already is).
            refs = _flatten_refs((args[1:], kwargs), names)
        if base in {"slice", "select"}:
            # C-OPS4-B: stride/offset view (ops/extra_ops.view).
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
                # source's storage absolutely. A full dense permutation names
                # the same storage as a contiguous view; prove coverage and
                # nonoverlap by sorting its non-singleton stride dimensions.
                # Slices, gaps, broadcasts and rebased input storage stay out.
                dense_source, dense_extent = True, 1
                for dimension, stride in sorted(zip(src.shape, src.strides), key=lambda item: item[1]):
                    if dimension == 1:
                        continue
                    if dimension <= 0 or stride != dense_extent:
                        dense_source = False
                        break
                    dense_extent *= dimension
                dense_source = src.contiguous or (dense_source and dense_extent == src.numel)
                root_value = values.get(src.root)
                if (src.offset or not dense_source or src.numel != layouts[src.root].numel
                        or (root_value is not None and int(_field(root_value, "storage_offset", 0) or 0))):
                    raise LoweringError(f"{name}: as_strided requires a dense offset-0 source allocation")
                strides = [int(v) for v in args[2]]
                offset = int(args[3]) if len(args) > 3 and args[3] is not None else 0
                extent = offset + sum((d - 1) * s for d, s in zip(shape, strides))
                if offset < 0 or len(strides) != len(shape) or any(s < 0 for s in strides) or (math.prod(shape) and extent >= src.numel):
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
        layouts[name] = TensorLayout(shape, _contiguous(shape) if name not in padded_operands else
                                     _padding.strides(shape, padded_operands[name]["pitch"]), name)
        if dynamic:
            dynamic.material(name, base)
        if name not in placement:
            roots.append(name)
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
                if "rngel" in features:
                    # P-OPS5-B: the mask is an elementwise value of (state, flat index), so consumers' groups
                    # (row engines, chains) may evaluate it where they use it; same Philox function and index.
                    reg = "r_" + name
                    add_phase("elementwise", op, name, "", refs, shape=shape, dtype="float32", registers={name: reg},
                              body=[f"const float {reg} = latch_philox_mask({_ident(state)}, i, {int(args[2])}ULL, "
                                    f"{_float(args[3])}); {out}[i] = {reg};"])
                    continue
                if dynamic:
                    from .dynamic_codegen import scalar
                    offset = scalar(args[2], integer=True) or f"{int(args[2])}ULL"
                    code = f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {dynamic.numel(name, shape)}; i += (long long)gridDim.x * blockDim.x) {out}[i] = latch_philox_mask({_ident(state)}, i, {offset}, {_float(args[3])});"
                else:
                    code = f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {_numel(shape)}LL; i += (long long)gridDim.x * blockDim.x) {out}[i] = latch_philox_mask({_ident(state)}, i, {int(args[2])}ULL, {_float(args[3])});"
            else:
                if dtypes[name] != "int64" or shape != (4,):
                    raise LoweringError("Philox advance requires int64[4] output")
                if dynamic:
                    from .dynamic_codegen import scalar
                    advance = scalar(args[1], integer=True) or f"{int(args[1])}ULL"
                else:
                    advance = f"{int(args[1])}ULL"
                code = f"if (blockIdx.x == 0 && threadIdx.x < 4) {out}[threadIdx.x] = latch_philox_advance_word({_ident(state)}, {advance}, threadIdx.x);"
            add_phase("random_state" if "advance" in target else "random_mask", op, name, code, refs)
            continue
        if conv_taps.defer(target, name, op, refs, args, kwargs, dtypes):
            if name in roots:
                roots.remove(name)
            continue
        from .transformer_lower import emit_operation
        from .exact_vocab_emit import emit as emit_exact_vocab, HEADERS as exact_vocab_headers
        exact_vocab_phase = emit_exact_vocab(
            _field(graph, "metadata", {}) or {}, target, args, kwargs, name, shape,
            layouts, dtypes, names)
        if exact_vocab_phase is not None:
            if not compose or target_arch != "sm_90" or staged or region_body:
                raise LoweringError("exact vocabulary rows require ordinary sm_90 common composition")
            kind, code, extra = exact_vocab_phase
            if name in padded_operands:
                if kind != "exact_vocab_grad" or padded_operands[name].get("kind") != "operand":
                    raise LoweringError("only the exact BF16 gradient writer supports operand row pitches")
                padded_writers.add(name)
            for header in exact_vocab_headers:
                if header not in extra_headers:
                    extra_headers.append(header)
                    helper_parts.append(f'#include "{header}"')
            # LATCH-MOE-EMA-FAST: a gradient whose other operands precede its statistics joins that row pass.
            from .exact_vocab_emit import fuse_rows as fuse_exact_vocab_rows
            fused_headers = fuse_exact_vocab_rows(kind, code, extra, name, args, refs, names, layouts,
                                                  phases, producers)
            if fused_headers is not None:
                for fused_header in fused_headers:
                    if fused_header not in extra_headers:
                        extra_headers.append(fused_header)
                        helper_parts.append(f'#include "{fused_header}"')
                continue
            add_phase(kind, op, name, code, refs, **extra)
            continue
        from .torch_logsoftmax import emit as emit_torch_logsoftmax, HEADER as torch_logsoftmax_header
        explicit_logsoftmax = emit_torch_logsoftmax(
            _field(graph, "metadata", {}) or {}, target, args, kwargs, name, shape,
            layouts, dtypes, names)
        if explicit_logsoftmax is not None:
            if not compose or target_arch != "sm_90" or staged or region_body:
                raise LoweringError("explicit CUDA log-softmax arithmetic requires ordinary sm_90 common composition")
            kind, code, extra = explicit_logsoftmax
            if torch_logsoftmax_header not in extra_headers:
                extra_headers.append(torch_logsoftmax_header)
                helper_parts.append(f'#include "{torch_logsoftmax_header}"')
            add_phase(kind, op, name, code, refs, **extra)
            continue
        if ordered_index_gradients and base == "embedding_dense_backward" and name not in accumulate:
            # LATCH-DISPATCH-GRAD: an index the graph proves to be a permutation (or a permutation // k) with a
            # known inverse reads its occurrences directly (ops/permutation_gradients.py); otherwise, and for
            # tagged arithmetic, LATCH-STEP-04 indexed_compose selects the base ordered emitter or an engine.
            indexed = None
            if permutation_facts and name not in embedding_arithmetic:
                from .permutation_gradients import emit as emit_permutation_gradient
                from .indexed_owner import _unravel
                indexed = emit_permutation_gradient(args, kwargs, name, shape, layouts, dtypes, names,
                                                    permutation_facts, ident=_ident, unravel=_unravel)
            if indexed is None:
                from .indexed_compose import embedding_backward
                indexed = embedding_backward(args, kwargs, name, shape, layouts, dtypes, names,
                                             cache=indexed_occurrence_cache, staged=bool(staged),
                                             arithmetic=embedding_arithmetic.get(name))
            if indexed is not None:
                if indexed["helpers"] and indexed["helpers"] not in helper_parts:
                    helper_parts.append(indexed["helpers"])
                for helper in indexed.get("outlined_helpers", ()):
                    if helper not in helper_parts:
                        helper_parts.append(helper)
                for value, value_shape, dtype in indexed["auxiliary"]:
                    layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
                    dtypes[value] = dtype
                    roots.append(value)
                for phase in indexed["phases"]:
                    phases.append({"names": phase["writes"],
                                   "targets": [str(_field(op, "target", _field(op, "op")))], **phase})
                    for value in phase["writes"]:
                        producers[value] = len(phases) - 1
                index_decisions.append({"output": name, **{key: value for key, value in indexed.items()
                                        if key not in {"auxiliary", "phases", "helpers", "outlined_helpers"}}})
                continue
        if target == "latch_ops.vocab_stats.default" and compose and not staged and not region_body:
            # LATCH-MOESUP: the logits product (a provider TMA body) is re-emitted with per-(row, N tile) softmax
            # partials in its staged epilogue; the statistics then merge the partials instead of re-reading logits.
            from . import vocab_head as _vocab
            x_ = _ref(args[0], names)
            q_ = producers.get(x_)
            record_ = (phases[q_].get("tma_respec") if _vocab.EPILOGUE_STATS and q_ is not None
                       and phases[q_].get("names") == [x_] and layouts[x_].root == x_ else None)
            if record_ is not None:
                from .matrix_adapters.bodies import Refusal as _Refusal, respec_tma_rowstats
                part_ = name + "__rowpart"
                try:
                    code_, extra_bytes_, tiles_ = respec_tma_rowstats(record_, ident=_ident, out=_ident(part_),
                                                                      valid=int(args[1]))
                except _Refusal:
                    code_ = None
                if code_ is not None and part_ not in layouts and part_ not in names:
                    rows_ = layouts[x_].shape[0]
                    layouts[part_] = TensorLayout((rows_, tiles_, 2), _contiguous((rows_, tiles_, 2)), part_)
                    dtypes[part_] = "float32"
                    roots.append(part_)
                    product_ = phases[q_]
                    product_["code"] = code_
                    product_["shared_bytes"] = int(product_.get("shared_bytes", 0)) + int(extra_bytes_)
                    product_["writes"] = [*product_["writes"], part_]
                    product_["row_partials"] = part_
                    product_.pop("tma_respec", None)   # re-emitted: later epilogue passes must not re-specify it
                    producers[part_] = q_
                    kind_, code2_, extra_, _ = _vocab.emit_stats_partials(_ident(part_), rows_, tiles_, _ident(name))
                    add_phase(kind_, op, name, code2_, [part_], **extra_)
                    continue

        if name in embedding_arithmetic:
            raise LoweringError("tagged embedding VJP was not retained by its explicit indexed emitter")
        from . import torch_softmax_backward
        explicit_softmax_backward = torch_softmax_backward.emit(
            _field(graph, "metadata", {}) or {}, target, args, kwargs, name, shape,
            layouts, dtypes, names)
        if explicit_softmax_backward is not None:
            if not compose or target_arch != "sm_90" or staged or region_body:
                raise LoweringError("explicit CUDA softmax backward requires ordinary sm_90 common composition")
            kind, code, extra = explicit_softmax_backward
            add_phase(kind, op, name, code, refs, **extra)
            continue
        emitted = None if name in virtual else emit_operation(
            target, args, kwargs, name, shape, layouts, dtypes, names, read,
            warp_rows="multi" if warp_softmax and "wsoft" in features else warp_softmax, inputs=tuple(inputs))
        if emitted is not None:
            kind, code, extra, helper = emitted
            if helper and helper not in helper_parts:
                helper_parts.append(helper)
            add_phase(kind, op, name, code, refs, **extra)
            continue
        if base in {"mm", "matmul", "bmm", "addmm"}:
            aa, bb = (args[0], args[1]) if base != "addmm" else (args[1], args[2])
            a, b = _ref(aa, names), _ref(bb, names)
            la, lb = layouts[a], layouts[b]
            if dynamic and any(v in dynamic.axes for v in (a, b, name)):
                if base == "addmm" or name in accumulate:
                    raise LoweringError("dynamic products require decomposed mm and ordinary SSA output")
                code, headers, extra = dynamic.matrix(a, b, name, layouts, dtypes,
                    split_k=split_k and matrix_policy != "frozen", target_arch=target_arch)
                for header in headers:
                    if header not in extra_headers:
                        extra_headers.append(header)
                    include = f'#include "{header}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                split_storage = extra.pop("split_storage", None)
                split_reduce_code = extra.pop("split_reduce_code", None)
                if split_storage:
                    scratch, scratch_shape = split_storage["name"], tuple(split_storage["shape"])
                    if scratch in layouts or scratch in names:
                        raise LoweringError("bounded split-K scratch name collides with a graph value")
                    layouts[scratch] = TensorLayout(scratch_shape, _contiguous(scratch_shape), scratch)
                    dtypes[scratch] = "float32"
                    roots.append(scratch)
                    add_phase("gemm", op, scratch, code, refs, **extra,
                              private_scratch=[{**split_storage, "partial_words": _numel(scratch_shape),
                                                "initialization": "active partial rows defined by this producer"}])
                    add_phase("matrix_split_reduce", op, name, split_reduce_code, [scratch],
                              engine="bounded_fixed_slice_sum", shared_bytes=0,
                              arithmetic="FP32 partials summed in ascending fixed slice order",
                              work_items=(_numel(shape) + 255) // 256, ksplit=extra["ksplit"])
                else:
                    add_phase("gemm", op, name, code, refs, **extra)
                matrix_decisions.append({"output": name, "selected": True, "reason": "bounded runtime extent and masked edges", "candidate": extra["engine"]})
                continue
            if matrix_policy != "frozen" and base != "addmm" and dtypes[a] == dtypes[b] == dtypes[name]:
                from .matrix_codegen import select
                from .matrix_rules import HEADER, emit_group
                p5c_ok = ("p5c" in features and ((bool(region_body) and staged) or (compose and _LOWERING_DEPTH.get() == 1)) and barrier_schedule == "dependency"
                          and name not in accumulate and not retain_all)
                if p5c_ok and "p5cw" in features:
                    p5c_ok = 'wide'
                if p5c_ok and "p5cv" in features and "p5ce" in features:
                    p5c_ok = 'wide_short' if p5c_ok == 'wide' else 'short'
                spec, reason = None, None
                exact_tensor = (_EXACT_BF16_PRODUCTS.get() == "tensor" and matrix_provider is not None
                                and not retain_all)
                if (COMPOSE_EXACT_BF16_PRODUCTS and (p5c_ok or exact_tensor) and base in ("mm", "matmul")
                        and dtypes[name] == "float32"
                        and name not in accumulate and not conv_taps.deferred and target_arch == "sm_90"):
                    # LATCH-DENSE-EXEC: an FP32 product of (views of) exact BF16 -> FP32 widenings (e.g. the opmath
                    # decomposition of a BF16 linear with bias) runs on p5c from the BF16 sources with FP32
                    # accumulation and an unrounded FP32 result (as fusion_p6d; strict exact3-class products).
                    if exact_bf16_producer is None:
                        exact_bf16_producer = {output_name(op): op for op in operations}
                    raw_sources = _exact_bf16_operands(la, lb, operations, output_name, names, layouts, dtypes, undefined,
                                                       exact_bf16_producer)
                    if raw_sources is not None:
                        (sa_, la_b), (sb_, lb_b) = raw_sources
                        spec_r, reason_r = (select(sa_, sb_, out, la_b, lb_b, shape, "bfloat16", arch=target_arch,
                                                   engine_ok=p5c_ok) if p5c_ok else (None, "exact_bf16_products"))
                        if spec_r is not None and spec_r.get("engine") == "p5c":
                            spec = dict(spec_r, out_raw=True)
                            reason = ("BF16-exact FP32 product on the warpgroup-MMA engine p5c (exact operand widenings; "
                                      "FP32 accumulation; unrounded FP32 result)")
                            refs = [sa_, sb_] + [r for r in refs if r not in (a, b)]
                        elif (RAW_PROVIDER_PRODUCTS and matrix_provider is not None and len(shape) == 2
                              and name not in placement and name not in set(writes.values())):
                            # LATCH-RESEARCH-EDITS: the P5C engine refuses the shape (M, N or K not multiples of 64,
                            # e.g. an awkward model width). The masked TMA wgmma body (latch_matrix pops4b_tma) runs
                            # the same BF16-exact product from the BF16 sources and stores the unrounded FP32
                            # accumulator (result mode store_raw): the arithmetic class of the P5C raw32 path. Before
                            # this rule the product fell to the generic FP32 SIMT tile (ffn-d600: 45x the control).
                            provided_raw = matrix_provider.product(a=sa_, b=sb_, name=name, out=out, la=la_b, lb=lb_b,
                                shape=shape, dtypes=("bfloat16", "bfloat16", "float32"), ident=_ident,
                                target_arch=target_arch, out_mode="store_raw")
                            if provided_raw is not None:
                                extra_headers.extend(h for h in provided_raw["headers"] if h not in extra_headers)
                                for header in provided_raw["headers"]:
                                    include = f'#include "{header}"'
                                    if include not in helper_parts:
                                        helper_parts.append(include)
                                helper_parts.extend(h for h in provided_raw["helpers"] if h not in helper_parts)
                                raw_refs = [sa_, sb_] + [r for r in refs if r not in (a, b)]
                                add_phase("gemm", op, name, provided_raw["code"], raw_refs,
                                          **provided_raw["extra"], **provided_raw["contract"])
                                matrix_decisions.append(dict(provided_raw["decision"], exact_bf16_sources=[sa_, sb_],
                                    reason="BF16-exact FP32 product on the masked TMA wgmma body (P5C refused the shape); "
                                           "unrounded FP32 result from the BF16 sources"))
                                continue
                from .compose import step14_flat as _step14_flat
                if (spec is None and _step14_flat() and compose and dtypes[name] == "float32" and len(shape) == 3 and name not in accumulate
                        and name not in placement and name not in outputs and name not in set(writes.values())
                        and _transposed_readers(name, value_readers.get(name, []), len(shape), names)):
                    # Release step 14: every reader views this batched FP32 product through its last-two-dimension
                    # transpose. Compute it as B^T A^T into storage with the readers' (contiguous) orientation;
                    # each element is the same ascending-k fmaf chain (fmaf is commutative in its factors).
                    swap = lambda lay: TensorLayout((*lay.shape[:-2], lay.shape[-1], lay.shape[-2]),
                                                    (*lay.strides[:-2], lay.strides[-1], lay.strides[-2]),
                                                    lay.root, lay.offset)
                    shape_t = (*shape[:-2], shape[-1], shape[-2])
                    spec_t, reason_t = select(b, a, out, swap(lb), swap(la), shape_t, dtypes[name], arch=target_arch,
                                              engine_ok=False)
                    if spec_t is not None:
                        spec, reason = spec_t, reason_t + "; computed as B^T A^T in its transposed readers' orientation"
                        layouts[name] = swap(TensorLayout(shape_t, _contiguous(shape_t), name))
                if spec is None:
                    spec, reason = select(a, b, out, la, lb, shape, dtypes[name], arch=target_arch, engine_ok=p5c_ok)
                if conv_taps.deferred:
                    spec, reason = conv_taps.select(spec, reason, a, b, out, la, lb, shape, dtypes, p5c_ok=p5c_ok,
                                                    accumulate=name in accumulate)
                if name in accumulate and spec is not None:
                    spec, reason = None, 'direct accumulated output retains the exact accumulation epilogue'
                out_pitch = (padded_operands[name]["pitch"] if padded_operands.get(name, {}).get("kind") == "product_result"
                             else None)
                provided = None
                if (("tsplit" in features or "tsplitx" in features) and matrix_provider is not None
                        and name not in accumulate and compose
                        and not staged and not region_body and not retain_all and out_pitch is None
                        and current_precision() == "practical" and (spec or {}).get("engine") == "p5c"):
                    # LATCH-TINYMOE-MATRIX: very long contractions (vocabulary-scale input gradients) kept by the
                    # chunked P5C/P8WS bodies run as deterministic K slices of <= 2048 on the TMA body instead
                    # (ops/matrix_tma_split.py choose_long), so their few long tiles become many grid tasks.
                    from . import matrix_tma_split
                    from .matrix_rules_p7b import TARGET_GRID as _split_grid
                    long_ = None
                    if int(la.shape[-1]) >= matrix_tma_split.LONG_K:
                        tma_ = matrix_provider.tma_emission(a=a, b=b, name=name, out=out, la=la, lb=lb, shape=shape,
                            dtypes=(dtypes[a], dtypes[b], dtypes[name]), ident=_ident, target_arch=target_arch,
                            out_pitch=None)
                        if tma_ is not None:
                            # LATCH-B1-FAST 'tsplitx': chunk-aligned exact slices (matrix_tma_split.choose_long_exact).
                            long_ = matrix_tma_split.emit(tma_, part=f"{name}_tsk", ident=_ident, grid=_split_grid,
                                                          long=True if "tsplit" in features else "exact")
                            if long_ is None:
                                matrix_provider.discard(tma_["marks"])
                    if long_ is not None:
                        part_ = f"{name}_tsk"
                        if part_ in names or part_ in layouts:
                            raise LoweringError(f"split-K partial name collision: {part_}")
                        extra_headers.extend(h for h in tma_["headers"] if h not in extra_headers)
                        for header in tma_["headers"]:
                            if f'#include "{header}"' not in helper_parts:
                                helper_parts.append(f'#include "{header}"')
                        helper_parts.extend(h for h in tma_["helpers"] if h not in helper_parts)
                        layouts[part_] = TensorLayout((long_["elements"],), (1,), part_)
                        dtypes[part_] = "float32"
                        roots.append(part_)
                        add_phase("gemm", op, part_, long_["code"], refs, **{**tma_["contract"], **long_["extra"]})
                        _split_sum(op, name, out, part_, long_)
                        matrix_decisions.append(dict(tma_["decision"], split_k=long_["splits"], partial=part_,
                                                     replaced_native="p5c"))
                        continue
                consult_native = getattr(matrix_provider, "consult_native", False)
                if matrix_provider is not None and name not in accumulate and (
                        (spec or {}).get("engine") != "p5c" or consult_native):
                    native = {"native": None if spec is None else spec.get("engine", "popsc")} if consult_native else {}
                    provided = matrix_provider.product(a=a, b=b, name=name, out=out, la=la, lb=lb, shape=shape,
                        dtypes=(dtypes[a], dtypes[b], dtypes[name]), ident=_ident, target_arch=target_arch,
                        out_pitch=out_pitch, **native)
                if out_pitch is not None and provided is not None and provided.get("out_pitch") == out_pitch:
                    padded_writers.add(name)
                if out_pitch is not None and (provided is None or provided.get("out_pitch") != out_pitch):
                    # Only the masked TMA body stores at a padded pitch; other bodies keep contiguous rows.
                    padded_operands.pop(name)
                    layouts[name] = TensorLayout(shape, _contiguous(shape), name)
                if matrix_provider is not None and name not in accumulate and (
                        (spec or {}).get("engine") != "p5c" or consult_native):
                    if provided is not None:
                        extra_headers.extend(h for h in provided["headers"] if h not in extra_headers)
                        for header in provided["headers"]:
                            include = f'#include "{header}"'
                            if include not in helper_parts:
                                helper_parts.append(include)
                        helper_parts.extend(h for h in provided["helpers"] if h not in helper_parts)
                        split_ = None
                        if ("tsplit" in features and compose and not staged and not region_body and not retain_all
                                and current_precision() == "practical" and out_pitch is None):
                            # LATCH-TINYMOE-MATRIX: underfilled long-K TMA products as deterministic K slices plus an
                            # ascending FP32 slice sum rounded once (ops/matrix_tma_split.py).
                            from . import matrix_tma_split
                            from .matrix_rules_p7b import TARGET_GRID as _split_grid
                            split_ = matrix_tma_split.emit(provided, part=f"{name}_tsk", ident=_ident,
                                                           grid=_split_grid)
                        if split_ is not None:
                            part_ = f"{name}_tsk"
                            if part_ in names or part_ in layouts:
                                raise LoweringError(f"split-K partial name collision: {part_}")
                            layouts[part_] = TensorLayout((split_["elements"],), (1,), part_)
                            dtypes[part_] = "float32"
                            roots.append(part_)
                            add_phase("gemm", op, part_, split_["code"], refs, **{**provided["contract"], **split_["extra"]})
                            _split_sum(op, name, out, part_, split_)
                            matrix_decisions.append(dict(provided["decision"], split_k=split_["splits"], partial=part_))
                            continue
                        add_phase("gemm", op, name, provided["code"], refs,
                                  **provided["extra"], **provided["contract"])
                        matrix_decisions.append(provided["decision"])
                        continue
                if spec is not None and spec.get('engine') == 'p5c':
                    if conv_taps.deferred:
                        spec, refs = conv_taps.virtualize_any(spec, a, b, la, lb, refs, layouts, names)
                    # P7-C: P-OPS5-C TMA + wgmma engine in staged loop regions. Operand maps are bound per region
                    # instance by the loop composer (loop_pipeline.finalize_region_maps); the product starts at the
                    # stage's CTA rotation like the popsc bodies.
                    spec = dict(spec, **({"pointers": (_ident(la.root), _ident(lb.root))} if staged else {}), rot_token=True)
                    if compose and not staged and not region_body and _LOWERING_DEPTH.get() == 1 and not spec.get("out_raw"):
                        # Reuse the existing flat P8WS feasibility and heavy-ALU
                        # epilogue guards for an independently admitted product.
                        spec["ws_ok"] = True
                    readers = value_readers.get(name, [])
                    conv = output_name(readers[0]) if len(readers) == 1 else None
                    rbase = (str(_field(readers[0], "target", "")).removeprefix("torch.ops.").removeprefix("aten.")
                             .split(".")[0] if readers else "")
                    if (conv is not None and rbase == "_to_copy" and dtypes.get(conv) == "float32" and not spec.get("out_raw")
                            and tuple(int(x) for x in _field(values[conv], "shape")) == shape
                            and name not in outputs and name not in set(writes.values()) and name not in placement
                            and conv not in undefined and conv not in write_done
                            and conv not in loop_members and conv not in attention_members and conv not in elided
                            and not (P9_KEEP_BF16 and conv not in accumulate)):
                        # Exact rule: the product's only reader is its BF16 -> FP32 conversion; the matrix phase stores
                        # float(bf16_rn(acc)) into the conversion's storage and the BF16 value has none.
                        converted_products[conv] = name
                        if name in roots:
                            roots.remove(name)
                        layouts.pop(name, None)
                        layouts[conv] = TensorLayout(shape, _contiguous(shape), conv)
                        if conv not in placement:
                            roots.append(conv)
                        spec = dict(spec, out=_ident(conv), out_f32=True, converted_from=name,
                                    out_acc=conv in accumulate)
                        reason += ('; FP32-converting ' + ('accumulation' if conv in accumulate else 'store')
                                   + ' into its only reader ' + conv)
                        name = conv
                matrix_decisions.append({'output': name, 'selected': spec is not None, 'reason': reason})
                if spec is not None:
                    extra_headers.extend(h for h in ('tgemm.cuh', 'bgemm.cuh', HEADER) if h not in extra_headers)
                    include = f'#include "{HEADER}"'
                    if include not in helper_parts:
                        helper_parts.append(include)
                    from .matrix_rules import spec_headers
                    for header in spec_headers(spec):
                        if header not in extra_headers:
                            extra_headers.append(header)
                        if f'#include "{header}"' not in helper_parts:
                            helper_parts.append(f'#include "{header}"')
                    sk_scratch = None
                    if (spec.get('engine') == 'p5c' and "p5ck" in features and persistent_counters
                            and not spec.get('epi') and not spec.get('out_raw')):
                        from .p5c_matrix import split_count
                        sk = split_count(spec['m'], spec['n'], spec['k'])
                        if sk > 1:
                            # P7-C p5ck: deterministic split-K; slice partials in region scratch, per-tile arrival
                            # counters in the persistent counter area (restored to zero by each last arriver).
                            tiles_ = (spec['m'] // 128) * (spec['n'] // 128)
                            sk_scratch = "p7c_sk_" + name
                            layouts[sk_scratch] = TensorLayout((sk * tiles_ * 128 * 128,), (1,), sk_scratch)
                            dtypes[sk_scratch] = "float32"
                            roots.append(sk_scratch)
                            spec = dict(spec, splitk=sk, part=_ident(sk_scratch),
                                        cnt=f"reinterpret_cast<unsigned*>(latch_ctr + {counter_top}LL)")
                            counter_top += tiles_
                            reason += f'; split-K {sk}'
                            matrix_decisions[-1]['reason'] = reason
                    split_part = None
                    if spec.get('dtype') == 'float32' and not spec.get('engine') and split_k:
                        from .matrix_rules import _plan as _rules_plan
                        split_plan = _rules_plan(spec, 128)
                        if split_plan.get('split', 1) > 1 and name not in accumulate:
                            # Release step 14: ordered FP32 split-K for an underfilled long-K edge product; slice
                            # partials in private storage, summed in ascending slice order by the next phase.
                            split_part = "smmsplit_" + name
                            while split_part in names or split_part in layouts:
                                split_part += "_"
                            total_ = _numel(shape)
                            layouts[split_part] = TensorLayout((split_plan['split'] * total_,), (1,), split_part)
                            dtypes[split_part] = "float32"
                            roots.append(split_part)
                            spec = dict(spec, part=_ident(split_part))
                            for header in spec_headers(spec):
                                if header not in extra_headers:
                                    extra_headers.append(header)
                                if f'#include "{header}"' not in helper_parts:
                                    helper_parts.append(f'#include "{header}"')
                            matrix_decisions[-1]['reason'] += f"; ordered FP32 split-K {split_plan['split']}"
                    if split_part is not None:
                        code, extra = emit_group([spec], ident=_ident)
                        add_phase('gemm', op, split_part, code, refs, k=la.shape[-1], matrix_spec=spec, **extra)
                        S_, total_ = split_plan['split'], _numel(shape)
                        terms = " ".join(f"acc = acc + {_ident(split_part)}[{s_ * total_}LL + e];" for s_ in range(1, S_))
                        reduce_code = (f"for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < {total_}LL; "
                                       f"e += (long long)gridDim.x * blockDim.x) {{\n  float acc = {_ident(split_part)}[e];\n  "
                                       f"{terms}\n  {out}[e] = acc;\n}}")
                        add_phase('splitk_reduce', op, name, reduce_code, [split_part], reduction_elements=S_,
                                  arithmetic='ordered FP32 sum of K-slice partials (ascending slice)')
                        continue
                    conv_counters = conv_padded = None
                    if (spec.get('engine') == 'p5c' and spec.get('conv_splitk', 1) > 1 and sk_scratch is None
                            and not spec.get('epi')):
                        # Step 10 (conv_ops.TapsOperands.select): the same p5c split-K body in a flat program, with
                        # private per-tile arrival counters zeroed by an explicit phase each launch.
                        bm_, bn_, wm_ = spec.get('p5c_tile') or (128, 128, 2)
                        sk, tiles_ = spec['conv_splitk'], (spec['m'] // bm_) * (spec['n'] // bn_)
                        if _conv_ops.SPLITK_PHASE:
                            # LATCH-STEP10-SPEED: the (tile, slice) units only store their FP32 partials (no counters);
                            # the next phase sums each element's partials in slice order over the whole grid and
                            # writes the true (M, N) block (padded tiles included), instead of one last-arriving CTA
                            # per tile reading every slice. Same sum order and BF16 rounding: bitwise unchanged.
                            sk_scratch = "conv_sk_" + name
                            layouts[sk_scratch] = TensorLayout((sk * tiles_ * bm_ * bn_,), (1,), sk_scratch)
                            dtypes[sk_scratch] = "float32"
                            roots.append(sk_scratch)
                            spec = dict(spec, splitk=sk, part=_ident(sk_scratch), cnt="nullptr", out=_ident(sk_scratch))
                            code, extra = emit_group([spec], ident=_ident)
                            add_phase('gemm', op, sk_scratch, code, refs, k=la.shape[-1], matrix_spec=spec, **extra)
                            phases[-1]["tiles"] = max(int(phases[-1].get("tiles", 0)), _conv_ops.FULL_GRID_UNITS)
                            add_phase('splitk_reduce', op, name, _conv_ops.splitk_reduce_code(
                                part=_ident(sk_scratch), out=out, dtype=dtypes[name], slices=sk, tiles=tiles_,
                                tile=(bm_, bn_, wm_), padded_n=spec['n'], rows=spec.get('m_extent', spec['m']),
                                cols=spec.get('n_extent', spec['n'])), [sk_scratch], reduction_elements=sk,
                                arithmetic='slice-ordered FP32 sum of split-K partials, one BF16 rounding')
                            reason += f'; split-K {sk} (grid-wide slice reduction phase)'
                            matrix_decisions[-1]['reason'] = reason
                            continue
                        if 'm_extent' in spec or 'n_extent' in spec:
                            # LATCH-STEP10-SPEED: padded product extents land in a private padded buffer; its
                            # (M, N) block is copied out by the phase emitted after the product.
                            conv_padded = "conv_pad_" + name
                            layouts[conv_padded] = TensorLayout((spec['m'], spec['n']), (spec['n'], 1), conv_padded)
                            dtypes[conv_padded] = dtypes[name]
                            roots.append(conv_padded)
                            spec = dict(spec, out=_ident(conv_padded))
                        sk_scratch, conv_counters = "conv_sk_" + name, "conv_skc_" + name
                        layouts[sk_scratch] = TensorLayout((sk * tiles_ * bm_ * bn_,), (1,), sk_scratch)
                        layouts[conv_counters] = TensorLayout((-(-tiles_ // 64) * 64,), (1,), conv_counters)
                        dtypes[sk_scratch] = dtypes[conv_counters] = "float32"
                        roots.extend([sk_scratch, conv_counters])
                        phases.append({'kind': 'matrix_scratch_init', 'names': [conv_counters],
                            'targets': ['conv_ops.private_counter_zero'], 'reads': list(dict.fromkeys(refs)),
                            'writes': [conv_counters],
                            'code': (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {tiles_}LL; '
                                     f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<unsigned*>({_ident(conv_counters)})[i] = 0u;'),
                            'work_items': 1, 'counter_words': tiles_, 'partial_words': 0,
                            'initialization': 'per-launch zero of private split-K arrival counters'})
                        producers[conv_counters] = len(phases) - 1
                        spec = dict(spec, splitk=sk, part=_ident(sk_scratch),
                                    cnt=f"reinterpret_cast<unsigned*>({_ident(conv_counters)})")
                        reason += f'; split-K {sk}'
                        matrix_decisions[-1]['reason'] = reason
                    if (spec.get('engine') == 'p5c' and spec.get('conv_splitk') == 1
                            and ('m_extent' in spec or 'n_extent' in spec) and not spec.get('epi')):
                        # A single K slice still writes full physical tiles. TMA zero-fill protects operand
                        # reads, not these stores: keep the padded product separate from its logical tensor,
                        # exactly as for the counter-based split-K path above, then extract its true extent.
                        conv_padded = "conv_pad_" + name
                        layouts[conv_padded] = TensorLayout((spec['m'], spec['n']), (spec['n'], 1), conv_padded)
                        dtypes[conv_padded] = dtypes[name]
                        roots.append(conv_padded)
                        spec = dict(spec, out=_ident(conv_padded))
                    code, extra = emit_group([spec], ident=_ident)
                    add_phase('gemm', op, conv_padded or name, code, [*refs, *([conv_counters] if conv_counters else [])],
                              k=la.shape[-1], matrix_spec=spec, **extra)
                    if conv_counters is not None:
                        phases[-1]["writes"].extend([sk_scratch, conv_counters])
                        phases[-1]["internal_mutations"] = [sk_scratch, conv_counters]
                        phases[-1]["private_scratch"] = [sk_scratch, conv_counters]
                        producers[sk_scratch] = len(phases) - 1
                        sk_scratch = None
                        # LATCH-STEP10-SPEED: the conv split-K slicing is sized to fill the planned grid; record full
                        # occupancy so idle-CTA placement (step-14 'ecta') does not confine co-scheduled elementwise
                        # work to the few CTAs the (tile, slice) units leave.
                        phases[-1]["tiles"] = max(int(phases[-1].get("tiles", 0)), _conv_ops.FULL_GRID_UNITS)
                    if conv_padded is not None:
                        rows_, cols_, pad_cols = spec['m_extent'] if 'm_extent' in spec else spec['m'], \
                            spec['n_extent'] if 'n_extent' in spec else spec['n'], spec['n']
                        add_phase('gather', op, name,
                                  (f"for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < {rows_ * cols_}LL; "
                                   f"e += (long long)gridDim.x * blockDim.x) {out}[e] = {_ident(conv_padded)}"
                                   f"[(e / {cols_}LL) * {pad_cols}LL + e % {cols_}LL];"), [conv_padded],
                                  initialization="extract the (M, N) block of a padded convolution product")
                    if sk_scratch is not None:
                        phases[-1]["writes"].append(sk_scratch)
                        phases[-1]["internal_mutations"] = [sk_scratch]
                        phases[-1]["counter_effects"] = [{"storage": "runtime_counter_arena",
                            "word_offset": counter_top - tiles_, "words": tiles_,
                            "initialization": "zero once per launch", "completion": "last arriver restores zero"}]
                        producers[sk_scratch] = len(phases) - 1
                    continue
            if all(dtypes[v] == "bfloat16" for v in (a, b, name)):
                if base == "addmm":
                    raise LoweringError("BF16 addmm must be decomposed into mm and pointwise add")
                from .mixed_gemm import CUDA_HELPERS, HEADERS, emit
                if CUDA_HELPERS not in helper_parts:
                    helper_parts.append(CUDA_HELPERS)
                extra_headers.extend(h for h in HEADERS if h not in extra_headers)
                split_grid = 0
                # LATCH-SPLITK-SELECT: the split bodies' contract is the optimized lowering's split-K admission
                # (not 'reference' policy, frozen matrices or a non-sm_90 target), not the convolution lane flag:
                # fixed K parts and part-order FP32 sum (association only, as the admitted FP32 split-K), private
                # partials, per-launch zeroed counters, grid-stride tasks. emit() keeps the shape/layout conditions.
                if GENERIC_SPLIT and split_k and matrix_policy != "frozen":
                    from .p5c_matrix import ADMIT_GRID
                    split_grid = ADMIT_GRID
                code, extra = emit(a, b, out, la, lb, shape, split_grid=split_grid)
                p2 = extra.pop("p2_split", None)
                if p2:
                    # LATCH-STEP10-SPEED: the existing split generic/skinny BF16 bodies (deterministic K parts, fixed
                    # part-order sum by the last arriver) for long-K products the generic tile underfills (e.g. a
                    # 3-channel stem's weight gradient), with private partials and per-launch zeroed counters.
                    part_, cnt_ = "gsk_" + name, "gskc_" + name
                    layouts[part_] = TensorLayout((p2["partial_words"],), (1,), part_)
                    layouts[cnt_] = TensorLayout((p2["counter_words"],), (1,), cnt_)
                    dtypes[part_] = dtypes[cnt_] = "float32"
                    roots.extend([part_, cnt_])
                    phases.append({'kind': 'matrix_scratch_init', 'names': [cnt_],
                        'targets': ['generic_split_counter_zero'], 'reads': list(dict.fromkeys(refs)), 'writes': [cnt_],
                        'code': (f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; '
                                 f'i < {p2["counter_words"]}LL; i += (long long)gridDim.x * blockDim.x) '
                                 f'reinterpret_cast<unsigned*>({_ident(cnt_)})[i] = 0u;'),
                        'work_items': 1, 'counter_words': p2["counter_words"], 'partial_words': 0,
                        'initialization': 'per-launch zero of private split arrival counters'})
                    producers[cnt_] = len(phases) - 1
                    code = (code.replace("/*P2_CNT*/", f"reinterpret_cast<unsigned*>({_ident(cnt_)})")
                            .replace("/*P2_PART*/", _ident(part_)))
                    add_phase("gemm", op, name, code, [*refs, cnt_], **extra)
                    phases[-1]["writes"].extend([part_, cnt_])
                    phases[-1]["internal_mutations"] = [part_, cnt_]
                    phases[-1]["private_scratch"] = [part_, cnt_]
                    producers[part_] = len(phases) - 1   # storage planning needs every root's producer (as conv_sk_)
                    continue
                add_phase("gemm", op, name, code, refs, **extra)
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
            sam, sak = la.strides[-2:]
            sbk, sbn = lb.strides[-2:]
            bsa, bsb = (la.strides[0], lb.strides[0]) if batched else (0, 0)
            vector_safe = ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)
                           and all(s % 4 == 0 for s in (sam if sak == 1 else sak,
                                                       sbn if sbk == 1 else sbk, bsa, bsb,
                                                       la.offset, lb.offset)))
            tiled = not (m % 64 or n % 64 or k % 16) and vector_safe
            if name in accumulate and batched:
                raise LoweringError(f"{name}: accumulating placement requires an unbatched product")
            if matrix_policy != "frozen" and base != "addmm":
                from .matrix_codegen import fp32_async
                destination = None
                if name in write_plans:
                    from .pops_b_policy import FLayout, FactorError, _two
                    plan = write_plans[name]
                    copy_shape = tuple(int(x) for x in _field(values[plan["copy"]], "shape"))
                    try:
                        fl = FLayout.of(TensorLayout(copy_shape, _contiguous(copy_shape), plan["copy"]))
                        for view_op in reversed(plan["chain"]):
                            view_base = str(_field(view_op, "target", _field(view_op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
                            view_args = list(_field(view_op, "args", []))
                            input_shape = tuple(int(x) for x in _field(values[_ref(view_args[0], names)], "shape"))
                            if view_base == "permute":
                                order = [int(d) % len(view_args[1]) for d in view_args[1]]
                                fl = fl.permute([order.index(k) for k in range(len(order))])
                            elif view_base == "transpose":
                                fl = fl.transpose(int(view_args[1]), int(view_args[2]))
                            elif view_base == "t":
                                fl = fl.permute(list(reversed(range(len(fl.dims)))))
                            else:
                                fl = fl.reshape(input_shape)
                        if tuple(fl.shape) != tuple(shape) or len(fl.dims) != 2:
                            raise FactorError("write-through layout rank/shape")
                        (mic, scm, scm1), (nic, scn, scn1) = _two(fl.dims[0]), _two(fl.dims[1])
                        destination = {"mic": mic, "scm": scm, "scm1": scm1, "nic": nic, "scn": scn, "scn1": scn1,
                                       "offset": fl.offset, "pointer": f"{_ident(plan['copy'])} + {fl.offset}LL"}
                    except FactorError:
                        destination = None
                bypassed = []
                la_used, lb_used = la, lb
                if copy_sources:
                    from .pops_b_policy import FLayout, FactorError
                    def through_copy(layout):
                        copy = layout.root
                        if (copy not in copy_sources or copy in write_done or copy_sources[copy] not in layouts
                                or layout.offset):
                            return None
                        try:
                            fl = FLayout.of(layouts[copy_sources[copy]])
                            if tuple(fl.shape) != tuple(layouts[copy].shape):
                                return None
                            if all(d == 1 or st == c for d, st, c in zip(layout.shape, layout.strides, _contiguous(layout.shape))):
                                return fl.reshape(tuple(layout.shape))
                            if len(layout.shape) >= 2:
                                swapped = (*layout.shape[:-2], layout.shape[-1], layout.shape[-2])
                                expect = list(_contiguous(swapped))
                                expect[-1], expect[-2] = expect[-2], expect[-1]
                                if all(d == 1 or st == c for d, st, c in zip(layout.shape, layout.strides, expect)):
                                    return fl.reshape(swapped).transpose(-1, -2)
                        except FactorError:
                            return None
                        return None
                    fa, fb = through_copy(la), through_copy(lb)
                    if fa is not None:
                        la_used = fa; bypassed.append(la.root)
                    if fb is not None:
                        lb_used = fb; bypassed.append(lb.root)
                asynchronous, reason = (None, None)
                if bypassed:
                    asynchronous, reason = fp32_async(a, b, name, la_used, lb_used, shape, ident=_ident, dest=destination,
                        arch=target_arch, split=split_k, occupied_names=names | set(layouts), engine=matrix_engine,
                        grid=(matrix_grids or {}).get(name, 128),
                        counters_at=f"latch_ctr + {counter_top}LL" if persistent_counters else None,
                        accumulate=name in accumulate, preserve_plan=preserve_matrix_plans)
                    if asynchronous is None:
                        bypassed = []
                if asynchronous is None:
                  asynchronous, reason = fp32_async(a, b, name, la, lb, shape, ident=_ident, dest=destination,
                    arch=target_arch, split=split_k, occupied_names=names | set(layouts), engine=matrix_engine,
                    grid=(matrix_grids or {}).get(name, 128),
                    counters_at=f"latch_ctr + {counter_top}LL" if persistent_counters else None,
                    accumulate=name in accumulate, preserve_plan=preserve_matrix_plans)
                matrix_decisions.append({'output': name, 'selected': asynchronous is not None,
                                         'reason': reason,
                                         'candidate': asynchronous['metadata']['engine'] if asynchronous else 'fp32_matrix_selection'})
                if bypassed and asynchronous is not None:
                    matrix_decisions[-1]['copy_bypass'] = bypassed
                if asynchronous is not None:
                    if asynchronous.get("selection_contract") is not None:
                        pops2_layouts[name] = asynchronous["selection_contract"]
                    helpers = asynchronous['helpers']
                    if helpers not in helper_parts:
                        helper_parts.append(helpers)
                    extra_headers.extend(header for header in asynchronous['headers']
                                         if header not in extra_headers)
                    scratch_names = []
                    for scratch in asynchronous['scratch']:
                        value, value_shape = scratch['name'], tuple(scratch['shape'])
                        if value in layouts or value in names:
                            raise LoweringError('internal: admitted matrix scratch name collides')
                        layouts[value] = TensorLayout(value_shape, _contiguous(value_shape), value)
                        dtypes[value] = scratch['dtype']
                        roots.append(value)
                        scratch_names.append(value)
                    initialization = asynchronous['counter_init']
                    if initialization is not None:
                        phase_writes = initialization['writes']
                        # These are conservative scheduling dependencies: the
                        # zero code does not load matrix operands. Waiting for
                        # them prevents hoisting every private scratch lifetime
                        # to interval zero, while retaining independent products.
                        phases.append({'kind': 'matrix_scratch_init', 'names': list(phase_writes),
                            'targets': ['p_ops_b.private_counter_zero'],
                            'reads': list(asynchronous['operand_roots']),
                            'writes': list(phase_writes), 'code': initialization['code'],
                            'work_items': (initialization['counter_words'] + 255) // 256,
                            'counter_words': initialization['counter_words'],
                            'partial_words': initialization['partial_words'],
                            'initialization': 'counter prefix zero; private partials defined by matrix protocol'})
                        for value in phase_writes:
                            producers[value] = len(phases) - 1
                    # One opaque operation owns its private scratch after the
                    # initialization barrier. It stores every partial before
                    # the arrival protocol reads it; only the result is SSA.
                    result_name = name
                    if destination is not None:
                        # The product's result lives in the copy's storage and layout.
                        copy = write_plans[name]["copy"]
                        copy_shape = tuple(int(x) for x in _field(values[copy], "shape"))
                        layouts[copy] = TensorLayout(copy_shape, _contiguous(copy_shape), copy)
                        if copy not in placement:
                            roots.append(copy)
                        roots.remove(name)
                        layouts[name] = TensorLayout(shape, _contiguous(shape), copy)
                        write_done.add(copy)
                        result_name = copy
                        matrix_decisions[-1]['write_through'] = copy
                    add_phase('gemm', op, result_name, asynchronous['code'],
                              [*asynchronous['operand_roots'], *scratch_names],
                              private_scratch=asynchronous['scratch'], **asynchronous['metadata'])
                    phases[-1]['product'] = name   # the product's own output (selection key)
                    if persistent_counters:
                        # The product defines its private partials before reading them.
                        phases[-1]['writes'].extend(scratch_names)
                        for value in scratch_names:
                            producers[value] = len(phases) - 1
                        phases[-1]["counter_effects"] = [{"storage": "runtime_counter_arena",
                            "word_offset": counter_top, "words": asynchronous['metadata']['persistent_counter_words'],
                            "initialization": "zero once per launch", "completion": "last arriver restores zero"}]
                        counter_top += asynchronous['metadata']['persistent_counter_words']
                    continue
            # EpiResid(X=C, Y=C) stores C + v: the caller's former C += v, same operands.
            epilogue = (f"s101h::EpiResid{{{out}, {out}, {n}LL}}" if name in accumulate
                        else f"s101h::EpiStore{{{out}, {n}LL, {m*n}LL}}")
            if tiled and split_k and target_arch == 'sm_90' and base != 'addmm':
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
                    roots.append(part)
                    add_phase('gemm', op, part, split['code'], refs, **split['metadata'])
                    reduce_code = split['reduce']
                    if name in accumulate:
                        reduce_code = reduce_code.replace(f's101h::EpiStore{{{out}, {n}LL, {m*n}LL}}', epilogue)
                    add_phase('splitk_reduce', op, name, reduce_code, [part],
                              reduction_elements=split['metadata']['ksplit'])
                    matrix_decisions.append({'output': name, 'selected': True,
                        'reason': 'underfilled FP32 ordered split-K', 'ksplit': split['metadata']['ksplit']})
                    continue
            if tiled:
                code = f'''{{
  s101h::GemmArgs<1> g{{}};
  g.p[0] = {{{_ident(a)}, {sam}LL, {sak}LL, {bsa}LL, 0, {_ident(b)}, {sbk}LL, {sbn}LL, {bsb}LL, 0, {k}}};
  g.M = {m}; g.N = {n}; g.nb = {batch}; g.nb2 = 1; g.ksplit = 1;
  s101h::gemm_phase<64,64,{str(sak == 1).lower()},{str(sbk == 1).lower()},1>(g, {epilogue}, smem/*VBASE*/);
}}'''
                tiles = batch * (m // 64) * (n // 64)
            else:
                mt, nt = (m + 15) // 16, (n + 15) // 16
                tiles = batch * mt * nt
                # Small generic tile for ragged dimensions and arbitrary positive
                # strides. Fixed ascending-K fmaf accumulation, no padding reads.
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
  if (row < {m} && col < {n}) {out}[batch * {m*n}LL + row * {n}LL + col] = {f"{out}[batch * {m*n}LL + row * {n}LL + col] + " if name in accumulate else ""}acc;
}}'''
            if base == "addmm":
                raise LoweringError("addmm must be decomposed into mm and pointwise add before lowering")
            add_phase("gemm", op, name, code, refs, tiles=tiles,
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
            from . import torch_column_sum
            column_plan = torch_column_sum.plan(_field(graph, "metadata", {}) or {},
                target=str(_field(op, "target", _field(op, "op"))),
                source=source, output=name, source_layout=src, output_layout=layouts[name],
                source_dtype=dtypes[source], output_dtype=dtypes[name], dims=dims,
                accumulate=name in accumulate)
            if column_plan is not None:
                if not compose or target_arch != "sm_90" or dynamic or staged or region_body:
                    raise LoweringError("explicit column-sum requires static composed sm_90 lowering")
                if src.root in virtual:
                    raise LoweringError("explicit column-sum source must retain materialized FP32 storage")
                scratch = name + "__torch_column_partials"
                count = column_plan["geometry"]["partial_floats"]
                if count:
                    if scratch in layouts or scratch in names:
                        raise LoweringError("explicit column-sum scratch name collision")
                    layouts[scratch] = TensorLayout((count,), (1,), scratch)
                    dtypes[scratch] = "float32"
                    roots.append(scratch)
                if torch_column_sum.HEADER not in extra_headers:
                    extra_headers.append(torch_column_sum.HEADER)
                    helper_parts.append(f'#include "{torch_column_sum.HEADER}"')
                for body in torch_column_sum.phases(column_plan, ident=_ident, scratch=scratch):
                    add_phase(body["kind"], op, body["name"], body["code"], body["refs"], **body["extra"])
                continue
            if dynamic and source in dynamic.axes:
                if integer_sum:
                    raise LoweringError("bounded batch currently admits floating reductions")
                code = dynamic.reduction(name, source, src, dims, shape, dtypes[source], dtypes[name],
                                         base == "mean", name in accumulate)
                add_phase("reduction", op, name, code, refs, reduction_elements=_numel(reduction_shape),
                          engine="bounded_active_cta_tree", active_extent=dynamic.axes[source] in dims)
                continue
            terms = []
            if _index32() and _numel(src.shape) < _U32_LIMIT:
                for sequence, subshape, index in [(dims, reduction_shape, "r"), (kept, kept_shape, "o")]:
                    entries = []
                    for j, axis in enumerate(sequence):
                        if src.shape[axis] == 1:
                            continue
                        if entries and entries[-1] is not None and sequence[j - 1] != axis - 1:
                            entries.append(None)  # not adjacent in the source: no merge
                        entries.append((src.shape[axis], src.strides[axis], _numel(subshape[j+1:])))
                    terms += _coord_terms(index, entries)
            else:
                for sequence, subshape, index in [(dims, reduction_shape, "r"), (kept, kept_shape, "o")]:
                    for j, axis in enumerate(sequence):
                        terms.append(f"(({index} / {_numel(subshape[j+1:])}LL) % {src.shape[axis]}LL) * {src.strides[axis]}LL")
            pos = " + ".join(terms) or "0"
            scale = f" / {_float(_numel(reduction_shape))}" if base == "mean" else ""
            reduction_read = f"{_ident(source)}[{pos}]"
            # LATCH-STEP10-SPEED: a reduction over a contiguous run of middle dimensions of a contiguous source whose
            # trailing kept extent is unit-stride (e.g. a spatial mean of pixel-major values: segments = images,
            # rows = pixels, columns = channels) runs one segment per CTA with threads over columns.
            seg_lead = dims[0] if dims else 0
            seg_cols = _numel(src.shape[dims[-1] + 1:]) if dims else 0
            segmented = (SEGMENTED_COLSUM and conv_program and compose and not staged and not region_body
                         and exact_reductions and src.contiguous and not src.offset and dims
                         and dims == list(range(dims[0], dims[-1] + 1)) and seg_lead >= 1 and dims[-1] < len(src.shape) - 1
                         and 32 <= seg_cols <= 1024 and seg_cols % 32 == 0 and _numel(reduction_shape) >= 64
                         and name not in accumulate and dtypes[name] == "float32" and dtypes[source] == "float32")
            inline_text = None
            if region_expression_inline and source in virtual_text:
                # Only a contiguous root source was admitted above. Preserve
                # the reducer's physical leaf indices, tree, scale, scratch,
                # and persistent-counter protocol; change its leaf load only.
                inline_text = virtual_text[source]
                reduction_read = re.sub(r"\bi\b", f"({pos})", inline_text)
                refs = real_refs(refs)
                inlined_reductions.append({"reduction": name, "source": source,
                                           "policy": "same expression at each unchanged reduction leaf"})
            elif (REDUCTION_INLINE and segmented and source in standalone_text and len(value_readers.get(source, [])) == 1
                    and source not in outputs and source not in set(writes.values()) and source not in placement
                    and layouts[source].root == source and src.contiguous and not src.offset
                    and dtypes[source] == "float32" and source not in accumulate
                    # only when the source's operands are read elsewhere too (stored anyway): inlining must not
                    # create new stores of values that only the source consumed
                    and all(len(value_readers.get(r, [])) >= 2 or r not in layouts or layouts[r].root != r
                            for r in standalone_refs[source])):
                # LATCH-STEP10-SPEED: this reduction is its source's only reader; every leaf evaluates the source's
                # own expression over its stored operands at that element (same operations, same rounding), so the
                # source needs no store.
                inline_text = standalone_text[source]
                reduction_read = re.sub(r"\bi\b", f"({pos})", inline_text)
                refs = [r for r in refs if r != source] + [r for r in standalone_refs[source] if r not in refs]
                inlined_reductions.append({"reduction": name, "source": source})
            if integer_sum:
                code = f'''for (long long o = blockIdx.x; o < {_numel(shape)}LL; o += gridDim.x) {{
  unsigned long long acc = 0;
  for (long long r = threadIdx.x; r < {_numel(reduction_shape)}LL; r += blockDim.x)
    acc += static_cast<unsigned long long>({reduction_read});
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
                add_phase("reduction", op, name, code, refs, reduction_elements=_numel(reduction_shape),
                          arithmetic="integer modulo2**64 sum; no FP32 conversion")
                continue
            if dtypes[source] == "bfloat16":
                reduction_read = f"__bfloat162float({reduction_read})"
            from . import torch_row_reduction
            if (exact_reductions and name not in accumulate
                    and torch_row_reduction.eligible(
                        metadata=_field(graph, "metadata", {}),
                        target=str(_field(op, "target", _field(op, "op"))),
                        dtype=dtypes[name], source_dtype=dtypes[source], source_shape=src.shape,
                        source_strides=src.strides, source_offset=src.offset,
                        dims=dims, rows=_numel(shape), extent=_numel(reduction_shape))):
                # This is a separately declared arithmetic contract, not a
                # faster replay of the old CTA tree. Keep the concrete phase
                # out of row_fusion, whose emitter implements the old tree.
                row_position = re.sub(r"\br\b", "pcol", re.sub(r"\bo\b", "prow", pos))
                row_leaf = f"{_ident(source)}[{row_position}]"
                if inline_text is not None:
                    row_leaf = re.sub(r"\bi\b", f"({row_position})", inline_text)
                scale = f" * {_float(1.0 / _numel(reduction_shape))}" if base == "mean" else ""
                code = torch_row_reduction.row_code(out=out, rows=_numel(shape),
                    extent=_numel(reduction_shape), leaf=row_leaf,
                    rounded=lambda expression: rounded(expression, dtypes[name]), scale=scale)
                add_phase("reduction", op, name, code, refs, engine="torch_cuda_vector4_" + base,
                          reduction_elements=_numel(reduction_shape), shared_bytes=0,
                          row_arithmetic=torch_row_reduction.CONTRACT,
                          arithmetic="four FP32 vector accumulators; left-associated vector combine; "
                                     "ascending shuffle-down warp tree; " +
                                     ("FP32 reciprocal multiply" if base == "mean" else "identity projection"))
                continue
            if exact_reductions:
                from .codegen_rules import exact_reduction
                def leaf(position):
                    value = f"{_ident(source)}[{position}]"
                    return f"__bfloat162float({value})" if dtypes[source] == "bfloat16" else value
                if inline_text is not None:
                    def leaf(position, _text=inline_text):
                        return re.sub(r"\bi\b", f"({position})", _text)
                def result(expression):
                    return (f"{out}[o] + ({expression})" if name in accumulate
                            else rounded(expression, dtypes[name]))
                extent_, results_ = _numel(reduction_shape), _numel(shape)
                lead = len(dims)
                if ("colsum" in features and src.contiguous and dims == list(range(lead)) and lead < len(src.shape)
                        and 32 < extent_ <= 768 and (extent_ <= 256 or "wide" in features)
                        and results_ % 32 == 0 and results_ >= 32):
                    # P6-C colsum: reduction over the leading rows of a contiguous [R, C] value. A CTA stages an
                    # R x 32 column tile in shared memory with coalesced loads; one warp per column replays the
                    # identical per-warp tree (slot t = 32 w + lane, folded passes t + 256 p, warp butterflies,
                    # second stage). Same leaves, order and rounding as exact_reduction: bitwise equal.
                    R, C = extent_, results_
                    passes = -(-R // 256)
                    folds = " ".join(f"{{ const int rr = {256 * p} + 32 * w + lane; if (rr < {R}) v[w] = v[w] + T[rr * 33 + cc]; }}"
                                     for p in range(1, passes))
                    code = f'''for (long long cb = blockIdx.x; cb < {C // 32}LL; cb += gridDim.x) {{
  float* T = smem;
  for (int e = threadIdx.x; e < {R * 32}; e += blockDim.x) {{
    const int r = e >> 5, c = e & 31;
    T[r * 33 + c] = {leaf(f"(long long)r * {C}LL + cb * 32LL + c")};
  }}
  __syncthreads();
  const int lane = threadIdx.x & 31;
  for (int cc = threadIdx.x >> 5; cc < 32; cc += blockDim.x >> 5) {{
    const long long o = cb * 32LL + cc;
    float v[8];
#pragma unroll
    for (int w = 0; w < 8; ++w) {{
      {{ const int rr = 32 * w + lane; v[w] = rr < {R} ? 0.0f + T[rr * 33 + cc] : 0.0f; }}
      {folds}
    }}
#pragma unroll
    for (int w = 0; w < 8; ++w) v[w] = s101h::warp_sum(v[w]);
    float second = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {{
      const float partial = __shfl_sync(0xffffffffu, v[w], 0);
      if (lane == w) second = partial;
    }}
    second = s101h::warp_sum(second);
    if (lane == 0) {out}[o] = {result('second' + scale)};
  }}
  __syncthreads();
}}'''
                    add_phase('reduction', op, name, code, refs, engine='exact_col_tile', reduction_elements=R,
                              shared_bytes=R * 33 * 4)
                    continue
                if (_norm_fusion.column_group(name) and name not in accumulate and src.contiguous
                        and dims == list(range(lead)) and lead < len(src.shape) and 32 < extent_ <= 4096
                        and results_ >= 32):
                    # LATCH-TINYMOE-FUSE colgrp: the P-OPS3-C coalesced exact column engine (4 columns per CTA) for
                    # a recognized normalization weight-gradient sum; same frozen per-warp CTA tree, bitwise equal.
                    from . import norm_rules
                    R, C = extent_, results_
                    code = norm_rules.column_code(out=out, columns=C, extent=R, leaf=leaf(f"(long long)pr * {C}LL + o"),
                                                  rounded=result, scale=scale, group=4)
                    add_phase('reduction', op, name, code, refs, engine='exact_col_group4', reduction_elements=R,
                              shared_bytes=8 * 4 * 4)
                    continue
                if (("colreg" in features or _norm_fusion.column_sum(name))
                        and src.contiguous and dims == list(range(lead)) and lead < len(src.shape)
                        and 256 < extent_ <= 2048 and results_ % 32 == 0 and results_ >= 32):
                    # P8-B colreg: reduction over the leading rows of a contiguous [R, C] value with R up to 2048 (no
                    # shared staging). A CTA owns 32 consecutive columns (lane = column, coalesced 128-byte loads); warp
                    # w evaluates CTA-tree slots t = 32 w + l (l = 0..31) for its lane's column: slot sum
                    # 0.0f + x[t] + x[t + 256] + ..., then the lane-0 xor butterfly over the 32 slots symbolically;
                    # the eight warp partials meet in shared memory and warp 0 evaluates the second-stage butterfly.
                    # Same leaves, order and rounding as exact_reduction's per-warp tree: bitwise equal.
                    from .codegen_rules import _butterfly0
                    R, C = extent_, results_
                    passes = -(-R // 256)
                    slot_lines = []
                    for l in range(32):
                        terms = [f"(0.0f + {leaf(f'(long long)(32 * w + {l}) * {C}LL + o')})"]
                        for p_ in range(1, passes):
                            row = f"{256 * p_} + 32 * w + {l}"
                            value = leaf(f"(long long)({row}) * {C}LL + o")
                            terms.append(value if 256 * p_ + 255 < R else f"(({row}) < {R} ? {value} : 0.0f)")
                        acc = terms[0]
                        for t_ in terms[1:]:
                            acc = f"({acc} + {t_})"
                        guard = (f"(32 * w + {l}) < {R} ? " if 255 >= R else "")
                        slot_lines.append(f"const float s{l}_ = {guard}{acc}{' : 0.0f' if guard else ''};")
                    tree = _butterfly0([f"s{l}_" for l in range(32)])
                    second = _butterfly0([f"P[{k} * 32 + lane]" for k in range(8)])
                    body = "\n    ".join(slot_lines)
                    code = f'''for (long long cb = blockIdx.x; cb < {C // 32}LL; cb += gridDim.x) {{
  float* P = smem;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const long long o = cb * 32LL + lane;
  {{
    {body}
    P[w * 32 + lane] = {tree};
  }}
  __syncthreads();
  if (w == 0) {out}[o] = {result(f'({second}){scale}')};
  __syncthreads();
}}'''
                    add_phase('reduction', op, name, code, refs, engine='exact_col_registers', reduction_elements=R,
                              shared_bytes=8 * 32 * 4)
                    continue
                if ("colsplit" in features and src.contiguous and dims == list(range(lead)) and lead < len(src.shape)
                        and extent_ > 768 and 32 <= results_ <= 1024 and results_ % 32 == 0 and persistent_counters
                        and os.environ.get("P9C_COLSPLIT", "1") != "0"):
                    # P9-C colsplit: a long reduction over the leading rows of a contiguous [R, C] value. CTA b sums
                    # its row block for every column (threads over columns: coalesced rows, ascending rows per
                    # column), stores the block partials, and the last CTA to arrive (ticket counter) sums them in
                    # block order. FP32 reassociation of the per-column sum (declared, as msum / split-K; the gates'
                    # tolerances apply); deterministic for a grid size.
                    R, C = extent_, results_
                    scratch = "p9c_colsplit_" + name
                    layouts[scratch] = TensorLayout((1024 * C,), (1,), scratch)
                    dtypes[scratch] = "float32"
                    roots.append(scratch)
                    cnt = f"reinterpret_cast<unsigned*>(latch_ctr + {counter_top}LL)"
                    counter_top += 1
                    part = _ident(scratch)
                    scale_ = scale
                    code = f'''if (gridDim.x <= 1024) {{
  __shared__ unsigned colsplit_last_;
  const int rb_ = ({R} + (int)gridDim.x - 1) / (int)gridDim.x, lo_ = (int)blockIdx.x * rb_;
  const int hi_ = lo_ + rb_ < {R} ? lo_ + rb_ : {R};
  for (int c = threadIdx.x; c < {C}; c += blockDim.x) {{
    float acc = 0.0f;
    int r = lo_;
    for (; r + 15 < hi_; r += 16) {{
      float x_[16];
#pragma unroll
      for (int u = 0; u < 16; ++u) x_[u] = {leaf(f"(long long)(r + u) * {C}LL + c")};
#pragma unroll
      for (int u = 0; u < 16; ++u) acc += x_[u];
    }}
    for (; r < hi_; ++r) acc += {leaf(f"(long long)r * {C}LL + c")};
    {part}[(long long)blockIdx.x * {C}LL + c] = acc;
  }}
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) colsplit_last_ = atomicAdd({cnt}, 1u) == gridDim.x - 1u;
  __syncthreads();
  if (colsplit_last_) {{
    __threadfence();
    for (int c = threadIdx.x; c < {C}; c += blockDim.x) {{
      float acc = 0.0f;
      unsigned b = 0;
      for (; b + 15 < gridDim.x; b += 16) {{
        float x_[16];
#pragma unroll
        for (int u = 0; u < 16; ++u) x_[u] = __ldcg({part} + (long long)(b + u) * {C}LL + c);
#pragma unroll
        for (int u = 0; u < 16; ++u) acc += x_[u];
      }}
      for (; b < gridDim.x; ++b) acc += __ldcg({part} + (long long)b * {C}LL + c);
      const long long o = c;
      {out}[o] = {result('acc' + scale_)};
    }}
    if (threadIdx.x == 0) *{cnt} = 0u;
  }}
  __syncthreads();
}}'''
                    add_phase("reduction", op, name, code, refs, reduction_elements=R, engine="p9c_colsplit")
                    phases[-1]["counter_effects"] = [{"storage": "runtime_counter_arena", "word_offset": counter_top - 1,
                        "words": 1, "initialization": "zero once per launch", "completion": "last arriver restores zero"}]
                    phases[-1]["writes"].append(scratch)
                    producers[scratch] = len(phases) - 1
                    continue
                if segmented:
                    # Segment s (kept leading index) owns rows [s * R, (s + 1) * R) of the [S * R, C] source: G row
                    # groups of threads over columns sum ascending rows (coalesced rows), the groups combine in order.
                    # FP32 reassociation of the segment sum (declared, as colsplit; the gates' tolerances apply);
                    # deterministic for every grid size.
                    R, C = _numel(reduction_shape), seg_cols
                    S = _numel(src.shape[:seg_lead])
                    G = 256 // C if (C <= 256 and 256 % C == 0) else 1
                    if G > 1:
                        body = f'''    {{
      const int c = (int)threadIdx.x % {C}, g = (int)threadIdx.x / {C};
      float acc = 0.0f;
#pragma unroll 8
      for (int r = g; r < {R}; r += {G}) acc += {leaf(f"base_ + (long long)r * {C}LL + c")};
      smem[threadIdx.x] = acc;
      __syncthreads();
      if (threadIdx.x < {C}) {{
        float s_ = 0.0f;
        for (int q = 0; q < {G}; ++q) s_ += smem[q * {C} + threadIdx.x];
        const long long o = (long long)seg * {C}LL + threadIdx.x;
        {out}[o] = {result('s_' + scale)};
      }}
      __syncthreads();
    }}'''
                    else:
                        body = f'''    for (int c = threadIdx.x; c < {C}; c += blockDim.x) {{
      float acc = 0.0f;
#pragma unroll 8
      for (int r = 0; r < {R}; ++r) acc += {leaf(f"base_ + (long long)r * {C}LL + c")};
      const long long o = (long long)seg * {C}LL + c;
      {out}[o] = {result('acc' + scale)};
    }}'''
                    code = f'''for (int seg = blockIdx.x; seg < {S}; seg += gridDim.x) {{
  const long long base_ = (long long)seg * {R * C}LL;
{body}
}}'''
                    add_phase("reduction", op, name, code, refs, reduction_elements=R, engine="segmented_colsum",
                              shared_bytes=1024, arithmetic=f"one CTA per segment ({G} ascending row groups, combined in "
                                                            "order)" + ("; source evaluated per leaf" if inline_text else ""))
                    continue
                split_column = (FLAT_COLSPLIT and (conv_program or FLAT_COLSPLIT_ALL) and compose and not staged and not region_body
                        and src.contiguous
                        and dims == list(range(lead)) and lead < len(src.shape) and extent_ > 2048
                        and 32 <= results_ <= 1024 and name not in accumulate and dtypes[name] == "float32")
                if split_column and not conv_program and _FLAT_COLSPLIT_PROBE.get() is not None:
                    _FLAT_COLSPLIT_PROBE.get().append(name)
                    split_column = False
                if split_column:
                    # LATCH-STEP10-SPEED: a long reduction over the leading rows of a contiguous [R, C] value in a
                    # flat program (e.g. a bias gradient over N*H*W pixel-major rows). A fixed number NB of row
                    # blocks (independent of the grid): block b sums its rows for every column (threads over
                    # columns, G row groups: coalesced rows, ascending rows per group, groups combined in order),
                    # stores the block partials, and the last block to arrive sums them in block order. FP32
                    # reassociation of the per-column sum (declared, as colsplit/msum; the gates' tolerances
                    # apply); deterministic for every grid size.
                    R, C = extent_, results_
                    NB = min(128, -(-R // 64))
                    rb = -(-R // NB)
                    G = 256 // C if (C <= 256 and 256 % C == 0) else 1
                    scratch = "flat_colsplit_" + name
                    while scratch in names or scratch in layouts:
                        scratch += "_"
                    layouts[scratch] = TensorLayout((64 + NB * C,), (1,), scratch)
                    dtypes[scratch] = "float32"
                    roots.append(scratch)
                    cnt = f"reinterpret_cast<unsigned*>({_ident(scratch)})"
                    part = f"({_ident(scratch)} + 64LL)"
                    phases.append({"kind": "reduction_counter_init", "names": [scratch], "targets": [], "reads": [],
                                   "writes": [scratch], "code": f"if (blockIdx.x == 0 && threadIdx.x == 0) *{cnt} = 0u;"})
                    producers[scratch] = len(phases) - 1
                    if G > 1:
                        block_sum = f'''    {{
      const int c = (int)threadIdx.x % {C}, g = (int)threadIdx.x / {C};
      float acc = 0.0f;
      if (g < {G}) {{
#pragma unroll 8
        for (int r = lo_ + g; r < hi_; r += {G}) acc += {leaf(f"(long long)r * {C}LL + c")};
      }}
      smem[threadIdx.x] = acc;
      __syncthreads();
      if (threadIdx.x < {C}) {{
        float s = 0.0f;
        for (int q = 0; q < {G}; ++q) s += smem[q * {C} + threadIdx.x];
        {part}[(long long)blk * {C}LL + threadIdx.x] = s;
      }}
    }}'''
                    elif COLSPLIT_PIPE and C <= 1024:
                        # LATCH-MOESUP-ROW: a thread's K = ceil(C/256) columns advance together, D rows of leaves
                        # loaded before they are added; each column still adds its rows in ascending order from
                        # 0.0f (the same sum, bit for bit), with K*D loads in flight per thread instead of 8.
                        K, D = -(-C // 256), 16
                        valid = lambda k: "true" if 256 * (k + 1) <= C else f"threadIdx.x + {256 * k} < {C}"
                        loads_ = "\n".join(f"          if ({valid(k)}) x_[{k}][d] = {leaf(f'(long long)(r + d) * {C}LL + threadIdx.x + {256 * k}')};"
                                           for k in range(K))
                        adds_ = "\n".join(f"          if ({valid(k)}) acc_[{k}] += x_[{k}][d];" for k in range(K))
                        tails_ = "\n".join(f"        if ({valid(k)}) acc_[{k}] += {leaf(f'(long long)r * {C}LL + threadIdx.x + {256 * k}')};"
                                           for k in range(K))
                        stores_ = "\n".join(f"      if ({valid(k)}) {part}[(long long)blk * {C}LL + threadIdx.x + {256 * k}] = acc_[{k}];"
                                            for k in range(K))
                        block_sum = f'''    {{
      float acc_[{K}];
#pragma unroll
      for (int k = 0; k < {K}; ++k) acc_[k] = 0.0f;
      int r = lo_;
      for (; r + {D} <= hi_; r += {D}) {{
        float x_[{K}][{D}];
#pragma unroll
        for (int d = 0; d < {D}; ++d) {{
{loads_}
        }}
#pragma unroll
        for (int d = 0; d < {D}; ++d) {{
{adds_}
        }}
      }}
      for (; r < hi_; ++r) {{
{tails_}
      }}
{stores_}
    }}'''
                    else:
                        block_sum = f'''    for (int c = threadIdx.x; c < {C}; c += blockDim.x) {{
      float acc = 0.0f;
#pragma unroll 8
      for (int r = lo_; r < hi_; ++r) acc += {leaf(f"(long long)r * {C}LL + c")};
      {part}[(long long)blk * {C}LL + c] = acc;
    }}'''
                    code = f'''{{
  __shared__ unsigned fcs_last_;
  for (int blk = blockIdx.x; blk < {NB}; blk += gridDim.x) {{
    const int lo_ = blk * {rb}, hi_ = lo_ + {rb} < {R} ? lo_ + {rb} : {R};
{block_sum}
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) fcs_last_ = atomicAdd({cnt}, 1u) == {NB - 1}u;
    __syncthreads();
    if (fcs_last_) {{
      __threadfence();
      for (int c = threadIdx.x; c < {C}; c += blockDim.x) {{
        float acc = 0.0f;
        for (int b = 0; b < {NB}; ++b) acc += __ldcg({part} + (long long)b * {C}LL + c);
        const long long o = c;
        {out}[o] = {result('acc' + scale)};
      }}
      if (threadIdx.x == 0) *{cnt} = 0u;
    }}
    __syncthreads();
  }}
}}'''
                    add_phase("reduction", op, name, code, refs + [scratch], reduction_elements=R,
                              engine="flat_colsplit", shared_bytes=1024, internal_mutations=[scratch],
                              private_scratch=[{"name": scratch, "counter_words": 1, "partial_word_offset": 64,
                                                "partial_words": NB * C}],
                              counter_effects=[{"storage": scratch, "word_offset": 0, "words": 1,
                                                "initialization": "zeroed by dependency predecessor",
                                                "completion": "last arriver restores zero"}],
                              arithmetic=f"{NB} fixed row blocks ({G} ascending row groups each), then fixed-order "
                                         "combine of block partials")
                    continue
                # LATCH-STEP10-SPEED: the P6-B thread-per-result replay of the same tree (bitwise equal) when the
                # kept innermost axis is unit-stride, e.g. a spatial mean of pixel-major values.
                coalesced = (FLAT_COALESCED_REDUCTION and conv_program and compose and not staged and not region_body and bool(kept)
                             and src.strides[kept[-1]] == 1 and src.shape[kept[-1]] >= 32)
                selected = exact_reduction(out=out, position=pos, extent=_numel(reduction_shape),
                                           results=_numel(shape), read=leaf, rounded=result, scale=scale,
                                           wide="wide" in features, coalesced_threads=coalesced)
                if selected is not None:
                    code, extra = selected
                    add_phase('reduction', op, name, code, refs, **extra)
                    if (row_fusion and inline_text is None and extra.get('engine') == 'exact_cta_tree_per_warp'
                            and dims == [len(src.shape) - 1] and src.contiguous and src.shape[-1] % 32 == 0
                            and name not in accumulate and dtypes[source] == dtypes[name] == 'float32'):
                        # Structured form of this exact warp-per-row tree for row fusion.
                        phases[-1]['row_reduction'] = {'source': source, 'width': src.shape[-1],
                                                       'rows': _numel(shape), 'scale': scale}
                    continue
            E_ = _numel(reduction_shape)
            preserved_full = (preserve_full_reductions and exact_reductions
                              and _numel(shape) == 1 and E_ >= 64 * 256 and dtypes[name] == "float32")
            flat_full = (preserved_full or (compose and not staged and not region_body and exact_reductions
                         and _numel(shape) == 1 and E_ >= 64 * 256 and dtypes[name] == "float32"
                         and name not in accumulate))
            if flat_full or ("msum" in features and _numel(shape) == 1 and E_ >= 64 * 256
                             and persistent_counters and name not in accumulate):
                # P6-C msum: a long full reduction on every CTA. CTA b sums the contiguous chunk b with the CTA tree
                # (per-thread strided chain, warp butterflies, second stage); the last CTA to arrive (ticket
                # counter) combines the partials in fixed CTA order with the same tree. FP32 reassociation of the
                # single-CTA order (declared; the gates' tolerances apply); deterministic for a grid size.
                scratch = ("compose_full_" if flat_full else "p6c_msum_") + name
                if flat_full:
                    while scratch in names or scratch in layouts:
                        scratch += "_"
                private_full_counter = flat_full and not persistent_counters
                layouts[scratch] = TensorLayout((1088 if private_full_counter else 1024,), (1,), scratch)
                dtypes[scratch] = "float32"
                roots.append(scratch)
                if private_full_counter:
                    # The same reduction calculation used by flat reference
                    # needs only its own initialized counter and partials. Keep
                    # the initializer as allocation producer: the reduction's
                    # internal mutation must not shorten scratch lifetime.
                    cnt = f"reinterpret_cast<unsigned*>({_ident(scratch)})"
                    part = f"({_ident(scratch)} + 64LL)"
                    phases.append({"kind": "reduction_counter_init", "names": [scratch],
                        "targets": [], "reads": [], "writes": [scratch],
                        "code": f"if (blockIdx.x == 0 && threadIdx.x == 0) *{cnt} = 0u;"})
                    producers[scratch] = len(phases) - 1
                else:
                    cnt = f"reinterpret_cast<unsigned*>(latch_ctr + {counter_top}LL)"
                    counter_top += 1
                    part = _ident(scratch)
                full_result = rounded('acc' + scale, dtypes[name])
                if name in accumulate:
                    # Preserve the donor's completed reduction and its one
                    # separately rounded caller addition; never seed the tree
                    # with the existing destination or mix it into partials.
                    full_result = f"{out}[0] + ({full_result})"
                tail = f"{out}[0] = {full_result};"
                full_chunk_loop = f'''  for (long long r = lo_ + threadIdx.x; r < hi_; r += blockDim.x)
    acc += {reduction_read};'''
                if name in balanced_square_sums:
                    full_chunk_loop = square_sums.marker(name) + "\n" + square_sums.lane_loop(reduction_read)
                elif FULL_SUM_BATCH > 1:
                    # LATCH-MOE-EMA-FAST: evaluate FULL_SUM_BATCH strided leaves of this thread's chain before
                    # adding them in the unchanged order (acc += x0; acc += x1; ...), as the single-CTA P-OPS2-D
                    # batched loads do. Leaf values, per-thread order, CTA trees and combine are unchanged; one
                    # 256-thread CTA per SM otherwise keeps one load per thread in flight.
                    depth = FULL_SUM_BATCH
                    terms = '\n'.join(f"    const float x{u} = latch_term(r + {u}LL * blockDim.x);" for u in range(depth))
                    adds = ' '.join(f'acc += x{u};' for u in range(depth))
                    full_chunk_loop = f'''  auto latch_term = [&](long long r) -> float {{ return {reduction_read}; }};
  long long r = lo_ + threadIdx.x;
  for (; r + {depth - 1}LL * blockDim.x < hi_; r += {depth}LL * blockDim.x) {{
{terms}
    {adds}
  }}
  for (; r < hi_; r += blockDim.x)
    acc += latch_term(r);'''
                code = f'''if (gridDim.x <= 1024) {{
  __shared__ unsigned msum_last_;
  const long long chunk_ = ({E_}LL + gridDim.x - 1) / gridDim.x, lo_ = (long long)blockIdx.x * chunk_;
  const long long hi_ = lo_ + chunk_ < {E_}LL ? lo_ + chunk_ : {E_}LL;
  float acc = 0.0f;
{full_chunk_loop}
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {{
      {part}[blockIdx.x] = acc;
      __threadfence();
      msum_last_ = atomicAdd({cnt}, 1u) == gridDim.x - 1u;
    }}
  }}
  __syncthreads();
  if (msum_last_) {{
    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += __ldcg({part} + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {{
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) {{ {tail} *{cnt} = 0u; }}
    }}
  }}
  __syncthreads();
}}'''
                full_arithmetic = ("eight FP32 strided thread lanes, adjacent-pair lane combine; "
                                   "per-CTA contiguous chunk trees, then fixed-order combine of CTA partials"
                                   if name in balanced_square_sums else
                                   "per-CTA contiguous chunk trees, then fixed-order combine of CTA partials")
                if flat_full:
                    # Results==1 still permits retained size-one dimensions in
                    # the address expression. Reference also binds o to zero.
                    code = "{ const long long o = 0;\n" + code + "\n}"
                    if private_full_counter:
                        add_phase("reduction", op, name, code, refs + [scratch], reduction_elements=E_,
                                  engine="multi_cta_full", requires_full_grid=True, max_grid_blocks=1024,
                                  internal_mutations=[scratch], private_scratch=[{"name": scratch,
                                      "counter_words": 1, "partial_word_offset": 64, "partial_words": 1024}],
                                  counter_effects=[{"storage": scratch, "word_offset": 0, "words": 1,
                                      "initialization": "zeroed by dependency predecessor",
                                      "completion": "last arriver restores zero"}],
                                  arithmetic=full_arithmetic)
                    else:
                        # Same body and tree, with a dedicated invocation-owned
                        # counter initialized by the enclosing executor. Every
                        # last arriver restores zero before the next call.
                        add_phase("reduction", op, name, code, refs, reduction_elements=E_,
                                  engine="multi_cta_full", requires_full_grid=True, max_grid_blocks=1024,
                                  internal_mutations=[scratch], private_scratch=[{"name": scratch,
                                      "counter_words": 0, "partial_word_offset": 0, "partial_words": 1024}],
                                  counter_effects=[{"storage": "runtime_counter_arena",
                                      "word_offset": counter_top - 1, "words": 1,
                                      "initialization": "zero once per enclosing executor invocation",
                                      "completion": "last arriver restores zero"}],
                                  arithmetic=full_arithmetic)
                        phases[-1]["writes"].append(scratch)
                        producers[scratch] = len(phases) - 1
                else:
                    add_phase("reduction", op, name, code, refs, reduction_elements=E_, engine="p6c_msum")
                    phases[-1]["counter_effects"] = [{"storage": "runtime_counter_arena", "word_offset": counter_top - 1,
                        "words": 1, "initialization": "zero once per launch", "completion": "last arriver restores zero"}]
                    phases[-1]["writes"].append(scratch)
                    producers[scratch] = len(phases) - 1
                continue
            accumulate_loop = f'''  for (long long r = threadIdx.x; r < {_numel(reduction_shape)}LL; r += blockDim.x)
    acc += {reduction_read};'''
            if batched_reduction_loads and _numel(reduction_shape) >= 8 * 256:
                # P-OPS2-D: issue eight strided loads before adding them in the
                # unchanged per-thread order (acc += x0; acc += x1; ...): exact.
                # P-OPS5-B 'deepred': 32 loads in flight for long single-CTA reductions (same order).
                depth = 32 if "deepred" in features and _numel(reduction_shape) >= 32 * 256 else 8
                loads = '\n'.join(f"    const float x{u} = {reduction_read.replace('(r / ', f'((r + {u * 256}LL) / ').replace('(unsigned)(r)', f'(unsigned)(r + {u * 256}LL)')};"
                                  for u in range(depth))
                adds = ' '.join(f'acc += x{u};' for u in range(depth))
                accumulate_loop = f'''  long long r = threadIdx.x;
  for (; r + {(depth - 1) * 256}LL < {_numel(reduction_shape)}LL; r += {depth * 256}LL) {{
{loads}
    {adds}
  }}
  for (; r < {_numel(reduction_shape)}LL; r += blockDim.x)
    acc += {reduction_read};'''
            code = f'''for (long long o = blockIdx.x; o < {_numel(shape)}LL; o += gridDim.x) {{
  float acc = 0.0f;
{accumulate_loop}
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {out}[o] = {f"{out}[o] + (acc{scale})" if name in accumulate else rounded('acc' + scale, dtypes[name])};
  }}
  __syncthreads();
}}'''
            add_phase("reduction", op, name, code, refs, reduction_elements=_numel(reduction_shape))
            continue
        # Each expression is kept in a register inside a shared-shape region.
        # Storing every SSA result preserves inspectability, with one barrier per region.
        from . import extra_ops
        if extra_ops.uses_expression(base, kwargs, dtypes[name], target):
            for helper in (extra_ops.EXTRA_HELPERS, extra_ops.FIELD_HELPERS):
                if helper not in helper_parts:
                    helper_parts.append(helper)

        def expression(regs):
            rd = lambda i: read(args[i], shape, register_values=regs)
            if target == "latch_ops.scaled_add_f32.default":
                from .scaled_add import expression as scaled_add_expression
                return scaled_add_expression(target, args, kwargs, name, dtypes, names, rd, layouts)
            if target == "latch_ops.mixed_fma64.default":
                from .mixed_fma import expression as mixed_fma_expression
                return mixed_fma_expression(target, args, kwargs, name, dtypes, names, rd, layouts)
            extra = extra_ops.expression(base, target, args, kwargs, name, dtypes, names, rd, "i", shape,
                                         input_names=tuple(inputs))
            if extra is not None:
                return extra
            if base == "slice_backward":
                grad = _ref(args[0], names)
                if rope_u32(name, shape):
                    # LATCH-TINYMOE-FUSE rope: 32-bit index map (index space below 2**31; same element).
                    condition, position = slice_backward_index(args, shape, layouts[grad].shape, "((unsigned)(i))",
                                                               unsigned=True)
                else:
                    condition, position = slice_backward_index(args, shape, layouts[grad].shape, "i")
                return f"({condition} ? {read(args[0], layouts[grad].shape, idx=position)} : 0.0f)"
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
            if base == "lerp":
                return _lerp_scalar_expression(target, args, kwargs, name, dtypes, names, rd,
                    lambda: read(kwargs["weight"], shape, register_values=regs))
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
                if base == "div":
                    from . import torch_scalar_div
                    selection = torch_scalar_div.select(_field(graph, "metadata", {}) or {},
                        target, args, kwargs, name, dtypes, names)
                    if selection is not None:
                        scalar_division_decisions[name] = selection
                        return torch_scalar_div.expression(selection, rd(0))
                return f"({rd(0)} {'*' if base == 'mul' else '/'} {rd(1)})"
            if base == "pow":
                if args[1] == 2:
                    return f"({rd(0)} * {rd(0)})"
                if args[1] == 0.5 and target == "pow.Tensor_Scalar":
                    return f"sqrtf({rd(0)})"
                if "powx" in features and isinstance(args[1], (int, float)) and float(args[1]) in (1.0, 3.0):
                    # P7-C powx: x^1 is x exactly; x^3 as two correctly rounded products (FP32-level, vs powf's
                    # up to 2 ulp); avoids the general powf sequence per element in normalization backward rows.
                    return f"({rd(0)})" if float(args[1]) == 1.0 else f"(({rd(0)} * {rd(0)}) * {rd(0)})"
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
                return f"(1.0f / {rd(0)})"
            if base in {"tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs"}:
                fn = {"abs": "fabsf"}.get(base, base + "f")
                return f"{fn}({rd(0)})"
            if base == "sigmoid":
                if compose and current_precision() == "practical":
                    return f"__fdividef(1.0f, 1.0f + __expf(-{rd(0)}))"
                return f"(1.0f / (1.0f + expf(-{rd(0)})))"
            if base == "silu":
                if compose and current_precision() == "practical":
                    return f"__fdividef({rd(0)}, 1.0f + __expf(-{rd(0)}))"
                return f"({rd(0)} / (1.0f + expf(-{rd(0)})))"
            if base == "silu_backward":
                sigmoid = (f"__fdividef(1.0f, 1.0f + __expf(-{rd(1)}))" if compose and current_precision() == "practical"
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
            if base in ("embedding", "index_select") and name in virtual:
                if "latch_gather_row" not in "".join(helper_parts):
                    helper_parts.append(GATHER_HELPER)
                table = _ref(args[0], names)
                index = _ref(args[1] if base == "embedding" else args[2], names)
                rows_, width_ = layouts[table].shape[0], _numel(layouts[table].shape[1:])
                row = read(args[1] if base == "embedding" else args[2], layouts[index].shape, idx=f"(i / {width_}LL)")
                return read(args[0], layouts[table].shape,
                            idx=f"(latch_gather_row({row}, {rows_}LL) * {width_}LL + i % {width_}LL)")
            if base == "select_backward" and name in virtual:
                grad = _ref(args[0], names)
                rank = len(shape)
                dim_ = int(args[2]) % rank
                pick = int(args[3]) % shape[dim_]
                tail = _numel(shape[dim_ + 1:])
                sfx, iv = ("u", "((unsigned)(i))") if rope_u32(name, shape) else ("LL", "i")
                coord = f"(({iv} / {tail}{sfx}) % {shape[dim_]}{sfx})"
                position = (f"(({iv} / {shape[dim_] * tail}{sfx}) * {tail}{sfx} + {iv} % {tail}{sfx})" if tail != 1
                            else f"({iv} / {shape[dim_]}{sfx})")
                return f"({coord} == {pick}{sfx} ? {read(args[0], layouts[grad].shape, idx=position)} : 0.0f)"
            if base == "cat" and name in virtual:
                pieces = [_ref(a, names) for a in args[0]]
                dim_ = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % len(shape)
                tail = _numel(shape[dim_ + 1:])
                sfx, iv = ("u", "((unsigned)(i))") if rope_u32(name, shape) else ("LL", "i")
                coord = f"(({iv} / {tail}{sfx}) % {shape[dim_]}{sfx})"
                outer = f"({iv} / {shape[dim_] * tail}{sfx})"
                inner = f"({iv} % {tail}{sfx})" if tail != 1 else f"0{sfx}"
                text, start = None, 0
                branches = []
                for piece in pieces:
                    extent = layouts[piece].shape[dim_]
                    position = f"({outer} * {extent * tail}{sfx} + ({coord} - {start}{sfx}) * {tail}{sfx} + {inner})"
                    branches.append((start + extent, read(piece, layouts[piece].shape, idx=position)))
                    start += extent
                text = branches[-1][1]
                for stop, value in reversed(branches[:-1]):
                    text = f"({coord} < {stop}{sfx} ? {value} : {text})"
                return text
            raise LoweringError(f"unsupported operation {target} at {name}")
        if name in virtual:
            virtual_text[name] = rounded(expression({}), dtypes[name])
            virtual_reads[name] = real_refs(refs)
            if name in roots:
                roots.remove(name)
            continue
        # P8-C vprod: values reached through inlined producers are read at mapped (not identity) indices.
        inline_real = real_refs([r for r in refs if r in layouts and layouts[r].root in virtual_text])
        refs = real_refs(refs)
        if REDUCTION_INLINE and conv_program and compose and not staged and not region_body and dtypes[name] == "float32" \
                and not inline_real:
            # LATCH-STEP10-SPEED: the value's expression over stored operands, for a reduction that is its only
            # reader (the reduction evaluates it per leaf; the value itself then needs no store). Identity reads of
            # single-use FP32 producers that have such an expression are inlined too (nested), so a chain like
            # silu(widen(y) + x) is evaluated from y and x.
            text_, srefs_ = expression({}), list(refs)
            for r_ in list(refs):
                lay_ = layouts.get(r_)
                if (REDUCTION_INLINE_NESTED and r_ in standalone_text and len(value_readers.get(r_, [])) == 1
                        and lay_ is not None and lay_.root == r_ and lay_.contiguous and not lay_.offset
                        and tuple(lay_.shape) == tuple(shape)):
                    for form_ in (f"{_ident(r_)}[(i)]", f"{_ident(r_)}[i]"):
                        text_ = text_.replace(form_, f"({standalone_text[r_]})")
                    if _ident(r_) + "[" not in text_:
                        srefs_ = [x for x in srefs_ if x != r_] + [x for x in standalone_refs[r_] if x not in srefs_]
            standalone_text[name] = text_
            standalone_refs[name] = srefs_
        # Store rendered expressions now; closure variables intentionally never escape.
        previous = phases[-1] if phases else None
        safe_region = previous and all(layouts[ref].root not in previous["writes"] or (layouts[ref].contiguous and layouts[ref].shape == shape) for ref in refs)
        if safe_region and inline_real:
            safe_region = all(layouts[ref].root not in previous["writes"] for ref in inline_real)
        # P-OPS5-B group_rules: a conversion joins its same-shape producer chain (registers carry their own
        # C type; reads of BF16 registers widen exactly as reads of BF16 memory). Unrelated chains are split
        # again below, so merging never delays an independent value.
        same_dtype = previous is not None and (previous.get("dtype", "float32") == dtypes[name] or (
            group_rules and dtypes[name] in ("float32", "bfloat16")
            and all(dtypes[n] in ("float32", "bfloat16") for n in previous["names"])))
        if dynamic and previous:
            safe_region = safe_region and all(dynamic.axes.get(v) == dynamic.axes.get(name) for v in previous["writes"])
        if fuse_elementwise and safe_region and previous["kind"] == "elementwise" and previous["shape"] == shape and same_dtype:
            regs = previous["registers"]
            expr = expression(regs)
            reg = "r_" + name
            previous["body"].append(f"const {_CTYPES[dtypes[name]]} {reg} = {rounded(expr, dtypes[name])}; {_padded_store(name, out)} = {f'{out}[i] + ' if name in accumulate else ''}{reg};")
            regs[name] = reg
            previous["names"].append(name)
            previous["targets"].append(str(_field(op, "target", _field(op, "op"))))
            previous["reads"] = list(dict.fromkeys(previous["reads"] + refs))
            previous["writes"].append(name)
            if inline_real:
                previous["vprod_inline"] = True
            producers[name] = len(phases) - 1
        else:
            reg = "r_" + name
            add_phase("elementwise", op, name, "", refs, shape=shape, dtype=dtypes[name], registers={name: reg}, body=[f"const {_CTYPES[dtypes[name]]} {reg} = {rounded(expression({}), dtypes[name])}; {_padded_store(name, out)} = {f'{out}[i] + ' if name in accumulate else ''}{reg};"])
            if inline_real:
                phases[-1]["vprod_inline"] = True
    # LATCH-TINYMOE-XMAT: a deferred segmented product whose pairing partner did not match is emitted here.
    for phase in ragged_experts_lower.flush(phases):
        if phase.get("tma_operands"):
            if '#include "ragged_tma.cuh"\n' not in helper_parts:
                helper_parts.append('#include "ragged_tma.cuh"\n')
            extra_headers.extend(h for h in ragged_experts_lower.TMA_HEADERS if h not in extra_headers)
        if phase.get("flat_claim"):
            if ragged_experts_lower._feature("sepi") and '#define LATCH_CLAIM_SEPI 1\n' not in helper_parts:
                helper_parts.append('#define LATCH_CLAIM_SEPI 1\n')
            if '#include "ragged_claim.cuh"\n' not in helper_parts:
                helper_parts.append('#include "ragged_claim.cuh"\n')
            if ragged_experts_lower.CLAIM_HEADER not in extra_headers:
                extra_headers.append(ragged_experts_lower.CLAIM_HEADER)
        phases.append(phase)
        for value in phase["writes"]:
            producers[value] = len(phases) - 1
    if dynamic:
        all_active_axes = {**dynamic.axes, **dynamic.storage_axes}
        for phase in phases:
            active_values = {value: all_active_axes[value] for value in [*phase["reads"], *phase["writes"]]
                             if value in all_active_axes}
            if active_values or "latch_active_batch" in phase.get("code", "") or any(
                    "latch_active_batch" in line for line in phase.get("body", ())):
                phase["reads"] = list(dict.fromkeys([*phase["reads"], dynamic.input]))
                phase["active_values"] = active_values
    # P8-C vprod: phases that evaluate inlined producers keep their own loop and are opaque to the fusion passes
    # below (which assume same-phase values are read element for element).
    for phase in phases:
        if phase.get("vprod_inline") and phase["kind"] == "elementwise":
            phase["kind"] = "elementwise_inline"
            phase["code"] = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {_numel(phase['shape'])}LL; "
                             f"i += (long long)gridDim.x * blockDim.x) {{\n" + "\n".join(phase["body"]) + "\n}")
    group_records = {}
    if unchanged_record is not None:
        group_records["unchanged_state_writes"] = unchanged_record
    prefix_dead = set()
    if compose and not staged and not region_body and not retain_all:
        from .k8_serial_sum import fuse as fuse_k8_serial_sum, HEADER as k8_serial_sum_header
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["k8_serial_sums"], prefix_dead = fuse_k8_serial_sum(
            phases, layouts, dtypes, _field(graph, "metadata", {}).get("serial_bf16_readout"),
            ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots)
        chain_dead = ()
        if K8_SERIAL_CHAINS:
            from .k8_serial_sum import fuse_chains as fuse_k8_serial_chains
            phases, group_records["k8_serial_chains"], chain_dead = fuse_k8_serial_chains(
                phases, layouts, dtypes, _field(graph, "metadata", {}).get("serial_bf16_readout"),
                ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots)
        prefix_dead = set(prefix_dead) | set(chain_dead)
        if prefix_dead:
            from .matrix_adapters import mma_k8 as k8_body
            wgmma_sum = k8_body.WIDE and k8_body.WGMMA and 'weight_partial' in k8_body.WGMMA_ROLES
            for header_ in ((k8_serial_sum_header,) + (k8_body.WGMMA_HEADERS if wgmma_sum else ())):
                if header_ not in extra_headers:
                    extra_headers.append(header_)
                if f'#include "{header_}"' not in helper_parts:
                    helper_parts.append(f'#include "{header_}"')
            roots = [n for n in roots if n not in prefix_dead]
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    if inlined_reductions:
        group_records["inlined_reduction_sources"] = inlined_reductions
    if group_rules and not retain_all:
        # P-OPS5-B: one elementwise phase per independent register chain.
        from .group_rules import split_components
        phases, group_records["split"] = split_components(phases, dtypes, ctypes=_CTYPES, layouts=layouts,
                                                         level_split="lsplit" in features,
                                                         gated="gatedbw" in features)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    if "csink" in features and compose and fuse_elementwise and not staged and not region_body and not retain_all:
        # LATCH-TINYMOE-25: a same-shape elementwise chain read by exactly one later chain joins it there
        # (values unchanged; removes the store/reload and the early interval of the single-reader chain).
        from .group_rules import sink_single_reader_chains
        inplace_ = {}
        for r_, (_, mode_) in placement.items():
            if mode_.startswith("inplace:"):
                target_ = inputs[int(mode_[8:])]
                inplace_[layouts[target_].root if target_ in layouts else target_] = r_
        phases, group_records["sunk_chains"] = sink_single_reader_chains(
            phases, layouts, ident=_ident, index=_index, inplace=inplace_)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        if "cmulti" in features:
            # LATCH-SELECTIVE-10 (ops.accumulation_rules cmulti): chains read by several same-shape chains join them.
            from .group_rules import sink_multi_reader_chains
            phases, group_records["sunk_multi_reader_chains"] = sink_multi_reader_chains(
                phases, layouts, ident=_ident, index=_index, inplace=inplace_)
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    if "emerge" in features and compose and fuse_elementwise and not staged and not region_body and not retain_all:
        # Release step 14: dependency-aware merging of same-shape elementwise chains (values unchanged).
        from .group_rules import merge_elementwise_chains
        inplace_roots_ = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["merged_chains"] = merge_elementwise_chains(
            phases, layouts, ident=_ident, index=_index, inplace_roots=inplace_roots_)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        if "csink" in features:
            # LATCH-TINYMOE-25: chains that became single-reader through the merges above sink as well.
            phases, group_records["sunk_chains_after_merge"] = sink_single_reader_chains(
                phases, layouts, ident=_ident, index=_index, inplace=inplace_)
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    copy_dead = set(prefix_dead)
    if copy_bypass and not retain_all:
        # Remove elementwise values nothing reads (e.g. copies every product now
        # bypasses) unless returned, committed or placed; repeat to a fixed point.
        kept_dead = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                     | set(placement) | set(writes))
        changed = True
        while changed:
            changed = False
            readers_dead = {}
            for k, phase in enumerate(phases):
                for r in phase["reads"]:
                    if r in layouts:
                        readers_dead.setdefault(layouts[r].root, set()).add(k)
            for k, phase in enumerate(phases):
                if phase["kind"] != "elementwise":
                    continue
                for name in list(phase["names"]):
                    if name in kept_dead or name not in layouts or layouts[name].root != name:
                        continue
                    if readers_dead.get(name, set()) - {k}:
                        continue
                    lines = phase["body"]
                    index_line = next((j for j, line in enumerate(lines) if line.startswith(f"const {_CTYPES[dtypes[name]]} r_{name} =")), None)
                    if index_line is None or any(re.search(rf"\br_{name}\b", line) for line in lines[index_line + 1:]):
                        continue
                    del lines[index_line]
                    position = phase["names"].index(name)
                    phase["names"].pop(position)
                    phase["targets"].pop(position)
                    phase["writes"].remove(name)
                    if name in roots:
                        roots.remove(name)
                    copy_dead.add(name)
                    changed = True
            before = len(phases)
            phases[:] = [phase for phase in phases if phase["kind"] != "elementwise" or phase["names"]]
            changed |= len(phases) != before
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    row_fusions, row_dead = [], set()
    if row_fusion and not retain_all:
        # P-OPS2-D: one warp per row evaluates an exact last-dimension reduction
        # together with its same-row elementwise producers and consumers.
        from .row_fusion import fuse_rows
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, row_fusions, row_dead = fuse_rows(phases, layouts, dtypes, ident=_ident, index=_index,
                                                  kept=kept_roots, inplace_roots=inplace_roots,
                                                  flat=bool(group_rules),
                                                  multi_warp=8 if "wrow" in features else 2 if "wrow2" in features else 0,
                                                  absorb_early="rngel" in features, hoist="rhoist" in features,
                                                  sink="rsink" in features, clamp="rclamp" in features,
                                                  pipe="rpipe" in features,
                                                  orders=_gather_orders(phases, layouts) if GATHER_ORDER else None)
        if any("latch_row_gather" in (phase.get("code") or "") for phase in phases):
            from .row_fusion import ROW_GATHER_HELPER
            if ROW_GATHER_HELPER not in helper_parts:
                helper_parts.append(ROW_GATHER_HELPER)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        roots = [n for n in roots if n not in row_dead]
    epilogues = []
    if epilogue_fusion and not retain_all:
        # P-OPS2-D: same-index elementwise consumers evaluated in B product stores.
        from .epilogue_fusion import fuse_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, epilogues, epilogue_dead = fuse_epilogues(phases, layouts, dtypes, ident=_ident, index=_index,
                                                          kept=kept_roots, inplace_roots=inplace_roots,
                                                          accumulate=accumulate)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= epilogue_dead
        roots = [n for n in roots if n not in epilogue_dead]
    if "softmax" in features and not retain_all:
        # P-OPS5-B: same-size elementwise consumers attach to the warp-per-row softmax store loop.
        from .group_rules import fuse_softmax_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["softmax_epilogues"], softmax_dead = fuse_softmax_epilogues(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= softmax_dead
        roots = [n for n in roots if n not in softmax_dead]
    if "chains" in features and not retain_all:
        # P-OPS5-B: non-adjacent same-size elementwise chains share one loop and level.
        from .group_rules import merge_chains
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["merged_chains"], chain_dead = merge_chains(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= chain_dead
        roots = [n for n in roots if n not in chain_dead]
    if "prologue" in features and not retain_all:
        # P-OPS5-B: products recompute a row group's BF16 operand in their A prologue.
        from .group_rules import fuse_row_prologues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        phases, group_records["row_prologues"], prologue_dead = fuse_row_prologues(
            phases, layouts, dtypes, ident=_ident, kept=kept_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= prologue_dead
        roots = [n for n in roots if n not in prologue_dead]
    if "cluster" in features and not retain_all:
        # P-OPS5-B: a row group consuming one BF16 product runs in its cluster-distributed epilogue.
        from .group_rules import fuse_cluster_rows
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        phases, group_records["cluster_rows"], _ = fuse_cluster_rows(
            phases, layouts, dtypes, ident=_ident, kept=kept_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    if "rowepi" in features and not retain_all:
        # P-OPS5-B: a row group consuming one BF16 product runs in that product's full-row-tile epilogue.
        from .group_rules import fuse_row_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        phases, group_records["row_epilogues"], rowepi_dead = fuse_row_epilogues(
            phases, layouts, dtypes, ident=_ident, kept=kept_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= rowepi_dead
        roots = [n for n in roots if n not in rowepi_dead]
    flat_epilogue_cost = None
    if compose and not staged and not region_body:
        from .pops2b_rules import EPI_COST_LIMIT
        flat_epilogue_cost = EPI_COST_LIMIT
    if "p5ce" in features and not retain_all:
        # P7-C: same-index consumers evaluated in p5c (warpgroup-MMA) product register epilogues.
        from .group_rules import fuse_p5c_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["p5c_epilogues"], p5ce_dead = fuse_p5c_epilogues(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots,
            vector="p5cv" in features, cost_limit=flat_epilogue_cost)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= p5ce_dead
        roots = [n for n in roots if n not in p5ce_dead]
    if "tmae" in features and compose and not staged and not region_body and not retain_all:
        # Release step 14: same-index consumers evaluated in staged epilogues of provider TMA + wgmma products.
        from .group_rules import TMA_EPI_COST_LIMIT, fuse_tma_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        # Frozen matrix donor: retain BF16 rounding while omitting a sole
        # material widening. Admission proves layout, ownership and epilogue.
        from . import matrix_widen
        if matrix_widen.BATCH_TMA_WIDEN:
            phases, group_records["batched_tma_widenings"], widen_dead = matrix_widen.fuse(
                phases, layouts, dtypes, ident=_ident, index=_index,
                kept=kept_roots, inplace_roots=inplace_roots)
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
            row_dead |= widen_dead
            roots = [n for n in roots if n not in widen_dead]
        phases, group_records["tma_epilogues"], tmae_dead = fuse_tma_epilogues(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots,
            cost_limit=TMA_EPI_COST_LIMIT, prefetch="tepf" in features, xdiv="xdiv" in features,
            prefetch_rows="tepr" in features)
        if any(phase.get("tma_epilogue") for phase in phases):
            if "latch_tma_epi.cuh" not in extra_headers:
                extra_headers.append("latch_tma_epi.cuh")
            if '#include "latch_tma_epi.cuh"' not in helper_parts:
                helper_parts.append('#include "latch_tma_epi.cuh"')
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= tmae_dead
        roots = [n for n in roots if n not in tmae_dead]
    if compose and not staged and not region_body and not retain_all:
        from .k8_widen import fuse as fuse_k8_widen
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["k8_widenings"], k8_dead = fuse_k8_widen(
            phases, layouts, dtypes, ident=_ident, index=_index,
            kept=kept_roots, inplace_roots=inplace_roots)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= k8_dead
        roots = [n for n in roots if n not in k8_dead]
        # LATCH-MOE-EMA-FAST: a gathered summand and its vector4 row sum in one warp-per-row pass.
        from .gathered_row_sum import fuse as fuse_gathered_row_sums
        phases, group_records["gathered_row_sums"], gathered_dead = fuse_gathered_row_sums(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots)
        if group_records["gathered_row_sums"]:
            from .row_fusion import ROW_GATHER_HELPER
            if ROW_GATHER_HELPER not in helper_parts:
                helper_parts.append(ROW_GATHER_HELPER)
        row_dead |= gathered_dead
        # LATCH-MOE-EMA-FAST: an ordered indexed gradient evaluates its single-line elementwise source per read.
        from .indexed_source_inline import fuse as fuse_indexed_sources
        phases, group_records["indexed_source_inlines"], indexed_dead = fuse_indexed_sources(
            phases, layouts, dtypes, ident=_ident, kept=kept_roots, inplace_roots=inplace_roots)
        row_dead |= indexed_dead
        # LATCH-MOE-EMA-FAST: a multi-CTA full sum evaluates its single-line elementwise source at each leaf.
        from .full_sum_inline import fuse as fuse_full_sum_sources
        phases, group_records["full_sum_inlines"], full_sum_dead = fuse_full_sum_sources(
            phases, layouts, dtypes, ident=_ident, kept=kept_roots, inplace_roots=inplace_roots)
        row_dead |= full_sum_dead
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        roots = [n for n in roots if n not in gathered_dead and n not in indexed_dead and n not in full_sum_dead]
    if group_rules and not retain_all:
        # P-OPS5-B: same-index consumers evaluated in BF16 product fragment stores.
        from .group_rules import fuse_bf16_epilogues
        kept_roots = ({layouts[n].root for n in [*outputs, *writes.values()] if n in layouts}
                      | set(placement) | set(writes))
        inplace_roots = {r for r, (_, mode) in placement.items() if mode.startswith("inplace:")}
        phases, group_records["bf16_epilogues"], epilogue_dead = fuse_bf16_epilogues(
            phases, layouts, dtypes, ident=_ident, index=_index, kept=kept_roots, inplace_roots=inplace_roots,
            ctypes=_CTYPES, prefetch=("early" if "epiearly" in features else "late" if "epilate" in features
                                      else "pre" in features),
            pair="epipair" in features, cost_limit=flat_epilogue_cost)
        producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
        row_dead |= epilogue_dead
        roots = [n for n in roots if n not in epilogue_dead]
        if "rot" in features:
            from .group_rules import rotate_products
            group_records["rotated_products"] = rotate_products(phases)
    commit = len(phases)
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
    in_place_commits = {}
    if compose and not staged and not region_body and not placement and not retain_all:
        from .state_placement import last_use_state_aliases
        in_place_commits = last_use_state_aliases(phases, layouts, dtypes, writes, inputs, ident=_ident)
        aliased = {source: target for target, source in in_place_commits.items()}
        for name, layout in list(layouts.items()):
            if layout.root in aliased:
                layouts[name] = TensorLayout(layout.shape, layout.strides, aliased[layout.root], layout.offset)
        roots = [name for name in roots if name not in aliased]
    if barrier_schedule not in ("phase", "dependency"):
        raise LoweringError(f"unknown barrier schedule {barrier_schedule}")
    # P-OPS-E elide_stores: an elementwise result consumed only through its
    # register inside its own fused phase, and never returned, committed or
    # placed, need not be written to memory or allocated. Values are unchanged.
    dead = set(row_dead) | copy_dead
    taps_dead = conv_taps.unmaterialized()
    if taps_dead:
        stray = sorted({r for phase in phases for r in phase["reads"] if r in layouts and layouts[r].root in taps_dead})
        if stray:
            raise LoweringError(f"deferred convolution taps read without materialization: {stray}")
        dead |= taps_dead
        roots = [name for name in roots if name not in taps_dead]
    if elide_stores and not retain_all:
        kept = {layouts[name].root for name in [*outputs, *writes.values()] if name in layouts} | set(placement)
        readers = {}
        for phase_id, phase in enumerate(phases):
            for ref in phase["reads"]:
                if ref in layouts:
                    readers.setdefault(layouts[ref].root, []).append((phase_id, ref))
        for phase_id, phase in enumerate(phases):
            if phase["kind"] != "elementwise":
                continue
            for name in phase["names"]:
                if (layouts[name].root == name and name not in kept and name in roots
                        and all(pid == phase_id and ref == name for pid, ref in readers.get(name, []))):
                    dead.add(name)
                    store = f"; {_padded_store(name, _ident(name))} = r_{name};"
                    padded_operands.pop(name, None)       # step 02: an elided value has no storage to pad
                    phase["body"] = [line.replace(store, ";") if line.startswith(f"const {_CTYPES[dtypes[name]]} r_{name} =") else line
                                     for line in phase["body"]]
        roots = [name for name in roots if name not in dead]
    if PIXELS_EPILOGUE and conv_program and compose and not staged and not region_body and not retain_all:
        # LATCH-STEP10-SPEED: an elementwise phase that reads only a conv3_pixels transpose's (FP32) result, element
        # for element, is evaluated in that transpose's store loop (same expressions on the same value, at the same
        # flat index), saving a full pass over the transposed tensor.
        from .conv_ops import PIXELS_TARGET as _PIX
        absorbed_ = set()
        for g_, gp in enumerate(phases):
            if gp.get("kind") != "gather" or gp.get("targets") != [_PIX] or "/*PIXELS_EPI*/" not in gp.get("code", ""):
                continue
            root_ = gp["writes"][0]
            if dtypes.get(root_) != "float32" or root_ not in layouts:
                continue
            total_ = layouts[root_].numel
            for e_ in range(g_ + 1, len(phases)):
                ce = phases[e_]
                if (e_ in absorbed_ or ce.get("kind") != "elementwise" or not ce.get("body")
                        or len(ce["body"]) != len(ce["names"]) or ce.get("vprod_inline")
                        or _numel(tuple(ce["shape"])) != total_ or not ce["reads"]):
                    continue
                if any(r not in layouts or layouts[r].root != root_ or not layouts[r].contiguous or layouts[r].offset
                       or layouts[r].numel != total_ for r in ce["reads"]):
                    continue
                lines_ = []
                for line in ce["body"]:
                    for r in ce["reads"]:
                        line = line.replace(f"{_ident(r)}[(i)]", "v").replace(f"{_ident(r)}[i]", "v")
                    lines_.append(line)
                if any(_ident(r) + "[" in line for line in lines_ for r in ce["reads"]):
                    continue                                    # a read not at the identity index remains
                gp["code"] = gp["code"].replace("/*PIXELS_EPI*/", " ".join(lines_) + " /*PIXELS_EPI*/")
                gp["names"] = gp["names"] + ce["names"]
                gp["targets"] = gp["targets"] + ce["targets"]
                gp["writes"] = gp["writes"] + ce["writes"]
                absorbed_.add(e_)
        if absorbed_:
            phases[:] = [p_ for k_, p_ in enumerate(phases) if k_ not in absorbed_]
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    if compose and not staged and not region_body and not retain_all:
        # LATCH-TINYMOE-XMAT: tile-local elementwise consumers of flat segmented right products (feature "consume").
        from .ragged_experts_lower import absorb_consumers as _ragged_absorb
        _absorbed = _ragged_absorb(phases, layouts, _ident)
        if _absorbed:
            group_records["ragged_tile_consumers"] = [phases[k_]["names"] for k_ in sorted(_absorbed)]
            phases[:] = [p_ for k_, p_ in enumerate(phases) if k_ not in _absorbed]
            producers = {w: k for k, phase in enumerate(phases) for w in phase["writes"]}
    intervals = _barrier_intervals(phases, layouts, barrier_schedule == "dependency")
    if "ulate" in features and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # TINYMOE-SELECTIVE-CADENCE (ops.update_schedule late_updates): bodies no loss/gradient output depends on
        # start after the last loss/gradient interval (moved later only; dependency order and values unchanged).
        from .update_schedule import late_updates
        targets_ = ([structured_outputs["loss"]] + list(structured_outputs.get("grads", {}).values())
                    if isinstance(structured_outputs, dict) else list(outputs[:1]))
        # LATCH-SELECTIVE-10: gradient_outputs="consumed" removes the public gradients, not their role as targets.
        targets_ += [v for v in ((_field(graph, "metadata", {}) or {}).get("consumed_gradient_values") or ())
                     if v in layouts and v not in targets_]
        intervals, group_records["late_updates"] = late_updates(
            phases, intervals, layouts, [_ref(t, names) or t for t in targets_])
    if "lsink" in features and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # LATCH-TRAIN-MEMORY (ops.memory_rules "sink"): bodies move later, never earlier, to the interval before their
        # earliest complete-contract successor when that lowers live arena bytes (values unchanged).
        from .memory_rules import lifetime_sink
        _inplace_targets = {layouts[inputs[int(mode[8:])]].root for _, (_, mode) in placement.items()
                            if mode.startswith("inplace:")}
        _pinned = {producers[r] for r, (_, mode) in placement.items() if mode.startswith("inplace:") and r in producers}
        _pinned |= {k for k, ph in enumerate(phases)
                    if any(n in layouts and layouts[n].root in _inplace_targets for n in ph["reads"])}
        intervals, group_records["lifetime_sink"] = lifetime_sink(phases, intervals, layouts, dtypes, roots, _pinned)
    if K8_SPLIT_INTERVALS and barrier_schedule == "dependency" and not retain_all:
        # LATCH-MOE-EMA-100: k8 readout products that share an interval with a serial weight sum run in the next
        # interval instead (moved later only; no reader shares their interval, values unchanged).
        _split = []
        for _members in intervals:
            _serial = [i for i in _members if phases[i].get("engine") == "latch_matrix:mma_k8_serial_sum"]
            _wide = [i for i in _members if phases[i].get("engine") == "latch_matrix:mma_k8_readout"]
            if _serial and _wide:
                _split += [[i for i in _members if i not in _wide], _wide]
                group_records.setdefault("k8_split_intervals", []).append([phases[i]["names"][0] for i in _wide])
            else:
                _split.append(_members)
        intervals = _split
    from . import group_rules as _gr_split
    if _gr_split.SPLIT_STREAM_ROWS and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        intervals, split_ = _gr_split.split_stream_rows(phases, intervals)
        if split_:
            group_records["split_stream_rows"] = split_
    if "ecta" in features and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # Release step 14: elementwise phases wait in the interval before their first reader when its matrix tiles
        # leave CTAs idle (emission below restricts them to those CTAs).
        from .group_rules import sink_to_idle
        intervals, group_records["ecta_sunk"] = sink_to_idle(
            phases, intervals, layouts, lambda i: int(phases[i].get("tiles", 0)) if phases[i]["kind"] == "gemm" and any(
                token in phases[i].get("code", "") for token in ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/")) else 0)
    pair_streams_ = {}
    if "pstream" in features and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # LATCH-MOESUP-ROW: a streaming vocabulary gradient shares an independent store-bound product's interval on a
        # disjoint CTA slice (group_rules.pair_streams; values unchanged).
        from .group_rules import pair_streams
        intervals, pair_streams_, group_records["paired_streams"] = pair_streams(phases, intervals, layouts)
    if "tmaj" in features and compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # Release step 14: joint tile choice for provider TMA products sharing an interval (arithmetic unchanged).
        from .group_rules import joint_tma_plans
        phases, group_records["joint_tma_plans"] = joint_tma_plans(phases, intervals, ident=_ident)
    if compose and barrier_schedule == "dependency" and not staged and not retain_all:
        # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'p5c'): plain provider TMA products of short/long contractions
        # re-emitted on the same-class P5C body (bitwise; ops.matrix_bodies.respec_p5c).
        from .matrix_bodies import respec_p5c, active_rules
        from .group_rules import ECTA_GRID as _p5_grid
        if active_rules():
            group_records["matrix_bodies"] = active_rules()
        phases, p5_records_, p5_headers_ = respec_p5c(phases, ident=_ident, grid=_p5_grid, intervals=intervals)
        if p5_records_:
            group_records["matrix_bodies_p5c"] = p5_records_
            extra_headers.extend(h for h in p5_headers_ if h not in extra_headers)
            for header in p5_headers_:
                include = f'#include "{header}"'
                if include not in helper_parts:
                    helper_parts.append(include)
    deferred = []
    if defer_accumulate and barrier_schedule == "dependency" and not retain_all:
        intervals, deferred = _defer_accumulate(phases, intervals, layouts, accumulate, policy=defer_policy,
                                                rotated="defer" in features)
    if STEP06_SLACK and staged and barrier_schedule == "dependency" and not retain_all and "p5c" in features:
        from . import loop_slack
        from .group_rules import np_cost as _np_cost
        _side = {name for name, (_, mode) in placement.items() if mode in ("store", "accumulate")}
        intervals, group_records["step06_slack"] = loop_slack.place(phases, intervals, layouts, _side,
                                                                    footprint=_footprint, np_cost=_np_cost)
    # P7-D e0fcb8e: selected rkcq storage policy delays only explicit
    # reverse rematerializations. Experimental broader sinki remains excluded.
    if "sink" in features and barrier_schedule == "dependency" and not retain_all:
        metadata = _field(graph, "metadata", {}) or {}
        sinkable = set(metadata.get("recomputed_in_reverse", ()))
        if sinkable:
            intervals, group_records["sunk_recomputation"] = _sink_recomputed(phases, intervals, layouts, sinkable)
    if features & {"joint", "jointbig"} and barrier_schedule == "dependency" and not retain_all:
        # P-OPS5-B: products sharing a dependency interval are planned for their share of the grid.
        from .group_rules import joint_plans
        group_records["joint_plans"] = joint_plans(phases, intervals,
                                                   shared_cap=192 * 1024 if "jointbig" in features else None,
                                                   light="jointlight" in features)
    if "lpt" in features and barrier_schedule == "dependency" and not retain_all:
        # P6-C: joint tile choice and cost-aware CTA offsets for the products of one interval (bitwise equal).
        from .group_rules import balanced_plans
        group_records["balanced_plans"] = balanced_plans(phases, intervals,
                                                         hints=_LPT_HINTS.get() if "lptx" in features else None)
    if features & {"slicek", "slicek2"} and not retain_all:
        # P6-C: sliced-K BF16 product tiles (k groups across warps; FP32 reassociation of the K sum).
        from .group_rules import sliced_k
        group_records["sliced_k"] = sliced_k(phases, groups_first=(2, 4) if "slicek2" in features else (4, 2))
    if features & {"epism", "epismh"} and not retain_all:
        # P6-C: staged, coalesced BF16 product epilogues (bitwise identical values); epismh: heavy epilogues only.
        from .group_rules import staged_epilogues
        group_records["staged_epilogues"] = staged_epilogues(phases, heavy_only="epism" not in features,
                                                                unroll=1 if "epu1" in features else 2 if "epu2" in features else 4)
    if "ring" in features and not retain_all:
        # P6-C: BF16 product stage rings deep enough to keep every K slice of a task in flight.
        from .group_rules import deep_rings
        group_records["deep_rings"] = deep_rings(phases)
    if features & {"splitk", "splitk16"} and barrier_schedule == "dependency" and not retain_all and persistent_counters:
        # P-OPS5-B: deterministic split-K for long-K BF16 products (slice partials summed in fixed order by the
        # last-arriving slice; FP32 association of the K sum changes, the BF16 rounding point does not).
        from .group_rules import split_k_plans, _render
        group_records["split_k"] = []
        for index, splits in split_k_plans(phases, choices=(16, 8, 4, 2) if "splitk16" in features else (4, 2)):
            phase = phases[index]
            call = phase["popsc_call"]
            tiles = int(phase["tiles"])
            bm, bn = int(call["targs"][0]), int(call["targs"][1])
            scratch = "pops5b_sk_" + phase["names"][0]
            layouts[scratch] = TensorLayout((tiles * splits * bm * bn,), (1,), scratch)
            dtypes[scratch] = "float32"
            roots.append(scratch)
            phase["writes"].append(scratch)
            producers[scratch] = index
            call["split"] = {"S": splits, "part": _ident(scratch),
                             "cnt": f"reinterpret_cast<unsigned*>(latch_ctr + {counter_top}LL)"}
            phase["counter_effects"] = [{"storage": "runtime_counter_arena", "word_offset": counter_top,
                "words": tiles, "initialization": "zero once per launch", "completion": "last arriver restores zero"}]
            counter_top += tiles
            phase["tiles"] = tiles * splits
            _render(phase)
            group_records["split_k"].append({"product": phase["names"][0], "splits": splits, "tiles": tiles,
                                             "k": int(phase["matrix_spec"]["k"])})
    # LATCH-TINYMOE-XMAT: cross-product CTA rotation of flat segmented products (feature "rotate").
    from .ragged_experts_lower import rotate_intervals as _ragged_rotate
    _rotated = _ragged_rotate(phases, intervals)
    if _rotated:
        group_records["ragged_rotated"] = _rotated
    cast_fusions_ = {}
    if ("fscast" in features and "fsload" in features and "fsbatch" in features and compose
            and barrier_schedule == "dependency" and not staged and not region_body and not retain_all):
        # LATCH-TRAIN-SYNC (ops.cast_sums): exact square sums over slices of a single-line elementwise producer's root
        # move into the producer's interval (before arena planning); emission fuses them (values unchanged).
        from . import cast_sums as _cast_sums
        intervals, cast_fusions_, cast_records_ = _cast_sums.plan(phases, intervals, layouts, _ident,
                                                                  lambda s_: phases[s_].get("code", ""))
        if cast_records_:
            group_records["cast_sums"] = cast_records_
    interval_of = {index: k for k, members in enumerate(intervals) for index in members}
    for root, (index, mode) in placement.items():
        if not mode.startswith("inplace:"):
            continue
        # The result overwrites its input: every other read of that input must
        # precede the writing interval; in the writing phase, reads of the same
        # element must precede the store (same index, earlier or same line).
        target, writer = inputs[int(mode[8:])], producers[root]
        if not layouts[target].contiguous or phases[writer]["kind"] != "elementwise":
            raise LoweringError(f"in-place result {root} requires a contiguous elementwise update")
        lines = phases[writer]["body"]
        store_line = next(k for k, line in enumerate(lines) if line.startswith(f"const {_CTYPES[dtypes[root]]} r_{root} ="))
        if any(f"{_ident(target)}[" in line for line in lines[store_line + 1:]):
            raise LoweringError(f"in-place result {root} is followed by a read of its overwritten input")
        for phase_id, phase in enumerate(phases):
            if phase_id != writer and any(layouts[ref].root == target for ref in phase["reads"] if ref in layouts):
                if interval_of[phase_id] >= interval_of[writer]:
                    raise LoweringError(f"in-place result {root}: input {target} is read at or after its overwrite")
        if any(ref in layouts and layouts[ref].root == target and (ref != target or not layouts[ref].contiguous)
               for ref in phases[writer]["reads"]):
            raise LoweringError(f"in-place result {root}: same-phase read of {target} is not the same element")
    unwritten = sorted(set(padded_operands) - padded_writers)
    if unwritten:
        raise LoweringError(f"padded matrix values without a pitch-aware writer: {unwritten}")
    sizes = {name: ((layouts[name].numel if name not in padded_operands else
                     _padding.storage_elements(layouts[name].shape, layouts[name].strides)) * _BYTES[dtypes[name]] + 3) // 4
             for name in roots}
    # LATCH-TRAIN-CORRECTNESS: a root may be written by several phases (per-launch zeroed split-product arrival
    # counters are written by their initialization phase and then by the product; scratch reused by a later
    # phase). `producers` holds its LAST writer, so a lifetime that started there let the arena share the root's
    # storage with values still live while the EARLIER write runs (it clobbered the attention-backward cotangent
    # of gated TinyMoE programs). Every root is now live from its earliest writer.
    writers_of = {}
    for phase_id, phase in enumerate(phases):
        for name in phase["writes"]:
            if name in layouts:
                writers_of.setdefault(layouts[name].root, []).append(phase_id)
    first_writer = {name: min([producers[name], *writers_of.get(name, ())]) for name in roots}
    classes = {name: ("live", first_writer[name], producers[name]) for name in roots}
    for phase_id, phase in enumerate(phases):
        for name in phase["reads"]:
            root = layouts[name].root
            if root in classes:
                _, first, last = classes[root]
                classes[root] = ("live", first, max(last, phase_id))
    for name in outputs + list(writes.values()):
        root = layouts[name].root
        if root in classes:
            _, first, _ = classes[root]
            classes[root] = ("live", first, commit)
    if retain_all:
        classes = {name: ("live", first_writer[name], commit) for name in roots}
    plan_commit = commit
    if barrier_schedule == "dependency":
        # Lifetimes in interval units: earliest writer interval through the latest
        # interval of any reader (phase order no longer implies interval order).
        plan_commit = len(intervals)
        spans = {name: [min(interval_of[k] for k in [first, producers[name], *writers_of.get(name, ())])] * 2
                 for name, (_, first, _) in classes.items()}
        for phase_id, phase in enumerate(phases):
            for name in phase["reads"]:
                root = layouts[name].root
                if root in spans:
                    spans[root][1] = max(spans[root][1], interval_of[phase_id])
        classes = {name: (kind, spans[name][0], plan_commit if last >= commit else spans[name][1])
                   for name, (kind, first, last) in classes.items()}
    from . import arena_assign   # LATCH-BUILD-600: arena_plan.assign's offsets without its quadratic scan
    offsets, arena_words, _ = arena_assign.assign(roots, sizes, classes, plan_commit)
    errors = arena_plan.check(roots, sizes, classes, offsets, arena_words)
    if errors:
        raise LoweringError("invalid arena: " + "; ".join(errors))
    aliases = {name: {**layout.to_dict(), "dtype": dtypes[name]} for name, layout in layouts.items()}
    elided_layouts = {}
    if compose or dynamic:
        # Internal/register values remain in the compiler's layouts and body
        # contracts, but Runtime binds every public manifest layout to storage.
        # Remove only explicitly elided values, never unexplained allocations.
        unmaterialized = dead | set(virtual_text)
        required = set(inputs) | set(outputs) | set(writes) | set(writes.values()) | set(placement)
        lost = sorted(name for name in required if layouts[name].root in unmaterialized)
        if lost:
            raise LoweringError(f"required runtime values have elided storage: {lost}")
        elided_layouts = {name: desc for name, desc in aliases.items() if desc["root"] in unmaterialized}
        aliases = {name: desc for name, desc in aliases.items() if name not in elided_layouts}
        bound_roots = set(inputs) | set(offsets) | set(placement)
        unbound = {name: desc["root"] for name, desc in aliases.items() if desc["root"] not in bound_roots}
        if unbound:
            raise LoweringError(f"runtime layouts have no materialized storage: {unbound}")
    declarations = []
    for name, layout in layouts.items():
        if layout.root in dead or layout.root in virtual_text:
            continue
        pointer = (f"args[{inputs.index(layout.root)}]" if layout.root in inputs
                   else f"args[{placement[layout.root][1][8:]}]" if layout.root in placement
                   and placement[layout.root][1].startswith("inplace:")
                   else f"out[{placement[layout.root][0]}]" if layout.root in placement
                   else f"arena + {offsets[layout.root]}LL")
        if dtypes[name] != "float32":
            pointer = f"reinterpret_cast<{_CTYPES[dtypes[name]]}*>({pointer})"
        declarations.append(f"{_CTYPES[dtypes[name]]}* {_ident(name)} = {pointer}" + (f" + {layout.offset}LL" if layout.offset else "") + ";")
    if dynamic:
        declarations.append(f"const int latch_active_batch = (int){_ident(dynamic.input)}[0];")
        declarations.append(f"if (latch_active_batch < 1 || latch_active_batch > {dynamic.maximum}) return;")
    phase_source = []

    fill_code = {}
    if "zmerge" in features and barrier_schedule == "dependency":
        # P6-C zmerge: the zero-fill phases of one interval run as one vector loop (first phase; the others empty).
        from .group_rules import merged_fills
        for members in intervals:
            fills = [i for i in members if phases[i]["kind"] == "elementwise" and phases[i].get("body")
                     and all(line.startswith("const float r_") and " = 0.0f; v_" in line for line in phases[i]["body"])]
            if len(fills) < 2:
                continue
            code_ = merged_fills([(phases[i], _numel(phases[i]["shape"])) for i in fills])
            if code_ is None:
                continue
            fill_code[fills[0]] = code_
            for i in fills[1:]:
                fill_code[i] = ""

    def phase_code(i, vbase=0, tail_vbase=None):
        if i in fill_code:
            return fill_code[i]
        phase = phases[i]
        # B's physical-grid helper takes an explicit integer rotation before
        # its shared pointer; it must never run under a virtual CTA subset.
        code = phase["code"].replace("/*B_VBASE*/", f"(vrot + {vbase})" if staged else str(vbase))
        # LATCH-TRAIN-MATMUL (matrix_bodies 'tailrot'): native tail products take the rotated offset only where the
        # flat interval emitter supplies one (tail_vbase); every other path keeps rotation 0 (the unchanged call).
        code = code.replace("/*T_VBASE*/", str(tail_vbase) if tail_vbase is not None and not staged else "0")
        code = code.replace("/*C_VBASE*/0", f"(vrot + {vbase})" if staged else str(vbase))
        if staged:
            # Staged regions take the caller's CTA rotation for co-scheduled GEMMs.
            code = code.replace("/*VBASE*/", f", vrot + {vbase}")
        else:
            code = code.replace("/*VBASE*/", f", {vbase}" if vbase else "")
        if phase["kind"] == "elementwise" and EW_RECIPROCAL and conv_program and compose and phase.get("body"):
            # LATCH-STEP10-SPEED: the p5c epilogues' exact rewrite (1.0f / (x)) -> __frcp_rn(x) (both correctly
            # rounded IEEE results) in plain elementwise loops too; idempotent.
            from .p5c_epilogue import reciprocal
            phase["body"] = [reciprocal(line) for line in phase["body"]]
        if phase["kind"] == "elementwise" or (INLINE_VECTOR and phase["kind"] == "elementwise_inline"
                                              and phase.get("body") and not dynamic):
            if dynamic and any(name in dynamic.axes for name in phase["writes"]):
                active_name = next(name for name in phase["writes"] if name in dynamic.axes)
                return (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
                        f"i < {dynamic.numel(active_name, phase['shape'])}; i += (long long)gridDim.x * blockDim.x) {{\n"
                        + "\n".join(phase["body"]) + "\n}")
            unrolled = None
            from . import flat_elementwise as _flat
            numel_full = _numel(phase["shape"])
            numel_main, numel_tail = _flat.split(numel_full) if "vecbf" in features else (numel_full, 0)
            if "vec4" in features:
                from .group_rules import vector_elementwise
                texts = {}
                for ref in phase["reads"] + phase["writes"]:
                    lay = layouts.get(ref)
                    ok_dtype = dtypes.get(ref) == "float32" or (P9_VEC_BF16 and "vecbf" not in features and ref in phase["writes"]
                                                                 and ref not in phase["reads"]
                                                                 and dtypes.get(ref) == "bfloat16")
                    # LATCH-TINYMOE-EGRAD: an FP32 contiguous view at a whole-float4 offset of an arena root (roots are
                    # 64-word aligned, vendor/plan.py) is 16-byte aligned too -- e.g. one expert's slice of a stacked
                    # weight-gradient product -- and is read through its own offset pointer. Data movement only.
                    arena_view = (VEC4_ARENA_VIEWS and lay is not None and bool(lay.offset) and lay.offset % 4 == 0
                                  and dtypes.get(ref) == "float32" and lay.root in offsets
                                  and lay.root not in inputs and lay.root not in placement)
                    if lay is None or not ok_dtype or (lay.offset and not arena_view) or not lay.contiguous \
                            or lay.numel != _numel(phase["shape"]):
                        continue
                    try:
                        texts[f"{_ident(ref)}[{_index(lay, tuple(phase['shape']), 'i')}]"] = (
                            _ident(ref) if arena_view else _ident(lay.root) if lay.root in layouts else _ident(ref))
                    except LoweringError:
                        continue
                    texts[f"{_ident(ref)}[i]"] = _ident(ref)
                # P6-C vec4u: unroll only phases with more than two float4 per thread of the 128 x 256 grid.
                n4_ = numel_main // 4
                bulk_ = P9_BULK_MIN_FLOAT4 is not None and n4_ >= P9_BULK_MIN_FLOAT4
                from .serial_add import emit as emit_serial_add
                unrolled = emit_serial_add(phase, numel_main, texts)
                if unrolled is None:
                    # LATCH-B1-FAST 'vec8' / 'vec4m' (ops.accumulation_rules): wider rounds for queued small-batch bodies.
                    unroll_ = (8 if "vec8" in features and n4_ > 2 * 32768 else
                               4 if "vec4m" in features and 16384 <= n4_ <= 2 * 32768 else
                               P9_VEC_UNROLL_FACTOR if ("vec4u" in features and n4_ > 2 * 32768) or
                               (P9_VEC_UNROLL is not None and n4_ >= P9_VEC_UNROLL) else 1)
                    unrolled = vector_elementwise(phase, numel_main, texts, unroll=unroll_, bulk=bulk_)
                if bulk_ and unrolled and "p9bulk::" in unrolled:
                    if '#include "p9_bulk.cuh"\n' not in helper_parts:
                        helper_parts.append('#include "p9_bulk.cuh"\n')
                    if "p9_bulk.cuh" not in extra_headers:
                        extra_headers.append("p9_bulk.cuh")
            if unrolled is None and "vecbf" in features:
                # P8-C: identity-indexed FP32/BF16 phases as 16-byte vectors (bit-identical data movement).
                from .group_rules import vector_mixed
                texts = {}
                for ref in phase["reads"] + phase["writes"]:
                    lay = layouts.get(ref)
                    if lay is None or dtypes.get(ref) not in ("float32", "bfloat16") or lay.offset or not lay.contiguous \
                            or lay.numel != _numel(phase["shape"]):
                        continue
                    pointer = _ident(lay.root) if lay.root in layouts else _ident(ref)
                    entry = (pointer, dtypes[ref])
                    try:
                        texts[f"{_ident(ref)}[{_index(lay, tuple(phase['shape']), 'i')}]"] = entry
                    except LoweringError:
                        continue
                    texts[f"{_ident(ref)}[i]"] = entry
                gathered = {}
                for ref in phase["reads"]:
                    lay = layouts.get(ref)
                    entry_ = padded_operands.get(ref)
                    if (lay is None or entry_ is None or lay.root != ref or lay.offset
                            or tuple(lay.shape) != tuple(phase["shape"]) or lay.numel >= _U32_LIMIT):
                        continue
                    try:
                        gathered[f"{_ident(ref)}[{_index(lay, tuple(phase['shape']), 'i')}]"] = (
                            _ident(ref), lay.shape[-1], entry_["pitch"])
                    except LoweringError:
                        continue
                unrolled = vector_mixed(phase, numel_main, texts,
                                        unroll=(VEC_MIXED_UNROLL if (VEC_MIXED_UNROLL and conv_program) else
                                                4 if numel_main // 8 > 2 * 32768 else 1), gathered=gathered,
                                        scattered={n: (layouts[n].shape[-1], e["pitch"]) for n, e in padded_operands.items()
                                                   if layouts[n].numel < _U32_LIMIT})
            if unrolled is None and "vfill" in features:
                from .group_rules import vector_fill
                unrolled = vector_fill(phase, _numel(phase["shape"]))
            if unrolled is None and features & {"ilp", "ilp8"}:
                from .group_rules import ilp_elementwise
                unrolled = ilp_elementwise(phase, _numel(phase["shape"]), _ident,
                                           unroll=8 if "ilp8" in features else 4)
            if unrolled and numel_tail:
                unrolled += "\n" + _flat.tail_loop(numel_main, numel_full, phase["body"])
            if unrolled and "padacc::" in unrolled:
                if '#include "padded_access.cuh"\n' not in helper_parts:
                    helper_parts.append('#include "padded_access.cuh"\n')
                if "padded_access.cuh" not in extra_headers:
                    extra_headers.append("padded_access.cuh")
            code = unrolled or f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {_numel(phase['shape'])}LL; i += (long long)gridDim.x * blockDim.x) {{\n" + "\n".join(phase["body"]) + "\n}"
        return code

    stage_info = []
    rc_substages = {}       # LATCH-SELECTIVE-10 rowchain: merged interval -> ordered row-local sub-stages
    cs_substages = {}       # LATCH-TRAIN-SYNC cast_sums: interval -> [producers and others, fused sums]
    ctrace_slots = []
    ectaq_count = [0]
    fill_claims_ = [0]       # LATCH-TRAIN-MATMUL matrix_bodies 'fill' claim-mode products (counted in ectaq_count)
    if staged:
        if not region_body or writes or barrier_schedule != "dependency":
            raise LoweringError("staged regions require a pure dependency-scheduled region body")
        input_index = {name: k for k, name in enumerate(inputs)}
        # staged="phase": one stage per phase (stages of one interval are mutually
        # independent; callers keep interval order). staged=True: one per interval.
        units = ([(k, [i]) for k, members in enumerate(intervals) for i in members] if staged == "phase"
                 else list(enumerate(intervals)))

        def jobs(i):
            # Independent CTA jobs of one non-GEMM phase (launch-policy units).
            phase = phases[i]
            if "work_items" in phase:
                return int(phase["work_items"])
            elements = max(_numel(layouts[name].shape) for name in phase["writes"])
            if phase["kind"] == "reduction":
                return elements
            if phase["kind"] in {"softmax", "log_softmax"}:
                return elements // int(phase["reduction_elements"])
            if phase["kind"] == "gemm":
                return int(phase.get("tiles", 1))
            return (elements + 255) // 256

        def pinned(i):
            return phases[i]["kind"] == "gemm" and any(
                token in phases[i]["code"] for token in ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/"))

        for case_index, (k, members) in enumerate(units):
            parts, vbase = [], 0
            others = [i for i in members if not pinned(i)]
            footprint = max([_footprint(phases[i], layouts) for i in others], default=0)
            if "place" in features:
                pass  # P-OPS5-B: the caller's level rotation already starts after every member's other work
            elif defer_accumulate and others and any(pinned(i) for i in members):
                # Matrix tasks start after the CTAs the unit's other work occupies.
                used = max(_footprint(phases[i], layouts) for i in others)
                if used + sum(int(phases[i].get("tiles", 0)) for i in members if pinned(i)) <= 128:
                    vbase = used
            weights = [min(jobs(i), 1 << 20) for i in others]
            total, prefix = sum(weights), 0
            # P9-C pelem: a stage made only of self-contained elementwise / gather phases (inline loops whose only CTA
            # references are blockIdx.x / gridDim.x tokens in their own text, no helper functions) — and, with
            # 'pred', inline ticketed full reductions — runs on the CTA span (cta0, ctas) its caller passes; the loop
            # composer gives the safe stages of one level disjoint spans (loop_pipeline). Unlike the V4 'part' rule
            # (raced through helpers reading blockIdx), no other phase kind is rewritten. Values are unchanged for
            # elementwise/gather phases (any grid is correct); a reduction's CTA count changes its FP32 association.
            pred = not compose and "pred" in features and os.environ.get("P9C_PRED", "1") != "0"
            safe_kind = lambda i: (phases[i]["kind"] in _PELEM_KINDS or (pred and phases[i]["kind"] == "reduction"
                                                                        and phases[i].get("engine") in _PRED_ENGINES))
            stage_safe = ("pelem" in features and not partition_phases and os.environ.get("P9C_PELEM", "1") != "0"
                          and members and all(safe_kind(i) and not pinned(i) for i in members))
            reads_inputs, writes_outputs, accumulates, reads_outputs = set(), set(), set(), set()
            part_members, part_rot = [], {}
            if staged and not partition_phases:
                # P9-B: members of one stage are mutually independent; groupable p9ws products first (adjacent, so
                # the stage assembler can merge them), every other member after them in its original order.
                from . import p5c_matrix as _p5m0
                if _p5m0.P9WS and _p5m0.P9WS_ENGINE == 'p9ws' and _p5m0.P9WS_GROUP and P9_GROUP_ORDER:
                    first = [i for i in members if pinned(i) and _p5m0.ws_groupable(
                        phases[i].get("matrix_spec") or {}, phases[i].get("matrix_plan") or {})]
                    members = first + [i for i in members if i not in first]
                # LATCH-STEP-06: slack-placed side products run after the stage's other members (after the attention
                # items, whose phase may contain an in-phase grid synchronization).
                members = ([i for i in members if not phases[i].get("step06_slack")]
                           + [i for i in members if phases[i].get("step06_slack")])
            for i in members:
                part_rot[i] = phases[i]["lpt_offset"] if "lpt_offset" in phases[i] else vbase
                part_members.append(i)
                code = phase_code(i, phases[i]["lpt_offset"] if "lpt_offset" in phases[i] else vbase)
                if stage_safe:
                    if "blockIdx" in code.replace("blockIdx.x", "") or "gridDim" in code.replace("gridDim.x", ""):
                        raise LoweringError(f"pelem: phase {i} uses CTA indices outside blockIdx.x / gridDim.x")
                    code = code.replace("blockIdx.x", "lb_").replace("gridDim.x", "lg_")
                    code = (f"if ((int)blockIdx.x >= cta0 && (int)blockIdx.x < cta0 + ctas) {{\n"
                            f"  const unsigned lb_ = blockIdx.x - cta0, lg_ = ctas;\n{code}\n}}")
                    if compose:
                        phases[i]["execution_grid_scope"] = "explicit_cta_span_self_contained_elementwise"
                elif (partition_phases and not pinned(i) and not phases[i].get("requires_full_grid")
                      and (not compose or phases[i]["kind"] in _PELEM_KINDS)):
                    # Disjoint CTA sub-range of the caller's [cta0, cta0+ctas) for
                    # each non-GEMM phase, proportional to its jobs; the phase runs
                    # on a virtual grid (any size >= 1 is correct).
                    w = weights[others.index(i)]
                    code = code.replace("blockIdx.x", "lb_").replace("gridDim.x", "lg_")
                    code = (f"{{ int lo_ = cta0, n_ = ctas;\n"
                            f"  if (ctas >= {len(others)}) {{ lo_ = cta0 + (int)((long long)ctas * {prefix} / {total}); "
                            f"n_ = max(1, cta0 + (int)((long long)ctas * {prefix + w} / {total}) - lo_); }}\n"
                            f"  if ((int)blockIdx.x >= lo_ && (int)blockIdx.x < lo_ + n_) {{\n"
                            f"  const unsigned lb_ = blockIdx.x - lo_, lg_ = n_;\n{code}\n}} }}")
                    if compose:
                        phases[i]["execution_grid_scope"] = "explicit_cta_span_self_contained_elementwise"
                    prefix += w
                parts.append(f"// Phase {i}: {', '.join(phases[i]['names'])}\n{{\n{code}\n}}"
                             + (f"\nP6C_EV(__REGION_ID__, {i});" if _PHASE_EVENTS.get() else ""))
                if pinned(i):
                    vbase += int(phases[i].get("tiles", 0))
                for ref in phases[i]["reads"]:
                    if ref in layouts and layouts[ref].root in input_index:
                        reads_inputs.add(input_index[layouts[ref].root])
                    elif ref in layouts and layouts[ref].root in placement:
                        index, mode = placement[layouts[ref].root]
                        reads_outputs.add(index)
                for name in phases[i]["writes"]:
                    if name in placement:
                        index, mode = placement[name]
                        if mode.startswith("inplace:"):
                            writes_outputs.add(index)
                            reads_inputs.add(int(mode[8:]))
                        else:
                            writes_outputs.add(index)
                            if mode == "accumulate":
                                accumulates.add(index)
            def _ws_init(others_):
                # P9-B: modeled per-CTA time of the stage's non-matrix phases (they start at CTA 0) for the balanced
                # group schedule.
                from .group_rules import np_cost
                init_ = [0.0] * 128
                from .loop_slack import attention_load as _att_load
                for j in others_:
                    att_ = _att_load(phases[j]) if STEP06_SLACK else None
                    if att_ is not None:
                        init_ = [a_ + b_ for a_, b_ in zip(init_, att_)]
                        continue
                    fp_ = min(128, max(1, _footprint(phases[j], layouts)))
                    cost_ = np_cost(phases[j], fp_)
                    for c_ in range(fp_):
                        init_[c_] += cost_
                return init_
            if staged and not partition_phases:
                # P9-B: consecutive p9ws products of this stage become one continuous warp-specialized pipeline.
                from . import p5c_matrix as _p5m
                if _p5m.P9WS and _p5m.P9WS_ENGINE == 'p9ws' and _p5m.P9WS_GROUP:
                    ok = [pinned(i) and _p5m.ws_groupable(phases[i].get("matrix_spec") or {},
                                                          phases[i].get("matrix_plan") or {}) for i in part_members]
                    merged, k0 = [], 0
                    while k0 < len(parts):
                        k1 = k0
                        while k1 < len(parts) and ok[k1]:
                            k1 += 1
                        if k1 - k0 >= 2:
                            run = part_members[k0:k1]
                            code = _p5m.ws_group_code([phases[i]["matrix_spec"] for i in run],
                                                      [phases[i]["matrix_plan"] for i in run],
                                                      [f"(vrot + {part_rot[i]})" for i in run], _ident,
                                                      init=_ws_init(others))
                            if isinstance(code, tuple):
                                code, table = code
                                if table not in helper_parts:
                                    helper_parts.append(table)     # P9-B balanced group schedule
                            names = [n for i in run for n in phases[i]["names"]]
                            merged.append(f"// Phases {', '.join(str(i) for i in run)} (p9ws group): {', '.join(names)}\n{{\n{code}\n}}")
                            k0 = k1
                        elif ok[k0] and _p5m.P9WS_GROUPS_ONLY:
                            # P9-B: a warp-specialized product without a group partner runs the 8-warp p5c body
                            # (in-kernel p9ws singles measured slower than p5c; groups faster).
                            i = part_members[k0]
                            spec_ = phases[i]["matrix_spec"]
                            import re as _re
                            found = _re.search(r"\(g, smem, ([^,]+), ([^,]+), ([^,]+), ([^,)]+)", phases[i]["code"])
                            saved_ws = _p5m.P9WS
                            try:
                                _p5m.P9WS = False
                                plan_ = _p5m.plan(spec_, int(phases[i]["matrix_plan"].get("grid", 128)))
                            finally:
                                _p5m.P9WS = saved_ws
                            if found and plan_ and not plan_.get("ws") and plan_.get("shared_bytes", 0) <= _p5m.carveout():
                                original = phases[i]["code"]
                                phases[i]["code"] = _p5m.code(spec_, plan_, _ident, *found.groups())
                                try:
                                    code = phase_code(i, part_rot[i])
                                finally:
                                    phases[i]["code"] = original
                                merged.append(f"// Phase {i}: {', '.join(phases[i]['names'])} (p5c single)\n{{\n{code}\n}}")
                            else:
                                merged.append(parts[k0])
                            k0 += 1
                        else:
                            merged.append(parts[k0])
                            k0 += 1
                    parts = merged
            stage_prelude = ''
            if staged and not partition_phases and P9_STAGE_LPT and len(parts) == len(part_members):
                # P9-B: the stage's p5c products share one compile-time longest-first tile table (the products of a
                # stage are mutually independent; each CTA runs its listed tiles of each product in order). Other
                # products of the stage keep their rotation and count as initial loads, with the non-matrix phases.
                from . import p5c_matrix as _p5t
                from .group_rules import _tile_cost
                table_members, init_ = [], _ws_init(others)
                for k_, i in enumerate(part_members):
                    spec_, plan_ = phases[i].get("matrix_spec") or {}, phases[i].get("matrix_plan") or {}
                    if (phases[i].get("kind") == "gemm" and pinned(i) and spec_.get("engine") == "p5c"
                            and not spec_.get("splitk") and not plan_.get("ws") and not plan_.get("pingpong")
                            and plan_.get("mc", 1) == 1 and "bm" in plan_ and "(g, smem, 0, 1, 1," in parts[k_]
                            and parts[k_].count("p5c::Args g{") == 1):
                        table_members.append((k_, i))
                    elif phases[i].get("kind") == "gemm" and pinned(i) and plan_.get("bm"):
                        cost_ = _tile_cost(int(plan_["bm"]), int(plan_.get("bn", 64)), int(spec_.get("k", 256)))
                        for t_ in range(int(phases[i].get("tiles", 0))):
                            init_[(part_rot[i] + t_) % 128] += cost_
                _table_min = 1 if any(phases[i].get("step06_slack") for _, i in table_members) else P9_STAGE_TABLE_MIN
                if len(table_members) >= _table_min and P9_STAGE_TILES:
                    specs_ = [phases[i]["matrix_spec"] for _, i in table_members]
                    plans_ = [phases[i]["matrix_plan"] for _, i in table_members]
                    grid_ = int(plans_[0].get("grid", 128))
                    def _alts(spec_, plan_):
                        if (spec_.get("epi") and not P9_STAGE_TILES_EPI) or spec_.get("splitk") or spec_.get("p9_delta"):
                            return [plan_]
                        saved_ = (_p5t.P9_TILE64, _p5t.P5C_BN192)
                        _p5t.P9_TILE64 = True
                        _p5t.P5C_BN192 = P9_STAGE_TILES_192
                        try:
                            _, cands_ = _p5t.p5c_plan(spec_["m"], spec_["n"], spec_["k"], spec_["ak"], spec_["bkc"],
                                                      grid=grid_)
                        finally:
                            _p5t.P9_TILE64, _p5t.P5C_BN192 = saved_
                        out_ = [plan_]
                        for c_ in cands_:
                            shape_ = (c_["bm"], c_["bn"])
                            if shape_ == (128, 96) or (shape_ == (128, 192) and not P9_STAGE_TILES_192) \
                                    or c_["shared_bytes"] > _p5t.carveout():
                                continue
                            if spec_.get("epi") and shape_ == (128, 64):
                                continue
                            if shape_ == (plan_["bm"], plan_["bn"]):
                                continue
                            out_.append(dict(c_, release=plan_.get("release", False),
                                             out_f32=bool(spec_.get("out_f32")), mc=1))
                        return out_
                    def _span(plans__):
                        t_, loads_ = _p5t.stage_balance(specs_, plans__, grid_, init_)
                        return loads_[-1] if t_ else float("inf")
                    alts_ = [_alts(sp_, pl_) for sp_, pl_ in zip(specs_, plans_)]
                    cur_ = list(plans_)
                    best_ = _span(cur_)
                    for _pass in range(3):
                        changed_ = False
                        for m_, options_ in enumerate(alts_):
                            for o_ in options_:
                                if o_ is cur_[m_]:
                                    continue
                                trial_ = cur_[:m_] + [o_] + cur_[m_ + 1:]
                                v_ = _span(trial_)
                                if v_ < best_ * P9_STAGE_TILES_GAIN:
                                    best_, cur_, changed_ = v_, trial_, True
                        if not changed_:
                            break
                    for (k_, i), new_ in zip(table_members, cur_):
                        if new_ is phases[i]["matrix_plan"]:
                            continue
                        found_ = re.search(r"\(g, smem, ([^,]+), ([^,]+), ([^,]+), ([^,)]+)", phases[i]["code"])
                        if not found_:
                            continue
                        phases[i]["code"] = _p5t.code(phases[i]["matrix_spec"], new_, _ident, *found_.groups())
                        phases[i]["matrix_plan"] = new_
                        phases[i]["tiles"] = new_["tiles"]
                        code_ = phase_code(i, part_rot[i])
                        parts[k_] = (f"// Phase {i}: {', '.join(phases[i]['names'])} (P9 stage tiles "
                                     f"{new_['bm']}x{new_['bn']})\n{{\n{code_}\n}}"
                                     + (f"\nP6C_EV(__REGION_ID__, {i});" if _PHASE_EVENTS.get() else ""))
                if len(table_members) >= _table_min:
                    specs_ = [phases[i]["matrix_spec"] for _, i in table_members]
                    plans_ = [phases[i]["matrix_plan"] for _, i in table_members]
                    grid_ = int(plans_[0].get("grid", 128))
                    table, _ = _p5t.stage_balance(specs_, plans_, grid_, init_)
                    if table:
                        import hashlib as _h
                        nmem = len(table_members)
                        entries, lists = _p5t.stage_entries(table, nmem, grid_)
                        key_ = _h.sha256((','.join(map(str, entries)) + '|' + ','.join(map(str, lists))).encode()).hexdigest()[:12]
                        ename, lname = f'p9e_{key_}', f'p9l_{key_}'
                        decl = (f'#ifndef P9T_{key_}\n#define P9T_{key_}\n'
                                f'__device__ const unsigned long long {ename}[{len(entries)}] = {{'
                                + ', '.join(f'{x}ull' for x in entries) + '};\n'
                                f'__device__ const unsigned short {lname}[{max(1, len(lists))}] = {{'
                                + (', '.join(map(str, lists)) or '0') + '};\n#endif')
                        if decl not in helper_parts:
                            helper_parts.append(decl)
                        # Entries of every member, loaded together at stage entry (planned grid only).
                        stage_prelude = ''.join(
                            f'const unsigned long long p9r_{idx_} = (int)gridDim.x == {grid_} ? '
                            f'__ldg(&{ename}[blockIdx.x * {nmem} + {idx_}]) : 0ull;\n' for idx_ in range(nmem))
                        for idx_, (k_, i) in enumerate(table_members):
                            parts[k_] = re.sub(r"(p5c::Args g\{[^;]*\};)",
                                               lambda m_: m_.group(1) + f" g.srow = p9r_{idx_}; g.stl = {lname};",
                                               parts[k_], count=1)
                        if P9_CHAIN:
                            # P9-B: chains of same-shape, same-configuration plain products (one p5c pipeline).
                            call_re = re.compile(r"(p5c::Args g\{[^;]*\}; g\.srow = [^;]+; g\.stl = [^;]+;)\s*"
                                                 r"p5c::phase<([^>]+)>\(g, smem, 0, 1, 1, [^;]*\);")
                            groups_ = {}
                            for k_, i in table_members:
                                sp_ = phases[i]["matrix_spec"]
                                found_ = call_re.findall(parts[k_])
                                if sp_.get("epi") or sp_.get("p9_delta") or len(found_) != 1 or parts[k_].count("p5c::") != 2:
                                    continue
                                key_ = (found_[0][1], sp_["m"], sp_["n"], sp_["k"])
                                groups_.setdefault(key_, []).append((k_, found_[0][0]))
                            for (tpl_, _, _, _), mem_ in groups_.items():
                                for c0_ in range(0, len(mem_), 6):
                                    run_ = mem_[c0_:c0_ + 6]
                                    if len(run_) < 2:
                                        continue
                                    body_ = "".join(f"  {{ {stmt_} p9cg_[{q_}] = g; }}\n" for q_, (_, stmt_) in enumerate(run_))
                                    parts[run_[0][0]] = (f"// P9 chain of {len(run_)} products\n{{\n  p5c::Args p9cg_[{len(run_)}];\n"
                                                         + body_ + f"  p5c::phase_chain<{tpl_},{len(run_)}>(p9cg_, smem);\n}}")
                                    for k_, _ in run_[1:]:
                                        parts[k_] = "{}"

            if P9_MAP_PREFETCH and staged and len(parts) > 1:
                # P9-B: warm the TMA descriptor cache for the stage's later products while the first one runs.
                maps_ = []
                for part_ in parts[1:]:
                    pos_ = part_.find("p5c::Args g{")
                    while pos_ >= 0:
                        j_, depth_, got_, start_ = pos_ + len("p5c::Args g{"), 0, [], None
                        start_ = j_
                        while j_ < len(part_) and len(got_) < 2:      # first two top-level arguments
                            ch_ = part_[j_]
                            if ch_ in "([{":
                                depth_ += 1
                            elif ch_ in ")]}":
                                depth_ -= 1
                            elif ch_ == "," and depth_ == 0:
                                got_.append(part_[start_:j_].strip())
                                start_ = j_ + 1
                            j_ += 1
                        maps_.extend(x_ for x_ in got_ if x_.startswith(("P5C_TMAPR(", "P5C_TMAP(", "p5c_pick_map(")))
                        pos_ = part_.find("p5c::Args g{", j_)
                if maps_:
                    pf_ = "".join(f'  asm volatile("prefetch.tensormap [%0];" :: "l"({m_}) : "memory");\n' for m_ in maps_)
                    stage_prelude = stage_prelude + "if (threadIdx.x == 0) {\n" + pf_ + "}\n"
            if stage_prelude and parts:
                parts[0] = stage_prelude + parts[0]
            if P9_GEMM_REPEAT and parts and any(phases[i].get("kind") == "gemm" for i in members):
                # Diagnostic only (timing, values change): each CTA runs its share of the stage twice, either looping
                # over the same code (warm instructions and data) or through a second code copy (warm data only).
                body_ = "\n__syncthreads();\n".join(parts)
                if P9_GEMM_REPEAT == "loop":
                    parts = ["#pragma unroll 1\nfor (int p9rep = 0; p9rep < *(volatile int*)&p9_rep_n; ++p9rep) {\n"
                             + body_ + "\n__syncthreads();\n}"]
                    rep_decl_ = "#ifndef P9_REP_N_DEF\n#define P9_REP_N_DEF\n__device__ int p9_rep_n = 2;\n#endif"
                    if rep_decl_ not in helper_parts:
                        helper_parts.append(rep_decl_)
                else:
                    parts = ["{\n" + body_ + "\n}\n__syncthreads();\n{\n" + body_ + "\n}"]
            if "nosync" in features:
                # P6-C: phases of one stage are independent; a CTA barrier is needed only to order shared-memory
                # reuse, so it is dropped between two neighbouring phases that never touch shared memory.
                joined = parts[0] if parts else ""
                for previous, part in zip(parts, parts[1:]):
                    joined += ("\n" if "smem" not in previous and "smem" not in part else "\n__syncthreads();\n") + part
                phase_source.append(joined)
            else:
                phase_source.append("\n__syncthreads();\n".join(parts))
            gemm_load = None
            if features & {"lptl", "lptx"}:
                # P6-C lptl: modeled per-CTA product load of this stage (relative to its rotation) for level placement.
                from .group_rules import _tile_cost
                gemm_load, vb = [0.0] * 128, 0
                for i in members:
                    if not pinned(i):
                        continue
                    tiles = int(phases[i].get("tiles", 0))
                    call, plan = phases[i].get("popsc_call"), phases[i].get("matrix_plan") or {}
                    bm, bn = ((int(call["targs"][0]), int(call["targs"][1])) if call
                              else (int(plan.get("bm", 64)), int(plan.get("bn", 64))))
                    cost = _tile_cost(bm, bn, int((phases[i].get("matrix_spec") or {}).get("k", 256)))
                    off = phases[i].get("lpt_offset", vb)
                    for t in range(tiles):
                        gemm_load[(off + t) % 128] += cost
                    vb += tiles
                gemm_load = [round(x, 3) for x in gemm_load]
            extra_info = {}
            if "lptx" in features:
                # P6-C lptx: modeled non-matrix work (footprint from CTA 0, us) and absolute product placement.
                from .group_rules import np_cost
                extra_info["np_load"] = [[min(128, max(1, _footprint(phases[i], layouts))),
                                          np_cost(phases[i], _footprint(phases[i], layouts))] for i in others]
                extra_info["lpt_absolute"] = any(phases[i].get("lpt_abs") for i in members)
            stage_info.append({"phases": members, "interval": k, "reads_inputs": sorted(reads_inputs),
                               **({"global_fence": True} if compose and any(
                                   requires_grid_fence(phases[i]) for i in members) else {}),
                               **({"gemm_load": gemm_load} if gemm_load is not None else {}), **extra_info,
                               "writes_outputs": sorted(writes_outputs), "accumulates": sorted(accumulates),
                               "reads_outputs": sorted(reads_outputs), "gemm_tiles": vbase,
                               **({"footprint": footprint} if "place" in features else {}),
                               **({"pelem": True, "pelem_work": max(1, sum(
                                   int(phases[i].get("reduction_elements") or 0) * (_numel(phases[i]["shape"]) if phases[i].get("shape") else 1)
                                   if phases[i]["kind"] == "reduction" else
                                   max(_numel(layouts[w].shape) for w in phases[i]["writes"] if w in layouts)
                                   for i in members) // 256)} if stage_safe else {}),
                               "jobs": total})
        if "l2pf" in features:
            # P6-C l2pf: each stage starts bulk L2 prefetches of the operands its successor stage reads that it does
            # not read itself (saved tape slices, weights, earlier results). A prefetch is a cache hint only: it never
            # changes a value, whatever the operand's producer or the iteration the successor runs for.
            declared = {re.search(r"\b(v_\w+) =", line).group(1) for line in declarations}
            for u in range(len(units) - 1):
                mine = {layouts[r].root for i in units[u][1] for r in phases[i]["reads"] if r in layouts}
                chunks, total_ = [], 0
                for i in units[u + 1][1]:
                    for r in phases[i]["reads"]:
                        if r not in layouts:
                            continue
                        root = layouts[r].root
                        nbytes = layouts[root].numel * _BYTES.get(dtypes.get(root), 4) if root in layouts else 0
                        if (root in mine or _ident(root) not in declared or nbytes < 32768 or nbytes % 16
                                or any(c[0] == root for c in chunks) or total_ + nbytes > 16 * 1024 ** 2):
                            continue
                        chunks.append((root, nbytes))
                        total_ += nbytes
                if not chunks:
                    continue
                code = ["{ // P6-C l2pf: bulk L2 prefetch of the next stage's operands",
                        "  const unsigned pf_t = blockIdx.x * blockDim.x + threadIdx.x, pf_n = gridDim.x * blockDim.x;"]
                for root, nbytes in chunks:
                    count = -(-nbytes // 4096)
                    last = nbytes - (count - 1) * 4096
                    code.append(f"  for (unsigned c = pf_t; c < {count}u; c += pf_n) {{\n"
                                f"    const char* p = reinterpret_cast<const char*>({_ident(root)}) + (size_t)c * 4096u;\n"
                                f"    const unsigned bytes = c + 1u == {count}u ? {last}u : 4096u;\n"
                                f"    asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(p), \"r\"(bytes) : \"memory\");\n  }}")
                code.append("}")
                phase_source[u] = "\n".join(code) + "\n" + phase_source[u]
                stage_info[u]["l2_prefetch"] = [[root, nbytes] for root, nbytes in chunks]
    elif barrier_schedule == "phase":
        for i, phase in enumerate(phases):
            phase_source.append(f"// Phase {i}: {', '.join(phase['names'])}\n{{\n{phase_code(i)}\n}}\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i+1}] = stamp();")
    else:
        def ctrace_stamp(label):
            # Step-14 speed diagnostic ('ctrace'): per-CTA globaltimer stamps after the program's tick words,
            # 256 words per slot (one per CTA). Written only when the runtime passes a ticks buffer (profiling).
            ctrace_slots.append(label)
            return (f"\nif (ticks && threadIdx.x == 0 && blockIdx.x < 256u) "
                    f"ticks[LATCH_CTSTAMP_BASE + {len(ctrace_slots) - 1} * 256 + blockIdx.x] = stamp();")
        chain_runs, chain_skip = {}, set()
        if CHAIN_SMALL and conv_program and compose and not staged and not region_body:
            # LATCH-STEP10-SPEED: consecutive intervals made only of tiny phases (e.g. a classifier head and its
            # loss/backward between two large stages) run on CTA 0 with CTA barriers between them, then one grid
            # barrier, instead of one grid barrier per interval. Each phase keeps its code and arithmetic; only the
            # grid it distributes its work over becomes one CTA.
            run = []
            for k, members in enumerate(intervals + [[]]):
                if members and all(_chain_small(phases[i], phase_code(i, 0), layouts) for i in members):
                    run.append(k)
                    continue
                if len(run) >= 2:
                    chain_runs[run[0]] = list(run)
                    chain_skip.update(run[1:])
                run = []
        rc_map = {}
        if "rowchain" in features and compose and not staged and not region_body and not chain_runs:
            # LATCH-SELECTIVE-10 (ops.accumulation_rules rowchain): runs of row-local small intervals become one
            # dependency interval (one grid barrier); each CTA computes its own row block in every phase, phase
            # groups separated by CTA barriers (ops/row_chain.py). The manifest records the merged intervals.
            from . import row_chain as _row_chain
            rc_runs, _, rc_records_, rc_info = _row_chain.plan_runs(intervals, phases, layouts, phase_code)
            if rc_runs and _row_chain.ARENA_SAFE:
                # TINYMOE-FINER-TRANSFER: arena storage was planned per interval before merging; cut runs where a
                # later interval would reuse storage an earlier interval of the run still accesses (row_chain.py).
                rc_runs, rc_records_, rc_cuts_ = _row_chain.arena_safe_runs(rc_runs, intervals, phases, layouts,
                                                                           offsets, sizes)
                for record_ in rc_records_:
                    record_["rows"] = rc_info[record_["first_interval"]][0][1]
                group_records["row_chain_arena_cuts"] = rc_cuts_
            if rc_records_:
                group_records["row_chains"] = rc_records_
                merged_, k_ = [], 0
                while k_ < len(intervals):
                    if k_ in rc_runs:
                        rc_map[len(merged_)] = [rc_info[kk] for kk in rc_runs[k_]]
                        rc_substages[len(merged_)] = [list(intervals[kk]) for kk in rc_runs[k_]]
                        merged_.append([i for kk in rc_runs[k_] for i in intervals[kk]])
                        k_ = rc_runs[k_][-1] + 1
                    else:
                        merged_.append(intervals[k_])
                        k_ += 1
                intervals = merged_
        for k_interval, members in enumerate(intervals):
            if k_interval in chain_skip:
                continue
            if k_interval in rc_map:
                segments, chained = [], []
                R_ = rc_map[k_interval][0][0][1]
                for group_ in rc_map[k_interval]:
                    parts_ = []
                    for i, _, per_row_, form_, code_ in group_:
                        body_ = _row_chain.restrict(phases[i], form_, code_, per_row_)
                        parts_.append(f"// Phase {i}: {', '.join(phases[i]['names'])}\n{{\n{body_}\n}}")
                        if "ctrace" in features and _ctrace_phase_stamps():
                            parts_[-1] += ctrace_stamp(f"phase:{i}")
                        chained.append(i)
                    segments.append("\n__syncthreads();\n".join(parts_))
                stamps = " ".join(f"ticks[{i+1}] = t;" for i in chained)
                head_ = ctrace_stamp(f"start:{len(phase_source)}") + "\n" if "ctrace" in features else ""
                phase_source.append(head_ + _row_chain.open_block(R_) + "\n__syncthreads();\n".join(segments) + "\n}"
                                    + (ctrace_stamp(f"end:{len(phase_source)}") if "ctrace" in features else "")
                                    + "\ngrid.sync();\n"
                                    f"if (ticks && blockIdx.x == 0 && threadIdx.x == 0) {{ const unsigned long long t = stamp(); {stamps} }}")
                continue
            if k_interval in chain_runs:
                segments, chained = [], []
                for kk in chain_runs[k_interval]:
                    parts_ = []
                    for i in intervals[kk]:
                        body_ = _chain_restrict(phase_code(i, 0))
                        parts_.append(f"// Phase {i}: {', '.join(phases[i]['names'])}\n{{\n{body_}\n}}")
                        chained.append(i)
                    segments.append("\n__syncthreads();\n".join(parts_))
                stamps = " ".join(f"ticks[{i+1}] = t;" for i in chained)
                phase_source.append("if (blockIdx.x == 0) {  // chained small intervals on one CTA\n"
                                    "const unsigned chain_b_ = 0u, chain_g_ = 1u;\n"
                                    + "\n__syncthreads();\n".join(segments) + "\n}\ngrid.sync();\n"
                                    f"if (ticks && blockIdx.x == 0 && threadIdx.x == 0) {{ const unsigned long long t = stamp(); {stamps} }}")
                group_records.setdefault("chained_intervals", []).append(
                    {"intervals": len(chain_runs[k_interval]), "phases": [n for i in chained for n in phases[i]["names"]]})
                continue
            parts, vbase, part_ids = [], 0, []
            if "ecta" in features and compose:
                # Release step 14: the interval's TMA + wgmma products start first (rotation from CTA 0); other pinned
                # products follow on the CTAs they leave. Phases of one interval are independent; values unchanged.
                pinned_ = [i for i in members if phases[i]["kind"] == "gemm" and any(
                    token in phases[i]["code"] for token in ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/"))]
                ordered_ = ([i for i in pinned_ if phases[i].get("tma_respec")]
                            + [i for i in pinned_ if not phases[i].get("tma_respec")])
                slots_ = iter(ordered_)
                members = [next(slots_) if i in pinned_ else i for i in members]
            ecta_tiles = pinned_total_ = 0
            if "ecta" in features and compose:
                from .group_rules import ECTA_GRID, ecta_eligible, ecta_restrict
                ecta_tiles = sum(int(phases[i].get("tiles", 0)) for i in members if phases[i]["kind"] == "gemm" and any(
                    token in phases[i]["code"] for token in ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/")))
                pinned_total_ = ecta_tiles
                if not 0 < ecta_tiles < ECTA_GRID:
                    ecta_tiles = 0
            fp32_place = {}
            queued_ = set()
            if "ectaq" in features and pinned_total_ and phase_source:
                # Step-14 speed ('ectaq'): FP32 small-tile products of an interval with pinned matrix tiles claim
                # their tasks from a per-launch counter after the interval's other work (pinned tiles, full-grid
                # reductions, idle-CTA elementwise phases), so CTAs that finish early absorb them. Never in the
                # first interval (the counters are zeroed at kernel entry and ordered by its barrier).
                queued_ = {i for i in members if phases[i]["kind"] == "gemm"
                           and str(phases[i].get("engine", "")).startswith("popsc_fp32")
                           and re.search(r"smmsplit::(edge_phase|phase)<", phases[i]["code"])}
                members = [i for i in members if i not in queued_] + [i for i in members if i in queued_]
                if "tfirst" in features:
                    # Pinned TMA products start every CTA's interval; the interval's other independent phases
                    # (reductions, elementwise, split-K reduces) follow them, then the queued FP32 work.
                    tma_first_ = [i for i in members if phases[i]["kind"] == "gemm" and phases[i].get("tma_respec")
                                  and i not in queued_]
                    members = tma_first_ + [i for i in members if i not in tma_first_]
                if queued_:
                    group_records.setdefault("ecta_queue", []).append(
                        {"interval": len(phase_source), "pinned_tiles": pinned_total_,
                         "queued": [phases[i]["names"][0] for i in members if i in queued_]})
            claimed_ = []
            if "tmaq" in features and pinned_total_ and pinned_total_ > ECTA_GRID and phase_source:
                # Step-14 speed ('tmaq'): when an interval's pinned TMA tiles exceed the grid, every TMA product of the
                # interval claims its tiles from a per-launch counter (one ahead), longest estimated tile first, so
                # CTAs finishing early take the remaining tiles. Each tile is computed whole by one CTA (unchanged).
                tma_ = [i for i in members if phases[i]["kind"] == "gemm" and phases[i].get("tma_respec")
                        and "pops4b::TArgs g{" in phases[i]["code"]]
                def nk_ok_(i):
                    m_ = re.search(r"tma_phase_inline<(\d+),(\d+),\d+,\w+,\w+,(\d+),", phases[i]["code"])
                    k_ = int((phases[i].get("matrix_spec") or phases[i]["tma_respec"]["spec"]).get("k", 0))
                    return bool(m_) and -(-k_ // 64) >= int(m_.group(3))
                if len(tma_) >= 2 and all(nk_ok_(i) for i in tma_) and sum(int(phases[i].get("tiles", 0)) for i in tma_) == pinned_total_:
                    cost_ = lambda i: float((phases[i].get("matrix_plan") or {}).get("local_us", 0.0)) / max(
                        1, int((phases[i].get("matrix_plan") or {}).get("waves", 1) or 1))
                    order_ = sorted(tma_, key=lambda i: -cost_(i))
                    rest_ = [i for i in members if i not in tma_]
                    members = order_ + rest_
                    claimed_ = order_
                    group_records.setdefault("tma_claim", []).append(
                        {"interval": len(phase_source), "products": [phases[i]["names"][0] for i in order_],
                         "tiles": [int(phases[i].get("tiles", 0)) for i in order_]})
            fill_ = None
            # LATCH-TRAIN-INTEGRATE: ECTA_GRID was bound only inside the "ecta" branch above, so composed programs
            # without that feature raised UnboundLocalError here (11 tests/operations failures in c3); bind it here.
            from .group_rules import ECTA_GRID
            if compose and not claimed_ and not queued_:
                from .matrix_bodies import long_first
                members = long_first(phases, members, grid=ECTA_GRID)
            if compose and phase_source and not claimed_ and not queued_:
                # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'fill'): a wide TMA product after an interval's long rotated
                # tasks claims its tiles (or takes the CTAs those tasks leave idle). Never in the first interval (claim
                # counters are zeroed at kernel entry). Placement only; values unchanged.
                from .matrix_bodies import fill_plan
                fill_ = fill_plan(phases, members, grid=ECTA_GRID)
                if fill_ is not None:
                    group_records.setdefault("matrix_fill", []).append(
                        {"interval": len(phase_source), "wide": phases[fill_["wide"]]["names"][0], "mode": fill_["mode"],
                         "fixed": [phases[i]["names"][0] for i in fill_["fixed"]], "fixed_tiles": fill_["fixed_tiles"]})
            if ecta_tiles and features & {"ectaw", "ectac", "ectai"}:
                # Step-14 speed: work/capacity-aware placement of the interval's FP32 small-tile products.
                from .group_rules import fp32_idle_placement
                fp32_place, record_ = fp32_idle_placement(
                    phases, members, ecta_tiles,
                    policy="whole" if "ectaw" in features else "idle" if "ectai" in features else "capacity")
                if record_:
                    group_records.setdefault("ecta_fp32", []).append(record_)
            if "fsgather" in features and "fsbatch" in features and compose:
                # LATCH-SELECTIVE-10 (ops.accumulation_rules fsgather): the interval's exact multi-CTA full sums are
                # emitted consecutively after its other (independent) phases, so fsbatch shares one body among up
                # to FS_BATCH of them instead of runs broken by interleaved phases. Each phase's code is unchanged.
                sums_ = [i for i in members if phases[i].get("engine") == "multi_cta_full"]
                if sums_:
                    members = [i for i in members if i not in sums_] + sums_
            if any(phases[i].get("flat_claim") for i in members):
                # LATCH-TRAIN-EXPERTS (segmented_products "claim"): the interval's claimed weight-gradient (outer)
                # products start every CTA's interval, then its claimed right products, then its other phases (phases
                # of one dependency interval are independent; only the order inside each CTA changes). Their longest
                # tasks (the largest experts' k chains, claimed first) then start with the interval instead of after
                # the CTA's share of the other products.
                outer_ = [i for i in members if phases[i].get("flat_claim") and phases[i].get("tma_outer")]
                right_ = [i for i in members if phases[i].get("flat_claim") and not phases[i].get("tma_outer")]
                members = outer_ + right_ + [i for i in members if i not in outer_ and i not in right_]
                from .ragged_experts_lower import merge_claimed_outer
                for first_, second_ in merge_claimed_outer(phases, members):
                    group_records.setdefault("ragged_claim_merged", []).append(
                        {"interval": len(phase_source), "products": [phases[first_]["names"][0], phases[second_]["names"][0]]})
            claim_seen_ = False
            fused_here_ = [i for i in members if i in cast_fusions_]
            if fused_here_:
                cs_substages[k_interval] = [[i for i in members if i not in fused_here_], list(fused_here_)]
            fused_producers_ = {cast_fusions_[i]["producer"] for i in fused_here_}
            for i in members:
                # Independent pinned GEMMs in one interval start on rotated CTAs (vbase).
                body_ = phase_code(i, phases[i].get('lpt_offset', vbase), tail_vbase=vbase)
                if i in fused_producers_:
                    # LATCH-TRAIN-SYNC cast_sums: computed and stored by its sums' lanes in the fsbatch body below.
                    body_ = "// fused into the update_schedule fsbatch body of its square sums (ops.cast_sums)"
                if i in pair_streams_ or i in pair_streams_.values():
                    from .group_rules import PAIR_SHARE, pair_restrict
                    if i in pair_streams_:
                        body_ = pair_restrict(body_, PAIR_SHARE)
                    else:
                        body_, count_ = re.subn(r"(tma_phase_inline<[^>]*>)\(g, smem, 0, 1, 1, (\d+)\)",
                                                rf"\1(g, smem, 0, {PAIR_SHARE}, 1000, \2)", body_)
                        if count_ != 1:
                            raise RuntimeError("pstream: unexpected TMA product call form")
                if phases[i].get("flat_claim"):
                    # LATCH-TRAIN-EXPERTS: a claimed segmented product takes one zeroed per-launch counter word (the
                    # ectaq words past the arena, zeroed at kernel entry and ordered by the first grid barrier); the
                    # first claimed product of an interval keeps a static first task per CTA.
                    from .ragged_experts_lower import CLAIM_SLOT, CLAIM_STATIC0
                    if not phase_source:
                        raise LoweringError("claimed segmented products cannot run in the first interval")
                    if body_.count(CLAIM_SLOT) != 1 or body_.count(CLAIM_STATIC0) != 1:
                        raise RuntimeError("claim: unexpected segmented product call form")
                    body_ = body_.replace(CLAIM_SLOT, f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {ectaq_count[0]}LL)")
                    static0_ = not claim_seen_
                    body_ = body_.replace(CLAIM_STATIC0, "true" if static0_ else "false")
                    claim_seen_ = True
                    group_records.setdefault("ragged_claim", []).append(
                        {"interval": len(phase_source), "product": phases[i]["names"][0], "counter": ectaq_count[0],
                         "static_first": static0_})
                    ectaq_count[0] += 1
                if fill_ is not None and i == fill_["wide"]:
                    if fill_["mode"] == "claim":
                        counter_ = f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {ectaq_count[0]}LL)"
                        body_, count_ = re.subn(r"(pops4b::TArgs g\{[^;]*\};)",
                                                lambda m: m.group(1) + f" g.claim = {counter_};", body_, count=1)
                        if count_ != 1:
                            raise RuntimeError("matrix_bodies fill: unexpected TMA product call form")
                        ectaq_count[0] += 1
                        fill_claims_[0] += 1
                    else:
                        from .matrix_bodies import fill_slice
                        body_ = fill_slice(body_, fill_["fixed_tiles"])
                if i in claimed_:
                    counter_ = f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {ectaq_count[0]}LL)"
                    body_, count_ = re.subn(r"(pops4b::TArgs g\{[^;]*\};)", lambda m: m.group(1) + f" g.claim = {counter_};",
                                            body_, count=1)
                    if count_ != 1:
                        raise RuntimeError("tmaq: unexpected TMA product call form")
                    ectaq_count[0] += 1
                if i in queued_:
                    counter_ = f"reinterpret_cast<unsigned*>(arena + /*ECTAQ_BASE*/ + {ectaq_count[0]}LL)"
                    body_, count_ = re.subn(r"smmsplit::(edge_phase|phase)<([^<>]*)>\(g, ([^,]+), (\d+), (\d+), smem, "
                                            r"0, 1, 1, [^,()]+\);",
                                            lambda m: f"smmsplit::{m.group(1)}_q<{m.group(2)}>(g, {m.group(3)}, {m.group(4)}, "
                                                      f"{m.group(5)}, smem, {counter_});", body_)
                    if count_ != 1:
                        raise RuntimeError("ectaq: unexpected FP32 product call form")
                    ectaq_count[0] += 1
                elif ecta_tiles and ecta_eligible({"kind": phases[i]["kind"], "code": body_}):
                    body_ = ecta_restrict(body_, ecta_tiles)
                elif (ecta_tiles and phases[i]["kind"] == "gemm" and str(phases[i].get("engine", "")).startswith("popsc_fp32")
                        and body_.count("smem, 0, 1, 1, ") == 1):
                    if i in fp32_place:
                        # Whole grid, rotated so the CTAs the TMA tiles leave idle take the first tasks (consecutive
                        # FP32 products continue the rotation), or the exact idle slice [tiles, gridDim.x).
                        mode_, rot_ = fp32_place[i]
                        slice_ = ("0, 1, 1" if mode_ == "whole" else
                                  f"(int)gridDim.x > {ecta_tiles} ? {ecta_tiles} : 0, (int)gridDim.x, (int)gridDim.x")
                        body_, count_ = re.subn(r"smem, 0, 1, 1, ([^,()]+)\)", rf"smem, {slice_}, {rot_} + (\1))", body_)
                        if count_ != 1:
                            raise RuntimeError("ecta FP32 placement: unexpected slice call form")
                    else:
                        # Release step 14: FP32 small-tile products (whole-grid CTA slice, no rotation token) take the
                        # CTA slice the interval's TMA tiles leave idle; every task's arithmetic is unchanged.
                        body_ = body_.replace("smem, 0, 1, 1, ", f"smem, {ecta_tiles}, {ECTA_GRID}, {ECTA_GRID}, ")
                parts.append(f"// Phase {i}: {', '.join(phases[i]['names'])}\n{{\n{body_}\n}}")
                part_ids.append(i)
                if "ctrace" in features and _ctrace_phase_stamps():
                    # Diagnostic only (profile launches): CTA b's thread 0 stamps the end of its phase i.
                    parts[-1] += ctrace_stamp(f"phase:{i}")
                if phases[i]["kind"] == "gemm" and any(
                        token in phases[i]["code"] for token in ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/", "/*T_VBASE*/")):
                    vbase += int(phases[i].get("tiles", 0))
            stamps = " ".join(f"ticks[{i+1}] = t;" for i in members)
            head_ = ctrace_stamp(f"start:{len(phase_source)}") + "\n" if "ctrace" in features else ""
            if "fsbatch" in features and compose and parts:
                # LATCH-TINYMOE-25: consecutive multi-CTA full sums share one body (ops.update_schedule).
                from .update_schedule import batch_full_sums
                parts, fsb_ = batch_full_sums(parts, part_ids, phases, warp_combine="fswarp" in features,
                                              mix="fsmix" in features, fence="fsfence" in features,
                                              leaves=bool(features & {"fsmixl", "fsload"}),
                                              unconditional="fsload" in features,
                                              fused={i: cast_fusions_[i] for i in part_ids if i in cast_fusions_},
                                              group=8 if "fsmix8" in features else None,
                                              spread="fsspread" in features,
                                              # LATCH-TRAIN-SYNC diagnostic only (profile launches of ctrace builds)
                                              stamp=ctrace_stamp if ("ctrace" in features and
                                                                     os.environ.get("LATCH_DIAGNOSTIC_FSB") == "1")
                                              else None)
                if fsb_:
                    group_records.setdefault("fsbatch", []).extend(fsb_)
                missing_ = {i for i in part_ids if i in cast_fusions_} - {
                    i for r_ in fsb_ for i in r_.get("fused", ())}
                if missing_:
                    raise LoweringError(f"cast_sums: fused sums outside a fused fsbatch body: {sorted(missing_)}")
            if "nosync" in features and compose:
                # LATCH-TINYMOE-25 (P6-C rule in the flat compose emitter): phases of one interval are independent;
                # a CTA barrier only orders shared-memory reuse, so none is placed between two neighbouring
                # shared-memory-free elementwise / counter-initialization phases.
                free_ = [phases[i]["kind"] in ("elementwise", "elementwise_inline", "reduction_counter_init")
                         and not re.search(r"smem|__shared__|__syncthreads|__shfl|\basm\b|grid\.sync", part_)
                         for i, part_ in zip(part_ids, parts)]
                joined_ = parts[0] if parts else ""
                for k_ in range(1, len(parts)):
                    joined_ += ("\n" if free_[k_ - 1] and free_[k_] else "\n__syncthreads();\n") + parts[k_]
            else:
                joined_ = "\n__syncthreads();\n".join(parts)
            phase_source.append(head_ + joined_
                                + (ctrace_stamp(f"end:{len(phase_source)}") if "ctrace" in features else "")
                                + "\ngrid.sync();\n"
                                f"if (ticks && blockIdx.x == 0 && threadIdx.x == 0) {{ const unsigned long long t = stamp(); {stamps} }}")
    commit_code = []
    for target, source in writes.items():
        if target in in_place_commits:
            continue
        scalar_commit = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layouts[target].numel}LL; i += (long long)gridDim.x * blockDim.x) {_ident(target)}[i] = {rounded(read({'ref':source}, layouts[target].shape), dtypes[target])};")
        vector_commit = _vector_commit(target, source, layouts, dtypes, virtual_text) if COMMIT_VECTOR else None
        commit_code.append(scalar_commit if vector_commit is None else vector_commit.replace("/*SCALAR*/", scalar_commit))
    if staged:
        # One small device function per stage (__REGION___s<k>), declaring only
        # the storage pointers that stage references.
        named = [(re.search(r"\b(v_\w+) =", line).group(1), line) for line in declarations]
        functions = []
        for case_index, code in enumerate(phase_source):
            used = [line for _, line in _used_tensor_declarations(named, code)]
            functions.append(f"__device__ __noinline__ void __REGION___s{case_index}(float** args, float* arena, float** out, int vrot, int cta0, int ctas"
                             + (", float* latch_ctr" if persistent_counters and counter_top else "") + ") {\n"
                             "  extern __shared__ float smem[];\n" + "\n".join(used) + "\n" + code + "\n}\n")
        kernel_body = "\n".join(functions)
    else:
      commit_comment = ("All old-state reads have completed before remaining state copies." if in_place_commits
                        else "All old-state reads have completed before any state mutation.")
      if compose and barrier_schedule == "dependency" and features & {"lptr", "lsreg"}:
          # Step-14 speed (P9-A port, fusion_p6d P9_LOCAL_PTRS / P9_OPAQUE_SREG): every barrier interval re-derives
          # the tensor pointers it uses from opaque copies of args/out/arena (and, with lsreg, of the special
          # registers), so no pointer or index live range crosses a grid barrier. Addresses and values unchanged.
          named_ = [(re.search(r"\b(v_\w+) =", line).group(1), line) for line in declarations]
          if LPTR_GLOBAL:
              # LATCH-MOESUP-ROW: opaque arena OFFSETS (not an opaque arena pointer): every tensor pointer still derives
              # from the global kernel parameter, so loads/stores stay in the global space (LDG/STG instead of generic
              # LD/ST); pointers read from args/out are asserted global. No live range crosses a barrier either way.
              def localize_(ident, line):
                  if "arena +" in line:
                      return re.sub(r"arena \+ (\d+)LL", r"arena + latch_ofs(\1LL)", line)
                  if "args[" in line or "out[" in line:
                      line = re.sub(r"\b(args|out)\[(\d+)\]", r"\1[latch_ofs(\2LL)]", line)
                      return line + f" __builtin_assume(__isGlobal({ident}));"
                  return line
              local_ = {ident: localize_(ident, line) for ident, line in named_}
          else:
              local_ = {ident: line.replace("args[", "latch_pa[").replace("out[", "latch_po[").replace("arena +", "latch_pr +")
                        for ident, line in named_}

          def scope_(code):
              used_ = [local_[ident] for ident, _ in _used_tensor_declarations(named_, code)]
              sregs_ = ""
              if "lsreg" in features:
                  for reg_, var_ in (("threadIdx.x", "lsr_tx"), ("blockIdx.x", "lsr_bx"), ("blockDim.x", "lsr_nt"),
                                     ("gridDim.x", "lsr_nb")):
                      if reg_ in code:
                          code = code.replace(reg_, var_)
                          sregs_ += f"unsigned {var_} = {reg_}; asm volatile(\"\" : \"+r\"({var_}));\n"
              if "lptr" not in features:
                  return "{\n" + sregs_ + code + "\n}"
              if LPTR_GLOBAL:
                  return "{\n" + sregs_ + "\n".join(used_) + "\n" + code + "\n}"
              opaque_ = "float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);"
              if any("latch_po[" in line for line in used_):
                  opaque_ += " float** latch_po = latch_opq(out);"
              return "{\n" + opaque_ + "\n" + sregs_ + "\n".join(used_) + "\n" + code + "\n}"
          if "lptr" in features:
              declarations = []
              commit_code = [scope_("\n".join(commit_code))] if commit_code else []
          phase_source = [scope_(code) for code in phase_source]
          helper_parts.append("#ifndef LATCH_OPQ_DEFINED\n#define LATCH_OPQ_DEFINED\n"
                              "template <class T> __device__ __forceinline__ T* latch_opq(T* p) { asm volatile(\"\" : \"+l\"(p)); return p; }\n"
                              "__device__ __forceinline__ long long latch_ofs(long long x) { asm volatile(\"\" : \"+l\"(x)); return x; }\n#endif")
          group_records["local_pointers"] = sorted(features & {"lptr", "lsreg"})
      kernel_body ="  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n" + "\n".join(declarations) + "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n" + "\n".join(phase_source) + f"\n// {commit_comment}\n" + "\n".join(commit_code) + ("" if region_body and not writes else "\n  grid.sync();") + f"\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{commit+1}] = stamp();\n"
    if "tmpf" in features and compose and not staged and barrier_schedule == "dependency" and len(phase_source) > 1:
        # Step-14 speed: before an interval's grid barrier, thread 0 of every CTA prefetches the tensor maps of the
        # next interval's TMA products (descriptor fetch leaves the next interval's critical path). Cache hint only.
        map_ref = re.compile(r"reinterpret_cast<const char\*>\(args\[(\d+)\]\) \+ (\d+)\)")
        prefetched_ = []
        for k_ in range(len(phase_source) - 1):
            maps_ = list(dict.fromkeys(map_ref.findall(phase_source[k_ + 1])))
            if not maps_:
                continue
            code_ = ("if (threadIdx.x == 0) {\n" + "\n".join(
                f'  asm volatile("prefetch.tensormap [%0];" :: "l"(reinterpret_cast<const char*>(args[{a}]) + {o}) : "memory");'
                for a, o in maps_) + "\n}\n")
            head_, sep_, tail_ = phase_source[k_].rpartition("grid.sync();")
            if sep_:
                phase_source[k_] = head_ + code_ + sep_ + tail_
                prefetched_.append(len(maps_))
        if prefetched_:
            kernel_body = ("  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n" + "\n".join(declarations)
                           + "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n" + "\n".join(phase_source)
                           + kernel_body[kernel_body.index("\n// All old-state reads"):] if "\n// All old-state reads" in kernel_body
                           else kernel_body)
            group_records["tensormap_prefetch"] = prefetched_
    if "/*XCLAIM*/nullptr" in kernel_body or "/*XCLAIM_S0*/" in kernel_body:
        raise LoweringError("a claimed segmented product was emitted outside the interval emitter (no counter)")
    if ectaq_count[0]:
        # Claim counters past the planned arena (never aliased), zeroed by CTA 0 at kernel entry; the first grid
        # barrier orders the zeroing before any claim (no queued product runs in the first interval).
        ectaq_base = (arena_words + 63) // 64 * 64
        arena_words = ectaq_base + -(-ectaq_count[0] // 64) * 64
        if ectaq_count[0] > 256:
            raise LoweringError("more than 256 per-launch claim counters (CTA 0 zeroes one word per thread)")
        kernel_body = kernel_body.replace("/*ECTAQ_BASE*/", f"{ectaq_base}LL")
        kernel_body = kernel_body.replace(
            "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n",
            f"\n  if (blockIdx.x == 0 && threadIdx.x < {ectaq_count[0]}u) "
            f"reinterpret_cast<unsigned*>(arena + {ectaq_base}LL)[threadIdx.x] = 0u;\n"
            "  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n", 1)
        group_records["ecta_queue_counters"] = {"base_word": ectaq_base, "count": ectaq_count[0]}
        if ("tmaq" in features and group_records.get("tma_claim")) or fill_claims_[0]:
            helper_parts.insert(0, "#define POPS4B_CLAIM_AHEAD 1")
    # LATCH-TINYMOE-MATRIX research screen only: existing compile-time variants of the TMA body (same arithmetic).
    for define_ in filter(None, os.environ.get("LATCH_SCREEN_DEFINES", "").split(";")):
        if define_ not in ("POPS4B_LAST_ARRIVER 1",):
            raise LoweringError(f"unknown screen define {define_!r}")
        helper_parts.insert(0, "#define " + define_)
        group_records.setdefault("screen_defines", []).append(define_)
    helpers = "\n".join(helper_parts)
    if "ctrace" in features and barrier_schedule == "dependency" and not staged:
        helpers = f"#define LATCH_CTSTAMP_BASE {commit + 2}\n" + helpers
    btrace_calls, btrace_phases = 0, []
    if "btrace" in features and compose and not staged and "pops4b_tma.cuh" in extra_headers:
        # Step-14 speed diagnostic: pops4b TMA body stamps (POPS4B_TRACE) after the program's other tick words.
        btrace_calls = kernel_body.count("pops4b::tma_phase_inline<")
        btrace_phases = [int(chunk.split(":", 1)[0]) for chunk in kernel_body.split("// Phase ")[1:]
                         for _ in range(chunk.count("pops4b::tma_phase_inline<"))]
        btrace_base = commit + 2 + 256 * len(ctrace_slots)
        helpers = "#define POPS4B_TRACE 1\n" + helpers
        kernel_body = kernel_body.replace(
            "\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n",
            f"\n  pops4b::trace_begin(ticks ? reinterpret_cast<unsigned long long*>(ticks) + {btrace_base} : nullptr);\n"
            "  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n", 1)
    if compose and not staged and "pops4b_tma.cuh" in extra_headers and features & {"tla", "teu4", "teu8", "tper"}:
        # Step-14 speed: the P7-B / P9-E pops4b TMA body switches in the flat common planner (same arithmetic):
        # last-arriver stage refill, staged-epilogue chunk unroll, launch-persistent stage barriers.
        if "tla" in features:
            helpers = "#define POPS4B_LAST_ARRIVER 1\n" + helpers
        if features & {"teu4", "teu8"}:
            helpers = f"#define POPS4B_EPI_UNROLL {8 if 'teu8' in features else 4}\n" + helpers
        if "tper" in features:
            helpers = "#define POPS4B_PERSIST 1\n" + helpers
            kernel_body = kernel_body.replace("\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n",
                                              "\n  pops4b::tma_persist_init();\n"
                                              "  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n", 1)
        group_records["tma_body_switches"] = sorted(features & {"tla", "teu4", "teu8", "tper"})
    shared_bytes = max([17408] + [phase.get("shared_bytes", 0) for phase in phases])
    # Exact row statistics may reuse storage already required by another
    # phase. Selection cannot raise the launch allocation or change its grid.
    from .exact_vocab_cache import select as select_exact_vocab_cache, HEADER as exact_vocab_cache_header
    kernel_body, cached_vocab = select_exact_vocab_cache(phases, kernel_body, shared_bytes)
    if cached_vocab:
        extra_headers.append(exact_vocab_cache_header)
        helpers += f'\n#include "{exact_vocab_cache_header}"\n'
    from .resources import select_program_resources
    resource_policy = select_program_resources(shared_bytes, phases=phases, arch=target_arch)
    grid_bounds = [p["max_grid_blocks"] for p in phases if p.get("max_grid_blocks") is not None]
    max_grid_blocks = min(grid_bounds) if grid_bounds else None
    source = _translation_unit(kernel_body, helpers=helpers, shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'],
                               max_grid_blocks=max_grid_blocks)
    header_hashes = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest() for name in ("core.cuh", "gemm.cuh", "epi.cuh", "plan.py", *extra_headers)}
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    # Count only expressions surviving final emission (including inlined ones),
    # not eligible source IR hidden by an opaque body or dead-code elimination.
    emitted_scalar_divisions = []
    for selection in scalar_division_decisions.values():
        marker = selection["marker"]
        if marker in source:
            record = {key: value for key, value in selection.items() if key != "marker"}
            record["phase_indices"] = [index for index, phase in enumerate(phases)
                if marker in phase.get("code", "") or any(marker in line for line in phase.get("body", ()))]
            emitted_scalar_divisions.append(record)
    manifest = {
        "format": "latch-ops-cuda-v1", "source_hash": source_hash, "header_hashes": header_hashes,
        "inputs": inputs, "outputs": outputs, "structured_outputs": structured_outputs,
        "input_roles": {name: {"role": _field(values[name], "role", "input"), "parameter": _field(values[name], "parameter")} for name in inputs},
        "state_writes": writes, "layouts": aliases,
        **({"in_place_commits": in_place_commits,
            "copied_commits": {t: s for t, s in writes.items() if t not in in_place_commits}} if compose else {}),
        "optimizer_state_schema": _field(graph, "metadata", {}).get("optimizer_state_schema", []),
        "offsets": offsets, "arena_words": arena_words, "arena_bytes": 4 * arena_words,
        "lifetimes": classes, "retained_all": retain_all, "views": views,
        "undefined_metadata_only": sorted(undefined),
        "attention_groups": attention_decisions, "attention_refusals": attention_refusals,
        "ordered_index_gradients": index_decisions,
        **({"selective_scans": scan_decisions} if scan_decisions else {}),
        "phases": [{k: v for k, v in p.items() if k not in {"code", "body", "registers", "popsc_call", "code_r1", "row_parts"}} for p in phases],
        "phase_count": commit + 1, "barrier_count": len(intervals) + 1,
        "barrier_schedule": barrier_schedule, "barrier_intervals": intervals,
        **({"row_chain_substages": {str(k): v for k, v in rc_substages.items()}} if rc_substages else {}),
        **({"cast_sum_substages": {str(k): v for k, v in cs_substages.items()}} if cs_substages else {}),
        "max_gemm_tiles": max([p.get("tiles", 0) for p in phases] or [1]),
        "shared_bytes": shared_bytes,
        **({"max_grid_blocks": max_grid_blocks} if max_grid_blocks is not None else {}),
        "matrix_decisions": matrix_decisions,
        **({"scalar_divisions": emitted_scalar_divisions} if emitted_scalar_divisions else {}),
        **({"conv_virtual_operands": conv_taps.virtual,
            "conv_materialized_taps": sorted(set(n for p in phases for n in p["names"] if p.get("targets") in ([_conv_ops.TAPS_TARGET], [_conv_ops.IM2COL_TARGET])))}
           if conv_taps.virtual or conv_taps.enabled else {}),
        "primitive_policy": {"matrix_policy": matrix_policy, "split_k": split_k,
                             "exact_reductions": exact_reductions, "target_arch": target_arch},
        "numerical_policy": ("Explicit BF16 casts/operands/results; BF16 MMA with FP32 accumulation; FP32 master state; fixed reduction order; no TF32/data atomics"
                             if "bfloat16" in dtypes.values() else
                             "FP32 SIMT fmaf GEMM; separately rounded pointwise arithmetic; fixed CTA reduction tree; no TF32/atomics on data"),
        "execution": f"one cooperative CUDA launch, 256 threads/CTA, {shared_bytes} shared bytes/CTA; occupancy admitted; final state commit",
        "memory_plan": "pinned plan.assign + independent plan.check; inclusive phase intervals and alias roots",
    }
    if resource_policy['has_popsb']:
        manifest['resource_policy'] = resource_policy
    if preserve_full_reductions:
        manifest['region_reduction_policy'] = {
            'policy': 'preserve ordered full-grid scalar reduction trees',
            'outputs': [p['writes'][0] for p in phases if p.get('engine') == 'multi_cta_full'],
            'arithmetic': 'per-CTA contiguous chunk trees, then fixed-order combine of CTA partials',
            'grid_dependency': 'same arithmetic as the ordered donor at the same physical cooperative grid size',
            'accumulation': 'one FP32 caller addition after the completed reduction and scale',
        }
    if preserve_matrix_plans:
        manifest['region_matrix_policy'] = {
            'policy': 'ordinary FP32 selector before output placement; preserve body and K-slice plan',
            'outputs': [p.get('product', p['writes'][0]) for p in phases if p.get('preserved_store_plan')],
            'accumulation': 'one FP32 caller addition after the completed product',
            'factored_copy_optimization': 'explicit copies retain the ordinary product layout and selection',
        }
    if region_expression_inline:
        manifest['region_expression_policy'] = {
            'policy': 'bounded pure FP32 producers evaluated by existing mapped expression renderer',
            'inlined': sorted(virtual_text), 'expression_costs': region_inline_costs,
            'per_use_cost_limit': 12, 'total_use_cost_limit': 24,
            'reduction_sources': inlined_reductions,
            'arithmetic': 'unchanged per-operation expressions; unchanged reduction leaves and trees',
            'protected': 'outputs, tape bindings, accumulation destinations, state writes and matrix operands',
        }
    if dynamic:
        import copy
        manifest["dynamic_batch"] = copy.deepcopy(dynamic.declaration)
        manifest["dynamic_batch"]["storage_policy"] = "capacity layouts and strides; active material rows only"
        if dynamic.storage_axes:
            manifest["dynamic_batch"]["internal_value_axes"] = dict(dynamic.storage_axes)
        for name, axis in {**dynamic.axes, **dynamic.storage_axes}.items():
            if name in manifest["layouts"]:
                manifest["layouts"][name]["active_axis"] = axis
    if output_bindings is not None:
        manifest["output_placement"] = {name: {"output": index, "mode": mode,
                                               "zero_omitted": name in zero_contributions}
                                        for name, (index, mode) in placement.items()}
        manifest["elided_clones"] = elided
        manifest["placed_outputs"] = sorted(index for index, _ in placement.values())
    if region_body:
        manifest["region_body"] = True
    if staged:
        manifest["stages"] = stage_info
    if persistent_counters and counter_top:
        manifest["persistent_counter_words"] = counter_top
    if row_fusions:
        manifest["row_fusions"] = row_fusions
    if deferred:
        manifest["deferred_products"] = deferred
    if padded_operands:
        manifest["padded_matrix_operands"] = {"policy": "BF16 matrix-only operands: row pitch rounded to 8 elements",
                                              "values": padded_operands}
    if epilogues:
        manifest["matrix_epilogues"] = epilogues
    if "ctrace" in features and barrier_schedule == "dependency" and not staged and ctrace_slots:
        manifest["tick_words"] = manifest["phase_count"] + 1 + 256 * len(ctrace_slots)
        group_records["ctrace"] = {"base": manifest["phase_count"] + 1, "stride": 256, "slots": ctrace_slots}
    if btrace_calls:
        base_ = manifest.get("tick_words", manifest["phase_count"] + 1)
        manifest["tick_words"] = base_ + btrace_calls * 6 * 256
        group_records["btrace"] = {"base": base_, "calls": btrace_calls, "events": 6, "stride": 256,
                                   "call_phases": btrace_phases}
    if group_records:
        manifest["group_rules"] = group_records
    if dead:
        manifest["elided_stores"] = sorted(dead)
    if elided_layouts:
        manifest["elided_layouts"] = elided_layouts
        used_identifiers = set(re.findall(r"\bv_\w+\b", kernel_body + "\n" + helpers))
        dangling = sorted(name for name in elided_layouts if _ident(name) in used_identifiers)
        if dangling:
            raise LoweringError("generated code references elided storage: " + ", ".join(dangling))
    if compose:
        record_plan(manifest, phases=phases, layouts=layouts, dtypes=dtypes)
        if dynamic:
            manifest["execution_plan"]["active_extents"] = {
                "input": dynamic.input, "minimum": 1, "maximum": dynamic.maximum,
                "value_axes": dict(dynamic.axes),
                "storage": "capacity-sized inclusive lifetimes; active extents do not alter alias/storage hazards"}
    program = GeneratedProgram(source, manifest, kernel_body, helpers)
    if pops2_layouts:
        program._selection_contracts = {"pops2_layouts": pops2_layouts}
    return program


def _translation_unit(body, *, helpers="", functions="", shared_bytes=17408, resident_blocks_per_sm=1, cluster=0,
                      max_grid_blocks=None):
    from .launch_cache import admitted_launch_source
    if type(shared_bytes) is not int or not 0 < shared_bytes <= 232448:
        raise LoweringError("cooperative program shared storage exceeds the admitted 227 KiB opt-in limit")
    if type(resident_blocks_per_sm) is not int or resident_blocks_per_sm not in (1, 2):
        raise LoweringError("cooperative program supports one or two resident CTAs/SM")
    if max_grid_blocks is not None and (type(max_grid_blocks) is not int or max_grid_blocks <= 0):
        raise LoweringError("cooperative body grid bound must be a positive integer")
    launch_bounds = '__launch_bounds__(256, 2) ' if resident_blocks_per_sm == 2 else ''
    optin = (f"  e = cudaFuncSetAttribute(latch_ops, cudaFuncAttributeMaxDynamicSharedMemorySize, {shared_bytes});\n"
             "  if (e != cudaSuccess) return e;\n" if shared_bytes > 49152 else "")
    if cluster > 8:
        # P8-D: clusters above the portable size of 8 (H100 admits 16) need an explicit opt-in.
        optin += ("  e = cudaFuncSetAttribute(latch_ops, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n"
                  "  if (e != cudaSuccess) return e;\n")
    return '''// Generated LATCH-OPS operation program. Do not edit: see manifest.json.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "gemm.cuh"
#include "epi.cuh"
namespace cg = cooperative_groups;
__device__ __forceinline__ unsigned long long stamp() {
  unsigned long long x; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(x)); return x;
}
''' + helpers + "\n" + functions + '''
extern "C" __global__ ''' + launch_bounds + '''void latch_ops(float** args, float* arena, unsigned long long* ticks) {
''' + body + "\n}\n" + '''
extern "C" int latch_query(int* max_blocks, int* multiprocessors) {
  int device; cudaError_t e = cudaGetDevice(&device); if (e != cudaSuccess) return e;
  cudaDeviceProp prop; e = cudaGetDeviceProperties(&prop, device); if (e != cudaSuccess) return e;
  if (!prop.cooperativeLaunch) return cudaErrorNotSupported;
''' + optin + '''  int active; e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, latch_ops, 256, ''' + str(shared_bytes) + ''');
  if (e != cudaSuccess) return e;
  *multiprocessors = prop.multiProcessorCount; *max_blocks = active * prop.multiProcessorCount;
''' + (_cluster_query(shared_bytes, cluster) if cluster else '') + (f"  if (*max_blocks > {max_grid_blocks}) *max_blocks = {max_grid_blocks};\n"
       if max_grid_blocks is not None else '') + '''  return *max_blocks > 0 ? cudaSuccess : cudaErrorLaunchOutOfResources;
}
extern "C" int latch_launch(float** args, float* arena, unsigned long long* ticks, int blocks, void* stream) {
  int max_blocks, sms; int e = latch_query(&max_blocks, &sms); if (e) return e;
  if (blocks <= 0 || blocks > max_blocks) return cudaErrorInvalidConfiguration;
  void* kernel_args[] = {&args, &arena, &ticks};
''' + (_cluster_launch(shared_bytes, cluster) if cluster else '''  return cudaLaunchCooperativeKernel((const void*)latch_ops, dim3(blocks), dim3(256), kernel_args, ''' + str(shared_bytes) + ''', (cudaStream_t)stream);
''') + '''}
extern "C" const char* latch_error(int error) { return cudaGetErrorString((cudaError_t)error); }
''' + admitted_launch_source(shared_bytes, cluster=cluster)


def _cluster_query(shared_bytes, cluster):
    """P-OPS5-B: co-resident capacity of a cluster launch (whole clusters only)."""
    return f'''  {{
    cudaLaunchConfig_t cfg = {{}};
    cfg.gridDim = dim3({cluster}); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = {shared_bytes};
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = {cluster}; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    int clusters = 0; e = cudaOccupancyMaxActiveClusters(&clusters, latch_ops, &cfg); if (e != cudaSuccess) return e;
    if (clusters * {cluster} < *max_blocks) *max_blocks = clusters * {cluster};
  }}
'''


def _cluster_launch(shared_bytes, cluster):
    """P-OPS5-B: cooperative launch with a thread-block cluster dimension (grid rounded down to whole clusters)."""
    return f'''  (void)kernel_args;
  blocks -= blocks % {cluster};
  if (blocks <= 0) return cudaErrorInvalidConfiguration;
  cudaLaunchConfig_t cfg = {{}};
  cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = {shared_bytes}; cfg.stream = (cudaStream_t)stream;
  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = {cluster}; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeCooperative; at[1].val.cooperative = 1;
  cfg.attrs = at; cfg.numAttrs = 2;
  return cudaLaunchKernelEx(&cfg, latch_ops, args, arena, ticks);
'''


def lower_conditional(region, *, retain_all=False, fuse_elementwise=True, policy="auto", **options):
    """Compose two ordinary AD regions under one uniform device predicate.

    Only the selected region executes or commits state. Its scratch arena may
    overlap the other region's scratch, while stable merged outputs live beyond
    both. Differentiation was performed by PyTorch within each region; the bool
    predicate has no derivative. This is one region boundary, not a masked blend.
    """
    policy_record = conditional_policy()
    branch_token = _CONDITIONAL_BRANCH.set(_LOWERING_DEPTH.get() + 1)
    try:
        branches = [lower(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise,
                          policy=policy, **options)
                    for graph in (region.true_branch, region.false_branch)]
    finally:
        _CONDITIONAL_BRANCH.reset(branch_token)
    predicate = region.predicate
    if _dtype(predicate) != "bool" or _numel(predicate.shape) != 1:
        raise LoweringError("uniform conditional requires one explicit boolean predicate")
    first, second = [program.manifest for program in branches]
    if len(first["inputs"]) != len(second["inputs"]) or len(first["outputs"]) != len(second["outputs"]):
        raise LoweringError("conditional branch ABI arity mismatch")
    for a, b in zip(first["inputs"], second["inputs"]):
        for key in ("shape", "strides", "dtype"):
            if first["layouts"][a][key] != second["layouts"][b][key]:
                raise LoweringError("conditional branch input metadata mismatch")
    inputs, outputs = list(region.inputs), list(region.output_names())
    if len(inputs) not in (len(first["inputs"]), len(first["inputs"]) + 1) or predicate.name not in inputs:
        raise LoweringError("conditional predicate must be an explicit common input or one final selector")
    layouts = {name: {**first["layouts"][local], "root": name}
               for name, local in zip(inputs, first["inputs"])}
    layouts[predicate.name] = {"shape": list(predicate.shape), "strides": list(predicate.strides),
                               "root": predicate.name, "offset": 0, "dtype": "bool"}
    offsets = {}
    scratch_words = arena_plan.a64(max(program.manifest["arena_words"] for program in branches))
    top = scratch_words
    # STEP-11: output slot -> region input whose binding both branches commit it to.
    committed = {}
    if policy_record["alias_committed_state"]:
        sources = [{value: manifest["inputs"].index(target) for target, value in manifest["state_writes"].items()}
                   for manifest in (first, second)]
        for name, a, b in zip(outputs, first["outputs"], second["outputs"]):
            ia, ib = sources[0].get(a), sources[1].get(b)
            if ia is not None and ia == ib:
                target = inputs[ia]
                la, lt = first["layouts"][a], layouts[target]
                if la["shape"] == lt["shape"] and la["dtype"] == lt["dtype"]:
                    committed[name] = target
    for name, a, b in zip(outputs, first["outputs"], second["outputs"]):
        la, lb = first["layouts"][a], second["layouts"][b]
        if la["shape"] != lb["shape"] or la["dtype"] != lb["dtype"]:
            raise LoweringError("conditional branch output metadata mismatch")
        if name in committed:
            # The selected branch's state commit has already written this value
            # into the input binding before the region's final grid barrier.
            layouts[name] = {**layouts[committed[name]], "root": committed[name]}
            continue
        layouts[name] = {"shape": la["shape"], "strides": list(_contiguous(la["shape"])),
                         "root": name, "offset": 0, "dtype": la["dtype"]}
        offsets[name] = top
        top += arena_plan.a64((_numel(la["shape"]) * _BYTES[la["dtype"]] + 3) // 4)
    functions = []
    for label, program in zip(("true", "false"), branches):
        manifest = program.manifest
        copies = []
        for result, local in zip(outputs, manifest["outputs"]):
            if result in committed or policy_record["skip_copies"]:
                continue
            desc = manifest["layouts"][local]
            root = desc["root"]
            ptr = (f"args[{manifest['inputs'].index(root)}]" if root in manifest["inputs"]
                   else f"arena + {manifest['offsets'][root]}LL")
            ctype = _CTYPES[desc["dtype"]]
            layout = TensorLayout(tuple(desc["shape"]), tuple(desc["strides"]), root, desc["offset"])
            source = f"reinterpret_cast<{ctype}*>({ptr})[{layout.offset}LL + {_index(layout, layout.shape)}]"
            source_words = (0 if root in manifest["inputs"] else manifest["offsets"][root]) + int(desc["offset"])
            if (policy_record["vector_copies"] and _BYTES[desc["dtype"]] == 4 and layout.numel % 4 == 0
                    and list(desc["strides"]) == list(_contiguous(desc["shape"]))
                    and source_words % 4 == 0 and offsets[result] % 4 == 0):
                # STEP-11: dense 4-byte outputs copy as 16-byte words (bitwise; args are 16-byte aligned).
                copies.append(f"{{ const float4* s = reinterpret_cast<const float4*>(reinterpret_cast<const float*>({ptr}) + {int(desc['offset'])}LL); "
                              f"float4* d = reinterpret_cast<float4*>(arena + {offsets[result]}LL); "
                              f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel // 4}LL; "
                              f"i += (long long)gridDim.x * blockDim.x) d[i] = s[i]; }}")
                continue
            copies.append(f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(arena + {offsets[result]}LL)[i] = {source};")
        functions.append(f"namespace latch_detail_{label} {{\n{program.functions}\n"
                         f"__device__ __forceinline__ void run(float** args, float* arena, unsigned long long* ticks) {{\n{program.kernel_body}\n" + "\n".join(copies) + "\n}\n}\n"
                         if program.functions else
                         f"__device__ __forceinline__ void latch_region_{label}(float** args, float* arena, unsigned long long* ticks) {{\n{program.kernel_body}\n" + "\n".join(copies) + "\n}\n")
    helpers = "\n".join(dict.fromkeys(program.helpers for program in branches if program.helpers))
    body = f'''  cg::grid_group grid = cg::this_grid();
  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();
  // Predicate is immutable for this launch and identical for the whole grid.
  if (*reinterpret_cast<const bool*>(args[{inputs.index(predicate.name)}])) {{
    {'latch_detail_true::run' if branches[0].functions else 'latch_region_true'}(args, arena, nullptr);
  }} else {{
    {'latch_detail_false::run' if branches[1].functions else 'latch_region_false'}(args, arena, nullptr);
  }}
  grid.sync();
  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();
'''
    shared_bytes = max(program.manifest.get("shared_bytes", 17408) for program in branches)
    header_hashes = {}
    for program in branches:
        for name, digest in program.manifest["header_hashes"].items():
            if name in header_hashes and header_hashes[name] != digest:
                raise LoweringError("conditional branches disagree on pinned header identity")
            header_hashes[name] = digest
    from .resources import select_program_resources
    resource_policy = select_program_resources(shared_bytes,
        children=[program.manifest for program in branches], arch=options.get('target_arch', 'sm_90'))
    source = _translation_unit(body, helpers=helpers, functions="\n".join(functions), shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource_policy['resident_blocks_per_sm'])
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    remap = dict(zip(first["outputs"], outputs))
    writes = {name: remap[value] for name, value in first["state_writes"].items()}
    manifest = {
        "format": "latch-ops-cuda-v2", "source_hash": source_hash, "header_hashes": header_hashes,
        "inputs": inputs, "outputs": outputs, "structured_outputs": region.outputs,
        "input_roles": {**first["input_roles"], predicate.name: {"role": "predicate", "parameter": None}},
        "state_writes": writes, "layouts": layouts, "offsets": offsets, "arena_words": top,
        "optimizer_state_schema": first.get("optimizer_state_schema", []),
        "arena_bytes": 4 * top, "branch_scratch_words": scratch_words,
        "retained_all": retain_all, "phases": [], "phase_count": 1,
        "barrier_count": {"true": first["barrier_count"] + 1, "false": second["barrier_count"] + 1},
        "max_gemm_tiles": max(first["max_gemm_tiles"], second["max_gemm_tiles"]),
        "shared_bytes": shared_bytes,
        "branches": {"true": first, "false": second}, "predicate_input": predicate.name,
        **({"resource_policy": resource_policy} if resource_policy['has_popsb'] else {}),
        "execution": "one cooperative launch; uniform device if executes and commits selected AD region only",
        "conditional_policy": {**policy_record, "committed_state_views": committed},
        "memory_plan": "independently checked branch arenas overlap by mutual exclusion; merged outputs are disjoint",
        "numerical_policy": "; ".join(dict.fromkeys(p.manifest["numerical_policy"] for p in branches)),
    }
    return GeneratedProgram(source, manifest, body, helpers, "\n".join(functions))
