"""LATCH-STEP-04: indexed-gradient accumulation selection for the common composition path.

One narrow entry point replaces the direct `transformer_lower.ordered_embedding_backward`
call inside `_lower_serial`. It returns a normalized phase list (dict phases with
explicit reads/writes) for every implementation:

* outside `execution="compose"` with no explicit policy, the frozen base emitter
  (P-OPS-D ordered occurrence lists) is returned unchanged;
* under compose, the automatic engine is selected from operation, shape, dtype and
  layout only (no model or workload names); an explicit
  `indexed_reduction.policy(...)` always wins, including "baseline".

Donor implementations: OMNI-P1-A `segmented_v1` (82afa77f7) in indexed_segmented.py and
the OMNI-C1 stable-merge reducers in indexed_reduction.py.
"""
from __future__ import annotations

COMPOSE_DEFAULT = "segmented_v1"


# LATCH-STEP-04: automatic engine. Global float atomics (atomic_v1) are admitted FP32 accumulation for natives as
# for conventional controls and measured fastest; deterministic accumulation stays available through
# indexed_reduction.policy("owner_v1") (bitwise repeatable for a fixed grid) or "segmented_v1".
AUTOMATIC_ATOMIC = True


def automatic_engine(n, rows, width):
    """Shape-only choice: atomic accumulation; else row ownership within its bounds; else segmented."""
    from .indexed_owner import supported
    if AUTOMATIC_ATOMIC:
        from .indexed_atomic import supported as atomic_supported
        if atomic_supported(n, rows, width)[0]:
            return "atomic_v1"
    return "owner_v1" if supported(n, rows, width)[0] else "segmented_v1"


def _normalize_base(indexed, args, names):
    from .lower import _ref
    phases = []
    for kind, writes, code, extra in indexed["phases"]:
        reads = ([_ref(args[1], names)] if len(writes) > 1 else [_ref(args[0], names), *indexed["internal"]])
        phases.append({"kind": kind, "writes": list(writes), "reads": reads, "code": code, **extra})
    return {"auxiliary": indexed["auxiliary"], "phases": phases, "helpers": "",
            "decision": "ordered_occurrence_list", "source_commit": indexed["source_commit"],
            "accumulation_order": indexed["accumulation_order"], "indexed_reduction_policy": "frozen_base"}


def embedding_backward(args, kwargs, name, shape, layouts, dtypes, names, *, cache, staged=False,
                       arithmetic=None):
    """Normalized phases for `aten.embedding_dense_backward`, or None for ordinary lowering."""
    from .execution_plan import current_execution
    from . import indexed_reduction
    requested = indexed_reduction.requested_policy()
    # Packed BF16 gathers need a BF16 VJP boundary. Accumulate exact widened
    # occurrence rows with the existing deterministic segmented FP32 engine,
    # then round each completed result once. Inactive rows stay exact +0.
    # Even a single occurrence uses +0 + value, so a negative zero cotangent
    # is canonicalized to +0, unlike an explicitly bijective copy VJP.
    # This is operation/dtype admission, independent of router/model identity.
    if dtypes.get(name) == "bfloat16":
        from .lower import LoweringError, TensorLayout, _contiguous, _ident, _ref
        grad = _ref(args[0], names)
        if dtypes[grad] != "bfloat16" or staged or arithmetic is not None:
            raise LoweringError("BF16 embedding VJP requires ordinary untagged BF16 segmented composition")
        temporary = name + "__fp32_accumulation"
        if temporary in names or temporary in layouts:
            raise LoweringError("BF16 embedding accumulation scratch collision")
        temporary_layouts = dict(layouts)
        temporary_layouts[temporary] = TensorLayout(tuple(shape), _contiguous(shape), temporary)
        temporary_dtypes = dict(dtypes, **{temporary: "float32"})
        with indexed_reduction.policy("segmented_v1"):
            indexed = indexed_reduction.ordered_embedding_backward(
                args, dict(kwargs, latch_row_widen=True), temporary, shape, temporary_layouts,
                temporary_dtypes, names | {temporary}, cache=cache)
        elements = layouts[name].numel
        indexed["auxiliary"].append((temporary, tuple(shape), "float32"))
        code = f"""for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < {elements}LL; i += (long long)gridDim.x * blockDim.x) {{
  {_ident(name)}[i] = __float2bfloat16_rn({_ident(temporary)}[i]);
}}"""
        indexed["phases"].append(dict(kind="indexed_cast", writes=[name], reads=[temporary],
            code=code, work_items=(elements + 255) // 256, shared_bytes=0,
            engine="bf16_indexed_final_round", arithmetic="one BF16 rounding after completed FP32 row sum"))
        indexed["indexed_selection"] = dict(requested=requested, selected="bf16_segmented_v1",
            rule="BF16 gather derivative: widened FP32 accumulation and one final BF16 rounding")
        return indexed
    if arithmetic is not None:
        if staged or current_execution() != "compose":
            from .lower import LoweringError
            raise LoweringError("explicit embedding arithmetic requires ordinary common composition")
        indexed = indexed_reduction.ordered_embedding_backward(
            args, kwargs, name, shape, layouts, dtypes, names, cache=cache, arithmetic=arithmetic)
        indexed["indexed_selection"] = {"requested": requested, "selected": "ordered_embedding_segment10",
                                         "rule": "explicit per-operation learned embedding VJP contract"}
        return indexed
    if requested == "frozen_base" or (requested is None and (staged or current_execution() != "compose")):
        # Staged loop functions have no grid-group argument and must not contain
        # internal grid barriers. Preserve their qualified ordered occurrence
        # body; standalone compose keeps the automatic atomic implementation.
        from .transformer_lower import ordered_embedding_backward
        indexed = ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names)
        return None if indexed is None else _normalize_base(indexed, args, names)
    if requested is None:
        from .lower import _ref
        n = layouts[_ref(args[1], names)].numel
        selected = automatic_engine(n, int(args[2]), shape[1])
    else:
        selected = requested
    with indexed_reduction.policy(selected):
        indexed = indexed_reduction.ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names,
                                                               cache=cache)
    if indexed is not None:
        if staged and any(phase.get("internal_grid_barriers") for phase in indexed["phases"]):
            from .lower import LoweringError
            raise LoweringError("staged indexed bodies cannot contain internal grid barriers; "
                                "use the automatic ordered staged selection")
        indexed["indexed_selection"] = {"requested": requested, "selected": selected,
                                        "rule": "explicit policy" if requested else "compose automatic engine"}
    return indexed


def plan_norms(operations, names, *, field, output_name, ref, flatten_refs):
    """Clip-norm terms sum(mul(g, g)) over embedding gradients g: {g: (square, sum)}.

    Only the exact capture form (optim.py global clipping) with a square consumed solely by a
    full sum.default is matched; anything else keeps ordinary lowering.
    """
    produced, consumers = {}, {}
    for op in operations:
        out = output_name(op)
        produced[out] = (str(field(op, "target", field(op, "op"))), op)
        for r in set(flatten_refs((list(field(op, "args", [])), dict(field(op, "kwargs", {}))), names)):
            consumers.setdefault(r, []).append(out)
    plan = {}
    for out, (target, op) in produced.items():
        if "embedding_dense_backward" not in target:
            continue
        for square in consumers.get(out, []):
            s_target, s_op = produced[square]
            s_args = list(field(s_op, "args", []))
            if (not s_target.endswith("mul.Tensor") or len(s_args) != 2
                    or ref(s_args[0], names) != out or ref(s_args[1], names) != out):
                continue
            users = consumers.get(square, [])
            if len(users) != 1:
                continue
            u_target, u_op = produced[users[0]]
            u_args = list(field(u_op, "args", []))
            if u_target.endswith("sum.default") and len(u_args) == 1 and not field(u_op, "kwargs", {}):
                plan[out] = (square, users[0])
                break
    return plan


def legacy_engine(args, kwargs, name, shape, layouts, dtypes, names, *, cache, norm=None, flags=None, read=None):
    """Static fused emitters: a reusable engine only when a non-base policy is explicitly selected.

    Compose passes the engine of its common program explicitly when it retains the
    established static emitter, so both schedules use the same indexed arithmetic.
    """
    from . import indexed_reduction
    requested = indexed_reduction.requested_policy()
    if requested in (None, "frozen_base"):
        return None
    indexed = indexed_reduction.ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names,
                                                           cache=cache, norm=norm, flags=flags, read=read)
    if indexed is not None:
        if any(len(phase["writes"]) != 1 and not phase.get("fused_norm") for phase in indexed["phases"]):
            from .lower import LoweringError
            raise LoweringError("static fused emitters admit single-output indexed phases only")
        indexed["indexed_selection"] = {"requested": requested, "selected": requested, "rule": "explicit policy"}
    return indexed


def engine_of(manifest):
    """The indexed policy a lowered program used (None for the frozen base emitter)."""
    policies = {row.get("indexed_reduction_policy") for row in manifest.get("ordered_index_gradients", [])}
    policies.discard(None)
    policies.discard("frozen_base")
    if len(policies) > 1:
        return "mixed"
    return next(iter(policies), None)


# LATCH-STEP-04: forward-gather inlining is chosen per program by a static count. Retained static programs are
# lowered with and without inlining embedding gathers into their consumers; the variant with fewer grid barriers
# (phase boundaries plus barriers inside bodies) is kept, ties keep the gathers materialized. Static structure only.
GATHER_SELECTION = True


def choose_gather_inlining(first, relower):
    from . import fusion_reference
    from .lower import LoweringError
    if not GATHER_SELECTION or fusion_reference.INLINE_EMBEDDING_GATHER:
        return first
    fusion_reference.INLINE_EMBEDDING_GATHER = True
    try:
        alternative = relower()
    except LoweringError as error:
        alternative, refusal = None, str(error)
    else:
        refusal = None
    finally:
        fusion_reference.INLINE_EMBEDDING_GATHER = False
    def barriers(program):
        return len(program.manifest["phases"]) + int(program.manifest.get("internal_grid_barriers", 0))
    phases = barriers(first)
    alt_phases = None if alternative is None else barriers(alternative)
    keep = (alternative is not None and engine_of(alternative.manifest) == engine_of(first.manifest)
            and alt_phases < phases)
    chosen = alternative if keep else first
    chosen.manifest["gather_inlining_selection"] = {
        "rule": "fewer grid barriers (phases + internal body barriers); ties keep materialized gathers",
        "materialized_barriers": phases, "inlined_barriers": alt_phases, "selected": "inlined" if keep else "materialized",
        **({"inlined_refusal": refusal} if refusal else {})}
    return chosen
