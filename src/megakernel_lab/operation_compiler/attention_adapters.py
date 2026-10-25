# Portable registry and phase contracts from LATCH-ATTENTION commit 1ad3e78.
# Integrated here without the peer legacy-host callbacks; ComposedAttention owns scheduling.
"""LATCH-ATTENTION: existing attention bodies as reusable execution adapters.

No new attention arithmetic lives here. Each registered body is an existing,
attributed implementation (its matcher, emitter and CUDA header are unchanged);
this module states what every body needs from a host and makes the phases
self-describing for the shared execution contract (``ops.execution_plan``):

* layouts: the 4-D (batch, head, row, unit) operand/result strides each body
  resolves from the captured graph, recorded per phase;
* dependencies: complete reads/writes of every phase (forward -> backward
  hand-off values such as row statistics are ordinary reads/writes);
* intermediate storage: each synthetic value's shape, dtype and lifetime class
  (``phase`` scratch, ``forward_to_backward`` hand-off, ``zeroed_counter``
  scratch that must be zero at entry and is restored to zero by the body);
* resources: 256-thread CTA ABI, dynamic shared bytes, work items, target
  architecture (``sm_90a`` for warpgroup bodies) and program-level requirements
  (private CUDA module state, program-wide device arrays);
* synchronization: bodies never assume more than a publication barrier between
  dependent phases; bodies with internal grid barriers declare them.

Selection is per attention group and depends only on the group's operand class
(declared precision recipe, recognized by graph dtypes), shape, mask and the
target, never on unrelated surrounding operations. This intake retains the
portable registry and phase interface from peer commit 1ad3e78. It does not
install the peer's legacy-host callbacks. The common planner currently consumes
the bf16b_short phase interface; existing long/mixed and loop adapters retain
their established paths. Strict short attention keeps all three pieces.

Peer qualification includes a preserved shared-bf16b scalar-loss failure.
Body admission is not a claim that every complete training update passes its
numerical gate; fresh integrated qualification remains required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

VERSION = 1
THREADS = 256
MAX_SHARED = 232448


@dataclass(frozen=True)
class Body:
    """Static contract of one existing attention body family."""
    name: str
    lineage: str
    module: str
    headers: tuple
    operand_class: str          # fp32 | bf16_boundary | mixed_bf16 | bf16_operands | mixed_bf16_rect
    context: str                # flat | loop_tape | loop_opaque
    head_widths: tuple
    sequence: str               # human-readable admitted context rule
    mask: str                   # additive FP32 mask support
    arch: str                   # sm_90 (mma.sync / SIMT) or sm_90a (warpgroup products)
    strict: str                 # arithmetic under the strict compiler policy
    practical: str              # arithmetic under declared practical arithmetic
    phases: tuple               # phase kinds in installation order
    storage: tuple              # (synthetic value, lifetime) declarations
    synchronization: str
    legacy_hosts: tuple         # whole-program schedules installing it at the frozen base
    compose_hosts: tuple        # schedules installing it under execution='compose' (this lane)
    limits: tuple = field(default_factory=tuple)


BODIES = {body.name: body for body in (
    Body("fp32_tiles", "P-OPS-D 775505d7 (after Latch v0.6.0 S101H / FlashAttention-2)", "attention_lower",
         ("attention_fp32.cuh",), "fp32", "flat", (32, 64), "1 <= T <= 4096", "any broadcast",
         "sm_90", "FP32 fmaf, expf/logf, no tensor cores, fixed order", "same (no practical variant)",
         ("attention_prepare", "attention_forward", "attention_delta", "attention_backward",
          "attention_layout_restore?"),
         (("__qn/__kmax/__mmax", "forward_to_backward proof tables"), ("__lse", "forward_to_backward"),
          ("__delta/__skip", "phase (delta -> backward)")),
         "publication barrier between phases; no atomics; one writer per element",
         ("reference", "p6e", "p6d", "p7a", "p7b", "loop regions"), ("reference", "p6e", "p6d", "p7a", "p7b"),
         ("FP32 operands only; BF16-boundary groups read FP32 widened copies",)),
    Body("fp32_stored", "P-OPS2-C e7fe4b16", "attention_lower",
         ("attention_fp32.cuh", "attention_fp32_v2.cuh"), "fp32", "flat", (64,), "1024 <= T <= 4096",
         "any broadcast (shared mask tables)", "sm_90", "FP32 fmaf, fixed order; stores scores and dS",
         "same", ("attention_prepare", "attention_forward", "attention_backward", "attention_dq"),
         (("__xs", "forward_to_backward [BH][Tp][Tp] scores"), ("__ds", "phase (backward -> dq)"),
          ("__lse", "forward_to_backward")),
         "publication barrier between phases",
         ("reference", "p6e"), ("reference", "p6e", "p6d"),
         ("T x T FP32 score storage", "p6d did not include its header before this lane")),
    Body("bf16b_short", "P-OPS3-B e68b4986", "short_attention", ("attention_short.cuh",), "bf16_boundary", "flat",
         (32, 64), "1 <= T <= 128", "any broadcast", "sm_90",
         "mma.sync BF16 with exact 3-piece P/dS splits (every product equals FP32)", "same (NS=3; NS<3 undeclared)",
         ("attention_short_forward", "attention_short_backward"),
         (("__short_o32", "forward_to_backward FP32 output"), ("__short_lse", "forward_to_backward")),
         "publication barrier between forward and backward; CTA barrier between items",
         (), ("reference", "p6e", "p6d"),
         ("no flat host after integration (only fusion.py@e68b4986 hosted it)",)),
    Body("bf16b_flat_long", "P-OPS4-D v7 + P9-A key-norm epilogue", "flat_long_attention",
         ("attention_long_flat.cuh",), "bf16_boundary", "flat", (64,), "256 <= T <= 4096, T % 128 == 0",
         "any broadcast (shared mask tables)", "sm_90a", "exact3 (3-piece splits, FP32 O for delta)",
         "bf16 (one-piece P/dS, BF16 O for delta)",
         ("attention_long_prepare?", "attention_long_forward", "attention_long_dq", "attention_long_kv"),
         (("__long_kn/__long_mmax/__long_mzero/__long_rm", "forward_to_backward proof tables"),
          ("__long_lse2/__long_live", "forward_to_backward"), ("__long_o32", "forward_to_backward (exact3)"),
          ("__long_delta", "phase (dq -> kv)")),
         "publication barrier between phases; two independent warpgroup workers per CTA",
         ("p6d",), ("p6d", "reference", "p6e"),
         ("key-norm epilogue only in p6d (needs the K product's p5c writer)",)),
    Body("mixed", "P-OPS5-D 65cc7e20 + P8-E long items + P9-E v2/v3", "mixed_attention",
         ("attention_mixed.cuh",), "mixed_bf16", "flat", (32, 64), "1 <= T <= 256 (long items above 128)",
         "none (masked/causal mixed groups are refused)", "sm_90",
         "declared mixed_bf16 rounding points, exact softmax", "PMIX_FAST_SOFTMAX (P9-E practical)",
         ("attention_mixed_forward", "attention_mixed_backward", "attention_mixed_bias?"),
         (("__mixed_stats", "forward_to_backward (m, l) per row"), ("__mixed_pds", "phase BF16 P/dS (T > 128)"),
          ("P2 counters", "zeroed_counter (T > 128 stage split)"), ("__mixed_partials", "phase (bias sums)")),
         "publication barrier; long backward stage split uses per-item counters restored to zero",
         ("p7a", "p7b"), ("p7a", "p7b"),
         ("equal query/key lengths", "no mask", "forward-only refused")),
    Body("loop_short", "P-OPS4-E 65a82829 (P-OPS3-B items through the tape)", "loop_attention",
         ("attention_short.cuh",), "bf16_boundary", "loop_tape", (32, 64), "1 <= T <= 128", "any broadcast",
         "sm_90", "as bf16b_short; tape holds O (FP32) and lse instead of P", "same",
         ("attention_short_forward", "attention_short_backward"),
         (("lse, O", "loop tape slots (forward body -> reverse body)"),),
         "forward and reverse halves live in different regions; tape order",
         ("loop regions",), ("loop regions",), ("saved-body tape policy precision_remat/storage_remat",)),
    Body("loop_long", "P7-C/P8-B v2 + P9-B v4 resident items", "long_attention",
         ("attention_short.cuh", "attention_long.cuh", "attention_p9.cuh"), "bf16_boundary", "loop_tape", (64,),
         "128 < T <= 4096, T % 128 == 0 (one-piece resident T <= 512)", "any broadcast (slot mask tables)",
         "sm_90", "3-piece exact; one piece only for declared bf16_operands", "fast_exp only if declared",
         ("attention_short_forward", "attention_short_backward"),
         (("lse, O", "loop tape slots"), ("plong_mcnt/plong_delta/p9_dsbuf", "program-wide device arrays")),
         "internal monotonic in-phase grid barriers (plong_grid_sync); private CUDA module per runtime",
         ("loop regions",), ("loop regions",), ("module-global device arrays", "MT_SLOTS=32 mask slots")),
    Body("window", "P8-C attention_window + P9-C fusion", "window_attention",
         ("attention_window.cuh", "attention_window_v4.cuh"), "mixed_bf16_rect", "loop_opaque", (32,),
         "M, N <= 512, multiples of 64", "broadcast over two leading dims", "sm_90",
         "mixed_bf16 rounding points (exact)", "mixed_bf16_online (declared online arithmetic)",
         ("mattn_table", "mattn_fwd", "mattn_bwd"), (("stats [Z, M, 2]", "loop tape"),),
         "opaque operations rewritten at capture; one phase each",
         ("loop init regions",), ("loop init regions",), ("capture rewrite only inside loop.py",)),
)}


# Measured preference among compatible bodies (sources: P-OPS3-B / P-OPS4-D / P-OPS2-C results). Flat-long beats
# FP32 tiles wherever both admit; the short body replaced FP32 tiles in its lane (update 1.953 -> 1.473 ms).
PREFERENCE = {
    "bf16_boundary": ("bf16b_flat_long", "bf16b_short", "fp32_stored", "fp32_tiles"),
    "fp32": ("fp32_stored", "fp32_tiles"),
    "mixed_bf16": ("mixed",),
}


_VIEWS = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand", "squeeze",
          "unsqueeze", "slice", "select", "as_strided"}


def _layouts(values, field, operations=None):
    """Captured layouts; with operations, view outputs share their source's alias root (captured offsets)."""
    from .lower import TensorLayout, _contiguous, _ref
    roots = {}
    if operations is not None:
        names = set(values)
        for op in operations:
            target = str(field(op, "target", field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.")
            outputs = field(op, "outputs", [field(op, "name")])
            args = list(field(op, "args", []))
            source = _ref(args[0], names) if args else None
            if target.split(".")[0] in _VIEWS and source is not None and len(outputs) == 1:
                roots[outputs[0]] = roots.get(source, source)
    result = {}
    for name, value in values.items():
        shape = tuple(int(x) for x in field(value, "shape"))
        result[name] = TensorLayout(shape, tuple(int(x) for x in field(value, "strides", field(
            value, "stride", _contiguous(shape)))), roots.get(name, name), int(field(value, "storage_offset", 0) or 0))
    return result


def _graph_parts(graph):
    from .lower import _field, _ref
    raw = _field(graph, "values")
    values = raw if isinstance(raw, dict) else {_field(v, "name"): v for v in raw}
    operations = list(_field(graph, "operations", _field(graph, "nodes", [])))
    output_name = lambda op: _field(op, "outputs", [_field(op, "name")])[0]
    protected = set()

    def visit(item):
        if isinstance(item, dict):
            for x in item.values():
                visit(x)
        elif isinstance(item, (tuple, list)):
            for x in item:
                visit(x)
        else:
            r = _ref(item, set(values))
            if r is not None:
                protected.add(r)
            elif isinstance(item, str) and item in values:
                protected.add(item)
    visit(_field(graph, "outputs", ()))
    visit(_field(graph, "state_writes", {}))
    return operations, values, set(values), _ref, _field, output_name, protected


def _fp32_check(group, values, field, layouts):
    """Static form of attention_lower.emit's operand preconditions (no code emitted)."""
    from . import attention_lower as A
    dtype = lambda name: str(field(values[name], "dtype", "float32")).removeprefix("torch.")
    if any(dtype(group[k]) != "float32" for k in ("q", "kt", "v") if group.get(k)):
        return None, "FP32 operands only"
    lq, lkt, lv = layouts[group["q"]], layouts[group["kt"]], layouts[group["v"]]
    if len(lq.shape) != 3:
        return None, "rank-3 batched operands required"
    bh, t, hd = lq.shape
    if hd not in A.HEAD_WIDTHS or not 1 <= t <= A.MAX_SEQUENCE:
        return None, f"head width {hd} / context {t} outside {A.HEAD_WIDTHS} / [1, {A.MAX_SEQUENCE}]"
    kb, kd, kr = lkt.strides
    if (A._head_layout(lq, bh, t, hd) is None or A._head_layout(lv, bh, t, hd) is None
            or not (kd == 1 and all(x % 4 == 0 for x in (kb, kr, lkt.offset)))):
        return None, "Q/K/V need unit inner stride and 16-byte aligned rows"
    if group.get("mask") is not None:
        from . import padding_masks
        if padding_masks.mask_strides(group, layouts, bh, t, A._mask_strides) is None:
            return None, "mask is not broadcastable to (batch*heads, T, T)"
    return (bh, t, hd), None


def survey(graph, *, target_arch="sm_90"):
    """Every recognized attention group with every registered flat body's admission decision.

    Pure host analysis (no lowering): the same matchers the schedulers call. Returns a JSON-able dict.
    """
    from . import attention_lower, flat_long_attention, mixed_attention, short_attention
    operations, values, names, ref, field, output_name, protected = _graph_parts(graph)
    graph_args = (operations, values, names, ref, field, output_name)
    layouts = _layouts(values, field, operations)
    groups, refusals = attention_lower.plan(operations, values, names, ref=ref, field=field,
                                            output_name=output_name, protected=protected, masks=True)
    rows = []
    for group in groups:
        row = {"group": group["label"], "mask": group.get("mask"), "backward": bool(group.get("backward")),
               "candidates": {}}
        if group.get("safe") or group.get("mask_lead") is not None:
            row["padding_mask"] = {"safe_rows": bool(group.get("safe")), "mask_lead": group.get("mask_lead")}
        shape, why = _fp32_check(group, values, field, layouts)
        if shape:
            row["shape"] = dict(zip(("head_batch", "sequence", "head_width"), shape))
        row["candidates"]["fp32_tiles"] = {"admitted": shape is not None, "reason": why}
        v2 = shape is not None and attention_lower.select_variant(shape[2], shape[1], shape[0]) == "v2"
        v2_reason = None if v2 else "v2 shape rule: head width 64 and T >= 1024"
        if v2 and "padding_mask" in row:
            v2, v2_reason = False, "release step 09 padding-mask groups use the v1 tiles"
        row["candidates"]["fp32_stored"] = {"admitted": v2, "reason": v2_reason}
        plans, refused = short_attention.admit([group], graph_args, layouts, protected, cast_first=True)
        row["candidates"]["bf16b_short"] = {"admitted": bool(plans), "reason": refused[0]["reason"] if refused else None}
        long_plans, long_refused = flat_long_attention.admit([group], graph_args, layouts, protected)
        ok = bool(long_plans) and target_arch == "sm_90"
        row["candidates"]["bf16b_flat_long"] = {"admitted": ok, "reason": long_refused[0]["reason"] if long_refused
                                                else (None if ok else "requires sm_90a warpgroup target")}
        row["operand_class"] = "bf16_boundary" if (plans or long_plans) else "fp32"
        rows.append(row)
    plans, refused = mixed_attention.admit(graph_args, lambda n: (layouts[n].shape, layouts[n].strides), protected)
    for label, plan in plans.items():
        rows.append({"group": label, "operand_class": "mixed_bf16", "backward": True, "mask": None,
                     "shape": {"head_batch": plan["bh"], "sequence": plan["t"], "head_width": plan["hd"]},
                     "candidates": {"mixed": {"admitted": True, "reason": None,
                                              "engine": "long" if mixed_attention.is_long(plan["t"]) else "short"}}})
    for refusal in refused:
        rows.append({"group": refusal["group"], "operand_class": "mixed_bf16", "candidates": {
            "mixed": {"admitted": False, "reason": refusal["reason"]}}})
    has_bf16 = any(str(field(v, "dtype", "float32")).removeprefix("torch.") == "bfloat16" for v in values.values())
    for row in rows:
        row["selected"] = select(row, has_bf16=has_bf16)
    return {"version": VERSION, "groups": rows, "unmatched_refusals": refusals, "program_has_bf16": has_bf16}


# P-OPS2-A measured rule kept by the reference/P6-E/P7 hosts: FP32 programs (no BF16 value) with at most this many
# keys run materialized FP32 product bodies plus softmax phases instead of fused FP32 tiles.
FP32_MATERIALIZED_KEYS = 512


def select(row, *, has_bf16=True):
    """First admitted body in the measured preference of the group's operand class (None: ordinary lowering).

    ``has_bf16`` is whether the program holds any BF16 value (the hosts apply the FP32 materialization rule only to
    all-FP32 programs).
    """
    sequence = (row.get("shape") or {}).get("sequence")
    if (row.get("operand_class") == "fp32" and not has_bf16 and sequence is not None
            and sequence <= FP32_MATERIALIZED_KEYS):
        return "fp32_products"
    for name in PREFERENCE.get(row.get("operand_class"), ()):
        if row["candidates"].get(name, {}).get("admitted"):
            return name
    return None


def legacy_choice(row, schedule, *, has_bf16=True):
    """What the frozen whole-program schedule installs for this group (for audit/A-B records)."""
    admitted = lambda name: row["candidates"].get(name, {}).get("admitted")
    sequence = (row.get("shape") or {}).get("sequence")
    if (not has_bf16 and schedule in ("reference", "p6e") and sequence is not None
            and sequence <= FP32_MATERIALIZED_KEYS):
        return "fp32_products"
    if row.get("operand_class") == "mixed_bf16":
        return "mixed" if schedule in ("p7a", "p7b") and admitted("mixed") else None
    if schedule == "p6d" and admitted("bf16b_flat_long"):
        return "bf16b_flat_long"
    if admitted("fp32_tiles"):
        return "fp32_stored" if admitted("fp32_stored") and schedule in ("reference", "p6e") else "fp32_tiles"
    return None


# ----------------------------------------------------------------------------------------- phase contracts
def _outputs(extra, values, field, group_label):
    """{name: (shape, dtype, lifetime)} of one emitted phase; None specs are the captured BF16 finals."""
    result = {}
    for name, spec in extra["outputs"].items():
        if spec is None:
            result[name] = (tuple(int(x) for x in field(values[name], "shape")), "bfloat16", "captured_final")
        else:
            result[name] = (tuple(spec[0]), spec[1], "synthetic")
    return result


def _lifetimes(phases):
    """Classify synthetic storage by its producing and last consuming phase."""
    produced, consumed = {}, {}
    for index, phase in enumerate(phases):
        for name in phase["writes"]:
            produced.setdefault(name, index)
        for name in phase["reads"]:
            consumed[name] = index
    for index, phase in enumerate(phases):
        storage = {}
        for name, (shape, dtype, origin) in phase["outputs"].items():
            if origin == "captured_final":
                lifetime = "captured_final"
            elif name not in consumed:
                lifetime = "phase"
            else:
                first = phases[produced[name]]["role"]
                last = phases[consumed[name]]["role"]
                lifetime = "forward_to_backward" if first == "forward" and last == "backward" else "phase"
            storage[name] = {"shape": list(shape), "dtype": dtype, "lifetime": lifetime}
        phase["storage"] = storage
    return phases


def emit_phases(body, plan, *, layouts, dtypes, values, field, ident, mask_strides=None, practical=False,
                version=None, keynorm_epilogue=False, mask_tables=None):
    """Execution-contract phase dicts of one admitted group (forward then backward), in dependency order.

    The code strings are the bodies' own emissions; this adds explicit headers, defines, resource and storage
    declarations. Hosts must give each phase a publication barrier after it unless the contract planner proves
    the next phase independent (``ops.execution_plan.contracts``).
    """
    if body == "bf16b_short":
        from . import short_attention as S
        phases = []
        for which in ("forward", "backward"):
            kind, code, extra = S.emit(plan, layouts, ident, mask_strides, which=which)
            used = ("q", "k", "v") if which == "forward" else ("q", "k", "v", "do")
            reads = [plan["sources"][key] for key in used]
            if which == "backward":
                aux = S.aux_names(plan["group"])
                reads += [aux["o32"], aux["lse"]]
            if plan["group"].get("mask") is not None:
                reads.append(plan["group"]["mask"])
            phases.append(_phase(body, which, kind, code, extra, reads, values, field, plan["group"]["label"],
                                 headers=(S.HEADER,), defines={}, arithmetic=f"short splits={plan['splits']}"))
        return _lifetimes(phases)
    if body == "bf16b_flat_long":
        from . import flat_long_attention as L
        recipe = "bf16" if practical else L.DEFAULT_RECIPE
        version = L.DEFAULT_VERSION if version is None else int(version)
        if version in L.BF16_ONLY_VERSIONS and L.RECIPES[recipe][0] != 1:
            version = 7
        defines = _parse_defines(L.defines(recipe, version))
        phases = []
        for which in ("prepare", "forward", *L.backward_phases(version)):
            emitted = L.emit(plan, ident, mask_strides, which=which, recipe=recipe, version=version,
                             mask_tables=mask_tables, keynorm_epilogue=keynorm_epilogue)
            if emitted is None:
                continue
            kind, code, extra = emitted
            role = "forward" if which in ("prepare", "forward") else "backward"
            phases.append(_phase(body, role, kind, code, extra, extra["reads"], values, field,
                                 plan["group"]["label"], headers=(L.HEADER,), defines=defines,
                                 arithmetic=f"flat long recipe={recipe} v{version}", arch="sm_90a"))
        return _lifetimes(phases)
    if body in ("fp32_tiles", "fp32_stored"):
        from . import attention_lower as A
        group = dict(plan, v2_enabled=body == "fp32_stored")   # the plan is attention_lower's group dict
        phases, defined = [], {}

        def define(name, shape, root=None, strides=None):
            defined[name] = (tuple(shape), root, strides)

        def add_phase(kind, code, metadata, reads, writes):
            role = "forward" if kind in ("attention_prepare", "attention_forward") else "backward"
            extra = dict(metadata, outputs={w: (defined[w][0], "float32") for w in writes})
            headers = (A.HEADER,) + ((A.HEADER_V2,) if metadata.get("variant") == "v2" else ())
            phases.append(_phase(body, role, kind, code, extra, reads, values, field, group["label"],
                                 headers=headers, defines={}, arithmetic="FP32 SIMT exact order"))
        anchors = (group["anchor_forward"], group.get("anchor_backward"), group.get("last_backward"))
        for name in dict.fromkeys(a for a in anchors if a is not None):
            A.emit_member(group, name, layouts, dtypes, values, field=field, ident=ident, define=define,
                          add_phase=add_phase, add_view=lambda *a: None, record=lambda r: None)
        return _lifetimes(phases)
    if body == "mixed":
        from . import mixed_attention as M
        phases = []
        for which in ("forward", "backward"):
            kind, code, extra = M.emit(plan, ident, dtypes.__getitem__, which=which)
            reads = list(plan["sources"].values()) if which == "backward" else [
                plan["sources"][k] for k in ("q", "k", "v")]
            reads += list(plan["biases"].values()) + ([M.stats_name(plan["group"])] if which == "backward" else [])
            reads += extra.get("aux_reads", [])
            phase = _phase(body, which, kind, code, extra, reads, values, field, plan["group"]["label"],
                           headers=(M.HEADER,), defines={"PMIX_LONG_V2": int(M.LONG_V2), "PMIX_LONG_V3": int(M.LONG_V3),
                                                         "PMIX_LONG_FWD_V2": int(M.LONG_FWD_V2),
                                                         "PMIX_FA_DELTA": int(M.FA_DELTA)},
                           arithmetic="mixed_bf16 declared rounding points")
            if extra.get("p2_split"):
                phase["zeroed_counters"] = dict(extra["p2_split"], placeholders=["/*P2_CNT*/", "/*P2_PART*/"],
                                                contract="zero at entry; the key stage restores zero")
            phases.append(phase)
            if extra.get("bias_phase"):
                bias_code, bias_extra = extra["bias_phase"]
                phases.append(_phase(body, "backward", "attention_mixed_bias", bias_code,
                                     dict(bias_extra, outputs={k: None for k in bias_extra["outputs"]}),
                                     bias_extra["aux_reads"], values, field, plan["group"]["label"],
                                     headers=(M.HEADER,), defines={}, arithmetic="fixed-tree bias sums"))
        return _lifetimes(phases)
    raise ValueError(f"no flat phase emitter for body {body!r} (loop bodies are installed per region)")


def _parse_defines(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "#define":
            result[parts[1]] = parts[2] if len(parts) > 2 else ""
    return result


def _phase(body, role, kind, code, extra, reads, values, field, label, *, headers, defines, arithmetic, arch="sm_90"):
    shared = int(extra.get("shared_bytes", 0) or 0)
    if not 0 <= shared <= MAX_SHARED:
        raise ValueError(f"{label}: {kind} needs {shared} shared bytes")
    outputs = _outputs(extra, values, field, label)
    return {"kind": kind, "engine": body, "role": role, "group": label, "code": code,
            "reads": list(dict.fromkeys(reads)), "writes": list(outputs), "outputs": outputs,
            "headers": list(headers), "defines": dict(defines), "threads": THREADS, "shared_bytes": shared,
            "work_items": int(extra.get("work_items", 0)), "arch": arch,
            "internal_grid_barrier": "grid.sync(" in code, "arithmetic": arithmetic,
            "layouts": {k: extra[k] for k in ("head_width", "sequence", "head_batch", "heads") if k in extra}}


def contract(phases, layouts, dtypes):
    """Validate the phases against the shared execution contract (RAW/WAR/WAW, resources)."""
    from .execution_plan import contracts
    from .lower import TensorLayout, _contiguous
    layouts = dict(layouts)
    dtypes = dict(dtypes)
    for phase in phases:
        for name, (shape, dtype, _) in phase["outputs"].items():
            layouts.setdefault(name, TensorLayout(tuple(shape), _contiguous(tuple(shape)), name))
            dtypes.setdefault(name, dtype)
    return contracts(phases, layouts, dtypes)




def registry():
    """JSON-able static contracts of every registered body."""
    return {"version": VERSION, "bodies": {name: asdict(body) for name, body in BODIES.items()},
            "portable_source_commit": "1ad3e78", "common_phase_intake": ["bf16b_short"],
            "preference": {k: list(v) for k, v in PREFERENCE.items()},
            "selection_rules": [f"all-FP32 programs with <= {FP32_MATERIALIZED_KEYS} keys: materialized FP32 products "
                                "(fp32_products, P-OPS2-A measured rule; not a fused body)",
                                "mixed_bf16 groups are installed only by the P7-A/P7-B hosts",
                                "loop-region bodies are selected per region by the tape rewrite (loop_attention)"]}
