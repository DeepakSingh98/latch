"""One structural selector for mutually exclusive frozen flat execution protocols.

No workload names, parameter names, model classes or benchmark sizes participate.
The full effect schedule is chosen once; schedules are never applied successively.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import threading
from contextlib import contextmanager

_LOCK = threading.RLock()
PINS = {
    "reference": "af3138238f1b2b84fa17a5b0a71884fe80c1dea0",
    "p6d": "81a52fc02783e0c537dcf1a909083916f2deaa66",
    "p6e": "5f78a0ef1d044fccb5813bea5329b77901bdcab8",
    "p7b": "f6f60bd1325cae4ded5fba17083feb3b85a791b0",
    "p7a": "f36b4737cee732e6915d11e233bcc068126b3e64",
}
INTEGRATED_INPUTS = {
    "ordinary": "2d8350f0fbe1029da6d10a7d706e47940e716ead",
    "recurrence": "5398776d40df99ed74b862235f85e4f1274e4724",
    "adaptive_memory": "cf127fc03de79bc3b2b4c02eb69e1f9ad71672c5",
    "energy": "585e7ad5cb354695ff4bfa496f8c061931b6656b",
    "diffusion": "5de9dd06a2790e1fa79fcdbf323c02eddfd4cb5c",
}


def _graph_parts(graph):
    from .lower import TensorLayout, _contiguous, _field, _ref
    raw = _field(graph, "values")
    values = raw if isinstance(raw, dict) else {_field(v, "name"): v for v in raw}
    names = set(values)
    operations = _field(graph, "operations", _field(graph, "nodes", []))
    output_name = lambda op: _field(op, "outputs", [_field(op, "name")])[0]
    layouts = {}
    for name, value in values.items():
        shape = tuple(_field(value, "shape"))
        layouts[name] = TensorLayout(shape, tuple(_field(value, "strides", _contiguous(shape))), name)
    protected = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            ref = _ref(value, names)
            if ref is not None:
                protected.add(ref)
            elif isinstance(value, str) and value in names:
                protected.add(value)
    visit(_field(graph, "outputs", ()))
    visit(_field(graph, "state_writes", {}))
    return (operations, values, names, _ref, _field, output_name), layouts, protected


def select_schedule(graph, *, target_arch="sm_90", retain_all=False, fuse_attention=True):
    """Return an admitted schedule and auditable structural reasons, without lowering."""
    from .lower import _field
    if target_arch != "sm_90" or retain_all:
        return "reference", ["frozen resource protocols require sm_90 and packed storage"]
    parts, layouts, protected = _graph_parts(graph)
    operations, values, names, ref, field, output_name = parts
    # Registration names come from the primitive module, never from a model.
    from .scan import TARGETS
    if any(str(_field(op, "target", _field(op, "op"))) in TARGETS for op in operations):
        return "p6e", ["selective-scan primitive requires the one-CTA scan/staging protocol"]
    if not fuse_attention:
        return "reference", ["attention fusion explicitly disabled"]
    from . import attention_lower, flat_long_attention, mixed_attention
    groups, refused = attention_lower.plan(operations, values, names, ref=ref, field=field,
        output_name=output_name, protected=protected, exclude=attention_lower.excluded_groups(), masks=True)
    long, long_refused = flat_long_attention.admit(groups, parts, layouts, protected)
    if long:
        return "p6d", ["matched BF16-boundary head64 attention with admitted long-context layouts",
                        "one-CTA WGMMA schedule; FP32 probability exact3 arithmetic"]
    # Admission uses the same shared-memory mode as the selected B protocol.
    # A context tail alone does not distinguish one- and two-CTA execution.
    from .performance_defaults import P7B_SELECTED
    limit = int(P7B_SELECTED["matrix_rule_overrides"]["mixed_attention.BWD_FIT_LIMIT"])
    with _LOCK, _configuration("reference", {"mixed_attention.BWD_FIT_LIMIT": limit}):
        short, short_refused = mixed_attention.admit(parts,
            lambda name: (layouts[name].shape, layouts[name].strides), protected,
            exclude=attention_lower.excluded_groups())
        requirements = [(mixed_attention.forward_shared(p["hd"], p["t"]),
                         mixed_attention.backward_shared(p["hd"], p["t"])) for p in short.values()]
    if short and all(max(pair) <= limit for pair in requirements):
        return "p7b", ["matched mixed BF16/FP32 attention whose forward and backward fit two CTAs",
                        f"attention shared requirements {requirements}; ceiling {limit} bytes",
                        "final whole-program shared-memory proof remains mandatory"]
    if short:
        with _LOCK, _configuration("reference", {"mixed_attention.BWD_FIT_LIMIT": None}):
            one, one_refused = mixed_attention.admit(parts,
                lambda name: (layouts[name].shape, layouts[name].strides), protected,
                exclude=attention_lower.excluded_groups())
        if set(short).issubset(one):
            return "p7a", ["matched mixed attention exceeds the two-CTA budget but admits the one-CTA protocol",
                            f"two-CTA attention shared requirements {requirements}; ceiling {limit} bytes"]
    # Pure matrix/elementwise graphs need not manufacture an attention group to
    # reach the same admitted engine. Preserve the generic schedule for an
    # unmatched attention/batched-product graph: its resource protocol has not
    # been proved by the attention matchers above.
    targets = [str(field(op, "target", field(op, "op"))) for op in operations]
    if not any("bmm" in target or "softmax" in target or "scaled_dot_product" in target
               for target in targets):
        from .lower import _dtype
        from .matrix_codegen import select
        with _LOCK, _configuration("reference", {}):
            for op, target in zip(operations, targets):
                if target != "aten.mm.default":
                    continue
                args, out = field(op, "args", ()), output_name(op)
                if len(args) < 2:
                    continue
                a, b = ref(args[0], names), ref(args[1], names)
                if (a is None or b is None or any(_dtype(values[x]) != "bfloat16" for x in (a, b, out))
                        or any(len(layouts[x].shape) != 2 for x in (a, b, out))
                        or any(int(field(values[x], "storage_offset", 0)) != 0 for x in (a, b))):
                    continue
                spec, reason = select(a, b, out, layouts[a], layouts[b], layouts[out].shape,
                                      "bfloat16", arch=target_arch, engine_ok=True)
                if spec and spec.get("engine") == "p5c":
                    return "p6d", ["rank2 BF16 product passes the shared warpgroup tile/layout/resource admission",
                                    reason, "no unmatched attention or batched-product protocol in the graph"]
    return "reference", ["no admitted scan or BF16-boundary attention protocol; generic dependency schedule"]


@contextmanager
def _configuration(schedule, overrides):
    """Restore module settings on every exit; serialize concurrent host lowerings."""
    names = ["transformer_lower", "p5c_matrix", "p6d_rules", "pops5e_rules",
             "mixed_attention", "norm_rules", "p7e_codegen", "mixed_gemm", "resources",
             "flat_long_attention", "p9_rowblock"]
    names += ["fusion_" + schedule]
    if schedule == "p7a":
        names += ["norm_rules_p7a", "pops4a_rules_p7a"]
    if schedule != "reference":
        names += ["matrix_rules_" + schedule]
        if schedule != "p6d":
            names += ["pops2b_rules_" + schedule]
    modules = {name: importlib.import_module("." + name, __package__) for name in names}
    from . import lower as lower_module
    modules["lower"] = lower_module
    snapshots = {name: {k: copy.deepcopy(v) for k, v in vars(module).items()
                        if k.isupper() and isinstance(v, (str, int, float, bool, tuple, list, dict, set, type(None)))}
                 for name, module in modules.items()}
    try:
        for key, value in overrides.items():
            if "." in key:
                module_name, attribute = key.split(".", 1)
                if module_name in ("matrix_rules", "pops2b_rules"):
                    module_name += "_" + schedule
            else:
                module_name, attribute = "matrix_rules_" + schedule, key
            module = modules.get(module_name)
            if module is None or not hasattr(module, attribute):
                raise ValueError(f"unknown performance setting {key!r} for {schedule}")
            setattr(module, attribute, copy.deepcopy(value))
        yield
    finally:
        for name, module in modules.items():
            for key in list(vars(module)):
                if key.isupper() and key not in snapshots[name] and isinstance(getattr(module, key),
                        (str, int, float, bool, tuple, list, dict, set, type(None))):
                    delattr(module, key)
            for key, value in snapshots[name].items():
                setattr(module, key, value)


def lower_selected(graph, *, schedule="auto", schedule_options=None, compact_codegen="off", precision_policy=None, **options):
    from .lower import LoweringError, _validate_flat_precision_policy
    from .performance_defaults import P6E_SELECTED, P6E_MATRIX, P7B_SELECTED, P7B_STRICT_SELECTED, P7A_SELECTED
    if schedule not in ("auto", *PINS):
        raise LoweringError(f"unknown flat performance schedule {schedule!r}")
    if compact_codegen not in ("off", "fold"):
        raise LoweringError("compact_codegen must be 'off' or 'fold'")
    schedule_options = dict(schedule_options or {})
    aliases = [v for v in (options.pop("p8_precision", None), schedule_options.pop("p8_precision", None)) if v is not None]
    if any(v not in ("strict", "practical") for v in aliases):
        raise LoweringError("legacy p8_precision must be strict or practical")
    if precision_policy not in (None, "strict", "practical"):
        raise LoweringError("precision_policy must be strict or practical")
    declared_policy = precision_policy or (aliases[0] if aliases else "strict")
    if any(v != declared_policy for v in aliases):
        raise LoweringError("conflicting precision_policy and legacy p8_precision declarations")
    requested = schedule
    with _LOCK:
        if schedule == "auto":
            schedule, reasons = select_schedule(graph, target_arch=options.get("target_arch", "sm_90"),
                retain_all=options.get("retain_all", False), fuse_attention=options.get("fuse_attention", True))
        else:
            reasons = ["explicit compiler schedule override"]
        defaults = copy.deepcopy(P6E_SELECTED if schedule == "p6e" else (P7B_SELECTED if declared_policy == "practical" else P7B_STRICT_SELECTED) if schedule == "p7b" else P7A_SELECTED if schedule == "p7a" else {})
        overrides = defaults.pop("matrix_rule_overrides", {})
        overrides.update(defaults.pop("matrix_overrides", {}))
        if schedule == "p6e":
            overrides.update(P6E_MATRIX)
            defaults["p6e_fast_act"] = declared_policy == "practical"
        if schedule == "p6d":
            from .p5c_matrix import FLAT_SELECTED_DEFAULTS
            overrides.update({"p5c_matrix." + key: value for key, value in FLAT_SELECTED_DEFAULTS.items()})
        selected_options = dict(defaults)
        selected_options.update(options)
        selected_options.update(dict(schedule_options or {}))
        if schedule == "p6d":
            selected_options["p8_precision"] = declared_policy
        if declared_policy == "practical" and schedule not in ("p6d", "p7b", "p6e"):
            raise LoweringError("practical flat arithmetic requires a compatible matrix, mixed-attention or scan protocol")
        if isinstance(defaults.get("pops5d_defines"), dict):   # P9-E: requested defines extend the selected ones
            selected_options["pops5d_defines"] = {**defaults["pops5d_defines"],
                                                  **(options.get("pops5d_defines") or {}),
                                                  **((schedule_options or {}).get("pops5d_defines") or {})}
        if schedule == "p7a" and selected_options.get("p6a_two_cta"):
            raise LoweringError("P7-A two-CTA diagnostic is outside the accepted v13 resource protocol")
        overrides.update(selected_options.pop("matrix_rule_overrides", {}))
        overrides.update(selected_options.pop("matrix_overrides", {}))
        # Validate after all supported option/define maps have merged, before
        # any module mutation or the catchable automatic schedule fallback.
        _validate_flat_precision_policy(declared_policy, selected_options, overrides)
        planned_grid = selected_options.pop("p6a_grid", None)
        late_remat = selected_options.pop("remat", False) if schedule == "p6d" else False
        if late_remat:
            # Step 15: an unknown policy is a caller error, not a reason for the catchable schedule fallback.
            from .temporary_memory import remat_admission
            remat_admission(late_remat)
        if planned_grid is not None:
            overrides.update(TARGET_GRID=int(planned_grid), **{
                "norm_rules_p7a.GRID_CAP": int(planned_grid),
                "fusion_p7a.GRID_MODEL": int(planned_grid),
                "fusion_p7a.RESIDENT_THREADS": int(planned_grid) * 256})
        module = importlib.import_module(".fusion_" + schedule, __package__)
        accepted = inspect.signature(module.lower_fused).parameters
        ignored = sorted(set(selected_options) - set(accepted))
        # Shared options absent from an older frozen scheduler have no analogue;
        # record them explicitly rather than mutating that protocol indirectly.
        filtered = {k: v for k, v in selected_options.items() if k in accepted}
        try:
            with _configuration(schedule, overrides):
                graph_for_lower = graph
                cse_report, remat_report = None, None
                if schedule == "p6d":
                    from .p6d_rules import cse
                    graph_for_lower, cse_report = cse(graph_for_lower)
                    # LATCH-DENSE-EXEC: readers of a BF16-exact FP32 value read its materialized BF16 copy (exact).
                    from .bf16_exact import reuse as bf16_exact_reuse
                    graph_for_lower, exact_report = bf16_exact_reuse(graph_for_lower)
                    if late_remat:
                        from .remat import plan
                        from .temporary_memory import remat_admission
                        graph_for_lower, remat_report = plan(graph_for_lower, admit=remat_admission(late_remat))
                program = module.lower_fused(graph_for_lower, **filtered)
                if schedule == "p6d":
                    from .p5c_matrix import finalize
                    selection_contracts = getattr(program, "_selection_contracts", None)
                    program = finalize(program)
                    if cse_report["merged"]:
                        program.manifest["p6d_cse"] = cse_report
                    if exact_report["rewritten"]:
                        program.manifest["bf16_exact_reuse"] = exact_report
                    if remat_report is not None:
                        program.manifest["late_operand_remat"] = {k: v for k, v in remat_report.items() if k != "candidates"}
                        program.manifest["late_operand_remat"]["values"] = sorted(remat_report["candidates"])
                if schedule == "p7b" and int(filtered.get("matrix_residency", 1)) == 2:
                    resource = program.manifest.get("resource_policy", {})
                    if resource.get("resident_blocks_per_sm") != 2:
                        raise LoweringError("selected two-CTA protocol exceeds whole-program resources")
                if compact_codegen == "fold" and schedule == "p6d":
                    from . import p7e_codegen
                    p7e_codegen.P7E_MODE = "fold"
                    p7e_codegen.P7E_OUTLINE = True
                    p7e_codegen.P7E_OUTLINE_KEEP = ("p5c::", "p8ws::", "pflatlong::", "plong::", "pops4b::", "p7a4b::", "p6e::", "grid.sync()")
                    # The integrated long-attention body is not safe with
                    # dynamic shared-body pointer slots. Keep its constant
                    # bindings; sharing the other admitted bodies is qualified
                    # separately. An explicit diagnostic cannot drop this guard.
                    p7e_codegen.P7E_NOSHARE = tuple(dict.fromkeys(
                        (*p7e_codegen.P7E_NOSHARE, "pflatlong::")))
                    program = p7e_codegen.apply(program)
                    program.manifest["p7e"]["integration_safety"] = {
                        "unshared_body_markers": ["pflatlong::"],
                        "reason": "shared long-attention bodies failed integrated backward qualification; constant-binding bodies passed",
                        "evidence_attempt": "latch-ops-int3-ordinary-current-gpu",
                        "rejected_plain_fold": True,
                        "default_compact_mode": "off",
                    }
                if schedule == "p6d" and selection_contracts is not None:
                    # Finalization/compaction may create a new program. Keep
                    # selection evidence ephemeral and tied to the actual IR
                    # consumed by the unchanged emitter.
                    program._selection_contracts = selection_contracts
                    if "p6d" in selection_contracts:
                        selection_contracts["p6d"].update(
                            lowered_graph=graph_for_lower, cse_report=copy.deepcopy(cse_report))
        except LoweringError as error:
            forced_scan = selected_options.get("scan_engine", "auto") not in ("auto", "generic")
            if requested != "auto" or schedule == "reference" or forced_scan:
                raise
            refused_schedule = schedule
            schedule = "reference"
            planned_grid = None
            if refused_schedule == "p6e":
                from .lower import _lower_serial
                accepted = inspect.signature(_lower_serial).parameters
                filtered = {k: v for k, v in options.items() if k in accepted}
                filtered["scan_engine"] = "generic"
                ignored = sorted(set(options) - set(accepted))
                reasons.append(f"scan schedule refused: {error}; ordered generic scan lowering")
                program = _lower_serial(graph, **filtered)
            else:
                from . import fusion_reference
                reasons.append(f"admitted protocol lowering refused: {error}; generic dependency schedule")
                accepted = inspect.signature(fusion_reference.lower_fused).parameters
                filtered = {k: v for k, v in options.items() if k in accepted}
                ignored = sorted(set(options) - set(accepted))
                program = fusion_reference.lower_fused(graph, **filtered)
        if compact_codegen == "fold" and schedule != "p6d":
            program.manifest["p7e"] = {"mode": "fold", "applied": False,
                "reason": "compact generation is admitted only for the P6D flat protocol"}
        if schedule in ("p6d", "p7a"):
            program.manifest["full_grid"] = True
        if planned_grid is not None:
            program.manifest["planned_grid"] = int(planned_grid)
        program.manifest["performance_schedule"] = {
            "version": 1, "requested": requested, "selected": schedule, "source_commit": PINS[schedule],
            "source_commit_role": "preserved protocol foundation; integrated changes identified separately",
            "integrated_inputs": dict(INTEGRATED_INPUTS),
            "reasons": reasons, "options": filtered, "unconsumed_shared_options": ignored,
            "compact_codegen": compact_codegen,
            "precision_policy": declared_policy,
            "selection": "operation semantics, layout, precision and whole-program resources; no model names",
            "qualification": "combined compiler requires independent numerical and runtime measurement",
        }
        return program
