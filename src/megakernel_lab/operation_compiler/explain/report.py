"""Structured compilation report built from existing Latch manifests (LATCH-EXPLAIN).

The report is a pure function of what the compiler already wrote (``manifest.json``,
``program.cu``, ``artifact.json``), the captured IR, optional collector records and an
optional measured profile. It changes nothing about lowering. Every quantity carries a
label: ``measured`` (from a supplied profile/receipt), ``static`` (a count read from the
plan), ``estimate`` (the compiler's own planning model) or ``unknown``. Rejection
reasons are quoted verbatim from the records that hold them; when no record exists the
report says so rather than inferring one. No bottleneck is inferred from operation
counts: slow-path candidates are ranked only by measured time.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from . import profile as profile_module

REPORT_SCHEMA = "latch-explain-report-v1"

_ITEM_BYTES = {"float32": 4, "float": 4, "bfloat16": 2, "float16": 2, "half": 2, "int64": 8, "long": 8,
               "int32": 4, "int": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1, "float64": 8, "double": 8}
_CAST_TARGETS = ("aten._to_copy.default", "prims.convert_element_type.default")
_COPY_TARGETS = ("aten.clone.default", "aten.copy.default", "aten.copy_.default", "aten.contiguous.default")
_MATRIX_KINDS = {"gemm", "splitk_reduce", "matrix_split_reduce", "matrix_scratch_init"}
_ATTENTION_PREFIX = "attention"
_REDUCTION_KINDS = {"reduction", "reduction_counter_init", "log_softmax", "softmax"}
_INDEXED_KINDS = {"embedding", "embedding_gradient", "gather", "scatter", "index_stable_merge",
                  "indexed_accumulation"}


# --------------------------------------------------------------------------------------------- helpers
def q(value, label, basis=None, unit=None):
    """A labelled quantity. label in {measured, static, estimate, unknown}."""
    assert label in ("measured", "static", "estimate", "unknown"), label
    row = {"value": value, "label": label}
    if unit:
        row["unit"] = unit
    if basis:
        row["basis"] = basis
    return row


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _field(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _numel(shape):
    try:
        return int(math.prod(int(x) for x in shape))
    except Exception:
        return None


def _bytes(layout):
    if not isinstance(layout, dict):
        return None
    n = _numel(layout.get("shape") or ())
    size = _ITEM_BYTES.get(str(layout.get("dtype", "")).removeprefix("torch."))
    return None if n is None or size is None else n * size


def ir_to_dict(ir):
    """Accept a TrainingIR, a dict from ``TrainingIR.to_dict`` or None."""
    if ir is None or isinstance(ir, dict):
        return ir
    if hasattr(ir, "to_dict"):
        return ir.to_dict()
    raise TypeError("ir must be a TrainingIR, its to_dict() dict or None")


class _Graph:
    """Read-only view of the captured IR for joining operations to the plan."""

    def __init__(self, ir):
        self.ops = {}
        self.values = {}
        self.metadata = {}
        if not ir:
            return
        for op in ir.get("operations", []):
            self.ops[op["name"]] = op
        for name, value in ir.get("values", {}).items():
            self.values[name] = value
        self.metadata = ir.get("metadata", {}) or {}

    @property
    def present(self):
        return bool(self.ops)

    def dtype(self, name):
        value = self.values.get(name)
        return None if value is None else str(value.get("dtype", "")).removeprefix("torch.")

    def op(self, name):
        return self.ops.get(name)

    def locate(self, name):
        """Optional source location recorded by the opt-in capture hook."""
        return (self.metadata.get("source_locations") or {}).get(name)


# --------------------------------------------------------------------------------------------- phases
def _phase_ops(phase):
    """(name, target) pairs for a phase; targets are parallel to names when the emitter kept them."""
    names = _as_list(phase.get("names"))
    targets = _as_list(phase.get("targets"))
    if len(targets) == len(names):
        return list(zip(names, targets))
    items = phase.get("items")
    if isinstance(items, list):
        pairs = []
        for item in items:
            for n in _as_list(item.get("names") or item.get("name")):
                pairs.append((n, item.get("target")))
        if pairs:
            return pairs
    return [(n, None) for n in names]


def _operation_identity(name, target, layouts, graph):
    """Unambiguous identity for an operation: captured name, aten target, result shape/dtype, source if known."""
    layout = layouts.get(name) or {}
    op = graph.op(name)
    row = {"operation": name, "target": target or (op or {}).get("target"),
           "shape": layout.get("shape"), "dtype": layout.get("dtype")}
    if op is not None:
        row["inputs"] = list(op.get("inputs", ()))
    location = graph.locate(name)
    if location:
        row["source"] = location
    return row


def _walk(manifest, path="program"):
    """Yield (path, node) for the root and every embedded region/branch manifest."""
    yield path, manifest
    regions = manifest.get("regions")
    if isinstance(regions, dict):
        for name, child in regions.items():
            if isinstance(child, dict):
                yield from _walk(child, f"{path}/{name}")
    branches = manifest.get("branches")
    if isinstance(branches, dict):
        for name, child in branches.items():
            if isinstance(child, dict) and ("phases" in child or "regions" in child):
                yield from _walk(child, f"{path}/branch:{name}")


# --------------------------------------------------------------------------------------------- node analysis
def _node_summary(path, node):
    phases = [p for p in _as_list(node.get("phases")) if isinstance(p, dict)]
    kinds = {}
    for p in phases:
        kinds[p.get("kind", "?")] = kinds.get(p.get("kind", "?"), 0) + 1
    selection = node.get("execution_selection") or {}
    barrier = node.get("barrier_count")
    row = {"path": path, "format": node.get("format"), "schedule": node.get("schedule"),
           "phase_count": q(len(phases), "static"), "phase_kinds": kinds,
           "barrier_count": q(barrier, "static", "manifest.barrier_count") if barrier is not None else q(None, "unknown"),
           "barrier_schedule": node.get("barrier_schedule"),
           "arena_bytes": q(node.get("arena_bytes"), "static", "manifest.arena_bytes") if node.get("arena_bytes") is not None else q(None, "unknown"),
           "shared_bytes": q(node.get("shared_bytes"), "static") if node.get("shared_bytes") is not None else None,
           "execution_selected": selection.get("selected"), "execution_requested": selection.get("requested"),
           "execution_reasons": list(selection.get("reasons") or []),
           "compiler_selection": {k: (node.get("compiler_selection") or {}).get(k)
                                  for k in ("policy", "schedule", "reasons", "dispatch")},
           "execution_statement": node.get("execution")}
    if node.get("runtime_loop"):
        row["runtime_loop"] = node["runtime_loop"]
    if node.get("loop_schedule_policy"):
        pol = node["loop_schedule_policy"]
        row["loop_schedule_policy"] = {k: pol.get(k) for k in ("selected", "reasons", "requested_options")}
    if node.get("compiled_regions"):
        row["compiled_regions"] = list(node["compiled_regions"])
    if node.get("predicate_input"):
        row["predicate_input"] = node["predicate_input"]
        row["conditional_policy"] = node.get("conditional_policy")
    if node.get("performance_schedule"):
        ps = node["performance_schedule"]
        row["performance_schedule"] = {k: ps.get(k) for k in ("requested", "selected", "reasons", "precision_policy")}
    if node.get("dynamic_batch"):
        row["dynamic_batch"] = node["dynamic_batch"]
    return row


def _matrix_rows(path, node, graph):
    layouts = node.get("layouts") or {}
    phases = [p for p in _as_list(node.get("phases")) if isinstance(p, dict)]
    decisions = {d.get("output"): d for d in _as_list(node.get("matrix_decisions")) if isinstance(d, dict)}
    provider = {d.get("output"): d for d in _as_list((node.get("latch_matrix") or {}).get("decisions"))
                if isinstance(d, dict)}
    rows = []
    seen = set()
    for index, phase in enumerate(phases):
        if phase.get("kind") not in _MATRIX_KINDS and not str(phase.get("engine", "")).startswith(
                ("p5c", "pops", "popsc", "latch_matrix", "generic_simt", "simt", "mma")):
            continue
        outputs = [n for n, t in _phase_ops(phase) if t is None or "mm" in str(t) or "bmm" in str(t)
                   or "addmm" in str(t) or "conv" in str(t)]
        spec = phase.get("matrix_spec") or {}
        plan = phase.get("matrix_plan") or {}
        for out in outputs or _as_list(phase.get("names"))[:1]:
            seen.add(out)
            decision = decisions.get(out) or {}
            prov = provider.get(out) or {}
            row = {"region": path, "phase": index, "output": out,
                   "identity": _operation_identity(out, dict(_phase_ops(phase)).get(out), layouts, graph),
                   "engine": phase.get("engine") or spec.get("engine"),
                   "matrix_shape": (phase.get("matrix_shape") or ([spec.get(k) for k in ("m", "n", "k")] if spec.get("m") else None)
                                    or {"result_shape": (layouts.get(out) or {}).get("shape"), "basis": "no matrix_shape recorded; result layout only"}),
                   "batch": phase.get("batch", spec.get("batch")),
                   "dtype": spec.get("dtype") or (layouts.get(out) or {}).get("dtype"),
                   "arithmetic": phase.get("arithmetic"),
                   "tiles": q(phase.get("tiles"), "static") if phase.get("tiles") is not None else None,
                   "shared_bytes": q(phase.get("shared_bytes"), "static") if phase.get("shared_bytes") is not None else None,
                   "selected_reason": decision.get("reason"),
                   "selected_flag": decision.get("selected"),
                   "provider_rule": prov.get("rule"), "provider_admitted": prov.get("admitted"),
                   "plan": {k: v for k, v in plan.items() if k != "estimate_us"}}
            if "estimate_us" in plan:
                row["planner_estimate_us"] = q(plan["estimate_us"], "estimate", "compiler matrix planner model")
            if phase.get("group_estimate_us") is not None:
                row["group_estimate_us"] = q(phase["group_estimate_us"], "estimate", "compiler group planner model")
            if phase.get("matrix_alternatives"):
                row["planner_alternatives"] = [{**alt, "estimate_us": q(alt.get("estimate_us"), "estimate")}
                                               for alt in phase["matrix_alternatives"]]
            rows.append(row)
    # decisions without an emitted phase (e.g. refused or absorbed products)
    for out, decision in decisions.items():
        if out in seen:
            continue
        rows.append({"region": path, "phase": None, "output": out,
                     "identity": _operation_identity(out, None, layouts, graph),
                     "engine": None, "selected_reason": decision.get("reason"),
                     "selected_flag": decision.get("selected"), "note": "decision recorded without its own matrix phase"})
    return rows


def _attention_rows(path, node):
    rows = []
    for group in _as_list(node.get("attention_groups")):
        if not isinstance(group, dict):
            continue
        rows.append({"region": path, "group": group.get("group"), "decision": group.get("decision"),
                     "recipe": group.get("recipe"), "variant": group.get("variant"),
                     "head_width": group.get("head_width"), "sequence": group.get("sequence"),
                     "head_batch": group.get("head_batch"), "mask": group.get("mask"),
                     "sources": group.get("sources"), "outputs": group.get("outputs"),
                     "removed_operations": q(len(_as_list(group.get("removed"))), "static",
                                             "captured operations replaced by the fused body"),
                     "source_commit": group.get("source_commit")})
    return rows


def _reduction_rows(path, node, graph):
    layouts = node.get("layouts") or {}
    rows = []
    for index, phase in enumerate(_as_list(node.get("phases"))):
        if not isinstance(phase, dict):
            continue
        kind = phase.get("kind")
        if kind not in _REDUCTION_KINDS and kind not in _INDEXED_KINDS:
            continue
        names = _as_list(phase.get("names"))
        row = {"region": path, "phase": index, "kind": kind, "outputs": names,
               "identity": [_operation_identity(n, t, layouts, graph) for n, t in _phase_ops(phase)][:8],
               "engine": phase.get("engine") or phase.get("indexed_engine"),
               "arithmetic": phase.get("arithmetic"),
               "reduction_elements": q(phase.get("reduction_elements"), "static") if phase.get("reduction_elements") is not None else None,
               "requires_full_grid": phase.get("requires_full_grid"),
               "counter_effects": _as_list(phase.get("counter_effects")) or None}
        if phase.get("group_estimate_us") is not None:
            row["group_estimate_us"] = q(phase["group_estimate_us"], "estimate", "compiler group planner model")
        rows.append(row)
    for entry in _as_list(node.get("ordered_index_gradients")):
        if isinstance(entry, dict):
            rows.append({"region": path, "kind": "indexed_gradient", **{k: entry.get(k) for k in
                         ("output", "decision", "indexed_reduction_policy", "source_commit")}})
    return rows


_REFUSAL_WORDS = ("refus", "reject", "not admit", "unsupported", "does not match", "requires", "no unique",
                  "disagree", "did not", "missing", "cannot", "outside")


def _rejection_rows(path, node, diagnostics):
    rows = []

    def add(category, subject, constraint, recorded_by, alternative=None, **extra):
        rows.append({"region": path, "category": category, "subject": subject,
                     "constraint": constraint if constraint else None,
                     "recorded_by": recorded_by, "alternative_selected": alternative, **extra})

    for decision in _as_list(node.get("matrix_decisions")):
        if not isinstance(decision, dict):
            continue
        reason = str(decision.get("reason") or "")
        if decision.get("selected") is False:
            add("matrix", decision.get("output"), reason or None, "manifest.matrix_decisions",
                alternative="generic/reference matrix path", candidate=decision.get("candidate"))
            continue
        # A selected decision may carry the compiler's own clause about a refused alternative.
        for clause in [c.strip() for c in reason.split(";") if c.strip()]:
            if any(w in clause.lower() for w in _REFUSAL_WORDS):
                add("matrix_alternative", decision.get("output"), clause, "manifest.matrix_decisions[].reason",
                    alternative=reason.split(";")[0].strip())
    for decision in _as_list((node.get("latch_matrix") or {}).get("decisions")):
        if not isinstance(decision, dict):
            continue
        for body, reasons in (decision.get("refused") or {}).items():
            add("matrix_body", decision.get("output"), "; ".join(map(str, _as_list(reasons))) or None,
                "manifest.latch_matrix.decisions[].refused", alternative=decision.get("selected"), body=body)
    for refusal in _as_list(node.get("attention_refusals")):
        if isinstance(refusal, dict):
            add("attention", refusal.get("group"), refusal.get("reason"), "manifest.attention_refusals",
                alternative="captured operations lowered individually")
    for refusal in _as_list((node.get("vocabulary_head") or {}).get("refused")):
        if isinstance(refusal, dict):
            add("vocabulary_head", refusal.get("log_softmax") or refusal.get("output"), refusal.get("reason"),
                "manifest.vocabulary_head.refused")
    selection = node.get("execution_selection") or {}
    if selection:
        selected = selection.get("selected")
        for reason in _as_list(selection.get("reasons")):
            text = str(reason)
            if selected == "common_static" and any(w in text.lower() for w in _REFUSAL_WORDS):
                add("execution_retention", "legacy_static (established emitter)", text,
                    "manifest.execution_selection.reasons", alternative="common_static")
        hazards = selection.get("legacy_view_hazards")
        if hazards and any(hazards.values()):
            add("execution_retention", "legacy_static (established emitter)",
                "new-value view reads: " + json.dumps(hazards, sort_keys=True),
                "manifest.execution_selection.legacy_view_hazards", alternative="common_static")
    for kind_key, subject_key in (("performance_schedule", "requested"), ("loop_schedule_policy", "requested_options")):
        pol = node.get(kind_key) or {}
        for reason in _as_list(pol.get("reasons")):
            if any(w in str(reason).lower() for w in _REFUSAL_WORDS):
                add(kind_key, str(pol.get(subject_key)), str(reason), f"manifest.{kind_key}.reasons",
                    alternative=pol.get("selected"))
    for key in ("epilogue_rejections", "row_epilogue_rejections", "refused_products"):
        for entry in _as_list(node.get(key)):
            if isinstance(entry, dict):
                add(key, entry.get("output") or entry.get("product") or entry.get("name"),
                    entry.get("reason"), f"manifest.{key}")
    group_rules = node.get("group_rules")
    if isinstance(group_rules, dict):
        for name, value in group_rules.items():
            if isinstance(value, dict):
                for entry in _as_list(value.get("refused") or value.get("rejected")):
                    if isinstance(entry, dict):
                        add("group_rule:" + name, entry.get("output") or entry.get("name"), entry.get("reason"),
                            f"manifest.group_rules.{name}")
    # collector records (only present when lowering ran under ops.explain.collect())
    for record in diagnostics or []:
        kind = record.get("kind")
        if kind == "matrix_provider_decision":
            decision = record.get("decision") or {}
            for candidate in _as_list(decision.get("candidates")):
                if isinstance(candidate, dict) and not candidate.get("admitted"):
                    add("matrix_body", record.get("output"), "; ".join(map(str, _as_list(candidate.get("reasons")))) or None,
                        "collector.matrix_provider_decision.candidates", alternative=decision.get("selected"),
                        body=candidate.get("body"), plan=candidate.get("plan"))
            for body, why in (decision.get("refused") or {}).items():
                add("matrix_body", record.get("output"), str(why), "collector.matrix_provider_decision.refused",
                    alternative=decision.get("selected"), body=body)
        elif kind == "selection_superseded":
            prior = record.get("previous") or {}
            add("execution_selection_superseded", prior.get("selected"),
                "; ".join(map(str, _as_list(prior.get("reasons")))) or None,
                "collector.selection_superseded", alternative=(record.get("replacement") or {}).get("selected"),
                stage=record.get("stage"))
        elif kind == "indexed_fallback":
            add("indexed_engine", record.get("output"), record.get("reason"), "collector.indexed_fallback",
                alternative=record.get("fallback"), engine=record.get("engine"))
        elif kind == "cast_rewrite":
            report = record.get("report") or {}
            for entry in _as_list(report.get("refused") or report.get("rejected")):
                add("cast_rewrite:" + str(record.get("pass")), entry.get("output") if isinstance(entry, dict) else str(entry),
                    entry.get("reason") if isinstance(entry, dict) else None, "collector.cast_rewrite")
    return rows


def _data_movement(path, node, graph):
    layouts = node.get("layouts") or {}
    inputs = set(_as_list(node.get("inputs")))
    outputs = set(_as_list(node.get("outputs")) if isinstance(node.get("outputs"), list) else (node.get("outputs") or {}).values())
    casts, layout_copies, clones = [], [], []
    phase_names = set()
    for index, phase in enumerate(_as_list(node.get("phases"))):
        if not isinstance(phase, dict):
            continue
        reads = _as_list(phase.get("reads"))
        for name, target in _phase_ops(phase):
            phase_names.add(name)
            if target in _CAST_TARGETS:
                out_dtype = (layouts.get(name) or {}).get("dtype")
                op = graph.op(name)
                src = op["inputs"][0] if op and op.get("inputs") else (reads[0] if len(reads) == 1 else None)
                in_dtype = graph.dtype(src) if src else None
                if in_dtype is None and src:
                    in_dtype = (layouts.get(src) or {}).get("dtype")
                entry = {"region": path, "phase": index, "operation": name, "source": src,
                         "from_dtype": in_dtype, "to_dtype": out_dtype,
                         "bytes_written": q(_bytes(layouts.get(name)), "static", "result layout") if layouts.get(name) else q(None, "unknown"),
                         "stored": name in (node.get("offsets") or {})}
                if graph.locate(name):
                    entry["source_location"] = graph.locate(name)
                if out_dtype is None:
                    entry["classification"] = "copy without a materialized result (no result layout; elided or absorbed)"
                    layout_copies.append(entry)
                elif in_dtype is None or in_dtype == out_dtype:
                    entry["classification"] = "layout/copy (dtype unchanged)" if in_dtype else "cast (input dtype unknown without IR)"
                    (layout_copies if in_dtype else casts).append(entry)
                else:
                    entry["classification"] = "dtype cast"
                    casts.append(entry)
            elif target in _COPY_TARGETS:
                clones.append({"region": path, "phase": index, "operation": name, "target": target,
                               "bytes_written": q(_bytes(layouts.get(name)), "static") if layouts.get(name) else q(None, "unknown")})
        if str(phase.get("kind", "")).startswith("attention_layout_restore"):
            clones.append({"region": path, "phase": index, "operation": ",".join(_as_list(phase.get("names"))),
                           "target": "attention_layout_restore", "reason": phase.get("reason")})
    offsets = node.get("offsets") or {}
    lifetimes = node.get("lifetimes") or {}
    temporaries, placed_outputs, scratch = [], [], []
    for name, word in offsets.items():
        layout = layouts.get(name)
        entry = {"name": name, "arena_word_offset": word, "bytes": q(_bytes(layout), "static") if layout else q(None, "unknown"),
                 "dtype": (layout or {}).get("dtype"), "shape": (layout or {}).get("shape"),
                 "lifetime": lifetimes.get(name)}
        if name in outputs:
            placed_outputs.append(entry)
        elif name in phase_names or (layout and name not in inputs):
            (scratch if name not in phase_names else temporaries).append(entry)
        else:
            scratch.append(entry)
    temporaries.sort(key=lambda e: -(e["bytes"]["value"] or 0))
    materialization = node.get("materialization") or {}
    plan = node.get("execution_plan") or {}
    bodies = _as_list(plan.get("bodies"))
    sync = {"barrier_count": q(node.get("barrier_count"), "static") if node.get("barrier_count") is not None else q(None, "unknown"),
            "barrier_schedule": node.get("barrier_schedule") or (
                "phase (one publication barrier per phase; manifest records no barrier_schedule)"
                if isinstance(node.get("barrier_count"), int) and node.get("barrier_count") == len(_as_list(node.get("phases"))) else None),
            "stages": q(len(_as_list(node.get("barrier_intervals"))), "static", "manifest.barrier_intervals") if node.get("barrier_intervals") else None,
            "full_grid_phases": q(sum(1 for p in _as_list(node.get("phases")) if isinstance(p, dict) and p.get("requires_full_grid")), "static"),
            "counter_barrier_phases": q(sum(1 for p in _as_list(node.get("phases")) if isinstance(p, dict) and p.get("counter_effects")), "static"),
            "global_fences": q(sum(1 for b in bodies if isinstance(b, dict) and b.get("global_fence")), "static", "execution_plan.bodies") if bodies else None,
            "internal_grid_barriers": node.get("internal_grid_barriers"),
            "publication": plan.get("publication"), "state_commit": plan.get("state_commit") or node.get("state_commit")}
    if node.get("barrier_storage"):
        sync["loop_barrier"] = node["barrier_storage"]
    return {"region": path,
            "casts": casts, "layout_copies": layout_copies, "clones": clones,
            "state_commits": {"in_place": q(len(node.get("in_place_commits") or {}), "static"),
                              "copied": q(len(node.get("copied_commits") or {}), "static"),
                              "copied_names": sorted((node.get("copied_commits") or {}).keys())},
            "temporaries": {"count": q(len(temporaries), "static"),
                            "bytes": q(sum((e["bytes"]["value"] or 0) for e in temporaries), "static", "sum of result layouts; arena packing may overlap lifetimes"),
                            "largest": temporaries[:12]},
            "placed_outputs": {"count": q(len(placed_outputs), "static"), "names": [e["name"] for e in placed_outputs]},
            "scratch": {"count": q(len(scratch), "static"), "names": [e["name"] for e in scratch][:40]},
            "arena_bytes": q(node.get("arena_bytes"), "static", "manifest.arena_bytes") if node.get("arena_bytes") is not None else q(None, "unknown"),
            "elided": {"stores": q(len(_as_list(node.get("elided_stores"))), "static"),
                       "layouts": q(len(_as_list(node.get("elided_layouts"))), "static"),
                       "materialization": {k: (q(v, "static") if isinstance(v, int) else q(len(_as_list(v)), "static"))
                                           for k, v in materialization.items()} if materialization else None},
            "synchronization": sync}


# --------------------------------------------------------------------------------------------- structure
def _structure(manifest, graph, profile_data, artifact):
    nodes = list(_walk(manifest))
    region_program = len(nodes) > 1 or bool(manifest.get("regions")) or bool(manifest.get("branches"))
    kind = "region_program" if region_program else "single_kernel"
    statement = manifest.get("execution")
    measured = None
    one_kernel = (profile_data or {}).get("one_kernel")
    if one_kernel:
        measured = {"cuda_events": one_kernel.get("cuda_events"), "runtime_launches": one_kernel.get("runtime_launches"),
                    "passed": one_kernel.get("passed"), "basis": one_kernel.get("basis")}
    return {"kind": kind,
            "static_claim": statement,
            "static_claim_basis": "manifest.execution (static schedule statement, not a launch census)",
            "measured_launch_census": q(measured, "measured", one_kernel.get("basis")) if measured else q(None, "unknown", "no native_profile receipt supplied"),
            "generated_regions": q(len(nodes), "static", "root manifest plus embedded region/branch manifests"),
            "regions": [{"path": p, "phase_count": len(_as_list(n.get("phases"))), "barrier_count": n.get("barrier_count"),
                         "arena_bytes": n.get("arena_bytes"), "schedule": n.get("schedule"),
                         "execution_selected": (n.get("execution_selection") or {}).get("selected")} for p, n in nodes],
            "runtime_control": {k: manifest.get(k) for k in ("runtime_loop", "predicate_input", "conditional_policy",
                                                              "loop_schedule_policy", "dynamic_batch") if manifest.get(k)},
            "external_operations": [],
            "external_operations_basis": ("the manifest lists no external operations; this compiler admits none "
                                          "(unsupported operations fail capture or lowering)"),
            "artifact": ({k: artifact.get(k) for k in ("schema", "execution", "arch", "torch", "fast_build",
                                                       "recurrent_gradients", "gradient_outputs", "source_hash")}
                         if artifact else None)}


# --------------------------------------------------------------------------------------------- costs
def _costs(manifest, nodes, profile_data, selections):
    if not profile_data or not profile_data.get("intervals"):
        return {"status": "unmeasured",
                "statement": "No profile was supplied: execution costs are unmeasured. Static counts and the "
                             "compiler's planning estimates are labelled as such and are not measurements.",
                "clock": None, "intervals": [], "by_phase": [], "whole_update": profile_data.get("whole_update") if profile_data else None,
                "static_estimates": _static_estimates(selections)}
    intervals = profile_data["intervals"]
    by_phase = profile_module.attribute(manifest, intervals)
    # attach measured interval time to matrix/attention/reduction rows of the root node
    phase_cost = {row["phase"]: row for row in by_phase}
    for group in ("matrices", "reductions"):
        for row in selections[group]:
            if row.get("region") == "program" and row.get("phase") in phase_cost:
                cost = phase_cost[row["phase"]]
                row["measured_interval_ms"] = q(cost["interval_ms"], "measured",
                                                f"interval {cost['interval']} shared by phases {cost['interval_phases']}; includes synchronization")
    return {"status": "measured", "clock": profile_data.get("clock"), "source": profile_data.get("source"),
            "binding": profile_data.get("binding"),
            "total_ms": q(profile_data.get("total_ms"), "measured", profile_data.get("clock")) if profile_data.get("total_ms") is not None else None,
            "whole_update": profile_data.get("whole_update"),
            "intervals": intervals, "by_phase": by_phase,
            "static_estimates": _static_estimates(selections)}


def _static_estimates(selections):
    rows = []
    for row in selections["matrices"]:
        est = row.get("planner_estimate_us") or row.get("group_estimate_us")
        if est:
            rows.append({"region": row["region"], "phase": row["phase"], "output": row["output"],
                         "engine": row.get("engine"), "estimate_us": est})
    for row in selections["reductions"]:
        if row.get("group_estimate_us"):
            rows.append({"region": row["region"], "phase": row["phase"], "output": ",".join(row["outputs"]),
                         "engine": row.get("engine"), "estimate_us": row["group_estimate_us"]})
    return {"basis": "compiler planning model (estimate); not a measurement and not a ranking of bottlenecks",
            "rows": rows}


def _slow_path_candidates(costs, selections, rejections):
    """Ranked only by measured interval time; never by operation count."""
    if costs["status"] != "measured":
        return {"status": "unmeasured", "statement": "No slow-path ranking without a profile.", "rows": []}
    rows = []
    rejection_by_output = {}
    for r in rejections:
        rejection_by_output.setdefault(r.get("subject"), []).append(r)
    for cost in sorted(costs["by_phase"], key=lambda c: -c["interval_ms"])[:12]:
        members = [row for row in selections["matrices"] + selections["reductions"]
                   if row.get("region") == "program" and row.get("phase") == cost["phase"]]
        outputs = [m.get("output") or ",".join(m.get("outputs", [])) for m in members]
        related = [r for out in outputs for r in rejection_by_output.get(out, [])]
        interval_name = next((r.get("names") for r in costs["intervals"] if r.get("interval") == cost["interval"]), None)
        rows.append({"phase": cost["phase"], "interval": cost["interval"], "interval_ms": q(cost["interval_ms"], "measured"),
                     "share_of_total": q(cost.get("share"), "measured") if cost.get("share") is not None else None,
                     "kind": cost["kind"], "names": cost["names"] or interval_name,
                     "interval_name": interval_name,
                     "selected": [{"output": m.get("output") or m.get("outputs"), "engine": m.get("engine"),
                                   "reason": m.get("selected_reason")} for m in members],
                     "recorded_rejections": related,
                     "caveat": "interval time includes synchronization and every phase sharing the interval"})
    return {"status": "measured", "rows": rows}


# --------------------------------------------------------------------------------------------- provenance
def _provenance(manifest, graph):
    roles = manifest.get("input_roles") or {}
    parameters = {name: role.get("parameter") for name, role in roles.items() if isinstance(role, dict) and role.get("parameter")}
    located = 0
    total = 0
    sample = []
    for path, node in _walk(manifest):
        for phase in _as_list(node.get("phases")):
            if not isinstance(phase, dict):
                continue
            for name, target in _phase_ops(phase):
                total += 1
                loc = graph.locate(name)
                if loc:
                    located += 1
                    if len(sample) < 12:
                        sample.append({"region": path, "operation": name, "target": target, "source": loc})
    basis = ("ops.explain source locations recorded at capture (opt-in)" if located else
             "no source locations recorded; operations are identified by captured name, aten target, result shape/dtype "
             "and the parameter roles below")
    return {"operations_total": q(total, "static"), "operations_with_source_location": q(located, "static"),
            "basis": basis, "sample": sample,
            "parameter_bindings": parameters,
            "structured_outputs": manifest.get("structured_outputs"),
            "capture": {k: graph.metadata.get(k) for k in ("capture", "torch_version", "precision_policy", "layout_copies",
                                                            "exact_round_trips_removed", "layout_canonical", "attention_canonical")
                        if graph.metadata.get(k) is not None} if graph.present else None}


# --------------------------------------------------------------------------------------------- entry points
def report_manifest(manifest, *, source=None, ir=None, diagnostics=None, profile=None, artifact=None, subject=None):
    graph = _Graph(ir_to_dict(ir))
    nodes = list(_walk(manifest))
    profile_data = profile_module.normalize(profile, manifest) if profile is not None else None
    selections = {"matrices": [], "attention": [], "reductions": [], "execution": [], "schedule_policies": {}}
    rejections, data_movement = [], []
    for path, node in nodes:
        selections["matrices"] += _matrix_rows(path, node, graph)
        selections["attention"] += _attention_rows(path, node)
        selections["reductions"] += _reduction_rows(path, node, graph)
        selections["execution"].append(_node_summary(path, node))
        rejections += _rejection_rows(path, node, diagnostics if path == "program" else None)
        data_movement.append(_data_movement(path, node, graph))
    for key in ("performance_schedule", "loop_schedule_policy", "compiler_selection", "fusion_policy", "numerical_policy",
                "primitive_policy", "compiler_policy", "group_rules"):
        if manifest.get(key) is not None:
            value = manifest[key]
            selections["schedule_policies"][key] = (value if not isinstance(value, dict) else
                                                    {k: v for k, v in value.items() if k not in ("options",)
                                                     and (not isinstance(v, (list, dict)) or k in ("reasons", "selected", "requested"))})
    costs = _costs(manifest, nodes, profile_data, selections)
    structure = _structure(manifest, graph, profile_data, artifact)
    subject = dict(subject or {})
    subject.setdefault("kind", "program")
    subject.update({"source_hash": manifest.get("source_hash"), "format": manifest.get("format"),
                    "target_arch": manifest.get("target_arch"),
                    "manifest_sha256": _sha(json.dumps(manifest, sort_keys=True).encode()),
                    "source_sha256": _sha(source.encode()) if isinstance(source, str) else None,
                    "source_phase_markers": q(len(re.findall(r"^// Phases? ", source, re.M)), "static", "'// Phase' comments in program.cu") if isinstance(source, str) else None,
                    "source_grid_syncs": q(source.count("grid.sync()"), "static", "grid.sync() occurrences in program.cu") if isinstance(source, str) else None,
                    "collector_records": q(len(diagnostics or []), "static"),
                    "ir_supplied": graph.present})
    report = {"schema": REPORT_SCHEMA, "subject": subject, "structure": structure,
              "selections": selections, "rejections": rejections,
              "rejection_basis": ("every constraint is quoted from the record named in recorded_by; entries with "
                                  "constraint=null mark a rejection the compiler recorded without a reason"),
              "data_movement": data_movement, "costs": costs,
              "slow_path_candidates": _slow_path_candidates(costs, selections, rejections),
              "provenance": _provenance(manifest, graph),
              "labels": {"measured": "from a supplied profile or receipt (clock named in costs)",
                         "static": "a count read from the plan/manifest/emitted source",
                         "estimate": "the compiler's own planning model; not a measurement",
                         "unknown": "not recorded anywhere available to this report"}}
    report["consistency"] = _consistency(report, manifest, source)
    return report


def _consistency(report, manifest, source):
    """Cheap cross-checks that report claims match the manifest and emitted code."""
    checks = []
    phases = _as_list(manifest.get("phases"))
    if isinstance(source, str) and phases and not manifest.get("regions") and not manifest.get("branches"):
        markers = set(int(m) for m in re.findall(r"^// Phases? (\d+)", source, re.M))
        for group in re.findall(r"^// Phases ([\d, ]+)", source, re.M):
            markers |= {int(x) for x in group.replace(" ", "").split(",") if x}
        covered = len(markers & set(range(len(phases))))
        checks.append({"check": "phase markers in program.cu cover manifest phases",
                       "manifest_phases": len(phases), "marked_phases": covered,
                       "passed": covered == len(phases) or covered == 0,
                       "note": None if covered else "emitter kept no per-phase comments for this schedule"})
    counted = sum(1 for r in report["selections"]["matrices"] if r.get("phase") is not None and r["region"] == "program")
    gemm = sum(1 for p in phases if isinstance(p, dict) and p.get("kind") in _MATRIX_KINDS)
    checks.append({"check": "matrix rows correspond to matrix phases of the root manifest",
                   "matrix_phases": gemm, "matrix_rows_with_phase": counted, "passed": counted >= gemm})
    if manifest.get("barrier_count") is not None and isinstance(source, str) and not manifest.get("regions"):
        syncs = source.count("grid.sync()")
        checks.append({"check": "grid.sync() occurrences versus barrier_count", "barrier_count": manifest["barrier_count"],
                       "grid_sync_in_source": syncs,
                       "passed": True,
                       "note": "textual count includes helper code and the state-commit barrier; it is not the runtime count"})
    return checks


def report_program(program, ir=None, diagnostics=None, profile=None, subject=None):
    manifest = _field(program, "manifest")
    source = _field(program, "source")
    return report_manifest(manifest, source=source, ir=ir, diagnostics=diagnostics, profile=profile, subject=subject)


def report_artifact(directory, profile=None, ir=None, diagnostics=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    source = (directory / "program.cu").read_text() if (directory / "program.cu").exists() else None
    artifact = json.loads((directory / "artifact.json").read_text()) if (directory / "artifact.json").exists() else None
    side = directory / "explain"
    if ir is None and (side / "ir.json").exists():
        ir = json.loads((side / "ir.json").read_text())
    if diagnostics is None and (side / "diagnostics.json").exists():
        diagnostics = json.loads((side / "diagnostics.json").read_text())
    return report_manifest(manifest, source=source, ir=ir, diagnostics=diagnostics, profile=profile, artifact=artifact,
                           subject={"kind": "artifact", "path": str(directory)})


def write_report(report, directory, *, stem="report"):
    """Write ``<directory>/<stem>.json`` and ``<stem>.md``; returns their paths."""
    from .render import render_markdown
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path
