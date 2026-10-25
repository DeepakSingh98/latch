"""Bounded reporting adapter for the four-GPU MoE research assembly (read-only).

The assembly artifact (``latch-distributed-update-v1``) and its program audit
(``latch-moe4-assembly-program-audit-v1``) come from a separate research checkout whose
compiler fork is not integrated here. This adapter only reads those two JSON files (plus
an optional paired whole-update measurement) and renders them in the report schema. It
does not describe the assembly as one training megakernel: the artifact itself states
"generated local regions plus external all-to-all/SUM collectives", and packing,
unpacking and orchestration stay outside the generated code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .report import REPORT_SCHEMA, q, report_artifact, _REFUSAL_WORDS

_ARTIFACT_SCHEMA = "latch-distributed-update-v1"
_AUDIT_SCHEMA = "latch-moe4-assembly-program-audit-v1"


def _read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), len(raw)


def report_moe4(artifact_json, audit_json=None, measurement_json=None, *, region_manifests=None, phase_label=None):
    artifact, artifact_sha, artifact_bytes = _read(artifact_json)
    if artifact.get("schema") != _ARTIFACT_SCHEMA:
        raise ValueError(f"expected {_ARTIFACT_SCHEMA}, found {artifact.get('schema')}")
    audit = audit_sha = None
    if audit_json:
        audit, audit_sha, _ = _read(audit_json)
        if audit.get("schema") != _AUDIT_SCHEMA:
            raise ValueError(f"expected {_AUDIT_SCHEMA}, found {audit.get('schema')}")
        if audit.get("artifact_sha256") and audit["artifact_sha256"] != artifact_sha:
            binding = {"audit_binds_this_artifact": False, "audit_artifact_sha256": audit["artifact_sha256"]}
        else:
            binding = {"audit_binds_this_artifact": True}
    else:
        binding = {"audit_binds_this_artifact": None}
    events = artifact.get("events") or []
    boundaries = artifact.get("boundaries") or {}
    regions = artifact.get("regions") or []
    audit_regions = {r["directory"]: r for r in (audit or {}).get("regions", [])}

    # ---- external operations from the event schedule
    external = []
    sum_index = None
    for index, event in enumerate(events):
        if event.get("kind") != "collective":
            continue
        inp = event.get("input")
        boundary = boundaries.get(inp) or {}
        target = str(event.get("target"))
        detail = (f"split_dim={event.get('split_dim')} concat_dim={event.get('concat_dim')} world={event.get('world_size')}"
                  if "redistribute" in target else "gradient SUM over payload buckets")
        if "distributed_sum" in target:
            sum_index = index
        external.append({"index": index, "kind": "collective", "target": target, "input": inp, "output": event.get("output"),
                         "bytes": q(boundary.get("bytes"), "static", "artifact.boundaries[input].bytes") if boundary else q(None, "unknown"),
                         "detail": detail})
    gsc = artifact.get("gradient_sum_contract") or {}
    buckets = gsc.get("buckets") or []
    # ---- region rows
    region_rows = []
    for position, region in enumerate(regions):
        name = region.get("directory")
        a = audit_regions.get(name, {})
        event_index = next((i for i, e in enumerate(events) if e.get("kind") == "region" and e.get("region") == position), None)
        role = None
        if sum_index is not None and event_index is not None and event_index == sum_index + 1:
            role = "optimizer/update region (structural: the region event immediately after the gradient SUM collective)"
        exports = (events[event_index] or {}).get("exports") if event_index is not None else None
        sel = a.get("execution_selection") or region.get("execution_selection") or {}
        matrices = a.get("matrices") or []
        region_rows.append({
            "path": f"assembly/{name}", "position": position, "event_index": event_index, "structural_role": role,
            "source_hash": region.get("source_hash"), "library_sha256": region.get("library_sha256"),
            "phase_count": q(a.get("phase_count"), "static") if a else q(None, "unknown", "audit not supplied"),
            "barrier_count": q(a.get("barriers"), "static") if a else q(None, "unknown"),
            "phase_kinds": a.get("phase_kinds"), "arena_bytes": q(a.get("arena_bytes") if a else region.get("arena_words", 0) * 4, "static"),
            "shared_bytes": q(a.get("shared_bytes"), "static") if a else None,
            "execution_selected": sel.get("selected"), "execution_reasons": list(sel.get("reasons") or []),
            "matrices": [{"phase": m.get("phase"), "output": ",".join(m.get("names") or []), "engine": m.get("engine"),
                          "matrix_shape": m.get("matrix_shape"), "batch": m.get("batch"), "arithmetic": m.get("arithmetic"),
                          "planner_estimate_us": q((m.get("matrix_plan") or {}).get("estimate_us"), "estimate", "compiler matrix planner model")
                          if (m.get("matrix_plan") or {}).get("estimate_us") is not None else None,
                          "batch_stacked_tma": m.get("batch_stacked_tma"), "batch_widening": m.get("batch_widening")} for m in matrices],
            "matrix_decisions": a.get("matrix_decisions"),
            "row_phases": a.get("row_phases"), "level_split_groups": a.get("level_split_groups"),
            "local_pointer_modes": a.get("local_pointer_modes"), "exact_vocab_phases": a.get("exact_vocab_phases"),
            "exports": q(len(exports or {}), "static", "values exported across the region boundary") if exports is not None else None,
            "lower_seconds": q(region.get("lower_seconds"), "measured", "build-worker wall clock (compile time, not execution)") if region.get("lower_seconds") is not None else None,
            "build_seconds": q(region.get("build_seconds"), "measured", "build-worker wall clock; near zero when a cached library was reused") if region.get("build_seconds") is not None else None,
        })
    # ---- rejections quoted from per-region records
    rejections = []
    for row in region_rows:
        for decision in row.get("matrix_decisions") or []:
            reason = str(decision.get("reason") or "")
            if decision.get("selected") is False:
                rejections.append({"region": row["path"], "category": "matrix", "subject": decision.get("output"), "constraint": reason or None,
                                   "recorded_by": "assembly-audit.regions[].matrix_decisions", "alternative_selected": "generic matrix path"})
            else:
                for clause in [c.strip() for c in reason.split(";") if c.strip()]:
                    if any(w in clause.lower() for w in _REFUSAL_WORDS):
                        rejections.append({"region": row["path"], "category": "matrix_alternative", "subject": decision.get("output"),
                                           "constraint": clause, "recorded_by": "assembly-audit.regions[].matrix_decisions[].reason",
                                           "alternative_selected": reason.split(";")[0].strip()})
        for reason in row.get("execution_reasons") or []:
            if row.get("execution_selected") == "common_static" and any(w in str(reason).lower() for w in _REFUSAL_WORDS):
                rejections.append({"region": row["path"], "category": "execution_retention", "subject": "legacy_static (established emitter)",
                                   "constraint": str(reason), "recorded_by": "region execution_selection.reasons",
                                   "alternative_selected": "common_static"})
    # ---- optional full per-region reports from downloaded region manifests
    region_reports = {}
    if region_manifests:
        base = Path(region_manifests)
        for row in region_rows:
            directory = base / row["path"].split("/")[-1]
            if (directory / "manifest.json").exists():
                region_reports[row["path"]] = report_artifact(directory)
    # ---- whole-update measurement (not per region)
    whole = None
    if measurement_json:
        measurement, measurement_sha, _ = _read(measurement_json)
        summary = measurement.get("summary") or {}
        whole = {"schema": measurement.get("schema"), "sha256": measurement_sha, "clock": measurement.get("clock"),
                 "primary_S": q(measurement.get("primary_S") or measurement.get("S"), "measured", "paired whole-update score; not a per-region cost"),
                 "candidate_phases": {phase: {arm: q(v.get("median_seconds"), "measured", "median of round medians; synchronized max-rank host wall")
                                              for arm, v in (arms or {}).items() if isinstance(v, dict)}
                                      for phase, arms in ((summary.get("candidate") or {}).get("phases") or {}).items()},
                 "paired_native_control_ratio_median": {phase: q(arms.get("paired_native_control_ratio_median"), "measured", "paired whole-update ratio")
                                                        for phase, arms in ((summary.get("candidate") or {}).get("phases") or {}).items() if isinstance(arms, dict)},
                 "statement": "whole-update wall time on four GPUs; it does not attribute cost to any region or collective"}
    counts = {"generated_regions": q(len(regions), "static", "artifact.regions"),
              "generated_launches": q(artifact.get("generated_launches"), "static", "artifact.generated_launches (static schedule, not a CUDA launch census)"),
              "logical_collective_events": q(artifact.get("collective_calls"), "static"),
              "physical_payload_collectives": q(artifact.get("physical_collective_calls"), "static", "32 all-to-all + 33 SUM buckets"),
              "all_to_all": q(sum(1 for e in external if "redistribute" in e["target"]), "static"),
              "gradient_sum_buckets": q(gsc.get("payload_collective_calls") or len(buckets), "static", f"bucket cap {gsc.get('bucket_cap_bytes')} B; {gsc.get('total_elements')} elements"),
              "boundary_tensors": q(len(boundaries), "static"),
              "boundary_buffer_bytes": q(sum((b or {}).get("bytes") or 0 for b in boundaries.values()), "static"),
              "transport_bytes": q(artifact.get("transport_bytes"), "static"),
              "gradient_bucket_bytes": q(artifact.get("gradient_bucket_bytes"), "static")}
    structure = {"kind": "assembly",
                 "headline": (f"Not one training megakernel: {len(regions)} generated local regions ({artifact.get('generated_launches')} "
                              f"static launches) plus {artifact.get('physical_collective_calls')} external payload collectives; "
                              "packing/unpacking and runtime orchestration are external."),
                 "static_claim": artifact.get("execution"),
                 "static_claim_basis": "artifact.execution and the event schedule (static inventory, not a CUDA launch census)",
                 "measured_launch_census": q(None, "unknown", "profiler census pending in the donor lane (HANDOFF: PENDING)"),
                 "generated_regions": counts["generated_regions"], "counts": counts,
                 "regions": [{"path": r["path"], "phase_count": r["phase_count"]["value"], "barrier_count": r["barrier_count"]["value"],
                              "arena_bytes": r["arena_bytes"]["value"], "execution_selected": r["execution_selected"], "role": r["structural_role"]}
                             for r in region_rows],
                 "external_operations": external,
                 "external_operations_basis": "artifact.events collectives; gradient SUM expands into payload buckets per gradient_sum_contract",
                 "world_size": artifact.get("world_size"), "artifact": {"schema": artifact.get("schema"), "execution": artifact.get("execution")}}
    engines = (audit or {}).get("matrix_engines")
    selections = {"matrices": [dict(m, region=r["path"]) for r in region_rows for m in r["matrices"]],
                  "attention": [{"region": r["path"], "phase_kinds": {k: v for k, v in (r.get("phase_kinds") or {}).items() if k.startswith("attention")}}
                                for r in region_rows if any(k.startswith("attention") for k in (r.get("phase_kinds") or {}))],
                  "reductions": [{"region": r["path"], "phase_kinds": {k: v for k, v in (r.get("phase_kinds") or {}).items()
                                                                         if k in ("reduction", "embedding_gradient", "index_stable_merge", "embedding")}}
                                 for r in region_rows if r.get("phase_kinds")],
                  "execution": [{"path": r["path"], "execution_selected": r["execution_selected"], "execution_reasons": r["execution_reasons"],
                                 "phase_count": r["phase_count"], "phase_kinds": r["phase_kinds"], "barrier_count": r["barrier_count"],
                                 "arena_bytes": r["arena_bytes"], "schedule": None, "execution_requested": "compose",
                                 "compiler_selection": None, "execution_statement": None} for r in region_rows],
                  "schedule_policies": {"matrix_engines": engines, "row_selection_counts": (audit or {}).get("row_selection_counts"),
                                        "rounded_widening_phases": (audit or {}).get("rounded_widening_phases")}}
    report = {"schema": REPORT_SCHEMA,
              "subject": {"kind": "moe4_assembly", "path": str(artifact_json), "artifact_sha256": artifact_sha, "artifact_bytes": artifact_bytes,
                          "audit_path": str(audit_json) if audit_json else None, "audit_sha256": audit_sha, **binding,
                          "phase_label": phase_label, "source_hash": None, "format": artifact.get("schema"), "target_arch": None,
                          "manifest_sha256": artifact_sha, "ir_supplied": False, "collector_records": q(0, "static"),
                          "note": "read-only adapter over a separate research checkout; per-region manifests live in the donor's cloud storage"},
              "structure": structure, "selections": selections, "rejections": rejections,
              "rejection_basis": "constraints quoted verbatim from the assembly audit's per-region records",
              "data_movement": [{"region": "assembly", "casts": [], "layout_copies": [], "clones": [],
                                 "state_commits": {"in_place": q(None, "unknown"), "copied": q(None, "unknown"), "copied_names": []},
                                 "temporaries": {"count": q(None, "unknown", "per-region manifests not available locally"), "bytes": q(None, "unknown"), "largest": []},
                                 "placed_outputs": {"count": q(None, "unknown"), "names": []},
                                 "scratch": {"count": q(None, "unknown"), "names": []},
                                 "arena_bytes": q(sum(r["arena_bytes"]["value"] or 0 for r in region_rows), "static", "sum of per-region arenas"),
                                 "elided": {"stores": q(0, "static"), "layouts": q(0, "static"), "materialization": None},
                                 "synchronization": {"barrier_count": q(sum((r["barrier_count"]["value"] or 0) for r in region_rows), "static", "sum of per-region barrier counts"),
                                                     "barrier_schedule": None, "stages": None, "full_grid_phases": q(None, "unknown"),
                                                     "counter_barrier_phases": q(None, "unknown"), "global_fences": None, "internal_grid_barriers": None,
                                                     "publication": "region boundaries publish exports to external buffers; collectives synchronize ranks",
                                                     "state_commit": None},
                                 "boundaries": counts}],
              "costs": {"status": "unmeasured",
                        "statement": ("No per-region profile exists for this assembly (the donor lane's profile capture/reduction is pending). "
                                      "Per-region costs are unmeasured; lower/build seconds are compile-time clocks."),
                        "clock": None, "intervals": [], "by_phase": [], "whole_update": whole,
                        "static_estimates": {"basis": "compiler matrix planner model (estimate); not a measurement",
                                             "rows": [{"region": m["region"], "phase": m["phase"], "output": m["output"], "engine": m["engine"],
                                                       "estimate_us": m["planner_estimate_us"]} for m in selections["matrices"] if m.get("planner_estimate_us")]}},
              "slow_path_candidates": {"status": "unmeasured", "statement": "No slow-path ranking without a per-region profile.", "rows": []},
              "provenance": {"operations_total": q(sum((r["phase_count"]["value"] or 0) for r in region_rows), "static", "phases, not operations"),
                             "operations_with_source_location": q(0, "static"),
                             "basis": "regions are identified by index, source hash, phase kinds and exports; the optimizer region structurally follows the gradient SUM",
                             "sample": [], "parameter_bindings": {"parameter_names": len((artifact.get("metadata") or {}).get("parameter_names") or [])},
                             "structured_outputs": None, "capture": {k: (artifact.get("metadata") or {}).get(k) for k in ("capture", "torch_version", "numerical_policy")}},
              "region_reports": region_reports,
              "labels": {"measured": "from a supplied receipt", "static": "count from artifact/audit", "estimate": "compiler planning model", "unknown": "not recorded"},
              "consistency": [
                  {"check": "physical collectives = all-to-all + SUM buckets", "all_to_all": counts["all_to_all"]["value"],
                   "sum_buckets": counts["gradient_sum_buckets"]["value"], "physical": counts["physical_payload_collectives"]["value"],
                   "passed": counts["all_to_all"]["value"] + (counts["gradient_sum_buckets"]["value"] or 0) == counts["physical_payload_collectives"]["value"]},
                  {"check": "region events = regions", "region_events": sum(1 for e in events if e.get("kind") == "region"),
                   "regions": len(regions), "passed": sum(1 for e in events if e.get("kind") == "region") == len(regions)},
                  {"check": "audit matrix_engines total = audit matrix phases", "engines_total": sum((engines or {}).values()) if engines else None,
                   "matrix_phase_count": (audit or {}).get("matrix_phase_count"),
                   "passed": (sum(engines.values()) == audit.get("matrix_phase_count")) if engines and audit else None}]}
    return report
