"""Profile adapters: turn existing measured records into labelled interval costs.

Accepted inputs (no new instrumentation; nothing here runs on a GPU):

* rows from ``Runtime.profile_step()`` / ``runtime.dependency_profile_intervals`` --
  ``[{phase, kind, values, nanoseconds[, phases]}]``;
* the ``native_profile`` dict of ``research/latch_release/ffn/bench.py`` --
  ``{total_ms, by_category_ms, top_intervals:[{interval, ms, kinds, members, values}]}``;
* an FFN ``timing.json`` receipt (``research/latch_release/ffn/extract.py``) -- ``rounds[*].profile``
  for native arms plus ``median_ms`` complete-update clocks and ``artifact_bindings``;
* a ``native_profile.py`` receipt -- ``{cuda_events, runtime_launches, passed}`` (launch census);
* an already normalized ``{"intervals": [...], "clock": ...}`` dict; or a path to any of these.

Interval times come from CTA0 ``globaltimer`` stamps taken after each cooperative barrier,
so they include synchronization and every phase that shares the interval. The adapter
keeps that caveat on every row and binds the profile to the manifest by source hash
when the record carries one.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics

_CTA0 = "CTA0 globaltimer stamps after each cooperative barrier; intervals include synchronization"


def _load(profile):
    if isinstance(profile, (str, Path)):
        return json.loads(Path(profile).read_text()), str(profile)
    return profile, None


def normalize(profile, manifest, *, arm=None):
    """Return {intervals, total_ms, clock, source, binding, whole_update, one_kernel} or None."""
    data, origin = _load(profile)
    out = {"intervals": [], "total_ms": None, "clock": None, "source": origin, "binding": None,
           "whole_update": None, "one_kernel": None}
    if data is None:
        return None
    if isinstance(data, dict) and "intervals" in data and "clock" in data:
        out.update({k: data.get(k, out[k]) for k in out})
        return out
    if isinstance(data, dict) and data.get("schema") == "latch-explain-profile-rows-v1":
        # lane.py profile receipt: median profile_step rows plus a launch census
        inner = normalize(data["rows"], manifest)
        inner["clock"] = data.get("clock") or inner["clock"]
        inner["source"] = origin
        census = data.get("launch_census")
        if census:
            inner["one_kernel"] = {**census, "basis": census.get("basis", "torch.profiler CUDA events during one runtime.step()")}
            inner["binding"] = {"profile_source_hash": census.get("source_hash"), "manifest_source_hash": manifest.get("source_hash"),
                                "bound": census.get("source_hash") == manifest.get("source_hash"),
                                "basis": "exact generated-source hash equality between the profiled build and this manifest"}
        return inner
    if isinstance(data, list) and data and isinstance(data[0], dict) and "nanoseconds" in data[0]:
        for row in data:
            out["intervals"].append({"interval": row.get("phase"), "kind": row.get("kind"),
                                     "names": row.get("values"), "phases": row.get("phases"),
                                     "ms": row["nanoseconds"] / 1e6, "samples": 1})
        out["total_ms"] = sum(r["ms"] for r in out["intervals"])
        out["clock"] = _CTA0 + " (profile_step: one diagnostic update)"
        return out
    if isinstance(data, dict) and "top_intervals" in data:
        for row in data["top_intervals"]:
            out["intervals"].append({"interval": row.get("interval"), "kind": ",".join(row.get("kinds") or []) if isinstance(row.get("kinds"), list) else row.get("kinds"),
                                     "names": row.get("values"), "member_count": row.get("members"), "ms": row.get("ms"), "samples": None})
        out["total_ms"] = data.get("total_ms")
        out["by_category_ms"] = data.get("by_category_ms")
        out["clock"] = _CTA0 + " (bench.native_profile: mean over its profiled updates)"
        return out
    if isinstance(data, dict) and "rounds" in data:
        rounds = [r for r in data["rounds"] if isinstance(r, dict) and r.get("profile")]
        arms = sorted({r.get("arm") for r in rounds})
        chosen = arm or next((a for a in arms if str(a).startswith("native:current")), None) or (arms[0] if arms else None)
        rows = [r for r in rounds if r.get("arm") == chosen]
        per_interval = {}
        meta = {}
        for r in rows:
            for row in r["profile"].get("top_intervals", []):
                per_interval.setdefault(row["interval"], []).append(row["ms"])
                meta[row["interval"]] = row
        for interval, values in sorted(per_interval.items()):
            row = meta[interval]
            out["intervals"].append({"interval": interval, "kind": ",".join(row.get("kinds") or []) if isinstance(row.get("kinds"), list) else row.get("kinds"),
                                     "names": row.get("values"), "member_count": row.get("members"),
                                     "ms": statistics.median(values), "samples": len(values)})
        totals = [r["profile"].get("total_ms") for r in rows if r["profile"].get("total_ms") is not None]
        out["total_ms"] = statistics.median(totals) if totals else None
        out["clock"] = (_CTA0 + f" (timing.json arm {chosen}: median over {len(rows)} profiled rounds; "
                        "the complete-update clock below is separate)")
        out["whole_update"] = {"median_ms": data.get("median_ms"), "clock": data.get("clock"),
                               "label": "measured", "basis": "complete-update CUDA-event clock from the same receipt",
                               "arm": chosen, "numerical_eligibility": data.get("numerical_eligibility")}
        binding = ((data.get("artifact_bindings") or {}).get("current") or {})
        recorded = binding.get("source_hash") or data.get("source_hash")
        out["binding"] = {"profile_source_hash": recorded, "manifest_source_hash": manifest.get("source_hash"),
                          "bound": bool(recorded) and recorded == manifest.get("source_hash"),
                          "basis": "exact generated-source hash equality between the profiled artifact and this manifest"}
        return out
    if isinstance(data, dict) and "cuda_events" in data:
        out["one_kernel"] = {"cuda_events": data.get("cuda_events"), "runtime_launches": data.get("runtime_launches"),
                             "passed": data.get("passed"), "basis": "native_profile.py launch census (torch.profiler CUDA events)"}
        out["binding"] = {"profile_source_hash": data.get("source_hash"), "manifest_source_hash": manifest.get("source_hash"),
                          "bound": data.get("source_hash") == manifest.get("source_hash")}
        return out
    raise ValueError("unrecognized profile record; see ops.explain.profile for accepted forms")


def attribute(manifest, intervals):
    """Map measured intervals to manifest phases following the declared schedule.

    Dependency schedules publish several phases per interval (``barrier_intervals``); phase
    schedules publish one phase per interval; pipelined loops name their intervals
    (``profile_intervals``). The time of an interval is never split among its members.
    """
    phases = [p for p in (manifest.get("phases") or []) if isinstance(p, dict)]
    stages = manifest.get("barrier_intervals")
    named = manifest.get("profile_intervals")
    total = sum(r["ms"] for r in intervals if r.get("ms") is not None) or None
    rows = []
    for row in intervals:
        index = row.get("interval")
        if row.get("phases"):
            members = list(row["phases"])
        elif named and isinstance(index, int) and index < len(named):
            members = []
        elif stages and isinstance(index, int) and index < len(stages):
            members = list(stages[index])
        elif isinstance(index, int) and index < len(phases):
            members = [index]
        else:
            members = []
        share = (row["ms"] / total) if total and row.get("ms") is not None else None
        if not members:
            rows.append({"phase": None, "kind": row.get("kind"), "names": row.get("names"), "interval": index,
                         "interval_phases": [], "interval_ms": row.get("ms"), "share": share})
        for member in members:
            phase = phases[member] if member < len(phases) else {}
            rows.append({"phase": member, "kind": phase.get("kind"), "names": phase.get("names"),
                         "engine": phase.get("engine"), "interval": index, "interval_phases": members,
                         "interval_ms": row.get("ms"), "share": share})
    return rows
