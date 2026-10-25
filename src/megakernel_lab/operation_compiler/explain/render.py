"""Readable Markdown rendering of a compilation report (the JSON report is the structured output)."""
from __future__ import annotations


def _v(x, unit=None):
    """Render a labelled quantity as ``value [label]``."""
    if x is None:
        return "–"
    if isinstance(x, dict) and "label" in x:
        value = x.get("value")
        if value is None:
            return f"unknown"
        if isinstance(value, float):
            value = f"{value:.4g}"
        return f"{value}{(' ' + (x.get('unit') or unit)) if (x.get('unit') or unit) else ''} [{x['label']}]"
    return str(x)


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") if c is not None else "–" for c in row) + " |")
    return "\n".join(out)


def _short(text, n=110):
    text = "–" if text is None else str(text)
    return text if len(text) <= n else text[: n - 1] + "…"


def render_markdown(report):
    s = report["subject"]
    lines = [f"# Latch compilation report ({report['schema']})", ""]
    lines.append(f"Subject: `{s.get('kind')}` {s.get('path') or ''}  ")
    lines.append(f"Generated-source hash: `{s.get('source_hash')}`; manifest sha256 `{(s.get('manifest_sha256') or '')[:16]}…`; "
                 f"format `{s.get('format')}`; target `{s.get('target_arch')}`  ")
    lines.append(f"IR supplied: {s.get('ir_supplied')}; collector records: {_v(s.get('collector_records'))}; "
                 f"source phase markers: {_v(s.get('source_phase_markers'))}; grid.sync() in source: {_v(s.get('source_grid_syncs'))}")
    lines.append("")
    lines.append("Labels: **measured** = from a supplied profile/receipt; **static** = count from the plan; "
                 "**estimate** = compiler planning model, not a measurement; **unknown** = not recorded.")
    lines.append("")

    # ---- 1. structure
    st = report["structure"]
    lines += ["## 1. What executes", ""]
    if st["kind"] == "single_kernel":
        lines.append(f"**One generated program.** Static statement from the manifest: _{st.get('static_claim')}_. "
                     f"Generated regions: {_v(st['generated_regions'])}.")
    elif st["kind"] == "region_program":
        lines.append(f"**One generated program composed of {_v(st['generated_regions'])} regions** (root plus embedded "
                     f"region/branch manifests). Static statement: _{st.get('static_claim')}_.")
    else:
        lines.append(f"**{st.get('headline')}**")
    census = st.get("measured_launch_census")
    if census and census.get("value"):
        c = census["value"]
        lines.append(f"Measured launch census: events {c.get('cuda_events')}, launches {c.get('runtime_launches')}, "
                     f"passed={c.get('passed')} [measured; {c.get('basis')}].")
    else:
        lines.append("Measured launch census: unknown (no native_profile receipt supplied); the one-kernel statement above is static.")
    if st.get("regions") and len(st["regions"]) > 1:
        lines.append("")
        lines.append(_table(["region", "phases", "barriers", "arena bytes", "schedule", "execution", "role"],
                            [[r["path"], r["phase_count"], r["barrier_count"], r["arena_bytes"], r.get("schedule"), r.get("execution_selected"), _short(r.get("role"), 90)]
                             for r in st["regions"]]))
    if st.get("runtime_control"):
        lines.append("")
        lines.append("Runtime control: `" + _short(str(st["runtime_control"]), 400) + "`")
    ext = st.get("external_operations") or []
    lines.append("")
    if ext:
        lines.append(f"**External operations: {len(ext)}** (outside generated code). {st.get('external_operations_basis')}")
        lines.append("")
        lines.append(_table(["#", "kind", "target", "input→output", "bytes", "detail"],
                            [[e.get("index"), e.get("kind"), e.get("target"), f"{e.get('input')}→{e.get('output')}",
                              _v(e.get("bytes")), _short(e.get("detail"), 80)] for e in ext[:70]]))
    else:
        lines.append(f"External operations: none listed. {st.get('external_operations_basis')}")
    lines.append("")

    # ---- 2. selections
    sel = report["selections"]
    lines += ["## 2. Selected implementations", ""]
    executions = sel.get("execution", [])
    if len(executions) > 12:
        # Many regions (an assembly): group identical execution selections instead of repeating them.
        groups = {}
        for ex in executions:
            key = (ex.get("execution_selected"), tuple(ex.get("execution_reasons") or []))
            groups.setdefault(key, []).append(ex["path"])
        lines.append(f"{len(executions)} regions; {len(groups)} distinct execution selections:")
        for (selected, reasons), paths in groups.items():
            lines.append(f"- `{selected}` for {len(paths)} regions ({', '.join(p.split('/')[-1] for p in paths[:6])}{'…' if len(paths) > 6 else ''})")
            for reason in reasons:
                lines.append(f"  - reason: {reason}")
        executions = []
    for ex in executions:
        lines.append(f"- `{ex['path']}`: execution `{ex.get('execution_selected')}` (requested `{ex.get('execution_requested')}`), "
                     f"schedule `{ex.get('schedule')}`, phases {_v(ex['phase_count'])} {ex.get('phase_kinds')}, "
                     f"barriers {_v(ex['barrier_count'])}, arena {_v(ex['arena_bytes'], 'B')}.")
        for reason in ex.get("execution_reasons") or []:
            lines.append(f"  - reason: {reason}")
        if ex.get("loop_schedule_policy"):
            lines.append(f"  - loop schedule: `{ex['loop_schedule_policy'].get('selected')}` — {ex['loop_schedule_policy'].get('reasons')}")
        if ex.get("runtime_loop"):
            lines.append(f"  - runtime loop: `{_short(str(ex['runtime_loop']), 200)}`")
        if ex.get("performance_schedule"):
            lines.append(f"  - performance schedule: `{ex['performance_schedule'].get('selected')}` — {_short(str(ex['performance_schedule'].get('reasons')), 300)}")
    lines.append("")
    if sel["matrices"]:
        lines.append("### Matrix products")
        lines.append("")
        rows = []
        for m in sel["matrices"]:
            cost = m.get("measured_interval_ms")
            est = m.get("planner_estimate_us") or m.get("group_estimate_us")
            rows.append([m["region"], m.get("phase"), m["output"], m.get("engine"), m.get("matrix_shape"), m.get("dtype"),
                         _short(m.get("selected_reason"), 90), _v(cost, "ms") if cost else "unmeasured", _v(est, "us") if est else "–"])
        lines.append(_table(["region", "phase", "output", "engine", "shape", "dtype", "recorded reason", "interval cost", "planner estimate"], rows))
        lines.append("")
    if sel["attention"]:
        lines.append("### Attention")
        lines.append("")
        lines.append(_table(["region", "group", "decision", "recipe/variant", "head width", "sequence", "mask", "replaced ops", "source"],
                            [[a.get("region"), a.get("group") or _short(str(a.get("phase_kinds")), 60), a.get("decision"), a.get("recipe") or a.get("variant"),
                              a.get("head_width"), a.get("sequence"), a.get("mask"), _v(a.get("removed_operations")), a.get("source_commit")]
                             for a in sel["attention"]]))
        lines.append("")
    if sel["reductions"]:
        lines.append("### Reductions, norms and indexed operations")
        lines.append("")
        rows = []
        for r in sel["reductions"]:
            cost = r.get("measured_interval_ms")
            rows.append([r.get("region"), r.get("phase"), r.get("kind") or _short(str(r.get("phase_kinds")), 50), _short(",".join(r.get("outputs") or []) or r.get("output"), 60),
                         r.get("engine") or r.get("decision"), _v(r.get("reduction_elements")), _short(r.get("arithmetic"), 80),
                         _v(cost, "ms") if cost else "unmeasured"])
        lines.append(_table(["region", "phase", "kind", "outputs", "engine", "elements", "arithmetic", "interval cost"], rows))
        lines.append("")

    # ---- 3. rejections
    rej = report["rejections"]
    lines += ["## 3. Rejected alternatives and their recorded constraints", ""]
    if rej:
        lines.append(report.get("rejection_basis", ""))
        lines.append("")
        lines.append(_table(["region", "category", "subject", "body/alt", "constraint (verbatim)", "selected instead", "recorded by"],
                            [[r["region"], r["category"], _short(r.get("subject"), 40), r.get("body"), _short(r.get("constraint"), 140),
                              _short(r.get("alternative_selected"), 50), r["recorded_by"]] for r in rej]))
    else:
        lines.append("No rejection record exists in the manifest or collector output for this program. "
                     "This is the absence of a record, not evidence that no alternative was considered.")
    lines.append("")

    # ---- 4. data movement
    lines += ["## 4. Copies, casts, temporaries and synchronization", ""]
    for dm in report["data_movement"]:
        casts = dm["casts"]
        pairs = {}
        for c in casts:
            key = f"{c.get('from_dtype') or '?'}→{c.get('to_dtype')}"
            pairs.setdefault(key, [0, 0])
            pairs[key][0] += 1
            pairs[key][1] += (c["bytes_written"].get("value") or 0)
        sync = dm["synchronization"]
        lines.append(f"### `{dm['region']}`")
        lines.append("")
        lines.append(f"- Casts: {len(casts)} [static]" + (" — " + "; ".join(f"{k}: {n} ops, {b:,} B" for k, (n, b) in pairs.items()) if pairs else "") + ".")
        lines.append(f"- Layout copies (`_to_copy` without dtype change): {len(dm['layout_copies'])}; clones/copies: {len(dm['clones'])} [static].")
        sc = dm["state_commits"]
        lines.append(f"- State commits: {_v(sc['in_place'])} in place, {_v(sc['copied'])} copied" + (f" ({', '.join(sc['copied_names'][:8])})" if sc["copied_names"] else "") + ".")
        t = dm["temporaries"]
        lines.append(f"- Arena: {_v(dm['arena_bytes'], 'B')}; temporaries {_v(t['count'])} summing {_v(t['bytes'], 'B')} of result layouts; "
                     f"placed outputs {_v(dm['placed_outputs']['count'])}; scratch roots {_v(dm['scratch']['count'])}.")
        if t["largest"]:
            lines.append("  - largest temporaries: " + "; ".join(f"`{e['name']}` {e['dtype']} {e['shape']} {_v(e['bytes'], 'B')} lifetime {e['lifetime']}" for e in t["largest"][:6]))
        el = dm["elided"]
        if el.get("materialization"):
            lines.append(f"- Materialization: " + ", ".join(f"{k} {_v(v)}" for k, v in el["materialization"].items()))
        if el["stores"]["value"] or el["layouts"]["value"]:
            lines.append(f"- Elided stores {_v(el['stores'])}, elided layouts {_v(el['layouts'])}.")
        lines.append(f"- Synchronization: barriers {_v(sync['barrier_count'])}, schedule `{sync.get('barrier_schedule')}`, stages {_v(sync.get('stages'))}, "
                     f"full-grid phases {_v(sync['full_grid_phases'])}, counter-barrier phases {_v(sync['counter_barrier_phases'])}, "
                     f"global fences {_v(sync.get('global_fences'))}.")
        if sync.get("publication"):
            lines.append(f"  - publication: {sync['publication']}")
        if sync.get("loop_barrier"):
            lines.append(f"  - loop barrier: `{sync['loop_barrier']}`")
        if casts:
            lines.append("")
            lines.append(_table(["phase", "operation", "source", "from", "to", "bytes", "stored"],
                                [[c["phase"], c["operation"], c.get("source"), c.get("from_dtype"), c.get("to_dtype"), _v(c["bytes_written"]), c.get("stored")]
                                 for c in casts[:40]]))
        lines.append("")

    # ---- 5. costs
    costs = report["costs"]
    lines += ["## 5. Costs: measured, static and estimated", ""]
    if costs["status"] == "measured":
        lines.append(f"Clock: {costs.get('clock')}. Source: `{costs.get('source')}`.")
        b = costs.get("binding")
        if b:
            lines.append(f"Binding: {'bound' if b.get('bound') else 'NOT bound'} — profile source hash `{b.get('profile_source_hash')}` "
                         f"vs manifest `{b.get('manifest_source_hash')}`.")
        if costs.get("total_ms"):
            lines.append(f"Profiled total: {_v(costs['total_ms'], 'ms')}.")
        wu = costs.get("whole_update")
        if wu:
            lines.append(f"Complete-update clock (separate): {wu.get('median_ms')} ms [{wu.get('label')}] — {wu.get('basis')}; {_short(wu.get('clock'), 160)}")
        lines.append("")
        members = {}
        for row in costs.get("by_phase") or []:
            if row.get("phase") is not None:
                members.setdefault(row["interval"], []).append(row["phase"])
        total = sum((r.get("ms") or 0) for r in costs["intervals"]) or None
        lines.append(_table(["interval", "ms", "share", "phases", "kind", "names"],
                            [[r["interval"], f"{r['ms']:.4f}", f"{100 * r['ms'] / total:.1f}%" if total else "–",
                              members.get(r["interval"]) or r.get("phases") or "–", r.get("kind"), _short(r.get("names"), 70)]
                             for r in sorted(costs["intervals"], key=lambda r: -(r.get("ms") or 0))[:25]]))
    else:
        lines.append(costs["statement"])
        wu = costs.get("whole_update")
        if wu:
            lines.append("")
            lines.append(f"Whole-update measurement supplied separately (schema `{wu.get('schema')}`): {wu.get('statement')}")
            lines.append(f"Clock: {wu.get('clock')}")
            if wu.get("primary_S"):
                lines.append(f"Primary score S: {_v(wu['primary_S'])} ({wu['primary_S'].get('basis')})")
            rows = []
            for phase, arms in (wu.get("candidate_phases") or {}).items():
                for arm, value in arms.items():
                    rows.append([phase, arm, _v(value, "s")])
            for phase, value in (wu.get("paired_native_control_ratio_median") or {}).items():
                rows.append([phase, "paired native/control ratio", _v(value)])
            if rows:
                lines.append("")
                lines.append(_table(["phase", "arm", "median"], rows))
    est = costs.get("static_estimates") or {}
    if est.get("rows"):
        lines.append("")
        lines.append(f"Compiler planning estimates ({est['basis']}):")
        lines.append("")
        lines.append(_table(["region", "phase", "output", "engine", "estimate"],
                            [[r["region"], r["phase"], r["output"], r.get("engine"), _v(r["estimate_us"], "us")] for r in est["rows"][:30]]))
    lines.append("")

    # ---- 6. slow path
    sp = report["slow_path_candidates"]
    lines += ["## 6. Slow-path candidates", ""]
    if sp["status"] == "measured":
        lines.append("Ranked by measured interval time only (an interval's time includes synchronization and every phase sharing it).")
        lines.append("")
        for row in sp["rows"][:8]:
            selected = "; ".join(f"`{x['output']}` via `{x['engine']}` ({_short(x.get('reason'), 60)})" for x in row["selected"])
            if not selected:
                selected = (f"compiler-named interval: {row.get('interval_name')} (regions inside it are not split by this clock)"
                            if row.get("phase") is None and row.get("interval_name") else "no matrix/reduction member")
            where = f"phase {row['phase']}" if row.get("phase") is not None else "named interval"
            lines.append(f"- interval {row['interval']} {where}: {_v(row['interval_ms'], 'ms')}, kind `{row['kind']}` — {selected}")
            for r in row["recorded_rejections"][:4]:
                lines.append(f"  - recorded rejection ({r['category']}): {_short(r.get('constraint'), 140)} [{r['recorded_by']}]")
    else:
        lines.append(sp["statement"] + " Static counts are not used to infer a bottleneck.")
    lines.append("")

    # ---- 7. provenance
    pv = report["provenance"]
    lines += ["## 7. Provenance", ""]
    lines.append(f"Operations: {_v(pv['operations_total'])}; with source locations: {_v(pv['operations_with_source_location'])}. {pv['basis']}.")
    if pv.get("sample"):
        lines.append("")
        lines.append(_table(["region", "operation", "target", "module stack / source"],
                            [[r["region"], r["operation"], r["target"], _short(str(r["source"]), 160)] for r in pv["sample"]]))
    if pv.get("parameter_bindings"):
        lines.append("")
        lines.append("Parameter bindings: " + ", ".join(f"`{k}`={v}" for k, v in list(pv["parameter_bindings"].items())[:24]))
    lines.append("")

    # ---- 8. consistency
    if report.get("consistency"):
        lines += ["## 8. Consistency checks", ""]
        for c in report["consistency"]:
            lines.append(f"- {c['check']}: {'ok' if c.get('passed') else 'MISMATCH'} — " +
                         ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("check", "passed", "note")) +
                         (f" ({c['note']})" if c.get("note") else ""))
        lines.append("")
    return "\n".join(lines)
