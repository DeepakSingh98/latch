"""LATCH-EXPLAIN contracts: reports are a pure function of existing manifests and never change lowering.

CPU only (pinned torch); no device. The unchanged-program contracts compare generated source,
header hashes and the complete manifest with and without the reporting hooks enabled.
"""
import copy
import json

import pytest

from megakernel_lab.operation_compiler.explain import collect, note, render_markdown, report_manifest, report_program
from megakernel_lab.operation_compiler.explain.profile import attribute, normalize
from megakernel_lab.operation_compiler.lower import lower
from .test_compose import _value


def small_graph():
    return {"inputs": ["x"], "outputs": ["y"],
            "values": {name: _value([64], "float32") for name in ("x", "y")},
            "operations": [{"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, 2]}]}


def _identical(a, b):
    assert a.source == b.source
    assert a.manifest["header_hashes"] == b.manifest["header_hashes"]
    assert a.cache_key == b.cache_key
    assert json.dumps(a.manifest, sort_keys=True) == json.dumps(b.manifest, sort_keys=True)


# ---------------------------------------------------------------------------------------- collector
def test_collector_is_inert_without_scope():
    note("anything", value=1)            # no sink: must be a no-op
    with collect() as records:
        note("inner", value={"a": [1, 2]})
        with collect() as nested:
            note("deep", value=3)
        note("after", value=4)
    assert [r["kind"] for r in records] == ["inner", "after"]
    assert nested == [{"kind": "deep", "value": 3}]


def test_collector_leaves_lowering_unchanged_on_dict_graphs():
    graph = small_graph()
    plain = lower(graph, execution="compose")
    with collect() as records:
        hooked = lower(copy.deepcopy(graph), execution="compose")
    _identical(plain, hooked)
    assert any(r["kind"] == "selection_snapshot" for r in records)
    forced_plain = lower(graph, execution="compose", compose_selection="common")
    with collect():
        forced_hooked = lower(copy.deepcopy(graph), execution="compose", compose_selection="common")
    _identical(forced_plain, forced_hooked)


# ---------------------------------------------------------------------------------------- report over lower()
def test_report_program_from_lowered_dict_graph():
    graph = small_graph()
    with collect() as records:
        program = lower(graph, execution="compose", compose_selection="common")
    report = report_program(program, ir=graph, diagnostics=records)
    assert report["schema"] == "latch-explain-report-v1"
    assert report["structure"]["kind"] == "single_kernel"
    assert report["structure"]["external_operations"] == []
    assert report["costs"]["status"] == "unmeasured"
    assert report["slow_path_candidates"]["status"] == "unmeasured"
    assert report["subject"]["source_hash"] == program.cache_key
    assert report["subject"]["collector_records"]["value"] == len(records)
    text = render_markdown(report)
    assert "unmeasured" in text and "## 3. Rejected alternatives" in text
    # every quantity is labelled
    def labelled(node):
        if isinstance(node, dict):
            if "label" in node and "value" in node:
                assert node["label"] in ("measured", "static", "estimate", "unknown")
            for v in node.values():
                labelled(v)
        elif isinstance(node, list):
            for v in node:
                labelled(v)
    labelled(report)


def test_module_capture_with_source_locations_is_metadata_only(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from megakernel_lab.operation_compiler import runtime
    from megakernel_lab.operation_compiler.api import capture_module

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32, bias=False)

        def forward(self, x):
            h = (x.to(torch.bfloat16) @ self.proj.weight.to(torch.bfloat16).t()).float()
            return h + 0.25 * torch.nn.functional.silu(h)

    loss = lambda prediction, target: (prediction - target).square().mean()
    x, y = torch.full((16, 32), 0.1), torch.zeros((16, 32))
    torch.manual_seed(0)
    plain = capture_module(Model(), loss, (x,), (y,))
    torch.manual_seed(0)
    located = capture_module(Model(), loss, (x,), (y,), source_locations=True)
    a, b = plain.ir.to_dict(), located.ir.to_dict()
    assert a["operations"] == b["operations"] and a["values"] == b["values"]
    assert "source_locations" not in a["metadata"]
    assert "source_locations" in b["metadata"]
    _identical(lower(plain.ir, execution="compose"), lower(located.ir, execution="compose"))

    programs = []

    def fake_build(program, *, cache_dir, arch):
        programs.append(program)
        library = tmp_path / (str(len(programs)) + ".so")
        library.write_bytes(b"CPU contract only; no executable library")
        return library

    monkeypatch.setattr(runtime, "build", fake_build)
    first = located.build(tmp_path / "plain", cache_dir=tmp_path / "cache", execution="compose")
    second = located.build(tmp_path / "explained", cache_dir=tmp_path / "cache", execution="compose", explain=True)
    assert programs[0].source == programs[1].source and programs[0].manifest == programs[1].manifest
    assert first["source_hash"] == second["source_hash"] and first["files"] == second["files"]
    side = tmp_path / "explained" / "explain"
    assert (side / "report.json").exists() and (side / "report.md").exists() and (side / "ir.json").exists()
    report = json.loads((side / "report.json").read_text())
    assert report["subject"]["ir_supplied"] is True
    assert report["provenance"]["operations_total"]["value"] >= 1
    # the manifest written to disk is identical with and without explain
    assert (tmp_path / "plain" / "manifest.json").read_text() == (tmp_path / "explained" / "manifest.json").read_text()
    assert (tmp_path / "plain" / "program.cu").read_text() == (tmp_path / "explained" / "program.cu").read_text()


# ---------------------------------------------------------------------------------------- pure-manifest contracts
def _manifest():
    return {"format": "latch-ops-cuda-v1", "source_hash": "s" * 64, "target_arch": "sm_90",
            "execution": "one cooperative CUDA launch", "barrier_count": 3, "arena_bytes": 4096,
            "inputs": ["arg0_1", "arg1_1"], "outputs": ["mm_1"],
            "input_roles": {"arg0_1": {"role": "parameter", "parameter": "w"}, "arg1_1": {"role": "input", "parameter": None}},
            "layouts": {"arg0_1": {"shape": [8, 8], "dtype": "float32"}, "arg1_1": {"shape": [8, 8], "dtype": "float32"},
                        "_to_copy": {"shape": [8, 8], "dtype": "bfloat16"}, "mm": {"shape": [8, 8], "dtype": "float32"},
                        "mm_1": {"shape": [8, 8], "dtype": "float32"}},
            "offsets": {"_to_copy": 0, "mm": 64, "mm_1": 128}, "lifetimes": {"_to_copy": ["live", 0, 1], "mm": ["live", 1, 2]},
            "phases": [{"kind": "elementwise", "names": ["_to_copy"], "targets": ["aten._to_copy.default"], "reads": ["arg1_1"], "writes": ["_to_copy"]},
                       {"kind": "gemm", "names": ["mm"], "targets": ["aten.mm.default"], "engine": "generic_simt_16",
                        "matrix_shape": [8, 8, 8], "tiles": 1, "matrix_plan": {"estimate_us": 1.5}, "reads": ["_to_copy", "arg0_1"], "writes": ["mm"]},
                       {"kind": "gemm", "names": ["mm_1"], "targets": ["aten.mm.default"], "engine": "p5c_ws_128x128",
                        "matrix_shape": [8, 8, 8], "reads": ["mm", "arg0_1"], "writes": ["mm_1"], "group_estimate_us": 2.0}],
            "matrix_decisions": [{"output": "mm", "selected": False, "reason": "existing FP32 matrix body; small-tile eligibility not met"},
                                 {"output": "mm_1", "selected": True, "reason": "aligned BF16 exact-order small tile; warpgroup shape/layout/resource admission refused"}],
            "attention_refusals": [{"group": "bmm", "reason": "head width 48 is not a supported attention width"}],
            "in_place_commits": {"arg0_1": "sub"}, "copied_commits": {},
            "execution_selection": {"requested": "compose", "selected": "common_static",
                                    "reasons": ["established reduction engine, extent or arithmetic does not match: sum_1"]}}


def test_rejections_quote_recorded_constraints_only():
    report = report_manifest(_manifest())
    by = {(r["category"], r["subject"]): r for r in report["rejections"]}
    assert by[("matrix", "mm")]["constraint"] == "existing FP32 matrix body; small-tile eligibility not met"
    assert by[("matrix_alternative", "mm_1")]["constraint"] == "warpgroup shape/layout/resource admission refused"
    assert by[("attention", "bmm")]["constraint"].startswith("head width 48")
    assert by[("execution_retention", "legacy_static (established emitter)")]["recorded_by"] == "manifest.execution_selection.reasons"
    for row in report["rejections"]:
        assert row["recorded_by"].startswith(("manifest.", "collector."))


def test_casts_temporaries_and_sync_are_static_counts():
    ir = {"operations": [{"name": "_to_copy", "target": "aten._to_copy.default", "inputs": ["arg1_1"]}],
          "values": {"arg1_1": {"dtype": "float32"}, "_to_copy": {"dtype": "bfloat16"}}, "metadata": {}}
    report = report_manifest(_manifest(), ir=ir)
    dm = report["data_movement"][0]
    assert len(dm["casts"]) == 1 and dm["casts"][0]["from_dtype"] == "float32" and dm["casts"][0]["to_dtype"] == "bfloat16"
    assert dm["casts"][0]["bytes_written"] == {"value": 128, "label": "static", "basis": "result layout"}
    assert dm["temporaries"]["count"]["value"] == 2 and dm["placed_outputs"]["names"] == ["mm_1"]
    assert dm["synchronization"]["barrier_count"]["value"] == 3
    assert dm["state_commits"]["in_place"]["value"] == 1


def test_costs_unmeasured_without_profile_and_estimates_labelled():
    report = report_manifest(_manifest())
    assert report["costs"]["status"] == "unmeasured"
    rows = report["costs"]["static_estimates"]["rows"]
    assert {r["output"]: r["estimate_us"]["label"] for r in rows} == {"mm": "estimate", "mm_1": "estimate"}
    assert report["slow_path_candidates"]["rows"] == []


def test_profile_rows_attach_measured_interval_costs():
    manifest = _manifest()
    rows = [{"phase": 0, "kind": "elementwise", "values": "_to_copy", "nanoseconds": 1000},
            {"phase": 1, "kind": "gemm", "values": "mm", "nanoseconds": 5000},
            {"phase": 2, "kind": "gemm", "values": "mm_1", "nanoseconds": 2000},
            {"phase": 3, "kind": "state_commit", "values": "state_commit", "nanoseconds": 500}]
    report = report_manifest(manifest, profile=rows)
    assert report["costs"]["status"] == "measured"
    matrices = {m["output"]: m for m in report["selections"]["matrices"]}
    assert matrices["mm"]["measured_interval_ms"]["value"] == pytest.approx(0.005)
    assert matrices["mm"]["measured_interval_ms"]["label"] == "measured"
    top = report["slow_path_candidates"]["rows"][0]
    assert top["phase"] == 1 and top["recorded_rejections"][0]["constraint"].startswith("existing FP32")
    # dependency schedules keep interval members together
    dep = dict(manifest, barrier_schedule="dependency", barrier_intervals=[[0, 1], [2]])
    by_phase = attribute(dep, normalize([{"phase": 0, "kind": "dependency_interval", "phases": [0, 1], "values": "a", "nanoseconds": 4000},
                                         {"phase": 1, "kind": "dependency_interval", "phases": [2], "values": "b", "nanoseconds": 1000}], dep)["intervals"])
    assert [(r["phase"], r["interval_ms"]) for r in by_phase] == [(0, 0.004), (1, 0.004), (2, 0.001)]


def test_timing_receipt_binding_requires_exact_source_hash():
    manifest = _manifest()
    receipt = {"rounds": [{"arm": "native:current", "profile": {"total_ms": 1.0, "top_intervals": [{"interval": 1, "ms": 0.7, "kinds": ["gemm"], "members": 0, "values": "mm"}]}}],
               "median_ms": {"native:current": 1.0}, "clock": "CUDA events", "artifact_bindings": {"current": {"source_hash": "x" * 64}}}
    report = report_manifest(manifest, profile=receipt)
    assert report["costs"]["binding"]["bound"] is False
    receipt["artifact_bindings"]["current"]["source_hash"] = manifest["source_hash"]
    assert report_manifest(manifest, profile=receipt)["costs"]["binding"]["bound"] is True


def test_region_programs_are_reported_per_region():
    child = _manifest()
    root = {"format": "latch-ops-cuda-loop-v1", "source_hash": "r" * 64, "execution": "one cooperative kernel; runtime loop",
            "regions": {"body": child, "optimizer": copy.deepcopy(child)}, "compiled_regions": ["body", "optimizer"],
            "runtime_loop": {"max_steps": 4, "storage_mode": "save_body_tape"}, "phases": [{"kind": "forward_loop", "names": ["forward_loop"]}],
            "loop_schedule_policy": {"selected": "pipelined", "reasons": ["flat saved-boundary operation regions expose complete stage effects"]}}
    report = report_manifest(root)
    assert report["structure"]["kind"] == "region_program"
    assert report["structure"]["generated_regions"]["value"] == 3
    assert {r["path"] for r in report["structure"]["regions"]} == {"program", "program/body", "program/optimizer"}
    assert {r["region"] for r in report["selections"]["matrices"]} == {"program/body", "program/optimizer"}
    text = render_markdown(report)
    assert "composed of 3 [static] regions" in text and "program/optimizer" in text
