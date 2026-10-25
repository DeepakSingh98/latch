"""CPU source contracts for fixed legacy and online common vocabulary statistics.

These synthetic IRs require no model execution and make no numerical claim.
"""
from __future__ import annotations

import functools
import importlib

import pytest

from megakernel_lab.operation_compiler import compose, vocab_head
from megakernel_lab.operation_compiler.execution_plan import execution_scope
from megakernel_lab.operation_compiler.lower import lower
from .test_dense_exec import _graph
from .test_ffn_retain import explicit_graph


def _vocabulary_graph():
    graph = explicit_graph()

    def stats(value, op, r):
        value("logits", (4096, 2048), dtype="bfloat16", role="input")
        value("row_stats", (4096, 2))
        op("row_stats", vocab_head.TARGETS["stats"], [r("logits"), 2048])
        return ["logits"], ["row_stats"]

    other = _graph(stats)
    graph["values"].update(other["values"])
    for key in ("operations", "inputs", "outputs"):
        graph[key].extend(other[key])
    return graph


def _stats(program):
    return [p for p in program.manifest["phases"] if p.get("kind") == "vocab_rows"]


@pytest.mark.parametrize("execution", ["legacy", "compose"])
def test_established_execution_keeps_fixed_vocabulary_tree(execution):
    graph = _vocabulary_graph()
    program = lower(graph, execution=execution)
    legacy = lower(graph, execution="legacy")
    rows = _stats(program)
    assert len(rows) == 1
    assert "fixed CTA tree" in rows[0]["arithmetic"]
    assert rows[0].get("engine") != "vocab_stats_online_warp"
    assert program.source == legacy.source
    assert program.cache_key == legacy.cache_key
    assert program.manifest["header_hashes"] == legacy.manifest["header_hashes"]
    if execution == "compose":
        selection = program.manifest["execution_selection"]
        assert selection["selected"] == "legacy_static"
        assert selection["retained_protocol"]["matrix_bodies"]
    assert "deferred_vocabulary_stats" not in program.manifest
    assert vocab_head._STATS_PROBE.get() is None


@pytest.mark.parametrize("refusal", ["unfused", "body_proof"])
def test_common_selection_reemits_the_online_body(monkeypatch, refusal):
    options = {"fuse_elementwise": False} if refusal == "unfused" else {}
    if refusal == "body_proof":
        monkeypatch.setattr(compose, "verify_legacy_long_protocol",
                            lambda *args: (None, "injected mismatched body contract"))
    graph = _vocabulary_graph()
    automatic = lower(graph, execution="compose", **options)
    forced = lower(graph, execution="compose", compose_selection="common", **options)
    selection = automatic.manifest["execution_selection"]
    assert selection["selected"] == "common_static"
    assert selection["common_body_upgrade"]["vocabulary_stats"] == [
        {"output": "v_row_stats", "rows": 4096, "padded": 2048, "vocabulary": 2048}]
    assert _stats(automatic)[0]["engine"] == "vocab_stats_online_warp"
    assert "declared reassociation" in _stats(automatic)[0]["arithmetic"]
    assert automatic.source == forced.source
    assert automatic.cache_key == forced.cache_key
    assert automatic.manifest["header_hashes"] == forced.manifest["header_hashes"]
    assert automatic.manifest["phases"] == forced.manifest["phases"]
    assert "deferred_vocabulary_stats" not in automatic.manifest
    assert vocab_head._STATS_PROBE.get() is None


def test_execution_scope_preserves_common_region_stats_and_restores_legacy():
    args = ("x", "float", 8, 2048, 2047, "stats")
    legacy = vocab_head._emit_stats(*args)
    assert "fixed CTA tree" in legacy[2]["arithmetic"]
    with execution_scope("compose"):
        common = vocab_head._emit_stats(*args)
        assert common == vocab_head._emit_stats_online(*args)
        with execution_scope("legacy"):
            assert vocab_head._emit_stats(*args) == legacy
        assert vocab_head._emit_stats(*args) == common
    assert vocab_head._emit_stats(*args) == legacy


@pytest.mark.parametrize("failure", [False, True])
def test_probe_context_restores_on_success_and_failure(monkeypatch, failure):
    module = importlib.import_module('megakernel_lab.operation_compiler.lower')
    original = module._lower_serial

    @functools.wraps(original)
    def emitter(*args, **kwargs):
        assert vocab_head._STATS_PROBE.get() == []
        if failure:
            raise RuntimeError("injected vocabulary emitter failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_lower_serial", emitter)
    outer = ["outer probe"]
    token = vocab_head._STATS_PROBE.set(outer)
    try:
        with execution_scope("compose"):
            if failure:
                with pytest.raises(RuntimeError, match="injected vocabulary emitter failure"):
                    compose.lower_composed_flat(_vocabulary_graph(), selection_probe=True)
            else:
                probe, _ = compose.lower_composed_flat(_vocabulary_graph(), selection_probe=True)
                assert len(probe.manifest["deferred_vocabulary_stats"]) == 1
                assert "fixed CTA tree" in _stats(probe)[0]["arithmetic"]
        assert vocab_head._STATS_PROBE.get() is outer
    finally:
        vocab_head._STATS_PROBE.reset(token)


def test_program_without_vocabulary_has_no_new_metadata():
    for mode in ("auto", "common"):
        program = lower(explicit_graph(), execution="compose", compose_selection=mode)
        assert not _stats(program)
        assert "deferred_vocabulary_stats" not in program.manifest
        assert "vocabulary_stats" not in program.manifest.get("execution_selection", {}).get("common_body_upgrade", {})
