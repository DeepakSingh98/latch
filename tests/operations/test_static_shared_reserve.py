"""Common-flat static headroom contracts; CPU source/planner checks only."""
from __future__ import annotations

import copy

import pytest

from megakernel_lab.operation_compiler import p5c_matrix
from megakernel_lab.operation_compiler.compose import static_shared_budget
from megakernel_lab.operation_compiler.lower import GeneratedProgram, LoweringError, lower
from megakernel_lab.operation_compiler.matrix_adapters import dense
from megakernel_lab.operation_compiler.matrix_adapters.bodies import BY_NAME, _popsc_spec


def graph_for(problem):
    return {
        "inputs": ["A", "B"], "outputs": ["C"], "state_writes": {}, "metadata": {},
        "values": {
            "A": {"shape": problem.a_shape(), "strides": problem.a_strides(), "dtype": "bfloat16", "role": "input"},
            "B": {"shape": problem.b_shape(), "strides": problem.b_strides(), "dtype": "bfloat16", "role": "input"},
            "C": {"shape": problem.out_shape(), "dtype": "bfloat16"},
        },
        "operations": [{"name": "C", "target": "aten.mm.default",
                        "args": [{"ref": "A"}, {"ref": "B"}], "kwargs": {}}],
    }


def spec_for(problem):
    return dict(_popsc_spec(problem, "A", "B", "v_C"), engine="p5c",
                roots=[("A", 0), ("B", 0)], ws_ok=True)


def compile_graph(graph, **options):
    return lower(copy.deepcopy(graph), execution="compose", compose_selection="common", **options)


@pytest.mark.parametrize("rows", [64, 128])
def test_exact_five_to_four_stage_budget(rows, monkeypatch):
    spec = spec_for(dense(rows, 128, 384))
    old = p5c_matrix.flat_ws_plan(spec, 132)
    assert (old["nst"], old["shared_bytes"]) == (5, 230400)
    assert static_shared_budget(8192) == 224256
    monkeypatch.setattr(p5c_matrix, "P8WS_SHARED_LIMIT", 224256)
    assert p5c_matrix.flat_ws_plan(spec, 132) is None  # legacy low-cap refusal
    monkeypatch.setattr(p5c_matrix, "P8WS_FIT_SHARED_LIMIT", True)
    new = p5c_matrix.flat_ws_plan(spec, 132)
    assert (new["nst"], new["shared_bytes"]) == (4, 197632)
    assert {k: v for k, v in new.items() if k not in ("nst", "shared_bytes")} == {
        k: v for k, v in old.items() if k not in ("nst", "shared_bytes")}
    before = p5c_matrix.group_code([spec], [old], [0], lambda name: "v_" + name)
    after = p5c_matrix.group_code([spec], [new], [0], lambda name: "v_" + name)
    assert after == before.replace("p8ws::group<0,5,", "p8ws::group<0,4,")


def test_provider_uses_same_dynamic_cap(monkeypatch):
    problem = dense(1024, 32000, 384)
    old = BY_NAME["p8ws"].plans(problem, 132)
    from megakernel_lab.operation_compiler.matrix_adapters.bodies import Refusal
    with pytest.raises(Refusal, match="plan not admitted"):
        BY_NAME["p8ws"].plans(problem, 132, shared_limit=224256)
    monkeypatch.setattr(p5c_matrix, "P8WS_FIT_SHARED_LIMIT", True)
    new = BY_NAME["p8ws"].plans(problem, 132, shared_limit=224256)
    assert old and new
    assert all(p["nst"] == 5 and p["shared_bytes"] == 230400 for p in old)
    assert all(p["nst"] == 4 and p["shared_bytes"] == 197632 for p in new)


def test_common_program_cap_and_zero_identity(monkeypatch):
    # A generic square product admits the native five-stage flat engine.
    # Vocabulary-scale products may legitimately prefer a wider four-stage
    # tile, so they cannot establish this particular pipeline transition.
    graph = graph_for(dense(1024, 1024, 1024))
    ordinary = compile_graph(graph)
    zero = compile_graph(graph, static_shared_reserve_bytes=0)
    assert (ordinary.source, ordinary.manifest) == (zero.source, zero.manifest)
    assert ordinary.manifest["shared_bytes"] == 230400
    assert ordinary.manifest["phases"][0]["matrix_plan"]["nst"] == 5
    before = {key: getattr(p5c_matrix, key) for key in
              ("P4B_SHARED_LIMIT", "P5C_SHARED_LIMIT", "P8WS_SHARED_LIMIT", "P9WS_SHARED_LIMIT", "P8WS_FIT_SHARED_LIMIT")}
    reserved = compile_graph(graph, static_shared_reserve_bytes=8192)
    assert reserved.manifest["shared_bytes"] == 197632
    assert reserved.manifest["phases"][0]["matrix_plan"]["nst"] == 4
    assert all(p.get("shared_bytes", 0) <= 224256 for p in reserved.manifest["phases"])
    budget = reserved.manifest["shared_memory_budget"]
    assert budget["total_optin_limit_bytes"] == 232448
    assert budget["dynamic_limit_bytes"] + budget["static_reserve_bytes"] == 232448
    assert budget["dynamic_required_bytes"] == 197632
    assert "actual compiled static plus dynamic" in budget["admission"]
    assert before == {key: getattr(p5c_matrix, key) for key in before}
    after = compile_graph(graph)
    assert (ordinary.source, ordinary.manifest) == (after.source, after.manifest)
    assert "shared_memory_budget" not in ordinary.manifest
    # Historical flat_ws_plan first chose min(5,K/64), then refused if it did
    # not fit. At the unchanged default limit its entire emitted program must
    # remain identical, including the default selection/manifest.
    current = p5c_matrix.flat_ws_plan
    def historical(spec, grid, rows=None):
        shared = min(p5c_matrix.P8WS_STAGES, spec["k"] // 64) * 32768 + p5c_matrix.P8WS_STAGING * 32768 + 1024
        return None if shared > p5c_matrix.P8WS_SHARED_LIMIT else current(spec, grid, rows)
    monkeypatch.setattr(p5c_matrix, "flat_ws_plan", historical)
    frozen = compile_graph(graph)
    assert (ordinary.source, ordinary.manifest) == (frozen.source, frozen.manifest)


@pytest.mark.parametrize("reserve", [True, False, -1, 1.0, "8192", None, 232448, 215041])
def test_invalid_reserve_refuses_before_lowering(reserve):
    with pytest.raises(LoweringError, match="static_shared_reserve_bytes"):
        compile_graph(graph_for(dense(64, 128, 384)), static_shared_reserve_bytes=reserve)


@pytest.mark.parametrize("options", [
    {"execution": "legacy"}, {"policy": "reference"}, {"retain_all": True},
    {"target_arch": "sm_80"}, {"region_body": False}, {"staged": False},
])
def test_unsupported_scope_refuses(options):
    with pytest.raises(LoweringError, match="standalone static common compose"):
        lower(graph_for(dense(64, 128, 384)),
              **{"execution": "compose", "static_shared_reserve_bytes": 8192, **options})


def test_dynamic_scope_refuses():
    graph = graph_for(dense(64, 128, 384))
    graph["metadata"]["dynamic_batch"] = {"maximum": 64}
    with pytest.raises(LoweringError, match="standalone static common compose"):
        compile_graph(graph, static_shared_reserve_bytes=8192)


def test_unbudgeted_body_refuses_and_restores_configuration(monkeypatch):
    import importlib
    lower_module = importlib.import_module("megakernel_lab.operation_compiler.lower")
    limit = p5c_matrix.P8WS_SHARED_LIMIT
    fitting = p5c_matrix.P8WS_FIT_SHARED_LIMIT
    def unbudgeted(*args, **kwargs):
        assert p5c_matrix.P8WS_SHARED_LIMIT == 224256
        assert p5c_matrix.P8WS_FIT_SHARED_LIMIT is True
        assert kwargs["matrix_provider"].resources.dynamic_shared_limit == 224256
        return GeneratedProgram("", {"shared_bytes": 224257})
    monkeypatch.setattr(lower_module, "_lower_serial", unbudgeted)
    # compose validates its keyword interface from the real emitter signature.
    import inspect
    unbudgeted.__signature__ = inspect.signature(original_serial)
    with pytest.raises(LoweringError, match="needs 224257 dynamic shared bytes"):
        compile_graph(graph_for(dense(64, 128, 384)), static_shared_reserve_bytes=8192)
    assert p5c_matrix.P8WS_SHARED_LIMIT == limit
    assert p5c_matrix.P8WS_FIT_SHARED_LIMIT is fitting


# Capture the real interface before a refusal test installs its fake emitter.
from megakernel_lab.operation_compiler.lower import _lower_serial as original_serial


def test_insufficient_two_stage_ring_refuses(monkeypatch):
    monkeypatch.setattr(p5c_matrix, "P8WS_SHARED_LIMIT", 132095)
    monkeypatch.setattr(p5c_matrix, "P8WS_FIT_SHARED_LIMIT", True)
    assert p5c_matrix.flat_ws_plan(spec_for(dense(64, 128, 384)), 132) is None
