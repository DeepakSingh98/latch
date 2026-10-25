"""CPU-only lowering regressions for the opt-in vector4 LayerNorm phase.

These exercise ordinary lowering, publication and selection. They do not compile
CUDA or establish arithmetic equality; actual-input component receipts do that.
Run after the production header has been installed from its measured candidate.
"""
# Accepted c24: two independent rows per CTA; affine serial chains and
# their unchanged combine tree are separate phases. Numerical gates live
# in the component/device qualifications and are not changed here.
from copy import deepcopy

import pytest

from megakernel_lab.operation_compiler.lower import LoweringError, lower
from .test_torch_layer_norm import fixture as metadata_fixture
from megakernel_lab.operation_compiler.torch_layer_norm import CONTRACT, KEY


def fixture(prefix=""):
    graph = metadata_fixture(prefix)
    # Ordinary inference lowering uses a flat sequence of output tensor names;
    # dictionary outputs are the distinct complete-training state ABI.
    graph["outputs"] = [prefix + name for name in ("out", "mean", "rstd")]
    return graph


def emit(graph, **kwargs):
    return lower(graph, execution="compose", **kwargs)


def phases(program):
    return [p for p in program.manifest["phases"] if p.get("row_arithmetic") == CONTRACT]


def _append(graph, name, target, args, shape):
    stride, strides = 1, []
    for size in reversed(shape):
        strides.insert(0, stride); stride *= size
    graph["values"][name] = dict(name=name, shape=shape, strides=strides, dtype="float32",
                                  storage_offset=0, alias_of=None, output_roles=[])
    refs = [arg["ref"] for arg in args if isinstance(arg, dict) and "ref" in arg]
    graph["operations"].append(dict(name=name, target=target, args=args, kwargs={},
        inputs=refs, outputs=[name], reads=refs, writes=[name], effect="pure"))


def test_unselected_equation_keeps_default_source_and_no_header():
    graph = fixture(); graph["metadata"] = {}
    before = deepcopy(graph)
    first = emit(graph)
    assert not phases(first)
    assert "torch_layer_norm.cuh" not in first.manifest["header_hashes"]
    # Lowering an explicitly selected program must not mutate process defaults.
    assert phases(emit(fixture()))
    assert emit(graph).source == first.source
    assert graph == before


def test_one_phase_publishes_all_three_outputs_and_exact_resources():
    graph = fixture(); before = deepcopy(graph)
    program = emit(graph)
    phase, = phases(program)
    assert phase["names"] == phase["writes"] == ["out", "mean", "rstd"]
    assert phase["reads"] == ["x", "gain", "bias"]
    assert (phase["threads"], phase["shared_bytes"], phase["work_items"], phase["reduction_elements"]) == (256, 96, 2, 8)
    assert "torch_ln_vector4::torch_ln_row2(" in program.source
    assert "torch_layer_norm.cuh" in program.manifest["header_hashes"]
    body, = [b for b in program.manifest["execution_plan"]["bodies"]
             if b["body"] == "torch_cuda_layer_norm_vector4"]
    assert {w["name"] for w in body["writes"]} == {"out", "mean", "rstd"}
    assert {r["name"] for r in body["reads"]} == {"x", "gain", "bias"}
    assert (body["threads"], body["shared_bytes"], body["work_items"]) == (256, 96, 2)
    for name, shape in (("out", [3, 8]), ("mean", [3, 1]), ("rstd", [3, 1])):
        assert list(program.manifest["layouts"][name]["shape"]) == shape
    assert graph == before


def test_saved_statistics_order_a_later_consumer():
    graph = fixture()
    _append(graph, "stat_consumer", "aten.add.Tensor", [{"ref": "mean"}, {"ref": "rstd"}], [3, 1])
    graph["outputs"] = ["out", "stat_consumer"]
    program = emit(graph, compose_selection="common")
    bodies = program.manifest["execution_plan"]["bodies"]
    producer, = [b for b in bodies if b["body"] == "torch_cuda_layer_norm_vector4"]
    consumer, = [b for b in bodies if any(w["name"] == "stat_consumer" for w in b["writes"])]
    assert producer["index"] in consumer["dependencies"]
    assert {r["root"] for r in consumer["reads"]} >= {"mean", "rstd"}
    assert {w["root"] for w in producer["writes"]} >= {"mean", "rstd"}


def test_auto_selection_retains_measured_common_phase():
    program = emit(fixture())
    selection = program.manifest["execution_selection"]
    assert selection["selected"] == "common_static"
    assert any("LayerNorm arithmetic requires common lowering" in reason for reason in selection["reasons"])
    assert len(phases(program)) == 1


def test_primary_shape_keeps_all_rows_and_saved_statistics():
    graph = fixture()
    for value in graph["values"].values():
        old = value["shape"]
        shape = [16384, 768] if old == [3, 8] else [16384, 1] if old == [3, 1] else [768]
        value["shape"] = shape
        value["strides"] = [shape[1], 1] if len(shape) == 2 else [1]
    next(op for op in graph["operations"] if op["name"] == "var")["args"][1] = 768
    program = emit(graph)
    phase, = phases(program)
    assert phase["work_items"] == 8192 and phase["reduction_elements"] == 768
    assert phase["writes"] == ["out", "mean", "rstd"]
    assert "row < 16384LL" in program.source


def test_row_fusion_does_not_rewrite_opaque_welford_body():
    graph = fixture()
    _append(graph, "tail", "aten.mul.Tensor", [{"ref": "out"}, 0.5], [3, 8])
    graph["outputs"].append("tail")
    program = emit(graph, compose_selection="common", row_fusion=True)
    phase, = phases(program)
    assert phase["writes"] == ["out", "mean", "rstd"]
    assert "row_reduction" not in phase
    assert all("mean" not in group.get("names", ()) for group in program.manifest.get("row_fusions", ()))


def test_escaping_center_refuses_replacement_and_preserves_old_equation():
    graph = fixture(); graph["outputs"].append("center")
    original = deepcopy(graph); original["metadata"] = {}
    flagged, unflagged = emit(graph), emit(original)
    assert not phases(flagged)
    assert "torch_layer_norm.cuh" not in flagged.manifest["header_hashes"]
    assert flagged.source == unflagged.source


def test_two_independent_groups_keep_distinct_publications():
    graph, second = fixture("a_"), fixture("b_")
    graph["values"].update(second["values"])
    graph["operations"].extend(second["operations"])
    graph["inputs"].extend(second["inputs"])
    graph["outputs"].extend(second["outputs"])
    program = emit(graph, compose_selection="common", row_fusion=True)
    selected = phases(program)
    assert len(selected) == 2
    assert {tuple(p["writes"]) for p in selected} == {
        ("a_out", "a_mean", "a_rstd"), ("b_out", "b_mean", "b_rstd")}
    assert all(p["shared_bytes"] == 96 and p["threads"] == 256 for p in selected)


def test_operation_names_may_differ_from_explicit_output_names():
    graph = fixture()
    for op in graph["operations"]:
        op["output"] = op["outputs"][0]
        op["name"] = "operation_" + op["name"]
    phase, = phases(emit(graph))
    assert phase["layer_norm_plan"]["anchor"] == "operation_mean"
    assert phase["writes"] == ["out", "mean", "rstd"]


def test_selected_legacy_mode_refuses_instead_of_silently_changing_arithmetic():
    with pytest.raises(LoweringError, match="LayerNorm arithmetic requires"):
        lower(fixture(), execution="legacy")


def test_unknown_contract_is_not_silently_ignored():
    graph = fixture(); graph["metadata"][KEY] = "unknown"
    with pytest.raises(ValueError, match="unknown explicit LayerNorm"):
        emit(graph)
