"""Structural contracts for the optional practical square-sum association."""
import copy
import math
import struct

import pytest

from megakernel_lab.operation_compiler import practical_bf16, square_sum_association as S


def fixture(count=7):
    values, operations = {}, []
    def value(name, shape):
        values[name] = dict(name=name, shape=shape, strides=[1] if shape else [], dtype="float32",
            role="temporary", alias_of=None, storage_offset=0, producer=name, output_roles=[])
    def op(name, target, args):
        refs = [x["ref"] for x in args]
        operations.append(dict(name="operation_"+name, target=target, args=args, kwargs={},
            outputs=[name], inputs=refs, reads=refs, writes=[name], dependencies=refs, effect="pure"))
    for i in range(count):
        value(f"input{i}", [3]); value(f"square{i}", [3]); value(f"term{i}", [])
        op(f"square{i}", "aten.mul.Tensor", [{"ref": f"input{i}"}]*2)
        op(f"term{i}", "aten.sum.default", [{"ref": f"square{i}"}])
    root = "term0"
    for i in range(1, count):
        name = f"subtotal{i}"
        value(name, [])
        op(name, "aten.add.Tensor", [{"ref": root}, {"ref": f"term{i}"}])
        root = name
    return dict(values=values, operations=operations, inputs=[f"input{i}" for i in range(count)],
                outputs={"result": root}, metadata={}, state_writes={})


def bound(count=7):
    return practical_bf16.bind(fixture(count), "practical")


def additions(graph):
    return [op for op in graph["operations"] if op["target"] == "aten.add.Tensor"]


def test_strict_and_disabled_graphs_are_identical_and_enabling_requires_permission():
    graph = fixture()
    before = copy.deepcopy(graph)
    assert S.rewrite(graph) == (graph, None)
    assert graph == before
    with pytest.raises(ValueError, match="practical execution permission"):
        S.rewrite(graph, option='balanced')
    with pytest.raises(ValueError, match="None or balanced"):
        S.rewrite(bound(), option='invalid')
    for policy in (None, "strict"):
        with pytest.raises(ValueError, match="precision_policy='practical'"):
            S.validate_option(policy, "balanced")


@pytest.mark.parametrize("count", [4, 5, 7, 75, 2048])
def test_balanced_tree_preserves_leaves_typed_operations_and_caller_graph(count):
    graph = bound(count); before = copy.deepcopy(graph)
    rewritten, record = S.rewrite(graph, option='balanced')
    assert graph == before and rewritten["values"] == graph["values"]
    assert rewritten["outputs"] == graph["outputs"] and len(additions(rewritten)) == count-1
    assert record == S.validate(rewritten)
    assert record["trees"][0]["leaves"] == [dict(sum=f"term{i}", square=f"square{i}",
        operand=f"input{i}", shape=[3]) for i in range(count)]
    depth = {f"term{i}": 0 for i in range(count)}
    for op in additions(rewritten):
        left, right = [x["ref"] for x in op["args"]]
        assert left in depth and right in depth
        depth[op["outputs"][0]] = 1 + max(depth[left], depth[right])
    assert depth[rewritten["outputs"]["result"]] == math.ceil(math.log2(count))
    again, retained = S.rewrite(rewritten, option='balanced')
    assert again is rewritten and retained == record and retained is not record
    assert S.rewrite(rewritten)[0] is rewritten


@pytest.mark.parametrize("count", [1, 2, 3])
def test_already_balanced_short_chain_is_unchanged(count):
    graph = bound(count)
    result, record = S.rewrite(graph, option='balanced')
    assert result is graph and record is None


@pytest.mark.parametrize("case", ["nested_output", "alias", "extra_reader", "negative_leaf", "bf16", "partial_sum"])
def test_declines_escaping_or_non_norm_scalar_chains(case):
    graph = bound(); ops = {op["outputs"][0]: op for op in graph["operations"]}
    if case == "nested_output": graph["outputs"]["nested"] = [{"value": "subtotal1"}]
    elif case == "alias": graph["values"]["alias"] = dict(graph["values"]["subtotal1"], name="alias", alias_of="subtotal1")
    elif case == "extra_reader":
        graph["operations"].append(dict(name="reader", target="aten.neg.default",
            args=[{"ref": "subtotal1"}], outputs=["reader"], kwargs={}))
    elif case == "negative_leaf": ops["term0"]["target"] = "aten.neg.default"
    elif case == "bf16": graph["values"]["square0"]["dtype"] = "bfloat16"
    else: ops["term0"]["target"] = "aten.sum.dim_IntList"
    before = copy.deepcopy(graph)
    result, record = S.rewrite(graph, option='balanced')
    assert result is graph and record is None and graph == before


@pytest.mark.parametrize("case", ["drop_leaf", "change_order", "wrong_dtype", "escape", "permission"])
def test_proof_rejects_modified_arithmetic_or_inventory(case):
    graph, _ = S.rewrite(bound(), option='balanced')
    row = graph["metadata"][S.KEY]["trees"][0]
    if case == "drop_leaf": row["leaves"].pop()
    elif case == "change_order": additions(graph)[0]["args"].reverse()
    elif case == "wrong_dtype": graph["values"][row["root"]]["dtype"] = "bfloat16"
    elif case == "escape": graph["outputs"]["subtotal"] = row["additions"][0]["name"]
    else: graph["metadata"].pop(practical_bf16.KEY)
    with pytest.raises(ValueError): S.validate(graph)


def test_adversarial_positive_scalar_terms_follow_declared_fp32_tree():
    graph, _ = S.rewrite(bound(75), option='balanced')
    fp32 = lambda x: struct.unpack("f", struct.pack("f", x))[0]
    terms = [1e8]+[1.]*74
    result = {f"term{i}": term for i, term in enumerate(terms)}
    for op in additions(graph):
        result[op["outputs"][0]] = fp32(sum(result[x["ref"]] for x in op["args"]))
    serial = terms[0]
    for value in terms[1:]: serial = fp32(serial + value)
    assert abs(result[graph["outputs"]["result"]] - sum(terms)) < abs(serial - sum(terms))


def long_fixture():
    graph = bound(4)
    for key in ("input0", "square0"):
        graph["values"][key]["shape"] = [16387]
    return graph


def test_only_full_scalar_fp32_square_reductions_select_thread_lanes():
    graph, record = S.rewrite(long_fixture(), option="balanced")
    assert record["thread_lanes"] == 8
    assert record["full_sums"] == [dict(sum="term0", square="square0", operand="input0", shape=[16387])]
    assert S.selected_full_sums(graph) == {"term0"}
    # A batch of per-matrix norms (e.g. an optimizer's captured normalization)
    # has retained dimensions and is deliberately outside full-scalar matching.
    per_matrix = long_fixture()
    per_matrix["values"]["term0"]["shape"] = [8, 1, 1]
    unchanged, _ = S.rewrite(per_matrix, option="balanced")
    assert unchanged is per_matrix


def test_all_lane_tail_coordinates_are_covered_once_for_odd_extents():
    for extent in (1, 255, 256, 2047, 2048, 2049, 16387):
        covered = []
        for thread in range(256):
            r = thread
            while r + 7*256 < extent:
                covered.extend(r + lane*256 for lane in range(8))
                r += 8*256
            covered.extend(r + lane*256 for lane in range(8) if r + lane*256 < extent)
        assert sorted(covered) == list(range(extent))


def test_manifest_and_distributed_inventory_fail_closed_on_missing_lanes():
    from types import SimpleNamespace
    graph, record = S.rewrite(long_fixture(), option="balanced")
    program = SimpleNamespace(source=S.marker("term0"), manifest={"execution_precision": practical_bf16.describe()})
    S.annotate_program(graph, program)
    S.validate_manifest(program.manifest, program.source)
    receipt = {S.KEY: record}
    S.validate_artifact(receipt, graph, [program.manifest])
    for case in ("missing_lane", "duplicate_region", "missing_permission", "source_marker"):
        changed = copy.deepcopy(program.manifest)
        if case == "missing_lane":
            changed["square_sum_lane_outputs"] = []
            with pytest.raises(ValueError): S.validate_artifact(receipt, graph, [changed])
        elif case == "duplicate_region":
            with pytest.raises(ValueError): S.validate_artifact(receipt, graph, [changed, changed])
        elif case == "missing_permission":
            changed.pop(S.KEY)
            with pytest.raises(ValueError): S.validate_artifact(receipt, graph, [changed])
        else:
            with pytest.raises(ValueError): S.validate_manifest(changed, "")


def test_partitioned_record_validates_without_unrelated_producers():
    graph, record = S.rewrite(long_fixture(), option="balanced")
    empty = dict(values={}, operations=[], inputs=[], outputs=[], metadata={
        "distributed_event": 0, practical_bf16.KEY: practical_bf16.describe(), S.KEY: record})
    assert S.validate(empty) == record
    assert S.selected_full_sums(empty) == set()


def test_normal_lowerer_emits_declared_lanes_and_keeps_default_arithmetic():
    from megakernel_lab.operation_compiler.lower import lower
    graph = long_fixture()
    graph["outputs"] = [graph["outputs"]["result"]]
    for operation in graph["operations"]:
        operation["name"] = operation["outputs"][0]
    # An operation-only fixture avoids model capture and device execution.
    baseline = lower(copy.deepcopy(graph), execution="compose", precision_policy="practical")
    changed = lower(graph, execution="compose", precision_policy="practical", square_sum_association="balanced")
    assert S.KEY not in baseline.manifest and S.marker("term0") not in baseline.source
    assert changed.manifest["square_sum_lane_outputs"] == ["term0"]
    assert "lane7 += latch_term" in changed.source
    S.validate_manifest(changed.manifest, changed.source)
    with pytest.raises(ValueError, match="precision_policy='practical'"):
        lower(fixture(), execution="compose", square_sum_association="balanced")
