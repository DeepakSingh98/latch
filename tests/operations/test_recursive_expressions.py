"""Opt-in recursive expression scheduling keeps the existing arithmetic bodies.

Run on the lane CPU worker. Native bitwise, poison and trajectory comparisons
remain required before enabling this diagnostic by default.
"""
import re

import pytest

from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.loop_pipeline import PIPELINED_MATRIX_ENGINE


def _graph(*, tape=False, matrix=False, reduction="mean"):
    shapes = {"x": [128, 128], "bias": [128], "target": [128, 128],
              "scaled": [128, 128], "scaled_bias": [128], "sum": [128, 128],
              "difference": [128, 128], "square": [128, 128], "result": [],
              "derivative": [128, 128]}
    inputs = ["x", "bias", "target"]
    operations = []
    if matrix:
        inputs = ["a", "b", "bias", "target"]
        shapes.update(a=[128, 128], b=[128, 128])
        operations.append({"name": "x", "target": "aten.mm.default", "args": [{"ref": "a"}, {"ref": "b"}]})
    for name, target, args in (
        ("scaled", "mul.Tensor", [{"ref": "x"}, 0.25]),
        ("scaled_bias", "mul.Tensor", [{"ref": "bias"}, 0.5]),
        ("sum", "add.Tensor", [{"ref": "scaled"}, {"ref": "scaled_bias"}]),
        ("difference", "sub.Tensor", [{"ref": "sum"}, {"ref": "target"}]),
        ("square", "pow.Tensor_Scalar", [{"ref": "difference"}, 2]),
        ("result", reduction + ".default", [{"ref": "square"}]),
        ("derivative", "mul.Tensor", [{"ref": "difference"}, 2.0]),
    ):
        operations.append({"name": name, "target": "aten." + target, "args": args})
    return {"inputs": inputs, "outputs": ["result", "derivative"] + (["difference"] if tape else []),
            "values": {name: {"shape": shape, "dtype": "float32"} for name, shape in shapes.items()},
            "operations": operations}


def _lower(graph, *, binding="store", group_rules=False, **options):
    return lower(graph, execution="compose", precision_policy="strict", staged=True, region_body=True,
                 output_bindings=[binding] + ["store"] * (len(graph["outputs"]) - 1),
                 barrier_schedule="dependency", group_rules=group_rules,
                 preserve_full_reductions=True, preserve_matrix_plans=True,
                 matrix_engine=dict(PIPELINED_MATRIX_ENGINE), split_counters="persistent", **options)


def _full_body(source):
    start = source.index("if (gridDim.x <= 1024) {")
    end = source.index("{", start) + 1
    depth = 1
    while depth:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
    return source[start:end]


@pytest.mark.parametrize("binding", ["store", "accumulate"])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_expression_inlining_changes_only_full_reduction_leaf_load(binding, reduction):
    graph = _graph(reduction=reduction)
    before = _lower(graph, binding=binding)
    after = _lower(graph, binding=binding, region_expression_inline=True)
    old_body, new_body = _full_body(before.source), _full_body(after.source)
    # Every synchronization, ticket/reset, partial, tree and final store stays
    # literally identical. Only the original leaf buffer load is substituted.
    replace_leaf = lambda body: re.sub(r"acc \+= [^\n]+;", "acc += LEAF;", body, count=1)
    assert old_body != new_body
    assert replace_leaf(old_body) == replace_leaf(new_body)
    assert "v_square[" not in new_body
    assert "v_x[" in new_body and "v_bias[" in new_body and "v_target[" in new_body
    old = next(p for p in before.manifest["phases"] if p.get("engine") == "multi_cta_full")
    new = next(p for p in after.manifest["phases"] if p.get("engine") == "multi_cta_full")
    for key in ("engine", "arithmetic", "reduction_elements", "counter_effects", "private_scratch",
                "internal_mutations", "writes", "requires_full_grid", "max_grid_blocks"):
        assert old[key] == new[key], key
    assert set(new["reads"]) == {"x", "bias", "target"}
    assert after.manifest["persistent_counter_words"] == before.manifest["persistent_counter_words"] == 1
    assert len(after.manifest["stages"]) < len(before.manifest["stages"])
    assert after.manifest["output_placement"]["result"]["mode"] == binding
    assert set(after.manifest["region_expression_policy"]["inlined"]) == {
        "scaled", "scaled_bias", "sum", "difference", "square"}


def test_published_tape_value_is_stored_and_read_after_its_dependency():
    graph = _graph(tape=True)
    program = _lower(graph, region_expression_inline=True)
    manifest = program.manifest
    assert "difference" not in manifest["region_expression_policy"]["inlined"]
    assert manifest["output_placement"]["difference"] == {"output": 2, "mode": "store", "zero_omitted": False}
    producer = next(i for i, p in enumerate(manifest["phases"]) if "difference" in p["writes"])
    consumer = next(i for i, p in enumerate(manifest["phases"]) if p.get("engine") == "multi_cta_full")
    intervals = {p: i for i, members in enumerate(manifest["barrier_intervals"]) for p in members}
    assert intervals[producer] < intervals[consumer]
    assert "v_difference[" in _full_body(program.source)


def test_published_view_preserves_its_source_storage():
    graph = _graph()
    graph["values"]["tape_view"] = {"shape": [16384], "dtype": "float32", "alias_of": "difference"}
    graph["operations"].append({"name": "tape_view", "target": "aten.view.default",
                                "args": [{"ref": "difference"}, [16384]]})
    graph["outputs"].append("tape_view")
    manifest = _lower(graph, region_expression_inline=True).manifest
    assert "difference" not in manifest["region_expression_policy"]["inlined"]
    assert manifest["layouts"]["tape_view"]["root"] == "difference"


def test_partial_reduction_keeps_its_source_buffer():
    graph = _graph(reduction="sum")
    graph["values"]["result"]["shape"] = [128]
    graph["operations"][5] = {"name": "result", "target": "aten.sum.dim_IntList",
                                "args": [{"ref": "square"}, [0], False]}
    manifest = _lower(graph, region_expression_inline=True).manifest
    assert "square" not in manifest["region_expression_policy"]["inlined"]
    assert manifest["region_expression_policy"]["reduction_sources"] == []
    reduction = next(p for p in manifest["phases"] if p["kind"] == "reduction")
    assert "square" in reduction["reads"] and "square" in manifest["layouts"]


def test_expression_inlining_keeps_selected_matrix_plan_and_scratch_hazards():
    graph = _graph(matrix=True)
    before = _lower(graph)
    after = _lower(graph, region_expression_inline=True)
    old, = [p for p in before.manifest["phases"] if p["kind"] == "gemm"]
    new, = [p for p in after.manifest["phases"] if p["kind"] == "gemm"]
    assert old == new
    assert before._selection_contracts == after._selection_contracts
    assert "x" not in after.manifest["region_expression_policy"]["inlined"]
    assert set(new["internal_mutations"]) <= set(new["reads"]) & set(new["writes"])


def test_matrix_operand_never_becomes_a_virtual_expression():
    graph = _graph(matrix=True)
    graph["values"]["prepared"] = {"shape": [128, 128], "dtype": "float32"}
    graph["operations"].insert(0, {"name": "prepared", "target": "aten.mul.Tensor",
                                    "args": [{"ref": "a"}, 0.5]})
    graph["operations"][1]["args"][0] = {"ref": "prepared"}
    manifest = _lower(graph, region_expression_inline=True).manifest
    assert "prepared" not in manifest["region_expression_policy"]["inlined"]
    assert "prepared" in manifest["layouts"]


def test_recomputation_is_bounded_and_option_off_keeps_existing_source():
    graph = _graph()
    assert _lower(graph).source == _lower(graph, region_expression_inline=False).source
    assert "region_expression_policy" not in _lower(graph).manifest
    # A long arithmetic chain must retain intermediate materialization instead
    # of generating one unbounded expression at each reduction leaf.
    operations = []
    for i in range(30):
        name, source = f"chain_{i}", f"chain_{i - 1}" if i else "x"
        graph["values"][name] = {"shape": [128, 128], "dtype": "float32"}
        operations.append({"name": name, "target": "aten.mul.Tensor", "args": [{"ref": source}, 1.001]})
    graph["operations"][0]["args"][0] = {"ref": "chain_29"}
    graph["operations"] = operations + graph["operations"]
    manifest = _lower(graph, region_expression_inline=True).manifest
    report = manifest["region_expression_policy"]
    assert all(cost <= 12 for cost in report["expression_costs"].values())
    assert len(set(f"chain_{i}" for i in range(30)) - set(report["inlined"])) >= 2


@pytest.mark.parametrize("consumer,uses,side_reader,shared_operand,view,group_rules", [
    ("silu", 3, True, False, False, False),
    ("tanh_backward", 3, True, False, False, False),
    ("sigmoid_backward", 3, True, False, False, False),
    ("silu_backward", 3, False, False, False, False),
    ("silu_backward", 3, False, False, True, False),
    ("silu_backward", 4, False, True, False, False),
    ("square", 3, True, False, False, False),
    ("cube", 3, False, False, False, "powx"),
])
@pytest.mark.parametrize("over_budget", [False, True])
def test_repeated_renderer_operands_enforce_total_expression_budget(
        consumer, uses, side_reader, shared_operand, view, group_rules, over_budget):
    # Each multiplication costs one. At exactly 24 rendered uses the complete
    # chain is admitted; adding one producer must retain its final materialized
    # value. An extra consumer counts together with repeated uses in one body.
    depth = 24 // uses + int(over_budget)
    values = {name: {"shape": [4], "dtype": "float32"} for name in ("x", "grad", "result")}
    operations = []
    source = "x"
    for i in range(depth):
        name = f"chain_{i}"
        values[name] = {"shape": [4], "dtype": "float32"}
        operations.append({"name": name, "target": "aten.mul.Tensor", "args": [{"ref": source}, 1.001]})
        source = name
    final_producer = source
    if view:
        values["view"] = {"shape": [4], "dtype": "float32", "alias_of": source}
        operations.append({"name": "view", "target": "aten.view.default", "args": [{"ref": source}, [4]]})
        source = "view"
    if consumer in {"square", "cube"}:
        target, args = "pow.Tensor_Scalar", [{"ref": source}, 2 if consumer == "square" else 3]
    elif consumer == "silu":
        target, args = "silu.default", [{"ref": source}]
    else:
        target = consumer + ".default"
        args = [{"ref": source if shared_operand else "grad"}, {"ref": source}]
    operations.append({"name": "result", "target": "aten." + target, "args": args})
    outputs = ["result"]
    if side_reader:
        values["side"] = {"shape": [4], "dtype": "float32"}
        operations.append({"name": "side", "target": "aten.neg.default", "args": [{"ref": source}]})
        outputs.append("side")
    graph = {"inputs": ["x", "grad"], "outputs": outputs, "values": values, "operations": operations}
    program = _lower(graph, region_expression_inline=True, group_rules=group_rules)
    manifest = program.manifest
    report = manifest["region_expression_policy"]
    assert (final_producer in report["inlined"]) is not over_budget
    if over_budget:
        assert any(final_producer in phase["writes"] for phase in manifest["phases"])
    else:
        assert report["expression_costs"][final_producer] * uses == report["total_use_cost_limit"]
        assert not any(final_producer in phase["writes"] for phase in manifest["phases"])
    assert manifest["outputs"] == outputs
    assert set(manifest["output_placement"]) == set(outputs)


@pytest.mark.parametrize("option", [1, "yes", None])
def test_expression_inline_option_requires_boolean(option):
    with pytest.raises(LoweringError, match="must be boolean"):
        _lower(_graph(), region_expression_inline=option)


def test_expression_inline_option_requires_a_staged_region():
    with pytest.raises(LoweringError, match="staged region"):
        lower(_graph(), execution="compose", region_expression_inline=True)
