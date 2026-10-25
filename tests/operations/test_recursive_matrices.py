"""Recursive output placement retains ordinary FP32 matrix calculations.

Run on the lane CPU worker. Native trajectory and poison checks remain the
independent validation of the generated accumulating store and counter protocol.
"""
import pytest

from megakernel_lab.operation_compiler.lower import LoweringError, TensorLayout, lower
from megakernel_lab.operation_compiler.loop_pipeline import PIPELINED_MATRIX_ENGINE
from megakernel_lab.operation_compiler.matrix_codegen import fp32_async, fp32_selected
from megakernel_lab.operation_compiler.matrix_select import SELECTION_GRID


def _graph(m, n, k):
    return {"inputs": ["a", "b"], "outputs": ["result"], "values": {
        "a": {"shape": [m, k], "strides": [1, m], "dtype": "float32"},
        "b": {"shape": [k, n], "strides": [n, 1], "dtype": "float32"},
        "result": {"shape": [m, n], "strides": [n, 1], "dtype": "float32"}},
        "operations": [{"name": "result", "target": "aten.mm.default",
                        "args": [{"ref": "a"}, {"ref": "b"}]}]}


@pytest.mark.parametrize("shape", [(256, 256, 256), (512, 256, 1024), (128, 32, 128)])
@pytest.mark.parametrize("binding", ["store", "accumulate"])
def test_staged_matrix_plan_and_operands_match_actual_ordered_donor(shape, binding):
    graph = _graph(*shape)
    donor = lower(graph, execution="legacy", schedule="reference")
    staged = lower(graph, execution="compose", staged=True, region_body=True,
                   output_bindings=[binding], barrier_schedule="dependency",
                   matrix_engine=dict(PIPELINED_MATRIX_ENGINE), preserve_matrix_plans=True,
                   split_counters="persistent")
    contracts = [item for row in donor._selection_contracts["pops2"] for item in row["items"]
                 if item["name"] == "result"]
    before, = contracts
    after, = [p for p in staged.manifest["phases"] if p["kind"] == "gemm"]
    for key in ("engine", "body", "matrix_shape", "matrix_plan", "ksplit", "arithmetic",
                "transposed_product", "requires_full_grid", "shared_bytes"):
        # The serialized common phase omits ``body`` because elementwise
        # phases use that key for CUDA text. Its physical plan retains the
        # selected engine body; the donor's private evidence keeps both.
        if key == "body":
            assert after["matrix_plan"][key] == before["body"][key]
        # The donor's private evidence embeds ksplit in the physical plan.
        elif key == "ksplit":
            assert after[key] == before["body"]["matrix_plan"][key]
        else:
            assert after[key] == before["body"][key], key
    assert staged._selection_contracts["pops2_layouts"]["result"] == before["layout"]
    assert after["selection_grid"] == SELECTION_GRID
    assert after["preserved_store_plan"] is True
    assert staged.manifest["region_matrix_policy"]["outputs"] == ["result"]
    assert staged.manifest["output_placement"]["result"]["mode"] == binding


@pytest.mark.parametrize("shape,body", [((256, 256, 256), "popsb"),
                                        ((512, 256, 1024), "simt"),
                                        ((128, 32, 128), "popsb")])
def test_accumulation_changes_only_selected_final_store_call(shape, body):
    m, n, k = shape
    args = ("a", "b", "result", TensorLayout((m, k), (1, m), "a"),
            TensorLayout((k, n), (n, 1), "b"), (m, n))
    options = dict(ident=lambda name: "v_" + name, counters_at="counter_words", inline=False)
    expected, _ = fp32_selected(*args, grid=SELECTION_GRID, **options)
    selected, _ = fp32_async(*args, ident=options["ident"], counters_at=options["counters_at"],
                             accumulate=True, engine=dict(PIPELINED_MATRIX_ENGINE), preserve_plan=True)
    assert expected["metadata"]["body"] == body
    assert selected["metadata"]["matrix_plan"] == expected["metadata"]["matrix_plan"]
    assert selected["selection_contract"] == expected["selection_contract"]
    assert selected["scratch"] == expected["scratch"]
    assert selected["counter_init"] == expected["counter_init"] is None
    assert selected["helpers"] == expected["helpers"]
    assert selected["code"].replace(",true>(", ">(", 1) == expected["code"]
    assert selected["code"] != expected["code"]


def test_preservation_does_not_replan_through_accumulation_policy():
    m = n = k = 256
    args = ("a", "b", "result", TensorLayout((m, k), (1, m), "a"),
            TensorLayout((k, n), (n, 1), "b"), (m, n))
    options = dict(ident=lambda name: "v_" + name, accumulate=True,
                   engine=dict(PIPELINED_MATRIX_ENGINE))
    old, _ = fp32_async(*args, **options)
    unchanged, _ = fp32_async(*args, preserve_plan=False, **options)
    retained, _ = fp32_async(*args, preserve_plan=True, **options)
    assert old == unchanged
    assert old["metadata"]["ksplit"] == 2
    assert retained["metadata"]["ksplit"] == 4


def test_refused_ordinary_matrix_keeps_the_pinned_fallback():
    # K=8 has no admitted A configuration (even BK=8 needs two stages).
    # Preservation must not silently replan it through B's distinct admission.
    args = ("a", "b", "result", TensorLayout((128, 8), (8, 1), "a"),
            TensorLayout((8, 128), (128, 1), "b"), (128, 128))
    ordinary, reason = fp32_selected(*args, ident=lambda name: "v_" + name, grid=SELECTION_GRID)
    selected, selected_reason = fp32_async(*args, ident=lambda name: "v_" + name, accumulate=True,
                             engine=dict(PIPELINED_MATRIX_ENGINE), preserve_plan=True)
    assert ordinary is None and selected is None
    assert reason
    assert selected_reason == reason


def test_matrix_preservation_requires_a_staged_region():
    with pytest.raises(LoweringError, match="staged region"):
        lower(_graph(256, 256, 256), preserve_matrix_plans=True, policy="reference")
