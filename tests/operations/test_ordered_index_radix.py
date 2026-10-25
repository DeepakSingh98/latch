"""CPU structural contracts; the paired CUDA component checks execution bits."""
import re

import pytest

from megakernel_lab.operation_compiler import ordered_index_radix
from megakernel_lab.operation_compiler.indexed_reduction import ordered_embedding_backward, policy
from megakernel_lab.operation_compiler.lower import TensorLayout


def emitted(enabled, *, count=4103, rows=257, width=33, contract=False, cache=None):
    layouts = {
        "g": TensorLayout((count, width), (width, 1), "g"),
        "ids": TensorLayout((count,), (2,), "ids"),
        "dw": TensorLayout((rows, width), (width, 1), "dw"),
    }
    arithmetic = None
    if contract:
        from megakernel_lab.operation_compiler.embedding_backward_contract import CONTRACT
        arithmetic = dict(contract=CONTRACT, output="dw", partial_rows=10,
                          indices_count=count, rows=rows, width=width,
                          accumulation="ascending ten-row partials")
    old = ordered_index_radix.ENABLED
    try:
        ordered_index_radix.ENABLED = enabled
        with policy("batched8_v1"):
            return ordered_embedding_backward(
                [{"ref": "g"}, {"ref": "ids"}, rows, 17, False], {}, "dw", (rows, width),
                layouts, {"g": "float32", "ids": "int64", "dw": "float32"}, set(layouts),
                arithmetic=arithmetic, cache=cache)
    finally:
        ordered_index_radix.ENABLED = old


@pytest.mark.parametrize("contract", [False, True])
def test_only_integer_construction_changes(contract):
    old, new = emitted(False, contract=contract), emitted(True, contract=contract)
    assert new["decision"] == "stable_radix_occurrence_list"
    assert len(new["phases"]) < len(old["phases"])
    normalize = lambda code: re.sub(r"dw__ordinal_\w+", "ORDER", code)
    assert normalize(old["phases"][-1]["code"]) == normalize(new["phases"][-1]["code"])
    assert old["accumulation_order"] == new["accumulation_order"]
    assert all(dtype == "int64" for _, _, dtype in new["auxiliary"])
    assert "* 2LL" in new["phases"][0]["code"]  # strided IDs retained
    assert all("float" not in p["code"] for p in new["phases"][:-1])


@pytest.mark.parametrize("count", [4095, 262145])
def test_register_budget_fallback(count):
    result = emitted(True, count=count)
    assert result["decision"] == "stable_merge_occurrence_list"


def test_same_indices_reuse_the_identical_occurrence_order():
    cache = {}
    first = emitted(True, cache=cache)
    second = emitted(True, cache=cache)
    assert not first["reused_occurrences"] and second["reused_occurrences"]
    assert not second["auxiliary"]
    assert len(second["phases"]) == 1
    assert first["phases"][-1]["code"] == second["phases"][-1]["code"]


@pytest.mark.parametrize("contract", [False, True])
def test_linear_bounds_retain_reduction_and_readiness(monkeypatch, contract):
    original = emitted(True, contract=contract)
    monkeypatch.setattr(ordered_index_radix, "LINEAR_BOUNDS", True)
    cache = {}
    linear = emitted(True, contract=contract, cache=cache)
    assert linear["row_bounds"] == "sorted_neighbor_scatter"
    assert linear["phases"][-1]["code"] == original["phases"][-1]["code"]
    assert len(linear["phases"]) == len(original["phases"]) - 1
    assert not any(p["kind"] == "index_row_bounds" for p in linear["phases"])
    bound = next(v for v, shape, dtype in linear["auxiliary"] if v.endswith("__row_bounds"))
    first, scatter, consumer = linear["phases"][0], linear["phases"][-2], linear["phases"][-1]
    assert bound in first["writes"] and f"{bound}[i] = 0" in first["code"]
    assert bound in scatter["reads"] and bound in consumer["reads"]
    assert scatter["writes"][0] in consumer["reads"]
    assert "keys_[i - 1] >> 32" in scatter["code"] and "keys_[i + 1] >> 32" in scatter["code"]
    again = emitted(True, contract=contract, cache=cache)
    assert again["reused_occurrences"] and len(again["phases"]) == 1
    assert again["phases"][-1]["code"] == consumer["code"]


def test_linear_bounds_cannot_change_merge_fallback(monkeypatch):
    original = emitted(True, count=4095)
    monkeypatch.setattr(ordered_index_radix, "LINEAR_BOUNDS", True)
    assert emitted(True, count=4095) == original
