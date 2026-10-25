"""Flat P6D selection preserves requested arithmetic and actual emitted contracts."""
from copy import deepcopy
from functools import lru_cache

import pytest

from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.p6d_retention import verify
from .test_execution_selection import identical_legacy


@lru_cache(maxsize=1)
def training_graph():
    import torch
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.precision import BF16_PROJECTIONS

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Parameter(torch.full((64, 1024), 0.01))
            self.weight = torch.nn.Parameter(torch.full((1024, 1024), 0.001))
            self.bias = torch.nn.Parameter(torch.zeros(1024))
            self.head = torch.nn.Parameter(torch.full((1024, 64), 0.01))

        def forward(self, x):
            h = torch.tanh(x @ self.input)
            for _ in range(3):
                h = torch.tanh(h @ self.weight + self.bias) + 0.2 * h
            return h @ self.head

    return capture_module(Shared(), lambda y, t: (y - t).square().sum(-1).mean(),
        (torch.full((1280, 64), 0.1),), (torch.zeros((1280, 64)),), precision=BF16_PROJECTIONS).ir


@lru_cache(maxsize=2)
def _candidates(policy):
    g = training_graph()
    return (g, lower(g, execution="compose", compose_selection="common", precision_policy=policy),
            lower(g, execution="legacy", precision_policy=policy))


def candidates(policy="practical"):
    return deepcopy(_candidates(policy))


def check(g, common, legacy, policy="practical"):
    return verify(g, common.manifest, legacy.manifest, contracts=legacy._selection_contracts, policy=policy)


def test_cse_reference_normalization_preserves_order_and_aliases():
    from megakernel_lab.operation_compiler.ir import Ref
    from megakernel_lab.operation_compiler.p6d_retention import _canonical
    aliases = {"duplicate": "original"}
    assert _canonical((Ref("duplicate"), {"ref": "other"}), aliases) == [{"ref": "original"}, {"ref": "other"}]
    assert _canonical((Ref("other"), Ref("duplicate")), aliases) != [{"ref": "original"}, {"ref": "other"}]


@pytest.mark.parametrize("policy", ["strict", "practical"])
def test_normal_api_keeps_complete_existing_p6d_training_kernel(policy):
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    g, common, legacy = candidates(policy)
    proof, refusal = check(g, common, legacy, policy)
    assert refusal is None
    assert proof["cse_aliases"] and proof["epilogues"]
    assert any(row["p9_record"] for row in legacy._selection_contracts["p6d"]["reduction_phases"])
    assert any(body["engine"].startswith("popsc_bf16") for body in proof["matrix_bodies"])
    assert any(body["engine"].startswith("p5c") for body in proof["matrix_bodies"])
    actual = lower(g, execution="compose", precision_policy=policy)
    identical_legacy(actual, legacy)
    selected = actual.manifest["execution_selection"]
    assert selected["requested_precision_policy"] == selected["effective_precision_policy"] == policy
    assert selected["retained_p6d_matrices"] == proof
    # The admitted load pipeline remains visible even though the complete,
    # independently proved P6D schedule wins this normal selection.
    assert selected["admitted_capabilities"]["pipelined_rows"] == 1
    assert "retained_bf16" not in selected
    for occupancy in (132, 264, 1024):
        assert select_launch_policy(actual.manifest, occupancy) == select_launch_policy(legacy.manifest, occupancy)
    assert common.manifest["execution_selection"]["selected"] == "common_static"


def test_pipeline_exemption_does_not_bypass_complete_p6d_proof(monkeypatch):
    from megakernel_lab.operation_compiler import p6d_retention
    g, _, _ = candidates()
    calls = []
    def refuse(*args, **kwargs):
        calls.append(True)
        return None, "test: complete P6D proof refused"
    monkeypatch.setattr(p6d_retention, "verify", refuse)
    actual = lower(g, execution="compose", precision_policy="practical")
    selection = actual.manifest["execution_selection"]
    assert calls and selection["p6d_matrix_retention_candidate"]
    assert selection["selected"] == "common_static"
    assert "test: complete P6D proof refused" in selection["reasons"]
    assert "retained_p6d_matrices" not in selection
    assert any(p.get("row_fusion", {}).get("next_row_pipelined")
               for p in actual.manifest["phases"])


@pytest.mark.parametrize("change", ["body", "split", "destination", "view", "alias", "scalar", "missing_scalar", "rounding", "store", "header", "scratch", "resource", "tensor_map", "precision",
    "reduction_missing", "reduction_extent", "reduction_leaf", "reduction_tree", "reduction_source", "reduction_stage", "reduction_setting", "copied_code", "escaped_product", "epilogue_order", "malformed"])
def test_p6d_retention_rejects_changed_emission_contracts(change):
    g, common, legacy = candidates()
    assert check(g, common, legacy)[1] is None
    evidence = legacy._selection_contracts["p6d"]
    product = next(item for item in evidence["products"] if item["spec"].get("engine") == "p5c")
    epi = next(item["epilogue"] for item in evidence["products"] if item.get("epilogue"))
    if change == "body":
        product["member"]["engine"] = "unknown"
    elif change == "split":
        product["spec"]["splitk"] = 2
    elif change == "destination":
        product["spec"]["out"] = "nullptr"
        product["stored_product"] = False
    elif change == "view":
        product["base_spec"]["strides"][0] += 8
    elif change == "alias":
        evidence["cse_report"]["merged_values"][0][1] = g.inputs[0]
    elif change == "scalar":
        evidence["scalars"][0]["expression"] = "0.0f"
    elif change == "missing_scalar":
        evidence["scalars"].pop()
    elif change == "rounding":
        product["member"]["arithmetic"] = "unrounded FP32 accumulator"
    elif change == "store":
        epi["store_names"] = []
        epi["store_code"] = ""
    elif change == "header":
        legacy.manifest["header_hashes"]["p8ws.cuh"] = "changed"
    elif change == "scratch":
        evidence["allocation"]["p9_offset"] = 0
    elif change == "resource":
        evidence["shared_bytes"] += 256
    elif change == "tensor_map":
        legacy.manifest["tensor_maps"][0]["word_offset"] += 16
    elif change == "precision":
        legacy.manifest["performance_schedule"]["precision_policy"] = "strict"
    elif change == "reduction_missing":
        evidence["reductions"].pop()
    elif change == "reduction_extent":
        evidence["reductions"][0]["extra"]["reduction_elements"] += 1
    elif change == "reduction_leaf":
        evidence["reductions"][0]["storage_reads"][0]["position"] = "0"
    elif change == "reduction_tree":
        evidence["reductions"][0]["node_code_sha256"] = "changed"
    elif change == "reduction_source":
        evidence["calculation_sources"]["p9_rowblock.py"] = "changed"
    elif change == "reduction_stage":
        phase = next(row for row in evidence["reduction_phases"] if row["p9_record"])
        phase["p9_record"]["p9_rowblock"]["rows"] += 1
    elif change == "reduction_setting":
        evidence["settings"]["p9_defer_finalize"] = True
    elif change == "copied_code":
        evidence["copied_commits"]["code"].append("v_state[0] = 0.0f;")
    elif change == "escaped_product":
        elided = next(item for item in evidence["products"] if item["spec"]["out"] == "nullptr")
        reader = next(row["name"] for row in evidence["scalars"] if row["name"] not in elided["epilogue"]["members"])
        evidence["loads_of"][reader].append({"root": elided["name"], "layout": evidence["logical_layouts"][elided["name"]],
                                            "shape": evidence["logical_layouts"][elided["name"]]["shape"], "index": "i"})
    elif change == "epilogue_order":
        ordered = next(item["epilogue"] for item in evidence["products"]
                       if item.get("epilogue") and len(item["epilogue"]["members"]) > 1)
        ordered["members"].reverse()
    elif change == "malformed":
        del evidence["logical_layouts"]
    assert check(g, common, legacy)[1]
