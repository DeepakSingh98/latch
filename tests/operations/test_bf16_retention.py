"""Exact captured BF16 schedule retention and its refusal boundaries."""
from copy import deepcopy
from functools import lru_cache
from unittest.mock import patch

import pytest

from megakernel_lab.operation_compiler.bf16_retention import exact_projection_capture, verify_legacy_bf16
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.precision import BF16_PROJECTIONS
from megakernel_lab.operation_compiler.practical_bf16 import describe as execution_precision
from megakernel_lab.operation_compiler import widen_cast
from .test_compose import _value
from .test_execution_selection import identical_legacy


def graph(width=64):
    result = {"inputs": ["x", "w"], "outputs": ["xb", "wb", "add3"],
            "metadata": {"precision_policy": BF16_PROJECTIONS.describe()},
            "values": {name: _value([width, width], "bfloat16" if name in ("xb", "wb", "m") else "float32")
                       for name in ("x", "w", "xb", "wb", "m", "f")},
            "operations": [
                {"name": "xb", "target": "aten._to_copy.default", "args": [{"ref": "x"}], "kwargs": {"dtype": "bfloat16"}},
                {"name": "wb", "target": "aten._to_copy.default", "args": [{"ref": "w"}], "kwargs": {"dtype": "bfloat16"}},
                {"name": "m", "target": "aten.mm.default", "args": [{"ref": "xb"}, {"ref": "wb"}]},
                {"name": "f", "target": "aten._to_copy.default", "args": [{"ref": "m"}], "kwargs": {"dtype": "float32"}}]}
    # Keep cast outputs material and put this FP32 chain above the established
    # epilogue cost gate. The positive fixture exercises plain POPSC retention;
    # absorbed operands/POPS2-B epilogues have a separate refusal test below.
    previous = "f"
    for index in range(4):
        name = f"add{index}"
        result["values"][name] = _value([width, width], "float32")
        result["operations"].append({"name": name, "target": "aten.add.Tensor",
            "args": [{"ref": previous}, {"ref": "x" if index % 2 else "w"}]})
        previous = name
    return result


def candidates():
    g = graph()
    return g, lower(g, execution="compose", compose_selection="common", precision_policy="practical"), lower(g, execution="legacy", precision_policy="strict")


def verify(g, common, legacy):
    return verify_legacy_bf16(g, common.manifest, legacy.manifest, contracts=legacy._selection_contracts)


@pytest.mark.parametrize("width", [64, 96, 512])
@pytest.mark.parametrize("policy", ["strict", "practical"])
def test_normal_api_keeps_exact_bf16_source_and_reports_effective_policy(width, policy):
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    g = graph(width)
    legacy = lower(g, execution="legacy", precision_policy="strict")
    actual = lower(g, execution="compose", precision_policy=policy)
    identical_legacy(actual, legacy, execution_precision=execution_precision() if policy == "practical" else None)
    selection = actual.manifest["execution_selection"]
    assert selection["requested_precision_policy"] == policy
    assert selection["effective_precision_policy"] == "strict"
    assert len(selection["retained_bf16"]["matrix_bodies"]) == 1
    assert len(selection["retained_bf16"]["casts"]) == 3
    assert actual.manifest["performance_schedule"]["precision_policy"] == "strict"
    for occupancy in (132, 264, 1024):
        assert select_launch_policy(actual.manifest, occupancy) == select_launch_policy(legacy.manifest, occupancy)
    common = lower(g, execution="compose", compose_selection="common", precision_policy=policy)
    assert common.manifest["execution_selection"]["selected"] == "common_static"


@pytest.mark.parametrize("change", ["activation", "attention", "compute_dtype", "product_arithmetic", "applied", "unknown", "products"])
def test_unknown_or_fast_capture_arithmetic_is_not_exempted(change):
    g = graph()
    policy = g["metadata"]["precision_policy"]
    policy[change] = ["unrecognized"] if change == "products" else "unrecognized"
    if change == "activation":
        policy[change] = "fast"
    assert not exact_projection_capture(g)
    result = lower(g, execution="compose", precision_policy="practical")
    assert result.manifest["execution_selection"]["selected"] == "common_static"


def test_capture_name_is_not_a_dispatch_key():
    g = graph()
    g["metadata"]["precision_policy"]["name"] = "independent exact cast recipe"
    assert exact_projection_capture(g)
    identical_legacy(lower(g, execution="compose", precision_policy="practical"), lower(g, execution="legacy", precision_policy="strict"),
                     execution_precision=execution_precision())


def test_list_valued_ir_keeps_normal_compiler_support():
    g = graph(96)
    g["values"] = [{"name": name, **value} for name, value in g["values"].items()]
    identical_legacy(lower(g, execution="compose", precision_policy="practical"), lower(g, execution="legacy", precision_policy="strict"),
                     execution_precision=execution_precision())


def test_absorbed_cast_or_legacy_epilogue_stays_common_until_proven():
    g = graph()
    g["operations"] = g["operations"][:4]
    g["values"] = {name: value for name, value in g["values"].items() if not name.startswith("add")}
    g["outputs"] = ["f"]
    result = lower(g, execution="compose", precision_policy="practical")
    assert result.manifest["execution_selection"]["selected"] == "common_static"
    assert "one-rounding arithmetic is unproven" in result.manifest["execution_selection"]["reasons"][0]


def test_nonlinear_common_epilogue_is_outside_this_proof():
    g, common, legacy = candidates()
    phase = next(p for p in common.manifest["phases"] if p.get("matrix_spec"))
    # Model an additional nonlinear member while preserving the declared
    # rounded product. This proof intentionally recognizes widening only.
    g["values"]["nonlinear"] = _value([64, 64], "float32")
    g["operations"].append({"name": "nonlinear", "target": "aten.sigmoid.default", "args": [{"ref": "f"}]})
    phase["bf16_epilogue"] = {"consumers": ["nonlinear"], "product_stored": False}
    common.manifest["group_rules"]["bf16_epilogues"] = [{"product": "m", "consumer_names": ["nonlinear"],
        "product_stored": False, "arithmetic": "unchanged consumer expressions on the stored BF16 product value"}]
    assert "not an exact widening" in verify(g, common, legacy)[1]


@lru_cache(maxsize=1)
def _shared_training_candidates():
    import torch
    from megakernel_lab.operation_compiler.api import capture_module

    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((96, 96), 0.015))

        def forward(self, x):
            for _ in range(3):
                x = torch.tanh(x @ self.weight) + 0.3 * x
            return x

    captured = capture_module(Shared(), lambda y, t: (y - t).square().mean(),
        (torch.full((32, 96), 0.12),), (torch.zeros((32, 96)),), precision=BF16_PROJECTIONS)
    g = captured.ir.to_dict()
    common = lower(g, execution="compose", compose_selection="common", precision_policy="practical")
    legacy = lower(g, execution="legacy", precision_policy="strict")
    return g, common, legacy


def accumulation_candidates():
    return deepcopy(_shared_training_candidates())


def test_projection_retention_preserves_materialized_casts_before_rewrites():
    from megakernel_lab.operation_compiler.lower import _rewrite_compose_casts
    g, common, legacy = accumulation_candidates()
    # Release repair preserves this canonical graph before the optional storage
    # rewrites. Both explicit and named BF16 projection spellings must still be
    # checked by the retained implementation's original arithmetic proof.
    assert _rewrite_compose_casts(g, retain_all=False, policy="auto", target_arch="sm_90") is g
    proof, refusal = verify(g, common, legacy)
    assert proof is not None and refusal is None
    result = lower(g, execution="compose", precision_policy="practical")
    assert result.manifest["execution_selection"]["selected"] == "legacy_static"


def test_round_each_contribution_before_fp32_accumulation():
    g, common, legacy = accumulation_candidates()
    proof, refusal = verify(g, common, legacy)
    assert refusal is None
    assert proof["fp32_additions"]
    assert all(item["arithmetic"] == "(operand0 + (operand1 * 1.0f))" for item in proof["fp32_additions"])
    with patch.object(widen_cast, "UNDER_PRECISION_POLICY", False):
        identical_legacy(lower(g, execution="compose", precision_policy="practical"), legacy,
                         execution_precision=execution_precision())


@pytest.mark.parametrize("change", ["order", "dtype", "expression", "destination", "pre_widen", "alpha"])
def test_fp32_accumulation_proof_rejects_changed_order_or_rounding(change):
    g, common, legacy = accumulation_candidates()
    proof, refusal = verify(g, common, legacy)
    assert refusal is None and proof["fp32_additions"]
    output = proof["fp32_additions"][0]["output"]
    item = next(item for item in legacy._selection_contracts["bf16_adds"] if item["name"] == output)
    op = next(op for op in g["operations"] if op["outputs"][0] == output)
    if change == "order":
        item["inputs"].reverse()
    elif change == "dtype":
        item["input_layouts"][1]["dtype"] = "bfloat16"
    elif change == "expression":
        item["expression"] = "__bfloat162float(__hadd(operand0, operand1))"
    elif change == "destination":
        item["output_layout"]["offset"] += 1
    elif change == "pre_widen":
        cast = next(op for op in g["operations"] if op["outputs"][0] == item["inputs"][1])
        op["args"] = [op["args"][0], cast["args"][0]]
    else:
        op["kwargs"] = {"alpha": 2}
    assert verify(g, common, legacy)[1]


@pytest.mark.parametrize("options", [{"matrix_engine": "auto"}, {"fuse_attention": False}, {"pops2b": False},
                                     {"p8e_fast_tanh": True}, {"exact_reductions": False}])
def test_explicit_controls_never_receive_precision_exemption(options):
    # Unsupported diagnostic options may refuse at admission. Supported ones
    # must stay on common planning, never silently select the strict recipe.
    from megakernel_lab.operation_compiler.lower import LoweringError
    try:
        result = lower(graph(), execution="compose", precision_policy="practical", **options)
    except LoweringError:
        return
    assert result.manifest["execution_selection"]["selected"] == "common_static"


@pytest.mark.parametrize("change", ["body", "arithmetic", "view", "destination", "cast_rounding", "cast_layout", "cast_missing", "header", "duplicate", "common_rounding"])
def test_retention_requires_matrix_and_cast_proofs(change):
    g, common, legacy = candidates()
    assert verify(g, common, legacy)[1] is None
    item = legacy._selection_contracts["bf16"][0]
    cast = legacy._selection_contracts["bf16_casts"][0]
    if change == "body":
        item["member"]["engine"] = "unproven"
    elif change == "arithmetic":
        item["member"]["arithmetic"] = "FP32 output without rounding"
    elif change == "view":
        item["spec"]["strides"] = [32, 2, 64, 1, 0, 0]
    elif change == "destination":
        legacy.manifest["layouts"]["m"]["offset"] += 16
    elif change == "cast_rounding":
        cast["expression"] = "operand"
    elif change == "cast_layout":
        cast["input_layout"]["offset"] += 8
    elif change == "cast_missing":
        legacy._selection_contracts["bf16_casts"].pop()
    elif change == "header":
        legacy.manifest["header_hashes"]["popsc_mm.cuh"] = "changed"
    elif change == "duplicate":
        legacy._selection_contracts["bf16"].append(deepcopy(item))
    else:
        next(p for p in common.manifest["phases"] if p.get("matrix_spec"))["members"][0]["arithmetic"] = "unrounded epilogue"
    assert verify(g, common, legacy)[1]


def test_new_matrix_edge_body_keeps_common():
    result = lower(graph(200), execution="compose", precision_policy="practical")
    assert result.manifest["execution_selection"]["selected"] == "common_static"
    assert result.manifest["execution_selection"]["admitted_capabilities"]["matrix_adapters"]


def test_untuned_training_module_normal_build(tmp_path, monkeypatch):
    import torch
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler import runtime

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((96, 96), 0.02))

        def forward(self, x):
            return torch.tanh(x @ self.weight)

    captured = capture_module(Model(), lambda y, t: (y - t).square().mean(),
        (torch.full((32, 96), 0.1),), (torch.zeros((32, 96)),), precision=BF16_PROJECTIONS)
    programs = []

    def fake_build(program, *, cache_dir, arch):
        programs.append(program)
        library = tmp_path / f"{len(programs)}.so"
        library.write_bytes(b"CPU build forwarding check; not an executable")
        return library

    monkeypatch.setattr(runtime, "build", fake_build)
    captured.build(tmp_path / "auto", cache_dir=tmp_path / "cache", execution="compose", precision_policy="practical")
    expected = lower(captured.ir, execution="legacy", precision_policy="strict")
    expected.write(tmp_path / "expected")
    identical_legacy(programs[0], expected, execution_precision=execution_precision())
    assert captured.ir.metadata["precision_policy"]["activation"] == "exact"
    assert programs[0].manifest["state_writes"]
