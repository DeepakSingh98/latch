"""Normal API execution selection; pinned-Torch CPU checks, no device needed."""
from types import SimpleNamespace
from copy import deepcopy
import re

import pytest

from megakernel_lab.operation_compiler.compose import (flat_selection, homogeneous_long_candidate, reduction_retention_candidates,
                      verify_legacy_long_protocol, verify_legacy_reductions, view_read_hazards)
from megakernel_lab.operation_compiler.execution_plan import current_execution
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from .test_compose import _value


def small_graph():
    return {"inputs": ["x"], "outputs": ["y"],
            "values": {name: _value([64], "float32") for name in ("x", "y")},
            "operations": [{"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, 2]}]}


def identical_legacy(actual, expected, *, execution_precision=None):
    assert actual.source == expected.source
    assert actual.manifest["header_hashes"] == expected.manifest["header_hashes"]
    assert actual.cache_key == expected.cache_key
    # Stronger than a runtime-field whitelist: preserve every previous manifest
    # field, including allocation, launch policy, roots, roles and precision.
    manifest = {k: v for k, v in actual.manifest.items() if k != "execution_selection"}
    if execution_precision is not None:
        # A practical caller's permission is authenticated separately from the
        # stricter retained implementation. All historical fields stay exact.
        assert manifest.pop("execution_precision", None) == execution_precision
    assert manifest == expected.manifest
    assert "execution_plan" not in actual.manifest
    assert actual.manifest["execution_selection"]["selected"] == "legacy_static"


def test_auto_preserves_established_static_source_and_context():
    graph = small_graph()
    expected = lower(graph, execution="legacy")
    for _ in range(3):
        identical_legacy(lower(graph, execution="compose"), expected)
        forced = lower(graph, execution="compose", compose_selection="common")
        assert forced.manifest["execution_selection"]["selected"] == "common_static"
        assert forced.manifest["execution_plan"]["selected"] == "compose"
        assert current_execution() == "legacy"
        identical_legacy(lower(graph), expected)
        assert lower(graph).source == expected.source


def test_normal_module_build_forwards_auto_and_common(tmp_path, monkeypatch):
    import torch
    from megakernel_lab.operation_compiler import runtime, widen_cast
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import _rewrite_compose_casts

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((32, 32), 0.02))

        def forward(self, x):
            h = (x.to(torch.bfloat16) @ self.weight.to(torch.bfloat16)).float()
            return h + 0.25 * torch.nn.functional.silu(h)

    captured = capture_module(Model(), lambda prediction, target: (prediction - target).square().mean(),
                              (torch.full((16, 32), 0.1),), (torch.zeros((16, 32)),))
    original_ir = deepcopy(captured.ir.to_dict())
    # Compose applies the qualified cast rewrite before selecting an emitter.
    # Retaining legacy emission must preserve the transformed graph exactly.
    transformed = _rewrite_compose_casts(captured.ir, retain_all=False,
                                         policy="auto", target_arch="sm_90")
    expected = lower(transformed, execution="legacy")
    identical_legacy(lower(captured.ir, execution="compose"), expected)
    with monkeypatch.context() as patch:
        patch.setattr(widen_cast, "UNARY", False)
        identical_legacy(lower(captured.ir, execution="compose"), lower(captured.ir, execution="legacy"))
    programs = []

    def fake_build(program, *, cache_dir, arch):
        programs.append(program)
        library = tmp_path / (str(len(programs)) + ".so")
        library.write_bytes(b"CPU forwarding contract only; no executable library")
        return library

    monkeypatch.setattr(runtime, "build", fake_build)
    first = captured.build(tmp_path / "auto", cache_dir=tmp_path / "cache", execution="compose")
    second = captured.build(tmp_path / "common", cache_dir=tmp_path / "cache", execution="compose", compose_selection="common")
    assert first["execution_selection"]["selected"] == "legacy_static" and first["execution"] == "legacy"
    assert second["execution_selection"]["selected"] == "common_static" and second["execution"] == "compose"
    assert programs[0].source == expected.source
    assert captured.ir.to_dict() == original_ir


@pytest.mark.parametrize("options", [
    {"staged": True, "region_body": True}, {"region_body": True}, {"output_bindings": [None]},
    {"barrier_schedule": "dependency"}, {"group_rules": "vec4"},
    {"exact_reductions": False}, {"matrix_engine": "auto"}, {"fuse_attention": False},
])
def test_explicit_lowering_controls_retain_common(options):
    manifest = lower(small_graph(), execution="compose", **options).manifest
    assert manifest["execution_selection"]["selected"] == "common_static"
    assert any("explicit lowering controls" in reason for reason in manifest["execution_selection"]["reasons"])
    assert "execution_plan" in manifest


def test_nested_and_unfused_and_practical_keep_common():
    from megakernel_lab.operation_compiler.lower import _LOWERING_DEPTH
    token = _LOWERING_DEPTH.set(1)
    try:
        nested = lower(small_graph(), execution="compose")
    finally:
        _LOWERING_DEPTH.reset(token)
    assert nested.manifest["execution_selection"]["selected"] == "common_static"
    for options in ({"fuse_elementwise": False}, {"precision_policy": "practical"}):
        assert lower(small_graph(), execution="compose", **options).manifest["execution_selection"]["selected"] == "common_static"


@pytest.mark.parametrize("extent,kind", [(1024, "p5c")])
def test_actual_admitted_bodies_keep_common(extent, kind):
    if kind == "p5c":
        graph = {"inputs": ["x", "w"], "outputs": ["y"], "values": {
            "x": _value([extent, 512], "bfloat16"), "w": _value([512, extent], "bfloat16"),
            "y": _value([extent, extent], "bfloat16")}, "operations": [
                {"name": "y", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]}]}
    else:
        graph = {"inputs": ["x"], "outputs": ["y"], "values": {
            "x": _value([extent], "float32"), "y": _value([], "float32")}, "operations": [
                {"name": "y", "target": "aten.mean.default", "args": [{"ref": "x"}]}]}
    from megakernel_lab.operation_compiler import compose as _compose
    # LATCH-FFN-RETAIN: an attention-free explicit-cast program whose P5C bodies the homogeneous protocol proof binds
    # retains the established schedule (EXPLICIT_FLAT_PROTOCOL); with the rule off, the original contract holds.
    with_rule = lower(graph, execution="compose")
    if kind == "p5c":
        assert with_rule.manifest["execution_selection"]["selected"] == "legacy_static"
        assert with_rule.manifest["execution_selection"]["protocol_retention_candidate"] == "p6d_explicit_flat"
    old_rule = _compose.EXPLICIT_FLAT_PROTOCOL
    _compose.EXPLICIT_FLAT_PROTOCOL = False
    try:
        actual = lower(graph, execution="compose")
        forced = lower(graph, execution="compose", compose_selection="common")
    finally:
        _compose.EXPLICIT_FLAT_PROTOCOL = old_rule
    assert actual.manifest["execution_selection"]["selected"] == "common_static"
    assert actual.manifest["execution_selection"]["admitted_capabilities"][kind] == 1
    assert actual.source == forced.source and actual.manifest["header_hashes"] == forced.manifest["header_hashes"]


@pytest.mark.parametrize("phase,body,capability", [
    ({"kind": "attention_long_forward"}, {}, "attention"),
    ({"kind": "attention_mixed_forward", "body_interface": "compose-v1"}, {}, "attention"),
    ({"kind": "custom", "internal_mutations": ["scratch"]}, {}, "opaque_effects"),
    ({"kind": "custom", "private_scratch": [{"name": "scratch"}]}, {}, "opaque_effects"),
    ({"kind": "custom", "counter_effects": [{"storage": "scratch"}]}, {}, "opaque_effects"),
    ({"kind": "custom", "global_effects": ["module_scratch"]}, {}, "opaque_effects"),
    ({"kind": "custom"}, {"global_fence": True}, "global_fences"),
])
def test_opaque_body_contract_guards(phase, body, capability):
    # A body adapter can hide its protocol in a helper; selection must consume
    # its declared contract rather than scanning or recognizing the CUDA text.
    program = SimpleNamespace(manifest={"phases": [phase], "layouts": {}, "state_writes": {},
                                       "execution_plan": {"bodies": [body]}})
    selection = flat_selection({"operations": []}, program, mode="auto", depth=1,
                               explicit_options=set(), fuse_elementwise=True)
    assert selection["selected"] == "common_static" and selection["admitted_capabilities"][capability]


@pytest.mark.parametrize("transitive", [False, True])
def test_state_result_view_consumer_cannot_select_old_emitter(transitive):
    graph = {"inputs": ["p"], "outputs": ["updated", "q"], "state_writes": {"p": "updated"},
             "values": {name: _value([4096], "float32") for name in ("p", "updated", "v", "v2", "q")},
             "operations": [
                 {"name": "updated", "target": "aten.add.Tensor", "args": [{"ref": "p"}, 1]},
                 {"name": "v", "target": "aten.view.default", "args": [{"ref": "updated"}, [4096]]}]}
    if transitive:
        graph["operations"].append({"name": "v2", "target": "aten.alias.default", "args": [{"ref": "v"}]})
    graph["operations"].append({"name": "q", "target": "aten.add.Tensor", "args": [{"ref": "v2" if transitive else "v"}, 2]})
    program = lower(graph, execution="compose")
    assert program.manifest["execution_selection"]["selected"] == "common_static"
    assert program.manifest["execution_selection"]["admitted_capabilities"]["state_result_view_consumers"] == ["q"]
    assert "r_q_0 = (r_updated_0 +" in program.source


@pytest.mark.parametrize("transitive", [False, True])
def test_nonstate_same_phase_view_consumer_retains_common(transitive):
    graph = {"inputs": ["x"], "outputs": ["u", "q"],
             "values": {name: _value([64], "float32") for name in ("x", "u", "v", "v2", "q")},
             "operations": [
                 {"name": "u", "target": "aten.add.Tensor", "args": [{"ref": "x"}, 1]},
                 {"name": "v", "target": "aten.view.default", "args": [{"ref": "u"}, [64]]}]}
    if transitive:
        graph["operations"].append({"name": "v2", "target": "aten.alias.default", "args": [{"ref": "v"}]})
    graph["operations"].append({"name": "q", "target": "aten.add.Tensor", "args": [{"ref": "v2" if transitive else "v"}, 2]})
    result = lower(graph, execution="compose")
    selection = result.manifest["execution_selection"]
    assert selection["selected"] == "common_static"
    assert selection["admitted_capabilities"]["same_phase_view_consumers"] == ["q"]
    assert not selection["admitted_capabilities"]["state_result_view_consumers"]
    # Small bodies use scalar temporaries; vector lanes add a numeric suffix.
    # In either form every new-value read must use its producer register.
    assignments = re.findall(r"\br_q((?:_\d+)?)\s*=\s*([^;]+);", result.kernel_body)
    assert assignments
    for suffix, expression in assignments:
        assert re.match(rf"\(r_u{re.escape(suffix)}\s*\+", expression)
    assert "v_v[" not in result.kernel_body and "v_v2[" not in result.kernel_body
    # Different emitters may place the same nodes together or separately.
    manifest = {"layouts": {}, "phases": [{"kind": "elementwise", "names": ["u"]}, {"kind": "elementwise", "names": ["q"]}]}
    assert not any(view_read_hazards(graph, manifest).values())
    manifest["phases"] = [{"kind": "elementwise", "names": ["u", "q"]}]
    assert view_read_hazards(graph, manifest)["same_phase_view_consumers"] == ["q"]
    manifest["phases"] = [{"names": ["u", "q"], "items": [
        {"kind": "elementwise", "names": ["u"]}, {"kind": "gemm", "name": "q"}]}]
    assert not any(view_read_hazards(graph, manifest).values())


def test_input_transpose_view_does_not_block_established_emitter():
    graph = {"inputs": ["x", "w"], "outputs": ["y"], "values": {
        name: _value([32, 32], "float32") for name in ("x", "w", "wt", "y")},
        "operations": [
            {"name": "wt", "target": "aten.t.default", "args": [{"ref": "w"}]},
            {"name": "y", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "wt"}]}]}
    identical_legacy(lower(graph, execution="compose"), lower(graph, execution="legacy"))


def test_established_candidate_view_grouping_is_checked(monkeypatch):
    from megakernel_lab.operation_compiler import compose
    original = compose.view_read_hazards
    calls = []

    def candidate_hazards(graph, manifest):
        calls.append(manifest)
        return ({"state_result_view_consumers": [], "same_phase_view_consumers": ["y"]}
                if len(calls) == 2 else original(graph, manifest))

    monkeypatch.setattr(compose, "view_read_hazards", candidate_hazards)
    result = lower(small_graph(), execution="compose")
    assert len(calls) == 2
    assert result.manifest["execution_selection"]["selected"] == "common_static"
    assert result.manifest["execution_selection"]["legacy_view_hazards"]["same_phase_view_consumers"] == ["y"]
    assert "execution_plan" in result.manifest and current_execution() == "legacy"


def test_legacy_refusal_preserves_common_and_restores_context(monkeypatch):
    from megakernel_lab.operation_compiler import fusion

    def refuse(*args, **kwargs):
        assert current_execution() == "legacy"
        raise LoweringError("test body refusal")

    monkeypatch.setattr(fusion, "lower_fused", refuse)
    result = lower(small_graph(), execution="compose")
    assert result.manifest["execution_selection"]["selected"] == "common_static"
    assert "test body refusal" in result.manifest["execution_selection"]["reasons"][0]
    assert current_execution() == "legacy"


def test_unknown_selection_and_legacy_override_refused():
    for options in ({"execution": "compose", "compose_selection": "fast"}, {"execution": "legacy", "compose_selection": "common"}):
        with pytest.raises(LoweringError, match="compose_selection"):
            lower(small_graph(), **options)


def reduction_graph():
    return {"inputs": ["x"], "outputs": ["y"], "values": {
        "x": _value([16384], "float32"), "y": _value([], "float32")}, "operations": [
            {"name": "y", "target": "aten.mean.default", "args": [{"ref": "x"}]}]}


def test_existing_full_reduction_protocol_can_be_retained():
    graph = reduction_graph()
    actual, legacy = lower(graph, execution="compose"), lower(graph, execution="legacy")
    identical_legacy(actual, legacy)
    selection = actual.manifest["execution_selection"]
    assert selection["retained_reductions"][0]["output"] == "y"
    assert selection["retained_reductions"][0]["engine"] == "multi_cta_full"
    assert selection["max_grid_blocks"] == 1024
    forced = lower(graph, execution="compose", compose_selection="common")
    assert forced.manifest["execution_selection"]["selected"] == "common_static"
    assert "execution_plan" in forced.manifest


@pytest.mark.parametrize("field,value", [
    ("engine", "single_cta"), ("reduction_elements", 16385), ("arithmetic", "different tree"),
    ("items", [{"kind": "reduction", "name": "y"}, {"kind": "reduction", "name": "other"}]),
    ("targets", ["aten.sum.default"]), ("writes", ["other"]),
])
def test_reduction_retention_requires_actual_matching_body(field, value):
    graph = reduction_graph()
    common = lower(graph, execution="compose", compose_selection="common").manifest
    legacy = deepcopy(lower(graph, execution="legacy").manifest)
    candidates = reduction_retention_candidates(common)
    assert candidates and verify_legacy_reductions(common, legacy, candidates)[1] is None
    phase = next(p for p in legacy["phases"] if "y" in p.get("writes", ()))
    phase[field] = value
    proofs, refusal = verify_legacy_reductions(common, legacy, candidates)
    assert not proofs and refusal


@pytest.mark.parametrize("field,value", [
    ("engine", "single_cta"), ("reduction_elements", 16385), ("arithmetic", "different tree"),
])
def test_reduction_retention_checks_individual_body_contract(field, value):
    graph = reduction_graph()
    common = lower(graph, execution="compose", compose_selection="common").manifest
    legacy = deepcopy(lower(graph, execution="legacy").manifest)
    candidates = reduction_retention_candidates(common)
    phase = next(p for p in legacy["phases"] if "y" in p.get("writes", ()))
    body = next(item for item in phase["items"] if item.get("name") == "y")
    body[field] = value
    proofs, refusal = verify_legacy_reductions(common, legacy, candidates)
    assert not proofs and refusal


def test_unknown_reduction_effects_and_unproven_scratch_keep_common():
    graph = reduction_graph()
    common = lower(graph, execution="compose", compose_selection="common").manifest
    legacy = deepcopy(lower(graph, execution="legacy").manifest)
    candidates = reduction_retention_candidates(common)
    legacy["matrix_scratch"]["partial_words"] = 1023
    assert verify_legacy_reductions(common, legacy, candidates)[1]
    phase = common["phases"][candidates[0]["phase"]]
    phase["global_effects"] = ["unknown scratch"]
    assert not reduction_retention_candidates(common)
    selected = flat_selection(graph, SimpleNamespace(manifest=common), mode="auto", depth=1,
                              explicit_options=set(), fuse_elementwise=True)
    assert selected["selected"] == "common_static"


def test_selected_reduction_grid_bound_is_enforced_by_normal_launch_policy():
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    graph = reduction_graph()
    selected = lower(graph, execution="compose").manifest
    legacy = lower(graph, execution="legacy").manifest
    assert select_launch_policy(selected, 2048)["selected_blocks"] == select_launch_policy(legacy, 2048)["selected_blocks"]
    assert select_launch_policy(selected, 2048, 1024)["selected_blocks"] == 1024
    assert select_launch_policy(legacy, 2048, 1025)["selected_blocks"] == 1025
    assert select_launch_policy(selected, 256, 256)["selected_blocks"] == 256
    for occupancy, override in ((2048, 1025), (256, 257)):
        with pytest.raises(ValueError, match="cooperative grid"):
            select_launch_policy(selected, occupancy, override)
    for invalid in (True, 0, -1, 1.5):
        changed = deepcopy(selected)
        changed["execution_selection"]["max_grid_blocks"] = invalid
        with pytest.raises(ValueError, match="selected body grid bound"):
            select_launch_policy(changed, 2048)


def test_embedded_reduction_grid_bound_is_enforced_without_legacy_selection():
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    manifest = {"max_grid_blocks": 1024, "max_gemm_tiles": 2048}
    assert select_launch_policy(manifest, 2048, 1024)["selected_blocks"] == 1024
    assert select_launch_policy(manifest, 2048)["selected_body_grid_limit"] == 1024
    with pytest.raises(ValueError, match="cooperative grid"):
        select_launch_policy(manifest, 2048, 1025)
    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="program grid bound"):
            select_launch_policy({**manifest, "max_grid_blocks": invalid}, 2048)


def long_protocol_manifests(converted=False):
    group = {"group": "attention", "head_width": 64, "sequence": 512, "source_commit": "P-OPS4-D",
             "version": 7, "recipe": "bf16", "sources": {"q": "q"}, "outputs": {"o": "o"}, "removed": ["bmm"]}
    body = {"engine": "p5c_ws_128x128", "matrix_shape": [1024, 1024, 512], "batch": 1,
            "operand_layout": {"a_k_contiguous": True, "b_k_contiguous": False}, "arithmetic": "BF16 rounded result"}
    matrix = {"kind": "gemm", "names": ["product"], "targets": ["aten.mm.default"], "writes": ["product"],
              "members": [body], "matrix_spec": {"engine": "p5c"}}
    common = {"layouts": {"product": {"shape": [1024, 1024], "dtype": "bfloat16"}},
              "numerical_policy": "same declared math", "outputs": ["converted"], "execution_plan": {"bodies": []},
              "attention_groups": [{**group, "decision": "compose_long_bf16_boundary_attention"}],
              "phases": [matrix] + [{"kind": "attention_long_" + kind, "group": "attention", "body_interface": "compose-v1"}
                                     for kind in ("prepare", "forward", "dq", "kv")]}
    old_output = "converted" if converted else "product"
    legacy = {"layouts": {old_output: {"shape": [1024, 1024], "dtype": "float32" if converted else "bfloat16"}},
              "numerical_policy": common["numerical_policy"],
              "performance_schedule": {"selected": "p6d", "precision_policy": "strict"},
              "attention_groups": [{**group, "decision": "long_bf16_boundary_attention"}],
              "phases": [{"kind": "gemm", "names": [old_output], "items": [{"kind": "gemm", "name": old_output}],
                          "members": [{**body, **({"engine": body["engine"] + "_f32", "matrix_plan": {"out_f32": True}} if converted else {})}]}]
                        + [{"items": [{"kind": "attention_long_" + kind, "group": "attention", "source_commit": "P-OPS4-D"}]}
                           for kind in ("forward", "dq", "kv")]}
    graph = {"operations": [{"name": "converted", "target": "aten._to_copy.default", "args": [{"ref": "product"}]}]}
    return graph, common, legacy


@pytest.mark.parametrize("converted", [False, True])
def test_homogeneous_long_protocol_requires_complete_actual_body_coverage(converted):
    graph, common, legacy = long_protocol_manifests(converted)
    assert homogeneous_long_candidate(common)
    proof, refusal = verify_legacy_long_protocol(graph, common, legacy)
    assert refusal is None and len(proof["matrix_bodies"]) == 1
    if converted:
        assert proof["matrix_bodies"][0]["conversion"] == "float(bf16_rn(acc))"
        graph["operations"].append({"name": "extra", "target": "aten.mul.Tensor", "args": [{"ref": "product"}, 2]})
        assert verify_legacy_long_protocol(graph, common, legacy)[1]
    legacy["phases"][0]["members"][0]["arithmetic"] = "different math"
    assert verify_legacy_long_protocol(graph, common, legacy)[1]


@pytest.mark.parametrize("mutation", ["missing_attention_body", "different_recipe", "different_precision", "missing_matrix"])
def test_homogeneous_protocol_refuses_incomplete_or_changed_implementation(mutation):
    graph, common, legacy = long_protocol_manifests()
    if mutation == "missing_attention_body":
        legacy["phases"].pop()
    elif mutation == "different_recipe":
        legacy["attention_groups"][0]["recipe"] = "exact3"
    elif mutation == "different_precision":
        legacy["performance_schedule"]["precision_policy"] = "practical"
    else:
        legacy["phases"][0]["members"][0]["engine"] = "simt"
    assert verify_legacy_long_protocol(graph, common, legacy)[1]


def long_protocol_with_mask_prepare():
    graph, common, legacy = long_protocol_manifests()
    layout = {"shape": [1, 8, 8], "strides": [64, 8, 1], "dtype": "float32"}
    common["layouts"]["mask_proof"] = dict(layout)
    legacy["layouts"]["mask_proof"] = dict(layout)
    common["phases"][1]["writes"] = ["key_norm", "mask_proof"]
    legacy["phases"].append({"items": [{"kind": "attention_long_prepare", "group": "attention",
        "source_commit": "P-OPS4-D", "phase_writes": ["mask_proof"]}]})
    return graph, common, legacy


def test_homogeneous_protocol_retains_shared_mask_preparation_body():
    graph, common, legacy = long_protocol_with_mask_prepare()
    proof, refusal = verify_legacy_long_protocol(graph, common, legacy)
    assert refusal is None
    assert proof["preparation_bodies"] == [{"group": "attention", "outputs": ["mask_proof"]}]


@pytest.mark.parametrize("mutation", ["duplicate_prepare", "unknown_body", "different_donor", "missing_writes",
                                     "unknown_output", "different_shape", "different_strides", "different_dtype"])
def test_homogeneous_protocol_refuses_unproven_prepare(mutation):
    graph, common, legacy = long_protocol_with_mask_prepare()
    item = legacy["phases"][-1]["items"][0]
    if mutation == "duplicate_prepare":
        legacy["phases"][-1]["items"].append(dict(item))
    elif mutation == "unknown_body":
        item["kind"] = "attention_long_unknown"
    elif mutation == "different_donor":
        item["source_commit"] = "unknown"
    elif mutation == "missing_writes":
        item.pop("phase_writes")
    elif mutation == "unknown_output":
        item["phase_writes"] = ["undeclared"]
    else:
        key = mutation.removeprefix("different_")
        legacy["layouts"]["mask_proof"][key] = {"shape": [1, 8, 9], "strides": [72, 9, 1],
                                                    "dtype": "bfloat16"}[key]
    assert verify_legacy_long_protocol(graph, common, legacy)[1]
