"""Stage-two CPU contracts; activate only after the saved-executor checkpoint.

This file is a scratch draft, intended for research/latch_ops/ops/. Run on the
cloud CPU worker after the matching compiler patch. Native dropout, changing
counts, independent reference and donor gates remain separate requirements.
"""
from copy import deepcopy
from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import CaptureError, NestedLoopBody, SGDConfig, capture_loop_training
from megakernel_lab.operation_compiler.loop_lower import select_loop_policy
from megakernel_lab.operation_compiler.loop_region_executor import lower_region, wrap_region
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.precision import PrecisionPolicy
from . import test_nested_loop


@lru_cache(maxsize=None)
def _pure_capture(nested=True, outer_max=3, inner_max=4, batch=2, precision=None,
                  storage_mode="recompute_boundaries"):
    """Fixed tensor shapes, parameter-dependent init/finish, full SGD history."""
    params = {"shared": torch.full((4,), 0.1)}
    x = torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 4) * 0.001
    inputs = (x, torch.tensor(0))

    def initial(p, x, inner_count):
        return {"h": x * p["shared"], "loss": x.new_zeros(())}

    def inner_init(p, outer, x, inner_count):
        return {"h": outer["h"] + 0.1 * torch.tanh(outer["h"] * p["shared"]),
                "loss": outer["loss"]}

    def inner_body(p, outer, inner, state, x, inner_count):
        h = inner["h"] + 0.05 * torch.tanh(inner["h"] * p["shared"] + 0.02 * outer["h"])
        return {"h": h, "loss": inner["loss"] + 0.03 * h.square().mean()}, state

    def finish(p, outer, inner, x, inner_count):
        h = inner["h"] + 0.01 * torch.tanh(inner["h"] * p["shared"])
        return {"h": h, "loss": inner["loss"] + 0.02 * (h - outer["h"]).square().mean()}

    def single_body(p, carry, state, x, inner_count):
        h = carry["h"] + 0.05 * torch.tanh(carry["h"] * p["shared"])
        return {"h": h, "loss": carry["loss"] + 0.03 * h.square().mean()}, state

    def terminal(p, carry, x, inner_count):
        return (carry["h"] * p["shared"]).square().mean() + carry["loss"]

    body = (NestedLoopBody(inner_init, inner_body, finish, count_input=1, max_steps=inner_max)
            if nested else single_body)
    return capture_loop_training(initial, body, terminal, params, None, inputs, torch.tensor(0),
                                 SGDConfig(lr=0.01, momentum=0.8), max_steps=outer_max,
                                 storage_mode=storage_mode, init_storage="save_tape", precision=precision)


@lru_cache(maxsize=None)
def _program(nested=True, outer_max=3, inner_max=4, batch=2, recursive=True):
    return lower(_pure_capture(nested, outer_max, inner_max, batch).ir,
                 execution="compose", precision_policy="strict",
                 loop_options={"recursive_executor": recursive})


@pytest.mark.parametrize("nested", [False, True])
def test_plain_fp32_sgd_recompute_selects_staged_and_keeps_public_abi(nested):
    graph = _pure_capture(nested).ir
    chosen, policy = select_loop_policy(graph)
    assert chosen["recursive_executor"] and chosen["barrier"] == "counter"
    assert policy["selected"] == "recursive_staged"
    assert not any("ordered recomputation" in reason for reason in policy["reasons"])
    assert chosen["recursive_boundary_elision"] is True
    assert chosen["region_options"]["recursive_inline_stages"] is True
    assert chosen["region_options"]["region_expression_inline"] is True
    assert chosen["region_options"].get("recursive_recompute_boundary_elision", False) is True
    requested, _ = select_loop_policy(graph, options={"recursive_boundary_elision": True})
    assert requested["recursive_executor"] and requested["recursive_boundary_elision"]
    old, new = _program(nested, recursive=False).manifest, _program(nested).manifest
    assert old["loop_schedule_policy"]["selected"] == "ordered"
    assert new["loop_schedule_policy"]["selected"] == "recursive_staged"
    for key in ("inputs", "outputs", "structured_outputs", "input_roles", "state_writes",
                "optimizer_state_schema", "runtime_loop"):
        assert old[key] == new[key], key
    assert new["body_tape_bytes"] == 0
    assert set(new["compiled_regions"]) == {
        "init_saved", "body", "terminal_vjp", "body_vjp", "init_vjp_saved", "optimizer"}
    assert "body_saved" not in new["regions"] and "body_vjp_saved" not in new["regions"]


@pytest.mark.parametrize("nested", [False, True])
def test_normal_recompute_matches_explicit_qualified_region_options(nested):
    graph = _pure_capture(nested).ir
    automatic = _program(nested)
    explicit = lower(graph, execution="compose", precision_policy="strict", loop_options={
        "recursive_boundary_elision": True,
        "region_options": {"recursive_inline_stages": True, "region_expression_inline": True,
                           "recursive_recompute_boundary_elision": True}})
    assert automatic.source == explicit.source
    for key in ("loop_options", "inputs", "outputs", "structured_outputs", "state_writes",
                "runtime_loop", "arena_bytes", "regions"):
        assert automatic.manifest[key] == explicit.manifest[key], key
    assert ("resource_policy" in automatic.manifest) == ("resource_policy" in explicit.manifest)
    assert automatic.manifest.get("resource_policy") == explicit.manifest.get("resource_policy")
    assert automatic.manifest["body_tape_bytes"] == explicit.manifest["body_tape_bytes"] == 0


@pytest.mark.parametrize("options,retained,reason", [
    ({"recursive_executor": False}, False, None),
    ({"pipeline": False}, False, "explicit ordered"),
    ({"region_policy": "reference"}, False, None),
    ({}, True, "diagnostic"),
])
@pytest.mark.parametrize("nested", [False, True])
def test_recompute_diagnostic_controls_keep_explicit_ordered_fallback(options, retained, reason, nested):
    requested = {**options, "recursive_boundary_elision": True, "region_options": {
        "recursive_inline_stages": True, "region_expression_inline": True,
        "recursive_recompute_boundary_elision": True}}
    chosen, policy = select_loop_policy(_pure_capture(nested).ir, retain_all=retained, options=requested)
    assert policy["selected"] == "ordered" and not chosen["recursive_executor"]
    assert not chosen["recursive_boundary_elision"]
    assert chosen["region_options"] == {}
    assert policy["requested_options"] == requested
    if reason is not None:
        assert any(reason in entry for entry in policy["reasons"])


_UNQUALIFIED_FEATURES = ("sequence_specs", "sequence_names", "invariant_casts",
                         "precision_policy", "advanced_optimizer", "missing_optimizer", "bf16_value")


def _declare_unqualified_feature(graph, feature):
    # These are policy-only mutations, never lowered as a synthetic ABI.
    if feature == "advanced_optimizer":
        graph.metadata["optimizer"] = {"kind": "adamw"}
        return "momentum SGD"
    if feature == "missing_optimizer":
        graph.metadata.pop("optimizer", None)
        return "optimizer kind unknown"
    if feature == "bf16_value":
        next(value for value in graph.values.values() if value.dtype == "float32").dtype = "bfloat16"
        return "FP32"
    graph.metadata[feature] = {"name": "mixed_bf16"} if feature == "precision_policy" else ["unqualified"]
    return {"sequence_specs": "sequence", "sequence_names": "sequence",
            "invariant_casts": "cast", "precision_policy": "precision"}[feature]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("feature", _UNQUALIFIED_FEATURES)
def test_unqualified_recompute_semantics_have_an_explicit_ordered_reason(nested, feature):
    graph = deepcopy(_pure_capture(nested).ir)
    reason = _declare_unqualified_feature(graph, feature)
    chosen, policy = select_loop_policy(graph)
    assert policy["selected"] == "ordered" and not chosen["recursive_executor"]
    assert chosen["region_options"] == {} and not chosen["recursive_boundary_elision"]
    assert any(reason.lower() in entry.lower() for entry in policy["reasons"]), policy


@pytest.mark.parametrize("nested", [False, True])
def test_recompute_precision_gate_inspects_leaf_values(nested):
    graph = deepcopy(_pure_capture(nested).ir)
    leaf = graph.body.regions["body"] if nested else graph.body
    next(value for value in leaf.values.values() if value.dtype == "float32").dtype = "bfloat16"
    chosen, policy = select_loop_policy(graph)
    assert not chosen["recursive_executor"] and policy["selected"] == "ordered"
    path = "body/body" if nested else "body"
    assert any(f"region {path} has ['bfloat16']" in reason for reason in policy["reasons"])


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("feature", _UNQUALIFIED_FEATURES)
def test_recompute_admission_gates_do_not_change_saved_selection(nested, feature):
    # Isolate the selector contract; this does not assert that a fabricated
    # saved ABI can be compiled or extend saved-history numerical coverage.
    graph = deepcopy(_pure_capture(nested, storage_mode="save_body_tape").ir)
    before, before_policy = select_loop_policy(graph)
    _declare_unqualified_feature(graph, feature)
    after, after_policy = select_loop_policy(graph)
    assert after == before
    assert after_policy["selected"] == before_policy["selected"]
    assert after_policy["reasons"] == before_policy["reasons"]
    assert after["recursive_executor"] is nested


_EXACT_PRECISIONS = ("float32", PrecisionPolicy(
    "exact_fp32_noop", products=(), attention="float32", activation="exact"))


@pytest.mark.parametrize("precision", _EXACT_PRECISIONS)
def test_single_explicit_fp32_and_truthy_noop_capture_precision_admit_recompute(precision):
    graph = _pure_capture(False, precision=precision).ir
    declared = graph.metadata.get("precision_policy")
    if precision == "float32":
        assert not declared
    else:
        assert declared and declared["products"] == []
        assert declared["attention"] == "float32" and declared["activation"] == "exact"
    chosen, policy = select_loop_policy(graph)
    assert chosen["recursive_executor"] and policy["selected"] == "recursive_staged"
    assert chosen == select_loop_policy(_pure_capture(False).ir)[0]


@pytest.mark.parametrize("declared", [None, _EXACT_PRECISIONS[1].describe()])
def test_nested_selector_admits_exact_noop_precision_metadata(declared):
    # The nested capture API intentionally rejects explicit precision options.
    # Test only the compiler metadata gate on a copied, valid FP32 capture.
    graph = deepcopy(_pure_capture().ir)
    graph.metadata["precision_policy"] = declared
    chosen, policy = select_loop_policy(graph)
    assert chosen["recursive_executor"] and policy["selected"] == "recursive_staged"
    assert chosen == select_loop_policy(_pure_capture().ir)[0]


@pytest.mark.parametrize("precision", _EXACT_PRECISIONS)
def test_nested_capture_still_rejects_explicit_precision_options(precision):
    with pytest.raises(CaptureError, match="NESTED_SCOPE"):
        _pure_capture(precision=precision)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("declared", [
    {"name": "mixed_bf16"},
    {"attention": "float32", "activation": "exact"},
    {"products": [], "activation": "exact"},
    {"products": [], "attention": "float32"},
    {"products": [], "attention": "float32", "activation": "fast"},
    {"products": ["linear"], "attention": "float32", "activation": "exact"},
    {"products": [], "attention": "bfloat16", "activation": "exact"},
])
def test_incomplete_or_arithmetic_changing_precision_metadata_refuses_recompute(nested, declared):
    graph = deepcopy(_pure_capture(nested).ir)
    graph.metadata["precision_policy"] = declared
    chosen, policy = select_loop_policy(graph)
    assert not chosen["recursive_executor"] and policy["selected"] == "ordered"
    assert "recursive staging requires an exact FP32 capture precision policy" in policy["reasons"]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["scan", "conditional"])
def test_recompute_control_capabilities_still_refuse_staging(nested, kind):
    graph = deepcopy(_pure_capture(nested).ir)
    leaf = graph.body.regions["body"] if nested else graph.body
    if kind == "scan":
        from megakernel_lab.operation_compiler.scan import FWD_TARGET
        leaf.operations.append(SimpleNamespace(target=FWD_TARGET))
        reason = "selective-scan"
    else:
        conditional = SimpleNamespace(true_branch=leaf, false_branch=leaf, values=leaf.values)
        if nested:
            graph.body.regions["body"] = conditional
        else:
            graph.body = conditional
        reason = "conditional"
    chosen, policy = select_loop_policy(graph)
    assert policy["selected"] == "ordered" and not chosen["recursive_executor"]
    assert any(reason in entry for entry in policy["reasons"]), policy


def test_nested_recompute_uses_unsaved_children_and_local_gradient_totals():
    captured = _pure_capture()
    manifest = _program().manifest
    forward, reverse = (manifest["regions"][name] for name in ("body", "body_vjp"))
    assert set(forward["regions"]) == {"init", "body", "finish"}
    assert set(reverse["regions"]) == {"init", "body", "finish_vjp", "body_vjp", "init_vjp"}
    n, no = len(captured.parameter_names), len(captured.carry_names)
    for child in (forward, reverse):
        contract = child["recursive_executor"]
        assert contract["storage_mode"] == "recompute_boundaries"
        assert contract["barrier_context"] == "forwarded from enclosing launch; no reset"
        assert contract["merged_boundary_barriers"] is True
        assert contract["merged_boundary_children"] == (["init", "finish"] if child is forward else ["init"])
        assert set(contract["staged_children"]) == set(child["regions"])
        assert all("_saved" not in name for name in child["regions"])
    bindings = reverse["recursive_executor"]["region_bindings"]
    assert all(mode == "store" for name in ("init", "body", "finish_vjp") for mode in bindings[name])
    for name in ("body_vjp", "init_vjp"):
        assert bindings[name][:n + no] == ["accumulate"] * (n + no)
        assert all(mode == "store" for mode in bindings[name][n + no:])
    assert reverse["recursive_executor"]["private_accumulated_outputs"] == list(range(n))
    assert reverse["placed_outputs"] == list(range(n, n + no))
    region = lower_region(captured.ir.body_vjp, ["accumulate"] * n + ["store"] * no)
    wrapper = wrap_region(region, "replayed_total", ["accumulate"] * n + ["store"] * no, counter=True)
    tail = wrapper[wrapper.index(region.kernel_body) + len(region.kernel_body):]
    assert tail.count("[i] += reinterpret_cast<const float*>") == n
    assert "latch_loop_bar_begin(loop_counter);" not in wrapper


@pytest.mark.parametrize("nested", [False, True])
def test_recompute_preserves_explicit_replay_loops_and_device_count_control(nested):
    before, after = _program(nested, recursive=False), _program(nested)
    fragments = ["for (int iteration = 0; iteration < count; ++iteration)",
                 "for (int reverse = count - 1; reverse >= 0; --reverse)",
                 "for (int replay = 0; replay < reverse; ++replay)",
                 "requested_count < 0 ? 0 : (requested_count > 3LL ? 3LL : requested_count)"]
    if nested:
        fragments += ["for (int inner_iteration=0; inner_iteration<inner_count; ++inner_iteration)",
                      "for (int inner_reverse=inner_count-1; inner_reverse>=0; --inner_reverse)",
                      "for (int inner_replay=0; inner_replay<inner_reverse; ++inner_replay)",
                      "inner_requested < 0 ? 0 : (inner_requested > 4LL ? 4LL : inner_requested)"]
    for fragment in fragments:
        assert before.source.count(fragment) == after.source.count(fragment) > 0, fragment
    assert after.source.count("latch_loop_bar_begin(loop_counter);") == 1
    assert after.manifest["runtime_loop"] == before.manifest["runtime_loop"]


@pytest.mark.parametrize("nested", [False, True])
def test_replay_storage_is_independent_of_outer_and_inner_bounds(nested):
    bounds = ((0, 0), (1, 1), (3, 4), (7, 4), (3, 9)) if nested else ((0, 4), (1, 4), (3, 4), (7, 4))
    manifests = [_program(nested, outer, inner).manifest for outer, inner in bounds]
    for mf, (outer, inner) in zip(manifests, bounds):
        assert mf["loop_schedule_policy"]["selected"] == "recursive_staged"
        assert mf["runtime_loop"]["max_steps"] == outer
        assert mf["body_tape_bytes"] == 0
        if nested:
            assert mf["runtime_loop"]["inner"]["max_steps"] == inner
    for key in ("arena_bytes", "region_scratch_bytes", "boundary_tape_bytes", "init_tape_bytes"):
        assert len({mf[key] for mf in manifests}) == 1, key
    assert manifests[0]["boundary_tape_bytes"] > 0
    if nested:
        for name in ("body", "body_vjp"):
            for key in ("arena_bytes", "region_scratch_bytes"):
                assert len({mf["regions"][name][key] for mf in manifests}) == 1, (name, key)
            assert all(mf["regions"][name]["runtime_loop"]["storage_mode"] == "recompute_boundaries"
                       for mf in manifests)


def test_replayed_children_have_dedicated_counters_and_one_enclosing_context():
    # Full scalar reductions over 16384 elements make this counter check
    # non-vacuous; it is compile/source-only work on the cloud CPU worker.
    captured = _pure_capture(batch=4096)
    n, no = len(captured.parameter_names), len(captured.carry_names)
    for name, bindings in (("body", ["store"] * len(captured.ir.body.outputs)),
                           ("body_vjp", ["accumulate"] * n + ["store"] * no)):
        region = lower_region(getattr(captured.ir, name), bindings)
        mf = region.manifest
        counters = mf["recursive_executor"]["persistent_counters"]
        assert counters and "body" in counters
        occupied = set()
        first_init = region.kernel_body.index("nested_init(region_args,arena,region_outputs,loop_counter)")
        for child, desc in counters.items():
            start, words = desc["word_offset"], desc["words"]
            assert start >= mf["region_scratch_bytes"] // 4
            assert start + words <= mf["arena_words"]
            slots = set(range(start, start + words))
            assert not occupied.intersection(slots)
            occupied.update(slots)
            assert words == mf["regions"][child]["persistent_counter_words"]
            zero = f"reinterpret_cast<unsigned*>(arena + {start}LL)[i]=0u;"
            assert region.kernel_body.count(zero) == 1
            assert region.kernel_body.index(zero) < first_init
        assert "latch_loop_bar_begin(loop_counter);" not in region.kernel_body + region.functions
        assert "grid.sync();" not in region.kernel_body
        assert mf["recursive_executor"]["barrier_context"] == "forwarded from enclosing launch; no reset"


def test_dropout_recompute_replays_rng_operations_instead_of_reading_saved_masks():
    captured, _ = test_nested_loop.fixture(storage_mode="recompute_boundaries")
    mf = lower(captured.ir, execution="compose", precision_policy="strict").manifest
    assert mf["loop_schedule_policy"]["selected"] == "recursive_staged" and mf["body_tape_bytes"] == 0
    reverse = mf["regions"]["body_vjp"]
    assert "body_vjp" in reverse["regions"] and "body_vjp_saved" not in reverse["regions"]
    replay_targets = {op.target for op in captured.ir.body_vjp.regions["body_vjp"].operations}
    assert any("philox_mask" in target for target in replay_targets)
    # Returned RNG/visits have the donor's same public types and state commits.
    donor = lower(captured.ir, execution="compose", precision_policy="strict",
                  loop_options={"recursive_executor": False}).manifest
    assert mf["state_writes"] == donor["state_writes"]
    assert mf["runtime_loop"] == donor["runtime_loop"]
