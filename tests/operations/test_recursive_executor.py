"""CPU compiler contracts for recursive loop execution; run on the lane CPU VM.

These check the executor ABI and arithmetic grouping, not native numerics or
speed. The lane's GPU qualification compares changing runtime counts, trained
states, dropout replay and restart against the donor and independent references.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import json
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler.ir import Operation, Ref, Value
from megakernel_lab.operation_compiler.loop import BoundedLoopIR, OperationIR
from megakernel_lab.operation_compiler.loop_lower import LOOP_OPTIONS, select_loop_policy
from megakernel_lab.operation_compiler.loop_region_executor import lower_region, wrap_region
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from . import test_loop, test_nested_loop


# Saved and replayed nested histories share the executor ABI. The dedicated
# replay suite checks actual unsaved children and bounded scratch independently.
SUPPORTED_STORAGE = ("save_body_tape", "recompute_boundaries")


@lru_cache(maxsize=None)
def _capture(storage_mode="save_body_tape"):
    return test_nested_loop.fixture(storage_mode=storage_mode)[0]


def _lower(captured, recursive=None, **kwargs):
    loop_options = {} if recursive is None else {"recursive_executor": recursive}
    return lower(captured.ir, execution="compose", precision_policy="strict",
                 loop_options=loop_options, **kwargs)


def _walk_regions(manifest):
    for child in manifest.get("regions", {}).values():
        yield child
        yield from _walk_regions(child)


def _assert_strict_fp32_regions(manifest, *, staged):
    leaves = [region for region in _walk_regions(manifest) if not region.get("regions")]
    assert leaves
    for region in leaves:
        # The established emitter and common planner expose different policy
        # prose. Check their machine-readable precision settings and admitted
        # FP32 bodies instead. This does not assert identical reduction trees;
        # independent native numerical and donor comparisons remain required.
        plan = region.get("execution_plan", {})
        arithmetic = plan.get("arithmetic", region.get("performance_schedule", {}).get("precision_policy"))
        assert arithmetic == "strict"
        assert {d["dtype"] for d in region["layouts"].values()} <= {
            "float32", "int64", "int32", "bool", "uint8",
        }
        assert all(phase.get("engine") in {
            "generic_simt_16", "pinned_simt_64", "pinned_simt_64_splitk", "pops2_fp32", "popsb_cpasync",
        } for phase in region["phases"] if phase["kind"] == "gemm")
        if staged:
            assert plan["fast_activation"] is False
            assert region["primitive_policy"]["exact_reductions"] is True


@pytest.mark.parametrize("storage_mode", SUPPORTED_STORAGE)
def test_default_selects_recursive_executor_without_changing_public_abi(storage_mode):
    captured = _capture(storage_mode)
    before_ir = json.dumps(captured.ir.to_dict(), sort_keys=True)
    donor, automatic, explicit = (_lower(captured, choice) for choice in (False, None, True))
    old, new = donor.manifest, automatic.manifest
    assert new["loop_schedule_policy"]["selected"] == "recursive_staged"
    assert new["recursive_executor"] is True
    assert not any("requires ordered recursive" in reason for reason in new["loop_schedule_policy"]["reasons"])
    assert old["loop_schedule_policy"]["selected"] == "ordered"
    assert not old.get("recursive_executor")
    assert automatic.source == explicit.source
    assert automatic.source != donor.source
    for key in ("inputs", "outputs", "structured_outputs", "input_roles", "state_writes",
                "optimizer_state_schema", "runtime_loop"):
        assert new[key] == old[key], key
    _assert_strict_fp32_regions(old, staged=False)
    _assert_strict_fp32_regions(new, staged=True)
    for name in new["inputs"] + new["outputs"]:
        for key in ("shape", "strides", "dtype"):
            assert new["layouts"][name][key] == old["layouts"][name][key], (name, key)
    assert any(region.get("placed_outputs") for region in _walk_regions(new))
    assert any(region.get("stages") for region in _walk_regions(new))
    assert json.dumps(captured.ir.to_dict(), sort_keys=True) == before_ir


@pytest.mark.parametrize("storage_mode", SUPPORTED_STORAGE)
def test_runtime_counts_and_history_level_control_are_preserved(storage_mode):
    captured = _capture(storage_mode)
    program = _lower(captured)
    control = program.manifest["runtime_loop"]
    assert control["max_steps"] == 3 and control["inner"]["max_steps"] == 4
    assert control["count_input"] == captured.ir.inputs[-1]
    assert control["inner"]["storage_mode"] == storage_mode
    # Forward and reverse loops use the clamped device counts, including zero.
    # A static unroll or a compile-time count cannot pass these contracts.
    for fragment in (
        "const int64_t requested_count = *reinterpret_cast<const int64_t*>",
        "requested_count < 0 ? 0 : (requested_count > 3LL ? 3LL : requested_count)",
        "for (int iteration = 0; iteration < count; ++iteration)",
        "for (int reverse = count - 1; reverse >= 0; --reverse)",
        "const int64_t inner_requested = *reinterpret_cast<const int64_t*>",
        "inner_requested < 0 ? 0 : (inner_requested > 4LL ? 4LL : inner_requested)",
        "for (int inner_iteration=0; inner_iteration<inner_count; ++inner_iteration)",
        "for (int inner_reverse=inner_count-1; inner_reverse>=0; --inner_reverse)",
    ):
        assert fragment in program.source, fragment
    restored = BoundedLoopIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    restored.metadata["case"] = "reserved-name-must-not-select-execution"
    assert lower(restored, execution="compose", precision_policy="strict").source == program.source


def test_retained_diagnostics_and_explicit_ordering_keep_donor_executor():
    captured = _capture()
    for kwargs in ({"retain_all": True}, {"loop_options": {"pipeline": False}},
                   {"policy": "reference"}):
        options = dict(kwargs.pop("loop_options", {}))
        programs = [lower(captured.ir, execution="compose", precision_policy="strict",
                          loop_options={**options, "recursive_executor": choice}, **kwargs)
                    for choice in (False, True)]
        assert programs[0].source == programs[1].source
        assert all(p.manifest["loop_schedule_policy"]["selected"] == "ordered" for p in programs)
        assert not any(p.manifest.get("recursive_executor") for p in programs)


@pytest.mark.parametrize("storage_mode", SUPPORTED_STORAGE)
def test_inner_gradients_finish_locally_before_one_enclosing_addition(storage_mode):
    captured = _capture(storage_mode)
    manifest = _lower(captured).manifest
    reverse_name = "body_vjp_saved" if storage_mode == "save_body_tape" else "body_vjp"
    nested = manifest["regions"][reverse_name]
    contract = nested["recursive_executor"]
    n = len(captured.parameter_names)
    outer_carries = len(captured.carry_names)
    # The nested total stays local. Sending each inner VJP directly into the
    # enclosing gradient would change outer + ((finish + body) + init) into
    # ((outer + finish) + body) + init, which is different FP32 arithmetic.
    assert contract["private_accumulated_outputs"] == list(range(n))
    assert set(nested["placed_outputs"]) == set(range(n, n + outer_carries))
    for i, name in enumerate(nested["outputs"]):
        assert (name in nested["offsets"]) == (i < n)
    assert contract["barrier_context"] == "forwarded from enclosing launch; no reset"
    bindings = contract["region_bindings"]
    suffix = "_saved" if storage_mode == "save_body_tape" else ""
    assert all(mode == "store" for mode in bindings[f"finish_vjp{suffix}"])
    for name in (f"body_vjp{suffix}", f"init_vjp{suffix}"):
        assert bindings[name][:n + outer_carries] == ["accumulate"] * (n + outer_carries)
        assert all(mode == "store" for mode in bindings[name][n + outer_carries:])
    # Wrapper emission proves that these private totals receive one addition
    # after the whole recursive body, rather than losing the outer history.
    from megakernel_lab.operation_compiler.nested_loop_lower import lower_nested
    region = getattr(captured.ir, reverse_name)
    modes = ["accumulate"] * n + ["store"] * outer_carries
    program = lower_nested(region, output_bindings=modes, region_body=True, loop_counter=True)
    wrapper = wrap_region(program, "nested_total", modes, counter=True)
    tail = wrapper[wrapper.index(program.kernel_body) + len(program.kernel_body):]
    assert tail.count("[i] += reinterpret_cast<const float*>") == n
    assert "latch_loop_bar_begin" not in wrapper


def test_explicit_ordered_recompute_preserves_replay_and_memory_contract():
    captured = _capture("recompute_boundaries")
    donor = _lower(captured, False)
    requested = lower(captured.ir, execution="compose", precision_policy="strict",
                      loop_options={"recursive_executor": True, "pipeline": False})
    assert donor.source == requested.source
    assert requested.manifest["loop_schedule_policy"]["selected"] == "ordered"
    assert requested.manifest["runtime_loop"]["storage_mode"] == "recompute_boundaries"
    assert requested.manifest["body_tape_bytes"] == 0
    assert "for (int replay = 0; replay < reverse; ++replay)" in requested.source
    assert "for (int inner_replay=0; inner_replay<inner_reverse; ++inner_replay)" in requested.source


def _option_policy_graph(storage_mode="save_body_tape", *, nested=True):
    leaf = SimpleNamespace(operations=[])
    body = SimpleNamespace(nested_operation=True, regions={"body": leaf}) if nested else leaf
    metadata = {"optimizer": {"kind": "sgd_momentum"}} if storage_mode == "recompute_boundaries" else {}
    return SimpleNamespace(storage_mode=storage_mode, max_steps=3, metadata=metadata,
                           init_saved=leaf, init_vjp_saved=leaf, body_saved=body, body=body,
                           terminal_vjp=leaf, body_vjp_saved=body, body_vjp=body, optimizer=leaf)


@pytest.mark.parametrize("storage_mode,nested,selected", [
    ("save_body_tape", True, "recursive_staged"),
    ("save_body_tape", False, "pipelined"),
    ("recompute_boundaries", True, "recursive_staged"),
    ("recompute_boundaries", False, "recursive_staged"),
])
def test_qualified_defaults_apply_only_to_admitted_recursive_regions(storage_mode, nested, selected):
    chosen, policy = select_loop_policy(_option_policy_graph(storage_mode, nested=nested))
    promoted = selected == "recursive_staged"
    assert policy["selected"] == selected
    assert chosen["recursive_boundary_elision"] is promoted
    expected_regions = ({"recursive_inline_stages": True,
                         "region_expression_inline": True} if promoted else {})
    if storage_mode == "recompute_boundaries":
        expected_regions["recursive_recompute_boundary_elision"] = True
    assert chosen["region_options"] == expected_regions
    assert policy["requested_options"] == {}
    # Effective choices must not mutate global or caller defaults and
    # thereby leak into later diagnostic or ordinary-pipeline compilation.
    assert LOOP_OPTIONS["region_options"] == {}
    assert LOOP_OPTIONS["recursive_boundary_elision"] is False
    for key in ("row_fusion", "warp_softmax", "batched_reduction_loads", "defer_accumulate",
                "epilogue_fusion", "write_through", "persistent_counters"):
        assert chosen[key] == LOOP_OPTIONS[key]


@pytest.mark.parametrize("storage_mode,nested", [
    ("save_body_tape", True), ("recompute_boundaries", True), ("recompute_boundaries", False),
])
@pytest.mark.parametrize("disabled", ["recursive_inline_stages", "region_expression_inline",
                                     "recursive_boundary_elision", "recursive_recompute_boundary_elision", "all"])
def test_recursive_defaults_preserve_false_overrides_and_other_region_policy(storage_mode, nested, disabled):
    requested = {"region_options": {"row_fusion": False, "split_counters": "phase"}}
    for key in ("recursive_inline_stages", "region_expression_inline", "recursive_recompute_boundary_elision"):
        if disabled in (key, "all"):
            requested["region_options"][key] = False
    if disabled in ("recursive_boundary_elision", "all"):
        requested["recursive_boundary_elision"] = False
    before = json.dumps(requested, sort_keys=True)
    chosen, policy = select_loop_policy(_option_policy_graph(storage_mode, nested=nested), options=requested)
    assert policy["selected"] == "recursive_staged"
    default_boundary = True
    assert chosen["recursive_boundary_elision"] is (default_boundary and disabled not in ("recursive_boundary_elision", "all"))
    for key in ("recursive_inline_stages", "region_expression_inline"):
        assert chosen["region_options"][key] is (disabled not in (key, "all"))
    nested_merge = (storage_mode == "recompute_boundaries" and disabled not in ("recursive_recompute_boundary_elision", "all"))
    assert chosen["region_options"].get("recursive_recompute_boundary_elision", False) is nested_merge
    assert chosen["region_options"]["row_fusion"] is False
    assert chosen["region_options"]["split_counters"] == "phase"
    assert json.dumps(requested, sort_keys=True) == before
    assert json.dumps(policy["requested_options"], sort_keys=True) == before
    assert chosen["region_options"] is not requested["region_options"]


@pytest.mark.parametrize("options,retained", [
    ({"recursive_executor": False}, False), ({"pipeline": False}, False),
    ({"region_policy": "reference"}, False), ({}, True),
])
@pytest.mark.parametrize("storage_mode", ["save_body_tape", "recompute_boundaries"])
def test_donor_and_diagnostic_fallback_do_not_inherit_promoted_defaults(options, retained, storage_mode):
    chosen, policy = select_loop_policy(_option_policy_graph(storage_mode), options=options, retain_all=retained)
    assert policy["selected"] == "ordered" and not chosen["recursive_executor"]
    assert chosen["region_options"] == {} and not chosen["recursive_boundary_elision"]


@pytest.mark.parametrize("inline,boundary,expression", list(product((False, True), repeat=3)))
@pytest.mark.parametrize("storage_mode,overrides,retained", [
    ("recompute_boundaries", {"recursive_executor": False}, False),
    ("save_body_tape", {"recursive_executor": False}, False),
    ("save_body_tape", {"pipeline": False}, False),
    ("save_body_tape", {"region_policy": "reference"}, False),
    ("save_body_tape", {}, True),
])
def test_ordered_fallback_filters_executor_controls_without_losing_request(
        inline, boundary, expression, storage_mode, overrides, retained):
    graph = _option_policy_graph(storage_mode)
    requested = {**overrides, "recursive_boundary_elision": boundary, "region_options": {
        "recursive_inline_stages": inline, "region_expression_inline": expression,
        "recursive_recompute_boundary_elision": boundary,
        "matrix_engine": "generic"}}
    before = json.dumps(requested, sort_keys=True)
    baseline, original_policy = select_loop_policy(graph, retain_all=retained, options=overrides)
    chosen, policy = select_loop_policy(graph, retain_all=retained, options=requested)
    assert policy["selected"] == original_policy["selected"] == "ordered"
    assert policy["reasons"] == original_policy["reasons"]
    assert chosen["recursive_executor"] == baseline["recursive_executor"] is False
    assert chosen["recursive_boundary_elision"] is False
    assert chosen["region_options"] == {"matrix_engine": "generic"}
    assert json.dumps(policy["requested_options"], sort_keys=True) == before
    assert json.dumps(requested, sort_keys=True) == before


@pytest.mark.parametrize("inline,expression", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("nested", [False, True])
def test_staged_selection_preserves_applicable_region_controls(inline, expression, nested):
    requested = {"recursive_boundary_elision": True, "inline_stages": False, "region_options": {
        "recursive_inline_stages": inline, "region_expression_inline": expression,
        "recursive_recompute_boundary_elision": True,
        "matrix_engine": "generic"}}
    chosen, policy = select_loop_policy(_option_policy_graph(nested=nested), options=requested)
    assert policy["selected"] == ("recursive_staged" if nested else "pipelined")
    assert chosen["recursive_executor"] is nested
    assert chosen["recursive_boundary_elision"] is nested
    assert chosen["inline_stages"] is False
    assert chosen["region_options"]["region_expression_inline"] is expression
    assert chosen["region_options"]["matrix_engine"] == "generic"
    assert ("recursive_inline_stages" in chosen["region_options"]) is nested
    assert ("recursive_recompute_boundary_elision" in chosen["region_options"]) is nested
    if nested:
        assert chosen["region_options"]["recursive_inline_stages"] is inline
        assert chosen["region_options"]["recursive_recompute_boundary_elision"] is True
    assert policy["requested_options"] == requested


@pytest.mark.parametrize("name", ["recursive_inline_stages", "region_expression_inline",
                                 "recursive_recompute_boundary_elision"])
@pytest.mark.parametrize("invalid", [None, "true", 0, 1])
def test_executor_region_controls_require_boolean_before_fallback(name, invalid):
    with pytest.raises(LoweringError, match=f"{name} must be boolean"):
        select_loop_policy(_option_policy_graph("recompute_boundaries"),
                           options={"region_options": {name: invalid}})


@pytest.mark.parametrize("storage_mode", ["save_body_tape", "recompute_boundaries"])
def test_combined_recursive_controls_lower_unchanged_when_ordered(storage_mode):
    captured = _capture(storage_mode)
    # Explicit donor ordering in both storage modes uses the established
    # emitter. Compiler-only controls must neither fail admission nor alter it.
    baseline_options = {"recursive_executor": False}
    requested_options = {**baseline_options, "recursive_boundary_elision": True,
                         "region_options": {"recursive_inline_stages": True,
                                            "region_expression_inline": True,
                                            "recursive_recompute_boundary_elision": True}}
    donor = lower(captured.ir, execution="compose", precision_policy="strict",
                  loop_options=baseline_options)
    requested = lower(captured.ir, execution="compose", precision_policy="strict",
                      loop_options=requested_options)
    assert requested.source == donor.source
    assert requested.manifest["loop_schedule_policy"]["selected"] == "ordered"
    recorded = requested.manifest["loop_schedule_policy"]["requested_options"]
    assert recorded["recursive_boundary_elision"] is True
    for name, value in requested_options["region_options"].items():
        assert recorded["region_options"][name] is value
    assert requested.manifest["loop_options"]["region_options"] == donor.manifest["loop_options"]["region_options"]
    assert requested.manifest["loop_options"]["recursive_boundary_elision"] is False


def test_existing_flat_pipeline_source_is_unchanged():
    captured, _ = test_loop._capture(random=True, storage_mode="save_body_tape", max_steps=4)
    donor, candidate = _lower(captured, False), _lower(captured, True)
    assert donor.source == candidate.source
    assert donor.manifest["header_hashes"] == candidate.manifest["header_hashes"]
    assert donor.manifest["source_hash"] == candidate.manifest["source_hash"]
    assert candidate.manifest["loop_schedule_policy"]["selected"] == "pipelined"
    assert donor.manifest["regions"] == candidate.manifest["regions"]


@pytest.mark.parametrize("invalid", [None, "auto", 0, 1])
def test_recursive_executor_option_requires_boolean(invalid):
    with pytest.raises(LoweringError, match="recursive_executor must be boolean"):
        select_loop_policy(_capture().ir, options={"recursive_executor": invalid})


def test_alias_only_region_keeps_typed_residual_copy_and_completion_barrier():
    # State aliases have no operation stages and cannot be directly placed.
    # They still need a typed caller copy and a completion barrier.
    value = Value("state", (2,), "int64", (1,), role="input")
    graph = OperationIR({"state": value}, [], ("state",), ("state",))
    program = lower_region(graph, ["store"])
    assert program.manifest["stages"] == []
    assert program.manifest["placed_outputs"] == []
    wrapper = wrap_region(program, "alias_state", ["store"], counter=True)
    assert "reinterpret_cast<int64_t*>(out[0])[i] =" in wrapper
    assert "reinterpret_cast<const int64_t*>(args[0])" in wrapper
    assert wrapper.index("reinterpret_cast<int64_t*>(out[0])") < wrapper.index("latch_loop_barrier(loop_counter);")


def test_residual_view_copies_from_direct_output_storage():
    values = {
        "x": Value("x", (2, 3), "float32", (3, 1)),
        "y": Value("y", (2, 3), "float32", (3, 1), producer="y"),
        "transposed": Value("transposed", (3, 2), "float32", (1, 3), alias_of="y", producer="transposed"),
    }
    operations = [
        Operation("y", "aten.mul.Tensor", (Ref("x"), 2), {},
                  ("x",), ("y",), (), ("x",), ("y",)),
        Operation("transposed", "aten.t.default", (Ref("y"),), {},
                  ("y",), ("transposed",), ("y",), ("y",), (), synchronization="view"),
    ]
    graph = OperationIR(values, operations, ("x",), ("y", "transposed"))
    program = lower_region(graph, ["store", "store"])
    assert program.manifest["placed_outputs"] == [0]
    assert "y" not in program.manifest["offsets"]
    assert program.manifest["layouts"]["transposed"]["root"] == "y"
    wrapper = wrap_region(program, "view_result", ["store", "store"], counter=True)
    assert "reinterpret_cast<float*>(out[1])[i] = reinterpret_cast<const float*>(out[0])" in wrapper


@pytest.mark.parametrize("kind", ["scan_forward", "scan_backward", "conditional"])
@pytest.mark.parametrize("location", ["init_saved", "body_saved/body_saved"])
def test_unsupported_leaf_control_selects_ordered_before_lowering(kind, location):
    # Minimal policy-only IR: recurse through the same regions hierarchy as a
    # nested operation without invoking an unsupported stage to discover it.
    leaf = SimpleNamespace(operations=[])
    nested = SimpleNamespace(nested_operation=True, regions={"body_saved": leaf})
    graph = SimpleNamespace(storage_mode="save_body_tape", max_steps=3, metadata={},
                            init_saved=leaf, init_vjp_saved=leaf, body_saved=nested,
                            terminal_vjp=leaf, body_vjp_saved=nested, optimizer=leaf)
    if kind == "conditional":
        unsupported = SimpleNamespace(true_branch=leaf, false_branch=leaf)
        reason = "uniform conditional"
    else:
        from megakernel_lab.operation_compiler.scan import FWD_TARGET, BWD_TARGET
        unsupported = SimpleNamespace(operations=[SimpleNamespace(
            target=FWD_TARGET if kind == "scan_forward" else BWD_TARGET)])
        reason = "selective-scan grid barriers"
    if "/" in location:
        nested.regions = {"body_saved": unsupported}
    else:
        graph.init_saved = unsupported
    options, policy = select_loop_policy(graph)
    assert options["recursive_executor"] is False
    assert policy["selected"] == "ordered"
    assert any(reason in entry for entry in policy["reasons"])
    if "/" in location or kind != "conditional":
        assert any(location in entry for entry in policy["reasons"])


def test_zero_gradient_contribution_preserves_explicit_fp32_addition():
    values = {name: Value(name, (4,), "float32", (1,)) for name in ("x", "zero")}
    op = Operation("zero", "aten.zeros_like.default", (Ref("x"),), {},
                   ("x",), ("zero",), (), ("x",), ("zero",))
    graph = OperationIR(values, [op], ("x",), ("zero",))
    program = lower_region(graph, ["accumulate"])
    placed = program.manifest["output_placement"]["zero"]
    assert placed["mode"] == "accumulate" and not placed["zero_omitted"]
    assert "v_zero[i] + r_zero" in program.kernel_body
