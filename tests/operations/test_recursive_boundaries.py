"""Source dependency contracts for recursive loop boundary merges.

These run on the CPU worker; native count/RNG/gradient gates remain required.
"""
from dataclasses import replace
from functools import lru_cache
import re
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler.loop_lower import select_loop_policy
from megakernel_lab.operation_compiler.loop_region_executor import lower_region
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from .test_nested_loop import fixture


SYNC = "latch_loop_barrier(loop_counter);"


@lru_cache(maxsize=None)
def _programs(max_steps=3, storage_mode="save_body_tape"):
    captured, _ = fixture(storage_mode=storage_mode)
    graph = replace(captured.ir, max_steps=max_steps)
    options = {"execution": "compose", "precision_policy": "strict"}
    # Isolate this boundary transformation from the other promoted defaults.
    region_options = {"recursive_inline_stages": False, "region_expression_inline": False,
                      "recursive_recompute_boundary_elision": False}
    before = lower(graph, **options, loop_options={"recursive_boundary_elision": False,
                                                 "region_options": region_options})
    after = lower(graph, **options, loop_options={"recursive_boundary_elision": True,
                                                "region_options": region_options})
    return captured, graph, before, after


def _namespace(source, name):
    start = source.index(f"namespace {name} {{")
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
    return source[start:end]


def _terminal_call(body):
    match = re.search(r"\{\nfloat\* region_args\[\] = \{([^\n]*)\};\n"
                      r"float\* region_outputs\[\] = \{([^\n]*)\};\n"
                      r"loop_terminal_vjp\(region_args, arena, region_outputs, loop_counter\);\n\}", body)
    assert match
    return match


def test_saved_normal_selection_matches_explicit_qualified_options():
    captured, _ = fixture()
    graph = captured.ir
    default = lower(graph, execution="compose", precision_policy="strict")
    explicit = lower(graph, execution="compose", precision_policy="strict", loop_options={
        "recursive_boundary_elision": True,
        "region_options": {"recursive_inline_stages": True, "region_expression_inline": True}})
    assert default.source == explicit.source
    assert default.manifest["loop_options"] == explicit.manifest["loop_options"]
    assert default.manifest["loop_options"]["recursive_boundary_elision"] is True
    for key in ("recursive_inline_stages", "region_expression_inline"):
        assert default.manifest["loop_options"]["region_options"][key] is True


@pytest.mark.parametrize("max_steps", [0, 3])
@pytest.mark.parametrize("storage_mode", ["save_body_tape", "recompute_boundaries"])
def test_only_init_final_publication_and_export_dependency_are_elided(max_steps, storage_mode):
    captured, graph, before, after = _programs(max_steps, storage_mode)
    old, new = before.manifest, after.manifest
    assert new["recursive_executor"] and new["loop_options"]["recursive_boundary_elision"]
    for key in ("inputs", "outputs", "structured_outputs", "state_writes", "layouts", "offsets",
                "arena_words", "runtime_loop", "regions"):
        assert old[key] == new[key], key

    # Every operation region, including terminal gradients and RNG production,
    # is identical. Only init's trailing publication is merged with its caller.
    init_name = "init_saved" if graph.init_saved is not None else "init"
    old_init = _namespace(before.functions, f"loop_{init_name}_impl")
    new_init = _namespace(after.functions, f"loop_{init_name}_impl")
    head, tail = old_init.rsplit(SYNC, 1)
    assert new_init == head + tail
    assert after.functions == before.functions.replace(old_init, new_init, 1)

    old_call, new_call = _terminal_call(before.kernel_body), _terminal_call(after.kernel_body)
    old_args, new_args = old_call.group(1).split(", "), new_call.group(1).split(", ")
    np, nc = len(captured.parameter_names), len(captured.carry_names)
    assert old_args[:np] == new_args[:np] and old_args[np + nc:] == new_args[np + nc:]
    assert old_args[np:np + nc] == [
        f'arena + {old["offsets"][graph.outputs["carry"][name]]}LL' for name in captured.carry_names]
    final_index = ("count" if storage_mode == "save_body_tape"
                   else "count == 0 ? 0 : (1 + ((count-1) % 2))")
    assert all(f"+ ({final_index}) *" in ptr for ptr in new_args[np:np + nc])
    assert old_call.group(2) == new_call.group(2)  # loss, gradients and cotangent destinations

    # Export copies of every carry and state (including RNG) are unchanged;
    # only their following barrier is removed. Everything after terminal is
    # byte-identical, including reverse count control and final mutable commit.
    old_prefix, new_prefix = before.kernel_body[:old_call.start()], after.kernel_body[:new_call.start()]
    assert old_prefix.endswith(SYNC + "\n")
    assert new_prefix == old_prefix[:-len(SYNC + "\n")]
    assert before.kernel_body[old_call.end():] == after.kernel_body[new_call.end():]
    if storage_mode == "recompute_boundaries":
        assert "for (int replay = 0; replay < reverse; ++replay)" in after.kernel_body[new_call.end():]
        assert new["body_tape_bytes"] == 0
    assert after.source.count(SYNC) == before.source.count(SYNC) - 2

    # The caller's init barrier remains unconditional, including count == 0.
    init_end = after.kernel_body.index(f"loop_{init_name}(region_args, arena, region_outputs, loop_counter);")
    first_loop = after.kernel_body.index("for (int iteration = 0; iteration < count; ++iteration)")
    assert SYNC in after.kernel_body[init_end:first_loop]
    assert f"requested_count > {max_steps}LL ? {max_steps}LL" in after.kernel_body


def _policy_graph(storage="save_body_tape"):
    leaf = SimpleNamespace(operations=[])
    nested = SimpleNamespace(nested_operation=True, regions={"body": leaf})
    return SimpleNamespace(storage_mode=storage, max_steps=3, metadata={}, init_saved=leaf,
                           init_vjp_saved=leaf, body_saved=nested, body=nested,
                           terminal_vjp=leaf, body_vjp_saved=nested, body_vjp=nested, optimizer=leaf)


@pytest.mark.parametrize("options,storage,retained", [
    ({"recursive_executor": False}, "save_body_tape", False),
    ({"pipeline": False}, "save_body_tape", False),
    ({"recursive_executor": False}, "recompute_boundaries", False),
    ({}, "save_body_tape", True),
])
def test_boundary_elision_does_not_change_ordered_fallback(options, storage, retained):
    chosen, policy = select_loop_policy(_policy_graph(storage), retain_all=retained,
                                       options={**options, "recursive_boundary_elision": True})
    assert not chosen["recursive_executor"] and not chosen["recursive_boundary_elision"]
    assert policy["selected"] == "ordered"


@pytest.mark.parametrize("value", [1, None, "true"])
def test_boundary_elision_requires_explicit_boolean(value):
    with pytest.raises(LoweringError, match="recursive_boundary_elision must be boolean"):
        select_loop_policy(_policy_graph(), options={"recursive_boundary_elision": value})


RECOMPUTE_MERGE = "recursive_recompute_boundary_elision"


@lru_cache(maxsize=None)
def _alias_capture():
    """Init and finish return input aliases; explicit state exports stay separate."""
    import torch
    from megakernel_lab.operation_compiler import NestedLoopBody, SGDConfig, capture_loop_training

    def initial(p, x, inner_count):
        return {"h": x * p["scale"], "loss": x.new_zeros(())}

    def init(p, outer, x, inner_count):
        return {"h": outer["h"], "loss": outer["loss"]}

    def body(p, outer, inner, state, x, inner_count):
        h = inner["h"] + 0.1 * (inner["h"] * p["scale"] + outer["h"])
        return {"h": h, "loss": inner["loss"] + 0.01 * h.square().mean()}, {
            "visits": state["visits"] + 1}

    def finish(p, outer, inner, x, inner_count):
        return {"h": inner["h"], "loss": inner["loss"]}

    def terminal(p, carry, x, inner_count):
        return carry["h"].square().mean() + carry["loss"]

    nested = NestedLoopBody(init, body, finish, count_input=1, max_steps=3)
    captured = capture_loop_training(
        initial, nested, terminal, {"scale": torch.full((4,), 0.1)}, None,
        (torch.arange(8, dtype=torch.float32).reshape(2, 4) * 0.01, torch.tensor(0)),
        torch.tensor(0), SGDConfig(lr=0.01, momentum=0.8), max_steps=3,
        state={"visits": torch.tensor(0)}, storage_mode="recompute_boundaries", init_storage="save_tape")
    return captured, (initial, nested, terminal)


def _input_alias_region(region):
    # Public capture clones its returned tensors. Exercise the supported
    # no-stage alias-output IR as well: remove only these input clones, with
    # the same ordered shapes/dtypes and an unchanged caller pointer ABI.
    assert len(region.operations) == len(region.outputs)
    aliases = {}
    for op in region.operations:
        assert op.target.removeprefix("torch.ops.") == "aten.clone.default"
        assert len(op.outputs) == 1 and op.args[0].name in region.inputs
        aliases[op.outputs[0]] = op.args[0].name
    outputs = tuple(aliases[name] for name in region.outputs)
    for original, alias in zip(region.outputs, outputs):
        a, b = region.values[original], region.values[alias]
        assert (a.shape, a.dtype, a.strides) == (b.shape, b.dtype, b.strides)
    return replace(region, values={name: region.values[name] for name in region.inputs},
                   operations=[], outputs=outputs)


@lru_cache(maxsize=None)
def _nested_programs(reverse=False, max_steps=3, *, aliases=False, counters=False,
                     storage_mode="recompute_boundaries"):
    if aliases:
        captured, _ = _alias_capture()
    elif counters:
        from .test_recursive_recompute import _pure_capture
        captured = _pure_capture(batch=4096)
    else:
        captured, _ = fixture(storage_mode=storage_mode)
    name = "body_vjp" if reverse else "body"
    if storage_mode == "save_body_tape":
        name += "_saved"
    region = getattr(captured.ir, name)
    region = replace(region, metadata={**region.metadata, "max_steps": max_steps})
    if aliases:
        region = replace(region, regions={**region.regions, **{
            name: _input_alias_region(region.regions[name]) for name in ("init", "finish")}})
    n = len(captured.parameter_names)
    bindings = (["accumulate"] * n + ["store"] * len(captured.carry_names)
                if reverse else ["store"] * len(region.outputs))
    before = lower_region(region, bindings, **{RECOMPUTE_MERGE: False})
    after = lower_region(region, bindings, **{RECOMPUTE_MERGE: True})
    return captured, region, before, after


def _nested_call(body, name):
    match = re.search(r"\{\nfloat\* region_args\[\] = \{([^\n]*)\};\n"
                      r"float\* region_outputs\[\] = \{([^\n]*)\};\n"
                      + rf"nested_{name}\(region_args,arena,region_outputs,loop_counter\);\n\}}", body)
    assert match, name
    return match


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("max_steps", [0, 3])
def test_nested_recompute_merge_only_changes_eligible_wrapper_publications(reverse, max_steps):
    _, _, before, after = _nested_programs(reverse, max_steps)
    old, new = before.manifest, after.manifest
    merged = ["init"] if reverse else ["init", "finish"]
    assert not old["recursive_executor"]["merged_boundary_barriers"]
    assert old["recursive_executor"].get("merged_boundary_children", []) == []
    assert new["recursive_executor"]["merged_boundary_children"] == merged
    assert new["recursive_executor"]["merged_boundary_barriers"] is True
    for key in old:
        if key not in ("source_hash", "recursive_executor"):
            assert old[key] == new[key], key
    assert {k: v for k, v in old["recursive_executor"].items() if k != "merged_boundary_barriers"} == {
        k: v for k, v in new["recursive_executor"].items()
        if k not in ("merged_boundary_barriers", "merged_boundary_children")}
    # This includes count clamps, zero-count execution, RNG/visit exports,
    # reverse traversal, replay loops and all call argument/output tables.
    assert before.kernel_body == after.kernel_body
    assert f"inner_requested > {max_steps}LL ? {max_steps}LL" in after.kernel_body
    expected_functions = before.functions
    for name in old["regions"]:
        original = _namespace(before.functions, f"nested_{name}_impl")
        modified = _namespace(after.functions, f"nested_{name}_impl")
        if name in merged:
            head, tail = original.rsplit(SYNC, 1)
            assert modified == head + tail
            expected_functions = expected_functions.replace(original, modified, 1)
        else:
            assert modified == original, name
    assert after.functions == expected_functions
    assert after.source.count(SYNC) == before.source.count(SYNC) - len(merged)
    if reverse:
        assert "for (int inner_replay=0; inner_replay<inner_reverse; ++inner_replay)" in after.kernel_body


@pytest.mark.parametrize("max_steps", [0, 3])
def test_alias_init_finish_copies_join_disjoint_state_exports_before_publication(max_steps):
    captured, region, before, after = _nested_programs(max_steps=max_steps, aliases=True)
    assert before.kernel_body == after.kernel_body
    n, no, nc, ns = (region.metadata[k] for k in ("n_parameters", "n_outer_carry", "n_inner_carry", "n_state"))
    assert ns == 1 and captured.state_names == ("visits",)
    for name in ("init", "finish"):
        child = after.manifest["regions"][name]
        assert child["stages"] == [] and child["placed_outputs"] == []
        wrapper = _namespace(after.functions, f"nested_{name}_impl")
        assert "reinterpret_cast<const float*>(args[" in wrapper
        assert wrapper.count("reinterpret_cast<float*>(out[") == nc
        assert SYNC not in wrapper  # Caller must publish these residual alias copies.
    init = _nested_call(after.kernel_body, "init")
    first_body = after.kernel_body.index("for (int inner_iteration=")
    initial_exports = after.kernel_body[init.end():first_body]
    assert f"reinterpret_cast<const int64_t*>(args[{n + no}])" in initial_exports
    assert initial_exports.count(SYNC) == 1
    finish = _nested_call(after.kernel_body, "finish")
    assert finish.group(2).split(", ") == [f"out[{i}]" for i in range(no)]
    final_exports = after.kernel_body[finish.end():]
    assert f"reinterpret_cast<int64_t*>(out[{no}])[i] =" in final_exports
    assert "reinterpret_cast<const int64_t*>(reinterpret_cast<float*>" in final_exports
    assert final_exports.count(SYNC) == 1 and final_exports.rstrip().endswith(SYNC)


def test_alias_fixture_preserves_zero_counts_and_full_history_oracle():
    import torch
    from .test_nested_loop import assert_close, eager
    captured, functions = _alias_capture()
    for outer, inner in ((0, 3), (1, 0), (1, 1), (2, 1), (1, 3), (3, 3)):
        args = (*captured.example_args[:-2], torch.tensor(inner), torch.tensor(outer))
        actual = captured.graph_step(*args)
        assert_close(actual, eager(captured, functions, args))
        assert captured.unpack(actual)["state"]["visits"].item() == outer * inner


@pytest.mark.parametrize("reverse", [False, True])
def test_recompute_merge_keeps_counter_initialization_and_early_publication(reverse):
    _, _, before, after = _nested_programs(reverse, counters=True)
    counters = after.manifest["recursive_executor"]["persistent_counters"]
    assert counters and "body" in counters  # A 16384-element reduction makes this non-vacuous.
    assert counters == before.manifest["recursive_executor"]["persistent_counters"]
    assert before.kernel_body == after.kernel_body
    first_init = _nested_call(after.kernel_body, "init")
    initialized = []
    for counter in counters.values():
        zero = f"reinterpret_cast<unsigned*>(arena + {counter['word_offset']}LL)[i]=0u;"
        assert after.kernel_body.count(zero) == 1 and zero not in after.functions
        initialized.append(after.kernel_body.index(zero) + len(zero))
    assert SYNC in after.kernel_body[max(initialized):first_init.start()]
    assert "latch_loop_bar_begin(loop_counter);" not in after.functions + after.kernel_body


@pytest.mark.parametrize("reverse", [False, True])
def test_recompute_boundary_option_does_not_change_saved_nested_execution(reverse):
    _, _, before, after = _nested_programs(reverse, storage_mode="save_body_tape")
    assert before.source == after.source and before.manifest == after.manifest


def test_default_recompute_and_scoped_override_leave_no_inherited_merge():
    _, region, before, after = _nested_programs()
    default = lower_region(region, ["store"] * len(region.outputs))
    assert default.source == before.source and default.manifest == before.manifest
    assert default.source != after.source


def test_combined_recompute_boundary_merges_compose_without_changing_replay():
    _, graph, baseline, outer_only = _programs(storage_mode="recompute_boundaries")
    combined = lower(graph, execution="compose", precision_policy="strict", loop_options={
        "recursive_boundary_elision": True, "region_options": {
            "recursive_inline_stages": False, "region_expression_inline": False,
            RECOMPUTE_MERGE: True}})
    assert combined.kernel_body == outer_only.kernel_body
    assert combined.source.count(SYNC) == baseline.source.count(SYNC) - 5
    for name, merged in (("body", ["init", "finish"]), ("body_vjp", ["init"])):
        previous = baseline.manifest["regions"][name]
        updated = combined.manifest["regions"][name]
        assert updated["recursive_executor"]["merged_boundary_children"] == merged
        assert previous["regions"] == updated["regions"]
    assert combined.manifest["body_tape_bytes"] == baseline.manifest["body_tape_bytes"] == 0


@pytest.mark.parametrize("invalid", [None, "true", 0, 1])
def test_direct_region_recompute_merge_requires_boolean(invalid):
    _, region, _, _ = _nested_programs()
    with pytest.raises(LoweringError, match=f"{RECOMPUTE_MERGE} must be boolean"):
        lower_region(region, ["store"] * len(region.outputs), **{RECOMPUTE_MERGE: invalid})
