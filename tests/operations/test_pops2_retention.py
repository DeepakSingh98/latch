"""Retain compatible emitted POPS2 schedules through the normal compiler API."""
from copy import deepcopy

import pytest

from megakernel_lab.operation_compiler.compose import matrix_retention_candidates, verify_legacy_matrices
from megakernel_lab.operation_compiler.lower import lower
from .test_compose import _value
from .test_execution_selection import identical_legacy


def graph(width=64):
    return {"inputs": ["x", "w", "v"], "outputs": ["a", "b", "total"],
            "values": {name: _value([width, width], "float32") for name in ("x", "w", "v", "a", "b", "total")},
            "operations": [
                {"name": "a", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]},
                {"name": "b", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "v"}]},
                {"name": "total", "target": "aten.add.Tensor", "args": [{"ref": "a"}, {"ref": "b"}]}]}


def candidates():
    common = lower(graph(), execution="compose", compose_selection="common")
    legacy = lower(graph(), execution="legacy")
    found = matrix_retention_candidates(common.manifest)
    assert len(found) == 2
    return common, legacy, found


def verify(common, legacy, found):
    return verify_legacy_matrices(common.manifest, legacy.manifest, found,
                                  contracts=legacy._selection_contracts,
                                  common_contracts=common._selection_contracts)


@pytest.mark.parametrize("width", [64, 128, 512])
def test_normal_api_retains_identical_established_pops2_source(width):
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    selected, legacy = lower(graph(width), execution="compose"), lower(graph(width), execution="legacy")
    identical_legacy(selected, legacy)
    assert selected.manifest["execution_selection"]["retained_matrices"]
    for occupancy in (132, 264, 1024):
        assert select_launch_policy(selected.manifest, occupancy) == select_launch_policy(legacy.manifest, occupancy)
    forced = lower(graph(width), execution="compose", compose_selection="common")
    assert forced.manifest["execution_selection"]["selected"] == "common_static"
    assert forced.source != legacy.source
    # Private compiler evidence is absent from every serialized manifest.
    assert not any("pops2_body" in item for phase in legacy.manifest["phases"] for item in phase.get("items", []))


@pytest.mark.parametrize("change", ["body", "arithmetic", "plan", "view", "operand", "output", "header", "evidence"])
def test_retention_requires_each_emitted_calculation_and_layout(change):
    common, legacy, found = candidates()
    assert verify(common, legacy, found)[1] is None
    row = legacy._selection_contracts["pops2"][0]
    item = row["items"][0]
    output = item["name"]
    if change == "body":
        item["body"]["body"] = "unknown"
    elif change == "arithmetic":
        item["body"]["arithmetic"] = "different sum order"
    elif change == "plan":
        item["body"]["matrix_plan"] = {**item["body"]["matrix_plan"], "ksplit": 8}
    elif change == "view":
        item["layout"] = {**item["layout"], "a_offset": 16}
    elif change == "operand":
        item["operand_roots"] = list(reversed(item["operand_roots"]))
    elif change == "output":
        legacy.manifest["layouts"][output]["root"] = "other"
    elif change == "header":
        legacy.manifest["header_hashes"]["fp32_matrix.cuh"] = "different"
    else:
        row["items"].append(deepcopy(item))
    assert verify(common, legacy, found)[1]


@pytest.mark.parametrize("change", ["capacity", "offset", "overlap", "requirement", "tensor_overlap"])
def test_retention_checks_actual_disjoint_program_scratch(change):
    common, legacy, found = candidates()
    assert verify(common, legacy, found)[1] is None
    row = legacy._selection_contracts["pops2"][0]
    first, second = row["split_bindings"].values()
    if change == "capacity":
        legacy.manifest["matrix_scratch"]["counter_words"] = 0
    elif change == "offset":
        first["partial_word_offset"] = legacy.manifest["arena_words"]
    elif change == "overlap":
        second["counter_word_offset"] = first["counter_word_offset"]
    elif change == "tensor_overlap":
        legacy.manifest["matrix_scratch"]["counter_word_offset"] = 0
    else:
        first["counter_words"] += 1
    assert verify(common, legacy, found)[1]


@pytest.mark.parametrize("change", ["effects", "scratch", "initializer", "other_user", "persistent"])
def test_unknown_effects_never_receive_pops2_retention_exemption(change):
    common, _, found = candidates()
    target = found[0]
    phase = common.manifest["phases"][target["phase"]]
    if change == "effects":
        phase["global_effects"] = ["unproven"]
    elif change == "scratch":
        phase["private_scratch"][0]["partial_words"] += 64
    elif change == "initializer":
        common.manifest["phases"][target["initializer"]]["reads"] = []
    elif change == "other_user":
        common.manifest["phases"].append({"kind": "custom", "reads": [target["scratch"]]})
    else:
        phase["persistent_counter_words"] = 64
    assert target["output"] not in [item["output"] for item in matrix_retention_candidates(common.manifest)]
