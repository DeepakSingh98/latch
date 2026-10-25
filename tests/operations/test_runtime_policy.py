"""CPU metadata tests for launch selection; no CUDA device or model required."""
from __future__ import annotations

import copy
import json

from megakernel_lab.operation_compiler.runtime import select_launch_policy


def _phase(kind, shape, **metadata):
    return {"max_gemm_tiles": 0, "layouts": {"out": {"shape": list(shape)}},
            "phases": [{"kind": kind, "writes": ["out"], **metadata}],
            "outputs": ["out"], "state_writes": {}}


def test_small_and_empty_graphs_remain_bounded():
    assert select_launch_policy({}, 132)["selected_blocks"] == 1
    for elements, expected in ((1, 1), (256, 1), (257, 2), (1024, 4)):
        policy = select_launch_policy(_phase("elementwise", (elements,)), 132)
        assert policy["selected_blocks"] == expected
    # Sequential phases are not added together, nor are unrelated allocations.
    manifest = _phase("elementwise", (257,))
    manifest["phases"] *= 100
    manifest["layouts"]["unused_tape"] = {"shape": [10**9]}
    assert select_launch_policy(manifest, 132)["selected_blocks"] == 2


def test_gemm_tiles_and_thread_work_use_different_units():
    assert select_launch_policy({"max_gemm_tiles": 7}, 132)["selected_blocks"] == 7
    matrix = _phase("gemm", (64, 64), tiles=1)
    assert select_launch_policy(matrix, 132)["selected_blocks"] == 1
    for kind in ("elementwise", "embedding", "embedding_gradient", "gather", "scatter_add", "random_mask"):
        manifest = _phase(kind, (257,))
        assert select_launch_policy(manifest, 132)["selected_blocks"] == 2, kind
    assert select_launch_policy(_phase("random_state", (4,)), 132)["selected_blocks"] == 1


def test_reductions_and_softmax_count_output_rows():
    # Reduced input length does not create independent CTA jobs.
    scalar = _phase("reduction", (), reduction_elements=65536)
    assert select_launch_policy(scalar, 132)["selected_blocks"] == 1
    short = _phase("reduction", (1, 64, 256), reduction_elements=4)
    policy = select_launch_policy(short, 132)
    assert policy["estimates"]["reduction"] == 16384
    assert policy["selected_blocks"] == 128
    for kind in ("softmax", "log_softmax"):
        policy = select_launch_policy(_phase(kind, (2, 3, 64), reduction_elements=64), 132)
        assert policy["selected_blocks"] == 6
        assert policy["estimates"]["softmax"] == 6


def test_commit_copies_count_each_sequential_tensor():
    manifest = {"layouts": {"p": {"shape": [257]}, "m": {"shape": [257]}},
                "state_writes": {"p": "new_p", "m": "new_m"}}
    policy = select_launch_policy(manifest, 132)
    assert policy["selected_blocks"] == 2
    assert policy["estimates"]["copies"] == 2
    for kind in ("copy", "state_commit"):
        assert select_launch_policy(_phase(kind, (1025,)), 132)["selected_blocks"] == 5


def test_fused_softmax_counts_only_its_rows():
    manifest = _phase("log_softmax", (16, 10), reduction_elements=10)
    manifest["layouts"]["independent"] = {"shape": [1024]}
    manifest["phases"][0].update(writes=["out", "independent"], items=[
        {"kind": "log_softmax", "name": "out"},
        {"kind": "elementwise", "elements": 1024, "names": ["independent"]}])
    policy = select_launch_policy(manifest, 132)
    assert policy["estimates"]["softmax"] == 16
    assert policy["estimates"]["pointwise"] == 4
    assert policy["selected_blocks"] == 16
    manifest["layouts"]["out"]["shape"] = [161]
    try:
        select_launch_policy(manifest, 132)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed softmax row metadata admitted")


def test_recursive_regions_and_boundary_alias_copies():
    copied_alias = {"layouts": {"alias": {"shape": [1025]}}, "outputs": ["alias"]}
    conditional = {"branches": {"true": copied_alias,
                                "false": _phase("elementwise", (513,))}}
    manifest = {"regions": {"body": conditional, "optimizer": {"max_gemm_tiles": 3}},
                "runtime_loop": {"max_steps": 1000}}
    policy = select_launch_policy(manifest, 132)
    assert policy["selected_blocks"] == 5
    assert policy["estimates"]["copies"] == 5
    manifest["regions"]["body"]["branches"]["false"] = _phase("reduction", (128,), reduction_elements=4)
    assert select_launch_policy(manifest, 132)["selected_blocks"] == 128


def test_occupancy_cap_and_explicit_override():
    large = _phase("elementwise", (10**6,))
    assert select_launch_policy(large, 132)["selected_blocks"] == 128
    assert select_launch_policy(large, 48)["selected_blocks"] == 48
    policy = select_launch_policy(large, 132, 32)
    assert policy["selected_blocks"] == policy["explicit_override"] == 32
    # An explicit admitted override can exceed the default's cap/useful work.
    assert select_launch_policy({}, 264, 200)["selected_blocks"] == 200
    for bad in (0, -1, 133, 1.5, "32", True):
        try:
            select_launch_policy(large, 132, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid explicit blocks admitted: {bad!r}")


def test_policy_does_not_mutate_manifest_or_source_identity():
    manifest = _phase("elementwise", (65536,))
    manifest.update(source_hash="preserved-source-hash", header_hashes={"core.cuh": "preserved-header-hash"})
    before = copy.deepcopy(manifest)
    policy = select_launch_policy(manifest, 132)
    assert manifest == before
    assert select_launch_policy(json.loads(json.dumps(manifest, sort_keys=True)), 132) == policy
    assert json.loads(json.dumps(policy)) == policy


def run():
    tests = [test_small_and_empty_graphs_remain_bounded,
             test_gemm_tiles_and_thread_work_use_different_units,
             test_reductions_and_softmax_count_output_rows,
             test_commit_copies_count_each_sequential_tensor,
             test_fused_softmax_counts_only_its_rows,
             test_recursive_regions_and_boundary_alias_copies,
             test_occupancy_cap_and_explicit_override,
             test_policy_does_not_mutate_manifest_or_source_identity]
    for test in tests:
        test()
    return {"passed": True, "tests": len(tests), "names": [test.__name__ for test in tests],
            "scope": "CPU structural launch policy; no model arithmetic or CUDA execution"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
