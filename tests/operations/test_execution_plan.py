"""Tiny dependency/resource contracts; numerical qualification runs remotely."""
import json

import pytest

from megakernel_lab.operation_compiler.execution_plan import (contracts, execution_scope, current_execution, static_intervals,
                             record_plan, region_stage_effects)
from megakernel_lab.operation_compiler.lower import LoweringError, TensorLayout, _defer_accumulate, _sink_recomputed


def _fixtures():
    layouts = {n: TensorLayout((64,), (1,), n) for n in ("x", "a", "b", "c")}
    layouts["av"] = TensorLayout((32,), (1,), "a", 16)
    phases = [dict(kind="gemm", reads=["x"], writes=["a"]),
              dict(kind="elementwise", reads=["x"], writes=["b"]),
              dict(kind="reduction", reads=["av", "b"], writes=["c"])]
    return phases, layouts


def test_independent_bodies_share_stage_and_alias_reads_wait():
    phases, layouts = _fixtures()
    assert static_intervals(phases, layouts) == [[0, 1], [2]]
    assert contracts(phases, layouts)[2].dependencies == (0, 1)


def test_write_after_read_and_write_are_ordered():
    phases, layouts = _fixtures()
    phases.append(dict(kind="elementwise", reads=["x"], writes=["av"]))
    assert contracts(phases, layouts)[3].dependencies == (0, 2)
    assert static_intervals(phases, layouts) == [[0, 1], [2], [3]]


def test_internal_grid_barrier_is_fence():
    phases, layouts = _fixtures()
    phases[1]["code"] = "grid.sync();"
    assert static_intervals(phases, layouts) == [[0], [1], [2]]


def test_helper_declared_grid_barrier_is_fence():
    phases, layouts = _fixtures()
    phases[1]["internal_grid_barriers"] = 2
    assert static_intervals(phases, layouts) == [[0], [1], [2]]


def test_shared_module_scratch_body_is_fence_without_textual_barrier():
    phases, layouts = _fixtures()
    phases[1]["global_fence"] = True
    phases[1]["global_effects"] = [{"storage": "private_runtime_cuda_module", "access": "read_write"}]
    assert static_intervals(phases, layouts) == [[0], [1], [2]]
    assert not contracts(phases, layouts)[1].internal_grid_barrier
    assert contracts(phases, layouts)[1].global_fence


def test_region_fence_conflicts_across_tracks_only_when_required():
    ar, aw = region_stage_effects({"a"}, {"b"})
    br, bw = region_stage_effects({"c"}, {"d"})
    assert not (aw & (br | bw) or ar & bw)
    fr, fw = region_stage_effects({"e"}, {"f"}, global_fence=True)
    assert ar & fw and fw & br


def test_private_mutation_follows_initialization_and_orders_reader():
    phases, layouts = _fixtures()
    phases[1]["internal_mutations"] = ["av"]
    phases[1]["counter_effects"] = [{"storage": "runtime_counter_arena", "word_offset": 0, "words": 64}]
    assert contracts(phases, layouts)[1].dependencies == (0,)
    assert contracts(phases, layouts)[2].dependencies == (1,)
    assert contracts(phases, layouts)[1].internal_mutations[0].root == "a"
    assert contracts(phases, layouts)[1].counter_effects[0]["words"] == 64


def test_invalid_emitted_schedule_is_rejected():
    phases, layouts = _fixtures()
    with pytest.raises(LoweringError, match="violates dependency") as caught:
        record_plan({"barrier_intervals": [[0, 1, 2]]}, phases=phases, layouts=layouts)
    detail = json.loads(str(caught.value).split(": ", 1)[1])
    assert detail["body"]["index"] == 2 and detail["body"]["stage"] == 0
    assert [body["index"] for body in detail["dependencies"]] == [0, 1]
    with pytest.raises(LoweringError, match="exactly once"):
        record_plan({"barrier_intervals": [[0, 1], [2, 2]]}, phases=phases, layouts=layouts)


@pytest.mark.parametrize("policy", ["earliest", "duration"])
@pytest.mark.parametrize("middle_kind", ["elementwise", "gemm"])
def test_accumulate_deferral_preserves_slack_without_crossing_fence(policy, middle_kind):
    layouts = {n: TensorLayout((64,), (1,), n) for n in ("x", "grad", "a", "b", "c", "d")}
    phases = [dict(kind="gemm", reads=["x"], writes=["grad"], code="/*B_VBASE*/", tiles=1),
              dict(kind="elementwise", reads=["x"], writes=["a"]),
              dict(kind=middle_kind, reads=["a"], writes=["b"]),
              dict(kind="elementwise", reads=["b"], writes=["c"], global_fence=True),
              dict(kind="elementwise", reads=["c"], writes=["d"])]
    with execution_scope("compose"):
        intervals, moved = _defer_accumulate(phases, static_intervals(phases, layouts), layouts,
                                            {"grad"}, policy=policy)
        record_plan({"barrier_intervals": intervals}, phases=phases, layouts=layouts)
    levels = {i: stage for stage, members in enumerate(intervals) for i in members}
    assert levels[0] == (1 if middle_kind == "elementwise" else 0)
    assert bool(moved) == (middle_kind == "elementwise")
    assert levels[0] < levels[3]


def test_accumulate_deferral_respects_later_internal_mutation_of_input():
    layouts = {n: TensorLayout((64,), (1,), n) for n in ("x", "grad", "a", "b")}
    phases = [dict(kind="gemm", reads=["x"], writes=["grad"], code="/*B_VBASE*/", tiles=1),
              dict(kind="elementwise", reads=["x"], writes=["a"]),
              dict(kind="elementwise", reads=["a"], writes=["b"], internal_mutations=["x"])]
    with execution_scope("compose"):
        intervals, moved = _defer_accumulate(phases, static_intervals(phases, layouts), layouts, {"grad"})
        record_plan({"barrier_intervals": intervals}, phases=phases, layouts=layouts)
    assert intervals == [[0, 1], [2]] and not moved


def test_recomputation_sink_stops_before_fence_but_uses_available_slack():
    layouts = {n: TensorLayout((64,), (1,), n) for n in ("x", "r", "a", "b", "c", "d", "out")}
    phases = [dict(kind="elementwise", reads=["x"], writes=["r"]),
              dict(kind="elementwise", reads=["x"], writes=["a"]),
              dict(kind="elementwise", reads=["a"], writes=["b"]),
              dict(kind="elementwise", reads=["b"], writes=["c"], global_fence=True),
              dict(kind="elementwise", reads=["c"], writes=["d"]),
              dict(kind="elementwise", reads=["d", "r"], writes=["out"])]
    with execution_scope("compose"):
        intervals, moved = _sink_recomputed(phases, static_intervals(phases, layouts), layouts, {"r"})
        record_plan({"barrier_intervals": intervals}, phases=phases, layouts=layouts)
    assert moved == [{"phase": 0, "from": 0, "to": 1}]
    assert intervals == [[1], [0, 2], [3], [4], [5]]


def test_scope_restores_after_failure():
    assert current_execution() == "legacy"
    with pytest.raises(RuntimeError):
        with execution_scope("compose"):
            assert current_execution() == "compose"
            raise RuntimeError("test")
    assert current_execution() == "legacy"


def test_resource_contract_rejects_oversized_body():
    phases, layouts = _fixtures()
    phases[0]["shared_bytes"] = 232449
    with pytest.raises(LoweringError, match="resource ABI"):
        contracts(phases, layouts)
