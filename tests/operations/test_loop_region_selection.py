"""LATCH-LOOP-GAP: ordered loop regions use top-level flat emitter selection (run on the lane CPU VM)."""

from __future__ import annotations

import contextlib

from megakernel_lab.operation_compiler.lower import LOOP_REGION_POLICY, lower
from . import test_loop, test_nested_loop


@contextlib.contextmanager
def _policy(value):
    token = LOOP_REGION_POLICY.set({"ordered_region_selection": value})
    try:
        yield
    finally:
        LOOP_REGION_POLICY.reset(token)


def _selections(manifest, where="program"):
    rows = {}
    for name, child in (manifest.get("regions") or {}).items():
        selection = child.get("execution_selection")
        if selection:
            rows[f"{where}/{name}"] = selection
        rows.update(_selections(child, f"{where}/{name}"))
    return rows


def _lowered(captured, value):
    with _policy(value):
        # This donor regression isolates flat-body emitter selection within the
        # ordered executor. Recursive staging has its own compiler contracts.
        return lower(captured.ir, execution="compose", precision_policy="strict",
                     loop_options={"recursive_executor": False})


def _check_ordered(captured):
    old, new = _lowered(captured, False), _lowered(captured, True)
    before, after = _selections(old.manifest), _selections(new.manifest)
    assert before.keys() == after.keys() and before
    changed = 0
    for region, selection in before.items():
        if "nested region retains common planning" not in selection["reasons"]:
            # Composite regions (e.g. the nested operation wrapper) record their own selection; unchanged.
            assert after[region] == selection, region
            continue
        assert selection["selected"] == "common_static"
        reasons = [r for r in selection["reasons"] if r != "nested region retains common planning"]
        if after[region]["selected"] == "legacy_static":
            changed += 1
        else:
            # Any other reason (admitted capability, refusal, hazard) still keeps common planning.
            assert after[region]["selected"] == "common_static" and (reasons or after[region]["reasons"]), region
        assert "nested region retains common planning" not in after[region]["reasons"]
    assert changed, "no ordered region reached the established emitter"
    return changed, len(before)


def test_ordered_single_loop_regions_use_flat_selection():
    captured, _ = test_loop._capture(random=True, storage_mode="recompute_boundaries", max_steps=4)
    _check_ordered(captured)


def test_nested_inner_and_outer_regions_use_flat_selection():
    for mode in ("save_body_tape", "recompute_boundaries"):
        captured, _ = test_nested_loop.fixture(storage_mode=mode)
        _check_ordered(captured)


def test_pipelined_regions_unchanged():
    # Staged pipelined regions keep explicit region options; the rule must not change their source.
    captured, _ = test_loop._capture(random=True, storage_mode="save_body_tape", max_steps=4)
    old, new = _lowered(captured, False), _lowered(captured, True)
    assert old.manifest["loop_schedule_policy"]["selected"] == "pipelined"
    assert old.source == new.source


if __name__ == "__main__":
    test_ordered_single_loop_regions_use_flat_selection()
    test_nested_inner_and_outer_regions_use_flat_selection()
    test_pipelined_regions_unchanged()
    print("ok")
