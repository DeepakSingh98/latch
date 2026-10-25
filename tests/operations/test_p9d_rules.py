"""P9-D CPU contracts: panel spill (ops/sampling_panels.py), train-region placement (ops/sampling_lower.py),
in-place index_copy scatter (ops/extra_ops.py + fusion_reference) and loop-component partition.

Run on a Modal CPU worker (torch 2.9.1): python -m pytest -q ops/test_p9d_rules.py
Lowering is inspected only; device behaviour is qualified by the GPU gates (p9_d/POLICY.json).
"""
from __future__ import annotations

from megakernel_lab.operation_compiler.lower import lower
from .test_sampling_panels import _sampled


def _spill_on():
    return {"panels": "on", "panel_options": {"spill": "on", "spill_gap": 1, "spill_ahead": 1}}


def test_spill_off_by_default_when_slices_fit():
    program = lower(_sampled().ir, sampled_options={"panels": "on"})
    engine = program.manifest["panel_engine"]
    assert engine["spill"]["bytes_per_cta"] == 0 and not engine["spill"]["prefetches"]
    # Shared segment functions retain a nullable spill-pointer argument even
    # when this graph needs no spill allocation or address arithmetic.
    assert engine['spill_area_bytes'] == 0
    assert 'spill_cta = spill_all +' not in program.source
    assert 'spill_cta +' not in program.source


def test_spill_forced_records_prefetches_and_area():
    program = lower(_sampled(K=3).ir, sampled_options=_spill_on())
    engine = program.manifest["panel_engine"]
    assert engine["mode"] == "slice" and engine["spill"]["bytes_per_cta"] > 0
    assert engine["spill_area_bytes"] == engine["spill"]["bytes_per_cta"] * engine["engine_ctas"]
    for pf in engine["spill"]["prefetches"]:
        assert 0 <= pf["hop"] < pf["use"] and pf["bytes"] % 16 == 0
    assert "fence.proxy.async.global" in program.source and "spill_cta" in program.source
    assert "__PANEL_SPILL__" not in program.source


def test_train_place_off_reproduces_parent_and_auto_commits_in_region():
    ir = _sampled().ir
    parent = lower(ir, execution="legacy", sampled_options={"train_place": "off"}, loop_partition=False)
    placed = lower(ir, execution="legacy", sampled_options={})
    assert parent.manifest["train_placement"]["skipped_copies"] == []
    p = placed.manifest["train_placement"]
    assert p["region_state_writes"] and p["skipped_copies"]
    # parameters / moments are committed by the region itself (in place or by its tail commit)
    assert set(p["region_state_writes"]) == set(p["in_place"]) | set(p["tail_commits"])
    # the replay buffer index_copy is committed in place as a scatter (no full-buffer copy)
    region = placed.manifest["regions"]["train"]
    assert any(v.startswith("index_copy") for v in p["in_place"].values())
    assert "if (k != j &&" in placed.source
    assert placed.manifest["arena_bytes"] < parent.manifest["arena_bytes"]


def test_loop_partition_option_is_byte_neutral_when_off():
    ir = _sampled().ir
    a = lower(ir, sampled_options={"train_place": "off"}, loop_partition=False)
    b = lower(ir, sampled_options={"train_place": "off"}, loop_partition=False)
    assert a.manifest["source_hash"] == b.manifest["source_hash"]
