"""LATCH-TINYMOE-FUSE: ops.norm_fusion structural rules (CPU; lowering only, no nvcc).

A tiny TinyMoE-shaped training program (the production adapter, objective and small-batch update rule with a
small random Mixtral configuration) is captured once and lowered with and without the rules. The tests check
recognition, the selected engines/placements, unchanged lowering when the option is omitted, and fallback for
unrecognized forms. Bitwise equality of executed updates is checked on the GPU worker (lane receipts).
"""
from __future__ import annotations

import functools
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@functools.lru_cache(maxsize=None)
def _ir(layers=1, window=1024):
    from transformers import MixtralConfig, MixtralForCausalLM
    from megakernel_lab.tinymoe.compiled_model import CapturedTinyMoE, objective
    from megakernel_lab.tinymoe.optimizer import initial_model_state
    from megakernel_lab.tinymoe.learning import sb_rule
    from megakernel_lab.operation_compiler import capture_module, bf16_sdpa
    from megakernel_lab.operation_compiler.compose_attention import rewrite_tape
    torch.manual_seed(0)
    cfg = MixtralConfig(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=layers,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_local_experts=4,
                        num_experts_per_tok=2, sliding_window=window, max_position_embeddings=1024, rope_theta=1e6,
                        router_jitter_noise=0.0, attention_dropout=0.0, tie_word_embeddings=False)
    model = CapturedTinyMoE(MixtralForCausalLM(cfg).float())
    tokens = torch.zeros(2, 256, dtype=torch.int64)
    with bf16_sdpa.capture_bf16_sdpa():
        captured = capture_module(model, objective, (tokens,), (torch.zeros_like(tokens), torch.tensor([1.0, 0.0])),
                                  optimizer=sb_rule.make_rule(model), state=initial_model_state(model))
    ir = captured.ir
    if any(op.target == bf16_sdpa.FORWARD for op in ir.operations):
        ir, _ = rewrite_tape(ir, version=7)
    return ir


def _lower(rules, ir=None):
    from megakernel_lab.operation_compiler import update_schedule, norm_fusion
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    with policy("segmented_v1"), update_schedule.scope("streamed"), norm_fusion.scope(rules):
        return lower(ir or _ir(), target_arch="sm_90", execution="compose", precision_policy="practical",
                     bf16_weight_chunk_k=8192, square_sum_association="balanced",
                     compact_codegen="elementwise_full_sum_resource_pure_intervals",
                     static_shared_reserve_bytes=8192)


@functools.lru_cache(maxsize=None)
def _program(rules):
    return _lower(rules)


def _phase_of(program, name):
    for k, phase in enumerate(program.manifest["phases"]):
        if name in phase["names"]:
            return k, phase
    raise KeyError(name)


def test_option_validation():
    from megakernel_lab.operation_compiler import norm_fusion
    assert norm_fusion.rules_of(None) == ()
    assert norm_fusion.rules_of("rows") == ("colgrp", "nremat", "castcse", "resid", "rpack", "rpackb")
    assert norm_fusion.rules_of("colreg+resid") == ("colreg", "resid")
    for bad in ("", "nope", "colreg+nope", 1):
        with pytest.raises(ValueError):
            norm_fusion.validate(bad)


def test_omitted_option_leaves_lowering_unchanged():
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler import update_schedule
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    base = _program(None)
    with policy("segmented_v1"), update_schedule.scope("streamed"):
        plain = lower(_ir(), target_arch="sm_90", execution="compose", precision_policy="practical",
                      bf16_weight_chunk_k=8192, square_sum_association="balanced",
                      compact_codegen="elementwise_full_sum_resource_pure_intervals", static_shared_reserve_bytes=8192)
    assert plain.manifest["source_hash"] == base.manifest["source_hash"]
    assert "norm_fusion" not in base.manifest


def test_recognition_report():
    report = _program("colreg+nremat+gremat+gatedbw+castcse+resid").manifest["norm_fusion"]
    assert _program("rows").manifest["norm_fusion"]["rules"] == ["colgrp", "nremat", "castcse", "resid", "rpack",
                                                               "rpackb"]
    # Three normalization weight-gradient column sums (final, post-attention and input norm).
    assert len(report["column_sums"]) == 3
    # Both normalized rows of the layer and the gated activation are recomputed late.
    sources = report["remat"]["sources"]
    assert any(s.startswith("silu") for s in sources) and sum(s.startswith("mul") for s in sources) == 2
    # q/k/v read one cast of the normalized row; the up/gate products read one cast of its row gather.
    assert len(report["merged_casts"]) == 3
    assert [(r["k"], r["rows"], r["width"]) for r in report["residual_addends"]] == [(2, 512, 128)]


def test_weight_gradient_sums_use_the_coalesced_exact_column_engine():
    base, rows = _program(None), _program("colreg+nremat+gremat+gatedbw+castcse+resid")
    for name in rows.manifest["norm_fusion"]["column_sums"]:
        assert _phase_of(base, name)[1].get("engine") == "exact_cta_tree_per_warp"
        assert _phase_of(rows, name)[1].get("engine") == "exact_col_registers"
    # Other reductions keep their engines.
    other = {p["names"][0]: p.get("engine") for p in base.manifest["phases"] if p["kind"] == "reduction"}
    after = {p["names"][0]: p.get("engine") for p in rows.manifest["phases"] if p["kind"] == "reduction"}
    changed = {n for n in other if n in after and other[n] != after[n]}
    assert changed == set(rows.manifest["norm_fusion"]["column_sums"])


def test_recomputed_values_are_not_stored():
    rows = _program("colreg+nremat+gremat+gatedbw+castcse+resid")
    report = rows.manifest["norm_fusion"]
    elided = set(rows.manifest.get("elided_stores") or [])
    for source, copy in zip(report["remat"]["sources"], report["remat"]["copied"]):
        assert source in elided and copy in elided, (source, copy)
        _, phase = _phase_of(rows, copy)
        assert phase["kind"] in ("row_fused", "elementwise")
        assert len(phase["names"]) > 1          # evaluated inside its reader's loop


def test_gated_backward_chains_share_one_loop():
    rows = _program("colreg+nremat+gremat+gatedbw+castcse+resid")
    copy = next(c for c in rows.manifest["norm_fusion"]["remat"]["copied"] if c.startswith("silu"))
    _, phase = _phase_of(rows, copy)
    assert any(n.startswith("sigmoid") for n in phase["names"])
    base = _program(None)
    _, plain = _phase_of(base, next(n for p in base.manifest["phases"] for n in p["names"] if n.startswith("sigmoid")))
    assert not any(n.startswith("silu") for n in plain["names"])


def test_residual_addend_joins_the_normalization_row_group():
    base, rows = _program(None), _program("colreg+nremat+gremat+gatedbw+castcse+resid")
    record = rows.manifest["norm_fusion"]["residual_addends"][0]
    _, group = _phase_of(rows, record["sum"])
    assert group["kind"] == "row_fused" and any(n.startswith("pow") for n in group["names"])
    assert record["product"] not in {n for p in rows.manifest["phases"] for n in p["writes"]}
    assert len(rows.manifest["barrier_intervals"]) < len(base.manifest["barrier_intervals"])
    s = record["sum"]
    # Leaves are 0.0f + p (as p + (0.0f * 1.0f)) and the k = 2 butterfly adds them in lane order.
    assert re.search(rf"const float r_{s}_l0__rk1 = \(r_\w+_p0__rk1 \+ \(0\.0f \* 1\.0f\)\);", rows.source)
    assert re.search(rf"const float r_{s} = \(r_{s}_l0__rk1 \+ \(r_{s}_l1__rk1 \* 1\.0f\)\);", rows.source)


def test_rule_subsets_apply_independently():
    only = _lower("colreg")
    report = only.manifest["norm_fusion"]
    assert report["rules"] == ["colreg"] and "remat" not in report and "residual_addends" not in report
    assert len(only.manifest["barrier_intervals"]) == len(_program(None).manifest["barrier_intervals"])


def test_unrecognized_forms_fall_back():
    from dataclasses import replace
    from megakernel_lab.operation_compiler import norm_fusion
    ir = _ir()
    # Replace every rsqrt statistic by sqrt: no normalized row is recognized, so no weight-gradient sum,
    # normalized-row remat or cast merge applies (the gated activation and residual addend still do).
    ops = [replace(op, target="aten.sqrt.default") if op.target == "aten.rsqrt.default" else op
           for op in ir.operations]
    with norm_fusion.scope("rows"):
        _, report = norm_fusion.apply_ir(replace(ir, operations=ops))
    assert report["column_sums"] == [] and "merged_casts" not in report
    assert all(s.startswith("silu") for s in report["remat"]["sources"])


def test_rope_chains_are_evaluated_inline_around_attention():
    base, rope = _program(None), _lower("rope")
    report = rope.manifest["norm_fusion"]["rope"]
    assert report["rotary_signatures"] >= 2 and report["values"]
    # The three stored attention operands are computed by one inline phase straight from the projections.
    producers = {n: p for p in rope.manifest["phases"] for n in p["writes"]}
    operands = [r for p in rope.manifest["phases"] if p["kind"].startswith("attention_long_forward")
                for r in p["reads"] if r.startswith("clone")]
    assert operands and len({id(producers[r]) for r in operands}) == 1
    assert producers[operands[0]]["kind"] == "elementwise_inline"
    assert len(rope.manifest["barrier_intervals"]) < len(base.manifest["barrier_intervals"])


def test_rope_requires_a_rotary_signature():
    from dataclasses import replace
    from megakernel_lab.operation_compiler import norm_fusion
    ir = _ir()
    # Without the negation of the rotated half (no rotary embedding) the region is not admitted.
    ops = [replace(op, target="aten.clone.default") if op.target == "aten.neg.default" else op for op in ir.operations]
    names, rotary = norm_fusion.rope_region(replace(ir, operations=ops))
    assert names == [] and rotary == 0


def test_cast_before_gather():
    rows = _lower("dcast+castcse")
    report = rows.manifest["norm_fusion"]
    (record,) = report["cast_before_gather"]
    written = {n for p in rows.manifest["phases"] for n in p["writes"]}
    assert record["gather"] not in written and record["cast"] in written
    gather = next(p for p in rows.manifest["phases"] if record["cast"] in p["writes"])
    assert gather["kind"] == "embedding"
    # The gathered table is the router projection's cast of the same normalized rows (merged by castcse).
    assert any(record["table"] + "__dc" in pair[0] for pair in report["merged_casts"])


@pytest.mark.parametrize("window", [1024, 256])
def test_rotary_packs_replace_the_attention_operand_chains(window):
    ir = _ir(1, window)
    base, rows = _lower(None, ir), _lower("rpack+rpackb", ir)
    report = rows.manifest["norm_fusion"]
    packs = {r["operand"]: r for r in report["rotary_packs"]}
    assert sorted((r["rotary"], r["group"]) for r in packs.values()) == [(False, 2), (True, 1), (True, 2)]
    modes = sorted(r["mode"] for r in report["rotary_unpacks"])
    assert modes == ["pad", "rotary", "rotary", "sum"]
    kinds = {p["kind"] for p in rows.manifest["phases"]}
    assert {"rotary_pack", "rotary_unpack"} <= kinds
    assert len(rows.manifest["barrier_intervals"]) < len(base.manifest["barrier_intervals"])
    # Captured expressions and roundings (rotary forward and backward).
    assert "__float2bfloat16_rn((m5_ + (m6_ * 1.0f)))" in rows.source
    assert "__float2bfloat16_rn((-__bfloat162float(sp_))))" in rows.source
    assert "o_[k] = __float2bfloat16_rn((__bfloat162float(a_) + (__bfloat162float(cp_) * 1.0f)));" in rows.source


def test_rotary_packs_fall_back_without_rotate_half():
    from dataclasses import replace
    from megakernel_lab.operation_compiler import norm_fusion
    ir = _ir()
    ops = [replace(op, target="aten.clone.default") if op.target == "aten.neg.default" else op for op in ir.operations]
    with norm_fusion.scope("rpack+rpackb"):
        _, report = norm_fusion.apply_ir(replace(ir, operations=ops))
    assert all(not r["rotary"] for r in report.get("rotary_packs", []))
    assert all(r["mode"] != "rotary" for r in report.get("rotary_unpacks", []))
