"""FP32 attention tiles keep an ordinary additive mask bound (regression for the release step-09 field insertion).

The step-09 padding-mask fields were inserted between ``if mask is not None`` and its ``else``, so every ordinary
additive mask was reset to ``nullptr`` / ``has_mask = 0`` while the prepare phase still scheduled its mask-table
items (CUDA illegal address in ``pattn::prepare_item``; LATCH-RELEASE-EVAL lm-e-awk, LATCH-FGM-COMPOSE legacy
build). Lowering-only checks on pinned CPU; device behaviour is checked by research/latch_release/attn_recog.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training

NULL_MASK = "A.mask.p = nullptr;"


def _graph(*, bf16, t, d=128, h=2, b=2, mask="additive"):
    g = torch.Generator().manual_seed(11)
    params = {n: torch.randn(d, d, generator=g) / math.sqrt(d) for n in ("wq", "wk", "wv", "wo")}
    x = torch.randn(b, t, d, generator=g)
    additive = torch.triu(torch.full((t, t), -10000.0), diagonal=1)
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[1, t // 2:] = False
    hd = d // h

    def linear(v, w):
        return (v.to(torch.bfloat16) @ w.to(torch.bfloat16)).float() if bf16 else v @ w

    def loss(p, x, additive, valid):
        if mask == "boolean":
            def heads4(w):
                return linear(x.reshape(b * t, d), w).reshape(b, t, h, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(heads4(p["wq"]), heads4(p["wk"]), heads4(p["wv"]),
                                               attn_mask=valid[:, None, None, :])
            return linear(o.transpose(1, 2).reshape(b * t, d), p["wo"]).square().mean()

        def heads(w):
            return linear(x.reshape(b * t, d), w).reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)
        s = torch.bmm(heads(p["wq"]), heads(p["wk"]).transpose(1, 2)) * (1.0 / math.sqrt(hd))
        if mask == "additive":
            s = s + additive
        o = torch.bmm(torch.softmax(s, -1), heads(p["wv"]))
        o = o.reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
        return linear(o, p["wo"]).square().mean()
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    return capture_training(loss, params, momentum, (x, additive, valid), SGDConfig(lr=0.01, momentum=0.9)).ir


def _programs(graph):
    from megakernel_lab.operation_compiler.lower import lower
    return {"legacy": lower(graph, execution="legacy"), "compose": lower(graph, execution="compose")}


def _fp32_tile_decisions(program):
    return [g for g in program.manifest.get("attention_groups", []) if g.get("decision") == "fused_fp32_attention"]


@pytest.mark.parametrize("bf16,t,variant", [(True, 192, "v1"), (False, 1024, "v2")])
def test_additive_mask_stays_bound_in_fp32_tiles(bf16, t, variant):
    # BF16-boundary T=192 has no BF16 body (short <= 128, long >= 256): FP32 tiles v1 on widened copies.
    # All-FP32 T=1024 (> 512 keys) with head width 64: the stored-score v2 tiles.
    for arm, program in _programs(_graph(bf16=bf16, t=t)).items():
        source = program.source
        assert "A.has_mask = 1;" in source, arm
        assert NULL_MASK not in source and "A.has_mask = 0;" not in source, arm
        if arm == "legacy":
            decisions = _fp32_tile_decisions(program)
            assert decisions and {d["variant"] for d in decisions} == {variant}, decisions


@pytest.mark.parametrize("bf16,t", [(True, 192), (False, 1024)])
def test_unmasked_fp32_tiles_clear_the_mask_binding(bf16, t):
    for arm, program in _programs(_graph(bf16=bf16, t=t, mask="none")).items():
        assert NULL_MASK in program.source and "A.has_mask = 1;" not in program.source, arm


def test_padding_mask_fields_keep_the_mask_pointer():
    # Boolean key padding (release step 09) on the legacy FP32 tiles: extended fields AND the mask binding.
    program = _programs(_graph(bf16=True, t=192, mask="boolean"))["legacy"]
    source = program.source
    assert "A.m_heads = 2;" in source and "A.safe_rows = 1;" in source
    assert "A.has_mask = 1;" in source and NULL_MASK not in source
