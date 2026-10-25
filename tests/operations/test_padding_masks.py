"""Release step 09 CPU contracts: boolean SDPA padding masks (capture, matching, addressing, emission).

Device numerics (forward, gradients, complete AdamW updates, forbidden-key invariance) are checked on an H100 by
research/latch_release/masks/jobs.py; these tests pin the compiler-side contract on pinned cloud CPU.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import padding_masks as PM
from megakernel_lab.operation_compiler import standard_ops as S
from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training
from megakernel_lab.operation_compiler.ir import CaptureError


class _Layout:
    def __init__(self, shape, strides=None):
        self.shape = tuple(shape)
        if strides is None:
            strides, step = [], 1
            for n in reversed(self.shape):
                strides.insert(0, step)
                step *= n
        self.strides = tuple(strides)


def test_mask_addressing_resolves_head_broadcast_without_copies():
    # key padding [B, 1, 1, T] read by bh = B*H heads: batch stride T, no query stride, head divisor H
    s = PM.resolve(_Layout((3, 1, 1, 8)), (3, 4), 12, 8)
    assert tuple(s) == (8, 0, 1) and s.heads == 4 and s.batches == 3
    # per-example [B, 1, T, T] (e.g. causal & padding)
    s = PM.resolve(_Layout((3, 1, 8, 8)), (3, 4), 12, 8)
    assert tuple(s) == (64, 8, 1) and s.heads == 4 and s.batches == 3
    # one mask for every example and head: linear form with no batch stride
    s = PM.resolve(_Layout((1, 1, 8, 8)), (3, 4), 12, 8)
    assert tuple(s) == (0, 8, 1) and s.heads == 1 and s.batches == 1
    # full [B, H, T, T] merges into the original linear (batch*heads) form
    s = PM.resolve(_Layout((3, 4, 8, 8)), (3, 4), 12, 8)
    assert tuple(s) == (64, 8, 1) and s.heads == 1 and s.batches == 12
    # per-head masks shared by every example are not linear in bh: refused (ordinary lowering)
    assert PM.resolve(_Layout((1, 4, 8, 8)), (3, 4), 12, 8) is None
    # wrong key extent
    assert PM.resolve(_Layout((3, 1, 1, 7)), (3, 4), 12, 8) is None


def test_composition_matches_pytorch_boolean_sdpa_including_rows_without_keys():
    g = torch.Generator().manual_seed(3)
    b, h, t, d = 3, 2, 9, 16
    valid = torch.tensor([[1] * 9, [1] * 4 + [0] * 5, [0] * 9], dtype=torch.bool)
    causal = torch.arange(t)[None, :] <= torch.arange(t)[:, None]
    left = torch.tensor([[0] * 3 + [1] * 6, [1] * 9, [0] * 8 + [1]], dtype=torch.bool)
    for mask in (valid[:, None, None, :], causal[None, None] & valid[:, None, None, :],
                 causal[None, None] & left[:, None, None, :], valid[:, None, None, :].expand(b, h, t, t)):
        q, k, v = (torch.randn(b, h, t, d, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(3))
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = S.sdpa_composition(q, k, v, attn_mask=mask)
        assert torch.allclose(out, ref, rtol=1e-12, atol=1e-12) and torch.isfinite(out).all()
        gy = torch.randn(ref.shape, generator=g, dtype=torch.float64)
        expected = torch.autograd.grad(ref, (q, k, v), gy)
        actual = torch.autograd.grad(out, (q, k, v), gy)
        assert all(torch.allclose(a, e, rtol=1e-11, atol=1e-11) for a, e in zip(actual, expected))
        rows = ~mask.expand(b, h, t, t).any(-1)
        assert bool((out[rows] == 0).all()) and bool((actual[0][rows] == 0).all())


def test_capture_refuses_mask_with_is_causal_and_non_broadcast_masks():
    # LATCH-ATTN-RECOG admits additive float masks of the query dtype (test_attention_recognition.py).
    q = torch.randn(1, 2, 4, 8)
    for kwargs in ({"attn_mask": torch.zeros(1, 1, 1, 4, dtype=torch.float64)},
                   {"attn_mask": torch.zeros(1, 1, 1, 4), "is_causal": True},
                   {"attn_mask": torch.ones(1, 1, 1, 4, dtype=torch.bool), "is_causal": True},
                   {"attn_mask": torch.ones(1, 1, 1, 5, dtype=torch.bool)},
                   {"attn_mask": torch.ones(1, 1, 4, 1, dtype=torch.bool)}):
        with pytest.raises(CaptureError):
            S.sdpa_composition(q, q, q, **kwargs)


def _graph(*, bf16, b=2, t=64, d=128, h=2, causal=False):
    g = torch.Generator().manual_seed(9)
    params = {n: torch.randn(d, d, generator=g) / math.sqrt(d) for n in ("wq", "wk", "wv", "wo")}
    x = torch.randn(b, t, d, generator=g)
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[1, t // 2:] = False

    def linear(v, w):
        return (v.to(torch.bfloat16) @ w.to(torch.bfloat16)).float() if bf16 else v @ w

    def loss(p, x, valid):
        def heads(w):
            return linear(x.reshape(b * t, d), w).reshape(b, t, h, d // h).transpose(1, 2)
        mask = valid[:, None, None, :]
        if causal:
            pos = torch.arange(t)
            mask = (pos[None, :] <= pos[:, None])[None, None] & mask
        o = F.scaled_dot_product_attention(heads(p["wq"]), heads(p["wk"]), heads(p["wv"]), attn_mask=mask)
        return linear(o.transpose(1, 2).reshape(b * t, d), p["wo"]).square().mean()
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    return capture_training(loss, params, momentum, (x, valid), SGDConfig(lr=0.01, momentum=0.9)).ir


def _plan(graph, masks):
    from megakernel_lab.operation_compiler import attention_adapters as AA, attention_lower
    operations, values, names, ref, field, output_name, protected = AA._graph_parts(graph)
    return attention_lower.plan(operations, values, names, ref=ref, field=field, output_name=output_name,
                                protected=protected, masks=masks), values, field


@pytest.mark.parametrize("causal", [False, True])
def test_capture_writes_safe_softmax_and_matcher_absorbs_head_expansion(causal):
    graph = _graph(bf16=True, causal=causal)
    bases = [str(getattr(op, "target", "")).split(".")[-2] for op in graph.operations]
    assert bases.count("_safe_softmax") == 1 and bases.count("_softmax") == 0
    (groups, refusals), values, field = _plan(graph, masks=True)
    assert len(groups) == 1 and not refusals
    group = groups[0]
    assert group["safe"] and tuple(group["mask_lead"]) == (2, 2) and group.get("backward")
    source_shape = tuple(int(x) for x in field(values[group["mask"]], "shape"))
    assert source_shape == ((2, 1, 64, 64) if causal else (2, 1, 1, 64))
    assert len(group["mask_members"]) == 3 and set(group["mask_members"]) <= group["skip"]
    # Without masks=True the safe softmax is not claimed by any host that does not implement it.
    (old, _), _, _ = _plan(graph, masks=False)
    assert old == []


def test_unmasked_groups_are_identical_with_and_without_mask_matching():
    from .test_attention_adapters import _graph as unmasked
    graph = unmasked(bf16=True)
    (new, a), _, _ = _plan(graph, masks=True)
    (old, b), _, _ = _plan(graph, masks=False)
    assert [g["label"] for g in new] == [g["label"] for g in old] and a == b
    assert all("safe" not in g and "mask_lead" not in g for g in new)
    assert [sorted(g["skip"]) for g in new] == [sorted(g["skip"]) for g in old]


def test_bodies_receive_head_divisor_and_safe_rows():
    from megakernel_lab.operation_compiler.lower import lower
    graph = _graph(bf16=True)
    composed = lower(graph, execution="compose")
    assert [g["decision"] for g in composed.manifest["attention_groups"]] == ["adapter:bf16b_short"]
    assert composed.source.count("A.m_heads = 2;") >= 2 and composed.source.count("A.safe_rows = 1;") >= 2
    legacy = lower(graph, execution="legacy")  # reference host: P-OPS-D FP32 tiles on widened copies
    decisions = legacy.manifest["attention_groups"]
    assert decisions and decisions[0]["decision"] == "fused_fp32_attention" and decisions[0]["variant"] == "v1"
    assert decisions[0]["padding_mask"]["safe_rows"]
    assert "A.m_heads = 2;" in legacy.source and "A.safe_rows = 1;" in legacy.source
    # the [B*H, 1, T] expanded copy is not materialized by either program
    for program in (composed, legacy):
        assert not any(p.get("targets") and any("expand" in t for t in p["targets"]) for p in program.manifest["phases"])


def test_long_body_receives_per_example_mask():
    from megakernel_lab.operation_compiler.lower import lower
    graph = _graph(bf16=True, t=256, d=128, h=2, causal=True)
    for program in (lower(graph, execution="legacy"), lower(graph, execution="compose")):
        kinds = {p["kind"] for p in program.manifest["phases"]}
        assert "attention_long_forward" in kinds, kinds
        assert "A.m_heads = 2;" in program.source and "A.safe_rows = 1;" in program.source


def test_fp32_short_keys_use_generic_safe_softmax_and_unfused_lowering_is_complete():
    from megakernel_lab.operation_compiler.lower import lower
    graph = _graph(bf16=False)
    program = lower(graph, execution="legacy")  # all-FP32, <= 512 keys: materialized products + softmax phases
    assert "maximum == -INFINITY ? 0.0f" in program.source
    unfused = lower(_graph(bf16=True), execution="legacy", fuse_attention=False)
    assert "maximum == -INFINITY ? 0.0f" in unfused.source and "pshort::" not in unfused.source


def test_expansion_shared_after_cse_is_claimed_once():
    """P6-D's CSE merges identical per-layer mask expansions; both groups still read the per-example source."""
    from megakernel_lab.operation_compiler.p6d_rules import cse
    g = torch.Generator().manual_seed(4)
    b, t, d, h = 2, 256, 128, 2
    params = {n: torch.randn(d, d, generator=g) / math.sqrt(d) for n in ("wq", "wk", "wv", "wq2", "wk2", "wv2")}
    x = torch.randn(b, t, d, generator=g)
    valid = torch.ones(b, t, dtype=torch.bool)
    valid[0, 100:] = False

    def linear(v, w):
        return (v.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()

    def loss(p, x, valid):
        mask = valid[:, None, None, :]
        y = x
        for suffix in ("", "2"):
            def heads(w):
                return linear(y.reshape(b * t, d), w).reshape(b, t, h, d // h).transpose(1, 2)
            o = F.scaled_dot_product_attention(heads(p["wq" + suffix]), heads(p["wk" + suffix]),
                                               heads(p["wv" + suffix]), attn_mask=mask)
            y = y + linear(o.transpose(1, 2).reshape(b * t, d), p["wq" + suffix]).reshape(b, t, d)
        return y.square().mean()
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    graph = capture_training(loss, params, momentum, (x, valid), SGDConfig(lr=0.01, momentum=0.9)).ir
    merged, report = cse(graph)
    assert report["merged"] > 0
    (groups, refusals), _, _ = _plan(merged, masks=True)
    assert len(groups) == 2 and all(tuple(g["mask_lead"]) == (2, 2) for g in groups)
    assert groups[0]["mask"] == groups[1]["mask"]
    assert all(g["mask_shared"] and not g["mask_members"] for g in groups)
    from megakernel_lab.operation_compiler.lower import lower
    program = lower(graph, execution="legacy")  # selects P6-D (CSE inside) for these long BF16-boundary groups
    assert program.source.count("A.m_heads = 2;") >= 8, program.source.count("A.m_heads = 2;")
    assert program.source.count("pflatlong::forward_item") >= 2 and "pattn::" not in program.source
