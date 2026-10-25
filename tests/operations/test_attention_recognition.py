"""LATCH-ATTN-RECOG CPU contracts: ordinary attention spellings reach the existing BF16-boundary bodies.

Capture/lowering only (pinned cloud CPU). Device numerics, restart and timing are checked by
research/latch_release/attn_recog (spelling-versus-qualified-expression and conventional controls).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import attention_adapters as AA
from megakernel_lab.operation_compiler import standard_ops as S
from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training

BF16 = torch.bfloat16


def _math(q, k, v, **kwargs):
    return torch.ops.aten._scaled_dot_product_attention_math(q, k, v, **kwargs)[0]


@pytest.mark.parametrize("causal", [False, True])
def test_bf16_sdpa_is_the_fp32_composition_rounded_once(causal):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(2, 3, 40, 32, generator=g).to(BF16) for _ in range(3))
    out = S.sdpa_composition(q, k, v, is_causal=causal)
    assert out.dtype == BF16
    exact = S.sdpa_composition(q.float(), k.float(), v.float(), is_causal=causal).to(BF16)
    assert torch.equal(out, exact)
    # PyTorch's math reference rounds only its output (it splits the scale over q and k, a different FP32
    # rounding of the scores); both stay within one BF16 rounding of each other.
    ref = _math(q, k, v, is_causal=causal)
    assert ref.dtype == BF16 and (out.float() - ref.float()).abs().max() <= 2 ** -7 * ref.float().abs().max()
    qs, ks, vs = (x.clone().requires_grad_() for x in (q, k, v))
    grads = torch.autograd.grad(S.sdpa_composition(qs, ks, vs, is_causal=causal), (qs, ks, vs),
                                torch.ones(out.shape, dtype=BF16))
    assert all(x.dtype == BF16 for x in grads)


def test_float_mask_matches_math_reference_including_fully_masked_rows():
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(2, 2, 12, 16, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(3))
    for mask in (torch.triu(torch.full((12, 12), -10000.0, dtype=torch.float64), 1),
                 torch.randn(2, 1, 12, 12, generator=g, dtype=torch.float64),
                 torch.zeros(12, 12, dtype=torch.float64).index_fill_(0, torch.tensor([3]), float("-inf"))):
        ref = _math(q, k, v, attn_mask=mask)
        out = S.sdpa_composition(q, k, v, attn_mask=mask)
        finite = torch.isfinite(ref)
        assert torch.equal(finite, torch.isfinite(out))
        assert torch.allclose(out[finite], ref[finite], rtol=1e-12, atol=1e-12)
        if S.FLOAT_MASK_SOFTMAX == "safe":
            assert torch.isfinite(out).all()


def _graph(spelling, *, t=256, d=256, h=4, b=2, mask="causal"):
    """One attention layer and output projection in the explicit BF16 projection recipe, spelled ``spelling``."""
    g = torch.Generator().manual_seed(7)
    hd = d // h
    params = {"wqkv": torch.randn(d, 3 * d, generator=g) / math.sqrt(d),
              "wo": torch.randn(d, d, generator=g) / math.sqrt(d)}
    x = torch.randn(b, t, d, generator=g)
    additive = torch.triu(torch.full((t, t), -10000.0), diagonal=1)

    def bf16_product(v, w):
        return v.to(BF16) @ w.to(BF16)

    def loss(p, x, additive):
        flat = x.reshape(b * t, d)
        sdpa = {"is_causal": True} if mask == "causal" else {}
        if spelling == "bmm_add":
            def heads(j):
                return (bf16_product(flat, p["wqkv"][:, j * d:(j + 1) * d]).float()
                        .reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd))
            s = torch.bmm(heads(0), heads(1).transpose(1, 2)) * (1.0 / math.sqrt(hd))
            s = s + additive if mask == "causal" else s
            o = torch.bmm(torch.softmax(s, -1), heads(2)).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
        elif spelling == "sdpa_bf16":
            def heads(j):
                return bf16_product(flat, p["wqkv"][:, j * d:(j + 1) * d]).view(b, t, h, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(heads(0), heads(1), heads(2), **sdpa).transpose(1, 2).reshape(b * t, d)
        elif spelling == "matmul4d":
            def heads(j):
                return bf16_product(flat, p["wqkv"][:, j * d:(j + 1) * d]).float().view(b, t, h, hd).transpose(1, 2)
            s = heads(0) @ heads(1).transpose(-2, -1) * (1.0 / math.sqrt(hd))
            s = s + additive if mask == "causal" else s
            o = (torch.softmax(s, dim=-1) @ heads(2)).transpose(1, 2).reshape(b * t, d)
        elif spelling == "sdpa_float":
            def heads(j):
                return bf16_product(flat, p["wqkv"][:, j * d:(j + 1) * d]).float().view(b, t, h, hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(heads(0), heads(1), heads(2), attn_mask=additive)
            o = o.transpose(1, 2).reshape(b * t, d)
        else:
            packed = bf16_product(flat, p["wqkv"])
            if spelling == "packed_heads":
                q, k, v = packed.float().view(b, t, 3, h, hd).permute(2, 0, 3, 1, 4).unbind(0)
            else:
                parts = (packed.float().chunk(3, dim=-1) if spelling == "packed_chunk" else
                         packed.split(d, dim=-1) if spelling == "packed_bf16" else
                         (packed.float()[:, :d], packed.float()[:, d:2 * d], packed.float()[:, 2 * d:]))
                q, k, v = (x_.view(b, t, h, hd).transpose(1, 2) for x_ in parts)
            o = F.scaled_dot_product_attention(q, k, v, **sdpa).transpose(1, 2).reshape(b * t, d)
        return bf16_product(o, p["wo"]).float().square().mean()
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    return capture_training(loss, params, momentum, (x, additive), SGDConfig(lr=0.01, momentum=0.9)).ir


@pytest.mark.parametrize("spelling", ["sdpa_bf16", "sdpa_float", "matmul4d", "packed_heads", "packed_chunk",
                                      "packed_slices", "packed_bf16"])
@pytest.mark.parametrize("t,body", [(256, "bf16b_flat_long"), (64, "bf16b_short")])
def test_ordinary_spellings_select_the_bf16_boundary_bodies(spelling, t, body):
    survey = AA.survey(_graph(spelling, t=t), target_arch="sm_90")
    (row,) = survey["groups"]
    assert row["operand_class"] == "bf16_boundary", row
    assert row["candidates"][body]["admitted"], row
    assert row["selected"] == body


@pytest.mark.parametrize("spelling", ["packed_heads", "packed_chunk", "packed_slices"])
def test_packed_operands_are_read_in_place_and_gradients_cast_per_operand(spelling):
    graph = _graph(spelling)
    records = graph.metadata.get("attention_canonical", {}).get("narrowed_gradient_assemblies", [])
    # one assembly of three gradients, or (slices of separately widened copies) three single placements
    assert sum(r["pieces"] for r in records) == 3, graph.metadata.get("attention_canonical")
    from megakernel_lab.operation_compiler.lower import lower
    program = lower(graph, execution="compose")
    decisions = program.manifest["attention_groups"]
    assert [g["decision"] for g in decisions] in (["compose_long_bf16_boundary_attention"],
                                                  ["long_bf16_boundary_attention"]), decisions
    # q/k/v are read from the one packed BF16 projection result at element offsets d and 2d (no operand copies)
    assert " + 256LL; A.lk = " in program.source and " + 512LL; A.lv = " in program.source


def test_unpacked_programs_are_unchanged_by_the_assembly_rewrite():
    graph = _graph("bmm_add")
    assert "attention_canonical" not in graph.metadata


def test_rank4_matmul_attention_is_lifted_to_the_rank3_products():
    from megakernel_lab.operation_compiler.lower import _field
    graph = _graph("matmul4d")
    regions = graph.metadata.get("attention_canonical", {}).get("rank3_score_regions", [])
    assert len(regions) == 1, graph.metadata.get("attention_canonical")
    base = lambda op: str(_field(op, "target", "")).removeprefix("aten.").split(".")[0]
    softmax = [_field(op, "outputs", [_field(op, "name")])[0] for op in graph.operations
               if base(op) in ("_softmax", "_softmax_backward_data")]
    assert len(softmax) == 2 and all(len(_field(graph.values[n], "shape")) == 3 for n in softmax)
