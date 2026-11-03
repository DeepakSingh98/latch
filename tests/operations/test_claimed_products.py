"""Claimed segmented-product tokens (ops/claimed_products.py) lowering coverage (CPU).

* Tokens off: the generated source is identical to the program without the module's rewrites.
* 'scatter' alone keeps the plain row-scatter body; with 'bsearch' it takes latch_grp::tflat_claim_scatter_rb (same
  call form, +512 shared bytes for the row table).
* 'bsearch': claimed right products take the binary-search bodies (claim_tasks.cuh, staged after its header chain).
* 'fit': claimed weight-gradient products change only their tile template arguments by the padded-area rule; the rule
  is the identity for tight shapes and with 'fit' off.
* 'route': the routing rewrite keeps the staged-epilogue macro ahead of every claimed-body header.
* Feature dependencies are checked.
"""
from __future__ import annotations

import pytest
import torch

from .test_pack_fusion import FEATURES, _unpack


def _lower(monkeypatch, module, args, state, products):
    from megakernel_lab.operation_compiler import capture_module, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", 0)
    captured = capture_module(module, lambda y: y.float().square().mean(), args, optimizer=SGDConfig(), state=state)
    with products_scope(products):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def _experts(tokens=64, choices=3, width=128, hidden=160, groups=8):
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, route_order

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1032)
            self.x = torch.nn.Parameter(torch.randn(tokens * choices, width))
            self.up = torch.nn.Parameter(torch.randn(groups, width, hidden))
            self.down = torch.nn.Parameter(torch.randn(groups, hidden, width))

        def forward(self, selected, **state):
            order, offsets, inverse = route_order(selected, groups)
            h = ragged_mm(self.x.to(torch.bfloat16), self.up.to(torch.bfloat16), offsets)
            y = ragged_mm(torch.nn.functional.silu(h.float()).to(torch.bfloat16), self.down.to(torch.bfloat16), offsets)
            return torch.nn.functional.embedding(inverse, y).float()

    selected = (torch.arange(tokens * choices) * 5 % groups).view(tokens, choices)
    return Experts(), (selected,), None


@pytest.mark.parametrize("kind", ["permutation", "embedding"])
def test_tokens_off_source_unchanged(monkeypatch, kind):
    from megakernel_lab.operation_compiler import claimed_products
    base = _lower(monkeypatch, *_unpack(kind), FEATURES + "+scatter")
    monkeypatch.setattr(claimed_products, "rewrite", lambda emitted: [])
    monkeypatch.setattr(claimed_products, "claimed_tile", lambda **kw: kw["current"])
    plain = _lower(monkeypatch, *_unpack(kind), FEATURES + "+scatter")
    assert base.source == plain.source
    assert "latch_grp::" not in base.source and "claim_tasks.cuh" not in base.source


@pytest.mark.parametrize("kind", ["permutation", "embedding"])
def test_scatter_bodies(monkeypatch, kind):
    plain = _lower(monkeypatch, *_unpack(kind), FEATURES + "+scatter")
    rows = _lower(monkeypatch, *_unpack(kind), FEATURES + "+scatter+bsearch")
    assert plain.source.count("latch_pack::tflat_claim_scatter<") == 1 and "latch_grp::" not in plain.source
    assert rows.source.count("latch_grp::tflat_claim_scatter_rb<") == 1
    assert "latch_pack::tflat_claim_scatter<" not in rows.source
    assert '#include "claim_tasks.cuh"' in rows.source
    # Only the body name changes: identical template and runtime arguments.
    norm = lambda s: s.replace("latch_grp::tflat_claim_scatter_rb<", "latch_pack::tflat_claim_scatter<")
    body = lambda s: [ln for ln in norm(s).splitlines() if "tflat_claim_scatter<" in ln]
    assert body(plain.source) == body(rows.source)
    (p,) = [p for p in rows.manifest["phases"] if p.get("pack_scatter")]
    (q,) = [p for p in plain.manifest["phases"] if p.get("pack_scatter")]
    assert p["engine"] == "ragged_flat_tma_claim_scatter_rows_wgmma" and p["shared_bytes"] == q["shared_bytes"] + 512
    assert q["engine"] == "ragged_flat_tma_claim_scatter_wgmma"


def test_tile_fit_templates(monkeypatch):
    base = _lower(monkeypatch, *_experts(), FEATURES)
    fit = _lower(monkeypatch, *_experts(), FEATURES + "+fit")
    outer = lambda prog: sorted(p["claim_outer"]["tmpl"] for p in prog.manifest["phases"] if p.get("claim_outer"))
    assert outer(base) == ["8,128,128,2,4"] * 2
    # Few tasks: both weight gradients take the narrow 128 x 64 tile; the [160, 128] down-bank gradient then fits
    # 64 x 128 tiles (192 instead of 256 padded rows).
    assert outer(fit) == ["8,128,64,2,6", "8,64,128,1,6"]
    names = lambda prog: sorted(n for p in prog.manifest["phases"] if p.get("kind") == "gemm" for n in p["names"])
    assert names(base) == names(fit)                       # same products, only tiles differ


def test_tile_rule(monkeypatch):
    from megakernel_lab.operation_compiler.claimed_products import claimed_tile
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    cases = [(352, 160, (128, 128, 2, 4)), (160, 224, (128, 64, 2, 6))]
    with products_scope(FEATURES):
        for m, n, cur in cases:
            assert claimed_tile(outer=True, kmajor=False, m=m, n=n, current=cur) == cur
        assert claimed_tile(outer=False, kmajor=False, m=0, n=160, current=(128, 128, 2, 4)) == (128, 128, 2, 4)
    with products_scope(FEATURES + "+fit"):
        for m, n in ((352, 352), (352, 96), (352, 224), (96, 352), (352, 64)):
            cur = (128, 64, 2, 6) if 64 in (m, n) or m == 96 else (128, 128, 2, 4)
            assert claimed_tile(outer=True, kmajor=False, m=m, n=n, current=cur) == cur
        # Ties: larger tiles for short segments, smaller tiles (more parallel k chains) for long ones.
        assert claimed_tile(outer=True, kmajor=False, m=352, n=160, current=(128, 128, 2, 4)) == (128, 192, 2, 4)
        assert claimed_tile(outer=True, kmajor=False, m=352, n=160, current=(128, 128, 2, 4),
                            rows=8 * 4096, groups=8) == (128, 64, 2, 6)
        assert claimed_tile(outer=True, kmajor=False, m=160, n=224, current=(128, 64, 2, 6)) == (64, 256, 1, 4)
        assert claimed_tile(outer=True, kmajor=False, m=160, n=224, current=(128, 64, 2, 6),
                            rows=8 * 4096, groups=8) == (64, 128, 1, 6)
        # N-major right products refit only with 'bsearch'.
        assert claimed_tile(outer=False, kmajor=False, m=0, n=160, current=(128, 128, 2, 4)) == (128, 128, 2, 4)
    with products_scope(FEATURES + "+fit+bsearch"):
        for n, bn in ((160, 160), (136, 160), (96, 96), (64, 64), (128, 128), (200, 128)):
            assert claimed_tile(outer=False, kmajor=False, m=0, n=n, current=(128, 128, 2, 4)) == (128, bn, 2, 4)
        assert claimed_tile(outer=False, kmajor=True, m=0, n=160, current=(128, 160, 2, 4)) == (128, 160, 2, 4)


def test_pair_bn_limit():
    from megakernel_lab.operation_compiler.claimed_products import pair_bn_limit
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    for products, limit in ((None, None), (FEATURES, None), (FEATURES + "+fit", 128), (FEATURES + "+bsearch", 128),
                            (FEATURES + "+fit+bsearch", 160)):
        with products_scope(products):
            assert pair_bn_limit() == limit, products


def test_bsearch_call_forms_and_headers(monkeypatch):
    base = _lower(monkeypatch, *_experts(), FEATURES)
    bs = _lower(monkeypatch, *_experts(), FEATURES + "+bsearch")
    assert "latch_grp::" not in base.source
    assert "latch_ragged::tflat_claim<" not in bs.source and "latch_ragged::tflat2_claim<" not in bs.source
    assert "latch_grp::tflat_claim_h<" in bs.source or "latch_grp::tflat2_claim_h<" in bs.source
    assert "pack_fusion.cuh" in bs.manifest["header_hashes"] and "claim_tasks.cuh" in bs.manifest["header_hashes"]
    src = bs.source
    position = src.index('#include "claim_tasks.cuh"')
    for header in ("ragged_tma.cuh", "ragged_claim.cuh", "pack_fusion.cuh"):
        assert src.index(f'#include "{header}"') < position, header
    norm = lambda s: (s.replace("latch_grp::tflat_claim_h<", "latch_ragged::tflat_claim<")
                      .replace("latch_grp::tflat2_claim_h<", "latch_ragged::tflat2_claim<"))
    calls = lambda s: [ln for ln in norm(s).splitlines() if "claim<" in ln]
    assert calls(base.source) == calls(src)                # same templates and runtime arguments
    assert any(p.get("task_bsearch") for p in bs.manifest["phases"])


def test_route_keeps_staged_epilogue_macro_first(monkeypatch):
    """The routing rewrite (route_order before any claimed product) must not include the claimed bodies ahead of
    LATCH_CLAIM_SEPI (every claimed product would otherwise compile without the staged epilogue)."""
    prog = _lower(monkeypatch, *_experts(), FEATURES + "+route+bsearch")
    src = prog.source
    assert "latch_grp::route_order<" in src or "latch_grp::route_counts<" in src
    sepi = src.index("#define LATCH_CLAIM_SEPI 1")
    for header in ("ragged_claim.cuh", "pack_fusion.cuh", "claim_tasks.cuh"):
        if f'#include "{header}"' in src:
            assert sepi < src.index(f'#include "{header}"'), header
    assert '#include "bucketing.cuh"' in src
    routed = [p for p in prog.manifest["phases"] if p.get("route_rows")]
    assert routed and all(p["route_fallback"].startswith("latch_ragged::") for p in routed)
    plain = _lower(monkeypatch, *_experts(), FEATURES)
    assert "bucketing.cuh" not in plain.source and "latch_grp::" not in plain.source


@pytest.mark.parametrize("value", ["flat+ktail", "flat+claim+fit", "flat+tma+fit", "flat+tma+bsearch",
                                   "flat+claim+scatter", "flat+tma+claim+kcat"])
def test_feature_dependencies(value):
    from megakernel_lab.operation_compiler.ragged_experts_lower import validate_products
    with pytest.raises(ValueError, match="requires"):
        validate_products(value)
