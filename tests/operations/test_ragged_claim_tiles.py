"""Claimed segmented-product tiles and observed products (CPU lowering coverage).

* 'ktail': N-major stacked weights with K % 64 != 0 take the TMA (claimed) bodies; the kernel zeroes the k-tail rows.
  Without it they keep the cp.async body.
* Observed products (executed-work counters) take the claimed bodies, and their counter records describe the claimed
  tile.
* 'fit': claimed K-major right products fit the column tile to N (160 / 96 / 64; 800 = 5 x 160; 384 keeps 128), and
  task-starved claimed weight-gradient products take 128 x 64 tiles. Without it the release tiles are kept.
"""
from __future__ import annotations

import re

import torch

from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, ragged_mm_observed

FEATURES = "flat+tma+rotate+claim+sepi"


def _program(make, monkeypatch, *, min_rows=0, products=FEATURES, **patches):
    from megakernel_lab.operation_compiler import capture_module, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", min_rows)
    for name, value in patches.items():
        monkeypatch.setattr(ragged_experts_lower, name, value)
    module, args, state = make()
    captured = capture_module(module, lambda y: y.float().square().mean(), args, optimizer=SGDConfig(), state=state)
    with products_scope(products):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def _experts(groups=4, rows=64, width=224, hidden=64):
    """Two segmented products with N-major stacked banks [G, K, N]."""
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1028)
            self.x = torch.nn.Parameter(torch.randn(rows, width))
            self.up = torch.nn.Parameter(torch.randn(groups, width, hidden))
            self.down = torch.nn.Parameter(torch.randn(groups, hidden, width))

        def forward(self, offsets):
            h = ragged_mm(self.x.to(torch.bfloat16), self.up.to(torch.bfloat16), offsets)
            return ragged_mm(h, self.down.to(torch.bfloat16), offsets)

    # Uneven segments with an empty one: [10, 0, 30, rows - 40].
    offsets = torch.tensor([0, 10, 10, 40, rows])
    assert groups == 4
    return lambda: (Experts(), (offsets,), None)


def _projection(groups=8, rows=96, width=128, head=96):
    """Observed projection pair with K-major stacked banks [G, N, K] read transposed."""
    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1029)
            self.x = torch.nn.Parameter(torch.randn(rows, width))
            self.wq = torch.nn.Parameter(torch.randn(groups, head, width))
            self.wo = torch.nn.Parameter(torch.randn(groups, width, head))

        def forward(self, offsets, *, state):
            q, work_q = ragged_mm_observed(self.x.to(torch.bfloat16), self.wq.transpose(1, 2).to(torch.bfloat16), offsets)
            out, work_o = ragged_mm_observed(q, self.wo.transpose(1, 2).to(torch.bfloat16), offsets)
            return out, dict(work_q=work_q, work_o=work_o)

    bounds = [0] + [min(rows, (i + 1) * rows // groups - (3 if i % 3 == 0 else 0)) for i in range(groups - 1)] + [rows]
    state = dict(work_q=torch.zeros(groups, 4, dtype=torch.int64), work_o=torch.zeros(groups, 4, dtype=torch.int64))
    return lambda: (Projection(), (torch.tensor(bounds),), state)


def _gemms(program):
    return [p for p in program.manifest["phases"] if p.get("kind") == "gemm"]


def test_n_major_k_tail_products_take_claimed_tma_bodies(monkeypatch):
    program = _program(_experts(), monkeypatch, products=FEATURES + "+ktail")
    # Forward products: N-major banks with K = 224 and 64 (224 % 64 = 32 needs the zeroed k tail).
    calls = re.findall(r"latch_ragged::(tflat2?_claim|tflat2?|flat2?)<([^>]*)>", program.source)
    assert calls and all(name.endswith("_claim") for name, _ in calls)
    assert any(args.split(",")[4] == "false" for name, args in calls if name == "tflat_claim")   # BKC false
    assert "latch_ragged::flat<" not in program.source and "latch_ragged::flat2<" not in program.source
    off = _program(_experts(), monkeypatch)
    assert "latch_ragged::flat" in off.source                         # the cp.async body without the k-tail rule


def test_observed_products_are_claimed_and_record_claimed_tiles(monkeypatch):
    # The q input gradient reads its bank N-major with K = 96, so it takes the claimed body through 'ktail'.
    program = _program(_projection(), monkeypatch, products=FEATURES + "+ktail")
    records = program.manifest["ragged_projection_work"]
    assert sorted(r["work_product"] for r in records) == ["outer", "outer", "right", "right", "right", "right"]
    assert all("claim" in r["engine"] for r in records)
    for r in records:
        phase = next(p for p in _gemms(program) if r["name"] in p["writes"])
        assert "claimed" in phase["work_instrumentation"] and phase["work_output_tile"] == r["work_output_tile"]
    # Every claimed observed call passes its counter pointer; no static observed body remains.
    assert not re.search(r"latch_ragged::(tflat|touter|flat)<", program.source)
    static = _program(_projection(), monkeypatch, products=FEATURES + "+ktail", OBSERVED_CLAIM=False)
    assert "latch_ragged::tflat<" in static.source or "latch_ragged::flat<" in static.source
    assert all("claim" not in r["engine"] for r in static.manifest["ragged_projection_work"])


def test_claimed_k_major_products_fit_their_column_tile():
    from megakernel_lab.operation_compiler.ragged_experts_lower import fit_bn
    assert [fit_bn(n, 128) for n in (64, 96, 128, 160, 320, 384, 800, 768)] == [64, 96, 128, 160, 160, 128, 160, 128]


def _tiles(program):
    tiles = {}
    for p in _gemms(program):
        if p.get("work_output_tile"):
            tiles.setdefault(tuple(p["matrix_shape"]), set()).add(tuple(p["work_output_tile"]))
    return tiles


def test_observed_projection_tiles(monkeypatch):
    tiles = _tiles(_program(_projection(head=160), monkeypatch, products=FEATURES + "+ktail+fit"))
    rows, width, head = 96, 128, 160
    # Forward q (K-major bank, N = 160) takes one 160-wide tile; the o input gradient has the same extents but an
    # N-major bank (64-wide swizzle chunks), so it keeps 128.
    assert tiles[(rows, head, width)] == {(128, 160), (128, 128)}
    # Weight gradients of 8 banks: 1 x 2 tasks per bank at 128 x 128 -> fewer than 128 -> 128 x 64 tiles; the
    # [160, 128] o-bank gradient then fits 64 x 128 tiles (192 instead of 256 padded rows), the [128, 160] q-bank
    # gradient keeps 128 x 64.
    assert set().union(*(tiles[k] for k in tiles if k[2] == rows)) == {(128, 64), (64, 128)}
    # Without 'fit' every claimed product keeps the 128 x 128 tile.
    plain = _tiles(_program(_projection(head=160), monkeypatch, products=FEATURES + "+ktail"))
    assert plain[(rows, head, width)] == {(128, 128)}
    assert set().union(*(plain[k] for k in plain if k[2] == rows)) == {(128, 128)}
