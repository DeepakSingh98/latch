"""Routing-pack rewrites (ops/pack_fusion.py) lowering coverage (CPU).

* graph_rules 'castshare': an FP32 row gather read by several identical BF16 casts becomes one table cast and one BF16
  gather; the duplicate casts and the FP32 rows disappear; the gradients' VJP phases are unchanged.
* segmented_products 'scatter' (GEMM -> unpack): products read only by a bijective unpack store at the unpacked rows;
  products that do not take a claimed TMA body keep the standalone unpack.
* Neither selected: the graph and the generated program are unchanged.
"""
from __future__ import annotations

import torch

from megakernel_lab.operation_compiler.ragged_experts import ragged_mm_observed

FEATURES = "flat+tma+rotate+claim+sepi"
SCATTER = FEATURES + "+scatter"


def _qkv(tokens=64, choices=3, width=128, heads=3, groups=8):
    class QKV(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1030)
            self.x = torch.nn.Parameter(torch.randn(tokens, width))
            self.w = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(groups, heads * 32, width)) for _ in range(3)])

        def forward(self, token, offsets, *, state):
            rows = torch.nn.functional.embedding(token, self.x)
            outs, work = [], {}
            for i, w in enumerate(self.w):
                o, wk = ragged_mm_observed(rows.to(torch.bfloat16), w.transpose(1, 2).to(torch.bfloat16), offsets)
                outs.append(o.float())
                work[f"work_{i}"] = wk
            return outs[0] + outs[1] * outs[2], work

    rows = tokens * choices
    token = torch.arange(rows) % tokens
    offsets = torch.tensor([0, 10, 10, 40, 70, 100, 130, 160, rows])
    state = {f"work_{i}": torch.zeros(groups, 4, dtype=torch.int64) for i in range(3)}
    return QKV(), (token, offsets), state


def _captured(make):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.optim import SGDConfig
    module, args, state = make()
    return capture_module(module, lambda y: y.float().square().mean(), args, optimizer=SGDConfig(), state=state)


def _program(monkeypatch, *, share, min_rows=0):
    from megakernel_lab.operation_compiler import graph_rules, ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", min_rows)
    captured = _captured(_qkv)
    with products_scope(FEATURES), graph_rules.scope("castshare" if share else None):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def _phases(program, kind):
    return [p for p in program.manifest["phases"] if p.get("kind") == kind]


def test_rewrite_is_identity_without_tokens(monkeypatch):
    from megakernel_lab.operation_compiler import pack_fusion
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    captured = _captured(_qkv)
    for products in (None, FEATURES):
        with products_scope(products):
            graph, report = pack_fusion.rewrite(captured.ir, target_arch="sm_90")
        assert graph is captured.ir and report is None


def test_rewrite_shares_casts(monkeypatch):
    from megakernel_lab.operation_compiler import graph_rules, pack_fusion
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    captured = _captured(_qkv)
    with products_scope(FEATURES), graph_rules.scope("castshare"):
        graph, report = pack_fusion.rewrite(captured.ir, target_arch="sm_90")
    (shared,) = report["shared_casts"]
    assert len(shared["removed_casts"]) == 3 and not report["scattered_outputs"]
    data = graph.to_dict()
    names = {op["name"] for op in data["operations"]}
    assert shared["table_cast"] in names and not set(shared["removed_casts"]) & names
    assert data["values"][shared["gather"]]["dtype"] == "bfloat16"
    # The gather's VJP still reads only its indices; the duplicate-token sums keep their own (ordered) phases.
    assert any(op["target"] == "aten.embedding_dense_backward.default" for op in data["operations"])
    # Input graph unchanged (the rewrite never mutates its argument).
    assert sum(op.target == "aten._to_copy.default" for op in captured.ir.operations) >= 3


def test_lowering_shares_pack_casts(monkeypatch):
    base = _program(monkeypatch, share=False)
    shared = _program(monkeypatch, share=True)

    def casts(program):
        return [p for p in program.manifest["phases"] if "aten._to_copy.default" in (p.get("targets") or ())
                and any("__bf16_pack_" in n for n in p.get("names") or ())]
    # Packs: base = FP32 gather + three casts; shared = table cast + BF16 gather.
    assert len(_phases(base, "embedding")) == 1 and not casts(base)
    assert len(_phases(shared, "embedding")) == 1 and len(casts(shared)) == 1
    assert "pack_fusion.cuh" not in base.source and "latch_pack::" not in shared.source
    assert shared.manifest["arena_bytes"] <= base.manifest["arena_bytes"]


def _unpack(kind, tokens=64, choices=3, width=128, groups=8):
    """Product -> permutation_gather unpack (observed product) or product -> embedding by route_order's inverse."""
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, route_order
    from megakernel_lab.operation_compiler.row_permutation import permutation_gather

    class Unpack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1031)
            self.x = torch.nn.Parameter(torch.randn(tokens * choices, width))
            self.w = torch.nn.Parameter(torch.randn(groups, width, width))

        def forward(self, selected, **state):
            order, offsets, inverse = route_order(selected, groups)
            rows = self.x.to(torch.bfloat16)
            if kind == "permutation":
                y, work = ragged_mm_observed(rows, self.w.transpose(1, 2).to(torch.bfloat16), offsets)
                return permutation_gather(y, inverse, order).float(), dict(work=work)
            y = ragged_mm(rows, self.w.to(torch.bfloat16), offsets)
            return torch.nn.functional.embedding(inverse, y).float()

    selected = (torch.arange(tokens * choices) * 5 % groups).view(tokens, choices)
    state = dict(work=torch.zeros(groups, 4, dtype=torch.int64)) if kind == "permutation" else None
    return Unpack(), (selected,), state


def _unpack_program(monkeypatch, kind, products, *, min_rows=0):
    from megakernel_lab.operation_compiler import ragged_experts_lower
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    monkeypatch.setattr(ragged_experts_lower, "CLAIM_MIN_ROWS", min_rows)
    captured = _captured(lambda: _unpack(kind))
    with products_scope(products):
        return lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")


def test_scatter_epilogue_replaces_unpacks(monkeypatch):
    for kind, unpack_kind in (("permutation", "indexed_permutation"), ("embedding", "embedding")):
        base = _unpack_program(monkeypatch, kind, FEATURES)
        fused = _unpack_program(monkeypatch, kind, SCATTER)
        scattered = [p for p in _phases(fused, "gemm") if p.get("pack_scatter")]
        assert len(scattered) == 1, kind
        (p,) = scattered
        assert p["engine"].endswith("_scatter_wgmma") and "flat_consumable" not in p
        assert fused.source.count("latch_pack::tflat_claim_scatter<") == 1
        assert '#include "pack_fusion.cuh"' in fused.source
        assert p["pack_scatter"]["dest"] in p["reads"] and p["pack_scatter"]["check"] in p["reads"]
        # The forward unpack phase is gone; the VJP's row movement (cotangent gather) keeps its own phase.
        assert len(_phases(fused, unpack_kind)) < len(_phases(base, unpack_kind)), kind
        assert "latch_pack::" not in base.source


def test_unclaimed_products_keep_the_unpack(monkeypatch):
    # Claim admission refused (rows per segment below the structural minimum): no scatter epilogue, unpack retained.
    for kind in ("permutation", "embedding"):
        base = _unpack_program(monkeypatch, kind, FEATURES, min_rows=10**6)
        program = _unpack_program(monkeypatch, kind, SCATTER, min_rows=10**6)
        assert "latch_pack::" not in program.source
        assert program.source == base.source
