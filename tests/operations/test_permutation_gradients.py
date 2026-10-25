"""LATCH-DISPATCH-GRAD: graph-proved permutation indices and their direct embedding-gradient lowering.

CPU-only structure tests (capture + lowering, no CUDA). Numerical execution is checked on an H100 by
research/latch_dispatch_grad/opcheck.py against the general engine and an FP64 reference.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import permutation_gradients as pg
from megakernel_lab.operation_compiler.indexed_reduction import policy
from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, route_order


class Dispatch(torch.nn.Module):
    """Packed token-choice dispatch/combine (the TinyMoE adapter's form) with variants of the dispatch index."""

    def __init__(self, k=2, experts=4, width=8, hidden=12, divisor=None, modulo=False, dtype=torch.float32,
                 tokens_from_input=False):
        super().__init__()
        torch.manual_seed(1740)
        self.k, self.experts, self.divisor, self.modulo, self.dtype = k, experts, divisor, modulo, dtype
        self.tokens_from_input = tokens_from_input
        self.weight = torch.nn.Parameter(torch.randn(experts, width, hidden) * 0.2)
        self.down = torch.nn.Parameter(torch.randn(experts, hidden, width) * 0.2)
        self.table = torch.nn.Parameter(torch.randn(11, width) * 0.2)
        self.gain = torch.nn.Parameter(torch.ones(width))   # x needs a gradient: dispatch VJP exists

    def forward(self, x, ids, tokens):
        x = x * self.gain
        if self.tokens_from_input:
            x = x + F.embedding(tokens, self.table)
        order, offsets, inverse = route_order(ids, self.experts)
        divisor = self.k if self.divisor is None else self.divisor
        source = order % ids.shape[0] if self.modulo else order // divisor
        packed = F.embedding(source, x)
        up = ragged_mm(packed.to(self.dtype), self.weight.to(self.dtype), offsets)
        down = ragged_mm(F.silu(up), self.down.to(self.dtype), offsets)
        choices = F.embedding(inverse, down).reshape(x.shape[0], self.k, x.shape[1])
        return choices.float().sum(1)


def _capture(model, tokens=5, width=8, k=2, ids=None):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.optim import SGDConfig
    if ids is None:
        ids = torch.tensor([[(t + s) % model.experts for s in range(k)] for t in range(tokens)])
    args = (torch.randn(tokens, width), ids, torch.randint(0, 11, (tokens,)))
    return capture_module(model, lambda y: y.square().mean(), args, optimizer=SGDConfig())


def _lower(captured, engine="segmented_v1"):
    from megakernel_lab.operation_compiler.lower import lower
    with policy(engine):
        return lower(captured.ir, precision_policy="practical", execution="compose")


def _facts(captured):
    ir = captured.ir
    return pg.analyze(ir.operations, ir.values, set(ir.values))


def _rows(program):
    return {row["output"]: row for row in program.manifest["ordered_index_gradients"]}


def test_facts_prove_order_inverse_and_repeat():
    captured = _capture(Dispatch())
    facts = _facts(captured)
    kinds = sorted((f["kind"], f.get("repeat", 1), f.get("rows", f.get("extent"))) for f in facts.values())
    # order and inverse are permutations of the 10 choices; order // 2 repeats each of 5 tokens twice.
    assert ("perm", 1, 10) in kinds and ("repeat", 2, 5) in kinds
    perms = {name: f for name, f in facts.items() if f["kind"] == "perm"}
    assert len(perms) == 2 and {f["inverse"] for f in perms.values()} == set(perms)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_dispatch_and_combine_use_direct_gradients(k):
    captured = _capture(Dispatch(k=k, experts=6), tokens=7, k=k)
    program = _lower(captured)
    rows = _rows(program)
    engines = sorted(row.get("engine", row.get("indexed_reduction_policy")) for row in rows.values())
    # Two direct indexed gradients (combine gather, dispatch fixed-k sum); no sorting phases for them.
    direct = [row for row in rows.values() if row.get("decision", "").startswith("permutation_")]
    assert len(direct) == 2, engines
    assert {row["index_proof"]["occurrences_per_row"] for row in direct} == {1, k}
    for row in direct:
        assert row["arithmetic_policy"] == "ordered_fp32_v1"
        assert row["indexed_reduction_policy"] == "segmented_v1"   # engine_of() unchanged
    kinds = [p["kind"] for p in program.manifest["phases"]]
    assert "index_radix_histogram" not in kinds and "index_segment_partials" not in kinds
    assert "latch_pgrad::" in program.source and "__trap()" in program.source


def test_arbitrary_indices_keep_general_engine():
    captured = _capture(Dispatch(tokens_from_input=True))
    rows = _rows(_lower(captured))
    general = [row for row in rows.values() if not row.get("decision", "").startswith("permutation_")]
    direct = [row for row in rows.values() if row.get("decision", "").startswith("permutation_")]
    assert len(direct) == 2
    # The token-table gradient (arbitrary, repeated input ids) stays on the segmented engine.
    assert len(general) == 1 and general[0]["engine"] == "segmented_v1"


@pytest.mark.parametrize("variant", [dict(divisor=3), dict(modulo=True)])
def test_unproved_index_arithmetic_falls_back(variant):
    # order // 3 over 10 choices is not a uniform repeat; order % T is not a recognized arrangement.
    captured = _capture(Dispatch(**variant))
    rows = _rows(_lower(captured))
    direct = [row for row in rows.values() if row.get("decision", "").startswith("permutation_")]
    assert len(direct) == 1 and direct[0]["index_proof"]["occurrences_per_row"] == 1   # combine only


def test_disable_switch_and_frozen_policies_restore_general_lowering(monkeypatch):
    captured = _capture(Dispatch())
    candidate = _lower(captured)
    monkeypatch.setattr(pg, "ENABLED", False)
    general = _lower(captured)
    assert not any(r.get("decision", "").startswith("permutation_") for r in _rows(general).values())
    assert general.source != candidate.source
    monkeypatch.setattr(pg, "ENABLED", True)
    baseline = _lower(captured, "baseline")
    assert not any(r.get("decision", "").startswith("permutation_") for r in _rows(baseline).values())


def test_bf16_combine_rounds_once_and_vectorizes():
    captured = _capture(Dispatch(dtype=torch.bfloat16, width=16, hidden=16), width=16)
    program = _lower(captured)
    direct = [r for r in _rows(program).values() if r.get("decision", "").startswith("permutation_")]
    combine = [r for r in direct if r["index_proof"]["occurrences_per_row"] == 1]
    assert len(combine) == 1 and combine[0]["contribution"] == "widen" and combine[0]["vector_columns"]
    assert "latch_pgrad::round4" in program.source


def test_facts_reject_noncontiguous_or_wrong_extent_copies():
    values = {
        "sel": dict(shape=[3, 2], strides=[2, 1], dtype="int64"),
        "o": dict(shape=[6], strides=[1], dtype="int64"), "off": dict(shape=[5], strides=[1], dtype="int64"),
        "inv": dict(shape=[6], strides=[1], dtype="int64"),
        "v": dict(shape=[2, 3], strides=[3, 1], dtype="int64"),          # 2-D view: not admitted as a vector
        "s": dict(shape=[3], strides=[2], dtype="int64"),                 # strided slice
        "f": dict(shape=[6], strides=[1], dtype="int64"),
        "g": dict(shape=[6], strides=[1], dtype="int64"),
    }
    ops = [
        dict(target="latch_ops.ragged_order.default", args=[{"ref": "sel"}, 4], kwargs={}, outputs=["o", "off", "inv"]),
        dict(target="aten.view.default", args=[{"ref": "o"}, [2, 3]], kwargs={}, outputs=["v"]),
        dict(target="aten.slice.Tensor", args=[{"ref": "o"}, 0, 0, 6, 2], kwargs={}, outputs=["s"]),
        dict(target="aten.floor_divide.default", args=[{"ref": "o"}, 2], kwargs={}, outputs=["f"]),
        dict(target="aten.floor_divide.default", args=[{"ref": "o"}, 4], kwargs={}, outputs=["g"]),
    ]
    facts = pg.analyze(ops, values, set(values))
    assert set(facts) == {"o", "inv", "f"}
    assert facts["f"] == dict(kind="repeat", rows=3, repeat=2, inverse="inv", source=pg.ORDER)
