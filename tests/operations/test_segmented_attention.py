"""Packed segmented attention: CPU reference, capture, native lowering and the opt-in width-32 rules.

CPU only: pytest -q tests/operations/test_segmented_attention.py
"""
from __future__ import annotations

import re

import pytest
import torch

from megakernel_lab.operation_compiler import segmented_attention as sa
from megakernel_lab.operation_compiler import segmented_attention_rules as rules


def reference(q, k, v, offsets, scale):
    """Independent per-query equation; packed positions preserve segment order."""
    out = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        for row in range(lo, hi):
            scores = torch.einsum("hd,khd->hk", q[row], k[lo:row+1]) * scale
            out.append(torch.einsum("hk,khd->hd", scores.softmax(-1), v[lo:row+1]))
    return torch.stack(out)


@pytest.mark.parametrize("width", [1, 30, 32, 64])
@pytest.mark.parametrize("boundaries", [[0, 0, 1, 3, 9], [0, 9, 9, 9, 9], [0, 2, 5, 6, 9]])
def test_cpu_reference_empty_singleton_uneven_changing_segments(width, boundaries):
    torch.manual_seed(711)
    q, k, v = [torch.randn(9, 2, width).bfloat16().requires_grad_() for _ in range(3)]
    cotangent = torch.randn_like(q)
    actual, work = sa.sdpa(q, k, v, torch.tensor(boundaries), max_seqlen=9, scale=1/30, return_work=True)
    got = torch.autograd.grad(actual, (q, k, v), cotangent)
    qr, kr, vr = [x.detach().double().requires_grad_() for x in (q, k, v)]
    expected = reference(qr, kr, vr, boundaries, 1/30)
    want = torch.autograd.grad(expected, (qr, kr, vr), cotangent.double())
    torch.testing.assert_close(actual.float(), expected.float(), atol=.015, rtol=.008)
    for a, b in zip(got, want):
        torch.testing.assert_close(a.float(), b.float(), atol=.015, rtol=.008)
    assert work.shape == (4, 2, 3) and (work == -1).all()  # eager counters are explicitly unmeasured


def test_cpu_offsets_are_validated():
    x = torch.zeros(4, 1, 30, dtype=torch.bfloat16)
    for boundaries in ([1, 4], [0, 3], [0, 3, 2, 4], [0, 4]):
        with pytest.raises(ValueError, match="offsets"):
            sa.sdpa(x, x, x, torch.tensor(boundaries), max_seqlen=3)


def test_raw_forward_statistics_are_log2_and_packed():
    torch.manual_seed(713)
    q, k, v = [torch.randn(7, 2, 64).bfloat16() for _ in range(3)]
    _out, lse, _work = sa.forward(q, k, v, torch.tensor([0, 0, 1, 7]), 7, 1/30)
    expected = torch.empty(2, 7, dtype=torch.float64)
    for lo, hi in ((0, 1), (1, 7)):
        for row in range(lo, hi):
            scores = torch.einsum("hd,khd->hk", q[row].double(), k[lo:row+1].double())/30
            expected[:, row] = scores.logsumexp(-1)/torch.log(torch.tensor(2., dtype=torch.float64))
    torch.testing.assert_close(lse.double(), expected, rtol=2e-6, atol=2e-6)


def fixture(rows=259, width=30, heads=2, groups=6):
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    torch.manual_seed(712)
    parameters = {name: torch.randn(rows, heads, width).bfloat16().float() for name in ("q", "k", "v")}
    cotangent = torch.randn(rows, heads, width).bfloat16()
    offsets = torch.zeros(groups+1, dtype=torch.int64)
    offsets[-1] = rows
    state = {"output": torch.zeros(rows, heads, width, dtype=torch.bfloat16),
             "work": torch.zeros(groups, heads, 3, dtype=torch.int64)}
    def objective(params, state, cotangent, offsets):
        output, work = sa.sdpa(*(params[n].bfloat16() for n in ("q", "k", "v")),
                              offsets, max_seqlen=4096, scale=1/30, return_work=True)
        return (output.float() * cotangent.float()).sum(), {"output": output, "work": work}
    rule = tensor_update(lambda p, g, s, hp: ({n: x-0.0*g[n] for n, x in p.items()}, {}),
                         name="segmented_attention_probe")
    return capture_training(objective, parameters, None, inputs=(cotangent, offsets), state=state, config=rule)


def lowered_fixture(width=30, rows=259, segmented_attention=None):
    from megakernel_lab.operation_compiler.lower import lower
    captured = fixture(width=width, rows=rows)
    with rules.scope(segmented_attention):
        program = lower(captured.ir, target_arch="sm_90", execution="compose")
    return captured, program


@pytest.mark.parametrize("width", [30, 32, 64])
def test_capture_native_lowering_and_retained_work(width):
    captured, program = lowered_fixture(width)
    targets = [op.target for op in captured.ir.operations]
    assert targets.count(sa.FORWARD) == targets.count(sa.BACKWARD) == 1
    assert not any("softmax" in t or "bmm" in t for t in targets)
    assert all(f"latch_segmented::run_{phase}" in program.source for phase in ("forward", "dq", "kv"))
    native = 64 if width > 32 else 32
    assert all(f"latch_segmented::run_{phase}<{native}," in program.source for phase in ("forward", "dq", "kv"))
    bodies = {p.get("native_body") for p in program.manifest["phases"] if p.get("native_body")}
    from megakernel_lab.operation_compiler import segmented_attention_lower as sal
    assert bodies == {sal.native_body(native, sal.DEFAULT_W32_VARIANT if native == 32 else 0)}
    # Width <= 32 enters the native body without the former zero-pad to 64 and output slice.
    if width == 32:
        assert not any("cat" in t or "slice" in t for t in targets)
    assert "attention_segmented.cuh" in program.manifest["header_hashes"]
    counters = program.manifest["segmented_attention_work"]
    assert [row["half"] for row in counters] == ["forward", "backward"]
    end = len(program.manifest["barrier_intervals"])
    for row in counters:
        layout = program.manifest["layouts"][row["name"]]
        root = layout["root"]
        assert program.manifest["lifetimes"][root][-1] == end
        assert row["retained_through_commit"]
        assert tuple(layout["shape"]) == (6, 2, 3 if row["half"] == "forward" else 6)
    lse = next(op for op in captured.ir.operations if op.target == sa.FORWARD).outputs[1]
    assert tuple(captured.ir.values[lse].shape) == (2, 259)
    assert all(r["row_capacity_padding"] == 0 for r in program.manifest["attention_groups"]
               if r.get("decision") == "packed_segmented_native")
    # Counter initialization and both read/modify/write phases must retain their
    # effects after source inlining and topological reordering (widths32/64 used
    # to form a false dq -> future-kv counter dependency and fail lowering).
    phases = program.manifest["phases"]
    stages = {p: level for level, members in enumerate(program.manifest["barrier_intervals"]) for p in members}
    dq = next(i for i, phase in enumerate(phases) if phase["kind"] == "segmented_attention_dq")
    kv = next(i for i, phase in enumerate(phases) if phase["kind"] == "segmented_attention_kv")
    work = counters[1]["name"]
    prepare = next(i for i, phase in enumerate(phases)
                   if phase["kind"] == "segmented_attention_prepare" and work in phase["writes"])
    assert stages[prepare] < stages[dq] < stages[kv]
    assert work in phases[dq]["reads"] and work in phases[kv]["reads"]
    assert work in phases[dq]["writes"] and work in phases[kv]["writes"]
    assert any(name.endswith("__segment_delta") for name in set(phases[dq]["writes"]) & set(phases[kv]["reads"]))


def test_source_tail_predication_and_dynamic_queue():
    from megakernel_lab.operation_compiler.lower import _header_path
    from megakernel_lab.operation_compiler.step17_intervals import wgmma_inline, SEGMENTED_ATTENTION
    source = _header_path("attention_segmented.cuh").read_text()
    assert "prefix[g+1]=prefix[g]+((hi-lo+63)/64)*heads" in source
    assert "if (idx>=total) break" in source
    assert "cp.async.cg.shared.global [%0], [%1], 16, %2" in source
    assert "if (r_lo < a.t) store_bf16x2" in source
    assert "if (k_hi < a.t) store_bf16x2" in source
    assert "add_count(a.work + 2, 1)" in source and "add_count(a.work + 5, 1)" in source
    assert "softmax_tile6<" not in source and "ds_rows6<" not in source
    code = re.sub(r"//[^\n]*", "", source)
    assert " = ex2(" not in code  # another operation's FTZ flag cannot change this contract
    assert SEGMENTED_ATTENTION in wgmma_inline("latch_segmented::run_forward(A, offsets, prefix, smem);")


def matrix_reference(q, k, v, offsets, scale):
    """CPU-only FP64 reference for the 4096 stress fixture; no native helper use."""
    output = []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        if lo == hi:
            continue
        qs, ks, vs = (x[lo:hi].transpose(0, 1) for x in (q, k, v))
        scores = qs @ ks.transpose(-1, -2) * scale
        mask = torch.ones(hi-lo, hi-lo, dtype=torch.bool, device=q.device).tril()
        output.append((scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vs).transpose(0, 1))
    return torch.cat(output)


def test_matrix_reference_matches_query_equations_and_vjp():
    torch.manual_seed(714)
    leaves = [torch.randn(9, 2, 30, dtype=torch.float64).requires_grad_() for _ in range(3)]
    offsets = [0, 0, 1, 4, 9]
    a = reference(*leaves, offsets, 1/30)
    b = matrix_reference(*leaves, offsets, 1/30)
    cotangent = torch.randn_like(a)
    ga = torch.autograd.grad(a, leaves, cotangent, retain_graph=True)
    gb = torch.autograd.grad(b, leaves, cotangent)
    torch.testing.assert_close(a, b, rtol=2e-14, atol=2e-14)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=2e-14, atol=2e-14)


def test_native_width_selection():
    assert [sa.native_width_for(w) for w in (1, 30, 32, 33, 64)] == [32, 32, 32, 64, 64]
    assert sa.native_width_for(30, 64) == 64
    with pytest.raises(ValueError):
        sa.native_width_for(40, 32)


def test_folded_forward_and_fused_backward_lowering():
    from megakernel_lab.operation_compiler import segmented_attention_lower as sal
    _captured, program = lowered_fixture(width=32, segmented_attention="folded_forward+fused_backward")
    source = program.source
    assert "latch_segmented::run_forward<32,152>(A," in source
    assert "latch_segmented::run_kv<32,168>(A," in source
    assert "latch_segmented::fused_bprep(" in source and "latch_segmented::fused_dqcvt_hr(" in source
    assert "latch_segmented::run_dq" not in source and "latch_segmented::prepare2(" in source  # forward prepare only
    phases = [p for p in program.manifest["phases"] if p["kind"].startswith("segmented_attention_")]
    by_kind = {}
    for phase in phases:
        by_kind.setdefault(phase["kind"], []).append(phase)
    forward, = by_kind["segmented_attention_forward"]
    kv, = by_kind["segmented_attention_kv"]
    dq, = by_kind["segmented_attention_dq"]
    assert forward["shared_bytes"] == kv["shared_bytes"] == 1024 + 2 * 88 * 1024 == 181248
    assert forward["native_body"] == kv["native_body"] == sal.native_body(32, 8)
    assert forward["arithmetic_contract"].endswith("+ftz-ex2+folded-fwd")
    assert kv["arithmetic_contract"].endswith("+ftz-ex2+fused-bwd")
    accumulators = [n for n in kv["writes"] if n.endswith("__fused_dqacc")]
    assert len(accumulators) == 1 and accumulators[0] in dq["reads"]
    assert tuple(program.manifest["layouts"][accumulators[0]]["shape"]) == (259, 2, 32)
    assert len([p for p in by_kind["segmented_attention_prepare"] if accumulators[0] in p["writes"]]) == 1
    # Each rule alone changes only its half.
    _captured, forward_only = lowered_fixture(width=32, segmented_attention="folded_forward")
    assert "run_forward<32,152>" in forward_only.source and "latch_segmented::run_dq<32,6>" in forward_only.source
    _captured, backward_only = lowered_fixture(width=32, segmented_attention="fused_backward")
    assert "run_forward<32,6>" in backward_only.source and "run_kv<32,168>" in backward_only.source


def test_default_lowering_ignores_bitwise_only_rules():
    _captured, default = lowered_fixture(width=32)
    _captured, rotary = lowered_fixture(width=32, segmented_attention="rotary")
    assert rotary.source == default.source
    assert "run_forward<32,6>" in default.source and "latch_segmented::fused_" not in default.source


@pytest.mark.parametrize("value", ["folded_forward", "fused_backward", "claim+fused_backward"])
def test_folded_and_fused_forms_require_practical_precision(tmp_path, value):
    from megakernel_lab.operation_compiler.api import prepare_module
    captured = fixture(width=32)
    with pytest.raises(ValueError, match="requires precision_policy='practical'"):
        prepare_module(captured, tmp_path / "artifact", cache_dir=tmp_path / "cache", segmented_attention=value)
