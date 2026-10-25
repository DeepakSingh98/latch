"""Step-05 layout canonicalization contracts (CPU; capture and lowering only, no CUDA)."""
from __future__ import annotations

import copy

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from megakernel_lab.operation_compiler.layout_canonical import canonicalize_layouts, flatten_broadcast_products, model_layouts

BF16 = torch.bfloat16


def _trace(fn, *args):
    return make_fx(fn, tracing_mode="fake")(*args)


def _targets(gm):
    return [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]


def _broadcast_step(b, t, k, h, *, transpose_left=False, dtype=torch.float32):
    def step(x, w):
        xb = x.to(dtype)
        if transpose_left:
            xb = xb.transpose(0, 1)         # batch/rows no longer merge into one stride
        y = torch.bmm(xb, w.to(dtype).expand(xb.shape[0], k, h)).float()
        loss = y.square().mean()
        return (loss, *torch.autograd.grad(loss, [w]))
    return step


def test_broadcast_product_flattened_with_identical_values():
    torch.manual_seed(0)
    x, w = torch.randn(3, 8, 16), torch.randn(16, 24, requires_grad=True)
    gm = _trace(_broadcast_step(3, 8, 16, 24), x, w)
    before = gm(x, w)
    records = flatten_broadcast_products(gm)
    assert [r["batch"] for r in records] == [3]          # forward product; the gradient sum is a separate rule
    after = gm(x, w)
    for a, b in zip(before, after):                      # same dot products; CPU mm/bmm kernels may block apart
        torch.testing.assert_close(b, a, rtol=1e-6, atol=1e-7)
    targets = _targets(gm)
    assert "aten.mm.default" in targets
    assert targets.count("aten.bmm.default") == 1
    assert "aten.sum.dim_IntList" in targets


def test_bf16_broadcast_product_is_left_batched():
    x, w = torch.randn(3, 8, 16), torch.randn(16, 24, requires_grad=True)
    gm = _trace(_broadcast_step(3, 8, 16, 24, dtype=BF16), x, w)
    source = gm.code
    assert canonicalize_layouts(gm) == {}
    assert gm.code == source


def test_rewrite_adds_no_copy_in_storage_model():
    x, w = torch.randn(2, 8, 16), torch.randn(16, 32, requires_grad=True)
    gm = _trace(_broadcast_step(2, 8, 16, 32), x, w)
    flatten_broadcast_products(gm)
    layouts = model_layouts(gm)
    assert all(v is not None for v in layouts.values())  # every view (incl. the new reshapes) is a view
    assert "aten.clone.default" not in _targets(gm)


def test_unmergeable_left_operand_is_left_batched():
    x, w = torch.randn(8, 2, 16), torch.randn(16, 32, requires_grad=True)
    gm = _trace(_broadcast_step(2, 8, 16, 32, transpose_left=True), x, w)
    # The traced left operand is a transposed view of an input, whose batch and row strides do not merge.
    source = gm.code
    assert flatten_broadcast_products(gm) == []
    assert gm.code == source


def test_graph_without_pattern_is_byte_identical():
    def step(x, w):
        y = (x.to(BF16) @ w.to(BF16)).float()
        loss = y.square().mean()
        return (loss, *torch.autograd.grad(loss, [w]))
    x, w = torch.randn(2, 8, 16), torch.randn(16, 32, requires_grad=True)
    gm = _trace(step, x, w)
    source = gm.code
    assert canonicalize_layouts(gm) == {}
    assert gm.code == source


def _families():
    from .test_fixtures import layout_workloads
    return layout_workloads


def _workload_trace(name, **shape):
    """make_fx of the workload's update (policy applied as at capture) at a tiny shape."""
    from dataclasses import replace
    from megakernel_lab.operation_compiler.precision import precision_policy
    workloads = _families()
    case = replace(workloads.case(name), **(shape or dict(batch=2, rows=8, inputs=16, width=32, outputs=8)))
    model = workloads.Model(case)
    (x,), targets = workloads.fixture(case)
    params = [p.detach().clone().requires_grad_() for p in model.parameters()]

    def step(p1, p2, x, *targets):
        with precision_policy("bf16_projections" if case.precision == "policy" else None):
            loss = workloads.loss_fn(case)(workloads.forward(case, {"fc1": p1, "fc2": p2}, x), *targets)
        return (loss, *torch.autograd.grad(loss, [p1, p2]))
    args = (*params, x, *targets)
    return _trace(step, *args), args


def _ops(gm):
    from collections import Counter
    return Counter(str(n.target) for n in gm.graph.nodes if n.op == "call_function")


@pytest.mark.parametrize("canonical,alternate", [
    ("linear-linear-dev-p", "linear-tt-dev-p"), ("linear-linear-dev-p", "linear-matmul_t-dev-p"),
    ("linear-linear-dev-f", "linear-tt-dev-f"), ("linear-linear-dev", "linear-tt-dev"),
    ("proj3d-matmul3d-dev-p", "proj3d-viewchain-dev-p")])
def test_equivalent_forms_reach_the_canonical_operations(canonical, alternate):
    graphs = {}
    for name in (canonical, alternate):
        gm, args = _workload_trace(name)
        before = gm(*args)
        canonicalize_layouts(gm)
        after = gm(*args)
        # Exact rewrites: FP32 graphs agree to rounding; BF16 CPU GEMM kernels may block transposed products
        # differently, so BF16 graphs agree to BF16 rounding.
        tol = dict(rtol=1e-6, atol=1e-9) if name.endswith("-f") else dict(rtol=2e-2, atol=1e-5)
        for a, b in zip(before, after):
            torch.testing.assert_close(b, a, **tol)
        graphs[name] = _ops(gm)
    assert graphs[canonical] == graphs[alternate]


def test_orientation_keeps_free_operand_transposes():
    """A product read only through transposes that feed matrix products is not re-oriented."""
    def step(x, w, v):
        h = (x @ w).t()
        loss = (h @ v).square().mean()
        return (loss, *torch.autograd.grad(loss, [w, v]))
    x, w, v = torch.randn(8, 16), torch.randn(16, 4, requires_grad=True), torch.randn(8, 3, requires_grad=True)
    gm = _trace(step, x, w, v)
    from megakernel_lab.operation_compiler.layout_canonical import orient_transposed_products
    forward = [n for n in gm.graph.nodes if str(n.target) == "aten.mm.default"][0]
    orient_transposed_products(gm)
    assert forward in gm.graph.nodes


def test_bf16_batch_sum_is_not_folded_but_fp32_is():
    from megakernel_lab.operation_compiler.layout_canonical import fold_fp32_batch_sums
    for dtype, folded in ((BF16, 0), (torch.float32, 1)):
        def step(x, w):
            y = torch.bmm(x.to(dtype), w.to(dtype).expand(3, 16, 24)).float()
            loss = y.square().mean()
            return (loss, *torch.autograd.grad(loss, [w]))
        x, w = torch.randn(3, 8, 16), torch.randn(16, 24, requires_grad=True)
        gm = _trace(step, x, w)
        before = gm(x, w)
        assert len(fold_fp32_batch_sums(gm)) == folded
        after = gm(x, w)
        for a, b in zip(before, after):
            torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-6)


def test_disabled_switch_leaves_graph_unchanged():
    import megakernel_lab.operation_compiler.layout_canonical as lc
    gm, _ = _workload_trace("linear-tt-dev-p")
    source = gm.code
    lc.ENABLED = False
    try:
        assert lc.canonicalize_layouts(gm) == {}
    finally:
        lc.ENABLED = True
    assert gm.code == source


@pytest.mark.parametrize("canonical,alternate", [
    ("linear-linear-dev-p", "linear-tt-dev-p"), ("linear-linear-dev-p", "linear-matmul_t-dev-p"),
    ("proj3d-matmul3d-dev-p", "proj3d-viewchain-dev-p"), ("linear-linear-dev-f", "linear-tt-dev-f"),
    ("proj3d-flat-dev-f", "proj3d-bmm_expand-dev-f")])
def test_normal_api_lowering_matches_canonical_structure(canonical, alternate):
    """capture_module + lower(execution='compose'): equal phases, barriers, bodies and arena."""
    from dataclasses import replace
    from megakernel_lab.operation_compiler.lower import lower
    workloads = _families()
    shape = dict(batch=4, rows=64, inputs=128, width=256, outputs=64)
    summary = {}
    for name in (canonical, alternate):
        _, captured, _, _ = workloads.capture(replace(workloads.case(name), **shape))
        manifest = lower(captured.ir, execution="compose").manifest
        bodies = sorted((q.get("matrix_spec") or {}).get("engine") or q.get("engine") or "generic"
                        for q in manifest["phases"] if q.get("kind") == "gemm")
        summary[name] = (manifest["phase_count"], manifest["barrier_count"], bodies, manifest["arena_bytes"])
    assert summary[canonical] == summary[alternate]


@pytest.mark.parametrize("precision", ["f"])
def test_normal_api_broadcast_form_reaches_flat_bodies(precision):
    """Through capture_module + lower(compose): the FP32 batch-broadcast form lowers every product (forward,
    input gradient and the batch-summed weight gradient) as the canonical flat form's products."""
    from dataclasses import replace
    from megakernel_lab.operation_compiler.lower import lower
    workloads = _families()
    tail = f"-{precision}" if precision else ""
    shape = dict(batch=4, rows=64, inputs=128, width=256, outputs=64)
    flat = replace(workloads.case(f"proj3d-flat-dev{tail}"), **shape)
    alt = replace(workloads.case(f"proj3d-bmm_expand-dev{tail}"), **shape)
    _, cf, _, _ = workloads.capture(flat)
    _, ca, _, _ = workloads.capture(alt)
    assert "layout_canonical" not in cf.ir.metadata
    assert len(ca.ir.metadata["layout_canonical"]["flattened_broadcast_products"]) == 3
    assert len(ca.ir.metadata["layout_canonical"]["folded_fp32_batch_sums"]) == 2
    pf, pa = lower(cf.ir, execution="compose"), lower(ca.ir, execution="compose")
    products = lambda p: sorted((q.get("matrix_spec") or {}).get("engine") or q.get("engine") or "generic"
                                for q in p.manifest["phases"] if q.get("kind") == "gemm")
    # FP32: every product, including both weight-gradient batch sums, is the canonical flat product.
    assert products(pf) == products(pa)
    assert pf.manifest["phase_count"] == pa.manifest["phase_count"]
    assert not [q for q in pa.manifest["phases"] if "bmm" in str(q.get("targets"))]


def test_capture_metadata_records_rewrites_and_values_match_reference():
    from dataclasses import replace
    workloads = _families()
    case = replace(workloads.case("proj3d-bmm_expand-dev-f"), batch=2, rows=16, inputs=32, width=64, outputs=16)
    model, captured, inputs, targets = workloads.capture(case)
    record = captured.ir.metadata["layout_canonical"]["flattened_broadcast_products"]
    assert {r["rule"] for r in record} == {"batch-broadcast right operand: one flat product over batch*rows"}
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    leaves = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    loss = workloads.reference_loss(case, leaves, inputs, targets)
    grads = torch.autograd.grad(loss, list(leaves.values()))
    # The captured (rewritten) graph is the native program's specification; executing it eagerly on CPU gives
    # the as-written conventional loss and gradients (CPU bmm and mm kernels may round differently, so the
    # bitwise statement is the FX-level test above).
    outputs = captured.graph_module(*captured.example_args)
    unpacked = captured.unpack(outputs)
    torch.testing.assert_close(unpacked["loss"].reshape(()), loss.reshape(()), rtol=1e-3, atol=1e-6)
    for name, want in zip(leaves, grads):
        torch.testing.assert_close(unpacked["grads"][name], want, rtol=2e-2, atol=1e-4)


def _stacked_expert_step(dtype):
    """HF-style per-expert [out, in] leaves stacked, cast and transposed to the [E, in, out] product operand."""
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm

    def step(x, offsets, *weights):
        w = torch.stack(weights).to(dtype).transpose(1, 2)
        y = ragged_mm(x.to(dtype), w, offsets).float()
        loss = y.square().sum()
        return (loss, *torch.autograd.grad(loss, list(weights)))
    return step


@pytest.mark.parametrize("dtype", [torch.float32, BF16])
def test_transposed_expert_weight_gradient_is_oriented(dtype):
    from megakernel_lab.operation_compiler.layout_canonical import orient_transposed_segment_products
    torch.manual_seed(0)
    groups, rows, inner, outer = 3, 10, 8, 12
    x = torch.randn(rows, inner)
    offsets = torch.tensor([0, 4, 4, rows], dtype=torch.int64)          # one empty segment
    weights = [torch.randn(outer, inner, requires_grad=True) for _ in range(groups)]
    from megakernel_lab.operation_compiler.capture import capture_decompositions
    # Capture's decompositions: the stack VJP reaches per-expert selects, as in a captured update.
    gm = make_fx(_stacked_expert_step(dtype), decomposition_table=capture_decompositions(),
                 tracing_mode="fake")(x, offsets, *weights)
    before = gm(x, offsets, *weights)
    applied = canonicalize_layouts(gm)
    records = applied.get("oriented_transposed_segment_products", [])
    assert len(records) == 1 and records[0]["through_cast"] == (dtype != torch.float32)
    after = gm(x, offsets, *weights)
    for a, b in zip(before, after):                      # same products and per-segment contraction order
        assert torch.equal(a, b)
    layout = model_layouts(gm)
    outputs = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
    for grad in outputs[1:]:                             # each expert gradient is now a contiguous slice
        shape, strides = layout[grad]
        assert tuple(shape) == (outer, inner) and tuple(strides) == (inner, 1)
    assert orient_transposed_segment_products(gm) == []  # idempotent: nothing reads it transposed any more
    widened = applied.get("widened_segment_products", [])
    targets = _targets(gm)
    if dtype == BF16:
        assert widened == []                             # widening is an opt-in rule (measured slower on TinyMoE)
        # The stacked FP32 weights' BF16 product operand is cast while concatenating (one pass).
        assert len(applied.get("fused_cat_casts", [])) == 1
        assert "latch_ops.cat_cast.default" in targets and "aten.cat.default" not in targets
    else:
        assert widened == [] and "latch_ops.ragged_outer_widened.default" not in targets


def test_bf16_product_with_another_reader_is_not_widened():
    from megakernel_lab.operation_compiler import ragged_experts  # noqa: F401  (registers the ops)
    from megakernel_lab.operation_compiler.layout_canonical import widen_segment_products

    def step(x, dy, offsets):
        product = torch.ops.latch_ops.ragged_outer(x, dy, offsets)
        return product, product.float()
    x, dy = torch.randn(6, 4).to(BF16), torch.randn(6, 5).to(BF16)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
    gm = _trace(step, x, dy, offsets)
    source = gm.code
    assert widen_segment_products(gm) == []              # the BF16 result is also returned: keep it
    assert gm.code == source


def test_untransposed_segment_product_is_left_unchanged():
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
    x, offsets = torch.randn(6, 4), torch.tensor([0, 2, 6], dtype=torch.int64)
    w = torch.randn(2, 4, 5, requires_grad=True)

    def step(x, offsets, w):
        loss = ragged_mm(x, w, offsets).square().sum()
        return (loss, *torch.autograd.grad(loss, [w]))
    gm = _trace(step, x, offsets, w)
    source = gm.code
    assert "oriented_transposed_segment_products" not in canonicalize_layouts(gm)
    assert gm.code == source


def test_cat_cast_requires_a_single_cast_reader():
    from megakernel_lab.operation_compiler.layout_canonical import fuse_cat_casts

    def step(a, b):
        joined = torch.cat([a, b], 0)
        return joined.to(BF16), joined.sum()
    a, b = torch.randn(4, 8), torch.randn(6, 8)
    gm = _trace(step, a, b)
    source = gm.code
    assert fuse_cat_casts(gm) == []                      # the FP32 concatenation has another reader
    assert gm.code == source

    def fused(a, b):
        return (torch.cat([a, b], 0).reshape(5, 2, 8).to(BF16),)
    gm = _trace(fused, a, b)
    before = gm(a, b)
    assert len(fuse_cat_casts(gm)) == 1
    assert torch.equal(gm(a, b)[0], before[0])


def test_opt_in_widening_folds_the_fp32_cast(monkeypatch):
    from megakernel_lab.operation_compiler import layout_canonical
    from megakernel_lab.operation_compiler.capture import capture_decompositions
    monkeypatch.setattr(layout_canonical, "WIDEN_SEGMENT_PRODUCTS", True)
    torch.manual_seed(0)
    x, offsets = torch.randn(10, 8), torch.tensor([0, 4, 4, 10], dtype=torch.int64)
    weights = [torch.randn(12, 8, requires_grad=True) for _ in range(3)]
    gm = make_fx(_stacked_expert_step(BF16), decomposition_table=capture_decompositions(),
                 tracing_mode="fake")(x, offsets, *weights)
    before = gm(x, offsets, *weights)
    applied = canonicalize_layouts(gm)
    assert len(applied.get("widened_segment_products", [])) == 1
    targets = _targets(gm)
    assert "latch_ops.ragged_outer_widened.default" in targets and "latch_ops.ragged_outer.default" not in targets
    for a, b in zip(before, gm(x, offsets, *weights)):
        assert torch.equal(a, b)
