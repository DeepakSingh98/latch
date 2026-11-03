"""Observed segmented products (``ragged_mm_observed``): eager semantics and executed-work counter lowering (CPU)."""
from __future__ import annotations

import pytest
import torch


def test_observed_projection_has_same_primal_and_vjp_without_fake_native_counts():
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, ragged_mm_observed
    torch.manual_seed(261028)
    x = torch.randn(7, 3, dtype=torch.float64).requires_grad_()
    w = torch.randn(4, 3, 5, dtype=torch.float64).requires_grad_()
    offsets = torch.tensor([0, 0, 1, 7, 7])
    result, work = ragged_mm_observed(x, w, offsets)
    expected = ragged_mm(x, w, offsets)
    assert work.shape == (4, 4) and work.dtype == torch.int64
    assert torch.all(work == -1)  # eager must never pretend to report executed native work
    dy = torch.randn_like(result)
    actual_grad = torch.autograd.grad(result, (x, w), dy, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (x, w), dy)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    for actual, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,products", [(torch.float32, None), (torch.bfloat16, "flat"),
                                          (torch.bfloat16, "flat+tma+rotate+claim+sepi")])
def test_observed_projection_lowering_counts_executed_native_tasks(dtype, products):
    from megakernel_lab.operation_compiler import capture_module
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm_observed
    from megakernel_lab.operation_compiler.ragged_experts_lower import products_scope
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 64, 32))
            self.shift = torch.nn.Parameter(torch.zeros(64))

        def forward(self, x, offsets):
            y, work = ragged_mm_observed((x + self.shift).to(dtype), self.weight.to(dtype), offsets)
            return {"value": y.float(), "work": work}

    captured = capture_module(Projection(), lambda y: y["value"].square().mean(),
        (torch.randn(9, 64), torch.tensor([0, 0, 3, 9, 9])), optimizer=SGDConfig())
    targets = [str(op.target) for op in captured.ir.operations]
    assert targets.count("latch_ops.ragged_mm_observed.default") == 2
    assert targets.count("latch_ops.ragged_outer_observed.default") == 1
    with products_scope(products):
        program = lower(captured.ir, precision_policy="practical", execution="compose", target_arch="sm_90")
    phases = [phase for phase in program.manifest["phases"] if phase.get("ragged_work_counters")]
    assert len(phases) == 3
    assert all(phase["packed_rows"] == 9 for phase in phases)
    assert sorted(phase["work_product"] for phase in phases) == ["outer", "right", "right"]
    assert all(not phase.get("flat_claim") and not phase.get("flat_paired") for phase in phases)
    assert all(phase["work_counter_columns"] == ["valid_rows", "output_tiles", "contraction_tiles", "output_elements"]
               for phase in phases)
    assert all(phase["ragged_work_counters"] in phase["writes"] for phase in phases)
