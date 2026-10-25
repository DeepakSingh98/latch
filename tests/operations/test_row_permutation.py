import pytest
import torch

from megakernel_lab.operation_compiler.row_permutation import permutation_gather


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
def test_bijection_primal_and_vjp_preserve_bits(dtype, strided):
    torch.manual_seed(1793)
    x = torch.randn(19, 14, dtype=dtype)
    x[0, 0] = -0.
    x = (x[:, ::2] if strided else x).requires_grad_()
    order = torch.randperm(19)
    inverse = torch.empty_like(order); inverse[order] = torch.arange(19)
    y = permutation_gather(x, order, inverse)
    grad = torch.randn_like(y); grad[3, 0] = -0.
    dx, = torch.autograd.grad(y, x, grad)
    bits = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16}[dtype]
    assert torch.equal(y.contiguous().view(bits), x[order].contiguous().view(bits))
    assert torch.equal(dx.contiguous().view(bits), grad[inverse].contiguous().view(bits))


@pytest.mark.parametrize("order,inverse", [([0, 0, 2], [0, 1, 2]),
    ([2, 0, 1], [0, 1, 2]), ([-1, 0, 1], [1, 2, 0]), ([3, 0, 1], [1, 2, 0])])
def test_invalid_bijection_is_rejected(order, inverse):
    with pytest.raises(ValueError, match="bijective"):
        permutation_gather(torch.ones(3, 2), torch.tensor(order), torch.tensor(inverse))


def test_common_capture_uses_inverse_copy_for_vjp():
    from megakernel_lab.operation_compiler import capture_module, SGDConfig
    from megakernel_lab.operation_compiler.lower import lower
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.arange(21, dtype=torch.float32).reshape(3, 7))
        def forward(self, order, inverse):
            return permutation_gather(self.x.bfloat16(), order, inverse).float()
    captured = capture_module(Model(), lambda y: y.sum(),
        (torch.tensor([2, 0, 1]), torch.tensor([1, 2, 0])), optimizer=SGDConfig(lr=.01))
    program = lower(captured.ir, execution="compose", precision_policy="practical")
    copies = [p for p in program.manifest["phases"] if p.get("engine") == "validated_bijective_row_copy"]
    assert len(copies) == 2
    assert "embedding_dense_backward" not in program.source
