"""Independent RMSNorm forward/backward and ordinary capture checks (cloud CPU)."""
import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import norms


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalized_shape", [(4,), (3, 4)])
@pytest.mark.parametrize("weighted", [False, True])
def test_analytic_rms_matches_independent_autograd(dtype, normalized_shape, weighted):
    gen = torch.Generator().manual_seed(20104)
    x = torch.randn(2, 3, 4, generator=gen, dtype=dtype).requires_grad_()
    weight = torch.randn(normalized_shape, generator=gen, dtype=dtype).requires_grad_() if weighted else None
    grad = torch.randn(x.shape, generator=gen, dtype=dtype)
    actual = norms.rms_norm(x, normalized_shape, weight, eps=1e-4)
    actual_grad = torch.autograd.grad(actual, (x, weight) if weighted else (x,), grad)
    refx = x.detach().clone().requires_grad_()
    refw = weight.detach().clone().requires_grad_() if weighted else None
    expected = F.rms_norm(refx, normalized_shape, refw, eps=1e-4)
    expected_grad = torch.autograd.grad(expected, (refx, refw) if weighted else (refx,), grad)
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("normalized_shape,weighted", [((4,), True), ((3, 4), True), ((4,), False)])
def test_public_rms_capture_matches_fp64_complete_update(normalized_shape, weighted):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.randn(3, 4) * .2 + 1)
            self.weight = torch.nn.Parameter(torch.randn(normalized_shape) * .2 + 1) if weighted else None

        def forward(self, x):
            return F.rms_norm(x * self.gain, normalized_shape, self.weight, eps=1e-4)

    torch.manual_seed(20105)
    model = Model()
    x, target = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    captured = capture_module(model, F.mse_loss, (x,), (target,), optimizer=SGDConfig(lr=.03))
    actual = captured.unpack(captured.graph_module(*captured.example_args))
    params = {name: value.detach().double().requires_grad_() for name, value in model.named_parameters()}
    expected = F.mse_loss(torch.func.functional_call(model, params, (x.double(),)), target.double())
    grads = torch.autograd.grad(expected, tuple(params.values()))
    torch.testing.assert_close(actual["loss"].double(), expected, rtol=2e-6, atol=2e-7)
    for (name, value), grad in zip(params.items(), grads):
        torch.testing.assert_close(actual["grads"][name].double(), grad, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(actual["params"][name].double(), value - .03 * grad, rtol=2e-6, atol=2e-7)
    program = lower(captured.ir, execution="compose")
    assert program.manifest["execution_selection"]["selected"] in ("common_static", "legacy_static")
