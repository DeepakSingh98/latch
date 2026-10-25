"""Declared semantics of operation_compiler.split_precision (LATCH-TRAIN-PRECISION), CPU, tiny shapes."""
import torch

from megakernel_lab.operation_compiler.split_precision import split_bf16_linear, split_weight_terms

BF16 = torch.bfloat16


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(6, 32, generator=g, dtype=torch.float32)
    w = torch.randn(48, 32, generator=g, dtype=torch.float32)
    dy = torch.randn(6, 48, generator=g, dtype=torch.float32).to(BF16)
    return x, w, dy


def test_one_term_is_released_linear():
    x, w, _ = _data()
    assert torch.equal(split_bf16_linear(x, w, weight_terms=1), torch.nn.functional.linear(x.to(BF16), w.to(BF16)))


def test_two_terms_represent_weight_to_2e_minus_16():
    _, w, _ = _data(1)
    hi, lo = (p.to(BF16).double() for p in split_weight_terms(w, 2))
    assert torch.equal(hi, w.to(BF16).double())
    assert float(((hi + lo) - w.double()).abs().max() / w.double().abs().max()) < 2.0 ** -15


def test_forward_matches_fp64_product_of_bf16_input_and_fp32_weight():
    x, w, _ = _data(2)
    y = split_bf16_linear(x, w, weight_terms=2).double()
    exact = x.to(BF16).double() @ w.double().T
    released = torch.nn.functional.linear(x.to(BF16), w.to(BF16)).double()
    err_split = float((y - exact).norm() / exact.norm())
    err_released = float((released - exact).norm() / exact.norm())
    assert err_split <= err_released and err_split < 4e-3


def test_gradients():
    x, w, dy = _data(3)
    xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
    split_bf16_linear(xs, ws, weight_terms=2).backward(dy)
    xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
    torch.nn.functional.linear(xr.to(BF16), wr.to(BF16)).backward(dy)
    # weight gradient: the released expression bit for bit (residual terms are detached)
    assert torch.equal(ws.grad, wr.grad)
    # input gradient: sum of the widened BF16 term products, summed in FP32
    hi, lo = (p.to(BF16) for p in split_weight_terms(w, 2))
    expected = (dy @ hi).float() + (dy @ lo).float()
    assert torch.equal(xs.grad, expected)


def test_three_term_product_forward_and_gradients():
    x, w, dy = _data(4)
    y = split_bf16_linear(x, w, weight_terms=2, input_terms=2).double()
    exact = x.double() @ w.double().T
    # only the one BF16 rounding of the result remains (~2^-9 relative per element)
    assert float((y - exact).norm() / exact.norm()) < 3e-3
    assert float((y - exact.to(BF16).double()).abs().max()) <= float(exact.abs().max()) * 2.0 ** -7
    xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
    split_bf16_linear(xs, ws, weight_terms=2, input_terms=2).backward(dy)
    hi, lo = (p.to(BF16) for p in split_weight_terms(w, 2))
    x_hi = x.to(BF16)
    x_lo = (x - x_hi.float()).to(BF16)
    assert torch.equal(xs.grad, (dy @ hi).float() + (dy @ lo).float())
    assert torch.equal(ws.grad, (dy.t() @ x_hi).float() + (dy.t() @ x_lo).float())


def test_fp32_result_linear_forward_and_released_backward():
    from megakernel_lab.operation_compiler.split_precision import raw_forward_linear
    x, w, dy = _data(4)
    # mm(out_dtype=float32) (the conventional spelling) is CUDA-only.
    for conventional in ((False, True) if torch.cuda.is_available() else (False,)):
        xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = raw_forward_linear(xr, wr, input_terms=2, conventional=conventional)
        assert y.dtype == torch.float32
        exact = x.double() @ w.double().T
        assert float((y.detach().double() - exact).norm() / exact.norm()) < 2e-5
        y.backward(dy.float())
        xb, wb = x.clone().requires_grad_(), w.clone().requires_grad_()
        torch.nn.functional.linear(xb.to(BF16), wb.to(BF16)).backward(dy)
        assert torch.equal(xr.grad, xb.grad) and torch.equal(wr.grad, wb.grad)


def test_fp32_result_ragged_forward_and_released_backward():
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
    from megakernel_lab.operation_compiler.split_precision import ragged_split_operands, raw_forward_ragged_mm
    g = torch.Generator().manual_seed(5)
    x = torch.randn(10, 32, generator=g)
    w = torch.randn(3, 16, 32, generator=g)                                # [E, out, in] FP32 masters
    offsets = torch.tensor([0, 4, 4, 10])
    dy = torch.randn(10, 16, generator=g).to(BF16)
    wr, xr = w.clone().requires_grad_(), x.clone().requires_grad_()
    hi = wr.to(BF16)
    lo = (w - w.to(BF16).float()).to(BF16)
    y = raw_forward_ragged_mm(*ragged_split_operands(xr, hi, lo, input_terms=2), offsets)
    assert y.dtype == torch.float32
    exact = torch.cat([x[a:b].double() @ w[e].double().T for e, (a, b) in enumerate([(0, 4), (4, 4), (4, 10)])])
    assert float((y.detach().double() - exact).norm() / exact.norm()) < 2e-5
    y.backward(dy.float())
    wb, xb = w.clone().requires_grad_(), x.clone().requires_grad_()
    ragged_mm(xb.to(BF16), wb.to(BF16).transpose(1, 2), offsets).backward(dy)
    assert torch.equal(xr.grad, xb.grad) and torch.equal(wr.grad, wb.grad)


def test_shared_split_input_keeps_separate_input_casts():
    """w1/w3 sharing one split forward operand: the input gradient is still the released sum of separately widened
    BF16 cotangents (bitwise), and each forward equals its own unshared product."""
    from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
    from megakernel_lab.operation_compiler.split_precision import ragged_split_operands, raw_forward_ragged_mm
    g = torch.Generator().manual_seed(6)
    x = torch.randn(10, 32, generator=g)
    w1, w3 = torch.randn(3, 16, 32, generator=g), torch.randn(3, 16, 32, generator=g)
    offsets = torch.tensor([0, 4, 4, 10])
    d1, d3 = (torch.randn(10, 16, generator=g).to(BF16) for _ in range(2))
    lo = lambda w: (w - w.to(BF16).float()).to(BF16)
    xr = x.clone().requires_grad_()
    first = ragged_split_operands(xr, w1.to(BF16), lo(w1), input_terms=2)
    up = raw_forward_ragged_mm(*first, offsets)
    second = ragged_split_operands(xr, w3.to(BF16), lo(w3), input_terms=2, x_operands=(first[0], first[2]))
    gate = raw_forward_ragged_mm(*second, offsets)
    alone = raw_forward_ragged_mm(*ragged_split_operands(x, w3.to(BF16), lo(w3), input_terms=2), offsets)
    assert torch.equal(gate.detach(), alone)
    (up * d1.float()).sum().add((gate * d3.float()).sum()).backward()
    xb = x.clone().requires_grad_()
    (ragged_mm(xb.to(BF16), w1.to(BF16).transpose(1, 2), offsets).float() * d1.float()).sum().add(
        (ragged_mm(xb.to(BF16), w3.to(BF16).transpose(1, 2), offsets).float() * d3.float()).sum()).backward()
    assert torch.equal(xr.grad, xb.grad)
