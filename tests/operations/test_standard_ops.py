"""CPU contracts for C-OPS2-B operation decompositions and refusals (cloud CPU)."""
from __future__ import annotations

import traceback

import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler import standard_ops as S


def _close(a, b, tol=1e-12):
    return bool(torch.allclose(a, b, rtol=tol, atol=tol))


def test_convolution_forward_backward():
    g = torch.Generator().manual_seed(1)
    for (n, c, h, w, o, k) in ((2, 3, 8, 12, 5, 4), (1, 1, 28, 28, 7, 4), (3, 2, 6, 6, 4, 2), (2, 4, 9, 9, 3, 3)):
        x = torch.randn(n, c, h, w, generator=g, dtype=torch.float64, requires_grad=True)
        wt = torch.randn(o, c, k, k, generator=g, dtype=torch.float64, requires_grad=True)
        b = torch.randn(o, generator=g, dtype=torch.float64, requires_grad=True)
        y = S.convolution(x, wt, b, (k, k), (0, 0), (1, 1), False, (0, 0), 1)
        ref = F.conv2d(x, wt, b, stride=k)
        assert y.shape == ref.shape and y.is_contiguous() and _close(y, ref)
        gy = torch.randn(ref.shape, generator=g, dtype=torch.float64)
        expected = torch.autograd.grad(ref, (x, wt, b), gy)
        actual = S.convolution_backward(gy, x.detach(), wt.detach(), [o], (k, k), (0, 0), (1, 1), False, (0, 0), 1,
                                        [True, True, True])
        assert all(_close(a, e) for a, e in zip(actual, expected))
        partial = S.convolution_backward(gy, x.detach(), wt.detach(), [o], (k, k), (0, 0), (1, 1), False, (0, 0), 1,
                                         [False, True, False])
        assert partial[0] is None and partial[2] is None and _close(partial[1], expected[1])


def test_convolution_refusals():
    x, w = torch.randn(1, 3, 8, 8), torch.randn(4, 3, 3, 3)
    for kwargs in ({'stride': (1, 1)}, {'stride': (3, 3), 'padding': (1, 1)}, {'stride': (2, 2)}):
        try:
            S.convolution(x, w, None, kwargs.get('stride'), kwargs.get('padding', (0, 0)), (1, 1), False, (0, 0), 1)
        except CaptureError as error:
            assert error.code == 'CONVOLUTION'
        else:
            raise AssertionError('overlapping/strided/padded convolution was admitted: ' + repr(kwargs))
    try:
        S.convolution(torch.randn(1, 4, 8, 8), torch.randn(4, 2, 4, 4), None, (4, 4), (0, 0), (1, 1), False, (0, 0), 2)
    except CaptureError:
        pass
    else:
        raise AssertionError('grouped convolution was admitted')


def test_sdpa_composition():
    g = torch.Generator().manual_seed(2)
    for causal in (False, True):
        for shape in ((2, 3, 17, 32), (4, 9, 24), (1, 2, 5, 16)):
            q, k, v = (torch.randn(*shape, generator=g, dtype=torch.float64, requires_grad=True) for _ in range(3))
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            out = S.sdpa_composition(q, k, v, is_causal=causal)
            assert _close(out, ref, 1e-10)
            gy = torch.randn(ref.shape, generator=g, dtype=torch.float64)
            assert all(_close(a, b, 1e-10) for a, b in zip(torch.autograd.grad(out, (q, k, v), gy),
                                                            torch.autograd.grad(ref, (q, k, v), gy)))
    q = torch.randn(1, 2, 4, 8)
    out = S.sdpa_composition(q, q, q, scale=0.3)
    assert _close(out.double(), F.scaled_dot_product_attention(q, q, q, scale=0.3).double(), 1e-6)
    # LATCH-ATTN-RECOG admits float masks of the query dtype (test_attention_recognition.py); a float mask of
    # another dtype stays refused, as in PyTorch.
    for kwargs in ({'attn_mask': torch.zeros(4, 4, dtype=torch.float64)}, {'dropout_p': 0.1}, {'enable_gqa': True}):
        try:
            S.sdpa_composition(q, q, q, **kwargs)
        except CaptureError as error:
            assert error.code == 'SDPA'
        else:
            raise AssertionError('refused SDPA option was admitted: ' + str(list(kwargs)))


def test_var_mean():
    x = torch.randn(3, 5, 7, dtype=torch.float64)
    for dim, correction, keepdim in (((-1,), 0, True), ((1, 2), 1, False), (None, 0, False), (0, None, True)):
        a = S.var_mean(x, dim, correction=correction, keepdim=keepdim)
        b = torch.var_mean(x, dim=dim, correction=correction, keepdim=keepdim)
        assert all(_close(u, w) for u, w in zip(a, b))


def test_decomposition_table():
    table = S.decompositions()
    aten = torch.ops.aten
    for op in (aten.native_layer_norm.default, aten.native_layer_norm_backward.default, aten.gelu.default,
               aten.gelu_backward.default, aten.addmm.default, aten.convolution.default,
               aten.convolution_backward.default, aten.var_mean.correction):
        assert op in table


def run():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                results[name] = {'passed': True}
            except Exception:
                results[name] = {'passed': False, 'traceback': traceback.format_exc()}
    return {'passed': all(r['passed'] for r in results.values()), 'count': len(results), 'results': results}
