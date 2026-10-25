"""CPU contracts for the overlapping 3x3 convolution class (release step 10; cloud CPU).

Independent references are PyTorch's own FP64 ``F.conv2d`` and its autograd. They cover border pixels
(1- and 2-pixel extents, non-square images), overlapping input-gradient contributions, bias on/off, partial
output masks, the captured complete update (forward, AD and optimizer in the captured graph) and lowering.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler import conv_ops as C
from megakernel_lab.operation_compiler import standard_ops as S

SHAPES = ((2, 3, 5, 4, 5), (1, 1, 1, 1, 2), (1, 2, 1, 6, 3), (2, 4, 2, 2, 4), (3, 2, 7, 3, 1), (1, 5, 9, 9, 6))


def _close(a, b, tol=1e-12):
    return a.shape == b.shape and bool(torch.allclose(a, b, rtol=tol, atol=tol))


def _args(stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1):
    return stride, padding, dilation, transposed, output_padding, groups


def test_patch_gather_matches_unfold():
    g = torch.Generator().manual_seed(10)
    for n, c, h, w, _ in SHAPES:
        x = torch.randn(n, c, h, w, generator=g, dtype=torch.float64)
        patches = C.conv3_patches(x, False)
        # F.unfold: (N, C*9, H*W) with (c, kh, kw) rows; the gather is tap-major (kh, kw, c).
        expected = F.unfold(x, 3, padding=1).reshape(n, c, 9, h * w).permute(2, 1, 0, 3).reshape(c * 9, n * h * w)
        assert torch.equal(patches, expected)
        flipped = C.conv3_patches(x, True)
        expected = F.unfold(x.flip(2, 3), 3, padding=1).reshape(n, c, 9, h, w).flip(3, 4)
        expected = expected.permute(2, 1, 0, 3, 4).reshape(c * 9, n * h * w)
        assert torch.equal(flipped, expected)


def _forms():
    previous = C.FORM
    try:
        for form in ("padded", "patches", "auto"):
            C.FORM = form
            yield form
    finally:
        C.FORM = previous


def test_padded_operands_match_patches():
    g = torch.Generator().manual_seed(12)
    for n, c, h, w, _ in SHAPES:
        x = torch.randn(n, c, h, w, generator=g, dtype=torch.float64)
        width, columns, margin, stored = C.padded_geometry(n, h, w)
        assert width % 8 == 0 and margin % 64 == 0 and margin >= width and columns % C.PAD_ALIGN == 0
        padded = C.conv3_pad(x)
        assert padded.shape == (3 * c, stored)
        centre = padded[c:2 * c, margin:margin + columns]
        ring = centre[:, :n * (h + 2) * width].reshape(c, n, h + 2, width)
        assert torch.equal(ring[:, :, 1:h + 1, 1:w + 1], x.permute(1, 0, 2, 3))
        outside = ring.clone()
        outside[:, :, 1:h + 1, 1:w + 1] = 0
        assert not outside.any() and not centre[:, n * (h + 2) * width:].any()
        for flip in (False, True):
            taps = C.conv3_taps(padded, width, margin, flip)
            frame = taps[:, :n * (h + 2) * width].reshape(9 * c, n, h + 2, width)[:, :, 1:h + 1, 1:w + 1]
            assert torch.equal(frame.reshape(9 * c, n * h * w), C.conv3_patches(x, flip))


def test_forward_backward_against_conv2d():
    for form in _forms():
        _forward_backward_against_conv2d()


def _forward_backward_against_conv2d():
    g = torch.Generator().manual_seed(11)
    for n, c, h, w, o in SHAPES:
        for bias in (True, False):
            x = torch.randn(n, c, h, w, generator=g, dtype=torch.float64, requires_grad=True)
            wt = torch.randn(o, c, 3, 3, generator=g, dtype=torch.float64, requires_grad=True)
            b = torch.randn(o, generator=g, dtype=torch.float64, requires_grad=True) if bias else None
            y = S.convolution(x, wt, b, *_args())
            ref = F.conv2d(x, wt, b, padding=1)
            assert y.is_contiguous() and _close(y, ref)
            gy = torch.randn(ref.shape, generator=g, dtype=torch.float64)
            leaves = (x, wt) + ((b,) if bias else ())
            expected = torch.autograd.grad(ref, leaves, gy)
            actual = S.convolution_backward(gy, x.detach(), wt.detach(), [o], *_args(), [True, True, bias])
            assert _close(actual[0], expected[0]) and _close(actual[1], expected[1])
            assert (actual[2] is None) if not bias else _close(actual[2], expected[2])
            for mask in ([True, False, False], [False, True, False], [False, False, bias]):
                partial = S.convolution_backward(gy, x.detach(), wt.detach(), [o], *_args(), mask)
                for got, want, on in zip(partial, expected + (None,) * (3 - len(expected)), mask):
                    assert (got is None) if not on else _close(got, want)


def test_overlapping_input_gradient_contributions():
    for form in _forms():
        _overlapping_input_gradient_contributions()


def _overlapping_input_gradient_contributions():
    # One unit output gradient at an interior pixel reaches exactly its 3x3 neighbourhood with the flipped
    # weights; a corner pixel reaches only its 2x2 in-image neighbourhood.
    wt = torch.arange(1.0, 10.0, dtype=torch.float64).reshape(1, 1, 3, 3)
    x = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
    for (i, j) in ((2, 2), (0, 0), (4, 3)):
        gy = torch.zeros(1, 1, 5, 5, dtype=torch.float64)
        gy[0, 0, i, j] = 1.0
        dx = S.convolution_backward(gy, x, wt, [1], *_args(), [True, False, False])[0]
        expected = torch.zeros(5, 5, dtype=torch.float64)
        for dh in range(3):
            for dw in range(3):
                r, s = i + dh - 1, j + dw - 1
                if 0 <= r < 5 and 0 <= s < 5:
                    expected[r, s] = wt[0, 0, dh, dw]
        assert torch.equal(dx[0, 0], expected)
    # Every interior input pixel collects nine overlapping contributions; a corner pixel four.
    ones = torch.ones(1, 1, 5, 5, dtype=torch.float64)
    dx = S.convolution_backward(ones, x, wt, [1], *_args(), [True, False, False])[0]
    assert float(dx[0, 0, 2, 2]) == 45.0 and float(dx[0, 0, 0, 0]) == 1 + 2 + 4 + 5


def test_refusals_unchanged_outside_class():
    x = torch.randn(1, 3, 8, 8)
    cases = [(torch.randn(4, 3, 3, 3), _args(padding=(0, 0))),       # 3x3 without padding
             (torch.randn(4, 3, 3, 3), _args(stride=(2, 2))),         # strided
             (torch.randn(4, 3, 3, 3), _args(dilation=(2, 2), padding=(2, 2))),
             (torch.randn(4, 3, 5, 5), _args(padding=(2, 2))),        # 5x5
             (torch.randn(4, 3, 3, 1), _args(padding=(1, 0))),        # 3x1
             (torch.randn(4, 1, 3, 3), _args(groups=3))]
    for weight, args in cases:
        try:
            S.convolution(x, weight, None, *args)
        except CaptureError as error:
            assert error.code == 'CONVOLUTION'
        else:
            raise AssertionError(f'convolution outside the admitted classes was admitted: {args}')
    try:
        S.convolution(x, torch.randn(4, 3, 3, 3), None, *_args(transposed=True))
    except CaptureError:
        pass
    else:
        raise AssertionError('transposed convolution was admitted')


class Block(nn.Module):
    """Patch stem (existing non-overlapping path) + two overlapping 3x3 convolutions + head."""

    def __init__(self, width=8, classes=5, bias2=False):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 2, 2)
        self.conv1 = nn.Conv2d(width, width, 3, padding=1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=bias2)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = self.stem(x)
        h = h + self.conv2(F.silu(self.conv1(h)))
        return self.head(F.silu(h).mean(dim=(2, 3)))


def _case(bias2):
    torch.manual_seed(5 + bias2)
    model = Block(bias2=bias2)
    x = torch.randn(3, 3, 10, 6)
    y = torch.randint(0, 5, (3,))
    return model, x, y


def test_captured_update_matches_fp64_autograd():
    for form in _forms():
        _captured_update_matches_fp64_autograd(form)


def _captured_update_matches_fp64_autograd(form):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.optim import SGDConfig
    for bias2 in (False, True):
        model, x, y = _case(bias2)
        config = SGDConfig(lr=0.05, momentum=0.9, weight_decay=0.0)
        captured = capture_module(model, F.cross_entropy, (x,), (y,), optimizer=config)
        targets = [op.target for op in captured.ir.operations]
        gathers = sum(t in C.TARGETS for t in targets)
        # padded per conv: 3 taps, 3 pads (2 when the forward copy is reused), 2 interior extractions (optional)
        per_conv = 3 + (2 if C.REUSE_PAD else 3) + (2 if C.INTERIOR_OP else 0)
        assert gathers == (2 * per_conv if form == "padded" else 6), targets
        assert not any('convolution' in t for t in targets)
        out = captured.unpack(captured.graph_module(*captured.example_args))
        params = {k: v.detach().double().requires_grad_(True) for k, v in model.named_parameters()}
        loss = F.cross_entropy(torch.func.functional_call(model, params, (x.double(),)), y)
        grads = dict(zip(params, torch.autograd.grad(loss, tuple(params.values()))))
        assert abs(float(out['loss']) - float(loss)) < 1e-5
        for k, g in grads.items():
            got = out['grads'][k].double()
            assert float((got - g).norm() / g.norm().clamp_min(1e-30)) < 1e-5, k
            assert float((out['params'][k].double() - (params[k] - 0.05 * g)).abs().max()) < 1e-6, k


def test_bf16_policy_capture():
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.optim import SGDConfig
    from megakernel_lab.operation_compiler.precision import precision_policy
    model, x, y = _case(True)
    captured = capture_module(model, F.cross_entropy, (x,), (y,), optimizer=SGDConfig(lr=0.05, momentum=0.9),
                              precision='bf16_projections')
    assert captured.ir.metadata['precision_policy']['covered_calls'].get('conv2d') == 3
    patches = [op for op in captured.ir.operations if op.target in C.TARGETS]
    assert patches and all(captured.ir.values[op.outputs[0]].dtype == 'bfloat16' for op in patches)
    out = captured.unpack(captured.graph_module(*captured.example_args))
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
    with precision_policy('bf16_projections'):
        loss = F.cross_entropy(torch.func.functional_call(model, params, (x,)), y)
    grads = dict(zip(params, torch.autograd.grad(loss, tuple(params.values()))))
    assert abs(float(out['loss']) - float(loss)) < 2e-3
    for k, g in grads.items():
        assert float((out['grads'][k] - g).norm() / g.norm().clamp_min(1e-30)) < 2e-2, k


def test_virtual_taps_lowering():
    """BF16 compose lowering reads admitted tap operands in place (no taps storage); FP32 materializes them."""
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig

    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv2d(128, 128, 3, padding=1)
            self.b = nn.Conv2d(128, 128, 3, padding=1, bias=False)

        def forward(self, x):
            return self.b(F.silu(self.a(x))).square().mean()

    torch.manual_seed(3)
    x = torch.randn(32, 128, 14, 14)
    previous, C.NHWC = C.NHWC, False      # the step-10 padded form (LATCH-STEP10-SPEED's form: test_conv_nhwc.py)
    try:
        _virtual_taps_lowering(Pair, x)
    finally:
        C.NHWC = previous


def _virtual_taps_lowering(Pair, x):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    for precision in ('bf16_projections', None):
        captured = capture_module(Pair(), lambda out: out, (x,), (), optimizer=SGDConfig(lr=0.01, momentum=0.9),
                                  precision=precision)
        program = lower(captured.ir, execution='compose')
        virtual = program.manifest.get('conv_virtual_operands', [])
        if precision:
            assert len(virtual) >= 4 and 'g.vb_c = 128' in program.source, virtual
            dead = {row['taps'] for row in virtual} - set(program.manifest.get('conv_materialized_taps', []))
            assert dead and not dead & set(program.manifest['offsets']), dead
        else:
            assert not virtual and 'g.vb_c' not in program.source


def test_lowering_emits_gather_and_products():
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    model, x, y = _case(True)
    captured = capture_module(model, F.cross_entropy, (x,), (y,), optimizer=SGDConfig(lr=0.05, momentum=0.9))
    for execution in (None, 'compose'):
        program = lower(captured.ir, execution=execution)
        assert 'tap' in program.source and 'g.vb_c' not in program.source
