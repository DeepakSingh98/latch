"""CPU contracts for the pixel-major convolution form and channels-last propagation (LATCH-STEP10-SPEED; cloud CPU).

Independent references are FP64 ``F.conv2d``/autograd and the step-10 patch gather. Covered: the im2col operand
against the patch matrix (borders, 1-pixel extents, non-square images, flipped taps), forward and every gradient
of the decomposition, the capture-time channels-last pass (converted operations, no remaining NCHW round trips,
unchanged values of the captured complete update), BF16 policy capture, and compose lowering that reads the tap
operands through im2col maps without storage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler import conv_ops as C
from . import test_conv_ops as T

SHAPES = T.SHAPES


def _nhwc():
    previous = C.FORM
    C.FORM = "nhwc"
    return previous


def test_im2col_matches_patches():
    g = torch.Generator().manual_seed(30)
    for n, c, h, w, _ in SHAPES:
        x = torch.randn(n, c, h, w, generator=g, dtype=torch.float64)
        xp = C.conv3_pixels(x)
        assert torch.equal(xp, x.permute(0, 2, 3, 1).reshape(n * h * w, c))
        assert torch.equal(C.conv3_unpixels(xp, n, h, w), x)
        for flip in (False, True):
            assert torch.equal(C.conv3_im2col(xp, n, h, w, flip), C.conv3_patches(x, flip).t())


def test_nhwc_forward_backward_against_conv2d():
    previous = _nhwc()
    try:
        T._forward_backward_against_conv2d()
        T._overlapping_input_gradient_contributions()
    finally:
        C.FORM = previous


class Block(nn.Module):
    """Two overlapping 3x3 convolutions with a residual, as the step-10 block programs (input is a feature map)."""

    def __init__(self, width=8, classes=5, bias1=True, bias2=False):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=bias1)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=bias2)
        self.head = nn.Linear(width, classes)

    def forward(self, x):
        h = x + self.conv2(F.silu(self.conv1(x)))
        return self.head(F.silu(h).mean(dim=(2, 3)))


def _capture(model, x, y, precision=None):
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.optim import SGDConfig
    return capture_module(model, F.cross_entropy, (x,), (y,),
                          optimizer=SGDConfig(lr=0.05, momentum=0.9, weight_decay=0.0), precision=precision)


def test_channels_last_capture_matches_fp64_autograd():
    previous = _nhwc()
    try:
        for bias1, bias2, shape in ((True, False, (3, 8, 5, 7)), (False, True, (2, 8, 1, 6)), (True, True, (1, 8, 4, 4))):
            torch.manual_seed(31)
            model = Block(bias1=bias1, bias2=bias2)
            x = torch.randn(*shape)
            y = torch.randint(0, 5, (shape[0],))
            captured = _capture(model, x, y)
            targets = [op.target for op in captured.ir.operations]
            record = captured.ir.metadata.get('channels_last')
            assert record and record['converted'] > 0 and record['sources'] == 3, record
            assert C.UNPIXELS_TARGET not in targets, targets
            assert sum(t == C.IM2COL_TARGET for t in targets) == 5, targets
            out = captured.unpack(captured.graph_module(*captured.example_args))
            params = {k: v.detach().double().requires_grad_(True) for k, v in model.named_parameters()}
            loss = F.cross_entropy(torch.func.functional_call(model, params, (x.double(),)), y)
            grads = dict(zip(params, torch.autograd.grad(loss, tuple(params.values()))))
            assert abs(float(out['loss']) - float(loss)) < 1e-5
            for k, grad in grads.items():
                got = out['grads'][k].double()
                assert float((got - grad).norm() / grad.norm().clamp_min(1e-30)) < 1e-5, k
                assert float((out['params'][k].double() - (params[k] - 0.05 * grad)).abs().max()) < 1e-6, k
    finally:
        C.FORM = previous


def test_channels_last_off_is_step10():
    """With the pass disabled the same capture keeps NCHW elementwise operations and explicit (un)pixel copies."""
    from megakernel_lab.operation_compiler import channels_last
    previous = _nhwc()
    channels_last.ENABLED = False
    try:
        torch.manual_seed(32)
        model = Block()
        x, y = torch.randn(2, 8, 4, 4), torch.randint(0, 5, (2,))
        captured = _capture(model, x, y)
        targets = [op.target for op in captured.ir.operations]
        assert 'channels_last' not in captured.ir.metadata and C.UNPIXELS_TARGET in targets
        out = captured.unpack(captured.graph_module(*captured.example_args))
        assert torch.isfinite(out['loss'])
    finally:
        channels_last.ENABLED = True
        C.FORM = previous


def test_bf16_policy_auto_selects_nhwc():
    from megakernel_lab.operation_compiler.precision import precision_policy
    torch.manual_seed(33)
    model = Block(width=64)
    x, y = torch.randn(2, 64, 6, 5), torch.randint(0, 5, (2,))
    captured = _capture(model, x, y, precision='bf16_projections')
    targets = [op.target for op in captured.ir.operations]
    assert C.IM2COL_TARGET in targets and captured.ir.metadata.get('channels_last'), targets
    taps = [op for op in captured.ir.operations if op.target == C.IM2COL_TARGET]
    assert all(captured.ir.values[op.outputs[0]].dtype == 'bfloat16' for op in taps)
    out = captured.unpack(captured.graph_module(*captured.example_args))
    params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
    with precision_policy('bf16_projections'):
        loss = F.cross_entropy(torch.func.functional_call(model, params, (x,)), y)
    grads = dict(zip(params, torch.autograd.grad(loss, tuple(params.values()))))
    assert abs(float(out['loss']) - float(loss)) < 2e-3
    for k, grad in grads.items():
        assert float((out['grads'][k] - grad).norm() / grad.norm().clamp_min(1e-30)) < 2e-2, k


def test_nhwc_lowering_reads_im2col_in_place():
    """BF16 compose lowering reads every tap operand through an im2col map (no taps storage, no NCHW copies)."""
    from megakernel_lab.operation_compiler.lower import lower
    torch.manual_seed(34)
    model = Block(width=128)
    x, y = torch.randn(8, 128, 8, 8), torch.randint(0, 5, (8,))
    captured = _capture(model, x, y, precision='bf16_projections')
    program = lower(captured.ir, execution='compose')
    virtual = program.manifest.get('conv_virtual_operands', [])
    operands = sorted(row.get('operand') for row in virtual)
    assert operands.count('A') == 3 and operands.count('B') == 2, virtual
    assert 'g.vi_c = 128' in program.source and 'i2c' not in program.source  # markers became map-table entries
    assert not program.manifest.get('conv_materialized_taps'), program.manifest.get('conv_materialized_taps')
    dead = {row['taps'] for row in virtual}
    assert not dead & set(program.manifest['offsets']), dead
    kinds = [phase.get('targets') for phase in program.manifest.get('phases', [])]
    assert not any(t and t[0] in (C.UNPIXELS_TARGET,) for t in kinds), kinds
    # Weight gradients: grid-wide split-K reduction phases, no private arrival counters.
    phases = program.manifest.get('phases', [])
    split = [p for p in phases if p.get('kind') == 'gemm' and (p.get('matrix_spec') or {}).get('conv_splitk', 1) > 1]
    assert [p.get('kind') for p in phases].count('splitk_reduce') == len(split)
    assert all(p['matrix_spec'].get('cnt') == 'nullptr' for p in split), split


def test_vector_loops_keep_broadcast_reads():
    """An identity-indexed phase with a broadcast operand is vectorized; the broadcast stays a per-element load."""
    from megakernel_lab.operation_compiler import group_rules
    from megakernel_lab.operation_compiler.lower import _CONV_PROGRAM
    token = _CONV_PROGRAM.set(True)          # the lane's rules apply to pixel-major convolution programs
    try:
        _vector_loops_keep_broadcast_reads(group_rules)
    finally:
        _CONV_PROGRAM.reset(token)
    assert group_rules.vector_mixed(*_broadcast_phase()) is None   # other programs: unchanged (scalar loop)


def _broadcast_phase():
    names = ['a', 'b']
    body = ['const float r_a = (v_x[(i)] + v_bias[(long long)((unsigned)(i) % 128u) * 1LL]);',
            'const __nv_bfloat16 r_b = __float2bfloat16_rn(r_a); v_b[i] = r_b;']
    texts = {'v_x[(i)]': ('v_x', 'float32'), 'v_x[i]': ('v_x', 'float32'),
             'v_b[(i)]': ('v_b', 'bfloat16'), 'v_b[i]': ('v_b', 'bfloat16')}
    return {'names': names, 'body': body}, 1 << 20, texts, 2


def _vector_loops_keep_broadcast_reads(group_rules):
    names = ['a', 'b']
    body = ['const float r_a = (v_x[(i)] + v_bias[(long long)((unsigned)(i) % 128u) * 1LL]);',
            'const __nv_bfloat16 r_b = __float2bfloat16_rn(r_a); v_b[i] = r_b;']
    texts = {'v_x[(i)]': ('v_x', 'float32'), 'v_x[i]': ('v_x', 'float32'),
             'v_b[(i)]': ('v_b', 'bfloat16'), 'v_b[i]': ('v_b', 'bfloat16')}
    code = group_rules.vector_mixed({'names': names, 'body': body}, 1 << 20, texts, unroll=2)
    # chunk-uniform index terms (exact): element 3 of the chunk at 8 * jc0 reads column (8 * jc0) % 128 + 3
    assert code and 'v_bias[(long long)(((unsigned)(8 * jc0) % 128u + 3u))' in code and 'float4' in code, code
    previous, group_rules.VEC_BROADCAST = group_rules.VEC_BROADCAST, False
    try:
        assert group_rules.vector_mixed({'names': names, 'body': body}, 1 << 20, texts, unroll=2) is None
    finally:
        group_rules.VEC_BROADCAST = previous
    # the doubly parenthesized index form of the lowering hoists the same way
    body2 = [body[0].replace('(unsigned)(i)', '(unsigned)((i))'), body[1]]
    code2 = group_rules.vector_mixed({'names': names, 'body': body2}, 1 << 20, texts, unroll=2)
    assert code2 and '((unsigned)(8 * jc0) % 128u + 3u)' in code2, code2
    # pipelined form: the next iteration's 16-byte loads are issued before the current compute and stores
    previous, group_rules.VEC_PIPELINE = group_rules.VEC_PIPELINE, True
    try:
        piped = group_rules.vector_mixed({'names': names, 'body': body}, 1 << 20, texts, unroll=2)
    finally:
        group_rules.VEC_PIPELINE = previous
    assert piped.count('nq0_0_0 = reinterpret_cast') == 1 and 'q0_0_0 = nq0_0_0;' in piped, piped
    assert piped.index('nq0_1_1 = ') < piped.index('const float r_a_0_0') < piped.index('q0_1_1 = nq0_1_1;'), piped


class StemBlock(nn.Module):
    """A 3-channel 2x2 patch stem (as the development program: a (C, 12) weight gradient with K = pixels) before a
    pixel-major block."""

    def __init__(self, width=64):
        super().__init__()
        self.stem = nn.Conv2d(3, width, 2, stride=2)
        self.block = Block(width=width)

    def forward(self, x):
        return self.block(F.silu(self.stem(x)))


def test_generic_split_long_k_stem():
    """The stem's long-K weight gradient uses the existing split BF16 body with private partials and counters."""
    from megakernel_lab.operation_compiler.lower import lower
    torch.manual_seed(36)
    model = StemBlock()
    x, y = torch.randn(16, 3, 32, 32), torch.randint(0, 5, (16,))
    captured = _capture(model, x, y, precision='bf16_projections')
    program = lower(captured.ir, execution='compose')
    phases = program.manifest.get('phases', [])
    split = [p for p in phases if p.get('engine') in ('generic_bf16_simt_16_split', 'p9e_skinny_split')]
    assert split, '\n'.join(str((p.get('names')[:2], p.get('engine'), p.get('k'), p.get('tiles'),
                                  (p.get('matrix_spec') or {}).get('m'), (p.get('matrix_spec') or {}).get('n')))
                             for p in phases if p.get('kind') == 'gemm')
    assert '/*P2_' not in program.source
    assert sum('generic_split_counter_zero' in (p.get('targets') or []) for p in phases) == len(split)


def test_nhwc_lowering_64_channels():
    """64 channels: 128 x 64 product tiles and padded weight-gradient extents, every tap operand still in place."""
    from megakernel_lab.operation_compiler.lower import lower
    torch.manual_seed(35)
    model = Block(width=64)
    x, y = torch.randn(8, 64, 8, 8), torch.randint(0, 5, (8,))
    captured = _capture(model, x, y, precision='bf16_projections')
    program = lower(captured.ir, execution='compose')
    virtual = program.manifest.get('conv_virtual_operands', [])
    operands = sorted(row.get('operand') for row in virtual)
    assert operands.count('A') == 3 and operands.count('B') == 2, virtual
    reasons = [d.get('reason', '') for d in program.manifest['matrix_decisions']]
    assert sum('computed padded to 128x640' in r for r in reasons) == 2, reasons
    assert not program.manifest.get('conv_materialized_taps')


def test_single_slice_padded_weight_gradient_owns_full_output_storage():
    """The installed BF16 MSE+AdamW example must not write 128x640 into a 64x576 root.

    Padding is independent of K slicing. The old split-K-only allocation left
    this single-slice weight gradient pointing at the compact logical output,
    overwriting the arena before its first complete update could finish.
    """
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig

    class Convolution(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 64, 3, 3) * 0.02)

        def forward(self, x):
            return F.conv2d(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), padding=1).float()

    for optimizer in (AdamWConfig(lr=0.001, eps=1e-4), SGDConfig()):
        torch.manual_seed(19099)
        model = Convolution()
        x = torch.randn(2, 64, 8, 8, generator=torch.Generator().manual_seed(19100))
        captured = capture_module(model, F.mse_loss, (x,), (0.1 * x,), optimizer=optimizer)
        program = lower(captured.ir, execution='compose')
        manifest = program.manifest
        padded = [phase for phase in manifest['phases'] if
                  phase.get('matrix_spec', {}).get('conv_splitk') == 1 and
                  'm_extent' in phase.get('matrix_spec', {})]
        assert len(padded) == 1, padded
        phase = padded[0]
        spec = phase['matrix_spec']
        assert (spec['m'], spec['n'], spec['k'], spec['m_extent'], spec['n_extent']) == (128, 640, 128, 64, 576)
        assert not spec.get('splitk') and spec['out'].startswith('v_conv_pad_'), spec
        root = spec['out'][2:]
        layout = manifest['layouts'][root]
        assert layout['root'] == root and layout['shape'] == [128, 640] and layout['strides'] == [640, 1]
        assert layout['dtype'] == 'bfloat16'
        assert 4 * manifest['offsets'][root] + 2 * 128 * 640 <= manifest['arena_bytes']
        assert root in phase['writes']
        extracts = [p for p in manifest['phases'] if root in p.get('reads', []) and
                    p.get('initialization') == 'extract the (M, N) block of a padded convolution product']
        assert len(extracts) == 1, extracts
        assert any(manifest['layouts'][name]['shape'] == [64, 576] for name in extracts[0]['writes'])


def test_exact_division_rewrite():
    """Branch-free exact division: division and reciprocal text forms, an exact fallback per chunk in the program."""
    from megakernel_lab.operation_compiler.p5c_epilogue import exact_division
    assert (exact_division('const float r_s = (r_a / (1.0f + expf(-r_a))); s81_[j] = r_s;')
            == 'const float r_s = latch_xdiv(r_a, (1.0f + expf(-r_a)), xok_); s81_[j] = r_s;')
    assert (exact_division('const float r_g = __frcp_rn((1.0f + expf(-p8f0_[j])));')
            == 'const float r_g = latch_xdiv(1.0f, ((1.0f + expf(-p8f0_[j]))), xok_);')
    assert exact_division('const float r_d = (r_x / 256.0f);') == 'const float r_d = (r_x / 256.0f);'
    # The former Block fixture no longer admits this vector rewrite in accepted
    # c24 either (conv-c24-probe01). Exercise the helper and its exact fallback
    # directly; the convolution/autograd contracts above remain independent.
    from megakernel_lab.operation_compiler import group_rules
    from megakernel_lab.operation_compiler.lower import _CONV_PROGRAM
    from megakernel_lab.operation_compiler.p5c_matrix import _include_exact_division
    phase = {'names': ['a', 's'], 'body': [
        'const float r_a = v_x[i];',
        'const float r_s = (r_a / (1.0f + expf(-r_a))); v_s[i] = r_s;',
    ]}
    texts = {'v_x[i]': ('v_x', 'float32'), 'v_s[i]': ('v_s', 'float32')}
    token = _CONV_PROGRAM.set(True)
    try:
        code = group_rules.vector_mixed(phase, 8192, texts, unroll=2)
    finally:
        _CONV_PROGRAM.reset(token)
    assert code and 'latch_xdiv(' in code and 'if (!xok_)' in code
    fast, fallback = code.split('if (!xok_)', 1)
    assert 'if (xok_)' in fast and 'reinterpret_cast<float4*>(v_s)' in fast
    assert 'latch_xdiv(' not in fallback and '/ (1.0f + expf(' in fallback
    assert 'reinterpret_cast<float4*>(v_s)' in fallback
    source = _include_exact_division(code)
    assert source.count('static __device__ __forceinline__ float latch_xdiv') == 1
    assert _include_exact_division(source) == source
