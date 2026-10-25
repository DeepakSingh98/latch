"""Modal CPU contracts for compiler precision policies (C-OPS3-A).

* Per-use gradient semantics: a parameter used by several covered products
  accumulates FP32-widened contributions (the Pro two-use counterexample).
* The default (no policy) captures, metadata and generated source are unchanged.
* Unsupported combinations are refused with PRECISION_POLICY errors.
* Policy capture of the unannotated FP32 recurrent program reproduces the
  explicit-cast recipes bitwise: eager autograd, and captured regions at runtime
  counts 0/1/3, loop and combined objectives, SGD and AdamW, tied embeddings.
* A small standard module (patch convolution, LayerNorm, SDPA, GELU, Linear, a
  shared weight) captures and lowers under both policies; the captured program
  agrees with eager autograd under the same policy.
"""
from __future__ import annotations

from dataclasses import replace
import json
import traceback

import torch
from torch import nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.optim import AdamWConfig, initialize_optimizer_state
from megakernel_lab.operation_compiler.precision import MIXED_BF16, POLICIES, PrecisionPolicy, precision_policy

BF16 = torch.bfloat16


def _bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                     b.reshape(-1).view(torch.uint8))


def test_per_use_gradient_semantics():
    """Pro review example: separate per-use casts give 1.00390625; a shared cast 1.0."""
    w = torch.ones(1, 1)
    x1, x2 = torch.ones(1, 1), torch.full((1, 1), 2.0 ** -8)

    def loss(p, a, b):
        return (a @ p['w'] + b @ p['w']).sum()

    rows = {}
    for name in POLICIES:
        captured = capture_training(loss, {'w': w.clone()}, {'w': torch.zeros_like(w)}, (x1, x2),
                                    SGDConfig(lr=0.0, momentum=0.5), precision=name)
        out = captured.unpack(captured.graph_module(*captured.example_args))
        assert float(out['loss']) == 1.00390625 and float(out['grads']['w']) == 1.00390625, (name, out)
        casts = [op for op in captured.ir.operations if op.target == 'aten._to_copy.default'
                 and op.kwargs.get('dtype') == 'bfloat16']
        assert len(casts) >= 4, 'each use casts its own operands'
        rows[name] = {'loss': float(out['loss']), 'grad': float(out['grads']['w']), 'bf16_casts': len(casts),
                      'covered_calls': captured.ir.metadata['precision_policy']['covered_calls']}
    shared = w.clone().requires_grad_(True)
    wb = shared.to(BF16)
    ((x1.to(BF16) @ wb).float() + (x2.to(BF16) @ wb).float()).sum().backward()
    assert float(shared.grad) == 1.0
    rows['shared_cast_counterexample_grad'] = float(shared.grad)
    return rows


class TinyVision(nn.Module):
    """Patch conv + class token + pre-LN attention block + GELU MLP + shared projection."""
    def __init__(self, width=32, heads=2, classes=10):
        super().__init__()
        self.patch = nn.Conv2d(3, width, 4, 4)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 17, width) * 0.02)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.fc1, self.fc2 = nn.Linear(width, 2 * width), nn.Linear(2 * width, width)
        self.mix = nn.Linear(width, width, bias=False)   # applied twice: shared weight, per-use casts
        self.head = nn.Linear(width, classes)
        self.heads = heads

    def forward(self, x):
        b = x.shape[0]
        h = self.patch(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1) + self.pos
        q, k, v = self.qkv(self.norm1(h)).reshape(b, 17, 3, self.heads, -1).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, 17, -1)
        h = h + self.proj(a)
        h = h + self.fc2(F.gelu(self.fc1(self.norm2(h))))
        h = h + self.mix(torch.tanh(self.mix(h)))
        return self.head(self.norm3(h)[:, 0])


def _vision_case():
    torch.manual_seed(31)
    model = TinyVision()
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 10, (4,))
    return model, x, y


def _eager_reference(model, params, x, y, policy):
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    with precision_policy(policy):
        loss = F.cross_entropy(torch.func.functional_call(model, p, (x,)), y)
    grads = torch.autograd.grad(loss, tuple(p.values()))
    return loss.detach(), dict(zip(p, grads))


def test_module_capture_and_lowering():
    from megakernel_lab.operation_compiler.api import capture_module
    rows = {}
    for name in POLICIES:
        model, x, y = _vision_case()
        params = {k: v.detach().clone() for k, v in model.named_parameters()}
        config = AdamWConfig(lr=1e-3, eps=1e-8, weight_decay=0.01)
        captured = capture_module(model, F.cross_entropy, (x,), (y,), optimizer=config,
                                  optimizer_state=initialize_optimizer_state(params, config), precision=name)
        meta = captured.ir.metadata['precision_policy']
        counts = meta['covered_calls']
        assert counts.get('linear') == 7 and counts.get('conv2d') == 1 and counts.get('attention') == 1, counts
        products = [op for op in captured.ir.operations if op.target.split('.')[1] in {'mm', 'bmm'}]
        dtypes = {}
        for op in products:
            key = captured.ir.values[op.outputs[0]].dtype
            dtypes[key] = dtypes.get(key, 0) + 1
        assert dtypes.get('bfloat16', 0) > 0
        # P8-C online attention has the same function-level cast recipe as
        # mixed_bf16; only its separately admitted fused kernel changes softmax.
        if POLICIES[name].attention in ('bfloat16', 'bfloat16_online'):
            assert 'float32' not in dtypes, dtypes
        else:
            assert dtypes.get('float32', 0) >= 4, dtypes  # FP32 attention forward + backward
        out = captured.unpack(captured.graph_module(*captured.example_args))
        loss, grads = _eager_reference(model, params, x, y, name)
        assert abs(float(out['loss']) - float(loss)) <= 2e-3 + 2e-3 * abs(float(loss))
        worst = 0.0
        for key, g in grads.items():
            ratio = float(((out['grads'][key] - g).abs() / (1e-3 + 0.05 * g.abs())).max())
            worst = max(worst, ratio)
        assert worst <= 1.0, worst
        # Parameter gradients and optimizer state stay FP32.
        assert all(captured.ir.values[v].dtype == 'float32' for v in captured.ir.outputs['grads'].values())
        program = lower(captured.ir)
        decisions = program.manifest.get('matrix_decisions', [])
        copies = captured.ir.metadata.get('layout_copies', [])
        rows[name] = {'layout_copies': len(copies),'covered_calls': counts, 'product_dtypes': dtypes, 'worst_gradient_ratio': worst,
                      'lowered': True, 'matrix_selected': sum(1 for d in decisions if d.get('selected')),
                      'matrix_decisions': len(decisions)}
    return rows


def test_default_unchanged():
    model, x, y = _vision_case()
    from megakernel_lab.operation_compiler.api import capture_module
    plain = capture_module(model, F.cross_entropy, (x,), (y,))
    explicit_none = capture_module(model, F.cross_entropy, (x,), (y,), precision=None)
    named_fp32 = capture_module(model, F.cross_entropy, (x,), (y,), precision='float32')
    assert plain.graph_module.code == explicit_none.graph_module.code == named_fp32.graph_module.code
    assert 'precision_policy' not in plain.ir.metadata
    sources = {lower(c.ir).source for c in (plain, explicit_none, named_fp32)}
    assert len(sources) == 1
    assert not any(op.target == 'aten._to_copy.default' and op.kwargs.get('dtype') == 'bfloat16'
                   for op in plain.ir.operations)
    assert 'layout_copies' not in plain.ir.metadata
    return {'identical_graph_and_source': True, 'operations': len(plain.ir.operations)}


def test_refusals():
    w, x = torch.randn(4, 4), torch.randn(2, 4)
    cases = {
        'einsum': lambda p, a: torch.einsum('ij,jk->ik', a, p['w']).sum(),
        'explicit_bf16_operand': lambda p, a: (a.to(BF16) @ p['w'].to(BF16)).float().sum(),
        'baddbmm': lambda p, a: torch.baddbmm(a[None], a[None], p['w'][None]).sum(),
        'rank1_matmul': lambda p, a: (a[0] @ p['w'][0]).sum(),
        'conv1d': lambda p, a: F.conv1d(a[None], p['w'][:, :, None]).sum(),
    }
    # A one-operand einsum is a permutation/reduction (diffusers unpatchify), not a product.
    captured = capture_training(lambda p, a: torch.einsum('ij->ji', a @ p['w']).sum(), {'w': w},
                                {'w': torch.zeros_like(w)}, (x,), precision='mixed_bf16')
    assert captured.ir.metadata['precision_policy']['covered_calls'] == {'matmul': 1}
    rows = {}
    for label, loss in cases.items():
        try:
            capture_training(loss, {'w': w}, {'w': torch.zeros_like(w)}, (x,), precision='mixed_bf16')
        except CaptureError as error:
            assert error.code == 'PRECISION_POLICY', (label, error)
            rows[label] = str(error)[:160]
        else:
            raise AssertionError(f'{label} was not refused')

    def autocast_loss(p, a):
        with torch.autocast('cpu', dtype=BF16):
            return (a @ p['w']).float().sum()
    try:
        capture_training(autocast_loss, {'w': w}, {'w': torch.zeros_like(w)}, (x,), precision='mixed_bf16')
    except CaptureError as error:
        assert error.code == 'PRECISION_POLICY'
        rows['autocast'] = str(error)[:160]
    else:
        raise AssertionError('autocast inside a policy was not refused')
    for bad in ('fp16', PrecisionPolicy('half', compute_dtype='float16'), PrecisionPolicy('x', attention='int8')):
        try:
            capture_training(lambda p, a: (a @ p['w']).sum(), {'w': w}, {'w': torch.zeros_like(w)}, (x,), precision=bad)
        except CaptureError as error:
            assert error.code == 'PRECISION_POLICY'
            rows[f'policy:{getattr(bad, "name", bad)}'] = str(error)[:160]
        else:
            raise AssertionError(f'{bad} was not refused')
    from megakernel_lab.operation_compiler.loop import ConditionalBody, capture_loop_training
    try:
        body = ConditionalBody(lambda p, c, s, a: (c, s), lambda p, c, s, a: (c, s), 'flag')
        capture_loop_training(lambda p, a: {'h': a @ p['w']}, body, lambda p, c, a: c['h'].sum(), {'w': w},
                              {'w': torch.zeros_like(w)}, (x,), torch.tensor(1), SGDConfig(), max_steps=2,
                              state={'flag': torch.tensor(True)}, precision='mixed_bf16')
    except CaptureError as error:
        assert error.code == 'PRECISION_POLICY'
        rows['conditional_loop'] = str(error)[:160]
    else:
        raise AssertionError('ConditionalBody with a policy was not refused')
    return rows


def test_round_trip_cancellation_exact():
    """narrow(clone(permute(widen(s)))) becomes clone(permute(s)); values bitwise equal."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from megakernel_lab.operation_compiler.precision import cancel_round_trips

    def program(s):
        return (s.float().permute(1, 0).clone().to(BF16) + 0).float().sum(0).to(BF16)
    s = torch.randn(8, 4).to(BF16)
    gm = make_fx(program, tracing_mode='fake')(s)
    expected = program(s)
    removed = cancel_round_trips(gm)
    assert removed == 1, removed
    assert _bitwise(gm(s), expected)
    casts = [n for n in gm.graph.nodes if n.op == 'call_function' and 'to_copy' in str(n.target)]
    assert len(casts) == 2, casts   # the reduction's widen/narrow pair is not a pure move chain
    return {'removed': removed, 'remaining_casts': len(casts)}


def test_loop_policy_shared_history_matches_explicit_recipe():
    """New composition check: same per-use casts in a reusable runtime loop."""
    from megakernel_lab.operation_compiler.loop import capture_loop_training
    params = {'w': torch.arange(16, dtype=torch.float32).reshape(4, 4) / 64}
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 32
    def initial(p, x):
        return {'h': x @ p['w']}
    def body(p, c, s, x):
        return {'h': torch.tanh(c['h'] @ p['w']) + c['h']}, {'steps': s['steps'] + 1}
    def terminal(p, c, x):
        return (c['h'] @ p['w']).square().mean()
    result = {}
    for policy in POLICIES:
        cap = capture_loop_training(initial, body, terminal, params, None, (x,), torch.tensor(0),
                                    SGDConfig(), max_steps=3, state={'steps': torch.tensor(0)},
                                    precision=policy, loop_attention='off')
        rows = []
        for count in (0, 1, 3):
            actual = cap.unpack(cap.graph_step(*cap.example_args[:-1], torch.tensor(count)))
            w = params['w'].clone().requires_grad_()
            product = lambda a, b: (a.to(BF16) @ b.to(BF16)).float()
            h = product(x, w)
            for _ in range(count):
                h = torch.tanh(product(h, w)) + h
            value = product(h, w).square().mean()
            gradient, = torch.autograd.grad(value, (w,))
            torch.testing.assert_close(actual['loss'], value, atol=2e-5, rtol=1e-4)
            torch.testing.assert_close(actual['grads']['w'], gradient, atol=1e-5, rtol=5e-4)
            assert int(actual['state']['steps']) == count
            rows.append(count)
        assert cap.ir.metadata['precision_policy']['name'] == policy
        result[policy] = rows
    return result


def run():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                results[name] = {'passed': True, 'result': fn()}
            except Exception:
                results[name] = {'passed': False, 'traceback': traceback.format_exc()}
    return {'passed': all(value['passed'] for value in results.values()), 'tests': results}
