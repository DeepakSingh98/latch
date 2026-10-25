"""Modify the attention example into a new small learning program, then train and restart it.

Starting from ``train_attention_forms.py`` (packed form), this program:

* adds a frozen random-feature input projection (``requires_grad=False``) beside
  trainable attention weights, so the optimizer updates only part of the module;
* adds an auxiliary loss term (a penalty on the attention output) to the main
  regression loss, by returning two values from ``forward``;
* exposes the sequence length, width and head count as command-line options.

``prepare`` captures and compiles on a CPU worker that has ``nvcc`` (CUDA 12.8).
``run`` loads the prepared artifact on an H100, trains, and writes a snapshot.
Running ``run`` again with ``--resume`` continues in a *new process* from that
snapshot; ``compare`` checks that a split run is bit-identical to an unsplit one.
``check`` compares the first update's loss and gradients with eager PyTorch.
For release validation, ``check-snapshot`` saves native tensors on H100 and
``check-audit`` computes the reference and errors on a separate CPU worker.

Changing any dimension produces a different program: prepare it again into a new
directory. Snapshots only load into the exact program that wrote them.
This is a synthetic usage demonstration; it makes no speed or task-quality claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler import AdamWConfig, ModuleUpdate, capture_module


class FrozenFeatureAttention(nn.Module):
    def __init__(self, batch, tokens, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by heads')
        self.shape = (batch, tokens, width, heads)
        # Frozen: captured as a read-only snapshot with no gradient or optimizer state.
        self.features = nn.Parameter(torch.randn(width, width) / width**0.5, requires_grad=False)
        self.qkv = nn.Parameter(torch.randn(width, 3 * width) / width**0.5)
        self.output = nn.Parameter(torch.randn(width, width) / (2 * width)**0.5)
        self.register_buffer('mask', torch.ones(tokens, tokens, dtype=torch.bool).tril())

    def forward(self, x):
        batch, tokens, width, heads = self.shape
        def project(value, weight):   # explicit BF16 products, FP32 results (as in the base example)
            return (value.to(torch.bfloat16) @ weight.to(torch.bfloat16)).float()
        flat = torch.tanh(project(x.reshape(batch * tokens, width), self.features))
        q, k, v = project(flat, self.qkv).chunk(3, dim=-1)
        def split(value):
            return value.reshape(batch, tokens, heads, width // heads).transpose(1, 2)
        mixed = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=self.mask)
        mixed = mixed.transpose(1, 2).reshape(batch * tokens, width)
        prediction = x + project(mixed, self.output).reshape(batch, tokens, width)
        return prediction, mixed          # the second output feeds the auxiliary loss


def make_loss(aux_weight):
    def loss_fn(outputs, target):
        prediction, mixed = outputs
        return F.mse_loss(prediction, target) + aux_weight * mixed.pow(2).mean()
    return loss_fn


def batch_at(step, config):
    generator = torch.Generator().manual_seed(9100 + step)
    shape = (config['batch'], config['tokens'], config['width'])
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator) * 0.1


def build_model(config):
    torch.manual_seed(9100)
    return FrozenFeatureAttention(config['batch'], config['tokens'], config['width'], config['heads'])


def prepare(directory, cache_dir, batch, tokens, width, heads, aux_weight):
    if torch.cuda.is_available():
        raise RuntimeError('Prepare on a CPU-only build worker (for example CUDA_VISIBLE_DEVICES=).')
    config = dict(batch=batch, tokens=tokens, width=width, heads=heads, aux_weight=aux_weight)
    x, target = batch_at(0, config)
    captured = capture_module(build_model(config), make_loss(aux_weight), (x,), (target,),
                              optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1))
    receipt = captured.build(directory, cache_dir=cache_dir, arch='sm_90', execution='compose')
    (Path(directory) / 'example-config.json').write_text(json.dumps(config, indent=2) + '\n')
    selection = receipt.get('execution_selection', {})
    # 0 attention bodies means SDPA was lowered as generic operations (for example head width 96):
    # it still trains, but without the fast attention implementation. Nothing else warns about this.
    return {'config': config, 'source_hash': receipt['source_hash'], 'execution': receipt['execution'],
            'attention_bodies': selection.get('admitted_capabilities', {}).get('attention'),
            'prepare_seconds': receipt['prepare_seconds'], 'build_seconds': receipt['build_seconds'],
            'frozen': sorted(receipt['module_api'].get('input_indices', {}).get('frozen', {}))}


def run(directory, steps, start_step, resume, save):
    if steps < 1:
        raise ValueError('steps must be positive')
    config = json.loads((Path(directory) / 'example-config.json').read_text())
    update = ModuleUpdate.load(directory)
    frozen_before = {k: v.detach().cpu().clone() for k, v in update.frozen_parameters().items()}
    if resume is not None:
        update.load_state_dict(torch.load(resume, map_location='cpu', weights_only=True))
    losses = []
    for step in range(start_step, start_step + steps):
        x, target = batch_at(step, config)
        update.stage((x,), (target,))
        update.step()
        losses.append(float(update.loss))
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError('a training loss is non-finite')
    frozen_unchanged = all(torch.equal(frozen_before[k], v.detach().cpu())
                           for k, v in update.frozen_parameters().items())
    if not frozen_unchanged:
        raise RuntimeError('a frozen parameter changed during training')
    if save is not None:
        torch.save(update.state_dict(), save)
    return {'steps': list(range(start_step, start_step + steps)), 'losses': losses,
            'resumed_from': None if resume is None else str(resume), 'saved': None if save is None else str(save),
            'frozen_unchanged': frozen_unchanged, 'source_hash': update.program.cache_key}


def compare(left, right):
    a = torch.load(left, map_location='cpu', weights_only=True)
    b = torch.load(right, map_location='cpu', weights_only=True)
    def same(u, v):
        if isinstance(u, dict):
            return set(u) == set(v) and all(same(u[k], v[k]) for k in u)
        if isinstance(u, torch.Tensor):
            return u.dtype == v.dtype and u.shape == v.shape and torch.equal(
                u.contiguous().view(-1).view(torch.uint8), v.contiguous().view(-1).view(torch.uint8))
        return u == v
    result = {'bitwise_equal': same(a, b), 'left': str(left), 'right': str(right)}
    if not result['bitwise_equal']:
        raise SystemExit(json.dumps(result))
    return result


def check(directory):
    """Smoke check: first-update loss and gradients against eager PyTorch on the same GPU.

    A loose tolerance catches a wrong objective (a dropped term, a frozen weight that
    trains). It is not a numerical qualification of the compiled program.
    """
    config = json.loads((Path(directory) / 'example-config.json').read_text())
    update = ModuleUpdate.load(directory)
    x, target = batch_at(0, config)
    update.stage((x,), (target,))
    update.step()
    model = build_model(config).cuda()
    loss = make_loss(config['aux_weight'])(model(x.cuda()), target.cuda())
    loss.backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
    rows = {'loss': (float(update.loss), float(loss.detach()))}
    worst = abs(rows['loss'][0] - rows['loss'][1]) / max(abs(rows['loss'][1]), 1e-12)
    finite = all(math.isfinite(value) for value in rows['loss']) and math.isfinite(worst)
    for name, reference in grads.items():
        native = update.grads[name].float()
        error = float((native - reference).norm() / reference.norm().clamp_min(1e-12))
        rows[name] = error
        finite = finite and math.isfinite(error)
        worst = max(worst, error)
    trained = sorted(update.grads)
    result = {'native_vs_eager': rows, 'worst_relative_error': worst, 'trained_parameters': trained,
              'frozen_not_trained': 'features' not in trained, 'finite': finite, 'tolerance': 2e-2,
              'scope': 'smoke check against eager PyTorch (same BF16 casts); not a qualification'}
    if not finite or worst > 2e-2 or 'features' in trained:
        raise SystemExit(json.dumps(result))
    return result


def check_snapshot(directory, save):
    """Execute one native update and save raw CPU copies, without reference arithmetic."""
    config = json.loads((Path(directory) / 'example-config.json').read_text())
    update = ModuleUpdate.load(directory)
    x, target = batch_at(0, config)
    initial = update.state_dict()
    frozen_before = {name: value.detach().cpu().clone() for name, value in update.frozen_parameters().items()}
    update.stage((x,), (target,))
    update.step()
    snapshot = {'schema': 'latch-modified-attention-smoke-v1', 'config': config,
                'source_hash': update.program.cache_key, 'initial_state': initial,
                'x': x, 'target': target, 'loss': update.loss.detach().cpu().clone(),
                'grads': {name: value.detach().cpu().clone() for name, value in update.grads.items()},
                'frozen_before': frozen_before,
                'frozen_after': {name: value.detach().cpu().clone() for name, value in update.frozen_parameters().items()}}
    torch.save(snapshot, save)
    return {'saved': str(save), 'source_hash': update.program.cache_key,
            'scope': 'Native execution and raw tensor copies; CPU smoke audit is pending.'}


def check_audit(snapshot):
    """CPU eager smoke reference with the same BF16 casts and unchanged 2e-2 bound."""
    if torch.cuda.is_available():
        raise RuntimeError('check-audit requires a CPU-only worker')
    torch.set_num_threads(1)
    data = torch.load(snapshot, map_location='cpu', weights_only=True)
    if data['schema'] != 'latch-modified-attention-smoke-v1':
        raise ValueError('unknown modified-attention smoke snapshot')
    if data['source_hash'] != data['initial_state']['source_hash']:
        raise ValueError('initial state belongs to a different program')
    config = data['config']
    model = build_model(config)
    def exact(left, right):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8))
    x, target = batch_at(0, config)
    if not exact(x, data['x']) or not exact(target, data['target']):
        raise ValueError('smoke minibatch differs from the documented seed')
    initial = data['initial_state']['parameters']
    expected_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected_frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    if set(initial) != expected_trainable or set(data['grads']) != expected_trainable:
        raise ValueError('trainable parameter or gradient coverage differs')
    if set(data['frozen_before']) != expected_frozen or set(data['frozen_after']) != expected_frozen:
        raise ValueError('frozen parameter coverage differs')
    for name, parameter in model.named_parameters():
        value = initial[name] if parameter.requires_grad else data['frozen_before'][name]
        if not exact(parameter.detach(), value):
            raise ValueError('starting parameter differs from the documented initialization: ' + name)
    frozen_unchanged = all(exact(data['frozen_before'][name], data['frozen_after'][name]) for name in expected_frozen)
    # CPU math attention is an independent eager implementation of the same
    # explicit-cast objective. This remains a loose smoke check, not qualification.
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        loss = make_loss(config['aux_weight'])(model(x), target)
        loss.backward()
    rows = {'loss': (float(data['loss']), float(loss.detach()))}
    worst = abs(rows['loss'][0] - rows['loss'][1]) / max(abs(rows['loss'][1]), 1e-12)
    finite = all(math.isfinite(value) for value in rows['loss']) and math.isfinite(worst)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        native, reference = data['grads'][name], parameter.grad
        if native.dtype != reference.dtype or native.shape != reference.shape:
            raise ValueError('gradient metadata differs: ' + name)
        error = float((native.float() - reference).norm() / reference.norm().clamp_min(1e-12))
        rows[name] = error
        finite = finite and math.isfinite(error)
        worst = max(worst, error)
    return {'passed': finite and worst <= 2e-2 and frozen_unchanged,
            'native_vs_eager': rows, 'worst_relative_error': worst, 'tolerance': 2e-2,
            'trained_parameters': sorted(data['grads']), 'frozen_not_trained': 'features' not in data['grads'],
            'frozen_unchanged': frozen_unchanged, 'starting_parameters_exact': True, 'minibatch_exact': True,
            'finite': finite, 'reference_device': 'cpu', 'attention_reference': 'PyTorch math SDPA',
            'source_hash': data['source_hash'],
            'scope': 'CPU eager smoke reference with the same explicit BF16 casts and unchanged 2e-2 bound; not numerical qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest='action', required=True)
    cpu = actions.add_parser('prepare', help='capture and compile (CPU worker with nvcc)')
    cpu.add_argument('directory', type=Path)
    cpu.add_argument('--cache-dir', type=Path, required=True)
    cpu.add_argument('--batch', type=int, default=2)
    cpu.add_argument('--tokens', type=int, default=128)
    cpu.add_argument('--width', type=int, default=128)
    cpu.add_argument('--heads', type=int, default=2)
    cpu.add_argument('--aux-weight', type=float, default=0.01)
    gpu = actions.add_parser('run', help='train on an H100; optionally resume and save snapshots')
    gpu.add_argument('directory', type=Path)
    gpu.add_argument('--steps', type=int, default=4)
    gpu.add_argument('--start-step', type=int, default=0, help='index of the first minibatch')
    gpu.add_argument('--resume', type=Path, help='snapshot written by an earlier run of this program')
    gpu.add_argument('--save', type=Path, help='write a snapshot after the last step')
    same = actions.add_parser('compare', help='check two snapshots are bit-identical')
    same.add_argument('left', type=Path)
    same.add_argument('right', type=Path)
    smoke = actions.add_parser('check', help='first-update loss/gradients against eager PyTorch (H100)')
    smoke.add_argument('directory', type=Path)
    snapshot = actions.add_parser('check-snapshot', help='execute native update and save raw tensors (H100)')
    snapshot.add_argument('directory', type=Path)
    snapshot.add_argument('--save', type=Path, required=True)
    audit = actions.add_parser('check-audit', help='compute eager smoke reference and errors on CPU')
    audit.add_argument('snapshot', type=Path)
    options = vars(parser.parse_args())
    action = options.pop('action')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {'prepare': prepare, 'run': run, 'compare': compare, 'check': check,
              'check-snapshot': check_snapshot, 'check-audit': check_audit}[action](**options)
    print(json.dumps(result, indent=2))
    if action == 'check-audit' and result['passed'] is not True:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
