"""Train a small attention block through the installed public compose API.

CPU prepare captures and compiles ordinary PyTorch attention. GPU run only
loads the saved artifact, executes updates and checks exact same/fresh-object
resume. This synthetic example makes no throughput or task-quality claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler import AdamWConfig, ModuleUpdate, capture_module


BATCH, TOKENS, WIDTH, HEADS = 2, 128, 128, 2


class AttentionBlock(nn.Module):
    def __init__(self, form):
        super().__init__()
        self.form = form
        if form == 'packed':
            self.qkv = nn.Parameter(torch.randn(WIDTH, 3 * WIDTH) / WIDTH**0.5)
        else:
            self.query = nn.Parameter(torch.randn(WIDTH, WIDTH) / WIDTH**0.5)
            self.key = nn.Parameter(torch.randn(WIDTH, WIDTH) / WIDTH**0.5)
            self.value = nn.Parameter(torch.randn(WIDTH, WIDTH) / WIDTH**0.5)
        self.output = nn.Parameter(torch.randn(WIDTH, WIDTH) / (2 * WIDTH)**0.5)
        self.register_buffer('mask', torch.ones(TOKENS, TOKENS, dtype=torch.bool).tril())

    def forward(self, x):
        flat = x.reshape(BATCH * TOKENS, WIDTH)
        def project(weight):
            return (flat.to(torch.bfloat16) @ weight.to(torch.bfloat16)).float()
        if self.form == 'packed':
            q, k, v = project(self.qkv).chunk(3, dim=-1)
        else:
            q, k, v = project(self.query), project(self.key), project(self.value)
        def heads(value):
            return value.reshape(BATCH, TOKENS, HEADS, WIDTH // HEADS).transpose(1, 2)
        value = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=self.mask)
        value = value.transpose(1, 2).reshape(BATCH * TOKENS, WIDTH)
        projected = (value.to(torch.bfloat16) @ self.output.to(torch.bfloat16)).float()
        return x + projected.reshape(BATCH, TOKENS, WIDTH)


def batch(step):
    generator = torch.Generator().manual_seed(8100 + step)
    x = torch.randn(BATCH, TOKENS, WIDTH, generator=generator)
    target = torch.randn(BATCH, TOKENS, WIDTH, generator=generator) * 0.1
    return x, target


def prepare(directory, cache_dir, form):
    if torch.cuda.is_available():
        raise RuntimeError('Prepare on a CPU-only build worker; run uses the prepared GPU artifact.')
    torch.manual_seed(8100)
    x, target = batch(0)
    captured = capture_module(AttentionBlock(form), F.mse_loss, (x,), (target,),
                              optimizer=AdamWConfig(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1))
    return captured.build(directory, cache_dir=cache_dir, arch='sm_90', execution='compose')


def exact(left, right):
    if isinstance(left, dict):
        return set(left) == set(right) and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape and
                torch.equal(left.cpu().contiguous().reshape(-1).view(torch.uint8),
                            right.cpu().contiguous().reshape(-1).view(torch.uint8)))
    return left == right


def run(directory, steps):
    if steps < 1:
        raise ValueError('steps must be positive')
    update = ModuleUpdate.load(directory)
    losses = []
    for step in range(steps):
        x, target = batch(step)
        update.stage((x,), (target,))
        update.step()
        losses.append(float(update.loss))
    saved = update.state_dict()
    snapshot_path = Path(directory) / 'attention-example-state.pt'
    torch.save(saved, snapshot_path)
    x, target = batch(steps)
    update.stage((x,), (target,))
    update.step()
    expected = update.state_dict()
    expected_loss = update.loss.detach().cpu().clone()
    update.load_state_dict(torch.load(snapshot_path, map_location='cpu', weights_only=False))
    update.stage((x,), (target,))
    update.step()
    same = exact(update.state_dict(), expected) and exact(update.loss.detach().cpu(), expected_loss)
    resumed = ModuleUpdate.load(directory)
    resumed.load_state_dict(torch.load(snapshot_path, map_location='cpu', weights_only=False))
    resumed.stage((x,), (target,))
    resumed.step()
    fresh = exact(resumed.state_dict(), expected) and exact(resumed.loss.detach().cpu(), expected_loss)
    assert same and fresh, 'Exact resume differs'
    assert all(torch.isfinite(torch.tensor(losses))), 'A training loss is non-finite'
    return {'passed': True, 'steps': steps, 'losses': losses,
            'source_hash': update.program.cache_key, 'execution_selection': update.runtime.manifest['execution_selection'],
            'same_object_exact': same, 'fresh_object_exact': fresh,
            'scope': 'Synthetic installed public-API and exact-resume demonstration; independent numerical qualification is separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    cpu = actions.add_parser('prepare')
    cpu.add_argument('directory', type=Path)
    cpu.add_argument('--cache-dir', type=Path, required=True)
    cpu.add_argument('--form', choices=('packed', 'separate'), default='packed')
    gpu = actions.add_parser('run')
    gpu.add_argument('directory', type=Path)
    gpu.add_argument('--steps', type=int, default=3)
    options = vars(parser.parse_args())
    action = options.pop('action')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    print(json.dumps((prepare if action == 'prepare' else run)(**options), indent=2))


if __name__ == '__main__':
    main()
