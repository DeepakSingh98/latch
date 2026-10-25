"""Prepare a BF16-recipe convolution update on CPU, then run and resume on H100.

The ordinary compose API chooses the admitted convolution layout and native
body. This synthetic example makes no learning or performance claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from megakernel_lab.operation_compiler import AdamWConfig, ModuleUpdate, capture_module


class Convolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 64, 3, 3) * 0.02)

    def forward(self, x):
        return F.conv2d(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), padding=1).float()


def batch(step):
    generator = torch.Generator().manual_seed(19100 + step)
    x = torch.randn(2, 64, 8, 8, generator=generator)
    return (x,), (0.1 * x,)


def prepare(directory, cache_dir):
    torch.manual_seed(19099)
    inputs, targets = batch(0)
    captured = capture_module(Convolution(), F.mse_loss, inputs, targets,
                              optimizer=AdamWConfig(lr=0.001, eps=1e-4))
    return captured.build(directory, cache_dir=cache_dir, arch='sm_90', execution='compose')


def run(directory, steps):
    if steps < 1:
        raise ValueError('steps must be positive')
    update = ModuleUpdate.load(directory)
    losses = []
    for step in range(steps):
        update.stage(*batch(step))
        update.step()
        losses.append(float(update.loss))
    before = update.state_dict()
    torch.save(before, Path(directory) / 'example-state.pt')
    update.stage(*batch(steps))
    update.step()
    after = update.state_dict()
    restored = ModuleUpdate.load(directory)
    restored.load_state_dict(torch.load(Path(directory) / 'example-state.pt', weights_only=False))
    restored.stage(*batch(steps))
    restored.step()
    replay = restored.state_dict()
    for group in ('parameters', 'optimizer_state', 'state'):
        assert set(replay[group]) == set(after[group])
        for name, value in after[group].items():
            assert torch.equal(value, replay[group][name]), (group, name)
            assert not value.is_floating_point() or torch.isfinite(value).all(), (group, name)
    assert all(torch.isfinite(torch.tensor(losses)))
    return {'passed': True, 'steps': steps, 'losses': losses, 'restart_bitwise': True,
            'execution': 'compose', 'source_hash': update.program.cache_key,
            'scope': 'Synthetic installed API and resume demonstration; no performance or learning claim'}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    cpu = actions.add_parser('prepare')
    cpu.add_argument('directory', type=Path)
    cpu.add_argument('--cache-dir', type=Path, required=True)
    gpu = actions.add_parser('run')
    gpu.add_argument('directory', type=Path)
    gpu.add_argument('--steps', type=int, default=10)
    args = vars(parser.parse_args())
    action = args.pop('action')
    print(json.dumps((prepare if action == 'prepare' else run)(**args), indent=2))


if __name__ == '__main__':
    main()
