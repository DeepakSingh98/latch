"""Small synthetic nn.Module training example for the installed operation compiler.

prepare runs on a CPU build worker with torch 2.9.1 and nvcc. run loads that
exact artifact on an H100, with no model tracing or compilation on the GPU.
The worker artifact and cache paths must remain available at their saved paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from megakernel_lab.operation_compiler import AdamWConfig, ModuleUpdate, capture_module


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(64, 64, bias=False)
        self.second = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.second(torch.tanh(self.first(x)))


def prepare(directory, cache_dir):
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    model = MLP()
    inputs, targets = torch.randn(64, 64), torch.randn(64, 64) * 0.1
    captured = capture_module(model, nn.functional.mse_loss, (inputs,), (targets,),
                              optimizer=AdamWConfig(lr=0.001, eps=1e-4))
    return captured.build(directory, cache_dir=cache_dir, arch="sm_90", execution="compose")


def run(directory, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    torch.set_num_threads(1)
    update = ModuleUpdate.load(directory)
    # Stage copies data into the already bound device tensors; shapes stay fixed.
    generator = torch.Generator().manual_seed(1730)
    inputs = torch.randn(64, 64, generator=generator)
    targets = inputs * 0.1
    update.stage((inputs,), (targets,))
    pointers = tuple(value.data_ptr() for value in update.runtime.args)
    losses = []
    for _ in range(steps):
        update.step()
        losses.append(float(update.loss.item()))
    snapshot = update.state_dict()
    torch.save(snapshot, Path(directory) / "example-state.pt")
    assert pointers == tuple(value.data_ptr() for value in update.runtime.args)
    return {"steps": steps, "losses": losses, "source_hash": update.program.cache_key,
            "optimizer_counters": {key: int(value) for key, value in snapshot["optimizer_state"].items()
                                   if key.startswith("step:")},
            "scope": "synthetic API demonstration; no performance or real-data quality claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    cpu = actions.add_parser("prepare")
    cpu.add_argument("directory", type=Path)
    cpu.add_argument("--cache-dir", type=Path, required=True)
    gpu = actions.add_parser("run")
    gpu.add_argument("directory", type=Path)
    gpu.add_argument("--steps", type=int, default=20)
    options = vars(parser.parse_args())
    action = options.pop("action")
    print(json.dumps((prepare if action == "prepare" else run)(**options), indent=2))


if __name__ == "__main__":
    main()
