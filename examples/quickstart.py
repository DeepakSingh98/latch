"""Compile and run a complete synthetic MLP training update.

The model and optimizer use the same APIs as operations/train_mlp.py. No dataset
or weights are downloaded. Prepare on a CPU host with CUDA 12.8, then run on an
H100 with the artifact and cache at the same absolute paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare(directory, cache_dir, hidden, lr):
    import torch
    from torch import nn
    from megakernel_lab.operation_compiler import AdamWConfig, capture_module

    if torch.cuda.is_available():
        raise RuntimeError("Build on CPU: set CUDA_VISIBLE_DEVICES= for this command")
    if hidden < 1 or lr <= 0:
        raise ValueError("hidden and lr must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(1729)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(64, hidden, bias=False)
            self.second = nn.Linear(hidden, 64, bias=False)

        def forward(self, x):
            return self.second(torch.tanh(self.first(x)))

    model = MLP()
    inputs = torch.randn(64, 64)
    targets = inputs * 0.1
    captured = capture_module(model, nn.functional.mse_loss, (inputs,), (targets,),
                              optimizer=AdamWConfig(lr=lr, eps=1e-4))
    receipt = captured.build(directory, cache_dir=cache_dir, arch="sm_90", execution="compose")
    return {"artifact": str(directory), "hidden": hidden, "lr": lr,
            "source_hash": receipt["source_hash"],
            "next": f"python examples/quickstart.py run {directory}"}


def run(directory, steps):
    import torch
    from megakernel_lab.operation_compiler import ModuleUpdate

    if steps < 1:
        raise ValueError("steps must be positive")
    torch.set_num_threads(1)
    update = ModuleUpdate.load(directory)
    generator = torch.Generator().manual_seed(1730)
    inputs = torch.randn(64, 64, generator=generator)
    update.stage((inputs,), (inputs * 0.1,))
    losses = []
    for _ in range(steps):
        update.step()
        losses.append(float(update.loss.item()))
    return {"steps": steps, "losses": losses, "source_hash": update.program.cache_key,
            "scope": "synthetic API demonstration; no speed or real-data learning claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    cpu = actions.add_parser("prepare", help="capture, lower and compile on CPU")
    cpu.add_argument("directory", type=Path)
    cpu.add_argument("--cache-dir", type=Path, required=True)
    cpu.add_argument("--hidden", type=int, default=64)
    cpu.add_argument("--lr", type=float, default=0.001)
    gpu = actions.add_parser("run", help="run generated training updates on H100")
    gpu.add_argument("directory", type=Path)
    gpu.add_argument("--steps", type=int, default=10)
    options = vars(parser.parse_args())
    action = options.pop("action")
    options["directory"] = options["directory"].expanduser().resolve()
    if action == "prepare":
        options["cache_dir"] = options["cache_dir"].expanduser().resolve()
    print(json.dumps((prepare if action == "prepare" else run)(**options), indent=2))


if __name__ == "__main__":
    main()
