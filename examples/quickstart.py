"""Compile a small MLP's training step, then train it.

    python examples/quickstart.py prepare out       # capture and compile; needs nvcc, not a GPU
    python examples/quickstart.py run out           # train on the GPU, one kernel launch per step

Add ``--target cpu`` to ``prepare`` to try it without a GPU (slow host emulation).
"""
import argparse

import torch
from torch import nn

import latch


def prepare(directory, hidden, lr, target):
    torch.manual_seed(1729)
    model = nn.Sequential(nn.Linear(64, hidden, bias=False), nn.Tanh(), nn.Linear(hidden, 64, bias=False))
    inputs = torch.randn(64, 64)
    captured = latch.capture_module(model, nn.functional.mse_loss, (inputs,), (inputs * 0.1,),
                                    optimizer=latch.AdamWConfig(lr=lr, eps=1e-4))
    manifest = captured.build(directory, target=target)
    print(f"compiled {len(captured.graph.ops)} operations into {manifest['items']} items "
          f"in {manifest['phases']} phases: {directory}/step.cu")


def run(directory, steps):
    update = latch.ModuleUpdate.load(directory)
    inputs = torch.randn(64, 64, generator=torch.Generator().manual_seed(1730))
    update.stage((inputs,), (inputs * 0.1,))
    for step in range(steps):
        print(f"step {step}: loss {update.step().item():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("directory")
    build.add_argument("--hidden", type=int, default=64)
    build.add_argument("--lr", type=float, default=0.001)
    build.add_argument("--target", choices=["cuda", "cpu"], default="cuda")
    train = actions.add_parser("run")
    train.add_argument("directory")
    train.add_argument("--steps", type=int, default=10)
    options = vars(parser.parse_args())
    {"prepare": prepare, "run": run}[options.pop("action")](**options)
