"""Time one training step, Latch against ordinary PyTorch on the same GPU.

    python examples/bench.py            # every model in the README's table
    python examples/bench.py mlp        # only the MLPs (or: gpt)
"""
import copy
import sys
import tempfile
import time

import torch
from torch import nn
import torch.nn.functional as F

import latch
from gpt import GPT, loss_fn as token_loss


def mlp(width):
    return nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width))


def time_latch(model, loss_fn, batch, optimizer, steps):
    directory = tempfile.mkdtemp(prefix="latch-")
    started = time.perf_counter()
    manifest = latch.capture_module(model, loss_fn, *batch, optimizer=optimizer).build(directory)
    compile_time = time.perf_counter() - started
    update = latch.ModuleUpdate.load(directory)
    update.stage(*batch)
    for _ in range(5):
        update.step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        update.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / steps, update.loss.item(), compile_time, manifest


def time_pytorch(model, loss_fn, batch, optimizer, steps):
    model = copy.deepcopy(model).cuda()
    inputs, targets = ([t.cuda() for t in group] for group in batch)
    reference = torch.optim.AdamW(model.parameters(), lr=optimizer.lr, betas=optimizer.betas, eps=optimizer.eps,
                                  weight_decay=optimizer.weight_decay, fused=True)

    def step():
        reference.zero_grad(set_to_none=True)
        loss = loss_fn(model(*inputs), *targets)
        loss.backward()
        reference.step()
        return loss

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        loss = step()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / steps, loss.item()


def main():
    torch.manual_seed(0)
    optimizer, which = latch.AdamWConfig(lr=1e-3), sys.argv[1:] or ["mlp", "gpt"]
    cases = []
    if "mlp" in which:
        for width, steps in ((64, 500), (512, 200), (2048, 50)):
            batch = ((torch.randn(width, width),), (torch.randn(width, width),))
            cases.append((f"MLP, 3 layers of width {width}, batch {width}", mlp(width), F.mse_loss, batch, steps))
    if "gpt" in which:
        for layers, width, context, size, steps in ((2, 64, 32, 8, 200), (4, 128, 64, 16, 100), (6, 384, 256, 32, 20)):
            batch = ((torch.randint(0, 65, (size, context)),), (torch.randint(0, 65, (size, context)),))
            cases.append((f"GPT, {layers} layers, width {width}, context {context}, batch {size}",
                          GPT(65, context, width, heads=4 if width < 384 else 6, layers=layers), token_loss, batch, steps))
    print(f"{torch.cuda.get_device_name()}, PyTorch {torch.__version__}")
    for name, model, loss_fn, batch, steps in cases:
        ours, loss, compile_time, manifest = time_latch(model, loss_fn, batch, optimizer, steps)
        theirs, expected = time_pytorch(model, loss_fn, batch, optimizer, steps)
        print(f"{name}: latch {ours * 1e6:.0f} us, pytorch {theirs * 1e6:.0f} us per step "
              f"({theirs / ours:.2f}x); loss {loss:.4f} vs {expected:.4f}; "
              f"{manifest['items']} items in {manifest['phases']} phases, compiled in {compile_time:.0f} s", flush=True)


if __name__ == "__main__":
    main()
