"""Train a small character-level GPT with one kernel launch per step.

    python examples/gpt.py                 # compile, then train on the GPU
    python examples/gpt.py --compare       # also time the same training in ordinary PyTorch
    python examples/gpt.py --target cpu --steps 5 --batch 2 --context 8   # no GPU: slow host emulation

The model is plain PyTorch. Latch traces its forward pass, loss, gradients and
AdamW update, and compiles all of it into the kernel in ``<out>/step.cu``.
"""
import argparse
import math
import time

import torch
from torch import nn
import torch.nn.functional as F

import latch

TEXT = ("The quick brown fox jumps over the lazy dog. A journey of a thousand miles begins with a single step. "
        "All that glitters is not gold. Actions speak louder than words. The pen is mightier than the sword. "
        "When in Rome, do as the Romans do. Fortune favors the bold. Practice makes perfect. ") * 4


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.fc, self.out, self.heads = nn.Linear(width, 4 * width), nn.Linear(4 * width, width), heads

    def forward(self, x):
        batch, time, width = x.shape
        q, k, v = (t.view(batch, time, self.heads, -1).transpose(1, 2) for t in self.qkv(self.ln1(x)).split(width, dim=2))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        mask = torch.tril(torch.ones(time, time, dtype=torch.bool, device=x.device))
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        x = x + self.proj((weights @ v).transpose(1, 2).contiguous().view(batch, time, width))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, context, width, heads, layers):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, width), nn.Embedding(context, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(width), nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.head.weight = self.tok.weight  # tied

    def forward(self, tokens):
        x = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="latch-gpt")
    parser.add_argument("--target", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--compare", action="store_true", help="time the same training in ordinary PyTorch")
    args = parser.parse_args()

    torch.manual_seed(0)
    alphabet = sorted(set(TEXT))
    data = torch.tensor([alphabet.index(c) for c in TEXT])
    starts = torch.randint(0, len(data) - args.context - 1, (args.steps, args.batch))
    offsets = torch.arange(args.context)
    batches = [((data[s[:, None] + offsets],), (data[s[:, None] + offsets + 1],)) for s in starts]
    model = GPT(len(alphabet), args.context, args.width, heads=4, layers=args.layers)
    optimizer = latch.AdamWConfig(lr=3e-3)

    started = time.perf_counter()
    manifest = latch.capture_module(model, loss_fn, *batches[0], optimizer=optimizer).build(args.out, target=args.target)
    print(f"compiled in {time.perf_counter() - started:.1f} s: {manifest['items']} items in {manifest['phases']} phases")
    update = latch.ModuleUpdate.load(args.out)
    device = update.device
    batches = [(tuple(t.to(device) for t in x), tuple(t.to(device) for t in y)) for x, y in batches]

    started = time.perf_counter()
    for step, (inputs, targets) in enumerate(batches):
        update.stage(inputs, targets)
        loss = update.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step}: loss {loss.item():.4f}")
    final = loss.item()  # waits for the GPU
    latch_time = (time.perf_counter() - started) / args.steps
    print(f"latch: {latch_time * 1e6:.0f} us per step")

    if args.compare:
        model = model.to(device)
        reference = torch.optim.AdamW(model.parameters(), lr=optimizer.lr, betas=optimizer.betas, eps=optimizer.eps,
                                      weight_decay=optimizer.weight_decay)
        started = time.perf_counter()
        for inputs, targets in batches:
            reference.zero_grad()
            loss = loss_fn(model(*inputs), *targets)
            loss.backward()
            reference.step()
        expected = loss.item()
        eager_time = (time.perf_counter() - started) / args.steps
        print(f"pytorch: {eager_time * 1e6:.0f} us per step, final loss {expected:.4f} (latch {final:.4f})")


if __name__ == "__main__":
    main()
