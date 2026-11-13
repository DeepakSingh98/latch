"""Whole training steps, compiled and run, against the same steps in ordinary PyTorch."""
import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import latch
from conftest import torch_optim


class MLP(nn.Module):
    def __init__(self, sizes, activation=torch.tanh, bias=True, norm=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b, bias=bias) for a, b in zip(sizes, sizes[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(b) for b in sizes[1:-1]) if norm else None
        self.activation = activation

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            if self.norms is not None:
                x = self.norms[i](x)
        return self.layers[-1](x)


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
    def __init__(self, vocab=40, context=12, width=24, heads=3, layers=2, tie=True):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, width), nn.Embedding(context, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(width), nn.Linear(width, vocab, bias=False)
        if tie:
            self.head.weight = self.tok.weight

    def forward(self, tokens):
        x = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


def token_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def regression(batch, inputs, outputs, steps=3):
    return [((torch.randn(batch, inputs),), (torch.randn(batch, outputs),)) for _ in range(steps)]


def classification(batch, inputs, classes, steps=3):
    return [((torch.randn(batch, inputs),), (torch.randint(0, classes, (batch,)),)) for _ in range(steps)]


def tokens(batch, context, vocab, steps=3):
    return [((torch.randint(0, vocab, (batch, context)),), (torch.randint(0, vocab, (batch, context)),))
            for _ in range(steps)]


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_quickstart_mlp(train):
    train(MLP([64, 64, 64], bias=False), F.mse_loss, regression(64, 64, 64), latch.AdamWConfig(lr=1e-3, eps=1e-4))


@pytest.mark.parametrize("optimizer", [
    latch.SGDConfig(lr=0.1), latch.SGDConfig(lr=0.1, momentum=0.9, weight_decay=1e-3),
    latch.AdamWConfig(lr=1e-2, eps=1e-3)])
def test_optimizers_match_torch_optim(train, optimizer):
    train(MLP([16, 32, 10], F.gelu), F.cross_entropy, classification(8, 16, 10, steps=4), optimizer,
          reference=torch_optim)


@pytest.mark.parametrize("activation", [torch.relu, torch.sigmoid, F.gelu, torch.tanh])
def test_activations(train, activation):
    train(MLP([9, 20, 4], activation), F.mse_loss, regression(6, 9, 4), latch.SGDConfig(lr=0.05))


def test_odd_sizes_and_layer_norm(train):
    """Extents that are not multiples of a matrix tile, and the reductions inside LayerNorm."""
    train(MLP([13, 37, 21, 5], F.gelu, norm=True), F.cross_entropy, classification(7, 13, 5),
          latch.SGDConfig(lr=0.1, momentum=0.9))


def test_large_tiles_and_wide_reductions(train):
    """Big enough for the larger matrix tiles, and for bias gradients reduced by whole blocks."""
    train(MLP([70, 300, 130]), F.mse_loss, regression(1100, 70, 130, steps=2), latch.SGDConfig(lr=0.05))


def test_gpt(train):
    train(GPT(), token_loss, tokens(3, 12, 40), latch.SGDConfig(lr=0.1, momentum=0.9), tolerance=5e-5)


def test_gpt_adamw_untied(train):
    train(GPT(layers=1, tie=False), token_loss, tokens(2, 12, 40), latch.AdamWConfig(lr=1e-3, eps=1e-3),
          tolerance=5e-5)


def test_frozen_parameters_and_buffers_are_constants(train):
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen, self.head = nn.Linear(6, 8), nn.Linear(8, 3)
            self.frozen.requires_grad_(False)
            self.register_buffer("scale", torch.linspace(0.5, 1.5, 8))

        def forward(self, x):
            return self.head(torch.tanh(self.frozen(x)) * self.scale)

    model = Scaled()
    update = train(model, F.mse_loss, regression(5, 6, 3), latch.SGDConfig(lr=0.1))
    assert set(update.parameters) == {"head.weight", "head.bias"}


def test_embedding_with_repeated_tokens(train):
    """Repeated indices accumulate into one row of the embedding gradient."""
    class Bag(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed, self.head = nn.Embedding(11, 16), nn.Linear(16, 4)

        def forward(self, tokens):
            return self.head(self.embed(tokens).mean(dim=1))

    batches = [((torch.randint(0, 3, (6, 9)),), (torch.randint(0, 4, (6,)),)) for _ in range(3)]
    train(Bag(), F.cross_entropy, batches, latch.SGDConfig(lr=0.1, momentum=0.5))


def test_state_dict_round_trip(train):
    model, batches = MLP([8, 16, 2]), regression(4, 8, 2, steps=4)
    update = train(model, F.mse_loss, batches[:2], latch.AdamWConfig(lr=1e-2, eps=1e-3))
    saved = update.state_dict()
    update.stage(*batches[2])
    first = update.step().item()
    update.load_state_dict(saved)
    assert update.step().item() == first
    with pytest.raises(ValueError):
        update.stage((torch.randn(5, 8),), batches[0][1])
