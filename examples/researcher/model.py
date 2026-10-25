"""One model, one loss and one update rule, shared by every execution path.

This file is the researcher's editing surface. ``ByteLM`` and ``objective`` run
unchanged in eager PyTorch, under ``torch.compile`` and through Latch's
``capture_module``; nothing here imports the compiler. Three supported learning
computations are expressed by one ``Config``:

* ordinary training: ``recurrence=1``, ``exit_after=None``;
* intermediate supervision: ``exit_after=k`` adds a second readout through the
  same tied head after the k-th block application, weighted by ``aux_weight``;
  the weight is a per-update runtime scalar, the position is a program choice;
* shared-weight recurrence: ``recurrence=R`` applies the same block stack R times
  (a static Python loop, specialized at capture).

Arithmetic is spelled explicitly because Latch has no autocast: FP32 master
parameters, residual stream, RMSNorm, GELU and cross entropy; BF16 operands and
BF16 results (FP32 accumulation) for every matrix product and for attention,
followed by ``.float()`` where the FP32 stream continues. Changing a shape, the
exit position or the recurrence count produces a different program; changing the
auxiliary weight, learning rate schedule data or minibatch does not.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

BF16 = torch.bfloat16
NORM_EPS = 1e-6


@dataclass(frozen=True)
class Config:
    name: str = 'ordinary'
    vocab: int = 256            # byte-level language model
    d_model: int = 256
    n_heads: int = 4            # head width 64 is an admitted fast attention form
    n_layers: int = 2
    d_ff: int = 1024
    seq_len: int = 128
    batch: int = 4
    recurrence: int = 1         # passes over the shared block stack (static)
    exit_after: int | None = None   # auxiliary readout after this many block applications (1 .. recurrence*n_layers-1)
    aux_weight: float = 0.0     # runtime scalar staged with every minibatch; no recompilation
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    clip: float | None = 1.0    # global gradient-norm clip before AdamW (None disables)
    seed: int = 1729

    @property
    def block_applications(self):
        return self.recurrence * self.n_layers

    @property
    def tokens_per_update(self):
        return self.batch * self.seq_len

    def check(self):
        if min(self.vocab, self.d_model, self.n_heads, self.n_layers, self.d_ff, self.seq_len, self.batch, self.recurrence) < 1:
            raise ValueError('all dimensions and the recurrence count must be positive')
        if self.d_model % self.n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        if self.exit_after is not None and not 1 <= self.exit_after < self.block_applications:
            raise ValueError(f'exit_after must lie in 1..{self.block_applications - 1} (block applications before the final readout)')
        if self.exit_after is None and self.aux_weight:
            raise ValueError('aux_weight needs an exit_after position')
        if self.aux_weight < 0 or self.lr < 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError('aux_weight, lr and weight_decay must be nonnegative; eps positive')
        if self.clip is not None and self.clip <= 0:
            raise ValueError('clip must be positive or None')
        return self

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError('unknown config keys: ' + ', '.join(sorted(unknown)))
        return cls(**values).check()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def edited(self, assignments):
        """Apply ``key=value`` edits (strings) with the field's type; ``none`` clears an optional field."""
        fields = {f.name: f for f in dataclasses.fields(self)}
        values = self.to_dict()
        for item in assignments:
            key, _, raw = item.partition('=')
            if key not in fields or not _:
                raise ValueError(f'edit must be key=value with a known key, got {item!r}')
            if raw.lower() == 'none':
                values[key] = None
                continue
            current = values[key]
            kind = type(current) if current is not None else (int if key == 'exit_after' else float)
            values[key] = raw if kind is str else (kind(raw) if kind is not bool else raw.lower() in ('1', 'true'))
        return Config.from_dict(values)


def bf16_matmul(x, weight):
    """BF16 operands, FP32 accumulation, BF16 result: ``x @ weight.T`` for a [out, in] weight."""
    return F.linear(x.to(BF16), weight.to(BF16))


class Block(nn.Module):
    """Pre-norm attention + GELU feed-forward block with explicit precision."""

    def __init__(self, cfg, generator):
        super().__init__()
        d, f = cfg.d_model, cfg.d_ff
        self.cfg = cfg
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.qkv = nn.Parameter(torch.randn(3 * d, d, generator=generator) * d ** -0.5)
        self.out = nn.Parameter(torch.randn(d, d, generator=generator) * (2 * d) ** -0.5)
        self.ff_norm = nn.Parameter(torch.ones(d))
        self.w1 = nn.Parameter(torch.randn(f, d, generator=generator) * d ** -0.5)
        self.w2 = nn.Parameter(torch.randn(d, f, generator=generator) * (2 * f) ** -0.5)

    def forward(self, h):
        b, t, d = h.shape
        heads, width = self.cfg.n_heads, d // self.cfg.n_heads
        x = F.rms_norm(h, (d,), self.attn_norm, eps=NORM_EPS)
        q, k, v = (part.view(b, t, heads, width).transpose(1, 2)
                   for part in bf16_matmul(x.reshape(b * t, d), self.qkv).split(d, dim=-1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # BF16 in, BF16 out
        h = h + bf16_matmul(y.transpose(1, 2).reshape(b * t, d), self.out).float().reshape(b, t, d)
        x = F.rms_norm(h, (d,), self.ff_norm, eps=NORM_EPS)
        u = F.gelu(bf16_matmul(x.reshape(b * t, d), self.w1).float())        # FP32 activation
        return h + bf16_matmul(u, self.w2).float().reshape(b, t, d)


class ByteLM(nn.Module):
    """Byte language model with a tied readout; returns ``(logits, *intermediate_logits)``."""

    def __init__(self, cfg: Config):
        super().__init__()
        cfg.check()
        self.cfg = cfg
        generator = torch.Generator().manual_seed(cfg.seed)
        d = cfg.d_model
        self.embed = nn.Parameter(torch.randn(cfg.vocab, d, generator=generator) * 0.02)   # also the readout weight
        self.position = nn.Parameter(torch.randn(cfg.seq_len, d, generator=generator) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg, generator) for _ in range(cfg.n_layers))
        self.final_norm = nn.Parameter(torch.ones(d))

    def readout(self, h):
        b, t, d = h.shape
        z = F.rms_norm(h, (d,), self.final_norm, eps=NORM_EPS)
        return bf16_matmul(z.reshape(b * t, d), self.embed).float()           # [B*T, vocab] FP32 view of BF16 products

    def forward(self, tokens):
        h = F.embedding(tokens, self.embed) + self.position
        exits = []
        applied = 0
        for _ in range(self.cfg.recurrence):           # shared weights across passes; unrolled at capture
            for block in self.blocks:
                h = block(h)
                applied += 1
                if applied == self.cfg.exit_after:
                    exits.append(self.readout(h))
        return (self.readout(h), *exits)


def objective(outputs, labels, aux_weight):
    """Mean next-byte cross entropy, plus ``aux_weight`` times the cross entropy of each intermediate readout.

    ``aux_weight`` is a 0-d FP32 tensor supplied with every minibatch (a runtime input to the generated program).
    """
    logits, *exits = outputs
    labels = labels.reshape(-1)
    loss = F.cross_entropy(logits, labels)
    for intermediate in exits:
        loss = loss + aux_weight * F.cross_entropy(intermediate, labels)
    return loss


def adamw_hyperparameters(cfg: Config):
    """The one AdamW rule both paths implement: decoupled decay, bias correction, eps after the square root."""
    return dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)


def synthetic_batch(cfg: Config, step: int):
    """Deterministic changing minibatch: next-byte targets of a seeded random byte sequence."""
    generator = torch.Generator().manual_seed(cfg.seed * 1000003 + step)
    sequence = torch.randint(cfg.vocab, (cfg.batch, cfg.seq_len + 1), generator=generator)
    return sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()


def example_inputs(cfg: Config):
    """Capture-time example tensors: ``(inputs, targets)`` with the auxiliary weight as the last target."""
    tokens, labels = synthetic_batch(cfg, 0)
    return (tokens,), (labels, torch.tensor(cfg.aux_weight, dtype=torch.float32))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
