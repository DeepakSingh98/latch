"""LATCH-FFN-RETAIN: small extracted forward/backward/update reproducers of the ordinary model's dominant calculations.

Every program uses the primary's spelling (model.py: explicit BF16 projection operands/results, FP32 LayerNorm and
residual), its token count (32 x 512 = 16384 rows) and width (640), and the primary's AdamW (betas 0.9/0.999,
eps 1e-8, wd 0.01, global clip 1.0 unless named otherwise). Loss: mean squared error to a fixed target (no head).

  ln        affine LayerNorm only (LATCH-FGM-COMPOSE repro_rows; its optimizer had no clip)
  ln-clip   ln with the primary's global clip
  ffn       x + down(relu(up(ff_norm(x)))) (primary FFN sublayer)
  attn      x + out(attention(attn_norm(x))) (primary attention sublayer: packed QKV, long BF16 boundary body)
  block     one full primary block (attn sublayer then FFN sublayer)

Run with python -m megakernel_lab.fast_examples prepare/compare/verify ffn.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

from megakernel_lab.fast_examples.recovered_release.ffn import model as fm, reference

ROWS, BATCH, SEQ, WIDTH = 16384, 32, 512, 640
# LATCH-FFN-RETAIN: "ffn" is the frozen primary. The two reserved FFN programs were declared in DECLARED.json
# before any compiler change: a nearby shape (D768, d_ff 3072, B16) and a nearby composition (D512, d_ff 2048,
# GELU-tanh activation instead of ReLU). Same spelling (explicit BF16 projections with bias, FP32 LayerNorm/residual).
SHAPES = {"ffn-d768": dict(batch=16, seq=512, width=768, d_ff=3072, activation="relu"),
          "ffn-gelu": dict(batch=32, seq=512, width=512, d_ff=2048, activation="gelu_tanh")}
PROGRAMS = ("ln", "ln-clip", "ffn", "attn", "block") + tuple(SHAPES)


def shape(program):
    return SHAPES.get(program, dict(batch=BATCH, seq=SEQ, width=WIDTH, d_ff=4 * WIDTH, activation="relu"))


def config(program="ffn"):
    s = shape(program)
    return fm.make_config("ordinary", n_layers=1, d_model=s["width"], d_ff=s["d_ff"], seq_len=s["seq"],
                          routing_group=s["batch"], n_heads=s["width"] // 64)


def ffn(ff, x, activation):
    b, t, d = x.shape
    flat = x.reshape(b * t, d)
    up = fm.bf16_linear(flat, ff.up.weight, ff.up.bias)
    up = F.gelu(up, approximate="tanh") if activation == "gelu_tanh" else F.relu(up)
    return fm.bf16_linear(up, ff.down.weight, ff.down.bias).reshape(b, t, d)


class Program(torch.nn.Module):
    """One primary block's parameters (reference.Block names/initialization); forward per program."""

    def __init__(self, program):
        super().__init__()
        self.program = program
        cfg = config(program)
        self.activation = shape(program)["activation"]
        torch.manual_seed(1729)
        self.block = reference.Block(cfg, 0)
        self.register_buffer("causal_mask", torch.triu(torch.full((cfg.seq_len, cfg.seq_len), -10000.0),
                                                       diagonal=1), persistent=False)
        if program.startswith("ln"):      # parameters of the LayerNorm only
            self.block.attn = None
            self.block.ff = None
            self.block.ff_norm = None
        elif program.startswith("ffn"):
            self.block.attn = None
            self.block.attn_norm = None
        elif program == "attn":
            self.block.ff = None
            self.block.ff_norm = None

    def forward(self, x):
        b = self.block
        if self.program.startswith("ln"):
            return fm.layer_norm(b.attn_norm, x)
        if self.program in ("attn", "block"):
            x = x + fm.attention(b.attn, fm.layer_norm(b.attn_norm, x), self.causal_mask)
        if self.program.startswith("ffn") or self.program == "block":
            x = x + ffn(b.ff, fm.layer_norm(b.ff_norm, x), self.activation)
        return x


def mse(y, target):
    return ((y - target) ** 2).mean()


def adamw(program):
    from megakernel_lab.operation_compiler import AdamWConfig
    clip = None if program == "ln" else 1.0
    return AdamWConfig(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, max_grad_norm=clip)


def inputs(device="cpu", program="ffn"):
    s = shape(program)
    g = torch.Generator(device="cpu").manual_seed(5)
    x = torch.randn(s["batch"], s["seq"], s["width"], generator=g)
    target = torch.randn(s["batch"], s["seq"], s["width"], generator=g)
    return x.to(device), target.to(device)

