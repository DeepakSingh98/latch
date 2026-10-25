"""Gated diagonal selective state-space language model for the common compiler (C-OPS2-C).

Ordinary PyTorch modules for the accepted scan family s99k-gdssm-v1 (S99K SPEC; S108F declarations, P1-A/P4-D
evidence), written with the operation-level ``ops.scan.selective_scan`` primitive. Per ssm block::

    u = rms(x) * g;  z = u . w_in  (columns v | q | c | k | r, widths E, E, E, N, N)
    o = selective_scan(v, c, -exp(a_log), k, r, d_skip, q, dt_bias, reset)   (softplus step, gate q)
    x = x + o . w_out
per ffn block ``x = x + silu(rms(x) * g . w1) . w2``; head ``logits = rms(x) * g_f . emb^T`` (tied); mean byte
cross entropy. Parameter names, shapes, order and initialization are the S108F inventory and s99k-init-v1
(``initialize`` below is a byte copy of the rule in s108f/family.py at P1-A tag p1a-final). The ``bfloat16``
matrix recipe casts matrix operands explicitly, reproducing the conventional control's BF16 autocast rounding
points (BF16 operands and outputs, FP32 accumulation; FP32 residual, norms, scan state, loss and masters).

This module contains no derivative, schedule or generator; capture differentiates it and the compiler lowers the
resulting operation graph. Nothing in the compiler inspects these class or parameter names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from megakernel_lab.operation_compiler.scan import selective_scan

EPS = 1e-5
MATRIX_ROLES = ('emb', 'w_in', 'w_out', 'w1', 'w2')


@dataclass(frozen=True)
class SSMConfig:
    """``blocks``: tuple of ('ssm', E, N) / ('ffn', F). Defaults follow S108F ``make``."""
    d_model: int
    blocks: tuple
    batch: int
    seq_len: int
    vocab: int = 256
    seed: int = 1081
    matrix_dtype: str = 'bfloat16'
    lr: float = 2e-3
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    def validate(self):
        if self.matrix_dtype not in ('bfloat16', 'float32'):
            raise ValueError('explicit BF16 matrix recipe or FP32 only')
        if not any(b[0] == 'ssm' for b in self.blocks):
            raise ValueError('at least one ssm block')


def make(D, E, N, F_, pairs, L, tokens=16384, **kw):
    """S108F ``make``: ``pairs`` x [ssm(E, N), ffn(F)], batch = tokens / L."""
    if tokens % L:
        raise ValueError('tokens % L')
    return SSMConfig(d_model=D, blocks=tuple(b for _ in range(pairs) for b in (('ssm', E, N), ('ffn', F_))),
                     batch=tokens // L, seq_len=L, **kw)


# S108F suite (s108f/suite.py at p1a-final) and P1-A reserved shapes; T* = S108F/S109D check fixtures.
CASES = {
    's5': dict(D=256, E=512, N=16, F_=512, pairs=6, L=2048),
    's5L4': dict(D=256, E=512, N=16, F_=512, pairs=6, L=4096),
    'm30w': dict(D=512, E=1024, N=16, F_=2048, pairs=6, L=2048),
    'm30d': dict(D=384, E=768, N=32, F_=1536, pairs=10, L=2048),
    'l100w': dict(D=1024, E=2048, N=16, F_=4096, pairs=6, L=2048),
    'h30': dict(D=576, E=1152, N=8, F_=2304, pairs=6, L=1024),
    'tiny16': dict(D=64, E=64, N=16, F_=128, pairs=2, L=512, tokens=1024),
    'tiny4': dict(D=128, E=192, N=4, F_=256, pairs=1, L=256, tokens=1024),
    'tiny32': dict(D=64, E=96, N=32, F_=64, pairs=2, L=1024, tokens=2048),
    'tinyT': dict(D=64, E=64, N=8, F_=128, pairs=1, L=264, tokens=264 * 3),
    'tinyC': dict(D=64, E=64, N=8, F_=128, pairs=1, L=520, tokens=1040),
}


def config(case, **kw):
    return make(**{**CASES[case], **kw})


def declaration(cfg):
    """The equivalent S108F declaration (for the conventional control and FP64 reference)."""
    blocks = [dict(kind='ssm', d_inner=b[1], d_state=b[2]) if b[0] == 'ssm' else dict(kind='ffn', d_hidden=b[1],
              activation='silu') for b in cfg.blocks]
    return dict(schema='s108f-decl-v1', family='s99k-gdssm-v1', name='cops2c', vocab=cfg.vocab, d_model=cfg.d_model,
                batch=cfg.batch, seq_len=cfg.seq_len, blocks=blocks, norm_eps=EPS, seed=cfg.seed, head={'tied': True},
                precision='s108f-pbf16-v1', init='s99k-init-v1',
                optimizer=dict(kind='adamw', lr=cfg.lr, beta1=cfg.betas[0], beta2=cfg.betas[1], eps=cfg.eps,
                               weight_decay=cfg.weight_decay),
                clip=dict(max_norm=cfg.max_grad_norm), state=dict(carry_across_updates=False, reset='document_start'),
                execution={'devices': 1})


def inventory(cfg):
    V, D = cfg.vocab, cfg.d_model
    rows = [('emb', 'emb', (V, D))]
    for i, b in enumerate(cfg.blocks):
        p = f'blocks.{i}.'
        if b[0] == 'ssm':
            E, N = b[1], b[2]
            rows += [(p + 'norm', 'norm', (D,)), (p + 'w_in', 'w_in', (D, 3 * E + 2 * N)), (p + 'a_log', 'a_log', (E, N)),
                     (p + 'dt_bias', 'dt_bias', (E,)), (p + 'd_skip', 'd_skip', (E,)), (p + 'w_out', 'w_out', (E, D))]
        else:
            rows += [(p + 'norm', 'norm', (D,)), (p + 'w1', 'w1', (D, b[1])), (p + 'w2', 'w2', (b[1], D))]
    rows.append(('final_norm', 'norm', (D,)))
    return rows


def _softplus_inv(y):
    return y + np.log(-np.expm1(-y))


def initialize(cfg, seed=None):
    """s99k-init-v1 FP32 masters in inventory order (rule copied from s108f/family.py initialize)."""
    rng = np.random.Generator(np.random.PCG64(cfg.seed if seed is None else seed))
    nb = len(cfg.blocks)
    out = {}
    for name, role, shape in inventory(cfg):
        if role == 'emb':
            w = rng.normal(0.0, 0.02, shape)
        elif role == 'norm':
            w = np.ones(shape)
        elif role in ('w_in', 'w1'):
            w = rng.normal(0.0, 1.0 / math.sqrt(shape[0]), shape)
        elif role == 'a_log':
            w = np.broadcast_to(np.log(np.arange(1, shape[1] + 1, dtype=np.float64)), shape).copy()
        elif role == 'dt_bias':
            dt = np.exp(rng.uniform(math.log(1e-3), math.log(1e-1), shape))
            w = _softplus_inv(dt)
        elif role == 'd_skip':
            w = np.ones(shape)
        elif role in ('w_out', 'w2'):
            w = rng.normal(0.0, 1.0 / math.sqrt(2 * nb * shape[0]), shape)
        else:
            raise AssertionError(role)
        out[name] = np.ascontiguousarray(w, dtype=np.float32)
    return out


def rms(x, g, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


class SSMBlock(nn.Module):
    def __init__(self, D, E, N, matrix_dtype):
        super().__init__()
        self.E, self.N, self.matrix_dtype = E, N, matrix_dtype
        self.norm = nn.Parameter(torch.ones(D))
        self.w_in = nn.Parameter(torch.zeros(D, 3 * E + 2 * N))
        self.a_log = nn.Parameter(torch.zeros(E, N))
        self.dt_bias = nn.Parameter(torch.zeros(E))
        self.d_skip = nn.Parameter(torch.ones(E))
        self.w_out = nn.Parameter(torch.zeros(E, D))

    def forward(self, x, reset=None):
        E, N = self.E, self.N
        u = rms(x, self.norm)
        z = _mm(u, self.w_in, self.matrix_dtype)
        v, q, c = z[..., :E], z[..., E:2 * E], z[..., 2 * E:3 * E]
        k, r = z[..., 3 * E:3 * E + N], z[..., 3 * E + N:]
        o = selective_scan(v, c, -torch.exp(self.a_log), k, r, self.d_skip, q, self.dt_bias, reset=reset)
        return x + _mm(o, self.w_out, self.matrix_dtype)


class FFNBlock(nn.Module):
    def __init__(self, D, F_, matrix_dtype):
        super().__init__()
        self.matrix_dtype = matrix_dtype
        self.norm = nn.Parameter(torch.ones(D))
        self.w1 = nn.Parameter(torch.zeros(D, F_))
        self.w2 = nn.Parameter(torch.zeros(F_, D))

    def forward(self, x, reset=None):
        u = rms(x, self.norm)
        return x + _mm(F.silu(_mm(u, self.w1, self.matrix_dtype)), self.w2, self.matrix_dtype)


def _mm(a, b, matrix_dtype):
    if matrix_dtype == 'bfloat16':
        return a.to(torch.bfloat16) @ b.to(torch.bfloat16)
    return a @ b


class GatedSSMLM(nn.Module):
    def __init__(self, cfg: SSMConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.emb = nn.Parameter(torch.zeros(cfg.vocab, cfg.d_model))
        self.blocks = nn.ModuleList(SSMBlock(cfg.d_model, b[1], b[2], cfg.matrix_dtype) if b[0] == 'ssm'
                                    else FFNBlock(cfg.d_model, b[1], cfg.matrix_dtype) for b in cfg.blocks)
        self.final_norm = nn.Parameter(torch.ones(cfg.d_model))

    def forward(self, tok, reset=None):
        x = F.embedding(tok, self.emb)
        for block in self.blocks:
            x = block(x, reset)
        return _mm(rms(x, self.final_norm), self.emb.t(), self.cfg.matrix_dtype)

    @torch.no_grad()
    def load_numpy(self, params):
        named = dict(self.named_parameters())
        if set(named) != set(params):
            raise ValueError('parameter inventory differs')
        for name, value in params.items():
            named[name].copy_(torch.as_tensor(np.asarray(value, np.float32)))


def build(cfg, seed=None):
    model = GatedSSMLM(cfg)
    model.load_numpy(initialize(cfg, seed))
    return model


def loss_fn(logits, tgt):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))


def optimizer_config(cfg, *, runtime_lr=True):
    from megakernel_lab.operation_compiler.optim import AdamWConfig
    no_decay = {name: {'weight_decay': 0.0} for name, role, _ in inventory(cfg) if role not in MATRIX_ROLES}
    return AdamWConfig(lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay,
                       param_options=no_decay, max_grad_norm=cfg.max_grad_norm, runtime_lr=runtime_lr)
