"""Energy-based model training by short-run Langevin sampling with a replay buffer, in ordinary PyTorch.

This re-expresses the S100L/S101E/S103H/S108H recipe (research/s100/S100L/SPEC.md: model `s100l-rmlp-ebm-v1`,
precision `s100l-p1`, RNG `s100l-rng-v1`, sampler `s100l-cd-v1`, init `s100l-init-v1`, data `s100l-fmnist-v1`)
as three plain functions for ops.sampling.capture_sampled_training. Nothing here schedules device work; the
compiler never sees these names.

Model (residual SiLU MLP, energy head without bias):
    h_0 = x W_in^T + b_in;  u_l = silu(h_{l-1}) W1_l^T + b1_l;  h_l = h_{l-1} + silu(u_l) W2_l^T + b2_l
    E(x) = sum_h silu(h_L)_h w_o_h
Every matrix product has BF16 operands, FP32 accumulation and a BF16-rounded output used as FP32
(`(a.bfloat16() @ w.bfloat16().T).float()`); autograd of that expression gives the policy's backward products.

One update:
    init     negatives: replay rows buf[idx_i] or, with probability p_fresh, uniform fresh rows (INIT/U_FRESH fields)
    chain    K steps, weights fixed at their pre-update values, input gradients only:
             y = x + sigma xi_k;  G = clamp(grad_x sum_i E(y_i), -c, c);  x = clamp(y - lambda G, lo, hi)
    loss     mean E(x+) - mean E(x-) + alpha (mean E(x+)^2 + mean E(x-)^2), x- = chain output (a constant:
             no derivative flows through sampling); parameter gradients of one forward over [x+; x-]
    update   global-norm clip (max_norm) + AdamW (beta1 = 0), decay on matrices only
    commit   buf[(ptr + i) mod R] = x-_i, ptr += B (mod R), t += 1 (after every read of buf)

Known differences from the S100L device programs, all deliberate and recorded in the C-OPS4-B handoff:
the Langevin and fresh-row arithmetic are written x + sigma*xi etc. (two FP32 roundings; the generated code
does not contract to FMA), the AdamW bias corrections use the compiler's int64 step and expm1/log form, and
reduction orders are the compiler's. Equations, RNG words and derivative boundaries are the recipe's.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.random import make_key, philox_field

BF = torch.bfloat16
INIT_TAG, FRESH_TAG, BUF_TAG, BUF_UPDATE = 0xFFFFFF01, 0xFFFFFF02, 0xFFFFFF03, 0xFFFFFFFF

SAMPLER = dict(step=10.0, noise=0.005, grad_clip=0.01, lo=0.0, hi=1.0, p_fresh=0.05, replay=10000)
OPTIMIZER = dict(lr=1e-4, beta1=0.0, beta2=0.999, eps=1e-8, weight_decay=0.01, max_norm=1.0)
BASE = dict(crop=28, alpha=1.0, seed=20260926, data_seed=1003, sampler=SAMPLER, optimizer=OPTIMIZER)

# hidden H, inner F, blocks L, batch B, chain K (S101E/S108H declaration names in comments).
CONFIGS = {
    'tiny': dict(BASE, crop=8, hidden=32, inner=64, blocks=2, batch=8, chain=3,
                 sampler=dict(SAMPLER, replay=40)),
    'small': dict(BASE, crop=16, hidden=64, inner=128, blocks=2, batch=32, chain=4,
                  sampler=dict(SAMPLER, replay=256)),
    'wb': dict(BASE, hidden=384, inner=768, blocks=4, batch=256, chain=64),      # S101E/S103H/S108H WB anchor
    'sd': dict(BASE, hidden=384, inner=768, blocks=8, batch=256, chain=32),      # S108H SD: depth 8, chain 32
}


def dims(cfg):
    return dict(D=cfg['crop'] ** 2, H=cfg['hidden'], F=cfg['inner'], L=cfg['blocks'], B=cfg['batch'],
                K=cfg['chain'], R=cfg['sampler']['replay'])


def inventory(cfg):
    """(name, shape, decay) in the S100L parameter inventory order."""
    z = dims(cfg)
    rows = [('w_in', (z['H'], z['D']), True), ('b_in', (z['H'],), False)]
    for l in range(z['L']):
        rows += [(f'w1_{l}', (z['F'], z['H']), True), (f'b1_{l}', (z['F'],), False),
                 (f'w2_{l}', (z['H'], z['F']), True), (f'b2_{l}', (z['H'],), False)]
    return rows + [('w_o', (z['H'],), False)]


def init_parameters(cfg):
    """`s100l-init-v1`: one PCG64(seed) generator in inventory order, float64 draws cast to FP32."""
    z = dims(cfg)
    gen = np.random.Generator(np.random.PCG64(cfg['seed']))
    std = {'w_in': 1 / z['D'], 'w1': 1 / z['H'], 'w2': 1 / (z['L'] * z['F']), 'w_o': 1 / z['H']}
    params = {}
    for name, shape, _ in inventory(cfg):
        kind = name.split('_')[0] if name != 'w_in' else 'w_in'
        kind = 'w_o' if name == 'w_o' else kind
        if name.startswith('b'):
            params[name] = torch.zeros(shape, dtype=torch.float32)
        else:
            params[name] = torch.from_numpy(gen.normal(0.0, math.sqrt(std[kind]), size=shape).astype(np.float32))
    return params


def optimizer_config(cfg):
    from megakernel_lab.operation_compiler.optim import AdamWConfig
    o = cfg['optimizer']
    return AdamWConfig(lr=o['lr'], betas=(o['beta1'], o['beta2']), eps=o['eps'], weight_decay=o['weight_decay'],
                       max_grad_norm=o['max_norm'],
                       param_options={name: {'weight_decay': 0.0} for name, _, decay in inventory(cfg) if not decay})


def initial_state(cfg, *, device='cpu'):
    """Replay buffer `buf[r, j] = lo + (hi - lo) U_BUF(r, j)`, counters t = ptr = 0 and the RNG key."""
    z, s = dims(cfg), cfg['sampler']
    key = make_key(cfg['seed'], device=device)
    u = philox_field(key, torch.tensor(BUF_TAG, device=device), torch.tensor(BUF_UPDATE, device=device),
                     [z['R'], z['D']], 'uniform')
    buf = (s['hi'] - s['lo']) * u + s['lo']
    return {'buf': buf.contiguous(), 'ptr': torch.zeros((), dtype=torch.int64, device=device),
            't': torch.zeros((), dtype=torch.int64, device=device), 'key': key}


def linear(a, w, b=None):
    """Policy s100l-p1 product: BF16 operands, FP32 accumulation, BF16-rounded output used as FP32."""
    y = (a.to(BF) @ w.to(BF).t()).float()
    return y if b is None else y + b


def energy(p, x, L):
    h = linear(x, p['w_in'], p['b_in'])
    for l in range(L):
        u = linear(F.silu(h), p[f'w1_{l}'], p[f'b1_{l}'])
        h = h + linear(F.silu(u), p[f'w2_{l}']) + p[f'b2_{l}']
    return (F.silu(h) * p['w_o']).sum(-1)


def make_functions(cfg):
    """(init, step, loss) for capture_sampled_training. Inputs: (images [N, D] FP32, order [T, B] int64)."""
    z, s = dims(cfg), cfg['sampler']
    B, D, R, L = z['B'], z['D'], z['R'], z['L']
    sigma, lam, c, lo, hi = s['noise'], s['step'], s['grad_clip'], s['lo'], s['hi']
    threshold = int(round(s['p_fresh'] * 2 ** 24)) << 8           # (w0 >> 8) < P  <=>  w0 < P * 2**8
    alpha = cfg['alpha']

    def init(params, state, images, order):
        t, key = state['t'], state['key']
        words = philox_field(key, t.new_full((), INIT_TAG), t, [B, 2], 'bits')
        fresh = words[:, 0] < threshold
        index = torch.div(words[:, 1] * R, 2 ** 32, rounding_mode='floor')
        drawn = (hi - lo) * philox_field(key, t.new_full((), FRESH_TAG), t, [B, D], 'uniform') + lo
        return {'x': torch.where(fresh[:, None], drawn, F.embedding(index, state['buf']))}

    def step(params, carry, state, k, images, order):
        xi = philox_field(state['key'], k, state['t'], [B, D], 'normal')
        y = carry['x'] + sigma * xi
        grad = torch.func.grad(lambda v: energy(params, v, L).sum())(y)
        x = (y - lam * grad.clamp(-c, c)).clamp(lo, hi)
        return {'x': x}, state

    def loss(params, carry, state, images, order):
        t, ptr = state['t'], state['ptr']
        positives = F.embedding(torch.index_select(order, 0, t.reshape(1)).reshape(B), images)
        negatives = carry['x']
        e = energy(params, torch.cat([positives, negatives]), L)
        ep, en = e[:B], e[B:]
        value = ep.mean() - en.mean() + alpha * ((ep * ep).mean() + (en * en).mean())
        slots = torch.remainder(ptr + torch.arange(B, dtype=torch.int64, device=ptr.device), R)
        new_state = {'buf': state['buf'].index_copy(0, slots, negatives),
                     'ptr': torch.remainder(ptr + B, R), 't': t + 1, 'key': state['key']}
        return value, new_state

    return init, step, loss
