"""Independent FP64 reference for the energy-model recipe (examples/energy.py), with hand-derived gradients.

Ported from the S100L NumPy reference (src/s100l_ebm/reference.py at 0936b034e, released family s100l) to torch
float64 so it can run on CPU or GPU; the equations are unchanged. No autograd, no compiler code and no
examples/energy.py code is used: RNG words come from an independent NumPy Philox (rng_numpy below, ported from
src/s100l_ebm/rng.py) and the Box-Muller transform is evaluated in FP64.

mode 'fp64': exact-real FP64 arithmetic (acceptance reference R).
mode 'p1'  : FP64 emulation of policy s100l-p1 (BF16-rounded product operands and outputs), diagnostic only.
"""
from __future__ import annotations

import numpy as np
import torch

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
MASK = np.uint64(0xFFFFFFFF)
INIT_TAG, FRESH_TAG, BUF_TAG, BUF_UPDATE = 0xFFFFFF01, 0xFFFFFF02, 0xFFFFFF03, 0xFFFFFFFF
KAT = [((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
       ((0xffffffff,) * 4, (0xffffffff, 0xffffffff), (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
       ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
        (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1))]


def philox(c0, c1, c2, c3, k0, k1):
    c = [np.asarray(x, dtype=np.uint64) & MASK for x in (c0, c1, c2, c3)]
    k0, k1 = np.asarray(k0, dtype=np.uint64) & MASK, np.asarray(k1, dtype=np.uint64) & MASK
    for _ in range(10):
        p0, p1 = M0 * c[0], M1 * c[2]
        hi0, lo0, hi1, lo1 = p0 >> np.uint64(32), p0 & MASK, p1 >> np.uint64(32), p1 & MASK
        c = [(hi1 ^ c[1] ^ k0) & MASK, lo1, (hi0 ^ c[3] ^ k1) & MASK, lo0]
        k0, k1 = (k0 + np.uint64(W0)) & MASK, (k1 + np.uint64(W1)) & MASK
    return [x.astype(np.uint32) for x in c]


def kat_ok():
    return all(tuple(int(x) for x in philox(*c, *k)) == o for c, k, o in KAT)


def field_words(seed, c2, c3, rows, cols):
    """[rows, cols] uint32 words of the counter grid (j >> 2, i, c2, c3), word j & 3."""
    k0, k1 = seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF
    i = np.arange(rows, dtype=np.uint64)[:, None]
    j = np.arange(cols, dtype=np.uint64)[None, :]
    w = philox(j >> np.uint64(2), i, np.uint64(c2 & 0xFFFFFFFF), np.uint64(c3 & 0xFFFFFFFF), k0, k1)
    return np.choose(np.broadcast_to((j & np.uint64(3)).astype(np.int64), w[0].shape), w)


def uniform(words):
    return (np.asarray(words, np.uint32) >> np.uint32(8)).astype(np.float64) * 2.0 ** -24


def normal64(seed, c2, c3, rows, cols):
    """FP64 Box-Muller of the exact words (independent of the device libdevice transform)."""
    padded = (cols + 3) // 4 * 4
    w = field_words(seed, c2, c3, rows, padded).reshape(rows, padded // 4, 4)
    a = ((w[..., 0::2] >> np.uint32(8)).astype(np.float64) + 1.0) * 2.0 ** -24
    b = uniform(w[..., 1::2])
    r = np.sqrt(-2.0 * np.log(a))
    n = np.stack((r * np.cos(2.0 * np.pi * b), r * np.sin(2.0 * np.pi * b)), axis=-1)
    return n.reshape(rows, padded)[:, :cols]


def bf16(x):
    """RNE rounding of FP32 values to BF16, returned as float64 (x is any float tensor)."""
    return x.to(torch.float32).to(torch.bfloat16).to(torch.float64)


class Arith:
    def __init__(self, mode):
        assert mode in ('fp64', 'p1')
        self.mode = mode

    def mm(self, a, b):
        if self.mode == 'fp64':
            return a @ b
        return bf16(bf16(a) @ bf16(b))


def sig(z):
    return 1.0 / (1.0 + torch.exp(-z))


def silu(z):
    return z * sig(z)


def dsilu(z):
    s = sig(z)
    return s * (1.0 + z * (1.0 - s))


def forward(P, x, A, L):
    h = [A.mm(x, P['w_in'].T) + P['b_in']]
    us = []
    for l in range(L):
        u = A.mm(silu(h[-1]), P[f'w1_{l}'].T) + P[f'b1_{l}']
        us.append(u)
        h.append(h[-1] + A.mm(silu(u), P[f'w2_{l}'].T) + P[f'b2_{l}'])
    return (silu(h[-1]) * P['w_o']).sum(-1), dict(x=x, h=h, u=us)


def backward(P, cache, seed, A, L, want_params):
    """Gradient of sum_i seed_i E_i: (grad_x, parameter gradients or None)."""
    h, us, x = cache['h'], cache['u'], cache['x']
    seed = seed[:, None]
    g = seed * (P['w_o'] * dsilu(h[-1]))
    G = {}
    if want_params:
        G['w_o'] = (seed * silu(h[-1])).sum(0)
    for l in reversed(range(L)):
        da = A.mm(g, P[f'w2_{l}'])
        du = da * dsilu(us[l])
        if want_params:
            G[f'w2_{l}'] = A.mm(g.T, silu(us[l]))
            G[f'b2_{l}'] = g.sum(0)
            G[f'w1_{l}'] = A.mm(du.T, silu(h[l]))
            G[f'b1_{l}'] = du.sum(0)
        g = g + A.mm(du, P[f'w1_{l}']) * dsilu(h[l])
    if want_params:
        G['w_in'] = A.mm(g.T, x)
        G['b_in'] = g.sum(0)
        return None, G
    return A.mm(g, P['w_in']), None


def input_gradient(P, y, A, L):
    E, cache = forward(P, y, A, L)
    gx, _ = backward(P, cache, torch.ones_like(E), A, L, False)
    return E, gx


def chain_step(P, x, k, t, cfg, A, L, seed, xi=None):
    """One Langevin step from x (FP64 tensor) at chain index k of update t. Returns (y, gx, x_next)."""
    s = cfg['sampler']
    if xi is None:
        xi = torch.as_tensor(normal64(seed, k, t, x.shape[0], x.shape[1]), dtype=torch.float64, device=x.device)
    y = x + s['noise'] * xi
    _, gx = input_gradient(P, y, A, L)
    G = gx.clamp(-s['grad_clip'], s['grad_clip'])
    return y, gx, (y - s['step'] * G).clamp(s['lo'], s['hi'])


def init_negatives(cfg, buf, t, seed):
    """(x0 FP32 tensor, fresh bool array, idx int array) exactly as the recipe (FP32 values)."""
    B, D, R, s = cfg['batch'], cfg['crop'] ** 2, cfg['sampler']['replay'], cfg['sampler']
    w = field_words(seed, INIT_TAG, t, B, 2)
    P = int(round(s['p_fresh'] * 2 ** 24))
    fresh = (w[:, 0] >> np.uint32(8)).astype(np.int64) < P
    idx = ((w[:, 1].astype(np.uint64) * np.uint64(R)) >> np.uint64(32)).astype(np.int64)
    u = uniform(field_words(seed, FRESH_TAG, t, B, D)).astype(np.float32)
    drawn = (np.float32(s['hi'] - s['lo']) * u + np.float32(s['lo'])).astype(np.float32)
    host = buf.detach().cpu().numpy()
    x0 = np.where(fresh[:, None], drawn, host[idx])
    return torch.as_tensor(x0, device=buf.device), fresh, idx


def buffer_init(cfg, seed):
    R, D, s = cfg['sampler']['replay'], cfg['crop'] ** 2, cfg['sampler']
    u = uniform(field_words(seed, BUF_TAG, BUF_UPDATE, R, D)).astype(np.float32)
    return (np.float32(s['hi'] - s['lo']) * u + np.float32(s['lo'])).astype(np.float32)


def loss_and_grads(P, xpos, xneg, alpha, A, L):
    B = xpos.shape[0]
    E, cache = forward(P, torch.cat([xpos, xneg]), A, L)
    Ep, En = E[:B], E[B:]
    loss = Ep.mean() - En.mean() + alpha * ((Ep ** 2).mean() + (En ** 2).mean())
    seed = torch.cat([(1.0 + 2.0 * alpha * Ep) / B, (-1.0 + 2.0 * alpha * En) / B])
    _, G = backward(P, cache, seed, A, L, True)
    return loss, E, G


def adamw(cfg, names, decay, P, G, m, v, step):
    """Global-norm clip + AdamW in FP64 (S100L SPEC section 7 expression). step: int counts before the update."""
    o = cfg['optimizer']
    norm = torch.sqrt(sum((G[n].double() ** 2).sum() for n in names))
    coef = torch.clamp(o['max_norm'] / (norm + 1e-6), max=1.0)
    out = {}
    for n in names:
        wd = o['weight_decay'] if decay[n] else 0.0
        g = G[n].double() * coef
        mm = m[n].double() + (1 - o['beta1']) * (g - m[n].double())
        vv = o['beta2'] * v[n].double() + (1 - o['beta2']) * g * g
        st = step[n] + 1
        bc1, bc2 = 1 - o['beta1'] ** st, 1 - o['beta2'] ** st
        w = P[n].double() * (1 - o['lr'] * wd) - (o['lr'] / bc1) * mm / (torch.sqrt(vv) / bc2 ** 0.5 + o['eps'])
        out[n] = (w, mm, vv)
    return out, norm, coef
