"""Conventional PyTorch controls for the energy-model update (same equations, RNG bits and state as the native).

Adapted from the S100L controls (src/s100l_ebm/conventional.py at 0936b034e) and the S105G whole-update
controls (research/s108/S108H/overlay/s105g_ebm.py at 0936b034e), the strongest measured conventional
implementations of this recipe. Changes: RNG fields come from megakernel_lab.operation_compiler.random.philox_field (Triton on CUDA, bitwise
equal to the generated device code), images are the staged FP32 crop (identical values to the S100L LUT), and
the optimizer is the expression the compiler captures (ops/optim.py: AdamW with global-norm clip, int64 steps,
expm1/log bias corrections), so controls and the native differ only in execution and reduction order.

Variants (every one runs the COMPLETE update: init, K-step chain, positives, gradients, clip+AdamW, commit):
  eager            autograd (torch.autograd.grad for the chain, autograd for the loss), eager ops
  manual           eager ops with the hand-written S100L backward (second independent FP32 implementation)
  pieces_autotune  historical ctl_whole_autotune: compiled unrolled chain / grads / optimizer (manual backward,
                   max-autotune-no-cudagraphs), eager glue
  fused_default    the whole update as one torch.compile(fullgraph) function (default mode)
  fused_autotune   same, mode max-autotune-no-cudagraphs (S105G fused_at)
Each is captured in one CUDA graph for timing (capture_update).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from megakernel_lab.fast_examples.vendor.energy.examples import energy as recipe
from megakernel_lab.operation_compiler.random import philox_field

BF = torch.bfloat16
VARIANTS = ('eager', 'manual', 'pieces_autotune', 'fused_default', 'fused_autotune')


def mm(a, b):
    return (a.to(BF) @ b.to(BF)).float()


def energy_fwd(P, x, L):
    h = [mm(x, P['w_in'].t()) + P['b_in']]
    us = []
    for l in range(L):
        u = mm(F.silu(h[-1]), P[f'w1_{l}'].t()) + P[f'b1_{l}']
        us.append(u)
        h.append(h[-1] + mm(F.silu(u), P[f'w2_{l}'].t()) + P[f'b2_{l}'])
    return (F.silu(h[-1]) * P['w_o']).sum(-1), h, us


def dsilu(z):
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


def manual_backward(P, x, h, us, seed, L, want_params):
    g = seed[:, None] * (P['w_o'] * dsilu(h[-1]))
    G = {}
    if want_params:
        G['w_o'] = (seed[:, None] * F.silu(h[-1])).sum(0)
    for l in reversed(range(L)):
        da = mm(g, P[f'w2_{l}'])
        du = da * dsilu(us[l])
        if want_params:
            G[f'w2_{l}'] = mm(g.t(), F.silu(us[l]))
            G[f'b2_{l}'] = g.sum(0)
            G[f'w1_{l}'] = mm(du.t(), F.silu(h[l]))
            G[f'b1_{l}'] = du.sum(0)
        g = g + mm(du, P[f'w1_{l}']) * dsilu(h[l])
    if want_params:
        G['w_in'] = mm(g.t(), x)
        G['b_in'] = g.sum(0)
        return None, G
    return mm(g, P['w_in']), None


class State:
    """GPU state of one control: parameters, AdamW moments/steps, replay buffer, counters, key, data."""

    def __init__(self, cfg, params, opt, state, images, order):
        self.cfg, self.z = cfg, recipe.dims(cfg)
        self.names = [n for n, _, _ in recipe.inventory(cfg)]
        self.decay = {n: d for n, _, d in recipe.inventory(cfg)}
        self.P = {n: params[n].clone() for n in self.names}
        self.m = {n: opt[f'exp_avg:{n}'].clone() for n in self.names}
        self.v = {n: opt[f'exp_avg_sq:{n}'].clone() for n in self.names}
        self.step = {n: opt[f'step:{n}'].clone() for n in self.names}
        self.buf, self.ptr, self.t, self.key = (state[k].clone() for k in ('buf', 'ptr', 't', 'key'))
        self.images, self.order = images, order
        self.slots = torch.arange(self.z['B'], device=images.device, dtype=torch.int64)

    def snapshot(self):
        return {k: v.clone() for k, v in self.flat().items()}

    def restore(self, snap):
        for k, v in self.flat().items():
            v.copy_(snap[k])

    def flat(self):
        out = {f'p/{n}': self.P[n] for n in self.names}
        out.update({f'm/{n}': self.m[n] for n in self.names})
        out.update({f'v/{n}': self.v[n] for n in self.names})
        out.update({f's/{n}': self.step[n] for n in self.names})
        out.update(buf=self.buf, ptr=self.ptr, t=self.t, key=self.key)
        return out


class Control:
    def __init__(self, cfg, variant):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.cfg, self.variant, self.z = cfg, variant, recipe.dims(cfg)
        s, o = cfg['sampler'], cfg['optimizer']
        self.sig, self.lam, self.c, self.lo, self.hi = s['noise'], s['step'], s['grad_clip'], s['lo'], s['hi']
        self.threshold = int(round(s['p_fresh'] * 2 ** 24)) << 8
        self.alpha, self.o = cfg['alpha'], o
        kw = dict(fullgraph=True, dynamic=False)
        if variant == 'pieces_autotune':
            kw['mode'] = 'max-autotune-no-cudagraphs'
            self._chain = torch.compile(self._chain_all, **kw)
            self._grads = torch.compile(self._param_grads, **kw)
            self._opt = torch.compile(self._optimizer, **kw)
        elif variant.startswith('fused'):
            if variant != 'fused_default':
                kw['mode'] = 'max-autotune-no-cudagraphs'
            self._all = torch.compile(self._update_all, **kw)

    # ---------------------------------------------------------------- pieces
    def _noise(self, key, t, k):
        return philox_field(key, k, t, [self.z['B'], self.z['D']], 'normal')

    def _init(self, key, t, buf):
        B, D, R = self.z['B'], self.z['D'], self.z['R']
        words = philox_field(key, t.new_full((), recipe.INIT_TAG), t, [B, 2], 'bits')
        fresh = words[:, 0] < self.threshold
        index = torch.div(words[:, 1] * R, 2 ** 32, rounding_mode='floor')
        drawn = (self.hi - self.lo) * philox_field(key, t.new_full((), recipe.FRESH_TAG), t, [B, D], 'uniform') + self.lo
        return torch.where(fresh[:, None], drawn, buf[index])

    def _grad_x(self, P, y):
        if self.variant == 'eager':
            y = y.detach().requires_grad_(True)
            with torch.enable_grad():
                E, _, _ = energy_fwd(P, y, self.z['L'])
                (gx,) = torch.autograd.grad(E.sum(), y)
            return gx
        E, h, us = energy_fwd(P, y, self.z['L'])
        gx, _ = manual_backward(P, y, h, us, torch.ones_like(E), self.z['L'], False)
        return gx

    def _chain_step(self, P, x, xi):
        y = x + self.sig * xi
        G = self._grad_x(P, y).clamp(-self.c, self.c)
        return (y - self.lam * G).clamp(self.lo, self.hi)

    def _chain_all(self, Pm, x, key, t, xi_all=None):
        for k in range(self.z['K']):
            xi = xi_all[k] if xi_all is not None else self._noise(key, t, t.new_full((), k))
            x = self._chain_step(Pm, x, xi)
        return x

    def _param_grads(self, P, xpos, xneg):
        B = self.z['B']
        xin = torch.cat([xpos, xneg])
        if self.variant == 'eager':
            Pl = {n: p.detach().requires_grad_(True) for n, p in P.items()}
            with torch.enable_grad():
                E, _, _ = energy_fwd(Pl, xin, self.z['L'])
                Ep, En = E[:B], E[B:]
                loss = Ep.mean() - En.mean() + self.alpha * ((Ep * Ep).mean() + (En * En).mean())
                gs = torch.autograd.grad(loss, [Pl[n] for n in P])
            return loss.detach(), E.detach(), dict(zip(P, gs))
        E, h, us = energy_fwd(P, xin, self.z['L'])
        Ep, En = E[:B], E[B:]
        loss = Ep.mean() - En.mean() + self.alpha * ((Ep * Ep).mean() + (En * En).mean())
        sgn = torch.cat([torch.ones_like(Ep), -torch.ones_like(En)])
        seed = (sgn + (2 * self.alpha) * E) / B
        _, G = manual_backward(P, xin, h, us, seed, self.z['L'], True)
        return loss, E, G

    def _optimizer(self, P, m, v, step, G, names, decay):
        """ops/optim.py AdamW expression with global-norm clipping (identical equations to the capture)."""
        o = self.o
        total = None
        for n in names:
            sq = (G[n] * G[n]).sum()
            total = sq if total is None else total + sq
        denominator = torch.sqrt(total) + 1e-6
        coef = torch.full_like(denominator, o['max_norm']) / denominator
        coef = torch.where(coef != coef, coef, coef.clamp(max=1.0))
        b1, b2 = o['beta1'], o['beta2']
        out = {}
        for n in names:
            g = G[n] * coef
            st = step[n] + 1
            sf = st.to(torch.float32)
            bc1 = -torch.expm1(sf * math.log(b1)) if b1 else sf * 0.0 + 1.0
            bc2 = -torch.expm1(sf * math.log(b2)) if b2 else sf * 0.0 + 1.0
            mm_ = b1 * m[n] + (1.0 - b1) * g
            vv = b2 * v[n] + (1.0 - b2) * (g * g)
            wd = o['weight_decay'] if decay[n] else 0.0
            w = P[n] * (1.0 - o['lr'] * wd) - (o['lr'] / bc1) * mm_ / (torch.sqrt(vv) / torch.sqrt(bc2) + o['eps'])
            out[n] = (w, mm_, vv, st)
        return out

    def _positives(self, S):
        rows = torch.index_select(S.order, 0, S.t.reshape(1)).reshape(self.z['B'])
        return S.images[rows]

    def _commit(self, S, x, out):
        for n in S.names:
            w, mm_, vv, st = out[n]
            S.P[n].copy_(w)
            S.m[n].copy_(mm_)
            S.v[n].copy_(vv)
            S.step[n].copy_(st)
        S.buf.index_copy_(0, torch.remainder(S.ptr + S.slots, self.z['R']), x)
        S.ptr.copy_(torch.remainder(S.ptr + self.z['B'], self.z['R']))
        S.t.add_(1)

    def _update_all(self, P, m, v, step, buf, ptr, t, key, images, order, slots, names, decay):
        x = self._init(key, t, buf)
        x0 = x
        Pm = {n: (P[n].to(BF) if P[n].dim() == 2 else P[n]) for n in names}
        x = self._chain_all(Pm, x, key, t)
        rows = torch.index_select(order, 0, t.reshape(1)).reshape(self.z['B'])
        loss, E, G = self._param_grads(P, images[rows], x)
        out = self._optimizer(P, m, v, step, G, names, decay)
        for n in names:
            w, mm_, vv, st = out[n]
            P[n].copy_(w)
            m[n].copy_(mm_)
            v[n].copy_(vv)
            step[n].copy_(st)
        buf.index_copy_(0, torch.remainder(ptr + slots, self.z['R']), x)
        ptr.copy_(torch.remainder(ptr + self.z['B'], self.z['R']))
        t.add_(1)
        return loss, E, x, x0, G

    # ---------------------------------------------------------------- one complete update (mutates S)
    def run_update(self, S):
        if self.variant.startswith('fused'):
            loss, E, x, x0, G = self._all(S.P, S.m, S.v, S.step, S.buf, S.ptr, S.t, S.key, S.images, S.order,
                                          S.slots, list(S.names), dict(S.decay))
            self.last = dict(loss=loss, E=E, xneg=x, x0=x0, grads=G)
            return self.last
        x0 = self._init(S.key, S.t, S.buf)
        Pm = {n: (S.P[n].to(BF) if S.P[n].dim() == 2 else S.P[n]) for n in S.names}
        if self.variant in ('eager', 'manual'):
            x = x0
            for k in range(self.z['K']):
                x = self._chain_step(Pm, x, self._noise(S.key, S.t, S.t.new_full((), k)))
            loss, E, G = self._param_grads(S.P, self._positives(S), x)
            out = self._optimizer(S.P, S.m, S.v, S.step, G, S.names, S.decay)
        else:
            x = self._chain(Pm, x0, S.key, S.t)
            loss, E, G = self._grads(S.P, self._positives(S), x)
            out = self._opt(S.P, S.m, S.v, S.step, G, list(S.names), dict(S.decay))
        self._commit(S, x, out)
        self.last = dict(loss=loss, E=E, xneg=x, x0=x0, grads=G)
        return self.last


def capture_update(ctrl, S, warmup=2):
    """Warm up on the state, restore it, and capture ONE complete update in a CUDA graph."""
    snap = S.snapshot()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            ctrl.run_update(S)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    S.restore(snap)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ctrl.run_update(S)
    torch.cuda.synchronize()
    S.restore(snap)
    return graph
