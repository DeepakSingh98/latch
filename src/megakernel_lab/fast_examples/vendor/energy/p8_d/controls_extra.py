"""Controls added by P8-D before any panel-engine timing (declared in p8_d/POLICY.json).

fused_autotune_cd   c_ops4_b fused_autotune with Inductor coordinate_descent_tuning (same equations/policy).
bf16_fused_autotune practical BF16: network activations, products and their gradients stay BF16 (no FP32
                    round trip between products); sampler state x/y, energies, loss, parameter master copies and
                    AdamW stay FP32. Whole update torch.compile(max-autotune-no-cudagraphs). Changed arithmetic:
                    judged as a denominator by the 400-update learning gate only (POLICY.json).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F

from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import controls as base

BF = torch.bfloat16
EXTRA = ('fused_autotune_cd', 'bf16_fused_autotune')


def _mmb(a, b):
    return a.to(BF) @ b.to(BF)


def _dsilu(z):
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


def energy_fwd_bf16(P, x, L):
    h = [_mmb(x, P['w_in'].t()) + P['b_in'].to(BF)]
    us = []
    for l in range(L):
        u = _mmb(F.silu(h[-1]), P[f'w1_{l}'].t()) + P[f'b1_{l}'].to(BF)
        us.append(u)
        h.append(h[-1] + _mmb(F.silu(u), P[f'w2_{l}'].t()) + P[f'b2_{l}'].to(BF))
    return (F.silu(h[-1]).float() * P['w_o']).sum(-1), h, us


def manual_backward_bf16(P, x, h, us, seed, L, want_params):
    g = (seed[:, None] * (P['w_o'] * _dsilu(h[-1].float()))).to(BF)
    G = {}
    if want_params:
        G['w_o'] = (seed[:, None] * F.silu(h[-1].float())).sum(0)
    for l in reversed(range(L)):
        da = _mmb(g, P[f'w2_{l}'])
        du = da * _dsilu(us[l])
        if want_params:
            G[f'w2_{l}'] = _mmb(g.t(), F.silu(us[l])).float()
            G[f'b2_{l}'] = g.float().sum(0)
            G[f'w1_{l}'] = _mmb(du.t(), F.silu(h[l])).float()
            G[f'b1_{l}'] = du.float().sum(0)
        g = g + _mmb(du, P[f'w1_{l}']) * _dsilu(h[l])
    if want_params:
        G['w_in'] = _mmb(g.t(), x).float()
        G['b_in'] = g.float().sum(0)
        return None, G
    return _mmb(g, P['w_in']).float(), None


class ExtraControl(base.Control):
    def __init__(self, cfg, variant):
        if variant not in EXTRA:
            raise ValueError(variant)
        super().__init__(cfg, 'fused_autotune')
        self.variant_name = variant
        self.bf16 = variant == 'bf16_fused_autotune'
        self.cd = variant == 'fused_autotune_cd'

    def _grad_x(self, P, y):
        if not self.bf16:
            return super()._grad_x(P, y)
        E, h, us = energy_fwd_bf16(P, y, self.z['L'])
        gx, _ = manual_backward_bf16(P, y, h, us, torch.ones_like(E), self.z['L'], False)
        return gx

    def _param_grads(self, P, xpos, xneg):
        if not self.bf16:
            return super()._param_grads(P, xpos, xneg)
        B = self.z['B']
        xin = torch.cat([xpos, xneg])
        E, h, us = energy_fwd_bf16(P, xin, self.z['L'])
        Ep, En = E[:B], E[B:]
        loss = Ep.mean() - En.mean() + self.alpha * ((Ep * Ep).mean() + (En * En).mean())
        sgn = torch.cat([torch.ones_like(Ep), -torch.ones_like(En)])
        seed = (sgn + (2 * self.alpha) * E) / B
        _, G = manual_backward_bf16(P, xin, h, us, seed, self.z['L'], True)
        return loss, E, G

    def run_update(self, S):
        context = contextlib.nullcontext()
        if self.cd:
            import torch._inductor.config as inductor_config
            context = inductor_config.patch(coordinate_descent_tuning=True)
        with context:
            return super().run_update(S)


def make_control(cfg, variant):
    return ExtraControl(cfg, variant) if variant in EXTRA else base.Control(cfg, variant)
