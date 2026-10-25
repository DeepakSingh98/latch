"""Plain AdamW + global clip as a captured tensor-update rule with staged hyper-parameters.

Same arithmetic as megakernel_lab.tinymoe.optimizer.make_rule (unchanged recipe,
no EMA), except that the per-run values which differ between learning-rate trials
and batch sizes are staged 0-d scalars instead of capture-time constants:

  schedule_learning_rate   LR for this update (host token schedule)
  schedule_weight_decay    decoupled decay coefficient (protocol.weight_decay)
  schedule_beta2           AdamW beta2 (protocol.beta2_for)
  schedule_one_minus_beta2 1 - beta2, rounded separately from beta2 on the host
  schedule_log_beta2       log(beta2) for the per-parameter bias correction

beta1 = 0.9, eps = 1e-6 and clip = 1.0 are common constants of both arms
(protocol.COMMON). One captured program therefore serves every LR/beta2 trial
of one shape. Experts with no routed token keep HF absent-gradient behaviour
(parameter, moments and counter unchanged), exactly as the donor rule.
"""
from __future__ import annotations

import math

import torch

from megakernel_lab.tinymoe.optimizer import CLIP, EXPERT, LR, NORM

from .protocol import COMMON

WD = "schedule_weight_decay"
BETA2 = "schedule_beta2"
OMB2 = "schedule_one_minus_beta2"
LOGB2 = "schedule_log_beta2"
STAGED = (LR, WD, BETA2, OMB2, LOGB2)


def make_rule(model):
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    beta1, eps, clip_norm = COMMON["beta1"], COMMON["eps"], COMMON["clip_norm"]
    roles = {name: EXPERT.match(name) for name, _ in model.named_parameters()}
    roles = {name: (int(m[1]), int(m[2])) if m else None for name, m in roles.items()}
    decay = {name: p.ndim >= 2 for name, p in model.named_parameters()}

    def initialize(parameters, _hp):
        first = next(iter(parameters.values()))
        state = {LR: first.new_zeros(()), WD: first.new_zeros(()), BETA2: first.new_zeros(()),
                 OMB2: first.new_zeros(()), LOGB2: first.new_zeros(()),
                 NORM: first.new_zeros(()), CLIP: first.new_ones(())}
        for name, p in parameters.items():
            state["exp_avg:" + name] = torch.zeros_like(p)
            state["exp_avg_sq:" + name] = torch.zeros_like(p)
            state["step:" + name] = torch.zeros((), dtype=torch.int64, device=p.device)
        return state

    def update(parameters, gradients, state, _hp, forward_state):
        squared = None
        active = {}
        for name, g in gradients.items():
            role = roles[name]
            if role is not None:
                layer, expert = role
                active[name] = forward_state[f"active.{layer}"][expert] > 0
                g = torch.where(active[name], g, torch.zeros_like(g))
            term = (g * g).sum()
            squared = term if squared is None else squared + term
        norm = squared.sqrt()
        coef = torch.clamp(clip_norm / (norm + 1e-6), max=1.0)
        new = {name: state[name] for name in STAGED}
        new[NORM], new[CLIP] = norm, coef
        out = {}
        rate, wd, beta2, omb2, logb2 = (state[name] for name in STAGED)
        for name, p in parameters.items():
            g = gradients[name] * coef
            m0, v0, t0 = (state[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
            t = t0 + 1
            sf = t.to(torch.float32)
            bc1 = -torch.expm1(sf * math.log(beta1))
            bc2 = -torch.expm1(sf * logb2)
            m = torch.lerp(m0, g, 1.0 - beta1)
            v = beta2 * v0 + omb2 * g.square()
            decayed = p * (1.0 - rate * wd) if decay[name] else p
            updated = decayed - (rate / bc1) * m / (v.sqrt() / bc2.sqrt() + eps)
            if name in active:
                flag = active[name]
                updated = torch.where(flag, updated, p)
                m, v, t = (torch.where(flag, a, b) for a, b in ((m, m0), (v, v0), (t, t0)))
            out[name] = updated
            new["exp_avg:" + name], new["exp_avg_sq:" + name], new["step:" + name] = m, v, t
        return out, new

    return tensor_update(update, init_state=initialize, name="participating_adamw_staged_hp",
                         forward_state=True)


def stage(state, hp, learning_rate):
    """Fill every staged scalar for one update (host values from protocol)."""
    state[LR].fill_(learning_rate)
    state[WD].fill_(hp["weight_decay"])
    state[BETA2].fill_(hp["beta2"])
    state[OMB2].fill_(1.0 - hp["beta2"])
    state[LOGB2].fill_(math.log(hp["beta2"]))
