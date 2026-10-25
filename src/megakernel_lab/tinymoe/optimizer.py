"""AdamW and donor EMA as ordinary, captured tensor-update arithmetic.

Every expert retains its original Hugging Face parameter identity. A forward
participation mask, rather than gradient magnitude, decides whether AdamW may
change that expert. The evaluation EMA still observes every raw parameter.
"""
from __future__ import annotations

import math
import re

import torch

LR = "schedule_learning_rate"
EMA_ALPHA = "schedule_ema_alpha"
NORM = "diagnostic_grad_norm"
CLIP = "diagnostic_clip_coefficient"
EXPERT = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")


def initial_model_state(model):
    parameter = next(model.parameters())
    return {f"active.{i}": parameter.new_zeros(model.config.num_local_experts)
            for i in range(model.config.num_hidden_layers)}


def make_rule(model, recipe, *, ema_active=False):
    """Return the common compiler's stateful tensor-rule API, no special lowering."""
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    roles = {name: EXPERT.match(name) for name, _ in model.named_parameters()}
    roles = {name: (int(m[1]), int(m[2])) if m else None for name, m in roles.items()}
    decay = {name: p.ndim >= 2 for name, p in model.named_parameters()}
    keep_ema = recipe.name == "supervision_ema"
    if ema_active and not keep_ema:
        raise ValueError("EMA phase requires supervision_ema recipe")

    def initialize(parameters, hp):
        first = next(iter(parameters.values()))
        state = {LR: first.new_tensor(recipe.lr(0)), EMA_ALPHA: first.new_zeros(()),
                 NORM: first.new_zeros(()), CLIP: first.new_ones(())}
        for name, p in parameters.items():
            state["exp_avg:" + name] = torch.zeros_like(p)
            state["exp_avg_sq:" + name] = torch.zeros_like(p)
            state["step:" + name] = torch.zeros((), dtype=torch.int64, device=p.device)
            if keep_ema:
                state["ema:" + name] = torch.zeros_like(p)
        return state

    def update(parameters, gradients, state, hp, forward_state):
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
        coef = torch.clamp(recipe.clip_norm / (norm + 1e-6), max=1.0)
        out, new = {}, {LR: state[LR], EMA_ALPHA: state[EMA_ALPHA], NORM: norm, CLIP: coef}
        rate = state[LR]
        for name, p in parameters.items():
            g = gradients[name] * coef
            m0, v0, t0 = (state[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
            t = t0 + 1
            sf = t.to(torch.float32)
            bc1 = -torch.expm1(sf * math.log(recipe.beta1))
            bc2 = -torch.expm1(sf * math.log(recipe.beta2))
            m = torch.lerp(m0, g, 1.0 - recipe.beta1)
            v = recipe.beta2 * v0 + (1.0 - recipe.beta2) * g.square()
            decayed = p * (1.0 - rate * recipe.weight_decay) if decay[name] else p
            updated = decayed - (rate / bc1) * m / (v.sqrt() / bc2.sqrt() + recipe.eps)
            if name in active:
                flag = active[name]
                updated = torch.where(flag, updated, p)
                m, v, t = (torch.where(flag, a, b) for a, b in ((m, m0), (v, v0), (t, t0)))
            out[name] = updated
            new["exp_avg:" + name], new["exp_avg_sq:" + name], new["step:" + name] = m, v, t
            if keep_ema:
                old = state["ema:" + name]
                # alpha=1 copies the just-updated raw state at the donor's
                # activation boundary; later alpha=1-exp(-16/H).
                new["ema:" + name] = torch.lerp(old, updated, state[EMA_ALPHA]) if ema_active else old
        return out, new

    return tensor_update(update, init_state=initialize, name="participating_adamw_donor_ema",
                         forward_state=True)


def stage_schedule(state, recipe, update_index):
    """Stage only explicit schedule bindings; all parameter work stays generated."""
    completed = update_index + 1
    start = (recipe.horizon + 1) // 2
    alpha = (1.0 if completed == start else 1.0 - math.exp(-16.0 / recipe.horizon))
    if recipe.name != "supervision_ema" or completed < start:
        alpha = 0.0
    state[LR].fill_(recipe.lr(update_index))
    state[EMA_ALPHA].fill_(alpha)
