"""Gradient accumulation expressed as an ordinary captured tensor-update rule.

No compiler change. One generated program processes one 8x1024 microbatch.
A staged scalar FINAL (0/1) selects the behaviour of the same program:

* FINAL=0: acc += g (experts masked by their microbatch participation),
  seen |= active; parameters, AdamW moments and counters pass through bitwise.
* FINAL=1: G = acc + g; one global clip of G; one AdamW update of every
  participating parameter (published hyper-parameters); acc and seen reset.

The objective is the unchanged captured TinyMoE objective with the staged CE
coefficient 1/16 and the fixed 0.02 router term per microbatch, i.e. the
Transformers 4.55 Trainer reduction for 16 equal microbatches. Experts that
received no token in any microbatch keep HF absent-gradient behaviour.
The where-gated AdamW arithmetic also runs (and is discarded) on non-final
microbatches; that cost is included in the Latch clock.
"""
from __future__ import annotations

import math

import torch

from megakernel_lab.tinymoe.optimizer import CLIP, EXPERT, LR, NORM

from .ga_common import PUBLISHED

FINAL = "schedule_final_microbatch"


def make_accumulating_rule(model, hp=None):
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    hp = dict(PUBLISHED if hp is None else hp)
    beta1, beta2, eps = hp["adam_beta1"], hp["adam_beta2"], hp["adam_epsilon"]
    weight_decay, clip_norm = hp["weight_decay"], hp["max_grad_norm"]
    layers, experts = model.config.num_hidden_layers, model.config.num_local_experts
    roles = {name: EXPERT.match(name) for name, _ in model.named_parameters()}
    roles = {name: (int(m[1]), int(m[2])) if m else None for name, m in roles.items()}
    # HF Trainer decays every parameter except bias/norm names; TinyMoE has no
    # biases, and its norm gammas are exactly the 1-D parameters.
    decay = {name: p.ndim >= 2 for name, p in model.named_parameters()}

    def initialize(parameters, _hp):
        first = next(iter(parameters.values()))
        state = {LR: first.new_zeros(()), FINAL: first.new_zeros(()),
                 NORM: first.new_zeros(()), CLIP: first.new_ones(())}
        for name, p in parameters.items():
            state["exp_avg:" + name] = torch.zeros_like(p)
            state["exp_avg_sq:" + name] = torch.zeros_like(p)
            state["step:" + name] = torch.zeros((), dtype=torch.int64, device=p.device)
            state["acc:" + name] = torch.zeros_like(p)
        for layer in range(layers):
            state[f"seen:{layer}"] = first.new_zeros(experts)
        return state

    def update(parameters, gradients, state, _hp, forward_state):
        final = state[FINAL] > 0
        seen = {layer: torch.maximum(state[f"seen:{layer}"], forward_state[f"active.{layer}"])
                for layer in range(layers)}
        totals, flags, squared = {}, {}, None
        for name, g in gradients.items():
            role = roles[name]
            if role is not None:
                layer, expert = role
                g = torch.where(forward_state[f"active.{layer}"][expert] > 0, g, torch.zeros_like(g))
                flags[name] = seen[layer][expert] > 0
            total = state["acc:" + name] + g
            totals[name] = total
            term = (total * total).sum()
            squared = term if squared is None else squared + term
        norm = squared.sqrt()
        coef = torch.clamp(clip_norm / (norm + 1e-6), max=1.0)
        out = {}
        new = {LR: state[LR], FINAL: state[FINAL],
               NORM: torch.where(final, norm, state[NORM]), CLIP: torch.where(final, coef, state[CLIP])}
        rate = state[LR]
        for name, p in parameters.items():
            g = totals[name] * coef
            m0, v0, t0 = (state[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
            t = t0 + 1
            sf = t.to(torch.float32)
            bc1 = -torch.expm1(sf * math.log(beta1))
            bc2 = -torch.expm1(sf * math.log(beta2))
            m = torch.lerp(m0, g, 1.0 - beta1)
            v = beta2 * v0 + (1.0 - beta2) * g.square()
            decayed = p * (1.0 - rate * weight_decay) if decay[name] else p
            updated = decayed - (rate / bc1) * m / (v.sqrt() / bc2.sqrt() + eps)
            apply = final & flags[name] if name in flags else final
            out[name] = torch.where(apply, updated, p)
            new["exp_avg:" + name] = torch.where(apply, m, m0)
            new["exp_avg_sq:" + name] = torch.where(apply, v, v0)
            new["step:" + name] = torch.where(apply, t, t0)
            new["acc:" + name] = torch.where(final, torch.zeros_like(totals[name]), totals[name])
        for layer in range(layers):
            new[f"seen:{layer}"] = torch.where(final, torch.zeros_like(seen[layer]), seen[layer])
        return out, new

    return tensor_update(update, init_state=initialize, name="accumulating_published_adamw",
                         forward_state=True)
