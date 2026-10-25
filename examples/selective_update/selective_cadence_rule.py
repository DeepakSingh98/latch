# Provenance: copied from research/latch_selective_cadence/cadence_rule.py at latch commit 6fcad414ad86
# (LATCH-SELECTIVE-10 final; git blob 5aeb45cebe6f67b9b1f88bb6e42e8a68f0150064). Body unchanged byte for byte:
# the numerical rule, its constants (WINDOW 16, beta1 0.9, eps 1e-6, clip 1.0) and staged-scalar names are the
# donor's. Only this provenance header was added. Its only import outside the standard library is the installed
# megakernel_lab.operation_compiler.user_optimizer.tensor_update.
"""Selective-cadence AdamW as an ordinary captured tensor-update rule (DESIGN.md "Update semantics").

One rule factory, two phases of the same persistent state (same names, shapes, dtypes):

  accumulate  fast group: participating AdamW + clip, identical arithmetic to the study's sb_rule restricted to the
              group; slow group: a <- a + g (absent experts contribute exact zeros), s <- max(s, active);
              slow parameters, moments and counters are returned unchanged (with update_schedule="streamed_ustate"
              they commit nothing).
  apply       fast group as above; slow group: G = (a + g)/16, own clip, one AdamW update with the staged window
              learning rate, window decay product and beta2_32; counter +16 (microbatch-equivalents) for every slow
              tensor present in the window; a and s reset.

The host stages every per-microbatch scalar (`stage`): fast LR and the B2 constants exactly as sb_rule, and for the
slow group the window learning rate kappa*mean(lr), the window decay product prod(1 - lr*wd_2), beta2_2^16,
1 - beta2_2^16 and log(beta2_2). Constants: beta1 0.9, eps 1e-6, clip 1.0, window 16.
"""
from __future__ import annotations

import math
import re

import torch

WINDOW = 16
BETA1, EPS, CLIP_NORM = 0.9, 1e-6, 1.0
EXPERT = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.")
ROUTER = re.compile(r"^model\.layers\.(\d+)\.block_sparse_moe\.gate\.weight$")
EMBEDDING, READOUT = "model.embed_tokens.weight", "lm_head.weight"

# Fast-group staged scalars and diagnostics: the sb_rule names (megakernel_lab.tinymoe.optimizer / sb_rule).
LR, WD, BETA2, OMB2, LOGB2 = ("schedule_learning_rate", "schedule_weight_decay", "schedule_beta2",
                              "schedule_one_minus_beta2", "schedule_log_beta2")
NORM, CLIP = "diagnostic_grad_norm", "diagnostic_clip_coefficient"
# Slow-group staged scalars and diagnostics.
S_LR, S_DECAY, S_BETA2, S_OMB2, S_LOGB2 = ("slow_window_learning_rate", "slow_window_decay_factor", "slow_beta2",
                                           "slow_one_minus_beta2", "slow_log_beta2_per_microbatch")
S_NORM, S_CLIP = "slow_diagnostic_grad_norm", "slow_diagnostic_clip_coefficient"
FAST_STAGED = (LR, WD, BETA2, OMB2, LOGB2)
SLOW_STAGED = (S_LR, S_DECAY, S_BETA2, S_OMB2, S_LOGB2)
PHASES = ("accumulate", "apply")
GROUPS = ("all", "experts", "shared", "shared_without_routers", "shared_without_io", "experts_and_routers")


def slow_set(names, group):
    """Names of the slow group. 'experts'/'shared' are the frozen A4/A3 partitions (routers with shared);
    the last three are the bounded refinement variants of DESIGN.md."""
    names = list(names)
    expert = {n for n in names if EXPERT.match(n)}
    router = {n for n in names if ROUTER.match(n)}
    shared = set(names) - expert
    table = dict(all=set(names), experts=expert, shared=shared,
                 shared_without_routers=shared - router,          # A3 refinement: routers fast with experts
                 shared_without_io=shared - {EMBEDDING, READOUT},  # A3 refinement: embedding/readout fast
                 experts_and_routers=expert | router)              # A4 refinement: routers slow with experts
    if group not in table:
        raise ValueError(f"unknown slow group {group}")
    return table[group]


def make_rule(model, *, slow_group, phase):
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    if phase not in PHASES:
        raise ValueError("phase must be accumulate or apply")
    names = [name for name, _ in model.named_parameters()]
    slow = slow_set(names, slow_group)
    roles = {name: EXPERT.match(name) for name in names}
    roles = {name: (int(m[1]), int(m[2])) if m else None for name, m in roles.items()}
    decay = {name: p.ndim >= 2 for name, p in model.named_parameters()}
    layers = model.config.num_hidden_layers
    slow_layers = sorted({roles[n][0] for n in slow if roles[n] is not None})
    experts = model.config.num_local_experts

    def initialize(parameters, _hp):
        first = next(iter(parameters.values()))
        state = {name: first.new_zeros(()) for name in FAST_STAGED + SLOW_STAGED + (NORM, S_NORM)}
        state[CLIP], state[S_CLIP] = first.new_ones(()), first.new_ones(())
        for name, p in parameters.items():
            state["exp_avg:" + name] = torch.zeros_like(p)
            state["exp_avg_sq:" + name] = torch.zeros_like(p)
            state["step:" + name] = torch.zeros((), dtype=torch.int64, device=p.device)
            if name in slow:
                state["acc:" + name] = torch.zeros_like(p)
        for layer in slow_layers:
            state[f"seen:{layer}"] = first.new_zeros(experts)
        return state

    def adamw(p, g, m0, v0, t0, t, rate, beta2, omb2, logb2, decayed):
        sf = t.to(torch.float32)
        bc1 = -torch.expm1(sf * math.log(BETA1))
        bc2 = -torch.expm1(sf * logb2)
        m = torch.lerp(m0, g, 1.0 - BETA1)
        v = beta2 * v0 + omb2 * g.square()
        return decayed - (rate / bc1) * m / (v.sqrt() / bc2.sqrt() + EPS), m, v

    def update(parameters, gradients, state, _hp, forward_state):
        out, new = {}, {}
        for name in FAST_STAGED + SLOW_STAGED:
            new[name] = state[name]
        active = {}
        for name, g in gradients.items():
            role = roles[name]
            if role is not None:
                active[name] = forward_state[f"active.{role[0]}"][role[1]] > 0
        masked = {name: (torch.where(active[name], g, torch.zeros_like(g)) if name in active else g)
                  for name, g in gradients.items()}

        # Fast group: sb_rule arithmetic restricted to the group (its own clip).
        fast = [name for name in names if name not in slow]
        if fast:
            squared = None
            for name in fast:
                g = masked[name]
                term = (g * g).sum()
                squared = term if squared is None else squared + term
            norm = squared.sqrt()
            coef = torch.clamp(CLIP_NORM / (norm + 1e-6), max=1.0)
            new[NORM], new[CLIP] = norm, coef
            rate, wd, beta2, omb2, logb2 = (state[k] for k in FAST_STAGED)
            for name in fast:
                p = parameters[name]
                g = gradients[name] * coef
                m0, v0, t0 = (state[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
                t = t0 + 1
                decayed = p * (1.0 - rate * wd) if decay[name] else p
                updated, m, v = adamw(p, g, m0, v0, t0, t, rate, beta2, omb2, logb2, decayed)
                if name in active:
                    flag = active[name]
                    updated = torch.where(flag, updated, p)
                    m, v, t = (torch.where(flag, a, b) for a, b in ((m, m0), (v, v0), (t, t0)))
                out[name] = updated
                new["exp_avg:" + name], new["exp_avg_sq:" + name], new["step:" + name] = m, v, t
        else:
            new[NORM], new[CLIP] = state[NORM], state[CLIP]

        # Slow group.
        seen = {layer: torch.maximum(state[f"seen:{layer}"], forward_state[f"active.{layer}"]) for layer in slow_layers}
        totals = {name: state["acc:" + name] + masked[name] for name in names if name in slow}
        if phase == "accumulate":
            for name in names:
                if name not in slow:
                    continue
                out[name] = parameters[name]
                for k in ("exp_avg", "exp_avg_sq", "step"):
                    new[k + ":" + name] = state[k + ":" + name]
                new["acc:" + name] = totals[name]
            for layer in slow_layers:
                new[f"seen:{layer}"] = seen[layer]
            new[S_NORM], new[S_CLIP] = state[S_NORM], state[S_CLIP]
            return out, new

        inverse = 1.0 / WINDOW
        means = {name: total * inverse for name, total in totals.items()}
        squared = None
        for name in names:
            if name not in slow:
                continue
            term = (means[name] * means[name]).sum()
            squared = term if squared is None else squared + term
        norm = squared.sqrt()
        coef = torch.clamp(CLIP_NORM / (norm + 1e-6), max=1.0)
        new[S_NORM], new[S_CLIP] = norm, coef
        rate, factor, beta2, omb2, logb2 = (state[k] for k in SLOW_STAGED)
        for name in names:
            if name not in slow:
                continue
            p = parameters[name]
            g = means[name] * coef
            m0, v0, t0 = (state[k + ":" + name] for k in ("exp_avg", "exp_avg_sq", "step"))
            t = t0 + WINDOW
            decayed = p * factor if decay[name] else p
            updated, m, v = adamw(p, g, m0, v0, t0, t, rate, beta2, omb2, logb2, decayed)
            role = roles[name]
            if role is not None:
                flag = seen[role[0]][role[1]] > 0
                updated = torch.where(flag, updated, p)
                m, v, t = (torch.where(flag, a, b) for a, b in ((m, m0), (v, v0), (t, t0)))
            out[name] = updated
            new["exp_avg:" + name], new["exp_avg_sq:" + name], new["step:" + name] = m, v, t
            new["acc:" + name] = torch.zeros_like(totals[name])
        for layer in slow_layers:
            new[f"seen:{layer}"] = torch.zeros_like(seen[layer])
        return out, new

    return tensor_update(update, init_state=initialize, name=f"selective_cadence_{slow_group}_{phase}",
                         forward_state=True)


def slow_scalars(learning_rates, weight_decay, beta2_2, kappa):
    """Host values (float64) for the window that ends now; learning_rates = the 16 scheduled B2 LRs."""
    if len(learning_rates) != WINDOW:
        raise ValueError("a slow update needs the window's 16 scheduled learning rates")
    factor = 1.0
    for lr in learning_rates:
        factor *= 1.0 - lr * weight_decay
    log2 = math.log(beta2_2)
    return {S_LR: kappa * math.fsum(learning_rates) / WINDOW, S_DECAY: factor, S_BETA2: math.exp(WINDOW * log2),
            S_OMB2: -math.expm1(WINDOW * log2), S_LOGB2: log2}


def stage(state, hp, learning_rate, slow=None):
    """Fill the fast scalars exactly as sb_rule.stage; the slow ones only when supplied (apply microbatches)."""
    state[LR].fill_(learning_rate)
    state[WD].fill_(hp["weight_decay"])
    state[BETA2].fill_(hp["beta2"])
    state[OMB2].fill_(1.0 - hp["beta2"])
    state[LOGB2].fill_(math.log(hp["beta2"]))
    if slow is not None:
        for name, value in slow.items():
            state[name].fill_(value)
