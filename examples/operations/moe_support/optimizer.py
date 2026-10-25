"""The recipe's optimizer as a user-written tensor rule (release step 12 ``tensor_update``): no compiler change.

Mathematics of ``research.learning_scaling.optim.MixedMuon`` (fused ``torch.optim.AdamW`` + the vendored
``muon_update``) after ``clip_grad_norm_(..., 1.0)``, written in the operation order of the PyTorch kernels:

* global clip: ``coef = min(clip / (||g||_2 + 1e-6), 1)`` over every gradient (tied embedding/readout: one summed
  gradient), applied to every gradient;
* AdamW (embeddings, gains, expert banks, routers, output norms) in the fused kernel's form:
  ``p -= lr*wd*p`` (decayed tensors only), ``m = b1 m + (1-b1) g``, ``v = b2 v + (1-b2) g g``,
  ``p -= (lr/bc1) m / (sqrt(v)/sqrt(bc2) + eps)``;
* Muon (attention qkv/out), stacked by shape as MixedMuon: ``B = lerp(B, G, 1-0.95)``,
  ``U = lerp(G, B, 0.95)`` (PyTorch's lerp branches), Newton-Schulz quintic in BF16 (5 steps, (3.4445, -4.7750,
  2.0315), BF16 Frobenius normalization + 1e-7), ``U *= sqrt(max(1, r/c))`` rounded to BF16 in place, then
  ``p = p*(1-lr*wd) + (-gain*lr) U``.

The per-update host scalars live in one FP32 optimizer-state vector ``schedule`` that the host writes before each
update (staging, like ``set_learning_rate``) and the rule returns unchanged:
``[lr, lr*wd, lr/bc1, sqrt(bc2), -gain*lr, 1-lr*wd]``. Adam's step counter is host state (all parameters advance
together, as in the recipe). Hyperparameters (betas, eps, gain, Newton-Schulz constants) are static.
"""
from __future__ import annotations

import math

import torch

NS = (3.4445, -4.7750, 2.0315)
SCHEDULE = "schedule"


def roles(model):
    """Canonical (deduplicated) parameter names -> 'muon' | 'adamw_decay' | 'adamw' (``build_optimizer`` rules)."""
    out = {}
    for name, p in model.named_parameters():
        hidden = (name.startswith(("blocks.", "second_head.", "fast.")) and p.ndim == 2 and
                  name.endswith((".up.weight", ".down.weight", ".qkv.weight", ".out.weight")))
        out[name] = "muon" if hidden else ("adamw_decay" if p.ndim >= 2 else "adamw")
    return out


def newton_schulz_bf16(G, steps=5):
    """vendor/muon.py ``zeropower_via_newtonschulz5`` on a stack [k, r, c]; the BF16 norm spelled as FP32
    accumulation of the BF16 squares, square root and one BF16 rounding (PyTorch's BF16 norm kernel)."""
    a, b, c = NS
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    wide = X.float()
    norm = torch.sqrt((wide * wide).sum(dim=(-2, -1), keepdim=True)).to(torch.bfloat16)
    X = X / (norm + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def make_rule(model, *, betas=(0.9, 0.95), eps=1e-8, clip=1.0, momentum=0.95):
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    role = roles(model)
    params = dict(model.named_parameters())
    groups = {}
    for name, kind in role.items():
        if kind == "muon":
            groups.setdefault(tuple(params[name].shape), []).append(name)
    b1, b2 = betas
    lerp_small = 1 - momentum                               # momentum.lerp_(grad, 0.05): weight < 0.5 branch
    lerp_large = float(1 - torch.tensor(momentum, dtype=torch.float32).item())  # grad.lerp_(B, 0.95): 1 - weight

    def update(p, g, state, hp):
        s = state[SCHEDULE]
        lr_wd, step_size, bc2_sqrt, neg, decay = s[1], s[2], s[3], s[4], s[5]
        total = None
        for name in p:
            term = (g[name] * g[name]).sum()
            total = term if total is None else total + term
        coef = torch.clamp(clip / (torch.sqrt(total) + 1e-6), max=1.0)
        new_p, new_s = {}, {SCHEDULE: s}
        for name, kind in role.items():
            if kind == "muon":
                continue
            h = g[name] * coef
            w = p[name] - lr_wd * p[name] if kind == "adamw_decay" else p[name]
            m = state["exp_avg:" + name] * b1 + h * (1 - b1)
            v = state["exp_avg_sq:" + name] * b2 + (h * h) * (1 - b2)
            new_p[name] = w - step_size * m / (torch.sqrt(v) / bc2_sqrt + eps)
            new_s["exp_avg:" + name], new_s["exp_avg_sq:" + name] = m, v
        for shape, names in groups.items():
            G = torch.stack([g[n] * coef for n in names])
            B0 = torch.stack([state["momentum:" + n] for n in names])
            B = B0 + lerp_small * (G - B0)
            U = B - (B - G) * lerp_large
            X = newton_schulz_bf16(U)
            X = (X.float() * max(1.0, shape[0] / shape[1]) ** 0.5).to(torch.bfloat16).float()
            for j, n in enumerate(names):
                new_p[n] = p[n] * decay + X[j] * neg
                new_s["momentum:" + n] = B[j]
        return new_p, new_s

    def init_state(params_, hp):
        state = {SCHEDULE: torch.zeros(6, dtype=torch.float32, device=next(iter(params_.values())).device)}
        for name, value in params_.items():
            if role[name] == "muon":
                state["momentum:" + name] = torch.zeros_like(value)
            else:
                state["exp_avg:" + name] = torch.zeros_like(value)
                state["exp_avg_sq:" + name] = torch.zeros_like(value)
        return state

    return tensor_update(update, init_state=init_state, name="ls_mixed_muon_adamw",
                         hyperparameters=dict(beta1=b1, beta2=b2, eps=eps, clip=clip, momentum=momentum, gain=20.0,
                                              ns_steps=5))


def schedule_vector(lr, adam_step, *, weight_decay=0.1, gain=20.0, betas=(0.9, 0.95)):
    """Host values for the update whose Adam step counter becomes ``adam_step`` (1-based, after increment)."""
    bc1 = 1 - betas[0] ** adam_step
    bc2 = 1 - betas[1] ** adam_step
    return torch.tensor([lr, lr * weight_decay, lr / bc1, math.sqrt(bc2), -gain * lr, 1 - lr * weight_decay],
                        dtype=torch.float32)

