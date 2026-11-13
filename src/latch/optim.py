"""Optimizers written as pure tensor functions.

``update`` is traced together with the model, so its arithmetic becomes part of
the same graph and the same kernel. State is explicit: every call reads the old
tensors and returns new ones; the compiler decides how to write them back.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SGDConfig:
    """torch.optim.SGD with optional momentum and coupled weight decay."""
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0

    def init(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.momentum:
            return {}
        return {f"momentum:{name}": torch.zeros_like(p) for name, p in params.items()}

    def update(self, params, grads, state):
        new_params, new_state = {}, {}
        for name, p in params.items():
            g = grads[name]
            if self.weight_decay:
                g = g + self.weight_decay * p
            if self.momentum:
                g = new_state[f"momentum:{name}"] = self.momentum * state[f"momentum:{name}"] + g
            new_params[name] = p - self.lr * g
        return new_params, new_state


@dataclass(frozen=True)
class AdamWConfig:
    """torch.optim.AdamW (decoupled weight decay, bias-corrected moments)."""
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    def init(self, params):
        state = {"step": torch.zeros((), dtype=torch.int64)}
        for name, p in params.items():
            state[f"exp_avg:{name}"] = torch.zeros_like(p)
            state[f"exp_avg_sq:{name}"] = torch.zeros_like(p)
        return state

    def update(self, params, grads, state):
        beta1, beta2 = self.betas
        step = state["step"] + 1
        t = step.to(torch.float32)
        # 1 - beta**t, written with expm1 so float32 keeps its precision at small t.
        correction1 = -torch.expm1(t * math.log(beta1))
        correction2 = -torch.expm1(t * math.log(beta2))
        step_size = self.lr / correction1
        new_params, new_state = {}, {"step": step}
        for name, p in params.items():
            g = grads[name]
            m = new_state[f"exp_avg:{name}"] = beta1 * state[f"exp_avg:{name}"] + (1 - beta1) * g
            v = new_state[f"exp_avg_sq:{name}"] = beta2 * state[f"exp_avg_sq:{name}"] + (1 - beta2) * (g * g)
            denom = torch.sqrt(v) / torch.sqrt(correction2) + self.eps
            new_params[name] = p * (1 - self.lr * self.weight_decay) - step_size * (m / denom)
        return new_params, new_state
