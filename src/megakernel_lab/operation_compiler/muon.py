"""Muon + AdamW with one global clip and a runtime schedule vector (C-OPS4-A).

Pure captured optimizer arithmetic, like ``optim.py``: every update reads old
parameters/state and returns new values; the native runtime commits them once
after differentiation. Nothing here inspects model or parameter-name patterns:
the caller lists which canonical parameters use Muon and which AdamW parameters
receive decoupled decay.

Semantics are those of the S101H/S105F/S106B ordinary Muon recipe (donor
``research/s106/S106B/s106b/train.py`` at tag s106b-final, ``Engine._update``
and ``ns5``), written as ordinary PyTorch operations:

* one global L2 clip ``coef = clamp(clip / (sqrt(sum_p sum(g_p^2)) + clip_eps), max=1)``;
* AdamW on the listed parameters with lerp-form moments
  ``m += (1-b1)(h-m)``, ``v = b2 v + (1-b2) h^2``, ``w = p * decay`` only where listed,
  ``p' = w - ss m / (sqrt(v) rden + eps)``;
* Muon on the other parameters, grouped by shape and stacked as the donor does:
  ``B += (1-mu)(G-B)``; ``U = NS5(G + mu (B-G))`` in BF16 (Newton-Schulz quintic,
  coefficients (3.4445, -4.7750, 2.0315), Frobenius normalization + 1e-7);
  ``p' = p * decay + U * sqrt(max(1, rows/cols)) * neg``.

The schedule is one explicit FP32 data input (``schedule_input`` names its
index among the captured data inputs). Its first five entries are, per update,
``[lr, 1 - lr*wd, lr/bc1, 1/sqrt(bc2), -gain*lr]`` exactly as the donor's host
schedule writes them; further entries are free for the model (the donor keeps
the inner write rate there). Changing values needs no recapture.

The only spelled-out difference from the donor source: the donor's BF16
``X.norm(dim=(-2,-1))`` is written as FP32 accumulation of the BF16 values'
squares, square root, then BF16 rounding -- the same arithmetic PyTorch's BF16
norm kernel performs, but an explicit operation sequence the compiler lowers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .ir import CaptureError

NS5_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


@dataclass(frozen=True)
class MuonConfig:
    muon_parameters: tuple[str, ...]
    decay_parameters: tuple[str, ...] = ()
    momentum: float = 0.95
    ns_steps: int = 5
    ns_coefficients: tuple[float, float, float] = NS5_COEFFICIENTS
    ns_eps: float = 1e-7
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    clip: float = 1.0
    clip_eps: float = 1e-6
    schedule_input: int = -1          # index into the captured explicit data inputs
    schedule_entries: int = 5

    def validate(self):
        numbers = (self.momentum, self.ns_eps, *self.betas, self.eps, self.clip, self.clip_eps, *self.ns_coefficients)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in numbers):
            raise CaptureError("OPTIMIZER", "Muon constants must be finite Python numbers")
        if not 0 < self.momentum < 1 or not all(0 <= b < 1 for b in self.betas) or self.eps <= 0 or self.clip <= 0:
            raise CaptureError("OPTIMIZER", "require 0<momentum<1, betas in [0,1), eps>0, clip>0")
        if type(self.ns_steps) is not int or self.ns_steps < 1:
            raise CaptureError("OPTIMIZER", "Muon needs at least one Newton-Schulz step")
        if type(self.schedule_input) is not int:
            raise CaptureError("OPTIMIZER", "schedule_input must be a static data-input index")
        if type(self.schedule_entries) is not int or self.schedule_entries < 5:
            raise CaptureError("OPTIMIZER", "Muon schedule_entries must be at least five")
        if not isinstance(self.muon_parameters, tuple) or not isinstance(self.decay_parameters, tuple):
            raise CaptureError("OPTIMIZER", "parameter lists must be tuples of canonical names")


def newton_schulz(G, config):
    """Quintic Newton-Schulz orthogonalization of a stack [k, r, c] in BF16."""
    a, b, c = config.ns_coefficients
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    wide = X.float()
    norm = torch.sqrt((wide * wide).sum(dim=(-2, -1), keepdim=True)).to(torch.bfloat16)
    X = X / (norm + config.ns_eps)
    for _ in range(config.ns_steps):
        A = X @ X.mT
        B = b * A + c * A @ A         # donor precedence: (c * A) @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


@dataclass
class PreparedMuon:
    config: MuonConfig
    parameter_names: tuple[str, ...]
    state: dict[str, torch.Tensor]
    state_schema: tuple[dict[str, Any], ...]
    groups: dict[tuple[int, ...], tuple[str, ...]] = field(default_factory=dict)

    legacy_momentum = False
    uses_participation = False

    @property
    def state_names(self):
        return tuple(self.state)

    @property
    def state_values(self):
        return tuple(value.detach() for value in self.state.values())

    @property
    def options(self):
        return {}

    def metadata(self):
        return {"optimizer_state_version": 1,
                "optimizer_state_names": list(self.state_names),
                "optimizer_state_schema": list(self.state_schema),
                "optimizer": {"kind": "muon_adamw", **{k: (list(v) if isinstance(v, tuple) else v)
                                                       for k, v in vars(self.config).items()},
                              "muon_groups": [list(names) for names in self.groups.values()]},
                "optimizer_state_commit": "read old versions; commit every new parameter/optimizer-state output once after all graph reads",
                "optimizer_gradient_policy": "every canonical parameter participates (donor torch.autograd.grad semantics)",
                "optimizer_schedule": "runtime FP32 data input [lr, 1-lr*wd, lr/bc1, 1/sqrt(bc2), -gain*lr, ...]"}

    def update(self, params, grads, state, participation=None, inputs=()):
        cfg = self.config
        if len(params) != len(self.parameter_names) or len(grads) != len(params) or len(state) != len(self.state_names):
            raise CaptureError("OPTIMIZER_ABI", "optimizer parameter/gradient/state arity mismatch")
        if len(inputs) != 1:
            raise CaptureError("OPTIMIZER_ABI", "Muon reads exactly one schedule input")
        sc = inputs[0]
        lr_decay, ss, rden, neg = sc[1], sc[2], sc[3], sc[4]
        if any(g is None for g in grads):
            raise CaptureError("OPTIMIZER", "Muon requires a materialized gradient for every parameter")
        names = self.parameter_names
        index = {name: i for i, name in enumerate(names)}
        old = dict(zip(self.state_names, state))
        total = None
        for g in grads:
            term = (g * g).sum()
            total = term if total is None else total + term
        norm = torch.sqrt(total)
        coef = torch.clamp(cfg.clip / (norm + cfg.clip_eps), max=1.0)
        b1, b2 = cfg.betas
        new_params, new_state = list(params), dict(old)
        muon = set(cfg.muon_parameters)
        for name in names:
            if name in muon:
                continue
            i = index[name]
            h = grads[i] * coef
            m, v = old[f"exp_avg:{name}"], old[f"exp_avg_sq:{name}"]
            m_new = m + (1 - b1) * (h - m)
            v_new = v * b2 + (h * h) * (1 - b2)
            w = params[i] * lr_decay if name in cfg.decay_parameters else params[i]
            new_params[i] = w - (ss * m_new) / (torch.sqrt(v_new) * rden + cfg.eps)
            new_state[f"exp_avg:{name}"], new_state[f"exp_avg_sq:{name}"] = m_new, v_new
        for shape, group in self.groups.items():
            G = torch.stack([grads[index[n]] * coef for n in group])
            B = torch.stack([old[f"momentum:{n}"] for n in group])
            B = B + (1 - cfg.momentum) * (G - B)
            U = newton_schulz(G + cfg.momentum * (B - G), cfg).float() * max(1.0, shape[0] / shape[1]) ** 0.5
            for j, n in enumerate(group):
                new_params[index[n]] = params[index[n]] * lr_decay + U[j] * neg
                new_state[f"momentum:{n}"] = B[j]
        return tuple(new_params), tuple(new_state[k] for k in self.state_names)


def prepare_muon(config, params, *, optimizer_state=None):
    config.validate()
    names = tuple(params)
    unknown = (set(config.muon_parameters) | set(config.decay_parameters)) - set(names)
    if unknown:
        raise CaptureError("OPTIMIZER", f"Muon configuration names unknown parameters {sorted(unknown)}")
    if set(config.muon_parameters) & set(config.decay_parameters):
        raise CaptureError("OPTIMIZER", "decay_parameters lists AdamW parameters; Muon decay is always applied")
    groups = {}
    for name in names:
        if name in config.muon_parameters:
            if params[name].ndim != 2:
                raise CaptureError("OPTIMIZER", f"Muon parameter {name} must be a matrix")
            groups.setdefault(tuple(int(x) for x in params[name].shape), []).append(name)
    schema = []
    for name in names:
        kinds = ("momentum",) if name in config.muon_parameters else ("exp_avg", "exp_avg_sq")
        schema.extend({"name": f"{kind}:{name}", "parameter": name, "kind": kind, "dtype": "float32",
                       "shape": list(params[name].shape)} for kind in kinds)
    supplied = optimizer_state or {}
    if supplied and set(supplied) != {entry["name"] for entry in schema}:
        raise CaptureError("OPTIMIZER_STATE", "Muon state keys must exactly match the named schema")
    state = {}
    for entry in schema:
        value = supplied.get(entry["name"])
        if value is None:
            value = torch.zeros_like(params[entry["parameter"]])
        if (tuple(value.shape) != tuple(entry["shape"]) or value.dtype != torch.float32
                or not value.is_contiguous()):
            raise CaptureError("OPTIMIZER_STATE", f"state {entry['name']} does not match its tensor schema")
        state[entry["name"]] = value
    return PreparedMuon(config, names, state, tuple(schema), {k: tuple(v) for k, v in groups.items()})
