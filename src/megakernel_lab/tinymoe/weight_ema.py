"""Evaluation-only per-update FP32 EMA, with strict restart and tied parameters.

Policy: copy after ceil(H/2) completed optimizer updates, then foreach lerp
after every update with beta=exp(-16/H). Never rebind a Parameter or touch an
optimizer state. This is not averaging sparse saved checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import torch

POLICY = "half-horizon-tau-h16-v1"


class WeightEMA:
    def __init__(self, model, horizon):
        if type(horizon) is not int or horizon < 2:
            raise ValueError("EMA needs an original horizon of at least two updates")
        self.horizon = horizon
        self.start = (horizon + 1) // 2
        self.beta = math.exp(-16. / horizon)
        self.params = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
        self.aliases = {}
        canonical = {id(p): n for n, p in self.params.items()}
        for name, p in model.named_parameters(remove_duplicate=False):
            if p.requires_grad:
                self.aliases[name] = canonical[id(p)]
        self.average = {}
        self.last_step = None
        self.updates = 0
        self._installed = False

    def policy(self):
        return dict(name=POLICY, horizon_updates=self.horizon, start_update=self.start,
                    beta=self.beta, time_constant_updates=self.horizon / 16.,
                    dtype="torch.float32", initialization="raw after start_update",
                    trainable_aliases=self.aliases)

    @property
    def ready(self):
        return bool(self.average)

    @property
    def bytes(self):
        return sum(p.numel() * 4 for p in self.params.values())

    @torch.no_grad()
    def update(self, completed_step):
        if self._installed:
            raise RuntimeError("cannot update EMA during averaged evaluation")
        if type(completed_step) is not int or completed_step < 0:
            raise ValueError("invalid EMA update cursor")
        if self.last_step is not None and completed_step != self.last_step + 1:
            raise ValueError("EMA must observe every optimizer update exactly once")
        if self.last_step is None and completed_step > self.start:
            raise ValueError("cannot initialize EMA after its declared start")
        if completed_step == self.start:
            self.average = {n: p.detach().to(dtype=torch.float32, copy=True) for n, p in self.params.items()}
        elif completed_step > self.start:
            if not self.ready:
                raise ValueError("missing EMA state")
            # Separate averages cannot alter the original FP32 master weights.
            torch._foreach_lerp_(list(self.average.values()),
                                 [p.detach().float() for p in self.params.values()], 1. - self.beta)
            self.updates += 1
        self.last_step = completed_step

    def state_dict(self):
        if self._installed:
            raise RuntimeError("cannot checkpoint while averaged weights are installed")
        return dict(policy=self.policy(), last_step=self.last_step, updates=self.updates,
                    average=self.average)

    def load_state_dict(self, state, completed_step):
        if set(state) != {"policy", "last_step", "updates", "average"} or state["policy"] != self.policy():
            raise ValueError("EMA policy/parameter relationship mismatch")
        if state["last_step"] != completed_step or state["updates"] != max(0, completed_step - self.start):
            raise ValueError("EMA cursor mismatch")
        expected = self.params if completed_step >= self.start else {}
        if state["average"].keys() != expected.keys():
            raise ValueError("EMA trainable parameter coverage mismatch")
        average = {}
        for name, value in state["average"].items():
            if value.dtype != torch.float32 or value.shape != self.params[name].shape or not torch.isfinite(value).all():
                raise ValueError("EMA tensor dtype/shape/finite mismatch: " + name)
            average[name] = value.to(device=self.params[name].device, copy=True)
        self.average, self.last_step, self.updates = average, completed_step, state["updates"]

    @contextmanager
    @torch.no_grad()
    def apply(self):
        if not self.ready or self._installed:
            raise ValueError("EMA is unavailable or already installed")
        # Copy values in place: tied readouts and optimizer references survive.
        backup = {n: p.detach().clone() for n, p in self.params.items()}
        self._installed = True
        try:
            for n, p in self.params.items():
                p.copy_(self.average[n])
            yield
        finally:
            for n, p in self.params.items():
                p.copy_(backup[n])
            self._installed = False


def validate_import(old, new, spec, step):
    """Explicit one-time import of a hash-pinned legacy raw checkpoint.

Only source identity, the declared equivalent legacy arm metadata, and added
EMA policy may differ. All numerical/data/topology/runtime fields stay exact.
Native restart never uses this migration path.
"""
    import copy
    if old != spec["contract"]:
        raise ValueError("legacy checkpoint differs from pinned import contract")
    a, b = copy.deepcopy(old), copy.deepcopy(new)
    if not a.get("source_sha256") or not b.get("source_sha256"):
        raise ValueError("source identities required for explicit import")
    a.pop("source_sha256"); b.pop("source_sha256")
    policy = b.pop("weight_ema", None)
    if policy is not None and step > policy["start_update"]:
        raise ValueError("legacy import would skip EMA updates")
    if "weight_ema" in a:
        raise ValueError("EMA checkpoints require exact native restart")
    arm = a["model"]["arm"]
    heads = {"fine_grained_moe": 0, "moe_intermediate_supervision": 2}.get(arm)
    if heads is None:
        raise ValueError("only pinned E32 legacy MoE arms may be imported")
    a["model"].update(arm="expert_frontier", expert_granularity=8, supervision_heads=heads)
    if a != b:
        changed = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
        raise ValueError("legacy import numerical contract mismatch: " + repr(changed))
