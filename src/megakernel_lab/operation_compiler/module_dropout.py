"""Release step 08: ordinary ``nn.Dropout`` modules -> explicit Philox state before AD.

Capture cannot use PyTorch's implicit generator (capture.py refuses seeded random
operations). This module translates one bounded form, exact ``torch.nn.Dropout``
instances, into the existing explicit-state operation ``ops.random.dropout``
(Philox4x32-10 ``philox_mask`` + ``philox_advance``). No other random-number
implementation is introduced. The translation happens while the user's forward
runs under ``make_fx``, i.e. before ``grad_and_value`` differentiates it, so AD
only sees ``x * mask`` with one opaque mask value per call. Backward reads that
same forward mask value; nothing in the derivative draws again.

Contract (module training mode, FP32 input, static p):

* One explicit int64[4] state (``STATE_NAME``) is threaded through every active
  call in Python execution order. Each call with 0 < p < 1 consumes
  ceil(numel/4) Philox blocks exactly like ``ops.random.dropout``; the update's
  final state is committed as ordinary model state and therefore checkpointed by
  ``ModuleUpdate.state_dict``. A module instance called twice draws twice.
* p == 0: identity, no state consumed (PyTorch draws nothing either).
* p == 1: ``input * 0`` exactly as PyTorch's ``at::dropout`` (zero output, zero
  gradient, NaN/Inf propagate as NaN), no state consumed. ``ops.random.dropout``
  itself still refuses p == 1 because its inverted scale would be infinite.
* ``module.training`` False: identity (evaluation), no state consumed.
* Drop probability is quantized to floor(p * 2**32) / 2**32 (< 2**-32 from p);
  the kept scale is FP32(1 / (1 - p)). Masks are not PyTorch's random stream.

Refused with ``CaptureError("DROPOUT_FORM")``: ``inplace=True`` in training mode,
non-FP32 inputs, subclasses of ``nn.Dropout`` and the other ``_DropoutNd``
modules (Dropout1d/2d/3d, AlphaDropout, FeatureAlphaDropout) in training mode.
Functional ``F.dropout``/``torch.dropout``/``bernoulli`` calls and SDPA
``dropout_p`` stay refused by the generic capture RNG check.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import torch
from torch import nn

from .ir import CaptureError
from .random import U64, dropout as explicit_dropout, make_rng_state

STATE_NAME = "dropout_rng"
SCHEMA = "latch-module-dropout-v1"


@dataclass(frozen=True)
class Site:
    name: str
    p: float
    mode: str      # "philox" | "identity_p0" | "zero_p1" | "identity_eval"


@dataclass
class Plan:
    """Static description of the module's dropout sites at capture time."""
    sites: tuple[Site, ...]
    seed: int
    counter: int
    seed_source: str
    calls: list = field(default_factory=list)

    @property
    def random(self):
        return any(site.mode == "philox" for site in self.sites)

    def initial_state(self, device="cpu"):
        return make_rng_state(self.seed, self.counter, device=device)

    def metadata(self):
        return {"schema": SCHEMA, "state_name": STATE_NAME if self.random else None,
                "seed": self.seed, "initial_counter": self.counter, "seed_source": self.seed_source,
                "sites": [{"name": s.name, "p": s.p, "mode": s.mode} for s in self.sites],
                "calls": list(self.calls),
                "operation": "ops.random.dropout (Philox4x32-10 philox_mask + philox_advance)",
                "semantics": "inverted dropout; drop iff word < floor(p*2**32); keep scale FP32(1/(1-p)); "
                             "state chained across calls in Python execution order; one mask per call reused by AD",
                "scope": "exact torch.nn.Dropout modules only; not PyTorch's random stream"}


def _active_kind(module):
    return type(module) is nn.Dropout


def plan_dropout(model: nn.Module, seed=None, counter: int = 0) -> Plan | None:
    """Return None when the model has no exact nn.Dropout and no refused dropout form."""
    sites = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.modules.dropout._DropoutNd):
            continue
        p = float(module.p)
        if not _active_kind(module):
            if module.training and p > 0:
                raise CaptureError("DROPOUT_FORM", f"{name}: {type(module).__name__} is not the supported dropout form; "
                                   "only exact torch.nn.Dropout is translated to explicit random state")
            sites.append(Site(name, p, "identity_eval" if not module.training else "identity_p0"))
            continue
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise CaptureError("DROPOUT_FORM", f"{name}: dropout probability must be in [0, 1]")
        if not module.training:
            mode = "identity_eval"
        elif module.inplace and p > 0:
            raise CaptureError("DROPOUT_FORM", f"{name}: inplace=True dropout mutates its input; use inplace=False")
        else:
            mode = "identity_p0" if p == 0 else "zero_p1" if p == 1 else "philox"
        sites.append(Site(name, p, mode))
    if not sites:
        return None
    source = "explicit"
    if seed is None:
        # torch.manual_seed(s) callers get a reproducible explicit seed; the
        # stream itself is Philox4x32-10 and differs from PyTorch's.
        seed, source = int(torch.initial_seed()) & U64, "torch.initial_seed()"
    if type(seed) is not int or not 0 <= seed <= U64 or type(counter) is not int or not 0 <= counter <= U64:
        raise CaptureError("DROPOUT_SEED", "dropout seed/counter must be unsigned 64-bit integers")
    return Plan(tuple(sites), seed, counter, source)


class _Carrier:
    def __init__(self, state):
        self.state, self.calls = state, []


@contextmanager
def bind(model: nn.Module, plan: Plan, state):
    """Temporarily route every exact nn.Dropout forward through explicit state.

    Yields a carrier whose ``state`` holds the advanced int64[4] state after the
    forward ran inside the context. Usable in capture_module and inside a loop
    body passed to capture_loop_training. The module is restored on exit.
    """
    carrier = _Carrier(state)
    patched = []

    def make_forward(name, module):
        def forward(input):
            if not module.training or module.p == 0:
                return input
            if not isinstance(input, torch.Tensor) or input.dtype != torch.float32:
                raise CaptureError("DROPOUT_FORM", f"{name}: translated dropout requires an FP32 Tensor input")
            if module.inplace:
                raise CaptureError("DROPOUT_FORM", f"{name}: inplace=True dropout mutates its input")
            if module.p == 1:
                return input * 0.0
            if carrier.state is None:
                raise CaptureError("DROPOUT_STATE", f"{name}: no explicit dropout state is bound")
            output, carrier.state = explicit_dropout(input, carrier.state, float(module.p))
            carrier.calls.append({"site": name, "shape": list(input.shape), "blocks": (input.numel() + 3) // 4})
            return output
        return forward

    try:
        for name, module in model.named_modules():
            if _active_kind(module):
                had = "forward" in module.__dict__
                patched.append((module, had, module.__dict__.get("forward")))
                module.forward = make_forward(name, module)
        yield carrier
    finally:
        for module, had, previous in patched:
            if had:
                module.forward = previous
            else:
                del module.forward
        # Keep the most recent trace's call record (all traces see the same static program).
        if carrier.calls:
            plan.calls[:] = carrier.calls


def blocks_per_update(plan: Plan) -> int:
    return sum(call["blocks"] for call in plan.calls)


__all__ = ["STATE_NAME", "Plan", "Site", "bind", "blocks_per_update", "plan_dropout"]
