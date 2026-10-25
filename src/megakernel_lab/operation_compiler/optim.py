"""Pure captured optimizer arithmetic and named persistent-state ABI.

State is versioned: every update reads old parameters/state and returns new
values. Native runtime commits those values only after differentiation finishes.
No optimizer tensor belongs to model state, RNG state or a loop's saved tape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .ir import CaptureError


ADAMW_STEP_LIMIT = 2**24


@dataclass(frozen=True)
class SGDConfig:
    """Existing momentum-SGD contract: zero materialized unused gradients.

    No dampening/Nesterov; coupled decay. An explicit zero initial momentum
    agrees with torch.optim.SGD's first-use buffer when dampening is zero.
    """
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0

    def validate(self):
        if not all(math.isfinite(v) for v in (self.lr, self.momentum, self.weight_decay)):
            raise CaptureError("OPTIMIZER", "optimizer constants must be finite")
        if self.lr < 0 or not 0 < self.momentum < 1 or self.weight_decay < 0:
            raise CaptureError("OPTIMIZER", "require lr >= 0, 0 < momentum < 1, weight_decay >= 0")


@dataclass(frozen=True)
class AdamWConfig:
    """Dense FP32 AdamW, static hyperparameters and per-parameter int64 steps.

    ``param_options`` overrides lr/betas/eps/weight_decay by canonical parameter
    name. Unused (None-gradient) parameters skip decay, moments and step; a
    participating zero gradient still updates. AMSGrad, maximize, sparse gradients,
    differentiable optimizers and host-mutated schedules are not admitted.
    Every update must start with step < 2**24, where the pinned conventional
    FP32 step counter and the FP32 bias-correction exponent remain exact.
    Callers must end/re-capture a training lifetime before exceeding this bound;
    native counters advance as int64 and are never silently saturated.

    runtime_lr appends a scalar FP32 input to flat/module capture. Its value is
    an absolute learning rate, staged without recapture; callers must keep it
    finite and nonnegative. Per-parameter LR overrides are refused. Loop and
    sampled AdamW retain static rates; scheduled Muon has a separate loop ABI.
    warmup_steps applies the accepted P7-A/B linear warmup to each parameter's
    base rate using its own participating-update counter. It is mutually
    exclusive with an externally supplied absolute runtime_lr.
    """
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    param_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    amsgrad: bool = False
    maximize: bool = False
    # C-OPS4-B: optional global-norm clipping before AdamW (s80u boundary form, S100L SPEC section 7):
    # norm = sqrt(sum of squared participating raw gradients), coef = max_norm / (norm + 1e-6), NaN kept, else
    # min(coef, 1); moments use g * coef. Returned gradients stay raw. None (default) reproduces plain AdamW.
    max_grad_norm: float | None = None
    # P6-E / C-OPS2-C: absolute AdamW rate is one explicit FP32 scalar binding.
    runtime_lr: bool = False
    # C-OPS2-B prerequisite, preserved by P7-A/B: scale lr by
    # min(1, (old_step + 1) / warmup_steps); a None gradient skips its counter.
    warmup_steps: int = 0

    def validate(self):
        if self.amsgrad or self.maximize:
            raise CaptureError("OPTIMIZER", "AdamW AMSGrad/maximize are not supported")
        if self.max_grad_norm is not None and (isinstance(self.max_grad_norm, bool) or
                                               not isinstance(self.max_grad_norm, (int, float)) or
                                               not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise CaptureError("OPTIMIZER", "max_grad_norm must be a finite positive number or None")
        _validate_adam_options(self.options())
        if not isinstance(self.param_options, dict):
            raise CaptureError("OPTIMIZER", "param_options must map parameter names to static options")
        if type(self.runtime_lr) is not bool:
            raise CaptureError("OPTIMIZER", "runtime_lr must be a bool")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < ADAMW_STEP_LIMIT:
            raise CaptureError("OPTIMIZER", "warmup_steps must be an integer in [0, 2**24)")
        if self.runtime_lr and self.warmup_steps:
            raise CaptureError("OPTIMIZER", "runtime_lr is an absolute rate and cannot also apply counter-based warmup")
        for name, options in self.param_options.items():
            if not isinstance(name, str) or not isinstance(options, dict):
                raise CaptureError("OPTIMIZER", "param_options requires named dictionaries")
            if set(options) - {"lr", "betas", "eps", "weight_decay"}:
                raise CaptureError("OPTIMIZER", f"unsupported AdamW parameter options for {name}")
            if self.runtime_lr and "lr" in options:
                raise CaptureError("OPTIMIZER", "runtime_lr refuses per-parameter learning-rate overrides")
            _validate_adam_options({**self.options(), **options})

    def options(self):
        return {"lr": self.lr, "betas": self.betas, "eps": self.eps,
                "weight_decay": self.weight_decay}


def _validate_adam_options(options):
    try:
        b1, b2 = options["betas"]
        vals = (options["lr"], b1, b2, options["eps"], options["weight_decay"])
        if any(isinstance(v, (bool, torch.Tensor)) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in vals):
            raise ValueError("constants must be finite Python numbers")
        if (options["lr"] < 0 or not 0 <= b1 < 1 or not 0 <= b2 < 1
                or options["eps"] <= 0 or options["weight_decay"] < 0):
            raise ValueError("require lr >= 0, betas in [0,1), eps > 0 and decay >= 0")
    except (TypeError, ValueError) as exc:
        raise CaptureError("OPTIMIZER", f"invalid AdamW options: {exc}") from exc


def _canonical(params):
    ids, canonical, aliases = {}, {}, {}
    for name, value in params.items():
        key = ids.setdefault(id(value), name)
        canonical.setdefault(key, value)
        aliases[name] = key
    return canonical, aliases


def initialize_optimizer_state(params, config=None):
    """Allocate explicit zero state once; tied Tensor identities share one entry."""
    config = config or SGDConfig()
    config.validate()
    canonical, _ = _canonical(params)
    from .user_optimizer import TensorUpdateRule, initialize_user_state
    if isinstance(config, TensorUpdateRule):
        return initialize_user_state(config, canonical)
    from .muon import MuonConfig, prepare_muon
    if isinstance(config, MuonConfig):
        return prepare_muon(config, canonical).state
    if isinstance(config, SGDConfig):
        return {name: torch.zeros_like(p) for name, p in canonical.items()}
    if not isinstance(config, AdamWConfig):
        raise CaptureError("OPTIMIZER", "expected SGDConfig, AdamWConfig or MuonConfig")
    result = {}
    for name, p in canonical.items():
        result[f"exp_avg:{name}"] = torch.zeros_like(p)
        result[f"exp_avg_sq:{name}"] = torch.zeros_like(p)
        result[f"step:{name}"] = torch.zeros((), dtype=torch.int64, device=p.device)
    return result


@dataclass
class PreparedOptimizer:
    config: SGDConfig | AdamWConfig
    parameter_names: tuple[str, ...]
    state: dict[str, torch.Tensor]
    options: dict[str, dict[str, Any]]
    state_schema: tuple[dict[str, Any], ...]

    @property
    def legacy_momentum(self):
        return isinstance(self.config, SGDConfig)

    @property
    def uses_participation(self):
        return not self.legacy_momentum

    @property
    def runtime_lr(self):
        return not self.legacy_momentum and self.config.runtime_lr

    @property
    def state_names(self):
        return tuple(self.state)

    @property
    def state_values(self):
        return tuple(value.detach() for value in self.state.values())

    def metadata(self):
        return {"optimizer_state_version": 1,
                "optimizer_state_names": list(self.state_names),
                "optimizer_state_schema": list(self.state_schema),
                "optimizer": ({"kind": "sgd_momentum", **vars(self.config)} if self.legacy_momentum else
                              {"kind": "adamw", **vars(self.config), "parameter_options": self.options}),
                "optimizer_state_commit": "read old versions; commit every new parameter/optimizer-state output once after all graph reads",
                "optimizer_gradient_policy": ("legacy materialized zeros participate" if self.legacy_momentum
                                               else "None skips parameter/decay/moments/step; participating zero advances"),
                **({"optimizer_global_clip": {"max_norm": self.config.max_grad_norm, "coef": "max_norm/(norm+1e-6), NaN kept else min(.,1)",
                                              "returned_gradients": "raw (unclipped)"}}
                   if not self.legacy_momentum and self.config.max_grad_norm is not None else {}),
                **({"optimizer_lr_schedule": {"kind": "linear_warmup", "warmup_steps": self.config.warmup_steps,
                    "factor": "min(1, (old_parameter_step+1)/warmup_steps)",
                    "counter": "per-parameter participating-update int64 state", "after_warmup": "constant base learning rate"}}
                   if not self.legacy_momentum and self.config.warmup_steps else {}),
                **({} if self.legacy_momentum else {
                    "optimizer_step_input_bound": {"minimum": 0, "exclusive_maximum": ADAMW_STEP_LIMIT},
                    "optimizer_step_lifetime_contract": "each update starts below the bound; native int64 advancement does not saturate or read host counters"})}

    def update(self, params, grads, state, participation=None, inputs=(), lr=None):
        """Return flat new P/O. Optional bool Tensor participation supports regions.

        A dynamic false predicate selects unchanged values, including the counter;
        arithmetic is pure and never advances externally visible state itself.
        """
        if len(params) != len(self.parameter_names) or len(grads) != len(params) or len(state) != len(self.state_names):
            raise CaptureError("OPTIMIZER_ABI", "optimizer parameter/gradient/state arity mismatch")
        if participation is not None and len(participation) != len(params):
            raise CaptureError("OPTIMIZER_ABI", "gradient participation arity mismatch")
        if (lr is not None) != self.runtime_lr:
            raise CaptureError("OPTIMIZER_ABI", "runtime learning-rate input must be supplied exactly when runtime_lr is set")
        if lr is not None and (not isinstance(lr, torch.Tensor) or lr.dtype != torch.float32 or lr.ndim != 0):
            raise CaptureError("OPTIMIZER_ABI", "runtime learning rate must be a scalar FP32 tensor")
        if self.legacy_momentum:
            momentum = tuple(self.config.momentum * m + (g + self.config.weight_decay * p)
                             for p, g, m in zip(params, grads, state))
            return tuple(p - self.config.lr * m for p, m in zip(params, momentum)), momentum
        new_params, new_state = [], []
        coef = None
        if self.config.max_grad_norm is not None:
            squares = []
            for i, g in enumerate(grads):
                active = participation[i] if participation is not None else g is not None
                if g is None or active is False:
                    continue
                # Conditional-loop participation is a device scalar. An inactive
                # gradient (including a nonfinite placeholder) must not affect
                # any participating parameter's shared clipping coefficient.
                clip_gradient = (torch.where(active, g, torch.zeros_like(g))
                                 if isinstance(active, torch.Tensor) else g)
                squares.append((clip_gradient * clip_gradient).sum())
            if squares:
                total = squares[0]
                for value in squares[1:]:
                    total = total + value
                denominator = torch.sqrt(total) + 1e-6
                coef = torch.full_like(denominator, self.config.max_grad_norm) / denominator
                coef = torch.where(coef != coef, coef, coef.clamp(max=1.0))
        for i, (name, p, g) in enumerate(zip(self.parameter_names, params, grads)):
            old_m, old_v, old_step = state[3*i:3*i+3]
            active = participation[i] if participation is not None else g is not None
            if g is None or active is False:
                new_params.append(p.clone())
                new_state.extend((old_m.clone(), old_v.clone(), old_step.clone()))
                continue
            options = self.options[name]
            if coef is not None:
                g = g * coef
            b1, b2 = options["betas"]
            step = old_step + 1
            # log/expm1 preserves small bias corrections even for beta close to
            # one. The log is a static scalar; no host reads of the step occur.
            sf = step.to(torch.float32)
            bc1 = -torch.expm1(sf * math.log(b1)) if b1 else sf * 0.0 + 1.0
            bc2 = -torch.expm1(sf * math.log(b2)) if b2 else sf * 0.0 + 1.0
            m = b1 * old_m + (1.0 - b1) * g
            v = b2 * old_v + (1.0 - b2) * (g * g)
            rate = options["lr"] if lr is None else lr
            if self.config.warmup_steps:
                rate = options["lr"] * torch.clamp(sf / float(self.config.warmup_steps), max=1.0)
            decayed = p * (1.0 - rate * options["weight_decay"] if lr is None else -(rate * options["weight_decay"]) + 1.0)
            updated = decayed - (rate / bc1) * m / (torch.sqrt(v) / torch.sqrt(bc2) + options["eps"])
            if isinstance(active, torch.Tensor):
                updated = torch.where(active, updated, p)
                m, v, step = (torch.where(active, new, old)
                              for new, old in ((m, old_m), (v, old_v), (step, old_step)))
            new_params.append(updated)
            new_state.extend((m, v, step))
        return tuple(new_params), tuple(new_state)


def prepare_optimizer(config, params, *, optimizer_state=None, momentum=None, aliases=None):
    """Validate named state and return reusable pure arithmetic plus ABI metadata."""
    config = config or SGDConfig()
    from .user_optimizer import TensorUpdateRule, prepare_tensor_rule
    if isinstance(config, TensorUpdateRule):
        # Step 12: user-written pure update; tied aliases are already canonical.
        return prepare_tensor_rule(config, _canonical(params)[0], optimizer_state=optimizer_state, momentum=momentum)
    from .muon import MuonConfig, prepare_muon
    if isinstance(config, MuonConfig):
        if momentum is not None:
            raise CaptureError("OPTIMIZER_STATE", "Muon requires optimizer_state; legacy momentum is SGD-only")
        return prepare_muon(config, _canonical(params)[0], optimizer_state=optimizer_state)
    if not isinstance(config, (SGDConfig, AdamWConfig)):
        raise CaptureError("OPTIMIZER", "expected SGDConfig, AdamWConfig or MuonConfig")
    config.validate()
    params, inferred_aliases = _canonical(params)
    aliases = aliases or inferred_aliases
    if optimizer_state is not None and momentum is not None:
        raise CaptureError("OPTIMIZER_STATE", "supply optimizer_state or legacy momentum, not both")
    options = {}
    if isinstance(config, SGDConfig):
        supplied = optimizer_state if optimizer_state is not None else (momentum or {})
        if set(supplied) - set(aliases):
            raise CaptureError("STATE", "momentum has unknown parameter keys")
        state = {}
        for name, p in params.items():
            entries = [supplied[alias] for alias, canonical in aliases.items() if canonical == name and alias in supplied]
            if entries and any(id(value) != id(entries[0]) for value in entries[1:]):
                raise CaptureError("STATE_ALIAS", f"shared parameter {name} has distinct momentum objects")
            state[name] = entries[0] if entries else torch.zeros_like(p)
        schema = tuple({"name": name, "parameter": name, "kind": "momentum", "dtype": "float32", "shape": list(p.shape)}
                       for name, p in params.items())
    else:
        if momentum is not None:
            raise CaptureError("OPTIMIZER_STATE", "AdamW requires optimizer_state; legacy momentum is SGD-only")
        if set(config.param_options) - set(aliases):
            raise CaptureError("OPTIMIZER", "parameter options name an unknown parameter")
        for name in params:
            entries = [config.param_options[alias] for alias, canonical in aliases.items()
                       if canonical == name and alias in config.param_options]
            if entries and any(entry != entries[0] for entry in entries[1:]):
                raise CaptureError("OPTIMIZER_ALIAS", f"tied parameter {name} has conflicting optimizer options")
            options[name] = {**config.options(), **(entries[0] if entries else {})}
        schema = tuple({"name": f"{kind}:{name}", "parameter": name, "kind": kind,
                        "dtype": "int64" if kind == "step" else "float32",
                        "shape": [] if kind == "step" else list(p.shape)}
                       for name, p in params.items() for kind in ("exp_avg", "exp_avg_sq", "step"))
        expected = {entry["name"] for entry in schema}
        if optimizer_state is not None and set(optimizer_state) != expected:
            raise CaptureError("OPTIMIZER_STATE", "AdamW state keys must exactly match the canonical named schema")
        supplied = optimizer_state if optimizer_state is not None else initialize_optimizer_state(params, config)
        state = {entry["name"]: supplied[entry["name"]] for entry in schema}
    for entry in schema:
        value, p = state[entry["name"]], params[entry["parameter"]]
        if isinstance(value, torch.Tensor) and (value.is_neg() or value.is_conj()):
            raise CaptureError("LAZY_VIEW", f"optimizer state {entry['name']}: resolve lazy negative/conjugate views")
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or tuple(value.shape) != tuple(entry["shape"]) or str(value.dtype).removeprefix("torch.") != entry["dtype"]
                or value.device != p.device or value.is_neg() or value.is_conj()):
            raise CaptureError("OPTIMIZER_STATE", f"state {entry['name']} does not match its tensor schema")
        if entry["kind"] == "step" and not 0 <= int(value.item()) < ADAMW_STEP_LIMIT:
            raise CaptureError("OPTIMIZER_STATE", f"AdamW input steps must be in [0,{ADAMW_STEP_LIMIT}); training lifetimes must preserve this bound")
        if entry["kind"] in {"exp_avg", "exp_avg_sq"}:
            if not bool(torch.isfinite(value).all()):
                raise CaptureError("OPTIMIZER_STATE", "AdamW restored moments must be finite")
            if entry["kind"] == "exp_avg_sq" and bool((value < 0).any()):
                raise CaptureError("OPTIMIZER_STATE", "AdamW restored second moments must be nonnegative")
    return PreparedOptimizer(config, tuple(params), state, options, schema)
