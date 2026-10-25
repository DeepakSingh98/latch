"""User-written pure tensor weight-update rules (release step 12).

A researcher supplies one pure function

    update(parameters, gradients, optimizer_state, hyperparameters)
        -> (new_parameters, new_optimizer_state)

and passes ``tensor_update(update, ...)`` wherever a built-in optimizer config
is accepted by ``capture_training`` or ``capture_module``. Capture traces the
function with the same make_fx/decomposition path as the built-in SGD/AdamW
arithmetic in ``optim.py``. Its operations are appended to the captured
forward/loss/backward graph and lower into the same single native update.
No formula is special-cased: the compiler sees ordinary ATen operations.

Bounded interface (refused otherwise, with a CaptureError):

* ``parameters`` and ``gradients`` map canonical parameter names to FP32
  tensors. Tied parameters (one Tensor registered under several names) appear
  once under the canonical (first) name with their summed gradient. Every
  parameter receives a materialized gradient; an unused parameter's gradient
  is zero. There is no implicit None/participation signal.
* With ``forward_state=True``, a stateful model's newly returned state mapping
  is passed as a fifth argument, ``forward_state``. These detached tensors can
  carry participation masks or statistics computed by the forward pass. They
  are explicit graph dependencies, read-only like the other arguments; the
  update must not infer participation from gradient magnitude. Stateless
  captures refuse this option.
* ``optimizer_state`` maps user-chosen names to strided FP32 tensors (any
  shape) or int64 scalar counters. ``init_state(parameters, hyperparameters)``
  builds the initial values eagerly, once, outside capture; an explicit
  ``optimizer_state`` passed to capture replaces them and must match the same
  names, shapes and dtypes. A ``"<slot>:<parameter>"`` naming convention is
  recorded in the schema but not required.
* ``hyperparameters`` are static finite Python numbers baked into the program.
  Changing them requires recapture. Schedules can be written with a state
  counter.
* The function must be pure: it reads old parameters/state/gradients and
  returns new tensors. In-place mutation of any argument, Python control flow
  on tensor values, randomness, host reads and external side effects are
  refused by capture. All right-hand sides read the old versions; the runtime
  commits every new parameter/state value once after all graph reads.
* Returned dictionaries must contain exactly the input keys with unchanged
  shape and dtype. A returned input (unchanged value) is copied.
* Only flat ``capture_training``/``capture_module`` programs are admitted.
  Loop, sampled and conditional-region captures refuse a user rule.
  Differentiating through the update itself is out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import math
from typing import Any, Callable, Mapping

import torch

from .ir import CaptureError

_STATE_DTYPES = (torch.float32, torch.int64, torch.float64)


def _static_number(value):
    return (isinstance(value, (bool, int, float)) and not isinstance(value, torch.Tensor)
            and (isinstance(value, bool) or math.isfinite(value)))


@dataclass(frozen=True)
class TensorUpdateRule:
    """Configuration object wrapping a user update function; see module docstring."""
    update: Callable
    init_state: Callable | None = None
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    name: str = "user_rule"
    forward_state: bool = False

    def validate(self):
        if type(self.forward_state) is not bool:
            raise CaptureError("OPTIMIZER", "forward_state must be bool")
        if not callable(self.update):
            raise CaptureError("OPTIMIZER", "tensor_update requires a callable update function")
        if self.init_state is not None and not callable(self.init_state):
            raise CaptureError("OPTIMIZER", "init_state must be callable or None")
        if not isinstance(self.hyperparameters, Mapping):
            raise CaptureError("OPTIMIZER", "hyperparameters must be a mapping of names to Python numbers")
        for key, value in self.hyperparameters.items():
            if not isinstance(key, str):
                raise CaptureError("OPTIMIZER", "hyperparameter names must be strings")
            if not _static_number(value):
                raise CaptureError("OPTIMIZER", f"hyperparameter {key} must be a finite static Python number; "
                                   "tensor/runtime hyperparameters are not admitted")
        if not isinstance(self.name, str) or not self.name:
            raise CaptureError("OPTIMIZER", "rule name must be a nonempty string")

    def describe(self):
        try:
            source = inspect.getsource(self.update)
        except (OSError, TypeError):
            source = None
        return {"kind": "user_tensor_rule", "name": self.name,
                "function": getattr(self.update, "__qualname__", repr(self.update)),
                "module": getattr(self.update, "__module__", None),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source else None,
                "hyperparameters": dict(sorted(self.hyperparameters.items())),
                "forward_state": self.forward_state}


def tensor_update(update, *, init_state=None, hyperparameters=None, name=None, forward_state=False):
    """Public constructor: wrap a pure ``update`` function as an optimizer config."""
    rule = TensorUpdateRule(update, init_state, dict(hyperparameters or {}),
                            name or getattr(update, "__name__", "user_rule"), forward_state)
    rule.validate()
    return rule


def _check_state_tensor(name, value):
    if not isinstance(name, str) or not name:
        raise CaptureError("OPTIMIZER_STATE", "optimizer state names must be nonempty strings")
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name} must be a strided Tensor")
    if value.dtype not in _STATE_DTYPES or (value.dtype in (torch.int64, torch.float64) and value.ndim != 0):
        raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name}: FP32 tensors, int64 counters, or FP64 scalar coefficients only")
    if value.is_neg() or value.is_conj():
        raise CaptureError("LAZY_VIEW", f"optimizer state {name}: resolve lazy negative/conjugate views")
    if value.dtype in (torch.float32, torch.float64) and not bool(torch.isfinite(value).all()):
        raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name} must be finite")


def initialize_user_state(rule, params):
    """Run the user's eager init_state once on canonical parameters."""
    if rule.init_state is None:
        return {}
    with torch.no_grad():
        detached = {name: value.detach() for name, value in params.items()}
        result = rule.init_state(detached, dict(rule.hyperparameters))
    if not isinstance(result, Mapping):
        raise CaptureError("OPTIMIZER_STATE", "init_state must return a dict of named tensors")
    state = {}
    devices = {value.device for value in params.values()}
    for name, value in result.items():
        _check_state_tensor(name, value)
        if value.device not in devices:
            raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name} must share the parameters' device")
        for p in params.values():
            if value.numel() and p.numel() and value.untyped_storage().data_ptr() == p.untyped_storage().data_ptr():
                raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name} aliases a parameter; return a new tensor")
        state[name] = value.detach().clone()
    return state


@dataclass
class PreparedTensorRule:
    """Implements the PreparedOptimizer protocol consumed by capture."""
    config: TensorUpdateRule
    parameter_names: tuple[str, ...]
    state: dict[str, torch.Tensor]
    state_schema: tuple[dict[str, Any], ...]

    legacy_momentum = False
    uses_participation = False
    runtime_lr = False

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
                "optimizer": self.config.describe(),
                "optimizer_state_commit": "read old versions; commit every new parameter/optimizer-state output once after all graph reads",
                "optimizer_gradient_policy": "every canonical parameter receives a materialized gradient (zero when unused)",
                "optimizer_interface": ("update(parameters, gradients, optimizer_state, hyperparameters" +
                    (", forward_state" if self.config.forward_state else "") +
                    ") -> (new_parameters, new_optimizer_state)")}

    def update(self, params, grads, state, participation=None, inputs=(), lr=None, forward_state=None):
        if len(params) != len(self.parameter_names) or len(grads) != len(params) or len(state) != len(self.state_names):
            raise CaptureError("OPTIMIZER_ABI", "optimizer parameter/gradient/state arity mismatch")
        if inputs or lr is not None:
            raise CaptureError("OPTIMIZER_ABI", "user tensor rules take no runtime optimizer inputs")
        grads = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, grads))
        if self.config.forward_state != (forward_state is not None):
            raise CaptureError("OPTIMIZER_ABI", "forward_state rule requires explicit stateful model outputs")
        arguments = (*params, *grads, *state, *(forward_state or {}).values())
        versions = [value._version for value in arguments]
        p_in = dict(zip(self.parameter_names, params))
        g_in = dict(zip(self.parameter_names, grads))
        s_in = dict(zip(self.state_names, state))
        call_args = (dict(p_in), dict(g_in), dict(s_in), dict(self.config.hyperparameters))
        forward_inputs = {}
        if self.config.forward_state:
            # Forward-produced participation and statistics are explicit SSA
            # values. Detach them: the update is never differentiated through.
            forward_inputs = {k: v.detach() for k, v in forward_state.items()}
            call_args += (forward_inputs,)
        result = self.config.update(*call_args)
        if any(value._version != version for value, version in zip(arguments, versions)):
            raise CaptureError("OPTIMIZER_MUTATION", "the update mutated a parameter, gradient or state argument "
                               "in place; return new tensors instead")
        if not isinstance(result, tuple) or len(result) != 2 or not all(isinstance(x, Mapping) for x in result):
            raise CaptureError("OPTIMIZER_RETURN", "update must return (new_parameters_dict, new_optimizer_state_dict)")
        new_params, new_state = result
        if set(new_params) != set(p_in):
            raise CaptureError("OPTIMIZER_RETURN", "new_parameters keys must exactly match the canonical parameters")
        if set(new_state) != set(s_in):
            raise CaptureError("OPTIMIZER_RETURN", "new_optimizer_state keys must exactly match the state names")
        # detach() creates a new Tensor object with no _base while retaining
        # storage. Storage identity works for fake tensors without data access
        # and keeps this check linear in the number of optimizer slots.
        original_storage = {value.untyped_storage()._cdata for value in arguments}

        def checked(kind, name, before, after):
            if not isinstance(after, torch.Tensor):
                raise CaptureError("OPTIMIZER_RETURN", f"{kind} {name} must be a Tensor")
            if after.shape != before.shape or after.dtype != before.dtype or after.device != before.device:
                raise CaptureError("OPTIMIZER_RETURN", f"{kind} {name} must keep shape {tuple(before.shape)}, "
                                   f"dtype {before.dtype} and device")
            # Never expose an input binding (or its view) as a committed output.
            return after.clone() if (after.untyped_storage()._cdata in original_storage
                                     or after._base is not None) else after

        return (tuple(checked("parameter", name, p_in[name], new_params[name]) for name in self.parameter_names),
                tuple(checked("state", name, s_in[name], new_state[name]) for name in self.state_names))


def prepare_tensor_rule(rule, params, *, optimizer_state=None, momentum=None):
    """Validate the user's state against its own initializer and return the protocol object."""
    rule.validate()
    if momentum is not None:
        raise CaptureError("OPTIMIZER_STATE", "user tensor rules use optimizer_state; legacy momentum is SGD-only")
    initial = initialize_user_state(rule, params)
    if optimizer_state is not None:
        if set(optimizer_state) != set(initial):
            raise CaptureError("OPTIMIZER_STATE", "optimizer_state keys must exactly match init_state's names")
        state = {}
        for name, value in initial.items():
            supplied = optimizer_state[name]
            _check_state_tensor(name, supplied)
            if supplied.shape != value.shape or supplied.dtype != value.dtype or supplied.device != value.device:
                raise CaptureError("OPTIMIZER_STATE", f"optimizer state {name} does not match its initializer schema")
            state[name] = supplied
    else:
        state = initial
    schema = []
    for name, value in state.items():
        slot, _, owner = name.rpartition(":")
        schema.append({"name": name, "parameter": owner if slot and owner in params else None,
                       "kind": "user", "dtype": str(value.dtype).removeprefix("torch."),
                       "shape": list(value.shape)})
    return PreparedTensorRule(rule, tuple(params), state, tuple(schema))
