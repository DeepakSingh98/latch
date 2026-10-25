"""Release step 07: ordinary modules mixing trainable and frozen parameters.

A registered parameter whose ``requires_grad`` is False when a module is
captured is *frozen*. Frozen weights are bound exactly like registered buffers:
a private snapshot taken at capture, passed to the captured program as an
explicit read-only input. They participate in forward and backward arithmetic,
so gradients still flow THROUGH a frozen layer to trainable layers before it,
but they are not differentiation inputs. They therefore receive no parameter
gradient output, no optimizer state, no weight decay and no update output.

Trainability is fixed at capture. Toggling ``requires_grad`` or editing a
frozen value in the Python module afterwards does not affect an existing
captured program or native artifact; recapture and rebuild instead, because
the native argument list and generated code differ. Native checkpoints keep
frozen values out of the mutable state dictionary and bind them through the
immutable-bindings identity, so a snapshot cannot be restored into a program
compiled against different frozen weights.

Tied entries share one Python Tensor and therefore one ``requires_grad``.
They stay canonicalized by identity, as for trainable parameters and buffers.
Distinct Tensors that share storage across the trainable/frozen boundary are
refused: the frozen snapshot would silently go stale as the trainable view
updated. An entirely frozen model has nothing to train and is refused with an
explicit message; use ``capture_module_inference`` for forward-only programs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ir import CaptureError

POLICY = ("trainability read from requires_grad at capture; frozen parameters are private read-only "
          "snapshots bound like buffers; no gradient, optimizer state, weight decay or update; "
          "changing trainability or frozen values requires recapture and rebuild")


def _storage_key(value):
    if not isinstance(value, torch.Tensor) or not value.numel():
        return None
    return str(value.device), value.untyped_storage().data_ptr()


@dataclass
class ParameterPartition:
    """Trainable aliases (live Tensors) and frozen canonical snapshots."""
    trainable: dict[str, torch.Tensor]
    frozen: dict[str, torch.Tensor]          # canonical name -> private snapshot
    frozen_aliases: dict[str, str]           # every frozen alias -> canonical name
    frozen_live: dict[str, torch.Tensor] = field(default_factory=dict)  # canonical -> user Tensor

    @property
    def frozen_names(self):
        return tuple(self.frozen)

    def bound(self, canonical_values):
        """Map every frozen alias to the supplied per-canonical binding."""
        return {alias: canonical_values[canonical] for alias, canonical in self.frozen_aliases.items()}

    def describe(self, optimizer=None):
        """Static accounting of state avoided; values are elements/bytes, not timings."""
        unique_trainable = {id(value): value for value in self.trainable.values()}.values()
        trainable_elements = sum(value.numel() for value in unique_trainable)
        frozen_elements = sum(value.numel() for value in self.frozen.values())
        frozen_bytes = sum(value.numel() * value.element_size() for value in self.frozen.values())
        kind = type(optimizer).__name__ if optimizer is not None else None
        # Per frozen FP32 element a trainable binding would add: a gradient
        # output, a new-parameter output, and optimizer state plus its new value.
        state_copies = {"SGDConfig": 1, "AdamWConfig": 2}.get(kind)
        result = {"policy": POLICY, "frozen_names": list(self.frozen),
                  "frozen_aliases": dict(self.frozen_aliases),
                  "trainable_elements": trainable_elements, "frozen_elements": frozen_elements,
                  "frozen_bytes": frozen_bytes, "optimizer": kind}
        if state_copies is not None:
            per_element = 4 * (2 + 2 * state_copies)
            result["avoided_bytes_upper_bound"] = frozen_elements * per_element
            result["avoided_breakdown_per_frozen_element_bytes"] = {
                "gradient_output": 4, "new_parameter_output": 4,
                "optimizer_state": 4 * state_copies, "new_optimizer_state": 4 * state_copies}
            if kind == "AdamWConfig":
                result["avoided_step_counters"] = len(self.frozen)
        return result


def partition_parameters(named, *, mutable=()):
    """Split ``named_parameters(remove_duplicate=False)`` by requires_grad.

    ``mutable`` is an iterable of other mutable Tensors (optimizer or explicit
    model state) whose storage a frozen snapshot must not share.
    """
    named = dict(named)
    trainable, frozen, aliases, live, identities = {}, {}, {}, {}, {}
    for name, value in named.items():
        if value.requires_grad:
            trainable[name] = value
            continue
        canonical = identities.setdefault(id(value), name)
        aliases[name] = canonical
        if canonical not in frozen:
            if value.layout != torch.strided:
                raise CaptureError("FROZEN_PARAMETER", f"{name}: frozen parameters must use strided storage")
            if value.is_neg() or value.is_conj():
                raise CaptureError("LAZY_VIEW", f"frozen parameter {name}: resolve lazy views before capture")
            live[canonical] = value
            frozen[canonical] = value.detach().clone()
    if named and not trainable:
        raise CaptureError("ALL_FROZEN", "every registered parameter has requires_grad=False, so there is "
                           "nothing to train; set requires_grad=True on at least one parameter before capture, "
                           "or use capture_module_inference for a forward-only program")
    guarded = {}
    for name, value in trainable.items():
        key = _storage_key(value)
        if key is not None:
            guarded.setdefault(key, f"trainable parameter {name}")
    for index, value in enumerate(mutable):
        key = _storage_key(value)
        if key is not None:
            guarded.setdefault(key, f"mutable optimizer/model state {index}")
    for canonical, value in live.items():
        other = guarded.get(_storage_key(value))
        if other is not None:
            raise CaptureError("FROZEN_STORAGE", f"frozen parameter {canonical} shares storage with {other}; "
                               "its read-only snapshot would go stale. Freeze or train both, or clone it")
    return ParameterPartition(trainable, frozen, aliases, live)


def check_optimizer_names(config, partition, optimizer_state=None):
    """Refuse optimizer options or supplied state naming frozen parameters."""
    frozen = set(partition.frozen_aliases)
    options = getattr(config, "param_options", None) or {}
    named = set(options) & frozen
    for key in optimizer_state or {}:
        if key in frozen or (":" in key and key.split(":", 1)[1] in frozen):
            named.add(key)
    if named:
        raise CaptureError("FROZEN_OPTIMIZER", f"optimizer options/state name frozen parameters {sorted(named)}; "
                           "frozen parameters have no optimizer state or update")
