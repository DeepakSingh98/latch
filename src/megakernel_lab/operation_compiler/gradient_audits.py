"""Reusable ordinary-autograd checks for objective-gradient contributions.

Run on Modal CPU. Callables take an explicit parameter dictionary and return a
scalar objective, closing over the same readonly data and explicit initial RNG.
No capture graph or handwritten derivative is used by this oracle.
"""
from __future__ import annotations

import torch

from .numerical_reference import compare


def intermediate_contribution_audit(params, total_fn, terminal_fn, auxiliary_fn, *, weight):
    """Prove d(total)=d(terminal)+weight*d(auxiliary), and reject missing d(aux).

    The negative control keeps the auxiliary forward value but stops its
    gradient. Shared Tensor identities are retained when cloning parameters.
    """
    ids, unique, aliases = {}, {}, {}
    for name, value in params.items():
        canonical = ids.setdefault(id(value), name)
        unique.setdefault(canonical, value)
        aliases[name] = canonical

    def differentiated(function):
        canonical = {name: value.detach().clone().requires_grad_(True) for name, value in unique.items()}
        mapped = {name: canonical[key] for name, key in aliases.items()}
        loss = function(mapped)
        derivatives = torch.autograd.grad(loss, tuple(canonical.values()), allow_unused=True)
        return loss.detach(), {name: torch.zeros_like(value) if derivative is None else derivative.detach()
                               for (name, value), derivative in zip(canonical.items(), derivatives)}

    total, total_grad = differentiated(total_fn)
    terminal, terminal_grad = differentiated(terminal_fn)
    auxiliary, auxiliary_grad = differentiated(auxiliary_fn)
    loss_check = compare(total, terminal + weight * auxiliary, "loss")
    contributions = {name: compare(total_grad[name], terminal_grad[name] + weight * auxiliary_grad[name], "gradients")
                     for name in unique}
    omitted = {name: compare(terminal_grad[name], total_grad[name], "gradients") for name in unique}
    rejected = [name for name, result in omitted.items() if not result["passed"]]
    control = {"passed": bool(rejected), "rejected_parameters": rejected,
               "worst_error_to_bound_ratio": max(value["max_ratio"] for value in omitted.values()),
               "forward_value": float(terminal + weight * auxiliary),
               "meaning": "auxiliary value retained, auxiliary derivative stopped"}
    return {"passed": loss_check["passed"] and all(value["passed"] for value in contributions.values()) and control["passed"],
            "weight": weight, "loss_decomposition": loss_check, "gradient_additivity": contributions,
            "terminal_loss": float(terminal), "auxiliary_loss": float(auxiliary),
            "negative_control": control}
