"""Independent ordinary-autograd/torch.optim optimizer references; cloud only.

No captured optimizer arithmetic or native graph executes in this reference.
Tied aliases and None gradients retain actual torch.optim.AdamW semantics.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .optim import AdamWConfig, initialize_optimizer_state


def adamw_update_from_gradients(params, grads, optimizer_state, config=None, *, dtype=None):
    """Actual torch.optim update with caller-provided identical gradients.

    This diagnostic isolates optimizer arithmetic from AD/reduction differences;
    it intentionally performs no model forward or gradient computation.
    Parameter and gradient dictionaries are already canonical.
    """
    config = config or AdamWConfig()
    config.validate()
    p = {name:value.detach().to(dtype=dtype or value.dtype).clone().requires_grad_(True)
         for name,value in params.items()}
    groups = [{"params":[value], **config.param_options.get(name,{})} for name,value in p.items()]
    opt = torch.optim.AdamW(groups,lr=config.lr,betas=config.betas,eps=config.eps,
                            weight_decay=config.weight_decay,amsgrad=False,maximize=False,foreach=False,fused=False)
    for name,value in p.items():
        gradient=grads[name]
        value.grad=None if gradient is None else gradient.detach().to(value.dtype).clone()
        opt.state[value].update(exp_avg=optimizer_state[f"exp_avg:{name}"].detach().to(value.dtype).clone(),
                               exp_avg_sq=optimizer_state[f"exp_avg_sq:{name}"].detach().to(value.dtype).clone(),
                               step=torch.tensor(float(optimizer_state[f"step:{name}"].item()),dtype=torch.float32,device=value.device))
    opt.step()
    state={}
    for name,value in p.items():
        for kind in ("exp_avg","exp_avg_sq"):
            state[f"{kind}:{name}"]=opt.state[value][kind].detach().clone()
        state[f"step:{name}"]=opt.state[value]["step"].to(torch.int64).detach().clone()
    return {"params":{name:value.detach().clone() for name,value in p.items()},"optimizer_state":state}


def adamw_reference_step(loss_fn, params, optimizer_state, inputs, *, state=None,
                         config=None, optimizer=None, dtype=None):
    config = config or optimizer or AdamWConfig()
    config.validate()
    ids, unique, aliases = {}, {}, {}
    for name, value in params.items():
        canonical = ids.setdefault(id(value), name)
        aliases[name] = canonical
        if canonical not in unique:
            unique[canonical] = value.detach().to(dtype=dtype or value.dtype).clone().requires_grad_(True)
    mapping = {name: unique[canonical] for name, canonical in aliases.items()}
    data = tuple(value.detach().to(dtype=dtype) if dtype and value.is_floating_point() else value.detach()
                 for value in inputs)
    old_state = None if state is None else {name: value.detach().clone() for name, value in state.items()}
    result = loss_fn(mapping, *data) if state is None else loss_fn(mapping, old_state, *data)
    loss, new_state = (result, {}) if state is None else result
    gradients = (torch.autograd.grad(loss, tuple(unique.values()), allow_unused=True, materialize_grads=False)
                 if loss.requires_grad else (None,) * len(unique))
    groups = []
    for name, value in unique.items():
        overrides = [config.param_options[alias] for alias, canonical in aliases.items()
                     if canonical == name and alias in config.param_options]
        groups.append({"params": [value], **(overrides[0] if overrides else {})})
    opt = torch.optim.AdamW(groups, lr=config.lr, betas=config.betas, eps=config.eps,
                            weight_decay=config.weight_decay, amsgrad=False, maximize=False,
                            foreach=False, fused=False)
    old = optimizer_state if optimizer_state is not None else initialize_optimizer_state(params, config)
    participation = {}
    for (name, value), gradient in zip(unique.items(), gradients):
        participation[name] = gradient is not None
        value.grad = None if gradient is None else gradient.detach().clone()
        opt.state[value].update(exp_avg=old[f"exp_avg:{name}"].detach().to(value.dtype).clone(),
                                exp_avg_sq=old[f"exp_avg_sq:{name}"].detach().to(value.dtype).clone(),
                                step=torch.tensor(float(old[f"step:{name}"].item()), dtype=torch.float32,
                                                  device=value.device))
    opt.step()
    new_optimizer = {}
    for name, value in unique.items():
        new_optimizer[f"exp_avg:{name}"] = opt.state[value]["exp_avg"].detach().clone()
        new_optimizer[f"exp_avg_sq:{name}"] = opt.state[value]["exp_avg_sq"].detach().clone()
        new_optimizer[f"step:{name}"] = opt.state[value]["step"].to(torch.int64).detach().clone()
    return {"loss": loss.detach(),
            "grads": {name: torch.zeros_like(value) if gradient is None else gradient.detach().clone()
                      for (name, value), gradient in zip(unique.items(), gradients)},
            "params": {name: value.detach().clone() for name, value in unique.items()},
            "optimizer_state": new_optimizer,
            "state": {name: value.detach().clone() for name, value in new_state.items()},
            "gradient_participation": participation}


def adamw_fp64_formula(params, grads, optimizer_state, config):
    """Small independent scalar-formula oracle, not used by compiler/reference.

    Bias powers use Python FP64 counters/constants; this intentionally does not
    reuse the native log/expm1 expression or its operation ordering.
    """
    new_params, new_state = {}, {}
    for name, value in params.items():
        p = value.detach().double()
        m, v = (optimizer_state[f"{kind}:{name}"].detach().double() for kind in ("exp_avg", "exp_avg_sq"))
        step = int(optimizer_state[f"step:{name}"].item())
        g = grads[name]
        if g is None:
            np, nm, nv, nt = p, m, v, step
        else:
            options = {**config.options(), **config.param_options.get(name, {})}
            b1, b2 = options["betas"]
            nt = step + 1
            nm = m * b1 + g.double() * (1-b1)
            nv = v * b2 + g.double().square() * (1-b2)
            mh, vh = nm / (1-b1**nt), nv / (1-b2**nt)
            np = p * (1-options["lr"]*options["weight_decay"]) - options["lr"] * mh / (vh.sqrt()+options["eps"])
        new_params[name] = np
        new_state[f"exp_avg:{name}"] = nm
        new_state[f"exp_avg_sq:{name}"] = nv
        new_state[f"step:{name}"] = torch.tensor(nt, dtype=torch.int64, device=p.device)
    return new_params, new_state


def compare_optimizer_step(actual, expected, old_params, *, arithmetic="float32"):
    from .numerical_reference import compare_arithmetic
    policy = json.loads(Path(__file__).with_name("STAGE4_POLICY.json").read_text())
    bounds = policy["mixed_precision" if arithmetic == "bfloat16" else "adamw"]
    checks = {"loss": compare_arithmetic(actual["loss"], expected["loss"], "loss", arithmetic)}
    for group, category in (("grads", "gradients"), ("params", "parameters")):
        checks[group] = {name: compare_arithmetic(value, expected[group][name], category, arithmetic)
                         for name, value in actual[group].items()}
    checks["updates"] = {name: compare_arithmetic(actual["params"][name]-value,
                          expected["params"][name]-value, "updates", arithmetic)
                         for name, value in old_params.items()}
    checks["optimizer_state"] = {}
    for name, value in actual["optimizer_state"].items():
        ref = expected["optimizer_state"][name]
        if name.startswith("step:"):
            check = {"passed": value.dtype == ref.dtype and value.shape == ref.shape and torch.equal(value, ref),
                     "exact_integer": True}
        else:
            bound = bounds["second_moment" if name.startswith("exp_avg_sq:") else "first_moment"]
            a, b = value.detach().double(), ref.detach().double()
            delta = (a-b).abs()
            ratio = delta / (bound["atol"]+bound["rtol"]*b.abs())
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            check = {"passed": value.shape == ref.shape and finite and bool((ratio <= 1).all()),
                     "finite": finite, "max_ratio": float(ratio.max()) if a.numel() else 0,
                     "max_abs": float(delta.max()) if a.numel() else 0,
                     "elements": a.numel(), "bounds": bound}
        checks["optimizer_state"][name] = check
    checks["state"] = {name: {"passed": value.shape == expected["state"][name].shape
                                        and value.dtype == expected["state"][name].dtype
                                        and torch.equal(value, expected["state"][name]), "exact": True}
                       for name, value in actual.get("state", {}).items()}
    return checks
