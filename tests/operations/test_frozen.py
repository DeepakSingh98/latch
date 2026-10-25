"""Release step 07 frozen-parameter contracts; run on a CPU worker (torch 2.9.1).

Every numerical check compares the captured update with an independent
ordinary-PyTorch reference: a deep copy of the module with the same
requires_grad flags, real autograd and an unmodified torch.optim optimizer over
``model.parameters()`` (PyTorch skips parameters whose gradient is None).
Comparisons are same-current-state: each update starts the reference from the
captured program's current trainable/optimizer state, so accumulation error is
not confused with trajectory drift. Frozen values are compared bitwise.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import traceback

import torch

from megakernel_lab.operation_compiler.api import _immutable_identity, capture_module
from megakernel_lab.operation_compiler.capture import capture_training, module_loss
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig

POLICY = json.loads((Path(__file__).resolve().parent / "fixtures" / "NUMERICAL_POLICY.json").read_text())
SHAPES = {
    # Shape A: used while selecting the implementation.
    "A": {"stack": (16, 32, 32, 32, 8), "tied": (16, 24, 8), "batch": 4},
    # Shape B: reserved in research/latch_release/step07/RESERVED_SHAPE.json
    # before any run; evaluated only after the implementation is selected.
    "B": {"stack": (24, 40, 56, 40, 12), "tied": (20, 36, 12), "batch": 6},
}
FREEZE = {
    "first": ("layers.0.weight", "layers.0.bias"),
    "middle": ("layers.1.weight", "layers.1.bias"),
    "output": ("layers.3.weight", "layers.3.bias"),
    "mixed": ("layers.0.weight", "layers.2.bias", "layers.3.weight"),
    "none": (),
    "tied_frozen": ("left.weight",),
    "tied_trainable": ("inp.weight", "inp.bias", "out.weight", "out.bias"),
}
OPTIMIZERS = {
    "sgd": lambda: SGDConfig(lr=0.05, momentum=0.9, weight_decay=0.01),
    "adamw": lambda: AdamWConfig(lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
}


class Stack(torch.nn.Module):
    """Linear/tanh stack with a registered buffer between the first two layers."""
    def __init__(self, dims):
        super().__init__()
        self.layers = torch.nn.ModuleList(torch.nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.register_buffer("shift", torch.linspace(-0.1, 0.1, dims[1]))

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index == 0:
                x = x + self.shift
            if index < len(self.layers) - 1:
                x = torch.tanh(x)
        return x


class Tied(torch.nn.Module):
    """``right.weight`` is the same Parameter object as ``left.weight``."""
    def __init__(self, dims):
        super().__init__()
        d_in, width, d_out = dims
        self.inp = torch.nn.Linear(d_in, width)
        self.left = torch.nn.Linear(width, width, bias=False)
        self.mid = torch.nn.Linear(width, width)
        self.right = torch.nn.Linear(width, width, bias=False)
        self.right.weight = self.left.weight
        self.out = torch.nn.Linear(width, d_out)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        h = torch.tanh(self.left(h))
        h = torch.tanh(self.mid(h))
        h = torch.tanh(self.right(h))
        return self.out(h)


def objective(prediction, target):
    return (prediction - target).square().mean()


def make_model(case, shape="A", seed=7031):
    """Deterministic CPU construction; frozen flags applied by name (all aliases)."""
    torch.manual_seed(seed)
    tied = case.startswith("tied")
    model = Tied(SHAPES[shape]["tied"]) if tied else Stack(SHAPES[shape]["stack"])
    frozen = set(FREEZE[case])
    for name, value in model.named_parameters(remove_duplicate=False):
        if name in frozen:
            value.requires_grad_(False)
    return model


def batches(case, shape="A", count=6, seed=9100, device="cpu"):
    dims = SHAPES[shape]["tied" if case.startswith("tied") else "stack"]
    generator = torch.Generator().manual_seed(seed)
    batch = SHAPES[shape]["batch"]
    return [(torch.randn(batch, dims[0], generator=generator).to(device),
             torch.randn(batch, dims[-1], generator=generator).to(device)) for _ in range(count)]


def compare(actual, expected, category):
    policy = POLICY[category]
    if tuple(actual.shape) != tuple(expected.shape):
        return {"passed": False, "shape_mismatch": [list(actual.shape), list(expected.shape)]}
    a, e = actual.detach().double().cpu(), expected.detach().double().cpu()
    ratio = (a - e).abs() / (policy["atol"] + policy["rtol"] * e.abs())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    return {"passed": finite and bool((ratio <= 1).all()), "max_ratio": float(ratio.max()) if a.numel() else 0.0,
            "max_abs": float((a - e).abs().max()) if a.numel() else 0.0, "policy": category}


def torch_optimizer(config, parameters):
    if isinstance(config, SGDConfig):
        return torch.optim.SGD(parameters, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    return torch.optim.AdamW(parameters, lr=config.lr, betas=config.betas, eps=config.eps,
                             weight_decay=config.weight_decay)


def reference_step(template, config, params, optimizer_state, x, y, *, gradients=None, dtype=None):
    """One ordinary eager update starting from the supplied trainable/optimizer state.

    ``template`` holds the original frozen values; trainable values are copied in.
    Returns loss, gradients and new values keyed by canonical parameter name.
    Diagnostics only: ``gradients`` replaces autograd's gradients before the
    unmodified torch.optim step (isolating optimizer arithmetic from gradient
    accumulation order); ``dtype=torch.float64`` evaluates a CPU FP64 reference.
    """
    model = copy.deepcopy(template)
    if dtype is not None:
        model = model.to(device="cpu", dtype=dtype)
        x, y = x.to("cpu", dtype), y.to("cpu", dtype)
        params = {k: v.to("cpu", dtype) for k, v in params.items()}
        optimizer_state = {k: v.cpu() if k.startswith("step:") else v.to("cpu", dtype)
                           for k, v in optimizer_state.items()}
    named = dict(model.named_parameters())          # canonical (first) names; frozen included
    with torch.no_grad():
        for name, value in params.items():
            named[name].copy_(value)
    optimizer = torch_optimizer(config, model.parameters())
    for name, value in named.items():
        if not value.requires_grad:
            continue
        if isinstance(config, SGDConfig):
            optimizer.state[value]["momentum_buffer"] = optimizer_state[name].detach().clone()
        else:
            optimizer.state[value] = {
                "step": torch.tensor(float(optimizer_state["step:" + name].item()), dtype=torch.float32),
                "exp_avg": optimizer_state["exp_avg:" + name].detach().clone(),
                "exp_avg_sq": optimizer_state["exp_avg_sq:" + name].detach().clone()}
    loss = objective(model(x), y)
    if gradients is None:
        loss.backward()
    else:
        for name, value in named.items():
            if value.requires_grad:
                value.grad = gradients[name].detach().to(value.dtype).clone()
    grads = {name: value.grad.detach().clone() for name, value in named.items() if value.grad is not None}
    optimizer.step()
    state = {}
    for name, value in named.items():
        if value in optimizer.state:
            entry = optimizer.state[value]
            if isinstance(config, SGDConfig):
                state[name] = entry["momentum_buffer"].detach().clone()
            else:
                state.update({"exp_avg:" + name: entry["exp_avg"].detach().clone(),
                              "exp_avg_sq:" + name: entry["exp_avg_sq"].detach().clone(),
                              "step:" + name: entry["step"].detach().clone()})
    return {"loss": loss.detach(), "grads": grads,
            "params": {name: value.detach().clone() for name, value in named.items() if value.requires_grad},
            "frozen": {name: value.detach().clone() for name, value in named.items() if not value.requires_grad},
            "optimizer_state": state, "optimizer_state_entries": len(optimizer.state)}


def compare_update(observed, expected, frozen_initial):
    """Policy comparisons of one update plus exact frozen and structural checks."""
    result = {"loss": compare(observed["loss"], expected["loss"], "loss"),
              "grads": {k: compare(observed["grads"][k], v, "gradients") for k, v in expected["grads"].items()},
              "params": {k: compare(observed["params"][k], v, "parameters") for k, v in expected["params"].items()}}
    states = {}
    for name, value in expected["optimizer_state"].items():
        got = observed["optimizer_state"][name]
        if name.startswith("step:"):
            states[name] = {"passed": float(got.item()) == float(value.item())}
        else:
            states[name] = compare(got, value, "momentum")
    result["optimizer_state"] = states
    result["structure"] = {
        "passed": (set(observed["grads"]) == set(expected["grads"]) == set(expected["params"])
                   and set(observed["optimizer_state"]) == set(expected["optimizer_state"])
                   and not set(observed["grads"]) & set(frozen_initial)),
        "trainable": sorted(observed["grads"]), "frozen": sorted(frozen_initial)}
    result["reference_frozen_exact"] = {
        "passed": all(torch.equal(expected["frozen"][k], v) for k, v in frozen_initial.items())}
    return result


def passed(value):
    if isinstance(value, dict):
        return value.get("passed", True) is not False and all(passed(v) for v in value.values())
    return True


def captured_trajectory(case, optimizer, shape="A", steps=6):
    """Capture through the normal module API and evolve the captured graph."""
    torch.set_num_threads(1)
    config = OPTIMIZERS[optimizer]()
    model = make_model(case, shape)
    template = copy.deepcopy(model)
    data = batches(case, shape, steps)
    captured = capture_module(model, objective, (data[0][0],), (data[0][1],), optimizer=config)
    indices = captured.input_indices
    frozen_initial = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    args = list(captured.example_args)
    n, names, state_names = len(captured.parameter_names), captured.parameter_names, captured.optimizer_state_names
    record = {"case": case, "optimizer": optimizer, "shape": shape, "parameter_names": list(names),
              "frozen_names": sorted(indices.get("frozen", {})), "steps": []}
    for step, (x, y) in enumerate(data):
        args[indices["inputs"][0]], args[indices["targets"][0]] = x, y
        observed = captured.unpack(captured.graph_module(*args))
        params = dict(zip(names, args[:n]))
        opt_state = dict(zip(state_names, args[n:n + len(state_names)]))
        expected = reference_step(template, config, params, opt_state, x, y)
        row = compare_update(observed, expected, frozen_initial)
        row["captured_frozen_exact"] = {"passed": all(
            torch.equal(args[index], frozen_initial[name]) for name, index in indices.get("frozen", {}).items())}
        row["no_frozen_optimizer_state"] = {"passed": expected["optimizer_state_entries"] == n and not any(
            name.split(":")[-1] in frozen_initial for name in state_names)}
        record["steps"].append({"step": step, "passed": passed(row), **row})
        for i, name in enumerate(names):
            args[i] = observed["params"][name]
        for j, name in enumerate(state_names):
            args[n + j] = observed["optimizer_state"][name]
    record["module_frozen_untouched"] = {"passed": all(
        torch.equal(dict(model.named_parameters())[k], v) for k, v in frozen_initial.items())}
    record["passed"] = passed(record)
    return record


def _refusal(fn, codes):
    try:
        fn()
    except CaptureError as exc:
        assert exc.code in codes, (exc.code, str(exc))
        return {"code": exc.code, "message": str(exc)}
    raise AssertionError("unsupported frozen configuration unexpectedly captured")


def run(shape="A"):
    torch.set_num_threads(1)
    results = {}

    def check(name, fn):
        try:
            value = fn()
            ok = passed(value) if isinstance(value, dict) else True
            results[name] = {"passed": ok, **({"detail": value} if isinstance(value, dict) else {})}
        except Exception:
            results[name] = {"passed": False, "traceback": traceback.format_exc()}

    for case in ("first", "middle", "output", "mixed", "none", "tied_frozen", "tied_trainable"):
        for optimizer in OPTIMIZERS:
            check(f"trajectory:{case}:{optimizer}", lambda c=case, o=optimizer: captured_trajectory(c, o, shape))

    def gradient_through_frozen():
        # The first layer's gradient exists only through later frozen layers.
        for case in ("middle", "output"):
            model = make_model(case, shape)
            x, y = batches(case, shape, 1)[0]
            captured = capture_module(model, objective, (x,), (y,))
            grads = captured.unpack(captured.graph_module(*captured.example_args))["grads"]
            norm = float(grads["layers.0.weight"].norm())
            assert norm > 1e-6, (case, norm)
        return {"passed": True}

    check("gradient_reaches_layers_before_frozen", gradient_through_frozen)

    def tied_identity():
        model = make_model("tied_frozen", shape)
        x, y = batches("tied_frozen", shape, 1)[0]
        captured = capture_module(model, objective, (x,), (y,))
        assert captured.input_indices["frozen"].keys() == {"left.weight"}
        assert captured.ir.metadata["module_api"]["frozen_aliases"] == {"left.weight": "left.weight",
                                                                      "right.weight": "left.weight"}
        model = make_model("tied_trainable", shape)
        captured = capture_module(model, objective, (x,), (y,))
        assert captured.parameter_aliases["right.weight"] == "left.weight"
        assert "right.weight" not in captured.parameter_names
        return {"passed": True}

    check("tied_identity_preserved", tied_identity)

    def frozen_fixed_at_capture():
        model = make_model("middle", shape)
        x, y = batches("middle", shape, 1)[0]
        captured = capture_module(model, objective, (x,), (y,))
        before = captured.unpack(captured.graph_module(*captured.example_args))
        with torch.no_grad():
            model.layers[1].weight.fill_(3.0)          # edit the user's frozen weight
        model.layers[1].bias.requires_grad_(True)      # and toggle trainability
        after = captured.unpack(captured.graph_module(*captured.example_args))
        assert torch.equal(before["loss"], after["loss"]), "frozen snapshot followed a later module edit"
        assert "layers.1.bias" not in captured.parameter_names, "trainability changed without recapture"
        recaptured = capture_module(model, objective, (x,), (y,))
        assert "layers.1.bias" in recaptured.parameter_names
        assert set(recaptured.input_indices["frozen"]) == {"layers.1.weight"}
        return {"passed": True}

    check("trainability_and_values_fixed_at_capture", frozen_fixed_at_capture)

    def refusals():
        x, y = batches("first", shape, 1)[0]
        detail = {}
        model = make_model("none", shape)
        for value in model.parameters():
            value.requires_grad_(False)
        detail["all_frozen"] = _refusal(lambda: capture_module(model, objective, (x,), (y,)), {"ALL_FROZEN"})
        detail["all_frozen_module_loss"] = _refusal(
            lambda: module_loss(model, lambda m, x, y: objective(m(x), y)), {"ALL_FROZEN"})
        model = make_model("none", shape)
        model.extra = torch.nn.Parameter(model.layers[0].weight.detach()[:2], requires_grad=False)
        detail["frozen_view_of_trainable"] = _refusal(
            lambda: capture_module(model, objective, (x,), (y,)), {"FROZEN_STORAGE"})
        model = make_model("first", shape)
        config = AdamWConfig(param_options={"layers.0.weight": {"weight_decay": 0.0}})
        detail["optimizer_options_name_frozen"] = _refusal(
            lambda: capture_module(model, objective, (x,), (y,), optimizer=config), {"FROZEN_OPTIMIZER"})
        state = {name: torch.zeros_like(value) for name, value in model.named_parameters()}
        detail["optimizer_state_names_frozen"] = _refusal(
            lambda: capture_module(model, objective, (x,), (y,), optimizer_state=state), {"FROZEN_OPTIMIZER"})
        return detail

    check("explicit_refusals", refusals)

    def module_loss_adapter():
        model = make_model("middle", shape)
        x, y = batches("middle", shape, 1)[0]
        fn, params = module_loss(model, lambda m, x, y: objective(m(x), y))
        assert not set(params) & {"model.layers.1.weight", "model.layers.1.bias"}
        captured = capture_training(fn, params, None, (x, y), SGDConfig(lr=0.05, momentum=0.9, weight_decay=0.01))
        roles = [captured.ir.values[name].role for name in captured.ir.inputs]
        assert roles.count("constant_input") == 3, roles     # two frozen tensors plus the buffer
        observed = captured.unpack(captured.graph_module(*captured.example_args))
        expected = reference_step(model, SGDConfig(lr=0.05, momentum=0.9, weight_decay=0.01),
                                  {k.removeprefix("model."): v for k, v in params.items()},
                                  {k.removeprefix("model."): torch.zeros_like(v) for k, v in params.items()}, x, y)
        observed = {"loss": observed["loss"],
                    "grads": {k.removeprefix("model."): v for k, v in observed["grads"].items()},
                    "params": {k.removeprefix("model."): v for k, v in observed["params"].items()},
                    "optimizer_state": {k.removeprefix("model."): v for k, v in observed["optimizer_state"].items()}}
        frozen = {k: v.detach().clone() for k, v in model.named_parameters() if not v.requires_grad}
        return compare_update(observed, expected, frozen)

    check("module_loss_adapter", module_loss_adapter)

    def checkpoint_identity():
        model = make_model("mixed", shape)
        x, y = batches("mixed", shape, 1)[0]
        captured = capture_module(model, objective, (x,), (y,))
        manifest = {"inputs": list(captured.ir.inputs),
                    "input_roles": {name: {"role": captured.ir.values[name].role} for name in captured.ir.inputs}}
        meta = captured.ir.metadata["module_api"]
        args = list(captured.example_args)
        identity = _immutable_identity(args, manifest, meta)
        index = next(iter(meta["input_indices"]["frozen"].values()))
        changed = list(args)
        changed[index] = args[index] + 1
        assert _immutable_identity(changed, manifest, meta) != identity, "frozen value outside checkpoint identity"
        trainable = list(args)
        trainable[0] = args[0] + 1
        assert _immutable_identity(trainable, manifest, meta) == identity
        return {"passed": True}

    check("checkpoint_identity_binds_frozen_values", checkpoint_identity)

    def memory_accounting():
        model = make_model("first", shape)
        x, y = batches("first", shape, 1)[0]
        captured = capture_module(model, objective, (x,), (y,), optimizer=OPTIMIZERS["adamw"]())
        policy = captured.ir.metadata["module_api"]["frozen_policy"]
        frozen = model.layers[0].weight.numel() + model.layers[0].bias.numel()
        assert policy["frozen_elements"] == frozen
        assert policy["avoided_bytes_upper_bound"] == frozen * 24
        outputs = captured.ir.output_names()
        assert len(outputs) == 1 + 2 * len(captured.parameter_names) + len(captured.optimizer_state_names)
        return {"passed": True, "policy": policy}

    check("state_avoided_accounting", memory_accounting)
    return {"passed": all(row["passed"] for row in results.values()), "shape": shape,
            "contract_count": len(results), "tests": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
