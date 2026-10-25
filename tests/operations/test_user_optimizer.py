"""Step 12 user tensor update contracts. Run on the admitted Modal CPU worker.

Captured-graph semantics are checked against independent ordinary autograd and
actual torch.optim (SGD/RMSprop/Adam). Native GPU checks live in the step 12
harness; this file never executes generated CUDA.
"""
from __future__ import annotations

import json
import math

import torch

from megakernel_lab.operation_compiler.capture import capture_training, capture_conditional_training
from megakernel_lab.operation_compiler.ir import CaptureError, TrainingIR
from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig, initialize_optimizer_state, prepare_optimizer
from megakernel_lab.operation_compiler.user_optimizer import tensor_update

BOUNDS = {"loss": (2e-5, 1e-4), "grads": (1e-5, 5e-4), "params": (1e-5, 2e-4),
          "updates": (2e-7, 1e-3), "state": (2e-5, 5e-4)}   # NUMERICAL_POLICY.json values


# ----- user rules (the same formulas as the examples; written as user code) -----
def sgd_rule(p, g, s, hp):
    out_p, out_s = {}, {}
    for n in p:
        buf = hp["momentum"] * s["momentum:" + n] + (g[n] + hp["weight_decay"] * p[n])
        out_s["momentum:" + n], out_p[n] = buf, p[n] - hp["lr"] * buf
    return out_p, out_s


def rmsprop_rule(p, g, s, hp):
    out_p, out_s = {}, {}
    for n in p:
        sq = hp["alpha"] * s["square_avg:" + n] + (1 - hp["alpha"]) * g[n] * g[n]
        out_s["square_avg:" + n], out_p[n] = sq, p[n] - hp["lr"] * g[n] / (torch.sqrt(sq) + hp["eps"])
    return out_p, out_s


def adam_rule(p, g, s, hp):
    """Adam with one shared int64 step counter; bias powers from the counter on device."""
    b1, b2 = hp["beta1"], hp["beta2"]
    step = s["step"] + 1
    t = step.to(torch.float32)
    bc1 = 1 - torch.pow(torch.full_like(t, b1), t)
    bc2 = 1 - torch.pow(torch.full_like(t, b2), t)
    out_p, out_s = {}, {"step": step}
    for n in p:
        m = b1 * s["exp_avg:" + n] + (1 - b1) * g[n]
        v = b2 * s["exp_avg_sq:" + n] + (1 - b2) * g[n] * g[n]
        out_s["exp_avg:" + n], out_s["exp_avg_sq:" + n] = m, v
        out_p[n] = p[n] - (hp["lr"] / bc1) * m / (torch.sqrt(v) / torch.sqrt(bc2) + hp["eps"])
    return out_p, out_s


def zeros(*slots):
    def init(params, hp):
        return {f"{slot}:{n}": torch.zeros_like(v) for slot in slots for n, v in params.items()}
    return init


def adam_init(params, hp):
    return {"step": torch.zeros((), dtype=torch.int64), **zeros("exp_avg", "exp_avg_sq")(params, hp)}


RULES = {
    "sgd": (tensor_update(sgd_rule, init_state=zeros("momentum"),
                          hyperparameters={"lr": .05, "momentum": .9, "weight_decay": 1e-3}),
            lambda ps, hp: torch.optim.SGD(ps, lr=hp["lr"], momentum=hp["momentum"],
                                           weight_decay=hp["weight_decay"], foreach=False)),
    "rmsprop": (tensor_update(rmsprop_rule, init_state=zeros("square_avg"),
                              hyperparameters={"lr": 1e-3, "alpha": .99, "eps": 1e-8}),
                lambda ps, hp: torch.optim.RMSprop(ps, lr=hp["lr"], alpha=hp["alpha"], eps=hp["eps"], foreach=False)),
    "adam": (tensor_update(adam_rule, init_state=adam_init,
                           hyperparameters={"lr": 2e-3, "beta1": .9, "beta2": .999, "eps": 1e-8}),
             lambda ps, hp: torch.optim.Adam(ps, lr=hp["lr"], betas=(hp["beta1"], hp["beta2"]),
                                             eps=hp["eps"], foreach=False)),
}


def _fixture():
    gen = torch.Generator().manual_seed(1212)
    w2 = torch.randn(7, 7, generator=gen) * .3
    params = {"w1": torch.randn(5, 7, generator=gen) * .3, "b1": torch.randn(7, generator=gen) * .1,
              "w2": w2, "w2_again": w2, "unused": torch.randn(3, generator=gen)}

    def loss(p, x, y, weight):
        h = torch.tanh(torch.tanh(x @ p["w1"] + p["b1"]) @ p["w2"]) @ p["w2_again"]
        return weight * (h - y).square().mean()

    batches = []
    for step in range(6):
        weight = torch.tensor(0.0 if step in (2, 3) else 1.0)   # steps 2,3: all-zero gradients
        batches.append((torch.randn(4, 5, generator=gen), torch.randn(4, 7, generator=gen), weight))
    return params, loss, batches


def _compare(actual, expected, kind):
    atol, rtol = BOUNDS[kind]
    if actual.dtype == torch.int64 or expected.dtype == torch.int64:
        return {"passed": actual.dtype == expected.dtype and torch.equal(actual, expected), "exact": True}
    a, e = actual.detach().double(), expected.detach().double()
    ratio = (a - e).abs() / (atol + rtol * e.abs())
    return {"passed": a.shape == e.shape and bool(torch.isfinite(a).all()) and bool((ratio <= 1).all()),
            "max_ratio": float(ratio.max()) if a.numel() else 0.0}


def _torch_reference(kind, loss, canonical, aliases, state, batch, hp, slots):
    """Ordinary autograd + actual torch.optim from identical old parameters/state."""
    p = {n: v.detach().clone().requires_grad_(True) for n, v in canonical.items()}
    value = loss({a: p[c] for a, c in aliases.items()}, *batch)
    grads = torch.autograd.grad(value, tuple(p.values()), allow_unused=True)
    grads = {n: torch.zeros_like(v) if g is None else g for (n, v), g in zip(p.items(), grads)}
    opt = RULES[kind][1](list(p.values()), hp)
    for n, v in p.items():
        v.grad = grads[n].clone()
        if kind == "sgd":
            opt.state[v]["momentum_buffer"] = state["momentum:" + n].clone()
        elif kind == "rmsprop":
            opt.state[v].update(step=torch.tensor(0.0), square_avg=state["square_avg:" + n].clone())
        else:
            opt.state[v].update(step=torch.tensor(float(state["step"])), exp_avg=state["exp_avg:" + n].clone(),
                                exp_avg_sq=state["exp_avg_sq:" + n].clone())
    opt.step()
    new_state = {}
    for n, v in p.items():
        for slot, key in slots:
            new_state[f"{slot}:{n}"] = opt.state[v][key].detach().clone()
    if kind == "adam":
        new_state["step"] = state["step"] + 1
    return {"loss": value.detach(), "grads": grads, "params": {n: v.detach() for n, v in p.items()},
            "optimizer_state": new_state}


SLOTS = {"sgd": (("momentum", "momentum_buffer"),), "rmsprop": (("square_avg", "square_avg"),),
         "adam": (("exp_avg", "exp_avg"), ("exp_avg_sq", "exp_avg_sq"))}


def test_captured_rules_match_torch_optim():
    results = {}
    for kind, (rule, _) in RULES.items():
        params, loss, batches = _fixture()
        state = initialize_optimizer_state(params, rule)
        captured = capture_training(loss, params, None, batches[0], rule, optimizer_state=state)
        restored = TrainingIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
        assert restored.output_names() == captured.ir.output_names()
        assert captured.parameter_names == ("w1", "b1", "w2", "unused")
        assert captured.optimizer_state_names == tuple(state)
        assert captured.ir.metadata["optimizer"]["kind"] == "user_tensor_rule"
        assert {e["kind"] for e in captured.ir.metadata["optimizer_state_schema"]} == {"user"}
        canonical = {n: params[n] for n in captured.parameter_names}
        steps = []
        for index, batch in enumerate(batches):
            args = (*canonical.values(), *state.values(), *batch)
            got = captured.unpack(captured.graph_module(*args))
            expected = _torch_reference(kind, loss, canonical, captured.parameter_aliases, state, batch,
                                        dict(rule.hyperparameters), SLOTS[kind])
            checks = {"loss": _compare(got["loss"], expected["loss"], "loss")}
            for group, bound in (("grads", "grads"), ("params", "params"), ("optimizer_state", "state")):
                checks[group] = {n: _compare(v, expected[group][n], bound) for n, v in got[group].items()}
            checks["updates"] = {n: _compare(got["params"][n] - v, expected["params"][n] - v, "updates")
                                 for n, v in canonical.items()}
            zero = float(batch[2]) == 0.0
            if zero:
                checks["zero_gradients_exact"] = {"passed": all(bool((g == 0).all()) for g in got["grads"].values())}
            passed = all(c["passed"] for group in checks.values()
                         for c in (group.values() if "passed" not in group else (group,)))
            steps.append({"step": index, "zero_gradient_step": zero, "passed": passed, "checks": checks})
            assert passed, json.dumps(steps[-1])
            canonical = {n: v.clone() for n, v in got["params"].items()}
            state = {n: v.clone() for n, v in got["optimizer_state"].items()}
        assert torch.equal(canonical["unused"], params["unused"]) or kind == "sgd"   # no decay except SGD
        results[kind] = {"steps": len(steps), "passed": True}
    return results


def test_simultaneous_update_reads_old_values():
    """Swap and previous-weight rules expose any sequential (non-simultaneous) commit."""
    gen = torch.Generator().manual_seed(7)
    params = {"a": torch.randn(4, generator=gen), "b": torch.randn(4, generator=gen)}

    def loss(p, x):
        return (p["a"] * p["b"] * x).sum()

    def swap(p, g, s, hp):
        return ({"a": p["b"] - hp["lr"] * g["b"], "b": p["a"] - hp["lr"] * g["a"]},
                {"prev:a": p["a"], "prev:b": p["b"]})

    rule = tensor_update(swap, init_state=lambda p, hp: {"prev:" + n: torch.zeros_like(v) for n, v in p.items()},
                         hyperparameters={"lr": .1})
    state = initialize_optimizer_state(params, rule)
    x = torch.randn(4, generator=gen)
    captured = capture_training(loss, params, None, (x,), rule, optimizer_state=state)
    out = captured.unpack(captured.graph_module(*params.values(), *state.values(), x))
    ga, gb = params["b"] * x, params["a"] * x
    assert torch.allclose(out["params"]["a"], params["b"] - .1 * gb)
    assert torch.allclose(out["params"]["b"], params["a"] - .1 * ga)
    assert torch.equal(out["optimizer_state"]["prev:a"], params["a"])
    assert torch.equal(out["optimizer_state"]["prev:b"], params["b"])
    assert [e["parameter"] for e in captured.ir.metadata["optimizer_state_schema"]] == ["a", "b"]
    return {"passed": True}


def _refusal(function, code=None):
    try:
        function()
    except CaptureError as exc:
        if code is not None:
            assert exc.code == code, (exc.code, str(exc))
        return {"code": exc.code, "message": str(exc)[:200]}
    raise AssertionError("invalid user tensor rule was accepted")


def test_refusals():
    params, loss, batches = _fixture()
    sgd = RULES["sgd"][0]

    def capture(rule, **kw):
        return capture_training(loss, params, None, batches[0], rule, **kw)

    def inplace(p, g, s, hp):
        for n in p:
            p[n].sub_(hp["lr"] * g[n])
        return p, s

    def branch(p, g, s, hp):
        return ({n: (p[n] - g[n] if float(g[n].sum()) > 0 else p[n]) for n in p}, s)

    def noisy(p, g, s, hp):
        return ({n: p[n] - g[n] + 1e-3 * torch.rand_like(p[n]) for n in p}, s)

    def missing(p, g, s, hp):
        return ({n: p[n] for n in list(p)[1:]}, s)

    def reshaped(p, g, s, hp):
        return ({n: (p[n] - g[n]).reshape(-1) for n in p}, s)

    def scalar(p, g, s, hp):
        return ({n: p[n] - g[n] for n in p}, {"renamed": s["extra"] + 1})

    refusals = {
        "inplace_parameter_mutation": _refusal(lambda: capture(tensor_update(inplace, hyperparameters={"lr": .1})),
                                               "OPTIMIZER_MUTATION"),
        "tensor_dependent_python_branch": _refusal(lambda: capture(tensor_update(branch)), "RUNTIME_CONTROL"),
        "randomness": _refusal(lambda: capture(tensor_update(noisy))),
        "missing_parameter": _refusal(lambda: capture(tensor_update(missing)), "OPTIMIZER_RETURN"),
        "changed_shape": _refusal(lambda: capture(tensor_update(reshaped)), "OPTIMIZER_RETURN"),
        "changed_state_keys": _refusal(lambda: capture(tensor_update(
            scalar, init_state=lambda p, hp: {"extra": torch.zeros(())})), "OPTIMIZER_RETURN"),
        "tensor_hyperparameter": _refusal(lambda: tensor_update(sgd_rule, hyperparameters={"lr": torch.tensor(.1)}),
                                          "OPTIMIZER"),
        "nonfinite_hyperparameter": _refusal(lambda: tensor_update(sgd_rule, hyperparameters={"lr": math.inf}),
                                             "OPTIMIZER"),
        "int64_tensor_state": _refusal(lambda: capture(tensor_update(
            sgd_rule, init_state=lambda p, hp: {"bad": torch.zeros(3, dtype=torch.int64)})), "OPTIMIZER_STATE"),
        "supplied_state_mismatch": _refusal(lambda: capture(sgd, optimizer_state={"momentum:w1": torch.zeros(5, 7)}),
                                            "OPTIMIZER_STATE"),
        "legacy_momentum_argument": _refusal(lambda: capture_training(loss, params, {"w1": torch.zeros(5, 7)},
                                                                      batches[0], sgd), "OPTIMIZER_STATE"),
        "conditional_region": _refusal(lambda: capture_conditional_training(
            loss, loss, params, None, batches[0], torch.tensor(True), sgd), "OPTIMIZER"),
    }
    from megakernel_lab.operation_compiler.loop import capture_loop_training
    loop_params = {n: params[n].contiguous() for n in ("w1", "b1")}
    refusals["loop_capture"] = _refusal(lambda: capture_loop_training(
        lambda p, x: {"h": x @ p["w1"]}, lambda p, c, s, x: ({"h": torch.tanh(c["h"] + p["b1"])}, s),
        lambda p, c, x: c["h"].square().mean(), loop_params, None, (batches[0][0],),
        torch.tensor(2, dtype=torch.int64), sgd, max_steps=2), "OPTIMIZER")
    assert "user tensor update rules" in refusals["loop_capture"]["message"]
    return refusals


def test_builtin_optimizers_still_dispatch():
    """Built-in configs keep their protocol objects, names and state layouts."""
    params, _, _ = _fixture()
    canonical = {n: params[n] for n in ("w1", "b1", "w2", "unused")}
    sgd = prepare_optimizer(SGDConfig(), canonical)
    adamw = prepare_optimizer(AdamWConfig(), canonical)
    assert sgd.legacy_momentum and sgd.state_names == tuple(canonical)
    assert adamw.state_names[:3] == ("exp_avg:w1", "exp_avg_sq:w1", "step:w1")
    assert type(adamw).__name__ == "PreparedOptimizer"
    return {"passed": True}


def run_all():
    results, failures = {}, {}
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                results[name] = function()
            except Exception as exc:
                import traceback
                failures[name] = traceback.format_exc()[-4000:]
    return {"passed": not failures, "results": results, "failures": failures}


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, default=str))
