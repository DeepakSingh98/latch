"""Modal CPU checks: selected branch, complete reverse history and AdamW state."""
from __future__ import annotations

import json
import traceback

import torch

from megakernel_lab.operation_compiler.loop import BoundedLoopIR, ConditionalBody, STORAGE_MODES, capture_loop_training
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.optim import AdamWConfig
from megakernel_lab.operation_compiler.random import dropout, make_rng_state


CONFIG = AdamWConfig(lr=0.002, betas=(0.8, 0.97), eps=1e-5, weight_decay=0.03)
COUNTS = (0, 1, 3, 5)


def fixture(width=8, batch=5):
    generator = torch.Generator().manual_seed(640411)
    params = {name: torch.randn(width, width, generator=generator) * 0.04
              for name in ("input", "shared", "true_only", "false_only", "head", "dead", "zero")}
    params["tied"] = params["shared"]
    inputs = (torch.randn(batch, width, generator=generator) * 0.15,
              torch.randn(batch, width, generator=generator) * 0.1)
    state = {"route": torch.tensor(True), "visits": torch.tensor(0, dtype=torch.int64),
             "rng": make_rng_state(0x123456789ABCDEF0, (1 << 40) + 27)}

    def initial(p, x, target):
        return {"hidden": x @ p["input"] + 0.05 * (x @ p["shared"]) + p["zero"].sum() * 0.0,
                "auxiliary": x.new_zeros(()), "unused": x @ p["dead"]}

    def selected(p, carry, state, x, target, *, branch):
        hidden = carry["hidden"]
        scale = 0.2 if branch else 0.35
        hidden = hidden + scale * torch.tanh(hidden @ p["shared"] + 0.1 * (hidden @ p["tied"])
                                             + x @ p["true_only" if branch else "false_only"])
        hidden, rng = dropout(hidden, state["rng"], 0.25 if branch else 0.5)
        if not branch:
            hidden, rng = dropout(hidden, rng, 0.125)
        auxiliary = carry["auxiliary"] + ((hidden @ p["head"] - target) ** 2).mean()
        return {"hidden": hidden, "auxiliary": auxiliary, "unused": carry["unused"]}, {
            "route": torch.logical_not(state["route"]), "visits": state["visits"] + (1 if branch else 2), "rng": rng}

    def true_body(p, carry, state, x, target):
        return selected(p, carry, state, x, target, branch=True)

    def false_body(p, carry, state, x, target):
        return selected(p, carry, state, x, target, branch=False)

    body = ConditionalBody(true_body, false_body, "route")

    def terminal(p, carry, x, target):
        return ((carry["hidden"] @ p["head"] - target) ** 2).mean() + 0.07 * carry["auxiliary"]

    return params, inputs, state, initial, body, terminal


def capture_fixture(width=8, batch=5):
    params, inputs, state, initial, body, terminal = fixture(width, batch)
    captured = capture_loop_training(initial, body, terminal, params, None, inputs, torch.tensor(0), CONFIG,
                                     max_steps=5, state=state)
    return captured, (initial, body, terminal)


def eager_step(captured, functions, flat, *, truncate=False, wrong_branch=False):
    """Independent ordinary autograd and actual torch.optim.AdamW; no captured AD."""
    initial, body, terminal = functions
    n, no, ns = len(captured.parameter_names), len(captured.optimizer_state_names), len(captured.state_names)
    p = {key: value.detach().clone().requires_grad_(True) for key, value in zip(captured.parameter_names, flat[:n])}
    mapped = {alias: p[key] for alias, key in captured.ir.metadata["parameter_aliases"].items()}
    old_optimizer = dict(zip(captured.optimizer_state_names, flat[n:n+no]))
    state = {key: value.detach().clone() for key, value in zip(captured.state_names, flat[n+no:n+no+ns])}
    data, count = flat[n+no+ns:-1], flat[-1]
    requested = int(count.item())
    steps = min(max(requested, 0), captured.ir.max_steps)
    carry = initial(mapped, *data)
    for index in range(steps):
        if truncate and index:
            carry = {key: value.detach() for key, value in carry.items()}
        if wrong_branch:
            fn = body.false_fn if bool(state["route"].item()) else body.true_fn
            carry, state = fn(mapped, carry, state, *data)
        else:
            carry, state = body(mapped, carry, state, *data)
    loss = terminal(mapped, carry, *data)
    gradients = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True, materialize_grads=False)
    reports = tuple(torch.zeros_like(value) if grad is None else grad.detach().clone()
                    for value, grad in zip(p.values(), gradients))
    config = captured.config
    groups = [{"params": [value], **config.param_options.get(key, {})} for key, value in p.items()]
    optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay, foreach=False, fused=False)
    for (key, value), gradient in zip(p.items(), gradients):
        value.grad = gradient
        optimizer.state[value] = {"exp_avg": old_optimizer[f"exp_avg:{key}"].detach().clone(),
                                  "exp_avg_sq": old_optimizer[f"exp_avg_sq:{key}"].detach().clone(),
                                  "step": old_optimizer[f"step:{key}"].detach().clone().to(torch.float32)}
    optimizer.step()
    next_optimizer = {f"{kind}:{key}": optimizer.state[value][kind].detach().clone().to(
        torch.int64 if kind == "step" else torch.float32)
                      for key, value in p.items() for kind in ("exp_avg", "exp_avg_sq", "step")}
    return (loss.detach(), *reports, *(value.detach().clone() for value in p.values()),
            *(next_optimizer[key] for key in captured.optimizer_state_names),
            *(state[key].detach().clone() for key in captured.state_names),
            *(carry[key].detach().clone() for key in captured.carry_names),
            count.new_tensor(steps), count.new_tensor(int(steps != requested)))


def next_args(captured, outputs, data, count, route):
    result = captured.unpack(outputs)
    state = dict(result["state"])
    state["route"] = torch.tensor(route)
    return (*result["params"].values(), *result["optimizer_state"].values(), *state.values(), *data, torch.tensor(count))


def assert_close(actual, expected):
    maximum = 0.0
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=4e-5, atol=3e-7)
            maximum = max(maximum, float((a-b).abs().max()))
        else:
            assert torch.equal(a, b), (a, b)
    return maximum


def test_nested_counts_routes_and_optimizer_participation():
    captured, functions = capture_fixture()
    n, no = len(captured.parameter_names), len(captured.optimizer_state_names)
    route_index = n + no + captured.state_names.index("route")
    rows = []
    for route in (True, False):
        for count in COUNTS:
            flat = list(captured.example_args)
            flat[route_index], flat[-1] = torch.tensor(route), torch.tensor(count)
            expected = eager_step(captured, functions, flat)
            error = assert_close(captured.step_fn(*flat), expected)
            for mode in STORAGE_MODES:
                error = max(error, assert_close(captured.graph_step(*flat, storage_mode=mode), expected))
            state = captured.unpack(expected)["optimizer_state"]
            assert state["step:dead"].item() == 0
            assert state["step:zero"].item() == 1
            if count == 0:
                assert state["step:true_only"].item() == state["step:false_only"].item() == 0
            elif count == 1:
                assert state["step:true_only"].item() == int(route)
                assert state["step:false_only"].item() == int(not route)
            rows.append({"route": route, "count": count, "max_absolute_error": error})
    return {"checks": rows, "None_vs_zero_and_unused_carry": True, "storage_modes": list(STORAGE_MODES)}


def test_nested_trained_state_and_roundtrip():
    captured, functions = capture_fixture()
    flat, rows = captured.example_args, []
    for step in range(41):
        flat = next_args(captured, eager_step(captured, functions, flat), flat[-3:-1], COUNTS[step % 4], bool(step % 2))
        if step in (0, 8, 40):
            expected = eager_step(captured, functions, flat)
            rows.append({"step": step, "max_absolute_error": assert_close(captured.graph_step(*flat), expected)})
    restored = BoundedLoopIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    assert restored.output_names() == captured.ir.output_names()
    assert lower(restored).cache_key == lower(captured.ir).cache_key
    return {"checks": rows, "sorted_json_source_identity": True}


def test_nested_negative_controls():
    captured, functions = capture_fixture()
    flat = (*captured.example_args[:-1], torch.tensor(5))
    correct = captured.unpack(eager_step(captured, functions, flat))
    truncated = captured.unpack(eager_step(captured, functions, flat, truncate=True))
    wrong = captured.unpack(eager_step(captured, functions, flat, wrong_branch=True))
    assert torch.equal(correct["loss"], truncated["loss"])
    assert not torch.allclose(correct["grads"]["input"], truncated["grads"]["input"], rtol=0.05, atol=1e-7)
    assert not torch.equal(correct["state"]["rng"], wrong["state"]["rng"])
    assert not torch.allclose(correct["loss"], wrong["loss"], rtol=1e-4, atol=1e-8)
    return {"truncated_history_detected": True, "wrong_selected_path_detected": True, "wrong_rng_advancement_detected": True}


def run():
    rows = []
    for function in (test_nested_counts_routes_and_optimizer_participation,
                     test_nested_trained_state_and_roundtrip, test_nested_negative_controls):
        try:
            rows.append({"name": function.__name__, "passed": True, "result": function()})
        except Exception as exc:
            rows.append({"name": function.__name__, "passed": False, "error": str(exc), "traceback": traceback.format_exc()})
    return {"passed": all(row["passed"] for row in rows), "tests": rows}
