"""TINYMOE-SELECTIVE-CADENCE: update_schedule="streamed_ustate" drops commits of unchanged state.

A tensor rule that returns some parameters/state unchanged is captured twice and lowered through the
standalone flat compose planner, once with "streamed" and once with "streamed_ustate". The ustate program
must commit exactly the changed values, name the unchanged bindings as their returned outputs, and keep
every other part of the program (the changed values' writes, loss and gradients) the same.
"""
from __future__ import annotations

import torch

from megakernel_lab.operation_compiler import update_schedule
from megakernel_lab.operation_compiler.capture import capture_training
from megakernel_lab.operation_compiler.compose import lower_composed_flat
from megakernel_lab.operation_compiler.user_optimizer import tensor_update


def _rule():
    def init(parameters, _hp):
        state = {}
        for name, p in parameters.items():
            state["m:" + name] = torch.zeros_like(p)
            state["acc:" + name] = torch.zeros_like(p)
            state["t:" + name] = torch.zeros((), dtype=torch.int64)
        return state

    def update(parameters, gradients, state, _hp):
        out, new = {}, {}
        for name, p in parameters.items():
            g = gradients[name]
            if name == "fast":   # updates every step
                m = 0.9 * state["m:" + name] + 0.1 * g
                out[name] = p - 0.01 * m
                new["m:" + name], new["t:" + name] = m, state["t:" + name] + 1
                new["acc:" + name] = state["acc:" + name]
            else:                # accumulates only: parameter, moment and counter unchanged
                out[name] = p
                new["m:" + name], new["t:" + name] = state["m:" + name], state["t:" + name]
                new["acc:" + name] = state["acc:" + name] + g
        return out, new

    return tensor_update(update, init_state=init, name="cadence_probe")


def _lower(schedule):
    torch.manual_seed(0)
    params = {"fast": torch.randn(8, 16), "slow": torch.randn(16, 4)}
    x = torch.randn(32, 8)

    def loss(p, x):
        return torch.relu(x @ p["fast"]).matmul(p["slow"]).square().mean()

    rule = _rule()
    captured = capture_training(loss, params, None, (x,), rule)
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    with execution_scope("compose"), update_schedule.scope(schedule):
        program, _ = lower_composed_flat(captured.ir if hasattr(captured, "ir") else captured)
    return program.manifest


def test_unchanged_state_commits_nothing():
    base, ustate = _lower("streamed"), _lower("streamed_ustate")
    role = lambda m, name: m["input_roles"][name]
    named = lambda m: {(role(m, t)["role"], role(m, t)["parameter"]): s for t, s in m["state_writes"].items()}
    base_writes, ustate_writes = named(base), named(ustate)
    unchanged = {k for k in base_writes if k[1] in ("slow", "m:slow", "t:slow", "acc:fast", "m:fast")
                 and not (k[1] == "m:fast")}
    # Changed values keep their commits; unchanged ones are gone.
    assert ("parameter", "fast") in ustate_writes and ("optimizer_state", "acc:slow") in ustate_writes
    assert ("optimizer_state", "m:fast") in ustate_writes and ("optimizer_state", "t:fast") in ustate_writes
    for key in [("parameter", "slow"), ("optimizer_state", "m:slow"), ("optimizer_state", "t:slow"),
                ("optimizer_state", "acc:fast")]:
        assert key in base_writes and key not in ustate_writes, key
    assert unchanged
    record = ustate["group_rules"]["unchanged_state_writes"]
    assert record["writes"] == 4 and record["removed_copies"] >= 4
    # Returned outputs of unchanged values name their own input bindings.
    inputs = set(ustate["inputs"])
    by_param = {role(ustate, n)["parameter"]: n for n in ustate["inputs"] if role(ustate, n)["role"] == "parameter"}
    assert by_param["slow"] in ustate["outputs"] and by_param["slow"] in inputs
    assert "unchanged_state_writes" not in base.get("group_rules", {})


def test_late_updates_follow_every_gradient():
    late = _lower("streamed_ustate_late")
    record = late["group_rules"]["late_updates"]
    assert record["needed"] > 0 and record["intervals_after"] >= 1
    phases, intervals = late["phases"], late["barrier_intervals"]
    where = {i: k for k, members in enumerate(intervals) for i in members}
    producer = {w: i for i, p in enumerate(phases) for w in p["writes"]}
    so = late["structured_outputs"]
    last_grad = max(where[producer[v]] for v in so["grads"].values() if v in producer)
    committed = {s for s in late["state_writes"].values() if s in producer}
    assert committed and all(where[producer[s]] > last_grad for s in committed)
