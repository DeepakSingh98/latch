"""Modal CPU contract tests for bounded-loop capture and full-history AD."""
from __future__ import annotations

import json
import traceback

import torch

from megakernel_lab.operation_compiler.capture import SGDConfig
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.loop import BoundedLoopIR, STORAGE_MODES, capture_loop_training
from megakernel_lab.operation_compiler.random import dropout, make_rng_state


CONFIG = SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.001)


def fixture(width=4, batch=3, *, random=False):
    generator = torch.Generator().manual_seed(6113)
    shared = torch.randn(width, width, generator=generator) * 0.1
    params = {"input": torch.randn(width, width, generator=generator) * 0.1,
              "shared": shared, "tied": shared,
              "head": torch.randn(width, width, generator=generator) * 0.1}
    inputs = (torch.randn(batch, width, generator=generator) * 0.2,
              torch.randn(batch, width, generator=generator) * 0.1)
    state = {"count": torch.tensor((1 << 40) + 13, dtype=torch.int64)}
    if random:
        state["rng"] = make_rng_state(0x123456789ABCDEF0, (1 << 40) + (1 << 32) - 2)

    def initial(p, x, target):
        return {"hidden": x @ p["input"] + 0.05 * (x @ p["shared"]),
                "auxiliary": x.new_zeros(())}

    def body(p, carry, state, x, target):
        hidden = carry["hidden"] + 0.3 * torch.tanh(carry["hidden"] @ p["shared"] + 0.1 * (carry["hidden"] @ p["tied"]))
        new_state = {"count": state["count"] + 1}
        if random:
            hidden, new_state["rng"] = dropout(hidden, state["rng"], 0.25)
        auxiliary = carry["auxiliary"] + ((hidden @ p["head"] - target) ** 2).mean()
        return {"hidden": hidden, "auxiliary": auxiliary}, new_state

    def terminal(p, carry, x, target):
        return ((carry["hidden"] @ p["head"] - target) ** 2).mean() + 0.03 * carry["auxiliary"]

    return params, inputs, state, initial, body, terminal


def eager_step(capture, functions, flat):
    """Independent ordinary autograd and torch.optim.SGD for one loop update."""
    initial, body, terminal = functions
    names, state_names = capture.parameter_names, capture.state_names
    n, ns = len(names), len(state_names)
    p = {name: value.detach().clone().requires_grad_(True) for name, value in zip(names, flat[:n])}
    mapped = {alias: p[name] for alias, name in capture.ir.metadata["parameter_aliases"].items()}
    momentum = dict(zip(names, flat[n:2*n]))
    state = {name: value.detach().clone() for name, value in zip(state_names, flat[2*n:2*n+ns])}
    data, count = flat[2*n+ns:-1], flat[-1]
    requested = int(count.item())
    steps = min(max(requested, 0), capture.ir.max_steps)
    carry = initial(mapped, *data)
    for _ in range(steps):
        carry, state = body(mapped, carry, state, *data)
    loss = terminal(mapped, carry, *data)
    derivatives = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True)
    gradients = tuple(torch.zeros_like(value) if grad is None else grad.detach().clone()
                      for value, grad in zip(p.values(), derivatives))
    optimizer = torch.optim.SGD(list(p.values()), lr=CONFIG.lr, momentum=CONFIG.momentum,
                                weight_decay=CONFIG.weight_decay, foreach=False)
    for (name, value), grad in zip(p.items(), gradients):
        value.grad = grad
        optimizer.state[value]["momentum_buffer"] = momentum[name].detach().clone()
    optimizer.step()
    return (loss.detach(), *gradients, *(value.detach().clone() for value in p.values()),
            *(optimizer.state[value]["momentum_buffer"].detach().clone() for value in p.values()),
            *(state[name].detach().clone() for name in state_names),
            *(carry[name].detach().clone() for name in capture.carry_names),
            count.new_tensor(steps), count.new_tensor(int(requested != steps)))


def _capture(*, random=False, storage_mode="save_boundaries", max_steps=5):
    params, inputs, state, initial, body, terminal = fixture(random=random)
    captured = capture_loop_training(initial, body, terminal, params, None, inputs, torch.tensor(1),
                                     CONFIG, max_steps=max_steps, state=state, storage_mode=storage_mode)
    return captured, (initial, body, terminal)


def _assert_outputs(actual, expected):
    assert len(actual) == len(expected)
    maximum = 0.0
    for a, b in zip(actual, expected):
        if a.dtype == torch.float32:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-7)
            maximum = max(maximum, float((a-b).abs().max()))
        else:
            assert torch.equal(a, b)
    return maximum


def test_runtime_counts_and_full_shared_history():
    captured, functions = _capture()
    assert captured.parameter_names == ("input", "shared", "head")
    assert captured.ir.metadata["parameter_aliases"]["tied"] == "shared"
    errors = []
    codes = {key: graph.code for key, graph in captured.graph_modules.items()}
    for count in (0, 1, 3, 5):
        flat = (*captured.example_args[:-1], torch.tensor(count))
        expected = eager_step(captured, functions, flat)
        errors.append(_assert_outputs(captured.step_fn(*flat), expected))
        for mode in STORAGE_MODES:
            errors.append(_assert_outputs(captured.graph_step(*flat, storage_mode=mode), expected))
    assert codes == {key: graph.code for key, graph in captured.graph_modules.items()}
    return {"counts": [0, 1, 3, 5], "storage_modes": list(STORAGE_MODES),
            "max_absolute_error": max(errors), "canonical_shared_parameter": "shared"}


def test_clamped_count_is_observable():
    captured, functions = _capture()
    result = []
    for requested, executed, status in ((-3, 0, 1), (0, 0, 0), (5, 5, 0), (9, 5, 1)):
        flat = (*captured.example_args[:-1], torch.tensor(requested))
        actual = captured.graph_step(*flat)
        _assert_outputs(actual, eager_step(captured, functions, flat))
        assert actual[-2].item() == executed and actual[-1].item() == status
        result.append({"requested": requested, "executed": executed, "status": status})
    return result


def test_rng_replay_and_boundary_recomputation():
    captured, functions = _capture(random=True)
    old_state = [value.clone() for value in captured.example_args]
    rows = []
    for count in (0, 1, 3, 5):
        flat = (*captured.example_args[:-1], torch.tensor(count))
        expected = eager_step(captured, functions, flat)
        saved = captured.graph_step(*flat, storage_mode="save_boundaries")
        replay = captured.graph_step(*flat, storage_mode="recompute_boundaries")
        tape = captured.graph_step(*flat, storage_mode="save_body_tape")
        _assert_outputs(saved, expected)
        _assert_outputs(replay, expected)
        assert all(torch.equal(a, b) for a, b in zip(saved, replay))
        assert all(torch.equal(a, b) for a, b in zip(saved, tape))
        unpacked = captured.unpack(saved)
        assert unpacked["state"]["count"].item() == (1 << 40) + 13 + count
        rows.append({"count": count, "rng": unpacked["state"]["rng"].tolist(), "save_recompute_bitwise": True})
    assert all(torch.equal(a, b) for a, b in zip(captured.example_args, old_state))
    return {"checks": rows, "inputs_unmodified": True,
            "body_vjp_rng_targets": [op.target for op in captured.ir.body_vjp.operations if op.target.startswith("latch_ops.")]}


def test_sorted_json_roundtrip_and_region_abi():
    captured, _ = _capture(random=True)
    saved = json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True))
    restored = BoundedLoopIR.from_dict(saved)
    assert restored.inputs == captured.ir.inputs
    assert restored.output_names() == captured.ir.output_names()
    assert restored.outputs == captured.ir.outputs
    assert json.loads(json.dumps(restored.to_dict())) == saved
    for graph in restored.regions.values():
        assert isinstance(graph.outputs, tuple) and graph.state_writes == {}
    n, nc, ns, ni = len(captured.parameter_names), len(captured.carry_names), len(captured.state_names), captured.ir.metadata["n_inputs"]
    expected = {"init": (n+ni, nc), "body": (n+nc+ns+ni, nc+ns),
                "terminal_vjp": (n+nc+ni, 1+n+nc), "body_vjp": (n+nc+ns+ni+nc, n+nc),
                "init_vjp": (n+ni+nc, n), "optimizer": (3*n, 2*n)}
    nt = len(captured.ir.metadata["body_tape"])
    expected.update(body_saved=(n+nc+ns+ni, nc+ns+nt), body_vjp_saved=(n+nc+ns+ni+nt+nc, n+nc))
    assert {key: (len(graph.inputs), len(graph.outputs)) for key, graph in restored.regions.items()} == expected
    return {"region_arity": expected, "state_order": list(captured.state_names)}


def test_body_tape_uses_material_roots_and_rebuilds_views():
    captured, functions = _capture(random=True)
    descriptors = captured.ir.metadata["body_tape"]
    assert descriptors
    for value in descriptors:
        original = captured.ir.body_saved.values[value["name"]]
        assert original.alias_of is None
        assert value["strides"] == list(captured.ir.body_vjp_saved.values[value["name"]].strides)
    backward_targets = [operation.target for operation in captured.ir.body_vjp_saved.operations]
    assert not any(target.startswith("latch_ops.philox_") for target in backward_targets)
    assert "aten.t.default" in backward_targets
    assert any("zeros_like" in target or "full_like" in target for target in backward_targets)
    flat = (*captured.example_args[:-1], torch.tensor(3))
    _assert_outputs(captured.graph_step(*flat, storage_mode="save_body_tape"), eager_step(captured, functions, flat))
    return {"material_tape_values": len(descriptors), "backward_rng_calls": 0,
            "transpose_views_rematerialized": backward_targets.count("aten.t.default"),
            "unused_parameter_zeros_rematerialized": True}


def test_capture_does_not_unroll_maximum():
    small, _ = _capture(max_steps=1)
    large, _ = _capture(max_steps=8)
    assert {key: value.code for key, value in small.graph_modules.items()} == {key: value.code for key, value in large.graph_modules.items()}
    return {"maximums": [1, 8], "region_sources_identical": True,
            "operation_counts": {key: len(graph.operations) for key, graph in small.ir.regions.items()}}


def test_truncated_history_negative_control():
    from megakernel_lab.operation_compiler.numerical_reference import compare
    captured, functions = _capture()
    initial, body, terminal = functions
    flat = (*captured.example_args[:-1], torch.tensor(5))
    expected = captured.unpack(eager_step(captured, functions, flat))
    n, ns = len(captured.parameter_names), len(captured.state_names)
    p = {name: value.detach().clone().requires_grad_(True) for name, value in zip(captured.parameter_names, flat[:n])}
    mapped = {alias: p[name] for alias, name in captured.ir.metadata["parameter_aliases"].items()}
    state, inputs = dict(zip(captured.state_names, flat[2*n:2*n+ns])), flat[2*n+ns:-1]
    carry = initial(mapped, *inputs)
    for index in range(5):
        if index:
            carry = {key: value.detach() for key, value in carry.items()}
        carry, state = body(mapped, carry, state, *inputs)
    wrong_loss = terminal(mapped, carry, *inputs)
    wrong_gradients = torch.autograd.grad(wrong_loss, tuple(p.values()), allow_unused=True)
    rejected, worst = [], 0.0
    for (name, value), gradient in zip(p.items(), wrong_gradients):
        gradient = torch.zeros_like(value) if gradient is None else gradient
        result = compare(gradient, expected["grads"][name], "gradients")
        worst = max(worst, result["max_ratio"])
        if not result["passed"]:
            rejected.append(name)
    assert torch.equal(wrong_loss.detach(), expected["loss"])
    assert "input" in rejected, (rejected, worst)
    return {"unchanged_forward_loss": True, "truncated_history_rejected_parameters": rejected,
            "worst_error_to_bound_ratio": worst}


def test_intermediate_objective_contribution_negative_control():
    from megakernel_lab.operation_compiler.gradient_audits import intermediate_contribution_audit
    params, inputs, state, initial, body, _ = fixture()
    def components(p):
        carry, active = initial(p, *inputs), state
        for _ in range(5):
            carry, active = body(p, carry, active, *inputs)
        terminal = ((carry["hidden"] @ p["head"] - inputs[1]) ** 2).mean()
        return terminal, carry["auxiliary"]
    weight = 0.5
    result = intermediate_contribution_audit(
        params, lambda p: components(p)[0] + weight * components(p)[1],
        lambda p: components(p)[0], lambda p: components(p)[1], weight=weight)
    assert result["passed"], result
    return result


def test_invalid_carry_state_effects_refused():
    params, inputs, state, initial, body, terminal = fixture()
    checks = []
    def expect(fn, code):
        try:
            fn()
        except CaptureError as exc:
            assert exc.code == code, (code, exc.code, str(exc))
            checks.append({"code": code, "message": str(exc)})
        else:
            raise AssertionError(f"missing {code} refusal")
    def capture(body_override=body, **kwargs):
        options = {"state": state, "max_steps": 5, **kwargs}
        return capture_loop_training(initial, body_override, terminal, params, None, inputs,
                                     torch.tensor(1), CONFIG, **options)
    expect(lambda: capture(max_steps=-1), "LOOP_BOUND")
    expect(lambda: capture(state={"count": torch.tensor(0.0)}), "LOOP_STATE")
    def changed(p, carry, state, x, y):
        return {"hidden": carry["hidden"].flatten(), "auxiliary": carry["auxiliary"]}, state
    expect(lambda: capture(changed), "LOOP_CARRY")
    def mutation(p, carry, state, x, y):
        state["count"].add_(1)
        return carry, state
    expect(lambda: capture(mutation), "MUTATION")
    def random(p, carry, state, x, y):
        return {"hidden": carry["hidden"] + torch.rand_like(carry["hidden"]), "auxiliary": carry["auxiliary"]}, state
    expect(lambda: capture(random), "RNG")
    transposed = {**params, "input": params["input"].t()}
    expect(lambda: capture_loop_training(initial, body, terminal, transposed, None, inputs, torch.tensor(1),
                                          CONFIG, max_steps=5, state=state), "LOOP_LAYOUT")
    return checks


def test_lower_refuses_deserialized_mutable_strides():
    from megakernel_lab.operation_compiler.loop_lower import lower_loop
    from megakernel_lab.operation_compiler.lower import LoweringError
    captured, _ = _capture(random=True)
    n = len(captured.parameter_names)
    # A manually edited/deserialized IR must not bypass capture's layout guard.
    rows = []
    for index, strides, role in ((0, (1, 4), "parameter"), (n, (1, 4), "momentum"),
                                 (2*n+1, (2,), "state")):
        changed = BoundedLoopIR.from_dict(json.loads(json.dumps(captured.ir.to_dict())))
        changed.external_values[changed.inputs[index]].strides = strides
        try:
            lower_loop(changed)
        except LoweringError as exc:
            assert "must be contiguous before final commit" in str(exc), str(exc)
            rows.append({"role": role, "refused": str(exc)})
        else:
            raise AssertionError(f"noncontiguous {role} was admitted")
    return rows


def run():
    torch.set_num_threads(1)
    results = {}
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                results[name] = {"passed": True, "details": function()}
            except Exception as exc:
                results[name] = {"passed": False, "error": str(exc), "traceback": traceback.format_exc()}
    return {"kind": "bounded-loop-cpu-contract", "torch": torch.__version__, "count": len(results),
            "tests": results, "passed": all(value["passed"] for value in results.values())}
