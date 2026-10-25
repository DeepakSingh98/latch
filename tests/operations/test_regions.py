"""Stage 2 region/state tests and upstream compatibility probes; Modal CPU only."""
from __future__ import annotations

import traceback
import json

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training, capture_conditional_training
from megakernel_lab.operation_compiler.ir import CaptureError, TrainingIR, UniformConditionalIR


def _fixture():
    gen = torch.Generator().manual_seed(8817)
    params = {"w": torch.randn(4, 4, generator=gen) * 0.1}
    inputs = (torch.randn(3, 4, generator=gen), torch.randn(3, 4, generator=gen))
    return params, inputs


def _expect_error(fn, code):
    try:
        fn()
    except CaptureError as exc:
        assert exc.code == code, (exc.code, str(exc))
        return str(exc)
    raise AssertionError(f"expected refusal {code}")


def test_persistent_state_and_loss_only_derivative():
    params, inputs = _fixture()
    state = {"count": torch.tensor(4, dtype=torch.int64), "running": torch.tensor(0.25)}
    def loss(p, s, x, y):
        objective = ((x @ p["w"] - y) ** 2).mean()
        return objective, {"running": s["running"] * 0.9 + objective * 0.1,
                           "count": s["count"] + 1}
    captured = capture_training(loss, params, None, inputs, state=state)
    got = captured.unpack(captured.graph_module(*captured.example_args))
    p = params["w"].clone().requires_grad_()
    ref = ((inputs[0] @ p - inputs[1]) ** 2).mean()
    (grad,) = torch.autograd.grad(ref, p)
    torch.testing.assert_close(got["grads"]["w"], grad)
    torch.testing.assert_close(got["state"]["running"], 0.9 * state["running"] + 0.1 * ref.detach())
    assert got["state"]["count"].item() == 5
    assert captured.state_names == ("count", "running")
    assert len(captured.example_args) == 6
    assert captured.ir.metadata["state_names"] == ["count", "running"]
    roles = [captured.ir.values[name].role for name in captured.ir.inputs]
    assert roles == ["parameter", "momentum", "state", "state", "input", "input"]
    restored = TrainingIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    assert restored.output_names() == captured.ir.output_names()
    assert restored.lifetimes() == captured.ir.lifetimes()
    return {"state_dtypes": {k: str(v.dtype) for k, v in got["state"].items()},
            "count_after": int(got["state"]["count"])}


def test_returned_state_is_new_version_even_when_unchanged():
    params, inputs = _fixture()
    state = {"unchanged": torch.ones(4, 4), "old_weight": torch.zeros(4, 4)}
    def loss(p, s, x, y):
        return ((x @ p["w"] - y) ** 2).mean(), {"unchanged": s["unchanged"], "old_weight": p["w"]}
    captured = capture_training(loss, params, None, inputs, state=state)
    got = captured.unpack(captured.graph_module(*captured.example_args))
    assert torch.equal(got["state"]["unchanged"], state["unchanged"])
    assert torch.equal(got["state"]["old_weight"], params["w"])
    for key in captured.state_names:
        value = captured.ir.values[captured.ir.outputs["state"][key]]
        assert value.alias_of is None
        assert got["state"][key].untyped_storage().data_ptr() not in {
            arg.untyped_storage().data_ptr() for arg in captured.example_args}


def test_uniform_conditional_toggle_without_recapture():
    params, inputs = _fixture()
    def yes(p, x, y):
        return ((torch.tanh(x @ p["w"]) - y) ** 2).mean()
    def no(p, x, y):
        return ((x @ p["w"] + y) ** 2).mean()
    captured = capture_conditional_training(yes, no, params, None, inputs, torch.tensor(True))
    grads = []
    for selection, branch in ((True, captured.true_branch), (False, captured.false_branch)):
        args = (*captured.example_args[:-1], torch.tensor(selection))
        out = captured.step_fn(*args)
        graph_out = branch.graph_module(*args[:-1])
        assert all(torch.equal(a, b) for a, b in zip(out, graph_out))
        p = params["w"].clone().requires_grad_()
        ref = (yes if selection else no)({"w": p}, *inputs)
        (grad,) = torch.autograd.grad(ref, p)
        actual = captured.unpack(out)
        torch.testing.assert_close(actual["grads"]["w"], grad)
        grads.append(actual["grads"]["w"])
    assert not torch.equal(*grads)
    assert captured.ir.inputs[-1] == captured.ir.predicate.name
    assert captured.ir.predicate.dtype == "bool"
    assert captured.ir.metadata["predicate_index"] == len(captured.example_args) - 1
    return {"capture_count": 1, "runtime_predicates_exercised": [True, False],
            "scope": "explicit conditional reference and both captured branches; not native execution"}


def test_conditional_only_selected_state_transition():
    params, inputs = _fixture()
    state = {"count": torch.tensor(0, dtype=torch.int64)}
    def yes(p, s, x, y):
        return ((x @ p["w"] - y) ** 2).mean(), {"count": s["count"] + 1}
    def no(p, s, x, y):
        return ((x @ p["w"] + y) ** 2).mean(), {"count": s["count"] + 7}
    captured = capture_conditional_training(yes, no, params, None, inputs, torch.tensor(False), state=state)
    flat = captured.example_args[:-1]
    for predicate in (True, False, False, True):
        got = captured.unpack(captured.step_fn(*flat, torch.tensor(predicate)))
        flat = (*got["params"].values(), *got["momentum"].values(), *got["state"].values(), *inputs)
    assert got["state"]["count"].item() == 16
    assert state["count"].item() == 0
    return {"final_count": 16, "counterfactual_both_branches_count": 32}


def test_conditional_json_roundtrip_order():
    params, inputs = _fixture()
    state = {"z_state": torch.tensor(0, dtype=torch.int64), "a_state": torch.tensor(2, dtype=torch.int64)}
    def loss(p, s, x, y):
        return ((x @ p["w"] - y) ** 2).mean(), {"a_state": s["a_state"] + 2, "z_state": s["z_state"] + 1}
    captured = capture_conditional_training(loss, loss, params, None, inputs, torch.tensor(True), state=state)
    packed = json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True))
    restored = UniformConditionalIR.from_dict(packed)
    assert restored.inputs == captured.ir.inputs
    assert restored.output_names() == captured.ir.output_names()
    assert restored.outputs == captured.ir.outputs
    assert restored.metadata["state_names"] == ["z_state", "a_state"]
    assert json.loads(json.dumps(restored.to_dict())) == packed
    for a, b in zip(restored.true_branch.output_names(), restored.false_branch.output_names()):
        assert restored.true_branch.values[a].dtype == restored.false_branch.values[b].dtype


def test_region_predicate_and_state_refusals():
    params, inputs = _fixture()
    loss = lambda p, x, y: ((x @ p["w"] - y) ** 2).mean()
    a = _expect_error(lambda: capture_conditional_training(loss, loss, params, None, inputs, torch.tensor(1.0)),
                      "PREDICATE")
    b = _expect_error(lambda: capture_conditional_training(loss, loss, params, None, inputs, torch.ones(2, dtype=torch.bool)),
                      "PREDICATE")
    wrong_state = lambda p, s, x, y: (loss(p, x, y), {"count": s["count"].float()})
    c = _expect_error(lambda: capture_training(wrong_state, params, None, inputs,
                                              state={"count": torch.tensor(0, dtype=torch.int64)}), "STATE_RETURN")
    def mutates(p, s, x, y):
        s["count"].add_(1)
        return loss(p, x, y), s
    d = _expect_error(lambda: capture_training(mutates, params, None, inputs,
                                              state={"count": torch.tensor(0, dtype=torch.int64)}), "MUTATION")
    return {"float_predicate": a, "vector_predicate": b, "state_dtype_change": c, "state_mutation": d}


def probe_torch_cond():
    """Probe the pinned upstream API before choosing branch-wise capture.

    Failures are observations, not test passes or an assertion of fundamental
    impossibility. Each path gets a fresh differentiable parameter; full failure
    context is retained for the smallest self-contained example.
    """
    torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(9812)
    w = torch.randn(4, 4, generator=gen) * 0.1
    x = torch.randn(3, 4, generator=gen)
    y = torch.randn(3, 4, generator=gen)
    pred = torch.tensor(True)

    def true_loss(w, x, y):
        return ((torch.tanh(x @ w) - y) ** 2).mean()

    def false_loss(w, x, y):
        return ((x @ w + y) ** 2).mean()

    def conditional(w, x, y, predicate):
        return torch.cond(predicate, true_loss, false_loss, (w, x, y))

    def eager_autograd():
        p = w.clone().requires_grad_()
        loss = conditional(p, x, y, pred)
        grad = torch.autograd.grad(loss, p)[0]
        return {"loss": float(loss), "gradient_norm": float(grad.norm())}

    def func_ad():
        grad, loss = torch.func.grad_and_value(conditional)(w, x, y, pred)
        return {"loss": float(loss), "gradient_norm": float(grad.norm())}

    def func_ad_make_fx():
        graph = make_fx(torch.func.grad_and_value(conditional), tracing_mode="fake")(w, x, y, pred)
        grad, loss = graph(w, x, y, pred)
        return {"loss": float(loss), "gradient_norm": float(grad.norm()), "source": graph.code,
                "targets": [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]}

    results = {}
    for name, fn in (("ordinary_autograd", eager_autograd), ("torch_func_grad", func_ad),
                     ("torch_func_grad_make_fx", func_ad_make_fx)):
        try:
            results[name] = {"succeeded": True, "result": fn()}
        except Exception as exc:
            results[name] = {"succeeded": False, "error_type": type(exc).__name__,
                             "message": str(exc), "traceback": traceback.format_exc()}
    return {"torch": torch.__version__, "shape": {"weight": [4, 4], "input": [3, 4]},
            "predicate": "runtime scalar torch.bool", "results": results}


def run():
    torch.set_num_threads(1)
    tests = {name: fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)}
    results = {}
    for name, fn in tests.items():
        try:
            results[name] = {"passed": True, "details": fn()}
        except Exception:
            results[name] = {"passed": False, "traceback": traceback.format_exc()}
    return {"torch": torch.__version__, "passed": all(row["passed"] for row in results.values()),
            "count": len(results), "tests": results, "upstream_torch_cond_probe": probe_torch_cond()}


if __name__ == "__main__":
    import json
    receipt = run()
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt["passed"] else 1)
