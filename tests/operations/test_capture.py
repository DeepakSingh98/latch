"""Substantive capture-contract checks, run on Modal CPU (one numerical thread).

Run ``python -m ops.test_capture`` for a compact structured receipt, or collect
these functions with pytest. The higher-order check validates capture only.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import torch

from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training, module_loss
from megakernel_lab.operation_compiler.ir import CaptureError, Ref, TrainingIR


def _fixture():
    gen = torch.Generator().manual_seed(903)
    return (torch.randn(4, 4, generator=gen) * 0.1,
            torch.randn(3, 4, generator=gen), torch.randn(3, 4, generator=gen))


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def _refusal(fn, expected):
    try:
        fn()
    except CaptureError as exc:
        assert exc.code in expected, (exc.code, str(exc), expected)
        return {"code": exc.code, "message": str(exc)}
    raise AssertionError("unsupported capture unexpectedly succeeded")


def test_shared_alias_accumulates_once():
    w, x, y = _fixture()
    config = SGDConfig(lr=0.03, momentum=0.8, weight_decay=0.01)
    m = torch.full_like(w, 0.07)

    def loss(p, x, y):
        return ((torch.tanh(x @ p["w"]) + x @ p["tied"] - y) ** 2).mean()

    c = capture_training(loss, {"w": w, "tied": w}, {"w": m}, (x, y), config)
    assert c.parameter_names == ("w",)
    assert c.parameter_aliases == {"w": "w", "tied": "w"}
    assert len(c.ir.inputs) == 4
    out = c.unpack(c.graph_module(*c.example_args))
    # Untie the two occurrences only in an independent reference; each path's
    # contribution must reach the one canonical compiler parameter gradient.
    w1, w2 = w.clone().requires_grad_(), w.clone().requires_grad_()
    ref_loss = loss({"w": w1, "tied": w2}, x, y)
    a, b = torch.autograd.grad(ref_loss, (w1, w2))
    expected_grad = a + b
    expected_m = config.momentum * m + expected_grad + config.weight_decay * w
    _close(out["loss"], ref_loss)
    _close(out["grads"]["w"], expected_grad)
    _close(out["momentum"]["w"], expected_m)
    _close(out["params"]["w"], w - config.lr * expected_m)


def test_ir_roundtrip_and_lifetimes():
    w, x, y = _fixture()
    c = capture_training(lambda p, x, y: ((x @ p["w"] - y) ** 2).mean(),
                         {"w": w}, None, (x, y))
    serialized = json.loads(json.dumps(c.ir.to_dict()))
    restored = TrainingIR.from_dict(serialized)
    assert json.loads(json.dumps(restored.to_dict())) == serialized
    assert restored.output_names() == c.ir.output_names()
    assert restored.lifetimes() == c.ir.lifetimes()
    end = len(c.ir.operations)
    for name in c.ir.output_names():
        assert c.ir.lifetimes()[name][1] == end
    for op in restored.operations:
        for name in op.inputs:
            assert name in restored.values
    assert any(isinstance(arg, Ref) for op in restored.operations for arg in op.args)


def test_graphmodule_save_load():
    w, x, y = _fixture()
    c = capture_training(lambda p, x, y: ((x @ p["w"] - y) ** 2).mean(),
                         {"w": w}, None, (x, y))
    data = io.BytesIO()
    torch.save(c.graph_module, data)
    data.seek(0)
    restored = torch.load(data, weights_only=False)
    for got, ref in zip(restored(*c.example_args), c.graph_module(*c.example_args)):
        assert torch.equal(got, ref)


def test_sorted_json_keeps_output_order_and_codegen():
    from megakernel_lab.operation_compiler.lower import lower
    # Deliberately nonalphabetic parameter order; matrix sizes also exercise
    # lowering without compiling or running a CUDA kernel in this CPU test.
    weights = {"z_first": torch.eye(64) * 0.2, "a_second": torch.eye(64) * 0.3}
    x, y = torch.ones(64, 64) * 0.1, torch.zeros(64, 64)

    def loss(p, x, y):
        h = torch.tanh(x @ p["z_first"])
        return ((h @ p["a_second"] - y) ** 2).mean()

    captured = capture_training(loss, weights, None, (x, y))
    restored = TrainingIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    assert restored.output_names() == captured.ir.output_names()
    assert restored.lifetimes() == captured.ir.lifetimes()
    hashes = {}
    for retained in (False, True):
        original = lower(captured.ir, retain_all=retained)
        roundtrip = lower(restored, retain_all=retained)
        assert original.source == roundtrip.source
        assert original.cache_key == roundtrip.cache_key
        assert original.manifest["outputs"] == roundtrip.manifest["outputs"]
        hashes["diagnostic" if retained else "packed"] = original.cache_key
    return {"parameter_order": captured.parameter_names, "source_hashes": hashes,
            "sorted_json_output_and_lifetime_identity": True}


def test_explicit_placeholder_storage_alias_refused():
    """Cover the observed make_fx specialization of duplicate input arguments."""
    w, x, y = _fixture()
    def loss(p, x, y):
        return ((x @ p["w"] + 0.3 * y) ** 2).mean()
    duplicate = _refusal(lambda: capture_training(loss, {"w": w}, None, (x, x)), {"PLACEHOLDER_STORAGE"})
    view = _refusal(lambda: capture_training(loss, {"w": w}, None, (x, x.view_as(x))), {"PLACEHOLDER_STORAGE"})
    parameter_state = _refusal(lambda: capture_training(loss, {"w": w}, {"w": w}, (x, y)), {"PLACEHOLDER_STORAGE"})
    parameter_input = _refusal(lambda: capture_training(loss, {"w": w}, None, (w, torch.zeros_like(w))), {"PLACEHOLDER_STORAGE"})
    m = torch.zeros_like(w)
    state_input = _refusal(lambda: capture_training(loss, {"w": w}, {"w": m}, (m, torch.zeros_like(m))), {"PLACEHOLDER_STORAGE"})
    # Distinct capture storage preserves both placeholders. Pure readonly uses
    # may subsequently receive equal values or alias each other when executed.
    captured = capture_training(loss, {"w": w}, None, (x, y))
    for rebound in ((x, x.clone()), (x, x)):
        rebound_args = (*captured.example_args[:-2], *rebound)
        actual = captured.graph_module(*rebound_args)
        expected = captured.step_fn(*rebound_args)
        for got, ref in zip(actual, expected):
            _close(got, ref)
    return {"duplicate_inputs": duplicate, "shared_storage_view": view,
            "parameter_momentum": parameter_state, "parameter_input": parameter_input,
            "momentum_input": state_input, "distinct_capture_readonly_rebinding_matches": True}


def test_graphmodule_load_fresh_process():
    """Cover GPU-01's initialization-order failure, hidden by same-process capture."""
    w, x, y = _fixture()
    c = capture_training(lambda p, x, y: ((x @ p["w"] - y) ** 2).mean(),
                         {"w": w}, None, (x, y))
    expected = c.graph_module(*c.example_args)
    with tempfile.TemporaryDirectory(prefix="latch-ops-fresh-load-") as temporary:
        folder = Path(temporary)
        torch.save(c.graph_module, folder / "graph.pt")
        torch.save(c.example_args, folder / "args.pt")
        script = """
import sys
from pathlib import Path
import torch
torch.set_num_threads(1)
# PyTorch 2.9.1 FX deserialization retraces source. Its tracer consults this
# submodule attribute, which normal same-process capture already initialized.
import torch._higher_order_ops.triton_kernel_wrap
folder = Path(sys.argv[1])
graph = torch.load(folder / 'graph.pt', weights_only=False)
args = torch.load(folder / 'args.pt', weights_only=True)
torch.save(graph(*args), folder / 'outputs.pt')
"""
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-B", "-c", script, str(folder)],
                              capture_output=True, text=True, timeout=60, env=env)
        if proc.returncode:
            raise AssertionError(f"fresh-process FX load failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        actual = torch.load(folder / "outputs.pt", weights_only=True)
        assert len(actual) == len(expected)
        for got, ref in zip(actual, expected):
            assert torch.equal(got, ref)
    return {"fresh_interpreter": True, "same_process_capture_dependency": False,
            "explicit_import": "torch._higher_order_ops.triton_kernel_wrap", "output_bitwise_equal": True}


def test_module_tied_parameter_functional_call():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            w, _, _ = _fixture()
            self.first = torch.nn.Linear(4, 4, bias=False)
            self.second = torch.nn.Linear(4, 4, bias=False)
            self.first.weight = torch.nn.Parameter(w)
            self.second.weight = self.first.weight

        def forward(self, x):
            return self.second(torch.tanh(self.first(x)))

    model = Shared()
    _, x, y = _fixture()
    loss = lambda m, x, y: ((m(x) - y) ** 2).mean()
    fn, p = module_loss(model, loss)
    c = capture_training(fn, p, None, (x, y))
    assert len(c.parameter_names) == 1
    got = c.unpack(c.graph_module(*c.example_args))
    expected_loss = loss(model, x, y)
    (expected_grad,) = torch.autograd.grad(expected_loss, tuple(model.parameters()))
    _close(got["loss"], expected_loss)
    _close(next(iter(got["grads"].values())), expected_grad)
    assert model.first.weight is model.second.weight


def test_mutation_refused():
    w, x, y = _fixture()
    def loss(p, x, y):
        x.add_(1.0)
        return ((x @ p["w"] - y) ** 2).mean()
    return _refusal(lambda: capture_training(loss, {"w": w}, None, (x, y)), {"MUTATION"})


def test_private_temporary_fill_functionalized():
    w, x, y = _fixture()
    def loss(p, x, y):
        private = torch.empty_like(x)
        private.fill_(0.25)
        return (((x + private) @ p['w'] - y) ** 2).mean()
    captured = capture_training(loss, {'w': w}, None, (x, y))
    actual = captured.unpack(captured.graph_module(*captured.example_args))
    weight = w.clone().requires_grad_(True)
    expected = (((x + 0.25) @ weight - y) ** 2).mean()
    gradient, = torch.autograd.grad(expected, weight)
    _close(actual['loss'], expected)
    _close(actual['grads']['w'], gradient)
    assert all(not getattr(node.target, '_schema', None) or not node.target._schema.is_mutable
               for node in captured.graph_module.graph.nodes if node.op == 'call_function')
    return {'private_fill_removed': True, 'external_input_mutation_still_refused_by_separate_test': True}


def test_rng_refused_even_if_unused():
    w, x, y = _fixture()
    def loss(p, x, y):
        torch.rand_like(x)
        return ((x @ p["w"] - y) ** 2).mean()
    return _refusal(lambda: capture_training(loss, {"w": w}, None, (x, y)), {"RNG"})


def test_runtime_tensor_branch_refused():
    w, x, y = _fixture()
    def loss(p, x, y):
        if x.sum() > 0:
            return ((x @ p["w"] - y) ** 2).mean()
        return ((x @ p["w"] + y) ** 2).mean()
    return _refusal(lambda: capture_training(loss, {"w": w}, None, (x, y)), {"RUNTIME_CONTROL"})


def test_storage_alias_and_state_conflict_refused():
    w, x, y = _fixture()
    loss = lambda p, x, y: ((x @ p["w"] - y) ** 2).mean()
    a = _refusal(lambda: capture_training(loss, {"w": w, "view": w.view_as(w)}, None, (x, y)),
                 {"PARAMETER_STORAGE"})
    b = _refusal(lambda: capture_training(loss, {"w": w, "tied": w},
                                          {"w": torch.zeros_like(w), "tied": torch.ones_like(w)}, (x, y)),
                 {"STATE_ALIAS"})
    return {"storage": a, "state": b}


def test_nonscalar_loss_refused():
    w, x, y = _fixture()
    return _refusal(lambda: capture_training(lambda p, x, y: (x @ p["w"] - y) ** 2,
                                             {"w": w}, None, (x, y)), {"LOSS"})


def test_zero_momentum_refused():
    return _refusal(lambda: SGDConfig(momentum=0).validate(), {"OPTIMIZER"})


def test_lazy_tensor_bits_refused():
    w, x, y = _fixture()
    loss = lambda p, x, y: ((x @ p["w"] - y) ** 2).mean()
    negative_w = torch._neg_view(w)
    negative_x = torch._neg_view(x)
    assert negative_w.is_neg() and negative_w.data_ptr() == w.data_ptr()
    assert negative_w.stride() == w.stride()
    parameter = _refusal(lambda: capture_training(loss, {"w": negative_w}, None, (x, y)), {"LAZY_VIEW"})
    momentum = _refusal(lambda: capture_training(loss, {"w": w}, {"w": negative_w}, (x, y)), {"LAZY_VIEW"})
    inputs = _refusal(lambda: capture_training(loss, {"w": w}, None, (negative_x, y)), {"LAZY_VIEW"})
    conjugate_x = x.to(torch.complex64).conj()
    assert conjugate_x.is_conj()
    conjugate = _refusal(lambda: capture_training(loss, {"w": w}, None, (conjugate_x, y)), {"LAZY_VIEW"})
    # Explicitly resolving the metadata restores an ordinary admissible tensor.
    resolved = capture_training(loss, {"w": negative_w.resolve_neg()}, None, (x, y))
    actual = resolved.graph_module(*resolved.example_args)
    expected = resolved.step_fn(*resolved.example_args)
    for got, ref in zip(actual, expected):
        _close(got, ref)
    return {"parameter": parameter, "momentum": momentum, "inputs": inputs,
            "conjugate_input": conjugate, "resolved_negative_admitted": True}


def test_inner_learning_second_derivative_capture():
    """Nested AD reaches the same outer derivative as a quadratic Hessian formula."""
    w, x, y = _fixture()
    q, u = x * 0.6 + 0.1, y * 0.2 - 0.3
    alpha = 0.05

    def inner(p, x, y):
        return ((x @ p["w"] - y) ** 2).sum()

    def outer(p, x, y, q, u):
        g = torch.func.grad(inner)(p, x, y)
        learned = p["w"] - alpha * g["w"]
        return ((q @ learned - u) ** 2).sum()

    c = capture_training(outer, {"w": w}, None, (x, y, q, u))
    got = c.unpack(c.graph_module(*c.example_args))
    wd, xd, yd, qd, ud = (v.double() for v in (w, x, y, q, u))
    learned = wd - alpha * (2 * xd.T @ (xd @ wd - yd))
    g_outer = 2 * qd.T @ (qd @ learned - ud)
    expected = g_outer - alpha * (2 * xd.T @ xd @ g_outer)
    _close(got["grads"]["w"].double(), expected)
    return {"scope": "CPU capture and graph execution only; no native qualification",
            "max_abs_error": float((got["grads"]["w"].double() - expected).abs().max()),
            "operations": sorted({op.target for op in c.ir.operations})}


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
            "count": len(results), "tests": results}


if __name__ == "__main__":
    receipt = run()
    print(json.dumps(receipt, indent=2))
    raise SystemExit(0 if receipt["passed"] else 1)
