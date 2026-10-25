"""LATCH-STEP-11 torch.cond frontend contracts; cloud CPU only (pinned torch 2.9.1).

The independent reference is ordinary PyTorch: eager torch.cond with autograd and
torch.optim. No captured arithmetic or native code participates in a reference.
"""
from __future__ import annotations

import json

import torch
from torch import nn
from torch.func import functional_call

from megakernel_lab.operation_compiler.api import capture_module
from megakernel_lab.operation_compiler.capture import CapturedConditional, CapturedTraining, capture_conditional_training, capture_training
from megakernel_lab.operation_compiler.ir import CaptureError, UniformConditionalIR
from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig, initialize_optimizer_state
from megakernel_lab.operation_compiler.optimizer_reference import adamw_reference_step


class Gated(nn.Module):
    """Shared parameter in both branches; `up`/`down` true-only; `mix` false-only."""

    def __init__(self, d_in=6, d=8, hidden=12, d_out=5):
        super().__init__()
        self.inp = nn.Linear(d_in, d, bias=False)
        self.shared = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        self.mix = nn.Linear(d, d)
        self.head = nn.Linear(d, d_out, bias=False)

    def deep(self, h):
        return h + self.down(torch.nn.functional.silu(self.up(self.shared(h))))

    def cheap(self, h):
        return h + 0.5 * torch.tanh(self.mix(self.shared(h)))

    def forward(self, x, flag):
        h = torch.tanh(self.inp(x))
        h = torch.cond(flag, self.deep, self.cheap, (h,))
        return self.head(h)


def _mse(prediction, target):
    return (prediction - target).square().mean()


def _fixture(seed=1101):
    torch.manual_seed(seed)
    model = Gated()
    gen = torch.Generator().manual_seed(seed + 1)
    x, target = torch.randn(4, 6, generator=gen), torch.randn(4, 5, generator=gen)
    return model, x, target


def _functional(model):
    def loss(params, x, flag, target):
        return _mse(functional_call(model, params, (x, flag)), target)
    return loss


def _expect(fn, code):
    try:
        fn()
    except CaptureError as exc:
        assert exc.code == code, (exc.code, str(exc))
        return str(exc)
    raise AssertionError(f"expected refusal {code}")


def test_reference_torch_cond_gradient_semantics():
    """Record what ordinary torch.cond autograd gives a parameter unused by the selected branch."""
    model, x, target = _fixture()
    observed = {}
    for flag in (True, False):
        params = {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}
        loss = _functional(model)(params, x, torch.tensor(flag), target)
        grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True, materialize_grads=False)
        observed[flag] = {name: ("none" if g is None else "zero" if not bool(g.abs().max()) else "nonzero")
                          for name, g in zip(params, grads)}
    # The frontend's union participation must agree with this reference behaviour.
    assert observed[True]["mix.weight"] in ("zero", "none") and observed[False]["up.weight"] in ("zero", "none")
    assert observed[True]["mix.weight"] == "zero" and observed[False]["up.weight"] == "zero", observed
    return observed


def test_capture_returns_region_with_matching_explicit_branches():
    model, x, target = _fixture()
    loss = _functional(model)
    params = dict(model.named_parameters())
    flag = torch.tensor(True)
    config = AdamWConfig(lr=1e-2, weight_decay=0.1)
    state = initialize_optimizer_state(params, config)
    captured = capture_training(loss, params, None, (x, flag, target), config, optimizer_state=state)
    assert isinstance(captured, CapturedConditional)
    meta = captured.ir.metadata["cond_frontend"]
    assert meta["predicate_input"] == 1 and meta["cond_calls"] == 1
    assert captured.ir.inputs[-1] == captured.ir.predicate.name
    assert captured.example_args[-1] is not flag and bool(captured.example_args[-1]) is True

    # Hand-written explicit branches with identical arithmetic (existing two-callable API).
    def specialized(outcome):
        def fn(p, x, target):
            return _mse(_call_specialized(model, p, x, outcome), target)
        return fn
    reference = capture_conditional_training(specialized(True), specialized(False), params, None, (x, target), flag,
                                             config, optimizer_state=state, participation="union")
    for a, b in ((captured.ir.true_branch, reference.ir.true_branch),
                 (captured.ir.false_branch, reference.ir.false_branch)):
        assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    roundtrip = UniformConditionalIR.from_dict(json.loads(json.dumps(captured.ir.to_dict(), sort_keys=True)))
    assert roundtrip.metadata["cond_frontend"] == meta
    return {"branch_ir_identical_to_explicit_union": True, "predicate_input": meta["predicate_input"]}


def _call_specialized(model, params, x, outcome):
    """Same module arithmetic with the branch chosen in Python (explicit-API control)."""
    class Specialized(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, x):
            m = self.model
            h = torch.tanh(m.inp(x))
            return m.head(m.deep(h) if outcome else m.cheap(h))
    wrapper = Specialized()
    return functional_call(wrapper, {f"model.{k}": v for k, v in params.items()}, (x,))


def test_both_outcomes_match_eager_torch_cond_and_adamw():
    model, x, target = _fixture()
    loss = _functional(model)
    params = {k: v.detach().clone() for k, v in model.named_parameters()}
    config = AdamWConfig(lr=1e-2, weight_decay=0.1)
    state = initialize_optimizer_state(params, config)
    captured = capture_training(loss, params, None, (x, torch.tensor(False), target), config, optimizer_state=state)
    names = captured.parameter_names
    counters = {name: 0 for name in names}
    rows = []
    for flag in (True, False, False, True, False, True):
        flat = (*[params[n] for n in names], *[state[n] for n in captured.optimizer_state_names], x, target,
                torch.tensor(flag))
        got = captured.unpack(captured.step_fn(*flat))
        expected = adamw_reference_step(lambda p, x, f, t: loss(p, x, f, t), params, state,
                                        (x, torch.tensor(flag), target), config=config)
        torch.testing.assert_close(got["loss"], expected["loss"], atol=2e-6, rtol=1e-5)
        for name in names:
            torch.testing.assert_close(got["grads"][name], expected["grads"][name], atol=2e-6, rtol=1e-4)
            torch.testing.assert_close(got["params"][name], expected["params"][name], atol=2e-6, rtol=1e-5)
            counters[name] += 1
            assert int(got["optimizer_state"][f"step:{name}"]) == counters[name]
        unused = "mix.weight" if flag else "up.weight"
        assert not bool(got["grads"][unused].abs().max())
        assert not torch.equal(got["params"][unused], params[unused])  # zero gradient still updates (decay)
        rows.append({"flag": flag, "loss": float(got["loss"])})
        params, state = got["params"], got["optimizer_state"]
    assert len({r["loss"] for r in rows}) == len(rows)
    return {"steps": rows, "counters": counters}


def test_module_api_binds_predicate_and_lowers():
    model, x, target = _fixture()
    flag = torch.tensor(True)
    captured = capture_module(model, _mse, (x, flag), (target,), optimizer=SGDConfig(lr=0.05))
    assert isinstance(captured.captured, CapturedConditional)
    indices = captured.input_indices
    assert indices["inputs"][1] == len(captured.ir.inputs) - 1
    assert captured.ir.values[captured.ir.inputs[indices["inputs"][0]]].shape == (4, 6)
    assert captured.ir.values[captured.ir.inputs[indices["targets"][0]]].shape == (4, 5)
    from megakernel_lab.operation_compiler.lower import lower
    program = lower(captured.ir, execution="compose")
    manifest = program.manifest
    assert manifest["predicate_input"] == captured.ir.predicate.name
    assert "if (*reinterpret_cast<const bool*>(args[" in program.kernel_body
    return {"inputs": indices, "source_hash": manifest["source_hash"],
            "barriers": manifest["barrier_count"]}


def test_flat_program_unchanged_type_and_refusal_switch():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 5))
    x, target = torch.randn(4, 6), torch.randn(4, 5)
    a = capture_module(model, _mse, (x,), (target,))
    assert isinstance(a.captured, CapturedTraining)
    params = dict(model.named_parameters())
    loss = lambda p, x, t: _mse(functional_call(model, p, (x,)), t)
    b = capture_training(loss, params, None, (x, target), conditional="refuse")
    c = capture_training(loss, params, None, (x, target))
    assert json.dumps(b.ir.to_dict(), sort_keys=True) == json.dumps(c.ir.to_dict(), sort_keys=True)
    assert torch.cond is not None and torch.cond.__name__ == "cond"  # patch restored
    m, x2, t2 = _fixture()
    _expect(lambda: capture_training(_functional(m), dict(m.named_parameters()), None, (x2, torch.tensor(True), t2),
                                     conditional="refuse"), "COND_SCOPE")


def test_stateful_and_two_calls_same_predicate():
    torch.manual_seed(5)
    w1, w2 = torch.randn(6, 6) * 0.2, torch.randn(6, 6) * 0.2
    params = {"w1": w1, "w2": w2}
    x = torch.randn(3, 6)

    def loss(p, s, x, flag):
        h = torch.cond(flag, lambda h: torch.tanh(h @ p["w1"]), lambda h: h @ p["w2"], (x,))
        h = torch.cond(flag, lambda h: h * 2.0, lambda h: h - 1.0, (h,))
        value = h.square().mean()
        return value, {"count": s["count"] + 1, "last": value.detach().reshape(1)}
    state = {"count": torch.tensor(0, dtype=torch.int64), "last": torch.zeros(1)}
    captured = capture_training(loss, params, None, (x, torch.tensor(True)), state=state)
    assert captured.ir.metadata["cond_frontend"]["cond_calls"] == 2
    for flag in (True, False):
        flat = (w1, w2, torch.zeros_like(w1), torch.zeros_like(w2), *state.values(), x, torch.tensor(flag))
        got = captured.unpack(captured.step_fn(*flat))
        p = {"w1": w1.clone().requires_grad_(), "w2": w2.clone().requires_grad_()}
        ref, _ = loss(p, state, x, torch.tensor(flag))
        grads = torch.autograd.grad(ref, list(p.values()), allow_unused=True)
        for name, g in zip(p, grads):
            torch.testing.assert_close(got["grads"][name], torch.zeros_like(p[name]) if g is None else g)
        assert int(got["state"]["count"]) == 1
    return {"cond_calls": 2}


def test_refusals():
    model, x, target = _fixture()
    params = dict(model.named_parameters())
    refusals = {}

    def computed(p, x, t):
        return _mse(functional_call(model, p, (x, x.sum() > 0)), t)
    refusals["computed_predicate"] = _expect(lambda: capture_training(computed, params, None, (x, target)),
                                             "COND_PREDICATE")
    refusals["vector_predicate"] = _expect(lambda: capture_training(
        _functional(model), params, None, (x, torch.ones(2, dtype=torch.bool), target)), "COND_PREDICATE")
    refusals["float_predicate"] = _expect(lambda: capture_training(
        _functional(model), params, None, (x, torch.tensor(1.0), target)), "COND_PREDICATE")

    def python_bool(p, x, t):
        return _mse(functional_call(model, p, (x, True)), t)
    refusals["python_predicate"] = _expect(lambda: capture_training(python_bool, params, None, (x, target)),
                                           "COND_PREDICATE")

    class Mismatch(Gated):
        def cheap(self, h):
            return (h + 0.5 * torch.tanh(self.mix(self.shared(h))))[:, :1]
    bad = Mismatch()
    refusals["branch_shape"] = _expect(lambda: capture_training(
        _functional(bad), dict(bad.named_parameters()), None, (x, torch.tensor(True), target)), "COND_BRANCH_MISMATCH")

    class Nested(Gated):
        def deep(self, h):
            return torch.cond(self._flag, lambda u: u + 1, lambda u: u - 1, (h,))

        def forward(self, x, flag):
            self._flag = flag
            return super().forward(x, flag)
    nested = Nested()
    refusals["nested"] = _expect(lambda: capture_training(
        _functional(nested), dict(nested.named_parameters()), None, (x, torch.tensor(True), target)), "COND_NESTED")

    class Reads(Gated):
        def forward(self, x, flag):
            return super().forward(x, flag) * flag
    reads = Reads()
    refusals["predicate_arithmetic"] = _expect(lambda: capture_training(
        _functional(reads), dict(reads.named_parameters()), None, (x, torch.tensor(True), target)),
        "COND_PREDICATE_USE")

    def two(p, x, a, b, t):
        h = torch.cond(a, lambda h: h + 1, lambda h: h - 1, (x,))
        h = torch.cond(b, lambda h: h * 2, lambda h: h / 2, (h,))
        return _mse(functional_call(model, p, (h, a)), t)
    refusals["second_predicate"] = _expect(lambda: capture_training(
        two, params, None, (x, torch.tensor(True), torch.tensor(False), target)), "COND_PREDICATE")

    class Buffered(Gated):
        def __init__(self):
            super().__init__()
            self.register_buffer("flag", torch.tensor(True))

        def forward(self, x):
            return super().forward(x, self.flag)
    buffered = Buffered()
    refusals["buffer_predicate"] = _expect(lambda: capture_module(buffered, _mse, (x,), (target,)), "COND_PREDICATE")

    class PerExample(Gated):
        def forward(self, x, flag):
            h = torch.tanh(self.inp(x))
            return self.head(torch.cond(flag, self.deep, self.cheap, [h, 3]))
    per = PerExample()
    refusals["non_tensor_operand"] = _expect(lambda: capture_training(
        _functional(per), dict(per.named_parameters()), None, (x, torch.tensor(True), target)), "COND_OPERANDS")
    assert torch.cond.__name__ == "cond"
    return refusals
