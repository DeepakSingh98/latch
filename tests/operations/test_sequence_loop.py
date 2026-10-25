"""Modal CPU contract tests for C-OPS4-A compiler additions.

Sequence inputs (scan xs) for bounded loops, the optional init saved tape,
the Muon/AdamW optimizer, logsumexp lowering and transposed FP32 products.
Independent references use ordinary eager autograd; no compiler code path
is used to build an expected value.
"""
from __future__ import annotations

import torch

from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.loop import BoundedLoopIR, capture_loop_training
from megakernel_lab.operation_compiler.muon import MuonConfig, newton_schulz

STEPS = 4


def fixture():
    g = torch.Generator().manual_seed(4401)
    params = {"embed": torch.randn(8, 16, generator=g) * 0.2, "mix": torch.randn(16, 16, generator=g) * 0.2,
              "write": torch.randn(16, 16, generator=g) * 0.2, "gain": torch.ones(16)}
    x = torch.randn(STEPS * 3, 8, generator=g)
    y = torch.randint(0, 16, (STEPS * 3,), generator=g)
    sc = torch.tensor([1e-2, 1 - 1e-3, 1e-2, 1.0, -5e-2, 0.5])
    return params, (x, y, sc)


def sequence(p, x, y, sc):
    h = torch.tanh(x @ p["embed"]) * p["gain"]
    return {"h": h.reshape(STEPS, 3, 16).contiguous(), "y": y.reshape(STEPS, 3).contiguous()}


def init(p, x, y, sc):
    return {"W": p["write"].clone(), "loss": torch.zeros((), dtype=torch.float32)}


def body(p, carry, state, xs, x, y, sc):
    logits = (xs["h"] @ carry["W"]) @ p["mix"]
    ce = torch.logsumexp(logits, -1) - logits.gather(-1, xs["y"][:, None])[:, 0]
    # score before write: the write uses this step's (already scored) labels
    grad = xs["h"].T @ (torch.softmax(logits, -1) - torch.zeros_like(logits).scatter(-1, xs["y"][:, None], 1.0))
    return {"W": carry["W"] - sc[5] * grad @ p["mix"].T, "loss": carry["loss"] + ce.sum()}, {}


def terminal(p, carry, x, y, sc):
    return carry["loss"] / (STEPS * 3)


def config():
    return MuonConfig(muon_parameters=("embed", "mix", "write"), decay_parameters=(), schedule_input=2)


def eager(params, inputs, count):
    p = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    xs = sequence(p, *inputs)
    carry = init(p, *inputs)
    for t in range(count):
        carry, _ = body(p, carry, {}, {k: v[t] for k, v in xs.items()}, *inputs)
    loss = terminal(p, carry, *inputs)
    names = list(p)
    if not loss.requires_grad:
        return loss.detach(), {k: torch.zeros_like(p[k]) for k in names}
    grads = torch.autograd.grad(loss, [p[k] for k in names], allow_unused=True)
    return loss.detach(), {k: torch.zeros_like(p[k]) if g is None else g for k, g in zip(names, grads)}


def test_sequence_loop_matches_eager_autograd():
    params, inputs = fixture()
    for storage, init_storage in (("save_body_tape", "recompute"), ("save_boundaries", "recompute"),
                                  ("save_body_tape", "save_tape")):
        cap = capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(STEPS), config(),
                                    max_steps=STEPS, storage_mode=storage, sequence_fn=sequence,
                                    init_storage=init_storage)
        assert [s["name"] for s in cap.ir.metadata["sequence_specs"]] == ["h", "y"]
        assert (cap.ir.init_saved is not None) == (init_storage == "save_tape")
        for count in (0, 1, STEPS):
            flat = (*cap.example_args[:-1], torch.tensor(count))
            got = cap.unpack(cap.graph_step(*flat))
            loss, grads = eager(params, inputs, count)
            assert torch.allclose(got["loss"], loss, rtol=1e-6, atol=1e-7)
            for k, g in grads.items():
                assert torch.allclose(got["grads"][k], g, rtol=1e-5, atol=1e-6), (storage, init_storage, count, k)
        restored = BoundedLoopIR.from_dict(cap.ir.to_dict())
        assert restored.metadata["sequence_specs"] == cap.ir.metadata["sequence_specs"]


def test_sequence_rejections():
    params, inputs = fixture()
    bad = lambda p, x, y, sc: {"h": torch.zeros(STEPS + 1, 3)}
    try:
        capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(STEPS), config(),
                              max_steps=STEPS, sequence_fn=bad)
    except CaptureError as error:
        assert error.code == "LOOP_SEQUENCE"
    else:
        raise AssertionError("wrong leading dimension accepted")


def donor_update(params, grads, state, sc, cfg):
    """Independent restatement of the donor S106B Engine._update for the fixture's Muon/AdamW split."""
    names = list(params)
    norm = torch.sqrt(sum((grads[n] * grads[n]).sum() for n in names))
    coef = torch.clamp(cfg.clip / (norm + 1e-6), max=1.0)
    out, new_state = {}, {}
    for n in names:
        if n in cfg.muon_parameters:
            G = grads[n] * coef
            B = state[f"momentum:{n}"] + (1 - cfg.momentum) * (G - state[f"momentum:{n}"])
            U = newton_schulz((G + cfg.momentum * (B - G))[None], cfg)[0].float()
            U = U * max(1.0, params[n].shape[0] / params[n].shape[1]) ** 0.5
            out[n] = params[n] * sc[1] + U * sc[4]
            new_state[f"momentum:{n}"] = B
        else:
            h = grads[n] * coef
            m = state[f"exp_avg:{n}"] + (1 - 0.9) * (h - state[f"exp_avg:{n}"])
            v = state[f"exp_avg_sq:{n}"] * 0.95 + (h * h) * (1 - 0.95)
            out[n] = params[n] - (sc[2] * m) / (torch.sqrt(v) * sc[3] + 1e-8)
            new_state[f"exp_avg:{n}"], new_state[f"exp_avg_sq:{n}"] = m, v
    return out, new_state


def test_muon_optimizer_matches_independent_update():
    params, inputs = fixture()
    cap = capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(STEPS), config(),
                                max_steps=STEPS, sequence_fn=sequence)
    got = cap.unpack(cap.graph_step(*cap.example_args))
    state = {k: torch.zeros_like(params[k.split(":", 1)[1]]) for k in cap.optimizer_state_names}
    expected, expected_state = donor_update(params, got["grads"], state, inputs[2], config())
    for k, v in expected.items():
        assert torch.allclose(got["params"][k], v, rtol=1e-5, atol=1e-7), k
    for k, v in expected_state.items():
        assert torch.allclose(got["optimizer_state"][k], v, rtol=1e-6, atol=1e-8), k


def test_transposed_product_plan():
    from megakernel_lab.operation_compiler.lower import TensorLayout
    from megakernel_lab.operation_compiler.matrix_select import plan
    a = TensorLayout((512, 256, 512), (256 * 512, 512, 1), "p")
    b = TensorLayout((512, 512, 32), (512 * 32, 32, 1), "v")
    selected, reason = plan("p", "v", "o", a, b, (512, 256, 32), ident=lambda s: s)
    assert selected is not None and selected["metadata"]["transposed_product"], reason
    assert "SCM = 1LL" in selected["helpers"] and "SCN = 32LL" in selected["helpers"]


def test_logsumexp_lowering_emits_row_reduction():
    from megakernel_lab.operation_compiler.transformer_lower import emit_operation
    from megakernel_lab.operation_compiler.lower import TensorLayout
    layouts = {"x": TensorLayout((6, 40), (40, 1), "x")}
    emitted = emit_operation("logsumexp.default", ({"ref": "x"}, [-1]), {}, "out", (6,), layouts,
                             {"x": "float32", "out": "float32"}, {"x": "x"}, read=None)
    assert emitted is not None and emitted[0] == "reduction" and "isinf(maximum)" in emitted[1]


def probe():
    import traceback
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results[name] = "passed"
            except Exception:  # noqa: BLE001
                results[name] = traceback.format_exc()[-3000:]
    return results
