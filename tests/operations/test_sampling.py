"""C-OPS4-B CPU contracts: random fields, extra operations, sampled-training capture/lowering, row tasks.

Run on a Modal CPU worker (torch 2.9.1): python -m pytest -q ops/test_sampling.py
No CUDA is required: lowering is inspected and captured graphs are interpreted on the CPU.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.capture import capture_training
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig, initialize_optimizer_state
from megakernel_lab.operation_compiler.random import field_words, make_key, philox_field
from megakernel_lab.operation_compiler.sampling import capture_sampled_training
from megakernel_lab.operation_compiler.sampling_rows import RowRefusal, row_axes, slice_region, virtualize


def _numpy_words(seed, c2, c3, rows, cols):
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.reference import field_words as reference
    return reference(seed, c2, c3, rows, cols).astype(np.int64)


@pytest.mark.parametrize("seed,c2,c3,shape", [(0, 0, 0, (3, 5)), (2 ** 64 - 1, 7, 2 ** 32 - 1, (4, 9)),
                                              (20260926, 0xFFFFFF01, 3, (6, 2))])
def test_field_words_are_the_counter_grid(seed, c2, c3, shape):
    key = make_key(seed)
    words = philox_field(key, torch.tensor(c2), torch.tensor(c3), list(shape), "bits")
    assert torch.equal(words, torch.as_tensor(_numpy_words(seed, c2, c3, *shape)))
    uniform = philox_field(key, torch.tensor(c2), torch.tensor(c3), list(shape), "uniform")
    assert torch.equal(uniform, ((words >> 8).double() * 2.0 ** -24).float())


def test_field_rows_are_independent_of_extent():
    key = make_key(5)
    full = philox_field(key, torch.tensor(1), torch.tensor(2), [8, 12], "normal")
    assert torch.equal(full[:3], philox_field(key, torch.tensor(1), torch.tensor(2), [3, 12], "normal"))
    assert not torch.equal(full, philox_field(key, torch.tensor(2), torch.tensor(2), [8, 12], "normal"))


def _sampled(B=8, D=6, H=8, K=2):
    torch.manual_seed(0)
    params = {"w": torch.randn(H, D) * 0.3, "b": torch.zeros(H), "v": torch.randn(H) * 0.3}
    state = {"buf": torch.rand(20, D), "ptr": torch.zeros((), dtype=torch.int64),
             "t": torch.zeros((), dtype=torch.int64), "key": make_key(11)}
    data = torch.rand(30, D)

    def energy(p, x):
        return (F.silu((x.to(torch.bfloat16) @ p["w"].to(torch.bfloat16).t()).float() + p["b"]) * p["v"]).sum(-1)

    def init(p, s, data):
        words = philox_field(s["key"], s["t"].new_full((), 9), s["t"], [B, 2], "bits")
        index = torch.div(words[:, 1] * 20, 2 ** 32, rounding_mode="floor")
        return {"x": F.embedding(index, s["buf"])}

    def step(p, c, s, k, data):
        xi = philox_field(s["key"], k, s["t"], [B, D], "normal")
        y = c["x"] + 0.01 * xi
        g = torch.func.grad(lambda v: energy(p, v).sum())(y)
        return {"x": (y - 0.5 * g.clamp(-0.1, 0.1)).clamp(0, 1)}, s

    def loss(p, c, s, data):
        pos = data[:B]
        e = energy(p, torch.cat([pos, c["x"]]))
        slots = torch.remainder(s["ptr"] + torch.arange(B, dtype=torch.int64), 20)
        new = {"buf": s["buf"].index_copy(0, slots, c["x"]), "ptr": torch.remainder(s["ptr"] + B, 20),
               "t": s["t"] + 1, "key": s["key"]}
        return e[:B].mean() - e[B:].mean(), new

    config = AdamWConfig(lr=1e-3, betas=(0.0, 0.999), max_grad_norm=1.0, param_options={"b": {"weight_decay": 0.0}})
    opt = initialize_optimizer_state(params, config)
    captured = capture_sampled_training(init, step, loss, params, (data,), torch.tensor(K), config, max_steps=K,
                                        state=state, optimizer_state=opt)
    args = (*params.values(), *opt.values(), *state.values(), data, torch.tensor(K))
    return captured, args


def test_sampled_capture_matches_eager_and_commits_replay():
    captured, args = _sampled()
    graph, eager = captured.graph_step(*args), captured.step_fn(*args)
    assert all(torch.equal(a, b) for a, b in zip(graph, eager))
    out = captured.unpack(graph)
    assert int(out["executed_count"]) == 2 and int(out["status"]) == 0
    assert int(out["state"]["t"]) == 1 and int(out["state"]["ptr"]) == 8
    assert torch.equal(out["state"]["buf"][:8], out["carry"]["x"])
    assert captured.ir.metadata["body_state_changes"] == []
    clamped = captured.graph_step(*args[:-1], torch.tensor(5))
    assert int(captured.unpack(clamped)["status"]) == 1


def test_sampled_parameter_gradient_stops_at_samples():
    """The loss gradient equals the gradient with the chain output supplied as a constant input."""
    captured, args = _sampled()
    out = captured.unpack(captured.graph_step(*args))
    params = dict(zip(captured.parameter_names, args[:3]))
    x = out["carry"]["x"]

    def energy(p, v):
        return (F.silu((v.to(torch.bfloat16) @ p["w"].to(torch.bfloat16).t()).float() + p["b"]) * p["v"]).sum(-1)
    data = args[-2]
    def objective(p):
        # One energy evaluation over the concatenated rows, as the loss defines it (the BF16 policy rounds a
        # weight-gradient product over [x+; x-] once, not two products separately).
        e = energy(p, torch.cat([data[:8], x]))
        return e[:8].mean() - e[8:].mean()
    grads = torch.func.grad(objective)(params)
    for name in captured.parameter_names:
        assert torch.allclose(out["grads"][name], grads[name], rtol=1e-5, atol=1e-7)


def test_sampled_lowering_emits_one_program_with_new_rules():
    captured, _ = _sampled()
    program = lower(captured.ir)
    source = program.source
    assert program.manifest["format"] == "latch-ops-cuda-sampled-v1"
    for token in ("latch_field_normal", "latch_imod", "latch_ifloordiv", "latch_clamp"):
        assert token in source
    assert source.count("__global__") == 1


def test_clip_coefficient_matches_definition():
    torch.manual_seed(1)
    params = {"a": torch.randn(4, 3), "b": torch.randn(3)}
    x = torch.randn(5, 4)
    config = AdamWConfig(lr=0.1, betas=(0.0, 0.999), max_grad_norm=0.5)
    captured = capture_training(lambda p, x: ((x @ p["a"] + p["b"]) ** 2).sum(), params, None, (x,), config,
                                optimizer_state=initialize_optimizer_state(params, config))
    out = captured.unpack(captured.step_fn(*captured.example_args))
    g = out["grads"]
    norm = math.sqrt(sum(float((v.double() ** 2).sum()) for v in g.values()))
    coef = min(0.5 / (norm + 1e-6), 1.0)
    for name in params:
        assert torch.allclose(out["optimizer_state"][f"exp_avg:{name}"].double(), g[name].double() * coef, rtol=1e-5)


def test_row_axes_accept_the_recipe_body_and_refuse_row_mixing():
    captured, _ = _sampled()
    body = captured.ir.body
    n = len(captured.parameter_names) + len(captured.ir.metadata.get("invariant_casts", []))
    carry = {body.inputs[n]}
    axes = row_axes(body, carry, 8)
    assert all(axes[o] == 0 for o in body.outputs)
    sliced = slice_region(body, axes, 8, 4)
    assert sliced.values[sliced.outputs[0]].shape[0] == 4
    program = lower(sliced)
    text, problems, _ = virtualize(program.kernel_body)
    assert not problems and "blockIdx.x" not in text and "grid.sync()" not in text

    from megakernel_lab.operation_compiler.loop import _capture_region
    region, _ = _capture_region(lambda x: (x - x.mean(0, keepdim=True),), (torch.randn(8, 6),), "mix", ["carry:x"])
    with pytest.raises(RowRefusal):
        row_axes(region, {region.inputs[0]}, 8)
    region, _ = _capture_region(lambda x, w: (x @ (x.t() @ w),), (torch.randn(8, 6), torch.randn(8, 6)), "contract",
                                ["carry:x", "parameter:w"])
    with pytest.raises(RowRefusal):
        row_axes(region, {region.inputs[0]}, 8)


def test_index_copy_lowering_refuses_non_row_copies():
    from megakernel_lab.operation_compiler.loop import _capture_region
    region, _ = _capture_region(lambda b, i, x: (b.index_copy(1, i, x),),
                                (torch.randn(4, 6), torch.tensor([0, 2]), torch.randn(4, 2)), "copy",
                                ["input:b", "input:i", "input:x"])
    with pytest.raises(LoweringError):
        lower(region)


def test_bf16_tail_products_use_tensor_core_tail_body():
    """K or N a multiple of 16 but not of 32 selects popsc::bmm_tail instead of the SIMT fallback."""
    from megakernel_lab.operation_compiler.loop import _capture_region
    a, w = torch.randn(64, 784), torch.randn(96, 784)
    region, _ = _capture_region(lambda a, w: ((a.to(torch.bfloat16) @ w.to(torch.bfloat16).t()).float(),
                                              (a.to(torch.bfloat16)[:, :96] @ w.to(torch.bfloat16)).float()),
                                (a, w), "tails", ["input:a", "input:w"])
    program = lower(region)
    gemms = [p for p in program.manifest["phases"] if p["kind"] == "gemm"]
    engines = [m["engine"] for p in gemms for m in p.get("members", [p])]
    assert engines and all(str(e).startswith("popsc_tail_bf16_") for e in engines), engines
    assert program.source.count("popsc::bmm_tail<") == 2 and "generic_bf16_simt" not in str(gemms)


def test_input_gradient_step_drops_the_unused_forward_value():
    """grad(sum E) seeds ones_like(sum): rewritten to full(shape), so the discarded E is not computed."""
    captured, args = _sampled()
    body = captured.ir.body
    targets = [op.target for op in body.operations]
    assert body.metadata["shape_only_constants_rewritten"] >= 1
    assert "aten.ones_like.default" not in targets and "aten.sum.default" not in targets
    graph, eager = captured.graph_step(*args), captured.step_fn(*args)
    assert all(torch.equal(a, b) for a, b in zip(graph, eager))
