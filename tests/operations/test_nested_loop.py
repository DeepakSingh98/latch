"""CPU qualification for actual inner/outer loops (run on the lane CPU VM)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import NestedLoopBody, SGDConfig, capture_loop_training
from megakernel_lab.operation_compiler.loop import BoundedLoopIR
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.ir import CaptureError, _contiguous_strides
from megakernel_lab.operation_compiler.random import dropout, make_rng_state


def fixture(*, width=4, batch=2, storage_mode="save_body_tape"):
    gen = torch.Generator().manual_seed(2041)
    params = {
        "shared": torch.randn(width, width, generator=gen) * 0.08,
        "bias": torch.randn(width, generator=gen) * 0.02,
        "head": torch.randn(width, width, generator=gen) * 0.13,
        "unused": torch.randn(width, generator=gen) * 0.1,
    }
    params["tied"] = params["shared"]
    inputs = (
        torch.randn(batch, width, generator=gen) * 0.2,
        torch.randn(batch, width, generator=gen) * 0.1,
        torch.tensor(0),
    )
    state = {"rng": make_rng_state(0xFACE1234DEADBEEF, (1 << 33) + 17), "visits": torch.tensor(0)}

    def init(p, x, target, inner_count):
        return {"h": F.linear(x, p["shared"], p["bias"]), "aux": x.new_zeros(())}

    def inner_init(p, outer, x, target, inner_count):
        return {
            "hidden": outer["h"] + 0.12 * F.linear(outer["h"], p["tied"], p["bias"]),
            "loss": outer["aux"],
        }

    def inner_body(p, outer, inner, state, x, target, inner_count):
        h = inner["hidden"] + 0.2 * torch.tanh(
            F.linear(inner["hidden"], p["shared"], p["bias"]) + 0.1 * outer["h"]
        )
        h, rng = dropout(h, state["rng"], 0.25)
        # The previous inner hidden value is read after computing the new one.
        loss = (
            inner["loss"]
            + 0.13 * ((F.linear(h, p["head"]) - target) ** 2).mean()
            + 0.03 * inner["hidden"].square().mean()
        )
        return {"hidden": h, "loss": loss}, {"rng": rng, "visits": state["visits"] + 1}

    def finish(p, outer, inner, x, target, inner_count):
        h = inner["hidden"] + 0.11 * torch.tanh(F.linear(inner["hidden"], p["tied"], p["bias"]))
        return {"h": h, "aux": inner["loss"] + 0.07 * (h - outer["h"]).square().mean()}

    def terminal(p, carry, x, target, inner_count):
        return (F.linear(carry["h"], p["head"]) - target).square().mean() + carry["aux"]

    body = NestedLoopBody(inner_init, inner_body, finish, count_input=2, max_steps=4)
    config = SGDConfig(lr=0.015, momentum=0.83, weight_decay=0.025)
    captured = capture_loop_training(
        init,
        body,
        terminal,
        params,
        None,
        inputs,
        torch.tensor(0),
        config,
        state=state,
        max_steps=3,
        storage_mode=storage_mode,
        init_storage="save_tape",
    )
    return captured, (init, body, terminal)


def eager(captured, functions, flat, *, truncate_outer=False, truncate_inner=False, stale_carry=False):
    initial, nested, terminal = functions
    n, nm, ns = len(captured.parameter_names), len(captured.optimizer_state_names), len(captured.state_names)
    p = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in zip(captured.parameter_names, flat[:n])
    }
    mapped = {a: p[k] for a, k in captured.ir.metadata["parameter_aliases"].items()}
    state = {name: value.clone() for name, value in zip(captured.state_names, flat[n + nm : n + nm + ns])}
    data = flat[n + nm + ns : -1]
    requested = int(flat[-1].item())
    count = min(max(requested, 0), captured.ir.max_steps)
    inner_requested = int(data[nested.count_input].item())
    inner_count = min(max(inner_requested, 0), nested.max_steps)
    carry = initial(mapped, *data)
    for i in range(count):
        if truncate_outer and i:
            carry = {k: v.detach() for k, v in carry.items()}
        inner = nested.init_fn(mapped, carry, *data)
        for j in range(inner_count):
            if truncate_inner and j:
                inner = {k: v.detach() for k, v in inner.items()}
            new, state = nested.body_fn(mapped, carry, inner, state, *data)
            inner = inner if stale_carry else new
        carry = nested.finish_fn(mapped, carry, inner, *data)
    loss = terminal(mapped, carry, *data)
    gradients = torch.autograd.grad(loss, tuple(p.values()), allow_unused=True)
    gradients = tuple(torch.zeros_like(v) if g is None else g for v, g in zip(p.values(), gradients))
    cfg = captured.config
    optimizer = torch.optim.SGD(
        p.values(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay, foreach=False
    )
    for value, grad, old in zip(p.values(), gradients, flat[n : n + nm]):
        value.grad = grad.clone()
        optimizer.state[value]["momentum_buffer"] = old.clone()
    optimizer.step()
    return (
        loss.detach(),
        *(v.detach() for v in gradients),
        *(v.detach().clone() for v in p.values()),
        *(optimizer.state[v]["momentum_buffer"].clone() for v in p.values()),
        *(state[k] for k in captured.state_names),
        *(carry[k].detach() for k in captured.carry_names),
        flat[-1].new_tensor(count),
        flat[-1].new_tensor(int(requested != count or inner_requested != inner_count)),
    )


def assert_close(actual, expected):
    error = 0.0
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        if a.is_floating_point():
            torch.testing.assert_close(a, b, rtol=4e-5, atol=4e-7)
            error = max(error, float((a - b).abs().max()))
        else:
            assert torch.equal(a, b), (a, b)
    return error


def test_counts_and_saved_recomputed_histories():
    capture, functions = fixture()
    rows = []
    originals = tuple(v.clone() for v in capture.example_args)
    for outer_count, inner_count in (
        (0, 0),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 3),
        (3, 1),
        (3, 4),
        (-1, 2),
        (2, -1),
        (7, 8),
    ):
        args = list(capture.example_args)
        args[-1], args[-2] = torch.tensor(outer_count), torch.tensor(inner_count)
        reference = eager(capture, functions, args)
        eager_error = assert_close(capture.step_fn(*args), reference)
        saved = capture.graph_step(*args, storage_mode="save_body_tape")
        replay = capture.graph_step(*args, storage_mode="recompute_boundaries")
        error = max(eager_error, assert_close(saved, reference), assert_close(replay, reference))
        assert all(torch.equal(a, b) for a, b in zip(saved, replay))
        out = capture.unpack(saved)
        assert out["state"]["visits"].item() == min(max(outer_count, 0), 3) * min(max(inner_count, 0), 4)
        rows.append(
            {
                "outer": outer_count,
                "inner": inner_count,
                "max_absolute_error": error,
                "save_recompute_bitwise": True,
            }
        )
    assert all(torch.equal(a, b) for a, b in zip(originals, capture.example_args))
    return rows


def test_history_controls_and_serialization():
    capture, functions = fixture()
    args = list(capture.example_args)
    args[-1], args[-2] = torch.tensor(3), torch.tensor(3)
    good = eager(capture, functions, args)
    controls = {}
    for option in ("truncate_outer", "truncate_inner", "stale_carry"):
        wrong = eager(capture, functions, args, **{option: True})
        gap = float((wrong[1] - good[1]).abs().max())
        assert gap > 1e-5, (option, gap)
        controls[option] = gap
    assert len(capture.parameter_names) == 4 and capture.ir.metadata["parameter_aliases"]["tied"] == "shared"
    # Doubling the shared contribution is distinguishable from the exact history.
    assert float(good[1].abs().max()) > 1e-5
    data = json.loads(json.dumps(capture.ir.to_dict(), sort_keys=True))
    restored = BoundedLoopIR.from_dict(data)
    assert json.loads(json.dumps(restored.to_dict(), sort_keys=True)) == data
    for label in ("body_vjp_saved",):
        regions = getattr(restored, label).regions
        assert not any(
            op.target.startswith("latch_ops.philox_") for op in regions["body_vjp_saved"].operations
        )
    return controls


def test_lowered_runtime_control():
    rows = []
    for mode in ("save_body_tape", "recompute_boundaries"):
        capture, _ = fixture(storage_mode=mode)
        program = lower(capture.ir, execution="compose", precision_policy="strict")
        assert "for (int iteration = 0; iteration < count;" in program.source
        assert "for (int inner_iteration=0; inner_iteration<inner_count;" in program.source
        assert "for (int inner_reverse=inner_count-1;" in program.source
        expected = "recursive_staged"
        assert program.manifest["loop_schedule_policy"]["selected"] == expected
        rows.append(
            {
                "mode": mode,
                "arena_bytes": program.manifest["arena_bytes"],
                "body_tape_bytes": program.manifest["body_tape_bytes"],
                "source_hash": program.manifest["source_hash"],
            }
        )
    assert rows[1]["arena_bytes"] < rows[0]["arena_bytes"]
    return rows


def test_scope_refusals():
    p = {"weight": torch.ones(2)}
    inputs = (torch.ones(2), torch.tensor(1))
    initial = lambda p, x, count: {"h": x * p["weight"]}
    inner_init = lambda p, outer, x, count: {"h": outer["h"]}
    inner_body = lambda p, outer, inner, state, x, count: (
        {"h": (inner["h"] * p["weight"]).bfloat16().float()},
        state,
    )
    finish = lambda p, outer, inner, x, count: {"h": inner["h"]}
    terminal = lambda p, carry, x, count: carry["h"].square().mean()
    nested = NestedLoopBody(inner_init, inner_body, finish, count_input=1, max_steps=2)
    try:
        capture_loop_training(initial, nested, terminal, p, None, inputs, torch.tensor(1), max_steps=2)
    except CaptureError as exc:
        assert exc.code == "NESTED_PRECISION", str(exc)
    else:
        raise AssertionError("internal BF16 arithmetic was silently admitted")
    capture, _ = fixture()
    data = capture.ir.body_saved.to_dict()
    data["metadata"] = {**data["metadata"], "max_steps": -1}
    from megakernel_lab.operation_compiler.nested_loop import NestedOperationIR

    try:
        NestedOperationIR.from_dict(data)
    except CaptureError as exc:
        assert exc.code == "NESTED_BOUND", str(exc)
    else:
        raise AssertionError("invalid serialized nested bound was admitted")
    return {"internal_bf16": "refused", "negative_serialized_bound": "refused"}


def transposed_fixture(*, noncontiguous_input=False):
    params = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) * 0.03 + 0.2}
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3) * 0.02
    if noncontiguous_input:
        x = x.t().contiguous().t()
    inputs = (x, torch.tensor(2))

    def initial(p, x, count):
        return {"h": (x * p["weight"]).t()}

    def inner_init(p, outer, x, count):
        return {"z": outer["h"].t()}

    def inner_body(p, outer, inner, state, x, count):
        return {"z": inner["z"] + 0.1 * outer["h"].t() * p["weight"]}, {
            "flag": torch.logical_not(state["flag"])
        }

    def finish(p, outer, inner, x, count):
        # reshape is valid for both eager noncontiguous tensors and native
        # contiguous boundary storage, with the same logical index order.
        return {"h": inner["z"].t().reshape(-1).reshape(3, 2) + 0.03 * p["weight"].t()}

    def terminal(p, carry, x, count):
        return carry["h"].reshape(-1).square().mean()

    nested = NestedLoopBody(inner_init, inner_body, finish, count_input=1, max_steps=3)
    return capture_loop_training(
        initial,
        nested,
        terminal,
        params,
        None,
        inputs,
        torch.tensor(2),
        max_steps=3,
        state={"flag": torch.tensor(True)},
    )


def test_transposed_boundary_layout():
    rows = []
    for noncontiguous in (False, True):
        captured = transposed_fixture(noncontiguous_input=noncontiguous)
        for counts in ((0, 0), (1, 0), (1, 2), (3, 3)):
            args = (*captured.example_args[:-2], torch.tensor(counts[1]), torch.tensor(counts[0]))
            reference = captured.step_fn(*args)
            error = max(
                assert_close(captured.graph_step(*args, storage_mode=mode), reference)
                for mode in ("save_body_tape", "recompute_boundaries")
            )
            rows.append({"counts": counts, "noncontiguous_input": noncontiguous, "max_absolute_error": error})
        for name in ("finish", "finish_vjp", "finish_saved", "finish_vjp_saved"):
            region = captured.ir.body_saved.regions[name]
            for key, label in zip(region.inputs, region.metadata["input_labels"]):
                if label.startswith(("carry:", "cotangent:")):
                    value = region.values[key]
                    assert value.strides == _contiguous_strides(value.shape), (name, label, value)
        region = captured.ir.body_saved
        x_index = region.metadata["input_labels"].index("input:0")
        assert region.values[region.inputs[x_index]].strides == captured.example_args[-3].stride()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--capture-only", action="store_true")
    args = parser.parse_args()
    result = {
        "counts": test_counts_and_saved_recomputed_histories(),
        "controls": test_history_controls_and_serialization(),
        "scope": test_scope_refusals(),
        "transposed": test_transposed_boundary_layout(),
    }
    if not args.capture_only:
        result["lowering"] = test_lowered_runtime_control()
    result["status"] = "PASS"
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
