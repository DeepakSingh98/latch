"""CPU contracts for the merged C-OPS4 / P7 capture and optimizer boundaries.

Execute on an admitted Modal CPU worker only. These use small independent
PyTorch references; no source-only local check imports this module.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess
import tempfile
import traceback

import torch

from megakernel_lab.operation_compiler.capture import capture_training
from megakernel_lab.operation_compiler.ir import CaptureError
from megakernel_lab.operation_compiler.loop import BoundedLoopIR, LOW_MEMORY_TAPE_OPTIONS, _resolve_memory_policy, capture_loop_training
from megakernel_lab.operation_compiler.muon import MuonConfig
from megakernel_lab.operation_compiler.optim import AdamWConfig, SGDConfig, initialize_optimizer_state, prepare_optimizer


def test_clipped_adamw_ignores_inactive_device_gradient():
    """An inactive large/NaN gradient cannot change another parameter's clip."""
    config = AdamWConfig(lr=0.002, betas=(0.9, 0.95), eps=1e-8,
                         weight_decay=0.1, max_grad_norm=0.25)
    params = {"active": torch.tensor([0.2, -0.3]), "inactive": torch.tensor([0.7, 0.9])}
    gradient = torch.tensor([0.6, -0.8])
    for inactive in (torch.full((2,), 1e6), torch.full((2,), float("nan"))):
        prepared = prepare_optimizer(config, params)
        actual, state = prepared.update(tuple(params.values()), (gradient, inactive), prepared.state_values,
                                        participation=(torch.tensor(True), torch.tensor(False)))
        ref = [value.clone().requires_grad_() for value in params.values()]
        optimizer = torch.optim.AdamW(ref, lr=config.lr, betas=config.betas, eps=config.eps,
                                      weight_decay=config.weight_decay, foreach=False, fused=False)
        ref[0].grad = gradient.clone()
        torch.nn.utils.clip_grad_norm_(ref, config.max_grad_norm, foreach=False)
        optimizer.step()
        torch.testing.assert_close(actual[0], ref[0], atol=2e-7, rtol=1e-6)
        assert torch.equal(actual[1], ref[1])
        named = dict(zip(prepared.state_names, state))
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(named[f"{key}:active"], optimizer.state[ref[0]][key],
                                       atol=2e-9, rtol=1e-6)
            assert torch.count_nonzero(named[f"{key}:inactive"]) == 0
        assert int(named["step:active"]) == 1 and int(named["step:inactive"]) == 0


def test_runtime_adamw_rate_changes_values_without_recapture():
    params = {"w": torch.tensor([0.2, -0.3])}
    x = torch.tensor([0.4, 0.7])
    config = AdamWConfig(lr=0.001, runtime_lr=True, weight_decay=0.1, max_grad_norm=0.25)
    def loss(p, x):
        return (p["w"] * x).square().sum()
    captured = capture_training(loss, params, None, (x,), config)
    index = captured.ir.metadata["optimizer_lr_input"]["index"]
    assert index == len(captured.example_args) - 1
    graph_identity = id(captured.graph_module)
    for rate in (0.0, 0.003):
        args = list(captured.example_args)
        args[index] = torch.tensor(rate, dtype=torch.float32)
        actual = captured.unpack(captured.graph_module(*args))
        p = params["w"].clone().requires_grad_()
        optimizer = torch.optim.AdamW([p], lr=rate, betas=config.betas, eps=config.eps,
                                      weight_decay=config.weight_decay, foreach=False, fused=False)
        loss({"w": p}, x).backward()
        torch.nn.utils.clip_grad_norm_([p], config.max_grad_norm, foreach=False)
        optimizer.step()
        torch.testing.assert_close(actual["params"]["w"], p, atol=2e-7, rtol=1e-6)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual["optimizer_state"][f"{key}:w"], optimizer.state[p][key],
                                       atol=2e-9, rtol=1e-6)
        assert int(actual["optimizer_state"]["step:w"]) == 1
        if rate == 0.0:
            assert torch.equal(actual["params"]["w"], params["w"])
    assert id(captured.graph_module) == graph_identity


def test_clamp_helpers_match_torch_including_reversed_bounds():
    """Compile the shared scalar C++ helper on CPU; compare its values to torch.

    This catches semantic differences in the actual emitted helper, including
    reversed bounds and NaN propagation, without needing a GPU or a model build.
    """
    from megakernel_lab.operation_compiler.extra_ops import EXTRA_HELPERS
    source = ("#include <cstdint>\n#define __device__\n#define __forceinline__ inline\n"
              "#define __trap __builtin_trap\n" + EXTRA_HELPERS +
              '\nextern "C" float clamp_test(float x,float lo,float hi) { return latch_clamp(x,lo,hi); }\n')
    with tempfile.TemporaryDirectory(prefix="cops4-clamp-") as folder:
        path = Path(folder)
        (path / "helper.cpp").write_text(source)
        subprocess.run(["c++", "-std=c++17", "-shared", "-fPIC", "-O2", str(path / "helper.cpp"),
                        "-o", str(path / "helper.so")], check=True, capture_output=True, timeout=30)
        helper = ctypes.CDLL(str(path / "helper.so")).clamp_test
        helper.argtypes = [ctypes.c_float] * 3
        helper.restype = ctypes.c_float
        for lo, hi in ((-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)):
            for x in (-2.0, 0.0, 2.0, float("nan")):
                expected = float(torch.tensor(x).clamp(lo, hi))
                actual = helper(x, lo, hi)
                assert math.isnan(actual) if math.isnan(expected) else actual == expected


def test_direct_fp16_sdpa_capture_is_explicitly_refused():
    import torch.nn.functional as F
    params = {"w": torch.ones(8, 8) * 0.01}
    x = torch.ones(1, 4, 8, dtype=torch.float16)
    def loss(p, x):
        q = x @ p["w"].to(torch.float16)
        return F.scaled_dot_product_attention(q, q, q).float().square().mean()
    try:
        capture_training(loss, params, None, (x,), SGDConfig())
    except CaptureError as exc:
        assert exc.code == "SDPA" and "cast FP16 inputs" in str(exc)
    else:
        raise AssertionError("direct FP16 SDPA capture silently changed its intermediate precision")
    # BF16 is now deliberately admitted by pinned attention donor39a0e20f2.
    # test_attention_recognition independently checks its FP32 math composition,
    # single BF16 output rounding, VJP dtypes and normal capture/body recognition.


def _sequence_fixture():
    generator = torch.Generator().manual_seed(404)
    params = {"w": torch.randn(4, 4, generator=generator) * 0.1}
    inputs = (torch.randn(3, 4, 4, generator=generator),)
    def sequence(p, x):
        return {"features": torch.tanh(x @ p["w"])}
    def init(p, x):
        return {"h": p["w"].clone(), "loss": p["w"].sum() * 0.0}
    def body(p, carry, state, xs, x):
        h = torch.tanh(carry["h"] @ p["w"] + xs["features"])
        return {"h": h, "loss": carry["loss"] + h.square().mean()}, state
    def terminal(p, carry, x):
        return carry["loss"]
    return params, inputs, sequence, init, body, terminal


def test_sequence_storage_remat_merge_keeps_full_history_and_zero_tail():
    params, inputs, sequence, init, body, terminal = _sequence_fixture()
    for init_storage in ("recompute", "save_tape"):
        cap = capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(3), SGDConfig(),
                                    max_steps=3, sequence_fn=sequence, init_storage=init_storage,
                                    tape_policy="storage_remat", tape_options={"cse": True},
                                    loop_attention="long")
        restored = BoundedLoopIR.from_dict(cap.ir.to_dict())
        assert restored.metadata["sequence_specs"] == cap.ir.metadata["sequence_specs"]
        assert not cap.ir.metadata["cast_reuse"]["enabled"] and cap.ir.invariant_casts is None
        assert (restored.init_saved is not None) == (init_storage == "save_tape")
        for count in (0, 1, 3):
            actual = cap.unpack(cap.graph_step(*cap.example_args[:-1], torch.tensor(count)))
            p = {key: value.clone().requires_grad_() for key, value in params.items()}
            xs, carry = sequence(p, *inputs), init(p, *inputs)
            for step in range(count):
                carry, _ = body(p, carry, {}, {key: value[step] for key, value in xs.items()}, *inputs)
            loss = terminal(p, carry, *inputs)
            grad, = torch.autograd.grad(loss, tuple(p.values()))
            torch.testing.assert_close(actual["loss"], loss, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(actual["grads"]["w"], grad, atol=1e-6, rtol=1e-5)
            if count == 0:
                assert torch.count_nonzero(actual["grads"]["w"]) == 0


def test_muon_schedule_scope_and_index_admission():
    params, inputs, sequence, init, body, terminal = _sequence_fixture()
    config = MuonConfig(("w",), schedule_input=4)
    state = initialize_optimizer_state(params, config)
    assert tuple(state) == ("momentum:w",)
    for operation in (
        lambda: capture_training(lambda p, x: p["w"].sum(), params, None, inputs, config),
        lambda: capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(3), config,
                                      max_steps=3, sequence_fn=sequence),
    ):
        try:
            operation()
        except CaptureError as exc:
            assert exc.code == "OPTIMIZER_ABI"
        else:
            raise AssertionError("invalid scheduled optimizer ABI was accepted")


def test_public_memory_policy_is_exact_and_rejects_conflicts():
    defaults = dict(storage_mode="save_body_tape", tape_policy="auto", tape_options=None,
                    cast_reuse=None, conditional=False, sequenced=False, init_storage="recompute")
    policy, options, casts = _resolve_memory_policy("low_memory", **defaults)
    assert policy == "storage_remat" and options == LOW_MEMORY_TAPE_OPTIONS and casts is True
    assert _resolve_memory_policy("default", **defaults) == ("auto", None, True)
    assert _resolve_memory_policy("default", **{**defaults,"cast_reuse":False}) == ("auto", None, False)
    for exclusion in ({"conditional": True}, {"sequenced": True}, {"init_storage": "save_tape"},
                      {"storage_mode": "save_boundaries"}, {"storage_mode": "recompute_boundaries"}):
        assert _resolve_memory_policy("default", **{**defaults, **exclusion}) == ("auto", None, False)
        # Explicit diagnostic requests retain their existing downstream checks.
        assert _resolve_memory_policy("default", **{**defaults, **exclusion,"cast_reuse":True}) == ("auto", None, True)
    for invalid in (0, 1, "auto"):
        try:
            _resolve_memory_policy("default", **{**defaults,"cast_reuse":invalid})
        except CaptureError as exc:
            assert exc.code == "LOOP_MEMORY"
        else:
            raise AssertionError("non-boolean explicit cast override was admitted")
    for override in ({"tape_policy": "stage3"}, {"cast_reuse": False}, {"storage_mode": "save_boundaries"},
                     {"tape_options": {"flops_per_byte": 65}}, {"tape_options": {"unknown": 1}},
                     {"conditional": True}, {"sequenced": True}, {"init_storage": "save_tape"}):
        try:
            _resolve_memory_policy("low_memory", **{**defaults, **override})
        except CaptureError as exc:
            assert exc.code == "LOOP_MEMORY"
        else:
            raise AssertionError(f"conflicting memory policy accepted: {override}")
    from .test_loop import CONFIG, fixture, eager_step
    params, inputs, state, init, body, terminal = fixture(random=True)
    captured = capture_loop_training(init, body, terminal, params, None, inputs, torch.tensor(3), CONFIG,
                                     max_steps=3, state=state, memory_policy="low_memory")
    selection = captured.ir.metadata["memory_selection"]
    assert selection["requested"] == "low_memory" and selection["tape_options"] == LOW_MEMORY_TAPE_OPTIONS
    restored = BoundedLoopIR.from_dict(captured.ir.to_dict())
    assert restored.metadata["memory_selection"] == selection
    for count in (0, 1, 3):
        flat = (*captured.example_args[:-1], torch.tensor(count))
        actual = captured.unpack(captured.graph_step(*flat))
        expected = captured.unpack(eager_step(captured, (init, body, terminal), flat))
        torch.testing.assert_close(actual["loss"], expected["loss"], atol=1e-7, rtol=1e-6)
        for key, value in actual["grads"].items():
            torch.testing.assert_close(value, expected["grads"][key], atol=1e-6, rtol=1e-5)
        for key, value in actual["state"].items():
            assert torch.equal(value, expected["state"][key])


def test_automatic_invariant_casts_match_explicit_pass_and_autograd():
    """The new default selects the existing pass, preserving all VJPs and commits."""
    from megakernel_lab.operation_compiler.lower import lower
    from .test_loop import CONFIG, eager_step, _assert_outputs
    generator = torch.Generator().manual_seed(17039)
    params = {"w":torch.randn(8,8,generator=generator)*.1}
    inputs = (torch.randn(4,8,generator=generator)*.2,)
    state = {"iterations":torch.zeros((),dtype=torch.int64)}
    def product(x,w):
        return (x.bfloat16() @ w.bfloat16()).float()
    def initial(p,x):
        return {"hidden":product(x,p["w"])}
    def body(p,c,s,x):
        return {"hidden":.9*c["hidden"]+.1*torch.tanh(product(c["hidden"],p["w"]))}, {"iterations":s["iterations"]+1}
    def terminal(p,c,x):
        return product(c["hidden"],p["w"]).square().mean()
    def capture(**options):
        return capture_loop_training(initial,body,terminal,params,None,inputs,torch.tensor(1),CONFIG,
                                      max_steps=3,state=state,**options)
    automatic, explicit, disabled = capture(),capture(cast_reuse=True),capture(cast_reuse=False)
    selected = automatic.ir.metadata["cast_reuse"]
    assert selected["requested"] is None and selected["enabled"] and selected["selected"]
    assert selected["selection_mode"] == "automatic" and selected["reason"]
    assert automatic.ir.invariant_casts is not None and disabled.ir.invariant_casts is None
    assert automatic.ir.metadata["invariant_casts"] == explicit.ir.metadata["invariant_casts"]
    assert all(row["source"].startswith(("parameter:","input:")) for row in automatic.ir.metadata["invariant_casts"])
    assert lower(automatic.ir).cache_key == lower(explicit.ir).cache_key
    restored = BoundedLoopIR.from_dict(automatic.ir.to_dict())
    assert restored.metadata["cast_reuse"] == selected
    for count in (0,1,3):
        flat = (*automatic.example_args[:-1],torch.tensor(count))
        expected = eager_step(automatic,(initial,body,terminal),flat)
        actual = automatic.graph_step(*flat)
        _assert_outputs(actual,expected)
        assert all(torch.equal(a,b) for a,b in zip(actual,explicit.graph_step(*flat)))
        _assert_outputs(actual,disabled.graph_step(*flat))
    # Unsupported automatic ABIs keep their previous behavior and explicit
    # invalid sequence/saved-init requests still fail instead of being ignored.
    for options in ({"init_storage":"save_tape"}, {"storage_mode":"save_boundaries"},
                    {"storage_mode":"recompute_boundaries"}):
        old = capture(**options)
        assert old.ir.invariant_casts is None and not old.ir.metadata["cast_reuse"]["enabled"]
    try:
        capture(init_storage="save_tape",cast_reuse=True)
    except CaptureError as exc:
        assert exc.code == "LOOP_INIT_STORAGE"
    else:
        raise AssertionError("saved-init/cast-reuse conflict was admitted")
    from megakernel_lab.operation_compiler.loop import ConditionalBody
    def branch(p,c,s,x):
        carry,new_state = body(p,c,s,x)
        return carry,{**new_state,"route":s["route"]}
    conditional = capture_loop_training(initial,ConditionalBody(branch,branch,predicate_state="route"),
        terminal,params,None,inputs,torch.tensor(1),CONFIG,max_steps=3,
        state={**state,"route":torch.tensor(True)})
    assert conditional.ir.invariant_casts is None
    assert not conditional.ir.metadata["cast_reuse"]["enabled"]
    assert conditional.ir.metadata["cast_reuse"]["requested"] is None
    seq_params,seq_inputs,sequence,seq_init,seq_body,seq_loss = _sequence_fixture()
    try:
        capture_loop_training(seq_init,seq_body,seq_loss,seq_params,None,seq_inputs,torch.tensor(1),CONFIG,
                              max_steps=3,sequence_fn=sequence,cast_reuse=True)
    except CaptureError as exc:
        assert exc.code == "LOOP_SEQUENCE"
    else:
        raise AssertionError("sequence/cast-reuse conflict was admitted")


def test_auto_long_attention_admission_and_policy_metadata():
    from megakernel_lab.fast_examples.recovered.transformer import TransformerConfig
    from .test_loop_attention import _capture
    # Multiple heads retain the materializing head-layout copy required by the
    # pinned attention boundary. A singleton-head alias is separately refused.
    model = TransformerConfig(batch=2, sequence=256, width=128, heads=2, layers=1,
                              expansion=1, matrix_dtype="bfloat16")
    captured, details = _capture(model=model)
    policy = captured.ir.metadata["attention_tape"]
    assert policy["selection"] == {"requested": "auto", "admitted": True, "short_groups": 0,
                                    "long_groups": 1, "fallback": "ordinary captured primitives for inadmissible groups"}, policy
    assert policy["groups"][0]["kernel"] == "long"
    assert policy["numerical_policy"] == {"attention": "exact3", "splits": 3, "comparison": "tolerance-gated",
                                           "bitwise_fp32": False, "long_resident_max_sequence": 512}
    off, _ = _capture(model=model, loop_attention="off")
    assert off.ir.metadata["attention_tape"]["groups"] == []
    from dataclasses import replace
    refused, _ = _capture(model=replace(model, matrix_dtype="float32"))
    assert refused.ir.metadata["attention_tape"]["groups"] == []
    singleton = replace(model, batch=1, width=64, heads=1)
    fallback, fallback_details = _capture(model=singleton)
    refused_policy = fallback.ir.metadata["attention_tape"]
    assert refused_policy["groups"] == [] and not refused_policy["selection"]["admitted"], refused_policy
    assert any("output chain needs exactly one materializing copy" in row["reason"]
               for row in refused_policy["refusals"]), refused_policy
    explicit_off, _ = _capture(model=singleton, loop_attention="off")
    from megakernel_lab.operation_compiler.lower import lower
    assert lower(fallback.ir).source == lower(explicit_off.ir).source
    from .fixtures.stage3_training import compare_step, reference_step, _loss_for
    def passed(value):
        return (value.get("passed", True) and all(passed(item) for item in value.values() if isinstance(item, dict))
                if isinstance(value, dict) else True)
    for cap, info in ((captured, details), (fallback, fallback_details)):
        cfg, params, momentum, state, data, objective = info
        for count in (0, 1):
            flat = (*params.values(), *momentum.values(), *state.values(), *data, torch.tensor(count))
            actual = cap.unpack(cap.graph_step(*flat))
            expected = reference_step(_loss_for(cfg, objective, count), params, momentum, data, state=state)
            checks = compare_step(actual, expected, params, arithmetic="bfloat16")
            assert passed(checks), {"shape": (cfg.batch, cfg.sequence, cfg.width, cfg.heads),
                                    "count": count, "checks": checks}
            assert torch.equal(actual["state"]["rng"], expected["state"]["rng"])
            if cap is fallback:
                off_values = explicit_off.graph_step(*flat)
                auto_values = cap.graph_step(*flat)
                assert all(torch.equal(a, b) for a, b in zip(auto_values, off_values))


def test_adamw_device_warmup_matches_independent_scheduler():
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.optim import AdamWConfig, initialize_optimizer_state, prepare_optimizer
    from megakernel_lab.operation_compiler.optimizer_reference import compare_optimizer_step
    from megakernel_lab.operation_compiler.ir import CaptureError
    config = AdamWConfig(lr=.004, weight_decay=.2, max_grad_norm=.25, warmup_steps=4,
                         param_options={"bias": {"weight_decay": 0.0}})
    params = {"weight": torch.tensor([[.1, -.2], [.3, .4]]), "bias": torch.tensor([.2, -.15])}
    data = (torch.tensor([[2., -1.], [3., 4.], [-2., 3.]]),)
    def loss(p, x):
        return (x @ p["weight"] + p["bias"]).square().mean()
    state = initialize_optimizer_state(params, config)
    captured = capture_training(loss, params, None, data, config, optimizer_state=state)
    reference = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    optimizer = torch.optim.AdamW([{"params": [reference["weight"]], "weight_decay": .2},
                                   {"params": [reference["bias"]], "weight_decay": 0.}],
                                  lr=.004, betas=config.betas, eps=config.eps, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: min(1., (k + 1) / 4))
    def passed(value):
        if isinstance(value, dict):
            return value.get("passed", True) and all(passed(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(passed(v) for v in value)
        return True
    for step in range(8):
        got = captured.unpack(captured.graph_module(*params.values(), *state.values(), *data))
        optimizer.zero_grad(set_to_none=True)
        expected_loss = loss(reference, *data)
        expected_loss.backward()
        raw = {k: v.grad.detach().clone() for k, v in reference.items()}
        torch.nn.utils.clip_grad_norm_(list(reference.values()), .25, foreach=False)
        optimizer.step(); scheduler.step()
        expected_state = {}
        for name, value in reference.items():
            current = optimizer.state[value]
            for kind in ("exp_avg", "exp_avg_sq"):
                expected_state[kind + ":" + name] = current[kind].clone()
            expected_state["step:" + name] = current["step"].to(torch.int64)
        expected = {"loss": expected_loss.detach(), "grads": raw,
                    "params": {k: v.detach().clone() for k, v in reference.items()},
                    "optimizer_state": expected_state, "state": {}}
        checks = compare_optimizer_step(got, expected, params)
        assert passed(checks), {"step": step, "checks": checks}
        params, state = expected["params"], expected_state
    assert captured.ir.metadata["optimizer_lr_schedule"]["warmup_steps"] == 4
    prepared = prepare_optimizer(config, params, optimizer_state=state)
    newp, newo = prepared.update(tuple(params.values()), (torch.zeros_like(params["weight"]), None),
                                 tuple(state.values()))
    assert newo[2].item() == state["step:weight"].item() + 1
    assert newo[5].item() == state["step:bias"].item()
    assert torch.equal(newp[1], params["bias"])
    for invalid in (True, -1, 1.5, 2**24):
        try:
            AdamWConfig(warmup_steps=invalid).validate()
        except CaptureError:
            pass
        else:
            raise AssertionError("invalid warmup was admitted")
    try:
        AdamWConfig(runtime_lr=True, warmup_steps=4).validate()
    except CaptureError:
        pass
    else:
        raise AssertionError("two distinct learning-rate providers were admitted")


def test_sampled_boundary_memory_reports_two_buffers():
    from megakernel_lab.operation_compiler.lower import lower
    from .test_sampling import _sampled
    captured, _ = _sampled()
    program = lower(captured.ir)
    manifest = program.manifest
    changed = captured.ir.metadata["body_state_changes"]
    nc = len(captured.ir.metadata["carry_names"])
    by = {"float32": 4, "bfloat16": 2, "int64": 8, "bool": 1}
    init = manifest["regions"]["init"]
    raw = sum(math.prod(init["layouts"][name]["shape"]) * by[init["layouts"][name]["dtype"]]
              for name in init["outputs"][:nc])
    raw += sum(math.prod(captured.ir.values[name].shape) * by[captured.ir.values[name].dtype]
               for name in captured.ir.inputs
               if captured.ir.values[name].role == "state" and captured.ir.values[name].parameter in changed)
    assert manifest["boundary_bytes"] == 2 * raw


def run():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results[name] = {"passed": True}
            except Exception:
                results[name] = {"passed": False, "traceback": traceback.format_exc()}
    return {"passed": all(item["passed"] for item in results.values()), "tests": results}
