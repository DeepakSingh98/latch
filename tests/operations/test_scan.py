"""CPU contracts for the selective-scan primitive, its capture and lowering (C-OPS2-C).

Run on Modal CPU with torch 2.9.1 (single thread). Independent references: the NumPy FP64 equations in
ops.scan.reference_float64 and the hand-derived S99K/S108F scan reference vendored at cops2_c/s108f/reference.py (P-OPS4-G copy: pops4_g/s108f).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from megakernel_lab.operation_compiler import scan as S

torch.set_num_threads(1)
LANE = Path(__file__).resolve().parents[1]


def _inputs(Bt=2, T=21, E=6, N=4, dtype=torch.float32, seed=0, reset=False):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g)
    u, delta, z = r(Bt, T, E), r(Bt, T, E) - 1.0, r(Bt, T, E)
    A = -torch.exp(0.5 * r(E, N))
    B, C = r(Bt, T, N), r(Bt, T, N)
    D, bias = r(E), 0.5 * r(E)
    rst = (torch.rand(Bt, T, generator=g) < 0.15) if reset else None
    if rst is not None:
        rst[:, 0] = True
    cast = lambda x: x.to(dtype)
    return cast(u), cast(delta), A, cast(B), cast(C), D, cast(z), bias, rst


def _rel(a, b):
    a = np.asarray(a.detach().double() if isinstance(a, torch.Tensor) else a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("softplus", [True, False])
def test_eager_definition_matches_float64(reset, softplus):
    args = list(_inputs(reset=reset))
    if not softplus:
        args[1] = 0.3 * args[1].abs()    # a raw (non-softplus) step must stay positive for a decaying state
    args = tuple(args)
    out, ckpt = S.selective_scan_fwd(*args, softplus)
    dout = torch.randn(out.shape, generator=torch.Generator().manual_seed(9))
    grads = S.selective_scan_bwd(*args, dout, ckpt, softplus)
    ref_out, ref = S.reference_float64(*args, delta_softplus=softplus, dout=dout)
    assert _rel(out, ref_out) < 2e-6
    for name, value in zip(S.GRAD_NAMES, grads):
        assert value.dtype == (torch.float32 if name in ("dA", "dD", "ddelta_bias") else args[0].dtype)
        assert _rel(value, ref[name]) < 2e-5, name


def test_checkpoint_is_state_entering_every_eighth_step():
    args = _inputs(T=27, reset=True)
    _, ckpt = S.selective_scan_fwd(*args, True)
    _, _, states = S._forward_states(args[0], args[1], args[2], args[3], args[4], args[7], args[8], True)
    assert tuple(ckpt.shape) == (2, 4, 4, 6)
    assert torch.equal(ckpt[:, 0], torch.zeros_like(ckpt[:, 0]))
    for j in range(1, 4):
        assert torch.equal(ckpt[:, j], states[8 * j - 1].transpose(1, 2))


def _s108f_reference():
    spec = importlib.util.spec_from_file_location("s108f_reference", Path(__file__).resolve().parent / "fixtures" / "scan_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("reset", [False, True])
def test_agrees_with_independent_s99k_scan_reference(reset):
    """The S108F hand-derived scan (a_log parameterization, packed z) and this operation agree."""
    REF = _s108f_reference()
    Bt, T, E, N = 2, 19, 5, 4
    g = np.random.default_rng(3)
    z = g.normal(size=(Bt, T, 3 * E + 2 * N))
    a_log = g.normal(size=(E, N)) * 0.3
    dtb, dsk = g.normal(size=E) * 0.5, g.normal(size=E)
    seg = np.zeros((Bt, T), np.int32)
    seg[:, 0] = 1
    if reset:
        seg[0, 7] = seg[1, 11] = 1
    keep = 1.0 - seg
    o, sc = REF.scan_forward(z, E, N, a_log, dtb, dsk, keep)
    sc["dsk"] = dsk
    do = g.normal(size=o.shape)
    dz_ref, shared = REF.scan_backward(do, sc, E, N, keep)
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    v, q, c = t(z[..., :E]), t(z[..., E:2 * E]), t(z[..., 2 * E:3 * E])
    k, r = t(z[..., 3 * E:3 * E + N]), t(z[..., 3 * E + N:])
    A = -torch.exp(t(a_log))
    rst = torch.tensor(seg.astype(bool)) if reset else None
    out, ckpt = S.selective_scan_fwd(v, c, A, k, r, t(dsk), q, t(dtb), rst, True)
    assert _rel(out, o) < 1e-5
    du, ddl, dA, dB, dC, dD, dzg, dbias = S.selective_scan_bwd(v, c, A, k, r, t(dsk), q, t(dtb), rst, t(do), ckpt, True)
    mine = torch.cat([du, dzg, ddl, dB, dC], -1)
    assert _rel(mine, dz_ref) < 1e-5
    assert _rel(dA * A, shared["a_log"]) < 1e-5          # d a_log = dA * dA/da_log = dA * A
    assert _rel(dbias, shared["dt_bias"]) < 1e-5
    assert _rel(dD, shared["d_skip"]) < 1e-5


def test_autograd_uses_backward_operation():
    args = list(_inputs(reset=True))
    leaves = [x.clone().requires_grad_(True) if i != 8 else x for i, x in enumerate(args)]
    out = S.selective_scan(*leaves[:8], reset=leaves[8])
    w = torch.randn(out.shape, generator=torch.Generator().manual_seed(4))
    (out * w).sum().backward()
    _, ref = S.reference_float64(*args, dout=w)
    for leaf, name in zip(leaves[:8], ("du", "ddelta", "dA", "dB", "dC", "dD", "dz", "ddelta_bias")):
        assert _rel(leaf.grad, ref[name]) < 2e-5, name


def test_bf16_rounds_outputs_once():
    args = _inputs(dtype=torch.bfloat16)
    out, _ = S.selective_scan_fwd(*args, True)
    exact, _ = S.reference_float64(*args)
    assert out.dtype == torch.bfloat16
    assert _rel(out.float(), exact) < 8e-3


# ---------------------------------------------------------------------------------------------- capture/lowering
def _tiny(matrix_dtype="bfloat16", E=64, N=16, T=32, batch=2, F=128):
    from .fixtures import ssm_models as M
    cfg = M.SSMConfig(d_model=64, blocks=(("ssm", E, N), ("ffn", F)), batch=batch, seq_len=T,
                      matrix_dtype=matrix_dtype, seed=5)
    return M, cfg


def _capture(matrix_dtype="bfloat16", reset=False, **kw):
    from megakernel_lab.operation_compiler.api import capture_module
    M, cfg = _tiny(matrix_dtype, **kw)
    model = M.build(cfg)
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(0, 256, (cfg.batch, cfg.seq_len), generator=g)
    tgt = torch.randint(0, 256, (cfg.batch, cfg.seq_len), generator=g)
    inputs = (tok,)
    if reset:
        rst = torch.zeros(cfg.batch, cfg.seq_len, dtype=torch.bool)
        rst[:, 0] = True
        rst[0, 9] = True
        inputs = (tok, rst)
    captured = capture_module(model, M.loss_fn, inputs, (tgt,), optimizer=M.optimizer_config(cfg))
    return M, cfg, model, captured, inputs, tgt


def test_initialization_and_declaration_match_s108f_family():
    spec = importlib.util.spec_from_file_location("s108f_family", Path(__file__).resolve().parent / "fixtures" / "scan_family.py")
    family = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family)
    from .fixtures import ssm_models as M
    for case in ("tiny16", "tiny4", "s5", "m30w"):
        cfg = M.config(case)
        decl = family.admit(M.declaration(cfg))
        ref = family.initialize(decl)
        mine = M.initialize(cfg)
        assert [p["id"] for p in family.inventory(decl)] == [name for name, _, _ in M.inventory(cfg)]
        assert all(np.array_equal(mine[k], ref[k]) and mine[k].dtype == ref[k].dtype for k in ref), case
        assert decl["optimizer"] == dict(kind="adamw", lr=2e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1)
        assert sum(int(np.prod(s)) for _, _, s in M.inventory(cfg)) == family.n_params(decl)


def test_capture_keeps_scan_operations_and_slice_views():
    M, cfg, model, captured, inputs, tgt = _capture()
    ir = captured.ir
    scans = [op for op in ir.operations if op.target in S.TARGETS]
    assert [op.target for op in scans] == [S.FWD_TARGET, S.BWD_TARGET]
    assert [len(op.outputs) for op in scans] == [2, 8]
    for op in scans:
        assert op.writes == op.outputs and all(ir.values[name].producer == op.name for name in op.outputs)
    slices = [op for op in ir.operations if op.target == "aten.slice.Tensor"]
    assert len(slices) >= 5 and all(ir.values[op.outputs[0]].alias_of for op in slices)
    assert sum(op.target == "aten.slice_backward.default" for op in ir.operations) == 5
    assert ir.metadata["optimizer"]["max_grad_norm"] == 1.0
    assert "optimizer_lr" in ir.metadata["module_api"]["input_indices"]
    # CPU BF16 GEMMs of eager torch.func execution and the traced graph may sum in different orders,
    # so the replay check is bitwise for FP32 and BF16-close for the BF16 recipe.
    replay = captured.graph_module(*captured.example_args)
    direct = captured.step_fn(*captured.example_args)
    assert abs(float(replay[0]) - float(direct[0])) < 1e-4
    M, cfg, model, captured, inputs, tgt = _capture("float32")
    replay = captured.graph_module(*captured.example_args)
    direct = captured.step_fn(*captured.example_args)
    assert all(torch.equal(a, b) for a, b in zip(replay, direct))


def _torch_reference_update(M, cfg, inputs, tgt, lr):
    """Ordinary autograd + torch clip_grad_norm_ + torch.optim.AdamW (independent conventional path)."""
    model = M.build(cfg)
    params = dict(model.named_parameters())
    decay = [p for name, role, _ in M.inventory(cfg) for p in [params[name]] if role in M.MATRIX_ROLES]
    plain = [p for name, role, _ in M.inventory(cfg) for p in [params[name]] if role not in M.MATRIX_ROLES]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg.weight_decay), dict(params=plain, weight_decay=0.0)],
                            lr=lr, betas=cfg.betas, eps=cfg.eps, foreach=False)
    loss = M.loss_fn(model(*inputs), tgt)
    loss.backward()
    grads = {name: p.grad.detach().clone() for name, p in params.items()}
    torch.nn.utils.clip_grad_norm_(list(params.values()), cfg.max_grad_norm)
    opt.step()
    return float(loss), grads, {name: p.detach().clone() for name, p in params.items()}


@pytest.mark.parametrize("matrix_dtype", ["float32", "bfloat16"])
def test_captured_update_matches_autograd_clip_adamw(matrix_dtype):
    M, cfg, model, captured, inputs, tgt = _capture(matrix_dtype)
    lr = 3e-3
    args = list(captured.example_args)
    args[captured.ir.metadata["optimizer_lr_input"]["index"]] = torch.tensor(lr)
    outputs = captured.step_fn(*args)
    result = captured.unpack(outputs)
    loss, grads, params = _torch_reference_update(M, cfg, inputs, tgt, lr)
    fp32 = matrix_dtype == "float32"
    assert abs(float(result["loss"]) - loss) < (1e-6 if fp32 else 1e-3)
    for name in grads:
        assert _rel(result["grads"][name], grads[name].numpy()) < (1e-5 if fp32 else 2e-2), name
        if fp32:   # BF16 rounding flips change near-zero AdamW first-step signs; checked natively against FP64
            before = dict(model.named_parameters())[name].detach()
            assert _rel(result["params"][name] - before, (params[name] - before).numpy()) < 1e-4, name


def test_clip_coefficient_is_applied_and_nan_propagates():
    from megakernel_lab.operation_compiler.optim import AdamWConfig, prepare_optimizer
    p = {"w": torch.ones(3)}
    opt = prepare_optimizer(AdamWConfig(lr=0.1, max_grad_norm=0.5, weight_decay=0.0), p)
    g = torch.tensor([3.0, 4.0, 0.0])
    new_p, state = opt.update((p["w"],), (g,), opt.state_values)
    m = state[0]
    assert torch.allclose(m, 0.1 * g * (0.5 / (5.0 + 1e-6)))
    _, state = opt.update((p["w"],), (torch.tensor([float("nan"), 0.0, 0.0]),), opt.state_values)
    assert torch.isnan(state[0]).all()


def test_runtime_lr_is_an_explicit_input():
    from megakernel_lab.operation_compiler.ir import CaptureError
    from megakernel_lab.operation_compiler.optim import AdamWConfig
    with pytest.raises(CaptureError):
        AdamWConfig(runtime_lr=True, param_options={"w": {"lr": 0.1}}).validate()
    M, cfg, model, captured, inputs, tgt = _capture("float32")
    index = captured.ir.metadata["optimizer_lr_input"]["index"]
    name = captured.ir.inputs[index]
    assert captured.ir.values[name].role == "input" and captured.ir.values[name].shape == ()
    args = list(captured.example_args)
    zero = list(args)
    zero[index] = torch.tensor(0.0)
    out = captured.unpack(captured.step_fn(*zero))
    for pname, value in dict(model.named_parameters()).items():
        assert torch.equal(out["params"][pname], value.detach()), pname


def _lowered(captured, **kw):
    from megakernel_lab.operation_compiler.lower import lower
    # These checks exercise the retained P6-E schedule and its explicit knobs.
    return lower(captured.ir, execution="legacy", **kw)


def test_lowering_selects_p1a_body_for_eligible_bf16_layouts():
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16")
    program = _lowered(captured)
    decisions = program.manifest["selective_scans"]
    assert [d["selected"] for d in decisions] == ["p1a", "p1a"]
    assert "latch_scan_p1a::backward<16, true>" in program.source
    assert "scan_p1a.cuh" in program.manifest["header_hashes"]
    phases = [p for p in program.manifest["phases"] if p["kind"].startswith("selective_scan")]
    # One exclusive heavy item per phase (independent elementwise loops may follow it in the same phase).
    assert len(phases) == 2 and all(sum(i["kind"] != "elementwise" for i in p["items"]) == 1 for p in phases)
    assert max(p.get("shared_bytes", 0) for p in phases) == program.manifest["shared_bytes"]
    assert program.manifest["compiler_selection"]["schedule"] == "fused_dependency"


def test_reference_policy_and_ineligible_layouts_use_generic_body():
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16")
    program = _lowered(captured, policy="reference")
    assert {d["selected"] for d in program.manifest["selective_scans"]} == {"generic"}
    M, cfg, model, captured, inputs, tgt = _capture("float32")
    decisions = _lowered(captured).manifest["selective_scans"]
    assert {d["selected"] for d in decisions} == {"generic"}
    assert any("not BF16" in reason for reason in decisions[0]["p1a_refusals"])
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16", N=4, E=32)
    decisions = _lowered(captured).manifest["selective_scans"]
    assert {d["selected"] for d in decisions} == {"generic"}
    assert any("N=4" in reason for reason in decisions[0]["p1a_refusals"])


def test_reset_input_lowers_and_forced_engine_refusal_is_explicit():
    from megakernel_lab.operation_compiler.lower import LoweringError, lower
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16", reset=True)
    program = _lowered(captured)
    assert {d["selected"] for d in program.manifest["selective_scans"]} == {"p1a"}
    assert "sa.rst = reinterpret_cast<const unsigned char*>" in program.source
    M, cfg, model, captured, inputs, tgt = _capture("float32")
    with pytest.raises(LoweringError, match="P1-A scan body refused"):
        lower(captured.ir, scan_engine="p1a")


def test_slice_backward_lowering_places_gradient_segments():
    from megakernel_lab.operation_compiler.lower import slice_backward_index
    condition, position = slice_backward_index([None, [2, 3, 10], 2, 4, 7, 1], (2, 3, 10), (2, 3, 3), "i")
    for i in range(60):
        b, t, c = i // 30, (i // 10) % 3, i % 10
        ok = eval(condition.replace("LL", "").replace("&&", " and ").replace("/", "//"))
        assert ok == (4 <= c < 7)
        if ok:
            assert eval(position.replace("LL", "").replace("/", "//")) == (b * 3 + t) * 3 + (c - 4)


def test_long_few_result_reductions_use_split_two_pass_rule():
    """Automatic P6-E reductions and the explicit primitive split rule remain distinct."""
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16", T=8192, batch=2)
    automatic = _lowered(captured)
    assert automatic.manifest["performance_schedule"]["selected"] == "p6e"
    auto_engines = {p.get("engine") for p in automatic.manifest["phases"] if p["kind"] == "reduction"}
    assert {"pops5e_row", "pops5e_fused"} <= auto_engines
    # Exercise the original split rule by explicitly disabling the admitted
    # replacement reductions. This does not alter the public automatic policy.
    program = _lowered(captured, schedule_options={"pops3c": False, "pops4a": False,
                        "pops5e_full": False, "pops5e_store": False, "pops5e_stream": False})
    engines = [p.get("engine") for p in program.manifest["phases"] if p["kind"] == "reduction"]
    assert "split_two_pass_column" in engines and "split_two_pass_row" in engines
    # The default schedule may substitute its release/acquire counter barrier for the internal grid barrier.
    assert ("grid.sync();\nfor (long long o" in program.source
            or "latch_loop_barrier(latch_bar);\nfor (long long o" in program.source)
    reference = _lowered(captured, policy="reference")
    assert not any(str(p.get("engine", "")).startswith("split_") for p in reference.manifest["phases"])


def test_scan_options_defaults_and_validation():
    """P-OPS4-G: contraction is the P1-A default; options are validated; the diagnostic C-OPS2-C body takes none."""
    from megakernel_lab.operation_compiler import scan_lower as SL
    assert SL.option_defines(None) == ["LSP_CONTRACT"]
    assert SL.option_defines({"contract": False}) == []
    assert SL.option_defines({"backward_group": 4}) == ["LSP_CONTRACT", "LSP_BG 4"]
    assert SL.option_defines({"body": "cops2c"}) == []
    for bad in ({"contract": 1.5}, {"backward_group": 3}, {"unknown": True}, {"body": "cops2c", "contract": True}):
        with pytest.raises(SL.ScanRefused):
            SL.option_defines(bad)


def test_lowering_emits_scan_option_defines_and_diagnostic_body():
    M, cfg, model, captured, inputs, tgt = _capture("bfloat16")
    default = _lowered(captured)
    assert default.manifest["performance_schedule"]["selected"] == "p6e"
    assert '#define LSP_CONTRACT\n#define LSP_BG 4\n#define LSP_FUNROLL 16\n#include "scan_p1a.cuh"' in default.source
    scans = default.manifest["selective_scans"]
    assert scans and all(d["selected"] == "p1a"
                         and d["scan_options"] == {"contract": True, "backward_group": 4, "forward_unroll": 16}
                         and d["scan_defines"] == ["LSP_CONTRACT", "LSP_BG 4", "LSP_FUNROLL 16"] for d in scans)
    primitive = _lowered(captured, scan_options={"contract": True})
    assert '#define LSP_CONTRACT\n#include "scan_p1a.cuh"' in primitive.source
    assert all(d["scan_options"] == {"contract": True} and d["scan_defines"] == ["LSP_CONTRACT"]
               for d in primitive.manifest["selective_scans"])
    plain = _lowered(captured, scan_options={"contract": False})
    assert '#include "scan_p1a.cuh"' in plain.source and "#define LSP_CONTRACT" not in plain.source
    old = _lowered(captured, scan_options={"body": "cops2c"})
    assert '#include "scan_p1a_cops2c.cuh"' in old.source and '#include "scan_p1a.cuh"' not in old.source
    assert len({default.source, primitive.source, plain.source, old.source}) == 4
