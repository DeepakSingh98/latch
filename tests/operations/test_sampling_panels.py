"""P8-D CPU contracts for the panel engine rule (ops/sampling_panels.py) and commit aliasing.

Run on a Modal CPU worker (torch 2.9.1): python -m pytest -q ops/test_sampling_panels.py
Lowering is inspected only; device behaviour is qualified by the GPU gates (p8_d/POLICY.json).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.optim import AdamWConfig, initialize_optimizer_state
from megakernel_lab.operation_compiler.random import make_key, philox_field
from megakernel_lab.operation_compiler.sampling import capture_sampled_training


def _sampled(B=8, D=16, H=32, K=2, change_state=False, reduce_rows=False):
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
        x = (y - 0.5 * g.clamp(-0.1, 0.1)).clamp(0, 1)
        if reduce_rows:
            x = x - x.mean(-1, keepdim=True)
        if change_state:
            s = {**s, "ptr": s["ptr"] + 1}
        return {"x": x}, s

    def loss(p, c, s, data):
        pos = data[:B]
        e = energy(p, torch.cat([pos, c["x"]]))
        slots = torch.remainder(s["ptr"] + torch.arange(B, dtype=torch.int64), 20)
        new = {"buf": s["buf"].index_copy(0, slots, c["x"]), "ptr": torch.remainder(s["ptr"] + B, 20),
               "t": s["t"] + 1, "key": s["key"]}
        return e[:B].mean() - e[B:].mean(), new

    config = AdamWConfig(lr=1e-3, betas=(0.0, 0.999), max_grad_norm=1.0, param_options={"b": {"weight_decay": 0.0}})
    opt = initialize_optimizer_state(params, config)
    return capture_sampled_training(init, step, loss, params, (data,), torch.tensor(K), config, max_steps=K,
                                    state=state, optimizer_state=opt)


def test_panels_admit_row_separable_step():
    program = lower(_sampled().ir, sampled_options={"panels": "on"})
    engine = program.manifest["panel_engine"]
    assert engine["selected"] and engine["rule"] == "p8d-panel-engine-v1"
    assert [p["name"] for p in engine["products"]]  # every product of the step is a panel product
    assert "panel_engine.cuh" in program.manifest["header_hashes"]
    assert "panel_loop(" in program.source and "for (int iteration = 0; iteration < count; ++iteration)" in program.source
    # the ordered body region is not emitted when the panel loop replaces it
    assert "sampled_body(" not in program.source


def test_panels_auto_records_refusal_and_falls_back():
    narrow = _sampled(D=6, H=8)
    program = lower(narrow.ir, sampled_options={"panels": "auto"})
    assert program.manifest["panel_engine"]["selected"] is False
    assert "multiple of 16" in program.manifest["panel_engine"]["reason"]
    with pytest.raises(LoweringError, match="panel engine refused"):
        lower(narrow.ir, sampled_options={"panels": "on"})


@pytest.mark.parametrize("kind,reason", [("change_state", "changes explicit state"),
                                         ("reduce_rows", "mean")])
def test_panels_refuse_non_admitted_steps(kind, reason):
    captured = _sampled(**{kind: True})
    program = lower(captured.ir, sampled_options={"panels": "auto"})
    assert program.manifest["panel_engine"]["selected"] is False
    assert reason in program.manifest["panel_engine"]["reason"]


def test_defaults_select_the_rule_and_parent_is_reproducible():
    captured = _sampled()
    default = lower(captured.ir)
    explicit = lower(captured.ir, sampled_options={"panels": "auto", "commit": "alias"})
    assert default.source == explicit.source and default.manifest["panel_engine"]["selected"]
    parent = lower(captured.ir, sampled_options={"panels": "off", "commit": "copy"})
    assert parent.manifest["panel_engine"] == {"selected": False, "reason": "panels off"}
    assert parent.manifest["committed_aliases"] == []
    assert "panel_loop(" not in parent.source


def test_commit_alias_outputs_are_bindings():
    captured = _sampled()
    program = lower(captured.ir, sampled_options={"commit": "alias"})
    m = program.manifest
    aliases = set(m["committed_aliases"])
    s = m["structured_outputs"]
    names = list(captured.parameter_names)
    assert {s["params"][k] for k in names} <= aliases
    for name in aliases:
        assert m["layouts"][name]["root"] in m["inputs"]
        assert name not in m["offsets"]
