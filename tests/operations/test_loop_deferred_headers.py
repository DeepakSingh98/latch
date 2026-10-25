"""LATCH-TRAIN-CORRECTNESS: a deferred-recurrent-gradient training loop records every header its build needs (CPU).

loop_pipeline added only kernels/step06_dgrad.cuh to a loop program's headers. That header includes the P5C
warpgroup body (p5c_wg.cuh and its own includes), which regions with generic matrix bodies do not record. Such a
program could not build (p5c_wg.cuh was not written next to program.cu) and runtime.build would have chosen sm_90 for
wgmma code. The loop below is the shape of examples/operations/train_runtime_loop.py with explicit BF16 products
small enough for generic region bodies; it must record the deferred stage's whole include closure.
"""
from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")


def _captured():
    from megakernel_lab.operation_compiler import SGDConfig, capture_loop_training

    def mm(a, b):
        return (a.to(torch.bfloat16) @ b.to(torch.bfloat16)).float()

    def initial(params, x, target):
        hidden = mm(x, params["encoder"])
        return {"hidden": hidden, "auxiliary": hidden.sum() * 0.0}

    def body(params, carry, state, x, target):
        hidden = carry["hidden"] + 0.1 * torch.tanh(mm(carry["hidden"], params["shared"]))
        auxiliary = carry["auxiliary"] + (mm(hidden, params["head"]) - target).square().mean()
        return {"hidden": hidden, "auxiliary": auxiliary}, {"iterations": state["iterations"] + 1}

    def loss(params, carry, x, target):
        return (mm(carry["hidden"], params["head"]) - target).square().mean() + 0.05 * carry["auxiliary"]

    g = torch.Generator().manual_seed(17)
    params = {n: torch.randn(256, 256, generator=g) * 0.05 for n in ("encoder", "shared", "head")}
    inputs = (torch.randn(256, 256, generator=g) * 0.2, torch.randn(256, 256, generator=g) * 0.1)
    return capture_loop_training(initial, body, loss, params, None, inputs, torch.tensor(0),
                                 SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.001), max_steps=4,
                                 state={"iterations": torch.zeros((), dtype=torch.int64)})


def _includes(name):
    from megakernel_lab.operation_compiler.lower import _header_path
    return [n for n in re.findall(r'#include\s+"([^"/]+)"', _header_path(name).read_text()) if n.endswith(".cuh")]


def test_deferred_stage_records_its_include_closure():
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.p5c_matrix import ARCH_HEADERS
    captured = _captured()
    program = lower(captured.ir, execution="compose", recurrent_gradients="deferred")
    headers = set(program.manifest["header_hashes"])
    assert "step06_dgrad.cuh" in headers
    for name in [h for h in headers if h.endswith(".cuh")]:
        missing = set(_includes(name)) - headers
        assert not missing, (name, missing)
    # The recorded names now select the warpgroup target in runtime.build.
    assert headers & set(ARCH_HEADERS)
    inline = lower(captured.ir, execution="compose", recurrent_gradients="inline")
    assert "step06_dgrad.cuh" not in inline.manifest["header_hashes"]
