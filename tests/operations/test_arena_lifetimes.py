"""LATCH-TRAIN-CORRECTNESS: arena lifetimes start at a root's EARLIEST writer (CPU; lowering only, no nvcc).

A root can be written by more than one phase: the per-launch zeroed arrival counters of a split BF16 product
(mixed_gemm 'p2_split') are written by their matrix_scratch_init phase and again by the product, which restores
them to zero. Lifetimes used to start at the LAST writer (the product), so the arena could give the counters
storage that another value still occupied while the initialization ran one dependency interval earlier. In gated
TinyMoE programs that zeroed part of the attention-backward cotangent (TINYMOE-SKIP-GATE's k_proj defect).

The test lowers a small program whose skinny long-K weight gradient selects the split body (an [8, 12] gate
weight over 2,048 rows, as TinyMoE's per-head attention gate) next to independent work, and checks on the
manifest that every arena root's planned lifetime covers every phase that reads or writes it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _Gate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.gate = torch.nn.Parameter(torch.randn(8, 12, generator=g) * 0.2)
        self.proj = torch.nn.Parameter(torch.randn(64, 96, generator=g) * 0.1)

    def forward(self, x):
        from torch.nn import functional as F
        gates = torch.sigmoid(F.linear(x[..., :12].to(torch.bfloat16), self.gate.to(torch.bfloat16)).float())
        values = F.linear(x.to(torch.bfloat16), self.proj.to(torch.bfloat16)).float()   # [R, 64]
        y = values.reshape(*values.shape[:-1], 8, 8) * gates.unsqueeze(-1)
        return (y * y).mean()


def _program():
    from megakernel_lab.operation_compiler.api import capture_module
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig
    x = torch.randn(2048, 96, generator=torch.Generator().manual_seed(1))
    captured = capture_module(_Gate(), lambda out: out, (x,), (), optimizer=SGDConfig(lr=0.01, momentum=0.9))
    return lower(captured.ir, target_arch="sm_90", execution="compose", precision_policy="practical")


def _interval_of(manifest):
    assert not manifest.get("row_chain_substages"), "test program unexpectedly uses row chains"
    return {i: e for e, members in enumerate(manifest["barrier_intervals"]) for i in members}


def test_split_counters_are_live_from_their_initialization():
    program = _program()
    manifest = program.manifest
    inits = [i for i, phase in enumerate(manifest["phases"])
             if phase["kind"] == "matrix_scratch_init" and "generic_split_counter_zero" in phase.get("targets", ())]
    assert inits, "the skinny long-K weight gradient no longer selects the split body; update this test"
    interval = _interval_of(manifest)
    layouts, lifetimes, offsets = manifest["layouts"], manifest["lifetimes"], manifest["offsets"]
    for i in inits:
        for name in manifest["phases"][i]["writes"]:
            root = layouts[name]["root"]
            kind, first, last = lifetimes[root]
            assert first <= interval[i] <= last, (root, lifetimes[root], interval[i])
    # The general property, over every arena root of the program.
    for i, phase in enumerate(manifest["phases"]):
        for name in list(phase["reads"]) + list(phase["writes"]):
            root = layouts.get(name, {}).get("root", name)
            life = lifetimes.get(root)
            if root in offsets and isinstance(life, list):
                assert life[1] <= interval[i] <= life[2], (root, life, i, interval[i], phase["kind"])
