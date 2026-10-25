"""Step-16 programs: ordinary modules with explicit BF16 products, FP32 master state and SGD momentum.

``ComposedMLP`` is the LATCH-COMPOSE fixture (research/latch_compose/workloads.py at 37361c773), copied with its
arithmetic unchanged so this lane's snapshot is self-contained. The reserved programs below were declared in
RESERVED.json before any tuning measurement; ``dev`` is a previously measured LATCH-COMPOSE peer shape used only
for mechanism smoke checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch.func import functional_call

LR, MOMENTUM = 0.01, 0.9


@dataclass(frozen=True)
class Case:
    name: str
    batch: int = 128
    inputs: int = 128
    width: int = 256
    outputs: int = 64
    depth: int = 3
    auxiliary: float = 0.0
    branch: str = "dense"
    chunk: int = 16
    shared: bool = False
    seed: int = 83017


CASES = {
    # Reserved before tuning (RESERVED.json): two unmeasured shapes and one mixed-operation program.
    "held_awkward": Case("held_awkward", batch=1200, inputs=840, width=1480, outputs=200, depth=2,
                         auxiliary=0.15, seed=1601),
    "held_aligned": Case("held_aligned", batch=1536, inputs=384, width=768, outputs=192, depth=3,
                         auxiliary=0.2, seed=1603),
    "held_mixed": Case("held_mixed", batch=960, inputs=600, width=896, outputs=160, depth=2, auxiliary=0.2,
                       branch="pairwise", chunk=40, shared=True, seed=1605),
    # Relevant configuration change for the cache demonstration (batch differs; otherwise held_awkward).
    "held_awkward_b1000": Case("held_awkward_b1000", batch=1000, inputs=840, width=1480, outputs=200, depth=2,
                               auxiliary=0.15, seed=1601),
    # Mechanism smoke only (LATCH-COMPOSE peer_tma shape, previously measured there).
    "dev": Case("dev", batch=1000, inputs=1000, width=1304, outputs=248, depth=2, auxiliary=0.15, seed=83101),
    # ---- Step 13 (RESERVED13.json). Development suite: step-16 programs (already measured) plus two small shapes.
    "dev_small": Case("dev_small", batch=512, inputs=320, width=512, outputs=128, depth=2, auxiliary=0.15, seed=1311),
    "dev_small_awkward": Case("dev_small_awkward", batch=600, inputs=300, width=520, outputs=72, depth=2,
                              auxiliary=0.15, seed=1313),
    # Step-13 reserved programs: declared before any step-13 rule change or measurement; never used to choose rules.
    "r13_awkward": Case("r13_awkward", batch=1100, inputs=712, width=1176, outputs=136, depth=3, auxiliary=0.1,
                        seed=1321),
    "r13_aligned": Case("r13_aligned", batch=2048, inputs=512, width=1024, outputs=256, depth=2, auxiliary=0.2,
                        seed=1323),
    "r13_mixed": Case("r13_mixed", batch=1152, inputs=520, width=1056, outputs=176, depth=3, auxiliary=0.15,
                      branch="pairwise", chunk=48, shared=True, seed=1325),
    "r13_small": Case("r13_small", batch=640, inputs=200, width=448, outputs=80, depth=3, auxiliary=0.2, seed=1327),
    "r13_wide": Case("r13_wide", batch=768, inputs=1536, width=2048, outputs=1000, depth=1, auxiliary=0.1,
                     seed=1329),
    # ---- Step 14 (RESERVED14.json). r13_mixed is now the development primary; these two nearby mixed-operation
    # programs (pairwise bmm auxiliary + shared block) were declared before any step-14 measurement or change.
    "r14_mixed_a": Case("r14_mixed_a", batch=1248, inputs=456, width=992, outputs=152, depth=3, auxiliary=0.15,
                        branch="pairwise", chunk=48, shared=True, seed=1411),
    "r14_mixed_b": Case("r14_mixed_b", batch=1080, inputs=584, width=1120, outputs=200, depth=4, auxiliary=0.2,
                        branch="pairwise", chunk=40, shared=True, seed=1413),
    # ---- Step 14 speed lane (RESERVED14S.json). r13_mixed and r14_mixed_a/b are development data; these two
    # mixed-operation programs (pairwise bmm auxiliary) were declared before any step-14-speed measurement or change.
    "r14s_mixed_c": Case("r14s_mixed_c", batch=1216, inputs=648, width=1152, outputs=168, depth=3, auxiliary=0.2,
                         branch="pairwise", chunk=32, shared=False, seed=1521),
    "r14s_mixed_d": Case("r14s_mixed_d", batch=1344, inputs=520, width=1248, outputs=232, depth=4, auxiliary=0.1,
                         branch="pairwise", chunk=56, shared=True, seed=1523),
    # Cache demonstration only (batch differs from r13_awkward).
    "r13_awkward_b1000": Case("r13_awkward_b1000", batch=1000, inputs=712, width=1176, outputs=136, depth=3,
                              auxiliary=0.1, seed=1321),
}
RESERVED = ("held_awkward", "held_aligned", "held_mixed")
DEV13 = ("dev", "held_awkward", "held_aligned", "held_mixed", "dev_small", "dev_small_awkward")
RESERVED13 = ("r13_awkward", "r13_aligned", "r13_mixed", "r13_small", "r13_wide")
PRIMARY14 = "r13_mixed"
RESERVED14 = ("r14_mixed_a", "r14_mixed_b")
# Step-14 speed lane: r14_mixed_b is the development target (formerly reserved); new reserved transfers below.
TARGET14S = "r14_mixed_b"
DEV14S = ("r13_mixed", "r14_mixed_a", "r14_mixed_b", "held_mixed")
RESERVED14S = ("r14s_mixed_c", "r14s_mixed_d")
# Established fast paths checked for regressions (already-measured programs; not reserved).
REGRESSION14 = ("r13_awkward", "r13_aligned", "r13_small", "r13_wide", "dev_small")


def product_bf16(x, weight):
    return (x.to(torch.bfloat16) @ weight.to(torch.bfloat16)).float()


def forward(case, params, x, product=product_bf16):
    """ComposedMLP.forward with an injectable product (the FP64 reference rounds where the recipe rounds)."""
    hidden = torch.nn.functional.silu(product(x, params["input"]))
    auxiliary = None
    for index in range(case.depth):
        weight = params["blocks.0" if case.shared else f"blocks.{index}"]
        hidden = hidden + 0.25 * torch.nn.functional.silu(product(hidden, weight))
        if index == 0 and case.auxiliary:
            if case.branch == "pairwise":
                rows = hidden.reshape(case.batch // case.chunk, case.chunk, case.width)
                auxiliary = torch.bmm(rows, rows.transpose(1, 2)) / math.sqrt(case.width)
            else:
                auxiliary = product(hidden, params["auxiliary_head"])
    return product(hidden, params["head"]), auxiliary


class ComposedMLP(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        self.case = case
        generator = torch.Generator().manual_seed(case.seed)

        def weight(n, m, gain=0.4):
            return torch.nn.Parameter(torch.randn(n, m, generator=generator) * (gain / math.sqrt(n)))
        self.input = weight(case.inputs, case.width)
        self.blocks = torch.nn.ParameterList(
            [weight(case.width, case.width) for _ in range(1 if case.shared else case.depth)])
        self.head = weight(case.width, case.outputs, 0.2)
        if case.auxiliary and case.branch == "dense":
            self.auxiliary_head = weight(case.width, case.outputs, 0.2)

    def forward(self, x):
        # Attribute access (as in the donor fixture) so functional_call's substituted tensors are used.
        params = {"input": self.input, "head": self.head,
                  **{f"blocks.{i}": block for i, block in enumerate(self.blocks)}}
        if hasattr(self, "auxiliary_head"):
            params["auxiliary_head"] = self.auxiliary_head
        return forward(self.case, params, x)


def fixture(case, seed_offset=1):
    generator = torch.Generator().manual_seed(case.seed + seed_offset)
    x = torch.randn(case.batch, case.inputs, generator=generator) * 0.4
    y = torch.randn(case.batch, case.outputs, generator=generator) * 0.1
    if case.branch == "pairwise":
        rows = y.reshape(case.batch // case.chunk, case.chunk, case.outputs)
        pairwise_target = torch.bmm(rows, rows.transpose(1, 2)) / math.sqrt(case.outputs)
        return (x,), (y, pairwise_target)
    return (x,), (y,)


def loss_fn(case):
    def loss(prediction, target, pairwise_target=None):
        terminal, auxiliary = prediction
        value = (terminal - target).square().mean()
        if auxiliary is not None:
            expected = pairwise_target if case.branch == "pairwise" else target
            value = value + case.auxiliary * (auxiliary - expected).square().mean()
        return value
    return loss


def capture(name, *, lr=LR, weight_seed=None):
    """Normal-API capture. ``weight_seed`` changes initial values only (structure unchanged)."""
    from megakernel_lab.operation_compiler import SGDConfig, capture_module
    case = CASES[name]
    model = ComposedMLP(replace(case, seed=weight_seed) if weight_seed is not None else case)
    model.case = case
    inputs, targets = fixture(case)
    captured = capture_module(model, loss_fn(case), inputs, targets,
                              optimizer=SGDConfig(lr=lr, momentum=MOMENTUM, weight_decay=0.0))
    return model, captured, inputs, targets


def reference_loss(case, params, inputs, targets, *, fp64=False):
    """Conventional explicit-cast recipe (cuBLAS BF16 products) or its FP64 evaluation (same roundings)."""
    if fp64:
        r = lambda t: t.to(torch.bfloat16).to(torch.float64)
        product = lambda x, w: r(r(x) @ r(w))
        params = {k: v.double() for k, v in params.items()}
        inputs = tuple(t.double() for t in inputs)
        targets = tuple(t.double() for t in targets)
    else:
        product = product_bf16
    return loss_fn(case)(forward(case, params, inputs[0], product), *targets)


def describe(name):
    return asdict(CASES[name])
