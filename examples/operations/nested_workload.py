"""Nested refinement using ordinary tensor layers and the public loop seam.

The reference below does not call captured graphs, the compiler's VJP, or its
optimizer. Its CPU dropout uses Python unsigned integers, independently of the
tensor Philox implementation used by native capture and conventional controls.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    inputs: int
    width: int
    outer_max: int = 4
    inner_max: int = 5
    seed: int = 200731
    auxiliary: float = 0.17
    dropout: float = 0.125


CASES = {
    "tiny": Case("tiny", 2, 3, 4, seed=200729),
    "held_out": Case("held_out", 3, 12, 20),
    "real_data": Case("real_data", 4, 64, 64, seed=200733),
}
COUNTS = ((0, 0), (0, 3), (1, 0), (1, 1), (2, 3), (3, 2), (4, 5))
MODES = ("save_body_tape", "recompute_boundaries")
TRAIN_COUNTS = ((1, 1), (2, 3), (3, 2), (4, 5))
PERFORMANCE_COUNTS = ((0, 0), (1, 1), (2, 3), (3, 2), (4, 5))
LR, MOMENTUM = 0.03, 0.8


class RefinementModel(torch.nn.Module):
    """A shared layer occurs outside and inside each inner recurrence."""
    def __init__(self, case):
        super().__init__()
        self.input = torch.nn.Linear(case.inputs, case.width)
        self.shared = torch.nn.Linear(case.width, case.width)
        self.head = torch.nn.Linear(case.width, case.inputs)
        generator = torch.Generator().manual_seed(case.seed)
        with torch.no_grad():
            for name, value in self.named_parameters():
                if value.ndim == 2:
                    value.copy_(torch.randn(value.shape, generator=generator) * (0.3 / math.sqrt(value.shape[1])))
                else:
                    value.zero_()


def fixture(case):
    model = RefinementModel(case)
    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    generator = torch.Generator().manual_seed(case.seed + 1)
    data = (torch.randn(case.batch, case.inputs, generator=generator) * 0.2,
            torch.randn(case.batch, case.inputs, generator=generator) * 0.1)
    # Cross a low-counter carry during the first nontrivial refinement.
    state = {"rng": torch.tensor([0x9ABCDEF0, 0x12345678, (1 << 32) - 2, 1 << 8], dtype=torch.int64),
             "visits": torch.tensor(0, dtype=torch.int64)}
    return params, data, state


def capture(case, mode, *, data=None):
    from megakernel_lab.operation_compiler.capture import SGDConfig
    from megakernel_lab.operation_compiler import NestedLoopBody, capture_loop_training
    from megakernel_lab.operation_compiler.random import dropout
    params, examples, state = fixture(case)
    x, target = data if data is not None else examples

    def linear(h, p, name):
        return F.linear(h, p[name + ".weight"], p[name + ".bias"])

    def initial(p, x, y, inner_count):
        return {"h": torch.tanh(linear(x, p, "input")), "aux": x.new_zeros(())}

    def inner_initial(p, outer, x, y, inner_count):
        h = outer["h"] + 0.1 * torch.tanh(linear(outer["h"], p, "shared"))
        return {"h": h, "aux": outer["aux"]}

    def inner_body(p, outer, inner, state, x, y, inner_count):
        h = inner["h"] + 0.2 * torch.tanh(linear(inner["h"], p, "shared") + 0.1 * outer["h"])
        h, rng = dropout(h, state["rng"], case.dropout)
        error = (linear(h, p, "head") - y).square().mean()
        return {"h": h, "aux": inner["aux"] + error}, {"rng": rng, "visits": state["visits"] + 1}

    def finish(p, outer, inner, x, y, inner_count):
        h = inner["h"] + 0.1 * torch.tanh(linear(inner["h"], p, "shared"))
        error = (linear(h, p, "head") - y).square().mean()
        return {"h": h, "aux": inner["aux"] + error}

    def terminal(p, carry, x, y, inner_count):
        return (linear(carry["h"], p, "head") - y).square().mean() + case.auxiliary * carry["aux"]

    nested = NestedLoopBody(inner_initial, inner_body, finish, count_input=2, max_steps=case.inner_max)
    return capture_loop_training(initial, nested, terminal, params, None,
                                 (x, target, torch.tensor(2, dtype=torch.int64)),
                                 torch.tensor(3, dtype=torch.int64),
                                 SGDConfig(lr=LR, momentum=MOMENTUM, weight_decay=0.0),
                                 max_steps=case.outer_max, state=state, storage_mode=mode,
                                 cast_reuse=False)


def independent_masks(state, shape, count, probability):
    """Python-integer Philox4x32-10; fixed masks are constants to autograd.

    Algorithm attribution: Random123 Philox4x32-10, whose contract is documented
    in ops/random.py. This implementation shares no arithmetic helpers with it.
    """
    u32 = (1 << 32) - 1
    k0, k1, low, high = [int(v) & u32 for v in state.cpu().tolist()]
    counter = low | (high << 32)
    n = math.prod(shape)
    blocks = (n + 3) // 4
    threshold = math.floor(probability * (1 << 32))
    # Native inverted-mask scale is rounded to FP32 before multiplication.
    scale = float(torch.tensor(1 / (1 - probability), dtype=torch.float32))
    masks = []
    for _ in range(count):
        words = []
        for index in range(blocks):
            offset = (counter + index) & ((1 << 64) - 1)
            a, b, c, d = offset & u32, offset >> 32, 0, 0
            ka, kb = k0, k1
            for _round in range(10):
                ma, mc = 0xD2511F53 * a, 0xCD9E8D57 * c
                a, b, c, d = ((mc >> 32) ^ b ^ ka) & u32, mc & u32, ((ma >> 32) ^ d ^ kb) & u32, ma & u32
                ka, kb = (ka + 0x9E3779B9) & u32, (kb + 0xBB67AE85) & u32
            words.extend((a, b, c, d))
        masks.append(torch.tensor([scale if w >= threshold else 0.0 for w in words[:n]], dtype=torch.float32).reshape(shape))
        counter = (counter + blocks) & ((1 << 64) - 1)
    final = torch.tensor([k0, k1, counter & u32, counter >> 32], dtype=torch.int64)
    return masks, final


def tensor_dropout(h, rng, probability):
    """Ordinary tensor control; torch.compile can see all RNG arithmetic."""
    from megakernel_lab.operation_compiler.random import _words
    mask = (_words(rng, h.numel()) >= math.floor(probability * (1 << 32))).to(torch.float32).reshape(h.shape)
    mask = mask * (1.0 / (1.0 - probability))
    blocks, u32 = (h.numel() + 3) // 4, (1 << 32) - 1
    low = (rng[2] & u32) + blocks
    high = ((rng[3] & u32) + (low >> 32)) & u32
    return h * mask, torch.stack((rng[0] & u32, rng[1] & u32, low & u32, high))


def objective(params, x, target, rng, outer_count, inner_count, case, *, bf16=False,
              masks=None, omit=None):
    """Independent explicit full-history model; no nested/compiler helper calls.

    The CPU reference deliberately uses x @ W.T + bias, so F.linear lowering
    defects cannot be shared by the implementation and its expected gradients.
    Negative controls preserve the forward value while dropping derivatives.
    """
    def dense(value, prefix, detached=False):
        weight, bias = params[prefix + ".weight"], params[prefix + ".bias"]
        if detached:
            weight, bias = weight.detach(), bias.detach()
        if bf16:
            # FP32 residual/loss and FP32 master parameters; BF16 products.
            return (value.to(torch.bfloat16) @ weight.T.to(torch.bfloat16)).float() + bias
        return value @ weight.T + bias

    h = torch.tanh(dense(x, "input"))
    aux = x.new_zeros(())
    mask_index = 0
    for oi in range(outer_count):
        if omit == "outer_history" and oi:
            h = h.detach()
        outer = h
        h = h + 0.1 * torch.tanh(dense(h, "shared", omit == "outer_shared"))
        for ii in range(inner_count):
            if omit == "inner_history" and ii:
                h = h.detach()
            h = h + 0.2 * torch.tanh(dense(h, "shared") + 0.1 * outer)
            if masks is None:
                h, rng = tensor_dropout(h, rng, case.dropout)
            else:
                h = h * masks[mask_index].to(device=h.device, dtype=h.dtype)
                mask_index += 1
            error = (dense(h, "head") - target).square().mean()
            aux = aux + (error.detach() if omit == "intermediate_loss" else error)
        h = h + 0.1 * torch.tanh(dense(h, "shared", omit == "outer_shared"))
        error = (dense(h, "head") - target).square().mean()
        aux = aux + (error.detach() if omit == "intermediate_loss" else error)
    loss = (dense(h, "head") - target).square().mean() + case.auxiliary * aux
    return loss, (h, aux, rng)


def reference_step(case, names, state_names, carry_names, flat, *, bf16=False):
    """CPU ordinary autograd plus independent explicit SGD equations."""
    n, ns = len(names), len(state_names)
    params = {k: v.detach().clone().requires_grad_(True) for k, v in zip(names, flat[:n])}
    momentum = dict(zip(names, flat[n:2*n]))
    state = dict(zip(state_names, flat[2*n:2*n+ns]))
    x, target, inner_requested, outer_requested = flat[2*n+ns:]
    oc = min(max(int(outer_requested), 0), case.outer_max)
    ic = min(max(int(inner_requested), 0), case.inner_max)
    masks, rng = independent_masks(state["rng"], (case.batch, case.width), oc * ic, case.dropout)
    loss, (h, aux, _) = objective(params, x, target, state["rng"], oc, ic, case, masks=masks, bf16=bf16)
    grads = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
    grads = tuple(torch.zeros_like(value) if g is None else g for value, g in zip(params.values(), grads))
    new_momentum = tuple(MOMENTUM * momentum[k] + g for k, g in zip(names, grads))
    updated = tuple(params[k] - LR * m for k, m in zip(names, new_momentum))
    new_state = {"rng": rng, "visits": state["visits"] + oc * ic}
    carry = {"h": h, "aux": aux}
    # The nested count's clamp status is checked separately against metadata;
    # only in-bound pairs enter this oracle roster.
    result = (loss, *grads, *updated, *new_momentum,
              *(new_state[k] for k in state_names), *(carry[k] for k in carry_names),
              outer_requested.new_tensor(oc), outer_requested.new_tensor(int(oc != int(outer_requested))))
    return tuple(v.detach().clone() for v in result)


def fp64_audit():
    """All tiny parameter derivatives, both histories and finite differences."""
    case = CASES["tiny"]
    params, (x, target), state = fixture(case)
    params = {k: v.double().requires_grad_(True) for k, v in params.items()}
    x, target = x.double(), target.double()
    masks, _ = independent_masks(state["rng"], (case.batch, case.width), 6, case.dropout)
    loss_fn = lambda p, omit=None: objective(p, x, target, state["rng"], 3, 2, case, masks=masks, omit=omit)[0]
    loss = loss_fn(params)
    gradients = dict(zip(params, torch.autograd.grad(loss, tuple(params.values()))))
    worst, checked = 0.0, 0
    for name, value in params.items():
        for index in range(value.numel()):
            plus = {k: v.detach().clone() for k, v in params.items()}
            minus = {k: v.detach().clone() for k, v in params.items()}
            plus[name].view(-1)[index] += 1e-5
            minus[name].view(-1)[index] -= 1e-5
            numerical = (loss_fn(plus) - loss_fn(minus)) / 2e-5
            exact = gradients[name].view(-1)[index]
            ratio = float((numerical - exact).abs() / (2e-8 + 2e-5 * exact.abs()))
            worst = max(worst, ratio)
            checked += 1
    negatives = {}
    for omit in ("outer_history", "inner_history", "outer_shared", "intermediate_loss"):
        wrong_loss = loss_fn(params, omit)
        wrong = torch.autograd.grad(wrong_loss, tuple(params.values()), allow_unused=True)
        differences = {name: float((gradients[name] - (torch.zeros_like(params[name]) if g is None else g)).abs().max())
                       for name, g in zip(params, wrong)}
        negatives[omit] = {"identical_forward": bool(torch.equal(loss, wrong_loss)),
                           "max_gradient_difference": max(differences.values()),
                           "per_parameter": differences,
                           "passed": bool(torch.equal(loss, wrong_loss)) and max(differences.values()) > 1e-6}
    return {"precision": "FP64 products, carry, loss and autograd; fixed FP32 dropout masks",
            "finite_difference_parameters": checked, "worst_error_to_bound_ratio": worst,
            "negative_controls": negatives,
            "passed": worst <= 1 and all(v["passed"] for v in negatives.values())}


def recipe(case):
    return {"case": asdict(case), "model": "nested shared-linear residual refinement",
            "objective": "next-byte normalized regression plus intermediate MSE",
            "scope": "short real-data compiler training smoke; no language-model or data-efficiency claim",
            "dropout": "explicit Philox4x32-10; inverted FP32 masks; nested body consumes one mask",
            "optimizer": {"name": "SGD", "lr": LR, "momentum": MOMENTUM, "weight_decay": 0}}
