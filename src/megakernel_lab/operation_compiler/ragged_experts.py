"""Packed token-choice routing and segmented matrix products for common capture.

The row extent is static (tokens * selected experts), while every segment length
is a device value. Empty segments are valid, every selected row is retained, and
changing assignments does not change the compiled program. The three primitives
are useful beyond MoE: stable bucket ordering, segmented right products, and
segmented outer-product reductions. Autograd composes the latter two as VJPs.

BF16 products accumulate in FP32 and round once to BF16, as ordinary BF16 mm.
FP32 is also supported for independent small operation/gradient references.
"""
from __future__ import annotations

import torch

ORDER = "latch_ops.ragged_order.default"
MM = "latch_ops.ragged_mm.default"
OUTER = "latch_ops.ragged_outer.default"
# Capture-time form of ``ragged_outer(...).float()`` for BF16 operands (layout_canonical.widen_segment_products).
OUTER_WIDE = "latch_ops.ragged_outer_widened.default"
# LATCH-SELECTIVE-10 (ops.accumulation_rules accfuse): an IR-only form (no capture-time op) of a BF16 ragged_outer
# whose per-segment results feed acc_e + where(flag_e, float(result_e), +0); lowered with an accumulating epilogue.
OUTER_ACC = "latch_ops.ragged_outer_acc"
# LATCH-TRAIN-PRECISION: BF16 operands, FP32 accumulation, unrounded FP32 result (opt-in training precision policies).
MM_RAW = "latch_ops.ragged_mm_raw.default"
TARGETS = (ORDER, MM, MM_RAW, OUTER, OUTER_WIDE, OUTER_ACC)


def _segments(offsets, rows):
    if offsets.dtype != torch.int64 or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("segment offsets must be an int64 vector of length groups + 1")
    # Concrete custom-op implementations validate the runtime ABI. Fake capture
    # and native lowering check metadata; generated helpers check bounds too.
    boundaries = offsets.tolist()
    if boundaries[0] != 0 or boundaries[-1] != rows or any(a > b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("segment offsets must monotonically cover all packed rows")
    return boundaries


def _check_mm(x, weight, offsets):
    if x.ndim != 2 or weight.ndim != 3 or x.shape[1] != weight.shape[1]:
        raise ValueError("ragged_mm expects [rows, in] and [groups, in, out]")
    if x.dtype != weight.dtype or x.dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("ragged products require matching floating operand dtypes")
    if offsets.dtype != torch.int64 or offsets.shape != (weight.shape[0] + 1,):
        raise ValueError("ragged_mm offsets must have groups + 1 entries")
    if x.device != weight.device or x.device != offsets.device:
        raise ValueError("ragged product operands must share a device")


@torch.library.custom_op("latch_ops::ragged_order", mutates_args=())
def _order(selected: torch.Tensor, groups: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if selected.dtype != torch.int64 or selected.ndim != 2 or not 1 <= groups <= 256:
        raise ValueError("routing requires int64 [tokens, choices] indices and 1..256 groups")
    ids = selected.reshape(-1)
    if bool(((ids < 0) | (ids >= groups)).any()):
        raise ValueError("routing index outside group range")
    order = torch.argsort(ids, stable=True)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(ids.numel(), device=ids.device)
    counts = torch.bincount(ids, minlength=groups)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return order.contiguous(), offsets.contiguous(), inverse.contiguous()


@_order.register_fake
def _order_fake(selected, groups):
    if selected.dtype != torch.int64 or selected.ndim != 2 or not 1 <= groups <= 256:
        raise ValueError("routing requires int64 [tokens, choices] indices and 1..256 groups")
    return (selected.new_empty((selected.numel(),)), selected.new_empty((groups + 1,)),
            selected.new_empty((selected.numel(),)))


def route_order(selected, num_experts):
    """Return (stable packed-to-choice order, segment offsets, inverse order).

    A choice index is ``token * choices + slot``. Thus a packed input is
    ``embedding(order // choices, x)``; unpack with ``embedding(inverse, y)``.
    No capacity padding, overflow path or dropped-token convention is involved.
    """
    return _order(selected.contiguous(), int(num_experts))


@torch.library.custom_op("latch_ops::ragged_mm", mutates_args=())
def _mm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    _check_mm(x, weight, offsets)
    boundaries = _segments(offsets, x.shape[0])
    return torch.cat([x[lo:hi] @ weight[e] for e, (lo, hi) in enumerate(zip(boundaries, boundaries[1:]))], dim=0)


@_mm.register_fake
def _mm_fake(x, weight, offsets):
    _check_mm(x, weight, offsets)
    return x.new_empty((x.shape[0], weight.shape[2]))


@torch.library.custom_op("latch_ops::ragged_outer", mutates_args=())
def _outer(x: torch.Tensor, dy: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or dy.ndim != 2 or x.shape[0] != dy.shape[0] or x.dtype != dy.dtype:
        raise ValueError("ragged_outer requires matching packed row extents and dtypes")
    boundaries = _segments(offsets, x.shape[0])
    return torch.stack([x[lo:hi].T @ dy[lo:hi] for lo, hi in zip(boundaries, boundaries[1:])])


@_outer.register_fake
def _outer_fake(x, dy, offsets):
    if x.ndim != 2 or dy.ndim != 2 or x.shape[0] != dy.shape[0] or x.dtype != dy.dtype:
        raise ValueError("ragged_outer requires matching packed row extents and dtypes")
    if offsets.ndim != 1 or offsets.dtype != torch.int64 or offsets.numel() < 2:
        raise ValueError("ragged_outer requires int64 segment offsets")
    return x.new_empty((offsets.numel() - 1, x.shape[1], dy.shape[1]))


@torch.library.custom_op("latch_ops::ragged_outer_widened", mutates_args=())
def _outer_widened(x: torch.Tensor, dy: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``ragged_outer(x, dy, offsets).float()``: the BF16-rounded product, exactly widened to FP32 storage.

    Never produced by autograd; capture introduces it only where a BF16 product's one reader is that cast.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError("ragged_outer_widened requires BF16 operands")
    return _outer(x, dy, offsets).float()


@_outer_widened.register_fake
def _outer_widened_fake(x, dy, offsets):
    if x.dtype != torch.bfloat16:
        raise ValueError("ragged_outer_widened requires BF16 operands")
    return _outer_fake(x, dy, offsets).float()


@torch.library.custom_op("latch_ops::ragged_mm_raw", mutates_args=())
def _mm_raw(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``ragged_mm`` of BF16 operands whose FP32 accumulators are returned unrounded (FP32 storage).

    The reference evaluates each segment product in FP32 from the exactly widened BF16 operands.
    """
    _check_mm(x, weight, offsets)
    if x.dtype != torch.bfloat16:
        raise ValueError("ragged_mm_raw requires BF16 operands")
    boundaries = _segments(offsets, x.shape[0])
    return torch.cat([x[lo:hi].float() @ weight[e].float()
                      for e, (lo, hi) in enumerate(zip(boundaries, boundaries[1:]))], dim=0)


@_mm_raw.register_fake
def _mm_raw_fake(x, weight, offsets):
    _check_mm(x, weight, offsets)
    if x.dtype != torch.bfloat16:
        raise ValueError("ragged_mm_raw requires BF16 operands")
    return x.new_empty((x.shape[0], weight.shape[2]), dtype=torch.float32)


class _RaggedMM(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, offsets):
        return _mm(x, weight, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, dy):
        x, weight, offsets = ctx.saved_tensors
        # First-order capture: the derivative operations themselves are opaque
        # under torch.func, as the existing SDPA and scan paired primitives are.
        with torch.no_grad():
            dx = _mm(dy.contiguous(), weight.transpose(1, 2), offsets)
            dw = _outer(x, dy.contiguous(), offsets)
        return dx, dw, None


def ragged_mm(x, weight, offsets):
    """Differentiable packed segmented right product; first-order AD supported."""
    return _RaggedMM.apply(x, weight, offsets)


def token_topk(scores, k=2):
    """CUDA small-row top-k tie law for FP32 softmax probabilities, width <=32.

    Selection and its gradient use common argmax/where/gather operations. The
    small bitonic output-order law is shared with the existing 32-way helper.
    """
    from .torch_topk32 import topk_small
    return topk_small(scores, k)


def token_choice_experts(x, selected, weights, w1, w2, w3):
    """SwiGLU selected experts from ordinary elementwise/gather ops + products.

    ``x`` is [tokens,D], ``selected/weights`` [tokens,K], w1/w3 [E,D,F],
    and w2 [E,F,D]. The caller supplies already-normalized routing weights.
    """
    from torch.nn import functional as F
    order, offsets, inverse = route_order(selected, w1.shape[0])
    packed = F.embedding(order // selected.shape[1], x)
    activated = F.silu(ragged_mm(packed, w1, offsets)) * ragged_mm(packed, w3, offsets)
    output = ragged_mm(activated, w2, offsets)
    choices = F.embedding(inverse, output).reshape(x.shape[0], selected.shape[1], x.shape[1])
    return (choices * weights.unsqueeze(-1)).sum(dim=1)


def router_auxiliary_loss(router_logits, top_k=2, *, probability_dtype=torch.float32):
    """Global token-choice balancing objective, before its external coefficient.

    For L equal-token layers concatenate all L*N router rows. With probabilities
    p, one-hot selected expert I, and E experts, return
    ``E * sum_{slot,expert} mean_rows(I) * mean_rows(p)``.
    ``probability_dtype`` is explicit because CUDA and CPU autocast can choose
    different softmax policies. FP32 is the CUDA-autocast contract; a caller
    requesting BF16 retains that softmax/mean rounding boundary. The dispatch
    selected-weight normalization remains a separate objective input.
    """
    logits = torch.cat(tuple(router_logits), dim=0)
    # The explicit widen/softmax/narrow exposes the BF16 softmax storage boundary
    # to common lowering while retaining PyTorch's FP32 softmax accumulator.
    if probability_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("router probability policy must be FP32 or BF16")
    probabilities = logits.float().softmax(-1).to(probability_dtype)
    _, selected = token_topk(probabilities.float(), top_k)
    experts = probabilities.shape[-1]
    membership = (selected.unsqueeze(-1) == torch.arange(experts, device=logits.device)).float()
    return (membership.mean(0) * probabilities.mean(0).unsqueeze(0)).sum() * experts
