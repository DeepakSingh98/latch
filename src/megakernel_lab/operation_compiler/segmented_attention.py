"""Packed causal BF16 attention with device-resident segment boundaries.

Rows are concatenated without per-segment capacity padding. ``offsets`` describes
every packed row exactly once; empty segments and changing lengths are valid.
The native implementation reuses the flat-long v7 BF16 WGMMA arithmetic with
predicated tail loads/stores and only enumerates live 64-row tiles.

Op contract (``latch_ops.segmented_sdpa_fwd`` / ``latch_ops.segmented_sdpa_bwd``,
wrapped by ``sdpa``): BF16 ``[packed_rows, heads, width]`` with native width 32 or
64, device int32/int64 offsets ``[segments + 1]``, a static ``max_seqlen`` in
1..4096 and a finite positive static scale. ``sdpa`` accepts any width 1..64 and
zero-pads it to the next native width: heads of width <= 32 (for example 30) use
the width-32 body and are zero-padded to 32; wider heads up to 64 use the v7
width-64 body. Native work counters are incremented at the executed items/tiles,
not inferred from lengths (``sdpa(..., return_work=True)``).
CUDA eager uses PyTorch's paired packed Flash operators with device offsets.
CPU eager is a small mathematical reference, never the native execution path.
The opaque primal's private FP32 row statistics are log2(sum(exp(scores))) in
packed [heads,rows] order. The Flash eager adapter converts its natural-log ABI
at this boundary; native forward/backward keep the existing log2 recipe.

Native exponentials are explicit, so another operation's FTZ option cannot
alter this primitive when the two are composed into the same program. The width-64
body uses exp2f. Lowering reaches exactly three width-32 schedules:

* schedule 6, the default (contract suffix "+ftz-ex2"), uses ex2.approx.ftz, the
  exponential of Latch's audited dense v7 items; it differs from exp2f only for
  probabilities below 2^-126, which become 0 instead of subnormal;
* schedule 8 with forward form 9, selected by ``segmented_attention`` token
  ``folded_forward`` (folded-scale forward);
* schedule 8 with backward form 10, selected by ``segmented_attention`` token
  ``fused_backward`` (single-pass backward).

The other numbered schedules and forms in kernels/attention_segmented.cuh
(width-32 schedules 0..5 and 7, the remaining forward and backward forms and the
one-warpgroup diagnostic arms) cannot be selected from any build option; they
are legacy forms kept in the header until a separate pruning change removes them.

The build option ``segmented_attention`` (ops.segmented_attention_rules) opts
width-32 heads into the folded-scale forward, the single-pass backward, claimed
items and fused rotary operand passes; without it the lowering above is
unchanged. Those rules, and ``tile_readiness``, apply only to width-32 heads.
"""
from __future__ import annotations

import math
import torch

FORWARD = "latch_ops.segmented_sdpa_fwd.default"
BACKWARD = "latch_ops.segmented_sdpa_bwd.default"
TARGETS = (FORWARD, BACKWARD)
CONTRACT = "packed-segmented-bf16-causal-v1"
NATIVE_WIDTHS = (32, 64)
LOG2E = math.log2(math.e)
FORWARD_WORK = ("valid_rows", "query_tiles", "pair_tiles")
BACKWARD_WORK = ("dq_valid_rows", "dq_query_tiles", "dq_pair_tiles",
                 "dkv_valid_rows", "dkv_key_tiles", "dkv_pair_tiles")


def _check(q, k, v, offsets, max_seqlen, scale):
    if any(x.ndim != 3 or x.dtype != torch.bfloat16 for x in (q, k, v)):
        raise ValueError("segmented attention expects BF16 [packed_rows, heads, 32 or 64]")
    if q.shape != k.shape or q.shape != v.shape or q.shape[-1] not in NATIVE_WIDTHS or q.shape[1] < 1:
        raise ValueError("segmented attention requires equal shapes and native width 32 or 64")
    if any(x.device != q.device for x in (k, v, offsets)):
        raise ValueError("segmented attention operands and offsets must share a device")
    if offsets.dtype not in (torch.int32, torch.int64) or offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("segment offsets must be device int32/int64 [segments + 1]")
    if type(max_seqlen) is not int or not 1 <= max_seqlen <= 4096:
        raise ValueError("max_seqlen is a static bound in 1..4096")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("segmented attention needs a finite positive static scale")


def _cpu_segments(offsets, rows, bound):
    boundaries = offsets.tolist()  # CPU reference only; never CUDA/native.
    if boundaries[0] != 0 or boundaries[-1] != rows or any(
            not 0 <= b - a <= bound for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("offsets must monotonically cover all packed rows within max_seqlen")
    return zip(boundaries, boundaries[1:])


def _cpu_forward(q, k, v, offsets, max_seqlen, scale):
    out = torch.empty_like(q)
    lse = q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32)
    for lo, hi in _cpu_segments(offsets, q.shape[0], max_seqlen):
        if lo == hi:
            continue
        qs, ks, vs = (x[lo:hi].transpose(0, 1).float() for x in (q, k, v))
        scores = qs @ ks.transpose(-1, -2) * scale
        causal = torch.ones((hi-lo, hi-lo), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, -torch.inf)
        probs = scores.softmax(-1)
        out[lo:hi] = (probs @ vs).transpose(0, 1).to(q.dtype)
        lse[:, lo:hi] = scores.logsumexp(-1) * LOG2E
    return out, lse


@torch.library.custom_op("latch_ops::segmented_sdpa_fwd", mutates_args=())
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor,
            max_seqlen: int, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(q, k, v, offsets, max_seqlen, scale)
    if q.is_cuda:
        cu = offsets.to(torch.int32).contiguous()
        result = torch.ops.aten._flash_attention_forward.default(
            q, k, v, cu, cu, max_seqlen, max_seqlen, 0.0, True, False, scale=scale)
        out, lse = result[:2]
        # Packed Flash stores row statistics as [heads, total packed rows].
        if lse.shape != (q.shape[1], q.shape[0]):
            raise RuntimeError("packed Flash returned an unsupported row-statistics layout")
        lse = lse * LOG2E
    else:
        out, lse = _cpu_forward(q, k, v, offsets, max_seqlen, scale)
    # External eager backend tile counts are deliberately not presented as native measurements.
    work = torch.full((offsets.numel()-1, q.shape[1], 3), -1, dtype=torch.int64, device=q.device)
    return out.contiguous(), lse.contiguous(), work


@forward.register_fake
def _forward_fake(q, k, v, offsets, max_seqlen, scale):
    _check(q, k, v, offsets, max_seqlen, scale)
    return (q.new_empty(q.shape), q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
            offsets.new_empty((offsets.numel()-1, q.shape[1], 3), dtype=torch.int64))


@torch.library.custom_op("latch_ops::segmented_sdpa_bwd", mutates_args=())
def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
             lse: torch.Tensor, dout: torch.Tensor, offsets: torch.Tensor,
             max_seqlen: int, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(q, k, v, offsets, max_seqlen, scale)
    if q.is_cuda:
        cu = offsets.to(torch.int32).contiguous()
        seed = torch.empty(2, dtype=torch.uint64, device=q.device)
        offset = torch.empty((), dtype=torch.uint64, device=q.device)
        grads = torch.ops.aten._flash_attention_backward.default(
            dout.contiguous(), q, k, v, o, lse / LOG2E, cu, cu, max_seqlen, max_seqlen,
            0.0, True, seed, offset, scale=scale)
    else:
        grads = [torch.empty_like(q) for _ in range(3)]
        for lo, hi in _cpu_segments(offsets, q.shape[0], max_seqlen):
            if lo == hi:
                continue
            qs, ks, vs, ds = (x[lo:hi].transpose(0, 1).float() for x in (q, k, v, dout))
            scores = qs @ ks.transpose(-1, -2) * scale
            causal = torch.ones((hi-lo, hi-lo), device=q.device, dtype=torch.bool).tril()
            p = scores.masked_fill(~causal, -torch.inf).softmax(-1)
            dp = ds @ vs.transpose(-1, -2)
            dscores = p * (dp - (p * dp).sum(-1, keepdim=True)) * scale
            for g, z in zip(grads, (dscores @ ks, dscores.transpose(-1, -2) @ qs, p.transpose(-1, -2) @ ds)):
                g[lo:hi] = z.transpose(0, 1).to(q.dtype)
    work = torch.full((offsets.numel()-1, q.shape[1], 6), -1, dtype=torch.int64, device=q.device)
    return *(g.contiguous() for g in grads), work


@backward.register_fake
def _backward_fake(q, k, v, o, lse, dout, offsets, max_seqlen, scale):
    _check(q, k, v, offsets, max_seqlen, scale)
    if o.shape != q.shape or dout.shape != q.shape or o.dtype != q.dtype or dout.dtype != q.dtype:
        raise ValueError("output and cotangent must preserve packed BF16 shape")
    if lse.shape != (q.shape[1], q.shape[0]) or lse.dtype != torch.float32:
        raise ValueError("log2 row statistics must be packed FP32 [heads, packed_rows]")
    return *(x.new_empty(x.shape) for x in (q, k, v)), offsets.new_empty(
        (offsets.numel()-1, q.shape[1], 6), dtype=torch.int64)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, offsets, max_seqlen, scale):
        return forward(q, k, v, offsets, max_seqlen, scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, offsets, ctx.max_seqlen, ctx.scale = inputs
        ctx.save_for_backward(q, k, v, offsets, output[0], output[1])
        ctx.mark_non_differentiable(output[1], output[2])

    @staticmethod
    def backward(ctx, dout, _dlse, _dwork):
        q, k, v, offsets, o, lse = ctx.saved_tensors
        with torch.no_grad():
            dq, dk, dv, _work = backward(q, k, v, o, lse, dout.contiguous(), offsets, ctx.max_seqlen, ctx.scale)
        return dq, dk, dv, None, None, None


def native_width_for(width, native_width=None):
    """Smallest native packed width holding ``width`` (or an explicitly requested wider one)."""
    chosen = next(w for w in NATIVE_WIDTHS if width <= w) if native_width is None else native_width
    if chosen not in NATIVE_WIDTHS or chosen < width:
        raise ValueError(f"native width must be one of {NATIVE_WIDTHS} and hold the head width")
    return chosen


def sdpa(query, key, value, offsets, *, max_seqlen=4096, scale=None, return_work=False, native_width=None):
    """Causal attention on ordered, concatenated segments; no row padding.

    Input/output: BF16 ``[total_rows, heads, width]`` with width 1..64. Offsets
    are device int32/int64 and must monotonically span [0,total_rows], with each
    length <= max_seqlen. Head padding is exactly zero and keeps the caller's
    scale. Apply RoPE at original positions before packing/calling this API.
    Width 32 or 64 enters the native body unchanged; other widths are zero-padded
    to the next native width (``native_width`` may request 64 explicitly, for
    example to compare a narrow head on the width-64 body).
    ``return_work`` exposes native executed [valid_rows, query_tiles, pair_tiles]
    counters per [segment,head]; eager backends return -1 (not instrumented).
    """
    if query.ndim != 3 or not 1 <= query.shape[-1] <= 64:
        raise ValueError("segmented SDPA expects [packed_rows, heads, width<=64]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("segmented SDPA expects equal query/key/value shapes")
    width = query.shape[-1]
    native = native_width_for(width, native_width)
    factor = width ** -.5 if scale is None else float(scale)
    def padded(x):
        if width < native:
            x = torch.cat((x, x.new_zeros((*x.shape[:-1], native-width))), dim=-1)
        return x.contiguous()
    q, k, v = (padded(x) for x in (query, key, value))
    out, _lse, work = _Attention.apply(q, k, v, offsets.contiguous(), max_seqlen, factor)
    out = out if width == native else out[..., :width]
    return (out, work) if return_work else out
