"""Explicit BF16 Flash SDPA primal/VJP contract for operation capture.

The default compiler SDPA composition is unchanged. This opt-in contract keeps
BF16 Q/K/V, output and cotangent boundaries, FP32 softmax statistics, and a
paired first-order derivative. Its eager CUDA definition calls PyTorch's
Flash operators, or their memory-efficient peer for windows, with zero dropout. No FP32 math-SDPA substitution is
performed. Native lowering may use the existing BF16 tiled body only after
paired admission; its rounding/tiling still requires workload qualification.

The causal or causal-window bias is an explicit captured constant for the
native body. Capture records its provenance; an arbitrary caller mask cannot
acquire that proof. Grouped query heads and widths below 64 are expressed as
ordinary differentiable layout/padding operations around this primitive.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
from typing import Optional

import torch

from .ir import CaptureError

FORWARD = "latch_ops.bf16_sdpa_fwd.default"
BACKWARD = "latch_ops.bf16_sdpa_bwd.default"
CONTRACT = "bf16-flash-sdpa-v1"
_CAPTURE = ContextVar("latch_bf16_sdpa_capture", default=False)


def enabled():
    return _CAPTURE.get()


@contextmanager
def capture_bf16_sdpa():
    """Select the explicit BF16 SDPA contract inside ordinary fake capture."""
    token = _CAPTURE.set(True)
    try:
        yield
    finally:
        _CAPTURE.reset(token)


def _check(q, k, v, bias, causal, scale):
    if any(x.dtype != torch.bfloat16 or x.dim() != 4 for x in (q, k, v)):
        raise CaptureError("BF16_SDPA", "expected BF16 [batch, heads, sequence, width] operands")
    if q.shape != k.shape or q.shape != v.shape or any(x.device != q.device for x in (k, v)):
        raise CaptureError("BF16_SDPA", "equal self-attention operand shapes/devices required")
    if q.shape[-1] != 64 or not 256 <= q.shape[-2] <= 4096 or q.shape[-2] % 128:
        raise CaptureError("BF16_SDPA", "existing body requires width 64 and sequence 256..4096 divisible by 128")
    if not math.isfinite(scale) or scale <= 0:
        raise CaptureError("BF16_SDPA", "a finite positive static scale is required")
    if causal and bias is None:
        raise CaptureError("BF16_SDPA", "causal mode requires its explicit constant bias")
    if bias is not None and (bias.dtype != torch.float32 or bias.shape != q.shape[-2:-1]*2 or bias.device != q.device):
        raise CaptureError("BF16_SDPA", "causal bias must be a matching FP32 square matrix")


@torch.library.custom_op("latch_ops::bf16_sdpa_fwd", mutates_args=())
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            bias: Optional[torch.Tensor], causal: bool, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    _check(q, k, v, bias, causal, scale)
    if not q.is_cuda:
        raise RuntimeError("BF16 SDPA eager definition requires CUDA Flash; CPU is fake-capture only")
    if bias is not None:
        from .standard_ops import causal_constant_key
        key = causal_constant_key(bias)
        if key is None or (causal and len(key) != 4):
            raise ValueError("BF16 SDPA bias is not the declared generated constant")
    if bias is not None and not causal:
        # PyTorch Flash's dense ABI has no window argument. Its memory-efficient
        # peer supports an additive bias and paired VJP without materializing P.
        # These are eager/reference calls; native execution uses the same common
        # tiled body as other explicit BF16 attention groups.
        result = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q, k, v, bias.expand(q.shape[0], q.shape[1], *bias.shape), True, 0.0, False, scale=scale)
        return result[0].contiguous(), result[1].contiguous()
    result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q, k, v, 0.0, causal, False, scale=scale)
    return result[0].contiguous(), result[1].contiguous()


@forward.register_fake
def _forward_fake(q, k, v, bias, causal, scale):
    _check(q, k, v, bias, causal, scale)
    return q.new_empty(q.shape), q.new_empty(q.shape[:-1], dtype=torch.float32)


@torch.library.custom_op("latch_ops::bf16_sdpa_bwd", mutates_args=())
def backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
             lse: torch.Tensor, dout: torch.Tensor, bias: Optional[torch.Tensor],
             causal: bool, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(q, k, v, bias, causal, scale)
    if not q.is_cuda:
        raise RuntimeError("BF16 SDPA eager VJP requires CUDA Flash; CPU is fake-capture only")
    # Dense, dropout-free PyTorch Flash uses undefined cumulative lengths and
    # does not consume Philox state. Unused uint64 storage matches the forward ABI.
    seed = torch.empty(2, device=q.device, dtype=torch.uint64)
    offset = torch.empty((), device=q.device, dtype=torch.uint64)
    if bias is not None and not causal:
        # With dropout zero the efficient ABI likewise does not consume Philox.
        seed = torch.zeros((), device=q.device, dtype=torch.int64)
        offset = torch.zeros((), device=q.device, dtype=torch.int64)
        result = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(
            dout, q, k, v, bias.expand(q.shape[0], q.shape[1], *bias.shape), o, lse,
            seed, offset, 0.0, [True, True, True, False], False, scale=scale)
        return tuple(x.contiguous() for x in result[:3])
    result = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        dout, q, k, v, o, lse, None, None, q.shape[-2], k.shape[-2],
        0.0, causal, seed, offset, scale=scale)
    return tuple(x.contiguous() for x in result)


@backward.register_fake
def _backward_fake(q, k, v, o, lse, dout, bias, causal, scale):
    _check(q, k, v, bias, causal, scale)
    if o.shape != q.shape or dout.shape != q.shape or o.dtype != q.dtype or dout.dtype != q.dtype:
        raise CaptureError("BF16_SDPA", "output and cotangent must preserve the BF16 query shape")
    if lse.shape != q.shape[:-1] or lse.dtype != torch.float32:
        raise CaptureError("BF16_SDPA", "row statistics must be FP32 [batch, heads, sequence]")
    return tuple(x.new_empty(x.shape) for x in (q, k, v))


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(q, k, v, bias, causal, scale):
        return forward(q, k, v, bias, causal, scale)

    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, bias, ctx.causal, ctx.scale = inputs
        ctx.save_for_backward(q, k, v, bias, *output)
        ctx.mark_non_differentiable(output[1])

    @staticmethod
    def backward(ctx, dout, _dlse):
        q, k, v, bias, o, lse = ctx.saved_tensors
        with torch.no_grad():
            grads = backward(q, k, v, o, lse, dout.clone(memory_format=torch.contiguous_format),
                             bias, ctx.causal, ctx.scale)
        return (*grads, None, None, None)


def _expand_grouped_heads(query, key, value, enable_gqa):
    """Repeat consecutive KV heads through ordinary ops whose VJP sums each group."""
    if any(x.dim() != 4 for x in (query, key, value)):
        raise CaptureError("BF16_SDPA", "expected rank-4 query/key/value")
    if (key.shape != value.shape or query.shape[0] != key.shape[0]
            or query.shape[2:] != key.shape[2:]):
        raise CaptureError("BF16_SDPA", "matching batch, sequence and head widths required")
    hq, hk = query.shape[1], key.shape[1]
    if hq <= 0 or hk <= 0 or hq % hk or (hq != hk and not enable_gqa):
        raise CaptureError("BF16_SDPA", "query heads must be a multiple of KV heads with enable_gqa=True")
    def repeat(x):
        if hq == hk:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, hq // hk, t, d).reshape(b, hq, t, d)
    return query, repeat(key), repeat(value)


def _pad_head_width(x):
    """Exact zero padding; AD slices the real channels and creates no parameter."""
    d = x.shape[-1]
    if not 1 <= d <= 64:
        raise CaptureError("BF16_SDPA", "native BF16 attention supports head widths 1..64 by zero padding")
    if d == 64:
        return x
    zeros = torch.zeros((*x.shape[:-1], 64 - d), dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=-1)


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
         scale=None, enable_gqa=False, *, sliding_window=None):
    """No-dropout BF16 self-attention, including GQA, widths <=64 and causal windows.

    Width padding retains the original scale, and RoPE must be applied before
    this call. Repetition and padding are captured ordinary ops, so their
    gradients include every query head and exclude padded channels. The
    existing native body requires sequence 256..4096 divisible by 128.
    """
    if attn_mask is not None or dropout_p != 0.0:
        raise CaptureError("BF16_SDPA", "caller-supplied masks and dropout are not admitted")
    if sliding_window is not None and (not is_causal or type(sliding_window) is not int or sliding_window <= 0):
        raise CaptureError("BF16_SDPA", "sliding_window requires causal mode and a positive static integer")
    from .standard_ops import causal_mask_constant
    factor = query.shape[-1]**-.5 if scale is None else float(scale)
    original_width = query.shape[-1]
    query, key, value = _expand_grouped_heads(query, key, value, enable_gqa)
    query, key, value = (_pad_head_width(x) for x in (query, key, value))
    bias = causal_mask_constant(query.shape[-2], key.shape[-2], torch.float32, query.device,
                                sliding_window=sliding_window) if is_causal else None
    full_causal = bool(is_causal and (sliding_window is None or sliding_window >= query.shape[-2]))
    _check(query, key, value, bias, full_causal, factor)
    # Exact copies establish independent contiguous roots across region cuts.
    q, k, v = (x.clone(memory_format=torch.contiguous_format) for x in (query, key, value))
    output = _Attention.apply(q, k, v, bias, full_causal, factor)[0]
    return output if original_width == 64 else output[..., :original_width]


def _contiguity_clone_source(ir, name, producers, consumers, readers, protected):
    """ops.attention_rules ``clones``: the value an exact contiguity clone copies, when reading it is identical.

    The clone must be an unconditional contiguous copy (no dtype/layout arguments besides the contiguous
    memory format) read only by the paired primal/VJP, of an unaliased, contiguous, same-shape BF16 value
    that is not an input, output or committed state and is read by nothing else. Returns None otherwise.
    """
    from .ir import _contiguous_strides
    op = producers.get(name)
    if (op is None or op.target != "aten.clone.default" or tuple(op.outputs) != (name,) or len(op.inputs) != 1
            or len(op.args) != 1 or name in protected or not consumers.get(name, set()) <= readers):
        return None
    if op.kwargs and set(op.kwargs) != {"memory_format"} or (
            op.kwargs and op.kwargs["memory_format"] not in (torch.contiguous_format, "contiguous_format",
                                                             "torch.contiguous_format")):
        return None
    source = op.inputs[0]
    value, original = ir.values[source], ir.values[name]
    if (source in ir.inputs or source in protected or value.alias_of is not None or value.storage_offset
            or value.dtype != original.dtype or tuple(value.shape) != tuple(original.shape)
            or tuple(value.strides) != _contiguous_strides(tuple(value.shape))
            or consumers.get(source, set()) != {op.name}):
        return None
    return source


def plans(ir):
    """Prove paired opaque primal/VJP and describe existing flat-body inputs."""
    from .ir import Ref, _contiguous_strides
    operations = ir.operations
    forwards = [op for op in operations if op.target == FORWARD]
    backwards = [op for op in operations if op.target == BACKWARD]
    result, used = {}, set()
    protected = set(ir.output_names())
    causal_constants = set(ir.metadata.get("causal_mask_constants", []))
    consumers = producers = None
    for fwd in forwards:
        if len(fwd.outputs) != 2 or len(fwd.args) != 6:
            raise CaptureError("BF16_SDPA", "invalid forward ABI")
        q, k, v, bias, causal, scale = fwd.args
        o, lse = fwd.outputs
        matches = [op for op in backwards if op.args[:5] == (q, k, v, Ref(o), Ref(lse))]
        if len(matches) != 1:
            raise CaptureError("BF16_SDPA", f"{fwd.name}: exactly one paired VJP required")
        bwd = matches[0]
        if bwd.args[6:] != (bias, causal, scale) or len(bwd.outputs) != 3 or bwd.name in used:
            raise CaptureError("BF16_SDPA", "forward/backward arguments differ")
        if lse in protected or any(lse in op.inputs and op is not bwd for op in operations):
            raise CaptureError("BF16_SDPA", "private row statistics escape their paired VJP")
        if (causal and bias is None) or (bias is not None and bias.name not in causal_constants):
            raise CaptureError("BF16_SDPA", "causal bias lacks captured constant provenance")
        sources = dict(q=q.name, k=k.name, v=v.name, do=bwd.args[5].name)
        shape = ir.values[q.name].shape
        if len(shape) != 4 or any(ir.values[n].shape != shape or ir.values[n].dtype != "bfloat16"
                                  or ir.values[n].strides != _contiguous_strides(shape)
                                  for n in sources.values()):
            raise CaptureError("BF16_SDPA", "operands/cotangent must be contiguous BF16 with matching rank-4 shape")
        b, h, t, d = shape
        if d != 64 or not 256 <= t <= 4096 or t % 128:
            raise CaptureError("BF16_SDPA", "shape outside the existing flat-long body")
        finals = dict(o=o, dq=bwd.outputs[0], dkt=bwd.outputs[1], dv=bwd.outputs[2])
        for name in finals.values():
            value = ir.values[name]
            if value.shape != shape or value.dtype != "bfloat16" or value.strides != _contiguous_strides(shape):
                raise CaptureError("BF16_SDPA", "primal/gradient boundary differs from BF16 contract")
        strides = (h*t*d, t*d, d)
        group = dict(o=o, label=fwd.name, mask=bias.name if bias else None, scale_mode=1, scale=float(scale),
                     mask_kind="causal" if causal else "causal_window" if bias is not None else "none")
        elided = set()
        from . import attention_rules
        if attention_rules.enabled("clones"):
            if consumers is None:
                consumers, producers = {}, {}
                for op in operations:
                    for name in op.inputs:
                        consumers.setdefault(name, set()).add(op.name)
                    for name in op.outputs:
                        producers[name] = op
            for key, name in list(sources.items()):
                source = _contiguity_clone_source(ir, name, producers, consumers, {fwd.name, bwd.name}, protected)
                if source is not None:
                    sources[key] = source
                    elided.add(name)
        result[fwd.name] = dict(group=group, bh=b*h, t=t, hd=d, heads=h, sources=sources, finals=finals,
            source_offsets={}, in_strides={key: strides for key in sources},
            out_strides={key: strides for key in finals}, skip=set((*fwd.outputs, *bwd.outputs, *elided)),
            anchor_forward=o, anchor_backward=bwd.outputs[0], arithmetic_contract=CONTRACT,
            removed=set((*fwd.outputs, *bwd.outputs, *elided)),
            replaced_statistics=lse, **({"elided_clones": sorted(elided)} if elided else {}))
        used.add(bwd.name)
    if len(used) != len(backwards):
        raise CaptureError("BF16_SDPA", "unpaired backward operation")
    return result
