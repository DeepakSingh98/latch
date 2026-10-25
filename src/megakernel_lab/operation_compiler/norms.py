"""RMS normalization with a compact analytic backward (LATCH-MOESUP).

``F.rms_norm(x, normalized_shape, weight, eps)`` reaches :func:`rms_norm` during capture
(``standard_ops._CaptureFunctions``). The forward is PyTorch's composition in the input dtype, in its operation order:

    r = rsqrt(mean(x * x over the normalized dims) + eps);   y = (x * r) * weight

Differentiating that composition directly materializes several full-size intermediates in the backward (squares,
broadcast scale chains, partial products). The analytic backward evaluates the same derivative with one row
reduction and saves only ``x``, the per-row ``r`` and the weight:

    xhat = x * r;  gw = g * weight;  dx = r * (gw - xhat * mean(gw * xhat));  dweight = sum_rows(g * xhat)

This is the same derivative of the same function (an FP32 reassociation of the composed backward's arithmetic, as
fused normalization kernels do); forward values are unchanged. Scope: FP32/FP64 inputs, normalized trailing
dimensions, optional weight. ``ANALYTIC = False`` restores the plain composition (A/B arms).
"""
from __future__ import annotations

import math

import torch

ANALYTIC = True


def _dims(x, normalized_shape):
    n = len(normalized_shape)
    if tuple(x.shape[-n:]) != tuple(normalized_shape):
        raise ValueError("rms_norm normalized_shape must match the trailing input dimensions")
    return tuple(range(x.dim() - n, x.dim()))


def _composition(x, normalized_shape, weight, eps):
    dims = _dims(x, normalized_shape)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    y = x * torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
    return y * weight if weight is not None else y


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, dims, eps):
        r = torch.rsqrt(x.pow(2).mean(dims, keepdim=True) + eps)
        y = x * r
        return (y * weight if weight is not None else y), r

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, dims, eps = inputs
        ctx.dims = dims
        ctx.save_for_backward(x, weight, output[1])
        ctx.mark_non_differentiable(output[1])

    @staticmethod
    def backward(ctx, g, _gr):
        x, weight, r = ctx.saved_tensors
        dims = ctx.dims
        xhat = x * r
        gw = g * weight if weight is not None else g
        dot = (gw * xhat).mean(dims, keepdim=True)
        dx = r * (gw - xhat * dot)
        dweight = None
        if weight is not None and ctx.needs_input_grad[1]:
            lead = tuple(range(x.dim() - len(dims)))
            dweight = (g * xhat).sum(lead) if lead else g * xhat
        return dx, dweight, None, None


def rms_norm(x, normalized_shape, weight=None, eps=None):
    normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if not ANALYTIC or x.dtype not in (torch.float32, torch.float64) or (
            weight is not None and weight.dtype != x.dtype):
        return _composition(x, normalized_shape, weight, eps)
    dims = _dims(x, normalized_shape)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    if not math.isfinite(float(eps)):
        raise ValueError("rms_norm eps must be finite")
    return _RMSNorm.apply(x, weight, dims, float(eps))[0]
