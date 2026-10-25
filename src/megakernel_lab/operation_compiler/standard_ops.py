"""General operations needed by standard vision and diffusion transformer modules (C-OPS2-B).

Two mechanisms, neither of which inspects module classes, parameter names or model
families:

1. Capture-time decompositions. PyTorch's own pinned upstream decompositions
   (torch 2.9.1 ``torch._decomp``) expose LayerNorm, GELU, addmm, MSE loss and
   multi-result view operations (split/unbind/stack) as admitted single-Tensor
   ATen operations. They are applied below autograd by ``make_fx``, so derivative
   formulas are PyTorch's. Two rules are written here because PyTorch has no
   decomposition for them:

   * ``aten.convolution`` / ``aten.convolution_backward`` for the admitted
     *non-overlapping* class (stride == kernel, zero padding, unit dilation, one
     group, input extent divisible by the kernel). This is exact patch extraction
     by reshape/permute followed by one matrix product; the backward is the
     transposed products plus a reduction for the bias. The overlapping
     3x3/stride-1/padding-1 class is dispatched to ``conv_ops`` (release step 10:
     patch gather plus matrix products). Every other convolution is refused at
     capture (no silent fallback).
   * ``F.scaled_dot_product_attention`` is replaced, above autograd, by its
     defining composition ``softmax(Q K^T * scale [+ causal mask]) V`` written with
     ``bmm``/``_softmax`` on rank-3 views. AD then derives the ordinary softmax
     attention backward, which the existing operation-level attention lowering
     can recognize. Boolean ``attn_mask`` (True = allowed; release step 09,
     ops/padding_masks.py) adds a 0/-inf bias and uses PyTorch's ``_safe_softmax``
     (fully masked rows are zero). Float masks and dropout_p > 0 are refused.
     This preserves SDPA's declared semantics; fused-kernel rounding order is not
     reproduced (it is not part of the contract).

2. Lowering emitters for operations without a previous rule: ``cat``,
   ``slice_backward``, ``select_backward``, single-index ``index.Tensor`` gathers
   and ``arange``. View rules for ``slice``/``select`` live with the other view
   rules in ``lower.py``/``fusion.py``; pointwise ``erf``/``sin``/``cos``/``clamp``
   rules live with the other pointwise rules.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch

from .ir import CaptureError
from . import conv_ops  # noqa: E402  step 10: registers latch_ops::conv3_patches

aten = torch.ops.aten


# ----------------------------------------------------------------------------- capture
def _patch_geometry(input, weight, stride, padding, dilation, transposed, output_padding, groups):
    if input.dim() != 4 or weight.dim() != 4:
        raise CaptureError("CONVOLUTION", "only rank-4 (N,C,H,W) 2-D convolution is admitted")
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    stride, padding, dilation = (tuple(int(v) for v in x) for x in (stride, padding, dilation))
    n, c, h, w = (int(v) for v in input.shape)
    if (transposed or int(groups) != 1 or stride != (kh, kw) or any(padding) or dilation != (1, 1)
            or any(int(v) for v in output_padding) or h % kh or w % kw or int(weight.shape[1]) != c):
        raise CaptureError("CONVOLUTION", "admitted convolution class is non-overlapping patch extraction: "
                           "stride == kernel, zero padding, unit dilation, one group, divisible input; "
                           f"got stride {stride}, kernel {(kh, kw)}, padding {padding}, dilation {dilation}, "
                           f"groups {groups}, transposed {transposed}, input {(n, c, h, w)}")
    return n, c, h, w, kh, kw, int(weight.shape[0])


def _patches(input, n, c, h, w, kh, kw):
    return input.reshape(n, c, h // kh, kh, w // kw, kw).permute(0, 2, 4, 1, 3, 5).reshape(
        n * (h // kh) * (w // kw), c * kh * kw)


def convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    # Step 10: the overlapping 3x3/stride-1/padding-1 class has its own rule (ops/conv_ops.py).
    overlapping = conv_ops.geometry(input, weight, stride, padding, dilation, transposed, output_padding, groups)
    if overlapping is not None:
        return conv_ops.convolution(input, weight, bias, *overlapping)
    n, c, h, w, kh, kw, o = _patch_geometry(input, weight, stride, padding, dilation, transposed,
                                            output_padding, groups)
    out = torch.mm(_patches(input, n, c, h, w, kh, kw), weight.reshape(o, c * kh * kw).t())
    if bias is not None:
        out = out + bias
    # Return the dense NCHW result a convolution produces (strides and values).
    return out.reshape(n, h // kh, w // kw, o).permute(0, 3, 1, 2).contiguous()


def convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed,
                         output_padding, groups, output_mask):
    overlapping = conv_ops.geometry(input, weight, stride, padding, dilation, transposed, output_padding, groups)
    if overlapping is not None:
        return conv_ops.convolution_backward(grad_output, input, weight, output_mask, *overlapping)
    n, c, h, w, kh, kw, o = _patch_geometry(input, weight, stride, padding, dilation, transposed,
                                            output_padding, groups)
    g = grad_output.permute(0, 2, 3, 1).reshape(n * (h // kh) * (w // kw), o)
    grad_input = grad_weight = grad_bias = None
    if output_mask[0]:
        grad_input = torch.mm(g, weight.reshape(o, c * kh * kw)).reshape(
            n, h // kh, w // kw, c, kh, kw).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)
    if output_mask[1]:
        grad_weight = torch.mm(g.t(), _patches(input, n, c, h, w, kh, kw)).reshape(o, c, kh, kw)
    if output_mask[2]:
        grad_bias = grad_output.sum(dim=(0, 2, 3))
    return grad_input, grad_weight, grad_bias


UPSTREAM = (
    "native_layer_norm.default", "native_layer_norm_backward.default",
    "gelu.default", "gelu_backward.default",
    "addmm.default", "mse_loss.default", "mse_loss_backward.default",
    "split.Tensor", "split_with_sizes.default", "unbind.int", "stack.default",
)


def var_mean(input, dim=None, *, correction=None, keepdim=False):
    """aten.var_mean.correction as two-pass mean/centered-square reductions.

    PyTorch's own reference decomposes to prims (broadcast_in_dim, var) that
    functionalization refuses; this keeps the identical definition in ATen ops.
    """
    dims = list(range(input.dim())) if dim is None else ([dim] if isinstance(dim, int) else list(dim))
    dims = [d % input.dim() for d in dims] or list(range(input.dim()))
    count = math.prod(int(input.shape[d]) for d in dims)
    correction = 1 if correction is None else correction
    mean = torch.mean(input, dims, keepdim=True)
    centered = input - mean
    var = torch.sum(centered * centered, dims, keepdim=True) / max(0, count - correction)
    if not keepdim:
        var, mean = var.squeeze(dims), mean.squeeze(dims)
    return var, mean


def decompositions():
    """Pinned upstream decompositions plus the two local convolution rules."""
    from torch._decomp import get_decompositions
    overloads = []
    for spec in UPSTREAM:
        packet, overload = spec.split(".")
        overloads.append(getattr(getattr(aten, packet), overload))
    table = get_decompositions(overloads)
    missing = [spec for spec, op in zip(UPSTREAM, overloads) if op not in table]
    if missing:
        raise CaptureError("DECOMPOSITION", f"pinned torch lacks upstream decompositions: {missing}")
    table[aten.var_mean.correction] = var_mean
    table[aten.convolution.default] = convolution
    table[aten.convolution_backward.default] = convolution_backward
    return table


_CAUSAL_CONSTANTS = {}


def causal_mask_constant(tq, tk, dtype, device, sliding_window=None):
    """Immutable causal bias, optionally allowing only the last ``sliding_window`` keys.

    A window of W includes the query's own key and W-1 earlier keys. The
    original four-field cache key is retained when the window cannot mask an
    additional position. Capture authenticates each constant's actual bytes.
    """
    if sliding_window is not None and (type(sliding_window) is not int or sliding_window <= 0):
        raise CaptureError("SDPA", "sliding_window must be a positive static integer")
    window = sliding_window if sliding_window is not None and sliding_window < int(tq) else None
    key = (int(tq), int(tk), dtype, str(device)) + (() if window is None else (window,))
    if key not in _CAUSAL_CONSTANTS:
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
        with unset_fake_temporarily(), disable_proxy_modes_tracing():
            queries, keys = torch.arange(int(tq))[:, None], torch.arange(int(tk))[None, :]
            masked = keys > queries
            if window is not None:
                masked = masked | (keys <= queries - window)
            value = (masked.to(dtype) * torch.finfo(dtype).min).to(device)
        _CAUSAL_CONSTANTS[key] = value
    return _CAUSAL_CONSTANTS[key]


def causal_constant_key(value):
    """Prove a captured immutable snapshot equals a generated causal constant.

    Functionalization/AD may replace the original object with multiple frozen
    attributes. Match their actual bytes, not their shape or an operator label.
    This runs after fake tracing, on the real captured constant storage.
    """
    for key, canonical in _CAUSAL_CONSTANTS.items():
        if value is canonical:
            return key
        if (value.shape == canonical.shape and value.dtype == canonical.dtype
                and value.device == canonical.device and value.layout == torch.strided
                and torch.equal(value.detach().contiguous().view(torch.uint8),
                                canonical.detach().contiguous().view(torch.uint8))):
            return key
    return None


def is_causal_constant(value):
    return causal_constant_key(value) is not None


# PyTorch 2.9's math SDPA softmax for float (additive) masks: "safe" (aten._safe_softmax: rows whose masked scores
# are all -inf give zeros) or "plain" (torch.softmax). Finite masks give identical values either way.
FLOAT_MASK_SOFTMAX = "safe"


def sdpa_composition(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
                     enable_gqa=False):
    """Defining composition of F.scaled_dot_product_attention (admitted subset)."""
    if query.dtype == key.dtype == value.dtype == torch.bfloat16:
        # LATCH-ATTN-RECOG: PyTorch 2.9's math reference (aten._scaled_dot_product_attention_math) widens BF16
        # query/key/value to FP32, keeps scores, mask, softmax and the value product in FP32 and rounds the output
        # once to BF16; its gradients are the FP32 gradients rounded once to BF16 (the widenings' VJPs). Pinned by
        # research/latch_release/attn_recog/sdpa_semantics.py. The same composition is written explicitly here, so
        # no score or probability is rounded to BF16 (the FP32 composition below, with BF16-exact operands).
        mask = attn_mask
        if mask is not None and mask.dtype == torch.bfloat16:
            mask = mask.float()  # the math reference adds a BF16 mask into FP32 scores (exact widening)
        return sdpa_composition(query.float(), key.float(), value.float(), mask, dropout_p, is_causal, scale,
                                enable_gqa).to(torch.bfloat16)
    # FP16 inputs stay outside the qualified scope (no BF16-style recognition or qualification exists for them).
    if any(t.dtype not in (torch.float32, torch.float64) for t in (query, key, value)):
        raise CaptureError("SDPA", "direct SDPA capture admits FP32/FP64 or BF16 query/key/value only (BF16 as "
                           "PyTorch's math reference: FP32 scores/softmax, one BF16 output rounding); cast FP16 "
                           "inputs, or use an explicit mixed-precision composition with its own qualification")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise CaptureError("SDPA", "query/key/value must have the same dtype")
    if attn_mask is not None:
        if attn_mask.dtype != torch.bool and attn_mask.dtype != query.dtype:
            raise CaptureError("SDPA", "a float attention mask must have the query dtype (PyTorch's SDPA contract)")
        if is_causal:
            raise CaptureError("SDPA", "attn_mask together with is_causal=True is backend-dependent in PyTorch (the "
                               "math backend rejects it); combine the causal pattern into the mask, "
                               "e.g. causal[None, None] & valid[:, None, None, :]")
    if dropout_p != 0.0:
        raise CaptureError("SDPA", "attention dropout needs explicit random state; dropout_p must be 0")
    if enable_gqa or query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
        raise CaptureError("SDPA", "grouped/broadcast heads are not admitted by this composition")
    if query.dim() < 3 or key.shape[-1] != query.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise CaptureError("SDPA", "query/key/value shapes are inconsistent")
    lead = tuple(query.shape[:-2])
    tq, tk, d, dv = query.shape[-2], key.shape[-2], query.shape[-1], value.shape[-1]
    batch = math.prod(lead)
    factor = (1.0 / math.sqrt(d)) if scale is None else float(scale)
    q = query.reshape(batch, tq, d)
    k = key.reshape(batch, tk, d)
    v = value.reshape(batch, tk, dv)
    scores = torch.bmm(q, k.transpose(1, 2)) * factor
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        # LATCH-ATTN-RECOG: additive float mask, as PyTorch's math reference: scores + mask, then the reference's
        # softmax rule for masked attention (FLOAT_MASK_SOFTMAX, pinned by attn_recog/sdpa_semantics.py).
        from .padding_masks import float_bias, safe_softmax
        try:
            bias = float_bias(attn_mask, lead, tq, tk)
        except ValueError as error:
            raise CaptureError("SDPA", f"float attention mask: {error}") from error
        scores = scores + bias
        probabilities = safe_softmax(scores) if FLOAT_MASK_SOFTMAX == "safe" else torch.softmax(scores, dim=-1)
        return torch.bmm(probabilities, v).reshape(*lead, tq, dv)
    if attn_mask is not None:
        from .padding_masks import boolean_bias, safe_softmax
        try:
            bias = boolean_bias(attn_mask, lead, tq, tk, scores.dtype)
        except ValueError as error:
            raise CaptureError("SDPA", f"boolean attention mask: {error}") from error
        return torch.bmm(safe_softmax(scores + bias), v).reshape(*lead, tq, dv)
    if is_causal:
        # SDPA's causal mask is -inf above the diagonal. The diagonal is never
        # masked, so every row max is finite and exp(finfo.min - max) underflows
        # to exactly 0: the most negative finite additive mask gives identical
        # probabilities and gradients while staying within admitted finite constants.
        # LATCH-ATTN-RECOG: the mask is one captured constant per (shape, dtype, device), shared by every call,
        # so it is bound once like a mask buffer instead of being recomputed per layer.
        scores = scores + causal_mask_constant(tq, tk, scores.dtype, scores.device)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.bmm(probabilities, v).reshape(*lead, tq, dv)


class _CaptureFunctions(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.scaled_dot_product_attention:
            from . import bf16_sdpa
            if bf16_sdpa.enabled() and (args[0] if args else kwargs["query"]).dtype == torch.bfloat16:
                return bf16_sdpa.sdpa(*args, **kwargs)
            return sdpa_composition(*args, **kwargs)
        if func is torch.nn.functional.rms_norm:
            # LATCH-MOESUP: RMS normalization with its compact analytic backward (ops/norms.py).
            from .norms import rms_norm
            return rms_norm(*args, **kwargs)
        return func(*args, **kwargs)


@contextmanager
def capture_functions():
    """Active while tracing; replaces torch-level calls that have no ATen-level rule."""
    with _CaptureFunctions():
        yield


# ----------------------------------------------------------------------------- lowering
SUPPORTED = {"cat", "slice_backward", "select_backward", "index", "arange"}


def emit(target, args, kwargs, name, shape, layouts, dtypes, names, read):
    from .lower import LoweringError, _ident, _ref, _CTYPES

    base = target.split(".")[0]
    if base not in SUPPORTED:
        return None
    out = _ident(name)
    numel = math.prod(shape)
    loop = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; "
            f"i += (long long)gridDim.x * blockDim.x)")
    from .lower import _div32, _mod32
    # C-OPS4-A: 32-bit coordinate arithmetic when the idx32 rule is active (results below 2**31 elements).
    coords = [_mod32(f"({_div32('i', math.prod(shape[d + 1:]))})", shape[d]) if numel < 2 ** 31
              else f"((i / {math.prod(shape[d + 1:])}LL) % {shape[d]}LL)" for d in range(len(shape))]

    def at(ref, coordinates):
        layout = layouts[ref]
        terms = [f"({c}) * {s}LL" for c, s, e in zip(coordinates, layout.strides, layout.shape) if e != 1 and s != 0]
        return f"{_ident(ref)}[{' + '.join(terms) or '0'}]"

    def same_dtype(*refs):
        if any(dtypes[r] != dtypes[name] for r in refs):
            raise LoweringError(f"{target}: operand/result dtypes must agree")

    if base == "cat":
        tensors = [_ref(a, names) for a in args[0]]
        if any(t is None for t in tensors):
            raise LoweringError("cat requires tensor operands")
        same_dtype(*tensors)
        dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % len(shape)
        start, branches = 0, []
        for ref in tensors:
            ls = layouts[ref]
            if len(ls.shape) != len(shape) or any(ls.shape[d] != shape[d] for d in range(len(shape)) if d != dim):
                raise LoweringError("cat operand shapes disagree outside the concatenated dimension")
            extent = ls.shape[dim]
            local = [f"(c - {start}LL)" if d == dim else coords[d] for d in range(len(shape))]
            branches.append((start + extent, at(ref, local)))
            start += extent
        if start != shape[dim]:
            raise LoweringError("cat result extent mismatch")
        body = "".join(f"  {'if' if k == 0 else 'else if'} (c < {end}LL) {out}[i] = {value};\n"
                       for k, (end, value) in enumerate(branches[:-1]))
        body += f"  {'else ' if len(branches) > 1 else ''}{out}[i] = {branches[-1][1]};\n"
        code = f"{loop} {{\n  const long long c = {coords[dim]};\n{body}}}"
        return "concatenate", code, {}, ""

    if base in {"slice_backward", "select_backward"}:
        grad = _ref(args[0], names)
        same_dtype(grad)
        sizes = tuple(int(v) for v in args[1])
        if sizes != shape:
            raise LoweringError(f"{base}: declared input sizes disagree with the result")
        dim = int(args[2]) % len(shape)
        zero = "static_cast<int64_t>(0)" if dtypes[name] == "int64" else "0.0f"
        if dtypes[name] not in {"float32", "int64"}:
            raise LoweringError(f"{base}: FP32/int64 operands required")
        if base == "slice_backward":
            extent = shape[dim]
            start, end, step = int(args[3]), int(args[4]), int(args[5])
            start = max(0, min(extent, start + extent if start < 0 else start))
            end = max(start, min(extent, end + extent if end < 0 else end))
            if step <= 0:
                raise LoweringError("slice_backward requires a positive step")
            local = [f"((c - {start}LL) / {step}LL)" if d == dim else coords[d] for d in range(len(shape))]
            guard = f"c >= {start}LL && c < {end}LL && ((c - {start}LL) % {step}LL) == 0"
        else:
            index = int(args[3]) % shape[dim]
            local = [coords[d] for d in range(len(shape)) if d != dim]
            guard = f"c == {index}LL"
        code = f"{loop} {{\n  const long long c = {coords[dim]};\n  {out}[i] = ({guard}) ? {at(grad, local)} : {zero};\n}}"
        return "embed_gradient", code, {}, ""

    if base == "index":
        if target != "index.Tensor":
            raise LoweringError(f"{target}: only index.Tensor is admitted")
        source = _ref(args[0], names)
        indices = list(args[1])
        if len(indices) != 1 or _ref(indices[0], names) is None:
            raise LoweringError("index.Tensor admits exactly one leading int64 index tensor")
        ids = _ref(indices[0], names)
        if dtypes[ids] != "int64":
            raise LoweringError("index.Tensor requires int64 indices")
        same_dtype(source)
        ls, li = layouts[source], layouts[ids]
        if shape != (*li.shape, *ls.shape[1:]):
            raise LoweringError("index.Tensor result shape mismatch")
        k = len(li.shape)
        row = at(ids, coords[:k])
        local = ["row"] + coords[k:]
        code = (f"{loop} {{\n  const int64_t row = {row};\n  assert(row >= 0 && row < {ls.shape[0]}LL);\n"
                f"  {out}[i] = {at(source, local)};\n}}")
        return "gather", code, {}, "#include <cassert>\n"

    if base == "arange":
        numbers = [a for a in args if isinstance(a, (int, float)) and not isinstance(a, bool)]
        if target == "arange.default":
            start, end, step = 0, numbers[0], 1
        elif target == "arange.start":
            start, end, step = numbers[0], numbers[1], 1
        elif target == "arange.start_step":
            start, end, step = numbers[0], numbers[1], numbers[2]
        else:
            raise LoweringError(f"{target}: unsupported arange overload")
        if len(shape) != 1 or shape[0] != max(0, math.ceil((end - start) / step)):
            raise LoweringError("arange result length mismatch")
        if dtypes[name] not in {"float32", "int64"}:
            raise LoweringError("arange admits FP32/int64 results")
        ctype = _CTYPES[dtypes[name]]
        # torch's CPU/CUDA arange forms start + step * i in the accumulation type.
        code = f"{loop} {{\n  {out}[i] = static_cast<{ctype}>((double){float(start)!r} + (double){float(step)!r} * (double)i);\n}}"
        return "arange", code, {}, ""
    return None
