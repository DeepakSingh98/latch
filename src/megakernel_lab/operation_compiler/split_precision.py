"""Split-BF16 weight products: an opt-in training precision policy built from ordinary BF16 products (LATCH-TRAIN-PRECISION).

``split_bf16_linear(x, w, weight_terms=T)`` is a BF16 linear whose FP32 master weight is represented by T BF16
terms instead of one:

    w_0 = bf16(w),  w_t = bf16(w - sum_{s<t} w_s)        (residuals formed in FP32; w - sum_t w_t ~ 2^(-8T) |w|)
    y   = bf16( sum_t bf16(x) . w_t^T )                   (one product over the concatenated K, FP32 accumulation)

It is written as one ordinary BF16 matrix product, ``linear(cat([x] * T, -1).bf16(), cat([w, r_1, ...], 1).bf16())``,
so every existing BF16 matrix body (and fused consumer rules, e.g. the vocabulary head) applies with K multiplied by
T; no new kernel is introduced. Arithmetic, per term, is exactly the released BF16 product arithmetic (BF16 operands,
FP32 accumulation, one BF16 rounding of the result); only the weight operand's representation error changes.

Gradients (ordinary AD of that expression):
  * input: dx = sum_t widen(bf16(dy . w_t)), summed in FP32 (the cast follows the FP32 concatenation, so the T
    cotangent slices widen before they are added) -- the released input gradient with the weight error removed;
  * weight: the residual terms are detached, so dw = widen(bf16(dy^T . bf16(x))), bit-for-bit the released weight
    gradient expression (the extra columns of the weight-gradient product are computed and discarded).

T = 1 is the released ``linear(x.bf16(), w.bf16())``. The policy is explicit at the model function level (the
user's forward); there is no implicit global precision change, and an FP32 program written without casts is
unaffected. Cost: T times the K extent of the selected product and of its input-gradient product.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

BF16 = torch.bfloat16


def split_weight_terms(weight, terms):
    """FP32 pieces whose BF16 casts are the T split terms of ``weight``: [w, r_1, ..., r_{T-1}], residuals detached."""
    if type(terms) is not int or terms < 1:
        raise ValueError("weight_terms must be a positive integer")
    if weight.dtype != torch.float32:
        raise ValueError("split_bf16_linear needs the FP32 master weight")
    pieces, residual = [weight], weight.detach()
    for _ in range(terms - 1):
        residual = residual - residual.to(BF16).float()
        pieces.append(residual)
    return pieces


def split_bf16_linear(input, weight, *, weight_terms=2, input_terms=1):
    """BF16 result of ``input @ weight.T`` with the FP32 weight (and optionally the FP32 input) carried as BF16 terms.

    Term pairs (input term i, weight term j) with i + j < max(input_terms, weight_terms) are summed in one product
    over the concatenated K extent; for 1 x 2: x.w0 + x.w1, for 2 x 2: x0.w0 + x0.w1 + x1.w0 (the 3-term split
    product; the dropped x1.w1 is ~2^-16 relative). Residual terms of the input are detached (the input gradient is
    sum_j dy.w_j over the leading input term), residual weight terms are detached (the weight gradient is
    sum_i dy^T.x_i over the leading weight term)."""
    if input.dtype not in (torch.float32, BF16):
        raise ValueError("split_bf16_linear input must be FP32 or BF16")
    if weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError("split_bf16_linear expects input [..., K] and weight [N, K]")
    if type(input_terms) is not int or input_terms < 1 or (input_terms > 1 and input.dtype != torch.float32):
        raise ValueError("input_terms > 1 needs an FP32 input")
    if weight_terms == 1 and input_terms == 1:
        return F.linear(input.to(BF16), weight.to(BF16))
    w_pieces = split_weight_terms(weight, weight_terms)
    x_pieces = [input]
    residual = input.detach()
    for _ in range(input_terms - 1):
        residual = residual - residual.to(BF16).float()
        x_pieces.append(residual)
    order = max(input_terms, weight_terms)
    pairs = [(i, j) for i in range(input_terms) for j in range(weight_terms) if i + j < order]
    operand = torch.cat([x_pieces[i] for i, _ in pairs], dim=-1).to(BF16)
    return F.linear(operand, torch.cat([w_pieces[j] for _, j in pairs], dim=1).to(BF16))


def _forward_operands(x, x_bf16, weight_pieces_bf16, input_terms, weight_terms, *, k_dim=-1, w_dim=1):
    """Concatenated forward operands of the term pairs (i, j), i + j < max(terms): (x_split, w_split)."""
    x_pieces = [x_bf16]
    if input_terms > 1:
        residual = x.detach().float()
        for _ in range(input_terms - 1):
            residual = residual - residual.to(BF16).float()
            x_pieces.append(residual.to(BF16))
    order = max(input_terms, weight_terms)
    pairs = [(i, j) for i in range(input_terms) for j in range(weight_terms) if i + j < order]
    return (torch.cat([x_pieces[i] for i, _ in pairs], dim=k_dim),
            torch.cat([weight_pieces_bf16[j] for _, j in pairs], dim=w_dim))


class _SplitForwardLinear(torch.autograd.Function):
    """Forward: one BF16 product over the split operands. Backward: the released BF16 products on (x_hi, w_hi)."""

    @staticmethod
    def forward(x_hi, w_hi, x_split, w_split):
        return F.linear(x_split, w_split)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_hi, w_hi, _, _ = inputs
        ctx.save_for_backward(x_hi, w_hi)

    @staticmethod
    def backward(ctx, dy):
        x, w_hi = ctx.saved_tensors
        with torch.no_grad():
            dx = torch.matmul(dy, w_hi)
            dw = torch.mm(dy.reshape(-1, dy.shape[-1]).t(), x.reshape(-1, x.shape[-1]))
        return dx, dw, None, None


def split_forward_linear(input, weight, *, weight_terms=2, input_terms=1):
    """BF16 linear whose FORWARD product reads the FP32 weight (and, with input_terms == 2, the FP32 input) as BF16
    terms (pairs i + j < max(terms) in one product over the concatenated K); its backward is the released
    ``linear(bf16(x), bf16(w))`` backward (dx = dy . bf16(w), dw = dy^T . bf16(x), widened by the casts' VJPs).
    T = 1 is the released linear."""
    if weight_terms == 1 and input_terms == 1:
        return F.linear(input.to(BF16), weight.to(BF16))
    if input_terms > 1 and input.dtype != torch.float32:
        raise ValueError("input_terms > 1 needs an FP32 input")
    x = input.to(BF16)
    w_hi = weight.to(BF16)
    w_pieces = [p.to(BF16) for p in split_weight_terms(weight.detach(), weight_terms)]
    x_split, w_split = _forward_operands(input, x, w_pieces, input_terms, weight_terms)
    return _SplitForwardLinear.apply(x, w_hi, x_split, w_split)


class _SplitForwardRagged(torch.autograd.Function):
    """Packed segmented product, forward over the split operands; backward = released ragged_mm VJP on (x_hi, w_hi)."""

    @staticmethod
    def forward(x_hi, w_hi, x_split, w_split, offsets):
        from .ragged_experts import _mm
        return _mm(x_split.contiguous(), w_split, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_hi, w_hi, _, _, offsets = inputs
        ctx.save_for_backward(x_hi, w_hi, offsets)

    @staticmethod
    def backward(ctx, dy):
        from .ragged_experts import _mm, _outer
        x, w_hi, offsets = ctx.saved_tensors
        with torch.no_grad():
            dx = _mm(dy.contiguous(), w_hi.transpose(1, 2), offsets)
            dw = _outer(x, dy.contiguous(), offsets)
        return dx, dw, None, None, None


def ragged_split_operands(x, hi, lo, *, input_terms=1, x_operands=None):
    """Forward operands of a split expert product in the released layouts.

    ``x``: FP32 (input_terms 1 or 2) or BF16 (input_terms 1) packed rows [R, in]; ``hi``/``lo``: BF16 [E, out, in]
    storage of the weight terms (hi = bf16(w), lo = bf16(w - hi)). Returns (x_hi, w_hi, x_split, w_split) with w_hi
    and w_split transposed views of contiguous [E, out, T*in] storage (K-contiguous, as the released operand)."""
    order = max(input_terms, 2)
    pairs = [(i, j) for i in range(input_terms) for j in range(2) if i + j < order]
    w_pieces = [hi, lo]
    w_split = torch.cat([w_pieces[j] for _, j in pairs], dim=2).transpose(1, 2)
    if x_operands is not None:
        # The x_split of an earlier call on the same input and term count: one shared forward A operand, so products
        # reading it may pair (e.g. the w1/w3 expert products). x_hi is a fresh cast per product, so each product's
        # BF16 input cotangent widens separately before the FP32 sum (as the released separate casts).
        _, x_split = x_operands
        return x.to(BF16), hi.transpose(1, 2), x_split, w_split
    x_hi = x.to(BF16)
    x_pieces = [x_hi]
    if input_terms > 1:
        if x.dtype != torch.float32:
            raise ValueError("input_terms > 1 needs the FP32 input")
        x_pieces.append((x.detach() - x_hi.float()).to(BF16))
    x_split = torch.cat([x_pieces[i] for i, _ in pairs], dim=-1)
    return x_hi, hi.transpose(1, 2), x_split, w_split


def split_forward_ragged_mm(x_hi, w_hi, x_split, w_split, offsets):
    """``ragged_mm(x_hi, w_hi, offsets)`` whose forward is ragged_mm(x_split, w_split) (``ragged_split_operands``);
    backward = the released ragged_mm VJP on (x_hi, w_hi)."""
    if any(t.dtype != BF16 for t in (x_hi, w_hi, x_split, w_split)):
        raise ValueError("split ragged products take BF16 operands")
    if w_split.shape[1] != x_split.shape[1] or w_split.shape[0] != w_hi.shape[0] or w_split.shape[2] != w_hi.shape[2]:
        raise ValueError("split operands disagree")
    return _SplitForwardRagged.apply(x_hi, w_hi, x_split, w_split, offsets)


class _SplitForwardGrouped(torch.autograd.Function):
    """torch._grouped_mm forward over split operands; backward = PyTorch's _grouped_mm VJP on (a_hi, b_hi)
    (da = grouped(g, b_hi^T), db = grouped(a_hi^T, g)), i.e. the released conventional expert products."""

    @staticmethod
    def forward(a_hi, b_hi, a_split, b_split, offs):
        return torch._grouped_mm(a_split, b_split, offs=offs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_hi, b_hi, _, _, offs = inputs
        ctx.save_for_backward(a_hi, b_hi, offs)

    @staticmethod
    def backward(ctx, g):
        a, b_hi, offs = ctx.saved_tensors
        with torch.no_grad():
            da = torch._grouped_mm(g, b_hi.transpose(-2, -1), offs=offs)
            db = torch._grouped_mm(a.transpose(-2, -1), g, offs=offs)
        return da, db, None, None, None


def split_forward_grouped_mm(a_hi, b_hi, a_split, b_split, offs):
    """Conventional (torch._grouped_mm) counterpart of ``split_forward_ragged_mm`` (operands from
    ``ragged_split_operands``)."""
    if any(t.dtype != BF16 for t in (a_hi, b_hi, a_split, b_split)):
        raise ValueError("split grouped products take BF16 operands")
    return _SplitForwardGrouped.apply(a_hi, b_hi, a_split, b_split, offs)


def expert_terms(weights, dtype=BF16):
    """BF16 storage (hi, lo) [E, out, in] of a list of FP32 [out, in] expert weights (lo detached)."""
    hi = torch.stack(list(weights)).to(dtype)
    lo = torch.stack([(w - w.to(dtype).float()).detach() for w in weights]).to(dtype)
    return hi, lo


def split_ragged_mm(x, weights, offsets, *, weight_terms=2, input_terms=1):
    """Packed segmented product ``ragged_mm(x, stack(weights).T, offsets)`` with the FP32 expert weights (and the FP32
    input when input_terms == 2) carried as BF16 terms over the concatenated K extent; ordinary AD of the
    concatenations gives (residual terms detached): dx = sum_j dy . w_j^T (each widened, summed in FP32), and per
    expert dw = sum_i x_i^T . dy (each widened, summed in FP32). ``weights``: list of FP32 [out, in] expert weights."""
    from .ragged_experts import ragged_mm
    if x.dtype not in (torch.float32, BF16) or (input_terms > 1 and x.dtype != torch.float32):
        raise ValueError("split_ragged_mm input must be FP32 (BF16 only with one input term)")
    stacked = torch.stack(list(weights))                                  # [E, out, in] FP32
    w_pieces = split_weight_terms(stacked, weight_terms)
    x_pieces, residual = [x if x.dtype == torch.float32 else x.float()], x.detach().float()
    for _ in range(input_terms - 1):
        residual = residual - residual.to(BF16).float()
        x_pieces.append(residual)
    order = max(input_terms, weight_terms)
    pairs = [(i, j) for i in range(input_terms) for j in range(weight_terms) if i + j < order]
    operand = torch.cat([x_pieces[i] for i, _ in pairs], dim=-1).to(BF16)
    matrix = torch.cat([w_pieces[j] for _, j in pairs], dim=2).to(BF16).transpose(1, 2)
    return ragged_mm(operand, matrix, offsets)


def split_grouped_mm(x, weights, offs, *, weight_terms=2, input_terms=1):
    """Conventional (torch._grouped_mm, ordinary AD) counterpart of ``split_ragged_mm``: the same term pairs over the
    concatenated K extent, residual terms detached."""
    stacked = torch.stack(list(weights))
    w_pieces = split_weight_terms(stacked, weight_terms)
    x_pieces, residual = [x if x.dtype == torch.float32 else x.float()], x.detach().float()
    for _ in range(input_terms - 1):
        residual = residual - residual.to(BF16).float()
        x_pieces.append(residual)
    order = max(input_terms, weight_terms)
    pairs = [(i, j) for i in range(input_terms) for j in range(weight_terms) if i + j < order]
    operand = torch.cat([x_pieces[i] for i, _ in pairs], dim=-1).to(BF16)
    matrix = torch.cat([w_pieces[j] for _, j in pairs], dim=2).to(BF16).transpose(1, 2)
    return torch._grouped_mm(operand, matrix, offs=offs)


# ----------------------------------------------------------------------------- FP32-result split products
# LATCH-TRAIN-PRECISION forward_fp32: the forward product of a split policy keeps its FP32 accumulators (no BF16 result
# rounding); BF16 operands (the hi/lo terms), FP32 accumulation, unrounded FP32 result. Backward: the incoming FP32
# cotangent is rounded once to BF16 (the cotangent the released BF16 product receives) and the released BF16 VJP
# products run on (bf16(x), bf16(w)). Latch spells the dense product as an FP32 mm of exact BF16 widenings (lowered
# to BF16 WGMMA with an unrounded FP32 result); the conventional control uses mm/_grouped_mm with out_dtype=float32.


@torch.library.custom_op("ltp_precision::mm_fp32", mutates_args=())
def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Conventional control: BF16 [M, K] x [K, N] -> unrounded FP32 (cuBLAS, FP32 accumulation). An opaque op so
    torch.compile(fullgraph=True) traces it (PyTorch 2.9 has no fake rule for mm(out_dtype=...))."""
    if a.dtype != BF16 or b.dtype != BF16:
        raise ValueError("mm_fp32 takes BF16 operands")
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


@_mm_fp32.register_fake
def _mm_fp32_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


@torch.library.custom_op("ltp_precision::grouped_mm_fp32", mutates_args=())
def _grouped_mm_fp32(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Conventional control: grouped product of BF16 operands with an unrounded FP32 result.

    PyTorch 2.9.1's CUDA _grouped_mm requires out_dtype == input dtype, so the exactly widened FP32 operands are used
    (FP32 accumulation; PyTorch runs its per-group fallback for FP32 inputs, ~30x slower than the BF16 grouped GEMM
    on H100 - a library limitation of this control, recorded with its timing)."""
    if a.dtype != BF16 or b.dtype != BF16:
        raise ValueError("grouped_mm_fp32 takes BF16 operands")
    return torch._grouped_mm(a.float(), b.float(), offs=offs)


@_grouped_mm_fp32.register_fake
def _grouped_mm_fp32_fake(a, b, offs):
    return a.new_empty((a.shape[0], b.shape[-1]), dtype=torch.float32)


def _dense_fp32(a, b, conventional):
    """[M, K] x [K, N] BF16 operands -> unrounded FP32 [M, N] (FP32 accumulation)."""
    if conventional:
        return _mm_fp32(a, b)
    return torch.mm(a.float(), b.float())


class _RawForwardLinear(torch.autograd.Function):
    @staticmethod
    def forward(x_hi, w_hi, x_split, w_split, conventional):
        flat = x_split.reshape(-1, x_split.shape[-1])
        y = _dense_fp32(flat, w_split.t(), conventional)
        return y.reshape(*x_split.shape[:-1], w_split.shape[0])

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_hi, w_hi, _, _, _ = inputs
        ctx.save_for_backward(x_hi, w_hi)

    @staticmethod
    def backward(ctx, dy):
        x, w_hi = ctx.saved_tensors
        with torch.no_grad():
            g = dy.to(BF16)
            dx = torch.matmul(g, w_hi)
            dw = torch.mm(g.reshape(-1, g.shape[-1]).t(), x.reshape(-1, x.shape[-1]))
        return dx, dw, None, None, None


def raw_forward_linear(input, weight, *, input_terms=1, conventional=False):
    """FP32 result of ``input @ weight.T`` from the BF16 split operands (weight hi + lo; with input_terms == 2 the FP32
    input hi + lo: 3-term product x0.w0 + x0.w1 + x1.w0 over 3K); backward = released BF16 VJP on (bf16(x), bf16(w))
    after one BF16 rounding of the cotangent."""
    if input_terms > 1 and input.dtype != torch.float32:
        raise ValueError("input_terms > 1 needs an FP32 input")
    x = input.to(BF16)
    w_hi = weight.to(BF16)
    w_pieces = [p.to(BF16) for p in split_weight_terms(weight.detach(), 2)]
    x_split, w_split = _forward_operands(input, x, w_pieces, input_terms, 2)
    return _RawForwardLinear.apply(x, w_hi, x_split, w_split, bool(conventional))


class _RawForwardRagged(torch.autograd.Function):
    @staticmethod
    def forward(x_hi, w_hi, x_split, w_split, offsets):
        from .ragged_experts import _mm_raw
        return _mm_raw(x_split.contiguous(), w_split, offsets)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x_hi, w_hi, _, _, offsets = inputs
        ctx.save_for_backward(x_hi, w_hi, offsets)

    @staticmethod
    def backward(ctx, dy):
        from .ragged_experts import _mm, _outer
        x, w_hi, offsets = ctx.saved_tensors
        with torch.no_grad():
            g = dy.to(BF16).contiguous()
            dx = _mm(g, w_hi.transpose(1, 2), offsets)
            dw = _outer(x, g, offsets)
        return dx, dw, None, None, None


def raw_forward_ragged_mm(x_hi, w_hi, x_split, w_split, offsets):
    """FP32-result ``ragged_mm`` over the split operands (``ragged_split_operands``); backward = the released ragged VJP
    on (x_hi, w_hi) after one BF16 rounding of the cotangent."""
    if any(t.dtype != BF16 for t in (x_hi, w_hi, x_split, w_split)):
        raise ValueError("split ragged products take BF16 operands")
    return _RawForwardRagged.apply(x_hi, w_hi, x_split, w_split, offsets)


class _RawForwardGrouped(torch.autograd.Function):
    @staticmethod
    def forward(a_hi, b_hi, a_split, b_split, offs):
        return _grouped_mm_fp32(a_split, b_split, offs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        a_hi, b_hi, _, _, offs = inputs
        ctx.save_for_backward(a_hi, b_hi, offs)

    @staticmethod
    def backward(ctx, g):
        a, b_hi, offs = ctx.saved_tensors
        with torch.no_grad():
            g = g.to(BF16)
            da = torch._grouped_mm(g, b_hi.transpose(-2, -1), offs=offs)
            db = torch._grouped_mm(a.transpose(-2, -1), g, offs=offs)
        return da, db, None, None, None


def raw_forward_grouped_mm(a_hi, b_hi, a_split, b_split, offs):
    """Conventional (torch._grouped_mm, out_dtype FP32) counterpart of ``raw_forward_ragged_mm``."""
    if any(t.dtype != BF16 for t in (a_hi, b_hi, a_split, b_split)):
        raise ValueError("split grouped products take BF16 operands")
    return _RawForwardGrouped.apply(a_hi, b_hi, a_split, b_split, offs)


def describe(weight_terms):
    return dict(policy=f"split_bf16_weight_x{weight_terms}", weight_terms=weight_terms,
                forward="sum over terms of BF16(x) . BF16 weight term, FP32 accumulation, one BF16 result rounding",
                weight_representation=f"{weight_terms} BF16 terms (residual ~2^-{8 * weight_terms} relative)",
                input_gradient="sum over terms of widened BF16 products dY . term, summed in FP32",
                weight_gradient="released BF16 product dY^T . BF16(x) (residual terms detached)",
                cost=f"{weight_terms}x K extent of the forward and input-gradient products")
