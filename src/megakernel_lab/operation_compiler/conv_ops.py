"""Overlapping 3x3 convolution through the common capture/AD/lowering path (Latch release step 10).

Admitted class: ``nn.Conv2d``/``F.conv2d`` with a 3x3 kernel, stride 1, padding 1, dilation 1, one group,
static NCHW shapes, optional bias. Every other overlapping geometry is still refused at capture.

Like the admitted non-overlapping patch rule (``standard_ops.convolution``), this is a capture-time
decomposition below autograd: PyTorch's own ``aten.convolution`` derivative selects ``convolution_backward``,
and both are rewritten into a patch gather plus ordinary matrix products, which the existing matrix bodies,
split-K and epilogue rules lower. There is no CNN generator and no external convolution library call.

The one new operation is the gather ``latch_ops::conv3_patches(x, flip)``. For ``x`` of shape (N, C, H, W) it
returns the (9C, N*H*W) patch matrix

    P[(dh, dw, c), (n, h, w)] = x[n, c, h + s*(dh - 1), w + s*(dw - 1)]    (zero outside the image)

with ``s = 1`` (``flip=False``) or ``s = -1`` (``flip=True``). Rows are tap-major, (kh, kw, c): every block of
consecutive channel rows belongs to one tap, so a 64-row matrix tile is one spatially shifted slab of the
source image (the form a direct/implicit operand load can address). The weight operand is the small reordered
copy ``weight.permute(0, 2, 3, 1).reshape(O, 9C)``. Columns are (n, h, w): the product output
(O, N*H*W) is channel-major (CNHW), and a contiguous NCHW result is one permuting copy that the elementwise
fuser can merge with the bias/activation consumer.

Decomposition (all products are single unbatched GEMMs, so they remain eligible for the warpgroup engines):

    forward   Y^T (O, NHW)  = W (O, 9C)  @ P(x)          (K = 9C)   W[o, (kh,kw,c)]  = weight[o, c, kh, kw]
    dgrad     dX^T (C, NHW) = W' (C, 9O) @ Pflip(dY)     (K = 9O)   W'[c, (kh,kw,o)] = weight[o, c, kh, kw]
    wgrad     dW (O, 9C)    = dY_c (O, NHW) @ P(x)^T     (K = NHW)
    bias      db            = sum over (n, h, w) of dY

The input gradient of a stride-1, padding-1 convolution is the same convolution of dY with the spatially
flipped, channel-transposed kernel, so the overlapping contributions of neighbouring outputs are summed inside
the matrix product's FP32 accumulation instead of by a separate scatter/col2im accumulation. dY_c is the centre
tap of Pflip(dY) (rows (1, 1, o)): a contiguous row block, no extra copy. The backward recomputes P(x) from the saved
input rather than keeping the 9x forward patch matrix alive between the forward and backward passes.

Pixel-major form (LATCH-STEP10-SPEED, ``nhwc``; BF16 operands with C and O multiples of 64). Products run over
dense (pixels, channels) operands P = N*H*W:

    forward   Y (P, O)    = T(x) (P, 9C)    @ W (O, 9C)^T           T: conv3_im2col of the pixel-major input
    dgrad     dX (P, C)   = Tflip(dY) (P, 9O) @ W' (C, 9O)^T
    wgrad     dW (O, 9C)  = dY (P, O)^T     @ T(x) (P, 9C)

``conv3_pixels``/``conv3_unpixels`` convert between NCHW values and pixel-major rows; capture-time channels-last
propagation (ops/channels_last.py) removes them around elementwise neighbourhoods. A flat p5c product reads T
through a TMA im2col map of the dense pixel-major source (zero fill outside the image supplies the padding), so the
9x tap operand, padded copies and interior extraction of the step-10 forms disappear.

Precision: operands keep their dtype. Under an explicit BF16 policy (``ops.precision``), the gather moves
BF16 values exactly and each product accumulates in FP32 with one BF16 result rounding, as any BF16 product.
"""
from __future__ import annotations

import math

import torch

from .ir import CaptureError

aten = torch.ops.aten
TARGET = "latch_ops.conv3_patches.default"
PAD_TARGET = "latch_ops.conv3_pad.default"
TAPS_TARGET = "latch_ops.conv3_taps.default"
INTERIOR_TARGET = "latch_ops.conv3_interior.default"
PIXELS_TARGET = "latch_ops.conv3_pixels.default"
UNPIXELS_TARGET = "latch_ops.conv3_unpixels.default"
IM2COL_TARGET = "latch_ops.conv3_im2col.default"
CMAJOR_TARGET = "latch_ops.conv3_cmajor.default"
TARGETS = (TARGET, PAD_TARGET, TAPS_TARGET, INTERIOR_TARGET, PIXELS_TARGET, UNPIXELS_TARGET, IM2COL_TARGET, CMAJOR_TARGET)
# LATCH-STEP10-SPEED: pixel-major ("nhwc") decomposition. Products run over dense (pixels, channels) operands; the
# 3x3 tap operand is never stored when a p5c product reads it through a TMA im2col map (implicit GEMM, no padding
# copies), and capture-time channels-last propagation (ops/channels_last.py) keeps the surrounding elementwise
# operations in the same pixel-major order. NHWC = False restores the step-10 forms.
NHWC = True
# Capture-time decomposition form: "padded" (tap-shifted operand of a padded channel-major image, which a TMA
# matrix body can read in place), "patches" (materialized unpadded patch matrix) or "auto". Both are exact
# rewrites of the same convolution. "auto" (measured, research/latch_release/HANDOFF.md) uses the padded form
# only where the in-place operand is possible: BF16 operands with input and output channels multiples of 64.
FORM = "auto"
# Lowering choices for the padded form (diagnostics/measurements): read tap operands in place, and give the
# weight-gradient product the existing p5c split-K body.
VIRTUAL = True
SPLITK = True
# Padded-form choices measured on the development block: reuse the forward padded copy of the input in the
# backward pass (memory for one fewer copy) and extract outputs with a dedicated gather instead of a strided view.
REUSE_PAD = True
# LATCH-STEP10-SPEED: weight-gradient split-K over an im2col operand rounds K up to whole slices (zero-filled rows).
PAD_SLICES = True
PAD_EXTENTS = True
# Weight-gradient split-K tile over an im2col operand (128 x 256 halves the dY re-reads; N is padded as needed).
WGRAD_TILE = (128, 128, 2)
WGRAD_STAGES = 6           # stage ring of the weight-gradient split-K body (flat default: 4)
WGRAD_KMAJOR = False       # weight gradients read a channel-major (K-major) copy of dY (A/B option)
FWD_STAGES = None          # stage ring of im2col-A (forward/input-gradient) products (None: the flat default)
FULL_GRID_UNITS = 128      # occupancy recorded for grid-filling conv split-K products (group_rules.ECTA_GRID)
# Split-K partials summed by a grid-wide phase (False: each tile's last-arriving CTA, the P7-C body's own reduction).
SPLITK_PHASE = True


def splitk_reduce_code(*, part, out, dtype, slices, tiles, tile, padded_n, rows, cols):
    """Grid-wide reduction of p5c split-K partials ([slice][tile][256 threads][fragment], kernels/p5c_wg.cuh body):
    each 4-value fragment group is summed over slices 0..S-1 in order (as the last-arriver reduction) and stored at its
    warpgroup-MMA accumulator position (rows r, r + 8; columns c, c + 1), clipped to the true (rows, cols) block."""
    bm, bn, wm = tile
    wn = 2 // wm
    wtm, wtn = bm // wm, bn // wn
    mt, r_ = wtm // 64, wtn // 2
    fr = mt * r_
    total = tiles * 256 * fr
    tn = padded_n // bn
    adds = " ".join(f"{{ const float4 p_ = __ldcg(reinterpret_cast<const float4*>({part} + {s * total}LL + e)); "
                    f"a.x += p_.x; a.y += p_.y; a.z += p_.z; a.w += p_.w; }}" for s in range(1, slices))
    rnd = (lambda v: f"__float2bfloat16_rn({v})") if dtype == "bfloat16" else (lambda v: v)
    return f"""for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < {total // 4}LL; v += (long long)gridDim.x * blockDim.x) {{
  const long long e = v * 4;
  float4 a = __ldcg(reinterpret_cast<const float4*>({part} + e));
  {adds}
  const int tile = (int)(e / {256 * fr}LL), rem = (int)(e % {256 * fr}LL), tid = rem / {fr}, f = rem % {fr};
  const int i = f / {r_}, nb = (f % {r_}) / 4, wg = tid >> 7, warp = (tid >> 5) & 3, lane = tid & 31;
  const int row = (tile / {tn}) * {bm} + {'wg' if wm == 2 else '0'} * {wtm} + i * 64 + warp * 16 + (lane >> 2);
  const int col = (tile % {tn}) * {bn} + {'0' if wm == 2 else 'wg'} * {wtn} + nb * 8 + 2 * (lane & 3);
  if (col < {cols}) {{
    if (row < {rows}) {{ {out}[(long long)row * {cols} + col] = {rnd('a.x')}; {out}[(long long)row * {cols} + col + 1] = {rnd('a.y')}; }}
    if (row + 8 < {rows}) {{ {out}[(long long)(row + 8) * {cols} + col] = {rnd('a.z')}; {out}[(long long)(row + 8) * {cols} + col + 1] = {rnd('a.w')}; }}
  }}
}}"""
# Tiles for im2col-operand products the flat admission refuses (first dividing tile; bkc-free N-major B excluded).
CONV_TILES = ((128, 256, 2), (128, 128, 2), (128, 192, 2), (128, 64, 2))
# Forward/input-gradient (im2col A) products: explicit p5c tile (None: flat selection, the warp-specialized 128-row
# body). A 256-row tile with two 128-row warpgroups re-reads the weight operand half as often from L2.
FWD_TILE = None
INTERIOR_OP = True
PIXELS_PREFETCH = True     # (gpu-20: input interval 22.0 -> 18.7 us) LATCH-STEP10-SPEED: the pixel-major transpose loads its next tile before storing the current
PAD_ALIGN = 1024    # padded columns: whole 64-wide K stages in 16 equal split-K slices


def geometry(input, weight, stride, padding, dilation, transposed, output_padding, groups):
    """(n, c, h, w, o) for the admitted 3x3/stride-1/padding-1 class, else None."""
    if input.dim() != 4 or weight.dim() != 4 or tuple(int(v) for v in weight.shape[2:]) != (3, 3):
        return None
    stride, padding, dilation = (tuple(int(v) for v in x) for x in (stride, padding, dilation))
    if (transposed or int(groups) != 1 or stride != (1, 1) or padding != (1, 1) or dilation != (1, 1)
            or any(int(v) for v in output_padding)):
        return None
    n, c, h, w = (int(v) for v in input.shape)
    if int(weight.shape[1]) != c:
        raise CaptureError("CONVOLUTION", f"weight input channels {int(weight.shape[1])} != input channels {c}")
    return n, c, h, w, int(weight.shape[0])


def _reference_patches(x, flip):
    n, c, h, w = x.shape
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1))
    taps = []
    for dh in range(3):
        for dw in range(3):
            oh, ow = (2 - dh, 2 - dw) if flip else (dh, dw)
            taps.append(padded[:, :, oh:oh + h, ow:ow + w])
    # (9, N, C, H, W) -> (9, C, N, H, W) -> (9C, NHW)
    return torch.stack(taps).permute(0, 2, 1, 3, 4).reshape(c * 9, n * h * w)


@torch.library.custom_op("latch_ops::conv3_patches", mutates_args=())
def conv3_patches(x: torch.Tensor, flip: bool) -> torch.Tensor:
    """3x3 patch matrix (9C, N*H*W) of an NCHW tensor with zero padding 1 (see module docstring)."""
    if x.dim() != 4:
        raise ValueError("conv3_patches expects an NCHW tensor")
    return _reference_patches(x, flip).contiguous()


@conv3_patches.register_fake
def _fake_patches(x, flip):
    n, c, h, w = x.shape
    return x.new_empty((c * 9, n * h * w))


def _nchw(product, n, channels, h, w):
    """(channels, N*H*W) product -> dense NCHW (one permuting copy; fusable with an elementwise consumer)."""
    return product.reshape(channels, n, h, w).permute(1, 0, 2, 3).contiguous()


def padded_geometry(n, h, w):
    """(row width Wp, columns Lp, margin M, stored columns Lm) of the padded layout.

    Each image is an (H + 2) x Wp frame with a zero ring; Wp = W + 2 rounded up to 8 elements so that whole-row
    tap shifts keep 16-byte alignment. Lp = N frames rounded up to PAD_ALIGN; M zero columns on both sides keep
    every tap read inside the stored row, at a non-negative, 16-byte-aligned coordinate (a TMA requirement found
    on the device: an unaligned/negative box start faults).
    """
    width = -(-(w + 2) // 8) * 8
    columns = -(-n * (h + 2) * width // PAD_ALIGN) * PAD_ALIGN
    margin = -(-width // 64) * 64
    return width, columns, margin, columns + 2 * margin


def _padded_flat(x):
    """(C, Lp) padded frames: x at row h+1, column w+1 of its image's frame, zero elsewhere."""
    n, c, h, w = x.shape
    width, columns, _, _ = padded_geometry(n, h, w)
    framed = torch.nn.functional.pad(x, (1, width - w - 1, 1, 1)).permute(1, 0, 2, 3).reshape(c, n * (h + 2) * width)
    return torch.nn.functional.pad(framed, (0, columns - framed.shape[1]))


@torch.library.custom_op("latch_ops::conv3_pad", mutates_args=())
def conv3_pad(x: torch.Tensor) -> torch.Tensor:
    """(3C, Lm): three column-shifted copies of the padded frames with zero margins.

    Row block d (d = 0, 1, 2) holds padded[c, j - M + d - 1] at stored column j, so the operand of tap (kh, kw)
    is block kw read at column offset M + (kh - 1) * Wp: whole rows only. Block 1 at columns [M, M + Lp) is the
    padded image itself.
    """
    n, c, h, w = x.shape
    _, columns, margin, stored = padded_geometry(n, h, w)
    flat = _padded_flat(x)
    blocks = []
    for d in range(3):
        block = x.new_zeros((c, stored))
        lo, hi = max(0, margin - d + 1), min(stored, margin - d + 1 + columns)
        block[:, lo:hi] = flat[:, lo - (margin - d + 1):hi - (margin - d + 1)]
        blocks.append(block)
    return torch.cat(blocks).contiguous()


@conv3_pad.register_fake
def _fake_pad(x):
    n, c, h, w = x.shape
    return x.new_empty((3 * c, padded_geometry(n, h, w)[3]))


def tap_source(tap, flip):
    """(row shift kh - 1, shifted block kw) of a tap; flipped taps use the opposite neighbours."""
    kh, kw = divmod(tap, 3)
    return ((1 - kh, 2 - kw) if flip else (kh - 1, kw))


@torch.library.custom_op("latch_ops::conv3_taps", mutates_args=())
def conv3_taps(padded: torch.Tensor, width: int, margin: int, flip: bool) -> torch.Tensor:
    """(9C, Lp): row (tap, c), column q holds block kw of ``padded`` at stored column M + q + (kh - 1) * Wp.

    At interior columns this is the 3x3 patch matrix (flipped taps for ``flip``); ring/tail columns hold values
    that every consumer discards (forward/dgrad) or multiplies by an exactly zero ring (wgrad).
    """
    rows3, stored = padded.shape
    c, columns = rows3 // 3, stored - 2 * margin
    rows = []
    for tap in range(9):
        shift, block = tap_source(tap, flip)
        start = margin + shift * width
        rows.append(padded[block * c:(block + 1) * c, start:start + columns])
    return torch.cat(rows).contiguous()


@conv3_taps.register_fake
def _fake_taps(padded, width, margin, flip):
    rows3, stored = padded.shape
    return padded.new_empty((3 * rows3, stored - 2 * margin))


@torch.library.custom_op("latch_ops::conv3_interior", mutates_args=())
def conv3_interior(product: torch.Tensor, n: int, h: int, w: int) -> torch.Tensor:
    """Dense NCHW interior of a (channels, Lp) padded-grid product."""
    width = padded_geometry(n, h, w)[0]
    framed = product[:, :n * (h + 2) * width].reshape(product.shape[0], n, h + 2, width)
    return framed[:, :, 1:h + 1, 1:w + 1].permute(1, 0, 2, 3).clone(memory_format=torch.contiguous_format)


@conv3_interior.register_fake
def _fake_interior(product, n, h, w):
    return product.new_empty((n, product.shape[0], h, w))


def _interior(product, n, channels, h, w):
    """(channels, Lp) padded-grid product -> dense NCHW interior (one permuting/extracting copy)."""
    if INTERIOR_OP:
        return conv3_interior(product, n, h, w)
    width = padded_geometry(n, h, w)[0]
    framed = product[:, :n * (h + 2) * width].reshape(channels, n, h + 2, width)
    return framed[:, :, 1:h + 1, 1:w + 1].permute(1, 0, 2, 3).contiguous()


# Forward padded copies by input tensor identity, for the backward of the same capture (REUSE_PAD). Entries hold
# the input so an identity is never reused while cached; the table is bounded.
_PADDED = {}


def _padded_input(input):
    entry = _PADDED.get(id(input))
    if REUSE_PAD and entry is not None and entry[0] is input:
        return entry[1]
    padded = conv3_pad(input)
    if REUSE_PAD:
        if len(_PADDED) > 256:
            _PADDED.clear()
        _PADDED[id(input)] = (input, padded)
    return padded


def _padded_convolution(input, matrix, n, c, h, w, o):
    width, _, margin, _ = padded_geometry(n, h, w)
    taps = conv3_taps(_padded_input(input), width, margin, False)            # (9C, Lp), virtual when admitted
    return _interior(torch.mm(matrix, taps), n, o, h, w)


def _padded_backward(grad_output, input, weight, output_mask, n, c, h, w, o):
    grad_input = grad_weight = None
    width, columns, margin, _ = padded_geometry(n, h, w)
    padded_grad = conv3_pad(grad_output)                                    # (3O, Lm)
    if output_mask[0]:
        transposed = weight.permute(1, 2, 3, 0).reshape(c, 9 * o)
        taps = conv3_taps(padded_grad, width, margin, True)
        grad_input = _interior(torch.mm(transposed, taps), n, c, h, w)
    if output_mask[1]:
        centre = padded_grad[o:2 * o, margin:margin + columns]              # dY frames with a zero ring: a view
        taps = conv3_taps(_padded_input(input), width, margin, False)       # forward copy, or recomputed
        grad_weight = torch.mm(centre, taps.t()).reshape(o, 3, 3, c).permute(0, 3, 1, 2).contiguous()
    return grad_input, grad_weight


# ----------------------------------------------------------------------------- pixel-major (nhwc) form
@torch.library.custom_op("latch_ops::conv3_pixels", mutates_args=())
def conv3_pixels(x: torch.Tensor) -> torch.Tensor:
    """(N*H*W, C) pixel-major copy of an (N, C, H, W) tensor (a view when its storage is already channels-last)."""
    n, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(n * h * w, c).clone(memory_format=torch.contiguous_format)


@conv3_pixels.register_fake
def _fake_pixels(x):
    n, c, h, w = x.shape
    return x.new_empty((n * h * w, c))


@torch.library.custom_op("latch_ops::conv3_unpixels", mutates_args=())
def conv3_unpixels(y: torch.Tensor, n: int, h: int, w: int) -> torch.Tensor:
    """(N, C, H, W) values of a pixel-major (N*H*W, C) product (contiguous NCHW result, eager semantics)."""
    return y.reshape(n, h, w, y.shape[1]).permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)


@conv3_unpixels.register_fake
def _fake_unpixels(y, n, h, w):
    return y.new_empty((n, y.shape[1], h, w))


def _reference_im2col(xp, n, h, w, flip):
    c = xp.shape[1]
    image = xp.reshape(n, h, w, c)
    padded = torch.nn.functional.pad(image, (0, 0, 1, 1, 1, 1))
    taps = []
    for kh in range(3):
        for kw in range(3):
            oh, ow = (2 - kh, 2 - kw) if flip else (kh, kw)
            taps.append(padded[:, oh:oh + h, ow:ow + w, :])
    return torch.cat(taps, dim=3).reshape(n * h * w, 9 * c)


@torch.library.custom_op("latch_ops::conv3_im2col", mutates_args=())
def conv3_im2col(xp: torch.Tensor, n: int, h: int, w: int, flip: bool) -> torch.Tensor:
    """(N*H*W, 9C) tap operand of a pixel-major image: column (kh, kw, c) of pixel (n, y, x) is
    image[n, y + kh - 1, x + kw - 1, c] (flip: y + 1 - kh, x + 1 - kw), zero outside the image."""
    return _reference_im2col(xp, n, h, w, flip).contiguous()


@conv3_im2col.register_fake
def _fake_im2col(xp, n, h, w, flip):
    return xp.new_empty((xp.shape[0], 9 * xp.shape[1]))


@torch.library.custom_op("latch_ops::conv3_cmajor", mutates_args=())
def conv3_cmajor(y: torch.Tensor) -> torch.Tensor:
    """(C, P) channel-major copy of a pixel-major (P, C) matrix."""
    return y.t().clone(memory_format=torch.contiguous_format)


@conv3_cmajor.register_fake
def _fake_cmajor(y):
    return y.new_empty((y.shape[1], y.shape[0]))


# Forward pixel-major copies by input tensor identity, reused by the backward of the same capture (as _PADDED).
_PIXELS = {}


def _pixels_input(input):
    entry = _PIXELS.get(id(input))
    if entry is not None and entry[0] is input:
        return entry[1]
    pixels = conv3_pixels(input)
    if len(_PIXELS) > 256:
        _PIXELS.clear()
    _PIXELS[id(input)] = (input, pixels)
    return pixels


def _nhwc_convolution(input, weight, bias, n, c, h, w, o):
    matrix = weight.permute(0, 2, 3, 1).reshape(o, 9 * c)                       # (O, 9C), tap-major columns
    product = torch.mm(conv3_im2col(_pixels_input(input), n, h, w, False), matrix.t())   # (P, O)
    out = conv3_unpixels(product, n, h, w)
    return out if bias is None else out + bias.reshape(1, o, 1, 1)


def _nhwc_backward(grad_output, input, weight, output_mask, n, c, h, w, o):
    grad_input = grad_weight = grad_bias = None
    pixels = conv3_pixels(grad_output)                                          # (P, O)
    if output_mask[0]:
        transposed = weight.permute(1, 2, 3, 0).reshape(c, 9 * o)                # (C, 9O): flipped taps below
        grad_input = conv3_unpixels(torch.mm(conv3_im2col(pixels, n, h, w, True), transposed.t()), n, h, w)
    if output_mask[1]:
        # WGRAD_KMAJOR: a channel-major copy of dY makes the weight-gradient A operand K-major (an A/B option).
        rows = conv3_cmajor(pixels) if WGRAD_KMAJOR else pixels.t()
        grad = torch.mm(rows, conv3_im2col(_pixels_input(input), n, h, w, False))    # (O, 9C)
        grad_weight = grad.reshape(o, 3, 3, c).permute(0, 3, 1, 2).contiguous()
    if output_mask[2]:
        grad_bias = grad_output.sum(dim=(0, 2, 3))
    return grad_input, grad_weight, grad_bias


def form(input, weight):
    """The decomposition used for this convolution (fixed by dtype and channel counts, the same in backward)."""
    if FORM != "auto":
        return FORM
    n, c, h, w = (int(v) for v in input.shape)
    o = int(weight.shape[0])
    if (NHWC and input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and c % 64 == 0 and o % 64 == 0):
        # The im2col operand reads 64-channel slabs of one tap; products tile 64-multiples of channels.
        return "nhwc"
    # The in-place operand needs the flat p5c body: 128-row tiles over O (forward/weight gradient) and C (input
    # gradient), and 9C weight-gradient columns. Padding inflates matrix work by the frame ratio; beyond 2x the
    # materialized patch form is kept.
    n, c, h, w = (int(v) for v in input.shape)
    o = int(weight.shape[0])
    width, _, _, _ = padded_geometry(n, h, w)
    ratio = (h + 2) * width / (h * w)
    eligible = c % 128 == 0 and o % 128 == 0 and ratio <= 2.0
    return "padded" if input.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and eligible else "patches"


def convolution(input, weight, bias, n, c, h, w, o):
    if form(input, weight) == "nhwc":
        return _nhwc_convolution(input, weight, bias, n, c, h, w, o)
    if form(input, weight) == "padded":
        matrix = weight.permute(0, 2, 3, 1).reshape(o, 9 * c)
        out = _padded_convolution(input, matrix, n, c, h, w, o)
        return out if bias is None else out + bias.reshape(1, o, 1, 1)
    patches = conv3_patches(input, False)                          # (9C, NHW)
    matrix = weight.permute(0, 2, 3, 1).reshape(o, 9 * c)          # (O, 9C), tap-major; small copy
    out = _nchw(torch.mm(matrix, patches), n, o, h, w)
    # Bias after the permuting copy: the product's only consumer is then the copy (a write-through candidate)
    # and the bias joins the following elementwise chain.
    return out if bias is None else out + bias.reshape(1, o, 1, 1)


def convolution_backward(grad_output, input, weight, output_mask, n, c, h, w, o):
    grad_input = grad_weight = grad_bias = None
    if form(input, weight) == "nhwc":
        return _nhwc_backward(grad_output, input, weight, output_mask, n, c, h, w, o)
    if form(input, weight) == "padded" and (output_mask[0] or output_mask[1]):
        grad_input, grad_weight = _padded_backward(grad_output, input, weight, output_mask, n, c, h, w, o)
        if output_mask[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias
    if output_mask[0] or output_mask[1]:
        flipped = conv3_patches(grad_output, True)                 # (9O, NHW)
    if output_mask[0]:
        transposed = weight.permute(1, 2, 3, 0).reshape(c, 9 * o)  # (C, 9O), tap-major; small copy
        grad_input = _nchw(torch.mm(transposed, flipped), n, c, h, w)
    if output_mask[1]:
        centre = flipped[4 * o:5 * o]                              # dY in (O, NHW) order: a row-block view
        patches = conv3_patches(input, False)                      # recomputed from the saved input
        grad_weight = torch.mm(centre, patches.t()).reshape(o, 3, 3, c).permute(0, 3, 1, 2).contiguous()
    if output_mask[2]:
        grad_bias = grad_output.sum(dim=(0, 2, 3))
    return grad_input, grad_weight, grad_bias


# ----------------------------------------------------------------------------- lowering
def emit(target, args, kwargs, name, shape, layouts, dtypes, names):
    """Grid-stride gather phases for the three conv operations; None for any other target."""
    if target == PAD_TARGET:
        return _emit_pad(args, name, shape, layouts, dtypes, names)
    if target == TAPS_TARGET:
        return _emit_taps(args, kwargs, name, shape, layouts, dtypes, names)
    if target == TARGET:
        return _emit_patches(args, kwargs, name, shape, layouts, dtypes, names)
    if target == INTERIOR_TARGET:
        return _emit_interior(args, kwargs, name, shape, layouts, dtypes, names)
    if target == PIXELS_TARGET:
        return _emit_pixels(args, name, shape, layouts, dtypes, names)
    if target == UNPIXELS_TARGET:
        return _emit_unpixels(args, kwargs, name, shape, layouts, dtypes, names)
    if target == IM2COL_TARGET:
        return _emit_im2col(args, kwargs, name, shape, layouts, dtypes, names)
    if target == CMAJOR_TARGET:
        return _emit_cmajor(args, name, shape, layouts, dtypes, names)
    return None


def _emit_cmajor(args, name, shape, layouts, dtypes, names):
    """(C, P) copy of a (P, C) matrix: 64 x 64 shared-memory tiles (16 loads in flight per thread)."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    if len(layout.shape) != 2 or shape != (layout.shape[1], layout.shape[0]):
        raise LoweringError(f"conv3_cmajor result {shape} disagrees with its operand")
    rows, cols = layout.shape
    sr, sc = layout.strides
    rb, cb = -(-rows // 64), -(-cols // 64)
    x, out = _ident(source), _ident(name)
    ctype = "float" if dtypes[name] == "float32" else "__nv_bfloat16"
    load = f"(float){x}" if dtypes[name] == "float32" else f"__bfloat162float({x}"
    close = "" if dtypes[name] == "float32" else ")"
    store = "v" if dtypes[name] == "float32" else "__float2bfloat16_rn(v)"
    code = f"""for (long long t = blockIdx.x; t < {rb * cb}LL; t += gridDim.x) {{
  float* T = smem;
  const int r0 = (int)(t / {cb}) * 64, c0 = (int)(t % {cb}) * 64;
  float v_[16];
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x, r = r0 + (e >> 6), c = c0 + (e & 63);
    v_[u] = (r < {rows} && c < {cols}) ? {load}[(long long)r * {sr}LL + (long long)c * {sc}LL]{close} : 0.0f;
  }}
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x;
    T[(e >> 6) * 65 + (e & 63)] = v_[u];
  }}
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x, c = c0 + (e >> 6), r = r0 + (e & 63);
    if (r < {rows} && c < {cols}) {{ const float v = T[(e & 63) * 65 + (e >> 6)]; {out}[(long long)c * {rows}LL + r] = ({ctype}){store}; }}
  }}
  __syncthreads();
}}"""
    return "gather", code, {"shared_bytes": 64 * 65 * 4}, ""



def _emit_pixels(args, name, shape, layouts, dtypes, names):
    """(P, C) pixel-major copy of an any-strided (N, C, H, W) operand (an operand that channels-last propagation
    could not supply, e.g. an NCHW graph input): 64 x 64 shared-memory tiles of each image's (C, H*W) plane, read
    along pixels and written along channels (both coalesced for NCHW sources). Values are copied exactly."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    if len(layout.shape) != 4:
        raise LoweringError("conv3_pixels requires an NCHW operand")
    n, c, h, w = layout.shape
    if shape != (n * h * w, c):
        raise LoweringError(f"conv3_pixels result {shape} disagrees with its operand")
    sn, sc, sh, sw = layout.strides
    hw, cb, pb = h * w, -(-c // 64), -(-(h * w) // 64)
    x, out = _ident(source), _ident(name)
    ctype = "float" if dtypes[name] == "float32" else "__nv_bfloat16"
    load = f"(float){x}" if dtypes[name] == "float32" else f"__bfloat162float({x}"
    close = "" if dtypes[name] == "float32" else ")"
    store = "v" if dtypes[name] == "float32" else "__float2bfloat16_rn(v)"
    pix = (f"(long long)p" if (sw == 1 and sh == w) else f"(long long)(p / {w}) * {sh}LL + (long long)(p % {w}) * {sw}LL")
    # 64 x 64 tiles; each thread issues its 16 loads before the shared-memory exchange (memory-level parallelism).
    if PIXELS_PREFETCH:
        # LATCH-STEP10-SPEED: the next tile's 16 loads are issued (into the registers just staged in shared memory)
        # before the current tile's stores, so a CTA keeps loads in flight while it writes. Same copies.
        loads = lambda tt: (f"""    const long long b = {tt} / {cb * pb}, rest = {tt} % {cb * pb};
    const int c0 = (int)(rest / {pb}) * 64, p0 = (int)(rest % {pb}) * 64;
#pragma unroll
    for (int u = 0; u < 16; ++u) {{
      const int e = u * 256 + (int)threadIdx.x, r = e >> 6, q = e & 63, ch = c0 + r, p = p0 + q;
      v_[u] = (ch < {c} && p < {hw}) ? {load}[b * {sn}LL + (long long)ch * {sc}LL + {pix}]{close} : 0.0f;
    }}""")
        code = f"""{{
  float v_[16];
  if ((long long)blockIdx.x < {n * cb * pb}LL) {{
{loads('(long long)blockIdx.x')}
  }}
  for (long long t = blockIdx.x; t < {n * cb * pb}LL; t += gridDim.x) {{
    float* T = smem;
    const long long b = t / {cb * pb}, rest = t % {cb * pb};
    const int c0 = (int)(rest / {pb}) * 64, p0 = (int)(rest % {pb}) * 64;
#pragma unroll
    for (int u = 0; u < 16; ++u) {{
      const int e = u * 256 + (int)threadIdx.x;
      T[(e >> 6) * 65 + (e & 63)] = v_[u];
    }}
    __syncthreads();
    if (t + gridDim.x < {n * cb * pb}LL) {{
{loads('(t + gridDim.x)')}
    }}
#pragma unroll
    for (int u = 0; u < 16; ++u) {{
      const int e = u * 256 + (int)threadIdx.x, r = e >> 6, q = e & 63, p = p0 + r, ch = c0 + q;
      if (ch < {c} && p < {hw}) {{ const float v = T[q * 65 + r]; const long long i = (b * {hw}LL + p) * {c}LL + ch;
        {out}[i] = ({ctype}){store}; /*PIXELS_EPI*/ }}
    }}
    __syncthreads();
  }}
}}"""
        return "gather", code, {"shared_bytes": 64 * 65 * 4}, ""
    # 64 x 64 tiles; each thread issues its 16 loads before the shared-memory exchange (memory-level parallelism).
    code = f"""for (long long t = blockIdx.x; t < {n * cb * pb}LL; t += gridDim.x) {{
  float* T = smem;
  const long long b = t / {cb * pb}, rest = t % {cb * pb};
  const int c0 = (int)(rest / {pb}) * 64, p0 = (int)(rest % {pb}) * 64;
  float v_[16];
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x, r = e >> 6, q = e & 63, ch = c0 + r, p = p0 + q;
    v_[u] = (ch < {c} && p < {hw}) ? {load}[b * {sn}LL + (long long)ch * {sc}LL + {pix}]{close} : 0.0f;
  }}
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x;
    T[(e >> 6) * 65 + (e & 63)] = v_[u];
  }}
  __syncthreads();
#pragma unroll
  for (int u = 0; u < 16; ++u) {{
    const int e = u * 256 + (int)threadIdx.x, r = e >> 6, q = e & 63, p = p0 + r, ch = c0 + q;
    if (ch < {c} && p < {hw}) {{ const float v = T[q * 65 + r]; const long long i = (b * {hw}LL + p) * {c}LL + ch;
      {out}[i] = ({ctype}){store}; /*PIXELS_EPI*/ }}
  }}
  __syncthreads();
}}"""
    return "gather", code, {"shared_bytes": 64 * 65 * 4}, ""


def unpixels_parameters(args, kwargs):
    return tuple(int(args[1 + i]) if len(args) > 1 + i else int(kwargs[k]) for i, k in enumerate("nhw"))


def _emit_unpixels(args, kwargs, name, shape, layouts, dtypes, names):
    """(N, C, H, W) contiguous copy of a pixel-major (P, C) operand (when channels-last propagation did not apply)."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    n, h, w = unpixels_parameters(args, kwargs)
    if len(layout.shape) != 2 or layout.shape[0] != n * h * w or shape != (n, layout.shape[1], h, w):
        raise LoweringError(f"conv3_unpixels operand {layout.shape} disagrees with result {shape}")
    c = layout.shape[1]
    sp, sc = layout.strides
    y = _ident(source)
    prelude = f"  const long long b = (long long)row / {c}LL, ch = (long long)row % {c}LL;"

    def element(e):
        return f"{y}[(b * {h * w}LL + (long long)(q + {e}u)) * {sp}LL + ch * {sc}LL]"
    return "gather", _gather_loop(math.prod(shape), h * w, dtypes[name], _ident(name), _Element(prelude, element)), {}, ""


def im2col_parameters(args, kwargs):
    names = ("n", "h", "w", "flip")
    values = [args[1 + i] if len(args) > 1 + i else kwargs[k] for i, k in enumerate(names)]
    return int(values[0]), int(values[1]), int(values[2]), bool(values[3])


def _emit_im2col(args, kwargs, name, shape, layouts, dtypes, names):
    """Materialized (P, 9C) tap operand (only for a reader that does not read it through an im2col map)."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    n, h, w, flip = im2col_parameters(args, kwargs)
    if len(layout.shape) != 2 or layout.shape[0] != n * h * w or shape != (n * h * w, 9 * layout.shape[1]):
        raise LoweringError(f"conv3_im2col operand {layout.shape} disagrees with result {shape}")
    c = layout.shape[1]
    sp, sc = layout.strides
    x, zero = _ident(source), _zero(dtypes[name])
    sign = "-" if flip else "+"
    prelude = (f"  const long long pix = (long long)row, b = pix / {h * w}LL, yx = pix % {h * w}LL;\n"
               f"  const int y0 = (int)(yx / {w}LL), x0 = (int)(yx % {w}LL);")

    def element(e):
        col = f"(q + {e}u)"
        return (f"([&]{{ const int tap = (int)({col} / {c}u), ch = (int)({col} % {c}u); "
                f"const int iy = y0 {sign} (tap / 3 - 1), ix = x0 {sign} (tap % 3 - 1); "
                f"return (iy >= 0 && iy < {h} && ix >= 0 && ix < {w}) ? "
                f"{x}[(b * {h * w}LL + (long long)iy * {w}LL + ix) * {sp}LL + (long long)ch * {sc}LL] : {zero}; }}())")
    return "gather", _gather_loop(math.prod(shape), shape[1], dtypes[name], _ident(name), _Element(prelude, element)), {}, ""


def _gather_loop(numel, columns, dtype, out, element):
    """Each thread writes VEC consecutive outputs of one row with one vector store.

    ``element(e)`` returns C++ for output column ``q + e`` of row ``row`` (with ``IT`` the index type).
    32-bit unsigned index arithmetic is used when every index fits.
    """
    vec = 4 if columns % 4 == 0 else 1
    it = "unsigned" if numel < 2 ** 31 else "long long"
    ctype = "float" if dtype == "float32" else "__nv_bfloat16"
    lines = [f"for ({it} v = ({it})blockIdx.x * blockDim.x + threadIdx.x; v < ({it}){numel // vec}; "
             f"v += ({it})gridDim.x * blockDim.x) {{",
             f"  const {it} i = v * {vec}, row = i / ({it}){columns}, q = i - row * ({it}){columns};"]
    lines.append(element.prelude)
    if vec == 1:
        lines.append(f"  {out}[i] = {element(0)};")
    else:
        lines.append(f"  alignas(16) {ctype} r[4];")
        lines.extend(f"  r[{e}] = {element(e)};" for e in range(4))
        store = "float4" if dtype == "float32" else "uint2"
        lines.append(f"  *reinterpret_cast<{store}*>({out} + i) = *reinterpret_cast<const {store}*>(r);")
    lines.append("}")
    return "\n".join(lines)


class _Element:
    def __init__(self, prelude, body):
        self.prelude, self.body = prelude, body

    def __call__(self, e):
        return self.body(e)


def _zero(dtype):
    return "0.0f" if dtype == "float32" else "__float2bfloat16_rn(0.0f)"


def _check(source, name, layouts, dtypes):
    from .lower import LoweringError
    if source is None:
        raise LoweringError("convolution gathers require a tensor operand")
    if dtypes[source] != dtypes[name] or dtypes[name] not in {"float32", "bfloat16"}:
        raise LoweringError("convolution gathers require matching FP32 or BF16 operand/result")
    return layouts[source]


def _emit_patches(args, kwargs, name, shape, layouts, dtypes, names):
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    flip = bool(args[1]) if len(args) > 1 else bool(kwargs.get("flip", False))
    if len(layout.shape) != 4:
        raise LoweringError("conv3_patches requires an NCHW operand")
    n, c, h, w = layout.shape
    if shape != (9 * c, n * h * w):
        raise LoweringError(f"conv3_patches result shape {shape} != {(9 * c, n * h * w)}")
    sn, sc, sh, sw = layout.strides
    sign = "-" if flip else "+"
    x, zero = _ident(source), _zero(dtypes[name])     # view pointers already include their element offset
    prelude = (f"  const int tap = (int)(row / {c}u), ch = (int)(row % {c}u);\n"
               f"  const int dh = {sign}(tap / 3 - 1), dw = {sign}(tap % 3 - 1);")

    def element(e):
        col = f"(q + {e}u)"
        return (f"([&]{{ const int b = (int)({col} / {h * w}u), pix = (int)({col} % {h * w}u); "
                f"const int ih = pix / {w} + dh, iw = pix % {w} + dw; "
                f"return (ih >= 0 && ih < {h} && iw >= 0 && iw < {w}) ? "
                f"{x}[(long long)b * {sn}LL + (long long)ch * {sc}LL + (long long)ih * {sh}LL + (long long)iw * {sw}LL] "
                f": {zero}; }}())")
    return "gather", _gather_loop(math.prod(shape), shape[1], dtypes[name], _ident(name), _Element(prelude, element)), {}, ""


def _emit_pad(args, name, shape, layouts, dtypes, names):
    """Three shifted padded copies; each thread writes 8 consecutive stored columns of one row (16/32 bytes),
    walking the (image, frame row, column) position incrementally instead of dividing per element."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    if len(layout.shape) != 4:
        raise LoweringError("conv3_pad requires an NCHW operand")
    n, c, h, w = layout.shape
    width, columns, margin, stored = padded_geometry(n, h, w)
    if shape != (3 * c, stored):
        raise LoweringError(f"conv3_pad result {shape} disagrees with its operand")
    frame, numel = (h + 2) * width, math.prod(shape)
    sn, sc, sh, sw = layout.strides
    extent = max(sn * (n - 1) + sc * (c - 1) + sh * (h - 1) + sw * (w - 1) + 1, numel)
    it = "int" if extent < 2 ** 31 and numel < 2 ** 31 else "long long"
    x, out, zero = _ident(source), _ident(name), _zero(dtypes[name])
    ctype = "float" if dtypes[name] == "float32" else "__nv_bfloat16"
    groups = numel // 8                                             # stored % 8 == 0
    stores = ("  *reinterpret_cast<float4*>(o) = *reinterpret_cast<const float4*>(r8);\n"
              "  *reinterpret_cast<float4*>(o + 4) = *reinterpret_cast<const float4*>(r8 + 4);"
              if dtypes[name] == "float32" else
              "  *reinterpret_cast<uint4*>(o) = *reinterpret_cast<const uint4*>(r8);")
    code = f"""for ({it} v = ({it})blockIdx.x * blockDim.x + threadIdx.x; v < ({it}){groups}; v += ({it})gridDim.x * blockDim.x) {{
  const {it} i = v * 8, row = i / {stored}, j = i - row * {stored};
  const {it} block = row / {c}, ch = row - block * {c};
  {it} src = j - {margin} + block - 1;
  {it} b = src >= 0 ? src / {frame} : -((-src + {frame} - 1) / {frame});
  {it} f = src - b * {frame}, r = f / {width}, col = f - r * {width};
  alignas(16) {ctype} r8[8];
  const {ctype}* xc = {x} + ch * ({it}){sc};
#pragma unroll
  for (int e = 0; e < 8; ++e) {{
    r8[e] = (b >= 0 && b < {n} && r >= 1 && r <= {h} && col >= 1 && col <= {w})
        ? xc[b * ({it}){sn} + (r - 1) * ({it}){sh} + (col - 1) * ({it}){sw}] : {zero};
    if (++col == {width}) {{ col = 0; if (++r == {h + 2}) {{ r = 0; ++b; }} }}
  }}
  {ctype}* o = {out} + i;
{stores}
}}"""
    return "gather", code, {}, ""


def _emit_interior(args, kwargs, name, shape, layouts, dtypes, names):
    """Dense NCHW interior of a padded-grid product: 4 consecutive pixels of one image row per thread."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    n, h, w = (int(args[1 + i]) if len(args) > 1 + i else int(kwargs[k]) for i, k in enumerate("nhw"))
    channels = layout.shape[0]
    width, columns, _, _ = padded_geometry(n, h, w)
    if (len(layout.shape) != 2 or layout.shape[1] != columns or layout.strides[1] != 1
            or shape != (n, channels, h, w)):
        raise LoweringError(f"conv3_interior operand {layout.shape} disagrees with result {shape}")
    stride, frame, numel = layout.strides[0], (h + 2) * width, math.prod(shape)
    vec = 4 if w % 4 == 0 else 1
    it = "int" if numel < 2 ** 31 and stride * channels < 2 ** 31 else "long long"
    x, out = _ident(source), _ident(name)
    ctype = "float" if dtypes[name] == "float32" else "__nv_bfloat16"
    store = ("float4" if dtypes[name] == "float32" else "uint2") if vec == 4 else None
    body = (f"  alignas(16) {ctype} r4[4];\n"
            + "".join(f"  r4[{e}] = src[{e}];\n" for e in range(4))
            + f"  *reinterpret_cast<{store}*>({out} + i) = *reinterpret_cast<const {store}*>(r4);\n"
            if vec == 4 else f"  {out}[i] = src[0];\n")
    code = f"""for ({it} v = ({it})blockIdx.x * blockDim.x + threadIdx.x; v < ({it}){numel // vec}; v += ({it})gridDim.x * blockDim.x) {{
  const {it} i = v * {vec}, row = i / {w}, x0 = i - row * {w};
  const {it} y0 = row % {h}, plane = row / {h}, ch = plane % {channels}, b = plane / {channels};
  const {ctype}* src = {x} + ch * ({it}){stride} + b * ({it}){frame} + (y0 + 1) * ({it}){width} + x0 + 1;
{body}}}"""
    return "gather", code, {}, ""


def taps_parameters(args, kwargs):
    names = ("width", "margin", "flip")
    values = [args[1 + i] if len(args) > 1 + i else kwargs[k] for i, k in enumerate(names)]
    return int(values[0]), int(values[1]), bool(values[2])


def _emit_taps(args, kwargs, name, shape, layouts, dtypes, names):
    """Materialized tap operand (used when no consumer reads it in place)."""
    from .lower import LoweringError, _ident, _ref
    source = _ref(args[0], names)
    layout = _check(source, name, layouts, dtypes)
    if len(layout.shape) != 2 or not layout.contiguous or layout.shape[0] % 3:
        raise LoweringError("conv3_taps requires a contiguous (3C, Lm) padded operand")
    rows3, stored = layout.shape
    c = rows3 // 3
    width, margin, flip = taps_parameters(args, kwargs)
    columns = stored - 2 * margin
    if shape != (9 * c, columns):
        raise LoweringError(f"conv3_taps result {shape} disagrees with its operand")
    sign = "-" if flip else ""
    x = _ident(source)
    prelude = (f"  const int tap = (int)(row / {c}u), ch = (int)(row % {c}u);\n"
               f"  const int kh = tap / 3, kw = tap % 3;\n"
               f"  const int shift = {sign}(kh - 1), block = {'2 - kw' if flip else 'kw'};\n"
               f"  const long long base = (long long)(block * {c} + ch) * {stored}LL + {margin}LL + "
               f"(long long)shift * {width}LL + (long long)q;")

    def element(e):
        return f"{x}[base + {e}]"
    return "gather", _gather_loop(math.prod(shape), columns, dtypes[name], _ident(name), _Element(prelude, element)), {}, ""


class TapsOperands:
    """Lowering helper: tap-shifted operands read in place by a TMA matrix body (narrow hooks in lower.py).

    A ``conv3_taps`` value is deferred at its own position. A BF16 product whose selected body is the flat
    ``p5c`` warpgroup engine reads it as its B operand directly from the padded image (``virtual_b``; see
    kernels/p5c_wg.cuh ``vb_issue``). Any other reader first materializes it with the ordinary gather phase,
    emitted immediately before that reader. A deferred value that no phase materialized has no storage.
    """

    def __init__(self, *, enabled):
        self.enabled = enabled
        self.deferred = {}       # taps root -> (op, refs, args, kwargs)
        self.targets = {}        # taps root -> capture target (step-10 taps or im2col)
        self.virtual = []        # manifest records

    def defer(self, target, name, op, refs, args, kwargs, dtypes):
        if not self.enabled or target not in (TAPS_TARGET, IM2COL_TARGET) or dtypes[name] != "bfloat16":
            return False
        self.deferred[name] = (op, list(refs), list(args), dict(kwargs))
        self.targets[name] = target
        return True

    def virtualize_any(self, spec, a, b, la, lb, refs, layouts, names):
        """Read a deferred im2col operand (A or B) or a step-10 tap operand (B) of a p5c product in place."""
        spec, refs = self.virtualize_im2col(spec, la, lb, refs, layouts, names)
        if spec.get('virtual_a') or spec.get('virtual_b'):
            return spec, refs
        return self.virtualize(spec, b, lb, refs, layouts, names)

    def virtualize_im2col(self, spec, la, lb, refs, layouts, names):
        """LATCH-STEP10-SPEED: an admitted flat p5c product reads a deferred ``conv3_im2col`` operand through a TMA
        im2col map of its dense pixel-major source: as A (K-major; rows are pixels, K = 9C) for forward and
        input-gradient products, or as B (N-major; columns 9C, K = pixels) for weight-gradient products."""
        from .lower import _ref
        if not VIRTUAL or spec.get('engine') != 'p5c' or spec.get('batch', 1) != 1:
            return spec, refs
        for operand, lay in ((0, la), (1, lb)):
            root = lay.root
            if self.targets.get(root) != IM2COL_TARGET or root not in self.deferred:
                continue
            if operand == 0 and not spec['ak'] or operand == 1 and spec['bkc']:
                continue
            _, _, args, kwargs = self.deferred[root]
            n, h, w, flip = im2col_parameters(args, kwargs)
            source = _ref(args[0], names)
            src = layouts[source]
            if (len(src.shape) != 2 or not src.contiguous or src.offset % 8 or src.shape[0] != n * h * w
                    or src.shape[1] % 64 or lay.offset or not lay.contiguous or lay.shape != layouts[root].shape):
                continue
            channels = src.shape[1]
            k = spec.get('k_extent', spec['k'])
            rows, cols = (spec['m'], k) if operand == 0 else (k, spec.get('n_extent', spec['n']))
            if (rows, cols) != (n * h * w, 9 * channels):
                continue
            key = 'virtual_a' if operand == 0 else 'virtual_b'
            spec = dict(spec, **{'a' if operand == 0 else 'b': source,
                                 key: {'kind': 'im2col', 'root': src.root, 'offset': src.offset, 'channels': channels,
                                       'n': n, 'h': h, 'w': w, 'flip': flip}})
            if operand == 0 and FWD_STAGES and not spec.get('max_stages'):
                spec['max_stages'] = FWD_STAGES
            if (operand == 0 and FWD_TILE and not spec.get('p5c_tile') and not spec.get('splitk')
                    and spec['m'] % FWD_TILE[0] == 0 and spec['n'] % FWD_TILE[1] == 0):
                spec['p5c_tile'] = tuple(FWD_TILE)
            refs = [source if (r in layouts and layouts[r].root == root) else r for r in refs]
            self.virtual.append({'operand': 'A' if operand == 0 else 'B', 'taps': root, 'source': source,
                                 'channels': channels, 'pixels': n * h * w, 'image': [h, w], 'flip': flip,
                                 'mnk': [spec['m'], spec['n'], spec['k']]})
            return spec, refs
        return spec, refs

    def select(self, spec, reason, a, b, out, la, lb, shape, dtypes, *, p5c_ok, accumulate):
        """Conv-owned body choice for a deferred tap operand the flat p5c admission refused.

        Only the weight-gradient product reaches this: its tile count is (O/128)(9C/128) with a long padded-pixel
        K, which the flat admission's occupancy rule refuses. The alternative is materializing the 9x tap matrix
        for a small-tile body, so this product instead uses the existing p5c split-K body (deterministic slice
        partials, last-arriver reduction) with private counters, reading its operand in place.
        """
        from . import matrix_rules, p5c_matrix
        # LATCH-COMPOSE-EVAL proposed patch: a weight-gradient product that the flat P5C admission accepted but whose
        # flat body underfills the grid (fewer than half the admitted grid in 128 x 128 tiles) is not split by the
        # flat path; offer it the convolution split-K body, keeping the admitted body if that path refuses.
        if ((spec or {}).get("engine") == "p5c" and p5c_ok and not accumulate and len(shape) == 2
                and lb.root in self.deferred and (spec["m"] // 128) * (spec["n"] // 128)
                < p5c_matrix.ADMIT_GRID // 2):
            alt, alt_reason = self.select(None, reason, a, b, out, la, lb, shape, dtypes, p5c_ok=p5c_ok,
                                          accumulate=accumulate)
            if alt is not None:
                return alt, alt_reason + "; flat admission relied on a long-K split the flat path does not apply"
            return spec, reason
        if ((spec or {}).get("engine") == "p5c" or not p5c_ok or accumulate or len(shape) != 2):
            return spec, reason
        if self.targets.get(la.root) == IM2COL_TARGET and la.root in self.deferred:
            # LATCH-STEP10-SPEED: forward/input-gradient product over an im2col A operand that the flat admission
            # refused (e.g. 64 or 192 output channels, which no default tile divides): the plain p5c body with the
            # first conv tile that divides the product (the operand is read in place by virtualize_im2col).
            try:
                base = matrix_rules.bf16_spec(a, b, out, la, lb, shape, engine_ok=False)
            except matrix_rules.MatrixRuleError:
                return spec, reason
            plans = []
            for tile in CONV_TILES:
                try:
                    plans.append(p5c_matrix.tile_plan(base["m"], base["n"], base["k"], base["ak"], base["bkc"], tile,
                                                      grid=p5c_matrix.ADMIT_GRID))
                except p5c_matrix.EngineRefusal:
                    continue
            if not plans:
                return spec, reason
            # The p5c cost model's estimate decides (an underfilled wide tile loses to more, narrower tiles).
            best = min(plans, key=lambda p: (p['estimate_us'], -p['bn']))
            tile = (best['bm'], best['bn'], best['wm'])
            spec = dict(base, engine="p5c", roots=[(la.root, la.offset), (lb.root, lb.offset)], p5c_tile=tile)
            return spec, (f"conv im2col operand: p5c tile {tile}, estimate {best['estimate_us']} us "
                          f"(flat admission refused: {reason})")
        if not SPLITK or lb.root not in self.deferred:
            return spec, reason
        try:
            base = matrix_rules.bf16_spec(a, b, out, la, lb, shape, engine_ok=False)
        except matrix_rules.MatrixRuleError:
            return spec, reason
        m, n, k = base["m"], base["n"], base["k"]
        padded = PAD_SLICES and self.targets.get(lb.root) == IM2COL_TARGET
        if base["batch"] != 1 or (k % 64 and not padded):
            return spec, reason
        mp, np_ = m, n
        bm, bn, wm = WGRAD_TILE if padded else (128, 128, 2)
        if m % bm or n % bn:
            # LATCH-STEP10-SPEED: output extents that the tile does not divide (e.g. 64 or 192 channels) are
            # computed padded: A rows beyond M and im2col columns beyond 9C read as zero (TMA fill), the product
            # lands in a private padded buffer and its (M, N) block is extracted.
            if not (padded and PAD_EXTENTS):
                return spec, reason
            mp, np_ = -(-m // bm) * bm, -(-n // bn) * bn
        tiles, stages = (mp // bm) * (np_ // bn), -(-k // 64)
        if padded:
            # LATCH-STEP10-SPEED: TMA zero-fills operand rows beyond the true K (pixels) extent, so K is rounded up
            # to whole slices: the slicing fills the resident grid (at most one unit per admitted CTA, >= 8 stages
            # per slice) instead of dividing K exactly.
            split = max([s for s in range(1, 65) if -(-stages // s) >= 8 and tiles * s <= p5c_matrix.ADMIT_GRID]
                        or [1])
            kpad = split * -(-stages // split) * 64
        else:
            # Fill the resident grid with (tile, K-slice) units: the largest even slicing with at most one unit per
            # admitted CTA and at least 8 64-wide K stages per slice.
            split = max([s for s in range(1, 65) if stages % s == 0 and stages // s >= 8
                         and tiles * s <= p5c_matrix.ADMIT_GRID] or [1])
            kpad = k
        try:
            p5c_matrix.p5c_plan(mp, np_, kpad, base["ak"], base["bkc"], grid=p5c_matrix.ADMIT_GRID)
        except p5c_matrix.EngineRefusal:
            return spec, reason
        spec = dict(base, engine="p5c", roots=[(la.root, la.offset), (lb.root, lb.offset)], conv_splitk=split,
                    p5c_tile=(bm, bn, wm), max_stages=WGRAD_STAGES,
                    **({"k": kpad, "k_extent": k} if kpad != k else {}),
                    **({"m": mp, "n": np_, "m_extent": m, "n_extent": n} if (mp, np_) != (m, n) else {}))
        return spec, (f"conv tap operand: existing p5c body, split-K {split} with private counters"
                      + (f", K {k} zero-padded to {kpad}" if kpad != k else "")
                      + (f", output {m}x{n} computed padded to {mp}x{np_}" if (mp, np_) != (m, n) else "")
                      + f" (flat admission refused: {reason})")

    def virtualize(self, spec, b, lb, refs, layouts, names):
        """Rewrite an admitted p5c product spec to read its deferred B operand in place."""
        from .lower import _ref
        root = lb.root
        if not VIRTUAL or root not in self.deferred or spec.get("engine") != "p5c" or spec.get("batch", 1) != 1:
            return spec, refs
        op, _, args, kwargs = self.deferred[root]
        padded = _ref(args[0], names)
        rows3, stored = layouts[padded].shape
        channels = rows3 // 3
        width, margin, flip = taps_parameters(args, kwargs)
        columns = stored - 2 * margin
        rows, cols = layouts[root].shape
        bkc = spec["bkc"]
        # B is the taps matrix (K = 9C rows, N = columns) or its transpose (N = 9C rows of output, K = columns).
        expected = ((spec["k"], spec["n"]) if not bkc else (spec["n"], spec["k"]))
        if (channels % 64 or lb.offset or (rows, cols) != expected or (rows, cols) != (9 * channels, columns)
                or not layouts[padded].contiguous or layouts[padded].offset or width % 8 or margin % 8
                or margin < width or stored % 8):
            return spec, refs
        spec = dict(spec, b=padded, virtual_b={"root": layouts[padded].root, "offset": 0, "channels": channels,
                                               "stored": stored, "margin": margin, "width": width, "flip": flip})
        refs = [padded if (r in layouts and layouts[r].root == root) else r for r in refs]
        self.virtual.append({"product_b": b, "taps": root, "padded": padded, "channels": channels,
                             "columns": columns, "width": width, "margin": margin, "flip": flip, "k_major_b": bkc})
        return spec, refs

    def before_phase(self, refs, layouts, emit_phase):
        """Materialize deferred taps read by an ordinary phase (emitted just before it)."""
        for ref in refs:
            root = layouts[ref].root if ref in layouts else None
            if root in self.deferred:
                op, taps_refs, args, kwargs = self.deferred.pop(root)
                emit_phase(root, op, taps_refs, args, kwargs)

    def unmaterialized(self):
        return set(self.deferred)
