"""Explicit CUDA2.9.1 softmax-score top-k ordering, ordinary tensor operations.

For 32 candidates CUDA TensorTopK.cu sbtopk gathers strictly-above-threshold
entries in input-index order, then the earliest threshold ties. Sort.cu uses
SmallBitonicSort (32 padded slots), whose ties are deliberately not stable.
This independently expressed composition reuses argmax, gather and where;
there is no custom kernel or derivative. Default compiler selection is unchanged.
Source (BSD-3): pytorch/pytorch v2.9.1 aten/src/ATen/native/cuda/
TensorTopK.cu, Sort.cu, SortUtils.cuh. The admitted caller supplies finite
nonnegative softmax scores (positive zero is allowed; negative zero is not).
CUDA radix distinguishes signed zero; this comparison composition does not
claim that broader input domain. No eager host-value check is added to capture.
"""
from __future__ import annotations

CONTRACT = "torch_cuda_topk32_bitonic_v1"


def topk32(scores, k=8):
    """Last-axis top-k for finite nonnegative scores without negative zero."""
    if scores.shape[-1] != 32:
        raise ValueError("topk32 requires exactly 32 candidates")
    return topk_small(scores, k)


def topk_small(scores, k=2):
    """The same CUDA gather/bitonic law for 1..32 softmax candidates.

    The output sort pads to 32 slots independently of the input row width.
    Numeric priorities encode the gather's two passes without rounding.
    """
    import torch
    width = scores.shape[-1]
    if scores.dtype != torch.float32 or not 1 <= width <= 32 or type(k) is not int or not 1 <= k <= width:
        raise ValueError("topk_small requires FP32 softmax scores, width1..32, and integer k in[1,width]")
    detached = scores.detach()
    positions = torch.arange(width, device=scores.device)
    masked = detached
    for _ in range(k):
        index = masked.argmax(-1, keepdim=True)
        threshold = detached.gather(-1, index)
        masked = torch.where(positions == index, -3.4028234663852886e38, masked)
    # The threshold is only used for selection. Integer-valued priorities are
    # exactly represented in FP32 and preserve the original gather's two passes.
    priority = torch.where(detached > threshold, 64.0-positions,
                          torch.where(detached == threshold, 32.0-positions, -100.0))
    keys, indices = [], []
    for _ in range(k):
        index = priority.argmax(-1, keepdim=True)
        keys.append(detached.gather(-1, index))
        indices.append(index)
        priority = torch.where(positions == index, -100.0, priority)
    # Invalid padded slots have a data-independent trajectory. Keeping them as
    # None avoids creating arithmetic leaves that the original never compares.
    keys += [None]*(32-k)
    indices += [None]*(32-k)
    stages = []
    for size in (2, 4, 8, 16):
        stride = size//2
        while stride:
            stages.append((size, stride))
            stride //= 2
    stages += [(32, stride) for stride in (16, 8, 4, 2, 1)]
    for size, stride in stages:
        for thread in range(16):
            a = 2*thread-(thread & (stride-1))
            b = a+stride
            direction = bool(thread & (size//2)) if size < 32 else False
            valid_a, valid_b = keys[a] is not None, keys[b] is not None
            if valid_a and valid_b:
                swap = (keys[a] > keys[b]) == direction
                ka,kb,ia,ib = keys[a],keys[b],indices[a],indices[b]
                keys[a],keys[b] = torch.where(swap,kb,ka),torch.where(swap,ka,kb)
                indices[a],indices[b] = torch.where(swap,ib,ia),torch.where(swap,ia,ib)
            else:
                # ((comp && validA) || !validB) is known for unequal validity.
                swap = (not valid_b) if not valid_a else True
                if swap == direction:
                    keys[a],keys[b] = keys[b],keys[a]
                    indices[a],indices[b] = indices[b],indices[a]
    if any(value is None for value in indices[:k]):
        raise AssertionError("bitonic validity network did not retain the selected prefix")
    selected = torch.cat(indices[:k], dim=-1)
    return scores.gather(-1, selected), selected
