"""Boolean attention padding masks (Latch release step 09).

Convention (PyTorch ``F.scaled_dot_product_attention`` boolean masks): ``attn_mask[b, h, i, j] = True`` means
query ``i`` MAY attend to key ``j``. The supported common form is a key-padding validity mask ``valid[batch, keys]``
passed as ``valid[:, None, None, :]`` (shape ``[batch, 1, 1, keys]``). A per-example mask ``[batch, 1, queries,
keys]`` (for example causal & padding; a mask with ``is_causal=True`` is backend-dependent in PyTorch) uses the
same machinery. Mask VALUES are ordinary runtime inputs: a fixed shape compiles once, new masks are staged per update.

Semantics are PyTorch's for boolean masks: forbidden keys receive -inf before the softmax; a query row with no
allowed key produces a zero output and zero gradients (``aten._safe_softmax``, as the PyTorch 2.9 math backend).

Capture writes exactly this definition with ordinary ATen operations (``log`` of the 0/1 mask gives 0/-inf,
expanded over heads, added to the scaled scores, ``_safe_softmax``), so any lowering that does not recognize the
pattern still computes it. ``attention_lower.plan(..., masks=True)`` recognizes it for the existing attention bodies:

* ``_safe_softmax`` is admitted where ``_softmax`` was (group key ``safe``); bodies zero rows whose softmax
  denominator is 0 and store lse = +inf, so the backward recomputation P = exp(x - lse) is exactly 0.
* The head expansion ``expand -> clone -> view`` of the mask is absorbed: bodies read the per-example mask
  ``[batch, 1, queries|1, keys]`` directly with ``mask batch = (batch*heads) / heads`` (group key ``mask_lead``),
  so no [batch*heads, ...] copy and no query x key matrix is materialized for a key-padding mask.

Bodies implementing both: P-OPS-D FP32 tiles (v1), P-OPS3-B short BF16-boundary, P-OPS4-D flat-long v7. Other
bodies refuse these groups (recorded); the ordinary lowering then evaluates the captured definition.
"""
from __future__ import annotations

import math

import torch

SAFE_SOFTMAX = "_safe_softmax"
_VIEW = {"view", "_unsafe_view", "reshape"}
_COPY = {"clone"}


class MaskStrides(tuple):
    """(batch stride, query stride, key stride) of an additive mask, plus ``heads``.

    The mask element of attention batch ``bh`` (= batch * heads + head), query i and key j is at
    ``(bh / heads) * sb + i * si + j * sj``. heads == 1 is the original linear (batch*heads) addressing.
    """
    def __new__(cls, strides, heads=1):
        value = super().__new__(cls, tuple(int(s) for s in strides))
        value.heads = int(heads)
        return value

    @property
    def batches(self):
        """Number of distinct mask batches given the attention batch (set by ``resolve``)."""
        return getattr(self, "_batches", None)


# ------------------------------------------------------------------------------------------------------- capture
def boolean_bias(attn_mask, lead, tq, tk, dtype):
    """Additive rank-3 bias [prod(lead), tq|1, tk] of a boolean SDPA mask (True = allowed): 0 / -inf.

    Raises ValueError when the mask does not broadcast to (*lead, tq, tk). Head/batch broadcast dimensions are
    expanded only over ``lead`` (never over queries), so a key-padding mask stays [batch*heads, 1, keys] when
    materialized by an ordinary lowering, and a body reads the [batch, 1, 1, keys] source directly.
    """
    shape = tuple(attn_mask.shape)
    rank = len(lead) + 2
    if len(shape) > rank:
        raise ValueError(f"mask rank {len(shape)} exceeds the attention rank {rank}")
    full = (1,) * (rank - len(shape)) + shape
    target = tuple(lead) + (tq, tk)
    if any(m not in (1, n) for m, n in zip(full, target)):
        raise ValueError(f"mask shape {shape} does not broadcast to {target}")
    if full[-1] != tk:
        raise ValueError("a boolean padding mask must give one value per key (key extent broadcast is not admitted)")
    mq = full[-2]
    # log(1) = 0 and log(0) = -inf exactly; an ordinary pointwise op (no non-finite scalar constant).
    if all(m == 1 for m in full[:-2]):
        # LATCH-ATTN-RECOG: one mask for every batch/head entry (e.g. a causal [T, T] buffer) is added by
        # broadcasting, as a hand-written additive mask; no head expansion is captured. Same values.
        shared = attn_mask if tuple(attn_mask.shape) == (mq, tk) else attn_mask.reshape(mq, tk)
        return torch.log(shared.to(dtype))
    bias = torch.log(attn_mask.reshape(full).to(dtype))
    return bias.expand(*lead, mq, tk).reshape(math.prod(lead), mq, tk)


def float_bias(attn_mask, lead, tq, tk):
    """Additive mask of a float SDPA mask for scores [prod(lead), tq, tk] (LATCH-ATTN-RECOG).

    A mask shared by every batch/head entry (all leading extents 1) is added as ``[tq|1, tk]`` by broadcasting,
    exactly as a hand-written ``scores + mask`` buffer; a per-entry mask is expanded over ``lead`` like
    ``boolean_bias`` (the attention matcher absorbs that expansion). Raises ValueError when it does not broadcast.
    """
    shape = tuple(attn_mask.shape)
    rank = len(lead) + 2
    if len(shape) > rank:
        raise ValueError(f"mask rank {len(shape)} exceeds the attention rank {rank}")
    full = (1,) * (rank - len(shape)) + shape
    target = tuple(lead) + (tq, tk)
    if any(m not in (1, n) for m, n in zip(full, target)):
        raise ValueError(f"mask shape {shape} does not broadcast to {target}")
    if full[-1] != tk:
        raise ValueError("a float attention mask must give one value per key (key extent broadcast is not admitted)")
    mq = full[-2]
    if all(m == 1 for m in full[:-2]):
        return attn_mask if tuple(attn_mask.shape) == (mq, tk) else attn_mask.reshape(mq, tk)
    return attn_mask.reshape(full).expand(*lead, mq, tk).reshape(math.prod(lead), mq, tk)


def safe_softmax(scores):
    """PyTorch's boolean-mask softmax: rows whose scores are all -inf give zeros (and zero gradients)."""
    return torch.ops.aten._safe_softmax(scores, -1)


# ----------------------------------------------------------------------------------------------------- matching
def look_through(mask, *, base, args_of, uses, shape_of):
    """Absorb the capture's head expansion of a mask. Returns (source, lead, members) or None.

    Pattern: mask = view|_unsafe_view|reshape(clone(expand(source, [*lead, mq, tk]))), each intermediate used once.
    The source keeps its own (possibly broadcast) strides; ``members`` become part of the attention group.
    """
    chain = [mask]
    node = mask
    if base(node) not in _VIEW:
        return None
    node = args_of(node)[0]
    if node is None:
        return None
    if base(node) in _COPY:
        chain.append(node)
        node = args_of(node)[0]
    if node is None or base(node) != "expand":
        return None
    chain.append(node)
    source = args_of(node)[0]
    if source is None:
        return None
    expanded = tuple(shape_of(node))
    if len(expanded) < 3 or tuple(shape_of(mask)) != (math.prod(expanded[:-2]),) + expanded[-2:]:
        return None
    for name in chain[1:]:
        if len(uses(name)) != 1:
            return None
    return source, expanded[:-2], chain


def resolve(layout, lead, bh, t):
    """MaskStrides of a [*lead', mq, tk] source read by attention batch bh = prod(lead), or None.

    Trailing lead dimensions with zero stride (broadcast heads) become the head divisor; the remaining leading
    dimensions must merge into one linear batch stride.
    """
    shape, strides = tuple(layout.shape), tuple(layout.strides)
    rank = len(lead) + 2
    if len(shape) > rank or math.prod(lead) != bh:
        return None
    shape = (1,) * (rank - len(shape)) + shape
    strides = (0,) * (rank - len(strides)) + strides
    if shape[-1] != t or shape[-2] not in (1, t):
        return None
    eff = [0 if extent == 1 else stride for extent, stride in zip(shape[:-2], strides[:-2])]
    if any(extent not in (1, full) for extent, full in zip(shape[:-2], lead)):
        return None
    heads = 1
    k = len(lead)
    while k > 0 and eff[k - 1] == 0:
        heads *= lead[k - 1]
        k -= 1
    prefix = [(lead[d], eff[d]) for d in range(k) if lead[d] != 1]
    sb = 0
    if prefix:
        for (outer, so), (inner, si_) in zip(prefix, prefix[1:]):
            if so != si_ * inner:
                return None
        sb = prefix[-1][1]
    else:
        heads = 1  # every lead dimension is broadcast: one mask batch, linear form with sb = 0
    si = 0 if shape[-2] == 1 else strides[-2]
    result = MaskStrides((sb, si, strides[-1]), heads if sb else 1)
    result._batches = (bh // result.heads) if sb else 1
    return result


def mask_strides(group, layouts, bh, t, legacy):
    """Mask addressing of a matched group: legacy(layout, bh, t, True) for ordinary groups."""
    name = group.get("mask")
    if name is None:
        return None
    lead = group.get("mask_lead")
    if lead is None:
        strides = legacy(layouts[name], bh, t, True)
        if strides is None:
            return None
        result = MaskStrides(strides, 1)
        result._batches = bh if strides[0] else 1
        return result
    return resolve(layouts[name], tuple(lead), bh, t)


def extended(group, strides):
    """True when a group needs a body that implements this module's fields."""
    return bool(group.get("safe")) or (strides is not None and getattr(strides, "heads", 1) != 1)


def fields(group, strides, prefix="A."):
    """C++ field assignments for the extended body semantics; empty for ordinary groups (unchanged code)."""
    parts = []
    if strides is not None and getattr(strides, "heads", 1) != 1:
        parts.append(f"{prefix}m_heads = {strides.heads};")
    if group.get("safe"):
        parts.append(f"{prefix}safe_rows = 1;")
    return " ".join(parts)


def record(group, strides):
    """Decision metadata for manifests and adapters."""
    if not extended(group, strides):
        return {}
    return {"padding_mask": {"safe_rows": bool(group.get("safe")), "mask_heads": getattr(strides, "heads", 1),
                             "mask_batches": getattr(strides, "batches", None), "mask_source": group.get("mask"),
                             "convention": "boolean True = allowed; -inf additive; fully masked rows -> 0"}}
