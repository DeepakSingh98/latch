"""Claimed width-32 segmented-attention items (``segmented_attention='claim'``; off by default, program unchanged).

Packed segmented attention (width-32 items: forward, dQ, dK/dV) statically assigns its (segment, head, tile) items
to the 2 x gridDim warpgroups in snake order. Item costs differ strongly (causal tiles, unequal segments), so the
slowest CTA of an attention phase can finish well after the mean CTA. With the rule, each width-32 attention phase
gets one entry-zeroed ticket counter (one of the emitter's ECTAQ words, zeroed at kernel entry and ordered by the
first grid barrier) and its warpgroups take items in ticket order: the next ticket is claimed while the current item
runs (``A.tkt_ahead = 1``), and tickets are decoded in rank-major order (``A.tkt_rank = 1``: every segment's largest
tiles before any segment's smaller ones, from the warpgroup's cached segment prefix) where the body supports it.
Items are independent and write disjoint rows; the set of items, every item's arithmetic and the executed-work
counters are unchanged, only the warpgroup that computes an item changes, so stored values are bitwise those of the
static order. The width-64 bodies and tile-ready row publication are untouched. The header code exists only under
``LATCH_SEG_ATTN_CLAIM``.
"""
from __future__ import annotations

import re

DEFINE = "#define LATCH_SEG_ATTN_CLAIM 1"
_CALL = re.compile(r"latch_segmented::run_(forward|dq|kv)<32,\s*\d+>\(A,")


def enabled():
    from .segmented_attention_rules import active
    return active("claim")


def attention_claim(phase, body, alloc):
    """Bind a per-phase claim counter to one width-32 packed segmented attention call (body unchanged otherwise).

    Returns ``(body, bound)``; ``alloc(1)`` is called (one entry-zeroed counter word) only when a call is bound."""
    if not enabled() or not str(phase.get("kind", "")).startswith("segmented_attention_"):
        return body, False
    if phase.get("kind") == "segmented_attention_prepare":
        return body, False
    calls = _CALL.findall(body)
    if len(calls) != 1 or body.count("latch_segmented::run_") != 1:
        return body, False
    counter = alloc(1)
    bound = body.replace("latch_segmented::run_",
                         f"A.tkt_claim = {counter}; A.tkt_ahead = 1; A.tkt_rank = 1;  "
                         "// attention items claimed by tickets\nlatch_segmented::run_", 1)
    return bound, True
