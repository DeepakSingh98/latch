"""Opt-in attention preparation/layout rules for composed flat-long attention (LATCH-TINYMOE-ATTN).

Selected through the normal build API: ``CapturedModule.build(..., attention_rules="streamed")``.
Omitting it (None) leaves the attention tape, lowering and generated source unchanged.

"streamed" adds structural rules for explicit BF16 SDPA groups lowered to the P-OPS4-D flat-long
body (compose_attention.rewrite_tape). None depends on a model name or shape table, and none changes
an arithmetic operation of the attention body:

* ``tables``: groups reading the same captured causal-constant mask with the same head batch,
  context and mask strides reuse the first group's mask proof tables (mask tile maxima, all-zero flags,
  per-tile row maxima). The tables are a function of the mask bytes only.
* ``kn_item``: v7 groups compute their head's key norms inside each forward item (``forward_item6k``:
  identical ``prepare_unit`` arithmetic and proof). A prepare phase then holds only mask-table units,
  which depend on the constant mask alone; with ``tables`` only the first group has one, so the other
  groups have no prepare phase and no extra dependency interval between their operands and forward.
* ``clones``: an operand or cotangent that is an exact contiguous clone of a contiguous, unaliased,
  single-reader BF16 value is read from that value; the copy is removed.
* ``merged_backward``: the v7 backward becomes the P9-A delta phase (``delta_unit6``, the dQ item's exact
  per-row arithmetic) plus one phase whose dQ (``dq_item6n``) and dK/dV (``kv_item6``) items are
  interleaved by decreasing causal cost. Every item is unchanged, so outputs are bitwise identical; the
  causal dQ and dK/dV critical paths overlap instead of running in two consecutive phases.
* ``pack`` (ops.attention_pack): the captured cast / grouped-query repetition / zero-channel chain that
  materializes each padded operand is replaced by one pack phase per group, reading each operand's
  pre-layout source once and writing the same padded BF16 values plus the key-norm proof table (exact
  prepare_unit arithmetic). Groups with a pack phase use it instead of ``kn_item``. An operand whose FP32
  source is a half-rotation rotary product of a BF16 value (x * cos + cat(-x2, x1) * sin, FP32 tables) is
  computed in the pack from x and the tables with the captured elementwise expression (version 4).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = (None, "streamed")
_RULES = ContextVar("latch_attention_rules", default=None)
_FEATURES = {"streamed": frozenset({"tables", "kn_item", "clones", "merged_backward", "pack"})}
VERSION = 4


def validate(value):
    if value not in RULES:
        raise ValueError(f"attention_rules must be one of {RULES}")
    return value


def current():
    return _RULES.get()


def enabled(feature):
    return feature in _FEATURES.get(_RULES.get(), frozenset())


@contextmanager
def scope(value):
    token = _RULES.set(validate(value))
    try:
        yield
    finally:
        _RULES.reset(token)
