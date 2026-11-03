"""Opt-in rules for packed segmented attention (``latch_ops.segmented_attention``) of width-32 heads.

Selected through the normal build API: ``CapturedModule.build(..., segmented_attention="claim+rotary")`` (a
'+'-joined list of distinct names from ``RULES``). Omitting it (None) leaves lowering and generated source unchanged:
width-32 heads keep the default two-warpgroup schedule (``segmented_attention_lower.DEFAULT_W32_VARIANT``) and the
three-phase backward. Every rule is dispatched on the recognized segmented-attention operation and its operand
layouts, never on model names.

* ``folded_forward`` -- the width-32 forward runs 128-row items with the softmax scale folded into the base-2
  exponent and Q read by the warpgroup MMA from shared memory (``run_forward<32,152>``). Dynamic shared memory is
  1024 + 2 * 88 KiB per CTA. The folded scale changes FP32 rounding, so it requires ``precision_policy='practical'``.
* ``fused_backward`` -- the width-32 backward is one pass in three steps: ``fused_bprep`` forms the row deltas and
  zeroes an FP32 accumulator ``<dq>__fused_dqacc`` of shape [rows, heads, 32]; ``run_kv<32,168>`` items compute dK/dV,
  form dS and add staged dQ tiles to that accumulator with bulk asynchronous reductions; ``fused_dqcvt_hr`` converts
  the accumulator to dQ. The order of the dQ additions depends on item timing (not bitwise reproducible run to run),
  so it requires ``precision_policy='practical'``.
* ``claim`` -- width-32 attention items are claimed from an entry-zeroed per-phase ticket counter; each CTA claims
  its next ticket one item ahead, in rank order where the body supports it. The same items compute the same bits;
  only which CTA computes an item changes.
* ``rotary`` -- rotary pack/unpack chains around segmented attention (``segmented_rotary``) each become one row pass,
  one thread per (row, head), with paired FP32 table loads when aligned. Bitwise.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("folded_forward", "fused_backward", "claim", "rotary")
PRACTICAL_ONLY = ("folded_forward", "fused_backward")     # refused by prepare_module unless precision is practical
VERSION = 1
_RULES = ContextVar("latch_segmented_attention_rules", default=(None, frozenset()))


def rules_of(value):
    """The frozenset of rule names selected by ``value`` (None: empty); anything else raises ValueError."""
    if value is None:
        return frozenset()
    names = value.split("+") if isinstance(value, str) and value else None
    if not names or any(name not in RULES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"segmented_attention must be None or a '+'-joined list of distinct rules from {RULES}; "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()[0]


def active(name):
    if name not in RULES:
        raise ValueError(f"unknown segmented_attention rule {name!r}; rules are {RULES}")
    return name in _RULES.get()[1]


@contextmanager
def scope(value):
    token = _RULES.set((value, rules_of(value)))
    try:
        yield
    finally:
        _RULES.reset(token)
