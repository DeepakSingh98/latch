"""Opt-in rules for the vocabulary head of flat compose programs (padded logits product, row statistics, gradient).

Selected through the normal build API: ``CapturedModule.build(..., vocab_rules="stream+views")`` (a '+'-joined list
of distinct names from ``RULES``). Omitting it (None) leaves lowering and generated source unchanged.

* ``stream`` -- the padded logits product is re-emitted on the streamed vocabulary body
  (``latch_vocab_stream::forward<4, NK, 0, false, false, 0, 2>``, kernels/vocab_stream.cuh): X rows stay resident in
  shared memory (K <= 512), W streams through a 4-stage ring, and the 64 x 64 result tiles leave through a TMA-store
  epilogue (its tensor map appended to the product's arguments). The row-statistics phase is kept unchanged. Where
  that statistics-free form is refused, the fused-statistics form ``forward<4, NK>`` writes the row statistics itself
  and replaces the statistics phase. Same products and roundings (bitwise).
* ``views`` -- vocabulary-head product phases address offset-view operands inline as ``(root + offsetLL)`` instead of
  through separately declared view pointers. Addressing only.
* ``rows`` -- the row-statistics phase and the logits-gradient phase fuse into one pass that stages one row per CTA.
  The FP32 exp-sum is reassociated, so it requires ``precision_policy='practical'``.

Shared-memory bounds. Both bodies use dynamic shared memory within ``VOCAB_DYNAMIC_SHARED_BOUND`` (the H100 opt-in
limit of 232,448 bytes less an assumed static reserve of 8,192 bytes, i.e. 224,256 bytes):

* ``stream`` keeps 128 X rows resident, so it applies only when the product's K <= 512.
* ``rows`` stages one whole padded BF16 row, ``256 + 2 * Vp`` bytes, so it applies only when Vp <= 112,000 padded
  columns.

A head outside these bounds is not an error: the rule declines and the head keeps its default body. The bound
assumes ``static_shared_reserve_bytes <= 8192``; with a larger reserve an admitted body can exceed the remaining
dynamic budget, and lowering then fails with the program's shared-memory error instead of the rule declining.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("stream", "views", "rows")
PRACTICAL_ONLY = ("rows",)            # refused by prepare_module unless precision is practical
VERSION = 1
# Dynamic shared bytes the vocabulary bodies may use: the H100 opt-in limit (232,448) less the static reserve these
# rules assume (static_shared_reserve_bytes <= 8,192). Used by vocab_rows (rows) and vocab_stream (stream).
VOCAB_DYNAMIC_SHARED_BOUND = 232448 - 8192
_RULES = ContextVar("latch_vocab_rules", default=(None, frozenset()))


def rules_of(value):
    """The frozenset of rule names selected by ``value`` (None: empty); anything else raises ValueError."""
    if value is None:
        return frozenset()
    names = value.split("+") if isinstance(value, str) and value else None
    if not names or any(name not in RULES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"vocab_rules must be None or a '+'-joined list of distinct rules from {RULES}; "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()[0]


def active(name):
    if name not in RULES:
        raise ValueError(f"unknown vocab_rules rule {name!r}; rules are {RULES}")
    return name in _RULES.get()[1]


@contextmanager
def scope(value):
    token = _RULES.set((value, rules_of(value)))
    try:
        yield
    finally:
        _RULES.reset(token)
