"""Opt-in per-tile readiness in place of grid barriers (``tile_ready``) for flat dependency-scheduled programs.

Selected through the normal build API: ``CapturedModule.build(..., tile_readiness="attention")``. Omitting it (None)
leaves lowering and generated source unchanged.

* ``attention`` -- width-32 segmented-attention items publish, per 128-row block of their output, a count of finished
  items. A grid barrier whose crossing reads are all row-local edges from publishing attention phases becomes
  ``latch_tr::elided_barrier()``, and each consumer waits only for the row blocks it reads. Two consumer forms wait:
  claimed segmented right products (``latch_tr::ReadyT``, waiting on their A rows) and elementwise row-block loops
  (more than 2**18 elements, static round robin over the grid in ascending order). Arena lifetimes are planned in
  barrier epochs (intervals joined by an elided barrier are one epoch), so no storage is reused between intervals that
  overlap in time. Each consumer computes the same values as behind the barrier (bitwise).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("attention",)
VERSION = 1
_RULES = ContextVar("latch_tile_readiness", default=(None, frozenset()))


def rules_of(value):
    """The frozenset of rule names selected by ``value`` (None: empty); anything else raises ValueError."""
    if value is None:
        return frozenset()
    names = value.split("+") if isinstance(value, str) and value else None
    if not names or any(name not in RULES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"tile_readiness must be None or a '+'-joined list of distinct rules from {RULES}; "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()[0]


def active(name):
    if name not in RULES:
        raise ValueError(f"unknown tile_readiness rule {name!r}; rules are {RULES}")
    return name in _RULES.get()[1]


@contextmanager
def scope(value):
    token = _RULES.set((value, rules_of(value)))
    try:
        yield
    finally:
        _RULES.reset(token)
