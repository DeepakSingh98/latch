"""Opt-in diagnostics collector for compilation reports (LATCH-EXPLAIN).

The compiler already records most selections in the manifest. A few records are
computed and then discarded (the full matrix-provider candidate table, the
selection snapshot a later refusal overwrites, cast-rewrite reports, indexed
fallback reasons). When no collector is active, every ``note`` call returns
immediately and nothing about lowering changes: no manifest field, no source
byte. With a collector active, records accumulate in Python memory only; the
report joins them to the manifest afterwards. Lowering never reads them.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import threading

_STATE = threading.local()


def active():
    """The innermost active collector list, or None."""
    stack = getattr(_STATE, "stack", None)
    return stack[-1] if stack else None


def note(kind, **record):
    """Record a diagnostic when a collector is active; otherwise a no-op."""
    sink = active()
    if sink is None:
        return
    try:
        sink.append({"kind": kind, **copy.deepcopy(record)})
    except Exception:  # deepcopy of exotic objects must never break lowering
        sink.append({"kind": kind, "record": repr(record)[:4000]})


@contextmanager
def collect():
    """``with collect() as diagnostics: program = lower(...)``; yields the record list."""
    stack = getattr(_STATE, "stack", None)
    if stack is None:
        stack = _STATE.stack = []
    records = []
    stack.append(records)
    try:
        yield records
    finally:
        stack.pop()
