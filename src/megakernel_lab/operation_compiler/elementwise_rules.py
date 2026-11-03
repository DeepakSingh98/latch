"""Opt-in elementwise, gather and copy body rules for flat compose programs.

Selected through the normal build API: ``CapturedModule.build(..., elementwise_rules="bf16vec+rolling")`` (a
'+'-joined list of distinct names from ``RULES``). Omitting it (None) leaves lowering and generated source unchanged.
Every rule is dispatched on recognized body structure (dtypes, layouts, index maps, readers and extents), never on
model names, and keeps every value bit for bit: it changes how many loads are in flight, how indices are formed or
where an element is evaluated, never an expression, rounding or reduction order.

* ``lanecse`` -- float4 lanes write a lane index ``(4j+u)/C`` with ``C % 4 == 0`` as ``(4j)/C``, so the four lanes of
  a vector share one row index.
* ``bf16vec`` -- identity-indexed contiguous BF16 operands of float4 bodies are loaded four elements at a time as one
  ``uint2``.
* ``rolling`` -- unroll-1 float4 loops issue each operand load for the next index right after its last use
  (including lane-invariant per-row reads), and the unroll of wide float4 loops is capped so that unroll x operand
  registers <= 64.
* ``divc`` -- an IEEE FP32 division ``(P / C)`` by a literal C from ``VERIFIED_DIVISORS`` becomes the inline,
  branch-free ``latch_divc`` (verified equal to ``x / C`` for every FP32 bit pattern x). The shipped table
  (ops/row_rewrites.py) lists the literals verified so far, 3, 5, 6, 7, 480, sqrt(480) and 16380; every other literal
  keeps its IEEE division. Per-literal counts are recorded in the manifest under ``group_rules["exact_divisions"]``.
* ``indexed`` -- ``select_backward`` / ``slice_backward`` placements read only at their readers' own element index are
  evaluated in those readers (the same element or the same +0) and not stored expanded.
* ``permrows`` -- permutation embedding-gradient bodies run one warp per rows: 8 row loads per batch, 4 occurrences
  per batch and one deferred bijection check, and a 16-byte warp-per-8-rows form for K = 1 BF16 gathers. Phases that
  refuse this form keep their body with the deferred bijection check only.
* ``embunroll`` -- the scalar embedding gather keeps 8 elements per thread in flight.
* ``rowcopy`` -- ``permutation_gather`` copies 16-byte row runs and checks the bijection once per row, with a run-time
  alignment fallback; rows of at most 128 vectors take one warp per 16 rows.
* ``acopy`` -- cat / slice_backward results of at least 2**20 elements that the 16-byte copy form declines are
  assembled into 16-byte vectors from raw-bit loads, with an alignment fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("lanecse", "bf16vec", "rolling", "divc", "indexed", "permrows", "embunroll", "rowcopy", "acopy")
VERSION = 1
_RULES = ContextVar("latch_elementwise_rules", default=(None, frozenset()))


def rules_of(value):
    """The frozenset of rule names selected by ``value`` (None: empty); anything else raises ValueError."""
    if value is None:
        return frozenset()
    names = value.split("+") if isinstance(value, str) and value else None
    if not names or any(name not in RULES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"elementwise_rules must be None or a '+'-joined list of distinct rules from {RULES}; "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()[0]


def active(name):
    if name not in RULES:
        raise ValueError(f"unknown elementwise_rules rule {name!r}; rules are {RULES}")
    return name in _RULES.get()[1]


@contextmanager
def scope(value):
    token = _RULES.set((value, rules_of(value)))
    try:
        yield
    finally:
        _RULES.reset(token)
