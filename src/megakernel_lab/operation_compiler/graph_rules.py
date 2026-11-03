"""Opt-in exact rewrites of the captured training graph before and during lowering.

Selected through the normal build API: ``CapturedModule.build(..., graph_rules="cse+castshare")`` (a '+'-joined list
of distinct names from ``RULES``). Omitting it (None) leaves lowering and generated source unchanged. Every rule
recognizes operation structure (targets, dtypes, shapes and readers), never model names, and keeps every value bit for
bit.

* ``cse`` -- identical pure ``aten.*`` operations of the captured graph (same target, operands, arguments and result
  layout) are merged (``exact_cse``). It runs in ``prepare_module`` before lowering, which passes the option value
  explicitly.
* ``depth_reads`` -- softmax-weighted mixtures of earlier block outputs that feed an RMSNorm (depth reads) are
  recognized (``depth_read_region``) and two rules apply: the RMSNorm's FP32 normalized value is recomputed in front
  of each backward reader instead of being stored (``<N>__nsr_rm<k>``), and up to 8 forward logit GEMVs that share
  one BF16 operand are stacked into one product (``<mm>__nsr_fq/fb/fp``).
* ``castshare`` -- an FP32 row gather read only by two or more identical BF16 casts becomes one cast of the gathered
  table plus one BF16 gather (``pack_fusion``).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("cse", "depth_reads", "castshare")
VERSION = 1
_RULES = ContextVar("latch_graph_rules", default=(None, frozenset()))


def rules_of(value):
    """The frozenset of rule names selected by ``value`` (None: empty); anything else raises ValueError."""
    if value is None:
        return frozenset()
    names = value.split("+") if isinstance(value, str) and value else None
    if not names or any(name not in RULES for name in names) or len(set(names)) != len(names):
        raise ValueError(f"graph_rules must be None or a '+'-joined list of distinct rules from {RULES}; "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()[0]


def active(name):
    if name not in RULES:
        raise ValueError(f"unknown graph_rules rule {name!r}; rules are {RULES}")
    return name in _RULES.get()[1]


@contextmanager
def scope(value):
    token = _RULES.set((value, rules_of(value)))
    try:
        yield
    finally:
        _RULES.reset(token)
