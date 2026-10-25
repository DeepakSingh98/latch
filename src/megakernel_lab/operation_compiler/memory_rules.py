"""Opt-in activation lifetime rules for flat common compose training programs (LATCH-TRAIN-MEMORY).

Selected through the normal build API: ``CapturedModule.build(..., memory_rules="sink")``. Omitting it (None) leaves
lowering and generated source unchanged.

The compose dependency schedule places every body in its as-soon-as-possible interval. A body whose inputs are
available early but whose result is first read much later (typically backward-pass arithmetic that depends only on
saved forward values, or a forward value used only by the backward pass) then holds its result in the arena across
every interval in between. The arena is packed over these interval lifetimes, so such results raise the peak.

``"sink"`` (``lsink``): visiting bodies from the last interval backwards, a body moves to the interval just before
its earliest complete-contract successor (RAW, WAR, WAW, alias and fence order from ``execution_plan.contracts``)
when the bytes of the arena roots it first writes are at least the bytes of the arena roots whose last access it is
(those inputs stay live until the new interval). Inputs smaller than 4 KiB (scalar constants) are not weighed. No
interval the body skips gains more than those tiny inputs, so the packing lower bound does not rise materially; equal sizes let a chain of same-sized bodies (e.g. a recomputed score -> mask -> softmax
chain) sink together, since the sweep visits later bodies first. Bodies move later only; every predecessor
stays earlier and every successor later, each body computes the same values, and no interval is added (an emptied
interval is dropped). Bodies with internal grid barriers or fences, counters, private scratch, global effects or
in-place commits keep their place. No model names, shapes or derivative patterns are recognized.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = (None, "sink", "remat", "remat_all", "remat_rows", "remat_absorbed", "remat_attn")
_RULES = ContextVar("latch_memory_rules", default=None)
_FEATURES = {"sink": "+lsink", "remat": "+lsink", "remat_all": "+lsink", "remat_rows": "+lsink", "remat_absorbed": "+lsink", "remat_attn": "+lsink"}
# Late-operand rematerialization (ops/remat.py) admission per rule: "remat" keeps the step-15 "activations" filter
# (never recompute a cast of a trainable parameter: that rereads FP32 masters); "remat_all" admits every exact
# candidate. Copies are placed by ``lsink`` (compose has no fusion-scheduler anchors), i.e. just before their
# earliest reader, never earlier than their inputs.
_REMAT = {"remat": "activations", "remat_all": "all", "remat_rows": "activations", "remat_absorbed": "absorbed", "remat_attn": "activations"}
# "remat_attn" (targeted recomputation): additionally batched matrix products with an FP32 result (e.g. attention
# scores of BF16-exact operands) and row softmax: the same body is selected for the same shapes/layouts, so a copy
# has the bits of the original; it trades one product and a row pass for the saved FP32 probabilities.
ATTN_OPS = frozenset({"bmm", "_softmax"})
ATTN_DTYPES = {"bmm": ("float32",)}
# "remat_absorbed" (step-15 "absorbed" filter) also refuses copies read by a late matrix product, which would be a
# separate stored elementwise pass; the remaining copies evaluate inside their consumers' bodies.
# "remat_rows" additionally recomputes last-axis row operations (softmax, log-softmax, logsumexp): each row is one
# fixed-order body selected by shape, so a copy of the same operation on the same row values has the same bits.
ROW_OPS = frozenset({"_softmax", "_log_softmax", "logsumexp"})
VERSION = 1


def validate(value):
    if value not in RULES:
        raise ValueError(f"memory_rules must be one of {RULES}")
    return value


def current():
    return _RULES.get()


@contextmanager
def scope(value):
    token = _RULES.set(validate(value))
    try:
        yield
    finally:
        _RULES.reset(token)


# Exact per-element operations (--fmad=false: every operation rounds separately, so a copy evaluated in another body
# has the bits of the original), comparisons/selections and constant creation, in addition to ops/remat.py's set.
from .remat import RECOMPUTABLE as _BASE
RECOMPUTABLE = frozenset(_BASE) | {"where", "minimum", "maximum", "clamp", "clamp_min", "clamp_max", "rsub",
                                   "reciprocal", "erf", "expm1", "log1p", "gt", "lt", "ge", "le", "eq", "ne",
                                   "logical_not", "masked_fill", "scalar_tensor", "full", "zeros", "ones"}
DTYPES = ("float32", "bfloat16", "bool")
MAX_CHAIN = 5          # view links count toward the chain depth


def remat_mode():
    return _REMAT.get(_RULES.get())


def plan_full(graph, mode):
    """ops/remat.plan under this module's widened exact set; returns (graph, full report incl. candidates)."""
    from .remat import plan
    from .temporary_memory import remat_admission
    rule = _RULES.get()
    ops = RECOMPUTABLE | ROW_OPS if rule == "remat_rows" else RECOMPUTABLE | ATTN_OPS if rule == "remat_attn" else RECOMPUTABLE
    return plan(graph, admit=remat_admission(mode), recomputable_ops=ops, dtypes=DTYPES,
                view_links=True, max_chain=MAX_CHAIN + (3 if rule == "remat_attn" else 0), extend_inputs=True,
                require_anchor=False, op_dtypes=ATTN_DTYPES if rule == "remat_attn" else None)


def apply_ir(graph):
    """Rematerialize late operands of a flat training graph (exact; same ATen operations on the same inputs)."""
    mode = remat_mode()
    if mode is None:
        return graph, None
    graph, report = plan_full(graph, mode)
    return graph, {k: v for k, v in report.items() if k != "candidates"} | {
        "mode": mode, "values": len(report.get("candidates", {}))}


def features():
    """Extra group-rule features for the standalone flat common planner."""
    return _FEATURES.get(_RULES.get(), "")


TINY_BYTES = 4096   # inputs below this (scalar constants) are not weighed; a move adds at most this much per input
_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "int64": 8, "int32": 4, "bool": 1, "uint8": 1, "float64": 8}


def lifetime_sink(phases, intervals, layouts, dtypes, arena_roots, pinned=()):
    """Return (intervals, record). ``arena_roots``: roots planned in the arena; ``pinned``: body indices that keep
    their interval (e.g. in-place commits and readers of their targets)."""
    from .execution_plan import dependency_successors, requires_grid_fence
    successors = dependency_successors(phases, layouts)
    level = {i: k for k, members in enumerate(intervals) for i in members}
    arena = set(arena_roots)

    def root(name):
        layout = layouts.get(name)
        return getattr(layout, "root", name) if layout is not None else None

    def nbytes(r):
        layout = layouts[r]
        return int(layout.numel) * _BYTES.get(dtypes.get(r, "float32"), 4)

    first_writer, readers = {}, {}
    for index, phase in enumerate(phases):
        for name in phase.get("writes", ()):
            r = root(name)
            if r is not None:
                first_writer.setdefault(r, index)
        for name in phase.get("reads", ()):
            r = root(name)
            if r is not None:
                readers.setdefault(r, set()).add(index)
    pinned = set(pinned)
    moved, saved = [], 0
    for index in sorted(level, key=lambda i: (level[i], i), reverse=True):
        phase = phases[index]
        if (index in pinned or requires_grid_fence(phase) or phase.get("internal_mutations")
                or phase.get("counter_effects") or phase.get("private_scratch") or phase.get("global_effects")
                or phase.get("step06_slack")):
            continue
        later = [level[s] for s in successors[index] if s in level]
        if not later:
            continue
        target = min(later) - 1
        current_level = level[index]
        if target <= current_level:
            continue
        written = {root(n) for n in phase.get("writes", ())} & arena
        out = sum(nbytes(r) for r in written if first_writer.get(r) == index)
        if not out:
            continue
        extended = 0
        for r in ({root(n) for n in phase.get("reads", ())} & arena) - written:
            others = [level[j] for j in readers.get(r, ()) if j != index and j in level]
            if r in first_writer and first_writer[r] in level:
                others.append(level[first_writer[r]])
            if max(others, default=-1) < target and nbytes(r) >= TINY_BYTES:
                extended += nbytes(r)
        if out < extended:
            continue
        level[index] = target
        moved.append({"phase": index, "from": current_level, "to": target, "out_bytes": out, "extended_bytes": extended})
        saved += out - extended
    if not moved:
        return intervals, {"moved": 0}
    result = [[] for _ in intervals]
    for i, k in level.items():
        result[k].append(i)
    kept = [sorted(m) for m in result if m]
    return kept, {"moved": len(moved), "intervals_before": len(intervals), "intervals_after": len(kept),
                  "net_bytes_per_skipped_interval": saved, "moves": moved[:200]}
