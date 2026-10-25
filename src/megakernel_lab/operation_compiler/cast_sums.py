"""Producer-fused exact square sums (LATCH-TRAIN-SYNC, ops.accumulation_rules ``fscast``; requires fsbatch + fsload).

Pattern (structural, never by name): an elementwise phase C writes a whole contiguous FP32 root R with one per-element
body line ``const float r_R = EXPR; v_R[i] = r_R;`` (e.g. the FP32 copy of a stacked BF16 expert-weight gradient),
and exact multi-CTA full sums S_0..S_{E-1} (engine ``multi_cta_full``, eight-lane form, fsbatch-eligible) read R
only through contiguous leading-dimension slices V_e = R[e*N:(e+1)*N] (``select(R, 0, e)``) inside a leaf
``where(FLAG, V_e, 0)**2``, whose other operands (flags, counters) exist before C's interval; the slices partition R.
No other phase of C's interval reads R.

Planning (before arena lifetimes): the sums move into C's interval. Emission: C's grid-stride body is not emitted;
each sum's thread lanes evaluate EXPR for exactly the elements of its own CTA chunk and lane positions, store them to
R (each element exactly once: the chunks/lanes of the sums over the slices cover R exactly once) and add the leaf of
the same register value. So:

* every element of R is computed by the same expression and stored once, as before (by a different thread);
* every sum adds, per lane and in the same order, the leaf of the same value it previously re-read from R;
* within the interval, a sum consumes only values its own thread produced (CTA-local ownership; no cross-CTA read of
  R before the next grid barrier); later readers of R are unchanged and follow at least one grid barrier;
* R's arena lifetime starts in the same interval as before; the sums' outputs start one interval earlier, and arena
  storage is planned after the move.

The 32 FP32 expert-gradient re-reads per layer of a clipped MoE update (and their separate dependent load rounds)
disappear; values are bitwise unchanged. Anything outside the pattern keeps its lowering.
"""
from __future__ import annotations

import math
import re

_BODY = re.compile(r"\Aconst float (r_\w+) = (.+); (v_\w+)\[i\] = \1;\Z")


def _numel(shape):
    return math.prod(int(d) for d in shape)


def _contiguous(layout):
    expected, stride = [], 1
    for d in reversed([int(x) for x in layout.shape]):
        expected.append(stride)
        stride *= d
    return tuple(int(s) for s in layout.strides) == tuple(reversed(expected))


def producer_line(phase, root_ident):
    """(register, expression) of an admitted single-line elementwise producer body, or None."""
    body = phase.get("body") or ()
    if (phase.get("kind") != "elementwise" or len(body) != 1 or len(phase.get("names", ())) != 1
            or phase.get("vprod_inline") or phase.get("requires_full_grid") or phase.get("counter_effects")
            or phase.get("internal_mutations")):
        return None
    m = _BODY.match(body[0])
    if m is None or m[3] != root_ident:
        return None
    return m[1], m[2]


def leaf_value(prelude, view_ident):
    """Leaf lambda(s) rewritten to take the summand value instead of reading ``view[(i)]``, or None.

    ``prelude`` is the full sum's code before its exact body: one-line ``auto latch_* = [&](long long i) -> float {...};``
    lambdas. Exactly one lambda reads the view, exactly once as ``view[(i)]``, and uses i nowhere else."""
    from .update_schedule import _leaf_prelude
    defs = _leaf_prelude(prelude)
    if not defs or len(defs) != 1:
        return None
    name, text = defs[0]
    token = f"{view_ident}[(i)]"
    if text.count(token) != 1:
        return None
    head = "auto %s = [&](long long i) -> float { " % name
    if not text.startswith(head):
        return None
    body = text[len(head):].replace(token, "latch_v")
    if re.search(r"\bi\b", body):
        return None
    return name, "auto %s = [&](float latch_v) -> float { %s" % (name, body)


def plan(phases, intervals, layouts, ident, sum_code):
    """Move admitted sums into their producer's interval. Returns (intervals, fusions, records).

    fusions: {sum phase: dict(producer=C, root=R, offset=e*N, n=N, register, expression, store=v_R)}.
    ``sum_code(i)`` returns the full sum's generated code (the same text the emitter batches)."""
    from .update_schedule import _full_sum_parts, _LANES_CALC
    interval_of = {p: k for k, members in enumerate(intervals) for p in members}
    writer = {}
    for p, phase in enumerate(phases):
        for w in phase.get("writes", ()):
            if w in layouts:
                writer.setdefault(layouts[w].root, []).append(p)
    readers = {}
    for p, phase in enumerate(phases):
        for r in phase.get("reads", ()):
            if r in layouts:
                readers.setdefault(layouts[r].root, set()).add(p)
    candidates = {}
    for s, phase in enumerate(phases):
        if phase.get("engine") != "multi_cta_full" or s not in interval_of:
            continue
        views = [r for r in phase.get("reads", ()) if r in layouts and layouts[r].root in writer
                 and len(writer[layouts[r].root]) == 1 and phases[writer[layouts[r].root][0]].get("kind") == "elementwise"
                 and layouts[r].root != r]
        if len(views) != 1:
            continue
        view = views[0]
        lay = layouts[view]
        root = lay.root
        c = writer[root][0]
        n = _numel(lay.shape)
        if not _contiguous(lay) or n <= 0 or int(lay.offset) % n or root not in layouts or not _contiguous(layouts[root]):
            continue
        candidates.setdefault((c, root), []).append((s, view, int(lay.offset), n))
    fusions, records = {}, []
    moved = {}
    for (c, root), sums in candidates.items():
        total = _numel(layouts[root].shape)
        offsets = sorted(o for _, _, o, _ in sums)
        ns = {n for _, _, _, n in sums}
        if len(ns) != 1 or len(sums) < 2:
            continue
        n = ns.pop()
        if offsets != list(range(0, total, n)):
            continue
        line = producer_line(phases[c], ident(root))
        if line is None:
            continue
        kc = interval_of[c]
        # every other operand of the sums exists before C's interval; nothing else in C's interval reads R
        ok = True
        for s, view, _, _ in sums:
            for r in phases[s].get("reads", ()):
                rr = layouts[r].root if r in layouts else r
                if rr == root:
                    continue
                if any(interval_of.get(w, -1) >= kc for w in writer.get(rr, ())):
                    ok = False
            parts = _full_sum_parts(sum_code(s))
            if parts is None or not parts[0].strip() or _LANES_CALC.match(parts[1]) is None:
                ok = False
            elif leaf_value(parts[0], ident(view)) is None:
                ok = False
            elif _LANES_CALC.match(parts[1])["expr"] != (
                    f"{leaf_value(parts[0], ident(view))[0]}((long long)((unsigned)(r) % {n}u) * 1LL)"):
                ok = False
        if any(interval_of.get(p) == kc and p != c for p in readers.get(root, ())):
            ok = False
        if not ok:
            continue
        reg, expr = line
        for s, view, offset, _ in sums:
            fusions[s] = dict(producer=c, root=root, view=view, offset=offset, n=n, register=reg, expression=expr,
                              store=ident(root))
            moved[s] = kc
        records.append(dict(producer=phases[c]["names"][0], root=root, sums=[phases[s]["names"][0] for s, *_ in sums],
                            from_interval=interval_of[sums[0][0]], to_interval=kc, slice_elements=n))
    if not moved:
        return intervals, {}, []
    out = [[p for p in members if p not in moved] for members in intervals]
    for s, kc in sorted(moved.items()):
        out[kc].append(s)
    out = [sorted(m) for m in out]
    keep = [m for m in out if m]
    return keep, fusions, records
