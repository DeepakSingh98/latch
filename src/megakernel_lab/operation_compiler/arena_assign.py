"""First-fit arena offsets with exactly the results of the pinned ``vendor/plan.py`` ``assign`` (LATCH-BUILD-600).

The pinned packer rescans every placed buffer for each new one (O(live^2) Python tuples: 126 of 1,239 profiled
seconds in the 4-layer 8,192-token E2-TTT build). ``assign`` here runs the same steps over arrays: persistent buffers
first in order, live buffers in the same sort order, and for each live buffer the same overlap filter over the
placed spans (phase intervals intersect: not (y < a or b < x)) and the same first-fit gap search. The gap search is
the reference's ``off = base; for lo, hi in sorted(busy): if off + sz <= lo: break; off = max(off, hi)``: ``off``
on reaching span i is the running maximum of ``base`` and the earlier ends, so the first span whose start leaves
room gives the same offset; spans with equal starts cannot change that result in either order, so ordering by start
alone is enough. The pinned file stays unchanged (its hash is part of every manifest) and ``vendor.plan.check``
still independently re-checks every assignment. Non-integer phase bounds fall back to the pinned packer.
``CHECK = True`` (validation runs only) additionally runs the pinned packer and requires identical results.
"""
from __future__ import annotations

import os

from .vendor import plan as pinned
from .vendor.plan import a64

FAST = True
CHECK = os.environ.get('LATCH_ARENA_ASSIGN_CHECK') == '1'


def assign(order, size, cls, end_phase, strategy='largest'):
    if not FAST or not all(type(cls[n]) is not tuple or all(type(x) is int for x in cls[n][1:]) for n in order):
        return pinned.assign(order, size, cls, end_phase, strategy)
    result = _assign(order, size, cls, strategy)
    if CHECK and result != pinned.assign(order, size, cls, end_phase, strategy):
        raise AssertionError('arena_assign differs from the pinned plan.assign')
    return result


def _assign(order, size, cls, strategy):
    import numpy as np
    plan, top = {}, 0
    for n in order:
        if cls[n] == 'persistent':
            plan[n] = top
            top += a64(size[n])
    base = top
    live = [n for n in order if isinstance(cls[n], tuple)]
    if strategy == 'largest':
        live.sort(key=lambda n: (-a64(size[n]), cls[n][1], n))
    elif strategy == 'adversarial':
        live.sort(key=lambda n: (a64(size[n]), -cls[n][1], n))
    else:
        raise ValueError(strategy)
    cap = max(1, len(live))
    lo_, hi_, a_, b_ = (np.empty(cap, dtype=np.int64) for _ in range(4))
    m = 0
    for n in live:
        _, a, b = cls[n]
        sz = a64(size[n])
        if sz == 0:
            plan[n] = base
            continue
        off = base
        if m:
            mask = (b_[:m] >= a) & (a_[:m] <= b)
            lo = lo_[:m][mask]
            if lo.size:
                by_start = np.argsort(lo, kind='stable')
                lo = lo[by_start]
                ends = np.maximum.accumulate(hi_[:m][mask][by_start])
                before = np.empty_like(lo)
                before[0] = base
                np.maximum(ends[:-1], base, out=before[1:])
                fit = np.flatnonzero(before + sz <= lo)
                off = int(before[fit[0]]) if fit.size else max(base, int(ends[-1]))
        plan[n] = off
        lo_[m], hi_[m], a_[m], b_[m] = off, off + sz, a, b
        m += 1
        top = max(top, off + sz)
    for n in order:
        if cls[n] == 'unused':
            plan[n] = 0
    return plan, top, base
