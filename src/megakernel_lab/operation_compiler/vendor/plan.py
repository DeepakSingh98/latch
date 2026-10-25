"""Offset assignment for classified buffers.

Inputs: per buffer an allocation size (words, 64-aligned) and a class:
  'persistent'  storage of its own for the whole program lifetime (state, constants, host-visible, unknown);
  ('live', a, b) phase interval [a, b] inclusive within one update (b may be END = live until the host reads);
  'unused'      no device access and no host access in its contract: no storage of its own.
Persistent buffers keep the baseline relative order at the start of the arena. Live buffers are packed by first fit
(largest first, then earliest start, then name): two buffers may share words only if their phase intervals are
disjoint. Unused buffers are placed at offset 0 (never accessed). Every assignment is re-checked independently
(check()); the packer is not trusted.
"""
from __future__ import annotations

ALIGN = 64


def a64(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def assign(order, size, cls, end_phase, strategy='largest'):
    plan, top = {}, 0
    for n in order:
        if cls[n] == 'persistent':
            plan[n] = top
            top += a64(size[n])
    base = top
    live = [n for n in order if isinstance(cls[n], tuple)]
    if strategy == 'largest':
        live.sort(key=lambda n: (-a64(size[n]), cls[n][1], n))
    elif strategy == 'adversarial':                    # validation arm only: different alias partners
        live.sort(key=lambda n: (a64(size[n]), -cls[n][1], n))
    else:
        raise ValueError(strategy)
    placed = []                                    # (lo, hi, a, b)
    for n in live:
        _, a, b = cls[n]
        sz = a64(size[n])
        if sz == 0:
            plan[n] = base
            continue
        busy = sorted((lo, hi) for lo, hi, x, y in placed if not (y < a or b < x))
        off = base
        for lo, hi in busy:
            if off + sz <= lo:
                break
            off = max(off, hi)
        plan[n] = off
        placed.append((off, off + sz, a, b))
        top = max(top, off + sz)
    for n in order:
        if cls[n] == 'unused':
            plan[n] = 0
    return plan, top, base


def check(order, size, cls, plan, total):
    """independent re-check: alignment, bounds, persistent exclusivity, live overlap only with disjoint intervals"""
    errs = []
    spans = []
    for n in order:
        o, sz = plan[n], a64(size[n])
        if o % ALIGN or o < 0 or o + sz > total:
            errs.append(f'bounds/alignment {n}')
        if cls[n] == 'unused' or sz == 0:
            continue
        spans.append((o, o + sz, n))
    spans.sort()
    import bisect
    # sweep: compare each span with every later span that starts before it ends
    for i, (lo, hi, n) in enumerate(spans):
        j = i + 1
        while j < len(spans) and spans[j][0] < hi:
            m = spans[j][2]
            cn, cm = cls[n], cls[m]
            if cn == 'persistent' or cm == 'persistent':
                errs.append(f'persistent overlap {n} {m}')
            elif not (cn[2] < cm[1] or cm[2] < cn[1]):
                errs.append(f'live overlap {n} [{cn[1]},{cn[2]}] {m} [{cm[1]},{cm[2]}]')
            j += 1
    return errs


def peak_live(order, size, cls, nphase):
    """lower bound: max over phases of live words + persistent words"""
    pers = sum(a64(size[n]) for n in order if cls[n] == 'persistent')
    prof = [0] * (nphase + 1)
    for n in order:
        c = cls[n]
        if isinstance(c, tuple):
            for p in range(c[1], c[2] + 1):
                prof[p] += a64(size[n])
    return pers + max(prof), prof
