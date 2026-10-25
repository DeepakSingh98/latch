"""LATCH-MOE-EMA-FAST: a multi-CTA full sum evaluates its single-line elementwise source at each leaf.

    x = EXPR(i)        elementwise FP32, one line, read only by S (e.g. a gradient's square for its norm)
    S: sum(x)          multi_cta_full (per-CTA contiguous chunk trees, fixed-order combine)

Each leaf read of x becomes the same expression at the same element, so every leaf value, the per-CTA trees and
the fixed-order combine are unchanged; x is not written or re-read. ``ENABLED = False`` keeps x.
"""
from __future__ import annotations

import math

from .gathered_row_sum import _gather_reads
from .group_rules import _identity, _topological
from .indexed_source_inline import LINE

ENABLED = True


def fuse(phases, layouts, dtypes, *, ident, kept, inplace_roots):
    """Returns (phases, records, elided_roots)."""
    if not ENABLED:
        return phases, [], set()
    root = lambda name: layouts[name].root if name in layouts else name
    writer, readers = {}, {}
    for k, phase in enumerate(phases):
        for w in phase['writes']:
            writer[root(w)] = k
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    merged, removed, records, elided = {}, set(), [], set()
    for s, phase in enumerate(phases):
        if phase.get('kind') != 'reduction' or phase.get('engine') != 'multi_cta_full':
            continue
        sources = [r for r in phase['reads'] if root(r) in writer and phases[writer[root(r)]].get('kind') == 'elementwise']
        if len(sources) != 1:
            continue
        source = sources[0]
        x, e = root(source), writer[root(source)]
        elem = phases[e]
        names, body = list(elem.get('names', ())), list(elem.get('body', ()))
        if (e >= s or e in removed or not names or len(body) != len(names)
                or root(names[-1]) != x or x in kept or x in inplace_roots or readers.get(x, set()) != {s}
                or dtypes.get(x) != 'float32' or layouts[source].offset
                or math.prod(layouts[source].shape) != math.prod(elem['shape'])
                or not _identity(layouts[source], tuple(layouts[source].shape))):
            continue
        match = LINE.match(body[-1].strip())
        if match is None or match.group('array') != ident(x) or match.group('name') != names[-1]:
            continue
        # LATCH-TINYMOE-25: a multi-line source chain also qualifies when every earlier value is private
        # to it (read only inside the chain, never kept or placed); each leaf evaluates the same lines in
        # order, so leaf values, the per-CTA chunk trees and the fixed-order combine stay unchanged.
        own, prelude = {root(n) for n in names}, []
        for name, line in zip(names[:-1], body[:-1]):
            head = line.strip().partition('; ')[0]
            if (root(name) in kept or root(name) in inplace_roots or not readers.get(root(name), set()) <= {e}
                    or not head.startswith('const ') or f' r_{name} = ' not in head):
                prelude = None
                break
            prelude.append(head + '; ')
        if prelude is None:
            continue
        code = _gather_reads(phase.get('code') or '', [ident(source)], lambda q: f'latch_leaf({q})')
        if code is None or code == phase.get('code'):
            continue
        new = dict(phase)
        new.update(code='{\nauto latch_leaf = [&](long long i) -> float { ' + ''.join(prelude) + 'return '
                        + match.group('expr') + '; };\n' + code + '\n}',
                   reads=list(dict.fromkeys([*(r for r in elem['reads'] if root(r) not in own),
                                             *(r for r in phase['reads'] if root(r) != x)])),
                   targets=[*elem['targets'], *phase['targets']],
                   full_sum_inline=dict(source=source, expression_of=names[-1], chain=names,
                                        preserves='leaf values, per-CTA chunk trees and the fixed-order combine'))
        merged[s] = new
        removed.add(e)
        elided |= own
        records.append(dict(sum=phase['names'][0], source=names[-1], lines=len(names)))
    if not records:
        return phases, [], set()
    phases = [merged.get(k, p) for k, p in enumerate(phases) if k not in removed]
    return _topological(phases, root), records, elided
