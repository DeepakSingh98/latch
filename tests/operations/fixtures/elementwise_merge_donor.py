"""Frozen scheduling oracle from practicalBF16 donor41995adaf94cac937a2ca27fefc816e540fa891b.

Only used for byte-for-byte planner regression tests; never used at runtime.
"""
from megakernel_lab.operation_compiler.group_rules import _identity


def merge_elementwise_chains(phases, layouts, *, ident, index, inplace_roots):
    """Release step 14 ('emerge'): a later elementwise phase B joins an earlier elementwise phase A of the same shape
    when B reads A's results at the identity index and every other operand of B is produced by a phase before A at a
    lower dependency level (so A is never delayed). B's lines run after A's at the same element, reading A's register
    values; program-order adjacency no longer decides fusion. Expressions and roundings are unchanged.
    Returns (phases, records)."""
    root = lambda name: layouts[name].root if name in layouts else name
    records = []
    changed = True
    while changed:
        changed = False
        writer, level = {}, []
        for k, phase in enumerate(phases):
            level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
            for w in phase['writes']:
                writer[root(w)] = k

        def plain(p):
            return (p['kind'] == 'elementwise' and p.get('body') and len(p['body']) == len(p['names'])
                    and not p.get('vprod_inline') and not any(root(n) in inplace_roots for n in p['names']))
        for b, phase_b in enumerate(phases):
            if not plain(phase_b):
                continue
            shape = tuple(phase_b['shape'])
            sources = {writer[root(r)] for r in phase_b['reads'] if root(r) in writer and writer[root(r)] != b}
            for a in sorted(sources, reverse=True):
                phase_a = phases[a]
                if a >= b or not plain(phase_a) or tuple(phase_a['shape']) != shape:
                    continue
                own_a = {root(w) for w in phase_a['writes']}
                # When A consumes a matrix product, B's other operands must precede that product's level too, so a
                # later epilogue pass (which admits A only on that condition) still absorbs the merged chain.
                bound = level[a] - (1 if any(root(r) in writer and phases[writer[root(r)]]['kind'] == 'gemm'
                                             for r in phase_a['reads']) else 0)
                own_b = {root(w) for w in phase_b['writes']}
                ok = True
                for r in phase_b['reads']:
                    y = root(r)
                    if y in own_b:
                        continue
                    if y in own_a:
                        if not _identity(layouts[r], shape):
                            ok = False
                    elif y in writer and (writer[y] >= a or level[writer[y]] >= bound):
                        ok = False
                if not ok:
                    continue
                names_a = sorted((n for n in layouts if root(n) in own_a), key=len, reverse=True)
                lines = []
                for line in phase_b['body']:
                    head, sep, tail = line.partition('; ')
                    for n in names_a:
                        try:
                            text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                        except ValueError:
                            continue
                        head = head.replace(text, f'r_{root(n)}')
                        # Level splitting materializes an earlier register at
                        # the loop index, rather than its canonical layout
                        # expression (for scalars that expression is 0).
                        # Rejoining the pieces must restore that register read
                        # before the now-private producer store is elided.
                        if _identity(layouts[n], shape):
                            head = head.replace(f'{ident(n)}[i]', f'r_{root(n)}')
                    lines.append(head + sep + tail)
                merged = dict(phase_a)
                merged.update(names=[*phase_a['names'], *phase_b['names']],
                              targets=[*phase_a['targets'], *phase_b['targets']],
                              reads=list(dict.fromkeys([*phase_a['reads'], *(r for r in phase_b['reads']
                                                                            if root(r) not in own_a)])),
                              writes=[*phase_a['writes'], *phase_b['writes']], body=[*phase_a['body'], *lines])
                phases = [merged if k == a else p for k, p in enumerate(phases) if k != b]
                records.append({'into': phase_a['names'][0], 'merged': list(phase_b['names'])})
                changed = True
                break
            if changed:
                break
    return phases, records


def _topological(phases, root):
    produced_by = {}
    for k, phase in enumerate(phases):
        for w in phase['writes']:
            produced_by[root(w)] = k
    deps = [{produced_by[root(r)] for r in phase['reads'] if root(r) in produced_by and produced_by[root(r)] != k}
            for k, phase in enumerate(phases)]
    order, done = [], set()
    while len(order) < len(phases):
        for k in range(len(phases)):
            if k not in done and deps[k] <= done:
                order.append(k)
                done.add(k)
                break
        else:
            raise ValueError('group rules produced a dependency cycle')
    return [phases[k] for k in order]
