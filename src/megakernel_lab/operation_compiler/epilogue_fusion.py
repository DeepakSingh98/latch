"""Elementwise consumers fused into P-OPS-B matrix-product epilogues (P-OPS2-D).

A product phase whose result is a plain row-major FP32 matrix (one batch, no
accumulating placement) can evaluate an elementwise consumer phase of the same
element count in its store loop: the engine calls a generated device lambda with
the flat output index and the final value (after any split-K slice sum). The
lambda evaluates the consumer's unchanged per-operation expressions, reading the
product result from the register and every other operand at the same flat index
as before, and performs the consumer's stores. The product result is still
stored when anything else reads it.

Admission follows dependencies only: the consumer must read the product result
element-for-element, and all its other operands must be produced strictly before
the product's dependency interval, so no outside consumer moves later.
"""
from __future__ import annotations

import math
import re


def _contiguous(shape):
    result, stride = [], 1
    for d in reversed(shape):
        result.append(stride)
        stride *= d
    return tuple(reversed(result))


def _identity(layout, shape):
    return (math.prod(layout.shape) == math.prod(shape) and layout.offset == 0
            and tuple(layout.shape) == tuple(shape)
            and all(d == 1 or s == c for d, s, c in zip(layout.shape, layout.strides, _contiguous(shape))))


def fuse_epilogues(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, accumulate):
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)

    absorbed, records = {}, []
    for g, phase in enumerate(phases):
        if phase['kind'] != 'gemm' or phase.get('engine') != 'popsb_cpasync' or int(phase.get('batch', 1)) != 1:
            continue
        code = phase.get('code') or ''
        if ',true>(' in code or code.count('pops::gemm_phase<') != 1:
            continue  # accumulating store or unexpected emission
        scratch = {item['name'] for item in phase.get('private_scratch', [])}
        results = [w for w in phase['writes'] if w not in scratch]
        if len(results) != 1 or results[0] in accumulate or dtypes.get(results[0]) != 'float32':
            continue
        out = results[0]
        out_layout = layouts[out]
        if len(out_layout.shape) != 2 or not _identity(out_layout, out_layout.shape):
            continue
        numel = math.prod(out_layout.shape)
        candidates = sorted(readers.get(out, set()) - {g})
        for e in candidates:
            consumer = phases[e]
            if e in absorbed or consumer['kind'] != 'elementwise' or consumer.get('dtype', 'float32') != 'float32':
                continue
            shape = tuple(consumer['shape'])
            if math.prod(shape) != numel or any(root(n) in inplace_roots for n in consumer['names']):
                continue
            if any(dtypes.get(n) != 'float32' for n in consumer['names']):
                continue
            own = {root(w) for w in consumer['writes']}
            ok = True
            for r in consumer['reads']:
                y = root(r)
                if y in own:
                    continue
                if y == out:
                    if not _identity(layouts[r], shape):
                        ok = False
                elif y in writer and level[writer[y]] >= level[g]:
                    ok = False
            if not ok:
                continue
            absorbed[e] = g
            records.append({'product': out, 'consumer_names': list(consumer['names']),
                            'arithmetic': 'unchanged consumer expressions evaluated in the product store loop'})
            break  # one consumer phase per product
    if not absorbed:
        return phases, [], set()
    by_product = {g: e for e, g in absorbed.items()}
    result, elided = [], set()
    for k, phase in enumerate(phases):
        if k in absorbed:
            continue
        if k not in by_product:
            result.append(phase)
            continue
        e = by_product[k]
        consumer = phases[e]
        scratch = {item['name'] for item in phase.get('private_scratch', [])}
        out = [w for w in phase['writes'] if w not in scratch][0]
        shape = tuple(consumer['shape'])
        register = f'r_{out}'
        lines = []
        names = [n for n in layouts if root(n) == out]
        for line in consumer['body']:
            for n in sorted(names, key=len, reverse=True):
                try:
                    text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                except ValueError:
                    continue
                line = line.replace(text, register)
            lines.append(line)
        outside = (out in kept) or any(q not in (k, e) for q in readers.get(out, ()))
        store = f' {ident(out)}[i] = {register};' if outside else ''
        for name in consumer['names']:
            y = root(name)
            if y not in kept and not any(q not in (k, e) for q in readers.get(y, ())):
                lines = [re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', line)
                         if line.startswith(f'const float r_{name} =') else line for line in lines]
                elided.add(name)
        lam = (f'[=](long long i, float {register}) {{{store} ' + ' '.join(lines) + ' }')
        code = phase['code']
        if ',false>(' not in code:
            code = code.replace('>(', ',false>(', 1)
        code = code.rstrip()
        assert code.endswith(', smem);'), code[-40:]
        code = code[:-len(', smem);')] + f', smem, {lam});'
        merged = dict(phase)
        merged.update(code=code, names=[*phase['names'], *consumer['names']],
                      targets=[*phase['targets'], *consumer['targets']],
                      reads=list(dict.fromkeys([*phase['reads'], *(r for r in consumer['reads']
                                                                    if root(r) not in {root(w) for w in consumer['writes']}
                                                                    and root(r) != out)])),
                      writes=[*phase['writes'], *consumer['names']],
                      epilogue={'consumer': list(consumer['names']), 'product_stored': outside})
        result.append(merged)
    # Restore a dependency-respecting order (stable topological sort).
    produced_by = {}
    for k, phase in enumerate(result):
        for w in phase['writes']:
            produced_by[root(w)] = k
    deps = [{produced_by[root(r)] for r in phase['reads'] if root(r) in produced_by and produced_by[root(r)] != k}
            for k, phase in enumerate(result)]
    order, done = [], set()
    while len(order) < len(result):
        ready = next((k for k in range(len(result)) if k not in done and deps[k] <= done), None)
        if ready is None:
            raise ValueError('epilogue fusion produced a dependency cycle')
        order.append(ready)
        done.add(ready)
    return [result[k] for k in order], records, elided
