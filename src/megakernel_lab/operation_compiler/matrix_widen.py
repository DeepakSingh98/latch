"""Exact rounded widening in the existing batch-stacked TMA matrix epilogue.

This pass changes neither matrix plans nor row/reduction lowering. A materialized
BF16 product followed by its sole FP32 cast is replaced by the existing TMA
pair epilogue: each value is rounded to BF16 by the matrix body, widened exactly,
and stored as FP32. The pair epilogue needs no additional shared-memory tile.

Ported from frozen LATCH-MATRIX-MOESUP compiler 6c72bf2c6d; its recorded
measurement scope is separate from qualification of the assembled compiler.
"""
from __future__ import annotations

import math
import re

BATCH_TMA_WIDEN = True
SOURCE_COMMIT = '6c72bf2c6db7428438388d12dc20d48ceb75471a'


def fuse(phases, layouts, dtypes, *, ident, index, kept, inplace_roots):
    """Fuse only an identity, sole-reader cast of a physically stacked BMM."""
    from .matrix_adapters.bodies import Refusal, respec_tma
    from .matrix_rules_p7b import batch_stacked, batch_stacked_tiles

    root = lambda name: layouts[name].root if name in layouts else name
    readers = {}
    for i, phase in enumerate(phases):
        for name in phase['reads']:
            readers.setdefault(root(name), set()).add(i)
    replace, removed, elided, records = {}, set(), set(), []
    for i, phase in enumerate(phases):
        record = phase.get('batch_widen_respec')
        if phase['kind'] != 'gemm' or not record or len(phase['writes']) != 1:
            continue
        source = phase['writes'][0]
        if (source not in layouts or root(source) != source or source in kept or source in inplace_roots
                or dtypes.get(source) != 'bfloat16'):
            continue
        layout, spec, plan = layouts[source], record['spec'], record['plan']
        shape = tuple(layout.shape)
        if (shape != (spec['batch'], spec['m'], spec['n']) or not layout.contiguous or layout.offset
                or spec.get('ldc', spec['n']) != spec['n'] or spec['n'] % 2
                or math.prod(shape) >= 2 ** 32 or not spec.get('tma_batch')
                or not batch_stacked(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'], spec['strides'])
                or not batch_stacked_tiles(spec, plan)):
            continue
        users = readers.get(source, set())
        if len(users) != 1:
            continue
        j = next(iter(users))
        consumer = phases[j]
        if (j <= i or j in removed or consumer['kind'] != 'elementwise'
                or consumer.get('targets') != ['aten._to_copy.default']
                or len(consumer.get('names', [])) != 1 or consumer['reads'] != [source]
                or len(consumer.get('body', [])) != 1):
            continue
        out = consumer['names'][0]
        if (consumer['writes'] != [out] or out not in layouts or root(out) != out or out in inplace_roots
                or dtypes.get(out) != 'float32' or tuple(layouts[out].shape) != shape
                or not layouts[out].contiguous or layouts[out].offset):
            continue
        # The graph target alone does not establish the expression after other
        # passes. Require the actual emitted line to be exactly this widening.
        # Normal lowering deliberately distinguishes identity loads [(i)]
        # from stores [i]. Reuse its index emitter rather than guessing the
        # spelling, including when a different index policy is selected.
        source_index = index(layout, shape)
        expected = f'const float r_{out} = __bfloat162float({ident(source)}[{source_index}]); {ident(out)}[i] = r_{out};'
        compact = lambda text: re.sub(r'\s+', '', text)
        if compact(consumer['body'][0]) != compact(expected):
            continue
        epi = {'code': ('[&](unsigned i, float x, float y) { '
                        f'*reinterpret_cast<float2*>({ident(out)} + i) = make_float2(x, y); }}'),
               'store': False, 'staged': False, 'batch_stacked_widen': True}
        try:
            code, extra_shared = respec_tma(record, ident=ident, epi=epi)
        except Refusal:
            continue
        assert extra_shared == 0
        metadata = {'source_commit': SOURCE_COMMIT, 'source': source, 'output': out,
                    'shape': list(shape), 'product_stored': False,
                    'extra_shared_bytes': 0, 'arithmetic': 'exact FP32 widening after the original BF16 rounding',
                    'avoided_bf16_store_and_load_bytes': math.prod(shape) * 4}
        replace[i] = dict(phase, code=code, names=[*phase['names'], out],
                          targets=[*phase['targets'], *consumer['targets']], writes=[out],
                          arithmetic=phase.get('arithmetic', '') + '; exact widening of the rounded BF16 result to FP32',
                          batch_widening=metadata)
        removed.add(j)
        elided.add(source)
        records.append(metadata)
    return [replace.get(i, p) for i, p in enumerate(phases) if i not in removed], records, elided
