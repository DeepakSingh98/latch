"""P-OPS4-A contract checks (CPU; lowering only, no CUDA): eligibility, plans, storage order, merging, scope."""
from __future__ import annotations

import itertools
import random

from megakernel_lab.operation_compiler import pops4a_rules as R


def _value(name, shape, dtype='float32', role='temporary'):
    strides, s = [], 1
    for d in reversed(shape):
        strides.append(s)
        s *= d
    return {'name': name, 'shape': list(shape), 'dtype': dtype, 'strides': list(reversed(strides)), 'role': role,
            'parameter': None, 'alias_of': None, 'storage_offset': 0, 'producer': None, 'output_roles': []}


def _graph(dtype='float32'):
    """Two same-shape leading reductions sharing an operand (merge), one full reduction, one small reduction."""
    values = {n: _value(n, s, dtype if n in ('x', 'y') else 'float32', 'input' if n in ('x', 'y') else 'temporary')
              for n, s in [('x', (8320, 64)), ('y', (8320, 64)), ('xy', (8320, 64)), ('sx', (64,)),
                           ('sxy', (64,)), ('sq', (8320, 64)), ('nsq', ()), ('small', (16,))]}
    values['xs'] = _value('xs', (16, 4))
    ops = [{'name': 'xy', 'target': 'aten.mul.Tensor', 'args': [{'ref': 'x'}, {'ref': 'y'}], 'kwargs': {}},
           {'name': 'sxy', 'target': 'aten.sum.dim_IntList', 'args': [{'ref': 'xy'}, [0]], 'kwargs': {}},
           {'name': 'sx', 'target': 'aten.sum.dim_IntList', 'args': [{'ref': 'x'}, [0]], 'kwargs': {}},
           {'name': 'sq', 'target': 'aten.mul.Tensor', 'args': [{'ref': 'y'}, {'ref': 'y'}], 'kwargs': {}},
           {'name': 'nsq', 'target': 'aten.sum.default', 'args': [{'ref': 'sq'}], 'kwargs': {}}]
    return {'inputs': ['x', 'y'], 'outputs': ['sxy', 'sx', 'nsq'],
            'values': {k: v for k, v in values.items() if k not in ('small', 'xs')}, 'operations': ops}


def run():
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    # Eligibility and plans.
    check(R.column_eligible((8320, 512), (512, 1), [0], [1], 8320, 512, False), 'column [8320,512] eligible')
    check(not R.column_eligible((8320, 512), (512, 1), [0], [1], 8320, 512, True), 'exact engine keeps priority')
    check(not R.column_eligible((8320, 512), (1, 8320), [0], [1], 8320, 512, False), 'non-contiguous kept refused')
    check(not R.column_eligible((128, 16), (16, 1), [0], [1], 128, 16, False), 'fewer than 32 results refused')
    for extent, results, cols in itertools.product((4097, 6272, 8320, 16384), (32, 512, 1536, 2048), (1, 2, 4)):
        plan = R.column_plan(extent, results, cols=cols)
        check(plan['tiles'] * plan['rows'] >= extent and (plan['tiles'] - 1) * plan['rows'] < extent,
              f'column tiles cover rows exactly once {extent}x{results}')
        check(plan['blocks'] * 32 * cols >= results, f'column blocks cover results {results}')
    for extent in (33, 8192, 8193, 262144, 1048576):
        plan = R.full_plan(extent)
        check(plan['mode'] == ('direct' if extent <= R.DIRECT_MAX else 'full'), f'full plan mode {extent}')
        if plan['mode'] == 'full':
            check(plan['chunks'] * plan['chunk'] >= extent, f'full chunks cover {extent}')

    # Storage-order positions: a bijection onto the logical indices of dense permuted views.
    rng = random.Random(0)
    for _ in range(40):
        shape = [rng.choice((1, 2, 3, 5, 8)) for _ in range(rng.choice((2, 3, 4)))]
        order = list(range(len(shape)))
        rng.shuffle(order)
        strides, s = [0] * len(shape), 1
        for axis in reversed(order):
            strides[axis] = s
            s *= shape[axis]
        check(R.dense_permutation(shape, strides, 0), f'dense permutation {shape} {strides}')
        text = R.storage_order_position(shape, strides, 's')
        seen = set()
        for storage in range(s):
            logical = eval(text.replace('u', '').replace('/', '//'), {'s': storage, '__builtins__': {}})
            coords, rem = [], logical
            for d in reversed(shape):
                coords.append(rem % d)
                rem //= d
            coords = list(reversed(coords))
            check(sum(c * st for c, st in zip(coords, strides)) == storage, f'storage order maps back {shape}')
            seen.add(logical)
        check(seen == set(range(s)), f'storage order is a bijection {shape}')
    check(not R.dense_permutation([4, 4], [8, 1], 0), 'gapped view is not dense')

    # The facade selects the older fusion_reference for these synthetic graphs.
    # P7B retains this mechanism; normal default dispatch is tested separately.
    from megakernel_lab.operation_compiler import fusion_p7b
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK, _configuration
    def lower(graph, **options):
        with _LOCK, execution_scope('legacy'), _configuration('p7b', {}):
            return fusion_p7b.lower_fused(graph, **options)
    manifest = lower(_graph()).manifest
    decisions = {d['name']: d for d in manifest.get('pops4a', {}).get('reductions', [])}
    check(set(decisions) >= {'sxy', 'sx', 'nsq'}, f'automatic rule covers the leading/full reductions {sorted(decisions)}')
    check(decisions.get('sx', {}).get('merged_into') == 'sxy_p4a', 'same-shape reduction sharing operands merged')
    check(decisions.get('nsq', {}).get('mode') == 'full', 'full reduction over 532480 elements uses chunk partials')
    off = lower(_graph(), pops4a=False).manifest
    check('pops4a' not in off, 'pops4a=False reproduces the previous rules')
    bf16 = lower(_graph('bfloat16')).manifest
    check('pops4a' not in bf16, 'automatic scope leaves programs with BF16 values unchanged')
    forced = lower(_graph('bfloat16'), pops4a=True).manifest
    check(bool(forced.get('pops4a')), 'explicit pops4a=True applies to BF16 programs')
    return {'passed': not failures, 'failed': len(failures), 'failures': failures[:20]}


if __name__ == '__main__':
    print(run())
