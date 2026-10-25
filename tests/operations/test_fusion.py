"""CPU contracts for the fused elementwise/state schedule (no CUDA required).

Numerical behaviour is qualified on GPU against independent CPU oracles and by
bitwise pairing with the frozen schedule; these checks pin the generated
structure: elided BF16 values keep their rounding, in-place commits happen only
when no later old-state read exists, register-only values have no storage, and
the independent schedule check rejects illegal plans.
"""
from __future__ import annotations

import json
import re

from megakernel_lab.operation_compiler.lower import LoweringError, _lower_serial, lower


def _value(shape, dtype='float32', **extra):
    strides, stride = [], 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    return {'shape': list(shape), 'strides': list(reversed(strides)), 'dtype': dtype, **extra}


def _ref(name):
    return {'ref': name}


def _op(name, target, args, kwargs=None):
    return {'name': name, 'target': 'aten.' + target, 'args': args, 'kwargs': kwargs or {}}


def _cast_graph():
    return {'inputs': ['x'], 'outputs': ['c'], 'values': {
        'x': _value((64,)), 'a': _value((64,), 'bfloat16'), 'b': _value((64,), 'bfloat16'), 'c': _value((64,))},
        'operations': [_op('a', '_to_copy.default', [_ref('x')], {'dtype': 'bfloat16'}),
                       _op('b', 'mul.Tensor', [_ref('a'), 0.37]),
                       _op('c', '_to_copy.default', [_ref('b')], {'dtype': 'float32'})]}


def reference_policy_is_stage4_serial_function():
    graph = _cast_graph()
    reference = dict(matrix_policy='frozen', split_k=False, exact_reductions=False,
                     fuse_attention=False, ordered_index_gradients=False)
    assert lower(graph, policy='reference').source == _lower_serial(graph, **reference).source
    assert lower(graph, policy='reference', fuse_elementwise=False).source == _lower_serial(
        graph, fuse_elementwise=False, **reference).source


def bf16_elision_keeps_rounding():
    program = lower(_cast_graph())
    m = program.manifest
    assert m['materialization']['inlined'] == ['a', 'b'], m['materialization']
    assert 'v_a' not in program.source and 'v_b' not in program.source
    expected = ('__bfloat162float(__float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn(v_x[i])) * 0.37f)))')
    assert expected in program.source, program.source[-2000:]
    assert m['phase_count'] == 2 and m['arena_bytes'] == 256


def _state_graph(later_read):
    values = {'p': _value((8,), role='parameter', parameter='w'), 'g': _value((8,)),
              'm': _value((8,), role='momentum', parameter='w'),
              'h': _value((8,)), 'mn': _value((8,)), 'pn': _value((8,)), 'q': _value((8,))}
    operations = [_op('h', 'add.Tensor', [_ref('g'), _ref('p')], {'alpha': 0.0001}),
                  _op('mn', 'add.Tensor', [_ref('h'), _ref('m')], {'alpha': 0.9}),
                  _op('pn', 'sub.Tensor', [_ref('p'), _ref('mn')], {'alpha': 0.01})]
    outputs = ['pn', 'mn']
    if later_read:
        operations.append(_op('q', 'mul.Tensor', [_ref('p'), 2.0]))
        outputs.append('q')
    return {'inputs': ['p', 'm', 'g'], 'outputs': outputs, 'values': values, 'operations': operations,
            'state_writes': {'p': 'pn', 'm': 'mn'}}


def inplace_commit_only_without_later_old_read():
    program = lower(_state_graph(False))
    m = program.manifest
    assert m['in_place_commits'] == {'p': 'pn', 'm': 'mn'} and m['copied_commits'] == {}
    assert m['layouts']['pn']['root'] == 'p' and m['layouts']['mn']['root'] == 'm'
    assert 'All old-state reads' not in program.source
    loop = program.source[program.source.index('// 8 elements'):]
    assert loop.index('v_pn[i] =') > loop.index('r_pn[v] =') > loop.index('r_mn[v] =')
    late = lower(_state_graph(True)).manifest
    assert late['in_place_commits'] == {'m': 'mn'}, late['in_place_commits']
    assert late['copied_commits'] == {'p': 'pn'}
    frozen = lower(_state_graph(False), policy='reference').manifest
    assert 'in_place_commits' not in frozen


def register_only_values_have_no_storage():
    uses = [f'u{k}' for k in range(7)]
    values = {'x': _value((32,)), 'y': _value((32,)), **{u: _value((32,)) for u in uses}}
    operations = [_op('y', 'exp.default', [_ref('x')])]
    operations += [_op(u, 'mul.Tensor', [_ref('y'), float(k + 2)]) for k, u in enumerate(uses)]
    program = lower({'inputs': ['x'], 'outputs': uses, 'values': values, 'operations': operations})
    m = program.manifest
    assert m['materialization']['register_only'] == ['y'], m['materialization']
    assert 'v_y' not in program.source and 'r_y[v]' in program.source
    assert m['phase_count'] == 2


def broadcast_operands_are_not_vectorized():
    n = 256 * 512
    values = {'x': _value((256, 512)), 'w': _value((512,)), 'y': _value((256, 512))}
    program = lower({'inputs': ['x', 'w'], 'outputs': ['y'], 'values': values,
                     'operations': [_op('y', 'mul.Tensor', [_ref('x'), _ref('w')])]})
    assert '(x4 vector)' in program.source
    assert 'latch_ld4(v_x + 4u * q' in program.source and 'latch_ld4(v_w' not in program.source
    assert 'v_w[(i % 512u)]' in program.source, program.source[-1500:]
    assert n % 4 == 0


def retain_all_materializes_everything():
    program = lower(_cast_graph(), retain_all=True)
    m = program.manifest
    assert m['retained_all'] and {'a', 'b', 'c'} <= set(m['layouts'])
    assert all(m['layouts'][name]['root'] == name for name in ('a', 'b', 'c'))
    assert all('v_' + name in program.source for name in ('a', 'b', 'c'))
    state = lower(_state_graph(False), retain_all=True).manifest
    assert state.get('in_place_commits', {}) == {}
    assert state['layouts']['pn']['root'] != 'p' and state['layouts']['mn']['root'] != 'm'


def schedule_check_rejects_illegal_plans():
    # Patch the actual selected implementation, not the public dispatcher.
    from megakernel_lab.operation_compiler import fusion_reference as fusion
    calls = []
    original = fusion._check_schedule

    def corrupt(items, loads_of, phase_of, register_load, stored, in_place, nodes, error):
        calls.append(True)
        bad = dict(phase_of)
        bad['mn'] = bad['pn'] + 1           # consumer before producer
        try:
            original(items, loads_of, bad, register_load, stored, in_place, nodes, error)
        except LoweringError:
            raise
        raise AssertionError('corrupted schedule accepted')
    fusion._check_schedule = corrupt
    try:
        lower(_state_graph(False))
    except LoweringError as exc:
        assert 'schedule check' in str(exc)
    else:
        raise AssertionError('corrupted schedule was not rejected')
    finally:
        fusion._check_schedule = original
    assert calls


def transformer_programs_lower_all_schedules():
    import torch
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES, fixture, initial_state, make_loop_functions, make_loss, parameters
    from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training
    from megakernel_lab.operation_compiler.loop import capture_loop_training
    config = SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0001)
    summary = {}
    for recipe in ('development', 'development_bf16'):
        model = RECIPES[recipe]
        p = parameters(model)
        momentum = {k: torch.zeros_like(v) for k, v in p.items()}
        captured = capture_training(make_loss(model, intermediate=True), p, momentum, fixture(model), config)
        programs = {s: lower(captured.ir, policy=s) for s in ('reference', 'auto')}
        summary[recipe] = {s: (pr.manifest['phase_count'], pr.manifest['arena_bytes']) for s, pr in programs.items()}
        assert programs['auto'].manifest['phase_count'] < programs['reference'].manifest['phase_count']
        assert len(programs['auto'].manifest['in_place_commits']) == 2 * len(p)
    model = RECIPES['development']
    p = parameters(model)
    momentum = {k: torch.zeros_like(v) for k, v in p.items()}
    captured = capture_loop_training(*make_loop_functions(model, intermediate=True), p, momentum, fixture(model),
                                     torch.tensor(1, dtype=torch.int64), max_steps=3, state=initial_state(),
                                     config=config, storage_mode='save_boundaries')
    loop = {s: lower(captured.ir, policy=s) for s in ('reference', 'auto')}
    regions = loop['auto'].manifest['regions']
    assert all(not region.get('in_place_commits') for region in regions.values())
    summary['combined_loop_region_phases'] = {name: (loop['reference'].manifest['regions'][name]['phase_count'],
                                                     region['phase_count']) for name, region in regions.items()}
    return summary


def run():
    tests = [reference_policy_is_stage4_serial_function, bf16_elision_keeps_rounding,
             inplace_commit_only_without_later_old_read, register_only_values_have_no_storage,
             broadcast_operands_are_not_vectorized,
             retain_all_materializes_everything, schedule_check_rejects_illegal_plans,
             transformer_programs_lower_all_schedules]
    results = []
    for test in tests:
        try:
            detail = test()
            results.append({'name': test.__name__, 'passed': True, **({'detail': detail} if detail else {})})
        except Exception as exc:
            results.append({'name': test.__name__, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'kind': 'fused-schedule-contracts', 'tests': results, 'passed': all(row['passed'] for row in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
