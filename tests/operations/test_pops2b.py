# Imported from P-OPS2-B d8f8c0d981354e3b81ef4c8671ae6cc2481a3af9.
"""CPU contracts for the P-OPS2-B BF16 matrix rules (no CUDA required).

Numerical behaviour is qualified on GPU against independent CPU oracles and by bitwise pairing with the base
build; these checks pin the generated structure: the base policy is unchanged, epilogue/prologue attachment never
reads a value before it exists or old state after its commit, permuted epilogue stores invert the view exactly,
cast modes follow their admission rules, prefetches only touch graph inputs and FP32 programs are untouched.
"""
from __future__ import annotations

import itertools
import json
import math

from megakernel_lab.operation_compiler.lower import LoweringError, TensorLayout, lower
from .test_fusion import _op, _ref, _value


def _sgd_bf16_graph():
    """y = (x.bf16 @ w.bf16).float(); loss-like output; backward-like dW/dX products and an SGD commit."""
    values = {
        'x': _value((64, 64)), 'w': _value((64, 64), role='parameter', parameter='w'),
        'm': _value((64, 64), role='momentum', parameter='w'), 'g': _value((64, 64)),
        'xb': _value((64, 64), 'bfloat16'), 'wb': _value((64, 64), 'bfloat16'),
        'y': _value((64, 64), 'bfloat16'), 'yf': _value((64, 64)),
        'gy': _value((64, 64)), 'gb': _value((64, 64), 'bfloat16'), 'xt': {'shape': [64, 64], 'strides': [1, 64], 'dtype': 'bfloat16'},
        'wt': {'shape': [64, 64], 'strides': [1, 64], 'dtype': 'bfloat16'},
        'dw': _value((64, 64), 'bfloat16'), 'dx': _value((64, 64), 'bfloat16'),
        'dwf': _value((64, 64)), 'dxf': _value((64, 64)),
        'h': _value((64, 64)), 'mn': _value((64, 64)), 'wn': _value((64, 64))}
    operations = [
        _op('xb', '_to_copy.default', [_ref('x')], {'dtype': 'bfloat16'}),
        _op('wb', '_to_copy.default', [_ref('w')], {'dtype': 'bfloat16'}),
        _op('y', 'mm.default', [_ref('xb'), _ref('wb')]),
        _op('yf', '_to_copy.default', [_ref('y')], {'dtype': 'float32'}),
        _op('gy', 'mul.Tensor', [_ref('yf'), _ref('g')]),
        _op('gb', '_to_copy.default', [_ref('gy')], {'dtype': 'bfloat16'}),
        _op('xt', 't.default', [_ref('xb')]),
        _op('wt', 't.default', [_ref('wb')]),
        _op('dw', 'mm.default', [_ref('xt'), _ref('gb')]),
        _op('dx', 'mm.default', [_ref('gb'), _ref('wt')]),
        _op('dwf', '_to_copy.default', [_ref('dw')], {'dtype': 'float32'}),
        _op('dxf', '_to_copy.default', [_ref('dx')], {'dtype': 'float32'}),
        _op('h', 'add.Tensor', [_ref('dwf'), _ref('w')], {'alpha': 0.0001}),
        _op('mn', 'add.Tensor', [_ref('h'), _ref('m')], {'alpha': 0.9}),
        _op('wn', 'sub.Tensor', [_ref('w'), _ref('mn')], {'alpha': 0.01})]
    return {'inputs': ['w', 'm', 'x', 'g'], 'outputs': ['yf', 'dxf', 'dwf', 'wn', 'mn'], 'values': values,
            'operations': operations, 'state_writes': {'w': 'wn', 'm': 'mn'}}


def base_policy_has_no_rules():
    graph = _sgd_bf16_graph()
    base = lower(graph, pops2b=False)
    assert 'pops2b' not in base.manifest and 'pops2b::' not in base.source
    assert 'pops2b_mm.cuh' not in base.manifest['header_hashes']
    return {'phases': base.manifest['phase_count']}


def epilogue_and_prologue_rules_apply_safely():
    from megakernel_lab.operation_compiler import pops2b_rules
    graph = _sgd_bf16_graph()
    limit = pops2b_rules.EPI_COST_LIMIT
    pops2b_rules.EPI_COST_LIMIT = 64          # pin the structural rule independently of the cost gate
    try:
        base, auto = lower(graph, pops2b=False), lower(graph, pops2b_casts=True, pops2b_prefetch=True)
    finally:
        pops2b_rules.EPI_COST_LIMIT = limit
    rules = auto.manifest['pops2b']
    assert auto.manifest['phase_count'] < base.manifest['phase_count'], (auto.manifest['phase_count'],
                                                                        base.manifest['phase_count'])
    epilogues = rules['matrix_epilogues']
    # The weight-gradient product carries the gradient output and the in-place SGD chain.
    assert any({'dwf', 'mn', 'wn'} <= set(members) for members in epilogues.values()), epilogues
    assert auto.manifest['in_place_commits'] == {'w': 'wn', 'm': 'mn'}
    casts = rules['operand_casts']
    # The weight cast is read again (transposed) after its first product: materialized, never recomputed after
    # the parameter commit; the data-gradient cast feeds only products and reads no state: recomputed.
    assert casts['wb']['mode'] == 'materialize', casts
    assert casts['gb']['mode'] == 'recompute', casts
    assert 'wb' in auto.manifest['layouts'] and 'gb' not in auto.manifest['layouts']
    for record in rules['l2_prefetch']:
        assert all(root in graph['inputs'] for root, _ in record['operands'])
    return {'phases': [base.manifest['phase_count'], auto.manifest['phase_count']], 'casts': casts}


def corrupted_epilogue_is_rejected():
    from megakernel_lab.operation_compiler import pops2b_rules
    original = pops2b_rules.attach_epilogues
    limit = pops2b_rules.EPI_COST_LIMIT
    pops2b_rules.EPI_COST_LIMIT = 64

    def corrupt(**kw):
        epilogue_of, epilogues, permuted = original(**kw)
        phase_of, nodes = kw['phase_of'], kw['nodes']
        # Pretend a product's result was read by an epilogue member of an earlier product (impossible order).
        products = sorted((n for n in phase_of if nodes[n].kind == 'gemm'), key=phase_of.get)
        member = max(epilogue_of, key=lambda n: phase_of[epilogue_of[n]])
        victim = next(g for g in products if phase_of[g] < phase_of[epilogue_of[member]])
        phase_of[member] = phase_of[victim]
        return epilogue_of, epilogues, permuted
    pops2b_rules.attach_epilogues = corrupt
    try:
        lower(_sgd_bf16_graph())
    except (LoweringError, KeyError, AssertionError) as exc:
        return {'rejected': type(exc).__name__}
    finally:
        pops2b_rules.attach_epilogues = original
        pops2b_rules.EPI_COST_LIMIT = limit
    raise AssertionError('corrupted epilogue schedule accepted')


def permutation_inverse_is_exact():
    from megakernel_lab.operation_compiler.pops2b_rules import permutation_inverse
    checked = 0
    for dims in ((4, 8, 64, 64), (2, 3, 4), (6, 5)):
        contiguous = [math.prod(dims[k + 1:]) for k in range(len(dims))]
        for order in itertools.permutations(range(len(dims))):
            shape = tuple(dims[d] for d in order)
            strides = tuple(contiguous[d] for d in order)
            layout = TensorLayout(shape, strides, 'root', 0)
            inverse = permutation_inverse(layout, shape)
            assert inverse is not None
            cstrides = [math.prod(shape[k + 1:]) for k in range(len(shape))]
            seen = set()
            for p in range(0, math.prod(dims), max(1, math.prod(dims) // 997)):
                coords = [(p // st) % ex for st, ex in inverse]
                assert sum(c * st for c, (st, _) in zip(coords, inverse)) == p
                seen.add(sum(c * cs for c, cs in zip(coords, cstrides)))
                checked += 1
    bad = TensorLayout((4, 4), (8, 1), 'root', 0)     # gaps: not a bijection of a contiguous root
    assert permutation_inverse(bad, (4, 4)) is None
    return {'points': checked}


def fp32_programs_unchanged():
    graph = json.loads(json.dumps(_sgd_bf16_graph()))
    for value in graph['values'].values():
        value['dtype'] = 'float32'
    for op in graph['operations']:
        if op['target'] == 'aten._to_copy.default':
            op['kwargs'] = {'dtype': 'float32'}
    assert lower(graph).source == lower(graph, pops2b=False).source


def transformer_bf16_rules():
    import torch
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES, fixture, make_loss, parameters
    from megakernel_lab.operation_compiler.capture import SGDConfig, capture_training
    config = SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0001)
    summary = {}
    for recipe in ('development_bf16', 'development'):
        model = RECIPES[recipe]
        p = parameters(model)
        momentum = {k: torch.zeros_like(v) for k, v in p.items()}
        captured = capture_training(make_loss(model), p, momentum, fixture(model), config)
        base, auto = lower(captured.ir, pops2b=False), lower(captured.ir)
        if model.matrix_dtype == 'bfloat16':
            assert auto.source != base.source
            assert len(auto.manifest['in_place_commits']) == 2 * len(p)
        else:
            assert auto.source == base.source
        summary[recipe] = [base.manifest['phase_count'], auto.manifest['phase_count']]
    return summary


def run():
    tests = [base_policy_has_no_rules, epilogue_and_prologue_rules_apply_safely, corrupted_epilogue_is_rejected,
             permutation_inverse_is_exact, fp32_programs_unchanged, transformer_bf16_rules]
    results = []
    for test in tests:
        try:
            detail = test()
            results.append({'name': test.__name__, 'passed': True, **({'detail': detail} if detail else {})})
        except Exception as exc:
            results.append({'name': test.__name__, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'kind': 'pops2b-matrix-contracts', 'tests': results, 'passed': all(row['passed'] for row in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result['passed'] else 1)
