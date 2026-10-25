"""CPU contracts for the P6-D whole-update rules (lowering structure only; GPU arithmetic is checked by bitwise A/B
builds and the unchanged update gates).

Each rule must (a) select from operation/shape/stride properties, (b) leave the source unchanged when its flag is
off, and (c) keep the documented exact arithmetic (same expressions, same frozen reduction order).
"""
from __future__ import annotations

import contextlib
import hashlib

from megakernel_lab.operation_compiler import p6d_rules


def _graph(build):
    values, ops = {}, []

    def value(name, shape, dtype='float32', strides=None, role='intermediate'):
        if strides is None:
            strides, acc = [], 1
            for d in reversed(shape):
                strides.insert(0, acc)
                acc *= d
        values[name] = {'name': name, 'shape': list(shape), 'dtype': dtype, 'strides': list(strides), 'role': role}

    def op(name, target, args, kwargs=None):
        ops.append({'name': name, 'target': target, 'args': args, 'kwargs': kwargs or {}})
    inputs, outputs = build(value, op, lambda n: {'ref': n})
    return {'values': values, 'operations': ops, 'inputs': inputs, 'outputs': outputs}


def norm_graph(rows=2048, width=1024):
    """RMSNorm forward/backward fragment: row mean, rsqrt^3 broadcast, pow(x, 1), column (weight) and row sums."""
    def build(value, op, r):
        for name, shape in (('x', (rows, width)), ('w', (width,)), ('g', (rows, width))):
            value(name, shape, role='input')
        value('sq', (rows, width)); op('sq', 'aten.pow.Tensor_Scalar', [r('x'), 2])
        value('mean', (rows, 1)); op('mean', 'aten.mean.dim', [r('sq'), [-1], True])
        value('eps', (rows, 1)); op('eps', 'aten.add.Tensor', [r('mean'), 1e-05])
        value('rs', (rows, 1)); op('rs', 'aten.rsqrt.default', [r('eps')])
        value('gw', (rows, width)); op('gw', 'aten.mul.Tensor', [r('g'), r('w')])
        value('gx', (rows, width)); op('gx', 'aten.mul.Tensor', [r('gw'), r('x')])
        value('rowsum', (rows, 1)); op('rowsum', 'aten.sum.dim_IntList', [r('gx'), [1], True])
        value('half', (rows, 1)); op('half', 'aten.mul.Tensor', [r('rowsum'), -0.5])
        value('p3', (rows, 1)); op('p3', 'aten.pow.Tensor_Scalar', [r('rs'), 3])
        value('m', (rows, 1)); op('m', 'aten.mul.Tensor', [r('half'), r('p3')])
        value('e', (rows, width), strides=(1, 0)); op('e', 'aten.expand.default', [r('m'), [rows, width]])
        value('d', (rows, width)); op('d', 'aten.div.Scalar', [r('e'), width])
        value('p1', (rows, width)); op('p1', 'aten.pow.Tensor_Scalar', [r('x'), 1])
        value('t', (rows, width)); op('t', 'aten.mul.Tensor', [r('p1'), 2.0])
        value('dx', (rows, width)); op('dx', 'aten.mul.Tensor', [r('d'), r('t')])
        value('xn', (rows, width)); op('xn', 'aten.mul.Tensor', [r('x'), r('rs')])
        value('gn', (rows, width)); op('gn', 'aten.mul.Tensor', [r('g'), r('xn')])
        value('dw', (width,)); op('dw', 'aten.sum.dim_IntList', [r('gn'), [0], False])
        return ['x', 'w', 'g'], ['dx', 'dw']
    return _graph(build)


@contextlib.contextmanager
def _flags(**values):
    old = {k: getattr(p6d_rules, k) for k in values}
    try:
        for k, v in values.items():
            setattr(p6d_rules, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(p6d_rules, k, v)


OFF = dict(POW_ONE=False, BROADCAST_MATERIALIZE=False, ROW_REDUCTION=False, COLUMN_REDUCTION=False, CSE=False,
           OPTIMIZER_EPILOGUE=False, ROW_CONSUMERS=False, SINK_TINY=False, MULTI_REDUCTION=0,
           EMBEDDING_INDEX=False, MATRIX_GROUP_MAX=3, COLUMN_WIDE=False)


def _lower(graph, *, legacy_reduction=False):
    from megakernel_lab.operation_compiler.lower import lower
    # This synthetic reduction-only graph tests the P6D rule family. The public
    # selector correctly leaves graphs without an admitted matrix/attention on
    # the generic schedule; select the rule family explicitly for this contract.
    # These two legacy engine contracts opt out of the accepted P9 rowblock
    # replacement; current default rowblock admission is qualified separately.
    options = {'matrix_rule_overrides': {'p9_rowblock.ROWBLOCK': False}} if legacy_reduction else {}
    return lower(graph, schedule='p6d', schedule_options=options)


def pow_one_and_broadcast_rules_change_only_selected_expressions():
    graph = norm_graph()
    on = _lower(graph)
    with _flags(**OFF):
        off = _lower(graph)
    assert 'powf(' in off.source and ', 1.0f)' in off.source
    assert ', 1.0f)' not in on.source.split('powf(', 1)[-1].split(';')[0]
    assert on.source.count('powf(') < off.source.count('powf(')
    assert 'p3' in on.manifest['materialization']['p6d_broadcast_materialized']
    assert off.manifest['materialization']['p6d_broadcast_materialized'] == []
    return {'powf_on': on.source.count('powf('), 'powf_off': off.source.count('powf(')}


def reduction_engines_follow_strides_and_replay_the_frozen_tree():
    graph = norm_graph()
    with _flags(MULTI_REDUCTION=0, ROW_CONSUMERS=False):   # one reduction per phase: one engine record each
        on = _lower(graph, legacy_reduction=True)
    engines = sorted(p.get('engine') for p in on.manifest['phases'] if p['kind'] == 'reduction')
    assert engines.count('p6d_exact_row_per_warp') == 2, engines      # mean and row sum (unit-stride reduced axis)
    assert sum(e in ('p6d_exact_column_cta', 'p6d_exact_column_wide') for e in engines) == 1, engines
    # The engines evaluate the frozen order: per virtual thread t sequential leaves r = t + 256 j from +0, then
    # s101h::warp_sum per warp of 32 and over the 8 warp partials.
    assert '256LL * j + 32 * k + lane' in on.source                      # row engine: virtual threads 32k + lane
    assert ('256LL * j + 32 * k + g' in on.source) or ('r += 256' in on.source)   # column engine (wide or CTA)
    with _flags(ROW_REDUCTION=False, COLUMN_REDUCTION=False):
        off = _lower(graph, legacy_reduction=True)
    assert not any(str(p.get('engine', '')).startswith('p6d_') for p in off.manifest['phases'])
    return {'engines': engines}


def all_rules_off_reproduce_previous_lowering():
    graph = norm_graph()
    with _flags(**OFF):
        a = _lower(graph)
    from megakernel_lab.operation_compiler.lower import lower
    with _flags(**OFF):
        b = lower(graph, schedule='p6d', pops4d=False)
    assert a.source == b.source  # no attention group: pops4d cannot change this graph
    return {'sha256': hashlib.sha256(a.source.encode()).hexdigest()}


def cse_merges_identical_pure_operations_and_keeps_protected_values():
    import torch
    from megakernel_lab.operation_compiler.ir import TrainingIR

    def loss(p, x):
        a = (x.to(torch.bfloat16) @ p['w1'].to(torch.bfloat16)).float()
        b = (x.to(torch.bfloat16) @ p['w2'].to(torch.bfloat16)).float()
        return (a * b).sum()
    from megakernel_lab.operation_compiler.capture import capture_training as capture
    from megakernel_lab.operation_compiler.optim import SGDConfig
    g = torch.Generator().manual_seed(0)
    params = {'w1': torch.randn(64, 64, generator=g), 'w2': torch.randn(64, 64, generator=g)}
    momentum = {k: torch.zeros_like(v) for k, v in params.items()}
    x = torch.randn(64, 64, generator=g)
    ir = capture(loss, params, momentum, (x,), SGDConfig()).ir
    assert isinstance(ir, TrainingIR)
    merged, report = p6d_rules.cse(ir)
    casts = lambda graph: sum(1 for op in graph.operations if op.target.startswith('aten._to_copy')
                              and graph.values[op.outputs[0]].dtype == 'bfloat16'
                              and op.reads and op.reads[0] == ir.inputs[-1])
    assert report['merged'] >= 1 and casts(merged) == 1 < casts(ir), (report, casts(ir), casts(merged))
    protected = set(ir.output_names()) | set(ir.inputs)
    assert not any(a in protected for a, _ in report['merged_values'])
    with _flags(CSE=False):
        same, none = p6d_rules.cse(ir)
    assert same is ir and none['merged'] == 0
    return {'merged': report['merged']}


def optimizer_update_joins_weight_gradient_epilogue():
    """SGD of a BF16-recipe weight runs in its FP32-converting gradient product's register epilogue."""
    import torch
    from megakernel_lab.operation_compiler.capture import capture_training as capture
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.optim import SGDConfig

    def loss(p, x, y):
        h = (x.to(torch.bfloat16) @ p['w'].to(torch.bfloat16)).float()
        return ((h - y) ** 2).mean()
    g = torch.Generator().manual_seed(0)
    params = {'w': torch.randn(1024, 1024, generator=g) * 0.03}
    momentum = {'w': torch.zeros(1024, 1024)}
    x, y = torch.randn(2048, 1024, generator=g), torch.randn(2048, 1024, generator=g)
    ir = capture(loss, params, momentum, (x, y), SGDConfig(lr=0.01, momentum=0.9, weight_decay=1e-4)).ir
    on = lower(ir)
    with _flags(OPTIMIZER_EPILOGUE=False):
        off = lower(ir)
    import re
    # P8-A: the epilogue may run in the p5c body (phase_epi_out) or as a warp-specialized item with an FP32-converting
    # store (OUT = 1) carrying generated epilogue functors.
    joined = lambda src: ('phase_epi_out<' in src
                          or re.search(r'p8ws::item<\d+,\w+,\w+,1,\w+>\(g\d+, \[&\]', src) is not None)
    assert joined(on.source) and not joined(off.source)
    assert len(on.manifest['phases']) < len(off.manifest['phases'])
    return {'phases_on': len(on.manifest['phases']), 'phases_off': len(off.manifest['phases'])}


def row_consumers_join_the_row_reduction_phase():
    graph = norm_graph()
    with _flags(ROW_CONSUMERS=False, SINK_TINY=False, MULTI_REDUCTION=0):
        off = _lower(graph, legacy_reduction=True)
    with _flags(ROW_CONSUMERS=True, SINK_TINY=True, MULTI_REDUCTION=2):
        on = _lower(graph, legacy_reduction=True)
    assert len(on.manifest['phases']) <= len(off.manifest['phases']), (len(on.manifest['phases']), len(off.manifest['phases']))
    assert any(len([h for h in ph['names'] if h in ('mean', 'rowsum')]) == 2 for ph in on.manifest['phases'])
    assert '(row of warp o)' in on.source and p6d_rules.ROW_MARK not in on.source and p6d_rules.ROW_MARK not in off.source
    assert '__syncwarp();  // the row' in on.source
    return {'phases_on': len(on.manifest['phases']), 'phases_off': len(off.manifest['phases'])}


def run():
    tests = [row_consumers_join_the_row_reduction_phase,optimizer_update_joins_weight_gradient_epilogue,cse_merges_identical_pure_operations_and_keeps_protected_values,pow_one_and_broadcast_rules_change_only_selected_expressions,
             reduction_engines_follow_strides_and_replay_the_frozen_tree,
             all_rules_off_reproduce_previous_lowering]
    results = []
    for test in tests:
        try:
            results.append({'test': test.__name__, 'passed': True, 'detail': test()})
        except Exception as exc:  # noqa: BLE001 - receipt keeps every failure
            import traceback
            results.append({'test': test.__name__, 'passed': False, 'error': repr(exc),
                            'traceback': traceback.format_exc()})
    return {'tests': results, 'passed': all(r['passed'] for r in results)}


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=1))
