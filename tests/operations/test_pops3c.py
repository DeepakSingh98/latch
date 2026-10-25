"""CPU contracts for the P-OPS3-C normalization rules (no CUDA required).

GPU attempts qualify numerics against the independent CPU oracles and pair outputs with the P-OPS2-B build. These
checks pin the structure: the rules are off-switchable to the previous source, the row and column engines evaluate
exactly the frozen CTA reduction tree (a float32 simulation of both orders on adversarial data), statistic consumers
attach only when every other operand is ready, a row and a column reduction share one phase, and the independent
schedule check rejects a member that reads a same-phase value it cannot own.
"""
from __future__ import annotations

import json
import re

import numpy as np

from megakernel_lab.operation_compiler.lower import LoweringError
from .test_fusion import _op, _ref, _value

F32 = np.float32

def lower(graph, **options):
    # The public facade selects fusion_reference for these synthetic graphs;
    # that frozen emitter predates these rules. Exercise the retained P7B
    # mechanism itself. Normal default dispatch is tested separately.
    from megakernel_lab.operation_compiler import fusion_p7b
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK, _configuration
    with _LOCK, execution_scope('legacy'), _configuration('p7b', {}):
        return fusion_p7b.lower_fused(graph, **options)


# --------------------------------------------------------------------------------------------- order simulators
def _butterfly(lanes):
    """Lane 0 of v += shfl_xor(v, o) for o = 16..1 (all lanes hold the same value afterwards)."""
    v = [F32(x) for x in lanes]
    for o in (16, 8, 4, 2, 1):
        v = [F32(v[l] + v[l ^ o]) for l in range(32)]
    return v[0]


def frozen_cta_tree(leaves):
    """The frozen generic CTA reduction: thread t sums t, t+256, ... from +0; per-warp xor butterflies;
    the second stage butterflies the eight warp partials with zero lanes."""
    extent = len(leaves)
    acc = []
    for t in range(256):
        a = F32(0.0)
        for r in range(t, extent, 256):
            a = F32(a + leaves[r])
        acc.append(a)
    partials = [_butterfly(acc[32 * w:32 * w + 32]) for w in range(8)]
    return _butterfly(partials + [F32(0.0)] * 24)


def row_engine(leaves):
    """norm_rules.row_code: lane l holds a[w] for frozen thread 32 w + l."""
    extent, a = len(leaves), [[F32(0.0)] * 32 for _ in range(8)]
    for w in range(8):
        for lane in range(32):
            acc = F32(0.0)
            for j in range(-(-extent // 256)):
                col = 32 * w + lane + 256 * j
                if col < extent:
                    acc = F32(acc + leaves[col])
            a[w][lane] = acc
    sums = [_butterfly(a[w]) for w in range(8)]
    second = [sums[lane] if lane < 8 else F32(0.0) for lane in range(32)]
    return _butterfly(second)


def column_engine(leaves):
    """norm_rules.column_code for one column: groups of four frozen threads per (w, sub, h), xor 16/8 levels."""
    extent = len(leaves)
    part = []
    for w in range(8):
        v = {}
        for sub in range(4):
            t2 = []
            for h in range(2):
                x = []
                for q in range(4):
                    t = 32 * w + sub + 4 * h + 8 * q
                    acc = F32(0.0)
                    for j in range(-(-extent // 256)):
                        if t + 256 * j < extent:
                            acc = F32(acc + leaves[t + 256 * j])
                    x.append(acc)
                t2.append(F32(F32(x[0] + x[2]) + F32(x[1] + x[3])))
            v[sub] = F32(t2[0] + t2[1])
        v = {s: F32(v[s] + v[s ^ 2]) for s in range(4)}
        v = {s: F32(v[s] + v[s ^ 1]) for s in range(4)}
        part.append(v[0])
    s = [F32(F32(p + F32(0.0)) + F32(0.0)) for p in part]
    return F32(F32(F32(s[0] + s[4]) + F32(s[2] + s[6])) + F32(F32(s[1] + s[5]) + F32(s[3] + s[7])))


def column_engine4(leaves):
    """norm_rules._column_code4 for one column: one four-leaf subtree per sub-group, xor 16/8/4 levels."""
    extent = len(leaves)
    part = []
    for w in range(8):
        v = {}
        for sub in range(8):
            x = []
            for q in range(4):
                t = 32 * w + sub + 8 * q
                acc = F32(0.0)
                for j in range(-(-extent // 256)):
                    if t + 256 * j < extent:
                        acc = F32(acc + leaves[t + 256 * j])
                x.append(acc)
            v[sub] = F32(F32(x[0] + x[2]) + F32(x[1] + x[3]))
        for flip in (4, 2, 1):
            v = {s_: F32(v[s_] + v[s_ ^ flip]) for s_ in range(8)}
        part.append(v[0])
    s = [F32(F32(p + F32(0.0)) + F32(0.0)) for p in part]
    return F32(F32(F32(s[0] + s[4]) + F32(s[2] + s[6])) + F32(F32(s[1] + s[5]) + F32(s[3] + s[7])))


def engines_replay_frozen_tree():
    rng = np.random.default_rng(20261003)
    checked = 0
    for extent in (33, 64, 255, 256, 257, 320, 384, 512, 768, 1000, 2048):
        for trial in range(6):
            # Wide dynamic range, cancellation and signed zeros make any order change visible.
            values = (rng.standard_normal(extent) * np.exp(rng.uniform(-30, 30, extent))).astype(F32)
            if trial == 1:
                pairs = np.arange(0, extent - 1, 2)
                values[pairs + 1] = -values[pairs]
            if trial == 2:
                values[:] = F32(-0.0)
            ref = frozen_cta_tree(values)
            for engine in (row_engine, column_engine, column_engine4):
                got = engine(values)
                assert np.array_equal(np.array([ref]).view(np.uint32), np.array([got]).view(np.uint32)), (
                    engine.__name__, extent, trial, ref, got)
            checked += 1
    # A different order is distinguishable on the same data (the simulation is not vacuous).
    distinguished = 0
    for _ in range(8):
        values = rng.standard_normal(512).astype(F32)
        sequential = F32(0.0)
        for x in values:
            sequential = F32(sequential + x)
        distinguished += int(sequential != frozen_cta_tree(values))
    assert distinguished >= 4, distinguished
    return {'cases': checked, 'sequential_order_distinguished': distinguished}


# --------------------------------------------------------------------------------------------- lowered structure
def _norm_graph(rows=96, width=320, dtype='bfloat16'):
    """RMS-normalization forward, its row derivative term and the weight-gradient (column) reduction."""
    shape = (rows, width)
    values = {
        'x': _value(shape), 'w': _value((width,)), 'g': _value(shape),
        'sq': _value(shape), 'ms': _value((rows, 1)), 'me': _value((rows, 1)), 'r': _value((rows, 1)),
        'xn': _value(shape), 'y': _value(shape), 'yb': _value(shape, dtype),
        'gx': _value(shape), 'dw': _value((1, width)), 'gw': _value(shape), 'gwx': _value(shape),
        's': _value((rows, 1)), 'c': _value((rows, 1)), 'dc': _value(shape), 'gr': _value(shape),
        'dx': _value(shape), 'dxb': _value(shape, dtype)}
    operations = [
        _op('sq', 'pow.Tensor_Scalar', [_ref('x'), 2]),
        _op('ms', 'mean.dim', [_ref('sq'), [-1], True]),
        _op('me', 'add.Tensor', [_ref('ms'), 1e-5]),
        _op('r', 'rsqrt.default', [_ref('me')]),
        _op('xn', 'mul.Tensor', [_ref('x'), _ref('r')]),
        _op('y', 'mul.Tensor', [_ref('xn'), _ref('w')]),
        _op('yb', '_to_copy.default', [_ref('y')], {'dtype': dtype}),
        _op('gx', 'mul.Tensor', [_ref('g'), _ref('xn')]),
        _op('dw', 'sum.dim_IntList', [_ref('gx'), [0], True]),
        _op('gw', 'mul.Tensor', [_ref('g'), _ref('w')]),
        _op('gwx', 'mul.Tensor', [_ref('gw'), _ref('x')]),
        _op('s', 'sum.dim_IntList', [_ref('gwx'), [1], True]),
        _op('c', 'mul.Tensor', [_ref('s'), -0.5]),
        _op('dc', 'mul.Tensor', [_ref('c'), _ref('x')]),
        _op('gr', 'mul.Tensor', [_ref('gw'), _ref('r')]),
        _op('dx', 'add.Tensor', [_ref('gr'), _ref('dc')]),
        _op('dxb', '_to_copy.default', [_ref('dx')], {'dtype': dtype})]
    return {'inputs': ['x', 'w', 'g'], 'outputs': ['yb', 'dw', 'dxb', 'ms'], 'values': values,
            'operations': operations}


def off_switch_restores_previous_schedule():
    graph = _norm_graph()
    off, on = lower(graph, pops3c=False), lower(graph)
    assert 'pops3c' not in off.manifest and 'P-OPS3-C' not in off.source
    assert 'pops3c' in on.manifest and off.source != on.source
    return {'phases': [off.manifest['phase_count'], on.manifest['phase_count']]}


def normalization_phases_fuse():
    graph = _norm_graph()
    off, on = lower(graph, pops3c=False), lower(graph)
    p3 = on.manifest['pops3c']
    assert p3['row_engines'] == ['ms', 's'] and p3['column_engines'] == ['dw'], p3
    attached = {k: sorted(v) for k, v in p3['attached'].items()}
    # Default: consumers attach to a row engine that owns its phase; in a phase shared with a column engine the
    # element-wise derivative stays a full-grid loop (measured cheaper), only per-row values attach there.
    assert 'yb' in attached['ms'] and 's' not in attached, attached
    assert p3['shared_phases'] == 1, p3
    phases = [set(p['names']) for p in on.manifest['phases']]
    assert any({'ms', 'yb'} <= names for names in phases)
    assert any({'dw', 's'} <= names for names in phases)
    assert on.manifest['phase_count'] < off.manifest['phase_count']
    assert len(re.findall(r'P-OPS3-C row engine', on.source)) == 2
    assert len(re.findall(r'P-OPS3-C column engine', on.source)) == 1
    shared = lower(graph, pops3c_attach_shared=True).manifest['pops3c']['attached']
    assert 'dxb' in shared['s'], shared
    # Attached members never read the statistic from memory inside the owning phase.
    assert 'latch_rowv' in on.source and 'v_ms[g]' not in on.source.split('attached:')[1].split('}')[0]
    return {'phases': [off.manifest['phase_count'], on.manifest['phase_count']], 'attached': attached}


def _factor_graph(rows=128, width=256):
    """Row statistic s, a pointwise op of its broadcast (the RMSNorm derivative's row term) and a consumer."""
    shape = (rows, width)
    values = {'x': _value(shape), 'g': _value(shape), 'gx': _value(shape), 's': _value((rows, 1)),
              'c': _value((rows, 1)), 'e': {'shape': list(shape), 'strides': [1, 0], 'dtype': 'float32'},
              'd': _value(shape), 'p': _value(shape), 'y': _value(shape), 'z': _value(shape)}
    operations = [
        _op('gx', 'mul.Tensor', [_ref('g'), _ref('x')]),
        _op('s', 'sum.dim_IntList', [_ref('gx'), [1], True]),
        _op('c', 'mul.Tensor', [_ref('s'), -0.5]),
        _op('e', 'expand.default', [_ref('c'), list(shape)]),
        _op('d', 'div.Scalar', [_ref('e'), float(width)]),
        _op('p', 'pow.Tensor_Scalar', [_ref('x'), 1.0]),
        _op('y', 'mul.Tensor', [_ref('d'), _ref('p')]),
        _op('z', 'add.Tensor', [_ref('y'), _ref('g')]),
        _op('d2', 'mul.Tensor', [_ref('d'), 2.0]),               # chained: pointwise of a factored value
        _op('w', 'add.Tensor', [_ref('d2'), _ref('x')])]
    values.update(d2=_value(shape), w=_value(shape))
    return {'inputs': ['x', 'g'], 'outputs': ['z', 'w'], 'values': values, 'operations': operations}


def factoring_and_identities():
    graph = _factor_graph()
    off = lower(graph, pops3c_factor=False, pops3c_pow1=False, pops3c_outline=False, pops3c_outline_ops=False)
    on = lower(graph)
    # The broadcast operand is evaluated once per row (here inlined into the attached consumer at the row index).
    assert 'd__p3row' in on.manifest['materialization']['inlined'] + sum(on.manifest['pops3c']['attached'].values(), [])
    assert 'd__p3row' not in off.source
    assert 'powf(' in off.source and 'powf(v_x' not in on.source and 'latch_powf(v_x' not in on.source
    return {'phases': [off.manifest['phase_count'], on.manifest['phase_count']]}


def outlining_is_textual_and_complete():
    from megakernel_lab.operation_compiler.norm_rules import OUTLINED, cse_calls, outline
    body, used = outline("a = powf(x, 3.0f) + expf(y) + s101h::expf(z) + __expf(w) + my_powf(q);")
    assert body == "a = latch_powf(x, 3.0f) + latch_expf(y) + s101h::expf(z) + __expf(w) + my_powf(q);" and used == 2
    prelude, expr = cse_calls("(g * latch_sigmoid(x)) * (1.0f - latch_sigmoid(x)) + latch_fdiv(latch_sigmoid(x), y)", "c")
    assert prelude == ["const float c_0 = latch_sigmoid(x);"] and expr == "(g * c_0) * (1.0f - c_0) + latch_fdiv(c_0, y)"
    graph = {'inputs': ['x'], 'outputs': ['y'], 'values': {'x': _value((4096,)), 'y': _value((4096,))},
             'operations': [_op('y', 'sigmoid.default', [_ref('x')])]}
    on, off = lower(graph), lower(graph, pops3c_outline=False)
    assert 'latch_sigmoid(' in on.source and 'latch_sigmoid(' not in off.source.split('#endif')[-1]
    assert 'LATCH_OPS_OUTLINED' in on.source and 'LATCH_OPS_OUTLINED' not in off.source
    for name in ('latch_powf', 'latch_expf', 'latch_logf', 'latch_fdiv', 'latch_sigmoid', 'latch_silu'):
        assert name + '(' in OUTLINED
    return {'outlined_calls': on.source.count('latch_')}


def engines_respect_eligibility():
    # Strided (non-contiguous) reduced axis keeps the frozen engine; tiny row counts keep the CTA tree.
    graph = _norm_graph(rows=16)
    program = lower(graph)
    assert program.manifest.get('pops3c', {}).get('row_engines', []) == [], program.manifest.get('pops3c')
    graph = _norm_graph(rows=128, width=24)
    program = lower(graph)
    assert not program.manifest.get('pops3c', {}).get('row_engines'), program.manifest.get('pops3c')
    return {'small_rows': 'frozen', 'short_rows': 'frozen'}


def corrupted_attachment_is_rejected():
    from megakernel_lab.operation_compiler import fusion_p7b as fusion
    calls = []
    original = fusion._check_schedule

    def corrupt(items, loads_of, phase_of, register_load, stored, in_place, nodes, error, **kw):
        calls.append(True)
        row_of = dict(kw.pop('row_of', {}))
        assert row_of, 'expected attached members'
        member = next(iter(row_of))
        row_of[member] = 'dw'                 # claim ownership by a reduction that cannot supply the statistic
        try:
            original(items, loads_of, phase_of, register_load, stored, in_place, nodes, error, row_of=row_of, **kw)
        except LoweringError:
            raise
        raise AssertionError('corrupted attachment accepted')
    fusion._check_schedule = corrupt
    try:
        lower(_norm_graph())
    except LoweringError as exc:
        assert 'schedule check' in str(exc)
    else:
        raise AssertionError('corrupted attachment was not rejected')
    finally:
        fusion._check_schedule = original
    assert calls


def transformer_programs_use_rules():
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
        off, on = lower(captured.ir, pops3c=False), lower(captured.ir)
        p3 = on.manifest['pops3c']
        assert len(p3['row_engines']) == 2 * (2 * model.layers + 1), p3['row_engines']
        assert len(p3['column_engines']) >= 2 * model.layers + 1
        assert len(on.manifest['in_place_commits']) == len(off.manifest['in_place_commits'])
        summary[recipe] = [off.manifest['phase_count'], on.manifest['phase_count']]
    return summary


def run():
    tests = [engines_replay_frozen_tree, off_switch_restores_previous_schedule, normalization_phases_fuse,
             factoring_and_identities, outlining_is_textual_and_complete,
             engines_respect_eligibility, corrupted_attachment_is_rejected, transformer_programs_use_rules]
    results = []
    for test in tests:
        try:
            detail = test()
            results.append({'name': test.__name__, 'passed': True, **({'detail': detail} if detail else {})})
        except Exception as exc:
            results.append({'name': test.__name__, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'kind': 'pops3c-normalization-contracts', 'tests': results, 'passed': all(r['passed'] for r in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
