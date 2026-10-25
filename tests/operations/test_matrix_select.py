# P-OPS2-A adaptation from 4dcee751fab59f5ecea67ebada3167ca93fba841.
"""CPU contracts for the P-OPS2-A FP32 product selection and its schedule/launch rules (no GPU)."""
from __future__ import annotations

import re
import traceback

from megakernel_lab.operation_compiler.lower import TensorLayout, _contiguous, _header_path


def _layout(shape, root, strides=None):
    return TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), root)


def _graph(m, k, n, *, a_t=False, b_t=False, names=('a', 'b', 'c')):
    """C = A @ B with optional transposed storage (k-contiguous or m-contiguous A, n- or k-contiguous B)."""
    a, b, c = names
    values = {a: {'shape': [k, m] if a_t else [m, k], 'dtype': 'float32'},
              b: {'shape': [n, k] if b_t else [k, n], 'dtype': 'float32'},
              c: {'shape': [m, n], 'dtype': 'float32'}}
    operations, left, right = [], a, b
    if a_t:
        values[a + 't'] = {'shape': [m, k], 'dtype': 'float32'}
        operations.append({'name': a + 't', 'target': 'aten.t.default', 'args': [{'ref': a}], 'kwargs': {}})
        left = a + 't'
    if b_t:
        values[b + 't'] = {'shape': [k, n], 'dtype': 'float32'}
        operations.append({'name': b + 't', 'target': 'aten.t.default', 'args': [{'ref': b}], 'kwargs': {}})
        right = b + 't'
    operations.append({'name': c, 'target': 'aten.mm.default', 'args': [{'ref': left}, {'ref': right}], 'kwargs': {}})
    return {'inputs': [a, b], 'outputs': [c], 'values': values, 'operations': operations, 'state_writes': {}}


def selection_uses_properties_only():
    from megakernel_lab.operation_compiler.matrix_select import plan
    rows = []
    for (m, k, n), (at, bt) in [((256, 512, 2048), (False, False)), ((256, 2048, 512), (False, True)),
                                ((512, 256, 2048), (True, False)), ((256, 512, 512), (False, False))]:
        a = _layout((m, k), 'p', (1, m) if at else None)
        b = _layout((k, n), 'q', (1, k) if bt else None)
        first, _ = plan('p', 'q', 'r', a, b, (m, n), ident=lambda s: 'v_' + s)
        a2 = TensorLayout(a.shape, a.strides, 'other_weight_name')
        b2 = TensorLayout(b.shape, b.strides, 'layer7.up')
        second, _ = plan('other_weight_name', 'layer7.up', 'z', a2, b2, (m, n), ident=lambda s: 'v_' + s)
        assert first['metadata']['matrix_plan'] == second['metadata']['matrix_plan']
        rows.append([m, k, n, at, bt, first['metadata']['matrix_plan']])
    return rows


def selected_bodies_are_strict_fp32():
    from megakernel_lab.operation_compiler.matrix_select import plan
    for at in (False, True):
        for bt in (False, True):
            a = _layout((256, 512), 'a', (1, 256) if at else None)
            b = _layout((512, 512), 'b', (1, 512) if bt else None)
            p, reason = plan('a', 'b', 'c', a, b, (256, 512), ident=lambda s: 'v_' + s)
            assert p, reason
            text = p['code'] + p['helpers'] + ''.join(_header_path(h).read_text() for h in p['headers'])
            assert 'fmaf(' in text and '/*B_VBASE*/' in p['code']
            for banned in ('mma.sync', 'wgmma.', '.tf32', '__float2bfloat16', 'atomicAdd(C', 'red.global.add.f32'):
                assert banned not in text, banned
    return True


def split_scratch_is_bound_without_init_phases():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    program = lower_fused(_graph(256, 2048, 512, b_t=True))
    gemm = [p for p in program.manifest['phases'] if p['kind'] == 'gemm']
    assert gemm and gemm[0]['engine'] == 'pops2_fp32'
    assert not any(p['kind'] == 'matrix_counter_init' for p in program.manifest['phases'])
    split = [d for d in program.manifest.get('matrix_decisions', []) if d.get('plan', {}).get('ksplit', 1) > 1]
    assert split, 'expected a split-K plan for K=2048 on 256x512'
    assert '/*P2_' not in program.source
    assert 'every last arriver restores zero' in program.source
    bound = [int(x) for x in re.findall(r'reinterpret_cast<unsigned\*>\(arena \+ (\d+)LL\)', program.source)]
    assert bound and max(bound) < program.manifest['arena_words']
    base = lower_fused(_graph(256, 2048, 512, b_t=True), pops2_policy=False)
    assert any(p['kind'] == 'matrix_counter_init' for p in base.manifest['phases'])
    return {'phases': program.manifest['phase_count'], 'base_phases': base.manifest['phase_count'],
            'arena_words': program.manifest['arena_words']}


def base_policy_switch():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    graph = _graph(256, 512, 2048)
    base = lower_fused(graph, pops2_policy=False)
    explicit = lower_fused(graph, pops2_policy=True, matrix_engine='popsb', matrix_bodies=None,
                           attention_min_keys=None, warp_rows=False)
    assert base.source == explicit.source
    assert lower_fused(graph).source != base.source
    return True


def attention_key_rule():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    from .test_attention import fixture
    operations, values, _ = fixture(delayed_dout=True)
    graph = {"operations": operations, "values": values, "inputs": ["q", "k", "v", "dout", "mask"],
             "outputs": ["out", "dq", "dv"], "state_writes": {}}
    keys = values['k']['shape'][-2] if 'k' in values else None
    selected = lower_fused(graph)
    assert len(selected.manifest['attention_groups']) == 0
    assert any('keys <=' in r.get('reason', '') for r in selected.manifest.get('attention_refusals', []))
    fused = lower_fused(graph, attention_min_keys=0)
    assert len(fused.manifest['attention_groups']) == 1
    return {'keys': keys}


def warp_row_softmax_option():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    graph = {'inputs': ['x'], 'outputs': ['y'], 'state_writes': {},
             'values': {'x': {'shape': [256, 64], 'dtype': 'float32'}, 'y': {'shape': [256, 64], 'dtype': 'float32'}},
             'operations': [{'name': 'y', 'target': 'aten._softmax.default', 'args': [{'ref': 'x'}, -1, False],
                             'kwargs': {}}]}
    warp = lower_fused(graph)
    block = lower_fused(graph, warp_rows=False)
    assert 'maximum_w' in warp.source and 's101h::warp_max(maximum_w)' in warp.source
    assert 'maximum_w' not in block.source
    phase = [p for p in warp.manifest['phases'] if p['kind'] == 'softmax'][0]
    return {'phase': {k: phase[k] for k in ('kind', 'reduction_elements') if k in phase}}


def full_device_launch_for_selected_products():
    from megakernel_lab.operation_compiler.runtime import select_launch_policy
    manifest = {'phases': [{'kind': 'gemm', 'engine': 'pops2_fp32', 'tiles': 4096}]}
    policy = select_launch_policy(manifest, 132, multiprocessors=132)
    assert policy['selected_blocks'] == 132 and policy['version'].endswith('+pops2-full-device')
    legacy = select_launch_policy({'phases': [{'kind': 'gemm', 'tiles': 4096}]}, 132, multiprocessors=132)
    assert legacy['selected_blocks'] == 128
    return True


def run():
    tests = []
    for fn in (selection_uses_properties_only, selected_bodies_are_strict_fp32, split_scratch_is_bound_without_init_phases,
               base_policy_switch, attention_key_rule, warp_row_softmax_option, full_device_launch_for_selected_products):
        try:
            tests.append({'name': fn.__name__, 'passed': True, 'detail': fn()})
        except Exception:
            tests.append({'name': fn.__name__, 'passed': False, 'error': traceback.format_exc()})
    return {'kind': 'P-OPS2-A-matrix-selection-contract', 'tests': tests, 'passed': all(t['passed'] for t in tests)}
