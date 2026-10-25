"""Metadata-only lowering admission regressions; run through the CPU job adapter."""
from __future__ import annotations

import json

from megakernel_lab.operation_compiler.lower import LoweringError, _reshape_strides, lower


def _value(shape, dtype='float32'):
    strides, stride = [], 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    return {'shape': list(shape), 'strides': list(reversed(strides)), 'dtype': dtype}


def _ref(name):
    return {'ref': name}


def _op(name, target, args, kwargs=None):
    return {'name': name, 'target': 'aten.' + target, 'args': args, 'kwargs': kwargs or {}}


def _fill_graph():
    return {'inputs': ['input'], 'outputs': ['filled'], 'values': {
        'input': _value((2, 3)), 'empty': _value((2, 3)), 'filled': _value((2, 3))},
        'operations': [_op('empty', 'empty_like.default', [_ref('input')]),
                       _op('filled', 'fill.Scalar', [_ref('empty'), 2.0])]}


def _refused(graph, message):
    try:
        lower(graph)
    except LoweringError as exc:
        assert message in str(exc), (message, str(exc))
        return
    raise AssertionError('unsafe graph was admitted')


def filled_empty_has_no_storage_or_value_read():
    program = lower(_fill_graph(), retain_all=True)
    assert 'empty' not in program.manifest['layouts']
    assert program.manifest['undefined_metadata_only'] == ['empty']
    assert program.manifest['phases'][0]['reads'] == []
    assert program.manifest['phases'][0]['names'] == ['filled']
    assert 'v_empty' not in program.source


def empty_cannot_be_returned():
    graph = _fill_graph()
    graph['outputs'] = ['empty']
    _refused(graph, 'cannot be returned or committed')


def empty_cannot_be_committed():
    graph = _fill_graph()
    graph['state_writes'] = {'input': 'empty'}
    _refused(graph, 'cannot be returned or committed')


def empty_cannot_be_read_or_viewed():
    for target, arguments in (
        ('clone.default', [_ref('empty')]),
        ('view.default', [_ref('empty'), [2, 3]]),
        ('_to_copy.default', [_ref('empty')]),
        ('add.Tensor', [_ref('empty'), _ref('input')]),
    ):
        graph = _fill_graph()
        graph['operations'][1] = _op('filled', target, arguments)
        _refused(graph, 'reads undefined empty_like')


def empty_cannot_supply_fill_value():
    graph = _fill_graph()
    graph['operations'][1] = _op('filled', 'fill.Tensor', [_ref('input'), _ref('empty')])
    _refused(graph, 'reads undefined empty_like')


def empty_cannot_hide_in_fill_kwargs():
    graph = _fill_graph()
    graph['operations'][1] = _op('filled', 'fill.Scalar', [_ref('empty'), 2.0], {'other': _ref('empty')})
    _refused(graph, 'reads undefined empty_like')


def fill_preserves_shape_and_dtype():
    for value in (_value((3, 2)), _value((2, 3), 'int64')):
        graph = _fill_graph()
        graph['values']['filled'] = value
        _refused(graph, 'preserve template shape/dtype')


def multiple_functional_fills_are_independent():
    graph = _fill_graph()
    graph['values']['other'] = _value((2, 3))
    graph['outputs'].append('other')
    graph['operations'].append(_op('other', 'fill.Scalar', [_ref('empty'), -3.0]))
    program = lower(graph)
    assert 'empty' not in program.manifest['layouts']
    assert program.manifest['phases'][0]['reads'] == []
    assert set(program.manifest['phases'][0]['names']) == {'filled', 'other'}


def _matrix_graph(dtype='float32', width=64):
    return {'inputs': ['a', 'b'], 'outputs': ['result'], 'values': {
        'a': _value((width, 128), dtype), 'b': _value((128, width), dtype),
        'result': _value((width, width), dtype)},
        'operations': [_op('result', 'mm.default', [_ref('a'), _ref('b')])]}


def fp32_arithmetic_stays_simt():
    from megakernel_lab.operation_compiler.lower import _header_path
    reference = lower(_matrix_graph(), policy='reference')
    assert reference.manifest['phases'][0]['engine'] == 'pinned_simt_64'
    assert reference.manifest['shared_bytes'] == 17408
    program = lower(_matrix_graph(), pops2_policy=False)
    products = [phase for phase in program.manifest['phases'] if phase['kind'] == 'gemm']
    assert len(products) == 1 and products[0]['engine'] == 'popsb_cpasync'
    assert 'bgemm.cuh' not in program.manifest['header_hashes']
    source = program.source + _header_path('fp32_matrix.cuh').read_text()
    assert 'fmaf(' in source and 'cp.async' in source
    assert '__float2bfloat16_rn' not in source
    assert 'mma.sync' not in source and 'wgmma.' not in source and '.tf32' not in source


def bf16_engine_declares_resources_and_headers():
    import hashlib
    from megakernel_lab.operation_compiler.lower import _header_path
    from megakernel_lab.operation_compiler.matrix_rules import HEADER, bf16_plan

    program = lower(_matrix_graph('bfloat16'), policy='auto')
    phase = program.manifest['phases'][0]
    plan, candidates = bf16_plan(64, 64, 128, 1, True, False)
    assert plan in candidates and phase['matrix_plan'] == plan
    assert phase['engine'] == f'popsc_bf16_{plan["bm"]}x{plan["bn"]}'
    assert phase['matrix_shape'] == [64, 64, 128] and phase['batch'] == 1
    assert phase['shared_bytes'] == plan['shared_bytes']
    shared = max(17408, plan['shared_bytes'])
    assert program.manifest['shared_bytes'] == shared
    for header in ('bgemm.cuh', 'tgemm.cuh', HEADER):
        assert program.manifest['header_hashes'][header] == hashlib.sha256(_header_path(header).read_bytes()).hexdigest()
    call = (f'popsc::bmm_phase<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},'
            f'true,false,{plan["ks"]},{plan["nst"]}>')
    assert call in program.source and f'kernel_args, {shared},' in program.source
    assert 'ascending k16; no split-K' in phase['arithmetic']

    # Keep the original exact-body/resource contract as an explicit reference
    # assertion; admitting C's automatic body must not replace this check.
    reference = lower(_matrix_graph('bfloat16'), policy='reference')
    assert reference.manifest['phases'][0]['engine'] == 'pinned_bf16_64'
    assert reference.manifest['shared_bytes'] == 29184
    assert {'bgemm.cuh', 'tgemm.cuh'} <= reference.manifest['header_hashes'].keys()
    assert HEADER not in reference.manifest['header_hashes']
    assert 'bgemm_phase<64,64,true,false,1,32,false>' in reference.source
    assert 'kernel_args, 29184,' in reference.source


def bf16_tail_admitted_and_implicit_mixed_gemm_refused():
    program = lower(_matrix_graph('bfloat16', width=63))
    assert program.manifest['phases'][0]['engine'] == 'generic_bf16_simt_16'
    assert 'row < 63 && ak < 128' in program.source
    assert '__float2bfloat16_rn(acc)' in program.source
    graph = _matrix_graph('bfloat16')
    graph['values']['b']['dtype'] = 'float32'
    _refused(graph, 'require matching FP32 or explicit BF16')


def lifted_constant_is_a_typed_copy():
    for dtype in ('float32', 'bfloat16', 'int64', 'bool'):
        graph = {'inputs': ['constant'], 'outputs': ['copy'], 'values': {
            'constant': _value((3, 5), dtype), 'copy': _value((3, 5), dtype)},
            'operations': [_op('copy', 'lift_fresh_copy.default', [_ref('constant')])]}
        program = lower(graph)
        assert program.manifest['layouts']['copy']['root'] == 'copy'
        assert program.manifest['phases'][0]['reads'] == ['constant']
        assert program.manifest['layouts']['copy']['dtype'] == dtype


def cross_entropy_decomposition_integer_paths_are_exact():
    graph = {'inputs': ['labels'], 'outputs': ['total'], 'values': {
        'labels': _value((5,), 'int64'), 'valid': _value((5,), 'bool'),
        'safe': _value((5,), 'int64'), 'total': _value((), 'int64')},
        'operations': [_op('valid', 'ne.Scalar', [_ref('labels'), -100]),
                       _op('safe', 'where.ScalarOther', [_ref('valid'), _ref('labels'), 0]),
                       _op('total', 'sum.default', [_ref('valid')])]}
    program = lower(graph)
    assert program.manifest['phases'][-1]['arithmetic'] == 'integer modulo2**64 sum; no FP32 conversion'
    assert 'static_cast<int64_t>(18446744073709551516ULL)' in program.source
    assert 'unsigned long long acc = 0;' in program.source


def scalar_scatter_handles_tail_class_count():
    graph = {'inputs': ['zeros', 'labels'], 'outputs': ['gradient'], 'values': {
        'zeros': _value((5, 3)), 'labels': _value((5, 1), 'int64'), 'gradient': _value((5, 3))},
        'operations': [_op('gradient', 'scatter.value', [_ref('zeros'), 1, _ref('labels'), -1.0])]}
    program = lower(graph)
    assert program.manifest['phases'][0]['kind'] == 'scatter'
    assert 'acc = -1.0f;' in program.source
    assert 'assert(index >= 0 && index < 3LL)' in program.source


def scalar_tensor_keeps_integer_precision_and_dtype():
    for dtype, scalar in (('float32', 0.5), ('bfloat16', -0.25), ('bool', True), ('int64', (1 << 40) + 3)):
        graph = {'inputs': [], 'outputs': ['scalar'], 'values': {'scalar': _value((), dtype)},
                 'operations': [_op('scalar', 'scalar_tensor.default', [scalar])]}
        program = lower(graph)
        assert program.manifest['layouts']['scalar']['dtype'] == dtype
        if dtype == 'int64':
            assert 'static_cast<int64_t>(1099511627779ULL)' in program.source


def transposed_batch_view_splits_contiguous_subspace():
    # Exact transformer backward failure: split the batch dimension while the
    # last two matrix axes remain transposed. No data reordering is required.
    graph = {'inputs': ['source'], 'outputs': ['view'], 'values': {
        'source': {**_value((16, 64, 64)), 'strides': [4096, 1, 64]},
        'view': _value((4, 4, 64, 64))},
        'operations': [_op('view', 'view.default', [_ref('source'), [4, 4, 64, 64]])]}
    program = lower(graph)
    assert program.manifest['layouts']['view']['strides'] == [16384, 4096, 1, 64]
    assert program.manifest['layouts']['view']['root'] == 'source'
    assert program.manifest['arena_bytes'] == 0


def view_stride_inference_matches_pytorch_logical_values():
    # CPU job only: check values and refusal, not arbitrary singleton strides.
    import torch
    transposed = torch.arange(24).reshape(2, 3, 4).transpose(0, 1)
    cases = [
        (transposed, (3, 2, 2, 2)),
        (transposed, (6, 4)),
        (transposed, (24,)),
        (torch.arange(120).reshape(2, 3, 4, 5).permute(1, 0, 2, 3), (3, 2, 20)),
        (torch.arange(12).reshape(3, 4)[:, ::2], (6,)),
        (torch.arange(3).reshape(1, 3).expand(4, 3), (2, 2, 3)),
        (torch.arange(3).reshape(1, 3).expand(4, 3), (12,)),
        (torch.tensor(3).expand(2, 3), (6,)),
        (torch.arange(6).reshape(2, 3).unsqueeze(1).permute(2, 1, 0), (3, 2)),
        (torch.tensor(3), (1, 1)),
    ]
    for tensor, shape in cases:
        inferred = _reshape_strides(tuple(tensor.shape), tensor.stride(), shape)
        try:
            expected = tensor.view(shape)
        except RuntimeError:
            assert inferred is None, (tensor.shape, tensor.stride(), shape, inferred)
        else:
            assert inferred is not None, (tensor.shape, tensor.stride(), shape)
            actual = torch.as_strided(tensor, shape, inferred, tensor.storage_offset())
            assert torch.equal(actual, expected), (tensor.shape, tensor.stride(), shape, inferred)


def run():
    tests = [filled_empty_has_no_storage_or_value_read, empty_cannot_be_returned,
             empty_cannot_be_committed, empty_cannot_be_read_or_viewed,
             empty_cannot_supply_fill_value, empty_cannot_hide_in_fill_kwargs,
             fill_preserves_shape_and_dtype, multiple_functional_fills_are_independent,
             fp32_arithmetic_stays_simt, bf16_engine_declares_resources_and_headers,
             bf16_tail_admitted_and_implicit_mixed_gemm_refused, lifted_constant_is_a_typed_copy,
             cross_entropy_decomposition_integer_paths_are_exact, scalar_scatter_handles_tail_class_count,
             scalar_tensor_keeps_integer_precision_and_dtype,
             transposed_batch_view_splits_contiguous_subspace,
             view_stride_inference_matches_pytorch_logical_values]
    results = []
    for test in tests:
        try:
            test()
            results.append({'name': test.__name__, 'passed': True})
        except Exception as exc:
            results.append({'name': test.__name__, 'passed': False,
                            'error': f'{type(exc).__name__}: {exc}'})
    return {'kind': 'lowering-metadata-admission', 'tests': results, 'passed': all(row['passed'] for row in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
