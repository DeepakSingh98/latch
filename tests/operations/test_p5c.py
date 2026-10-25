"""CPU contracts for the P-OPS5-C warpgroup-MMA matrix rules (lowering only; GPU arithmetic is gated separately).

The graph is the explicit-BF16 feed-forward forward/backward of one ordinary-transformer block, written in the
captured-graph operation format (the same operations pops5c_group.extract slices from a captured update).
"""
from __future__ import annotations

import hashlib


def ffn_group(m=2048, d=1024, f=4096):
    values, ops = {}, []

    def value(name, shape, dtype, strides=None, role='intermediate'):
        if strides is None:
            strides = [shape[1], 1]
        values[name] = {'name': name, 'shape': list(shape), 'dtype': dtype, 'strides': list(strides), 'role': role}

    def op(name, target, args, kwargs=None):
        ops.append({'name': name, 'target': target, 'args': args, 'kwargs': kwargs or {}})

    r = lambda n: {'ref': n}
    for name, shape in (('n2', (m, d)), ('W1', (d, f)), ('W2', (f, d)), ('dY', (m, d))):
        value(name, shape, 'float32', role='input')
    value('cn', (m, d), 'bfloat16'); op('cn', 'aten._to_copy.default', [r('n2')], {'dtype': 'bfloat16'})
    value('cw1', (d, f), 'bfloat16'); op('cw1', 'aten._to_copy.default', [r('W1')], {'dtype': 'bfloat16'})
    value('z', (m, f), 'bfloat16'); op('z', 'aten.mm.default', [r('cn'), r('cw1')])
    value('z32', (m, f), 'float32'); op('z32', 'aten._to_copy.default', [r('z')], {'dtype': 'float32'})
    value('a', (m, f), 'float32'); op('a', 'aten.silu.default', [r('z32')])
    value('ca', (m, f), 'bfloat16'); op('ca', 'aten._to_copy.default', [r('a')], {'dtype': 'bfloat16'})
    value('cw2', (f, d), 'bfloat16'); op('cw2', 'aten._to_copy.default', [r('W2')], {'dtype': 'bfloat16'})
    value('y', (m, d), 'bfloat16'); op('y', 'aten.mm.default', [r('ca'), r('cw2')])
    value('y32', (m, d), 'float32'); op('y32', 'aten._to_copy.default', [r('y')], {'dtype': 'float32'})
    value('zero', (m, d), 'float32'); op('zero', 'aten.mul.Tensor', [r('y32'), 0.0])
    value('dye', (m, d), 'float32'); op('dye', 'aten.add.Tensor', [r('dY'), r('zero')])
    value('cdy', (m, d), 'bfloat16'); op('cdy', 'aten._to_copy.default', [r('dye')], {'dtype': 'bfloat16'})
    value('ta', (f, m), 'bfloat16', [1, f]); op('ta', 'aten.t.default', [r('ca')])
    value('dw2', (f, d), 'bfloat16'); op('dw2', 'aten.mm.default', [r('ta'), r('cdy')])
    value('tw2', (d, f), 'bfloat16', [1, d]); op('tw2', 'aten.t.default', [r('cw2')])
    value('da', (m, f), 'bfloat16'); op('da', 'aten.mm.default', [r('cdy'), r('tw2')])
    value('dw2f', (f, d), 'float32'); op('dw2f', 'aten._to_copy.default', [r('dw2')], {'dtype': 'float32'})
    value('daf', (m, f), 'float32'); op('daf', 'aten._to_copy.default', [r('da')], {'dtype': 'float32'})
    value('s', (m, f), 'float32'); op('s', 'aten.sigmoid.default', [r('z32')])
    value('e', (m, f), 'float32'); op('e', 'aten.empty_like.default', [r('s')], {'memory_format': 'preserve_format'})
    value('one', (m, f), 'float32'); op('one', 'aten.fill.Scalar', [r('e'), 1.0])
    value('oms', (m, f), 'float32'); op('oms', 'aten.sub.Tensor', [r('one'), r('s')])
    value('zo', (m, f), 'float32'); op('zo', 'aten.mul.Tensor', [r('z32'), r('oms')])
    value('zp', (m, f), 'float32'); op('zp', 'aten.add.Scalar', [r('zo'), 1.0])
    value('g1', (m, f), 'float32'); op('g1', 'aten.mul.Tensor', [r('daf'), r('s')])
    value('dz', (m, f), 'float32'); op('dz', 'aten.mul.Tensor', [r('g1'), r('zp')])
    value('cdz', (m, f), 'bfloat16'); op('cdz', 'aten._to_copy.default', [r('dz')], {'dtype': 'bfloat16'})
    value('tn', (d, m), 'bfloat16', [1, d]); op('tn', 'aten.t.default', [r('cn')])
    value('dw1', (d, f), 'bfloat16'); op('dw1', 'aten.mm.default', [r('tn'), r('cdz')])
    value('tw1', (f, d), 'bfloat16', [1, f]); op('tw1', 'aten.t.default', [r('cw1')])
    value('dn', (m, d), 'bfloat16'); op('dn', 'aten.mm.default', [r('cdz'), r('tw1')])
    value('dw1f', (d, f), 'float32'); op('dw1f', 'aten._to_copy.default', [r('dw1')], {'dtype': 'float32'})
    value('dnf', (m, d), 'float32'); op('dnf', 'aten._to_copy.default', [r('dn')], {'dtype': 'float32'})
    return {'values': values, 'operations': ops, 'inputs': ['n2', 'W1', 'W2', 'dY'],
            'outputs': ['y32', 'dnf', 'dw1f', 'dw2f']}


def _lower(**settings):
    from megakernel_lab.operation_compiler import p5c_matrix
    from megakernel_lab.operation_compiler.lower import lower
    saved = {k: getattr(p5c_matrix, k) for k in settings}
    try:
        for k, v in settings.items():
            setattr(p5c_matrix, k, v)
        return lower(ffn_group(), schedule='p6d')
    finally:
        for k, v in saved.items():
            setattr(p5c_matrix, k, v)


def popsc_engine_ignores_lane_switches():
    a = _lower(ENGINE='popsc', P5C_CONVERT=False, P5C_EPILOGUE=False)
    b = _lower(ENGINE='popsc', P5C_CONVERT=True, P5C_EPILOGUE=True, P5C_PINGPONG='always')
    assert a.cache_key == b.cache_key and 'p5c::' not in a.source and 'tensor_maps' not in a.manifest
    return {'source_sha256': hashlib.sha256(a.source.encode()).hexdigest()}


def admission_is_property_based():
    from megakernel_lab.operation_compiler import p5c_matrix as P
    expect = {(2048, 1024, 1024): 'p5c', (2048, 4096, 1024): 'p5c', (1024, 1024, 2048): 'p5c',
              (2048, 256, 1024): 'p5c', (512, 512, 2048): None, (8192, 256, 256): 'p5c', (8192, 1024, 256): 'p5c',
              (8192, 256, 1024): 'p5c', (256, 256, 8192): None, (2048, 1000, 1024): None}
    got = {shape: P.admit(*shape, 1, True, True, 0, 0, engine='p5c') for shape in expect}
    assert got == expect, got
    # 96742fdfe selected P9's 4-stage/quarter-grid default before the maintained
    # 2f9bbc3f3 baseline. Keep the original P5C 8-stage/half-grid boundaries as
    # explicit policy checks instead of requiring obsolete global defaults.
    historical = dict(expect)
    for shape in ((2048, 256, 1024), (8192, 256, 256), (8192, 1024, 256)):
        historical[shape] = None
    old_policy = {shape: P.admit(*shape, 1, True, True, 0, 0, engine='p5c',
                                min_k=8, min_tiles=64) for shape in historical}
    assert old_policy == historical, old_policy
    assert P.admit(2048, 1024, 1024, 2, True, True, 0, 0, engine='p5c') is None    # batched products excluded
    return {'selected_policy': {str(k): v for k, v in got.items()},
            'historical_policy': {str(k): v for k, v in old_policy.items()}}


def engine_program_structure():
    prog = _lower(ENGINE='p5c', P5C_CONVERT=True, P5C_EPILOGUE=True)
    src, man = prog.source, prog.manifest
    assert 'P5C_TMAP(' not in src and man['tensor_maps'], 'tensor maps finalized'
    assert all(m['storage'] in ('input', 'arena') for m in man['tensor_maps'])
    assert src.count('extern "C" __global__') == 1 and '#define assert(c) do { if (!(c)) __trap(); }' in src
    gemm = [p for p in man['phases'] if p['kind'] == 'gemm']
    names = [n for p in gemm for n in p['names']]
    # Conversion rule: returned FP32 conversions are written by their products; their BF16 values have no storage.
    for conv, prod in (('y32', 'y'), ('dnf', 'dn'), ('dw1f', 'dw1'), ('dw2f', 'dw2')):
        assert conv in names and prod not in man['layouts'], (conv, prod)
    # Epilogue rule: SiLU and its derivative chain run in the products' register epilogues.
    assert src.count('p5c::phase_epi<') == 2 and 'silu' not in ''.join(p['kind'] for p in man['phases'])
    assert 'da' not in man['layouts'], 'dA product is read only by its epilogue chain'
    plain = _lower(ENGINE='p5c', P5C_CONVERT=False, P5C_EPILOGUE=False)
    return {'phases': len(man['phases']), 'phases_without_rules': len(plain.manifest['phases']),
            'tensor_maps': len(man['tensor_maps'])}


def wide_admission_keeps_supported_tail():
    from megakernel_lab.operation_compiler import p5c_matrix as P
    from megakernel_lab.operation_compiler.lower import TensorLayout
    from megakernel_lab.operation_compiler.matrix_codegen import select
    m, n, k = 1792, 448, 512
    decisions = []
    for ak in (False, True):
        for bkc in (False, True):
            assert P.admit(m, n, k, 1, ak, bkc, 0, 0,
                           engine='p5c', min_tiles=P.WIDE_MIN_TILES) is None
            la = TensorLayout((m, k), (k, 1) if ak else (1, m), 'a')
            lb = TensorLayout((k, n), (1, k) if bkc else (n, 1), 'b')
            spec, reason = select('a', 'b', 'out', la, lb, (m, n),
                                  'bfloat16', engine_ok='wide')
            assert spec is not None and not spec.get('engine')
            assert 'admission refused' in reason
            decisions.append({'ak': ak, 'bkc': bkc, 'reason': reason})
    return {'shape': [m, n, k], 'fallbacks': decisions}


def run():
    tests = [popsc_engine_ignores_lane_switches, admission_is_property_based, engine_program_structure,
             wide_admission_keeps_supported_tail]
    results = []
    for test in tests:
        try:
            detail = test()
            results.append({'name': test.__name__, 'passed': True, **({'detail': detail} if detail else {})})
        except Exception as exc:
            import traceback
            results.append({'name': test.__name__, 'passed': False, 'error': f'{type(exc).__name__}: {exc}',
                            'traceback': traceback.format_exc()[-3000:]})
    return {'passed': all(r['passed'] for r in results), 'tests': results}


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=1))
