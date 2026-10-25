"""Modal CPU contracts for P-OPS5-B operation grouping rules (ops/group_rules.py, row_fusion flat members).

* The normal saved-attention loop uses the qualified recurrent rule set. Explicit
  off/on baselines isolate the original grouping mechanism, which does not alter FP32 programs.
* Structure (BF16 combined recurrence): fewer forward/reverse stages; no stage-level phase whose only work is
  widening a BF16 product remains; every absorbed consumer reads its product element-for-element.
* Admission is dependency-safe on synthetic phases: a consumer that also reads a value produced at the
  product's level is refused; one that reads only earlier values is absorbed; the product store is kept when
  any outside phase reads it.
* Rotation only rewrites whole-grid single BF16 products and is reported per region.
The numerical evidence is on the GPU (bitwise equality with the base build at every oracle state).
"""
from __future__ import annotations

import json
import traceback

import torch

from megakernel_lab.operation_compiler.loop import capture_loop_training
from megakernel_lab.operation_compiler.lower import TensorLayout, _index, _ident, lower
from megakernel_lab.operation_compiler import group_rules


def _capture(recipe='primary_bf16', *, objective='combined'):
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES, fixture, initial_state, make_loop_functions, parameters
    from .fixtures.stage3_training import OPTIMIZER
    model = RECIPES[recipe]
    p = parameters(model)
    momentum = {k: torch.zeros_like(v) for k, v in p.items()}
    functions = make_loop_functions(model, intermediate=objective == 'combined')
    return capture_loop_training(*functions, p, momentum, fixture(model), torch.tensor(1), OPTIMIZER, max_steps=3,
                                 state=initial_state(), storage_mode='save_body_tape')


def test_qualified_default_and_fp32_unchanged():
    from megakernel_lab.operation_compiler.execution_policy import RECURRENT_RULES
    captured = _capture()
    assert lower(captured.ir).source == lower(captured.ir, loop_options={'group_rules': RECURRENT_RULES}).source
    fp32 = _capture('primary')
    same = (lower(fp32.ir, loop_options={'group_rules': False}).source
            == lower(fp32.ir, loop_options={'group_rules': True}).source)
    assert same, 'FP32 program changed under group rules'
    return {'qualified_default': RECURRENT_RULES, 'fp32_identical': same}


def test_structure():
    captured = _capture()
    base = lower(captured.ir, execution="legacy", loop_options={'group_rules': False}).manifest
    grouped = lower(captured.ir, execution="legacy", loop_options={'group_rules': True}).manifest
    result = {'splits_base': base['splits'], 'splits_grouped': grouped['splits'], 'regions': {}}
    assert grouped['splits']['forward'] < base['splits']['forward']
    assert grouped['splits']['reverse'] < base['splits']['reverse']
    for name in ('body_saved', 'body_vjp_saved'):
        region = grouped['regions'][name]
        layouts = region['layouts']
        epilogues = [p for p in region['phases'] if p.get('bf16_epilogue')]
        assert epilogues, name
        products = {p['names'][0] for p in region['phases'] if p['kind'] == 'gemm'}
        for phase in region['phases']:
            if phase['kind'] != 'elementwise':
                continue
            widen_only = all(layouts[n]['dtype'] == 'float32' for n in phase['names']) and any(
                layouts[r]['root'] in products for r in phase['reads'])
            assert not widen_only or len(phase['names']) > 1, f'{name}: lone widening phase {phase["names"]}'
        result['regions'][name] = {'stages': len(region['stages']), 'epilogues': len(epilogues),
                                   'rotated': len(region['group_rules'].get('rotated_products', []))}
    return result


def _layout(name, shape, root=None):
    stride, strides = 1, []
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    return TensorLayout(tuple(shape), tuple(reversed(strides)), root or name)


def _product(name, a, b):
    return {'kind': 'gemm', 'names': [name], 'targets': ['aten.mm.default'], 'reads': [a, b], 'writes': [name],
            'engine': 'popsc_bf16_32x32', 'members': [{}], 'matrix_spec': {'dtype': 'bfloat16', 'batch': 1},
            'code': '{\n  popsc::BArgs g{};\n  popsc::bmm_phase<32,32,2,4,true,false,64,5>(g, smem, 0, 1, 1, 0);\n}',
            'tiles': 64}


def _widen(name, source, extra=None):
    at = _index(_layout(source, (64, 64)), (64, 64), 'i')  # the lowering's rendered same-shape read
    expression = f'__bfloat162float(v_{source}[{at}])' + (f' + v_{extra}[{at}]' if extra else '')
    return {'kind': 'elementwise', 'names': [name], 'targets': ['aten._to_copy.default'],
            'reads': [source] + ([extra] if extra else []), 'writes': [name], 'shape': [64, 64], 'dtype': 'float32',
            'body': [f'const float r_{name} = {expression}; v_{name}[i] = r_{name};']}


def test_synthetic_admission():
    dtypes = {'a': 'bfloat16', 'w': 'bfloat16', 'w2': 'bfloat16', 'p': 'bfloat16', 'q': 'bfloat16',
              'c_ok': 'float32', 'c_late': 'float32', 'early': 'float32', 'reader': 'float32'}
    layouts = {n: _layout(n, (64, 64)) for n in dtypes}
    early = {'kind': 'elementwise', 'names': ['early'], 'targets': ['x'], 'reads': [], 'writes': ['early'],
             'shape': [64, 64], 'dtype': 'float32', 'body': ['const float r_early = 1.0f; v_early[i] = r_early;']}
    first, second = _product('p', 'a', 'w'), _product('q', 'a', 'w2')
    for product in (first, second):
        product['reads'].append('early')  # both products sit one level after 'early'
    phases = [early, first, second,
              _widen('c_ok', 'p', 'early'), _widen('c_late', 'p', 'q'),
              {'kind': 'reduction', 'names': ['reader'], 'targets': ['sum'], 'reads': ['p'], 'writes': ['reader'],
               'code': ''}]
    out, records, elided = group_rules.fuse_bf16_epilogues(
        phases, layouts, dtypes, ident=_ident, index=_index, kept=set(), inplace_roots=set(), ctypes={})
    merged = [p for p in out if p.get('bf16_epilogue')]
    assert len(merged) == 1 and merged[0]['bf16_epilogue']['consumers'] == ['c_ok'], records
    assert merged[0]['bf16_epilogue']['product_stored'] is True, 'outside reader requires the product store'
    assert any(p['names'] == ['c_late'] for p in out), 'consumer reading a same-level product must stay separate'
    assert 'popsc::bmm_phase<32,32,2,4,true,false,64,5,true>(g, smem, 0, 1, 1, 0, [=](long long i, __nv_bfloat16 r_p)' \
        in merged[0]['code']
    assert '__bfloat162float(r_p)' in merged[0]['code'] and 'v_p[' not in merged[0]['code'].split('[=]')[1]
    rotated = group_rules.rotate_products(out)
    assert sorted(rotated) == ['p', 'q'] and all('/*C_VBASE*/0' in p['code'] for p in out if p['kind'] == 'gemm')
    return {'absorbed': merged[0]['bf16_epilogue'], 'rotated': rotated, 'phases_after': [p['names'] for p in out]}


def test_split_components_by_level():
    dtypes = {'x': 'float32', 'y': 'float32', 'p': 'bfloat16', 'u': 'float32', 'v': 'bfloat16'}
    layouts = {n: _layout(n, (64, 64)) for n in dtypes}
    producer = {'kind': 'elementwise', 'names': ['x'], 'targets': ['t'], 'reads': [], 'writes': ['x'],
                'shape': [64, 64], 'dtype': 'float32', 'body': ['const float r_x = 0.0f; v_x[i] = r_x;']}
    product = _product('p', 'x', 'x')
    mixed = {'kind': 'elementwise', 'names': ['u', 'v'], 'targets': ['a', 'b'], 'reads': ['p', 'y'],
             'writes': ['u', 'v'], 'shape': [64, 64], 'dtype': 'float32',
             'body': ['const float r_u = __bfloat162float(v_p[i]); v_u[i] = r_u;',
                      'const __nv_bfloat16 r_v = __float2bfloat16_rn(v_y[i]); v_v[i] = r_v;']}
    out, records = group_rules.split_components([producer, product, mixed], dtypes, ctypes={}, layouts=layouts)
    names = [p['names'] for p in out if p['kind'] == 'elementwise']
    assert ['u'] in names and ['v'] in names, names
    assert next(p for p in out if p['names'] == ['v'])['dtype'] == 'bfloat16'
    joined = {'kind': 'elementwise', 'names': ['u', 'v'], 'targets': ['a', 'b'], 'reads': ['p'],
              'writes': ['u', 'v'], 'shape': [64, 64], 'dtype': 'float32',
              'body': ['const float r_u = __bfloat162float(v_p[i]); v_u[i] = r_u;',
                       'const __nv_bfloat16 r_v = __float2bfloat16_rn(r_u); v_v[i] = r_v;']}
    kept, _ = group_rules.split_components([producer, product, joined], dtypes, ctypes={}, layouts=layouts)
    assert [p['names'] for p in kept if p['kind'] == 'elementwise'][-1] == ['u', 'v'], 'dependent chain split'
    return {'split': names, 'records': records}


def test_row_prologue_structure():
    captured = _capture()
    grouped = lower(captured.ir, loop_options={'group_rules': True}).manifest
    prologue = lower(captured.ir, loop_options={'group_rules': 'prologue'}).manifest
    assert prologue['splits']['forward'] < grouped['splits']['forward']
    assert prologue['splits']['reverse'] < grouped['splits']['reverse']
    result = {'splits': prologue['splits'], 'regions': {}}
    for name in ('body_saved', 'body_vjp_saved'):
        region = prologue['regions'][name]
        layouts = region['layouts']
        products = [p for p in region['phases'] if p.get('row_prologue')]
        assert products, name
        for phase in products:
            operand = phase['row_prologue']['operand']
            assert operand not in phase['reads'], 'a prologue product must not read its recomputed operand'
            assert phase['matrix_spec']['k'] == phase['row_prologue']['width']
        rows = [p for p in region['phases'] if p['kind'] == 'row_fused']
        result['regions'][name] = {'prologue_products': len(products), 'row_groups': len(rows),
                                   'shared_bytes': region['shared_bytes']}
    return result


def test_chain_merge_synthetic():
    dtypes = {'p': 'float32', 'g': 'float32', 'm': 'float32', 'buf': 'float32', 'upd': 'float32', 'late': 'float32'}
    layouts = {n: _layout(n, (64, 64)) for n in dtypes}
    at = _index(_layout('x', (64, 64)), (64, 64), 'i')
    late = {'kind': 'reduction', 'names': ['late'], 'targets': ['sum'], 'reads': ['g'], 'writes': ['late'], 'code': ''}
    momentum = {'kind': 'elementwise', 'names': ['buf'], 'targets': ['add'], 'reads': ['g', 'm'], 'writes': ['buf'],
                'shape': [64, 64], 'dtype': 'float32',
                'body': [f'const float r_buf = (v_m[{at}] * 0.9f + v_g[{at}]); v_buf[i] = r_buf;']}
    update = {'kind': 'elementwise', 'names': ['upd'], 'targets': ['sub'], 'reads': ['p', 'buf'], 'writes': ['upd'],
              'shape': [64, 64], 'dtype': 'float32',
              'body': [f'const float r_upd = (v_p[{at}] - v_buf[{at}] * 0.01f); v_upd[i] = r_upd;']}
    blocked = dict(update, names=['upd2'], writes=['upd2'], reads=['p', 'buf', 'late'],
                   body=[f'const float r_upd2 = (v_p[{at}] - v_buf[{at}] * v_late[0]); v_upd2[i] = r_upd2;'])
    layouts['upd2'] = _layout('upd2', (64, 64))
    dtypes['upd2'] = 'float32'
    out, records, elided = group_rules.merge_chains([late, momentum, update], layouts, dtypes, ident=_ident,
                                                    index=_index, kept={'upd'}, inplace_roots=set())
    merged = [p for p in out if p['kind'] == 'elementwise']
    assert len(merged) == 1 and merged[0]['names'] == ['buf', 'upd'], records
    assert 'r_buf * 0.01f' in merged[0]['body'][1] and 'buf' in elided
    out2, records2, _ = group_rules.merge_chains([momentum, late, blocked], layouts, dtypes, ident=_ident,
                                                 index=_index, kept={'upd2'}, inplace_roots=set())
    assert not records2, 'a phase also waiting on a same-level value must not merge into an earlier producer'
    return {'merged': records, 'refused': len(out2)}


TESTS = (test_qualified_default_and_fp32_unchanged, test_structure, test_synthetic_admission, test_split_components_by_level,
         test_row_prologue_structure, test_chain_merge_synthetic)


def run():
    results = []
    for test in TESTS:
        try:
            results.append({'test': test.__name__, 'passed': True, 'details': test()})
        except Exception:
            results.append({'test': test.__name__, 'passed': False, 'error': traceback.format_exc()})
    return {'kind': 'pops5b-group-rules-contract', 'tests': results, 'passed': all(t['passed'] for t in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result['passed'] else 1)
