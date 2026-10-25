"""CPU ownership and exact-rounding contracts for k8 direct widening."""
import copy

from megakernel_lab.operation_compiler.k8_widen import fuse
from megakernel_lab.operation_compiler.lower import TensorLayout, _ident, _index


def fixture(tail=False):
    names = ['a', 'b', 'p', 'f', 'prior', 'sum']
    layouts = {n: TensorLayout((128, 256), (256, 1), n) for n in names}
    dtypes = {n: 'bfloat16' if n in ('a', 'b', 'p') else 'float32' for n in names}
    at = _index(layouts['p'], (128,256))
    phases = [dict(kind='gemm', reads=['a', 'b'], writes=['p'], names=['p'],
        targets=['aten.mm.default'], code='g.C = reinterpret_cast<s101h::bf16*>(v_p); /*K8_ROUNDED_F32*/false',
        matrix_spec=dict(out='v_p'),
        k8_widen_respec=dict(output='v_p', shape=[128, 256], pitch=256, enabled=True)),
        dict(kind='elementwise', reads=['p'], writes=['f'], names=['f'], shape=(128, 256),
            targets=['aten._to_copy.default'], registers={'f': 'r_f'},
            body=[f'const float r_f = __bfloat162float(v_p[{at}]); v_f[i] = r_f;'])]
    if tail:
        phases[1]['names'].append('sum')
        phases[1]['writes'].append('sum')
        phases[1]['reads'] += ['prior', 'f']
        phases[1]['targets'].append('aten.add.Tensor')
        phases[1]['body'].append(f'const float r_sum = (v_prior[{at}] + (r_f * 1.0f)); v_sum[i] = r_sum;')
    return phases, layouts, dtypes


def apply(parts, **kw):
    phases, layouts, dtypes = parts
    return fuse(phases, layouts, dtypes, ident=_ident, index=_index,
                kept=kw.get('kept', set()), inplace_roots=kw.get('inplace_roots', set()))


def test_private_identity_widening_uses_exact_rounded_fp32_store():
    parts = fixture()
    before = copy.deepcopy(parts[0])
    phases, records, dead = apply(parts, kept={'f'})
    assert parts[0] == before
    assert len(phases) == len(records) == 1 and dead == {'p'}
    assert phases[0]['writes'] == ['f']
    assert 'reinterpret_cast<s101h::bf16*>(v_f)' in phases[0]['code']
    assert '/*K8_ROUNDED_F32*/true' in phases[0]['code']
    assert records[0]['avoided_bf16_store_and_load_bytes'] == 128 * 256 * 4


def test_widened_value_feeds_remaining_original_addition():
    phases, records, dead = apply(fixture(tail=True))
    assert len(phases) == 2 and dead == {'p'}
    assert phases[1]['names'] == phases[1]['writes'] == ['sum']
    assert f'v_f[{_index(TensorLayout((128,256),(256,1),"f"),(128,256))}]' in phases[1]['body'][0]
    assert 'r_f' not in phases[1]['body'][0]
    assert set(phases[1]['reads']) == {'prior', 'f'}


def test_product_escape_layout_or_arithmetic_change_refused():
    for kind in ('export', 'reader', 'pitch', 'partition', 'cast', 'inplace'):
        parts = fixture()
        kw = {}
        if kind == 'export':
            kw['kept'] = {'p'}
        elif kind == 'reader':
            parts[0].append(dict(kind='diagnostic', reads=['p']))
        elif kind == 'pitch':
            parts[0][0]['k8_widen_respec']['pitch'] = 264
        elif kind == 'partition':
            parts[0][0]['k8_widen_respec']['enabled'] = False
        elif kind == 'cast':
            parts[0][1]['body'][0] = parts[0][1]['body'][0].replace('= __bfloat162float(', '= 2.0f * __bfloat162float(')
        else:
            kw['inplace_roots'] = {'f'}
        phases, records, dead = apply(parts, **kw)
        assert phases == parts[0] and records == [] and dead == set()


def test_private_bf16_storage_uses_in_remaining_consumer_are_refused():
    for storage in ('source', 'alias', 'offset_alias'):
        for use in ('load', 'pointer_cast', 'register'):
            parts = fixture()
            phases, layouts, dtypes = parts
            source = 'p'
            if storage != 'source':
                source = 'private_view'
                offset = 256 if storage == 'offset_alias' else 0
                layouts[source] = TensorLayout((127 if offset else 128, 256), (256, 1), 'p', offset)
                dtypes[source] = 'bfloat16'
            layouts['escaped'] = TensorLayout((128, 256), (256, 1), 'escaped')
            dtypes['escaped'] = 'float32'
            consumer = phases[-1]
            consumer['names'].append('escaped'); consumer['writes'].append('escaped')
            consumer['reads'].append(source)
            consumer['targets'].append('aten._to_copy.default')
            if use == 'load':
                read = f'__bfloat162float({_ident(source)}[(i)])'
            elif use == 'pointer_cast':
                read = f'float(reinterpret_cast<const unsigned short*>({_ident(source)})[i])'
            else:
                read = f'__bfloat162float(r_{source})'
            consumer['body'].append(f'const float r_escaped = {read}; v_escaped[i] = r_escaped;')
            before = copy.deepcopy(parts)
            result, records, dead = apply(parts, kept={'escaped'})
            assert parts == before and result == before[0] and not records and not dead, (storage, use)
