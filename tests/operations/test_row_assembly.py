"""CPU source/admission checks for the frozen c2a325e441 ROW scheduling port.

No donor imports, tensors, CUDA compilation, or claims of GPU qualification.
"""
import copy
import json

import pytest

from megakernel_lab.operation_compiler import compose, group_rules, row_fusion
from megakernel_lab.operation_compiler.lower import TensorLayout, _ident, _index, lower
from megakernel_lab.operation_compiler.torch_row_reduction import CONTRACT


def layout(name, shape):
    strides, stride = [], 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return TensorLayout(tuple(shape), tuple(reversed(strides)), name)


def chain_fixture(size=1):
    first = dict(kind='reduction', names=['late'], targets=['sum'], reads=['x'], writes=['late'], code='')
    chain = dict(kind='elementwise', names=['early', 'last'], targets=['mul', 'add'],
                 reads=['x', 'late'], writes=['early', 'last'], shape=[size], dtype='float32',
                 body=['const float r_early = v_x[i] * 2.0f; v_early[i] = r_early;',
                       'const float r_last = r_early + v_late[i]; v_last[i] = r_last;'],
                 registers={'early': 'r_early', 'last': 'r_last'})
    return [first, chain], dict.fromkeys(('x', 'late', 'early', 'last'), 'float32')


def test_small_chain_split_preserves_expressions_and_crossing_storage():
    phases, dtypes = chain_fixture()
    old, _ = group_rules.split_components(copy.deepcopy(phases), dtypes, ctypes={})
    assert old[-1] == phases[-1]
    new, records = group_rules.split_components(copy.deepcopy(phases), dtypes, ctypes={}, level_split=True)
    assert [p['names'] for p in new] == [['late'], ['early'], ['last']]
    assert new[1]['body'] == phases[1]['body'][:1]
    assert new[2]['body'] == [phases[1]['body'][1].replace('r_early', 'v_early[i]')]
    assert new[1]['reads'] == ['x'] and new[2]['reads'] == ['late', 'early']
    assert new[1]['registers'] == {'early': 'r_early'}
    assert new[2]['registers'] == {'last': 'r_last'}
    assert records[-1]['level_split'] == [['early'], ['last']]


@pytest.mark.parametrize('case', ['large', 'no_store', 'rounded_store', 'wrong_name', 'memory_dependency',
                                  'forward_register', 'same_level'])
def test_chain_refusals_preserve_existing_result(case):
    phases, dtypes = chain_fixture(65537 if case == 'large' else 1)
    body = phases[1]['body']
    if case == 'no_store':
        body[0] = 'const float r_early = v_x[i] * 2.0f;'
    elif case == 'rounded_store':
        body[0] = 'const float r_early = v_x[i] * 2.0f; v_early[i] = __float2bfloat16_rn(r_early);'
    elif case == 'wrong_name':
        body[0] = body[0].replace('r_early =', 'r_other =')
    elif case == 'memory_dependency':
        body[1] = body[1].replace('r_early +', 'v_early[0] +')
    elif case == 'forward_register':
        body[0] = body[0].replace('v_x[i]', 'r_last')
    elif case == 'same_level':
        phases = phases[1:]
    base = group_rules.split_components(copy.deepcopy(phases), dtypes, ctypes={})
    candidate = group_rules.split_components(copy.deepcopy(phases), dtypes, ctypes={}, level_split=True)
    assert candidate == base


def row_fixture(rows=128, width=128, *, random=False):
    full, reduced = (rows, width), (rows, 1)
    layouts = {n: layout(n, full) for n in ('x', 'square')}
    layouts['mean'] = layout('mean', reduced)
    at = _index(layouts['x'], full, 'i')
    expression = f'v_x[{at}] * v_x[{at}]'
    if random:
        expression += ' * latch_philox_mask(v_rng, i, 1ULL, 0.5f)'
    phases = [dict(kind='elementwise', names=['square'], targets=['aten.mul.Tensor'], reads=['x'],
                   writes=['square'], shape=list(full), dtype='float32',
                   body=[f'const float r_square = {expression}; v_square[i] = r_square;']),
              dict(kind='reduction', names=['mean'], targets=['aten.mean.dim'], reads=['square'],
                   writes=['mean'], code='', row_reduction=dict(source='square', width=width, rows=rows,
                                                               scale=f' * (1.0f / {width}.0f)'))]
    return phases, layouts, dict.fromkeys(layouts, 'float32')


def fused(fixture, **kwargs):
    phases, layouts, dtypes = fixture
    return row_fusion.fuse_rows(copy.deepcopy(phases), layouts, dtypes, ident=_ident, index=_index,
                               kept={'mean'}, inplace_roots=kwargs.pop('inplace_roots', set()),
                               hoist=kwargs.pop('hoist', True), **kwargs)


def test_pipeline_preserves_arithmetic_and_plain_epilogue_form():
    fixture = row_fixture()
    base, _, dead = fused(fixture)
    candidate, records, candidate_dead = fused(fixture, pipe=True)
    old, new = base[0], candidate[0]
    assert old['kind'] == new['kind'] == 'row_fused'
    assert isinstance(new['code_r1'], str) and new['code_r1'] == old['code']
    for key in ('names', 'targets', 'reads', 'writes', 'row_parts', 'work_items', 'reduction_elements'):
        assert new[key] == old[key]
    assert candidate_dead == dead
    assert records[0]['next_row_pipelined'] is True
    assert 'h0_[w] = p0_[w];' in new['code']
    assert 'if (on_ < 128LL)' in new['code'] and 'if (o < 128LL)' in new['code']
    assert new['code'].index('h0_[w] = p0_[w];') < new['code'].index('const float r_square =')
    assert new['code'][new['code'].index('  float v[8];'):] == old['code'][old['code'].index('  float v[8];'):] + '\n}'


@pytest.mark.parametrize('case', ['unhoisted', 'inplace', 'multiwarp', 'philox'])
def test_pipeline_refusals_are_source_identical(case):
    fixture = row_fixture(rows=128 if case == 'multiwarp' else 512, random=case == 'philox')
    kwargs = dict(hoist=False) if case == 'unhoisted' else dict(inplace_roots={'square'}) if case == 'inplace' \
        else dict(multi_warp=8) if case == 'multiwarp' else {}
    assert fused(fixture, **kwargs) == fused(fixture, pipe=True, **kwargs)


def reduction_graph(*, contract=False):
    return dict(inputs=['x'], outputs=['mean'], state_writes={},
                values={'x': dict(shape=[128,768], dtype='float32', role='input'),
                        'square': dict(shape=[128,768], dtype='float32'),
                        'mean': dict(shape=[128,1], dtype='float32')},
                operations=[dict(name='square', target='aten.mul.Tensor', args=[{'ref':'x'}, {'ref':'x'}], kwargs={}),
                            dict(name='mean', target='aten.mean.dim', args=[{'ref':'square'}, [-1], True], kwargs={})],
                metadata={'row_reduction_contract': CONTRACT} if contract else {})


def test_normal_compiler_row_toggle_and_protected_reduction(monkeypatch):
    programs = []
    protected = []
    for enabled in (False, True, False):
        monkeypatch.setattr(compose, 'ROW_RULES', enabled)
        features = group_rules.features_of(compose.MOESUP_FLAT_RULES())
        assert {'rpipe', 'lsplit', 'lptr', 'lsreg', 'deepred'} <= features if enabled else not features & {'rpipe', 'lsplit'}
        programs.append(lower(reduction_graph(), execution='compose', target_arch='sm_90', vocabulary_head='off'))
        protected.append(lower(reduction_graph(contract=True), execution='compose', target_arch='sm_90', vocabulary_head='off'))
    assert programs[0].source == programs[2].source
    selected = [p for p in programs[1].manifest['phases'] if p.get('row_fusion', {}).get('next_row_pipelined')]
    print(json.dumps({'selection': programs[1].manifest.get('execution_selection'),
                      'phases': [{k:p.get(k) for k in ('kind','names','engine','row_fusion','row_reduction')}
                                 for p in programs[1].manifest['phases']]}, default=str))
    assert len(selected) == 1
    for program in protected:
        exact = [p for p in program.manifest['phases'] if p.get('row_arithmetic') == CONTRACT]
        assert len(exact) == 1 and exact[0]['engine'] == 'torch_cuda_vector4_mean'
        assert all('mean' not in p['names'] for p in program.manifest['phases'] if p.get('row_fusion'))
    assert protected[0].source == protected[2].source


def test_provider_addition_does_not_remove_existing_bodies():
    assert set(('pops4b_tma', 'pops4b_wg', 'popsc_bf16_edge', 'p8ws')) <= set(compose.PROVIDER_BODIES)


@pytest.mark.parametrize('shape', [[], [64]])
def test_no_admitted_pipeline_preserves_legacy_selection_and_source(monkeypatch, shape):
    graph=dict(inputs=['x'],outputs=['y'],state_writes={},metadata={},
               values={'x':dict(shape=shape,dtype='float32',role='input'),
                       'y':dict(shape=shape,dtype='float32')},
               operations=[dict(name='y',target='aten.mul.Tensor',args=[{'ref':'x'},2.0],kwargs={})])
    programs=[]
    for enabled in (False,True):
        monkeypatch.setattr(compose,'ROW_RULES',enabled)
        programs.append(lower(graph,execution='compose',target_arch='sm_90',vocabulary_head='off'))
    for program in programs:
        selection=program.manifest['execution_selection']
        assert selection['selected']=='legacy_static'
        assert selection['admitted_capabilities']['pipelined_rows']==0
    assert programs[0].source==programs[1].source
