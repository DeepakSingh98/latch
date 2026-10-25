"""CPU-only source/admission tests for the frozen matrix donor assembly.

Adapted from 02e43acc80 contracts.py/widen_contracts.py, paired with compiler
6c72bf2c6d and stacked-TMA prerequisite 8292d176. No CUDA compilation or tensors.
"""
import copy
from dataclasses import replace
import importlib

import pytest

from megakernel_lab.operation_compiler import matrix_widen, p5c_matrix
from megakernel_lab.operation_compiler.lower import TensorLayout, lower
from megakernel_lab.operation_compiler.matrix_adapters import dense, select
from megakernel_lab.operation_compiler.matrix_adapters.bodies import BY_NAME, _popsc_spec

selector = importlib.import_module('megakernel_lab.operation_compiler.matrix_adapters.candidates')


def graph_for(problem, widen=False):
    values = {'A': dict(shape=problem.a_shape(), strides=problem.a_strides(), dtype='bfloat16', role='input'),
              'B': dict(shape=problem.b_shape(), strides=problem.b_strides(), dtype='bfloat16', role='input'),
              'C': dict(shape=problem.out_shape(), dtype='bfloat16')}
    operations = [dict(name='C', target='aten.bmm.default' if problem.batch > 1 else 'aten.mm.default',
                       args=[{'ref': 'A'}, {'ref': 'B'}], kwargs={})]
    if widen:
        values['D'] = dict(shape=problem.out_shape(), dtype='float32')
        operations.append(dict(name='D', target='aten._to_copy.default', args=[{'ref': 'C'}], kwargs={'dtype': 'float32'}))
    return dict(inputs=['A', 'B'], outputs=['D' if widen else 'C'], values=values,
                operations=operations, metadata={}, state_writes={})


def spec_for(problem):
    return dict(_popsc_spec(problem, problem.a.root, problem.b.root, 'v_C'), engine='p5c',
                roots=[(problem.a.root, problem.a.offset), (problem.b.root, problem.b.offset)], ws_ok=True)


def test_short_k_store_selection_and_feature_off(monkeypatch):
    p = dense(65536, 50304, 384, ak=True, bkc=True)
    selected = []
    for enabled in (False, True):
        monkeypatch.setattr(selector, 'STORE_BOUND_WS', enabled)
        selected.append(select(p, policy='step13')[0].body)
    assert selected == ['pops4b_tma', 'p8ws']
    for p in (dense(65536, 1152, 384), dense(65536, 50304, 512), dense(4096, 384, 384, batch=32)):
        outcomes = []
        for enabled in (False, True):
            monkeypatch.setattr(selector, 'STORE_BOUND_WS', enabled)
            c, _ = select(p, policy='step13'); outcomes.append((c.body, c.plan))
        assert outcomes[0] == outcomes[1]


def test_cooperative_chunks_keep_order_and_scope(monkeypatch):
    spec = spec_for(dense(50304, 384, 65536, ak=False, bkc=False))
    monkeypatch.setattr(p5c_matrix, 'P5C_COOPERATIVE_CHUNKS', True)
    plan = p5c_matrix.plan(spec, 132)
    code = p5c_matrix.code(spec, plan, lambda n: 'v_'+n, 0, 1, 1, 0)
    assert plan['chunk_stages'] == 32 and not plan.get('ws') and plan['chunk_selection']
    assert 'p5c::phase_k<' in code
    monkeypatch.setattr(p5c_matrix, 'P5C_COOPERATIVE_CHUNKS', False)
    assert p5c_matrix.plan(spec, 132)['ws_engine'] == 'p8ws_flat'
    monkeypatch.setattr(p5c_matrix, 'P8WS', False)
    reference = p5c_matrix.plan(spec, 132)
    assert p5c_matrix.code(spec, reference, lambda n: 'v_'+n, 0, 1, 1, 0) == code
    monkeypatch.setattr(p5c_matrix, 'P5C_COOPERATIVE_CHUNKS', True)
    for extra in ({'pointers': ('a', 'b')}, {'splitk': 2}, {'out_raw': True}, {'virtual_a': True},
                  {'virtual_b': True}, {'p9_delta': True}, {'ws_ok': False}, {'bkc': True},
                  {'m': 1536, 'n': 512}, {'m': 65472}):
        assert not p5c_matrix.prefer_cooperative_chunks(dict(spec, **extra), 132)
    assert not p5c_matrix.prefer_cooperative_chunks(dict(spec, m=128*127, n=512), 8)
    assert p5c_matrix.prefer_cooperative_chunks(dict(spec, m=128*128, n=512), 8)


@pytest.mark.parametrize('k', [128, 256, 384])
def test_actual_p8ws_ring_and_group_stage_count(k):
    spec = spec_for(dense(64, 128, k))
    plan = p5c_matrix.flat_ws_plan(spec, 132)
    stages = min(k//64, 5)
    assert plan['nst'] == stages
    assert plan['shared_bytes'] == stages * 256 * 128 + 2 * 128 * 128 * 2 + 1024
    code = p5c_matrix.group_code([spec], [plan], [0], lambda n: 'v_'+n)
    assert f'p8ws::group<0,{stages},' in code


@pytest.mark.parametrize('shape,bkc', [((32,4096,768,384), True), ((32,4096,384,768), False),
                                    ((3,16512,448,192), True), ((3,16512,448,192), False)])
def test_normal_api_stacked_widen_preserves_plan_and_rounding(monkeypatch, shape, bkc):
    batch,m,n,k = shape
    p = dense(m,n,k,batch=batch,ak=True,bkc=bkc)
    programs = []
    for enabled in (False, True):
        monkeypatch.setattr(matrix_widen, 'BATCH_TMA_WIDEN', enabled)
        programs.append(lower(graph_for(p, True), target_arch='sm_90', execution='compose', vocabulary_head='off'))
    base,candidate = programs
    old, = [phase for phase in base.manifest['phases'] if phase['kind']=='gemm']
    fused, = [phase for phase in candidate.manifest['phases'] if phase.get('batch_widening')]
    assert old['engine'] == fused['engine'] == 'latch_matrix:pops4b_tma'
    assert old['matrix_plan'] == fused['matrix_plan'] and old['shared_bytes'] == fused['shared_bytes']
    assert fused['writes'] == ['D'] and fused['batch_widening']['extra_shared_bytes'] == 0
    assert fused['batch_stacked_tma']['rows_per_batch'] == m
    assert f'g.bgrp = {m}; g.bstep = {n if bkc else k};' in candidate.source
    assert 'make_float2(x, y)' in candidate.source and 'C' not in candidate.manifest['offsets']
    assert base.manifest['arena_bytes'] - candidate.manifest['arena_bytes'] == batch*m*n*2


def widen_fixture():
    p = dense(16384,384,192,batch=32,ak=True,bkc=True)
    body = BY_NAME['pops4b_tma']; plan = body.plans(p,132)[0]
    shape = p.out_shape(); strides=(16384*384,384,1)
    layouts={n:TensorLayout(shape,strides,n) for n in ('C','D')}
    phases=[dict(kind='gemm',names=['C'],targets=['aten.bmm.default'],reads=['A','B'],writes=['C'],
                 batch_widen_respec=dict(spec=body._spec(p,'A','B','v_C'),plan=plan,rot=0)),
            dict(kind='elementwise',names=['D'],targets=['aten._to_copy.default'],reads=['C'],writes=['D'],
                 body=['const float r_D = __bfloat162float(v_C[i]); v_D[i] = r_D;'])]
    return phases,layouts,{'C':'bfloat16','D':'float32'}


@pytest.mark.parametrize('case', ['retained','mutated_input','mutated_output','alias_reader',
                                 'nonidentity','shifted','nonstacked','serial'])
def test_widen_refuses_escaping_or_changed_arithmetic(case):
    phases,layouts,dtypes=widen_fixture();kept=set();inplace=set()
    if case=='retained':kept.add('C')
    elif case=='mutated_input':inplace.add('C')
    elif case=='mutated_output':inplace.add('D')
    elif case=='alias_reader':
        layouts['view']=replace(layouts['C'],root='C')
        phases.append(dict(kind='elementwise',reads=['view'],writes=['other']))
    elif case=='nonidentity':phases[1]['body'][0]=phases[1]['body'][0].replace('__bfloat162float','2.f * __bfloat162float')
    elif case=='shifted':layouts['D']=replace(layouts['D'],offset=1)
    elif case=='nonstacked':
        spec=phases[0]['batch_widen_respec']['spec'];spec['strides']=list(spec['strides']);spec['strides'][4]+=8
    elif case=='serial':phases[0]['batch_widen_respec']['spec']['serial_bf16_k']=64
    result,records,dead=matrix_widen.fuse(phases,layouts,dtypes,ident=lambda n:'v_'+n,index=lambda *_:'i',
                                        kept=kept,inplace_roots=inplace)
    assert not records and not dead and result==phases


def test_stacked_admission_rejects_cross_batch_tiles_and_serial_contract():
    from megakernel_lab.operation_compiler.matrix_rules_p7b import batch_stacked, batch_stacked_tiles
    p=dense(4096,768,384,batch=32,ak=True,bkc=True)
    spec=BY_NAME['pops4b_tma']._spec(p,'A','B','v_C');plan=BY_NAME['pops4b_tma'].plans(p,132)[0]
    assert batch_stacked(p.m,p.n,p.k,p.batch,p.ak,p.bkc,spec['strides'])
    assert batch_stacked_tiles(spec,plan)
    assert not batch_stacked_tiles(dict(spec,serial_bf16_k=64),plan)
    assert not batch_stacked_tiles(dict(spec,m=p.m-1),plan)
    assert not batch_stacked_tiles(spec,plan,dict(staged=True,store=True))
