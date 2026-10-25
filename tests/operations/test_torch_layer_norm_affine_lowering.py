"""CPU-only production lowering checks, with no native compilation."""
# Accepted c24: two independent rows per CTA; affine serial chains and
# their unchanged combine tree are separate phases. Numerical gates live
# in the component/device qualifications and are not changed here.
from copy import deepcopy
import pytest

from megakernel_lab.operation_compiler.lower import LoweringError,lower
from .test_torch_layer_norm_affine import fixture
from megakernel_lab.operation_compiler.torch_layer_norm_backward import AFFINE_KEY,AFFINE_CONTRACT


def selected(program):
    return [p for p in program.manifest['phases'] if p.get('column_arithmetic')==AFFINE_CONTRACT]


def test_primary_affine_both_outputs_and_virtual_geometry():
    g=fixture();before=deepcopy(g);program=lower(g,execution='compose');chains,p=selected(program)
    assert p['names']==p['writes']==['dgain','dbias']
    assert chains['reads']==['x','dy','mean','rstd']
    assert chains['engine']=='torch_cuda_layer_norm_gamma_beta_chains'
    assert chains['names']==chains['writes']==['dgain__affine_chains']
    assert (chains['threads'],chains['shared_bytes'],chains['work_items'])==(256,0,96)
    assert p['reads']==chains['writes']
    assert p['layer_norm_affine_split']['chains']==chains['writes'][0]
    assert list(program.manifest['layouts'][chains['writes'][0]]['shape'])==[2,24,32,32]
    assert (p['threads'],p['logical_threads'],p['shared_bytes'],p['work_items'])==(256,1024,8448,24)
    assert p['layer_norm_affine_plan']['members']==('gain_product','dgain','dbias')
    assert program.manifest['execution_selection']['selected']=='common_static'
    assert 'torch_ln_affine::chains(' in program.source
    assert 'torch_ln_affine::combine(' in program.source
    assert 'torch_layer_norm_affine.cuh' in program.manifest['header_hashes']
    assert len([p for p in program.manifest['phases'] if 'dgain' in p['writes']])==1
    assert not any('gain_product' in p['writes'] for p in program.manifest['phases'])
    assert g==before


def test_default_source_exact_and_escaped_product_preserved():
    g=fixture();g['metadata'].pop(AFFINE_KEY);first=lower(g,execution='compose')
    assert not selected(first)
    assert 'torch_layer_norm_affine.cuh' not in first.manifest['header_hashes']
    h=fixture();h['outputs'].append('gain_product');candidate=lower(h,execution='compose')
    _,p=selected(candidate);assert p['layer_norm_affine_plan']['intermediates']==()
    assert any('gain_product' in p['writes'] for p in candidate.manifest['phases'])
    assert lower(g,execution='compose').source==first.source


def test_reference_refuses_explicit_affine():
    with pytest.raises(LoweringError,match='LayerNorm.*requires'):
        lower(fixture(),execution="legacy")
