"""CPU-worker tests for explicit arithmetic admission and emitted contracts."""
from dataclasses import replace
from pathlib import Path
from importlib.resources import files

import pytest
import torch

from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.matrix_accumulation import CONTRACT as SINGLE
from megakernel_lab.operation_compiler.matrix_adapters.problem import MatrixProblem, Operand
from megakernel_lab.operation_compiler.matrix_adapters.provider import MatrixProvider, problem_key
from megakernel_lab.operation_compiler.matrix_adapters.requirements import ProgramResources
from megakernel_lab.operation_compiler.matrix_splitk import BODY, CONTRACT, KEY, applies, cuts, record, validate
from .test_matrix_accumulation import emit


def problem(k=16384):
    return MatrixProblem(768,32,k,Operand('x',1,768),Operand('dy',32,1))


def test_measured_interval_boundaries_and_narrow_scope():
    assert cuts(16384) == [0,2752,5504,8256,11008,13760,16384]
    assert cuts(65536) == [0,10944,21888,32832,43776,54720,65536]
    assert applies(problem()) and not applies(None)
    assert applies(replace(problem(), out='unrelated', role='not_a_weight'))
    for change in (dict(batch=2),dict(n=64),dict(dtype='float32'),dict(out_dtype='float32'),
                   dict(out_mode='store_f32'),dict(k=768),dict(a=Operand('x',16384,1)),
                   dict(b=Operand('dy',1,16384))):
        assert not applies(replace(problem(),**change))


@pytest.mark.parametrize('k',[0,64,320,16385,False])
def test_bad_intervals_refused(k):
    with pytest.raises(ValueError): cuts(k)


def test_reuses_existing_raw_fp32_body_and_records_exact_order():
    provider=MatrixProvider(matrix_accumulation_contract=SINGLE,matrix_splitk_contract=CONTRACT)
    result=emit(provider,problem())
    assert result['extra']['engine']=='latch_matrix:'+BODY
    assert result['extra'][KEY]==record(problem())
    assert result['extra']['latch_matrix']['bitwise_class']==CONTRACT
    assert result['contract']['requires_full_grid'] and result['contract']['private_scratch']==[]
    from megakernel_lab.operation_compiler.matrix_adapters import splitk as adapter
    if adapter.PARALLEL_SPLITS:
        # LATCH-MOE4-160: six interval warps per 16x32 tile; same fragments and ascending interval sum.
        assert result['extra']['shared_bytes']==adapter.WARPS_SHARED_BYTES==110592
        assert 'latch_matrix_splitk::phase_warps(g,smem);' in result['code']
    else:
        assert result['extra']['shared_bytes']==29696
    assert 'matrix_accumulation_contract' not in result['extra']
    assert 'step13_body' not in result['decision']
    source=(files('megakernel_lab.operation_compiler')/'kernels/matrix_splitk.cuh').read_text()
    assert 'btile_edge<BM,BN,4,2,false,false,KS,NST,false,true,true>' in source
    assert '__fadd_rn(sum[i], partial[' in source
    serial_body, warp_body = source.split('void phase_warps(')
    assert serial_body.count('__float2bfloat16_rn')==1 and warp_body.count('__float2bfloat16_rn')==1


@pytest.mark.parametrize('change',[dict(m=736),dict(k=16385),dict(a=Operand('x',1,768,offset=1))])
def test_matching_but_unsupported_layout_fails_closed(change):
    with pytest.raises(ValueError): emit(MatrixProvider(matrix_splitk_contract=CONTRACT),replace(problem(),**change))


def test_resource_and_explicit_choice_conflicts():
    for resources in (ProgramResources(threads_per_cta=128),ProgramResources(dynamic_shared_limit=16384),
                      ProgramResources(allow_split_k=False)):
        with pytest.raises(ValueError,match='body refused'):
            emit(MatrixProvider(matrix_splitk_contract=CONTRACT,resources=resources),problem())
    for options in (dict(force='popsc_bf16'),dict(choices={problem_key(problem()):'native'})):
        with pytest.raises(ValueError,match='conflicts'):
            emit(MatrixProvider(matrix_splitk_contract=CONTRACT,**options),problem())
    with pytest.raises(ValueError,match='unpadded'):
        emit(MatrixProvider(matrix_splitk_contract=CONTRACT),problem(),out_pitch=40)
    with pytest.raises(ValueError,match='sm_90'):
        emit(MatrixProvider(matrix_splitk_contract=CONTRACT),problem(),target_arch='sm_80')


def test_defaults_broad_contract_and_readout_precedence_preserved():
    assert validate(None) is None and validate('legacy') is None
    with pytest.raises(ValueError): validate('guess')
    assert emit(MatrixProvider(),problem())==emit(MatrixProvider(matrix_splitk_contract='legacy'),problem())
    wide=replace(problem(),n=64,b=Operand('dy',64,1))
    assert emit(MatrixProvider(matrix_accumulation_contract=SINGLE),wide)==emit(
        MatrixProvider(matrix_accumulation_contract=SINGLE,matrix_splitk_contract=CONTRACT),wide)
    from megakernel_lab.operation_compiler.serial_bf16 import MMA_K8_CONTRACT
    serial=dict(products={'dw':dict(contract=MMA_K8_CONTRACT,a='a',b='b',shape=[768,32,16384],
                                  role='weight_partial',partition_k=0)})
    assert emit(MatrixProvider(serial_bf16=serial),problem())==emit(
        MatrixProvider(serial_bf16=serial,matrix_splitk_contract=CONTRACT),problem())


def test_capture_selection_and_default_source_identity():
    torch.set_num_threads(1)
    capture=capture_inference(lambda _p,x,dy:x.t()@dy,{},
        (torch.zeros(1024,64,dtype=torch.bfloat16),torch.zeros(1024,32,dtype=torch.bfloat16)))
    options=dict(execution='compose',target_arch='sm_90',vocabulary_head='off')
    legacy=lower(capture.ir,**options)
    capture.ir.metadata[KEY]=CONTRACT
    current=lower(capture.ir,**options)
    phases=[p for p in current.manifest['phases'] if KEY in p]
    assert len(phases)==1 and phases[0][KEY]['cuts']==cuts(1024)
    assert current.manifest['execution_selection']['selected']=='common_static'
    assert 'latch_matrix_splitk::phase' in current.source
    del capture.ir.metadata[KEY]
    assert lower(capture.ir,**options).source==legacy.source


def test_warp_interval_body_keeps_fragments_and_ascending_interval_sum():
    source=(files('megakernel_lab.operation_compiler')/'kernels/matrix_splitk.cuh').read_text()
    body=source[source.index('void phase_warps('):]
    assert 'for (int split = 0; split < 6; ++split) sum = __fadd_rn(sum, partial[split * WBM * BN + index]);' in body
    assert 's101h::bmma(acc[j], af[0], af[1], af[2], af[3], bb[0], bb[1]);' in body
    assert '__float2bfloat16_rn(sum)' in body
