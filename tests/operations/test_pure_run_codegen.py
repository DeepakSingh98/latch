"""Pure-run byte inverse, retained resource boundaries and six-value ABI."""
import copy
import hashlib
import re

import pytest
from megakernel_lab.operation_compiler import elementwise_codegen as ec
from megakernel_lab.operation_compiler import mixed_phase_pointer_codegen as mixed
from megakernel_lab.operation_compiler import resource_phase_pointer_codegen as resource
from megakernel_lab.operation_compiler import pure_interval_codegen as pi
from megakernel_lab.operation_compiler.pure_run_codegen import fold, KEY, Refusal
from .fixtures.tinymoe_source_proofs import close_brace

from .test_phase_pointer_codegen import fixture as base_fixture


def fixture(count=100, counters=(0,33,66)):
    p=base_fixture(count)
    old='if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();'
    tail='if (ticks && lsr_bx == 0 && lsr_tx == 0) { const unsigned long long t = stamp(); '
    tail+=' '.join(f'ticks[{i+1}] = t;' for i in range(count))+' }'
    p.source=p.source.replace(old,tail)
    for i,a,b in reversed(list(ec._phase_bodies(p.source))):
        if i not in counters:continue
        body=f'if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_y_{i}) = 0u;\n'
        p.source=p.source[:a]+body+p.source[b:]
        p.manifest['phases'][i]=dict(kind='reduction_counter_init',names=[f'y_{i}'],writes=[f'y_{i}'],reads=[],targets=[])
    return pi.fold(resource.fold(mixed.fold(p)))


def independently_expand(before,after):
    source=after.source
    pattern=re.compile(r'static __device__ __noinline__ void (latch_prun_body\d+)\(float\*\* args, float\* arena, unsigned lsr_tx, unsigned lsr_bx, unsigned lsr_nt, unsigned lsr_nb\) \{\n')
    found=list(pattern.finditer(source));groups=after.manifest[KEY]['groups']
    assert {m[1] for m in found}=={g['helper'] for g in groups} and len(found)==len(groups)
    bodies={}
    for m in reversed(found):
        end=close_brace(source,m.end()-2)
        assert source[end-1:end+2]=='\n}\n'
        bodies[m[1]]=source[m.end():end-1]
        source=source[:m.start()]+source[end+2:]
    for g in groups:
        code=bodies[g['helper']]
        assert hashlib.sha256(code.encode()).hexdigest()==g['body_sha256']
        assert g['parameter_types']==['float**','float*','unsigned','unsigned','unsigned','unsigned']
        assert g['argument_names']==['args','arena','lsr_tx','lsr_bx','lsr_nt','lsr_nb']
        call=g['helper']+'(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);'
        assert source.count(call)==1
        source=source.replace(call,code)
    assert source==before.source,'prior helpers, source bodies, actual bindings, resources or caller boundaries changed'
    assert {k:v for k,v in before.manifest.items() if k!='source_hash'}=={
        k:v for k,v in after.manifest.items() if k not in (KEY,'source_hash')}


def test_three_runs_move_with_resource_bodies_and_boundary_barriers_unchanged():
    before=fixture();after=fold(before);proof=after.manifest[KEY]
    assert proof['outlined_runs']==3 and proof['outlined_phases']==97
    assert [g['phase_count'] for g in proof['groups']]==[32,32,33]
    assert [(g['previous_phase'],g['next_phase']) for g in proof['groups']]==[(0,33),(33,66),(66,None)]
    independently_expand(before,after)
    entry=after.source[after.source.index('extern "C" __global__'):]
    assert entry.count('__syncthreads();')==5
    for i in (0,33,66):assert f'*reinterpret_cast<unsigned*>(v_y_{i}) = 0u;' in entry
    assert 'grid.sync();' not in after.source[:after.source.index('extern "C" __global__')]
    assert all(after.manifest[k]==before.manifest[k] for k in (pi.KEY,resource.KEY,mixed.KEY))


@pytest.mark.parametrize('count,expected',[(32,0),(33,1)])
def test_exact_minimum_run_length(count,expected):
    before=fixture(count,counters=(0,));after=fold(before)
    assert after.manifest[KEY]['outlined_runs']==expected
    independently_expand(before,after)


@pytest.mark.parametrize('suffix',['return;','break;','continue;','goto bad;',
    'unknown(v_y_1);','unknown(&v_y_1);','float* escaped=v_y_1;',
    '__shared__ float sm[4];','atomicAdd(v_y_1,1.f);','lsr_tx = 5;',
    'unknown(&lsr_bx);','unsigned& escaped = lsr_nt;','float local[2];'])
def test_unsupported_phase_terminates_run_without_changing_its_bytes(suffix):
    before=fixture();marker='// latch_mixed_phase_pointers: 1\n'
    a=before.source.index(marker);b=before.source.index('// latch_mixed_phase_pointers: end\n',a)+len('// latch_mixed_phase_pointers: end\n')
    before.source=before.source[:b]+suffix+'\n'+before.source[b:]
    after=fold(before)
    assert after.manifest[KEY]['outlined_runs']==2
    assert all(1 not in g['phases'] for g in after.manifest[KEY]['groups'])
    independently_expand(before,after)


@pytest.mark.parametrize('mutation',['scalar_type','scalar_value','missing_scalar','outer_reference','outer_qualified',
    'barrier','pre_run_barrier','publication','outer_tensor_use','output_base'])
def test_caller_or_capture_mutations_are_refused(mutation):
    before=fixture()
    if mutation=='scalar_type':before.source=before.source.replace('unsigned lsr_tx =','int lsr_tx =',1)
    elif mutation=='scalar_value':before.source=before.source.replace('lsr_tx = threadIdx.x','lsr_tx = blockIdx.x',1)
    elif mutation=='missing_scalar':before.source=re.sub(r'unsigned lsr_tx =[^\n]*\n','',before.source,count=1)
    elif mutation=='outer_reference':before.source=before.source.replace('// Phase 0:','float& shared = arena[0];\n// Phase 0:',1)
    elif mutation=='outer_qualified':before.source=before.source.replace('float* v_x_0 =','const float* v_x_0 =',1)
    elif mutation in ('barrier','pre_run_barrier'):before.source=before.source.replace('__syncthreads();','__syncthreads();\nside_effect();',1)
    elif mutation=='publication':before.source=before.source.replace('ticks[1] = t;','ticks[1] = t+1;',1)
    elif mutation=='outer_tensor_use':before.source=before.source.replace('v_x_1[j]','v_x_0[j]',1)
    else:before.source=before.source.replace('float* v_x_1 = latch_pa[1];','float* v_x_1 = latch_po[1];',1)
    after=fold(before)
    if mutation in ('outer_tensor_use','output_base'):
        assert all(1 not in g['phases'] for g in after.manifest[KEY]['groups'])
    else:assert after.manifest[KEY]['outlined_runs']==0 and after.source==before.source
    independently_expand(before,after)


@pytest.mark.parametrize('change',['body','signature','call','scalar_order','scalar_type','caller_scalar',
    'binding_rhs','resource','boundary_barrier','publication','manifest','parameter_escape'])
def test_independent_inverse_rejects_actual_emitted_corruption(change):
    before=fixture();after=fold(before)
    if change=='body':after.source=after.source.replace('v_x_1[j] * 2.0f','v_x_1[j] * 3.0f',1)
    elif change=='signature':after.source=after.source.replace('void latch_prun_body0(float** args','void latch_prun_body0(float* args',1)
    elif change=='call':after.source=after.source.replace('latch_prun_body0(args, arena,','latch_prun_body0(out, arena,',1)
    elif change=='scalar_order':after.source=after.source.replace('latch_prun_body0(args, arena, lsr_tx, lsr_bx,','latch_prun_body0(args, arena, lsr_bx, lsr_tx,',1)
    elif change=='scalar_type':after.source=after.source.replace('unsigned lsr_nt,','int lsr_nt,',1)
    elif change=='caller_scalar':after.source=after.source.replace('lsr_tx = threadIdx.x','lsr_tx = blockIdx.x',1)
    elif change=='binding_rhs':after.source=after.source.replace('v_x_1 = latch_pa[1]','v_x_1 = latch_pa[2]',1)
    elif change=='resource':after.source=after.source.replace('v_y_0) = 0u','v_y_0) = 1u',1)
    elif change=='boundary_barrier':
        a=after.source.index('extern "C" __global__');after.source=after.source[:a]+after.source[a:].replace('__syncthreads();','',1)
    elif change=='publication':after.source=after.source.replace('ticks[1] = t;','ticks[1] = t+1;',1)
    elif change=='parameter_escape':after.source=after.source.replace('// Phase 1:','unknown(&lsr_tx);\n// Phase 1:',1)
    else:after.manifest=copy.deepcopy(after.manifest);after.manifest[pi.KEY]['proof']='changed'
    with pytest.raises(AssertionError):independently_expand(before,after)


@pytest.mark.parametrize('argument',range(6))
@pytest.mark.parametrize('location',['signature_name','signature_type','call_actual'])
def test_every_abi_argument_identity_type_and_position_is_independently_bound(argument,location):
    before=fixture();after=fold(before)
    names=['args','arena','lsr_tx','lsr_bx','lsr_nt','lsr_nb']
    types=['float**','float*','unsigned','unsigned','unsigned','unsigned']
    if location=='call_actual':
        old='latch_prun_body0('+', '.join(names)+');'
        names[argument]=names[(argument+1)%6]
        new='latch_prun_body0('+', '.join(names)+');'
    else:
        old='void latch_prun_body0('+', '.join(t+' '+n for t,n in zip(types,names))+')'
        if location=='signature_name':names[argument]+='_different'
        else:types[argument]='int'
        new='void latch_prun_body0('+', '.join(t+' '+n for t,n in zip(types,names))+')'
    assert old in after.source;after.source=after.source.replace(old,new,1)
    with pytest.raises(AssertionError):independently_expand(before,after)


def test_post_run_barrier_is_retained_and_its_input_or_output_corruption_is_rejected():
    before=fixture();after=fold(before)
    boundary='latch_prun_body0(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);\n__syncthreads();\n// Phase 33:'
    assert boundary in after.source
    after.source=after.source.replace(boundary,boundary.replace('__syncthreads();',''),1)
    with pytest.raises(AssertionError):independently_expand(before,after)
    before.source=before.source.replace('__syncthreads();\n// Phase 33:', 'side_effect();\n// Phase 33:',1)
    refused=fold(before)
    assert refused.source==before.source and refused.manifest[KEY]['outlined_runs']==0


def test_predecessor_helpers_remain_exact_and_wrong_abi_is_rejected():
    before=fixture(32,counters=())
    assert before.manifest[pi.KEY]['outlined_intervals']==1
    after=fold(before);assert after.source==before.source and after.manifest[KEY]['outlined_runs']==0
    independently_expand(before,after)
    for old,new in [('void latch_pi_body0(float** args','void latch_pi_body0(float* args'),
                    ('latch_pi_body0(args, arena, lsr_tx, lsr_bx,','latch_pi_body0(args, arena, lsr_bx, lsr_tx,'),
                    ('v_x_0[j] * 2.0f','v_x_0[j] * 3.0f')]:
        damaged=copy.deepcopy(before);damaged.source=damaged.source.replace(old,new,1)
        with pytest.raises(Refusal):fold(damaged)


def test_missing_predecessor_and_duplicate_modes_fail_closed():
    with pytest.raises(Refusal):fold(base_fixture())
    with pytest.raises(Refusal):fold(fold(fixture()))
