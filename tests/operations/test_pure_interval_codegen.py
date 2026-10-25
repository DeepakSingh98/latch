"""Whole pure-interval outlining preserves every computation/publication byte."""
import copy
import hashlib
import re

import pytest
from megakernel_lab.operation_compiler.lower import GeneratedProgram
from megakernel_lab.operation_compiler import mixed_phase_pointer_codegen as mixed
from megakernel_lab.operation_compiler import resource_phase_pointer_codegen as resource
from megakernel_lab.operation_compiler.pure_interval_codegen import fold,KEY,PREFIX,Refusal
from .fixtures.tinymoe_source_proofs import close_brace

from .test_phase_pointer_codegen import fixture as base_fixture


def fixture():
    value=base_fixture(32)
    old='if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();'
    new='if (ticks && lsr_bx == 0 && lsr_tx == 0) { const unsigned long long t = stamp(); '
    new+=' '.join(f'ticks[{i+1}] = t;' for i in range(32))+' }'
    value.source=value.source.replace(old,new)
    return resource.fold(mixed.fold(value))


def independently_expand(before,after):
    source=after.source
    pattern=re.compile(r'static __device__ __noinline__ void (latch_pi_body\d+)\(float\*\* args, float\* arena, unsigned lsr_tx, unsigned lsr_bx, unsigned lsr_nt, unsigned lsr_nb\) \{\n')
    found=list(pattern.finditer(source));groups=after.manifest[KEY]['groups']
    assert {m[1] for m in found}=={g['helper'] for g in groups} and len(found)==len(groups)
    bodies={}
    for m in reversed(found):
        end=close_brace(source,m.end()-2)
        assert source[end-1:end+2]=='\n}\n'
        bodies[m[1]]=source[m.end():end-1]
        source=source[:m.start()]+source[end+2:]
    for g in groups:
        name=g['helper'];code=bodies[name]
        assert hashlib.sha256(code.encode()).hexdigest()==g['body_sha256']
        call=name+'(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);'
        assert source.count(call)==1
        source=source.replace(call,code)
    assert source==before.source,'computation, binding, barrier, tick, commit or framing bytes changed'
    assert {k:v for k,v in before.manifest.items() if k!='source_hash'}=={
        k:v for k,v in after.manifest.items() if k not in (KEY,'source_hash')}


def test_complete_interval_body_moves_but_publication_remains_caller():
    before=fixture();after=fold(before);proof=after.manifest[KEY]
    assert proof['outlined_intervals']==1 and proof['outlined_phases']==32
    independently_expand(before,after)
    start=after.source.index('extern "C" __global__')
    helper=after.source[:start];entry=after.source[start:]
    assert helper.count('__syncthreads();')==31 and '__syncthreads();' not in entry
    assert 'unsigned lsr_tx = threadIdx.x;' in entry
    assert 'unsigned lsr_tx = threadIdx.x;' not in helper
    assert 'float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);' in entry
    assert 'grid.sync();' not in helper and 'ticks[' not in helper
    assert entry.count('grid.sync();')==before.source.count('grid.sync();')
    assert after.manifest[mixed.KEY]==before.manifest[mixed.KEY]
    assert after.manifest[resource.KEY]==before.manifest[resource.KEY]


@pytest.mark.parametrize('suffix',[
    'return;', 'break;', 'continue;', 'goto label;', 'unknown(v_y_0);', 'use(&v_y_0);',
    '__shared__ float smem[4];', 'atomicAdd(v_y_0,1.f);', 'asm volatile("wgmma.fence.sync.aligned;");',
    'const float local_array[2]={0.f,1.f};', 'float* saved=v_y_0;'])
def test_unknown_resource_control_or_binding_forms_stay_inline(suffix):
    before=fixture()
    marker='// latch_mixed_phase_pointers: end\n'
    before.source=before.source.replace(marker,marker+suffix+'\n',1)
    after=fold(before);assert after.source==before.source and after.manifest[KEY]['outlined_intervals']==0


@pytest.mark.parametrize('key',['kind','counter_effects','private_scratch','requires_full_grid','shared_bytes'])
def test_metadata_cannot_admit_mixed_or_resource_interval(key):
    before=fixture();before.manifest['phases'][0][key]='counter' if key=='kind' else ['resource']
    after=fold(before);assert after.source==before.source


@pytest.mark.parametrize('mutation',['outer_scalar','outer_pointer','outer_shared','gap','tail','unbound','out','prefix','constant_abi'])
def test_exact_enclosing_state_and_publication_refusals(mutation):
    before=fixture()
    if mutation=='outer_scalar': before.source=before.source.replace('// Phase 0:', 'float persistent=1.f;\n// Phase 0:',1)
    elif mutation=='outer_pointer': before.source=before.source.replace('// Phase 0:', 'float* v_outer = latch_pa[0];\n// Phase 0:',1)
    elif mutation=='outer_shared': before.source=before.source.replace('// Phase 0:', '__shared__ float saved[4];\n// Phase 0:',1)
    elif mutation=='gap': before.source=before.source.replace('__syncthreads();','__syncthreads();\nside_effect();',1)
    elif mutation=='tail': before.source=before.source.replace('ticks[1] = t;','ticks[1] = t + 1;',1)
    elif mutation=='unbound': before.source=before.source.replace('v_x_0[j]','v_outer[j]',1)
    elif mutation=='out': before.source=before.source.replace('latch_pa[0]','latch_po[0]',1)
    elif mutation=='prefix':
        pos=before.source.index('// latch_mixed_phase_pointers: 0')
        before.source=before.source[:pos]+before.source[pos:].replace('latch_opq(args)','latch_opq(out)',1)
    else:
        before.source=before.source.replace('float** args, float* arena, unsigned long long* ticks','const float** args, float* arena, unsigned long long* ticks',1)
        with pytest.raises(Refusal): fold(before)
        return
    after=fold(before);assert after.source==before.source


@pytest.mark.parametrize('change',['body','signature','call','swapped_scalar_call','swapped_scalar_signature',
                                  'scalar_type','caller_scalar_value','helper_parameter_assignment',
                                  'helper_parameter_escape','publication','manifest'])
def test_independent_inverse_rejects_actual_emitted_corruption(change):
    before=fixture();after=fold(before)
    if change=='body': after.source=after.source.replace('v_x_0 = latch_pa[0]','v_x_0 = latch_pa[1]',1)
    elif change=='signature': after.source=after.source.replace('void latch_pi_body0(float** args','void latch_pi_body0(float* args',1)
    elif change=='call': after.source=after.source.replace('latch_pi_body0(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);','latch_pi_body0(out, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);',1)
    elif change=='swapped_scalar_call': after.source=after.source.replace('latch_pi_body0(args, arena, lsr_tx, lsr_bx, lsr_nt, lsr_nb);','latch_pi_body0(args, arena, lsr_bx, lsr_tx, lsr_nt, lsr_nb);',1)
    elif change=='swapped_scalar_signature': after.source=after.source.replace('unsigned lsr_tx, unsigned lsr_bx','unsigned lsr_bx, unsigned lsr_tx',1)
    elif change=='scalar_type': after.source=after.source.replace('unsigned lsr_tx,','int lsr_tx,',1)
    elif change=='caller_scalar_value': after.source=after.source.replace('unsigned lsr_tx = threadIdx.x;','unsigned lsr_tx = blockIdx.x;',1)
    elif change in ('helper_parameter_assignment','helper_parameter_escape'):
        suffix='lsr_tx = 7;' if change=='helper_parameter_assignment' else 'unknown(&lsr_tx);'
        after.source=after.source.replace('// Phase 0:',suffix+'\n// Phase 0:',1)
    elif change=='publication': after.source=after.source.replace('ticks[1] = t;','ticks[1] = t+1;',1)
    else:
        after.manifest=copy.deepcopy(after.manifest);after.manifest['phases'][0]['counter_effects']=['changed']
    with pytest.raises(AssertionError): independently_expand(before,after)


@pytest.mark.parametrize('body',['lsr_tx = 7;', '++lsr_bx;', 'unknown(&lsr_nt);', 'unsigned& escaped = lsr_nb;'])
def test_hardware_index_binding_mutation_or_escape_is_not_admitted(body):
    before=fixture()
    marker='// latch_mixed_phase_pointers: end\n'
    before.source=before.source.replace(marker,marker+body+'\n',1)
    after=fold(before)
    assert after.source==before.source and after.manifest[KEY]['outlined_intervals']==0


def test_requires_preceding_pass_and_refuses_double_outline():
    with pytest.raises(Refusal):fold(base_fixture())
    with pytest.raises(Refusal):fold(fold(fixture()))


@pytest.mark.parametrize('mutation',['missing_scalar','changed_scalar','free_outer_scalar','outer_reference'])
def test_scalar_definition_and_free_capture_refusals(mutation):
    before=fixture()
    line='unsigned lsr_tx = threadIdx.x; asm volatile("" : "+r"(lsr_tx));\n'
    if mutation=='missing_scalar':before.source=before.source.replace(line,'',1)
    elif mutation=='changed_scalar':before.source=before.source.replace(line,line.replace('threadIdx.x','blockIdx.x'),1)
    elif mutation=='free_outer_scalar':before.source=before.source.replace('v_x_0[j] * 2.0f','v_x_0[j] * enclosing_value',1)
    else:before.source=before.source.replace('// Phase 0:','float& escaped = arena[0];\n// Phase 0:',1)
    after=fold(before);assert after.source==before.source


def test_existing_compact_helper_signature_or_call_cannot_change():
    from megakernel_lab.operation_compiler.elementwise_codegen import fold as compact
    base=base_fixture(32)
    old='if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();'
    new='if (ticks && lsr_bx == 0 && lsr_tx == 0) { const unsigned long long t = stamp(); '
    new+=' '.join(f'ticks[{i+1}] = t;' for i in range(32))+' }'
    base.source=base.source.replace(old,new)
    before=resource.fold(mixed.fold(compact(base)))
    assert before.manifest['elementwise_codegen']['shared_phases']==32
    independently_expand(before,fold(before))
    signature=re.search(r'void latch_ec_body0\([^\n]*',before.source)[0]
    assert 'float*' in signature
    damaged=GeneratedProgram(before.source.replace(signature,signature.replace('float*','const float*',1),1),before.manifest)
    with pytest.raises(Refusal):fold(damaged)
    call=re.search(r'(?m)^latch_ec_body0\([^;\n]*\);',before.source)[0]
    assert 'v_x_0' in call
    damaged=GeneratedProgram(before.source.replace(call,call.replace('v_x_0','enclosing_pointer'),1),before.manifest)
    after=fold(damaged);assert after.source==damaged.source
