"""Typed elementwise sharing, rejection boundaries and emitted-source inversion."""
import hashlib
import json
import re

import pytest
import torch

from megakernel_lab.operation_compiler import capture_module
from megakernel_lab.operation_compiler.elementwise_codegen import Refusal, canonicalize, fold, _phase_bodies, _tokens
from megakernel_lab.operation_compiler.lower import GeneratedProgram, LoweringError, lower


def body(x='v_x', y='v_y', register='r_result', literal='2.0f'):
    register += '_optimizer_intermediate'
    return (f'for (long long i = (long long)lsr_bx * lsr_nt + lsr_tx; i < 8192LL; '
            f'i += (long long)lsr_nb * lsr_nt) {{\n'
            f'const float {register} = fmaf({x}[i], {literal}, sqrtf({x}[i]));\n'
            f'{y}[i] = ({register} > 0.5f ? {register} : expm1f({register}));\n}}\n')


def fixture(bodies=None, kinds=None, types=None):
    bodies = bodies or [body(), body('v_a','v_b','r_second'), body('v_c','v_d','r_third')]
    kinds = kinds or ['elementwise'] * len(bodies)
    types = types or {name:'float' for name in ('x','y','a','b','c','d')}
    declarations = ''.join(f'{ty}* v_{name} = args[{i}];\n' for i,(name,ty) in enumerate(types.items()))
    source = ('// common fixture\nextern "C" __global__ void latch_ops(float** args, float* arena, '
              'unsigned long long* ticks) {\n'
              'unsigned lsr_tx = threadIdx.x;\nunsigned lsr_bx = blockIdx.x;\n'
              'unsigned lsr_nt = blockDim.x;\nunsigned lsr_nb = gridDim.x;\n' + declarations)
    for index,value in enumerate(bodies):
        source += f'// Phase {index}: fixture\n{{\n{value}}}\n__syncthreads();\ngrid.sync();\nticks[{index}] = stamp();\n'
    source += '// Commit remains entry-resident\nv_y[0] = v_x[0];\n}\n'
    manifest = dict(phases=[dict(kind=k) for k in kinds],header_hashes={},execution_selection={'selected':'common_static'})
    manifest['source_hash'] = hashlib.sha256((source+json.dumps({},sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source,manifest)


def helpers(source):
    """Read emitted helper declarations/bodies rather than the compiler's patch log."""
    result = {}
    for head in re.finditer(r'static __device__ __noinline__ void (latch_ec_body\d+)\(([^)]*)\) \{\n',source):
        depth=1
        for kind,text,start,end in _tokens(source[head.end():]):
            if kind=='other' and text=='{':depth+=1
            elif kind=='other' and text=='}':depth-=1
            if depth==0:
                result[head[1]]=(head.start(),head.end()+end+1,head[2],source[head.end():head.end()+start]);break
    return result


def independently_expand(original, emitted):
    groups=helpers(emitted.source);expected=dict((p,original.source[a:b]) for p,a,b in _phase_bodies(original.source))
    edits=[]
    for phase,start,end in _phase_bodies(emitted.source):
        call=re.fullmatch(r'(latch_ec_body\d+)\(([^;]*)\);\n',emitted.source[start:end])
        if call is None:continue
        _,_,signature,code=groups[call[1]]
        formals=[arg.rsplit(' ',1)[1] for arg in signature.split(', ')]
        bindings=dict(zip(formals,call[2].split(', ')))
        # Local identifiers have no caller bindings. Recover the original local
        # spelling using independently located declarations in the two bodies.
        local_pattern=r'\b(?:long long|float) (\w+) ='
        locals_old=re.findall(local_pattern,expected[phase]);locals_new=re.findall(local_pattern,code)
        bindings.update(zip(locals_new,locals_old))
        expanded=re.sub(r'\b\w+\b',lambda m:bindings.get(m[0],m[0]),code)
        assert expanded==expected[phase]
        edits.append((start,end,expanded))
    edits += [(start,end,'') for start,end,_,_ in groups.values()]
    source=emitted.source
    for start,end,value in sorted(edits,reverse=True):source=source[:start]+value+source[end:]
    assert source==original.source


def test_repeated_typed_bodies_expand_to_every_original_byte():
    original=fixture(); emitted=fold(original)
    assert emitted.manifest['elementwise_codegen']['shared_phases']==3
    assert emitted.manifest['elementwise_codegen']['shared_bodies']==1
    assert len(emitted.source)<len(original.source)
    assert emitted.manifest['source_hash']!=original.manifest['source_hash']
    assert 'restrict' not in emitted.source
    assert emitted.manifest['elementwise_codegen']['groups'][0]['parameter_types']==[
        'unsigned','unsigned','unsigned','unsigned','float*','float*']
    independently_expand(original,emitted)
    # The independent inverse must catch an actual emitted arithmetic mutation.
    damaged=GeneratedProgram(emitted.source.replace('2.0f','3.0f'),emitted.manifest)
    with pytest.raises(AssertionError):independently_expand(original,damaged)


def test_alias_arguments_are_preserved_without_restrict_or_pointer_rebasing():
    original=fixture([body('v_x','v_x'),body('v_a','v_a','r_second'),body('v_b','v_b','r_third')])
    emitted=fold(original)
    assert emitted.manifest['elementwise_codegen']['shared_phases']==3
    independently_expand(original,emitted)


def test_non_elementwise_bodies_and_publication_bytes_remain_unchanged():
    matrix='wgmma_do_not_touch(v_x, smem);\nasm volatile("cp.async.bulk");\n'
    original=fixture([body(),matrix,body('v_a','v_b','r_second'),body('v_c','v_d','r_third')],
                     ['elementwise','gemm','elementwise','elementwise'])
    emitted=fold(original)
    assert matrix in emitted.source
    assert emitted.manifest['elementwise_codegen']['shared_phases']==3
    independently_expand(original,emitted)


def test_literals_and_pointer_types_are_part_of_the_group_identity():
    original=fixture([body(),body('v_a','v_b','r_second','3.0f'),body('v_c','v_d','r_third')],
                     types={'x':'float','y':'float','a':'float','b':'float','c':'double','d':'double'})
    emitted=fold(original)
    assert emitted.manifest['elementwise_codegen']['shared_phases']==0
    assert emitted.source==original.source


@pytest.mark.parametrize('addition',[
    'asm volatile("bar.sync 0;");','__syncthreads();','atomicAdd(v_y, 1.0f);','smem[0] = 1.0f;',
    'mbarrier_wait();','unknown_pure(v_x);','return;','continue;','v_x += 1;','++v_x;',
    '(v_x) = v_y;','unsigned* p = &lsr_tx;','v_y[0] = hidden_scalar;', 'v_y[0] = pragma;', 'v_y[0] = unroll;',
    '{ float leaked = 1.0f; } v_y[0] = leaked;',
    'float local = 1.0f; v_y[0] = (float)(uint64_t)&local;',
    'float local[4]; v_y[0] = (float)reinterpret_cast<int64_t>(local);',
    'float safe, local[4]; v_y[0] = (float)reinterpret_cast<int64_t>(local);',
    'if (v_x[0] > 0.0f) float leaked = 1.0f; v_y[0] = leaked;',
    'if (int hidden = 1) { v_y[0] = hidden; } v_y[1] = hidden;',
    'v_y[0] = (float)(uint64_t)&threadIdx.x;',
    'for (int loop_i = 0; loop_i < 2; ++loop_i) { v_y[0] += 1.0f; } v_y[0] = loop_i;', 'float r_external = 1.0f; v_y[0] = undeclared_r;',
])
def test_unsafe_or_unbound_source_is_refused(addition):
    with pytest.raises(Refusal):canonicalize(body()+addition,{'v_x':'float*','v_y':'float*'})


def test_comments_members_and_shadowed_local_scopes_roundtrip():
    text=('// comment with r_same and v_x\nfloat4 x, y;\nx.x = v_x[0];\n'
          '{ float r_same = x.x; v_y[0] = r_same; }\n'
          '{ float r_same = x.y; v_y[1] = r_same; }\n')
    canonical,parameters,inverse=canonicalize(text,{'v_x':'float*','v_y':'float*'})
    assert '.x' in canonical and '.y' in canonical
    assert '// comment with r_same and v_x' in canonical
    assert ('float*','v_x') in parameters


def test_unbound_qualified_pointer_and_manifest_mismatch_fail_closed():
    original=fixture();original.source=original.source.replace('float* v_x','volatile float* v_x')
    emitted=fold(original)
    assert emitted.manifest['elementwise_codegen']['refused_bodies']['unbound or qualified tensor pointer']==1
    original=fixture();original.manifest['phases'].pop()
    with pytest.raises(Refusal,match='identity mismatch'):fold(original)


def test_public_lowering_default_bytes_and_explicit_common_mode():
    module=torch.nn.Linear(8,8)
    captured=capture_module(module,lambda value,target:(value-target).square().mean(),
                            (torch.ones(8,8),),(torch.zeros(8,8),))
    first=lower(captured.ir,execution='compose',compose_selection='common')
    second=lower(captured.ir,execution='compose',compose_selection='common',compact_codegen='off')
    assert first.source==second.source and first.manifest==second.manifest
    compact=lower(captured.ir,execution='compose',compose_selection='common',compact_codegen='elementwise')
    assert 'elementwise_codegen' in compact.manifest
    with pytest.raises(LoweringError,match='standalone static common'):
        lower(captured.ir,compact_codegen='elementwise')
    with pytest.raises(LoweringError,match='legacy schedule'):
        lower(captured.ir,execution='compose',compact_codegen='fold')


def test_auto_legacy_selection_cannot_silently_ignore_explicit_sharing(monkeypatch):
    from megakernel_lab.operation_compiler import compose
    captured=capture_module(torch.nn.Linear(4,4),lambda value,target:(value-target).square().mean(),
                            (torch.ones(4,4),),(torch.zeros(4,4),))
    select=compose.flat_selection
    def legacy(*args,**kwargs):
        result=select(*args,**kwargs)
        return dict(result,selected='legacy_static')
    monkeypatch.setattr(compose,'flat_selection',legacy)
    with pytest.raises(LoweringError,match="set compose_selection='common'"):
        lower(captured.ir,execution='compose',compact_codegen='elementwise')


def test_mixed_pointer_qualifications_and_scalar_widths_cannot_be_erased():
    original=fixture()
    original.source=original.source.replace('// Phase 1:', 'volatile float* v_x = args[0];\n// Phase 1:')
    result=fold(original)
    assert result.manifest['elementwise_codegen']['refused_bodies']['unbound or qualified tensor pointer']==1
    original=fixture()
    original.source=original.source.replace('unsigned lsr_tx =', 'unsigned long long lsr_tx =')
    with pytest.raises(Refusal,match='scalar type'):fold(original)


@pytest.mark.parametrize('declaration',[
    'float* volatile v_x', 'float* const v_x', 'float* __restrict__ v_x',
    'float*volatile v_x', 'float const* v_x', 'volatile float* v_x',
    'float* const* v_x', 'float*& v_x',
])
def test_every_nonplain_pointer_spelling_is_refused_globally(declaration):
    original=fixture()
    original.source=original.source.replace('// Phase 1:', declaration+' = args[0];\n// Phase 1:')
    result=fold(original)
    assert result.manifest['elementwise_codegen']['refused_bodies']['unbound or qualified tensor pointer']==1
