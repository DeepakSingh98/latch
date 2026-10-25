"""Independent expansion and refusal tests for exact full-sum outlining."""
import hashlib
import json
import re

import pytest
import torch

from megakernel_lab.operation_compiler import capture_module
from megakernel_lab.operation_compiler.elementwise_codegen import _phase_bodies, _tokens
from megakernel_lab.operation_compiler.full_sum_codegen import ARITHMETIC, HEAD, TAIL, Refusal, fold, ordinary_body
from megakernel_lab.operation_compiler.lower import GeneratedProgram, LoweringError, lower
from megakernel_lab.operation_compiler.square_sum_association import marker
from megakernel_lab.operation_compiler.user_optimizer import TensorUpdateRule


def fixture(sizes=(32768, 32768, 32768)):
    source = ('extern "C" __global__ void latch_ops(float** args, float* arena, unsigned long long* ticks) {\n'
              'extern __shared__ float smem[];\n'
              'unsigned lsr_tx = threadIdx.x;\nunsigned lsr_bx = blockIdx.x;\n'
              'unsigned lsr_nt = blockDim.x;\nunsigned lsr_nb = gridDim.x;\n')
    phases=[]
    for index,elements in enumerate(sizes):
        x,s,y=(f'{prefix}{index}' for prefix in ('input','scratch','output'))
        source+=''.join(f'float* v_{name} = args[{3*index+j}];\n' for j,name in enumerate((x,s,y)))
        phases.append(dict(kind='reduction_counter_init', names=[s],reads=[],writes=[s],targets=[]))
        source+=f'// Phase {len(phases)-1}: {s}\n{{\nif (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_{s}) = 0u;\n}}\ngrid.sync();\n'
        phases.append(dict(kind='reduction',engine='multi_cta_full',requires_full_grid=True,max_grid_blocks=1024,
            arithmetic=ARITHMETIC,reduction_elements=elements,targets=['aten.sum.default'],names=[y],writes=[y],
            reads=[x,s],internal_mutations=[s],private_scratch=[dict(name=s,counter_words=1,partial_word_offset=64,partial_words=1024)],
            counter_effects=[dict(storage=s,word_offset=0,words=1,initialization='zeroed by dependency predecessor',
                                 completion='last arriver restores zero')]))
        source+=f'// Phase {len(phases)-1}: {y}\n{{\n'+ordinary_body(elements,'v_'+x,'v_'+s,'v_'+y,marker(y))+'}\n'
        source+='__syncthreads();\ngrid.sync();\nif (ticks) ticks[1] = stamp();\n'
    source+='// caller commit remains in place\n}\n'
    manifest=dict(phases=phases,header_hashes={},full_grid=True,execution_selection={'selected':'common_static'},
                  shared_bytes=104857,resource_policy={'unchanged':'all original admission metadata'},
                  phase_count=len(phases)+1,block_threads=256)
    manifest['source_hash']=hashlib.sha256((source+json.dumps({},sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source,manifest)


def independently_expand(original, emitted):
    helpers={};edits=[]
    for h in re.finditer(r'static __device__ __noinline__ void (latch_fs_body\d+)\(([^)]*)\) \{\n',emitted.source):
        depth=1
        for kind,word,start,end in _tokens(emitted.source[h.end():]):
            if kind=='other' and word=='{':depth+=1
            elif kind=='other' and word=='}':depth-=1
            if not depth:
                helpers[h[1]]=(h[2],emitted.source[h.end():h.end()+start])
                edits.append((h.start(),h.end()+end+1,''));break
    old={phase:original.source[a:b] for phase,a,b in _phase_bodies(original.source)}
    changed=0
    for phase,start,end in _phase_bodies(emitted.source):
        body=emitted.source[start:end]
        call=re.search(r'(latch_fs_body\d+)\(([^;]+)\);',body)
        if call is None:continue
        signature,code=helpers[call[1]]
        formals=[arg.rsplit(' ',1)[1] for arg in signature.split(', ')]
        bindings=dict(zip(formals,call[2].split(', ')))
        assert bindings.pop('latch_fs_last')=='&msum_last_'
        code=code.replace('(*latch_fs_last)','msum_last_')
        code=re.sub(r'\b\w+\b',lambda m:bindings.get(m[0],m[0]),code)
        stamp=re.search(r'/\* square_sum_lanes8:[0-9a-f]{64} \*/',body)[0]
        code=code.replace('  auto latch_term =',stamp+'\n  auto latch_term =',1)
        expanded=HEAD+code.rstrip('\n')+TAIL
        assert expanded==old[phase]
        edits.append((start,end,expanded));changed+=1
    assert changed==emitted.manifest['full_sum_codegen']['shared_phases']
    result=emitted.source
    for start,end,value in sorted(edits,reverse=True):result=result[:start]+value+result[end:]
    assert result==original.source


def test_shared_flag_and_every_original_source_byte_reexpand():
    original=fixture();emitted=fold(original)
    assert emitted.manifest['full_sum_codegen']['shared_phases']==3
    assert emitted.manifest['full_sum_codegen']['shared_bodies']==1
    assert len(emitted.source)<len(original.source)
    assert emitted.source.count('__shared__ unsigned msum_last_;')==3
    assert 'restrict' not in emitted.source
    independently_expand(original,emitted)
    assert {k:v for k,v in emitted.manifest.items() if k not in ('source_hash','full_sum_codegen')}=={
        k:v for k,v in original.manifest.items() if k!='source_hash'}
    damaged=GeneratedProgram(emitted.source.replace('lane0 + lane1','lane0 - lane1'),emitted.manifest)
    with pytest.raises(AssertionError):independently_expand(original,damaged)


def test_sizes_remain_literal_specializations_and_singletons_inline():
    original=fixture((32768,65536,32768))
    result=fold(original)
    assert result.manifest['full_sum_codegen']['shared_phases']==2
    assert result.manifest['full_sum_codegen']['groups'][0]['elements']==32768
    independently_expand(original,result)


@pytest.mark.parametrize('mutation',[
    'arithmetic','counter','partial_offset','scratch_words','full_grid','grid_limit','wrong_target',
    'initializer_missing','initializer_value','fence','barrier','slice','association','source_width',
    'pointer_width','pointer_qualifier','foreign_shared','marker','input_index','size',
])
def test_no_relaxation_of_body_or_counter_contract(mutation):
    original=fixture();phase=original.manifest['phases'][1]
    if mutation=='arithmetic':phase['arithmetic']='other'
    elif mutation=='counter':phase['counter_effects'][0]['word_offset']=1
    elif mutation=='partial_offset':phase['private_scratch'][0]['partial_word_offset']=1
    elif mutation=='scratch_words':phase['private_scratch'][0]['partial_words']=1
    elif mutation=='full_grid':phase['requires_full_grid']=False
    elif mutation=='grid_limit':phase['max_grid_blocks']=2048
    elif mutation=='wrong_target':phase['targets']=['aten.mean.default']
    elif mutation=='initializer_missing':original.manifest['phases'][0]['kind']='other'
    elif mutation=='initializer_value':original.source=original.source.replace('v_scratch0) = 0u','v_scratch0) = 1u',1)
    elif mutation=='fence':original.source=original.source.replace('__threadfence();','',1)
    elif mutation=='barrier':original.source=original.source.replace('  __syncthreads();','',1)
    elif mutation=='slice':original.source=original.source.replace('if (lsr_nb <= 1024)','if (lsr_bx < 12 && lsr_nb <= 1024)',1)
    elif mutation=='association':original.source=original.source.replace('lane0 + lane1','lane0 - lane1',1)
    elif mutation=='source_width':original.source=original.source.replace('unsigned lsr_tx =','unsigned long long lsr_tx =',1)
    elif mutation=='pointer_width':original.source=original.source.replace('float* v_input0','double* v_input0',1)
    elif mutation=='pointer_qualifier':original.source=original.source.replace('float* v_input0','float* volatile v_input0',1)
    elif mutation=='foreign_shared':original.source=original.source.replace('float acc = 0.0f;','float acc = foreign_smem[0];',1)
    elif mutation=='marker':original.source=original.source.replace(marker('output0'),marker('wrong'),1)
    elif mutation=='input_index':original.source=original.source.replace('v_input0[(long long)','v_input0[1+(long long)',1)
    elif mutation=='size':phase['reduction_elements']=32769
    if mutation=='source_width':
        with pytest.raises(Refusal,match='register type'):fold(original)
    else:
        result=fold(original)
        assert result.manifest['full_sum_codegen']['shared_phases']==2
        assert sum(result.manifest['full_sum_codegen']['refused_bodies'].values())==1
        independently_expand(original,result)


def test_nonreduction_attention_and_gemm_source_never_moves():
    original=fixture()
    original.source=original.source.replace('// caller commit', 'wgmma_external(smem);\n// caller commit')
    emitted=fold(original)
    independently_expand(original,emitted)
    assert 'wgmma_external(smem);' in emitted.source


def test_public_mode_uses_ordinary_captured_norm_and_keeps_prior_modes_unchanged():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.ones(16384)) for _ in range(3)])
        def forward(self,x):
            return sum((i+1)*p*x for i,p in enumerate(self.weights))
    def init(params,hyper):return {'norm':torch.zeros(())}
    def update(params,grads,state,hyper):
        # The explicit association policy admits multiply-self sums, matching
        # the production optimizer's captured graph (not aten.pow/square).
        norm=sum((g*g).sum() for g in grads.values())
        return {k:p-.001*grads[k]/(norm.sqrt()+1) for k,p in params.items()},{'norm':norm}
    captured=capture_module(Model(),lambda value,target:(value-target).square().mean(),
        (torch.ones(16384),),(torch.zeros(16384),),optimizer=TensorUpdateRule(update,init_state=init))
    options=dict(execution='compose',compose_selection='common',precision_policy='practical',square_sum_association='balanced')
    ordinary=lower(captured.ir,**options)
    off=lower(captured.ir,compact_codegen='off',**options)
    assert ordinary.source==off.source and ordinary.manifest==off.manifest
    elementwise=lower(captured.ir,compact_codegen='elementwise',**options)
    candidate=lower(captured.ir,compact_codegen='elementwise_full_sum',**options)
    assert candidate.manifest['elementwise_codegen']==elementwise.manifest['elementwise_codegen']
    assert candidate.manifest['full_sum_codegen']['shared_phases']>=3, candidate.manifest['full_sum_codegen']
    independently_expand(elementwise,candidate)
    again=lower(captured.ir,compact_codegen='elementwise',**options)
    assert elementwise.source==again.source and elementwise.manifest==again.manifest
    with pytest.raises(LoweringError,match='standalone static common'):
        lower(captured.ir,compact_codegen='full_sum')


@pytest.mark.parametrize('elementwise,outline,mode',[
    (False,False,'off'),(True,False,'elementwise'),
    (False,True,'full_sum'),(True,True,'elementwise_full_sum'),
])
def test_tinymoe_build_selects_only_requested_passes(tmp_path,monkeypatch,elementwise,outline,mode):
    from types import SimpleNamespace
    from megakernel_lab.tinymoe import build
    calls=[]
    def captured_build(directory,**kwargs):
        directory.mkdir();calls.append(kwargs);return {'source_hash':'fixture'}
    captured=SimpleNamespace(ir=SimpleNamespace(metadata={'tinymoe':{}},operations=[]),build=captured_build)
    monkeypatch.setattr(build,'capture_model',lambda *args,**kwargs:(torch.nn.Linear(1,1),captured))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    build.prepare('model',tmp_path/'artifact',tmp_path/'cache',recipe=None,phase='plain',
        batch_size=1,sequence_length=2,compact_elementwise=elementwise,outline_full_sums=outline)
    assert calls[0]['compact_codegen']==mode


def test_tinymoe_cli_forwards_explicit_passes_and_assembler_preference(monkeypatch):
    from megakernel_lab.tinymoe import build
    calls=[]
    monkeypatch.setattr(build,'prepare',lambda *args,**kwargs:calls.append(kwargs) or {})
    monkeypatch.setattr('sys.argv',['build','--model','model','--out','artifact','--cache-dir','cache',
                                  '--compact-elementwise','--outline-full-sums','--disable-expensive-optimizations'])
    build.main()
    assert calls[0]['compact_elementwise'] is True and calls[0]['outline_full_sums'] is True
    assert calls[0]['disable_expensive_optimizations'] is True
