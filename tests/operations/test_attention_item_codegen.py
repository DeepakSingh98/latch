"""Exact v7 item call boundaries and independent source/header inverse."""
import copy
import hashlib
import json
import re

import pytest
from megakernel_lab.operation_compiler import attention_item_codegen as ai
from megakernel_lab.operation_compiler import elementwise_codegen as ec
from megakernel_lab.operation_compiler.lower import GeneratedProgram, _header_path


def fixture():
    source=''.join(f'#define {k} {v}\n' for k,v in ai.MACROS.items())
    source+='#include "attention_long_flat.cuh"\n'
    source+='extern "C" __global__ void latch_ops(float** args, float* arena, unsigned long long* ticks) {\n'
    phases=[]
    for i,kind in enumerate(('forward','dq','kv')):
        tile='(int)(idx / 4LL)' if kind=='kv' else '3 - (int)(idx / 4LL)'
        source+=f'// Phase {i}: attention\n{{\npflatlong::Args A{{}};\nA.q = v_q; A.lq = pflatlong::Layout4{{65536LL, 16384LL, 64LL}};\n'
        source+=f'for (long long r = 0;; ++r) {{\n  const long long idx = pflatlong::snake2(r);\n  if (idx >= 16LL) break;\n  pflatlong::{kind}_item6<1, false>(A, (int)(idx % 4LL), {tile}, smem);\n}}\n}}\ngrid.sync();\n'
        phases.append(dict(kind='attention_long_'+kind,schedule_version=7,recipe='bf16',pieces=1,head_width=64,
                           shared_bytes=103424,sequence=256,head_batch=4,work_items=16))
    source+='}\n'
    manifest=dict(header_hashes={ai.ORIGINAL:ai.AUDITED_SHA256},phases=phases,
                  execution_selection={'selected':'common_static'},barrier_schedule='dependency')
    manifest['source_hash']=hashlib.sha256((source+json.dumps(manifest['header_hashes'],sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source,manifest)


def independently_expand(before,after):
    record=after.manifest[ai.KEY]
    source=after.source
    assert record['source_before']==before.cache_key
    actual_by_phase={i:source[a:b] for i,a,b in ec._phase_bodies(source)}
    assert len({g['phase'] for g in record['groups']})==len(record['groups'])
    assert record['shared_wrappers']==len({g['wrapper'] for g in record['groups']})
    for g in record['groups']:
        actual=list(re.finditer(r'latch_attention_v7::(forward|dq|kv)_item<1, (true|false)>\([^;]+;',actual_by_phase[g['phase']]))
        assert len(actual)==1
        m=actual[0]
        assert g['wrapper']=='latch_attention_v7::'+m[1]+'_item' and g['target']=='pflatlong::'+m[1]+'_item6'
        assert g['pieces']==1 and g['pow2']==(m[2]=='true')
        assert hashlib.sha256(m[0].replace(g['wrapper'],g['target'],1).encode()).hexdigest()==g['call_sha256']
    assert record['original_header_sha256']==ai.AUDITED_SHA256
    assert hashlib.sha256(_header_path(ai.ORIGINAL).read_bytes()).hexdigest()==ai.AUDITED_SHA256
    if record['outlined_items']:
        assert after.manifest['header_hashes'][ai.HEADER]==ai.WRAPPER_SHA256
        assert hashlib.sha256(_header_path(ai.HEADER).read_bytes()).hexdigest()==ai.WRAPPER_SHA256
        text=_header_path(ai.HEADER).read_text()
        assert '__syncthreads' not in text
        for kind in ('forward','dq','kv'):
            expected=(f'template <int NS, bool POW2>\nstatic __device__ __noinline__ void {kind}_item'
                      f'(const pflatlong::Args& a, int bh, int tile, void* smem) {{\n'
                      f'  pflatlong::{kind}_item6<NS, POW2>(a, bh, tile, smem);\n}}')
            assert text.count(expected)==1
        include='#include "'+ai.HEADER+'"\n';assert source.count(include)==1;source=source.replace(include,'')
    calls=list(re.finditer(r'latch_attention_v7::(forward|dq|kv)_item<1, (true|false)>\([^;]+;',source))
    assert len(calls)==record['outlined_items']==len(record['groups'])
    for m in reversed(calls):
        old=m[0].replace('latch_attention_v7::'+m[1]+'_item','pflatlong::'+m[1]+'_item6',1)
        assert hashlib.sha256(old.encode()).hexdigest() in {g['call_sha256'] for g in record['groups']}
        source=source[:m.start()]+old+source[m.end():]
    assert source==before.source,'body, call ABI, macros, schedule or publication changed'
    headers=dict(after.manifest['header_hashes']);headers.pop(ai.HEADER,None)
    assert headers==before.manifest['header_hashes']
    assert {k:v for k,v in before.manifest.items() if k not in ('source_hash','header_hashes')}=={
        k:v for k,v in after.manifest.items() if k not in ('source_hash','header_hashes',ai.KEY)}


def test_three_exact_calls_and_header_round_trip():
    before=fixture();after=ai.fold(before);independently_expand(before,after)
    assert after.manifest[ai.KEY]['outlined_items']==3 and after.manifest[ai.KEY]['shared_wrappers']==3
    assert before.manifest['header_hashes']=={ai.ORIGINAL:ai.AUDITED_SHA256}


@pytest.mark.parametrize('old,new',[
    ('forward_item6','forward_item6s'),('<1, false>','<3, false>'),('snake2(r)','snake2(r+1)'),
    ('idx >= 16LL','idx >= 15LL'),('idx % 4LL','idx % 2LL'),('3 - (int)(idx / 4LL)','2 - (int)(idx / 4LL)'),
    (', smem);',', arena);'),('A.q = v_q;','A.q = unknown(v_q);'),('A.q = v_q;','A.q = v_q; A.q = v_k;'),
    ('A.q = v_q;','A.q = v_q; __syncthreads();'),('Args A{}','Args A{v_q}'),
    ('PFLATLONG_FTZ 1','PFLATLONG_FTZ 0'),('PFLATLONG_WGMMA 1','PFLATLONG_WGMMA 1\n#define PFLATLONG_KV6_PTR 1'),
    ('#include "attention_long_flat.cuh"','#include "attention_long_flat.cuh"\n#define P9_TRACE 1')])
def test_unknown_abi_schedule_macro_and_prefetch_forms_refuse(old,new):
    before=fixture();before.source=before.source.replace(old,new,1)
    with pytest.raises(ai.Refusal):ai.fold(before)


@pytest.mark.parametrize('key,value',[('schedule_version',8),('pieces',3),('recipe','exact3'),
    ('shared_bytes',102400),('work_items',15),('head_width',48),('sequence',192)])
def test_resource_metadata_refuses(key,value):
    before=fixture();before.manifest['phases'][0][key]=value
    with pytest.raises(ai.Refusal):ai.fold(before)


@pytest.mark.parametrize('change',['actual','template','barrier','macro','header','manifest','body'])
def test_independent_inverse_rejects_actual_output_damage(change):
    before=fixture();after=ai.fold(before)
    if change=='actual':after.source=after.source.replace('(A, (int)','(A, 1+(int)',1)
    elif change=='template':after.source=after.source.replace('<1, false>','<1, true>',1)
    elif change=='barrier':after.source=after.source.replace('grid.sync();','__syncthreads();',1)
    elif change=='macro':after.source=after.source.replace('PFLATLONG_FTZ 1','PFLATLONG_FTZ 0',1)
    elif change=='header':after.manifest['header_hashes'][ai.ORIGINAL]='bad'
    elif change=='manifest':after.manifest=copy.deepcopy(after.manifest);after.manifest['phases'][0]['shared_bytes']=0
    else:after.source=after.source.replace('A.q = v_q;','A.q = v_k;',1)
    with pytest.raises(AssertionError):independently_expand(before,after)


def test_audited_old_header_and_duplicate_modes_fail_closed():
    before=fixture();before.manifest['header_hashes'][ai.ORIGINAL]='bad'
    with pytest.raises(ai.Refusal):ai.fold(before)
    with pytest.raises(ai.Refusal):ai.fold(ai.fold(fixture()))


@pytest.mark.parametrize('token',['forward_item6','forward_item','NS','POW2','smem','Args','pflatlong','__noinline__'])
def test_wrapper_abi_macro_shadowing_refuses(token):
    before=fixture();before.source+='\n#define '+token+' changed\n'
    with pytest.raises(ai.Refusal):ai.fold(before)

def test_namespace_reopening_refuses():
    before=fixture();before.source+='\nnamespace pflatlong {}\n'
    with pytest.raises(ai.Refusal):ai.fold(before)


@pytest.mark.parametrize('field,value',[('phase',2),('pieces',3),('pow2',True),('wrapper','wrong'),('target','wrong')])
def test_inverse_binds_every_proof_call_to_its_actual_phase(field,value):
    before=fixture();after=ai.fold(before);after.manifest=copy.deepcopy(after.manifest)
    after.manifest[ai.KEY]['groups'][0][field]=value
    with pytest.raises(AssertionError):independently_expand(before,after)
