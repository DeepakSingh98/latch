"""Exact emitter/body inverse and independent typed ABI qualification."""
from copy import deepcopy
import difflib
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import indexed_segmented as emitter
from megakernel_lab.operation_compiler import indexed_segmented_outline as rule


def emit(*, mode="off", n=513, width=48, rows=17, sorter="radix", reuse=False,
         scale=False, widen=False, stride=1, padding=-1):
    cache = {}
    def one(name):
        return emitter.emit(name, "gradient", "ids",
            SimpleNamespace(shape=(n, width), strides=(width * stride, stride)),
            SimpleNamespace(shape=(n,), strides=(1,), numel=n), rows, padding,
            width, lambda name: "v_" + name,
            lambda suffix, extent, dtype: name + "__" + suffix, cache,
            sorter=sorter, scale=("scale", SimpleNamespace(shape=(n,), strides=(1,))) if scale else None,
            widen=widen)
    with rule.preference(mode):
        first = one("output")
        return one("output2") if reuse else first


def independently_expand_phase(before, after, definitions):
    proof = after[rule.KEY]
    name = proof["helper"]
    definition, = [d for d in definitions if d.startswith("static __device__ __noinline__ void " + name + "(")]
    # Parse declaration and call independently; do not invoke the product renamer.
    match = re.fullmatch(r"static __device__ __noinline__ void " + name + r"\(([^()]*)\) (\{[\s\S]*\})\n", definition)
    assert match
    parameters = [x.rsplit(" ", 1) for x in match[1].split(", ")]
    types, formals = map(list, zip(*parameters))
    assert types == proof["parameter_types"]
    assert formals == proof["formal_names"]
    assert all(t in ("float*", "int64_t*", "__nv_bfloat16*", "unsigned") for t in types)
    call = re.fullmatch(re.escape(name) + r"\(([^()]*)\);", after["code"])
    assert call
    actuals = call[1].split(", ")
    assert actuals == proof["argument_names"] and len(actuals) == len(formals)
    assert types[actuals.index("smem")] == "float*"
    mapping = dict(zip(formals, actuals))
    expanded = re.sub(r"\blatch_is_arg\d+\b", lambda m: mapping[m[0]], match[2])
    assert expanded == before["code"]
    assert hashlib.sha256(expanded.encode()).hexdigest() == proof["original_body_sha256"]
    assert hashlib.sha256(match[2].encode()).hexdigest() == proof["normalized_body_sha256"]
    assert hashlib.sha256(definition.encode()).hexdigest() == proof["definition_sha256"]
    assert hashlib.sha256(after["code"].encode()).hexdigest() == proof["call_sha256"]
    assert expanded.count("__syncthreads();") == proof["body_barriers"]
    assert sorted(set(re.findall(r'\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b',expanded)))==proof['resource_pointer_lexical_bases']
    assert {k:v for k,v in after.items() if k not in ("code", rule.KEY)} == {k:v for k,v in before.items() if k != "code"}


@pytest.mark.parametrize("options", [
    {}, {"width":3}, {"width":192}, {"width":1024}, {"n":1}, {"n":8193},
    {"sorter":"tile"}, {"sorter":"tile", "n":20000}, {"reuse":True},
    {"sorter":"tile", "reuse":True}, {"padding":0}, {"padding":16},
    {"scale":True}, {"widen":True}, {"stride":2},
    {"scale":True,"stride":2}, {"widen":True,"width":3}])
def test_exact_emitter_inverse_and_metadata(options):
    old, decision = emit(**options)
    new, changed = emit(mode="typed_v1", **options)
    helpers = changed.pop("outlined_helpers")
    assert changed == decision and len(old) == len(new)
    count=0
    for a,b in zip(old,new):
        if rule.KEY in b:
            independently_expand_phase(a,b,helpers);count+=1
        else: assert a==b
    assert count == len(helpers) == 2
    assert all("__restrict__" not in h for h in helpers)


def test_default_off_and_context_restoration():
    baseline=emit()
    with rule.preference("typed_v1"):
        assert rule.current_preference()=="typed_v1"
        with rule.preference("off"):assert rule.current_preference()=="off"
        assert rule.current_preference()=="typed_v1"
    assert rule.current_preference()=="off" and emit()==baseline
    with pytest.raises(RuntimeError):
        with rule.preference("typed_v1"):raise RuntimeError()
    assert rule.current_preference()=="off"


@pytest.mark.parametrize("mode", [None, True, "segmented_v1", "typed_v2"])
def test_unknown_preference_refused(mode):
    with pytest.raises(rule.Refusal):
        with rule.preference(mode): pass


def simple():
    return {"kind":"index_segment_partials", "reads":["x"], "writes":["y"],
            "code":"{ smem[threadIdx.x] = v_x[blockIdx.x]; __syncthreads(); v_y[threadIdx.x] = smem[threadIdx.x]; }"}


def test_argument_order_is_first_use_and_types_preserve_aliases():
    phase=simple();bindings=[("v_y","float*"),("v_x","float*"),("smem","float*")]
    a,h=rule.outline(phase,bindings)
    b,j=rule.outline(phase,list(reversed(bindings))+[("v_x","float*")])
    assert (a,h)==(b,j)
    assert a[rule.KEY]["argument_names"]==["smem","threadIdx.x","v_x","blockIdx.x","v_y"]
    independently_expand_phase(phase,a,[h])


def test_comments_and_pragma_strings_are_not_renamed():
    phase=simple();phase["code"]=phase["code"].replace("{ ",'{ /* v_x smem */ _Pragma("v_x") // v_y\n',1)
    a,h=rule.outline(phase,[("smem","float*"),("v_x","float*"),("v_y","float*")])
    assert '/* v_x smem */ _Pragma("v_x") // v_y' in h
    independently_expand_phase(phase,a,[h])


@pytest.mark.parametrize("mutation", ["kind", "already", "scope", "unknown", "reserved", "alias_type", "smem_type", "type", "expression"])
def test_unproven_emitter_bindings_refused(mutation):
    p=simple();bindings=[("smem","float*"),("v_x","float*"),("v_y","float*")]
    if mutation=="kind":p["kind"]="elementwise"
    elif mutation=="already":p[rule.KEY]={}
    elif mutation=="scope":p["code"]=p["code"][1:]
    elif mutation=="unknown":p["code"]=p["code"].replace("v_x","v_hidden")
    elif mutation=="reserved":p["code"]=p["code"].replace("v_x","latch_is_arg0")
    elif mutation=="alias_type":bindings.append(("v_x","int64_t*"))
    elif mutation=="smem_type":bindings[0]=("smem","int64_t*")
    elif mutation=="type":bindings[1]=("v_x","void*")
    else:bindings[1]=("v_x + 1","float*")
    with pytest.raises(rule.Refusal):rule.outline(p,bindings)


@pytest.mark.parametrize("mutation", ["body", "signature", "call", "shared_type", "barrier", "reads", "work", "proof"])
def test_independent_inverse_detects_corruption(mutation):
    old,_=emit();new,d=emit(mode="typed_v1")
    i=next(i for i,p in enumerate(new) if rule.KEY in p);a,b=old[i],deepcopy(new[i]);helpers=list(d["outlined_helpers"])
    if mutation=="body":helpers[0]=helpers[0].replace("__fadd_rn", "bad_add",1) if "__fadd_rn" in helpers[0] else helpers[0].replace("seg_add", "bad_add",1)
    elif mutation=="signature":helpers[0]=helpers[0].replace("int64_t*", "float*",1)
    elif mutation=="call":b["code"]=b["code"].replace("v_gradient", "v_output")
    elif mutation=="shared_type":helpers[0]=helpers[0].replace("float*", "void*",1)
    elif mutation=="barrier":helpers[0]=helpers[0].replace("__syncthreads();", "",1)
    elif mutation=="reads":b["reads"]=[]
    elif mutation=="work":b["work_items"]+=1
    else:b[rule.KEY]["argument_names"].reverse()
    with pytest.raises((AssertionError,KeyError)):independently_expand_phase(a,b,helpers)


def independently_expand_program(before, after):
    """Undo actual helper definitions/calls, then compare the whole old source."""
    from .fixtures.tinymoe_source_proofs import close_brace
    source=after.source
    pattern=re.compile(r"static __device__ __noinline__ void (latch_is_\w+)\(([^()]*)\) \{")
    definitions={}
    for match in reversed(list(pattern.finditer(source))):
        end=close_brace(source,match.end()-1)
        definitions[match[1]]=([s.rsplit(" ",1) for s in match[2].split(", ")],source[match.end()-1:end+1])
        assert source[match.start()-1]=="\n" and source[end+1]=="\n"
        source=source[:match.start()-1]+source[end+2:]
    proofs=[p[rule.KEY] for p in after.manifest["phases"] if rule.KEY in p]
    assert set(definitions)=={p["helper"] for p in proofs}
    for proof in proofs:
        parameters,body=definitions[proof["helper"]]
        types,formals=map(list,zip(*parameters))
        assert types==proof["parameter_types"] and formals==proof["formal_names"]
        expected=[]
        for actual in proof["argument_names"]:
            if actual=="smem":expected.append("float*")
            elif actual in ("threadIdx.x","blockIdx.x","blockDim.x","gridDim.x"):expected.append("unsigned")
            else:
                assert actual.startswith("v_")
                dtype=before.manifest['layouts'][actual[2:]]['dtype']
                expected.append({'float32':'float*','int64':'int64_t*','bfloat16':'__nv_bfloat16*'}[dtype])
        assert types==expected, 'helper ABI differs from original typed storage/builtins'
        original_map=dict(zip(formals,proof['argument_names']))
        original_body=re.sub(r"\blatch_is_arg\d+\b",lambda m:original_map[m[0]],body)
        assert hashlib.sha256(original_body.encode()).hexdigest()==proof['original_body_sha256']
        assert hashlib.sha256(body.encode()).hexdigest()==proof['normalized_body_sha256']
        definition='static __device__ __noinline__ void '+proof['helper']+'('+', '.join(t+' '+n for t,n in parameters)+') '+body+'\n'
        assert hashlib.sha256(definition.encode()).hexdigest()==proof['definition_sha256']
        original_call=proof['helper']+'('+', '.join(proof['argument_names'])+');'
        assert hashlib.sha256(original_call.encode()).hexdigest()==proof['call_sha256']
        assert body.count('__syncthreads();')==proof['body_barriers']
        assert sorted(set(re.findall(r'\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b',original_body)))==proof['resource_pointer_lexical_bases']
        # The normal lowerer captures CUDA builtins as unsigned opaque registers.
        aliases={"threadIdx.x":"lsr_tx","blockIdx.x":"lsr_bx","blockDim.x":"lsr_nt","gridDim.x":"lsr_nb"}
        actuals=[aliases.get(n,n) for n in proof["argument_names"]]
        call=proof["helper"]+"("+", ".join(actuals)+");"
        assert source.count(call)==1
        mapping=dict(zip(formals,actuals))
        source=source.replace(call,re.sub(r"\blatch_is_arg\d+\b",lambda m:mapping[m[0]],body),1)
    assert source==before.source, "non-helper source, bindings, arithmetic or barriers changed\n"+'\n'.join(
        list(difflib.unified_diff(before.source.splitlines(),source.splitlines(),n=2))[:100])
    clean=deepcopy(after.manifest);prior=deepcopy(before.manifest)
    clean.pop(rule.KEY)
    for p in clean["phases"]:p.pop(rule.KEY,None)
    # Existing post-emission passes record input hashes and byte counts. The
    # complete source inverse above binds all of their actual generated code;
    # only these derived diagnostics change when bodies move into helpers.
    derived=("source_before","source_bytes_before","source_bytes_after",
             "entry_body_bytes_before","entry_body_bytes_after")
    for key in ("elementwise_codegen","full_sum_codegen","mixed_phase_pointer_codegen",
                "resource_phase_pointer_codegen","pure_interval_codegen","pure_run_codegen"):
        assert (key in clean)==(key in prior)
        if key in clean:
            assert clean[key].keys()==prior[key].keys()
            for field in derived:
                clean[key].pop(field,None);prior[key].pop(field,None)
    def differences(a,b,path=''):
        if type(a) is not type(b):return [(path,str(a),str(b))]
        if isinstance(a,dict):
            if a.keys()!=b.keys():return [(path,sorted(a),sorted(b))]
            return [row for k in a for row in differences(a[k],b[k],path+'.'+str(k))]
        if isinstance(a,list):
            if len(a)!=len(b):return [(path,len(a),len(b))]
            return [row for i,(x,y) in enumerate(zip(a,b)) for row in differences(x,y,path+'.'+str(i))]
        return [] if a==b else [(path,a,b)]
    diff=differences({k:v for k,v in clean.items() if k!="source_hash"},{k:v for k,v in prior.items() if k!="source_hash"})
    assert not diff,json.dumps(diff,indent=2)


def test_manifest_integer_keys_keep_exact_value_comparison():
    before=SimpleNamespace(source='',manifest={'phases':[], 'mapping':{3:{'value':7},'named':9}})
    after=deepcopy(before);after.manifest[rule.KEY]={'mode':'typed_v1'}
    independently_expand_program(before,after)
    after.manifest['mapping'][3]['value']=8
    with pytest.raises(AssertionError,match=r'mapping\.3\.value'):
        independently_expand_program(before,after)


@pytest.fixture(scope="module")
def captured():
    import torch
    import torch.nn.functional as F
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    def objective(p,ids,g):
        return ((F.embedding(ids,p['a'])+F.embedding(ids,p['b']))*g).sum()
    return capture_training(objective,{'a':torch.zeros(17,48),'b':torch.zeros(17,48)},None,
        (torch.zeros(513,dtype=torch.int64),torch.zeros(513,48)),
        tensor_update(lambda p,g,s,hp:(dict(p),{}),name='indexed_outline_identity'))


@pytest.mark.parametrize("compact", ["off","elementwise_full_sum_resource_pure_runs"])
def test_ordinary_common_source_inverse(captured,compact):
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    kwargs=dict(execution="compose",compose_selection="common",compact_codegen=compact)
    with policy("segmented_v1"):
        old=lower(captured.ir,**kwargs)
        with rule.preference("off"):off=lower(captured.ir,**kwargs)
        with rule.preference("typed_v1"):new=lower(captured.ir,**kwargs)
    assert old.source==off.source and old.manifest==off.manifest
    independently_expand_program(old,new)
    assert new.manifest[rule.KEY]["outlined_phases"]==4


@pytest.mark.parametrize("kwargs", [{"execution":"legacy"},{"compose_selection":"unknown"},{"retain_all":True},{"staged":False}])
def test_inappropriate_normal_lowering_refused(captured,kwargs):
    from megakernel_lab.operation_compiler.lower import lower,LoweringError
    options=dict(execution="compose",compose_selection="common");options.update(kwargs)
    with rule.preference("typed_v1"),pytest.raises((LoweringError,ValueError)):
        lower(captured.ir,**options)


def test_auto_option_does_not_admit_a_noncommon_selected_program():
    with rule.preference('typed_v1'):
        rule.validate_lowering(SimpleNamespace(metadata={}),compose=True,common=True,depth=1,retain_all=False,options={})
        bad=SimpleNamespace(manifest={'phases':[{rule.KEY:{'helper':'unused'}}],
            'execution_selection':{'selected':'flat_static'}},source='')
        with pytest.raises(rule.Refusal,match='common_static'):rule.annotate(bad)


@pytest.mark.parametrize('change',['typed_metadata','original_hash','normalized_hash','call_hash','barriers','dependencies','lexical_context'])
def test_whole_program_inverse_rejects_metadata_or_abi_damage(captured,change):
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    with policy('segmented_v1'):
        before=lower(captured.ir,execution='compose',compose_selection='common')
        with rule.preference('typed_v1'):
            after=lower(captured.ir,execution='compose',compose_selection='common')
    phase=next(p for p in after.manifest['phases'] if rule.KEY in p);proof=phase[rule.KEY]
    if change=='typed_metadata':
        i=proof['parameter_types'].index('int64_t*')
        after.source=after.source.replace('int64_t* '+proof['formal_names'][i],'float* '+proof['formal_names'][i],1)
        proof['parameter_types'][i]='float*'
    elif change=='original_hash':proof['original_body_sha256']='0'*64
    elif change=='normalized_hash':proof['normalized_body_sha256']='0'*64
    elif change=='call_hash':proof['call_sha256']='0'*64
    elif change=='barriers':proof['body_barriers']+=1
    elif change=='lexical_context':proof['resource_pointer_lexical_bases']=['args','arena']
    else:phase['reads']=[]
    with pytest.raises(AssertionError):independently_expand_program(before,after)
