"""CPU metadata contracts for the pinned B adapter and global resource choice."""
from __future__ import annotations
import traceback

from megakernel_lab.operation_compiler.lower import TensorLayout, _contiguous
from megakernel_lab.operation_compiler.matrix_codegen import fp32_async, select
from megakernel_lab.operation_compiler.resources import select_program_resources
from megakernel_lab.operation_compiler.runtime import select_launch_policy


def layout(shape,name,strides=None,offset=0):
    return TensorLayout(shape,strides or _contiguous(shape),name,offset)


def test_layout_admission_and_existing_fallbacks():
    rows=[]
    for at in (False,True):
        for bt in (False,True):
            a=layout((64,1024),'a',(1,64) if at else None)
            b=layout((1024,128),'b',(1,1024) if bt else None)
            plan,reason=fp32_async('a','b','out',a,b,(64,128),ident=lambda n:'v_'+n)
            assert plan,reason
            assert plan['metadata']['requires_full_grid']
            assert plan['metadata']['engine']=='popsb_cpasync'
            assert '/*B_VBASE*/' in plan['code']
            rows.append([at,bt,plan['metadata']['ksplit']])
    a,b=layout((64,1024),'a'),layout((1024,64),'b')
    for kwargs in ({'arch':'sm_80'},{'accumulate':True}):
        assert fp32_async('a','b','out',a,b,(64,64),ident=str,**kwargs)[0] is None
    assert fp32_async('a','b','out',layout((16,48),'a'),layout((48,10),'b'),(16,10),ident=str)[0] is None
    assert fp32_async('a','b','out',layout((64,1024),'a',offset=1),b,(64,64),ident=str)[0] is None
    # C keeps first refusal for eligible BF16 and small batched FP32.
    ba,bb=layout((2,64,64),'a'),layout((2,64,64),'b')
    assert select('a','b','out',ba,bb,(2,64,64),'float32')[0] is not None
    return {'four_layouts':rows,'tails_alignment_arch_accumulation_fallback':True}


def test_private_split_scratch_and_collision():
    a,b=layout((64,1024),'a'),layout((1024,64),'b')
    first,_=fp32_async('a','b','first',a,b,(64,64),ident=lambda n:'v_'+n)
    second,_=fp32_async('a','b','second',a,b,(64,64),ident=lambda n:'v_'+n)
    for plan in (first,second):
        assert plan['metadata']['ksplit']>1
        root,=plan['scratch']
        assert root['counter_words']%64==0 and root['partial_words']>0
        assert root['shape']==(root['counter_words']+root['partial_words'],)
        assert plan['counter_init']['writes']==[root['name']]
        assert str(root['counter_words'])+'LL' in plan['counter_init']['code']
        assert plan['metadata']['internal_mutations']==[root['name']]
        assert 'counter words zero; partial words defined' in root['initialization']
    assert first['scratch'][0]['name']!=second['scratch'][0]['name']
    assert fp32_async('a','b','first',a,b,(64,64),ident=str,
                     occupied_names=['first__popsb_scratch'])[0] is None
    unsplit,_=fp32_async('a','b','first',a,b,(64,64),ident=str,split=False)
    assert not unsplit['scratch'] and unsplit['counter_init'] is None
    assert 'nullptr, nullptr' in unsplit['code']
    return {'per_product_isolation':True,'initialization_explicit':True,'collision_fallback':True}


def test_whole_program_resource_ceiling_and_launch():
    phase={'kind':'gemm','engine':'popsb_cpasync','tiles':500,'selection_grid':128}
    low=select_program_resources(17408,phases=[phase])
    assert low['resident_blocks_per_sm']==2
    for shared in (98305,131840,232448):
        high=select_program_resources(shared,children=[{'phases':[phase]}])
        assert high['resident_blocks_per_sm']==1 and high['has_popsb']
    ordinary=select_program_resources(17408,phases=[{'engine':'pinned_simt_64'}])
    assert ordinary['resident_blocks_per_sm']==1
    nested=select_program_resources(131840,children=[{'branches':{'true':{'resource_policy':low},
                  'false':{'shared_bytes':131840}}}])
    assert nested['has_popsb'] and nested['resident_blocks_per_sm']==1
    mf={'shared_bytes':17408,'resource_policy':low,'phases':[phase]}
    assert select_launch_policy(mf,396,multiprocessors=132)['selected_blocks']==264
    assert select_launch_policy(mf,132,multiprocessors=132)['selected_blocks']==132
    assert select_launch_policy(mf,396,blocks=7,multiprocessors=132)['selected_blocks']==7
    old={'shared_bytes':17408,'phases':[phase]}
    assert select_launch_policy(old,396)['selected_blocks']==128
    try:
        select_launch_policy({**mf,'shared_bytes':131840},396,multiprocessors=132)
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete resource proof admitted')
    return {'low_shared_two_cta':True,'attention_large_ring_one_cta':True,
            'actual_occupancy_clamp':True,'legacy128_unchanged':True}


def test_flat_fusion_scratch_dependencies_and_grouping():
    from megakernel_lab.operation_compiler.lower import lower
    shapes={'a':(64,1024),'b':(1024,64),'c':(1024,64),'x':(64,64),'y':(64,64)}
    graph={'inputs':['a','b','c'],'outputs':['x','y'],
           'values':{name:{'shape':shape,'strides':_contiguous(shape),'dtype':'float32'}
                     for name,shape in shapes.items()},
           'operations':[{'name':out,'target':'aten.mm.default',
                          'args':[{'ref':'a'},{'ref':other}],'kwargs':{}}
                         for out,other in [('x','b'),('y','c')]]}
    program=lower(graph, execution="legacy", pops2_policy=False)
    phases=program.manifest['phases']
    products=[(i,p) for i,p in enumerate(phases) if p.get('engine')=='popsb_cpasync']
    assert len(products)==1 and set(products[0][1]['names'])=={'x','y'}
    for root in ('x__popsb_scratch','y__popsb_scratch'):
        producers=[i for i,p in enumerate(phases) if root in p['writes']]
        assert len(producers)==1 and producers[0]<products[0][0]
        assert root in products[0][1]['reads']
        assert program.manifest['lifetimes'][root][2]>=products[0][0]
    assert program.manifest['offsets']['x__popsb_scratch']!=program.manifest['offsets']['y__popsb_scratch']
    assert '/*B_VBASE*/' not in program.source and 'latch_opq(args)' in program.source
    assert program.manifest['resource_policy']['resident_blocks_per_sm']==2
    return {'common_scheduler_groups_products':True,'private_scratch_live_through_product':True}


def run():
    tests=[]
    for fn in (test_layout_admission_and_existing_fallbacks,test_private_split_scratch_and_collision,
               test_whole_program_resource_ceiling_and_launch,test_flat_fusion_scratch_dependencies_and_grouping):
        try:tests.append({'name':fn.__name__,'passed':True,'detail':fn()})
        except Exception:tests.append({'name':fn.__name__,'passed':False,'error':traceback.format_exc()})
    return {'kind':'B-primitive-and-program-resource-contract','tests':tests,
            'passed':all(row['passed'] for row in tests)}
