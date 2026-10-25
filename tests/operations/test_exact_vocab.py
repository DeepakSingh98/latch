"""Metadata-only proofs for the exact unpadded CE rewrite; no CUDA/tensor work."""
import copy

import pytest

from megakernel_lab.operation_compiler import exact_vocab as rule
from megakernel_lab.operation_compiler.ir import TrainingIR
from megakernel_lab.operation_compiler import vocab_head as head


def fixture(*, rows=8, vocab=10001, width=4, backward=True, scalar=False, chain=''):
    data = dict(format_version=1, inputs=['x','w','ids','dy','factor'], values={}, operations=[],
                outputs={'loss':'pick','grads':{'w':'dlogits'}} if backward else {'loss':'pick'},
                metadata={'log_softmax_contract':rule.LOG_SOFTMAX_CONTRACT,
                          'serial_bf16': {'products': {'example': {'forward_product':'product',
                              'dlogits':'dlogits','log_softmax':'ls'}}}}, state_writes={})
    def value(name, shape, dtype, **kw):
        data['values'][name] = dict(name=name, shape=list(shape), dtype=dtype,
            strides=head._contiguous(shape), role='temporary', parameter=None,
            alias_of=None, storage_offset=0, producer=name, output_roles=[], **kw)
    def op(name, target, args, kwargs=None):
        data['operations'].append(rule._operation(data,name,target,args,kwargs or {}))
    ref=lambda n:{'ref':n}
    for name,shape,dtype in [('x',[rows,width],'bfloat16'),('w',[vocab,width],'float32'),
                             ('ids',[rows,1],'int64'),('dy',[rows,1],'float32'),('factor',[],'float32')]:
        value(name,shape,dtype)
        data['values'][name].update(role='input',producer=None)
    data['values']['w'].update(role='parameter',parameter='table')
    for name,shape,dtype in [('wc',[vocab,width],'bfloat16'),('wt',[width,vocab],'bfloat16'),
                            ('product',[rows,vocab],'bfloat16'),('wide',[rows,vocab],'float32'),
                            ('ls',[rows,vocab],'float32'),('pick',[rows,1],'float32')]:
        value(name,shape,dtype)
    data['values']['wt'].update(strides=[1,width],alias_of='wc')
    op('wc','aten._to_copy.default',[ref('w')],{'dtype':'bfloat16'})
    op('wt','aten.t.default',[ref('wc')])
    op('product','aten.mm.default',[ref('x'),ref('wt')])
    op('wide','aten._to_copy.default',[ref('product')],{'dtype':'float32'})
    op('ls','aten._log_softmax.default',[ref('wide'),-1,False])
    op('pick','aten.gather.default',[ref('ls'),1,ref('ids'),False])
    if backward:
        for name,dtype in [('zeros','float32'),('scatter','float32'),('back','float32'),('dlogits','bfloat16')]:
            value(name,[rows,vocab],dtype)
        op('zeros','aten.zeros_like.default',[ref('ls')])
        op('scatter','aten.scatter.value' if scalar else 'aten.scatter_add.default',
           [ref('zeros'),1,ref('ids'),-0.25 if scalar else ref('dy')])
        previous='scatter'
        for i,kind in enumerate(chain):
            name=f'chain_{i}';value(name,[rows,vocab],'float32')
            op(name,{'*':'aten.mul.Tensor','~':'aten.neg.default','/':'aten.div.Tensor'}[kind],
               [ref(previous)] if kind=='~' else [ref(previous),ref('factor')])
            previous=name
        op('back','aten._log_softmax_backward_data.default',[ref(previous),ref('ls'),-1,'float32'])
        op('dlogits','aten._to_copy.default',[ref('back')],{'dtype':'bfloat16'})
    return data


@pytest.mark.parametrize('scalar,chain',[(False,''),(False,'*~'),(True,'~*')])
def test_rewrite_preserves_matrix_extent_names_metadata_and_rounding(scalar,chain):
    old=fixture(scalar=scalar,chain=chain);saved=copy.deepcopy(old)
    new,report=rule.rewrite(old)
    assert old==saved and len(report['rewritten'])==1 and not report['refused']
    assert new['metadata']['serial_bf16']==old['metadata']['serial_bf16']
    record=report['rewritten'][0]
    assert record['padded'] is False and record['vocabulary']==10001
    assert record['product']=='product' and record['dlogits']=='dlogits'
    for name in ('x','w','wc','wt','product','pick','dlogits'):
        assert new['values'][name]==old['values'][name]
    for name in ('wc','wt','product'):
        assert next(op for op in new['operations'] if op['name']==name)==next(op for op in old['operations'] if op['name']==name)
    assert not set(record['removed']) & set(new['values'])
    assert not set(record['removed']) & {op['name'] for op in new['operations']}
    grad=next(op for op in new['operations'] if op['target']==rule.TARGETS['grad'])
    assert grad['name']=='dlogits' and grad['kwargs']['chain']==chain
    assert grad['kwargs']['leaf_add'] is (not scalar)
    assert grad['args'][3]==(-0.25 if scalar else {'ref':'dy'})
    assert grad['args'][4:]==[None if c=='~' else {'ref':'factor'} for c in chain]
    assert all(r in new['values'] for op in new['operations'] for r in head._refs(op['args']))


def test_forward_only_and_training_ir_roundtrip():
    data=fixture(backward=False)
    data.pop('state_writes')
    old=TrainingIR.from_dict(data)
    before=copy.deepcopy(old.to_dict())
    new,report=rule.rewrite(old)
    assert isinstance(new,TrainingIR) and report['rewritten'][0]['dlogits'] is None
    assert sum(op.target==rule.TARGETS['stats'] for op in new.operations)==1
    assert sum(op.target==rule.TARGETS['pick'] for op in new.operations)==1
    assert not any(op.target==rule.TARGETS['grad'] for op in new.operations)
    assert old.to_dict()==before


@pytest.mark.parametrize('case',['no_policy','off','below','aligned_width','fp32_product','shifted_logits',
                                 'strided_logits','half_backward','division','state_value','state_key',
                                 'nested_output','output_role','alias','terminal_grad_reader','stats_collision',
                                 'negative_zero_fill'])
def test_refusals_and_off_preserve_input(case):
    old=fixture(vocab=9215 if case=='below' else 10000 if case=='aligned_width' else 10001,
                chain='/' if case=='division' else '')
    if case=='no_policy':old['metadata'].pop('log_softmax_contract')
    elif case=='fp32_product':old['values']['product']['dtype']='float32'
    elif case=='shifted_logits':old['values']['wide']['storage_offset']=4
    elif case=='strided_logits':old['values']['wide']['strides']=[10002,1]
    elif case=='half_backward':next(op for op in old['operations'] if op['name']=='back')['args'][3]='bfloat16'
    elif case=='state_value':old['state_writes']={'destination':'scatter'}
    elif case=='state_key':old['state_writes']={'scatter':'dy'}
    elif case=='nested_output':old['outputs']['nested']={'a':[{'b':'wide'}]}
    elif case=='output_role':old['values']['ls']['output_roles']=['diagnostic']
    elif case=='alias':old['values']['alias']={**old['values']['ls'],'name':'alias','alias_of':'ls'}
    elif case=='terminal_grad_reader':
        old['operations'].append(rule._operation(old,'extra','aten.neg.default',[{'ref':'scatter'}],{}))
        old['values']['extra']={**old['values']['scatter'],'name':'extra','producer':'extra'}
    elif case=='stats_collision':old['values']['exact_vocab_stats_product']=copy.deepcopy(old['values']['pick'])
    elif case=='negative_zero_fill':
        old['operations'][next(i for i,op in enumerate(old['operations']) if op['name']=='zeros')]=\
            rule._operation(old,'zeros','aten.full.default',[[8,10001],-0.0],{'dtype':'float32'})
    saved=copy.deepcopy(old)
    new,report=rule.rewrite(old,enabled=case!='off')
    assert new is old and old==saved
    if case not in ('no_policy','off'):
        assert report['refused'] and not report['rewritten']


def test_toggle_and_idempotence(monkeypatch):
    old=fixture()
    monkeypatch.setattr(rule,'EXACT_ROWS',False)
    assert rule.rewrite(old)==(old,None)
    new,report=rule.rewrite(old,enabled=True)
    assert report['rewritten']
    assert rule.rewrite(new,enabled=True)==(new,None)
    with pytest.raises(ValueError,match='Boolean'):rule.rewrite(old,enabled='on')
