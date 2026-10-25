"""Mutation checks for the independent complete-update evidence gate."""
from dataclasses import asdict
import copy
import importlib.util
import json
from pathlib import Path

import torch
import pytest

from megakernel_lab.tinymoe.model import Recipe
from megakernel_lab.tinymoe.optimizer import make_rule,initial_model_state,stage_schedule,CLIP

from megakernel_lab.tinymoe import audit
POLICY=json.loads(audit.POLICY_PATH.read_text())


def observation():
    model=torch.nn.Linear(3,2);model.config=type('Config',(),{'num_hidden_layers':0})()
    recipe=Recipe(name='supervision_ema',horizon=400)
    params={k:p.detach().clone() for k,p in model.named_parameters()}
    rule=make_rule(model,recipe,ema_active=True);state=rule.init_state(params,{})
    stage_schedule(state,recipe,199)
    gradients={k:torch.full_like(p,3.) for k,p in params.items()}
    updated,new=rule.update(params,gradients,state,{}, {})
    return dict(before=dict(parameters=params,optimizer_state=state),
        after=dict(parameters=updated,optimizer_state=new,model_state={}),
        recipe=asdict(recipe),update_index=199,gradients=gradients)


def test_own_gradient_formula_and_activation_pass():
    assert audit.optimizer_audit(observation(),POLICY)['passed']


def test_corrupt_clip_coefficient_is_rejected():
    item=observation();item['after']['optimizer_state'][CLIP].fill_(1.)
    result=audit.optimizer_audit(item,POLICY)
    assert not result['passed']
    assert result['coefficient_relative_error']>POLICY['same_state_fixed_membership_complete_update']['clip_norm_relative_max']


def test_corrupt_ema_boundary_is_rejected():
    item=observation();item['after']['optimizer_state']['ema:weight'].add_(.01)
    assert not audit.optimizer_audit(item,POLICY)['passed']


def test_exact_gate_distinguishes_signed_zero_including_scalars():
    assert not audit.bitwise_equal(torch.tensor(0.),torch.tensor(-0.))
    assert audit.bitwise_equal(torch.tensor(3,dtype=torch.int64),torch.tensor(3,dtype=torch.int64))


def test_all_observation_tensors_must_be_finite():
    item=observation();item['before']['optimizer_state']['ema:bias'].fill_(float('inf'))
    result=audit.optimizer_audit(item,POLICY)
    assert not result['passed']
    assert 'observation.before.optimizer_state.ema:bias' in result['nonfinite_tensors']


def test_participation_flags_must_match_actual_memberships():
    item=observation();item['fixed_routes']=torch.tensor([[[0,2],[2,3]]])
    item['after']['model_state']={'active.0':torch.tensor([1.,0.,1.,1.])}
    assert audit.activity_audit(item)['passed']
    item['after']['model_state']['active.0'][2]=0
    assert not audit.activity_audit(item)['passed']


def complete_observation(native=False):
    item=observation()
    item['before']['next_step']=199;item['after']['next_step']=200
    item['loss']=torch.tensor(3.)
    if native:
        identity={'artifacts':{'plain':'frozen-artifact'},'wrappers':{'runtime.py':'frozen-runtime'}}
        item['identity']=copy.deepcopy(identity)
        for key in ('before','after'):
            item[key].update(schema='latch-tinymoe-native-state-v1',
                             recipe=copy.deepcopy(item['recipe']),identity=copy.deepcopy(identity))
        state=item['after']['optimizer_state']
        item['after']['diagnostics']=dict(loss=item['loss'].clone(),
            gradient_norm=state[audit.NORM].clone(),clip_coefficient=state[CLIP].clone())
    return item


def source_receipt():
    return dict(sha256='a'*64,bytes=123,completed_update=200)


@pytest.mark.parametrize('native',[False,True])
def test_complete_observation_contract_binds_cursor_and_source(native):
    item=complete_observation(native)
    result=audit.observation_contract(item,source_receipt())
    assert result['passed'] and result['completed_update']==200
    assert result['observation_sha256']=='a'*64
    assert result['backend_kind']==('native' if native else 'conventional')


@pytest.mark.parametrize('state,key,value',[
    ('before','next_step',198),('after','next_step',201),('after','next_step',200.),
    ('before','schema','wrong'),('after','schema','wrong'),
    ('before','identity',{'artifacts':'different'}),('after','identity',{'artifacts':'different'}),
    ('before','recipe',{'name':'unchanged'}),('after','recipe',{'name':'unchanged'}),
])
def test_mislabeled_native_state_is_rejected_before_numerical_work(state,key,value):
    item=complete_observation(True);item[state][key]=value
    with pytest.raises(ValueError):audit.observation_contract(item,source_receipt())


@pytest.mark.parametrize('change',['index','loss','norm','clip','top_identity','sidecar_cursor','sidecar_type','sidecar_identity'])
def test_duplicate_labels_and_diagnostics_cannot_disagree(change):
    item=complete_observation(True);receipt=source_receipt()
    if change=='index':item['update_index']=198
    elif change=='loss':item['after']['diagnostics']['loss'].add_(1)
    elif change=='norm':item['after']['diagnostics']['gradient_norm'].add_(1)
    elif change=='clip':item['after']['diagnostics']['clip_coefficient'].add_(1)
    elif change=='top_identity':item['identity']={'artifacts':'different'}
    elif change=='sidecar_cursor':receipt['completed_update']=201
    elif change=='sidecar_type':receipt['completed_update']=200.
    elif change=='sidecar_identity':receipt['identity']={'artifacts':'different'}
    with pytest.raises(ValueError):audit.observation_contract(item,receipt)


def test_canonical_snapshot_view_preserves_payload_and_checks_duplicate_routes():
    item=complete_observation(True)
    item['after']['model_state']={'selected.0':torch.tensor([[0,1],[1,2]])}
    item.pop('update_index');item.pop('recipe');item.pop('loss')
    view=audit.canonical_observation(item)
    assert 'update_index' not in item and 'fixed_routes' not in item
    assert audit.observation_contract(view,source_receipt())['passed']
    view['fixed_routes'][0,0,0]=2
    with pytest.raises(ValueError,match='memberships'):
        audit.observation_contract(view,source_receipt())
    view['fixed_routes']=view['fixed_routes'][:0]
    with pytest.raises(ValueError,match='layers'):
        audit.observation_contract(view,source_receipt())


def test_public_snapshot_duplicate_gradient_spellings_must_match():
    item=complete_observation(True)
    item['after']['gradients_before_clip']={key:value.clone() for key,value in item['gradients'].items()}
    assert audit.observation_contract(item,source_receipt())['passed']
    item['after']['gradients_before_clip']['weight'].add_(1)
    with pytest.raises(ValueError,match='gradients'):
        audit.observation_contract(item,source_receipt())
