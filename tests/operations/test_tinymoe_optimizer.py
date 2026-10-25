import torch
from megakernel_lab.tinymoe.model import Recipe
from megakernel_lab.tinymoe.optimizer import make_rule, initial_model_state, stage_schedule, EXPERT


def tiny_model():
    from transformers import MixtralConfig, MixtralForCausalLM
    return MixtralForCausalLM(MixtralConfig(hidden_size=16,intermediate_size=24,
        num_attention_heads=2,num_key_value_heads=1,num_hidden_layers=1,
        num_local_experts=4,num_experts_per_tok=2,vocab_size=32))


@torch.no_grad()
def test_unused_expert_moments_steps_and_decay_stay_unchanged():
    model=tiny_model();recipe=Recipe(horizon=400)
    rule=make_rule(model,recipe)
    params=dict(model.named_parameters());state=rule.init_state(params,{})
    gradients={k:torch.zeros_like(v) for k,v in params.items()}
    for k,v in state.items():
        if k.startswith('exp_avg:'):v.fill_(.03)
        if k.startswith('exp_avg_sq:'):v.fill_(.01)
        if k.startswith('step:'):v.fill_(7)
    active=initial_model_state(model);active['active.0'][:2]=1
    stage_schedule(state,recipe,7)
    updated,new=rule.update(params,gradients,state,{},active)
    for name,p in params.items():
        match=EXPERT.match(name)
        if match and int(match[2])>=2:
            assert torch.equal(updated[name],p)
            for slot in ('exp_avg','exp_avg_sq','step'):
                assert torch.equal(new[slot+':'+name],state[slot+':'+name])
        else:
            assert int(new['step:'+name])==8
            assert not torch.equal(updated[name],p)


@torch.no_grad()
def test_ema_activation_copies_updated_raw_including_unused_experts():
    model=tiny_model();recipe=Recipe(name='supervision_ema',horizon=400)
    rule=make_rule(model,recipe,ema_active=True)
    params=dict(model.named_parameters());state=rule.init_state(params,{})
    gradients={k:torch.zeros_like(v) for k,v in params.items()}
    active=initial_model_state(model);active['active.0'][:2]=1
    stage_schedule(state,recipe,199)
    updated,new=rule.update(params,gradients,state,{},active)
    for name,p in params.items():
        assert torch.equal(updated[name],new['ema:'+name])
