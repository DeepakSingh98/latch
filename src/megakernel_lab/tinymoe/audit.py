"""Independent CPU audits for observed complete updates and fixed membership.

Run on an admitted CPU-only worker. Observations are saved by the GPU harness;
this module never rebuilds or executes the generated program. Reference model
execution uses installed Transformers and ordinary autograd, with only routing
membership supplied and FP64 RMS normalization for the high-precision audit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import types

import torch
from torch.nn import functional as F

from megakernel_lab.tinymoe.model import Recipe, load_model
from megakernel_lab.tinymoe.optimizer import EXPERT, NORM, CLIP


POLICY_PATH = Path(__file__).with_name('POLICY.json')


def canonical_observation(observation):
    """Normalize public snapshot spellings in memory, preserving source evidence."""
    value=dict(observation)
    value.setdefault('gradients',value['after'].get('gradients_before_clip'))
    value.setdefault('recipe',value['before'].get('recipe'))
    value.setdefault('update_index',value['before'].get('next_step'))
    value.setdefault('loss',value['after'].get('loss',
        value['after'].get('diagnostics',{}).get('loss')))
    if 'fixed_routes' not in value:
        states=value['after'].get('model_state',value['after'].get('state',{}))
        selected={int(k.split('.')[1]):v for k,v in states.items() if k.startswith('selected.')}
        if selected:
            if sorted(selected)!=list(range(len(selected))):
                raise ValueError('native route observations have missing layers')
            value['fixed_routes']=torch.stack([selected[i] for i in sorted(selected)])
    return value


def observation_contract(observation, receipt):
    """Bind the numerical update to its actual cursor, recipe and native identity."""
    before,after=observation['before'],observation['after']
    recipe=observation['recipe'];canonical=asdict(Recipe(**recipe))
    index=observation['update_index']
    if (canonical!=recipe or type(canonical['horizon']) is not int
            or type(index) is not int or not 0<=index<canonical['horizon']):
        raise ValueError('observation recipe or update index is invalid')
    if (type(before.get('next_step')) is not int or before['next_step']!=index
            or type(after.get('next_step')) is not int or after['next_step']!=index+1):
        raise ValueError('observation before/after cursor does not match the update')
    if 'completed_update' in receipt and (type(receipt['completed_update']) is not int
            or receipt['completed_update']!=index+1):
        raise ValueError('observation sidecar cursor differs from payload')
    native=any('schema' in state or 'identity' in state for state in (before,after)) or 'identity' in observation
    identity=None
    if native:
        identity=observation.get('identity',before.get('identity'))
        if not isinstance(identity,dict) or not identity:
            raise ValueError('native observation identity is missing')
        for state in (before,after):
            if (state.get('schema')!='latch-tinymoe-native-state-v1'
                    or state.get('recipe')!=recipe or state.get('identity')!=identity):
                raise ValueError('native observation schema, recipe or identity differs across update')
        if 'identity' in receipt and receipt['identity']!=identity:
            raise ValueError('native observation sidecar identity differs from payload')
    else:
        for state in (before,after):
            if 'recipe' in state and state['recipe']!=recipe:
                raise ValueError('observation state recipe differs from objective recipe')
    loss=observation['loss']
    if not isinstance(loss,torch.Tensor) or loss.numel()!=1 or not bool(torch.isfinite(loss).all()):
        raise ValueError('observation loss must be one finite tensor value')
    diagnostics=after.get('diagnostics',{})
    for key,value in (('loss',loss),('gradient_norm',after['optimizer_state'][NORM]),
                      ('clip_coefficient',after['optimizer_state'][CLIP])):
        if key in diagnostics and not bitwise_equal(diagnostics[key],value):
            raise ValueError('observation duplicate diagnostic differs: '+key)
    if 'loss' in after and not bitwise_equal(after['loss'],loss):
        raise ValueError('observation duplicate loss differs')
    if 'gradients_before_clip' in after:
        nested=after['gradients_before_clip'];top=observation['gradients']
        if (nested.keys()!=top.keys() or any(nested[key] is not top[key]
                and not bitwise_equal(nested[key],top[key]) for key in nested)):
            raise ValueError('observation duplicate gradients differ')
    selected=after.get('model_state',{})
    if 'fixed_routes' in observation:
        selected_layers={int(key.split('.')[1]) for key in selected if key.startswith('selected.')}
        if selected_layers and selected_layers!=set(range(len(observation['fixed_routes']))):
            raise ValueError('observation duplicate membership layers differ')
        for layer,routes in enumerate(observation['fixed_routes']):
            if f'selected.{layer}' in selected and not bitwise_equal(selected[f'selected.{layer}'],routes):
                raise ValueError('observation duplicate memberships differ')
    return dict(passed=True,update_index=index,completed_update=index+1,recipe=recipe,
                backend_kind='native' if native else 'conventional',identity=identity,
                observation_sha256=receipt['sha256'],observation_bytes=receipt['bytes'],
                validator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def bitwise_equal(actual, expected):
    return (actual.shape==expected.shape and actual.dtype==expected.dtype and
            torch.equal(actual.detach().contiguous().reshape(-1).view(torch.uint8),
                        expected.detach().contiguous().reshape(-1).view(torch.uint8)))


def nonfinite_tensors(value, path='observation'):
    if isinstance(value,torch.Tensor):
        return [] if bool(torch.isfinite(value).all()) else [path]
    if isinstance(value,dict):
        return [p for k,v in value.items() for p in nonfinite_tensors(v,path+'.'+str(k))]
    if isinstance(value,(tuple,list)):
        return [p for i,v in enumerate(value) for p in nonfinite_tensors(v,path+'.'+str(i))]
    return []


def distance(actual, expected, *, floor=1e-20):
    a, b = actual.detach().double(), expected.detach().double()
    d = a-b
    return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
                max_abs=float(d.abs().max()),
                relative_l2=float(d.norm()/b.norm().clamp_min(floor*math.sqrt(max(1,b.numel())))),
                reference_rms=float(b.norm()/math.sqrt(max(1,b.numel()))))


def activity_audit(observation):
    routes=observation.get('fixed_routes')
    if routes is None:return dict(applicable=False,passed=True)
    state=observation['after']['model_state'];rows=[]
    for layer,selected in enumerate(routes):
        active=state[f'active.{layer}']
        counts=F.one_hot(selected.long(),active.numel()).sum((0,1))
        correct=torch.equal(active>0,counts>0)
        duplicate=bool((selected.sort(-1).values[:,1:]==selected.sort(-1).values[:,:-1]).any())
        rows.append(dict(layer=layer,expert_counts=counts.tolist(),
                         unused_experts=int((counts==0).sum()),passed=correct and not duplicate))
    return dict(applicable=True,layers=rows,passed=all(r['passed'] for r in rows))


def optimizer_audit(observation, policy):
    """FP64 formula driven by the candidate's own raw gradients and membership."""
    before, after = observation['before'], observation['after']
    recipe = Recipe(**observation['recipe'])
    params, opt = before['parameters'], before['optimizer_state']
    new_params, new_opt = after['parameters'], after['optimizer_state']
    grads = observation['gradients']
    nonfinite=nonfinite_tensors(observation)
    activity=activity_audit(observation)
    if set(grads)!=set(params) or set(new_params)!=set(params):
        raise ValueError('gradient/parameter inventories differ across complete update')
    active_masks = after.get('model_state', after.get('state'))
    gates = policy['same_state_fixed_membership_complete_update']
    active = {}
    squares = 0.0
    for name, g in grads.items():
        match = EXPERT.match(name)
        active[name] = not match or bool(active_masks[f'active.{int(match[1])}'][int(match[2])] > 0)
        if active[name]:
            squares += float(g.double().square().sum())
    norm = math.sqrt(squares)
    coefficient = min(1., recipe.clip_norm/(norm+1e-6))
    measured_norm = float(new_opt[NORM])
    norm_error = abs(measured_norm-norm)/max(norm,1e-20)
    # Formula audit uses the observed generated coefficient; clipping has its
    # own independent FP64 gate and cannot be hidden in optimizer residuals.
    generated_coefficient = float(new_opt[CLIP])
    coefficient_error = abs(generated_coefficient-coefficient)/max(coefficient,1e-20)
    rate = recipe.lr(observation['update_index'])
    rows = []
    for name, p in params.items():
        m0, v0, t0 = (opt[k+':'+name] for k in ('exp_avg','exp_avg_sq','step'))
        m, v, t = (new_opt[k+':'+name] for k in ('exp_avg','exp_avg_sq','step'))
        if not active[name]:
            passed = not bool(grads[name].count_nonzero()) and all(bitwise_equal(a,b) for a,b in ((p,new_params[name]),(m0,m),(v0,v),(t0,t)))
            rows.append(dict(name=name, inactive=True, passed=passed))
            continue
        grad = grads[name].double()*generated_coefficient
        count = int(t0)+1
        expected_m = recipe.beta1*m0.double()+(1-recipe.beta1)*grad
        expected_v = recipe.beta2*v0.double()+(1-recipe.beta2)*grad.square()
        expected_p = p.double()*(1-rate*recipe.weight_decay) if p.ndim>=2 else p.double()
        expected_p = expected_p - rate/(1-recipe.beta1**count)*expected_m/(expected_v.sqrt()/math.sqrt(1-recipe.beta2**count)+recipe.eps)
        metrics = {key:distance(a,b) for key,a,b in (('parameter',new_params[name],expected_p),('exp_avg',m,expected_m),('exp_avg_sq',v,expected_v))}
        passed = int(t)==count and all(row['finite'] and (
            row['max_abs']<=gates['own_gradient_optimizer_absolute_max'] or
            row['relative_l2']<=gates['own_gradient_optimizer_relative_l2_max']) for row in metrics.values())
        rows.append(dict(name=name,inactive=False,passed=passed,metrics=metrics))
    ema=[]
    if recipe.name=='supervision_ema':
        completed=observation['update_index']+1;start=(recipe.horizon+1)//2
        alpha=1. if completed==start else 1.-math.exp(-16./recipe.horizon)
        for name in params:
            old=opt['ema:'+name];new=new_opt['ema:'+name]
            if completed<start:
                metric=dict(passed=bitwise_equal(old,new))
            elif completed==start:
                metric=dict(passed=bitwise_equal(new,new_params[name]))
            else:
                metric=distance(new,old.double()+alpha*(new_params[name].double()-old.double()))
                metric['passed']=metric['finite'] and metric['max_abs']<=gates['own_gradient_optimizer_absolute_max']
            ema.append(dict(name=name,**metric))
    return dict(norm_fp64=norm,observed_norm=measured_norm,norm_relative_error=norm_error,
                coefficient_fp64=coefficient,observed_coefficient=generated_coefficient,
                coefficient_relative_error=coefficient_error,
                parameters=rows,ema=ema,nonfinite_tensors=nonfinite,activity=activity,
                passed=not nonfinite and activity['passed'] and norm_error<=gates['clip_norm_relative_max'] and
                coefficient_error<=gates['clip_norm_relative_max'] and
                all(r['passed'] for r in rows+ema))


def routing_audit(observation, reference_probabilities):
    """Describe free selections at the same supplied upstream memberships.

    The independent FP64 path may produce slightly different probabilities.
    Report its cutoff margin and the measured per-token probability error;
    exact same-probability top-k ordering is qualified by the separate GPU
    primitive probe. Whole free-running trajectories are separate evidence.
    """
    state=observation['after']['model_state'];routes=observation['fixed_routes']
    rows=[]
    for layer,expected in enumerate(reference_probabilities):
        actual=state[f'probabilities.{layer}'].double()
        selected=routes[layer]
        expected=expected.detach().double()
        error=(actual-expected).abs().amax(-1)
        ref_top=expected.topk(selected.shape[-1]+1,dim=-1)
        k=selected.shape[-1]
        disagreement=(selected.sort(-1).values!=ref_top.indices[:,:k].sort(-1).values).any(-1)
        margin=ref_top.values[:,k-1]-ref_top.values[:,k]
        outside=disagreement & (margin>2*error)
        # A differing set beyond this interval requires investigation. This
        # comparison reports arithmetic sensitivity, not exact CPU/GPU routes.
        rows.append(dict(layer=layer,tokens=selected.shape[0],
            set_disagreements=int(disagreement.sum()),outside_error_interval=int(outside.sum()),
            maximum_probability_error=float(error.max()),
            minimum_cutoff_margin=float(margin.min()),
            maximum_disagreement_margin=float(margin[disagreement].max()) if disagreement.any() else 0.))
    return dict(basis='same upstream membership; observed BF16 vs independent installed FP64 probabilities',
                layers=rows,passed=all(row['outside_error_interval']==0 for row in rows))


def force_membership(model, routes):
    """Independent expert-loop reference; probabilities remain differentiable."""
    for index, layer in enumerate(model.model.layers):
        fixed=routes[index].cpu()
        def forward(module, hidden, selected=fixed):
            shape=hidden.shape;x=hidden.reshape(-1,shape[-1])
            logits=module.gate(x)
            probabilities=F.softmax(logits,dim=-1,dtype=torch.float64)
            weights=probabilities.gather(1,selected)
            weights=weights/weights.sum(-1,keepdim=True)
            out=torch.zeros_like(x)
            # Same installed parameter modules and ordinary autograd; no ragged
            # custom operators or compiler implementations enter this reference.
            for expert_index, expert in enumerate(module.experts):
                tokens,slots=torch.where(selected==expert_index)
                if tokens.numel():
                    contribution=expert(x[tokens])*weights[tokens,slots,None]
                    out=out.index_add(0,tokens,contribution)
            return out.reshape(shape),logits
        layer.block_sparse_moe.forward=types.MethodType(forward,layer.block_sparse_moe)
    for module in model.modules():
        if module.__class__.__name__=='MixtralRMSNorm':
            def norm(self,x):
                return self.weight*x*torch.rsqrt(x.square().mean(-1,keepdim=True)+self.variance_epsilon)
            module.forward=types.MethodType(norm,module)


def reference_audit(observation, model_dir, policy):
    model=load_model(model_dir,attention='eager').double()
    if set(dict(model.named_parameters())) != set(observation['before']['parameters']):
        raise ValueError('observation parameter inventory differs from installed model')
    model.load_state_dict(observation['before']['parameters'],strict=False)
    routes=observation['fixed_routes'].cpu()
    force_membership(model,routes)
    x,y=observation['tokens'].cpu(),observation['targets'].cpu()
    recipe=Recipe(**observation['recipe'])
    weight=recipe.supervision_weight(observation['update_index'])
    exit_states=[];hooks=[]
    if weight>0:
        for index in (3,7):
            hooks.append(model.model.layers[index].register_forward_hook(
                lambda _m,_a,value: exit_states.append(value)))
    try:
        output=model(input_ids=x,use_cache=False,output_router_logits=True)
    finally:
        for hook in hooks:hook.remove()
    final=F.cross_entropy(output.logits.reshape(-1,model.config.vocab_size),y.reshape(-1))
    extra=[F.cross_entropy(model.lm_head(model.model.norm(h)).reshape(-1,model.config.vocab_size),y.reshape(-1))
           for h in exit_states]
    reference_probabilities=[F.softmax(logits,dim=-1) for logits in output.router_logits]
    probability=torch.cat(reference_probabilities,dim=0)
    member=F.one_hot(routes.reshape(-1,routes.shape[-1]),model.num_experts).double().mean(0)
    auxiliary=model.num_experts*(member*probability.mean(0)[None,:]).sum()
    loss=(final+weight*sum(extra)/2)/(1+weight)+model.router_aux_loss_coef*auxiliary
    loss.backward()
    gates=policy['same_state_fixed_membership_complete_update'];rows=[]
    for name,p in model.named_parameters():
        expected=p.grad if p.grad is not None else torch.zeros_like(p)
        row=distance(observation['gradients'][name],expected,floor=gates['gradient_absolute_rms_floor'])
        row.update(name=name,passed=row['finite'] and row['relative_l2']<=gates['per_parameter_gradient_relative_l2_max'])
        rows.append(row)
    reference_loss=float(loss.detach())
    loss_error=abs(float(observation['loss'])-reference_loss)
    routing=routing_audit(observation,reference_probabilities)
    return dict(reference='installed Transformers FP64 eager attention and norms; independent expert loop at supplied membership',
                loss=reference_loss,observed_loss=float(observation['loss']),loss_absolute_error=loss_error,
                gradients=rows,routing=routing,
                passed=loss_error<=gates['objective_absolute_max'] and all(r['passed'] for r in rows) and routing['passed'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-dir');a=p.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError('independent references and receipt reductions belong on CPU workers')
    torch.set_num_threads(1)
    raw=POLICY_PATH.read_bytes();policy=json.loads(raw)
    path=Path(a.observation)
    sidecar=path.with_suffix('.json')
    if not sidecar.exists():sidecar=path.with_suffix(path.suffix+'.json')
    receipt=json.loads(sidecar.read_text())
    from megakernel_lab.tinymoe.data import digest
    if receipt['bytes']!=path.stat().st_size or receipt['sha256']!=digest(path):
        raise ValueError('observation size/hash mismatch')
    observation=canonical_observation(torch.load(a.observation,map_location='cpu',weights_only=True))
    contract=observation_contract(observation,receipt)
    report=dict(schema='latch-tinymoe-independent-update-audit-v1',
                policy_sha256=hashlib.sha256(raw).hexdigest(),observation_contract=contract,
                optimizer=optimizer_audit(observation,policy))
    if a.model_dir:
        report['fixed_membership']=reference_audit(observation,a.model_dir,policy)
    report['passed']=all(v['passed'] for k,v in report.items() if isinstance(v,dict) and 'passed' in v)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],policy_sha256=report['policy_sha256'])))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':main()
