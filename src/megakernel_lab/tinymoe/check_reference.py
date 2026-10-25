"""Independent execution checks for the pinned conventional objective; CPU worker."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .model import Recipe, features, installed_objective, load_model, optimized_backward, parameter_inventory
from .pins import public_identity


def compare_objective(model, x, y, weight, chunk_tokens=7):
    model.zero_grad(set_to_none=True)
    installed, im = installed_objective(model, x, y, weight)
    installed.backward()
    expected = {n: None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
    installed_value = float(installed.detach())
    model.zero_grad(set_to_none=True)
    candidate, cm = optimized_backward(model, x, y, weight, chunk_tokens=chunk_tokens)
    rows = []
    for name, p in model.named_parameters():
        e = expected.pop(name)
        if e is None or p.grad is None:
            rows.append(dict(name=name, passed=e is None and p.grad is None, absent=e is None))
            continue
        diff = (p.grad-e).double()
        absolute = float(diff.abs().max())
        relative = float(diff.norm()/e.double().norm().clamp_min(1e-20))
        passed = bool(torch.allclose(p.grad, e, rtol=1e-4, atol=1e-6))
        rows.append(dict(name=name, passed=passed, max_abs=absolute, relative_l2=relative))
    return dict(weight=weight, chunk_tokens=chunk_tokens, installed_objective=installed_value, bridge_objective=float(candidate),
        loss_abs=abs(installed_value-float(candidate)), gradients=rows,
        passed=abs(installed_value-float(candidate))<2e-5 and all(r["passed"] for r in rows),
        gates=dict(loss_abs=2e-5, gradient_allclose_rtol=1e-4, gradient_allclose_atol=1e-6))


def readout_association(model,x,y):
    """Attribute FP32 chunk-association error against an isolated FP64 head VJP.

    This is diagnostic, with no replacement acceptance gate. The full-model
    chunk7 failure remains a failure of its original strict coordinate gate.
    """
    with torch.no_grad(): hidden=features(model,x)[0].detach().reshape(-1,model.config.hidden_size)
    target=y.reshape(-1)
    grads={}
    for name,chunk,dtype in (("whole_fp32",hidden.shape[0],torch.float32),
                              ("chunk7_fp32",7,torch.float32),("whole_fp64",hidden.shape[0],torch.float64)):
        value=hidden.to(dtype).detach().requires_grad_(True)
        weight=model.lm_head.weight.detach().to(dtype)
        for start in range(0,len(target),chunk):
            logits=F.linear(value[start:start+chunk],weight)
            loss=F.cross_entropy(logits,target[start:start+chunk],reduction="sum")/len(target)
            loss.backward()
        grads[name]=value.grad.double()
    oracle=grads["whole_fp64"]
    return {name:dict(max_abs=float((grad-oracle).abs().max()),
                     relative_l2=float((grad-oracle).norm()/oracle.norm()))
            for name,grad in grads.items() if name!="whole_fp64"}


def inspect_loaded(model):
    with torch.no_grad():
        x = torch.tensor([[1, 42, 73, 92, 2, 11, 9, 6]], dtype=torch.long)
        installed = model(x, use_cache=False, output_router_logits=True)
        final, _, routers = features(model,x)
        direct_logits=model.lm_head(final)
        router_dtype=[str(t.dtype) for t in routers]
        result=dict(parameter_count=sum(p.numel() for p in model.parameters()),
            head_dims=[layer.self_attn.head_dim for layer in model.model.layers],
            logits_max_abs=float((installed.logits-direct_logits).abs().max()),
            router_logits_dtypes=router_dtype, router_loss_dtype=str(installed.aux_loss.dtype),
            aux_loss=float(installed.aux_loss), config=model.config.to_dict())
    return result


def tiny_model():
    from transformers import MixtralConfig, MixtralForCausalLM
    c=MixtralConfig(vocab_size=96, hidden_size=32, intermediate_size=48,
        num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2,
        num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        sliding_window=16, output_router_logits=True, use_cache=False,
        attention_dropout=0., router_jitter_noise=0., router_aux_loss_coef=.02,
        rope_theta=1000000., tie_word_embeddings=False)
    c._attn_implementation="eager"
    return MixtralForCausalLM(c)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir");p.add_argument("--out",required=True)
    p.add_argument("--production-gradients",action="store_true")
    p.add_argument("--diagnose-association",action="store_true")
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1729)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    checks=[]
    tiny=tiny_model()
    x=torch.randint(0,96,(2,9));y=torch.randint(0,96,(2,9))
    for weight in (0.,.25):
        checks.append(compare_objective(tiny,x,y,weight))
    report=dict(schema="latch-tinymoe-reference-audit-v1",identity=public_identity(),tiny=checks)
    if a.model_dir:
        model=load_model(a.model_dir,attention="eager")
        report["loaded"]=inspect_loaded(model)
        (out/"parameters.json").write_text(json.dumps(parameter_inventory(model),indent=2)+"\n")
        if a.production_gradients:
            x=torch.tensor([[1,42,73,92,2,11,9,6]],dtype=torch.long)
            y=torch.tensor([[42,73,92,2,11,9,6,21]],dtype=torch.long)
            report["production"]=[compare_objective(model,x,y,w) for w in (0.,.25)]
        if a.diagnose_association:
            x=torch.tensor([[1,42,73,92,2,11,9,6]],dtype=torch.long)
            y=torch.tensor([[42,73,92,2,11,9,6,21]],dtype=torch.long)
            report["matched_readout_geometry"]=compare_objective(model,x,y,0.,chunk_tokens=x.numel())
            report["isolated_readout_fp64"]=readout_association(model,x,y)
    report["passed"]=all(r["passed"] for r in checks+report.get("production",[])) and report.get("loaded",{}).get("logits_max_abs",0)<1e-5
    (out/"result.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("tiny","production","loaded")},indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__=="__main__":
    main()
