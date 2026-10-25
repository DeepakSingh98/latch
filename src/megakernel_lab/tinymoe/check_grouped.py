"""GPU same-routing checks for conventional grouped experts, including empties."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from .model import GroupedExpertsMoe, select_experts


def metric(actual, expected, *, rel=0.025, absolute=0.002):
    a,e=actual.detach().float(),expected.detach().float()
    difference=(a-e).double()
    relative=float(difference.norm()/e.double().norm().clamp_min(1e-12))
    max_abs=float(difference.abs().max())
    return dict(relative_l2=relative,max_abs=max_abs,passed=relative<=rel or max_abs<=absolute)


def case(original, inputs, *, compiled=False):
    # Frozen independent copies of precisely the same live FP32 parameters.
    stock=copy.deepcopy(original).cuda()
    grouped=GroupedExpertsMoe(copy.deepcopy(original)).cuda()
    if compiled:
        grouped.forward=torch.compile(grouped.forward,fullgraph=True)
    x=inputs.cuda().requires_grad_(True)
    y=inputs.cuda().requires_grad_(True)
    upstream=torch.randn_like(x)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        left,lr=stock(x)
        right,rr=grouped(y)
        membership_equal=torch.equal(select_experts(lr,stock.top_k,x.dtype)[1],
                                    select_experts(rr,stock.top_k,y.dtype)[1])
        (left*upstream).sum().backward()
        (right*upstream).sum().backward()
    active=grouped.active_experts.cpu().tolist()
    stock_parameters=dict(stock.named_parameters())
    rows=[]
    for name,p in grouped.named_parameters():
        expected=stock_parameters[name].grad
        if name.startswith("experts.") and not active[int(name.split(".")[1])]:
            # Exactly zero mathematical gradient; restore absent optimizer leaf.
            zero=bool(torch.equal(p.grad,torch.zeros_like(p.grad)))
            rows.append(dict(name=name,absent=True,passed=expected is None and zero))
            p.grad=None
        else:
            rows.append(dict(name=name,**metric(p.grad,expected)))
    output=metric(right,left);input_gradient=metric(y.grad,x.grad)
    return dict(compiled=compiled,active_experts=active,membership_equal=membership_equal,
        output=output,input_gradient=input_gradient,gradients=rows,
        passed=membership_equal and output["passed"] and input_gradient["passed"] and all(r["passed"] for r in rows))


def main():
    from .conventional import configure_execution
    execution_policy=configure_execution()
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True);p.add_argument("--compiled",action="store_true")
    a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32=False
    c=MixtralConfig(hidden_size=384,intermediate_size=768,num_local_experts=16,num_experts_per_tok=2)
    module=MixtralSparseMoeBlock(c)
    x=torch.randn(2,37,384)
    rows=[dict(kind="uneven",**case(module,x,compiled=a.compiled))]
    # Deliberately force only two experts, then change their identities. This
    # exercises empty groups at the beginning, middle and end of offsets.
    for chosen in ((0,1),(7,15)):
        with torch.no_grad():
            module.gate.weight.zero_();module.gate.weight[:,0]=-10
            module.gate.weight[chosen[0],0]=10;module.gate.weight[chosen[1],0]=9
            x[:,:,0]=1
        rows.append(dict(kind="unused-and-changing",chosen=chosen,**case(module,x,compiled=a.compiled)))
    result=dict(schema="latch-tinymoe-grouped-conventional-audit-v1",passed=all(r["passed"] for r in rows),
        gates=dict(relative_l2=.025,max_abs=.002,rule="relative_l2 OR max_abs per tensor; exact same supplied router matmul inputs/outputs"),
        torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),cases=rows)
    result["execution_policy"]=execution_policy
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(passed=result["passed"],cases=[dict(kind=r["kind"],passed=r["passed"]) for r in rows])))
    if not result["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
