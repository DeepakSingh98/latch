"""Observe CUDA BF16 shared-readout accumulation against isolated contributions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def error(actual, expected):
    difference=actual.double()-expected.double()
    return dict(max_abs=float(difference.abs().max()),
                relative_l2=float(difference.norm()/expected.double().norm().clamp_min(1e-30)),
                equal=bool(torch.equal(actual.double(),expected.double())))


def check(*,vocabulary=32000,width=384,tokens=17):
    torch.manual_seed(712)
    weight=torch.nn.Parameter(torch.randn(vocabulary,width,device="cuda")/width**.5)
    hidden=[torch.randn(tokens,width,device="cuda") for _ in range(3)]
    labels=torch.randint(vocabulary,(tokens,),device="cuda")
    coefficients=(.8,.1,.1)

    def term(index,explicit=False):
        h,w=hidden[index],weight
        if explicit: h,w=h.bfloat16(),w.bfloat16()
        return coefficients[index]*F.cross_entropy(F.linear(h,w).float(),labels)

    # Each isolated backward is an actual CUDA BF16 contribution. FP64 summing
    # these observed gradients independently identifies aggregation error,
    # without conflating it with BF16 GEMM error or softmax derivatives.
    contributions=[]
    for index in range(3):
        weight.grad=None
        with torch.autocast("cuda",dtype=torch.bfloat16): term(index).backward()
        contributions.append(weight.grad.detach().clone())
    reference=sum(g.double() for g in contributions)
    actual={}
    for mode in ("installed_single_backward","bridge_separate_backward","generated_separate_casts"):
        weight.grad=None
        if mode=="generated_separate_casts":
            sum(term(i,explicit=True) for i in range(3)).backward()
        else:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                if mode=="bridge_separate_backward":
                    for i in range(3): term(i).backward()
                else: sum(term(i) for i in range(3)).backward()
        actual[mode]=weight.grad.detach().clone()
    rows={mode:error(value,reference) for mode,value in actual.items()}
    # Omit-one references document that an apparently small exit contribution
    # is nevertheless measurable and present in the complete shared gradient.
    missing=[error(actual["generated_separate_casts"],reference-g.double()) for g in contributions]
    agreement=error(actual["bridge_separate_backward"],actual["generated_separate_casts"])
    passed=(agreement["relative_l2"]<=1e-6
            and rows["generated_separate_casts"]["relative_l2"]<=1e-6
            and all(row["relative_l2"]>1e-3 for row in missing))
    return dict(shape=dict(vocabulary=vocabulary,width=width,tokens=tokens),coefficients=coefficients,
                contribution_norms=[float(g.double().norm()) for g in contributions],
                errors_against_fp64_sum_of_isolated_bf16_contributions=rows,
                bridge_vs_generated=agreement,omit_one_errors=missing,passed=passed,
                gate="bridge/generated and generated/independent sum relative L2 <= 1e-6; each omitted contribution changes relative L2 > 1e-3")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True)
    a=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    result=dict(schema="latch-tinymoe-shared-readout-contributions-v1",torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(),**check())
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
