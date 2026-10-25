"""CUDA-order source oracle plus ordinary capture/lowering; run on CPU worker."""
from copy import deepcopy
import pytest
import torch
from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.torch_topk32 import topk32


def source_order(row,k):
    # Independent scalar spelling of CUDA sbtopk's threshold partition and
    # SortUtils bitonicSwap. CPU torch.topk is not the CUDA tie-order oracle.
    threshold=sorted(row,reverse=True)[k-1]
    indices=[i for i,v in enumerate(row) if v>threshold]
    indices += [i for i,v in enumerate(row) if v==threshold][:k-len(indices)]
    keys=[row[i] for i in indices]+[0.]*(32-k)
    indices += [0]*(32-k);valid=[i<k for i in range(32)]
    def step(size,stride,final=False):
        for t in range(16):
            a=2*t-(t&(stride-1));b=a+stride;direction=False if final else bool(t&(size//2))
            swap=(keys[a]>keys[b] and valid[a]) or not valid[b]
            if swap==direction:
                keys[a],keys[b]=keys[b],keys[a];indices[a],indices[b]=indices[b],indices[a]
                valid[a],valid[b]=valid[b],valid[a]
    for size in (2,4,8,16):
        stride=size//2
        while stride:step(size,stride);stride//=2
    for stride in (16,8,4,2,1):step(32,stride,True)
    assert all(valid[:k])
    return indices[:k]


@pytest.mark.parametrize("k",[1,2,8,32])
def test_finite_source_order_and_ordinary_ad(k):
    generator=torch.Generator().manual_seed(31)
    rows=torch.randn(8,32,generator=generator);rows[1].fill_(.25);rows[2:]=(rows[2:]*3).round().abs()
    rows.requires_grad_(True);values,indices=topk32(rows,k)
    expected=torch.tensor([source_order(row,k) for row in rows.detach().tolist()])
    assert torch.equal(indices,expected) and torch.equal(values,rows.gather(-1,expected))
    dy=torch.randn(values.shape,generator=generator)
    gradient,=torch.autograd.grad((values*dy).sum(),rows)
    assert torch.equal(gradient,torch.zeros_like(rows).scatter(-1,expected,dy))


@pytest.mark.parametrize("case",["dtype","width","zero","too_large","bool"])
def test_refusals(case):
    scores=torch.zeros(2,31 if case=="width" else 32,dtype=torch.float64 if case=="dtype" else torch.float32)
    k={"zero":0,"too_large":33,"bool":True}.get(case,8)
    with pytest.raises(ValueError):topk32(scores,k)


@pytest.mark.parametrize("backward",[False,True])
def test_captured_composition_lowers_and_keeps_dynamic_indices(backward):
    from .fixtures.topk_components import component_forward,component_ad
    scores=torch.randn(2,3,32,generator=torch.Generator().manual_seed(14));cotangent=torch.ones(2,3,8)
    fn=component_ad if backward else component_forward;args=(scores,cotangent) if backward else (scores,)
    captured=capture_inference(fn,{},args);before=deepcopy(captured.ir.to_dict())
    actual=captured.unpack(captured.graph_step(*captured.example_args))["predictions"]
    expected=fn({},*args)
    assert all(torch.equal(actual[key],value) for key,value in expected.items())
    targets=[op.target for op in captured.ir.operations]
    assert any("argmax" in target for target in targets)
    assert any("gather" in target for target in targets)
    assert not any("topk" in target or "torch_topk32" in target for target in targets)
    if backward:assert any("scatter_add" in target for target in targets)
    program=lower(captured.ir,execution="compose",vocabulary_head="off")
    assert program.manifest["phases"] and captured.ir.to_dict()==before
