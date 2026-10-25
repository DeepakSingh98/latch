"""Captured ordinary SUM, materialization, arithmetic metadata and refusals."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
import torch
from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import lower,LoweringError
from megakernel_lab.operation_compiler import torch_column_sum as rule


def captured(shape=(8,2048,768),dims=(0,1),*,product=False):
    if product:
        fn=lambda _p,x,y:dict(sum=(x*y).sum(dims))
        args=(torch.empty(shape),torch.empty(shape))
    else:
        fn=lambda _p,x:dict(sum=x.sum(dims))
        args=(torch.empty(shape),)
    result=capture_inference(fn,{},args)
    result.ir.metadata[rule.KEY]=rule.CONTRACT
    return result


@pytest.mark.parametrize("dims,expected", [((0,1),["partial","final"]),((0,),["single"])])
def test_captured_primary_phases_and_materialized_product(dims,expected):
    graph=captured(dims=dims,product=True).ir;before=deepcopy(graph.to_dict())
    program=lower(graph,execution="compose",vocabulary_head="off")
    selected=[p for p in program.manifest["phases"] if p.get("column_arithmetic")==rule.CONTRACT]
    assert [p[rule.KEY]["stage"] for p in selected]==expected
    detail=selected[0][rule.KEY]
    source=detail["source"]
    assert detail["materialized_input"] and source in program.manifest["layouts"]
    # The product has its own actual write preceding the consuming body; it
    # cannot be inlined into a reassociated or contracted column accumulation.
    producer=[i for i,p in enumerate(program.manifest["phases"]) if source in p["writes"]]
    consumer=program.manifest["phases"].index(selected[0])
    assert producer and max(producer)<consumer
    assert program.manifest["execution_selection"]["selected"]=="common_static"
    assert rule.required_sms(program.manifest)==132
    header=(Path(rule.__file__).parent/"kernels"/rule.HEADER).read_bytes()
    assert f'#include "{rule.HEADER}"' in program.source
    assert program.manifest["header_hashes"][rule.HEADER]==hashlib.sha256(header).hexdigest()
    assert b"__fadd_rn" in header
    assert b"for(int a=1;a<4;++a)" in header
    assert b"for(int cta=ty;cta<CTAS;cta+=BY)" in header
    if len(selected)==2:
        scratch=selected[0]["writes"][0]
        assert selected[1]["reads"]==[scratch]
        intervals=program.manifest["barrier_intervals"]
        indexes=[program.manifest["phases"].index(p) for p in selected]
        assert next(i for i,x in enumerate(intervals) if indexes[0] in x)<next(i for i,x in enumerate(intervals) if indexes[1] in x)
    assert graph.to_dict()==before


def test_geometry_matches_measured_configuration():
    rms=rule.geometry(16384,768);pos=rule.geometry(8,2048*768)
    assert rms["block"]==pos["block"]==[32,4,1]
    assert rms["grid"]==[6,256,1] and rms["partial_floats"]==196608
    assert pos["grid"]==[3072,1,1] and pos["partial_floats"]==0


def test_default_and_unselected_row_sum_unchanged():
    graph=captured(shape=(128,768),dims=(0,)).ir
    del graph.metadata[rule.KEY]
    default=lower(graph,execution="compose",vocabulary_head="off")
    graph.metadata[rule.KEY]=rule.CONTRACT
    lower(graph,execution="compose",vocabulary_head="off")
    del graph.metadata[rule.KEY]
    assert lower(graph,execution="compose",vocabulary_head="off").source==default.source
    row=captured(shape=(128,768),dims=(1,)).ir
    selected=lower(row,execution="compose",vocabulary_head="off")
    del row.metadata[rule.KEY]
    assert selected.source==lower(row,execution="compose",vocabulary_head="off").source
    assert not any(p.get("column_arithmetic") for p in selected.manifest["phases"])


@pytest.mark.parametrize("case", ["width","strides","unknown","legacy"])
def test_explicit_refusals(case):
    graph=captured(shape=(128,770 if case=="width" else 768),dims=(0,)).ir
    if case=="strides":
        value=graph.values[graph.inputs[0]]
        from dataclasses import replace
        graph.values[graph.inputs[0]]=replace(value,strides=(1,128))
    if case=="unknown":graph.metadata[rule.KEY]="unknown"
    with pytest.raises(LoweringError,match="column-sum"):
        lower(graph,execution="legacy" if case=="legacy" else "compose",vocabulary_head="off")


def test_runtime_manifest_binding_fail_closed():
    assert rule.required_sms({"phases":[]}) is None
    assert rule.required_sms({"phases":[{"column_arithmetic":rule.CONTRACT,
        rule.KEY:{"required_multiprocessors":132}}]})==132
    with pytest.raises(ValueError,match="132-SM"):
        rule.required_sms({"phases":[{"column_arithmetic":rule.CONTRACT}]})
