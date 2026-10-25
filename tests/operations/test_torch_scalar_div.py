"""Literal provenance, actual emission, and isolation of the scalar contract."""
from copy import deepcopy
from fractions import Fraction
import struct

import pytest
import torch

from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler import torch_scalar_div as rule


def fixture(denominator=768, target="aten.div.Scalar", dtype="float32"):
    values = {n:dict(name=n, shape=[4,8], strides=[8,1], dtype=dtype,
                    storage_offset=0, alias_of=None, output_roles=[]) for n in ("x","y","z")}
    return dict(values=values, inputs=["x"], outputs=["z"], metadata={rule.KEY:rule.CONTRACT},
        operations=[dict(name="y", target=target, args=[{"ref":"x"},denominator], kwargs={},
                         inputs=["x"], outputs=["y"], reads=["x"], writes=["y"], effect="pure"),
                    dict(name="z", target="aten.add.Tensor", args=[{"ref":"y"},{"ref":"x"}], kwargs={},
                         inputs=["y","x"], outputs=["z"], reads=["y","x"], writes=["z"], effect="pure")])


@pytest.mark.parametrize("target", ["aten.div.Scalar","aten.div.Tensor"])
@pytest.mark.parametrize("denominator", [768,3.0,-7.25,16777217])
def test_literal_division_records_actual_emitter_and_exact_reciprocal(target, denominator):
    graph=fixture(denominator,target); before=deepcopy(graph)
    program=lower(graph,execution="compose",vocabulary_head="off")
    records=program.manifest["scalar_divisions"]
    assert len(records)==1
    record=records[0]
    assert record["denominator"]==denominator and record["target"]==target
    assert record["emitter"]=="common_pointwise_fmul_rn"
    assert record["phase_indices"]
    assert program.manifest["execution_selection"]["selected"]=="common_static"
    assert "/*latch_scalar_div:y*/ __fmul_rn(" in program.source
    assert f"__uint_as_float({record['reciprocal_bits']}u)" in program.source
    divisor=struct.unpack("<f",struct.pack("<f",float(denominator)))[0]
    reciprocal=struct.unpack("<f",struct.pack("<f",1./divisor))[0]
    assert record["denominator_fp32"]==divisor
    assert record["reciprocal_fp32"]==reciprocal
    assert graph==before


def test_default_unchanged_and_metadata_without_selection_does_not_force_common():
    graph=fixture(); graph["metadata"]={}
    default=lower(graph,execution="compose",vocabulary_head="off")
    assert "scalar_divisions" not in default.manifest
    lower(fixture(),execution="compose",vocabulary_head="off")
    assert lower(graph,execution="compose",vocabulary_head="off").source==default.source
    graph["operations"][0]["target"]="aten.mul.Tensor"
    previous=lower(graph,execution="compose",vocabulary_head="off")
    graph["metadata"]={rule.KEY:rule.CONTRACT}
    selected=lower(graph,execution="compose",vocabulary_head="off")
    assert selected.source==previous.source
    assert selected.manifest["execution_selection"]["selected"]==previous.manifest["execution_selection"]["selected"]
    assert "scalar_divisions" not in selected.manifest


@pytest.mark.parametrize("scalar_shape", [False,True])
def test_tensor_denominator_is_not_reinterpreted_as_cpu_literal(scalar_shape):
    graph=fixture({"ref":"denominator"},"aten.div.Tensor")
    graph["values"]["denominator"]=dict(name="denominator", shape=[] if scalar_shape else [4,8],
        strides=[] if scalar_shape else [8,1], dtype="float32",storage_offset=0,alias_of=None,output_roles=[])
    graph["inputs"].append("denominator")
    graph["operations"][0]["inputs"].append("denominator")
    graph["operations"][0]["reads"].append("denominator")
    program=lower(graph,execution="compose",vocabulary_head="off")
    graph["metadata"]={}
    original=lower(graph,execution="compose",vocabulary_head="off")
    assert "scalar_divisions" not in program.manifest
    assert program.source==original.source


def test_bf16_division_is_outside_explicit_fp32_rule():
    graph=fixture(dtype="bfloat16")
    program=lower(graph,execution="compose",vocabulary_head="off")
    graph["metadata"]={}
    assert "scalar_divisions" not in program.manifest
    assert program.source==lower(graph,execution="compose",vocabulary_head="off").source


@pytest.mark.parametrize("denominator", [0.,float("inf"),float("nan"),1.e-40,1.e40])
def test_unadmitted_constant_domain_refused(denominator):
    with pytest.raises(LoweringError,match="finite normal FP32"):
        lower(fixture(denominator),execution="compose",vocabulary_head="off")


def test_unknown_contract_and_legacy_emitter_refused():
    graph=fixture()
    with pytest.raises(LoweringError,match="requires execution='compose'"):
        lower(graph,execution="legacy",vocabulary_head="off")
    graph["metadata"][rule.KEY]="unmeasured"
    with pytest.raises(LoweringError,match="unknown explicit"):
        lower(graph,execution="compose",vocabulary_head="off")


def test_capture_distinguishes_wrapped_literal_from_real_scalar_tensor():
    torch.set_num_threads(1)
    captured=capture_inference(lambda p,x,d:dict(literal=x/768,tensor=x/d),{},
                               (torch.ones(4,8),torch.tensor(768.)))
    captured.ir.metadata[rule.KEY]=rule.CONTRACT
    divisions=[op for op in captured.ir.operations if op.target.startswith("aten.div.")]
    assert len(divisions)==2
    assert type(divisions[0].args[1]) is int
    assert type(divisions[1].args[1]) is not int
    program=lower(captured.ir,execution="compose",vocabulary_head="off")
    records=program.manifest["scalar_divisions"]
    assert len(records)==1 and records[0]["denominator"]==768


def test_inlined_division_is_counted_only_when_its_marker_reaches_source():
    graph=fixture(); program=lower(graph,execution="compose",vocabulary_head="off",elide_stores=True)
    records=program.manifest["scalar_divisions"]
    assert len(records)==1
    assert records[0]["phase_indices"]
    assert "/*latch_scalar_div:y*/" in program.source


def test_exact_rounding_helper_resolves_fp32_ties_and_integer_conversion():
    assert rule._normal_f32(Fraction(1)+Fraction(1,2**24))[1]==0x3f800000
    assert rule._normal_f32(Fraction(1)+Fraction(3,2**24))[1]==0x3f800002
    assert rule._normal_f32(2**60+2**36+1)[1]==0x5d800001
