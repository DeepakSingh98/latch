"""CPU-only metadata/lowering checks; no CUDA compilation or numerical claim."""
# Accepted c24: two independent rows per CTA; affine serial chains and
# their unchanged combine tree are separate phases. Numerical gates live
# in the component/device qualifications and are not changed here.
from copy import deepcopy

import pytest

from megakernel_lab.operation_compiler.lower import LoweringError, lower
from .test_torch_layer_norm_backward import fixture as metadata_fixture
from megakernel_lab.operation_compiler.torch_layer_norm_backward import ROW_ARITHMETIC


def fixture():
    graph=metadata_fixture()
    graph["outputs"]=["output","gain_product"]
    for name,value in graph["values"].items():value["name"]=name
    return graph


def phases(program):
    return [p for p in program.manifest["phases"] if p.get("row_arithmetic")==ROW_ARITHMETIC]


def test_backward_single_publication_preserves_gain_computation():
    graph=fixture();before=deepcopy(graph)
    program=lower(graph,execution="compose")
    phase,=phases(program)
    assert phase["writes"]==phase["names"]==["output"]
    assert phase["reads"]==["x","dy","gain","mean","rstd"]
    assert phase["layer_norm_backward_plan"]["preserved"]==("center","norm")
    assert (phase["threads"],phase["shared_bytes"],phase["work_items"])==(256,64,2)
    assert "torch_ln_dx::row2(" in program.source
    assert "torch_layer_norm_backward.cuh" in program.manifest["header_hashes"]
    assert any("gain_product" in p["writes"] for p in program.manifest["phases"])
    assert program.manifest["execution_selection"]["selected"]=="common_static"
    assert graph==before


def test_default_equation_is_unchanged_after_selected_lowering():
    graph=fixture();graph["metadata"]={}
    first=lower(graph,execution="compose")
    assert not phases(first)
    assert "torch_layer_norm_backward.cuh" not in first.manifest["header_hashes"]
    assert phases(lower(fixture(),execution="compose"))
    assert lower(graph,execution="compose").source==first.source


def test_shared_intermediate_remains_available():
    graph=fixture();graph["outputs"].append("mixed")
    program=lower(graph,execution="compose",row_fusion=True)
    phase,=phases(program)
    assert "mixed" in phase["layer_norm_backward_plan"]["preserved"]
    assert any("mixed" in p["writes"] for p in program.manifest["phases"])
    assert "row_reduction" not in phase


def test_primary_dimensions_and_renamed_operations():
    graph=fixture()
    for value in graph["values"].values():
        old=value["shape"]
        value["shape"]=shape=[16384,768] if old==[3,8] else [16384,1] if old==[3,1] else [768]
        value["strides"]=[shape[1],1] if len(shape)==2 else [1]
    for op in graph["operations"]:
        if op["name"] in ("scale","scaled"):op["args"][1]=768
        op["output"]=op["outputs"][0];op["name"]="op_"+op["name"]
    phase,=phases(lower(graph,execution="compose"))
    assert phase["layer_norm_backward_plan"]["anchor"]=="op_weighted"
    assert (phase["work_items"],phase["reduction_elements"])==(8192,768)


def test_legacy_lowering_refuses_explicit_backward_contract():
    with pytest.raises(LoweringError,match="LayerNorm arithmetic requires"):
        lower(fixture(),execution="legacy")
