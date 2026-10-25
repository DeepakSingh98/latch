"""CPU lowering checks: no CUDA compilation or numerical claims."""
from copy import deepcopy
import pytest
from megakernel_lab.operation_compiler.lower import lower,LoweringError
from megakernel_lab.operation_compiler.torch_logsoftmax import KEY,CONTRACT,HEADER


def fixture(width=50257):
    values={n:dict(name=n,shape=[2,width],strides=[width,1],dtype="float32",
        storage_offset=0,alias_of=None,output_roles=[]) for n in ("x","g","y","dx")}
    def op(n,target,args,inputs):
        return dict(name=n,target=target,args=args,kwargs={},inputs=inputs,
            outputs=[n],reads=inputs,writes=[n],effect="pure")
    return dict(values=values,inputs=["x","g"],outputs=["y","dx"],metadata={KEY:CONTRACT},
        operations=[op("y","aten._log_softmax.default",[{"ref":"x"},-1,False],["x"]),
            op("dx","aten._log_softmax_backward_data.default",[{"ref":"g"},{"ref":"y"},-1,"float32"],["g","y"])])


def test_measured_contract_uses_both_bodies_and_preserves_graph():
    g=fixture();before=deepcopy(g);p=lower(g,execution="compose",vocabulary_head="off")
    phases=[x for x in p.manifest["phases"] if x.get("row_arithmetic","").startswith(CONTRACT)]
    assert len(phases)==2
    assert [x["shared_bytes"] for x in phases]==[128,2048]
    assert [x["log_softmax_contract"]["logical_threads"] for x in phases]==[1024,512]
    assert all(x["threads"]==256 for x in phases)
    assert HEADER in p.manifest["header_hashes"]
    assert "backward_row" in p.source and ",row,true,smem)" in p.source
    assert p.manifest["execution_selection"]["selected"]=="common_static"
    assert g==before


def test_default_source_unchanged_after_selected_lowering():
    g=fixture();g["metadata"]={};first=lower(g,execution="compose",vocabulary_head="off")
    assert HEADER not in first.manifest["header_hashes"]
    lower(fixture(),execution="compose",vocabulary_head="off")
    assert lower(g,execution="compose",vocabulary_head="off").source==first.source


def test_propagated_metadata_without_operation_keeps_previous_schedule():
    g=fixture();g["outputs"]=["y"]
    g["operations"]=[dict(name="y",target="aten.add.Tensor",args=[{"ref":"x"},{"ref":"g"}],
        kwargs={},inputs=["x","g"],outputs=["y"],reads=["x","g"],writes=["y"],effect="pure")]
    selected=lower(g,execution="compose",vocabulary_head="off")
    g["metadata"]={};original=lower(g,execution="compose",vocabulary_head="off")
    assert selected.source==original.source
    assert selected.manifest["execution_selection"]["selected"]==original.manifest["execution_selection"]["selected"]
    assert HEADER not in selected.manifest["header_hashes"]


@pytest.mark.parametrize("width",[768,8193,9216,50256])
def test_other_pinned_dispatch_paths_refused(width):
    with pytest.raises(LoweringError,match="regular ILP4"):
        lower(fixture(width),execution="compose",vocabulary_head="off")


def test_physical_padding_fp32_refused():
    g=fixture();g["values"]["x"]["strides"]=[50304,1]
    with pytest.raises(LoweringError,match="contiguous FP32"):
        lower(g,execution="compose",vocabulary_head="off")


def test_unknown_contract_and_legacy_execution_refused():
    with pytest.raises(LoweringError,match="requires execution"):
        lower(fixture(),execution="legacy",vocabulary_head="off")
    g=fixture();g["metadata"][KEY]="unmeasured"
    with pytest.raises(LoweringError,match="unknown explicit"):
        lower(g,execution="compose",vocabulary_head="off")
