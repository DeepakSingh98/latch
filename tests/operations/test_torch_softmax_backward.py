"""Metadata/source checks for the measured opt-in warp32 VJP contract."""
from copy import deepcopy
import pytest
from megakernel_lab.operation_compiler.lower import lower, LoweringError
from megakernel_lab.operation_compiler.torch_softmax_backward import KEY, CONTRACT, ROW_ARITHMETIC


def fixture(width=32, *, strided=False):
    shape = [2, 128, width]
    values = {n: dict(name=n, shape=shape, strides=[128*width,width,1], dtype="float32",
        storage_offset=0, alias_of=None, output_roles=[]) for n in ("p", "g", "dx")}
    if strided:
        values["g"]["strides"] = [128, 1, 256]
    return dict(values=values, inputs=["g", "p"], outputs=["dx"], metadata={KEY:CONTRACT},
        operations=[dict(name="dx", target="aten._softmax_backward_data.default",
            args=[{"ref":"g"}, {"ref":"p"}, -1, "float32"], kwargs={}, inputs=["g","p"],
            outputs=["dx"], reads=["g","p"], writes=["dx"], effect="pure")])


@pytest.mark.parametrize("strided", [False, True])
def test_measured_body_and_actual_contract_retained(strided):
    graph = fixture(strided=strided); before = deepcopy(graph)
    program = lower(graph, execution="compose", vocabulary_head="off")
    selected = [p for p in program.manifest["phases"] if p.get("row_arithmetic") == ROW_ARITHMETIC]
    assert len(selected) == 1
    assert selected[0]["softmax_backward_contract"]["input_gradient_strides"] == graph["values"]["g"]["strides"]
    assert "__fmul_rn(" in program.source and "fmaf(-probability,total,product)" in program.source
    assert "offset=16;offset>0;offset>>=1" in program.source
    assert program.manifest["execution_selection"]["selected"] == "common_static"
    assert graph == before


def test_default_source_unchanged_and_forward_not_selected():
    graph = fixture(); graph["metadata"] = {}
    old = lower(graph, execution="compose", vocabulary_head="off")
    lower(fixture(), execution="compose", vocabulary_head="off")
    assert lower(graph, execution="compose", vocabulary_head="off").source == old.source
    graph = fixture(); op = graph["operations"][0]
    op.update(target="aten._softmax.default", args=[{"ref":"p"},-1,False], inputs=["p"], reads=["p"])
    selected = lower(graph, execution="compose", vocabulary_head="off")
    graph["metadata"] = {}
    default = lower(graph, execution="compose", vocabulary_head="off")
    assert selected.source == default.source
    assert not any(p.get("row_arithmetic") == ROW_ARITHMETIC for p in selected.manifest["phases"])


@pytest.mark.parametrize("width", [16, 64, 768])
def test_other_widths_refused(width):
    with pytest.raises(LoweringError, match="last-axis width32"):
        lower(fixture(width), execution="compose", vocabulary_head="off")


@pytest.mark.parametrize("change", ["axis", "dtype", "shape", "unknown"])
def test_unmeasured_contracts_refused(change):
    graph = fixture()
    if change == "axis": graph["operations"][0]["args"][2] = 0
    elif change == "dtype": graph["operations"][0]["args"][3] = "bfloat16"
    elif change == "shape": graph["values"]["g"]["shape"] = [2,127,32]
    else: graph["metadata"][KEY] = "unknown"
    with pytest.raises(LoweringError, match="explicit softmax backward"):
        lower(graph, execution="compose", vocabulary_head="off")


def test_legacy_execution_refused():
    with pytest.raises(LoweringError, match="requires execution"):
        lower(fixture(), execution="legacy", vocabulary_head="off")
