"""Tensor lerp broadcasting/capture admission; numerical CUDA proof is separate."""
import pytest
import torch

from megakernel_lab.operation_compiler import capture_module, SGDConfig
from megakernel_lab.operation_compiler.lower import lower, LoweringError
from .test_lower_safety import _value, _op, _ref
from .test_lerp import run as scalar_regression


def graph(weight_shape=(), weight_dtype="float32", keyword=False):
    args = [_ref("start"), _ref("end")]
    kwargs = {"weight": _ref("weight")} if keyword else {}
    if not keyword:
        args.append(_ref("weight"))
    return dict(inputs=["start", "end", "weight"], outputs=["out"],
        values=dict(start=_value((2, 3)), end=_value((1, 3)),
                    weight=_value(weight_shape, weight_dtype), out=_value((2, 3))),
        operations=[_op("out", "lerp.Tensor", args, kwargs)])


@pytest.mark.parametrize("shape", [(), (1,), (2, 1), (2, 3)])
@pytest.mark.parametrize("keyword", [False, True])
def test_tensor_lerp_broadcast_admission(shape, keyword):
    for options in ({}, {"execution": "compose"}, {"policy": "reference"}):
        program = lower(graph(shape, keyword=keyword), **options)
        assert "fabsf(" in program.source and "< 0.5f ? fmaf(" in program.source
        assert "1.0f - " in program.source
        assert program.manifest["layouts"]["out"]["shape"] == [2, 3]


@pytest.mark.parametrize("dtype", ["bfloat16", "float64", "int64"])
def test_tensor_lerp_rejects_implicit_weight_dtype(dtype):
    with pytest.raises(LoweringError, match="three FP32 tensor operands"):
        lower(graph(weight_dtype=dtype))


def test_tensor_lerp_survives_common_capture():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.start = torch.nn.Parameter(torch.zeros(2, 3))
            self.end = torch.nn.Parameter(torch.ones(1, 3))

        def forward(self, weight):
            return torch.lerp(self.start, self.end, weight)

    captured = capture_module(Model(), lambda y: y.square().sum(),
        (torch.tensor([[.05], [1.]]),), optimizer=SGDConfig(lr=.01))
    program = lower(captured.ir, execution="compose")
    assert "fabsf(" in program.source


def test_existing_scalar_contract_is_preserved():
    assert scalar_regression()["passed"]
