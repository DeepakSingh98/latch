"""Explicit FP32 scalar scaled addition with one final rounding.

``scaled_add(input, other, alpha)`` means fmaf(other, alpha, input),
with matching FP32 tensor shapes and a runtime FP32 scalar alpha. It is an
operation contract, not an optimizer recognition or global contraction rule.
The eager CUDA oracle uses torch.add's scalar-alpha API; reading alpha there
synchronizes and is diagnostic only. Fake capture preserves a dynamic tensor
operand, and generated execution reads it directly without a host operation.
"""
import torch

from .ir import CaptureError

TARGET = "latch_ops.scaled_add_f32.default"


def _check(input, other, alpha):
    if any(x.dtype != torch.float32 for x in (input, other, alpha)):
        raise CaptureError("SCALED_ADD", "scaled_add requires three FP32 operands")
    if input.shape != other.shape or alpha.shape != ():
        raise CaptureError("SCALED_ADD", "scaled_add requires equal value shapes and scalar alpha")
    if other.device != input.device or alpha.device != input.device:
        raise CaptureError("SCALED_ADD", "scaled_add operands must share a device")


@torch.library.custom_op("latch_ops::scaled_add_f32", mutates_args=())
def _forward(input: torch.Tensor, other: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    _check(input, other, alpha)
    if not input.is_cuda:
        raise RuntimeError("scaled_add eager oracle requires CUDA; CPU supports fake capture")
    return torch.add(input, other, alpha=float(alpha))


@_forward.register_fake
def _fake(input, other, alpha):
    _check(input, other, alpha)
    return input.new_empty(input.shape)


class _ScaledAdd(torch.autograd.Function):
    @staticmethod
    def forward(input, other, alpha):
        return _forward(input, other, alpha)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1], inputs[2])

    @staticmethod
    def backward(ctx, gradient):
        other, alpha = ctx.saved_tensors
        return gradient, gradient * alpha, (gradient * other).sum()


def scaled_add(input, other, alpha):
    return _ScaledAdd.apply(input, other, alpha)


def expression(target, args, kwargs, name, dtypes, names, rd, layouts):
    """Independent metadata admission shared by all expression schedules."""
    from .lower import LoweringError, _ref
    refs = [_ref(value, names) for value in args]
    if (target != TARGET or len(refs) != 3 or kwargs or name is None or layouts is None
            or dtypes.get(name) != "float32" or any(dtypes.get(ref) != "float32" for ref in refs)
            or layouts[refs[0]].shape != layouts[refs[1]].shape
            or layouts[refs[2]].shape != ()
            or layouts[name].shape != layouts[refs[0]].shape):
        raise LoweringError("scaled_add_f32 requires matching FP32 values and one FP32 scalar alpha")
    return f"fmaf({rd(1)}, {rd(2)}, {rd(0)})"
