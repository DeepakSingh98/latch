"""FP32 result of an explicitly ordered FP64 weighted fused multiply-add.

For FP32 values a,b,c and scalar FP64 weights alpha,beta:
  float32(fma64(alpha, float64(a), (beta*float64(b))*float64(c))).
The two right-hand products round separately in FP64. Values may be scalar
or have a common shape. This preserves scalar coefficient precision without
enabling FP64 products/reductions elsewhere in the compiler.
"""
import torch

from .ir import CaptureError

TARGET = "latch_ops.mixed_fma64.default"


def _check(a, b, c, alpha, beta):
    if any(x.dtype != torch.float32 for x in (a, b, c)):
        raise CaptureError("MIXED_FMA", "mixed_fma64 values must be FP32")
    if any(x.dtype != torch.float64 or x.shape != () for x in (alpha, beta)):
        raise CaptureError("MIXED_FMA", "mixed_fma64 weights must be scalar FP64 tensors")
    shapes = {tuple(x.shape) for x in (a, b, c) if x.ndim}
    if len(shapes) > 1 or any(x.device != a.device for x in (b, c, alpha, beta)):
        raise CaptureError("MIXED_FMA", "mixed_fma64 requires matching value shapes and one device")
    return next(iter(shapes), ())


@torch.library.custom_op("latch_ops::mixed_fma64", mutates_args=())
def _forward(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
             alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    _check(a, b, c, alpha, beta)
    if not a.is_cuda:
        raise RuntimeError("mixed_fma64 eager oracle requires CUDA; CPU supports fake capture")
    # TensorIterator scalar-alpha add contracts its final FP64 multiply/add.
    # Reading alpha synchronizes only this diagnostic oracle, never native code.
    return torch.add((b.double() * beta) * c.double(), a.double(), alpha=float(alpha)).float()


@_forward.register_fake
def _fake(a, b, c, alpha, beta):
    return a.new_empty(_check(a, b, c, alpha, beta))


class _MixedFma(torch.autograd.Function):
    @staticmethod
    def forward(a, b, c, alpha, beta):
        return _forward(a, b, c, alpha, beta)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, gradient):
        a, b, c, alpha, beta = ctx.saved_tensors
        g = gradient.double()
        return ((g * alpha).sum_to_size(a.shape).float(),
                (g * beta * c.double()).sum_to_size(b.shape).float(),
                (g * beta * b.double()).sum_to_size(c.shape).float(),
                (g * a.double()).sum(), (g * b.double() * c.double()).sum())


def mixed_fma64(a, b, c, alpha, beta):
    """Eager CUDA VJP and fake primal capture; native VJP FP64 arrays are refused."""
    return _MixedFma.apply(a, b, c, alpha, beta)


def expression(target, args, kwargs, name, dtypes, names, rd, layouts):
    from .lower import LoweringError, _ref
    refs = [_ref(value, names) for value in args]
    if (target != TARGET or len(refs) != 5 or kwargs or name is None or layouts is None
            or dtypes.get(name) != "float32" or any(ref is None for ref in refs)
            or any(dtypes.get(ref) != "float32" for ref in refs[:3])
            or any(dtypes.get(ref) != "float64" or layouts[ref].shape != () for ref in refs[3:])
            or any(layouts[ref].shape not in ((), layouts[name].shape) for ref in refs[:3])):
        raise LoweringError("mixed_fma64 requires FP32 values and scalar FP64 weights")
    return (f"static_cast<float>(__fma_rn({rd(3)}, static_cast<double>({rd(0)}), "
            f"__dmul_rn(__dmul_rn({rd(4)}, static_cast<double>({rd(1)})), "
            f"static_cast<double>({rd(2)}))))")
