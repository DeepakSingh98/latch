"""Opt-in FP32 division by a literal using pinned CUDA scalar arithmetic.

PyTorch v2.9.1 BinaryOps.cpp wraps Scalar divisors as CPU tensors, and
BinaryDivTrueKernel.cu computes an FP32 reciprocal on the host before launching
a multiplication functor. This is different from a correctly rounded quotient.
Tensor-valued operands have no such provenance and are never rewritten here.

Sources (independent implementation, no copied kernel body):
https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/BinaryOps.cpp
https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/ScalarOps.h
https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu
https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/jit/python/pybind_utils.cpp
"""
from fractions import Fraction
import math
import struct

KEY = "scalar_division_contract"
CONTRACT = "torch_cuda_reciprocal_v1"


def validate(metadata):
    from .lower import LoweringError
    selected = metadata.get(KEY)
    if selected not in (None, "legacy", CONTRACT):
        raise LoweringError("unknown explicit scalar-division arithmetic contract")
    return selected == CONTRACT


def _normal_f32(value):
    """Round an exact rational to normal binary32, ties to even.

The binary64 estimate is only used to find nearby candidates. Exact rational
distances select the result, including integer-to-float and reciprocal ties.
Subnormal/zero/overflow constants are outside this deliberately narrow rule.
"""
    from .lower import LoweringError
    exact = Fraction(value)
    sign = 0x80000000 if exact < 0 else 0
    magnitude = abs(exact)
    try:
        estimate = struct.unpack("<I", struct.pack("<f", float(magnitude)))[0]
    except (OverflowError, ValueError):
        raise LoweringError("scalar-division contract requires finite normal FP32 divisor and reciprocal") from None
    if not (0x00800000 <= estimate < 0x7f800000):
        raise LoweringError("scalar-division contract requires finite normal FP32 divisor and reciprocal")
    candidates = [bits for bits in (estimate-1, estimate, estimate+1)
                  if 0x00800000 <= bits < 0x7f800000]
    def score(bits):
        number = struct.unpack("<f", struct.pack("<I", bits))[0]
        return abs(Fraction(number)-magnitude), bits & 1
    bits = min(candidates, key=score) | sign
    return struct.unpack("<f", struct.pack("<I", bits))[0], bits


def select(metadata, target, args, kwargs, name, dtypes, names):
    """Return an admitted operation record; never infer scalarhood from shape."""
    from .lower import LoweringError, _ref
    if not validate(metadata):
        return None
    target = str(target).removeprefix("torch.ops.").removeprefix("aten.")
    # make_fx can expose a CPU wrapped number as a literal in div.Tensor.
    # Pinned toPyObject unwraps only is_wrapped_number tensors and asserts CPU
    # device. A Ref remains tensor-valued even if its layout has one element.
    if target not in ("div.Scalar", "div.Tensor") or len(args) != 2:
        return None
    source = _ref(args[0], names)
    if (source is None or dtypes.get(source) != "float32" or dtypes.get(name) != "float32"
            or type(args[1]) not in (int, float)):
        return None
    if kwargs.get("rounding_mode") is not None:
        raise LoweringError("rounded division unsupported by scalar-division contract")
    if not math.isfinite(args[1]) or args[1] == 0:
        raise LoweringError("scalar-division contract requires finite normal FP32 divisor and reciprocal")
    divisor, divisor_bits = _normal_f32(args[1])
    reciprocal, reciprocal_bits = _normal_f32(1 / Fraction(divisor))
    return dict(contract=CONTRACT, name=name, target="aten."+target,
                numerator=source, denominator=args[1], denominator_fp32=divisor,
                denominator_bits=f"0x{divisor_bits:08x}", reciprocal_fp32=reciprocal,
                reciprocal_bits=f"0x{reciprocal_bits:08x}",
                emitter="common_pointwise_fmul_rn", dtype="float32",
                marker=f"/*latch_scalar_div:{name}*/")


def expression(record, numerator):
    return (f"({record['marker']} __fmul_rn({numerator}, "
            f"__uint_as_float({record['reciprocal_bits']}u)))")


def graph_has_selection(graph):
    """Entry-point guard for legacy emitters which cannot preserve this rule."""
    from .lower import _field, _dtype, _ref
    metadata = _field(graph, "metadata", {}) or {}
    if not validate(metadata):
        return False
    raw = _field(graph, "values", {})
    values = raw if isinstance(raw, dict) else {_field(v, "name"): v for v in raw}
    names = set(values)
    dtypes = {name: _dtype(value) for name, value in values.items()}
    for op in _field(graph, "operations", _field(graph, "nodes", [])):
        output = _field(op, "output", _field(op, "name"))
        if output is None:
            outputs = _field(op, "outputs", [])
            if len(outputs) != 1:
                continue
            output = outputs[0]
        name = _ref(output, names) or output
        if select(metadata, _field(op, "target", _field(op, "op", "")),
                  list(_field(op, "args", [])), dict(_field(op, "kwargs", {})),
                  name, dtypes, names):
            return True
    return False
