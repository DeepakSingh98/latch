"""Opt-in FP32 warp32 CUDA softmax VJP arithmetic.

Independently expressed from PyTorch v2.9.1 (BSD-3-Clause) SoftMax.cu's
materialized ``grad * output`` and PersistentSoftmax.cuh's warp reduction and
backward epilogue. LATCH-MOE4's identical-input component measured explicit FMA
bitwise equal on all 32 real rank/layer fixtures; this is not full-model approval.
"""
import math

KEY = "softmax_backward_contract"
CONTRACT = "torch_cuda_warp32_v1"
ROW_ARITHMETIC = CONTRACT + "_backward"
TARGET = "_softmax_backward_data.default"
REFERENCES = (
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/SoftMax.cu",
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/PersistentSoftmax.cuh",
)


def validate(metadata):
    from .lower import LoweringError
    selected = metadata.get(KEY)
    if selected not in (None, "legacy", CONTRACT):
        raise LoweringError("unknown explicit softmax backward arithmetic contract")
    return selected == CONTRACT


def emit(metadata, target, args, kwargs, name, shape, layouts, dtypes, names):
    from .lower import LoweringError, _ident, _index, _ref
    if not validate(metadata) or target.removeprefix("aten.") != TARGET:
        return None
    if (not shape or any(v <= 0 for v in shape) or shape[-1] != 32
            or int(args[2]) % len(shape) != len(shape) - 1):
        raise LoweringError("explicit softmax backward requires last-axis width32")
    if str(args[3]).removeprefix("torch.") != "float32":
        raise LoweringError("explicit softmax backward requires FP32 input dtype")
    grad, probability = (_ref(args[i], names) for i in (0, 1))
    for value in (grad, probability, name):
        layout = layouts[value]
        if (dtypes[value] != "float32" or tuple(layout.shape) != tuple(shape)
                or layout.offset < 0 or any(s < 0 for s in layout.strides)):
            raise LoweringError("explicit softmax backward requires matching FP32 positive-stride rows")
    if not layouts[name].contiguous:
        raise LoweringError("explicit softmax backward requires contiguous FP32 output")
    rows = math.prod(shape) // 32
    index = "(row*32LL+lane)"
    g = f"{_ident(grad)}[{_index(layouts[grad], shape, index)}]"
    p = f"{_ident(probability)}[{_index(layouts[probability], shape, index)}]"
    code = f'''for(long long row=((long long)blockIdx.x*blockDim.x+threadIdx.x)/32;
    row<{rows}LL;row+=(long long)gridDim.x*blockDim.x/32) {{
  const int lane=threadIdx.x&31;
  const float probability={p};
  const float product=__fmul_rn({g},probability);
  float total=__fadd_rn(0.f,product);
#pragma unroll
  for(int offset=16;offset>0;offset>>=1)
    total=__fadd_rn(total,__shfl_xor_sync(0xffffffffu,total,offset));
  {_ident(name)}[row*32LL+lane]=fmaf(-probability,total,product);
}}'''
    extra = dict(engine="torch_cuda_softmax_backward_warp32", row_arithmetic=ROW_ARITHMETIC,
        reduction_elements=32, work_items=rows, threads=256, shared_bytes=0,
        softmax_backward_contract=dict(contract=CONTRACT, kind="backward", width=32, rows=rows, dtype="float32",
            source_order="materialize_grad_times_probability_then_descending_xor_sum_then_fma",
            arithmetic="FP32 materialized product; XOR16,8,4,2,1 sum; explicit FMA epilogue",
            input_gradient_strides=list(layouts[grad].strides),
            qualification="identical-input warp32 evidence; complete workload must qualify"))
    return "softmax", code, extra
