"""Explicit pinned CUDA FP32 log-softmax arithmetic, independently generated.

The selected regular-ILP4 dispatch is structural. Its numerical evidence is
currently the primary width50257 fixture, not a blanket claim for other widths.
"""
import math

KEY="log_softmax_contract"
CONTRACT="torch_cuda_logsoftmax_ilp4_v1"
HEADER="torch_logsoftmax.cuh"


def emit(metadata,target,args,kwargs,name,shape,layouts,dtypes,names):
    from .lower import LoweringError,_ident,_ref
    selected=metadata.get(KEY)
    if selected in (None,"legacy"):return None
    if selected!=CONTRACT:raise LoweringError("unknown explicit log-softmax arithmetic contract")
    target=target.removeprefix("aten.")
    if target not in ("_log_softmax.default","_log_softmax_backward_data.default"):return None
    backward="backward" in target
    source=_ref(args[1] if backward else args[0],names)
    dimension=int(args[2] if backward else args[1])
    width=shape[-1] if shape else 0
    # These widths take pinned host_softmax's regular1024-thread ILP4 path:
    # register_count>=10, no aligned shared-cache dispatch. Backward is512.
    if (not shape or any(v<=0 for v in shape) or dimension%len(shape)!=len(shape)-1
            or width<=9216 or width%4==0 or width>=2**31):
        raise LoweringError("explicit log-softmax requires regular ILP4 last-axis width>9216, width%4!=0")
    operands=[source]
    if backward:
        operands.insert(0,_ref(args[0],names))
        if str(args[3]).removeprefix("torch.")!="float32":
            raise LoweringError("explicit log-softmax VJP input dtype must be FP32")
    elif bool(args[2] if len(args)>2 else kwargs.get("half_to_float",False)):
        raise LoweringError("explicit log-softmax half_to_float is refused")
    for value in [*operands,name]:
        layout=layouts[value]
        if (dtypes[value]!="float32" or tuple(layout.shape)!=tuple(shape)
                or not layout.contiguous or int(layout.offset)%4):
            raise LoweringError("explicit log-softmax requires aligned contiguous FP32 logical rows")
    rows=math.prod(shape)//width
    direction="backward" if backward else "forward"
    if backward:
        call=f"latch_torch_logsoftmax::backward_row({_ident(operands[0])},{_ident(source)},{_ident(name)},{width},row,true,smem);"
    else:
        call=f"latch_torch_logsoftmax::forward_row({_ident(source)},{_ident(name)},{width},row,smem);"
    code=f"for(long long row=blockIdx.x;row<{rows}LL;row+=gridDim.x) {{ {call} }}"
    extra=dict(engine="torch_cuda_logsoftmax_ilp4_"+direction,
        row_arithmetic=CONTRACT+"_"+direction,reduction_elements=width,work_items=rows,
        threads=256,shared_bytes=2048 if backward else 128,
        log_softmax_contract=dict(contract=CONTRACT,direction=direction,width=width,rows=rows,
            logical_threads=512 if backward else 1024,physical_threads=256,
            arithmetic="pinned ILP4 and ordered reduction; explicit FMA backward epilogue",
            qualification="measured width50257 only; full workload must qualify"))
    return "log_softmax",code,extra
