"""Opt-in pinned PyTorch2.9.1 leading-axis FP32 vector4 SUM arithmetic.

Independent expression of Reduce.cuh setReduceConfig/thread_reduce_impl,
block_y_reduce/global_reduce and ReduceSumProdKernel.cu sum_functor (BSD-3).
The arithmetic was measured for [16384,768] and [8,1572864] on 132-SM H100;
parameterized admission does not qualify other shapes or a complete workload.
"""
import math

KEY = "column_sum_contract"
CONTRACT = "torch_cuda_column_vector4_v1"
HEADER = "torch_column_sum.cuh"


def validate(metadata):
    from .lower import LoweringError
    selected = metadata.get(KEY)
    if selected not in (None, "legacy", CONTRACT):
        raise LoweringError("unknown explicit column-sum arithmetic contract")
    return selected == CONTRACT


def geometry(rows, columns):
    if (any(type(v) is not int or v <= 0 for v in (rows, columns))
            or columns % 4 or rows*columns*4 >= 2**31):
        raise ValueError("column sum requires positive int32-indexed rows and vector4 output")
    div = lambda a,b:(a+b-1)//b
    pow2 = lambda n:1 << (n.bit_length()-1)
    dx,dy = min(pow2(columns//4),128),min(pow2(rows),128)
    bx = min(dx,32); by = min(dy,128//bx); bx = min(dx,128//by)
    reduce_y = rows >= min(by*16,256)
    step_input = by if reduce_y else 1
    step_output = bx if reduce_y else bx*by
    gx = div(columns//4,step_output)
    target = 132*(2048//(bx*by)); per_thread = div(rows,step_input)
    ctas = 1
    if reduce_y and per_thread >= 256 and gx <= target:
        ctas = max(min(div(target,gx),div(per_thread,16)),div(per_thread,256))
    return dict(rows=rows,columns=columns,block=[bx,by,1],grid=[gx,ctas,1],
        reduce_y=reduce_y,step_output=step_output,input_stride=step_input*ctas,
        shared_bytes=bx*by*16 if reduce_y else 0,
        partial_floats=gx*ctas*bx*4 if ctas>1 else 0,
        required_multiprocessors=132,threads_per_sm=2048,
        output_vector=4,accumulators=4)


def leading_sum(target, source_dtype, shape, dims):
    return (target.removeprefix("aten.") == "sum.dim_IntList"
            and source_dtype == "float32" and 0 < len(dims) < len(shape)
            and dims == list(range(len(dims))))


def graph_has_selection(graph):
    """Do not block legacy candidate evaluation for an unrelated region."""
    from .lower import _field,_dtype,_ref
    values=_field(graph,"values")
    if not isinstance(values,dict):values={_field(v,"name"):v for v in values}
    for op in _field(graph,"operations",_field(graph,"nodes",[])):
        target=str(_field(op,"target",_field(op,"op",""))).removeprefix("torch.ops.")
        args=_field(op,"args",[]);kwargs=_field(op,"kwargs",{})
        if target.removeprefix("aten.")!="sum.dim_IntList" or not args:continue
        source=_ref(args[0],set(values))
        if source not in values:continue
        shape=tuple(_field(values[source],"shape"))
        if not shape:continue
        dims=args[1] if len(args)>1 else kwargs.get("dim")
        dims=list(range(len(shape))) if dims is None or dims==[] or dims==() else ([dims] if isinstance(dims,int) else list(dims))
        dims=sorted({int(d)%len(shape) for d in dims})
        if leading_sum(target,_dtype(values[source]),shape,dims):return True
    return False


def plan(metadata, *, target, source, output, source_layout, output_layout,
         source_dtype, output_dtype, dims, accumulate=False):
    from .lower import LoweringError
    if not validate(metadata) or not leading_sum(target,source_dtype,source_layout.shape,dims):
        return None
    if (not source_layout.contiguous or not output_layout.contiguous
            or source_layout.offset % 4 or output_layout.offset % 4
            or output_dtype != "float32" or accumulate):
        raise LoweringError("explicit column-sum requires materialized aligned contiguous FP32 source/result")
    try:
        config = geometry(math.prod(source_layout.shape[:len(dims)]),
                          math.prod(source_layout.shape[len(dims):]))
    except ValueError as exc:
        raise LoweringError("explicit column-sum admission: "+str(exc)) from exc
    return dict(contract=CONTRACT,output=output,source=source,rows=config["rows"],
        columns=config["columns"],geometry=config,materialized_input=True,
        required_multiprocessors=132,source_shape=list(source_layout.shape),
        source_strides=list(source_layout.strides),source_offset=source_layout.offset,
        dimensions=list(dims),qualification="measured two primary shapes; other shapes/full workload require qualification")


def phases(record, *, ident, scratch):
    """Return ordinary body records; caller owns typed layouts and dependencies."""
    c = record["geometry"]; bx,by,_ = c["block"]; gx,ctas,_ = c["grid"]
    source,output = record["source"],record["output"]
    template = ",".join(map(str,(c["rows"],c["columns"],bx,by,gx,ctas,
                                  int(c["reduce_y"]),c["step_output"])))
    partial_output = scratch if ctas>1 else output
    code = (f"for(int task=blockIdx.x;task<{gx*ctas};task+=gridDim.x) {{ "
            f"latch_torch_column_sum::partial<{template}>({ident(source)},"
            f"{ident(partial_output)},task,smem); }}")
    def phase(stage,write,reads,body,work):
        return dict(kind="column_sum",name=write,refs=reads,code=body,
            extra=dict(engine="torch_cuda_column_vector4_"+stage,
                column_arithmetic=CONTRACT,column_sum_contract={**record,"stage":stage},
                work_items=work,shared_bytes=c["shared_bytes"],threads=256,
                reduction_elements=c["rows"],
                arithmetic="materialized FP32 leaves; four independent accumulators; ascending combine; ordered Y tree and CTA partials; no FMA"))
    answer=[phase("partial" if ctas>1 else "single",partial_output,[source],code,gx*ctas)]
    if ctas>1:
        code=(f"for(int task=blockIdx.x;task<{gx};task+=gridDim.x) {{ "
              f"latch_torch_column_sum::finish<{c['columns']},{bx},{by},{ctas}>"
              f"({ident(scratch)},{ident(output)},task,smem); }}")
        answer.append(phase("final",output,[scratch],code,gx))
    return answer


def required_sms(manifest):
    """Runtime checks actual emitted bodies, including nested plans."""
    found=[]
    def visit(value):
        if isinstance(value,dict):
            if value.get("column_arithmetic") == CONTRACT:
                detail=value.get(KEY,{})
                if detail.get("required_multiprocessors")!=132:
                    raise ValueError("column-sum manifest omitted its 132-SM arithmetic binding")
                found.append(132)
            for child in value.values():visit(child)
        elif isinstance(value,(tuple,list)):
            for child in value:visit(child)
    visit(manifest)
    return 132 if found else None
