"""Bijective row movement with an explicitly supplied, validated inverse.

Unlike a general gather, this operation has one source and one destination per
row. Its first-order VJP is another row copy, with the two permutations swapped;
there is no floating reduction or rounding boundary. The runtime inverse check
establishes bijectivity and rejects duplicate/missing rows or a stale inverse.
"""
from __future__ import annotations

import torch

TARGET = "latch_ops.permutation_gather.default"


def _metadata(x, order, inverse):
    if x.ndim != 2 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("permutation_gather requires a nonempty floating row matrix")
    if any(p.dtype != torch.int64 or p.shape != (x.shape[0],) or p.device != x.device
           for p in (order, inverse)):
        raise ValueError("row permutation and inverse must be matching device int64 row vectors")


@torch.library.custom_op("latch_ops::permutation_gather", mutates_args=())
def _gather(x: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    _metadata(x, order, inverse)
    rows = x.shape[0]
    if bool(((order < 0) | (order >= rows)).any()) or not torch.equal(
            inverse[order], torch.arange(rows, device=x.device)):
        raise ValueError("row permutation is not bijective or its inverse differs")
    return x.index_select(0, order)


@_gather.register_fake
def _gather_fake(x, order, inverse):
    _metadata(x, order, inverse)
    return x.new_empty(x.shape)


class _PermutationGather(torch.autograd.Function):
    @staticmethod
    def forward(x, order, inverse):
        return _gather(x, order, inverse)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1], inputs[2])

    @staticmethod
    def backward(ctx, dy):
        order, inverse = ctx.saved_tensors
        with torch.no_grad():
            dx = _gather(dy, inverse, order)
        return dx, None, None


def permutation_gather(x, order, inverse):
    """Copy ``x[order]``; require ``inverse[order] == arange(rows)`` at runtime.

    Both primal and VJP preserve the source storage bits, including signed zero.
    Supply route_order's (order,inverse) or (inverse,order), depending on direction.
    The explicit operation is optional; ordinary gathers retain their own VJP.
    """
    return _PermutationGather.apply(x, order.contiguous(), inverse.contiguous())


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """Generic generated copy with runtime inverse validation; no model pattern."""
    from .lower import LoweringError
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ())) or (field(op, "output", field(op, "name")),)
    if len(args) != 3 or len(outputs) != 1 or field(op, "kwargs", {}):
        raise LoweringError("permutation_gather has three tensor operands and one result")
    x, order, inverse = [ref(a, names) for a in args]
    name = outputs[0]
    lx, lo, li = (layouts[a] for a in (x, order, inverse))
    shape = tuple(field(values[name], "shape"))
    if (len(lx.shape) != 2 or min(lx.shape) < 1 or shape != lx.shape
            or lo.shape != (lx.shape[0],) or li.shape != lo.shape
            or not lo.contiguous or not li.contiguous
            or dtypes[order] != "int64" or dtypes[inverse] != "int64"
            or dtypes[x] not in ("float32", "bfloat16") or dtypes[name] != dtypes[x]):
        raise LoweringError("generated permutation_gather requires BF16/FP32 rows and contiguous int64 permutations")
    define(name, shape, None, dtypes[name])
    rows, width = shape
    def ptr(key):
        layout = layouts[key]
        return f"({ident(layout.root)} + {layout.offset}LL)"
    code = f"""for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
     i < {rows * width}LL; i += (long long)gridDim.x * blockDim.x) {{
  const long long r = i / {width}LL, c = i % {width}LL;
  const int64_t source = {ptr(order)}[r];
  if (source < 0 || source >= {rows}LL) __trap();
  if ({ptr(inverse)}[source] != r) __trap();
  {ptr(name)}[i] = {ptr(x)}[source * {lx.strides[0]}LL + c * {lx.strides[1]}LL];
}}"""
    return [dict(kind="indexed_permutation", names=[name], targets=[TARGET], reads=[x, order, inverse],
        writes=[name], code=code, threads=256, work_items=(rows * width + 255) // 256,
        shared_bytes=0, requires_full_grid=True, engine="validated_bijective_row_copy",
        arithmetic="storage-preserving row movement; inverse-gather VJP without floating reduction")]
