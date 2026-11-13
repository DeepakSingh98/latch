"""Rules that turn ATen operations into C expressions.

A pointwise rule computes one element of its result from the elements of its
operands at the same (broadcast) position, so any chain of them can share one
loop. A rule may also declare *memory* operands: tensors it reads at computed
positions (a gather, a reduction over a few elements), which must be in memory
before the loop starts. ``emit(e, op)`` receives the loop emitter ``e`` from
codegen.py and returns the C expression for the result element.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from .ir import CompileError, Op, Ref

CTYPE = {"float32": "float", "int64": "long long", "bool": "bool"}
SIZE = {"float32": 4, "int64": 8, "bool": 1}


def literal(value, dtype: str) -> str:
    if dtype == "bool":
        return "true" if value else "false"
    if dtype == "int64":
        return f"{int(value)}LL"
    value = float(value)
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "(-INFINITY)"
    return f"{value!r}f"


def kind(e, x) -> str:
    """The dtype an operand brings to type promotion; Python scalars are weakly typed."""
    if isinstance(x, Ref):
        return e.dtype(x)
    return "bool" if isinstance(x, bool) else "int64" if isinstance(x, int) else "float32"


def promote(e, *xs) -> str:
    kinds = [kind(e, x) for x in xs]
    return "float32" if "float32" in kinds else "int64" if "int64" in kinds else "bool"


@dataclass
class Pointwise:
    emit: Callable
    memory: Callable[[Op], list[str]] = lambda op: []


POINTWISE: dict[str, Pointwise] = {}


def rule(*targets, memory=None):
    def register(emit):
        for target in targets:
            POINTWISE[target] = Pointwise(emit, memory) if memory else Pointwise(emit)
        return emit
    return register


def template(text, *targets, dtype=None):
    """``text`` formats the operands, all converted to ``dtype`` (default: the result's dtype)."""
    @rule(*targets)
    def emit(e, op):
        return text.format(*(e.arg(x, dtype) for x in op.args))


# ---------------------------------------------------------------- arithmetic
@rule("aten.add.Tensor", "aten.add.Scalar", "aten.sub.Tensor", "aten.sub.Scalar")
def _add(e, op):
    a, b = e.arg(op.args[0]), e.arg(op.args[1])
    if op.kwargs.get("alpha", 1) != 1:
        b = f"({e.arg(op.kwargs['alpha'])} * {b})"
    return f"({a} {'+' if '.add.' in op.target else '-'} {b})"


template("({0} * {1})", "aten.mul.Tensor", "aten.mul.Scalar")
template("({0} / {1})", "aten.div.Tensor", "aten.div.Scalar")
template("(-{0})", "aten.neg.default")
template("(1.0f / {0})", "aten.reciprocal.default")
template("fmaxf({0}, {1})", "aten.maximum.default", "aten.clamp_min.default")
template("fminf({0}, {1})", "aten.minimum.default", "aten.clamp_max.default")
template("powf({0}, {1})", "aten.pow.Tensor_Tensor", "aten.pow.Scalar")
template("{0}", "latch.copy", "aten._to_copy.default")
for _name, _c in {"tanh": "tanhf({0})", "exp": "expf({0})", "log": "logf({0})", "sqrt": "sqrtf({0})",
                  "rsqrt": "(1.0f / sqrtf({0}))", "erf": "erff({0})", "expm1": "expm1f({0})",
                  "log1p": "log1pf({0})", "sin": "sinf({0})", "cos": "cosf({0})", "abs": "fabsf({0})",
                  "relu": "fmaxf({0}, 0.0f)", "sigmoid": "(1.0f / (1.0f + expf(-{0})))"}.items():
    template(_c, f"aten.{_name}.default", dtype="float32")


@rule("aten.pow.Tensor_Scalar")
def _pow(e, op):
    x, exponent = e.arg(op.args[0]), op.args[1]
    if exponent == 2:
        return f"({x} * {x})"
    return f"sqrtf({x})" if exponent == 0.5 else f"powf({x}, {literal(exponent, 'float32')})"


@rule("aten.clamp.default")
def _clamp(e, op):
    x = e.arg(op.args[0])
    low, high = (list(op.args[1:]) + [None, None])[:2]
    low, high = op.kwargs.get("min", low), op.kwargs.get("max", high)
    if low is not None:
        x = f"fmaxf({x}, {e.arg(low)})"
    return x if high is None else f"fminf({x}, {e.arg(high)})"


# ---------------------------------------------------------------- comparison, logic, selection
for _name, _c in {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}.items():
    @rule(f"aten.{_name}.Scalar", f"aten.{_name}.Tensor")
    def _compare(e, op, c=_c):
        common = promote(e, *op.args)
        return f"({e.arg(op.args[0], common)} {c} {e.arg(op.args[1], common)})"

template("({0} && {1})", "aten.logical_and.default", "aten.bitwise_and.Tensor", dtype="bool")
template("({0} || {1})", "aten.logical_or.default", "aten.bitwise_or.Tensor", dtype="bool")
template("(!{0})", "aten.logical_not.default", "aten.bitwise_not.default", dtype="bool")


@rule("aten.where.self")
def _where(e, op):
    return f"({e.arg(op.args[0], 'bool')} ? {e.arg(op.args[1])} : {e.arg(op.args[2])})"


# ---------------------------------------------------------------- constants
@rule("aten.full.default", "aten.full_like.default")
def _full(e, op):
    return literal(op.args[1], e.out.dtype)


@rule("aten.scalar_tensor.default")
def _scalar(e, op):
    return literal(op.args[0], e.out.dtype)


@rule("aten.arange.start_step")
def _arange(e, op):
    start, step = op.args[0], op.args[2] if len(op.args) > 2 else 1
    return f"({literal(start, e.out.dtype)} + {literal(step, e.out.dtype)} * {e.coord(0)})"


# ---------------------------------------------------------------- reads at computed positions
@rule("aten.gather.default", memory=lambda op: [op.args[0].name])
def _gather(e, op):
    x, dim, index = op.args[:3]
    coords = [e.coord(d) for d in range(len(e.out.shape))]
    coords[dim] = e.arg(index, "int64")
    return e.at(x, coords)


@rule("aten.index_select.default", memory=lambda op: [op.args[0].name, op.args[2].name])
def _index_select(e, op):
    x, dim, index = op.args
    coords = [e.coord(d) for d in range(len(e.out.shape))]
    coords[dim] = e.at(index, [coords[dim]])
    return e.at(x, coords)


@rule("aten.index.Tensor", memory=lambda op: [r.name for r in (op.args[0], *op.args[1])])
def _index(e, op):
    """``x[i0, i1, ...]`` with index tensors on the leading dimensions (an embedding lookup)."""
    x, indices = op.args
    if any(i is None for i in indices):
        raise CompileError(f"{op.name}: index tensors must cover leading dimensions")
    shape = _broadcast([e.shape(i) for i in indices])
    lead = [e.coord(d) for d in range(len(shape))]
    rest = [e.coord(d) for d in range(len(shape), len(e.out.shape))]
    picked = [e.at(i, lead[len(lead) - len(e.shape(i)):]) for i in indices]
    return e.at(x, picked + rest)


@rule("aten.scatter.value", memory=lambda op: [op.args[2].name])
def _scatter(e, op):
    """One value written per slice along ``dim`` (the one-hot of a class index)."""
    base, dim, index, value = op.args
    dim %= len(e.out.shape)
    if e.shape(index)[dim] != 1:
        raise CompileError(f"{op.name}: scatter needs an index of extent 1 along its dimension")
    coords = [e.coord(d) for d in range(len(e.out.shape))]
    target = e.at(index, coords[:dim] + ["0"] + coords[dim + 1:])
    return f"({coords[dim]} == {target} ? {literal(value, e.out.dtype)} : {e.arg(base)})"


@rule("aten.cat.default", memory=lambda op: [r.name for r in op.args[0]])
def _cat(e, op):
    tensors, dim = op.args[0], (op.args[1] if len(op.args) > 1 else op.kwargs.get("dim", 0))
    dim = dim + len(e.out.shape) if dim < 0 else dim
    coords, text, start = [e.coord(d) for d in range(len(e.out.shape))], "", 0
    for position, x in enumerate(tensors):
        extent = e.shape(x)[dim]
        here = e.at(x, coords[:dim] + [f"({coords[dim]} - {start})"] + coords[dim + 1:])
        start += extent
        text += here if position == len(tensors) - 1 else f"{coords[dim]} < {start} ? {here} : "
    return f"({text})"


def _broadcast(shapes):
    rank = max(len(s) for s in shapes)
    padded = [(1,) * (rank - len(s)) + tuple(s) for s in shapes]
    return tuple(max(extents) for extents in zip(*padded))


# ---------------------------------------------------------------- reductions
@dataclass
class Reduction:
    initial: Callable[[str], str]        # dtype -> starting value
    combine: Callable[[str, str], str]   # two C expressions -> their combination
    mean: bool = False


REDUCTIONS = {
    "aten.sum.dim_IntList": Reduction(lambda dtype: literal(0, dtype), lambda a, b: f"{a} + {b}"),
    "aten.sum.default": Reduction(lambda dtype: literal(0, dtype), lambda a, b: f"{a} + {b}"),
    "aten.mean.dim": Reduction(lambda dtype: literal(0, dtype), lambda a, b: f"{a} + {b}", mean=True),
    "aten.mean.default": Reduction(lambda dtype: literal(0, dtype), lambda a, b: f"{a} + {b}", mean=True),
    "aten.amax.default": Reduction(lambda dtype: "(-INFINITY)", lambda a, b: f"fmaxf({a}, {b})"),
    "aten.amin.default": Reduction(lambda dtype: "INFINITY", lambda a, b: f"fminf({a}, {b})"),
}


def reduced(op: Op, shape) -> tuple[list[int], bool]:
    """(dimensions reduced, keepdim) of a reduction over a source of ``shape``."""
    dims = op.args[1] if len(op.args) > 1 else op.kwargs.get("dim")
    keepdim = op.args[2] if len(op.args) > 2 else op.kwargs.get("keepdim", False)
    if dims is None or len(dims) == 0:
        dims = range(len(shape))
    return sorted(d + len(shape) if d < 0 else d for d in dims), bool(keepdim)


def reduced_extent(op: Op, shape) -> int:
    return math.prod(shape[d] for d in reduced(op, shape)[0])


def _reduce(e, op):
    """A short reduction, run by one thread per result element."""
    spec, x = REDUCTIONS[op.target], op.args[0]
    shape = e.shape(x)
    dims, keepdim = reduced(op, shape)
    coords, loops, position = [], [], 0
    for d, extent in enumerate(shape):
        if d in dims:
            loops.append(f"for (int r{d} = 0; r{d} < {extent}; ++r{d})")
            coords.append(f"r{d}")
            position += keepdim
        else:
            coords.append(e.coord(position))
            position += 1
    ctype, acc = CTYPE[e.out.dtype], f"r_{op.name}"
    element = f"({ctype}){e.at(x, coords)}"
    e.before(f"{ctype} {acc} = {spec.initial(e.out.dtype)}; {' '.join(loops)} {acc} = {spec.combine(acc, element)};")
    return f"({acc} / {literal(reduced_extent(op, shape), 'float32')})" if spec.mean else acc


for _target in REDUCTIONS:
    POINTWISE[_target] = Pointwise(_reduce, lambda op: [op.args[0].name])
