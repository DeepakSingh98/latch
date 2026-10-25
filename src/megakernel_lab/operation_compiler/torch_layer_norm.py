"""Metadata-only proof for an opt-in CUDA FP32 affine LayerNorm contract.

Recognizes the ordinary captured two-pass equation independently of its caller,
then describes one replacement producing output, mean and reciprocal standard
deviation at the mean anchor. No operations, source graph, tensors or default
arithmetic are changed here. A caller must explicitly select the contract and
qualify the resulting implementation; width admission is not a numerical claim.

The intended vector4 Welford arithmetic is the pinned PyTorch 2.9.1 CUDA
layer_norm_kernel.cu algorithm. This module contains only structural recognition,
not that implementation, and has no torch or device dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

KEY = "layer_norm_contract"
CONTRACT = "torch_cuda_layer_norm_vector4_v1"
# LATCH-MOE-EMA-FAST: rows per 256-thread CTA in the generated forward phase. 2 runs one pinned 128-thread row
# computation in each half of the CTA (torch_ln_row2); per-row arithmetic is unchanged. 1 restores torch_ln_row.
ROWS_PER_CTA = 2
# LATCH-MOE-EMA-100 (n01): rows per 128-thread half in the forward phase (torch_ln_rows<R>); 1 keeps torch_ln_row2.
LN_ROWS_PER_HALF = int(__import__("os").environ.get("LATCH_OPS_LN_ROWS_PER_HALF", "4"))


class Refused(ValueError):
    """The complete local dataflow or storage proof did not hold."""


@dataclass(frozen=True)
class LayerNormPlan:
    anchor: str
    x: str
    gain: str
    bias: str
    output: str
    mean: str
    rstd: str
    rows: int
    width: int
    epsilon: float
    members: tuple[str, ...]
    intermediates: tuple[str, ...]
    contract: str = CONTRACT

    @property
    def finals(self):
        return (self.output, self.mean, self.rstd)

    def to_dict(self):
        return asdict(self)


def _data(ir):
    return ir if isinstance(ir, dict) else ir.to_dict()


def _ref(value):
    return value.get("ref") if isinstance(value, dict) and set(value) == {"ref"} else None


def _references(value):
    if isinstance(value, dict):
        if set(value) == {"ref"}:
            return {value["ref"]}
        return set().union(*(_references(v) for v in value.values())) if value else set()
    if isinstance(value, (tuple, list)):
        return set().union(*(_references(v) for v in value)) if value else set()
    return set()


def _strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_strings(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(_strings(v) for v in value)) if value else set()
    return set()


def _shape(value):
    return tuple(value["shape"])


def _contiguous(value, *, aligned=False):
    shape, strides = _shape(value), tuple(value["strides"])
    if len(shape) != len(strides) or any(type(n) is not int or n <= 0 for n in shape):
        return False
    expected = 1
    for n, stride in zip(reversed(shape), reversed(strides)):
        if n != 1 and stride != expected:
            return False
        expected *= n
    offset = value.get("storage_offset", 0)
    return (type(offset) is int and offset >= 0 and (not aligned or offset % 4 == 0)
            and expected < 2**31)


def _one(rows, description):
    if len(rows) != 1:
        raise Refused(f"expected one {description}; found {len(rows)}")
    return rows[0]


def _args(op):
    return list(op.get("args", ()))


def _binary(op, target, left, right):
    args = _args(op)
    return (op["target"] == target and len(args) == 2
            and _ref(args[0]) == left and _ref(args[1]) == right)


def _reduction(op, target, source, axis, rank):
    args = _args(op)
    if op["target"] != target or len(args) != 3 or _ref(args[0]) != source or args[2] is not True:
        return False
    dims = args[1]
    return (isinstance(dims, (tuple, list)) and len(dims) == 1
            and type(dims[0]) is int and dims[0] % rank == axis
            and -rank <= dims[0] < rank)


def _match(data, mean):
    operations, values = data["operations"], data["values"]
    order = {op["name"]: i for i, op in enumerate(operations)}
    producers = {name: op for op in operations for name in op["outputs"]}
    if len(mean["outputs"]) != 1 or not _args(mean):
        raise Refused("mean must have one output and a direct input")
    x = _ref(_args(mean)[0])
    if x not in values:
        raise Refused("mean input is not a tensor value")
    shape = _shape(values[x])
    if not shape or not _reduction(mean, "aten.mean.dim", x, len(shape)-1, len(shape)):
        raise Refused("mean must retain the final dimension")
    width, rows = shape[-1], math.prod(shape[:-1])
    if type(width) is not int or width <= 0 or width % 4 or width > 4096:
        raise Refused("vector4 width must be positive, divisible by four and at most4096")
    if values[x]["dtype"] != "float32" or not _contiguous(values[x], aligned=True):
        raise Refused("input must be aligned contiguous FP32")
    mean_name = mean["outputs"][0]
    centers = [op for op in operations if _binary(op, "aten.sub.Tensor", x, mean_name)]
    centered_names = {op["outputs"][0] for op in centers if len(op["outputs"]) == 1}
    square = _one([op for op in operations if any(
        _binary(op, "aten.mul.Tensor", n, n) for n in centered_names)], "centered square")
    summed = _one([op for op in operations if _reduction(op, "aten.sum.dim_IntList",
        square["outputs"][0], len(shape)-1, len(shape))], "centered-square sum")
    variance = _one([op for op in operations if op["target"] == "aten.div.Tensor"
        and len(_args(op)) == 2 and _ref(_args(op)[0]) == summed["outputs"][0]
        and type(_args(op)[1]) in (float, int) and _args(op)[1] == width], "population variance")
    eps = _one([op for op in operations if op["target"] == "aten.add.Tensor"
        and len(_args(op)) == 2 and _ref(_args(op)[0]) == variance["outputs"][0]
        and type(_args(op)[1]) in (float, int)], "variance plus scalar epsilon")
    epsilon = float(_args(eps)[1])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise Refused("epsilon must be a finite positive scalar")
    reciprocal = _one([op for op in operations if op["target"] == "aten.rsqrt.default"
        and len(_args(op)) == 1 and _ref(_args(op)[0]) == eps["outputs"][0]], "reciprocal standard deviation")
    rstd = reciprocal["outputs"][0]
    normalized = _one([op for op in operations if any(
        _binary(op, "aten.mul.Tensor", n, rstd) for n in centered_names)], "centered normalization")
    scaled = _one([op for op in operations if op["target"] == "aten.mul.Tensor"
        and len(_args(op)) == 2 and _ref(_args(op)[0]) == normalized["outputs"][0]
        and _ref(_args(op)[1]) in values], "affine gain")
    final = _one([op for op in operations if op["target"] == "aten.add.Tensor"
        and len(_args(op)) == 2 and _ref(_args(op)[0]) == scaled["outputs"][0]
        and _ref(_args(op)[1]) in values], "affine bias")
    gain, bias, output = _ref(_args(scaled)[1]), _ref(_args(final)[1]), final["outputs"][0]
    used_centers = {_ref(_args(square)[0]), _ref(_args(normalized)[0])}
    selected = [mean, *(op for op in centers if op["outputs"][0] in used_centers), square,
                summed, variance, eps, reciprocal, normalized, scaled, final]
    members = {op["name"] for op in selected}
    finals = {output, mean_name, rstd}
    made = {name for op in selected for name in op["outputs"]}
    intermediates = made - finals
    anchor_index = order[mean["name"]]
    for name in (x, gain, bias):
        value = values[name]
        if value["dtype"] != "float32" or not _contiguous(value, aligned=True):
            raise Refused("all source bindings must be aligned contiguous FP32")
        if name in producers:
            if order[producers[name]["name"]] >= anchor_index:
                raise Refused("all source bindings must be ready before the mean anchor")
        elif name not in data["inputs"]:
            raise Refused("source binding is neither an input nor an earlier producer")
    if _shape(values[gain]) != (width,) or _shape(values[bias]) != (width,):
        raise Refused("affine gain and bias must each have shape[width]")
    stats_shape = shape[:-1] + (1,)
    row_values = {mean_name, summed["outputs"][0], variance["outputs"][0], eps["outputs"][0], rstd}
    for op in selected:
        if len(op["outputs"]) != 1 or op.get("effect", "pure") != "pure":
            raise Refused("members must be pure single-output operations")
        kwargs = op.get("kwargs", {})
        allowed = {"alpha": 1} if op["target"] in ("aten.add.Tensor", "aten.sub.Tensor") else (
            {"dtype": "float32"} if op["target"] in ("aten.mean.dim", "aten.sum.dim_IntList") else {})
        if any(k not in allowed or v != allowed[k] for k, v in kwargs.items()):
            raise Refused("member has unsupported nondefault keyword semantics")
        name = op["outputs"][0]
        if values[name]["dtype"] != "float32" or _shape(values[name]) != (stats_shape if name in row_values else shape):
            raise Refused("member dtype/shape does not preserve FP32 row arithmetic")
        if order[op["name"]] < anchor_index:
            raise Refused("member precedes the mean anchor")
    if any(not _contiguous(values[n], aligned=n == output) for n in finals):
        raise Refused("the three emitted finals must have contiguous storage")
    # A skipped temporary cannot feed a different operation or leave the graph.
    # Mean/rstd/output remain material outputs and may have arbitrary later readers.
    escaping = _strings(data.get("outputs", {})) | set(data.get("state_writes", {}))
    escaping |= set(data.get("state_writes", {}).values())
    escaping |= {name for name, v in values.items() if v.get("output_roles")}
    if intermediates & escaping:
        raise Refused("a skipped intermediate escapes the group")
    for op in operations:
        reads = set(op.get("inputs", ())) | set(op.get("reads", ())) | _references((op.get("args"), op.get("kwargs")))
        if op["name"] not in members and reads & intermediates:
            raise Refused("a skipped intermediate has an external reader")
        if op["name"] not in members and anchor_index <= order[op["name"]] <= order[final["name"]]:
            if op.get("effect", "pure") != "pure" or set(op.get("writes", ())) & {x, gain, bias}:
                raise Refused("an intervening effect can alter replacement operands")
    for name, value in values.items():
        if value.get("alias_of") in intermediates and name not in made:
            raise Refused("a skipped intermediate has an external alias")
    # Every internal dependency must already exist at its original operation;
    # this catches invalid metadata instead of depending on accidental op names.
    for op in selected:
        for name in _references((op.get("args"), op.get("kwargs"))):
            if name in producers and order[producers[name]["name"]] >= order[op["name"]]:
                raise Refused("member dependency is not topologically ready")
    return LayerNormPlan(mean["name"], x, gain, bias, output, mean_name, rstd, rows, width,
                         epsilon, tuple(op["name"] for op in operations if op["name"] in members),
                         tuple(name for op in operations if op["name"] in members
                               for name in op["outputs"] if name in intermediates))


def match_mean(ir: Any, operation_name: str) -> LayerNormPlan:
    data = _data(ir)
    if data.get("metadata", {}).get(KEY) != CONTRACT:
        raise Refused("the explicit LayerNorm contract is not selected")
    op = _one([op for op in data["operations"] if op["name"] == operation_name], "named mean anchor")
    return _match(data, op)


def analyze(ir: Any):
    """Return plans and explicit refusals; default metadata has no effect."""
    data = _data(ir)
    selected = data.get("metadata", {}).get(KEY)
    if selected is None:
        return dict(contract=CONTRACT, enabled=False, plans=[], refused=[])
    if selected != CONTRACT:
        raise ValueError("unknown explicit LayerNorm arithmetic contract: " + str(selected))
    plans, refused, occupied = [], [], set()
    for op in data["operations"]:
        if op["target"] != "aten.mean.dim":
            continue
        try:
            candidate = _match(data, op)
            if occupied.intersection(candidate.members):
                raise Refused("replacement groups overlap")
        except Refused as error:
            refused.append(dict(anchor=op["name"], reason=str(error)))
            continue
        plans.append(candidate)
        occupied.update(candidate.members)
    return dict(contract=CONTRACT, enabled=True, plans=plans, refused=refused)


def find_plans(ir: Any) -> list[LayerNormPlan]:
    return analyze(ir)["plans"]
