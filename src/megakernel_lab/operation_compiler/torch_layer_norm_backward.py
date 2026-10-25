"""Metadata proof for the explicit CUDA LayerNorm input-gradient arithmetic.

Recognizes ordinary captured dX equations without caller/model names. Shared
ancestors stay in the graph for gain/bias gradients and arbitrary other readers;
only proved-private arithmetic and the replaced dX result become body members.
This module selects no kernel and has no Torch/device dependency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .torch_layer_norm import (KEY, CONTRACT, Refused, _args, _contiguous, _data,
                              _one, _ref, _references, _reduction, _shape, _strings)

ROW_ARITHMETIC = CONTRACT + "_backward_dx"
AFFINE_KEY = "layer_norm_affine_contract"
AFFINE_CONTRACT = "torch_cuda_layer_norm_gamma_beta_v1"
# LATCH-MOE-EMA-FAST: run the affine gradient's logical-thread chains on four CTAs per column tile (then the same
# tree); False keeps one CTA per column tile.
AFFINE_SPLIT = True


@dataclass(frozen=True)
class LayerNormBackwardPlan:
    anchor: str
    x: str
    dy: str
    gain: str
    mean: str
    rstd: str
    output: str
    rows: int
    width: int
    members: tuple[str, ...]
    intermediates: tuple[str, ...]
    preserved: tuple[str, ...]
    contract: str = CONTRACT

    @property
    def finals(self):
        return (self.output,)

    def to_dict(self):
        return asdict(self)


def _match(data, final):
    ops, values = data["operations"], data["values"]
    producers = {name: op for op in ops for name in op["outputs"]}
    order = {op["name"]: i for i, op in enumerate(ops)}
    selected = {}

    def node(name, target):
        op = producers.get(name)
        if op is None or op["target"] != target or len(op["outputs"]) != 1:
            raise Refused("dX expression has an unexpected producer")
        selected[op["name"]] = op
        return op

    def ref(args, i):
        name = _ref(args[i]) if i < len(args) else None
        if name not in values:
            raise Refused("dX expression requires a direct tensor operand")
        return name

    if len(final["outputs"]) != 1 or len(_args(final)) != 2:
        raise Refused("dX final must be a binary product")
    output = final["outputs"][0]
    node(output, "aten.mul.Tensor")
    scaling = node(ref(_args(final), 0), "aten.div.Tensor")
    difference = node(ref(_args(final), 1), "aten.sub.Tensor")
    scale_args = _args(scaling)
    if len(scale_args) != 2 or type(scale_args[1]) not in (int, float):
        raise Refused("dX scale requires scalar width division")
    rstd, width = ref(scale_args, 0), scale_args[1]
    if not math.isfinite(width) or int(width) != width or width <= 0 or width % 4 or width > 4096:
        raise Refused("dX vector4 width must be positive, divisible by four and at most4096")
    width = int(width)
    subtraction = node(ref(_args(difference), 0), "aten.sub.Tensor")
    norm_sum = node(ref(_args(difference), 1), "aten.mul.Tensor")
    scaled_dy = node(ref(_args(subtraction), 0), "aten.mul.Tensor")
    sum1 = node(ref(_args(subtraction), 1), "aten.sum.dim_IntList")
    normalized = node(ref(_args(norm_sum), 0), "aten.mul.Tensor")
    sum2 = node(ref(_args(norm_sum), 1), "aten.sum.dim_IntList")
    mixed = node(ref(_args(sum2), 0), "aten.mul.Tensor")
    weighted = node(ref(_args(scaled_dy), 0), "aten.mul.Tensor")
    center = node(ref(_args(normalized), 0), "aten.sub.Tensor")
    x, mean = ref(_args(center), 0), ref(_args(center), 1)
    dy, gain = ref(_args(weighted), 0), ref(_args(weighted), 1)
    shape = _shape(values[x]); stats_shape = shape[:-1] + (1,)
    if not shape or shape[-1] != width or _shape(values[dy]) != shape:
        raise Refused("dX input/cotangent shapes differ from the normalized width")
    rank, rows = len(shape), math.prod(shape[:-1])
    norm_name, weighted_name = normalized["outputs"][0], weighted["outputs"][0]
    if (_args(scaled_dy) != [{"ref": weighted_name}, width]
            or _args(normalized) != [{"ref": center["outputs"][0]}, {"ref": rstd}]
            or _args(mixed) != [{"ref": weighted_name}, {"ref": norm_name}]
            or not _reduction(sum1, "aten.sum.dim_IntList", weighted_name, rank-1, rank)
            or not _reduction(sum2, "aten.sum.dim_IntList", mixed["outputs"][0], rank-1, rank)):
        raise Refused("dX row statistics do not match the captured LayerNorm equation")
    sources = (x, dy, gain, mean, rstd)
    for name in sources:
        if (values[name]["dtype"] != "float32" or not _contiguous(values[name], aligned=name in (x, dy, gain))):
            raise Refused("dX operands require contiguous FP32 and vector-aligned rows/affine")
    if _shape(values[gain]) != (width,) or any(_shape(values[n]) != stats_shape for n in (mean, rstd)):
        raise Refused("dX affine or saved-statistic shape differs")
    row_values = {scaling["outputs"][0], sum1["outputs"][0], sum2["outputs"][0]}
    made = {op["outputs"][0] for op in selected.values()}
    for op in selected.values():
        if op.get("effect", "pure") != "pure":
            raise Refused("dX members must have pure semantics")
        allowed = {"alpha": 1} if op["target"] == "aten.sub.Tensor" else (
            {"dtype": "float32"} if op["target"] == "aten.sum.dim_IntList" else {})
        if any(k not in allowed or v != allowed[k] for k,v in op.get("kwargs", {}).items()):
            raise Refused("dX member has unsupported keyword semantics")
        name = op["outputs"][0]
        if (values[name]["dtype"] != "float32"
                or _shape(values[name]) != (stats_shape if name in row_values else shape)):
            raise Refused("dX member shape/dtype does not preserve FP32 row arithmetic")
        expected_args = 3 if op["target"] == "aten.sum.dim_IntList" else 2
        if len(_args(op)) != expected_args:
            raise Refused("dX member argument arity differs")
        for dependency in _references((op.get("args"), op.get("kwargs"))):
            if dependency in producers and order[producers[dependency]["name"]] >= order[op["name"]]:
                raise Refused("dX member dependency is not ready")
    if not _contiguous(values[output], aligned=True):
        raise Refused("dX result must have aligned contiguous storage")
    # Preserve every shared/escaping ancestor, then its ancestors. In particular
    # the normalized value remains available to the independent gain VJP.
    escaping = _strings(data.get("outputs", {})) | set(data.get("state_writes", {}))
    escaping |= set(data.get("state_writes", {}).values())
    escaping |= {n for n,v in values.items() if v.get("output_roles")}
    protected = (made - {output}) & escaping
    for op in ops:
        if op["name"] not in selected:
            protected |= (set(op.get("inputs", ())) | set(op.get("reads", ()))
                          | _references((op.get("args"), op.get("kwargs")))) & (made - {output})
    for name, value in values.items():
        if name not in made and value.get("alias_of") in made - {output}:
            protected.add(value["alias_of"])
    while True:
        previous = set(protected)
        for name in previous:
            protected |= _references((_args(producers[name]), producers[name].get("kwargs"))) & made
        if protected == previous:
            break
    private = made - protected
    members = {op["name"] for op in selected.values() if op["outputs"][0] in private}
    anchor = min(members, key=order.get); anchor_index = order[anchor]
    for name in sources:
        if name in producers:
            if order[producers[name]["name"]] >= anchor_index:
                raise Refused("dX replacement operands are not ready at its anchor")
        elif name not in data["inputs"]:
            raise Refused("dX source has no binding")
    for op in ops:
        reads = set(op.get("inputs", ())) | set(op.get("reads", ())) | _references((op.get("args"), op.get("kwargs")))
        if op["name"] not in members and reads & (private - {output}):
            raise Refused("a skipped dX intermediate has an outside reader")
        if op["name"] not in members and anchor_index <= order[op["name"]] <= order[final["name"]]:
            if op.get("effect", "pure") != "pure" or set(op.get("writes", ())) & set(sources):
                raise Refused("an intervening effect can alter dX source operands")
    return LayerNormBackwardPlan(anchor, x, dy, gain, mean, rstd, output, rows, width,
        tuple(op["name"] for op in ops if op["name"] in members),
        tuple(op["outputs"][0] for op in ops if op["name"] in members and op["outputs"][0] != output),
        tuple(op["name"] for op in ops if op["name"] in selected and op["outputs"][0] in protected))


def analyze(ir):
    data = _data(ir)
    selected = data.get("metadata", {}).get(KEY)
    if selected is None:
        return dict(contract=CONTRACT, enabled=False, plans=[], refused=[])
    if selected != CONTRACT:
        raise ValueError("unknown explicit LayerNorm arithmetic contract: " + str(selected))
    producers = {n: op for op in data["operations"] for n in op["outputs"]}
    plans, refused, occupied = [], [], set()
    for op in data["operations"]:
        args = _args(op)
        # Narrow discovery to a direct rstd/scalar times two-subtraction VJP;
        # the full proof above never infers semantics from a tensor name.
        if (op["target"] != "aten.mul.Tensor" or len(args) != 2
                or producers.get(_ref(args[0]), {}).get("target") != "aten.div.Tensor"
                or producers.get(_ref(args[1]), {}).get("target") != "aten.sub.Tensor"):
            continue
        try:
            plan = _match(data, op)
            if occupied.intersection(plan.members):
                raise Refused("dX replacement groups overlap")
        except Refused as error:
            refused.append(dict(output=op["outputs"][0], reason=str(error)))
            continue
        plans.append(plan); occupied.update(plan.members)
    return dict(contract=CONTRACT, enabled=True, plans=plans, refused=refused)


def find_plans(ir):
    return analyze(ir)["plans"]


@dataclass(frozen=True)
class LayerNormAffinePlan:
    anchor: str
    x: str
    dy: str
    mean: str
    rstd: str
    dgain: str
    dbias: str
    rows: int
    width: int
    members: tuple[str, ...]
    intermediates: tuple[str, ...]
    contract: str = AFFINE_CONTRACT

    @property
    def finals(self):
        return (self.dgain, self.dbias)

    def to_dict(self):
        return asdict(self)


def _match_affine(data, dx):
    """Bind original saved LayerNorm statistics to ordinary affine derivatives.

    The dX structural proof identifies the statistics, independently of tensor
    names. Only the two column sums and an unshared gain product are removed.
    The normalized primal remains available to every other original consumer.
    """
    if (dx.rows, dx.width) != (16384, 768):
        raise Refused("affine contraction is measured only for 16384 rows and width768")
    ops, values = data["operations"], data["values"]
    producers = {n: op for op in ops for n in op["outputs"]}
    order = {op["name"]: i for i, op in enumerate(ops)}
    rank = len(_shape(values[dx.x]))
    leading = list(range(rank-1))

    def column_sum(op):
        args = _args(op)
        return (op["target"] == "aten.sum.dim_IntList" and len(args) in (2,3)
                and args[1] == leading and (len(args) == 2 or args[2] is False)
                and len(op["outputs"]) == 1)

    dbias_op = _one([op for op in ops if column_sum(op) and _args(op)[0] == {"ref": dx.dy}],
                   "direct LayerNorm bias derivative")
    candidates = []
    for op in ops:
        if not column_sum(op):
            continue
        product = producers.get(_ref(_args(op)[0]), {})
        if product.get("target") != "aten.mul.Tensor" or len(_args(product)) != 2:
            continue
        if _args(product)[0] != {"ref": dx.dy}:
            continue
        normalized = producers.get(_ref(_args(product)[1]), {})
        if normalized.get("target") != "aten.mul.Tensor" or len(_args(normalized)) != 2:
            continue
        center = producers.get(_ref(_args(normalized)[0]), {})
        if (center.get("target") == "aten.sub.Tensor"
                and _args(center) == [{"ref": dx.x}, {"ref": dx.mean}]
                and _args(normalized)[1] == {"ref": dx.rstd}
                and center.get("kwargs", {}) in ({}, {"alpha": 1})
                and not normalized.get("kwargs", {})
                and center.get("effect", "pure") == normalized.get("effect", "pure") == "pure"):
            candidates.append((op, product))
    gain_op, product = _one(candidates, "saved-statistic LayerNorm gain derivative")
    made = {op["outputs"][0] for op in (gain_op, dbias_op, product)}
    for op in (gain_op, dbias_op, product):
        allowed = {"dtype": "float32"} if column_sum(op) else {}
        if (op.get("effect", "pure") != "pure"
                or any(k not in allowed or v != allowed[k] for k,v in op.get("kwargs", {}).items())):
            raise Refused("affine derivative has unsupported effect or keyword semantics")
        name = op["outputs"][0]
        expected = (dx.width,) if op is not product else _shape(values[dx.x])
        if (values[name]["dtype"] != "float32" or _shape(values[name]) != expected
                or not _contiguous(values[name])):
            raise Refused("affine derivative must preserve contiguous FP32 storage")
    product_name = product["outputs"][0]
    escaping = _strings(data.get("outputs", {})) | set(data.get("state_writes", {}))
    escaping |= set(data.get("state_writes", {}).values())
    escaping |= {n for n,v in values.items() if v.get("output_roles")}
    product_shared = product_name in escaping or any(
        value.get("alias_of") == product_name for value in values.values())
    product_shared |= any(op["name"] != gain_op["name"] and product_name in (
        set(op.get("inputs", ())) | set(op.get("reads", ()))
        | _references((op.get("args"), op.get("kwargs")))) for op in ops)
    selected = {gain_op["name"], dbias_op["name"]}
    if not product_shared:
        selected.add(product["name"])
    anchor = min(selected, key=order.get)
    sources = (dx.x, dx.dy, dx.mean, dx.rstd)
    for name in sources:
        if name in producers:
            if order[producers[name]["name"]] >= order[anchor]:
                raise Refused("affine saved source is unavailable at replacement anchor")
        elif name not in data["inputs"]:
            raise Refused("affine saved source has no input binding")
    end = max(order[name] for name in selected)
    for op in ops[order[anchor]:end+1]:
        if op["name"] not in selected and (op.get("effect", "pure") != "pure"
                or set(op.get("writes", ())) & set(sources)):
            raise Refused("intervening effect may alter affine saved tape")
    return LayerNormAffinePlan(anchor, dx.x, dx.dy, dx.mean, dx.rstd,
        gain_op["outputs"][0], dbias_op["outputs"][0], dx.rows, dx.width,
        tuple(op["name"] for op in ops if op["name"] in selected),
        () if product_shared else (product_name,))


def analyze_affine(ir):
    data = _data(ir)
    selected = data.get("metadata", {}).get(AFFINE_KEY)
    if selected in (None, "legacy"):
        return dict(contract=AFFINE_CONTRACT, enabled=False, plans=[], refused=[])
    if selected != AFFINE_CONTRACT:
        raise ValueError("unknown explicit LayerNorm affine arithmetic contract: " + str(selected))
    producers = {n: op for op in data["operations"] for n in op["outputs"]}
    plans, refused, occupied = [], [], set()
    for op in data["operations"]:
        args = _args(op)
        if (op["target"] != "aten.mul.Tensor" or len(args) != 2
                or producers.get(_ref(args[0]), {}).get("target") != "aten.div.Tensor"
                or producers.get(_ref(args[1]), {}).get("target") != "aten.sub.Tensor"):
            continue
        try:
            plan = _match_affine(data, _match(data, op))
            if occupied.intersection(plan.members):
                raise Refused("affine replacement groups overlap")
        except Refused as error:
            refused.append(dict(output=op["outputs"][0], reason=str(error)))
            continue
        plans.append(plan); occupied.update(plan.members)
    return dict(contract=AFFINE_CONTRACT, enabled=True, plans=plans, refused=refused)


def find_affine_plans(ir):
    return analyze_affine(ir)["plans"]
