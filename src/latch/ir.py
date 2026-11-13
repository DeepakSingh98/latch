"""The training-step graph: one ATen operation per node, static shapes, no control flow.

A Graph holds everything one update computes: the forward pass, the loss, the
gradients PyTorch derived, and the optimizer arithmetic. The compiler never sees
a model class or a layer name, only these operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class CompileError(RuntimeError):
    """The program uses something the compiler has no rule for."""


@dataclass(frozen=True)
class Ref:
    """A reference to another value, as an operation argument."""
    name: str


@dataclass
class Value:
    name: str
    shape: tuple[int, ...]
    dtype: str  # "float32", "int64" or "bool"

    @property
    def numel(self) -> int:
        n = 1
        for extent in self.shape:
            n *= extent
        return n


@dataclass
class Op:
    """``name = target(*args, **kwargs)``; the result value has the same name."""
    name: str
    target: str  # for example "aten.mm.default"
    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def refs(self) -> list[str]:
        return list(_refs(self.args)) + list(_refs(self.kwargs))


@dataclass
class Graph:
    values: dict[str, Value]
    ops: list[Op]
    inputs: list[str]             # bound tensors, in argument-table order
    roles: dict[str, list]        # input name -> [kind, key]; kind is parameter/optimizer/input/target/constant
    loss: str                     # scalar float32 value
    updates: dict[str, str]       # persistent input name -> value that replaces it after the step

    def to_dict(self) -> dict:
        data = asdict(self)
        for entry, op in zip(data["ops"], self.ops):
            entry["args"], entry["kwargs"] = _encode(op.args), _encode(op.kwargs)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        values = {name: Value(v["name"], tuple(v["shape"]), v["dtype"]) for name, v in data["values"].items()}
        ops = [Op(o["name"], o["target"], _decode(o["args"]), _decode(o["kwargs"])) for o in data["ops"]]
        return cls(values, ops, list(data["inputs"]), dict(data["roles"]), data["loss"], dict(data["updates"]))


def _refs(item):
    if isinstance(item, Ref):
        yield item.name
    elif isinstance(item, (tuple, list)):
        for x in item:
            yield from _refs(x)
    elif isinstance(item, dict):
        for x in item.values():
            yield from _refs(x)


def _encode(item):
    if isinstance(item, Ref):
        return {"ref": item.name}
    if isinstance(item, (tuple, list)):
        return [_encode(x) for x in item]
    if isinstance(item, dict):
        return {k: _encode(v) for k, v in item.items()}
    if isinstance(item, float) and item != item:
        return {"float": "nan"}
    if isinstance(item, float) and item in (float("inf"), float("-inf")):
        return {"float": "inf" if item > 0 else "-inf"}
    return item


def _decode(item):
    if isinstance(item, dict):
        if set(item) == {"ref"}:
            return Ref(item["ref"])
        if set(item) == {"float"}:
            return float(item["float"])
        return {k: _decode(v) for k, v in item.items()}
    if isinstance(item, list):
        return tuple(_decode(x) for x in item)
    return item
