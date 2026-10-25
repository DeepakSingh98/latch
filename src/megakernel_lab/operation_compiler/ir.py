"""Operation/dependency IR for the LATCH-OPS research checkpoint.

Unlike the released frontend's recognized-family Program, this represents every
operation of the captured loss, automatically derived gradients, and state update.
There are no model names or architecture dimensions in the compiler contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class CaptureError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass
class Value:
    name: str
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    role: str = "temporary"
    parameter: str | None = None
    alias_of: str | None = None
    storage_offset: int = 0
    producer: str | None = None
    output_roles: tuple[str, ...] = ()

    @property
    def numel(self) -> int:
        result = 1
        for size in self.shape:
            result *= size
        return result


@dataclass
class Operation:
    name: str
    target: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    reduction_axes: tuple[int, ...] = ()
    synchronization: str = "global_phase_boundary"
    effect: str = "pure"


@dataclass
class TrainingIR:
    values: dict[str, Value]
    operations: list[Operation]
    inputs: tuple[str, ...]
    outputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Small metadata receipt; tensor constant bytes live outside the IR."""
        def encode(item):
            if isinstance(item, Ref):
                return {"ref": item.name}
            if isinstance(item, (tuple, list)):
                return [encode(v) for v in item]
            if isinstance(item, dict):
                return {k: encode(v) for k, v in item.items()}
            return item
        result = asdict(self)
        for entry, op in zip(result["operations"], self.operations):
            entry["args"] = encode(op.args)
            entry["kwargs"] = encode(op.kwargs)
        result["format_version"] = 1
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingIR":
        if data.get("format_version") != 1:
            raise CaptureError("IR_VERSION", "unknown operation graph format")
        def decode(item):
            if isinstance(item, dict):
                if set(item) == {"ref"}:
                    return Ref(item["ref"])
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return tuple(decode(v) for v in item)
            return item
        values = {}
        for name, raw in data["values"].items():
            entry = dict(raw)
            for key in ("shape", "strides", "output_roles"):
                entry[key] = tuple(entry[key])
            values[name] = Value(**entry)
        operations = []
        for raw in data["operations"]:
            entry = dict(raw)
            entry["args"] = decode(entry["args"])
            entry["kwargs"] = decode(entry["kwargs"])
            for key in ("inputs", "outputs", "dependencies", "reads", "writes", "reduction_axes"):
                entry[key] = tuple(entry[key])
            operations.append(Operation(**entry))
        return cls(values, operations, tuple(data["inputs"]), data["outputs"], data["metadata"])

    def output_names(self) -> tuple[str, ...]:
        out = [self.outputs["loss"]]
        parameter_names = self.metadata.get("parameter_names", tuple(self.outputs.get("grads", self.outputs["params"])))
        for kind in (("grads",) if "grads" in self.outputs else ()) + ("params",):  # step 15: consumed gradients
            out.extend(self.outputs[kind][name] for name in parameter_names)
        if "optimizer_state" in self.outputs:
            out.extend(self.outputs["optimizer_state"][name]
                       for name in self.metadata["optimizer_state_names"])
        else:
            out.extend(self.outputs["momentum"][name] for name in parameter_names)
        for name in self.metadata.get("state_names", ()):
            out.append(self.outputs["state"][name])
        return tuple(out)

    def lifetimes(self) -> dict[str, tuple[int, int]]:
        """Inclusive phase lifetimes; aliases and returned values stay live.

        Reuse is optional. All persistent state is read as the old version until
        a final commit; no optimizer output may overwrite an old input early.
        """
        indices = {op.name: i for i, op in enumerate(self.operations)}
        bounds = {name: [indices.get(v.producer, -1), indices.get(v.producer, -1)]
                  for name, v in self.values.items()}
        for i, op in enumerate(self.operations):
            for name in op.reads:
                bounds[name][1] = max(bounds[name][1], i)
        end = len(self.operations)
        for name in self.output_names():
            bounds[name][1] = end
        # Aliased views keep their source allocation live through their consumers.
        # JSON object-key order is not graph topology. The operation list keeps
        # the producer order even after a sorted-key JSON round trip.
        for operation in reversed(self.operations):
            for name in operation.outputs:
                parent = self.values[name].alias_of
                if parent is not None:
                    bounds[parent][1] = max(bounds[parent][1], bounds[name][1])
        return {name: tuple(pair) for name, pair in bounds.items()}


@dataclass
class UniformConditionalIR:
    """One uniform runtime choice between two complete differentiable regions.

    Branch inputs have the same positional ABI; their local SSA names may differ.
    The external predicate is last and is never included among derivative targets.
    Only the selected branch's forward/AD/optimizer/state transition executes.
    Branch-local outputs map by explicit parameter/state ordering to stable outer
    outputs, so the runtime never reads a host-side copy of the predicate.
    """
    true_branch: TrainingIR
    false_branch: TrainingIR
    predicate: Value
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> tuple[str, ...]:
        return (*self.true_branch.inputs, self.predicate.name)

    @property
    def values(self) -> dict[str, Value]:
        # Outer names describe only the external ABI, not branch-local temporaries.
        result = {name: self.true_branch.values[name] for name in self.true_branch.inputs}
        result[self.predicate.name] = self.predicate
        true_outputs = self.true_branch.output_names()
        for i, local in enumerate(true_outputs):
            original = self.true_branch.values[local]
            name = f"region_result_{i}"
            result[name] = Value(name, original.shape, original.dtype, _contiguous_strides(original.shape),
                                 role="region_output", output_roles=original.output_roles)
        return result

    def output_names(self) -> tuple[str, ...]:
        return tuple(f"region_result_{i}" for i in range(len(self.true_branch.output_names())))

    @property
    def outputs(self) -> dict[str, Any]:
        names = iter(self.output_names())
        result: dict[str, Any] = {"loss": next(names)}
        for kind in ("grads", "params"):
            result[kind] = {name: next(names) for name in self.metadata["parameter_names"]}
        if "optimizer_state_names" in self.metadata:
            result["optimizer_state"] = {name: next(names) for name in self.metadata["optimizer_state_names"]}
            if "momentum" in self.true_branch.outputs:
                result["momentum"] = result["optimizer_state"]
        else:
            result["momentum"] = {name: next(names) for name in self.metadata["parameter_names"]}
        if "state_names" in self.metadata:
            result["state"] = {name: next(names) for name in self.metadata["state_names"]}
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"format_version": 2, "kind": "uniform_conditional",
                "true_branch": self.true_branch.to_dict(), "false_branch": self.false_branch.to_dict(),
                "predicate": asdict(self.predicate), "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniformConditionalIR":
        if data.get("format_version") != 2 or data.get("kind") != "uniform_conditional":
            raise CaptureError("REGION_VERSION", "unknown uniform conditional format")
        predicate = dict(data["predicate"])
        for key in ("shape", "strides", "output_roles"):
            predicate[key] = tuple(predicate[key])
        return cls(TrainingIR.from_dict(data["true_branch"]), TrainingIR.from_dict(data["false_branch"]),
                   Value(**predicate), data["metadata"])


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))
