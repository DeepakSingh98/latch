"""Opt-in CUDA embedding AD association, proved on the complete training IR.

Ordinary ATen operations remain unchanged. A learned-table primal before the
scalar loss must pair uniquely with an EDB contribution after that loss in the
same parameter's canonical local gradient. Intermediate-table gathers and
direct primal EDB operations receive no arithmetic tag.

Arithmetic source: PyTorch v2.9.1 Embedding.cu / EmbeddingBackwardKernel.cu
(BSD-3-Clause): stable occurrences, ten-row FP32 partials, ascending partial sum.
The small-index CUDA path (<=3072 indices) is deliberately outside this contract.
"""
from __future__ import annotations

from copy import deepcopy
import math

KEY = "embedding_backward_contract"
CONTRACT = "torch_cuda_embedding_segment10_v1"
FORWARD = "aten.embedding.default"
BACKWARD = "aten.embedding_dense_backward.default"
ALIASES = {"aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
           "aten.clone.default", "aten.detach.default", "aten.alias.default",
           "aten.contiguous.default", "aten._to_copy.default"}


def _ref(value):
    return value.get("ref") if isinstance(value, dict) and set(value) == {"ref"} else None


def _data(ir):
    return ir if isinstance(ir, dict) else ir.to_dict()


def _flags(op, start, defaults):
    args, kwargs = op["args"], op.get("kwargs", {})
    if len(args) > start + len(defaults) or set(kwargs) - {name for name, _ in defaults}:
        raise ValueError("embedding arithmetic has an unsupported operator ABI")
    result = []
    for index, (name, default) in enumerate(defaults, start):
        if len(args) > index and name in kwargs:
            raise ValueError("embedding argument supplied twice")
        result.append(args[index] if len(args) > index else kwargs.get(name, default))
    return result


def plan(ir, contract=CONTRACT):
    """Return exact operation tags; refuse absent or ambiguous eligible VJPs."""
    if contract != CONTRACT:
        raise ValueError("unknown embedding backward arithmetic contract")
    graph = _data(ir)
    values, operations = graph["values"], graph["operations"]
    metadata = graph.get("metadata", {})
    if metadata.get("capture") != "torch.func.grad_and_value + make_fx(fake)":
        raise ValueError("embedding VJP binding requires ordinary complete training capture")
    producers = {name: op for op in operations for name in op["outputs"]}
    positions = {name: i for i, op in enumerate(operations) for name in op["outputs"]}
    loss = graph.get("outputs", {}).get("loss")
    if (loss not in producers or not producers[loss]["target"].startswith("aten.")
            or values[loss]["shape"] not in ([], ()) or values[loss]["dtype"] != "float32"):
        raise ValueError("embedding VJP binding requires an ordinary scalar-loss producer")
    loss_position = positions[loss]
    # Reuse the complete canonical cat/SUM/public-slice ownership proof. The
    # byte cap is only a bounded metadata-plan argument, not a transport change.
    from .gradient_buckets import plan as gradient_plan
    ownership = gradient_plan(graph, 25 << 20)
    if metadata.get("gradient_transform", {}).get("canonical_sum") != dict(
            contract="canonical_cat_sum_views_v1", world_size=ownership["world_size"],
            local_gradients="unmodified AD outputs"):
        raise ValueError("embedding VJP binding requires captured canonical unmodified AD-gradient transform provenance")
    parameters = {v["parameter"]: name for name, v in values.items() if v.get("role") == "parameter"}

    def alias_source(name):
        op = producers.get(name)
        if op is None or op["target"] not in ALIASES or len(op.get("inputs", ())) != 1:
            return None
        source = _ref(op["args"][0])
        if (source is None or values[source]["dtype"] != "float32" or values[name]["dtype"] != "float32"
                or math.prod(values[source]["shape"]) != math.prod(values[name]["shape"])):
            return None
        return source

    def parameter_root(name):
        seen = set()
        while name not in seen:
            if values[name].get("role") == "parameter":
                return name
            seen.add(name)
            source = alias_source(name)
            if source is None or values[source]["shape"] != values[name]["shape"]:
                return None
            name = source
        return None

    forwards = []
    for op in operations:
        if op["target"] != FORWARD or len(op["outputs"]) != 1 or len(op["args"]) < 2:
            continue
        table, indices = map(_ref, op["args"][:2])
        weight = parameter_root(table) if table in values else None
        if weight is None or indices not in values:
            continue
        padding, frequency, sparse = _flags(op, 2, (("padding_idx", -1), ("scale_grad_by_freq", False), ("sparse", False)))
        forwards.append((op, weight, indices, padding, frequency, sparse))

    def contributions(name, shape):
        found, seen, pending = set(), set(), [name]
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            source = alias_source(name)
            if source is not None:
                pending.append(source)
                continue
            op = producers.get(name)
            if op is None or values[name]["dtype"] != "float32" or list(values[name]["shape"]) != shape:
                continue
            if op["target"] == BACKWARD:
                found.add(name)
            elif (op["target"] == "aten.add.Tensor" and len(op["args"]) == 2
                  and op.get("kwargs", {}) in ({}, {"alpha": 1})):
                refs = [_ref(arg) for arg in op["args"]]
                # widen_cast may elide a private BF16->FP32 copy feeding this
                # FP32-result add. Reading BF16 in that add widens exactly;
                # the other FP32 EDB contribution still enters with scale one.
                # Follow only FP32 branches below: a narrowed EDB is never a
                # proof of the original FP32 learned-table contribution.
                if all(ref in values and list(values[ref]["shape"]) == shape
                       and values[ref]["dtype"] in ("float32", "bfloat16") for ref in refs):
                    pending.extend(ref for ref in refs if values[ref]["dtype"] == "float32")
        return found

    records, paired_forwards = {}, set()
    for owner in ownership["parameters"]:
        weight = parameters[owner["name"]]
        shape = list(values[weight]["shape"])
        if len(shape) != 2 or values[weight]["dtype"] != "float32":
            continue
        for output in sorted(contributions(owner["input"], shape)):
            if positions[output] <= loss_position:
                continue  # A direct primal EDB is not a learned-table VJP.
            op = producers[output]
            if len(op["args"]) < 3 or len(op["outputs"]) != 1:
                raise ValueError("embedding VJP ABI differs")
            gradient, indices = map(_ref, op["args"][:2])
            rows = op["args"][2]
            padding, frequency = _flags(op, 3, (("padding_idx", -1), ("scale_grad_by_freq", False)))
            if gradient not in values or indices not in values:
                raise ValueError("embedding VJP tensor operands are missing")
            index_count = math.prod(values[indices]["shape"])
            if index_count <= 3072:
                continue
            matches = [entry for entry in forwards if entry[1:5] == (weight, indices, padding, frequency)]
            if len(matches) != 1:
                raise ValueError("embedding VJP lacks one unique learned-table forward")
            forward, _, _, _, _, sparse = matches[0]
            forward_output = forward["outputs"][0]
            grad_shape = [*values[indices]["shape"], shape[1]]
            if (positions[forward_output] >= loss_position or sparse is not False or frequency is not False
                    or type(rows) is not int or rows != shape[0] or type(padding) is not int
                    or not -1 <= padding < rows or values[indices]["dtype"] != "int64"
                    or values[gradient]["dtype"] != "float32"
                    or list(values[gradient]["shape"]) != grad_shape
                    or list(values[forward_output]["shape"]) != grad_shape
                    or values[forward_output]["dtype"] != "float32"):
                raise ValueError("embedding VJP does not preserve the admitted primal/derivative ABI")
            if output in records:
                raise ValueError("embedding VJP contributes to more than one parameter gradient")
            if forward_output in paired_forwards:
                raise ValueError("learned-table forward has more than one candidate embedding VJP")
            paired_forwards.add(forward_output)
            records[output] = dict(contract=CONTRACT, operation=op["name"], output=output,
                forward_operation=forward["name"], forward_output=forward_output,
                weight=weight, parameter=owner["name"], local_gradient=owner["input"],
                gradient=gradient, indices=indices, indices_shape=list(values[indices]["shape"]),
                gradient_shape=grad_shape, rows=rows, width=shape[1], indices_count=index_count,
                padding_idx=padding, scale_grad_by_freq=False, dtype="float32", indices_dtype="int64",
                partial_rows=10, accumulation="ascending ten-occurrence FP32 partials then ascending FP32 sum",
                provenance="unique learned-table primal before scalar loss; VJP after loss in canonical parameter gradient")
    if not records:
        raise ValueError("embedding contract has no uniquely proved large-index learned-table VJP")
    return dict(contract=CONTRACT, operations=records)


def bind(ir, contract=None):
    existing = ir.metadata.get(KEY)
    if contract is None and existing is None:
        return ir, None
    contract = existing.get("contract") if contract is None and isinstance(existing, dict) else contract
    record = plan(ir, contract)
    if existing is not None and existing != record:
        raise ValueError("existing embedding arithmetic differs from complete training IR proof")
    result = deepcopy(ir)
    result.metadata[KEY] = record
    return result, record


def region_record(metadata, outputs):
    record = metadata.get(KEY)
    if record is None:
        return None
    if record.get("contract") != CONTRACT or not isinstance(record.get("operations"), dict):
        raise ValueError("malformed embedding arithmetic record")
    outputs = set(outputs)
    selected = {name: deepcopy(value) for name, value in record["operations"].items() if name in outputs}
    return dict(contract=CONTRACT, operations=selected) if selected else None


def validate_local(ir):
    """Recheck every tagged local EDB after graph rewrites/region partition."""
    metadata = ir.get("metadata", {}) if isinstance(ir, dict) else getattr(ir, "metadata", {})
    if not metadata or KEY not in metadata:
        return {}
    graph = _data(ir)
    record = graph.get("metadata", {}).get(KEY)
    if record is None:
        return {}
    if record.get("contract") != CONTRACT or not record.get("operations"):
        raise ValueError("malformed local embedding arithmetic record")
    producers = {name: op for op in graph["operations"] for name in op["outputs"]}
    values = graph["values"]
    for name, proof in record["operations"].items():
        op = producers.get(name, {})
        args = op.get("args", ())
        if (proof.get("contract") != CONTRACT or proof.get("output") != name or proof.get("partial_rows") != 10
                or op.get("target") != BACKWARD or op.get("outputs") not in ([name], (name,))
                or list(args[:3]) != [{"ref": proof.get("gradient")}, {"ref": proof.get("indices")}, proof.get("rows")]
                or _flags(op, 3, (("padding_idx", -1), ("scale_grad_by_freq", False))) != [proof.get("padding_idx"), False]
                or proof.get("dtype") != "float32" or proof.get("indices_dtype") != "int64"
                or proof.get("indices_count", 0) <= 3072 or proof.get("scale_grad_by_freq") is not False):
            raise ValueError("local embedding arithmetic differs from bound EDB")
        for tensor, shape, dtype in ((name, [proof["rows"], proof["width"]], "float32"),
                (proof["gradient"], proof["gradient_shape"], "float32"),
                (proof["indices"], proof["indices_shape"], "int64")):
            if (tensor not in values or list(values[tensor]["shape"]) != shape or values[tensor]["dtype"] != dtype):
                raise ValueError("local embedding arithmetic tensor ABI differs")
    return record["operations"]


def phase_signature(record):
    if record is None:
        return None
    if record.get("contract") != CONTRACT or not record.get("operations"):
        raise ValueError("malformed embedding arithmetic phase signature")
    keys = ("parameter", "rows", "width", "indices_shape", "indices_count", "padding_idx",
            "scale_grad_by_freq", "dtype", "indices_dtype", "partial_rows", "accumulation")
    return dict(contract=CONTRACT, parameters=sorted(
        ({key: value[key] for key in keys} for value in record["operations"].values()),
        key=lambda value: value["parameter"]))


def receipt_record(receipt):
    record = receipt.get(KEY)
    if receipt.get("metadata", {}).get(KEY) != record:
        raise ValueError("embedding arithmetic artifact and metadata disagree")
    if record is not None:
        phase_signature(record)
    return record


def validate_artifact(receipt, graph):
    record = receipt_record(receipt)
    if graph.get("metadata", {}).get(KEY) != record:
        raise ValueError("embedding arithmetic was lost from retained training IR")
    if record is not None and plan(graph, record["contract"]) != record:
        raise ValueError("embedding arithmetic differs from complete training IR proof")
    return record
