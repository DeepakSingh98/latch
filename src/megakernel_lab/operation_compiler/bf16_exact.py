"""LATCH-DENSE-EXEC: reuse of an existing BF16 copy of a BF16-exact FP32 value (exact graph rewrite before P6D lowering).

An FP32 value v is BF16-exact when it is an exact widening of a BF16 value, or relu/neg/abs of a BF16-exact value of
the same shape (each maps BF16-representable inputs to BF16-representable outputs exactly). If the graph already
narrows v to n = bf16(v) and n is materialized for a matrix product, then widen(n) == v bit for bit, so the other
readers of v may read widen(n) instead. This turns, e.g., the backward ReLU mask threshold_backward(g, relu(...)) into
a read of the stored BF16 activation instead of a recomputation from the FP32 pre-activation product; the FP32
product then has no reader outside its epilogue. Values, VJPs and rounding points are unchanged; protected values
(graph outputs, inputs, state) are never rewritten.
"""
from __future__ import annotations

from dataclasses import replace

BF16_EXACT_REUSE = True
_PRESERVING = {"relu", "neg", "abs"}
_VIEWS = {"view", "_unsafe_view", "reshape", "t", "transpose", "permute", "squeeze", "unsqueeze", "expand", "alias",
          "detach", "slice", "select", "as_strided"}
_PRODUCTS = {"mm", "bmm", "matmul"}


def _base(op):
    return op.target.removeprefix("aten.").split(".")[0]


def _contiguous(shape):
    strides, acc = [], 1
    for d in reversed(shape):
        strides.insert(0, acc)
        acc *= d
    return tuple(strides)


def reuse(graph):
    """Returns (graph, report). Applies only to TrainingIR with dict outputs (the P6D CSE contract)."""
    from .ir import Ref, TrainingIR, Value, Operation
    from .remat import _names, _rewrite
    report = {"rewritten": []}
    if not BF16_EXACT_REUSE or not isinstance(graph, TrainingIR) or not isinstance(graph.outputs, dict):
        return graph, report
    values = graph.values
    protected = _names(graph.outputs, set(graph.inputs))
    _names(getattr(graph, "state_writes", None) or {}, protected)

    def single(op):
        return len(op.outputs) == 1 and op.effect == "pure" and tuple(op.writes) in ((), tuple(op.outputs))

    def source(op):
        return op.args[0].name if op.args and isinstance(op.args[0], Ref) else None

    def plain_cast(op):
        return single(op) and _base(op) == "_to_copy" and set(op.kwargs) <= {"dtype"} and source(op) in values

    exact, widen_kwargs, narrow_of, readers = set(), None, {}, {}
    for op in graph.operations:
        for name in _names((op.args, op.kwargs), set()):
            if name in values:
                readers.setdefault(name, []).append(op)
        if not single(op):
            continue
        out, src = op.outputs[0], source(op)
        if src is None or src not in values or values[src].shape != values[out].shape:
            continue
        if plain_cast(op) and values[out].dtype == "float32" and values[src].dtype == "bfloat16":
            exact.add(out)
            widen_kwargs = widen_kwargs or dict(op.kwargs)
        elif _base(op) in _PRESERVING and values[out].dtype == "float32" and src in exact and len(op.args) == 1:
            exact.add(out)
        elif plain_cast(op) and values[out].dtype == "bfloat16" and src in exact:
            narrow_of.setdefault(src, op)
    if widen_kwargs is None:
        return graph, report
    mapping, inserts = {}, {}
    for value, narrow in narrow_of.items():
        n = narrow.outputs[0]
        if value in protected or n in protected or values[n].strides != _contiguous(values[n].shape):
            continue
        if not any(_base(r) in _PRODUCTS for r in readers.get(n, ())):
            continue        # the BF16 copy is not materialized for a product; a widened read could add storage
        others = [r for r in readers.get(value, ()) if r is not narrow and _base(r) not in _VIEWS]
        if not others:
            continue
        name = "bf16x_" + n
        if name in values:
            continue
        values = dict(values) if values is graph.values else values
        values[name] = replace(values[value], name=name, role="temporary", parameter=None, alias_of=None,
                               storage_offset=0, producer=name, output_roles=(), strides=_contiguous(values[value].shape))
        inserts[narrow.name] = Operation(name=name, target="aten._to_copy.default", args=(Ref(n),),
                                         kwargs=dict(widen_kwargs), inputs=(n,), outputs=(name,),
                                         dependencies=(narrow.name,), reads=(n,), writes=())
        for r in others:
            mapping.setdefault(r.name, {})[value] = name
        report["rewritten"].append({"value": value, "bf16_copy": n, "widened": name,
                                    "readers": sorted(r.name for r in others)})
    if not report["rewritten"]:
        return graph, report
    operations = []
    for op in graph.operations:
        if op.name in mapping:
            m = mapping[op.name]
            op = replace(op, args=_rewrite(op.args, m), kwargs=_rewrite(op.kwargs, m),
                         inputs=tuple(m.get(x, x) for x in op.inputs), reads=tuple(m.get(x, x) for x in op.reads),
                         dependencies=tuple(dict.fromkeys(tuple(op.dependencies) + tuple(m.values()))))
        operations.append(op)
        if op.name in inserts:
            operations.append(inserts[op.name])
    metadata = dict(graph.metadata)
    metadata["bf16_exact_reuse"] = {"rewritten": len(report["rewritten"])}
    return replace(graph, values=values, operations=operations, metadata=metadata), report
