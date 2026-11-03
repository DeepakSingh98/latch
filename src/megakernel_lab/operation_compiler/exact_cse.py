"""Exact common-subexpression elimination on the captured training IR (graph_rules ``cse``).

Captured programs can repeat the same pure operation on the same operands -- for example a statistic
``rsqrt(mean(s * s) + eps)`` or a BF16 operand copy ``s.to(bf16)`` recomputed by every later reader of ``s`` -- each
stored separately and each a separate latency-bound phase. This pass keeps the first operation of each identical
(target, operands, attributes, result metadata) class and redirects later readers to its results.

Exactness: a pure operation's results are a function of its target, attributes and operand values; Latch lowers every
operation with its own canonical expression and reduction order (fusion and scheduling never change an operation's
values), so the kept results are bit-identical to the removed ones. The IR is in SSA form -- persistent state keeps
its old version until the final commit -- so equal operand names are equal values. Backward operations of the
separate uses stay separate (their gradient operands differ); only identical operations merge. Operand order is
significant: ``mul(a, b)`` and ``mul(b, a)`` are different classes.

Refused (kept as they are):

* operations whose target is not an ``aten.*`` operator (opaque ``latch_ops.*`` calls such as composed attention
  tapes are lowered by pattern-specific engines that expect each call);
* operations that are not ``effect == "pure"``, whose writes are not exactly their own fresh results, or whose
  target is random/stateful (``philox``/``rand``/``dropout``/``bernoulli``/``normal``/``uniform``/``noise``);
* operations with a result that is returned, committed, a parameter/state/input or otherwise not a temporary;
* results whose metadata (shape, dtype, strides, storage offset, alias parent) differs from the kept class.

``apply`` always runs; the caller (``prepare_module``) applies it only when ``graph_rules`` selects ``cse``.
"""
from __future__ import annotations

from dataclasses import replace
import re

VERSION = 1
_RANDOM = re.compile(r"philox|rand|dropout|bernoulli|normal|uniform|noise|seed", re.I)


def _flatten_names(item, out):
    if isinstance(item, str):
        out.add(item)
    elif isinstance(item, dict):
        for v in item.values():
            _flatten_names(v, out)
    elif isinstance(item, (list, tuple, set)):
        for v in item:
            _flatten_names(v, out)
    elif hasattr(item, "name") and isinstance(getattr(item, "name"), str):
        out.add(item.name)


def _canon(item, rename, Ref):
    if isinstance(item, Ref):
        return ("ref", rename.get(item.name, item.name))
    if isinstance(item, (list, tuple)):
        return (type(item).__name__, tuple(_canon(v, rename, Ref) for v in item))
    if isinstance(item, dict):
        return ("dict", tuple(sorted((k, _canon(v, rename, Ref)) for k, v in item.items())))
    if isinstance(item, float):
        return ("float", item.hex() if item == item else "nan")
    return (type(item).__name__, repr(item))


def _remap(item, rename, Ref):
    if isinstance(item, Ref):
        return Ref(rename[item.name]) if item.name in rename else item
    if isinstance(item, tuple):
        return tuple(_remap(v, rename, Ref) for v in item)
    if isinstance(item, list):
        return [_remap(v, rename, Ref) for v in item]
    if isinstance(item, dict):
        return {k: _remap(v, rename, Ref) for k, v in item.items()}
    return item


def apply(graph):
    """Return (graph, report). The graph is unchanged (same object) when nothing merges; a graph without
    ``operations``/``values`` is returned as is with report None."""
    if not hasattr(graph, "operations") or not hasattr(graph, "values"):
        return graph, None
    from .ir import Ref
    values = graph.values
    protected = set()
    _flatten_names(graph.outputs, protected)
    try:
        protected |= set(graph.output_names())
    except Exception:   # noqa: BLE001 - an IR without the training output layout: only explicit outputs count
        pass
    protected |= set(getattr(graph, "inputs", ()) or ())
    rename, op_rename, seen, removed = {}, {}, {}, set()
    by_target = {}
    for op in graph.operations:
        outs = tuple(op.outputs)
        ok = (op.effect == "pure" and outs and op.target.startswith("aten.") and not _RANDOM.search(op.target)
              and set(op.writes) <= set(outs)
              and all(o in values and o not in protected and values[o].role == "temporary"
                      and not values[o].output_roles for o in outs))
        if not ok:
            continue
        meta = tuple((tuple(values[o].shape), values[o].dtype, tuple(values[o].strides), values[o].storage_offset,
                      rename.get(values[o].alias_of, values[o].alias_of)) for o in outs)
        args_key = _canon(op.args, rename, Ref)
        key = (op.target, args_key, _canon(op.kwargs, rename, Ref), meta,
               tuple(op.reduction_axes), op.synchronization, len(op.writes))
        first = seen.get(key)
        if first is None:
            seen[key] = op
            continue
        for a, b in zip(outs, first.outputs):
            rename[a] = b
        op_rename[op.name] = first.name
        removed.add(op.name)
        by_target[op.target] = by_target.get(op.target, 0) + 1
    if not removed:
        return graph, dict(merged=0)
    operations = []
    for op in graph.operations:
        if op.name in removed:
            continue
        if any(r in rename for r in op.reads) or any(d in op_rename for d in op.dependencies):
            op = replace(op, args=_remap(op.args, rename, Ref), kwargs=_remap(op.kwargs, rename, Ref),
                         inputs=tuple(dict.fromkeys(rename.get(x, x) for x in op.inputs)),
                         reads=tuple(dict.fromkeys(rename.get(x, x) for x in op.reads)),
                         dependencies=tuple(dict.fromkeys(op_rename.get(d, d) for d in op.dependencies)))
        operations.append(op)
    new_values = {}
    for name, v in values.items():
        if name in rename:
            continue
        if v.alias_of in rename:
            v = replace(v, alias_of=rename[v.alias_of])
        new_values[name] = v
    leftover = {r for op in operations for r in op.reads if r in rename}
    if leftover:
        raise RuntimeError(f"exact_cse: unresolved reads {sorted(leftover)[:5]}")
    report = dict(merged=len(removed), values_removed=len(rename),
                  by_target=dict(sorted(by_target.items(), key=lambda kv: -kv[1])))
    return replace(graph, values=new_values, operations=operations), report
