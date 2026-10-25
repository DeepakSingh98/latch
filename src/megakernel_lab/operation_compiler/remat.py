"""Late-operand rematerialization for flat training graphs (C-OPS2-D).

A pure per-element value that is read early (forward) and again much later
(typically by a weight-gradient product) would otherwise occupy arena storage
across that whole interval. When every input it needs is already live at the
late use (or is a bound input), an identical copy of the producing operation is
emitted just before the first late consumer, and late consumers read the copy.
Short producer chains are copied together when their own inputs are live.

This changes where an operation runs, never what it computes: the copy applies
the same ATen operation to the same stored inputs, so results are bitwise equal
(the same exactness argument as the saved-tape rule in ops/loop.py, P-OPS-E).
The fusion scheduler receives an anchor per copy so it is not hoisted back into
the forward pass. No model names, shapes or derivative patterns are recognized.
"""
from __future__ import annotations

from dataclasses import replace

from .ir import Ref, TrainingIR

# Same exact per-element set as ops/loop.py, plus dtype-only casts (deterministic rounding).
RECOMPUTABLE = {"add", "sub", "mul", "div", "pow", "neg", "tanh", "exp", "log", "sqrt", "rsqrt", "abs",
                "sigmoid", "silu", "silu_backward", "tanh_backward", "sigmoid_backward", "relu",
                "threshold_backward", "_to_copy"}
VIEWS = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand",
         "squeeze", "unsqueeze"}
MIN_ELEMENTS = 4096
MAX_CHAIN = 3


def _base(op):
    return op.target.removeprefix("aten.").split(".")[0]


def _rewrite(item, mapping):
    if isinstance(item, Ref):
        return Ref(mapping.get(item.name, item.name))
    if isinstance(item, tuple):
        return tuple(_rewrite(x, mapping) for x in item)
    if isinstance(item, list):
        return [_rewrite(x, mapping) for x in item]
    if isinstance(item, dict):
        return {k: _rewrite(v, mapping) for k, v in item.items()}
    return item


def _names(item, into):
    if isinstance(item, Ref):
        into.add(item.name)
    elif isinstance(item, str):
        into.add(item)
    elif isinstance(item, (list, tuple)):
        for x in item:
            _names(x, into)
    elif isinstance(item, dict):
        for x in item.values():
            _names(x, into)
    return into


def plan(graph: TrainingIR, *, gap=None, admit=None, recomputable_ops=None, max_chain=None, dtypes=None,
         view_links=False, extend_inputs=False, require_anchor=True, op_dtypes=None):
    """Return (new_graph, report); the graph is returned unchanged when nothing qualifies.

    ``admit(graph, copied_source_operations, late_consumers)`` optionally refuses a candidate group (release step 15).
    ``recomputable_ops`` / ``max_chain`` / ``dtypes`` widen the exact per-element set, chain depth and copied result
    dtypes; ``view_links`` lets a copied chain pass through view operations (aliases, no arithmetic) of values that
    are not live late (LATCH-TRAIN-MEMORY ops.memory_rules); omitted, the established defaults apply unchanged.
    ``extend_inputs`` (min-cut swap): a chain input that is neither live late nor recomputable may instead be kept
    alive until the copy when the bytes newly kept alive are strictly fewer than the bytes of the value no longer
    kept; later candidates read earlier copies (placed no later than their own) instead of the originals.
    ``require_anchor=False`` admits late consumers without another produced operand (no fusion-scheduler anchor is
    needed by compose, where ops.memory_rules' lifetime sink places copies). ``op_dtypes`` restricts the result dtypes
    of particular operations (e.g. matrix products recomputed only when their result is FP32)."""
    recomputable_ops = RECOMPUTABLE if recomputable_ops is None else recomputable_ops
    max_chain = MAX_CHAIN if max_chain is None else max_chain
    dtypes = ("float32", "bfloat16") if dtypes is None else dtypes
    report = {"applied": 0, "copied_operations": 0, "removed_dead_views": 0, "candidates": {}}
    if not isinstance(graph, TrainingIR):
        return graph, {**report, "reason": "only captured TrainingIR graphs are rewritten"}
    ops = list(graph.operations)
    if not ops or not isinstance(graph.outputs, dict):
        return graph, {**report, "reason": "not a structured flat training graph"}
    gap = max(32, len(ops) // 16) if gap is None else gap
    report["gap_operations"] = gap
    order = {op.name: i for i, op in enumerate(ops)}
    by_name = {op.name: op for op in ops}
    producer = {op.outputs[0]: op for op in ops if len(op.outputs) == 1}
    readers = {}
    for op in ops:
        for name in dict.fromkeys(op.reads):
            readers.setdefault(name, []).append(op)
    protected = _names(graph.outputs, set(graph.inputs))
    _names(getattr(graph, "state_writes", None) or {}, protected)
    used_names = set(graph.values)

    def fresh(name):
        k = 1
        while f"{name}__remat{k}" in used_names:
            k += 1
        used_names.add(f"{name}__remat{k}")
        return f"{name}__remat{k}"

    def is_view(op):
        return _base(op) in VIEWS and len(op.outputs) == 1

    def recomputable(op):
        if op.effect != "pure" or len(op.outputs) != 1 or _base(op) not in recomputable_ops:
            return False
        if _base(op) == "_to_copy" and set(op.kwargs) - {"dtype"}:
            return False
        value = graph.values[op.outputs[0]]
        allowed = (op_dtypes or {}).get(_base(op), dtypes)
        return value.alias_of is None and value.dtype in dtypes and value.dtype in allowed

    redirected = {}        # extend_inputs: original value -> late readers already redirected to a copy
    copy_reads = {}        # extend_inputs: original value -> positions of accepted copies that read it

    def live_at(name, position, ignoring=()):
        if name in graph.inputs:
            return True
        gone = redirected.get(name, ())
        return (any(order[op.name] >= position for op in readers.get(name, ())
                    if op.name not in ignoring and op.name not in gone)
                or any(q >= position for q in copy_reads.get(name, ())))

    taken = set()          # values already copied (each value at most once)
    extended = set()       # values kept alive for late copies (extend_inputs)
    late_copy = {}         # original value -> (copy name, position) of an earlier accepted group

    def value_bytes(name):
        value = graph.values[name]
        return value.numel * {"bfloat16": 2, "float16": 2, "bool": 1, "int64": 8, "float64": 8}.get(value.dtype, 4)
    groups = []            # (position, [(source_op, new_name, mapping)], late terminal names, mapping)
    anchors = {}
    for op in ops:
        if len(op.outputs) != 1:
            continue
        v = op.outputs[0]
        if v in protected or v in taken or not recomputable(op) or graph.values[v].numel < MIN_ELEMENTS:
            continue
        # Views of v (transitively) and the non-view terminals reached through them.
        views, terminals, stack = [], [], [v]
        while stack:
            current = stack.pop()
            for reader in readers.get(current, ()):
                if is_view(reader):
                    views.append(reader)
                    stack.append(reader.outputs[0])
                else:
                    terminals.append(reader)
        positions = sorted({order[t.name] for t in terminals})
        if len(positions) < 2:
            continue
        size, split = max((positions[i + 1] - positions[i], i) for i in range(len(positions) - 1))
        if size < gap:
            continue
        first_late = positions[split + 1]
        late = [t for t in terminals if order[t.name] >= first_late]
        late_names = {t.name for t in late}
        chain = {v} | {u.outputs[0] for u in views}
        consumer = ops[first_late]
        external = [r for r in consumer.reads if r not in chain and r in producer]
        if not external and require_anchor:
            continue
        anchor = max(external, key=lambda r: order[producer[r].name]) if external else None
        # Inputs of v (and of any copied producer) must be live at first_late,
        # not counting the late reads being redirected.
        new_ops, mapping, ok = [], {}, True
        keep = {}

        def late_version(name, depth):
            if name in mapping:
                return mapping[name]
            if extend_inputs and name in late_copy and late_copy[name][1] <= first_late:
                return late_copy[name][0]
            if live_at(name, first_late) or name in extended:
                return name
            source = producer.get(name)
            if source is None or name in protected or (name in taken and not extend_inputs):
                return None
            if depth >= max_chain or not (recomputable(source) or (view_links and is_view(source))):
                if extend_inputs and graph.values[name].alias_of is None and source.effect == "pure":
                    keep[name] = value_bytes(name)
                    return name
                return None
            local = {}
            for read in dict.fromkeys(source.reads):
                late_read = late_version(read, depth + 1)
                if late_read is None:
                    return None
                local[read] = late_read
            mapping[name] = fresh(name)
            new_ops.append((source, mapping[name], local))
            return mapping[name]

        local = {}
        for read in dict.fromkeys(op.reads):
            late_read = late_version(read, 1)
            if late_read is None:
                ok = False
                break
            local[read] = late_read
        if not ok:
            continue
        if keep and sum(keep.values()) >= value_bytes(v):
            report.setdefault("refused_by_bytes", 0)
            report["refused_by_bytes"] += 1
            continue
        mapping[v] = fresh(v)
        new_ops.append((op, mapping[v], local))
        # Copy every view on a path from v to a late terminal (views are aliases, free).
        pending = [v]
        while pending:
            current = pending.pop(0)
            for reader in readers.get(current, ()):
                if is_view(reader) and _reaches(reader, late_names, readers, is_view):
                    if reader.outputs[0] not in mapping:
                        mapping[reader.outputs[0]] = fresh(reader.outputs[0])
                        new_ops.append((reader, mapping[reader.outputs[0]], mapping))
                        pending.append(reader.outputs[0])
        if admit is not None and not admit(graph, [source for source, _, _ in new_ops], late):
            report.setdefault("refused_by_policy", []).append(v)
            continue
        for _, name, _ in new_ops:
            if anchor is not None:
                anchors[name] = anchor
        taken.update(mapping)
        extended.update(keep)
        if extend_inputs:
            for original, copy in mapping.items():
                late_copy.setdefault(original, (copy, first_late))
                redirected.setdefault(original, set()).update(
                    t for t in late_names if original in by_name[t].reads)
            for source, _, local in new_ops:
                for read in source.reads:
                    target = local.get(read, read)
                    if target == read:                  # this copy reads the original value
                        copy_reads.setdefault(read, []).append(first_late)
        groups.append((first_late, new_ops, late_names, dict(mapping)))
        value = graph.values[v]
        report["candidates"][v] = {"copy": mapping[v], "first_late_operation": first_late, "anchor": anchor,
                                   "kept_alive": dict(keep),
                                   "gap_operations": size, "copied": [n for _, n, _ in new_ops],
                                   "bytes": value.numel * (2 if value.dtype == "bfloat16" else 4)}
    if not groups:
        return graph, report
    by_position, redirect = {}, {}
    for position, new_ops, late_names, mapping in groups:
        by_position.setdefault(position, []).extend(new_ops)
        for name in late_names:
            redirect.setdefault(name, {}).update(mapping)
    values, new_producer, result = dict(graph.values), {}, []

    def dependencies(reads):
        out = []
        for name in reads:
            if name in new_producer:
                out.append(new_producer[name])
            elif name in producer:
                out.append(producer[name].name)
        return tuple(dict.fromkeys(out))

    for index, op in enumerate(ops):
        for source, name, local in by_position.get(index, ()):
            reads = tuple(local.get(x, x) for x in source.reads)
            result.append(replace(source, name=name, outputs=(name,), writes=(name,), reads=reads,
                                  args=_rewrite(source.args, local), kwargs=_rewrite(source.kwargs, local),
                                  inputs=tuple(local.get(x, x) for x in source.inputs),
                                  dependencies=dependencies(reads)))
            old = graph.values[source.outputs[0]]
            alias = local.get(old.alias_of, old.alias_of) if old.alias_of else None
            values[name] = replace(old, name=name, producer=name, alias_of=alias, output_roles=())
            new_producer[name] = name
            report["copied_operations"] += 1
        mapping = redirect.get(op.name)
        if mapping:
            reads = tuple(mapping.get(x, x) for x in op.reads)
            op = replace(op, args=_rewrite(op.args, mapping), kwargs=_rewrite(op.kwargs, mapping),
                         inputs=tuple(mapping.get(x, x) for x in op.inputs), reads=reads,
                         dependencies=dependencies(reads))
        result.append(op)
    # Drop original views left without readers (pure aliases; no effect).
    while True:
        read = set()
        for op in result:
            read.update(op.reads)
        dead = [op for op in result if is_view(op) and op.outputs[0] not in read and op.outputs[0] not in protected
                and op.outputs[0] in graph.values]
        if not dead:
            break
        dead_names = {op.name for op in dead}
        result = [op for op in result if op.name not in dead_names]
        for op in dead:
            values.pop(op.outputs[0], None)
        report["removed_dead_views"] += len(dead)
    metadata = dict(graph.metadata)
    metadata["remat_anchors"] = anchors
    report.update(applied=len(groups), saved_bytes_upper_bound=sum(c["bytes"] for c in report["candidates"].values()))
    metadata["remat_report"] = {k: v for k, v in report.items() if k != "candidates"}
    return replace(graph, values=values, operations=result, metadata=metadata), report


def _reaches(view_op, late_names, readers, is_view, limit=16):
    frontier = [view_op.outputs[0]]
    for _ in range(limit):
        following = []
        for name in frontier:
            for op in readers.get(name, ()):
                if op.name in late_names:
                    return True
                if is_view(op):
                    following.append(op.outputs[0])
        if not following:
            return False
        frontier = following
    return False
