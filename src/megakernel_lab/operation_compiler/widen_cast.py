"""Widening-cast elision (LATCH-MOESUP): a graph rewrite applied by ``lower`` before schedule selection.

    c = _to_copy(b, dtype=float32)          b BFloat16, c read only by FP32-result arithmetic (mul/add/sub/div)
    y = op(c, other)

becomes ``y = op(b, other)``. A BF16 operand of an FP32-result elementwise expression is read by widening, and
the widening is exact, so ``y`` is bit for bit the value computed from ``c``. The FP32 copy (e.g. the expert
output ``expert_output.float()`` kept for the selection-weight gradient: twice the BF16 bytes, written once and
read in both passes) is neither written nor re-read. Scope: ``c`` has the same shape as ``b``, is neither a graph
output nor viewed, and each of its readers is an aten mul/add/sub/div (Tensor overload) with an FP32 result.
``ENABLED = False`` disables it.
"""
from __future__ import annotations

import copy

ENABLED = True
# LATCH-MOE-EMA-FAST: also applied to programs with a captured precision policy (bit-exact storage commutation;
# the policy's values and roundings are unchanged). False restores the policy exclusion.
UNDER_PRECISION_POLICY = True
CAST = "aten._to_copy.default"
READERS = ("aten.mul.Tensor", "aten.add.Tensor", "aten.sub.Tensor", "aten.div.Tensor")
# LATCH-MOESUP-ROW: FP32-result elementwise functions whose rendered expression reads the operand as a float (a BF16
# operand by exact widening, lower.read), e.g. the expert activation's silu and its backward sigmoid reading the FP32
# copy of the up-projection kept for the backward pass. ``UNARY = False`` restores the arithmetic-only reader set.
UNARY = True
UNARY_READERS = ("aten.silu.default", "aten.sigmoid.default", "aten.tanh.default", "aten.exp.default",
                 "aten.neg.default", "aten.silu_backward.default", "aten.sigmoid_backward.default",
                 "aten.tanh_backward.default")


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _replace(obj, old, new):
    if isinstance(obj, dict):
        if set(obj) == {"ref"} and obj["ref"] == old:
            return {"ref": new}
        return {k: _replace(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace(v, old, new) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace(v, old, new) for v in obj)
    return obj


def _swap(seq, old, new):
    if not isinstance(seq, (list, tuple)):
        return seq
    return list(dict.fromkeys(new if x == old else x for x in seq))


def _output_names(data):
    """Every value name in the IR's (nested) outputs structure, plus values carrying output roles."""
    names = set()

    def walk(item):
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            if set(item) == {"ref"}:
                names.add(item["ref"])
            else:
                for v in item.values():
                    walk(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                walk(v)
    walk(data.get("outputs", {}))
    names |= {n for n, v in (data.get("values") or {}).items() if v.get("output_roles")}
    return names


def rewrite(graph):
    """Return (graph, report); the input is never mutated."""
    if not ENABLED:
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or not isinstance(values, dict):
        return graph, None
    outputs = _output_names(data)
    viewed = {v.get("alias_of") for v in values.values() if v.get("alias_of")}
    # A graph-bound serial subtotal chain names its exact widening producers.
    # Keep these boundaries available for k8_serial_sum's private-chain proof;
    # eliding them here would silently preempt that later scheduling rule.
    # Flattened source-only groups have no subtotal chain left to select.
    serial = (data.get("metadata") or {}).get("serial_bf16_readout") or {}
    serial_widenings = {name for weight in serial.get("weight_gradients", [])
                        if len(weight.get("partials", [])) >= 2
                        and not weight.get("source_only_removed_subtotals")
                        for name in weight.get("widened_partials", [])}
    # LATCH-MOE-EMA-FAST: a shared readout's flattened last-forward-first partial chain is an audited widening
    # sequence (addition by addition, on the joint IR); it is reached here when the rule runs before partitioning.
    serial_widenings |= {name for group in serial.get("weight_accumulation_groups", [])
                         for name in group.get("widened_partials", [])}
    report = {"version": 1, "rewritten": []}
    # LATCH-BUILD-600: readers come from an index of each operation's read set (built once, in operation order)
    # instead of rescanning every operation for every cast (O(casts x operations)). Exact: a rewrite only replaces
    # the elided FP32 name ``c`` by its BF16 source ``b`` in its readers; later casts query FP32 names, never a BF16
    # source or an elided name, so the read sets they see are the ones indexed. Removed casts are skipped as the
    # reference's ``ops.remove`` would, and are dropped from the list once, in order, at the end.
    read_index = {}
    for position, o in enumerate(ops):
        for name in set(o.get("reads") or o.get("inputs") or ()):
            read_index.setdefault(name, []).append(position)
    removed = set()
    for cast in list(ops):
        if cast.get("target") != CAST:
            continue
        c = cast["name"]
        b = _ref((cast.get("args") or [None])[0])
        if (b is None or c in outputs or c in viewed or c in serial_widenings
                or c not in values or b not in values
                or values[c].get("dtype") != "float32" or values[b].get("dtype") != "bfloat16"
                or list(values[b]["shape"]) != list(values[c]["shape"])
                or any(k not in ("dtype", "layout", "device", "memory_format") for k in (cast.get("kwargs") or {}))):
            continue
        readers = [ops[p] for p in read_index.get(c, ()) if ops[p] is not cast and id(ops[p]) not in removed]
        admitted = READERS + (UNARY_READERS if UNARY else ())
        if not readers or any(o.get("target") not in admitted or values.get(o["name"], {}).get("dtype") != "float32"
                              for o in readers):
            continue
        removed.add(id(cast))
        for o in readers:
            o["args"] = _replace(o.get("args"), c, b)
            o["kwargs"] = _replace(o.get("kwargs"), c, b)
            for key in ("inputs", "dependencies", "reads"):
                o[key] = _swap(o.get(key), c, b)
        del values[c]
        report["rewritten"].append({"cast": c, "source": b, "readers": [o["name"] for o in readers],
                                    "shape": values[b]["shape"]})
    if removed:
        ops[:] = [o for o in ops if id(o) not in removed]
    if not report["rewritten"]:
        return graph, None
    data.setdefault("metadata", {})["widen_cast"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report
