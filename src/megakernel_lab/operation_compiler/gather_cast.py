"""Cast-before-gather commutation (LATCH-MOESUP): a graph rewrite applied by ``lower`` before schedule selection.

    g = embedding(T, idx) | index_select(T, 0, idx)   FP32 rows      (g read only through views by one cast)
    c = _to_copy(view(g), dtype=bfloat16)

becomes

    Tb = _to_copy(T, dtype=bfloat16);   g' = gather(Tb, idx)  (BF16);   readers of c read view(g') instead.

Rounding is elementwise, so cast(gather(T)) and gather(cast(T)) are the same values bit for bit; out-of-range
indices fault in both. The FP32 gathered rows (e.g. expert-choice dispatch: 8x the table) are neither written nor
re-read; only the table is cast once. Gradients are unaffected: the gather's derivative reads only the indices and
the cast's derivative is computed from the cotangent. Scope: the gather's every non-view reader is that one cast,
neither value is a graph output, the cast reads an identity-shaped view. ``ENABLED = False`` disables it.
"""
from __future__ import annotations

import copy

ENABLED = True
# LATCH-MOE-EMA-FAST: also applied to programs with a captured precision policy (bit-exact storage commutation;
# the policy's values and roundings are unchanged). False restores the policy exclusion.
UNDER_PRECISION_POLICY = True
GATHERS = ("aten.embedding.default", "aten.index_select.default")
CAST = "aten._to_copy.default"


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


def _swap_names(seq, old, new):
    return [new if x == old else x for x in seq] if isinstance(seq, (list, tuple)) else seq


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
    # List-valued hand-authored IR is supported by lowering; this optional rewrite
    # only understands the keyed captured form and must leave other forms intact.
    if not ops or not isinstance(values, dict) or not any(op.get("target") in GATHERS for op in ops):
        return graph, None
    outputs = _output_names(data)

    def root(name):
        for _ in range(256):
            alias = values.get(name, {}).get("alias_of")
            if not alias:
                return name
            name = alias
        return name

    report = {"version": 1, "rewritten": []}
    for op in list(ops):
        if op.get("target") not in GATHERS:
            continue
        g = op["name"]
        if g not in values or values[g].get("dtype") != "float32" or g in outputs:
            continue
        aliases = {n for n in values if root(n) == g}
        if aliases & outputs:
            continue
        readers = [o for o in ops if o is not op and set(o.get("reads") or o.get("inputs") or ()) & aliases]
        nonview = [o for o in readers if o.get("synchronization") != "view"]
        if len(nonview) != 1 or nonview[0].get("target") != CAST:
            continue
        cast = nonview[0]
        c = cast["name"]
        src = _ref((cast.get("args") or [None])[0])
        if (c in outputs or values.get(c, {}).get("dtype") != "bfloat16" or src not in aliases
                or list(values[src]["shape"]) != list(values[c]["shape"])
                or any(k not in ("dtype", "layout", "device", "memory_format") for k in (cast.get("kwargs") or {}))):
            continue
        if any(values[n].get("alias_of") == c for n in values if n != c) and c in outputs:
            continue
        table_arg = 0
        table = _ref(op["args"][table_arg])
        if table is None or values[table].get("dtype") != "float32":
            continue
        tb = f"{table}__bf16_{g}"
        if tb in values:
            continue
        # 1. The table's BF16 cast, inserted before the gather.
        values[tb] = {**copy.deepcopy(values[table]), "name": tb, "dtype": "bfloat16", "role": "temporary",
                      "parameter": None, "alias_of": None, "storage_offset": 0, "producer": tb, "output_roles": [],
                      "strides": _contiguous(values[table]["shape"])}
        cast_table = {**copy.deepcopy(cast), "name": tb, "args": [{"ref": table}, *copy.deepcopy(cast["args"][1:])],
                      "inputs": [table], "outputs": [tb], "dependencies": [] if table in data.get("inputs", ()) else [table],
                      "reads": [table], "writes": [tb]}
        ops.insert(ops.index(op), cast_table)
        # 2. The gather reads the BF16 table and produces BF16 rows (its views follow).
        op["args"] = _replace(op["args"], table, tb)
        for key in ("inputs", "dependencies", "reads"):
            op[key] = _swap_names(op.get(key), table, tb)
        for n in aliases:
            values[n]["dtype"] = "bfloat16"
        # 3. Readers of the cast read its (identity-shaped) source view directly; the cast disappears.
        ops.remove(cast)
        for o in ops:
            o["args"] = _replace(o.get("args"), c, src)
            o["kwargs"] = _replace(o.get("kwargs"), c, src)
            for key in ("inputs", "dependencies", "reads"):
                o[key] = _swap_names(o.get(key), c, src)
        for n, v in values.items():
            if v.get("alias_of") == c:
                v["alias_of"] = src
        del values[c]
        report["rewritten"].append({"gather": g, "table": table, "table_cast": tb, "removed_cast": c,
                                    "rows": values[g]["shape"][0] if values[g]["shape"] else None})
    if not report["rewritten"]:
        return graph, None
    data.setdefault("metadata", {})["gather_cast"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return list(reversed(strides))
