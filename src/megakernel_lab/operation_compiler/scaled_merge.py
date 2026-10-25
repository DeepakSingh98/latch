"""Row-scaled indexed merge (LATCH-MOESUP): a graph rewrite applied by ``lower`` before schedule selection.

    m = mul(b, s)                         b BFloat16 [..., R, D] contiguous, s FP32 [..., R, 1] (one factor per row)
    g = view(m, [N, D])                   identity row order (N = number of rows of m)
    e = embedding_dense_backward(g, idx, V, -1, False)

becomes ``e = embedding_dense_backward(view(b, [N, D]), idx, V, -1, False, latch_row_scale=s)``: the segmented
engine (ops/indexed_segmented.py, the only engine admitting the factor) forms each contribution as
``__fmul_rn(float(b), s[o])`` -- the stored FP32 product's value, rounded once -- immediately before its
unchanged ordered FP32 sum. The FP32 product (e.g. the expert-choice merge input ``expert_output * weight``:
twice the BF16 bytes) is neither written nor re-read. Scope: ``m`` is neither an output nor read except through
identity views by this one merge. Applied only when the requested indexed policy is the segmented engine.
``ENABLED = False`` disables it.

LATCH-MOESUP-ROW: the same merge input formed by an exact widening, ``m = _to_copy(b, dtype=float32)`` (e.g. the
expert backward output widened for the token-gradient merge), becomes
``embedding_dense_backward(view(b, [N, D]), ..., latch_row_widen=True)``: the engine reads the BF16 rows and widens
each contribution at load, so the summed FP32 values -- and the result -- are bit for bit those of the FP32 copy,
which is neither written nor re-read. ``WIDEN = False`` disables this case.
"""
from __future__ import annotations

import copy
import math

from .gather_cast import _output_names, _ref, _replace, _swap_names

ENABLED = True
MERGE = "aten.embedding_dense_backward.default"
MUL = "aten.mul.Tensor"
CAST = "aten._to_copy.default"
WIDEN = True


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return list(reversed(strides))


def _identity(value):
    return int(value.get("storage_offset") or 0) == 0 and all(
        int(e) == 1 or int(s) == c for e, s, c in zip(value["shape"], value["strides"], _contiguous(value["shape"])))


def rewrite(graph):
    """Return (graph, report); the input is never mutated."""
    if not ENABLED:
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or not isinstance(values, dict) or not any(op.get("target") == MERGE for op in ops):
        return graph, None
    outputs = _output_names(data)

    def root(name):
        for _ in range(256):
            alias = values.get(name, {}).get("alias_of")
            if not alias:
                return name
            name = alias
        return name

    producer = {op["name"]: op for op in ops}
    report = {"version": 1, "rewritten": []}
    for merge in [op for op in ops if op.get("target") == MERGE]:
        args = merge.get("args") or []
        g = _ref(args[0]) if args else None
        if (g is None or "latch_row_scale" in (merge.get("kwargs") or {}) or "latch_row_widen" in (merge.get("kwargs") or {})
                or len(args) < 3):
            continue
        if len(args) > 3 and args[3] not in (-1, None) or len(args) > 4 and args[4]:
            continue
        m = root(g)
        mul = producer.get(m)
        if mul is None or mul.get("target") not in (MUL, CAST) or m in outputs or values[m].get("dtype") != "float32":
            continue
        widen = mul.get("target") == CAST
        if widen:
            source = _ref((mul.get("args") or [None])[0])
            if (not WIDEN or source is None or len(mul.get("args") or []) != 1
                    or any(k not in ("dtype", "layout", "device", "memory_format") for k in (mul.get("kwargs") or {}))):
                continue
            operands = [source]
        else:
            operands = [_ref(a) for a in (mul.get("args") or [])[:2]]
            if len(operands) != 2 or None in operands:
                continue
        shape = list(values[m]["shape"])
        pick = None
        if widen:
            vb = values.get(operands[0])
            if (vb is not None and vb.get("dtype") == "bfloat16" and list(vb["shape"]) == shape and _identity(vb)
                    and len(shape) >= 2):
                pick = (operands[0], None)
        for b, s in (() if widen else (operands, operands[::-1])):
            vb, vs = values.get(b), values.get(s)
            if (vb is not None and vs is not None and vb.get("dtype") == "bfloat16" and vs.get("dtype") == "float32"
                    and list(vb["shape"]) == shape and _identity(vb) and len(shape) >= 2
                    and list(vs["shape"]) == shape[:-1] + [1]):
                pick = (b, s)
                break
        if pick is None or not _identity(values[m]):
            continue
        # Outside the segmented body's bounds the original FP32 merge has a
        # valid fallback. Keep that path; it cannot consume row-scaled BF16.
        from .indexed_segmented import supported
        if not isinstance(args[2], int) or not supported(math.prod(shape[:-1]), args[2], shape[-1])[0]:
            continue
        b, s = pick
        aliases = {n for n in values if root(n) == m}
        if aliases & outputs:
            continue
        readers = [o for o in ops if o is not mul and set(o.get("reads") or o.get("inputs") or ()) & aliases]
        nonview = [o for o in readers if o.get("synchronization") != "view"]
        if (len(nonview) != 1 or nonview[0] is not merge or not _identity(values[g])
                or list(values[g]["shape"]) != [math.prod(shape[:-1]), shape[-1]]):
            continue
        # 1. The merge's identity row view reads the BF16 product operand instead of the FP32 product.
        # Rank-two inputs may reach the merge directly, without any view to rewrite.
        if g == m:
            merge["args"] = _replace(merge.get("args"), m, b)
            for key in ("inputs", "dependencies", "reads"):
                merge[key] = _swap_names(merge.get(key), m, b)
        for o in readers:
            if o is merge:
                continue
            o["args"] = _replace(o.get("args"), m, b)
            for key in ("inputs", "dependencies", "reads"):
                o[key] = _swap_names(o.get(key), m, b)
        for n in aliases - {m}:
            values[n]["dtype"] = "bfloat16"
            if values[n].get("alias_of") == m:
                values[n]["alias_of"] = b
        if widen:
            # 2'. The merge widens each BF16 contribution at load; the FP32 copy disappears.
            merge["kwargs"] = {**(merge.get("kwargs") or {}), "latch_row_widen": True}
            ops.remove(mul)
            del values[m]
            report["rewritten"].append({"merge": merge["name"], "widened": m, "rows": b, "shape": shape})
            continue
        # 2. The factor becomes the merge's row scale; the FP32 product disappears.
        merge["kwargs"] = {**(merge.get("kwargs") or {}), "latch_row_scale": {"ref": s}}
        for key in ("inputs", "reads"):
            if isinstance(merge.get(key), (list, tuple)) and s not in merge[key]:
                merge[key] = [*merge[key], s]
        if isinstance(merge.get("dependencies"), (list, tuple)) and s not in merge["dependencies"] \
                and s not in data.get("inputs", ()):
            merge["dependencies"] = [*merge["dependencies"], s]
        ops.remove(mul)
        del values[m]
        report["rewritten"].append({"merge": merge["name"], "product": m, "rows": b, "scale": s, "shape": shape})
    if not report["rewritten"]:
        return graph, None
    data.setdefault("metadata", {})["scaled_merge"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report
