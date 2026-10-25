"""Row-scatter / matrix-product commutation (LATCH-MOE-EMA-100 s01): a joint-IR rewrite applied before partition.

    X = embedding_dense_backward(S, idx, N, -1, False)     FP32 [N, d]   (rows idx[m] receive S[m], others +0)
    Y = mm(X, W)                                            FP32 [N, n]

becomes

    Yc = mm(S, W)                                           FP32 [M, n]   (only the M scattered rows)
    Y  = embedding_dense_backward(Yc, idx, N, -1, False)    FP32 [N, n]   (latch_unique_rows: traps on a repeat)

With unique indices both forms give the same bits. A scattered row of X is +0 + S[m], so it differs from S[m] at
most in the sign of a zero element; every FP32 product row of the shared matrix bodies is an fmaf chain from +0, and
such a chain never holds -0 under round-to-nearest (x + (-x) = +0, +0 + -0 = +0), so a zero addend of either sign
leaves it unchanged: Y's scattered rows equal Yc's rows, and they are never -0, so +0 + Yc[m] = Yc[m]. Every other
row of X is +0, hence its product row is +0, as the scatter writes. Repeated indices would make X's rows sums, so
the new scatter carries ``latch_unique_rows`` and the lowering traps when a row receives more than one occurrence
(both scatters share the cached occurrence order of idx). X keeps its other readers (e.g. the backward pass), so
the gradients are unchanged. ``ENABLED = False`` disables it.
"""
from __future__ import annotations

import copy
import os

ENABLED = os.environ.get("LATCH_OPS_ROW_SCATTER_MM", "1") != "0"
# s02: G = index_select(cast(mm(A, B)), 0, idx) -> cast(mm(index_select(A, 0, idx), B)). A row gather commutes
# with a row-wise cast and with the product's rows (each product row is its own k-chain of the same body class), so
# only the gathered rows are multiplied, cast and stored; out-of-range indices fault in the gather either way.
GATHER_PRODUCTS = os.environ.get("LATCH_OPS_GATHER_PRODUCTS", "0") == "1"   # opt-in (audit tracing pending)
# s03: G = index_select(cast(P), 0, idx) -> cast(index_select(P, 0, idx)) when the cast's only reader is the gather:
# casts are elementwise, so only the gathered rows are converted (the product P itself is unchanged).
GATHER_CASTS = os.environ.get("LATCH_OPS_GATHER_CASTS", "1") != "0"
INDEX_SELECT = "aten.index_select.default"
CAST = "aten._to_copy.default"
EDB = "aten.embedding_dense_backward.default"
MM = "aten.mm.default"


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return list(reversed(strides))


def rewrite(graph, *, gather_products=None):
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
    from .gather_cast import _output_names
    outputs = _output_names(data)
    producers = {o: op for op in ops for o in op.get("outputs", ())}
    report = {"version": 1, "rewritten": []}
    for op in list(ops):
        if op.get("target") != MM or op.get("kwargs") or len(op.get("args", ())) != 2:
            continue
        y = op["outputs"][0] if len(op.get("outputs", ())) == 1 else None
        x, w = _ref(op["args"][0]), _ref(op["args"][1])
        scatter = producers.get(x)
        if (y is None or y in outputs or w is None or scatter is None or scatter.get("target") != EDB
                or scatter.get("kwargs") or len(scatter.get("args", ())) != 5):
            continue
        s, idx = _ref(scatter["args"][0]), _ref(scatter["args"][1])
        rows, padding, by_freq = scatter["args"][2:5]
        if (s is None or idx is None or padding != -1 or by_freq is not False or type(rows) is not int
                or any(values.get(n, {}).get("dtype") != "float32" for n in (x, y, s, w))
                or values[idx].get("dtype") != "int64" or len(values[idx]["shape"]) != 1
                or len(values[s]["shape"]) != 2 or len(values[w]["shape"]) != 2
                or list(values[x]["shape"]) != [rows, values[s]["shape"][1]]
                or values[w]["shape"][0] != values[s]["shape"][1]
                or list(values[y]["shape"]) != [rows, values[w]["shape"][1]]
                or values[s]["shape"][0] != values[idx]["shape"][0] or values[s]["shape"][0] >= rows):
            continue
        compact = f"{y}__rows"
        if compact in values:
            continue
        m, n = values[s]["shape"][0], values[w]["shape"][1]
        values[compact] = {**copy.deepcopy(values[y]), "name": compact, "shape": [m, n], "strides": _contiguous([m, n]),
                           "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                           "producer": compact, "output_roles": []}
        product = {**copy.deepcopy(op), "name": compact, "args": [{"ref": s}, {"ref": w}], "outputs": [compact],
                   "inputs": [s, w], "reads": [s, w], "writes": [compact],
                   "dependencies": [n_ for n_ in (s, w) if n_ not in data.get("inputs", ())]}
        ops.insert(ops.index(op), product)
        op.update(target=EDB, args=[{"ref": compact}, {"ref": idx}, rows, -1, False],
                  kwargs={"latch_unique_rows": True}, inputs=[compact, idx], reads=[compact, idx],
                  dependencies=[compact] + ([idx] if idx not in data.get("inputs", ()) else []))
        report["rewritten"].append({"product": y, "scatter": x, "rows": rows, "scattered_rows": m,
                                    "compact_product": compact, "indices": idx})
    report["gathered"] = _gather_products(data, ops, values, outputs, enabled=gather_products)
    report["gathered_casts"] = _gather_casts(data, ops, values, outputs)
    if not report["rewritten"] and not report["gathered"] and not report["gathered_casts"]:
        return graph, None
    data.setdefault("metadata", {})["row_scatter_mm"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def _gather_products(data, ops, values, outputs, *, enabled=None):
    if not (GATHER_PRODUCTS if enabled is None else enabled):
        return []
    producers = {o: op for op in ops for o in op.get("outputs", ())}

    def readers(name):
        return [o for o in ops if name in (o.get("inputs") or o.get("reads") or ())]

    done = []
    for gather in list(ops):
        if gather.get("target") != INDEX_SELECT or gather.get("kwargs") or len(gather.get("args", ())) != 3:
            continue
        c, dim, idx = _ref(gather["args"][0]), gather["args"][1], _ref(gather["args"][2])
        g = gather["outputs"][0]
        cast = producers.get(c)
        if (c is None or idx is None or dim != 0 or cast is None or cast.get("target") != CAST
                or len(cast.get("args", ())) != 1 or set(cast.get("kwargs") or {}) - {"dtype"}):
            continue
        prod_name = _ref(cast["args"][0])
        product = producers.get(prod_name)
        if (product is None or product.get("target") != MM or product.get("kwargs")
                or len(product.get("args", ())) != 2 or len(product.get("outputs", ())) != 1):
            continue
        a, b = _ref(product["args"][0]), _ref(product["args"][1])
        if (a is None or b is None or {c, prod_name} & outputs
                or any(values[n].get("alias_of") in (c, prod_name) for n in values)
                or [o["name"] for o in readers(c)] != [gather["name"]]
                or [o["name"] for o in readers(prod_name)] != [cast["name"]]
                or len(values[a]["shape"]) != 2 or len(values[idx]["shape"]) != 1
                or values[idx].get("dtype") != "int64"):
            continue
        m = values[idx]["shape"][0]
        k = values[a]["shape"][1]
        n = values[prod_name]["shape"][1]
        ag, pg = f"{a}__rows_{g}", f"{prod_name}__rows_{g}"
        if ag in values or pg in values:
            continue
        values[ag] = {**copy.deepcopy(values[a]), "name": ag, "shape": [m, k], "strides": _contiguous([m, k]),
                      "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                      "producer": ag, "output_roles": []}
        values[pg] = {**copy.deepcopy(values[prod_name]), "name": pg, "shape": [m, n], "strides": _contiguous([m, n]),
                      "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                      "producer": pg, "output_roles": []}
        inputs = set(data.get("inputs", ()))
        new_gather = {**copy.deepcopy(gather), "name": ag, "args": [{"ref": a}, 0, {"ref": idx}], "outputs": [ag],
                      "inputs": [a, idx], "reads": [a, idx], "writes": [ag],
                      "dependencies": [x for x in (a, idx) if x not in inputs]}
        new_product = {**copy.deepcopy(product), "name": pg, "args": [{"ref": ag}, {"ref": b}], "outputs": [pg],
                       "inputs": [ag, b], "reads": [ag, b], "writes": [pg],
                       "dependencies": [ag] + ([b] if b not in inputs else [])}
        new_cast = {**copy.deepcopy(cast), "name": g, "args": [{"ref": pg}], "outputs": [g],
                    "inputs": [pg], "reads": [pg], "writes": [g], "dependencies": [pg]}
        at = ops.index(gather)
        ops[at:at + 1] = [new_gather, new_product, new_cast]
        ops.remove(cast)
        ops.remove(product)
        del values[c], values[prod_name]
        done.append({"gather": g, "product": prod_name, "cast": c, "rows": values[a]["shape"][0], "gathered_rows": m,
                     "indices": idx})
    return done


def _gather_casts(data, ops, values, outputs):
    if not GATHER_CASTS:
        return []
    producers = {o: op for op in ops for o in op.get("outputs", ())}
    done = []
    for gather in list(ops):
        if gather.get("target") != INDEX_SELECT or gather.get("kwargs") or len(gather.get("args", ())) != 3:
            continue
        c, dim, idx = _ref(gather["args"][0]), gather["args"][1], _ref(gather["args"][2])
        g = gather["outputs"][0]
        cast = producers.get(c)
        if (c is None or idx is None or dim != 0 or cast is None or cast.get("target") != CAST
                or len(cast.get("args", ())) != 1 or set(cast.get("kwargs") or {}) - {"dtype"} or c in outputs):
            continue
        p = _ref(cast["args"][0])
        readers = [o["name"] for o in ops if c in (o.get("inputs") or o.get("reads") or ())]
        if (p is None or readers != [gather["name"]] or any(values[n].get("alias_of") == c for n in values)
                or len(values[p]["shape"]) != 2 or values[idx].get("dtype") != "int64"):
            continue
        pg = f"{p}__rows_{g}"
        if pg in values:
            continue
        m, n = values[idx]["shape"][0], values[p]["shape"][1]
        values[pg] = {**copy.deepcopy(values[p]), "name": pg, "shape": [m, n], "strides": _contiguous([m, n]),
                      "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                      "producer": pg, "output_roles": []}
        inputs = set(data.get("inputs", ()))
        new_gather = {**copy.deepcopy(gather), "name": pg, "args": [{"ref": p}, 0, {"ref": idx}], "outputs": [pg],
                      "inputs": [p, idx], "reads": [p, idx], "writes": [pg],
                      "dependencies": [x for x in (p, idx) if x not in inputs]}
        new_cast = {**copy.deepcopy(cast), "name": g, "args": [{"ref": pg}], "outputs": [g],
                    "inputs": [pg], "reads": [pg], "writes": [g], "dependencies": [pg]}
        at = ops.index(gather)
        ops[at:at + 1] = [new_gather, new_cast]
        ops.remove(cast)
        del values[c]
        done.append({"gather": g, "cast": c, "source": p, "rows": values[p]["shape"][0], "gathered_rows": m})
    return done
