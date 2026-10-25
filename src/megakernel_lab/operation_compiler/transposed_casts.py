"""P9-B: transposed copies of update-invariant weight casts for N-major product operands.

A forward product ``mm(x, W)`` with an invariant BF16 weight cast ``W`` ([in, out], row-major) reads its B operand
N-major. This pass adds, per such weight, one more once-per-launch cast ``Wt = clone(t(W))`` ([out, in], contiguous)
to the ``invariant_casts`` region, binds it in every consumer region right after the existing casts (the shared cast
ABI), and rewrites the forward products to ``mm(x, t(Wt))``: the same values in a K-major B operand. Every output
element keeps its FP32 accumulation in ascending k16 steps from +0 and one rounding; only the operand layout changes.
"""
from __future__ import annotations

import json

CONSUMER_SKIP = ("optimizer", "invariant_casts")


def _cast_names(n_casts):
    return [f"invariant_cast_{i}" for i in range(n_casts)]


def transpose_forward_casts(data, regions=("body_saved",)):
    """(new IR dict, report) for a BoundedLoopIR dict; unchanged when nothing qualifies."""
    md = data.get("metadata", {})
    casts = list(md.get("invariant_casts") or [])
    R = data.get("regions", {})
    if not casts or "invariant_casts" not in R:
        return data, {"transposed": []}
    names = _cast_names(len(casts))
    targets = []
    for rname in regions:
        region = R.get(rname)
        if not region:
            continue
        for op in region["operations"]:
            if (op["target"] == "aten.mm.default" and isinstance(op["args"][1], dict)
                    and op["args"][1].get("ref") in names):
                k = names.index(op["args"][1]["ref"])
                value = region["values"][names[k]]
                if (len(value["shape"]) == 2 and list(value["strides"]) == [value["shape"][1], 1]
                        and value["dtype"] == "bfloat16" and k not in targets):
                    targets.append(k)
    if not targets:
        return data, {"transposed": []}
    data = json.loads(json.dumps(data))          # plain lists/dicts (to_dict may hold tuples)
    md, R = data["metadata"], data["regions"]
    n_par, n_cast = len(md["parameter_names"]), len(casts)
    ic = R["invariant_casts"]
    new_names = []
    for j, k in enumerate(targets):
        rows, cols = casts[k]["shape"]
        nm = f"invariant_cast_{n_cast + j}"
        new_names.append(nm)
        md["invariant_casts"].append({"name": nm, "source": casts[k]["source"], "dtype": "bfloat16",
                                      "shape": [cols, rows], "transposed_of": casts[k]["name"]})
        src = ic["outputs"][k]
        src_op = ic["values"][src]["producer"]
        view, out = f"p9t_view_{j}", f"p9t_cast_{j}"
        ic["values"][view] = {"name": view, "shape": [cols, rows], "dtype": "bfloat16", "strides": [1, cols],
                              "role": "temporary", "parameter": None, "alias_of": src, "storage_offset": 0,
                              "producer": view, "output_roles": []}
        ic["values"][out] = {"name": out, "shape": [cols, rows], "dtype": "bfloat16", "strides": [rows, 1],
                             "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                             "producer": out, "output_roles": [f"region_output:{n_cast + j}"]}
        ic["operations"].append({"name": view, "target": "aten.t.default", "args": [{"ref": src}], "kwargs": {},
                                 "inputs": [src], "outputs": [view], "dependencies": [src_op] if src_op else [],
                                 "reads": [src], "writes": [], "reduction_axes": [], "synchronization": "view",
                                 "effect": "pure"})
        ic["operations"].append({"name": out, "target": "aten.clone.default", "args": [{"ref": view}],
                                 "kwargs": {"memory_format": "contiguous_format"}, "inputs": [view],
                                 "outputs": [out], "dependencies": [view], "reads": [view], "writes": [out],
                                 "reduction_axes": [], "synchronization": "global_phase_boundary", "effect": "pure"})
        ic["outputs"].append(out)
    # Shared cast ABI: the new casts follow the existing ones in every consumer region.
    for rname, region in R.items():
        if rname in CONSUMER_SKIP or region is None:
            continue
        labels = region.get("metadata", {}).get("input_labels")
        if len(region["inputs"]) < n_par + n_cast or region["inputs"][n_par:n_par + n_cast] != names:
            continue
        at = n_par + n_cast
        for j, (k, nm) in enumerate(zip(targets, new_names)):
            rows, cols = casts[k]["shape"]
            region["inputs"].insert(at + j, nm)
            region["values"][nm] = {"name": nm, "shape": [cols, rows], "dtype": "bfloat16", "strides": [rows, 1],
                                    "role": "input", "parameter": None, "alias_of": None, "storage_offset": 0,
                                    "producer": None, "output_roles": []}
            if labels is not None:
                labels.insert(at + j, f"invariant_cast:{casts[k]['source']}:bfloat16:transposed")
    # Forward products read the transposed copy through a transposed view (K-major B).
    rewritten = 0
    for rname in regions:
        region = R.get(rname)
        if not region:
            continue
        ops = []
        for op in region["operations"]:
            if (op["target"] == "aten.mm.default" and isinstance(op["args"][1], dict)
                    and op["args"][1].get("ref") in names and names.index(op["args"][1]["ref"]) in targets):
                k = names.index(op["args"][1]["ref"])
                nm = new_names[targets.index(k)]
                rows, cols = casts[k]["shape"]
                view = f"p9t_b_{op['name']}"
                region["values"][view] = {"name": view, "shape": [rows, cols], "dtype": "bfloat16",
                                          "strides": [1, rows], "role": "temporary", "parameter": None,
                                          "alias_of": nm, "storage_offset": 0, "producer": view, "output_roles": []}
                ops.append({"name": view, "target": "aten.t.default", "args": [{"ref": nm}], "kwargs": {},
                            "inputs": [nm], "outputs": [view], "dependencies": [], "reads": [nm], "writes": [],
                            "reduction_axes": [], "synchronization": "view", "effect": "pure"})
                old = names[k]
                op = dict(op, args=[op["args"][0], {"ref": view}],
                          inputs=[view if x == old else x for x in op["inputs"]],
                          reads=[view if x == old else x for x in op["reads"]],
                          dependencies=list(op.get("dependencies", [])) + [view])
                rewritten += 1
            ops.append(op)
        region["operations"] = ops
    return data, {"transposed": [casts[k]["name"] for k in targets], "products": rewritten}


__all__ = ["transpose_forward_casts"]
