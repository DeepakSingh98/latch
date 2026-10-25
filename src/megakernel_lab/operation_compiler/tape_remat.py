"""P9-B: recompute cheap elementwise tape entries in the reverse body instead of saving them.

A saved-tape loop body may save the result of an FP32 elementwise product ``aten.mul.Tensor(a, b)`` whose operands
are themselves available to the reverse region: other saved tape entries, or forward-region inputs the reverse region
also binds (same boundary label). RMS normalization is the case on the primary: the tape holds both the normalized
input x * rstd and its factors x and rstd. This pass copies such a product into the reverse region (operands bound to
the reverse region's tape/boundary inputs), rewires its readers and drops the entry from the tape ABI. The copy is
the same IEEE operation on the same operand values, so every recomputed element has the bits of the saved one.
Each saved product costs one FP32 write in the forward body and one read in the reverse body; the recomputation
reads operands the reverse readers usually load anyway.
"""
from __future__ import annotations

import json

TARGETS = ("aten.mul.Tensor", "aten.add.Tensor", "aten.sub.Tensor", "aten.div.Tensor")   # (wtrm2, gpu-81)
LATE = True     # insert each recomputed product right before its first reverse reader (fusable into it)
MIN_NUMEL = 1 << 18


def _ref_map(x, mapping):
    if isinstance(x, dict):
        if set(x) == {"ref"}:
            return {"ref": mapping.get(x["ref"], x["ref"])}
        return {k: _ref_map(v, mapping) for k, v in x.items()}
    if isinstance(x, list):
        return [_ref_map(v, mapping) for v in x]
    return x


def _refs(x):
    if isinstance(x, dict):
        if set(x) == {"ref"}:
            return [x["ref"]]
        return [r for v in x.values() for r in _refs(v)]
    if isinstance(x, list):
        return [r for v in x for r in _refs(v)]
    return []


def _numel(value):
    n = 1
    for d in value.get("shape") or ():
        n *= int(d)
    return n


def remat_tape_products(data):
    if data.get("storage_mode") != "save_body_tape":
        return data, {"recomputed": []}
    R = data.get("regions", {})
    fwd, rev = R.get("body_saved"), R.get("body_vjp_saved")
    tape = data.get("metadata", {}).get("body_tape")
    if not fwd or not rev or not tape:
        return data, {"recomputed": []}
    if "operations" not in fwd or "operations" not in rev:
        return data, {"recomputed": [], "refused": "nested regions retain their original tape ABI"}
    producer = {o["outputs"][0]: o for o in fwd["operations"] if o["outputs"]}
    flabels = fwd.get("metadata", {}).get("input_labels") or []
    rlabels = rev.get("metadata", {}).get("input_labels") or []
    finput_label = dict(zip(fwd["inputs"], flabels))
    rinput_by_label = dict(zip(rlabels, rev["inputs"]))
    tape_names = [t["name"] for t in tape]
    candidates = {}
    for name in tape_names:
        op = producer.get(name)
        value = fwd["values"].get(name, {})
        if (op is None or op["target"] not in TARGETS or op.get("kwargs") or value.get("dtype") != "float32"
                or _numel(value) < MIN_NUMEL or f"tape:{name}" not in rinput_by_label):
            continue
        refs = _refs(op["args"])
        if len(refs) != 2:
            continue
        candidates[name] = (op, refs)

    def bound(ref, dropping):
        if ref in finput_label and finput_label[ref] in rinput_by_label:
            return rinput_by_label[finput_label[ref]]
        if ref in tape_names and ref not in dropping and f"tape:{ref}" in rinput_by_label:
            return rinput_by_label[f"tape:{ref}"]
        return None

    dropping = set(candidates)
    changed = True
    while changed:                       # operands must stay available (no chains of recomputed entries)
        changed = False
        for name in sorted(dropping):
            if any(bound(r, dropping) is None for r in candidates[name][1]):
                dropping.discard(name)
                changed = True
    if not dropping:
        return data, {"recomputed": []}
    data = json.loads(json.dumps(data))
    R, md = data["regions"], data["metadata"]
    fwd, rev, tape = R["body_saved"], R["body_vjp_saved"], md["body_tape"]
    rlabels = rev["metadata"]["input_labels"]
    rinput_by_label = dict(zip(rlabels, rev["inputs"]))
    new_ops, rename = [], {}
    for name in [n for n in tape_names if n in dropping]:
        op, refs = candidates[name]
        mapping = {r: bound(r, dropping) for r in refs}
        new = f"p9t_{name}"
        rev["values"][new] = dict(fwd["values"][name], name=new, producer=f"p9t_{op['name']}", alias_of=None,
                                  output_roles=[])
        new_ops.append(dict(op, name=f"p9t_{op['name']}", outputs=[new], writes=[new],
                            args=_ref_map(op["args"], mapping),
                            inputs=[mapping.get(x, x) for x in op["inputs"]],
                            reads=[mapping.get(x, x) for x in op["reads"]], dependencies=[]))
        rename[rinput_by_label[f"tape:{name}"]] = new
    n_tape = len(tape)
    lead = len(fwd["outputs"]) - n_tape
    keep = [k for k, t in enumerate(tape) if t["name"] not in dropping]
    md["body_tape"] = [tape[k] for k in keep]
    fwd["outputs"] = fwd["outputs"][:lead] + [fwd["outputs"][lead + k] for k in keep]
    for idx, out in enumerate(fwd["outputs"]):
        if out in fwd["values"]:
            fwd["values"][out]["output_roles"] = [r for r in fwd["values"][out].get("output_roles", [])
                                                 if not r.startswith("region_output:")] + [f"region_output:{idx}"]
    for name in dropping:                # the forward value is no longer a region output
        roles = fwd["values"].get(name, {}).get("output_roles", [])
        if name not in fwd["outputs"] and name in fwd["values"]:
            fwd["values"][name]["output_roles"] = [r for r in roles if not r.startswith("region_output:")]
    drop_inputs = set(rename)
    keep_in = [i for i, n in enumerate(rev["inputs"]) if n not in drop_inputs]
    rev["inputs"] = [rev["inputs"][i] for i in keep_in]
    rev["metadata"]["input_labels"] = [rlabels[i] for i in keep_in]
    for n in drop_inputs:
        rev["values"].pop(n, None)
    rewired = [dict(o, args=_ref_map(o["args"], rename), kwargs=_ref_map(o["kwargs"], rename),
                    inputs=[rename.get(x, x) for x in o["inputs"]],
                    reads=[rename.get(x, x) for x in o["reads"]]) for o in rev["operations"]]
    if not LATE:
        rev["operations"] = new_ops + rewired
    else:
        produced = {op["outputs"][0]: op for op in new_ops}
        placed, order = set(), []
        for o in rewired:
            for x in o["inputs"]:
                if x in produced and produced[x]["name"] not in placed:
                    placed.add(produced[x]["name"])
                    order.append(produced[x])
            order.append(o)
        order[:0] = [op for op in new_ops if op["name"] not in placed]
        rev["operations"] = order
    return data, {"recomputed": sorted(dropping)}


__all__ = ["remat_tape_products"]
