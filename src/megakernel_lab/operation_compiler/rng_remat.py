"""P9-B: recompute saved Philox dropout masks in the reverse body instead of saving them on the tape.

A saved-tape loop body may save ``latch_ops.philox_mask`` results (FP32, one value per element) for its reverse
region. Each mask is a pure function of the iteration's RNG state, which the reverse region already reads (the same
``state:rng`` boundary binding as the forward region), through a chain of ``philox_advance`` transitions. This pass
copies that chain into the reverse region, rewires the mask readers to the recomputed values and drops the mask
entries from the tape ABI. The recomputation is the same Philox stream at the same offsets, so every mask element
has the same bits as the saved one.
"""
from __future__ import annotations

import json

MASK = "latch_ops.philox_mask.default"
# P9-B: insert each recomputed chain immediately before its first reader in the reverse region (instead of at the
# region start), so that elementwise fusion can absorb the mask into its consumer instead of materializing it.
# Default since gpu-74: 7.222 vs 7.365 ms, bitwise 12/12, peak -36 MiB.
LATE = True
ADVANCE = "latch_ops.philox_advance.default"


def _ref_map(x, mapping):
    if isinstance(x, dict):
        if set(x) == {"ref"}:
            return {"ref": mapping.get(x["ref"], x["ref"])}
        return {k: _ref_map(v, mapping) for k, v in x.items()}
    if isinstance(x, list):
        return [_ref_map(v, mapping) for v in x]
    return x


def remat_philox_masks(data):
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
    targets = []
    for entry in tape:
        op = producer.get(entry["name"])
        if op is None or op["target"] != MASK:
            continue
        # The chain back to a region input must be philox_advance ops only, from an input the reverse region reads.
        chain, ref, ok = [], op["args"][0].get("ref") if isinstance(op["args"][0], dict) else None, True
        while ref is not None and ref in producer:
            p = producer[ref]
            if p["target"] != ADVANCE:
                ok = False
                break
            chain.append(p)
            ref = p["args"][0].get("ref") if isinstance(p["args"][0], dict) else None
        label = finput_label.get(ref)
        if not ok or ref is None or label is None or label not in rinput_by_label:
            continue
        targets.append((entry["name"], op, list(reversed(chain)), ref, label))
    if not targets:
        return data, {"recomputed": []}
    data = json.loads(json.dumps(data))
    R, md = data["regions"], data["metadata"]
    fwd, rev, tape = R["body_saved"], R["body_vjp_saved"], md["body_tape"]
    rlabels = rev["metadata"]["input_labels"]
    position = {label: i for i, label in enumerate(rlabels)}
    new_ops, made, rename = [], {}, {}
    for name, op, chain, state_in, label in targets:
        mapping = {state_in: rinput_by_label[label]}
        for step in chain + [op]:
            out = step["outputs"][0]
            new = made.get(out) or f"p9r_{out}"
            if out not in made:
                made[out] = new
                value = dict(fwd["values"][out], name=new, producer=f"p9r_{step['name']}", alias_of=None,
                             output_roles=[])
                rev["values"][new] = value
                new_ops.append(dict(step, name=f"p9r_{step['name']}", outputs=[new], writes=[new],
                                    args=_ref_map(step["args"], {**mapping, **made}),
                                    inputs=[{**mapping, **made}.get(x, x) for x in step["inputs"]],
                                    reads=[{**mapping, **made}.get(x, x) for x in step["reads"]],
                                    dependencies=[f"p9r_{d}" for d in step.get("dependencies", []) if f"p9r_{d}" in
                                                  {o["name"] for o in new_ops}]))
            mapping[out] = made[out]
        rename[rev["inputs"][position[f"tape:{name}"]]] = made[op["outputs"][0]]
    dropped = {name for name, *_ in targets}
    n_tape = len(tape)
    lead = len(fwd["outputs"]) - n_tape
    keep = [k for k, t in enumerate(tape) if t["name"] not in dropped]
    md["body_tape"] = [tape[k] for k in keep]
    fwd["outputs"] = fwd["outputs"][:lead] + [fwd["outputs"][lead + k] for k in keep]
    for idx, out in enumerate(fwd["outputs"]):
        if out in fwd["values"]:
            fwd["values"][out]["output_roles"] = [r for r in fwd["values"][out].get("output_roles", [])
                                                 if not r.startswith("region_output:")] + [f"region_output:{idx}"]
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
        # Each new op goes right before the first rewired op that reads one of its outputs (chains stay in order).
        produced = {op["outputs"][0]: op for op in new_ops}
        placed, order = set(), []
        def place(op_):
            for dep in op_["inputs"]:
                if dep in produced and produced[dep]["name"] not in placed:
                    place(produced[dep])
            if op_["name"] not in placed:
                placed.add(op_["name"])
                order.append(op_)
        for o in rewired:
            for x in o["inputs"]:
                if x in produced:
                    place(produced[x])
            order.append(o)
        for op_ in new_ops:                      # unread recomputations (none expected) keep the region start
            if op_["name"] not in placed:
                order.insert(0, op_)
        rev["operations"] = order
    return data, {"recomputed": sorted(dropped)}


__all__ = ["remat_philox_masks"]
