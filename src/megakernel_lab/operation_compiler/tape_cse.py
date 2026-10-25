"""P9-B: common-subexpression elimination of the saved loop body and its tape.

The forward body of a saved-tape loop (``body_saved``) can compute the same pure value several times (for example
three identical BF16 casts of one normalized activation, one per query/key/value product, each from its own view of
the same tensor) and save every copy on the tape. This pass merges pure operations with the same target, the same
(canonicalized) arguments and keyword arguments and the same output layout, keeping the first; tape entries that
become duplicates are dropped from the tape ABI (``metadata.body_tape``, the forward region's outputs and the reverse
region's tape inputs, whose readers read the kept entry). Every merged value was computed by the same operation from
the same operands, so every read sees the same bits.
"""
from __future__ import annotations

import json

_NO_CSE = ("empty", "rand", "philox", "dropout", "bernoulli", "latch_ops")


def _ref_map(x, mapping):
    if isinstance(x, dict):
        if set(x) == {"ref"}:
            return {"ref": mapping.get(x["ref"], x["ref"])}
        return {k: _ref_map(v, mapping) for k, v in x.items()}
    if isinstance(x, list):
        return [_ref_map(v, mapping) for v in x]
    return x


def cse_saved_body(data):
    """(new BoundedLoopIR dict, report). Unchanged when nothing merges or the IR is not a saved-tape loop."""
    if data.get("storage_mode") != "save_body_tape":
        return data, {"merged": []}
    R = data.get("regions", {})
    fwd, rev = R.get("body_saved"), R.get("body_vjp_saved")
    tape = data.get("metadata", {}).get("body_tape")
    if not fwd or not rev or not tape:
        return data, {"merged": []}
    if "operations" not in fwd or "operations" not in rev:
        return data, {"merged": [], "refused": "nested regions retain their original tape ABI"}
    data = json.loads(json.dumps(data))
    R, md = data["regions"], data["metadata"]
    fwd, rev, tape = R["body_saved"], R["body_vjp_saved"], md["body_tape"]
    outputs = fwd["outputs"]
    n_out = len(outputs)
    n_tape = len(tape)
    lead = n_out - n_tape                                       # carries + states precede the tape entries
    if lead < 0 or [outputs[lead + k] for k in range(n_tape)] != [t["name"] for t in tape]:
        return data, {"merged": [], "refused": "tape entries are not the trailing forward outputs"}
    protected = set(outputs[:lead])
    values = fwd["values"]
    mapping, op_map, seen, kept = {}, {}, {}, []
    for op in fwd["operations"]:
        op = dict(op, args=_ref_map(op["args"], mapping), kwargs=_ref_map(op["kwargs"], mapping),
                  inputs=[mapping.get(x, x) for x in op["inputs"]], reads=[mapping.get(x, x) for x in op["reads"]],
                  dependencies=list(dict.fromkeys(op_map.get(d, d) for d in op.get("dependencies", []))))
        base = op["target"].removeprefix("aten.").split(".")[0]
        if (op.get("effect") == "pure" and len(op["outputs"]) == 1 and op["outputs"][0] not in protected
                and not any(word in op["target"] for word in _NO_CSE) and list(op.get("writes", [])) in ([], op["outputs"])):
            out = values[op["outputs"][0]]
            alias = mapping.get(out.get("alias_of"), out.get("alias_of"))
            key = (op["target"], json.dumps(op["args"], sort_keys=True), json.dumps(op["kwargs"], sort_keys=True),
                   out["dtype"], json.dumps(out["shape"]), json.dumps(out["strides"]), out.get("storage_offset", 0),
                   alias)
            if key in seen:
                canonical_op, canonical_value = seen[key]
                mapping[op["outputs"][0]] = canonical_value
                op_map[op["name"]] = canonical_op
                continue
            seen[key] = (op["name"], op["outputs"][0])
        kept.append(op)
    merged = {k: v for k, v in mapping.items()}
    if not merged:
        return data, {"merged": []}
    fwd["operations"] = kept
    for name in merged:
        values.pop(name, None)
    for name, value in values.items():
        if value.get("alias_of") in mapping:
            value["alias_of"] = mapping[value["alias_of"]]
    # Tape entries that became duplicates: drop them; reverse readers read the kept entry's tape input.
    tape_names = [t["name"] for t in tape]
    dropped = [k for k, name in enumerate(tape_names) if name in merged]
    if dropped:
        labels = rev.get("metadata", {}).get("input_labels")
        if labels is None:
            raise ValueError("reverse region lacks input labels")
        position = {label: i for i, label in enumerate(labels)}
        rename = {}
        for k in dropped:
            name = tape_names[k]
            canonical = merged[name]
            if canonical not in tape_names:
                # The kept value is not itself saved: keep this entry, now holding the kept value.
                outputs[lead + k] = canonical
                tape[k]["name"] = canonical
                continue
            i_drop, i_keep = position[f"tape:{name}"], position[f"tape:{canonical}"]
            rename[rev["inputs"][i_drop]] = rev["inputs"][i_keep]
        drop_names = {tape_names[k] for k in dropped if merged[tape_names[k]] in tape_names}
        keep_idx = [k for k in range(n_tape) if tape_names[k] not in drop_names]
        md["body_tape"] = [tape[k] for k in keep_idx]
        fwd["outputs"] = outputs[:lead] + [outputs[lead + k] for k in keep_idx]
        drop_inputs = {rev["inputs"][position[f"tape:{n}"]] for n in drop_names}
        keep_in = [i for i, name in enumerate(rev["inputs"]) if name not in drop_inputs]
        rev["inputs"] = [rev["inputs"][i] for i in keep_in]
        if labels is not None:
            rev["metadata"]["input_labels"] = [labels[i] for i in keep_in]
        for name in drop_inputs:
            rev["values"].pop(name, None)
        rev["operations"] = [dict(op, args=_ref_map(op["args"], rename), kwargs=_ref_map(op["kwargs"], rename),
                                  inputs=[rename.get(x, x) for x in op["inputs"]],
                                  reads=[rename.get(x, x) for x in op["reads"]]) for op in rev["operations"]]
        for value in rev["values"].values():
            if value.get("alias_of") in rename:
                value["alias_of"] = rename[value["alias_of"]]
        # Output roles of the forward values follow the new positions.
        for idx, name in enumerate(fwd["outputs"]):
            if name in fwd["values"]:
                fwd["values"][name]["output_roles"] = [r for r in fwd["values"][name].get("output_roles", [])
                                                      if not r.startswith("region_output:")] + [f"region_output:{idx}"]
    return data, {"merged": sorted(merged), "tape_dropped": sorted(tape_names[k] for k in dropped)}


__all__ = ["cse_saved_body"]
