"""LATCH-STEP-06: shared-weight gradient accumulation leaves the reverse loop body (exact).

MODE 'tape' (default): every reverse step keeps its weight-gradient product, but the product stores its BF16
result bf16(A_r^T B_r) into a per-iteration reverse-tape slot instead of read-modify-writing the FP32 running
gradient; one elementwise post-loop stage adds the slots in execution order (G = G + float(P_r), r = count-1 .. 0),
optionally fused with the elementwise optimizer update. Same IEEE operations per element as the in-body epilogue.

MODE 'batch' (measured, slower: the post-loop products are L2-bandwidth bound) is described below.

In a saved-tape recurrent update every reverse step accumulates each shared weight's gradient contribution
fp32(bf16(A_r^T @ B_r)) into the running FP32 gradient: A_r is a saved forward tape entry of iteration r, B_r a
BF16 cotangent computed by that reverse step. Those products are on the reverse loop's critical levels although no
later reverse step reads them.

This pass removes the product chain ``clone(_to_copy[f32](mm(t(A_tape), B)))`` of such a gradient output from the
reverse body and exposes B as a new per-iteration reverse-tape output instead. The loop composer
(``loop_pipeline``) then evaluates every deferred contribution once per update, after the last reverse step, in the
original order: for each output tile the running gradient is loaded once, then for execution steps e = 0..count-1
(iteration r = count-1-e) the same BF16 x BF16 -> FP32 wgmma k-sequence over that iteration's T rows is formed from
+0, rounded once to BF16, widened and added in FP32 (``kernels/step06_dgrad.cuh``). Every element therefore sees the
identical sequence of IEEE operations as the per-step read-modify-write epilogue it replaces: the result is bitwise
equal. Runtime counts, full-history gradients and recomputation behaviour are unchanged; the cost is one BF16
cotangent slot per deferred weight and iteration.
"""
from __future__ import annotations

import json

ENABLED = True
MODE = "tape"        # 'tape': BF16 per-iteration products + ordered post-loop sum; 'batch': post-loop products
ZERO_INIT = True     # elide the terminal +0 fill of a deferred gradient; the deferred stage starts from +0 (exact)
FUSE_OPTIMIZER = True    # g-03: 6.173 vs 6.190 ms (bitwise). Apply an elementwise optimizer update of a +0-initialized deferred weight in the stage epilogue
_OPT_TARGETS = {"aten.mul.Tensor": "*", "aten.add.Tensor": "+", "aten.sub.Tensor": "-"}


def _optimizer_chain(data, md, k):
    """The elementwise update of parameter k in the optimizer region as an ordered expression list, or None.

    Only mul/add/sub of the parameter, its optimizer state and its gradient with FP32 scalars are admitted; the
    expressions use the operand order and form of the generated elementwise code (``a + (b * 1.0f)``)."""
    opt = data["regions"].get("optimizer")
    if not opt or "operations" not in opt:
        return None
    names = md["parameter_names"]
    states = md.get("optimizer_state_names", names)
    labels = dict(zip(opt.get("metadata", {}).get("input_labels") or [], opt["inputs"]))
    np_ = len(names)
    if len(opt["outputs"]) != np_ + len(states) or names[k] not in states:
        return None
    j = states.index(names[k])
    leaves = {labels.get(f"parameter:{names[k]}"): "p", labels.get(f"momentum:{names[k]}"): "m",
              labels.get(f"gradient:{names[k]}"): "g"}
    if None in leaves:
        return None
    outs = {opt["outputs"][k]: "p_new", opt["outputs"][np_ + j]: "m_new"}
    if len(outs) != 2:
        return None
    prod = {o: op for op in opt["operations"] for o in op["outputs"]}
    users = {}
    for op in opt["operations"]:
        for r in _refs(op["args"]):
            users.setdefault(r, set()).add(op["name"])
    chain, seen = [], set()

    def visit(name):
        if name in leaves or name in seen:
            return True
        op = prod.get(name)
        if op is None or op["target"] not in _OPT_TARGETS or op.get("kwargs") or len(op["args"]) != 2:
            return False
        for a in op["args"]:
            if isinstance(a, dict):
                if not visit(a["ref"]):
                    return False
            elif not isinstance(a, (int, float)) or isinstance(a, bool):
                return False
        seen.add(name)
        chain.append(op)
        return True

    if not all(visit(o) for o in outs):
        return None
    inside = {op["name"] for op in chain}
    for op in chain:                      # no value of the chain is read outside it (other than as an output)
        out = op["outputs"][0]
        if any(u not in inside for u in users.get(out, ())) or (out in opt["outputs"] and out not in outs):
            return None
    order = {op["name"]: i for i, op in enumerate(opt["operations"])}
    chain.sort(key=lambda op: order[op["name"]])
    exprs = []
    for op in chain:
        a, b = op["args"]
        exprs.append({"out": op["outputs"][0], "op": _OPT_TARGETS[op["target"]],
                      "a": leaves.get(a["ref"], a["ref"]) if isinstance(a, dict) else float(a),
                      "b": leaves.get(b["ref"], b["ref"]) if isinstance(b, dict) else float(b)})
    return {"p_arg": k, "m_arg": np_ + j, "exprs": exprs, "outputs": {v: n for n, v in outs.items()}}


def _refs(x):
    if isinstance(x, dict):
        if set(x) == {"ref"}:
            return [x["ref"]]
        return [r for v in x.values() for r in _refs(v)]
    if isinstance(x, list):
        return [r for v in x for r in _refs(v)]
    return []


def _contiguous(value):
    shape, strides = list(value.get("shape") or ()), list(value.get("strides") or ())
    expect, s = [], 1
    for d in reversed(shape):
        expect.append(s)
        s *= int(d)
    return strides == list(reversed(expect))


def defer_weight_gradients(data):
    if not ENABLED or data.get("storage_mode") != "save_body_tape":
        return data, {"deferred": []}
    regions, md = data.get("regions", {}), data.get("metadata", {})
    rev = regions.get("body_vjp_saved")
    tape = md.get("body_tape") or []
    if not rev or "operations" not in rev or md.get("sequence_specs") or md.get("deferred_grads"):
        return data, {"deferred": [], "refused": "plain saved-tape reverse body required"}
    np_ = len(md["parameter_names"])
    labels = rev.get("metadata", {}).get("input_labels") or []
    label_of = dict(zip(rev["inputs"], labels))
    tape_index = {t["name"]: i for i, t in enumerate(tape)}
    ops = rev["operations"]
    producer = {o: op for op in ops for o in op["outputs"]}
    readers = {}
    for op in ops:
        for r in set(op["reads"]) | set(_refs(op["args"])) | set(_refs(op.get("kwargs", {}))):
            readers.setdefault(r, []).append(op["name"])
    outputs = rev["outputs"]
    out_uses = {}
    for o in outputs:
        out_uses[o] = out_uses.get(o, 0) + 1
    vals = rev["values"]
    found = []
    for k in range(np_):
        out = outputs[k]
        clone = producer.get(out)
        if not clone or clone["target"] != "aten.clone.default" or out_uses.get(out) != 1 or readers.get(out):
            continue
        (cast_name,) = _refs(clone["args"]) or [None]
        cast = producer.get(cast_name)
        if (not cast or cast["target"] != "aten._to_copy.default" or vals[cast_name].get("dtype") != "float32"
                or len(readers.get(cast_name, [])) != 1 or cast_name in out_uses):
            continue
        refs = _refs(cast["args"])
        if len(refs) != 1:
            continue
        mm = producer.get(refs[0])
        if (not mm or mm["target"] != "aten.mm.default" or vals[refs[0]].get("dtype") != "bfloat16"
                or len(readers.get(refs[0], [])) != 1 or refs[0] in out_uses or mm.get("kwargs")):
            continue
        a_t, b = _refs(mm["args"])
        t = producer.get(a_t)
        if not t or t["target"] != "aten.t.default" or len(readers.get(a_t, [])) != 1 or a_t in out_uses:
            continue
        (a,) = _refs(t["args"])
        label = label_of.get(a, "")
        if not label.startswith("tape:") or label[5:] not in tape_index:
            continue
        av, bv, wv = vals[a], vals[b], vals[refs[0]]
        if (av.get("dtype") != "bfloat16" or bv.get("dtype") != "bfloat16" or len(av["shape"]) != 2
                or len(bv["shape"]) != 2 or not _contiguous(av) or not _contiguous(bv) or b in rev["inputs"]
                or av["shape"][0] != bv["shape"][0]):
            continue
        T, M, N = int(av["shape"][0]), int(av["shape"][1]), int(bv["shape"][1])
        if [M, N] != list(wv["shape"]) or M % 128 or N % 128 or T % 64:
            continue
        if MODE == "batch":
            # Only products the in-body schedule runs on the p5c warpgroup engine (same k16 association as the
            # deferred body); smaller products keep their in-body mma.sync bodies.
            from .p5c_matrix import admit
            if admit(M, N, T, 1, False, False, 0, 0) != "p5c":
                continue
        elif (M * N) % 4:
            continue
        found.append({"param": k, "output": out, "product": refs[0],
                      "ops": [clone["name"], cast["name"]] + ([mm["name"], t["name"]] if MODE == "batch" else []),
                      "values": [out, cast_name] + ([refs[0], a_t] if MODE == "batch" else []),
                      "tape": label[5:], "tape_index": tape_index[label[5:]], "operand": b,
                      "M": M, "N": N, "T": T})
    if not found:
        return data, {"deferred": []}
    data = json.loads(json.dumps(data))
    rev, md = data["regions"]["body_vjp_saved"], data["metadata"]
    drop_ops = {name for f in found for name in f["ops"]}
    rev["operations"] = [op for op in rev["operations"] if op["name"] not in drop_ops]
    deferred_params = {f["param"] for f in found}
    old = rev["outputs"]
    operands = []
    for f in found:
        key = f["operand"] if MODE == "batch" else f["product"]
        if key not in operands:
            operands.append(key)
    # Values of the removed chain (clone output, FP32 cast, BF16 product, transposed tape view). The tape input
    # stays bound: the deferred stage reads the saved tape slots directly.
    for f in found:
        for name in f["values"]:
            rev["values"].pop(name, None)
    rev["outputs"] = [o for k, o in enumerate(old) if not (k < len(md["parameter_names"]) and k in deferred_params)]
    rev["outputs"] += operands
    for idx, out in enumerate(rev["outputs"]):
        v = rev["values"].get(out)
        if v is not None:
            v["output_roles"] = [r for r in v.get("output_roles", []) if not r.startswith("region_output:")] + [
                f"region_output:{idx}"]
    # A gradient whose terminal-VJP contribution is an explicit +0 fill (zeros_like) starts from +0 in registers:
    # the composer elides that fill (``zero_init``) and the deferred stage stores every element, also for count 0.
    def producers(region):
        return {o: op for op in (region or {}).get("operations", []) for o in op["outputs"]}

    term = data["regions"].get("terminal_vjp") or {}
    init_vjp = data["regions"].get("init_vjp_saved") or data["regions"].get("init_vjp") or {}
    tprod, iprod = producers(term), producers(init_vjp)

    def zero_fill(prod, name, depth=0):
        op = prod.get(name)
        if op is None or depth > 2:
            return False
        if op["target"] in ("aten.zeros_like.default", "aten.new_zeros.default", "aten.zeros.default"):
            return True
        return (op["target"] == "aten.clone.default" and len(_refs(op["args"])) == 1
                and zero_fill(prod, _refs(op["args"])[0], depth + 1))

    entries = []
    for f in found:
        k = f["param"]
        touts = term.get("outputs", [])
        zero = (ZERO_INIT and 1 + k < len(touts) and touts.count(touts[1 + k]) == 1 and zero_fill(tprod, touts[1 + k]))
        entry = {"param": k, "name": md["parameter_names"][k], "tape": f["tape"],
                 "tape_index": f["tape_index"], "operand": f["operand"],
                 "operand_index": operands.index(f["operand"] if MODE == "batch" else f["product"]),
                 "M": f["M"], "N": f["N"], "T": f["T"], "zero_init": bool(zero)}
        iouts = init_vjp.get("outputs", [])
        if zero and FUSE_OPTIMIZER and k < len(iouts) and zero_fill(iprod, iouts[k]):
            chain = _optimizer_chain(data, md, k)
            if chain is not None:
                entry["optimizer"] = chain
        entries.append(entry)
    md["deferred_grads"] = {
        "version": 1, "mode": MODE, "params": sorted(deferred_params), "entries": entries,
        "reverse_tape": [{"name": n, "shape": list(rev["values"][n]["shape"]), "dtype": "bfloat16"} for n in operands],
        "arithmetic": "per iteration fp32(bf16_rn(A_r^T B_r)) with A_r, B_r BF16 and an FP32 wgmma k-sequence from +0; "
                      "added in FP32 to the running gradient in execution order (r = count-1 .. 0): bitwise equal "
                      "to the in-body read-modify-write epilogue"}
    return data, {"deferred": [e["name"] for e in entries], "operands": operands}


__all__ = ["defer_weight_gradients"]
