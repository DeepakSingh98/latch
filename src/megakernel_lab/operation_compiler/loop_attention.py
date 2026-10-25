"""P-OPS4-E: short BF16-boundary attention inside differentiated saved-body loops.

A saved-body loop splits PyTorch's joint primal/VJP graph into a forward body region and a
reverse region joined by an explicit per-iteration tape (ops/loop.py). The P-OPS3-B short
attention rule (ops/short_attention.py) needs its forward phase to hand the FP32 output O and the
row log-sum-exp to its backward phase, and it never materializes the probabilities P. Inside a
loop that hand-off must cross the tape. This module makes it a compiler-visible tape decision:

Capture (``rewrite_joint``), for each attention group of the joint graph that the short rule
admits by structure, dtype, shape and stride:

* the forward gains explicit row statistics ``lse = m + log(sum(exp(x - m)))`` of the softmax
  input x (m = row max), ordinary ATen operations;
* every reverse use of P reads ``exp(x' - lse)``, where x' recomputes the scaled, masked scores
  from the same Q and K^T values in the reverse region;
* ``_softmax_backward_data(dP, P)`` becomes ``P * (dP - rowsum(dO * O))``, mathematically equal
  (sum_j P_j dP_j = sum_d dO_d O_d) and the form the fused backward uses for its key columns.

The partitioner is told to save ``lse`` and O (``forced_save``) and to recompute the score chain
and the Q/K/V head-layout copies/widenings in reverse (``forced_remat``), so the tape holds the
BF16 projection products, O and lse instead of P and FP32 head copies. P is no longer a tape
value; nothing pretends a saved value is dead. The rewritten graph remains an ordinary, exactly
interpretable ATen program (CPU graph_step and the reference lowering execute it as written).

Lowering (``region_plans``): when the regions carry this metadata, sm_90 auto lowering installs
the short forward phase in the forward body (writing the BF16 output in its consumer's layout,
O as FP32 and lse directly into their tape slots) and the short backward phase in the reverse
region (reading the tape's BF16 sources, O and lse). A region that cannot install a group keeps
the rewritten graph's ordinary lowering with a recorded reason; semantics never change.
"""
from __future__ import annotations

import math

from . import attention_lower, short_attention
from .short_attention import Refused

SCHEMA = "pops4e-loop-attention-tape-v1"
SOURCE_COMMIT = "P-OPS4-E (P-OPS3-B e68b4986b work items)"


def _names(joint):
    return set(joint.values)


def _output_name(op):
    return op.outputs[0]


def _static_layouts(values):
    from .lower import TensorLayout
    return {name: TensorLayout(tuple(int(x) for x in value.shape), tuple(int(x) for x in value.strides), name)
            for name, value in values.items()}


def _chain_nodes(graph, value):
    """Names of the view/clone/widening chain from an FP32 attention operand to its BF16 source."""
    _, chain = short_attention._input_chain(graph, value)
    return chain


def rewrite_joint(joint, module, *, splits=short_attention.DEFAULT_SPLITS,
                  max_sequence=short_attention.MAX_SEQUENCE, long=False):
    """Rewrite admitted attention groups of a joint primal/VJP graph (FX module, in place).

    Returns (records, refusals, forced_remat, forced_save). ``records`` describe the rewritten
    groups by value name; an empty list leaves the module untouched.
    """
    import torch
    from .lower import _field, _ref
    names = _names(joint)
    groups, refusals = attention_lower.plan(joint.operations, joint.values, names, ref=_ref, field=_field,
                                            output_name=_output_name, protected=set(joint.outputs),
                                            allow_rounding=True)
    groups = [g for g in groups if g.get("backward")]
    graph_args = (joint.operations, joint.values, names, _ref, _field, _output_name)
    plans, refused = short_attention.admit(groups, graph_args, _static_layouts(joint.values),
                                           set(joint.outputs), splits=splits, max_sequence=max_sequence)
    if long:
        # P7-C: groups refused only for their context length take the long rule (same tape contract).
        from . import long_attention
        remaining = [g for g in groups if g["label"] not in plans]
        long_plans, long_refused = long_attention.admit(remaining, graph_args, _static_layouts(joint.values),
                                                        set(joint.outputs))
        plans.update(long_plans)
        refused = [r for r in refused if r["group"] not in long_plans] + long_refused
    refusals = [*refusals, *refused]
    if not plans:
        return [], refusals, set(), set()
    graph = module.graph
    nodes = {node.name: node for node in graph.nodes}
    sgraph = short_attention._Graph(*graph_args)
    aten = torch.ops.aten
    records, forced_remat, forced_save = [], set(), set()
    for label, plan in plans.items():
        group = plan["group"]
        forward = list(group["forward"])
        p, o = group["p"], forward[-1]
        chain = list(group["scores"])  # score product, optional value identity, optional scale/mask addition
        x = chain[-1]
        pt, dv, dp, ds2 = group["backward"][:4]
        p_node, x_node = nodes[p], nodes[x]
        tag = f"pops4e_{label}"

        def make(target, args, suffix):
            return graph.create_node("call_function", target, tuple(args), {}, name=f"{tag}_{suffix}")
        # Forward statistics, immediately after the softmax (its input is already available).
        with graph.inserting_before(p_node.next):
            m = make(aten.amax.default, (x_node, [-1], True), "max")
            centered = make(aten.sub.Tensor, (x_node, m), "centered")
            e = make(aten.exp.default, (centered,), "exp")
            total = make(aten.sum.dim_IntList, (e, [-1], True), "sum")
            logged = make(aten.log.default, (total,), "log")
            lse = make(aten.add.Tensor, (m, logged), "lse")
        # Reverse recomputation of P from the same Q, K^T, scale and mask, before its first reverse use.
        first = min((nodes[n] for n in group["backward"]), key=lambda node: list(graph.nodes).index(node))
        recompute = {}
        with graph.inserting_before(first):
            for index, name in enumerate(chain):
                source = nodes[name]
                args = tuple(recompute.get(a.name, a) if isinstance(a, torch.fx.Node) else a for a in source.args)
                recompute[name] = graph.create_node("call_function", source.target, args, dict(source.kwargs),
                                                    name=f"{tag}_score{index}")
            shifted = make(aten.sub.Tensor, (recompute[x], lse), "shifted")
            p_reverse = make(aten.exp.default, (shifted,), "p")
            p_round_reverse = []
            if group.get("p_round"):
                # P8-B: dV reads the same declared BF16 rounding of the recomputed probabilities.
                narrow_r = graph.create_node("call_function", aten._to_copy.default, (p_reverse,),
                                             {"dtype": torch.bfloat16}, name=f"{tag}_p_bf16")
                wide_r = graph.create_node("call_function", aten._to_copy.default, (narrow_r,),
                                           {"dtype": torch.float32}, name=f"{tag}_p_round")
                p_round_reverse = [narrow_r, wide_r]
        if p_round_reverse:
            nodes[pt].replace_input_with(nodes[group["pv"]], p_round_reverse[-1])
        else:
            nodes[pt].replace_input_with(p_node, p_reverse)
        # Softmax derivative with the row term from dO and O.
        ds_node = nodes[ds2]
        dp_node, do_node, o_node = nodes[dp], nodes[group["do"]], nodes[o]
        with graph.inserting_before(ds_node):
            product = make(aten.mul.Tensor, (do_node, o_node), "do_o")
            delta = make(aten.sum.dim_IntList, (product, [-1], True), "delta")
            difference = make(aten.sub.Tensor, (dp_node, delta), "dp_delta")
            ds = make(aten.mul.Tensor, (p_reverse, difference), "ds")
        ds_node.replace_all_uses_with(ds)
        graph.erase_node(ds_node)
        remat = ([node.name for node in recompute.values()] + [shifted.name, p_reverse.name]
                 + [node.name for node in p_round_reverse])
        operand_chains = {key: _chain_nodes(sgraph, group[key]) for key in ("q", "kt", "v")}
        forced_remat.update(remat)
        forced_remat.update(n for chain_names in operand_chains.values() for n in chain_names)
        forced_save.update((lse.name, o))
        ds0 = group["backward"][4] if group["scale_mode"] else None
        records.append({
            "label": label, "q": group["q"], "kt": group["kt"], "v": group["v"], "mask": group["mask"],
            "do": group["do"], "k_view": group["k_view"], "qt_view": group["qt_view"],
            "vt_view": group["vt_view"], "scale": group["scale"], "scale_mode": group["scale_mode"],
            "forward": {"scores": chain, "p": p, "o": o,
                        "stats": [m.name, centered.name, e.name, total.name, logged.name, lse.name],
                        "p_round": list(group.get("p_round_nodes", []))},
            "lse": lse.name, "o": o,
            "reverse": {"recompute": remat, "p": p_reverse.name, "pt": pt, "dv": dv, "dp": dp,
                        "delta": [product.name, delta.name, difference.name], "ds": ds.name,
                        "ds_scaled": ds0 or ds.name, "dq": group["dq"], "dkt": group["dkt"],
                        "ds_round": list(group.get("ds_round_nodes", []))},
            "bf16_operands": bool(group.get("p_round")),
            "operand_chains": operand_chains, "head_width": plan["hd"], "sequence": plan["t"],
            "head_batch": plan["bh"], "heads": plan["heads"], "splits": plan["splits"],
            **({"kernel": "long"} if plan.get("kernel") == "long" else {})})
    graph.lint()
    module.recompile()
    _propagate(module)
    return records, refusals, forced_remat, forced_save


def _propagate(module):
    """Recompute fake tensor metadata of every node from the placeholders' fake values."""
    from torch._guards import detect_fake_mode
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp
    placeholders = [node.meta["val"] for node in module.graph.nodes if node.op == "placeholder"]
    mode = detect_fake_mode(placeholders)
    FakeTensorProp(module, mode).propagate_dont_convert_inputs(*placeholders)


# --------------------------------------------------------------------------------- region lowering
def _operand(graph, layouts, key, value):
    source, chain, axes = short_attention._input_map(graph, layouts, value)
    return source, chain, axes


def _transposed(axes):
    return [axes[0], axes[2], axes[1]]


def _resolve_heads(rows, outputs, bh):
    """Common (batch, head) decomposition of operand/result layouts, as short_attention.finalize."""
    candidates = short_attention._heads_candidates([axes for pairs in rows.values() for _, _, axes in pairs]
                                                   + [axes for axes in outputs.values()])
    error = None
    for heads in candidates:
        if bh % heads:
            continue
        try:
            strides = {}
            for key, pairs in rows.items():
                found = {tuple(short_attention._strides4(axes, bh, heads, name)) for name, _, axes in pairs}
                owners = {source for _, source, _ in pairs}
                if len(found) != 1 or len(owners) != 1:
                    raise Refused(f"{key}: operands read different views")
                strides[key] = (owners.pop(), found.pop())
            out = {key: short_attention._strides4(axes, bh, heads, key) for key, axes in outputs.items()}
            return heads, strides, out
        except Refused as exc:
            error = exc
    raise error or Refused("no consistent batch/head decomposition")


def region_plans(operations, values, names, metadata, *, ref, field, output_name, protected, inputs,
                 excluded=()):
    """Install plans for this region's halves of rewritten tape groups. Returns (plans, refusals)."""
    tape = (metadata or {}).get("attention_tape") or {}
    if tape.get("schema") != SCHEMA:
        return [], []
    graph_args = (operations, values, names, ref, field, output_name)
    graph = short_attention._Graph(*graph_args)
    layouts = _static_layouts(values)
    present = set(graph.ops)
    plans, refusals = [], []
    for record in tape.get("groups", []):
        label = record["label"]
        if label in excluded:
            continue
        try:
            if record["o"] in present and record["lse"] in present:
                plans.append(_forward_plan(graph, layouts, record, protected))
            elif record["reverse"]["ds"] in present:
                plans.append(_backward_plan(graph, layouts, record, protected, inputs))
        except Refused as exc:
            refusals.append({"group": label, "reason": "loop short attention: " + str(exc)})
    return plans, refusals


def _check_removed(graph, removed, skip, protected):
    for name in removed:
        if name in protected:
            raise Refused(f"{name} is returned or committed")
        outside = [u for u in graph.consumers.get(name, []) if u not in skip]
        if outside:
            raise Refused(f"{name} is used outside the attention region by {outside}")


def _common(record):
    bh, t, hd = record["head_batch"], record["sequence"], record["head_width"]
    return bh, t, hd


def _forward_plan(graph, layouts, record, protected):
    bh, t, hd = _common(record)
    fwd = record["forward"]
    o, lse = record["o"], record["lse"]
    members = set(fwd["scores"]) | {fwd["p"], o} | set(fwd["stats"]) | {record["kt"]} | set(fwd.get("p_round", []))
    if graph.shape(o) != (bh, t, hd) or graph.dtype(o) != "float32" or graph.strides(o) != (t * hd, hd, 1):
        raise Refused(f"{o}: FP32 output must be contiguous [{bh}, {t}, {hd}]")
    if graph.shape(lse) != (bh, t, 1) or graph.dtype(lse) != "float32":
        raise Refused(f"{lse}: statistics must be FP32 [{bh}, {t}, 1]")
    if o not in protected or lse not in protected:
        raise Refused("forward statistics are not tape outputs of this region")
    maps, chains = {}, {}
    for key in ("q", "kt", "v"):
        source, chain, axes = _operand(graph, layouts, key, record[key])
        maps[key], chains[key] = (key, source, axes), chain
    rows = {"q": [maps["q"]], "k": [("kt", maps["kt"][1], _transposed(maps["kt"][2]))], "v": [maps["v"]]}
    out_chain = short_attention._output_chain(graph, o, members)
    out_axes = short_attention._output_map(graph, o, ("bh", "i", "d"), out_chain)
    heads, strides, out = _resolve_heads(rows, {"o": out_axes}, bh)
    for key, (source, (sb, sh, sr)) in strides.items():
        if any(x % 8 for x in (sb, sh, sr)):
            raise Refused(f"{key}: BF16 rows must be 16-byte aligned")
    if any(x % 2 for x in out["o"]):
        raise Refused("o: BF16 output pairs must be 4-byte aligned")
    chain_names = {n for c in chains.values() for n in c}
    removed = (members - {o, lse}) | chain_names | set(out_chain[:-1])
    skip = removed | {o, lse, out_chain[-1]}
    _check_removed(graph, removed, skip, protected)
    group = {"label": record["label"], "o": o, "mask": record["mask"], "scale": record["scale"],
             "scale_mode": record["scale_mode"]}
    return dict(which="forward", label=record["label"], group=group, bh=bh, t=t, hd=hd, heads=heads,
                splits=record["splits"], kernel=record.get("kernel", "short"), sources={k: v[0] for k, v in strides.items()},
                in_strides={k: v[1] for k, v in strides.items()}, finals={"o": out_chain[-1]},
                out_strides=out, aux={"o32": o, "lse": lse}, skip=skip, removed=removed,
                anchor=min((o, *fwd["stats"]), key=graph.order.__getitem__),
                writes=[out_chain[-1], o, lse])


def _backward_plan(graph, layouts, record, protected, inputs):
    bh, t, hd = _common(record)
    rev = record["reverse"]
    o, lse = record["o"], record["lse"]
    for name in (o, lse):
        if name not in inputs:
            raise Refused(f"{name}: tape statistic is not an input of the reverse region")
    if layouts[o].shape != (bh, t, hd) or layouts[o].strides != (t * hd, hd, 1) or graph.dtype(o) != "float32":
        raise Refused(f"{o}: tape O must be contiguous FP32")
    if layouts[lse].shape != (bh, t, 1) or graph.dtype(lse) != "float32" or not layouts[lse].contiguous:
        raise Refused(f"{lse}: tape lse must be contiguous FP32")
    members = (set(rev["recompute"]) | {rev["p"], rev["pt"], rev["dv"], rev["dp"], rev["ds"], rev["ds_scaled"],
               rev["dq"], rev["dkt"]} | set(rev["delta"]) | set(rev.get("ds_round", []))
               | {record["kt"], record["k_view"], record["qt_view"], record["vt_view"]})
    maps = {}
    for key in ("q", "kt", "v", "do", "k_view", "qt_view", "vt_view"):
        source, chain, axes = _operand(graph, layouts, key, record[key])
        maps[key] = (source, chain, axes)
    rows = {"q": [("q", maps["q"][0], maps["q"][2]), ("qt_view", maps["qt_view"][0], _transposed(maps["qt_view"][2]))],
            "k": [("kt", maps["kt"][0], _transposed(maps["kt"][2])), ("k_view", maps["k_view"][0], maps["k_view"][2])],
            "v": [("v", maps["v"][0], maps["v"][2]), ("vt_view", maps["vt_view"][0], _transposed(maps["vt_view"][2]))],
            "do": [("do", maps["do"][0], maps["do"][2])]}
    outs = {"dq": (rev["dq"], ("bh", "i", "d")), "dkt": (rev["dkt"], ("bh", "d", "i")), "dv": (rev["dv"], ("bh", "i", "d"))}
    out_chains, out_maps = {}, {}
    for key, (value, labels) in outs.items():
        chain = short_attention._output_chain(graph, value, members)
        axes = short_attention._output_map(graph, value, labels, chain)
        out_chains[key], out_maps[key] = chain, (_transposed(axes) if key == "dkt" else axes)
    heads, strides, out = _resolve_heads(rows, out_maps, bh)
    for key, (source, (sb, sh, sr)) in strides.items():
        if any(x % 8 for x in (sb, sh, sr)):
            raise Refused(f"{key}: BF16 rows must be 16-byte aligned")
    for key, (sb, sh, sr) in out.items():
        if any(x % 2 for x in (sb, sh, sr)):
            raise Refused(f"{key}: BF16 output pairs must be 4-byte aligned")
    chain_names = {n for _, chain, _ in maps.values() for n in chain}
    removed = members | chain_names | {n for c in out_chains.values() for n in c[:-1]}
    skip = removed | {c[-1] for c in out_chains.values()}
    _check_removed(graph, removed, skip, protected)
    group = {"label": record["label"], "o": o, "mask": record["mask"], "scale": record["scale"],
             "scale_mode": record["scale_mode"]}
    finals = {key: chain[-1] for key, chain in out_chains.items()}
    return dict(which="backward", label=record["label"], group=group, bh=bh, t=t, hd=hd, heads=heads,
                splits=record["splits"], kernel=record.get("kernel", "short"), sources={k: v[0] for k, v in strides.items()},
                in_strides={k: v[1] for k, v in strides.items()}, finals=finals, out_strides=out,
                aux={"o32": o, "lse": lse}, skip=skip, removed=removed,
                anchor=min((rev["dv"], rev["dq"], rev["dkt"]), key=graph.order.__getitem__),
                writes=[finals["dq"], finals["dkt"], finals["dv"]])


def emit(plan, layouts, ident, mask_strides, *, fast_exp=False):
    """Phase code and resources; the P-OPS3-B emission with tape-bound statistics (P7-C: long-context body)."""
    if plan.get("kernel") == "long":
        from . import long_attention
        return long_attention.emit(plan, layouts, ident, mask_strides, which=plan["which"], aux=plan["aux"], fast_exp=fast_exp)
    kind, code, extra = short_attention.emit(plan, layouts, ident, mask_strides, which=plan["which"],
                                             aux=plan["aux"])
    extra.pop("prefetch", None)
    return "attention_short_" + plan["which"], code, extra


def reads(plan):
    used = ("q", "k", "v") if plan["which"] == "forward" else ("q", "k", "v", "do")
    result = [plan["sources"][key] for key in used]
    if plan["which"] == "backward":
        result += [plan["aux"]["o32"], plan["aux"]["lse"]]
    if plan["group"].get("mask") is not None:
        result.append(plan["group"]["mask"])
    return result


def shared_estimate(hd, t, splits):
    tp = (t + 15) // 16 * 16
    pieces = 2 * (2 * splits * tp * 24 + splits * 16 * (tp + 8))
    return (2 * 4 * tp * (hd + 8) + max(pieces, 4 * tp * (hd + 4))
            + 4 * ((tp * 20 + 16 * (tp + 4) if tp <= 96 else 0) + 2 * tp + (tp // 8) * 16))


__all__ = ["SCHEMA", "emit", "reads", "region_plans", "rewrite_joint"]
