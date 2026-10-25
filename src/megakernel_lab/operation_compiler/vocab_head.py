"""Large-vocabulary prediction heads: padded execution storage and fused row cross-entropy.

Release step 3 (LATCH-STEP-03). A graph rewrite applied by ``lower`` before schedule selection,
plus three indexed operation emitters reached through ``transformer_lower.emit_operation``
(shared by every flat schedule). Selection reads only operation targets, shapes, dtypes and
dataflow; it never inspects module, parameter or value names.

Recognized dataflow (both the explicit ``-log_softmax(z).gather(...)`` form and
``F.cross_entropy``'s captured nll decomposition reach it after AD):

    Wc  = _to_copy(W, bf16)                      (or W itself for FP32 products)
    P   = mm(A, view(Wc))           [R, V]       logits product
    L   = _to_copy(P, f32)                       (absent for FP32 products)
    LS  = _log_softmax(L, -1)
    y   = gather(LS, -1, idx)                    forward loss rows
    G   = chain(scatter*(zeros, -1, idx', s))    one nonzero per row (mul/div/neg by row factors)
    GL  = _log_softmax_backward_data(G, LS, -1)
    dP  = _to_copy(GL, bf16)                     (absent for FP32 products)
    ... = mm(dP, view(Wc)), mm(view(A), dP), ... ordinary AD products

Rewrite (exact up to the declared reduction order of the row sum):

* Execution storage of width ``Vp = ceil(V / 128) * 128``. ``Wp = vocab_pad(W)`` is the cast with zero
  columns/rows beyond V; every product reading the vocabulary dimension is re-emitted on padded
  twins, so the shared planner sees M/N/K multiples of 128 and 16-byte leading strides. A padded
  contraction adds exact zero products. Each original value becomes a ``slice(twin, d, 0, V)`` view,
  so other consumers are unchanged.
* ``stats = vocab_stats(Pp)`` [R, 2]: row max and log of the shifted exponential sum over the first V
  columns only. Padding never becomes a probability class.
* ``y = vocab_pick(Pp, stats, idx)`` replaces the gather: ``(x_t - max) - logsum`` exactly as the
  log_softmax element formula.
* ``dPp = vocab_grad(Pp, stats, idx', leaf, factors...)`` replaces scatter/chain/backward/cast:
  ``g_j = [j == t] c - expf((x_j - max) - logsum) * c`` rounded once to the product dtype, with
  ``c`` the chain evaluated at the row's nonzero element (identical operation order) and exact zero in
  padded columns.

Differences from the unfused lowering: established legacy execution uses the fixed per-row
order (one CTA per row, eight-element thread chunks, CTA tree). Common execution uses the
declared online warp reassociation when ONLINE_STATS is enabled; a retention probe first
checks the fixed tree. Max, element formulas, BF16 rounding points and the gradient formula
are unchanged. Not admitted (left unchanged): other readers
of the logits/log-probabilities/scatter chain, non-last class dimensions, rank > 2 logits, bias
products (addmm), widths below ``MIN_VOCABULARY``.
"""
from __future__ import annotations

import copy
import math
from contextvars import ContextVar

MIN_VOCABULARY = 2048
PAD_MULTIPLE = 128
ROW_ILP = 8            # independent 16-byte row loads issued together per thread (stats and gradient passes); LATCH-MOESUP: 12 measured: head component 26.25 -> 25.30 ms but full aux update 139.7 -> 141.6 ms (match-c7), kept 8
ONLINE_STATS = True    # LATCH-MOESUP: common-planner one-pass warp statistics; legacy keeps the fixed CTA tree
# A retention probe must compare the established reduction, not claim that the
# online reassociation is bitwise equivalent. Common selections are re-emitted
# after the probe with their selected online body. Region compose bodies retain
# online statistics through the existing execution scope.
_STATS_PROBE = ContextVar("latch_vocab_stats_probe", default=None)
# LATCH-MOESUP-ROW: the online statistics loop keeps the raw 16-byte vectors of a lane's next ROW_ILP-vector group in
# flight while the current group is reduced (and the next row's first group during a row's tail and butterfly).
# Same per-lane element order and operations: bit-identical statistics. BF16 logits only.
VOCAB_PIPE = True
# LATCH-MOESUP-ROW: the gradient pass runs one warp per row with the same lookahead (standalone production shape:
# 6.47 -> 5.73 ms vs a 4.82 ms read+write roofline; bit-identical). BF16 logits only.
VOCAB_GRAD_WARP = True
GRAD_FLAT = False      # LATCH-MOESUP: flat chunk stream for the gradient pass; measured slower (head 26.3 -> 31.7 ms)
# Products with a vocabulary-scale extent (>= WIDE_EXTENT) and a short contraction stream every operand tile from L2
# far more often than they compute on it: a 128 x 128 tile reads (128 + 128) * K * 2 bytes per 2 * 128 * 128 * K
# flops (64 flop/byte, ~350 TF/s at H100 L2 bandwidth). The existing P5C 256 x 128 / 128 x 256 bodies halve the
# re-read stream. Restricting the rule to such extents leaves every existing program's selection unchanged.
WIDE_TILES = True
WIDE_EXTENT = 16384
TARGETS = {
    "pad": "latch_ops.vocab_pad.default",
    "stats": "latch_ops.vocab_stats.default",
    "pick": "latch_ops.vocab_pick.default",
    "grad": "latch_ops.vocab_grad.default",
}
_ZERO_FILLS = {"aten.zeros_like.default", "aten.new_zeros.default", "aten.zeros.default"}
_FULL_FILLS = {"aten.full_like.default", "aten.new_full.default", "aten.full.default"}
_TRANSPOSING_VIEWS = {"aten.t.default", "aten.transpose.int", "aten.permute.default",
                      "aten.alias.default", "aten.detach.default"}
_CHAIN = {"aten.mul.Tensor": "*", "aten.div.Tensor": "/", "aten.neg.default": "~"}


class Refused(Exception):
    pass


def padded(width):
    return -(-width // PAD_MULTIPLE) * PAD_MULTIPLE


# ----------------------------------------------------------------------------------------- graph rewrite
def rewrite(graph, mode="auto"):
    """Return (graph, report). The input is never mutated; unmatched graphs are returned as given.

    mode: "auto" rewrites widths >= MIN_VOCABULARY, "force" any width (small test fixtures), "off" nothing.
    """
    if mode not in ("auto", "force", "off"):
        raise ValueError("vocabulary_head must be auto, force or off")
    if mode == "off":
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    if "operations" not in data or "values" not in data:
        return graph, None
    if not any(op.get("target") == "aten._log_softmax.default" for op in data["operations"]):
        return graph, None
    report = {"version": 1, "rewritten": [], "refused": []}
    changed = False
    for op in list(data["operations"]):
        if op.get("target") != "aten._log_softmax.default" or op not in data["operations"]:
            continue
        try:
            plan = _match(data, op, mode)
        except Refused as exc:
            report["refused"].append({"log_softmax": op["name"], "reason": str(exc)})
            continue
        if plan is None:
            continue
        try:
            _apply(data, plan)
        except Refused as exc:
            # Application failures leave the whole program unchanged.
            report["refused"].append({"log_softmax": op["name"], "reason": "apply: " + str(exc)})
            return graph, report
        report["rewritten"].append(plan["record"])
        changed = True
    if not changed:
        if report["refused"]:
            return graph, report
        return graph, None
    data.setdefault("metadata", {})["vocabulary_head"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _index(data):
    producer, consumers = {}, {}
    for op in data["operations"]:
        for out in op.get("outputs") or [op["name"]]:
            producer[out] = op
        for name in _refs(op.get("args", [])) + _refs(op.get("kwargs", {})):
            consumers.setdefault(name, []).append(op)
    return producer, consumers


def _refs(item):
    found = []
    if isinstance(item, dict):
        if set(item) == {"ref"}:
            return [item["ref"]]
        for value in item.values():
            found += _refs(value)
    elif isinstance(item, (list, tuple)):
        for value in item:
            found += _refs(value)
    return found


def _outputs(data):
    raw = data.get("outputs", {})
    return set(_refs(raw)) | {v for v in (raw.values() if isinstance(raw, dict) else raw) if isinstance(v, str)} | \
        {v for group in (raw.values() if isinstance(raw, dict) else []) if isinstance(group, (list, tuple))
         for v in group if isinstance(v, str)}


def _only(consumers, name, outputs):
    users = consumers.get(name, [])
    if name in outputs or len(users) != 1:
        raise Refused(f"{name} has other readers")
    return users[0]


def _arg(op, i, key=None, default=None):
    args = op.get("args", [])
    if len(args) > i:
        return args[i]
    return op.get("kwargs", {}).get(key, default) if key else default


def _view_chain(name, producer, root):
    """Transposing view ops from ``root`` to ``name`` (empty if name is root), else None."""
    chain = []
    while name != root:
        op = producer.get(name)
        if op is None or op["target"] not in _TRANSPOSING_VIEWS:
            return None
        chain.append(op)
        name = _ref(op["args"][0])
    return list(reversed(chain))


def _match(data, ls_op, mode="auto"):
    values = data["values"]
    producer, consumers = _index(data)
    outputs = _outputs(data)
    ls = ls_op["name"]
    logits = _ref(ls_op["args"][0])
    shape = list(values[logits]["shape"])
    if len(shape) != 2 or int(ls_op["args"][1]) % 2 != 1 or bool(_arg(ls_op, 2, "half_to_float", False)):
        raise Refused("rank-2 last-dimension log_softmax required")
    rows, vocab = shape
    if vocab < MIN_VOCABULARY and mode != "force":
        return None
    if values[logits]["dtype"] != "float32" or values[ls]["dtype"] != "float32":
        raise Refused("FP32 logits required")
    _only(consumers, logits, outputs)
    cast_op = producer.get(logits)
    if cast_op is None:
        raise Refused("logits have no producer")
    if cast_op["target"] == "aten._to_copy.default":
        product = _ref(cast_op["args"][0])
        if values[product]["dtype"] != "bfloat16" or list(values[product]["shape"]) != shape:
            raise Refused("logits cast must widen a same-shape BF16 product")
        _only(consumers, product, outputs)
    else:
        cast_op, product = None, logits
    mm = producer.get(product)
    if mm is None or mm["target"] != "aten.mm.default":
        raise Refused("logits must come from a plain mm (no bias/addmm)")
    dtype = values[product]["dtype"]
    a, b = _ref(mm["args"][0]), _ref(mm["args"][1])
    if values[a]["dtype"] != dtype or values[b]["dtype"] != dtype:
        raise Refused("mixed product operand dtypes")
    # Root of the vocabulary operand: transposing views over a cast of a parameter, or the parameter.
    root = b
    while producer.get(root) is not None and producer[root]["target"] in _TRANSPOSING_VIEWS:
        root = _ref(producer[root]["args"][0])
    weight_op = producer.get(root)
    if weight_op is not None and weight_op["target"] == "aten._to_copy.default":
        source = _ref(weight_op["args"][0])
        if values[source]["dtype"] != "float32" or dtype != "bfloat16":
            raise Refused("vocabulary weight cast must narrow FP32 to BF16")
    elif weight_op is None and dtype == "float32":
        source, weight_op = root, None
    else:
        raise Refused("vocabulary operand must be a (viewed) weight or its BF16 cast")
    wshape = list(values[root]["shape"])
    if len(wshape) != 2 or wshape.count(vocab) < 1 or vocab not in wshape:
        raise Refused("vocabulary weight must be rank 2")
    # Which weight dimension is the vocabulary: follow the chain's permutation of shapes.
    vdim = _vocab_dim_of_root(values, producer, b, root, 1, vocab)
    if vdim is None:
        raise Refused("cannot map the vocabulary dimension through operand views")
    if weight_op is None and list(values[root]["strides"]) != _contiguous(wshape):
        # A computed cast is materialized contiguously by every lowering; an input keeps its bound strides.
        raise Refused("vocabulary weight input must be contiguous")
    # Forward readers of the log-probabilities.
    readers = consumers.get(ls, [])
    if ls in outputs or not readers:
        raise Refused("log-probabilities are a program output")
    gathers = [op for op in readers if op["target"] == "aten.gather.default"]
    backs = [op for op in readers if op["target"] == "aten._log_softmax_backward_data.default"]
    templates = [op for op in readers if op["target"] in _ZERO_FILLS | _FULL_FILLS]
    if len(gathers) + len(backs) + len(templates) != len(readers) or len(gathers) != 1 or len(backs) > 1:
        raise Refused("log-probabilities need exactly one gather and at most one backward reader")
    gather = gathers[0]
    if int(gather["args"][1]) % 2 != 1 or list(values[_ref(gather["args"][2])]["shape"]) != [rows, 1]:
        raise Refused("gather must select one class per row on the last dimension")
    if bool(_arg(gather, 3, "sparse_grad", False)):
        raise Refused("sparse gather")
    if templates and not backs:
        raise Refused("log-probability template without a backward reader")
    plan = {"rows": rows, "vocab": vocab, "vp": padded(vocab), "dtype": dtype, "product": product,
            "mm": mm, "a": a, "b": b, "root": root, "weight_op": weight_op, "source": source, "vdim": vdim,
            "cast_op": cast_op, "logits": logits, "ls_op": ls_op, "gather": gather, "back": None}
    if backs:
        back = backs[0]
        if _ref(back["args"][1]) != ls or int(back["args"][2]) % 2 != 1:
            raise Refused("backward must read these log-probabilities on the last dimension")
        grad = _ref(back["args"][0])
        leaf, factors, ops, chain_ops = _scatter_chain(values, producer, consumers, outputs, grad, rows, vocab)
        out = back["name"]
        if dtype == "bfloat16":
            down = _only(consumers, out, outputs)
            if down["target"] != "aten._to_copy.default" or values[down["name"]]["dtype"] != "bfloat16":
                raise Refused("BF16 logits need the gradient narrowed once to BF16")
            dlogits, dcast = down["name"], down
        else:
            dlogits, dcast = out, None
        if any(t is not leaf["zeros"] for t in templates):
            raise Refused("log-probabilities template a value other than the backward scatter destination")
        plan.update(back=back, leaf=leaf, factors=factors, chain=ops, chain_ops=chain_ops,
                    dlogits=dlogits, dcast=dcast)
    plan["record"] = {"rows": rows, "vocabulary": vocab, "padded": plan["vp"], "dtype": dtype,
                      "weight_vocabulary_dim": vdim, "backward": bool(backs),
                      "chain": plan.get("chain", ""), "logits_product": mm["name"]}
    return plan


def _contiguous(shape):
    strides, stride = [], 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    return list(reversed(strides))


def _vocab_dim_of_root(values, producer, name, root, dim, vocab):
    """Dimension of ``root`` that ``name``'s dimension ``dim`` views (transposing views only)."""
    while name != root:
        op = producer[name]
        target = op["target"]
        rank = len(values[name]["shape"])
        if target == "aten.t.default":
            dim = rank - 1 - dim if rank == 2 else dim
        elif target == "aten.transpose.int":
            x, y = int(op["args"][1]) % rank, int(op["args"][2]) % rank
            dim = y if dim == x else x if dim == y else dim
        elif target == "aten.permute.default":
            dim = [int(d) % rank for d in op["args"][1]][dim]
        name = _ref(op["args"][0])
    return dim if values[root]["shape"][dim] == vocab else None


def _scatter_chain(values, producer, consumers, outputs, grad, rows, vocab):
    """(leaf, factors, ops, chain_ops) for G = chain(scatter(zeros, -1, idx, s)); refuses anything else."""
    ops, factors, chain_ops = [], [], []
    name = grad
    if list(values[name]["shape"]) != [rows, vocab] or values[name]["dtype"] != "float32":
        raise Refused("upstream log-probability gradient must be FP32 [R, V]")
    while True:
        op = producer.get(name)
        if op is None:
            raise Refused("gradient has no producer")
        target = op["target"]
        if target in _CHAIN:
            if target == "aten.neg.default":
                inner, factor = _ref(op["args"][0]), None
            else:
                x, y = _ref(op["args"][0]), _ref(op["args"][1])
                big = [v for v in (x, y) if v is not None and list(values[v]["shape"]) == [rows, vocab]]
                if len(big) != 1:
                    raise Refused("chain step must combine the [R, V] gradient with a row-broadcast factor")
                inner = big[0]
                factor = y if inner == x else x
                if target == "aten.div.Tensor" and inner != x:
                    raise Refused("chain division must divide the sparse gradient")
                if factor is None or values[factor]["dtype"] != "float32":
                    raise Refused("chain factors must be FP32 tensors")
                fshape = list(values[factor]["shape"])
                padded_shape = [1] * (2 - len(fshape)) + fshape
                if len(fshape) > 2 or padded_shape[1] != 1 or padded_shape[0] not in (1, rows):
                    raise Refused("chain factor must broadcast along the vocabulary dimension")
            if name != grad:
                _only(consumers, name, outputs)
            ops.append(_CHAIN[target])
            factors.append(factor)
            chain_ops.append(op)
            name = inner
            continue
        if target in {"aten.scatter.value", "aten.scatter_add.default", "aten.scatter.src"}:
            if name != grad:
                _only(consumers, name, outputs)
            if int(op["args"][1]) % 2 != 1:
                raise Refused("scatter must address the last dimension")
            idx = _ref(op["args"][2])
            if list(values[idx]["shape"]) != [rows, 1] or values[idx]["dtype"] != "int64":
                raise Refused("one int64 class index per row required")
            zeros = _ref(op["args"][0])
            _only(consumers, zeros, outputs)
            if not _is_zero_fill(producer.get(zeros), values, rows, vocab):
                raise Refused("scatter destination must be a zero fill")
            if target == "aten.scatter.value":
                value = op["args"][3]
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise Refused("scalar scatter value required")
                leaf = {"value": float(value)}
            else:
                src = _ref(op["args"][3])
                if list(values[src]["shape"]) != [rows, 1] or values[src]["dtype"] != "float32":
                    raise Refused("scatter source must be FP32 [R, 1]")
                leaf = {"source": src, "add": target == "aten.scatter_add.default"}
            leaf.update(index=idx, scatter=op, zeros=producer[zeros])
            ops.reverse(); factors.reverse(); chain_ops.reverse()
            return leaf, factors, "".join(ops), chain_ops
        raise Refused(f"unsupported gradient producer {target}")


def _is_zero_fill(op, values, rows, vocab):
    if op is None or list(values[op["name"]]["shape"]) != [rows, vocab] or values[op["name"]]["dtype"] != "float32":
        return False
    if op["target"] in _ZERO_FILLS:
        return True
    if op["target"] in _FULL_FILLS:
        fill = op["args"][-1] if op["target"] != "aten.full.default" else op["args"][1]
        return isinstance(fill, (int, float)) and float(fill) == 0.0
    if op["target"] == "aten.fill.Scalar":
        return isinstance(op["args"][1], (int, float)) and float(op["args"][1]) == 0.0
    return False


# ----------------------------------------------------------------------------------------- application
def _new_value(data, name, shape, dtype, *, alias_of=None, strides=None, offset=0):
    if name in data["values"]:
        raise Refused("generated value name collision: " + name)
    data["values"][name] = {"name": name, "shape": list(shape), "dtype": dtype,
                            "strides": list(strides if strides is not None else _contiguous(shape)),
                            "role": "temporary", "parameter": None, "alias_of": alias_of,
                            "storage_offset": int(offset), "producer": name, "output_roles": []}


_GRAPH_INPUTS = ()


def _new_op(name, target, args, kwargs=None, *, view=False):
    refs = []
    for r in _refs(args) + _refs(kwargs or {}):
        if r not in refs:
            refs.append(r)
    return {"name": name, "target": target, "args": args, "kwargs": kwargs or {},
            "inputs": list(refs), "outputs": [name], "dependencies": [r for r in refs if r not in _GRAPH_INPUTS],
            "reads": list(refs),
            "writes": [] if view else [name], "reduction_axes": [],
            "synchronization": "view" if view else "global_phase_boundary", "effect": "pure"}


def _root_alias(values, name):
    while values[name].get("alias_of"):
        name = values[name]["alias_of"]
    return name


def _apply(data, plan):
    global _GRAPH_INPUTS
    _GRAPH_INPUTS = tuple(data.get("inputs", ()))
    values, operations = data["values"], data["operations"]
    vocab, vp, rows, dtype = plan["vocab"], plan["vp"], plan["rows"], plan["dtype"]
    tag = plan["mm"]["name"]
    fresh = lambda suffix: f"vocab_{suffix}_{tag}"
    position = {op["name"]: i for i, op in enumerate(operations)}
    inserts = {}       # index -> ops inserted before the op at that index
    replace = {}       # op name -> list of ops replacing it (same position)
    remove = set()

    def add_before(op_name, *ops):
        inserts.setdefault(position[op_name], []).extend(ops)

    def slice_view(name, twin, dim, after_op):
        """Turn ``name`` into slice(twin, dim, 0, V); twin must exist before."""
        tshape = values[twin]["shape"]
        strides = _contiguous(tshape) if not values[twin].get("alias_of") else values[twin]["strides"]
        values[name].update(alias_of=twin, strides=list(strides), producer=name, storage_offset=0)
        return _new_op(name, "aten.slice.Tensor", [{"ref": twin}, dim, 0, vocab, 1], view=True)

    # 1. Padded weight: Wp = vocab_pad(source) in the cast's dtype, padded along the weight's vocabulary dim.
    root, vdim, source = plan["root"], plan["vdim"], plan["source"]
    wshape = list(values[root]["shape"])
    pshape = list(wshape)
    pshape[vdim] = vp
    wp = fresh("weight")
    _new_value(data, wp, pshape, dtype)
    pad_op = _new_op(wp, TARGETS["pad"], [{"ref": source}, vdim, vp], {"dtype": dtype})
    weight_op = plan["weight_op"]
    if weight_op is not None:
        # The original cast becomes a view of the padded cast; its other readers keep their values.
        replace[weight_op["name"]] = [pad_op, slice_view(root, wp, vdim, weight_op)]
    else:
        add_before(plan["mm"]["name"], pad_op)
    twins = {root: (wp, vdim)}

    producer, consumers = _index(data)

    def twin_view(name):
        """Twin of a transposing view chain over a value with a twin (created on demand)."""
        if name in twins:
            return twins[name]
        op = producer.get(name)
        if op is None or op["target"] not in _TRANSPOSING_VIEWS:
            return None
        base = twin_view(_ref(op["args"][0]))
        if base is None:
            return None
        bname, bdim = base
        bshape = values[bname]["shape"]
        rank = len(bshape)
        target = op["target"]
        if target in ("aten.alias.default", "aten.detach.default"):
            perm = list(range(rank))
        elif target == "aten.t.default":
            perm = list(reversed(range(rank)))
        elif target == "aten.transpose.int":
            perm = list(range(rank))
            x, y = int(op["args"][1]) % rank, int(op["args"][2]) % rank
            perm[x], perm[y] = perm[y], perm[x]
        else:
            perm = [int(d) % rank for d in op["args"][1]]
        new = fresh("view_" + name)
        bstr = values[bname]["strides"]
        _new_value(data, new, [bshape[p] for p in perm], values[bname]["dtype"],
                   alias_of=bname, strides=[bstr[p] for p in perm])
        twins[name] = (new, perm.index(bdim))
        new_ops.append(_new_op(new, target, [{"ref": bname}, *op["args"][1:]], op.get("kwargs"), view=True))
        return twins[name]

    # 2. Logits product on padded storage; the original product becomes a view.
    new_ops = []
    mm = plan["mm"]
    a, b = plan["a"], plan["b"]
    bt = twin_view(b)
    if bt is None or bt[1] != 1:
        raise Refused("vocabulary operand twin does not expose the vocabulary as product columns")
    pp = fresh("logits")
    _new_value(data, pp, [rows, vp], dtype)
    new_ops.append(_new_op(pp, "aten.mm.default", [{"ref": a}, {"ref": bt[0]}]))
    replace[mm["name"]] = new_ops + [slice_view(plan["product"], pp, 1, mm)]
    twins[plan["product"]] = (pp, 1)

    # 3. Row statistics and the forward pick.
    stats = fresh("stats")
    _new_value(data, stats, [rows, 2], "float32")
    stats_op = _new_op(stats, TARGETS["stats"], [{"ref": pp}, vocab])
    gather = plan["gather"]
    pick = _new_op(gather["name"], TARGETS["pick"], [{"ref": pp}, {"ref": stats}, gather["args"][2], vocab])
    replace[gather["name"]] = [stats_op, pick]
    remove.update({plan["ls_op"]["name"]} | ({plan["cast_op"]["name"]} if plan["cast_op"] else set()))

    # 4. Fused gradient into padded storage, then padded products for every vocabulary-reading product.
    if plan["back"] is not None:
        leaf = plan["leaf"]
        dpp = fresh("dlogits")
        _new_value(data, dpp, [rows, vp], dtype)
        args = [{"ref": pp}, {"ref": stats}, {"ref": leaf["index"]},
                {"ref": leaf["source"]} if "source" in leaf else float(leaf["value"])]
        args += [{"ref": f} if f is not None else None for f in plan["factors"]]
        grad_op = _new_op(dpp, TARGETS["grad"], args, {"vocabulary": vocab, "chain": plan["chain"],
                                                         "scatter_add": bool(leaf.get("add"))})
        dlogits = plan["dlogits"]
        last = plan["dcast"] or plan["back"]
        replace[last["name"]] = [grad_op, slice_view(dlogits, dpp, 1, last)]
        remove.update({plan["back"]["name"], leaf["scatter"]["name"], leaf["zeros"]["name"]}
                      | {op["name"] for op in plan["chain_ops"]} | ({plan["back"]["name"]} if plan["dcast"] else set()))
        twins[dlogits] = (dpp, 1)
        # Products reading the vocabulary dimension through views of dlogits or the weight. Choose the
        # operand combination (twin or original slice view) whose shapes agree; a padded contraction
        # needs zero-padded twins on both sides, and at most one output dimension may be padded.
        for op in list(operations):
            if op["target"] != "aten.mm.default" or op is mm or op["name"] in replace:
                continue
            x, y = _ref(op["args"][0]), _ref(op["args"][1])
            new_ops = []
            tx, ty = twin_view(x), twin_view(y)
            if tx is None and ty is None:
                continue
            original = list(values[op["name"]]["shape"])
            chosen = None
            for use_x, use_y in ((True, True), (True, False), (False, True)):
                if (use_x and tx is None) or (use_y and ty is None):
                    continue
                xs, ys = (tx[0] if use_x else x), (ty[0] if use_y else y)
                if values[xs]["shape"][1] != values[ys]["shape"][0]:
                    continue
                shape = [values[xs]["shape"][0], values[ys]["shape"][1]]
                changed = [d for d in range(2) if shape[d] != original[d]]
                if len(changed) > 1 or any(original[d] != vocab or shape[d] != vp for d in changed):
                    continue
                chosen = xs, ys, shape, changed
                break
            if chosen is None or (chosen[0] == x and chosen[1] == y):
                continue
            xs, ys, shape, changed = chosen
            if not changed:
                new_ops.append(_new_op(op["name"], "aten.mm.default", [{"ref": xs}, {"ref": ys}]))
                replace[op["name"]] = new_ops
                continue
            twin = fresh("product_" + op["name"])
            _new_value(data, twin, shape, values[op["name"]]["dtype"])
            new_ops.append(_new_op(twin, "aten.mm.default", [{"ref": xs}, {"ref": ys}]))
            replace[op["name"]] = new_ops + [slice_view(op["name"], twin, changed[0], op)]
            twins[op["name"]] = (twin, changed[0])
    result = []
    for i, op in enumerate(operations):
        result.extend(inserts.get(i, []))
        if op["name"] in replace:
            result.extend(replace[op["name"]])
        elif op["name"] not in remove:
            result.append(op)
    for name in remove:
        if name not in replace and name in values and not any(name in (o.get("outputs") or [o["name"]]) for o in result):
            del values[name]
    data["operations"] = result
    _refresh_view_strides(data)


def _refresh_view_strides(data):
    """Recompute alias/stride metadata of every view from its (possibly re-rooted) source."""
    values = data["values"]
    for op in data["operations"]:
        target = op["target"]
        if op.get("synchronization") != "view" and not target.startswith(("aten.t.", "aten.transpose", "aten.permute",
                                                                           "aten.slice", "aten.alias", "aten.detach")):
            continue
        name, source = op["name"], _ref(op["args"][0]) if op.get("args") else None
        if source is None or source not in values or name not in values:
            continue
        src = values[source]
        strides, rank = list(src["strides"]), len(src["shape"])
        if target == "aten.t.default":
            strides = list(reversed(strides))
        elif target == "aten.transpose.int":
            x, y = int(op["args"][1]) % rank, int(op["args"][2]) % rank
            strides[x], strides[y] = strides[y], strides[x]
        elif target == "aten.permute.default":
            strides = [src["strides"][int(d) % rank] for d in op["args"][1]]
        elif target == "aten.slice.Tensor":
            dim = int(op["args"][1]) % rank
            step = int(op["args"][4]) if len(op["args"]) > 4 else 1
            strides[dim] *= step
        elif target in ("aten.alias.default", "aten.detach.default"):
            pass
        elif target == "aten.unsqueeze.default":
            dim = int(op["args"][1]) % (rank + 1)
            strides.insert(dim, src["strides"][dim] * src["shape"][dim] if dim < rank else 1)
        elif target in ("aten.view.default", "aten._unsafe_view.default", "aten.reshape.default"):
            if list(src["strides"]) != _contiguous(src["shape"]):
                continue
            strides = _contiguous(values[name]["shape"])
        else:
            continue
        if len(strides) == len(values[name]["shape"]):
            values[name]["strides"] = strides
            values[name]["alias_of"] = source


# ----------------------------------------------------------------------------------------- emitters
def emit(target, args, kwargs, name, shape, layouts, dtypes, names, read):
    """(kind, code, metadata, helpers) for a vocabulary operation, or None."""
    if target not in TARGETS.values():
        return None
    from .lower import LoweringError, _ident, _index, _ref as ref_of
    ref = lambda i: ref_of(args[i], names)
    out = _ident(name)
    if target == TARGETS["pad"]:
        return _emit_pad(ref(0), int(args[1]), int(args[2]), name, shape, layouts, dtypes, out, _ident, LoweringError)
    product = ref(0)
    lp = layouts[product]
    rows, vp = lp.shape
    if tuple(lp.strides) != (vp, 1) or lp.offset or vp % 8:
        raise LoweringError(f"{target}: padded contiguous logits required")
    ctype = "__nv_bfloat16" if dtypes[product] == "bfloat16" else "float"
    x = _ident(product)
    if target == TARGETS["stats"]:
        vocab = int(args[1])
        return _emit_stats(x, ctype, rows, vp, vocab, out)
    stats = _ident(ref(1))
    if layouts[ref(1)].shape != (rows, 2) or not layouts[ref(1)].contiguous or layouts[ref(1)].offset:
        raise LoweringError(f"{target}: contiguous row statistics required")
    index = ref(2)
    li = layouts[index]
    idx = lambda row: f"{_ident(index)}[{_index(li, (rows, 1), row)}]"
    if target == TARGETS["pick"]:
        vocab = int(args[3])
        code = f"""for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x; row < {rows}LL; row += (long long)gridDim.x * blockDim.x) {{
  const int64_t t = {idx("row")};
  assert(t >= 0 && t < {vocab}LL);
  {out}[row] = ({_load(x, ctype, f"row * {vp}LL + t")} - {stats}[2 * row]) - {stats}[2 * row + 1];
}}"""
        return "vocab_pick", code, {"vocab_rows": rows}, "#include <cassert>\n"
    vocab = int(kwargs["vocabulary"])
    chain = kwargs["chain"]
    leaf = args[3]
    factors = list(args[4:])
    if len(factors) != len(chain):
        raise LoweringError("vocab_grad chain/factor arity mismatch")
    if isinstance(leaf, (int, float)):
        if not math.isfinite(float(leaf)):
            raise LoweringError("non-finite scatter value")
        c = f"{float(leaf)!r}f"
    else:
        src = ref_of(leaf, names)
        c = f"{_ident(src)}[{_index(layouts[src], (rows, 1), 'row')}]"
        if kwargs.get("scatter_add"):
            c = f"(0.0f + {c})"
    steps = []
    for op, factor in zip(chain, factors):
        if op == "~":
            steps.append("c = -c;")
            continue
        f = ref_of(factor, names)
        if dtypes[f] != "float32":
            raise LoweringError("vocab_grad factors must be FP32")
        value = f"{_ident(f)}[{_index(layouts[f], (rows, vocab), f'(row * {vocab}LL)')}]"
        steps.append(f"c = c {'*' if op == '*' else '/'} {value};")
    return _emit_grad(x, ctype, rows, vp, vocab, stats, idx("row"), c, steps, out, dtypes[name])


def _load(pointer, ctype, position):
    if ctype == "float":
        return f"{pointer}[{position}]"
    return f"__bfloat162float({pointer}[{position}])"


def _tree(var, operation, identity):
    return f"""{var} = s101h::warp_{operation}({var});
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = {var};
    __syncthreads();
    if (threadIdx.x < 32) {{
      float partial = threadIdx.x < 8 ? smem[threadIdx.x] : {identity};
      partial = s101h::warp_{operation}(partial);
      if (threadIdx.x == 0) smem[0] = partial;
    }}
    __syncthreads();
    {var} = smem[0];
    __syncthreads();"""


_UNPACK = """__device__ __forceinline__ void latch_vocab_unpack(const uint4 raw, float* v) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int e = 0; e < 4; ++e) { const float2 f = __bfloat1622float2(h[e]); v[2 * e] = f.x; v[2 * e + 1] = f.y; }
}
__device__ __forceinline__ void latch_vocab_unpack(const float4 lo, const float4 hi, float* v) {
  v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w; v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
}
"""
_HELPERS = "#include <cassert>\n#ifndef LATCH_VOCAB_HELPERS\n#define LATCH_VOCAB_HELPERS\n" + _UNPACK + "#endif\n"


def _chunk_load(x, ctype, base):
    """Eight consecutive elements at ``base`` (a multiple of eight) into float v[8]."""
    if ctype == "float":
        return (f"latch_vocab_unpack(*reinterpret_cast<const float4*>({x} + {base}), "
                f"*reinterpret_cast<const float4*>({x} + {base} + 4), v);")
    return f"latch_vocab_unpack(*reinterpret_cast<const uint4*>({x} + {base}), v);"


def _batched(body, full, ilp=None):
    """Loop over row chunks [threadIdx.x, full) with ``ilp`` independent 16-byte loads issued together."""
    ilp = ROW_ILP if ilp is None else ilp
    loads = "\n".join(f"        float v{u}[8]; LOAD(v{u}, base + 8LL * (q + {u} * (int)blockDim.x))" for u in range(ilp))
    uses = "\n".join(body.replace("v[", f"v{u}[") for u in range(ilp))
    return f"""    {{ int q = threadIdx.x;
    for (; q + {ilp - 1} * (int)blockDim.x < {full}; q += {ilp} * (int)blockDim.x) {{
{loads}
{uses}
    }}
    for (; q < {full}; q += blockDim.x) {{
        float v[8]; LOAD(v, base + 8LL * q)
{body}
    }} }}"""


def _load_macro(x, ctype):
    if ctype == "float":
        return (f"#define LOAD(dst, at) latch_vocab_unpack(*reinterpret_cast<const float4*>({x} + (at)), "
                f"*reinterpret_cast<const float4*>({x} + (at) + 4), dst);")
    return f"#define LOAD(dst, at) latch_vocab_unpack(*reinterpret_cast<const uint4*>({x} + (at)), dst);"


def _emit_stats_online(x, ctype, rows, vp, vocab, out):
    """LATCH-MOESUP: one warp per row, one pass (online max/rescaled exp-sum per lane, then an xor-butterfly merge).

    The row maximum is the exact maximum over the first V columns (fmaxf is order-free); the exp-sum is accumulated
    as sum_k expf(x_k - m_running) rescaled by expf(m_old - m_new) whenever a lane's running maximum grows, then
    lanes merge with the same rescaling -- the online softmax normalizer (Milakov & Gimelshein; PyTorch Inductor's
    prepare_softmax_online). A declared FP32 reassociation of the CTA-tree exp-sum; one read of each row instead of
    two, no block barriers, eight rows in flight per CTA."""
    chunks = vp // 8
    full = vocab // 8
    ilp = ROW_ILP
    loads = "\n".join(f"      float v{u}[8]; const bool k{u} = q + {32 * u} < {full}; if (k{u}) LOAD(v{u}, base + 8LL * (q + {32 * u}))"
                      for u in range(ilp))
    uses = "\n".join(f"""      if (k{u}) {{ float mc = v{u}[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) mc = fmaxf(mc, v{u}[e]);
        if (mc > m) {{ s = s * expf(m - mc); m = mc; }}
#pragma unroll
        for (int e = 0; e < 8; ++e) s += expf(v{u}[e] - m); }}""" for u in range(ilp))
    tail = "" if full == chunks else f"""
    for (int q = {full} + lane; q < {chunks}; q += 32) {{
      float v[8]; LOAD(v, base + 8LL * q)
#pragma unroll
      for (int e = 0; e < 8; ++e) if (8 * q + e < {vocab}) {{
        if (v[e] > m) {{ s = s * expf(m - v[e]); m = v[e]; }}
        s += expf(v[e] - m);
      }}
    }}"""
    if VOCAB_PIPE and ctype != "float":
        def issue(dst, base, q):
            return "\n".join(f"      if ({q} + {32 * u} < {full}) {dst}[{u}] = "
                             f"*reinterpret_cast<const uint4*>({x} + {base} + 8LL * ({q} + {32 * u}));" for u in range(ilp))
        steps = "\n".join(f"""      {{ const bool k = q + {32 * u} < {full}; float v{u}[8]; if (k) latch_vocab_unpack(cur_[{u}], v{u});
      if (k) {{ float mc = v{u}[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) mc = fmaxf(mc, v{u}[e]);
        if (mc > m) {{ s = s * expf(m - mc); m = mc; }}
#pragma unroll
        for (int e = 0; e < 8; ++e) s += expf(v{u}[e] - m); }} }}""" for u in range(ilp))
        head = f"""{{
{_load_macro(x, ctype)}
  const int lane = threadIdx.x & 31;
  const long long stride_ = ((long long)gridDim.x * blockDim.x) >> 5;
  long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint4 cur_[{ilp}];
  if (row < {rows}LL) {{
{issue("cur_", f"row * {vp}LL", "lane")}
  }}
  for (; row < {rows}LL; row += stride_) {{
    const long long base = row * {vp}LL;
    float m = -INFINITY, s = 0.0f;
    for (int q = lane; q < {full}; q += {32 * ilp}) {{
      uint4 nxt_[{ilp}];
      if (q + {32 * ilp} < {full}) {{
{issue("nxt_", "base", f"(q + {32 * ilp})")}
      }} else if (row + stride_ < {rows}LL) {{
{issue("nxt_", f"(row + stride_) * {vp}LL", "lane")}
      }}
{steps}
#pragma unroll
      for (int u = 0; u < {ilp}; ++u) cur_[u] = nxt_[u];
    }}{tail}"""
    else:
        head = f"""{{
{_load_macro(x, ctype)}
  const int lane = threadIdx.x & 31;
  for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; row += ((long long)gridDim.x * blockDim.x) >> 5) {{
    const long long base = row * {vp}LL;
    float m = -INFINITY, s = 0.0f;
    for (int q = lane; q < {full}; q += {32 * ilp}) {{
{loads}
{uses}
    }}{tail}"""
    code = head + f"""
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {{
      const float m2 = __shfl_xor_sync(0xffffffffu, m, delta), s2 = __shfl_xor_sync(0xffffffffu, s, delta);
      const float mm = fmaxf(m, m2);
      s = (m == -INFINITY ? 0.0f : s * expf(m - mm)) + (m2 == -INFINITY ? 0.0f : s2 * expf(m2 - mm));
      m = mm;
    }}
    if (lane == 0) {{ {out}[2 * row] = m; {out}[2 * row + 1] = logf(s); }}
  }}
#undef LOAD
}}"""
    return "vocab_rows", code, {"vocab_rows": rows, "reduction_elements": vocab, "engine": "vocab_stats_online_warp",
                                "arithmetic": "FP32 exact row max; online rescaled exp-sum per lane then xor-butterfly "
                                              "merge over the first V columns (declared reassociation)"}, _HELPERS


# LATCH-MOESUP: the logits product's epilogue writes per-(row, N tile) softmax partials (pops4b rowpart) and
# vocab_stats only merges them (lower.py). Measured slower on the 25M head: the TMA body's epilogue does not overlap
# the next tile's main loop, so the partials' exp work lands on the product's critical path (plain update 68.41 ->
# 69.56 ms staged, 71.05 ms register-direct, vs the 4.5 ms statistics pass it removes). Off by default.
EPILOGUE_STATS = False


def emit_stats_partials(part, rows, tiles, out):
    """Merge per-tile (max, exp-sum) partials into (max, log exp-sum) rows: one warp per row, each lane merges its
    tiles in ascending order, then an xor butterfly (the online statistics merge). Declared FP32 reassociation."""
    code = f"""{{
  const int lane = threadIdx.x & 31;
  const float2* part_ = reinterpret_cast<const float2*>({part});
  for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; row += ((long long)gridDim.x * blockDim.x) >> 5) {{
    float m = -INFINITY, s = 0.0f;
    for (int t = lane; t < {tiles}; t += 32) {{
      const float2 p = part_[row * {tiles}LL + t];
      if (p.x != -INFINITY) {{
        if (m == -INFINITY) {{ m = p.x; s = p.y; }}
        else {{ const float mm = fmaxf(m, p.x); s = s * expf(m - mm) + p.y * expf(p.x - mm); m = mm; }}
      }}
    }}
#pragma unroll
    for (int delta = 16; delta > 0; delta /= 2) {{
      const float m2 = __shfl_xor_sync(0xffffffffu, m, delta), s2 = __shfl_xor_sync(0xffffffffu, s, delta);
      const float mm = fmaxf(m, m2);
      s = (m == -INFINITY ? 0.0f : s * expf(m - mm)) + (m2 == -INFINITY ? 0.0f : s2 * expf(m2 - mm));
      m = mm;
    }}
    if (lane == 0) {{ {out}[2 * row] = m; {out}[2 * row + 1] = logf(s); }}
  }}
}}"""
    return "vocab_rows", code, {"vocab_rows": rows, "reduction_elements": tiles, "engine": "vocab_stats_epilogue_partials",
                                "arithmetic": "per-tile row max / exp-sum of the rounded logits in the product's staged "
                                              "epilogue, merged online in tile order then xor-butterfly (declared "
                                              "reassociation)"}, ""


def _emit_stats(x, ctype, rows, vp, vocab, out):
    from .execution_plan import current_execution
    if ONLINE_STATS and current_execution() == "compose":
        probe = _STATS_PROBE.get()
        if probe is None:
            return _emit_stats_online(x, ctype, rows, vp, vocab, out)
        probe.append({"output": out, "rows": rows, "padded": vp, "vocabulary": vocab})
    chunks = vp // 8
    full = vocab // 8          # chunks entirely inside the vocabulary
    tail = "" if full == chunks else f"""
    for (int q = {full} + threadIdx.x; q < {chunks}; q += blockDim.x) {{
      float v[8]; LOAD(v, base + 8LL * q)
#pragma unroll
      for (int e = 0; e < 8; ++e) if (8 * q + e < {vocab}) ACC;
    }}"""
    max_body = "#pragma unroll\n        for (int e = 0; e < 8; ++e) maximum = fmaxf(maximum, v[e]);"
    sum_body = "#pragma unroll\n        for (int e = 0; e < 8; ++e) total += expf(v[e] - maximum);"
    code = f"""{{
{_load_macro(x, ctype)}
  for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
    const long long base = row * {vp}LL;
    float maximum = -INFINITY;
{_batched(max_body, full)}{tail.replace("ACC", "maximum = fmaxf(maximum, v[e])")}
    {_tree("maximum", "max", "-INFINITY")}
    float total = 0.0f;
{_batched(sum_body, full)}{tail.replace("ACC", "total += expf(v[e] - maximum)")}
    {_tree("total", "sum", "0.0f")}
    if (threadIdx.x == 0) {{ {out}[2 * row] = maximum; {out}[2 * row + 1] = logf(total); }}
    __syncthreads();
  }}
#undef LOAD
}}"""
    return "vocab_rows", code, {"vocab_rows": rows, "reduction_elements": vocab,
                                "arithmetic": "FP32 row max then shifted exp-sum over the first V columns; "
                                              "per-thread chunk order then fixed CTA tree"}, _HELPERS


def _emit_grad_flat(x, ctype, rows, vp, vocab, stats, index, c, steps, out, out_dtype):
    """LATCH-MOESUP: the same per-element gradient over a flat grid-stride stream of 16-byte chunks (all rows),
    ROW_ILP chunks in flight per thread; each chunk evaluates its row's target, chain factor and statistics (cached
    row reads). Identical element expressions and roundings to the CTA-per-row form; only the work mapping changes."""
    chunks = vp // 8
    total = rows * chunks
    if out_dtype == "bfloat16":
        store = ("{ uint4 packed; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&packed);\n"
                 "#pragma unroll\n        for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(g[2 * e], g[2 * e + 1]);\n"
                 f"        *reinterpret_cast<uint4*>({out} + AT) = packed; }}")
    else:
        store = (f"{{ *reinterpret_cast<float4*>({out} + AT) = make_float4(g[0], g[1], g[2], g[3]);\n"
                 f"        *reinterpret_cast<float4*>({out} + AT + 4) = make_float4(g[4], g[5], g[6], g[7]); }}")
    ilp = ROW_ILP
    step_code = "\n      ".join(steps)
    loads = "\n".join(f"    const unsigned q{u} = qb_ + {u}u * s_; const bool k{u} = q{u} < {total}u; float v{u}[8];\n"
                      f"    if (k{u}) LOAD(v{u}, 8LL * (long long)q{u})" for u in range(ilp))
    uses = "\n".join(f"""    if (k{u}) {{
      const long long row = (long long)(q{u} / {chunks}u), col = 8LL * (long long)(q{u} % {chunks}u);
      const int64_t t = {index};
      assert(t >= 0 && t < {vocab}LL);
      float c = {c};
      {step_code}
      const float maximum = {stats}[2 * row], logsum = {stats}[2 * row + 1];
      float g[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {{
        const long long j = col + e;
        g[e] = j < {vocab}LL ? ((j == t ? c : 0.0f) - expf((v{u}[e] - maximum) - logsum) * c) : 0.0f;
      }}
      {store.replace("AT", f"8LL * (long long)q{u}")}
    }}""" for u in range(ilp))
    code = f"""{{
{_load_macro(x, ctype)}
  const unsigned s_ = gridDim.x * blockDim.x;
  for (unsigned qb_ = blockIdx.x * blockDim.x + threadIdx.x; qb_ < {total}u; qb_ += {ilp}u * s_) {{
{loads}
{uses}
  }}
#undef LOAD
}}"""
    return "vocab_gradient", code, {"vocab_rows": rows, "reduction_elements": 1, "engine": "vocab_gradient_flat",
                                    "arithmetic": "log_softmax backward of a one-hot cotangent: "
                                                  "[j==t]c - exp((x-max)-logsum)*c, one rounding to the product dtype; "
                                                  "exact zero in padded columns"}, _HELPERS


def _emit_grad(x, ctype, rows, vp, vocab, stats, index, c, steps, out, out_dtype):
    if GRAD_FLAT and rows * (vp // 8) < 2 ** 31:
        return _emit_grad_flat(x, ctype, rows, vp, vocab, stats, index, c, steps, out, out_dtype)
    chunks = vp // 8
    if out_dtype == "bfloat16":
        store = ("{ uint4 packed; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&packed);\n"
                 "#pragma unroll\n          for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(g[2 * e], g[2 * e + 1]);\n"
                 f"          *reinterpret_cast<uint4*>({out} + AT) = packed; }}")
    else:
        store = (f"{{ *reinterpret_cast<float4*>({out} + AT) = make_float4(g[0], g[1], g[2], g[3]);\n"
                 f"          *reinterpret_cast<float4*>({out} + AT + 4) = make_float4(g[4], g[5], g[6], g[7]); }}")
    body = f"""        {{ const long long at = base + 8LL * QQ; float g[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) {{
            const long long j = 8LL * QQ + e;
            g[e] = j < {vocab}LL ? ((j == t ? c : 0.0f) - expf((v[e] - maximum) - logsum) * c) : 0.0f;
          }}
          {store.replace("AT", "at")} }}"""
    ilp = ROW_ILP
    step_code = "\n    ".join(steps)
    if VOCAB_GRAD_WARP and ctype != "float":
        # LATCH-MOESUP-ROW: one warp per row; lanes stride the row's 16-byte chunks with the next ROW_ILP-chunk group
        # (and, at a row's last group, the warp's next row's first group) in flight while the current group's
        # gradients are formed and stored. Identical per-element expression and rounding: bit-identical output.
        def issue(dst, base, q):
            return "\n".join(f"      if ({q} + {32 * u} < {chunks}) {dst}[{u}] = "
                             f"*reinterpret_cast<const uint4*>({x} + {base} + 8LL * ({q} + {32 * u}));" for u in range(ilp))
        # Fast path (same value): a chunk wholly inside the vocabulary that does not hold the target evaluates
        # 0.0f - expf((x - max) - logsum) * c, the general expression with j < V and j != t (standalone production
        # shape: 5.73 -> 4.81 ms, the 4.82 ms read+write roofline).
        fast_body = f"""        {{ const long long at = base + 8LL * QQ; float g[8];
#pragma unroll
          for (int e = 0; e < 8; ++e) g[e] = 0.0f - expf((v[e] - maximum) - logsum) * c;
          {store.replace("AT", "at")} }}"""
        full = vocab // 8
        uses = "\n".join(f"      if (q + {32 * u} < {chunks}) {{ float v[8]; latch_vocab_unpack(cur_[{u}], v);\n"
                         f"        if (q + {32 * u} < {full} && q + {32 * u} != tq_) {{\n"
                         + fast_body.replace("QQ", f"(q + {32 * u})") + "\n        } else {\n"
                         + body.replace("QQ", f"(q + {32 * u})") + " } }" for u in range(ilp))
        code = f"""{{
  const int lane = threadIdx.x & 31;
  const long long stride_ = ((long long)gridDim.x * blockDim.x) >> 5;
  long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint4 cur_[{ilp}];
  if (row < {rows}LL) {{
{issue("cur_", f"row * {vp}LL", "lane")}
  }}
  for (; row < {rows}LL; row += stride_) {{
    const long long base = row * {vp}LL;
    const int64_t t = {index};
    assert(t >= 0 && t < {vocab}LL);
    float c = {c};
    {step_code}
    const float maximum = {stats}[2 * row], logsum = {stats}[2 * row + 1];
    const int tq_ = (int)(t >> 3);
    for (int q = lane; q < {chunks}; q += {32 * ilp}) {{
      uint4 nxt_[{ilp}];
      if (q + {32 * ilp} < {chunks}) {{
{issue("nxt_", "base", f"(q + {32 * ilp})")}
      }} else if (row + stride_ < {rows}LL) {{
{issue("nxt_", f"(row + stride_) * {vp}LL", "lane")}
      }}
{uses}
#pragma unroll
      for (int u = 0; u < {ilp}; ++u) cur_[u] = nxt_[u];
    }}
  }}
}}"""
        return "vocab_gradient", code, {"vocab_rows": rows, "reduction_elements": 1, "engine": "vocab_gradient_warp_pipelined",
                                        "arithmetic": "log_softmax backward of a one-hot cotangent: "
                                                      "[j==t]c - exp((x-max)-logsum)*c, one rounding to the product dtype; "
                                                      "exact zero in padded columns"}, _HELPERS
    loads = "\n".join(f"        float v{u}[8]; LOAD(v{u}, base + 8LL * (q + {u} * (int)blockDim.x))" for u in range(ilp))
    uses = "\n".join(body.replace("v[", f"v{u}[").replace("QQ", f"(q + {u} * (int)blockDim.x)") for u in range(ilp))
    code = f"""{{
{_load_macro(x, ctype)}
  for (long long row = blockIdx.x; row < {rows}LL; row += gridDim.x) {{
    const long long base = row * {vp}LL;
    const int64_t t = {index};
    assert(t >= 0 && t < {vocab}LL);
    float c = {c};
    {step_code}
    const float maximum = {stats}[2 * row], logsum = {stats}[2 * row + 1];
    int q = threadIdx.x;
    for (; q + {ilp - 1} * (int)blockDim.x < {chunks}; q += {ilp} * (int)blockDim.x) {{
{loads}
{uses}
    }}
    for (; q < {chunks}; q += blockDim.x) {{
        float v[8]; LOAD(v, base + 8LL * q)
{body.replace("QQ", "q")}
    }}
  }}
#undef LOAD
}}"""
    return "vocab_gradient", code, {"vocab_rows": rows, "reduction_elements": 1,
                                    "arithmetic": "log_softmax backward of a one-hot cotangent: "
                                                  "[j==t]c - exp((x-max)-logsum)*c, one rounding to the product dtype; "
                                                  "exact zero in padded columns"}, _HELPERS


def _emit_pad(source, vdim, vp, name, shape, layouts, dtypes, out, _ident, LoweringError):
    ls = layouts[source]
    if len(ls.shape) != 2 or len(shape) != 2 or ls.offset:
        raise LoweringError("vocab_pad requires a rank-2 source")
    vocab = ls.shape[vdim]
    if shape[vdim] != vp or shape[1 - vdim] != ls.shape[1 - vdim] or vp < vocab:
        raise LoweringError("vocab_pad shape mismatch")
    narrow = dtypes[name] == "bfloat16"
    if dtypes[source] != "float32" or dtypes[name] not in ("bfloat16", "float32"):
        raise LoweringError("vocab_pad reads FP32 weights")
    rows, cols = shape
    s0, s1 = ls.strides
    numel = rows * cols
    meta = {"vocab_pad": {"vocabulary": vocab, "padded": vp, "dim": vdim}}
    src = _ident(source)
    if cols % 8 == 0 and numel < 2 ** 31:
        # Eight consecutive outputs of one row per chunk: one 16-byte (BF16) or two 16-byte (FP32) stores,
        # 32-bit index arithmetic, four independent chunks in flight per thread.
        per_row = cols // 8
        valid = (lambda: f"(k + e) < {vocab}") if vdim == 1 else (lambda: f"r < {vocab}")
        if narrow:
            store = ("uint4 packed; __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&packed);\n"
                     "#pragma unroll\n        for (int e = 0; e < 4; ++e) h[e] = __floats2bfloat162_rn(w[2 * e], w[2 * e + 1]);\n"
                     f"        *reinterpret_cast<uint4*>({out} + 8u * c) = packed;")
        else:
            store = (f"*reinterpret_cast<float4*>({out} + 8u * c) = make_float4(w[0], w[1], w[2], w[3]);\n"
                     f"        *reinterpret_cast<float4*>({out} + 8u * c + 4) = make_float4(w[4], w[5], w[6], w[7]);")
        code = f"""for (unsigned c0 = blockIdx.x * blockDim.x + threadIdx.x; c0 < {numel // 8}u; c0 += 4u * gridDim.x * blockDim.x) {{
#pragma unroll
  for (int u = 0; u < 4; ++u) {{
    const unsigned c = c0 + (unsigned)u * gridDim.x * blockDim.x;
    if (c < {numel // 8}u) {{
      const unsigned r = c / {per_row}u, k = (c % {per_row}u) * 8u;
      float w[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) w[e] = ({valid()}) ? {src}[(long long)r * {s0}LL + (long long)(k + e) * {s1}LL] : 0.0f;
        {store}
    }}
  }}
}}"""
        return "elementwise_copy", code, meta, ""
    value = f"{src}[r * {s0}LL + k * {s1}LL]"
    convert = (lambda v: f"__float2bfloat16_rn({v})") if narrow else (lambda v: v)
    zero = "__float2bfloat16_rn(0.0f)" if narrow else "0.0f"
    valid = "k < " + str(vocab) if vdim == 1 else "r < " + str(vocab)
    code = f"""for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; i += (long long)gridDim.x * blockDim.x) {{
  const long long r = i / {cols}LL, k = i % {cols}LL;
  {out}[i] = ({valid}) ? {convert(value)} : {zero};
}}"""
    return "elementwise_copy", code, meta, ""


def prefer_wide_tile(spec, ws, grid):
    """Existing P5C wide-tile plan for a vocabulary-scale product whose warp-specialized 128 x 128 plan is
    L2-stream bound, or None (keep ``ws``). Called from p5c_matrix.plan; shapes and the cost model only."""
    if not WIDE_TILES or spec.get('epi') or spec.get('splitk') or spec.get('out_f32') or spec.get('pointers'):
        return None
    m, n, k = spec['m'], spec['n'], spec['k']
    if max(m, n) < WIDE_EXTENT or ws.get('traffic_us', 0) <= ws.get('local_us', 0):
        return None
    from . import p5c_matrix
    try:
        _, candidates = p5c_matrix.p5c_plan(m, n, k, spec['ak'], spec['bkc'], grid=grid)
    except p5c_matrix.EngineRefusal:
        return None
    wide = [c for c in candidates if c['bm'] * c['bn'] >= 2 * 128 * 128 and c['traffic_us'] < ws['traffic_us']]
    if not wide:
        return None
    chosen = min(wide, key=lambda c: (c['traffic_us'], c['estimate_us'], -c['bm']))
    return dict(chosen, release=p5c_matrix.P5C_RELEASE, out_f32=False, mc=1, vocabulary_wide_tile=True)
