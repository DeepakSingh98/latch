"""P-OPS3-B short-context training attention lowering.

Extends an attention group matched by attention_lower.plan (scores, scale, additive mask, softmax, value product
and the PyTorch-derived backward) with the dtype conversions and head-layout copies around it:

    Q, K, V, dO : FP32 values reached from BF16 sources only through views, clones and one BF16->FP32 _to_copy
    O, dQ, dK^T, dV : consumed only through views, one clone and one FP32->BF16 _to_copy (the projection operands)

When every boundary absorbs, the group becomes two atomic phases per attention: a forward phase writing the BF16
output directly in its consumer's layout and a backward phase writing the three BF16 gradients, reading the BF16
sources through 4-D strides. No probabilities, log-sum-exp or FP32 head copies are stored. Admission depends only
on graph structure, dtypes, shapes and strides (attention_short.cuh documents the arithmetic). Anything else is a
recorded refusal and the group keeps the existing lowering.
"""
from __future__ import annotations

import math

HEADER = "attention_short.cuh"
HEAD_WIDTHS = (32, 64)
MAX_SEQUENCE = 128
DEFAULT_SPLITS = 3
SHARED_LIMIT = 133120  # P-OPS-D: crossing the H100 L1/shared carve-out at 132 KB slows every product phase
_VIEWS = {"view", "_unsafe_view", "reshape", "permute", "transpose", "t", "unsqueeze", "squeeze", "alias", "detach"}
_IDENTITY = {"clone", "contiguous", "alias", "detach"}


class Refused(Exception):
    pass


def _base(target):
    return str(target).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]


# ------------------------------------------------------------------------------------------ logical layouts
# A logical tensor is a list of axes; each axis is a list of factors (outer -> inner). A factor is
# [extent, stride, label]: stride is an element stride in a physical buffer (or None before materialization),
# label is a tuple path naming the factor's origin (output direction only).
def _flatten(axes):
    return [f for axis in axes for f in axis if f[0] != 1]


def _regroup(factors, shape):
    factors = [list(f) for f in factors if f[0] != 1]
    axes = []
    for n in shape:
        n = int(n)
        axis, have = [], 1
        while have < n:
            if not factors:
                raise Refused("reshape regroup underflow")
            extent, stride, label = factors[0]
            if have * extent <= n:
                if n % (have * extent):
                    raise Refused("reshape does not regroup whole factors")
                axis.append(factors.pop(0))
                have *= extent
            else:
                r = n // have
                if n % have or extent % r:
                    raise Refused("reshape splits a factor unevenly")
                inner = extent // r
                outer = [r, None if stride is None else stride * inner, None if label is None else label + (0,)]
                factors[0] = [inner, stride, None if label is None else label + (1,)]
                axis.append(outer)
                have *= r
        axes.append(axis)
    if factors:
        raise Refused("reshape regroup overflow")
    return axes


def _apply(axes, op, *, shape):
    base = _base(op["target"])
    args = list(op.get("args", []))
    if base in {"view", "_unsafe_view", "reshape"}:
        return _regroup(_flatten(axes), shape)
    if base == "permute":
        return [axes[int(d) % len(axes)] for d in args[1]]
    if base == "transpose":
        a, b = int(args[1]) % len(axes), int(args[2]) % len(axes)
        axes = list(axes)
        axes[a], axes[b] = axes[b], axes[a]
        return axes
    if base == "t":
        return list(reversed(axes)) if len(axes) == 2 else axes
    if base in {"unsqueeze", "squeeze"}:
        return _regroup(_flatten(axes), shape)
    if base in _IDENTITY or base == "_to_copy":
        return axes
    if base == "expand":
        extents = tuple(math.prod(f[0] for f in a) for a in axes)
        sizes = tuple(int(x) for x in args[1])
        if len(sizes) == len(extents) and all(s in (-1, e) for s, e in zip(sizes, extents)):
            return axes  # no-op expand (same shape): identity view
    raise Refused(f"{base} is not a layout or conversion operation")


def _source_axes(layout_shape, layout_strides):
    return [[[int(e), int(s), None]] for e, s in zip(layout_shape, layout_strides)]


# LATCH-ATTN-RECOG: operand views that select part of their source (packed QKV projections). They are exact reads:
# the body reads the same elements through a pointer offset and strides. Input direction only (physical strides).
_OFFSET_VIEWS = {"slice", "select", "as_strided"}


def _offset_view(axes, op, *, root_identity):
    """(axes, element offset) of a slice/select/as_strided read of ``axes`` (physical strides). Raises Refused."""
    base = _base(op["target"])
    args = list(op.get("args", []))
    if base == "as_strided":
        # Functionalized split/chunk: absolute (size, stride, offset) of the source storage. Admitted only directly
        # on the dense, offset-0 root (as lower.py's view rule), where the source is the root itself.
        if not root_identity:
            raise Refused("as_strided operand view of a transformed source")
        size, stride = [int(x) for x in args[1]], [int(x) for x in args[2]]
        offset = int(args[3]) if len(args) > 3 and args[3] is not None else 0
        if len(size) != len(stride) or offset < 0 or any(s < 0 for s in stride):
            raise Refused("as_strided operand view is not a nonnegative strided view")
        return [[[e, s, None]] for e, s in zip(size, stride)], offset
    dim = int(args[1]) % len(axes) if len(args) > 1 else 0
    factors = [f for f in axes[dim] if f[0] != 1]
    extent = math.prod(f[0] for f in factors)
    if base == "select":
        index = int(args[2])
        index = index + extent if index < 0 else index
        if not 0 <= index < extent:
            raise Refused("select index out of range")
        offset, rest = 0, index
        for f in reversed(factors):
            offset += (rest % f[0]) * f[1]
            rest //= f[0]
        return axes[:dim] + axes[dim + 1:], offset
    start = args[2] if len(args) > 2 and args[2] is not None else 0
    end = args[3] if len(args) > 3 and args[3] is not None else extent
    step = int(args[4]) if len(args) > 4 else 1
    start, end = int(start), int(end)
    start = max(0, min(extent, start + extent if start < 0 else start))
    end = max(start, min(extent, end + extent if end < 0 else end))
    if step <= 0:
        raise Refused("slice requires a positive step")
    if (start, end, step) == (0, extent, 1):
        return axes, 0
    length = -(-(end - start) // step)
    if len(factors) == 1:
        e, s = factors[0][0], factors[0][1]
        axis = [[length, s * step, None]] if length != 1 else []
        return axes[:dim] + [axis] + axes[dim + 1:], start * s
    inner = math.prod(f[0] for f in factors[1:])
    if not factors or step != 1 or start % inner or end % inner:
        raise Refused("slice does not select whole outer factors of a merged axis")
    outer = factors[0]
    axis = [[length // inner, outer[1], None]] + [list(f) for f in factors[1:]]
    return axes[:dim] + [axis] + axes[dim + 1:], (start // inner) * outer[1]


def _split_heads(axis, bh, heads):
    """(outer stride, inner stride) of a batch*heads axis decomposed as (bh // heads, bh % heads)."""
    parts = _regroup([list(f) for f in axis], (bh // heads, heads))
    strides = []
    for part in parts:
        if len(part) > 1:
            raise Refused("batch-head axis is not two strided factors")
        strides.append(part[0][1] if part else 0)
    return strides


class _Graph:
    def __init__(self, operations, values, names, ref, field, output_name):
        self.ops, self.values, self.names, self.ref, self.field = {}, values, names, ref, field
        self.consumers = {}
        self.order = {}
        for index, op in enumerate(operations):
            name = output_name(op)
            self.ops[name] = op
            self.order[name] = index
            seen = set()

            def walk(item):
                r = ref(item, names)
                if r is not None:
                    if r not in seen:
                        seen.add(r)
                        self.consumers.setdefault(r, []).append(name)
                    return
                if isinstance(item, (list, tuple)):
                    for x in item:
                        walk(x)
                elif isinstance(item, dict):
                    for x in item.values():
                        walk(x)
            walk(list(field(op, "args", [])))
            walk(dict(field(op, "kwargs", {})))

    def op(self, name):
        op = self.ops[name]
        return {"target": str(self.field(op, "target", self.field(op, "op"))),
                "args": list(self.field(op, "args", [])), "kwargs": dict(self.field(op, "kwargs", {}))}

    def shape(self, name):
        return tuple(int(x) for x in self.field(self.values[name], "shape"))

    def strides(self, name):
        return tuple(int(x) for x in self.field(self.values[name], "strides", ()))

    def dtype(self, name):
        return str(self.field(self.values[name], "dtype", "float32")).removeprefix("torch.")

    def source_of(self, name):
        args = self.op(name)["args"]
        return self.ref(args[0], self.names) if args else None


def _noop_expand(graph, name):
    """An ``expand`` to its source's own shape (rank-4 ``matmul`` broadcasting preparation): an identity view."""
    op = graph.op(name)
    return _base(op["target"]) == "expand" and graph.shape(name) == graph.shape(graph.source_of(name))


def _input_chain(graph, value, offsets=False):
    """Walk from an FP32 operand back to its BF16 source. Returns (source, chain ops in forward order).

    ``offsets`` (LATCH-ATTN-RECOG, flat bodies): the chain may also select part of its source (slice/select/
    as_strided of a packed projection); the source is then the packed BF16 value, read at an element offset.
    """
    chain, current, converts = [], value, 0
    while current in graph.ops:
        op = graph.op(current)
        base = _base(op["target"])
        if base not in _VIEWS and base not in _IDENTITY and base != "_to_copy" and not (
                offsets and (base in _OFFSET_VIEWS or _noop_expand(graph, current))):
            break
        if base == "_to_copy":
            dtype = str(op["kwargs"].get("dtype", "")).removeprefix("torch.")
            if dtype != "float32" or graph.dtype(graph.source_of(current)) != "bfloat16":
                raise Refused(f"{current}: conversion is not BF16 -> FP32")
            if set(op["kwargs"]) - {"dtype", "layout", "device", "pin_memory", "memory_format"}:
                raise Refused(f"{current}: conversion has unsupported keywords")
            converts += 1
        chain.append(current)
        current = graph.source_of(current)
    if converts != 1:
        raise Refused(f"{value}: operand is not a single BF16 -> FP32 widening (FP32 operands keep the FP32 path)")
    if graph.dtype(current) != "bfloat16":
        raise Refused(f"{value}: source {current} is not BF16")
    return current, list(reversed(chain))


def _input_map(graph, layouts, value, offsets=False):
    """(source, chain, axes) or, with ``offsets``, (source, chain, axes, element offset into the source)."""
    source, chain = _input_chain(graph, value, offsets)
    layout = layouts[source]
    axes = _source_axes(layout.shape, layout.strides)
    offset, identity = 0, True
    dense = tuple(layout.strides) == tuple(math.prod(layout.shape[d + 1:]) for d in range(len(layout.shape)))
    for name in chain:
        op = graph.op(name)
        if _base(op["target"]) in _OFFSET_VIEWS:
            axes, delta = _offset_view(axes, op, root_identity=identity and dense and offset == 0)
            offset += delta
            identity = False
            continue
        axes = _apply(axes, op, shape=graph.shape(name))
        identity = identity and _base(op["target"]) in _IDENTITY | {"_to_copy"}
    if offsets:
        return source, chain, axes, offset
    return source, chain, axes


def _output_chain(graph, value, group_names, cast_first=False):
    """Layout ops, at most one materializing copy and exactly one FP32 -> BF16 cast after an attention result.

    The cast and the copy are both exact and elementwise, so their order does not change any value: the BF16 result
    is materialized at the copy when there is one (before or after the cast; LATCH-ATTN-RECOG: BF16 SDPA and the
    transposed reads of BF16 q/k/v put the cast first), otherwise at the cast. The chain ends at the materialized
    BF16 buffer; later views are ordinary reads of it. ``cast_first=False`` (loop tape callers) keeps the original
    rule: exactly one copy, before the cast.
    """
    chain, current, clones, cast_at = [], value, 0, None
    while True:
        users = [u for u in graph.consumers.get(current, []) if u not in group_names]
        user = users[0] if len(users) == 1 else None
        base = _base(graph.op(user)["target"]) if user is not None else None
        if cast_at is not None and (user is None or base not in _VIEWS | {"clone", "contiguous"}):
            # Cast first and no later copy: the cast result is the materialized buffer.
            del chain[cast_at + 1:]
            break
        if user is None:
            raise Refused(f"{current}: attention result must have exactly one layout/cast consumer")
        if base in {"clone", "contiguous"}:
            clones += 1
            if cast_at is not None:
                chain.append(user)
                break
        elif base == "_to_copy":
            op = graph.op(user)
            if str(op["kwargs"].get("dtype", "")).removeprefix("torch.") != "bfloat16":
                raise Refused(f"{user}: result is not cast to BF16")
            chain.append(user)
            if clones:
                break
            if not cast_first:
                raise Refused(f"{value}: output chain needs exactly one materializing copy")
            cast_at = len(chain) - 1
            current = user
            continue
        elif base not in _VIEWS:
            raise Refused(f"{user}: {base} is not a layout or BF16 cast consumer")
        chain.append(user)
        current = user
    if clones > 1 or (not cast_first and clones != 1):
        raise Refused(f"{value}: output chain needs {'at most' if cast_first else 'exactly'} one materializing copy")
    return chain


def _output_map(graph, value, axes_labels, chain):
    """Physical strides (in the final BF16 buffer) of the attention result's labeled factors."""
    shape = graph.shape(value)
    axes = [[[int(e), None, (label,)]] for e, label in zip(shape, axes_labels)]
    materialized = False
    copies = any(_base(graph.op(name)["target"]) in {"clone", "contiguous"} for name in chain)
    for name in chain:
        op = graph.op(name)
        base = _base(op["target"])
        if base in {"clone", "contiguous"} or (base == "_to_copy" and not copies):
            factors = _flatten(axes)
            stride = 1
            for factor in reversed(factors):
                factor[1] = stride
                stride *= factor[0]
            materialized = True
            continue
        if materialized and base in {"permute", "transpose", "t"}:
            raise Refused(f"{name}: permutation after materialization")
        axes = _apply(axes, op, shape=graph.shape(name))
    final = chain[-1]
    expected, stride = [], 1
    for extent in reversed(graph.shape(final)):
        expected.append(stride)
        stride *= extent
    if tuple(reversed(expected)) != graph.strides(final) and math.prod(graph.shape(final)) > 1:
        raise Refused(f"{final}: BF16 result is not contiguous")
    by_label = {}
    for factor in _flatten(axes):
        by_label.setdefault(factor[2][0], []).append(factor)
    result = []
    for label in axes_labels:
        factors = sorted(by_label.get(label, []), key=lambda f: f[2])
        result.append([[f[0], f[1], None] for f in factors])
    return result


def _strides4(axes, bh, heads, label):
    if len(axes) != 3:
        raise Refused(f"{label}: rank-3 head operand required")
    sb, sh = _split_heads(axes[0], bh, heads)
    rows = [f for f in axes[1] if f[0] != 1]
    units = [f for f in axes[2] if f[0] != 1]
    if len(rows) > 1 or len(units) > 1 or (units and units[0][1] != 1):
        raise Refused(f"{label}: rows need one stride and the head width unit stride")
    return (sb, sh, rows[0][1] if rows else 0)


def _heads_candidates(axes_list):
    found = []
    for axes in axes_list:
        factors = [f for f in axes[0] if f[0] != 1]
        if len(factors) == 2 and factors[1][0] not in found:
            found.append(factors[1][0])
    return found or [1]


def admit(groups, graph_args, static_layouts, protected, *, splits=DEFAULT_SPLITS, max_sequence=MAX_SEQUENCE,
          cast_first=False):
    """Return ({group label: resolved plan}, refusals) for matched attention groups.

    graph_args = (operations, values, names, ref, field, output_name). static_layouts gives every graph value's
    captured layout; the caller re-checks the BF16 sources' lowered layouts when installing the phases.
    """
    graph = _Graph(*graph_args)
    plans, refusals = {}, []
    for group in groups:
        label = group["label"]
        try:
            plan = _plan_group(graph, group, protected, splits=splits, max_sequence=max_sequence)
            plans[label] = finalize(plan, graph_args, static_layouts, protected, cast_first=cast_first)
        except Refused as exc:
            refusals.append({"group": label, "reason": "short attention: " + str(exc)})
    return plans, refusals


def _plan_group(graph, group, protected, *, splits, max_sequence):
    if not group.get("backward"):
        raise Refused("forward-only attention keeps the existing lowering")
    if splits not in (1, 2, 3):
        raise Refused("split count must be 1, 2 or 3")
    if group.get("p_round") or group.get("ds_round"):
        # P8-B: declared 'bf16_operands' roundings are exactly one BF16 piece per P / dS operand.
        if not (group.get("p_round") and group.get("ds_round")):
            raise Refused("BF16 operand roundings must cover both P and dS")
        splits = 1
    bh, t, hd = graph.shape(group["q"])
    if graph.shape(group["kt"]) != (bh, hd, t) or graph.shape(group["v"]) != (bh, t, hd):
        raise Refused("self-attention with equal key/value head width required")
    if hd not in HEAD_WIDTHS or not 1 <= t <= max_sequence:
        raise Refused(f"head width {hd} / context {t} outside {HEAD_WIDTHS} / [1, {max_sequence}]")
    tp = (t + 15) // 16 * 16
    pieces = 2 * (2 * splits * tp * 24 + splits * 16 * (tp + 8))
    backward = (2 * 4 * tp * (hd + 8) + max(pieces, 4 * tp * (hd + 4))
                + 4 * ((tp * 20 + 16 * (tp + 4) if tp <= 96 else 0) + 2 * tp + (tp // 8) * 16))
    if backward > SHARED_LIMIT:
        raise Refused(f"backward needs {backward} shared bytes > {SHARED_LIMIT} (L1 carve-out limit)")
    members = set(group["skip"])
    views = {group[k] for k in ("kt", "k_view", "qt_view", "vt_view") if group.get(k)}
    group_names = members | views
    return dict(group=group, bh=bh, t=t, hd=hd, splits=splits, group_names=group_names)


def finalize(plan, graph_args, layouts, protected, *, cast_first=False):
    """Resolve layouts (at the anchor, when source layouts exist) and the full skip set. Raises Refused."""
    graph = _Graph(*graph_args)
    group, bh, t, hd = plan["group"], plan["bh"], plan["t"], plan["hd"]
    group_names = set(plan["group_names"])
    sources, chains, maps, offsets = {}, {}, {}, {}
    for key, operand in (("q", group["q"]), ("kt", group["kt"]), ("v", group["v"]), ("do", group["do"]),
                         ("k_view", group["k_view"]), ("qt_view", group["qt_view"]), ("vt_view", group["vt_view"])):
        if operand in group["skip"]:
            raise Refused(f"{key} operand is produced inside the group")
        # Flat callers (cast_first) also read operands at an offset into a packed BF16 source (LATCH-ATTN-RECOG).
        source, chain, axes, offset = _input_map(graph, layouts, operand, offsets=True) if cast_first else (
            *_input_map(graph, layouts, operand), 0)
        sources[key], chains[key], maps[key], offsets[key] = source, chain, axes, offset

    def transposed(axes):
        return [axes[0], axes[2], axes[1]]
    pairs = {"q": [("q", maps["q"]), ("qt_view", transposed(maps["qt_view"]))],
             "k": [("kt", transposed(maps["kt"])), ("k_view", maps["k_view"])],
             "v": [("v", maps["v"]), ("vt_view", transposed(maps["vt_view"]))],
             "do": [("do", maps["do"])]}
    outputs = {"o": (group["o"], ("bh", "i", "d")), "dq": (group["dq"], ("bh", "i", "d")),
               "dkt": (group["dkt"], ("bh", "d", "i")), "dv": (group["dv"], ("bh", "i", "d"))}
    out_chains, out_maps = {}, {}
    for key, (value, labels) in outputs.items():
        chain = _output_chain(graph, value, group_names, cast_first)
        axes = _output_map(graph, value, labels, chain)
        if key == "dkt":
            axes = [axes[0], axes[2], axes[1]]
        out_chains[key], out_maps[key] = chain, axes
    candidates = _heads_candidates([axes for rows in pairs.values() for _, axes in rows] + list(out_maps.values()))
    error = None
    for heads in candidates:
        if bh % heads:
            continue
        try:
            strides = {}
            for key, rows in pairs.items():
                found = {tuple(_strides4(axes, bh, heads, name)) for name, axes in rows}
                owners = {(sources[name], offsets[name]) for name, _ in rows}
                if len(found) != 1 or len(owners) != 1:
                    raise Refused(f"{key}: forward and backward operands read different views")
                strides[key] = (*owners.pop(), found.pop())
            out_strides = {key: (out_chains[key][-1], _strides4(axes, bh, heads, key)) for key, axes in out_maps.items()}
            break
        except Refused as exc:
            error = exc
    else:
        raise error or Refused("no consistent batch/head decomposition")
    for key, (source, offset, (sb, sh, sr)) in strides.items():
        layout = layouts[source]
        if any(x % 8 for x in (sb, sh, sr, layout.offset, offset)):
            raise Refused(f"{key}: BF16 rows must be 16-byte aligned")
    for key, (final, (sb, sh, sr)) in out_strides.items():
        if any(x % 2 for x in (sb, sh, sr)):
            raise Refused(f"{key}: BF16 output pairs must be 4-byte aligned")
    removed = set(group["skip"]) | {name for chain in chains.values() for name in chain}
    removed |= {name for key, chain in out_chains.items() for name in chain[:-1]} | {group["o"], group["dq"],
                                                                                      group["dkt"], group["dv"]}
    removed |= group_names
    skip = removed | {chain[-1] for chain in out_chains.values()}
    for name in removed:
        if name in protected:
            raise Refused(f"{name} is returned or committed")
        outside = [u for u in graph.consumers.get(name, []) if u not in skip]
        if outside:
            raise Refused(f"{name} is used outside the attention region by {outside}")
    finals = {key: chain[-1] for key, chain in out_chains.items()}
    first_backward = min((group["dv"], group["dq"], group["dkt"]), key=graph.order.__getitem__)
    return dict(plan, sources={k: v[0] for k, v in strides.items()}, in_strides={k: v[2] for k, v in strides.items()},
                source_offsets={k: v[1] for k, v in strides.items() if v[1]}, finals=finals, out_strides={k: v[1] for k, v in out_strides.items()}, heads=heads,
                skip=skip, removed=removed, anchor_forward=group["o"], anchor_backward=first_backward,
                chains={k: list(v) for k, v in chains.items()}, out_chains=out_chains)


def source_pointer(plan, key, ident):
    """C++ operand pointer of a resolved source: the value itself, or an element offset into a packed source."""
    offset = plan.get("source_offsets", {}).get(key, 0)
    return ident(plan["sources"][key]) + (f" + {offset}LL" if offset else "")


def aux_names(group):
    """Forward-to-backward FP32 values: the unrounded attention output and the row log-sum-exp."""
    return {"o32": group["o"] + "__short_o32", "lse": group["o"] + "__short_lse",
            "trace": group["o"] + "__short_trace"}


def emit(plan, layouts, ident, mask_strides, *, which, trace=False, aux=None):
    # aux (P-OPS4-E): explicit O32/lse value names, e.g. loop tape slots; default internal names.
    group, bh, t, hd, ns = plan["group"], plan["bh"], plan["t"], plan["hd"], plan["splits"]
    tp = (t + 15) // 16 * 16
    s = plan["in_strides"]
    o = plan["out_strides"]
    mode = group["scale_mode"]
    scale = float(group["scale"]) if mode else 1.0

    def l4(x):
        return "{" + f"{x[0]}LL, {x[1]}LL, {x[2]}LL" + "}"
    ptr = lambda key: source_pointer(plan, key, ident)
    fields = [
        f"A.q = {ptr('q')}; A.lq = pshort::Layout4{l4(s['q'])};",
        f"A.k = {ptr('k')}; A.lk = pshort::Layout4{l4(s['k'])};",
        f"A.v = {ptr('v')}; A.lv = pshort::Layout4{l4(s['v'])};",
        f"A.heads = {plan['heads']}; A.t = {t}; A.scale = {repr(scale)}f; A.scale_mode = {mode};",
    ]
    if mask_strides is not None:
        fields.append(f"A.mask = {ident(group['mask'])}; A.m_b = {mask_strides[0]}LL; A.m_i = {mask_strides[1]}LL; "
                      f"A.m_j = {mask_strides[2]}LL; A.has_mask = 1;")
    else:
        fields.append("A.mask = nullptr; A.m_b = A.m_i = A.m_j = 0; A.has_mask = 0;")
    from . import padding_masks
    extended = padding_masks.fields(group, mask_strides)
    if extended:
        if aux is not None:
            raise Refused("padding-mask groups (release step 09) are not admitted by the loop tape rules")
        fields.append(extended)  # release step 09: per-example mask batches / boolean-mask empty rows
    nb = tp // 16
    items = bh * nb
    aux = {**aux_names(group), **(aux or {})}
    fields.append(f"A.o32 = {ident(aux['o32'])}; A.lse = {ident(aux['lse'])};")
    if trace:
        fields.append(f"A.trace = {ident(aux['trace'])}; A.trace_offset = {0 if which == 'forward' else items};")
    if which == "forward":
        fields.append(f"A.o = {ident(plan['finals']['o'])}; A.lo = pshort::Layout4{l4(o['o'])};")
        # Split-K forward where measured faster on H100 (T >= 128 or head width 32; neutral at T=64/HD=64).
        groups = tp // 16
        splitk = (8 % groups == 0 and (hd // 8) % (8 // groups) == 0) and (tp >= 128 or hd == 32)
        body = (f"pshort::forward_item<{hd}, {tp}, {ns}, {'true' if splitk else 'false'}>(A, (int)(item % {bh}LL), "
                f"(int)(item / {bh}LL), smem);")
        shared = 2 * (2 * tp * (hd + 8) + 16 * (hd + 8) + ns * 16 * (tp + 8)) + 4 * (16 * (tp + 4) + 2 * 8 * 16)
        # split-K forward variant (PSHORT_FWD_SPLITK) footprint; the program reserves the larger of the two
        splitk = (2 * (2 * tp * (hd + 8) + 16 * (hd + 8))
                  + 4 * (16 * (tp + 4) + (tp // 16) * 16 * (hd + 4) + 2 * (tp // 16) * 16))
        shared = max(shared, splitk)
        outputs = {plan["finals"]["o"]: None, aux["o32"]: ((bh, t, hd), "float32"), aux["lse"]: ((bh, t), "float32")}
        if trace:
            outputs[aux["trace"]] = ((2 * items, 16), "int64")
    else:
        fields.append(f"A.dout = {ptr('do')}; A.ld = pshort::Layout4{l4(s['do'])};")
        fields.append(f"A.dq = {ident(plan['finals']['dq'])}; A.ldq = pshort::Layout4{l4(o['dq'])};")
        fields.append(f"A.dk = {ident(plan['finals']['dkt'])}; A.ldk = pshort::Layout4{l4(o['dkt'])};")
        fields.append(f"A.dv = {ident(plan['finals']['dv'])}; A.ldv = pshort::Layout4{l4(o['dv'])};")
        body = (f"pshort::backward_item<{hd}, {tp}, {ns}>(A, (int)(item % {bh}LL), (int)(item / {bh}LL), smem, "
                "/*PSHORT_PREFETCHED*/0);")
        pieces = 2 * (2 * ns * tp * 24 + ns * 16 * (tp + 8))
        masks = tp * 20 + 16 * (tp + 4) if tp <= 96 else 0
        shared = 2 * 4 * tp * (hd + 8) + max(pieces, 4 * tp * (hd + 4)) + 4 * (masks + 2 * tp + (tp // 8) * 16)
        outputs = {plan["finals"]["dq"]: None, plan["finals"]["dkt"]: None, plan["finals"]["dv"]: None}
    code = ("pshort::Args A{};\n" + "\n".join(fields) +
            f"\nfor (long long item = blockIdx.x; item < {items}LL; item += gridDim.x) {{\n"
            f"  if (item != blockIdx.x) __syncthreads();  // shared reuse by the next item\n"
            + ("  A.trace_item = (int)item;\n" if trace else "") + f"  {body}\n}}")
    kind = "attention_short_" + which
    extra = dict(work_items=items, shared_bytes=shared, head_width=hd, sequence=t, head_batch=bh,
                 splits=ns, outputs=outputs)
    if which == "forward":
        extra["forward_schedule"] = "split_k" if splitk else "row_block"
    if which == "backward":
        # Early operand copies for this CTA's first item, issued by the preceding phase (see fusion.lower_fused).
        sources = {"q": (plan["sources"]["q"], 1), "k": (plan["sources"]["k"], 2), "v": (plan["sources"]["v"], 4),
                   "do": (plan["sources"]["do"], 8)}
        if mask_strides is not None:
            sources["mask"] = (group["mask"], 16)
        extra["prefetch"] = {"args": "pshort::Args A{};\n" + "\n".join(fields), "items": items,
                             "call": f"pshort::backward_prefetch<{hd}, {tp}, {ns}>(A, (int)(blockIdx.x % {bh}LL), "
                                     f"(int)(blockIdx.x / {bh}LL), smem, /*BITS*/)",
                             "sources": sources}
    return kind, code, extra
