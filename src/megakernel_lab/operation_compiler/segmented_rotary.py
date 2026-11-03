"""Segmented-attention rotary operand passes (``segmented_attention='rotary'``; off by default).

Technique transferred from Resin (github.com/ekzhang/resin @ e4807a66a0db17bcd06be2a2ac7274351d7571b5),
``models/cuda/resin_models_cuda/deepseek_v41/upstream/cuda/fused_transform.py`` (itself adapted from DeepInfer
774ce244b8): an attention operand's rotary embedding, rounding and storage layout are produced by ONE row pass in
which each token's cos/sin row is read once and reused by every head, and the inverse (``conjugate``) rotation is
the same pass run on gradients. Only the idea is reused; no Resin/DeepInfer code is copied. That kernel is an
inference KV-cache store (TileLang, head_dim 128/512, FP8/FP4 casts); here the pass is generated CUDA for H100
training, BF16 storage and Latch's existing FP32 expressions, in both forward and backward.

Joint-IR rewrite applied in ``lower._rewrite_compose_casts`` next to ``pack_fusion``. Dispatched only on graph
structure (targets, dtypes, shapes, views and readers), never on model or interval names:

``pack`` (forward). A segmented attention Q or K operand built as

    X32 = _to_copy(X, float32)      X: BF16 [R, H, P] (a projection head view)
    T   = slice(X32, -1, 0, D)      D even, D <= P
    Rr  = cat([neg(slice(T, -1, D/2, D)), slice(T, -1, 0, D/2)], -1)
    O   = cat([_to_copy(add(mul(T, C), mul(Rr, S)), bfloat16), new_zeros([R, H, P - D])], -1)

with FP32 C, S broadcast over heads ([R, 1, D]) becomes one ``segmented_rotary_pack(X, C, S)``. Per element the
generated pass evaluates the captured expressions with the build's ``--fmad=false`` roundings: ``m1 = x * c``,
``m2 = r * s``, ``bf16(m1 + (m2 * 1.0f))`` (operand order as captured), padding ``+0``. The FP32 widened copy, the
negated half, the rotated cat, the BF16 rounding buffer and the zero pad are not materialized; the five dependent
elementwise phases (each otherwise followed by a grid barrier) become one phase.

``unpack`` (backward). A segmented attention gradient G (dQ or dK, BF16 [R, H, P]) whose only consumer chain is

    Gs = slice(G, -1, 0, D);  Mc = mul(Gs, C);  Ms = mul(Gs, S)
    A  = add(add(Mc, slice_backward(slice(Ms, -1, D/2, D), [R,H,D], -1, 0, D/2)),
             slice_backward(neg(slice(Ms, -1, 0, D/2)), [R,H,D], -1, D/2, D))      (either add nesting)
    O  = _to_copy(slice_backward(A, [R, H, P], -1, 0, D), bfloat16)

becomes one ``segmented_rotary_unpack(G, C, S)`` evaluating the same products, the two ``+ (v * 1.0f)`` additions in the
captured nesting (the filler of each slice_backward is ``0.0f``, as in the generated index maps) and the BF16
rounding; padding channels are ``bf16(0.0f)``.

Every replaced intermediate must be read only inside its chain and be neither a graph output nor aliased outside
it; otherwise the chain keeps its lowering. Both rules (``pack`` and ``unpack``) run whenever the option is on.

Each pass runs one thread per (row, head): consecutive threads own consecutive head rows, and a token's cos/sin row
is loaded as FP32 pairs when both table rows are 8-byte aligned (exact copies), otherwise element by element.
"""
from __future__ import annotations

import copy

PACK_TARGET = "latch_ops.segmented_rotary_pack.default"
UNPACK_TARGET = "latch_ops.segmented_rotary_unpack.default"
TARGETS = (PACK_TARGET, UNPACK_TARGET)
FORWARD = "latch_ops.segmented_sdpa_fwd.default"
BACKWARD = "latch_ops.segmented_sdpa_bwd.default"
VERSION = 1


KINDS = frozenset({"pack", "unpack"})


def enabled():
    from .segmented_attention_rules import active
    return active("rotary")


def _target(op):
    return str(op.get("target") or "").removeprefix("torch.ops.")


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _args(op):
    return list(op.get("args") or ())


def _reads(op):
    return list(dict.fromkeys(op.get("reads") or op.get("inputs") or ()))


def _output_names(data):
    """Same walk as gather_cast._output_names: nested output refs plus values carrying output roles."""
    names = set()

    def walk(item):
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            if set(item) == {"ref"}:
                names.add(item["ref"])
            else:
                for v in item.values():
                    walk(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                walk(v)
    walk(data.get("outputs", {}))
    names |= {n for n, v in (data.get("values") or {}).items() if v.get("output_roles")}
    return names


class _Graph:
    def __init__(self, data):
        self.data = data
        self.ops = data["operations"]
        self.values = data["values"]
        self.producer = {}
        self.readers = {}
        for op in self.ops:
            for name in op.get("outputs") or ():
                self.producer[name] = op
            for name in _reads(op):
                self.readers.setdefault(name, []).append(op)
        self.outputs = _output_names(data)

    def value(self, name):
        return self.values.get(name) or {}

    def shape(self, name):
        return tuple(self.value(name).get("shape") or ())

    def dtype(self, name):
        return self.value(name).get("dtype")

    def prod(self, name, target):
        op = self.producer.get(name)
        if op is None or _target(op) != target or len(op.get("outputs") or ()) != 1:
            return None
        if op.get("effect", "pure") != "pure":
            return None
        return op

    def last_slice(self, name, src, start, stop):
        """name = slice(src, last dim, start, stop, 1) (a view)."""
        op = self.prod(name, "aten.slice.Tensor")
        if op is None:
            return False
        args = _args(op) + [None] * 5
        rank = len(self.shape(src))
        if _ref(args[0]) != src or rank == 0:
            return False
        dim, lo, hi, step = args[1], args[2], args[3], args[4] if args[4] is not None else 1
        return (isinstance(dim, int) and dim % rank == rank - 1 and lo == start
                and (hi == stop or (isinstance(hi, int) and hi >= 2 ** 62 and stop == self.shape(src)[-1]))
                and step == 1)

    def plain_cast(self, name, dtype):
        op = self.prod(name, "aten._to_copy.default")
        if op is None or self.dtype(name) != dtype or len(_args(op)) != 1:
            return None
        if any(k not in ("dtype", "layout", "device", "memory_format") for k in (op.get("kwargs") or {})):
            return None
        return op

    def binary(self, name, target):
        """(a, b) of a plain binary op (alpha 1 for add); None otherwise."""
        op = self.prod(name, target)
        if op is None:
            return None
        args, kwargs = _args(op), dict(op.get("kwargs") or {})
        if len(args) == 3 and target == "aten.add.Tensor":
            kwargs["alpha"] = args.pop()
        if len(args) != 2 or any(k != "alpha" for k in kwargs) or kwargs.get("alpha", 1) != 1:
            return None
        a, b = _ref(args[0]), _ref(args[1])
        if a is None or b is None:
            return None
        return op, a, b

    def slice_backward(self, name, src, size, start, stop):
        op = self.prod(name, "aten.slice_backward.default")
        if op is None:
            return False
        args = _args(op) + [None] * 6
        if _ref(args[0]) != src:
            return False
        sizes, dim, lo, hi, step = args[1], args[2], args[3], args[4], args[5]
        rank = len(size)
        # A VJP of x[..., a:] records end = INT64_MAX; it denotes the same range when stop is the last extent.
        open_end = isinstance(hi, int) and hi >= 2 ** 62 and stop == size[-1]
        return (list(sizes or ()) == list(size) and isinstance(dim, int) and dim % rank == rank - 1
                and lo == start and (hi == stop or open_end) and (step in (None, 1)))

    def table(self, name, rows, width):
        """C/S: FP32 [rows, 1, width] (any strides) broadcast over heads."""
        shape = self.shape(name)
        return self.dtype(name) == "float32" and shape == (rows, 1, width)


_REFUSALS = []
_REFUSAL_DETAIL = []


def _refuse():
    """None, recording the refusing source line (diagnostic for the rewrite report)."""
    import sys
    _REFUSALS.append(sys._getframe(1).f_lineno)
    return None


def _pack_chain(g, operand):
    """Recognized forward chain for one attention operand: (record, removed op names) or None."""
    shape = g.shape(operand)
    if len(shape) != 3 or g.dtype(operand) != "bfloat16" or operand in g.outputs:
        return _refuse()
    R, H, P = shape
    cat = g.prod(operand, "aten.cat.default")
    if cat is None:
        return _refuse()
    args = _args(cat)
    if len(args) != 2 or not isinstance(args[0], (list, tuple)) or len(args[0]) != 2 or args[1] % 3 != 2:
        return _refuse()
    p_name, z_name = (_ref(a) for a in args[0])
    zop = g.prod(z_name, "aten.new_zeros.default") if z_name else None
    if p_name is None or zop is None or g.dtype(z_name) != "bfloat16":
        return _refuse()
    D = g.shape(p_name)[-1] if len(g.shape(p_name)) == 3 else 0
    if g.shape(z_name) != (R, H, P - D) or not 0 < D < P or D % 2:
        return _refuse()
    half = D // 2
    cast = g.plain_cast(p_name, "bfloat16")
    if cast is None:
        return _refuse()
    a_name = _ref(_args(cast)[0])
    add = g.binary(a_name, "aten.add.Tensor") if a_name else None
    if add is None or g.dtype(a_name) != "float32" or g.shape(a_name) != (R, H, D):
        return _refuse()
    add_op, m_a, m_b = add
    muls = [g.binary(m, "aten.mul.Tensor") for m in (m_a, m_b)]
    if None in muls:
        return _refuse()
    # Identify the direct product mul(T, C) and the rotated one mul(cat([neg(T_hi), T_lo]), S).
    direct = rotated = None
    for position, (op, x, y) in zip(("first", "second"), muls):
        for t, c in ((x, y), (y, x)):
            rop = g.prod(t, "aten.cat.default")
            if rop is not None:
                rargs = _args(rop)
                if (len(rargs) == 2 and isinstance(rargs[0], (list, tuple)) and len(rargs[0]) == 2
                        and rargs[1] % 3 == 2):
                    rotated = (position, op, t, c, rop)
                    break
        else:
            direct = (position, op, x, y)
    if direct is None or rotated is None or direct[0] == rotated[0]:
        return _refuse()
    _, dop, dx, dy = direct
    # T is the operand of the direct product that is a last-dim [0, D) slice of an FP32 widened BF16 projection.
    T = coef_c = None
    for t, c in ((dx, dy), (dy, dx)):
        sop = g.prod(t, "aten.slice.Tensor")
        if sop is None:
            continue
        src = _ref(_args(sop)[0])
        if src and g.dtype(src) == "float32" and len(g.shape(src)) == 3 and g.shape(src)[:2] == (R, H):
            if g.last_slice(t, src, 0, D):
                T, coef_c, x32 = t, c, src
                break
    if T is None:
        return _refuse()
    _, rmul, rot, coef_s, rop = rotated
    neg_name, lo_name = (_ref(a) for a in _args(rop)[0])
    negop = g.prod(neg_name, "aten.neg.default") if neg_name else None
    if negop is None or len(_args(negop)) != 1:
        return _refuse()
    hi_name = _ref(_args(negop)[0])
    if not hi_name or not g.last_slice(hi_name, T, half, D) or not g.last_slice(lo_name, T, 0, half):
        return _refuse()
    x32cast = g.plain_cast(x32, "float32")
    if x32cast is None:
        return _refuse()
    X = _ref(_args(x32cast)[0])
    if X is None or g.dtype(X) != "bfloat16" or g.shape(X) != (R, H, g.shape(x32)[-1]) or g.shape(x32)[-1] < D:
        return _refuse()
    if not (g.table(coef_c, R, D) and g.table(coef_s, R, D)):
        return _refuse()
    chain = [cat, zop, cast, add_op, dop, rmul, rop, negop, g.producer[T], g.producer[hi_name],
             g.producer[lo_name], x32cast]
    # add(m_a, m_b) is emitted as m_a + (m_b * 1.0f); record which captured product comes first.
    record = dict(kind="pack", operand=operand, source=X, cos=coef_c, sin=coef_s, rows=R, heads=H, width=D,
                  padded=P, add_order="direct_first" if direct[0] == "first" else "rotated_first")
    return record, chain


def _unpack_chain(g, grad):
    """Recognized backward chain read from one attention gradient: (record, chain ops) or None."""
    shape = g.shape(grad)
    if len(shape) != 3 or g.dtype(grad) != "bfloat16":
        return _refuse()
    R, H, P = shape
    readers = [op for op in g.readers.get(grad, ()) if _target(op) == "aten.slice.Tensor"]
    if len(readers) != 1:
        return _refuse()
    sop = readers[0]
    gs = sop["outputs"][0]
    if len(g.shape(gs)) != 3:
        return _refuse()
    D = g.shape(gs)[-1]
    half = D // 2
    if not 0 < D < P or D % 2 or not g.last_slice(gs, grad, 0, D):
        return _refuse()
    widen = None
    first = g.readers.get(gs, [])
    if len(first) == 1 and _target(first[0]) == "aten._to_copy.default":
        # VJP of the forward BF16 rounding: an exact widening of the gradient slice (BF16 -> FP32).
        widen = first[0]
        wname = widen["outputs"][0]
        if g.plain_cast(wname, "float32") is None or g.shape(wname) != (R, H, D):
            return _refuse()
        gs_read = wname
    else:
        gs_read = gs
    muls = g.readers.get(gs_read, [])
    if len(muls) != 2 or any(_target(m) != "aten.mul.Tensor" for m in muls):
        _REFUSAL_DETAIL.append(dict(readers=[(_target(m), m["outputs"][0]) for m in g.readers.get(gs, [])]))
        return _refuse()
    info = {}
    for m in muls:
        b = g.binary(m["outputs"][0], "aten.mul.Tensor")
        if b is None:
            return _refuse()
        _, x, y = b
        other = y if x == gs_read else x if y == gs_read else None
        if other is None or not g.table(other, R, D) or g.dtype(m["outputs"][0]) != "float32":
            return _refuse()
        info[m["outputs"][0]] = other
    # Ms is sliced into halves; Mc is added directly.
    ms = mc = None
    for name in info:
        rs = g.readers.get(name, [])
        if rs and all(_target(r) == "aten.slice.Tensor" for r in rs):
            ms = name
        elif rs and all(_target(r) == "aten.add.Tensor" for r in rs):
            mc = name
    if ms is None or mc is None or len(g.readers[mc]) != 1:
        return _refuse()
    slices = g.readers[ms]
    if len(slices) != 2:
        return _refuse()
    s_hi = next((s["outputs"][0] for s in slices if g.last_slice(s["outputs"][0], ms, half, D)), None)
    s_lo = next((s["outputs"][0] for s in slices if g.last_slice(s["outputs"][0], ms, 0, half)), None)
    if s_hi is None or s_lo is None:
        return _refuse()
    hi_readers, lo_readers = g.readers.get(s_hi, []), g.readers.get(s_lo, [])
    if len(hi_readers) != 1 or len(lo_readers) != 1 or _target(lo_readers[0]) != "aten.neg.default":
        return _refuse()
    neg = lo_readers[0]
    nname = neg["outputs"][0]
    sb_hi_op = hi_readers[0]
    sb_hi = sb_hi_op["outputs"][0]
    neg_readers = g.readers.get(nname, [])
    if len(neg_readers) != 1:
        return _refuse()
    sb_lo_op = neg_readers[0]
    sb_lo = sb_lo_op["outputs"][0]
    if (not g.slice_backward(sb_hi, s_hi, (R, H, D), 0, half)
            or not g.slice_backward(sb_lo, nname, (R, H, D), half, D)):
        return _refuse()
    # Add nest: inner = add(x, y) over two of {mc, sb_hi, sb_lo}; outer = add(inner, third) (either operand order).
    inner_op = g.readers[mc][0]
    inner = g.binary(inner_op["outputs"][0], "aten.add.Tensor")
    if inner is None:
        return _refuse()
    _, ia, ib = inner
    leaves = {mc: "c", sb_hi: "h", sb_lo: "l"}
    if {ia, ib} - set(leaves) or mc not in (ia, ib) or ia == ib:
        return _refuse()
    inner_name = inner_op["outputs"][0]
    if len(g.readers.get(inner_name, [])) != 1:
        return _refuse()
    outer_op = g.readers[inner_name][0]
    outer = g.binary(outer_op["outputs"][0], "aten.add.Tensor")
    if outer is None:
        return _refuse()
    _, oa, ob = outer
    third = ({sb_hi, sb_lo} - {ia, ib})
    if len(third) != 1:
        return _refuse()
    third = third.pop()
    if {oa, ob} != {inner_name, third}:
        return _refuse()
    for leaf in (sb_hi, sb_lo):
        if len(g.readers.get(leaf, [])) != 1:
            return _refuse()
    # Expression strings in the captured operand order: a + (b * 1.0f).
    inner_expr = (leaves[ia], leaves[ib])
    outer_expr = ("i", leaves[third]) if oa == inner_name else (leaves[third], "i")
    a_name = outer_op["outputs"][0]
    if g.dtype(a_name) != "float32" or g.shape(a_name) != (R, H, D):
        return _refuse()
    a_readers = g.readers.get(a_name, [])
    if len(a_readers) != 1 or _target(a_readers[0]) != "aten.slice_backward.default":
        return _refuse()
    pad_op = a_readers[0]
    pad = pad_op["outputs"][0]
    if not g.slice_backward(pad, a_name, (R, H, P), 0, D):
        return _refuse()
    pad_readers = g.readers.get(pad, [])
    if len(pad_readers) != 1:
        return _refuse()
    cast = pad_readers[0]
    out = cast["outputs"][0]
    if g.plain_cast(out, "bfloat16") is None or g.shape(out) != (R, H, P):
        return _refuse()
    cos = info[mc]
    sin = info[ms]
    chain = [sop, *([widen] if widen else []), *muls, *slices, neg, sb_hi_op, sb_lo_op, inner_op, outer_op, pad_op, cast]
    record = dict(kind="unpack", gradient=grad, result=out, cos=cos, sin=sin, rows=R, heads=H, width=D, padded=P,
                  inner=list(inner_expr), outer=list(outer_expr))
    return record, chain


def _closed(g, chain, keep):
    """Every chain value is read only inside the chain (or is the kept result) and is not a graph output."""
    names = {op["name"] for op in chain}
    for op in chain:
        for v in op.get("outputs") or ():
            if v == keep:
                continue
            if v in g.outputs:
                return False
            if any(r["name"] not in names for r in g.readers.get(v, ())):
                return False
            if any(val.get("alias_of") == v and n not in {o for c in chain for o in c.get("outputs") or ()}
                   for n, val in g.values.items()):
                return False
    return True


def rewrite(graph):
    """(graph, report); the input is never mutated. Unchanged when disabled or nothing is recognized."""
    if not enabled():
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or not isinstance(values, dict):
        return graph, None
    chosen = KINDS
    g = _Graph(data)
    attention = [op for op in ops if _target(op) in (FORWARD, BACKWARD)]
    if not attention:
        return graph, None
    report = {"version": VERSION, "kinds": sorted(chosen), "packs": [], "unpacks": [], "refused": []}
    replacements = {}   # removed op name -> new op (placed at the position of the result's producer)
    removed = set()
    for att in attention:
        args = _args(att)
        if _target(att) == FORWARD and "pack" in chosen:
            for operand in (_ref(args[0]), _ref(args[1])):
                if operand is None or operand in replacements:
                    continue
                found = _pack_chain(g, operand)
                if found is None:
                    report["refused"].append(dict(kind="pack", operand=operand, reason="pattern",
                                                  line=_REFUSALS[-1] if _REFUSALS else None))
                    continue
                record, chain = found
                if not _closed(g, chain, operand) or removed & {op["name"] for op in chain}:
                    report["refused"].append(dict(kind="pack", operand=operand, reason="intermediate read outside"))
                    continue
                reads = [record["source"], record["cos"], record["sin"]]
                new = dict(name=operand, target=PACK_TARGET,
                           args=[{"ref": record["source"]}, {"ref": record["cos"]}, {"ref": record["sin"]},
                                 record["width"], record["add_order"]],
                           kwargs={}, inputs=reads, outputs=[operand], dependencies=[], reads=reads, writes=[operand],
                           reduction_axes=[], synchronization="global_phase_boundary", effect="pure")
                replacements[operand] = new
                removed |= {op["name"] for op in chain}
                report["packs"].append(record)
        if _target(att) == BACKWARD and "unpack" in chosen:
            for grad in list(att.get("outputs") or ())[:2]:
                found = _unpack_chain(g, grad)
                if found is None:
                    report["refused"].append(dict(kind="unpack", gradient=grad, reason="pattern",
                                                  line=_REFUSALS[-1] if _REFUSALS else None,
                                                  detail=_REFUSAL_DETAIL.pop() if _REFUSAL_DETAIL else None))
                    continue
                record, chain = found
                out = record["result"]
                if not _closed(g, chain, out) or removed & {op["name"] for op in chain}:
                    report["refused"].append(dict(kind="unpack", gradient=grad, reason="intermediate read outside"))
                    continue
                reads = [grad, record["cos"], record["sin"]]
                new = dict(name=out, target=UNPACK_TARGET,
                           args=[{"ref": grad}, {"ref": record["cos"]}, {"ref": record["sin"]}, record["width"],
                                 "".join(record["inner"]) + "|" + "".join(record["outer"])],
                           kwargs={}, inputs=reads, outputs=[out], dependencies=[], reads=reads, writes=[out],
                           reduction_axes=[], synchronization="global_phase_boundary", effect="pure")
                replacements[out] = new
                removed |= {op["name"] for op in chain}
                report["unpacks"].append(record)
    if not replacements:
        return graph, (report if report["refused"] else None)
    producer_of, new_ops, placed = {}, [], set()
    for op in ops:
        if op["name"] in removed:
            for out in op.get("outputs") or ():
                if out in replacements and out not in placed:
                    new = replacements[out]
                    new["dependencies"] = list(dict.fromkeys(producer_of[r] for r in new["reads"] if r in producer_of))
                    new_ops.append(new)
                    producer_of[out] = new["name"]
                    placed.add(out)
            continue
        new_ops.append(op)
        for o in op.get("outputs") or ():
            producer_of[o] = op["name"]
    if placed != set(replacements):
        raise AssertionError("segmented_rotary: replacement placement incomplete")
    produced = {o for op in new_ops for o in op.get("outputs") or ()}
    read = {r for op in new_ops for r in _reads(op)}
    dropped = []
    for name in list(values):
        v = values[name]
        if name in produced or name in data.get("inputs", ()) or name in g.outputs:
            continue
        producer = (g.producer.get(name) or {}).get("name")
        if producer in removed and name not in read:
            dropped.append(name)
            del values[name]
    for name in replacements:
        values[name] = {**values[name], "producer": name, "alias_of": None}
    data["operations"] = new_ops
    data.setdefault("metadata", {})["segmented_rotary"] = report
    report["dropped_values"] = len(dropped)
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def _ptr(layouts, ident, key):
    layout = layouts[key]
    return f"({ident(layout.root)} + {layout.offset}LL)"


def _table_access(layouts, dtypes, name, rows, width):
    from .lower import LoweringError
    layout = layouts[name]
    if tuple(layout.shape) != (rows, 1, width) or dtypes[name] != "float32":
        raise LoweringError("segmented rotary tables must be FP32 [rows, 1, width]")
    return layout.strides[0], layout.strides[2]


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """One generated row pass per recognized rotary pack/unpack (see the module docstring)."""
    from .lower import LoweringError
    target = str(field(op, "target")).removeprefix("torch.ops.")
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ()))
    if target not in TARGETS or len(args) != 5 or len(outputs) != 1:
        raise LoweringError("segmented rotary ABI")
    src, cos, sin = (ref(a, names) for a in args[:3])
    D, order = int(args[3]), str(args[4])
    name = outputs[0]
    shape = tuple(field(values[name], "shape"))
    R, H, P = shape
    ls = layouts[src]
    if (len(ls.shape) != 3 or tuple(ls.shape[:2]) != (R, H) or dtypes[src] != "bfloat16"
            or dtypes[name] != "bfloat16" or D % 2 or not 0 < D < P or P % 8 or ls.shape[-1] < D
            or R * H * P >= 2 ** 31):
        raise LoweringError("segmented rotary shape/dtype contract")
    define(name, shape, None, "bfloat16")
    half = D // 2
    cr, cd = _table_access(layouts, dtypes, cos, R, D)
    sr, sd = _table_access(layouts, dtypes, sin, R, D)
    s0, s1, s2 = ls.strides
    pc, ps, px, po = (_ptr(layouts, ident, k) for k in (cos, sin, src, name))
    # Per output channel d < D the captured expressions; x_[] holds the widened source row (exact BF16->FP32).
    if target == PACK_TARGET:
        if order not in ("direct_first", "rotated_first"):
            raise LoweringError("segmented rotary pack order")
        add = "(m1_ + (m2_ * 1.0f))" if order == "direct_first" else "(m2_ + (m1_ * 1.0f))"
        body = f"""        const float r_ = d < {half} ? (-x_[d + {half}]) : x_[d - {half}];
        const float m1_ = (x_[d] * c_[d]);
        const float m2_ = (r_ * s_[d]);
        o_[d] = __float2bfloat16_rn({add});"""
        kind, engine = "segmented_rotary_pack", "segmented_rotary_pack"
    else:
        inner, outer = order.split("|")
        if sorted(inner) not in (["c", "h"], ["c", "l"]) or len(outer) != 2 or "i" not in outer:
            raise LoweringError("segmented rotary unpack order")
        # Leaves: c = G*cos at d; h = slice_backward of (G*sin)[D/2:] (rows d < D/2); l = slice_backward of
        # -(G*sin)[:D/2] (rows d >= D/2); fillers 0.0f as in the generated index maps.
        leaf = {"c": "gc_", "h": f"(d < {half} ? (x_[d + {half}] * s_[d + {half}]) : 0.0f)",
                "l": f"(d >= {half} ? (-(x_[d - {half}] * s_[d - {half}])) : 0.0f)", "i": "in_"}
        third = outer.replace("i", "")
        inner_expr = f"({leaf[inner[0]]} + ({leaf[inner[1]]} * 1.0f))"
        outer_expr = (f"(in_ + ({leaf[third]} * 1.0f))" if outer[0] == "i" else f"({leaf[third]} + (in_ * 1.0f))")
        body = f"""        const float gc_ = (x_[d] * c_[d]);
        const float in_ = {inner_expr};
        o_[d] = __float2bfloat16_rn({outer_expr});"""
        kind, engine = "segmented_rotary_unpack", "segmented_rotary_unpack"
    vec = s2 == 1 and s1 % 8 == 0 and s0 % 8 == 0 and ls.offset % 8 == 0
    if vec:
        load = f"""      const __nv_bfloat16* xs_ = {px} + (long long)row_ * {s0}LL + (long long)h_ * {s1}LL;
#pragma unroll
      for (int q = 0; q < {(D + 7) // 8}; ++q) {{
        const uint4 w_ = *reinterpret_cast<const uint4*>(xs_ + 8 * q);
        const __nv_bfloat16* e_ = reinterpret_cast<const __nv_bfloat16*>(&w_);
#pragma unroll
        for (int k = 0; k < 8; ++k) if (8 * q + k < {D}) x_[8 * q + k] = __bfloat162float(e_[k]);
      }}"""
    else:
        load = f"""      const __nv_bfloat16* xs_ = {px} + (long long)row_ * {s0}LL + (long long)h_ * {s1}LL;
#pragma unroll
      for (int d = 0; d < {D}; ++d) x_[d] = __bfloat162float(xs_[(long long)d * {s2}LL]);"""
    store = f"""    __nv_bfloat16 o_[{P}];
#pragma unroll
    for (int d = 0; d < {P}; ++d) {{
      if (d < {D}) {{
{body}
      }} else {{
        o_[d] = __float2bfloat16_rn(0.0f);
      }}
    }}
    __nv_bfloat16* dst_ = {po} + ((long long)row_ * {H}LL + (long long)h_) * {P}LL;
    if ((reinterpret_cast<uintptr_t>(dst_) & 15u) == 0) {{
#pragma unroll
      for (int q = 0; q < {P // 8}; ++q) reinterpret_cast<uint4*>(dst_)[q] = reinterpret_cast<const uint4*>(o_)[q];
    }} else {{
#pragma unroll
      for (int d = 0; d < {P}; ++d) dst_[d] = o_[d];
    }}"""
    tables = f"""  float c_[{D}], s_[{D}];
#pragma unroll
  for (int d = 0; d < {D}; ++d) {{
    c_[d] = {pc}[(long long)row_ * {cr}LL + (long long)d * {cd}LL];
    s_[d] = {ps}[(long long)row_ * {sr}LL + (long long)d * {sd}LL];
  }}"""
    if cd == 1 and sd == 1 and cr % 2 == 0 and sr % 2 == 0 and D % 2 == 0:
        # Same values loaded as FP32 pairs when both table rows are 8-byte aligned (exact copies).
        tables = f"""  float c_[{D}], s_[{D}];
  const float* cp_ = {pc} + (long long)row_ * {cr}LL;
  const float* sp_ = {ps} + (long long)row_ * {sr}LL;
  if (((reinterpret_cast<uintptr_t>(cp_) | reinterpret_cast<uintptr_t>(sp_)) & 7u) == 0) {{
#pragma unroll
    for (int d = 0; d < {D}; d += 2) {{
      const float2 cv_ = *reinterpret_cast<const float2*>(cp_ + d);
      const float2 sv_ = *reinterpret_cast<const float2*>(sp_ + d);
      c_[d] = cv_.x; c_[d + 1] = cv_.y; s_[d] = sv_.x; s_[d + 1] = sv_.y;
    }}
  }} else {{
#pragma unroll
    for (int d = 0; d < {D}; ++d) {{ c_[d] = cp_[d]; s_[d] = sp_[d]; }}
  }}"""
    # One thread per (row, head): consecutive threads own consecutive head rows; the row's cos/sin are re-read by
    # its H threads from L1 (one table row serves every head of a token).
    code = f"""for (unsigned t_ = blockIdx.x * blockDim.x + threadIdx.x; t_ < {R * H}u; t_ += gridDim.x * blockDim.x) {{
  const unsigned row_ = t_ / {H}u, h_ = t_ % {H}u;
{tables}
  {{
    float x_[{D}];
{load}
{store}
  }}
}}"""
    items = (R * H + 255) // 256
    reads = [src, cos, sin]
    return [dict(kind=kind, names=[name], targets=[target], reads=reads, writes=[name], code=code, threads=256,
                 work_items=items, shared_bytes=0, engine=engine + "_rowhead",
                 arithmetic="per element the captured rotary/padding expressions and roundings (--fmad=false)")]
