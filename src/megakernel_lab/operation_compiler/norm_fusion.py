"""Opt-in normalization / activation / residual fusion rules for flat common compose programs (LATCH-TINYMOE-FUSE).

Selected through the normal build API: ``CapturedModule.build(..., norm_fusion="rows")`` (every rule) or a
'+'-joined subset of ``RULES`` (ablation). Omitting it (None) leaves lowering and generated source unchanged.

Every rule is dispatched on recognized operation structure (targets, shapes, dtypes, layouts and dependencies),
never on model names, and falls back to the unchanged lowering when its pattern is not proven:

* ``colreg`` -- normalization weight gradients. A reduction over the leading rows of a contiguous FP32 [R, C]
  product ``dy * n`` whose factor ``n = x * rsqrt(mean(x**2) + eps)`` is a recognized RMS-normalized row uses the
  existing coalesced exact column engine (P8-B ``colreg``: lanes over columns, the frozen per-warp CTA tree
  replayed symbolically; bitwise equal to the warp-per-column engine it replaces, whose lanes stride rows).
* ``colgrp`` -- the same recognized sums with the P-OPS3-C column engine, 4 columns per CTA (96 CTAs for 384
  columns instead of colreg's 12): identical leaves, per-thread order, butterfly and second stage.
* ``rope`` -- (user-approved scope extension) the element-wise / index-map chains between the q/k/v projections
  and a composed attention call, and between its gradients and the projection backward products, when they carry a
  rotary embedding (``x * cos + cat(-x[..., h:], x[..., :h]) * sin`` forward; ``slice_backward``/``neg`` pairs
  backward): casts, products, sums, negations, cats (rotate-half, head-width padding), slice_backwards and copies of
  that region only are admitted to the existing P8-C ``vprod`` inline evaluation (an index-map/elementwise producer
  whose readers are all elementwise is evaluated inside its readers instead of being stored). Same expressions and
  roundings; the stored attention operands and projection inputs are computed in one pass from the projections.
* ``dcast`` -- (user-approved scope extension, forward dispatch only) a row gather ``embedding(x, idx)`` of a
  recognized weight-scaled normalized row whose only readers are identical dtype casts is replaced by the gather of
  the cast row ``embedding(cast(x), idx)``: a cast commutes with a row copy, so every element has the same bits; the
  FP32 gathered rows are no longer written and re-read, and with castcse the row cast is the one the router
  projection already reads.
* ``rpack`` -- (user-approved scope extension, forward only) an attention operand built as
  ``clone(cat([P, zeros], -1))`` (head-width padding) where P is a projection head view, its rotary embedding
  ``bf16(x * cos + cat(-x[..., h:], x[..., :h]) * sin)`` and/or its grouped-query expansion
  ``view(clone(expand(unsqueeze(R, 2))))`` is produced by one generated row pass (``RPACK_TARGET``): per output
  element the same expressions and roundings as the captured chain (``bf16(-x)``, ``bf16(x) * cos``,
  ``bf16(r) * sin``, ``a + (b * 1.0f)``, ``bf16(.)``, +0 padding), indices computed once per row instead of through
  seven separately stored and re-read intermediates. Unrecognized chains keep their lowering.
* ``rpackb`` -- (user-approved scope extension) the backward counterparts: an attention gradient's contiguous copy
  for its projection product ``clone(transpose(...))`` built from the rotary backward (``bf16(F * sin)`` rotated
  back through ``slice_backward``/``neg`` with the explicit +0 additions, plus ``bf16(F * cos)``, BF16 additions),
  from a grouped-query head sum (``sum(view(.), 2)``, the exact two-level leaf tree) or both, and the zero-padded
  output cotangent ``clone(slice_backward(.))``, each produced by one generated row pass with the same expressions
  and roundings (``RPACKB_TARGET``). Unrecognized chains keep their lowering.
* ``nremat`` -- recognized RMS-normalized rows ``n = x * r`` (FP32, ``r`` a row-broadcast rsqrt statistic) that
  are read again by the backward pass are recomputed right before their first late reader from the same stored
  ``x`` and ``r`` (C-OPS2-D late-operand rematerialization, restricted to this pattern), instead of being written
  in the forward pass and re-read. The late readers already load ``x`` and ``r``; the copy is the same IEEE
  multiply of the same operand values, so every element has the bits of the stored one.
* ``gremat`` -- the same rule for a gated activation's ``silu(u)`` read again by its backward product.
* ``gatedbw`` -- the two backward chains of a gated activation ``silu(u) * g`` (the ``g`` cotangent
  ``dy * silu(u)`` and the ``u`` cotangent ``dy * g * silu'(u)``) stay in one elementwise loop: they wait for the
  same newest product ``dy`` and read the same ``u``; split_components would otherwise separate them because their
  older forward operands differ. Expressions are unchanged; each loads ``dy`` and ``u`` once.
* ``castcse`` -- identical dtype casts of one normalized row (one per projection that reads it), or of one row
  gather of it (one per gated-activation input product), are evaluated and stored once; every reader reads the
  same bits.
* ``resid`` -- a residual addition ``h + sum(a * w, dim=1)`` whose addend is a small k-way (k <= 8) weighted sum
  over the middle axis of [T, k, D] (read only by residual additions) is expanded into k same-shape products and the
  exact tree the k-leaf reduction evaluates (leaves ``0.0f + p_j``, the xor-butterfly pairing of exact_reduction);
  (SMELT-LATCH) a scalar-constant scale between the sum and the residual additions (``h + c * sum(...)``, the
  looped-span 1/r residual scale) is admitted and kept as written.
  The products, the tree and the residual addition then form one elementwise chain, which the row fusion of the
  following normalization takes as its prologue: the [T, k, D] FP32 products and the [T, D] sum are no longer
  written, re-read or separated by two grid barriers. Same operands, operation order and rounding.

Rules change where values are computed or stored and which threads evaluate them, never an expression, operand
or reduction order.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

RULES = ("colreg", "nremat", "gremat", "gatedbw", "castcse", "resid", "colgrp", "rope", "dcast", "rpack", "rpackb",
         "pcse")
# "rows" is the selected set (LATCH-TINYMOE-FUSE screens): colreg/gremat/gatedbw/rope/dcast measured no gain or
# a loss on TinyMoE and remain selectable individually. rope, dcast, rpack and rpackb belong to the user-approved
# scope extension (attention-boundary and forward-dispatch data movement).
PRESETS = {"rows": ("colgrp", "nremat", "castcse", "resid", "rpack", "rpackb")}
_RULES = ContextVar("latch_norm_fusion", default=None)
# Names of recognized normalization weight-gradient column sums for the current lowering (colreg scope).
_COLUMN_SUMS = ContextVar("latch_norm_fusion_column_sums", default=frozenset())
VERSION = 1


def rules_of(value):
    if value is None:
        return ()
    if not isinstance(value, str) or not value:
        raise ValueError(f"norm_fusion must be None, one of {sorted(PRESETS)} or '+'-joined rules from {RULES}")
    if value in PRESETS:
        return tuple(PRESETS[value])
    # A preset name may appear inside a '+'-joined list (e.g. "rows+pcse"); it expands in place.
    chosen = tuple(dict.fromkeys(r for part in value.split("+") for r in PRESETS.get(part, (part,))))
    if not chosen or set(chosen) - set(RULES):
        raise ValueError(f"norm_fusion must be None, one of {sorted(PRESETS)} or '+'-joined rules from {RULES}")
    return chosen


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()


def active(rule):
    return rule in rules_of(_RULES.get())


@contextmanager
def scope(value):
    token = _RULES.set(validate(value))
    try:
        yield
    finally:
        _RULES.reset(token)


def features():
    """Extra group-rule features for the standalone flat common planner."""
    return "+gatedbw" if active("gatedbw") else ""


def column_sum(name):
    """colreg scope: `name` is a recognized normalization weight-gradient column sum."""
    return active("colreg") and not active("colgrp") and name in _COLUMN_SUMS.get()


def column_group(name):
    """colgrp scope: `name` is a recognized weight-gradient column sum for the 4-column P-OPS3-C engine."""
    return active("colgrp") and name in _COLUMN_SUMS.get()


_ROPE = ContextVar("latch_norm_fusion_rope", default=frozenset())


def rope_value(name):
    """rope scope: `name` belongs to a recognized rotary attention-boundary chain (vprod admission)."""
    return name in _ROPE.get()


def rope_scope():
    return bool(_ROPE.get())


@contextmanager
def rope_values(names):
    token = _ROPE.set(frozenset(names))
    try:
        yield
    finally:
        _ROPE.reset(token)


@contextmanager
def column_sums(names):
    token = _COLUMN_SUMS.set(frozenset(names))
    try:
        yield
    finally:
        _COLUMN_SUMS.reset(token)


# ---------------------------------------------------------------- recognition

def _base(op):
    return op.target.removeprefix("aten.").split(".")[0]


def _ref(item):
    from .ir import Ref
    return item.name if isinstance(item, Ref) else None


def _plain_cast(op):
    """A dtype conversion whose other keyword arguments do not affect values (strided layout, device)."""
    kw = op.kwargs
    return (set(kw) <= {"dtype", "layout", "device"} and str(kw.get("layout", "strided")).endswith("strided")
            and not kw.get("memory_format"))


def _contiguous(value):
    stride, expected = 1, []
    for d in reversed(value.shape):
        expected.append(stride)
        stride *= d
    return tuple(reversed(expected)) == tuple(value.strides) and value.storage_offset == 0


class _Graph:
    def __init__(self, graph):
        self.graph = graph
        self.values = graph.values
        self.producer = {op.outputs[0]: op for op in graph.operations if len(op.outputs) == 1}
        self.readers = {}
        for op in graph.operations:
            for name in dict.fromkeys(op.reads):
                self.readers.setdefault(name, []).append(op)

    def through_views(self, name, limit=8):
        """The non-view producer behind a chain of pure views/identity casts of `name`."""
        for _ in range(limit):
            op = self.producer.get(name)
            if op is None:
                return name, None
            base = _base(op)
            same = (base in ("alias", "detach", "view", "_unsafe_view", "reshape", "clone")
                    or (base == "_to_copy" and self.values[op.outputs[0]].dtype == self.values[op.reads[0]].dtype
                        and set(op.kwargs) <= {"dtype"}))
            if not same or not op.reads:
                return name, op
            name = op.reads[0]
        return name, self.producer.get(name)

    def rms_statistic(self, name):
        """`name` = rsqrt(mean(pow(x, 2), [-1], keepdim) + eps) over the last axis of FP32 x; returns x or None."""
        _, op = self.through_views(name)
        if op is None or _base(op) != "rsqrt" or self.values[op.outputs[0]].dtype != "float32":
            return None
        _, add = self.through_views(op.reads[0])
        if add is None or _base(add) != "add" or len(add.reads) != 1:
            return None
        _, mean = self.through_views(add.reads[0])
        if mean is None or _base(mean) != "mean" or len(mean.reads) != 1:
            return None
        dims = mean.args[1] if len(mean.args) > 1 else mean.kwargs.get("dim")
        keep = mean.args[2] if len(mean.args) > 2 else mean.kwargs.get("keepdim", False)
        _, square = self.through_views(mean.reads[0])
        if square is None or _base(square) != "pow" or len(square.reads) != 1 or square.args[1:2] != (2,):
            return None
        x = square.reads[0]
        xv = self.values[x]
        if xv.dtype != "float32" or not keep or list(dims or ()) not in ([-1], [len(xv.shape) - 1]):
            return None
        return x

    def normalized(self, op):
        """op = mul(x, r) / mul(r, x) of a recognized RMS statistic r of x: the FP32 normalized row."""
        if _base(op) != "mul" or op.target != "aten.mul.Tensor" or len(op.reads) != 2 or len(op.outputs) != 1:
            return False
        out = self.values[op.outputs[0]]
        if out.dtype != "float32" or not _contiguous(out):
            return False
        a, b = (_ref(x) for x in op.args[:2])
        if a is None or b is None:
            return False
        for x, r in ((a, b), (b, a)):
            source = self.rms_statistic(r)
            if source is not None and self.through_views(x)[0] == self.through_views(source)[0]:
                return True
        return False

    def normalized_source(self, op, gathers=True):
        """op is a normalized row, its weight-scaled form w * n, or (gathers) a row embedding gather of either."""
        if self.normalized(op):
            return True
        if op.target == "aten.mul.Tensor" and len(op.reads) == 2:
            for r in op.reads:
                q = self.through_views(r)[1]
                if q is not None and self.normalized(q):
                    return True
        if gathers and op.target == "aten.embedding.default" and op.args and _ref(op.args[0]) is not None:
            q = self.through_views(_ref(op.args[0]))[1]
            return q is not None and self.normalized_source(q, gathers=False)
        return False

    def gated_silu(self, op):
        """op = silu(u) whose result multiplies another value of the same shape (a gated activation)."""
        if op.target != "aten.silu.default" or len(op.outputs) != 1:
            return False
        v = op.outputs[0]
        return any(_base(r) == "mul" and len(r.reads) == 2 and self.values[r.outputs[0]].shape == self.values[v].shape
                   for r in self.readers.get(v, ()))


def weight_gradient_sums(graph):
    """Leading-axis sums of dy * n where n is a recognized normalized row (normalization weight gradients)."""
    g = _Graph(graph)
    found = []
    for op in graph.operations:
        if _base(op) != "sum" or len(op.reads) != 1 or len(op.outputs) != 1:
            continue
        src_name, src = g.through_views(op.reads[0])
        if src is None or src.target != "aten.mul.Tensor" or len(src.reads) != 2:
            continue
        sv = g.values[src_name]
        dims = op.args[1] if len(op.args) > 1 else op.kwargs.get("dim")
        dims = [d % len(sv.shape) for d in (dims or ())]
        if not dims or dims != list(range(len(dims))) or len(dims) >= len(sv.shape) or sv.dtype != "float32":
            continue
        if any(g.normalized(g.through_views(r)[1]) for r in src.reads if g.through_views(r)[1] is not None):
            found.append(op.outputs[0])
    return found


def _cse_casts(graph, report):
    """castcse: merge identical dtype casts of one recognized normalized-row value (same source, same kwargs)."""
    from .remat import _rewrite
    g = _Graph(graph)
    protected = set(graph.inputs)
    from .remat import _names
    _names(graph.outputs, protected)
    _names(getattr(graph, "state_writes", None) or {}, protected)
    first, mapping = {}, {}
    for op in graph.operations:
        if (op.target != "aten._to_copy.default" or len(op.reads) != 1 or op.effect != "pure"
                or len(op.outputs) != 1 or op.outputs[0] in protected or set(op.kwargs) - {"dtype"}):
            continue
        out = g.values[op.outputs[0]]
        if out.alias_of is not None or not _contiguous(out):
            continue
        src_name, src = g.through_views(op.reads[0])
        # The cast source is a normalized row, its weight-scaled form w * n, or a row gather of either.
        if src is None or not g.normalized_source(src):
            continue
        key = (op.reads[0], repr(op.kwargs.get("dtype")), tuple(out.shape), tuple(out.strides))
        if key in first:
            mapping[op.outputs[0]] = first[key]
        else:
            first[key] = op.outputs[0]
    if not mapping:
        return graph
    removed = {g.producer[m].name: g.producer[mapping[m]].name for m in mapping}
    ops = []
    for op in graph.operations:
        if op.name in removed:
            continue
        if any(r in mapping for r in op.reads) or any(d in removed for d in op.dependencies):
            op = replace(op, args=_rewrite(op.args, mapping), kwargs=_rewrite(op.kwargs, mapping),
                         inputs=tuple(mapping.get(x, x) for x in op.inputs),
                         reads=tuple(mapping.get(x, x) for x in op.reads),
                         dependencies=tuple(dict.fromkeys(removed.get(d, d) for d in op.dependencies)))
        ops.append(op)
    values = {k: (replace(v, alias_of=mapping[v.alias_of]) if v.alias_of in mapping else v)
              for k, v in graph.values.items() if k not in mapping}
    report["merged_casts"] = sorted(mapping.items())
    return replace(graph, operations=ops, values=values)


def _cse_parameter_derivations(graph, report):
    """pcse (SMELT-LATCH, opt-in): pure operations whose every read is a parameter input, or a value already derived
    only from parameter inputs (casts, stacks, transposes/views of weights), are evaluated once per update; identical
    operations (same target, same mapped arguments/keyword arguments, same result layout) read the first result.

    Parameters are not written before the update's state commit, so every such operation reads the same bits
    wherever it appears; a weight-shared (looped) layer otherwise recasts/restacks its weights at every visit.
    Expressions and rounding are unchanged; only duplicates are removed."""
    from .remat import _names, _rewrite
    from .ir import Ref
    g = _Graph(graph)
    protected = set()
    _names(graph.outputs, protected)
    _names(getattr(graph, "state_writes", None) or {}, protected)
    invariant = {name for name in graph.inputs if g.values[name].role == "parameter"}
    first, mapping = {}, {}

    def key_of(item):
        if isinstance(item, Ref):
            return ("ref", mapping.get(item.name, item.name))
        if isinstance(item, (list, tuple)):
            return (type(item).__name__, tuple(key_of(x) for x in item))
        if isinstance(item, dict):
            return ("dict", tuple(sorted((k, key_of(v)) for k, v in item.items())))
        return ("const", repr(item))

    for op in graph.operations:
        if (op.effect != "pure" or len(op.outputs) != 1 or not op.reads or op.outputs[0] in protected
                or not all(mapping.get(r, r) in invariant for r in op.reads)):
            continue
        out = g.values[op.outputs[0]]
        key = (op.target, key_of(op.args), key_of(op.kwargs), tuple(out.shape), tuple(out.strides), out.dtype,
               None if out.alias_of is None else mapping.get(out.alias_of, out.alias_of), out.storage_offset)
        if key in first:
            mapping[op.outputs[0]] = first[key]
        else:
            first[key] = op.outputs[0]
            invariant.add(op.outputs[0])
    if not mapping:
        report["parameter_cse"] = dict(merged=0)
        return graph
    removed = {g.producer[m].name: g.producer[mapping[m]].name for m in mapping}
    ops = []
    for op in graph.operations:
        if op.name in removed:
            continue
        if any(r in mapping for r in op.reads) or any(d in removed for d in op.dependencies):
            op = replace(op, args=_rewrite(op.args, mapping), kwargs=_rewrite(op.kwargs, mapping),
                         inputs=tuple(mapping.get(x, x) for x in op.inputs),
                         reads=tuple(dict.fromkeys(mapping.get(x, x) for x in op.reads)),
                         dependencies=tuple(dict.fromkeys(removed.get(d, d) for d in op.dependencies)))
        ops.append(op)
    values = {k: (replace(v, alias_of=mapping[v.alias_of]) if v.alias_of in mapping else v)
              for k, v in graph.values.items() if k not in mapping}
    report["parameter_cse"] = dict(merged=len(mapping), targets=dict(sorted(
        __import__("collections").Counter(g.producer[m].target for m in mapping).items())))
    return replace(graph, operations=ops, values=values)


def _butterfly_pairs(k):
    """Nested (a, b) pairing of leaves 0..k-1 in codegen_rules._butterfly0 order (a + b evaluated as written)."""
    lanes = list(range(k)) + [None] * (32 - k)
    for offset in (16, 8, 4, 2, 1):
        lanes = [a if b is None else b if a is None else (a, b)
                 for a, b in zip(lanes, (lanes[i ^ offset] for i in range(32)))]
    return lanes[0]


RESID_MAX_K = 8


def _expand_residual_addends(graph, report):
    """resid: expand recognized k-way weighted residual addends into same-shape elementwise products and tree."""
    from .ir import Ref, Operation, Value
    g = _Graph(graph)
    protected = set(graph.inputs)
    from .remat import _names
    _names(graph.outputs, protected)
    _names(getattr(graph, "state_writes", None) or {}, protected)
    used = set(graph.values)

    def fresh(stem):
        k = 1
        while f"{stem}__rk{k}" in used:
            k += 1
        used.add(f"{stem}__rk{k}")
        return f"{stem}__rk{k}"

    def terminals(name, limit=8):
        """Non-view readers reached from `name` through pure views."""
        out, stack = [], [(name, 0)]
        while stack:
            v, depth = stack.pop()
            for r in g.readers.get(v, ()):
                if _base(r) in ("view", "_unsafe_view", "reshape", "alias") and len(r.outputs) == 1 and depth < limit:
                    stack.append((r.outputs[0], depth + 1))
                else:
                    out.append(r)
        return out

    replaced, rewrites = {}, []
    for op in graph.operations:
        if op.target != "aten.sum.dim_IntList" or len(op.reads) != 1 or len(op.outputs) != 1:
            continue
        dims = op.args[1] if len(op.args) > 1 else op.kwargs.get("dim")
        keep = op.args[2] if len(op.args) > 2 else op.kwargs.get("keepdim", False)
        s_name = op.outputs[0]
        m_name = op.reads[0]
        mv, sv = g.values[m_name], g.values[s_name]
        mop = g.producer.get(m_name)
        if (keep or op.kwargs.get("dtype") is not None or len(mv.shape) != 3 or list(dims or ()) not in ([1], [-2])
                or not 2 <= mv.shape[1] <= RESID_MAX_K or sv.dtype != "float32" or mv.dtype != "float32"
                or not _contiguous(sv) or not _contiguous(mv) or s_name in protected or m_name in protected
                or mop is None or mop.target != "aten.mul.Tensor" or len(mop.reads) != 2 or len(mop.args) != 2
                or mop.kwargs or [r.name for r in g.readers.get(m_name, ())] != [op.name]):
            continue
        T, k, D = mv.shape
        factors = [_ref(a) for a in mop.args]
        if None in factors:
            continue
        fv = [g.values[f] for f in factors]
        shapes = sorted(tuple(v.shape) for v in fv)
        if shapes != sorted([(T, k, D), (T, k, 1)]):
            continue
        # The addend is read only by residual additions of FP32 values of its shape, or (SMELT-LATCH) by an
        # FP32 scalar-constant scale ``c * sum`` (looped-span residual scale 1/r) that is itself read only by such
        # residual additions. The scale operation is kept unchanged; only the sum is expanded, so expressions and
        # rounding are those of the captured program.
        def residual(e):
            return (e.target == "aten.add.Tensor" and len(e.reads) == 2 and e.kwargs.get("alpha", 1) == 1
                    and len(e.args) <= 2 and g.values[e.outputs[0]].dtype == "float32"
                    and g.values[e.outputs[0]].numel == sv.numel)

        def constant_scale(e):
            if (e.target not in ("aten.mul.Tensor", "aten.mul.Scalar") or len(e.reads) != 1 or e.kwargs
                    or len(e.args) != 2 or len(e.outputs) != 1 or e.outputs[0] in protected):
                return False
            constants = [a for a in e.args if _ref(a) is None]
            out = g.values[e.outputs[0]]
            if (len(constants) != 1 or type(constants[0]) not in (float, int) or out.dtype != "float32"
                    or out.numel != sv.numel or not _contiguous(out)):
                return False
            later = terminals(e.outputs[0])
            return bool(later) and all(residual(x) for x in later)

        ends = terminals(s_name)
        if not ends or not all(residual(e) or constant_scale(e) for e in ends):
            continue
        scaled = [e.name for e in ends if not residual(e)]
        new_ops, new_values, leaves = [], {}, []
        for j in range(k):
            refs = []
            for f, v in zip(factors, fv):
                sel = fresh(f"{f}_s{j}")
                root = v.alias_of or f
                new_values[sel] = Value(sel, (T, v.shape[2]), v.dtype, (v.strides[0], v.strides[2]), role="temporary",
                                        alias_of=root, storage_offset=v.storage_offset + j * v.strides[1], producer=sel)
                new_ops.append(Operation(sel, "aten.select.int", (Ref(f), 1, j), {}, (f,), (sel,), (), (f,), (),
                                         synchronization="view"))
                refs.append(sel)
            prod = fresh(f"{m_name}_p{j}")
            new_values[prod] = Value(prod, (T, D), "float32", (D, 1), role="temporary", producer=prod)
            new_ops.append(Operation(prod, "aten.mul.Tensor", tuple(Ref(r) for r in refs), {}, tuple(refs), (prod,), (),
                                     tuple(refs), (prod,)))
            leaf = fresh(f"{s_name}_l{j}")
            new_values[leaf] = Value(leaf, (T, D), "float32", (D, 1), role="temporary", producer=leaf)
            new_ops.append(Operation(leaf, "aten.add.Scalar", (Ref(prod), 0.0), {}, (prod,), (leaf,), (), (prod,), (leaf,)))
            leaves.append(leaf)

        def build(node, top):
            if isinstance(node, int):
                return leaves[node]
            a, b = build(node[0], False), build(node[1], False)
            name = s_name if top else fresh(f"{s_name}_t")
            if not top:
                new_values[name] = Value(name, (T, D), "float32", (D, 1), role="temporary", producer=name)
            new_ops.append(Operation(name, "aten.add.Tensor", (Ref(a), Ref(b)), {}, (a, b), (name,), (), (a, b),
                                     (name,)))
            return name
        build(_butterfly_pairs(k), True)
        replaced[op.name] = new_ops
        replaced[mop.name] = []
        new_values[s_name] = replace(sv, producer=s_name)
        rewrites.append(dict(sum=s_name, product=m_name, k=k, rows=T, width=D, **({"scaled_by": scaled} if scaled else {})))
        for name, value in new_values.items():
            g.values[name] = value
        g.values.pop(m_name, None)
    if not rewrites:
        return graph
    producer_of = {}
    ops = []
    for op in graph.operations:
        if op.name in replaced:
            for new in replaced[op.name]:
                deps = tuple(dict.fromkeys(producer_of[r] for r in new.reads if r in producer_of))
                new = replace(new, dependencies=deps)
                ops.append(new)
                producer_of[new.outputs[0]] = new.name
            continue
        ops.append(op)
        for out in op.outputs:
            producer_of[out] = op.name
    values = {k_: v for k_, v in g.values.items()}
    report["residual_addends"] = rewrites
    return replace(graph, operations=ops, values=values)


ATTENTION_TARGETS = ("latch_ops.composed_attention_tape.default", "latch_ops.bf16_sdpa_fwd.default",
                     "latch_ops.bf16_sdpa_bwd.default")
_ROPE_VIEWS = {"view", "_unsafe_view", "reshape", "transpose", "t", "permute", "unsqueeze", "squeeze", "expand",
               "slice", "select", "alias", "detach"}
_ROPE_OPS = {"clone", "contiguous", "cat", "neg", "mul", "add", "_to_copy", "slice_backward", "select_backward"}


def rope_region(graph):
    """Values of the element-wise chains bordering composed attention calls that carry a rotary embedding."""
    g = _Graph(graph)
    attention = [op for op in graph.operations if op.target in ATTENTION_TARGETS]
    if not attention:
        return [], 0
    att_names = {op.name for op in attention}
    att_outputs = {o for op in attention for o in op.outputs}
    region, seen = set(), set()

    def chain_op(op):
        return op is not None and op.effect == "pure" and len(op.outputs) == 1 and (
            _base(op) in _ROPE_VIEWS | _ROPE_OPS or op.target == "aten.sum.dim_IntList")
    # Producers of attention operands (forward preparation and the padded output cotangent).
    stack = [r for op in attention for r in op.reads if r not in att_outputs]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        op = g.producer.get(v)
        if not chain_op(op) or op.target == "aten.sum.dim_IntList":
            continue
        region.add(v)
        stack.extend(op.reads)
    # Readers of attention results (output un-padding and the q/k/v cotangent chains).
    stack, seen = list(att_outputs), set()
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        for op in g.readers.get(v, ()):
            if op.name in att_names or not chain_op(op):
                continue
            region.add(op.outputs[0])
            stack.append(op.outputs[0])
    # Rotary signature: rotate-half cats (forward) and negated slice_backward pairs (backward).
    rotary = 0
    for v in region:
        op = g.producer.get(v)
        if op is None:
            continue
        if _base(op) == "cat" and op.args and isinstance(op.args[0], (list, tuple)) and len(op.args[0]) == 2:
            first = g.producer.get(_ref(op.args[0][0]))
            if first is not None and _base(first) == "neg":
                rotary += 1
        if _base(op) == "slice_backward":
            src = g.producer.get(op.reads[0]) if op.reads else None
            if src is not None and _base(src) == "neg":
                rotary += 1
    if not rotary:
        return [], 0
    return sorted(v for v in region if _base(g.producer[v]) in _ROPE_OPS), rotary


def _cast_before_gather(graph, report):
    """dcast: embedding(x, idx) read only by identical casts -> embedding(cast(x), idx)."""
    from .ir import Ref, Operation, Value
    g = _Graph(graph)
    used = set(graph.values)
    protected = set(graph.inputs)
    from .remat import _names
    _names(graph.outputs, protected)
    _names(getattr(graph, "state_writes", None) or {}, protected)
    replaced, records = {}, []
    for op in graph.operations:
        if op.target != "aten.embedding.default" or len(op.args) < 2 or len(op.outputs) != 1:
            continue
        table, idx = _ref(op.args[0]), _ref(op.args[1])
        e = op.outputs[0]
        if (table is None or idx is None or e in protected or len(op.args) > 2 or op.kwargs
                or g.values[e].dtype != "float32" or not _contiguous(g.values[e])):
            continue
        source = g.through_views(table)[1]
        if source is None or not g.normalized_source(source, gathers=False):
            continue
        casts = g.readers.get(e, ())
        if (not casts or any(c.target != "aten._to_copy.default" or set(c.kwargs) - {"dtype"} or c.reads != (e,)
                             or c.outputs[0] in protected for c in casts)
                or len({repr(c.kwargs.get("dtype")) for c in casts}) != 1):
            continue
        keep = casts[0].outputs[0]
        out = g.values[keep]
        if not _contiguous(out) or out.alias_of is not None:
            continue
        tv = g.values[table]
        name = f"{table}__dc1"
        k = 1
        while name in used:
            k += 1
            name = f"{table}__dc{k}"
        used.add(name)
        strides, acc = [], 1
        for d in reversed(tv.shape):
            strides.insert(0, acc)
            acc *= d
        cast = Operation(name, "aten._to_copy.default", (Ref(table),), dict(casts[0].kwargs), (table,), (name,), (),
                         (table,), (name,))
        gather = Operation(keep, "aten.embedding.default", (Ref(name), Ref(idx)), {}, (name, idx), (keep,), (),
                           (name, idx), (keep,))
        replaced[op.name] = [cast, gather]
        for c in casts:
            replaced[c.name] = []
        replaced["__values__" + op.name] = (name, Value(name, tuple(tv.shape), out.dtype, tuple(strides),
                                                        role="temporary", producer=name), e,
                                            [c.outputs[0] for c in casts[1:]], keep)
        records.append(dict(gather=e, cast=keep, table=table))
    if not records:
        return graph
    values = dict(graph.values)
    alias = {}
    for key, entry in list(replaced.items()):
        if key.startswith("__values__"):
            name, value, e, others, keep = entry
            values[name] = value
            values.pop(e, None)
            values[keep] = replace(values[keep], producer=keep)
            for o in others:
                alias[o] = keep
                values.pop(o, None)
            del replaced[key]
    from .remat import _rewrite
    producer_of, ops = {}, []
    for op in graph.operations:
        if op.name in replaced:
            for new in replaced[op.name]:
                new = replace(new, dependencies=tuple(dict.fromkeys(producer_of[r] for r in new.reads if r in producer_of)))
                ops.append(new)
                producer_of[new.outputs[0]] = new.name
            continue
        if any(r in alias for r in op.reads):
            op = replace(op, args=_rewrite(op.args, alias), kwargs=_rewrite(op.kwargs, alias),
                         inputs=tuple(alias.get(x, x) for x in op.inputs), reads=tuple(alias.get(x, x) for x in op.reads),
                         dependencies=tuple(dict.fromkeys(producer_of.get(alias.get(r, r), d)
                                                          for r, d in zip(op.reads, op.dependencies)))
                         if len(op.reads) == len(op.dependencies) else op.dependencies)
        ops.append(op)
        for o in op.outputs:
            producer_of[o] = op.name
    values = {k_: (replace(v, alias_of=alias[v.alias_of]) if v.alias_of in alias else v) for k_, v in values.items()}
    report["cast_before_gather"] = records
    return replace(graph, operations=ops, values=values)


RPACK_TARGET = "latch_ops.norm_fusion_rotary_pack.default"


def _rotary_packs(graph, report):
    """rpack: replace recognized padded rotary/GQA attention-operand chains by one rotary-pack operation."""
    from .ir import Ref, Operation
    g = _Graph(graph)
    attention = [op for op in graph.operations if op.target in ATTENTION_TARGETS]
    if not attention:
        return graph

    def prod(name, base, target=None):
        op = g.producer.get(name)
        if op is None or op.effect != "pure" or len(op.outputs) != 1:
            return None
        if target is not None:
            return op if op.target == target else None
        return op if _base(op) == base else None

    def last_slice(op, start, stop, src):
        if op is None or op.target != "aten.slice.Tensor" or op.reads != (src,):
            return False
        args = list(op.args) + [None] * 5
        dim, lo, hi, step = args[1], args[2], args[3], args[4] if args[4] is not None else 1
        rank = len(g.values[src].shape)
        return dim % rank == rank - 1 and lo == start and (hi == stop or hi >= 2**62) and step == 1

    def bf16_cast(name):
        cast = prod(name, "_to_copy")
        if cast is None or not _plain_cast(cast) or g.values[name].dtype != "bfloat16":
            return None
        return cast

    def rope(name):
        """name = bf16(add(...)) (see rope_add): (T, C, S, chain ops)."""
        cast = bf16_cast(name)
        if cast is None:
            return None
        r = rope_add(cast.reads[0])
        return None if r is None else (r[0], r[1], r[2], [cast] + r[3])

    def rope_add(name):
        """name = add(mul(T, C), mul(cat([neg(T[..., h:]), T[..., :h]], -1), S)) in FP32: (T, C, S, chain ops)."""
        add = prod(name, None, "aten.add.Tensor")
        if add is None or len(add.args) != 2 or add.kwargs or len(add.reads) != 2:
            return None
        m5, m6 = (prod(_ref(a), None, "aten.mul.Tensor") for a in add.args)
        if m5 is None or m6 is None or len(m5.args) != 2 or len(m6.args) != 2 or m5.kwargs or m6.kwargs:
            return None
        t, c = (_ref(a) for a in m5.args)
        cat, s = (_ref(a) for a in m6.args)
        if None in (t, c, cat, s):
            return None
        tv = g.values[t]
        D = tv.shape[-1]
        h = D // 2
        catop = prod(cat, None, "aten.cat.default")
        if (tv.dtype != "bfloat16" or D % 2 or catop is None or len(catop.args) < 2
                or catop.args[1] % len(tv.shape) != len(tv.shape) - 1 or len(catop.args[0]) != 2):
            return None
        neg_name, lo_name = (_ref(a) for a in catop.args[0])
        negop = prod(neg_name, "neg")
        if (negop is None or not last_slice(g.producer.get(negop.reads[0]), h, D, t)
                or not last_slice(g.producer.get(lo_name), 0, h, t)):
            return None
        for v in (c, s):
            vv = g.values[v]
            if vv.dtype != "float32" or vv.shape[-1] != D or vv.shape[-2] != tv.shape[-2] or any(
                    d != 1 for d in vv.shape[:-2]):
                return None
        if g.values[name].dtype != "float32":
            return None
        chain = [add, m5, m6, catop, negop, g.producer[negop.reads[0]], g.producer[lo_name]]
        return t, c, s, chain

    def plain(name):
        """A view chain of a BF16 rank-4 projection head layout [B, Hs, S, D] (no copies)."""
        v = g.values.get(name)
        op = g.producer.get(name)
        return v is not None and v.dtype == "bfloat16" and len(v.shape) == 4 and op is not None and _base(op) in (
            "transpose", "view", "_unsafe_view", "reshape", "permute")

    replaced, records, removed = {}, [], set()
    for att in attention:
        for operand in att.reads:
            if operand in replaced or g.values.get(operand) is None:
                continue
            ov = g.values[operand]
            clone = prod(operand, None, "aten.clone.default")
            if clone is None or ov.dtype != "bfloat16" or len(ov.shape) != 4 or not _contiguous(ov) or ov.storage_offset:
                continue
            pad = prod(clone.reads[0], None, "aten.cat.default")
            if pad is None or len(pad.args) < 2 or pad.args[1] % 4 != 3 or len(pad.args[0]) != 2:
                continue
            p_name, z_name = (_ref(a) for a in pad.args[0])
            zop = prod(z_name, None, "aten.zeros.default")
            if p_name is None or zop is None or g.values[z_name].dtype != "bfloat16":
                continue
            B, H, S, Dp = ov.shape
            chain = [clone, pad, zop]
            rot, source = None, None

            def strip_gqa(name):
                """view(clone(expand(unsqueeze(R, 2)))) -> (R, ops, group); otherwise (name, [], 1)."""
                view = prod(name, "_unsafe_view") or prod(name, "view")
                if view is None or prod(view.reads[0], None, "aten.clone.default") is None:
                    return name, [], 1
                gclone = g.producer[view.reads[0]]
                expand = prod(gclone.reads[0], "expand")
                unsq = prod(expand.reads[0], "unsqueeze") if expand is not None else None
                if unsq is None or unsq.args[1] % 5 != 2:
                    return None, None, None
                ev = g.values[expand.outputs[0]]
                if len(ev.shape) != 5 or ev.shape[0] != B or ev.shape[1] * ev.shape[2] != H or ev.shape[3] != S:
                    return None, None, None
                return unsq.reads[0], [view, gclone, expand, unsq], ev.shape[2]
            late_cast = bf16_cast(p_name)
            if late_cast is not None:
                # The BF16 rounding after a grouped-query copy (a copy commutes with the rounding).
                inner, gops, group = strip_gqa(late_cast.reads[0])
                r = rope_add(inner) if inner is not None else None
                if r is None:
                    continue
                source, cos, sin, ops = r
                rot = (cos, sin)
                chain += [late_cast] + gops + ops
            else:
                inner, gops, group = strip_gqa(p_name)
                if inner is None:
                    continue
                chain += gops
                r = rope(inner)
                if r is not None:
                    source, cos, sin, ops = r
                    rot = (cos, sin)
                    chain += ops
                elif plain(inner):
                    source = inner
                else:
                    continue
            sv = g.values[source]
            D = sv.shape[-1]
            if (tuple(sv.shape) != (B, H // group, S, D) or not 0 < D <= Dp or Dp % 8
                    or g.values[p_name].shape[-1] != D):
                continue
            # Every replaced intermediate must be read only inside the chain (or by this operand's clone).
            outs = {op.outputs[0] for op in chain}
            if any(r_.name not in {op.name for op in chain}
                   for v in outs - {operand} for r_ in g.readers.get(v, ())):
                continue
            reads = (source,) + (rot if rot else ())
            args = (Ref(source), Ref(rot[0]) if rot else None, Ref(rot[1]) if rot else None, group, D, Dp)
            new = Operation(operand, RPACK_TARGET, args, {}, reads, (operand,), (), reads, (operand,))
            replaced[operand] = new
            removed |= {op.name for op in chain}
            records.append(dict(operand=operand, source=source, rotary=bool(rot), group=group, width=D,
                                padded=Dp, replaced=len(chain)))
    if not records:
        return graph
    producer_of, ops = {}, []
    placed = set()
    for op in graph.operations:
        if op.name in removed:
            out = op.outputs[0]
            if out in replaced and out not in placed:
                new = replaced[out]
                new = replace(new, dependencies=tuple(dict.fromkeys(producer_of[r] for r in new.reads if r in producer_of)))
                ops.append(new)
                producer_of[out] = new.name
                placed.add(out)
            continue
        ops.append(op)
        for o in op.outputs:
            producer_of[o] = op.name
    keep = {o for op in ops for o in op.outputs} | set(graph.inputs)
    values = {k: v for k, v in graph.values.items() if k in keep or v.alias_of in keep and k in
              {r for op in ops for r in op.reads}}
    values = {k: (replace(v, producer=k) if k in replaced else v) for k, v in values.items()}
    report["rotary_packs"] = records
    return replace(graph, operations=ops, values=values)


RPACKB_TARGET = "latch_ops.norm_fusion_rotary_unpack.default"


def _sum_leaves(k):
    """exact_reduction's per-thread tree over k <= 32 leaves (lane-0 xor butterfly, leaves 0.0f + x)."""
    from .codegen_rules import _butterfly0
    return _butterfly0([f"(0.0f + l{j})" for j in range(k)])


def _rotary_unpacks(graph, report):
    """rpackb: recognized attention-gradient chains -> one rotary-unpack operation each."""
    from .ir import Ref, Operation
    g = _Graph(graph)
    attention = [op for op in graph.operations if op.target in ATTENTION_TARGETS]
    att_out = {o for op in attention for o in op.outputs}
    if not attention:
        return graph

    def prod(name, base=None, target=None):
        op = g.producer.get(name)
        if op is None or op.effect != "pure" or len(op.outputs) != 1:
            return None
        if target is not None:
            return op if op.target == target else None
        return op if _base(op) == base else None

    def head_slice(name):
        """slice(GRAD, -1, 0, D) of a rank-4 attention gradient: (GRAD, D, op)."""
        op = prod(name, None, "aten.slice.Tensor")
        if op is None or op.reads[0] not in att_out:
            return None
        args = list(op.args) + [None] * 5
        rank = len(g.values[op.reads[0]].shape)
        if rank != 4 or args[1] % rank != 3 or args[2] != 0 or (args[4] or 1) != 1:
            return None
        return op.reads[0], g.values[name].shape[-1], op

    def widen(name):
        """Optional FP32 widening of a value: (inner, ops)."""
        op = prod(name, "_to_copy")
        if op is not None and _plain_cast(op) and g.values[name].dtype == "float32" \
                and g.values[op.reads[0]].dtype == "bfloat16":
            return op.reads[0], [op]
        return name, []

    def head_source(name):
        """F: (grad, D, group, sum_bf16, ops) for slice / widened slice / grouped head sum of either."""
        inner, ops = widen(name)
        hs = head_slice(inner)
        if hs is not None:
            return hs[0], hs[1], 1, None, ops + [hs[2]]
        sq = prod(inner, "squeeze")
        sm = prod(sq.reads[0], None, "aten.sum.dim_IntList") if sq is not None else None
        if sm is None or list(sm.args[1]) != [2] or not (len(sm.args) > 2 and sm.args[2]) or sm.kwargs:
            return None
        vw = prod(sm.reads[0], "view")
        if vw is None:
            return None
        leaf, lops = widen(vw.reads[0])
        hs = head_slice(leaf)
        if hs is None or len(g.values[vw.outputs[0]].shape) != 5:
            return None
        group = g.values[vw.outputs[0]].shape[2]
        if group > 32:
            return None
        return hs[0], hs[1], group, g.values[sm.outputs[0]].dtype == "bfloat16", ops + [sq, sm, vw] + lops + [hs[2]]

    def mul_by(name, target_dtype="bfloat16"):
        """bf16(mul(F, T)): (F, T, ops)."""
        cast = prod(name, "_to_copy")
        if cast is None or not _plain_cast(cast) or g.values[name].dtype != target_dtype:
            return None
        m = prod(cast.reads[0], None, "aten.mul.Tensor")
        if m is None or len(m.args) != 2 or m.kwargs or g.values[m.outputs[0]].dtype != "float32":
            return None
        f, t = (_ref(a) for a in m.args)
        if f is None or t is None or g.values[t].dtype != "float32":
            return None
        return f, t, [cast, m]

    def add_of(name):
        op = prod(name, None, "aten.add.Tensor")
        if op is None or len(op.args) != 2 or op.kwargs or g.values[name].dtype != "bfloat16":
            return None
        a, b = (_ref(x) for x in op.args)
        return (a, b, op) if a is not None and b is not None else None

    def placed(name, lo, hi, D):
        """slice_backward(x, sizes, -1, lo, hi_or_end, 1): (x, op)."""
        op = prod(name, "slice_backward")
        if op is None or len(op.args) < 5:
            return None
        args = list(op.args) + [1]
        if args[2] % 4 != 3 or args[3] != lo or not (args[4] == hi or args[4] >= 2**62) or args[5] != 1:
            return None
        return op.reads[0], op

    replaced, records, removed = {}, [], set()
    for op in graph.operations:
        if op.target != "aten.clone.default" or len(op.outputs) != 1:
            continue
        gname = op.outputs[0]
        gv = g.values[gname]
        if gv.dtype != "bfloat16" or len(gv.shape) != 4 or not _contiguous(gv) or gv.storage_offset:
            continue
        tr = prod(op.reads[0], "transpose")
        chain = [op]
        if tr is not None and sorted(a % 4 for a in tr.args[1:3]) == [1, 2]:
            # Gradient copies for the projection products: [B, S, Hx, D] = transpose of [B, Hx, S, D].
            chain.append(tr)
            a_name = tr.reads[0]
            B, S, Hx, D = gv.shape
            spec = None
            top = add_of(a_name)
            if top is not None:
                a1, c_name, top_op = top
                cmul = mul_by(c_name)
                inner = add_of(a1)
                if cmul is None or inner is None:
                    continue
                sb1, sb2, inner_op = inner
                p1, p2 = placed(sb1, 0, D // 2, D), placed(sb2, D // 2, D, D)
                if p1 is None or p2 is None:
                    continue
                neg = prod(p2[0], "neg")
                hi = prod(p1[0], None, "aten.slice.Tensor")
                lo = prod(neg.reads[0], None, "aten.slice.Tensor") if neg is not None else None
                if hi is None or lo is None or hi.reads[0] != lo.reads[0]:
                    continue
                smul = mul_by(hi.reads[0])
                if smul is None or smul[0] != cmul[0]:
                    continue
                src = head_source(cmul[0])
                if src is None or src[1] != D:
                    continue
                grad, _, group, sum_bf16, sops = src
                spec = dict(mode="rotary", grad=grad, sin=smul[1], cos=cmul[1], group=group, sum_bf16=sum_bf16)
                chain += [top_op, inner_op, p1[1], p2[1], neg, hi, lo] + cmul[2] + smul[2] + sops
            else:
                src = head_source(a_name)
                if src is None or src[2] == 1 or src[1] != D or not src[3]:
                    continue
                grad, _, group, sum_bf16, sops = src
                spec = dict(mode="sum", grad=grad, sin=None, cos=None, group=group, sum_bf16=sum_bf16)
                chain += sops
            gshape = g.values[spec["grad"]].shape
            if gshape[0] != B or gshape[2] != S or gshape[1] != Hx * spec["group"]:
                continue
        else:
            sbk_op = prod(op.reads[0], "slice_backward")
            if sbk_op is None:
                continue
            x = sbk_op.reads[0]
            xv = g.values[x]
            sbk = placed(op.reads[0], 0, xv.shape[-1], None) if len(xv.shape) == 4 else None
            if sbk is None:
                continue
            B, H, S, Dp = gv.shape
            if xv.dtype != "bfloat16" or len(xv.shape) != 4 or tuple(xv.shape[:3]) != (B, H, S):
                continue
            # Every consumer of the padded copy must be an attention call (the output cotangent operand).
            if not g.readers.get(gname) or any(r.target not in ATTENTION_TARGETS for r in g.readers[gname]):
                continue
            spec = dict(mode="pad", grad=x, sin=None, cos=None, group=1, sum_bf16=None)
            chain.append(sbk[1])
            D = xv.shape[3]
        outs = {c.outputs[0] for c in chain}
        names_ = {c.name for c in chain}
        if any(r.name not in names_ for v in outs - {gname} for r in g.readers.get(v, ())):
            continue
        reads = tuple(x for x in (spec["grad"], spec["sin"], spec["cos"]) if x is not None)
        args = (Ref(spec["grad"]), Ref(spec["sin"]) if spec["sin"] else None, Ref(spec["cos"]) if spec["cos"] else None,
                spec["mode"], spec["group"], bool(spec["sum_bf16"]), D)
        replaced[gname] = Operation(gname, RPACKB_TARGET, args, {}, reads, (gname,), (), reads, (gname,))
        removed |= names_
        records.append(dict(value=gname, mode=spec["mode"], grad=spec["grad"], group=spec["group"], width=D,
                            replaced=len(chain)))
    if not records:
        return graph
    producer_of, ops, placed_ = {}, [], set()
    for op in graph.operations:
        if op.name in removed:
            out = op.outputs[0]
            if out in replaced and out not in placed_:
                new = replaced[out]
                ops.append(replace(new, dependencies=tuple(dict.fromkeys(
                    producer_of[r] for r in new.reads if r in producer_of))))
                producer_of[out] = new.name
                placed_.add(out)
            continue
        ops.append(op)
        for o in op.outputs:
            producer_of[o] = op.name
    live = {o for op in ops for o in op.outputs} | set(graph.inputs)
    read = {r for op in ops for r in op.reads}
    values = {k: v for k, v in graph.values.items() if k in live or (k in read)}
    values = {k: (replace(v, producer=k) if k in replaced else v) for k, v in values.items()}
    report["rotary_unpacks"] = records
    return replace(graph, operations=ops, values=values)


def emit_rotary_unpack(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """Generated row pass for an rpackb operation (see module rule ``rpackb``)."""
    from .lower import LoweringError
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ()))
    if len(args) != 7 or len(outputs) != 1:
        raise LoweringError("rotary_unpack has seven arguments and one result")
    grad = ref(args[0], names)
    sin = ref(args[1], names) if args[1] is not None else None
    cos = ref(args[2], names) if args[2] is not None else None
    mode, group, sum_bf16, D = args[3], int(args[4]), bool(args[5]), int(args[6])
    name = outputs[0]
    shape = tuple(field(values[name], "shape"))
    lg = layouts[grad]
    if dtypes[name] != "bfloat16" or dtypes[grad] != "bfloat16" or len(shape) != 4 or math_prod(shape) >= 2**31:
        raise LoweringError("rotary_unpack shape/dtype contract")
    define(name, shape, None, "bfloat16")

    def ptr(key):
        layout = layouts[key]
        return f"({ident(layout.root)} + {layout.offset}LL)"
    gs = lg.strides
    if mode == "pad":
        B, H, S, Dp = shape
        if Dp % 8:
            raise LoweringError("rotary_unpack pad width")
        chunks = B * H * S * Dp // 8
        code = f"""for (unsigned c_ = blockIdx.x * blockDim.x + threadIdx.x; c_ < {chunks}u; c_ += gridDim.x * blockDim.x) {{
  const unsigned row_ = c_ / {Dp // 8}u, d0_ = (c_ % {Dp // 8}u) * 8u;
  const unsigned s = row_ % {S}u, h_ = (row_ / {S}u) % {H}u, b_ = row_ / {S * H}u;
  const __nv_bfloat16* src_ = {ptr(grad)} + (long long)b_ * {gs[0]}LL + (long long)h_ * {gs[1]}LL + (long long)s * {gs[2]}LL;
  __nv_bfloat16 o_[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {{
    const unsigned d = d0_ + (unsigned)k;
    o_[k] = d < {D}u ? src_[(long long)d * {gs[3]}LL] : __float2bfloat16_rn(0.0f);
  }}
  __nv_bfloat16* dst_ = {ptr(name)} + (long long)c_ * 8LL;
  if ((reinterpret_cast<uintptr_t>(dst_) & 15u) == 0) {{ *reinterpret_cast<uint4*>(dst_) = *reinterpret_cast<const uint4*>(o_); }}
  else {{
#pragma unroll
    for (int k = 0; k < 8; ++k) dst_[k] = o_[k];
  }}
}}"""
        reads = [grad]
    else:
        B, S, Hx, D_ = shape
        if D_ != D or D % 8 or D % 2:
            raise LoweringError("rotary_unpack head width")
        half = D // 2
        leaves = "\n".join(f"      const float l{j} = __bfloat162float(g_[(long long){j} * {gs[1]}LL + (long long)(dd) * {gs[3]}LL]);"
                           for j in range(group))
        if group == 1:
            fexpr = f"__bfloat162float(g_[(long long)(dd) * {gs[3]}LL])"
            fdef = ""
        else:
            tree = _sum_leaves(group)
            fdef = leaves.replace("(dd)", "(dd)")
            fexpr = f"__bfloat162float(__float2bfloat16_rn({tree}))" if sum_bf16 else tree
        if mode == "sum":
            body = f"""      const unsigned dd = d;
{fdef}
      o_[k] = __float2bfloat16_rn({_sum_leaves(group)});"""
            if not sum_bf16:
                raise LoweringError("rotary_unpack sum mode requires a BF16 head sum")
            reads = [grad]
        else:
            ls_, lc_ = layouts[sin], layouts[cos]
            sst = (0,) * (4 - len(ls_.shape)) + tuple(ls_.strides)
            cst = (0,) * (4 - len(lc_.shape)) + tuple(lc_.strides)

            def F(var):
                return f"""      float {var};
      {{ const unsigned dd = d_{var};
{fdef}
        {var} = {fexpr}; }}"""
            body = f"""      const unsigned d_fd = d, d_fr = d < {half}u ? d + {half}u : d - {half}u;
{F('fd')}
{F('fr')}
      const __nv_bfloat16 sp_ = __float2bfloat16_rn((fr * {ptr(sin)}[(long long)s * {sst[2]}LL + (long long)d_fr * {sst[3]}LL]));
      const __nv_bfloat16 a_ = d < {half}u
          ? __float2bfloat16_rn((__bfloat162float(sp_) + (__bfloat162float(__float2bfloat16_rn(0.0f)) * 1.0f)))
          : __float2bfloat16_rn((__bfloat162float(__float2bfloat16_rn(0.0f)) + (__bfloat162float(__float2bfloat16_rn((-__bfloat162float(sp_)))) * 1.0f)));
      const __nv_bfloat16 cp_ = __float2bfloat16_rn((fd * {ptr(cos)}[(long long)s * {cst[2]}LL + (long long)d * {cst[3]}LL]));
      o_[k] = __float2bfloat16_rn((__bfloat162float(a_) + (__bfloat162float(cp_) * 1.0f)));"""
            reads = [grad, sin, cos]
        chunks = B * S * Hx * D // 8
        code = f"""for (unsigned c_ = blockIdx.x * blockDim.x + threadIdx.x; c_ < {chunks}u; c_ += gridDim.x * blockDim.x) {{
  const unsigned row_ = c_ / {D // 8}u, d0_ = (c_ % {D // 8}u) * 8u;
  const unsigned h_ = row_ % {Hx}u, s = (row_ / {Hx}u) % {S}u, b_ = row_ / {Hx * S}u;
  const __nv_bfloat16* g_ = {ptr(grad)} + (long long)b_ * {gs[0]}LL + (long long)(h_ * {group}u) * {gs[1]}LL + (long long)s * {gs[2]}LL;
  __nv_bfloat16 o_[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {{
    const unsigned d = d0_ + (unsigned)k;
    {{
{body}
    }}
  }}
  __nv_bfloat16* dst_ = {ptr(name)} + (long long)c_ * 8LL;
  if ((reinterpret_cast<uintptr_t>(dst_) & 15u) == 0) {{ *reinterpret_cast<uint4*>(dst_) = *reinterpret_cast<const uint4*>(o_); }}
  else {{
#pragma unroll
    for (int k = 0; k < 8; ++k) dst_[k] = o_[k];
  }}
}}"""
    return [dict(kind="rotary_unpack", names=[name], targets=[RPACKB_TARGET], reads=reads, writes=[name], code=code,
                 threads=256, work_items=(math_prod(shape) // 8 + 255) // 256, shared_bytes=0,
                 engine="norm_fusion_rotary_unpack",
                 arithmetic="per element the captured rotary-backward/head-sum/padding expressions and roundings")]


def math_prod(shape):
    out = 1
    for d in shape:
        out *= d
    return out


def emit_rotary_pack(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """Generated row pass for an rpack operation (see module rule ``rpack``)."""
    from .lower import LoweringError
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ()))
    if len(args) != 6 or len(outputs) != 1:
        raise LoweringError("rotary_pack has six arguments and one result")
    src = ref(args[0], names)
    cos = ref(args[1], names) if args[1] is not None else None
    sin = ref(args[2], names) if args[2] is not None else None
    group, D, Dp = int(args[3]), int(args[4]), int(args[5])
    name = outputs[0]
    shape = tuple(field(values[name], "shape"))
    B, H, S, Dp_ = shape
    ls = layouts[src]
    if (Dp_ != Dp or tuple(ls.shape) != (B, H // group, S, D) or dtypes[src] != "bfloat16"
            or dtypes[name] != "bfloat16" or Dp % 8 or B * H * S * Dp >= 2**31):
        raise LoweringError("rotary_pack shape/dtype contract")
    define(name, shape, None, "bfloat16")
    half = D // 2

    def ptr(key):
        layout = layouts[key]
        return f"({ident(layout.root)} + {layout.offset}LL)"
    st = ls.strides
    if cos is not None:
        lc, lsn = layouts[cos], layouts[sin]
        cst = (0,) * (4 - len(lc.shape)) + tuple(lc.strides)
        sst = (0,) * (4 - len(lsn.shape)) + tuple(lsn.strides)
        cshape = (1,) * (4 - len(lc.shape)) + tuple(lc.shape)
        if any(d != 1 for d in cshape[:2]) or dtypes[cos] != "float32" or dtypes[sin] != "float32":
            raise LoweringError("rotary_pack tables must broadcast over batch and heads")
        body = f"""      const float x_ = __bfloat162float(src_[(long long)d * {st[3]}LL]);
      const __nv_bfloat16 r_ = d < {half}u ? __float2bfloat16_rn((-__bfloat162float(src_[(long long)(d + {half}u) * {st[3]}LL])))
                                           : src_[(long long)(d - {half}u) * {st[3]}LL];
      const float m5_ = (x_ * {ptr(cos)}[(long long)s * {cst[2]}LL + (long long)d * {cst[3]}LL]);
      const float m6_ = (__bfloat162float(r_) * {ptr(sin)}[(long long)s * {sst[2]}LL + (long long)d * {sst[3]}LL]);
      o_[k] = __float2bfloat16_rn((m5_ + (m6_ * 1.0f)));"""
    else:
        body = f"""      o_[k] = src_[(long long)d * {st[3]}LL];"""
    chunks = B * H * S * Dp // 8
    code = f"""for (unsigned c_ = blockIdx.x * blockDim.x + threadIdx.x; c_ < {chunks}u; c_ += gridDim.x * blockDim.x) {{
  const unsigned row_ = c_ / {Dp // 8}u, d0_ = (c_ % {Dp // 8}u) * 8u;
  const unsigned s = row_ % {S}u, h_ = (row_ / {S}u) % {H}u, b_ = row_ / {S * H}u;
  const __nv_bfloat16* src_ = {ptr(src)} + (long long)b_ * {st[0]}LL + (long long)(h_ / {group}u) * {st[1]}LL + (long long)s * {st[2]}LL;
  __nv_bfloat16 o_[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {{
    const unsigned d = d0_ + (unsigned)k;
    if (d < {D}u) {{
{body}
    }} else {{
      o_[k] = __float2bfloat16_rn(0.0f);
    }}
  }}
  __nv_bfloat16* dst_ = {ptr(name)} + (long long)c_ * 8LL;
  if ((reinterpret_cast<uintptr_t>(dst_) & 15u) == 0) {{
    *reinterpret_cast<uint4*>(dst_) = *reinterpret_cast<const uint4*>(o_);
  }} else {{
#pragma unroll
    for (int k = 0; k < 8; ++k) dst_[k] = o_[k];
  }}
}}"""
    reads = [src] + ([cos, sin] if cos is not None else [])
    return [dict(kind="rotary_pack", names=[name], targets=[RPACK_TARGET], reads=reads, writes=[name], code=code,
                 threads=256, work_items=(chunks + 255) // 256, shared_bytes=0, engine="norm_fusion_rotary_pack",
                 arithmetic="per element the captured rotary/expansion/padding expressions and roundings")]


def apply_ir(graph):
    """(graph, report) after the selected IR rules; unchanged graph when nothing is recognized."""
    from .ir import TrainingIR
    chosen = rules_of(_RULES.get())
    report = {"version": VERSION, "rules": list(chosen)}
    if not chosen or not isinstance(graph, TrainingIR) or not isinstance(graph.outputs, dict):
        report["reason"] = "no rules selected or not a structured flat training graph"
        return graph, report
    if "dcast" in chosen:
        graph = _cast_before_gather(graph, report)
    if "rpack" in chosen:
        graph = _rotary_packs(graph, report)
    if "rpackb" in chosen:
        graph = _rotary_unpacks(graph, report)
    if "pcse" in chosen:
        graph = _cse_parameter_derivations(graph, report)
    if "castcse" in chosen:
        graph = _cse_casts(graph, report)
    if "resid" in chosen:
        graph = _expand_residual_addends(graph, report)
    remat_kinds = {k for k in ("nremat", "gremat") if k in chosen}
    if remat_kinds:
        from . import remat
        g = _Graph(graph)

        def admit(_graph, copied, late):
            if len(copied) != 1:
                return False
            op = copied[0]
            return (("nremat" in remat_kinds and g.normalized(op))
                    or ("gremat" in remat_kinds and g.gated_silu(op)))
        graph, remat_report = remat.plan(graph, admit=admit)
        report["remat"] = {"copied": sorted(c["copy"] for c in remat_report["candidates"].values()),
                           "sources": sorted(remat_report["candidates"]),
                           "bytes_not_stored_upper_bound": remat_report.get("saved_bytes_upper_bound", 0)}
    if "colreg" in chosen or "colgrp" in chosen:
        report["column_sums"] = weight_gradient_sums(graph)
    if "rope" in chosen:
        names, rotary = rope_region(graph)
        report["rope"] = {"values": names, "rotary_signatures": rotary}
    return graph, report
