"""Layout canonicalization of equivalent product expressions (LATCH release step 05).

Equivalent PyTorch spellings of one computation reach the lowering in different operation forms, and the
storage model then loses fast paths: transposed products are read element-wise through strided views, casts of
transposed views materialize transposing copies, multi-step view chains hide in-phase reads (forcing stores),
and batch-broadcast products fall back to batched bodies. These FX rewrites run at capture, after AD and the
precision policy, and give the lowering one canonical form. Each is exact in value or (FP32 batch sums only) a
re-association inside FP32 accumulation:

* ``collapse_view_chains``: a chain of >= 2 views whose net storage-model layout is its source's (identity)
  becomes the source; a chain of reshape-family views with a contiguous net becomes one reshape.
* ``commute_cast_permutations``: ``cast(p(x))`` -> ``p(cast(x))`` for permutation views (P9-C rule 1).
* ``orient_transposed_products``: ``T(A @ B)`` -> ``T(B) @ T(A)`` for an unbatched product when every reader
  (or every reader of its one float cast) transposes it and some reader is not itself a matrix product.
* ``orient_transposed_segment_products``: the same for a segmented outer product,
  ``T(ragged_outer(x, dy))`` -> ``ragged_outer(dy, x)`` (LATCH-TINYMOE-EGRAD: expert weight gradients).
* ``widen_segment_products``: a BF16 ``ragged_outer`` whose one reader is an FP32 cast stores the exactly widened
  BF16-rounded result itself (``ragged_outer_widened``); the cast's values, not its copy, remain.
* ``fuse_cat_casts``: ``bf16(views(cat(xs, 0)))`` of FP32 pieces -> ``views(cat_cast(xs, 0, bf16))``, one pass
  instead of an FP32 concatenation copy and its cast (ops/cast_concat.py).
* ``flatten_broadcast_products``: FP32 ``bmm(a, w.expand(b, k, n))`` -> ``reshape(mm(reshape(a), w), ...)``.
* ``fold_fp32_batch_sums``: FP32 ``sum(bmm(A, B), [0])`` -> one flat product over batch*rows. BF16 sums are
  kept because each batch result is rounded before the sum.

The storage model is that of ``layout_normalize`` (placeholders keep captured strides, non-view results are
contiguous). No rewrite adds a copy. Graphs without a pattern are left byte-identical.
"""
from __future__ import annotations

import torch

from .layout_normalize import RESHAPES, VIEWS, _base, _shape, _view_strides
from .lower import _contiguous

aten = torch.ops.aten


_UNMODELLED = object()


def _node_layout(node, layout):
    """model_layouts' rule for one node given its sources' entries (_UNMODELLED: no entry is made)."""
    shape = _shape(node)
    if node.op in ("placeholder", "get_attr"):
        value = node.meta.get("val")
        return (shape, tuple(int(s) for s in value.stride())) if isinstance(value, torch.Tensor) else _UNMODELLED
    if node.op != "call_function" or shape is None:
        return _UNMODELLED
    base = _base(node)
    source = node.args[0] if node.args and isinstance(node.args[0], torch.fx.Node) else None
    if base in VIEWS:
        if source not in layout or layout[source] is None:
            return None
        src_shape, src_strides = layout[source]
        strides = _view_strides(base, node, src_shape, src_strides, shape)
        return None if strides is None else (shape, tuple(strides))
    return (shape, _contiguous(shape))


def model_layouts(gm):
    """Replay the lowering storage model: node -> (shape, strides), or None where a view is not modelled."""
    layout = {}
    for node in gm.graph.nodes:
        entry = _node_layout(node, layout)
        if entry is not _UNMODELLED:
            layout[node] = entry
    return layout


def _fake(node):
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _new(gm, target, args, value, like):
    node = gm.graph.call_function(target, args)
    node.meta = {k: v for k, v in like.meta.items() if k in ("stack_trace", "nn_module_stack")}
    node.meta["val"] = value
    return node


def flatten_broadcast_products(gm):
    """Rewrite batch-broadcast ``bmm`` into ``view(mm(view(a), select(b)))``. Returns rewrite records."""
    layout = model_layouts(gm)
    records = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or _base(node) != "bmm" or len(node.args) != 2 or node.kwargs:
            continue
        a, b = node.args
        if not (isinstance(a, torch.fx.Node) and isinstance(b, torch.fx.Node)):
            continue
        la, lb, out = layout.get(a), layout.get(b), _fake(node)
        fa, fb = _fake(a), _fake(b)
        if la is None or lb is None or out is None or fa is None or fb is None:
            continue
        (batch, m, k), sa = la
        (batch_b, kb, n), sb = lb
        if batch != batch_b or k != kb or batch < 2 or sb[0] != 0:
            continue
        if out.dtype != torch.float32:
            # BF16: the broadcast operand's gradient rounds each batch's product before its batch sum, so the
            # program cannot become the flat program; flattening only the forward measured no gain (and a
            # 1.06x loss at one reserved shape), so the batched form is kept unchanged.
            continue
        if sa[0] != m * sa[1] and m != 1:
            continue                 # batch and rows do not merge into one stride: keep the batched product
        with gm.graph.inserting_before(node):
            # reshape (not view): eager execution of the captured graph may hold other strides than the model.
            rows = _new(gm, aten.reshape.default, (a, [batch * m, k]), fa.reshape(batch * m, k), a)
            matrix = _new(gm, aten.select.int, (b, 0, 0), fb.select(0, 0), b)
            flat = _new(gm, aten.mm.default, (rows, matrix), torch.mm(rows.meta["val"], matrix.meta["val"]), node)
            result = _new(gm, aten.reshape.default, (flat, [batch, m, n]), flat.meta["val"].reshape(batch, m, n), node)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        records.append({"product": node.name, "flat": flat.name, "batch": batch, "m": m, "n": n, "k": k,
                        "left_strides": list(sa), "right_strides": list(sb),
                        "rule": "batch-broadcast right operand: one flat product over batch*rows"})
        layout = model_layouts(gm)
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


_TRANSPOSES = {"t", "transpose", "permute"}
_RESHAPE_VIEWS = {"view", "_unsafe_view", "reshape", "squeeze", "unsqueeze"}
_OFFSET_VIEWS = {"slice", "select", "as_strided"}


def _swaps_last_two(node, rank):
    """True when ``node`` (a view of a rank-``rank`` value) exchanges exactly its last two dimensions."""
    base = _base(node)
    if base == "t":
        return rank == 2
    if base == "transpose":
        dims = sorted(int(d) % rank for d in node.args[1:3])
        return dims == [rank - 2, rank - 1]
    if base == "permute":
        order = [int(d) % rank for d in node.args[1]]
        return order == [*range(rank - 2), rank - 1, rank - 2]
    return False


def _float_cast(node):
    """A same-device, strided, preserve-format floating-point ``_to_copy`` (``.to(dtype)``/``.float()``)."""
    if node.op != "call_function" or _base(node) != "_to_copy" or not node.args:
        return False
    if set(node.kwargs) - {"dtype", "memory_format", "layout", "device"}:
        return False
    if node.kwargs.get("memory_format", torch.preserve_format) is not torch.preserve_format:
        return False
    if node.kwargs.get("layout", torch.strided) is not torch.strided:
        return False
    source = node.args[0]
    out, src = _fake(node), _fake(source) if isinstance(source, torch.fx.Node) else None
    if out is None or src is None or out.device != src.device:
        return False
    return out.is_floating_point() and src.is_floating_point()


def commute_cast_permutations(gm):
    """``cast(p(x))`` -> ``p(cast(x))`` for a permutation view ``p`` (ops.precision.share_float_casts rule 1).

    A float cast is elementwise, so both orders give identical values; the cast then reads and writes its source
    order and the permutation stays a free operand view. Applied only where the traced result signature (shape,
    strides, dtype) is unchanged, exactly as the P9-C loop rule. Duplicate casts are not merged here.
    """
    from .precision import _signature
    records = []
    for node in list(gm.graph.nodes):
        if not _float_cast(node) or any(user.op == "output" for user in node.users):
            continue
        view = node.args[0]
        if view.op != "call_function" or _base(view) not in _TRANSPOSES:
            continue
        source, value = view.args[0], _fake(view.args[0]) if isinstance(view.args[0], torch.fx.Node) else None
        if value is None or not value.is_floating_point():
            continue
        try:
            cast_value = node.target(value, *node.args[1:], **node.kwargs)
            view_value = view.target(cast_value, *view.args[1:], **view.kwargs)
        except Exception:
            continue
        if _signature(view_value) != _signature(node.meta["val"]):
            continue
        with gm.graph.inserting_before(node):
            cast = gm.graph.call_function(node.target, (source, *node.args[1:]), dict(node.kwargs))
            cast.meta = {"val": cast_value}
            moved = gm.graph.call_function(view.target, (cast, *view.args[1:]), dict(view.kwargs))
            moved.meta = {"val": view_value}
        node.replace_all_uses_with(moved)
        gm.graph.erase_node(node)
        records.append({"cast": node.name, "permutation": view.name, "source": source.name})
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


def orient_transposed_products(gm):
    """Compute a product in the orientation its consumers read: ``T(A @ B)`` -> ``T(B) @ T(A)``.

    Applies when every reader of an ``mm``/``bmm`` result -- or of its one plain float cast -- is a view that
    exchanges the last two dimensions. Element (i, j) of ``T(B) @ T(A)`` is the dot product of the same operand
    elements over the same k as element (j, i) of ``A @ B``, with the same operand/result dtypes and cast.
    Elementwise consumers then read a contiguous result instead of a transposed one; operand transposes remain
    free views of the matrix bodies.
    """
    records = []
    for node in list(gm.graph.nodes):
        base = _base(node) if node.op == "call_function" else None
        # Unbatched products only: batched products (e.g. attention scores/values and their gradients) keep the
        # orientation their established pattern bodies match.
        if base != "mm" or len(node.args) != 2 or node.kwargs:
            continue
        rank = 2
        users = list(node.users)
        if not users or any(u.op == "output" for u in users):
            continue
        casts = [u for u in users if u.op == "call_function" and _float_cast(u)]
        direct = [u for u in users if u not in casts]
        if len(casts) > 1 or any(not (u.op == "call_function" and _swaps_last_two(u, rank)) for u in direct):
            continue
        cast_users = list(casts[0].users) if casts else []
        if casts and (not cast_users or any(u.op != "call_function" or not _swaps_last_two(u, rank)
                                             for u in cast_users)):
            continue
        readers, pending = [], [r for view in (*direct, *cast_users) for r in view.users]
        while pending:                      # look through further views to the consuming operations
            reader = pending.pop()
            if reader.op == "call_function" and _base(reader) in VIEWS:
                pending.extend(reader.users)
            else:
                readers.append(reader)
        if all(r.op == "call_function" and _base(r) in ("mm", "bmm") for r in readers):
            continue          # transposed operands of matrix bodies are already free views
        a, b = node.args
        with gm.graph.inserting_before(node):
            tb = _new(gm, aten.t.default, (b,), _fake(b).t(), b)
            ta = _new(gm, aten.t.default, (a,), _fake(a).t(), a)
            product = _new(gm, node.target, (tb, ta), node.target(tb.meta["val"], ta.meta["val"]), node)
        for user in direct:
            user.replace_all_uses_with(product)
            gm.graph.erase_node(user)
        if casts:
            cast = casts[0]
            with gm.graph.inserting_after(product):
                widened = gm.graph.call_function(cast.target, (product, *cast.args[1:]), dict(cast.kwargs))
                widened.meta = {"val": cast.target(product.meta["val"], *cast.args[1:], **cast.kwargs)}
            for user in cast_users:
                user.replace_all_uses_with(widened)
                gm.graph.erase_node(user)
            gm.graph.erase_node(cast)
        gm.graph.erase_node(node)
        records.append({"product": node.name, "oriented": product.name, "op": base, "through_cast": bool(casts),
                        "transposed_readers": len(direct) + len(cast_users)})
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


_RAGGED_OUTER = "latch_ops.ragged_outer.default"
# widen_segment_products is OFF by default: exact, but measured slower on TinyMoE B2xS1024 (LATCH-TINYMOE-EGRAD
# screens q1b/q2b/q3b, +0.11..+0.13 ms per update): the FP32 store doubles the products' epilogue bytes on the
# backward critical path, while the separate cast phases ran beside other work in their interval.
WIDEN_SEGMENT_PRODUCTS = False


def orient_transposed_segment_products(gm):
    """``T(ragged_outer(x, dy, s))`` -> ``ragged_outer(dy, x, s)`` when every reader swaps the last two dims.

    A segmented outer product's group ``e``, element ``(i, j)``, is ``sum_r x[r, i] * dy[r, j]`` over the rows of
    segment ``e``; element ``(j, i)`` of the swapped product is ``sum_r dy[r, j] * x[r, i]`` over the same rows,
    with the same operand/result dtypes, ascending contraction order and rounding. Typical case: the weight
    gradient of a stacked, transposed expert matrix, whose readers (one float cast, then the transpose back to the
    parameter layout and per-group selects) then read the result contiguously instead of through a stride.
    Same admission as ``orient_transposed_products``; the cast keeps its dtype and moves onto the new product.
    """
    records = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != _RAGGED_OUTER or len(node.args) != 3 or node.kwargs:
            continue
        found = _transposed_readers(node, 3)
        if found is None:
            continue
        direct, cast = found
        x, dy, offsets = node.args
        if not all(isinstance(a, torch.fx.Node) and _fake(a) is not None for a in (x, dy, offsets)):
            continue
        with gm.graph.inserting_before(node):
            product = _new(gm, node.target, (dy, x, offsets),
                           node.target(_fake(dy), _fake(x), _fake(offsets)), node)
        moved = _move_transposed_readers(gm, node, product, direct, cast)
        records.append({"product": node.name, "oriented": product.name, "op": "ragged_outer",
                        "through_cast": cast is not None, "transposed_readers": moved})
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


def widen_segment_products(gm):
    """``float(ragged_outer(x, dy, s))`` -> ``ragged_outer_widened(x, dy, s)`` for BF16 operands.

    Applies when the BF16 product's only reader is one plain FP32 cast. The widened product rounds each FP32
    accumulator once to BF16 exactly as before and stores that value exactly widened -- the bits the cast wrote --
    so the BF16 intermediate, its store, the cast's reread and the separate cast phase disappear.
    """
    from . import ragged_experts  # noqa: F401  (registers latch_ops::ragged_outer_widened)
    target = torch.ops.latch_ops.ragged_outer_widened.default
    records = []
    if not WIDEN_SEGMENT_PRODUCTS:
        return records
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or str(node.target) != _RAGGED_OUTER or len(node.args) != 3 or node.kwargs:
            continue
        value, users = _fake(node), list(node.users)
        if value is None or value.dtype != torch.bfloat16 or len(users) != 1:
            continue
        cast = users[0]
        if cast.op != "call_function" or not _float_cast(cast) or _fake(cast).dtype != torch.float32:
            continue
        with gm.graph.inserting_before(cast):
            widened = _new(gm, target, node.args, target(*(_fake(a) for a in node.args)), cast)
        cast.replace_all_uses_with(widened)
        gm.graph.erase_node(cast)
        gm.graph.erase_node(node)
        records.append({"product": node.name, "cast": cast.name, "widened": widened.name,
                        "shape": list(value.shape)})
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


# Ablation switch for fuse_cat_casts (paired same-GPU comparisons only).
FUSE_CAT_CASTS = True


def fuse_cat_casts(gm):
    """``bf16(views(cat(xs, 0)))`` -> ``views(cat_cast(xs, 0, bf16))`` for FP32 pieces (ops/cast_concat.py).

    The concatenation's only reader is a chain of reshape-family views whose only reader is one plain FP32 -> BF16
    cast (e.g. the BF16 product operand of stacked per-expert weights). A cast is elementwise, so casting the
    pieces into the concatenated layout gives the same values; one pass replaces the FP32 copy and its cast.
    """
    from . import cast_concat  # noqa: F401  (registers latch_ops::cat_cast)
    target = torch.ops.latch_ops.cat_cast.default
    layout = model_layouts(gm)
    records = []
    if not FUSE_CAT_CASTS:
        return records
    for cast in list(gm.graph.nodes):
        if cast.op != "call_function" or not _float_cast(cast):
            continue
        if _fake(cast).dtype != torch.bfloat16 or _fake(cast.args[0]).dtype != torch.float32:
            continue
        views, source = [], cast.args[0]
        while (source.op == "call_function" and _base(source) in _RESHAPE_VIEWS and len(source.users) == 1
               and not source.kwargs and isinstance(source.args[0], torch.fx.Node)):
            views.append(source)
            source = source.args[0]
        if (source.op != "call_function" or _base(source) != "cat" or len(source.users) != 1 or source.kwargs
                or not isinstance(source.args[0], (list, tuple))):
            continue
        dim = int(source.args[1]) if len(source.args) > 1 else 0
        pieces = list(source.args[0])
        if dim % _fake(source).dim() or not all(isinstance(p, torch.fx.Node) and _fake(p) is not None
                                                and _fake(p).dtype == torch.float32 for p in pieces):
            continue
        if any(layout.get(p) is None or tuple(layout[p][1]) != tuple(_contiguous(layout[p][0])) for p in pieces):
            continue          # pieces read in their contiguous order (the lowering re-checks)
        with gm.graph.inserting_before(cast):
            value = target([_fake(p) for p in pieces], 0, torch.bfloat16)
            result = _new(gm, target, (pieces, 0, torch.bfloat16), value, cast)
            for view in reversed(views):
                result = _new(gm, view.target, (result, *view.args[1:]),
                              view.target(result.meta["val"], *view.args[1:]), view)
        cast.replace_all_uses_with(result)
        gm.graph.erase_node(cast)
        for view in views:
            gm.graph.erase_node(view)
        gm.graph.erase_node(source)
        records.append({"cat": source.name, "cast": cast.name, "fused": result.name, "pieces": len(pieces),
                        "shape": list(value.shape)})
        layout = model_layouts(gm)
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


def _transposed_readers(node, rank):
    """(direct transposing views, the one float cast or None) when every reader transposes ``node``.

    Admission of ``orient_transposed_products``: no output use, at most one plain float cast whose readers all
    transpose, and some consuming operation beyond further views is not a matrix product (whose transposed
    operands are already free views).
    """
    users = list(node.users)
    if not users or any(u.op == "output" for u in users):
        return None
    casts = [u for u in users if u.op == "call_function" and _float_cast(u)]
    direct = [u for u in users if u not in casts]
    if len(casts) > 1 or any(not (u.op == "call_function" and _swaps_last_two(u, rank)) for u in direct):
        return None
    cast_users = list(casts[0].users) if casts else []
    if casts and (not cast_users or any(u.op != "call_function" or not _swaps_last_two(u, rank)
                                         for u in cast_users)):
        return None
    readers, pending = [], [r for view in (*direct, *cast_users) for r in view.users]
    while pending:
        reader = pending.pop()
        if reader.op == "call_function" and _base(reader) in VIEWS:
            pending.extend(reader.users)
        else:
            readers.append(reader)
    if all(r.op == "call_function" and _base(r) in ("mm", "bmm") for r in readers):
        return None
    return direct, (casts[0] if casts else None)


def _move_transposed_readers(gm, node, product, direct, cast):
    """Point every transposing reader of ``node`` (or of its cast) at the oriented ``product``; erase the old."""
    for user in direct:
        user.replace_all_uses_with(product)
        gm.graph.erase_node(user)
    moved = len(direct)
    if cast is not None:
        cast_users = list(cast.users)
        with gm.graph.inserting_after(product):
            widened = gm.graph.call_function(cast.target, (product, *cast.args[1:]), dict(cast.kwargs))
            widened.meta = {"val": cast.target(product.meta["val"], *cast.args[1:], **cast.kwargs)}
        for user in cast_users:
            user.replace_all_uses_with(widened)
            gm.graph.erase_node(user)
        gm.graph.erase_node(cast)
        moved += len(cast_users)
    gm.graph.erase_node(node)
    return moved


def collapse_view_chains(gm):
    """Replace a chain of two or more views by its source (identity) or by one reshape of a contiguous source.

    Uses the lowering storage model. Offset-changing views (slice/select/as_strided) are never collapsed.
    """
    layout = model_layouts(gm)
    records = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or _base(node) not in VIEWS or not node.users:
            continue
        chain, source = [node], node.args[0]
        while (isinstance(source, torch.fx.Node) and source.op == "call_function"
               and _base(source) in VIEWS):
            chain.append(source)
            source = source.args[0]
        # Intermediate views keep their other readers; only this view's readers move to the collapsed form.
        if (len(chain) < 2 or not isinstance(source, torch.fx.Node) or source not in layout
                or any(_base(v) in _OFFSET_VIEWS | {"detach", "alias", "expand"} for v in chain)):
            continue
        reshapes_only = all(_base(v) in _RESHAPE_VIEWS for v in chain)
        lv, ls = layout.get(node), layout.get(source)
        if lv is None or ls is None:
            continue
        shape = list(lv[0])
        if lv == ls:
            replacement, rule = source, "identity"
        elif (reshapes_only and tuple(lv[1]) == tuple(_contiguous(lv[0]))
              and tuple(ls[1]) == tuple(_contiguous(ls[0]))):
            with gm.graph.inserting_before(node):
                replacement = _new(gm, aten.reshape.default, (source, shape), _fake(source).reshape(shape), node)
            rule = "contiguous reshape"
        else:
            continue
        node.replace_all_uses_with(replacement)
        records.append({"view": node.name, "source": source.name, "length": len(chain), "rule": rule})
        # LATCH-E2TTT-OUTER: incremental form of "eliminate_dead_code(); layout = model_layouts(gm)" after every
        # rewrite (O(nodes x rewrites); 255 of 327 s of an 8-layer E2-TTT capture). Same results: the first rewrite
        # runs the full dead-code pass (it also removes pre-existing dead nodes); afterwards the only nodes a
        # rewrite can make dead are this chain's views (pure; the source stays read by the replacement), erased
        # in order. The replacement has exactly the replaced view's modelled layout, so no other node's layout
        # changes; only the new reshape node is added to the map, by model_layouts' own rule.
        if len(records) == 1:
            gm.graph.eliminate_dead_code()
        else:
            for view in chain:
                if view.users:
                    break
                gm.graph.erase_node(view)
        if replacement is not source:
            layout[replacement] = _node_layout(replacement, layout)
    if records:
        gm.graph.lint()
        gm.recompile()
    return records


def fold_fp32_batch_sums(gm):
    """FP32 only: ``sum(bmm(A, B), [0])`` -> one flat product contracting over batch and rows.

    This is the weight gradient of a batch-broadcast operand. In FP32 both forms are fixed-order FP32
    accumulations of the same products (a different association, as in split-K). BF16 products round each
    batch's result before the sum, so they are never folded.
    """
    layout = model_layouts(gm)
    records = []
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or _base(node) != "sum" or node.kwargs.get("dtype") is not None:
            continue
        product = node.args[0]
        dims = node.args[1] if len(node.args) > 1 else None
        keep = bool(node.args[2]) if len(node.args) > 2 else bool(node.kwargs.get("keepdim", False))
        if (not isinstance(product, torch.fx.Node) or product.op != "call_function" or _base(product) != "bmm"
                or len(product.users) != 1 or [int(d) % 3 for d in (dims or [])] != [0]):
            continue
        out = _fake(node)
        if out is None or out.dtype != torch.float32 or _fake(product).dtype != torch.float32:
            continue
        a, b = product.args
        la, lb = layout.get(a), layout.get(b)
        if la is None or lb is None:
            continue
        (batch, k, m), sa = la
        (_, mb, n), sb = lb
        if m != mb or sa[0] != m * sa[2] or sb[0] != m * sb[1]:
            continue          # batch does not merge with the contraction rows in both operands
        with gm.graph.inserting_before(node):
            moved = _new(gm, aten.permute.default, (a, [1, 0, 2]), _fake(a).permute(1, 0, 2), a)
            left = _new(gm, aten.reshape.default, (moved, [k, batch * m]), moved.meta["val"].reshape(k, batch * m), a)
            right = _new(gm, aten.reshape.default, (b, [batch * m, n]), _fake(b).reshape(batch * m, n), b)
            flat = _new(gm, aten.mm.default, (left, right), torch.mm(left.meta["val"], right.meta["val"]), product)
            shape = [1, k, n] if keep else [k, n]
            result = _new(gm, aten.reshape.default, (flat, shape), flat.meta["val"].reshape(shape), node)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        gm.graph.erase_node(product)
        records.append({"sum": node.name, "product": product.name, "flat": flat.name, "batch": batch,
                        "k": batch * m, "rule": "FP32 batch-sum of products: one flat fixed-order FP32 product"})
        layout = model_layouts(gm)
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


# Default on for every capture. The research harness clears it only to rebuild the pre-step-05 program for a
# paired same-GPU comparison (and checks that rebuild against the base commit's source hash); it is not a
# user-facing performance option.
ENABLED = True


def canonicalize_layouts(gm):
    """All step-05 layout canonicalizations, in order. Returns {rule: records} for non-empty rules."""
    applied = {}
    if not ENABLED:
        return applied
    # Each rewrite can expose a view chain (e.g. commuting a cast yields t(t(x))), so chains are collapsed
    # before every step that inspects readers.
    collapse = ("collapsed_view_chains", collapse_view_chains)
    for rule, function in (collapse, ("commuted_cast_permutations", commute_cast_permutations), collapse,
                           ("oriented_transposed_products", orient_transposed_products), collapse,
                           ("oriented_transposed_segment_products", orient_transposed_segment_products), collapse,
                           ("widened_segment_products", widen_segment_products),
                           ("fused_cat_casts", fuse_cat_casts),
                           ("flattened_broadcast_products", flatten_broadcast_products),
                           ("folded_fp32_batch_sums", fold_fp32_batch_sums), collapse):
        records = function(gm)
        if records:
            applied.setdefault(rule, []).extend(records)
    return applied
