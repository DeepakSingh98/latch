"""Canonical attention gradient assembly (LATCH-ATTN-RECOG).

A packed query/key/value projection (one ``[d, 3d]`` weight; ``unbind``/``chunk``/``split``/column slices of its
widened result) differentiates into ONE BF16 cast of the assembled FP32 gradient: the per-operand attention
gradients dQ, dK, dV are placed into the packed layout (``cat``, or disjoint ``slice_backward`` placements summed)
and the whole ``[tokens, 3d]`` result is rounded to BF16 once (the VJP of the single widening). The attention
bodies (ops/short_attention.py output rule) write each gradient as its own BF16 value. This capture rewrite moves
the cast before the assembly:

    cast(M(assemble(x_1, ..., x_n)))  ->  M(assemble(cast(x_1), ..., cast(x_n)))

where ``M`` is a chain of layout operations and copies and ``assemble`` is ``cat`` or a sum of ``slice_backward``
placements with pairwise disjoint ranges along one dimension. Every element of the result is the cast of exactly one
gradient element (or +0), so values are unchanged: rounding is elementwise, layout operations and ``cat`` move
values, and for the disjoint placement sum each element is ``g + (+0)...`` in both orders (a -0 gradient becomes +0
before or after the cast alike). Scope: every assembled piece must be reached from a batched product (``bmm``, an
attention gradient) through layout operations and copies; other graphs are unchanged.
"""
from __future__ import annotations

import torch

aten = torch.ops.aten
_LAYOUT = {"view", "_unsafe_view", "reshape", "permute", "transpose", "t", "unsqueeze", "squeeze", "alias",
           "detach", "expand"}
_COPIES = {"clone", "contiguous"}


def _base(node):
    if node.op != "call_function":
        return None
    return str(getattr(node.target, "__name__", node.target)).split(".")[0]


def _fake(node):
    value = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return value if isinstance(value, torch.Tensor) else None


def _narrowing_cast(node):
    """``_to_copy(x, dtype=bfloat16)`` of an FP32 tensor (strided, preserve format, same device)."""
    if node.op != "call_function" or _base(node) != "_to_copy" or len(node.args) != 1:
        return False
    if set(node.kwargs) - {"dtype", "memory_format", "layout", "device"} or node.kwargs.get("dtype") != torch.bfloat16:
        return False
    if node.kwargs.get("memory_format", torch.preserve_format) is not torch.preserve_format:
        return False
    if node.kwargs.get("layout", torch.strided) is not torch.strided:
        return False
    source, out = node.args[0], _fake(node)
    src = _fake(source)
    return out is not None and src is not None and src.dtype == torch.float32 and src.device == out.device


def _from_product(node):
    """True when ``node`` is reached from a ``bmm`` through layout operations and copies (single-use chain)."""
    while isinstance(node, torch.fx.Node):
        base = _base(node)
        if base == "bmm":
            return True
        if base not in _LAYOUT | _COPIES or len(node.users) != 1:
            return False
        node = node.args[0]
    return False


def _slice_range(node):
    """(dim, start, end, step, input sizes) of a slice_backward node."""
    grad, sizes, dim, start, end = node.args[:5]
    step = node.args[5] if len(node.args) > 5 else 1
    rank = len(sizes)
    dim = int(dim) % rank
    size = int(sizes[dim])
    start, end = int(start), int(end)
    start = max(0, min(size, start + size if start < 0 else start))
    end = max(start, min(size, end + size if end < 0 else end))
    return dim, start, end, int(step), tuple(int(s) for s in sizes)


def _assembly(node):
    """('cat', [pieces]) or ('placements', [slice_backward nodes]) or None."""
    base = _base(node)
    if base == "cat":
        pieces = list(node.args[0])
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        values = [_fake(p) for p in pieces]
        if not pieces or any(v is None or v.dtype != torch.float32 for v in values) or not isinstance(dim, int):
            return None
        return "cat", pieces
    if base == "slice_backward":
        # One placement (each operand widened separately, e.g. ``packed.float()[:, :d]``): cast(place(x)) =
        # place(cast(x)) exactly (the zeros stay +0).
        return ("placements", [node]) if _slice_range(node)[3] == 1 else None
    if base == "add":
        terms, stack = [], [node]
        while stack:
            item = stack.pop()
            b = _base(item)
            if b == "add" and len(item.args) == 2 and not item.kwargs and all(
                    isinstance(a, torch.fx.Node) for a in item.args) and (item is node or len(item.users) == 1):
                stack.extend(item.args)
            elif b == "slice_backward" and len(item.users) == 1:
                terms.append(item)
            else:
                return None
        ranges = [_slice_range(t) for t in terms]
        if len(terms) < 2 or len({(r[0], r[4]) for r in ranges}) != 1 or any(r[3] != 1 for r in ranges):
            return None
        spans = sorted((r[1], r[2]) for r in ranges)
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            return None  # overlapping placements are a real sum; keep the FP32 addition
        return "placements", terms
    return None


def narrow_assembled_gradients(gm):
    """Apply the rewrite above. Returns rewrite records."""
    records = []
    for cast in list(gm.graph.nodes):
        if cast.op != "call_function" or not _narrowing_cast(cast):
            continue
        chain, node = [], cast.args[0]
        while isinstance(node, torch.fx.Node) and _base(node) in _LAYOUT | _COPIES and len(node.users) == 1:
            chain.append(node)
            node = node.args[0]
        if not isinstance(node, torch.fx.Node) or len(node.users) != 1:
            continue
        found = _assembly(node)
        if found is None:
            continue
        kind, pieces = found
        sources = pieces if kind == "cat" else [t.args[0] for t in pieces]
        if not all(_from_product(p) for p in sources):
            continue
        kwargs = dict(cast.kwargs)
        try:
            with gm.graph.inserting_before(cast):
                def narrow(x):
                    value = aten._to_copy.default(_fake(x), **kwargs)
                    out = gm.graph.call_function(aten._to_copy.default, (x,), dict(kwargs))
                    out.meta = {"val": value}
                    if not value.is_contiguous():
                        # A transposed gradient (dK^T) keeps its strides through the cast: materialize the BF16
                        # value contiguously (the attention body writes this copy's layout directly).
                        copy = aten.clone.default(value, memory_format=torch.contiguous_format)
                        out = gm.graph.call_function(aten.clone.default, (out,),
                                                     {"memory_format": torch.contiguous_format})
                        out.meta = {"val": copy}
                    return out

                def rebuild(old, args):
                    value = old.target(*[(_fake(a) if isinstance(a, torch.fx.Node) else
                                          [(_fake(x) if isinstance(x, torch.fx.Node) else x) for x in a]
                                          if isinstance(a, (list, tuple)) and any(isinstance(x, torch.fx.Node) for x in a)
                                          else a) for a in args], **old.kwargs)
                    out = gm.graph.call_function(old.target, tuple(args), dict(old.kwargs))
                    out.meta = {k: v for k, v in old.meta.items() if k in ("stack_trace", "nn_module_stack")}
                    out.meta["val"] = value
                    return out

                if kind == "cat":
                    assembled = rebuild(node, (list(narrow(p) for p in pieces), *node.args[1:]))
                else:
                    placed = [rebuild(t, (narrow(t.args[0]), *t.args[1:])) for t in pieces]
                    assembled = placed[0]
                    for term in placed[1:]:
                        value = aten.add.Tensor(_fake(assembled), _fake(term))
                        added = gm.graph.call_function(aten.add.Tensor, (assembled, term))
                        added.meta = {"val": value}
                        assembled = added
                result = assembled
                for old in reversed(chain):
                    result = rebuild(old, (result, *old.args[1:]))
        except Exception:
            continue
        before, after = _fake(cast), _fake(result)
        if after is None or (tuple(before.shape), before.dtype) != (tuple(after.shape), after.dtype):
            continue
        cast.replace_all_uses_with(result)
        records.append({"cast": cast.name, "assembly": kind, "pieces": len(pieces), "through": [n.name for n in chain]})
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


# ----------------------------------------------------------------------------------- rank-4 score regions
_REGION_OPS = {"mul", "div", "add", "sub", "_softmax", "_safe_softmax", "_softmax_backward_data", "expand"}
_RESHAPES = {"view", "_unsafe_view", "reshape"}


def _merged_view(node, lead):
    """The rank-3 source of a reshape that splits its leading dimension into ``lead`` (else None)."""
    if _base(node) not in _RESHAPES:
        return None
    source = node.args[0]
    src, out = _fake(source), _fake(node)
    if src is None or out is None or src.dim() != 3 or out.dim() != 4:
        return None
    if tuple(out.shape[:2]) != lead or out.shape[0] * out.shape[1] != src.shape[0] or tuple(out.shape[2:]) != tuple(src.shape[1:]):
        return None
    return source if src.is_contiguous() else None


def flatten_score_regions(gm):
    """Rank-4 attention score/probability chains become the rank-3 form of their batched products.

    ``q @ k.transpose(-2, -1)`` on [b, h, t, d] tensors is a ``bmm`` over b*h viewed as [b, h, t, t]; the scale,
    additive mask, softmax over the key axis and the softmax backward then run on that view and are viewed back to
    [b*h, t, t] for the value/gradient products. Each of those operations is elementwise or per key row, so it is
    computed on the rank-3 product directly (same elements, same order within a row); masks whose leading
    dimensions broadcast (size 1) keep their own [t|1, t] form. Exact. Applied only to regions that contain a
    softmax and are entered only through such views, broadcast masks, scalars or other region values.
    """
    records = []
    for seed in list(gm.graph.nodes):
        if _base(seed) != "bmm" or len(seed.users) != 1:
            continue
        view = next(iter(seed.users))
        out = _fake(view)
        if out is None or out.dim() != 4 or _merged_view(view, tuple(out.shape[:2])) is not seed:
            continue
        lead, shape4 = tuple(out.shape[:2]), tuple(out.shape)
        # Candidate region: rank-4 values of this shape reached forward from the seed view (and from any other
        # merged view of a rank-3 product with the same shape, e.g. dP in the backward).
        region, frontier = {}, [view]
        while frontier:
            item = frontier.pop()
            for user in item.users:
                value = _fake(user)
                if (user in region or value is None or _base(user) not in _REGION_OPS
                        or tuple(value.shape) != shape4):
                    continue
                region[user] = None
                frontier.append(user)
                for arg in user.args:
                    if isinstance(arg, torch.fx.Node) and _merged_view(arg, lead) is not None and arg not in region:
                        frontier.append(arg)
        changed = True
        while changed:
            changed = False
            for node in list(region):
                ok = True
                for arg in list(node.args) + list(node.kwargs.values()):
                    if not isinstance(arg, torch.fx.Node):
                        continue
                    value = _fake(arg)
                    if arg in region or _merged_view(arg, lead) is not None:
                        continue
                    if value is not None and (value.dim() <= 2 or all(e == 1 for e in value.shape[:value.dim() - 2])):
                        continue  # broadcast mask
                    ok = False
                if _base(node) in ("_softmax", "_safe_softmax", "_softmax_backward_data"):
                    dim = node.args[1] if _base(node) != "_softmax_backward_data" else node.args[2]
                    ok = ok and int(dim) % 4 == 3
                if _base(node) == "expand" and tuple(int(x) for x in node.args[1]) not in (shape4,):
                    ok = False
                if node.kwargs and _base(node) not in ("_safe_softmax",):
                    ok = False
                if not ok:
                    del region[node]
                    changed = True
        if not any(_base(n) in ("_softmax", "_safe_softmax") for n in region):
            continue
        order = [n for n in gm.graph.nodes if n in region]
        mapped = {}

        def lifted(arg):
            if not isinstance(arg, torch.fx.Node):
                return arg
            if arg in mapped:
                return mapped[arg]
            source = _merged_view(arg, lead)
            if source is not None:
                return source
            value = _fake(arg)
            if value.dim() > 2:  # [1, ..., 1, t|1, t] broadcast mask: drop the unit leading dimensions
                with gm.graph.inserting_after(arg):
                    squeezed = gm.graph.call_function(aten.view.default, (arg, list(value.shape[-2:])))
                    squeezed.meta = {"val": value.view(*value.shape[-2:])}
                mapped[arg] = squeezed
                return squeezed
            return arg

        try:
            for node in order:
                if _base(node) == "expand":
                    mapped[node] = lifted(node.args[0])
                    continue
                args = [lifted(a) for a in node.args]
                base = _base(node)
                if base in ("_softmax", "_safe_softmax"):
                    args[1] = -1
                elif base == "_softmax_backward_data":
                    args[2] = -1
                with gm.graph.inserting_before(node):
                    value = node.target(*[_fake(a) if isinstance(a, torch.fx.Node) else a for a in args], **node.kwargs)
                    new = gm.graph.call_function(node.target, tuple(args), dict(node.kwargs))
                    new.meta = {k: v for k, v in node.meta.items() if k in ("stack_trace", "nn_module_stack")}
                    new.meta["val"] = value
                mapped[node] = new
        except Exception:
            continue
        rewired = 0
        for node in order:
            for user in list(node.users):
                if user in region:
                    continue
                value = _fake(user)
                if _base(user) in _RESHAPES and value is not None and value.dim() == 3 and \
                        tuple(value.shape) == (lead[0] * lead[1],) + shape4[2:]:
                    user.replace_all_uses_with(mapped[node])
                    rewired += 1
                    continue
                with gm.graph.inserting_after(mapped[node]):
                    back = gm.graph.call_function(aten.view.default, (mapped[node], list(shape4)))
                    back.meta = {"val": _fake(mapped[node]).view(*shape4)}
                user.replace_input_with(node, back)
        records.append({"seed": seed.name, "operations": len(order), "rank3_readers": rewired})
        gm.graph.eliminate_dead_code()  # the replaced rank-4 region; later seeds see only live values
    if records:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return records


# ----------------------------------------------------------------------------------------- shared masks
_MASK_OPS = {"view", "_unsafe_view", "reshape", "unsqueeze", "squeeze", "expand", "_to_copy", "log", "mul", "add",
             "gt", "lt", "ge", "le", "eq", "ne", "logical_and", "logical_or", "logical_not", "bitwise_not",
             "bitwise_and", "bitwise_or", "where", "arange", "full", "triu", "tril", "clone"}


def share_mask_values(gm):
    """One value per distinct attention mask computation.

    Each SDPA call (or each layer of a hand-written block) derives its additive mask from the same inputs/constants
    with the same pure operations; per-layer copies make every attention group prepare its own mask tables. Exact
    duplicates (same operation, same arguments) among the operations that only compute a softmax's additive mask
    are merged, and get_attr nodes of one shared causal-mask constant become one value. Exact (pure functions of
    identical arguments).
    """
    from .standard_ops import is_causal_constant
    records = {"constants": 0, "operations": 0}
    seen_attr = {}
    for node in list(gm.graph.nodes):
        if node.op != "get_attr":
            continue
        value = gm
        for part in str(node.target).split("."):
            value = getattr(value, part, None)
        if isinstance(value, torch.Tensor) and is_causal_constant(value):
            first = seen_attr.setdefault(id(value), node)
            if first is not node:
                node.replace_all_uses_with(first)
                gm.graph.erase_node(node)
                records["constants"] += 1
    # Mask operands: the operands of an add whose result is the input of a (safe) softmax.
    masks = []
    for node in gm.graph.nodes:
        if _base(node) in ("_softmax", "_safe_softmax"):
            source = node.args[0]
            if isinstance(source, torch.fx.Node) and _base(source) == "add":
                masks.extend(a for a in source.args if isinstance(a, torch.fx.Node))
    candidates, stack = set(), list(masks)
    while stack:
        item = stack.pop()
        if item in candidates or item.op != "call_function" or _base(item) not in _MASK_OPS:
            continue
        candidates.add(item)
        stack.extend(item.all_input_nodes)
    keys = {}
    for node in list(gm.graph.nodes):
        if node not in candidates:
            continue
        key = (node.target, repr(node.args), repr(sorted(node.kwargs.items(), key=lambda kv: kv[0])))
        if key in keys and keys[key] is not node:
            node.replace_all_uses_with(keys[key])
            gm.graph.erase_node(node)
            records["operations"] += 1
        else:
            keys[key] = node
    if records["constants"] or records["operations"]:
        gm.graph.lint()
        gm.recompile()
        return records
    return {}


def canonicalize_attention(gm):
    """Capture-time attention canonicalization (no-op without a pattern). Returns metadata or {}."""
    result = {}
    shared = share_mask_values(gm)
    if shared:
        result["shared_masks"] = shared
    flattened = flatten_score_regions(gm)
    if flattened:
        result["rank3_score_regions"] = flattened
    records = narrow_assembled_gradients(gm)
    if records:
        result["narrowed_gradient_assemblies"] = records
    return result
