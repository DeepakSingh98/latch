"""Capture-time channels-last propagation around pixel-major convolution products (LATCH-STEP10-SPEED).

The pixel-major convolution form (ops/conv_ops.py, ``nhwc``) produces dense (N*H*W, C) products and returns their
NCHW values through ``conv3_unpixels``; its operands enter through ``conv3_pixels``. Left alone, every elementwise
operation between two convolutions would run in NCHW order and each crossing would be a transposing copy.

This pass rewrites the functional FX graph so that such operations run on the pixel-major (N, H, W, C) values
directly. A rank-4 elementwise operation (or a sum/mean reduction) is *converted* when one of its rank-4 tensor
operands is converted, or when all of its readers are converted or are ``conv3_pixels`` (fixpoint). A converted
operation is re-emitted with the same target on permuted operands: converted operands are used directly, other
rank-4 operands through ``aten.permute(x, (0, 2, 3, 1))``, lower-rank broadcast operands through a leading-ones
view and the same permutation; reduction dimensions are renumbered. ``conv3_unpixels(y)`` becomes a view of ``y``
and ``conv3_pixels(v)`` of a converted value a view of its pixel-major form. Readers that are not converted see
``aten.permute(v_nhwc, (0, 3, 1, 2))``, a view. Graph outputs are never converted.

Every rewrite is a reindexing of the same elementwise expressions (same operands, operation order and dtypes), so
values are unchanged. Graphs without ``conv3_unpixels`` are untouched. No model names are inspected.
"""
from __future__ import annotations

import torch
from torch.fx import Node
from torch.fx.node import map_arg

aten = torch.ops.aten
PERM = [0, 2, 3, 1]                  # NCHW -> NHWC
INV = [0, 3, 1, 2]                   # NHWC -> NCHW
POS = (0, 3, 1, 2)                   # position of logical dim d in the permuted order
UNPIXELS = "latch_ops.conv3_unpixels.default"
PIXELS = "latch_ops.conv3_pixels.default"
ELEMENTWISE = {"add", "sub", "mul", "div", "neg", "rsub", "pow", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs",
               "reciprocal", "tanh", "sigmoid", "silu", "relu", "erf", "silu_backward", "tanh_backward",
               "sigmoid_backward", "threshold_backward", "clone", "_to_copy", "fill", "zeros_like", "ones_like",
               "full_like", "where", "masked_fill", "clamp", "clamp_min", "clamp_max", "minimum", "maximum",
               "eq", "ne", "lt", "le", "gt", "ge", "logical_not", "logical_and", "logical_or", "empty_like"}
# Operations that read only their template's shape/dtype: permuted by a view, never by a data copy.
TEMPLATES = {"empty_like", "zeros_like", "ones_like", "full_like"}
REDUCTIONS = {"sum", "mean"}
ENABLED = True
DENSE_COPY = True      # dense unconverted rank-4 operands: one pixel-major copy (else strided permuted reads)
REMAT_WIDENED = False  # (neutral on block_dev, gpu-12) backward readers recompute widen(BF16 product) +/- small operands (see _rematerialize_widened)
REMAT_FULL = False     # REMAT_WIDENED also with full-size input-derived operands (e.g. a residual input)


def _name(node):
    return str(node.target).removeprefix("torch.ops.")


def _base(node):
    return _name(node).removeprefix("aten.").split(".")[0]


def _val(node):
    value = node.meta.get("val") if isinstance(node, Node) else None
    return value if isinstance(value, torch.Tensor) else None


def _rank4(node):
    value = _val(node)
    return value is not None and value.dim() == 4


def _tensor_args(node):
    out = []
    for a in list(node.args) + list(node.kwargs.values()):
        if isinstance(a, (list, tuple)):
            if any(isinstance(x, Node) for x in a):
                return None                      # tensor lists are not admitted
            continue
        if isinstance(a, Node) and _val(a) is not None:
            out.append(a)
    return out


def _reduction_dims(node):
    """Logical reduced dims and keepdim of an admitted rank-4 sum/mean, or None."""
    target = _name(node)
    if target not in ("aten.sum.dim_IntList", "aten.mean.dim"):
        return None
    dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if dims is None:
        return None
    keep = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False))
    dims = sorted({int(d) % 4 for d in dims})
    kept = [d for d in range(4) if d not in dims]
    if [POS[d] for d in kept] != sorted(POS[d] for d in kept):
        return None                              # the result's element order would change
    return dims, keep


def propagate(gm):
    """Mutates ``gm``; returns a small record (converted operation names), empty when nothing applies."""
    graph = gm.graph
    nodes = list(graph.nodes)
    if not ENABLED or not any(n.op == "call_function" and _name(n) == UNPIXELS for n in nodes):
        return {}
    output = next(n for n in nodes if n.op == "output")
    outputs = {a for a in output.all_input_nodes}

    def kind(n):
        if n.op != "call_function":
            return None
        name = _name(n)
        if name == UNPIXELS:
            return "source"
        if name == PIXELS:
            return "sink"
        if n in outputs:
            return None
        base = _base(n)
        if base in ELEMENTWISE and _rank4(n):
            args = _tensor_args(n)
            if args is not None and all(_val(a).dim() <= 4 for a in args):
                return "elementwise"
        if base in REDUCTIONS and n.args and _rank4(n.args[0]) and _reduction_dims(n) is not None:
            return "reduction"
        return None

    kinds = {n: kind(n) for n in nodes}
    converted = {n for n in nodes if kinds[n] == "source"}
    changed = True
    while changed:
        changed = False
        for n in nodes:
            if n in converted or kinds[n] not in ("elementwise", "reduction"):
                continue
            inputs = [a for a in (_tensor_args(n) or []) if _rank4(a)]
            if kinds[n] == "reduction":
                take = bool(inputs) and inputs[0] in converted
            else:
                users = list(n.users)
                take = (any(a in converted for a in inputs)
                        or (bool(users) and all(u in converted or kinds[u] == "sink" for u in users)))
            if take:
                converted.add(n)
                changed = True
    if len(converted) == sum(1 for n in nodes if kinds[n] == "source") and not any(
            kinds[n] == "sink" for n in nodes):
        return {}

    from torch._guards import detect_fake_mode
    mode = detect_fake_mode([_val(n) for n in nodes if _val(n) is not None])

    def make(target, args, kwargs, like, *, contiguous=False):
        node = graph.call_function(target, args, kwargs)
        node.meta = {k: v for k, v in like.meta.items() if k in ("stack_trace", "nn_module_stack")}
        vals = map_arg(args, lambda a: a.meta["val"])
        kvals = map_arg(kwargs, lambda a: a.meta["val"])
        with mode:
            value = target(*vals, **kvals)
            if contiguous and not value.is_contiguous():
                value = value.contiguous()
        node.meta["val"] = value
        return node

    nhwc = {}             # original node -> pixel-major (N, H, W, C) node
    record = {"converted": [], "sources": [], "sinks": [], "permuted_operands": 0}

    copies = {}

    def operand(a, user, template=False):
        if not isinstance(a, Node) or _val(a) is None:
            return a
        if a in nhwc:
            return nhwc[a]
        value = _val(a)
        if value.dim() == 0:
            return a
        if (DENSE_COPY and not template and value.dim() == 4 and value.dtype in (torch.float32, torch.bfloat16)
                and 0 not in value.stride() and min(value.shape) > 1):
            # A dense NCHW operand (e.g. a graph input) is copied once to pixel-major order by conv3_pixels' tiled
            # transpose; every converted reader then reads it contiguously instead of through strided reads.
            if a not in copies:
                from .conv_ops import conv3_pixels
                b, c, h, w = (int(v) for v in value.shape)
                with graph.inserting_before(user):          # the first converted reader (topological order)
                    pixels = make(torch.ops.latch_ops.conv3_pixels.default, (a,), {}, a)
                    copies[a] = make(aten.reshape.default, (pixels, [b, h, w, c]), {}, a)
                record["dense_copies"] = record.get("dense_copies", 0) + 1
            return copies[a]
        with graph.inserting_before(user):
            x = a
            if value.dim() < 4:
                x = make(aten.view.default, (a, [1] * (4 - value.dim()) + list(value.shape)), {}, a)
            record["permuted_operands"] += 1
            return make(aten.permute.default, (x, PERM), {}, a)

    for n in nodes:
        k = kinds[n]
        if k == "source" and n in converted:
            y = n.args[0]
            b, h, w = (int(v) for v in n.args[1:4])
            with graph.inserting_before(n):
                nhwc[n] = make(aten.reshape.default, (y, [b, h, w, int(_val(y).shape[1])]), {}, n)
            record["sources"].append(n.name)
        elif k == "sink" and n.args[0] in nhwc:
            source = nhwc[n.args[0]]
            shape = list(_val(n).shape)
            with graph.inserting_before(n):
                view = make(aten.reshape.default, (source, shape), {}, n)   # eager results may be strided
            n.replace_all_uses_with(view)
            record["sinks"].append(n.name)
        elif k == "sink" and DENSE_COPY and isinstance(n.args[0], Node) and n.args[0] not in copies:
            # An explicit pixel-major copy of an unconverted operand also serves its converted readers.
            value = _val(n.args[0])
            b, c, h, w = (int(v) for v in value.shape)
            with graph.inserting_after(n):
                copies[n.args[0]] = make(aten.reshape.default, (n, [b, h, w, c]), {}, n)
        elif n in converted:
            template = _base(n) in TEMPLATES
            args = tuple(operand(a, n, template) if isinstance(a, Node) else a for a in n.args)
            kwargs = {key: operand(v, n, template) if isinstance(v, Node) else v for key, v in n.kwargs.items()}
            if "memory_format" in kwargs:
                kwargs["memory_format"] = torch.contiguous_format
            if k == "reduction":
                dims, keep = _reduction_dims(n)
                if len(args) > 1:
                    args = (args[0], [POS[d] for d in dims], *args[2:])
                else:
                    kwargs["dim"] = [POS[d] for d in dims]
                with graph.inserting_before(n):
                    new = make(n.target, args, kwargs, n)
                if keep:
                    nhwc[n] = new
                else:
                    n.replace_all_uses_with(new)
            else:
                with graph.inserting_before(n):
                    nhwc[n] = make(n.target, args, kwargs, n, contiguous=True)
            record["converted"].append(n.name)
    # Remaining readers of converted values (not converted themselves) see an NCHW view.
    for n, p in nhwc.items():
        if not n.users:
            continue
        with graph.inserting_after(p):
            view = make(aten.permute.default, (p, INV), {}, n)
        n.replace_all_uses_with(view)
    graph.eliminate_dead_code()
    if REMAT_WIDENED:
        record["rematerialized"] = _rematerialize_widened(graph, make)
    graph.eliminate_dead_code()
    graph.lint()
    gm.recompile()
    return record


def _input_derived(node, depth=6):
    """A graph input, or a view/pixel-major copy chain of one (its value exists for the whole update)."""
    for _ in range(depth):
        if not isinstance(node, Node):
            return False
        if node.op == "placeholder":
            return True
        if node.op != "call_function" or not node.args or not isinstance(node.args[0], Node):
            return False
        if _base(node) not in ("view", "reshape", "permute", "_unsafe_view") and _name(node) != PIXELS:
            return False
        node = node.args[0]
    return False


def _rematerialize_widened(graph, make):
    """Recompute ``v = widen(p) (+|-|*) small`` for its readers after the loss from the BF16 value ``p``.

    ``v`` (FP32, rank 4) is an exact widening of a BF16 tensor combined with broadcast operands (e.g. a pixel-major
    convolution product plus its bias). Readers before the loss keep ``v``; later (backward) readers get a copy of the
    same two operations placed before the first of them, so ``v`` is consumed where it is produced and only the BF16
    value lives until the backward pass (half the bytes). Same operations on the same values: bitwise unchanged."""
    nodes = list(graph.nodes)
    output = next(n for n in nodes if n.op == "output")
    flat = output.args[0] if isinstance(output.args[0], (list, tuple)) else [output.args[0]]
    loss = flat[0] if flat and isinstance(flat[0], Node) else None
    if loss is None:
        return []
    order = {n: k for k, n in enumerate(nodes)}
    done = []
    for v in nodes:
        if v.op != "call_function" or _base(v) not in ("add", "sub", "mul") or not _rank4(v):
            continue
        value = _val(v)
        if value.dtype != torch.float32 or v in flat:
            continue
        wide = [a for a in v.args if isinstance(a, Node) and _base(a) == "_to_copy" and _val(a) is not None
                and _val(a).dtype == torch.float32 and isinstance(a.args[0], Node) and _val(a.args[0]) is not None
                and _val(a.args[0]).dtype == torch.bfloat16 and tuple(_val(a.args[0]).shape) == tuple(value.shape)]
        others = [a for a in v.args if isinstance(a, Node) and a not in wide]
        # Small (broadcast) operands always qualify; with REMAT_FULL also full-size operands that are graph-input
        # values (or pixel-major copies of them), which stay alive for the whole update anyway.
        def cheap(a):
            if _val(a) is None:
                return False
            if _val(a).numel() * 64 <= value.numel():
                return True
            return REMAT_FULL and _input_derived(a)
        if len(wide) != 1 or not all(cheap(a) for a in others):
            continue
        late = [u for u in v.users if order.get(u, -1) > order[loss]]
        early = [u for u in v.users if order.get(u, -1) <= order[loss]]
        if not late or not early:
            continue
        first = min(late, key=lambda u: order[u])
        widen = wide[0]
        with graph.inserting_before(first):
            widen2 = make(widen.target, widen.args, dict(widen.kwargs), widen, contiguous=True)
            args2 = tuple(widen2 if a is widen else a for a in v.args)
            v2 = make(v.target, args2, dict(v.kwargs), v, contiguous=True)
        for u in late:
            u.replace_input_with(v, v2)
        done.append(v.name)
    return done
