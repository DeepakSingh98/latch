"""Materialize reshape copies the lowering's storage model requires (C-OPS3-A).

The tracer records PyTorch's eager strides: TensorIterator ops and ``_to_copy``
with ``preserve_format`` may return permuted (dense, non-contiguous) results, and
a later ``view`` can be legal only because of that permuted layout. Lowering
allocates every non-view result contiguously, so the same ``view`` would cross
non-contiguous stride subspaces there ("copy decomposition required").

This pass replays the lowering's layout model over the functional FX graph
(placeholders and constants keep their captured strides, every non-view result
is contiguous, view rules as in ``fusion.py``/``lower.py``) and inserts an
``aten.clone`` before each ``view``/``_unsafe_view``/``reshape`` that cannot be
represented as a view under that model. In a functional graph a view and a
copy of the same logical elements are interchangeable, so values are unchanged.
Graphs that already lower insert nothing and are left byte-identical.
"""
from __future__ import annotations

import torch

from .lower import _contiguous, _reshape_strides

aten = torch.ops.aten
VIEWS = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view", "reshape", "expand",
         "squeeze", "unsqueeze", "slice", "select", "as_strided"}
RESHAPES = {"view", "_unsafe_view", "reshape"}


def _base(node):
    target = str(node.target).removeprefix("torch.ops.").removeprefix("aten.")
    return target.split(".")[0]


def _shape(node):
    value = node.meta.get("val")
    return tuple(int(x) for x in value.shape) if isinstance(value, torch.Tensor) else None


def _view_strides(base, node, src_shape, src_strides, shape):
    args = node.args
    strides = list(src_strides)
    if base in ("detach", "alias"):
        return strides
    if base == "t":
        return list(reversed(strides))
    if base == "transpose":
        a, b = int(args[1]) % len(strides), int(args[2]) % len(strides)
        strides[a], strides[b] = strides[b], strides[a]
        return strides
    if base == "permute":
        return [src_strides[int(d) % len(src_strides)] for d in args[1]]
    if base == "expand":
        pad_shape = (1,) * (len(shape) - len(src_shape)) + tuple(src_shape)
        pad_stride = (0,) * (len(shape) - len(src_shape)) + tuple(src_strides)
        return [0 if old == 1 and new != 1 else stride for old, new, stride in zip(pad_shape, shape, pad_stride)]
    if base == "unsqueeze":
        dim = int(args[1]) % (len(src_shape) + 1)
        strides.insert(dim, src_strides[dim] * src_shape[dim] if dim < len(src_shape) else 1)
        return strides
    if base == "squeeze":
        if len(args) == 1:
            return [s for d, s in zip(src_shape, src_strides) if d != 1]
        dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
        dims = [int(d) % len(src_shape) for d in dims]
        return [s for k, (d, s) in enumerate(zip(src_shape, src_strides)) if k not in dims or d != 1]
    if base == "slice":
        dim = int(args[1]) % len(src_shape) if len(args) > 1 else 0
        step = int(args[4]) if len(args) > 4 else 1
        strides[dim] = src_strides[dim] * step
        return strides
    if base == "select":
        dim = int(args[1]) % len(src_shape)
        del strides[dim]
        return strides
    if base == "as_strided":
        return [int(s) for s in args[2]]
    if base in RESHAPES:
        contiguous = all(d == 1 or s == t for d, s, t in zip(src_shape, src_strides, _contiguous(src_shape)))
        result = _contiguous(shape) if contiguous else _reshape_strides(tuple(src_shape), tuple(src_strides), shape)
        return None if result is None else list(result)
    raise AssertionError(base)


def materialize_reshape_copies(gm):
    """Insert contiguous copies before reshapes that the lowering model cannot alias.

    Mutates ``gm`` in place; returns the list of inserted copies (empty when the
    graph already satisfies the model, in which case nothing is recompiled).
    """
    layout, inserted = {}, []
    for node in list(gm.graph.nodes):
        shape = _shape(node)
        if node.op in ("placeholder", "get_attr"):
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                layout[node] = (shape, tuple(int(s) for s in value.stride()))
            continue
        if node.op != "call_function" or shape is None:
            continue
        base = _base(node)
        source = node.args[0] if node.args and isinstance(node.args[0], torch.fx.Node) else None
        if base in VIEWS and source in layout:
            src_shape, src_strides = layout[source]
            strides = _view_strides(base, node, src_shape, src_strides, shape)
            if strides is None:
                with gm.graph.inserting_before(node):
                    copy = gm.graph.call_function(aten.clone.default, (source,),
                                                  {"memory_format": torch.contiguous_format})
                value = source.meta["val"]
                copy.meta = {k: v for k, v in source.meta.items() if k in ("stack_trace", "nn_module_stack")}
                copy.meta["val"] = value.clone(memory_format=torch.contiguous_format)
                node.replace_input_with(source, copy)
                layout[copy] = (src_shape, _contiguous(src_shape))
                strides = list(_contiguous(shape))
                inserted.append({"view": node.name, "source": source.name, "source_shape": list(src_shape),
                                 "model_strides": list(src_strides), "traced_strides": list(value.stride()),
                                 "result_shape": list(shape)})
            layout[node] = (shape, tuple(strides))
        else:
            layout[node] = (shape, _contiguous(shape))
    if inserted:
        gm.graph.lint()
        gm.recompile()
    return inserted
