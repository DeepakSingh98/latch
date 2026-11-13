"""Trace one complete PyTorch training step into a Graph.

PyTorch does the differentiation. ``torch.func.grad_and_value`` runs under
``make_fx``, so the traced ATen graph already contains the backward pass, and the
optimizer's update is traced right after it. Core ATen decompositions reduce the
long tail of operators to the small set ``ops.py`` has rules for.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import functional_call, functionalize, grad_and_value
from torch.fx.experimental.proxy_tensor import make_fx

from .ir import CompileError, Graph, Op, Ref, Value
from .optim import SGDConfig

DTYPES = {torch.float32: "float32", torch.int64: "int64", torch.bool: "bool"}


@dataclass
class Captured:
    graph: Graph
    tensors: dict[str, torch.Tensor]  # initial contents of every bound input, by graph input name

    def build(self, directory, **options):
        """Lower, generate and compile into ``directory``. See ``runtime.build``."""
        from .runtime import build
        return build(self, directory, **options)


def decompositions():
    from torch._decomp import core_aten_decompositions, get_decompositions
    aten = torch.ops.aten
    table = dict(core_aten_decompositions())
    # Core ATen keeps these whole; their upstream decompositions leave only simpler operations.
    table.update(get_decompositions([aten.addmm, aten._softmax, aten._log_softmax, aten.split_with_sizes,
                                     aten.embedding, aten.gelu]))
    table[aten.native_layer_norm.default] = _layer_norm
    return table


def _layer_norm(input, normalized_shape, weight, bias, eps):
    """LayerNorm from single-result operations (the upstream rule uses var_mean)."""
    dims = list(range(input.dim() - len(normalized_shape), input.dim()))
    mean = input.mean(dims, keepdim=True)
    centered = input - mean
    rstd = torch.rsqrt((centered * centered).mean(dims, keepdim=True) + eps)
    out = centered * rstd
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out, mean, rstd


def capture_module(model, loss_fn, inputs, targets=(), *, optimizer=None) -> Captured:
    """Capture ``loss_fn(model(*inputs), *targets)``, its gradients and one optimizer update.

    ``inputs`` and ``targets`` are example tensors; their shapes and dtypes are fixed
    from here on. Parameters with ``requires_grad=False`` and buffers are bound as
    read-only constants. Python control flow is specialized to what this call ran.
    """
    optimizer = optimizer or SGDConfig()
    inputs, targets = tuple(inputs), tuple(targets)
    params = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    constants = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    constants.update({name: b.detach().clone() for name, b in model.named_buffers()})
    state = optimizer.init(params)
    groups = [("parameter", list(params)), ("optimizer", list(state)), ("constant", list(constants)),
              ("input", list(range(len(inputs)))), ("target", list(range(len(targets))))]
    flat = [*params.values(), *state.values(), *constants.values(), *inputs, *targets]
    for value in flat:
        if not isinstance(value, torch.Tensor) or value.dtype not in DTYPES:
            raise CompileError(f"bound tensors must be float32, int64 or bool; got {getattr(value, 'dtype', type(value))}")
    np_, ns, nc, ni = len(params), len(state), len(constants), len(inputs)

    def step(*flat):
        p = dict(zip(params, flat[:np_]))
        s = dict(zip(state, flat[np_:np_ + ns]))
        c = dict(zip(constants, flat[np_ + ns:np_ + ns + nc]))
        x, y = flat[np_ + ns + nc:np_ + ns + nc + ni], flat[np_ + ns + nc + ni:]

        def loss_of(p):
            return loss_fn(functional_call(model, {**p, **c}, x), *y)

        grads, loss = grad_and_value(loss_of)(p)
        new_p, new_s = optimizer.update(p, grads, s)
        return (loss, *(new_p[name] for name in params), *(new_s.get(name, s[name]) for name in state))

    gm = make_fx(functionalize(step), decomposition_table=decompositions(), tracing_mode="fake")(*flat)
    gm.graph.eliminate_dead_code()
    return Captured(*_import(gm, flat, groups))


def _import(gm, flat, groups):
    """FX graph -> Graph, plus the tensors bound to its inputs."""
    values, ops, inputs, roles, tensors = {}, [], [], {}, {}
    keys = [[kind, key] for kind, names in groups for key in names]
    canonical, seen = {}, {}  # value name -> representative; operation signature -> first result

    def encode(item):
        if isinstance(item, torch.fx.Node):
            return Ref(canonical[item.name])
        if isinstance(item, (tuple, list)):
            return tuple(encode(x) for x in item)
        if isinstance(item, torch.dtype):
            if item not in DTYPES:
                raise CompileError(f"unsupported dtype {item}")
            return DTYPES[item]
        if isinstance(item, (torch.device, torch.layout, torch.memory_format)):
            return str(item)
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        raise CompileError(f"unsupported literal {item!r}")

    def declare(node, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in DTYPES:
            raise CompileError(f"{node.name} ({node.target}): results must be single float32/int64/bool tensors")
        values[node.name] = Value(node.name, tuple(int(n) for n in tensor.shape), DTYPES[tensor.dtype])
        canonical[node.name] = node.name

    result = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            declare(node, node.meta["val"])
            inputs.append(node.name)
            roles[node.name], tensors[node.name] = keys[len(inputs) - 1], flat[len(inputs) - 1]
        elif node.op == "get_attr":  # a tensor the trace closed over: bind it as one more constant
            tensor = getattr(gm, node.target).detach().clone()
            declare(node, tensor)
            inputs.append(node.name)
            roles[node.name], tensors[node.name] = ["constant", str(node.target)], tensor
        elif node.op == "call_function":
            target = str(node.target)
            op = Op(node.name, target, encode(node.args), {k: encode(v) for k, v in node.kwargs.items()})
            signature = repr((op.target, op.args, sorted(op.kwargs.items())))
            if signature in seen:  # common subexpression: the graph is pure, so reuse the first result
                canonical[node.name] = seen[signature]
                continue
            declare(node, node.meta["val"])
            seen[signature] = node.name
            ops.append(op)
        elif node.op == "output":
            result = encode(node.args[0])
        else:
            raise CompileError(f"unsupported graph node {node.op}")
    loss, new = result[0], result[1:]
    if not isinstance(loss, Ref) or values[loss.name].shape != () or values[loss.name].dtype != "float32":
        raise CompileError("the loss must be a float32 scalar tensor")
    persistent = [name for name in inputs if roles[name][0] in ("parameter", "optimizer")]
    updates = {}
    for name, value in zip(persistent, new):
        if not isinstance(value, Ref) or (values[value.name].shape, values[value.name].dtype) != (
                values[name].shape, values[name].dtype):
            raise CompileError(f"the update of {roles[name][1]} changes its shape or dtype")
        if value.name != name:
            updates[name] = value.name
    return Graph(values, ops, inputs, roles, loss.name, updates), tensors
