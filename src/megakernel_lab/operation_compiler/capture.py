"""Functional PyTorch training capture, pinned and tested against torch 2.9.1.

PyTorch provides differentiation; this module imports the resulting ATen graph.
Optimizer arithmetic is captured in the same graph, without handwritten backward
schedules. The scalar-loss wrapper pattern follows Latch 0.6 torch_frontend/capture.py
(0987d803f7b8be0d41cc653efd928e6581a97026), but no recognizer is used here.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import contextvars
from dataclasses import dataclass
import sys
import threading
from typing import Any, Callable

import torch
from torch.func import functional_call, functionalize, grad_and_value
from torch.fx.experimental.proxy_tensor import make_fx

from .ir import CaptureError, Operation, Ref, TrainingIR, UniformConditionalIR, Value
from .optim import AdamWConfig, SGDConfig, prepare_optimizer
from .standard_ops import capture_functions


@dataclass
class CapturedTraining:
    ir: TrainingIR
    graph_module: torch.fx.GraphModule
    step_fn: Callable
    example_args: tuple[torch.Tensor, ...]
    parameter_names: tuple[str, ...]
    parameter_aliases: dict[str, str]
    config: SGDConfig | AdamWConfig
    constants: dict[str, torch.Tensor]
    state_names: tuple[str, ...] = ()

    @property
    def optimizer_state_names(self):
        return tuple(self.ir.metadata.get("optimizer_state_names", self.parameter_names))

    def unpack(self, outputs) -> dict[str, Any]:
        n = len(self.parameter_names)
        no = len(self.optimizer_state_names)
        if len(outputs) != 1 + 2 * n + no + len(self.state_names):
            raise ValueError("unexpected training-step output arity")
        result = {"loss": outputs[0],
                "grads": dict(zip(self.parameter_names, outputs[1:1 + n])),
                "params": dict(zip(self.parameter_names, outputs[1 + n:1 + 2 * n])),
                "optimizer_state": dict(zip(self.optimizer_state_names, outputs[1 + 2 * n:1 + 2 * n + no]))}
        if "momentum" in self.ir.outputs:
            result["momentum"] = result["optimizer_state"]
        if "state_names" in self.ir.metadata:
            result["state"] = dict(zip(self.state_names, outputs[1 + 2 * n + no:]))
        return result


_ALIASES = {
    "aten.detach.default", "aten.alias.default", "aten.view.default",
    "aten._unsafe_view.default", "aten.t.default", "aten.transpose.int",
    "aten.permute.default", "aten.expand.default", "aten.squeeze.default",
    "aten.squeeze.dim", "aten.squeeze.dims", "aten.unsqueeze.default",
    "aten.slice.Tensor", "aten.select.int", "aten.as_strided.default",
}


def capture_decompositions():
    """Minimal pinned upstream operation decompositions, never model rewrites.

    PyTorch 2.9.1 nll_loss returns a tuple; its registered forward/backward
    decompositions expose gather, masks, reductions and scatter arithmetic to
    this single-Tensor operation IR. Other operators retain existing capture.
    Source: pytorch/v2.9.1/torch/_decomp/decompositions.py.
    """
    from torch._decomp import get_decompositions
    from .standard_ops import decompositions as standard_decompositions
    table = get_decompositions([torch.ops.aten.nll_loss_forward.default,
                                torch.ops.aten.nll_loss_backward.default])
    # C-OPS2-B: LayerNorm/GELU/addmm/MSE/multi-result views (pinned upstream) and
    # the admitted non-overlapping convolution class. Operators absent from a
    # graph are untouched, so earlier captured programs are unchanged.
    table.update(standard_decompositions())
    return table


def _reject_lazy_view(value: torch.Tensor, label: str) -> None:
    # CUDA bindings address physical storage. These metadata bits change the
    # logical tensor without changing its data pointer or ordinary strides.
    if value.is_neg() or value.is_conj():
        raise CaptureError("LAZY_VIEW", f"{label}: resolve lazy negative/conjugate views before capture")


def _reject_placeholder_storage_alias(values, labels) -> None:
    # make_fx can specialize duplicate Tensor arguments to one placeholder.
    # Canonical tied parameters have already been deduplicated. Conservatively
    # refuse any storage shared by the remaining explicit placeholders, even
    # disjoint views, until alias guards are represented in the capture contract.
    owners = {}
    for value, label in zip(values, labels):
        if value.layout != torch.strided:
            raise CaptureError("PLACEHOLDER_LAYOUT", f"{label}: explicit placeholders must use strided storage")
        if not value.numel():
            continue
        storage = (value.device, value.untyped_storage().data_ptr())
        if storage in owners:
            raise CaptureError("PLACEHOLDER_STORAGE", f"{label} and {owners[storage]} share explicit-placeholder "
                               "storage; clone inputs/state before capture (canonical tied parameters remain supported)")
        owners[storage] = label


def _reject_effects(node):
    if node.op != "call_function":
        return
    target = str(node.target)
    schema = getattr(node.target, "_schema", None)
    if schema is not None and schema.is_mutable:
        raise CaptureError("MUTATION", f"captured mutation {target}; use a functional update")
    tags = set(getattr(node.target, "tags", ()))
    if torch.Tag.nondeterministic_seeded in tags or any(
            text in target for text in ("dropout", "bernoulli", "rand", "normal", "multinomial")):
        raise CaptureError("RNG", f"{target} needs explicit random-state advancement before lowering")


def _functionalize_internal_mutations(gm, arguments, *, source_locations=False):
    """Remove AD's private temporary writes while refusing external mutation.

    PyTorch derivative graphs can fill freshly allocated buffers in place. The
    functionalization transform turns these into pure value-producing operations;
    writes to caller-owned inputs survive as copy_ effects and remain refused.
    Inspect seeded randomness before any dead-code elimination/functionalization.
    """
    mutable = False
    for node in gm.graph.nodes:
        schema = getattr(node.target, "_schema", None) if node.op == "call_function" else None
        if schema is not None and schema.is_mutable:
            mutable = True
        else:
            _reject_effects(node)
    if mutable:
        # LATCH-EXPLAIN (opt-in only): replaying the traced graph through torch.fx.Interpreter lets FX copy each
        # original node's recorded stack trace / module path onto the functionalized nodes (preserve_node_meta).
        # The executed operations are the same nodes in the same order; the plain path calls the module directly.
        traced = _InterpreterReplay(gm) if source_locations else gm
        gm = make_fx(functionalize(traced, remove="mutations"), decomposition_table=capture_decompositions(), tracing_mode="fake",
                     _allow_non_fake_inputs=True, _error_on_data_dependent_ops=True)(*arguments)
    for node in gm.graph.nodes:
        _reject_effects(node)
    return gm


class _InterpreterReplay:
    """Callable that runs a GraphModule through torch.fx.Interpreter (metadata-preserving replay)."""

    def __init__(self, gm):
        self.gm = gm

    def __call__(self, *args):
        return torch.fx.Interpreter(self.gm).run(*args)


def _refs(obj):
    if isinstance(obj, Ref):
        yield obj.name
    elif isinstance(obj, (tuple, list)):
        for item in obj:
            yield from _refs(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _refs(item)


def _encode(obj):
    if isinstance(obj, torch.fx.Node):
        return Ref(obj.name)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_encode(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, torch.dtype):
        return str(obj).removeprefix("torch.")
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.layout):
        return str(obj).removeprefix("torch.")
    if isinstance(obj, torch.memory_format):
        return str(obj).removeprefix("torch.")
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise CaptureError("LITERAL", f"unsupported captured argument {type(obj).__name__}: {obj}")


MULTI_RESULT_TARGETS = {"latch_ops.selective_scan_fwd.default", "latch_ops.selective_scan_bwd.default",
                        "latch_ops.ragged_order.default",
                        "latch_ops.bf16_sdpa_fwd.default", "latch_ops.bf16_sdpa_bwd.default",
                        # P8-C window attention (ops/window_attention.py)
                        "latch_ops.mattn_fwd.default", "latch_ops.mattn_bwd.default",
                        # P9-C absorbed window attention (ops/window_fusion.py)
                        "latch_ops.mattn_fwd2.default", "latch_ops.mattn_bwd2.default"}


def _value(node, **kwargs):
    return _tensor_value(node.name, node.meta.get("val"), **kwargs)


def _tensor_value(name, value, **kwargs):
    if not isinstance(value, torch.Tensor):
        raise CaptureError("VALUE", f"{name} does not return a single Tensor ({type(value).__name__})")
    try:
        shape = tuple(int(x) for x in value.shape)
        strides = tuple(int(x) for x in value.stride())
        offset = int(value.storage_offset())
    except (TypeError, ValueError) as exc:
        raise CaptureError("SHAPE", f"{name} requires static shapes and strides") from exc
    return Value(name, shape, str(value.dtype).removeprefix("torch."), strides,
                 storage_offset=offset, **kwargs)


def _multi_result_names(node):
    """P6-E: stable names for an explicitly admitted tuple-valued operation."""
    import operator
    names = [None] * len(node.meta["val"])
    for user in node.users:
        if user.op != "call_function" or user.target is not operator.getitem or user.kwargs or len(user.args) != 2:
            raise CaptureError("MULTI_RESULT", f"{node.name}: tuple results may only be selected by getitem")
        index = user.args[1]
        if type(index) is not int or not 0 <= index < len(names) or names[index] is not None:
            raise CaptureError("MULTI_RESULT", f"{node.name}: duplicate or invalid result selection")
        names[index] = user.name
    return [name or f"{node.name}_result_{i}" for i, name in enumerate(names)]


def discover_gradient_participation(function, params, arguments=(), *, has_aux=False):
    """Cloud CPU ordinary-autograd discovery, preserving None versus zero.

    ``function(pflat, *arguments)`` may return several floating outputs. This
    examines graph connectivity, not gradient values. Clone every supplied
    tensor so discovery cannot mutate caller-owned bindings. Call only after
    the fake trace has refused external effects/runtime Python control.
    """
    from torch.utils._pytree import tree_flatten
    pflat = tuple(p.detach().clone().requires_grad_(True) for p in params)
    args = tuple(a.detach().clone() for a in arguments)
    with torch.enable_grad():
        output = function(pflat, *args)
        if has_aux:
            output = output[0]
        leaves, _ = tree_flatten(output)
        differentiable = tuple(x for x in leaves if isinstance(x, torch.Tensor) and x.requires_grad)
        if not differentiable:
            return (False,) * len(pflat)
        gradients = torch.autograd.grad(differentiable, pflat,
                                        grad_outputs=tuple(torch.ones_like(x) for x in differentiable),
                                        allow_unused=True, materialize_grads=False)
    return tuple(value is not None for value in gradients)


def hoist_tensor_constants(gm):
    """Append readonly cloned constant placeholders; return names and bindings.

    Mutates the captured GraphModule in place. Constant attributes are snapshotted
    so mutating the original closure later cannot change this captured program.
    The returned placeholder names are in appended ABI order.
    """
    nodes = [node for node in gm.graph.nodes if node.op == "get_attr"]
    if not nodes:
        return (), (), {}
    anchor = next(node for node in gm.graph.nodes if node.op != "placeholder")
    attributes, values, bindings = [], [], {}
    causal_snapshots = {}
    for node in nodes:
        value = gm
        for part in node.target.split("."):
            value = getattr(value, part)
        if not isinstance(value, torch.Tensor):
            raise CaptureError("SUBGRAPH", f"nested graph/object {node.target} requires a region lowering")
        if value.layout != torch.strided:
            raise CaptureError("CONSTANT", "ordinary captured constants must be strided tensors")
        _reject_lazy_view(value, f"constant {node.target}")
        from .standard_ops import causal_constant_key
        causal_key = causal_constant_key(value)
        if causal_key is not None and causal_key in causal_snapshots:
            # The paired primal/VJP can hold different Python snapshots of
            # the same mask. Exact byte proof above permits one readonly ABI
            # binding and retains strict reference equality in pair admission.
            node.replace_all_uses_with(causal_snapshots[causal_key])
            gm.graph.erase_node(node)
            if anchor is node:
                anchor = next(n for n in gm.graph.nodes if n.op != "placeholder")
            continue
        frozen = value.detach().clone()
        with gm.graph.inserting_before(anchor):
            placeholder = gm.graph.placeholder(f"captured_constant_{len(values)}")
        placeholder.meta = dict(node.meta)
        # _value must describe the real frozen storage rather than a view into
        # a closure object's previous allocation.
        placeholder.meta["val"] = frozen
        if causal_key is not None:
            placeholder.meta["latch_causal_mask"] = True
            causal_snapshots[causal_key] = placeholder
        node.replace_all_uses_with(placeholder)
        gm.graph.erase_node(node)
        attributes.append(str(node.target))
        values.append(frozen)
        bindings[placeholder.name] = frozen
        if anchor is node:
            anchor = next(n for n in gm.graph.nodes if n.op != "placeholder")
    gm.graph.lint()
    gm.recompile()
    return tuple(attributes), tuple(values), bindings


# LATCH-STEP-11: torch.cond frontend. PyTorch 2.9.1 cannot trace torch.cond
# under torch.func.grad_and_value + make_fx (ops/test_regions.py probe), so the
# normal capture intercepts the call and specializes the complete update once
# per outcome. The existing uniform region then executes only the selected
# specialization on device. This is not a general higher-order-op importer.
_COND_STATE = contextvars.ContextVar("latch_cond_state", default=None)
_COND_LOCK = threading.RLock()
_COND_PATCH = {"depth": 0, "original": None, "owners": ()}
COND_GRADIENT_PARTICIPATION = "union"


class _ConditionalFound(Exception):
    """Internal stop signal: the first traced torch.cond call and its predicate."""

    def __init__(self, predicate, index):
        super().__init__("torch.cond found")
        self.predicate, self.index = predicate, index


class _PredicateMarker:
    """Stands in for the runtime predicate while one outcome is specialized.

    Any read other than passing it directly as the torch.cond predicate would be
    evaluated at a fixed specialization and silently change meaning; refuse it.
    """

    def _refuse(self, *args, **kwargs):
        raise CaptureError("COND_PREDICATE_USE", "the torch.cond predicate input may only be passed unchanged as "
                           "the torch.cond predicate; other reads of it are outside the bounded conditional form")

    __bool__ = __index__ = __int__ = __float__ = __len__ = __iter__ = __getitem__ = _refuse

    def __getattr__(self, name):
        self._refuse()

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        raise CaptureError("COND_PREDICATE_USE", f"{getattr(func, '__name__', func)} reads the torch.cond predicate "
                           "input; it may only be passed unchanged as the torch.cond predicate")


class _CondDetect:
    """First-trace detector: map the predicate to an explicit data input by identity."""

    def __init__(self):
        self.arguments = None

    def call(self, pred, true_fn, false_fn, operands, extra, kwargs):
        index = None
        if isinstance(pred, torch.Tensor) and self.arguments is not None:
            index = next((i for i, value in enumerate(self.arguments) if value is pred), None)
        raise _ConditionalFound(pred, index)


class _CondSelect:
    """Trace one outcome; record each call's output signature for branch matching."""

    def __init__(self, outcome, marker):
        self.outcome, self.marker, self.traces, self.depth = outcome, marker, [], 0

    def call(self, pred, true_fn, false_fn, operands, extra, kwargs):
        if extra or kwargs:
            raise CaptureError("COND_FORM", "only torch.cond(pred, true_fn, false_fn, operands) is supported")
        if self.depth:
            raise CaptureError("COND_NESTED", "nested torch.cond is outside the bounded conditional form")
        if pred is not self.marker:
            raise CaptureError("COND_PREDICATE", "every torch.cond in one update must use the same explicit predicate "
                               "input; computed, Python, parameter, buffer or second predicates are unsupported")
        if not callable(true_fn) or not callable(false_fn):
            raise CaptureError("COND_FORM", "torch.cond branches must be callables")
        if not isinstance(operands, (tuple, list)) or not all(isinstance(x, torch.Tensor) for x in operands):
            raise CaptureError("COND_OPERANDS", "torch.cond operands must be a tuple/list of tensors")
        from torch.utils._pytree import tree_flatten
        self.depth += 1
        try:
            result = (true_fn if self.outcome else false_fn)(*operands)
            other = self._other_signature(false_fn if self.outcome else true_fn, operands)
        finally:
            self.depth -= 1
        leaves, spec = tree_flatten(result)
        if not leaves or not all(isinstance(x, torch.Tensor) for x in leaves):
            raise CaptureError("COND_OUTPUT", "torch.cond branches must return a nonempty tensor or pytree of tensors")
        signature = {"structure": str(spec), "inputs": [[list(x.shape), str(x.dtype)] for x in operands],
                     "outputs": [[list(x.shape), str(x.dtype)] for x in leaves]}
        if other is not None and (other["structure"], other["outputs"]) != (signature["structure"],
                                                                            signature["outputs"]):
            raise CaptureError("COND_BRANCH_MISMATCH", "torch.cond true/false branch outputs differ in structure, "
                               f"shape or dtype: {signature['structure']} {signature['outputs']} vs "
                               f"{other['structure']} {other['outputs']}")
        self.traces[-1].append(signature)
        return result

    @staticmethod
    def _other_signature(fn, operands):
        """Output metadata of the unselected branch, evaluated on fake tensors only.

        Only during make_fx fake tracing; proxy recording is disabled, so nothing
        enters the captured graph. Real-tensor passes (participation discovery)
        skip it; the post-capture call-by-call comparison still applies.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, get_proxy_mode
        from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
        from torch.utils._pytree import tree_flatten
        if get_proxy_mode() is None or not any(isinstance(m, FakeTensorMode)
                                               for m in _get_current_dispatch_mode_stack()):
            return None
        try:
            with disable_proxy_modes_tracing(), torch.no_grad():
                value = fn(*operands)
        except CaptureError:
            raise
        except Exception:
            return None  # Signature unavailable here; the per-outcome comparison remains.
        leaves, spec = tree_flatten(value)
        if not leaves or not all(isinstance(x, torch.Tensor) for x in leaves):
            raise CaptureError("COND_OUTPUT", "torch.cond branches must return a nonempty tensor or pytree of tensors")
        return {"structure": str(spec), "outputs": [[list(x.shape), str(x.dtype)] for x in leaves]}


def _intercepted_cond(pred, true_fn, false_fn, operands=(), *extra, **kwargs):
    state = _COND_STATE.get()
    if state is None:
        return _COND_PATCH["original"](pred, true_fn, false_fn, operands, *extra, **kwargs)
    return state.call(pred, true_fn, false_fn, operands, extra, kwargs)


@contextmanager
def _cond_frontend():
    """Route torch.cond through the capture-time state for this context only."""
    with _COND_LOCK:
        if not _COND_PATCH["depth"]:
            original = torch.cond
            owners = [torch]
            for module_name in ("torch._higher_order_ops", "torch._higher_order_ops.cond"):
                module = sys.modules.get(module_name)
                if module is not None and getattr(module, "cond", None) is original:
                    owners.append(module)
            for owner in owners:
                owner.cond = _intercepted_cond
            _COND_PATCH.update(original=original, owners=tuple(owners))
        _COND_PATCH["depth"] += 1
    try:
        yield
    finally:
        with _COND_LOCK:
            _COND_PATCH["depth"] -= 1
            if not _COND_PATCH["depth"]:
                for owner in _COND_PATCH["owners"]:
                    owner.cond = _COND_PATCH["original"]
                _COND_PATCH.update(original=None, owners=())


def _import_graph(gm, names, num_inputs, aliases, config, state_names=None, *, optimizer,
                  gradient_participation=None, constant_inputs=()):
    values, operations, placeholders, constants = {}, [], [], {}
    ns = len(state_names or ())
    no = len(optimizer.state_names)
    out_refs = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            index = len(placeholders)
            n = len(names)
            role = ("parameter" if index < n else
                    ("momentum" if optimizer.legacy_momentum else "optimizer_state") if index < n + no else
                    "state" if index < n + no + ns else
                    "input" if index < n + no + ns + num_inputs else "constant_input")
            parameter = (names[index] if index < n else optimizer.state_names[index - n] if index < n + no else
                         state_names[index - n - no] if index < n + no + ns else None)
            values[node.name] = _value(node, role=role, parameter=parameter)
            placeholders.append(node.name)
        elif node.op == "get_attr":
            value = gm
            for part in node.target.split("."):
                value = getattr(value, part)
            if not isinstance(value, torch.Tensor):
                raise CaptureError("SUBGRAPH", f"nested graph/object {node.target} requires a region lowering")
            values[node.name] = _value(node, role="constant")
            constants[node.name] = value.detach().clone()
        elif node.op == "call_function":
            target = str(node.target)
            schema = getattr(node.target, "_schema", None)
            if schema is None and node.name in values and values[node.name].producer is not None:
                continue
            if schema is None:
                raise CaptureError("HIGHER_ORDER", f"{target} requires an explicit operation/region lowering")
            if isinstance(node.meta.get("val"), (tuple, list)):
                if target not in MULTI_RESULT_TARGETS:
                    raise CaptureError("MULTI_RESULT", f"{target} returns several tensors and has no operation rule")
                _reject_effects(node)
                args, kwargs = _encode(node.args), _encode(node.kwargs)
                reads = tuple(dict.fromkeys(_refs((args, kwargs))))
                dependencies = tuple(dict.fromkeys(values[x].producer for x in reads if values[x].producer is not None))
                results = _multi_result_names(node)
                for result, tensor in zip(results, node.meta["val"]):
                    values[result] = _tensor_value(result, tensor, producer=node.name)
                operations.append(Operation(node.name, target, args, kwargs, reads, tuple(results), dependencies,
                                            reads, tuple(results)))
                continue
            _reject_effects(node)
            args, kwargs = _encode(node.args), _encode(node.kwargs)
            reads = tuple(dict.fromkeys(_refs((args, kwargs))))
            dependencies = tuple(dict.fromkeys(values[x].producer for x in reads if values[x].producer is not None))
            alias = reads[0] if target in _ALIASES and reads else None
            values[node.name] = _value(node, producer=node.name, alias_of=alias)
            axes = ()
            if target.startswith(("aten.sum.", "aten.mean.", "aten.amax.")):
                dims = args[1] if len(args) > 1 and args[1] is not None else tuple(range(len(values[reads[0]].shape)))
                axes = (dims,) if isinstance(dims, int) else tuple(dims)
            operations.append(Operation(node.name, target, args, kwargs, reads, (node.name,), dependencies,
                                        reads, () if alias is not None else (node.name,), reduction_axes=axes,
                                        synchronization="view" if alias is not None else "global_phase_boundary",
                                        effect="rng_state_transition" if target == "latch_ops.philox_advance.default"
                                        else "pure"))
        elif node.op == "output":
            out_refs = _encode(node.args[0])
        else:
            raise CaptureError("NODE", f"unsupported FX node kind {node.op}: {node}")
    if len(placeholders) != len(names) + no + ns + num_inputs + len(constant_inputs):
        raise CaptureError("SIGNATURE", "captured placeholders do not match the explicit state/input signature")
    if not isinstance(out_refs, (tuple, list)) or len(out_refs) != 1 + 2 * len(names) + no + ns:
        raise CaptureError("SIGNATURE", "training graph output signature changed")
    if not all(isinstance(ref, Ref) for ref in out_refs):
        raise CaptureError("SIGNATURE", "every training output must be a Tensor value")
    n = len(names)
    outputs = {"loss": out_refs[0].name,
               "grads": dict(zip(names, (r.name for r in out_refs[1:1+n]))),
               "params": dict(zip(names, (r.name for r in out_refs[1+n:1+2*n])))}
    optimizer_outputs = dict(zip(optimizer.state_names, (r.name for r in out_refs[1+2*n:1+2*n+no])))
    outputs["optimizer_state"] = optimizer_outputs
    if optimizer.legacy_momentum:
        outputs["momentum"] = optimizer_outputs
    if state_names is not None:
        outputs["state"] = dict(zip(state_names, (r.name for r in out_refs[1+2*n+no:])))
    roles = [(outputs["loss"], "loss")]
    for kind in ("grads", "params", "momentum" if optimizer.legacy_momentum else "optimizer_state"):
        roles.extend((value, f"{kind}:{name}") for name, value in outputs[kind].items())
    for name, value in outputs.get("state", {}).items():
        roles.append((value, f"state:{name}"))
    for value, role in roles:
        values[value].output_roles += (role,)
    metadata = {
        "torch_version": torch.__version__, "capture": "torch.func.grad_and_value + make_fx(fake)",
        "parameter_names": list(names), "parameter_aliases": aliases,
        **optimizer.metadata(),
        "control_flow": "Python static loops/branches specialize at capture; no runtime regions admitted",
        "randomness": "refused until explicit state transition lowering",
        "state_commit": "all new parameter/optimizer/model-state outputs committed after all graph reads",
        "numerical_policy": "FP32 input/state; lowering defines and records matrix accumulation policy",
    }
    if gradient_participation is not None:
        metadata["gradient_participation"] = dict(zip(names, gradient_participation))
        metadata["gradient_participation_scope"] = "ordinary autograd None-v-zero discovery for the captured static loss path"
    if constant_inputs:
        metadata["constant_input_names"] = list(placeholders[-len(constant_inputs):])
        metadata["constant_input_attributes"] = list(constant_inputs)
    metadata["causal_mask_constants"] = [node.name for node in gm.graph.nodes
        if node.op == "placeholder" and node.meta.get("latch_causal_mask")]
    metadata["explicit_input_indices"] = list(range(n + no + ns, n + no + ns + num_inputs))
    if state_names is not None:
        metadata["state_names"] = list(state_names)
        metadata["state_derivatives"] = "parameter derivatives of loss only; returned state is auxiliary output"
        metadata["numerical_policy"] = "FP32 floating computation plus explicit typed persistent state"
    if any(op.target.startswith("latch_ops.philox_") for op in operations):
        metadata["randomness"] = "pure explicit Philox state reads and counter transitions; no global generator"
    return TrainingIR(values, operations, tuple(placeholders), outputs, metadata), constants


def capture_training(loss_fn, params: dict[str, torch.Tensor],
                     momentum: dict[str, torch.Tensor] | None,
                     inputs: tuple[torch.Tensor, ...], config: SGDConfig | AdamWConfig | None = None,
                     *, state: dict[str, torch.Tensor] | None = None,
                     optimizer_state: dict[str, torch.Tensor] | None = None,
                     precision=None, conditional="auto", _participation_floor=None, _cond_inputs=None,
                     gradient_transform=None, backward_autocast=None, source_locations=False):
    """Capture one pure update; expensive capture/reference work belongs on Modal CPU.

    ``loss_fn(params, *inputs)`` returns a scalar FP32 Tensor. Parameters appearing
    under several dictionary names are canonicalized by identity, never cloned into
    independent differentiable inputs. The callable's Python structure is static.
    With ``state`` supplied, ``loss_fn(params, state, *inputs)`` returns
    ``(loss, new_state_dict)``. Parameter gradients come only from the loss.
    The flat ABI is params, optimizer state, model state, inputs, readonly captured
    constants; outputs are loss, grads, new params, new optimizer state, new model
    state. SGD keeps its existing momentum layout. With hoisted constants,
    ``step_fn`` evaluates captured arithmetic so its bindings match the graph;
    independent qualification must use ordinary autograd and torch.optim.
    AdamWConfig(runtime_lr=True) appends its scalar FP32 rate after explicit
    data and before hoisted constants; optimizer_lr_input records its index.
    precision names an explicit forward arithmetic policy applied before AD;
    None leaves the user program unchanged, including all pre-existing casts.
    gradient_transform(grads, canonical_names), when supplied, is captured once
    after AD. Its shape/dtype-preserving outputs are both the public gradients
    and the optimizer inputs (e.g. a distributed SUM before global clipping).
    backward_autocast="bf16" captures AD under the example device's BF16
    autocast dispatcher, with autocast disabled for the forward callable,
    gradient transform and optimizer. Explicit forward casts remain unchanged.
    CPU examples use CPU dispatch, not an emulation of every CUDA autocast rule;
    the imported typed operations define the generated program. None preserves
    the existing caller context. This option currently requires a flat graph.

    LATCH-STEP-11: when the traced update calls ``torch.cond(pred, true_fn,
    false_fn, operands)`` and ``pred`` is one of the explicit ``inputs`` (one
    contiguous torch.bool element), the default ``conditional="auto"`` returns a
    CapturedConditional instead: both outcomes are specialized as complete
    differentiated updates and composed by the existing uniform region, so only
    the selected outcome executes. Its flat ABI removes the predicate from the
    data inputs and appends it last; ``ir.metadata["cond_frontend"]`` records
    the original input index. ``conditional="refuse"`` rejects torch.cond.
    Programs without torch.cond are captured exactly as before.
    """
    if conditional not in ("auto", "refuse"):
        raise CaptureError("COND_SCOPE", "conditional must be 'auto' or 'refuse'")
    if backward_autocast not in (None, "bf16"):
        raise CaptureError("BACKWARD_AUTOCAST", "backward_autocast must be None or 'bf16'")
    from .precision import PolicyRecord, resolve, wrap
    policy = resolve(precision)
    policy_records = []
    config = config or SGDConfig()
    config.validate()
    if hasattr(config, "schedule_input"):
        raise CaptureError("OPTIMIZER_ABI", "runtime-scheduled Muon is supported by capture_loop_training; "
                           "pass its schedule as an explicit loop data input")
    if torch.__version__.split("+")[0] != "2.9.1":
        raise CaptureError("TORCH_VERSION", f"this capture integration requires torch 2.9.1; got {torch.__version__}")
    if not params:
        raise CaptureError("PARAMETERS", "at least one trainable parameter is required")
    unique, aliases, ids, storages = {}, {}, {}, {}
    for name, p in params.items():
        if not isinstance(p, torch.Tensor) or p.dtype != torch.float32 or p.layout != torch.strided:
            raise CaptureError("PARAMETER", f"{name}: expected a strided FP32 Tensor")
        _reject_lazy_view(p, f"parameter {name}")
        canonical = ids.setdefault(id(p), name)
        aliases[name] = canonical
        unique.setdefault(canonical, p)
        if p.numel():
            storage = (p.device, p.untyped_storage().data_ptr())
            other = storages.setdefault(storage, canonical)
            if other != canonical:
                raise CaptureError("PARAMETER_STORAGE", f"{name} and {other} are distinct tensors sharing storage; "
                                   "use a single Tensor object for tied parameters")
    names = tuple(unique)
    name_to_index = {name: i for i, name in enumerate(names)}
    pvalues = tuple(p.detach() for p in unique.values())
    optimizer = prepare_optimizer(config, unique, optimizer_state=optimizer_state, momentum=momentum, aliases=aliases)
    ovalues = optimizer.state_values
    no = len(ovalues)
    inputs = tuple(inputs)
    data_inputs = inputs
    if optimizer.runtime_lr:
        inputs = (*inputs, torch.tensor(float(config.lr), dtype=torch.float32, device=pvalues[0].device))
    for i, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("INPUT", f"input {i} must be an explicit Tensor")
        _reject_lazy_view(value, f"input {i}")
    n = len(names)
    state_names = tuple(state) if state is not None else ()
    state_values = []
    for name in state_names:
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise CaptureError("STATE_TENSOR", f"state {name} must be a strided Tensor")
        _reject_lazy_view(value, f"state {name}")
        state_values.append(value.detach())
    ns = len(state_names)
    autocast_device = None
    if backward_autocast is not None:
        devices = {value.device for value in (*pvalues, *ovalues, *state_values, *inputs)}
        if len(devices) != 1 or pvalues[0].device.type not in ("cpu", "cuda"):
            raise CaptureError("BACKWARD_AUTOCAST", "BF16 backward autocast requires one CPU or CUDA device")
        autocast_device = pvalues[0].device.type

    def policy_loss(*args):
        policy_records.append(PolicyRecord())
        return wrap(loss_fn, policy, policy_records[-1])(*args)

    forward_loss = loss_fn if policy is None else policy_loss
    if backward_autocast is not None:
        original_forward_loss = forward_loss
        def forward_loss(mapping, *args):
            # This context ends before grad_and_value executes the VJP. It must
            # not encompass that transform or turn off its ambient autocast.
            device_type = next(iter(mapping.values())).device.type
            with torch.autocast(device_type, enabled=False):
                return original_forward_loss(mapping, *args)

    def scalar_loss(pflat, *args):
        mapping = {alias: pflat[name_to_index[canonical]] for alias, canonical in aliases.items()}
        new_state = ()
        if state is not None:
            state_in = dict(zip(state_names, args[:ns]))
            cond_state = _COND_STATE.get()
            if isinstance(cond_state, _CondDetect):
                cond_state.arguments = args[ns:]
            result = forward_loss(mapping, state_in, *args[ns:])
            if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict):
                raise CaptureError("STATE_RETURN", "stateful loss must return (scalar_loss, new_state_dict)")
            loss, state_out = result
            if set(state_out) != set(state_names):
                raise CaptureError("STATE_RETURN", "returned state keys must exactly match input state keys")
            for name in state_names:
                before, after = state_in[name], state_out[name]
                if (not isinstance(after, torch.Tensor) or after.shape != before.shape or after.dtype != before.dtype
                        or after.device != before.device):
                    raise CaptureError("STATE_RETURN", f"returned state {name} must preserve shape, dtype and device")
            # Aux state may intentionally remain unchanged or be a view of old
            # state/parameters. Materialize its new version before the final
            # mutable-state commit; do not expose an old-input alias as output.
            new_state = tuple(state_out[name].clone() for name in state_names)
        else:
            cond_state = _COND_STATE.get()
            if isinstance(cond_state, _CondDetect):
                cond_state.arguments = args
            loss = forward_loss(mapping, *args)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or loss.dtype != torch.float32:
            raise CaptureError("LOSS", "loss_fn must return a scalar FP32 Tensor")
        return (loss, new_state) if state is not None else loss

    differentiated = grad_and_value(scalar_loss, has_aux=state is not None)
    participation = (True,) * n

    def update_step(*flat):
        pflat, oflat, args = tuple(flat[:n]), tuple(flat[n:n+no]), flat[n+no:]
        lr = None
        if optimizer.runtime_lr:
            args, lr = args[:-1], args[-1]
        with (torch.autocast(pflat[0].device.type, dtype=torch.bfloat16, cache_enabled=False)
              if backward_autocast is not None else nullcontext()):
            grads, primal = differentiated(pflat, *args)
        loss, new_state = primal if state is not None else (primal, ())
        if gradient_transform is not None:
            transformed = tuple(gradient_transform(grads, names))
            if len(transformed) != len(grads) or any(
                    not isinstance(g, torch.Tensor) or g.shape != old.shape or g.dtype != old.dtype or g.device != old.device
                    for g, old in zip(transformed, grads)):
                raise CaptureError("GRADIENT_TRANSFORM", "gradient transform must preserve gradient arity, shapes, dtypes and devices")
            grads = transformed
        update_options = {}
        if getattr(optimizer.config, "forward_state", False):
            if state is None:
                raise CaptureError("OPTIMIZER_ABI", "forward_state optimizer requires a stateful loss")
            update_options["forward_state"] = dict(zip(state_names, new_state))
        new_params, new_optimizer = optimizer.update(pflat, grads, oflat, participation, lr=lr, **update_options)
        return (loss, *grads, *new_params, *new_optimizer, *new_state)

    def step(*flat):
        # Also isolate optimizer/gradient-transform arithmetic when the caller
        # already has autocast enabled. The default retains the prior behavior.
        with (torch.autocast(flat[0].device.type, enabled=False)
              if backward_autocast is not None else nullcontext()):
            return update_step(*flat)

    example_args = (*pvalues, *ovalues, *state_values, *inputs)
    labels = ([f"parameter {name}" for name in names] +
              [f"optimizer state {name}" for name in optimizer.state_names] +
              [f"state {name}" for name in state_names] +
              [f"input {i}" for i in range(len(data_inputs))] +
              (["optimizer learning rate"] if optimizer.runtime_lr else []))
    _reject_placeholder_storage_alias(example_args, labels)
    detect = _CondDetect() if _COND_STATE.get() is None else None
    try:
        with _cond_frontend():
            token = _COND_STATE.set(detect) if detect is not None else None
            try:
                with capture_functions(), _source_location_scope(source_locations):
                    gm = make_fx(step, decomposition_table=capture_decompositions(), tracing_mode="fake",
                                 _allow_non_fake_inputs=True, _error_on_data_dependent_ops=True,
                                 **_source_location_kwargs(source_locations))(*example_args)
            finally:
                if token is not None:
                    _COND_STATE.reset(token)
    except _ConditionalFound as found:
        if backward_autocast is not None:
            raise CaptureError("BACKWARD_AUTOCAST", "backward autocast currently requires a flat training graph") from None
        if gradient_transform is not None:
            raise CaptureError("GRADIENT_TRANSFORM", "gradient transforms currently require a flat training graph") from None
        if conditional != "auto":
            raise CaptureError("COND_SCOPE", "torch.cond is refused by this capture (conditional='refuse')") from None
        if found.index is None or found.index >= len(data_inputs):
            raise CaptureError("COND_PREDICATE", "torch.cond predicate must be one explicit input tensor passed "
                               "unchanged; computed, Python, parameter or buffer predicates are unsupported") from None
        if _cond_inputs is not None and found.index >= _cond_inputs:
            raise CaptureError("COND_PREDICATE", "a registered buffer is a frozen capture snapshot; pass the "
                               "torch.cond predicate as an explicit forward input or target") from None
        return _capture_cond_training(loss_fn, params, momentum, data_inputs, found.index, config, state=state,
                                      optimizer_state=optimizer_state, precision=precision)
    except CaptureError:
        raise
    except Exception as exc:
        if any(marker in str(exc).lower() for marker in ("data-dependent", "data dependent", "_local_scalar_dense")):
            raise CaptureError("RUNTIME_CONTROL", "tensor-dependent Python control/scalar extraction requires "
                               f"an explicit region capture: {type(exc).__name__}: {exc}") from exc
        raise CaptureError("TRACE", f"full update capture failed: {type(exc).__name__}: {exc}") from exc
    # Caller-visible state effects remain meaningful even when their result is dead.
    gm = _functionalize_internal_mutations(gm, example_args, source_locations=source_locations)
    if optimizer.uses_participation:
        with _cond_frontend(), capture_functions():
            participation = discover_gradient_participation(scalar_loss, pvalues, (*state_values, *data_inputs),
                                                            has_aux=state is not None)
        if _participation_floor is not None:
            # STEP-11 torch.cond semantics: a parameter used by either outcome
            # receives a materialized zero gradient when the other is selected.
            participation = tuple(bool(a) or bool(b) for a, b in zip(participation, _participation_floor))
        if not all(participation):
            with _cond_frontend(), capture_functions(), _source_location_scope(source_locations):
                gm = make_fx(step, decomposition_table=capture_decompositions(), tracing_mode="fake", _allow_non_fake_inputs=True,
                             _error_on_data_dependent_ops=True, **_source_location_kwargs(source_locations))(*example_args)
            gm = _functionalize_internal_mutations(gm, example_args, source_locations=source_locations)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    round_trips, layout_copies = 0, []
    from .channels_last import propagate as _channels_last
    if policy is not None:
        from .precision import cancel_round_trips
        from .layout_normalize import materialize_reshape_copies
        round_trips = cancel_round_trips(gm)
        channels_last = _channels_last(gm)      # LATCH-STEP10-SPEED: pixel-major convolution neighbourhoods
        layout_copies = materialize_reshape_copies(gm)
    else:
        channels_last = _channels_last(gm)
    # LATCH-ATTN-RECOG: packed attention gradients are cast per operand before their assembly (exact; no-op
    # without the pattern), so the attention bodies write each BF16 gradient directly.
    from .attention_canonical import canonicalize_attention
    attention_canonical = canonicalize_attention(gm)
    # Step 05: equivalent product spellings reach one canonical operation form (no-op without the pattern).
    from .layout_canonical import canonicalize_layouts
    layout_canonical = canonicalize_layouts(gm)
    constant_attributes, constant_values, constants = hoist_tensor_constants(gm)
    example_args = (*example_args, *constant_values)
    ir, unhoisted = _import_graph(gm, names, len(inputs), aliases, config, state_names if state is not None else None,
                                  optimizer=optimizer, gradient_participation=participation,
                                  constant_inputs=constant_attributes)
    if unhoisted:
        raise CaptureError("CONSTANT", "captured constants were not completely lifted")
    if not optimizer.uses_participation:
        ir.metadata["gradient_participation_scope"] = (
            "optimizer contract consumes materialized gradients, including zero for unused parameters; "
            "no eager None-versus-zero discovery is needed")
    if gradient_transform is not None:
        ir.metadata["gradient_transform"] = {
            "function": getattr(gradient_transform, "__qualname__", type(gradient_transform).__name__),
            "semantics": "applied once after AD; returned gradients and optimizer inputs are identical"}
        from .distributed import canonical_gradient_transform
        canonical_transform = canonical_gradient_transform(gradient_transform)
        if canonical_transform is not None:
            ir.metadata["gradient_transform"]["canonical_sum"] = canonical_transform
    if backward_autocast is not None:
        ir.metadata["backward_autocast"] = {
            "contract": "device_bf16_backward_v1", "dtype": "bfloat16",
            "capture_device_type": autocast_device, "forward": "disabled",
            "gradient_transform": "disabled", "optimizer": "disabled",
            "scope": "device-specific PyTorch autocast during AD; typed captured operations retained"}
    if policy is not None:
        ir.metadata["precision_policy"] = {**policy.describe(), **policy_records[-1].summary(),
                                           "exact_round_trips_removed": round_trips}
        ir.metadata["numerical_policy"] = (f"precision policy {policy.name}: covered products BF16 operands with FP32 "
                                           "accumulation and one BF16 result rounding; all other arithmetic as written (FP32)")
        if layout_copies:
            ir.metadata["layout_copies"] = layout_copies
    if layout_canonical:
        ir.metadata["layout_canonical"] = layout_canonical
    if attention_canonical:
        ir.metadata["attention_canonical"] = attention_canonical
    if channels_last:
        ir.metadata["channels_last"] = {k: v if isinstance(v, int) else len(v) for k, v in channels_last.items()}
    if optimizer.runtime_lr:
        ir.metadata["optimizer_lr_input"] = {"index": n + no + len(state_names) + len(data_inputs),
                                             "dtype": "float32", "shape": [],
                                             "semantics": "absolute AdamW learning rate staged by the host"}
    if constant_values:
        # The graph uses frozen/rebound constants; the original Python closure
        # could now contain different objects, so it is not a valid reference.
        step = gm
        ir.metadata["step_fn_semantics"] = "captured operation graph with explicit immutable constant bindings"
    if source_locations:
        # LATCH-EXPLAIN: metadata only (never read by lowering); maps captured operation names to the
        # nn.Module stack / user source frame FX recorded, when this torch recorded them.
        ir.metadata["source_locations"] = _collect_source_locations(gm)
    return CapturedTraining(ir, gm, step, tuple(example_args), names, aliases, config, constants, state_names)


# ---- LATCH-EXPLAIN optional provenance -------------------------------------------------------------------
def _source_location_kwargs(enabled):
    """make_fx(record_module_stack=True, record_stack_traces=True) where this torch supports them."""
    if not enabled:
        return {}
    import inspect
    try:
        parameters = inspect.signature(make_fx).parameters
    except (TypeError, ValueError):
        return {}
    return {flag: True for flag in ("record_module_stack", "record_stack_traces") if flag in parameters}


def source_location_support():
    """Which recording mechanisms this torch offers (plain data, for reports and diagnostics)."""
    import inspect
    try:
        parameters = sorted(inspect.signature(make_fx).parameters)
    except (TypeError, ValueError):
        parameters = []
    try:
        from torch.fx.traceback import preserve_node_meta  # noqa: F401
        preserve = True
    except ImportError:
        preserve = False
    return {"torch": torch.__version__, "make_fx_parameters": parameters, "preserve_node_meta": preserve}


@contextmanager
def module_path_recording(model, enabled):
    """While tracing ``model``'s forward, publish the running nn.Module path as FX ``nn_module_stack`` metadata.

    Opt-in only. Forward pre/post hooks maintain a path stack and mirror it into
    ``torch.fx.traceback``'s current preserved metadata, which the tracer copies onto every node it
    creates (the same mechanism AOT/export use). Hooks perform no tensor work and are removed afterwards.
    """
    if not enabled or model is None:
        yield
        return
    try:
        from torch.fx import traceback as fx_traceback
    except ImportError:
        yield
        return
    stack = []
    handles = []

    def publish():
        meta = getattr(fx_traceback, "current_meta", None)
        if isinstance(meta, dict):
            if stack:
                meta["nn_module_stack"] = {path: (path, kind) for path, kind in stack}
            else:
                meta.pop("nn_module_stack", None)

    def pre(path, kind):
        def hook(module, inputs):
            stack.append((path, kind))
            publish()
        return hook

    def post(module, inputs, output):
        if stack:
            stack.pop()
        publish()

    for path, module in model.named_modules():
        kind = type(module).__module__ + "." + type(module).__qualname__
        handles.append(module.register_forward_pre_hook(pre(path or "<root>", kind)))
        handles.append(module.register_forward_hook(post))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        meta = getattr(fx_traceback, "current_meta", None)
        if isinstance(meta, dict):
            meta.pop("nn_module_stack", None)


@contextmanager
def _source_location_scope(enabled):
    """Preserve FX node metadata (stack traces) during tracing when requested; a no-op otherwise."""
    if not enabled:
        yield
        return
    try:
        from torch.fx.traceback import preserve_node_meta
    except ImportError:
        yield
        return
    with preserve_node_meta():
        yield


def _collect_source_locations(gm):
    """{operation name: {module_stack: [...], source: 'file:line'}} from FX node metadata; plain data only."""
    located = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        entry = {}
        stack = node.meta.get("nn_module_stack")
        if stack:
            modules = []
            for key, value in stack.items():
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    path, kind = value
                    if not isinstance(kind, str):
                        kind = getattr(kind, "__module__", "") + "." + getattr(kind, "__qualname__", str(kind))
                    modules.append({"path": str(path), "type": str(kind).split("'")[1] if "'" in str(kind) else str(kind)})
                else:
                    modules.append({"path": str(key), "type": str(value)})
            entry["module_stack"] = modules
        trace = node.meta.get("stack_trace")
        if trace:
            frames = [line.strip() for line in str(trace).splitlines() if line.strip().startswith("File ")]
            user = [f for f in frames if "/torch/" not in f and "/ops/" not in f]
            entry["source"] = (user[-1] if user else frames[-1]) if frames else str(trace)[-200:]
        if entry:
            located[node.name] = entry
    return located


@dataclass
class CapturedConditional:
    ir: UniformConditionalIR
    true_branch: CapturedTraining
    false_branch: CapturedTraining
    example_args: tuple[torch.Tensor, ...]
    frontend: dict | None = None

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self.true_branch.parameter_names

    @property
    def parameter_aliases(self) -> dict[str, str]:
        return self.true_branch.parameter_aliases

    @property
    def constants(self) -> dict[str, torch.Tensor]:
        # Whole-training conditionals refuse hoisted constants (REGION_CONSTANT).
        return {}

    @property
    def state_names(self) -> tuple[str, ...]:
        return self.true_branch.state_names

    @property
    def optimizer_state_names(self) -> tuple[str, ...]:
        return self.true_branch.optimizer_state_names

    @property
    def config(self) -> SGDConfig | AdamWConfig:
        return self.true_branch.config

    def step_fn(self, *flat):
        """Conventional reference selection only; native execution never calls this."""
        selected = self.true_branch if bool(flat[-1].item()) else self.false_branch
        return selected.step_fn(*flat[:-1])

    def unpack(self, outputs):
        return self.true_branch.unpack(outputs)


def capture_conditional_training(true_loss, false_loss, params, momentum, inputs, predicate,
                                 config: SGDConfig | AdamWConfig | None = None, *, state=None,
                                 optimizer_state=None, precision=None,
                                 participation="per_branch") -> CapturedConditional:
    """Capture a uniform runtime choice between two PyTorch-defined training regions.

    Both branches are differentiated using the same PyTorch transform. This
    explicit region API does not claim to import arbitrary tensor-dependent
    Python branches or nested torch.cond. Branch-local arithmetic/state effects
    execute only when that branch is selected; the predicate is nondifferentiable.

    participation="per_branch" (default, unchanged) lets each branch discover its
    own None-v-zero parameter participation. "union" gives a parameter used by
    either branch a zero gradient in the other, matching torch.cond autograd;
    the torch.cond frontend (capture_training) uses it.
    """
    if participation not in ("per_branch", "union"):
        raise CaptureError("REGION_ABI", "participation must be per_branch or union")
    if (not isinstance(predicate, torch.Tensor) or predicate.dtype != torch.bool
            or predicate.numel() != 1 or predicate.layout != torch.strided or not predicate.is_contiguous()):
        raise CaptureError("PREDICATE", "uniform conditional requires one contiguous torch.bool Tensor element")
    _reject_lazy_view(predicate, "conditional predicate")
    from .user_optimizer import TensorUpdateRule
    if isinstance(config, TensorUpdateRule):
        raise CaptureError("OPTIMIZER", "user tensor update rules are admitted for flat capture_training/capture_module only")
    true_capture = capture_training(true_loss, params, momentum, inputs, config, state=state,
                                    optimizer_state=optimizer_state, precision=precision, conditional="refuse")
    false_capture = capture_training(false_loss, params, momentum, inputs, config, state=state,
                                     optimizer_state=optimizer_state, precision=precision, conditional="refuse")
    floors = [capture.ir.metadata.get("gradient_participation") for capture in (true_capture, false_capture)]
    if participation == "union" and floors[0] is not None and floors[0] != floors[1]:
        union = tuple(bool(floors[0][name]) or bool(floors[1][name]) for name in true_capture.parameter_names)
        if tuple(floors[0][name] for name in true_capture.parameter_names) != union:
            true_capture = capture_training(true_loss, params, momentum, inputs, config, state=state,
                                            optimizer_state=optimizer_state, precision=precision,
                                            conditional="refuse", _participation_floor=union)
        if tuple(floors[1][name] for name in false_capture.parameter_names) != union:
            false_capture = capture_training(false_loss, params, momentum, inputs, config, state=state,
                                             optimizer_state=optimizer_state, precision=precision,
                                             conditional="refuse", _participation_floor=union)
    if true_capture.parameter_names != false_capture.parameter_names or true_capture.state_names != false_capture.state_names:
        raise CaptureError("REGION_ABI", "conditional branch parameter/state order differs")
    true_ir, false_ir = true_capture.ir, false_capture.ir
    if true_capture.constants or false_capture.constants:
        raise CaptureError("REGION_CONSTANT", "whole-training conditional constants must be explicit shared inputs")
    if true_capture.optimizer_state_names != false_capture.optimizer_state_names:
        raise CaptureError("REGION_ABI", "conditional optimizer-state order differs")
    if len(true_ir.inputs) != len(false_ir.inputs):
        raise CaptureError("REGION_ABI", "conditional branch input counts differ")
    for left, right in zip(true_ir.inputs, false_ir.inputs):
        a, b = true_ir.values[left], false_ir.values[right]
        if (a.shape, a.dtype, a.strides, a.role, a.parameter) != (b.shape, b.dtype, b.strides, b.role, b.parameter):
            raise CaptureError("REGION_ABI", f"conditional branch input metadata differs at {left}/{right}")
    true_outputs, false_outputs = true_ir.output_names(), false_ir.output_names()
    if len(true_outputs) != len(false_outputs):
        raise CaptureError("REGION_ABI", "conditional branch output counts differ")
    for left, right in zip(true_outputs, false_outputs):
        a, b = true_ir.values[left], false_ir.values[right]
        if (a.shape, a.dtype) != (b.shape, b.dtype):
            raise CaptureError("REGION_ABI", f"conditional branch output metadata differs at {left}/{right}")
    if any(value.device != predicate.device for value in true_capture.example_args):
        raise CaptureError("PREDICATE", "predicate and all explicit inputs must be on one device")
    name = "region_predicate"
    if name in true_ir.values or name in false_ir.values:
        raise CaptureError("REGION_ABI", "reserved predicate value name collides with branch value")
    pred_value = Value(name, tuple(predicate.shape), "bool", tuple(predicate.stride()), role="predicate")
    metadata = {"parameter_names": list(true_capture.parameter_names),
                "state_names": list(true_capture.state_names),
                "parameter_aliases": true_capture.parameter_aliases,
                "optimizer": true_ir.metadata["optimizer"], "torch_version": torch.__version__,
                **{key: value for key, value in true_ir.metadata.items() if key.startswith("optimizer_")},
                "optimizer_state_version": true_ir.metadata["optimizer_state_version"],
                "optimizer_state_names": list(true_capture.optimizer_state_names),
                "optimizer_state_schema": true_ir.metadata["optimizer_state_schema"],
                "optimizer_gradient_policy": true_ir.metadata["optimizer_gradient_policy"],
                "control_flow": "one device-uniform runtime conditional; only selected complete region executes",
                "capture": "explicit uniform region; PyTorch grad_and_value + make_fx per branch",
                "predicate_index": len(true_capture.example_args),
                "derivatives": "selected branch loss differentiated with respect to canonical parameters",
                "branch_input_binding": "positional; external branch inputs follow true_branch.inputs",
                "branch_output_binding": "positional; loss, ordered grads/params/optimizer-state, ordered model state"}
    if participation != "per_branch":
        metadata["gradient_participation_policy"] = participation
    region = UniformConditionalIR(true_ir, false_ir, pred_value, metadata)
    example_args = (*true_capture.example_args, predicate.detach())
    _reject_placeholder_storage_alias(example_args, (*region.inputs,))
    return CapturedConditional(region, true_capture, false_capture, example_args)


def _capture_cond_training(loss_fn, params, momentum, data_inputs, index, config, *, state,
                           optimizer_state, precision) -> CapturedConditional:
    """STEP-11: specialize one torch.cond predicate input into the uniform region.

    The same user callable is traced once per outcome; within each trace every
    torch.cond call runs only the corresponding branch callable. Arithmetic and
    parameters outside the conditional appear in both complete updates, so the
    PyTorch transform differentiates through the selected branch together with
    the surrounding computation. Branch signatures must match call by call.
    """
    predicate = data_inputs[index]
    if (not isinstance(predicate, torch.Tensor) or predicate.dtype != torch.bool or predicate.numel() != 1
            or predicate.layout != torch.strided or not predicate.is_contiguous()):
        raise CaptureError("COND_PREDICATE", "torch.cond predicate input must be one contiguous torch.bool element "
                           "shared by the whole batch; per-example or non-boolean predicates are unsupported")
    rest = (*data_inputs[:index], *data_inputs[index + 1:])
    marker = _PredicateMarker()
    selections = {outcome: _CondSelect(outcome, marker) for outcome in (True, False)}

    def specialize(outcome):
        selection = selections[outcome]

        def specialized(p, *args):
            # Also routes the Python reference (CapturedConditional.step_fn), which
            # re-enters this callable after capture has restored torch.cond.
            selection.traces.append([])
            with _cond_frontend():
                token = _COND_STATE.set(selection)
                try:
                    if state is not None:
                        return loss_fn(p, args[0], *args[1:1 + index], marker, *args[1 + index:])
                    return loss_fn(p, *args[:index], marker, *args[index:])
                finally:
                    _COND_STATE.reset(token)
        return specialized

    with _cond_frontend():
        captured = capture_conditional_training(specialize(True), specialize(False), params, momentum, rest,
                                                predicate, config, state=state, optimizer_state=optimizer_state,
                                                precision=precision, participation=COND_GRADIENT_PARTICIPATION)
    signatures = {}
    for outcome, selection in selections.items():
        traces = selection.traces
        if not traces or not traces[0] or any(trace != traces[0] for trace in traces):
            raise CaptureError("COND_FORM", "torch.cond calls must be reached identically on every trace of an outcome")
        signatures[outcome] = traces[0]
    if len(signatures[True]) != len(signatures[False]):
        raise CaptureError("COND_BRANCH_MISMATCH", "the two outcomes reach different numbers of torch.cond calls")
    for call, (a, b) in enumerate(zip(signatures[True], signatures[False])):
        if a != b:
            raise CaptureError("COND_BRANCH_MISMATCH", f"torch.cond call {call}: true/false branch outputs differ "
                               f"in structure, shape or dtype: {a['structure']} {a['outputs']} vs "
                               f"{b['structure']} {b['outputs']}")
    frontend = {"version": 1, "source": "torch.cond inside the ordinary capture",
                "predicate_input": index, "region_predicate_index": len(captured.example_args) - 1,
                "cond_calls": len(signatures[True]), "branch_signatures": signatures[True],
                "gradient_participation": COND_GRADIENT_PARTICIPATION,
                "abi": "data inputs without the predicate, then the predicate as the final region input"}
    captured.ir.metadata["cond_frontend"] = frontend
    captured.ir.metadata["capture"] = ("torch.cond frontend: complete update specialized per outcome by "
                                       "grad_and_value + make_fx; existing uniform region")
    captured.frontend = frontend
    return captured


def module_loss(model: torch.nn.Module, loss_fn: Callable):
    """Adapt ``loss_fn(model, *inputs)`` with identity-preserving tied parameters.

    Buffers are explicit state in the functional call but read-only in this first
    checkpoint. The generic capture rejects traced mutation and random operations.
    Parameters with requires_grad=False when this adapter is created are frozen
    (ops/frozen.py): their private snapshots are closed over like buffers, become
    hoisted read-only constant inputs at capture, and are omitted from the
    returned trainable dictionary. Changing trainability requires a new adapter.
    """
    class LossModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, *args):
            return loss_fn(self.model, *args)

    wrapper = LossModule()
    from .frozen import partition_parameters
    partition = partition_parameters(wrapper.named_parameters())
    parameters = partition.trainable
    frozen = partition.bound(partition.frozen)
    buffers = dict(wrapper.named_buffers())

    def functional_loss(params, *inputs):
        return functional_call(wrapper, (params, frozen, buffers), inputs, tie_weights=True, strict=True)

    return functional_loss, parameters


def probe():
    """Small remote CPU probe callable, no CUDA required."""
    torch.set_num_threads(1)
    torch.manual_seed(4701)
    params = {"shared": torch.randn(8, 8) * 0.05, "head": torch.randn(8, 5) * 0.05}
    x, target = torch.randn(4, 8), torch.randn(4, 5)

    def loss(p, x, target):
        h = x
        for _ in range(3):
            h = h + torch.tanh(h @ p["shared"])
        pred = h @ p["head"]
        return ((pred - target) ** 2).mean()

    result = capture_training(loss, params, None, (x, target))
    actual = result.graph_module(*result.example_args)
    expected = result.step_fn(*result.example_args)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    return {"torch": torch.__version__, "operations": sorted({op.target for op in result.ir.operations}),
            "num_operations": len(result.ir.operations), "num_values": len(result.ir.values),
            "parameter_names": result.parameter_names, "loss": float(actual[0]), "fx_exact": True,
            "graph": result.graph_module.code}


if __name__ == "__main__":
    import json
    print(json.dumps(probe(), indent=2))
