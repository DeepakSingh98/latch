"""Bounded differentiable loop regions, with PyTorch-derived local VJPs.

Capture contains no iteration-count unrolling. A native loop lowerer orchestrates
ordinary operation graphs. ``save_body_tape`` partitions a PyTorch joint primal/VJP
graph and saves all material forward dependencies needed by its backward region.
``save_boundaries`` saves every carried
value and integer-state boundary; the VJP still recomputes body-local activations.
``recompute_boundaries`` reconstructs boundaries from the initial values, trading
quadratic replay work for history storage. Neither mode means a fully saved body
tape. Both replay random operations from the exact original explicit state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import os
from typing import Any, Callable

import torch
from torch.func import grad_and_value, vjp
from torch.fx.experimental.proxy_tensor import make_fx

from .capture import (SGDConfig, _ALIASES, _encode, _functionalize_internal_mutations, _refs, _reject_effects,
                      _reject_lazy_view, _reject_placeholder_storage_alias, _value,
                      discover_gradient_participation, capture_decompositions,
                      MULTI_RESULT_TARGETS, _multi_result_names, _tensor_value)
from .ir import CaptureError, Operation, Ref, TrainingIR, Value, _contiguous_strides


REGION_NAMES = ("init", "body", "terminal_vjp", "body_vjp", "init_vjp", "optimizer")
# invariant_casts (P-OPS4-F) runs once per launch, before init; see _hoist_invariant_casts.
OPTIONAL_REGION_NAMES = ("body_saved", "body_vjp_saved", "invariant_casts", "init_saved", "init_vjp_saved")
# init_saved/init_vjp_saved (C-OPS4-A): optional saved-tape partition of the
# once-per-update init (+ sequence) region and its VJP; see init_storage.
INIT_STORAGE = ("recompute", "save_tape")
# Differentiable carry dtypes. A BF16 carry has BF16 cotangents (PyTorch VJP
# semantics); shared-parameter gradients still accumulate in FP32.
CARRY_DTYPES = (torch.float32, torch.bfloat16)
STORAGE_MODES = ("save_body_tape", "save_boundaries", "recompute_boundaries")
LOOP_ALIASES = _ALIASES | {"aten.reshape.default"}
# P7-D selected storage-rematerialization policy (rkcq), preserved as an explicit
# public memory choice rather than a model-specific capture preset.
LOW_MEMORY_TAPE_OPTIONS = {"flops_per_byte": 64, "keep_operands": True,
                           "min_bytes": 1048576, "cse": True, "operand_copies": True}


def _automatic_cast_reuse(*, storage_mode, conditional, sequenced, init_storage):
    """Only the already supported saved-body, plain-region ABI is automatic."""
    return (storage_mode == "save_body_tape" and not conditional and not sequenced
            and init_storage == "recompute")


def _resolve_memory_policy(memory_policy, *, storage_mode, tape_policy, tape_options,
                           cast_reuse, conditional, sequenced, init_storage):
    if memory_policy not in ("default", "low_memory"):
        raise CaptureError("LOOP_MEMORY", "memory_policy must be 'default' or 'low_memory'")
    if cast_reuse is not None and type(cast_reuse) is not bool:
        raise CaptureError("LOOP_MEMORY", "cast_reuse must be a bool when specified")
    if tape_options is not None and not isinstance(tape_options, dict):
        raise CaptureError("LOOP_MEMORY", "tape_options must be a dict when specified")
    if memory_policy == "default":
        automatic = _automatic_cast_reuse(storage_mode=storage_mode, conditional=conditional,
                                           sequenced=sequenced, init_storage=init_storage)
        return tape_policy, tape_options, automatic if cast_reuse is None else cast_reuse
    if (storage_mode != "save_body_tape" or conditional or sequenced or init_storage != "recompute"):
        raise CaptureError("LOOP_MEMORY", "low_memory requires a nonconditional saved-body loop without sequences "
                           "and with init_storage='recompute'")
    if tape_policy not in ("auto", "storage_remat") or cast_reuse is False:
        raise CaptureError("LOOP_MEMORY", "low_memory requires storage_remat and cast_reuse=True; explicit overrides conflict")
    if any(key not in LOW_MEMORY_TAPE_OPTIONS or LOW_MEMORY_TAPE_OPTIONS[key] != value
           for key, value in (tape_options or {}).items()):
        raise CaptureError("LOOP_MEMORY", "tape_options conflict with the selected low_memory policy")
    return "storage_remat", dict(LOW_MEMORY_TAPE_OPTIONS), True


@dataclass
class OperationIR:
    """Flat pure operation region; no parameter or state commits occur here."""
    values: dict[str, Value]
    operations: list[Operation]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    state_writes: dict[str, str] = field(default_factory=dict)

    def output_names(self):
        return self.outputs

    def to_dict(self):
        # Reuse the existing value/operation encoder; TrainingIR's ordered
        # training-output helper is deliberately not called for this flat graph.
        result = TrainingIR(self.values, self.operations, self.inputs, self.outputs, self.metadata).to_dict()
        result.update(format_version=3, kind="operation_region", state_writes=self.state_writes)
        return result

    @classmethod
    def from_dict(cls, data):
        if data.get("kind") == "nested_operation":
            from .nested_loop import NestedOperationIR
            return NestedOperationIR.from_dict(data)
        if data.get("kind") == "uniform_operation":
            return UniformOperationIR.from_dict(data)
        if data.get("format_version") != 3 or data.get("kind") != "operation_region":
            raise CaptureError("REGION_VERSION", "unknown operation-region format")
        if data.get("state_writes"):
            raise CaptureError("REGION_EFFECT", "loop operation regions cannot commit mutable state")
        graph = TrainingIR.from_dict({**data, "format_version": 1})
        return cls(graph.values, graph.operations, graph.inputs, tuple(graph.outputs), graph.metadata)


@dataclass(frozen=True)
class ConditionalBody:
    """Explicit whole-grid branch selected by one scalar bool state value.

    Branches have the ordinary loop-body signature and may update the selector.
    The selector has no derivative. Each reverse step reads the old boundary's
    selector, not the final selector or a host-specialized trace decision.
    """
    true_fn: Callable
    false_fn: Callable
    predicate_state: str

    def __call__(self, params, carry, state, *inputs):
        selected = self.true_fn if bool(state[self.predicate_state].item()) else self.false_fn
        return selected(params, carry, state, *inputs)


@dataclass
class UniformOperationIR:
    true_branch: OperationIR
    false_branch: OperationIR
    predicate: Value
    values: dict[str, Value]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    state_writes: dict[str, str] = field(default_factory=dict)

    def output_names(self):
        return self.outputs

    def to_dict(self):
        return {"format_version": 5, "kind": "uniform_operation",
                "true_branch": self.true_branch.to_dict(), "false_branch": self.false_branch.to_dict(),
                "predicate": self.predicate.name, "inputs": list(self.inputs), "outputs": list(self.outputs),
                "values": {key: asdict(value) for key, value in self.values.items()}, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != 5 or data.get("kind") != "uniform_operation":
            raise CaptureError("REGION_VERSION", "unknown uniform-operation format")
        values = {}
        for key, entry in data["values"].items():
            entry = dict(entry)
            for field_name in ("shape", "strides", "output_roles"):
                entry[field_name] = tuple(entry[field_name])
            values[key] = Value(**entry)
        return cls(OperationIR.from_dict(data["true_branch"]), OperationIR.from_dict(data["false_branch"]),
                   values[data["predicate"]], values, tuple(data["inputs"]), tuple(data["outputs"]), data["metadata"])


@dataclass
class SelectedRegion:
    true_branch: Any
    false_branch: Any
    predicate_index: int

    def __call__(self, *args):
        branch = self.true_branch if bool(args[self.predicate_index].item()) else self.false_branch
        return branch(*args)

    @property
    def code(self):
        return f"uniform input {self.predicate_index}\n{self.true_branch.code}\n{self.false_branch.code}"


def _uniform_region(label, true_ir, false_ir, true_module, false_module, predicate_index):
    if len(true_ir.inputs) != len(false_ir.inputs) or len(true_ir.outputs) != len(false_ir.outputs):
        raise CaptureError("LOOP_BRANCH", f"{label}: branch arity differs")
    values = {name: replace(true_ir.values[name]) for name in true_ir.inputs}
    outputs = []
    for index, (a, b) in enumerate(zip(true_ir.outputs, false_ir.outputs)):
        va, vb = true_ir.values[a], false_ir.values[b]
        if (va.shape, va.dtype) != (vb.shape, vb.dtype):
            raise CaptureError("LOOP_BRANCH", f"{label}: branch result {index} differs in shape/dtype")
        name = f"{label}_result_{index}"
        values[name] = replace(va, name=name, alias_of=None, producer=None, storage_offset=0,
                               strides=_contiguous_strides(va.shape), role="region_output")
        outputs.append(name)
    predicate = values[true_ir.inputs[predicate_index]]
    if predicate.dtype != "bool" or predicate.shape != ():
        raise CaptureError("LOOP_BRANCH", "nested selector must be a scalar bool state")
    graph = UniformOperationIR(true_ir, false_ir, predicate, values, true_ir.inputs, tuple(outputs),
                               {"name": label, "input_labels": true_ir.metadata["input_labels"],
                                "dispatch": "uniform old-boundary scalar bool; selected branch only"})
    return graph, SelectedRegion(true_module, false_module, predicate_index)


@dataclass
class BoundedLoopIR:
    init: OperationIR
    body: OperationIR
    terminal_vjp: OperationIR
    body_vjp: OperationIR
    init_vjp: OperationIR
    optimizer: OperationIR
    external_values: dict[str, Value]
    inputs: tuple[str, ...]
    outputs: dict[str, Any]
    count: Value
    max_steps: int
    storage_mode: str
    metadata: dict[str, Any] = field(default_factory=dict)
    body_saved: OperationIR | None = None
    body_vjp_saved: OperationIR | None = None
    invariant_casts: OperationIR | None = None
    init_saved: OperationIR | None = None
    init_vjp_saved: OperationIR | None = None

    @property
    def values(self):
        return self.external_values

    @property
    def regions(self):
        return {name: getattr(self, name) for name in (*REGION_NAMES, *OPTIONAL_REGION_NAMES)
                if getattr(self, name) is not None}

    def output_names(self):
        result = [self.outputs["loss"]]
        for group in ("grads", "params"):
            result.extend(self.outputs[group][name] for name in self.metadata["parameter_names"])
        group = "optimizer_state" if "optimizer_state" in self.outputs else "momentum"
        result.extend(self.outputs[group][name] for name in self.metadata.get("optimizer_state_names", self.metadata["parameter_names"]))
        for group in ("state", "carry"):
            result.extend(self.outputs[group][name] for name in self.metadata[f"{group}_names"])
        return (*result, self.outputs["executed_count"], self.outputs["status"])

    def to_dict(self):
        return {"format_version": 4, "kind": "bounded_loop", "regions": {key: graph.to_dict() for key, graph in self.regions.items()},
                "external_values": {key: asdict(value) for key, value in self.external_values.items()},
                "inputs": list(self.inputs), "outputs": self.outputs, "count": asdict(self.count),
                "max_steps": self.max_steps, "storage_mode": self.storage_mode, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != 4 or data.get("kind") != "bounded_loop":
            raise CaptureError("LOOP_VERSION", "unknown bounded-loop format")
        if data["storage_mode"] not in STORAGE_MODES or type(data["max_steps"]) is not int or not 0 <= data["max_steps"] <= 1024:
            raise CaptureError("LOOP_BOUND", "invalid saved loop policy")
        def value(entry):
            entry = dict(entry)
            for key in ("shape", "strides", "output_roles"):
                entry[key] = tuple(entry[key])
            return Value(**entry)
        return cls(**{key: OperationIR.from_dict(data["regions"][key]) for key in REGION_NAMES},
                   external_values={key: value(raw) for key, raw in data["external_values"].items()},
                   inputs=tuple(data["inputs"]), outputs=data["outputs"], count=value(data["count"]),
                   max_steps=data["max_steps"], storage_mode=data["storage_mode"], metadata=data["metadata"],
                   **{key: OperationIR.from_dict(data["regions"][key]) if key in data["regions"] else None
                      for key in OPTIONAL_REGION_NAMES})


@dataclass
class CapturedLoop:
    ir: BoundedLoopIR
    graph_modules: dict[str, torch.fx.GraphModule]
    example_args: tuple[torch.Tensor, ...]
    step_fn: Callable
    config: SGDConfig

    @property
    def parameter_names(self):
        return tuple(self.ir.metadata["parameter_names"])

    @property
    def carry_names(self):
        return tuple(self.ir.metadata["carry_names"])

    @property
    def state_names(self):
        return tuple(self.ir.metadata["state_names"])

    @property
    def optimizer_state_names(self):
        return tuple(self.ir.metadata.get("optimizer_state_names", self.parameter_names))

    def unpack(self, values):
        n, no, ns, nc = len(self.parameter_names), len(self.optimizer_state_names), len(self.state_names), len(self.carry_names)
        if len(values) != 1 + 2 * n + no + ns + nc + 2:
            raise ValueError("bounded-loop output arity differs from its ABI")
        return {"loss": values[0], "grads": dict(zip(self.parameter_names, values[1:1+n])),
                "params": dict(zip(self.parameter_names, values[1+n:1+2*n])),
                ("optimizer_state" if "optimizer_state" in self.ir.outputs else "momentum"): dict(zip(self.optimizer_state_names, values[1+2*n:1+2*n+no])),
                "state": dict(zip(self.state_names, values[1+2*n+no:1+2*n+no+ns])),
                "carry": dict(zip(self.carry_names, values[1+2*n+no+ns:-2])),
                "executed_count": values[-2], "status": values[-1]}

    def graph_step(self, *flat, storage_mode=None):
        """CPU qualification of the native region schedule; no AD is invoked.

        The Python loop is a reference interpreter only. Device execution uses
        BoundedLoopIR and reads the count inside the cooperative kernel.
        """
        n, no, ns, nc = len(self.parameter_names), len(self.optimizer_state_names), len(self.state_names), len(self.carry_names)
        mode = storage_mode or self.ir.storage_mode
        if mode not in self.ir.metadata.get("storage_modes", STORAGE_MODES):
            raise CaptureError("LOOP_STORAGE", f"unknown loop storage mode {mode}")
        params, momentum = tuple(flat[:n]), tuple(flat[n:n+no])
        initial_state, inputs, count = tuple(flat[n+no:n+no+ns]), tuple(flat[n+no+ns:-1]), flat[-1]
        requested = int(count.item())
        steps = min(max(requested, 0), self.ir.max_steps)
        graphs = self.graph_modules
        # Update-invariant casts are computed once from the bindings of this
        # update; every non-optimizer region reads them after the parameters.
        casts = tuple(graphs["invariant_casts"](*params, *inputs)) if "invariant_casts" in graphs else ()
        optimizer_params, params = params, (*params, *casts)
        specs = self.ir.metadata.get("sequence_specs", [])
        init_tape = ()
        if "init_saved" in graphs:
            initialized = tuple(graphs["init_saved"](*params, *inputs))
            nx = len(specs)
            init_tape = tuple(value.contiguous().clone() for value in initialized[nc+nx:])
            initialized = initialized[:nc+nx]
        else:
            initialized = tuple(graphs["init"](*params, *inputs))
        if self.ir.metadata.get("nested_loop"):
            initialized = tuple(value.contiguous() for value in initialized)
        initial_carry, xs = initialized[:nc], initialized[nc:]
        xdiff = [i for i, spec in enumerate(specs) if spec["differentiable"]]
        dxs = [torch.zeros_like(xs[i]) for i in xdiff]
        carry, state, history, selected = initial_carry, initial_state, [], []
        for t in range(steps):
            old_carry, old_state = carry, state
            selector = self.ir.metadata.get("conditional_body", {}).get("predicate_state")
            selected.append("true" if selector is not None and bool(state[self.state_names.index(selector)].item()) else "false")
            result = graphs["body_saved" if mode == "save_body_tape" else "body"](*params, *carry, *state,
                                                                                    *(x[t] for x in xs), *inputs)
            carry, state = tuple(result[:nc]), tuple(result[nc:nc+ns])
            if mode != "recompute_boundaries":
                tape = tuple(value.contiguous().clone() for value in result[nc+ns:]) if mode == "save_body_tape" else ()
                history.append((old_carry, old_state, tape))
        final_carry, final_state = carry, state
        result = graphs["terminal_vjp"](*params, *carry, *inputs)
        loss, gradients, cotangents = result[0], tuple(result[1:1+n]), tuple(result[1+n:])
        for index in range(steps - 1, -1, -1):
            if mode != "recompute_boundaries":
                carry, state, tape = history[index]
            else:
                carry, state = initial_carry, initial_state
                for replay in range(index):
                    result = graphs["body"](*params, *carry, *state, *(x[replay] for x in xs), *inputs)
                    carry, state = tuple(result[:nc]), tuple(result[nc:])
            xt = tuple(x[index] for x in xs)
            if mode == "save_body_tape":
                result = graphs["body_vjp_saved"](*params, *carry, *state, *xt, *inputs, *tape, *cotangents)
            else:
                result = graphs["body_vjp"](*params, *carry, *state, *xt, *inputs, *cotangents)
            gradients = tuple(a + b for a, b in zip(gradients, result[:n]))
            cotangents = tuple(result[n:n+nc])
            for j, value in enumerate(result[n+nc:]):
                dxs[j][index] = value
        initial_gradients = (graphs["init_vjp_saved"](*params, *inputs, *init_tape, *cotangents, *dxs)
                             if "init_saved" in graphs else graphs["init_vjp"](*params, *inputs, *cotangents, *dxs))
        gradients = tuple(a + b for a, b in zip(gradients, initial_gradients))
        flags = tuple(torch.tensor(value, dtype=torch.bool, device=params[0].device)
                      for value in _loop_participation(self.ir.metadata, selected))
        extra = tuple(inputs[i] for i in self.ir.metadata.get("optimizer_inputs", ()))
        updated =graphs["optimizer"](*optimizer_params, *momentum, *gradients, *flags, *extra)
        nested = self.ir.metadata.get("nested_loop")
        inner_clamped = (nested is not None and not
                         0 <= int(inputs[nested["count_input"]].item()) <= nested["max_steps"])
        return (loss, *gradients, *updated[:n], *updated[n:],
                *(value.clone() for value in final_state), *(value.clone() for value in final_carry),
                count.new_tensor(steps), count.new_tensor(int(requested != steps or inner_clamped)))


def _loop_participation(metadata, selected):
    """Boolean reverse connectivity, independent of derivative values."""
    connectivity = metadata.get("gradient_connectivity")
    if connectivity is None:
        return ()
    n = len(metadata["parameter_names"])
    parameter = list(connectivity["terminal"][:n])
    carry = list(connectivity["terminal"][n:])
    for branch in reversed(selected):
        rows = connectivity["body"][branch] if "conditional_body" in metadata else connectivity["body"]
        parameter = [old or any(active and row[i] for active, row in zip(carry, rows))
                     for i, old in enumerate(parameter)]
        carry = [any(active and row[n+i] for active, row in zip(carry, rows)) for i in range(len(carry))]
    return tuple(old or any(active and row[i] for active, row in zip(carry, connectivity["init"]))
                 for i, old in enumerate(parameter))


def _import_region(gm, label, input_labels):
    values, operations, inputs, outputs = {}, [], [], None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            values[node.name] = _value(node, role="input")
            inputs.append(node.name)
        elif node.op == "get_attr":
            raise CaptureError("LOOP_CONSTANT", f"{label}: pass captured tensor constant {node.target} as an explicit input")
        elif node.op == "call_function":
            target = str(node.target)
            schema = getattr(node.target, "_schema", None)
            if schema is None and node.name in values and values[node.name].producer is not None:
                continue
            if schema is None:
                raise CaptureError("LOOP_HIGHER_ORDER", f"{label}: {target} needs an explicit operation lowering")
            if isinstance(node.meta.get("val"), (tuple, list)):
                if target not in MULTI_RESULT_TARGETS:
                    raise CaptureError("MULTI_RESULT", f"{label}: {target} returns several tensors and has no operation rule")
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
            if any(result.alias_info is not None for result in schema.returns) and target not in LOOP_ALIASES:
                raise CaptureError("LOOP_ALIAS", f"{label}: {target} needs an explicit alias rule before tape partitioning")
            _reject_effects(node)
            args, kwargs = _encode(node.args), _encode(node.kwargs)
            reads = tuple(dict.fromkeys(_refs((args, kwargs))))
            dependencies = tuple(dict.fromkeys(values[name].producer for name in reads if values[name].producer is not None))
            alias = reads[0] if target in LOOP_ALIASES and reads else None
            values[node.name] = _value(node, producer=node.name, alias_of=alias)
            axes = ()
            if target.startswith(("aten.sum.", "aten.mean.", "aten.amax.")):
                dims = args[1] if len(args) > 1 and args[1] is not None else tuple(range(len(values[reads[0]].shape)))
                axes = (dims,) if isinstance(dims, int) else tuple(dims)
            operations.append(Operation(node.name, target, args, kwargs, reads, (node.name,), dependencies,
                                        reads, () if alias else (node.name,), reduction_axes=axes,
                                        synchronization="view" if alias else "global_phase_boundary",
                                        effect="rng_state_transition" if target == "latch_ops.philox_advance.default" else "pure"))
        elif node.op == "output":
            refs = _encode(node.args[0])
            if not isinstance(refs, (tuple, list)) or not all(isinstance(ref, Ref) for ref in refs):
                raise CaptureError("LOOP_SIGNATURE", f"{label}: region outputs must be a flat Tensor tuple")
            outputs = tuple(ref.name for ref in refs)
        else:
            raise CaptureError("LOOP_NODE", f"{label}: unsupported node {node.op}")
    if len(inputs) != len(input_labels) or outputs is None:
        raise CaptureError("LOOP_SIGNATURE", f"{label}: region argument/output ABI changed")
    for index, name in enumerate(outputs):
        values[name].output_roles += (f"region_output:{index}",)
    return OperationIR(values, operations, tuple(inputs), outputs,
                       {"name": label, "input_labels": list(input_labels), "torch_version": torch.__version__,
                        "state_commit": "none; enclosing loop owns all commits"})


def _capture_region(function, arguments, label, labels, *, round_trips=False, share_casts=False):
    from .standard_ops import capture_functions
    try:
        with capture_functions():
            gm = make_fx(function, tracing_mode="fake", _allow_non_fake_inputs=False,
                         _error_on_data_dependent_ops=True, decomposition_table=capture_decompositions())(*arguments)
    except CaptureError:
        raise
    except Exception as exc:
        code = "LOOP_RUNTIME_CONTROL" if any(text in str(exc).lower() for text in ("data-dependent", "data dependent", "_local_scalar_dense")) else "LOOP_TRACE"
        raise CaptureError(code, f"{label}: {type(exc).__name__}: {exc}") from exc
    gm = _functionalize_internal_mutations(gm, arguments)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    removed, copies = 0, []
    if round_trips:
        from .precision import cancel_round_trips
        from .layout_normalize import materialize_reshape_copies
        removed = cancel_round_trips(gm)
        copies = materialize_reshape_copies(gm)
    shared = None
    if share_casts:
        from .precision import share_float_casts
        shared = share_float_casts(gm)
    region = _import_region(gm, label, labels)
    if removed:
        region.metadata["exact_round_trips_removed"] = removed
    if shared and (shared["commuted"] or shared["merged"]):
        region.metadata["shared_casts"] = shared
    if copies:
        region.metadata["layout_copies"] = copies
    return region, gm


# Adapted from pinned P-OPS-E 578fa6ea14f12b04b85148c73cf6ef44415910e1.
# "precision_remat" (C-OPS2-A) adds exact floating-cast rematerialization to
# "pointwise_remat"; it changes saved storage only, never computed values.
# P9-C: merge duplicate float casts of the differentiated body joint (precision.share_float_casts; exact).
SHARE_BODY_CASTS = os.environ.get("P9C_SHARECAST", "1") != "0"
TAPE_POLICIES = ("stage3", "pointwise_remat", "precision_remat", "storage_remat")
# P7-D storage_remat: matrix-product FLOPs admitted per byte of body tape removed by a rematerialization.
STORAGE_REMAT_FLOPS_PER_BYTE = 64
# Deterministic functions of the explicit Philox state (no state commit in a loop region).
_RNG_FUNCTIONS = ("latch_ops.philox_mask.default", "latch_ops.philox_advance.default")
_REMAT_REDUCTIONS = {"_softmax", "_log_softmax", "mean", "sum", "amax"}
_FLOAT_BYTES = {"float64": 8, "float32": 4, "bfloat16": 2, "float16": 2}
_VALUE_BYTES = {**_FLOAT_BYTES, "int64": 8, "int32": 4, "bool": 1, "uint8": 1}
# Pure per-element arithmetic whose recomputation from identical saved inputs is
# bitwise identical to the forward computation (no RNG, reductions or indexing).
_RECOMPUTABLE = {"add", "sub", "mul", "div", "pow", "neg", "tanh", "exp", "log", "sqrt", "rsqrt", "abs",
                 "sigmoid", "silu", "silu_backward", "tanh_backward", "sigmoid_backward", "relu",
                 "threshold_backward"}


def _float_cast_source(joint, producer, name):
    """Source of an exact, deterministic float-to-float dtype conversion, else None.

    Round-to-nearest narrowing (FP32->BF16) and widening (BF16->FP32) of the
    same source always reproduce identical bits, so the reverse region may
    recompute them instead of reading a saved copy.
    """
    operation = producer.get(name)
    if operation is None or operation.target != "aten._to_copy.default" or len(operation.reads) != 1:
        return None
    source = operation.reads[0]
    value, original = joint.values[name], joint.values[source]
    if (value.alias_of is not None or value.dtype == original.dtype or value.dtype not in _FLOAT_BYTES
            or original.dtype not in _FLOAT_BYTES or tuple(value.shape) != tuple(original.shape)
            or operation.effect != "pure"):
        return None
    return source


def _cse_joint(module):
    """P7-D: merge identical deterministic pure call_function nodes of a joint primal/VJP graph (same target,
    arguments and keyword arguments after merging their inputs), including identical views of one base (the loop's
    admitted alias operations; the graph is functional). Exact: each merged node computed the same value. Excludes
    uninitialized allocations, random and effectful operations. Returns the merged node names."""
    from torch.fx.node import map_arg
    seen, merged = {}, []
    for node in list(module.graph.nodes):
        if node.op != "call_function":
            continue
        target = str(node.target)
        schema = getattr(node.target, "_schema", None)
        if (schema is None or schema.is_mutable or any(word in target for word in ("empty", "rand", "philox", "dropout", "bernoulli"))
                or (any(result.alias_info is not None for result in schema.returns) and target not in LOOP_ALIASES)):
            continue
        key = (target, repr(map_arg(node.args, lambda n: n.name)), repr(map_arg(node.kwargs, lambda n: n.name)))
        if key in seen:
            node.replace_all_uses_with(seen[key])
            module.graph.erase_node(node)
            merged.append(node.name)
        else:
            seen[key] = node
    if merged:
        module.graph.eliminate_dead_code()
        module.recompile()
    return merged


def _value_bytes(value):
    return math.prod(value.shape) * _VALUE_BYTES.get(value.dtype, 4)


def _remat_class(joint, producer, operation):
    """Recomputation class of a cotangent-independent operation, or None if it must stay saved."""
    if len(operation.outputs) != 1:
        return None
    if operation.target in _RNG_FUNCTIONS:
        return "rng"
    if operation.effect != "pure":
        return None
    base = operation.target.removeprefix("aten.").split(".")[0]
    if base in ("mm", "bmm"):
        return "matrix"
    if base in _RECOMPUTABLE or base in ("clone", "contiguous"):
        return "elementwise"
    if base == "_to_copy" and _float_cast_source(joint, producer, operation.outputs[0]) is not None:
        return "elementwise"
    if base in _REMAT_REDUCTIONS:
        return "reduction"
    return None


def _matrix_flops(joint, operation):
    out = joint.values[operation.outputs[0]].shape
    inner = joint.values[operation.reads[0]].shape[-1]
    return 2 * math.prod(out) * inner


def _storage_remat(joint, producer, dependent, cut, remat_ops, decide, flops_per_byte, min_bytes=0,
                   keep_operands=False, operand_copies=False):
    """Greedy bytes-driven rematerialization over the current tape decision (see storage_remat).

    min_bytes: a replacement must remove at least this many tape bytes. keep_operands: a value read by a matrix
    product in the reverse region stays saved unless its recomputation chain itself contains a matrix product (a
    data-expanding recompute such as attention scores); recomputing a pure elementwise/copy operand would add a
    materializing pass in front of the product. operand_copies admits such an operand when its chain is only
    copies and exact casts (a layout/precision copy of a narrower saved value)."""
    def tape_bytes():
        return sum(_value_bytes(joint.values[name]) for name in cut)
    matrix_readers = set()
    for op in joint.operations:
        if op.target.removeprefix("aten.").split(".")[0] in ("mm", "bmm") and op.outputs[0] in dependent:
            for name in op.reads:
                source = name
                while joint.values[source].alias_of is not None:
                    source = joint.values[source].alias_of
                matrix_readers.update((name, source))
    accepted, rejected = [], []
    changed = True
    while changed:
        changed = False
        for name in sorted(cut, key=lambda key: (-_value_bytes(joint.values[key]), key)):
            if name not in cut or name in dependent:
                continue
            before_cut, before = set(cut), tape_bytes()
            trial, frontier, outcome = [], [name], None
            for _ in range(12):
                ops = [producer[v] for v in frontier if v in producer]
                new = [op for op in ops if op.name not in remat_ops and _remat_class(joint, producer, op)]
                if not new or len(new) != len(ops):
                    break
                for op in new:
                    if op.name not in trial:
                        trial.append(op.name)
                remat_ops.update(trial)
                decide()
                removed = before - tape_bytes()
                flops = sum(_matrix_flops(joint, producer_op) for producer_op in
                            (next(op for op in joint.operations if op.name == t) for t in trial)
                            if _remat_class(joint, producer, producer_op) == "matrix")
                copies = operand_copies and all(
                    next(op for op in joint.operations if op.name == t).target.removeprefix("aten.").split(".")[0]
                    in ("clone", "contiguous", "_to_copy") for t in trial)
                operand_cost = keep_operands and flops == 0 and name in matrix_readers and not copies
                if removed > max(0, min_bytes) and flops <= flops_per_byte * removed and not operand_cost:
                    outcome = {"value": name, "bytes": _value_bytes(joint.values[name]), "removed_bytes": removed,
                               "recomputed_operations": list(trial), "matrix_flops": flops,
                               "newly_saved": sorted(set(cut) - before_cut)}
                    break
                frontier = sorted(set(cut) - before_cut)
            if outcome:
                accepted.append(outcome)
                changed = True
            else:
                remat_ops.difference_update(trial)
                decide()
                if trial:
                    rejected.append({"value": name, "tried_operations": list(trial)})
    return {"flops_per_byte": flops_per_byte, "accepted": accepted, "rejected": rejected,
            "tape_bytes": tape_bytes()}


def _partition_body_tape(joint, module, *, n_primals, n_forward_outputs, policy="stage3",
                         forced_remat=(), forced_save=(), tape_options=None):
    """Generic dependency cut of a PyTorch-derived primal/VJP graph.

    All tensor values that depend on incoming cotangents belong to backward.
    Independent material roots crossing into backward become explicit tape.
    Alias/view chains and cheap zero/full/copy nodes are regenerated, so a
    noncontiguous view is never saved with an incompatible dense tape layout.
    No derivative formula or model/operation-size pattern is recognized here.

    policy "pointwise_remat" (P-OPS-E) changes only where cotangent-independent
    work runs, never what is computed: (1) per-element arithmetic used only by
    the reverse region (not by the forward outputs) is recomputed there, placed
    immediately before its first reverse consumer, instead of being computed in
    the forward body and saved; (2) a dense copy that the forward body computes
    anyway is saved rather than recopied in reverse. Cf. functorch's min-cut
    rematerialization partitioner, simplified to these two exact rules.

    policy "precision_remat" (C-OPS2-A) additionally moves exact float casts
    out of the tape: a cast whose source the reverse region already has (a
    primal input such as an FP32 master weight, or another saved value) is
    recomputed there, and an exact widening saves its narrower source instead
    of the wider copy. Explicit mixed-precision recipes otherwise save both a
    BF16 operand and its FP32 original/widening, so BF16 would enlarge the tape.

    forced_remat/forced_save (P-OPS4-E, ops/loop_attention.py) carry an explicit
    attention tape decision: listed cotangent-independent operations are recomputed
    in reverse, and listed values are saved even when a rule above would recompute
    them. Neither changes what the joint graph computes.

    policy "storage_remat" (P7-D) starts from the precision_remat decision and then,
    largest saved value first, tries to replace a saved value by recomputing its
    producer chain in reverse from values the reverse region already has (primal
    inputs, other saved values, recomputed values). The chain grows one producer
    level at a time through recomputable operations (per-element arithmetic, exact
    float casts, copies, explicit-Philox mask/advance functions, row reductions and
    matrix products). A replacement is accepted only if the total saved bytes fall
    and its matrix-product FLOPs per removed byte stay within
    tape_options["flops_per_byte"] (default STORAGE_REMAT_FLOPS_PER_BYTE). Only where
    values are produced changes; the joint graph's operations are unchanged.
    """
    if policy not in TAPE_POLICIES:
        raise CaptureError("LOOP_TAPE", f"unknown tape policy {policy}")
    primal_inputs, cotangent_inputs = joint.inputs[:n_primals], joint.inputs[n_primals:]
    dependent = set(cotangent_inputs)
    producer = {}
    for operation in joint.operations:
        for name in operation.outputs:
            producer[name] = operation
        if any(name in dependent for name in operation.reads):
            dependent.update(operation.outputs)
    forward_outputs, reverse_outputs = joint.outputs[:n_forward_outputs], joint.outputs[n_forward_outputs:]
    if any(name in dependent for name in forward_outputs):
        raise CaptureError("LOOP_TAPE", "a primal body output unexpectedly depends on a cotangent")
    cheap = {"clone", "contiguous", "fill", "zeros_like", "ones_like", "zeros", "ones", "full", "full_like", "new_zeros", "new_ones", "new_full"}
    retained, cut, recomputed = set(), set(), set()
    forward_core = set()

    def core(name):
        if name in primal_inputs or producer[name].name in forward_core:
            return
        forward_core.add(producer[name].name)
        for source in producer[name].reads:
            core(source)
    if policy != "stage3":
        for name in forward_outputs:
            core(name)

    cast_remat = set()
    remat_ops = set(forced_remat)

    def backward_needed(name):
        if name in joint.inputs or name in cut:
            return
        operation = producer[name]
        if operation.name in retained:
            return
        if name in forced_save and name not in dependent:
            cut.add(name)
            return
        if operation.name in remat_ops and name not in dependent:
            retained.add(operation.name)
            recomputed.add(operation.name)
            for source in operation.reads:
                backward_needed(source)
            return
        if operation.name in cast_remat:
            retained.add(operation.name)
            recomputed.add(operation.name)
            for source in operation.reads:
                backward_needed(source)
            return
        base = operation.target.removeprefix("aten.").split(".")[0]
        saved_copy = (policy != "stage3" and base in {"clone", "contiguous"} and operation.name in forward_core
                      and joint.values[name].alias_of is None)
        remat = (policy != "stage3" and base in _RECOMPUTABLE and operation.name not in forward_core
                 and len(operation.outputs) == 1 and operation.effect == "pure")
        # A functional fill's empty_like template is shape metadata only; a moved
        # fill regenerates it rather than saving undefined bytes.
        metadata_only = policy != "stage3" and base == "empty_like"
        if (name in dependent or operation.target in LOOP_ALIASES or (base in cheap and not saved_copy) or remat
                or metadata_only):
            retained.add(operation.name)
            if remat and name not in dependent:
                recomputed.add(operation.name)
            for source in operation.reads:
                backward_needed(source)
        else:
            if joint.values[name].alias_of is not None:
                raise CaptureError("LOOP_TAPE", f"unhandled aliased tape value {name}")
            cut.add(name)
    cast_report = []
    order = {op.name: index for index, op in enumerate(joint.operations)}

    def decide():
        retained.clear(); cut.clear(); recomputed.clear(); cast_remat.clear(); cast_report.clear()
        for name in reverse_outputs:
            backward_needed(name)
        if policy in ("precision_remat", "storage_remat"):
            # Second pass from the first pass's storage decision. Only casts that the
            # first pass saved are reconsidered; every other decision is unchanged.
            available = set(primal_inputs) | cut
            for name in sorted(cut, key=lambda key: order[producer[key].name]):
                source = _float_cast_source(joint, producer, name)
                if source is None or source in dependent:
                    continue
                value, original = joint.values[name], joint.values[source]
                widening = _FLOAT_BYTES[original.dtype] < _FLOAT_BYTES[value.dtype]
                if source in available or widening:
                    cast_remat.add(producer[name].name)
                    cast_report.append({"value": name, "source": source, "from": original.dtype, "to": value.dtype,
                                        "reason": "source already available in reverse" if source in available
                                                  else "save narrower source of exact widening"})
            if cast_remat:
                retained.clear(); cut.clear(); recomputed.clear()
                for name in reverse_outputs:
                    backward_needed(name)
    decide()
    storage_report = None
    if policy == "storage_remat":
        options = tape_options or {}
        storage_report = _storage_remat(joint, producer, dependent, cut, remat_ops, decide,
                                        float(options.get("flops_per_byte", STORAGE_REMAT_FLOPS_PER_BYTE)),
                                        min_bytes=int(options.get("min_bytes", 0)),
                                        keep_operands=bool(options.get("keep_operands", False)),
                                        operand_copies=bool(options.get("operand_copies", False)))
    tape = tuple(name for operation in joint.operations for name in operation.outputs if name in cut)
    forward_retained = set()

    def forward_needed(name):
        if name in primal_inputs:
            return
        if name in dependent:
            raise CaptureError("LOOP_TAPE", "forward tape depends on a cotangent")
        operation = producer[name]
        if operation.name in forward_retained:
            return
        forward_retained.add(operation.name)
        for source in operation.reads:
            forward_needed(source)
    for name in (*forward_outputs, *tape):
        forward_needed(name)

    def reverse_order():
        # Recomputed producers are emitted just before their first consumer so
        # that elementwise fusion can absorb them; all else keeps joint order.
        order, emitted = [], set()

        def emit(name):
            if name in emitted:
                return
            operation = by_name[name]
            for source in operation.reads:
                if source in producer and producer[source].name in recomputed:
                    emit(producer[source].name)
            emitted.add(name)
            order.append(name)
        for operation in joint.operations:
            if operation.name in retained and operation.name not in recomputed:
                emit(operation.name)
        for operation in joint.operations:
            if operation.name in recomputed:
                emit(operation.name)
        return order

    by_name = {operation.name: operation for operation in joint.operations}

    def subgraph(label, inputs, outputs, chosen, labels, order=None):
        values = {name: replace(joint.values[name], role="input", producer=None, alias_of=None,
                                strides=_contiguous_strides(joint.values[name].shape), storage_offset=0)
                  if name in cut else replace(joint.values[name], role="input", producer=None)
                  for name in inputs}
        operations = []
        for operation in (joint.operations if order is None else [by_name[name] for name in order]):
            if operation.name not in chosen:
                continue
            for name in operation.outputs:
                values[name] = replace(joint.values[name])
            dependencies = tuple(dict.fromkeys(values[name].producer for name in operation.reads
                                               if values[name].producer is not None))
            operations.append(replace(operation, dependencies=dependencies))
        result = OperationIR(values, operations, tuple(inputs), tuple(outputs),
                             {**joint.metadata, "name": label, "input_labels": list(labels),
                              "partition": "cotangent dependency cut; aliases/cheap constants rematerialized"})
        graph = torch.fx.Graph()
        source_nodes = {node.name: node for node in module.graph.nodes}
        copied = {}
        for name in inputs:
            node = graph.placeholder(name)
            node.meta = dict(source_nodes[name].meta)
            copied[name] = node
        for operation in operations:
            copied[operation.name] = graph.node_copy(source_nodes[operation.name], lambda node: copied[node.name])
            if operation.outputs != (operation.name,):
                # P8-C: an admitted multi-result operation's results are its getitem selector nodes.
                import operator as _operator
                for user in source_nodes[operation.name].users:
                    if user.target is _operator.getitem and user.name in operation.outputs:
                        copied[user.name] = graph.node_copy(user, lambda node: copied[node.name])
        graph.output(tuple(copied[name] for name in outputs))
        return result, torch.fx.GraphModule(module, graph)

    labels = joint.metadata["input_labels"]
    forward, forward_module = subgraph("body_saved", primal_inputs, (*forward_outputs, *tape),
                                       forward_retained, labels[:n_primals])
    backward, backward_module = subgraph("body_vjp_saved", (*primal_inputs, *tape, *cotangent_inputs),
                                         reverse_outputs, retained,
                                         [*labels[:n_primals], *(f"tape:{name}" for name in tape), *labels[n_primals:]],
                                         order=None if policy == "stage3" else reverse_order())
    backward.metadata["tape_policy"] = forward.metadata["tape_policy"] = policy
    if forced_remat or forced_save:
        backward.metadata["forced_rematerialization"] = sorted(forced_remat)
        backward.metadata["forced_tape"] = sorted(forced_save)
    backward.metadata["recomputed_in_reverse"] = sorted(recomputed)
    if policy in ("precision_remat", "storage_remat"):
        backward.metadata["cast_rematerialization"] = list(cast_report)
    if storage_report is not None:
        backward.metadata["storage_rematerialization"] = storage_report
    descriptors = [{"name": name, "shape": list(joint.values[name].shape), "dtype": joint.values[name].dtype,
                    "strides": list(_contiguous_strides(joint.values[name].shape))} for name in tape]
    return forward, forward_module, backward, backward_module, descriptors


# P-OPS4-F: update-invariant cast storage, applied AFTER differentiation.
#
# A differentiable cast that is hoisted before AD and shared by several uses is
# not equivalent: its single VJP adds the uses' cotangents in the narrow dtype
# before widening, while separate casts widen each contribution and add in FP32.
# This pass therefore never changes a VJP. It rewrites, in every PyTorch-derived
# non-optimizer region, the forward VALUE of an exact float cast of an
# update-invariant binding into a read of one cast buffer, computed once per
# launch by the invariant_casts region. Every cotangent widening and every
# accumulation (and its dtype) stays where PyTorch put it.
_PERMUTATION_VIEWS = {"aten.t.default", "aten.transpose.int", "aten.permute.default"}
# Bindings never written between the start of an update and its final commit.
# Carry, explicit state, tape and cotangents change within the update.
_INVARIANT_ROLES = ("parameter", "input")


def _cast_candidates(gm, labels):
    """Classify every float->float _to_copy of one region graph."""
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    role = dict(zip(placeholders, labels))
    rows = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or str(node.target) != "aten._to_copy.default":
            continue
        value, source = node.meta.get("val"), node.args[0] if node.args else None
        original = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
        if (not isinstance(value, torch.Tensor) or not isinstance(original, torch.Tensor)
                or not value.is_floating_point() or not original.is_floating_point() or value.dtype == original.dtype):
            continue
        chain, root = [], source
        while root.op == "call_function" and str(root.target) in _PERMUTATION_VIEWS:
            chain.append(root)
            root = root.args[0]
        chain.reverse()
        label = role.get(root) if root.op == "placeholder" else None
        reason = None
        if len(node.args) != 1 or set(node.kwargs) != {"dtype"}:
            reason = "conversion arguments other than dtype"
        elif label is None:
            reason = "source is computed inside the region, not a binding or a permutation view of one"
        elif label.split(":", 1)[0] not in _INVARIANT_ROLES:
            reason = f"source binding {label.split(':', 1)[0]} changes within the update"
        else:
            expected = torch.empty(tuple(root.meta["val"].shape), dtype=value.dtype, device="meta")
            for view in chain:
                expected = view.target(expected, *view.args[1:], **view.kwargs)
            if (tuple(expected.shape) != tuple(value.shape) or tuple(expected.stride()) != tuple(value.stride())
                    or int(value.storage_offset()) != 0 or not root.meta["val"].is_contiguous()):
                reason = "cast layout differs from the permuted contiguous cast of its binding"
        rows.append({"node": node, "root": root, "label": label, "chain": chain, "dtype": value.dtype,
                     "reason": reason})
    return rows


def _hoist_invariant_casts(regions, n, external_labels):
    """Rewrite eligible cast values into reads of update-invariant cast inputs.

    regions maps a region label to (GraphModule, input labels); every region's
    first n inputs are the parameters. Inserts one placeholder per (binding,
    dtype) after the parameters of EVERY region, so all consumers share one
    input ABI, and returns the cast keys, new labels and a report.
    """
    rows = {label: _cast_candidates(gm, labels) for label, (gm, labels) in regions.items()}
    keys = sorted({(row["label"], row["dtype"]) for region in rows.values() for row in region if row["reason"] is None},
                  key=lambda key: (external_labels.index(key[0]), str(key[1])))
    report = {"eligible": {label: sum(row["reason"] is None for row in region) for label, region in rows.items()},
              "ineligible": [{"region": label, "value": row["node"].name, "source": row["label"] or row["root"].name,
                              "reason": row["reason"]}
                             for label, region in rows.items() for row in region if row["reason"] is not None]}
    if not keys:
        return keys, {label: labels for label, (_, labels) in regions.items()}, report
    cast_labels = [f"invariant_cast:{source}:{str(dtype).removeprefix('torch.')}" for source, dtype in keys]
    new_labels = {}
    for label, (gm, labels) in regions.items():
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        anchor, inserted = placeholders[n - 1], {}
        sources = {labels[i]: node for i, node in enumerate(placeholders)}
        for index, (source, dtype) in enumerate(keys):
            with gm.graph.inserting_after(anchor):
                anchor = gm.graph.placeholder(f"invariant_cast_{index}")
            # P-OPS5-B merge fix: same (fake) device as the source binding, so later passes that
            # re-propagate the joint graph (P-OPS4-E loop attention) see one device.
            anchor.meta["val"] = sources[source].meta["val"].new_empty(tuple(sources[source].meta["val"].shape), dtype=dtype)
            inserted[(source, dtype)] = anchor
        for row in rows[label]:
            if row["reason"] is not None:
                continue
            replacement = inserted[(row["label"], row["dtype"])]
            for view in row["chain"]:
                with gm.graph.inserting_before(row["node"]):
                    node = gm.graph.call_function(view.target, (replacement, *view.args[1:]), dict(view.kwargs))
                node.meta["val"] = view.target(replacement.meta["val"], *view.args[1:], **view.kwargs)
                replacement = node
            row["node"].replace_all_uses_with(replacement)
            gm.graph.erase_node(row["node"])
        gm.graph.eliminate_dead_code()
        gm.recompile()
        new_labels[label] = [*labels[:n], *cast_labels, *labels[n:]]
    return keys, new_labels, report


def _check_invariant_cast_abi(graphs, metadata):
    """Every consumer reads the cast buffers right after the parameters; only
    the once-per-launch invariant_casts region writes them; no loop region
    commits state, so their bindings cannot change before the final commit."""
    casts = metadata.get("invariant_casts", [])
    if not casts:
        return
    n = len(metadata["parameter_names"])
    producer = graphs["invariant_casts"]
    for desc, name in zip(casts, producer.outputs):
        value = producer.values[name]
        if (list(value.shape), value.dtype) != (desc["shape"], desc["dtype"]):
            raise CaptureError("CAST_REUSE", f"invariant cast {desc['name']} differs from its producer output")
    expected = [f"invariant_cast:{desc['source']}:{desc['dtype']}" for desc in casts]
    for label in ("init", "body", "terminal_vjp", "body_vjp", "init_vjp", "body_saved", "body_vjp_saved"):
        region = graphs[label]
        if region.metadata["input_labels"][n:n+len(casts)] != expected:
            raise CaptureError("CAST_REUSE", f"{label}: invariant casts are not read at the shared ABI position")
        if getattr(region, "state_writes", None):
            raise CaptureError("CAST_REUSE", f"{label}: a loop region may not commit state")
    if any(label.startswith("invariant_cast:") for label in graphs["optimizer"].metadata["input_labels"]):
        raise CaptureError("CAST_REUSE", "the optimizer must read the master parameters, not cast storage")


def capture_loop_training(init_fn, body_fn, loss_fn, params, momentum, inputs, count,
                          config=None, *, max_steps, state=None, storage_mode="save_body_tape", optimizer_state=None,
                          tape_policy="auto", loop_attention="auto", cast_reuse=None, sequence_fn=None,
                          init_storage="recompute", tape_options=None, memory_policy="default", precision=None,
                          window_attention="auto", window_fusion="auto"):
    """Capture a bounded loop's reusable forward/VJP/optimizer regions.

    init_fn(params, *inputs) -> floating (FP32 or BF16) carry dict
    body_fn(params, carry, state, *inputs) -> (new_carry_dict, new_state_dict)
    loss_fn(params, final_carry, *inputs) -> scalar FP32 loss

    Every floating carried value participates in the full reverse history.
    Integer/bool state is explicitly nondifferentiable; use carry for floating
    recurrent state and intermediate-supervision accumulators. Count is a scalar
    int64 device input, clamped to [0,max_steps]; status=1 reports clamping.

    cast_reuse=True computes each exact float cast of a parameter/data binding
    once per update (P-OPS4-F) and reuses its storage in every iteration and
    region; derivatives are unchanged. Conditional bodies keep per-use casts.

    memory_policy="low_memory" selects the P7-D storage_remat policy: saved-body
    tape, invariant cast reuse, a 64 FLOP/byte budget, 1 MiB minimum saving,
    operand preservation, common-subexpression merging and operand-copy replay.
    It requires a nonconditional body, no sequences and recomputed init. Explicit
    conflicting tape/cast/storage choices are refused. "default" retains the
    ordinary policy and explicit diagnostic overrides. Omitted cast_reuse enables
    the existing invariant-cast pass for plain save_body_tape loops with recomputed
    init, and stays disabled for conditional, sequence and boundary-storage loops
    or saved init. Explicit False disables it. Reuse is selected only when the
    pass proves an eligible exact cast; it never hoists a carried/state value.

    loop_attention="auto" admits short or long BF16-boundary attention solely
    from the captured graph, shapes and layouts; inadmissible groups retain
    primitive lowering. The exact3 attention arithmetic is tolerance-gated,
    not bitwise FP32 attention. Long attention uses resident operands only up
    to sequence length 512; "off" preserves the primitive capture path.

    sequence_fn(params, *inputs) -> dict of tensors whose leading dimension is
    max_steps (C-OPS4-A, scan "xs"). It is evaluated once per update together
    with init; iteration t of the body reads slice t of every sequence and the
    body signature becomes body_fn(params, carry, state, sequence_slices, *inputs).
    Floating sequences are differentiable: each reverse step writes the
    cotangent of its own slice, unexecuted slices have zero cotangent, and one
    VJP of the sequence function (with init) returns their parameter gradients.
    Integer/bool sequences are nondifferentiable data. Sequences are neither
    carried nor saved per iteration: bulk work stays outside the chain.
    MuonConfig's schedule input is passed to the optimizer region.

    init_storage="save_tape" (C-OPS4-A) partitions init (+ sequences) and its
    VJP with the body-tape partitioner: values the VJP needs are saved once per
    update instead of the VJP region re-tracing init's forward. Default
    "recompute" keeps the original separate init/init_vjp regions.
    """
    from .nested_loop import NestedLoopBody, capture_nested_loop_training
    if isinstance(body_fn, NestedLoopBody):
        return capture_nested_loop_training(init_fn, body_fn, loss_fn, params, momentum, inputs, count, config,
            max_steps=max_steps, state=state, storage_mode=storage_mode, optimizer_state=optimizer_state,
            tape_policy=tape_policy, loop_attention=loop_attention, cast_reuse=cast_reuse, sequence_fn=sequence_fn,
            init_storage=init_storage, tape_options=tape_options, memory_policy=memory_policy, precision=precision,
            window_attention=window_attention, window_fusion=window_fusion)
    from .precision import PolicyRecord, resolve, wrap
    policy = resolve(precision)
    policy_record = PolicyRecord()
    if policy is not None:
        if isinstance(body_fn, ConditionalBody):
            raise CaptureError("PRECISION_POLICY", "precision policies are not qualified for conditional loops")
        init_fn, body_fn, loss_fn = (wrap(init_fn, policy, policy_record), wrap(body_fn, policy, policy_record),
                                     wrap(loss_fn, policy, policy_record))
        if sequence_fn is not None:
            # P8-C: the sequence function (bulk work evaluated once with init) is user forward code like
            # init/body/terminal; the same declared policy applies to it before AD. Sequence slices and their
            # cotangents keep the dtype the policy-wrapped function returns (FP32 unless the user casts).
            sequence_fn = wrap(sequence_fn, policy, policy_record)
    if torch.__version__.split("+")[0] != "2.9.1":
        raise CaptureError("TORCH_VERSION", f"loop capture requires torch 2.9.1, got {torch.__version__}")
    if type(max_steps) is not int or not 0 <= max_steps <= 1024 or storage_mode not in STORAGE_MODES:
        raise CaptureError("LOOP_BOUND", "require static max_steps in [0,1024] and a supported storage mode")
    if not isinstance(count, torch.Tensor) or count.dtype != torch.int64 or count.ndim != 0:
        raise CaptureError("LOOP_COUNT", "runtime count must be a scalar int64 Tensor")
    _reject_lazy_view(count, "loop count")
    config = config or SGDConfig()
    config.validate()
    if getattr(config, "runtime_lr", False):
        raise CaptureError("OPTIMIZER_ABI", "AdamW runtime_lr is admitted by capture_training/capture_module; "
                           "loop AdamW currently requires a static learning rate")
    if not params:
        raise CaptureError("PARAMETERS", "at least one trainable parameter is required")
    unique, aliases, identities = {}, {}, {}
    for name, value in params.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.layout != torch.strided:
            raise CaptureError("PARAMETER", f"{name}: loop parameters must be strided FP32 tensors")
        _reject_lazy_view(value, f"parameter {name}")
        if not value.is_contiguous():
            raise CaptureError("LOOP_LAYOUT", f"parameter {name}: loop mutable bindings must be contiguous")
        canonical = identities.setdefault(id(value), name)
        aliases[name] = canonical
        unique.setdefault(canonical, value)
    names, alias_index = tuple(unique), {key: tuple(unique).index(value) for key, value in aliases.items()}
    from .optim import prepare_optimizer
    from .user_optimizer import TensorUpdateRule
    if isinstance(config, TensorUpdateRule):
        raise CaptureError("OPTIMIZER", "user tensor update rules are admitted for flat capture_training/capture_module only")
    prepared = prepare_optimizer(config, unique, optimizer_state=optimizer_state, momentum=momentum, aliases=aliases)
    optimizer_names = prepared.state_names
    optimizer_role = "momentum" if prepared.legacy_momentum else "optimizer_state"
    pvalues, mvalues = tuple(value.detach() for value in unique.values()), prepared.state_values
    for name, value in zip(optimizer_names, mvalues):
        if not value.is_contiguous():
            raise CaptureError("LOOP_LAYOUT", f"optimizer state {name}: loop mutable bindings must be contiguous")
    state = state or {}
    state_names = tuple(state)
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype not in (torch.int64, torch.bool) or value.layout != torch.strided:
            raise CaptureError("LOOP_STATE", f"{name}: nondifferentiable state must be int64/bool; put floating state in carry")
        _reject_lazy_view(value, f"state {name}")
        if not value.is_contiguous():
            raise CaptureError("LOOP_LAYOUT", f"state {name}: loop mutable bindings must be contiguous")
    svalues, inputs = tuple(value.detach() for value in state.values()), tuple(inputs)
    conditional = isinstance(body_fn, ConditionalBody)
    sequenced = sequence_fn is not None
    requested_tape_policy = tape_policy
    requested_cast_reuse = cast_reuse
    tape_policy, tape_options, cast_reuse = _resolve_memory_policy(
        memory_policy, storage_mode=storage_mode, tape_policy=tape_policy, tape_options=tape_options,
        cast_reuse=cast_reuse, conditional=conditional, sequenced=sequenced, init_storage=init_storage)
    if init_storage not in INIT_STORAGE or (init_storage == "save_tape" and (conditional or cast_reuse)):
        raise CaptureError("LOOP_INIT_STORAGE", "init_storage must be recompute or save_tape (save_tape: nonconditional, no cast_reuse)")
    if sequenced and (conditional or cast_reuse or max_steps < 1):
        raise CaptureError("LOOP_SEQUENCE", "sequence inputs require a nonconditional body, cast_reuse=False and max_steps >= 1")
    if sequenced and prepared.uses_participation:
        raise CaptureError("LOOP_SEQUENCE", "sequence inputs with AdamW unused-parameter participation are not admitted")
    if tape_policy == "auto":
        tape_policy = "precision_remat" if not conditional and storage_mode == "save_body_tape" else "stage3"
    if tape_policy not in TAPE_POLICIES:
        raise CaptureError("LOOP_TAPE", f"unknown tape policy {tape_policy}")
    if conditional and (body_fn.predicate_state not in state or state[body_fn.predicate_state].dtype != torch.bool
                        or state[body_fn.predicate_state].ndim != 0):
        raise CaptureError("LOOP_BRANCH", "ConditionalBody requires a named scalar bool state predicate")
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("INPUT", f"loop input {index} must be a Tensor")
        _reject_lazy_view(value, f"input {index}")
    external = (*pvalues, *mvalues, *svalues, *inputs, count.detach())
    external_labels = ([f"parameter:{name}" for name in names] + [f"{optimizer_role}:{name}" for name in optimizer_names]
                       + [f"state:{name}" for name in state_names] + [f"input:{i}" for i in range(len(inputs))] + ["count"])
    _reject_placeholder_storage_alias(external, external_labels)
    if any(value.device != count.device for value in external):
        raise CaptureError("LOOP_DEVICE", "every loop input and count must be on the same device")
    n, no, ns, ni = len(names), len(optimizer_names), len(state_names), len(inputs)
    graphs, modules, carry_names, carry_specs = {}, {}, [], []
    sequence_names, sequence_specs = [], []
    schedule = getattr(config, "schedule_input", None)
    if schedule is None:
        optimizer_inputs = ()
    else:
        if not ni or not -ni <= schedule < ni:
            raise CaptureError("OPTIMIZER_ABI", "the optimizer schedule input index names no data input")
        optimizer_inputs = (schedule % ni,)
        schedule_value = inputs[optimizer_inputs[0]]
        if (schedule_value.dtype != torch.float32 or schedule_value.ndim != 1
                or schedule_value.numel() < max(5, config.schedule_entries)):
            raise CaptureError("OPTIMIZER_ABI", "Muon schedule must be a rank-1 FP32 input with at least five entries")
    p_labels = [f"parameter:{name}" for name in names]
    i_labels = [f"input:{i}" for i in range(ni)]
    s_labels = [f"state:{name}" for name in state_names]

    def mapping(flat):
        return {alias: flat[index] for alias, index in alias_index.items()}

    def checked_carry(result, *, initial=False, device=None):
        if not isinstance(result, dict) or not result:
            raise CaptureError("LOOP_CARRY", "loop carry must be a nonempty dictionary")
        if initial and not carry_names:
            carry_names.extend(result)
            for key in carry_names:
                value = result[key]
                if not isinstance(value, torch.Tensor) or value.dtype not in CARRY_DTYPES:
                    raise CaptureError("LOOP_CARRY", f"carry {key} must be FP32 or BF16; put integer state in state")
                carry_specs.append((tuple(value.shape), value.dtype, value.device))
        if set(result) != set(carry_names):
            raise CaptureError("LOOP_CARRY", "every iteration must preserve carry keys")
        for key, spec in zip(carry_names, carry_specs):
            value = result[key]
            if (not isinstance(value, torch.Tensor) or (tuple(value.shape), value.dtype) != spec[:2]
                    or value.device != device):
                raise CaptureError("LOOP_CARRY", f"iteration carry {key} changed shape, dtype or device")
        return tuple(result[key] for key in carry_names)

    def initial(pflat, *data):
        return checked_carry(init_fn(mapping(pflat), *data), initial=True, device=pflat[0].device)

    def sequences(pflat, *data):
        result = sequence_fn(mapping(pflat), *data)
        if not isinstance(result, dict) or not result:
            raise CaptureError("LOOP_SEQUENCE", "sequence_fn must return a nonempty dictionary")
        if not sequence_names:
            sequence_names.extend(result)
            for key in sequence_names:
                value = result[key]
                if (not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != max_steps
                        or value.dtype not in (*CARRY_DTYPES, torch.int64, torch.bool)):
                    raise CaptureError("LOOP_SEQUENCE", f"sequence {key} must be a FP32/BF16/int64/bool tensor "
                                       f"with leading dimension max_steps={max_steps}")
                sequence_specs.append((tuple(value.shape), value.dtype, value.device))
        if list(result) != sequence_names:
            raise CaptureError("LOOP_SEQUENCE", "sequence keys changed between traces")
        return tuple(result[key] for key in sequence_names)

    def differentiable(index):
        return sequence_specs[index][1] in CARRY_DTYPES

    active_body = body_fn

    def body(pflat, cflat, sflat, *data, xflat=()):
        if sequenced:
            result = active_body(mapping(pflat), dict(zip(carry_names, cflat)), dict(zip(state_names, sflat)),
                                 dict(zip(sequence_names, xflat)), *data)
        else:
            result = active_body(mapping(pflat), dict(zip(carry_names, cflat)), dict(zip(state_names, sflat)), *data)
        if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[1], dict) or set(result[1]) != set(state_names):
            raise CaptureError("LOOP_BODY", "body must return (new_carry, new_state) with unchanged state keys")
        carry, new_state = checked_carry(result[0], device=pflat[0].device), tuple(result[1][key] for key in state_names)
        for key, before, after in zip(state_names, sflat, new_state):
            if not isinstance(after, torch.Tensor) or (after.shape, after.dtype, after.device) != (before.shape, before.dtype, before.device):
                raise CaptureError("LOOP_STATE", f"body state {key} changed shape, dtype or device")
        return carry, new_state

    def terminal(pflat, cflat, *data):
        loss = loss_fn(mapping(pflat), dict(zip(carry_names, cflat)), *data)
        if not isinstance(loss, torch.Tensor) or loss.dtype != torch.float32 or loss.ndim != 0:
            raise CaptureError("LOSS", "loop terminal objective must return a scalar FP32 Tensor")
        return loss

    def init_region(*flat):
        carry = initial(tuple(flat[:n]), *flat[n:])
        xs = sequences(tuple(flat[:n]), *flat[n:]) if sequenced else ()
        return tuple(value.clone() for value in (*carry, *xs))

    graphs["init"], modules["init"] = _capture_region(init_region, (*pvalues, *inputs), "init", p_labels + i_labels,
                                                      round_trips=policy is not None)
    nc = len(carry_names)
    nx = len(sequence_names)
    xdiff = [i for i in range(nx) if differentiable(i)]
    x_labels = [f"sequence:{key}" for key in sequence_names]
    dx_labels = [f"sequence_cotangent:{sequence_names[i]}" for i in xdiff]
    xslices = tuple(torch.empty(shape[1:], dtype=dtype, device=device) for shape, dtype, device in sequence_specs)
    xfull_cotangents = tuple(torch.empty(sequence_specs[i][0], dtype=sequence_specs[i][1], device=sequence_specs[i][2])
                             for i in xdiff)

    def merged(xflat, xf):
        result, floats = list(xflat), iter(xf)
        for i in xdiff:
            result[i] = next(floats)
        return tuple(result)
    cvalues = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype, device in carry_specs)
    cotangents = tuple(torch.empty_like(value) for value in cvalues)
    gradients = tuple(torch.empty_like(value) for value in pvalues)
    c_labels = [f"carry:{key}" for key in carry_names]
    dc_labels = [f"carry_cotangent:{key}" for key in carry_names]

    def body_region(*flat):
        carry, new_state = body(tuple(flat[:n]), tuple(flat[n:n+nc]), tuple(flat[n+nc:n+nc+ns]), *flat[n+nc+ns+nx:],
                                xflat=tuple(flat[n+nc+ns:n+nc+ns+nx]))
        return tuple(value.clone() for value in (*carry, *new_state))

    def terminal_region(*flat):
        (dp, dc), loss = grad_and_value(terminal, argnums=(0, 1))(tuple(flat[:n]), tuple(flat[n:n+nc]), *flat[n+nc:])
        return tuple(value.clone() for value in (loss, *dp, *dc))

    def split_reverse(flat):
        p, c, s = tuple(flat[:n]), tuple(flat[n:n+nc]), tuple(flat[n+nc:n+nc+ns])
        x = tuple(flat[n+nc+ns:n+nc+ns+nx])
        data, dc = tuple(flat[n+nc+ns+nx:n+nc+ns+nx+ni]), tuple(flat[n+nc+ns+nx+ni:])
        return p, c, s, x, data, dc

    def pulled(p, c, s, x, data, dc):
        if not xdiff:
            def differentiated_body(pp, cc):
                return body(pp, cc, s, *data, xflat=x)
            primal, pullback, aux = vjp(differentiated_body, p, c, has_aux=True)
            return primal, aux, (*pullback(dc), ())
        def differentiated_sequence_body(pp, cc, xf):
            return body(pp, cc, s, *data, xflat=merged(x, xf))
        primal, pullback, aux = vjp(differentiated_sequence_body, p, c, tuple(x[i] for i in xdiff), has_aux=True)
        return primal, aux, pullback(dc)

    def body_reverse(*flat):
        _, _, (dp, old_dc, dx) = pulled(*split_reverse(flat))
        return tuple(value.clone() for value in (*dp, *old_dc, *dx))

    def body_joint(*flat):
        next_carry, next_state, (dp, old_dc, dx) = pulled(*split_reverse(flat))
        return tuple(value.clone() for value in (*next_carry, *next_state, *dp, *old_dc, *dx))

    def initial_reverse(*flat):
        p, data, dc, dx = tuple(flat[:n]), tuple(flat[n:n+ni]), tuple(flat[n+ni:n+ni+nc]), tuple(flat[n+ni+nc:])
        if not xdiff:
            _, pullback = vjp(lambda pp: initial(pp, *data), p)
            (dp,) = pullback(dc)
            return tuple(value.clone() for value in dp)
        def combined(pp):
            xs = sequences(pp, *data)
            return (initial(pp, *data), tuple(xs[i] for i in xdiff))
        _, pullback = vjp(combined, p)
        (dp,) = pullback((dc, dx))
        return tuple(value.clone() for value in dp)

    nf = len(names) if prepared.uses_participation else 0

    def optimizer_region(*flat):
        p, m, g = tuple(flat[:n]), tuple(flat[n:n+no]), tuple(flat[n+no:n+no+n])
        participation = tuple(flat[n+no+n:n+no+n+nf]) if prepared.uses_participation else None
        extra = tuple(flat[n+no+n+nf:])
        new_p, new_m = (prepared.update(p, g, m, participation=participation, inputs=extra) if optimizer_inputs
                        else prepared.update(p, g, m, participation=participation))
        return (*new_p, *new_m)

    flag_values = tuple(torch.tensor(True, device=count.device) for _ in names) if prepared.uses_participation else ()
    flag_labels = [f"participation:{key}" for key in names] if flag_values else []
    optimizer_data = tuple(inputs[i] for i in optimizer_inputs)
    specs = {
        "body": (body_region, (*pvalues, *cvalues, *svalues, *xslices, *inputs), p_labels+c_labels+s_labels+x_labels+i_labels),
        "terminal_vjp": (terminal_region, (*pvalues, *cvalues, *inputs), p_labels+c_labels+i_labels),
        "body_vjp": (body_reverse, (*pvalues, *cvalues, *svalues, *xslices, *inputs, *cotangents),
                     p_labels+c_labels+s_labels+x_labels+i_labels+dc_labels),
        "init_vjp": (initial_reverse, (*pvalues, *inputs, *cotangents, *xfull_cotangents), p_labels+i_labels+dc_labels+dx_labels),
        "optimizer": (optimizer_region, (*pvalues, *mvalues, *gradients, *flag_values, *optimizer_data),
                      p_labels+[f"{optimizer_role}:{key}" for key in optimizer_names]+[f"gradient:{key}" for key in names]
                      +flag_labels+[f"input:{i}" for i in optimizer_inputs]),
    }
    for label, (function, arguments, labels) in specs.items():
        if conditional and label in ("body", "body_vjp"):
            continue
        graphs[label], modules[label] = _capture_region(function, arguments, label, labels,
                                                        round_trips=policy is not None)
    attention_tape = None
    if loop_attention not in ("auto", "off", "long"):
        raise CaptureError("LOOP_ATTENTION", "loop_attention must be 'auto', 'long' or 'off'")
    cast_keys, cast_report = [], None
    if cast_reuse and conditional:
        cast_report = {"selected": False, "reason": "conditional bodies keep per-use casts (branch unions not rewritten)"}
    if not conditional:
        joint_ir, joint_module = _capture_region(body_joint, specs["body_vjp"][1], "body_joint", specs["body_vjp"][2],
                                                 round_trips=policy is not None,
                                                 share_casts=policy is not None and SHARE_BODY_CASTS)
        joint_labels = specs["body_vjp"][2]
        if cast_reuse:
            # P-OPS4-F (c7bd29251), after AD: every region below is a PyTorch-derived forward or VJP graph.
            hoisted = {label: (modules[label], specs[label][2]) for label in ("body", "terminal_vjp", "body_vjp", "init_vjp")}
            hoisted["init"] = (modules["init"], p_labels + i_labels)
            hoisted["body_joint"] = (joint_module, specs["body_vjp"][2])
            cast_keys, cast_labels, cast_report = _hoist_invariant_casts(hoisted, n, p_labels + i_labels)
            cast_report["selected"] = bool(cast_keys)
            if cast_keys:
                for label, (gm, _) in hoisted.items():
                    if label != "body_joint":
                        graphs[label] = _import_region(gm, label, cast_labels[label])
                joint_labels = cast_labels["body_joint"]
                joint_ir = _import_region(joint_module, "body_joint", joint_labels)
                sources = [(p_labels + i_labels).index(source) for source, _ in cast_keys]
                dtypes = [dtype for _, dtype in cast_keys]

                def cast_region(*flat):
                    return tuple(flat[index].to(dtype) for index, dtype in zip(sources, dtypes))
                graphs["invariant_casts"], modules["invariant_casts"] = _capture_region(
                    cast_region, (*pvalues, *inputs), "invariant_casts", p_labels + i_labels)
        nw = len(cast_keys)
        forced_remat, forced_save = set(), set()
        if (loop_attention in ("auto", "long") and storage_mode == "save_body_tape"
                and tape_policy in ("precision_remat", "storage_remat")):
            # P-OPS4-E / P7-C: admitted short and long groups save O and row
            # statistics, never probabilities; reverse recomputes P from them.
            from .loop_attention import SCHEMA, rewrite_joint
            # P7-C long-context attention composes with P7-D storage decisions;
            # the admitted attention rewrite owns its required saved values.
            records, refusals, forced_remat, forced_save = rewrite_joint(
                joint_ir, joint_module, long=True)
            if records:
                joint_module.graph.eliminate_dead_code()
                joint_module.recompile()
                joint_ir = _import_region(joint_module, "body_joint", joint_labels)
            attention_tape = {"schema": SCHEMA, "groups": records, "refusals": refusals,
                              "selection": {"requested": loop_attention, "admitted": bool(records),
                                            "short_groups": sum(row.get("kernel") != "long" for row in records),
                                            "long_groups": sum(row.get("kernel") == "long" for row in records),
                                            "fallback": "ordinary captured primitives for inadmissible groups"},
                              "numerical_policy": {"attention": ("bf16_operands" if any(r.get("bf16_operands") for r in records)
                                                                 else "exact3"),
                                                   "splits": (lambda v: v[0] if len(v) == 1 else v)(
                                                       sorted({r["splits"] for r in records})),
                                                   "comparison": "tolerance-gated", "bitwise_fp32": False,
                                                   "long_resident_max_sequence": 512}}
            joint_ir.metadata["attention_tape"] = attention_tape
        else:
            attention_tape = {"schema": "none", "groups": [], "refusals": [
                {"reason": "loop attention requires save_body_tape with precision_remat or storage_remat"
                 if loop_attention != "off" else "loop_attention='off'"}]}
        cse_merged = []
        if tape_policy == "storage_remat" and (tape_options or {}).get("cse", False):
            cse_merged = _cse_joint(joint_module)
            if cse_merged:
                joint_ir = _import_region(joint_module, "body_joint", joint_labels)
                if attention_tape is not None:
                    joint_ir.metadata["attention_tape"] = attention_tape
        (graphs["body_saved"], modules["body_saved"], graphs["body_vjp_saved"], modules["body_vjp_saved"], tape) = _partition_body_tape(
            joint_ir, joint_module, n_primals=n+nw+nc+ns+nx+ni, n_forward_outputs=nc+ns, policy=tape_policy,
            forced_remat=forced_remat, forced_save=forced_save, tape_options=tape_options)
        if cse_merged:
            graphs["body_vjp_saved"].metadata["cse_merged"] = cse_merged
    else:
        branches = {}
        for branch, function in (("true", body_fn.true_fn), ("false", body_fn.false_fn)):
            active_body = function
            records = {}
            for label in ("body", "body_vjp"):
                fn, arguments, labels = specs[label]
                records[label] = _capture_region(fn, arguments, f"{label}_{branch}", labels)
            joint_ir, joint_module = _capture_region(body_joint, specs["body_vjp"][1], f"body_joint_{branch}", specs["body_vjp"][2])
            records["partition"] = _partition_body_tape(joint_ir, joint_module, n_primals=n+nc+ns+ni, n_forward_outputs=nc+ns, policy=tape_policy)
            branches[branch] = records
        selector_index = n + nc + state_names.index(body_fn.predicate_state)
        for label in ("body", "body_vjp"):
            graphs[label], modules[label] = _uniform_region(label, branches["true"][label][0], branches["false"][label][0],
                                                             branches["true"][label][1], branches["false"][label][1], selector_index)
        # A fixed typed union preserves one branch-independent loop ABI. Only
        # the selected branch computes activations/RNG; inactive tape slots are
        # deterministic zeros and the selected backward never reads them.
        tape = [{**desc, "name": f"{branch}_{desc['name']}", "branch": branch}
                for branch in ("true", "false") for desc in branches[branch]["partition"][4]]
        tape_values = tuple(torch.empty(desc["shape"], dtype=getattr(torch, desc["dtype"]), device=count.device) for desc in tape)
        ntrue = len(branches["true"]["partition"][4])
        nprimals = n+nc+ns+ni
        padded = {}
        for branch in ("true", "false"):
            forward_module, backward_module = branches[branch]["partition"][1], branches[branch]["partition"][3]
            start, stop = (0, ntrue) if branch == "true" else (ntrue, len(tape))
            def padded_forward(*flat, _gm=forward_module, _start=start, _stop=stop):
                result = _gm(*flat)
                local_tape = iter(result[nc+ns:])
                union = tuple(next(local_tape) if _start <= i < _stop else
                              torch.zeros(desc["shape"], dtype=getattr(torch, desc["dtype"]), device=flat[0].device)
                              for i, desc in enumerate(tape))
                return (*result[:nc+ns], *union)
            def padded_backward(*flat, _gm=backward_module, _start=start, _stop=stop):
                return _gm(*flat[:nprimals], *flat[nprimals+_start:nprimals+_stop], *flat[nprimals+len(tape):])
            fwd = _capture_region(padded_forward, specs["body"][1], f"body_saved_{branch}", specs["body"][2])
            backward_args = (*specs["body"][1], *tape_values, *cotangents)
            backward_labels = [*specs["body"][2], *(f"tape:{desc['name']}" for desc in tape), *dc_labels]
            rev = _capture_region(padded_backward, backward_args, f"body_vjp_saved_{branch}", backward_labels)
            padded[branch] = {"body_saved": fwd, "body_vjp_saved": rev}
        for label in ("body_saved", "body_vjp_saved"):
            graphs[label], modules[label] = _uniform_region(label, padded["true"][label][0], padded["false"][label][0],
                                                             padded["true"][label][1], padded["false"][label][1], selector_index)
        active_body = body_fn

    init_tape = []
    init_window = None
    if init_storage == "save_tape":
        def init_joint(*flat):
            p, data = tuple(flat[:n]), tuple(flat[n:n+ni])
            dc, dx = tuple(flat[n+ni:n+ni+nc]), tuple(flat[n+ni+nc:])
            def combined(pp):
                xs = sequences(pp, *data) if sequenced else ()
                return (initial(pp, *data), tuple(xs[i] for i in xdiff)), xs
            (carry, _), pullback, xs = vjp(combined, p, has_aux=True)
            (dp,) = pullback((dc, dx))
            return tuple(value.clone() for value in (*carry, *xs, *dp))
        init_ir, init_module = _capture_region(init_joint, specs["init_vjp"][1], "init_joint", specs["init_vjp"][2],
                                               round_trips=policy is not None)
        if window_attention in ("auto", True) and policy is not None:
            # P8-C: rectangular mixed attention of the init/sequence joint graph becomes opaque operations
            # whose tape holds row statistics and BF16 operands instead of probabilities.
            from .window_attention import SCHEMA as WINDOW_SCHEMA, rewrite_joint as window_rewrite
            window_records, window_refusals = window_rewrite(
                init_module, arithmetic='online' if getattr(policy, 'attention', None) == 'bfloat16_online' else 'exact')
            fusion = {"schema": "none", "groups": [], "refusals": []}
            if window_records and window_fusion in ("auto", True):
                # P9-C: rotation / block-shift producers and their derivatives become operand loads and gradient
                # stores of the attention operations (ops/window_fusion.py).
                from .window_fusion import SCHEMA as FUSION_SCHEMA, rewrite_joint as fusion_rewrite
                fusion_records, fusion_refusals = fusion_rewrite(init_module)
                fusion = {"schema": FUSION_SCHEMA, "groups": fusion_records, "refusals": fusion_refusals}
            if window_records:
                init_ir = _import_region(init_module, "init_joint", specs["init_vjp"][2])
            init_window = {"schema": WINDOW_SCHEMA, "groups": window_records, "refusals": window_refusals,
                           "fusion": fusion}
        else:
            init_window = {"schema": "none", "groups": [], "refusals": [{"reason": "window_attention off or no precision policy"}]}
        (graphs["init_saved"], modules["init_saved"], graphs["init_vjp_saved"], modules["init_vjp_saved"],
         init_tape) = _partition_body_tape(init_ir, init_module, n_primals=n+ni, n_forward_outputs=nc+nx,
                                           policy=tape_policy if tape_policy != "stage3" else "pointwise_remat")
        graphs["init_saved"].metadata["name"], graphs["init_vjp_saved"].metadata["name"] = "init_saved", "init_vjp_saved"

    connectivity = None
    if prepared.uses_participation:
        # Discover autograd connectivity after every branch has passed effect
        # validation. These booleans describe graph edges, never gradient values.
        probes = tuple(torch.zeros_like(value) for value in cvalues)
        connectivity = {"terminal": list(discover_gradient_participation(
            lambda pc: terminal(pc[:n], pc[n:], *inputs), (*pvalues, *probes))),
            "init": [list(discover_gradient_participation(lambda p, j=j: initial(p, *inputs)[j], pvalues)) for j in range(nc)]}
        def body_connectivity(function):
            nonlocal active_body
            active_body = function
            return [list(discover_gradient_participation(
                lambda pc, j=j: body(pc[:n], pc[n:], svalues, *inputs)[0][j], (*pvalues, *probes))) for j in range(nc)]
        connectivity["body"] = ({branch: body_connectivity(fn) for branch, fn in (("true", body_fn.true_fn), ("false", body_fn.false_fn))}
                                if conditional else body_connectivity(body_fn))
        active_body = body_fn

    def step(*flat):
        p, m = tuple(flat[:n]), tuple(flat[n:n+no])
        s, data, runtime_count = tuple(flat[n+no:n+no+ns]), tuple(flat[n+no+ns:-1]), flat[-1]
        requested = int(runtime_count.item())
        steps = min(max(requested, 0), max_steps)
        def objective(pp):
            carry, active_state = initial(pp, *data), s
            xs = sequences(pp, *data) if sequenced else ()
            for t in range(steps):
                carry, active_state = body(pp, carry, active_state, *data, xflat=tuple(x[t] for x in xs))
            return terminal(pp, carry, *data), (active_state, carry)
        grad, (loss, (new_state, final_carry)) = grad_and_value(objective, has_aux=True)(p)
        flags = discover_gradient_participation(lambda pp: objective(pp)[0], p) if prepared.uses_participation else None
        extra = tuple(data[i] for i in optimizer_inputs)
        new_p, new_m = (prepared.update(p, grad, m, participation=flags, inputs=extra) if optimizer_inputs
                        else prepared.update(p, grad, m, participation=flags))
        updated = (*new_p, *new_m)
        return (loss, *grad, *updated[:n], *updated[n:],
                *(value.clone() for value in new_state), *(value.clone() for value in final_carry),
                runtime_count.new_tensor(steps), runtime_count.new_tensor(int(requested != steps)))

    external_values, input_names = {}, []
    for index, (tensor, label) in enumerate(zip(external, external_labels)):
        name = f"loop_arg_{index}"
        role, _, parameter = label.partition(":")
        external_values[name] = Value(name, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."),
                                     tuple(tensor.stride()), role=role, parameter=parameter or None)
        input_names.append(name)
    outputs, out_index = {}, 0
    def output(spec, role):
        nonlocal out_index
        name = f"loop_result_{out_index}"
        out_index += 1
        shape, dtype = spec
        external_values[name] = Value(name, tuple(shape), dtype, _contiguous_strides(shape), role="loop_output", output_roles=(role,))
        return name
    outputs["loss"] = output(((), "float32"), "loss")
    for group in ("grads", "params"):
        outputs[group] = {key: output((tensor.shape, "float32"), f"{group}:{key}") for key, tensor in zip(names, pvalues)}
    outputs[optimizer_role] = {key: output((tensor.shape, str(tensor.dtype).removeprefix("torch.")), f"{optimizer_role}:{key}")
                               for key, tensor in zip(optimizer_names, mvalues)}
    outputs["state"] = {key: output((tensor.shape, str(tensor.dtype).removeprefix("torch.")), f"state:{key}") for key, tensor in zip(state_names, svalues)}
    outputs["carry"] = {key: output((tensor.shape, str(tensor.dtype).removeprefix("torch.")), f"carry:{key}")
                        for key, tensor in zip(carry_names, cvalues)}
    outputs["executed_count"], outputs["status"] = output(((), "int64"), "executed_count"), output(((), "int64"), "status")
    metadata = {"parameter_names": list(names), "parameter_aliases": aliases, "carry_names": carry_names,
                "state_names": list(state_names), "n_inputs": ni, "torch_version": torch.__version__,
                **prepared.metadata(),
                "count_policy": "clamp to [0,max_steps]; status=1 iff requested count was clamped",
                "derivatives": "terminal parameter gradient + reverse-loop shared parameter VJPs + initial VJP; all floating carry differentiated",
                "body_tape": tape, "tape_policy": tape_policy,
                "memory_selection": {"requested": memory_policy, "storage_mode": storage_mode,
                                     "tape_policy": tape_policy, "cast_reuse": cast_reuse,
                                     "tape_options": dict(tape_options or {})},
                "optimizer_inputs": list(optimizer_inputs),
                "init_storage": init_storage, "init_tape": init_tape,
                **({"init_attention_tape": init_window} if init_window is not None else {}),
                **({"sequence_names": list(sequence_names),
                    "sequence_specs": [{"name": key, "shape": list(shape), "dtype": str(dtype).removeprefix("torch."),
                                        "differentiable": i in xdiff}
                                       for i, (key, (shape, dtype, _)) in enumerate(zip(sequence_names, sequence_specs))],
                    "sequence_policy": "evaluated once with init; body t reads slice t; reverse step t writes the cotangent "
                                       "of slice t; unexecuted slices have zero cotangent; init_vjp differentiates init and "
                                       "sequences together"} if sequenced else {}),
                **({"attention_tape": attention_tape} if attention_tape is not None else {}),
                "tape_selection": {"requested": requested_tape_policy, "selected": tape_policy,
                                   **({"tape_options": dict(tape_options)} if tape_options else {}),
                                   "reason": "P7-D bytes-driven rematerialization over the precision_remat tape" if tape_policy == "storage_remat"
                                             else "flat saved-body exact pointwise and float-cast rematerialization" if tape_policy == "precision_remat"
                                             else "flat saved-body exact pointwise rematerialization" if tape_policy == "pointwise_remat"
                                             else "reference partition for explicit choice or nested/recomputed regions"},
                "body_tape_policy": "save_body_tape saves material forward cut values; alias/view and cheap constant/copy nodes rematerialize; other modes recompute local forward activations",
                "storage_modes": list(STORAGE_MODES), "state_commit": "new parameters, momentum and final explicit state committed once after reverse loop"}
    if policy is not None:
        metadata["precision_policy"] = {**policy.describe(),
            "covered_calls_all_region_traces": policy_record.summary()["covered_calls"]}
    if connectivity is not None:
        metadata["gradient_connectivity"] = connectivity
    if not cast_reuse:
        cast_reason = ("explicit cast_reuse=False" if requested_cast_reuse is False else
                       "automatic cast reuse requires a plain saved-body loop without sequences and with recomputed init")
    elif conditional:
        cast_reason = "conditional bodies keep per-use casts; branch unions are not rewritten"
    elif cast_keys:
        cast_reason = "eligible exact casts of update-invariant parameter/data bindings are computed once per update"
    else:
        cast_reason = "pass enabled, but no exact parameter/data binding cast satisfies its layout and dtype admission"
    metadata["cast_reuse"] = {**(cast_report or {}), "requested": requested_cast_reuse,
            "enabled": cast_reuse, "selected": bool(cast_keys), "reason": cast_reason,
            "selection_mode": "low_memory" if memory_policy == "low_memory" else
                              "automatic" if requested_cast_reuse is None else "explicit",
            "policy": "exact float casts of parameter/data bindings computed once per launch before init and read "
                      "by every non-optimizer region; refreshed every update from the bindings it reads; "
                      "per-use cotangent widening and FP32 accumulation unchanged"}
    if cast_keys:
        bound = (*pvalues, *inputs)
        metadata["invariant_casts"] = [
            {"name": f"invariant_cast_{index}", "source": source, "dtype": str(dtype).removeprefix("torch."),
             "shape": list(bound[(p_labels + i_labels).index(source)].shape)}
            for index, (source, dtype) in enumerate(cast_keys)]
    if conditional:
        metadata["conditional_body"] = {"predicate_state": body_fn.predicate_state,
                                         "policy": "selected whole-grid body and matching old-boundary VJP; no predicate derivative",
                                         "tape_policy": "typed branch union; inactive slots zero; selected VJP reads its own tape"}
    if cast_reuse and not conditional:
        _check_invariant_cast_abi(graphs, metadata)
    ir = BoundedLoopIR(**graphs, external_values=external_values, inputs=tuple(input_names), outputs=outputs,
                       count=external_values[input_names[-1]], max_steps=max_steps, storage_mode=storage_mode, metadata=metadata)
    return CapturedLoop(ir, modules, tuple(external), step, config)
