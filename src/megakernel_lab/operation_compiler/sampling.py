"""Training with a nondifferentiated device sampling loop (C-OPS4-B).

Some learning procedures draw samples by iterating a model-defined update whose result then enters the loss as
a constant: short-run Langevin / contrastive-divergence energy models, persistent chains, sampling-based
negatives. The sampling iterations may use derivatives themselves (an input gradient), but the outer parameter
gradient must stop at the samples. This explicit region API expresses that boundary:

    init_fn(params, state, *inputs)            -> carry dict                      (no derivative)
    step_fn(params, carry, state, k, *inputs)  -> (new carry dict, new state dict) (no outer derivative)
    loss_fn(params, carry, state, *inputs)     -> (scalar FP32 loss, new state dict)

``k`` is the iteration index as a scalar int64 device tensor. The step runs a runtime count of times, clamped
to ``[0, max_steps]`` (status reports clamping), with every parameter at its pre-update value; ``step_fn`` may
call ``torch.func.grad`` with respect to its carry (an input gradient), never with respect to parameters.
The loss is differentiated with respect to the parameters only; carry and state are constants there. The
optimizer then updates parameters once and the final explicit state is committed after every read.

PyTorch (make_fx over each region, grad_and_value for the loss) derives every operation; lowering composes
them (ops.sampling_lower). There are no model names. Region capture follows ops.loop / ops.inference
(capture_loop_inference, C-OPS-I f121882295) and reuses P-OPS4-F invariant-cast hoisting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Callable

import torch
from torch.func import grad_and_value

from .capture import _reject_lazy_view, _reject_placeholder_storage_alias, discover_gradient_participation
from .ir import CaptureError, Value, _contiguous_strides
from .loop import OperationIR, _hoist_invariant_casts, _import_region

REGIONS = ("init", "body", "train")
_SHAPE_ONLY = ("ones_like", "zeros_like", "full_like")


def _shape_only_constants(gm):
    """Replace ones_like/zeros_like/full_like of a contiguous template by full(shape, value).

    These constructors use only the template's shape, dtype and device, but as graph inputs they keep the
    template's producer alive. PyTorch's grad of sum(E) seeds ones_like(sum), so without this rewrite the
    discarded forward value of an input-gradient step would still be computed. Values are unchanged."""
    changed = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or not str(node.target).startswith(tuple(f"aten.{b}." for b in _SHAPE_ONLY)):
            continue
        value = node.meta.get("val")
        template = node.args[0].meta.get("val") if node.args and isinstance(node.args[0], torch.fx.Node) else None
        if not isinstance(value, torch.Tensor) or not isinstance(template, torch.Tensor) or not value.is_contiguous():
            continue
        base = str(node.target).split(".")[1]
        fill = 1 if base == "ones_like" else 0 if base == "zeros_like" else node.args[1]
        if not isinstance(fill, (int, float, bool)):
            continue
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(torch.ops.aten.full.default, (list(value.shape), fill),
                                         {"dtype": value.dtype, "layout": torch.strided, "device": value.device,
                                          "pin_memory": False})
        new.meta = dict(node.meta)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        changed += 1
    return changed


def _capture_region(function, arguments, label, labels):
    """ops.loop._capture_region plus the shape-only constant rewrite before dead-code elimination."""
    from torch.fx.experimental.proxy_tensor import make_fx
    from .capture import _functionalize_internal_mutations, capture_decompositions
    from .standard_ops import capture_functions
    try:
        with capture_functions():
            gm = make_fx(function, tracing_mode="fake", _allow_non_fake_inputs=False,
                         _error_on_data_dependent_ops=True, decomposition_table=capture_decompositions())(*arguments)
    except CaptureError:
        raise
    except Exception as exc:
        code = ("SAMPLED_RUNTIME_CONTROL" if any(text in str(exc).lower() for text in
                                                 ("data-dependent", "data dependent", "_local_scalar_dense"))
                else "SAMPLED_TRACE")
        raise CaptureError(code, f"{label}: {type(exc).__name__}: {exc}") from exc
    gm = _functionalize_internal_mutations(gm, arguments)
    rewritten = _shape_only_constants(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    region = _import_region(gm, label, labels)
    region.metadata["shape_only_constants_rewritten"] = rewritten
    return region, gm
STATE_DTYPES = (torch.float32, torch.int64, torch.bool)


@dataclass
class SampledTrainingIR:
    init: OperationIR
    body: OperationIR
    train: OperationIR
    external_values: dict[str, Value]
    inputs: tuple[str, ...]
    outputs: dict[str, Any]
    count: Value
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)
    invariant_casts: OperationIR | None = None

    @property
    def values(self):
        return self.external_values

    @property
    def regions(self):
        names = (("invariant_casts",) if self.invariant_casts is not None else ()) + REGIONS
        return {name: getattr(self, name) for name in names}

    def output_names(self):
        m = self.metadata
        result = [self.outputs["loss"]]
        for group in ("grads", "params"):
            result.extend(self.outputs[group][name] for name in m["parameter_names"])
        result.extend(self.outputs["optimizer_state"][name] for name in m["optimizer_state_names"])
        result.extend(self.outputs["state"][name] for name in m["state_names"])
        result.extend(self.outputs["carry"][name] for name in m["carry_names"])
        return (*result, self.outputs["executed_count"], self.outputs["status"])

    def to_dict(self):
        return {"format_version": 6, "kind": "sampled_training",
                "regions": {key: graph.to_dict() for key, graph in self.regions.items()},
                "external_values": {key: asdict(value) for key, value in self.external_values.items()},
                "inputs": list(self.inputs), "outputs": self.outputs, "count": asdict(self.count),
                "max_steps": self.max_steps, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data):
        if data.get("format_version") != 6 or data.get("kind") != "sampled_training":
            raise CaptureError("SAMPLED_VERSION", "unknown sampled-training format")
        if type(data["max_steps"]) is not int or not 0 <= data["max_steps"] <= 4096:
            raise CaptureError("SAMPLED_BOUND", "invalid saved sampling bound")

        def value(entry):
            entry = dict(entry)
            for key in ("shape", "strides", "output_roles"):
                entry[key] = tuple(entry[key])
            return Value(**entry)
        regions = {key: OperationIR.from_dict(raw) for key, raw in data["regions"].items()}
        return cls(regions["init"], regions["body"], regions["train"],
                   external_values={key: value(raw) for key, raw in data["external_values"].items()},
                   inputs=tuple(data["inputs"]), outputs=data["outputs"], count=value(data["count"]),
                   max_steps=data["max_steps"], metadata=data["metadata"],
                   invariant_casts=regions.get("invariant_casts"))


@dataclass
class CapturedSampled:
    ir: SampledTrainingIR
    graph_modules: dict[str, torch.fx.GraphModule]
    example_args: tuple[torch.Tensor, ...]
    step_fn: Callable
    config: Any

    @property
    def parameter_names(self):
        return tuple(self.ir.metadata["parameter_names"])

    @property
    def optimizer_state_names(self):
        return tuple(self.ir.metadata["optimizer_state_names"])

    @property
    def state_names(self):
        return tuple(self.ir.metadata["state_names"])

    @property
    def carry_names(self):
        return tuple(self.ir.metadata["carry_names"])

    def unpack(self, values):
        n, no = len(self.parameter_names), len(self.optimizer_state_names)
        ns, nc = len(self.state_names), len(self.carry_names)
        if len(values) != 1 + 2 * n + no + ns + nc + 2:
            raise ValueError("sampled-training output arity differs from its ABI")
        return {"loss": values[0], "grads": dict(zip(self.parameter_names, values[1:1 + n])),
                "params": dict(zip(self.parameter_names, values[1 + n:1 + 2 * n])),
                "optimizer_state": dict(zip(self.optimizer_state_names, values[1 + 2 * n:1 + 2 * n + no])),
                "state": dict(zip(self.state_names, values[1 + 2 * n + no:1 + 2 * n + no + ns])),
                "carry": dict(zip(self.carry_names, values[1 + 2 * n + no + ns:-2])),
                "executed_count": values[-2], "status": values[-1]}

    def graph_step(self, *flat):
        """CPU interpreter of the captured regions in native order (diagnostic, not an independent reference)."""
        m = self.ir.metadata
        n, no, ns = len(self.parameter_names), len(self.optimizer_state_names), len(self.state_names)
        nc = len(self.carry_names)
        params, opt = tuple(flat[:n]), tuple(flat[n:n + no])
        state, data, count = list(flat[n + no:n + no + ns]), tuple(flat[n + no + ns:-1]), flat[-1]
        requested = int(count.item())
        steps = min(max(requested, 0), self.ir.max_steps)
        g = self.graph_modules
        casts = tuple(g["invariant_casts"](*params, *data)) if "invariant_casts" in g else ()
        pw = (*params, *casts)
        carry = tuple(g["init"](*pw, *state, *data))
        changed = [m["state_names"].index(key) for key in m["body_state_changes"]]
        for k in range(steps):
            result = g["body"](*pw, *carry, *state, count.new_tensor(k), *data)
            carry = tuple(result[:nc])
            for index, value in zip(changed, result[nc:]):
                state[index] = value
        result = g["train"](*pw, *opt, *carry, *state, *data)
        loss, grads = result[0], tuple(result[1:1 + n])
        new_params, new_opt = tuple(result[1 + n:1 + 2 * n]), tuple(result[1 + 2 * n:1 + 2 * n + no])
        new_state = tuple(result[1 + 2 * n + no:])
        return (loss, *grads, *new_params, *new_opt, *new_state, *(c.clone() for c in carry),
                count.new_tensor(steps), count.new_tensor(int(requested != steps)))


def capture_sampled_training(init_fn, step_fn, loss_fn, params, inputs, count, config, *, max_steps,
                             state=None, optimizer_state=None, cast_reuse=True):
    """Capture the init / sampling-step / training regions of one complete update (CPU worker)."""
    if torch.__version__.split("+")[0] != "2.9.1":
        raise CaptureError("TORCH_VERSION", f"sampled capture requires torch 2.9.1, got {torch.__version__}")
    if type(max_steps) is not int or not 0 <= max_steps <= 4096:
        raise CaptureError("SAMPLED_BOUND", "require static max_steps in [0, 4096]")
    if not isinstance(count, torch.Tensor) or count.dtype != torch.int64 or count.ndim != 0:
        raise CaptureError("SAMPLED_COUNT", "runtime count must be a scalar int64 Tensor")
    _reject_lazy_view(count, "sampling count")
    from .optim import AdamWConfig, SGDConfig, prepare_optimizer
    if not isinstance(config, (AdamWConfig, SGDConfig)):
        raise CaptureError("OPTIMIZER", "expected SGDConfig or AdamWConfig")
    config.validate()
    if getattr(config, "runtime_lr", False):
        raise CaptureError("OPTIMIZER_ABI", "sampled AdamW currently requires a static learning rate; "
                           "runtime_lr is admitted by capture_training/capture_module")
    unique, aliases, identities = {}, {}, {}
    for name, value in params.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not value.is_contiguous():
            raise CaptureError("PARAMETER", f"{name}: parameters must be contiguous FP32 tensors")
        _reject_lazy_view(value, f"parameter {name}")
        canonical = identities.setdefault(id(value), name)
        aliases[name] = canonical
        unique.setdefault(canonical, value)
    if not unique:
        raise CaptureError("PARAMETERS", "at least one trainable parameter is required")
    names = tuple(unique)
    alias_index = {key: names.index(value) for key, value in aliases.items()}
    prepared = prepare_optimizer(config, unique, optimizer_state=optimizer_state, aliases=aliases)
    optimizer_names = prepared.state_names
    role = "momentum" if prepared.legacy_momentum else "optimizer_state"
    pvalues, mvalues = tuple(v.detach() for v in unique.values()), prepared.state_values
    state = dict(state or {})
    state_names = tuple(state)
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.dtype not in STATE_DTYPES or not value.is_contiguous()
                or value.layout != torch.strided):
            raise CaptureError("SAMPLED_STATE", f"state {name}: contiguous FP32/int64/bool tensors only")
        _reject_lazy_view(value, f"state {name}")
    svalues, inputs = tuple(v.detach() for v in state.values()), tuple(inputs)
    for index, value in enumerate(inputs):
        if not isinstance(value, torch.Tensor):
            raise CaptureError("INPUT", f"input {index} must be a Tensor")
        _reject_lazy_view(value, f"input {index}")
    external = (*pvalues, *mvalues, *svalues, *inputs, count.detach())
    external_labels = ([f"parameter:{k}" for k in names] + [f"{role}:{k}" for k in optimizer_names]
                       + [f"state:{k}" for k in state_names] + [f"input:{i}" for i in range(len(inputs))] + ["count"])
    _reject_placeholder_storage_alias(external, external_labels)
    if any(value.device != count.device for value in external):
        raise CaptureError("SAMPLED_DEVICE", "every input and the count must be on one device")
    n, no, ns, ni = len(names), len(optimizer_names), len(state_names), len(inputs)
    carry_names, carry_specs, body_changes = [], [], []

    def mapping(flat):
        return {alias: flat[index] for alias, index in alias_index.items()}

    def checked_state(before, after, label):
        if not isinstance(after, dict) or set(after) != set(state_names):
            raise CaptureError("SAMPLED_STATE", f"{label} must return a state dict with unchanged keys")
        for key in state_names:
            new, old = after[key], before[key]
            if not isinstance(new, torch.Tensor) or (new.shape, new.dtype) != (old.shape, old.dtype):
                raise CaptureError("SAMPLED_STATE", f"{label} state {key} changed shape or dtype")
        return after

    def checked_carry(result, *, initial=False):
        if not isinstance(result, dict) or not result:
            raise CaptureError("SAMPLED_CARRY", "carry must be a nonempty dictionary of tensors")
        if initial and not carry_names:
            carry_names.extend(result)
            for key in carry_names:
                value = result[key]
                if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.bfloat16, torch.int64):
                    raise CaptureError("SAMPLED_CARRY", f"carry {key}: FP32/BF16/int64 tensors only")
                carry_specs.append((tuple(value.shape), value.dtype))
        if set(result) != set(carry_names):
            raise CaptureError("SAMPLED_CARRY", "every iteration must preserve carry keys")
        for key, spec in zip(carry_names, carry_specs):
            value = result[key]
            if not isinstance(value, torch.Tensor) or (tuple(value.shape), value.dtype) != spec:
                raise CaptureError("SAMPLED_CARRY", f"carry {key} changed shape or dtype")
        return tuple(result[key] for key in carry_names)

    def initial(pflat, sflat, data):
        with torch.no_grad():
            return checked_carry(init_fn(mapping(pflat), dict(zip(state_names, sflat)), *data), initial=True)

    def body(pflat, cflat, sflat, k, data):
        old = dict(zip(state_names, sflat))
        result = step_fn(mapping(pflat), dict(zip(carry_names, cflat)), old, k, *data)
        if not isinstance(result, tuple) or len(result) != 2:
            raise CaptureError("SAMPLED_STEP", "step must return (new_carry, new_state)")
        return checked_carry(result[0]), checked_state(old, result[1], "step")

    def objective(pflat, cflat, sflat, data):
        old = dict(zip(state_names, sflat))
        result = loss_fn(mapping(pflat), dict(zip(carry_names, cflat)), old, *data)
        if not isinstance(result, tuple) or len(result) != 2:
            raise CaptureError("SAMPLED_LOSS", "loss must return (scalar_loss, new_state)")
        loss, new_state = result
        if not isinstance(loss, torch.Tensor) or loss.dtype != torch.float32 or loss.ndim != 0:
            raise CaptureError("LOSS", "sampled training loss must be a scalar FP32 Tensor")
        checked_state(old, new_state, "loss")
        return loss, tuple(new_state[key] for key in state_names)

    started = time.perf_counter()
    p_labels = [f"parameter:{k}" for k in names]
    m_labels = [f"{role}:{k}" for k in optimizer_names]
    s_labels = [f"state:{k}" for k in state_names]
    i_labels = [f"input:{i}" for i in range(ni)]
    graphs, modules, labels = {}, {}, {}

    def init_region(*flat):
        return tuple(v.clone() for v in initial(flat[:n], flat[n:n + ns], flat[n + ns:]))

    labels["init"] = p_labels + s_labels + i_labels
    graphs["init"], modules["init"] = _capture_region(init_region, (*pvalues, *svalues, *inputs), "init",
                                                      labels["init"])
    nc = len(carry_names)
    cvalues = tuple(torch.empty(shape, dtype=dtype, device=count.device) for shape, dtype in carry_specs)
    c_labels = [f"carry:{k}" for k in carry_names]
    kvalue = count.new_zeros(())

    def probe_body(*flat):
        # Classify returned state: identity (the same Tensor object) is read-only for the whole loop.
        carry, new_state = body(flat[:n], flat[n:n + nc], flat[n + nc:n + nc + ns], flat[n + nc + ns],
                                flat[n + nc + ns + 1:])
        body_changes[:] = [k for k, old in zip(state_names, flat[n + nc:n + nc + ns]) if new_state[k] is not old]
        return tuple(v.clone() for v in (*carry, *(new_state[k] for k in body_changes)))

    labels["body"] = p_labels + c_labels + s_labels + ["step"] + i_labels
    graphs["body"], modules["body"] = _capture_region(probe_body, (*pvalues, *cvalues, *svalues, kvalue, *inputs),
                                                      "body", labels["body"])
    participation = tuple(discover_gradient_participation(
        lambda pf, *rest: objective(pf, rest[:nc], rest[nc:nc + ns], rest[nc + ns:])[0], pvalues,
        (*cvalues, *svalues, *inputs))) if not prepared.legacy_momentum else None

    def train_region(*flat):
        p, mo = tuple(flat[:n]), tuple(flat[n:n + no])
        carry, sflat, data = flat[n + no:n + no + nc], flat[n + no + nc:n + no + nc + ns], flat[n + no + nc + ns:]
        carry = tuple(c.detach() for c in carry)
        sflat = tuple(s.detach() for s in sflat)
        grads, (loss, new_state) = grad_and_value(
            lambda pp: objective(pp, carry, sflat, data), has_aux=True)(p)
        new_p, new_m = prepared.update(p, grads, mo, participation=participation)
        return tuple(v.clone() for v in (loss, *grads, *new_p, *new_m, *new_state))

    labels["train"] = p_labels + m_labels + c_labels + s_labels + i_labels
    graphs["train"], modules["train"] = _capture_region(
        train_region, (*pvalues, *mvalues, *cvalues, *svalues, *inputs), "train", labels["train"])
    cast_keys, cast_report = [], {"requested": bool(cast_reuse), "selected": False}
    if cast_reuse:
        hoisted = {label: (modules[label], labels[label]) for label in REGIONS}
        cast_keys, cast_labels, report = _hoist_invariant_casts(hoisted, n, p_labels + i_labels)
        cast_report.update(report, selected=bool(cast_keys))
        if cast_keys:
            for label in REGIONS:
                rewritten = graphs[label].metadata.get("shape_only_constants_rewritten", 0)
                graphs[label] = _import_region(modules[label], label, cast_labels[label])
                graphs[label].metadata["shape_only_constants_rewritten"] = rewritten
            sources = [(p_labels + i_labels).index(source) for source, _ in cast_keys]
            dtypes = [dtype for _, dtype in cast_keys]

            def cast_region(*flat):
                return tuple(flat[index].to(dtype) for index, dtype in zip(sources, dtypes))
            graphs["invariant_casts"], modules["invariant_casts"] = _capture_region(
                cast_region, (*pvalues, *inputs), "invariant_casts", p_labels + i_labels)

    def reference_step(*flat):
        """Conventional eager reference: ordinary PyTorch calls of the user functions, autograd and optimizer."""
        p, mo = tuple(flat[:n]), tuple(flat[n:n + no])
        s, data, runtime_count = tuple(flat[n + no:n + no + ns]), tuple(flat[n + no + ns:-1]), flat[-1]
        requested = int(runtime_count.item())
        steps = min(max(requested, 0), max_steps)
        detached = tuple(v.detach() for v in p)
        carry = initial(detached, s, data)
        current = dict(zip(state_names, s))
        for k in range(steps):
            carry, current = body(detached, carry, tuple(current[key] for key in state_names),
                                  runtime_count.new_tensor(k), data)
        carry = tuple(c.detach() for c in carry)
        sflat = tuple(current[key].detach() for key in state_names)
        grads, (loss, new_state) = grad_and_value(lambda pp: objective(pp, carry, sflat, data), has_aux=True)(p)
        new_p, new_m = prepared.update(p, grads, mo, participation=participation)
        return (loss, *grads, *new_p, *new_m, *(v.clone() for v in new_state), *(c.clone() for c in carry),
                runtime_count.new_tensor(steps), runtime_count.new_tensor(int(requested != steps)))

    external_values, input_names = {}, []
    for index, (tensor, label) in enumerate(zip(external, external_labels)):
        name = f"sampled_arg_{index}"
        kind, _, key = label.partition(":")
        external_values[name] = Value(name, tuple(tensor.shape), str(tensor.dtype).removeprefix("torch."),
                                      tuple(tensor.stride()), role=kind, parameter=key or None)
        input_names.append(name)
    outputs, counter = {}, [0]

    def output(shape, dtype, label):
        name = f"sampled_result_{counter[0]}"
        counter[0] += 1
        external_values[name] = Value(name, tuple(shape), dtype, _contiguous_strides(tuple(shape)),
                                      role="loop_output", output_roles=(label,))
        return name

    dt = lambda t: str(t.dtype).removeprefix("torch.")
    outputs["loss"] = output((), "float32", "loss")
    for group in ("grads", "params"):
        outputs[group] = {k: output(t.shape, "float32", f"{group}:{k}") for k, t in zip(names, pvalues)}
    outputs["optimizer_state"] = {k: output(t.shape, dt(t), f"{role}:{k}") for k, t in zip(optimizer_names, mvalues)}
    outputs["state"] = {k: output(t.shape, dt(t), f"state:{k}") for k, t in zip(state_names, svalues)}
    outputs["carry"] = {k: output(t.shape, dt(t), f"carry:{k}") for k, t in zip(carry_names, cvalues)}
    outputs["executed_count"] = output((), "int64", "executed_count")
    outputs["status"] = output((), "int64", "status")
    metadata = {"parameter_names": list(names), "parameter_aliases": aliases, "carry_names": list(carry_names),
                "state_names": list(state_names), "n_inputs": ni, "torch_version": torch.__version__,
                **prepared.metadata(), "optimizer_role": role,
                "body_state_changes": list(body_changes),
                "gradient_participation": list(participation) if participation is not None else None,
                "count_policy": "clamp to [0,max_steps]; status=1 iff the requested count was clamped",
                "derivatives": "parameter gradient of the loss region only; init and step regions are constants "
                               "for it (pre-update parameters, no parameter derivative operations)",
                "capture": "explicit init/step/train regions; make_fx(fake); grad_and_value for the train region",
                "state_commit": "parameters, optimizer state and final explicit state committed once after the "
                                "train region; step-changed state is double buffered, unchanged state read in place",
                "cast_reuse": cast_report,
                "capture_seconds": time.perf_counter() - started}
    if cast_keys:
        bound = (*pvalues, *inputs)
        metadata["invariant_casts"] = [
            {"name": f"invariant_cast_{index}", "source": source, "dtype": str(dtype).removeprefix("torch."),
             "shape": list(bound[(p_labels + i_labels).index(source)].shape)}
            for index, (source, dtype) in enumerate(cast_keys)]
    ir = SampledTrainingIR(graphs["init"], graphs["body"], graphs["train"], external_values, tuple(input_names),
                           outputs, external_values[input_names[-1]], max_steps, metadata,
                           invariant_casts=graphs.get("invariant_casts"))
    return CapturedSampled(ir, modules, tuple(external), reference_step, config)
