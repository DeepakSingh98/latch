"""Forward-only inference through the same operation compiler (C-OPS-I).

Adapted from pinned f121882295ecdcf094f8727e8d72f15fc51199ba.

Capture traces only the forward computation: no loss, automatic differentiation,
optimizer or zero-learning-rate step. The resulting operation graph uses the
existing IR values/operations and is lowered by the unmodified ``lower`` (flat
programs) or by composing ``lower`` regions under a device-count loop
(``inference_lower``). Parameters are read-only bindings; only explicit state is
committed, after all reads. Capture and nvcc belong on a CPU build worker;
``Inference.load`` binds a prebuilt artifact on a CUDA worker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch.func import functional_call
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree

from .api import _copy_bindings, _hash, _immutable_identity, _storage_key
from .capture import (_ALIASES, _functionalize_internal_mutations, _reject_lazy_view,
                      _reject_placeholder_storage_alias, capture_decompositions, hoist_tensor_constants)
from .ir import CaptureError, Operation, TrainingIR, Value
from .loop import OperationIR, _capture_region, _import_region

STORAGE_DTYPES = {torch.float32, torch.bfloat16, torch.int64, torch.bool, torch.float64}
ARTIFACT_SCHEMA = 'latch-ops-inference-artifact-v1'


def _check_version():
    if torch.__version__.split('+')[0] != '2.9.1':
        raise CaptureError('TORCH_VERSION', f'inference capture requires torch 2.9.1; got {torch.__version__}')


def _check_tensor(value, label, *, contiguous=False):
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise CaptureError('TENSOR', f'{label} must be a strided Tensor')
    if value.dtype not in STORAGE_DTYPES:
        raise CaptureError('DTYPE', f'{label}: dtype {value.dtype} has no qualified storage representation')
    if value.dtype == torch.float64 and value.shape != ():
        raise CaptureError('DTYPE', f'{label}: FP64 storage is restricted to scalar coefficients')
    _reject_lazy_view(value, label)
    if contiguous and not value.is_contiguous():
        raise CaptureError('STATE_LAYOUT', f'{label}: committed state must be contiguous')


def _canonical_parameters(params):
    """Tied parameters are one binding; distinct tensors may not share storage."""
    unique, aliases, ids, storages = {}, {}, {}, {}
    for name, value in params.items():
        _check_tensor(value, f'parameter {name}')
        canonical = ids.setdefault(id(value), name)
        aliases[name] = canonical
        unique.setdefault(canonical, value)
        if value.numel():
            other = storages.setdefault(_storage_key(value), canonical)
            if other != canonical:
                raise CaptureError('PARAMETER_STORAGE', f'{name} and {other} are distinct tensors sharing storage; '
                                   'use a single Tensor object for tied parameters')
    return unique, aliases


def _checked_state(state):
    state = dict(state or {})
    for name, value in state.items():
        _check_tensor(value, f'state {name}', contiguous=True)
    return state


def _validate_new_state(before, after):
    if not isinstance(after, dict) or set(after) != set(before):
        raise CaptureError('STATE_RETURN', 'returned state keys must exactly match input state keys')
    for name, old in before.items():
        new = after[name]
        if (not isinstance(new, torch.Tensor) or new.shape != old.shape or new.dtype != old.dtype
                or new.device != old.device):
            raise CaptureError('STATE_RETURN', f'returned state {name} must preserve shape, dtype and device')


def _flatten_predictions(predictions):
    leaves, spec = pytree.tree_flatten(predictions)
    if not leaves or not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
        raise CaptureError('PREDICTIONS', 'predictions must be a nonempty Tensor or tuple/list/dict of Tensors')
    for index, leaf in enumerate(leaves):
        if leaf.dtype not in STORAGE_DTYPES:
            raise CaptureError('PREDICTIONS', f'prediction {index}: dtype {leaf.dtype} unsupported')
        if leaf.dtype == torch.float64 and leaf.shape != ():
            raise CaptureError('PREDICTIONS', 'FP64 storage is restricted to scalar coefficients')
    return leaves, spec


def _trace(function, arguments, label):
    try:
        return make_fx(function, decomposition_table=capture_decompositions(), tracing_mode='fake',
                       _allow_non_fake_inputs=True, _error_on_data_dependent_ops=True)(*arguments)
    except CaptureError:
        raise
    except Exception as exc:
        if any(marker in str(exc).lower() for marker in ('data-dependent', 'data dependent', '_local_scalar_dense')):
            raise CaptureError('RUNTIME_CONTROL', f'{label}: tensor-dependent Python control/scalar extraction '
                               f'requires an explicit region capture: {type(exc).__name__}: {exc}') from exc
        raise CaptureError('TRACE', f'{label} capture failed: {type(exc).__name__}: {exc}') from exc


def _materialize_input_outputs(gm):
    """Outputs that are an input/constant (or a view of one) get a fresh copy.

    Bound inputs, parameters and state keep caller-owned storage. Returning one
    directly would expose an alias of a binding the next call overwrites (or a
    state version committed after reads), so only these outputs are copied.
    """
    output = next(node for node in gm.graph.nodes if node.op == 'output')
    results = list(output.args[0])
    changed = False
    for index, node in enumerate(results):
        root = node
        while (root.op == 'call_function' and str(root.target) in _ALIASES and root.args
               and isinstance(root.args[0], torch.fx.Node)):
            root = root.args[0]
        if root.op not in ('placeholder', 'get_attr'):
            continue
        with gm.graph.inserting_before(output):
            copy = gm.graph.call_function(torch.ops.aten.clone.default, (node,))
        value = node.meta['val']
        copy.meta['val'] = torch.empty(tuple(value.shape), dtype=value.dtype, device='meta')
        results[index] = copy
        changed = True
    if changed:
        output.args = (tuple(results),)
        gm.graph.lint()
        gm.recompile()
    return gm


def _refuse_derivatives(operations, label):
    for operation in operations:
        target = operation.target
        if '_backward' in target or target.endswith('_backward.default'):
            raise CaptureError('INFERENCE_DERIVATIVE', f'{label}: derivative operation {target} in forward-only capture')


@dataclass
class InferenceIR:
    """Flat forward program: prediction leaves then new explicit state."""
    values: dict[str, Value]
    operations: list[Operation]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    state_writes: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def output_names(self):
        return self.outputs

    def to_dict(self):
        result = TrainingIR(self.values, self.operations, self.inputs, {}, self.metadata).to_dict()
        result.update(format_version=6, kind='inference', outputs=list(self.outputs),
                      state_writes=dict(self.state_writes))
        return result

    @classmethod
    def from_dict(cls, data):
        if data.get('format_version') != 6 or data.get('kind') != 'inference':
            raise CaptureError('IR_VERSION', 'unknown inference graph format')
        graph = TrainingIR.from_dict({**data, 'format_version': 1})
        return cls(graph.values, graph.operations, graph.inputs, tuple(graph.outputs),
                   dict(data['state_writes']), graph.metadata)


@dataclass
class LoopInferenceIR:
    """init -> body repeated by a device count -> output; regions are ordinary graphs."""
    init: OperationIR
    body: OperationIR
    output: OperationIR
    external_values: dict[str, Value]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    count: Value
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def values(self):
        return self.external_values

    def output_names(self):
        return self.outputs

    def to_dict(self):
        return {'format_version': 7, 'kind': 'inference_loop',
                'regions': {name: getattr(self, name).to_dict() for name in ('init', 'body', 'output')},
                'external_values': {key: asdict(value) for key, value in self.external_values.items()},
                'inputs': list(self.inputs), 'outputs': list(self.outputs), 'count': asdict(self.count),
                'max_steps': self.max_steps, 'metadata': self.metadata}

    @classmethod
    def from_dict(cls, data):
        if data.get('format_version') != 7 or data.get('kind') != 'inference_loop':
            raise CaptureError('IR_VERSION', 'unknown inference loop format')

        def value(entry):
            entry = dict(entry)
            for key in ('shape', 'strides', 'output_roles'):
                entry[key] = tuple(entry[key])
            return Value(**entry)
        return cls(**{name: OperationIR.from_dict(data['regions'][name]) for name in ('init', 'body', 'output')},
                   external_values={key: value(raw) for key, raw in data['external_values'].items()},
                   inputs=tuple(data['inputs']), outputs=tuple(data['outputs']), count=value(data['count']),
                   max_steps=data['max_steps'], metadata=data['metadata'])


def load_ir(data):
    return (LoopInferenceIR if data.get('kind') == 'inference_loop' else InferenceIR).from_dict(data)


@dataclass
class CapturedInference:
    ir: InferenceIR | LoopInferenceIR
    graph_modules: dict[str, torch.fx.GraphModule]
    example_args: tuple[torch.Tensor, ...]
    parameter_names: tuple[str, ...]
    parameter_aliases: dict[str, str]
    state_names: tuple[str, ...]
    prediction_spec: Any
    api: dict[str, Any]
    constants: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def is_loop(self):
        return isinstance(self.ir, LoopInferenceIR)

    def unpack(self, outputs):
        """Map a flat output tuple to predictions, state and loop count/status."""
        n, ns = self.api['prediction_count'], len(self.state_names)
        result = {'predictions': pytree.tree_unflatten(list(outputs[:n]), self.prediction_spec),
                  'state': dict(zip(self.state_names, outputs[n:n + ns]))}
        if self.is_loop:
            result['executed_count'], result['status'] = outputs[n + ns], outputs[n + ns + 1]
        return result

    def graph_step(self, *flat):
        """CPU interpretation of the captured graphs; the native program never calls this."""
        modules = self.graph_modules
        if not self.is_loop:
            return tuple(modules['forward'](*flat))
        np_, ns, ni = len(self.parameter_names), len(self.state_names), self.api['input_count']
        params, state = tuple(flat[:np_]), tuple(flat[np_:np_ + ns])
        data, count = tuple(flat[np_ + ns:np_ + ns + ni]), flat[-1]
        requested = int(count.item())
        steps = min(max(requested, 0), self.ir.max_steps)
        carry = tuple(modules['init'](*params, *data))
        nc = len(carry)
        for _ in range(steps):
            result = modules['body'](*params, *carry, *state, *data)
            carry, state = tuple(result[:nc]), tuple(result[nc:])
        predictions = tuple(modules['output'](*params, *carry, *data))
        return (*predictions, *(value.clone() for value in state),
                count.new_tensor(steps), count.new_tensor(int(requested != steps)))

    def build(self, directory, *, cache_dir, arch='sm_90', lower_options=None):
        """Write a bound native artifact; calls nvcc only on the build worker."""
        return prepare_inference(self, directory, cache_dir=cache_dir, arch=arch, lower_options=lower_options)


def capture_inference(fn: Callable, params: dict[str, torch.Tensor], inputs, *, state=None) -> CapturedInference:
    """Capture ``fn(params, *inputs) -> predictions`` with no differentiation.

    With ``state`` supplied the signature is ``fn(params, state, *inputs) ->
    (predictions, new_state)``. Predictions are a Tensor or a tuple/list/dict of
    Tensors. Parameters are read-only bindings (tied entries canonicalized by
    identity); explicit state is the only committed effect. Python control
    specializes at capture; implicit randomness and caller-visible mutation are
    refused exactly as in training capture. Flat ABI: parameters, state, inputs,
    hoisted immutable constants. Outputs: prediction leaves, new state.
    """
    _check_version()
    unique, aliases = _canonical_parameters(params)
    names = tuple(unique)
    index = {name: i for i, name in enumerate(names)}
    state = _checked_state(state) if state is not None else None
    state_names = tuple(state or ())
    inputs = tuple(inputs)
    for position, value in enumerate(inputs):
        _check_tensor(value, f'input {position}')
    n, ns = len(names), len(state_names)
    spec_box = []

    def forward(*flat):
        mapping = {alias: flat[index[canonical]] for alias, canonical in aliases.items()}
        new_state = ()
        if state is not None:
            old = dict(zip(state_names, flat[n:n + ns]))
            result = fn(mapping, old, *flat[n + ns:])
            if not isinstance(result, tuple) or len(result) != 2:
                raise CaptureError('STATE_RETURN', 'stateful forward must return (predictions, new_state)')
            predictions, returned = result
            _validate_new_state(old, returned)
            # Materialize each new version before the final commit, as training does.
            new_state = tuple(returned[name].clone() for name in state_names)
        else:
            predictions = fn(mapping, *flat[n:])
        leaves, spec = _flatten_predictions(predictions)
        spec_box.append(spec)
        return (*leaves, *new_state)

    example_args = (*(value.detach() for value in unique.values()),
                    *(value.detach() for value in (state or {}).values()), *inputs)
    labels = ([f'parameter {name}' for name in names] + [f'state {name}' for name in state_names]
              + [f'input {i}' for i in range(len(inputs))])
    _reject_placeholder_storage_alias(example_args, labels)
    started = time.perf_counter()
    gm = _trace(forward, example_args, 'forward')
    gm = _functionalize_internal_mutations(gm, example_args)
    gm = _materialize_input_outputs(gm)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    constant_attributes, constant_values, constants = hoist_tensor_constants(gm)
    example_args = (*example_args, *constant_values)
    labels += [f'constant:{name}' for name in constant_attributes]
    region = _import_region(gm, 'forward', labels)
    _refuse_derivatives(region.operations, 'forward')
    spec = spec_box[-1]
    npred = len(region.outputs) - ns
    roles = (['parameter'] * n + ['state'] * ns + ['input'] * len(inputs)
             + ['constant_input'] * len(constant_attributes))
    keys = [*names, *state_names, *([None] * (len(inputs) + len(constant_attributes)))]
    for name, role, key in zip(region.inputs, roles, keys):
        region.values[name].role, region.values[name].parameter = role, key
    outputs = tuple(region.outputs)
    for position, name in enumerate(outputs):
        role = f'prediction:{position}' if position < npred else f'state:{state_names[position - npred]}'
        region.values[name].output_roles += (role,)
    state_inputs = region.inputs[n:n + ns]
    writes = dict(zip(state_inputs, outputs[npred:]))
    metadata = {
        'torch_version': torch.__version__, 'parameter_names': list(names), 'parameter_aliases': aliases,
        'state_names': list(state_names), 'prediction_count': npred,
        'prediction_spec': pytree.treespec_dumps(spec),
        'explicit_input_indices': list(range(n + ns, n + ns + len(inputs))),
        'constant_input_names': list(region.inputs[n + ns + len(inputs):]),
        'constant_input_attributes': list(constant_attributes),
        'capture': 'make_fx(fake) of forward only; no autograd transform, loss or optimizer',
        'state_commit': 'explicit new state committed once after all graph reads; parameters never written',
        'control_flow': 'Python static loops/branches specialize; runtime control requires explicit regions',
        'randomness': 'implicit generators refused; explicit Philox state only'}
    ir = InferenceIR(region.values, region.operations, region.inputs, outputs, writes, metadata)
    api = {'kind': 'flat', 'prediction_count': npred, 'input_count': len(inputs),
           'input_indices': {'inputs': list(range(n + ns, n + ns + len(inputs))), 'buffers': {}},
           'capture_seconds': time.perf_counter() - started}
    return CapturedInference(ir, {'forward': gm}, tuple(example_args), names, aliases, state_names, spec,
                             api, constants)


def _module_buffers(model, params, state):
    mutable = {*(_storage_key(value) for value in params.values()),
               *(_storage_key(value) for value in (state or {}).values())} - {None}
    buffers, buffer_aliases, identities = {}, {}, {}
    for name, value in model.named_buffers(remove_duplicate=False):
        if _storage_key(value) in mutable:
            raise CaptureError('MODULE_BUFFER_ALIAS', f'buffer {name} shares parameter/state storage')
        canonical = identities.setdefault(id(value), name)
        buffer_aliases[name] = canonical
        if canonical not in buffers:
            buffers[canonical] = value.detach().clone()  # frozen private snapshot
    return buffers, buffer_aliases


def capture_module_inference(model: torch.nn.Module, inputs, *, state=None) -> CapturedInference:
    """Capture ``model(*inputs)`` as a forward-only native program.

    All registered parameters (trainable or frozen) become read-only weight
    bindings, tied entries kept; registered buffers are frozen read-only inputs.
    With ``state``, forward receives ``state=state`` and returns
    ``(predictions, new_state)``; the new state is committed after the call.
    The module's current train/eval mode is traced as is: put it in eval mode,
    since implicit dropout randomness is refused.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError('capture_module_inference expects an nn.Module')
    if not isinstance(inputs, (tuple, list)):
        raise TypeError('inputs must be a tuple/list of tensors')
    inputs = tuple(inputs)
    params = dict(model.named_parameters(remove_duplicate=False))
    state = _checked_state(state) if state is not None else None
    buffers, buffer_aliases = _module_buffers(model, params, state)
    buffer_names = tuple(buffers)
    ni = len(inputs)

    def call(p, explicit_state, arguments):
        bound = dict(zip(buffer_names, arguments[ni:]))
        bound_buffers = {alias: bound[canonical] for alias, canonical in buffer_aliases.items()}
        kwargs = {'state': explicit_state} if state is not None else None
        result = functional_call(model, (p, bound_buffers), tuple(arguments[:ni]), kwargs=kwargs,
                                 tie_weights=True, strict=True)
        if state is not None and (not isinstance(result, tuple) or len(result) != 2):
            raise CaptureError('MODULE_STATE', 'stateful module must return (predictions, new_state)')
        return result

    if state is None:
        def fn(p, *arguments):
            return call(p, None, arguments)
    else:
        def fn(p, explicit_state, *arguments):
            return call(p, explicit_state, arguments)
    captured = capture_inference(fn, params, (*inputs, *buffers.values()), state=state)
    start = len(captured.parameter_names) + len(captured.state_names)
    captured.api.update(input_count=ni, input_indices={
        'inputs': list(range(start, start + ni)),
        'buffers': dict(zip(buffer_names, range(start + ni, start + ni + len(buffer_names))))},
        buffer_aliases=buffer_aliases, module=type(model).__name__,
        module_training_mode=bool(model.training))
    captured.ir.metadata['module_api'] = {k: captured.api[k] for k in ('input_indices', 'buffer_aliases')}
    return captured


def capture_loop_inference(init_fn, body_fn, output_fn, params, inputs, count, *, max_steps, state=None):
    """Forward-only runtime-count loop from the explicit region contract.

    init_fn(params, *inputs) -> carry dict
    body_fn(params, carry, state, *inputs) -> (new_carry, new_state)
    output_fn(params, carry, *inputs) -> predictions (Tensor/tuple/dict)

    ``count`` is a scalar int64 device input clamped to ``[0, max_steps]`` at
    runtime; executed count and status (1 iff clamped) are returned. Changing
    it changes no generated code. Carry restarts from ``init_fn`` each call;
    explicit state persists across calls through the final commit.
    """
    _check_version()
    if type(max_steps) is not int or not 0 <= max_steps <= 1024:
        raise CaptureError('LOOP_BOUND', 'require static max_steps in [0,1024]')
    if not isinstance(count, torch.Tensor) or count.dtype != torch.int64 or count.ndim != 0:
        raise CaptureError('LOOP_COUNT', 'runtime count must be a scalar int64 Tensor')
    _reject_lazy_view(count, 'loop count')
    unique, aliases = _canonical_parameters(params)
    names = tuple(unique)
    index = {name: i for i, name in enumerate(names)}
    state = _checked_state(state)
    state_names = tuple(state)
    inputs = tuple(inputs)
    for position, value in enumerate(inputs):
        _check_tensor(value, f'loop input {position}')
    pvalues = tuple(value.detach() for value in unique.values())
    svalues = tuple(value.detach() for value in state.values())
    external = (*pvalues, *svalues, *inputs, count.detach())
    external_labels = ([f'parameter:{name}' for name in names] + [f'state:{name}' for name in state_names]
                       + [f'input:{i}' for i in range(len(inputs))] + ['count'])
    _reject_placeholder_storage_alias(external, external_labels)
    if any(value.device != count.device for value in external):
        raise CaptureError('LOOP_DEVICE', 'every loop input and count must be on the same device')
    n, ns, ni = len(names), len(state_names), len(inputs)
    carry_names, carry_specs, spec_box = [], [], []

    def mapping(flat):
        return {alias: flat[index[canonical]] for alias, canonical in aliases.items()}

    def checked_carry(result, *, initial=False):
        if not isinstance(result, dict) or not result:
            raise CaptureError('LOOP_CARRY', 'loop carry must be a nonempty dictionary')
        if initial and not carry_names:
            carry_names.extend(result)
            for key in carry_names:
                _check_tensor(result[key], f'carry {key}')
                carry_specs.append((tuple(result[key].shape), result[key].dtype, result[key].device))
        if set(result) != set(carry_names):
            raise CaptureError('LOOP_CARRY', 'every iteration must preserve carry keys')
        for key, spec in zip(carry_names, carry_specs):
            value = result[key]
            if not isinstance(value, torch.Tensor) or (tuple(value.shape), value.dtype, value.device) != spec:
                raise CaptureError('LOOP_CARRY', f'iteration carry {key} changed shape, dtype or device')
        return tuple(result[key] for key in carry_names)

    def init_region(*flat):
        carry = checked_carry(init_fn(mapping(flat[:n]), *flat[n:]), initial=True)
        return tuple(value.clone() for value in carry)

    def body_region(*flat):
        nc = len(carry_names)
        carry = dict(zip(carry_names, flat[n:n + nc]))
        old = dict(zip(state_names, flat[n + nc:n + nc + ns]))
        result = body_fn(mapping(flat[:n]), carry, old, *flat[n + nc + ns:])
        if not isinstance(result, tuple) or len(result) != 2:
            raise CaptureError('LOOP_BODY', 'body must return (new_carry, new_state)')
        _validate_new_state(old, result[1])
        new_carry = checked_carry(result[0])
        return tuple(value.clone() for value in (*new_carry, *(result[1][key] for key in state_names)))

    def output_region(*flat):
        nc = len(carry_names)
        predictions = output_fn(mapping(flat[:n]), dict(zip(carry_names, flat[n:n + nc])), *flat[n + nc:])
        leaves, spec = _flatten_predictions(predictions)
        spec_box.append(spec)
        return tuple(value.clone() for value in leaves)

    started = time.perf_counter()
    p_labels = [f'parameter:{name}' for name in names]
    s_labels = [f'state:{name}' for name in state_names]
    i_labels = [f'input:{i}' for i in range(ni)]
    graphs, modules = {}, {}
    graphs['init'], modules['init'] = _capture_region(init_region, (*pvalues, *inputs), 'init', p_labels + i_labels)
    cvalues = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype, device in carry_specs)
    c_labels = [f'carry:{key}' for key in carry_names]
    graphs['body'], modules['body'] = _capture_region(
        body_region, (*pvalues, *cvalues, *svalues, *inputs), 'body', p_labels + c_labels + s_labels + i_labels)
    graphs['output'], modules['output'] = _capture_region(
        output_region, (*pvalues, *cvalues, *inputs), 'output', p_labels + c_labels + i_labels)
    for label, graph in graphs.items():
        _refuse_derivatives(graph.operations, label)
    spec = spec_box[-1]
    npred = len(graphs['output'].outputs)
    external_values, input_names = {}, []
    for position, (tensor, label) in enumerate(zip(external, external_labels)):
        name = f'inference_arg_{position}'
        role, _, key = label.partition(':')
        external_values[name] = Value(name, tuple(tensor.shape), str(tensor.dtype).removeprefix('torch.'),
                                      tuple(tensor.stride()), role=role, parameter=key or None)
        input_names.append(name)
    outputs = []

    def output(shape, dtype, role):
        name = f'inference_result_{len(outputs)}'
        external_values[name] = Value(name, tuple(shape), dtype, _contiguous(shape), role='loop_output',
                                      output_roles=(role,))
        outputs.append(name)

    for position, local in enumerate(graphs['output'].outputs):
        value = graphs['output'].values[local]
        output(value.shape, value.dtype, f'prediction:{position}')
    for key, tensor in zip(state_names, svalues):
        output(tensor.shape, str(tensor.dtype).removeprefix('torch.'), f'state:{key}')
    output((), 'int64', 'executed_count')
    output((), 'int64', 'status')
    metadata = {'torch_version': torch.__version__, 'parameter_names': list(names), 'parameter_aliases': aliases,
                'state_names': list(state_names), 'carry_names': list(carry_names), 'n_inputs': ni,
                'prediction_count': npred, 'prediction_spec': pytree.treespec_dumps(spec),
                'capture': 'explicit init/body/output regions, make_fx(fake) forward only; no AD/loss/optimizer',
                'count_policy': 'clamp to [0,max_steps]; status=1 iff requested count was clamped',
                'state_commit': 'final explicit state committed once after the output region; parameters never written'}
    ir = LoopInferenceIR(graphs['init'], graphs['body'], graphs['output'], external_values, tuple(input_names),
                         tuple(outputs), external_values[input_names[-1]], max_steps, metadata)
    api = {'kind': 'loop', 'prediction_count': npred, 'input_count': ni, 'max_steps': max_steps,
           'count_index': len(external) - 1,
           'input_indices': {'inputs': list(range(n + ns, n + ns + ni)), 'buffers': {}},
           'capture_seconds': time.perf_counter() - started}
    return CapturedInference(ir, modules, tuple(external), names, aliases, state_names, spec, api)


def _contiguous(shape):
    strides, stride = [], 1
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def lower_inference(ir, **options):
    from .lower import lower
    return lower(ir, **options)


def prepare_inference(captured, directory, *, cache_dir, arch='sm_90', lower_options=None):
    """Persist source, metadata and initial bindings with exact library identity.

    ``lower_options`` pass through to ``lower`` unchanged (for each loop region),
    so schedule options added to the shared lowerer apply to inference too.
    """
    from .runtime import build
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    lower_options = dict(lower_options or {})
    lower_options.setdefault('target_arch', arch)
    started = time.perf_counter()
    program = lower_inference(captured.ir, **lower_options)
    lower_seconds = time.perf_counter() - started
    program.write(directory)
    (directory / 'graph.json').write_text(json.dumps(captured.ir.to_dict(), sort_keys=True) + '\n')
    torch.save({'args': captured.example_args}, directory / 'bindings.pt')
    started_build = time.perf_counter()
    library = build(program, cache_dir=cache_dir, arch=arch)
    build_seconds = time.perf_counter() - started_build
    files = {path.name: _hash(path) for path in sorted(directory.iterdir()) if path.is_file()}
    api = {**captured.api, 'parameter_names': list(captured.parameter_names),
           'parameter_aliases': captured.parameter_aliases, 'state_names': list(captured.state_names),
           'prediction_spec': pytree.treespec_dumps(captured.prediction_spec)}
    receipt = {'schema': ARTIFACT_SCHEMA, 'source_hash': program.cache_key, 'files': files,
               'library': str(library), 'library_sha256': _hash(library), 'arch': arch,
               'torch': torch.__version__, 'lower_options': lower_options, 'lower_seconds': lower_seconds,
               'build_seconds': build_seconds, 'prepare_seconds': time.perf_counter() - started,
               'capture_seconds': captured.api.get('capture_seconds'), 'api': api,
               'arena_bytes': program.manifest['arena_bytes'], 'shared_bytes': program.manifest['shared_bytes']}
    receipt['immutable_bindings_sha256'] = _immutable_identity(captured.example_args, program.manifest, api)
    (directory / 'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


class Inference:
    """Bound forward program: stage explicit inputs, launch one kernel, read predictions.

    Prediction tensors are views of the native arena and are overwritten by the
    next call; clone them to keep them. Calls are asynchronous on PyTorch's
    current CUDA stream. Weights and buffers are never written by the program.
    """

    def __init__(self, runtime, receipt):
        self.runtime, self.receipt = runtime, receipt
        api = receipt['api']
        self.api = api
        self._spec = pytree.treespec_loads(api['prediction_spec'])
        n, ns = api['prediction_count'], len(api['state_names'])
        # Arena views are stable for the bound program, so the structured
        # predictions object is built once rather than on every launch.
        self._predictions = pytree.tree_unflatten(list(runtime.outputs[:n]), self._spec)
        self._input_views = tuple(runtime.args[index] for index in api['input_indices']['inputs'])
        self.count_binding = runtime.args[api['count_index']] if api['kind'] == 'loop' else None
        self.executed_count = runtime.outputs[n + ns] if api['kind'] == 'loop' else None
        self.loop_status = runtime.outputs[n + ns + 1] if api['kind'] == 'loop' else None

    @classmethod
    def load(cls, directory, *, device='cuda', blocks=None, launch_cache=True):
        """Verify and bind a CPU-prepared artifact; native compilation is disabled."""
        from .lower import GeneratedProgram
        from .runtime import Runtime
        directory = Path(directory)
        receipt = json.loads((directory / 'artifact.json').read_text())
        if receipt.get('schema') != ARTIFACT_SCHEMA:
            raise ValueError('unsupported inference artifact schema')
        for name, expected in receipt['files'].items():
            if Path(name).name != name or _hash(directory / name) != expected:
                raise ValueError(f'artifact identity mismatch: {name}')
        if _hash(receipt['library']) != receipt['library_sha256']:
            raise ValueError('native library identity mismatch')
        program = GeneratedProgram.load(directory)
        if program.cache_key != receipt['source_hash']:
            raise ValueError('native source identity mismatch')
        saved = torch.load(directory / 'bindings.pt', weights_only=True, map_location='cpu')['args']
        if _immutable_identity(saved, program.manifest, receipt['api']) != receipt['immutable_bindings_sha256']:
            raise ValueError('immutable buffer/constant identity mismatch')
        args = []
        for value in saved:
            bound = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=device)
            bound.copy_(value)
            args.append(bound)
        return cls(Runtime(program, tuple(args), library=receipt['library'], blocks=blocks,
                           launch_cache=launch_cache), receipt)

    @property
    def program(self):
        return self.runtime.program

    @property
    def launch_policy(self):
        return self.runtime.launch_policy

    @property
    def launch_cache_policy(self):
        return self.runtime.launch_cache_policy

    @property
    def parameters(self):
        return self.runtime.parameters

    @property
    def state(self):
        return self.runtime.state

    @property
    def inputs(self):
        return self._input_views

    @property
    def predictions(self):
        return self._predictions

    def stage(self, *inputs, count=None):
        """Copy explicit inputs (and a loop count) into stable device bindings."""
        if len(inputs) != len(self._input_views):
            raise ValueError(f'expected {len(self._input_views)} inputs, got {len(inputs)}')
        copies = []
        for position, (target, source) in enumerate(zip(self._input_views, inputs)):
            if not isinstance(source, torch.Tensor) or source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f'input {position} shape/dtype differs from capture')
            copies.append((target, source))
        if count is not None:
            if self.count_binding is None:
                raise ValueError('this program has no runtime iteration count')
            copies.append((self.count_binding, torch.as_tensor(count, dtype=torch.int64)))
        _copy_bindings(copies)

    def set_count(self, count):
        if self.count_binding is None:
            raise ValueError('this program has no runtime iteration count')
        self.count_binding.copy_(torch.as_tensor(count, dtype=torch.int64))

    def run(self):
        """One native launch on the current stream; returns arena prediction views."""
        self.runtime.step()
        return self.predictions

    def __call__(self, *inputs, count=None):
        self.stage(*inputs, count=count)
        return self.run()

    def state_dict(self):
        """Synchronized CPU snapshot of explicit state (weights are not state)."""
        return {'source_hash': self.runtime.program.cache_key,
                'immutable_bindings_sha256': self.receipt['immutable_bindings_sha256'],
                'state': {name: value.detach().cpu().clone() for name, value in self.state.items()}}

    def load_state_dict(self, snapshot):
        if snapshot.get('source_hash') != self.runtime.program.cache_key:
            raise ValueError('snapshot belongs to a different generated program')
        if snapshot.get('immutable_bindings_sha256') != self.receipt['immutable_bindings_sha256']:
            raise ValueError('snapshot belongs to different frozen buffers/constants')
        self._copy_named(self.state, snapshot.get('state', {}), 'state')

    def load_parameters(self, parameters):
        """Replace weight values in place (same names/shapes/dtypes); the program is unchanged."""
        canonical = {}
        for name, value in parameters.items():
            key = self.api['parameter_aliases'].get(name, name)
            if key in canonical and value is not canonical[key]:
                raise ValueError(f'ambiguous tied parameter values for {key}; supply one canonical value')
            canonical[key] = value
        self._copy_named(self.parameters, canonical, 'parameters')

    @staticmethod
    def _copy_named(current, supplied, label):
        if set(current) != set(supplied):
            raise ValueError(f'{label} keys differ from the bound program')
        copies = []
        for name, target in current.items():
            source = supplied[name]
            if not isinstance(source, torch.Tensor) or source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f'{label}:{name} shape/dtype differs')
            copies.append((target, source))
        _copy_bindings(copies)
