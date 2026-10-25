"""Bounded leading-batch capture with one capacity allocation and ABI.

Every admitted integer extent is fake-traced during CPU preparation. The traces
must have the same operation/dependency topology; only one capacity-sized IR is
emitted. This finite-domain audit catches shape-dependent Python branches even
when they occur at an interior batch size. Runtime never selects a traced graph.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct

import torch

from .ir import CaptureError, Ref, Value


@dataclass(frozen=True)
class DynamicBatch:
    """Admit integer batches ``1 <= B <= maximum`` on data/target axis zero.

    Supply capacity-shaped examples to capture_module. Preparation audits every
    integer B on its CPU worker; native execution has one scalar B input, stable
    capacity storage, and one generated training program.
    """
    maximum: int = 32
    axis: int = 0

    def __post_init__(self):
        if type(self.maximum) is not int or not 2 <= self.maximum <= 32:
            raise ValueError('DynamicBatch.maximum must be an integer in [2, 32]')
        if type(self.axis) is not int or self.axis != 0:
            raise ValueError('DynamicBatch currently supports only leading axis=0')


_ALIASES = {'detach', 'alias', 't', 'transpose', 'permute', 'view', '_unsafe_view',
            'reshape', 'expand', 'squeeze', 'unsqueeze'}
_POINTWISE = {'add', 'sub', 'mul', 'div', 'pow', 'neg', 'tanh', 'exp', 'log', 'sqrt',
              'rsqrt', 'abs', 'sigmoid', 'silu', 'silu_backward', 'tanh_backward',
              'sigmoid_backward', 'relu', 'threshold_backward', 'clone', 'contiguous',
              '_to_copy', 'fill', 'ones_like', 'zeros_like', 'full_like'}
_SHAPE_ARGS = {'view': 1, '_unsafe_view': 1, 'reshape': 1, 'expand': 1,
               'full': 0, 'ones': 0, 'zeros': 0, 'empty': 0, 'new_full': 1,
               'philox_mask': 1}
_ADMITTED = _ALIASES | _POINTWISE | set(_SHAPE_ARGS) | {'mm', 'sum', 'mean', 'philox_advance',
                                                    'empty_like'}


def _base(op):
    return op.target.split('.')[1]


def _fail(message):
    raise CaptureError('DYNAMIC_BATCH', message)


def _same(a, b):
    return type(a) is type(b) and a == b


def _number_equal(a, b):
    # Integer argument rules must be exact. Floating ATen scalar arguments
    # enter FP32 tensor arithmetic after Python's double calculation; match
    # that final conversion bit-for-bit, rather than fitting a tolerance.
    if type(a) is int:
        return a == b
    try:
        return struct.pack('!f', a) == struct.pack('!f', b)
    except (OverflowError, struct.error):
        return False


def _scalar_marker(values, sizes, *, rng_width=None):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        _fail('batch-dependent literals must have a finite numeric scalar rule')
    if rng_width is not None:
        if all(type(v) is int and v == (size * rng_width + 3) // 4
               for v, size in zip(values, sizes)):
            return {'dynamic_batch_scalar': {'coefficient': rng_width, 'power': 1, 'ceil_divisor': 4}}
        _fail('Philox counter advance does not match the active mask element count')
    for power in (1, -1):
        coefficient = values[0] / sizes[0] if power == 1 else values[0] * sizes[0]
        if all(_number_equal(v, coefficient * size if power == 1 else coefficient / size)
               for v, size in zip(values, sizes)):
            if float(coefficient).is_integer():
                coefficient = int(coefficient)
            return {'dynamic_batch_scalar': {'coefficient': coefficient, 'power': power}}
    _fail('batch-dependent scalar is not C*B or C/B over the entire admitted domain')


def _rewrite_arguments(op, probes, sizes, axes, ir):
    """Audit literal changes by operation role, never by equality to capacity."""
    base = _base(op)
    shape_position = _SHAPE_ARGS.get(base)
    rng_width = None
    if base == 'philox_advance':
        state = op.args[0]
        masks = [candidate for candidate in ir.operations
                 if candidate.target == 'latch_ops.philox_mask.default'
                 and candidate.args[0] == state and candidate.outputs[0] in axes]
        if len(masks) == 1:
            tensor = ir.values[masks[0].outputs[0]]
            rng_width = tensor.numel // sizes[0]

    def visit(values, path):
        first = values[0]
        if all(_same(first, value) for value in values[1:]):
            return first
        if isinstance(first, (tuple, list)):
            if not all(type(value) is type(first) and len(value) == len(first) for value in values):
                _fail(f'{op.name}: argument structure changes with B')
            return type(first)(visit([value[i] for value in values], (*path, i)) for i in range(len(first)))
        if isinstance(first, dict):
            if not all(type(value) is dict and value.keys() == first.keys() for value in values):
                _fail(f'{op.name}: keyword structure changes with B')
            return {key: visit([value[key] for value in values], (*path, key)) for key in first}
        if path[:1] == ('args',) and len(path) == 3 and path[1] == shape_position:
            # Storage/layout arguments stay at capacity. Lowering obtains the
            # execution extent from explicit value-axis provenance instead.
            axis = axes.get(op.outputs[0])
            if path[2] != axis or any(type(value) is not int or value != size
                                      for value, size in zip(values, sizes)):
                _fail(f'{op.name}: unsupported batch-dependent shape argument {path}')
            return first
        if path == ('args', 1) and base == 'philox_advance' and rng_width is not None:
            return _scalar_marker(values, sizes, rng_width=rng_width)
        if (path == ('args', 1) and base in {'div', 'mul'}
                and ir.values[op.outputs[0]].dtype in {'float32', 'bfloat16'}):
            # This includes AD's mean divisor. A user-written scalar factor
            # obeying the same formula is also exact for every admitted B,
            # including scalar sum/B losses whose output shape is static.
            return _scalar_marker(values, sizes)
        _fail(f'{op.name}: unsupported batch-dependent literal at {path}')

    op.args = visit([op.args, *(probe.args for probe in probes)], ('args',))
    op.kwargs = visit([op.kwargs, *(probe.kwargs for probe in probes)], ('kwargs',))


def _audit_axis_propagation(op, axes, ir):
    """Require operation semantics to agree with the observed changing axis.

    Shape equality alone is insufficient at capacity two: a B=1 input can
    broadcast into a fixed two-row tensor while B=2 does not broadcast. Its
    output shape then appears static, but capacity indexing would read an
    inactive input row. Reject such changes of the logical index mapping.
    """
    base = _base(op)
    output = ir.values[op.outputs[0]]
    observed = axes.get(output.name)

    def require(expected):
        if observed != expected:
            _fail(f'{op.name}: operation batch-axis propagation requires {expected}, '
                  f'but shape audit observed {observed}; static-capacity broadcasting is unsupported')

    if base in _POINTWISE | {'empty_like'}:
        for name in op.inputs:
            if name in axes:
                source = ir.values[name]
                expected = axes[name] + len(output.shape) - len(source.shape)
                if expected < 0:
                    _fail(f'{op.name}: pointwise operation removes an active axis')
                require(expected)
        return
    if base in {'sum', 'mean'}:
        source = op.args[0].name
        axis = axes.get(source)
        if axis is None:
            require(None)
            return
        rank = len(ir.values[source].shape)
        dims = op.args[1] if len(op.args) > 1 else op.kwargs.get('dim')
        if dims is None or dims == [] or dims == ():
            dims = list(range(rank))
        else:
            dims = [dims] if isinstance(dims, int) else dims
        dims = {int(dim) % rank for dim in dims}
        keep = bool(op.args[2] if len(op.args) > 2 else op.kwargs.get('keepdim', False))
        require(None if axis in dims else axis if keep else axis - sum(dim < axis for dim in dims))
        return
    if base == 'mm':
        left, right = (axes.get(op.args[index].name) for index in (0, 1))
        if (left == 1) != (right == 0):
            _fail(f'{op.name}: both product operands must share active reduction extent')
        if left == 0 and right == 1:
            _fail(f'{op.name}: two active product output axes are unsupported')
        require(0 if left == 0 else 1 if right == 1 else None)
        return
    if base not in _ALIASES:
        # Explicit-shape factories (including new_full's metadata template) and
        # Philox mask creation have their shape arguments audited separately.
        return
    source = ir.values[op.args[0].name]
    axis = axes.get(source.name)
    if axis is None:
        if base != 'expand':
            require(None)
        return
    rank = len(source.shape)
    if base in {'detach', 'alias'}:
        expected = axis
    elif base == 't':
        expected = rank - 1 - axis
    elif base == 'transpose':
        a, b = int(op.args[1]) % rank, int(op.args[2]) % rank
        expected = b if axis == a else a if axis == b else axis
    elif base == 'permute':
        expected = [int(dim) % rank for dim in op.args[1]].index(axis)
    elif base == 'expand':
        expected = axis + len(output.shape) - rank
    elif base == 'unsqueeze':
        dim = int(op.args[1]) % (rank + 1)
        expected = axis + int(dim <= axis)
    elif base == 'squeeze':
        dims = (range(rank) if len(op.args) == 1 else
                op.args[1] if isinstance(op.args[1], (tuple, list)) else [op.args[1]])
        removed = {int(dim) % rank for dim in dims if source.shape[int(dim) % rank] == 1}
        if axis in removed:
            _fail(f'{op.name}: squeeze removes an active axis')
        expected = axis - sum(dim < axis for dim in removed)
    elif base in {'view', '_unsafe_view', 'reshape'}:
        if source.shape == output.shape:
            expected = axis
        elif (axis == 0 and observed == 0
              and math.prod(source.shape[1:]) == math.prod(output.shape[1:])):
            expected = 0
        else:
            _fail(f'{op.name}: reshape must preserve the active axis and per-row element count')
    else:
        _fail(f'{op.name}: alias has no active-axis propagation rule')
    require(expected)


def capture_dynamic_module(capture, inputs, targets, spec):
    """Audit ordinary captures and annotate a single capacity IR in place."""
    if not isinstance(spec, DynamicBatch):
        raise TypeError('dynamic_batch must be a DynamicBatch annotation')
    if not inputs:
        _fail('at least one explicit batched model input is required')
    for label, tensors in (('input', inputs), ('target', targets)):
        for index, value in enumerate(tensors):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[0] != spec.maximum or not value.is_contiguous()):
                _fail(f'{label} {index}: expected contiguous rank-two capacity shape '
                      f'({spec.maximum}, features)')
    full = capture(inputs, targets)
    from .optim import SGDConfig
    if not isinstance(full.config, SGDConfig):
        _fail('dynamic batch currently admits momentum SGD only')
    if full.constants:
        _fail('batch capture currently requires explicit buffers instead of captured tensor constants')
    sizes = [spec.maximum, *range(1, spec.maximum)]
    probes = []
    for size in sizes[1:]:
        probe = capture(tuple(value[:size].clone() for value in inputs),
                        tuple(value[:size].clone() for value in targets))
        if probe.constants:
            _fail('batch-dependent captured constants are unsupported')
        probes.append(probe.ir)
    ir = full.ir
    for probe in probes:
        if (tuple(ir.values) != tuple(probe.values) or ir.inputs != probe.inputs
                or ir.outputs != probe.outputs or len(ir.operations) != len(probe.operations)
                or ir.metadata.get('gradient_participation') != probe.metadata.get('gradient_participation')):
            _fail('operation/value/gradient topology changes within the admitted batch domain')
    axes = {}
    for name, value in ir.values.items():
        shapes = [value.shape, *(probe.values[name].shape for probe in probes)]
        if any(len(shape) != len(value.shape) for shape in shapes):
            _fail(f'{name}: rank changes with active batch')
        changed = [axis for axis in range(len(value.shape))
                   if any(shape[axis] != value.shape[axis] for shape in shapes[1:])]
        if changed:
            if len(changed) != 1 or any(shape[changed[0]] != size for shape, size in zip(shapes, sizes)):
                _fail(f'{name}: extent must be exactly B on one dimension')
            if value.role in {'parameter', 'momentum', 'optimizer_state', 'state', 'constant_input'}:
                _fail(f'{name}: persistent state cannot have a dynamic extent')
            if value.alias_of is None and changed[0] != 0:
                _fail(f'{name}: material tensors require leading dynamic axis zero')
            axes[name] = changed[0]
        for probe in probes:
            other = probe.values[name]
            if (value.dtype, value.role, value.parameter, value.alias_of, value.storage_offset,
                value.producer, value.output_roles) != (other.dtype, other.role, other.parameter,
                other.alias_of, other.storage_offset, other.producer, other.output_roles):
                _fail(f'{name}: dtype, alias, storage offset or value role changes with B')
            if value.alias_of is not None and value.strides != other.strides:
                _fail(f'{name}: batch-dependent alias strides are unsupported')
    positions = full.input_indices['inputs'] + full.input_indices['targets']
    if any(axes.get(ir.inputs[index]) != 0 for index in positions):
        _fail('declared data and target inputs did not preserve batch-axis provenance')
    for index, op in enumerate(ir.operations):
        base = _base(op)
        namespace_ok = (op.target.startswith('aten.') or
                        op.target in {'latch_ops.philox_mask.default', 'latch_ops.philox_advance.default'})
        if base not in _ADMITTED or not namespace_ok:
            _fail(f'{op.target}: no dynamic-batch semantic rule')
        others = [probe.operations[index] for probe in probes]
        for other in others:
            for field in ('name', 'target', 'inputs', 'outputs', 'dependencies', 'reads', 'writes',
                          'reduction_axes', 'synchronization', 'effect'):
                if getattr(op, field) != getattr(other, field):
                    _fail(f'{op.name}: operation topology changes with B')
        _audit_axis_propagation(op, axes, ir)
        _rewrite_arguments(op, others, sizes, axes, ir)
    scalar_name = 'dynamic_batch_size'
    if scalar_name in ir.values:
        _fail('reserved dynamic batch binding name already exists')
    scalar_index = len(ir.inputs)
    ir.values[scalar_name] = Value(scalar_name, (), 'int64', (), role='input')
    ir.inputs = (*ir.inputs, scalar_name)
    ir.metadata['dynamic_batch'] = {
        'version': 1, 'maximum': spec.maximum, 'minimum': 1, 'axis': spec.axis,
        'input': scalar_name, 'input_index': scalar_index, 'value_axes': axes,
        'data_input_indices': positions, 'audited_batch_sizes': sorted(sizes),
        'capture_policy': 'exhaustive finite-domain fake-AD topology and scalar audit',
        'storage_policy': 'capacity shapes and stable pointers; active-prefix computation',
        'runtime_policy': 'one device scalar staged before the same generated update or graph replay'}
    full.input_indices['active_batch'] = scalar_index
    full.ir.metadata['module_api']['dynamic_batch'] = dict(ir.metadata['dynamic_batch'])
    original_step = full.captured.step_fn
    original_arity = len(full.example_args)

    def reference_step(*arguments):
        if len(arguments) != original_arity + 1:
            raise ValueError('dynamic training reference argument count differs')
        scalar = arguments[-1]
        if not isinstance(scalar, torch.Tensor) or scalar.dtype != torch.int64 or scalar.shape != ():
            raise ValueError('active batch reference input must be a scalar int64 tensor')
        batch = int(scalar.item())
        if not 1 <= batch <= spec.maximum:
            raise ValueError('active batch is outside captured capacity')
        active = list(arguments[:-1])
        for index in positions:
            active[index] = active[index][:batch]
        return original_step(*active)

    full.captured.step_fn = reference_step
    full.captured.example_args = (*full.example_args, torch.tensor(spec.maximum, dtype=torch.int64,
                                                                 device=full.example_args[0].device))
    # This FX object remains the auditable capacity graph. Its additional unused
    # placeholder matches the ABI; step_fn above is the active-shape CPU oracle.
    graph = full.graph_module.graph
    anchor = next(node for node in graph.nodes if node.op != 'placeholder')
    with graph.inserting_before(anchor):
        placeholder = graph.placeholder(scalar_name)
        placeholder.meta['val'] = full.example_args[-1]
    graph.lint()
    full.graph_module.recompile()
    return full
