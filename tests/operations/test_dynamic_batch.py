"""CPU qualification of bounded batch capture and public staging contracts."""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import traceback
import struct

import torch

from megakernel_lab.operation_compiler.api import ModuleUpdate, capture_module
from megakernel_lab.operation_compiler.dynamic_batch import DynamicBatch, _number_equal, _scalar_marker
from megakernel_lab.operation_compiler.ir import CaptureError, TrainingIR
from megakernel_lab.operation_compiler.random import dropout, make_rng_state
from megakernel_lab.operation_compiler.runtime import Runtime


def run():
    torch.set_num_threads(1)
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    def objective(prediction, target):
        return (prediction - target).square().mean()

    def collision_and_reference():
        # A feature/parameter dimension equal to maximum must remain static.
        torch.manual_seed(1901)
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32))
        x, y = torch.randn(32, 32), torch.randn(32, 32)
        captured = capture_module(model, objective, (x,), (y,), dynamic_batch=DynamicBatch(32))
        dynamic = captured.ir.metadata['dynamic_batch']
        assert dynamic['audited_batch_sizes'] == list(range(1, 33))
        assert not set(captured.ir.outputs['grads'].values()) & set(dynamic['value_axes'])
        assert not set(captured.ir.outputs['params'].values()) & set(dynamic['value_axes'])
        for name in captured.ir.inputs:
            if captured.ir.values[name].role in {'parameter', 'momentum'}:
                assert name not in dynamic['value_axes']
        for batch in (1, 3, 7, 16, 31, 32):
            args = (*captured.example_args[:-1], torch.tensor(batch, dtype=torch.int64))
            result = captured.unpack(captured.step_fn(*args))
            loss = objective(model(x[:batch]), y[:batch])
            grads = torch.autograd.grad(loss, tuple(model.parameters()))
            torch.testing.assert_close(result['loss'], loss)
            for observed, expected in zip(result['grads'].values(), grads):
                torch.testing.assert_close(observed, expected)
        roundtrip = TrainingIR.from_dict(captured.ir.to_dict())
        assert roundtrip.to_dict() == captured.ir.to_dict()
        assert any(isinstance(op.args[1], dict) and 'dynamic_batch_scalar' in op.args[1]
                   for op in captured.ir.operations if len(op.args) > 1)

    check('axis_provenance_mean_divisor_and_full_domain_reference', collision_and_reference)

    def fractional_scalar_rules():
        sizes = [8, *range(1, 8)]
        for power in (1, -1):
            values = [.1 * batch if power == 1 else .1 / batch for batch in sizes]
            rule = _scalar_marker(values, sizes)['dynamic_batch_scalar']
            assert rule['power'] == power
            for value, batch in zip(values, sizes):
                predicted = (rule['coefficient'] * batch if rule['power'] == 1
                             else rule['coefficient'] / batch)
                assert struct.pack('!f', value) == struct.pack('!f', predicted)
        # Nearby Python doubles on opposite sides of an FP32 rounding midpoint
        # are not interchangeable even when isclose would accept them.
        midpoint = 1.0 + 2.0 ** -24
        assert not _number_equal(midpoint - 2.0 ** -45, midpoint + 2.0 ** -45)
        assert not _number_equal(2**53 + 1, float(2**53 + 1))

        class Fractional(torch.nn.Linear):
            def __init__(self, power):
                super().__init__(3, 5)
                self.power = power

            def forward(self, x):
                scalar = .1 * x.shape[0] if self.power == 1 else .1 / x.shape[0]
                return super().forward(x) * scalar

        for power in (1, -1):
            model = Fractional(power)
            x, y = torch.ones(8, 3), torch.zeros(8, 5)
            captured = capture_module(model, objective, (x,), (y,), dynamic_batch=DynamicBatch(8))
            assert any(isinstance(op.args[1], dict)
                       and op.args[1].get('dynamic_batch_scalar', {}).get('power') == power
                       for op in captured.ir.operations if len(op.args) > 1)
            for batch in (1, 3, 7, 8):
                result = captured.unpack(captured.step_fn(*captured.example_args[:-1],
                                                         torch.tensor(batch, dtype=torch.int64)))
                loss = objective(model(x[:batch]), y[:batch])
                grads = torch.autograd.grad(loss, tuple(model.parameters()))
                torch.testing.assert_close(result['loss'], loss)
                for observed, expected in zip(result['grads'].values(), grads):
                    torch.testing.assert_close(observed, expected)

    check('fractional_batch_scalars_preserve_final_fp32_bits', fractional_scalar_rules)

    def scalar_sum_over_batch():
        model = torch.nn.Linear(3, 5)
        x, y = torch.randn(8, 3), torch.randn(8, 5)

        def loss_fn(prediction, target):
            return (prediction - target).square().sum() / prediction.shape[0]

        captured = capture_module(model, loss_fn, (x,), (y,), dynamic_batch=DynamicBatch(8))
        loss = captured.ir.outputs['loss']
        producer = next(op for op in captured.ir.operations if loss in op.outputs)
        assert producer.args[1] == {'dynamic_batch_scalar': {'coefficient': 1, 'power': 1}}
        assert loss not in captured.ir.metadata['dynamic_batch']['value_axes']
        for batch in (1, 3, 7, 8):
            result = captured.unpack(captured.step_fn(*captured.example_args[:-1],
                                                     torch.tensor(batch, dtype=torch.int64)))
            expected = loss_fn(model(x[:batch]), y[:batch])
            gradients = torch.autograd.grad(expected, tuple(model.parameters()))
            torch.testing.assert_close(result['loss'], expected)
            for observed, gradient in zip(result['grads'].values(), gradients):
                torch.testing.assert_close(observed, gradient)

    check('explicit_sum_over_batch_loss_uses_runtime_divisor', scalar_sum_over_batch)

    def explicit_rng():
        class Dropped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x, *, state):
                value, rng = dropout(torch.tanh(self.linear(x)), state['rng'], .25)
                return value, {'rng': rng}

        captured = capture_module(Dropped(), objective, (torch.ones(8, 3),), (torch.zeros(8, 5),),
                                  state={'rng': make_rng_state(42)}, dynamic_batch=DynamicBatch(8))
        advance = next(op for op in captured.ir.operations if op.target == 'latch_ops.philox_advance.default')
        assert advance.args[1] == {'dynamic_batch_scalar': {'coefficient': 5, 'power': 1, 'ceil_divisor': 4}}
        for batch in (1, 3, 7, 8):
            result = captured.unpack(captured.step_fn(*captured.example_args[:-1],
                                                     torch.tensor(batch, dtype=torch.int64)))
            assert int(result['state']['rng'][2]) == (batch * 5 + 3) // 4

    check('explicit_dropout_counter_uses_active_elements', explicit_rng)

    def refused_shapes_and_control():
        class Interior(torch.nn.Linear):
            def forward(self, x):
                result = super().forward(x)
                return -result if x.shape[0] == 3 else result

        class NonlinearScalar(torch.nn.Linear):
            def forward(self, x):
                return super().forward(x) / (x.shape[0] + 1)

        for model in (Interior(4, 4), NonlinearScalar(4, 4)):
            try:
                capture_module(model, objective, (torch.ones(4, 4),), (torch.zeros(4, 4),),
                               dynamic_batch=DynamicBatch(4))
            except CaptureError as error:
                assert error.code == 'DYNAMIC_BATCH'
            else:
                raise AssertionError('unsupported shape control or scalar admitted')
        for options in ({'maximum': 1}, {'maximum': 33}, {'maximum': True}, {'maximum': 4, 'axis': 1}):
            try:
                DynamicBatch(**options)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid annotation admitted')

    check('interior_control_nonaffine_scalar_and_annotation_refusals', refused_shapes_and_control)

    def static_capacity_broadcast_refused():
        class CapacityBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(3))
                self.register_buffer('rows', torch.tensor([[1., 2., 3.], [4., 5., 6.]]))

            def forward(self, x):
                # At B=1 eager broadcasts x row zero into both rows. A loop
                # inferred solely from this static output shape would instead
                # read inactive x row one from capacity storage.
                return (x + self.rows) * self.weight

        try:
            capture_module(CapacityBuffer(), lambda prediction: prediction.square().mean(),
                           (torch.ones(2, 3),), dynamic_batch=DynamicBatch(2))
        except CaptureError as error:
            assert error.code == 'DYNAMIC_BATCH'
        else:
            raise AssertionError('active input broadcast into static capacity was admitted')

    check('static_capacity_broadcast_cannot_erase_active_axis', static_capacity_broadcast_refused)

    def bound_update():
        runtime = Runtime.__new__(Runtime)
        runtime.manifest = {'dynamic_batch': {'maximum': 8, 'input_index': 2, 'input': 'batch'}}
        runtime.args = (torch.full((8, 3), float('nan')), torch.full((8, 2), float('nan')),
                        torch.tensor(8, dtype=torch.int64))
        runtime.active_batch_tensor = runtime.args[-1]
        runtime.active_batch = 8
        runtime.torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda _: nullcontext(),
                                         is_current_stream_capturing=lambda: False))
        runtime.device = 'cpu'
        runtime.program = SimpleNamespace(cache_key='bounded-program')
        runtime.parameters = {'w': torch.ones(2, 3)}
        runtime.optimizer_state = {'w': torch.zeros(2, 3)}
        runtime.state = {'rng': make_rng_state(71)}
        metadata = {'input_indices': {'inputs': [0], 'targets': [1], 'buffers': {}, 'active_batch': 2},
                    'immutable_bindings_sha256': 'frozen-values'}
        return ModuleUpdate(runtime, metadata)

    def staging_and_snapshot():
        update = bound_update()
        pointers = [value.data_ptr() for value in update.args]
        update.stage((torch.ones(3, 3),), (torch.zeros(3, 2),))
        assert update.active_batch == 3 and update.active_batch_tensor.item() == 3
        assert torch.isnan(update.args[0][3:]).all() and torch.isnan(update.args[1][3:]).all()
        saved = update.state_dict()
        update.set_active_batch(7)
        update.parameters['w'].fill_(9)
        update.state['rng'][2] = 99
        update.load_state_dict(saved)
        assert update.active_batch == 3 and update.active_batch_tensor.item() == 3
        assert torch.equal(update.parameters['w'], saved['parameters']['w'])
        assert torch.equal(update.state['rng'], saved['state']['rng'])
        assert pointers == [value.data_ptr() for value in update.args]
        for batch in (0, 9, True, 2.0):
            try:
                update.set_active_batch(batch)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid active batch admitted')
        for targets in ((torch.zeros(4, 2),), (torch.zeros(3, 3),)):
            before = update.args[0].clone()
            try:
                update.stage((torch.full((3, 3), 99.),), targets)
            except ValueError:
                pass
            else:
                raise AssertionError('inconsistent staging admitted')
            torch.testing.assert_close(update.args[0], before, equal_nan=True)
        invalid = update.state_dict()
        invalid['active_batch'].fill_(0)
        invalid['parameters']['w'].fill_(100)
        try:
            update.load_state_dict(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid snapshot batch admitted')
        torch.testing.assert_close(update.parameters['w'], saved['parameters']['w'])
        update.runtime.torch.cuda.is_current_stream_capturing = lambda: True
        try:
            update.set_active_batch(4)
        except RuntimeError:
            pass
        else:
            raise AssertionError('active scalar setter was captured into CUDA graph')

    check('stable_staging_snapshot_and_atomic_invalid_input_refusal', staging_and_snapshot)

    def live_reduction_addresses_and_elided_storage():
        from megakernel_lab.operation_compiler.dynamic_codegen import ActiveExtents
        from megakernel_lab.operation_compiler.lower import TensorLayout, lower
        extent = ActiveExtents({'maximum': 32, 'input': 'batch', 'value_axes': {'source': 0}},
                               {'source': {'shape': (32, 7)}})
        source = TensorLayout((32, 7), (7, 1), 'source')
        full = extent.reduction('loss', 'source', source, [0, 1], (), 'float32', 'float32', True)
        assert 'v_source[r]' in full and '% latch_active_batch' not in full
        columns = extent.reduction('bias', 'source', source, [0], (7,), 'float32', 'float32', False)
        assert 'v_source[r * 7LL + o]' in columns
        # A transposed reduction must keep its logical leaf order even though
        # the backing storage is contiguous.
        extent.axes['source'] = 1
        transpose = TensorLayout((7, 32), (1, 7), 'source')
        transposed = extent.reduction('loss', 'source', transpose, [0, 1], (), 'float32', 'float32', True)
        assert 'v_source[r]' not in transposed

        captured = capture_module(torch.nn.Linear(3, 5), objective,
                                  (torch.ones(4, 3),), (torch.zeros(4, 5),),
                                  dynamic_batch=DynamicBatch(4))
        for execution in (None, 'compose'):
            program = lower(captured.ir, execution=execution)
            assert program.manifest['elided_stores']
            roots = set(program.manifest['inputs']) | set(program.manifest['offsets'])
            assert all(layout['root'] in roots for layout in program.manifest['layouts'].values())
            retained = lower(captured.ir, execution=execution, retain_all=True)
            assert not retained.manifest.get('elided_stores')

    check('active_reduction_addressing_and_elided_runtime_storage', live_reduction_addresses_and_elided_storage)

    def static_k_splits_have_live_private_storage():
        from megakernel_lab.operation_compiler.lower import lower
        captured = capture_module(torch.nn.Linear(256, 64), objective,
                                  (torch.ones(32, 256),), (torch.zeros(32, 64),),
                                  dynamic_batch=DynamicBatch(32))
        before = captured.ir.to_dict()
        program = lower(captured.ir, execution='compose')
        phases = program.manifest['phases']
        partials = [(i, phase) for i, phase in enumerate(phases)
                    if phase.get('engine') == 'popsc_fp32_split_edge_dynamic']
        assert partials
        interval = {phase: i for i, members in enumerate(program.manifest['barrier_intervals'])
                    for phase in members}
        for index, phase in partials:
            scratch, = phase['writes']
            layout = program.manifest['layouts'][scratch]
            assert layout['shape'][0] == 32 and layout['active_axis'] == 0
            assert phase['ksplit'] in (2, 4, 8)
            consumers = [(i, p) for i, p in enumerate(phases) if scratch in p['reads']]
            assert len(consumers) == 1
            consumer, reduction = consumers[0]
            assert reduction['engine'] == 'bounded_fixed_slice_sum'
            assert interval[index] < interval[consumer]
            assert scratch in program.manifest['offsets']
            assert program.manifest['dynamic_batch']['internal_value_axes'][scratch] == 0
        # Compiler-owned scratch does not change the captured graph or turn
        # capacity examples into a runtime specialization table.
        assert captured.ir.to_dict() == before
        disabled = lower(captured.ir, execution='compose', split_k=False)
        assert not disabled.manifest['dynamic_batch'].get('internal_value_axes')

    check('fixed_static_k_splits_publish_only_live_private_rows', static_k_splits_have_live_private_storage)

    def bf16_active_k_uses_one_hardware_block_below_seventeen():
        from megakernel_lab.operation_compiler.dynamic_codegen import ActiveExtents
        from megakernel_lab.operation_compiler.lower import TensorLayout
        # Aligned but ragged M stays on the general masked tile path.
        values = {'a': {'shape': (24, 32)}, 'b': {'shape': (32, 64)}, 'out': {'shape': (24, 64)}}
        extent = ActiveExtents({'maximum': 32, 'input': 'batch', 'value_axes': {'a': 1, 'b': 0}}, values)
        layouts = {'a': TensorLayout((24, 32), (1, 24), 'a'),
                   'b': TensorLayout((32, 64), (64, 1), 'b'),
                   'out': TensorLayout((24, 64), (64, 1), 'out')}
        dtypes = dict.fromkeys(layouts, 'bfloat16')
        code, _, metadata = extent.matrix('a', 'b', 'out', layouts, dtypes)
        assert 'bmm_edge_phase<32,64,2,4,false,false,16,3,true>' in code
        assert metadata['stage_depth'] == 16 and metadata['shared_bytes'] == 10752
        assert metadata['hardware_k16_blocks'] == 'ceil(active_batch / 16)'
        # Static-K forward products retain the previous 32-deep stage.
        layouts['a'] = TensorLayout((32, 64), (64, 1), 'a')
        layouts['b'] = TensorLayout((64, 64), (64, 1), 'b')
        layouts['out'] = TensorLayout((32, 64), (64, 1), 'out')
        extent.axes = {'a': 0, 'out': 0}
        code, _, metadata = extent.matrix('a', 'b', 'out', layouts, dtypes)
        assert 'bmm_edge_phase<16,64,1,8,true,false,32,3,true>' in code
        assert metadata['stage_depth'] == 32 and metadata['hardware_k16_blocks'] == 4

    check('bf16_active_k_scales_at_hardware_k16_granularity', bf16_active_k_uses_one_hardware_block_below_seventeen)

    def bf16_capacity_tiles_keep_whole_grid_and_runtime_k16():
        from megakernel_lab.operation_compiler.dynamic_codegen import ActiveExtents
        from megakernel_lab.operation_compiler.lower import TensorLayout
        for m, n, bm, bn, wm, wn, shared in (
                (1024, 256, 64, 32, 4, 2, 10752),
                (512, 1024, 64, 64, 2, 4, 13824)):
            values = {'a': {'shape': (m, 32)}, 'b': {'shape': (32, n)}, 'out': {'shape': (m, n)}}
            extent = ActiveExtents({'maximum': 32, 'input': 'batch', 'value_axes': {'a': 1, 'b': 0}}, values)
            layouts = {'a': TensorLayout((m, 32), (1, m), 'a'),
                       'b': TensorLayout((32, n), (n, 1), 'b'),
                       'out': TensorLayout((m, n), (n, 1), 'out')}
            code, _, metadata = extent.matrix('a', 'b', 'out', layouts, dict.fromkeys(layouts, 'bfloat16'))
            assert f'bmm_edge_phase<{bm},{bn},{wm},{wn},false,false,16,3,true>' in code
            assert metadata['capacity_tile_selection'] == {
                'bm': bm, 'bn': bn, 'wm': wm, 'wn': wn, 'tiles': 128, 'grid': 128}
            assert metadata['tiles'] == 128 and metadata['shared_bytes'] == shared
            assert metadata['work_extent'] == [str(m), str(n), 'latch_active_batch']
            assert metadata['stage_depth'] == 16 and metadata['pipeline_stages'] == 3
            assert metadata['hardware_k16_blocks'] == 'ceil(active_batch / 16)'
            assert '(g, smem);' in code  # whole physical grid, no borrowed CTA partition

    check('bf16_capacity_tile_geometry_retains_existing_whole_grid_plan', bf16_capacity_tiles_keep_whole_grid_and_runtime_k16)
    return {'passed': all(result['passed'] for result in results.values()),
            'contract_count': len(results), 'tests': results}
