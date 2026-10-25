"""Module-API contracts; run on a CPU worker with the pinned PyTorch runtime."""
from __future__ import annotations

import traceback
from types import SimpleNamespace
import torch

from megakernel_lab.operation_compiler.api import capture_module, ModuleUpdate, _immutable_identity
from megakernel_lab.operation_compiler.capture import SGDConfig
from megakernel_lab.operation_compiler.ir import CaptureError


def run():
    torch.set_num_threads(1)
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {'passed': True}
        except Exception:
            results[name] = {'passed': False, 'traceback': traceback.format_exc()}

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(8, 8, bias=False)
            self.right = torch.nn.Linear(8, 8, bias=False)
            self.right.weight = self.left.weight
            self.register_buffer('offset', torch.arange(8, dtype=torch.float32) * .01)
            self.register_buffer('same_offset', self.offset)

        def forward(self, x):
            return self.right(torch.tanh(self.left(x) + self.offset)) + self.same_offset

    torch.manual_seed(42026)
    inputs, target = torch.randn(4, 8), torch.randn(4, 8)
    objective = lambda prediction, target: (prediction - target).square().mean()

    def tied_and_buffers():
        model = Tied()
        captured = capture_module(model, objective, (inputs,), (target,))
        assert captured.parameter_names == ('left.weight',)
        assert captured.parameter_aliases['right.weight'] == 'left.weight'
        assert captured.buffer_aliases == {'offset': 'offset', 'same_offset': 'offset'}
        assert len(captured.input_indices['buffers']) == 1
        observed = captured.unpack(captured.graph_module(*captured.example_args))
        loss = objective(model(inputs), target)
        derivative, = torch.autograd.grad(loss, (model.left.weight,))
        assert torch.allclose(observed['loss'], loss, atol=1e-7, rtol=1e-6)
        assert torch.allclose(observed['grads']['left.weight'], derivative, atol=1e-7, rtol=1e-6)
        assert set(observed['momentum']) == {'left.weight'}
        assert set(observed['optimizer_state']) == {'left.weight'}
        saved = observed['loss'].clone()
        model.offset.fill_(300)
        observed2 = captured.unpack(captured.graph_module(*captured.example_args))
        assert torch.equal(observed2['loss'], saved), 'read-only buffer snapshot was not frozen'

    check('tied_parameters_and_buffer_snapshots', tied_and_buffers)

    def structured_output():
        class Pair(Tied):
            def forward(self, x):
                value = super().forward(x)
                return {'prediction': value, 'intermediate': value.square()}
        captured = capture_module(Pair(), lambda output, y: objective(output['prediction'], y)
                                  + .1 * output['intermediate'].mean(), (inputs,), (target,))
        expected = captured.step_fn(*captured.example_args)
        actual = captured.graph_module(*captured.example_args)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))

    check('structured_forward_output', structured_output)

    def explicit_state():
        class Counter(Tied):
            def forward(self, x, *, state):
                return super().forward(x), {'calls': state['calls'] + 1}
        captured = capture_module(Counter(), objective, (inputs,), (target,),
                                  state={'calls': torch.tensor(11, dtype=torch.int64)})
        output = captured.unpack(captured.graph_module(*captured.example_args))
        assert output['state']['calls'].item() == 12
        assert captured.ir.metadata['module_api']['stateful_forward'] is True
        assert captured.state_names == ('calls',)

    check('explicit_model_state', explicit_state)

    def mutation_refused():
        class Mutating(Tied):
            def forward(self, x):
                self.offset.add_(1)
                return super().forward(x)
        try:
            capture_module(Mutating(), objective, (inputs,), (target,))
        except CaptureError as error:
            assert error.code in {'MUTATION', 'TRACE'}
        else:
            raise AssertionError('module buffer mutation admitted')

    check('readonly_buffer_mutation_refused', mutation_refused)

    def factory_constant():
        class Constant(Tied):
            def forward(self, x):
                return super().forward(x) + torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8])
        from megakernel_lab.operation_compiler.lower import lower
        captured = capture_module(Constant(), objective, (inputs,), (target,))
        generated = lower(captured.ir)
        assert captured.constants
        assert len(generated.manifest['inputs']) == len(captured.example_args)
        assert all(value.role != 'constant' for value in captured.ir.values.values())

    check('ordinary_tensor_constant_native_binding', factory_constant)

    def adamw_state_separate():
        from megakernel_lab.operation_compiler.optim import AdamWConfig
        captured = capture_module(Tied(), objective, (inputs,), (target,), optimizer=AdamWConfig())
        output = captured.unpack(captured.graph_module(*captured.example_args))
        assert set(output['optimizer_state']) == {
            'exp_avg:left.weight', 'exp_avg_sq:left.weight', 'step:left.weight'}
        assert output['optimizer_state']['step:left.weight'].item() == 1
        assert not captured.state_names
        assert len(captured.parameter_names) == 1

    check('adamw_tied_state_separate', adamw_state_separate)

    def mutable_buffer_alias_refused():
        for group in ('parameter', 'optimizer', 'state'):
            model = Tied()
            extra = {}
            if group == 'parameter':
                alias = model.left.weight.detach()
            elif group == 'optimizer':
                alias = torch.zeros_like(model.left.weight)
                extra['optimizer_state'] = {'left.weight': alias}
            else:
                alias = torch.tensor(1, dtype=torch.int64)
                extra['state'] = {'count': alias}
            model.register_buffer('aliased_mutable', alias)
            try:
                capture_module(model, objective, (inputs,), (target,), **extra)
            except CaptureError as error:
                assert error.code == 'MODULE_BUFFER_ALIAS'
            else:
                raise AssertionError(f'{group} buffer storage alias admitted')

    check('buffer_alias_to_mutable_storage_refused', mutable_buffer_alias_refused)

    def bound_update(*, optimizer_state=None, schema=()):
        runtime = SimpleNamespace(
            program=SimpleNamespace(cache_key='same-source'),
            args=(torch.tensor([1., 2.]), torch.tensor([3., 4.])),
            parameters={'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])},
            optimizer_state=optimizer_state or {}, state={},
            manifest={'optimizer_state_schema': schema})
        metadata = {'input_indices': {'inputs': [0, 1], 'targets': [], 'buffers': {}},
                    'immutable_bindings_sha256': 'frozen-values'}
        return ModuleUpdate(runtime, metadata)

    def simultaneous_alias_copies():
        update = bound_update()
        update.stage(tuple(reversed(update.runtime.args)))
        assert torch.equal(update.runtime.args[0], torch.tensor([3., 4.]))
        assert torch.equal(update.runtime.args[1], torch.tensor([1., 2.]))
        state = update.state_dict()
        a, b = update.runtime.parameters.values()
        state['parameters'] = {'a': b[:], 'b': a[:]}
        update.load_state_dict(state)
        assert torch.equal(a, torch.tensor([3., 4.])) and torch.equal(b, torch.tensor([1., 2.]))
        assert update.runtime.parameters['a'] is a and update.runtime.parameters['b'] is b

    check('staging_and_restore_have_simultaneous_alias_semantics', simultaneous_alias_copies)

    def optimizer_restore_admission():
        fields = {'exp_avg:a': torch.zeros(2), 'exp_avg_sq:a': torch.ones(2),
                  'step:a': torch.tensor(9, dtype=torch.int64)}
        schema = [{'name': name, 'kind': name.split(':')[0]} for name in fields]
        update = bound_update(optimizer_state=fields, schema=schema)
        for name, bad in [('exp_avg:a', float('nan')), ('exp_avg_sq:a', -.1),
                          ('exp_avg_sq:a', float('inf')), ('step:a', 2**24)]:
            state = update.state_dict()
            state['parameters']['a'].fill_(99)
            state['optimizer_state'][name].fill_(bad)
            try:
                update.load_state_dict(state)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid restored AdamW state admitted')
            assert torch.equal(update.runtime.parameters['a'], torch.tensor([1., 2.])), 'partial restore occurred'
        # A legal SGD parameter name must not be mistaken for an AdamW counter.
        update = bound_update(optimizer_state={'step:weight': torch.zeros(2)},
                              schema=[{'name': 'step:weight', 'kind': 'momentum'}])
        state = update.state_dict()
        state['optimizer_state']['step:weight'].fill_(.5)
        update.load_state_dict(state)
        assert torch.equal(update.optimizer_state['step:weight'], torch.full((2,), .5))

    check('optimizer_restore_schema_and_atomic_refusals', optimizer_restore_admission)

    def immutable_checkpoint_identity():
        args = (torch.tensor([1., 2.]), torch.tensor([.25, .5]), torch.tensor([.75]))
        manifest = {'inputs': ['p', 'buffer', 'constant'], 'input_roles': {
            'p': {'role': 'parameter'}, 'buffer': {'role': 'input'},
            'constant': {'role': 'constant_input'}}}
        metadata = {'input_indices': {'buffers': {'offset': 1}}}
        identity = _immutable_identity(args, manifest, metadata)
        changed_parameter = (args[0] + 5, *args[1:])
        assert _immutable_identity(changed_parameter, manifest, metadata) == identity
        for index in (1, 2):
            different = list(args)
            different[index] = args[index] + 1
            assert _immutable_identity(different, manifest, metadata) != identity
        update = bound_update()
        snapshot = update.state_dict()
        snapshot['immutable_bindings_sha256'] = 'different-frozen-values'
        snapshot['parameters']['a'].fill_(99)
        try:
            update.load_state_dict(snapshot)
        except ValueError:
            pass
        else:
            raise AssertionError('cross-buffer checkpoint restore admitted')
        assert torch.equal(update.parameters['a'], torch.tensor([1., 2.]))

    check('checkpoint_identity_includes_frozen_buffers_and_constants', immutable_checkpoint_identity)
    return {'passed': all(row['passed'] for row in results.values()),
            'contract_count': len(results), 'tests': results}
