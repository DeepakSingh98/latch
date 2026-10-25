"""Prepared batch variants bind CPU artifacts before loading any CUDA runtime."""
import hashlib
import json

import pytest
import torch

from megakernel_lab.operation_compiler.variants import BatchVariants, VariantContractError, persistent_keys, _identity_value
from megakernel_lab.operation_compiler.user_optimizer import tensor_update


class FrozenModel(torch.nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3, 2))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        self.register_buffer('offset', torch.zeros(2))
        self.scale = scale

    def forward(self, x):
        return (x @ self.weight + self.frozen + self.offset) * self.scale


def custom_update(params, grads, state, hparams):
    return {name: value - hparams['lr'] * grads[name] for name, value in params.items()}, state


def manager(tmp_path, **kwargs):
    return BatchVariants(FrozenModel(), torch.nn.functional.mse_loss,
                         optimizer=tensor_update(custom_update, hyperparameters={'lr': .01}),
                         device='cpu', work_dir=tmp_path / 'work', cache_dir=tmp_path / 'cache',
                         execution='compose', **kwargs)


def fake_builder(self, batch, inputs, targets, directory):
    directory.mkdir(parents=True)
    data = b'bound CPU payload'
    (directory / 'bindings.pt').write_bytes(data)
    receipt = {'source_hash': 'a' * 64, 'files': {'bindings.pt': hashlib.sha256(data).hexdigest()}}
    (directory / 'artifact.json').write_text(json.dumps(receipt))
    return receipt, {'build_seconds': 0.}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(BatchVariants, '_build_artifact', fake_builder)
    producer = manager(tmp_path)
    inputs, targets = (torch.zeros(2, 3),), (torch.zeros(2, 2),)
    directory = producer.prepare_artifact(inputs, targets)
    return tmp_path, directory, inputs, targets


def test_prepared_identity_survives_process_configuration_changes(prepared):
    tmp_path, directory, inputs, targets = prepared
    consumer = manager(tmp_path / 'other', prepared_artifacts={2: directory})
    consumer.device = torch.device('cuda')
    key, batch = consumer.signature(inputs, targets)
    assert consumer._read_prepared(key, batch)[0] == directory
    assert consumer.counters['captures'] == consumer.counters['native_compilations'] == 0


@pytest.mark.parametrize('change', ['weights', 'frozen', 'buffer', 'config', 'optimizer', 'dtype', 'batch'])
def test_prepared_rejects_mismatched_model_state_and_data(prepared, change):
    tmp_path, directory, inputs, targets = prepared
    consumer = manager(tmp_path / 'other', prepared_artifacts={2: directory, 3: directory})
    if change in {'weights', 'frozen', 'buffer'}:
        tensor = {'weights': consumer.model.weight, 'frozen': consumer.model.frozen,
                  'buffer': consumer.model.offset}[change]
        with torch.no_grad():
            tensor.add_(1)
    elif change == 'config':
        consumer.model.scale = 2.
    elif change == 'optimizer':
        consumer._optimizer.hyperparameters['lr'] = .02
        consumer.identity['optimizer'] = consumer._optimizer_identity()
    elif change == 'dtype':
        inputs = (inputs[0].double(),)
    else:
        inputs, targets = (torch.zeros(3, 3),), (torch.zeros(3, 2),)
    key, batch = consumer.signature(inputs, targets)
    with pytest.raises(VariantContractError, match='differs'):
        consumer._read_prepared(key, batch)


@pytest.mark.parametrize('name', ['artifact.json', 'bindings.pt'])
def test_prepared_rejects_artifact_tampering(prepared, name):
    tmp_path, directory, inputs, targets = prepared
    consumer = manager(tmp_path / 'other', prepared_artifacts={2: directory})
    (directory / name).write_bytes((directory / name).read_bytes() + b'changed')
    key, batch = consumer.signature(inputs, targets)
    with pytest.raises(VariantContractError, match='changed'):
        consumer._read_prepared(key, batch)


def test_prepared_missing_batch_never_enters_builder(prepared, monkeypatch):
    tmp_path, directory, inputs, targets = prepared
    consumer = manager(tmp_path / 'other', prepared_artifacts={3: directory})
    monkeypatch.setattr(consumer, '_build_artifact', lambda *a: pytest.fail('must not compile'))
    with pytest.raises(VariantContractError, match='compilation is disabled'):
        consumer.prepare(inputs, targets)
    with pytest.raises(VariantContractError, match='load-only'):
        consumer.prepare_artifact(inputs, targets)


def test_prepare_artifact_refuses_gpu_manager(tmp_path):
    producer = manager(tmp_path)
    producer.device = torch.device('cuda')
    with pytest.raises(VariantContractError, match='requires device'):
        producer.prepare_artifact((torch.zeros(2, 3),), (torch.zeros(2, 2),))


def test_frozen_bindings_participate_in_shared_contract():
    manifest = {'inputs': ['weight', 'data', 'frozen', 'buffer'],
                'input_roles': {'weight': {'role': 'parameter', 'parameter': 'weight'},
                                'data': {'role': 'input'}, 'frozen': {'role': 'constant_input'},
                                'buffer': {'role': 'input'}}}
    metadata = {'input_indices': {'buffers': {'offset': 3}, 'frozen': {'frozen': 2}}}
    assert persistent_keys(manifest, metadata) == {0: ('parameter', 'weight'),
                                                  2: ('frozen', 'frozen'), 3: ('buffer', 'offset')}


def test_dropout_contract_allows_batch_extent_but_preserves_policy(tmp_path):
    producer = manager(tmp_path)

    def contract(batch, probability=.25):
        receipt = {'optimizer': {}, 'parameter_aliases': {},
                   'module_api': {'buffer_aliases': {}, 'stateful_forward': False,
                                  'dropout': {'seed': 17,
                                              'sites': [{'name': 'drop', 'p': probability}],
                                              'calls': [{'site': 'drop', 'shape': [batch, 8],
                                                         'blocks': batch * 2}]}}}
        return producer._state_contract({}, receipt, {}, ())

    assert contract(2) == contract(4)
    assert contract(2) != contract(2, probability=.5)


def test_source_free_callable_identity_binds_constants_and_nested_code():
    scopes = []
    for constant in (1., 2.):
        scope = {}
        exec(f'def rule(values):\n    return {{key: value + {constant} for key, value in values.items()}}', scope)
        scopes.append(scope)
    assert scopes[0]['rule'].__code__.co_code == scopes[1]['rule'].__code__.co_code
    assert _identity_value(scopes[0]['rule']) != _identity_value(scopes[1]['rule'])
