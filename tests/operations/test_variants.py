"""Batch-variant entry-point contracts on a CPU worker (pinned PyTorch); no CUDA runtime is used."""
from __future__ import annotations

import tempfile
import traceback

import torch
from torch import nn

from megakernel_lab.operation_compiler.api import capture_module
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.optim import AdamWConfig
from megakernel_lab.operation_compiler.random import dropout, make_rng_state
from megakernel_lab.operation_compiler.variants import BatchVariants, VariantContractError, persistent_keys


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1818)
        self.inp = nn.Linear(12, 24)
        self.out = nn.Linear(24, 5)
        self.register_buffer('gain', torch.linspace(.9, 1.1, 24))

    def forward(self, x, *, state):
        h, rng = dropout(torch.tanh(self.inp(x)) * self.gain, state['rng'], p=.25)
        return self.out(h), {'rng': rng}


def loss(prediction, target):
    return torch.nn.functional.cross_entropy(prediction.reshape(-1, 5), target.reshape(-1))


def batch(size, tokens=7, features=12, dtype=torch.float32):
    return (torch.randn(size, tokens, features, dtype=dtype),), (torch.zeros(size, tokens, dtype=torch.int64),)


def trainer(model=None, optimizer=None, **options):
    directory = tempfile.mkdtemp(prefix='latch-variants-test-')
    return BatchVariants(model or Small(), loss, work_dir=directory, cache_dir=directory,
                         optimizer=optimizer or AdamWConfig(eps=1e-4), state={'rng': make_rng_state(3)}, **options)


def refused(fn, text):
    try:
        fn()
    except VariantContractError as error:
        assert text in str(error), str(error)
        return
    raise AssertionError('expected VariantContractError: ' + text)


def test_batch_changes_select_distinct_variants():
    t = trainer()
    k2, b2 = t.signature(*batch(2))
    k4, b4 = t.signature(*batch(4))
    again, _ = t.signature(*batch(2))
    assert (b2, b4) == (2, 4) and k2 != k4 and again == k2


def test_non_batch_changes_are_refused():
    t = trainer()
    t.signature(*batch(2))
    refused(lambda: t.signature(*batch(4, tokens=8)), 'non-batch shape')
    refused(lambda: t.signature(*batch(4, features=13)), 'non-batch shape')
    refused(lambda: t.signature(*batch(4, dtype=torch.float64)), 'dtype')
    refused(lambda: t.signature(batch(4)[0]), 'arity')
    x, y = batch(4)
    refused(lambda: t.signature(x, (y[0][:3],)), 'share one batch size')
    refused(lambda: t.signature((torch.tensor(1.),), ()), 'no batch dimension')


def test_declared_batch_set_and_first_refusal_does_not_fix_contract():
    t = trainer(batch_sizes=(2, 4))
    refused(lambda: t.signature(*batch(8)), 'declared set')
    t.signature(*batch(2, tokens=9))          # the refused request fixed nothing
    refused(lambda: t.signature(*batch(4)), 'non-batch shape')


def test_explicit_batch_dimensions():
    t = trainer(batch_dims=(1, None))
    first = ((torch.randn(7, 2, 12),), (torch.zeros(7, dtype=torch.int64),))
    second = ((torch.randn(7, 6, 12),), (torch.zeros(7, dtype=torch.int64),))
    assert t.signature(*first)[1] == 2 and t.signature(*second)[1] == 6
    refused(lambda: t.signature((torch.randn(7, 6, 12),), (torch.zeros(8, dtype=torch.int64),)), 'non-batch shape')


def test_module_and_optimizer_changes_are_refused():
    model = Small()
    t = trainer(model)
    t._check_module()
    with torch.no_grad():
        model.inp.weight.add_(1)
    refused(t._check_module, 'module parameters')
    model = Small()
    t = trainer(model)
    model.eval()
    refused(t._check_module, 'train/eval')
    optimizer = AdamWConfig(eps=1e-4, param_options={})
    t = trainer(optimizer=optimizer)
    optimizer.param_options['inp.weight'] = {'lr': .5}   # frozen dataclass, mutable field
    refused(t._check_module, 'optimizer configuration')


def test_variant_programs_share_one_state_contract():
    for optimizer in (AdamWConfig(eps=1e-4), AdamWConfig(eps=1e-4, runtime_lr=True)):
        rows = {}
        for size in (2, 4):
            inputs, targets = batch(size)
            captured = capture_module(Small(), loss, inputs, targets, optimizer=optimizer,
                                      state={'rng': make_rng_state(3)})
            program = lower(captured.ir, target_arch='sm_90')
            module_api = captured.ir.metadata['module_api']
            keys = persistent_keys(program.manifest, module_api)
            layouts = {key: (tuple(captured.example_args[i].shape), captured.example_args[i].stride(),
                             captured.example_args[i].dtype) for i, key in keys.items()}
            data = [tuple(captured.example_args[i].shape) for i in module_api['input_indices']['inputs']]
            rows[size] = (layouts, data, program.cache_key, program.manifest.get('optimizer_state_schema'))
        assert rows[2][0] == rows[4][0] and rows[2][3] == rows[4][3]
        roles = {role for role, _ in rows[2][0]}
        assert {'parameter', 'optimizer_state', 'state', 'buffer'} <= roles
        assert ('optimizer_lr', '') in rows[2][0] if optimizer.runtime_lr else ('optimizer_lr', '') not in rows[2][0]
        assert rows[2][1] != rows[4][1] and rows[2][2] != rows[4][2]


def run():
    torch.set_num_threads(1)
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                results[name] = {'passed': True}
            except Exception:
                results[name] = {'passed': False, 'traceback': traceback.format_exc()}
    return {'passed': all(r['passed'] for r in results.values()), 'tests': results,
            'scope': 'batch-variant signature/module/state contracts; capture and lowering on CPU only'}
