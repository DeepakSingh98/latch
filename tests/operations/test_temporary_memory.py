"""Release step 15 CPU contracts: consumed gradients and late-remat admission (lowering only, no native build)."""
from __future__ import annotations

import pytest
import torch

from megakernel_lab.operation_compiler.capture import SGDConfig
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.gradient_consumption import consume


def _transformer_ir():
    from megakernel_lab.fast_examples.recovered.transformer import TransformerConfig, parameters, fixture, make_loss
    from megakernel_lab.operation_compiler.capture import capture_training
    torch.manual_seed(0)
    config = TransformerConfig(batch=2, sequence=256, width=128, heads=2, layers=2, matrix_dtype='bfloat16')
    params = parameters(config)
    momentum = {name: torch.zeros_like(value) for name, value in params.items()}
    captured = capture_training(make_loss(config), params, momentum, fixture(config),
                                SGDConfig(lr=0.01, momentum=0.9, weight_decay=1e-4))
    return captured.ir


@pytest.fixture(scope='module')
def ir():
    torch.set_num_threads(1)
    return _transformer_ir()


def _lower(ir, **options):
    return lower(ir, policy='auto', precision_policy='practical', **options)


def test_default_is_unchanged(ir):
    base = _lower(ir)
    explicit = _lower(ir, gradient_outputs='returned')
    assert base.source == explicit.source and base.manifest['arena_bytes'] == explicit.manifest['arena_bytes']
    assert 'gradient_outputs' not in base.manifest
    assert set(base.manifest['structured_outputs']['grads']) == set(ir.metadata['parameter_names'])


def test_consumed_removes_public_gradients_only(ir):
    base, consumed = _lower(ir), _lower(ir, gradient_outputs='consumed')
    structured = consumed.manifest['structured_outputs']
    assert 'grads' not in structured
    assert structured['params'] == base.manifest['structured_outputs']['params']
    assert len(consumed.manifest['outputs']) == len(base.manifest['outputs']) - len(ir.metadata['parameter_names'])
    record = consumed.manifest['gradient_outputs']
    assert record['policy'] == 'consumed' and record['removed_public_outputs'] == len(ir.metadata['parameter_names'])
    assert consumed.manifest['arena_bytes'] <= base.manifest['arena_bytes']
    # The state commit set is unchanged: the same parameter/momentum inputs are written.
    assert consumed.manifest['state_writes'].keys() == base.manifest['state_writes'].keys()
    # Same selected flat schedule (no slower fallback).
    assert consumed.manifest['performance_schedule']['selected'] == base.manifest['performance_schedule']['selected']


def test_consumed_input_graph_is_not_mutated(ir):
    before = ir.to_dict()
    consume(ir, 'consumed')
    assert ir.to_dict() == before


def test_unknown_policy_and_non_flat_graphs_refused(ir):
    with pytest.raises(LoweringError):
        _lower(ir, gradient_outputs='dropped')

    class NotFlat:
        pass
    with pytest.raises(LoweringError):
        consume(NotFlat(), 'consumed')


def test_remat_activation_policy_never_recomputes_parameter_casts(ir):
    full = _lower(ir, schedule_options={'remat': True})
    act = _lower(ir, schedule_options={'remat': 'activations'})
    parameters = {name for name in ir.inputs if ir.values[name].role == 'parameter'}
    producers = {op.outputs[0]: op for op in ir.operations if op.outputs}
    chosen = act.manifest['late_operand_remat']['values']
    assert all(not set(producers[v].reads) & parameters for v in chosen)
    assert set(chosen) <= set(full.manifest['late_operand_remat']['values'])
    with pytest.raises(LoweringError):
        _lower(ir, schedule_options={'remat': 'everything'})


def test_compose_execution_accepts_consumed(ir):
    program = lower(ir, policy='auto', precision_policy='practical', execution='compose', gradient_outputs='consumed')
    assert 'grads' not in program.manifest['structured_outputs']
    assert program.manifest['gradient_outputs']['policy'] == 'consumed'
