"""Release-repair regressions: row storage, automatic selection and exact BF16 spellings."""
from copy import deepcopy

import pytest

from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.bf16_retention import exact_projection_capture, explicit_projection_contract
from .test_row_assembly import row_fixture, fused, layout
from .test_bf16_retention import graph as projection_graph
from .test_execution_selection import small_graph
from .test_ffn_retain import explicit_graph
from .test_row_assembly import reduction_graph


@pytest.mark.parametrize('width', [64, 128, 256, 512])
@pytest.mark.parametrize('pipe', [False, True])
def test_split_component_identity_read_stays_in_carried_register(width, pipe):
    phases, layouts, dtypes = row_fixture(width=width)
    shape = (128, width)
    layouts['derivative'] = layout('derivative', shape)
    layouts['gradient'] = layout('gradient', shape)
    dtypes.update(derivative='float32', gradient='float32')
    phases[0]['names'].append('derivative')
    phases[0]['writes'].append('derivative')
    phases[0]['targets'].append('aten.mul.Tensor')
    phases[0]['body'].append('const float r_derivative = v_x[i] * 2.0f; v_derivative[i] = r_derivative;')
    phases.append(dict(kind='elementwise', names=['gradient'], targets=['aten.mul.Tensor'],
        reads=['derivative'], writes=['gradient'], shape=list(shape), dtype='float32',
        body=['const float r_gradient = v_derivative[i] * 0.5f; v_gradient[i] = r_gradient;']))
    result, records, elided = fused((phases, layouts, dtypes), pipe=pipe)
    assert 'derivative' in elided
    code = result[0]['code']
    assert 'keep_derivative[w] = r_derivative;' in code
    assert 'const float r_derivative = keep_derivative[w];' in code
    assert 'r_derivative * 0.5f' in code
    assert 'v_derivative[' not in code
    assert all(not name.startswith('v_derivative') for name in result[0]['reads'])


def test_default_api_runs_the_compatible_selector_and_preserves_explicit_legacy():
    graph = small_graph()
    automatic, explicit = lower(graph), lower(graph, execution='legacy')
    assert automatic.manifest['execution_selection']['selected'] == 'legacy_static'
    assert automatic.source == explicit.source
    assert lower(graph, execution='auto').source == explicit.source
    assert 'execution_selection' not in explicit.manifest
    assert lower(graph, compose_selection='common').manifest['execution_selection']['selected'] == 'common_static'
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    with execution_scope('legacy'):
        assert lower(graph).manifest == explicit.manifest
    assert lower(graph).manifest['execution_selection']['selected'] == 'legacy_static'


def test_pipeline_does_not_preempt_a_proved_explicit_flat_protocol(monkeypatch):
    graph = explicit_graph()
    rows = reduction_graph()
    # Keep both computations independent and their inputs distinct.
    rows = {**rows, 'inputs': ['row_x'],
            'values': {('row_x' if k == 'x' else k): v for k, v in rows['values'].items()},
            'operations': deepcopy(rows['operations'])}
    rows['operations'][0]['args'] = [{'ref':'row_x'}, {'ref':'row_x'}]
    graph['values'].update(rows['values'])
    for key in ('operations', 'inputs', 'outputs'):
        graph[key].extend(rows[key])
    result = lower(graph)
    selection = result.manifest['execution_selection']
    assert selection['admitted_capabilities']['pipelined_rows'] > 0
    assert selection['selected'] == 'legacy_static'
    assert selection['retained_protocol']['matrix_bodies']
    from megakernel_lab.operation_compiler import compose
    monkeypatch.setattr(compose, 'verify_legacy_long_protocol', lambda *args: (None, 'injected protocol mismatch'))
    assert lower(graph).manifest['execution_selection']['selected'] == 'common_static'


@pytest.mark.parametrize('width', [64, 96])
def test_supported_explicit_projection_graph_uses_the_named_recipe_implementation(width):
    named = projection_graph(width)
    explicit = deepcopy(named)
    explicit['metadata'] = {}
    assert exact_projection_capture(explicit)
    assert explicit_projection_contract(explicit)
    a = lower(named, precision_policy='practical')
    b = lower(explicit, precision_policy='practical')
    assert a.manifest['execution_selection']['selected'] == b.manifest['execution_selection']['selected'] == 'legacy_static'
    assert a.source == b.source
    assert a.manifest['state_writes'] == b.manifest['state_writes']


@pytest.mark.parametrize('change', ['fp32_product', 'shared_bf16_gradient_add', 'unknown_policy',
                                  'backward_autocast', 'distributed_region'])
def test_unproved_bf16_spelling_never_claims_projection_equivalence(change):
    graph = projection_graph()
    graph['metadata'] = {}
    if change == 'fp32_product':
        graph['values']['m']['dtype'] = 'float32'
    elif change == 'shared_bf16_gradient_add':
        graph['values']['added'] = deepcopy(graph['values']['m'])
        graph['operations'].append(dict(name='added', target='aten.add.Tensor', args=[{'ref':'m'}, {'ref':'m'}]))
    elif change == 'unknown_policy':
        graph['metadata']['precision_policy'] = {'name': 'unproved'}
    elif change == 'backward_autocast':
        graph['metadata']['backward_autocast'] = {'dtype':'bfloat16'}
    else:
        graph['metadata']['distributed_event'] = 0
    assert not exact_projection_capture(graph)


@pytest.mark.parametrize('transpose', [False, True])
def test_explicit_and_named_projection_capture_match_update_semantics(transpose):
    import torch
    from megakernel_lab.operation_compiler import SGDConfig, capture_module

    class Projection(torch.nn.Module):
        def __init__(self, explicit):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64) * 1e-5)
            self.explicit = explicit

        def forward(self, x):
            weight = self.weight.t() if transpose else self.weight
            if self.explicit:
                # Each use owns its casts; changing that sharing before AD
                # would define a different gradient-accumulation recipe.
                a = (x.bfloat16() @ weight.bfloat16()).float()
                b = (x.bfloat16() @ weight.bfloat16()).float()
            else:
                a, b = x @ weight, x @ weight
            return torch.tanh(a) + 0.25 * b

    x = torch.linspace(-0.3, 0.3, 32 * 64).reshape(32, 64)
    target = torch.zeros_like(x)
    captures = [capture_module(Projection(explicit), lambda p, y: (p-y).square().mean(),
                              (x,), (target,), optimizer=SGDConfig(lr=0.01, momentum=0.9),
                              precision=None if explicit else 'bf16_projections')
                for explicit in (False, True)]
    reference = [c.step_fn(*c.example_args) for c in captures]
    assert len(reference[0]) == len(reference[1])
    assert all(torch.equal(a, b) for a, b in zip(*reference))
    assert exact_projection_capture(captures[1].ir)
    programs = [lower(c.ir, precision_policy='practical') for c in captures]
    assert all(p.manifest['execution_selection']['selected'] == 'legacy_static' for p in programs)
    body = lambda p: [m['engine'] for m in p.manifest['execution_selection']['retained_bf16']['matrix_bodies']]
    assert body(programs[0]) == body(programs[1])
