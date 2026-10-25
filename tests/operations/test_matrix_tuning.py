"""CPU contracts for step-16 matrix tuning (lowering only, no CUDA). Run on a CPU worker, one thread per case."""
from __future__ import annotations

import copy
import json

import pytest

from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.matrix_adapters.provider import matrix_choices
from megakernel_lab.operation_compiler import matrix_tuning as mt
from .test_matrix_adapters import _graph

AWKWARD = (1000, 1000, 1304, 248)     # LATCH-COMPOSE peer fixture shape (masked TMA products)
ALIGNED = (2048, 1024, 4096, 1024)    # LATCH-MATRIX development chain shape (P5C/P8WS products)


def _lower(graph, choices=None, log=None):
    if choices is None:
        return lower(copy.deepcopy(graph), execution='compose')
    with matrix_choices(choices, log=log):
        return lower(copy.deepcopy(graph), execution='compose')


def _engines(program):
    return {name: phase.get('engine') for phase in program.manifest['phases'] if phase.get('kind') == 'gemm'
            for name in phase.get('names', ())}


@pytest.mark.parametrize('shape', [AWKWARD, ALIGNED])
def test_empty_choice_scope_is_byte_identical_and_inventories_products(shape):
    graph = _graph(*shape)
    base = _lower(graph)
    log = []
    same = _lower(graph, {}, log)
    assert (same.source, same.manifest) == (base.source, base.manifest)
    assert log, 'every BF16 product is inventoried'
    for row in log:
        assert row['selected'] == row['default']
        assert set(row['admitted']) >= ({row['default']} - {'native'})
        assert all(body in row['classes'] for body in row['admitted'])


def test_choice_switches_one_product_to_an_admitted_body():
    graph = _graph(*AWKWARD)
    log = []
    _lower(graph, {}, log)
    row = next(r for r in log if r['default'] in mt.WGMMA and 'popsc_bf16_edge' in r['admitted'])
    chosen_log = []
    program = _lower(graph, {row['key']: 'popsc_bf16_edge'}, chosen_log)
    engines = _engines(program)
    assert engines[row['output']] == 'latch_matrix:popsc_bf16_edge'
    others = [r for r in chosen_log if r['key'] != row['key']]
    assert all(r['selected'] == r['default'] for r in others)


def test_native_choice_returns_product_to_the_existing_native_path():
    graph = _graph(*AWKWARD)
    log = []
    base = _lower(graph, {}, log)
    row = next(r for r in log if r['default'] != 'native')
    program = _lower(graph, {row['key']: 'native'}, [])
    assert not str(_engines(program).get(row['output'], '')).startswith('latch_matrix:')
    assert str(_engines(base)[row['output']]).startswith('latch_matrix:')


def test_p5c_products_are_visible_and_overridable():
    graph = _graph(*ALIGNED)
    log = []
    _lower(graph, {}, log)
    natives = [r for r in log if r['native'] == 'p5c']
    if not natives:
        pytest.skip('no P5C-native product in this lowering')
    row = natives[0]
    assert row['default'] == 'native'
    body = next(b for b in mt.MMA_SYNC if b in row['admitted'])
    program = _lower(graph, {row['key']: body}, [])
    assert _engines(program)[row['output']] == 'latch_matrix:' + body


def test_inadmissible_choice_is_rejected_and_keeps_existing_selection():
    graph = _graph(*AWKWARD)
    base = _lower(graph)
    log = []
    _lower(graph, {}, log)
    row = next(r for r in log if 'p8ws' not in r['admitted'])
    chosen_log = []
    program = _lower(graph, {row['key']: 'p8ws'}, chosen_log)
    rejected = [r for r in chosen_log if r.get('rejected')]
    assert rejected and rejected[0]['selected'] == rejected[0]['default']
    assert program.source == base.source


def _row(key, native, default, admitted, classes=None):
    classes = classes or {b: ('bf16_wgmma_k16' if b.startswith(('pops4b', 'p5c', 'p8ws')) else 'bf16_mma_sync_k16')
                          for b in admitted}
    return {'key': key, 'native': native, 'default': default, 'admitted': admitted, 'classes': classes}


def test_variants_are_bounded_deduplicated_and_respect_bitwise_mode():
    rows = [_row('a', None, 'pops4b_tma', ['pops4b_tma', 'pops4b_wg', 'popsc_bf16_edge']),
            _row('b', 'p5c', 'native', ['pops4b_tma', 'popsc_bf16', 'popsc_bf16_edge']),
            _row('b', 'p5c', 'native', ['pops4b_tma', 'popsc_bf16', 'popsc_bf16_edge']),
            _row('c', 'popsc', 'native', ['pops4b_tma', 'popsc_bf16'])]
    out = mt.variants(rows)
    labels = [label for label, _ in out]
    assert labels == ['existing', 'wgmma', 'mma_sync'] and len(out) <= 4
    assert out[0][1] == {}
    assert out[1][1] == {'c': 'pops4b_tma'}
    assert out[2][1] == {'a': 'popsc_bf16_edge', 'b': 'popsc_bf16'}
    assert mt.variants(rows, max_candidates=2) == out[:2]
    bitwise = mt.variants(rows, numerics='bitwise')
    assert bitwise == [('existing', {})]
    only_default = [_row('a', None, 'popsc_bf16_edge', ['popsc_bf16_edge'])]
    assert mt.variants(only_default) == [('existing', {})]


def test_cache_entry_identity_and_applicability(tmp_path):
    parts = {'version': mt.TUNING_VERSION, 'environment': {'gpu': 'H100'}, 'program': 'p'}
    key = mt._sha(parts)
    assert mt.load_entry(tmp_path, key, parts) == (None, 'miss')
    entry = {'schema': mt.TUNING_VERSION, 'key': key, 'parts': parts, 'verdict': 'tuned',
             'choices': {'a': 'popsc_bf16_edge'}}
    mt.store_entry(tmp_path, entry)
    loaded, state = mt.load_entry(tmp_path, key, parts)
    assert state == 'hit' and loaded['choices'] == entry['choices']
    other = dict(parts, environment={'gpu': 'A100'})
    assert mt.load_entry(tmp_path, key, other)[1].startswith('rejected: entry identity differs')
    products = [_row('a', None, 'pops4b_tma', ['pops4b_tma', 'popsc_bf16_edge'])]
    assert mt.validate_choices(loaded, products) is None
    assert mt.validate_choices(dict(loaded, choices={'a': 'p8ws'}), products).startswith('rejected')
    assert mt.validate_choices(dict(loaded, choices={'z': 'pops4b_tma'}), products).startswith('rejected')
    (tmp_path / f'{key}.json').write_text('{broken')
    assert mt.load_entry(tmp_path, key, parts)[1].startswith('rejected: unreadable')


def test_graph_signature_ignores_values_and_scalars_but_not_shapes():
    import torch
    from megakernel_lab.operation_compiler import SGDConfig, capture_module

    def capture(batch, lr, seed):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))
        x, y = torch.randn(batch, 8), torch.randn(batch, 8)
        return capture_module(model, lambda p, t: (p - t).square().mean(), (x,), (y,),
                              optimizer=SGDConfig(lr=lr, momentum=0.9))

    a = mt.graph_signature(capture(4, 0.01, 1).ir)
    assert a == mt.graph_signature(capture(4, 0.02, 2).ir)
    assert a != mt.graph_signature(capture(6, 0.01, 1).ir)


def test_tuning_bounds_are_validated():
    with pytest.raises(ValueError):
        mt.MatrixTuning(cache='c', max_candidates=5)
    with pytest.raises(ValueError):
        mt.MatrixTuning(cache='c', numerics='loose')
    assert json.loads(json.dumps(mt.MatrixTuning(cache='c').__dict__))['max_candidates'] == 4
