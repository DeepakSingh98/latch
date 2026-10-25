"""CPU contracts for step-13 structural groups, candidate sets and the bounded mixed search (no CUDA)."""
from __future__ import annotations

import copy
import json

import pytest

from megakernel_lab.operation_compiler import matrix_selection as ms
from megakernel_lab.operation_compiler import matrix_tuning as mt
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.matrix_adapters.provider import current_choices, matrix_choices
from .test_matrix_adapters import _graph

AWKWARD = (1000, 1000, 1304, 248)
ALIGNED = (2048, 1024, 4096, 1024)


def _key(m, n, k, ta=False, tb=False, batch=1):
    a = [1, m, 0, 0] if ta else [k, 1, 0, 0]
    b = [1, k, 0, 0] if tb else [n, 1, 0, 0]
    return json.dumps([m, n, k, batch, a, b, 'bfloat16', 'bfloat16', 'store'], separators=(',', ':'))


def _row(key, native, default, admitted, flops, output='mm', batch=1):
    return {'key': key, 'native': native, 'default': default, 'admitted': admitted, 'flops': flops,
            'batch': batch, 'output': output}


ALL = ['p8ws', 'p5c', 'pops4b_tma', 'pops4b_wg', 'popsc_bf16', 'popsc_bf16_tail', 'popsc_bf16_edge', 'mixed_generic']
AWK = ['pops4b_tma', 'pops4b_wg', 'popsc_bf16_edge', 'mixed_generic']


def _rows():
    return [_row(_key(1536, 768, 768), 'p5c', 'native', ALL, 1.8e9, 'mm_1'),
            _row(_key(1536, 768, 768), 'p5c', 'native', ALL, 1.8e9, 'mm_3'),
            _row(_key(768, 768, 1536, ta=True), 'popsc', 'native', ALL, 1.8e9, 'mm_8'),
            _row(_key(1536, 192, 768), 'popsc', 'native', ALL[2:], 4.5e8, 'mm_2'),
            _row(_key(1200, 1480, 840), 'fallback', 'pops4b_tma', AWK, 3.0e9, 'mm_9'),
            _row(_key(1200, 200, 1480), 'fallback', 'popsc_bf16_edge', AWK, 7.1e8, 'mm_10'),
            _row(_key(40, 40, 896, batch=24), 'fallback', 'native', [], 6.9e7, 'bmm', batch=24)]


def test_family_bodies_follow_native_engine_and_admission():
    p5c, popsc, small, awk, awk_small, bmm = [r for i, r in enumerate(_rows()) if i != 1]
    assert ms.family_body(p5c, 'P') == 'native' and ms.family_body(p5c, 'S') == 'popsc_bf16'
    assert ms.family_body(popsc, 'S') == 'native' and ms.family_body(popsc, 'P') == 'p8ws'
    assert ms.family_body(small, 'P') is None and ms.family_body(small, 'W') == 'pops4b_tma'
    assert ms.family_body(awk, 'S') == 'popsc_bf16_edge' and ms.family_body(awk, 'P') is None
    assert all(ms.family_body(bmm, f) in (None, 'native') for f in ms.FAMILIES)
    assert ms.family_of(p5c, 'native') == 'P' and ms.family_of(popsc, 'native') == 'S'
    assert ms.family_of(awk, 'pops4b_wg') == 'W'


def test_groups_are_structural_and_skip_products_without_alternatives():
    gs = ms.groups(_rows())
    assert all('bmm' not in g['outputs'] for g in gs.values())
    p5c = gs['p5c:NN:b' + str(ms.bucket(1.8e9))]
    assert p5c['flops'] == pytest.approx(3.6e9) and p5c['outputs'] == ['mm_1']   # repeated key: one choice
    assert list(gs) == sorted(gs, key=lambda g: -gs[g]['flops'])
    assert {g.split(':')[1] for g in gs} == {'NN', 'TN'}


def test_assignment_resolves_families_per_product():
    rows = _rows()
    assign = {g: 'W' for g in ms.groups(rows)}
    choices = ms.assignment_choices(rows, assign)
    keys = {r['output']: r['key'] for r in rows}
    assert choices[keys['mm_1']] == 'pops4b_tma' and choices[keys['mm_10']] == 'pops4b_tma'
    assert keys['mm_9'] not in choices            # already the default
    assert ms.assignment_choices(rows, {}) == {}


def test_candidate_set_is_bounded_distinct_and_starts_with_the_default():
    rows = _rows()
    items, truncated = ms.candidate_set(rows, cap=64)
    assert items[0][0] == 'uniform:D' and items[0][2] == {}
    identities = [tuple(sorted(c.items())) for _, _, c in items]
    assert len(identities) == len(set(identities)) <= 64
    small, cut = ms.candidate_set(rows, cap=5)
    assert len(small) == 5 and cut == len(items) - 5 + truncated
    assert truncated == 0
    assert [l for l, _, _ in small] == [l for l, _, _ in items[:5]]


def test_mixed_search_rounds_are_bounded_and_judged_by_supplied_times():
    rows = _rows()
    search = ms.MixedSearch(rows, top_groups=2, alternatives=2, max_builds=7)
    first = search.next_round({})
    assert first[0] == ('existing', {}) and 2 <= len(first) <= 3
    assert not any(label == 'uniform:S' for label, _ in first)
    times = {label: 1.0 for label, _ in first}
    times['uniform:W'] = 0.9
    second = search.next_round(times)
    assert second and all(label.startswith('uniform:W|') for label, _ in second)
    assert len(first) + len(second) <= 7
    for label, _ in second:
        times[label] = 0.95
    by_group = {}
    for label, _ in second:
        by_group.setdefault(label.split('|')[1].rsplit('=', 1)[0], label)
    improving = list(by_group.values())[:2]
    assert len(improving) == 2
    for label in improving:
        times[label] = 0.85
    third = search.next_round(times)
    assert third and third[0][0] == 'uniform:W|combined'
    merged = search.variants['uniform:W|combined'][0]
    assert all(merged[label.split('|')[1].rsplit('=', 1)[0]] == label.rsplit('=', 1)[1] for label in improving)
    assert len(search.variants) <= 7
    assert search.next_round(times) == []


def test_mixed_search_without_improvement_builds_no_combination():
    rows = _rows()
    search = ms.MixedSearch(rows)
    first = search.next_round({})
    times = {label: 1.0 for label, _ in first}
    second = search.next_round(times)
    times.update({label: 1.2 for label, _ in second})
    assert search.next_round(times) == []


def test_policy_reaches_the_provider_scope_and_empty_scopes_stay_identical():
    with matrix_choices({}, policy='donor'):
        assert current_choices()['policy'] == 'donor'
    with matrix_choices({}):
        assert 'policy' not in current_choices()
    assert current_choices() == {}
    graph = _graph(*AWKWARD)
    base = lower(copy.deepcopy(graph), execution='compose')
    log = []
    with matrix_choices({}, log=log):
        same = lower(copy.deepcopy(graph), execution='compose')
    assert (same.source, same.manifest) == (base.source, base.manifest)
    assert len({r['output'] for r in log}) == len(log), 'the re-lowering does not duplicate inventory rows'


def test_step13_common_rule_moves_non_p5c_mma_sync_products_only_on_common_programs():
    graph = _graph(*AWKWARD)
    base = lower(copy.deepcopy(graph), execution='compose')
    frozen_log, log = [], []
    with matrix_choices({}, policy='donor', log=frozen_log):
        frozen = lower(copy.deepcopy(graph), execution='compose')
    with matrix_choices({}, log=log):
        lower(copy.deepcopy(graph), execution='compose')
    sel = base.manifest['execution_selection']
    assert sel['selected'] == 'common_static'
    assert 'step13_default_rule' not in frozen.manifest['execution_selection']
    moved = {r['output']: r['to'] for r in sel['step13_default_rule']['changed_products']}
    assert moved and set(moved.values()) <= {'pops4b_tma', 'pops4b_wg'}
    frozen_default = {r['output']: r['default'] for r in frozen_log}
    default = {r['output']: r['default'] for r in log}
    for name, body in frozen_default.items():
        if name in moved:
            assert body in ('popsc_bf16_edge', 'native') and default[name] == moved[name]
        else:
            assert default[name] == body
    assert all(r['native'] != 'p5c' for r in frozen_log if r['output'] in moved)


def test_explicit_choices_win_over_the_step13_rule():
    graph = _graph(*AWKWARD)
    frozen_log = []
    with matrix_choices({}, policy='donor', log=frozen_log):
        lower(copy.deepcopy(graph), execution='compose')
    row = next(r for r in frozen_log if r['default'] == 'popsc_bf16_edge')
    log = []
    with matrix_choices({row['key']: 'popsc_bf16_edge'}, log=log):
        program = lower(copy.deepcopy(graph), execution='compose')
    chosen = next(r for r in log if r['key'] == row['key'])
    assert chosen['selected'] == 'popsc_bf16_edge'
    engines = {n: p.get('engine') for p in program.manifest['phases'] if p.get('kind') == 'gemm'
               for n in p.get('names', ())}
    assert engines[row['output']] == 'latch_matrix:popsc_bf16_edge'


@pytest.mark.parametrize('shape', [AWKWARD, ALIGNED])
def test_real_inventory_groups_and_uniform_choices_lower(shape):
    graph = _graph(*shape)
    log = []
    with matrix_choices({}, log=log):
        lower(copy.deepcopy(graph), execution='compose')
    gs = ms.groups(log)
    assert gs
    items, _ = ms.candidate_set(log, cap=12)
    for label, _, choices in items[:3]:
        chosen = []
        with matrix_choices(choices, log=chosen):
            lower(copy.deepcopy(graph), execution='compose')
        assert not [r for r in chosen if r.get('rejected')], label
        assert all(r['selected'] == choices.get(r['key'], r['default']) for r in chosen)


def test_tuning_strategy_bounds_are_validated():
    assert mt.MatrixTuning(cache='c').strategy == 'mixed'
    with pytest.raises(ValueError):
        mt.MatrixTuning(cache='c', strategy='exhaustive')
    with pytest.raises(ValueError):
        mt.MatrixTuning(cache='c', max_builds=40)
    with pytest.raises(ValueError):
        mt.MatrixTuning(cache='c', baseline_policy='fastest')


def test_mixed_search_stops_when_every_uniform_alternative_is_far_slower():
    search = ms.MixedSearch(_rows())
    first = search.next_round({})
    times = {label: 1.5 for label, _ in first}
    times['existing'] = 1.0
    assert search.next_round(times) == [] and search.stopped
    assert search.next_round(times) == []
