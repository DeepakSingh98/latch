"""CPU-worker lowering checks of the opt-in matrix_bodies rules (LATCH-TRAIN-MATMUL; no CUDA execution).

Rules must be chosen from structure alone (shapes, layouts, tile counts, contraction length), for several equivalent
graph spellings and untuned nearby shapes, and must leave lowering unchanged without the option.
"""
import json
import re

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import capture_training
from megakernel_lab.operation_compiler import matrix_bodies as mb
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.user_optimizer import tensor_update

SPELLINGS = {
    'linear': lambda x, w: F.linear(x, w),
    'matmul_t': lambda x, w: x @ w.t(),
    'transposed_product': lambda x, w: (w @ x.t()).t(),
}


def fixture(tokens, width, out, spelling='linear'):
    product = SPELLINGS[spelling]

    def objective(parameters, x):
        y = product(x.bfloat16(), parameters['weight'].bfloat16()).float()
        return (y * y).mean()
    rule = tensor_update(lambda p, g, s, hp: ({k: p[k] - .01 * g[k] for k in p}, {}), name='matrix_bodies_test_sgd')
    return capture_training(objective, {'weight': torch.zeros(out, width)}, None,
                            inputs=(torch.zeros(tokens, width),), config=rule)


def lowered(captured, monkeypatch, bodies):
    monkeypatch.delenv('LATCH_SCREEN_RULES', raising=False)
    with mb.scope(bodies):
        return lower(captured.ir, target_arch='sm_90', execution='compose', precision_policy='practical')


def test_option_validation_and_presets():
    assert mb.rules_of(None) == frozenset()
    assert mb.rules_of('exact') == frozenset(mb.RULES) - {'isolate'}
    assert mb.rules_of('tailrot+taildeep') == {'tailrot', 'taildeep'}
    for bad in ('bogus', 'tailrot+tailrot', '', 'exact+p5c'):
        with pytest.raises(ValueError):
            mb.validate(bad)


def test_tail_deepening_is_structural_and_keeps_tiles():
    plan = dict(bm=32, bn=32, wm=2, wn=4, ks=64, nst=4, tiles=12, shared_bytes=40960)
    elems = lambda bm, bn, ak, bkc, ks: (bm * (ks + 8) if ak else ks * (bm + 8)) + (bn * (ks + 8) if bkc else ks * (bn + 8))
    with mb.scope('taildeep'):
        deep = mb.deepen_tail(plan, 16, 384, 8192, False, False, stage_elements=elems, shared_limit=192 * 1024)
        assert (deep['bm'], deep['bn'], deep['tiles']) == (32, 32, 12)
        assert deep['ks'] * deep['nst'] > 64 * 4 and deep['shared_bytes'] <= 192 * 1024
        assert mb.deepen_tail(plan, 16, 384, 1024, False, False, stage_elements=elems,
                              shared_limit=192 * 1024) is plan       # short contraction: unchanged
    assert mb.deepen_tail(plan, 16, 384, 8192, False, False, stage_elements=elems, shared_limit=192 * 1024) is plan


def test_tail_rotation_token_only_under_the_rule():
    spec = {'tail': True, 'dtype': 'bfloat16'}
    assert mb.tail_rotation_token(spec) == 0
    with mb.scope('tailrot'):
        assert mb.tail_rotation_token(spec) == mb.TAIL_VBASE
        assert mb.tail_rotation_token(dict(spec, tail=False)) == 0
        assert mb.tail_rotation_token(dict(spec, active={'m': True})) == 0


def _phase(code, tiles, shape):
    return {'kind': 'gemm', 'code': code, 'tiles': tiles, 'matrix_shape': list(shape)}


TAIL = 'popsc::bmm_tail<32,32,2,4,false,false,128,8>(g, smem, 0, 1, 1, /*T_VBASE*/);'
WIDE = 'pops4b::TArgs g{a, b, c, 384LL, 32768, 384, KK}; pops4b::tma_phase_inline<128,192,2,true,false,NST,true,true>(g, smem, 0, 1, 1, /*B_VBASE*/);'
P5C = 'p5c::Args g{a, b, c, 384LL, 32768, 384, 384}; p5c::phase<128,128,2,true,false,6,0,false>(g, smem, 0, 1, 1, /*B_VBASE*/);'


def test_fill_plan_modes_and_refusals():
    tails = {i: _phase(TAIL, 12, (16, 384, 8192)) for i in range(4)}
    with mb.scope('fill'):
        phases = dict(tails, w=_phase(WIDE.replace('KK', '16').replace('NST', '2'), 512, (32768, 384, 16)))
        plan = mb.fill_plan(phases, [0, 1, 2, 3, 'w'], grid=128)
        assert plan['mode'] == 'slice' and plan['fixed_tiles'] == 48
        sliced = mb.fill_slice('x<1>(g, smem, 0, 1, 1, 0); pops4b::tma_phase_inline<128,192,2,true,false,2,true,true>'
                               '(g, smem, 0, 1, 1, 48);', 48)
        assert '(int)gridDim.x > 48 ? 48 : 0, (int)gridDim.x, (int)gridDim.x, 0)' in sliced
        phases['w'] = _phase(WIDE.replace('KK', '384').replace('NST', '4'), 512, (32768, 384, 384))
        assert mb.fill_plan(phases, [0, 1, 2, 3, 'w'], grid=128)['mode'] == 'claim'
        phases['w'] = _phase(P5C, 768, (32768, 384, 384))
        assert mb.fill_plan(phases, [0, 1, 2, 3, 'w'], grid=128)['mode'] == 'slice'
        assert mb.fill_plan(phases, ['w', 0, 1, 2, 3], grid=128) is None           # long tasks must run first
        assert mb.long_first(phases, ['w', 0, 1, 2, 3], grid=128) == [0, 1, 2, 3, 'w']
        short = {**phases, **{i: _phase(TAIL, 12, (16, 384, 1024)) for i in range(4)}}
        assert mb.fill_plan(short, [0, 1, 2, 3, 'w'], grid=128) is None             # not long tasks
        many = {**phases, **{i: _phase(TAIL, 40, (16, 384, 8192)) for i in range(4)}}
        assert mb.fill_plan(many, [0, 1, 2, 3, 'w'], grid=128) is None              # more than half the grid
    assert mb.fill_plan(dict(tails, w=_phase(P5C, 768, (32768, 384, 384))), [0, 1, 2, 3, 'w'], grid=128) is None


def bodies_of(program):
    return {(phase.get('names') or ['?'])[0]: phase.get('engine') for phase in program.manifest['phases']
            if phase['kind'] == 'gemm'}


@pytest.mark.parametrize('spelling', sorted(SPELLINGS))
@pytest.mark.parametrize('tokens,width,out', [(32768, 384, 384), (16384, 256, 512), (32768, 384, 192)])
def test_short_wide_products_take_the_p5c_body(monkeypatch, spelling, tokens, width, out):
    captured = fixture(tokens, width, out, spelling)
    with_rule = lowered(captured, monkeypatch, 'p5c')
    without = lowered(captured, monkeypatch, None)
    records = with_rule.manifest['group_rules'].get('matrix_bodies_p5c', [])
    assert records, 'no short wide product was re-emitted on the p5c body'
    for record in records:
        m, n, k = record['shape']
        assert record['to'][:2] in ([128, 128], [128, 192], [128, 64], [64, 128]) and k % 64 == 0
        assert (2 <= k // 64 < 8 and (m // 128) * (n // record['to'][1]) >= 128) or k // 64 >= mb.DEEP_MIN_STEPS
    assert 'matrix_bodies_p5c' not in without.manifest['group_rules']
    assert 'p5c::phase<' in with_rule.source
    assert any(engine == 'latch_matrix:p5c' for engine in bodies_of(with_rule).values())


def test_underfilled_short_products_keep_their_body(monkeypatch):
    program = lowered(fixture(2048, 384, 384), monkeypatch, 'p5c')        # 48 tiles: under one wave
    short = [r for r in program.manifest['group_rules'].get('matrix_bodies_p5c', []) if r['shape'][2] < 512]
    assert not short


def router_fixture(tokens, width, experts):
    def objective(parameters, x):
        logits = F.linear(x.bfloat16(), parameters['gate'].bfloat16()).float()
        return logits.softmax(-1).square().mean()
    rule = tensor_update(lambda p, g, s, hp: ({k: p[k] - .01 * g[k] for k in p}, {}), name='matrix_bodies_router_sgd')
    return capture_training(objective, {'gate': torch.zeros(experts, width)}, None,
                            inputs=(torch.zeros(tokens, width),), config=rule)


@pytest.mark.parametrize('tokens,width,experts', [(8192, 384, 16), (4096, 512, 16), (8192, 256, 32)])
def test_long_tail_products_deepen_and_rotate(monkeypatch, tokens, width, experts):
    captured = router_fixture(tokens, width, experts)
    with_rule = lowered(captured, monkeypatch, 'tailrot+taildeep')
    without = lowered(captured, monkeypatch, None)
    calls = re.findall(r'popsc::bmm_tail<(\d+),(\d+),\d+,\d+,\w+,\w+,(\d+),(\d+)>\(g, smem, 0, 1, 1, ([^)]*)\)',
                       with_rule.source)
    base_calls = re.findall(r'popsc::bmm_tail<(\d+),(\d+),\d+,\d+,\w+,\w+,(\d+),(\d+)>', without.source)
    if not base_calls:
        pytest.skip('no tail product in this spelling')
    assert calls and any(int(ks) * int(nst) > 256 for _, _, ks, nst, _ in calls)    # the long-K product deepens
    assert sorted(c[:2] for c in calls) == sorted(c[:2] for c in base_calls)          # same tiles
    assert '/*T_VBASE*/' not in with_rule.source and '/*T_VBASE*/' not in without.source
    assert all(rot.isdigit() for *_, rot in calls)


def readout_fixture(tokens, width, vocab):
    def objective(parameters, x):
        logits = F.linear(parameters['hidden'].bfloat16(), parameters['weight'].bfloat16()).float()
        return (logits * x[:, :1]).square().mean()
    rule = tensor_update(lambda p, g, s, hp: ({k: p[k] - .01 * g[k] for k in p}, {}), name='matrix_bodies_readout_sgd')
    return capture_training(objective, {'hidden': torch.zeros(tokens, width), 'weight': torch.zeros(vocab, width)},
                            None, inputs=(torch.zeros(tokens, 1),), config=rule)


@pytest.mark.parametrize('tokens,width,vocab', [(32768, 384, 32000), (16384, 512, 32768)])
def test_store_bound_short_k_products_take_wide_p5c_tiles(monkeypatch, tokens, width, vocab):
    captured = readout_fixture(tokens, width, vocab)
    with_rule = lowered(captured, monkeypatch, 'p5c')
    without = lowered(captured, monkeypatch, None)

    def engines(program):
        return [(phase.get('engine'), (phase.get('matrix_plan') or {}).get('bm'), (phase.get('matrix_plan') or {}).get('bn'))
                for phase in program.manifest['phases']
                if phase['kind'] == 'gemm' and list(phase.get('matrix_shape') or ()) == [tokens, vocab, width]]
    base, rule = engines(without), engines(with_rule)
    assert base and rule
    if not any(e == 'latch_matrix:p8ws' for e, _, _ in base):
        pytest.skip(f'the store-bound rule does not select p8ws for this product: {base}')
    assert all(e == 'latch_matrix:p5c' and (bm, bn) in ((256, 128), (128, 256)) for e, bm, bn in rule), rule
    assert 'popsc_bf16' not in ' '.join(str(e) for e, _, _ in rule)


def test_isolate_patterns_and_offset_view_split():
    from megakernel_lab.operation_compiler import step17_intervals as si
    assert mb.isolate_patterns({'group_rules': {}}) == ()
    patterns = mb.isolate_patterns({'group_rules': {'matrix_bodies': ['isolate', 'p5c']}})
    assert any(re.search(p, 'p5c::phase<128,128,2,true,false,6,0,false>(g') for p in patterns)
    assert not any(re.search(p, 'p5c::Args g{') for p in patterns)
    prelude = ('float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n'
               '__nv_bfloat16* v_a = reinterpret_cast<__nv_bfloat16*>(latch_pr + 0LL) + 131072LL;\n')
    block = ('{\n' + prelude + '{\n  p5c::phase<128,128,2,true,false,6,0,false>(g, smem, 0, 1, 1, 0);\n}\n'
             '__syncthreads();\n{\n  latch_ec_body1(v_a);\n}\ngrid.sync();\n}')
    assert si.split_interval('f', block, len(block), tuple(patterns[:-1])) is None     # offset view refused
    split = si.split_interval('f', block, len(block), tuple(patterns))
    assert split is not None and len(split[0]) == 2
    assert 'latch_ec_body1' not in split[0][0][1] and 'p5c::phase<' in split[0][0][1]


def test_partition_coarsening_keeps_long_tasks_within_the_grid():
    def phase(m, n, k):
        spec = {'m': m, 'n': n, 'k': k}
        return {'kind': 'gemm', 'tma_respec': {'spec': spec, 'plan': {}}}
    phases = [phase(192, 384, 8192)] * 8 + [phase(384, 384, 8192)] * 4 + [phase(32768, 384, 384)]
    members = list(range(len(phases)))
    coarse = mb.coarse_partitions(phases, [members], grid=132)        # 72 + 72 = 144 tiles > 132
    assert coarse and coarse <= set(range(8, 12))
    assert mb.coarse_partitions(phases, [list(range(8, 12)) + [12]], grid=132) == set()   # 72 tiles fit
