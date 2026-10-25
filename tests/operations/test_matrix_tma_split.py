"""CPU-worker lowering checks of the split-K rule for underfilled long-K TMA products (no CUDA execution).

Nearby shapes and equivalent graph spellings of the same weight-gradient product must select the rule from structure
alone; short contractions, strict precision and the rule's absence keep the unsplit product.
"""
import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import capture_training
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler import matrix_schedule
from megakernel_lab.operation_compiler import matrix_tma_split as split
from megakernel_lab.operation_compiler.user_optimizer import tensor_update

SPELLINGS = {
    'linear': lambda x, w: F.linear(x, w),
    'matmul_t': lambda x, w: x @ w.t(),
    'transpose_matmul': lambda x, w: torch.matmul(x, w.transpose(0, 1)),
    'batched_view': lambda x, w: F.linear(x.view(2, x.shape[0] // 2, x.shape[1]), w).reshape(x.shape[0], -1),
    'transposed_product': lambda x, w: (w @ x.t()).t(),
}


def fixture(tokens, width, out, spelling):
    product = SPELLINGS[spelling]

    def objective(parameters, x):
        y = product(x.bfloat16(), parameters['weight'].bfloat16()).float()
        return (y * y).mean()
    rule = tensor_update(lambda p, g, s, hp: ({k: p[k] - .01 * g[k] for k in p}, {}), name='tma_split_test_sgd')
    return capture_training(objective, {'weight': torch.zeros(out, width)}, None,
                            inputs=(torch.zeros(tokens, width),), config=rule)


def lowered(captured, monkeypatch, *, schedule='underfilled', precision='practical'):
    monkeypatch.delenv('LATCH_SCREEN_RULES', raising=False)
    with matrix_schedule.scope(schedule):
        return lower(captured.ir, target_arch='sm_90', execution='compose', precision_policy=precision)


def split_records(program):
    return [phase[split.KEY] for phase in program.manifest['phases'] if split.KEY in phase]


def test_choice_is_structural():
    spec = dict(m=384, n=384, k=2048, batch=1, pair=True, tma={'a': ('a', 0), 'b': ('b', 0)})
    assert split.choose(spec, {'tiles': 18}, 128) == (8 if split.MAX_TASK_GRIDS > 1 else 4)
    assert split.choose(dict(spec, m=192), {'tiles': 9}, 128) == 8
    assert split.choose(dict(spec, k=384), {'tiles': 18}, 128) is None          # short contraction
    assert split.choose(spec, {'tiles': 96}, 128) is None                       # grid already filled
    assert split.choose(dict(spec, n=388), {'tiles': 18}, 128) is None          # vector reduce needs N % 8
    assert split.choose(dict(spec, ldc=392), {'tiles': 18}, 128) is None        # padded result pitch
    assert split.choose(dict(spec, serial_bf16_k=64), {'tiles': 18}, 128) is None


@pytest.mark.parametrize('spelling', sorted(SPELLINGS))
@pytest.mark.parametrize('tokens,width,out', [(2048, 384, 384), (2048, 384, 192), (1536, 320, 320), (4096, 256, 192)])
def test_weight_gradient_products_split_for_every_spelling(monkeypatch, spelling, tokens, width, out):
    program = lowered(fixture(tokens, width, out, spelling), monkeypatch)
    records = split_records(program)
    assert records, 'the long-K weight-gradient product was not split'
    for record in records:
        assert record['shape'][2] == tokens
        assert record['slice_k'] * record['splits'] == tokens and record['slice_k'] >= split.MIN_SLICE
        assert record['tiles'] * record['splits'] <= split.MAX_TASK_GRIDS * 128
    sums = [phase for phase in program.manifest['phases'] if phase.get('tma_split_sum')]
    assert len(sums) == len(records)
    assert all(phase['kind'] == ('elementwise' if split.ELEMENTWISE_SUM else 'splitk_reduce') for phase in sums)


def test_short_contraction_strict_precision_and_absent_rule_keep_unsplit_products(monkeypatch):
    assert not split_records(lowered(fixture(256, 384, 384, 'linear'), monkeypatch))
    assert not split_records(lowered(fixture(2048, 384, 384, 'linear'), monkeypatch, schedule=None))
    assert not split_records(lowered(fixture(2048, 384, 384, 'linear'), monkeypatch, precision='strict'))


def test_schedule_option_is_validated():
    with pytest.raises(ValueError, match='matrix_schedule'):
        matrix_schedule.validate('fast')


def readout_fixture(tokens, width, vocab):
    def objective(parameters, x):
        logits = F.linear(parameters['hidden'].bfloat16(), parameters['weight'].bfloat16()).float()
        return (logits * x[:, :1]).square().mean()
    rule = tensor_update(lambda p, g, s, hp: ({k: p[k] - .01 * g[k] for k in p}, {}), name='tma_split_long_test_sgd')
    return capture_training(objective, {'hidden': torch.zeros(tokens, width), 'weight': torch.zeros(vocab, width)},
                            None, inputs=(torch.zeros(tokens, 1),), config=rule)


@pytest.mark.parametrize('tokens,width,vocab', [(2048, 384, 32000), (1024, 256, 16384), (2048, 512, 50304)])
def test_vocabulary_scale_input_gradient_splits_into_short_chains(monkeypatch, tokens, width, vocab):
    program = lowered(readout_fixture(tokens, width, vocab), monkeypatch)
    long = [record for record in split_records(program) if record.get('long_contraction')]
    for record in long:
        assert record['shape'][2] >= split.LONG_K
        assert record['slice_k'] * record['splits'] == record['shape'][2]
        assert record['slice_k'] <= 64 * split.LONG_SLICE_STEPS
    native = [d for d in program.manifest.get('matrix_decisions', []) if d.get('replaced_native') == 'p5c']
    assert len(native) == len(long)
    if (tokens, width, vocab) == (2048, 384, 32000):
        assert long and long[0]['splits'] == 20     # 500 k steps -> 20 slices of 25 (1600 <= 2048)


@pytest.mark.parametrize('spelling', ['linear', 'matmul_t', 'batched_view'])
@pytest.mark.parametrize('tokens,width,out', [(2048, 384, 384), (2048, 384, 192), (1536, 320, 320)])
def test_short_single_wave_contractions_take_the_cp_async_wgmma_body(monkeypatch, spelling, tokens, width, out):
    def bodies(schedule):
        program = lowered(fixture(tokens, width, out, spelling), monkeypatch, schedule=schedule)
        return {d['output']: d.get('selected') for d in program.manifest.get('latch_matrix', {}).get('decisions', [])
                if d.get('problem_key') and json_k(d['problem_key']) <= split.SHORT_K}
    with_rule, without = bodies('underfilled'), bodies(None)
    assert with_rule and set(with_rule.values()) <= {'pops4b_wg', None}
    assert 'pops4b_wg' in with_rule.values()
    assert 'pops4b_wg' not in without.values()


def json_k(key):
    import json
    return json.loads(key)[2]
