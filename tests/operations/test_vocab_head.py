"""CPU contracts for the large-vocabulary head rewrite (ops/vocab_head.py); no CUDA. Run remotely, one thread."""
from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from megakernel_lab.operation_compiler import AdamWConfig, capture_module
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler import vocab_head


class _Head(torch.nn.Module):
    def __init__(self, width, vocab, *, transposed=False, cast=True):
        super().__init__()
        g = torch.Generator().manual_seed(11)
        shape = (vocab, width) if transposed else (width, vocab)
        self.weight = torch.nn.Parameter(torch.randn(shape, generator=g) / math.sqrt(width))
        self.transposed, self.cast = transposed, cast

    def forward(self, hidden):
        w = self.weight.t() if self.transposed else self.weight
        if self.cast:
            return (hidden.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()
        return hidden @ w


def _capture(tokens=24, width=32, vocab=301, *, form='ce', transposed=False, cast=True, extra_reader=False):
    g = torch.Generator().manual_seed(5)
    hidden = torch.randn(tokens, width, generator=g)
    targets = torch.randint(0, vocab, (tokens,), generator=g)

    def ce(logits, t):
        loss = F.cross_entropy(logits, t)
        if extra_reader:
            loss = loss + 0.0 * logits.logsumexp(-1).mean()
        return loss

    def gather(logits, t):
        return -torch.log_softmax(logits, -1).gather(-1, t.unsqueeze(-1)).mean()

    model = _Head(width, vocab, transposed=transposed, cast=cast)
    return capture_module(model, ce if form == 'ce' else gather, (hidden,), (targets,),
                          optimizer=AdamWConfig(lr=1e-3, weight_decay=0.1))


def _targets(ir):
    data = ir.to_dict() if hasattr(ir, 'to_dict') else ir
    return [op['target'] for op in data['operations']]


@pytest.mark.parametrize('form', ['ce', 'gather'])
@pytest.mark.parametrize('transposed', [False, True])
def test_rewrite_replaces_softmax_chain_and_pads_products(form, transposed):
    captured = _capture(form=form, transposed=transposed)
    new, report = vocab_head.rewrite(captured.ir, 'force')
    assert report and len(report['rewritten']) == 1 and not report['refused']
    row = report['rewritten'][0]
    assert row['vocabulary'] == 301 and row['padded'] == 384 and row['backward']
    targets = _targets(new)
    for gone in ('aten._log_softmax.default', 'aten._log_softmax_backward_data.default', 'aten.gather.default'):
        assert gone not in targets
    for present in vocab_head.TARGETS.values():
        assert present in targets
    data = new.to_dict()
    # Every product touching the vocabulary dimension is padded; original-width values are views.
    for op in data['operations']:
        if op['target'] == 'aten.mm.default':
            shapes = [tuple(data['values'][a['ref']]['shape']) for a in op['args']] + [tuple(data['values'][op['name']]['shape'])]
            assert all(301 not in s for s in shapes), (op['name'], shapes)
    slices = [op for op in data['operations'] if op['target'] == 'aten.slice.Tensor']
    assert slices and all(op['args'][3] == 301 for op in slices)


def test_threshold_and_off_keep_the_program():
    captured = _capture(vocab=301)
    same, report = vocab_head.rewrite(captured.ir, 'auto')
    assert same is captured.ir and report is None
    off, report = vocab_head.rewrite(captured.ir, 'off')
    assert off is captured.ir and report is None
    a = lower(captured.ir)
    b = lower(captured.ir, vocabulary_head='off')
    assert a.source == b.source and 'vocabulary_head' not in a.manifest


def test_other_logit_readers_are_refused_unchanged():
    captured = _capture(extra_reader=True)
    same, report = vocab_head.rewrite(captured.ir, 'force')
    assert same is captured.ir
    assert report['refused'] and not report['rewritten']


@pytest.mark.parametrize('execution', [None, 'compose'])
def test_lowering_emits_vocabulary_phases_without_generic_matrix_tiles(execution):
    captured = _capture(tokens=64, width=64, vocab=1013)
    program = lower(captured.ir, execution=execution, vocabulary_head='force')
    phases = program.manifest['phases']
    kinds = {p['kind'] for p in phases}
    assert {'vocab_rows', 'vocab_gradient', 'vocab_pick', 'elementwise_copy'} <= kinds
    assert not any(p.get('engine') == 'generic_bf16_simt_16' for p in phases if p['kind'] == 'gemm')
    assert program.manifest['vocabulary_head']['rewritten'][0]['padded'] == 1024
    assert 'latch_vocab_unpack' in program.source


def test_fp32_products_pad_by_copy():
    captured = _capture(cast=False)
    new, report = vocab_head.rewrite(captured.ir, 'force')
    assert report['rewritten'][0]['dtype'] == 'float32'
    data = new.to_dict()
    pads = [op for op in data['operations'] if op['target'] == vocab_head.TARGETS['pad']]
    assert len(pads) == 1 and pads[0]['kwargs']['dtype'] == 'float32'
    lower(new)


def test_wide_tiles_only_at_vocabulary_extents():
    from megakernel_lab.operation_compiler import p5c_matrix
    small = {'m': 2048, 'n': 4096, 'k': 768, 'ak': True, 'bkc': True, 'ws_ok': True}
    large = {'m': 2048, 'n': 50304, 'k': 768, 'ak': True, 'bkc': True, 'ws_ok': True}
    assert vocab_head.prefer_wide_tile(small, p5c_matrix.flat_ws_plan(small, 132), 132) is None
    ws = p5c_matrix.flat_ws_plan(large, 132)
    assert ws['traffic_us'] > ws['local_us']          # the warp-specialized 128 x 128 plan is L2-stream bound
    plan = vocab_head.prefer_wide_tile(large, ws, 132)
    assert p5c_matrix.plan(dict(large, engine='p5c'), 132)['bm'] * p5c_matrix.plan(dict(large, engine='p5c'), 132)['bn'] == 2 * 128 * 128
    assert plan is not None and plan['bm'] * plan['bn'] == 2 * 128 * 128 and plan['vocabulary_wide_tile']
    saved = vocab_head.WIDE_TILES
    try:
        vocab_head.WIDE_TILES = False
        assert vocab_head.prefer_wide_tile(large, ws, 132) is None
    finally:
        vocab_head.WIDE_TILES = saved
    assert p5c_matrix.P8WS


def test_rewrite_does_not_mutate_input():
    captured = _capture()
    before = copy.deepcopy(captured.ir.to_dict())
    vocab_head.rewrite(captured.ir, 'force')
    assert captured.ir.to_dict() == before
