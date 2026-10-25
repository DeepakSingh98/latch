"""CPU contracts for padded BF16 matrix operands (step 02; no CUDA). Run on a CPU VM, one thread per case."""
from __future__ import annotations

import math
import re

import pytest
import torch

from megakernel_lab.operation_compiler import SGDConfig, capture_module
from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.matrix_adapters import padding


def _mlp(inputs, width, outputs, batch, *, aux_elementwise=False, seed=5):
    generator = torch.Generator().manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.randn(inputs, width, generator=generator) / math.sqrt(inputs))
            self.b = torch.nn.Parameter(torch.randn(width, outputs, generator=generator) / math.sqrt(width))

        def forward(self, x):
            xb = x.to(torch.bfloat16)
            h = torch.nn.functional.silu((xb @ self.a.to(torch.bfloat16)).float())
            hb = h.to(torch.bfloat16)
            y = (hb @ self.b.to(torch.bfloat16)).float()
            if aux_elementwise:
                y = y + 0.1 * hb.float()[:, :outputs]          # an elementwise reader of the BF16 operand
            return y

    x = torch.randn(batch, inputs, generator=generator)
    t = torch.randn(batch, outputs, generator=generator)
    return capture_module(Net(), lambda p, t: (p - t).square().mean(), (x,), (t,),
                          optimizer=SGDConfig(lr=0.01, momentum=0.9, weight_decay=0.0))


def _lower(captured, enabled=True, elementwise=True):
    from megakernel_lab.operation_compiler import flat_elementwise
    saved = padding.ENABLED, flat_elementwise.ENABLED
    padding.ENABLED, flat_elementwise.ENABLED = enabled, elementwise
    try:
        return lower(captured.ir, target_arch='sm_90', execution='compose')
    finally:
        padding.ENABLED, flat_elementwise.ENABLED = saved


def test_helpers():
    assert padding.pitch(1001) == 1024 and padding.pitch(37) == 64
    assert padding.strides((5, 37), 40) == (40, 1)
    assert padding.strides((3, 5, 37), 40) == (200, 40, 1)
    assert padding.storage_elements((5, 37), (40, 1)) == 4 * 40 + 37
    text = padding.store_index((5, 37), 40, lambda i, d: f'({i}/{d})', lambda i, d: f'({i}%{d})')
    assert ';' not in text and text == '(i/37) * 40LL + (i%37)'


def test_odd_operands_are_padded_and_admitted():
    program = _lower(_mlp(37, 45, 21, 64))
    record = program.manifest['padded_matrix_operands']['values']
    assert record, 'odd BF16 matrix-only operands must be padded'
    layouts = program.manifest['layouts']
    for name, entry in record.items():
        layout = layouts[name]
        assert layout['strides'][-1] == 1 and layout['strides'][-2] == entry['pitch']
        if entry['kind'] != 'operand':
            continue
        assert layout['dtype'] == 'bfloat16'
        assert layout['strides'][-2] == entry['pitch'] and entry['pitch'] % 8 == 0
        assert entry['pitch'] >= layout['shape'][-1] and entry['readers']
        # Arena storage covers the padded span (words), never only the logical numel.
        assert program.manifest['arena_words'] * 4 >= padding.storage_elements(layout['shape'], layout['strides']) * 2
        # The producer stores through the padded index; no identity store of the value remains.
        assert re.search(rf'v_{re.escape(name)}\[[^\]]*\* {entry["pitch"]}LL', program.source)
        assert f'v_{name}[i] =' not in program.source
    # Every product reading only padded/aligned BF16 operands left the scalar SIMT fallback.
    engines = [p.get('engine') for p in program.manifest['phases'] if p['kind'] == 'gemm']
    assert engines and all(str(e).startswith('latch_matrix:') for e in engines), engines


def test_ablation_switch_restores_unpadded_lowering():
    captured = _mlp(37, 45, 21, 64)
    off = _lower(captured, enabled=False)
    assert 'padded_matrix_operands' not in off.manifest
    engines = [str(p.get('engine')) for p in off.manifest['phases'] if p['kind'] == 'gemm']
    assert engines and not all(e.startswith('latch_matrix:') for e in engines), engines


def test_aligned_program_is_byte_identical():
    captured = _mlp(64, 128, 32, 64)
    on, off = _lower(captured, True), _lower(captured, False)
    assert 'padded_matrix_operands' not in on.manifest
    assert on.source == off.source and on.cache_key == off.cache_key


def test_elementwise_reader_keeps_contiguous_layout():
    """Values read by operations outside pointwise/matrix/transpose kinds (here a slice view) keep their layout."""
    program = _lower(_mlp(37, 45, 21, 64, aux_elementwise=True))
    record = (program.manifest.get('padded_matrix_operands') or {}).get('values', {})
    layouts = program.manifest['layouts']
    sliced = [v['source'] for v in program.manifest['views'] if 'slice' in str(v.get('target'))]
    assert sliced
    for source in sliced:
        assert layouts[source]['root'] not in record, source
    assert [n for n, e in record.items() if e['shape'] == [64, 37]]


def test_outputs_and_state_are_never_padded():
    program = _lower(_mlp(37, 45, 21, 64))
    record = program.manifest['padded_matrix_operands']['values']
    assert not set(record) & set(program.manifest['outputs'])
    assert not set(record) & set(program.manifest['state_writes'].values() if isinstance(
        program.manifest['state_writes'], dict) else ())
    assert not set(record) & set(program.manifest['inputs'])


def test_vector_loops_cover_padded_stores_and_odd_tails():
    """Large odd program: padded operands are produced inside 16-byte vector loops; odd counts get a scalar tail."""
    program = _lower(_mlp(1001, 515, 77, 512))
    record = program.manifest['padded_matrix_operands']['values']
    assert record
    source = program.source
    assert 'reinterpret_cast<const uint4*>' in source or 'reinterpret_cast<const float4*>' in source
    big = [n for n, e in record.items() if math.prod(e['shape']) >= 4096 and e['kind'] == 'operand']
    assert big
    for name in big:
        pitch = record[name]['pitch']
        stores = [l.strip()[:300] for l in source.splitlines() if f'v_{name}' in l]
        assert f'padacc::scatter8<{record[name]["shape"][-1]}u, {pitch}u>(v_{name},' in source, (name, stores)
    assert 'padded_access.cuh' in program.manifest['header_hashes']
    # 1001 x 515 weight: odd element count -> vector loop over a multiple of 8, then a scalar tail.
    assert f'for (long long i = {1001 * 515 - (1001 * 515) % 8}LL + ' in source


def test_elementwise_ablation_is_previous_code():
    captured = _mlp(64, 128, 32, 64)
    base = _lower(captured, enabled=False, elementwise=False)
    assert 'vecbf' not in str(base.manifest.get('group_rules', ''))


def test_padded_problem_admission():
    from megakernel_lab.operation_compiler.matrix_adapters import ProgramResources, candidates, dense, select
    # A padded odd forward product: X [4096, 1001 -> pitch 1024] @ W [1001, 2001 -> pitch 2048].
    p = dense(4096, 2001, 1001, ak=True, bkc=False, a_pad=23, b_pad=47)
    assert select(p, ProgramResources(), 'donor')[0].body == 'pops4b_tma'
    # Unpadded odd rows stay refused by every TMA/cp.async body (16-byte stride contract).
    raw = dense(4096, 2001, 1001, ak=True, bkc=False)
    assert all(c.body not in ('pops4b_tma', 'pops4b_wg', 'p8ws', 'p5c', 'popsc_bf16_edge') for c in candidates(raw)[0])
    # Aligned products keep the unchanged p8ws body.
    assert select(dense(4096, 2048, 1024, ak=True, bkc=False), ProgramResources(), 'donor')[0].body == 'p8ws'


def test_product_results_padded_only_for_masked_tma():
    """Large products taken by the masked TMA body store their odd-width BF16 result at a padded pitch with 16-byte
    stores; consumers read it through grouped padded loads. Small products keep contiguous results."""
    program = _lower(_mlp(1001, 515, 77, 2048))
    record = program.manifest['padded_matrix_operands']['values']
    results = {n: e for n, e in record.items() if e['kind'] == 'product_result'}
    assert results, record
    source = program.source
    replans = [phase['tma_respec'] for phase in program.manifest['phases'] if phase.get('tma_respec')]
    specs = {item['spec']['out'].removeprefix('v_'): item['spec'] for item in replans}
    for name, entry in results.items():
        assert specs[name]['ldc'] == entry['pitch']  # joint step-14 planning must preserve the physical layout
        assert f'v_{name}, {entry["pitch"]}LL, ' in source                       # TArgs ldc = pitch
        decision = [d for d in program.manifest['matrix_decisions'] if d.get('output') == name][0]
        assert decision['selected'] == 'pops4b_tma' and decision['out_pitch'] == entry['pitch']
    assert ',true>(g, smem' in source                                             # 16-byte store form
    assert 'padacc::gather8<' in source                                           # grouped padded reads
    small = _lower(_mlp(37, 45, 21, 64)).manifest['padded_matrix_operands']['values']
    assert not [n for n, e in small.items() if e['kind'] == 'product_result']
