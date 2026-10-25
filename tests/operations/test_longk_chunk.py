"""CPU contracts for chunked long-K accumulation (LATCH-STEP03-REPAIR, ops/p5c_matrix.py); no CUDA."""
from __future__ import annotations

import pytest
import torch

from megakernel_lab.operation_compiler import AdamWConfig, capture_module, p5c_matrix
from megakernel_lab.operation_compiler.lower import lower


def _spec(m, n, k, **extra):
    return dict(engine='p5c', m=m, n=n, k=k, ak=True, bkc=True, a='a', b='b', out='o',
                strides=[k, 1, 1, k, 0, 0], roots=[('ra', 0), ('rb', 0)], **extra)


def test_chunking_is_a_shape_rule():
    assert p5c_matrix.chunk_stages(p5c_matrix.LONGK_CHUNK_MIN_K) == 0
    assert p5c_matrix.chunk_stages(p5c_matrix.LONGK_CHUNK_MIN_K + 64) == p5c_matrix.LONGK_CHUNK_STAGES
    assert p5c_matrix.chunk_stages(50304) == p5c_matrix.LONGK_CHUNK_STAGES
    saved = p5c_matrix.LONGK_CHUNK
    try:
        p5c_matrix.LONGK_CHUNK = False
        assert p5c_matrix.chunk_stages(50304) == 0
    finally:
        p5c_matrix.LONGK_CHUNK = saved


def test_vocabulary_input_gradient_plans_chunked_warp_specialized_item():
    dx = _spec(2048, 768, 50304, ws_ok=True)
    plan = p5c_matrix.plan(dx, 128)
    assert plan['ws_engine'] == 'p8ws_flat' and plan['chunk_stages'] == p5c_matrix.LONGK_CHUNK_STAGES
    code = p5c_matrix.code(dx, plan, lambda x: x, 0, 1, 1, 0)
    assert f'p8ws::item_k<128,true,true,0,true,{plan["chunk_stages"]}>' in code
    dw = _spec(768, 4096, 2048, ws_ok=True)      # an ordinary warp-specialized member (no vocabulary wide tile)
    group = p5c_matrix.group_code([dx, dw], [plan, p5c_matrix.plan(dw, 128)], [0, 96], lambda x: x)
    assert group.count('p8ws::item_k<') == 1 and group.count('p8ws::item<') == 1


def test_non_warp_specialized_long_k_uses_one_slab_tiles():
    plan = p5c_matrix.plan(_spec(2048, 1024, 50304), 128)
    assert (plan['bm'], plan['bn'], plan['wm']) == (128, 128, 2) and plan['chunk_stages'] and plan['mc'] == 1
    assert not plan['release']
    code = p5c_matrix.code(_spec(2048, 1024, 50304), plan, lambda x: x, 0, 1, 1, 0)
    assert 'p5c::phase_k<128,128,2,' in code


def test_wide_tiles_and_region_bodies_refused_for_long_k():
    large = _spec(16384, 768, 50304, ws_ok=True)
    plan = p5c_matrix.plan(large, 128)
    assert plan['bm'] * plan['bn'] <= 128 * 128 and plan['chunk_stages']
    saved = p5c_matrix.P9WS
    try:
        p5c_matrix.P9WS = True
        assert p5c_matrix.region_ws_plan(_spec(2048, 768, 50304), 128) is None
    finally:
        p5c_matrix.P9WS = saved


@pytest.mark.parametrize('shape', [(2048, 768, 768), (768, 50304, 2048), (2048, 3072, 768), (2048, 768, 8192)])
def test_established_products_emit_identical_code(shape):
    spec = _spec(*shape, ws_ok=True)
    plan = p5c_matrix.plan(spec, 128)
    assert 'chunk_stages' not in plan
    code = p5c_matrix.code(spec, plan, lambda x: x, 0, 1, 1, 0)
    saved = p5c_matrix.LONGK_CHUNK
    try:
        p5c_matrix.LONGK_CHUNK = False
        assert p5c_matrix.code(spec, p5c_matrix.plan(spec, 128), lambda x: x, 0, 1, 1, 0) == code
    finally:
        p5c_matrix.LONGK_CHUNK = saved


class _HeadInput(torch.nn.Module):
    def __init__(self, tokens, width, vocab):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.weight = torch.nn.Parameter(torch.randn(width, vocab, generator=g) / width ** 0.5)
        self.hidden = torch.nn.Parameter(torch.randn(tokens, width, generator=g))

    def forward(self, scale):
        return ((self.hidden * scale).to(torch.bfloat16) @ self.weight.to(torch.bfloat16)).float()


def test_normal_lowering_selects_chunked_input_gradient():
    """The normal API lowers the vocabulary head's input gradient (K = padded vocabulary) with chunked accumulation."""
    tokens, width, vocab = 1024, 1024, 9001      # 64 output tiles: admitted to the warpgroup engine
    model = _HeadInput(tokens, width, vocab)
    targets = torch.randint(0, vocab, (tokens,), generator=torch.Generator().manual_seed(6))
    captured = capture_module(model, torch.nn.functional.cross_entropy, (torch.ones(()),), (targets,),
                              optimizer=AdamWConfig(lr=1e-3))
    program = lower(captured.ir, target_arch='sm_90')
    members = [mm for p in program.manifest['phases'] if p.get('kind') == 'gemm' for mm in (p.get('members') or [p])]
    chunked = [mm for mm in members if (mm.get('matrix_plan') or {}).get('chunk_stages')]
    assert len(chunked) == 1 and chunked[0]['matrix_shape'] == [tokens, width, 9088]
    assert any(form in program.source for form in ('item_k<', 'phase_k<', 'phase_epi_k<'))
    saved = p5c_matrix.LONGK_CHUNK
    try:
        p5c_matrix.LONGK_CHUNK = False
        off = lower(captured.ir, target_arch='sm_90')
    finally:
        p5c_matrix.LONGK_CHUNK = saved
    assert not any(form in off.source for form in ('item_k<', 'phase_k<', 'phase_epi_k<'))
