"""Masked-edge primitive contracts in the isolated P7-B matrix protocol.

The original assertions are retained. The integrated public auto selector can
choose another protocol, so this diagnostic intentionally selects the owner of
these historical mma.sync/WGMMA bodies. Public auto routing has separate tests.
"""
import pytest

from megakernel_lab.operation_compiler import matrix_rules_p7b as R
from megakernel_lab.operation_compiler.fusion_p7b import lower_fused


def lower(graph):
    return lower_fused(graph, matrix_policy="tiles")


def _graph(m, n, k, *, at=False, bt=False):
    """One BF16 product C = A B with A (m, k) and B (k, n), optionally given as transposed views of their roots."""
    values, ops = {}, []

    def value(name, shape, role, strides=None):
        values[name] = {'name': name, 'shape': list(shape), 'dtype': 'bfloat16', 'role': role,
                        'strides': strides or [shape[1], 1]}
    value('a', (k, m) if at else (m, k), 'input')
    value('b', (n, k) if bt else (k, n), 'input')
    a, b = 'a', 'b'
    if at:
        value('aT', (m, k), 'intermediate', [1, m])
        ops.append({'name': 'aT', 'target': 'aten.transpose.int', 'args': [{'ref': 'a'}, 0, 1], 'kwargs': {}})
        a = 'aT'
    if bt:
        value('bT', (k, n), 'intermediate', [1, k])
        ops.append({'name': 'bT', 'target': 'aten.transpose.int', 'args': [{'ref': 'b'}, 0, 1], 'kwargs': {}})
        b = 'bT'
    value('c', (m, n), 'intermediate')
    ops.append({'name': 'c', 'target': 'aten.mm.default', 'args': [{'ref': a}, {'ref': b}], 'kwargs': {}})
    return {'values': values, 'operations': ops, 'inputs': ['a', 'b'], 'outputs': ['c']}


def _engines(program):
    out = []
    for phase in program.manifest['phases']:
        if phase.get('kind') == 'gemm':
            out += [m['engine'] for m in phase.get('members', [])] or [phase.get('engine')]
    return out


FLAGS = ('EDGE_TAILS', 'WGMMA', 'WGMMA_ALIGNED', 'WG_INLINE', 'WG_TMA', 'WG_VEC_STORE', 'LARGE_TILES')


@pytest.fixture(autouse=True)
def _default_rule():
    saved = {f: getattr(R, f) for f in FLAGS}
    yield
    for f, v in saved.items():
        setattr(R, f, v)


@pytest.fixture
def mma_edges():
    """The mma.sync masked-edge arm (warpgroup bodies off)."""
    R.WGMMA = False


@pytest.mark.parametrize('m,n,k,at,bt', [
    (16384, 3472, 576, False, False),       # projection forward: N % 32 = 16
    (16384, 576, 3472, False, True),        # input gradient: K % 32 = 16
    (576, 3472, 1024, True, False),         # weight gradient: N tail, m-contiguous A
    (100, 72, 40, False, False),            # every extent a tail
    (96, 577, 64, False, True),             # odd N (k-contiguous B): scalar stores
    (37, 64, 128, True, False),             # odd M with m-contiguous A? stride 37 is misaligned -> refused
])
def test_edge_admission(m, n, k, at, bt, mma_edges):
    program = lower(_graph(m, n, k, at=at, bt=bt))
    engines = _engines(program)
    if at and m % 8:
        assert engines == ['generic_bf16_simt_16']       # layout contract (stride alignment) unchanged
        return
    assert len(engines) == 1 and engines[0].startswith('popsc_bf16_edge_'), engines
    assert 'popsc::bmm_edge_phase<' in program.source
    pair = n % 2 == 0
    assert (',true>(g' if pair else ',false>(g') in program.source.split('bmm_edge_phase<')[1].split('\n')[0]


@pytest.mark.parametrize('m,n,k,at,bt', [(256, 128, 64, False, False), (16384, 3456, 576, False, False),
                                         (16384, 576, 3456, False, True), (576, 3456, 1024, True, False)])
def test_aligned_shapes_emit_unchanged_source(m, n, k, at, bt):
    on = lower(_graph(m, n, k, at=at, bt=bt))
    R.EDGE_TAILS = False
    off = lower(_graph(m, n, k, at=at, bt=bt))
    assert on.source == off.source and on.cache_key == off.cache_key
    assert 'bmm_edge_phase<' not in on.source


def test_selected_default_uses_tma_body_for_large_tails():
    assert R.EDGE_TAILS and R.WGMMA and R.WG_TMA and R.WG_INLINE and R.WG_VEC_STORE and not R.WGMMA_ALIGNED
    program = lower(_graph(16384, 3472, 576))
    assert _engines(program) == ['pops4b_tma_edge_128x256'] and 'pops4b::tma_phase_inline<128,256,2,true,false,3,true,true>' in program.source
    assert _engines(lower(_graph(100, 72, 40)))[0].startswith('popsc_bf16_edge_')     # small tails: mma.sync edges


def test_disabled_rule_restores_fallback():
    R.EDGE_TAILS = False
    assert _engines(lower(_graph(100, 72, 40))) == ['generic_bf16_simt_16']


def test_edge_plan_covers_extents():
    for m, n, k in [(16384, 3472, 576), (16384, 576, 3472), (576, 3472, 16384), (1, 1, 1), (33, 65, 17)]:
        plan, cands = R.bf16_plan(m, n, k, 1, True, False, edge=True)
        mp, np_, kp = plan['padded']
        assert mp >= m and np_ >= n and kp >= k and kp % plan['ks'] == 0
        assert plan['tiles'] == -(-m // plan['bm']) * -(-n // plan['bn'])
        assert all(c['shared_bytes'] <= R.SHARED_LIMIT for c in cands)
    with pytest.raises(R.MatrixRuleError):
        R.bf16_plan(100, 72, 40, 1, True, False)              # the aligned planner still refuses tails


def test_edge_products_get_no_generated_epilogue():
    from megakernel_lab.operation_compiler import pops2b_rules_p7b as pops2b_rules

    class Node:
        kind, name = 'gemm', 'c'
        extra = {'matrix_spec': {'dtype': 'bfloat16', 'batch': 1, 'edge': True}}
    assert pops2b_rules.eligible_product(Node(), {}) is None


@pytest.mark.parametrize('m,n,k,at,bt', [(16384, 3472, 576, False, False), (16384, 576, 3472, False, True),
                                         (576, 3472, 16384, True, False), (1001, 577, 1024, False, True)])
def test_warpgroup_body_selection(m, n, k, at, bt):
    saved = R.WGMMA, R.WG_INLINE
    try:
        R.WGMMA, R.WG_TMA, R.WG_INLINE = True, False, False                  # cp.async form
        program = lower(_graph(m, n, k, at=at, bt=bt))
        engines = _engines(program)
        assert len(engines) == 1 and engines[0].startswith('pops4b_wg_edge_'), engines
        assert 'pops4b::wmm_phase<' in program.source and '#include "pops4b_wg.cuh"' in program.source
        plan = program.manifest['phases'][-1].get('matrix_plan') or program.manifest['phases'][-1]['members'][0]['matrix_plan']
        assert plan['shared_bytes'] <= program.manifest['shared_bytes']
        R.WG_INLINE = True
        assert 'pops4b::wmm_phase_inline<' in lower(_graph(m, n, k, at=at, bt=bt)).source
    finally:
        R.WGMMA, R.WG_INLINE = saved


def test_warpgroup_body_leaves_aligned_and_small_products():
    saved = R.WGMMA
    try:
        R.WGMMA = True
        assert 'wmm_phase' not in lower(_graph(16384, 3456, 576)).source        # aligned: WGMMA_ALIGNED is off
        assert 'wmm_phase' not in lower(_graph(100, 72, 40)).source             # below WG_MIN_FLOPS
    finally:
        R.WGMMA = saved


def test_tma_body_tensor_maps():
    saved = R.WGMMA, R.WG_TMA
    try:
        R.WGMMA, R.WG_TMA, R.WG_INLINE = True, True, False
        program = lower(_graph(16384, 576, 3472, bt=True))
        maps = program.manifest['tensor_maps']
        assert 'POPS4B_TMAP' not in program.source and 'pops4b::tma_phase<' in program.source
        assert len(maps) == 2 and all(m['stride_bytes'] % 16 == 0 and m['box'][0] == 64 for m in maps)
        a, b = maps
        assert a['root'] == 'a' and a['dims'] == [3472, 16384]          # K-major dZ: dims {K, rows}
        assert b['root'] == 'bT' and b['dims'] == [3472, 576]            # Wᵀ view of W [576, 3472]: K-major rows n
        assert '(reinterpret_cast<const char*>(args[2]) + 128)' in program.source
    finally:
        R.WGMMA, R.WG_TMA = saved
