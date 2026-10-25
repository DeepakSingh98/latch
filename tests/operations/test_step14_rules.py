"""CPU contracts for the release-step-14 flat common-planner rules (no GPU, no nvcc).

* merge_elementwise_chains: a later same-shape phase reading an earlier one at the identity index joins it (also
  when it reads its own intermediates); an operand produced at or after the earlier phase's level is refused.
* ecta: only self-contained elementwise loops are restricted to idle CTAs; the restriction keeps every element.
* FP32 edge planning: K is covered exactly by the slices (the k step divides K and every slice), and the
  established FP32 tile set still wins wherever it divides M and N; STEP14_RULES=False restores step-13 admission.
* Edge admission is scoped to the standalone flat planner (no effect outside it).
"""
from __future__ import annotations

import math

from megakernel_lab.operation_compiler import compose, group_rules, matrix_rules
from megakernel_lab.operation_compiler.lower import TensorLayout


def _lay(name, shape):
    strides, stride = [], 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    return TensorLayout(tuple(shape), tuple(reversed(strides)), name)


def _ident(n):
    return 'v_' + n


def _index(layout, shape, idx='i'):
    return f'({idx})'


def _phases(extra_read=None):
    shape = (24, 48, 48)
    layouts = {n: _lay(n, shape) for n in ('bmm', 'div', 'sub_1', 'pow_2', 'target', 'late')}
    phases = [
        {'kind': 'splitk_reduce', 'names': ['bmm'], 'targets': ['r'], 'reads': ['part'], 'writes': ['bmm']},
        {'kind': 'elementwise', 'names': ['div'], 'targets': ['d'], 'reads': ['bmm'], 'writes': ['div'],
         'shape': shape, 'body': ['const float r_div = (v_bmm[(i)] / 3.0f); v_div[i] = r_div;']},
        {'kind': 'elementwise', 'names': ['late'], 'targets': ['l'], 'reads': ['div'], 'writes': ['late'],
         'shape': shape, 'body': ['const float r_late = (v_div[(i)] * 2.0f); v_late[i] = r_late;']},
        {'kind': 'elementwise', 'names': ['sub_1', 'pow_2'], 'targets': ['s', 'p'],
         'reads': ['div', 'target', 'sub_1'] + ([extra_read] if extra_read else []), 'writes': ['sub_1', 'pow_2'],
         'shape': shape, 'body': ['const float r_sub_1 = (v_div[(i)] - v_target[(i)]); v_sub_1[i] = r_sub_1;',
                                  'const float r_pow_2 = (r_sub_1 * r_sub_1); v_pow_2[i] = r_pow_2;']},
    ]
    return phases, layouts


def test_chain_merge_with_self_reads():
    phases, layouts = _phases()
    merged, records = group_rules.merge_elementwise_chains(phases, layouts, ident=_ident, index=_index,
                                                           inplace_roots=set())
    names = [p['names'] for p in merged]
    assert ['div', 'late', 'sub_1', 'pow_2'] in names or ['div', 'sub_1', 'pow_2', 'late'] in names
    body = next(p for p in merged if 'sub_1' in p['names'])['body']
    assert any('r_div - v_target' in line or '(r_div -' in line for line in body)


def test_chain_merge_refuses_operand_at_or_after_level():
    phases, layouts = _phases(extra_read='late')
    merged, _ = group_rules.merge_elementwise_chains(phases, layouts, ident=_ident, index=_index,
                                                     inplace_roots=set())
    owner = {n: k for k, p in enumerate(merged) for n in p['names']}
    # 'late' sits at div's level + 1; sub_1 may join late's chain but never div's phase ahead of 'late'.
    assert owner['sub_1'] != owner['div'] or owner['late'] == owner['div']


def test_ecta_eligibility_and_restriction():
    loop = ('for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < 100LL; '
            'i += (long long)gridDim.x * blockDim.x) { v_y[i] = v_x[i]; }')
    assert group_rules.ecta_eligible({'kind': 'elementwise', 'code': loop})
    for unsafe in ('__syncthreads();', 'smem[0] = 1;', 'popsc::foo();', 'atomicAdd(p, 1);'):
        assert not group_rules.ecta_eligible({'kind': 'elementwise', 'code': loop + unsafe})
    restricted = group_rules.ecta_restrict(loop, 81)
    assert 'blockIdx.x >= 81u' in restricted and 'gridDim.x <= 81u' in restricted
    assert 'ecta_b_ * blockDim.x' in restricted and 'ecta_g_ * blockDim.x' in restricted


def test_fp32_edge_plans_cover_k_exactly():
    for m, n, k, batch in ((48, 48, 1056, 24), (40, 40, 1120, 27), (48, 48, 992, 26), (48, 1056, 48, 24),
                           (40, 1120, 40, 27), (1056, 48, 48, 24)):
        chosen, candidates = matrix_rules.fp32_edge_plan(m, n, k, batch, True, False)
        for c in candidates:
            assert k % c['kc'] == 0 and c['ks'] % c['kc'] == 0
            assert (c['split'] - 1) * c['ks'] < k <= c['split'] * c['ks']
            assert c['tiles'] == c['split'] * batch * math.ceil(m / c['bm']) * math.ceil(n / c['bn'])


def test_established_fp32_tiles_unchanged():
    # Products the step-13 rule admits keep their candidate set (FP32_TILES) and plan.
    chosen, candidates = matrix_rules.fp32_plan(64, 64, 128, 8, True, True)
    assert all((c['bm'], c['bn']) in matrix_rules.FP32_TILES for c in candidates)
    assert 'split' not in chosen


def test_edge_admission_scoped_to_flat_planner():
    token = compose.STEP14_FLAT_ACTIVE.set(False)
    try:
        assert not matrix_rules.fp32_edge_applies(48, 48, 1056)
    finally:
        compose.STEP14_FLAT_ACTIVE.reset(token)
    token = compose.STEP14_FLAT_ACTIVE.set(True)
    try:
        assert matrix_rules.fp32_edge_applies(48, 48, 1056)
        previous = compose.STEP14_RULES
        compose.STEP14_RULES = False
        try:
            assert not matrix_rules.fp32_edge_applies(48, 48, 1056)
        finally:
            compose.STEP14_RULES = previous
    finally:
        compose.STEP14_FLAT_ACTIVE.reset(token)


def test_step14_features_registered():
    for feature in ('tmae', 'tmaj', 'ecta', 'emerge', 'w96', 'tdeep'):
        assert feature in group_rules.FEATURES
    assert group_rules.features_of('p5c+p5ce+vec4+vec4u' + compose.STEP14_FEATURES) >= {'tmae', 'emerge', 'ecta'}
