"""Structural selection contracts; arithmetic and native clocks are cloud gates."""
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler.compose import long_protocol_common_upgrade
from megakernel_lab.operation_compiler.matrix_adapters.provider import compose_policy, matrix_choices


def fixture():
    provider = SimpleNamespace(policy='donor', force=None, inventory=None,
        decisions=[{'output': 'projection', 'problem_key': 'admitted-product',
                    'selected': 'popsc_bf16', 'step13_body': 'pops4b_tma'}])
    manifest = {'barrier_count': 8,
                'phases': [{'kind': 'gemm', 'members': [{'engine': 'popsc_bf16_64x32'}]}]}
    proof = {'protocol': 'p6d', 'attention_groups': ['attention'], 'matrix_bodies': []}
    return provider, manifest, proof


def test_verified_long_protocol_can_keep_existing_step13_product_admission():
    provider, manifest, proof = fixture()
    calls = []

    def relower():
        calls.append(compose_policy())
        return SimpleNamespace(manifest={'barrier_count': 6}), SimpleNamespace(policy='step13')

    program, changed_provider, rule = long_protocol_common_upgrade(provider, manifest, proof, relower)
    assert program.manifest['barrier_count'] == 6 and changed_provider.policy == 'step13'
    assert calls == ['step13'] and compose_policy() == 'donor'
    assert rule['changed_products'] == [{'output': 'projection', 'from': 'popsc_bf16', 'to': 'pops4b_tma'}]


@pytest.mark.parametrize('common,established', [(8, 8), (0, 0), (6, 8)])
def test_common_upgrade_preserves_or_reduces_global_synchronization(common, established):
    provider, manifest, proof = fixture()
    manifest['barrier_count'] = established
    assert long_protocol_common_upgrade(provider, manifest, proof,
        lambda: (SimpleNamespace(manifest={'barrier_count': common}), provider)) is not None


@pytest.mark.parametrize('common,established', [(9, 8), (92, 50), (None, 8), (8, None),
                                              (True, 8), (8, True), (-1, 8), (8, -1)])
def test_extra_or_unproven_global_synchronization_keeps_established(common, established):
    provider, manifest, proof = fixture()
    manifest['barrier_count'] = established
    provider.inventory = [{'prior': True}]
    provider.inventory_start = 0

    def relower():
        provider.inventory.append({'rejected_candidate': True})
        return SimpleNamespace(manifest={'barrier_count': common}), provider

    assert long_protocol_common_upgrade(provider, manifest, proof, relower) is None
    assert provider.inventory == [{'prior': True}]
    assert compose_policy() == 'donor'


@pytest.mark.parametrize('proof', [None, {}, {'protocol': 'reference'},
    {'protocol': 'p6d', 'attention_groups': []},
    {'protocol': 'p6d', 'attention_groups': ['a'], 'matrix_bodies': [{'engine': 'p5c'}]}])
def test_unverified_or_non_attention_programs_cannot_change_schedule(proof):
    provider, manifest, _ = fixture()
    assert long_protocol_common_upgrade(provider, manifest, proof, lambda: pytest.fail('unexpected lowering')) is None


@pytest.mark.parametrize('member', [
    {'engine': 'p5c'}, {'engine': 'p5c_ws_f32'}, {'matrix_spec': {'engine': 'p5c'}},
    {'engine': 'p8ws'}, {'items': [{'engine': 'p5c_128x128_f32'}]},
])
def test_any_established_p5c_product_keeps_fused_schedule(member):
    provider, manifest, proof = fixture()
    manifest['phases'][0]['members'].append(member)
    assert long_protocol_common_upgrade(provider, manifest, proof, lambda: pytest.fail('unexpected lowering')) is None


@pytest.mark.parametrize('change', ['no_admission', 'already_step13', 'forced', 'no_provider'])
def test_no_new_upgrade_is_invented(change):
    provider, manifest, proof = fixture()
    if change == 'no_admission':
        provider.decisions[0].pop('step13_body')
    elif change == 'already_step13':
        provider.policy = 'step13'
    elif change == 'forced':
        provider.force = 'popsc_bf16'
    else:
        provider = None
    assert long_protocol_common_upgrade(provider, manifest, proof, lambda: pytest.fail('unexpected lowering')) is None


def test_explicit_matrix_policy_is_respected():
    provider, manifest, proof = fixture()
    with matrix_choices({}, policy='donor'):
        assert long_protocol_common_upgrade(provider, manifest, proof, lambda: pytest.fail('unexpected lowering')) is None


def test_normal_compose_uses_the_qualified_common_candidate_without_a_flag():
    from megakernel_lab.operation_compiler.lower import lower
    from .test_attention_mask_binding import _graph
    graph = _graph(bf16=True, t=512, d=256, h=4, b=4, mask='none')
    normal = lower(graph, execution='compose', target_arch='sm_90')
    common = lower(graph, execution='compose', compose_selection='common', target_arch='sm_90')
    selection = normal.manifest['execution_selection']
    assert selection['selected'] == 'common_static'
    assert selection['performance_selection']['rule'] == 'homogeneous_long_without_p5c_step13_products'
    assert selection['step13_default_rule']['changed_products']
    assert normal.source == common.source
    assert normal.manifest['header_hashes'] == common.manifest['header_hashes']
