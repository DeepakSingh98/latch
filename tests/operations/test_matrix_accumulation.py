"""CPU-worker structural/emission checks; no CUDA compilation or execution."""
from dataclasses import replace

import pytest
import torch

from megakernel_lab.operation_compiler import capture_inference
from megakernel_lab.operation_compiler.lower import TensorLayout, lower
from megakernel_lab.operation_compiler.matrix_accumulation import BODY, WGMMA_BODY, CONTRACT, KEY, applies, record, validate
from megakernel_lab.operation_compiler.matrix_adapters.problem import MatrixProblem, Operand
from megakernel_lab.operation_compiler.matrix_adapters.provider import MatrixProvider, problem_key
from megakernel_lab.operation_compiler.matrix_adapters.requirements import ProgramResources


def problem():
    return MatrixProblem(64, 64, 256, Operand('dy', 1, 64), Operand('x', 64, 1))


def emit(provider, p=None, **changes):
    p = p or problem()
    arguments = dict(a='a', b='b', name='dw', out='output',
        la=TensorLayout(p.a_shape(), p.a_strides(), p.a.root, p.a.offset),
        lb=TensorLayout(p.b_shape(), p.b_strides(), p.b.root, p.b.offset),
        shape=p.out_shape(), dtypes=(p.dtype, p.dtype, p.out_dtype),
        ident=lambda name: name, target_arch='sm_90', native='p5c', out_pitch=None)
    return provider.product(**{**arguments, **changes})


def test_scope_is_structural_and_not_a_model_or_output_name():
    p = problem()
    assert applies(p)
    assert applies(replace(p, out='unrelated', role='not_a_weight'))
    assert record(p)['shape'] == [64, 64, 256]
    assert not applies(None)


@pytest.mark.parametrize('change', [
    dict(batch=2), dict(dtype='float32'), dict(out_dtype='float32'),
    dict(out_mode='store_f32'), dict(k=64),
    dict(a=Operand('a', 256, 1)), dict(b=Operand('b', 1, 256)),
])
def test_nonmatching_products_are_not_selected(change):
    assert not applies(replace(problem(), **change))


def test_explicit_contract_reuses_existing_body_and_overrides_default_native_selection():
    provider = MatrixProvider(only={'pops4b_tma'}, matrix_accumulation_contract=CONTRACT)
    assert provider.consult_native
    result = emit(provider)
    assert result['extra']['engine'] == 'latch_matrix:pops4b_tma'
    assert result['extra'][KEY] == record(problem(), WGMMA_BODY)
    assert result['extra']['latch_matrix']['bitwise_class'] == 'bf16_wgmma_k16'
    assert 'pops4b_tma.cuh' in result['headers']
    assert result['decision']['selected'] == WGMMA_BODY
    assert result['decision']['arithmetic_contract'] == record(problem(), WGMMA_BODY)
    assert 'step13_body' not in result['decision']
    assert provider.emitted == ['dw']
    # The same admitted body under the older diagnostic force emits identical
    # code. This contract changes selection, not the body implementation.
    forced = emit(MatrixProvider(force=WGMMA_BODY, take_frozen=True), native=None)
    assert result['code'] == forced['code']
    assert result['headers'] == forced['headers']


def test_step13_replanning_cannot_change_the_arithmetic_body():
    result = emit(MatrixProvider(policy='step13', matrix_accumulation_contract=CONTRACT))
    assert result['extra']['engine'] == 'latch_matrix:pops4b_tma'
    assert 'step13_body' not in result['decision']


@pytest.mark.parametrize('change, message', [
    (dict(out_pitch=72), 'unpadded result pitch'),
    (dict(target_arch='sm_80'), 'sm_90 matrix'),
])
def test_matching_product_cannot_silently_drop_the_contract(change, message):
    with pytest.raises(ValueError, match=message):
        emit(MatrixProvider(matrix_accumulation_contract=CONTRACT), **change)


def test_matching_layout_and_resource_refusals_are_errors(monkeypatch):
    provider = MatrixProvider(matrix_accumulation_contract=CONTRACT)
    with pytest.raises(ValueError, match='body refused'):
        emit(provider, replace(problem(), a=Operand('dy', 1, 64, offset=1)))
    # The accepted c24 TMA implementation admits this zero-filled K tail;
    # the retained mma.sync-only alternative still refuses it explicitly.
    tail = emit(provider, replace(problem(), k=257))
    assert tail['extra'][KEY] == record(replace(problem(), k=257), WGMMA_BODY)
    with pytest.raises(ValueError, match='body refused'):
        emit(MatrixProvider(resources=ProgramResources(threads_per_cta=128),
                            matrix_accumulation_contract=CONTRACT))
    from megakernel_lab.operation_compiler import matrix_accumulation
    monkeypatch.setattr(matrix_accumulation, 'WGMMA', False)
    with pytest.raises(ValueError, match='body refused'):
        emit(MatrixProvider(matrix_accumulation_contract=CONTRACT), replace(problem(), k=257))


def test_conflicting_explicit_body_choices_are_rejected():
    for options in (dict(force='pops4b_tma'), dict(choices={problem_key(problem()): 'native'})):
        with pytest.raises(ValueError, match='conflicts'):
            emit(MatrixProvider(matrix_accumulation_contract=CONTRACT, **options))


def test_legacy_and_unknown_contracts():
    assert validate(None) is None and validate('legacy') is None
    for value in (None, 'legacy'):
        provider = MatrixProvider(matrix_accumulation_contract=value)
        assert not provider.consult_native
        assert emit(provider) is None
    with pytest.raises(ValueError, match='unknown matrix accumulation'):
        MatrixProvider(matrix_accumulation_contract='approximate')


def test_graph_bound_readout_k8_takes_precedence_on_a_matching_product():
    from megakernel_lab.operation_compiler.serial_bf16 import MMA_K8_CONTRACT
    binding = dict(contract=MMA_K8_CONTRACT, a='a', b='b', shape=[64,64,256],
                   role='weight_partial', partition_k=0)
    serial = dict(products={'dw': binding})
    default = emit(MatrixProvider(serial_bf16=serial))
    marked = emit(MatrixProvider(serial_bf16=serial, matrix_accumulation_contract=CONTRACT))
    assert marked == default
    assert marked['extra']['engine'] == 'latch_matrix:mma_k8_readout'
    assert KEY not in marked['extra']


def test_captured_contract_survives_common_selection_and_leaves_defaults_unchanged():
    torch.set_num_threads(1)
    def products(_p, x, dy, weight):
        return dict(hidden=dy @ weight, weight=dy.t() @ x)
    captured = capture_inference(products, {}, tuple(torch.zeros(shape, dtype=torch.bfloat16)
        for shape in ((256,64), (256,64), (64,64))))
    options = dict(target_arch='sm_90', execution='compose', vocabulary_head='off')
    legacy = lower(captured.ir, **options)
    captured.ir.metadata[KEY] = CONTRACT
    current = lower(captured.ir, **options)
    selected = [p for p in current.manifest['phases'] if KEY in p]
    assert len(selected) == 1
    assert selected[0][KEY] == record(problem(), WGMMA_BODY)
    assert selected[0]['engine'] == 'latch_matrix:pops4b_tma'
    assert selected[0]['latch_matrix']['bitwise_class'] == 'bf16_wgmma_k16'
    assert current.manifest['execution_selection']['selected'] == 'common_static'
    del captured.ir.metadata[KEY]
    assert lower(captured.ir, **options).source == legacy.source
    unrelated = capture_inference(lambda _p, a, b: a @ b, {},
        (torch.zeros(256,64,dtype=torch.bfloat16), torch.zeros(64,64,dtype=torch.bfloat16)))
    before = lower(unrelated.ir, **options)
    unrelated.ir.metadata[KEY] = CONTRACT
    after = lower(unrelated.ir, **options)
    assert after.source == before.source
    assert not any(KEY in p for p in after.manifest['phases'])
