"""CPU contracts for LATCH-MATRIX candidates (no CUDA). Run on Modal CPU (one thread per case)."""
from __future__ import annotations

import copy
import re

import pytest

from megakernel_lab.operation_compiler.matrix_adapters import (MatrixProvider, ProgramResources, candidates, dense, expert_products, linear_products,
                          select)
from megakernel_lab.operation_compiler.matrix_adapters.bodies import BY_NAME, _popsc_spec

DEV_SHAPES = [(2048, 1024, 4096), (2048, 4096, 1024)]
HELD_OUT_SHAPES = [(3072, 1536, 6144), (2000, 1000, 2600), (1500, 1000, 3000)]
IDENT = lambda name: 'v_' + name


# ------------------------------------------------------------------------------------------- problem layouts
def test_linear_product_layouts():
    fwd, dx, dw = linear_products(2048, 1024, 4096)
    assert (fwd.m, fwd.n, fwd.k, fwd.ak, fwd.bkc) == (2048, 4096, 1024, True, True)
    assert (dx.m, dx.n, dx.k, dx.ak, dx.bkc) == (2048, 1024, 4096, True, False)
    assert (dw.m, dw.n, dw.k, dw.ak, dw.bkc) == (4096, 1024, 2048, False, False)
    assert dw.a.row_stride == 1 and dw.a.col_stride == 4096        # dY^T view of [T, out]
    e = expert_products(8, 128, 256, 512)[0]
    assert e.batch == 8 and e.a.batch_stride == 128 * 256 and e.b.batch_stride == 256 * 512


# --------------------------------------------------------------------------------- donor spec equivalence
def _layouts(p):
    la, lb = p.layouts()
    return la, lb, p.out_shape()


@pytest.mark.parametrize('shape', DEV_SHAPES + HELD_OUT_SHAPES)
@pytest.mark.parametrize('which', [0, 1, 2])
def test_spec_matches_donor_bf16_spec(shape, which):
    """The adapter builds exactly the spec the donor's own admission builds from TensorLayouts."""
    from megakernel_lab.operation_compiler import matrix_rules, matrix_rules_p7b
    p = linear_products(*shape)[which]
    la, lb, out_shape = _layouts(p)
    mine = _popsc_spec(p, p.a.root, p.b.root, 'v_C')
    if not (p.m % 32 or p.n % 32 or p.k % 32):
        donor = matrix_rules.bf16_spec(p.a.root, p.b.root, 'v_C', la, lb, out_shape)
        assert {k: donor[k] for k in mine} == mine
    donor7 = matrix_rules_p7b.bf16_spec(p.a.root, p.b.root, 'v_C', la, lb, out_shape)
    assert {k: donor7[k] for k in mine} == mine


@pytest.mark.parametrize('shape', DEV_SHAPES)
def test_popsc_emission_identical_to_frozen_emit_group(shape):
    """Single-product popsc emission equals the frozen reference schedule's emit_group text."""
    from megakernel_lab.operation_compiler import matrix_rules
    for p in linear_products(*shape):
        la, lb, out_shape = _layouts(p)
        spec = matrix_rules.bf16_spec(p.a.root, p.b.root, 'v_C', la, lb, out_shape)
        frozen, _ = matrix_rules.emit_group([spec], ident=IDENT)
        c = [x for x in candidates(p)[0] if x.body == 'popsc_bf16'][0]
        assert c.emit(ident=IDENT, a=p.a.root, b=p.b.root, out='v_C').code == frozen


def test_p5c_emission_identical_to_donor_code():
    from megakernel_lab.operation_compiler import p5c_matrix
    from megakernel_lab.operation_compiler.matrix_adapters.bodies import P5C_FLAT, pinned
    p = linear_products(2048, 1024, 4096)[0]
    c = [x for x in candidates(p)[0] if x.body == 'p5c'][0]
    with pinned('p5c_matrix', P5C_FLAT):
        spec = {**_popsc_spec(p, 'X', 'W', 'v_Y'), 'engine': 'p5c', 'roots': [('X', 0), ('W', 0)]}
        donor = p5c_matrix.code(spec, c.plan, IDENT, 0, 1, 1, 0)
    assert c.emit(ident=IDENT, a='X', b='W', out='v_Y').code == donor


# ------------------------------------------------------------------------------------------- selection table
EXPECTED = {
    ('forward', 2048, 4096, 1024): 'p8ws', ('grad_input', 2048, 1024, 4096): 'p8ws',
    ('grad_weight', 4096, 1024, 2048): 'p8ws',
}


def test_selection_dev_dense():
    for p in linear_products(2048, 1024, 4096):
        chosen, decision = select(p)
        assert chosen.body == EXPECTED[(p.role, p.m, p.n, p.k)], decision['rule']
        assert chosen.requirements.arch == 'sm_90a' and chosen.requirements.sync.entry_resident


@pytest.mark.parametrize('shape', HELD_OUT_SHAPES)
def test_selection_held_out_never_generic(shape):
    """Held-out dense shapes (aligned and awkward) all receive a tensor-core body, never the SIMT fallback."""
    for p in linear_products(*shape):
        chosen, decision = select(p)
        assert chosen is not None and chosen.body != 'mixed_generic', decision
        if p.m % 32 or p.n % 32 or p.k % 32:
            assert chosen.body in ('pops4b_tma', 'pops4b_wg', 'popsc_bf16_edge')


@pytest.mark.parametrize('batch_tma,expected', [
    (True, ('pops4b_tma', 'pops4b_tma', 'pops4b_tma')),
    (False, ('pops4b_wg', 'pops4b_wg', 'pops4b_wg')),
])
def test_selection_experts_batched(monkeypatch, batch_tma, expected):
    from megakernel_lab.operation_compiler.matrix_adapters import bodies
    monkeypatch.setattr(bodies, 'BATCH_TMA', batch_tma)
    # Accepted c24 extends the TMA body to M-major batched A (expert dW).
    for p, body in zip(expert_products(16, 256, 1024, 1024), expected):
        chosen, decision = select(p)
        assert chosen.body == body, (p.role, decision)
    small = dense(48, 80, 112, batch=4)
    assert select(small)[0].body == 'popsc_bf16_tail'


def test_frozen_reference_policy_reproduces_reference_order():
    assert select(linear_products(2048, 1024, 4096)[0], policy='frozen_reference')[0].body == 'popsc_bf16'
    assert select(dense(48, 80, 112), policy='frozen_reference')[0].body == 'popsc_bf16_tail'
    assert select(linear_products(2000, 1000, 2600)[0], policy='frozen_reference')[0].body == 'mixed_generic'


def test_odd_leading_dimension_falls_back_explicitly():
    p = dense(2000, 1001, 1000, ak=True, bkc=True)    # output stride odd is fine; operands aligned
    assert select(p)[0].body in ('pops4b_tma', 'pops4b_wg')
    q = dense(2000, 1000, 1001, ak=True, bkc=False)   # A row stride 1001: not 16-byte aligned
    chosen, decision = select(q)
    assert chosen.body == 'mixed_generic'
    refusals = {c['body']: c for c in decision['candidates']}
    assert 'pops4b_tma' in decision['refused'] and 'aligned' in decision['refused']['pops4b_tma']


# ------------------------------------------------------------------------------------------- resources
def test_resource_refusal_two_cta_and_arch():
    p = linear_products(2048, 1024, 4096)[0]
    two = ProgramResources(resident_ctas_per_sm=2, dynamic_shared_limit=98304)
    chosen, decision = select(p, two)
    refused = {c['body']: c['reasons'] for c in decision['candidates'] if not c['admitted']}
    assert 'p8ws' in decision['refused'] and any('CTA/SM' in r for r in refused['p5c'])
    assert chosen is not None and chosen.requirements.dynamic_shared_bytes <= 98304
    old = ProgramResources(arch='sm_90')
    chosen, decision = select(p, old)
    assert chosen.requirements.arch == 'sm_90' and chosen.body == 'popsc_bf16'
    small = ProgramResources(dynamic_shared_limit=120 * 1024)
    chosen, decision = select(p, small)
    assert chosen.requirements.dynamic_shared_bytes <= 120 * 1024


def test_bitwise_class_constraint():
    p = linear_products(2048, 1024, 4096)[0]
    chosen, _ = select(p, ProgramResources(bitwise_class='bf16_mma_sync_k16'))
    assert chosen.requirements.precision.bitwise_class == 'bf16_mma_sync_k16'


def test_requirements_record_every_field():
    for p in [*linear_products(2048, 1024, 4096), *linear_products(2000, 1000, 2600), *expert_products(8, 256, 512, 512)]:
        for c in candidates(p)[0]:
            r = c.record()
            req = r['requirements']
            assert req['arch'] in ('sm_90', 'sm_90a') and req['sync']['threads_per_cta'] == 256
            assert req['precision']['accumulate'] == 'float32' and req['dynamic_shared_bytes'] > 0
            assert r['provenance']




def _graph(m, d, f, classes, *, softmax=True, batch=None):
    """Captured-format graph: BF16 product chain with an optional auxiliary log_softmax head."""
    values, ops = {}, []

    def value(name, shape, dtype, strides=None, role='intermediate'):
        if strides is None:
            strides = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                strides[i] = strides[i + 1] * shape[i + 1]
        values[name] = {'name': name, 'shape': list(shape), 'dtype': dtype, 'strides': list(strides), 'role': role}

    def op(name, target, args, kwargs=None):
        ops.append({'name': name, 'target': target, 'args': args, 'kwargs': kwargs or {}})

    r = lambda n: {'ref': n}
    for name, shape in (('x', (m, d)), ('W1', (d, f)), ('W2', (f, d)), ('Wh', (d, classes))):
        value(name, shape, 'float32', role='input')
    value('cx', (m, d), 'bfloat16'); op('cx', 'aten._to_copy.default', [r('x')], {'dtype': 'bfloat16'})
    value('c1', (d, f), 'bfloat16'); op('c1', 'aten._to_copy.default', [r('W1')], {'dtype': 'bfloat16'})
    value('h', (m, f), 'bfloat16'); op('h', 'aten.mm.default', [r('cx'), r('c1')])
    value('c2', (f, d), 'bfloat16'); op('c2', 'aten._to_copy.default', [r('W2')], {'dtype': 'bfloat16'})
    value('y', (m, d), 'bfloat16'); op('y', 'aten.mm.default', [r('h'), r('c2')])
    value('ch', (d, classes), 'bfloat16'); op('ch', 'aten._to_copy.default', [r('Wh')], {'dtype': 'bfloat16'})
    value('z', (m, classes), 'bfloat16'); op('z', 'aten.mm.default', [r('y'), r('ch')])
    value('z32', (m, classes), 'float32'); op('z32', 'aten._to_copy.default', [r('z')], {'dtype': 'float32'})
    outputs = ['z32']
    if softmax:
        value('ls', (m, classes), 'float32'); op('ls', 'aten._log_softmax.default', [r('z32'), 1, False])
        outputs = ['ls']
    return {'values': values, 'operations': ops, 'inputs': ['x', 'W1', 'W2', 'Wh'], 'outputs': outputs}



def test_normal_compose_awkward_products_keep_masked_matrix_bodies():
    from megakernel_lab.operation_compiler.lower import lower
    graph = _graph(1000, 1000, 1304, 248)
    legacy = lower(copy.deepcopy(graph))
    composed = lower(copy.deepcopy(graph), execution="compose")
    phases = composed.manifest["phases"]
    adapted = [p for p in phases if p.get("engine", "").startswith("latch_matrix:")]
    assert adapted and any(p["engine"] == "latch_matrix:pops4b_tma" for p in adapted)
    assert composed.manifest["execution_selection"]["selected"] == "common_static"
    assert composed.manifest["tensor_maps"]
    assert "POPS4B_TMAP(" not in composed.source
    for phase in adapted:
        assert phase["reads"] and phase["writes"]
        assert phase["matrix_spec"]["roots"]
        assert phase["shared_bytes"] <= 232448
        assert phase["threads"] == 256
    # A normal composed call must not install state which changes legacy output.
    again = lower(copy.deepcopy(graph))
    assert (again.source, again.manifest) == (legacy.source, legacy.manifest)


def test_unused_provider_does_not_change_program_metadata():
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    provider = MatrixProvider(only={"pops4b_tma", "pops4b_wg", "popsc_bf16_edge"})
    program = GeneratedProgram("source", {"existing": True}, "body", "helpers")
    assert provider.finalize(program) is program
