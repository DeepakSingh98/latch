"""P8-B CPU contracts: declared BF16-operand attention, v2 attention items, epilogue/row/reduction codegen rules.

Run on Modal CPU (local torch cannot import ops). GPU qualification of the same rules is separate (p8_b/).
"""
from __future__ import annotations

import re

import torch

from megakernel_lab.operation_compiler.loop import capture_loop_training


def _tiny(precision=None, sequence=256):
    from megakernel_lab.fast_examples.recovered.transformer import TransformerConfig, fixture, initial_state, make_loop_functions, parameters
    from .fixtures.stage3_training import OPTIMIZER
    model = TransformerConfig(batch=2, sequence=sequence, width=128, heads=2, layers=1, expansion=1,
                              matrix_dtype='bfloat16')
    p = parameters(model)
    momentum = {k: torch.zeros_like(v) for k, v in p.items()}
    data = fixture(model)
    functions = make_loop_functions(model, intermediate=True)
    captured = capture_loop_training(*functions, p, momentum, data, torch.tensor(1), OPTIMIZER, max_steps=2,
                                     state=initial_state(), storage_mode='save_body_tape', precision=precision)
    return captured, (model, p, momentum, initial_state(), data)


def test_policy_long_attention_one_piece():
    from megakernel_lab.operation_compiler.lower import lower
    captured, _ = _tiny('bf16_attention_operands')
    tape = captured.ir.metadata['attention_tape']
    assert len(tape['groups']) == 1, tape['refusals']
    group = tape['groups'][0]
    assert group['kernel'] == 'long' and group['splits'] == 1 and group['bf16_operands'], group
    assert tape['numerical_policy']['attention'] == 'bf16_operands'
    assert captured.ir.metadata['precision_policy']['attention'] == 'bf16_operands'
    source = lower(captured.ir, policy='auto').source
    assert 'p9att::forward_phase4<' in source and 'p9att::backward_phase6<' in source
    assert 'A.practical_exp = false;' in source
    assert '#define P9A_FAST_EXP' not in source


def test_default_long_attention_exact_v2():
    from megakernel_lab.operation_compiler.lower import lower
    captured, _ = _tiny(None)
    group = captured.ir.metadata['attention_tape']['groups'][0]
    assert group['splits'] == 3 and not group['bf16_operands']
    assert captured.ir.metadata['attention_tape']['numerical_policy']['splits'] == 3
    source = lower(captured.ir, policy='auto').source
    assert 'plong::forward_phase2<3>' in source and 'plong::backward_phase2<3>' in source


def test_policy_graph_matches_policy_reference():
    from .fixtures.stage3_training import _loss_for, _passed, compare_step, reference_step
    from megakernel_lab.operation_compiler.precision import wrap
    captured, (model, p, momentum, state, data) = _tiny('bf16_attention_operands')
    flat = (*p.values(), *momentum.values(), *state.values(), *data, torch.tensor(2))
    actual = captured.unpack(captured.graph_step(*flat))
    expected = reference_step(wrap(_loss_for(model, 'combined', 2), 'bf16_attention_operands'), p, momentum, data,
                              state=state)
    checks = compare_step(actual, expected, p, arithmetic='bfloat16')
    assert _passed(checks), 'rewritten policy graph differs from the declared eager recipe'


def test_flat_plan_refuses_roundings():
    from megakernel_lab.operation_compiler import attention_lower
    from megakernel_lab.operation_compiler.lower import _field, _ref
    captured, _ = _tiny('bf16_attention_operands')
    # The tape rewrite consumed the group; the unrewritten joint is not retained, so check the rule directly on the
    # forward region: no group may be admitted there without allow_rounding.
    region = captured.ir.body_saved
    names = set(region.values)
    groups, _ = attention_lower.plan(region.operations, region.values, names, ref=_ref, field=_field,
                                     output_name=lambda op: op.outputs[0], protected=set(region.outputs))
    assert not any(g.get('p_round') for g in groups)


def test_philox_row_blocks():
    from megakernel_lab.operation_compiler.row_fusion import _philox_rows
    code = ['for (long long o = 0; o < 4LL; ++o) {', '  const int lane = threadIdx.x & 31;',
            '  for (int w = 0; w < 4; ++w) {', '    const long long i = o * 128LL + 32 * w + lane;',
            '    const float r_m = latch_philox_mask(v_s, i, 7ULL, 1.25f); y[i] = r_m;', '  }', '}']
    out = '\n'.join(_philox_rows(code, 128))
    assert 'latch_philox_block(v_s, ((unsigned long long)(o * 128LL) >> 2) + 32ULL * m_' in out
    assert 'latch_philox_mask_pre(pw0_[w >> 2], 8 * (w & 3) + (lane >> 2), lane & 3, 7ULL, 1.25f)' in out
    assert _philox_rows(code, 96) == code   # width % 128 != 0 keeps per-element evaluation


def test_vector_epilogue_and_fast_activation():
    from megakernel_lab.operation_compiler.group_rules import _fast_activation, _vector_epilogue
    lines = ['const float r_a = __bfloat162float(v_x[(i)]);',
             'const float r_s = (r_a / (1.0f + expf(-r_a)));',
             'const __nv_bfloat16 r_o = __float2bfloat16_rn(r_s); v_o[i] = r_o;']
    prelude, ld, decl, out, tail = _vector_epilogue(lines, {'v_x': 'bfloat16', 'v_o': 'bfloat16'})
    assert 'uint4 p8l0_[8];' == prelude and 'reinterpret_cast<const uint4*>(v_x + i0)' in ld
    assert 'p5c::upkb(p8l0_[e], j)' in out[0] and out[2].endswith('s80_[j] = r_o;') and tail == 'p5c::st8(v_o + i0, s80_);'
    assert _fast_activation(lines[1]) == 'const float r_s = __fdividef(r_a, 1.0f + __expf(-r_a));'
    assert _fast_activation('const float r_g = __frcp_rn((1.0f + expf(-r_x)));') == \
        'const float r_g = __fdividef(1.0f, 1.0f + __expf(-r_x));'


def test_colreg_lowering():
    from megakernel_lab.operation_compiler.execution_policy import RECURRENT_RULES
    from megakernel_lab.operation_compiler.lower import lower
    assert 'colreg' in RECURRENT_RULES.split('+')
    captured, _ = _tiny(None)
    source = lower(captured.ir, policy='auto').source
    assert re.search(r'P\[w \* 32 \+ lane\] = ', source), 'coalesced exact column reduction emitted'


def run():
    results = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            results[name] = True
    return {'passed': True, 'tests': results}
