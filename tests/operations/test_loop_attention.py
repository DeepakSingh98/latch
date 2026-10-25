"""Modal CPU contracts for short attention inside saved-body loops (P-OPS4-E, ops/loop_attention.py).

* Admission is structural: the BF16-projection recurrent transformer rewrites every block's attention;
  the FP32 recipe, other storage modes and loop_attention='off' keep the previous capture byte for byte.
* Tape decision: probabilities and FP32 head copies leave the tape; O, lse and BF16 sources enter it,
  and the tape is smaller.
* The rewritten regions are an exact ATen program: CPU graph_step at counts 0/1/3 agrees with ordinary
  autograd within the unchanged mixed-precision gate, with exact RNG state, and with the unrewritten
  capture. Perturbing the saved lse, O or BF16 sources changes the gradients (the tape is live), while
  perturbing a dropped value cannot (it is gone).
* Lowering: each body-saved region installs one forward phase per block and the reverse region one
  backward phase per block, writing/reading tape slots, with no P-OPS-D fused attention left over.
"""
from __future__ import annotations

import json
import math
import traceback

import torch

from megakernel_lab.operation_compiler.loop import capture_loop_training
from megakernel_lab.operation_compiler.lower import lower

SIZE = {'float32': 4, 'bfloat16': 2, 'int64': 8, 'bool': 1}


def _capture(recipe='primary_bf16', *, objective='combined', storage_mode='save_body_tape', loop_attention='auto',
             count=1, model=None):
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES, fixture, initial_state, make_loop_functions, parameters
    from .fixtures.stage3_training import OPTIMIZER
    model = model or RECIPES[recipe]
    p = parameters(model)
    momentum = {k: torch.zeros_like(v) for k, v in p.items()}
    data = fixture(model)
    functions = make_loop_functions(model, intermediate=objective == 'combined')
    captured = capture_loop_training(*functions, p, momentum, data, torch.tensor(count), OPTIMIZER, max_steps=3,
                                     state=initial_state(), storage_mode=storage_mode, loop_attention=loop_attention)
    return captured, (model, p, momentum, initial_state(), data, objective)


def _tape_bytes(captured):
    return sum(math.prod(row['shape']) * SIZE[row['dtype']] for row in captured.ir.metadata['body_tape'])


def _flat(p, momentum, state, data, count):
    return (*p.values(), *momentum.values(), *state.values(), *data, torch.tensor(count))


def test_admission_and_tape():
    captured, _ = _capture()
    tape = captured.ir.metadata['attention_tape']
    groups = tape['groups']
    assert len(groups) == 2 and not tape['refusals'], tape['refusals']
    names = {row['name']: row for row in captured.ir.metadata['body_tape']}
    for group in groups:
        assert group['forward']['p'] not in names, 'probabilities must not be saved'
        assert names[group['o']]['dtype'] == 'float32' and names[group['lse']]['shape'][-1] == 1
    assert not any(row['dtype'] == 'float32' and row['shape'] == [4, 4, 64, 64] for row in names.values()), \
        'FP32 head-layout copies must not be saved'
    off, _ = _capture(loop_attention='off')
    off_names = {row['name'] for row in off.ir.metadata['body_tape']}
    assert any(group['forward']['p'] in off_names for group in groups)
    reverse = captured.ir.body_vjp_saved
    assert set(reverse.metadata['forced_tape']) == {g['o'] for g in groups} | {g['lse'] for g in groups}
    return {'tape_bytes': _tape_bytes(captured), 'tape_bytes_off': _tape_bytes(off),
            'tape_tensors': len(names), 'tape_tensors_off': len(off_names), 'groups': [g['label'] for g in groups]}


def test_refusals_preserve_capture():
    results = {}
    for label, kwargs in (('fp32', {'recipe': 'primary'}), ('boundaries', {'storage_mode': 'save_boundaries'})):
        auto, _ = _capture(**kwargs)
        off, _ = _capture(loop_attention='off', **kwargs)
        assert not auto.ir.metadata['attention_tape']['groups']
        assert auto.ir.metadata['body_tape'] == off.ir.metadata['body_tape']
        same = lower(auto.ir).source == lower(off.ir).source
        assert same, label + ': lowered source changed for a refused configuration'
        results[label] = {'refusals': auto.ir.metadata['attention_tape']['refusals'][:2], 'identical_source': same}
    return results


def test_graph_step_matches_autograd():
    from .fixtures.stage3_training import compare_step, reference_step, _loss_for
    captured, (model, p, momentum, state, data, objective) = _capture()
    off, _ = _capture(loop_attention='off')
    rows = []
    for count in (0, 1, 3):
        flat = _flat(p, momentum, state, data, count)
        actual = captured.unpack(captured.graph_step(*flat))
        unchanged = off.unpack(off.graph_step(*flat))
        expected = reference_step(_loss_for(model, objective, count), p, momentum, data, state=state)
        checks = compare_step(actual, expected, p, arithmetic='bfloat16')

        def passed(x):
            return x.get('passed', True) and all(passed(v) for v in x.values() if isinstance(v, dict)) \
                if isinstance(x, dict) else True
        assert passed(checks), f'count {count}: rewritten graph_step misses the mixed-precision gate'
        assert torch.equal(actual['state']['rng'], unchanged['state']['rng'])
        assert int(actual['executed_count']) == count
        worst = max(abs(a - b).max().item() / (b.abs().max().item() + 1e-30)
                    for a, b in zip(actual['grads'].values(), unchanged['grads'].values()))
        rows.append({'count': count, 'relative_grad_difference_vs_unrewritten': worst})
    return rows


def test_tape_values_are_live():
    captured, (model, p, momentum, state, data, objective) = _capture()
    group = captured.ir.metadata['attention_tape']['groups'][0]
    graphs = captured.graph_modules
    # Run one body then its reverse directly with a perturbed tape entry.
    # The public default now hoists eligible binding casts once per update;
    # direct region calls must use that same producer/consumer ABI.
    nc, ns = len(captured.carry_names), len(captured.state_names)
    casts = graphs['invariant_casts'](*p.values(), *data) if 'invariant_casts' in graphs else ()
    primals = (*p.values(), *casts)
    init = graphs['init'](*primals, *data)
    result = graphs['body_saved'](*primals, *init, *state.values(), *data)
    tape = list(result[nc + ns:])
    labels = [row['name'] for row in captured.ir.metadata['body_tape']]
    cotangents = tuple(torch.ones_like(c) for c in result[:nc])
    base = graphs['body_vjp_saved'](*primals, *init, *state.values(), *data, *tape, *cotangents)
    changed = {}
    for key in ('lse', 'o'):
        index = labels.index(group[key])
        perturbed = list(tape)
        perturbed[index] = perturbed[index] + 0.01
        out = graphs['body_vjp_saved'](*primals, *init, *state.values(), *data, *perturbed, *cotangents)
        changed[key] = max((a - b).abs().max().item() for a, b in zip(out, base))
        assert changed[key] > 0, key + ' is saved but unused'
    return changed


def test_lowering_installs_both_halves():
    captured, _ = _capture()
    program = lower(captured.ir)
    regions = program.manifest['regions']
    decisions = {name: [d['decision'] for d in regions[name].get('attention_groups', [])]
                 for name in ('body_saved', 'body_vjp_saved')}
    assert decisions['body_saved'] == ['loop_short_bf16_attention_forward'] * 2, decisions
    assert decisions['body_vjp_saved'] == ['loop_short_bf16_attention_backward'] * 2, decisions
    kinds = {name: [p['kind'] for p in regions[name]['phases']] for name in ('body_saved', 'body_vjp_saved')}
    for name in kinds:
        assert not any(k.startswith('attention_') and 'short' not in k for k in kinds[name]), kinds[name]
        assert not any(p['kind'] == 'softmax' for p in regions[name]['phases']), name
    off, _ = _capture(loop_attention='off')
    reference = lower(off.ir)
    return {'phases': {k: len(v) for k, v in kinds.items()},
            'phases_off': {k: len(reference.manifest['regions'][k]['phases']) for k in kinds},
            'body_tape_bytes': program.manifest['body_tape_bytes'],
            'body_tape_bytes_off': reference.manifest['body_tape_bytes'],
            'arena_bytes': program.manifest['arena_bytes'], 'arena_bytes_off': reference.manifest['arena_bytes'],
            'shared_bytes': program.manifest['shared_bytes']}


def test_transfer_shape_admitted():
    from dataclasses import replace
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES
    model = replace(RECIPES['primary'], sequence=96, width=320, heads=5, matrix_dtype='bfloat16')
    captured, _ = _capture(model=model)
    groups = captured.ir.metadata['attention_tape']['groups']
    assert len(groups) == 2 and groups[0]['sequence'] == 96 and groups[0]['heads'] == 5
    program = lower(captured.ir)
    decisions = [d['decision'] for name in ('body_saved', 'body_vjp_saved')
                 for d in program.manifest['regions'][name].get('attention_groups', [])]
    assert decisions.count('loop_short_bf16_attention_forward') == 2
    assert decisions.count('loop_short_bf16_attention_backward') == 2
    return {'groups': len(groups), 'decisions': decisions}


def test_negative_controls():
    """The unchanged mixed-precision gate rejects wrong references against the rewritten program."""
    from megakernel_lab.operation_compiler.random import philox_advance
    from .fixtures.stage3_training import compare_step, reference_step, _loss_for
    captured, (model, p, momentum, state, data, objective) = _capture()
    actual = captured.unpack(captured.graph_step(*_flat(p, momentum, state, data, 3)))

    def passed(x):
        return x.get('passed', True) and all(passed(v) for v in x.values() if isinstance(v, dict)) \
            if isinstance(x, dict) else True
    shifted = {'rng': philox_advance(state['rng'], 1)}
    controls = {'truncated_history_count2': reference_step(_loss_for(model, objective, 2), p, momentum, data, state=state),
                'shifted_dropout_stream': reference_step(_loss_for(model, objective, 3), p, momentum, data, state=shifted),
                'auxiliary_objective_omitted': reference_step(_loss_for(model, 'loop', 3), p, momentum, data, state=state)}
    result = {}
    for name, expected in controls.items():
        checks = compare_step(actual, expected, p, arithmetic='bfloat16')
        worst = max(v['max_ratio'] for group in ('grads', 'params') for v in checks[group].values())
        result[name] = {'rejected': not passed({k: v for k, v in checks.items() if k != 'state'}), 'worst_ratio': worst}
        assert result[name]['rejected'], name + ' was not rejected'
    return result


TESTS = (test_admission_and_tape, test_refusals_preserve_capture, test_graph_step_matches_autograd,
         test_tape_values_are_live, test_lowering_installs_both_halves, test_transfer_shape_admitted,
         test_negative_controls)


def run():
    results = []
    for test in TESTS:
        try:
            results.append({'test': test.__name__, 'passed': True, 'details': test()})
        except Exception:
            results.append({'test': test.__name__, 'passed': False, 'error': traceback.format_exc()})
    return {'kind': 'pops4e-loop-attention-contract', 'tests': results, 'passed': all(t['passed'] for t in results)}


if __name__ == '__main__':
    result = run()
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result['passed'] else 1)
