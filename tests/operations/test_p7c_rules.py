"""Modal CPU contracts for P7-C rules (structure only; numerical evidence is GPU qualification and bitwise checks).

* long attention (``loop_attention='long'``): a T=256 shared loop admits its attention group under the long rule,
  saves O and lse (never the probabilities) on the tape and installs plong forward/backward phases. The
  integrated default (``'auto'``) selects the same admitted group; explicit ``'off'`` preserves primitives.
"""
from __future__ import annotations

import traceback


def _capture_small(loop_attention, *, sequence=256, width=128, heads=2, layers=1):
    import torch
    from dataclasses import replace
    from megakernel_lab.fast_examples.recovered.transformer import RECIPES, initial_state, make_loop_functions, parameters, static_inputs
    from .fixtures.stage3_training import OPTIMIZER
    from megakernel_lab.operation_compiler.loop import capture_loop_training
    torch.set_num_threads(1)
    model = replace(RECIPES['primary'], sequence=sequence, width=width, heads=heads, layers=layers,
                    matrix_dtype='bfloat16')
    p = parameters(model)
    m = {k: torch.zeros_like(v) for k, v in p.items()}
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, 256, (model.batch, model.sequence), generator=g)
    inputs = (tokens, tokens.clone(), *static_inputs(model))
    return capture_loop_training(*make_loop_functions(model, intermediate=True), p, m, inputs,
                                 torch.tensor(1, dtype=torch.int64), max_steps=3, state=initial_state(),
                                 config=OPTIMIZER, storage_mode='save_body_tape', loop_attention=loop_attention,
                                 cast_reuse=True)


def test_long_attention_structure():
    from megakernel_lab.operation_compiler.lower import lower
    long = _capture_small('long')
    tape = long.ir.metadata['attention_tape']
    groups = tape['groups']
    assert groups and all(g.get('kernel') == 'long' and g['sequence'] == 256 for g in groups), tape
    names = [d['name'] if isinstance(d, dict) and 'name' in d else d for d in long.ir.metadata.get('body_tape', [])]
    program = lower(long.ir, loop_options={'group_rules': 'rot+defer+jointbig'})
    source = program.source
    assert source.count('plong::forward_phase') >= 1 and source.count('plong::backward_phase') >= 1, 'phases'
    auto = _capture_small('auto')
    assert auto.ir.metadata['attention_tape']['groups'] == groups, 'automatic admission differs from explicit long'
    automatic = lower(auto.ir, loop_options={'group_rules': 'rot+defer+jointbig'})
    assert automatic.source == program.source, 'automatic policy changed the admitted long implementation'
    off = _capture_small('off')
    refusals = off.ir.metadata['attention_tape']['refusals']
    assert not off.ir.metadata['attention_tape']['groups'] and refusals, 'explicit off must preserve primitives'
    base = lower(off.ir, loop_options={'group_rules': 'rot+defer+jointbig'})
    assert 'plong::' not in base.source
    size = {'float32':4, 'bfloat16':2, 'int64':8, 'bool':1}
    tape_bytes = lambda ir: sum(size[d['dtype']] * __import__('math').prod(d['shape']) for d in ir.metadata.get('body_tape', [])
                                if isinstance(d, dict) and 'shape' in d)
    return {'groups': [g['label'] for g in groups], 'off_refusal': refusals[0], 'auto_equals_explicit_source':True,
            'tape_bytes_long': tape_bytes(long.ir), 'tape_bytes_auto': tape_bytes(auto.ir), 'tape_bytes_off':tape_bytes(off.ir),
            'arena_long': program.manifest.get('arena_bytes'), 'arena_auto': automatic.manifest.get('arena_bytes'),
            'arena_off':base.manifest.get('arena_bytes'),
            'tape_entries': len(names)}


FINAL1 = ('rot+defer+jointbig+place+chains+softmax+wrow+alap+part+wsoft+ilp+deepred+rngel+vfill+vec4+wide+idx32'
          '+idxid+rsink+epiearly+epipair+epismh+vec4u+lpt+lptl+zmerge+msum')


def test_p5c_loop_structure():
    """p5c in staged loop regions: admitted products use the wgmma engine with region-resolved maps; without the
    feature the program is byte-identical to the base rules."""
    from megakernel_lab.operation_compiler.lower import lower
    cap = _capture_small('long', sequence=512, width=512, heads=8, layers=1)
    base = lower(cap.ir, loop_options={'group_rules': FINAL1})
    again = lower(cap.ir, loop_options={'group_rules': FINAL1})
    assert base.source == again.source
    p5c = lower(cap.ir, loop_options={'group_rules': FINAL1 + '+p5c'})
    src = p5c.source
    assert 'p5c::phase<' in src and 'P5C_TMAP' not in src and 'p5c_pick_map(' in src, 'engine/maps'
    maps = p5c.manifest['tensor_maps']
    assert maps and p5c.manifest.get('tensor_map_addresses')
    engines = {}
    for region in p5c.manifest['regions'].values():
        for d in region.get('matrix_decisions', []):
            engines[d['reason'].split(';')[0]] = engines.get(d['reason'].split(';')[0], 0) + 1
    converted = sum('FP32-converting' in d['reason'] for r in p5c.manifest['regions'].values()
                    for d in r.get('matrix_decisions', []))
    epi = lower(cap.ir, loop_options={'group_rules': FINAL1 + '+p5c+p5cw+p5ce'})
    assert 'p5c::phase_epi<' in epi.source and 'P5C_TMAP' not in epi.source, 'p5c epilogues'
    fused = [r for r in epi.manifest['regions'].values() for r in [r.get('group_rules', {}).get('p5c_epilogues')] if r]
    return {'maps': len(maps), 'engines': engines, 'converted': converted, 'p5c_epilogue_regions': len(fused),
            'epi_sites': epi.source.count('p5c::phase_epi<'),
            'arena_base': base.manifest['arena_bytes'], 'arena_p5c': p5c.manifest['arena_bytes'],
            'shared_bytes': p5c.manifest['shared_bytes'], 'p5c_sites': src.count('p5c::phase<')}


def run():
    results, passed = {}, True
    for name, fn in [('long_attention_structure', test_long_attention_structure),
                     ('p5c_loop_structure', test_p5c_loop_structure)]:
        try:
            results[name] = {'passed': True, **fn()}
        except Exception:
            passed = False
            results[name] = {'passed': False, 'error': traceback.format_exc()}
    return {'passed': passed, 'results': results}
