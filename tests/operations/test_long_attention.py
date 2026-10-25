"""CPU contracts for the integrated flat long-context attention rule (no CUDA required).

Numerics are qualified on GPU (FP64 region references, complete-update gates, training); these checks pin admission
and the generated structure: the BF16-boundary region becomes exactly prepare/forward/dQ/dKdV phases with absorbed
layouts and no T x T storage, each explicit recipe selects its pieces and saved output, FP32 operands, unsupported
head widths and contexts and forward-only graphs keep the existing lowering with a recorded reason, and
pops4d=False preserves the eligible fallback. Singleton axes use the pinned attention donor's canonical layouts;
noncontiguous BF16 result views remain refused.
"""
from __future__ import annotations

import math

import torch

from megakernel_lab.operation_compiler.lower import lower


def _config(batch=2, sequence=256, heads=2, head_width=64):
    return dict(batch=batch, sequence=sequence, heads=heads, head_width=head_width, width=heads * head_width)


def _region(cfg, *, fp32_inputs=False, forward_only=False, noncausal=False):
    from .fixtures import pops4d_region as R
    from torch.fx.experimental.proxy_tensor import make_fx
    from megakernel_lab.operation_compiler.capture import capture_decompositions
    args = list(R.example_inputs(cfg))
    if fp32_inputs:
        args[0], args[1], args[2] = (x.float() for x in args[:3])
    if noncausal:
        args[3] = torch.zeros_like(args[3])

    def attention(qb, kb, vb, mask):
        return R.attention_region(qb, kb, vb, mask, cfg)

    def step(qb, kb, vb, mask, g):
        if forward_only:
            return (attention(qb, kb, vb, mask),)
        out, pullback = torch.func.vjp(lambda a, b, c: attention(a, b, c, mask), qb, kb, vb)
        return (out, *pullback(g))
    gm = make_fx(step, decomposition_table=capture_decompositions(), tracing_mode='fake',
                 _allow_non_fake_inputs=True)(*args)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return R.fx_to_graph(gm)


def _kinds(program):
    return [phase['kind'] for phase in program.manifest['phases']]


def _long(program):
    return [k for k in _kinds(program) if k.startswith('attention_long_')]


def _flat_admission(graph):
    """Pin the primitive refusal even when automatic selection bypasses this protocol."""
    from megakernel_lab.operation_compiler import attention_lower, flat_long_attention
    from megakernel_lab.operation_compiler.performance_schedule import _graph_parts
    parts, layouts, protected = _graph_parts(graph)
    operations, values, names, ref, field, output_name = parts
    groups, match_refusals = attention_lower.plan(
        operations, values, names, ref=ref, field=field,
        output_name=output_name, protected=protected)
    plans, refusals = flat_long_attention.admit(groups, parts, layouts, protected)
    return groups, plans, match_refusals + refusals


def run():
    checks = {}
    cfg = _config()
    graph = _region(cfg)
    default = lower(graph, precision_policy='strict')
    t, bh = cfg['sequence'], cfg['batch'] * cfg['heads']
    checks['admitted_four_phases'] = _long(default) == ['attention_long_prepare', 'attention_long_forward',
                                                         'attention_long_dq', 'attention_long_kv']
    decisions = default.manifest.get('attention_groups') or []
    checks['decision_recorded'] = (len(decisions) == 1 and decisions[0]['decision'] == 'long_bf16_boundary_attention'
                                   and decisions[0]['recipe'] == 'exact3')
    # No T x T storage: the arena is far below one FP32 score matrix per head.
    checks['no_score_storage'] = default.manifest['arena_bytes'] < bh * t * t * 4
    checks['one_cta_resource'] = default.manifest.get('pops4d', {}).get('one_cta') is True
    sources = {}
    for recipe in ('exact3', 'split2', 'bf16'):
        program = lower(graph, pops4d_recipe=recipe,
                        precision_policy='strict' if recipe == 'exact3' else 'practical')
        sources[recipe] = program.source
        # Flat and recurrent long-attention headers now have separate namespaces.
        o32 = '#define PFLATLONG_O32 1' in program.source
        pieces = {'exact3': 3, 'split2': 2, 'bf16': 1}[recipe]
        checks[f'{recipe}_structure'] = (_long(program) == _long(default) and o32 == (recipe != 'bf16')
                                        and f'forward_item6<{pieces}, true>' in program.source
                                        and f'kv_item6<{pieces}, true>' in program.source
                                        and program.manifest.get('requires_arch') == 'sm_90a')
        v1 = lower(graph, pops4d_recipe=recipe, pops4d_version=1,
                   precision_policy='strict' if recipe == 'exact3' else 'practical')
        checks[f'{recipe}_v1_structure'] = (_long(v1) == _long(default) and f'forward_item<{pieces}>' in v1.source
                                           and f'dq_item<{pieces}>' in v1.source)
    checks['default_is_exact3'] = sources['exact3'] == default.source
    # Schedule versions: v3 (mma.sync, plain sm_90), v9 (stored dS, BF16 only; exact recipes fall back to v7).
    v3 = lower(graph, pops4d_version=3, precision_policy='strict')
    checks['v3_mma_sync'] = 'forward_item3<3>' in v3.source and v3.manifest.get('requires_arch') is None
    v9 = lower(graph, pops4d_recipe='bf16', pops4d_version=9, precision_policy='practical')
    checks['v9_stored_ds'] = (_long(v9) == ['attention_long_prepare', 'attention_long_forward', 'attention_long_delta',
                                           'attention_long_kv', 'attention_long_dq'] and 'dq_item9<1, true>' in v9.source)
    v9x = lower(graph, pops4d_version=9, precision_policy='strict')
    checks['v9_exact_falls_back'] = 'dq_item9' not in v9x.source and 'kv_item6<3, true>' in v9x.source
    # v11: single-pass backward with fixed-point dQ (BF16 only; exact recipes fall back to v7).
    v11 = lower(graph, pops4d_recipe='bf16', pops4d_version=11, precision_policy='practical')
    checks['v11_fused_dq'] = (_long(v11) == ['attention_long_prepare', 'attention_long_forward', 'attention_long_delta',
                                            'attention_long_kv', 'attention_long_dq']
                              and 'kv_item11<1, true>' in v11.source and 'dq_convert_unit11<true>' in v11.source
                              and 'delta_unit11' in v11.source and 'forward_item6<1, true>' in v11.source
                              and v11.manifest.get('requires_arch') == 'sm_90a')
    v11x = lower(graph, pops4d_version=11, precision_policy='strict')
    checks['v11_exact_falls_back'] = 'kv_item11' not in v11x.source and 'kv_item6<3, true>' in v11x.source
    # v12: 128-key forward/dQ steps, v7 dK/dV (BF16 only; exact recipes fall back to v7).
    v12 = lower(graph, pops4d_recipe='bf16', pops4d_version=12, precision_policy='practical')
    checks['v12_wide_steps'] = (_long(v12) == _long(default) and 'forward_item12<1, true>' in v12.source
                                and 'dq_item12<1, true>' in v12.source and 'kv_item6<1, true>' in v12.source
                                and v12.manifest.get('requires_arch') == 'sm_90a')
    v12x = lower(graph, pops4d_version=12, precision_policy='strict')
    checks['v12_exact_falls_back'] = 'forward_item12' not in v12x.source and 'kv_item6<3, true>' in v12x.source
    checks['recipes_differ'] = len(set(sources.values())) == 3
    base = lower(graph, pops4d=False, precision_policy='strict')
    checks['disabled_keeps_integrated'] = (not _long(base)
        and 'attention_long_flat.cuh' not in base.manifest['header_hashes'])
    # Layout absorption: no elementwise copy/convert phases remain around the attention.
    checks['layouts_absorbed'] = all(k.startswith('attention_long_') for k in _kinds(default)
                                     if k not in ('elementwise',)) and _kinds(default).count('elementwise') == 0
    # Other context lengths and head counts.
    for name, other in (('c512_h4', _config(batch=2, sequence=512, heads=4)),
                        ('c1024_h2', _config(batch=2, sequence=1024, heads=2)),
                        ('c1024_h1', _config(batch=2, sequence=1024, heads=1))):
        other_graph = _region(other)
        program = lower(other_graph, precision_policy='strict')
        groups, plans, _ = _flat_admission(other_graph)
        checks[f'{name}_admitted'] = len(_long(program)) == 4 and len(groups) == len(plans) == 1
    noncausal = lower(_region(cfg, noncausal=True), precision_policy='strict')
    checks['zero_mask_admitted'] = len(_long(noncausal)) == 4
    # The private rule records the precise refusal. Automatic selection can choose
    # another admitted protocol before that private rule is invoked; its source
    # must then equal the source with this flat rule explicitly disabled.
    refusals = {
        'fp32_operands': (_region(cfg, fp32_inputs=True), 'single BF16 -> FP32 widening'),
        'head_width_32': (_region(_config(heads=4, head_width=32)), 'head width 32 != 64'),
        'context_192': (_region(_config(sequence=192)), 'context 192 outside'),
        'context_128': (_region(_config(sequence=128)), 'context 128 outside'),
        'forward_only': (_region(cfg, forward_only=True), 'forward-only'),
        # batch 1: PyTorch returns dK^T through views as a column-major BF16 cast (no materializing copy).
        'batch1_view_output': (_region(_config(batch=1)),
                               'BF16 result is not contiguous'),
    }
    refusal_details = {}
    for name, (other, expected_reason) in refusals.items():
        groups, plans, refused = _flat_admission(other)
        program = lower(other, precision_policy='strict')
        disabled = lower(other, pops4d=False, precision_policy='strict')
        reasons = [r.get('reason', '') for r in refused]
        reason_matches = any(expected_reason in reason for reason in reasons)
        if name == 'forward_only' and not groups:
            reason_matches = True  # No complete training-attention group exists.
        checks[f'refused_{name}'] = (not plans and not _long(program) and reason_matches
                                    and program.source == disabled.source)
        refusal_details[name] = {'reasons': reasons, 'matched_training_groups': len(groups),
                                 'selected': program.manifest.get('performance_schedule', {}).get('selected'),
                                 'disabled_source_equal': program.source == disabled.source}
    return {'checks': checks, 'refusal_details': refusal_details, 'passed': all(checks.values()),
            'scope': 'CPU admission/code-generation contracts; numerical qualification remains separate'}


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=1))
