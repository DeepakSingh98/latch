"""Small CPU structural contracts for the integrated frozen schedule protocols.

These are code-generation/admission tests. They make no numerical or timing claim.
"""
from __future__ import annotations

import traceback


def _graph():
    return {"inputs": ["x"], "outputs": ["z"], "values": {
        "x": {"shape": [64], "strides": [1], "dtype": "float32"},
        "z": {"shape": [64], "strides": [1], "dtype": "float32"}},
        "operations": [{"name": "z", "target": "aten.mul.Tensor",
                        "args": [{"ref": "x"}, 2.0], "kwargs": {}}]}


def ordinary_graph_uses_generic_schedule():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    auto = lower_fused(_graph())
    reference = lower_fused(_graph(), schedule="reference")
    assert auto.source == reference.source
    assert auto.manifest["performance_schedule"]["selected"] == "reference"
    return {"source_hash": auto.manifest["source_hash"]}


def every_protocol_lowers_common_ir():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    results = {}
    for schedule in ("reference", "p6d", "p6e", "p7b", "p7a"):
        p = lower_fused(_graph(), schedule=schedule)
        assert p.manifest["inputs"] == ["x"] and p.manifest["outputs"] == ["z"]
        assert p.manifest["performance_schedule"]["selected"] == schedule
        assert "@@LOAD" not in p.source
        if schedule in ("p6d", "p7a"):
            assert p.manifest["full_grid"] is True
        results[schedule] = {"source_hash": p.manifest["source_hash"], "shared_bytes": p.manifest["shared_bytes"]}
    return results


def configuration_is_scoped_on_error():
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK, _configuration
    from megakernel_lab.operation_compiler import matrix_rules_p7b as rules
    saved = rules.SHARED_LIMIT
    try:
        with _LOCK, _configuration("p7b", {"SHARED_LIMIT": 1024}):
            assert rules.SHARED_LIMIT == 1024
            raise RuntimeError("intentional scope exit")
    except RuntimeError:
        pass
    assert rules.SHARED_LIMIT == saved
    return {"restored_shared_limit": saved}


def parallel_builds_do_not_leak_settings():
    from concurrent.futures import ThreadPoolExecutor
    from megakernel_lab.operation_compiler.fusion import lower_fused
    def lower(schedule):
        p = lower_fused(_graph(), schedule=schedule)
        return p.manifest["source_hash"]
    expected = {s: lower(s) for s in ("reference", "p6e", "p7b", "p7a")}
    choices = ("p7a", "p6e", "p7b", "reference", "p7a", "p7b")
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(lower, choices))
    assert actual == [expected[s] for s in choices]
    return {"builds": len(choices), "stable": True}


def tensor_map_schemas_bind_identically():
    from megakernel_lab.operation_compiler.p5c_matrix import normalize_tensor_maps
    common = {"dims": [64, 64], "stride_bytes": 128, "box": [64, 64], "dtype": "bfloat16"}
    manifest = {"inputs": ["p"], "offsets": {"buf": 192},
                "layouts": {"view": {"root": "buf", "offset": 8}},
                "tensor_maps": [dict(common, root="view", offset=16),
                                dict(common, root="p", offset=0)]}
    actual = normalize_tensor_maps(manifest)
    assert actual[0]["storage"] == "arena" and actual[0]["word_offset"] == 192
    assert actual[0]["element_offset"] == 24
    assert actual[1]["storage"] == "input" and actual[1]["index"] == 0
    manifest["tensor_maps"] = actual
    assert normalize_tensor_maps(manifest) == actual
    return {"schemas": ["flat_root_offset", "loop_explicit_storage"], "idempotent": True}


def compact_generation_has_explicit_refusal():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    p = lower_fused(_graph(), schedule="p6d", compact_codegen="fold")
    assert "p7e" in p.manifest
    assert p.manifest["p7e"]["mode"] == "fold"
    return {"record": p.manifest["p7e"]}


def shared_reduction_interfaces_remain_distinct():
    from megakernel_lab.operation_compiler.codegen_rules import exact_reduction
    common = dict(out="out", position="(r / 1LL) * 1024LL + o", read=lambda i: f"input[{i}]",
                  rounded=lambda v: v, results=1024)
    wide, wide_meta = exact_reduction(**common, extent=512, wide=True)
    column, col_meta = exact_reduction(**common, extent=64, coalesced_threads=True)
    quad, quad_meta = exact_reduction(**common, extent=64, coalesced_threads=True, quad_threads=True)
    assert wide_meta["folded_passes"] == 2 and "const long long rr" in wide
    assert col_meta["engine"] == "exact_cta_tree_per_thread_column" and "latch_t" not in column
    assert quad_meta["engine"] == "exact_cta_tree_per_thread_column" and "latch_t >> 2" in quad
    return {"interfaces": ["C wide", "B coalesced", "B quad"], "structural_only": True}


def mixed_schedule_uses_resource_proof():
    from unittest.mock import patch
    from megakernel_lab.operation_compiler.performance_schedule import select_schedule
    from megakernel_lab.operation_compiler import mixed_attention
    results = {}
    # Mock only the already-matched layout plan. Real resource formulas and
    # scoped precision/resource settings choose the execution protocol.
    for head, context, expected in ((32, 49, "p7b"), (64, 49, "p7b"), (64, 65, "p7a")):
        plans = {"attention": {"hd": head, "t": context}}
        with patch.object(mixed_attention, "admit", return_value=(plans, [])):
            selected, reason = select_schedule(_graph())
        assert selected == expected, (head, context, selected, reason)
        assert mixed_attention.BWD_FIT_LIMIT is None
        results[f"head{head}_context{context}"] = selected
    return {"protocols": results, "structural_only": True}


def recomputed_chain_sinks_after_deferred_product():
    from megakernel_lab.operation_compiler.lower import TensorLayout, _barrier_intervals, _defer_accumulate, _sink_recomputed
    def phase(name, reads, kind='elementwise'):
        return {'kind': kind, 'names': [name], 'reads': reads, 'writes': [name],
                'code': '/*B_VBASE*/' if kind == 'gemm' else '', 'tiles': 1}
    phases = [phase('r', ['input']), phase('r2', ['r']), phase('u', ['input']),
              phase('v', ['u']), phase('z', ['v']), phase('gradient', ['r2_view'], 'gemm'),
              phase('consumer', ['r2_view', 'z'])]
    layouts = {name: TensorLayout((64,), (1,), name) for name in
               ('input', 'r', 'r2', 'u', 'v', 'z', 'gradient', 'consumer')}
    layouts['r2_view'] = TensorLayout((64,), (1,), 'r2')
    initial = _barrier_intervals(phases, layouts, True)
    deferred, defer_records = _defer_accumulate(phases, initial, layouts, {'gradient'}, grid=128)
    sunk, sink_records = _sink_recomputed(phases, deferred, layouts, {'r', 'r2'})
    assert defer_records and sink_records
    before = {p: level for level, members in enumerate(deferred) for p in members}
    after = {p: level for level, members in enumerate(sunk) for p in members}
    assert after[0] > before[0] and after[1] > before[1]
    assert after[0] < after[1] < min(after[5], after[6])
    assert after[5] == before[5] and after[6] == before[6]
    # Check the resulting schedule independently through all root-level RAW
    # edges, including the alias read; each producer must precede every reader.
    writer = {layouts[name].root: i for i, p in enumerate(phases) for name in p['writes']}
    for i, p in enumerate(phases):
        for name in p['reads']:
            producer = writer.get(layouts[name].root)
            if producer is not None and producer != i:
                assert after[producer] < after[i], (producer, i, sunk)
    return {'initial': initial, 'after_defer': deferred, 'after_sink': sunk,
            'deferred': defer_records, 'sunk': sink_records, 'alias_dependencies_checked': True}


def captured_bf16_conditional_keeps_region_bindings_local():
    import torch
    import torch.nn.functional as F
    from megakernel_lab.operation_compiler.capture import capture_conditional_training
    from megakernel_lab.operation_compiler.lower import LoweringError, _LOWERING_DEPTH, lower
    from megakernel_lab.operation_compiler.performance_schedule import select_schedule
    batch, tokens, heads, width = 2, 65, 2, 64
    channels = heads * width
    gen = torch.Generator().manual_seed(17031)
    params = {'qkv': torch.randn(3 * channels, channels, generator=gen) * .01,
              'bias': torch.randn(3 * channels, generator=gen) * .01,
              'out': torch.randn(channels, channels, generator=gen) * .01}
    inputs = (torch.randn(batch, tokens, channels, generator=gen),)

    def loss(p, x):
        qkv = F.linear(x, p['qkv'], p['bias'])
        q, k, v = qkv.reshape(batch, tokens, 3, heads, width).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v)
        output = F.linear(attended.transpose(1, 2).reshape(batch, tokens, channels), p['out'])
        return output.square().mean()

    capture = capture_conditional_training(loss, loss, params, None, inputs, torch.tensor(True),
                                           precision='mixed_bf16')
    # This is an actually admitted captured attention graph, so an accidental
    # return to automatic flat selection would exercise the unsafe private ABI.
    standalone, _ = select_schedule(capture.ir.true_branch)
    assert standalone == 'p7a', standalone
    program = lower(capture.ir)
    assert _LOWERING_DEPTH.get() == 0
    for branch in program.manifest['branches'].values():
        assert branch['performance_schedule']['selected'] == 'reference'
        assert not branch.get('tensor_maps') and not branch.get('p7_const_args')
        assert any('not composed' in reason for reason in branch['compiler_selection']['reasons'])
    assert not program.manifest.get('tensor_maps') and not program.manifest.get('p7_const_args')
    assert 'POPS4B_TMAP(' not in program.source and 'P5C_TMAP(' not in program.source
    assert 'latch_cargs' not in program.source
    try:
        lower(capture.ir, schedule='p7a')
    except LoweringError as error:
        assert 'standalone region' in str(error)
    else:
        raise AssertionError('unsafe explicit composed schedule was accepted')
    assert _LOWERING_DEPTH.get() == 0
    assert lower(capture.ir).source == program.source
    return {'standalone_selection': standalone, 'branch_selection': 'reference',
            'branch_count': 2, 'private_flat_bindings': False,
            'explicit_override_refused': True, 'depth_restored_on_error': True,
            'structural_only': True}


def typed_arange_rules_coexist_in_b_schedule():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    graph = {'inputs': [], 'outputs': ['float_range', 'int_range'], 'values': {
        'float_range': {'shape': [16], 'strides': [1], 'dtype': 'float32'},
        'int_range': {'shape': [16], 'strides': [1], 'dtype': 'int64'}}, 'operations': [
        {'name': 'float_range', 'target': 'aten.arange.start', 'args': [0, 16],
         'kwargs': {'dtype': 'float32'}},
        {'name': 'int_range', 'target': 'aten.arange.start', 'args': [0, 16],
         'kwargs': {'dtype': 'int64'}}]}
    program = lower_fused(graph, schedule='p7b')
    assert program.manifest['performance_schedule']['selected'] == 'p7b'
    assert 'static_cast<float>((double)0.0 + (double)1.0 * (double)' in program.source
    assert program.manifest['layouts']['int_range']['dtype'] == 'int64'
    assert program.manifest['outputs'] == ['float_range', 'int_range']
    return {'schedule': 'p7b', 'dtypes': ['float32', 'int64'], 'structural_only': True}


def compact_long_attention_guard_is_mandatory_and_scoped():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    from megakernel_lab.operation_compiler import p7e_codegen
    saved = p7e_codegen.P7E_NOSHARE
    program = lower_fused(_graph(), schedule='p6d', compact_codegen='fold',
        schedule_options={'matrix_rule_overrides': {'p7e_codegen.P7E_NOSHARE': []}})
    assert program.manifest['p7e']['mode'] == 'fold'
    assert program.manifest['p7e']['noshare'] == ['pflatlong::']
    assert program.manifest['p7e']['integration_safety']['rejected_plain_fold'] is True
    assert p7e_codegen.P7E_NOSHARE == saved
    ordinary = lower_fused(_graph(), schedule='p6d')
    assert 'p7e' not in ordinary.manifest
    return {'mandatory_unshared_marker': 'pflatlong::', 'compact_default': 'off',
            'globals_restored': True, 'structural_only': True}


def pure_bf16_matrix_graph_reaches_admitted_engine():
    from megakernel_lab.operation_compiler.lower import lower
    from .test_p5c import ffn_group
    from megakernel_lab.operation_compiler.performance_schedule import select_schedule
    graph = ffn_group()
    selected, reasons = select_schedule(graph)
    assert selected == 'p6d', reasons
    program = lower(graph)
    assert program.manifest['performance_schedule']['selected'] == 'p6d'
    assert program.manifest.get('tensor_maps') and 'p5c::' in program.source
    graph['operations'].append({'name': 'softmax_marker', 'target': 'aten._softmax.default',
                                'args': [{'ref': 'z32'}, -1, False], 'kwargs': {}})
    graph['values']['softmax_marker'] = dict(graph['values']['z32'], name='softmax_marker')
    assert select_schedule(graph)[0] == 'reference'
    return {'matrix_selection': selected, 'unmatched_attention_fallback': 'reference',
            'source_hash': program.cache_key, 'structural_only': True}


def run():
    rows = {}
    tests = (ordinary_graph_uses_generic_schedule, every_protocol_lowers_common_ir,
             configuration_is_scoped_on_error, parallel_builds_do_not_leak_settings,
             tensor_map_schemas_bind_identically, compact_generation_has_explicit_refusal,
             shared_reduction_interfaces_remain_distinct, mixed_schedule_uses_resource_proof,
             recomputed_chain_sinks_after_deferred_product,
             captured_bf16_conditional_keeps_region_bindings_local,
             typed_arange_rules_coexist_in_b_schedule,
             compact_long_attention_guard_is_mandatory_and_scoped,
             pure_bf16_matrix_graph_reaches_admitted_engine)
    for test in tests:
        try:
            rows[test.__name__] = {"passed": True, **test()}
        except Exception:
            rows[test.__name__] = {"passed": False, "error": traceback.format_exc()}
    return {"passed": all(row["passed"] for row in rows.values()), "tests": rows}
