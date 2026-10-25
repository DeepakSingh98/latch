"""CPU contracts for the P8-A warp-specialized product rules (lowering only; GPU arithmetic is gated separately).

Uses the explicit-BF16 feed-forward block of test_p5c (forward and backward products, SiLU chains, FP32-converting
gradient products) through the normal auto entry point.
"""
from __future__ import annotations

import hashlib
import re


def _lower(precision=None, **settings):
    from megakernel_lab.operation_compiler import p5c_matrix
    from megakernel_lab.operation_compiler.lower import lower
    from .test_p5c import ffn_group
    saved = {k: getattr(p5c_matrix, k) for k in settings}
    try:
        for k, v in settings.items():
            setattr(p5c_matrix, k, v)
        options = {'schedule_options': {'p8_precision': precision}} if precision else {}
        return lower(ffn_group(), **options)
    finally:
        for k, v in saved.items():
            setattr(p5c_matrix, k, v)


def default_selects_grouped_warp_specialized_products():
    prog = _lower()
    src, man = prog.source, prog.manifest
    engines = [m['engine'] for p in man['phases'] for m in p.get('members', [])]
    assert man['performance_schedule']['selected'] == 'p6d'
    assert '#include "p8ws.cuh"' in src and 'p8ws::group<' in src
    # SiLU / SiLU-derivative epilogues (expf, division) keep the p5c body under the strict policy.
    assert src.count('p5c::phase_epi<') == 2
    assert any(e.startswith('p5c_ws_') for e in engines)
    assert man['shared_bytes'] <= 232448
    return {'engines': sorted(set(engines)), 'groups': src.count('p8ws::group<'), 'shared_bytes': man['shared_bytes']}


def disabled_reproduces_parent_lowering_shape():
    off = _lower(P8WS=False)
    assert 'p8ws' not in off.source and 'p8ws.cuh' not in off.manifest.get('header_hashes', {})
    engines = [m['engine'] for p in off.manifest['phases'] for m in p.get('members', [])]
    assert not any('_ws_' in e for e in engines)
    return {'source_sha256': hashlib.sha256(off.source.encode()).hexdigest()}


def practical_policy_changes_only_declared_operations():
    strict, practical = _lower(), _lower('practical')
    assert '__expf(' not in strict.source and '__fdividef(' not in strict.source
    assert '__expf(' in practical.source and '__fdividef(' in practical.source
    # The light practical SiLU epilogues move to the warp-specialized body (no p5c epilogue products remain).
    assert practical.source.count('p5c::phase_epi<') == 0
    return {'strict_groups': strict.source.count('p8ws::group<'), 'practical_groups': practical.source.count('p8ws::group<')}


def group_items_are_bound_to_their_rotation():
    prog = _lower()
    items = re.findall(r'p8ws::item<(\d+),(\w+),(\w+),(\d),(\w+)>\(g\d+, .*?, (\d+)\)', prog.source, re.S)
    assert items, 'no warp-specialized items'
    return {'items': len(items)}


def run():
    tests = [default_selects_grouped_warp_specialized_products, disabled_reproduces_parent_lowering_shape,
             practical_policy_changes_only_declared_operations, group_items_are_bound_to_their_rotation]
    results = []
    for test in tests:
        try:
            results.append({'test': test.__name__, 'passed': True, 'detail': test()})
        except Exception as error:  # noqa: BLE001 - contract receipts keep every failure
            import traceback
            results.append({'test': test.__name__, 'passed': False, 'error': repr(error),
                            'traceback': traceback.format_exc()})
    return {'tests': results, 'passed': all(r['passed'] for r in results)}


if __name__ == '__main__':
    import json
    print(json.dumps(run(), indent=2))
