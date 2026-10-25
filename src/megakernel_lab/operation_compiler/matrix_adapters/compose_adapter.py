"""Thin adapter from LATCH-MATRIX candidates to the LATCH-COMPOSE execution contract v1.

LATCH-COMPOSE (research/latch_compose/API.md at ac3bf4710, implementation 712b4cb3a) plans opaque bodies from
phase dictionaries (``ops.execution_plan.contracts``): complete reads/writes, engine, shared bytes, the common
256-thread ABI, work items and grid scope. ``body_phase`` turns one selected candidate plus its emission into that
dictionary; ``MatrixProvider`` installs the same records inside ``lower(..., execution="compose")`` through the
opt-in hook in ``_lower_serial`` (top-level, unstaged composed bodies only).

Nothing here schedules: dependencies, intervals, arena lifetimes and publication stay with LATCH-COMPOSE.
"""
from __future__ import annotations

from .candidates import Candidate


def body_phase(candidate: Candidate, emission, *, a, b, out, target='aten.mm.default'):
    """Phase dictionary accepted by ``ops.execution_plan.contracts`` for one emitted product.

    ``a``/``b``/``out`` are value names with layouts in the planner's layout table (views resolve to roots there).
    """
    req = candidate.requirements
    prec = req.precision
    phase = {
        'kind': 'gemm', 'names': [out], 'targets': [target], 'reads': [a, b], 'writes': [out],
        'code': emission.code, 'engine': f'latch_matrix:{candidate.body}',
        'shared_bytes': int(req.dynamic_shared_bytes), 'threads': req.sync.threads_per_cta,
        'tiles': emission.tiles, 'work_items': emission.tiles,
        'execution_grid_scope': 'physical_cooperative_grid',
        'arithmetic': f'{prec.instruction}; {prec.order}; {prec.rounding}',
        'internal_mutations': [], 'counter_effects': [], 'private_scratch': [],
        'latch_matrix': {'body': candidate.body, 'bitwise_class': prec.bitwise_class, 'arch': req.arch,
                         'host_setup': list(req.host_setup), 'entry_resident': req.sync.entry_resident,
                         'requires_full_grid': req.sync.requires_full_grid, 'cta_subset': req.sync.cta_subset,
                         'plan': {k: v for k, v in candidate.plan.items() if isinstance(v, (int, float, str, bool))}},
    }
    if req.sync.requires_full_grid:
        phase['requires_full_grid'] = True
    return phase
