# P-OPS2-A adaptation from 4dcee751fab59f5ecea67ebada3167ca93fba841.
"""P-OPS2-A FP32 product engine and shape/resource selection for the fused operation schedule.

Every admitted strict-FP32 product (P-OPS-B layout contract: unit-stride m or k for A, unit-stride k
or n for B, 16-byte aligned leading dimensions, two-factor views) is planned by choosing one body,
tile, k-stage depth/pipeline and split-K count from a fixed candidate set by a fitted H100 cost
estimate. Inputs are operation properties only: M, N, K, batch, operand unit strides/alignment and
the resident cooperative grid. No model, layer or value names are inspected.

Bodies (all one fmaf chain per output element and k slice, ascending k, from +0; no TF32, tensor
cores or floating atomics; split slices summed in fixed order 0..KSPLIT-1):
  * popsb  - P-OPS-B cp.async engine (kernels/fp32_matrix.cuh), the integrated base's body;
  * simt   - kernels/fp32_simt.cuh: register-prefetched k-major staging, 4x8-lane warps;
  * sliceS - kernels/fp32_slice.cuh: 8x8 accumulators per thread, S in-CTA k slices reduced on chip
             in fixed slice order (a fixed-order split of every stage), multi-stage cp.async ring.

Cost model: a phase's time is the maximum resident-slot load under whole-grid vbase rotation; a task
costs a + b * (k-stages in its slice) + c * (BM*BN/4096) * (KSPLIT-1)/KSPLIT microseconds. (a, b, c)
were least-squares fitted per configuration (mean |error| 2-8%) to in-kernel repeated product phases
measured on H100 at one CTA/SM and a 128-CTA cooperative grid (P-OPS2-A attempt micro-09, ten
product shapes, all split counts), i.e. the integrated program's resource setting.

Split-K scratch is not a graph value: products use placeholders /*P2_CNT*/ (arrival counters) and
/*P2_PART*/ (partials) that the emitter binds after scheduling to a program-wide region beyond the
planned arena. Counters are zeroed once in the kernel prologue and every last arriver restores zero,
so no per-product initialization phase exists; partial offsets restart in every phase.
"""
from __future__ import annotations

import math

ENGINE = 'pops2_fp32'
HEADERS = {'simt': 'fp32_simt.cuh', 'slice': 'fp32_slice.cuh', 'popsb': 'fp32_matrix.cuh'}
POPSB_HELPERS = """#ifndef LATCH_OPS_POPSB_IMPORTED
#define LATCH_OPS_POPSB_IMPORTED
#define POPS_INLINE
#define POPS_RELACQ
#include "fp32_matrix.cuh"
#endif
"""
# (body, BM, BN, BK, pipeline stages, in-CTA slices): (a, b, c) microseconds.
FITTED = {
    ('popsb', 64, 64, 16, 2, 1): (1.00, 0.70, 3.0),
    ('popsb', 128, 64, 16, 2, 1): (1.00, 1.05, 3.0),
    ('popsb', 64, 128, 16, 2, 1): (2.75, 1.05, 2.0),
    ('popsb', 32, 64, 16, 2, 1): (1.00, 0.50, 3.0),
    ('simt', 64, 64, 16, 2, 1): (1.25, 0.55, 3.0),
    ('simt', 128, 64, 16, 2, 1): (2.00, 0.95, 2.0),
    ('simt', 128, 128, 8, 2, 1): (3.75, 0.80, 2.0),
    ('slice', 64, 64, 16, 4, 4): (0.00, 0.75, 3.0),
    ('slice', 64, 64, 32, 4, 4): (0.00, 1.25, 3.0),
    ('slice', 128, 64, 16, 4, 2): (0.75, 1.20, 2.0),
    ('slice', 64, 128, 16, 4, 2): (1.00, 1.10, 2.0),
    ('slice', 128, 64, 32, 4, 2): (0.00, 2.25, 3.0),
    ('slice', 64, 128, 32, 4, 2): (0.00, 2.05, 3.0),
    ('slice', 32, 64, 32, 4, 8): (0.25, 0.75, 4.0),
}
# Two resident CTAs per SM (264 slots; __launch_bounds__(256, 2) register cap): fitted the same way to
# in-kernel repeated phases at grid 264 (attempt micro-04). Tiles with poor fits (128x128) are excluded.
FITTED_2CTA = {
    ('popsb', 64, 64, 16, 2, 1): (1.25, 1.10, 4.0),
    ('popsb', 32, 64, 16, 2, 1): (1.00, 0.80, 4.0),
    ('simt', 64, 64, 16, 2, 1): (1.00, 1.05, 4.0),
    ('simt', 64, 64, 8, 2, 1): (1.25, 0.55, 4.0),
    ('simt', 128, 64, 16, 2, 1): (1.50, 1.65, 4.0),
    ('simt', 128, 64, 8, 2, 1): (0.50, 0.95, 4.0),
    ('simt', 32, 64, 16, 2, 1): (0.50, 0.85, 4.0),
    ('simt', 32, 64, 8, 2, 1): (0.25, 0.50, 4.0),
    ('simt', 64, 128, 8, 2, 1): (2.50, 0.95, 3.0),
}
SPLITS = (1, 2, 4, 8)
# LATCH-MOESUP: deeper ordered split-K for underfilled long-K products (e.g. [384, 32] router weight gradients
# over K = 65536: 6 output tiles), and split-K for the transposed C^T = B^T A^T form, whose split partials are
# summed in slice order before the same strided store. Selection stays with the fitted cost model.
# MOESUP_SPLITS = False reproduces the previous selection.
MOESUP_SPLITS = True
EXTRA_SPLITS = (16, 32)
FIT_GRID = 128       # grid of the fitting measurements
SELECTION_GRID = 132  # H100 SMs: the runtime launches programs with these products at one CTA per SM


def _shared_bytes(body, bm, bn, bk, nst, slices, ak, bkc):
    if body == 'popsb':
        sa = bk + 4 if ak else bm + 4
        sb = bk + 4 if bkc else bn + 4
        return 4 * nst * ((bm * sa if ak else bk * sa) + (bn * sb if bkc else bk * sb))
    if body == 'simt':
        return 4 * 2 * (bk * (bm + (4 if ak else 0)) + bk * (bn + (4 if bkc else 0)))
    return 4 * max(nst * bk * ((bm + 4) + (bn + 4)), slices * bm * bn)


def task_us(config, steps, ksplit, bm, bn, table=None):
    a, b, c = (table or FITTED)[config]
    return a + b * steps + (c * bm * bn / 4096 * (ksplit - 1) / ksplit if ksplit > 1 else 0.0)


def makespan(rows, grid):
    """Max slot load for products sharing a phase with cumulative whole-grid rotation."""
    load, vbase = [0.0] * grid, 0
    for row in rows:
        full, extra = divmod(row['tasks'], grid)
        if full:
            load = [x + full * row['task_us'] for x in load]
        for i in range(extra):
            load[(vbase + i) % grid] += row['task_us']
        vbase += row['tasks']
    return max(load)


def candidates(flags, m, n, k, batch, grid, configs=None, table=None):
    rows = []
    kin = math.gcd(flags['kia'], flags['kib'])
    for config in (configs or table or FITTED):
        body, bm, bn, bk, nst, slices = config
        if m % bm or flags['mia'] % bm or n % bn or flags['nib'] % bn or kin % bk:
            continue
        if body == 'slice' and ((bm * bk // 4) % 256 or (bn * bk // 4) % 256 or bk % slices):
            continue
        if body == 'popsb' and bn not in (64, 128):
            continue
        for ks in SPLITS + (EXTRA_SPLITS if MOESUP_SPLITS else ()):
            if k % (bk * ks) or k // (bk * ks) < 2:
                continue
            steps = k // (bk * ks)
            tiles = batch * (m // bm) * (n // bn)
            row = {'body': body, 'bm': bm, 'bn': bn, 'bk': bk, 'nst': nst, 'slices': slices, 'ksplit': ks,
                   'steps': steps, 'tiles': tiles, 'tasks': tiles * ks,
                   'task_us': task_us(config, steps, ks, bm, bn, table),
                   'shared_bytes': _shared_bytes(body, bm, bn, bk, nst, slices, flags['ak'], flags['bkc'])}
            row['estimate_us'] = round(makespan([row], grid), 4)
            rows.append(row)
    # Deterministic order: modeled time, then fewer partial bytes, then larger tiles.
    rows.sort(key=lambda r: (r['estimate_us'], r['ksplit'], -r['bm'] * r['bn'], r['body'], r['bk']))
    return rows


def layout_struct(spec):
    from .pops_b_policy import layout_struct as popsb_layout
    return popsb_layout(spec)


def plan(a, b, name, la, lb, shape, *, ident, grid=SELECTION_GRID, arch='sm_90', split=True, configs=None,
         bodies=None, runtime_layout=False, residency=1):
    """Plan one product (caller validated FP32 dtypes). Returns (plan | None, reason)."""
    if arch != 'sm_90':
        return None, 'P-OPS2-A cost model qualified on H100 sm_90 only'
    if len(la.shape) not in (2, 3) or len(lb.shape) != len(la.shape):
        return None, 'requires matching rank-2/rank-3 operands'
    m, k = la.shape[-2:]
    kb, n = lb.shape[-2:]
    batch = la.shape[0] if len(la.shape) == 3 else 1
    if k != kb or tuple(shape) != ((batch, m, n) if len(la.shape) == 3 else (m, n)):
        return None, 'matrix shape/batch mismatch'
    from .pops_b_policy import admit
    flags = admit(la, lb, m, n, k, batch)
    table = FITTED_2CTA if residency == 2 else FITTED
    grid = 2 * grid if residency == 2 else grid
    configs = [c for c in (configs or table) if bodies is None or c[0] in bodies]
    transposed = False
    if flags is None and n % 64 and not m % 64:
        # C-OPS4-A: a product whose column count admits no tile (e.g. a 32-wide attention head) is planned as
        # C^T = B^T A^T: the same per-element strict-FP32 dot products in the same ascending-k order, stored
        # through a transposed destination (row stride 1, column stride n). No split-K for this form.
        from .lower import TensorLayout
        swap = lambda l: TensorLayout(tuple(l.shape[:-2]) + (l.shape[-1], l.shape[-2]),
                                      tuple(l.strides[:-2]) + (l.strides[-1], l.strides[-2]), l.root, l.offset)
        flags = admit(swap(lb), swap(la), n, m, k, batch)
        if flags is not None:
            transposed, (m, n), (a, b) = True, (n, m), (b, a)
    if flags is None:
        return None, 'operand layouts outside the unit-stride/16-byte contract'
    rows = [r for r in candidates(flags, m, n, k, batch, grid, configs, table)
            if (split and (MOESUP_SPLITS or not transposed)) or r['ksplit'] == 1]
    if not rows:
        return None, 'no tile divides the product shape'
    row = rows[0]
    spec = {'a': a, 'b': b, 'out': name, 'm': m, 'n': n, 'k': k, 'batch': batch, **flags,
            'bm': row['bm'], 'bn': row['bn'], 'ksplit': row['ksplit'], 'tiles': row['tiles'], 'tasks': row['tasks'],
            'scm': 1 if transposed else n, 'scn': m if transposed else 1, 'bsc': m * n}
    lname, ldef = layout_struct(spec)
    ak, bkc = str(flags['ak']).lower(), str(flags['bkc']).lower()
    split_k = row['ksplit'] > 1
    part, cnt = ('/*P2_PART*/', '/*P2_CNT*/') if split_k else ('nullptr', 'nullptr')
    args = (f"{ident(flags['a_root'])} + {flags['a_offset']}LL, {ident(flags['b_root'])} + {flags['b_offset']}LL, "
            f"{ident(name)}, {part}, {cnt}, /*B_VBASE*/, smem")
    body = row['body']
    if body == 'popsb':
        call = f"pops::gemm_phase<{row['bm']},{row['bn']},{ak},{bkc},{row['nst']},{lname},{row['bk']}>({args});"
        helpers = POPSB_HELPERS + '\n' + ldef
    elif body == 'simt' and runtime_layout:
        # Shared out-of-line body per (tile, k depth, operand layout class); layout values are arguments.
        import re
        values = re.findall(r'static constexpr (int|long long) (\w+) = (-?\d+)(LL)?;', ldef)
        init = ', '.join(v + ('LL' if t == 'long long' else '') for t, _, v, _ in values)
        call = f"pops2::simt_rt<{row['bm']},{row['bn']},{row['bk']},{ak},{bkc}>(pops2::RtLayout{{{init}}}, {args});"
        helpers = '#include "fp32_simt.cuh"\n'
    elif body == 'simt':
        call = f"pops2::simt_phase<{row['bm']},{row['bn']},{row['bk']},{ak},{bkc},{lname}>({args});"
        helpers = '#include "fp32_simt.cuh"\n' + ldef
    else:
        call = (f"pops2s::slice_phase<{row['bm']},{row['bn']},{row['slices']},{row['bk']},{row['nst']},{ak},{bkc},"
                f"{lname}>({args});")
        helpers = '#include "fp32_slice.cuh"\n' + ldef
    a64 = lambda words: (words + 63) // 64 * 64
    metadata = {'engine': ENGINE, 'body': body + ('_rt' if body == 'simt' and runtime_layout else ''), 'tiles': row['tasks'], 'batch': batch, 'k': k,
                'ksplit': row['ksplit'], 'matrix_shape': [batch, m, n, k], 'shared_bytes': row['shared_bytes'],
                'matrix_plan': {key: row[key] for key in ('body', 'bm', 'bn', 'bk', 'nst', 'slices', 'ksplit',
                                                          'tiles', 'tasks', 'steps', 'estimate_us')},
                'matrix_alternatives': [{key: r[key] for key in ('body', 'bm', 'bn', 'bk', 'slices', 'ksplit',
                                                                 'estimate_us')} for r in rows[1:4]],
                'p2_split': {'counter_words': a64(row['tiles']), 'partial_words': a64(row['tasks'] * row['bm'] * row['bn'])}
                if split_k else None,
                'requires_full_grid': True, 'selection_grid': grid, 'transposed_product': transposed,
                'arithmetic': ('strict FP32 fmaf, ascending k within each k slice from +0; '
                               + (f"{row['slices']} in-CTA slices of every k stage summed in fixed order; " if body == 'slice' and row['slices'] > 1 else '')
                               + ('cross-CTA split-K slices summed in fixed order; ' if split_k else '')
                               + 'no TF32/tensor cores/data atomics')}
    return {'code': call, 'helpers': helpers, 'headers': [HEADERS[body]], 'metadata': metadata,
            'selection_contract': dict(spec),
            'operand_roots': list(dict.fromkeys((flags['a_root'], flags['b_root'])))}, (
                f'P-OPS2-A selected {body}' + (' (transposed product, C-OPS4-A)' if transposed else ''))
