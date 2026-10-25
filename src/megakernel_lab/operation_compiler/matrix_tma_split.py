"""Deterministic split-K for underfilled long-K provider TMA products (LATCH-TINYMOE-MATRIX).

A batch-1 BF16 product selected for the provider's pops4b TMA + wgmma body whose output has few tiles and whose
contraction is long (typically a weight gradient dW = dY^T X over the tokens) keeps most CTAs idle while each busy
CTA streams a long operand slab through one ring: its interval lasts as long as one tile's serial k loop. The body's
existing P6-B mode (pops4b::tma_body SPLIT > 1) runs (K slice, tile) tasks instead; each slice accumulates its
contiguous K range in FP32 from +0 over ascending k16 steps, exactly as the unsplit body, and stores the unrounded
FP32 slice tile. A 'splitk_reduce' phase then adds the slices in ascending slice order in FP32 and rounds once to
BF16 (round to nearest even), writing the product's ordinary BF16 result for its unchanged consumers.

Arithmetic contract: BF16 operands, FP32 accumulation, one BF16 rounding of the FP32 sum; only the association of
the K sum changes (S partial chains added in fixed order), which the practical precision policy permits
("reassociated reductions", "BF16 matrix partials accumulated in FP32"). Results are deterministic and independent
of the grid. Selection is structural (shape, tile count, contraction length, layout, precision policy); no model or
parameter names are consulted.
"""
from __future__ import annotations

KEY = "tma_split_k"
ELEMENTWISE_SUM = False  # the slice sum is an ordinary elementwise value (False: a separate splitk_reduce body)
SPLITS = (8, 4, 2)
MIN_STEPS = 16          # unsplit 64-wide k steps: only long contractions
MIN_SLICE = 256         # each slice keeps at least four 64-wide k steps
MAX_TILE_SHARE = 4      # admitted when the unsplit tiles occupy at most 1/4 of the grid
MAX_TASK_GRIDS = 1      # (slice, tile) tasks: at most this many grid-widths (2 measured slower: 21.66 vs 21.59 ms)
LONG_LEAST_TRAFFIC = False  # long slices on the least-traffic tile (measured with MAX_TASK_GRIDS 2: slower)


LONG_K = 8192           # very long contractions (vocabulary-scale input gradients)
LONG_SLICE_STEPS = 32   # slice chains of at most 2048 (the step-3 long-K chunk length of the P5C/P8WS bodies)
LONG_MAX_TASKS = 16     # at most this many grid-widths of (slice, tile) tasks


def choose_long(spec, plan, grid):
    """K slices for a very long contraction (K >= LONG_K): the fewest equal whole-step slices of <= 2048, or None."""
    if (spec.get('batch', 1) != 1 or spec.get('tma_batch') or spec.get('ksegs') or spec.get('serial_bf16_k')
            or spec.get('out_raw') or spec.get('rowstats') or spec.get('ldc') or not spec.get('pair', True)
            or not spec.get('tma') or spec['n'] % 8):
        return None
    k, tiles = int(spec['k']), int(plan['tiles'])
    if k % 64 or k < LONG_K or tiles > grid // 2:
        return None
    steps = k // 64
    for splits in range(2, steps + 1):
        if steps % splits == 0 and steps // splits <= LONG_SLICE_STEPS:
            return splits if tiles * splits <= LONG_MAX_TASKS * grid else None
    return None


def choose_long_exact(spec, plan, grid):
    """LATCH-B1-FAST (matrix_schedule "exact_split"): K slices that reproduce the chunked body's own association.

    The chunked P5C/P8WS bodies (p5c_matrix.chunk_stages) restart their wgmma chain every PK 64-wide stages and add
    the chunk results in chunk order with round-to-nearest FP32 additions. Slices of exactly PK stages, the last one
    padded to PK stages with out-of-range (zero-filled) TMA stages, compute the same chunk partials from +0 over the
    same ascending k16 steps; the ascending slice sum and the single BF16 rounding are the body's. A padded stage adds
    exact zeros to a nonzero partial. Returns (splits, padded K) or None."""
    from .p5c_matrix import chunk_stages
    if (spec.get('batch', 1) != 1 or spec.get('tma_batch') or spec.get('ksegs') or spec.get('serial_bf16_k')
            or spec.get('out_raw') or spec.get('rowstats') or spec.get('ldc') or not spec.get('pair', True)
            or not spec.get('tma') or spec['n'] % 8):
        return None
    k, tiles = int(spec['k']), int(plan['tiles'])
    pk = chunk_stages(k)
    if not pk or k % 64 or k < LONG_K or tiles > grid // 2:
        return None
    chunk = 64 * pk
    splits = -(-k // chunk)
    if splits < 2 or tiles * splits > LONG_MAX_TASKS * grid:
        return None
    return splits, splits * chunk


def choose(spec, plan, grid):
    """Number of K slices for a provider pops4b TMA product, or None."""
    if (spec.get('batch', 1) != 1 or spec.get('tma_batch') or spec.get('ksegs') or spec.get('serial_bf16_k')
            or spec.get('out_raw') or spec.get('rowstats') or spec.get('ldc') or not spec.get('pair', True)
            or not spec.get('tma') or spec['n'] % 8):
        return None
    k, tiles = int(spec['k']), int(plan['tiles'])
    if k % 64 or k // 64 < MIN_STEPS or tiles * MAX_TILE_SHARE > grid:
        return None
    for splits in SPLITS:
        if k % (64 * splits) == 0 and k // splits >= MIN_SLICE and tiles * splits <= MAX_TASK_GRIDS * grid:
            return splits
    return None


def emit(provided, *, part, ident, grid, long=False):
    """Split producer and reduce code for a provider record, or None when the rule does not apply.

    long=True: the product's native body was a chunked long-contraction body; slices follow choose_long."""
    from .matrix_adapters.bodies import BY_NAME, pinned
    extra = provided['extra']
    record = extra.get('tma_respec')
    if record is None or not extra.get('engine', '').endswith('pops4b_tma'):
        return None
    spec, plan = record['spec'], dict(record['plan'])
    k_pad = None
    if long == 'exact':
        chosen_ = choose_long_exact(spec, plan, grid)
        if chosen_ is None:
            return None
        splits, k_pad = chosen_
    else:
        splits = (choose_long if long else choose)(spec, plan, grid)
    if splits is None:
        return None
    m, n, k = int(spec['m']), int(spec['n']), int(spec['k'])
    k_loop = k if k_pad is None else k_pad        # the kernel's K loop; tensor maps keep the true extent k
    body = BY_NAME['pops4b_tma']
    if long and LONG_LEAST_TRAFFIC:
        # The slices already fill the grid many times over; among the body's admitted tiles take the one that
        # streams the fewest operand bytes (the planner's traffic term) while keeping at least a grid of tasks.
        with pinned(body.module, body.tile_settings()) as mod:
            try:
                _, candidates = mod.wg_plan(m, n, k, 1, spec['ak'], spec['bkc'], grid=grid)
            except mod.MatrixRuleError:
                candidates = []
        fits = [(c, choose_long(spec, c, grid)) for c in candidates]
        fits = [(c, s_) for c, s_ in fits if s_ is not None and c['tiles'] * s_ >= grid]
        if fits:
            best, splits = min(fits, key=lambda row: (row[0]['traffic_us'], row[0]['tiles'], -row[0]['bm']))
            plan = dict(best)
    plan['nst'] = max(2, min(int(plan['nst']), k_loop // splits // 64))
    with pinned(body.module, body.settings) as mod:
        depth = mod._LOWER_DEPTH[0]
        mod._LOWER_DEPTH[0] = 1
        try:
            code = mod._code(dict(spec, out=ident(part), split=splits,
                                  **({'k_pad': k_pad} if k_pad is not None else {})), plan, ident, 0, 1, 1, record['rot'])
        except mod.MatrixRuleError:
            return None
        finally:
            mod._LOWER_DEPTH[0] = depth
    tiles = int(plan['tiles'])
    mn = m * n
    out = spec['out']
    terms = "\n".join(f"    {{ const float4 t_ = p_[{s * mn // 4}LL + q]; s_.x += t_.x; s_.y += t_.y; s_.z += t_.z; "
                      f"s_.w += t_.w; }}" for s in range(1, splits))
    reduce = (
        "{ // tma_split_k: ascending FP32 slice sum, one BF16 rounding\n"
        f"  const float4* p_ = reinterpret_cast<const float4*>({ident(part)});\n"
        f"  for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < {mn // 4}LL;"
        " q += (long long)gridDim.x * blockDim.x) {\n"
        "    float4 s_ = p_[q];\n"
        f"{terms}\n"
        "    const __nv_bfloat162 lo_ = __floats2bfloat162_rn(s_.x, s_.y), hi_ = __floats2bfloat162_rn(s_.z, s_.w);\n"
        f"    reinterpret_cast<uint2*>({out})[q] = make_uint2(*reinterpret_cast<const unsigned*>(&lo_),"
        " *reinterpret_cast<const unsigned*>(&hi_));\n"
        "  }\n}")
    split_plan = dict({key: value for key, value in plan.items() if isinstance(value, (int, float, str, bool))},
                      split_k=splits, tiles=tiles * splits)
    split_extra = {key: value for key, value in extra.items() if key not in ('tma_respec',)}
    split_extra.update(
        tiles=tiles * splits, work_items=tiles * splits, matrix_plan=split_plan,
        k=k_loop // splits, shared_bytes=int(plan['nst']) * (int(plan['bm']) + int(plan['bn'])) * 128 + 1024,
        matrix_spec=dict(extra['matrix_spec'], out=part),
        arithmetic=(f'wgmma.mma_async; {splits} K slices, each FP32 accumulation from +0 over ascending k16 steps; '
                    'FP32 slice results (splitk_reduce: ascending slice sum, one round-to-nearest BF16 rounding)'
                    + (f'; slices of the chunked body\'s {k_loop // splits} chunk length, last slice padded to '
                       f'{k_loop} with zero-filled stages (exact)' if k_pad is not None else '')),
        latch_matrix=dict(extra['latch_matrix'], bitwise_class=f'bf16_wgmma_k16_split{splits}_ascending'))
    split_extra[KEY] = dict(long_contraction=bool(long), exact=k_pad is not None, k_pad=k_pad, splits=splits,
                            shape=[m, n, k], tiles=tiles, slice_k=k_loop // splits, partial=part,
                            tile=[int(plan['bm']), int(plan['bn']), int(plan['wm'])], ak=bool(spec['ak']),
                            bkc=bool(spec['bkc']), reduction='ascending FP32 slice sum, one BF16 RN rounding')
    # The same ascending slice sum as an ordinary elementwise value (flat index i over the [M, N] result): the
    # planner's elementwise rules then treat the reduction like any other same-shape producer chain.
    total = f"{ident(part)}[i]"
    for s in range(1, splits):
        total = f"({total} + {ident(part)}[{s * mn}LL + i])"
    return dict(code=code, extra=split_extra, reduce=reduce, splits=splits, elements=splits * mn,
                sum_expression=total, mn=mn)


SHORT_K = 512           # at most eight 64-wide k steps


def short_contraction(problem, chosen, resources, decision):
    """Body for a short single-wave contraction under ``matrix_schedule='underfilled'`` or ``'exact'`` (else ``chosen``).

    A provider TMA product with K <= SHORT_K whose tiles fit one wave of the grid runs each CTA's single tile
    through a few k steps: the TMA body's per-task setup (tensor-map fetch, barrier ring, staged epilogue) is not
    amortized. The admitted cp.async wgmma body of the same bitwise class computes identical values with a shorter
    fixed path. Plans and arithmetic are the bodies' own; only the body choice changes."""
    from . import matrix_schedule
    if (matrix_schedule.current() not in ('underfilled', 'exact', 'exact_split') or chosen is None or chosen.body != 'pops4b_tma'
            or problem.batch != 1 or problem.k > SHORT_K):
        return chosen
    from .matrix_rules_p7b import TARGET_GRID
    if int(chosen.plan.get('tiles', TARGET_GRID + 1)) > TARGET_GRID:
        return chosen
    from .matrix_adapters.candidates import candidates as all_candidates
    same = chosen.requirements.precision.bitwise_class
    alt = [c for c in all_candidates(problem, resources)[0]
           if c.admitted and c.body == 'pops4b_wg' and c.requirements.precision.bitwise_class == same]
    if not alt:
        return chosen
    decision['rule'] = decision.get('rule', '') + '; matrix_schedule: short single-wave contraction on the cp.async wgmma body'
    alt[0].short_contraction = True
    return alt[0]
