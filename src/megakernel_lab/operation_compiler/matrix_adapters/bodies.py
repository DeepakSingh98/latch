"""Adapters exposing the frozen compiler's existing matrix bodies as explicit candidates.

No matrix body is reimplemented here. Each adapter calls the donor planner (tile/stage choice) and the donor
emitter (the exact CUDA call string) from ``research/latch_ops/ops`` at base ab4703ee3, under explicitly pinned
module settings, and states the body's requirements. Donor provenance per body is in ``BODY_INFO``.

Emission contract (every body): the returned code is one block executed by all 256 threads of each CTA in a
CTA slice ``(lo, hi, den)`` of the cooperative grid starting at rotation ``rot``; it references the operand
identifiers ``ident(a)``/``ident(b)`` (so dependency analysis sees the reads), writes the result through the
``out`` pointer expression, and uses the program's dynamic shared buffer ``smem``. Operand visibility before
and result visibility after are the caller's barrier responsibility (``Synchronization.barrier_*``).
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib

from .problem import MatrixProblem
from .requirements import Precision, Requirements, Synchronization

OPS = 'megakernel_lab.operation_compiler'      # the frozen compiler package (research/latch_ops/ops); imported lazily (pulls in torch)


def _mod(name):
    return importlib.import_module(f'{OPS}.{name}')


@contextmanager
def pinned(module_name, settings):
    """Temporarily apply module-level settings; always restore (donor planners read them at call time)."""
    module = _mod(module_name)
    saved = {key: copy.deepcopy(getattr(module, key)) for key in settings}
    try:
        for key, value in settings.items():
            if not hasattr(module, key):
                raise AttributeError(f'{module_name} has no setting {key}')
            setattr(module, key, copy.deepcopy(value))
        yield module
    finally:
        for key, value in saved.items():
            setattr(module, key, value)


class Refusal(ValueError):
    pass


# ----------------------------------------------------------------------------------------------- precision
MMA_SYNC = Precision('bfloat16', 'float32', 'bfloat16', 'one round-to-nearest BF16 rounding (cvt.rn.bf16x2)',
                     'one FP32 accumulator per element from +0, ascending k16 blocks', 'mma.sync.m16n8k16',
                     'bf16_mma_sync_k16')
WGMMA = Precision('bfloat16', 'float32', 'bfloat16', 'one round-to-nearest BF16 rounding of the FP32 accumulator',
                  'tensor-core FP32 accumulation from +0, ascending k16 steps', 'wgmma.mma_async',
                  'bf16_wgmma_k16')
BF16_SIMT = Precision('bfloat16', 'float32', 'bfloat16', 'one round-to-nearest BF16 rounding',
                      'exact BF16->FP32 widening, one fmaf chain per element from +0, ascending k', 'fmaf',
                      'bf16_fmaf_k1')
FP32_SIMT = Precision('float32', 'float32', 'float32', 'none (FP32 result)',
                      'one fmaf chain per element from +0, ascending k; no TF32/tensor cores', 'fmaf',
                      'fp32_fmaf_k1')


def _with_result(prec, problem, split=1):
    from dataclasses import replace
    result = 'float32' if problem.out_mode != 'store' else prec.result
    rounding = prec.rounding
    if problem.out_mode == 'store_f32':
        rounding += '; stored widened to FP32 (exact)'
    elif problem.out_mode == 'store_raw':
        rounding = 'none: unrounded FP32 accumulator stored (BF16-exact FP32 product; LATCH-RESEARCH-EDITS)'
    elif problem.out_mode == 'accumulate_f32':
        rounding += '; widened and added to the FP32 destination'
    cls = prec.bitwise_class + (f'_split{split}' if split > 1 else '')
    order = prec.order + (f'; {split} K slices summed in fixed slice order (FP32 reassociation)' if split > 1 else '')
    return replace(prec, result=result, rounding=rounding, bitwise_class=cls, order=order, split_k=split)


def _with_chunks(prec, plan):
    """The long-K repair is a distinct arithmetic class from a single chain."""
    from dataclasses import replace
    stages = plan.get('chunk_stages')
    if not stages:
        return prec
    return replace(prec,
                   order=(f'ascending k16 steps in independent chains from +0 of at most {stages} 64-k stages; '
                          'chain results summed in ascending order with FP32 round-to-nearest additions'),
                   bitwise_class=prec.bitwise_class + f'_chunk{stages}')


# ----------------------------------------------------------------------------------------------- shared checks
def _aligned(problem, elements):
    """Leading/batch strides and offsets are multiples of ``elements`` (16-byte for BF16 when 8)."""
    a, b = problem.a, problem.b
    lead_a = a.row_stride if problem.ak else a.col_stride
    lead_b = b.col_stride if problem.bkc else b.row_stride
    return all(s % elements == 0 for s in (lead_a, lead_b, a.batch_stride, b.batch_stride, a.offset, b.offset))


def _unit(problem):
    a, b = problem.a, problem.b
    return (a.col_stride == 1 or a.row_stride == 1) and (b.row_stride == 1 or b.col_stride == 1)


def _spec_strides(problem):
    a, b = problem.a, problem.b
    return [a.row_stride, a.col_stride, b.row_stride, b.col_stride,
            a.batch_stride if problem.batch > 1 else 0, b.batch_stride if problem.batch > 1 else 0]


def _bf16_store(problem):
    return problem.dtype == problem.out_dtype == 'bfloat16' and problem.out_mode == 'store'


def _popsc_spec(problem, a, b, out, **extra):
    return {'dtype': problem.dtype, 'a': a, 'b': b, 'out': out, 'm': problem.m, 'n': problem.n, 'k': problem.k,
            'batch': problem.batch, 'strides': _spec_strides(problem), 'ak': problem.ak, 'bkc': problem.bkc, **extra}


class Body:
    name = ''
    module = ''
    headers = ()
    settings = {}

    def structural(self, problem: MatrixProblem):
        """(admissible, reason) from operation properties alone."""
        raise NotImplementedError

    def plans(self, problem, grid, shared_limit=None):
        """Admissible plans (dicts) ordered best-first by the donor's own cost estimate, planned within the
        program's dynamic shared-memory limit when one is given."""
        raise NotImplementedError

    def requirements(self, problem, plan) -> Requirements:
        raise NotImplementedError

    def emit(self, problem, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        raise NotImplementedError


# --------------------------------------------------------------------------------------------- P-OPS-C mma.sync
class PopscBF16(Body):
    """P-OPS-C mma.sync BF16 body, M/N/K multiples of 32 (matrix_rules.bf16_plan / popsc::bmm_phase)."""
    name, module, headers = 'popsc_bf16', 'matrix_rules', ('tgemm.cuh', 'bgemm.cuh', 'popsc_mm.cuh')
    settings = {'TARGET_GRID': 128, 'SHARED_LIMIT': 192 * 1024}

    def structural(self, p):
        if not _bf16_store(p):
            return False, 'BF16 operands and BF16 stored result only'
        if not _unit(p):
            return False, 'needs a unit-stride matrix dimension in A and in B'
        if not _aligned(p, 8):
            return False, 'needs 16-byte aligned leading/batch strides and offsets'
        if p.m % 32 or p.n % 32 or p.k % 32:
            return False, 'M, N, K must be multiples of 32'
        return True, ''

    def _settings(self, shared_limit):
        if shared_limit is None:
            return self.settings
        return {**self.settings, 'SHARED_LIMIT': min(self.settings['SHARED_LIMIT'], int(shared_limit))}

    def plans(self, p, grid, shared_limit=None):
        with pinned(self.module, self._settings(shared_limit)) as mod:
            try:
                chosen, cands = mod.bf16_plan(p.m, p.n, p.k, p.batch, p.ak, p.bkc, grid=grid)
            except mod.MatrixRuleError as error:
                raise Refusal(str(error))
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def requirements(self, p, plan):
        # Qualified at two resident CTAs/SM inside the P7-B protocol when the program shared ceiling is respected.
        return Requirements('sm_90', plan['shared_bytes'], headers=self.headers, max_resident_ctas_per_sm=2,
                            sync=Synchronization(notes='__noinline__ body (own register allocation); grid-strided tiles'),
                            precision=_with_result(MMA_SYNC, p))

    def _call(self, mod, spec, plan, ident, lo, hi, den, rot):
        return mod._code(spec, plan, ident, lo, hi, den, rot)

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            code = self._call(mod, _popsc_spec(p, a, b, out), plan, ident, lo, hi, den, rot)
        return Emission(code, self.headers, tiles=plan['tiles'])


class PopscBF16Tail(PopscBF16):
    """C-OPS4-B tail body for M/N/K multiples of 16 (matrix_rules.bf16_tail_plan / popsc::bmm_tail)."""
    name = 'popsc_bf16_tail'

    def structural(self, p):
        ok, why = super().structural(p)
        if ok or why != 'M, N, K must be multiples of 32':
            return ok, why
        if p.m % 16 or p.n % 16 or p.k % 16:
            return False, 'M, N, K must be multiples of 16'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        with pinned(self.module, self._settings(shared_limit)) as mod:
            try:
                chosen, cands = mod.bf16_tail_plan(p.m, p.n, p.k, p.batch, p.ak, p.bkc, grid=grid)
            except mod.MatrixRuleError as error:
                raise Refusal(str(error))
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            code = mod._code(_popsc_spec(p, a, b, out, tail=True), plan, ident, lo, hi, den, rot)
        return Emission(code, self.headers, tiles=plan['tiles'])


class PopscBF16Edge(PopscBF16):
    """P-OPS4-B masked-edge mma.sync body for arbitrary M/N/K (matrix_rules_p7b edge plan / popsc::bmm_edge_phase).

    Absent operand elements are staged as +0 and stores are masked: bitwise equal to the pinned 64x64 body on
    zero-padded operands (donor claim, re-verified by this lane's harness)."""
    name, module = 'popsc_bf16_edge', 'matrix_rules_p7b'
    settings = {'TARGET_GRID': 128, 'SHARED_LIMIT': 192 * 1024, 'LARGE_TILES': False, 'FORCE_BY_OUTPUT': {}}

    def structural(self, p):
        if not _bf16_store(p):
            return False, 'BF16 operands and BF16 stored result only'
        if not _unit(p):
            return False, 'needs a unit-stride matrix dimension in A and in B'
        if not _aligned(p, 8):
            return False, 'needs 16-byte aligned leading/batch strides and offsets'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        with pinned(self.module, self._settings(shared_limit)) as mod:
            try:
                chosen, cands = mod.bf16_plan(p.m, p.n, p.k, p.batch, p.ak, p.bkc, grid=grid, edge=True)
            except mod.MatrixRuleError as error:
                raise Refusal(str(error))
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            code = mod._code(_popsc_spec(p, a, b, out, edge=True, pair=p.n % 2 == 0), plan, ident, lo, hi, den, rot)
        return Emission(code, self.headers, tiles=plan['tiles'])


# ------------------------------------------------------------------------------------- P-OPS4-B warpgroup MMA
class Pops4bWG(Body):
    """P-OPS4-B cp.async + wgmma body, masked edges, batched (matrix_rules_p7b.wg_plan / pops4b::wmm_phase_inline)."""
    name, module = 'pops4b_wg', 'matrix_rules_p7b'
    headers = ('tgemm.cuh', 'bgemm.cuh', 'popsc_mm.cuh', 's107a_wgmma.cuh', 'pops4b_wg.cuh')
    settings = {'TARGET_GRID': 128, 'SHARED_LIMIT': 192 * 1024, 'WG_INLINE': True, 'WG_FLAT': False, 'WG_DIAG': 0,
                'WG_TILES': ((128, 256, 2), (128, 192, 2), (128, 128, 2), (128, 64, 2), (64, 256, 1), (64, 128, 1)),
                'P9E_STAGING_RESERVE': False, 'P9E_RING_RULE': False, 'P9E_WIDE_TILES': False,
                'P9E_TALL_TILES': False, 'P9E_WIDE_ONLY': False, 'FORCE_BY_OUTPUT': {}}
    tma = False

    def structural(self, p):
        if not _bf16_store(p):
            return False, 'BF16 operands and BF16 stored result only'
        if not _unit(p):
            return False, 'needs a unit-stride matrix dimension in A and in B'
        if not _aligned(p, 8):
            return False, 'needs 16-byte aligned leading/batch strides and offsets'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        limit = self.settings['SHARED_LIMIT'] if shared_limit is None else min(self.settings['SHARED_LIMIT'], shared_limit)
        with pinned(self.module, self.settings) as mod:
            try:
                chosen, cands = mod.wg_plan(p.m, p.n, p.k, p.batch, p.ak, p.bkc, grid=grid, shared_limit=limit)
            except mod.MatrixRuleError as error:
                raise Refusal(str(error))
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def requirements(self, p, plan):
        return Requirements('sm_90a', plan['shared_bytes'], headers=self.headers, max_resident_ctas_per_sm=2,
                            host_setup=('tensor_map:A', 'tensor_map:B') if self.tma else (),
                            sync=Synchronization(entry_resident=True, smem_mbarriers=plan['nst'] if self.tma else 0,
                                                 notes='entry-resident inline wgmma body (ptxas C7510 otherwise)'),
                            precision=_with_result(WGMMA, p))

    def _spec(self, p, a, b, out):
        spec = _popsc_spec(p, a, b, out, wg=True, pair=p.n % 2 == 0)
        if p.m % 32 or p.n % 32 or p.k % 32:
            spec['edge'] = True
        return spec

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            code = mod._code(self._spec(p, a, b, out), plan, ident, lo, hi, den, rot)
        return Emission(code, self.requirements(p, plan).headers, tiles=plan['tiles'])


import contextvars as _contextvars
# Release step 14: set by the flat common planner when its group rules name 'w96'.
TMA_EXTRA_TILES = _contextvars.ContextVar('latch_step14_tma_extra_tiles', default=False)
# Release step 14 ('tdeep'): deeper operand rings for underfilled TMA products.
TMA_DEEP_RINGS = _contextvars.ContextVar('latch_step14_tma_deep_rings', default=False)


def deepen(plan, grid, limit):
    """A product whose tiles fill at most half the grid runs one tile per CTA with a long serial k loop: keep up to
    eight 64-wide k stages in flight (pops4b admits NST <= 8) within the shared limit, reserving the staged-epilogue
    tile. Tile shape, k order and rounding are unchanged."""
    from .. import compose
    if not compose.STEP14_RULES or not TMA_DEEP_RINGS.get() or plan['tiles'] > grid // 2:
        return plan
    nk = plan['padded'][2] // 64
    stage = (plan['bm'] + plan['bn']) * 128
    reserve = plan['bm'] * (plan['bn'] + 8) * 2
    nst = min(8, nk, (limit - 1024 - reserve) // stage)
    if nst <= plan['nst']:
        return plan
    return {**plan, 'nst': nst, 'shared_bytes': nst * stage + 1024, 'deep_ring': True}


# LATCH-MOESUP: batched products whose batches stack into one product of batch x M rows (matrix_rules_p7b.
# batch_stacked) run on the TMA body too; the cp.async form keeps every other batched product.
BATCH_TMA = True
# Program-scoped admission, set by the composed planner from graph structure.
# It does not change the arithmetic or the public precision policy.
BATCH_TMA_PROGRAM = _contextvars.ContextVar('latch_batch_tma_program', default=True)


def _stacked(p):
    if p.batch == 1 or not BATCH_TMA or not BATCH_TMA_PROGRAM.get() or p.out_mode != 'store':
        return False
    from .. import matrix_rules_p7b
    return matrix_rules_p7b.batch_stacked(p.m, p.n, p.k, p.batch, p.ak, p.bkc, _spec_strides(p))


class Pops4bTMA(Pops4bWG):
    """P-OPS4-B TMA + wgmma body (batch 1 or batch-stacked), masked edges, 16-byte stores (pops4b::tma_phase_inline)."""
    name = 'pops4b_tma'
    headers = Pops4bWG.headers + ('pops4b_tma.cuh',)
    settings = {**Pops4bWG.settings, 'WG_TMA': True, 'WG_VEC_STORE': True, 'P7B_CLAIM': False, 'P7B_LPT': False}
    tma = True
    # Release step 14: 128 x 96 tiles (two 64 x 96 warpgroups; the body admits BN 96 for K-major B, planner-enforced
    # by wg_plan's whole-64-column rule for N-major B). A 1152 x 1056 output becomes 99 tiles instead of 81 on a
    # 132-SM grid. The cost model chooses among the candidates; arithmetic is the same k16 wgmma accumulation.
    STEP14_TILES = ((128, 96, 2),)

    def tile_settings(self):
        from .. import compose
        if not compose.STEP14_RULES or not TMA_EXTRA_TILES.get():
            return self.settings
        return {**self.settings, 'WG_TILES': tuple(self.settings['WG_TILES']) + self.STEP14_TILES}

    def plans(self, p, grid, shared_limit=None):
        limit = self.settings['SHARED_LIMIT'] if shared_limit is None else min(self.settings['SHARED_LIMIT'], shared_limit)
        with pinned(self.module, self.tile_settings()) as mod:
            try:
                chosen, cands = mod.wg_plan(p.m, p.n, p.k, p.batch, p.ak, p.bkc, grid=grid, shared_limit=limit)
            except mod.MatrixRuleError as error:
                raise Refusal(str(error))
        cands = [deepen(c, grid, limit) for c in cands]
        if p.batch > 1:
            # batch-stacked: tiles must not cross a batch (and never take the deepened single-tile rings)
            cands = [c for c in cands if p.m % c['bm'] == 0 and (not p.bkc or p.n % c['bn'] == 0)
                     and not c.get('deep_ring')]
            if not cands:
                raise Refusal('no batch-stacked tile divides the per-batch extents')
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def structural(self, p):
        if p.dtype == 'bfloat16' and p.out_dtype == 'float32' and p.out_mode == 'store_raw':
            # LATCH-RESEARCH-EDITS: unrounded FP32 result of a BF16-exact product (batch 1, even N: the raw store
            # writes column pairs); the same body, tiles, maps and k order as the BF16-storing form.
            if p.batch != 1 or p.n % 2:
                return False, 'raw FP32 result: batch 1 and even N only'
            if not _unit(p):
                return False, 'needs a unit-stride matrix dimension in A and in B'
            if not _aligned(p, 8):
                return False, 'needs 16-byte aligned leading/batch strides and offsets'
            return True, ''
        ok, why = super().structural(p)
        if not ok:
            return ok, why
        if p.batch != 1 and not _stacked(p):
            return False, 'TMA form: batch 1 or batch-stacked products only (the cp.async form covers batches)'
        return True, ''

    def _spec(self, p, a, b, out):
        spec = super()._spec(p, a, b, out)
        if p.out_mode == 'store_raw':
            spec['out_raw'] = True
        # Maps bind to the operand storage roots and element offsets; the Runtime encodes them per launch.
        spec['tma'] = {'a': (p.a.root, p.a.offset), 'b': (p.b.root, p.b.offset)}
        if p.batch > 1:
            spec['tma_batch'] = True
        return spec

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0, ldc=None, serial_bf16_k=None):
        spec = self._spec(p, a, b, out)
        if serial_bf16_k is not None:
            if type(serial_bf16_k) is not int or serial_bf16_k <= 0 or serial_bf16_k % 16:
                raise Refusal('serial BF16 partition must be a positive multiple of16')
            spec['serial_bf16_k'] = serial_bf16_k
        if ldc is not None:
            if ldc % 8 or ldc < -(-p.n // 8) * 8:
                raise Refusal('padded result pitch must be a multiple of 8 covering whole 8-column chunks')
            spec['ldc'] = int(ldc)        # step 02: result rows ldc apart; 16-byte stores of whole chunks
        with pinned(self.module, self.settings) as mod:
            depth = mod._LOWER_DEPTH[0]
            mod._LOWER_DEPTH[0] = 1         # outermost flat program: tensor maps admitted
            try:
                code = mod._code(spec, plan, ident, lo, hi, den, rot)
            finally:
                mod._LOWER_DEPTH[0] = depth
        return Emission(code, self.headers, tiles=plan['tiles'], map_marker='POPS4B_TMAP')


# ---------------------------------------------------------------------------------------- P-OPS5-C / P8-A WGMMA
P5C_FLAT = {'ENGINE': 'p5c', 'MIN_K_STAGES': 8, 'MIN_TILE_FRACTION': 0.5, 'P5C_MAX_STAGES': 4, 'P5C_STAGED': False,
            'P5C_MN_MAJOR_A_PENALTY': 1.0, 'P5C_FORCE': {}, 'P5C_PINGPONG': 'never', 'P5C_MULTICAST': False,
            'P5C_RELEASE': False, 'P5C_BN96': False, 'P9_TILE64': False, 'P5C_BN192': False,
            'P5C_DECOUPLED': False, 'P5C_SYNC2': False, 'P5C_TRACE': False, 'P5C_ACC_BATCH': False,
            'P5C_ACC_PREFETCH': False, 'P5C_PAR_PRE': False, 'P8WS': True, 'P9WS': False}


class P5C(Body):
    """P-OPS5-C single-chain diagnostic (p5c_matrix.p5c_plan / p5c::phase).

    Normal compilation separately applies its established long-K chunk policy;
    this explicit body retains the historical unchunked comparison.
    """
    name, module = 'p5c', 'p5c_matrix'
    settings = P5C_FLAT

    @property
    def headers(self):
        with pinned(self.module, self.settings) as mod:
            return ('tgemm.cuh', 'bgemm.cuh', 'popsc_mm.cuh') + tuple(mod.wg_headers())

    def structural(self, p):
        if not (p.dtype == 'bfloat16' and p.out_dtype in ('bfloat16', 'float32')):
            return False, 'BF16 operands only'
        if p.out_mode == 'store' and p.out_dtype != 'bfloat16':
            return False, 'FP32 result needs store_f32 or accumulate_f32'
        if p.out_mode == 'store_raw':
            return False, 'unrounded FP32 results take the native P5C raw32 path, not this adapter'
        if p.batch != 1:
            return False, 'batch 1 only'
        if not _unit(p) or not _aligned(p, 8):
            return False, 'needs unit-stride dimensions and 16-byte aligned strides/offsets'
        if p.m % 64 or p.n % 64 or p.k % 64:
            return False, 'M, N, K must be multiples of 64'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        settings = dict(self.settings)
        if shared_limit is not None:
            settings['P5C_SHARED_LIMIT'] = min(220 * 1024, int(shared_limit))
        with pinned(self.module, settings) as mod:
            try:
                chosen, cands = mod.p5c_plan(p.m, p.n, p.k, p.ak, p.bkc, grid=grid)
            except mod.EngineRefusal as error:
                raise Refusal(str(error))
            cands = [dict(c, release=False, out_f32=p.out_mode != 'store', mc=1) for c in cands]
        return sorted(cands, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def donor_admitted(self, p):
        """The frozen flat compiler's own occupancy/size admission rule (FLAT_SELECTED_DEFAULTS)."""
        with pinned(self.module, self.settings) as mod:
            return mod.admit(p.m, p.n, p.k, p.batch, p.ak, p.bkc, p.a.offset, p.b.offset) == 'p5c'

    def requirements(self, p, plan):
        return Requirements('sm_90a', plan['shared_bytes'], headers=self.headers,
                            host_setup=('tensor_map:A', 'tensor_map:B'),
                            sync=Synchronization(entry_resident=True, smem_mbarriers=plan['nst'],
                                                 notes='entry-resident wgmma body; static shared TMA mbarriers'),
                            precision=_with_result(_with_chunks(WGMMA, plan), p))

    def _spec(self, p, a, b, out, plan):
        spec = _popsc_spec(p, a, b, out, engine='p5c', roots=[(p.a.root, p.a.offset), (p.b.root, p.b.offset)])
        if p.out_mode != 'store':
            spec['out_f32'] = True
        if p.out_mode == 'accumulate_f32':
            spec['out_acc'] = True
        return spec

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            code = mod.code(self._spec(p, a, b, out, plan), plan, ident, lo, hi, den, rot)
        return Emission(code, self.headers, tiles=plan['tiles'], map_marker='P5C_TMAP')


class P8WS(P5C):
    """P8-A warp-specialized TMA + wgmma body (BM x 128 tiles, producer warp + epilogue warps; p8ws::phase_mode).

    Shares the WM = 2 p5c arithmetic only when both use the same chunk policy.
    Long-K plans carry their ordered chunk policy and a distinct arithmetic class."""
    name = 'p8ws'

    def structural(self, p):
        ok, why = super().structural(p)
        if not ok:
            return ok, why
        if p.n % 128:
            return False, 'N must be a multiple of 128'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        spec = {'m': p.m, 'n': p.n, 'k': p.k, 'ak': p.ak, 'bkc': p.bkc, 'ws_ok': True,
                'out_f32': p.out_mode != 'store'}
        settings = dict(self.settings)
        if shared_limit is not None:
            settings['P8WS_SHARED_LIMIT'] = min(231000, int(shared_limit))
        with pinned(self.module, settings) as mod:
            plans = [x for x in (mod.flat_ws_plan(spec, grid, rows=r) for r in (None, 128, 64)) if x is not None]
        unique = {}
        for x in plans:
            unique.setdefault(x['bm'], x)
        if not unique:
            raise Refusal('warp-specialized plan not admitted (tile/stage/shared limits)')
        return sorted(unique.values(), key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))

    def _spec(self, p, a, b, out, plan):
        spec = super()._spec(p, a, b, out, plan)
        spec['ws_ok'] = True
        return spec

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        with pinned(self.module, self.settings) as mod:
            from megakernel_lab.operation_compiler import flat_long_attention as fla
            if out in fla.KN_EPI_TARGETS:
                raise Refusal('output registered for the attention key-norm epilogue')
            code = mod.code(self._spec(p, a, b, out, plan), plan, ident, lo, hi, den, rot)
        return Emission(code, self.headers, tiles=plan['tiles'], map_marker='P5C_TMAP')


# ------------------------------------------------------------------------------------- generic BF16 fallback
class MixedGeneric(Body):
    """mixed_gemm.emit: pinned s101h 64x64 mma.sync for aligned shapes, else the 16x16 SIMT fmaf tile
    (with the P8-E/P9-E split and skinny rules). Admits every BF16 rank-2/3 problem; the reference fallback."""
    name, module = 'mixed_generic', 'mixed_gemm'
    headers = ('tgemm.cuh', 'bgemm.cuh')

    def structural(self, p):
        if not _bf16_store(p):
            return False, 'BF16 operands and BF16 stored result only'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        aligned = not (p.m % 64 or p.n % 64 or p.k % 32) and _unit(p) and _aligned(p, 8)
        tiles = p.batch * (-(-p.m // 64) * -(-p.n // 64) if aligned else -(-p.m // 16) * -(-p.n // 16))
        return [{'form': 'pinned_bf16_64' if aligned else 'generic_bf16_simt_16', 'tiles': tiles,
                 'shared_bytes': 64 * 1024 if aligned else 17408, 'estimate_us': None}]

    def requirements(self, p, plan):
        prec = MMA_SYNC if plan['form'] == 'pinned_bf16_64' else BF16_SIMT
        return Requirements('sm_90', plan['shared_bytes'], headers=self.headers,
                            sync=Synchronization(notes='generic fallback; split forms use arrival counters'),
                            precision=_with_result(prec, p))

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        if (lo, hi, den) != (0, 1, 1):
            raise Refusal('generic fallback runs on the whole grid')
        mod = _mod(self.module)
        la, lb = p.layouts()
        la = type(la)(la.shape, la.strides, a, la.offset)
        lb = type(lb)(lb.shape, lb.strides, b, lb.offset)
        code, extra = mod.emit(a, b, out, la, lb, p.out_shape())
        return Emission(code, self.headers, tiles=extra.get('tiles', plan['tiles']), helpers=(mod.CUDA_HELPERS,),
                        metadata=extra)


# ------------------------------------------------------------------------------------------------ FP32 bodies
class FP32Pops2(Body):
    """P-OPS2-A strict FP32 engine selection over popsb/simt/slice bodies (matrix_select.plan), unsplit form.

    Split-K forms need program-level scratch binding (/*P2_PART*/, /*P2_CNT*/) and are exposed with
    ``split=True`` only through the frozen compiler's own scratch binder."""
    name, module = 'fp32_pops2', 'matrix_select'

    def structural(self, p):
        if not (p.dtype == p.out_dtype == 'float32' and p.out_mode == 'store'):
            return False, 'strict FP32 operands/result only'
        return True, ''

    def plans(self, p, grid, shared_limit=None):
        mod = _mod(self.module)
        from megakernel_lab.operation_compiler.pops_b_policy import admit
        la, lb = p.layouts()
        flags = admit(la, lb, p.m, p.n, p.k, p.batch)
        if flags is None:
            raise Refusal('operand layouts outside the unit-stride/16-byte contract')
        rows = [r for r in mod.candidates(flags, p.m, p.n, p.k, p.batch, grid) if r['ksplit'] == 1]
        if not rows:
            raise Refusal('no tile divides the product shape')
        return rows

    def requirements(self, p, plan):
        return Requirements('sm_90', plan['shared_bytes'], headers=(mod_header(plan['body']),),
                            sync=Synchronization(requires_full_grid=True, cta_subset=False,
                                                 notes='tasks dealt over the physical grid (/*B_VBASE*/ rotation)'),
                            precision=_with_result(FP32_SIMT, p))

    def emit(self, p, plan, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0):
        if (lo, hi, den) != (0, 1, 1):
            raise Refusal('P-OPS2-A bodies require the full grid')
        mod = _mod(self.module)
        la, lb = p.layouts()
        la = type(la)(la.shape, la.strides, a, la.offset)
        lb = type(lb)(lb.shape, lb.strides, b, lb.offset)
        cfg = (plan['body'], plan['bm'], plan['bn'], plan['bk'], plan['nst'], plan['slices'])
        selected, reason = mod.plan(a, b, out, la, lb, p.out_shape(), ident=ident, split=False, configs=[cfg])
        if selected is None:
            raise Refusal(reason)
        code = selected['code'].replace('/*B_VBASE*/', str(rot))
        return Emission(code, tuple(selected['headers']), tiles=selected['metadata']['tiles'],
                        helpers=(selected['helpers'],), metadata=selected['metadata'])


def mod_header(body):
    return {'simt': 'fp32_simt.cuh', 'slice': 'fp32_slice.cuh', 'popsb': 'fp32_matrix.cuh'}[body]


class Emission:
    """One emitted product: code block, headers to bind, helper text and tensor-map marker style."""

    def __init__(self, code, headers, *, tiles, helpers=(), map_marker=None, metadata=None, scratch=()):
        self.code, self.headers, self.tiles = code, tuple(headers), int(tiles)
        self.helpers, self.map_marker = tuple(helpers), map_marker
        self.metadata, self.scratch = dict(metadata or {}), tuple(scratch)

    def to_dict(self):
        return {'code': self.code, 'headers': list(self.headers), 'tiles': self.tiles,
                'helpers': list(self.helpers), 'map_marker': self.map_marker}


BODIES = (P8WS(), P5C(), Pops4bTMA(), Pops4bWG(), PopscBF16(), PopscBF16Tail(), PopscBF16Edge(),
          MixedGeneric(), FP32Pops2())
BY_NAME = {body.name: body for body in BODIES}

# Exact donor provenance (PERFORMANCE_DONORS.json at ab4703ee3).
BODY_INFO = {
    'popsc_bf16': {'planner': 'ops/matrix_rules.py:bf16_plan', 'emitter': 'ops/matrix_rules.py:_code',
                   'device': 'kernels/popsc_mm.cuh:popsc::bmm_phase', 'donor': 'af3138238 + 6082a189f + 22c3231ad'},
    'popsc_bf16_tail': {'planner': 'ops/matrix_rules.py:bf16_tail_plan', 'emitter': 'ops/matrix_rules.py:_code',
                        'device': 'kernels/popsc_mm.cuh:popsc::bmm_tail', 'donor': 'C-OPS4-B via 22c3231ad'},
    'popsc_bf16_edge': {'planner': 'ops/matrix_rules_p7b.py:bf16_plan(edge=True)',
                        'emitter': 'ops/matrix_rules_p7b.py:_code',
                        'device': 'kernels/popsc_mm.cuh:popsc::bmm_edge_phase', 'donor': 'P-OPS4-B via p7b f6f60bd1'},
    'pops4b_wg': {'planner': 'ops/matrix_rules_p7b.py:wg_plan', 'emitter': 'ops/matrix_rules_p7b.py:_code',
                  'device': 'kernels/pops4b_wg.cuh:pops4b::wmm_phase_inline', 'donor': 'P-OPS4-B 92bb997a7 via p7b f6f60bd1'},
    'pops4b_tma': {'planner': 'ops/matrix_rules_p7b.py:wg_plan', 'emitter': 'ops/matrix_rules_p7b.py:_code',
                   'device': 'kernels/pops4b_tma.cuh:pops4b::tma_phase_inline', 'donor': 'P-OPS4-B 92bb997a7 via p7b f6f60bd1'},
    'p5c': {'planner': 'ops/p5c_matrix.py:p5c_plan', 'emitter': 'ops/p5c_matrix.py:code',
            'device': 'kernels/p5c_wg.cuh:p5c::phase', 'donor': '81a52fc0 + 6082a189'},
    'p8ws': {'planner': 'ops/p5c_matrix.py:flat_ws_plan', 'emitter': 'ops/p5c_matrix.py:code',
             'device': 'kernels/p8ws.cuh:p8ws::phase_mode', 'donor': 'P8-A 25f1d13b3 via P9-A'},
    'mixed_generic': {'planner': 'ops/mixed_gemm.py:emit', 'emitter': 'ops/mixed_gemm.py:emit',
                      'device': 'vendor/bgemm.cuh (pinned) + generated 16x16 SIMT tile',
                      'donor': 'private Latch 0987d803 bgemm/tgemm'},
    'fp32_pops2': {'planner': 'ops/matrix_select.py:candidates', 'emitter': 'ops/matrix_select.py:plan',
                   'device': 'kernels/fp32_{matrix,simt,slice}.cuh', 'donor': 'P-OPS2-A 4dcee751 / 2f04ba28'},
}


def respec_tma(record, *, ident, epi, lo=0, hi=1, den=1):
    """Re-emit a provider TMA product with a staged or register-pair epilogue.

    Return (code, extra shared bytes), reserving storage only for ``staged``
    epilogues. Preserve the provider's plan, maps, K order and result rounding.
    """
    body = BY_NAME['pops4b_tma']
    with pinned(body.module, body.settings) as mod:
        depth = mod._LOWER_DEPTH[0]
        mod._LOWER_DEPTH[0] = 1
        try:
            plan = dict(record['plan'])
            staged = bool(epi.get('staged'))
            if staged and (mod.staged_alias(record['spec'], plan) or mod.staged_half(plan)):
                raise Refusal('staged epilogue tile does not fit beside the operand ring')
            code = mod._code(record['spec'], plan, ident, lo, hi, den, record['rot'], epi=epi)
            return code, mod.staging_bytes(plan) if staged else 0
        except mod.MatrixRuleError as error:
            raise Refusal(str(error))
        finally:
            mod._LOWER_DEPTH[0] = depth


def respec_tma_rowstats(record, *, ident, out, valid, lo=0, hi=1, den=1):
    """LATCH-MOESUP: re-emit a provider pops4b_tma product (``tma_respec`` record) storing its result as before and
    also writing per-(row, N tile) softmax partials of the rounded values over columns < ``valid`` to ``out``
    (float2 [M][tiles]). Returns (code, extra shared bytes, tiles). Plan, maps, k order and rounding are unchanged."""
    body = BY_NAME['pops4b_tma']
    with pinned(body.module, body.settings) as mod:
        depth = mod._LOWER_DEPTH[0]
        mod._LOWER_DEPTH[0] = 1
        try:
            plan = dict(record['plan'])
            spec = dict(record['spec'])
            if spec.get('ldc') or spec.get('batch', 1) != 1:
                raise Refusal('row softmax partials: contiguous batch-1 results only')
            tiles = -(-spec['n'] // plan['bn'])
            spec['rowstats'] = {'out': out, 'valid': int(valid), 'tiles': tiles}
            if not mod.rowstats_admitted(spec, plan):
                raise Refusal('row softmax partials not admitted for this plan')
            code = mod._code(spec, plan, ident, lo, hi, den, record['rot'])
            return code, mod.rowstats_staging(plan), tiles
        except mod.MatrixRuleError as error:
            raise Refusal(str(error))
        finally:
            mod._LOWER_DEPTH[0] = depth
