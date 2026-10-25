"""P-OPS-C matrix lowering rules: tile/staging/schedule selection from operation properties.

Implementations are chosen only from dtype, rank, static M/N/K/batch, operand
layout (which stride is unit) and alignment. Nothing inspects model or value
names. The bodies live in kernels/popsc_mm.cuh and keep the pinned arithmetic:

* BF16 mm/bmm: BF16 operands, one FP32 accumulator per element from +0, one
  mma.sync m16n8k16 per k16 block in ascending k, one round-to-nearest BF16
  result. Bitwise equal to the pinned 64x64 body for every admitted shape.
* FP32 batched products: one fmaf chain per element from +0, k ascending,
  unrounded FP32 result. Bitwise equal to the pinned SIMT 64x64 body.

Selection is an H100-calibrated cost estimate over a fixed candidate set, not an
autotuner. MATRIX_POLICIES names the alternatives retained for comparison:
"tiles" (these bodies, one phase per product), "pops_c" (these bodies, and
adjacent independent products share one barrier phase, on disjoint CTA slices
when the same estimate prefers that) and "frozen".
"""
from __future__ import annotations

import math

HEADER = 'popsc_mm.cuh'
MATRIX_POLICIES = ('pops_c', 'tiles', 'frozen')

# Resident cooperative grid assumed when choosing tiles: the runtime's default
# block cap. A smaller actual grid remains correct (tasks are grid-strided and
# CTA slices are fractions of the actual grid).
TARGET_GRID = 128
# Dynamic shared memory admitted for matrix stage rings. One resident CTA per SM
# is already imposed by the cooperative kernel's register use on H100.
SHARED_LIMIT = 192 * 1024

# Cost model (microseconds, excluding the phase barrier), least-squares fitted to
# 160 in-compiler BF16 microbenchmark phases on H100 (P-OPS-C attempt gpu-02):
#   waves * (FIXED + max(stream, mma chain, mma throughput)) + operand bytes / L2.
CLOCK_MHZ = 1755
FIXED_US = 0.8                 # first-stage latency, epilogue, tile bookkeeping
SM_BYTES_PER_US = 100e3        # sustained L2->SMEM operand stream per SM
L2_BYTES_PER_US = 8e6          # aggregate operand traffic, all SMs
MMA_CHAIN_CYCLES = 34          # dependent mma.sync m16n8k16 on one accumulator
MMA_MACS_PER_CYCLE = 384       # effective mma.sync MAC rate per SM incl. fragment loads
SIMT_FMAS_PER_CYCLE = 64       # effective FP32 SIMT fmaf rate per SM incl. shared loads

# (BM, BN, WM, WN): eight warps arranged WM x WN.
BF16_TILES = ((32, 32, 2, 4), (32, 64, 2, 4), (64, 32, 4, 2), (64, 64, 2, 4),
              (64, 128, 2, 4), (128, 64, 4, 2))
FP32_TILES = ((32, 32), (32, 64), (64, 32), (64, 64))
# Release step 14: 16-multiple tiles of the same 16 x 16-thread body (popsc::stile; thread (ty, tx) owns rows ty + 16 i
# and columns tx + 16 j), at most 24 accumulators per thread. Used only where no FP32_TILES member divides M and N.
FP32_EDGE_TILES = tuple((bm, bn) for bm in (16, 32, 48, 64, 96, 128) for bn in (16, 32, 48, 64, 96, 128)
                        if (bm // 16) * (bn // 16) <= 24)
# Group schedule search (concurrent CTA slices). False is a diagnostic comparison
# arm only (grouped members back to back on the whole grid).
GROUP_PARTITION = True

class MatrixRuleError(ValueError):
    pass


def _bf16_stage_elements(bm, bn, ak, bkc, ks):
    a = bm * (ks + 8) if ak else ks * (bm + 8)
    b = bn * (ks + 8) if bkc else ks * (bn + 8)
    return a + b


def _estimate(m, n, k, batch, bm, bn, grid, element_bytes, chain_cycles, compute_rate):
    tiles = batch * (m // bm) * (n // bn)
    waves = math.ceil(tiles / grid)
    stream = (bm + bn) * k * element_bytes / SM_BYTES_PER_US
    compute = max(chain_cycles, bm * bn * k / compute_rate) / CLOCK_MHZ
    traffic = batch * (m * k * (n // bn) + k * n * (m // bm)) * element_bytes / L2_BYTES_PER_US
    return tiles, waves, waves * (FIXED_US + max(stream, compute)), traffic


def bf16_plan(m, n, k, batch, ak, bkc, *, grid=TARGET_GRID, shared_limit=SHARED_LIMIT, force=None):
    """Return the chosen BF16 tile/stage plan and every candidate's estimate."""
    if m % 32 or n % 32 or k % 32:
        raise MatrixRuleError(f'BF16 small-tile body requires M,N,K multiples of 32; got {(m, n, k)}')
    ks = 64 if k % 64 == 0 else 32
    if force is not None and force.get('ks'):
        ks = int(force['ks'])
        if k % ks:
            raise MatrixRuleError('forced k step does not divide K')
    nk = k // ks
    candidates = []
    for bm, bn, wm, wn in BF16_TILES:
        if m % bm or n % bn:
            continue
        stage_bytes = 2 * _bf16_stage_elements(bm, bn, ak, bkc, ks)
        # Measured on H100 (P-OPS-C gpu-02): a 4-deep ring beats issuing every
        # stage up front; short K slices (at most four steps) are issued whole.
        nst = min(nk + 1 if nk <= 4 else 4, shared_limit // stage_bytes)
        if force is not None and force.get('nst'):
            nst = int(force['nst'])
            if nst * stage_bytes > shared_limit:
                continue
        if nst < 2:
            continue
        mt, nt = (bm // wm) // 16, (bn // wn) // 8
        tiles, waves, local, traffic = _estimate(m, n, k, batch, bm, bn, grid, 2,
                                     (k // 16) * max(MMA_CHAIN_CYCLES, 4 + 3 * mt * nt), MMA_MACS_PER_CYCLE)
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'wn': wn, 'ks': ks, 'nst': nst, 'tiles': tiles,
                           'waves': waves, 'grid': grid, 'shared_bytes': nst * stage_bytes,
                           'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(local + traffic, 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted BF16 tile for {(m, n, k)}')
    if force is not None:
        chosen = [c for c in candidates if (c['bm'], c['bn']) == (force['bm'], force['bn'])]
        if not chosen:
            raise MatrixRuleError(f'forced tile {force} is not admitted for {(m, n, k)}')
        chosen = chosen[0]
    else:
        # Ties prefer fewer, larger tiles (less operand re-reading), then the
        # taller tile (measured faster for m-contiguous weight-gradient operands).
        chosen = min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))
    return chosen, candidates


def fp32_plan(m, n, k, batch, ak, bkc, *, grid=TARGET_GRID):
    kc = k if k <= 64 else next((c for c in (64, 32, 16) if k % c == 0), None)
    if kc is None:
        # Release step 14: long K that only a narrower slice divides (edge admission; K % 4 == 0).
        kc = next(c for c in (8, 4) if k % c == 0)
    nk = k // kc
    nst = max(2, min(nk + 1, 4))
    candidates = []
    # Release step 14: 16-multiple edge tiles only when no established tile divides the product (previously such
    # products took the generic 16 x 16 tile); established shapes keep their candidate set and choice.
    if not any(m % bm == 0 and n % bn == 0 for bm, bn in FP32_TILES):
        return fp32_edge_plan(m, n, k, batch, ak, bkc, grid=grid)
    for bm, bn in FP32_TILES:
        if m % bm or n % bn:
            continue
        stage_floats = (bm * (kc + 4) if ak else kc * (bm + 4)) + (bn * (kc + 4) if bkc else kc * (bn + 4))
        tiles, waves, local, traffic = _estimate(m, n, k, batch, bm, bn, grid, 4, 0, SIMT_FMAS_PER_CYCLE)
        candidates.append({'bm': bm, 'bn': bn, 'kc': kc, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                           'shared_bytes': 4 * nst * stage_floats, 'local_us': round(local, 4),
                           'traffic_us': round(traffic, 4), 'estimate_us': round(local + traffic, 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted FP32 tile for {(m, n, k)}')
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates


def _strides(la, lb):
    sam, sak = la.strides[-2:]
    sbk, sbn = lb.strides[-2:]
    bsa, bsb = (la.strides[0], lb.strides[0]) if len(la.shape) == 3 else (0, 0)
    return sam, sak, sbk, sbn, bsa, bsb


def _shape(la, lb, shape):
    if len(la.shape) not in (2, 3) or len(lb.shape) != len(la.shape):
        raise MatrixRuleError('mm/bmm requires equal rank-2 or rank-3 operands')
    m, k = la.shape[-2:]
    kb, n = lb.shape[-2:]
    batched = len(la.shape) == 3
    batch = la.shape[0] if batched else 1
    if k != kb or (batched and lb.shape[0] != batch) or tuple(shape) != ((batch, m, n) if batched else (m, n)):
        raise MatrixRuleError('matrix shape mismatch')
    return m, n, k, batch


def bf16_spec(a, b, out, la, lb, shape, *, engine_ok=False):
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    if not ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)):
        raise MatrixRuleError('BF16 GEMM requires contiguous/transposed matrix dimensions')
    if any(s % 8 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)):
        raise MatrixRuleError('BF16 GEMM needs 16-byte aligned leading/batch strides and offsets')
    bf16_plan(m, n, k, batch, sak == 1, sbk == 1)       # admission check
    spec = {'dtype': 'bfloat16', 'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch,
            'strides': [sam, sak, sbk, sbn, bsa, bsb], 'ak': sak == 1, 'bkc': sbk == 1}
    if engine_ok:
        # P-OPS5-C: flat fused programs may select a warpgroup-MMA engine (TMA operand maps bind storage roots).
        from . import p5c_matrix
        engine = p5c_matrix.admit(m, n, k, batch, sak == 1, sbk == 1, la.offset, lb.offset,
                                  min_tiles=p5c_matrix.WIDE_MIN_TILES if engine_ok in ('wide', 'wide_short') else None,
                                  min_k=p5c_matrix.SHORT_MIN_K_STAGES if engine_ok in ('short', 'wide_short') else None)
        if engine is not None:
            spec.update(engine=engine, roots=[(la.root, la.offset), (lb.root, lb.offset)])
    return spec


def spec_headers(spec):
    if spec.get('engine'):
        from .p5c_matrix import wg_headers
        return wg_headers()
    if spec.get('part') or spec.get('edge_fp32'):
        return ('latch_smm_split.cuh',)
    return ()



def bf16_tail_plan(m, n, k, batch, ak, bkc, *, grid=TARGET_GRID, shared_limit=SHARED_LIMIT):
    """C-OPS4-B: tile plan for BF16 products whose M, N, K are multiples of 16 but not all of 32 (bmm_tail)."""
    if m % 16 or n % 16 or k % 16:
        raise MatrixRuleError(f'BF16 tail body requires M,N,K multiples of 16; got {(m, n, k)}')
    ks = 64 if k >= 64 else 32
    nk = -(-k // ks)
    candidates = []
    for bm, bn, wm, wn in BF16_TILES:
        stage_bytes = 2 * _bf16_stage_elements(bm, bn, ak, bkc, ks)
        nst = min(nk + 1 if nk <= 4 else 4, shared_limit // stage_bytes)
        if nst < 2:
            continue
        mt, nt = (bm // wm) // 16, (bn // wn) // 8
        pm, pn, pk = -(-m // bm) * bm, -(-n // bn) * bn, nk * ks
        tiles, waves, local, traffic = _estimate(pm, pn, pk, batch, bm, bn, grid, 2,
                                     (pk // 16) * max(MMA_CHAIN_CYCLES, 4 + 3 * mt * nt), MMA_MACS_PER_CYCLE)
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'wn': wn, 'ks': ks, 'nst': nst, 'tiles': tiles,
                           'waves': waves, 'grid': grid, 'shared_bytes': nst * stage_bytes, 'tail': True,
                           'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(local + traffic, 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted BF16 tail tile for {(m, n, k)}')
    chosen = min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))
    # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'taildeep'): long contractions keep more operand bytes in flight
    # (same body, tile and k16 order; bitwise). None of this applies without the option.
    from .matrix_bodies import deepen_tail
    chosen = deepen_tail(chosen, m, n, k, ak, bkc, stage_elements=_bf16_stage_elements, shared_limit=shared_limit)
    return chosen, candidates

def bf16_tail_spec(a, b, out, la, lb, shape):
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    if not ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)):
        raise MatrixRuleError('BF16 GEMM requires contiguous/transposed matrix dimensions')
    if any(s % 8 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)):
        raise MatrixRuleError('BF16 GEMM needs 16-byte aligned leading/batch strides and offsets')
    bf16_tail_plan(m, n, k, batch, sak == 1, sbk == 1)
    return {'dtype': 'bfloat16', 'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch,
            'strides': [sam, sak, sbk, sbn, bsa, bsb], 'ak': sak == 1, 'bkc': sbk == 1, 'tail': True}

def fp32_small_applies(la, lb, shape, *, grid=TARGET_GRID):
    """Batched FP32 products whose pinned 64x64 tiling would fill under half the grid.

    LATCH-RESEARCH-EDITS: rank-2 FP32 products take only the step-14 masked edge planner (the same exact-order body
    the generic 16 x 16 tile computes), and only where no established FP32 tile divides M and N; the pinned
    small-tile rule itself stays batched-only."""
    if len(la.shape) not in (2, 3):
        return False
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    if not ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)):
        return False
    if any(s % 4 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)):
        return False
    if len(la.shape) == 2:
        return FP32_EDGE_RANK2 and fp32_edge_applies(m, n, k)
    if not (m % 32 or n % 32 or k % 16 or k > 256):
        pinned_tiles = batch * math.ceil(m / 64) * math.ceil(n / 64)
        return pinned_tiles < grid // 2
    return fp32_edge_applies(m, n, k)


FP32_EDGE_SPLITS = (1, 2, 3, 4, 6, 8)
FP32_EDGE_STEP_US = 0.3      # per staged k-step (cp.async wait + CTA barrier; measured ~0.35 us on H100, step 14)


def _fp32_kc(k):
    return k if k <= 64 else next(c for c in (64, 32, 16, 8, 4) if k % c == 0)


# Step-14 speed diagnostic ('fs<N>' flat group rules): force the FP32 edge K split when that split is admitted.
import contextvars as _contextvars
FP32_EDGE_SPLIT_FORCE = _contextvars.ContextVar('latch_fp32_edge_split_force', default=None)
# Step-14 speed ('ectaq'): FP32 edge products whose tasks are claimed from a queue next to other interval work are
# planned by the list-scheduling makespan bound, work / grid + longest task (+ aggregate traffic), instead of the
# whole-grid wave estimate: finer tasks bound the queue's tail.
FP32_EDGE_QUEUE = _contextvars.ContextVar('latch_fp32_edge_queue', default=False)
# LATCH-RESEARCH-EDITS: 'region' while a loop composer lowers a region body. Masked FP32 edge tiles are then admitted
# (the standalone planner's step-14 gate is off inside regions) without the ordered K split, whose slice partials
# live in flat-program private storage.
FP32_EDGE_SCOPE = _contextvars.ContextVar('latch_fp32_edge_scope', default=None)
# A/B switches (research arms only): rank-2 admission and region admission of the masked FP32 edge planner.
FP32_EDGE_RANK2 = True
FP32_EDGE_REGIONS = True


def fp32_edge_plan(m, n, k, batch, ak, bkc, *, grid=TARGET_GRID):
    """Release step 14: tile and K-split plan for edge-admitted FP32 batched products (popsc::stile; split > 1 runs
    smmsplit::phase plus an ordered slice reduction). The SIMT rate is scaled by the tile's register blocking: a
    thread with TM x TN accumulators issues TM + TN shared loads per TM * TN fmaf (16 x 16 tiles are load-bound)."""
    candidates = []
    for split in FP32_EDGE_SPLITS:
        # Slices of ks (a multiple of the k step kc, which divides K); the last slice may be shorter.
        kc = _fp32_kc(k)
        ks = -(-k // (split * kc)) * kc
        if (split - 1) * ks >= k or (split > 1 and ks < 128):
            continue
        nk = ks // kc
        nst = max(2, min(nk + 1, 4))
        for bm, bn in FP32_EDGE_TILES:
            masked = bool(m % bm or n % bn)
            if masked and (m % 4 or n % 4):
                continue
            stage_floats = (bm * (kc + 4) if ak else kc * (bm + 4)) + (bn * (kc + 4) if bkc else kc * (bn + 4))
            if 4 * nst * stage_floats > SHARED_LIMIT:
                continue
            tm, tn = bm // 16, bn // 16
            efficiency = min(1.0, tm * tn / (tm + tn))
            tiles = split * batch * math.ceil(m / bm) * math.ceil(n / bn)
            waves = math.ceil(tiles / grid)
            stream = (bm + bn) * ks * 4 / SM_BYTES_PER_US
            compute = bm * bn * ks / (SIMT_FMAS_PER_CYCLE * efficiency) / CLOCK_MHZ
            local = waves * (FIXED_US + max(stream, compute) + nk * FP32_EDGE_STEP_US)
            traffic = split * batch * (m * ks * (n // bn) + ks * n * (m // bm)) * 4 / L2_BYTES_PER_US
            reduce = (FIXED_US + 2 * split * batch * m * n * 4 / L2_BYTES_PER_US) if split > 1 else 0.0
            candidates.append({'bm': bm, 'bn': bn, 'kc': kc, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                               'split': split, 'ks': ks, 'masked': masked, 'shared_bytes': 4 * nst * stage_floats, 'local_us': round(local, 4),
                               'traffic_us': round(traffic + reduce, 4),
                               'estimate_us': round(local + traffic + reduce, 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted FP32 edge tile for {(m, n, k)}')
    force = FP32_EDGE_SPLIT_FORCE.get()
    if FP32_EDGE_SCOPE.get() == 'region':
        force = 1
    pool = [c for c in candidates if c['split'] == force] if force else []
    if FP32_EDGE_QUEUE.get():
        for c in candidates:
            task = c['local_us'] / max(1, c['waves'])
            c['queue_us'] = round(c['tiles'] * task / grid + task + c['traffic_us'], 4)
        return min(pool or candidates, key=lambda c: (c['queue_us'], c['split'], c['tiles'], -c['bm'])), candidates
    return min(pool or candidates, key=lambda c: (c['estimate_us'], c['split'], c['tiles'], -c['bm'])), candidates


def _step14():
    from . import compose
    return compose.step14_flat()


def fp32_edge_applies(m, n, k):
    """Release step 14: batched FP32 products that neither the established small tile nor the pinned 64 x 64 tile
    admits (M or N not a multiple of 64, or the small tile's 32-multiple/K <= 256 bounds unmet) and that the
    generic 16 x 16 tile would otherwise run. The same exact-order body (ascending-k fmaf per element from +0,
    cp.async operand ring) on 16-multiple tiles; bitwise equal to the generic tile's per-element order."""
    if m % 4 or n % 4 or k % 4 or not (_step14() or (FP32_EDGE_REGIONS and FP32_EDGE_SCOPE.get() == 'region')):
        return False
    return bool(m % 64 or n % 64)


def fp32_spec(a, b, out, la, lb, shape):
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    spec = {'dtype': 'float32', 'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch,
            'strides': [sam, sak, sbk, sbn, bsa, bsb], 'ak': sak == 1, 'bkc': sbk == 1}
    if not any(m % bm == 0 and n % bn == 0 for bm, bn in FP32_TILES):
        spec['edge_fp32'] = True          # release step 14 edge planner (may emit smmsplit:: bodies)
    return spec


def _plan(spec, grid):
    # LATCH-STEP19-SPEED: a dynamic co-scheduling decision may plan a product for part of the grid
    # (leaving CTAs to bounded reductions in the same phase). Static specs never carry plan_grid.
    grid = min(grid, spec.get('plan_grid', grid))
    if spec.get('tail'):
        return bf16_tail_plan(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'], grid=grid)[0]
    if spec.get('engine'):
        from . import p5c_matrix
        return p5c_matrix.plan(spec, grid)
    args = (spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'])
    if spec['dtype'] == 'bfloat16':
        chosen = bf16_plan(*args, grid=grid, force=None)[0]
        if spec.get('split'):                       # LATCH-STEP19-SPEED dynamic split-K: tasks = tiles x slices
            chosen = {**chosen, 'tiles': chosen['tiles'] * spec['split'], 'split': spec['split']}
        return chosen
    return fp32_plan(*args, grid=grid)[0]


def _code(spec, plan, ident, lo, hi, den, rot):
    if spec.get('engine'):
        from . import p5c_matrix
        return p5c_matrix.code(spec, plan, ident, lo, hi, den, rot)
    sam, sak, sbk, sbn, bsa, bsb = spec['strides']
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    t = lambda flag: str(flag).lower()
    if spec.get('active'):
        # LATCH-STEP19-SPEED: bounded dynamic batch. The capacity plan's tile/stage geometry with the
        # existing masked-edge body: runtime M (active rows) or K (active reduction extent); tiles that
        # cross the extent zero-fill absent operands without loading them and store only active outputs.
        active = spec['active']
        M = 'latch_active_batch' if active['m'] else m
        K = 'latch_active_batch' if active['k'] else k
        if plan.get('split'):
            call = (f'latch_dyn::bmm_phase2e_sk<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                    f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]},{plan["split"]},true,{t(n % 2 == 0)}>')
            return ('{\n'
                    f'  popsc::BArgs g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {bsa}LL, {ident(spec["b"])}, {sbk}LL, {sbn}LL, {bsb}LL,\n'
                    f'              {spec["out"]}, {n}LL, {m*n}LL, {M}, {n}, {K}, {batch}}};\n'
                    f'  {call}(g, smem, latch_dyn::NoEpiRows{{}}, latch_dyn::NoProE{{}}, latch_dyn::NoProE{{}}, '
                    f'latch_dyn::ProArgs{{0, 0, nullptr, nullptr}}, {lo}, {hi}, {den}, {rot}, '
                    f'/*P2_PART:{spec["out"]}*/, /*P2_CNT:{spec["out"]}*/);\n'
                    '}')
        call = (f'popsc::bmm_edge_phase<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]},{t(n % 2 == 0)}>')
        return ('{\n'
                f'  popsc::BArgs g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {bsa}LL, {ident(spec["b"])}, {sbk}LL, {sbn}LL, {bsb}LL,\n'
                f'              {spec["out"]}, {n}LL, {m*n}LL, {M}, {n}, {K}, {batch}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec['dtype'] == 'bfloat16':
        call = (f'popsc::{"bmm_tail" if spec.get("tail") else "bmm_phase"}<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]}>')
        struct = 'popsc::BArgs'
    elif plan.get('masked'):
        # Release step 14: masked-edge FP32 tile (any 4-multiple M, N), optional ordered K split.
        split = plan.get('split', 1)
        return ('{\n'
                f'  popsc::SArgs g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {bsa}LL, {ident(spec["b"])}, {sbk}LL, {sbn}LL, {bsb}LL,\n'
                f'              {"nullptr" if split > 1 else spec["out"]}, {n}LL, {m*n}LL, {m}, {n}, {k}, {batch}}};\n'
                f'  smmsplit::edge_phase<{plan["bm"]},{plan["bn"]},{t(spec["ak"])},{t(spec["bkc"])},{plan["kc"]},{plan["nst"]}>'
                f'(g, {spec["part"] if split > 1 else "nullptr"}, {split}, {plan.get("ks", k)}, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    elif plan.get('split', 1) > 1:
        # Release step 14: K-sliced FP32 edge product; partials to spec['part'], reduced by a separate phase.
        return ('{\n'
                f'  popsc::SArgs g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {bsa}LL, {ident(spec["b"])}, {sbk}LL, {sbn}LL, {bsb}LL,\n'
                f'              nullptr, {n}LL, {m*n}LL, {m}, {n}, {k}, {batch}}};\n'
                f'  smmsplit::phase<{plan["bm"]},{plan["bn"]},{t(spec["ak"])},{t(spec["bkc"])},{plan["kc"]},{plan["nst"]}>'
                f'(g, {spec["part"]}, {plan["split"]}, {plan["ks"]}, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    else:
        call = f'popsc::smm_phase<{plan["bm"]},{plan["bn"]},{t(spec["ak"])},{t(spec["bkc"])},{plan["kc"]},{plan["nst"]}>'
        struct = 'popsc::SArgs'
    return ('{\n'
            f'  {struct} g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {bsa}LL, {ident(spec["b"])}, {sbk}LL, {sbn}LL, {bsb}LL,\n'
            f'              {spec["out"]}, {n}LL, {m*n}LL, {m}, {n}, {k}, {batch}}};\n'
            f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
            '}')


def _member_record(spec, plan, slice_):
    if spec.get('engine'):
        from . import p5c_matrix
        return p5c_matrix.member_record(spec, plan, slice_)
    bf16 = spec['dtype'] == 'bfloat16'
    if spec.get('active'):
        return {'engine': f'popsc_bf16_edge_active_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
                'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
                'cta_slice': slice_, 'active_extents': dict(spec['active']),
                'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
                'arithmetic': ('explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; absent '
                               'operands zero-filled without loads; one cvt.rn.bf16x2 rounding; active outputs only')}
    return {'engine': f'popsc_{"tail_bf16" if spec.get("tail") else "bf16" if bf16 else "fp32"}_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
            'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
            'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
            'arithmetic': ('explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; no split-K; '
                           'one cvt.rn.bf16x2 rounding (bitwise = pinned 64x64 body)') if bf16 else
                          (f'FP32 SIMT fmaf chain from +0, ascending k within each of {plan["split"]} contiguous K '
                           'slices; slices added in ascending order (ordered FP32 split-K); no TF32')
                          if plan.get('split', 1) > 1 else
                          'FP32 SIMT fmaf chain from +0, ascending k; no TF32 (bitwise = pinned SIMT 64x64 body)'}


def emit_group(specs, *, ident, grid=TARGET_GRID, partition=None):
    """Emit one matrix phase executing every spec (mutually independent products).

    A single product uses the whole grid. Several products either run back to back
    on the whole grid or concurrently on disjoint CTA slices (fractions of the actual
    grid), whichever the cost model estimates faster; each slice's tile is planned
    for that slice. Partitions are searched in steps of grid/16.
    """
    partition = GROUP_PARTITION if partition is None else partition
    whole = [_plan(spec, grid) for spec in specs]
    best = ('serial', sum(p['estimate_us'] for p in whole), None)
    if partition and 2 <= len(specs) <= 3:
        step = grid // 16
        shares = range(step, grid, step)
        options = ([(g, grid - g) for g in shares] if len(specs) == 2 else
                   [(g1, g2, grid - g1 - g2) for g1 in shares for g2 in shares if g1 + g2 < grid])
        for option in options:
            plans = [_plan(spec, g) for spec, g in zip(specs, option)]
            # Concurrent members share the aggregate L2 operand bandwidth.
            us = max(p['local_us'] for p in plans) + sum(p['traffic_us'] for p in plans)
            if us < best[1]:
                best = ('partition', us, (option, plans))
    blocks, members, rot = [], [], 0
    if best[0] == 'serial':
        # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'tailrot'): a single native tail product starts on the interval's
        # rotated CTA offset (a token replaced at emission; 0 without the option).
        from .matrix_bodies import tail_rotation_token
        token = tail_rotation_token(specs[0]) if len(specs) == 1 else 0
        for spec, plan in zip(specs, whole):
            blocks.append(_code(spec, plan, ident, 0, 1, 1, token or rot))
            members.append(_member_record(spec, plan, [0, 1, 1]))
            rot += plan['tiles']
    else:
        option, plans = best[2]
        lo = 0
        for spec, plan, g in zip(specs, plans, option):
            blocks.append(_code(spec, plan, ident, lo, lo + g, grid, 0))
            members.append(_member_record(spec, plan, [lo, lo + g, grid]))
            lo += g
    engine = members[0]['engine'] if len(members) == 1 else f'popsc_group_{best[0]}'
    extra = {'tiles': sum(m['matrix_plan']['tiles'] for m in members), 'engine': engine,
             'batch': members[0]['batch'], 'shared_bytes': max(m['matrix_plan']['shared_bytes'] for m in members),
             'members': members, 'group_schedule': best[0], 'group_estimate_us': round(best[1], 4),
             'matrix_shape': members[0]['matrix_shape'], 'matrix_plan': members[0]['matrix_plan'],
             'operand_layout': members[0]['operand_layout'], 'arithmetic': members[0]['arithmetic']}
    return '\n'.join(blocks), extra
