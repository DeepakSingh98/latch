# Integrated from frozen donor 81a52fc02783e0c537dcf1a909083916f2deaa66; private scheduling/resource protocol.
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
    kc = k if k <= 64 else next(c for c in (64, 32, 16) if k % c == 0)
    nk = k // kc
    nst = max(2, min(nk + 1, 4))
    candidates = []
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
        engine = p5c_matrix.admit(m, n, k, batch, sak == 1, sbk == 1, la.offset, lb.offset)
        if engine is not None:
            spec.update(engine=engine, roots=[(la.root, la.offset), (lb.root, lb.offset)])
            spec['ws_ok'] = True   # P8-A: flat P6D protocol admits the warp-specialized body
    return spec


def spec_headers(spec):
    if spec.get('engine'):
        from .p5c_matrix import wg_headers
        return wg_headers()
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
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates

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
    """Batched FP32 products whose pinned 64x64 tiling would fill under half the grid."""
    if len(la.shape) != 3:
        return False
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    if not ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)):
        return False
    if any(s % 4 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)):
        return False
    if m % 32 or n % 32 or k % 16 or k > 256:
        return False
    pinned_tiles = batch * math.ceil(m / 64) * math.ceil(n / 64)
    return pinned_tiles < grid // 2


def fp32_spec(a, b, out, la, lb, shape):
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    return {'dtype': 'float32', 'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch,
            'strides': [sam, sak, sbk, sbn, bsa, bsb], 'ak': sak == 1, 'bkc': sbk == 1}


def _plan(spec, grid):
    if spec.get('tail'):
        return bf16_tail_plan(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'], grid=grid)[0]
    if spec.get('engine'):
        from . import p5c_matrix
        return p5c_matrix.plan(spec, grid)
    args = (spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'])
    if spec['dtype'] == 'bfloat16':
        return bf16_plan(*args, grid=grid, force=None)[0]
    return fp32_plan(*args, grid=grid)[0]


def _code(spec, plan, ident, lo, hi, den, rot):
    if spec.get('engine'):
        from . import p5c_matrix
        return p5c_matrix.code(spec, plan, ident, lo, hi, den, rot)
    sam, sak, sbk, sbn, bsa, bsb = spec['strides']
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    t = lambda flag: str(flag).lower()
    if spec['dtype'] == 'bfloat16':
        call = (f'popsc::{"bmm_tail" if spec.get("tail") else "bmm_phase"}<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]}>')
        struct = 'popsc::BArgs'
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
    return {'engine': f'popsc_{"tail_bf16" if spec.get("tail") else "bf16" if bf16 else "fp32"}_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
            'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
            'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
            'arithmetic': ('explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; no split-K; '
                           'one cvt.rn.bf16x2 rounding (bitwise = pinned 64x64 body)') if bf16 else
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
        from . import p5c_matrix
        run = []   # P8-A: consecutive warp-specialized members form one continuous pipeline

        def flush():
            if len(run) > 1 or (run and p5c_matrix.P8WS_GROUP):
                blocks.append(p5c_matrix.group_code(*zip(*run), ident))
            elif run:
                blocks.append(_code(run[0][0], run[0][1], ident, 0, 1, 1, run[0][2]))
            run.clear()
        if len(specs) > 1 and p5c_matrix.P8WS_GROUP_ROWS:
            # P8-A: within a multi-product group, few-tile members keep 128-row tiles (higher operand intensity);
            # the other members occupy the CTAs a single product would leave idle.
            whole = [p5c_matrix.ws_plan(spec, grid, rows=p5c_matrix.P8WS_GROUP_ROWS) or plan
                     if plan.get('ws') and plan['bm'] != p5c_matrix.P8WS_GROUP_ROWS else plan
                     for spec, plan in zip(specs, whole)]
        for spec, plan in zip(specs, whole):
            if plan.get('ws') and p5c_matrix.P8WS_GROUP:
                if run and bool(run[-1][0].get('out_raw')) != bool(spec.get('out_raw')):
                    flush()   # LATCH-FFN-RETAIN: raw FP32 (one-slot ring) and rounded items run as separate groups
                run.append((spec, plan, rot))
            else:
                flush()
                blocks.append(_code(spec, plan, ident, 0, 1, 1, rot))
            members.append(_member_record(spec, plan, [0, 1, 1]))
            rot += plan['tiles']
        flush()
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
