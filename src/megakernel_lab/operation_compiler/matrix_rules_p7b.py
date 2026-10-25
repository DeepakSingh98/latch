# Integrated from frozen donor f6f60bd1325cae4ded5fba17083feb3b85a791b0; private scheduling/resource protocol.
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
# P-OPS4-B diagnostic: larger eight-warp tiles (warp tiles up to 64x64) for large products. Off by default.
BF16_LARGE_TILES = ((128, 128, 2, 4), (128, 256, 2, 4), (256, 128, 4, 2))
LARGE_TILES = False
# P-OPS4-B warpgroup-MMA bodies (kernels/pops4b_tma.cuh; cp.async form kernels/pops4b_wg.cuh). Selected default (gpu-07):
# masked-edge BF16 products of at least WG_MIN_FLOPS use the entry-resident TMA body with 16-byte result stores; nested
# region lowerings and batched products use the cp.async form. WGMMA_ALIGNED (aligned products) stays a diagnostic arm.
WGMMA = True
WGMMA_ALIGNED = False
WG_INLINE = True
WG_FLAT = False
WG_TMA = True               # TMA staging (tensor maps encoded by the Runtime) instead of cp.async; batch 1 only
WG_VEC_STORE = True         # TMA body: quad-transposed 16-byte result stores when N % 8 == 0
WG_DIAG = 0                 # timing diagnostics only (wrong results): 1 = no main-loop loads, 2 = no wgmma
WG_MIN_FLOPS = 2 ** 30
WG_HEADERS = ('s107a_wgmma.cuh', 'pops4b_wg.cuh')
# (BM, BN, WM): WM = 2 splits M between the two warpgroups, WM = 1 splits N.
WG_TILES = ((128, 256, 2), (128, 192, 2), (128, 128, 2), (128, 64, 2), (64, 256, 1), (64, 128, 1))
# Fitted to the P-OPS4-B TMA body on H100 (gpu-07: weight-gradient 128x256 runs at the tensor bound on 70 CTAs;
# 128x192 / 64x256 and the 14-wave forward product are bound by aggregate L2->SM operand traffic of ~7 TB/s).
WG_SM_FLOPS_PER_US = 7.8e6      # sustained dense BF16 wgmma per SM
WG_L2_BYTES_PER_US = 7.0e6      # aggregate operand stream, all active CTAs
WG_STORE_BYTES_PER_US = 2.6e6   # aggregate BF16 result stores
WG_FIXED_US = 1.0
# Group schedule search (concurrent CTA slices). False is a diagnostic comparison
# arm only (grouped members back to back on the whole grid).
GROUP_PARTITION = True

# P-OPS4-B: BF16 products whose M, N or K is not a multiple of 32 use the masked-edge variant of the same body
# (zero-filled absent operand elements, masked stores) instead of the generic 16-wide SIMT fallback. False is a
# diagnostic comparison arm only.
EDGE_TAILS = True
# Diagnostic-only plan overrides keyed by output identifier (microbenchmarks); empty in every compiler use.
FORCE_BY_OUTPUT: dict = {}


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


def _bf16_tiles():
    return BF16_TILES + (BF16_LARGE_TILES if LARGE_TILES else ())


def _ceil(x, q):
    return -(-x // q) * q


def bf16_plan(m, n, k, batch, ak, bkc, *, grid=None, shared_limit=None, force=None, edge=False):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
    shared_limit = SHARED_LIMIT if shared_limit is None else shared_limit   # read at call time (scoped overrides)
    """Return the chosen BF16 tile/stage plan and every candidate's estimate.

    edge=True plans the masked-edge body: ceil-divided tiles, ceil(K/KS) stages, estimates on padded extents."""
    if edge:
        return _bf16_edge_plan(m, n, k, batch, ak, bkc, grid=grid, shared_limit=shared_limit, force=force)
    if m % 32 or n % 32 or k % 32:
        raise MatrixRuleError(f'BF16 small-tile body requires M,N,K multiples of 32; got {(m, n, k)}')
    ks = 64 if k % 64 == 0 else 32
    if force is not None and force.get('ks'):
        ks = int(force['ks'])
        if k % ks:
            raise MatrixRuleError('forced k step does not divide K')
    nk = k // ks
    candidates = []
    for bm, bn, wm, wn in _bf16_tiles():
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


def _bf16_edge_plan(m, n, k, batch, ak, bkc, *, grid, shared_limit, force):
    # The K step wastes the least padding (ties keep the longer step).
    ks = 64 if _ceil(k, 64) == _ceil(k, 32) else 32
    if force is not None and force.get('ks'):
        ks = int(force['ks'])
    nk = -(-k // ks)
    kp = nk * ks
    candidates = []
    for bm, bn, wm, wn in _bf16_tiles():
        stage_bytes = 2 * _bf16_stage_elements(bm, bn, ak, bkc, ks)
        nst = min(nk + 1 if nk <= 4 else 4, shared_limit // stage_bytes)
        if force is not None and force.get('nst'):
            nst = int(force['nst'])
            if nst * stage_bytes > shared_limit:
                continue
        if nst < 2:
            continue
        mt, nt = (bm // wm) // 16, (bn // wn) // 8
        mp, np_ = _ceil(m, bm), _ceil(n, bn)
        tiles, waves, local, traffic = _estimate(mp, np_, kp, batch, bm, bn, grid, 2,
                                     (kp // 16) * max(MMA_CHAIN_CYCLES, 4 + 3 * mt * nt), MMA_MACS_PER_CYCLE)
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'wn': wn, 'ks': ks, 'nst': nst, 'tiles': tiles,
                           'waves': waves, 'grid': grid, 'shared_bytes': nst * stage_bytes, 'edge': True,
                           'padded': [mp, np_, kp], 'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(local + traffic, 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted BF16 masked-edge tile for {(m, n, k)}')
    if force is not None and force.get('bm'):
        chosen = [c for c in candidates if (c['bm'], c['bn']) == (force['bm'], force['bn'])]
        if not chosen:
            raise MatrixRuleError(f'forced tile {force} is not admitted for {(m, n, k)}')
        return chosen[0], candidates
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates


def wg_plan(m, n, k, batch, ak, bkc, *, grid=None, shared_limit=None, force=None):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
    shared_limit = SHARED_LIMIT if shared_limit is None else shared_limit   # read at call time (scoped overrides)
    """warpgroup-MMA tile/stage plan (64-wide k stages, ceil-divided tiles, estimates on padded extents)."""
    nk = -(-k // 64)
    kp = 64 * nk
    candidates = []
    for bm, bn, wm in WG_TILES:
        wtn = bn // (2 // wm)
        if not bkc and wtn % 64:
            continue                       # N-major B: whole 64-column swizzle chunks per warpgroup
        stage_bytes = (bm + bn) * 128
        reserve = bm * (bn + 8) * 2 if P9E_STAGING_RESERVE else 0   # P9-E: room for a staged epilogue tile
        nst = min(4, nk + 1, (shared_limit - 1024 - reserve) // stage_bytes)
        if P9E_RING_RULE and nk < 16:
            nst = min(nst, 3)      # P9-E: a fourth stage pays off only for long K (matrix region bench, gpu-06a)
        if force is not None and force.get('nst'):
            nst = int(force['nst'])
            if nst * stage_bytes + 1024 > shared_limit:
                continue
        if nst < 2:
            continue
        mp, np_ = _ceil(m, bm), _ceil(n, bn)
        tiles = batch * (mp // bm) * (np_ // bn)
        waves = math.ceil(tiles / grid)
        active = min(grid, tiles)
        compute = 2 * bm * bn * kp / WG_SM_FLOPS_PER_US
        store = bm * bn * 2 * active / WG_STORE_BYTES_PER_US
        local = waves * (WG_FIXED_US + compute + store)
        traffic = tiles * (bm + bn) * kp * 2 / WG_L2_BYTES_PER_US
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                           'shared_bytes': nst * stage_bytes + 1024, 'padded': [mp, np_, kp], 'wg': True,
                           'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(max(local, traffic), 4)})
    if not candidates:
        raise MatrixRuleError(f'no admitted warpgroup-MMA tile for {(m, n, k)}')
    if force is not None and force.get('bm'):
        chosen = [c for c in candidates if (c['bm'], c['bn']) == (force['bm'], force['bn'])
                  and ('wm' not in force or c['wm'] == force['wm'])]
        if not chosen:
            raise MatrixRuleError(f'forced tile {force} is not admitted for {(m, n, k)}')
        return chosen[0], candidates
    if P9E_WIDE_TILES and batch == 1 and n >= 1024 and n % 128 == 0 and (not P9E_WIDE_ONLY or m >= 4096):
        # P9-E: wide outputs take 128 x 128 tiles (two 64 x 128 warpgroups; region bench: fc1 / fc2-dX 36.7 -> 29.3 us)
        wide = [c for c in candidates if (c['bm'], c['bn'], c['wm']) == (128, 128, 2)]
        if wide:
            return wide[0], candidates
    if P9E_TALL_TILES and batch == 1 and n % 64 == 0 and n <= 512 and m >= 4096:
        # P9-E: narrow outputs over many rows take 256 x 64 tiles (two 128 x 64 warpgroups): A is shared by twice the
        # rows per B load (operand stream -17 % against 128 x 64) and M = 12,544 gives 49 x N / 64 tasks.
        tall = [c for c in candidates if (c['bm'], c['bn'], c['wm']) == (256, 64, 2)]
        if tall:
            return tall[0], candidates
    if P9E_WIDE_ONLY:
        # P9-E: 128-column tiles only through the wide rule above (row-heavy wide outputs); every other product keeps
        # its previous candidates, so adding the tile does not change unrelated plans or split choices.
        narrow = [c for c in candidates if c['bn'] < 128]
        if narrow:
            return min(narrow, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates


def spec_headers(spec):
    if spec.get('tma'):
        return WG_HEADERS + ('pops4b_tma.cuh',)
    return WG_HEADERS if spec.get('wg') else ()


# Tensor maps are finalized only by the outermost lower() call's own fused/serial program; nested lowerings (loop,
# inference and conditional region bodies spliced into a parent program) emit the cp.async wgmma body instead.
_LOWER_DEPTH = [0]


def lowering_scope(fn):
    import functools

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        _LOWER_DEPTH[0] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _LOWER_DEPTH[0] -= 1
    return wrapped


def tma_allowed():
    return _LOWER_DEPTH[0] <= 1


TMAP_MARK = 'POPS4B_TMAP'


def _tmap_marker(root, offset, kmajor, rows, k, stride, box_rows):
    """Placeholder for the tensor map of one operand view, finalized by collect_tensor_maps (K-major: dims {K, rows},
    box {64, box_rows}; M/N-major: dims {rows, K}, box {64, 64}; BF16, 128-byte swizzle, OOB zero fill)."""
    dims, box = ((k, rows), (64, box_rows)) if kmajor else ((rows, k), (64, 64))
    return f'{TMAP_MARK}({root},{offset},{dims[0]},{dims[1]},{2 * stride},{box[0]},{box[1]})'


def collect_tensor_maps(source, n_inputs):
    """Replace tensor-map placeholders by entries of the map table passed as args[n_inputs]; return the table spec."""
    import re
    maps, index = [], {}

    def replace(match):
        key = match.group(1)
        if key not in index:
            root, offset, d0, d1, stride, b0, b1, *rest = key.split(',')
            index[key] = len(maps)
            maps.append({'root': root, 'offset': int(offset), 'dims': [int(d0), int(d1)], 'stride_bytes': int(stride),
                         'box': [int(b0), int(b1)], 'dtype': 'bfloat16', 'swizzle': int(rest[0]) if rest else 128, 'oob_fill': 'zero'})
        return f'(reinterpret_cast<const char*>(args[{n_inputs}]) + {128 * index[key]})'
    return re.sub(TMAP_MARK + r'\(([^()]*)\)', replace, source), maps


def fp32_plan(m, n, k, batch, ak, bkc, *, grid=None):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
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


def bf16_spec(a, b, out, la, lb, shape):
    m, n, k, batch = _shape(la, lb, shape)
    sam, sak, sbk, sbn, bsa, bsb = _strides(la, lb)
    if not ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)):
        raise MatrixRuleError('BF16 GEMM requires contiguous/transposed matrix dimensions')
    if any(s % 8 for s in (sam if sak == 1 else sak, sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)):
        raise MatrixRuleError('BF16 GEMM needs 16-byte aligned leading/batch strides and offsets')
    spec = {'dtype': 'bfloat16', 'a': a, 'b': b, 'out': out, 'm': m, 'n': n, 'k': k, 'batch': batch,
            'strides': [sam, sak, sbk, sbn, bsa, bsb], 'ak': sak == 1, 'bkc': sbk == 1}
    try:
        bf16_plan(m, n, k, batch, sak == 1, sbk == 1)       # admission check
    except MatrixRuleError:
        if not EDGE_TAILS or not (m % 32 or n % 32 or k % 32):
            raise
        bf16_plan(m, n, k, batch, sak == 1, sbk == 1, edge=True)
        spec.update(edge=True, pair=n % 2 == 0)
    if (WGMMA and (spec.get('edge') or WGMMA_ALIGNED) and 2 * m * n * k * batch >= WG_MIN_FLOPS):
        wg_plan(m, n, k, batch, sak == 1, sbk == 1)
        spec.update(wg=True, pair=n % 2 == 0)
        if WG_TMA and batch == 1:
            # Maps bind to the operand views themselves (the Runtime resolves their final storage and offset).
            spec['tma'] = {'a': (a, 0), 'b': (b, 0)}
    return spec


def fp32_small_applies(la, lb, shape, *, grid=None):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
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


def split_choice(spec, grid=None, gain=0.8):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
    """P6-B: K-slice count for a long-K, small-output BF16 product (deterministic split-K), or None.

    The slices run as a batched product over K (FP32 slice results); the choice is the modeled-fastest slice count
    whose estimate (plus the slice-sum read) is below `gain` x the unsplit estimate."""
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    if (spec['dtype'] != 'bfloat16' or batch != 1 or spec.get('edge') or spec.get('wg') or k < 2048
            or m % 32 or n % 32):
        return None
    try:
        base = bf16_plan(m, n, k, 1, spec['ak'], spec['bkc'], grid=grid)[0]['estimate_us']
    except MatrixRuleError:
        return None
    best = None
    for split in (2, 3, 4, 6, 7, 8, 12, 14, 16):
        if k % (32 * split) or k // split < 256:
            continue
        try:
            plan = bf16_plan(m, n, k // split, split, spec['ak'], spec['bkc'], grid=grid)[0]
        except MatrixRuleError:
            continue
        estimate = plan['estimate_us'] + split * m * n * 4 / L2_BYTES_PER_US
        if estimate < gain * base and (best is None or estimate < best[1]):
            best = (split, estimate)
    return best[0] if best else None


def wg_split_choice(spec, grid=None, gain=0.8, min_slice=256, edge=False, objective='model'):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
    """P6-B: K-slice count for a long-K warpgroup (TMA) product: slices run on the TMA wgmma body as (slice, tile)
    tasks writing FP32 slice results; chosen when the modeled estimate (plus the slice-sum read) is below `gain` x
    the unsplit warpgroup estimate. Slices are whole 64-wide k stages."""
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    if not (spec.get('wg') and spec.get('tma')) or (spec.get('edge') and not edge) or batch != 1 or k < 2048 or n % 2:
        return None
    try:
        base = wg_plan(m, n, k, 1, spec['ak'], spec['bkc'], grid=grid)[0]['estimate_us']
    except MatrixRuleError:
        return None
    best = None
    for split in range(2, 65):
        if k % (64 * split) or k // split < min_slice:
            continue
        try:
            plan = wg_plan(m, n, k // split, split, spec['ak'], spec['bkc'], grid=grid)[0]
        except MatrixRuleError:
            continue
        estimate = plan['estimate_us'] + split * m * n * 4 / L2_BYTES_PER_US
        if objective == 'p9e':
            # P9-E: busiest-CTA k steps at the measured per-CTA step time plus the FP32 slice results written and read
            # back by the consumer (charged at HBM rate); the unsplit body is the reference.
            if not (k // split) % 64:
                lat = (plan['waves'] * ((k // split) // 64 + P7B_LPT_TASK_US) * P8E_STEP_US
                       + 2 * split * m * n * 4 / P9E_SLICE_BYTES_PER_US)
                if best is None or lat < best[1]:
                    best = (split, lat)
            continue
        if objective == 'steps':
            # P6-B: latency objective: 64-wide k steps on the busiest CTA (waves x steps per slice), then the
            # modeled estimate; the (traffic) model must not predict a loss against the unsplit body.
            if estimate < base:
                key = (plan['waves'] * (-(-(k // split) // 64)), estimate)
                if best is None or key < best[1]:
                    best = (split, key)
            continue
        if estimate < gain * base and (best is None or estimate < best[1]):
            best = (split, estimate)
    if objective == 'p9e':
        unsplit = wg_plan(m, n, k, 1, spec['ak'], spec['bkc'], grid=grid)[0]
        base_lat = unsplit['waves'] * (-(-k // 64) + P7B_LPT_TASK_US) * P8E_STEP_US
        return best[0] if best and best[1] < base_lat else None
    if best is None and P8E_LATENCY_SPLIT:
        # P8-E: the traffic model above assumes one CTA streams a long K at the SM's full tensor rate. With few tiles
        # (one partial wave) each CTA runs its whole K serially at roughly P8E_STEP_US per 64-wide k step (measured).
        # When no split was chosen, split if the busiest CTA's modeled time (waves x (k steps + fixed per-task
        # cost)) falls to at most half; among splits within 10% of the best, the fewest slices (FP32 slice memory).
        unsplit = wg_plan(m, n, k, 1, spec['ak'], spec['bkc'], grid=grid)[0]
        base_lat = unsplit['waves'] * (-(-k // 64) + P7B_LPT_TASK_US) * P8E_STEP_US
        cands = []
        for split in range(2, 65):
            if k % (64 * split) or k // split < min_slice:
                continue
            try:
                plan = wg_plan(m, n, k // split, split, spec['ak'], spec['bkc'], grid=grid)[0]
            except MatrixRuleError:
                continue
            lat = (plan['waves'] * (-(-(k // split) // 64) + P7B_LPT_TASK_US) * P8E_STEP_US
                   + split * m * n * 4 / L2_BYTES_PER_US)
            if lat <= 0.5 * base_lat:
                cands.append((split, lat))
        if cands:
            low = min(lat for _, lat in cands)
            best = min((c for c in cands if c[1] <= 1.1 * low), key=lambda c: c[0])
    return best[0] if best else None


P9E_SLICE_BYTES_PER_US = 3.0e6   # P9-E 'p9e' split objective: HBM-rate charge for FP32 slice results (write + read)
P9E_STAGING_RESERVE = False
P9E_TALL_TILES = False       # P9-E: 256 x 64 warpgroup tiles for unsplit products with N <= 512 (needs (256, 64, 2) in WG_TILES)
P9E_WIDE_TILES = False       # P9-E: 128 x 128 warpgroup tiles for unsplit products with N >= 1024, N % 128 == 0
P9E_WIDE_ONLY = False        # P9-E: 128-column tiles only via the wide rule, and only for row-heavy outputs (M >= 4096)
P9E_RING_RULE = False        # P9-E: four-stage operand rings only for K >= 1024  # P9-E: warpgroup stage counts leave room for the staged epilogue tile (large tiles)
P8E_LATENCY_SPLIT = False   # P8-E: latency fallback for long-K products the traffic model leaves unsplit
P8E_STEP_US = 1.0           # P8-E: per-CTA time of one 64-wide k step (measured 0.75-2.1 us on H100 in this program)
P7B_CLAIM = False           # P7-B: TMA products claim tasks from a per-launch counter (pops4b TArgs.claim)
P7B_LPT = False             # P7-B: cost-balanced static task tables for serial groups of TMA products
P7B_LPT_GRID = 264          # resident grid the tables are built for (2 x 132 CTAs); other grids use round robin
P7B_LPT_TASK_US = 2.0       # modeled fixed cost per task (TMA latency, epilogue) in 64-wide k-step units


def lpt_tables(specs, plans, grid=None):
    """P7-B: longest-processing-time assignment of every task of the group's TMA products to the resident grid's CTAs
    (cost = 64-wide k steps of the task's K range + a fixed per-task term; ties: lowest load, then lowest CTA).
    Returns {spec index: (table name, C definition)}; products outside the TMA body keep their round robin."""
    import hashlib
    import heapq
    grid = P7B_LPT_GRID if grid is None else grid
    tasks = []
    for si, (spec, plan) in enumerate(zip(specs, plans)):
        if not (spec.get('tma') and tma_allowed()):
            continue
        split = spec.get('split') or 1
        steps = -(-(spec['k'] // split) // 64)
        for t in range(plan['tiles'] if not spec.get('split') else plan['tiles']):
            tasks.append((steps + P7B_LPT_TASK_US, si, t))
    if len({si for _, si, _ in tasks}) < 2:
        return {}
    tasks.sort(key=lambda x: (-x[0], x[1], x[2]))
    heap = [(0.0, c) for c in range(grid)]
    heapq.heapify(heap)
    lists = {}
    for cost, si, t in tasks:
        load, c = heapq.heappop(heap)
        lists.setdefault(si, [[] for _ in range(grid)])[c].append(t)
        heapq.heappush(heap, (load + cost, c))
    out = {}
    for si, per in lists.items():
        offsets, ids = [0], []
        for c in range(grid):
            ids.extend(per[c])
            offsets.append(len(ids))
        if len(ids) >= 65536:
            continue
        values = offsets + ids
        body = ", ".join(str(v) for v in values)
        name = "p7b_lpt_" + hashlib.sha256(body.encode()).hexdigest()[:16]
        out[si] = (name, f"__device__ const unsigned short {name}[] = {{{body}}};")
    return out


# P8-E: only split-K products (K-slice tasks of weight-gradient shape) claim their tasks, and they run after the
# statically assigned products of their serial group, so CTAs that finish their static tiles early take more slices.
P8E_SPLIT_CLAIM = False


def claim_order(specs):
    """P7-B: serial-group emission order; in claim mode the products with the longest tasks run first (LPT)."""
    if not P7B_CLAIM:
        if P8E_SPLIT_CLAIM:
            return sorted(range(len(specs)), key=lambda i: (_claims(specs[i]), i))
        return list(range(len(specs)))
    steps = lambda sp: -(-(sp['k'] // (sp.get('split') or 1)) // 64)
    return sorted(range(len(specs)), key=lambda i: (-steps(specs[i]), i))


def _claims(spec):
    """P7-B claim mode for a TMA product with at least two 64-wide k steps per task (P8-E: split-K products only)."""
    return ((P7B_CLAIM or (P8E_SPLIT_CLAIM and bool(spec.get('split')) and spec.get('tma') and tma_allowed()))
            and -(-(spec['k'] // (spec.get('split') or 1)) // 64) >= 2)


P7B_ALIAS = False           # P7-B: staged epilogues alias the drained operand ring when ring + staging exceed the budget


def staged_alias(spec, plan, table=None):
    """P7-B: does this staged TMA product stage its tile in the drained operand ring (pops4b ALIAS)?"""
    return (P7B_ALIAS and not plan.get('rows') and plan['shared_bytes'] + staging_bytes(plan) > SHARED_LIMIT
            and staging_bytes(plan) <= plan['nst'] * (plan['bm'] + plan['bn']) * 128 and table is None
            and not _claims(spec))


P9E_HALF_STAGING = False    # P9-E: staged epilogues of 128-column tiles stage two 64-column halves when a whole tile
                            # does not fit beside the operand ring (pops4b HALF; unaliased)


def staged_half(plan):
    """P9-E: does this staged TMA product stage its tile in two column halves (pops4b HALF)?"""
    full = plan['bm'] * (plan['bn'] + 8) * 2
    half = plan['bm'] * (plan['bn'] // 2 + 8) * 2
    return (P9E_HALF_STAGING and not plan.get('rows') and plan['bn'] % 128 == 0
            and plan['shared_bytes'] + full > SHARED_LIMIT and plan['shared_bytes'] + half <= SHARED_LIMIT)


def staging_bytes(plan):
    """P7-B: shared bytes of the staged epilogue tile (pops4b::Staging: BM x (BN + 8) BF16; P9-E: BN / 2 when halved)."""
    if staged_half(plan):
        return plan['bm'] * (plan['bn'] // 2 + 8) * 2
    return plan['bm'] * (plan['bn'] + 8) * 2


def row_plan(spec, grid=None):
    """P7-B: whole-row tile plan for a TMA warpgroup product whose row epilogue reads complete rows: BM 64, BN = N
    (two warpgroups split on N), two operand stages (the staged tile aliases the drained ring). None when the
    product does not admit it (N not a whole 128/192/256 tile, split-K, batched, edge, short K, shared limit)."""
    grid = TARGET_GRID if grid is None else grid
    m, n, k = spec['m'], spec['n'], spec['k']
    if (not (spec.get('wg') and spec.get('tma') and tma_allowed()) or spec.get('split') or spec['batch'] != 1
            or spec.get('edge') or n not in (128, 192, 256) or not spec.get('pair') or -(-k // 64) < 2):
        return None
    if not spec['bkc'] and (n // 2) % 64:
        return None
    stage = (64 + n) * 128
    if 2 * stage + 1024 > SHARED_LIMIT or 64 * (n + 8) * 2 > 2 * stage:
        return None
    tiles = -(-m // 64)
    return {'bm': 64, 'bn': n, 'wm': 1, 'nst': 2, 'tiles': tiles, 'waves': math.ceil(tiles / grid), 'grid': grid,
            'shared_bytes': 2 * stage + 1024, 'padded': [64 * tiles, n, 64 * -(-k // 64)], 'wg': True, 'rows': True}


def _plan(spec, grid):
    args = (spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'])
    if spec.get('split') and spec.get('wg'):
        split = spec['split']
        return wg_plan(spec['m'], spec['n'], spec['k'] // split, split, spec['ak'], spec['bkc'], grid=grid)[0]
    if spec.get('split'):
        split = spec['split']
        return bf16_plan(spec['m'], spec['n'], spec['k'] // split, split, spec['ak'], spec['bkc'], grid=grid)[0]
    if spec.get('wg'):
        return wg_plan(*args, grid=grid, force=FORCE_BY_OUTPUT.get(spec['out']))[0]
    if spec['dtype'] == 'bfloat16':
        return bf16_plan(*args, grid=grid, force=FORCE_BY_OUTPUT.get(spec['out']), edge=bool(spec.get('edge')))[0]
    return fp32_plan(*args, grid=grid)[0]


def rowstats_half(plan):
    """Row softmax partials stage the tile in two column halves when the whole tile does not fit beside the ring."""
    return plan['shared_bytes'] + plan['bm'] * (plan['bn'] + 8) * 2 > SHARED_LIMIT


def rowstats_staging(plan):
    return 0          # register-direct partials (pops4b non-staged path): no staging tile


def rowstats_admitted(spec, plan):
    """Register-direct row partials need every row's tile columns in one quad: two warpgroups splitting M (WM 2)."""
    return (plan['wm'] == 2 and spec.get('pair') and not spec.get('split') and not spec.get('ksegs')
            and not spec.get('tma_batch') and not _claims(spec))


MN_STACKED_A = True   # LATCH-MOE-EMA-FAST: M-major batched A admitted for the stacked TMA form (False: K-major only)


def batch_stacked(m, n, k, batch, ak, bkc, strides):
    """LATCH-MOESUP: a batched product the TMA body runs as one product of batch x M rows: A K-contiguous with the
    batches' rows stacked, B's map rows (K for N-major B, N for K-contiguous B) stacked, M a multiple of 128, and
    K (N-major B) a multiple of 64 / N (K-contiguous B) a multiple of 64 so no tile or k stage crosses a batch."""
    sam, sak, sbk, sbn, bsa, bsb = strides
    if batch < 2 or m % 128:
        return False
    if ak and bsa != m * sam:
        return False
    # LATCH-MOE-EMA-FAST: an M-major A stacks its batches along K (dims {M_b, batch * K_b}); no k stage crosses one.
    if not ak and not (MN_STACKED_A and bsa == k * sak and k % 64 == 0):
        return False
    if bkc:
        return bsb == n * sbn and n % 64 == 0
    return bsb == k * sbk and k % 64 == 0


STACKED_STAGED_EPILOGUES = True


def batch_stacked_tiles(spec, plan, epi=None):
    """Tiles remain within a batch; the direct rounded-widen epilogue and (LATCH-MOE-EMA-100) staged same-index
    epilogues are compatible: a stacked tile's rows are rows of one batch, so the staged chunk's flat index
    (stacked row * N + column) is the contiguous [batch, M, N] result index."""
    direct_widen = (epi is not None and epi.get('batch_stacked_widen') and not epi.get('store')
                    and not epi.get('staged') and not epi.get('rows'))
    staged = (epi is not None and STACKED_STAGED_EPILOGUES and (epi.get('staged') or epi.get('pair_direct'))
              and not epi.get('rows') and not epi.get('colpart') and not spec.get('ldc'))
    return ((epi is None or direct_widen or staged) and not spec.get('split') and not spec.get('ksegs')
            and not spec.get('serial_bf16_k') and spec['m'] % plan['bm'] == 0
            and (not spec['bkc'] or spec['n'] % plan['bn'] == 0))


def _code(spec, plan, ident, lo, hi, den, rot, epi=None, table=None):
    sam, sak, sbk, sbn, bsa, bsb = spec['strides']
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    t = lambda flag: str(flag).lower()
    serial_k = spec.get('serial_bf16_k')
    if serial_k is not None and (type(serial_k) is not int or serial_k <= 0 or serial_k % 16
            or serial_k >= k or not spec.get('tma') or not tma_allowed()
            or spec.get('split') or spec.get('ksegs') or batch != 1):
        raise MatrixRuleError('serial BF16 epilogues require an unsplit TMA product and16-aligned partial K')
    stacked = spec.get('tma_batch')
    if stacked and not batch_stacked_tiles(spec, plan, epi):
        stacked, spec = None, {key: value for key, value in spec.items() if key not in ('tma', 'tma_batch')}
    if spec.get('tma') and tma_allowed():
        (ra, oa), (rb, ob) = spec['tma']['a'], spec['tma']['b']
        km = spec.get('kseg') or k        # P9-E K segments: every map spans one segment's K
        ta = _tmap_marker(ra, oa, spec['ak'], m * batch if stacked and spec['ak'] else m,
                          km * batch if stacked and not spec['ak'] else km, sam if spec['ak'] else sak, plan['bm'])
        tb = _tmap_marker(rb, ob, spec['bkc'], n * batch if stacked and spec['bkc'] else n,
                          km * batch if stacked and not spec['bkc'] else km, sbn if spec['bkc'] else sbk, plan['bn'])
        ldc = spec.get('ldc', n)          # step 02: padded result pitch (default: contiguous rows)
        vec = WG_VEC_STORE and n % 8 == 0 or bool(spec.get('ldc')) and ldc % 8 == 0 and ldc >= -(-n // 8) * 8
        kseg_lines = f'  g.serial_bf16_k = {serial_k};\n' if serial_k is not None else ''
        if stacked:
            kseg_lines = (f'  g.bgrp = {m}; g.bstep = {n if spec["bkc"] else k};'
                          + ('' if spec['ak'] else f' g.astep = {k};') + '\n')
            m *= batch
        if spec.get('ksegs'):
            if spec.get('split') or _claims(spec) or len(spec['ksegs']) > 2 or km % 64:
                raise MatrixRuleError('P9-E K-segmented products: unsplit, unclaimed, <= 3 segments of whole k stages')
            for j_, seg in enumerate(spec['ksegs'], start=2):
                (ra2, oa2), (rb2, ob2) = seg['tma']['a'], seg['tma']['b']
                kseg_lines += (f'  g.tA{j_} = {_tmap_marker(ra2, oa2, spec["ak"], m, km, sam if spec["ak"] else sak, plan["bm"])}; '
                               f'g.tB{j_} = {_tmap_marker(rb2, ob2, spec["bkc"], n, km, sbn if spec["bkc"] else sbk, plan["bn"])};\n')
            kseg_lines += f'  g.kseg = {km};\n'
            # the segments' operand views are reads of this item (storage lifetimes come from referenced identifiers)
            kseg_lines += ''.join(f'  (void){ident(seg["a"])}; (void){ident(seg["b"])};\n' for seg in spec['ksegs'])
        if spec.get('split'):
            # P6-B split-K on the TMA body: maps span the whole K; tasks = (slice, tile); FP32 slice results.
            if not WG_INLINE:
                raise MatrixRuleError('split-K warpgroup products need the entry-resident TMA body')
            call = (f'pops4b::tma_phase_inline<{plan["bm"]},{plan["bn"]},{plan["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan["nst"]},true,false,pops4b::NoEpi,false,{spec["split"]}>')
            return ('{\n'
                    f'  pops4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, nullptr, {n}LL, {m}, {n}, '
                    f'{spec.get("k_pad", k)}, {spec["out"]}}};\n'
                    + (f'  g.tt = {table}; g.tg = {P7B_LPT_GRID};\n' if table else '')
                    + ('  g.claim = /*P7B_CLAIM*/;\n' if _claims(spec) else '') +
                    f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                    '}')
        if spec.get('out_raw'):
            # LATCH-RESEARCH-EDITS: unrounded FP32 result (RAW): BF16 maps and k order as above; FP32 [M][N] result.
            if not WG_INLINE or spec.get('rowstats') or spec.get('ksegs') or _claims(spec) or n % 2:
                raise MatrixRuleError('raw FP32 results need the entry-resident unsplit TMA body and even N')
            call = (f'pops4b::tma_phase_inline<{plan["bm"]},{plan["bn"]},{plan["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan["nst"]},true,false,pops4b::NoEpi,false,1,false,false,false,false,true>')
            return ('{\n'
                    f'  pops4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, nullptr, {n}LL, {m}, {n}, '
                    f'{k}, {spec["out"]}}};\n'
                    + (f'  g.tt = {table}; g.tg = {P7B_LPT_GRID};\n' if table else '') +
                    f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                    '}')
        if epi is not None:
            # P6-B: generated pair epilogue (pops4b::tma_body EPI) on the rounded product; STORE per elision.
            if not WG_INLINE:
                raise MatrixRuleError('pair epilogues need the entry-resident TMA body')
            staged = bool(epi.get("staged"))
            rows = bool(epi.get("rows"))
            if rows and not plan.get('rows'):
                raise MatrixRuleError('P7-B row epilogues need the whole-row plan (row_plan)')
            alias = staged and staged_alias(spec, plan, table)
            if staged and not rows and (n % 8 or (plan['shared_bytes'] + staging_bytes(plan) > SHARED_LIMIT
                                                  and not alias)):
                raise MatrixRuleError('P7-B staged epilogue needs N % 8 == 0 and staging within the shared limit')
            call = (f'pops4b::tma_phase_inline<{plan["bm"]},{plan["bn"]},{plan["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan["nst"]},{t(spec["pair"])},{t(vec)},decltype(epi),{t(epi["store"])}'
                    + (',1,true,true>' if rows else ',1,true,false,true>' if alias
                       else ',1,true,false,false,true>' if staged and staged_half(plan)
                       else ',1,true>' if staged else '>'))
            if epi.get("colpart") and not (staged and not rows):
                raise MatrixRuleError('P9-E epilogue column sums need the staged epilogue')
            return ('{\n'
                    f'  pops4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, '
                    f'{spec["out"] if epi["store"] else "nullptr"}, {n}LL, {m}, {n}, {k}}};\n'
                    + kseg_lines
                    + (f'  g.colpart = {epi["colpart"]};\n' if epi.get("colpart") else '')
                    + (f'  g.tt = {table}; g.tg = {P7B_LPT_GRID};\n' if table else '')
                    + ('  g.claim = /*P7B_CLAIM*/;\n' if _claims(spec) else '') +
                    f'  const auto epi = {epi["code"]};\n'
                    f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}, epi);\n'
                    '}')
        call = (f'pops4b::tma_phase{"_inline" if WG_INLINE else ""}<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{t(spec["pair"])}'
                + (',true>' if vec else '>'))
        if spec.get('rowstats'):
            # LATCH-MOESUP: row softmax partials from the staged (BF16-rounded) tile; plain store otherwise unchanged.
            rs = spec['rowstats']
            if not rowstats_admitted(spec, plan):
                raise MatrixRuleError('row softmax partials: BM 128, even N, unsplit, unclaimed, staged tile must fit')
            kseg_lines += (f'  g.rowpart = reinterpret_cast<float2*>({rs["out"]}); g.rowvalid = {rs["valid"]}; '
                           f'g.rowtiles = {rs["tiles"]};\n')
        return ('{\n'
                f'  pops4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {ldc}LL, '
                f'{m}, {n}, {k}}};\n'
                + kseg_lines
                + (f'  g.tt = {table}; g.tg = {P7B_LPT_GRID};\n' if table else '')
                + ('  g.claim = /*P7B_CLAIM*/;\n' if _claims(spec) else '') +
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec.get('ksegs'):
        raise MatrixRuleError('P9-E K-segmented products need the TMA warpgroup body')
    if spec.get('wg'):
        call = (f'pops4b::wmm_phase{"_inline" if WG_INLINE else ""}<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{t(spec["pair"])}' +
                (f',{t(WG_FLAT)},{int(WG_DIAG)}>' if WG_DIAG else (',true>' if WG_FLAT else '>')))
        struct = 'popsc::BArgs'
    elif spec.get('split'):
        split, ks = spec['split'], k // spec['split']
        call = (f'popsc::bmm_split_phase<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]}>')
        # Batch index = K slice: A/B advance by one slice along K; C is the FP32 slice-result buffer [split][m][n].
        return ('{\n'
                f'  popsc::BArgs g{{{ident(spec["a"])}, {sam}LL, {sak}LL, {ks * sak}LL, {ident(spec["b"])}, {sbk}LL, '
                f'{sbn}LL, {ks * sbk}LL,\n'
                f'              reinterpret_cast<popsc::bf16*>({spec["out"]}), {n}LL, {m * n}LL, {m}, {n}, {ks}, {split}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    elif spec['dtype'] == 'bfloat16' and spec.get('edge'):
        call = (f'popsc::bmm_edge_phase<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]},{t(spec["pair"])}>')
        struct = 'popsc::BArgs'
    elif spec['dtype'] == 'bfloat16':
        call = (f'popsc::bmm_phase<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
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
    bf16 = spec['dtype'] == 'bfloat16'
    edge = '_edge' if spec.get('edge') else ''
    if spec.get('wg'):
        if spec.get('split'):
            edge = f'_split{spec["split"]}'
        if spec.get('out_raw'):
            return {'engine': f'pops4b_tma_raw32_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
                    'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
                    'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
                    'arithmetic': ('explicit BF16 operands; warpgroup MMA (wgmma) FP32 accumulation from +0, ascending k16; '
                                   'no split-K; absent elements staged as +0, stores masked; unrounded FP32 result')}
        return {'engine': f'pops4b_{"tma" if spec.get("tma") and tma_allowed() else "wg"}{edge}_{plan["bm"]}x{plan["bn"]}',
                'output': spec['out'],
                'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
                'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
                'arithmetic': ('explicit BF16 operands; warpgroup MMA (wgmma) FP32 accumulation from +0 per K slice, '
                               f'ascending k16; {spec["split"]} unrounded FP32 slice results, fixed-order slice sum and '
                               'one rounding where consumed' if spec.get('split') else
                               'explicit BF16 operands/output; warpgroup MMA (wgmma) FP32 accumulation, ascending k16; '
                               'no split-K; absent elements staged as +0, stores masked; one cvt.rn.bf16x2 rounding')}
    if spec.get('split'):
        edge = f'_split{spec["split"]}'
    return {'engine': f'popsc_{"bf16" if bf16 else "fp32"}{edge}_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
            'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
            'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
            'arithmetic': ('explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; no split-K; '
                           'one cvt.rn.bf16x2 rounding (bitwise = pinned 64x64 body' +
                           (' on zero-padded operands; absent elements staged as +0, stores masked)' if edge else ')'))
                          if bf16 else
                          'FP32 SIMT fmaf chain from +0, ascending k; no TF32 (bitwise = pinned SIMT 64x64 body)'}


def emit_group(specs, *, ident, grid=None, partition=None):
    grid = TARGET_GRID if grid is None else grid   # read at call time (scoped overrides)
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
    tables = lpt_tables(specs, whole) if P7B_LPT and best[0] == 'serial' else {}
    if best[0] == 'serial':
        for si in claim_order(specs):
            spec, plan = specs[si], whole[si]
            blocks.append(_code(spec, plan, ident, 0, 1, 1, rot, table=tables[si][0] if si in tables else None))
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
    if tables:
        extra['p7b_tables'] = [definition for _, definition in tables.values()]
    return '\n'.join(blocks), extra
