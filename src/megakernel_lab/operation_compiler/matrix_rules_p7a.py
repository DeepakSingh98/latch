# Adapted from frozen P7-A f36b4737cee732e6915d11e233bcc068126b3e64; private scheduling/resource protocol.
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
# P-OPS4-B warpgroup-MMA bodies (kernels/pops4b_tma_p7a.cuh; cp.async form kernels/pops4b_wg_p7a.cuh). Selected default (gpu-07):
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
WG_HEADERS = ('s107a_wgmma.cuh', 'pops4b_wg_p7a.cuh')
# (BM, BN, WM): WM = 2 splits M between the two warpgroups, WM = 1 splits N.
WG_TILES = ((128, 256, 2), (128, 192, 2), (128, 128, 2), (128, 64, 2), (64, 256, 1), (64, 128, 1))
# Fitted to the P-OPS4-B TMA body on H100 (gpu-07: weight-gradient 128x256 runs at the tensor bound on 70 CTAs;
# 128x192 / 64x256 and the 14-wave forward product are bound by aggregate L2->SM operand traffic of ~7 TB/s).
WG_SM_FLOPS_PER_US = 7.8e6      # sustained dense BF16 wgmma per SM
WG_L2_BYTES_PER_US = 7.0e6      # aggregate operand stream, all active CTAs
WG_STORE_BYTES_PER_US = 2.6e6   # aggregate BF16 result stores
WG_FIXED_US = 1.0
# P6-A deterministic split-K for long-K TMA products that fill under half the grid (weight gradients): K-slices are
# planned as a batch of products, FP32 partials are summed in slice order and rounded once (FP32 reassociation).
WG_SPLITK = True
WG_SPLIT_MIN_K = 2048
WG_SPLIT_GAIN = 0.85           # accept a split when its modeled time (incl. partial traffic) is below this fraction
# P7-A: operand-stream rate used when planning split-K weight-gradient products (None: WG_L2_BYTES_PER_US) and
# plain-store (epilogue-free) warpgroup products (None: WG_L2_BYTES_PER_US). In-kernel measurement (gpu-26): long-K
# split products and plain products favour wider tiles than row-epilogue products at the same modeled rate.
WG_SPLIT_L2_BYTES_PER_US = None
# P7-A: the last warpgroup TMA product of a phase whose products run back to back (serial group schedule) claims its
# tiles dynamically (kernels/pops4b_tma_p7a.cuh DYN), so CTAs that ran earlier products of the phase take fewer tiles.
WG_DYN_TILES = False
WG_PLAIN_L2_BYTES_PER_US = None
WG_PARTIAL_BYTES_PER_US = 2.6e6
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
    shared_limit = SHARED_LIMIT if shared_limit is None else shared_limit
    grid = TARGET_GRID if grid is None else grid
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


# P6-A: widest warpgroup tile (columns per warpgroup) for products carrying a generated epilogue (None = any):
# a 256-wide warpgroup tile holds 128 FP32 accumulators per thread, which with epilogue temporaries spills.
WG_EPI_MAX_WTN = None
# P7-A: widest warpgroup tile for products whose generated epilogue is a vector staged epilogue (accumulators go
# to a shared staging tile before any epilogue arithmetic, so the epilogue holds no accumulators). None keeps
# WG_EPI_MAX_WTN for them. Candidates must have a power-of-two warpgroup width (vector traversal) and fit the
# staging tile inside the shared limit.
WG_VEC_EPI_MAX_WTN = None


def wg_plan(m, n, k, batch, ak, bkc, *, grid=None, shared_limit=None, force=None, max_wtn=None, extra_bytes=None,
            wtn_set=None, l2=None):
    shared_limit = SHARED_LIMIT if shared_limit is None else shared_limit
    grid = TARGET_GRID if grid is None else grid
    """warpgroup-MMA tile/stage plan (64-wide k stages, ceil-divided tiles, estimates on padded extents)."""
    nk = -(-k // 64)
    kp = 64 * nk
    candidates = []
    for bm, bn, wm in WG_TILES:
        wtn = bn // (2 // wm)
        if not bkc and wtn % 64:
            continue                       # N-major B: whole 64-column swizzle chunks per warpgroup
        if max_wtn is not None and wtn > max_wtn:
            continue
        if wtn_set is not None and wtn not in wtn_set:
            continue
        stage_bytes = (bm + bn) * 128
        extra = int(extra_bytes(bm, bn, wm)) if extra_bytes is not None else 0
        nst = min(4, nk + 1, (shared_limit - 1024 - extra) // stage_bytes)
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
        traffic = tiles * (bm + bn) * kp * 2 / (l2 or WG_L2_BYTES_PER_US)
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                           'shared_bytes': nst * stage_bytes + 1024 + extra, 'padded': [mp, np_, kp], 'wg': True,
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
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates


def spec_headers(spec):
    if spec.get('tma'):
        return WG_HEADERS + ('pops4b_tma_p7a.cuh',)
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
    grid = TARGET_GRID if grid is None else grid
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
    # P6-A: long-K products that would leave most of the grid idle also take the TMA body (with split-K) below the
    # FLOP threshold (e.g. a patch-embedding weight gradient).
    long_k_split = (WGMMA and WGMMA_ALIGNED and WG_TMA and WG_SPLITK and batch == 1 and k >= WG_SPLIT_MIN_K
                    and k % 64 == 0 and m % 2 == 0 and n % 2 == 0
                    and splitk_choice(m, n, k, sak == 1, sbk == 1) is not None)
    if (WGMMA and (spec.get('edge') or WGMMA_ALIGNED) and (2 * m * n * k * batch >= WG_MIN_FLOPS or long_k_split)):
        wg_plan(m, n, k, batch, sak == 1, sbk == 1)
        spec.update(wg=True, pair=n % 2 == 0)
        if WG_TMA and batch == 1:
            # Maps bind to the operand views themselves (the Runtime resolves their final storage and offset).
            spec['tma'] = {'a': (a, 0), 'b': (b, 0)}
            split = splitk_choice(m, n, k, sak == 1, sbk == 1) if WG_SPLITK and (not spec.get('edge') or long_k_split) else None
            if split is not None:
                spec.update(splitk=split, k_full=k, k=k // split, batch=split)
    return spec


def splitk_choice(m, n, k, ak, bkc, *, grid=None):
    grid = TARGET_GRID if grid is None else grid
    """Number of K-slices for a long-K product filling under half the grid, or None (P6-A)."""
    if k < WG_SPLIT_MIN_K or k % 64:
        return None
    whole = wg_plan(m, n, k, 1, ak, bkc, grid=grid, l2=WG_SPLIT_L2_BYTES_PER_US)[0]
    if whole['tiles'] * 2 > grid:
        return None
    nk, best = k // 64, None
    for ks in range(2, 33):
        if nk % ks:
            continue
        plan = wg_plan(m, n, k // ks, ks, ak, bkc, grid=grid, l2=WG_SPLIT_L2_BYTES_PER_US)[0]
        cost = plan['estimate_us'] + 2 * ks * m * n * 4 / WG_PARTIAL_BYTES_PER_US
        if best is None or cost < best[1]:
            best = (ks, cost)
    if best is None or best[1] >= WG_SPLIT_GAIN * whole['estimate_us']:
        return None
    return best[0]


def fp32_small_applies(la, lb, shape, *, grid=None):
    grid = TARGET_GRID if grid is None else grid
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
    args = (spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'])
    if spec.get('wg'):
        return wg_plan(*args, grid=grid, force=FORCE_BY_OUTPUT.get(spec['out']),
                       l2=WG_SPLIT_L2_BYTES_PER_US if spec.get('splitk') else WG_PLAIN_L2_BYTES_PER_US)[0]
    if spec['dtype'] == 'bfloat16':
        return bf16_plan(*args, grid=grid, force=FORCE_BY_OUTPUT.get(spec['out']), edge=bool(spec.get('edge')))[0]
    return fp32_plan(*args, grid=grid)[0]


def _code(spec, plan, ident, lo, hi, den, rot, epi=None, dyn=False):
    sam, sak, sbk, sbn, bsa, bsb = spec['strides']
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    t = lambda flag: str(flag).lower()
    if spec.get('splitk'):
        if not (spec.get('tma') and tma_allowed()):
            raise MatrixRuleError('split-K products need the outermost TMA lowering scope')
        (ra, oa), (rb, ob) = spec['tma']['a'], spec['tma']['b']
        kf = spec['k_full']
        ta = _tmap_marker(ra, oa, spec['ak'], m, kf, sam if spec['ak'] else sak, plan['bm'])
        tb = _tmap_marker(rb, ob, spec['bkc'], n, kf, sbn if spec['bkc'] else sbk, plan['bn'])
        call = (f'p7a4b::tma_split_phase_inline<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{spec["splitk"]}>')
        return ('{\n'
                f'  p7a4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, nullptr, {n}LL, '
                f'{m}, {n}, {kf}}};\n'
                f'  {call}(g, {spec["out"]}, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec.get('tma') and tma_allowed():
        (ra, oa), (rb, ob) = spec['tma']['a'], spec['tma']['b']
        ta = _tmap_marker(ra, oa, spec['ak'], m, k, sam if spec['ak'] else sak, plan['bm'])
        tb = _tmap_marker(rb, ob, spec['bkc'], n, k, sbn if spec['bkc'] else sbk, plan['bn'])
        vec = WG_VEC_STORE and n % 8 == 0
        dyn_field = ', /*P7_DYN*/nullptr' if dyn else ''
        if epi:
            # P6-A pair epilogue (register-direct); the product store is kept unless only the epilogue reads it.
            call = (f'p7a4b::tma_phase_inline<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                    f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{t(spec["pair"])},{t(vec)},'
                    f'decltype(epi),{t(epi["store"])}' + (',true>' if dyn else '>'))
            out = spec["out"] if epi["store"] else "nullptr"
            return ('{\n'
                    f'  p7a4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {out}, {n}LL, '
                    f'{m}, {n}, {k}{dyn_field}}};\n'
                    f'  const auto epi = {epi["code"]};\n'
                    f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}, epi);\n'
                    '}')
        call = (f'p7a4b::tma_phase{"_inline" if WG_INLINE else ""}<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{t(spec["pair"])}'
                + (f',{t(vec)},p7a4b::NoPairEpi,true,true>' if dyn and WG_INLINE else (',true>' if vec else '>')))
        return ('{\n'
                f'  p7a4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}{dyn_field if WG_INLINE else ""}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec.get('wg'):
        call = (f'p7a4b::wmm_phase{"_inline" if WG_INLINE else ""}<{plan["bm"]},{plan["bn"]},{plan["wm"]},'
                f'{t(spec["ak"])},{t(spec["bkc"])},{plan["nst"]},{t(spec["pair"])}' +
                (f',{t(WG_FLAT)},{int(WG_DIAG)}>' if WG_DIAG else (',true>' if WG_FLAT else '>')))
        struct = 'popsc::BArgs'
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
        return {'engine': f'pops4b_{"tma" if spec.get("tma") and tma_allowed() else "wg"}{edge}_{plan["bm"]}x{plan["bn"]}',
                'output': spec['out'],
                'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
                'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
                'arithmetic': 'explicit BF16 operands/output; warpgroup MMA (wgmma) FP32 accumulation, ascending k16; '
                              'no split-K; absent elements staged as +0, stores masked; one cvt.rn.bf16x2 rounding'}
    return {'engine': f'popsc_{"bf16" if bf16 else "fp32"}{edge}_{plan["bm"]}x{plan["bn"]}', 'output': spec['out'],
            'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': spec['batch'], 'matrix_plan': plan,
            'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
            'arithmetic': ('explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; no split-K; '
                           'one cvt.rn.bf16x2 rounding (bitwise = pinned 64x64 body' +
                           (' on zero-padded operands; absent elements staged as +0, stores masked)' if edge else ')'))
                          if bf16 else
                          'FP32 SIMT fmaf chain from +0, ascending k; no TF32 (bitwise = pinned SIMT 64x64 body)'}


def emit_group(specs, *, ident, grid=None, partition=None):
    grid = TARGET_GRID if grid is None else grid
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
        for idx_, (spec, plan) in enumerate(zip(specs, whole)):
            dyn = bool(WG_DYN_TILES and idx_ == len(specs) - 1 and idx_ > 0 and spec.get('wg') and spec.get('tma')
                       and tma_allowed() and not spec.get('splitk') and WG_INLINE)
            blocks.append(_code(spec, plan, ident, 0, 1, 1, rot, dyn=dyn))
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
