"""P-OPS5-C: warpgroup-MMA engines for already-supported, aligned BF16 products in flat fused programs.

Selection uses only dtype, rank, static M/N/K, operand layout (which stride is unit), alignment and the
resident grid; never value or model names. Engines (``ENGINE``):

* ``popsc``: the integrated P-OPS-C mma.sync bodies; the emitted source is byte-identical to the base compiler.
* ``p4b_tma``: P-OPS4-B's TMA + wgmma body imported unchanged from research checkpoint 92bb997a7
  (kernels/pops4b_tma.cuh, kernels/pops4b_wg.cuh, vendor/s107a_wgmma.cuh), entry-resident, 16-byte result stores,
  with P-OPS4-B's ``wg_plan`` tile model. Kept as the measured import arm.

Arithmetic of the wgmma engines: BF16 x BF16 products accumulated in FP32 by the tensor core in ascending k16
steps from +0, one round-to-nearest BF16 result. It is not bitwise equal to mma.sync (different instruction
association inside a k16 step); it is gated by the declared mixed-precision tolerances, like cuBLAS.

Operand tensor maps (TMA) are emitted as placeholders naming the operand's storage root and element offset;
``finalize`` (called once per flat fused program) replaces them by entries of a map table passed as
``args[len(inputs)]`` and records ``manifest["tensor_maps"]``. ``Runtime`` encodes the maps for its bound
storages. Maps describe addresses/extents/strides only: refreshed operand values are read every launch.
"""
from __future__ import annotations

import math
from pathlib import Path
import re

ENGINES = ('popsc', 'p4b_tma', 'p5c')
ENGINE = 'p5c'                 # 'popsc' reproduces the integrated base source byte for byte
MIN_FLOPS = 2 ** 27            # smaller aligned products keep the mma.sync small-tile bodies
# Admission (measured, P-OPS5-C gpu-06 long-context transfer): the warpgroup engine needs a K loop long enough to
# amortize stage fill and its epilogue (MIN_K_STAGES 64-wide stages), and enough 128 x 128 output tiles to occupy
# at least half the resident grid; otherwise the P-OPS-C small-tile body (more, smaller tiles) is kept.
MIN_K_STAGES = 4            # P9-B gpu-98: 4 / 0.25 (was 8 / 0.5): 6.885 vs 6.911 ms, bitwise
MIN_TILE_FRACTION = 0.25
ADMIT_GRID = 128
WG_HEADERS = ('s107a_wgmma.cuh', 'pops4b_wg.cuh', 'pops4b_tma.cuh', 'p5c_wg.cuh')

# Lane engine 'p5c' (kernels/p5c_wg.cuh). Options are module settings recorded in every member record.
P5C_TILES = ((128, 256, 2), (256, 128, 2), (128, 128, 2))
P5C_SHARED_LIMIT = 220 * 1024  # dynamic stage ring + 1 KiB alignment; leaves room for static shared storage
P5C_MAX_STAGES = 6           # P9-B (gpu-75): 6 stages for 128 x 128 tiles, 7.213 vs 7.289 ms, bitwise (was 4)
P5C_DECOUPLED = False          # P6-D: no CTA barrier per k-step; the second warpgroup to finish a stage refills it
P5C_STAGED = True              # P9-B: staged epilogue for 128 x 128 tiles (+32 KB shared, one stage fewer)
P5C_BULK_EPI = False           # LATCH-STEP-06: plain staged tiles store by per-row bulk async copies (272 B rows)


def staging_bytes():
    """Shared bytes of the 128 x 128 staged epilogue buffer (+2 KiB of row padding for the bulk store)."""
    return 32768 + (2048 if P5C_BULK_EPI else 0)


def carveout():
    """Largest dynamic shared size a p5c plan may take (the program's 196 KiB carve-out plus the bulk padding)."""
    return 197632 + (2048 if P5C_BULK_EPI else 0)
P5C_TRACE = False              # P9-B diagnostic: per-tile timeline stamps (kernels/p5c_wg.cuh, timing only)
P5C_ACC_BATCH = False          # P9-B: FP32-accumulate epilogues load a chunk group's old values together (exact)
P5C_ACC_PREFETCH = False       # P9-B: L2 prefetch of the FP32 accumulate destination at tile start (exact)
P5C_PAR_PRE = False            # P9-B: prologue stages issued by lanes 0..pre-1 of warp 0 in one pass (exact)
P5C_SYNC2 = False              # P6-D: barrier/refill every two k-steps (kernels/p5c_wg.cuh, P5C_SYNC2)
P5C_RELEASE = False            # per-stage empty mbarriers instead of a CTA barrier per 64-k step
P5C_CONVERT = True             # FP32-converting store when a product's only reader is its materialized FP32 conversion
P5C_FORCE = {}                 # diagnostic tile overrides keyed by (m, n, k, ak, bkc) -> (bm, bn)
P5C_EPILOGUE = True            # elementwise register chains in the register epilogue (ops/p5c_epilogue.py)
# Products carrying a generated epilogue use this tile when admissible (measured: the epilogue's ALU-bound work
# spreads over twice as many CTAs per output area; P-OPS5-C gpu-03/04/05 screens).
P5C_EPI_TILE = (128, 128)
# P6-D: False lets FP32-converting products that carry a (state-update) epilogue keep the cost model's tile.
P5C_EPI_TILE_F32 = True
# Exact rewrites inside generated epilogues: common transcendental subexpressions once, 1/x via __frcp_rn.
P5C_EXACT_REWRITES = True
P5C_MULTICAST = False
P5C_SHARED_SITES = False       # one entry-resident body per template configuration, reached by goto trampolines
# Ping-pong body (p5c::pp_body): per-warpgroup tiles and rings, main loops alternate so one warpgroup's epilogue
# overlaps the other's MMAs. 'never' | 'epilogue' (products carrying a generated epilogue) | 'always'.
P5C_PINGPONG = 'never'
# Diagnostic only: insert p5c::diag_call into phases: None | 'elementwise' | 'all' | 'first' | 'last'.
P5C_DIAG_CALL = None
P5C_DIAG_FN = 'diag_call'   # diag_call | diag_sync | diag_cpasync | diag_bmm (bmm_phase is itself out of line)
# Diagnostic only: admit the engine for at most this many products (in lowering order); None = all.
P5C_DIAG_ONLY_FIRST = None
_DIAG_ADMITTED = [0]
# P9-B: P8-A's warp-specialized body (kernels/p8ws.cuh, recovered from frozen 25f1d13b3) admitted for loop-region
# products as well (operand maps picked per region instance like the p5c body). One MMA warpgroup owns a BM x 128 tile
# (two m64n128k16 slabs, same chains/rounding as the WM = 2 p5c body), a producer thread fills the stage ring and the
# other warps run the generated register epilogue from a staged BF16 tile. K-major A only by default: P8-B measured
# 128-wide warpgroup tiles ~1.5x slower per k-step on M-major A (weight-gradient) products.
P9WS = False
P9WS_STAGES = 4
P9WS_MN_A = False
P9WS_ALU_RULE = True
P9WS_SHARED_LIMIT = 220 * 1024
# 'p8ws': P8-A's body (BN = 128). 'p9ws': kernels/p9ws.cuh, per-product BN in {128, 96, 64} chosen to minimize the
# busiest CTA's tile area after wave quantization (96/64 need a K-major B operand), ties to the wider tile.
P9WS_ENGINE = 'p8ws'

# P8-A warp-specialized body (kernels/p8ws.cuh): one MMA warpgroup owns a BM x 128 tile, a dedicated TMA producer
# thread fills the stage ring, three warps run the register epilogue from a double-buffered shared staging tile.
# Selected structurally for aligned p5c products without split-K, multicast or ping-pong; bitwise equal to the
# WM = 2 body (same m64n128k16 chains). 64-row tiles when 128-row tiles would leave over half the grid idle.
P8WS = True
P8WS_RAW = True                # LATCH-FFN-RETAIN: raw FP32 products (OUT == 3) on the flat warp-specialized body
P8WS_RAW_MIN_STEPS = 16        # ... when each tile has at least this many 64-k steps to hide its FP32 epilogue under

# Select the existing cooperative PK body for
# long contractions with several full waves of output tiles. Its two MMA
# warpgroups retain the ordered chunk sum in registers. The warp-specialized
# body remains available as an explicit diagnostic and for short/underfilled
# products. This flag changes scheduling, not the PK arithmetic policy.
P5C_COOPERATIVE_CHUNKS = True
P5C_COOPERATIVE_CHUNK_MIN_WAVES = 4
P5C_COOPERATIVE_CHUNK_MIN_TILES = 512   # retain small products even when the group assigns them a tiny CTA slice
P8WS_STAGES = 5                # P9-A c53: 5-stage ring once local pointers freed registers (P8-A: 4)
P8WS_MMA_SPEEDUP = 1.3         # measured standalone mainloop gain over the p5c body (P8-A gpu-04), cost model only
P8WS_SHARED_LIMIT = 231000     # P9-A c53: room for the 5-stage ring (P8-A: 220 KiB)
# Explicit common-flat static reserve only. Otherwise retain the historical
# refusal when the originally selected stage depth exceeds a caller's cap.
P8WS_FIT_SHARED_LIMIT = False
# Register epilogues with transcendental or division work (latency-bound on the 3 epilogue warps; P8-A gpu-06: SiLU
# 271 vs 332 TF/s for the 8-warp body) keep the p5c body; plain, conversion and memory-bound (state update) epilogues
# use the warp-specialized body. The class is read from the generated expression, not from operation or model names.
P8WS_ALU_RULE = True
IM2COL_WS_ALU = True           # LATCH-STEP10-SPEED: the ALU rule does not apply to im2col-A products
P8WS_MODE = 0                  # 0: dedicated producer warp + 3 epilogue warps; 1: self-refill + 4 epilogue warps
P8WS_GROUP = True              # serial members of one phase run as one continuous warp-specialized pipeline
P8WS_STAGING = 2               # shared staging buffers (P8WS_NSB); 1 saves 32 KiB of the program's shared memory
P8WS_TRACE = False             # diagnostic: per-role event trace in CTAs 0-3 (p8ws_trace_read)
P8WS_GROUP_ROWS = None         # serial members of a multi-product group keep 128-row tiles (other members fill the grid)
P8WS_HELP = False              # MMA warps drain staged epilogue units before a tile whose staging buffer is still busy
_ALU_HEAVY = re.compile(r'\b(expf|logf|log1pf|expm1f|rsqrtf|sqrtf|tanhf|powf|erff|sinf|cosf|__frcp_rn|__fdiv_rn)\(|[^/*]/[ (]')
# LATCH-FFN-RETAIN: a division of two floating literals (e.g. a mean's 1.0f / N scale) is a compile-time constant, not
# epilogue ALU work; the rule judges the run text with such constants removed.
_CONST_DIV = re.compile(r'\(\s*-?\d+(?:\.\d*)?(?:e[-+]?\d+)?f?\s*/\s*-?\d+(?:\.\d*)?(?:e[-+]?\d+)?f?\s*\)')


def _alu_heavy(run):
    return bool(_ALU_HEAVY.search(_CONST_DIV.sub('(1.0f)', run)))


P5C_PP_TILE = (128, 128)
# P9-B: admit ping-pong for any product with at least one tile per CTA (an odd per-CTA count makes the second
# warpgroup recompute its last task without storing, as the body already does)
P5C_PP_ANY_TILES = False
P5C_PP_STAGES = 4
# Cost model (us): per 64-k step max(MMA at a per-SM rate, operand stream at an aggregate L2 rate / grid).
P5C_SM_FLOPS_PER_US = 989e6 / 132 * 0.85
P5C_L2_BYTES_PER_US = 7.0e6
P5C_FIXED_US = 1.5             # first-stage latency + epilogue per tile wave
# P8-B (measured, p8_b/probes/gemm_probe.cu on H100, gpu-07): with an M-major A operand (weight-gradient products,
# A = X^T) a warpgroup N width below 256 runs each 64-k step ~1.5x slower (768x3072x2048: 128x128 38.2 us vs
# 128x256 25.7 us; K-major A products show no such gap).
P5C_MN_MAJOR_A_PENALTY = 1.5
ARCH_HEADERS = ('pops4b_wg.cuh', 'p5c_wg.cuh', 'p8ws.cuh', 'p8ws_region.cuh', 'p9ws.cuh', 'attention_long.cuh', 'attention_long_flat.cuh')  # WGMMA requires sm_90a

# P-OPS4-B wg_plan constants (92bb997a7 matrix_rules.py), unchanged for the import arm.
P4B_TILES = ((128, 256, 2), (128, 192, 2), (128, 128, 2), (128, 64, 2), (64, 256, 1), (64, 128, 1))
P4B_SM_FLOPS_PER_US = 989e6 / 132 * 0.8
P4B_FIXED_US = 1.0
P4B_SHARED_LIMIT = 192 * 1024
SM_BYTES_PER_US = 100e3
L2_BYTES_PER_US = 8e6

# Flat scheduling restores these within its host lowering context; region defaults above remain independent.
FLAT_SELECTED_DEFAULTS = {
    'MIN_K_STAGES': 8, 'MIN_TILE_FRACTION': 0.5, 'P5C_MAX_STAGES': 4,
    'P5C_STAGED': False, 'P5C_MN_MAJOR_A_PENALTY': 1.0,
}

TMAP_MARK = 'P5C_TMAP'


class EngineRefusal(ValueError):
    pass


# P7-C 'p5cw': in staged loop regions, products co-scheduled in one dependency level share the grid by rotation,
# so a product filling a quarter of the grid (16 of 128 x 128 tiles) is admitted when its K loop is long enough.
WIDE_MIN_TILES = 16
SHORT_MIN_K_STAGES = 4
LONGK_SPLIT = True
SPLIT_COUNTS = (2, 3, 4, 8, 16)
SPLIT_STEP_US = 0.33
SPLIT_PARTIAL_US = 0.65
LONGK_MIN_STAGES = 64
LONGK_MIN_UNITS = 96


P9_SPLIT_FRACTION = 0.5     # P9-B: p5ck splits products with fewer than this fraction of the grid in tiles


def split_count(m, n, k, *, grid=128):
    """P7-C p5ck: K slices for a few-tile, long-K product (128 x 128 tiles), or 1. Minimizes waves x slice length,
    then fewer slices (less partial traffic); slices keep >= 8 k-steps of 64."""
    tiles, nk = (m // 128) * (n // 128), k // 64
    if tiles >= grid * P9_SPLIT_FRACTION or m % 128 or n % 128:
        return 1
    long_k = LONGK_SPLIT and tiles < WIDE_MIN_TILES and nk >= LONGK_MIN_STAGES
    step = SPLIT_STEP_US if long_k else 1.0
    best = (-(-tiles // grid) * nk * step, 1)
    for sk in (SPLIT_COUNTS if long_k else (2, 3, 4)):
        if nk % sk or nk // sk < 8:
            continue
        cost = -(-tiles * sk // grid) * (nk // sk) * step + (sk * SPLIT_PARTIAL_US if long_k else 0.0)
        if (cost, sk) < best:
            best = (cost, sk)
    return best[1]


def longk_split_admissible(m, n, k):
    """P9-C: a few-tile product with a long K loop fills the grid through split-K slices."""
    if m % 128 or n % 128 or k % 64 or k // 64 < LONGK_MIN_STAGES:
        return False
    return (m // 128) * (n // 128) * split_count(m, n, k) >= LONGK_MIN_UNITS


# Release step 3 repair (long-K accumulation accuracy). A wgmma chain accumulates each k16 product into its FP32
# accumulator with truncation relative to the running sum, so its error grows with the chain length: a vocabulary-scale
# contraction (K ~ 50k, the prediction-head input gradient) loses several bits more than conventional GEMMs. Products
# whose K loop exceeds LONGK_CHUNK_MIN_K restart the chain every LONGK_CHUNK_STAGES 64-k stages and sum the chunk results
# with round-to-nearest FP32 additions in chunk order (kernels p8ws.cuh item_k, p5c_wg.cuh PK). Shape-only rule; products
# with K <= LONGK_CHUNK_MIN_K keep the single-chain bodies and identical generated code.
LONGK_CHUNK = True
LONGK_CHUNK_STAGES = 32
LONGK_CHUNK_MIN_K = 8192


def _out_mode(spec):
    """p5c OUT template value: 0 BF16 store, 1 exact widening of the rounded BF16 result, 2 its FP32 accumulation,
    3 (LATCH-DENSE-EXEC) the unrounded FP32 accumulator: the FP32 product of BF16-exact operands."""
    return 3 if spec.get('out_raw') else 2 if spec.get('out_acc') else 1 if spec.get('out_f32') else 0


def chunk_stages(k):
    """64-k stages per wgmma chain for a K loop of length ``k`` (0: one chain)."""
    if not LONGK_CHUNK or k <= LONGK_CHUNK_MIN_K or k % 64:
        return 0
    return int(LONGK_CHUNK_STAGES)


def admit(m, n, k, batch, ak, bkc, la_offset, lb_offset, *, engine=None, min_tiles=None, min_k=None):
    """Return the engine name for an aligned BF16 product, or None (P-OPS-C body)."""
    engine = ENGINE if engine is None else engine
    if engine not in ENGINES:
        raise EngineRefusal(f'unknown matrix engine {engine}')
    if engine == 'popsc' or batch != 1:
        return None
    if m % 64 or n % 64 or k % 64 or 2 * m * n * k < MIN_FLOPS:
        return None
    min_k = MIN_K_STAGES if min_k is None else min_k
    if k // 64 < min_k or ((m // 128) * (n // 128) < (MIN_TILE_FRACTION * ADMIT_GRID if min_tiles is None else min_tiles)
                           and not (LONGK_SPLIT and longk_split_admissible(m, n, k))):
        return None
    # Alignment and estimated occupancy do not establish tile feasibility.
    # In particular, wide loop admission can encounter N=64 mod 128 while
    # every selected P5C tile requires N divisible by128. Keep the supported
    # small-tile body instead of failing later during code emission.
    try:
        if engine == 'p5c':
            p5c_plan(m, n, k, ak, bkc, grid=ADMIT_GRID)
        else:
            p4b_plan(m, n, k, ak, bkc, grid=ADMIT_GRID)
    except EngineRefusal:
        return None
    if P5C_DIAG_ONLY_FIRST is not None:
        if _DIAG_ADMITTED[0] >= P5C_DIAG_ONLY_FIRST:
            return None
        _DIAG_ADMITTED[0] += 1
    return engine


def p4b_plan(m, n, k, ak, bkc, *, grid):
    """P-OPS4-B wg_plan (aligned extents): 64-wide k stages, cost = max(waves x (fixed + max(MMA, stream)), L2)."""
    nk = k // 64
    candidates = []
    for bm, bn, wm in P4B_TILES:
        wtn = bn // (2 // wm)
        if not bkc and wtn % 64:
            continue
        if m % bm or n % bn:
            continue
        stage_bytes = (bm + bn) * 128
        nst = min(4, nk + 1, (P4B_SHARED_LIMIT - 1024) // stage_bytes)
        if nst < 2:
            continue
        tiles = (m // bm) * (n // bn)
        waves = math.ceil(tiles / grid)
        compute = 2 * bm * bn * k / P4B_SM_FLOPS_PER_US
        stream = (bm + bn) * k * 2 / SM_BYTES_PER_US
        local = waves * (P4B_FIXED_US + max(compute, stream))
        traffic = (m * k * (n // bn) + k * n * (m // bm)) * 2 / L2_BYTES_PER_US
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                           'shared_bytes': nst * stage_bytes + 1024, 'local_us': round(local, 4),
                           'traffic_us': round(traffic, 4), 'estimate_us': round(max(local, traffic), 4)})
    if not candidates:
        raise EngineRefusal(f'no admitted warpgroup-MMA tile for {(m, n, k)}')
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm'])), candidates


# P9-B: 128 x 96 tiles for K-major B operands (an N = 768 output becomes 128 tiles: one full wave of the grid)
P5C_BN96 = False
P9_TILE64 = False   # P9-B: 128 x 64 tiles for products whose 128 x 128 plan fills at most half the grid
# P9-B: 128 x 192 tiles (two warpgroups of 64 x 192) when they fill whole waves of the grid and 128 x 256 does not
# (N = 3072 outputs with M = 2048: 256 tiles, two full waves, instead of 192 tiles of 128 x 256 in 1.5 waves).
P5C_BN192 = False


def p5c_plan(m, n, k, ak, bkc, *, grid):
    nk = k // 64
    candidates = []
    for bm, bn, wm in (tuple(tuple(t) for t in P5C_TILES) + (((128, 96, 2),) if P5C_BN96 and bkc else ())
                       + (((128, 64, 2),) if P9_TILE64 else ())
                       + (((128, 192, 2),) if P5C_BN192 else ())):
        if m % bm or n % bn:
            continue
        if (not ak and bm % 64) or (not bkc and (bn // (2 // wm)) % 64):
            continue
        if chunk_stages(k) and (bm // wm != 64 or bn // (2 // wm) > 128):
            continue    # chunked accumulation keeps a second register slab: one 64 x <=128 slab per warpgroup
        stage_bytes = (bm + bn) * 128
        staged = staging_bytes() if (P5C_STAGED and (bm, bn) == (128, 128)) else 0
        nst = min(P5C_MAX_STAGES, nk, (P5C_SHARED_LIMIT - 1024 - staged) // stage_bytes)
        if staged:
            nst = min(nst, (carveout() - 1024 - staged) // stage_bytes)    # keep the program's carve-out
        if nst < 2:
            continue
        tiles = (m // bm) * (n // bn)
        waves = math.ceil(tiles / grid)
        active = min(grid, tiles)
        mma = 2 * bm * bn * 64 / P5C_SM_FLOPS_PER_US
        if not ak and bn // (2 // wm) < 256:
            mma *= P5C_MN_MAJOR_A_PENALTY
        stream = stage_bytes * active / P5C_L2_BYTES_PER_US
        estimate = waves * (P5C_FIXED_US + nk * max(mma, stream))
        # local/traffic split for matrix_rules.emit_group's concurrent-slice search (compute vs shared L2 stream).
        local = waves * (P5C_FIXED_US + nk * mma)
        traffic = tiles * nk * stage_bytes / P5C_L2_BYTES_PER_US
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
                           'shared_bytes': nst * stage_bytes + 1024 + staged, 'step_us': round(max(mma, stream), 4),
                           'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(estimate, 4)})
    if not candidates:
        raise EngineRefusal(f'no admitted p5c tile for {(m, n, k)}')
    forced = P5C_FORCE.get((m, n, k, ak, bkc)) or P5C_FORCE.get('all')
    if forced:
        chosen = [c for c in candidates if (c['bm'], c['bn']) == tuple(forced)]
        if chosen:
            return dict(chosen[0], forced=True), candidates
    best = min([c for c in candidates if (c['bm'], c['bn']) != (128, 64)] or candidates,
               key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))
    if P9_TILE64:
        # P9-B: half-grid products (few long tiles, e.g. 768 x 768 weight gradients) take 128 x 64 tiles: twice the
        # tiles, half the longest tile; every output element keeps its k16 chain (bitwise).
        c64 = next((c for c in candidates if (c['bm'], c['bn']) == (128, 64)), None)
        if c64 and (best['bm'], best['bn']) == (128, 128) and 2 * best['tiles'] <= grid:
            best = c64
    if P5C_BN96:
        # P9-B: the L2-stream term overstates the cost of filling the grid (multicast probe gpu-28: these bodies are
        # not L2-bound); take 128 x 96 when it fills whole waves and 128 x 128 does not.
        c96 = next((c for c in candidates if (c['bm'], c['bn']) == (128, 96)), None)
        c128 = next((c for c in candidates if (c['bm'], c['bn']) == (128, 128)), None)
        if c96 and c128 and c96['tiles'] % grid == 0 and c128['tiles'] % grid:
            best = c96
    if P5C_BN192:
        c192 = next((c for c in candidates if (c['bm'], c['bn']) == (128, 192)), None)
        if c192 and c192['tiles'] % grid == 0 and best['tiles'] % grid and (best['bm'], best['bn']) != (128, 96):
            best = c192
        elif c192 and best is c192 and c192['tiles'] % grid:
            best = min((c for c in candidates if c is not c192), key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']),
                       default=c192)
    return best, candidates


def tile_plan(m, n, k, ak, bkc, tile, *, grid, max_stages=None):
    """LATCH-STEP10-SPEED: the plan of one explicit (bm, bn, wm) p5c tile, with p5c_plan's stage and cost model
    (used for products whose tile is chosen by their operand owner, e.g. im2col convolution operands)."""
    bm, bn, wm = tile
    if m % bm or n % bn or k % 64 or (not ak and bm % 64) or (not bkc and (bn // (2 // wm)) % 64):
        raise EngineRefusal(f'tile {tile} does not fit {(m, n, k, ak, bkc)}')
    nk = k // 64
    stage_bytes = (bm + bn) * 128
    nst = min(max_stages or P5C_MAX_STAGES, nk, (P5C_SHARED_LIMIT - 1024) // stage_bytes)
    if nst < 2:
        raise EngineRefusal(f'tile {tile}: fewer than two stages')
    tiles = (m // bm) * (n // bn)
    waves = math.ceil(tiles / grid)
    mma = 2 * bm * bn * 64 / P5C_SM_FLOPS_PER_US
    stream = stage_bytes * min(grid, tiles) / P5C_L2_BYTES_PER_US
    return {'bm': bm, 'bn': bn, 'wm': wm, 'nst': nst, 'tiles': tiles, 'waves': waves, 'grid': grid,
            'shared_bytes': nst * stage_bytes + 1024, 'step_us': round(max(mma, stream), 4),
            'local_us': round(waves * (P5C_FIXED_US + nk * mma), 4),
            'traffic_us': round(tiles * nk * stage_bytes / P5C_L2_BYTES_PER_US, 4),
            'estimate_us': round(waves * (P5C_FIXED_US + nk * max(mma, stream)), 4), 'forced': True}


def pingpong_plan(spec, grid):
    """Ping-pong per-warpgroup tiles when selected and every CTA gets an even number of tiles (no recompute)."""
    if P5C_PINGPONG == 'never' or (P5C_PINGPONG == 'epilogue' and not spec.get('epi')):
        return None
    # P6-D: 'optimizer' -- only FP32-converting products carrying a generated (memory-bound state-update) epilogue.
    if P5C_PINGPONG == 'optimizer' and not (spec.get('epi') and spec.get('out_f32')):
        return None
    bm, bn = P5C_PP_TILE
    m, n, k = spec['m'], spec['n'], spec['k']
    if m % bm or n % bn or (not spec['bkc'] and bn % 64):
        return None
    tiles = (m // bm) * (n // bn)
    if (tiles % (2 * grid)) if not P5C_PP_ANY_TILES else tiles < grid:
        return None
    stage_bytes = (bm + bn) * 128
    nstw = min(P5C_PP_STAGES, (P5C_SHARED_LIMIT - 1024) // (2 * stage_bytes))
    if nstw < 2:
        return None
    waves = math.ceil(tiles / grid)
    mma = 2 * bm * bn * 64 / P5C_SM_FLOPS_PER_US
    stream = stage_bytes * grid / P5C_L2_BYTES_PER_US
    estimate = waves * (k // 64) * max(mma, stream) + P5C_FIXED_US
    return {'bm': bm, 'bn': bn, 'wm': 0, 'nst': nstw, 'pingpong': True, 'tiles': tiles, 'waves': waves,
            'grid': grid, 'shared_bytes': 2 * nstw * stage_bytes + 1024, 'estimate_us': round(estimate, 4),
            'local_us': round(waves * (k // 64) * mma + P5C_FIXED_US, 4),
            'traffic_us': round(tiles * (k // 64) * stage_bytes / P5C_L2_BYTES_PER_US, 4),
            'release': False, 'out_f32': bool(spec.get('out_f32'))}


def wg_headers():
    """Headers of admitted warpgroup-MMA products (P9WS adds kernels/p8ws.cuh)."""
    # P9-B: the loop composer emits no source prelude, so main-loop selections travel as selector headers placed
    # before kernels/p5c_wg.cuh.
    pre = (('p5c_sync2.cuh',) if P5C_SYNC2 else ()) + (('p5c_decoupled.cuh',) if P5C_DECOUPLED else ())
    pre += ('p5c_trace.cuh',) if P5C_TRACE else ()
    pre += ('p5c_staged.cuh',) if P5C_STAGED else ()
    pre += ('p5c_accbatch.cuh',) if P5C_ACC_BATCH else ()
    pre += ('p5c_accpf.cuh',) if P5C_ACC_PREFETCH else ()
    pre += ('p5c_parpre.cuh',) if P5C_PAR_PRE else ()
    pre += ('p5c_epipf.cuh',) if P5C_EPI_PREFETCH else ()
    pre += ('p5c_bulk.cuh',) if P5C_BULK_EPI else ()
    base = WG_HEADERS[:-1] + pre + WG_HEADERS[-1:]
    return (base + (('p8ws.cuh',) if P8WS else ())
            + ((('p8ws_region.cuh', 'p9ws.cuh') if P9WS_ENGINE == 'p9ws' else ('p8ws_region.cuh',)) if P9WS else ()))


def flat_ws_plan(spec, grid, rows=None):
    """P8-A: warp-specialized plan (BM x 128 tiles) or None when the shapes/options do not admit it."""
    if not P8WS or not spec.get('ws_ok') or P5C_MULTICAST or spec.get('splitk') or spec.get('pointers'):
        return None
    if (P8WS_ALU_RULE and spec.get('epi') and _alu_heavy(spec['epi']['run'])
            and not (IM2COL_WS_ALU and spec.get('virtual_a'))):
        # LATCH-STEP10-SPEED: implicit-GEMM (im2col A) products keep the warp-specialized body with transcendental
        # epilogues (measured on the step-10 development block: 0.3222 -> 0.3081 ms).
        return None
    m, n, k = spec['m'], spec['n'], spec['k']
    if n % 128 or k % 64 or m % 64:
        return None
    bm = rows or (128 if m % 128 == 0 and (m // 128) * (n // 128) > grid // 2 else 64)
    if (P9_LONGK_BM64 and not rows and bm == 128 and k >= P9_LONGK_BM64 and (m // 128) * (n // 128) <= grid
            and m % 64 == 0):
        bm = 64   # P9-A: one long tile per CTA leaves its epilogue unoverlapped; two 64-row tiles overlap it
    if m % bm:
        return None
    pk = chunk_stages(k)
    if pk and bm * 128 * 4 > P8WS_STAGING * 128 * 128 * 2:
        return None     # the chunk sum lives in the staging buffers
    stage_bytes = (bm + 128) * 128
    # The flat p8ws group uses fixed 128-row operand slots so BM64 and
    # BM128 items can share one continuous ring (p8ws::STAGE/shared_bytes).
    # BM64 transfers fewer bytes but does not shrink those physical slots.
    ring_stage = (128 + 128) * 128
    staging = P8WS_STAGING * 128 * 128 * 2 + 1024
    nst = min(P8WS_STAGES, k // 64)
    if P8WS_FIT_SHARED_LIMIT:
        nst = min(nst, (P8WS_SHARED_LIMIT - staging) // ring_stage)
    shared = nst * ring_stage + staging
    if nst < 2 or shared > P8WS_SHARED_LIMIT:
        return None
    tiles = (m // bm) * (n // 128)
    waves = math.ceil(tiles / grid)
    active = min(grid, tiles)
    mma = 2 * bm * 128 * 64 / (P5C_SM_FLOPS_PER_US * P8WS_MMA_SPEEDUP)
    stream = stage_bytes * active / P5C_L2_BYTES_PER_US
    nk = k // 64
    return {'bm': bm, 'bn': 128, 'wm': 0, 'nst': nst, 'ws': True, 'ws_engine': 'p8ws_flat', 'tiles': tiles, 'waves': waves, 'grid': grid,
            'shared_bytes': shared, 'step_us': round(max(mma, stream), 4),
            'local_us': round(waves * (P5C_FIXED_US + nk * mma), 4),
            'traffic_us': round(tiles * nk * stage_bytes / P5C_L2_BYTES_PER_US, 4),
            'estimate_us': round(waves * (P5C_FIXED_US + nk * max(mma, stream)), 4),
            'release': False, 'out_f32': bool(spec.get('out_f32')), 'mc': 1,
            **({'chunk_stages': pk} if pk else {})}

def region_ws_plan(spec, grid):
    """P9-B: warp-specialized plan (BM x 128 tiles) or None when the shapes/options do not admit it."""
    if not P9WS or P5C_MULTICAST or spec.get('splitk') or chunk_stages(spec['k']):
        return None
    if not spec['ak'] and not P9WS_MN_A:
        return None
    if P9WS_ALU_RULE and spec.get('epi') and _alu_heavy(spec['epi']['run']):
        return None
    m, n, k = spec['m'], spec['n'], spec['k']
    bn = 128
    if P9WS_ENGINE == 'p9ws':
        # Busiest CTA's time = waves x (k steps x step time + per-tile staging); step times measured standalone
        # (p9_b/probes, gpu-10/12/20: 128x128 ~0.43 us, 128x96 ~0.37 us per 64-k step; ~1.6 us per tile).
        # 64-row tiles only when M is not a multiple of 128 (they run each step at about the 128-row tile's cost).
        step = {(128, 128): 0.43, (128, 96): 0.37, (64, 128): 0.40, (64, 96): 0.37}
        options = []
        for bm_ in ((128,) if m % 128 == 0 else (64,)):
            for bn_ in ((128, 96) if spec['bkc'] else (128,)):
                if n % bn_ or m % bm_:
                    continue
                tiles_ = (m // bm_) * (n // bn_)
                options.append((math.ceil(tiles_ / grid) * ((k // 64) * step[(bm_, bn_)] + 1.6), -bn_, bm_, bn_))
        if not options or k % 64:
            return None
        _, _, bm, bn = min(options)
    else:
        if n % 128 or k % 64 or m % 64:
            return None
        bm = 128 if m % 128 == 0 and (m // 128) * (n // 128) > grid // 2 else 64
    if m % bm:
        return None
    stage_bytes = (bm + 128) * 128
    nst = min(P9WS_STAGES, k // 64)
    shared = nst * stage_bytes + 2 * 128 * 128 * 2 + 1024
    if nst < 2 or shared > P9WS_SHARED_LIMIT:
        return None
    if P9WS_ENGINE == 'p9ws':
        shared = nst * 256 * 128 + 2 * 128 * 128 * 2 + 1024      # kernels/p9ws.cuh shared_bytes<NST>()
        if shared > P9WS_SHARED_LIMIT:
            return None
    tiles = (m // bm) * (n // bn)
    waves = math.ceil(tiles / grid)
    mma = 2 * bm * bn * 64 / (P5C_SM_FLOPS_PER_US * 1.3)
    nk = k // 64
    return {'bm': bm, 'bn': bn, 'ws_engine': P9WS_ENGINE, 'wm': 0, 'nst': nst, 'ws': True, 'tiles': tiles, 'waves': waves, 'grid': grid,
            'shared_bytes': shared, 'step_us': round(mma, 4), 'local_us': round(waves * (P5C_FIXED_US + nk * mma), 4),
            'traffic_us': round(tiles * nk * stage_bytes / P5C_L2_BYTES_PER_US, 4),
            'estimate_us': round(waves * (P5C_FIXED_US + nk * mma), 4),
            'release': False, 'out_f32': bool(spec.get('out_f32')), 'mc': 1}

def ws_plan(spec, grid, rows=None):
    """Choose the admitted flat or region protocol from the operand binding contract."""
    if spec.get('ws_ok') and not spec.get('pointers'):
        return flat_ws_plan(spec, grid, rows=rows)
    return region_ws_plan(spec, grid)


def prefer_cooperative_chunks(spec, grid):
    """Measured flat long-K regime; retain existing paths outside its contract.

    Selection uses shape, operand layout and epilogue capabilities only. The
    explicit P5C adapter keeps its historical single-chain diagnostic; this
    rule applies to normal compilation, whose existing PK policy is preserved.
    """
    if (not P5C_COOPERATIVE_CHUNKS or not spec.get('ws_ok')
            or not chunk_stages(spec['k']) or spec['bkc']
            or any(spec.get(key) for key in ('pointers', 'splitk', 'out_raw', 'virtual_a', 'virtual_b', 'p9_delta'))):
        return False
    m, n = spec['m'], spec['n']
    minimum = max(P5C_COOPERATIVE_CHUNK_MIN_TILES, P5C_COOPERATIVE_CHUNK_MIN_WAVES * grid)
    if m % 128 or n % 128 or (m // 128) * (n // 128) < minimum:
        return False
    from . import flat_long_attention
    return spec.get('out') not in flat_long_attention.KN_EPI_TARGETS


def plan(spec, grid):
    if spec['engine'] == 'p4b_tma':
        return p4b_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], grid=grid)[0]
    if spec['engine'] == 'p5c' and spec.get('splitk'):
        # P7-C split-K products use 128 x 128 tiles (plan of that tile, work units = tiles x slices).
        if spec.get('p5c_tile') or spec.get('max_stages'):
            # Convolution owns its explicit tile/ring; preserve the donor split arithmetic.
            chosen = tile_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'],
                               tuple(spec.get('p5c_tile') or (128, 128, 2)), grid=grid,
                               max_stages=spec.get('max_stages'))
            pk = None
        else:
            chosen, candidates = p5c_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], grid=grid)
            chosen = next(c for c in candidates if (c['bm'], c['bn']) == (128, 128))
            pk = chunk_stages(spec['k'] // spec['splitk'])
        return dict(chosen, tiles=chosen['tiles'] * spec['splitk'], release=False, out_f32=bool(spec.get('out_f32')),
                    mc=1, splitk=spec['splitk'], **({'chunk_stages': pk} if pk else {}))
    if spec['engine'] == 'p5c' and spec.get('p5c_tile'):
        # LATCH-STEP10-SPEED: an operand-owner tile (conv_ops.TapsOperands) on the plain p5c body.
        chosen = tile_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], tuple(spec['p5c_tile']), grid=grid,
                           max_stages=spec.get('max_stages'))
        return dict(chosen, release=P5C_RELEASE, out_f32=bool(spec.get('out_f32')), mc=1)
    if spec['engine'] == 'p5c' and spec.get('out_raw'):
        # LATCH-DENSE-EXEC: raw FP32 results are emitted only by the plain body's unstaged store (no warp-specialized,
        # ping-pong, multicast or split-K forms; those stage or exchange BF16-rounded tiles).
        # LATCH-FFN-RETAIN: the flat warp-specialized body stages raw tiles in FP32 (p8ws OUT == 3) when admitted.
        pk = chunk_stages(spec['k'])
        # Measured (LATCH-FFN-RETAIN v8): with 10 k-steps per tile the FP32 epilogue of 3 epilogue warps is the
        # bottleneck (0.150 vs 0.114 ms plain); with 40 it hides under the mainloop (0.113 vs 0.158 ms).
        if P8WS_RAW and not pk and P8WS_STAGING == 2 and spec['k'] // 64 >= P8WS_RAW_MIN_STEPS:
            ws = flat_ws_plan(spec, grid)
            if ws is not None and ws['bm'] in (64, 128):
                return dict(ws, out_raw=True)
        chosen, _ = p5c_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], grid=grid)
        if pk:
            return dict(chosen, release=False, out_f32=False, out_raw=True, mc=1, chunk_stages=pk)
        return dict(chosen, release=P5C_RELEASE, out_f32=False, out_raw=True, mc=1)
    if spec['engine'] == 'p5c':
        pk = chunk_stages(spec['k'])
        ws = ws_plan(spec, grid)
        cooperative_chunks = ws is not None and prefer_cooperative_chunks(spec, grid)
        if ws is not None and not cooperative_chunks:
            from .vocab_head import prefer_wide_tile    # release step 3: vocabulary-scale L2-bound products
            return (None if pk else prefer_wide_tile(spec, ws, grid)) or ws
        pp = None if pk else pingpong_plan(spec, grid)
        if pp is not None:
            return pp
        chosen, candidates = p5c_plan(spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], grid=grid)
        if (spec.get('epi') and P5C_EPI_TILE and not chosen.get('forced')
                and not (spec.get('out_f32') and not P5C_EPI_TILE_F32)):
            chosen = next((c for c in candidates if (c['bm'], c['bn']) == tuple(P5C_EPI_TILE)), chosen)
        mc = 2 if (P5C_MULTICAST and chosen['bm'] % 128 == 0 and (spec['n'] // chosen['bn']) % 2 == 0
                   and not spec.get('virtual_a') and not spec.get('virtual_b')) else 1
        if spec.get('max_stages'):
            # LATCH-STEP10-SPEED: operand-owner stage ring (e.g. im2col products), within the shared-memory limit.
            stage_bytes = (chosen['bm'] + chosen['bn']) * 128
            nst = max(2, min(int(spec['max_stages']), spec['k'] // 64, (P5C_SHARED_LIMIT - 1024) // stage_bytes))
            chosen = dict(chosen, nst=nst, shared_bytes=nst * stage_bytes + 1024)
        if pk:
            return dict(chosen, release=False, out_f32=bool(spec.get('out_f32')), mc=1, chunk_stages=pk,
                        **({'chunk_selection': 'cooperative register sums for full-wave N-major-B contractions'}
                           if cooperative_chunks else {}))
        return dict(chosen, release=P5C_RELEASE, out_f32=bool(spec.get('out_f32')), mc=mc)
    raise EngineRefusal(spec['engine'])


def _marker(root, offset, kmajor, rows, k, stride, box_rows, pointer=None):
    # K-major operand: dims {K, rows}, box {64, box_rows}; rows-major: dims {rows, K}, box {64, 64}.
    dims, box = ((k, rows), (64, box_rows)) if kmajor else ((rows, k), (64, 64))
    if pointer is not None:
        # P7-C loop regions: the operand's storage is bound per region instance (tape slot, scratch parity); the
        # loop composer lists every possible binding and the code picks the map whose address equals the pointer.
        return f'{TMAP_REGION}({root},{offset},{dims[0]},{dims[1]},{2 * stride},{box[0]},{box[1]}|{pointer})'
    return f'{TMAP_MARK}({root},{offset},{dims[0]},{dims[1]},{2 * stride},{box[0]},{box[1]})'


TMAP_REGION = 'P5C_TMAPR'


def im2col_marker(vi, ppc):
    """LATCH-STEP10-SPEED: im2col map of a dense pixel-major (N*H*W, C) BF16 image as a 4-D (C, W, H, N) tensor;
    64-channel x ppc-pixel boxes, bounding box corners (-1, -1) / (-1, -1) (3x3, stride 1, padding 1)."""
    return (f"{TMAP_MARK}({vi['root']},{vi['offset']},{vi['channels']},{vi['w']},{vi['h']},{vi['n']},"
            f"i2c,{ppc})")


def _im2col_code(spec, plan_, text):
    """Rewrite the tiled map of the virtual operand to its im2col map and set the Args::vi_* fields."""
    vi, operand = (spec['virtual_a'], 0) if spec.get('virtual_a') else (spec['virtual_b'], 1)
    if spec['engine'] != 'p5c' or 'p5c::Args g{' not in text or spec.get('pointers'):
        raise EngineRefusal('im2col operands require a flat p5c::Args body')
    sam, sak, sbk, sbn, _, _ = spec['strides']
    if operand == 0:
        if not spec['ak']:
            raise EngineRefusal('an im2col A operand must be K-major')
        (ra, oa) = spec['roots'][0]
        original = _marker(ra, oa, True, spec.get('m_extent', spec['m']), spec.get('k_extent', spec['k']), sam,
                           plan_['bm'])
        virtual = im2col_marker(vi, plan_['bm'])
    else:
        if spec['bkc']:
            raise EngineRefusal('an im2col B operand must be N-major')
        (rb, ob) = spec['roots'][1]
        original = _marker(rb, ob, False, spec.get('n_extent', spec['n']), spec.get('k_extent', spec['k']), sbk,
                           plan_['bn'])
        virtual = im2col_marker(vi, 64)
    if text.count(original) != 1:
        raise EngineRefusal('im2col operand map placeholder not found exactly once')
    text = text.replace(original, virtual)
    assign = (f'  g.vi_c = {vi["channels"]}; g.vi_h = {vi["h"]}; g.vi_w = {vi["w"]}; '
              f'g.vi_s = {-1 if vi["flip"] else 1}; g.vi_b = {operand};\n')
    text, count = re.subn(r'(  p5c::Args g\{[^\n]*\};\n)', lambda m: m.group(1) + assign, text, count=1)
    if count != 1:
        raise EngineRefusal('im2col operand needs one p5c::Args construction')
    return text


def code(spec, plan_, ident, lo, hi, den, rot):
    text = _code(spec, plan_, ident, lo, hi, den, rot)
    if spec.get('virtual_a') or (spec.get('virtual_b') or {}).get('kind') == 'im2col':
        return _im2col_code(spec, plan_, text)
    vb = spec.get('virtual_b')
    if not vb:
        return text
    # Step 10 (ops/conv_ops.py): the B operand is read in place from a padded channel-major image through
    # tap-shifted {64, 64} boxes (kernels/p5c_wg.cuh vb_issue); the logical tap matrix has no storage.
    if spec['engine'] != 'p5c' or 'p5c::Args g{' not in text or spec.get('pointers'):
        raise EngineRefusal('tap-shifted operands require a flat p5c::Args body')
    sam, sak, sbk, sbn, _, _ = spec['strides']
    (rb, ob) = spec['roots'][1]
    original = _marker(rb, ob, spec['bkc'], spec['n'], spec['k'], sbn if spec['bkc'] else sbk, plan_['bn'])
    stored, rows = vb['stored'], 3 * vb['channels']
    virtual = (_marker(vb['root'], vb['offset'], True, rows, stored, stored, 64) if spec['bkc'] else
               _marker(vb['root'], vb['offset'], False, stored, rows, stored, 64))
    if text.count(original) != 1:
        raise EngineRefusal('tap-shifted operand map placeholder not found exactly once')
    text = text.replace(original, virtual)
    assign = (f'  g.vb_c = {vb["channels"]}; g.vb_w = {vb["width"]}; g.vb_m = {vb["margin"]}; '
              f'g.vb_s = {-1 if vb["flip"] else 1};\n')
    text, count = re.subn(r'(  p5c::Args g\{[^\n]*\};\n)', lambda m: m.group(1) + assign, text, count=1)
    if count != 1:
        raise EngineRefusal('tap-shifted operand needs one p5c::Args construction')
    return text


def _code(spec, plan_, ident, lo, hi, den, rot):
    sam, sak, sbk, sbn, _, _ = spec['strides']
    m, n, k = spec['m'], spec['n'], spec['k']
    t = lambda flag: str(flag).lower()
    (ra, oa), (rb, ob) = spec['roots']
    mc = plan_.get('mc', 1) if (lo, hi, den) == (0, 1, 1) else 1      # multicast needs the whole (even) grid
    pa, pb = (spec['pointers'] if spec.get('pointers') else (None, None))
    # Maps describe the true extents; rows/columns of a padded product beyond them are zero-filled (conv_ops).
    ke, me, ne = spec.get('k_extent', k), spec.get('m_extent', m), spec.get('n_extent', n)
    ta = _marker(ra, oa, spec['ak'], me, ke, sam if spec['ak'] else sak, plan_['bm'] // mc, pa)
    tb = _marker(rb, ob, spec['bkc'], ne, ke, sbn if spec['bkc'] else sbk, plan_['bn'], pb)
    if spec.get('rot_token') and rot == 0 and (lo, hi, den) == (0, 1, 1):
        rot = '/*C_VBASE*/0'      # P7-C staged regions: the stage's CTA rotation (lower.phase_code)
    if spec['engine'] == 'p4b_tma':
        call = (f'pops4b::tma_phase_inline<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},'
                f'{t(spec["bkc"])},{plan_["nst"]},true,true>')
        return ('{\n'
                f'  pops4b::TArgs g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec['engine'] == 'p5c' and spec.get('splitk'):
        out_mode = _out_mode(spec)
        call = (f'p5c::phase_sk<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                f'{plan_["nst"]},{out_mode},{spec["splitk"]}' +
                (f',{plan_["chunk_stages"]}>' if plan_.get('chunk_stages') else '>'))
        return ('{\n'
                f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}, {spec["part"]}, {spec["cnt"]}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot});\n'
                '}')
    if spec['engine'] == 'p5c' and plan_.get('ws'):
        # Partitioned flat products must publish the same registered key-norm table as serial groups.
        from . import flat_long_attention as _fla
        if (plan_.get('ws_engine') == 'p8ws_flat' and spec['out'] in _fla.KN_EPI_TARGETS
                and not spec.get('epi')):
            return group_code([spec], [plan_], [rot], ident, lo, hi, den)
        out_mode = _out_mode(spec)
        store = spec['out'] != 'nullptr'
        ns = ('p8ws' if plan_.get('ws_engine') == 'p8ws_flat' else
              'p9ws' if plan_.get('ws_engine') == 'p9ws' else 'p8ws_region')
        mode = P8WS_MODE if plan_.get('ws_engine') == 'p8ws_flat' else 0
        call = (f'{ns}::phase_mode<{mode},{plan_["bm"]},{plan_["bn"]},{t(spec["ak"])},{t(spec["bkc"])},'
                f'{plan_["nst"]},{out_mode},{t(store)}>')
        tail = (f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}' if spec.get('epi') else ', p5c::NoEpi(), p5c::NoLd()')
        prelude = spec['epi']['prelude'] + '\n' if spec.get('epi') and spec['epi']['prelude'] else ''
        if plan_.get('chunk_stages'):
            # Release step 3 repair: chunked accumulation item (flat warp-specialized protocol only).
            if plan_.get('ws_engine') != 'p8ws_flat':
                raise EngineRefusal('chunked accumulation requires the flat warp-specialized body')
            functors = tail[2:]
            return ('{\n' + prelude +
                    f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                    f'{m}, {n}, {k}}};\n'
                    f'  p8ws::group<{mode},{plan_["nst"]}>(smem, {lo}, {hi}, {den}, p8ws::item_k<{plan_["bm"]},'
                    f'{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)},{plan_["chunk_stages"]}>(g, {functors}, {rot}));\n'
                    '}')
        return ('{\n' + prelude +
                f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}{tail});\n'
                '}')
    if spec['engine'] == 'p5c' and plan_.get('pingpong'):
        if spec.get('epi') and spec.get('out_f32'):
            store = spec['out'] != 'nullptr'
            call = (f'p5c::pp_phase_epi_out<{plan_["bm"]},{plan_["bn"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},1,{t(store)}>')
            tail = f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}'
        elif spec.get('epi'):
            store = spec['out'] != 'nullptr'
            call = (f'p5c::pp_phase_epi<{plan_["bm"]},{plan_["bn"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{t(store)}>')
            tail = f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}'
        else:
            call = (f'p5c::pp_phase<{plan_["bm"]},{plan_["bn"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{1 if spec.get("out_f32") else 0}>')
            tail = ''
        prelude = spec['epi']['prelude'] + '\n' if spec.get('epi') and spec['epi']['prelude'] else ''
        return ('{\n' + prelude +
                f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}{tail});\n'
                '}')
    if spec['engine'] == 'p5c' and plan_.get('chunk_stages'):
        # Release step 3 repair: chunked accumulation forms of the same body (no multicast, no per-stage release).
        out_mode = _out_mode(spec)
        if spec.get('p9_delta'):
            raise EngineRefusal('chunked accumulation does not carry the attention delta epilogue')
        if spec.get('epi'):
            store = spec['out'] != 'nullptr'
            call = (f'p5c::phase_epi_k<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{out_mode},{t(store)},{plan_["chunk_stages"]}>')
            tail = f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}'
        else:
            call = (f'p5c::phase_k<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{out_mode},{plan_["chunk_stages"]}>')
            tail = ''
        prelude = spec['epi']['prelude'] + '\n' if spec.get('epi') and spec['epi']['prelude'] else ''
        return ('{\n' + prelude +
                f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}}};\n'
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}{tail});\n'
                '}')
    if spec['engine'] == 'p5c':
        if spec.get('epi') and spec.get('out_f32'):
            # P6-D: register epilogue on an FP32-converting store (exact widening of the rounded BF16 result).
            store = spec['out'] != 'nullptr'
            call = (f'p5c::phase_epi_out<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},1,{t(plan_["release"])},{t(store)}>')
            tail = f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}'
        elif spec.get('epi'):
            store = spec['out'] != 'nullptr'
            out_mode = _out_mode(spec)
            call = (f'p5c::phase_epi<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{t(plan_["release"])},{t(store)},{out_mode}' +
                    (f',decltype(p5c_epi_run),decltype(p5c_epi_ld),2>' if mc == 2 else '>'))
            tail = f', {spec["epi"]["run"]}, {spec["epi"]["ld"]}'
            if mc == 2:
                # Explicit lambda types (the multicast argument follows the deduced functor parameters).
                call = f'auto p5c_epi_run = {spec["epi"]["run"]};\n  auto p5c_epi_ld = {spec["epi"]["ld"]};\n  ' + call
                tail = ', p5c_epi_run, p5c_epi_ld'
        else:
            call = (f'p5c::phase<{plan_["bm"]},{plan_["bn"]},{plan_["wm"]},{t(spec["ak"])},{t(spec["bkc"])},'
                    f'{plan_["nst"]},{_out_mode(spec)},{t(plan_["release"])}' +
                    (',2>' if mc == 2 else '>'))
            tail = ''
        pf = (prefetch_functor(spec['epi']['ld'], bn=plan_['bn'], ns='p5c')
              if P5C_EPI_PREFETCH and spec.get('epi') and mc == 1 else None)
        if pf:
            # LATCH-STEP-06: the same body with its epilogue operand rows L2-prefetched at tile start.
            call = call.replace('p5c::phase_epi_out<', 'p5c::phase_epi_out_pf<').replace('p5c::phase_epi<', 'p5c::phase_epi_pf<')
            tail += f', {pf}'
        if mc == 2:
            call = '/*P5C_CLUSTER2*/ ' + call
        prelude = spec['epi']['prelude'] + '\n' if spec.get('epi') and spec['epi']['prelude'] else ''
        dlt = spec.get('p9_delta')
        dlt_line = (f'  g.dlt_o32 = {dlt["o32"]}; g.dlt_out = plong::plong_delta[{dlt["slot"]}]; '
                    f'g.dlt_t = {dlt["t"]}; g.dlt_heads = {dlt["heads"]};\n') if dlt else ''
        if dlt and dlt.get("q"):
            dlt_line += (f'  g.qn_q = {dlt["q"]}; g.qn_out = p9att::p9_qnmax[{dlt["slot"]}]; g.qn_sb = {dlt["lq"][0]}LL; '
                         f'g.qn_sh = {dlt["lq"][1]}LL; g.qn_sr = {dlt["lq"][2]}LL;\n')
        return ('{\n' + prelude +
                f'  p5c::Args g{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                f'{m}, {n}, {k}}};\n' + dlt_line +
                f'  {call}(g, smem, {lo}, {hi}, {den}, {rot}{tail});\n'
                '}')
    raise EngineRefusal(spec['engine'])


P9WS_GROUP = True    # consecutive p9ws products of one stage run as one continuous pipeline (lower.py stage assembly)


def ws_groupable(spec, plan_):
    """A product the stage assembler may merge into one p9ws group call (empty epilogue prelude, whole grid)."""
    return bool(plan_ and plan_.get('ws') and plan_.get('ws_engine') == 'p9ws' and spec.get('engine') == 'p5c'
                and not (spec.get('epi') and spec['epi'].get('prelude')))


P9WS_GROUPS_ONLY = False   # P9-B: warp-specialized plans only for products that end up in a group of >= 2
P9WS_BALANCE = True    # P9-B: groups use a compile-time longest-first tile schedule over the planned grid
P9WS_STEP_US = {(128, 128): 0.43, (128, 96): 0.37, (64, 128): 0.40, (64, 96): 0.37, (64, 64): 0.35, (128, 64): 0.40}


def ws_balance(specs, plans, grid, init=None):
    """P9-B: (table, loads): longest-processing-time assignment of every tile of a group's products to the grid's CTAs
    (tile cost = 64-k steps x step time + 1.6 us staging, kernels measured standalone). The table holds, per CTA, the
    prefix offsets of each member's tiles followed by the concatenated per-CTA tile lists (ascending tile index)."""
    import heapq
    tiles = []
    for k, (spec, plan_) in enumerate(zip(specs, plans)):
        bm, bn = plan_['bm'], plan_['bn']
        cost = (spec['k'] // 64) * P9WS_STEP_US.get((bm, bn), 0.43) + 1.6
        for u in range((spec['m'] // bm) * (spec['n'] // bn)):
            tiles.append((-cost, k, u))
    tiles.sort()
    init = init if init and len(init) == grid else None
    heap = [(float(init[c]) if init else 0.0, c) for c in range(grid)]
    heapq.heapify(heap)
    owned = [[[] for _ in specs] for _ in range(grid)]
    for neg, k, u in tiles:
        load, c = heapq.heappop(heap)
        owned[c][k].append(u)
        heapq.heappush(heap, (load - neg, c))
    n = len(specs)
    offsets, lists = [], []
    for c in range(grid):
        for k in range(n):
            offsets.append(len(lists))
            lists += sorted(owned[c][k])
        offsets.append(len(lists))
    if len(offsets) + len(lists) >= 65536:
        return None, None
    return offsets + lists, sorted(load for load, _ in heap)


# P9-B: [[bm, bn], factor] step-cost factors relative to area scaling (None: area only). gpu-117: 6.379 vs 6.402 ms.
P9_TILE_EFF = (((128, 256), 0.72), ((256, 128), 0.72))
P9_TILE_COST = (2.5, 0.40, 0.55, 1.5, 4.0)   # fixed, K-major step, M/N-major step, FP32/acc store, epilogue (us)
P9_DELTA_EPI_US = 0.0  # LATCH-STEP-06: extra per-tile cost of the attention delta/q-norm epilogue in the stage balance


def p5c_tile_cost(spec, plan_):
    """P9-B: in-kernel time (us) of one p5c tile, fitted to the phase-tick profile (gpu-30): fixed cost, 64-k steps
    (K-major operands 0.40 us, M/N-major 0.55 us per 128 x 128 step), FP32/accumulating stores and generated
    epilogues per 128 x 128."""
    fixed, step_k, step_mn, out_us, epi_us = P9_TILE_COST
    scale = plan_['bm'] * plan_['bn'] / 16384
    if P9_TILE_EFF:
        # P9-B: per-shape step efficiency (per-tile trace gpu-85/106: a 128 x 256 step costs ~1.45x a 128 x 128 step)
        scale *= dict((tuple(k_), v_) for k_, v_ in P9_TILE_EFF).get((plan_['bm'], plan_['bn']), 1.0)
    step = step_k if spec['ak'] and spec['bkc'] else step_mn
    cost = fixed + (spec['k'] // 64) * step * scale
    if spec.get('out_f32') or spec.get('out_acc') or spec.get('out_raw'):
        cost += out_us * scale
    if spec.get('epi'):
        cost += epi_us * scale
    if spec.get('p9_delta'):
        # LATCH-STEP-06: the dO product's staged epilogue also forms the attention delta and q-norm tables from saved
        # O (FP32) and Q rows (per-tile trace, g-05: 9.1-9.9 us epilogue vs 4.5 us main loop).
        cost += P9_DELTA_EPI_US
    return cost


def stage_entries(table, n, grid):
    """P9-B: per-(CTA, member) 64-bit schedule entries (p5c::Args::srow) and the tile lists, from a balance table."""
    head = grid * (n + 1)
    entries = []
    for c in range(grid):
        for k in range(n):
            b, e = table[c * (n + 1) + k], table[c * (n + 1) + k + 1]
            first = table[head + b] if e > b else 0
            entries.append((1 << 63) | ((e - b) << 48) | (first << 32) | b)
    return entries, table[head:]


P9_BODY_US = None     # P9-B: per-(CTA, product) body cost (us) in the stage tile balance (None: plain LPT)


def stage_balance(specs, plans, grid, init=None):
    """P9-B: longest-first assignment of every tile of a stage's p5c products (table layout of ws_balance)."""
    import heapq
    tiles = []
    for k, (spec, plan_) in enumerate(zip(specs, plans)):
        cost = p5c_tile_cost(spec, plan_)
        for u in range((spec['m'] // plan_['bm']) * (spec['n'] // plan_['bn'])):
            tiles.append((-cost, k, u))
    tiles.sort()
    init = init if init and len(init) == grid else None
    heap = [(float(init[c]) if init else 0.0, c) for c in range(grid)]
    heapq.heapify(heap)
    owned = [[[] for _ in specs] for _ in range(grid)]
    if P9_BODY_US:
        # P9-B: a CTA's first tile of a product also pays the body start (barrier setup, cold descriptors, cold
        # epilogue code; per-tile trace gpu-85): place each tile where it finishes first, counting that cost.
        load = [float(init[c]) if init else 0.0 for c in range(grid)]
        for neg, k, u in tiles:
            best, bc = None, -1
            for c in range(grid):
                f = load[c] - neg + (0.0 if owned[c][k] else P9_BODY_US)
                if best is None or f < best:
                    best, bc = f, c
            owned[bc][k].append(u)
            load[bc] = best
        heap = [(load[c], c) for c in range(grid)]
    else:
        for neg, k, u in tiles:
            load, c = heapq.heappop(heap)
            owned[c][k].append(u)
            heapq.heappush(heap, (load - neg, c))
    n = len(specs)
    offsets, lists = [], []
    for c in range(grid):
        for k in range(n):
            offsets.append(len(lists))
            lists += sorted(owned[c][k])
        offsets.append(len(lists))
    if len(offsets) + len(lists) >= 65536:
        return None, None
    return offsets + lists, sorted(load for load, _ in heap)


def ws_group_code(specs, plans, rots, ident, init=None):
    """P9-B: one p9ws::group call over several independent products (each with its own CTA rotation); the stage ring,
    staging buffers and barriers carry across members, so the next product's loads and MMAs overlap the previous
    product's last epilogue (P8-A's grouping, kernels/p8ws.cuh group())."""
    if any(spec.get('virtual_b') or spec.get('virtual_a') for spec in specs):
        raise EngineRefusal('tap-shifted operands are emitted only by single-product p5c code')
    t = lambda flag: str(flag).lower()
    decls, items = [], []
    nst = min(plan_['nst'] for plan_ in plans)
    grid = int(plans[0].get('grid', 128))
    table = ws_balance(specs, plans, grid, init)[0] if P9WS_BALANCE else None
    helper, sched = None, ''
    if table:
        import hashlib as _h
        body = ', '.join(str(x) for x in table)
        name = 'p9s_' + _h.sha256(body.encode()).hexdigest()[:12]
        helper = f'__device__ const unsigned short {name}[{len(table)}] = {{{body}}};'
        sched = f'{name}, @K@, {len(specs)}, {grid}'

    for index, (spec, plan_, rot) in enumerate(zip(specs, plans, rots)):
        sam, sak, sbk, sbn, _, _ = spec['strides']
        m, n, k = spec['m'], spec['n'], spec['k']
        (ra, oa), (rb, ob) = spec['roots']
        pa, pb = (spec['pointers'] if spec.get('pointers') else (None, None))
        ta = _marker(ra, oa, spec['ak'], m, k, sam if spec['ak'] else sak, plan_['bm'], pa)
        tb = _marker(rb, ob, spec['bkc'], n, k, sbn if spec['bkc'] else sbk, plan_['bn'], pb)
        out_mode = _out_mode(spec)
        store = spec['out'] != 'nullptr'
        decls.append(f'  p5c::Args g{index}{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                     f'{m}, {n}, {k}}};')
        functors = (f'{spec["epi"]["run"]}, {spec["epi"]["ld"]}' if spec.get('epi') else 'p5c::NoEpi(), p5c::NoLd()')
        targs = f'<{plan_["bm"]},{plan_["bn"]},{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)}>'
        items.append(f'p9ws::bitem{targs}(g{index}, {functors}, ' + sched.replace('@K@', str(index)) + ')' if sched
                     else f'p9ws::item{targs}(g{index}, {functors}, {rot})')
    code = ('{\n' + '\n'.join(decls) + '\n'
            f'  p9ws::group<0,{nst}>(smem, 0, 1, 1,\n    ' + ',\n    '.join(items) + ');\n}')
    return (code, helper) if helper else code


P9_PF_AHEAD = 0                # P9-A: with P9_EPI_PREFETCH, epilogue L2 prefetch this many units ahead (0: next tile)
CONV_P8WS_UNITS = 32           # LATCH-STEP10-SPEED: conv programs (gpu-23 16: 0.2993 -> 0.2867 ms, gpu-24 32: 0.2854; transcendental
                               # epilogues bound the im2col products, smaller unrolled units schedule better)
P8WS_UNITS = 8                 # P9-A: epilogue units per tile (16 halves the unrolled epilogue code per unit)
P9_LONGK_BM64 = 0              # P9-A: products with K >= this and at most one 128-row tile per CTA use 64-row tiles (0: off)
P9_EARLY_LD = False            # P9-A: epilogue warps issue a unit's operand loads before waiting for its staged tile
P9_TMA_EVICT_LAST = False      # P9-A: product operand TMA loads carry an L2 evict_last cache policy
P9_OPERAND_PREFETCH = 0        # P9-A: producer L2-prefetches operand boxes this many steps beyond the stage ring (0: off)
P9_INTERLEAVE = False          # P9-A: interleaved tile order in multi-product p8ws groups (heavy epilogues earlier)
P9_HEAVY_WEIGHT = 70
P9_EPI_PREFETCH = False        # P9-A: epilogue warps L2-prefetch epilogue operand rows ahead of each tile (p8ws pf)
_LD_LOAD = re.compile(r'(\w+)\[e\](\[[01]\])? = \*reinterpret_cast<const uint4\*>\((v_\w+) \+ (.+?)\);')
# LATCH-STEP-06: the p5c register epilogues load through __ldg(...) (ops/fusion_p6e.py / p5c_epilogue forms).
_LD_LOAD_LDG = re.compile(r'(\w+)\[e\](\[[01]\])? = __ldg\(reinterpret_cast<const uint4\*>\((v_\w+) \+ (.+?)\)\);')
P5C_EPI_PREFETCH = False       # LATCH-STEP-06: p5c tiles L2-prefetch their epilogue's saved operand rows at tile start


def prefetch_functor(ld, bn=128, ns='p8ws'):
    """L2-prefetch functor for the row-contiguous operands of a generated epilogue `ld` functor, or None."""
    calls = []
    found = _LD_LOAD.findall(ld or '') + (_LD_LOAD_LDG.findall(ld or '') if ns == 'p5c' else [])
    for _, pair, ptr, idx in found:
        if pair == '[1]':
            continue
        m = re.fullmatch(r'\(i0 % (\d+)u\)', idx)
        if not (idx == 'i0' or (m and int(m.group(1)) % bn == 0)):
            continue
        call = f'{ns}::pf_l2({ptr} + {idx.replace("i0", "r0")}, {bn * (4 if pair else 2)}u);'
        if call not in calls:
            calls.append(call)
    return ('[&](unsigned r0) { ' + ' '.join(calls) + ' }') if calls else None


def group_code(specs, plans, rots, ident, lo=0, hi=1, den=1):
    """P8-A: one warp-specialized pipeline over several independent products (serial members of a phase)."""
    if any(spec.get('virtual_b') or spec.get('virtual_a') for spec in specs):
        raise EngineRefusal('tap-shifted operands are emitted only by single-product p5c code')
    # Every item shares the same physical operand ring. Short contractions may
    # plan fewer stages; use the largest declared ring, whose bytes the group
    # scheduler already reserves as max(member.shared_bytes).
    nst = max(plan_['nst'] for plan_ in plans)
    preludes, decls, items = [], [], []
    for index, (spec, plan_, rot) in enumerate(zip(specs, plans, rots)):
        if not plan_.get('ws'):
            raise EngineRefusal('group member without a warp-specialized plan')
        t = lambda flag: str(flag).lower()
        (ra, oa), (rb, ob) = spec['roots']
        m, n, k = spec['m'], spec['n'], spec['k']
        sam, sak, sbk, sbn, _, _ = spec['strides']
        pa, pb = (spec['pointers'] if spec.get('pointers') else (None, None))
        ta = _marker(ra, oa, spec['ak'], m, k, sam if spec['ak'] else sak, plan_['bm'], pa)
        tb = _marker(rb, ob, spec['bkc'], n, k, sbn if spec['bkc'] else sbk, plan_['bn'], pb)
        out_mode = _out_mode(spec)
        store = spec['out'] != 'nullptr'
        if spec.get('epi') and spec['epi']['prelude']:
            preludes.append(spec['epi']['prelude'])
        decls.append(f'  p5c::Args g{index}{{{ta}, {tb}, {ident(spec["a"])}, {ident(spec["b"])}, {spec["out"]}, {n}LL, '
                     f'{m}, {n}, {k}}};')
        functors = (f'{spec["epi"]["run"]}, {spec["epi"]["ld"]}' if spec.get('epi') else 'p5c::NoEpi(), p5c::NoLd()')
        from . import flat_long_attention as _fla
        kn = _fla.KN_EPI_TARGETS.get(spec['out']) if not spec.get('epi') and spec['out'] != 'nullptr' else None
        if kn and n == kn['ldc'] and store:
            # P9-A: key-norm table of prepare_unit from the stored BF16 K values (gv): 8 lanes x 8 units of one head row
            # summed in prepare_unit's order (fmaf over the chunk, xor 1/2/4), sqrt, the same (1 + 1/256) bound, and
            # the tile maximum as an atomic maximum of non-negative floats (order-independent): bitwise equal.
            functors = ('[&](int e, unsigned i0, const float (&gv)[8]) { float acc = 0.f;\n'
                        '#pragma unroll\n      for (int j = 0; j < 8; ++j) acc = fmaf(gv[j], gv[j], acc);\n'
                        '#pragma unroll\n      for (int o_ = 1; o_ < 8; o_ <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o_);\n'
                        '      float v_ = sqrtf(acc) * (1.0f + 1.0f / 256.0f);\n'
                        '      v_ = fmaxf(v_, __shfl_xor_sync(0xffffffffu, v_, 16));\n'
                        '      if ((threadIdx.x & 23) == 0) {\n'
                        f'        const unsigned r_ = i0 / {kn["ldc"]}u, h_ = (i0 % {kn["ldc"]}u) / 64u;\n'
                        f'        atomicMax(reinterpret_cast<int*>(v_p9kn + {kn["base"]} + ((r_ / {kn["t"]}u) * {kn["heads"]}u + h_) * {kn["nt"]}u'
                        f' + (r_ % {kn["t"]}u) / 64u), __float_as_int(v_)); }} }}, [&](int e, unsigned i0) {{}}')

        pf = prefetch_functor(spec['epi']['ld']) if P9_EPI_PREFETCH and spec.get('epi') else None
        heavy = spec.get('epi') and len(re.findall(r'\[e\]\[1\] = ', spec['epi']['ld'])) >= 2
        if plan_.get('chunk_stages'):
            # Release step 3 repair: chunked accumulation (no interleave weight or epilogue prefetch on this item).
            items.append(f'p8ws::item_k<{plan_["bm"]},{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)},'
                         f'{plan_["chunk_stages"]}>(g{index}, {functors}, {rot})')
        elif P9_INTERLEAVE and len(specs) > 1 and not pf:
            items.append(f'p8ws::item_w<{plan_["bm"]},{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)}>(g{index}, '
                         f'{functors}, {P9_HEAVY_WEIGHT if heavy else 100}, {rot})')
        elif pf:
            items.append(f'p8ws::item_pf<{plan_["bm"]},{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)}>(g{index}, '
                         f'{functors}, {pf}, {rot})')
        else:
            items.append(f'p8ws::item<{plan_["bm"]},{t(spec["ak"])},{t(spec["bkc"])},{out_mode},{t(store)}>(g{index}, '
                         f'{functors}, {rot})')
    return ('{\n' + ''.join(p + '\n' for p in preludes) + '\n'.join(decls) + '\n'
            f'  p8ws::group<{P8WS_MODE},{nst},{str(P8WS_HELP).lower()},'
            f'{str(bool(P9_INTERLEAVE and len(specs) > 1 and P8WS_MODE == 0 and not P8WS_HELP)).lower()}>(smem, {lo}, {hi}, {den},\n    '
            + ',\n    '.join(items) + ');\n}')


def member_record(spec, plan_, slice_):
    if spec.get('out_raw'):
        return {'engine': f'{spec["engine"]}{"_ws" if plan_.get("ws") else ""}_{plan_["bm"]}x{plan_["bn"]}_raw32',
                'output': spec['out'],
                'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': 1, 'matrix_plan': plan_,
                'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
                'arithmetic': ('FP32 product of exact BF16->FP32 operand widenings, read from their BF16 sources; '
                               'wgmma FP32 accumulation from +0 in ascending k16 steps; '
                               + (f'chains of {plan_["chunk_stages"]} 64-k stages summed in order with FP32 RN additions; '
                                  if plan_.get('chunk_stages') else '') +
                               'no split-K; unrounded FP32 result (tolerance-gated, not bitwise to the FP32 fmaf chain)')}
    return {'engine': f'{spec["engine"]}{"_pp" if plan_.get("pingpong") else "_ws" if plan_.get("ws") else ""}_{plan_["bm"]}x{plan_["bn"]}'
                      + ('_f32' if spec.get('out_f32') else ''),
            'output': spec['out'],
            'matrix_shape': [spec['m'], spec['n'], spec['k']], 'batch': 1, 'matrix_plan': plan_,
            'cta_slice': slice_, 'operand_layout': {'a_k_contiguous': spec['ak'], 'b_k_contiguous': spec['bkc']},
            'arithmetic': ('explicit BF16 operands/output; wgmma FP32 accumulation from +0 in ascending k16 steps; '
                           + (f'chains of {plan_["chunk_stages"]} 64-k stages summed in order with FP32 RN additions; '
                              if plan_.get('chunk_stages') else '') +
                           'no split-K; one cvt.rn.bf16x2 rounding (tolerance-gated, not bitwise to mma.sync)')}


P5C_TRAP_ASSERT = True         # see TRAP_ASSERT below
TRAP_ASSERT = '''// P-OPS5-C: generated index bounds checks use assert(); its __assertfail system call makes ptxas serialize every
// wgmma of the function (C7510, "pipeline crossing function boundary at a function call"). In programs with
// warpgroup-MMA products a failed check aborts the launch through __trap() instead (same abort-on-violation, no
// message).
#ifdef assert
#undef assert
#endif
#define assert(c) do { if (!(c)) __trap(); } while (0)
'''


REGION_MAP_HELPERS = r'''
// P7-C: loop-region operand maps. The table (args[inputs]) holds `total` 128-byte maps followed by their u64 global
// addresses; a region instance picks the map whose address equals its operand pointer (trap if none).
__shared__ const char* p5c_maps_sh;
__device__ __forceinline__ const void* p5c_pick_map(int total, int first, int count, const void* ptr) {
  const unsigned long long* addr = reinterpret_cast<const unsigned long long*>(p5c_maps_sh + 128ll * total);
  int k = -1;
  for (int c = 0; c < count; ++c) k = addr[first + c] == (unsigned long long)ptr ? first + c : k;
  if (k < 0) __trap();
  return p5c_maps_sh + 128ll * k;
}
'''

# P8-B: the map addresses are copied once per launch into shared memory (u64 per map, read by every product start:
# the global-table lookup was a dependent L2 round trip before each product's first TMA issue). Same selection.
# Measured P8-B gpu-11: the 3.7 KB static shared table moved the program to a larger shared carve-out (less L1)
# and cost 0.27-0.29 ms per update on the primary (9.08 vs 9.35 ms; 8.74 vs 8.97 ms): rejected, default off.
MAP_ADDR_SHARED_MAX = 0     # maps (0: keep the global lookup)


def region_map_helpers(total):
    if not 0 < total <= MAP_ADDR_SHARED_MAX:
        return REGION_MAP_HELPERS
    return f'''
// P7-C loop-region operand maps; P8-B: addresses resident in shared memory ({total} maps).
__shared__ const char* p5c_maps_sh;
__shared__ unsigned long long p5c_addr_sh[{total}];
__device__ __forceinline__ void p5c_load_addresses() {{
  const unsigned long long* addr = reinterpret_cast<const unsigned long long*>(p5c_maps_sh + 128ll * {total});
  for (int i = (int)threadIdx.x; i < {total}; i += (int)blockDim.x) p5c_addr_sh[i] = addr[i];
}}
__device__ __forceinline__ const void* p5c_pick_map(int total, int first, int count, const void* ptr) {{
  int k = -1;
#pragma unroll 8
  for (int c = 0; c < count; ++c) k = p5c_addr_sh[first + c] == (unsigned long long)ptr ? first + c : k;
  if (k < 0) __trap();
  return p5c_maps_sh + 128ll * k;
}}
'''


def _include_exact_division(source):
    if 'latch_xdiv(' in source and 'static __device__ __forceinline__ float latch_xdiv' not in source:
        # Exact-division element/row epilogues can also occur in a standalone
        # region with no TMA product. Include their helper independently of map
        # finalization, retaining the same arithmetic and chunk fallback.
        header = (Path(__file__).parent / 'kernels' / 'latch_xdiv.cuh').read_text().replace('#pragma once\n', '')
        source = header + source
    return source


def finalize(program):
    """Include element helpers and resolve any tensor-map placeholders."""
    import hashlib
    import json
    from .lower import GeneratedProgram
    if TMAP_MARK + '(' not in program.source:
        source = _include_exact_division(program.source)
        if source == program.source:
            return program
        manifest = dict(program.manifest)
        manifest['source_hash'] = hashlib.sha256((source + json.dumps(manifest['header_hashes'], sort_keys=True)).encode()).hexdigest()
        return GeneratedProgram(source, manifest, program.kernel_body, program.helpers, program.functions)
    manifest = dict(program.manifest)
    inputs = manifest['inputs']
    maps, index = [], {}

    def replace(match):
        key = match.group(1)
        if key not in index:
            fields = key.split(',')
            root = fields[0]
            if root in inputs:
                where = {'storage': 'input', 'index': inputs.index(root)}
            elif root in manifest['offsets']:
                where = {'storage': 'arena', 'word_offset': int(manifest['offsets'][root])}
            else:
                raise EngineRefusal(f'tensor-map operand root {root} has no storage')
            index[key] = len(maps)
            if len(fields) == 8 and fields[6] == 'i2c':
                # LATCH-STEP10-SPEED: im2col map of a pixel-major (N, H, W, C) image (im2col_marker).
                _, offset, c, w, h, n, _, ppc = fields
                c, w, h, n = int(c), int(w), int(h), int(n)
                maps.append({'root': root, **where, 'element_offset': int(offset), 'kind': 'im2col',
                             'dims': [c, w, h, n], 'stride_bytes': [2 * c, 2 * c * w, 2 * c * w * h],
                             'channels_per_pixel': 64, 'pixels_per_column': int(ppc), 'lower': [-1, -1],
                             'upper': [-1, -1], 'dtype': 'bfloat16', 'swizzle': 128, 'oob_fill': 'zero'})
            else:
                _, offset, d0, d1, stride, b0, b1 = fields
                maps.append({'root': root, **where, 'element_offset': int(offset), 'dims': [int(d0), int(d1)],
                             'stride_bytes': int(stride), 'box': [int(b0), int(b1)], 'dtype': 'bfloat16',
                             'swizzle': 128, 'oob_fill': 'zero'})
        return f'(reinterpret_cast<const char*>(args[{len(inputs)}]) + {128 * index[key]})'

    source = re.sub(TMAP_MARK + r'\(([^()]*)\)', replace, program.source)
    if P5C_DECOUPLED:
        source = '#define P5C_DECOUPLED 1  // P6-D: per-warpgroup stage release, second arrival refills\n' + source
    if P8WS and P8WS_TRACE and 'p8ws::' in source:
        source = '#define P8WS_TRACE 1  // P8-A diagnostic trace\n' + source
    from . import p9_rowblock
    if p9_rowblock.TRACE and 'P9T(' in source:
        source = '#define P9_TRACE 1  // P9-A staged-phase stamps\n' + source
    if P9_EPI_PREFETCH and 'p8ws::' in source:
        source = '#define P9_EPI_PREFETCH 1  // P9-A epilogue operand L2 prefetch\n' + source
    if P9_PF_AHEAD and 'p8ws::' in source:
        source = f'#define P9_PF_AHEAD {int(P9_PF_AHEAD)}  // P9-A epilogue prefetch distance (units)\n' + source
    source = _include_exact_division(source)
    from .lower import lane_rules_active
    units = CONV_P8WS_UNITS if (CONV_P8WS_UNITS and lane_rules_active()) else P8WS_UNITS
    if units != 8 and 'p8ws::' in source:
        source = f'#define P8WS_NU {int(units)}  // P9-A epilogue units per tile\n' + source
    if P9_EARLY_LD and 'p8ws::' in source:
        source = '#define P9_EARLY_LD 1  // P9-A early epilogue operand loads\n' + source
    if P9_TMA_EVICT_LAST and 'p8ws::' in source:
        source = '#define P9_TMA_EVICT_LAST 1  // P9-A operand L2 evict_last policy\n' + source
    if P8WS and P9_OPERAND_PREFETCH and 'p8ws::' in source:
        source = f'#define P9_OPF {int(P9_OPERAND_PREFETCH)}  // P9-A operand L2 prefetch distance\n' + source
    if P8WS and P8WS_STAGING != 2 and 'p8ws::' in source:
        source = f'#define P8WS_NSB {int(P8WS_STAGING)}  // P8-A staging buffers\n' + source
    if P5C_SYNC2:
        source = '#define P5C_SYNC2 1  // P6-D: one CTA barrier per two 64-k steps in p5c bodies\n' + source
    if '/*P5C_CLUSTER2*/' in source:
        source = cluster_launcher(source, int(manifest['shared_bytes']))
        manifest['cluster_dims'] = [2, 1, 1]
    if P5C_TRAP_ASSERT:
        entry = 'extern "C" __global__'
        if source.count(entry) != 1:
            raise EngineRefusal('expected exactly one kernel entry')
        source = source.replace(entry, TRAP_ASSERT + entry, 1)
        manifest['index_checks'] = 'trap (assert overridden for wgmma programs)'
    manifest['tensor_maps'] = maps
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(manifest['header_hashes'], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest, program.kernel_body, program.helpers, program.functions)


def cluster_launcher(source, shared_bytes):
    """Replace the cooperative launcher by a cooperative 2-CTA-cluster launch (cudaLaunchKernelEx)."""
    a = source.index('extern "C" int latch_query(')
    b = source.index('extern "C" const char* latch_error(')
    text = f'''extern "C" int latch_query(int* max_blocks, int* multiprocessors) {{
  int device; cudaError_t e = cudaGetDevice(&device); if (e != cudaSuccess) return e;
  cudaDeviceProp prop; e = cudaGetDeviceProperties(&prop, device); if (e != cudaSuccess) return e;
  if (!prop.cooperativeLaunch) return cudaErrorNotSupported;
  e = cudaFuncSetAttribute(latch_ops, cudaFuncAttributeMaxDynamicSharedMemorySize, {shared_bytes});
  if (e != cudaSuccess) return e;
  cudaLaunchConfig_t cfg = {{}}; cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(2 * prop.multiProcessorCount); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = {shared_bytes};
  cfg.attrs = at; cfg.numAttrs = 1;
  int clusters = 0; e = cudaOccupancyMaxActiveClusters(&clusters, (void*)latch_ops, &cfg);
  if (e != cudaSuccess) return e;
  *multiprocessors = prop.multiProcessorCount; *max_blocks = 2 * clusters;
  return *max_blocks > 0 ? cudaSuccess : cudaErrorLaunchOutOfResources;
}}
extern "C" int latch_launch(float** args, float* arena, unsigned long long* ticks, int blocks, void* stream) {{
  int max_blocks, sms; int e = latch_query(&max_blocks, &sms); if (e) return e;
  blocks &= ~1;                                                       // whole 2-CTA clusters
  if (blocks <= 0 || blocks > max_blocks) return cudaErrorInvalidConfiguration;
  void* kernel_args[] = {{&args, &arena, &ticks}};
  cudaLaunchConfig_t cfg = {{}}; cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeCooperative; at[1].val.cooperative = 1;
  cfg.gridDim = dim3(blocks); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = {shared_bytes};
  cfg.stream = (cudaStream_t)stream; cfg.attrs = at; cfg.numAttrs = 2;
  return cudaLaunchKernelExC(&cfg, (const void*)latch_ops, kernel_args);
}}
'''
    return source[:a] + text + source[b:]


def normalize_tensor_maps(manifest):
    """Normalize frozen flat root/offset maps and loop storage maps to one address ABI."""
    result = []
    inputs = list(manifest.get('inputs', ()))
    layouts = manifest.get('layouts', {})
    offsets = manifest.get('offsets', {})
    for original in manifest.get('tensor_maps') or ():
        spec = dict(original)
        if 'storage' not in spec:
            name = spec['root']
            layout = layouts.get(name, {})
            root = layout.get('root', name)
            spec['element_offset'] = int(spec.get('offset', 0)) + int(layout.get('offset', 0))
            if root in inputs:
                spec.update(storage='input', index=inputs.index(root))
            elif root in offsets:
                spec.update(storage='arena', word_offset=int(offsets[root]))
            else:
                raise EngineRefusal(f'tensor-map root {name} has no bound storage')
        if spec['storage'] not in ('input', 'arena'):
            raise EngineRefusal('invalid tensor-map storage kind')
        spec.setdefault('element_offset', 0)
        result.append(spec)
    return result


def encode_maps(manifest, args, arena, device):
    """Host-side cuTensorMapEncodeTiled for every map; returns a device uint8 table (128 B per map)."""
    import ctypes
    import torch
    specs = normalize_tensor_maps(manifest)
    if not specs:
        return None
    encode = ctypes.CDLL('libcuda.so.1').cuTensorMapEncodeTiled
    encode.restype = ctypes.c_int
    raw = ctypes.create_string_buffer(128 * len(specs) + 64)
    off0 = (-ctypes.addressof(raw)) % 64
    def tensor_base(spec):
        if spec['storage'] == 'input':
            tensor = args[spec['index']]
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f'tensor map: BF16 input storage required for {spec["root"]}')
            return tensor.data_ptr()
        return arena.data_ptr() + 4 * spec['word_offset']

    encode_im2col = None
    for i, spec in enumerate(specs):
        base = tensor_base(spec)
        address = base + 2 * spec['element_offset']
        if spec.get('kind') == 'im2col':
            # LATCH-STEP10-SPEED: cuTensorMapEncodeIm2col(map, BF16, rank 4, address, dims {C, W, H, N}, byte
            # strides of W/H/N, lower/upper pixel-box corners {w, h}, channels per pixel, pixels per column,
            # element strides, no interleave, 128-byte swizzle, 256-byte L2 promotion, zero OOB fill).
            if encode_im2col is None:
                encode_im2col = ctypes.CDLL('libcuda.so.1').cuTensorMapEncodeIm2col
                encode_im2col.restype = ctypes.c_int
            if address % 16 or any(s % 16 for s in spec['stride_bytes']):
                raise ValueError(f'im2col tensor map {i}: 16-byte aligned address and strides required')
            rc = encode_im2col(ctypes.c_void_p(ctypes.addressof(raw) + off0 + 128 * i), ctypes.c_int(9),
                               ctypes.c_uint32(4), ctypes.c_void_p(address), (ctypes.c_uint64 * 4)(*spec['dims']),
                               (ctypes.c_uint64 * 3)(*spec['stride_bytes']), (ctypes.c_int * 2)(*spec['lower']),
                               (ctypes.c_int * 2)(*spec['upper']), ctypes.c_uint32(spec['channels_per_pixel']),
                               ctypes.c_uint32(spec['pixels_per_column']), (ctypes.c_uint32 * 4)(1, 1, 1, 1),
                               ctypes.c_int(0), ctypes.c_int(3), ctypes.c_int(3), ctypes.c_int(0))
            if rc:
                raise RuntimeError(f'cuTensorMapEncodeIm2col({spec}) -> {rc}')
            continue
        if address % 16 or spec['stride_bytes'] % 16:
            raise ValueError(f'tensor map {i}: 16-byte aligned address and row stride required')
        rc = encode(ctypes.c_void_p(ctypes.addressof(raw) + off0 + 128 * i), ctypes.c_int(9), ctypes.c_uint32(2),
                    ctypes.c_void_p(address), (ctypes.c_uint64 * 2)(*spec['dims']),
                    (ctypes.c_uint64 * 1)(spec['stride_bytes']), (ctypes.c_uint32 * 2)(*spec['box']),
                    (ctypes.c_uint32 * 2)(1, 1), ctypes.c_int(0), ctypes.c_int({128: 3, 0: 0}[int(spec.get('swizzle', 128))]), ctypes.c_int(2), ctypes.c_int(0))
        if rc:
            raise RuntimeError(f'cuTensorMapEncodeTiled({spec}) -> {rc}')
    payload = bytearray(raw.raw[off0:off0 + 128 * len(specs)])
    if manifest.get('tensor_map_addresses'):
        # P7-C: u64 global address of every map after the maps (loop-region map selection by bound pointer).
        import struct
        for spec in specs:
            base = tensor_base(spec)
            payload += struct.pack('<Q', base + 2 * spec['element_offset'])
    host = torch.frombuffer(payload, dtype=torch.uint8)
    table = host.to(device)
    if table.data_ptr() % 64:
        raise RuntimeError('tensor map table must be 64-byte aligned')
    return table


class SharedSites:
    """Route epilogue-free p5c product calls to one entry-resident site per template configuration.

    Each call `p5c::phase<T>(g, smem, lo, hi, den, rot);` becomes: copy the arguments into kernel-scope variables,
    record a return index, `goto` the configuration's site (emitted once after the kernel's final statement) and
    come back through a uniform switch. Same function (no call: ptxas keeps wgmma pipelined), same arithmetic and
    schedule; only the code layout changes.
    """
    CALL = re.compile(r'\{\n  p5c::Args g\{(?P<args>[^;]*)\};\n  p5c::phase<(?P<tpl>[^>]*)>\(g, smem, (?P<lo>[^,]+), '
                      r'(?P<hi>[^,]+), (?P<den>[^,]+), (?P<rot>[^)]+)\);\n\}')

    def __init__(self):
        self.sites, self.returns = {}, 0

    def route(self, code):
        def replace(match):
            site = self.sites.setdefault(match.group('tpl'), len(self.sites))
            r = self.returns
            self.returns += 1
            return ('{\n'
                    f'  p5c_sg = p5c::Args{{{match.group("args")}}};\n'
                    f'  p5c_sl = {match.group("lo")}; p5c_sh = {match.group("hi")}; p5c_sd = {match.group("den")}; '
                    f'p5c_sr = {match.group("rot")}; p5c_ret = {r};\n'
                    f'}}\n  goto P5C_SITE_{site};\n  P5C_RET_{r}: ;\n')
        return self.CALL.sub(replace, code)

    def declarations(self):
        if not self.sites:
            return ''
        return '\n  p5c::Args p5c_sg{}; int p5c_sl = 0, p5c_sh = 1, p5c_sd = 1, p5c_sr = 0, p5c_ret = 0;\n'

    def definitions(self):
        if not self.sites:
            return ''
        cases = ' '.join(f'case {r}: goto P5C_RET_{r};' for r in range(self.returns))
        out = ['  return;']
        for tpl, site in self.sites.items():
            out.append(f'P5C_SITE_{site}:\n  p5c::phase<{tpl}>(p5c_sg, smem, p5c_sl, p5c_sh, p5c_sd, p5c_sr);\n'
                       f'  switch (p5c_ret) {{ {cases} default: return; }}')
        return '\n'.join(out) + '\n'
