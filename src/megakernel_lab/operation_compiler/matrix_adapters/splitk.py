"""Admission and emission of the explicit narrow six-part arithmetic contract."""
from .bodies import _aligned
from .requirements import Precision, Requirements, Synchronization, check
from ..matrix_splitk import BODY, CONTRACT, applies, record

HEADERS = ('tgemm.cuh', 'bgemm.cuh', 'popsc_mm.cuh', 'matrix_splitk.cuh')
SHARED_BYTES = 3 * (32 * 72 + 32 * 40) * 2 + 64 * 32 * 4
# LATCH-MOE4-160: the six intervals of one 16x32 tile run on six warps of a CTA
# (same fragments, instruction and ascending interval sum); False keeps one CTA
# per 64-row tile computing the intervals one after another.
PARALLEL_SPLITS = True
WARPS_SHARED_BYTES = 6 * 4 * (32 * 24 + 32 * 40) * 2 + 6 * 16 * 32 * 4


def emit(provider, p, *, a, b, name, out, ident, out_pitch):
    if not applies(p) or p.m % 64 or not _aligned(p, 8):
        raise ValueError(f'{name}: split6 requires an aligned transposed-left N32 product with M divisible by 64')
    arithmetic = record(p)  # verifies all six intervals are nonempty and aligned
    if out_pitch is not None and out_pitch != p.n:
        raise ValueError(f'{name}: split6 requires an unpadded result pitch')
    precision = Precision('bfloat16', 'float32', 'bfloat16',
        'one BF16 RN after ascending FP32 partial reduction',
        'six independent +0 FP32 accumulators; ascending k16 within aligned64 intervals; ascending FP32 interval sum',
        'mma.sync.m16n8k16', CONTRACT, split_k=6)
    parallel = PARALLEL_SPLITS and p.m % 16 == 0
    shared = WARPS_SHARED_BYTES if parallel else SHARED_BYTES
    req = Requirements('sm_90', shared, headers=HEADERS,
        sync=Synchronization(requires_full_grid=True), precision=precision)
    admitted, reasons = check(req, provider.resources)
    if not admitted:
        if not parallel:
            raise ValueError(f'{name}: split6 body refused: {reasons}')
        parallel, shared = False, SHARED_BYTES
        req = Requirements('sm_90', shared, headers=HEADERS,
            sync=Synchronization(requires_full_grid=True), precision=precision)
        admitted, reasons = check(req, provider.resources)
        if not admitted:
            raise ValueError(f'{name}: split6 body refused: {reasons}')
    code = f'''{{
  popsc::BArgs g{{}};
  g.A = reinterpret_cast<const s101h::bf16*>({ident(p.a.root)}) + {p.a.offset}LL;
  g.B = reinterpret_cast<const s101h::bf16*>({ident(p.b.root)}) + {p.b.offset}LL;
  g.C = reinterpret_cast<s101h::bf16*>({out});
  g.sam={p.a.row_stride}LL; g.sak={p.a.col_stride}LL; g.bsa=0;
  g.sbk={p.b.row_stride}LL; g.sbn={p.b.col_stride}LL; g.bsb=0;
  g.ldc={p.n}LL; g.bsc=0; g.M={p.m}; g.N={p.n}; g.K={p.k}; g.nb=1;
  latch_matrix_splitk::{'phase_warps' if parallel else 'phase'}(g,smem);
}}'''
    if parallel:
        tiles = p.m // 16
        plan = dict(bm=16, bn=32, interval_warps=6, ks=32, nst=4, tiles=tiles, shared_bytes=shared)
    else:
        tiles = p.m // 64
        plan = dict(bm=64, bn=32, wm=4, wn=2, ks=32, nst=3, tiles=tiles, shared_bytes=shared)
    decision = dict(output=name, provider='latch_matrix', policy=provider.policy,
        selected=BODY, rule='explicit narrow BF16 six-part mma.sync k16 arithmetic',
        admitted=[BODY], refused={}, arithmetic_contract=arithmetic, plan=plan)
    provider.decisions.append(decision)
    provider.emitted.append(name)
    extra = dict(tiles=tiles, k=p.k, shared_bytes=shared, threads=256,
        engine='latch_matrix:'+BODY, matrix_shape=[p.m,p.n,p.k], batch=1,
        matrix_plan=plan, arithmetic=precision.order+'; '+precision.rounding,
        matrix_splitk_contract=arithmetic,
        latch_matrix=dict(body=BODY, bitwise_class=CONTRACT, arch='sm_90', entry_resident=False, host_setup=[]),
        matrix_spec=dict(engine='latch_matrix:'+BODY, a=a, b=b, out=out, m=p.m,n=p.n,k=p.k,batch=1,
                         roots=[(p.a.root,int(p.a.offset)),(p.b.root,int(p.b.offset))]))
    contract = dict(work_items=tiles, execution_grid_scope='physical_cooperative_grid',
        requires_full_grid=True, internal_mutations=[], counter_effects=[], private_scratch=[])
    return dict(code=code, headers=list(HEADERS), helpers=[], extra=extra, decision=decision,
                contract=contract, out_pitch=out_pitch)
