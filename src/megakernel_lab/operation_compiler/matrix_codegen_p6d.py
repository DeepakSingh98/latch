# Integrated from frozen donor 81a52fc02783e0c537dcf1a909083916f2deaa66; private scheduling/resource protocol.
"""One matrix selection interface for flat and staged operation lowering.

P-OPS-C exact BF16/FP32 bodies take precedence on their admitted layouts;
P-OPS-D ordered FP32 split-K covers underfilled pinned matrices. The existing
Stage4 body remains the fallback, including BF16 tails. No value/model names
select an implementation. Recorded reasons describe every fallback.
"""
from . import matrix_rules_p6d as rules


def select(a, b, out, la, lb, shape, dtype, *, arch='sm_90', engine_ok=False):
    if arch != 'sm_90':
        return None, 'matrix cost model qualified only for sm_90; existing body retained'
    try:
        if dtype == 'bfloat16':
            try:
                spec = rules.bf16_spec(a, b, out, la, lb, shape, engine_ok=engine_ok)
            except rules.MatrixRuleError as aligned_error:
                try:
                    spec = rules.bf16_tail_spec(a, b, out, la, lb, shape)
                except rules.MatrixRuleError:
                    raise aligned_error
            return spec, (f'aligned BF16 warpgroup-MMA engine {spec["engine"]}' if spec.get('engine')
                          else 'aligned BF16 exact-order small tile')
        if dtype == 'float32' and rules.fp32_small_applies(la, lb, shape):
            return rules.fp32_spec(a, b, out, la, lb, shape), 'underfilled batched FP32 exact-order small tile'
    except rules.MatrixRuleError as error:
        return None, str(error) + '; existing tail/layout lowering retained'
    return None, 'existing FP32 matrix body; small-tile eligibility not met'


def split_k(a, b, name, la, lb, *, ident, enabled=True, grid=128):
    """Return partial/reduction phases, or None. Adapted from P-OPS-D lower.py.

    Ordered slice reduction changes FP32 association; it never splits BF16 MMA.
    Caller provides the already validated FP32 matrix layouts and a fresh name.
    """
    if not enabled:
        return None
    m, k = la.shape[-2:]; n = lb.shape[-1]
    batch = la.shape[0] if len(la.shape) == 3 else 1
    sam, sak = la.strides[-2:]; sbk, sbn = lb.strides[-2:]
    bsa, bsb = (la.strides[0], lb.strides[0]) if len(la.shape) == 3 else (0, 0)
    aligned = ((sak == 1 or sam == 1) and (sbk == 1 or sbn == 1)
               and all(s % 4 == 0 for s in (sam if sak == 1 else sak,
                       sbn if sbk == 1 else sbk, bsa, bsb, la.offset, lb.offset)))
    if m % 64 or n % 64 or k % 16 or not aligned:
        return None
    tiles = batch * (m // 64) * (n // 64)
    count = next((s for s in (8, 4, 2) if tiles * s <= grid
                  and k % (16 * s) == 0 and k // s >= 128), 1)
    if count == 1:
        return None
    part = name + '__splitk'
    out = ident(name)
    code = f'''{{
  s101h::GemmArgs<1> g{{}};
  g.p[0] = {{{ident(a)}, {sam}LL, {sak}LL, {bsa}LL, 0, {ident(b)}, {sbk}LL, {sbn}LL, {bsb}LL, 0, {k}}};
  g.M = {m}; g.N = {n}; g.nb = {batch}; g.nb2 = 1; g.ksplit = {count}; g.part = {ident(part)}; g.cnt = nullptr;
  s101h::gemm_phase<64,64,{str(sak == 1).lower()},{str(sbk == 1).lower()},1>(g, s101h::EpiStore{{{out}, {n}LL, {m*n}LL}}, smem);
}}'''
    reduce = f's101h::splitk_phase({ident(part)}, {count}, {batch}, {m}, {n}, s101h::EpiStore{{{out}, {n}LL, {m*n}LL}});'
    return {'part': part, 'elements': count * batch * m * n, 'code': code, 'reduce': reduce,
            'metadata': {'tiles': tiles * count, 'engine': 'pinned_simt_64_splitk',
                         'ksplit': count, 'batch': batch, 'k': k,
                         'arithmetic': 'FP32 ascending-K slices followed by ascending slice sum; no TF32'}}


B_DONOR_COMMIT = '286c2371624be08c6c7106c47df546e1e550f861'
B_HEADER = 'fp32_matrix.cuh'
B_HELPERS = """#ifndef LATCH_OPS_POPSB_IMPORTED
#define LATCH_OPS_POPSB_IMPORTED
#define POPS_INLINE
#define POPS_RELACQ
#include "fp32_matrix.cuh"
#endif
"""


def fp32_async(a, b, name, la, lb, shape, *, ident, arch='sm_90', grid=128,
               split=True, occupied_names=(), accumulate=False):
    """P-OPS-B engine adapter shared by A flat and E staged schedules.

    Admission assumes caller-validated FP32 operand/result dtypes. C's admitted
    exact small matrix bodies take precedence. One private scratch root contains
    padded counters followed by partials. Its explicit producer phase initializes
    counters only; the product writes all partial values before its fixed-order
    last-arriver reduction. No cross-product counters or implicit grid barrier.

    The emitted device helper uses the physical cooperative grid. Callers must
    not execute it under a virtual CTA subset; whole-grid rotation is supported
    by replacing /*B_VBASE*/ with an integer expression (zero by default).
    """
    if arch != 'sm_90':
        return None, 'P-OPS-B cost/admission policy is H100 sm_90 only'
    if accumulate:
        return None, 'direct accumulated output retains the existing exact accumulation epilogue'
    if len(la.shape) not in (2,3) or len(lb.shape) != len(la.shape):
        return None, 'P-OPS-B requires matching rank-2/rank-3 matrix operands'
    m,k = la.shape[-2:]; kb,n = lb.shape[-2:]
    batch = la.shape[0] if len(la.shape)==3 else 1
    expected = (batch,m,n) if len(la.shape)==3 else (m,n)
    if k != kb or tuple(shape)!=expected or (len(lb.shape)==3 and lb.shape[0]!=batch):
        return None, 'P-OPS-B matrix shape/batch mismatch'
    if type(grid) is not int or grid < 1:
        return None, 'P-OPS-B requires a positive declared grid estimate'
    from .pops_b_policy import MatrixPolicy, make_spec, layout_struct
    policy = MatrixPolicy(grid=grid,min_blocks_per_sm=1,pack_phases=False,
                          elide_copies=False,joint=False,l2_prefetch=False,
                          splits=(1,2,4,8) if split else (1,))
    spec = make_spec(policy,a,b,name,la,lb,shape,m,n,k,batch)
    if spec is None:
        return None, 'P-OPS-B layout/tile/alignment contract refused; existing body retained'
    scratch, counter_init, operands = [], None, list(dict.fromkeys((spec['a_root'],spec['b_root'])))
    counters = ((spec['tiles']+63)//64*64) if spec['ksplit']>1 else 0
    partials = spec['tiles']*spec['ksplit']*spec['bm']*spec['bn'] if counters else 0
    scratch_name = name+'__popsb_scratch'
    if counters and scratch_name in set(occupied_names):
        return None, 'P-OPS-B scratch name collision; existing body retained'
    if counters:
        scratch = [{'name':scratch_name,'shape':(counters+partials,), 'dtype':'float32',
                    'role':'counter_and_partials','counter_words':counters,'partial_words':partials,
                    'initialization':'counter words zero; partial words defined by matrix slice stores'}]
        pointer = ident(scratch_name)
        counter_init = {'code':f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {counters}LL; i += (long long)gridDim.x * blockDim.x) reinterpret_cast<unsigned*>({pointer})[i] = 0u;',
                        'writes':[scratch_name], 'counter_words':counters,'partial_words':partials}
        part, cnt = f'{pointer} + {counters}LL', f'reinterpret_cast<unsigned*>({pointer})'
    else:
        part, cnt = 'nullptr','nullptr'
    layout_name, layout_definition = layout_struct(spec)
    code = (f"pops::gemm_phase<{spec['bm']},{spec['bn']},{str(spec['ak']).lower()},"
            f"{str(spec['bkc']).lower()},{spec['stages']},{layout_name},{spec['kb']}>("
            f"{ident(spec['a_root'])} + {spec['a_offset']}LL, {ident(spec['b_root'])} + {spec['b_offset']}LL, "
            f"{ident(name)}, {part}, {cnt}, /*B_VBASE*/, smem);")
    metadata = {'engine':'popsb_cpasync','tiles':spec['tasks'],'batch':batch,'k':k,
        'ksplit':spec['ksplit'],'matrix_shape':[batch,m,n,k],
        'shared_bytes':spec['shared_bytes'],'matrix_plan':{key:spec[key] for key in
            ('bm','bn','kb','stages','ksplit','tiles','tasks','steps','waves','estimate_us')},
        'requires_full_grid':True,'selection_grid':grid,
        'internal_mutations':[scratch_name] if counters else [],
        'arithmetic':'strict FP32 ascending-K fmaf within slices; fixed ascending split sum; no TF32/tensor cores/data atomics',
        'copy_elision':'deferred: common attention/layout rewriting owns views',
        'phase_schedule':'common A/E effects scheduler; no donor packing pass',
        'donor_commit':B_DONOR_COMMIT,
        'model_context':'donor constants fitted at 2 CTAs/SM; integrated resource policy recorded separately'}
    return {'code':code,'helpers':B_HELPERS+'\n'+layout_definition,
            'headers':[B_HEADER],'metadata':metadata,'scratch':scratch,
            'counter_init':counter_init,'operand_roots':operands}, 'admitted pinned P-OPS-B cp.async FP32 body'
