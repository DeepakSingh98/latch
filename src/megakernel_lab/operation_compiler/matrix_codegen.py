# P-OPS2-D adaptation from 04ca670a17bb51ca6123a5c0d9a1002a27fbc83c.
"""One matrix selection interface for flat and staged operation lowering.

P-OPS-C exact BF16/FP32 bodies take precedence on their admitted layouts;
P-OPS-D ordered FP32 split-K covers underfilled pinned matrices. The existing
Stage4 body remains the fallback, including BF16 tails. No value/model names
select an implementation. Recorded reasons describe every fallback.
"""
from . import matrix_rules as rules


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
                          else 'aligned BF16 exact-order small tile' + (
                              '; warpgroup shape/layout/resource admission refused' if engine_ok else ''))
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
               split=True, occupied_names=(), accumulate=False, engine=None, counters_at=None, dest=None,
               preserve_plan=False):
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
    if isinstance(engine, str):
        if engine not in ('auto', 'popsb'):
            return None, 'unknown matrix engine: ' + engine
        engine = {'selector': 'auto'} if engine == 'auto' else None
    # Integration: A handles ordinary FP32 destinations inside the same D
    # staged schedule. D's exact accumulating and factored store epilogues keep
    # their qualified engine until the selected A bodies admit those contracts.
    requested = (engine or {}).get('selector')
    selection_fallback = None
    if preserve_plan and requested == 'auto':
        # A placed gradient must keep the ordinary donor's chosen body and K
        # slices. Choosing B's deferred-grid plan for an accumulating result
        # changes the sum tree even when its final += is separately rounded.
        from .matrix_select import SELECTION_GRID
        if dest is not None:
            from .lower import LoweringError
            raise LoweringError('preserved matrix plans require the ordinary destination layout')
        selected, reason = fp32_selected(a, b, name, la, lb, shape, ident=ident,
            arch=arch, grid=SELECTION_GRID, split=split, occupied_names=occupied_names,
            counters_at=counters_at, inline=(engine or {}).get('inline', True),
            bodies=(engine or {}).get('bodies', ('simt', 'popsb')), accumulate=accumulate)
        if selected is not None:
            selected['metadata'].update(preserved_store_plan=True,
                accumulation_epilogue='one FP32 destination addition after the completed product'
                    if accumulate else 'ordinary store')
        # If A refuses, retain the caller's ordinary pinned/generic fallback.
        # Do not silently replace it with B's independently selected split.
        return selected, reason
    if requested == 'auto' and not accumulate and dest is None:
        selected, selection_fallback = fp32_selected(a, b, name, la, lb, shape, ident=ident,
            arch=arch, grid=grid, split=split, occupied_names=occupied_names,
            counters_at=counters_at, inline=(engine or {}).get('inline', True),
            bodies=(engine or {}).get('bodies', ('simt', 'popsb')))
        if selected is not None:
            return selected, selection_fallback
    elif requested == 'auto':
        selection_fallback = ('A body does not admit direct accumulated outputs' if accumulate else
                              'A body does not admit factored write-through destinations')
    elif requested is not None:
        return None, 'unknown matrix selector: ' + str(requested)
    accumulating_epilogue = bool((engine or {}).get('accumulate'))
    if accumulate and not accumulating_epilogue:
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
    # P-OPS2-D: optional engine-selection overrides (pipeline depth, stage depth,
    # shared-memory limit, split set, tile set, resident-grid estimate, cost
    # constants). They select among B's admitted configurations; arithmetic per
    # slice and the fixed slice-order sum are unchanged.
    overrides = {key: (tuple(value) if isinstance(value, list) else value)
                 for key, value in (engine or {}).items() if key not in ('accumulate', 'inline', 'selector', 'bodies')}
    unknown = set(overrides) - {'stages', 'stage_depths', 'smem_limit', 'splits', 'tiles', 'grid', 'deferred_grid',
                                'task_us', 'step_us', 'split_store_us', 'split_reduce_us', 'min_slice_steps'}
    if unknown:
        return None, 'unknown P-OPS-B engine overrides: ' + ', '.join(sorted(unknown))
    if 'tiles' in overrides:
        overrides['tiles'] = tuple(tuple(tile) for tile in overrides['tiles'])
    grid = overrides.pop('grid', grid)
    deferred_grid = overrides.pop('deferred_grid', None)
    if accumulate and deferred_grid and not preserve_plan:
        # Accumulate-only products may be slack-placed beside small non-matrix
        # work; select among B's configurations for the CTAs left over.
        grid = int(deferred_grid)
    if not split:
        overrides['splits'] = (1,)
    policy = MatrixPolicy(grid=grid,min_blocks_per_sm=1,pack_phases=False,
                          elide_copies=False,joint=False,l2_prefetch=False,
                          **{'splits': (1,2,4,8), **overrides})
    spec = make_spec(policy,a,b,name,la,lb,shape,m,n,k,batch)
    if spec is None:
        return None, 'P-OPS-B layout/tile/alignment contract refused; existing body retained'
    if dest is not None:
        # P-OPS2-D write-through: store the result directly in a copy's layout
        # (two-factor row/column index); tiles must not straddle an atom.
        if (batch != 1 or accumulate or dest['mic'] % spec['bm'] or dest['nic'] % spec['bn']
                or (dest['scn'] != 1 and dest['scm'] != 1) or dest['offset'] % 4):
            return None, 'write-through destination layout not admitted by tile atoms'
        spec = {**spec, **{key: dest[key] for key in ('scm', 'scn', 'scm1', 'scn1', 'mic', 'nic')}}
    scratch, counter_init, operands = [], None, list(dict.fromkeys((spec['a_root'],spec['b_root'])))
    counters = ((spec['tiles']+63)//64*64) if spec['ksplit']>1 else 0
    partials = spec['tiles']*spec['ksplit']*spec['bm']*spec['bn'] if counters else 0
    scratch_name = name+'__popsb_scratch'
    if counters and scratch_name in set(occupied_names):
        return None, 'P-OPS-B scratch name collision; existing body retained'
    if counters and counters_at is not None:
        # P-OPS2-D persistent counters: the caller owns a program-level counter
        # area, zeroed once per launch; B's last arriver restores each counter to
        # zero after use. Private scratch then holds only partial tiles, which the
        # product defines before its own reduction reads them: no producer phase.
        scratch = [{'name':scratch_name,'shape':(partials,), 'dtype':'float32',
                    'role':'partials','counter_words':0,'partial_words':partials,
                    'initialization':'partial words defined by matrix slice stores'}]
        part, cnt = ident(scratch_name), f'reinterpret_cast<unsigned*>({counters_at})'
    elif counters:
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
            f"{str(spec['bkc']).lower()},{spec['stages']},{layout_name},{spec['kb']}{',true' if accumulate else ''}>("
            f"{ident(spec['a_root'])} + {spec['a_offset']}LL, {ident(spec['b_root'])} + {spec['b_offset']}LL, "
            f"{dest['pointer'] if dest is not None else ident(name)}, {part}, {cnt}, /*B_VBASE*/, smem);")
    metadata = {'engine':'popsb_cpasync','tiles':spec['tasks'],'batch':batch,'k':k,
        'ksplit':spec['ksplit'],'matrix_shape':[batch,m,n,k],
        'shared_bytes':spec['shared_bytes'],'matrix_plan':{key:spec[key] for key in
            ('bm','bn','kb','stages','ksplit','tiles','tasks','steps','waves','estimate_us')},
        'requires_full_grid':True,'selection_grid':grid,
        'internal_mutations':[scratch_name] if counters else [],
        'persistent_counter_words':counters if counters_at is not None else 0,
        'arithmetic':'strict FP32 ascending-K fmaf within slices; fixed ascending split sum; no TF32/tensor cores/data atomics',
        'copy_elision':'deferred: common attention/layout rewriting owns views',
        'phase_schedule':'common A/E effects scheduler; no donor packing pass',
        'donor_commit':B_DONOR_COMMIT,
        'model_context':'donor constants fitted at 2 CTAs/SM; integrated resource policy recorded separately'}
    if selection_fallback:
        metadata['matrix_selector_fallback'] = selection_fallback
    # P-OPS2-D: 'inline': False gives the product body its own (noinline) register
    # allocation instead of inlining it into the whole cooperative kernel.
    helpers = B_HELPERS if (engine or {}).get('inline', True) else B_HELPERS.replace('#define POPS_INLINE\n', '')
    return {'code':code,'helpers':helpers+'\n'+layout_definition,
            'headers':[B_HEADER],'metadata':metadata,'scratch':scratch,
            'counter_init':counter_init,'operand_roots':operands}, 'admitted pinned P-OPS-B cp.async FP32 body'


def fp32_selected(a, b, name, la, lb, shape, *, ident, arch='sm_90', grid=128,
                  split=True, occupied_names=(), counters_at=None, inline=True,
                  bodies=('simt', 'popsb'), accumulate=False):
    """Bind A's selected body to the serial/D region scratch and counter ABI.

    Full-grid rotation is required, as for the original B body. With persistent
    counters each region owns disjoint counter words, initialized once at kernel
    entry; only partial tiles participate in the region's ordinary lifetime plan.
    The selected headers/arithmetic are unchanged from the qualified A pin.
    """
    from .matrix_select import plan
    selected, reason = plan(a, b, name, la, lb, shape, ident=ident, arch=arch,
                            grid=grid, split=split, bodies=bodies)
    if selected is None:
        return None, reason
    code, helper, metadata = selected['code'], selected['helpers'], dict(selected['metadata'])
    if accumulate:
        # Plan first as an ordinary store. Only the final store adds the caller
        # value; no partial accumulator, FMA chain or split sum sees it.
        import re
        body = metadata.get('body')
        prefix = {'popsb': 'pops::gemm_phase', 'simt': 'pops2::simt_phase'}.get(body)
        if prefix is None:
            from .lower import LoweringError
            raise LoweringError(f'preserved accumulating matrix body is unsupported: {body}')
        code, count = re.subn(r'(' + re.escape(prefix) + r'<[^<>]+)>\(', r'\1,true>(', code, count=1)
        if count != 1:
            raise RuntimeError('preserved matrix accumulation has an unexpected store call')
    need = metadata.pop('p2_split', None)
    counters = need['counter_words'] if need else 0
    partials = need['partial_words'] if need else 0
    scratch, initialization = [], None
    scratch_name = name + '__pops2_scratch'
    if counters:
        if scratch_name in set(occupied_names):
            return None, 'A scratch name collision; D engine retained'
        persistent = counters_at is not None
        scratch = [{'name': scratch_name, 'shape': (partials if persistent else counters + partials,),
                    'dtype': 'float32', 'role': 'partials' if persistent else 'counter_and_partials',
                    'counter_words': 0 if persistent else counters, 'partial_words': partials,
                    'initialization': 'partial words defined by matrix slice stores'}]
        ptr = ident(scratch_name)
        if persistent:
            part, counter = ptr, f'reinterpret_cast<unsigned*>({counters_at})'
        else:
            part, counter = f'{ptr} + {counters}LL', f'reinterpret_cast<unsigned*>({ptr})'
            initialization = {'code': f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {counters}LL; i += (long long)gridDim.x * blockDim.x) reinterpret_cast<unsigned*>({ptr})[i] = 0u;',
                              'writes': [scratch_name], 'counter_words': counters, 'partial_words': partials}
        code = code.replace('/*P2_PART*/', part).replace('/*P2_CNT*/', counter)
    if not inline:
        helper = helper.replace('#define POPS_INLINE\n', '')
        if 'fp32_simt.cuh' in selected['headers']:
            helper = '#ifndef POPS2_NOINLINE\n#define POPS2_NOINLINE\n#endif\n' + helper
    metadata.update(persistent_counter_words=counters if counters_at is not None else 0,
                    internal_mutations=[scratch_name] if counters else [],
                    phase_schedule='common staged dependency scheduler; A body selection only',
                    donor_commit='4dcee751fab59f5ecea67ebada3167ca93fba841')
    return {**selected, 'code': code, 'helpers': helper, 'metadata': metadata,
            'scratch': scratch, 'counter_init': initialization}, reason
