"""Explicit graph-bound BF16 k8 emitter; never participates in default selection."""
from .bodies import _aligned, _unit
from .requirements import Precision, Requirements, Synchronization, check
from ..serial_bf16 import MMA_K8_CONTRACT, MMA_K8_RESIDUE_CONTRACT

BODY = 'mma_k8_readout'
HEADERS = ('tgemm.cuh', 'bgemm.cuh', 'popsc_mm.cuh', 'mma_k8.cuh')
# LATCH-MOE4-PARITY: wide CTA/warp tiles and grouped raster for the identical
# per-output ascending-k8 arithmetic (scheduling only). False restores 64x64.
WIDE = True
# Deeper vector stages amortize staging/barrier overhead for long contractions.
# The MMA instruction, ascending K order and partition boundaries are unchanged.
WIDE_KS = 64
# LATCH-MOE4-160: partitioned (serial input-VJP) products also take the 128x256
# tile; their exact BF16 partition subtotals are held packed. Scheduling only.
WIDE_PARTITION = True
# LATCH-MOE-EMA-FAST: issue the identical per-output k8 arithmetic as wgmma.m64nNk16 with a register A operand
# whose other k8 half is exact +0 (kernels/wgmma_k8.cuh). That instruction pair was measured bitwise equal to
# mma.sync.m16n8k8 on H100 (probes/k8_equiv*.cu receipts). Order, partitions, residue grouping, skipped edge
# halves and rounding are the reference's. Scheduling only; False restores the mma.sync body.
WGMMA = True
# Roles issued with wgmma, by component timing (LATCH-MOE-EMA-FAST a11: logits 6.71 -> 4.54 ms, 1024-token
# serial weight partials 8.37 -> 5.54 ms; the 2288-partition input VJP 9.39/9.26 ms stays on mma.sync).
# Every role is bitwise identical either way.
WGMMA_ROLES = frozenset({'forward', 'weight_partial', 'input_vjp'})
WGMMA_HEADERS = ('s107a_wgmma.cuh', 'pops4b_wg.cuh', 'pops4b_tma.cuh', 'wgmma_k8.cuh')
# Feed the warpgroup body with tensor-map (TMA) loads: one instruction per operand box instead of one cp.async per
# 16 bytes and thread. Maps are bound by the provider and encoded by the runtime (POPS4B_TMAP markers).
WGMMA_TMA = True


# LATCH-MOE-EMA-100: initial-residue input VJPs read the post-hole stages' A through an aligned, unswizzled 72-column
# raw box (latch_wgk8::TileT RAW) instead of element-wise staging; same values in registers and shared memory.
RAW_RESIDUE = True


def raw_residue_marker(a_root, a_offset, a_major_stride, m, k):
    """Unswizzled K-major raw A map: dims {K, rows}, box {72, 128} (POPS4B_TMAP swizzle field 0)."""
    return f'POPS4B_TMAP({a_root},{a_offset},{k},{m},{2 * a_major_stride},72,128,0)'


def wgmma_shared(ak, bn, tma=None, raw=False):
    """Shared bytes of latch_wgk8::Tile / TileT: four stages of A + swizzled B, plus 1024-byte alignment."""
    if WGMMA_TMA if tma is None else tma:
        return 4 * ((128 * 72 * 2 if raw else 128 * 128) + bn * 128) + 1024
    a_bytes = 128 * 72 * 2 if ak else 64 * 136 * 2
    return 4 * (a_bytes + bn * 128) + 1024


def tma_markers(a_root, a_offset, ak, a_major_stride, b_root, b_offset, bkc, b_major_stride, m, n, k, bn):
    """POPS4B_TMAP placeholders of the A / B operand views (latch_wgk8 TileT layouts)."""
    a = (f'POPS4B_TMAP({a_root},{a_offset},{k},{m},{2 * a_major_stride},64,128)' if ak else
         f'POPS4B_TMAP({a_root},{a_offset},{m},{k},{2 * a_major_stride},64,64)')
    b = (f'POPS4B_TMAP({b_root},{b_offset},{k},{n},{2 * b_major_stride},64,{bn})' if bkc else
         f'POPS4B_TMAP({b_root},{b_offset},{n},{k},{2 * b_major_stride},64,64)')
    return a, b


def emit(provider, problem, binding, *, a, b, name, out, ident, out_pitch):
    p = problem
    partition = binding.get('partition_k')
    role = binding.get('role')
    arithmetic_contract = binding.get('contract')
    initial_residue = arithmetic_contract == MMA_K8_RESIDUE_CONTRACT
    if (arithmetic_contract not in (MMA_K8_CONTRACT,MMA_K8_RESIDUE_CONTRACT)
            or binding.get('a') != a or binding.get('b') != b
            or binding.get('shape') != [p.m,p.n,p.k]
            or p.dtype != 'bfloat16' or p.out_dtype != 'bfloat16' or p.out_mode != 'store'
            or type(partition) is not int or role not in ('forward','input_vjp','weight_partial')
            or (role in ('forward','weight_partial') and partition != 0)
            or (role == 'input_vjp' and (partition <= 0 or partition % 16 or partition >= p.k))):
        raise ValueError(f'{name}: k8 readout contract changed')
    if initial_residue and (role != 'input_vjp' or binding.get('residue_tile_k') != 8
                            or binding.get('k_grouping') != 'initial-short-then-full-k8'):
        raise ValueError(f'{name}: k8 initial-residue contract changed')
    if not initial_residue and any(key in binding for key in ('residue_tile_k','k_grouping')):
        raise ValueError(f'{name}: residue fields need an explicit initial-residue contract')
    if p.batch != 1 or not _unit(p) or not _aligned(p,8):
        raise ValueError(f'{name}: k8 readout requires batch1 unit-stride matrices and aligned leading strides/offsets')
    pitch = out_pitch if out_pitch is not None else p.n
    if pitch < p.n or (out_pitch is not None and pitch % 8):
        raise ValueError(f'{name}: invalid physical k8 output pitch')
    # Reuse the existing edge body layout. Long wide contractions use 64 K
    # values per stage; the narrow/short path retains its 32-value stages.
    wgmma = WIDE and WGMMA and role in WGMMA_ROLES
    raw = bool(wgmma and WGMMA_TMA and RAW_RESIDUE and initial_residue and p.ak and not p.bkc)
    if wgmma:
        bm, bn, nst, wm, wn, ks = 128, 256, 4, 2, 1, 64
        shared = wgmma_shared(p.ak, bn, raw=raw)
    else:
        if WIDE:
            bm, bn, nst, wm, wn = 128, (128 if partition and not WIDE_PARTITION else 256), 4, 2, 4
            ks = WIDE_KS if p.k >= 256 and p.m >= bm and p.n >= bn else 32
        else:
            bm, bn, nst, wm, wn = 64, 64, 3, 4, 2
            ks = 32
        a_el = bm*(ks+8) if p.ak else ks*(bm+8)
        b_el = bn*(ks+8) if p.bkc else ks*(bn+8)
        shared = nst*(a_el+b_el)*2
    instruction = (('mma.sync.m16n8k8 arithmetic issued as wgmma.m64nNk16 register-A zero-half pairs'
                    + (' (TMA operands)' if WGMMA_TMA else '')) if wgmma else 'mma.sync.m16n8k8')
    headers = HEADERS + (WGMMA_HEADERS if wgmma else ())
    precision = Precision('bfloat16','float32','bfloat16',
        'BF16 RN once for forward/weight partial; BF16 RN(partial+previous BF16) at bound input-VJP partitions',
        ('ascending original K; initial short k8 group then full groups in final partition; FP32 accumulator from zero per partition'
         if initial_residue else 'ascending k8; FP32 accumulator from zero per partition'),
        instruction,arithmetic_contract)
    req = Requirements('sm_90a' if wgmma else 'sm_90', shared, headers=headers,
        host_setup=(('tensor_map:A', 'tensor_map:B') if wgmma and WGMMA_TMA else ()),
        sync=Synchronization(requires_full_grid=True, smem_mbarriers=4 if wgmma and WGMMA_TMA else 0), precision=precision)
    admitted, reasons = check(req, provider.resources)
    if not admitted:
        raise ValueError(f'{name}: k8 readout body refused: {reasons}')
    tiles = ((p.m+bm-1)//bm)*((p.n+bn-1)//bn)
    if wgmma and WGMMA_TMA:
        ta, tb = tma_markers(p.a.root, p.a.offset, p.ak, p.a.row_stride if p.ak else p.a.col_stride,
                             p.b.root, p.b.offset, p.bkc, p.b.col_stride if p.bkc else p.b.row_stride, p.m, p.n, p.k, bn)
        if raw:
            tr = raw_residue_marker(p.a.root, p.a.offset, p.a.row_stride, p.m, p.k)
            call = (f"latch_wgk8::phase_wide_tma<{str(p.ak).lower()},{str(p.bkc).lower()},{partition},"
                    f"{str(initial_residue).lower()},{p.k},/*K8_ROUNDED_F32*/false,true,true>(g,{ta},{tb},smem,0,{tr});")
        else:
            call = (f"latch_wgk8::phase_wide_tma<{str(p.ak).lower()},{str(p.bkc).lower()},{partition},"
                    f"{str(initial_residue).lower()},{p.k},/*K8_ROUNDED_F32*/false,true>(g,{ta},{tb},smem);")
    elif wgmma:
        call = (f"latch_wgk8::phase_wide<{str(p.ak).lower()},{str(p.bkc).lower()},{partition},"
                f"{str(initial_residue).lower()},{p.k},/*K8_ROUNDED_F32*/false,true>(g,smem);")
    else:
        call = (f"latch_mma_k8::{'phase_wide' if WIDE else 'phase'}<{str(p.ak).lower()},{str(p.bkc).lower()},{partition},"
                f"{str(initial_residue).lower()}{(',' + str(ks)) if WIDE else ''},/*K8_ROUNDED_F32*/false"
                f"{(',' + str(bool(partition and WIDE_PARTITION)).lower()) if WIDE else ''}>(g,smem);")
    code = f'''{{
  popsc::BArgs g{{}};
  g.A = reinterpret_cast<const s101h::bf16*>({ident(p.a.root)}) + {p.a.offset}LL;
  g.B = reinterpret_cast<const s101h::bf16*>({ident(p.b.root)}) + {p.b.offset}LL;
  g.C = reinterpret_cast<s101h::bf16*>({out});
  g.sam={p.a.row_stride}LL; g.sak={p.a.col_stride}LL; g.bsa=0;
  g.sbk={p.b.row_stride}LL; g.sbn={p.b.col_stride}LL; g.bsb=0;
  g.ldc={pitch}LL; g.bsc=0; g.M={p.m}; g.N={p.n}; g.K={p.k}; g.nb=1;
  {call}
}}'''
    decision = dict(output=name, provider='latch_matrix', policy=provider.policy,
        selected=BODY, rule=('explicit paired-readout k8 arithmetic, wgmma zero-half issue' if wgmma
                             else 'explicit paired-readout mma.sync k8 arithmetic'),
        admitted=[BODY], refused={}, arithmetic_contract=dict(binding),
        plan=dict(bm=bm,bn=bn,wm=wm,wn=wn,ks=ks,nst=nst,shared_bytes=shared,tiles=tiles,
                  raster='group8' if WIDE else 'n-fastest', instruction=instruction))
    if out_pitch is not None:
        decision['out_pitch'] = out_pitch
    provider.decisions.append(decision)
    provider.emitted.append(name)
    extra = dict(tiles=tiles,k=p.k,shared_bytes=shared,threads=256,
        engine='latch_matrix:'+BODY,matrix_shape=[p.m,p.n,p.k],batch=1,
        matrix_plan=decision['plan'],arithmetic=precision.order+'; '+precision.rounding,
        serial_bf16_contract=dict(binding),
        k8_widen_respec=dict(output=out, pitch=pitch, shape=[p.m,p.n], enabled=partition == 0),
        latch_matrix=dict(body=BODY,bitwise_class=arithmetic_contract,arch='sm_90a' if wgmma else 'sm_90',
                          entry_resident=False,host_setup=[],instruction=instruction),
        matrix_spec=dict(engine='latch_matrix:'+BODY,a=a,b=b,out=out,m=p.m,n=p.n,k=p.k,batch=1,
                         roots=[(p.a.root,int(p.a.offset)),(p.b.root,int(p.b.offset))]))
    contract = dict(work_items=tiles,execution_grid_scope='physical_cooperative_grid',
        requires_full_grid=True,internal_mutations=[],counter_effects=[],private_scratch=[])
    return dict(code=code,headers=list(headers),helpers=[],extra=extra,decision=decision,
                contract=contract,out_pitch=out_pitch)
