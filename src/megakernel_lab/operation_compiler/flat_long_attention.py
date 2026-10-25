"""P-OPS4-D long-context training attention lowering.

Applies to an attention group matched by attention_lower.plan (scores, scale, additive mask, softmax, value product
and the PyTorch-derived backward) whose operands are FP32 widenings of BF16 values and whose results are consumed
through one BF16 cast (P-OPS3-B admission and layout absorption, ops/short_attention.py). The group becomes four
atomic phases in the cooperative program, with no T x T storage:

    prepare  proof tables: per-tile key norms and mask tile statistics (ops/kernels/attention_long_flat.cuh)
    forward  tiled online-softmax forward: BF16 output in its consumer's layout, log2 row log-sum-exp, live tiles
    dq       delta = rowsum(dO * O) and dQ, query-block items
    kv       dK and dV, key-block items (no atomics: deterministic)

Admission depends only on graph structure, dtypes, shapes and strides; anything else is a recorded refusal and keeps
the existing lowering. The arithmetic recipe is an explicit option (see RECIPES); the default preserves every FP32
product of the captured attention (exact three-piece BF16 splits), so no automatic choice changes arithmetic class.
"""
from __future__ import annotations

import math

from . import short_attention as S

HEADER = "attention_long_flat.cuh"
HEAD_WIDTH = 64
MIN_SEQUENCE = 256
MAX_SEQUENCE = 4096          # live-tile proof words hold 64 key tiles
SEQUENCE_MULTIPLE = 128      # forward/dQ items are 128 query rows, dK/dV items 128 keys
# recipe -> (BF16 pieces of P and dS, delta from a saved FP32 output)
RECIPES = {"exact3": (3, True), "split2": (2, True), "bf16": (1, False)}
DEFAULT_RECIPE = "exact3"
LD = 72
VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
MERGE_BWD = False     # P9-A: v7 backward as delta phase + one merged dQ/dK-dV item phase (exact)
FWD_SPEC = False     # (also dq/kv items: dq_item6s / kv_item6s)
BWD_SPEC = False
_FWD_SPEC_DOC = True      # P9-A: v7 forward items speculatively load K/V tiles 0/1 with Q (forward_item6s, exact)
KN_IN_ITEM = False   # P9-A: v7 forward items compute their head's key norms (no per-layer key-norm prepare)
KN_EPILOGUE = True   # P9-A: per-layer key-norm tables come from the K product's epilogue (no prepare phase; bitwise)
KN_EPILOGUE_OWNER = True  # P9-A: also the mask-table owner (its prepare keeps only the mask units)
KN_EPI_TARGETS = {}  # K operand ident -> key-norm table placement (filled while lowering one program)
DQ_SMEM_A = False    # P9-A: v7 dQ items read Q/dO wgmma operands from shared memory (dq_item6a, bitwise)
KV_SMEM_A = False    # P9-A: v7 dK/dV items read K/V wgmma operands from shared memory (kv_item6a, bitwise)
FWD_WIDE = False     # P9-A: v7 forward items over 128 query rows (two m64 halves per warpgroup, forward_item6w, bitwise)
FWD_DEEP = False     # P9-A: v7 forward items use the four-stage K/V ring (forward_item6d, bitwise)
MASK_LIVE = False    # P9-A: masked groups decide tile liveness from the mask tables alone (no key-norm prepare)
DEFAULT_VERSION = 7  # measured best on H100 (P-OPS4-D gpu-14/16/17); v3 is the best mma.sync schedule
WGMMA_VERSIONS = (4, 5, 6, 7, 8, 9, 10, 11, 12)
# v11 = v7 forward + single-pass backward: delta (+ fixed-point scales, accumulator zeroing), dK/dV items that also
# add int32 fixed-point dQ^T partials with bulk shared->global reductions (order-independent, deterministic), dQ
# conversion. BF16 recipe only (exact recipes fall back to v7).
FUSED_DQ_VERSIONS = (11,)
BF16_ONLY_VERSIONS = (9, 11, 12, 13)
# v13: explicit BF16 SDPA only. Reverse 128-key forward tiles with the
# PyTorch 2.9.1 Flash softmax/backward operation order, reusing v3 staging/MMA.
# Flash source pin: 979702c87a8713a8e0a5e9fee122b90d2ef13be5.
# Deterministic dQ partition policy is bound to the declared 132-SM H100 target.
FLASH_TARGET_SMS = 132
# v12 = v7 arithmetic with 128-key steps (m64n128 score products) in the forward and dQ items; v7 dK/dV. BF16 only.
# v10 = v7 with FA3-style intra-warpgroup pipelining (deferred value/gradient product waits, dQ score lookahead).
# v9 = v7 forward + stored-dS backward (delta, dK/dV storing BF16 dS^T, dQ from stored tiles); BF16 recipe only.
STORED_DS_VERSIONS = (9,)
# v8 = v7 forward with half-tile (m64n32) backward items that overlap tensor and elementwise work.
# v7 = v6 items compiled with PFLATLONG_FTZ (flush-to-zero probability exponentials, declared) and PFLATLONG_DEFER_GSCALE.  # Hopper warpgroup products: programs must be built for sm_90a
# v2: two m16 tiles per warp, split-K/split-Q warp pairs, 3-stage (2-stage dK/dV) pipelines; mirrors *2_shared_bytes.
PAIR = 4 * 64 * LD
SHARED_BYTES_V2 = {"prepare": 0, "forward": 2 * 3 * PAIR + 128, "dq": 2 * 3 * PAIR + 128,
                   "kv": 2 * 2 * 128 * LD + 2 * (2 * PAIR + 4 * 2 * 2 * 64) + 128}
# v3: v1 shape, precomputed tile bits, three-stage ring with one barrier per tile, next-tile/chunk products issued
# before the current softmax/elementwise work; mirrors *3_shared_bytes.
KV1 = 2 * 64 * LD
SHARED_BYTES_V3 = {"prepare": 0, "forward": 2 * 3 * KV1 + 128, "dq": 2 * 3 * KV1 + 128,
                   "kv": 2 * 3 * KV1 + 3 * 4 * 2 * 64 + 128}
# v4: two warpgroups, wgmma m64n64k16 products (register A, 128B-swizzled shared B); mirrors *4_shared_bytes.
ST4 = 2 * 64 * 128
SHARED_BYTES_V4 = {"prepare": 0, "forward": 1024 + 3 * ST4 + 2 * 128 * LD + 128,
                   "dq": 1024 + 3 * ST4 + 2 * 2 * 128 * LD + 128,
                   "kv": 1024 + 3 * ST4 + 3 * 4 * 2 * 64 + 2 * 2 * 128 * LD + 128}
# v5: v4 products, two independent warpgroup workers on 64-row / 64-key items; mirrors v5_shared_bytes.
SHARED_BYTES_V5 = {"prepare": 0, "forward": 1024 + 2 * 50 * 1024, "dq": 1024 + 2 * 50 * 1024, "kv": 1024 + 2 * 50 * 1024}
SHARED_BYTES = {
    "prepare": 0,
    "forward": 2 * (128 * LD + 4 * 64 * LD) + 8 * 8 + 16,
    "dq": 2 * (2 * 128 * LD + 4 * 64 * LD) + 8 * 8 + 16,
    "kv": 2 * (2 * 128 * LD) + 2 * (2 * 2 * 64 * LD + 4 * 2 * 64) + 2 * 8 * 2 + 16,
}


class Refused(S.Refused):
    pass


def admit(groups, graph_args, static_layouts, protected, *, min_sequence=MIN_SEQUENCE):
    """Return ({group label: resolved plan}, refusals) for matched attention groups."""
    graph = S._Graph(*graph_args)
    plans, refusals = {}, []
    for group in groups:
        label = group["label"]
        try:
            plan = _plan_group(graph, group, min_sequence)
            plans[label] = S.finalize(plan, graph_args, static_layouts, protected, cast_first=True)
        except S.Refused as exc:
            refusals.append({"group": label, "reason": "long attention: " + str(exc)})
    return plans, refusals


def _plan_group(graph, group, min_sequence):
    if not group.get("backward"):
        raise Refused("forward-only attention keeps the existing lowering")
    bh, t, hd = graph.shape(group["q"])
    if graph.shape(group["kt"]) != (bh, hd, t) or graph.shape(group["v"]) != (bh, t, hd):
        raise Refused("self-attention with equal key/value head width required")
    if hd != HEAD_WIDTH:
        raise Refused(f"head width {hd} != {HEAD_WIDTH}")
    if not min_sequence <= t <= MAX_SEQUENCE or t % SEQUENCE_MULTIPLE:
        raise Refused(f"context {t} outside [{min_sequence}, {MAX_SEQUENCE}] or not a multiple of {SEQUENCE_MULTIPLE}")
    members = set(group["skip"])
    views = {group[k] for k in ("kt", "k_view", "qt_view", "vt_view") if group.get(k)}
    return dict(group=group, bh=bh, t=t, hd=hd, splits=0, group_names=members | views)


def aux_names(group):
    o = group["o"]
    return {key: o + "__long_" + key for key in ("kn", "mmax", "mzero", "rm", "lse2", "live", "delta", "o32", "dsbuf",
                                                 "vn", "dqf", "dqacc")}


def _l4(x):
    return "pflatlong::Layout4{" + f"{x[0]}LL, {x[1]}LL, {x[2]}LL" + "}"


def emit(plan, ident, mask_strides, *, which, recipe=DEFAULT_RECIPE, version=DEFAULT_VERSION, mask_tables=None,
         diag=None, keynorm_epilogue=False, keynorm_in_item=False, keynorm_external=False):
    """Return (kind, code, extra) for one phase: extra has outputs {name: None (BF16 final) | (shape, dtype)}, reads.

    keynorm_in_item=True selects the v7 forward items that compute their head's key norms (forward_item6k,
    identical prepare_unit arithmetic) instead of prepare-phase key-norm units (ops.attention_rules kn_item).
    keynorm_external=True: the key-norm table is written by another phase (ops.attention_pack, same arithmetic);
    the prepare phase keeps only mask-table units and the forward items read the table as usual.
    """
    if recipe not in RECIPES:
        raise Refused(f"unknown long-attention recipe {recipe!r}")
    diag = dict(diag or {})  # diagnostic overrides only: {"fold": bool, "phase_versions": {phase: version}}
    version = int(diag.get("phase_versions", {}).get(which, version))
    if version not in VERSIONS:
        raise Refused(f"unknown long-attention schedule version {version!r}")
    if version in BF16_ONLY_VERSIONS and RECIPES[recipe][0] != 1:
        raise Refused(f"schedule v{version} is defined for the BF16 recipe only")
    if version == 13 and plan.get("arithmetic_contract") != "bf16-flash-sdpa-v1":
        raise Refused("schedule v13 requires the explicit BF16 SDPA contract")
    suffix = "" if version == 1 else ("6" if version == 7 or (version in (8, 9, 11) and which in ("prepare", "forward")) or (version == 12 and which == "kv") else str(version))
    fused = version in FUSED_DQ_VERSIONS
    pieces, o32 = RECIPES[recipe]
    group, bh, t = plan["group"], plan["bh"], plan["t"]
    if version == 13 and group.get("mask_kind") == "causal_window":
        raise Refused("schedule v13 does not implement causal-window masks; select schedule v7")
    s, o = plan["in_strides"], plan["out_strides"]
    src, fin = plan["sources"], plan["finals"]
    mode = group["scale_mode"]
    scale = float(group["scale"]) if mode else 1.0
    if version == 13 and (mode != 1 or not math.isfinite(scale) or scale <= 0):
        raise Refused("schedule v13 requires a finite positive multiplicative scale")
    if version == 13 and math.frexp(scale)[0] != 0.5:
        raise Refused("schedule v13 Flash scale policy requires an exact positive power of two")
    # v6: an exact positive power-of-two multiplicative score scale folds into the exponent FMA (bitwise equal).
    pow2 = mode == 1 and scale > 0 and math.frexp(scale)[0] == 0.5 and diag.get("fold", True)
    template = f"{pieces}, {'true' if pow2 else 'false'}" if version in (6, 7, 8, 9, 10, 11, 12) else f"{pieces}"
    flash_splits = min((FLASH_TARGET_SMS + bh - 1) // bh, t // 128) if version == 13 else None
    if version == 13 and which == "dq":
        template = f"{pieces}, {flash_splits}"
    nt = t // 64
    aux = aux_names(group)
    # Mask statistics are a function of the mask tensor only: groups reading the same mask with the same
    # (head batch, context) reuse the first group's tables instead of recomputing them.
    owned_tables = mask_tables is None
    if not owned_tables:
        aux.update({key: mask_tables[key] for key in ('mmax', 'mzero', 'rm')})
    has_mask = mask_strides is not None
    from . import padding_masks
    extended = padding_masks.fields(group, mask_strides)
    if extended and version != 7:
        raise Refused(f"padding-mask groups (release step 09) are implemented by the v7 items only, not v{version}")
    # Mask tables are per mask batch: bh, or bh / heads for a per-example mask shared by every head.
    mb = ((getattr(mask_strides, "batches", None) or bh) if mask_strides[0] != 0 else 1) if has_mask else 0
    mask_live = MASK_LIVE and has_mask and not fused
    kn_item = (KN_IN_ITEM or keynorm_in_item) and version == 7 and not fused and not mask_live
    kn_epi = None
    if keynorm_epilogue and KN_EPILOGUE and version == 7 and not fused and not mask_live and not kn_item and (mask_tables is not None or KN_EPILOGUE_OWNER):
        sb, sh, sr = s['k'][0], s['k'][1], s['k'][2]
        if sh == 64 and sr == plan['heads'] * 64 and sb == t * sr and not plan.get('source_offsets', {}).get('k'):
            key = ident(src['k'])
            if key not in KN_EPI_TARGETS:
                KN_EPI_TARGETS[key] = dict(base=sum(v['size'] for v in KN_EPI_TARGETS.values()), size=bh * nt,
                                           t=t, heads=plan['heads'], nt=nt, ldc=sr)
            kn_epi = KN_EPI_TARGETS[key]
    fields = [
        f"A.q = {S.source_pointer(plan, 'q', ident)}; A.lq = {_l4(s['q'])};",
        f"A.k = {S.source_pointer(plan, 'k', ident)}; A.lk = {_l4(s['k'])};",
        f"A.v = {S.source_pointer(plan, 'v', ident)}; A.lv = {_l4(s['v'])};",
        f"A.heads = {plan['heads']}; A.bh = {bh}; A.t = {t}; A.scale = {repr(scale)}f; A.scale_mode = {mode};",
        "A.kn = nullptr;" if (mask_live or kn_item) else
        (f"A.kn = v_p9kn + {kn_epi['base']};" if kn_epi else f"A.kn = {ident(aux['kn'])};"),
    ]
    def fused_fields(*keys):
        parts = {"vn": f"A.vn = {ident(aux['vn'])};", "dqf": f"A.dqf = {ident(aux['dqf'])};",
                 "dqacc": f"A.dqacc = (int*){ident(aux['dqacc'])};"}
        return [parts[key] for key in keys]
    if has_mask:
        fields.append(f"A.mask = {ident(group['mask'])}; A.m_b = {mask_strides[0]}LL; A.m_i = {mask_strides[1]}LL; "
                      f"A.m_j = {mask_strides[2]}LL; A.has_mask = 1; A.mask_batched = {int(mask_strides[0] != 0)};")
        fields.append(f"A.mmax = {ident(aux['mmax'])}; A.mzero = {ident(aux['mzero'])}; A.rm = {ident(aux['rm'])};")
    else:
        fields.append("A.mask = nullptr; A.m_b = A.m_i = A.m_j = 0; A.has_mask = 0; A.mask_batched = 0; "
                      "A.mmax = nullptr; A.mzero = nullptr; A.rm = nullptr;")
    if extended:
        fields.append(extended)  # release step 09: per-example mask batches / boolean-mask empty rows
    mask_reads = [group["mask"]] if has_mask else []
    sources = [src["q"], src["k"], src["v"]]
    rows_per_item = 64 if version in (5, 6, 7, 8, 9, 10, 11, 12) else 128
    nqb = t // rows_per_item
    items = bh * nqb
    if which == "prepare":
        # no key-norm units: mask-only liveness / forward items / K epilogue / an operand pack phase (keynorm_external)
        skip = bh * nt if (mask_live or kn_item or kn_epi or keynorm_external) else 0
        units = bh * nt - skip + (mb * nt * nt if owned_tables else 0)
        if units == 0:
            return None
        code = ("pflatlong::Args A{};\n" + "\n".join(fields + (fused_fields("vn") if fused else [])) + f"""
// One warp per proof-table unit (key-norm tiles, then mask tiles); no block barriers.
for (long long u = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); u < {units}LL; u += (long long)gridDim.x * 8)
  pflatlong::prepare_unit(A, u + {skip}LL);""")
        outputs = {} if (mask_live or kn_item or kn_epi or keynorm_external) else {aux["kn"]: ((bh, nt), "float32")}
        if has_mask and owned_tables:
            outputs.update({aux["mmax"]: ((mb, nt, nt), "float32"), aux["mzero"]: ((mb, nt, nt), "float32"),
                            aux["rm"]: ((mb, nt, t), "float32")})
        if fused:
            outputs[aux["vn"]] = ((bh, nt), "float32")
        return "attention_long_prepare", code, dict(work_items=(units + 7) // 8, shared_bytes=0, head_width=64,
                                                     sequence=t, head_batch=bh, recipe=recipe, outputs=outputs,
                                                     reads=([] if (kn_epi or keynorm_external) else [src["k"]])
                                                     + ([src["v"]] if fused else []) + mask_reads)
    tables = ([] if (mask_live or kn_item or kn_epi) else [aux["kn"]]) + ([aux["mmax"], aux["mzero"], aux["rm"]] if has_mask else [])
    fields.append(f"A.lse2 = {ident(aux['lse2'])}; A.live = (unsigned long long*){ident(aux['live'])};")
    fields.append(f"A.o = {ident(fin['o'])}; A.lo = {_l4(o['o'])};")
    fields.append(f"A.o32 = {ident(aux['o32']) if o32 else 'nullptr'};")
    loop = "\n".join([
        f"for (long long r = 0;; ++r) {{",
        f"  const long long idx = pflatlong::snake(r);",
        f"  if (idx >= {items}LL) break;",
        f"  if (r) __syncthreads();  // shared reuse by the next item",
        f"  @@BODY@@",
        f"}}"]) if version not in (5, 6, 7, 8, 9, 10, 11, 12) else "\n".join([
        "// Two independent warpgroup workers per CTA (no block-wide barrier inside the phase).",
        f"for (long long r = 0;; ++r) {{",
        f"  const long long idx = pflatlong::snake2(r);",
        f"  if (idx >= {items}LL) break;",
        f"  @@BODY@@",
        f"}}"])
    if which == "forward":
        fsuffix = ("6d" if FWD_DEEP and suffix == "6" and version == 7 else "6k" if kn_item and suffix == "6" else
                   "6s" if FWD_SPEC and suffix == "6" and version == 7 else suffix)
        if extended and (fsuffix != "6" or FWD_WIDE):
            raise Refused("padding-mask empty rows are implemented in forward_item6 only")
        body = f"pflatlong::forward_item{fsuffix}<{template}>(A, (int)(idx % {bh}LL), {nqb - 1} - (int)(idx / {bh}LL), smem);"
        if FWD_WIDE and fsuffix == "6" and version == 7 and nqb % 2 == 0:
            body = f"pflatlong::forward_item6w<{template}>(A, (int)(idx % {bh}LL), {nqb // 2 - 1} - (int)(idx / {bh}LL), smem);"
            loop = loop.replace(f"if (idx >= {items}LL) break;", f"if (idx >= {items // 2}LL) break;")
            items = items // 2
        outputs = {fin["o"]: None, aux["lse2"]: ((bh, t), "float32"), aux["live"]: ((bh, t // 16), "int64")}
        if o32:
            outputs[aux["o32"]] = ((bh, t, 64), "float32")
        reads = sources + mask_reads + tables
        comment = "// 128-query items, decreasing causal cost, boustrophedon over CTAs (static LPT)."
    else:
        fields.append(f"A.dout = {S.source_pointer(plan, 'do', ident)}; A.ld = {_l4(s['do'])};")
        fields.append(f"A.delta = {ident(aux['delta'])};")
        reads = sources + [src["do"], aux["lse2"], aux["live"]] + mask_reads + ([aux["mzero"]] if has_mask else [])
        if version in STORED_DS_VERSIONS:
            fields.append(f"A.dsbuf = {ident(aux['dsbuf'])};")
        if which == "delta" and fused:
            units = bh * (t // 16)
            code = ("pflatlong::Args A{};\n" + "\n".join(fields + fused_fields("vn", "dqf", "dqacc")) + f"""
// delta = rowsum(dO * O), fixed-point dQ scales and accumulator zeroing: one warp per 16-row unit (v11).
for (long long u = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); u < {units}LL; u += (long long)gridDim.x * 8)
  pflatlong::delta_unit11(A, u);""")
            return "attention_long_delta", code, dict(
                work_items=(units + 7) // 8, shared_bytes=0, head_width=64, sequence=t, head_batch=bh, recipe=recipe,
                schedule_version=version,
                outputs={aux["delta"]: ((bh, t), "float32"), aux["dqf"]: ((bh, t), "float32"),
                         aux["dqacc"]: ((bh, nt, 64, 64), "float32")},  # int32 words in FP32-typed storage
                reads=[src["do"], fin["o"], aux["kn"], aux["vn"]])
        if which == "dq" and fused:
            units = bh * nt
            fields.append(f"A.dq = {ident(fin['dq'])}; A.ldq = {_l4(o['dq'])};")
            code = ("pflatlong::Args A{};\n" + "\n".join(fields + fused_fields("dqf", "dqacc")) + f"""
// dQ conversion from the fixed-point accumulator: one warp per (head batch, 64-query tile) unit (v11).
for (long long u = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); u < {units}LL; u += (long long)gridDim.x * 8)
  pflatlong::dq_convert_unit11<{'true' if pow2 else 'false'}>(A, u);""")
            # dK^T is read only to order this phase after the dK/dV/dQ phase that adds into the accumulator.
            return "attention_long_dq", code, dict(
                work_items=(units + 7) // 8, shared_bytes=0, head_width=64, sequence=t, head_batch=bh, recipe=recipe,
                schedule_version=version, outputs={fin["dq"]: None},
                reads=[aux["dqacc"], aux["dqf"], fin["dkt"], fin["dv"]])
        if which == "delta6":
            units = bh * (t // 16)
            code = ("pflatlong::Args A{};\n" + "\n".join(fields) + f"""
// delta = rowsum(dO * O) with the dQ item's per-row arithmetic: one warp per 16-row unit (P9-A merged backward).
for (long long u = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); u < {units}LL; u += (long long)gridDim.x * 8)
  pflatlong::delta_unit6(A, u);""")
            return "attention_long_delta", code, dict(work_items=(units + 7) // 8, shared_bytes=0, head_width=64,
                                                       sequence=t, head_batch=bh, recipe=recipe, schedule_version=version,
                                                       outputs={aux["delta"]: ((bh, t), "float32")},
                                                       reads=[src["do"]] + ([aux["o32"]] if o32 else [fin["o"]]))
        if which == "dqkv":
            fields.append(f"A.dq = {ident(fin['dq'])}; A.ldq = {_l4(o['dq'])};")
            fields.append(f"A.dk = {ident(fin['dkt'])}; A.ldk = {_l4(o['dkt'])};")
            fields.append(f"A.dv = {ident(fin['dv'])}; A.ldv = {_l4(o['dv'])};")
            nkb = t // 64
            body = (f"const long long lv = idx / {2 * bh}LL, j = idx % {2 * bh}LL;\n"
                    f"  if (j < {bh}LL) pflatlong::kv_item6<{template}>(A, (int)j, (int)lv, smem);\n"
                    f"  else pflatlong::dq_item6n<{template}>(A, (int)(j - {bh}LL), {nkb - 1} - (int)lv, smem);")
            outputs = {fin["dq"]: None, fin["dkt"]: None, fin["dv"]: None}
            reads += [aux["delta"]] + ([aux["o32"]] if o32 else [fin["o"]])
            comment = "// dQ and dK/dV items in one phase, interleaved by decreasing cost (P9-A merged backward)."
            items = 2 * bh * nkb
            loop = "\n".join([
                "// Two independent warpgroup workers per CTA (no block-wide barrier inside the phase).",
                f"for (long long r = 0;; ++r) {{",
                f"  const long long idx = pflatlong::snake2(r);",
                f"  if (idx >= {items}LL) break;",
                f"  @@BODY@@",
                f"}}"])
        elif which == "delta":
            units = bh * (t // 16)
            code = ("pflatlong::Args A{};\n" + "\n".join(fields) + f"""
// delta = rowsum(dO * O): one warp per 16-row unit (stored-dS backward).
for (long long u = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); u < {units}LL; u += (long long)gridDim.x * 8)
  pflatlong::delta_unit(A, u);""")
            return "attention_long_delta", code, dict(work_items=(units + 7) // 8, shared_bytes=0, head_width=64,
                                                       sequence=t, head_batch=bh, recipe=recipe, schedule_version=version,
                                                       outputs={aux["delta"]: ((bh, t), "float32")},
                                                       reads=[src["do"], fin["o"]])
        if which == "dqkv":
            pass
        elif which == "dq" and version in STORED_DS_VERSIONS:
            fields.append(f"A.dq = {ident(fin['dq'])}; A.ldq = {_l4(o['dq'])};")
            body = f"pflatlong::dq_item{suffix}<{template}>(A, (int)(idx % {bh}LL), {nqb - 1} - (int)(idx / {bh}LL), smem);"
            outputs = {fin["dq"]: None}
            reads = [src["k"], aux["live"], aux["dsbuf"]]
            comment = "// dQ from stored dS tiles: 64-query items, decreasing causal cost."
        elif which == "dq":
            fields.append(f"A.dq = {ident(fin['dq'])}; A.ldq = {_l4(o['dq'])};")
            dsuffix = "6s" if BWD_SPEC and suffix == "6" and version == 7 else suffix
            if DQ_SMEM_A and dsuffix == "6" and version == 7:
                dsuffix = "6a"
            body = f"pflatlong::dq_item{dsuffix}<{template}>(A, (int)(idx % {bh}LL), {nqb - 1} - (int)(idx / {bh}LL), smem);"
            outputs = {fin["dq"]: None, aux["delta"]: ((bh, t), "float32")}
            reads += [aux["o32"]] if o32 else [fin["o"]]
            comment = "// delta and dQ: 128-query items, decreasing causal cost."
        elif which == "kv":
            fields.append(f"A.dk = {ident(fin['dkt'])}; A.ldk = {_l4(o['dkt'])};")
            fields.append(f"A.dv = {ident(fin['dv'])}; A.ldv = {_l4(o['dv'])};")
            ksuffix = "6s" if BWD_SPEC and suffix == "6" and version == 7 else suffix
            if KV_SMEM_A and suffix == "6" and version == 7 and not BWD_SPEC:
                ksuffix = "6a"
            body = f"pflatlong::kv_item{ksuffix}<{template}>(A, (int)(idx % {bh}LL), (int)(idx / {bh}LL), smem);"
            outputs = {fin["dkt"]: None, fin["dv"]: None}
            if version in STORED_DS_VERSIONS:
                outputs[aux["dsbuf"]] = ((bh, nt, nt, 64, 64), "bfloat16")
            if fused:
                fields.extend(fused_fields("dqf", "dqacc"))
                reads += [aux["dqf"], aux["dqacc"]]  # the accumulator is added into in place (bulk reductions)
            reads += [aux["delta"]]
            comment = "// dK and dV: 128-key items, decreasing causal cost."
        else:
            raise ValueError("unknown long-attention phase " + which)
    from . import p9_rowblock
    if p9_rowblock.TRACE:
        kind_code = {"forward": 1, "dq": 2, "kv": 3}.get(which, 9)
        body = f"P9AS({kind_code}, 0); {body} P9AS({kind_code}, 1);"
    code = "pflatlong::Args A{};\n" + "\n".join(fields) + "\n" + comment + "\n" + loop.replace("@@BODY@@", body)
    shared = {1: SHARED_BYTES, 2: SHARED_BYTES_V2, 3: SHARED_BYTES_V3, 4: SHARED_BYTES_V4, 5: SHARED_BYTES_V5,
              6: SHARED_BYTES_V5, 7: SHARED_BYTES_V5, 8: SHARED_BYTES_V5,
              9: {**SHARED_BYTES_V5, "dq": 1024 + 2 * 54 * 1024}, 10: SHARED_BYTES_V5,
              11: {**SHARED_BYTES_V5, "kv": 1024 + 2 * 74 * 1024},
              12: {**SHARED_BYTES_V5, "forward": 1024 + 2 * 97 * 1024, "dq": 1024 + 2 * 97 * 1024},
              13: SHARED_BYTES_V3}[version][
                  "kv" if which == "dqkv" else which]
    if which == "dq" and DQ_SMEM_A and version == 7 and not BWD_SPEC:
        shared = 1024 + 2 * 66 * 1024
    if which == "kv" and KV_SMEM_A and version == 7 and not BWD_SPEC:
        shared = 1024 + 2 * 66 * 1024
    if which == "forward" and FWD_DEEP and version == 7:
        shared = 1024 + 2 * 66 * 1024
    return "attention_long_" + which, code, dict(work_items=items, shared_bytes=shared, head_width=64,
                                                  sequence=t, head_batch=bh, recipe=recipe, pieces=pieces,
                                                  schedule_version=version,
                                                  **(dict(deterministic_dq_target_sms=FLASH_TARGET_SMS,
                                                          deterministic_dq_partitions=flash_splits,
                                                          backward_arithmetic="flash-bf16-deterministic-v1")
                                                     if version == 13 else {}),
                                                  outputs=outputs, reads=list(dict.fromkeys(reads)))


def defines(recipe, version=DEFAULT_VERSION):
    return (("#define PFLATLONG_O32 1\n" if RECIPES[recipe][1] else "")
            + ("#define PFLATLONG_WGMMA 1\n" if version in WGMMA_VERSIONS else "")
            + ("#define PFLATLONG_FTZ 1\n#define PFLATLONG_DEFER_GSCALE 1\n" if version in (7, 8, 9, 10, 11, 12) else ""))


def backward_phases(version, merged=False):
    """Phase order installed at the first gradient product (merged: ops.attention_rules merged_backward)."""
    if (MERGE_BWD or merged) and version == 7:
        return ("delta6", "dqkv")
    return ("delta", "kv", "dq") if version in STORED_DS_VERSIONS + FUSED_DQ_VERSIONS else ("dq", "kv")
