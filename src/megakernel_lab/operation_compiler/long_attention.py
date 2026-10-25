"""P7-C: long-context BF16-boundary training attention inside saved-body loops (kernels/attention_long.cuh).

The P-OPS4-E tape contract (ops/loop_attention.py) admitted only short contexts (T <= 128, the P-OPS3-B body keeps
whole rows on chip). Longer attention in a loop therefore kept the ordinary lowering: FP32 score/probability
matrices, an FP32 SIMT batched product per score/value/gradient product and the probabilities saved on the loop
tape. This rule admits the same structural groups (BF16-exact query/key/value/output-gradient sources, additive
FP32 mask, captured scale, consumers that round the outputs to BF16) at head width 64 and T % 128 == 0, with the
same tape statistics (FP32 O and row log-sum-exp) and the same rewritten graph; only the installed body differs.
Arithmetic and the exact-zero tile guard are documented in the header. Selection uses shapes, dtypes and layouts.
"""
from __future__ import annotations

from contextlib import contextmanager

from . import short_attention
from .short_attention import Refused

HEADER = "attention_long.cuh"
HEAD_WIDTHS = (64,)
BLOCK = 128
MAX_SEQUENCE = 4096
SOURCE = "P7-C long attention (kernels/attention_long.cuh)"
FORWARD_SHARED = 2 * (128 + 64 + 64) * 72 + 4 * (128 + 16 + 8 * 64)
BACKWARD_SHARED = 2 * (2 * 128 + 2 * 64) * 72 + 4 * (4 * 128 + 16)
# Resident variant (T <= RES_MAX): the streamed head operands stay in shared memory (kernels/attention_long.cuh).
RES_MAX = 512
RES_FORWARD_SHARED = 2 * (128 + 2 * RES_MAX) * 72 + 4 * (128 + RES_MAX // 64 + 16)
RES_BACKWARD_SHARED = 2 * (256 + 2 * RES_MAX) * 72 + 4 * (3 * RES_MAX + 128 + 16)
RESIDENT = True     # module setting; False keeps the streaming items at every T
# P8-B (default, qualified 12/12 on the primary): 'v2' installs the pipelined single-pass resident items (kernels/attention_long.cuh, *_item2<NS>) for T <=
# RES_MAX; 'v1' keeps the P7-C resident items. One BF16 piece (declared 'bf16_operands' graphs) always uses v2.
ENGINE = 'v2'
V2_FORWARD_SHARED = 2 * (128 + 2 * RES_MAX) * 72 + 4 * (128 + RES_MAX // 64 + RES_MAX)
V2_BACKWARD_SHARED = 2 * (256 + 2 * RES_MAX) * 72 + 4 * (3 * RES_MAX + 128 + RES_MAX // 64 + 16)
# P9-B (kernels/attention_p9.cuh): one-piece resident items with the per-tile arithmetic of v2 (bitwise equal outputs):
# CTA-live tile copies, item tables staged in shared memory, key/query block norm tables for the exact-zero guard,
# lean mma.sync tiles; backward publishes BF16 dS^T from the key/value items and computes dQ = dS K in separate
# items (no query-side recomputation). 'v2' keeps P8-B's items.
P9_ENGINE = 'v4'
P9_FWD_PAIRS = False  # P9-B forward v7: paired 128-query blocks per item (kernels/attention_p9.cuh forward_phase7)
P9_BWD_V8 = False     # P9-B backward v8: no prologue tables / grid sync (kernels/attention_p9.cuh backward_phase8)
P9_WRAPPER = True    # out-of-line constant-specialized wrapper per attention shape (see emit)
P9_HEADER = "attention_p9.cuh"
# Exact macros defined before the header. Approximate exp is selected only through emit(fast_exp=True),
# declared in the captured precision policy; it is carried in Args rather than a translation-unit macro.
# Diagnostic macros (e.g. "P9A_NO_KNORM", "P9A_TRACE") are added to this tuple.
# P9A_SKIP_RESCALE (exact; gpu-68 7.600 vs 7.612 ms over 3 rounds): forward tiles skip the rescale multiply when no
# row maximum of the warp changed (factor exp(0) = 1).
# P9A_DQ_QUEUE (exact; gpu-89 6.892 vs 6.912 ms, all rounds lower): backward dQ items from a call-epoch work queue.
P9_DEFINES = ("P9A_SKIP_RESCALE", "P9A_DQ_QUEUE")
P9_DS_ELEMS = 64 * 512 * 512        # kernels/attention_p9.cuh P9A_DS_ELEMS (static dS^T scratch)
# kernels/attention_p9.cuh v4_forward_shared_bytes() / v4_backward_shared_bytes() (item tables after the operands,
# diagonal mask staging off)
P9_FORWARD_SHARED = 2 * (128 + 2 * RES_MAX) * 72 + 4 * 128 + 8 * 8 * (RES_MAX // 64) + 4 * (RES_MAX // 64) + 32
P9_BACKWARD_SHARED = max(2 * (256 + 2 * RES_MAX) * 72 + 4 * 3 * RES_MAX + 1152, 2 * (128 * 72 + 128 * 136) * 2)


def admits(hd, t):
    return hd in HEAD_WIDTHS and t > short_attention.MAX_SEQUENCE and t % BLOCK == 0 and t <= MAX_SEQUENCE


def admit(groups, graph_args, static_layouts, protected):
    """({label: plan}, refusals) for groups the short rule refuses by context length only."""
    graph = short_attention._Graph(*graph_args)
    plans, refusals = {}, []
    for group in groups:
        label = group["label"]
        try:
            if not group.get("backward"):
                raise Refused("forward-only attention keeps the existing lowering")
            bh, t, hd = graph.shape(group["q"])
            if graph.shape(group["kt"]) != (bh, hd, t) or graph.shape(group["v"]) != (bh, t, hd):
                raise Refused("self-attention with equal key/value head width required")
            if not admits(hd, t):
                raise Refused(f"head width {hd} / context {t} outside long rule ({HEAD_WIDTHS}, T % {BLOCK} == 0, "
                              f"{short_attention.MAX_SEQUENCE} < T <= {MAX_SEQUENCE})")
            members = set(group["skip"])
            views = {group[k] for k in ("kt", "k_view", "qt_view", "vt_view") if group.get(k)}
            # P8-B: the declared 'bf16_operands' roundings (both present, attention_lower.plan) are implemented
            # exactly by one BF16 piece per probability / score-gradient operand; otherwise the exact 3-piece split.
            splits = 1 if group.get("p_round") and group.get("ds_round") else 3
            if splits == 1 and t > RES_MAX:
                raise Refused("one-piece attention exceeds resident capacity; retain captured primitive operations")
            members |= set(group.get("p_round_nodes", [])) | set(group.get("ds_round_nodes", []))
            if group.get("score_alias"):
                members.add(group["score_alias"])
            plan = dict(group=group, bh=bh, t=t, hd=hd, splits=splits, group_names=members | views)
            plans[label] = dict(short_attention.finalize(plan, graph_args, static_layouts, protected), kernel="long")
        except Refused as exc:
            refusals.append({"group": label, "reason": "long attention: " + str(exc)})
    return plans, refusals


def backward_list(blocks):
    """Per-head-batch work list (kind << 7 | block) in decreasing estimated causal cost (stable)."""
    entries = [(16 * (blocks - b), b) for b in range(blocks)] + [(10 * (b + 1), 128 | b) for b in range(blocks)]
    entries.sort(key=lambda e: -e[0])
    return [code for _, code in entries]


_SLOTS = {}
MT_SLOTS = 32   # kernels/attention_long.cuh plong::MT_SLOTS
# P9-B: one mask-tile table per distinct mask operand (identity, strides, T) for the whole update, built by the first
# forward phase emitted for it (layer order = execution order of the first body iteration; with count 0 no attention
# phase runs at all). Slots are taken from the top of the slot range, apart from the per-phase norm/counter slots.
P9_SHARED_MASK = True
_MASK_SLOTS = {}
_REGION_NAMESPACE = None
_NEXT_REGION_NAMESPACE = 0


@contextmanager
def program_state(*, shared_masks=True):
    """Program-local table allocation; caller serializes the entire top-level lowering.

    Conditional region composition disables shared-mask builders: an earlier
    builder may be in an unexecuted branch. Individual emission also requires
    explicit immutable-mask provenance before sharing its table.
    """
    if type(shared_masks) is not bool:
        raise TypeError("shared_masks must be bool")
    global _SLOTS, _MASK_SLOTS, P9_SHARED_MASK, _REGION_NAMESPACE, _NEXT_REGION_NAMESPACE
    previous = _SLOTS, _MASK_SLOTS, P9_SHARED_MASK, _REGION_NAMESPACE, _NEXT_REGION_NAMESPACE
    _SLOTS, _MASK_SLOTS = {}, {}
    P9_SHARED_MASK = previous[2] and shared_masks
    _REGION_NAMESPACE, _NEXT_REGION_NAMESPACE = None, 0
    try:
        yield
    finally:
        _SLOTS, _MASK_SLOTS, P9_SHARED_MASK, _REGION_NAMESPACE, _NEXT_REGION_NAMESPACE = previous


@contextmanager
def region_namespace():
    """Deterministic call-order identity; local FX names are not outer bindings.

    Forward and backward regions conservatively build separate mask tables.
    Sharing across them requires an explicit outer-binding identity, not equal
    local argument names or equal user-provided region labels.
    """
    global _REGION_NAMESPACE, _NEXT_REGION_NAMESPACE
    previous = _REGION_NAMESPACE
    _REGION_NAMESPACE = _NEXT_REGION_NAMESPACE
    _NEXT_REGION_NAMESPACE += 1
    try:
        yield _REGION_NAMESPACE
    finally:
        _REGION_NAMESPACE = previous


@contextmanager
def allocation_transaction():
    """Discard allocations from a refused attempt; retain prior successful regions."""
    global _NEXT_REGION_NAMESPACE
    slots, masks, next_namespace = dict(_SLOTS), dict(_MASK_SLOTS), _NEXT_REGION_NAMESPACE
    try:
        yield
    except BaseException:
        _SLOTS.clear()
        _SLOTS.update(slots)
        _MASK_SLOTS.clear()
        _MASK_SLOTS.update(masks)
        _NEXT_REGION_NAMESPACE = next_namespace
        raise


def _free_slots():
    used = set(_SLOTS.values()) | {slot for slot, _ in _MASK_SLOTS.values()}
    return [slot for slot in range(MT_SLOTS) if slot not in used]


def mask_table(key, which, label):
    """An immutable mask's shared table, with a guaranteed distinct program-local slot."""
    key = (_REGION_NAMESPACE, key)
    if key not in _MASK_SLOTS:
        if which != "forward":
            return None
        free = _free_slots()
        if not free:
            raise Refused("long-attention table capacity exceeded; shared slots cannot alias")
        _MASK_SLOTS[key] = (free[-1], label)
    slot, builder = _MASK_SLOTS[key]
    return slot, which == "forward" and label == builder


def mask_slot(label, which):
    """Stable per-phase slot, never modulo-wrapped into another live table."""
    key = (_REGION_NAMESPACE, label, which)
    if key not in _SLOTS:
        free = _free_slots()
        if not free:
            raise Refused("long-attention phase capacity exceeded; slots cannot wrap")
        _SLOTS[key] = free[0]
    return _SLOTS[key]


def uses_p9(plan, mask_strides):
    """P9-B: whether this attention group runs the v4/v6 p9 items (the condition emit applies)."""
    bh, t = plan["bh"], plan["t"]
    splits = int(plan.get("splits", 3))
    v2 = t <= RES_MAX and (ENGINE == 'v2' or splits == 1)
    return (P9_ENGINE == 'v4' and v2 and splits == 1 and mask_strides is not None and t % 128 == 0
            and bh * (t // 64) <= 4096 and bh * t * t <= P9_DS_ELEMS)


def emit(plan, layouts, ident, mask_strides, *, which, aux, fast_exp=False):
    if type(fast_exp) is not bool:
        raise ValueError("fast_exp must be an explicit bool from the precision policy")
    group, bh, t = plan["group"], plan["bh"], plan["t"]
    s, o = plan["in_strides"], plan["out_strides"]
    mode = group["scale_mode"]
    scale = float(group["scale"]) if mode else 1.0

    def l4(x):
        return "{" + f"{x[0]}LL, {x[1]}LL, {x[2]}LL" + "}"
    fields = [
        f"A.q = {ident(plan['sources']['q'])}; A.lq = pshort::Layout4{l4(s['q'])};",
        f"A.k = {ident(plan['sources']['k'])}; A.lk = pshort::Layout4{l4(s['k'])};",
        f"A.v = {ident(plan['sources']['v'])}; A.lv = pshort::Layout4{l4(s['v'])};",
        f"A.heads = {plan['heads']}; A.t = {t}; A.scale = {repr(scale)}f; A.scale_mode = {mode};",
        f"A.o32 = {ident(aux['o32'])}; A.lse = {ident(aux['lse'])};",
    ]
    if mask_strides is not None:
        fields.append(f"A.mask = {ident(group['mask'])}; A.m_b = {mask_strides[0]}LL; A.m_i = {mask_strides[1]}LL; "
                      f"A.m_j = {mask_strides[2]}LL; A.has_mask = 1;")
    else:
        fields.append("A.mask = nullptr; A.m_b = A.m_i = A.m_j = 0; A.has_mask = 0;")
    blocks = t // BLOCK
    splits = int(plan.get("splits", 3))
    if splits not in (1, 3):
        raise ValueError("long attention implements one (declared BF16 operands) or three (exact) pieces")
    v2 = t <= RES_MAX and (ENGINE == 'v2' or splits == 1)
    if splits == 1 and not v2:
        raise ValueError("one-piece long attention requires the resident v2 items (T <= RES_MAX)")
    res = RESIDENT and t <= RES_MAX
    tpl = "<true>" if res else ""
    p9 = uses_p9(plan, mask_strides)
    fields.append(f"A.practical_exp = {str(bool(fast_exp and p9)).lower()};")
    if which == "forward":
        fields.append(f"A.o = {ident(plan['finals']['o'])}; A.lo = pshort::Layout4{l4(o['o'])};")
        if p9:
            body = f"p9att::forward_phase4<0>(A, {blocks}, {bh}, smem, {mask_slot(plan['label'], which)});"
        else:
            body = (f"plong::forward_phase2<{splits}>(A, {blocks}, {bh}, smem, {mask_slot(plan['label'], which)});" if v2 else
                    f"plong::forward_phase{tpl}(A, {blocks}, {bh}, smem);")
        items, shared = blocks * bh, (P9_FORWARD_SHARED if p9 else V2_FORWARD_SHARED if v2 else
                                      (RES_FORWARD_SHARED if res else FORWARD_SHARED))
        outputs = {plan["finals"]["o"]: None, aux["o32"]: ((bh, t, plan["hd"]), "float32"), aux["lse"]: ((bh, t), "float32")}
    else:
        fields.append(f"A.dout = {ident(plan['sources']['do'])}; A.ld = pshort::Layout4{l4(s['do'])};")
        fields.append(f"A.dq = {ident(plan['finals']['dq'])}; A.ldq = pshort::Layout4{l4(o['dq'])};")
        fields.append(f"A.dk = {ident(plan['finals']['dkt'])}; A.ldk = pshort::Layout4{l4(o['dkt'])};")
        fields.append(f"A.dv = {ident(plan['finals']['dv'])}; A.ldv = pshort::Layout4{l4(o['dv'])};")
        if plan.get("p9_delta_ready"):
            fields.append("A.delta_ready = 1;")      # P9-B: the dO product's epilogue wrote this call's delta table
        if plan.get("p9_qnorm_ready"):
            fields.append("A.qnorm_ready = 1;")      # P9-B: ... and its q-norm block maxima
        entries = backward_list(blocks)
        if p9:
            body = f"p9att::backward_phase4<0>(A, {bh}, smem, {mask_slot(plan['label'], which)}, p9att::p9_dsbuf, 0);"
            items, shared = 2 * blocks * bh, P9_BACKWARD_SHARED
        else:
            call = (f"plong::backward_phase2<{splits}>" if v2 else f"plong::backward_phase{tpl}")
            body = ("const unsigned char plong_list[] = {" + ", ".join(str(e) for e in entries) + "};\n"
                    f"{call}(A, plong_list, {len(entries)}, {bh}, smem" + (f", {mask_slot(plan['label'], which)});" if v2 else ");"))
            items, shared = len(entries) * bh, (V2_BACKWARD_SHARED if v2 else (RES_BACKWARD_SHARED if res else BACKWARD_SHARED))
        outputs = {plan["finals"]["dq"]: None, plan["finals"]["dkt"]: None, plan["finals"]["dv"]: None}
    code = "pshort::Args A{};\n" + "\n".join(fields) + "\n" + body
    helper = None
    if p9 and P9_WRAPPER:
        # P9-B: one out-of-line wrapper per (direction, shape constants): the operand pointers are its parameters,
        # every other Args field is a compile-time constant inside it (the inlined items fold them), and the item
        # bodies get the wrapper's own register allocation instead of the single kernel function's.
        names = {"q": "const __nv_bfloat16*", "k": "const __nv_bfloat16*", "v": "const __nv_bfloat16*",
                 "dout": "const __nv_bfloat16*", "o": "__nv_bfloat16*", "dq": "__nv_bfloat16*",
                 "dk": "__nv_bfloat16*", "dv": "__nv_bfloat16*", "o32": "float*", "lse": "float*",
                 "mask": "const float*"}
        pointers = {"q": ident(plan['sources']['q']), "k": ident(plan['sources']['k']),
                    "v": ident(plan['sources']['v']), "o32": ident(aux['o32']), "lse": ident(aux['lse']),
                    "mask": ident(group['mask'])}
        if which == "forward":
            pointers["o"] = ident(plan['finals']['o'])
        else:
            pointers.update(dout=ident(plan['sources']['do']), dq=ident(plan['finals']['dq']),
                            dk=ident(plan['finals']['dkt']), dv=ident(plan['finals']['dv']))
        wfields = []
        for line in fields:
            for key, value in pointers.items():
                line = line.replace(f"A.{key} = {value};", f"A.{key} = p_{key};")
            wfields.append(line)
        table = mask_table((pointers["mask"], tuple(mask_strides), t), which, plan["label"]) if P9_SHARED_MASK and plan.get("immutable_mask", False) else None
        pre, mslot = (0, -1) if table is None else (0 if table[1] else 1, table[0])
        if which == "forward":
            wbody = f"p9att::forward_phase{7 if P9_FWD_PAIRS else 4}<{pre}>(A, {blocks}, {bh}, smem, slot, {mslot});"
        else:
            wbody = (f"p9att::backward_phase{8 if (P9_BWD_V8 and pre == 1) else 6}<{pre}>(A, {bh}, smem, slot, "
                     f"p9att::p9_dsbuf, 0, {mslot});")
        import hashlib as _h
        key = _h.sha256(("\n".join(wfields) + wbody).encode()).hexdigest()[:12]
        fn = f"p9w_{which}_{key}"
        params = ", ".join(f"{names[k]} p_{k}" for k in pointers) + ", float* smem, int slot"
        helper = (f"__device__ __noinline__ void {fn}({params}) {{\npshort::Args A{{}};\n" + "\n".join(wfields)
                  + "\n" + wbody + "\n}\n")
        code = f"{fn}(" + ", ".join(pointers.values()) + f", smem, {mask_slot(plan['label'], which)});"
    return "attention_long_" + which, code, dict(work_items=items, shared_bytes=shared, head_width=plan["hd"],
                                                 sequence=t, head_batch=bh, splits=splits, outputs=outputs,
                                                 resident=res, engine='p9' if p9 else 'v2' if v2 else 'v1',
                                                 exponential_policy='approximate_ftz' if (fast_exp and p9) else 'accurate',
                                                 approximate_exp_requested=fast_exp,
                                                 shared_mask_admitted=bool(p9 and P9_WRAPPER and P9_SHARED_MASK
                                                                           and plan.get("immutable_mask", False)),
                                                 **({"helper": helper} if helper else {}))


__all__ = ["HEADER", "admit", "admits", "backward_list", "emit", "program_state"]
