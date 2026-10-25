"""Operation-level fused attention lowering (P-OPS-D).

Recognizes the *semantics* of a scaled, additively masked softmax attention
subgraph in a captured ATen training graph, independent of any model:

    S0 = bmm(Q, KT)            KT any layout of K^T (inner K stride must be 1)
    S1 = mul|div(S0, c)        optional, c a captured Python scalar
    S2 = add(S1, M)            optional, M a tensor broadcastable to S2
    P  = _softmax(S2, -1)
    O  = bmm(P, V)

and, when present, the reverse-mode operations PyTorch derived for it:

    dV  = bmm(transpose(P), dO)       dP = bmm(dO, VT)
    dS2 = _softmax_backward_data(dP, P, -1, float32)
    dS0 = mul|div(dS2, c)             (identity when S1 is absent)
    dKT = bmm(QT, dS0)                dQ = bmm(dS0, K)

Every intermediate (S0, S1, S2, P, dP, dS2, dS0 and transpose views of P) must be
used only inside the group, and no group value may be a returned or committed
output. The original path produces O and row log-sum-exp, then delta and backward
phases. The P-OPS2-C long-context path stores scores and dS for training;
forward-only inference does not retain those matrices. Unmatched dtype, head width or
layout leaves the ordinary operation lowering in place (recorded as a refusal).
This file never inspects parameter names or model structure beyond the group.

Imported from P-OPS-D commit 775505d79b1d7debc9b2865e092e30405eee8455.
The integration adds scheduler adapters and conservative admission checks; the
FP32 attention arithmetic lives in the attributed, pinned vendor header.
"""
from __future__ import annotations

import math
from contextvars import ContextVar
from functools import wraps

HEAD_WIDTHS = (32, 64)
MAX_SHARED_BYTES = 232448  # H100 opt-in dynamic shared memory per block
MAX_SEQUENCE = 4096  # per-item block flags hold at most 64 key/query blocks
SOURCE_COMMIT = "775505d79b1d7debc9b2865e092e30405eee8455"
SOURCE_COMMIT_V2 = "e7fe4b165f186fa1e86df3c49df6bbfb15e213f2"
HEADER = "attention_fp32.cuh"
# P-OPS2-C execution variant (ops/vendor/attention_fp32_v2.cuh). Selection is a
# pure function of the recognized group's shape; semantics are identical to v1.
HEADER_V2 = "attention_fp32_v2.cuh"
V2_HEAD_WIDTHS = (64,)
V2_MIN_SEQUENCE = 1024  # donor final: gain at 1024/2048; forced 512 regresses
# Mirrors FWD_FLOATS / KV_FLOATS / DQ_FLOATS in the header (row strides 68 / 132).
V2_SHARED_BYTES = {"forward": 4 * (128 * 68 + 64 * 68 + 64 * 64 + 64 * 132 + 4 * 128 + 64 + 4 * 64),
                   "backward": 4 * (2 * 128 * 68 + 2 * 32 * 68 + 64 + 2 * 32 * 132 + 128),
                   "backward_stored": 4 * (128 * 68 + 3 * 32 * 68 + 64 + 2 * 32 * 132 + 256),
                   "dq": 4 * (2 * (128 * 68 + 64 * 64) + 4 * 64 + 3 * 64)}


def select_variant(head_width, sequence, head_batch, enabled=True):
    """Shape-based execution selection for one recognized attention group.

    enabled: False keeps v1; True applies the shape rule; "force" (qualification
    fixtures only) selects v2 wherever its kernels admit the shape.
    """
    admitted = head_width in V2_HEAD_WIDTHS and 1 <= sequence <= MAX_SEQUENCE
    if enabled == "force":
        return "v2" if admitted else "v1"
    if enabled and admitted and sequence >= V2_MIN_SEQUENCE:
        return "v2"
    return "v1"
_EXCLUDED = ContextVar("latch_attention_excluded_groups", default=None)


class AttentionRefused(Exception):
    def __init__(self, group, reason):
        self.group = group
        self.reason = reason
        super().__init__(f"attention group {group}: {reason}")


def excluded_groups():
    return tuple((_EXCLUDED.get() or {}).keys())


def retry_lowering(function):
    """Discard provisional code after late view/layout refusal, then re-lower.

    Context-local state gives each nested region a separate exclusion set. Every
    retry excludes a new finite graph group; scientific/numerical errors are not
    caught. This is a compiler fallback, never a retry of device execution.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        excluded = {}
        token = _EXCLUDED.set(excluded)
        try:
            while True:
                try:
                    # A late refusal discards generated code, so it must also
                    # discard that attempt's shared-table builder/slot claims.
                    # Earlier successful regions remain in the enclosing
                    # whole-program allocation scope.
                    from .long_attention import allocation_transaction
                    with allocation_transaction():
                        result = function(*args, **kwargs)
                    if excluded:
                        result.manifest.setdefault("attention_refusals", []).extend(
                            {"group": group, "reason": reason} for group, reason in excluded.items())
                    return result
                except AttentionRefused as error:
                    if error.group in excluded:
                        raise RuntimeError("attention exclusion failed to make lowering progress") from error
                    excluded[error.group] = error.reason
        finally:
            _EXCLUDED.reset(token)
    return wrapped


def _base(op, field):
    target = str(field(op, "target", field(op, "op")))
    return target.removeprefix("torch.ops.").removeprefix("aten.").split(".")[0], target


def _dkt_restore_reason(group, operations, values, names, *, ref, field, output_name):
    """Check only aliases downstream of the one output whose layout changes.

    Native backward produces contiguous dK, while the original bmm produces
    contiguous dKT. Their logical values agree after transpose, but a later view
    may be legal for only the original layout. Follow the alias chain with the
    same reshape-stride rule as lowering; restore dKT only when such a view needs
    it. Arithmetic consumers terminate a chain because they have fresh storage.
    """
    from .lower import _contiguous, _reshape_strides
    if not group.get("backward"):
        return None
    initial = tuple(field(values[group["dkt"]], "shape"))
    if len(initial) != 3:
        return None  # The emitter refuses malformed rank before installation.
    batch, width, length = initial
    aliases = {group["dkt"]: (initial, (width * length, 1, width))}
    view_bases = {"detach", "alias", "t", "transpose", "permute", "view", "_unsafe_view",
                  "reshape", "expand", "squeeze", "unsqueeze"}
    for operation in operations:
        base, _ = _base(operation, field)
        args = list(field(operation, "args", []))
        source = ref(args[0], names) if args else None
        if base not in view_bases or source not in aliases:
            continue
        name = output_name(operation)
        old_shape, old_strides = aliases[source]
        shape = tuple(field(values[name], "shape"))
        strides = list(old_strides)
        if base == "t":
            strides.reverse()
        elif base == "transpose":
            left, right = int(args[1]) % len(strides), int(args[2]) % len(strides)
            strides[left], strides[right] = strides[right], strides[left]
        elif base == "permute":
            strides = [old_strides[int(dim)] for dim in args[1]]
        elif base == "expand":
            padded = (1,) * (len(shape) - len(old_shape)) + old_shape
            padded_strides = (0,) * (len(shape) - len(old_shape)) + old_strides
            strides = [0 if old == 1 and new != 1 else stride
                       for old, new, stride in zip(padded, shape, padded_strides)]
        elif base == "unsqueeze":
            dim = int(args[1]) % (len(old_shape) + 1)
            strides.insert(dim, old_strides[dim] * old_shape[dim] if dim < len(old_shape) else 1)
        elif base == "squeeze":
            dims = range(len(old_shape)) if len(args) == 1 else (
                args[1] if isinstance(args[1], (list, tuple)) else [args[1]])
            dims = {int(dim) % len(old_shape) for dim in dims}
            strides = [stride for dim, (extent, stride) in enumerate(zip(old_shape, old_strides))
                       if dim not in dims or extent != 1]
        elif base in {"view", "_unsafe_view", "reshape"}:
            contiguous = all(extent == 1 or stride == expected
                             for extent, stride, expected in zip(old_shape, old_strides, _contiguous(old_shape)))
            strides = _contiguous(shape) if contiguous else _reshape_strides(old_shape, old_strides, shape)
            if strides is None:
                return f"{name}: downstream view requires original contiguous dKT layout"
        aliases[name] = (shape, tuple(strides))
    return None


def plan(operations, values, names, *, ref, field, output_name, protected, exclude=(), allow_rounding=False,
         masks=False):
    """Structural matching only. Returns (groups, refusals).

    groups: list of dicts with keys forward (op names), backward (op names or
    None), skip (all removed value names), anchor_forward, anchor_backward,
    and tensor value names q, kt, v, mask, o, scale, scale_mode, do, dq, dkt, dv,
    plus the auxiliary view names whose layouts must be validated later.

    P8-B: groups carrying the declared 'bf16_operands' roundings (an exact BF16 rounding of P as the value-product
    operand and of the scaled score cotangent, ops/precision.py) are admitted only with allow_rounding=True (the
    loop tape rules, which implement them with one BF16 piece); every other caller refuses them, so their graphs
    keep the ordinary lowering of the casts as written.

    masks=True (release step 09, ops/padding_masks.py): also admit PyTorch's boolean-mask softmax
    (``_safe_softmax``, group key ``safe``) and absorb the capture's head expansion of a mask (group key
    ``mask_lead``: the mask is then the per-example source). Only callers whose bodies implement or refuse these
    keys pass it; the default leaves every group and refusal exactly as before.
    """
    by_name, order, consumers = {}, {}, {}
    for index, op in enumerate(operations):
        name = output_name(op)
        by_name[name] = op
        order[name] = index
        args = list(field(op, "args", []))
        kwargs = dict(field(op, "kwargs", {}))
        seen = set()

        def walk(item):
            r = ref(item, names)
            if r is not None:
                if r not in seen:
                    seen.add(r)
                    consumers.setdefault(r, []).append(name)
                return
            if isinstance(item, (list, tuple)):
                for x in item:
                    walk(x)
            elif isinstance(item, dict):
                for x in item.values():
                    walk(x)
        walk(args)
        walk(kwargs)

    def base(name):
        return _base(by_name[name], field)[0] if name in by_name else None

    def args_of(name):
        return list(field(by_name[name], "args", []))

    def kwargs_of(name):
        return dict(field(by_name[name], "kwargs", {}))

    def uses(name):
        return consumers.get(name, [])

    def only_use(name):
        users = uses(name)
        return users[0] if len(users) == 1 else None

    def is_scalar(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

    def is_transpose12(name, source):
        if base(name) != "transpose":
            return False
        a = args_of(name)
        if ref(a[0], names) != source:
            return False
        rank = len(field(values[source], "shape"))
        return rank == 3 and {int(a[1]) % 3, int(a[2]) % 3} == {1, 2}

    def cast_to(name, dtype):
        if base(name) != "_to_copy":
            return False
        kw = kwargs_of(name)
        if set(kw) - {"dtype", "layout", "device", "pin_memory", "memory_format"}:
            return False
        return str(kw.get("dtype", "")).removeprefix("torch.") == dtype

    def round_trip(name):
        """P8-B: the FP32 value of an exact BF16 rounding of `name` (name -> to(bf16) -> to(f32)), each node with
        one consumer, or None. Returns (rounded value name, [the two cast nodes])."""
        users = uses(name)
        if len(users) != 1 or not cast_to(users[0], "bfloat16"):
            return None
        narrow = users[0]
        wide_users = uses(narrow)
        if len(wide_users) != 1 or not cast_to(wide_users[0], "float32"):
            return None
        return wide_users[0], [narrow, wide_users[0]]

    groups, refusals = [], []
    for s0 in [output_name(op) for op in operations if _base(op, field)[0] == "bmm"]:
        label = s0
        if label in exclude:
            continue
        try:
            a = args_of(s0)
            q, kt = ref(a[0], names), ref(a[1], names)
            chain = [s0]
            current = s0
            scale, scale_mode, mask = None, 0, None
            nxt = only_use(current)
            score_alias = None
            if (nxt is not None and base(nxt) in ("view", "alias", "_unsafe_view", "clone")
                    and ref(args_of(nxt)[0], names) == current
                    and tuple(field(values[nxt], "shape")) == tuple(field(values[current], "shape"))):
                # P8-B: the value-identity of a cotangent-rounding score (precision 'bf16_operands').
                score_alias = nxt
                chain.append(nxt)
                current = nxt
                nxt = only_use(current)
            if nxt is not None and base(nxt) in ("mul", "div"):
                b = args_of(nxt)
                if ref(b[0], names) != current or not is_scalar(b[1]) or kwargs_of(nxt).get("rounding_mode") is not None:
                    continue  # Not a scalar-scaled score product: no attention semantics claimed.
                scale, scale_mode = float(b[1]), 1 if base(nxt) == "mul" else 2
                chain.append(nxt)
                current = nxt
                nxt = only_use(current)
            if nxt is not None and base(nxt) == "add":
                b = args_of(nxt)
                alpha = kwargs_of(nxt).get("alpha", b[2] if len(b) > 2 else 1)
                left, right = ref(b[0], names), ref(b[1], names)
                if alpha != 1 or None in (left, right) or current not in (left, right) or left == right:
                    continue  # Not one additive tensor term: no attention semantics claimed.
                mask = right if left == current else left
                chain.append(nxt)
                current = nxt
                nxt = only_use(current)
            mask_lead, mask_members, mask_view = None, [], None
            # A CSE pass (P6-D) can merge identical per-layer mask expansions: one view then feeds several score
            # additions. Such a shared chain is assigned to one group after matching (below).
            if masks and mask is not None and chain[-1] in uses(mask) and all(base(u) == "add" for u in uses(mask)):
                from . import padding_masks
                seen = padding_masks.look_through(
                    mask, base=base, args_of=lambda n: [ref(x, names) for x in args_of(n)] if n in by_name else [None],
                    uses=uses, shape_of=lambda n: field(values[n], "shape"))
                if seen is not None:
                    mask_view = mask
                    mask, mask_lead, mask_members = seen[0], seen[1], seen[2]
                    if len(uses(mask_view)) > 1:
                        mask_members = []  # claimed after matching when every user is a matched group
            safe = masks and nxt is not None and base(nxt) == "_safe_softmax"
            if nxt is None or base(nxt) not in ("_softmax", "_safe_softmax") or (base(nxt) != "_softmax" and not safe):
                continue  # Not an attention score product.
            b = args_of(nxt)
            if safe:
                dtype = b[2] if len(b) > 2 else kwargs_of(nxt).get("dtype")
                if ref(b[0], names) != current or int(b[1]) % 3 != 2 or dtype is not None:
                    raise AttentionRefused(label, "safe softmax is not over the key axis in FP32")
            elif ref(b[0], names) != current or int(b[1]) % 3 != 2 or bool(b[2]):
                raise AttentionRefused(label, "softmax is not over the key axis in FP32")
            p = nxt
            chain.append(p)
            p_users = uses(p)
            # P8-B (precision 'bf16_operands'): the value product reads an exact BF16 rounding of P (forward value
            # only; its derivative is the identity), and the transposed rounded P feeds dV.
            pv = p
            p_round = None
            rounded = [u for u in p_users if cast_to(u, "bfloat16")]
            if len(rounded) == 1:
                narrow = rounded[0]
                wide = uses(narrow)
                if len(wide) == 1 and cast_to(wide[0], "float32"):
                    pv, p_round = wide[0], [narrow, wide[0]]
            pv_users = uses(pv)
            forward_out = [u for u in pv_users if base(u) == "bmm" and ref(args_of(u)[0], names) == pv]
            if len(forward_out) != 1:
                raise AttentionRefused(label, "probabilities do not feed exactly one value product")
            o = forward_out[0]
            v = ref(args_of(o)[1], names)
            others = [u for u in p_users if u != o and (p_round is None or u != p_round[0])]
            if p_round is not None:
                others += [u for u in pv_users if u != o]
                chain += p_round
            skip = set(chain) | {o} | set(mask_members)
            if (p_round is not None or score_alias is not None) and not allow_rounding:
                raise AttentionRefused(label, "BF16-operand attention roundings are implemented only by the loop "
                                              "tape attention rules")
            scores = list(chain[:chain.index(p)])
            group = dict(label=label, q=q, kt=kt, v=v, mask=mask, o=o, scale=scale, scale_mode=scale_mode,
                         forward=list(chain) + [o], backward=None, anchor_forward=o,
                         p=p, pv=pv, scores=scores, p_round=p_round is not None, p_round_nodes=p_round or [],
                         score_alias=score_alias, ds_round_nodes=[])
            if safe or mask_lead is not None:
                group.update(safe=safe, mask_lead=mask_lead, mask_members=list(mask_members), mask_view=mask_view,
                             mask_add=chain[chain.index(p) - 1])
            if others:
                transposes = [u for u in others if is_transpose12(u, pv)]
                sbw = [u for u in others if base(u) == "_softmax_backward_data"]
                if len(transposes) != 1 or len(sbw) != 1 or len(others) != 2:
                    raise AttentionRefused(label, "probabilities have consumers outside the derived backward")
                pt, ds2 = transposes[0], sbw[0]
                dv_users = uses(pt)
                if len(dv_users) != 1 or base(dv_users[0]) != "bmm" or ref(args_of(dv_users[0])[0], names) != pt:
                    raise AttentionRefused(label, "transposed probabilities are not only the dV product input")
                dv = dv_users[0]
                do = ref(args_of(dv)[1], names)
                b = args_of(ds2)
                if (ref(b[1], names) != p or int(b[2]) % 3 != 2 or str(b[3]).removeprefix("torch.") != "float32"):
                    raise AttentionRefused(label, "softmax backward does not match the forward softmax")
                dp = ref(b[0], names)
                if base(dp) != "bmm" or uses(dp) != [ds2] or ref(args_of(dp)[0], names) != do:
                    raise AttentionRefused(label, "dP is not bmm(dO, V^T) used only by softmax backward")
                vt_view = ref(args_of(dp)[1], names)
                ds0 = ds2
                bwd = [pt, dv, dp, ds2]
                if scale_mode:
                    users = uses(ds2)
                    if len(users) != 1 or base(users[0]) != ("mul" if scale_mode == 1 else "div"):
                        raise AttentionRefused(label, "score-gradient scaling does not match the forward scale")
                    c = args_of(users[0])
                    if (ref(c[0], names) != ds2 or not is_scalar(c[1]) or float(c[1]) != scale
                            or kwargs_of(users[0]).get("rounding_mode") is not None):
                        raise AttentionRefused(label, "score-gradient scaling constant differs from the forward")
                    ds0 = users[0]
                    bwd.append(ds0)
                ds_round = round_trip(ds0)
                if ds_round is not None:
                    # P8-B: the cotangent rounding of the score (precision 'bf16_operands').
                    bwd += ds_round[1]
                    ds0 = ds_round[0]
                group["ds_round"] = ds_round is not None
                group["ds_round_nodes"] = ds_round[1] if ds_round is not None else []
                if group["p_round"] != group["ds_round"] or (group["ds_round"] and score_alias is None):
                    raise AttentionRefused(label, "BF16 operand roundings must cover both P and dS (declared "
                                                  "'bf16_operands' recipe) or neither")
                users = uses(ds0)
                dq = [u for u in users if base(u) == "bmm" and ref(args_of(u)[0], names) == ds0]
                dkt = [u for u in users if base(u) == "bmm" and ref(args_of(u)[1], names) == ds0]
                if len(users) != 2 or len(dq) != 1 or len(dkt) != 1 or dq[0] == dkt[0]:
                    raise AttentionRefused(label, "score gradient is not consumed by exactly dQ and dK^T products")
                dq, dkt = dq[0], dkt[0]
                bwd += [dq, dkt]
                group.update(backward=bwd, do=do, dq=dq, dkt=dkt, dv=dv, vt_view=vt_view,
                             k_view=ref(args_of(dq)[1], names), qt_view=ref(args_of(dkt)[0], names),
                             # A transpose(P) can precede the producer of dO.
                             # Every gradient product, unlike that view, reads
                             # dO (directly or through dS) and occurs after it.
                             anchor_backward=min((dv, dq, dkt), key=order.__getitem__),
                             last_backward=max(bwd, key=order.__getitem__))
                skip |= set(bwd)
            # Values that cease to exist must not escape the group.
            removed = (skip - {o, group.get("dq"), group.get("dkt"), group.get("dv")}) - {None}
            for name in removed:
                if name in protected:
                    raise AttentionRefused(label, f"{name} is returned or committed")
                outside = [u for u in uses(name) if u not in skip]
                if outside:
                    raise AttentionRefused(label, f"{name} is used outside the attention group by {outside}")
            for name in (o, group.get("dq"), group.get("dkt"), group.get("dv")):
                if name is not None and name in protected and name == group.get("dkt"):
                    raise AttentionRefused(label, "dK^T is a returned value; fused layout differs")
            group["skip"] = skip
            group["removed"] = removed
            group["restore_dkt_reason"] = _dkt_restore_reason(
                group, operations, values, names, ref=ref, field=field, output_name=output_name)
            generated = set(tables(group).values()) | {o + "__lse", o + "__delta"}
            generated.update(phase_root(group, kind) for kind in
                             ("attention_prepare", "attention_forward", "attention_delta", "attention_backward",
                              "attention_layout_restore", "attention_dq", "attention_ds", "attention_xs"))
            if group.get("dkt"):
                generated.add(group["dkt"] + "__dk")
            collisions = generated.intersection(names)
            if collisions:
                raise AttentionRefused(label, f"synthetic names already exist: {sorted(collisions)}")
            groups.append(group)
        except AttentionRefused as exc:
            refusals.append({"group": exc.group, "reason": exc.reason})
    if masks:
        _claim_shared_masks(groups, uses=lambda n: consumers.get(n, []), base=lambda n: _base(by_name[n], field)[0]
                            if n in by_name else None, args_of=lambda n: [ref(x, names) for x in
                            field(by_name[n], "args", [])] if n in by_name else [None], protected=protected)
    # Groups must be disjoint (a value cannot belong to two attentions).
    claimed = {}
    for g in groups:
        for name in g["skip"]:
            if name in claimed:
                raise AttentionRefused(g["label"], f"overlaps attention group {claimed[name]}")
            claimed[name] = g["label"]
    return groups, refusals


def _claim_shared_masks(groups, *, uses, base, args_of, protected):
    """Release step 09: a mask expansion shared by several groups (after CSE) stays an ordinary operation; every
    group reads the per-example source directly when all readers of the expansion are matched groups. Removing it
    with one group would fail that group's region check and break a reader a host later declines to fuse.
    Otherwise those groups keep reading the materialized expansion, as before."""
    shared = {}
    for group in groups:
        view = group.get("mask_view")
        if view is not None and len(uses(view)) > 1:
            shared.setdefault(view, []).append(group)
    for view, members in shared.items():
        if {g["mask_add"] for g in members} == set(uses(view)):
            for group in members:
                group["mask_shared"] = True
            continue
        for group in members:  # not every reader is fused: read the materialized expansion, as before
            group.update(mask=view, mask_lead=None, mask_members=[])


def _mask_strides(layout, bh, t, index_ok):
    shape, strides = tuple(layout.shape), tuple(layout.strides)
    if len(shape) > 3:
        if any(d != 1 for d in shape[:-3]):
            return None
        shape, strides = shape[-3:], strides[-3:]
    padded = (1,) * (3 - len(shape)) + shape
    pstrides = (0,) * (3 - len(shape)) + strides
    result = []
    for extent, stride, full in zip(padded, pstrides, (bh, t, t)):
        if extent == full:
            result.append(stride)
        elif extent == 1:
            result.append(0)
        else:
            return None
    return tuple(result)


def _head_layout(layout, bh, t, hd):
    """(batch stride, row stride) for a [BH][T][HD] view with unit inner stride."""
    if tuple(layout.shape) != (bh, t, hd):
        return None
    sb, sr, se = layout.strides
    if se != 1 and hd != 1:
        return None
    if any(x % 4 for x in (sb, sr, layout.offset)):
        return None
    return sb, sr


def _repeat(group, code):
    """Diagnostic only (attention_debug_repeat): run the phase's work twice, separated
    by a grid barrier; the repeat rewrites identical values (deterministic items)."""
    if not group.get("debug_repeat"):
        return code
    return "{\n" + code + "\n}\ngrid.sync();\n{\n" + code + "\n}"


def emit(group, layouts, dtypes, ident, *, which):
    """Return (kind, code, extra) for the forward, delta or backward phase.

    Raises AttentionRefused on any layout/dtype precondition, before any value
    is defined, so the caller can re-lower this group by ordinary operations.
    """
    label = group["label"]
    lq, lkt, lv = layouts[group["q"]], layouts[group["kt"]], layouts[group["v"]]
    for name in (group["q"], group["kt"], group["v"], group["o"], group.get("mask")):
        if name is not None and dtypes.get(name, "float32") != "float32":
            raise AttentionRefused(label, "fused attention admits FP32 operands only")
    if len(lq.shape) != 3 or len(lkt.shape) != 3 or len(lv.shape) != 3:
        raise AttentionRefused(label, "rank-3 batched operands required")
    bh, t, hd = lq.shape
    if tuple(lkt.shape) != (bh, hd, t) or tuple(lv.shape)[:2] != (bh, t) or lv.shape[2] != hd:
        raise AttentionRefused(label, f"self-attention shapes with equal key/value head width required: {lq.shape} {lkt.shape} {lv.shape}")
    if hd not in HEAD_WIDTHS:
        raise AttentionRefused(label, f"head width {hd} outside admitted {HEAD_WIDTHS}")
    if not 1 <= t <= MAX_SEQUENCE:
        raise AttentionRefused(label, f"context {t} outside admitted [1, {MAX_SEQUENCE}]")
    q = _head_layout(lq, bh, t, hd)
    # K rows from K^T: element (b, d, j) of KT is K[b][j][d].
    kb, kd, kr = lkt.strides
    k = (kb, kr) if kd == 1 and all(x % 4 == 0 for x in (kb, kr, lkt.offset)) else None
    v = _head_layout(lv, bh, t, hd)
    if None in (q, k, v):
        raise AttentionRefused(label, "Q/K/V need unit inner stride and 16-byte aligned rows")
    mask = None
    if group.get("mask") is not None:
        from . import padding_masks
        mask = padding_masks.mask_strides(group, layouts, bh, t, _mask_strides)
        if mask is None:
            raise AttentionRefused(label, "mask is not broadcastable to (batch*heads, T, T)")
    scale = group["scale"]
    mode = group["scale_mode"]
    if mode:
        c = abs(float(scale))
        if c == 0.0 and mode == 2:
            raise AttentionRefused(label, "division by zero scale")
        bound = c if mode == 1 else (math.inf if c == 0 else 1.0 / c)
        bound = math.nextafter(float(bound), math.inf) if math.isfinite(bound) else bound
        if not math.isfinite(bound):
            raise AttentionRefused(label, "score scale bound is not finite")
    else:
        bound = 1.0
    o_name = group["o"]
    lse, delta = o_name + "__lse", o_name + "__delta"
    names_ = tables(group)

    def f32(x):
        x = float(x)
        return repr(x) + "f"

    fields = [
        f"A.q = {ident(group['q'])}; A.q_b = {q[0]}LL; A.q_r = {q[1]}LL;",
        f"A.k = {ident(group['kt'])}; A.k_b = {k[0]}LL; A.k_r = {k[1]}LL;",
        f"A.v = {ident(group['v'])}; A.v_b = {v[0]}LL; A.v_r = {v[1]}LL;",
        f"A.bh = {bh}; A.t = {t}; A.scale = {f32(scale if mode else 1.0)}; A.scale_mode = {mode}; A.bound_scale = {f32(bound)};",
    ]
    if mask is not None:
        fields.append(f"A.mask.p = {ident(group['mask'])}; A.mask.sb = {mask[0]}LL; A.mask.si = {mask[1]}LL; A.mask.sj = {mask[2]}LL; A.has_mask = 1; A.mask_batched = {int(mask[0] != 0)};")
        fields.append(f"A.mmax = {ident(names_['mmax'])};")
    else:
        # Only an unmasked group clears the mask binding; an ordinary additive mask has no extended fields.
        fields.append("A.mask.p = nullptr; A.mask.sb = A.mask.si = A.mask.sj = 0; A.has_mask = 0; A.mask_batched = 0; A.mmax = nullptr;")
    from . import padding_masks
    extended = padding_masks.fields(group, mask)
    if extended:
        fields.append(extended)  # release step 09: per-example mask batches / PyTorch boolean-mask empty rows
    fields.append(f"A.qn = {ident(names_['qn'])}; A.kmax = {ident(names_['kmax'])};")
    nq = (t + 63) // 64
    mask_batches = (mask.batches if mask is not None and mask[0] != 0 else 1) if mask is not None else 0
    # P-OPS2-C v2 does not implement the step-09 fields; those groups keep the v1 tiles (same semantics).
    variant = "v1" if extended else select_variant(hd, t, bh, group.get("v2_enabled", True))
    nq2 = (t + 127) // 128
    mzero = ident(names_["mzero"]) if mask is not None else "nullptr"
    shared_tables = variant == "v2" and mask is not None and group.get("mask_tables_owner") is not None
    # Forward-only inference has no consumer for saved scores or dS.
    stored = variant == "v2" and bool(group.get("backward")) and bool(group.get("store_scores", True))
    tp = nq2 * 128
    if variant == "v2" and which == "prepare":
        items = bh * nq + (0 if shared_tables else mask_batches * nq * nq)
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
// One warp per prepare item (no block barriers).
for (long long item = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); item < {items}LL; item += (long long)gridDim.x * 8)
  pattn2::prepare_warp(A, {mzero}, item, true);""")
        outputs = [names_["qn"], names_["kmax"]] + ([names_["mmax"], names_["mzero"]] if mask is not None and not shared_tables else [])
        tile = (max(mask_batches, 1), nq, nq)
        shapes = {names_["qn"]: (bh, t), names_["kmax"]: (bh, nq), names_["mmax"]: tile, names_["mzero"]: tile}
        return "attention_prepare", code, dict(work_items=items, head_width=hd, sequence=t, head_batch=bh, variant=variant,
                                                  outputs=outputs, shapes={k: shapes[k] for k in outputs},
                                                  shared_reads=[names_["mmax"], names_["mzero"]] if shared_tables else [])
    if variant == "v2" and which == "forward":
        fields.append(f"A.o = {ident(o_name)}; A.lse = {ident(lse)};")
        items = bh * nq2
        xs = ident(names_["xs"]) if stored else "nullptr"
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
// 128-query items in decreasing causal cost, boustrophedon over CTAs (static LPT).
for (long long r = 0;; ++r) {{
  const long long idx = pattn2::snake(r);
  if (idx >= {items}LL) break;
  pattn2::forward_item<{mode}, {'true' if stored else 'false'}>(A, {mzero}, {xs}, (int)(idx % {bh}LL), {nq2 - 1} - (int)(idx / {bh}LL), smem);
}}""")
        outputs = [o_name, lse] + ([names_["xs"]] if stored else [])
        code = _repeat(group, code)
        return "attention_forward", code, dict(work_items=items, shared_bytes=V2_SHARED_BYTES["forward"], head_width=hd,
                                                  sequence=t, head_batch=bh, variant=variant, stored_scores=stored,
                                                  outputs=outputs, shapes={names_["xs"]: (bh, tp, tp)} if stored else {})
    if which == "prepare":
        items = bh * nq + mask_batches * nq * nq
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
for (long long item = blockIdx.x; item < {items}LL; item += gridDim.x)
  pattn::prepare_item<{hd}>(A, item, smem);""")
        outputs = [names_["qn"], names_["kmax"]] + ([names_["mmax"]] if mask is not None else [])
        shapes = {names_["qn"]: (bh, t), names_["kmax"]: (bh, nq), names_["mmax"]: (max(mask_batches, 1), nq, nq)}
        return "attention_prepare", code, dict(work_items=items, head_width=hd, sequence=t, head_batch=bh,
                                                  outputs=outputs, shapes={k: shapes[k] for k in outputs})
    if which == "forward":
        fields.append(f"A.o = {ident(o_name)}; A.lse = {ident(lse)};")
        items = bh * nq
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
for (long long item = blockIdx.x; item < {items}LL; item += gridDim.x) {{
  // Heaviest query blocks first (static longest-processing-time order).
  const int b = (int)(item % {bh}LL), qblk = {nq - 1} - (int)(item / {bh}LL);
  pattn::forward_item<{hd}>(A, b, qblk, smem);
}}""")
        shared = 4 * (3 * hd * 64 + 64 * 64 + 4 * 64)
        return "attention_forward", code, dict(work_items=items, shared_bytes=shared, head_width=hd,
                                                  sequence=t, head_batch=bh, outputs=[o_name, lse])
    lo, ldo = layouts[o_name], layouts[group["do"]]
    o = _head_layout(lo, bh, t, hd)
    do = _head_layout(ldo, bh, t, hd)
    if o is None or do is None or dtypes.get(group["do"], "float32") != "float32":
        raise AttentionRefused(label, "dO/O need FP32 unit inner stride and aligned rows")
    fields.append(f"A.oin = {ident(o_name)}; A.o_b = {o[0]}LL; A.o_r = {o[1]}LL; A.lse = {ident(lse)};")
    if stored:  # stored-score backward: no delta phase, no skip table
        fields.append(f"A.dout = {ident(group['do'])}; A.d_b = {do[0]}LL; A.d_r = {do[1]}LL; A.delta = nullptr;")
        fields.append("A.skip = nullptr;")
    else:
        fields.append(f"A.dout = {ident(group['do'])}; A.d_b = {do[0]}LL; A.d_r = {do[1]}LL; A.delta = {ident(delta)};")
        fields.append(f"A.skip = {ident(names_['skip'])};")
    if variant == "v2" and which == "delta":
        items = bh * nq
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
for (long long item = blockIdx.x; item < {items}LL; item += gridDim.x)
  pattn2::delta_item(A, (int)(item / {nq}LL), (int)(item % {nq}LL), smem);""")
        return "attention_delta", code, dict(work_items=items, rows=bh * t, variant=variant, outputs=[delta, names_["skip"]],
                                                shapes={delta: (bh, t), names_["skip"]: (bh, nq, nq)})
    if which == "delta":
        items = bh * nq
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
for (long long item = blockIdx.x; item < {items}LL; item += gridDim.x)
  pattn::delta_item<{hd}>(A, (int)(item / {nq}LL), (int)(item % {nq}LL));""")
        return "attention_delta", code, dict(work_items=items, rows=bh * t, outputs=[delta, names_["skip"]],
                                                shapes={delta: (bh, t), names_["skip"]: (bh, nq, nq)})
    dk_root = group["dkt"] + "__dk"
    fields.append(f"A.dq = {ident(group['dq'])}; A.dk = {ident(dk_root)}; A.dv = {ident(group['dv'])};")
    if stored and which == "backward":
        items = bh * nq2
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
// 128-key items from the forward's stored scores (dK, dV, dS scratch), decreasing causal cost.
for (long long r = 0;; ++r) {{
  const long long idx = pattn2::snake(r);
  if (idx >= {items}LL) break;
  pattn2::backward_kv_stored_item<{mode}>(A, {ident(names_['xs'])}, {ident(names_['ds'])}, (int)(idx % {bh}LL), (int)(idx / {bh}LL), smem);
}}""")
        code = _repeat(group, code)
        return "attention_backward", code, dict(work_items=items, shared_bytes=V2_SHARED_BYTES["backward_stored"], head_width=hd,
                                                   sequence=t, head_batch=bh, variant=variant, stored_scores=True,
                                                   outputs=[dk_root, group["dv"], names_["ds"]],
                                                   shapes={names_["ds"]: (bh, tp, tp)})
    if variant == "v2" and which == "backward":
        items = bh * nq2
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
// 128-key items (dK, dV, dS scratch) in decreasing causal cost, boustrophedon order.
for (long long r = 0;; ++r) {{
  const long long idx = pattn2::snake(r);
  if (idx >= {items}LL) break;
  pattn2::backward_kv_item<{mode}>(A, {ident(names_['ds'])}, {mzero}, (int)(idx % {bh}LL), (int)(idx / {bh}LL), smem);
}}""")
        return "attention_backward", code, dict(work_items=items, shared_bytes=V2_SHARED_BYTES["backward"], head_width=hd,
                                                   sequence=t, head_batch=bh, variant=variant,
                                                   outputs=[dk_root, group["dv"], names_["ds"]],
                                                   shapes={names_["ds"]: (bh, nq2 * 128, nq2 * 128)})
    if which == "dq":
        if variant != "v2":
            raise AttentionRefused(label, "the separate dQ pass exists only in the v2 schedule")
        items = bh * nq2
        code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
// 128-query dQ items from the dS scratch, decreasing causal cost, boustrophedon order.
for (long long r = 0;; ++r) {{
  const long long idx = pattn2::snake(r);
  if (idx >= {items}LL) break;
  pattn2::dq_item<{'true' if stored else 'false'}>(A, {ident(names_['ds'])}, (int)(idx % {bh}LL), {nq2 - 1} - (int)(idx / {bh}LL), smem);
}}""")
        code = _repeat(group, code)
        return "attention_dq", code, dict(work_items=items, shared_bytes=V2_SHARED_BYTES["dq"], head_width=hd,
                                             sequence=t, head_batch=bh, variant=variant, outputs=[group["dq"]])
    items = 2 * bh * nq
    code = ("pattn::Attn A{};\n" + "\n".join(fields) + f"""
for (long long item = blockIdx.x; item < {items}LL; item += gridDim.x) {{
  // Key-block items (heaviest first) then query-block items (lightest first):
  // with one item of each kind per block their causal costs roughly pair up.
  if (item < {bh * nq}LL) {{
    pattn::backward_kv_item<{hd}>(A, (int)(item % {bh}LL), (int)(item / {bh}LL), smem);
  }} else {{
    const long long r = item - {bh * nq}LL;
    pattn::backward_q_item<{hd}>(A, (int)(r % {bh}LL), (int)(r / {bh}LL), smem);
  }}
}}""")
    shared = 4 * (6 * hd * 64 + 2 * 64 * 64 + 3 * 64)
    return "attention_backward", code, dict(work_items=items, shared_bytes=shared, head_width=hd,
                                               sequence=t, head_batch=bh,
                                               outputs=[group["dq"], dk_root, group["dv"]])


def validate_views(group, layouts):
    """The backward products' other operands must be exactly K, Q^T and V^T."""
    label = group["label"]
    lq, lkt, lv = layouts[group["q"]], layouts[group["kt"]], layouts[group["v"]]

    def same(layout, root, offset, shape, strides):
        return (layout.root, layout.offset, tuple(layout.shape), tuple(layout.strides)) == (root, offset, tuple(shape), tuple(strides))

    def swapped(layout):
        s = layout.strides
        return (layout.shape[0], layout.shape[2], layout.shape[1]), (s[0], s[2], s[1])

    k_shape, k_strides = swapped(lkt)
    if not same(layouts[group["k_view"]], lkt.root, lkt.offset, k_shape, k_strides):
        raise AttentionRefused(label, "dQ product does not read the forward K")
    qt_shape, qt_strides = swapped(lq)
    if not same(layouts[group["qt_view"]], lq.root, lq.offset, qt_shape, qt_strides):
        raise AttentionRefused(label, "dK^T product does not read the forward Q^T")
    vt_shape, vt_strides = swapped(lv)
    if not same(layouts[group["vt_view"]], lv.root, lv.offset, vt_shape, vt_strides):
        raise AttentionRefused(label, "dP product does not read the forward V^T")


def tables(group):
    """Names of the small per-attention proof tables (never T x T)."""
    o = group["o"]
    owner = group.get("mask_tables_owner") or o  # P-OPS2-C: one mask-statistics pass per mask tensor
    return {"qn": o + "__qn", "kmax": o + "__kmax", "mmax": owner + "__mmax", "skip": o + "__skip",
            "mzero": owner + "__mzero", "ds": o + "__ds", "xs": o + "__xs"}


def share_mask_tables(groups, layouts, values, field):
    """P-OPS2-C: v2 groups reading the same mask tensor with the same (BH, T, HD) share
    the first such group's mask-tile max / all-zero tables (pure functions of the
    mask). v1 groups never share, so the v1 lowering is unchanged."""
    owners = {}
    for group in groups:
        mask = group.get("mask")
        q = tuple(int(x) for x in field(values[group["q"]], "shape")) if group.get("q") in values else None
        if mask is None or q is None or len(q) != 3:
            continue
        if select_variant(q[2], q[1], q[0], group.get("v2_enabled", True)) != "v2":
            continue
        if group.get("safe") or group.get("mask_lead") is not None:
            continue  # release step 09 groups use v1 tiles (no shared v2 tables)
        key = (mask, q)
        if key in owners:
            group["mask_tables_owner"] = owners[key]
        else:
            owners[key] = group["o"]


def phase_root(group, kind):
    """Collision-checked name for an atomic scheduler phase's packed storage."""
    return group["o"] + "__phase_" + kind


def emit_member(group, name, layouts, dtypes, values, *, field, ident,
                define, add_phase, add_view, record):
    """Install a matched group's phases using scheduler-owned callbacks.

    Call for every operation in ``group['skip']`` instead of ordinary lowering.
    Non-anchor operations do nothing, except the final backward member validates
    its transpose views. A later refusal requires discarding this provisional
    lowering and retrying with the group's label excluded from ``plan``.

    ``define(name, shape, root=None, strides=None)`` registers FP32 storage or a
    view; ``add_phase(kind, code, metadata, reads, writes)`` installs one atomic
    phase. ``add_view(name, target, source)`` records alias provenance.
    ``record(metadata)`` records the structural decision. Callers supply the
    header once and propagate ``shared_bytes`` and ``work_items`` to runtime.
    No callback commits model, optimizer or RNG state.
    """
    label, output = group["label"], group["o"]
    if len(layouts[group["q"]].shape) != 3:
        raise AttentionRefused(label, "rank-3 batched operands required")
    bh, length, width = layouts[group["q"]].shape
    head_shape = (bh, length, width)
    row_shape = (bh, length)
    proof = tables(group)
    base_reads = [group[key] for key in ("q", "kt", "v", "mask") if group.get(key) is not None]

    def check_shape(value, expected):
        if tuple(field(values[value], "shape")) != tuple(expected):
            raise AttentionRefused(label, f"{value}: captured shape differs from fused result {expected}")
        if dtypes[value] != "float32":
            raise AttentionRefused(label, f"{value}: fused output must be FP32")

    def install(emission, reads):
        kind, code, extra = emission
        writes = list(extra["outputs"])
        metadata = {key: value for key, value in extra.items() if key not in ("outputs", "shapes")}
        metadata.update(group=label, attention_output=output,
                        source_commit=SOURCE_COMMIT_V2 if extra.get("variant") == "v2" else SOURCE_COMMIT,
                        header=HEADER_V2 if extra.get("variant") == "v2" else HEADER, threads=256)
        add_phase(kind, code, metadata, list(dict.fromkeys(reads)), writes)

    if name == group["anchor_forward"]:
        # Both emissions validate before their outputs are installed.
        prepare = emit(group, layouts, dtypes, ident, which="prepare")
        forward = emit(group, layouts, dtypes, ident, which="forward")
        check_shape(output, head_shape)
        for value, shape in prepare[2]["shapes"].items():
            define(value, shape)
        install(prepare, base_reads)
        define(output, head_shape)
        define(output + "__lse", row_shape)
        for value, shape in forward[2].get("shapes", {}).items():
            define(value, shape)
        install(forward, base_reads + prepare[2]["outputs"] + prepare[2].get("shared_reads", []))
        record({"group": label, "forward": list(group["forward"]),
                "backward": list(group.get("backward") or ()), "head_width": width,
                "sequence": length, "head_batch": bh, "scale": group["scale"],
                "scale_mode": group["scale_mode"], "mask": group["mask"],
                "removed": sorted(group["removed"]),
                "source_commit": SOURCE_COMMIT_V2 if forward[2].get("variant") == "v2" else SOURCE_COMMIT,
                "decision": "fused_fp32_attention", "variant": forward[2].get("variant", "v1"),
                "layout_restore": group.get("restore_dkt_reason"),
                **({"padding_mask": {"safe_rows": bool(group.get("safe")), "mask_lead": group.get("mask_lead")}}
                   if group.get("safe") or group.get("mask_lead") is not None else {})})

    if group.get("backward") and name == group["anchor_backward"]:
        backward = emit(group, layouts, dtypes, ident, which="backward")
        stored = bool(backward[2].get("stored_scores"))
        delta = None if stored else emit(group, layouts, dtypes, ident, which="delta")
        for value in (group["dq"], group["dv"]):
            check_shape(value, head_shape)
        check_shape(group["dkt"], (bh, width, length))
        proof_reads = [proof["qn"], proof["kmax"]] + ([proof["mmax"]] if group["mask"] else [])
        read_outputs = [output, output + "__lse", group["do"]]
        if delta is not None:
            for value, shape in delta[2]["shapes"].items():
                define(value, shape)
            install(delta, base_reads + proof_reads + read_outputs)
        dk = group["dkt"] + "__dk"
        for value in (group["dq"], dk, group["dv"]):
            define(value, head_shape)
        # Preserve the cheap transpose alias when every downstream view admits
        # it. Otherwise restore the original bmm result's physical layout with
        # an exact typed copy; PyTorch's captured view ABI must remain valid.
        if not group.get("restore_dkt_reason"):
            define(group["dkt"], (bh, width, length), root=dk,
                   strides=(length * width, 1, width))
            add_view(group["dkt"], "p_ops_d.attention_dkt", dk)
        if stored:
            # P-OPS2-C stored scores: key-block phase reads the forward's scores (no delta
            # phase), then the dQ pass; both decide negligible steps identically.
            for value, shape in backward[2]["shapes"].items():
                define(value, shape)
            install(backward, base_reads + proof_reads + read_outputs + [proof["xs"]])
            install(emit(group, layouts, dtypes, ident, which="dq"),
                    [group["kt"], proof["ds"], output + "__lse"] + proof_reads)
        elif backward[2].get("variant") == "v2":
            # P-OPS2-C: key-block phase (dK, dV, dS scratch), then the dQ pass.
            if group["mask"]:
                proof_reads.append(proof["mzero"])
            for value, shape in backward[2]["shapes"].items():
                define(value, shape)
            install(backward, base_reads + proof_reads + read_outputs + delta[2]["outputs"])
            install(emit(group, layouts, dtypes, ident, which="dq"),
                    [group["kt"], proof["ds"], proof["skip"]])
        else:
            install(backward, base_reads + proof_reads + read_outputs + delta[2]["outputs"])
        if group.get("restore_dkt_reason"):
            define(group["dkt"], (bh, width, length))
            total = bh * width * length
            source_index = (f"(i / {width * length}LL) * {width * length}LL + "
                            f"(i % {length}LL) * {width}LL + (i / {length}LL) % {width}LL")
            code = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; "
                    f"i < {total}LL; i += (long long)gridDim.x * blockDim.x) "
                    f"{ident(group['dkt'])}[i] = {ident(dk)}[{source_index}];")
            install(("attention_layout_restore", code,
                     {"outputs": [group["dkt"]], "work_items": (total + 255) // 256,
                      "reason": group["restore_dkt_reason"], "arithmetic": "FP32 bit-preserving transpose copy"}), [dk])

    if group.get("backward") and name == group["last_backward"]:
        validate_views(group, layouts)


def add_fused_nodes(kind, code, metadata, reads, writes, *, layouts, dtypes,
                    make_layout, add_heavy, add_view, reserved_names=()):
    """Adapt one multi-output phase to a scheduler with one root per heavy node.

    Every same-dtype output becomes a view of one packed, contiguous allocation.
    Segments are aligned to four floats; existing aliases (notably dKT) are
    rebased too. The allocation therefore remains live through the final user of
    *any* sibling. This modest retention cost is explicit in ``packed_outputs``.

    ``add_heavy`` has fusion.py's (kind, op, name, code, refs, **extra) signature.
    ``make_layout`` has TensorLayout's (shape, strides, root, offset) signature.
    The central scheduler retains authority over placement and arena reuse.
    """
    group = {"o": metadata.get("attention_output", metadata.get("atomic_output"))}
    root = phase_root(group, kind)
    if root in layouts or root in reserved_names:
        raise AttentionRefused(metadata["group"], f"packed phase name collision: {root}")
    if len(set(writes)) != len(writes) or not writes:
        raise AttentionRefused(metadata["group"], "phase outputs must be unique and nonempty")
    dtype = dtypes[writes[0]]
    if any(dtypes[value] != dtype for value in writes):
        return _add_mixed_packed(kind, code, metadata, reads, writes, root, layouts=layouts, dtypes=dtypes,
                                 make_layout=make_layout, add_heavy=add_heavy, add_view=add_view)
    if dtype not in ("float32", "int64", "bfloat16"):
        raise AttentionRefused(metadata["group"], f"unsupported atomic packed dtype {dtype}")
    # BF16 segments (P-OPS3-B short attention outputs) start on 16-byte boundaries.
    align = 8 if dtype == "bfloat16" else 4
    packed, count = {}, 0
    old_roots = {}
    for value in writes:
        layout = layouts[value]
        if dtypes[value] != dtype or layout.root != value or layout.offset:
            raise AttentionRefused(metadata["group"], f"{value}: phase output must be fresh, contiguous and same dtype")
        expected = tuple(math.prod(layout.shape[i + 1:]) for i in range(len(layout.shape)))
        if tuple(layout.strides) != expected:
            raise AttentionRefused(metadata["group"], f"{value}: phase output must be contiguous")
        count = (count + align - 1) // align * align
        packed[value] = {"offset": count, "shape": list(layout.shape), "strides": list(layout.strides)}
        old_roots[value] = count
        count += math.prod(layout.shape)
    for value, layout in list(layouts.items()):
        if layout.root in old_roots:
            layouts[value] = make_layout(layout.shape, layout.strides, root,
                                         old_roots[layout.root] + layout.offset)
    layouts[root] = make_layout((count,), (1,), root, 0)
    dtypes[root] = dtype
    for value in writes:
        add_view(value, "p_ops_d.attention_packed_output", root)
    extra = dict(metadata, packed_outputs=packed, packed_elements=count,
                 phase_writes=list(writes))
    add_heavy(kind, {"target": "p_ops_d." + kind}, root, code, reads, **extra)
    return root


_PACK_BYTES = {"float32": 4, "bfloat16": 2, "int64": 8}


def _add_mixed_packed(kind, code, metadata, reads, writes, root, *, layouts, dtypes, make_layout, add_heavy, add_view):
    """P-OPS3-B: one packed FP32-word allocation holding outputs of different dtypes.

    Each output starts on a 16-byte boundary; its layout offset is expressed in its own element size, which the
    declarations and runtime views already apply per value dtype.
    """
    packed, offset_bytes, old_roots = {}, 0, {}
    for value in writes:
        layout, dtype = layouts[value], dtypes[value]
        if dtype not in _PACK_BYTES or layout.root != value or layout.offset:
            raise AttentionRefused(metadata["group"], f"{value}: mixed packed output must be fresh with a known dtype")
        expected = tuple(math.prod(layout.shape[i + 1:]) for i in range(len(layout.shape)))
        if tuple(layout.strides) != expected:
            raise AttentionRefused(metadata["group"], f"{value}: phase output must be contiguous")
        offset_bytes = (offset_bytes + 15) // 16 * 16
        size = _PACK_BYTES[dtype]
        packed[value] = {"offset_bytes": offset_bytes, "dtype": dtype, "shape": list(layout.shape),
                         "strides": list(layout.strides)}
        old_roots[value] = offset_bytes // size
        offset_bytes += math.prod(layout.shape) * size
    for value, layout in list(layouts.items()):
        if layout.root in old_roots:
            layouts[value] = make_layout(layout.shape, layout.strides, root, old_roots[layout.root] + layout.offset)
    words = (offset_bytes + 3) // 4
    layouts[root] = make_layout((words,), (1,), root, 0)
    dtypes[root] = "float32"
    for value in writes:
        add_view(value, "p_ops3_b.mixed_packed_output", root)
    extra = dict(metadata, packed_outputs=packed, packed_elements=words, packed_words=words,
                 phase_writes=list(writes))
    add_heavy(kind, {"target": "p_ops3_b." + kind}, root, code, reads, **extra)
    return root
