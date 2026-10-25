"""P-OPS5-D: fused mixed-precision short attention for training graphs (operation compiler rule).

Recognizes, by graph structure and dtypes only, the attention that the ``mixed_bf16`` precision policy (or an
identical explicit-cast program) produces:

    S  = bmm(Qb, KTb)                          BF16 operands, BF16 result
    X  = mul|div(_to_copy(S, float32), c)      optional captured scalar
    P  = _softmax(X, -1)                       FP32
    O  = bmm(_to_copy(P, bfloat16), Vb)        BF16

and the derivative PyTorch produced for it:

    dV  = bmm(transpose(Pb), dOb)              dP = _to_copy(bmm(dOb, transpose(Vb)), float32)
    dS  = _to_copy(mul|div(_softmax_backward_data(dP, P, -1, float32), c), bfloat16)
    dKT = bmm(transpose(Qb), dS)               dQ = bmm(dS, transpose(KTb))

Around the group, operand chains back to their storage (views, slices/selects, clones, casts, and the policy's
biased linear ``widen(x) + b``) and result chains forward to their first consumer that is not a layout or
cast operation (views, casts, one concatenation of sibling gradients, clones) are absorbed. Layout maps are
resolved *numerically*: the chain operations are replayed on index tensors and the resulting gather/scatter
maps must be exactly affine in (batch, head, row, unit) with unit stride, otherwise the group is refused. The
group then lowers to two phases (forward, backward) of ``attention_mixed.cuh`` work items, one per (batch,
head); nothing else is stored (no scores, probabilities, gradient matrices or head copies; the forward saves
two FP32 numbers per row). Anything not admitted keeps the existing lowering and is recorded as a refusal.
Selection never inspects names beyond graph references.
"""
from __future__ import annotations

import math

from .short_attention import _Graph, _base

HEADER = "attention_mixed.cuh"
HEAD_WIDTHS = (32, 64)
MAX_SEQUENCE = 128
# P8-E: 128 < T <= LONG_SEQUENCE lowers to the long-context items of the same header (forward_long_phase /
# backward_long_phase): the same declared arithmetic, keys streamed in 16-key steps, no T x T tile on chip.
LONG_SEQUENCE = 256
# P8-E: the long backward stores bf16(P)^T and bf16(dS)^T per item ([BH][2][TP][TP] BF16 scratch, written and read
# inside the backward phase) so its key stage reads them instead of recomputing scores and probabilities.
LONG_STORE_PDS = True
LONG_STAGE_SPLIT = True   # P8-E: long backward row and key stages are separate work units (per-item flags, split scratch)
# P9-E: long items v2 (attention_mixed.cuh PMIX_LONG_V2): register-resident scores in the forward, cp.async-staged
# operands (bias completed in shared memory), prefetched stored P/dS fragments; bitwise identical to the v1 items.
LONG_V2 = True
# P9-E: long backward v3 (PMIX_LONG_V3): half-item units claimed in order, per-warp bias partials combined in the v1
# order by the second finisher; split scratch: 3 BH + 2 counters, [BH][2][3][8][HD] FP32 partials.
LONG_V3 = True
# P9-E: v2 forward phase (PMIX_LONG_FWD_V2); measured slower than the v1 forward items (gpu-04a), default off.
LONG_FWD_V2 = False
# P9-E declared precision option (amendment A4, PMIX_FA_DELTA): long backward delta from the forward output O.
FA_DELTA = False
SHARED_LIMIT = 133120  # keep the program inside the H100 L1 carve-out (P-OPS-D / P-OPS3-B finding)
_VIEWS = {"view", "_unsafe_view", "reshape", "permute", "transpose", "t", "unsqueeze", "squeeze", "alias",
          "detach", "slice", "select", "expand"}
_COPIES = {"clone", "contiguous"}
_CAST_KWARGS = {"dtype", "layout", "device", "pin_memory", "memory_format"}


class Refused(Exception):
    pass


def _tp(t):
    return (t + 15) // 16 * 16


# Diagnostic only (default False): lower T <= 128 through the long items too, so they can be compared bitwise
# with the whole-row items at the same context.
FORCE_LONG = False


def is_long(t):
    return t > MAX_SEQUENCE or FORCE_LONG


def forward_shared(hd, t, single_only=False):
    """Bytes for every schedule (forward_item / pipelined forward_phase / forward_phase_split)."""
    tp = _tp(t)
    if is_long(t):
        if LONG_V2 and LONG_FWD_V2:  # forward_long_v2_shared_bytes: K, double-buffered V, staged biases
            return 2 * 3 * tp * (hd + 8) + 4 * 3 * hd
        return 2 * 2 * tp * (hd + 8)  # forward_long_shared_bytes: resident K and V
    pipelined = 2 * 2 * (3 * tp * (hd + 8) + 6 * hd)
    if single_only:
        return 2 * 3 * tp * (hd + 8)
    return max(2 * 3 * tp * (hd + 8), pipelined, pipelined + 2 * 2 * tp * (tp + 8), dual_forward_bytes(hd, tp))


# P8-E dual items (attention_mixed.cuh PMIX_DUAL, TP == 64): two (batch, head) items per CTA at once, one per half.
DUAL = True


def dual_forward_bytes(hd, tp):
    if not (DUAL and tp == 64):
        return 0
    return 2 * (2 if hd == 32 else 1) * (3 * tp * (hd + 8) + 6 * hd) * 2


def dual_backward_bytes(hd, tp):
    if not (DUAL and tp == 64 and hd == 32):
        return 0
    return 2 * (4 * tp * (hd + 8) + 6 * hd + 4 * tp + 2 * tp * (tp + 8) + 2 * (3 * 8 * hd + 72)) * 2


# P6-B: shared-memory ceiling the backward should fit (None: no constraint). A pipelined backward above it uses the
# single-buffer item schedule (backward_phase<HD, TP, false>), e.g. so a two-CTA program keeps its residency.
BWD_FIT_LIMIT = None


def backward_pipelined(hd, t):
    tp = _tp(t)
    colsum = 4 * 3 * 8 * hd + 4 * 72
    pipelined = 2 * (2 * (4 * tp * (hd + 8) + 6 * hd + 4 * tp) + 2 * tp * (tp + 8)) + colsum
    return BWD_FIT_LIMIT is None or max(pipelined, dual_backward_bytes(hd, tp)) <= BWD_FIT_LIMIT


def backward_shared(hd, t, single_only=False):
    tp = _tp(t)
    if is_long(t):  # backward_long_shared_bytes: K/V then Q/dO, (m, l), delta, column sums + flags
        return 2 * 2 * tp * (hd + 8) + 4 * 3 * tp + 4 * 3 * 8 * hd + 4 * 72 + (4 * 2 * hd if LONG_V2 else 0)
    colsum = 4 * 3 * 8 * hd + 4 * 72  # absorbed bias-gradient column sums [3][warps][HD] + counter flags
    single = 2 * (4 * tp * (hd + 8) + 2 * tp * (tp + 8)) + 8 * tp + colsum
    pipelined = 2 * (2 * (4 * tp * (hd + 8) + 6 * hd + 4 * tp) + 2 * tp * (tp + 8)) + colsum
    if not single_only and backward_pipelined(hd, t):
        return max(single, pipelined, dual_backward_bytes(hd, tp))
    return single


# --------------------------------------------------------------------------------------------- graph helpers
class _G(_Graph):
    def __init__(self, graph_args):
        super().__init__(*graph_args)
        self.args_tuple = graph_args

    def base(self, name):
        return _base(self.op(name)["target"]) if name in self.ops else None

    def arg(self, name, i):
        args = self.op(name)["args"]
        return self.ref(args[i], self.names) if i < len(args) else None

    def users(self, name):
        return self.consumers.get(name, [])

    def cast_dtype(self, name):
        """Target dtype of a pure dtype cast (_to_copy with only layout/device keywords), else None."""
        if self.base(name) != "_to_copy":
            return None
        kwargs = self.op(name)["kwargs"]
        if set(kwargs) - _CAST_KWARGS or "dtype" not in kwargs:
            return None
        return str(kwargs["dtype"]).removeprefix("torch.")

    def scalar(self, item):
        return item if isinstance(item, (int, float)) and not isinstance(item, bool) else None


def _transpose_source(g, name):
    """Source of a transpose(x, 1, 2) / transpose(x, -1, -2) of a rank-3 value, else None."""
    if g.base(name) != "transpose":
        return None
    args = g.op(name)["args"]
    if len(g.shape(name)) != 3 or {int(args[1]) % 3, int(args[2]) % 3} != {1, 2}:
        return None
    return g.arg(name, 0)


def _scale_of(g, name):
    """(source, mode, scalar) for mul/div by a captured scalar, else (name, 0, 1.0)."""
    base = g.base(name)
    if base in ("mul", "div") and len(g.op(name)["args"]) == 2:
        c = g.scalar(g.op(name)["args"][1])
        if c is not None:
            return g.arg(name, 0), (1 if base == "mul" else 2), float(c)
    return name, 0, 1.0


def match(graph_args, exclude=()):
    """Mixed attention groups in operation order: (groups, refusals)."""
    g = _G(graph_args)
    groups, refusals = [], []
    for name, op in g.ops.items():
        if g.base(name) != "_softmax":
            continue
        label = "mixed:" + name
        if label in exclude:
            continue
        try:
            groups.append(_match_one(g, name, label))
        except Refused as exc:
            if g.dtype(name) == "float32" and _looks_mixed(g, name):
                refusals.append({"group": label, "reason": "mixed attention: " + str(exc)})
    return groups, refusals


def _looks_mixed(g, softmax):
    users = g.users(softmax)
    return any(g.cast_dtype(u) == "bfloat16" for u in users)


def _match_one(g, softmax, label):
    args = g.op(softmax)["args"]
    if len(args) < 3 or int(args[1]) not in (-1, len(g.shape(softmax)) - 1) or bool(args[2]):
        raise Refused("softmax must be over the last dimension without dtype change")
    if g.dtype(softmax) != "float32" or len(g.shape(softmax)) != 3:
        raise Refused("FP32 rank-3 probabilities required")
    x = g.arg(softmax, 0)
    widened, mode, scale = _scale_of(g, x)
    if g.cast_dtype(widened) != "float32":
        raise Refused("scores are not a widened BF16 product (masks are not admitted)")
    s = g.arg(widened, 0)
    if g.base(s) != "bmm" or g.dtype(s) != "bfloat16":
        raise Refused("scores are not a BF16 batched product")
    q, kt = g.arg(s, 0), g.arg(s, 1)
    bh, t, hd = g.shape(q)
    if g.shape(kt) != (bh, hd, t) or g.shape(softmax) != (bh, t, t):
        raise Refused("self-attention with equal query/key lengths required")
    # forward value product
    casts = [u for u in g.users(softmax) if g.cast_dtype(u) == "bfloat16"]
    backs = [u for u in g.users(softmax) if g.base(u) == "_softmax_backward_data"]
    if len(casts) != 1 or len(g.users(softmax)) != 1 + len(backs):
        raise Refused("probabilities need one BF16 cast (and the softmax derivative)")
    pb = casts[0]
    value_products = [u for u in g.users(pb) if g.base(u) == "bmm" and g.arg(u, 0) == pb]
    if len(value_products) != 1:
        raise Refused("one value product of the BF16 probabilities required")
    o = value_products[0]
    v = g.arg(o, 1)
    if g.shape(v) != (bh, t, hd) or g.dtype(o) != "bfloat16" or g.dtype(v) != "bfloat16":
        raise Refused("BF16 values with the query head width required")
    if hd not in HEAD_WIDTHS or not 1 <= t <= LONG_SEQUENCE:
        raise Refused(f"head width {hd} / context {t} outside {HEAD_WIDTHS} / [1, {LONG_SEQUENCE}]")
    group = dict(label=label, softmax=softmax, x=x, widened=widened, s=s, pb=pb, o=o, q=q, kt=kt, v=v,
                 bh=bh, t=t, hd=hd, scale=scale, scale_mode=mode, backward=False,
                 members={softmax, widened, s, pb, o} | ({x} if x != widened else set()))
    if not backs:
        raise Refused("forward-only attention keeps the existing lowering")
    if len(backs) != 1:
        raise Refused("one softmax derivative required")
    sbd = backs[0]
    bargs = g.op(sbd)["args"]
    if g.arg(sbd, 1) != softmax or int(bargs[2]) not in (-1, 2) or str(bargs[3]).removeprefix("torch.") != "float32":
        raise Refused("softmax derivative does not match the forward softmax")
    dp32 = g.arg(sbd, 0)
    if g.cast_dtype(dp32) != "float32":
        raise Refused("probability gradient is not a widened BF16 product")
    dpb = g.arg(dp32, 0)
    if g.base(dpb) != "bmm" or g.dtype(dpb) != "bfloat16":
        raise Refused("probability gradient product is not BF16")
    dout, vt = g.arg(dpb, 0), g.arg(dpb, 1)
    if _transpose_source(g, vt) != v:
        raise Refused("probability gradient does not read the forward values")
    # dV = bmm(transpose(Pb), dOb)
    dv = [u for u in g.users(pb) if g.base(u) == "transpose" and _transpose_source(g, u) == pb]
    if len(dv) != 1:
        raise Refused("value gradient product not found")
    pbt = dv[0]
    dvs = [u for u in g.users(pbt) if g.base(u) == "bmm" and g.arg(u, 0) == pbt and g.arg(u, 1) == dout]
    if len(dvs) != 1 or len(g.users(pb)) != 2 or len(g.users(pbt)) != 1:
        raise Refused("probabilities are used outside the attention")
    dvv = dvs[0]
    # dS = cast(scale(sbd))
    users = g.users(sbd)
    if len(users) != 1:
        raise Refused("softmax derivative has extra consumers")
    scaled = users[0]
    if mode:
        src, m2, c2 = _scale_of(g, scaled)
        if src != sbd or m2 != mode or c2 != scale:
            raise Refused("score-gradient scale differs from the forward scale")
        users = g.users(scaled)
        if len(users) != 1:
            raise Refused("scaled score gradient has extra consumers")
        dsb = users[0]
    else:
        dsb, scaled = scaled, None
    if g.cast_dtype(dsb) != "bfloat16":
        raise Refused("score gradient is not rounded to BF16")
    dq = dkt = qt = kk = None
    for u in g.users(dsb):
        if g.base(u) != "bmm":
            raise Refused("score gradient used outside the attention")
        a0, a1 = g.arg(u, 0), g.arg(u, 1)
        if a0 == dsb and _transpose_source(g, a1) == kt:
            dq, kk = u, a1
        elif a1 == dsb and _transpose_source(g, a0) == q:
            dkt, qt = u, a0
        else:
            raise Refused("unrecognized score-gradient product")
    if dq is None or dkt is None or len(g.users(dsb)) != 2:
        raise Refused("query/key gradient products not found")
    members = group["members"] | {sbd, dp32, dpb, pbt, dvv, dsb, dq, dkt, vt, qt, kk} | ({scaled} if scaled else set())
    group.update(backward=True, sbd=sbd, dp32=dp32, dpb=dpb, dout=dout, dv=dvv, dq=dq, dkt=dkt, dsb=dsb,
                 vt=vt, qt=qt, k_view=kk, pbt=pbt, members=members)
    return group


# ------------------------------------------------------------------------------------ numeric layout maps
def _torch():
    import torch
    return torch


def _replay(g, name, x, shape):
    """Apply one layout/cast operation to an index tensor x (its single tensor input)."""
    torch = _torch()
    base = g.base(name)
    args = g.op(name)["args"]
    if base in ("view", "_unsafe_view", "reshape", "unsqueeze", "squeeze"):
        return x.reshape(shape)
    if base == "permute":
        return x.permute([int(d) for d in args[1]])
    if base == "transpose":
        return x.transpose(int(args[1]), int(args[2]))
    if base == "t":
        return x.t()
    if base == "expand":
        return x.expand(shape)
    if base == "slice":
        dim = int(args[1]) if len(args) > 1 else 0
        extent = x.shape[dim]
        start = int(args[2]) if len(args) > 2 and args[2] is not None else 0
        end = int(args[3]) if len(args) > 3 and args[3] is not None else extent
        step = int(args[4]) if len(args) > 4 else 1
        start, end = (start + extent if start < 0 else start), (end + extent if end < 0 else end)
        start, end = max(0, min(extent, start)), max(0, min(extent, end))
        return x.index_select(dim, torch.arange(start, end, step))
    if base == "select":
        return x.select(int(args[1]), int(args[2]))
    if base in _COPIES or base in ("alias", "detach", "_to_copy"):
        return x
    raise Refused(f"{base} cannot be replayed as a layout operation")


def _index_tensor(layout):
    """Element offsets of a value (relative to its own pointer) in its layout."""
    torch = _torch()
    shape, strides = tuple(layout[0]), tuple(layout[1])
    idx = torch.zeros(shape, dtype=torch.int64)
    for dim, (n, s) in enumerate(zip(shape, strides)):
        view = [1] * len(shape)
        view[dim] = n
        idx = idx + torch.arange(n, dtype=torch.int64).reshape(view) * int(s)
    return idx


def _affine(index, bh, heads, label):
    """(o, sb, sh, sr) of an exactly affine map index[bh, i, d] = o + sb*b + sh*h + sr*i + d; else Refused."""
    torch = _torch()
    n_bh, t, hd = index.shape
    o = int(index[0, 0, 0])
    if o < 0:
        raise Refused(f"{label}: negative base offset")
    sd = int(index[0, 0, 1] - o) if hd > 1 else 1
    sr = int(index[0, 1, 0] - o) if t > 1 else 0
    sh = int(index[1, 0, 0] - o) if heads > 1 else 0
    sb = int(index[heads, 0, 0] - o) if n_bh > heads else 0
    if sd != 1:
        raise Refused(f"{label}: head width is not unit-stride")
    b = torch.arange(n_bh) // heads
    h = torch.arange(n_bh) % heads
    model = (o + (b * sb + h * sh)[:, None, None] + torch.arange(t)[None, :, None] * sr
             + torch.arange(hd)[None, None, :])
    if not torch.equal(model, index):
        raise Refused(f"{label}: layout map is not affine in (batch, head, row, unit)")
    return o, sb, sh, sr


def _heads_for(maps, bh):
    """Head counts h | bh (smallest first, excluding 1 when bh > 1 is not affine) under which every map is affine."""
    errors = []
    for heads in [d for d in range(1, bh + 1) if bh % d == 0]:
        try:
            return heads, {k: _affine(m, bh, heads, k) for k, m in maps.items()}
        except Refused as exc:
            errors.append(str(exc))
    raise Refused("no batch/head decomposition makes every operand map affine: " + "; ".join(errors[-3:]))


# ------------------------------------------------------------------------------------------ operand chains
def _input_chain(g, value):
    """Walk an operand back to its storage. Returns a source description (kind, chain, ...)."""
    chain, current = [], value
    narrows = widens = 0
    while current in g.ops:
        base = g.base(current)
        dtype = g.cast_dtype(current)
        if base in _VIEWS or base in _COPIES:
            chain.append(current)
            current = g.arg(current, 0)
            continue
        if base == "_to_copy" and dtype in ("float32", "bfloat16"):
            if dtype == "bfloat16":
                narrows += 1
            else:
                widens += 1
            chain.append(current)
            current = g.arg(current, 0)
            continue
        if base == "add" and narrows == widens + 1 and g.dtype(current) == "float32":
            return _biased_source(g, current, list(reversed(chain)))
        break
    if g.dtype(current) == "bfloat16" and narrows == widens:
        return dict(kind=0, source=current, chain=list(reversed(chain)))
    if g.dtype(current) == "float32" and narrows == widens + 1:
        return dict(kind=2, source=current, chain=list(reversed(chain)))
    raise Refused(f"{value}: operand source {current} is not BF16, FP32 rounded once, or a biased BF16 product")


def _biased_source(g, add, after):
    op = g.op(add)
    if op["kwargs"] or len(op["args"]) != 2:
        raise Refused(f"{add}: bias add with alpha/extra arguments")
    a, b = g.arg(add, 0), g.arg(add, 1)
    for x, bias in ((a, b), (b, a)):
        if x is None or bias is None:
            continue
        inner = _input_chain_widen(g, x)
        if inner is None or g.dtype(bias) != "float32":
            continue
        bias_shape = g.shape(bias)
        if len(bias_shape) < 1 or (math.prod(bias_shape) != bias_shape[-1]) or bias_shape[-1] != g.shape(add)[-1]:
            continue
        source, before = inner
        return dict(kind=1, source=source, chain=before, add=add, bias=bias, after=after)
    raise Refused(f"{add}: not a widened BF16 product plus a last-dimension FP32 bias vector")


def _input_chain_widen(g, value):
    """value = layout ops of widen(BF16 source) -> (source, chain forward order) or None."""
    chain, current, widens = [], value, 0
    while current in g.ops:
        base = g.base(current)
        dtype = g.cast_dtype(current)
        if base in _VIEWS or base in _COPIES:
            chain.append(current)
            current = g.arg(current, 0)
            continue
        if base == "_to_copy" and dtype == "float32" and widens == 0:
            widens += 1
            chain.append(current)
            current = g.arg(current, 0)
            continue
        break
    if widens != 1 or g.dtype(current) != "bfloat16":
        return None
    return current, list(reversed(chain))


def _gather(g, desc, layout_of, value):
    """Index map (element offsets in the source value) of every element of the operand `value`."""
    src = desc["source"]
    x = _index_tensor(layout_of(src))
    chain = desc["chain"] if desc["kind"] != 1 else desc["chain"]
    for name in chain:
        x = _replay(g, name, x, g.shape(name))
    bias_map = None
    if desc["kind"] == 1:
        torch = _torch()
        add_shape = g.shape(desc["add"])
        x = x.expand(add_shape) if tuple(x.shape) != add_shape else x
        bias_idx = _index_tensor(layout_of(desc["bias"])).reshape(-1)
        bias_map = bias_idx.reshape([1] * (len(add_shape) - 1) + [-1]).expand(add_shape)
        for name in desc["after"]:
            x = _replay(g, name, x, g.shape(name))
            bias_map = _replay(g, name, bias_map, g.shape(name))
        del torch
    if tuple(x.shape) != g.shape(value):
        raise Refused(f"{value}: operand replay shape {tuple(x.shape)} != {g.shape(value)}")
    return x, bias_map


# ------------------------------------------------------------------------------------------- result chains
def _output_chain(g, value, group_names, siblings):
    """Forward walk through layout/cast/copy/concatenation consumers; returns (chain, final)."""
    chain, current, final = [], value, None
    while True:
        users = [u for u in g.users(current) if u not in group_names]
        if len(users) != 1:
            break
        user = users[0]
        base = g.base(user)
        if base in _VIEWS:
            pass
        elif base in _COPIES or g.cast_dtype(user) in ("float32", "bfloat16"):
            final = user
        elif base == "cat":
            parts = [g.ref(x, g.names) for x in g.op(user)["args"][0]]
            if any(_reaches(g, p, siblings) is None for p in parts if p != current):
                raise Refused(f"{user}: concatenation with values outside the attention gradients")
            final = user
        else:
            break
        chain.append(user)
        current = user
    if final is None:
        raise Refused(f"{value}: result needs a materializing consumer (copy, cast or concatenation)")
    chain = chain[:chain.index(final) + 1]
    return chain, final


def _reaches(g, value, targets):
    """The target reached from value backward through layout/cast/copy operations, else None."""
    current = value
    while current is not None and current not in targets:
        base = g.base(current)
        if base in _VIEWS or base in _COPIES or g.cast_dtype(current) in ("float32", "bfloat16"):
            current = g.arg(current, 0)
        else:
            return None
    return current


def _replay_forward(g, value, chain):
    """Replay a result chain on logical result indices (-1 marks sibling concatenation inputs)."""
    torch = _torch()
    shape = g.shape(value)
    x = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    for name in chain:
        if g.base(name) == "cat":
            parts = []
            for item in g.op(name)["args"][0]:
                ref = g.ref(item, g.names)
                parts.append(x if ref == chain_input(chain, name, value) else
                             torch.full(g.shape(ref), -1, dtype=torch.int64))
            dim = int(g.op(name)["args"][1]) if len(g.op(name)["args"]) > 1 else 0
            x = torch.cat(parts, dim)
        else:
            x = _replay(g, name, x, g.shape(name))
    return x


def _reduction_map(g, value, chain):
    """Logical result element -> flat index of a sum over all but the last dimension of chain[-1]."""
    torch = _torch()
    x = _replay_forward(g, value, chain)
    total = math.prod(g.shape(value))
    flat = x.reshape(-1)
    valid = flat >= 0
    if int(valid.sum()) != total:
        raise Refused(f"{value}: reduced elements are duplicated or dropped")
    last = x.shape[-1]
    out = torch.full((total,), -1, dtype=torch.int64)
    out[flat[valid]] = (torch.arange(flat.numel(), dtype=torch.int64) % last)[valid]
    return out.reshape(g.shape(value))


def _absorb_consumers(g, final, protected):
    """Consumers of an FP32 gradient materialization reached through views: exactly one BF16 cast (the next
    products' operand) and any number of sums over all but the last dimension (bias gradients).
    Returns (path to the cast [views..., cast], [(sum, path to its input [views...])]) or None."""
    frontier, paths, casts, sums = [final], {final: []}, [], []
    while frontier:
        current = frontier.pop(0)
        for user in g.users(current):
            base = g.base(user)
            path = paths[current]
            if base in ("view", "_unsafe_view", "reshape", "squeeze", "unsqueeze", "alias", "detach"):
                if user in protected:
                    return None
                paths[user] = path + [user]
                frontier.append(user)
            elif g.cast_dtype(user) == "bfloat16":
                casts.append((user, path + [user]))
            elif base == "sum" and "dim_IntList" in g.op(user)["target"]:
                args = g.op(user)["args"]
                rank = len(g.shape(current))
                dims = sorted(int(d) % rank for d in (args[1] if isinstance(args[1], (list, tuple)) else [args[1]]))
                kwargs = g.op(user)["kwargs"]
                if (dims != list(range(rank - 1)) or g.dtype(user) != "float32"
                        or (kwargs.get("dtype") not in (None, "float32", "torch.float32"))):
                    return None
                sums.append((user, path))
            else:
                return None
    if len(casts) != 1:
        return None
    path = list(casts[0][1])
    # an exact copy of the cast (a layout copy kept by the capture) materializes the same values: absorb it too
    while True:
        users = g.users(path[-1])
        if len(users) != 1:
            break
        base = g.base(users[0])
        tail = [users[0]]
        current = users[0]
        while base in ("view", "_unsafe_view", "reshape", "alias", "detach"):
            nxt = g.users(current)
            if len(nxt) != 1:
                break
            current = nxt[0]
            base = g.base(current)
            tail.append(current)
        if base in _COPIES and current not in protected and all(x not in protected for x in tail[:-1]):
            path += tail
        else:
            break
    return path, sums


def _scatter(g, value, chain, final, chains_by_value):
    """Map logical result elements -> element offsets in the final contiguous materialization."""
    torch = _torch()
    shape = g.shape(value)
    total = math.prod(shape)
    x = torch.arange(total, dtype=torch.int64).reshape(shape)
    for name in chain:
        if g.base(name) == "cat":
            parts = []
            for item in g.op(name)["args"][0]:
                ref = g.ref(item, g.names)
                parts.append(x if ref == chain_input(chain, name, value) else
                             torch.full(g.shape(ref), -1, dtype=torch.int64))
            dim = int(g.op(name)["args"][1]) if len(g.op(name)["args"]) > 1 else 0
            x = torch.cat(parts, dim)
        else:
            x = _replay(g, name, x, g.shape(name))
    flat = x.reshape(-1)
    if tuple(x.shape) != g.shape(final):
        raise Refused(f"{final}: result replay shape mismatch")
    valid = flat >= 0
    if int(valid.sum()) != total:
        raise Refused(f"{value}: result elements are duplicated or dropped by its layout chain")
    out = torch.full((total,), -1, dtype=torch.int64)
    out[flat[valid]] = torch.arange(flat.numel(), dtype=torch.int64)[valid]
    if bool((out < 0).any()):
        raise Refused(f"{value}: result elements are not all materialized")
    return out.reshape(shape)


def chain_input(chain, name, value):
    """The element of `chain` (or value) feeding `name`."""
    k = chain.index(name)
    return value if k == 0 else chain[k - 1]


# ------------------------------------------------------------------------------------------------ admission
def admit(graph_args, static_layout, protected, exclude=(), absorb=True):
    """({label: plan}, refusals) for every mixed attention group in the graph.

    static_layout(name) -> (shape, strides) of a captured value. Removal is restricted to operations whose
    every consumer is removed too; the anchors are the forward/backward result materializations.
    """
    g = _G(graph_args)
    groups, refusals = match(graph_args, exclude)
    plans = {}
    claimed = set()
    for group in groups:
        try:
            plan = _plan(g, group, static_layout, protected, absorb=absorb)
            if plan["removed"] & claimed:
                raise Refused("operations shared with another attention group")
            claimed |= plan["removed"]
            plans[group["label"]] = plan
        except Refused as exc:
            refusals.append({"group": group["label"], "reason": "mixed attention: " + str(exc)})
    return plans, refusals


def _plan(g, group, static_layout, protected, layout_of=None, absorb=True):
    layout_of = layout_of or static_layout
    bh, t, hd = group["bh"], group["t"], group["hd"]
    if backward_shared(hd, t) > SHARED_LIMIT:
        raise Refused(f"backward needs {backward_shared(hd, t)} shared bytes > {SHARED_LIMIT}")
    operands = {"q": [("q", group["q"], False), ("qt", group["qt"], True)],
                "k": [("kt", group["kt"], True), ("k_view", group["k_view"], False)],
                "v": [("v", group["v"], False), ("vt", group["vt"], True)],
                "do": [("do", group["dout"], False)]}
    descs, maps, bias_maps = {}, {}, {}
    for key, refs in operands.items():
        found = []
        for label, value, transposed in refs:
            desc = _input_chain(g, value)
            index, bias_map = _gather(g, desc, layout_of, value)
            if transposed:
                index = index.transpose(1, 2)
                bias_map = None if bias_map is None else bias_map.transpose(1, 2)
            found.append((desc, index, bias_map))
        d0, i0, b0 = found[0]
        torch = _torch()
        for d, i, b in found[1:]:
            same = (d["source"] == d0["source"] and d["kind"] == d0["kind"] and torch.equal(i, i0)
                    and (b is None) == (b0 is None) and (b is None or torch.equal(b, b0))
                    and d.get("bias") == d0.get("bias"))
            if not same:
                raise Refused(f"{key}: forward and backward read different operand values")
        descs[key], maps[key] = d0, i0
        if b0 is not None:
            bias_maps[key] = b0
    if descs["do"]["kind"] == 1:
        raise Refused("output gradient with a bias source is not admitted")
    group_names = set(group["members"])
    siblings = {group["dq"], group["dkt"], group["dv"]}
    results = {"o": group["o"], "dq": group["dq"], "dkt": group["dkt"], "dv": group["dv"]}
    chains, finals, scatters = {}, {}, {}
    for key, value in results.items():
        chain, final = _output_chain(g, value, group_names, siblings if key != "o" else set())
        chains[key], finals[key] = chain, final
        index = _scatter(g, value, chain, final, chains)
        scatters[key] = index.transpose(1, 2) if key == "dkt" else index
    # Absorb the gradient materialization's consumers: its BF16 cast becomes the result and its bias sums are
    # computed in the phase (no FP32 gradient matrix is stored).
    reductions, absorbed_sums = {}, set()
    if absorb:
        consumers = {}
        for key in ("dq", "dkt", "dv"):
            if g.dtype(finals[key]) == "float32" and finals[key] not in consumers:
                consumers[finals[key]] = _absorb_consumers(g, finals[key], protected)
        for key in ("dq", "dkt", "dv"):
            info = consumers.get(finals[key])
            if info is None:
                continue
            cast_path, sums = info
            base_chain = chains[key]
            chains[key] = base_chain + cast_path
            finals[key] = cast_path[-1]
            index = _scatter(g, results[key], chains[key], finals[key], chains)
            scatters[key] = index.transpose(1, 2) if key == "dkt" else index
            mine = []
            for total, path in sums:
                chain_to_input = base_chain + path
                rmap = _reduction_map(g, results[key], chain_to_input)
                mine.append((total, rmap.transpose(1, 2) if key == "dkt" else rmap, path))
            if len(mine) > 1:
                raise Refused(f"{key}: more than one reduction of the gradient")
            if mine:
                total, rmap, path = mine[0]
                reductions[key] = (total, rmap)
                chains[key] = chains[key] + [x for x in path if x not in chains[key]]
                absorbed_sums.add(total)
    if g.dtype(finals["o"]) not in ("bfloat16", "float32"):
        raise Refused("attention output materialization must be BF16 or FP32")
    for key in ("dq", "dkt", "dv"):
        if g.dtype(finals[key]) not in ("bfloat16", "float32"):
            raise Refused(f"{key} materialization must be BF16 or FP32")
    # one head decomposition for every operand/result map (bias maps are checked separately)
    all_maps = {**{f"in:{k}": m for k, m in maps.items()}, **{f"out:{k}": m for k, m in scatters.items()}}
    heads, strides = _heads_for(all_maps, bh)
    reduce = {}
    for key, (total, rmap) in reductions.items():
        o, sb, sh, sr = _affine(rmap, bh, heads, f"reduce:{key}")
        if sb or sr:
            raise Refused(f"{key}: reduction does not sum over batch and rows")
        reduce[key] = (total, o, sh)
    bias_strides = {}
    for key, bmap in bias_maps.items():
        torch = _torch()
        o = int(bmap[0, 0, 0])
        sb_b = int(bmap[heads, 0, 0] - o) if bh > heads else 0
        sh_b = int(bmap[1, 0, 0] - o) if heads > 1 else 0
        b = torch.arange(bh) // heads
        h = torch.arange(bh) % heads
        model = o + (b * sb_b + h * sh_b)[:, None, None] + torch.zeros(t, dtype=torch.int64)[None, :, None] \
            + torch.arange(hd)[None, None, :]
        if not torch.equal(model, bmap):
            raise Refused(f"{key}: bias map is not a broadcast unit-stride vector per head")
        if o % 4 or sb_b % 4 or sh_b % 4:
            raise Refused(f"{key}: bias rows must be 16-byte aligned")
        bias_strides[key] = (o, sb_b, sh_b)
    for key, desc in descs.items():
        unit = 8 if desc["kind"] in (0, 1) else 4
        if any(x % unit for x in strides[f"in:{key}"]):
            raise Refused(f"{key}: operand rows must be 16-byte aligned")
    for key in results:
        if any(x % 2 for x in strides[f"out:{key}"]):
            raise Refused(f"{key}: result pairs must be aligned")
    # removal: group operations, result chains (finals are rewritten by the phases), and operand chain
    # operations whose every consumer is removed (reverse dead-code closure)
    removed = set(group_names) | absorbed_sums
    for key, chain in chains.items():
        removed |= set(chain)
    finals_set = set(finals.values()) | absorbed_sums
    candidates = []
    for desc in descs.values():
        candidates += desc["chain"] + ([desc["add"]] + desc["after"] if desc["kind"] == 1 else [])
    candidates = [c for c in dict.fromkeys(candidates)]
    changed = True
    while changed:
        changed = False
        for c in sorted(candidates, key=lambda n: -g.order[n]):
            if c in removed or c in protected:
                continue
            if all(u in removed for u in g.users(c)) and g.users(c):
                removed.add(c)
                changed = True
    for name in removed:
        if name in protected and name not in finals_set:
            raise Refused(f"{name} is returned or committed")
        outside = [u for u in g.users(name) if u not in removed]
        if outside and name not in finals_set:
            raise Refused(f"{name} is used outside the attention region by {outside}")
    for key, desc in descs.items():
        if desc["source"] in removed:
            raise Refused(f"{key}: operand source is produced inside the group")
    anchor_backward = min([finals[k] for k in ("dq", "dkt", "dv")] + sorted(absorbed_sums), key=g.order.__getitem__)
    for key in ("q", "k", "v", "do"):
        for name in (descs[key]["source"], descs[key].get("bias")):
            if name is not None and name in g.ops and g.order[name] > g.order[anchor_backward]:
                raise Refused(f"{key}: source defined after the backward results")
    return dict(group=group, bh=bh, t=t, hd=hd, heads=heads, descs=descs, strides=strides,
                bias_strides=bias_strides, finals=finals, chains=chains, removed=removed, reduce=reduce,
                absorb=absorb,
                anchor_forward=finals["o"], anchor_backward=anchor_backward,
                sources={k: d["source"] for k, d in descs.items()},
                biases={k: d["bias"] for k, d in descs.items() if d["kind"] == 1})


def refresh(plan, graph_args, layout_of, protected):
    """Recompute layout maps from the lowered layouts at installation (raises Refused on any difference)."""
    g = _G(graph_args)
    fresh = _plan(g, plan["group"], layout_of, protected, layout_of=layout_of, absorb=plan["absorb"])
    if fresh["removed"] != plan["removed"] or fresh["finals"] != plan["finals"]:
        raise Refused("lowered layouts changed the absorbed region")
    return fresh


def stats_name(group):
    return group["o"] + "__mixed_stats"


def aux_names(group):
    return {"counters": group["o"] + "__mixed_counters", "partials": group["o"] + "__mixed_partials",
            "gpartials": group["o"] + "__mixed_gpartials"}


def reduction_layout(plan):
    """(batches, group size, groups, counters) of the absorbed bias reductions, or None."""
    if not plan.get("reduce"):
        return None
    batches = plan["bh"] // plan["heads"]
    size = max(1, int(math.ceil(math.sqrt(batches))))
    groups = (batches + size - 1) // size
    return batches, size, groups, plan["heads"] * groups + plan["heads"]


def emit(plan, ident, dtype_of, *, which, bias_phase=True, trace=False, store_probs=False, single_only=False):
    group, bh, t, hd, heads = plan["group"], plan["bh"], plan["t"], plan["hd"], plan["heads"]
    tp = _tp(t)
    st = plan["strides"]
    long_context = is_long(t)
    if long_context and trace:
        raise Refused("trace stamps are defined for T <= 128 items only")
    if long_context:
        store_probs = False  # long items recompute probabilities from the saved (m, l); nothing T x T is stored

    def operand(key):
        desc = plan["descs"][key]
        o_src, sb, sh, sr = st[f"in:{key}"]
        bias = "nullptr"
        bb = bhh = 0
        if desc["kind"] == 1:
            o, bb, bhh = plan["bias_strides"][key]
            bias = f"{ident(desc['bias'])} + {o}LL"
        field = "dout" if key == "do" else key
        return (f"A.{field} = pmix::Operand{{{ident(desc['source'])} + {o_src}LL, {bias}, {sb}LL, {sh}LL, {sr}LL, {bb}LL, "
                f"{bhh}LL, {desc['kind']}}};")

    def result(key, field):
        o_dst, sb, sh, sr = st[f"out:{key}"]
        final = plan["finals"][key]
        fp32 = 1 if dtype_of(final) == "float32" else 0
        return f"A.{field} = pmix::Result{{{ident(final)} + {o_dst}LL, {sb}LL, {sh}LL, {sr}LL, {fp32}}};"

    fields = [operand("q"), operand("k"), operand("v"),
              f"A.heads = {heads}; A.t = {t}; A.scale = {repr(float(group['scale']))}f; "
              f"A.scale_mode = {group['scale_mode']};",
              f"A.stats = {ident(stats_name(group))};"]
    layout = reduction_layout(plan)
    aux = aux_names(group)
    aux_reads = []
    if layout and not bias_phase:
        batches, size, groups, counters = layout
        fields.append(f"A.counters = reinterpret_cast<unsigned long long*>({ident(aux['counters'])}); "
                      f"A.ncounters = {counters}; A.batches = {batches}; A.group = {size};")
    elif layout:
        fields.append(f"A.batches = {layout[0]}; A.group = {layout[1]};")
    trace_name = group["o"] + "__mixed_trace"
    probs_name = group["o"] + "__mixed_probs"
    if store_probs:
        fields.append(f"A.probs = {ident(probs_name)};")
        if which == "backward":
            aux_reads.append(probs_name)
    if trace:
        fields.append(f"A.trace = reinterpret_cast<long long*>({ident(trace_name)});")
        if which == "backward":
            aux_reads.append(trace_name)
    if which == "forward":
        fields.append(result("o", "o"))
        body = (f"pmix::forward_long_phase<{hd}, {tp}>(A, {bh}LL, smem);" if long_context else
                f"pmix::forward_phase<{hd}, {tp}>(A, {bh}LL, smem);")
        shared = forward_shared(hd, t, single_only)
        outputs = {plan["finals"]["o"]: None, stats_name(group): ((bh, t, 2), "float32")}
        if layout and not bias_phase:
            outputs[aux["counters"]] = ((layout[3],), "int64")
        if trace:
            outputs[trace_name] = ((2 * 8 * 16,), "int64")
        if store_probs:
            outputs[probs_name] = ((bh * (tp // 16) * (tp // 8) * 32 * 4,), "float32")
    else:
        fields += [operand("do"), result("dq", "dq"), result("dkt", "dk"), result("dv", "dv")]
        if long_context and FA_DELTA:
            o_dst, osb, osh, osr = st["out:o"]
            okind = 2 if dtype_of(plan["finals"]["o"]) == "float32" else 0
            fields.append(f"A.ofwd = pmix::Operand{{{ident(plan['finals']['o'])} + {o_dst}LL, nullptr, {osb}LL, {osh}LL, "
                          f"{osr}LL, 0LL, 0LL, {okind}}};")
            aux_reads.append(plan["finals"]["o"])
        if long_context and LONG_STORE_PDS:
            fields.append(f"A.pds = reinterpret_cast<__nv_bfloat16*>({ident(group['o'] + '__mixed_pds')});")
        stage_split = long_context and LONG_STORE_PDS and LONG_STAGE_SPLIT and not (layout and not bias_phase)
        if stage_split:
            # per-item flags in the program's zeroed split scratch (restored to zero by the key stage)
            if LONG_V3 and _tp(t) // 16 > 8:
                fields.append("A.flags = /*P2_CNT*/; A.halves = /*P2_PART*/;")
            else:
                fields.append("A.flags = /*P2_CNT*/; (void)(/*P2_PART*/);")
        body = (f"pmix::backward_long_phase<{hd}, {tp}>(A, {bh}LL, smem);" if long_context else
                f"pmix::backward_phase<{hd}, {tp}>(A, {bh}LL, smem);" if not single_only and backward_pipelined(hd, t) else
                f"pmix::backward_phase<{hd}, {tp}, false>(A, {bh}LL, smem);")
        shared = backward_shared(hd, t, single_only)
        outputs = {}
        for key in ("dq", "dkt", "dv"):
            outputs.setdefault(plan["finals"][key], None)
        if long_context and LONG_STORE_PDS:
            outputs[group["o"] + "__mixed_pds"] = ((bh, 2, tp, tp), "bfloat16")
        if layout:
            batches, size, groups, counters = layout
            for slot, key in enumerate(("dq", "dkt", "dv")):
                if key in plan["reduce"]:
                    total, o, sh = plan["reduce"][key]
                    fields.append(f"A.red[{slot}] = pmix::Reduce{{{ident(total)}, {o}LL, {sh}LL}};")
                    if not bias_phase:
                        outputs.setdefault(total, None)
            fields.append(f"A.partials = {ident(aux['partials'])};")
            outputs[aux["partials"]] = ((batches, 3, heads, hd), "float32")
            if not bias_phase:
                fields.append(f"A.gpartials = {ident(aux['gpartials'])};")
                outputs[aux["gpartials"]] = ((groups, 3, heads, hd), "float32")
                aux_reads.append(aux["counters"])
    code = "pmix::Args A{};\n" + "\n".join(fields) + "\n" + body
    if which == "backward" and layout and bias_phase:
        bias_fields = [f for f in fields if f.startswith(("A.red[", "A.heads", "A.batches"))]
        bias_fields.append(f"A.partials = {ident(aux['partials'])};")
        bias_code = "pmix::Args A{};\n" + "\n".join(bias_fields) + f"\npmix::bias_phase<{hd}>(A, smem);"
        bias_outputs = {total: None for total, _, _ in plan["reduce"].values()}
        extra_bias = dict(work_items=(3 * heads * hd) // 8, shared_bytes=4 * 32 * 8, outputs=bias_outputs,
                          aux_reads=[aux["partials"]], head_width=hd, heads=heads)
    else:
        bias_code = extra_bias = None
    split_units = which == "backward" and stage_split
    extra = dict(work_items=2 * bh if split_units else bh, shared_bytes=shared, head_width=hd, sequence=t,
                 head_batch=bh, heads=heads, engine="long" if long_context else "short",
                 outputs=outputs, aux_reads=aux_reads,
                 reductions={k: v[0] for k, v in plan.get("reduce", {}).items()},
                 bias_phase=(bias_code, extra_bias) if bias_code else None)
    if split_units:
        from .vendor import plan as arena_plan
        v3 = LONG_V3 and tp // 16 > 8
        extra["p2_split"] = {"counter_words": arena_plan.a64(3 * bh + 2 if v3 else bh),
                             "partial_words": bh * 2 * 3 * 8 * hd if v3 else 0}
        extra["long_stage_split"] = True
    return "attention_mixed_" + which, code, extra
