"""P8-C: rectangular masked mixed-precision attention as opaque operations (capture rewrite + lowering rule).

The ``mixed_bf16`` precision policy turns an explicitly written attention

    S  = bmm(Qb, KTb)                    BF16 operands and result      [Z, M, N]
    X  = view*(add(view*(S.float() op c), mask))                       (views only regroup leading dims)
    P  = _softmax(X, -1)                                               FP32
    O  = bmm(P.bfloat16(), Vb)                                         BF16 [Z, M, D]

into ordinary ATen operations; PyTorch's derivative of it (dV = bmm(Pb^T, dOb), dP = bmm(dOb, Vb^T).float(),
dX = _softmax_backward_data(dP, P), dS = (view*(dX) op c).bfloat16(), dQ = bmm(dS, KTb^T), dKT = bmm(Qb^T, dS))
follows in the joint primal/VJP graph. Written as such, every [Z, M, N] score/probability/gradient matrix is
materialized (block-local sliding-window attention: 268 MB per layer in the S106B m8k learner) and the
probabilities become saved tape values.

``rewrite_joint`` recognizes that structure by graph references, dtypes and shapes only (no names) in a joint
FX graph and replaces it with the opaque operations below, which never form the matrices:

    table          = mattn_table(mask)                          (once per mask and grouping)
    O, stats       = mattn_fwd(Qb, KTb, Vb, mask, table)        stats [Z, M, 2]: row max m, l = sum exp(x - m)
    dQ, dKT, dV    = mattn_bwd(Qb, KTb, Vb, mask, table, dOb, stats)

Their eager definitions below are the executable semantics (CPU graph replay and reference interpretation):
P = exp(x - m) / l from the stored statistics, delta = sum_j P_j dP_j as _softmax_backward_data, and every rounding
point of the captured mixed graph. They differ from the replaced graph only by FP32 association (the softmax
normalization is recomputed from (m, l) instead of reusing the forward P). The tape partitioner then saves the
BF16 operands and the [Z, M, 2] statistics instead of [Z, M, N] probabilities. Device lowering uses
kernels/attention_window.cuh (head width 32, M % 64 == 0, N % 64 == 0, M, N <= 512); anything else is not
rewritten and keeps the ordinary lowering, with a recorded refusal.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

HEADER = 'attention_window.cuh'
INCLUDED_HEADERS = ('attention_window_v4.cuh',)   # included by HEADER (hashed with it)
SCHEMA = 'p8c-window-attention-v1'
HEAD_WIDTHS = (32,)
MAX_EXTENT = 512
KEY_BLOCK = 64
TARGETS = {f'latch_ops.mattn_{kind}.default': kind for kind in ('table', 'fwd', 'bwd')}
MULTI = ('latch_ops.mattn_fwd.default', 'latch_ops.mattn_bwd.default')
SHARED_LIMIT = 133120  # keep the program inside the H100 L1 carve-out (P-OPS-D / P-OPS3-B finding)
ITEM_SHARED = lambda m, n: 2 * (2 * m + 2 * n) * 40 + 4 * (5 * m + n + n // KEY_BLOCK)  # kernels: item_shared_bytes


# --------------------------------------------------------------------------------------- eager definitions
def _group(group, z, m, n):
    return list(group) if group else [z, m, n]


def _b(x):
    return x.to(torch.bfloat16)


def _scores(q, kt, mask, scale, mode, group):
    s = torch.bmm(_b(q), _b(kt)).float()
    if mode == 1:
        s = s * scale
    elif mode == 2:
        s = s / scale
    if mask is not None:
        s = (s.view(_group(group, *s.shape)) + mask).view(s.shape)
    return s


def _probs(x, stats):
    return torch.exp(x - stats[..., 0:1]) / stats[..., 1:2]


# ---- declared online-softmax arithmetic (scale mode bit ONLINE; policy mixed_bf16_online, see the kernel header)
ONLINE = 4


def _cmul(scale, mode):
    c = torch.tensor(float(scale), dtype=torch.float32)
    if mode & 3 == 1:
        return c
    return torch.ones((), dtype=torch.float32) / c if mode & 3 == 2 else torch.ones((), dtype=torch.float32)


def _scores_online(q, kt, mask, scale, mode, group):
    s = torch.bmm(_b(q), _b(kt)).float() * _cmul(scale, mode).to(q.device)
    if mask is not None:
        s = (s.view(_group(group, *s.shape)) + mask).view(s.shape)
    return s


def _block_classes(mask, group, Z, M, N, device):
    """[Z, M, N/64] booleans: True where the row's (16-row, 64-key) mask block is entirely below -1e30 (class 2)."""
    if mask is None:
        return torch.zeros((Z, M, N // KEY_BLOCK), dtype=torch.bool, device=device)
    g = _group(group, Z, M, N)
    full = mask.expand(*g).reshape(Z, M // 16, 16, N // KEY_BLOCK, KEY_BLOCK)
    return (full.amax(dim=(2, 4)) < -1e30).repeat_interleave(16, dim=1)


def _online_forward(q, kt, v, mask, scale, mode, group):
    """Online softmax over 64-key blocks: per row, unmasked blocks ascending, then fully masked blocks ascending."""
    x = _scores_online(q, kt, mask, scale, mode, group)
    Z, M, N = x.shape
    cls = _block_classes(mask, group, Z, M, N, x.device)
    m = torch.full((Z, M), -math.inf, dtype=torch.float32, device=x.device)
    l = torch.zeros((Z, M), dtype=torch.float32, device=x.device)
    o = torch.zeros((Z, M, v.shape[2]), dtype=torch.float32, device=x.device)
    vb = _b(v)
    for second in (False, True):
        for kb in range(N // KEY_BLOCK):
            act = cls[:, :, kb] == second
            xb = x[:, :, kb * KEY_BLOCK:(kb + 1) * KEY_BLOCK]
            mn = torch.where(act, torch.maximum(m, xb.amax(-1)), m)
            alpha = torch.where(act, torch.where(m == -math.inf, torch.zeros_like(m), torch.exp(m - mn)),
                                torch.ones_like(m))
            p = torch.where(act[..., None], torch.exp(xb - mn[..., None]), torch.zeros_like(xb))
            l = l * alpha + p.sum(-1)
            o = o * alpha[..., None] + torch.bmm(p.to(torch.bfloat16).float(), vb[:, kb * KEY_BLOCK:(kb + 1) * KEY_BLOCK].float())
            m = mn
    rl = torch.ones((), dtype=torch.float32, device=x.device) / l
    return (o * rl[..., None]).to(torch.bfloat16).contiguous(), torch.stack((m, l), -1).contiguous()


def _online_backward(q, kt, v, mask, dout, stats, o, scale, mode, group):
    x = _scores_online(q, kt, mask, scale, mode, group)
    rl = torch.ones((), dtype=torch.float32, device=x.device) / stats[..., 1:2]
    p = torch.exp(x - stats[..., 0:1]) * rl
    dp = _dp(dout, v)
    delta = (_b(dout).float() * _b(o).float()).sum(-1, keepdim=True)
    ds = ((p * (dp - delta)) * _cmul(scale, mode).to(x.device)).to(torch.bfloat16)
    return p, ds


TABLE_WORDS = 20   # kernels/attention_window.cuh TE


def _table_shape(mask, group):
    lead = list(group[:-2])
    ext = [1] * (len(lead) - (mask.dim() - 2)) + list(mask.shape[:-2])
    ext = [1] * (2 - len(ext)) + ext
    return (*ext, mask.shape[-2] // 16, mask.shape[-1] // KEY_BLOCK, TABLE_WORDS)


def _float_bits(x):
    return x.contiguous().view(torch.int32).to(torch.int64) & 0xffffffff


@torch.library.custom_op('latch_ops::mattn_table', mutates_args=())
def mattn_table(mask: torch.Tensor, group: List[int]) -> torch.Tensor:
    """Per (mask batch, 16-row block, 64-key block), int64 words: bits of the maximum mask value; 1 if every element is
    +-0; 1 if every element is +-0 or the block minimum v; bits of v; one bitmap per row (bit j: element j is +-0)."""
    shape = _table_shape(mask, group)
    m = mask.reshape(shape[0], shape[1], shape[2], 16, shape[3], KEY_BLOCK)
    mx, mn = m.amax(dim=(3, 5)), m.amin(dim=(3, 5))
    zero = (m == 0).all(dim=5).all(dim=3)
    binary = ((m == 0) | (m == mn[:, :, :, None, :, None])).all(dim=5).all(dim=3)
    weights = torch.ones((), dtype=torch.int64, device=mask.device) << torch.arange(KEY_BLOCK, device=mask.device)
    bits = ((m == 0).to(torch.int64) * weights).sum(dim=5).permute(0, 1, 2, 4, 3)
    head = torch.stack((_float_bits(mx), zero.to(torch.int64), binary.to(torch.int64), _float_bits(mn)), -1)
    return torch.cat((head, bits), -1).contiguous()


@mattn_table.register_fake
def _(mask, group):
    return torch.empty(_table_shape(mask, group), dtype=torch.int64, device=mask.device)


@torch.library.custom_op('latch_ops::mattn_fwd', mutates_args=())
def mattn_fwd(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor],
              table: Optional[torch.Tensor], scale: float, mode: int, group: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (O, stats): O = bf16(bf16(P) v) and stats [Z, M, 2] = (row max m, l = sum exp(x - m))."""
    if mode & ONLINE:
        return _online_forward(q, kt, v, mask, scale, mode, group)
    x = _scores(q, kt, mask, scale, mode, group)
    m = x.amax(-1, keepdim=True)
    l = torch.exp(x - m).sum(-1, keepdim=True)
    stats = torch.cat((m, l), -1).contiguous()
    return torch.bmm(_probs(x, stats).to(torch.bfloat16), _b(v)).contiguous(), stats


@mattn_fwd.register_fake
def _(q, kt, v, mask, table, scale, mode, group):
    return (q.new_empty((q.shape[0], q.shape[1], v.shape[2]), dtype=torch.bfloat16),
            q.new_empty((q.shape[0], q.shape[1], 2), dtype=torch.float32))


def _dp(dout, v):
    return torch.bmm(_b(dout), _b(v).transpose(1, 2)).float()


def _dk_strides(z, d, n):
    return (n * d, 1, d)   # K-major storage of dK^T: [Z][N][D] memory, logical [Z, D, N]


@torch.library.custom_op('latch_ops::mattn_bwd', mutates_args=())
def mattn_bwd(q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor],
              table: Optional[torch.Tensor], dout: torch.Tensor, stats: torch.Tensor, scale: float, mode: int,
              group: List[int], wide: List[int], o: Optional[torch.Tensor] = None
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (dQ, dK^T, dV): delta = sum_j P_j dP_j, dS = bf16((P (dP - delta)) op c), dQ = bf16(dS k),
    dK^T = bf16(q^T dS) stored K-major, dV = bf16(bf16(P)^T dO); result r widened exactly to FP32 if wide[r].
    Online arithmetic (mode & ONLINE): delta = dO . o with o the forward output (see _online_backward)."""
    if mode & ONLINE:
        p, ds = _online_backward(q, kt, v, mask, dout, stats, o, scale, mode, group)
    else:
        p = _probs(_scores(q, kt, mask, scale, mode, group), stats)
        dp = _dp(dout, v)
        ds = p * (dp - (p * dp).sum(-1, keepdim=True))
        if mode == 1:
            ds = ds * scale
        elif mode == 2:
            ds = ds / scale
        ds = ds.to(torch.bfloat16)
    dq = torch.bmm(ds, _b(kt).transpose(1, 2)).contiguous()
    dkt = torch.bmm(ds.transpose(1, 2), _b(q)).contiguous().transpose(1, 2)
    dv = torch.bmm(p.to(torch.bfloat16).transpose(1, 2), _b(dout)).contiguous()
    return tuple(x.float() if w else x for x, w in zip((dq, dkt, dv), wide))


@mattn_bwd.register_fake
def _(q, kt, v, mask, table, dout, stats, scale, mode, group, wide, o=None):
    dt = lambda r: torch.float32 if wide[r] else torch.bfloat16
    return (q.new_empty(q.shape, dtype=dt(0)),
            torch.empty_strided(tuple(kt.shape), _dk_strides(*kt.shape), dtype=dt(1), device=kt.device),
            v.new_empty(v.shape, dtype=dt(2)))


# ------------------------------------------------------------------------------------------ joint rewrite
_VIEWS = {'view', '_unsafe_view', 'reshape'}
_TRANSPARENT = {'detach', 'alias'}


class Refused(Exception):
    pass


def _base(node):
    if node.op != 'call_function':
        return None
    return str(node.target).removeprefix('aten.').split('.')[0]


def _meta(node):
    value = node.meta.get('val')
    return value if isinstance(value, torch.Tensor) else None


def _is_cast(node, dtype):
    if _base(node) != '_to_copy':
        return False
    out = _meta(node)
    return out is not None and out.dtype == dtype and set(node.kwargs) <= {'dtype', 'memory_format', 'layout', 'device'}


def _users(node):
    return list(node.users)


def _scalar(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _through(node, kinds):
    """Follow single-user chains of `kinds` forward; returns (last node, chain)."""
    chain = []
    while True:
        users = _users(node)
        if len(users) != 1 or _base(users[0]) not in kinds or users[0].args[0] is not node:
            return node, chain
        node = users[0]
        chain.append(node)


def _transposed_of(node, source):
    return (_base(node) == 'transpose' and node.args[0] is source and
            {int(node.args[1]) % 3, int(node.args[2]) % 3} == {1, 2})


def _match(sm):
    """Match one mixed attention around softmax node ``sm``. Raises Refused or returns a record."""
    if len(sm.args) < 3 or int(sm.args[1]) % _meta(sm).dim() != _meta(sm).dim() - 1 or bool(sm.args[2]):
        raise Refused('softmax is not an FP32 last-axis softmax')
    # Backward from the softmax input through views, the mask addition, views and the scale to the product.
    x = sm.args[0]
    after, cur = [], x
    while _base(cur) in _VIEWS:
        after.append(cur)
        cur = cur.args[0]
    add = mask = None
    if _base(cur) == 'add':
        a, b = cur.args[0], cur.args[1]
        if len(cur.args) > 2 or cur.kwargs.get('alpha', 1) != 1:
            raise Refused('mask addition has alpha')
        add = cur
        # The score operand is the one reached from a bmm through views/scale/cast; the other is the mask.
        cur, mask = (a, b) if _reaches_product(a) else (b, a)
        if not isinstance(mask, torch.fx.Node) or _meta(mask) is None or _meta(mask).dtype != torch.float32:
            raise Refused('mask is not an FP32 tensor')
    before = []
    while _base(cur) in _VIEWS:
        before.append(cur)
        cur = cur.args[0]
    scale_node, scale, mode = None, 1.0, 0
    if _base(cur) in ('mul', 'div') and _scalar(cur.args[1]) and not cur.kwargs:
        scale_node, scale, mode = cur, float(cur.args[1]), 1 if _base(cur) == 'mul' else 2
        cur = cur.args[0]
    if not _is_cast(cur, torch.float32):
        raise Refused('scores are not the FP32 widening of a BF16 product')
    widen = cur
    sb = widen.args[0]
    if _base(sb) != 'bmm' or _meta(sb).dtype != torch.bfloat16:
        raise Refused('scores are not a BF16 batched product')
    qb, ktb = sb.args
    Z, M, N = _meta(sb).shape
    D = _meta(qb).shape[2]
    for n in (*after, *before):
        shape = tuple(_meta(n).shape)
        if shape[-2:] != (M, N) or math.prod(shape[:-2]) != Z:
            raise Refused('views around the mask regroup more than leading dimensions')
    group = list(_meta(add).shape) if add is not None else []
    if add is not None:
        if len(group) - 2 > 2:
            raise Refused('more than two leading mask dimensions')
        ms = tuple(_meta(mask).shape)
        if ms[-2:] != (M, N) or len(ms) > len(group):
            raise Refused('mask must cover the full [M, N] score block')
        lead = [1] * (len(group) - len(ms)) + list(ms[:-2])
        if any(e not in (1, g) for e, g in zip(lead, group[:-2])):
            raise Refused('mask leading dimensions do not broadcast')
    # Every intermediate must be consumed only inside the group.
    chain = [sb, widen] + ([scale_node] if scale_node else []) + list(reversed(before)) + ([add] if add else []) + list(reversed(after))
    for a_, b_ in zip(chain, chain[1:]):
        if _users(a_) != [b_]:
            raise Refused(f'{a_.name} has consumers outside the score chain')
    if _users(chain[-1]) != [sm]:
        raise Refused('softmax input has other consumers')
    # Forward outputs and the derived backward.
    pb = sbd = None
    p_users = _users(sm)
    for u in p_users:
        if _is_cast(u, torch.bfloat16) and pb is None:
            pb = u
        elif _base(u) in _TRANSPARENT:
            t, _ = _through(u, _TRANSPARENT)
            t_users = _users(t)
            if len(t_users) != 1 or _base(t_users[0]) != '_softmax_backward_data':
                raise Refused('probabilities have consumers outside the derived backward')
            sbd = t_users[0]
        elif _base(u) == '_softmax_backward_data':
            sbd = u
        else:
            raise Refused(f'probabilities have consumer {u.name}')
    if pb is None:
        raise Refused('probabilities are not rounded to BF16 for the value product')
    ob = pbt = dvb = None
    for u in _users(pb):
        if _base(u) == 'bmm' and u.args[0] is pb:
            ob = u
        elif _transposed_of(u, pb):
            pbt = u
        else:
            raise Refused('BF16 probabilities have other consumers')
    if ob is None:
        raise Refused('no value product')
    vb = ob.args[1]
    record = dict(qb=qb, ktb=ktb, vb=vb, mask=mask, group=group, scale=scale, mode=mode, Z=Z, M=M, N=N, D=D,
                  ob=ob, removed=[*chain, sm, pb, ob], forward_only=sbd is None)
    if sbd is None:
        if pbt is not None:
            raise Refused('transposed probabilities without a softmax derivative')
        return record
    if pbt is None or len(_users(pbt)) != 1 or _base(_users(pbt)[0]) != 'bmm':
        raise Refused('no dV product')
    dvb = _users(pbt)[0]
    dob = dvb.args[1]
    dp, p_arg = sbd.args[0], sbd.args[1]
    if not _is_cast(dp, torch.float32) or _base(dp.args[0]) != 'bmm' or _users(dp) != [sbd]:
        raise Refused('dP is not the FP32 widening of a BF16 product')
    dpb = dp.args[0]
    if dpb.args[0] is not dob or not _transposed_of(dpb.args[1], vb) or _users(dpb) != [dp]:
        raise Refused('dP is not bmm(dO, V^T)')
    vt = dpb.args[1]
    # Backward views, scale and the BF16 rounding of dS.
    cur, bviews = sbd, []
    while len(_users(cur)) == 1 and _base(_users(cur)[0]) in _VIEWS:
        cur = _users(cur)[0]
        bviews.append(cur)
    dscale = None
    if mode:
        users = _users(cur)
        if (len(users) != 1 or _base(users[0]) != ('mul' if mode == 1 else 'div') or users[0].args[0] is not cur
                or float(users[0].args[1]) != scale):
            raise Refused('score-gradient scaling differs from the forward')
        dscale = cur = users[0]
    users = _users(cur)
    if len(users) != 1 or not _is_cast(users[0], torch.bfloat16):
        raise Refused('score gradient is not rounded to BF16')
    dsb = users[0]
    dqb = dktb = kt_t = q_t = None
    for u in _users(dsb):
        if _base(u) == 'bmm' and u.args[0] is dsb and _transposed_of(u.args[1], ktb):
            dqb, kt_t = u, u.args[1]
        elif _base(u) == 'bmm' and u.args[1] is dsb and _transposed_of(u.args[0], qb):
            dktb, q_t = u, u.args[0]
        else:
            raise Refused('BF16 score gradient has other consumers')
    if dqb is None or dktb is None:
        raise Refused('missing dQ or dK^T product')
    removed = record['removed'] + [pbt, dvb, dpb, dp, sbd, *bviews] + ([dscale] if dscale else []) + [dsb, dqb, dktb]
    removed += [n for n in (vt, kt_t, q_t) if all(u in removed for u in _users(n))]
    for n in list(p_users):
        if _base(n) in _TRANSPARENT:
            t = n
            while _base(t) in _TRANSPARENT:
                removed.append(t)
                t = _users(t)[0] if _users(t) and _base(_users(t)[0]) in _TRANSPARENT else t
                if t is removed[-1]:
                    break
    record.update(dob=dob, dvb=dvb, dqb=dqb, dktb=dktb, removed=removed, forward_only=False)
    return record


def _reaches_product(node):
    cur = node
    while _base(cur) in _VIEWS:
        cur = cur.args[0]
    if _base(cur) in ('mul', 'div') and _scalar(cur.args[1]):
        cur = cur.args[0]
    return _is_cast(cur, torch.float32) and _base(cur.args[0]) == 'bmm'


def admit(record):
    if record['D'] not in HEAD_WIDTHS:
        raise Refused(f"head width {record['D']} outside {HEAD_WIDTHS}")
    M, N = record['M'], record['N']
    if M % 64 or N % KEY_BLOCK or not 0 < M <= MAX_EXTENT or not 0 < N <= MAX_EXTENT:
        raise Refused(f'extents M={M}, N={N} outside M % 64 == N % 64 == 0, <= {MAX_EXTENT}')
    if ITEM_SHARED(M, N) > SHARED_LIMIT:
        raise Refused(f'one item needs {ITEM_SHARED(M, N)} shared bytes > {SHARED_LIMIT}')
    if record['forward_only']:
        raise Refused('forward-only attention keeps the existing lowering')
    for key in ('qb', 'ktb', 'vb', 'dob'):
        if _meta(record[key]).dtype != torch.bfloat16:
            raise Refused(f'{key} is not BF16')
    if tuple(_meta(record['vb']).shape) != (record['Z'], N, record['D']):
        raise Refused('value shape differs from keys')


def _absorb_casts(graph, fwd, bwd, results, removed):
    """Operand BF16 casts read only by the attention operations become rounding on load (their FP32 source is the
    operand); a result's only reader being an exact FP32 widening becomes a widening store. Same values."""
    ours = {fwd, bwd}
    absorbed = {'operands': [], 'results': []}
    for position in (0, 1, 2, 5):
        if position == 5:
            x = bwd.args[5]
        else:
            x = fwd.args[position]
        if not _is_cast(x, torch.bfloat16) or _meta(x.args[0]) is None or _meta(x.args[0]).dtype != torch.float32:
            continue
        if any(u not in ours and u not in removed for u in x.users):
            continue
        source = x.args[0]
        for node in ours:
            node.args = tuple(source if a is x else a for a in node.args)
        absorbed['operands'].append(position)
    wide = [0, 0, 0]
    for k, g in enumerate(results):
        users = list(g.users)
        if len(users) == 1 and _is_cast(users[0], torch.float32) and users[0].args[0] is g:
            widen = users[0]
            widen.replace_all_uses_with(g)
            graph.erase_node(widen)
            wide[k] = 1
            absorbed['results'].append(k)
    bwd.args = (*bwd.args[:10], wide, *bwd.args[11:])
    return absorbed


def rewrite_joint(module, arithmetic='exact'):
    """Rewrite admitted groups of an FX joint graph in place. Returns (records, refusals). ``arithmetic``: 'exact'
    (rounding points of the mixed graph) or 'online' (declared online-softmax arithmetic, policy mixed_bf16_online)."""
    if arithmetic not in ('exact', 'online'):
        raise ValueError(f'unknown attention arithmetic {arithmetic!r}')
    graph = module.graph
    records, refusals = [], []
    tables = {}
    ops = torch.ops.latch_ops
    for sm in [n for n in graph.nodes if _base(n) == '_softmax']:
        if sm.graph is not graph or sm not in graph.nodes:
            continue
        try:
            record = _match(sm)
            admit(record)
        except Refused as exc:
            refusals.append({'softmax': sm.name, 'reason': str(exc)})
            continue
        mode = record['mode'] | (ONLINE if arithmetic == 'online' else 0)
        args = (record['scale'], mode, record['group'])
        mask = record['mask']
        table = None
        if mask is not None:
            key = (mask, tuple(record['group']))
            if key not in tables:
                with graph.inserting_after(mask):
                    tables[key] = graph.call_function(ops.mattn_table.default, (mask, record['group']))
            table = tables[key]
        qb, ktb, vb, dob = record['qb'], record['ktb'], record['vb'], record['dob']
        import operator
        with graph.inserting_before(record['ob']):
            fwd = graph.call_function(ops.mattn_fwd.default, (qb, ktb, vb, mask, table, *args))
            out = graph.call_function(operator.getitem, (fwd, 0))
            stats = graph.call_function(operator.getitem, (fwd, 1))
        record['ob'].replace_all_uses_with(out)
        first = min((record['dvb'], record['dqb'], record['dktb']), key=lambda n: list(graph.nodes).index(n))
        with graph.inserting_before(first):
            bwd = graph.call_function(ops.mattn_bwd.default, (qb, ktb, vb, mask, table, dob, stats, *args, [0, 0, 0],
                                                              *((out,) if arithmetic == 'online' else ())))
            dq = graph.call_function(operator.getitem, (bwd, 0))
            dk = graph.call_function(operator.getitem, (bwd, 1))
            dv = graph.call_function(operator.getitem, (bwd, 2))
        record['dqb'].replace_all_uses_with(dq)
        record['dktb'].replace_all_uses_with(dk)
        record['dvb'].replace_all_uses_with(dv)
        absorbed = _absorb_casts(graph, fwd, bwd, (dq, dk, dv), removed=set(record['removed']))
        records.append({'softmax': sm.name, 'Z': record['Z'], 'M': record['M'], 'N': record['N'], 'D': record['D'],
                        'group': record['group'], 'scale': record['scale'], 'scale_mode': record['mode'],
                        'arithmetic': arithmetic,
                        'mask': mask.name if mask is not None else None,
                        'absorbed_casts': absorbed,
                        'nodes': {'fwd': fwd.name, 'bwd': bwd.name,
                                  'table': table.name if table is not None else None}})
    if records:
        graph.eliminate_dead_code()
        graph.lint()
        module.recompile()
        from .loop_attention import _propagate
        _propagate(module)
    return records, refusals


# ------------------------------------------------------------------------------------------------ lowering
def _dtype_of(value, field):
    return str(field(value, 'dtype')).removeprefix('torch.')


def _op3(layout, ident, name, transpose=False, fp32=False):
    """pwin::Op of a rank-3 BF16 (or FP32, rounded on load) value; transpose swaps the last two axes (K^T -> K)."""
    sz, s1, s2 = layout.strides
    sr, sd = (s2, s1) if transpose else (s1, s2)
    pointer = f"{ident(layout.root)} + {layout.offset}LL" if layout.offset else ident(layout.root)
    return (f"pwin::Op{{reinterpret_cast<const __nv_bfloat16*>({pointer}), {sz}LL, {sr}LL, {sd}LL, "
            f"{1 if fp32 else 0}}}")


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """Lower one mattn operation to a phase. ``define(name, shape, strides, dtype)`` registers a result.
    Returns (kind, code, metadata, reads, writes)."""
    target = str(field(op, 'target', field(op, 'op'))).removeprefix('torch.ops.')
    kind = TARGETS[target]
    args = list(field(op, 'args', []))
    results = list(field(op, 'outputs', None) or [field(op, 'name')])
    refs = [ref(a, names) for a in args]

    def result_meta(name):
        return (tuple(int(x) for x in field(values[name], 'shape')), tuple(int(x) for x in field(values[name], 'strides')))
    if kind == 'table':
        mask, name = refs[0], results[0]
        ml = layouts[mask]
        if dtypes[mask] != 'float32':
            raise ValueError('mattn_table: FP32 mask required')
        lead = [1] * (2 - (len(ml.shape) - 2)) + list(ml.shape[:-2])
        lst = [0] * (2 - (len(ml.shape) - 2)) + list(ml.strides[:-2])
        M, N = ml.shape[-2:]
        shape, _ = result_meta(name)
        define(name, shape, None, 'int64')
        mptr = f"{ident(ml.root)} + {ml.offset}LL" if ml.offset else ident(ml.root)
        code = (f"pwin::Args A{{}}; A.mask = {mptr}; A.mb0 = {lst[0]}LL; A.mb1 = {lst[1]}LL; "
                f"A.mi = {ml.strides[-2]}LL; A.mj = {ml.strides[-1]}LL; A.M = {M}; A.N = {N}; "
                f"A.table = reinterpret_cast<long long*>({ident(name)});\n"
                f"pwin::table_phase(A, {lead[0]}, {lead[1]}, smem);")
        blocks = lead[0] * lead[1] * (M // 16) * (N // KEY_BLOCK)
        return 'window_table', code, {'work_items': max(1, blocks // 8), 'shared_bytes': 0}, [mask], [name]
    q, kt, v, mask, table = refs[:5]
    scale, mode, group = (args[5], args[6], args[7]) if kind == 'fwd' else (args[7], args[8], args[9])
    rest = [r for r in refs[5:] if r is not None]
    lq, lk = layouts[q], layouts[kt]
    Z, M, D = lq.shape
    N = lk.shape[2]
    if D != 32 or tuple(lk.shape) != (Z, D, N):
        raise ValueError('mattn: unsupported shapes')
    wide_ = lambda r: dtypes[r] == 'float32'
    fields = [f"A.q = {_op3(lq, ident, q, fp32=wide_(q))};",
              f"A.k = {_op3(lk, ident, kt, transpose=True, fp32=wide_(kt))};",
              f"A.v = {_op3(layouts[v], ident, v, fp32=wide_(v))};",
              f"A.Z = {Z}; A.M = {M}; A.N = {N}; A.scale = {repr(float(scale))}f; A.scale_mode = {int(mode) & 3}; "
              f"A.rscale = __frcp_rn(A.scale); A.online = {1 if int(mode) & ONLINE else 0};"]
    online = bool(int(mode) & ONLINE)
    reads = [q, kt, v]
    if mask is not None:
        ml = layouts[mask]
        g = list(group) if group else [Z, M, N]
        lead_g = [1] * (2 - len(g[:-2])) + list(g[:-2])
        lead = [1] * (2 - (len(ml.shape) - 2)) + list(ml.shape[:-2])
        lst = [0] * (2 - (len(ml.shape) - 2)) + list(ml.strides[:-2])
        mb = [st if e != 1 else 0 for e, st in zip(lead, lst)]
        mptr = f"{ident(ml.root)} + {ml.offset}LL" if ml.offset else ident(ml.root)
        tb0 = lead[1] if lead[0] > 1 else 0
        tb1 = 1 if lead[1] > 1 else 0
        fields.append(f"A.mask = {mptr}; A.g1 = {lead_g[1]}; A.mb0 = {mb[0]}LL; A.mb1 = {mb[1]}LL; "
                      f"A.mi = {ml.strides[-2]}LL; A.mj = {ml.strides[-1]}LL; A.table = reinterpret_cast<long long*>({ident(table)}); "
                      f"A.tb0 = {tb0}LL; A.tb1 = {tb1}LL;")
        reads += [mask, table]
    else:
        fields.append("A.mask = nullptr; A.table = nullptr; A.g1 = 1;")

    def res(name, kmajor=False):
        shape, strides = result_meta(name)
        dt = _dtype_of(values[name], field)
        define(name, shape, strides, dt)
        pointer = f"reinterpret_cast<__nv_bfloat16*>({ident(name)})"
        flag = 1 if dt == 'float32' else 0
        if kmajor:   # logical [Z, D, N]: store element (z, key, d)
            return f"pwin::Res{{{pointer}, {strides[0]}LL, {strides[2]}LL, {strides[1]}LL, {flag}}}"
        return f"pwin::Res{{{pointer}, {strides[0]}LL, {strides[1]}LL, {strides[2]}LL, {flag}}}"
    if kind == 'fwd':
        out, stats = results
        fields.append(f"A.out = {res(out)};")
        define(stats, result_meta(stats)[0], None, 'float32')
        fields.append(f"A.stats = {ident(stats)};")
        call = 'pwin::fwd_online_phase(A, smem);' if online else 'pwin::fwd_phase(A, smem);'
    else:
        dout, st = rest[0], rest[1]
        fields.append(f"A.dout = {_op3(layouts[dout], ident, dout, fp32=wide_(dout))}; A.stats = {ident(st)};")
        reads += [dout, st]
        if online:
            o_ = refs[11] if len(refs) > 11 else None
            if o_ is None:
                raise ValueError('mattn_bwd (online arithmetic) needs the forward output')
            fields.append(f"A.o = {_op3(layouts[o_], ident, o_, fp32=wide_(o_))};")
            reads.append(o_)
        dq, dkt, dv = results
        fields.append(f"const pwin::Res dq_ = {res(dq)};")
        fields.append(f"const pwin::Res dk_ = {res(dkt, kmajor=True)};")
        fields.append(f"const pwin::Res dv_ = {res(dv)};")
        call = ('pwin::bwd_online_phase(A, dq_, dk_, dv_, smem);' if online
                else 'pwin::bwd_phase(A, dq_, dk_, dv_, smem);')
    code = "pwin::Args A{};\n" + "\n".join(fields) + "\n" + call
    return ('window_' + kind, code, {'work_items': Z, 'shared_bytes': ITEM_SHARED(M, N)},
            list(dict.fromkeys(reads)), results)


__all__ = ['HEADER', 'SCHEMA', 'TARGETS', 'emit', 'rewrite_joint']
