"""P9-C: operand and gradient absorption for the rectangular mixed attention (capture rewrite; new code).

After ``ops.window_attention.rewrite_joint``, a block-local attention reads operands produced by

    src (BF16 [B, L, 3*H*D], a fused projection result) -> widened FP32 -> view [B, L, 3, H, D] -> permute
    q = rope(src[0]), k = rope(src[1]) (half-split rotation by cos/sin [L, D/2] tables), v = src[2]
    keys / values of block n = (block n - 1 or zeros, block n)          (block-shifted concatenation)

and its outputs flow through the duals: O is head-merged by view/transpose/clone into [B, L, H*D]; dO is head-split
from the FP32 widening of a BF16 [B, L, H*D] product; dK / dV pass through the shifted concatenation's derivative
(slice / slice_backward / add), dQ / dK through the rotation's derivative (slice, mul, neg, add, slice_backward), and
the three are placed by select_backward / add / permute / clone into the projection gradient that is rounded to BF16.

``rewrite_joint`` recognizes exactly that structure by graph references, targets, arguments, dtypes and shapes (no
names) and replaces it with three opaque operations whose eager definitions below ARE the replaced subgraphs, op for
op, so CPU replay is bitwise identical:

    O, stats                 = mattn_fwd2(src, cos, sin, mask, table)    O head-merged BF16 [B, L, H*D]
    dq, dkc, dkp, dvc, dvp   = mattn_bwd2(src, cos, sin, mask, table, dout, stats)
    dsrc                     = mattn_dsrc(dq, dkc, dkp, dvc, dvp, cos, sin)

dq is the final BF16 value of the query slot of the projection gradient; dkc / dvc (current block) and dkp / dvp
(next block's previous-block contribution, shifted; zero for the last block) are the BF16 attention gradients whose
FP32 sum the eager graph forms. Device lowering (kernels/attention_window_v4.cuh) reads projection rows directly:
the rotation is applied while loading, key/value rows come from a row offset into the previous block, and O / dO /
the gradients are stored/read in head-merged order. Anything else is not rewritten (recorded refusal).
"""
from __future__ import annotations

import math
import operator
import sys
from typing import List, Optional

import torch

from . import window_attention as WA

SCHEMA = 'p9c-window-fusion-v1'
TARGETS = {f'latch_ops.mattn_{kind}.default': kind for kind in ('fwd2', 'bwd2', 'dsrc')}
MULTI = ('latch_ops.mattn_fwd2.default', 'latch_ops.mattn_bwd2.default')
MAXI = sys.maxsize


# --------------------------------------------------------------------------------------- eager definitions
def _rope(x, cos, sin, half):
    """Half-split rotation exactly as the captured graph: cat(x1 c - x2 s, x2 c + x1 s)."""
    x1 = torch.ops.aten.slice.Tensor(x, 3, 0, half)
    x2 = torch.ops.aten.slice.Tensor(x, 3, half, MAXI)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)


def _rope_grad(g, cos, sin, half):
    """The captured derivative of _rope: second half (g2 c + (-g1) s), first half (g2 s + g1 c), placed and summed."""
    shape = list(g.shape)
    g1 = torch.ops.aten.slice.Tensor(g, 3, 0, half)
    g2 = torch.ops.aten.slice.Tensor(g, 3, half, shape[3])
    second = g2 * cos + torch.neg(g1) * sin
    first = g2 * sin + g1 * cos
    return (torch.ops.aten.slice_backward(second, shape, 3, half, MAXI, 1)
            + torch.ops.aten.slice_backward(first, shape, 3, 0, half, 1))


def _geom(geom):
    B, L, H, D, W = (int(x) for x in geom[:5])
    nb = L // W
    return B, L, H, D, W, nb, B * H, B * H * nb


def _shifted(t, BH, nb, W, D, Z):
    return torch.cat((torch.cat((torch.zeros_like(t[:, :1]), t[:, :-1]), 1), t), 2).view(Z, 2 * W, D)


def _operands(src, cos, sin, geom):
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    x = src.float().view(B, L, 3, H, D).permute(2, 0, 3, 1, 4)
    q = _rope(x[0], cos, sin, D // 2)
    k = _rope(x[1], cos, sin, D // 2)
    v = x[2].contiguous()
    keys = _shifted(k.view(BH, nb, W, D), BH, nb, W, D, Z)
    vals = _shifted(v.view(BH, nb, W, D), BH, nb, W, D, Z)
    return q.view(Z, W, D), keys.transpose(1, 2), vals


def _split(t, BH, nb, W, D):
    """(current half, previous half of the next block shifted onto this block, zero for the last block)."""
    prev = torch.ops.aten.slice.Tensor(t, 2, 0, W)
    cur = torch.ops.aten.slice.Tensor(t, 2, W, 2 * W)
    shifted = torch.ops.aten.slice_backward(torch.ops.aten.slice.Tensor(prev, 1, 1, nb), [BH, nb, W, D], 1, 0, -1, 1)
    return cur, shifted


@torch.library.custom_op('latch_ops::mattn_fwd2', mutates_args=())
def mattn_fwd2(src: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor],
               table: Optional[torch.Tensor], scale: float, mode: int, group: List[int],
               geom: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(O head-merged BF16 [B, L, H*D], stats [Z, W, 2]) of mattn_fwd on the operands formed from src."""
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    q, kt, v = _operands(src, cos, sin, geom)
    o, stats = WA.mattn_fwd(q, kt, v, mask, table, scale, mode, group)
    return o.view(B, H, L, D).transpose(1, 2).contiguous().view(B, L, H * D), stats


@mattn_fwd2.register_fake
def _(src, cos, sin, mask, table, scale, mode, group, geom):
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    return (src.new_empty((B, L, H * D), dtype=torch.bfloat16),
            src.new_empty((Z, W, 2), dtype=torch.float32))


@torch.library.custom_op('latch_ops::mattn_bwd2', mutates_args=())
def mattn_bwd2(src: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor],
               table: Optional[torch.Tensor], dout: torch.Tensor, stats: torch.Tensor, scale: float, mode: int,
               group: List[int], geom: List[int]
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(dq, dkc, dkp, dvc, dvp), BF16 [B, H, L, D]: dq is the final query-slot value of the projection gradient."""
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    q, kt, v = _operands(src, cos, sin, geom)
    do = dout.float().view(B, L, H, D).transpose(1, 2).contiguous().view(Z, W, D)
    dq, dkt, dv = WA.mattn_bwd(q, kt, v, mask, table, do, stats, scale, mode, group, [1, 1, 1])
    dqr = _rope_grad(dq.view(B, H, L, D), cos, sin, D // 2)
    dq_final = (torch.zeros_like(dqr) + dqr).to(torch.bfloat16)
    dkc, dkp = _split(dkt.transpose(1, 2).view(BH, nb, 2 * W, D), BH, nb, W, D)
    dvc, dvp = _split(dv.view(BH, nb, 2 * W, D), BH, nb, W, D)
    out = [t.reshape(B, H, L, D).to(torch.bfloat16).contiguous() for t in (dkc, dkp, dvc, dvp)]
    return (dq_final.contiguous(), *out)


@mattn_bwd2.register_fake
def _(src, cos, sin, mask, table, dout, stats, scale, mode, group, geom):
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    return tuple(src.new_empty((B, H, L, D), dtype=torch.bfloat16) for _ in range(5))


@torch.library.custom_op('latch_ops::mattn_dsrc', mutates_args=())
def mattn_dsrc(dq: torch.Tensor, dkc: torch.Tensor, dkp: torch.Tensor, dvc: torch.Tensor, dvp: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor, geom: List[int]) -> torch.Tensor:
    """BF16 projection gradient [B, L, 3*H*D]: (q slot, rope derivative of dkc + dkp, dvc + dvp), as the graph."""
    B, L, H, D, W, nb, BH, Z = _geom(geom)
    dk = dkc.float() + dkp.float()
    dv = dvc.float() + dvp.float()
    dkr = _rope_grad(dk.contiguous().view(B, H, L, D), cos, sin, D // 2)
    shape = [3, B, H, L, D]
    acc = (torch.ops.aten.select_backward(dv.view(B, H, L, D), shape, 0, 2)
           + torch.ops.aten.select_backward(dkr, shape, 0, 1))
    acc = acc + torch.ops.aten.select_backward(dq.float(), shape, 0, 0)
    return acc.permute(1, 3, 0, 2, 4).contiguous().view(B, L, 3 * H * D).to(torch.bfloat16)


@mattn_dsrc.register_fake
def _(dq, dkc, dkp, dvc, dvp, cos, sin, geom):
    B, L, H, D = (int(x) for x in geom[:4])
    return dq.new_empty((B, L, 3 * H * D), dtype=torch.bfloat16)


# ------------------------------------------------------------------------------------------ joint rewrite
class Refused(Exception):
    pass


_base, _meta, _is_cast = WA._base, WA._meta, WA._is_cast
_VIEWS = {'view', '_unsafe_view', 'reshape'}


def _need(cond, why):
    if not cond:
        raise Refused(why)


def _users(n):
    return list(n.users)


def _only_user(n, base=None):
    users = _users(n)
    _need(len(users) == 1, f'{n.name} has {len(users)} users')
    _need(base is None or _base(users[0]) in ((base,) if isinstance(base, str) else base),
          f'{n.name} feeds {users[0].name}, not {base}')
    return users[0]


def _shape(n):
    m = _meta(n)
    _need(m is not None, f'{getattr(n, "name", n)} has no tensor metadata')
    return list(m.shape)


def _is_view(n, shape=None):
    return _base(n) in _VIEWS and (shape is None or _shape(n) == list(shape))


def _slice_args(n):
    """(source, dim, start, end) of an aten.slice with unit step."""
    _need(_base(n) == 'slice', f'{n.name} is not a slice')
    args = list(n.args) + [0, MAXI, 1][len(n.args) - 1:]
    src, dim, start, end = args[0], int(args[1]), args[2], args[3]
    step = args[4] if len(args) > 4 else 1
    _need(step == 1, 'strided slice')
    rank = len(_shape(src))
    return src, dim % rank, int(start if start is not None else 0), int(end if end is not None else MAXI)


def _clamp(end, size):
    return size if end >= size else (end if end >= 0 else size + end)


def _mul_parts(n):
    _need(_base(n) == 'mul' and len(n.args) == 2 and not n.kwargs, f'{n.name} is not a binary mul')
    return n.args


def _match_rope(cat, x_expected=None):
    """cat([x1 c - x2 s, x2 c + x1 s], -1) of the halves of x; returns (x, cos, sin)."""
    _need(_base(cat) == 'cat', f'{cat.name} is not a cat')
    parts = cat.args[0]
    dim = cat.args[1] if len(cat.args) > 1 else 0
    rank = len(_shape(cat))
    _need(isinstance(parts, (list, tuple)) and len(parts) == 2 and int(dim) % rank == rank - 1, 'not a last-axis pair')
    sub, add = parts
    _need(_base(sub) == 'sub' and _base(add) == 'add' and len(sub.args) == 2 and len(add.args) == 2
          and not sub.kwargs and not add.kwargs, 'rotation halves are not sub / add')
    a1, a2 = sub.args
    x1, c1 = _mul_parts(a1)
    x2, s1 = _mul_parts(a2)
    b1, b2 = add.args
    p, c2 = _mul_parts(b1)
    r, s2 = _mul_parts(b2)
    _need(p is x2 and r is x1 and c2 is c1 and s2 is s1, 'rotation operands differ from x2 c + x1 s')
    src1, d1, st1, en1 = _slice_args(x1)
    src2, d2, st2, en2 = _slice_args(x2)
    D = _shape(src1)[-1]
    _need(src1 is src2 and d1 == d2 == rank - 1 and st1 == 0 and _clamp(en1, D) == D // 2 and st2 == D // 2
          and _clamp(en2, D) == D and D % 2 == 0, 'rotation halves are not the two halves of one tensor')
    _need(x_expected is None or src1 is x_expected, 'rotation source differs')
    for t in (c1, s1):
        _need(isinstance(t, torch.fx.Node) and _meta(t) is not None and _meta(t).dtype == torch.float32,
              'rotation tables are not FP32 tensors')
    return src1, c1, s1


def _match_head_split(sel, index):
    """select(permute(view(widen(src), [B, L, 3, H, D]), [2, 0, 3, 1, 4]), 0, index) -> (permute, src, B, L, H, D)."""
    _need(_base(sel) == 'select' and int(sel.args[1]) == 0 and int(sel.args[2]) == index, 'not a head-split select')
    perm = sel.args[0]
    _need(_base(perm) == 'permute' and [int(x) for x in perm.args[1]] == [2, 0, 3, 1, 4], 'not the head-split permute')
    view = perm.args[0]
    B, L, three, H, D = _shape(view)
    _need(_is_view(view) and three == 3, 'not a [B, L, 3, H, D] view')
    widen = view.args[0]
    _need(_is_cast(widen, torch.float32), 'projection is not an FP32 widening')
    src = widen.args[0]
    _need(_meta(src).dtype == torch.bfloat16 and _shape(src) == [B, L, 3 * H * D], 'projection is not BF16 [B, L, 3HD]')
    _need(tuple(_meta(src).stride()) == (L * 3 * H * D, 3 * H * D, 1), 'projection is not contiguous')
    return perm, src, B, L, H, D


def _match_shifted(node, W, D):
    """view(cat([cat([zeros_like(t[:, :1]), t[:, :-1]], 1), t], 2), [Z, 2W, D]) -> t [BH, nb, W, D]."""
    _need(_is_view(node), f'{node.name} is not a view')
    cat3 = node.args[0]
    _need(_base(cat3) == 'cat' and int(cat3.args[1]) == 2 and len(cat3.args[0]) == 2, 'not the block concatenation')
    cat2, t = cat3.args[0]
    _need(_base(cat2) == 'cat' and int(cat2.args[1]) == 1 and len(cat2.args[0]) == 2, 'not the block shift')
    zeros, head = cat2.args[0]
    _need(_base(zeros) == 'zeros_like', 'shift does not insert zeros')
    zsrc, zd, zs, ze = _slice_args(zeros.args[0])
    hsrc, hd, hs, he = _slice_args(head)
    BH, nb, Wt, Dt = _shape(t)
    _need(zsrc is t and hsrc is t and zd == 1 and zs == 0 and _clamp(ze, nb) == 1 and hd == 1 and hs == 0
          and _clamp(he, nb) == nb - 1 and Wt == W and Dt == D, 'shift is not (zeros, blocks 0..nb-2)')
    _need(_shape(node) == [BH * nb, 2 * W, D], 'concatenation view differs')
    return t


def _match_forward(F):
    q, kt, v, mask, table, scale, mode, group = F.args[:8]
    _need(int(mode) in (0, 1, 2), 'only the exact arithmetic is absorbed')
    Z, W, D = _shape(q)
    _need(_is_view(q, [Z, W, D]), 'query operand is not a view')
    perm_q, src, B, L, H, D2 = None, None, None, None, None, None
    xq, cos, sin = _match_rope(q.args[0])
    perm, src, B, L, H, D2 = _match_head_split(xq, 0)
    _need(D2 == D and L % W == 0 and B * H * (L // W) == Z, 'geometry differs')
    nb = L // W
    _need(_base(kt) == 'transpose' and {int(kt.args[1]) % 3, int(kt.args[2]) % 3} == {1, 2}, 'key operand is not K^T')
    kb = _match_shifted(kt.args[0], W, D)
    _need(_is_view(kb, [B * H, nb, W, D]), 'key blocks are not a view')
    xk, cos_k, sin_k = _match_rope(kb.args[0])
    _need(cos_k is cos and sin_k is sin, 'query and key rotations use different tables')
    perm_k, *_ = _match_head_split(xk, 1)
    _need(perm_k is perm, 'keys come from another projection')
    vb = _match_shifted(v, W, D)
    _need(_is_view(vb, [B * H, nb, W, D]), 'value blocks are not a view')
    vc = vb.args[0]
    if _base(vc) == 'clone':
        vc = vc.args[0]
    _match_head_split(vc, 2)
    _need(vc.args[0] is perm, 'values come from another projection')
    cs = _shape(cos)
    _need(cs == [L, D // 2] and _shape(sin) == cs, 'rotation tables are not [L, D/2]')
    for t in (cos, sin):
        _need(tuple(_meta(t).stride()) == (D // 2, 1), 'rotation tables are not contiguous')
    _need(D in WA.HEAD_WIDTHS and W % 128 == 0 and 2 * W <= WA.MAX_EXTENT, 'absorbed loads need D = 32, W % 128 == 0')
    return dict(src=src, cos=cos, sin=sin, mask=mask, table=table, scale=scale, mode=int(mode), group=list(group),
                geom=[B, L, H, D, W], Z=Z)


def _getitems(node):
    out = {}
    for u in _users(node):
        _need(u.target is operator.getitem, f'{node.name} result consumed without getitem')
        out[int(u.args[1])] = u
    return out


def _match_out_merge(o, rec):
    """getitem O -> view [B, H, L, D] -> transpose(1, 2) -> clone (contiguous [B, L, H, D]); returns the clone."""
    B, L, H, D, W = rec['geom']
    view = _only_user(o, tuple(_VIEWS))
    _need(_shape(view) == [B, H, L, D], 'output view differs')
    tr = _only_user(view, 'transpose')
    _need({int(tr.args[1]) % 4, int(tr.args[2]) % 4} == {1, 2}, 'output transpose differs')
    clone = _only_user(tr, 'clone')
    return clone


def _match_dout(dout, rec):
    B, L, H, D, W = rec['geom']
    _need(_is_view(dout, [rec['Z'], W, D]), 'dO is not a view')
    clone = dout.args[0]
    _need(_base(clone) == 'clone', 'dO is not a head-split copy')
    tr = clone.args[0]
    _need(_base(tr) == 'transpose' and {int(tr.args[1]) % 4, int(tr.args[2]) % 4} == {1, 2}, 'dO transpose differs')
    view = tr.args[0]
    _need(_is_view(view, [B, L, H, D]), 'dO head view differs')
    widen = view.args[0]
    _need(_is_cast(widen, torch.float32), 'dO is not an FP32 widening')
    src = widen.args[0]
    _need(_meta(src).dtype == torch.bfloat16 and _shape(src) == [B, L, H * D]
          and tuple(_meta(src).stride()) == (L * H * D, H * D, 1), 'dO source is not contiguous BF16 [B, L, HD]')
    return src


def _match_rope_grad(g, rec):
    """Forward from g ([B, H, L, D] FP32): the captured rotation derivative; returns its sum node (add of two
    slice_backward placements)."""
    B, L, H, D, W = rec['geom']
    half = D // 2
    halves = {}
    for u in _users(g):
        s, d, st, en = _slice_args(u)
        _need(s is g and d == 3, 'derivative slices differ')
        halves['first' if st == 0 and _clamp(en, D) == half else 'second' if st == half and _clamp(en, D) == D
               else 'other'] = u
    _need(set(halves) == {'first', 'second'} and len(_users(g)) == 2, 'derivative halves differ')
    g1, g2 = halves['first'], halves['second']
    cos, sin = rec['cos'], rec['sin']

    def mul_of(x, table):
        for u in _users(x):
            if _base(u) == 'mul' and set(u.args) == {x, table} and len(u.args) == 2:
                return u
        raise Refused('derivative product missing')
    _need(len(_users(g2)) == 2 and len(_users(g1)) == 2, 'derivative halves have extra readers')
    a, b = mul_of(g2, sin), mul_of(g2, cos)
    neg = next((u for u in _users(g1) if _base(u) == 'neg'), None)
    _need(neg is not None, 'negated half missing')
    c = _only_user(neg, 'mul')
    _need(set(c.args) == {neg, sin}, 'negated product differs')
    e = mul_of(g1, cos)
    second = _only_user(b, 'add')
    _need(set(second.args) == {b, c} and _only_user(c) is second, 'second-half derivative differs')
    first = _only_user(a, 'add')
    _need(set(first.args) == {a, e} and _only_user(e) is first, 'first-half derivative differs')
    sb2 = _only_user(second, 'slice_backward')
    sb1 = _only_user(first, 'slice_backward')
    _need(int(sb2.args[2]) == 3 and int(sb2.args[3]) == half and int(sb1.args[2]) == 3 and int(sb1.args[3]) == 0
          and _clamp(int(sb1.args[4]), D) == half, 'derivative placement differs')
    total = _only_user(sb2, 'add')
    _need(set(total.args) == {sb1, sb2} and _only_user(sb1) is total, 'derivative sum differs')
    return total


def _match_split_add(t, rec):
    """t [Z, 2W, D] (view chain) -> slice halves -> slice(prev, 1, 1, nb) -> slice_backward -> add(cur, shifted)."""
    B, L, H, D, W = rec['geom']
    nb, BH = L // W, B * H
    view = t
    _need(_is_view(view, [BH, nb, 2 * W, D]), 'gradient block view differs')
    halves = {}
    for u in _users(view):
        s, d, st, en = _slice_args(u)
        _need(s is view and d == 2, 'gradient halves differ')
        halves['prev' if st == 0 and _clamp(en, 2 * W) == W else 'cur' if st == W and _clamp(en, 2 * W) == 2 * W
               else 'other'] = u
    _need(set(halves) == {'prev', 'cur'} and len(_users(view)) == 2, 'gradient halves differ')
    tail = _only_user(halves['prev'], 'slice')
    s, d, st, en = _slice_args(tail)
    _need(d == 1 and st == 1 and _clamp(en, nb) == nb, 'previous-half slice differs')
    sb = _only_user(tail, 'slice_backward')
    _need([int(x) for x in sb.args[1]] == [BH, nb, W, D] and int(sb.args[2]) == 1 and int(sb.args[3]) == 0
          and _clamp(int(sb.args[4]), nb) == nb - 1, 'previous-half placement differs')
    add = _only_user(sb, 'add')
    _need(set(add.args) == {halves['cur'], sb} and _only_user(halves['cur']) is add, 'block gradient sum differs')
    return add


def _match_backward(G, rec, fwd_stats):
    q, kt, v, mask, table, dout, stats, scale, mode, group, wide = G.args[:11]
    _need(stats is fwd_stats, 'backward statistics are not this forward')
    _need(list(wide) == [1, 1, 1], 'backward results are not widened')
    B, L, H, D, W = rec['geom']
    nb, BH = L // W, B * H
    dsrc_o = _match_dout(dout, rec)
    items = _getitems(G)
    _need(set(items) == {0, 1, 2}, 'backward results missing')
    # dQ: view [B, H, L, D] -> rotation derivative -> select_backward(.., 0, 0)
    vq = _only_user(items[0], tuple(_VIEWS))
    _need(_shape(vq) == [B, H, L, D], 'dQ view differs')
    tq = _match_rope_grad(vq, rec)
    sq = _only_user(tq, 'select_backward')
    # dK: transpose -> view [BH, nb, 2W, D] -> split/add -> clone -> view [B, H, L, D] -> rotation derivative
    trk = _only_user(items[1], 'transpose')
    vk = _only_user(trk, tuple(_VIEWS))
    addk = _match_split_add(vk, rec)
    ck = _only_user(addk, 'clone')
    vk2 = _only_user(ck, tuple(_VIEWS))
    _need(_shape(vk2) == [B, H, L, D], 'dK view differs')
    tk = _match_rope_grad(vk2, rec)
    sk = _only_user(tk, 'select_backward')
    # dV: view [BH, nb, 2W, D] -> split/add -> view [B, H, L, D] -> select_backward(.., 0, 2)
    vv = _only_user(items[2], tuple(_VIEWS))
    addv = _match_split_add(vv, rec)
    vv2 = _only_user(addv, tuple(_VIEWS))
    _need(_shape(vv2) == [B, H, L, D], 'dV view differs')
    sv = _only_user(vv2, 'select_backward')
    shape = [3, B, H, L, D]
    for node, index in ((sq, 0), (sk, 1), (sv, 2)):
        _need([int(x) for x in node.args[1]] == shape and int(node.args[2]) == 0 and int(node.args[3]) == index,
              'gradient placement differs')
    # (sv + sk) + sq in either operand order, then permute [1, 3, 0, 2, 4] -> clone -> view -> BF16 cast.
    a1 = _only_user(sv, 'add')
    _need(set(a1.args) == {sv, sk} and _only_user(sk) is a1, 'value/key gradient sum differs')
    a2 = _only_user(a1, 'add')
    _need(set(a2.args) == {a1, sq} and _only_user(sq) is a2, 'query gradient sum differs')
    perm = _only_user(a2, 'permute')
    _need([int(x) for x in perm.args[1]] == [1, 3, 0, 2, 4], 'gradient permute differs')
    clone = _only_user(perm, 'clone')
    view = _only_user(clone, tuple(_VIEWS))
    _need(_shape(view) == [B, L, 3 * H * D], 'projection-gradient view differs')
    cast = _only_user(view)
    _need(_is_cast(cast, torch.bfloat16), 'projection gradient is not rounded to BF16')
    return dict(dout_src=dsrc_o, cast=cast)


def rewrite_joint(module):
    """Rewrite admitted mattn_fwd / mattn_bwd pairs of a joint FX graph in place. Returns (records, refusals)."""
    graph = module.graph
    ops = torch.ops.latch_ops
    records, refusals = [], []
    bwd_of = {}
    for n in graph.nodes:
        if str(n.target) == 'latch_ops.mattn_bwd.default':
            bwd_of.setdefault(tuple(n.args[:5]), []).append(n)
    for F in [n for n in graph.nodes if str(n.target) == 'latch_ops.mattn_fwd.default']:
        try:
            rec = _match_forward(F)
            outs = _getitems(F)
            _need(set(outs) == {0, 1}, 'forward results missing')
            gs = bwd_of.get(tuple(F.args[:5]), [])
            _need(len(gs) == 1, 'no unique backward for this forward')
            G = gs[0]
            o_users = [u for u in _users(outs[0])]
            _need(len(o_users) == 1, 'forward output has other readers')
            o_clone = _match_out_merge(outs[0], rec)
            stats_users = [u for u in _users(outs[1]) if u is not G]
            bw = _match_backward(G, rec, outs[1])
        except Refused as exc:
            refusals.append({'attention': F.name, 'reason': str(exc)})
            continue
        args = (rec['scale'], rec['mode'], rec['group'], rec['geom'])
        with graph.inserting_before(F):
            f2 = graph.call_function(ops.mattn_fwd2.default, (rec['src'], rec['cos'], rec['sin'], rec['mask'],
                                                               rec['table'], *args))
            o2 = graph.call_function(operator.getitem, (f2, 0))
            st2 = graph.call_function(operator.getitem, (f2, 1))
        B, L, H, D, W = rec['geom']
        with graph.inserting_after(o_clone):
            o4 = graph.call_function(torch.ops.aten.view.default, (o2, [B, L, H, D]))
        o_clone.replace_all_uses_with(o4)
        for u in stats_users:
            u.replace_input_with(outs[1], st2)
        with graph.inserting_before(G):
            g2 = graph.call_function(ops.mattn_bwd2.default, (rec['src'], rec['cos'], rec['sin'], rec['mask'],
                                                               rec['table'], bw['dout_src'], st2, *args))
            parts = [graph.call_function(operator.getitem, (g2, i)) for i in range(5)]
        with graph.inserting_before(bw['cast']):
            ds = graph.call_function(ops.mattn_dsrc.default, (*parts, rec['cos'], rec['sin'], rec['geom']))
        bw['cast'].replace_all_uses_with(ds)
        records.append({'attention': F.name, 'geom': rec['geom'], 'Z': rec['Z'],
                        'nodes': {'fwd2': f2.name, 'bwd2': g2.name, 'dsrc': ds.name}})
    if records:
        graph.eliminate_dead_code()
        graph.lint()
        module.recompile()
        from .loop_attention import _propagate
        _propagate(module)
    return records, refusals


# ------------------------------------------------------------------------------------------------ lowering
def _ptr(layout, ident, ctype):
    pointer = f"{ident(layout.root)} + {layout.offset}LL" if layout.offset else ident(layout.root)
    return f"(({ctype})({pointer}))"


def _contiguous(layout, shape):
    expected, step = [], 1
    for extent in reversed(shape):
        expected.append(step)
        step *= extent
    return tuple(layout.shape) == tuple(shape) and tuple(layout.strides) == tuple(reversed(expected))


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define):
    """Lower one absorbed attention operation to a phase. Returns (kind, code, metadata, reads, writes)."""
    target = str(field(op, 'target', field(op, 'op'))).removeprefix('torch.ops.')
    kind = TARGETS[target]
    args = list(field(op, 'args', []))
    results = list(field(op, 'outputs', None) or [field(op, 'name')])
    refs = [ref(a, names) for a in args]

    def result_meta(name):
        return (tuple(int(x) for x in field(values[name], 'shape')),
                tuple(int(x) for x in field(values[name], 'strides')))

    def need(cond, why):
        if not cond:
            raise ValueError(f'{target}: {why}')
    geom = [int(x) for x in args[-1]]
    B, L, H, D, W = geom[:5]
    nb = L // W
    Z = B * H * nb
    common = [f"A.absorbed = 1; A.gB = {B}; A.gL = {L}; A.gH = {H}; A.gW = {W}; A.gnb = {nb};"]
    if kind == 'dsrc':
        dq, dkc, dkp, dvc, dvp, cos, sin = refs[:7]
        for r in (dq, dkc, dkp, dvc, dvp):
            need(dtypes[r] == 'bfloat16' and _contiguous(layouts[r], (B, H, L, D)), 'gradient parts are not BF16 [B, H, L, D]')
        for r in (cos, sin):
            need(dtypes[r] == 'float32' and _contiguous(layouts[r], (L, D // 2)), 'rotation tables are not [L, D/2]')
        out = results[0]
        shape, strides = result_meta(out)
        define(out, shape, strides, 'bfloat16')
        bp = '__nv_bfloat16*'
        code = ("pwin::Args A{};\n" + "".join(common) + "\n"
                f"A.dq_o = {_ptr(layouts[dq], ident, bp)}; A.dkc_o = {_ptr(layouts[dkc], ident, bp)}; "
                f"A.dkp_o = {_ptr(layouts[dkp], ident, bp)}; A.dvc_o = {_ptr(layouts[dvc], ident, bp)}; "
                f"A.dvp_o = {_ptr(layouts[dvp], ident, bp)};\n"
                f"A.rcos = {_ptr(layouts[cos], ident, 'const float*')}; A.rsin = {_ptr(layouts[sin], ident, 'const float*')};\n"
                f"A.proj_grad = reinterpret_cast<__nv_bfloat16*>({ident(out)});\n"
                "pwin::dsrc_phase(A);")
        tasks = B * L * H * 2
        return ('window_dsrc', code, {'work_items': max(1, (tasks + 255) // 256), 'shared_bytes': 0},
                list(dict.fromkeys([dq, dkc, dkp, dvc, dvp, cos, sin])), results)
    src, cos, sin, mask, table = refs[:5]
    need(dtypes[src] == 'bfloat16' and _contiguous(layouts[src], (B, L, 3 * H * D)), 'projection is not BF16 [B, L, 3HD]')
    for r in (cos, sin):
        need(dtypes[r] == 'float32' and _contiguous(layouts[r], (L, D // 2)), 'rotation tables are not [L, D/2]')
    need(D == 32 and W % 128 == 0, 'absorbed loads need D = 32 and W % 128 == 0')
    if kind == 'fwd2':
        scale, mode, group = args[5], args[6], args[7]
    else:
        scale, mode, group = args[7], args[8], args[9]
    fields = list(common)
    fields.append(f"A.src = {_ptr(layouts[src], ident, 'const __nv_bfloat16*')}; "
                  f"A.rcos = {_ptr(layouts[cos], ident, 'const float*')}; A.rsin = {_ptr(layouts[sin], ident, 'const float*')};")
    fields.append(f"A.Z = {Z}; A.M = {W}; A.N = {2 * W}; A.scale = {repr(float(scale))}f; A.scale_mode = {int(mode) & 3}; "
                  f"A.rscale = __frcp_rn(A.scale); A.online = 0;")
    reads = [src, cos, sin]
    if mask is not None:
        ml = layouts[mask]
        g = list(group) if group else [Z, W, 2 * W]
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
    if kind == 'fwd2':
        o, stats = results
        define(o, result_meta(o)[0], result_meta(o)[1], 'bfloat16')
        define(stats, result_meta(stats)[0], None, 'float32')
        need(result_meta(o)[1] == (L * H * D, H * D, 1), 'O is not contiguous head-merged')
        fields.append(f"A.o_m = reinterpret_cast<__nv_bfloat16*>({ident(o)}); A.stats = {ident(stats)};")
        call = "pwin::fwd2_phase(A, smem);"
    else:
        dout, st = refs[5], refs[6]
        need(dtypes[dout] == 'bfloat16' and _contiguous(layouts[dout], (B, L, H * D)), 'dO is not BF16 [B, L, HD]')
        fields.append(f"A.dsrc = {_ptr(layouts[dout], ident, 'const __nv_bfloat16*')}; A.stats = {ident(st)};")
        reads += [dout, st]
        names_ = ['dq_o', 'dkc_o', 'dkp_o', 'dvc_o', 'dvp_o']
        for res_, field_ in zip(results, names_):
            shape, strides = result_meta(res_)
            need(shape == (B, H, L, D), 'gradient part shape differs')
            define(res_, shape, strides, 'bfloat16')
            fields.append(f"A.{field_} = reinterpret_cast<__nv_bfloat16*>({ident(res_)});")
        call = "pwin::bwd2_phase(A, smem);"
    code = "pwin::Args A{};\n" + "\n".join(fields) + "\n" + call
    return ('window_' + kind, code, {'work_items': Z, 'shared_bytes': WA.ITEM_SHARED(W, 2 * W)},
            list(dict.fromkeys(reads)), results)


__all__ = ['SCHEMA', 'TARGETS', 'MULTI', 'emit', 'rewrite_joint']
