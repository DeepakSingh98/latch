"""Independent NumPy FP64 reference for s99k-gdssm-v1 (SPEC §2-§5), hand-derived backward.

No torch, no autograd: the backward is written out from the equations so that it is an
independent derivation from the conventional (autograd) trainer. `mode='exact'` evaluates the
real-arithmetic definition in float64; `mode='p1'` additionally applies the BF16 roundings of the
precision policy at every matrix product (float64 elsewhere) for attribution only.

`mutant` names a deliberately wrong variant used to prove the checks detect it:
  detach_state        backward drops the adjoint through alpha (dh_{t-1} gets no alpha*dh_t)
  ignore_reset        forward/backward ignore seg (state leaks across documents)
  shift_reset         reset applied one position late
  reverse_time        scan runs from t=T-1 down to 0 (anti-causal)
  drop_batch_shared   a_log/dt_bias/d_skip gradients only from batch row 0
  drop_tied_head      emb gradient omits the tied head contribution
  drop_channel_shared k/r gradients summed over the first half of channels only
"""
import math

import numpy as np

MUTANTS = ('detach_state', 'ignore_reset', 'shift_reset', 'reverse_time', 'drop_batch_shared',
           'drop_tied_head', 'drop_channel_shared')


def bf16(x):
    """Round-to-nearest-even to BF16 (via FP32, RNE at both steps as in the policy), as float64."""
    f = np.asarray(x, dtype=np.float64).astype(np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    nan = np.isnan(f)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    out = r.astype(np.uint32).view(np.float32).astype(np.float64)
    return np.where(nan, np.nan, out)


def _mm(a, b, mode):
    if mode == 'p1':
        return bf16(bf16(a) @ bf16(b))
    return a @ b


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rms(x, g, eps):
    rho = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rho * g, rho


def rms_bwd(du, x, rho, g):
    D = x.shape[-1]
    gdu = du * g
    dx = rho * gdu - x * rho ** 3 * np.sum(gdu * x, axis=-1, keepdims=True) / D
    dg = np.sum(du * x * rho, axis=tuple(range(x.ndim - 1)))
    return dx, dg


def _alpha_mask(seg, mutant):
    keep = 1.0 - seg.astype(np.float64)
    if mutant == 'ignore_reset':
        keep = np.ones_like(keep)
    elif mutant == 'shift_reset':
        keep = np.ones_like(keep)
        keep[:, 1:] = 1.0 - seg[:, :-1].astype(np.float64)
    return keep


def scan_forward(z, E, N, A, dtb, dsk, keep, mutant=None):
    B, T, _ = z.shape
    v, q, c = z[..., :E], z[..., E:2 * E], z[..., 2 * E:3 * E]
    k, r = z[..., 3 * E:3 * E + N], z[..., 3 * E + N:3 * E + 2 * N]
    pre = c + dtb
    dl = softplus(pre)
    lam = np.exp(A)
    alpha = keep[:, :, None, None] * np.exp(-dl[..., None] * lam)
    w = (dl * v)[..., None] * k[:, :, None, :]
    h = np.zeros((B, T, E, N))
    prev = np.zeros((B, E, N))
    order = range(T - 1, -1, -1) if mutant == 'reverse_time' else range(T)
    for t in order:
        prev = alpha[:, t] * prev + w[:, t]
        h[:, t] = prev
    s = np.einsum('bten,btn->bte', h, r) + dsk * v
    sg = sigmoid(q)
    o = s * (q * sg)
    return o, dict(v=v, q=q, c=c, k=k, r=r, pre=pre, dl=dl, lam=lam, alpha=alpha, h=h, s=s, sg=sg)


def scan_backward(do, sc, E, N, keep, mutant=None):
    v, q, k, r, pre, dl, lam = sc['v'], sc['q'], sc['k'], sc['r'], sc['pre'], sc['dl'], sc['lam']
    alpha, h, s, sg = sc['alpha'], sc['h'], sc['s'], sc['sg']
    B, T = v.shape[:2]
    gq = q * sg
    ds = do * gq
    dq = do * s * sg * (1.0 + q * (1.0 - sg))
    dsk_rows = ds * v
    dv = ds * sc['dsk']
    dr_full = h * ds[..., None]                                   # (B,T,E,N), summed over e below
    ddl = np.zeros((B, T, E))
    dk_full = np.zeros((B, T, E, N))
    dA_rows = np.zeros((B, E, N))
    carry = np.zeros((B, E, N))
    order = range(T) if mutant == 'reverse_time' else range(T - 1, -1, -1)
    reverse = mutant == 'reverse_time'
    for t in order:
        dh = carry + r[:, t, None, :] * ds[:, t, :, None]
        tp = (t + 1) if reverse else (t - 1)
        hprev = h[:, tp] if 0 <= tp < T else np.zeros((B, E, N))
        da = dh * hprev
        ddl[:, t] += np.sum(da * (-lam) * alpha[:, t], axis=-1) + np.sum(dh * v[:, t, :, None] * k[:, t, None, :], axis=-1)
        dv[:, t] += np.sum(dh * dl[:, t, :, None] * k[:, t, None, :], axis=-1)
        dk_full[:, t] = dh * (dl[:, t] * v[:, t])[..., None]
        dA_rows += da * (-dl[:, t, :, None] * lam) * alpha[:, t]
        carry = np.zeros_like(dh) if mutant == 'detach_state' else alpha[:, t] * dh
    dpre = ddl * np.where(pre > 20, 1.0, sigmoid(pre))
    if mutant == 'drop_channel_shared':
        dk = dk_full[:, :, :E // 2].sum(2)
        dr = dr_full[:, :, :E // 2].sum(2)
    else:
        dk = dk_full.sum(2)
        dr = dr_full.sum(2)
    dz = np.concatenate([dv, dq, dpre, dk, dr], axis=-1)
    sel = slice(0, 1) if mutant == 'drop_batch_shared' else slice(None)
    dA = dA_rows[sel].sum(0) * 1.0
    return dz, dict(a_log=dA, dt_bias=dpre[sel].sum((0, 1)), d_skip=dsk_rows[sel].sum((0, 1)))


def forward_backward(decl, params, batch, *, mode='exact', mutant=None, need_grad=True, trace=None):
    """Returns (loss, grads, aux). Params: dict id -> array (any float dtype; used as float64).
    `trace` (optional dict) receives intermediates for diagnosis; it never changes the math."""
    if mutant is not None and mutant not in MUTANTS:
        raise ValueError(mutant)
    P = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    eps = decl['norm_eps']
    tok, tgt, seg = batch['tok'], batch['tgt'], batch['seg']
    B, T = tok.shape
    keep = _alpha_mask(seg, mutant)
    mm = lambda a, b: _mm(a, b, mode)
    x = P['emb'][tok]
    caches = []
    for i, blk in enumerate(decl['blocks']):
        p = 'blocks.%d.' % i
        u, rho = rms(x, P[p + 'norm'], eps)
        if blk['kind'] == 'ssm':
            E, N = blk['d_inner'], blk['d_state']
            z = mm(u, P[p + 'w_in'])
            o, sc = scan_forward(z, E, N, P[p + 'a_log'], P[p + 'dt_bias'], P[p + 'd_skip'], keep, mutant)
            sc['dsk'] = P[p + 'd_skip']
            y = mm(o, P[p + 'w_out'])
            caches.append(dict(x=x, u=u, rho=rho, z=z, o=o, sc=sc))
            if trace is not None:
                trace['z:%d' % i], trace['o:%d' % i] = z, o
        else:
            pre = mm(u, P[p + 'w1'])
            act = pre * sigmoid(pre)
            y = mm(act, P[p + 'w2'])
            caches.append(dict(x=x, u=u, rho=rho, pre=pre, act=act))
        x = x + y
        if trace is not None:
            trace['x:%d' % (i + 1)] = x
    uf, rhof = rms(x, P['final_norm'], eps)
    logits = mm(uf, P['emb'].T)
    valid = tgt >= 0
    n = int(valid.sum())
    mx = logits.max(-1, keepdims=True)
    ex = np.exp(logits - mx)
    se = ex.sum(-1, keepdims=True)
    lse = (mx + np.log(se))[..., 0]
    safe = np.where(valid, tgt, 0)
    picked = np.take_along_axis(logits, safe[..., None], -1)[..., 0]
    loss = float(np.sum(np.where(valid, lse - picked, 0.0)) / n)
    aux = dict(logits=logits, n_valid=n)
    if trace is not None:
        trace['logits'], trace['x:0'] = logits, P['emb'][tok]
    if not need_grad:
        return loss, None, aux
    G = {}
    dlog = ex / se
    onehot = np.zeros_like(dlog)
    np.put_along_axis(onehot, safe[..., None], 1.0, -1)
    dlog = np.where(valid[..., None], (dlog - onehot) / n, 0.0)
    head = mm(dlog.reshape(-1, dlog.shape[-1]).T, uf.reshape(-1, uf.shape[-1]))
    duf = mm(dlog, P['emb'])
    dx, G['final_norm'] = rms_bwd(duf, x, rhof, P['final_norm'])
    if trace is not None:
        trace['dU:f'], trace['dX:f'] = duf, dx
    for i in range(len(decl['blocks']) - 1, -1, -1):
        blk, cch, p = decl['blocks'][i], caches[i], 'blocks.%d.' % i
        dy = dx
        M = B * T
        if blk['kind'] == 'ssm':
            E, N = blk['d_inner'], blk['d_state']
            G[p + 'w_out'] = mm(cch['o'].reshape(M, -1).T, dy.reshape(M, -1))
            do = mm(dy, P[p + 'w_out'].T)
            dz, sh = scan_backward(do, cch['sc'], E, N, keep, mutant)
            if trace is not None:
                trace['dO:%d' % i], trace['dZ:%d' % i] = do, dz
            G[p + 'a_log'], G[p + 'dt_bias'], G[p + 'd_skip'] = sh['a_log'], sh['dt_bias'], sh['d_skip']
            G[p + 'w_in'] = mm(cch['u'].reshape(M, -1).T, dz.reshape(M, -1))
            du = mm(dz, P[p + 'w_in'].T)
        else:
            G[p + 'w2'] = mm(cch['act'].reshape(M, -1).T, dy.reshape(M, -1))
            da = mm(dy, P[p + 'w2'].T)
            sg = sigmoid(cch['pre'])
            dpre = da * sg * (1.0 + cch['pre'] * (1.0 - sg))
            G[p + 'w1'] = mm(cch['u'].reshape(M, -1).T, dpre.reshape(M, -1))
            du = mm(dpre, P[p + 'w1'].T)
        dxn, G[p + 'norm'] = rms_bwd(du, cch['x'], cch['rho'], P[p + 'norm'])
        dx = dy + dxn
    if trace is not None:
        trace['dX:0'] = dx
    gather = np.zeros_like(P['emb'])
    np.add.at(gather, tok.reshape(-1), dx.reshape(-1, dx.shape[-1]))
    G['emb'] = gather if mutant == 'drop_tied_head' else gather + head
    aux.update(emb_gather=gather, emb_head=head)
    return loss, G, aux


def clip_adamw(decl, params, grads, moments, steps, lr):
    """FP64 evaluation of the s80u update (SPEC §5). moments: id -> (m, v); steps: id -> float.
    Returns new (params, moments, steps, norm, coef, clipped_grads)."""
    from .family import inventory
    o = decl['optimizer']
    b1, b2, eps = o['beta1'], o['beta2'], o['eps']
    norm = math.sqrt(sum(float(np.sum(np.asarray(g, np.float64) ** 2)) for g in grads.values()))
    c = decl['clip']['max_norm'] / (norm + 1e-6)
    coef = c if math.isnan(c) else min(c, 1.0)
    newp, newm, news, clipped = {}, {}, {}, {}
    for p in inventory(decl):
        k = p['id']
        g = np.asarray(grads[k], np.float64) * coef
        m, v = (np.asarray(a, np.float64) for a in moments[k])
        step = steps[k] + 1.0
        decay = o['weight_decay'] if p['group'] == 0 else 0.0
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        m = m + (1 - b1) * (g - m)
        v = v * b2 + g * g * (1 - b2)
        w = np.asarray(params[k], np.float64) * (1 - lr * decay) - (lr / bc1) * m / (np.sqrt(v) / math.sqrt(bc2) + eps)
        newp[k], newm[k], news[k], clipped[k] = w, (m, v), step, g
    return newp, newm, news, norm, coef, clipped
