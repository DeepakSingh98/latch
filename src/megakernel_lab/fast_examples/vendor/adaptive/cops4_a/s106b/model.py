"""S106B byte language model: full or sliding-window attention trunk with one of three final tails.

Shape (S98C/S101H production byte model, re-implemented here for long context): vocabulary 256, d 256, 8 heads
(head dim 32), 6 pre-norm blocks, non-gated SiLU FFN width 1024, RMSNorm (eps 1e-6, learned gain), RoPE base 10000
(half-split rotation), tied embedding, no biases, N(0, 0.02) matrices and unit gains.

Arms (ARMS):
  F  full causal attention in every block, ordinary dense last FFN.
  W  sliding-window attention (a query at position t attends to keys t-window+1 .. t) in every block, dense last FFN.
  M  W + adaptive memory: the last FFN is split, r = u + (M_static(h) + M_fast(h; W_c)) / sqrt(2) (static 512 + fast
     512, same parameter count as the dense 1024 FFN). The fast weights start from a trained initialization per
     document, chunk c (C bytes) is SCORED with W_c, then W_{c+1} = W_c - eta * grad_W CE_c(W_c) using only the labels
     of chunk c (already scored). The outer loss is differentiated through every write (exact, second order), as in
     S101H/S105D's A arm and TTT-E2E's end-to-end next-token inner loss. Only the last FFN adapts.
  R  W + trained retrieval cache (S102E 'T' form, multi-head): at the last block beside the dense FFN,
     cache_j = W_O softmax_i(q_j.k_i) v_i / sqrt(2) over every position i of COMPLETED earlier chunks of the same
     sequence (floor(i/C) < floor(j/C)), v_i = W_V (h_i + E[y_i]) carries the observed label of i. Same history
     access as M (everything already scored), but by exact attention over all of it instead of a fixed-size state.

Every tensor path is causal within a sequence; a sequence never crosses a document; positions restart per sequence.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn.functional as F

INV_SQRT2 = 1.0 / math.sqrt(2.0)
EMB = 'embedding.weight'


@dataclass(frozen=True)
class Arch:
    arm: str = 'F'
    ctx: int = 8192
    window: int = 256
    chunk: int = 128
    d: int = 256
    heads: int = 8
    layers: int = 6
    hidden: int = 1024
    vocab: int = 256
    m_static: int = 512
    m_fast: int = 512
    cache_heads: int = 4
    rope_base: float = 10000.0
    eps: float = 1e-6

    @property
    def attn(self):
        return 'full' if self.arm == 'F' else 'swa'

    @property
    def tail(self):
        return dict(F='dense', W='dense', M='memory', R='cache')[self.arm]

    @property
    def hd(self):
        return self.d // self.heads

    def as_dict(self):
        return dict(asdict(self), attn=self.attn, tail=self.tail)


ARMS = ('F', 'W', 'M', 'R')
TINY = Arch(arm='M', ctx=48, window=6, chunk=8, d=16, heads=2, layers=2, hidden=24, vocab=11, m_static=12, m_fast=12,
            cache_heads=2)


def inventory(a):
    """(name, shape, kind) in canonical order; kind: 'emb', 'gain', 'matrix'."""
    out = [(EMB, (a.vocab, a.d), 'emb')]
    for i in range(a.layers):
        p = f'blocks.{i}.'
        out += [(p + 'attn_norm', (a.d,), 'gain'), (p + 'qkv', (3 * a.d, a.d), 'matrix'),
                (p + 'out', (a.d, a.d), 'matrix'), (p + 'ffn_norm', (a.d,), 'gain')]
        if i < a.layers - 1 or a.tail != 'memory':
            out += [(p + 'up', (a.hidden, a.d), 'matrix'), (p + 'down', (a.d, a.hidden), 'matrix')]
        else:
            out += [(p + 'static_up', (a.m_static, a.d), 'matrix'), (p + 'static_down', (a.d, a.m_static), 'matrix'),
                    ('fast_init.up', (a.m_fast, a.d), 'matrix'), ('fast_init.down', (a.d, a.m_fast), 'matrix')]
    if a.tail == 'cache':
        out += [(f'cache.{k}', (a.d, a.d), 'matrix') for k in 'qkvo']
    out.append(('final_norm', (a.d,), 'gain'))
    return out


def init_params(a, seed, device='cpu', dtype=torch.float32):
    """N(0, 0.02) matrices / embedding, unit gains, drawn in inventory order from one CPU generator (float64)."""
    g = torch.Generator().manual_seed(int(seed))
    P = {}
    for n, shape, kind in inventory(a):
        t = torch.ones(shape, dtype=torch.float64) if kind == 'gain' else torch.randn(shape, generator=g, dtype=torch.float64) * 0.02
        P[n] = t.to(device=device, dtype=dtype)
    return P


def parameter_count(a):
    return sum(math.prod(s) for _, s, _ in inventory(a))


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def rope_tables(a, device, dtype=torch.float32):
    half = a.hd // 2
    inv = 1.0 / (a.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
    ang = torch.arange(a.ctx, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos().to(device=device, dtype=dtype), ang.sin().to(device=device, dtype=dtype)


def apply_rope(x, cos, sin):
    """x [B, H, L, hd], half-split rotation."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).to(x.dtype)


# ----------------------------------------------------------------------------------------------- attention backends
def ref_mask(L, kind, window, chunk, device):
    i = torch.arange(L, device=device)
    q, k = i[:, None], i[None, :]
    if kind == 'full':
        return k <= q
    if kind == 'swa':
        return (k <= q) & (q - k < window)
    if kind == 'chunk':
        return (k // chunk) < (q // chunk)
    raise ValueError(kind)


def ref_attention(q, k, v, mask):
    """Explicit masked softmax attention; a row without keys returns 0 (the cache's chunk-0 rows)."""
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(~mask, float('-inf'))
    anyk = mask.any(-1, keepdim=True)
    s = torch.where(anyk, s, torch.zeros_like(s))
    p = torch.softmax(up32(s), -1).to(v.dtype) * anyk.to(v.dtype)
    return p @ v


class Attn:
    """Attention dispatcher chosen per run: 'ref' (explicit), 'sdpa' (PyTorch SDPA causal, default dispatch =
    FlashAttention-2 for BF16; full only), 'cudnn' (SDPA cuDNN backend; full only), 'sdpa_mask' (SDPA with an explicit
    boolean mask, memory-efficient backend), 'flex' (FlexAttention block-sparse masks). The cache's chunk-causal
    attention always uses flex unless the backend is 'ref'."""

    def __init__(self, a, backend, device):
        self.a, self.backend, self.device = a, backend, device
        self.masks, self.block = {}, {}
        if backend != 'ref':
            from torch.nn.attention.flex_attention import create_block_mask
            W, C = a.window, a.chunk

            def full(b, h, q, k):
                return k <= q

            def swa(b, h, q, k):
                return (k <= q) & (q - k < W)

            def chunk(b, h, q, k):
                return (k // C) < (q // C)
            fns = dict(full=full, swa=swa, chunk=chunk)
            need = {a.attn} | ({'chunk'} if a.tail == 'cache' else set())
            for kind in need:
                self.block[kind] = create_block_mask(fns[kind], None, None, a.ctx, a.ctx, device=device)

    def mask(self, kind, L, device):
        key = (kind, L)
        if key not in self.masks:
            self.masks[key] = ref_mask(L, kind, self.a.window, self.a.chunk, device)
        return self.masks[key]

    def __call__(self, q, k, v, kind):
        be = self.backend
        if be == 'ref':
            return ref_attention(q, k, v, self.mask(kind, q.shape[-2], q.device))
        if kind == 'chunk':
            be = 'flex'
        if be in ('sdpa', 'cudnn'):
            if kind != 'full':
                raise ValueError(be + ' backend supports full causal attention only')
            if be == 'sdpa':
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if be == 'sdpa_mask':
            return F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(kind, q.shape[-2], q.device))
        from torch.nn.attention.flex_attention import flex_attention
        return flex_attention(q, k, v, block_mask=self.block[kind])


# ------------------------------------------------------------------------------------------------------ model pieces
def trunk(P, a, x, attn, rope):
    """Embedding and blocks through the last block's attention. x [B, L] int64. Returns u [B, L, d]."""
    cos, sin = rope
    B, L = x.shape
    z = F.embedding(x, P[EMB])
    kind = a.attn
    for i in range(a.layers):
        p = f'blocks.{i}.'
        hn = rmsnorm(z, P[p + 'attn_norm'], a.eps)
        qkv = (hn @ P[p + 'qkv'].T).view(B, L, 3, a.heads, a.hd).permute(2, 0, 3, 1, 4)
        q, k, v = apply_rope(qkv[0], cos[:L], sin[:L]), apply_rope(qkv[1], cos[:L], sin[:L]), qkv[2]
        o = attn(q, k, v, kind)
        z = z + o.transpose(1, 2).reshape(B, L, a.d) @ P[p + 'out'].T
        if i < a.layers - 1:
            hf = rmsnorm(z, P[p + 'ffn_norm'], a.eps)
            z = z + F.silu(hf @ P[p + 'up'].T) @ P[p + 'down'].T
    return z


def up32(t):
    """At least FP32 (FP64 stays FP64 for the CPU checks)."""
    return t if t.dtype == torch.float64 else t.float()


def head_ce(r, P, a, y):
    """Final norm, tied head, per-target cross entropy in FP32. r [B, S, d], y [B, S]."""
    logits = up32(rmsnorm(r, P['final_norm'], a.eps) @ P[EMB].T)
    return torch.logsumexp(logits, -1) - logits.gather(-1, y[..., None])[..., 0]


def dense_tail(P, a, u, y):
    last = f'blocks.{a.layers - 1}.'
    h = rmsnorm(u, P[last + 'ffn_norm'], a.eps)
    return head_ce(u + F.silu(h @ P[last + 'up'].T) @ P[last + 'down'].T, P, a, y)


def cache_residual(P, a, h, y, attn):
    B, L, d = h.shape
    H = a.cache_heads
    hd = d // H
    q = (h @ P['cache.q'].T).view(B, L, H, hd).transpose(1, 2)
    k = (h @ P['cache.k'].T).view(B, L, H, hd).transpose(1, 2)
    v = ((h + F.embedding(y, P[EMB]).to(h.dtype)) @ P['cache.v'].T).view(B, L, H, hd).transpose(1, 2)
    o = attn(q, k, v, 'chunk').transpose(1, 2).reshape(B, L, d)
    return (o @ P['cache.o'].T) * INV_SQRT2


def cache_tail(P, a, u, y, attn):
    last = f'blocks.{a.layers - 1}.'
    h = rmsnorm(u, P[last + 'ffn_norm'], a.eps)
    r = u + F.silu(h @ P[last + 'up'].T) @ P[last + 'down'].T + cache_residual(P, a, h, y, attn)
    return head_ce(r, P, a, y)


def memory_pre(P, a, u):
    """Fast-independent tail work: h = RMSNorm(u) g, static branch output."""
    last = f'blocks.{a.layers - 1}.'
    h = rmsnorm(u, P[last + 'ffn_norm'], a.eps)
    ms = F.silu(h @ P[last + 'static_up'].T) @ P[last + 'static_down'].T
    return h, ms


def memory_chunk(u_c, h_c, ms_c, y_c, U, Dn, gf, E, eta, eps, *, y_write=None):
    """Score chunk c with (U, Dn) = W_c, then write W_{c+1} = W_c - eta * grad CE_c(W_c) (chunk-mean CE per sequence).
    u_c/h_c/ms_c [B, C, d], y_c [B, C], U [B, m, d], Dn [B, d, m], eta a 0-d tensor (0 = writes off, exactly).
    The inner gradient is written explicitly (closed form) so outer autograd differentiates through it exactly."""
    zf = torch.bmm(h_c, U.transpose(1, 2).to(h_c.dtype))
    sf = torch.sigmoid(zf)
    af = zf * sf
    mf = torch.bmm(af, Dn.transpose(1, 2).to(af.dtype))
    r = u_c + (ms_c + mf) * INV_SQRT2
    rr = torch.rsqrt(r.square().mean(-1, keepdim=True) + eps)
    rh = r * rr
    logits = up32((rh * gf) @ E.T)
    lse = torch.logsumexp(logits, -1)
    ce = lse - logits.gather(-1, y_c[..., None])[..., 0]
    y_w = y_c if y_write is None else y_write        # y_write: deliberate fault injection only
    C = y_c.shape[1]
    dl = (torch.softmax(logits, -1) - F.one_hot(y_w, logits.shape[-1]).to(logits.dtype)) / C
    do = (dl.to(E.dtype) @ E)
    drh = do * gf
    dr = rr * (drh - rh * (drh * rh).mean(-1, keepdim=True))
    dmf = dr * INV_SQRT2
    dD = torch.bmm(dmf.transpose(1, 2), af.to(dmf.dtype))
    daf = torch.bmm(dmf, Dn.to(dmf.dtype))
    dz = daf * (sf * (1 + zf * (1 - sf))).to(daf.dtype)
    dU = torch.bmm(dz.transpose(1, 2), h_c.to(dz.dtype))
    return ce, U - eta * dU.to(U.dtype), Dn - eta * dD.to(Dn.dtype)


def memory_tail(P, a, u, y, eta, chunk_fn=None, *, mutant=None, pre_fn=None):
    B, L, _ = u.shape
    C = a.chunk
    h, ms = (pre_fn or memory_pre)(P, a, u)
    U = P['fast_init.up'].expand(B, -1, -1).contiguous()
    Dn = P['fast_init.down'].expand(B, -1, -1).contiguous()
    fn = chunk_fn or memory_chunk
    ces = []
    for c in range(L // C):
        sl = slice(c * C, (c + 1) * C)
        if mutant == 'read_after_write':
            _, U, Dn = fn(u[:, sl], h[:, sl], ms[:, sl], y[:, sl], U, Dn, P['final_norm'], P[EMB], eta, a.eps)
            ce, _, _ = fn(u[:, sl], h[:, sl], ms[:, sl], y[:, sl], U, Dn, P['final_norm'], P[EMB], eta, a.eps)
        elif mutant == 'future_label':     # fault: chunk c's write uses the labels of chunk c + 1
            nx = slice(((c + 1) % (L // C)) * C, ((c + 1) % (L // C) + 1) * C)
            ce, U, Dn = memory_chunk(u[:, sl], h[:, sl], ms[:, sl], y[:, sl], U, Dn, P['final_norm'], P[EMB], eta,
                                     a.eps, y_write=y[:, nx])
        else:
            ce, U, Dn = fn(u[:, sl], h[:, sl], ms[:, sl], y[:, sl], U, Dn, P['final_norm'], P[EMB], eta, a.eps)
        ces.append(ce)
    return torch.cat(ces, 1)


def forward_ce(P, a, x, y, attn, rope, eta=None, *, chunk_fn=None, trunk_fn=None, mutant=None):
    """Per-target cross entropy [B, L] (FP32) for inputs x and targets y (y[:, t] = next byte after x[:, t])."""
    u = (trunk_fn or trunk)(P, a, x, attn, rope)
    if a.tail == 'dense':
        return dense_tail(P, a, u, y)
    if a.tail == 'cache':
        return cache_tail(P, a, u, y, attn)
    return memory_tail(P, a, u, y, eta, chunk_fn, mutant=mutant)


def flops_per_update(a, tokens):
    """Analytic matmul FLOPs (forward + backward = 3x forward; memory tail second-order approximated as 2x its
    first-order cost, S43G measured 1.65x). Attention counts only allowed (q, k) pairs."""
    L = a.ctx
    d, hd = a.d, a.hd
    dense_per_tok = 2 * (a.layers * (3 * d * d + d * d) + a.layers * 2 * d * a.hidden + a.vocab * d)
    if a.attn == 'full':
        pairs = L * (L + 1) / 2
    else:
        W = a.window
        pairs = sum(min(t + 1, W) for t in range(L))
    attn_per_seq = a.layers * 2 * 2 * pairs * d          # QK^T and PV
    seqs = tokens / L
    fwd = dense_per_tok * tokens + attn_per_seq * seqs
    extra = 0.0
    if a.tail == 'cache':
        C = a.chunk
        cpairs = sum((t // C) * C for t in range(L))
        extra = 3 * (4 * 2 * d * d * tokens + 2 * 2 * cpairs * d * seqs)
    if a.tail == 'memory':
        # per target: inner backward through fast branch (2 bmm) + head gradient (V x d twice) ~ forward cost again
        inner = 2 * (2 * 2 * d * a.m_fast + 2 * a.vocab * d) * tokens
        extra = 2 * 3 * inner
    return dict(forward=fwd, total=3 * fwd + extra, attention_total=3 * attn_per_seq * seqs, tail_extra=extra,
                attention_pairs_per_seq=pairs)
