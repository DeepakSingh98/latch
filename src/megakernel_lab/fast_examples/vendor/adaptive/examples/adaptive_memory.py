"""Adaptive-memory byte LM (S106B arm M) as ordinary PyTorch over the loop API.

Model and learning semantics are those of the closed S106B lane (tag
s106b-final, ``research/s106/S106B/s106b/model.py``; vendored unchanged in
``cops4_a/s106b`` as the conventional reference): vocabulary 256, RMSNorm,
RoPE (half-split), sliding-window causal attention in every block, tied
embedding, and a last FFN split into a static branch and a fast branch whose
per-sequence weights W_c are *written* after scoring each chunk:

    score chunk c with W_c, then W_{c+1} = W_c - eta * dCE_c/dW_c

using only chunk c's labels (already scored). The inner gradient is the
donor's closed form, written as ordinary operations, so outer differentiation
passes exactly (second order) through every write. Nothing is detached.

Expression differences from the donor source, all exact reformulations of the
same function (no learning or numerical-policy change):

* Sliding-window attention is written block-locally: query block n attends to
  key blocks n-1 and n with an additive mask (0 / float32 min) selecting
  exactly the donor's keys ``t-window < s <= t``; masked probabilities are 0.
* The chunk loop is the compiler's bounded device loop. The trunk and the
  fast-independent part of the tail (u, h = RMSNorm(u), static branch) form the
  loop's sequence function (bulk work, computed once per update); chunk c reads
  slice c. Losses accumulate as a carried FP32 sum divided by B*L at the end.
* ``F.one_hot`` is written as the scatter it performs.

Optimizer: the donor's Muon (NS5 BF16, gain, momentum 0.95) + AdamW recipe via
``ops.muon.MuonConfig`` with the donor's runtime schedule vector
``sc = [lr, 1-lr*wd, lr/bc1, 1/sqrt(bc2), -gain*lr, eta]`` as a data input.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn.functional as F

INV_SQRT2 = 1.0 / math.sqrt(2.0)
EMB = 'embedding.weight'
WD, B1, B2 = 0.1, 0.9, 0.95


@dataclass(frozen=True)
class Arch:
    """Donor ``s106b.model.Arch`` fields for arm M, plus the per-update batch."""
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
    rope_base: float = 10000.0
    eps: float = 1e-6
    batch: int = 2

    @property
    def hd(self):
        return self.d // self.heads

    @property
    def chunks(self):
        return self.ctx // self.chunk

    def validate(self):
        if self.ctx % self.chunk or self.ctx % self.window or self.d % self.heads or self.hd % 2:
            raise ValueError('require ctx divisible by chunk and window, even head width')

    def donor(self):
        """The equivalent donor Arch (conventional reference)."""
        from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b import model as MD
        return MD.Arch(arm='M', ctx=self.ctx, window=self.window, chunk=self.chunk, d=self.d, heads=self.heads,
                       layers=self.layers, hidden=self.hidden, vocab=self.vocab, m_static=self.m_static,
                       m_fast=self.m_fast, rope_base=self.rope_base, eps=self.eps)


RECIPES = {
    # Donor TINY (s106b.model.TINY): the smallest faithful case for FP64 / finite differences.
    'small': Arch(ctx=256, window=32, chunk=32, d=32, heads=4, layers=2, hidden=64, vocab=64, m_static=32,
                  m_fast=32, batch=2),
    'tiny': Arch(ctx=48, window=6, chunk=8, d=16, heads=2, layers=2, hidden=24, vocab=11, m_static=12, m_fast=12,
                 batch=2),
    # S106B M8192: 4,787,456 parameters, 2 x 8192 tokens, 64 writes per sequence.
    'm8k': Arch(),
    # Same model and tokens per update at S106B's 2K arm shape: 8 x 2048, 16 writes per sequence.
    'm2k': Arch(ctx=2048, batch=8),
    # Longer chain (S107B unseen M8Kc64 declaration): 128 writes of 64 per sequence.
    'm8k_c64': Arch(chunk=64),
    # Wider state/trunk (S106E/S107B wide declaration, d512 / head 64 / fast 1024), 2K tokens x 8.
    'w2k': Arch(ctx=2048, batch=8, d=512, heads=8, layers=6, hidden=2048, m_static=1024, m_fast=1024),
}


def inventory(a):
    """(name, shape, kind) in the donor's canonical order for arm M."""
    out = [(EMB, (a.vocab, a.d), 'emb')]
    for i in range(a.layers):
        p = f'blocks.{i}.'
        out += [(p + 'attn_norm', (a.d,), 'gain'), (p + 'qkv', (3 * a.d, a.d), 'matrix'),
                (p + 'out', (a.d, a.d), 'matrix'), (p + 'ffn_norm', (a.d,), 'gain')]
        if i < a.layers - 1:
            out += [(p + 'up', (a.hidden, a.d), 'matrix'), (p + 'down', (a.d, a.hidden), 'matrix')]
        else:
            out += [(p + 'static_up', (a.m_static, a.d), 'matrix'), (p + 'static_down', (a.d, a.m_static), 'matrix'),
                    ('fast_init.up', (a.m_fast, a.d), 'matrix'), ('fast_init.down', (a.d, a.m_fast), 'matrix')]
    out.append(('final_norm', (a.d,), 'gain'))
    return out


def init_params(a, seed, device='cpu', dtype=torch.float32):
    """Donor initialization: N(0, 0.02) matrices/embedding, unit gains, inventory order, one float64 generator."""
    g = torch.Generator().manual_seed(int(seed))
    P = {}
    for n, shape, kind in inventory(a):
        t = (torch.ones(shape, dtype=torch.float64) if kind == 'gain'
             else torch.randn(shape, generator=g, dtype=torch.float64) * 0.02)
        P[n] = t.to(device=device, dtype=dtype)
    return P


def muon_config(a, schedule_input=2):
    from megakernel_lab.operation_compiler.muon import MuonConfig
    inv = inventory(a)
    return MuonConfig(muon_parameters=tuple(n for n, _, k in inv if k == 'matrix'), decay_parameters=(EMB,),
                      momentum=0.95, ns_steps=5, betas=(B1, B2), eps=1e-8, clip=1.0, schedule_input=schedule_input)


def schedule_rows(H, lr, eta, gain=5.0):
    """Donor Engine.schedule: warmup H/16 then cosine to 10%; returns float32 [H, 6]."""
    rows = []
    warm = max(1, H // 16)
    for step in range(1, H + 1):
        if step <= warm:
            l = lr * step / warm
        else:
            t = min(1.0, (step - warm) / max(1, H - warm))
            l = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))
        bc1, bc2 = 1 - B1 ** step, 1 - B2 ** step
        rows.append([l, 1 - l * WD, l / bc1, 1 / math.sqrt(bc2), -gain * l, eta])
    return torch.tensor(rows, dtype=torch.float32)


# ------------------------------------------------------------------------------------------------ constants
def rope_tables(a, dtype=torch.float32):
    half = a.hd // 2
    inv = 1.0 / (a.rope_base ** (torch.arange(half, dtype=torch.float64) / half))
    ang = torch.arange(a.ctx, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos().to(dtype), ang.sin().to(dtype)


def window_mask(a, dtype=torch.float32):
    """Additive [blocks, W, 2W] mask: key j of (previous, current) block is visible to query i iff
    its position is in (i - W, i] (and not before the sequence start)."""
    W, nb = a.window, a.ctx // a.window
    i = torch.arange(W)[:, None]
    j = torch.arange(2 * W)[None, :]
    visible = (j > i) & (j <= i + W)
    first = visible & (j >= W)
    allowed = torch.stack([first] + [visible] * (nb - 1))
    return torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), torch.finfo(dtype).min, dtype=dtype))


def constants(a):
    cos, sin = rope_tables(a)
    return cos, sin, window_mask(a)


# ---------------------------------------------------------------------------------------------------- model
def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)


def window_attention(q, k, v, mask, window):
    """Exact sliding-window causal attention, block-local. q/k/v [B, H, L, hd]."""
    B, H, L, hd = q.shape
    W, nb = window, L // window
    kb, vb = k.reshape(B * H, nb, W, hd), v.reshape(B * H, nb, W, hd)
    keys = torch.cat((torch.cat((torch.zeros_like(kb[:, :1]), kb[:, :-1]), 1), kb), 2).reshape(B * H * nb, 2 * W, hd)
    vals = torch.cat((torch.cat((torch.zeros_like(vb[:, :1]), vb[:, :-1]), 1), vb), 2).reshape(B * H * nb, 2 * W, hd)
    s = torch.bmm(q.reshape(B * H * nb, W, hd), keys.transpose(1, 2)) / math.sqrt(hd)
    s = (s.reshape(B * H, nb, W, 2 * W) + mask).reshape(B * H * nb, W, 2 * W)
    p = torch.softmax(s, -1)
    return torch.bmm(p, vals).reshape(B, H, L, hd)


def trunk(P, a, x, cos, sin, mask):
    """Embedding and blocks through the last block's attention. x [B, L] int64 -> u [B, L, d]."""
    B, L = x.shape
    z = F.embedding(x, P[EMB])
    for i in range(a.layers):
        p = f'blocks.{i}.'
        hn = rmsnorm(z, P[p + 'attn_norm'], a.eps)
        qkv = (hn @ P[p + 'qkv'].T).view(B, L, 3, a.heads, a.hd).permute(2, 0, 3, 1, 4)
        q, k, v = apply_rope(qkv[0], cos, sin), apply_rope(qkv[1], cos, sin), qkv[2]
        o = window_attention(q, k, v, mask, a.window)
        z = z + o.transpose(1, 2).reshape(B, L, a.d) @ P[p + 'out'].T
        if i < a.layers - 1:
            hf = rmsnorm(z, P[p + 'ffn_norm'], a.eps)
            z = z + F.silu(hf @ P[p + 'up'].T) @ P[p + 'down'].T
    return z


def memory_pre(P, a, u):
    last = f'blocks.{a.layers - 1}.'
    h = rmsnorm(u, P[last + 'ffn_norm'], a.eps)
    ms = F.silu(h @ P[last + 'static_up'].T) @ P[last + 'static_down'].T
    return h, ms


def memory_chunk(u_c, h_c, ms_c, y_c, U, Dn, gf, E, eta, eps, *, mutant=None):
    """Donor memory_chunk: score chunk c with (U, Dn) = W_c, then write W_{c+1} (closed-form inner gradient).

    ``mutant='detach_write'`` is a deliberate fault (writes without derivatives) used only by checks."""
    zf = torch.bmm(h_c, U.transpose(1, 2))
    sf = torch.sigmoid(zf)
    af = zf * sf
    mf = torch.bmm(af, Dn.transpose(1, 2))
    r = u_c + (ms_c + mf) * INV_SQRT2
    rr = torch.rsqrt(r.square().mean(-1, keepdim=True) + eps)
    rh = r * rr
    logits = (rh * gf) @ E.T
    lse = torch.logsumexp(logits, -1)
    ce = lse - logits.gather(-1, y_c[..., None])[..., 0]
    C = y_c.shape[1]
    onehot = torch.zeros_like(logits).scatter(-1, y_c[..., None], 1.0)
    dl = (torch.softmax(logits, -1) - onehot) / C
    do = dl @ E
    drh = do * gf
    dr = rr * (drh - rh * (drh * rh).mean(-1, keepdim=True))
    dmf = dr * INV_SQRT2
    dD = torch.bmm(dmf.transpose(1, 2), af)
    daf = torch.bmm(dmf, Dn)
    dz = daf * (sf * (1 + zf * (1 - sf)))
    dU = torch.bmm(dz.transpose(1, 2), h_c)
    if mutant == 'detach_write':
        dU, dD = dU.detach(), dD.detach()
    return ce, U - eta * dU, Dn - eta * dD


def chunk_major(t, a):
    """[B, L, ...] -> [chunks, B, C, ...] (contiguous)."""
    B = t.shape[0]
    return t.reshape(B, a.chunks, a.chunk, *t.shape[2:]).transpose(0, 1).contiguous()


def make_loop_functions(a, mutant=None):
    """Loop functions over explicit inputs (x, y, sc, cos, sin, mask). ``mutant`` is for fault checks only."""
    a.validate()
    last = f'blocks.{a.layers - 1}.'

    def sequence(P, x, y, sc, cos, sin, mask):
        u = trunk(P, a, x, cos, sin, mask)
        h, ms = memory_pre(P, a, u)
        return {'u': chunk_major(u, a), 'h': chunk_major(h, a), 'ms': chunk_major(ms, a), 'y': chunk_major(y, a)}

    def init(P, x, y, sc, cos, sin, mask):
        B = x.shape[0]
        return {'U': P['fast_init.up'].expand(B, -1, -1).contiguous(),
                'Dn': P['fast_init.down'].expand(B, -1, -1).contiguous(),
                'ce': torch.zeros((), dtype=torch.float32, device=x.device)}

    def body(P, carry, state, xs, x, y, sc, cos, sin, mask):
        ce, U, Dn = memory_chunk(xs['u'], xs['h'], xs['ms'], xs['y'], carry['U'], carry['Dn'],
                                 P['final_norm'], P[EMB], sc[5], a.eps, mutant=mutant)
        return {'U': U, 'Dn': Dn, 'ce': carry['ce'] + ce.sum()}, {}

    def terminal(P, carry, x, y, sc, cos, sin, mask):
        return carry['ce'] / (x.shape[0] * x.shape[1])

    return sequence, init, body, terminal


def reference_loss(P, a, x, y, sc, cos, sin, mask, count=None, mutant=None):
    """Eager evaluation of the same functions (count chunks); used by CPU checks."""
    sequence, init, body, terminal = make_loop_functions(a, mutant)
    xs = sequence(P, x, y, sc, cos, sin, mask)
    carry, state = init(P, x, y, sc, cos, sin, mask), {}
    for c in range(a.chunks if count is None else count):
        carry, state = body(P, carry, state, {k: v[c] for k, v in xs.items()}, x, y, sc, cos, sin, mask)
    return terminal(P, carry, x, y, sc, cos, sin, mask)


def batch_inputs(a, tokens, sc):
    """tokens int64 [B, ctx+1] -> explicit loop inputs (x, y, sc, cos, sin, mask)."""
    cos, sin, mask = constants(a)
    return (tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous(), sc.to(torch.float32).contiguous(),
            cos, sin, mask)


def synthetic_tokens(a, seed):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, a.vocab, (a.batch, a.ctx + 1), generator=g, dtype=torch.int64)


def capture(a, params, inputs, *, optimizer_state=None, storage_mode='save_body_tape', **options):
    from megakernel_lab.operation_compiler.loop import capture_loop_training
    sequence, init, body, terminal = make_loop_functions(a)
    count = torch.tensor(a.chunks, dtype=torch.int64)
    return capture_loop_training(init, body, terminal, params, None, inputs, count, muon_config(a),
                                 max_steps=a.chunks, storage_mode=storage_mode, optimizer_state=optimizer_state,
                                 sequence_fn=sequence, **options)
