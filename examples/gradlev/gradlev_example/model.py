# Provenance: copied from the Latch repository, commit df184a5f2579
# (branch codex/latch-gradlev), research/latch_gradlev/gl/model.py (git blob 056a2b9579af324b3da17ce68f676605ef4b9747).
# Changes: none to the code (byte-identical below this header). The prefiller Q and gradlev_loss/static_loss belong to helper training, which this example does not run; they are kept so the module stays identical to the donor.
"""GPT-2 small with a GradLev-adapted final MLP (paper-based adaptation of arXiv 2609.34174v2).

P is the public GPT-2 small (openai-community/gpt2, MIT) re-implemented in plain PyTorch so
the same parameters drive training, the conventional deployment loop, the Latch capture and
the FP64 references. Only the final block's MLP (h.11.mlp.c_fc / c_proj, GPT-2 Conv1D layout
x @ W) is adapted, as in the paper's final-MLP scope (Sec. 5.2). See SPEC.md for every choice
the paper leaves open.

Layout: W1 = c_fc.weight [d, m], W2 = c_proj.weight [m, d]; z = x @ W1 + b1, h = gelu_tanh(z),
y = h @ W2 + b2. The paper's m x d matrix W1 and write -eta g x^T appear here transposed:
W1 <- W1 - eta1 x g_z^T, W2 <- W2 - eta2 h g_y^T (the exact negative per-token gradients).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import torch
import torch.nn.functional as F

D, M, V, L, H, CTX = 768, 3072, 50257, 12, 12, 1024
EOT = 50256
LN_EPS = 1e-5
MU_MAX = 4.0          # paper: write-strength cap mu_max = 4
MU_INIT = 0.9         # paper: write strength starts at 0.9 (w = 0, b = log(0.9/3.1))
GATE_BIAS0 = math.log(MU_INIT / (MU_MAX - MU_INIT))
RMS_EPS = 1e-6


def hp(t):
    """Upcast 16-bit floats to FP32; FP32/FP64 unchanged (keeps FP64 diagnostics exact)."""
    return t.float() if t.dtype in (torch.bfloat16, torch.float16) else t


def gelu_tanh(z):
    return 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * z * z * z)))


def gelu_tanh_grad(z):
    """d/dz of the tanh-approximate GELU, written with forward operations only."""
    c = math.sqrt(2.0 / math.pi)
    inner = c * (z + 0.044715 * z * z * z)
    t = torch.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * z * z)


def layer_norm_parts(r, w, b):
    mean = r.mean(-1, keepdim=True)
    c = r - mean
    rstd = torch.rsqrt((c * c).mean(-1, keepdim=True) + LN_EPS)
    nh = c * rstd
    return nh * w + b, nh, rstd


def layer_norm_backward(gn, nh, rstd, w):
    gh = gn * w
    return rstd * (gh - gh.mean(-1, keepdim=True) - nh * (gh * nh).mean(-1, keepdim=True))


def head_costate(logits, targets, E, nh, rstd, lnf_w):
    """Per-token dl_t/dy through the tied head and LN_f (no batch/sequence averaging), no autograd.

    The head-gradient product takes BF16 operands when E/autocast is BF16, as in deployment.
    """
    gl = torch.softmax(hp(logits), -1)
    gl.scatter_add_(-1, targets[..., None], torch.full_like(targets[..., None], -1, dtype=gl.dtype))
    gn = gl.to(logits.dtype) @ E
    return layer_norm_backward(hp(gn).to(nh.dtype), nh, rstd, lnf_w.to(nh.dtype))


# ----------------------------------------------------------------------------- weights

def read_safetensors(path):
    """Minimal safetensors reader (format: u64 header length, JSON header, raw little-endian data)."""
    raw = Path(path).read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    base = 8 + n
    dtypes = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16}
    out = {}
    for key, meta in header.items():
        if key == '__metadata__':
            continue
        lo, hi = meta['data_offsets']
        tensor = torch.frombuffer(bytearray(raw[base + lo:base + hi]), dtype=dtypes[meta['dtype']])
        out[key] = tensor.reshape(meta['shape']).clone()
    return out


def gpt2_params(path):
    """FP32 parameter dict with canonical names; wte is tied to the output head."""
    raw = read_safetensors(path)
    raw = {k.removeprefix('transformer.'): v for k, v in raw.items()}
    p = {'wte': raw['wte.weight'], 'wpe': raw['wpe.weight'],
         'lnf_w': raw['ln_f.weight'], 'lnf_b': raw['ln_f.bias']}
    for i in range(L):
        s = f'h.{i}.'
        p.update({f'{i}.ln1_w': raw[s + 'ln_1.weight'], f'{i}.ln1_b': raw[s + 'ln_1.bias'],
                  f'{i}.qkv_w': raw[s + 'attn.c_attn.weight'], f'{i}.qkv_b': raw[s + 'attn.c_attn.bias'],
                  f'{i}.ao_w': raw[s + 'attn.c_proj.weight'], f'{i}.ao_b': raw[s + 'attn.c_proj.bias'],
                  f'{i}.ln2_w': raw[s + 'ln_2.weight'], f'{i}.ln2_b': raw[s + 'ln_2.bias'],
                  f'{i}.fc_w': raw[s + 'mlp.c_fc.weight'], f'{i}.fc_b': raw[s + 'mlp.c_fc.bias'],
                  f'{i}.pr_w': raw[s + 'mlp.c_proj.weight'], f'{i}.pr_b': raw[s + 'mlp.c_proj.bias']})
    p = {k: v.float().contiguous() for k, v in p.items()}
    assert p['wte'].shape == (V, D) and p[f'{L-1}.fc_w'].shape == (D, M)
    return p


def gate_params():
    return {'gate_w': torch.zeros(D), 'gate_b': torch.full((), GATE_BIAS0)}


# ----------------------------------------------------------------------------- trunk

def _attention(p, i, x):
    B, T, _ = x.shape
    qkv = F.linear(x, p[f'{i}.qkv_w'].t(), p[f'{i}.qkv_b'])
    q, k, v = qkv.split(D, dim=-1)
    q, k, v = (t.view(B, T, H, D // H).transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    o = o.transpose(1, 2).reshape(B, T, D)
    return F.linear(o, p[f'{i}.ao_w'].t(), p[f'{i}.ao_b'])


def _mlp(p, i, x):
    return F.linear(gelu_tanh(F.linear(x, p[f'{i}.fc_w'].t(), p[f'{i}.fc_b'])), p[f'{i}.pr_w'].t(), p[f'{i}.pr_b'])


def trunk(p, tokens):
    """Everything before the adapted map: blocks 0..L-2 and the last block's attention.

    Returns (u, x): u is the residual entering the final MLP, x = ln_2(u) its normalized input.
    Neither depends on the fast weights (the adapted map feeds only ln_f and the head), so a
    causal deployment may compute them for all positions at once without changing any read.
    Caller controls precision (training and deployment run this under BF16 autocast).
    """
    B, T = tokens.shape
    h = F.embedding(tokens, p['wte']) + p['wpe'][:T]
    for i in range(L):
        h = h + _attention(p, i, F.layer_norm(h, (D,), p[f'{i}.ln1_w'], p[f'{i}.ln1_b'], LN_EPS))
        if i == L - 1:
            break
        h = h + _mlp(p, i, F.layer_norm(h, (D,), p[f'{i}.ln2_w'], p[f'{i}.ln2_b'], LN_EPS))
    u = h
    x = F.layer_norm(u, (D,), p[f'{L-1}.ln2_w'], p[f'{L-1}.ln2_b'], LN_EPS)
    return u, x


def gates(g, x):
    """mu_t = mu_max sigmoid(w . sg(x_t) + b); eta1 = mu/d, eta2 = mu/m (paper Eq. 18)."""
    mu = MU_MAX * torch.sigmoid(x.detach().float() @ g['gate_w'].float() + g['gate_b'].float())
    return mu / D, mu / M


def static_logits(p, u, x):
    y = _mlp(p, L - 1, x)
    n = F.layer_norm(u + y, (D,), p['lnf_w'], p['lnf_b'], LN_EPS)
    return F.linear(n, p['wte'])


# ----------------------------------------------------------------------------- prefiller Q

def rope(x):
    """Rotary positions (base 10000) on [B, H, T, hd]."""
    B, Hh, T, hd = x.shape
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    ang = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def rmsnorm(x, w=None, eps=RMS_EPS):
    y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
    return y if w is None else y * w


def q_params(seed=0, blocks=2):
    """Two pre-norm causal blocks (RMSNorm, RoPE, GELU MLP) and two zero-initialized heads."""
    g = torch.Generator().manual_seed(seed)
    std = 0.02
    q = {}
    for i in range(blocks):
        q[f'q{i}.n1'] = torch.ones(D)
        q[f'q{i}.qkv'] = torch.randn(D, 3 * D, generator=g) * std
        q[f'q{i}.o'] = torch.randn(D, D, generator=g) * std / math.sqrt(2 * blocks)
        q[f'q{i}.n2'] = torch.ones(D)
        q[f'q{i}.fc'] = torch.randn(D, 4 * D, generator=g) * std
        q[f'q{i}.pr'] = torch.randn(4 * D, D, generator=g) * std / math.sqrt(2 * blocks)
    q['qn'] = torch.ones(D)
    q['head_z'] = torch.zeros(D, M)   # paper: Q's heads start at zero
    q['head_y'] = torch.zeros(D, D)
    return q


def prefiller(q, x, next_tokens, wte, blocks=2):
    """Q(sg(x_t) + RMSNorm(sg(E s_{t+1}))) -> (a_z [B,T,m], a_y [B,T,d]); causal within the row."""
    h = x.detach() + rmsnorm(F.embedding(next_tokens, wte.detach()))
    B, T, _ = h.shape
    for i in range(blocks):
        a = rmsnorm(h, q[f'q{i}.n1'])
        qq, kk, vv = (a @ q[f'q{i}.qkv']).split(D, dim=-1)
        qq, kk, vv = (t.view(B, T, H, D // H).transpose(1, 2) for t in (qq, kk, vv))
        o = F.scaled_dot_product_attention(rope(qq), rope(kk), vv, is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, D) @ q[f'q{i}.o']
        h = h + gelu_tanh(rmsnorm(h, q[f'q{i}.n2']) @ q[f'q{i}.fc']) @ q[f'q{i}.pr']
    h = rmsnorm(h, q['qn'])
    return h @ q['head_z'], h @ q['head_y']


# ----------------------------------------------------------------------------- GradLev training forward

def gradlev_loss(p, g, q, tokens, targets, lam=1.0, proposals=None):
    """Paper Eq. 3-5/13-14 on rows that each hold one document segment (fresh state per row).

    With retention alpha = 1 the exclusive scans are strictly-causal Gram products, computed here in
    their exact quadratic form: x_t @ Delta_t = -sum_{i<t} eta_i (x_t . x_i) g_i.
    Returns (total, ce_mean, consistency, diagnostics).
    """
    W1, b1, W2, b2 = p[f'{L-1}.fc_w'], p[f'{L-1}.fc_b'], p[f'{L-1}.pr_w'], p[f'{L-1}.pr_b']
    u, x = trunk(p, tokens)
    B, T, _ = x.shape
    eta1, eta2 = gates(g, x)                                   # [B,T] fp32, gate derivatives kept
    strict = torch.ones(T, T, device=x.device, dtype=torch.bool).tril(-1)
    z0 = x @ W1 + b1
    if proposals is None:
        a_z, a_y = prefiller(q, x, targets, p['wte'])
        gz_hat = gelu_tanh_grad(z0.detach().float()).to(a_z.dtype) * a_z   # Eq. 17
        gy_hat = a_y
    else:   # exactness diagnostics only: externally supplied costates (e.g. the serial learner's)
        gz_hat, gy_hat = proposals
    # layer 1 forward scan: reads keep x_t / gate derivatives; writes use sg(x_i), sg(g_i)
    s1 = (x @ x.detach().transpose(1, 2)).masked_fill(~strict, 0)
    z = z0 - s1 @ (eta1[..., None].to(x.dtype) * gz_hat.detach())
    h = gelu_tanh(z)
    s2 = (h @ h.detach().transpose(1, 2)).masked_fill(~strict, 0)
    y = h @ W2 + b2 - s2 @ (eta2[..., None].to(h.dtype) * gy_hat.detach())
    r = hp(u + y)
    n, nh, rstd = layer_norm_parts(r, p['lnf_w'], p['lnf_b'])
    logits = F.linear(n, p['wte'])
    ce = F.cross_entropy(hp(logits).reshape(-1, V), targets.reshape(-1), reduction='none').view(B, T)
    # reverse reconstruction (targets are stop-gradient): per-token costates before any averaging
    with torch.no_grad():
        gy_t = head_costate(logits.detach(), targets, p['wte'].detach(), nh.detach(), rstd.detach(),
                            p['lnf_w'].detach()).float()
        hf = h.detach().float()
        r2 = (gy_t @ gy_hat.detach().float().transpose(1, 2)).masked_fill(~strict, 0)
        gh = gy_t @ W2.detach().float().t() - r2 @ (eta2.detach()[..., None] * hf)
        gz_t = gelu_tanh_grad(z.detach().float()) * gh
    cons = 0.5 * (((gz_hat.float() - gz_t) ** 2).sum(-1) / M + ((gy_hat.float() - gy_t) ** 2).sum(-1) / D).mean()
    ce_mean = ce.mean()
    diag = {'gz_rel': ((hp(gz_hat) - gz_t).norm() / gz_t.norm().clamp_min(1e-30)).detach(),
            'gy_rel': ((hp(gy_hat) - gy_t).norm() / gy_t.norm().clamp_min(1e-30)).detach(),
            'mu_mean': (eta1 * D).mean().detach()}
    if proposals is not None:
        diag.update(ce_tokens=ce.detach(), gy_t=gy_t, gz_t=gz_t)
    return ce_mean + lam * cons, ce_mean, cons, diag


def static_loss(p, tokens, targets):
    u, x = trunk(p, tokens)
    logits = static_logits(p, u, x)
    return F.cross_entropy(logits.float().reshape(-1, V), targets.reshape(-1))
