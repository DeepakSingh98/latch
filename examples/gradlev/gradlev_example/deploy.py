# Provenance: copied from the Latch repository, commit df184a5f2579
# (branch codex/latch-gradlev), research/latch_gradlev/gl/deploy.py (git blob d6f0b6f5b2bf900facf380f8ef4efa9c82e99950).
# Changes: none to the code (byte-identical below this header; relative imports resolve inside gradlev_example).
"""Causal GradLev deployment: score token t, observe its target, write the exact gradient, predict t+1.

The helper Q does not exist here. For every scored position t of a fresh block (reset policy:
fresh fast state at every block = document segment / context window, paper App. E/F):

    z = x_t @ W1 + b1; h = gelu(z); y = h @ W2 + b2; n = LN_f(u_t + y); logits = E n
    nll_t = logsumexp(logits) - logits[s_{t+1}]                      (score BEFORE learning)
    g_y = dl_t/dy (softmax - onehot through the tied head and LN_f)   (observe the target)
    g_z = gelu'(z) * (W2 g_y)                                         (adapted transpose)
    W1 <- W1 - eta1_t x_t g_z^T ; W2 <- W2 - eta2_t h g_y^T          (centered decay alpha = 1)

and position t+1 reads the updated W1, W2. u_t, x_t and the gates eta_t depend only on the
frozen trunk (the adapted map feeds only LN_f/head), so they are computed for all positions by
one causal prefill; nothing in the dependent chain is reordered or detached.

Numerical policy P1 (identical for every backend; see SPEC.md): frozen trunk in BF16 autocast;
adapted MLP, fast state, LN_f, softmax, NLL and costates in FP32 (TF32 off); the tied head E is
stored BF16 and both head products take BF16 operands with FP32 accumulation and BF16 results,
i.e. exactly PyTorch autocast semantics for those two products.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import (D, M, V, L, LN_EPS, gates, gelu_tanh, gelu_tanh_grad, layer_norm_backward, layer_norm_parts,
                    trunk, static_logits)


def consts_from(p, device):
    """Frozen per-model constants for the adapted loop (fp32 except the BF16 head)."""
    return {'b1': p[f'{L-1}.fc_b'].float().to(device), 'b2': p[f'{L-1}.pr_b'].float().to(device),
            'lnf_w': p['lnf_w'].float().to(device), 'lnf_b': p['lnf_b'].float().to(device),
            'E': p['wte'].to(device=device, dtype=torch.bfloat16).contiguous()}


def initial_state(p, device, B=1):
    W1 = p[f'{L-1}.fc_w'].float().to(device)
    W2 = p[f'{L-1}.pr_w'].float().to(device)
    # clone: expand(1, ...).contiguous() would alias the slow weights for B == 1
    return {'W1': W1.expand(B, D, M).clone(), 'W2': W2.expand(B, M, D).clone()}


@torch.no_grad()
def prefill(p, g, tokens):
    """Causal trunk for every position at once: (u, x) fp32 [B,T,d], gates eta1/eta2 fp32 [B,T]."""
    with torch.autocast('cuda' if tokens.is_cuda else 'cpu', dtype=torch.bfloat16):
        u, x = trunk(p, tokens)
    u, x = u.float(), x.float()
    eta1, eta2 = gates(g, x)
    return u, x, eta1, eta2


HEAD_MODES = ('mixed', 'mixed_gl32', 'bf16', 'fp32', 'reduce', 'hybrid')  # hybrid: reduce logits, cuBLAS gradient


@torch.library.custom_op('glev::mm_f32out', mutates_args=())
def _mm_f32out(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b, out_dtype=torch.float32)


@_mm_f32out.register_fake
def _(a, b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


def mm_fp32_out(a, b):
    """BF16 operands, FP32 accumulation and FP32 result (cuBLAS GemmEx semantics).

    CUDA: ATen mm(out_dtype=float32), wrapped as an opaque custom op so torch.compile keeps cuBLAS.
    CPU (and Latch capture, which traces on CPU): exact BF16 values upcast, FP32 product.
    """
    if a.is_cuda:
        return torch.ops.glev.mm_f32out(a, b)
    return a.float() @ b.float()


def head_logits(n, E, mode):
    if mode == 'mixed_gl32':
        mode = 'mixed'
    if mode in ('reduce', 'hybrid'):   # same values as 'mixed' (exact BF16 products, FP32 sums) as an elementwise reduction
        return (n.to(torch.bfloat16).float()[:, None, :] * E.float()[None]).sum(-1)
    if mode == 'mixed':
        return mm_fp32_out(n.to(torch.bfloat16), E.t())
    if mode == 'bf16':
        return (n.to(E.dtype) @ E.t()).to(n.dtype)
    return n @ E.t().to(n.dtype)


def head_grad(gl, E, mode, E32=None):
    if mode == 'mixed_gl32':
        # policy P1b: FP32 softmax-gradient weights times the BF16-stored head (exact BF16 values, FP32 sums)
        return gl @ (E32 if E32 is not None else E.float())
    if mode == 'hybrid':
        mode = 'mixed'
    if mode == 'reduce':
        return (gl.to(torch.bfloat16).float()[:, :, None] * E.float()[None]).sum(1)
    if mode == 'mixed':
        return mm_fp32_out(gl.to(torch.bfloat16), E)
    if mode == 'bf16':
        return (gl.to(E.dtype) @ E).to(gl.dtype)
    return gl @ E.to(gl.dtype)


def step_math(W1, W2, c, x, u, tgt, eta1, eta2, head='mixed'):
    """One causal step for B streams. W1 [B,d,m], W2 [B,m,d], x/u [B,d], tgt int64 [B], eta [B].

    Returns (nll [B], new W1, new W2). Pure function with explicit dtypes: policy P1 when the
    loop dtype is fp32 and c['E'] is BF16; an all-FP64 instance (E in FP64) checks the formulas.
    """
    f = x.dtype
    z = torch.bmm(x[:, None, :], W1)[:, 0] + c['b1']
    h = gelu_tanh(z)
    y = torch.bmm(h[:, None, :], W2)[:, 0] + c['b2']
    n, nh, rstd = layer_norm_parts(u + y, c['lnf_w'], c['lnf_b'])
    logits = head_logits(n, c['E'], head) if f != torch.float64 else n @ c['E'].t()
    lsm = torch.log_softmax(logits, -1)
    onehot = (torch.arange(c['E'].shape[0], device=x.device)[None, :] == tgt[:, None]).to(f)
    nll = -(lsm * onehot).sum(-1)
    gl = torch.exp(lsm) - onehot
    gn = head_grad(gl, c['E'], head, c.get('E32')) if f != torch.float64 else gl @ c['E']
    gy = layer_norm_backward(gn, nh, rstd, c['lnf_w'])
    gz = torch.bmm(W2, gy[:, :, None])[:, :, 0] * gelu_tanh_grad(z)
    W1n = W1 - (eta1[:, None, None] * x[:, :, None]) * gz[:, None, :]
    W2n = W2 - (eta2[:, None, None] * h[:, :, None]) * gy[:, None, :]
    return nll, W1n, W2n


def adapt_eager(state, c, u, x, eta1, eta2, targets, head='mixed'):
    """Reference-order Python loop (B streams, T steps); returns nll [B,T]; mutates state."""
    B, T, _ = x.shape
    out = torch.empty(B, T, device=x.device, dtype=x.dtype)
    W1, W2 = state['W1'], state['W2']
    for t in range(T):
        out[:, t], W1, W2 = step_math(W1, W2, c, x[:, t], u[:, t], targets[:, t], eta1[:, t], eta2[:, t], head)
    state['W1'], state['W2'] = W1, W2
    return out


@torch.no_grad()
def frozen_nll(p, g, tokens, targets, head='mixed'):
    """Frozen inference (no writes), all positions in parallel; same head policy (BF16 operands)."""
    u, x, _, _ = prefill(p, g, tokens)
    c = consts_from(p, tokens.device)
    W1, W2 = p[f'{L-1}.fc_w'].float().to(tokens.device), p[f'{L-1}.pr_w'].float().to(tokens.device)
    y = gelu_tanh(x @ W1 + c['b1']) @ W2 + c['b2']
    n, _, _ = layer_norm_parts(u + y, c['lnf_w'], c['lnf_b'])
    logits = head_logits(n.reshape(-1, D), c['E'], head).view(*n.shape[:-1], V)
    return torch.logsumexp(logits, -1) - logits.gather(-1, targets[..., None])[..., 0]


# ----------------------------------------------------------------------------- FP64 references

def serial_reference(p, g, u, x, eta1, eta2, targets, *, head_bf16_values=True):
    """Independent serial learner in FP64 with torch.autograd for the costates and weight gradients.

    Uses the same frozen values (E optionally rounded to its BF16 storage values) but no BF16
    operand rounding, explicit costate formulas or fused updates. Returns (nll [B,T], W1, W2,
    per-step costates list). One stream at a time.
    """
    dd = torch.float64
    E = p['wte'].to(torch.bfloat16).to(dd) if head_bf16_values else p['wte'].to(dd)
    b1, b2 = p[f'{L-1}.fc_b'].to(dd), p[f'{L-1}.pr_b'].to(dd)
    lw, lb = p['lnf_w'].to(dd), p['lnf_b'].to(dd)
    B, T, _ = x.shape
    nll = torch.empty(B, T, dtype=dd)
    finals, costates = [], []
    for s in range(B):
        W1 = p[f'{L-1}.fc_w'].to(dd).clone()
        W2 = p[f'{L-1}.pr_w'].to(dd).clone()
        cs = []
        for t in range(T):
            W1r, W2r = W1.clone().requires_grad_(True), W2.clone().requires_grad_(True)
            xt, ut = x[s, t].to(dd), u[s, t].to(dd)
            z = xt @ W1r + b1
            z.retain_grad()
            hh = gelu_tanh(z)
            y = hh @ W2r + b2
            y.retain_grad()
            n = F.layer_norm(ut + y, (D,), lw, lb, LN_EPS)
            logits = E @ n
            loss = torch.logsumexp(logits, 0) - logits[targets[s, t]]
            loss.backward()
            nll[s, t] = loss.detach()
            cs.append((y.grad.detach().clone(), z.grad.detach().clone()))
            with torch.no_grad():
                W1 = W1 - eta1[s, t].to(dd) * W1r.grad
                W2 = W2 - eta2[s, t].to(dd) * W2r.grad
        finals.append((W1, W2))
        costates.append(cs)
    return nll, finals, costates


def parallel_reconstruction(p, u, x, eta1, eta2, targets, gy_hat, gz_hat, *, head_bf16_values=True,
                            slow_transpose=False):
    """Paper Sec. 3 stages 2-3 in FP64 given proposed costates (one stream, explicit causal sums).

    Returns (nll [T], gy_target [T,d], gz_target [T,m]). With proposals equal to the serial
    learner's costates, Theorem 1 says outputs/targets equal the serial ones in exact arithmetic.
    slow_transpose=True is the paper's control: the reverse reads W2_0 instead of W2_t.
    """
    dd = torch.float64
    E = p['wte'].to(torch.bfloat16).to(dd) if head_bf16_values else p['wte'].to(dd)
    W1, b1 = p[f'{L-1}.fc_w'].to(dd), p[f'{L-1}.fc_b'].to(dd)
    W2, b2 = p[f'{L-1}.pr_w'].to(dd), p[f'{L-1}.pr_b'].to(dd)
    lw, lb = p['lnf_w'].to(dd), p['lnf_b'].to(dd)
    x, u, e1, e2 = x.to(dd), u.to(dd), eta1.to(dd), eta2.to(dd)
    T = x.shape[0]
    strict = torch.ones(T, T, dtype=torch.bool).tril(-1)
    z = x @ W1 + b1 - ((x @ x.t()) * strict) @ (e1[:, None] * gz_hat)
    h = gelu_tanh(z)
    y = (h @ W2 + b2 - ((h @ h.t()) * strict) @ (e2[:, None] * gy_hat)).requires_grad_(True)
    n = F.layer_norm(u + y, (D,), lw, lb, LN_EPS)
    logits = n @ E.t()
    nll = torch.logsumexp(logits, -1) - logits.gather(-1, targets[:, None])[:, 0]
    gy, = torch.autograd.grad(nll.sum(), y)
    gh = gy @ W2.t()
    if not slow_transpose:
        gh = gh - ((gy @ gy_hat.t()) * strict) @ (e2[:, None] * h)
    gz = gelu_tanh_grad(z) * gh
    return nll.detach(), gy, gz, h.detach()
