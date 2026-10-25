"""Selective state-space scan as one operation-level primitive (C-OPS2-C).

This is an operation, not a model recognizer. The forward computes, for every
sequence b, channel e and state n (Mamba-1 ``selective_scan_fn`` semantics with
B/C shared over channels, plus an optional per-position document reset):

    dl_t  = softplus(delta_t + delta_bias)          (or delta_t + delta_bias)
    a_t   = 0 if reset_t else exp(dl_t * A[e, n])
    h_t   = a_t * h_{t-1} + dl_t * u_t * B_t[n]     (h_{-1} = 0)
    y_t   = sum_n C_t[n] * h_t[n] + D[e] * u_t
    out_t = y_t * z_t * sigmoid(z_t)

Inputs u, delta, z are [batch, time, channels]; B and C are [batch, time,
states]; A is [channels, states]; D and delta_bias are [channels]; reset is an
optional bool [batch, time]. u/delta/z/B/C share one floating dtype (FP32 or
BF16); A/D/delta_bias are FP32. Arithmetic is FP32 throughout and outputs are
rounded once to the input dtype. ``ckpt`` is the FP32 state entering every
eighth step, [batch, ceil(time/8), states, channels]; it is saved forward
storage consumed by the backward operation, not a differentiable result.

The backward operation returns du, ddelta, dA, dB, dC, dD, dz, ddelta_bias in
the dtypes of the corresponding inputs. PyTorch autograd connects the two
operations through ``register_autograd``; capture therefore differentiates
models that call ``selective_scan`` without unrolling time. Device lowering
lives in ``scan_lower``. The CPU/CUDA eager implementations here are the
executable definition used for capture-time graph replay and small CPU checks;
they loop over time and are not a performance path.

Source attribution: the equations, reset convention, checkpoint interval and
backward derivation follow the accepted S108F/S109D/P1-A scan family
(s99k-gdssm-v1; /private/tmp/megakernel-p1a research/p1/P1-A ssm/s108f/reference.py,
tag p1a-final) restated for an operation with explicit A instead of a_log; the
operand convention follows mamba_ssm 2.3.2.post1 selective_scan_fn.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

FWD_TARGET = 'latch_ops.selective_scan_fwd.default'
BWD_TARGET = 'latch_ops.selective_scan_bwd.default'
TARGETS = (FWD_TARGET, BWD_TARGET)
CHECKPOINT_INTERVAL = 8
FLOATING = (torch.float32, torch.bfloat16)


def checkpoint_shape(batch, time, channels, states):
    return (batch, -(-time // CHECKPOINT_INTERVAL), states, channels)


def _validate(u, delta, A, B, C, D, z, delta_bias, reset):
    if u.dim() != 3:
        raise ValueError('selective_scan: u must be [batch, time, channels]')
    Bt, T, E = u.shape
    if A.dim() != 2 or A.shape[0] != E:
        raise ValueError('selective_scan: A must be [channels, states]')
    N = A.shape[1]
    if min(Bt, T, E, N) <= 0:
        raise ValueError('selective_scan: empty extents are not admitted')
    for label, value, shape in (('delta', delta, (Bt, T, E)), ('z', z, (Bt, T, E)),
                                ('B', B, (Bt, T, N)), ('C', C, (Bt, T, N))):
        if tuple(value.shape) != shape:
            raise ValueError(f'selective_scan: {label} must have shape {shape}')
        if value.dtype != u.dtype:
            raise ValueError(f'selective_scan: {label} must share the dtype of u')
    if u.dtype not in FLOATING:
        raise ValueError('selective_scan: u/delta/z/B/C must be FP32 or BF16')
    for label, value, shape in (('A', A, (E, N)), ('D', D, (E,)), ('delta_bias', delta_bias, (E,))):
        if tuple(value.shape) != shape or value.dtype != torch.float32:
            raise ValueError(f'selective_scan: {label} must be FP32 {shape}')
    if reset is not None and (reset.dtype != torch.bool or tuple(reset.shape) != (Bt, T)):
        raise ValueError('selective_scan: reset must be bool [batch, time]')
    devices = {x.device for x in (u, delta, A, B, C, D, z, delta_bias) + ((reset,) if reset is not None else ())}
    if len(devices) != 1:
        raise ValueError('selective_scan: all operands must be on one device')
    return Bt, T, E, N


def _step_terms(delta, delta_bias, softplus):
    pre = delta.float() + delta_bias
    return pre, (F.softplus(pre) if softplus else pre)


def _forward_states(u, delta, A, B, C, delta_bias, reset, softplus):
    """FP32 states after every step, [time][batch, channels, states] (small cases only)."""
    Bt, T, E = u.shape
    pre, dl = _step_terms(delta, delta_bias, softplus)
    uf, Bf, Af = u.float(), B.float(), A.float()
    h = torch.zeros(Bt, E, A.shape[1], dtype=torch.float32, device=u.device)
    states = []
    for t in range(T):
        a = torch.exp(dl[:, t, :, None] * Af)
        if reset is not None:
            a = torch.where(reset[:, t, None, None], torch.zeros_like(a), a)
        h = a * h + (dl[:, t] * uf[:, t])[..., None] * Bf[:, t, None, :]
        states.append(h)
    return pre, dl, states


def _reference_forward(u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus):
    Bt, T, E, N = _validate(u, delta, A, B, C, D, z, delta_bias, reset)
    _, _, states = _forward_states(u, delta, A, B, C, delta_bias, reset, delta_softplus)
    Cf = C.float()
    ys = torch.stack([(states[t] * Cf[:, t, None, :]).sum(-1) for t in range(T)], 1)
    y = ys + D * u.float()
    zf = z.float()
    out = y * (zf * torch.sigmoid(zf))
    ckpt = torch.zeros(checkpoint_shape(Bt, T, E, N), dtype=torch.float32, device=u.device)
    for j in range(1, ckpt.shape[1]):
        ckpt[:, j] = states[j * CHECKPOINT_INTERVAL - 1].transpose(1, 2)
    return out.to(u.dtype), ckpt


def _reference_backward(u, delta, A, B, C, D, z, delta_bias, reset, dout, ckpt, delta_softplus):
    Bt, T, E, N = _validate(u, delta, A, B, C, D, z, delta_bias, reset)
    if tuple(dout.shape) != (Bt, T, E) or dout.dtype != u.dtype:
        raise ValueError('selective_scan_bwd: dout must match u')
    if tuple(ckpt.shape) != checkpoint_shape(Bt, T, E, N) or ckpt.dtype != torch.float32:
        raise ValueError('selective_scan_bwd: checkpoint shape/dtype mismatch')
    pre, dl, states = _forward_states(u, delta, A, B, C, delta_bias, reset, delta_softplus)
    uf, Bf, Cf, zf, Af, dof = u.float(), B.float(), C.float(), z.float(), A.float(), dout.float()
    s = torch.stack([(states[t] * Cf[:, t, None, :]).sum(-1) for t in range(T)], 1) + D * uf
    sg = torch.sigmoid(zf)
    ds = dof * zf * sg
    dz = dof * s * sg * (1.0 + zf * (1.0 - sg))
    dD = (ds * uf).sum((0, 1))
    du = ds * D
    ddl = torch.zeros_like(uf)
    dB = torch.zeros(Bt, T, N, dtype=torch.float32, device=u.device)
    dC = torch.zeros_like(dB)
    dA = torch.zeros(E, N, dtype=torch.float32, device=u.device)
    R = torch.zeros(Bt, E, N, dtype=torch.float32, device=u.device)
    for t in range(T - 1, -1, -1):
        dh = R + Cf[:, t, None, :] * ds[:, t, :, None]
        dC[:, t] = (states[t] * ds[:, t, :, None]).sum(1)
        hprev = states[t - 1] if t else torch.zeros_like(R)
        a = torch.exp(dl[:, t, :, None] * Af)
        if reset is not None:
            a = torch.where(reset[:, t, None, None], torch.zeros_like(a), a)
        ada = dh * a * hprev
        ddl[:, t] = (ada * Af).sum(-1) + (dh * uf[:, t, :, None] * Bf[:, t, None, :]).sum(-1)
        du[:, t] = du[:, t] + (dh * dl[:, t, :, None] * Bf[:, t, None, :]).sum(-1)
        dB[:, t] = (dh * (dl[:, t] * uf[:, t])[..., None]).sum(1)
        dA = dA + (ada * dl[:, t, :, None]).sum(0)
        R = a * dh
    dpre = ddl * torch.where(pre > 20.0, torch.ones_like(pre), torch.sigmoid(pre)) if delta_softplus else ddl
    dt = u.dtype
    return (du.to(dt), dpre.to(dt), dA, dB.to(dt), dC.to(dt), dD, dz.to(dt), dpre.sum((0, 1)))


@torch.library.custom_op('latch_ops::selective_scan_fwd', mutates_args=())
def selective_scan_fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                       D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor, reset: Optional[torch.Tensor],
                       delta_softplus: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out, ckpt). Eager definition; see module docstring."""
    return _reference_forward(u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus)


@selective_scan_fwd.register_fake
def _fake_forward(u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus):
    Bt, T, E, N = _validate(u, delta, A, B, C, D, z, delta_bias, reset)
    return (torch.empty((Bt, T, E), dtype=u.dtype, device=u.device),
            torch.empty(checkpoint_shape(Bt, T, E, N), dtype=torch.float32, device=u.device))


@torch.library.custom_op('latch_ops::selective_scan_bwd', mutates_args=())
def selective_scan_bwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
                       D: torch.Tensor, z: torch.Tensor, delta_bias: torch.Tensor, reset: Optional[torch.Tensor],
                       dout: torch.Tensor, ckpt: torch.Tensor, delta_softplus: bool
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (du, ddelta, dA, dB, dC, dD, dz, ddelta_bias)."""
    return _reference_backward(u, delta, A, B, C, D, z, delta_bias, reset, dout, ckpt, delta_softplus)


@selective_scan_bwd.register_fake
def _fake_backward(u, delta, A, B, C, D, z, delta_bias, reset, dout, ckpt, delta_softplus):
    _validate(u, delta, A, B, C, D, z, delta_bias, reset)
    like = lambda x: torch.empty(x.shape, dtype=x.dtype, device=x.device)
    return like(u), like(delta), like(A), like(B), like(C), like(D), like(z), like(delta_bias)


class _SelectiveScan(torch.autograd.Function):
    """Autograd binding of the two opaque operations.

    torch.library ``register_autograd`` is not admitted under torch.func
    transforms (PyTorch 2.9.1), which the capture uses; an autograd.Function
    with ``setup_context`` is. Its forward and backward call only the opaque
    operations, so the captured graph holds one forward and one backward node.
    """

    @staticmethod
    def forward(u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus):
        return selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus)

    @staticmethod
    def setup_context(ctx, inputs, output):
        u, delta, A, B, C, D, z, delta_bias, reset, delta_softplus = inputs
        ctx.mark_non_differentiable(output[1])
        ctx.delta_softplus = delta_softplus
        ctx.has_reset = reset is not None
        saved = (u, delta, A, B, C, D, z, delta_bias, output[1]) + ((reset,) if reset is not None else ())
        ctx.save_for_backward(*saved)

    @staticmethod
    def backward(ctx, dout, _dckpt):
        saved = ctx.saved_tensors
        u, delta, A, B, C, D, z, delta_bias, ckpt = saved[:9]
        reset = saved[9] if ctx.has_reset else None
        # First-order only: the backward operation itself is not differentiated. Running it with grad
        # disabled keeps the custom op's generated autograd kernel (refused under torch.func) out of the call.
        with torch.no_grad():
            grads = selective_scan_bwd(u, delta, A, B, C, D, z, delta_bias, reset, dout.contiguous(), ckpt,
                                       ctx.delta_softplus)
        return (*grads, None, None)


def selective_scan(u, delta, A, B, C, D, z, delta_bias, *, reset=None, delta_softplus=True):
    """Differentiable gated selective scan; returns ``out`` [batch, time, channels].

    D, z and delta_bias are required in this checkpoint (Mamba-1's gated form).
    ``reset[b, t]`` starts a new document at step t (its state does not see
    step t-1). Lowering selects an implementation from dtype/layout eligibility;
    it never inspects the calling model.
    """
    return _SelectiveScan.apply(u, delta, A, B, C, D, z, delta_bias, reset, bool(delta_softplus))[0]


def reference_float64(u, delta, A, B, C, D, z, delta_bias, reset=None, delta_softplus=True, dout=None):
    """Independent NumPy FP64 forward/backward for primitive checks (no torch autograd).

    Returns (out, grads-or-None) as float64 arrays, grads keyed like the
    backward outputs. Written directly from the equations above.
    """
    import numpy as np
    a = lambda x: np.asarray(x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else x, np.float64)
    u, delta, A, B, C, D, z, bias = map(a, (u, delta, A, B, C, D, z, delta_bias))
    Bt, T, E = u.shape
    N = A.shape[1]
    keep = np.ones((Bt, T)) if reset is None else 1.0 - a(reset)
    pre = delta + bias
    dl = np.where(pre > 20, pre, np.log1p(np.exp(np.minimum(pre, 20)))) if delta_softplus else pre
    alpha = keep[:, :, None, None] * np.exp(dl[..., None] * A)
    h = np.zeros((Bt, T, E, N))
    prev = np.zeros((Bt, E, N))
    for t in range(T):
        prev = alpha[:, t] * prev + (dl[:, t] * u[:, t])[..., None] * B[:, t, None, :]
        h[:, t] = prev
    s = np.einsum('bten,btn->bte', h, C) + D * u
    sg = 1.0 / (1.0 + np.exp(-z))
    out = s * z * sg
    if dout is None:
        return out, None
    do = a(dout)
    ds = do * z * sg
    dz = do * s * sg * (1.0 + z * (1.0 - sg))
    dD = (ds * u).sum((0, 1))
    du = ds * D
    ddl = np.zeros_like(u)
    dB = np.zeros((Bt, T, N))
    dC = np.einsum('bten,bte->btn', h, ds)
    dA = np.zeros((E, N))
    R = np.zeros((Bt, E, N))
    for t in range(T - 1, -1, -1):
        dh = R + C[:, t, None, :] * ds[:, t, :, None]
        hprev = h[:, t - 1] if t else np.zeros_like(R)
        ada = dh * alpha[:, t] * hprev
        ddl[:, t] = (ada * A).sum(-1) + (dh * u[:, t, :, None] * B[:, t, None, :]).sum(-1)
        du[:, t] += (dh * dl[:, t, :, None] * B[:, t, None, :]).sum(-1)
        dB[:, t] = (dh * (dl[:, t] * u[:, t])[..., None]).sum(1)
        dA += (ada * dl[:, t, :, None]).sum(0)
        R = alpha[:, t] * dh
    dpre = ddl * np.where(pre > 20, 1.0, 1.0 / (1.0 + np.exp(-pre))) if delta_softplus else ddl
    return out, dict(du=du, ddelta=dpre, dA=dA, dB=dB, dC=dC, dD=dD, dz=dz, ddelta_bias=dpre.sum((0, 1)))


GRAD_NAMES = ('du', 'ddelta', 'dA', 'dB', 'dC', 'dD', 'dz', 'ddelta_bias')
