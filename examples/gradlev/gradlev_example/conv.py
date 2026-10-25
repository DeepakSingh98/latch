# Provenance: copied from the Latch repository, commit df184a5f2579
# (branch codex/latch-gradlev), research/latch_gradlev/gl/conv.py (git blob 0d759df67e2de0d8d9cb52c6c38e4242b9f415ed).
# Changes: none to the code (byte-identical below this header; relative imports resolve inside gradlev_example).
"""Conventional executions of the same causal adaptation sequence (policy P1).

  eager     deploy.step_math in a Python loop (functional; new W tensors every step)
  inplace   hand-written eager step: cuBLAS GEMVs, in-place rank-1 writes (W.addr_-style)
  compiled  torch.compile(max-autotune-no-cudagraphs) of the in-place step, Python loop
  graph     the compiled in-place step for all T positions captured in ONE CUDA graph (plus the
            state reset), replayed per block: no per-step host launch work
Every arm includes all prediction, costate (backward), gate-scaled write and scoring work for
every position; the frozen trunk prefill is shared (deploy.prefill) and timed separately.
"""
from __future__ import annotations

import torch

from .deploy import gelu_tanh, gelu_tanh_grad, head_grad, head_logits, layer_norm_backward, layer_norm_parts, step_math
from .model import V


def step_inplace(W1, W2, c, x, u, tgt, eta1, eta2, nll_out):
    """Same arithmetic as step_math with in-place fast-weight writes; nll_out [B] is written."""
    z = torch.baddbmm(c['b1'][None, None, :], x[:, None, :], W1)[:, 0]
    h = gelu_tanh(z)
    y = torch.baddbmm(c['b2'][None, None, :], h[:, None, :], W2)[:, 0]
    n, nh, rstd = layer_norm_parts(u + y, c['lnf_w'], c['lnf_b'])
    logits = head_logits(n, c['E'], 'mixed')
    lsm = torch.log_softmax(logits, -1)
    onehot = (torch.arange(c['E'].shape[0], device=x.device)[None, :] == tgt[:, None]).float()
    nll_out.copy_(-(lsm * onehot).sum(-1))
    gl = torch.exp(lsm) - onehot
    gn = head_grad(gl, c['E'], 'mixed')
    gy = layer_norm_backward(gn, nh, rstd, c['lnf_w'])
    gz = torch.bmm(W2, gy[:, :, None])[:, :, 0] * gelu_tanh_grad(z)
    W1.baddbmm_((eta1[:, None] * x)[:, :, None], gz[:, None, :], alpha=-1.0)
    W2.baddbmm_((eta2[:, None] * h)[:, :, None], gy[:, None, :], alpha=-1.0)


def step_compiled_body(W1, W2, c, x, u, tgt, eta1, eta2, nll_out, head='mixed'):
    nll, W1n, W2n = step_math(W1, W2, c, x, u, tgt, eta1, eta2, head)
    W1.copy_(W1n)
    W2.copy_(W2n)
    nll_out.copy_(nll)


class ConvLoop:
    """Owns static buffers so that every arm (and the CUDA graph) reads/writes fixed storage."""

    def __init__(self, arm, c, W1_0, W2_0, B, T, device='cuda'):
        self.arm, self.c, self.B, self.T = arm, c, B, T
        self.W1_0, self.W2_0 = W1_0, W2_0                      # [d,m], [m,d] fp32 slow init
        self.W1 = W1_0.expand(B, *W1_0.shape).clone()     # never a view of the slow weights
        self.W2 = W2_0.expand(B, *W2_0.shape).clone()
        self.X = torch.zeros(T, B, W1_0.shape[0], device=device)
        self.U = torch.zeros_like(self.X)
        self.TGT = torch.zeros(T, B, dtype=torch.int64, device=device)
        self.E1 = torch.zeros(T, B, device=device)
        self.E2 = torch.zeros(T, B, device=device)
        self.NLL = torch.zeros(T, B, device=device)
        self.graph = None
        if arm in ('compiled', 'graph', 'graph_reduce', 'graph_hybrid', 'graph_gl32', 'graph_gl32red'):
            head = {'graph_reduce': 'reduce', 'graph_hybrid': 'hybrid', 'graph_gl32': 'mixed_gl32',
                    'graph_gl32red': 'mixed_gl32'}.get(arm, 'mixed')
            if arm == 'graph_gl32':
                # strongest P1b control: an FP32 copy of the BF16 head values for cuBLAS SGEMV (+154 MB)
                self.c = c = {**c, 'E32': c['E'].float()}
            fn = torch.compile(step_compiled_body, mode='max-autotune-no-cudagraphs', fullgraph=True, dynamic=False)
            self.fn = lambda *a: fn(*a, head=head)
        elif arm == 'inplace':
            self.fn = step_inplace

    def load(self, u, x, eta1, eta2, targets):
        """Stage one block's prefill rows ([B,T,...] -> [T,B,...])."""
        self.X.copy_(x.transpose(0, 1))
        self.U.copy_(u.transpose(0, 1))
        self.E1.copy_(eta1.transpose(0, 1))
        self.E2.copy_(eta2.transpose(0, 1))
        self.TGT.copy_(targets.transpose(0, 1))

    def reset(self):
        self.W1.copy_(self.W1_0.expand_as(self.W1))
        self.W2.copy_(self.W2_0.expand_as(self.W2))

    def _loop(self, start=0, stop=None):
        stop = self.T if stop is None else stop
        if self.arm == 'eager':
            W1, W2 = self.W1, self.W2
            for t in range(start, stop):
                self.NLL[t], W1, W2 = step_math(W1, W2, self.c, self.X[t], self.U[t], self.TGT[t],
                                                self.E1[t], self.E2[t])
            self.W1.copy_(W1)
            self.W2.copy_(W2)
            return
        for t in range(start, stop):
            self.fn(self.W1, self.W2, self.c, self.X[t], self.U[t], self.TGT[t], self.E1[t], self.E2[t], self.NLL[t])

    def capture(self):
        """CUDA graph of reset + all T steps (graph arm)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):           # compile/autotune and warm every specialization off-graph
                self.reset()
                self._loop()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.reset()
            self._loop()
        torch.cuda.synchronize()

    def capture_noreset(self):
        """Second CUDA graph: all T steps WITHOUT the state reset (carried-state deployment)."""
        if self.graph is None:
            self.capture()
        self.graph_noreset = torch.cuda.CUDAGraph()
        saved = (self.W1.clone(), self.W2.clone())
        with torch.cuda.graph(self.graph_noreset):
            self._loop()
        self.W1.copy_(saved[0])
        self.W2.copy_(saved[1])
        torch.cuda.synchronize()

    def run_block(self, reset=True, start=0, stop=None):
        """Adapted sequence over the staged block; returns the NLL buffer [T,B] (view)."""
        if self.arm.startswith('graph'):
            if self.graph is None:
                self.capture()
            if not (start == 0 and stop in (None, self.T)):
                raise ValueError('graph arm replays complete captured blocks')
            if reset:
                self.graph.replay()
            else:
                if getattr(self, 'graph_noreset', None) is None:
                    self.capture_noreset()
                self.graph_noreset.replay()
            return self.NLL
        if reset:
            self.reset()
        self._loop(start, stop)
        return self.NLL
