"""S106B conventional training/evaluation engine (one H100). Every arm uses the same execution policy:

  policy 'C2b-long' (conventional, labelled): BF16 autocast forward/backward (FP32 master weights, gradients, clip,
  AdamW and Muon momentum; Muon Newton-Schulz in BF16 as upstream), torch.compile (default mode, static shapes) of the
  trunk, the tail pieces and the outer update, and ONE CUDA graph per complete update (forward, backward through every
  write, clip, AdamW + Muon). Nondeterministic kernels allowed. Attention backend per (arm, context) chosen from the
  pilot's measured frontier (SDPA/FlashAttention-2, cuDNN, FlexAttention block-sparse, SDPA masked).
  Optimizer recipe (S101H/S105F ordinary Muon): Muon (momentum 0.95, Nesterov, NS5, shape scale) on every hidden
  matrix incl. the fast initialization and the cache matrices at gain x lr; AdamW (0.9, 0.95, eps 1e-8) on the tied
  embedding and gains; weight decay 0.1 on matrices and embedding (Muon's on the base lr); one global clip at 1.0;
  linear warmup H/16 then cosine to 10% at H. Tokens per update fixed at 16,384 (8K: 2 units; 2K: 8 quarter-units of
  the same 2 documents), every target of a unit is scored once.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import model as MD

TOKENS = 16384
# Every Engine builds closures over the same code objects; dynamo's per-code cache must hold all engines of a job
# (pilot S106B-pilot-v1 exhausted the default 8 and silently fell back to eager flex attention).
for _k, _v in (('recompile_limit', 512), ('cache_size_limit', 512), ('accumulated_recompile_limit', 8192),
               ('accumulated_cache_size_limit', 8192)):
    if hasattr(torch._dynamo.config, _k):
        setattr(torch._dynamo.config, _k, _v)
WD, B1, B2, AEPS, CLIP = 0.1, 0.9, 0.95, 1e-8, 1.0
MU_MOM, NS_STEPS = 0.95, 5


def ns5(G):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    tr = G.size(-2) > G.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS_STEPS):
        A = X @ X.mT
        Bm = b * A + c * A @ A
        X = a * X + Bm @ X
    if tr:
        X = X.mT
    return X


def segments(units, ctx):
    """units uint8 [N, 8193] -> (x, y) int64 [N * 8192 // ctx, ctx]: aligned sequences, targets never cross a unit."""
    N = units.shape[0]
    k = 8192 // ctx
    idx = np.arange(k)[:, None] * ctx + np.arange(ctx + 1)[None, :]          # [k, ctx+1]
    seq = units[:, idx].reshape(N * k, ctx + 1)
    return seq


class Engine:
    """Compiled + captured complete update for one (arm, ctx, backend); reused across runs (re-initialised)."""

    def __init__(self, arch, backend, *, gain=5.0, compile=True, capture=True, precision='bf16', device='cuda',
                 share=None):
        a = self.a = arch
        self.backend, self.gain, self.precision = backend, gain, precision
        self.B = TOKENS // a.ctx
        self.device = device
        self.inv = MD.inventory(a)
        self.P = share.P if share is not None else {
            n: torch.zeros(s, device=device, dtype=torch.float32, requires_grad=True) for n, s, _ in self.inv}
        self.kind = {n: k for n, _, k in self.inv}
        self.muon = [n for n, _, k in self.inv if k == 'matrix']
        self.adam = [n for n, _, k in self.inv if k != 'matrix']
        self.groups = {}
        for n in self.muon:
            self.groups.setdefault(tuple(self.P[n].shape), []).append(n)
        if share is not None:                      # same parameters and optimizer state, different shapes/graph
            self.mom, self.m, self.v = share.mom, share.m, share.v
        else:
            self.mom = {n: torch.zeros_like(self.P[n], requires_grad=False) for n in self.muon}
            self.m = {n: torch.zeros_like(self.P[n], requires_grad=False) for n in self.adam}
            self.v = {n: torch.zeros_like(self.P[n], requires_grad=False) for n in self.adam}
        # per-update scalars: lr, 1 - lr wd, lr / bc1, 1 / sqrt(bc2), -gain lr, eta
        self.sc = torch.zeros(6, device=device)
        self.x = torch.zeros(self.B, a.ctx, dtype=torch.long, device=device)
        self.y = torch.zeros(self.B, a.ctx, dtype=torch.long, device=device)
        self.loss = torch.zeros((), device=device)
        self.norm = torch.zeros((), device=device)
        self.attn = MD.Attn(a, backend, device)
        self.rope = MD.rope_tables(a, device)
        attn, rope = self.attn, self.rope

        def trunk_fn(P, x):
            return MD.trunk(P, a, x, attn, rope)

        def dense_fn(P, u, y):
            return MD.dense_tail(P, a, u, y)

        def cache_fn(P, u, y):
            return MD.cache_tail(P, a, u, y, attn)

        def pre_fn(P, a_, u):
            return MD.memory_pre(P, a, u)
        c = (lambda f: torch.compile(f, dynamic=False)) if compile else (lambda f: f)
        self.trunk_c = c(trunk_fn)
        self.tail_c = c({'dense': dense_fn, 'cache': cache_fn}.get(a.tail, dense_fn))
        self.pre_c = c(pre_fn)
        self.chunk_c = c(MD.memory_chunk)
        self.update_c = c(self._update)
        self.compiled, self.captured = compile, False
        self.graph = None
        self.capture_requested = capture

    # ------------------------------------------------------------------------------------------------ pieces
    def ce(self, x, y, eta):
        u = self.trunk_c(self.P, x)
        if self.a.tail == 'memory':
            if self.precision == 'bf16_tail32':        # BF16 trunk, FP32 adaptive tail (TF32 off, set by the job)
                with torch.autocast('cuda', enabled=False):
                    return MD.memory_tail(self.P, self.a, u.float(), y, eta, self.chunk_c, pre_fn=self.pre_c)
            return MD.memory_tail(self.P, self.a, u, y, eta, self.chunk_c, pre_fn=self.pre_c)
        return self.tail_c(self.P, u, y)

    def _update(self, params, grads, mom, m, v, sc):
        lr, dec, ss, rden, neg = sc[0], sc[1], sc[2], sc[3], sc[4]
        norm = torch.sqrt(sum((g * g).sum() for g in grads.values()))
        coef = torch.clamp(CLIP / (norm + 1e-6), max=1.0)
        new = {}
        for n in self.adam:
            h = grads[n] * coef
            m_ = m[n] + (1 - B1) * (h - m[n])
            v_ = v[n] * B2 + (h * h) * (1 - B2)
            w = params[n] * dec if self.kind[n] == 'emb' else params[n]
            new[n] = (w - (ss * m_) / (torch.sqrt(v_) * rden + AEPS), m_, v_)
        for shape, names in self.groups.items():
            G = torch.stack([grads[n] * coef for n in names])
            Bm = torch.stack([mom[n] for n in names])
            Bm = Bm + (1 - MU_MOM) * (G - Bm)
            U = ns5(G + MU_MOM * (Bm - G)).float() * max(1.0, shape[0] / shape[1]) ** 0.5
            for i, n in enumerate(names):
                new[n] = (params[n] * dec + U[i] * neg, Bm[i], None)
        return new, norm

    def body(self):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision != 'fp32'):
            loss = self.ce(self.x, self.y, self.sc[5]).mean()
        names = list(self.P)
        gs = torch.autograd.grad(loss, [self.P[n] for n in names])
        grads = {n: g.float() for n, g in zip(names, gs)}
        with torch.no_grad():
            new, norm = self.update_c({n: self.P[n].detach() for n in names}, grads, self.mom, self.m, self.v, self.sc)
            for n, (w, a_, b_) in new.items():
                self.P[n].copy_(w)
                if n in self.mom:
                    self.mom[n].copy_(a_)
                else:
                    self.m[n].copy_(a_)
                    self.v[n].copy_(b_)
            self.loss.copy_(loss.detach().float())
            self.norm.copy_(norm)

    # ------------------------------------------------------------------------------------------------ state
    def load(self, seed):
        P0 = MD.init_params(self.a, seed, device=self.device)
        with torch.no_grad():
            for n in self.P:
                self.P[n].copy_(P0[n])
            for d in (self.mom, self.m, self.v):
                for t in d.values():
                    t.zero_()

    def build(self, seed=0, warmup=3):
        """Compile (by running warmup updates on a side stream) and capture one update; then reset to seed init."""
        t0 = time.perf_counter()
        self.load(seed)
        with torch.no_grad():                               # no RNG use (capture-safe), deterministic dummy bytes
            ar = torch.arange(self.x.numel(), device=self.device).view_as(self.x)
            self.x.copy_(ar * 7 % 256)
            self.y.copy_(ar * 13 % 256)
        self.sc.copy_(torch.tensor([1e-4, 1.0, 1e-4, 1.0, -1e-4, 1.0]))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        err = []
        if self.capture_requested:
            for attempt in range(2):
                from torch._dynamo.utils import counters
                before = dict(counters['stats'])
                try:
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        self.body()
                    self.graph, self.captured = g, True
                    break
                except Exception as exc:  # noqa: BLE001 -- recorded; retried once after a fresh warmup
                    err.append(dict(error=repr(exc)[:300], dynamo_stats_before=before,
                                    dynamo_stats_after=dict(counters['stats'])))
                    self.graph, self.captured = None, False
                    torch.cuda.synchronize()
                    for _ in range(2):
                        self.body()
                    torch.cuda.synchronize()
        torch.cuda.synchronize()
        self.load(seed)
        return dict(compile_warmup_s=t1 - t0, capture_s=time.perf_counter() - t1, captured=self.captured,
                    capture_error=err or None)

    def step(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.body()

    # ------------------------------------------------------------------------------------------------ runs
    def schedule(self, H, lr, eta, gain=None):
        gain = self.gain if gain is None else gain
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
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    def train(self, seed, H, lr, eta, units, gain=None):
        """units uint8 [2H, 8193] (the stream's first 2H units). Returns losses, per-update ms, finite flag."""
        self.load(seed)
        need = 2 * H
        if units.shape[0] < need:
            raise ValueError('stream shorter than horizon (no replay allowed)')
        seq = segments(units[:need], self.a.ctx)                       # [2H * k, ctx+1]
        data = torch.from_numpy(np.ascontiguousarray(seq.reshape(H, self.B, self.a.ctx + 1))).to(self.device)
        table = self.schedule(H, lr, eta, gain)
        losses = torch.zeros(H, device=self.device)
        norms = torch.zeros(H, device=self.device)
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(H)]
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        for i in range(H):
            self.x.copy_(data[i, :, :-1])
            self.y.copy_(data[i, :, 1:])
            self.sc.copy_(table[i])
            ev[i][0].record()
            self.step()
            ev[i][1].record()
            losses[i] = self.loss
            norms[i] = self.norm
        torch.cuda.synchronize()
        wall = time.perf_counter() - w0
        ms = [s.elapsed_time(e) for s, e in ev]
        lh = losses.cpu().numpy()
        del data
        return dict(updates=H, lr=lr, eta=eta, seed=seed, gain=self.gain if gain is None else gain, train_ms=ms, train_seconds=sum(ms) / 1000,
                    loop_wall_seconds=wall, median_update_ms=float(np.median(ms)),
                    loss_curve=[float(v) for v in lh[:: max(1, H // 64)]], final_train_loss=float(lh[-32:].mean()),
                    finite=bool(np.isfinite(lh).all()), unique_units=need, targets=H * TOKENS,
                    grad_norm_tail=float(norms[-32:].mean()))

    @torch.no_grad()
    def evaluate(self, units, eta, score=None):
        """Per-position mean CE (nats/byte) over units uint8 [N, 8193]; the arm's own context (2K arms score each
        aligned quarter independently). Evaluation batches = training shapes. Returns summed CE per position [8192]
        and count; score (bool [N, 8192]) optionally restricts extra subset sums."""
        k = 8192 // self.a.ctx
        per = self.B // k                                   # units per call
        N = units.shape[0]
        pos_sum = torch.zeros(8192, dtype=torch.float64, device=self.device)
        sub_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        sub_n = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        etat = torch.tensor(float(eta), device=self.device)
        for i in range(0, N - N % per, per):
            seq = torch.from_numpy(segments(units[i:i + per], self.a.ctx)).to(self.device).long()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.precision != 'fp32'):
                ce = self.ce(seq[:, :-1].contiguous(), seq[:, 1:].contiguous(), etat).double()
            ce = ce.view(per, 8192)
            pos_sum += ce.sum(0)
            if score is not None:
                s = torch.from_numpy(score[i:i + per]).to(self.device)
                sub_sum += (ce * s).sum()
                sub_n += int(score[i:i + per].sum())
        torch.cuda.synchronize()
        n = N - N % per
        return dict(pos_sum=pos_sum.cpu().numpy(), n=n, seconds=time.perf_counter() - t0,
                    sub_sum=float(sub_sum), sub_n=sub_n)


class CurriculumEngine:
    """Full attention with a sequence-length curriculum (PLAN Amendment 1 contingency): the first `frac` of updates
    train on the aligned 2K quarters of the same units (a 2K-shaped graph), the rest on whole 8K units; both graphs
    share one set of parameters and optimizer state, the schedule is one continuous schedule. Evaluation at 8K."""

    def __init__(self, arch, backend8, backend2, *, frac=0.25, gain=5.0, precision='bf16', device='cuda'):
        from dataclasses import replace
        self.e8 = Engine(arch, backend8, gain=gain, precision=precision, device=device)
        self.e2 = Engine(replace(arch, ctx=2048), backend2, gain=gain, precision=precision, device=device,
                         share=self.e8)
        self.a, self.frac, self.gain, self.precision = arch, frac, gain, precision
        self.backend = f'{backend2}->{backend8}'

    @property
    def captured(self):
        return self.e8.captured and self.e2.captured

    def build(self, seed=0):
        r2 = self.e2.build(seed)
        r8 = self.e8.build(seed)
        return dict(compile_warmup_s=r2['compile_warmup_s'] + r8['compile_warmup_s'],
                    capture_s=r2['capture_s'] + r8['capture_s'], captured=self.captured,
                    capture_error=(r2['capture_error'] or r8['capture_error']))

    def train(self, seed, H, lr, eta, units, gain=None):
        e2, e8 = self.e2, self.e8
        e8.load(seed)
        n2 = int(round(H * self.frac))
        d2 = torch.from_numpy(np.ascontiguousarray(segments(units[:2 * H], 2048).reshape(H, e2.B, 2049))).to(e8.device)
        d8 = torch.from_numpy(np.ascontiguousarray(segments(units[:2 * H], 8192).reshape(H, e8.B, 8193))).to(e8.device)
        table = e8.schedule(H, lr, eta, gain)
        losses = torch.zeros(H, device=e8.device)
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(H)]
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        for i in range(H):
            e, d = (e2, d2) if i < n2 else (e8, d8)
            e.x.copy_(d[i, :, :-1])
            e.y.copy_(d[i, :, 1:])
            e.sc.copy_(table[i])
            ev[i][0].record()
            e.step()
            ev[i][1].record()
            losses[i] = e.loss
        torch.cuda.synchronize()
        ms = [s_.elapsed_time(e_) for s_, e_ in ev]
        lh = losses.cpu().numpy()
        return dict(updates=H, lr=lr, eta=eta, seed=seed, gain=self.gain if gain is None else gain, train_ms=ms,
                    train_seconds=sum(ms) / 1000, loop_wall_seconds=time.perf_counter() - w0,
                    median_update_ms=float(np.median(ms)), curriculum_updates_2k=n2,
                    loss_curve=[float(v) for v in lh[:: max(1, H // 64)]], final_train_loss=float(lh[-32:].mean()),
                    finite=bool(np.isfinite(lh).all()), unique_units=2 * H, targets=H * TOKENS, grad_norm_tail=None)

    def evaluate(self, units, eta, score=None):
        return self.e8.evaluate(units, eta, score)
