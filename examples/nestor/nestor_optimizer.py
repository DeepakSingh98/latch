"""Upstream Nestor MuonAdamW (nestor/scaling/muon.py @ 9c25256f), global clipping and gradient accumulation as one
Latch tensor-update rule per program mode.

* Parameter groups are the upstream optimizer's own groups (inventory ``optimizer_groups``: method, initial lr,
  weight decay, betas, eps, momentum, nesterov, ns_steps, member names), so the partition and per-group rates are
  read from upstream code, not re-derived.
* Per-update schedule scalars are staged (``sched`` [groups, 4] fp32 = lr, 1 - lr*wd, -lr/bc1, sqrt(1 - beta2^s)),
  computed on the host in double precision exactly as the upstream Python does (SequentialLR warm-up lambda
  k / 1526 times the group's initial lr; per-parameter AdamW step counts are equal because every parameter always
  has a gradient), then rounded to fp32 as the foreach kernels round their Python scalars.
* Modes: ``accumulate`` (acc += g; parameters unchanged), ``final`` (G = acc + g; clip; step; acc = 0),
  ``direct`` (G = g; clip; step).
* Clip: torch.nn.utils.clip_grad_norm_(max_norm=1.0) foreach form -- per-tensor L2 norms, the L2 norm of those,
  coef = min(1, 1 / (norm + 1e-6)), every gradient scaled by coef.
* Muon (nesterov, momentum 0.95): buf = 0.95 buf + G; d = G + 0.95 buf; U = NS5(d) per matrix / per stacked-bank
  slice in bf16 (Frobenius normalisation per slice, wide orientation); p = p * (1 - lr wd); p = p - lr U.
* AdamW: p = p * (1 - lr wd); m = lerp(m, G, 0.1); v = 0.95 v + 0.05 G G; p = p + (-lr/bc1) * m / (sqrt(v)/bc2 + eps).
"""
from __future__ import annotations

import math

import torch

BF16 = torch.bfloat16
SCHED = "sched"
NORM = "diag_grad_norm"
CLIP = "diag_clip_coef"
MODES = ("accumulate", "final", "direct")
NS = (3.4445, -4.7750, 2.0315)


def policy_from_groups(groups):
    """Upstream param groups -> {name: (group index, method)} plus the group table."""
    if isinstance(groups, dict) and "error" in groups:
        raise RuntimeError("inventory has no upstream optimizer groups: " + groups["error"])
    owner = {}
    for g in groups:
        for name in g["params"]:
            if name in owner:
                raise ValueError("parameter in two groups: " + name)
            owner[name] = (g["index"], g["method"])
    return owner


def schedule(groups, update_index, *, warmup=1526, lr_scale=1.0):
    """fp32 [groups, 4] schedule tensor for 0-based update ``update_index`` (lr factor = k / warmup in warm-up).
    ``lr_scale`` != 1 is a measurement-only device, never used for training."""
    if update_index >= warmup:
        raise NotImplementedError("only the warm-up phase is staged (qualification uses <= 512 updates)")
    factor = update_index / max(1, warmup)
    rows = []
    step = update_index + 1
    for g in groups:
        lr = g["lr"] * factor * lr_scale
        wd = g.get("weight_decay") or 0.0
        if g["method"] == "adamw":
            beta1, beta2 = g["betas"]
            bc1 = 1.0 - beta1 ** step
            bc2 = math.sqrt(1.0 - beta2 ** step)
            rows.append([lr, 1.0 - lr * wd, -lr / bc1, bc2])
        else:
            rows.append([lr, 1.0 - lr * wd, 0.0, 1.0])
    return torch.tensor(rows, dtype=torch.float32)


def _bf(x):
    return x.to(BF16)


def _f(x):
    return x.to(torch.float32)


def newton_schulz(d, steps, *, pad, tall):
    """zeropower_via_newtonschulz5[_batched] with every eager bf16 rounding point spelled out.

    ``pad=P`` (> 1) pads each slice's short dimension with zero rows (zero columns for a tall slice) to a multiple
    of P before the iteration and slices them off after. Zero rows and columns stay exactly zero through every
    step and contribute exact zeros to every real entry, so each real entry's arithmetic is unchanged; the padded
    shapes (e.g. 150 -> 152, 90 -> 96, 60 -> 64) admit tensor-core matrix bodies. ``pad`` of 0 or 1 pads nothing.

    ``tall=True`` iterates a tall slice (rows > cols) in its own orientation instead of through a transposed view:
    A = X^T X, B = bA + c(cA)A, X <- aX + X B^T. Every product is the transpose of the wide-orientation product
    (B Y)^T = Y^T B^T with Y = X^T, so each entry is the same sum of the same products (only the tensor-core
    accumulation layout differs) and no strided transposed elementwise passes are needed.
    """
    a, b, c = NS
    x = _bf(d)
    transposed = x.shape[-2] > x.shape[-1]
    tall = transposed and tall
    if transposed and not tall:
        x = x.transpose(-2, -1)
    dims = (-2, -1)
    norm = _bf(torch.sqrt((_f(x) * _f(x)).sum(dims, keepdim=True)))
    denom = _bf(_f(norm) + 1e-7)
    x = _bf(_f(x) / _f(denom))
    short = -1 if tall else -2
    size = x.shape[short]
    extra = (-size) % pad if pad > 1 else 0
    if extra:
        shape = list(x.shape)
        shape[short] = extra
        x = torch.cat([x, torch.zeros(*shape, dtype=BF16, device=x.device)], short)
    for _ in range(steps):
        if tall:
            aa = torch.matmul(x.transpose(-2, -1), x)
            bb = _bf(_f(aa) * b)
            cc = torch.matmul(_bf(_f(aa) * c), aa)
            bb = _bf(_f(bb) + _f(cc))
            x = _bf(_f(_bf(_f(x) * a)) + _f(torch.matmul(x, bb.transpose(-2, -1))))
        else:
            aa = torch.matmul(x, x.transpose(-2, -1))
            bb = _bf(_f(aa) * b)
            cc = torch.matmul(_bf(_f(aa) * c), aa)
            bb = _bf(_f(bb) + _f(cc))
            x = _bf(_f(_bf(_f(x) * a)) + _f(torch.matmul(bb, x)))
    if extra:
        x = x[..., :size] if tall else x[..., :size, :]
    if transposed and not tall:
        x = x.transpose(-2, -1)
    return _f(x)


def make_rule(model, groups, mode, *, max_grad_norm=1.0, ns_pad=8, ns_tall=True):
    """The upstream update of one program mode as a Latch ``tensor_update`` rule.

    ``ns_pad`` and ``ns_tall`` select the Newton-Schulz slice padding and tall orientation (see ``newton_schulz``);
    the defaults are the values the example's programs were built with. ``ns_pad=0, ns_tall=False`` gives the
    plain upstream spelling.
    """
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    if mode not in MODES:
        raise ValueError("mode must be accumulate, final or direct")
    owner = policy_from_groups(groups)
    table = {g["index"]: g for g in groups}
    names = [n for n, _ in model.named_parameters()]
    missing = [n for n in names if n not in owner]
    if missing:
        raise KeyError(f"{len(missing)} parameters without an upstream group, e.g. {missing[:3]}")

    def initialize(parameters, _hp):
        first = next(iter(parameters.values()))
        state = {}
        if mode != "accumulate":
            state[SCHED] = torch.zeros(len(groups), 4, dtype=torch.float32, device=first.device)
            state[NORM] = first.new_zeros(())
            state[CLIP] = first.new_ones(())
        for name, p in parameters.items():
            if mode != "accumulate":
                if owner[name][1] == "muon":
                    state["mom:" + name] = torch.zeros_like(p)
                else:
                    state["m:" + name] = torch.zeros_like(p)
                    state["v:" + name] = torch.zeros_like(p)
            if mode != "direct":
                state["acc:" + name] = torch.zeros_like(p)
        return state

    def update(parameters, gradients, state, _hp):
        if mode == "accumulate":
            return dict(parameters), {"acc:" + n: state["acc:" + n] + g for n, g in gradients.items()}
        totals = {n: (state["acc:" + n] + g if mode == "final" else g) for n, g in gradients.items()}
        norms = [torch.sqrt((t * t).sum()) for t in totals.values()]
        stacked = torch.stack(norms)
        total = torch.sqrt((stacked * stacked).sum())
        coef = torch.clamp(max_grad_norm / (total + 1e-6), max=1.0)
        sched = state[SCHED]
        out, new = {}, {SCHED: sched, NORM: total, CLIP: coef}
        # Muon directions, then ONE batched Newton-Schulz per matrix shape (upstream _orthogonalize_updates buckets
        # the same way; every slice is normalised and iterated independently, so bucketing changes no slice's math).
        directions, buckets = {}, {}
        for name, p in parameters.items():
            gi, method = owner[name]
            if method != "muon":
                continue
            g = totals[name] * coef
            momentum = table[gi].get("momentum", 0.95)
            buf = state["mom:" + name] * momentum + g
            new["mom:" + name] = buf
            directions[name] = g + momentum * buf if table[gi].get("nesterov", True) else buf
            steps = int(table[gi].get("ns_steps", 5))
            buckets.setdefault((tuple(p.shape[-2:]), steps), []).append(name)
        ortho = {}
        for (_, steps), members in buckets.items():
            stacked = torch.cat([directions[n] if directions[n].ndim == 3 else directions[n].unsqueeze(0)
                                 for n in members], 0)
            result = newton_schulz(stacked, steps, pad=ns_pad, tall=ns_tall)
            start = 0
            for n in members:
                count = directions[n].shape[0] if directions[n].ndim == 3 else 1
                piece = result[start:start + count]
                ortho[n] = piece if directions[n].ndim == 3 else piece[0]
                start += count
        for name, p in parameters.items():
            gi, method = owner[name]
            lr, decay, step_size, bc2 = sched[gi, 0], sched[gi, 1], sched[gi, 2], sched[gi, 3]
            wd = table[gi].get("weight_decay") or 0.0
            if method == "muon":
                q = p * decay if wd != 0.0 else p
                out[name] = q + (-lr) * ortho[name]
            else:
                g = totals[name] * coef
                beta1, beta2 = table[gi]["betas"]
                eps = table[gi].get("eps", 1e-8)
                q = p * decay if wd != 0.0 else p
                m = torch.lerp(state["m:" + name], g, 1.0 - beta1)
                v = state["v:" + name] * beta2 + ((1.0 - beta2) * g) * g
                denom = torch.sqrt(v) / bc2 + eps
                out[name] = q + step_size * (m / denom)
                new["m:" + name], new["v:" + name] = m, v
            if mode == "final":
                new["acc:" + name] = torch.zeros_like(p)
        return out, new

    return tensor_update(update, init_state=initialize, name=f"nestor_muon_adamw_{mode}")
