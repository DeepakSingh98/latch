"""Capture adapter: the frozen LS-MOE-SUP recipe expressed through the compiler's existing operations.

``CaptureLM`` uses the local ``backbone.LanguageModel`` constructor adapted from the pinned recipe. It retains the
supported single-rank MoE arm's parameter names, shapes, tying and seeded initialization; unrelated learning arms
and distributed helpers are omitted. ``SOURCE.json`` records the exact donor and each adaptation. The forward
spells the trainer's CUDA-autocast arithmetic with explicit casts (the compiler's precision policies are not autocast):

* FP32 master parameters, embeddings, residual stream, RMSNorm (``z * rsqrt(mean(z^2) + 1e-6) * g``), router
  logits/softmax (autocast disabled in the recipe), selected probabilities, expert merge and the experts' affine
  output LayerNorm; FP32 SiLU between the expert products (the recipe's ``.to(w1.dtype)``).
* BF16 operands and BF16 results (FP32 accumulation) for the attention projections, both expert ``bmm`` and the
  tied vocabulary readouts, then ``.float()`` where autocast promotes.
* Attention is the recipe's own spelling: one packed per-head QKV projection, ``reshape/transpose/chunk``, BF16
  ``F.scaled_dot_product_attention(is_causal=True)`` (admitted by LATCH-ATTN-RECOG recognition as the FP32
  composition of the widened operands with one BF16 output rounding; the recipe ran PyTorch's flash kernel).
* Expert-choice top-8 of the 32 simultaneous sequences per (expert, position) is the exact comparison rank of
  LATCH-FGM-COMPOSE (``stable_topk_slots``; ties by ascending sequence index); dispatch ``index_select`` is
  ``F.embedding``; the FP32 ``index_add`` merge is ``aten.embedding_dense_backward`` (same duplicate-row FP32
  accumulation). No renormalization of the selected probabilities.
* The forward returns the readout logits (FP32 views of BF16 products) of the final state and, in the auxiliary
  program, of both exits through the SAME final RMSNorm and tied head. ``objective`` forms the recipe's normalized
  loss ``(final + w (e1 + e2) / 2) / (1 + w)`` with the weight ``w`` a runtime FP32 input. The trainer's 1024-token
  chunks and detached head bridges only bounded memory: chunk sums vs one sum is an FP32 summation-order difference.

Two programs share all parameter and optimizer state: ``auxiliary=True`` (updates with w > 0, the first half of the
schedule) and ``auxiliary=False`` (w = 0, exits skipped exactly as the recipe skips them).
"""
from __future__ import annotations


import torch
import torch.nn.functional as F

from .backbone import LanguageModel, ModelConfig
from .auxiliary import supervision_exit_layers
from types import SimpleNamespace

BF16 = torch.bfloat16


def stable_topk_slots(probabilities, k):
    """Exact expert-choice selection over dim 1 of [E, R, P] (LATCH-FGM-COMPOSE ``model.stable_topk_slots``).

    Returns (weights [E, k, P] differentiable, index [E, k, P] int64): ``torch.topk(sorted=True)`` with ties broken
    by ascending sequence index.
    """
    e, r, p = probabilities.shape
    device = probabilities.device
    scores = probabilities.permute(0, 2, 1)                 # [E, P, R]
    a = scores.unsqueeze(-1)
    b = scores.unsqueeze(-2)
    earlier = (torch.arange(r, device=device)[None, :] < torch.arange(r, device=device)[:, None]).to(torch.float32)
    greater = (b > a).to(torch.float32).sum(dim=-1)
    ties = ((b == a).to(torch.float32) * earlier).sum(dim=-1)
    rank = greater + ties
    slots = torch.arange(k, device=device, dtype=torch.float32)
    onehot = (rank.unsqueeze(-2) == slots[:, None]).to(torch.float32)              # [E, P, k, R]
    weights = (onehot * scores.unsqueeze(-2)).sum(dim=-1)                           # [E, P, k]
    index = torch.argmax(onehot, dim=-1)                                            # [E, P, k]
    return weights.permute(0, 2, 1), index.permute(0, 2, 1)


def iterative_topk_slots(probabilities, k):
    """The same selection as ``stable_topk_slots`` by k rounds of first-maximum argmax and masking.

    Probabilities are nonnegative (softmax), so a masked entry is set to -1 and can never be selected again. Round j
    returns the largest remaining value, ties to the lowest sequence index (argmax keeps the first maximum): the
    sorted-descending, ascending-index-tie order of ``stable_topk_slots``. Weights are the selected original
    probabilities (gather; derivative = one-hot scatter). O(k R) work instead of O(R^2)."""
    scores = probabilities.permute(0, 2, 1)                 # [E, P, R]
    r = scores.shape[-1]
    positions = torch.arange(r, device=scores.device)
    masked = scores
    indices, weights = [], []
    for _ in range(k):
        index = torch.argmax(masked, dim=-1)               # [E, P]
        indices.append(index)
        weights.append(torch.gather(scores, -1, index.unsqueeze(-1)).squeeze(-1))
        masked = torch.where(positions == index.unsqueeze(-1), -1.0, masked)
    return torch.stack(weights, -1).permute(0, 2, 1), torch.stack(indices, -1).permute(0, 2, 1)


SELECTION = "argmax"


def rms_norm(module, x):
    """The recipe's RMSNorm, ``x * rsqrt(mean(x^2) + 1e-6) * weight`` in FP32, spelled as PyTorch's F.rms_norm (the
    same expression and operation order; capture maps it to the compiler's analytic-backward RMS normalization)."""
    return F.rms_norm(x, (x.shape[-1],), module.weight, eps=1e-6)


def bf16_linear(x, weight):
    return F.linear(x.to(BF16), weight.to(BF16))


ATTENTION_FORM = "blocks"


def attention(attn, x):
    """Recipe attention. ``blocks`` (default): the packed per-head QKV weight's rows are read in [q | k | v] block
    order (a view/permute of the same parameter), so the one packed BF16 product yields the same dot products and
    ``split`` gives each operand as a column block: LATCH-ATTN-RECOG's admitted ``packed_bf16`` spelling.
    ``heads``: the recipe's literal ``reshape(b, t, h, 3 hd).transpose(1, 2).chunk(3, -1)``."""
    cfg = attn.config
    b, t, d = x.shape
    h = cfg.n_heads
    hd = d // h
    if ATTENTION_FORM == "heads":
        projected = bf16_linear(x, attn.qkv.weight).reshape(b, t, h, 3 * hd)
        q, k, v = projected.transpose(1, 2).chunk(3, dim=-1)
    else:
        weight = attn.qkv.weight.view(h, 3, hd, d).permute(1, 0, 2, 3).reshape(3 * d, d)
        parts = bf16_linear(x.reshape(b * t, d), weight).split(d, dim=-1)
        q, k, v = (part.view(b, t, h, hd).transpose(1, 2) for part in parts)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return bf16_linear(y.transpose(1, 2).reshape(b * t, d), attn.out.weight).float().reshape(b, t, d)


def expert_choice(ff, x, routes=None):
    """The recipe's ``research.learning_scaling.moe.ExpertChoice.forward`` (FP32 SiLU) with routing group = batch."""
    cfg = ff.config
    d = cfg.d_model
    grouped = x.reshape(cfg.routing_group, -1, d)
    r, p = grouped.shape[0], grouped.shape[1]
    logits = torch.matmul(grouped, ff.router)                          # FP32 router (autocast disabled)
    probabilities = logits.softmax(dim=-1).permute(2, 0, 1)            # [E, R, P]
    select = iterative_topk_slots if SELECTION == "argmax" else stable_topk_slots
    weights, index = select(probabilities, ff.capacity)                 # [E, K, P]
    if FORCED_ROUTES is not None:   # eager diagnostic references only (never captured): reuse given selections
        index = FORCED_ROUTES[FORCED_LAYER[0]]
        FORCED_LAYER[0] += 1
        weights = probabilities.gather(1, index)
    if routes is not None:
        routes.append(index)
    flat = (index * p + torch.arange(p, device=x.device)).reshape(-1)
    extracted = F.embedding(flat, grouped.reshape(-1, d)).reshape(ff.n_experts, -1, d)
    projected = torch.bmm(extracted.to(BF16), ff.w1.to(BF16)).float()
    expert_output = torch.bmm(F.silu(projected).to(BF16), ff.w2.to(BF16))
    weighted = expert_output.float() * weights.reshape(ff.n_experts, -1, 1)
    merged = torch.ops.aten.embedding_dense_backward(weighted.reshape(-1, d), flat, r * p, -1, False)
    return F.layer_norm(merged.reshape(x.shape), (d,), ff.output_norm.weight, ff.output_norm.bias,
                        ff.output_norm.eps)


class CaptureLM(LanguageModel):
    """The recipe's LanguageModel; forward(tokens) returns readout logits [N, V] FP32 (final[, exit1, exit2])."""

    def __init__(self, config, *, auxiliary=True):
        # A collective-free single-rank context (whole routing group in one process); construction on the current
        # default device. Native runs bind the reference model's values, so capture-time values are placeholders.
        super().__init__(config, SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"), backend="none"))
        if config.arm != "moe_intermediate_supervision":
            raise ValueError("adapter covers the MoE + intermediate-supervision recipe only")
        self.auxiliary = auxiliary

    def _readout(self, h):
        flat = self.final_norm_value(h).reshape(-1, self.config.d_model)
        return bf16_linear(flat, self.lm_head.weight).float()

    def final_norm_value(self, h):
        return rms_norm(self.final_norm, h)

    def forward(self, tokens, routes=None, state=None):
        """``state`` (route-observation diagnostic builds only): returns (logits, {"routes": [L, E, K, P] FP32})."""
        if state is not None:
            observed = []
            logits = self.forward(tokens, observed)
            return logits, {"routes": torch.stack(observed).to(torch.float32)}
        cfg = self.config
        t = tokens.shape[1]
        x = F.embedding(tokens, self.token_embed.weight) + F.embedding(
            torch.arange(t, device=tokens.device), self.position_embed.weight)
        exits = supervision_exit_layers(cfg.n_layers) if self.auxiliary else ()
        outputs = []
        for i, block in enumerate(self.blocks):
            x = x + attention(block.attn, rms_norm(block.attn_norm, x))
            x = x + expert_choice(block.ff, rms_norm(block.ff_norm, x), routes)
            if i + 1 in exits:
                outputs.append(x if HEAD_FORM == "stacked" else self._readout(x))
        if HEAD_FORM == "stacked" and outputs:
            # One readout over the stacked [final; exit1; exit2] states: the same shared RMSNorm and tied product per
            # row, as one [3N, V] product (one statistics/gradient pass, one dX and one dW product with K = 3N).
            return (self._readout(torch.cat([x, *outputs], dim=0)),)
        return (self._readout(x), *outputs)


# Readout form of the auxiliary program: 'separate' (three readouts) or 'stacked' (one [3N, V] readout; experiment).
HEAD_FORM = "separate"
FORCED_ROUTES = None   # [L, E, K, P] int64, eager diagnostic references only
FORCED_LAYER = [0]


def route_state(config, device=None):
    return {"routes": torch.zeros((config.n_layers, 32, config.routing_group // 4, config.seq_len), device=device)}


def objective(logits, labels, w):
    """Recipe objective; ``w`` is the auxiliary weight, a 0-d FP32 runtime input (staged per update)."""
    labels = labels.reshape(-1)
    n = labels.numel()
    if len(logits) == 1 and logits[0].shape[0] == 3 * n:
        # Stacked form (HEAD_FORM='stacked'): the same normalized objective as per-row weights on one cross entropy:
        # final rows 1/((1+w) N), exit rows w/(2(1+w) N). Equal to the separate form up to FP32 summation order.
        per_row = F.cross_entropy(logits[0], torch.cat([labels, labels, labels]), reduction="none")
        final_w = (1 / (1 + w)) / n
        exit_w = (w / (2 * (1 + w))) / n
        coef = torch.cat([final_w.expand(n), exit_w.expand(2 * n)])
        return (per_row * coef).sum()
    final = F.cross_entropy(logits[0], labels)
    if len(logits) == 1:
        return final
    return final / (1 + w) + (F.cross_entropy(logits[1], labels) + F.cross_entropy(logits[2], labels)) * (w / (2 * (1 + w)))


def make_config(size="25m", **overrides):
    from .recipe import recipe
    return ModelConfig(**recipe(size, **overrides)["model"])


def tiny_config(**overrides):
    """Explicitly named small fixture (development only; never a timed or qualified program)."""
    values = dict(d_model=128, n_layers=3, n_heads=2, seq_len=128, vocab_size=97, routing_group=32,
                  arm="moe_intermediate_supervision", seed=1729)
    values.update(overrides)
    return ModelConfig(**values)
