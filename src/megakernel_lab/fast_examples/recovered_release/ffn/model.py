"""Capture adapter: the frozen FGM-A models expressed through the integrated compiler's existing operations.

``reference.py`` is FGM-A ``model.py`` at 1a46b26af (SHA256 4b623837...), unchanged. ``CaptureLM`` subclasses it, so
parameter names, shapes and the seeded initialization are the original ones; only ``forward`` differs. The forward
spells the original autocast arithmetic with explicit casts (the compiler's precision policies are not CUDA autocast):

* FP32 master parameters, embeddings, residual stream, LayerNorm, router logits/softmax, selected probabilities and
  the expert merge; BF16 operands for every projection, expert product and the vocabulary head (BF16 results, FP32
  accumulation), then ``.float()`` where the original promotes.
* Attention (``ATTENTION_FORM='boundary'``, default) is the FP32 score/softmax/value composition on BF16-exact
  q/k/v with an additive causal mask, the spelling the compiler's flat long BF16 attention body admits (strict:
  FP32-equivalent products; declared practical precision: one-piece BF16 P/dS, FlashAttention-like).
  ``'operands'`` calls the compiler's own ``bf16_operand_attention`` composition (the ``bf16_attention_operands``
  recipe: BF16-exact q/k/v, FP32 scores and softmax, BF16-rounded probabilities for the value product and
  BF16-rounded score cotangents). This is the compiler's declared BF16 attention recipe, not FlashAttention's exact
  rounding sequence. It is called directly because the precision policy would also claim the expert ``bmm``.
* Expert-choice selection is the original top-k over the 32 simultaneous sequences at each position, for every
  expert. There is no top-k operation in the compiler, so the selection is written with exact comparisons:
  rank_i = #{j: p_j > p_i} + #{j < i: p_j == p_i}; slot k holds the i with rank_i == k. This is ``torch.topk``
  (sorted, descending) with ties broken by ascending sequence index — the stable tie policy declared by FGM-LATCH;
  torch.topk leaves tie order unspecified. Selected probabilities are the original softmax values, gathered by an
  exact one-hot sum (one nonzero term), never renormalized; their gradient is the same one-hot scatter as gather's.
* Dispatch is ``F.embedding`` over the flattened group (backward: the step-4 indexed accumulation engine). The
  weighted merge, originally ``index_add``, is ``aten.embedding_dense_backward`` of the weighted rows: the same
  duplicate-row FP32 accumulation (its own derivative is the gather). No new operation or backend is added.
* The head computes the full mean cross entropy in one expression; FGM-A's 1024-token chunks only bounded memory.
  Chunk-sum vs single-sum is an FP32 summation-order difference.

The router backward product is FP32 here. FGM-A executed backward inside autocast, which also rounds the router
gradient products to BF16; this adapter is the more precise of the two and the difference is reported, not hidden.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from megakernel_lab.operation_compiler.precision import bf16_operand_attention

from megakernel_lab.fast_examples.recovered_release.ffn import reference
from megakernel_lab.fast_examples.recovered_release.ffn.reference import ModelConfig

BF16 = torch.bfloat16
# Attention spelling (both are existing compiler recipes; see module docstring):
#   'boundary' (default): FP32 composition on BF16-exact q/k/v -> the flat long BF16 attention body
#   'operands': precision.bf16_operand_attention (BF16 score/value operand roundings)
ATTENTION_FORM = os.environ.get("FGM_ATTENTION_FORM", "boundary")


def bf16_linear(x, weight, bias=None):
    """F.linear under BF16 autocast: BF16 operands (and bias), BF16 result, then promoted to FP32."""
    return F.linear(x.to(BF16), weight.to(BF16), None if bias is None else bias.to(BF16)).float()


def stable_topk_slots(probabilities, k):
    """Exact expert-choice selection over dim 1 of [E, R, P] probabilities.

    Returns (weights [E, k, P] differentiable, index [E, k, P] int64). The sequence index of slot k is the argmax of
    an exact one-hot (argmax keeps the first maximum), so no float-to-integer conversion is needed.
    """
    e, r, p = probabilities.shape
    device = probabilities.device
    scores = probabilities.permute(0, 2, 1)                 # [E, P, R]
    a = scores.unsqueeze(-1)                                 # [E, P, R(i), 1]
    b = scores.unsqueeze(-2)                                 # [E, P, 1, R(j)]
    earlier = (torch.arange(r, device=device)[None, :] < torch.arange(r, device=device)[:, None]).to(torch.float32)
    greater = (b > a).to(torch.float32).sum(dim=-1)
    ties = ((b == a).to(torch.float32) * earlier).sum(dim=-1)
    rank = greater + ties                                    # [E, P, R], exact integers in FP32
    slots = torch.arange(k, device=device, dtype=torch.float32)
    onehot = (rank.unsqueeze(-2) == slots[:, None]).to(torch.float32)              # [E, P, k, R]
    weights = (onehot * scores.unsqueeze(-2)).sum(dim=-1)                           # [E, P, k]
    index = torch.argmax(onehot, dim=-1)                                            # [E, P, k] int64
    return weights.permute(0, 2, 1), index.permute(0, 2, 1)


def expert_choice(ff, x, routes=None):
    cfg = ff.config
    b, t, d = x.shape
    grouped = x.reshape(cfg.routing_group, -1, d)                     # original upstream grouping
    r, p = grouped.shape[0], grouped.shape[1]
    logits = torch.matmul(grouped, ff.router)                         # FP32 router
    probabilities = logits.softmax(dim=-1).permute(2, 0, 1)           # [E, R, P]
    weights, index = stable_topk_slots(probabilities, ff.capacity)    # [E, K, P]
    if FORCED_ROUTES is not None:   # diagnostic references only (never captured): reuse given selections
        index = FORCED_ROUTES[FORCED_LAYER[0]]
        FORCED_LAYER[0] += 1
        weights = probabilities.gather(1, index)
    if routes is not None:
        routes.append(index.to(torch.float32))
    flat = (index * p + torch.arange(p, device=x.device)).reshape(-1)  # row = r * P + position (int64)
    rows = grouped.reshape(-1, d)
    extracted = F.embedding(flat, rows).reshape(ff.n_experts, -1, d)  # dispatch
    hidden = F.relu(torch.bmm(extracted.to(BF16), ff.w1.to(BF16)))
    expert_output = torch.bmm(hidden, ff.w2.to(BF16))
    weighted = expert_output.float() * weights.reshape(ff.n_experts, -1, 1)
    merged = torch.ops.aten.embedding_dense_backward(weighted.reshape(-1, d), flat, r * p, -1, False)
    return F.layer_norm(merged.reshape(b, t, d), (d,), ff.output_norm.weight, ff.output_norm.bias,
                        ff.output_norm.eps)


# LATCH-FFN-RETAIN reserved composition (DECLARED.json res-parallel-gelu): DENSE_COMPOSITION=parallel_gelu makes every
# block x + attn(attn_norm(x)) + down(gelu_tanh(up(ff_norm(x)))) with the same parameters, in this adapter and (via
# apply_reference_composition) in the frozen reference used by conventional controls and oracles.
COMPOSITION = os.environ.get("DENSE_COMPOSITION", "ordinary")
if COMPOSITION not in ("ordinary", "parallel_gelu"):
    raise ValueError("unknown DENSE_COMPOSITION " + COMPOSITION)


def _activation(x):
    return F.gelu(x, approximate="tanh") if COMPOSITION == "parallel_gelu" else F.relu(x)


def apply_reference_composition():
    if COMPOSITION != "parallel_gelu" or getattr(reference.Block, "_dense_composition", None) == COMPOSITION:
        return

    def block_forward(self, x):
        return x + self.attn(self.attn_norm(x)) + self.ff(self.ff_norm(x))

    def ffn_forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))
    reference.Block.forward = block_forward
    reference.DenseFFN.forward = ffn_forward
    reference.Block._dense_composition = COMPOSITION


apply_reference_composition()


def dense_ffn(ff, x):
    b, t, d = x.shape
    flat = x.reshape(b * t, d)
    up = _activation(bf16_linear(flat, ff.up.weight, ff.up.bias))
    return bf16_linear(up, ff.down.weight, ff.down.bias).reshape(b, t, d)


def attention(attn, x, mask=None):
    cfg = attn.config
    b, t, d = x.shape
    h = cfg.n_heads
    hd = d // h
    if ATTENTION_FORM == "operands":
        projected = bf16_linear(x.reshape(b * t, d), attn.qkv.weight).reshape(b, t, h, 3 * hd).transpose(1, 2)
        # Original per-head packing; slices (not chunk, whose functionalized as_strided needs a dense source).
        q, k, v = projected[..., :hd], projected[..., hd:2 * hd], projected[..., 2 * hd:]
        y = bf16_operand_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b * t, d)
    else:
        # 'boundary': FP32 composition on BF16-exact q/k/v with an additive causal mask, spelled exactly like the
        # compiler's qualified transformer example (examples/transformer.py block): one BF16 projection per
        # operand, widened once, then head layout. The original packed per-head QKV weight is read through its
        # per-head [q|k|v] row slices, so every q/k/v element is the same dot product as the packed projection.
        # exp(-10000 - max) is exactly 0 in FP32: masked probabilities equal is_causal's zeros.
        # One packed BF16 projection, as the original: its weight rows are reordered from per-head [q|k|v] packing
        # to [q-block | k-block | v-block], so each output element is the same dot product, the input gradient is
        # one BF16 product (a split per-operand projection would add BF16 roundings to it), and q/k/v are 2-D
        # column slices of the BF16 result, each widened once (the long BF16 attention body's operand contract).
        weight = attn.qkv.weight.view(h, 3, hd, d).permute(1, 0, 2, 3).reshape(3 * d, d)
        packed = F.linear(x.reshape(b * t, d).to(BF16), weight.to(BF16))

        def operand(j):
            # "* 1" is an exact BF16 copy: the attention body needs each operand's own materialized BF16 source
            # (offset views of the packed result are refused); its derivative is the identity.
            block = packed[:, j * d:(j + 1) * d] * 1
            return block.float().reshape(b, t, h, hd).permute(0, 2, 1, 3).reshape(b * h, t, hd)
        q, k, v = operand(0), operand(1), operand(2)
        scores = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(hd)) + mask
        y = torch.bmm(torch.softmax(scores, dim=-1), v).reshape(b, h, t, hd).permute(0, 2, 1, 3).reshape(b * t, d)
    return bf16_linear(y, attn.out.weight).reshape(b, t, d)


def layer_norm(module, x):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


class CaptureLM(reference.LanguageModel):
    """Original FGM-A LanguageModel (same names/initialization); forward returns FP32 logits [B*T, V].

    The additive causal mask is a non-persistent buffer (a read-only bound input; not in the state dict).
    """

    def __init__(self, config):
        super().__init__(config)
        t = config.seq_len
        self.register_buffer("causal_mask", torch.triu(torch.full((t, t), -10000.0), diagonal=1), persistent=False)

    def forward(self, tokens, state=None):
        """``state`` (diagnostic builds only): {"routes": [L, E, K, P]} receives every layer's selected indices."""
        cfg = self.config
        b, t = tokens.shape
        routes = [] if state is not None else None
        x = F.embedding(tokens, self.token_embed.weight) + F.embedding(
            torch.arange(t, device=tokens.device), self.position_embed.weight)
        for block in self.blocks:
            if COMPOSITION == "parallel_gelu":
                x = (x + attention(block.attn, layer_norm(block.attn_norm, x), self.causal_mask)
                     + dense_ffn(block.ff, layer_norm(block.ff_norm, x)))
                continue
            x = x + attention(block.attn, layer_norm(block.attn_norm, x), self.causal_mask)
            normalized = layer_norm(block.ff_norm, x)
            if cfg.architecture == "moe":
                x = x + expert_choice(block.ff, normalized, routes)
            else:
                x = x + dense_ffn(block.ff, normalized)
        hidden = layer_norm(self.final_norm, x).reshape(b * t, cfg.d_model)
        logits = bf16_linear(hidden, self.lm_head.weight)
        if state is None:
            return logits
        return logits, {"routes": torch.stack(routes)}

    def route_state(self, device=None):
        cfg = self.config
        return {"routes": torch.zeros((cfg.n_layers, cfg.n_experts, cfg.capacity, cfg.seq_len), device=device)}


def mean_cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets.reshape(-1))


FORCED_ROUTES = None   # [L, E, K, P] int64, diagnostic eager references only
FORCED_LAYER = [0]

PRECISION_POLICY = None   # attention recipe is written directly (bf16_operand_attention)


def make_config(name, **overrides):
    """The fixed comparison: ordinary, 16x640 (G4) and 32x320 (G8), plus explicitly named small fixtures."""
    base = dict(vocab_size=50257, d_model=640, n_layers=7, n_heads=10, seq_len=512, expansion=4,
                routing_group=32, init_scale=0.1, attention_precision="bf16", d_ff=2560)
    table = {"ordinary": dict(architecture="dense", granularity=1),
             "moe16x640": dict(architecture="moe", granularity=4),
             "moe32x320": dict(architecture="moe", granularity=8)}
    if name in table:
        values = {**base, **table[name]}
    elif name.startswith("tiny-"):
        kind = name[len("tiny-"):]
        values = dict(vocab_size=97, d_model=32, n_layers=1, n_heads=2, seq_len=8, expansion=4, routing_group=8,
                      init_scale=0.1, attention_precision="bf16", d_ff=128,
                      **{"ordinary": dict(architecture="dense", granularity=1),
                         "g4": dict(architecture="moe", granularity=4),
                         "g8": dict(architecture="moe", granularity=8)}[kind])
    else:
        raise ValueError("unknown configuration " + name)
    values.update(overrides)
    return ModelConfig(**values)


def batch_shape(config):
    return config.routing_group, config.seq_len


def adamw_config(lr=3e-4, *, runtime_lr=True):
    """FGM-A AdamW: betas (0.9, 0.999), eps 1e-8, decoupled decay 0.01 on every tensor, global clip 1.0."""
    from megakernel_lab.operation_compiler import AdamWConfig
    return AdamWConfig(lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, max_grad_norm=1.0,
                       runtime_lr=runtime_lr)


def learning_rate(step, tokens_per_update, *, peak=3e-4, minimum=3e-5, warmup=1_048_576, horizon=200_000_000):
    """FGM-A train.learning_rate (absolute token clock) for update index ``step``."""
    tokens = (step + 1) * tokens_per_update
    if tokens <= warmup:
        return peak * tokens / warmup
    progress = min(1.0, (tokens - warmup) / (horizon - warmup))
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
