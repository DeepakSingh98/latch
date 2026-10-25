"""FGM-A text transformer and adapted llm-random expert-choice FFN.

Modified adaptation (2026-10-06), Apache-2.0 upstream attribution in UPSTREAM.md.
Upstream commit: 810e6451a402f9a37c2f6f43c1bb75bc00f37140.
The routing and initialization below preserve the original expert-choice recipe;
this is a task adaptation, not a reproduction of the paper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 640
    n_layers: int = 7
    n_heads: int = 10
    seq_len: int = 512
    expansion: int = 4
    granularity: int = 1
    architecture: str = "moe"
    routing_group: int = 32
    seed: int = 0
    init_scale: float = 0.1
    attention_precision: str = "bf16"
    d_ff: int | None = None

    def __post_init__(self):
        if self.d_ff is None:
            object.__setattr__(self, "d_ff", 4 * self.d_model)
        for key in ("vocab_size", "d_model", "n_layers", "n_heads", "seq_len",
                    "expansion", "granularity", "routing_group", "d_ff"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.architecture not in ("dense", "moe"):
            raise ValueError("architecture must be dense or moe")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.attention_precision not in ("bf16", "fp32"):
            raise ValueError("attention_precision must be bf16 or fp32")
        if not math.isfinite(self.init_scale) or self.init_scale <= 0:
            raise ValueError("init_scale must be finite and positive")
        if self.architecture == "moe":
            if self.d_ff % self.granularity:
                raise ValueError("d_ff must be divisible by granularity: no width rounding")
            if self.routing_group % self.expansion:
                raise ValueError("routing_group must be divisible by expansion: no capacity rounding")

    @property
    def n_experts(self):
        return self.expansion * self.granularity

    @property
    def expert_width(self):
        return self.d_ff // self.granularity

    @property
    def capacity(self):
        return self.routing_group // self.expansion

    def to_dict(self):
        return asdict(self)


def _init_weight(parameter, fan_in, cfg, stream):
    # This intentionally matches upstream truncated_normal, including its
    # absolute bounds +/-2*scale (NOT +/-2*std or truncated_normal_fixed).
    digest = hashlib.sha256(f"fgm-a:{cfg.seed}:{stream}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    nn.init.trunc_normal_(parameter, std=math.sqrt(cfg.init_scale / fan_in),
                          a=-2 * cfg.init_scale, b=2 * cfg.init_scale,
                          generator=generator)


class Linear(nn.Module):
    """Linear without an initial, discarded random initialization."""
    def __init__(self, din, dout, cfg, stream, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dout, din))
        self.bias = nn.Parameter(torch.zeros(dout)) if bias else None
        _init_weight(self.weight, din, cfg, stream + ".weight")

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class DenseFFN(nn.Module):
    def __init__(self, cfg, stream="ff"):
        super().__init__()
        # llm-random FeedForward's ordinary control has both biases and no
        # additional output norm. MoE experts have neither bias.
        self.up = Linear(cfg.d_model, cfg.d_ff, cfg, stream + ".up", bias=True)
        self.down = Linear(cfg.d_ff, cfg.d_model, cfg, stream + ".down", bias=True)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class ExpertChoice(nn.Module):
    """Each expert selects K sequences independently at each token position.

    For B=R*M, match upstream reshape grouping: group m contains original
    sequence indices m, M+m, ..., (R-1)*M+m. No temporal positions are mixed.
    Accumulation steps and DDP ranks are NOT pooled into a routing group.
    """
    def __init__(self, cfg: ModelConfig, stream="ff"):
        super().__init__()
        if cfg.architecture != "moe":
            raise ValueError("ExpertChoice requires architecture='moe'")
        self.config = cfg
        self.n_experts = cfg.n_experts
        self.expert_width = cfg.expert_width
        self.capacity = cfg.capacity
        self.router = nn.Parameter(torch.empty(cfg.d_model, cfg.n_experts))
        self.w1 = nn.Parameter(torch.empty(cfg.n_experts, cfg.d_model, cfg.expert_width))
        self.w2 = nn.Parameter(torch.empty(cfg.n_experts, cfg.expert_width, cfg.d_model))
        _init_weight(self.router, cfg.d_model, cfg, stream + ".router")
        _init_weight(self.w1, cfg.d_model, cfg, stream + ".w1")
        # model_utils.get_expert_choice_args defaults use_topk_initialization=True:
        # fan_in = average_selected_experts * expert_width = G*H = d_ff.
        _init_weight(self.w2, cfg.granularity * cfg.expert_width, cfg, stream + ".w2")
        self.output_norm = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.capture_routing = False
        self._routing_snapshot = None

    def _group(self, x):
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError("expert-choice expects [batch,position,d_model]")
        if x.shape[0] % self.config.routing_group:
            raise ValueError("simultaneous batch must be divisible by routing_group; accumulation cannot substitute")
        return x.reshape(self.config.routing_group, -1, self.config.d_model)

    def route(self, x):
        grouped = self._group(x)
        # FP64 is supported for independent derivative audits. Production
        # routing uses FP32 master weights/logits/softmax under BF16 autocast.
        dtype = torch.float64 if self.router.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = torch.matmul(grouped.to(dtype), self.router.to(dtype))
            probabilities = logits.softmax(dim=-1).permute(2, 0, 1)
            weights, indices = probabilities.topk(self.capacity, dim=1, sorted=True)
        return weights, indices

    def forward(self, x):
        original_shape = x.shape
        grouped = self._group(x)
        positions = grouped.shape[1]
        weights, indices = self.route(x)
        flat_indices = indices * positions + torch.arange(positions, device=x.device)
        extracted = grouped.reshape(-1, self.config.d_model).index_select(
            0, flat_indices.reshape(-1)).reshape(self.n_experts, -1, self.config.d_model)
        expert_output = torch.bmm(F.relu(torch.bmm(extracted, self.w1)), self.w2)
        # Original selected probabilities weight outputs; never renormalize
        # them over selected experts. FP32 reduction avoids BF16 atomics.
        weighted = expert_output.to(weights.dtype) * weights.reshape(self.n_experts, -1, 1)
        merged = torch.zeros((grouped.shape[0] * positions, self.config.d_model),
                             dtype=weighted.dtype, device=x.device)
        merged = merged.index_add(0, flat_indices.reshape(-1), weighted.reshape(-1, self.config.d_model))
        if self.capture_routing:
            with torch.no_grad():
                counts = torch.bincount(flat_indices.reshape(-1), minlength=grouped.shape[0] * positions)
                self._routing_snapshot = {
                    "selection_histogram": torch.bincount(counts, minlength=self.n_experts + 1),
                    "expert_selections": torch.full((self.n_experts,), self.capacity * positions,
                                                    dtype=torch.int64, device=x.device),
                    "selected_weight_mean": weights.detach().mean(),
                    "selected_weight_min": weights.detach().min(),
                    "selected_weight_max": weights.detach().max(),
                }
        # Includes the affine bias for tokens selected by no expert, as upstream.
        return self.output_norm(merged.reshape(original_shape))

    def routing_stats(self):
        if self._routing_snapshot is None:
            return {}
        out = {key: value.detach().cpu().tolist() for key, value in self._routing_snapshot.items()}
        hist = out["selection_histogram"]
        n_tokens = sum(hist)
        out.update(n_experts=self.n_experts, expert_width=self.expert_width,
                   routing_group=self.config.routing_group, capacity=self.capacity,
                   tokens=n_tokens, zero_selection_fraction=hist[0] / n_tokens,
                   mean_selections=sum(i * n for i, n in enumerate(hist)) / n_tokens)
        for name in ("router", "w1", "w2"):
            parameter = getattr(self, name)
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                out[name + "_grad_norm"] = grad.norm().item()
                if name != "router":
                    out[name + "_expert_grad_norms"] = grad.reshape(self.n_experts, -1).norm(dim=1).cpu().tolist()
        return out


class Attention(nn.Module):
    def __init__(self, cfg, stream):
        super().__init__()
        self.config = cfg
        self.qkv = Linear(cfg.d_model, 3 * cfg.d_model, cfg, stream + ".qkv")
        self.out = Linear(cfg.d_model, cfg.d_model, cfg, stream + ".out")

    def forward(self, x):
        b, t, d = x.shape
        # Original per-head QKV packing; no rotary positions or gated attention.
        projected = self.qkv(x).reshape(b, t, self.config.n_heads, 3 * (d // self.config.n_heads))
        q, k, v = projected.transpose(1, 2).chunk(3, dim=-1)
        if self.config.attention_precision == "fp32":
            dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
            with torch.autocast(device_type=x.device.type, enabled=False):
                y = F.scaled_dot_product_attention(q.to(dtype), k.to(dtype), v.to(dtype), is_causal=True)
        else:
            # The name denotes permission to use BF16 under the trainer's
            # autocast. CPU and FP64 audits retain their incoming dtype.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        stream = f"blocks.{index}"
        self.attn_norm = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg, stream + ".attn")
        self.ff_norm = nn.LayerNorm(cfg.d_model)
        self.ff = (DenseFFN if cfg.architecture == "dense" else ExpertChoice)(cfg, stream + ".ff")

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ff(self.ff_norm(x))


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model,
                                       _weight=torch.empty(config.vocab_size, config.d_model))
        self.position_embed = nn.Embedding(config.seq_len, config.d_model,
                                          _weight=torch.empty(config.seq_len, config.d_model))
        _init_weight(self.token_embed.weight, 1, config, "token_embed.weight")
        _init_weight(self.position_embed.weight, 1, config, "position_embed.weight")
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layers)])
        # OMNI-L1 backbone convention: untied biasless output and final norm.
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, config, "lm_head")

    def hidden(self, tokens):
        if tokens.ndim != 2 or tokens.shape[1] > self.config.seq_len:
            raise ValueError("tokens must be [batch,position] within configured seq_len")
        x = self.token_embed(tokens) + self.position_embed(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, tokens):
        return self.lm_head(self.hidden(tokens))

    def set_routing_capture(self, enabled=True):
        for block in self.blocks:
            if isinstance(block.ff, ExpertChoice):
                block.ff.capture_routing = enabled
                if not enabled:
                    block.ff._routing_snapshot = None

    def routing_stats(self):
        return [dict(layer=i, **block.ff.routing_stats()) for i, block in enumerate(self.blocks)
                if isinstance(block.ff, ExpertChoice)]

    def parameter_accounting(self, seq_len=None):
        return parameter_accounting(self.config, seq_len=seq_len, actual_total=sum(p.numel() for p in self.parameters()))


def parameter_accounting(cfg: ModelConfig, seq_len=None, actual_total=None):
    """Exact parameter counts; explicitly scoped mathematical FLOP estimates.

    Matmul multiply+add=2; backward+forward projection FLOPs=3*forward.
    Excludes nonlinearities, normalization, softmax/top-k comparisons, optimizer,
    memory traffic and communication. Their elapsed cost remains in timing.
    Attention has both dense-square and ideal causal-triangle counts.
    """
    d, f, l, v = cfg.d_model, cfg.d_ff, cfg.n_layers, cfg.vocab_size
    t = cfg.seq_len if seq_len is None else seq_len
    moe = cfg.architecture == "moe"
    experts = 2 * d * f * (cfg.expansion if moe else 1) * l
    router = d * cfg.n_experts * l if moe else 0
    ff_bias = (d + f) * l if not moe else 0
    norms = 4 * d * l + 2 * d + (2 * d * l if moe else 0)
    attention = 4 * d * d * l
    embedding = (v + cfg.seq_len) * d
    head = v * d
    total = experts + router + ff_bias + norms + attention + embedding + head
    if actual_total is not None and actual_total != total:
        raise AssertionError(f"analytical count {total} != actual parameter count {actual_total}")
    active_expert = 2 * d * f * l
    active_excluding_embedding_lookup = active_expert + router + ff_bias + norms + attention + head
    forward_expert = 2 * active_expert
    forward_router = 2 * router
    forward_attention_projection = 2 * attention
    forward_head = 2 * head
    attention_square = 4 * t * d * l
    attention_causal = 2 * (t + 1) * d * l
    dense_matmul = forward_expert + forward_router + forward_attention_projection + forward_head
    result = {
        "total_parameters": total, "actual_total_parameters": actual_total,
        "embedding_parameters": embedding, "output_head_parameters": head,
        "attention_parameters": attention, "expert_or_dense_weight_parameters": experts,
        "router_parameters": router, "normalization_parameters": norms,
        "dense_ffn_bias_parameters": ff_bias,
        "average_active_expert_weight_parameters_per_token": active_expert,
        "average_active_parameters_excluding_embedding_lookup": active_excluding_embedding_lookup,
        "average_active_parameters_including_two_embedding_rows": active_excluding_embedding_lookup + 2 * d,
        "active_definition": "mean over simultaneous routing groups; per-token expert counts vary; head and all router weights included",
        "n_experts_per_layer": cfg.n_experts if moe else 0,
        "expert_width": cfg.expert_width if moe else f,
        "average_experts_per_token_per_layer": cfg.granularity if moe else 0,
        "capacity_per_expert_per_position": cfg.capacity if moe else 0,
        "routing_group": cfg.routing_group if moe else None,
        "forward_expert_matmul_flops_per_token": forward_expert,
        "forward_router_matmul_flops_per_token": forward_router,
        "forward_attention_projection_flops_per_token": forward_attention_projection,
        "forward_head_flops_per_token": forward_head,
        "forward_attention_matmul_flops_per_token_square": attention_square,
        "forward_attention_matmul_flops_per_token_causal_ideal": attention_causal,
        "training_matmul_flops_per_token_square": 3 * (dense_matmul + attention_square),
        "training_matmul_flops_per_token_causal_ideal": 3 * (dense_matmul + attention_causal),
        "router_logits_per_token": cfg.n_experts * l if moe else 0,
        "router_softmax_exponentials_per_token": cfg.n_experts * l if moe else 0,
        "router_topk_groups_per_update_at_batch_routing_group": cfg.n_experts * t * l if moe else 0,
        "router_topk_candidates_per_group": cfg.routing_group if moe else 0,
        "router_topk_selected_per_group": cfg.capacity if moe else 0,
        "router_topk_comparison_count": "implementation dependent; no comparison FLOPs invented",
        "expert_dispatch_vectors_per_token": cfg.granularity * l if moe else 0,
        "weighted_merge_scalar_terms_per_token": cfg.granularity * d * l if moe else 0,
        "weighted_merge_multiply_add_ops_per_token": 2 * cfg.granularity * d * l if moe else 0,
        "dispatch_gather_logical_scalar_reads_per_token": cfg.granularity * d * l if moe else 0,
        "dispatch_gather_logical_scalar_writes_per_token": cfg.granularity * d * l if moe else 0,
        "dispatch_count_scope": "forward only; average over routing groups; logical scalar transfers, not measured HBM traffic",
        "topk_update_scope": "one microbatch of routing_group sequences and seq_len tokens, accumulation=1; multiply by actual group/microbatch count",
        "flops_excluded": "softmax/top-k/ReLU/norm/weighting/scatter/embedding/optimizer; report separately through measured wall time",
    }
    return result
