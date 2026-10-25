"""Public Mixtral weights expressed with reusable captured training operations.

Parameter names and ownership are identical to the installed checkpoint. The
only execution padding is inside BF16 SDPA; token-choice experts pack every
selected token and retain their actual runtime segment lengths.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from megakernel_lab.operation_compiler import bf16_sdpa
from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, route_order, token_topk


def bf16_linear(x, weight):
    return F.linear(x.to(torch.bfloat16), weight.to(torch.bfloat16))


def rms_norm(module, hidden):
    # Pinned MixtralRMSNorm: residuals and masters remain FP32. Preserve the
    # checkpoint's epsilon and multiply gamma only after normalization.
    x = hidden.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + module.variance_epsilon)
    return module.weight * normalized.to(hidden.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class CapturedTinyMoE(nn.Module):
    """Static recipe phase over the public model; all tensor work remains dynamic."""
    def __init__(self, checkpoint, *, auxiliary=False, precision=None):
        super().__init__()
        from .precision import validate
        self.model = checkpoint.model
        self.lm_head = checkpoint.lm_head
        self.config = checkpoint.config
        self.auxiliary = bool(auxiliary)
        # LATCH-TRAIN-PRECISION: opt-in training precision policy (tinymoe/precision.py); None = released arithmetic.
        self.precision = validate(precision)
        from .precision import components, forward_fp32, forward_input_terms, forward_weight_terms
        self.forward_fp32 = forward_fp32(self.precision)
        self.forward_weight_terms = forward_weight_terms(self.precision)
        self.forward_input_terms = forward_input_terms(self.precision)
        self.expert_split = "experts_split3" in components(self.precision)
        self.num_experts = self.config.num_local_experts
        self.top_k = self.config.num_experts_per_tok
        if self.config.router_jitter_noise != 0 or self.config.attention_dropout != 0:
            raise ValueError("the explicit-state adapter requires zero router jitter and attention dropout")
        if len(self.model.layers) < 8 and self.auxiliary:
            raise ValueError("the supervision recipe requires exits after layers four and eight")
        if self.config.hidden_act != "silu":
            raise ValueError("the public adapter requires the checkpoint's SiLU experts")

    def _rope(self, hidden, tokens):
        # Default checkpoint RoPE, theta=1e6. The original rotary module owns
        # inv_freq; capture binds it as an immutable buffer. Positions reset at
        # each packed text example and caches are disabled, like the baseline.
        positions = torch.arange(tokens.shape[1], dtype=torch.float32, device=tokens.device)
        frequencies = positions[:, None] * self.model.rotary_emb.inv_freq.float()[None, :]
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        scaling = self.model.rotary_emb.attention_scaling
        return (embedding.cos() * scaling).to(hidden.dtype)[None, None, :, :], (
            embedding.sin() * scaling).to(hidden.dtype)[None, None, :, :]

    def _attention(self, module, hidden, cos, sin):
        batch, sequence, width = hidden.shape
        head_width = module.head_dim
        qheads, kvheads = self.config.num_attention_heads, self.config.num_key_value_heads
        q = self._linear(hidden, module.q_proj.weight).reshape(batch, sequence, qheads, head_width).transpose(1, 2)
        k = self._linear(hidden, module.k_proj.weight).reshape(batch, sequence, kvheads, head_width).transpose(1, 2)
        v = self._linear(hidden, module.v_proj.weight, fp32_result=False).reshape(
            batch, sequence, kvheads, head_width).transpose(1, 2)
        # BF16 projected Q/K times FP32 RoPE => FP32, then SDPA autocast would
        # narrow once. Apply RoPE in the real head width before execution padding.
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if sequence >= self.config.sliding_window:
            # Pinned HF4.55.4 materializes the sliding mask at S>=window and
            # repeats KV before SDPA autocast in that branch. K is still FP32,
            # so its shared-head VJP must sum widened contributions in FP32.
            # Shorter unmasked calls use native GQA after autocast (BF16 K).
            q, k, v = bf16_sdpa._expand_grouped_heads(q, k, v, True)
        q, k = q.to(torch.bfloat16), k.to(torch.bfloat16)
        y = bf16_sdpa.sdpa(q, k, v, is_causal=True, enable_gqa=True,
                          scale=head_width**-0.5, sliding_window=self.config.sliding_window)
        y = y.transpose(1, 2).reshape(batch, sequence, width)
        return self._linear(y, module.o_proj.weight).float()

    def _experts(self, module, hidden, fixed_route=None):
        x = hidden.reshape(-1, hidden.shape[-1])
        # LATCH-TRAIN-INTEGRATE: precision attributes default to the released path (bf16_linear, no split), so
        # callers with a minimal context (tests/operations/test_tinymoe_adapter.py) keep the released arithmetic.
        expert_split = getattr(self, "expert_split", False)
        forward_fp32 = getattr(self, "forward_fp32", False)
        weight_terms = getattr(self, "forward_weight_terms", 1)
        linear = self._linear if weight_terms != 1 else bf16_linear
        router_logits = linear(x, module.gate.weight)
        probabilities = F.softmax(router_logits.float(), dim=-1)
        if fixed_route is None:
            weights, selected = token_topk(probabilities, self.top_k)
        else:
            # Diagnostic membership is an explicit tensor input, never a
            # production routing substitute. Probabilities and their gradients
            # still come from this model's current router logits.
            selected = fixed_route
            weights = torch.gather(probabilities, 1, selected)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routing = weights
        order, offsets, inverse = route_order(selected, self.num_experts)
        # Gather the FP32 residual first, as HF does before each expert linear.
        # Its VJP widens each BF16 expert contribution before the FP32 token sum.
        packed = F.embedding(order // self.top_k, x)
        # Stacking is ordinary differentiable composition of the original
        # expert leaves; each VJP is returned to its own HF parameter identity.
        matrices = [torch.stack([getattr(expert, name).weight for expert in module.experts]).to(
            torch.bfloat16).transpose(1, 2) for name in ("w1", "w2", "w3")]
        product = ragged_mm
        if expert_split:
            # LATCH-TRAIN-PRECISION experts_split3 (tinymoe/precision.py): every expert product, forward and backward,
            # reads the FP32 expert weights as BF16 hi + lo terms; the w1/w3 products also read their FP32 input as
            # hi + lo terms (3-term products over 3K; residuals detached). Ordinary AD of the concatenations.
            from megakernel_lab.operation_compiler.split_precision import split_ragged_mm
            weights = {name: [getattr(expert, name).weight for expert in module.experts] for name in ("w1", "w2", "w3")}
            up = split_ragged_mm(packed, weights["w1"], offsets, weight_terms=2, input_terms=2)
            gate = split_ragged_mm(packed, weights["w3"], offsets, weight_terms=2, input_terms=2)
            activated = F.silu(up) * gate
            down = split_ragged_mm(activated, weights["w2"], offsets, weight_terms=2, input_terms=1)
            choices = F.embedding(inverse, down).reshape(x.shape[0], self.top_k, x.shape[1])
            result = (choices * routing.to(hidden.dtype).unsqueeze(-1)).sum(dim=1)
            active = (offsets[1:] > offsets[:-1]).to(torch.float32)
            return result.reshape(hidden.shape), probabilities, selected, active
        if forward_fp32:
            # LATCH-TRAIN-PRECISION forward_fp32 (tinymoe/precision.py): split forward products (w1/w3 3-term over 3K,
            # w2 2-term over 2K) with unrounded FP32 results (ragged_mm_raw); the released elementwise roundings
            # s = bf16(silu(up)), a = bf16(s * gate); backward = released ragged VJP after one BF16 cotangent rounding.
            from megakernel_lab.operation_compiler.split_precision import ragged_split_operands, raw_forward_ragged_mm
            lows = {index: torch.stack([(lambda w: (w - w.to(torch.bfloat16).float()).detach())(
                getattr(expert, name).weight) for expert in module.experts]).to(torch.bfloat16)
                for index, name in enumerate(("w1", "w2", "w3"))}

            def raw_product(index, x_in, fp32_input, x_operands=None):
                x_hi, w_hi, x_split, w_split = ragged_split_operands(
                    x_in, matrices[index].transpose(1, 2), lows[index], input_terms=2 if fp32_input else 1,
                    x_operands=x_operands)
                return raw_forward_ragged_mm(x_hi, w_hi, x_split, w_split, offsets), (x_hi, x_split)
            up, shared = raw_product(0, packed, True)
            gate, _ = raw_product(2, packed, True, shared)          # one split input for w1 and w3 (pairable)
            activated = (F.silu(up).to(torch.bfloat16).float() * gate).to(torch.bfloat16)
            down, _ = raw_product(1, activated, False)
            choices = F.embedding(inverse, down).reshape(x.shape[0], self.top_k, x.shape[1])
            result = (choices * routing.to(hidden.dtype).unsqueeze(-1)).sum(dim=1)
            active = (offsets[1:] > offsets[:-1]).to(torch.float32)
            return result.reshape(hidden.shape), probabilities, selected, active
        if weight_terms > 1:
            # LATCH-TRAIN-PRECISION forward_weights_split2 / forward_split3 (tinymoe/precision.py): the forward
            # products read the master weights as BF16 hi + lo terms (and, forward_split3, the FP32 w1/w3 input as
            # hi + lo terms: 3-term products); backward = the released ragged VJP on (bf16(x), hi).
            from megakernel_lab.operation_compiler.split_precision import ragged_split_operands, split_forward_ragged_mm
            lows = {}
            for index, name in enumerate(("w1", "w2", "w3")):
                lows[index] = torch.stack([(lambda w: (w - w.to(torch.bfloat16).float()).detach())(
                    getattr(expert, name).weight) for expert in module.experts]).to(torch.bfloat16)
            terms = self.forward_input_terms

            def split_product(index, x_in, fp32_input):
                x_hi, w_hi, x_split, w_split = ragged_split_operands(
                    x_in, matrices[index].transpose(1, 2), lows[index],
                    input_terms=terms if fp32_input else 1)
                return split_forward_ragged_mm(x_hi, w_hi, x_split, w_split, offsets)
            up = split_product(0, packed, True)
            gate = split_product(2, packed, True)
            activated = F.silu(up) * gate
            down = split_product(1, activated, False)
            choices = F.embedding(inverse, down).reshape(x.shape[0], self.top_k, x.shape[1])
            result = (choices * routing.to(hidden.dtype).unsqueeze(-1)).sum(dim=1)
            active = (offsets[1:] > offsets[:-1]).to(torch.float32)
            return result.reshape(hidden.shape), probabilities, selected, active
        # Autocast narrows each of HF's two input projections separately. Their
        # BF16 input cotangents therefore widen before the FP32 branch sum.
        up = product(packed.to(torch.bfloat16), matrices[0], offsets)
        gate = product(packed.to(torch.bfloat16), matrices[2], offsets)
        activated = F.silu(up) * gate
        down = product(activated, matrices[1], offsets)
        choices = F.embedding(inverse, down).reshape(x.shape[0], self.top_k, x.shape[1])
        result = (choices * weights.to(hidden.dtype).unsqueeze(-1)).sum(dim=1)
        active = (offsets[1:] > offsets[:-1]).to(torch.float32)
        return result.reshape(hidden.shape), probabilities, selected, active

    def _readout(self, hidden):
        normalized = rms_norm(self.model.norm, hidden)
        flat = normalized.reshape(-1, normalized.shape[-1])
        if self.precision == "release":
            return bf16_linear(flat, self.lm_head.weight).float()
        from .precision import readout_linear
        return readout_linear(self.precision)(flat, self.lm_head.weight).float()

    def _linear(self, x, weight, *, fp32_result=True):
        if self.forward_weight_terms == 1:
            return bf16_linear(x, weight)
        if self.forward_fp32 and fp32_result:
            from megakernel_lab.operation_compiler.split_precision import raw_forward_linear
            return raw_forward_linear(x, weight, input_terms=2 if x.dtype == torch.float32 else 1)
        from megakernel_lab.operation_compiler.split_precision import split_forward_linear
        return split_forward_linear(x, weight, weight_terms=self.forward_weight_terms,
                                    input_terms=self.forward_input_terms if x.dtype == torch.float32 else 1)

    def forward(self, tokens, fixed_routes=None, *, state):
        if fixed_routes is not None and (fixed_routes.dtype != torch.int64 or tuple(fixed_routes.shape) != (
                len(self.model.layers), tokens.numel(), self.top_k)):
            raise ValueError("diagnostic fixed_routes must be int64 [layers, tokens, top_k]")
        hidden = F.embedding(tokens, self.model.embed_tokens.weight)
        cos, sin = self._rope(hidden, tokens)
        exits, probabilities, choices = [], [], []
        new_state = dict(state)
        for index, layer in enumerate(self.model.layers):
            hidden = hidden + self._attention(layer.self_attn, rms_norm(layer.input_layernorm, hidden), cos, sin)
            delta, probability, selected, active = self._experts(
                layer.block_sparse_moe, rms_norm(layer.post_attention_layernorm, hidden),
                None if fixed_routes is None else fixed_routes[index])
            hidden = hidden + delta
            probabilities.append(probability)
            choices.append(selected)
            new_state[f"active.{index}"] = active
            if f"selected.{index}" in state:
                new_state[f"selected.{index}"] = selected
            if f"probabilities.{index}" in state:
                new_state[f"probabilities.{index}"] = probability
            if self.auxiliary and index + 1 in (4, 8):
                exits.append(self._readout(hidden))
        logits = (self._readout(hidden), *exits)
        # HF's auxiliary loss averages after concatenating ALL layer logits.
        # This differs from averaging independently formed layer auxiliary losses.
        probs = torch.cat(probabilities, dim=0)
        selected = torch.cat(choices, dim=0)
        expert_ids = torch.arange(self.num_experts, dtype=torch.int64, device=tokens.device)
        membership = (selected.unsqueeze(-1) == expert_ids).to(torch.float32)
        token_fraction = membership.mean(dim=0)
        probability_fraction = probs.mean(dim=0)
        router_aux = self.num_experts * (token_fraction * probability_fraction.unsqueeze(0)).sum()
        return {"logits": logits, "router_aux": router_aux,
                "selected": tuple(choices), "router_probabilities": tuple(probabilities)}, new_state


def objective(prediction, targets, coefficients):
    """Two coefficients are staged from the exact host schedule; router stays separate."""
    labels = targets.reshape(-1)
    logits = prediction["logits"]
    loss = F.cross_entropy(logits[0], labels, reduction="sum") / labels.numel() * coefficients[0]
    for output in logits[1:]:
        loss = loss + F.cross_entropy(output, labels, reduction="sum") / labels.numel() * coefficients[1]
    return loss + 0.02 * prediction["router_aux"]
