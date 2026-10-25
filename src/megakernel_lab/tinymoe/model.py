"""Exact public Mixtral adapter and explicit continued-training recipe.

The stock baseline uses installed Transformers. The optimized conventional
path shares its actual layers/weights, while replacing only expert execution.
No capacity limit is imposed and no token is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .pins import EXPECTED_CONFIG, EXPECTED_PARAMETERS, TRANSFORMERS_VERSION, public_identity


@dataclass(frozen=True)
class Recipe:
    name: str = "unchanged"
    horizon: int = 400
    learning_rate: float = 1e-4
    min_lr_fraction: float = 0.1
    warmup_fraction: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    clip_norm: float = 1.0

    def __post_init__(self):
        if self.name not in ("unchanged", "supervision_ema") or self.horizon < 2:
            raise ValueError("recipe must be unchanged or supervision_ema, with horizon >=2")

    def supervision_weight(self, update_index):
        if not 0 <= update_index < self.horizon:
            raise ValueError("update outside original horizon")
        return 0.5 * max(0.0, 1.0 - 2.0 * (update_index + 1) / self.horizon) if self.name == "supervision_ema" else 0.0

    def lr(self, update_index):
        import math
        warmup = max(1, int(self.horizon * self.warmup_fraction))
        if update_index < warmup:
            return self.learning_rate * (update_index + 1) / warmup
        progress = (update_index - warmup) / max(1, self.horizon - warmup - 1)
        return self.learning_rate * (self.min_lr_fraction + (1-self.min_lr_fraction) * 0.5 * (1+math.cos(math.pi*progress)))


def load_model(directory, *, device="cpu", attention="sdpa", verify=True):
    import transformers
    from transformers import AutoModelForCausalLM
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"requires transformers=={TRANSFORMERS_VERSION}; got {transformers.__version__}")
    directory = Path(directory)
    from . import variant
    if variant.is_variant(directory):
        # Explicit edited architecture (variant.py): placeholder weights only; every
        # caller overwrites all parameters with supplied fresh initial weights.
        model = variant.fresh_model(directory, attention=attention)
        if model.model.embed_tokens.weight is model.lm_head.weight:
            raise ValueError("variant must keep the untied readout")
        if any(layer.self_attn.head_dim != 48 for layer in model.model.layers):
            raise ValueError("variant changed the resolved head dimension")
        return model.to(device)
    if verify:
        from .data import digest
        identity = json.loads((directory / "latch_identity.json").read_text())
        if any(identity.get(k) != v for k, v in public_identity().items()):
            raise ValueError("public model/tokenizer identity mismatch")
        for name, spec in identity["files"].items():
            if (directory/name).stat().st_size != spec["bytes"] or digest(directory/name) != spec["sha256"]:
                raise ValueError("public snapshot hash mismatch: " + name)
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation=attention)
    for name, expected in EXPECTED_CONFIG.items():
        if getattr(model.config, name) != expected:
            raise ValueError(f"model config mismatch {name}")
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMETERS or model.model.embed_tokens.weight is model.lm_head.weight:
        raise ValueError("parameter count or untied readout mismatch")
    if any(layer.self_attn.head_dim != 48 for layer in model.model.layers):
        raise ValueError("actual resolved head dimension differs")
    return model.to(device)


def parameter_inventory(model):
    return {name: dict(shape=list(p.shape), dtype=str(p.dtype), elements=p.numel())
            for name, p in model.named_parameters()}


def select_experts(router_logits, top_k, residual_dtype):
    probabilities = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, top_k, dim=-1, largest=True, sorted=True)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.to(residual_dtype), selected


def evaluation_view(config,parameters,device):
    """Installed HF evaluator sharing live weights, without a GPU model copy."""
    from transformers import AutoModelForCausalLM
    from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
    config=copy.deepcopy(config);config._attn_implementation="sdpa"
    with torch.device("meta"):
        model=AutoModelForCausalLM.from_config(config)
    if dict(model.named_parameters()).keys()!=parameters.keys():
        raise ValueError("evaluator parameter names differ")
    for name,value in parameters.items():
        path,leaf=name.rsplit(".",1)
        model.get_submodule(path)._parameters[leaf]=nn.Parameter(value,requires_grad=False)
    model.model.rotary_emb=MixtralRotaryEmbedding(config,device=device)
    return model.eval()


LTP_DIAGNOSTIC_EXACT_ELEMENTWISE = False   # LATCH-TRAIN-PRECISION diagnostic switch; never set by a policy


class GroupedExpertsMoe(nn.Module):
    """Library grouped GEMMs with unchanged parameter names/optimizer ownership.

    Stack views are differentiable; optimizer_skip_unused restores HF's grad=None
    semantics for experts with zero membership after backward. This is necessary
    because zero-gradient AdamW and an absent gradient are different updates.
    """
    def __init__(self, original):
        super().__init__()
        self.gate, self.experts = original.gate, original.experts
        self.top_k, self.num_experts = original.top_k, original.num_experts
        self.register_buffer("active_experts", torch.zeros(self.num_experts, dtype=torch.bool,
                             device=original.gate.weight.device), persistent=False)
        if original.jitter_noise:
            raise ValueError("grouped adapter currently requires zero router jitter")

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        router_logits = self.gate(x)
        weights, selected = select_experts(router_logits, self.top_k, x.dtype)
        flat = selected.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = F.one_hot(flat, self.num_experts).sum(dim=0)
        self.active_experts.copy_(counts > 0)
        offsets = counts.cumsum(0).to(torch.int32)
        packed = x[order // self.top_k]
        autocast = torch.is_autocast_enabled(x.device.type)
        dtype = torch.get_autocast_dtype(x.device.type) if autocast else x.dtype
        if dtype != torch.bfloat16 or x.device.type != "cuda":
            raise ValueError("torch grouped MoE baseline requires BF16 CUDA autocast")
        terms = getattr(self, "forward_weight_terms", 1)
        if getattr(self, "forward_fp32", False):
            # LATCH-TRAIN-PRECISION forward_fp32: the Latch adapter's FP32-result split expert products with
            # torch._grouped_mm(out_dtype=float32); released elementwise roundings; backward = PyTorch's _grouped_mm
            # VJP on (bf16(x), hi) after one BF16 cotangent rounding.
            from megakernel_lab.operation_compiler.split_precision import (expert_terms, ragged_split_operands,
                                                                           raw_forward_grouped_mm)
            stores = {name: expert_terms([getattr(e, name).weight for e in self.experts], dtype)
                      for name in ("w1", "w2", "w3")}

            def raw_product(name, a, fp32_input, x_operands=None):
                a_hi, b_hi, a_split, b_split = ragged_split_operands(a, *stores[name], input_terms=2 if fp32_input else 1,
                                                                     x_operands=x_operands)
                return raw_forward_grouped_mm(a_hi, b_hi, a_split, b_split, offsets), (a_hi, a_split)
            with torch.autocast(x.device.type, enabled=False):
                up, shared = raw_product("w1", packed.float(), True)
                gate, _ = raw_product("w3", packed.float(), True, shared)
                if LTP_DIAGNOSTIC_EXACT_ELEMENTWISE:   # diagnostic only (research/latch_train_precision/observe.py)
                    down, _ = raw_product("w2", F.silu(up) * gate, True)
                else:
                    activated = (F.silu(up).to(dtype).float() * gate).to(dtype)
                    down, _ = raw_product("w2", activated, False)
        elif getattr(self, "expert_split", False):
            # LATCH-TRAIN-PRECISION experts_split3: the Latch adapter's split expert products (split_ragged_mm) with
            # torch._grouped_mm and ordinary AD: w1/w3 3-term over 3K, w2 2-term over 2K, forward and backward.
            from megakernel_lab.operation_compiler.split_precision import split_grouped_mm
            weights_of = lambda name: [getattr(e, name).weight for e in self.experts]
            with torch.autocast(x.device.type, enabled=False):
                up = split_grouped_mm(packed.float(), weights_of("w1"), offsets, weight_terms=2, input_terms=2)
                gate = split_grouped_mm(packed.float(), weights_of("w3"), offsets, weight_terms=2, input_terms=2)
                activated = F.silu(up) * gate
                down = split_grouped_mm(activated, weights_of("w2"), offsets, weight_terms=2, input_terms=1)
        elif terms > 1:
            # LATCH-TRAIN-PRECISION forward_weights_split2 / forward_split3 (tinymoe/precision.py): forward products
            # read the master weights as BF16 hi + lo terms (forward_split3: the FP32 w1/w3 input too); backward is
            # PyTorch's _grouped_mm VJP on (bf16(x), hi), the released arithmetic.
            from megakernel_lab.operation_compiler.split_precision import (expert_terms, ragged_split_operands,
                                                                           split_forward_grouped_mm)
            inputs = getattr(self, "forward_input_terms", 1)
            stores = {name: expert_terms([getattr(e, name).weight for e in self.experts], dtype)
                      for name in ("w1", "w2", "w3")}

            def product(name, a, fp32_input):
                a_hi, b_hi, a_split, b_split = ragged_split_operands(a, *stores[name],
                                                                     input_terms=inputs if fp32_input else 1)
                return split_forward_grouped_mm(a_hi, b_hi, a_split, b_split, offsets)
            with torch.autocast(x.device.type, enabled=False):
                up = product("w1", packed.float() if inputs > 1 else packed.to(dtype), True)
                gate = product("w3", packed.float() if inputs > 1 else packed.to(dtype), True)
                activated = F.silu(up) * gate
                down = product("w2", activated, False)
        else:
            matrices = [torch.stack([getattr(e, name).weight for e in self.experts]).to(dtype).transpose(1, 2)
                        for name in ("w1", "w2", "w3")]
            with torch.autocast(x.device.type, enabled=False):
                # The installed expert makes independent autocast input copies for
                # w1 and w3. Their cotangents must widen separately before the FP32
                # branch sum, rather than adding in a shared BF16 cast node.
                up = torch._grouped_mm(packed.to(dtype), matrices[0], offs=offsets)
                gate = torch._grouped_mm(packed.to(dtype), matrices[2], offs=offsets)
                activated = F.silu(up) * gate
                down = torch._grouped_mm(activated, matrices[1], offs=offsets)
        # Inverse permutation retains top-k order; K=2 addition is commutative.
        inverse = torch.argsort(order)
        out = (down[inverse].reshape(-1, self.top_k, shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
        return out.reshape(shape), router_logits


def apply_precision(model, policy):
    """Apply an opt-in TinyMoE training precision policy (tinymoe/precision.py) to a grouped conventional model.

    forward_weights_split2: attention q/k/v/o, router and expert products read the master weights as BF16 hi + lo
    terms in the forward (backward = the released autocast/grouped VJPs); the readout follows the policy through
    optimized_backward(precision=...). Must be applied before torch.compile of the layers (prepare_backend)."""
    from .precision import components, forward_fp32, forward_input_terms, forward_weight_terms, validate
    terms = forward_weight_terms(validate(policy))
    inputs = forward_input_terms(policy)
    raw = forward_fp32(policy)
    if "experts_split3" in components(policy):
        for layer in model.model.layers:
            if not isinstance(layer.block_sparse_moe, GroupedExpertsMoe):
                raise ValueError("expert split policy requires the grouped conventional experts")
            layer.block_sparse_moe.expert_split = True
    if terms == 1:
        return model
    from megakernel_lab.operation_compiler.split_precision import raw_forward_linear, split_forward_linear
    for layer in model.model.layers:
        linears = [getattr(layer.self_attn, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
        linears.append(layer.block_sparse_moe.gate)
        for module in linears:
            if module.bias is not None:
                raise ValueError("split forward linear policy covers bias-free projections")
            if raw and module is not layer.self_attn.v_proj:
                # forward_fp32: unrounded FP32 result (mm out_dtype=float32 of the BF16 split operands).
                module.forward = (lambda m: (lambda x: raw_forward_linear(
                    x, m.weight, input_terms=inputs if x.dtype == torch.float32 else 1, conventional=True)))(module)
                continue
            module.forward = (lambda m: (lambda x: split_forward_linear(
                x, m.weight, weight_terms=terms, input_terms=inputs if x.dtype == torch.float32 else 1)))(module)
        if not isinstance(layer.block_sparse_moe, GroupedExpertsMoe):
            raise ValueError("forward weight split policy requires the grouped conventional experts")
        layer.block_sparse_moe.forward_weight_terms = terms
        layer.block_sparse_moe.forward_input_terms = inputs
        layer.block_sparse_moe.forward_fp32 = raw
    return model


def use_grouped_experts(model):
    for layer in model.model.layers:
        layer.block_sparse_moe = GroupedExpertsMoe(layer.block_sparse_moe)
    return model


def optimizer_skip_unused(model):
    modules = [layer.block_sparse_moe for layer in model.model.layers]
    if not all(isinstance(m, GroupedExpertsMoe) for m in modules):
        return
    # One compact synchronization for the whole network, included in update time.
    masks = torch.stack([m.active_experts for m in modules]).cpu().tolist()
    for module, mask in zip(modules, masks):
        for expert, active in zip(module.experts, mask):
            if not active:
                for p in expert.parameters():
                    p.grad = None


def features(model, tokens, *, auxiliary=False):
    """Explicit common model operations, retaining router logits and exit states."""
    from transformers.masking_utils import create_sliding_window_causal_mask
    base = model.model
    h = base.embed_tokens(tokens)
    positions = torch.arange(tokens.shape[1], device=tokens.device)
    pos = positions.unsqueeze(0)
    mask = create_sliding_window_causal_mask(config=base.config, input_embeds=h,
        attention_mask=None, cache_position=positions, past_key_values=None, position_ids=pos)
    rope = base.rotary_emb(h, pos)
    exits, routers = [], []
    for index, layer in enumerate(base.layers):
        attention, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=rope,
            attention_mask=mask, position_ids=pos, cache_position=positions)
        h = h + attention
        delta, logits = layer.block_sparse_moe(layer.post_attention_layernorm(h))
        h = h + delta
        routers.append(logits)
        if auxiliary and index + 1 in (4, 8):
            exits.append(base.norm(h))
    return base.norm(h), tuple(exits), tuple(routers)


def installed_objective(model, tokens, targets, weight=0.0):
    """Unmodified installed Transformers objective plus optional shared exits."""
    captured = []
    hooks = []
    if weight > 0:
        for index in (3, 7):
            hooks.append(model.model.layers[index].register_forward_hook(lambda _m, _a, output: captured.append(output)))
    try:
        # Explicit shift_labels keeps all S supplied next-token targets, including
        # the last lookahead, and is the pinned stock ForCausalLMLoss API.
        output = model(input_ids=tokens, labels=targets, shift_labels=targets,
                       use_cache=False, output_router_logits=True)
    finally:
        for hook in hooks:
            hook.remove()
    router = output.aux_loss
    final = output.loss - model.router_aux_loss_coef * router
    extra = [F.cross_entropy(model.lm_head(model.model.norm(h)).float().reshape(-1, model.config.vocab_size),
                             targets.reshape(-1)) for h in captured]
    loss = output.loss if weight == 0 else (final + weight*sum(extra)/2)/(1+weight) + model.router_aux_loss_coef*router
    return loss, dict(final_ce=final.detach(), router_aux=router.detach(),
                      exit_ce=[x.detach() for x in extra])


def optimized_backward(model, tokens, targets, weight=0.0, *, chunk_tokens=512, precision=None):
    """Memory-bounded readout bridge, preserving every shared parameter gradient.

    Each chunk backpropagates into a detached hidden bridge AND the live shared
    readout; one final autograd call propagates every bridge and router loss into
    the common backbone. No readout or final-norm gradient is discarded.
    """
    from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
    final, exits, routers = features(model, tokens, auxiliary=weight > 0)
    router = load_balancing_loss_func(routers, model.num_experts, model.num_experts_per_tok)
    terms = [(final, 1/(1+weight))] + [(e, weight/(2*(1+weight))) for e in exits]
    values, hiddens, gradients = [], [], []
    flat_targets = targets.reshape(-1)
    # LATCH-TRAIN-PRECISION: an opt-in training precision policy (tinymoe/precision.py) replaces the readout
    # product with the policy's (same expression as the Latch adapter); None keeps the installed lm_head call.
    if precision is not None and precision != "release":
        from .precision import readout_linear
        policy_readout = readout_linear(precision)
        readout = lambda h: policy_readout(h, model.lm_head.weight)
    else:
        readout = model.lm_head
    for hidden, coefficient in terms:
        bridge = hidden.detach().requires_grad_(True)
        flat = bridge.reshape(-1, bridge.shape[-1])
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, flat.shape[0], chunk_tokens):
            logits = readout(flat[start:start+chunk_tokens])
            ce = F.cross_entropy(logits.float(), flat_targets[start:start+chunk_tokens], reduction="sum") / flat.shape[0]
            (ce * coefficient).backward()
            total = total + ce.detach()
        hiddens.append(hidden); gradients.append(bridge.grad); values.append(total)
    torch.autograd.backward(hiddens + [model.router_aux_loss_coef * router], gradients + [None])
    loss = values[0] if weight == 0 else (values[0] + weight*(values[1]+values[2])/2)/(1+weight)
    return loss + model.router_aux_loss_coef*router.detach(), dict(final_ce=values[0],
        router_aux=router.detach(), exit_ce=values[1:])


def make_optimizer(model, recipe, *, fused=True):
    # Gamma vectors are excluded from weight decay; all matrices, including
    # embeddings, routers and readout, share the declared AdamW settings.
    decay, nodecay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else nodecay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=recipe.weight_decay),
                             dict(params=nodecay, weight_decay=0.0)],
        lr=recipe.learning_rate, betas=(recipe.beta1, recipe.beta2), eps=recipe.eps,
        fused=fused)
