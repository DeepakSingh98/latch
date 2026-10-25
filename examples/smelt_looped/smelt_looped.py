# Provenance: copied from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch), original path research/smelt_latch/looped.py (blob 80ba398609e4,
# sha256 7974f168ba4e...; last changed at d2c1f2587).
# Kept unchanged: LoopSpec, _scaled, grouped_aux, looped_class, capture_factory, loop_grouped_moe_class,
# install_loop_experts, trunk_class, prepare_conventional (every numerical rule, the 1/r residual scale and the
# 8-sequence router-aux grouping are byte-identical). Imports are only from the installed megakernel_lab package,
# torch and transformers.
# Removed (CPU FP64 audit only, not used by this example): reference_fp64 and grouped_aux_fp64. The docstring's
# component list and PLAN.md reference below are the donor's text; PLAN.md is not shipped with this example (the
# deviation table is summarized in README.md).
"""SMELT-style looped TinyMoE: middle attention+MoE blocks repeated with shared weights.

Paper-based adaptation (no official code found on 2026-10-17 for arXiv 2609.01343 SMELT or
2609.40316 Loop Scaling Laws). SMELT loops "the middle half of layers twice": with L physical
layers and m = L/2, floor((L-m)/2) layers precede the looped span (1-based layers 4-9 at L=12,
its 200M configuration). Each sublayer residual update inside the looped span is scaled by 1/r,
every visit recomputes Q/K/V from the updated residual stream, and the shared layers receive
gradient contributions from every visit (full backpropagation through all passes).

Components here (TinyMoE-200m-2x16 public modules; deviations are documented in PLAN.md):

  LoopSpec            first looped layer (0-based), span, passes r; visits() = (layer, scale) list
  looped_class(spec)  CapturedTinyMoE subclass captured by Latch through the normal API
  capture_factory     drop-in replacement for megakernel_lab.tinymoe.build.capture_model
  LoopGroupedMoe      conventional grouped-GEMM experts returning the participation mask
  ConventionalTrunk   conventional looped trunk over the installed HF modules (eager or compiled)
  reference_fp64      independent FP64 reference at observed per-visit memberships (CPU audits)

Objective (both executors): mean CE over all targets + 0.02 * mean over 8-sequence groups of the
pinned HF load-balancing aux of that group, all MoE VISITS concatenated (a looped layer's two
visits are two router evaluations, exactly as two distinct layers would be). A batch of at most
8 sequences is one group. Participation (AdamW absent-expert semantics) is the union over visits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

AUX_GROUP_SEQUENCES = 8


@dataclass(frozen=True)
class LoopSpec:
    first: int = 3      # 0-based index of the first looped layer (1-based layer 4 at L=12)
    span: int = 6       # looped layers (m = L/2)
    passes: int = 2     # r
    scale: bool = True  # 1/r residual scale in the span (False: timing diagnostic only, not the SMELT model)

    @classmethod
    def smelt(cls, layers, passes=2):
        span = layers // 2
        return cls(first=(layers - span) // 2, span=span, passes=passes)

    def check(self, layers):
        if self.passes < 1 or self.span < 1 or self.first < 0 or self.first + self.span > layers:
            raise ValueError(f"invalid loop {self} for {layers} layers")

    def visits(self, layers):
        """Ordered (physical layer, residual scale) visits; scale None means unscaled (1)."""
        self.check(layers)
        scale = None if self.passes == 1 or not self.scale else 1.0 / self.passes
        out = [(i, None) for i in range(self.first)]
        for _ in range(self.passes):
            out += [(i, scale) for i in range(self.first, self.first + self.span)]
        out += [(i, None) for i in range(self.first + self.span, layers)]
        return out

    def label(self):
        return f"loop{self.first}x{self.span}r{self.passes}" + ("" if self.scale else "noscale")

    def asdict(self):
        return asdict(self)


def _scaled(delta, scale):
    # FP32 residual update times an FP32 constant (exact at r=2). Unscaled visits are untouched,
    # so passes=1 is the unlooped program.
    return delta if scale is None else delta * scale


def grouped_aux(probabilities, choices, num_experts, batch, length, device):
    """Pinned HF load-balancing aux per 8-sequence group (all visits concatenated), group mean."""
    import torch
    group = min(batch, AUX_GROUP_SEQUENCES)
    if batch % group:
        raise ValueError("batch must hold whole router-aux groups")
    groups, rows = batch // group, group * length
    expert_ids = torch.arange(num_experts, dtype=torch.int64, device=device)
    total = None
    for g in range(groups):
        probs = torch.cat([p[g * rows:(g + 1) * rows] for p in probabilities], dim=0)
        selected = torch.cat([c[g * rows:(g + 1) * rows] for c in choices], dim=0)
        membership = (selected.unsqueeze(-1) == expert_ids).to(torch.float32)
        term = num_experts * (membership.mean(dim=0) * probs.mean(dim=0).unsqueeze(0)).sum()
        total = term if total is None else total + term
    return total if groups == 1 else total / groups


# ------------------------------------------------------------------------------------- Latch capture
def looped_class(spec):
    import torch
    from megakernel_lab.tinymoe.compiled_model import CapturedTinyMoE, rms_norm

    class LoopedTinyMoE(CapturedTinyMoE):
        """CapturedTinyMoE whose forward visits the SMELT loop schedule (shared weights).

        Per visit it reuses the packaged _attention/_experts bodies unchanged; only the visit
        order, the 1/r residual scale, the union participation mask and the grouped aux differ."""
        loop = spec

        def forward(self, tokens, fixed_routes=None, *, state):
            if fixed_routes is not None or self.auxiliary:
                raise ValueError("the looped capture is plain-phase with unconstrained routing")
            layers = self.model.layers
            hidden = torch.nn.functional.embedding(tokens, self.model.embed_tokens.weight)
            cos, sin = self._rope(hidden, tokens)
            probabilities, choices, active = [], [], {}
            new_state = dict(state)
            for visit, (index, scale) in enumerate(spec.visits(len(layers))):
                layer = layers[index]
                hidden = hidden + _scaled(
                    self._attention(layer.self_attn, rms_norm(layer.input_layernorm, hidden), cos, sin), scale)
                delta, probability, selected, used = self._experts(
                    layer.block_sparse_moe, rms_norm(layer.post_attention_layernorm, hidden), None)
                hidden = hidden + _scaled(delta, scale)
                probabilities.append(probability)
                choices.append(selected)
                active[index] = used if index not in active else torch.maximum(active[index], used)
                if f"selected.v{visit}" in state:
                    new_state[f"selected.v{visit}"] = selected
                if f"probabilities.v{visit}" in state:
                    new_state[f"probabilities.v{visit}"] = probability
            for index, used in active.items():
                new_state[f"active.{index}"] = used
            batch, length = tokens.shape
            router_aux = grouped_aux(probabilities, choices, self.num_experts, batch, length, tokens.device)
            return {"logits": (self._readout(hidden),), "router_aux": router_aux,
                    "selected": tuple(choices), "router_probabilities": tuple(probabilities)}, new_state

    LoopedTinyMoE.__name__ = "LoopedTinyMoE_" + spec.label()
    return LoopedTinyMoE


def capture_factory(spec):
    """Same signature/behaviour as megakernel_lab.tinymoe.build.capture_model, looped model.

    Routing diagnostics are recorded per VISIT (selected.vJ / probabilities.vJ) so the FP64 audit
    can force each visit's actual membership. Everything else (BF16 SDPA capture, metadata,
    the normal capture_module API) is the packaged path."""
    def capture_model(model_directory, *, recipe=None, phase="plain", batch_size=1, sequence_length=512,
                      fixed_routes=None, diagnostics=True, optimizer_factory=None):
        import torch
        from megakernel_lab.operation_compiler import capture_module
        from megakernel_lab.operation_compiler.bf16_sdpa import capture_bf16_sdpa
        from megakernel_lab.tinymoe.compiled_model import objective
        from megakernel_lab.tinymoe.model import Recipe, load_model
        from megakernel_lab.tinymoe.optimizer import initial_model_state
        from megakernel_lab.tinymoe.pins import public_identity
        recipe = recipe or Recipe()
        if phase != "plain" or fixed_routes is not None or optimizer_factory is None:
            raise ValueError("looped capture: plain phase, unconstrained routing, explicit optimizer rule")
        if batch_size < 1 or not 256 <= sequence_length <= 1024 or sequence_length % 128:
            raise ValueError("requires batch>=1 and sequence 256..1024 divisible by 128")
        original = load_model(model_directory, device="cpu", attention="sdpa")
        model = looped_class(spec)(original, auxiliary=False)
        tokens = torch.zeros(batch_size, sequence_length, dtype=torch.int64)
        labels = torch.zeros_like(tokens)
        coefficients = torch.tensor([1.0, 0.0], dtype=torch.float32)
        model_state = initial_model_state(model)
        visits = spec.visits(len(model.model.layers))
        if diagnostics:
            for v in range(len(visits)):
                model_state[f"selected.v{v}"] = torch.zeros(batch_size * sequence_length, model.top_k, dtype=torch.int64)
                model_state[f"probabilities.v{v}"] = torch.zeros(batch_size * sequence_length, model.num_experts)
        with capture_bf16_sdpa():
            captured = capture_module(model, objective, (tokens,), (labels, coefficients),
                                      optimizer=optimizer_factory(model, recipe, ema_active=False), state=model_state)
        captured.ir.metadata["tinymoe"] = dict(public_identity(), recipe=asdict(recipe), phase=phase,
            batch_size=batch_size, sequence_length=sequence_length, ema_active=False, routing="unconstrained",
            routing_diagnostics=bool(diagnostics),
            objective="sum CE / target tokens + 0.02 x mean over 8-sequence groups of HF aux (all visits)",
            attention="BF16 GQA; real-width RoPE; zero pad 48 to 64; true 1/sqrt(48) scale; causal window1024",
            expert_policy=f"top{model.top_k} of {model.num_experts} token choice; no capacity/no dropped tokens; "
                          "AdamW participation = union over visits",
            loop=dict(spec.asdict(), visits=[[i, s] for i, s in visits],
                      scheme="SMELT middle-half loop, 1/r residual scale in span, shared weights, full backprop"))
        return model, captured
    return capture_model


# ------------------------------------------------------------------------------------- conventional
def loop_grouped_moe_class():
    import torch
    from torch.nn import functional as F
    from megakernel_lab.tinymoe.model import GroupedExpertsMoe, select_experts

    class LoopGroupedMoe(GroupedExpertsMoe):
        """GroupedExpertsMoe arithmetic unchanged; returns the visit's participation mask instead of
        overwriting a module buffer (a looped layer is visited more than once per update)."""

        def forward(self, hidden):
            shape = hidden.shape
            x = hidden.reshape(-1, shape[-1])
            router_logits = self.gate(x)
            weights, selected = select_experts(router_logits, self.top_k, x.dtype)
            flat = selected.reshape(-1)
            order = torch.argsort(flat, stable=True)
            counts = F.one_hot(flat, self.num_experts).sum(dim=0)
            offsets = counts.cumsum(0).to(torch.int32)
            packed = x[order // self.top_k]
            autocast = torch.is_autocast_enabled(x.device.type)
            dtype = torch.get_autocast_dtype(x.device.type) if autocast else x.dtype
            if dtype != torch.bfloat16 or x.device.type != "cuda":
                raise ValueError("torch grouped MoE baseline requires BF16 CUDA autocast")
            matrices = [torch.stack([getattr(e, name).weight for e in self.experts]).to(dtype).transpose(1, 2)
                        for name in ("w1", "w2", "w3")]
            with torch.autocast(x.device.type, enabled=False):
                up = torch._grouped_mm(packed.to(dtype), matrices[0], offs=offsets)
                gate = torch._grouped_mm(packed.to(dtype), matrices[2], offs=offsets)
                activated = F.silu(up) * gate
                down = torch._grouped_mm(activated, matrices[1], offs=offsets)
            inverse = torch.argsort(order)
            out = (down[inverse].reshape(-1, self.top_k, shape[-1]) * weights.unsqueeze(-1)).sum(dim=1)
            return out.reshape(shape), router_logits, counts > 0

    return LoopGroupedMoe


def install_loop_experts(model):
    cls = loop_grouped_moe_class()
    for layer in model.model.layers:
        layer.block_sparse_moe = cls(layer.block_sparse_moe)
    return model


def trunk_class():
    import torch
    from torch import nn

    class ConventionalTrunk(nn.Module):
        """Looped trunk over the installed HF modules: embedding/mask/RoPE as features(), then the
        visit schedule; returns final-norm hidden, router logits per visit and union masks."""

        def __init__(self, model, spec):
            super().__init__()
            self.model, self.spec = model, spec
            self.visits = spec.visits(len(model.model.layers))

        def blocks(self, h, rope, mask, pos, positions):
            base = self.model.model
            routers, active = [], {}
            for index, scale in self.visits:
                layer = base.layers[index]
                attention, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=rope,
                                               attention_mask=mask, position_ids=pos, cache_position=positions)
                h = h + _scaled(attention.float() if scale is not None else attention, scale)
                delta, logits, used = layer.block_sparse_moe(layer.post_attention_layernorm(h))
                h = h + _scaled(delta, scale)
                routers.append(logits)
                active[index] = used if index not in active else active[index] | used
            return base.norm(h), tuple(routers), torch.stack([active[i] for i in range(len(base.layers))])

        def forward(self, tokens):
            from transformers.masking_utils import create_sliding_window_causal_mask
            base = self.model.model
            h = base.embed_tokens(tokens)
            positions = torch.arange(tokens.shape[1], device=tokens.device)
            pos = positions.unsqueeze(0)
            mask = create_sliding_window_causal_mask(config=base.config, input_embeds=h, attention_mask=None,
                                                     cache_position=positions, past_key_values=None, position_ids=pos)
            rope = base.rotary_emb(h, pos)
            return self.blocks(h, rope, mask, pos, positions)

    return ConventionalTrunk


def prepare_conventional(model, spec, mode):
    """mode: grouped (eager), compiled-grouped (per-region torch.compile, the established C32 control
    applied to every visit), compiled-trunk (one fullgraph torch.compile of all visits)."""
    import torch
    install_loop_experts(model)
    trunk = trunk_class()(model, spec)
    regions = []
    if mode == "compiled-grouped":
        for index, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.forward = torch.compile(layer.block_sparse_moe.forward, fullgraph=True)
            layer.self_attn.forward = torch.compile(layer.self_attn.forward, fullgraph=True)
            regions += [f"layer{index}.moe", f"layer{index}.attention"]
    elif mode == "compiled-trunk":
        trunk.blocks = torch.compile(trunk.blocks, fullgraph=True)
        regions = ["trunk.blocks (all visits)"]
    elif mode != "grouped":
        raise ValueError("unknown conventional mode " + mode)
    return trunk, regions
