# Written against Latch operations; it does not import upstream Nestor. Parameter names follow upstream so
# upstream state dicts load unchanged.
"""Nestor (Ma-Lab-Berkeley/Nestor @ 9c25256f, ``nestor_100m`` and reduced fixtures) in Latch-capturable operations.

Parameters keep the upstream names, shapes and layouts exactly (stacked expert banks [E, in, out]), so upstream
state dicts load unchanged. The forward reproduces the upstream training forward under bf16 autocast with the paper's
speed stack (inline comments name each rounding point and the upstream behaviour it mirrors):

* embedding x0 = 12 * E[ids] (fp32); no residual adds -- layers communicate only through depth reads;
* per block: the column router (480->128->8 gelu score MLP, softplus noise MLP times an INJECTED N(0,1) draw,
  loss-free choice bias for selection only, renorm gate over the selected unbiased scores), shared by every layer;
* per layer: Kimi depth read -> RMSNorm -> column attention (each column's heads attend causally over its member
  tokens only, at original RoPE positions, scale 1/30) -> per-slot gated outputs; then the MLP read -> RMSNorm ->
  hybrid pool (public top-2 renorm + per-column private switch experts, path gate) -> per-slot updates;
* block exit exports per-column deviations and the aggregate; the head read over all of them -> RMSNorm -> logits/8.

Column attention packs selected token/slot rows once per block, sorted stably by (column, sequence, position).
Q/K/V and output projections operate only on those rows. Native segmented causal attention uses device offsets,
with RoPE applied at the original sequence positions. Head channels are zero-padded to a multiple of 8 for the
projections and to 64 inside attention; no sequence or column is padded to the longest segment.

Modes: ``precision="bf16"`` (the Latch program), ``"f32"`` (no autocast, as upstream eager without autocast) and
``"f64"`` (double everywhere); ``attention="latch"`` (native segmented attention), ``"math"`` (independent
gathered-attention reference), ``"packed_math"`` (packed-path eager diagnostic), or ``"dense_latch"`` (historical
dense-masked diagnostic only). ``update`` = ``accumulate`` | ``final`` | ``direct`` selects the loss-free bias rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

BF16 = torch.bfloat16
MASK_NEG = -65536.0          # exact in bf16; times the 1/30 scale -> -2184.5
HEAD_PAD = 64                # the native BF16 attention body's head width


@dataclass(frozen=True)
class ColumnPlan:
    """Device-resident stable packing for one block's fixed-shape column choices.

    ``order`` maps a packed row to ``token * choices + slot``; ``inverse`` reverses that bijection. Column-major
    segment IDs make each column's sequence segments adjacent, so the same packed rows admit eight ragged
    projection banks without another permutation. Offsets and membership remain runtime device values.
    """
    order: torch.Tensor
    inverse: torch.Tensor
    offsets: torch.Tensor
    projection_offsets: torch.Tensor
    token: torch.Tensor
    slot: torch.Tensor
    position: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def column_plan(selected, *, batch, context, cos, sin, columns=8):
    """Pack each selected row exactly once and retain original positions; reusable by every layer of a block."""
    from megakernel_lab.operation_compiler.ragged_experts import route_order
    if (selected.dtype != torch.int64 or selected.ndim != 2 or selected.shape[0] != batch * context
            or batch < 1 or context < 1 or selected.shape[1] < 1):
        raise ValueError("column choices must be int64 [batch * context, choices]")
    choices = selected.shape[1]
    token = torch.arange(batch * context, device=selected.device)
    sequence = torch.div(token, context, rounding_mode="floor")
    order, offsets, inverse = route_order(selected * batch + sequence.unsqueeze(-1), columns * batch)
    packed_token = torch.div(order, choices, rounding_mode="floor")
    slot = order - packed_token * choices
    position = packed_token - torch.div(packed_token, context, rounding_mode="floor") * context
    # Materialize the short vector: the ragged projection ABI requires contiguous offsets.
    projection_offsets = offsets[::batch].contiguous()
    return ColumnPlan(order, inverse, offsets, projection_offsets, packed_token, slot, position,
                      F.embedding(position, cos).unsqueeze(1), F.embedding(position, sin).unsqueeze(1))


@dataclass(frozen=True)
class Geometry:
    width: int
    vocab: int
    context: int
    columns: tuple
    top_k: tuple
    heads: tuple
    head_width: int
    layers: tuple
    public_experts: int
    public_top_k: int
    private_experts: int
    private_top_k: tuple
    rope_base: float
    bias_rate: float
    embedding_multiplier: float = 12.0
    logit_multiplier: float = 0.125
    norm_eps: float = 1e-5
    read_eps: float = 1e-6

    @classmethod
    def from_record(cls, shape):
        def one(v):
            values = set(v)
            if len(values) != 1:
                raise ValueError(f"per-block variation not supported: {v}")
            return values.pop()
        base = shape["rope_base"]
        if isinstance(base, (list, tuple)):
            base = one(base)
        head_width = shape["head_width"]
        if isinstance(head_width, (list, tuple)):
            head_width = one(head_width)
        return cls(width=shape["width"], vocab=shape["vocab"], context=shape["context"],
                   columns=tuple(shape["columns"]), top_k=tuple(shape["top_k"]), heads=tuple(shape["heads"]),
                   head_width=head_width, layers=tuple(shape["layers"]),
                   public_experts=one(shape["public_experts"]), public_top_k=one(shape["public_top_k"]),
                   private_experts=one(shape["private_experts"]), private_top_k=tuple(shape["private_top_k"]),
                   rope_base=float(base), bias_rate=float(shape["bias_rate"]))

    @property
    def blocks(self):
        return len(self.top_k)

    def layer_blocks(self):
        """[(global layer index, block, layer-in-block)]"""
        rows, g = [], 0
        for b, n in enumerate(self.layers):
            for l in range(n):
                rows.append((g, b, l))
                g += 1
        return rows

    @property
    def total_layers(self):
        return sum(self.layers)

    @property
    def head_sources(self):
        return 1 + sum(k + 1 for k in self.top_k)


def initial_state(geometry, *, device="cpu", observe=False, batch=1, choice_bias=None):
    """Model state: loss-free choice biases (fp32), window counts (int64) and optional routing observations."""
    G, Pn, Q, nb, nl = 8, geometry.public_experts, geometry.private_experts, geometry.blocks, geometry.total_layers
    if any(c != G for c in geometry.columns):
        raise ValueError("the port assumes 8 columns per block")
    state = {"bias_col": torch.zeros(nb, G, dtype=torch.float32, device=device),
             "bias_pub": torch.zeros(nl, Pn, dtype=torch.float32, device=device),
             "bias_priv": torch.zeros(nl, G * Q, dtype=torch.float32, device=device),
             "count_col": torch.zeros(nb, G, dtype=torch.int64, device=device),
             "count_pub": torch.zeros(nl, Pn, dtype=torch.int64, device=device),
             "count_priv": torch.zeros(nl, G * Q, dtype=torch.int64, device=device)}
    if choice_bias is not None:
        for key in ("bias_col", "bias_pub", "bias_priv"):
            state[key] = choice_bias[key].to(device=device, dtype=torch.float32).clone()
    if observe:
        tokens = batch * geometry.context
        for b, k in enumerate(geometry.top_k):
            state[f"sel_col.{b}"] = torch.zeros(tokens, k, dtype=torch.int64, device=device)
            state[f"column_offsets.{b}"] = torch.zeros(batch * G + 1, dtype=torch.int64, device=device)
        for g, b, _ in geometry.layer_blocks():
            k, kq = geometry.top_k[b], geometry.private_top_k[b]
            state[f"sel_pub.{g}"] = torch.zeros(tokens, geometry.public_top_k, dtype=torch.int64, device=device)
            state[f"sel_priv.{g}"] = torch.zeros(tokens, k * kq, dtype=torch.int64, device=device)
            state[f"attn_work.{g}"] = torch.zeros(batch * G, geometry.heads[b], 3,
                                                  dtype=torch.int64, device=device)
            for role in ("q", "k", "v", "o"):
                state[f"projection_work.{g}.{role}"] = torch.zeros(G, 4, dtype=torch.int64, device=device)
    return state


# ----------------------------------------------------------------------------------------------- upstream naming
def names(geometry):
    """Upstream parameter names (backbone stacked-bank layout) for every role, from the pinned source's modules."""
    n = {"embed": "adapter.embed.weight", "head": "head.projection.layers.0.weight",
         "final_norm": "backbone.final_norm.weight", "final_query": "backbone.final_depth_read.query",
         "final_gains": "backbone.final_depth_read.value_gains", "blocks": [], "layers": []}
    for b in range(geometry.blocks):
        r = f"backbone.blocks.{b}.group_router"
        n["blocks"].append(dict(w1=f"{r}.score_mlp.layers.0.weight", w2=f"{r}.score_mlp.layers.1.weight",
                                w3=f"{r}.noise_mlp.layers.0.weight", w4=f"{r}.noise_mlp.layers.1.weight",
                                bias=f"{r}.choice_bias"))
    for g, b, l in geometry.layer_blocks():
        p = f"backbone.blocks.{b}.layers.{l}"
        n["layers"].append(dict(
            q=[f"{p}.attention_groups.{c}.q_proj.weight" for c in range(8)],
            k=[f"{p}.attention_groups.{c}.k_proj.weight" for c in range(8)],
            v=[f"{p}.attention_groups.{c}.v_proj.weight" for c in range(8)],
            o=[f"{p}.attention_groups.{c}.out_proj.weight" for c in range(8)],
            attn_norm=f"{p}.attn_norm.weight", mlp_norm=f"{p}.mlp_norm.weight",
            attn_query=f"{p}.kimi_attn_read.query", attn_gains=f"{p}.kimi_attn_read.value_gains",
            mlp_query=f"{p}.kimi_mlp_read.query", mlp_gains=f"{p}.kimi_mlp_read.value_gains",
            pub_up=f"{p}.expert_pool.stacked_public.w_up", pub_gate=f"{p}.expert_pool.stacked_public.w_gate",
            pub_down=f"{p}.expert_pool.stacked_public.w_down",
            priv_up=f"{p}.expert_pool.stacked_private.w_up", priv_gate=f"{p}.expert_pool.stacked_private.w_gate",
            priv_down=f"{p}.expert_pool.stacked_private.w_down",
            pub_router=f"{p}.expert_pool.public_router.score_mlp.layers.0.weight",
            priv_router=[f"{p}.expert_pool.private_routers.{c}.score_mlp.layers.0.weight" for c in range(8)],
            path=f"{p}.expert_pool.path_gate.weight",
            pub_bias=f"{p}.expert_pool.public_router.choice_bias",
            priv_bias=[f"{p}.expert_pool.private_routers.{c}.choice_bias" for c in range(8)]))
    return n


def bias_state_from_upstream(geometry, state_dict):
    """Upstream choice_bias buffers -> the port's stacked bias state."""
    n = names(geometry)
    col = torch.stack([state_dict[x["bias"]].float() for x in n["blocks"]])
    pub = torch.stack([state_dict[x["pub_bias"]].float() for x in n["layers"]])
    priv = torch.stack([torch.cat([state_dict[c].float() for c in x["priv_bias"]]) for x in n["layers"]])
    return {"bias_col": col, "bias_pub": pub, "bias_priv": priv}


def bias_state_to_upstream(geometry, state):
    n = names(geometry)
    out = {}
    for b, x in enumerate(n["blocks"]):
        out[x["bias"]] = state["bias_col"][b].clone()
    for g, x in enumerate(n["layers"]):
        out[x["pub_bias"]] = state["bias_pub"][g].clone()
        for c, key in enumerate(x["priv_bias"]):
            out[key] = state["bias_priv"][g, 8 * c:8 * (c + 1)].clone()
    return out


def _module_tree(root, name, tensor):
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        if not hasattr(module, part):
            module.add_module(part, nn.Module())
        module = getattr(module, part)
    module.register_parameter(parts[-1], nn.Parameter(tensor))


class CapturedNestor(nn.Module):
    """One static microbatch shape [batch, context] over upstream-named parameters; every op is capturable."""

    def __init__(self, geometry, parameters, *, precision="bf16", attention="latch", update="direct",
                 observe=False):
        super().__init__()
        if precision not in ("bf16", "f32", "f64") or attention not in ("latch", "math", "packed_math", "dense_latch"):
            raise ValueError("precision bf16|f32|f64, attention latch|math|packed_math|dense_latch")
        if update not in ("accumulate", "final", "direct"):
            raise ValueError("update accumulate|final|direct")
        self.geometry = geometry
        self.precision, self.attention, self.update_mode, self.observe = precision, attention, update, observe
        self.forced = None   # reference audits: {"sel_col.b" | "sel_pub.g" | "sel_priv.g": int64 indices} (eager only)
        self.names = names(geometry)
        self.hi = {"bf16": torch.float32, "f32": torch.float32, "f64": torch.float64}[precision]
        self.lo = {"bf16": BF16, "f32": torch.float32, "f64": torch.float64}[precision]
        dtype = self.hi if precision == "f64" else torch.float32
        for name, tensor in parameters.items():
            _module_tree(self, name, tensor.detach().to(dtype).clone())
        missing = [x for x in self._all_names() if not self._has(x)]
        if missing:
            raise KeyError(f"{len(missing)} expected parameters missing, e.g. {missing[:4]}")
        unused = sorted(set(parameters) - set(self._all_names()))
        if unused:  # every upstream parameter must be consumed by the port (e.g. no silently ignored bias)
            raise KeyError(f"{len(unused)} upstream parameters not used by the port, e.g. {unused[:4]}")
        hw = geometry.head_width
        inv = 1.0 / (geometry.rope_base ** (torch.arange(0, hw, 2, dtype=torch.float32) / hw))
        self.register_buffer("inv_freq", inv, persistent=False)

    # ------------------------------------------------------------------ helpers
    def _all_names(self):
        n = self.names
        out = [n["embed"], n["head"], n["final_norm"], n["final_query"], n["final_gains"]]
        for x in n["blocks"]:
            out += [x["w1"], x["w2"], x["w3"], x["w4"]]
        for x in n["layers"]:
            out += x["q"] + x["k"] + x["v"] + x["o"] + x["priv_router"]
            out += [x[k] for k in ("attn_norm", "mlp_norm", "attn_query", "attn_gains", "mlp_query", "mlp_gains",
                                   "pub_up", "pub_gate", "pub_down", "priv_up", "priv_gate", "priv_down",
                                   "pub_router", "path")]
        return out

    def _forced(self, key):
        if self.forced is None:
            return None
        return self.forced[key]

    def _has(self, name):
        try:
            self.get_parameter(name)
            return True
        except AttributeError:
            return False

    def P(self, name):
        # Attribute traversal (not get_parameter): capture swaps parameters for traced tensors by attribute.
        obj = self
        for part in name.split("."):
            obj = getattr(obj, part)
        return obj

    def lin(self, x, w):
        """Autocast F.linear: operands in the low precision, output in the low precision."""
        return F.linear(x.to(self.lo), w.to(self.lo))

    def up(self, x):
        """``.float()`` of the upstream code (fp32, or f64 in the double reference mode)."""
        return x.to(self.hi)

    def low(self, x):
        return x.to(self.lo)

    def rms_norm(self, x, weight):
        if "norm64" in (getattr(self, "variant", "") or "") and x.dtype == self.hi:   # eager diagnosis only
            x64 = x.double()
            x64 = x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + self.geometry.norm_eps)
            return (x64 * weight.double()).to(x.dtype)
        x32 = self.up(x)
        x32 = x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + self.geometry.norm_eps)
        return x32.to(x.dtype) * weight.to(x.dtype)

    def gelu_lo(self, h):
        return F.gelu(self.up(h)).to(self.lo)

    def softplus_hi(self, x):
        x = self.up(x)
        return torch.where(x > 20.0, x, torch.log1p(torch.exp(torch.clamp(x, max=20.0))))

    def topk(self, scores, k):
        """torch.topk(scores, k).indices order (descending; CUDA tie law) by argmax rounds."""
        if self.precision == "bf16":
            from megakernel_lab.operation_compiler.ragged_experts import token_topk
            return token_topk(scores, k)[1]
        return torch.topk(scores, k, dim=-1).indices

    def sign(self, x):
        one = torch.ones_like(x)
        return torch.where(x > 0, one, torch.where(x < 0, -one, torch.zeros_like(x)))

    def depth_read(self, sources, query, gains):
        """DepthAttentionRead (ar:600-718): bf16 GEMV keys on raw sources, fp32 softmax * gains, fp32 mixture."""
        d = self.geometry.width
        scales, logits = [], []
        q = query.to(sources[0].dtype)
        for s in sources:
            r = torch.rsqrt((self.up(s) * self.up(s)).mean(-1, keepdim=True) + self.geometry.read_eps)
            projected = F.linear(s.to(self.lo), q.to(self.lo).unsqueeze(0))       # [T, 1] (autocast GEMV)
            scales.append(r)
            logits.append(self.up(projected * r))
        depth = torch.cat(logits, -1) / (d ** 0.5)
        variant = getattr(self, "variant", "") or ""          # eager diagnosis only (diag_gains.py); never captured
        if "gainred64" in variant:   # gains-gradient reduction over tokens in FP64 (forward value unchanged)
            weights = (torch.softmax(self.up(depth), -1).double() * gains[: len(sources)].double()).to(self.hi)
        else:
            weights = torch.softmax(self.up(depth), -1) * self.up(gains[: len(sources)])
        alpha = weights * torch.cat(scales, -1)
        if "mix64" in variant:       # mixture (and its per-token dot-product backward) in FP64
            out = sources[0].double() * alpha[:, 0:1].double()
            for i in range(1, len(sources)):
                out = out + sources[i].double() * alpha[:, i:i + 1].double()
            return out.to(self.hi)
        out = sources[0] * alpha[:, 0:1]
        for i in range(1, len(sources)):
            out = out + sources[i] * alpha[:, i:i + 1]
        return out

    def rope(self, x, positions_cos, positions_sin):
        x32 = self.up(x)
        half = x32.shape[-1] // 2
        rotated = torch.cat((-x32[..., half:], x32[..., :half]), dim=-1)
        return (x32 * positions_cos + rotated * positions_sin).to(x.dtype)

    # ------------------------------------------------------------------ routing
    def column_router(self, b, xr, eps, bias):
        x = self.names["blocks"][b]
        s = self.lin(self.gelu_lo(self.lin(xr, self.P(x["w1"]))), self.P(x["w2"]))              # bf16 [T, 8]
        std = self.softplus_hi(self.lin(self.gelu_lo(self.lin(xr, self.P(x["w3"]))), self.P(x["w4"])))
        scores = self.up(s) + self.up(eps.to(self.lo)) * std                                     # fp32 (training)
        idx = self._forced(f"sel_col.{b}")
        if idx is None:
            idx = self.topk(scores + bias, self.geometry.top_k[b])
        gate = torch.softmax(self.up(scores.gather(-1, idx)), -1)                                # renorm, unbiased
        return idx, gate

    # ------------------------------------------------------------------ attention
    def _packed_column_attention(self, g, b, n_a, idx, gate, plan, work_state=None):
        from megakernel_lab.operation_compiler.ragged_experts import ragged_mm, ragged_mm_observed
        from megakernel_lab.operation_compiler.row_permutation import permutation_gather
        geo, x = self.geometry, self.names["layers"][g]
        H, hw, D, T, choices = geo.heads[b], geo.head_width, geo.width, n_a.shape[0], idx.shape[1]
        padded = (hw + 7) // 8 * 8
        if hw % 2 or padded > HEAD_PAD:
            raise ValueError("packed column attention requires an even head width at most 64")
        # Gather while still in FP32: the duplicate-token VJP sums all selected-column cotangents before the
        # upstream high-precision depth-read path. Each separate BF16 projection retains its rounding boundary.
        rows = F.embedding(plan.token, n_a)

        def product(operand, bank, role):
            if work_state is None or self.attention != "latch":
                return ragged_mm(operand, bank, plan.projection_offsets)
            result, work = ragged_mm_observed(operand, bank, plan.projection_offsets)
            work_state[f"projection_work.{g}.{role}"] = work
            return result

        def project(role):
            banks = []
            for name in x[role]:
                weight = self.P(name).view(H, hw, D)
                banks.append(torch.cat((weight, weight.new_zeros((H, padded - hw, D))), 1)
                             .reshape(H * padded, D))
            # Preserve upstream K-major matrices when stacking so dynamic-row products admit the TMA body.
            # Keep Q/K/V casts distinct: each low-precision input cotangent is widened before their sum, as
            # in three upstream autocast F.linear calls. Sharing one BF16 row tensor would round that sum.
            return product(rows.to(self.lo), torch.stack(banks).transpose(1, 2).to(self.lo), role).view(-1, H, padded)

        q, k, v = (project(role) for role in ("q", "k", "v"))

        def rotate(value):
            # Slice in the high precision so the captured slice VJP is FP32. Padding channels carry exact zero;
            # the rotation is solely over the original upstream head width (30 in the published model).
            original = self.up(value)[..., :hw]
            result = self.rope(original, plan.cos, plan.sin).to(self.lo)
            return torch.cat((result, result.new_zeros((*result.shape[:-1], padded - hw))), -1)

        q, k = rotate(q), rotate(k)
        if self.attention == "latch":
            from megakernel_lab.operation_compiler.segmented_attention import sdpa
            if work_state is None:
                attended = sdpa(q, k, v, plan.offsets, max_seqlen=geo.context, scale=1.0 / hw)
            else:
                attended, work = sdpa(q, k, v, plan.offsets, max_seqlen=geo.context, scale=1.0 / hw,
                                      return_work=True)
                work_state[f"attn_work.{g}"] = work
        else:
            # Eager-only packed-path diagnostic; the normal Latch path above never runs an external attention.
            pieces = []
            boundaries = plan.offsets.tolist()
            for lo, hi in zip(boundaries, boundaries[1:]):
                qc, kc, vc = (self.up(t[lo:hi]).transpose(0, 1) for t in (q, k, v))
                scores = torch.matmul(qc, kc.transpose(-1, -2)) * (1.0 / hw)
                causal = torch.ones((hi - lo, hi - lo), dtype=torch.bool, device=q.device).tril()
                probabilities = torch.softmax(scores.masked_fill(~causal, float("-inf")), -1)
                pieces.append(torch.matmul(probabilities, vc).transpose(0, 1))
            attended = torch.cat(pieces).to(self.lo)
        banks = []
        for name in x["o"]:
            weight = self.P(name).view(D, H, hw)
            banks.append(torch.cat((weight, weight.new_zeros((D, H, padded - hw))), -1)
                         .reshape(D, H * padded))
        output = product(attended.reshape(T * choices, H * padded), torch.stack(banks).transpose(1, 2).to(self.lo), "o")
        # Inverse is bijective even when a token selected several columns. Its VJP is an exact row copy;
        # duplicate-token accumulation belongs to the input gather above, not to this unpacking operation.
        output = permutation_gather(output, plan.inverse, plan.order).view(T, choices, D)
        return self.up(output) * gate.unsqueeze(-1)

    def column_attention(self, g, b, n_a, idx, gate, member, cos, sin, batch, plan=None, work_state=None):
        if self.attention in ("latch", "packed_math"):
            if plan is None:
                raise ValueError("packed column attention requires its block's ColumnPlan")
            return self._packed_column_attention(g, b, n_a, idx, gate, plan, work_state=work_state)
        geo, x = self.geometry, self.names["layers"][g]
        H, hw, T, S = geo.heads[b], geo.head_width, n_a.shape[0], geo.context
        # three projections (columns concatenated): no bf16 select of a fused qkv (Latch lowers select_backward
        # only for FP32/int64 operands)
        q, k, v = (self.lin(n_a, torch.cat([self.P(n) for n in x[r]], 0)).view(batch, S, 8 * H, hw).transpose(1, 2)
                   for r in ("q", "k", "v"))                                                    # [B, 8H, S, hw]
        q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
        # member [B, S, 8] -> per head [B, 8H, S, 1]
        keep = member.view(batch, S, 8, 1).expand(batch, S, 8, H).reshape(batch, S, 8 * H).transpose(1, 2)
        if self.attention == "dense_latch":
            spare = torch.where(keep, 0.0, MASK_NEG).to(self.lo).unsqueeze(-1)
            ones = torch.ones_like(spare)
            q64 = torch.cat((q, ones, torch.zeros_like(q[..., :1]).expand(*q.shape[:-1], HEAD_PAD - hw - 1)), -1)
            k64 = torch.cat((k, spare, torch.zeros_like(k[..., :1]).expand(*k.shape[:-1], HEAD_PAD - hw - 1)), -1)
            v64 = torch.cat((v, torch.zeros_like(v[..., :1]).expand(*v.shape[:-1], HEAD_PAD - hw)), -1)
            from megakernel_lab.operation_compiler import bf16_sdpa
            y = bf16_sdpa.sdpa(q64, k64, v64, is_causal=True, scale=1.0 / hw)                 # [B, 8H, S, 64]
            # Output projection at the padded head width (exact): value channels hw..63 are zero, so the attention
            # output's padded channels are exactly zero; each column's out_proj weight is zero-padded per head to
            # [D, H*64]. The products are unchanged; the contraction width H*64 admits Latch's tensor-core matrix
            # bodies (H*30 = 150/90/60 is not a multiple of 8 and fell back to the SIMT body: ~45% of the update).
            y = y.transpose(1, 2).reshape(T, 8, H * HEAD_PAD).transpose(0, 1)                # [8, T, H*64]
            D = geo.width
            wo = torch.stack([torch.cat((self.P(n).view(D, H, hw),
                                         torch.zeros((D, H, HEAD_PAD - hw), dtype=self.P(n).dtype,
                                                     device=self.P(n).device)), -1).reshape(D, H * HEAD_PAD)
                              for n in x["o"]]).transpose(1, 2)                                 # [8, H*64, D]
            out = torch.bmm(y.to(self.lo), wo.to(self.lo)).transpose(0, 1).reshape(T * 8, D)    # [T*8, D]
            flat = torch.arange(T, device=idx.device).unsqueeze(-1) * 8 + idx                  # [T, k]
            return self.up(F.embedding(flat, out)) * gate.unsqueeze(-1)                        # [T, k, D] fp32
        else:
            # Reference spelling (independent of the spare-channel trick): genuinely gathered variable-length causal
            # attention per (sequence, column) over member tokens only, at their original RoPE positions, as
            # upstream's packed-varlen dispatch; non-member rows of a column are zero (never gathered downstream).
            rows = []
            for bi in range(batch):
                cols = []
                for c in range(8):
                    pos = torch.nonzero(member[bi, :, c]).squeeze(-1)                          # ascending order
                    heads = slice(c * H, (c + 1) * H)
                    yc = q.new_zeros((H, S, hw), dtype=self.hi)
                    if pos.numel():
                        qc, kc, vc = (self.up(t[bi, heads].index_select(1, pos)) for t in (q, k, v))
                        logits = torch.matmul(qc, kc.transpose(-1, -2)) * (1.0 / hw)
                        n = pos.numel()
                        causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
                        probs = torch.softmax(logits.masked_fill(~causal, float("-inf")), -1)
                        yc = yc.index_copy(1, pos, torch.matmul(probs, vc))
                    cols.append(yc)
                rows.append(torch.cat(cols, 0))
            y = torch.stack(rows).to(self.lo)                                                  # [B, 8H, S, hw]
        y = y.transpose(1, 2).reshape(T, 8, H * hw).transpose(0, 1)                           # [8, T, H*hw]
        wo = torch.stack([self.P(n) for n in x["o"]]).transpose(1, 2)                          # [8, H*hw, D]
        out = torch.bmm(y.to(self.lo), wo.to(self.lo)).transpose(0, 1).reshape(T * 8, geo.width)  # [T*8, D]
        flat = torch.arange(T, device=idx.device).unsqueeze(-1) * 8 + idx                      # [T, k]
        return self.up(F.embedding(flat, out)) * gate.unsqueeze(-1)                            # [T, k, D] fp32

    # ------------------------------------------------------------------ experts
    def swiglu_experts(self, rows, up_w, gate_w, down_w, offsets):
        from megakernel_lab.operation_compiler.ragged_experts import ragged_mm
        lo = self.lo
        upv = ragged_mm(rows, up_w.to(lo), offsets)
        gv = ragged_mm(rows, gate_w.to(lo), offsets)
        act = F.silu(self.up(gv)).to(lo)
        hidden = (self.up(act) * self.up(upv)).to(lo)
        return ragged_mm(hidden, down_w.to(lo), offsets)

    def expert_pool(self, g, b, n_m, idx, gate, bias_pub, bias_priv):
        from megakernel_lab.operation_compiler.ragged_experts import route_order
        geo, x = self.geometry, self.names["layers"][g]
        T, D, k, kq, Pk = n_m.shape[0], geo.width, geo.top_k[b], geo.private_top_k[b], geo.public_top_k
        # public pool: renorm top-2 of a bias-free linear router
        sp = self.lin(n_m, self.P(x["pub_router"]))                                            # bf16 [T, 8]
        sel_p = self._forced(f"sel_pub.{g}")
        if sel_p is None:
            sel_p = self.topk(self.up(sp) + bias_pub, Pk)
        g_p = torch.softmax(self.up(sp).gather(-1, sel_p), -1).to(sp.dtype)
        order, offsets, inverse = route_order(sel_p, geo.public_experts)
        rows = F.embedding(torch.div(order, Pk, rounding_mode="floor"), n_m).to(self.lo)
        down = self.swiglu_experts(rows, self.P(x["pub_up"]), self.P(x["pub_gate"]), self.P(x["pub_down"]), offsets)
        terms = (F.embedding(inverse, down).view(T, Pk, D) * g_p.unsqueeze(-1).to(down.dtype))   # per-term bf16
        pub = self.up(terms).sum(1)
        counts_pub = offsets[1:] - offsets[:-1]
        # private pools: per selected column, switch top-kq of that column's 8 experts (bank id = column*8+e)
        wr = torch.cat([self.P(n) for n in x["priv_router"]], 0)                               # [64, D]
        s_all = self.lin(n_m, wr).reshape(T * 8, 8)                                             # bf16
        flat = torch.arange(T, device=idx.device).unsqueeze(-1) * 8 + idx                       # [T, k]
        s_rows = F.embedding(flat, s_all)                                                       # [T, k, 8] bf16
        b_rows = F.embedding(idx, bias_priv.view(8, 8))                                         # [T, k, 8]
        forced_ids = self._forced(f"sel_priv.{g}")
        if forced_ids is None:
            sel_e = self.topk((self.up(s_rows) + b_rows).reshape(T * k, 8), kq).reshape(T, k, kq)
        else:
            sel_e = forced_ids.view(T, k, kq) - idx.unsqueeze(-1) * 8
            if bool(((sel_e < 0) | (sel_e >= 8)).any()):
                raise ValueError("forced private ids do not belong to the forced columns")
        g_e = torch.softmax(self.up(s_rows), -1).gather(-1, sel_e).to(s_rows.dtype)             # switch, bf16
        weight = self.up(g_e) * gate.unsqueeze(-1)                                              # [T, k, kq] fp32
        ids = (idx.unsqueeze(-1) * 8 + sel_e).reshape(T, k * kq)
        order, offsets, inverse = route_order(ids, 8 * geo.private_experts)
        rows = F.embedding(torch.div(order, k * kq, rounding_mode="floor"), n_m).to(self.lo)
        down = self.swiglu_experts(rows, self.P(x["priv_up"]), self.P(x["priv_gate"]), self.P(x["priv_down"]),
                                   offsets)
        choice = F.embedding(inverse, down).view(T, k, kq, D)
        priv = (self.up(choice) * weight.unsqueeze(-1)).sum(2)                                  # [T, k, D]
        counts_priv = offsets[1:] - offsets[:-1]
        alpha = torch.softmax(self.up(self.lin(n_m, self.P(x["path"]))), -1)                    # [T, 2]
        return (alpha[:, 0:1] * pub, alpha[:, 1].view(T, 1, 1) * priv, counts_pub, counts_priv,
                sel_p, ids)

    # ------------------------------------------------------------------ forward
    def forward(self, tokens, noise, *, state):
        geo, n = self.geometry, self.names
        batch, S = tokens.shape
        if S != geo.context:
            raise ValueError("static context mismatch")
        T, D = batch * S, geo.width
        x0 = F.embedding(tokens.reshape(-1), self.P(n["embed"])) * geo.embedding_multiplier      # fp32 [T, D]
        positions = torch.arange(S, dtype=torch.float32, device=tokens.device)
        angles = positions.unsqueeze(-1) * self.inv_freq.to(tokens.device)
        emb = torch.cat((angles, angles), -1)
        cos, sin = self.up(emb.cos()), self.up(emb.sin())                                       # [S, hw]
        new_state = dict(state)
        narrow, wide = [x0], [x0]
        xr = x0
        counts_col, counts_pub, counts_priv = [], [], []
        layer = 0
        for b in range(geo.blocks):
            k = geo.top_k[b]
            idx, gate = self.column_router(b, xr, noise[b].reshape(T, 8), state["bias_col"][b])
            if self.attention in ("latch", "packed_math"):
                plan = column_plan(idx, batch=batch, context=S, cos=cos, sin=sin)
                counts_col.append(plan.projection_offsets[1:] - plan.projection_offsets[:-1])
                member = None
            else:
                plan = None
                columns = torch.arange(8, device=idx.device)
                hits = (idx.unsqueeze(-1) == columns).to(torch.int64)                          # [T, k, 8]
                counts_col.append(hits.sum((0, 1)))
                member = hits.sum(1).view(batch, S, 8) > 0
            if self.observe:
                new_state[f"sel_col.{b}"] = idx
                if plan is not None:
                    new_state[f"column_offsets.{b}"] = plan.offsets
            streams, agg = None, None
            for _ in range(geo.layers[b]):
                x = n["layers"][layer]
                src_a = narrow if streams is None else narrow + [streams.mean(1)]
                n_a = self.rms_norm(self.depth_read(src_a, self.P(x["attn_query"]), self.P(x["attn_gains"])),
                                    self.P(x["attn_norm"]))
                u_a = self.column_attention(layer, b, n_a, idx, gate, member, cos, sin, batch,
                                             plan=plan, work_state=new_state if self.observe else None)
                run = agg if agg is not None else (streams.mean(1) if streams is not None else None)
                am = u_a.mean(1)
                src_m = narrow + [am if run is None else run + am]
                n_m = self.rms_norm(self.depth_read(src_m, self.P(x["mlp_query"]), self.P(x["mlp_gains"])),
                                    self.P(x["mlp_norm"]))
                pub, priv, c_pub, c_priv, sel_p, ids = self.expert_pool(
                    layer, b, n_m, idx, gate, state["bias_pub"][layer], state["bias_priv"][layer])
                counts_pub.append(c_pub)
                counts_priv.append(c_priv)
                if self.observe:
                    new_state[f"sel_pub.{layer}"] = sel_p
                    new_state[f"sel_priv.{layer}"] = ids
                update = u_a + priv + pub.unsqueeze(1)
                streams = update if streams is None else streams + update
                agg = update.mean(1) if run is None else run + update.mean(1)
                layer += 1
            aggregate = streams.mean(1)
            wide += [streams[:, j] - aggregate for j in range(k)] + [aggregate]
            narrow = narrow + [aggregate]
            xr = aggregate
        y = self.rms_norm(self.depth_read(wide, self.P(n["final_query"]), self.P(n["final_gains"])),
                          self.P(n["final_norm"]))
        rows = y.view(batch, S, D)[:, :-1].reshape(-1, D)
        logits = self.up(self.lin(rows * geo.logit_multiplier, self.P(n["head"])))
        self._update_routing_state(state, new_state, torch.stack(counts_col), torch.stack(counts_pub),
                                   torch.stack(counts_priv))
        return {"logits": logits}, new_state

    def _update_routing_state(self, state, new_state, c_col, c_pub, c_priv):
        totals = {"col": state["count_col"] + c_col, "pub": state["count_pub"] + c_pub,
                  "priv": state["count_priv"] + c_priv}
        if self.update_mode == "accumulate":
            new_state.update(count_col=totals["col"], count_pub=totals["pub"], count_priv=totals["priv"])
            return
        rate = self.geometry.bias_rate
        for key, groups in (("col", 8), ("pub", self.geometry.public_experts), ("priv", self.geometry.private_experts)):
            counts = totals[key].to(torch.float32)
            shaped = counts.view(counts.shape[0], -1, groups)
            error = shaped.mean(-1, keepdim=True) - shaped
            step = (self.sign(error) * rate).view(counts.shape)
            new_state["bias_" + key] = state["bias_" + key] + step
            new_state["count_" + key] = torch.zeros_like(state["count_" + key])


def objective(prediction, labels, coefficients):
    """Upstream CausalLMLoss: sum CE over the B*(S-1) shifted rows / their count, times the staged 1/accum."""
    targets = labels[:, 1:].reshape(-1)
    ce = F.cross_entropy(prediction["logits"], targets, reduction="sum") / targets.numel()
    return ce * coefficients[0]
