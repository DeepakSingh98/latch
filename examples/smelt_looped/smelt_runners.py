# Provenance: copied from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch), original path research/smelt_latch/runners.py (blob ee816db44d8b,
# sha256 dc93cc78ab1d...; last changed at 86c4e2f42, the conventional CUDA-graph gradient-binding fix).
# Changed: imports only. research.latch_tinymoe_batch_frontier.protocol / .observe -> local smelt_recipe
# (byte-identical hyperparameters / ROUTER_COEFFICIENT / adam_slots); relative .looped -> local smelt_looped;
# the docstring's build-module name. Nothing removed; executor logic, optimizer rules, participation semantics,
# chunked CE, CUDA-graph capture and the graph-owned-gradient guard are unchanged.
"""Complete-update executors of the SAME looped TinyMoE recipe (GPU worker).

Recipe (frozen; = LATCH-TINYMOE-BATCH-FRONTIER recipe Q at B32 except for the loop): B32 x 1024 per
update, FP32 masters/moments, BF16 projections/attention/expert GEMMs, FP32 CE, AdamW beta1 0.9,
eps 1e-6, beta2 = 0.999^(B/128), decoupled weight decay per token fixed, clip 1.0, absent experts
untouched (participation = union over visits), router aux 0.02 x mean over 8-sequence groups.

  LatchLooped         normal-API Latch artifact (smelt_build.py), one launch per update
  ConventionalLooped  installed HF modules + library grouped GEMMs + fused torch AdamW
      mode grouped          eager (diagnostic only)
      mode compiled-grouped per-region torch.compile (the established C32 control) at every visit
      mode compiled-trunk   one fullgraph torch.compile over all visits
      graph=True            forward+backward (+readout chunks) captured in ONE CUDA graph and
                            replayed; embedding dW, participation sync, clip and fused AdamW eager
"""
from __future__ import annotations

import json
from pathlib import Path

DEVICE = "cuda"


def hyper(lr, sequences=32):
    import smelt_recipe as protocol
    return protocol.hyperparameters(f"L{sequences}", lr) | dict(arm=f"looped-B{sequences}")


def spec_of(text):
    from smelt_looped import LoopSpec
    if isinstance(text, LoopSpec):
        return text
    values = [int(v) for v in text.split(",")]
    return LoopSpec(*values[:3], scale=bool(values[3]) if len(values) > 3 else True)


class ConventionalLooped:
    def __init__(self, *, model_dir, spec, hp, initial, mode="compiled-grouped", chunk=8192, graph=False):
        import torch
        from megakernel_lab.tinymoe.conventional import configure_execution
        from megakernel_lab.tinymoe.model import load_model
        from smelt_looped import prepare_conventional
        self.execution = configure_execution()
        torch.manual_seed(0)
        self.spec, self.hp, self.chunk, self.mode, self.graphed = spec, hp, chunk, mode, bool(graph)
        self.graph = None
        self.model = load_model(model_dir, device=DEVICE, attention="sdpa")
        self.initial = initial
        self.restore()
        self.trunk, self.regions = prepare_conventional(self.model, spec, mode)
        self.experts = [list(layer.block_sparse_moe.experts) for layer in self.model.model.layers]
        self.reset_optimizer()
        self.model.train()

    # ---------------------------------------------------------------- state
    def _drop_gradients(self):
        # A captured CUDA graph owns the gradient buffers (every replay overwrites them); they must stay bound to
        # the parameters, or clipping/AdamW after a later restore/load would silently see only the eager
        # embedding gradient (SMELT-LATCH bug found in the first conventional learning run; see RESULTS.md).
        if self.graph is None:
            for p in self.model.parameters():
                p.grad = None

    def restore(self):
        import torch
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(self.initial[name])
        self._drop_gradients()

    def reset_optimizer(self):
        import torch
        decay, nodecay = [], []
        for p in self.model.parameters():
            (decay if p.ndim >= 2 else nodecay).append(p)
        hp = self.hp
        self.optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=hp["weight_decay"]),
                                            dict(params=nodecay, weight_decay=0.0)],
                                           lr=hp["learning_rate"], betas=(hp["beta1"], hp["beta2"]), eps=hp["eps"],
                                           fused=True)

    def parameters(self):
        return dict(self.model.named_parameters())

    def state(self):
        from smelt_recipe import adam_slots
        return dict(parameters={n: p.detach().to("cpu", copy=True) for n, p in self.model.named_parameters()},
                    optimizer=adam_slots(self.model, self.optimizer))

    def load(self, state):
        """Latch-format slots (exp_avg:/exp_avg_sq:/step:) into fused AdamW (absent step 0 = no state)."""
        import torch
        self.reset_optimizer()
        with torch.no_grad():
            self._drop_gradients()
            for name, p in self.model.named_parameters():
                p.copy_(state["parameters"][name])
                step = int(state["optimizer"]["step:" + name])
                if step:
                    self.optimizer.state[p] = dict(
                        step=torch.tensor(float(step), dtype=torch.float32, device=DEVICE),
                        exp_avg=state["optimizer"]["exp_avg:" + name].to(DEVICE, copy=True),
                        exp_avg_sq=state["optimizer"]["exp_avg_sq:" + name].to(DEVICE, copy=True))

    # ---------------------------------------------------------------- objective
    def _forward_backward(self, x, y, embedded=None):
        """CE_mean + 0.02 * mean_g aux_g; gradients in .grad (embedding excluded when `embedded`)."""
        import torch
        from torch.nn import functional as F
        from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
        from smelt_recipe import ROUTER_COEFFICIENT
        from smelt_looped import AUX_GROUP_SEQUENCES
        model = self.model
        if embedded is None:
            final, routers, active = self.trunk(x)
        else:
            final, routers, active = self._trunk_from(embedded, x)
        batch = x.shape[0]
        group = min(batch, AUX_GROUP_SEQUENCES)
        groups, rows = batch // group, group * x.shape[1]
        if groups == 1:
            router = load_balancing_loss_func(routers, model.num_experts, model.num_experts_per_tok)
        else:
            router = sum(load_balancing_loss_func(tuple(r[g * rows:(g + 1) * rows] for r in routers),
                                                  model.num_experts, model.num_experts_per_tok)
                         for g in range(groups)) / groups
        bridge = final.detach().requires_grad_(True)
        flat = bridge.reshape(-1, bridge.shape[-1])
        targets = y.reshape(-1)
        total = final.new_zeros((), dtype=torch.float32)
        for start in range(0, flat.shape[0], self.chunk):
            logits = model.lm_head(flat[start:start + self.chunk])
            ce = F.cross_entropy(logits.float(), targets[start:start + self.chunk], reduction="sum") / flat.shape[0]
            ce.backward()
            total = total + ce.detach()
        torch.autograd.backward([final, ROUTER_COEFFICIENT * router], [bridge.grad, None])
        return total + ROUTER_COEFFICIENT * router.detach(), active

    def _trunk_from(self, h, tokens):
        import torch
        from transformers.masking_utils import create_sliding_window_causal_mask
        base = self.model.model
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        pos = positions.unsqueeze(0)
        mask = create_sliding_window_causal_mask(config=base.config, input_embeds=h, attention_mask=None,
                                                 cache_position=positions, past_key_values=None, position_ids=pos)
        rope = base.rotary_emb(h, pos)
        return self.trunk.blocks(h, rope, mask, pos, positions)

    def _finish(self, objective, active, lr):
        import torch
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        stash = []
        # HF absent-gradient semantics (union over visits): one compact host sync per update.
        for experts, flags in zip(self.experts, active.cpu().tolist()):
            for expert, flag in zip(experts, flags):
                if not flag:
                    for p in expert.parameters():
                        stash.append((p, p.grad))
                        p.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp["clip_norm"], foreach=True)
        self.optimizer.step()
        if self.graphed:  # graph-owned gradient storage stays referenced for the next replay
            for p, g in stash:
                p.grad = g
        return objective, norm

    def update(self, x, y, lr):
        import torch
        from megakernel_lab.tinymoe.conventional import precision_context
        if self.graphed:
            return self._graph_update(x, y, lr)
        self.optimizer.zero_grad(set_to_none=True)
        xs, ys = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with precision_context(DEVICE, True):
            objective, active = self._forward_backward(xs, ys)
        return self._finish(objective, active, lr)

    # ---------------------------------------------------------------- CUDA graph control
    def _graph_body(self):
        import torch
        from megakernel_lab.tinymoe.conventional import precision_context
        emb = self.model.model.embed_tokens.weight
        with torch.no_grad():
            h0 = torch.nn.functional.embedding(self.static_x, emb)
        h0.requires_grad_(True)
        with precision_context(DEVICE, True):
            objective, active = self._forward_backward(self.static_x, self.static_y, embedded=h0)
        return objective, active, h0

    def _embedding_grad(self):
        import torch
        emb = self.model.model.embed_tokens.weight
        # The kernel autograd itself dispatches for F.embedding's backward (no padding index/scale).
        emb.grad = torch.ops.aten.embedding_dense_backward(self.static_h0.grad, self.static_x, emb.shape[0], -1, False)

    def _graph_update(self, x, y, lr):
        import torch
        if self.graph is None:
            self.static_x = torch.empty(x.shape, dtype=x.dtype, device=DEVICE)
            self.static_y = torch.empty(y.shape, dtype=y.dtype, device=DEVICE)
            self.static_x.copy_(x, non_blocking=True)
            self.static_y.copy_(y, non_blocking=True)
            snapshot = {n: p.detach().clone() for n, p in self.model.named_parameters()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # compilation / autotuning outside capture; no state change
                for _ in range(3):
                    for p in self.model.parameters():
                        p.grad = None
                    self._graph_body()
            torch.cuda.current_stream().wait_stream(side)
            for p in self.model.parameters():
                p.grad = None
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    p.copy_(snapshot[n])
            del snapshot
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_objective, self.static_active, self.static_h0 = self._graph_body()
        self.static_x.copy_(x, non_blocking=True)
        self.static_y.copy_(y, non_blocking=True)
        self.graph.replay()
        self._embedding_grad()
        if any(p.grad is None for p in self.model.parameters()):
            raise RuntimeError("graph-owned gradient buffers were detached from their parameters")
        return self._finish(self.static_objective, self.static_active, lr)

    def identity(self):
        from megakernel_lab.tinymoe.conventional import source_identity
        return dict(backend=f"conventional looped ({self.mode}{', CUDA graph' if self.graphed else ''})",
                    loop=self.spec.asdict(), compiled_regions=self.regions, execution_policy=self.execution,
                    optimizer="torch.optim.AdamW fused=True", chunk_tokens=self.chunk, sources=source_identity())


def build_record(artifact):
    """smelt-build.json next to the artifact (build job tree) or, for restored packs (artifact + cache only), the
    equivalent fields of the artifact's own tinymoe-build.json (written by prepare; records the captured loop)."""
    artifact = Path(artifact)
    for path in (artifact / "smelt-build.json", artifact.parent / "smelt-build.json"):
        if path.is_file():
            return json.loads(path.read_text())
    meta = json.loads((artifact / "tinymoe-build.json").read_text())
    loop = meta["loop"]
    keys = ("update_schedule", "segmented_products", "norm_fusion", "attention_rules", "accumulation_rules",
            "gradient_outputs", "matrix_schedule")
    return dict(schema="smelt-latch-build-v1 (from tinymoe-build.json)",
                loop={k: loop[k] for k in ("first", "span", "passes", "scale") if k in loop},
                batch_size=meta["batch_size"], sequence_length=meta["sequence_length"],
                options={k: meta[k] for k in keys if k in meta}, lane_sources=None,
                source_hash=meta.get("artifact", {}).get("source_hash"))


class LatchLooped:
    def __init__(self, *, artifact, hp, initial, spec=None):
        import torch
        from megakernel_lab.operation_compiler import ModuleUpdate
        from megakernel_lab.tinymoe.native import artifact_identity
        from megakernel_lab.tinymoe.learning import sb_rule
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.sb_rule, self.hp = sb_rule, hp
        self.directory = Path(artifact)
        self.build = build_record(self.directory)
        if self.build["batch_size"] != hp["sequences"]:
            raise ValueError("artifact batch differs from the recipe")
        if spec is not None and {"scale": True, **self.build["loop"]} != spec.asdict():
            raise ValueError(f"artifact loop {self.build['loop']} differs from requested {spec.asdict()}")
        self.program = ModuleUpdate.load(self.directory, device=DEVICE)
        self.artifact = artifact_identity(self.directory)
        self.initial = initial
        self.coefficients = torch.tensor([1.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.restore()
        self.reset_optimizer()

    def restore(self):
        import torch
        with torch.no_grad():
            params = self.program.parameters
            if set(params) != set(self.initial):
                raise ValueError("artifact parameter names differ from the initial state")
            for name, p in params.items():
                p.copy_(self.initial[name])
            for value in self.program.state.values():
                value.zero_()

    def reset_optimizer(self):
        import torch
        from megakernel_lab.tinymoe.optimizer import CLIP
        with torch.no_grad():
            for name, value in self.program.optimizer_state.items():
                value.fill_(1.0) if name == CLIP else value.zero_()
        self.sb_rule.stage(self.program.optimizer_state, self.hp, self.hp["learning_rate"])

    def parameters(self):
        return self.program.parameters

    def state(self):
        host = lambda values: {k: v.detach().to("cpu", copy=True) for k, v in values.items()}
        return dict(parameters=host(self.program.parameters), optimizer=host(self.program.optimizer_state),
                    model_state=host(self.program.state))

    def load(self, state):
        import torch
        with torch.no_grad():
            for name, value in self.program.parameters.items():
                value.copy_(state["parameters"][name])
            for name, value in self.program.optimizer_state.items():
                if name in state["optimizer"]:
                    value.copy_(state["optimizer"][name])
            for name, value in self.program.state.items():
                if "model_state" in state and name in state["model_state"]:
                    value.copy_(state["model_state"][name])
                else:
                    value.zero_()
        self.sb_rule.stage(self.program.optimizer_state, self.hp, self.hp["learning_rate"])

    def update(self, x, y, lr):
        from megakernel_lab.tinymoe.optimizer import LR, NORM
        opt = self.program.optimizer_state
        opt[LR].fill_(lr)
        self.program.stage((x.to(DEVICE, non_blocking=True),), (y.to(DEVICE, non_blocking=True), self.coefficients))
        self.program.step()
        return self.program.loss, opt[NORM]

    def identity(self):
        runtime = self.program.runtime
        return dict(backend="Latch generated program (one cooperative latch_ops launch per update)",
                    artifact=self.artifact, loop=self.build["loop"], build_options=self.build["options"],
                    lane_sources=self.build["lane_sources"], blocks=int(runtime.blocks),
                    multiprocessors=int(runtime.multiprocessors), phases=len(runtime.manifest.get("phases", [])))


def make(spec_text, *, model_dir, hp, initial, kind, artifact=None, mode="compiled-grouped", chunk=8192, graph=False):
    spec = spec_of(spec_text)
    if kind == "latch":
        return LatchLooped(artifact=artifact, hp=hp, initial=initial, spec=spec)
    return ConventionalLooped(model_dir=model_dir, spec=spec, hp=hp, initial=initial, mode=mode, chunk=chunk,
                              graph=graph)
