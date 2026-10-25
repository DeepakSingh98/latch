"""Observe complete optimizer updates from the seed's random weights for the CPU audit (GPU worker).

Updates 1..N of the run's real schedule/data are executed exactly as train.py does.
Arm B (Latch): before/after persistent state, raw generated gradients, loss, norm, clip
and actual memberships/probabilities per update. Arm A (conventional): actual
memberships/probabilities of every microbatch are recorded by in-place buffer copies
inside the compiled MoE region (Python side effects are not traced), plus the
accumulated FP32 gradient G exactly as fed to clipping, and before/after state.
The instrumented arm-A update is compared bitwise with the uninstrumented timed path
from the same state; the result is recorded, not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import protocol
from .protocol import Windows, write_json

DEVICE = "cuda"


def host(values):
    return {k: v.detach().to("cpu", copy=True) for k, v in values.items()}


def adam_slots(model, optimizer):
    import torch
    slots = {}
    for name, p in model.named_parameters():
        live = optimizer.state.get(p, {})
        for key in ("exp_avg", "exp_avg_sq"):
            slots[key + ":" + name] = live[key].detach().to("cpu", copy=True) if key in live else torch.zeros(p.shape)
        slots["step:" + name] = torch.tensor(int(live.get("step", 0)), dtype=torch.int64)
    return slots


def observed_moe_class():
    import torch
    from torch.nn import functional as F
    from megakernel_lab.tinymoe.model import GroupedExpertsMoe, select_experts

    class ObservedGroupedExpertsMoe(GroupedExpertsMoe):
        """GroupedExpertsMoe plus in-place copies of its actual selections (audit only)."""

        def __init__(self, original, tokens):
            super().__init__(original)
            device = original.gate.weight.device
            self.register_buffer("observed_selected", torch.zeros(tokens, self.top_k, dtype=torch.int64, device=device),
                                 persistent=False)
            self.register_buffer("observed_probabilities", torch.zeros(tokens, self.num_experts, device=device),
                                 persistent=False)

        def forward(self, hidden):
            out, router_logits = super().forward(hidden)
            x = hidden.reshape(-1, hidden.shape[-1])
            # Same select_experts call on the same logits inside the same compiled region.
            _, selected = select_experts(router_logits, self.top_k, x.dtype)
            self.observed_selected.copy_(selected)
            self.observed_probabilities.copy_(F.softmax(router_logits.detach(), dim=-1, dtype=torch.float32))
            return out, router_logits

    return ObservedGroupedExpertsMoe


def observe_conventional(a, hp, train, initial, out):
    import torch
    from .train import Conventional
    spec = protocol.ARMS["A"]
    plain = Conventional(a, hp, initial)
    # Uninstrumented reference trajectory of the timed path.
    reference = []
    for u in range(a.updates):
        x, y = train.batch(u * spec["sequences"], spec["sequences"], pin=True)
        plain.update(x, y, protocol.lr_at(u, a.learning_rate, spec["sequences"], a.horizon_tokens))
        reference.append({k: v.detach().clone() for k, v in plain.model.named_parameters()})
    del plain
    torch.cuda.empty_cache()
    # Instrumented copy of the same backend.
    from megakernel_lab.tinymoe.conventional import configure_execution, precision_context
    from megakernel_lab.tinymoe.model import load_model
    configure_execution()
    model = load_model(a.model_dir, device=DEVICE, attention="sdpa")
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(initial[name])
    cls = observed_moe_class()
    tokens = spec["microbatch"] * protocol.SEQUENCE
    for layer in model.model.layers:
        layer.block_sparse_moe = cls(layer.block_sparse_moe, tokens)
        layer.block_sparse_moe.forward = torch.compile(layer.block_sparse_moe.forward, fullgraph=True)
        layer.self_attn.forward = torch.compile(layer.self_attn.forward, fullgraph=True)
    shell = Conventional.__new__(Conventional)
    shell.model, shell.hp, shell.chunk, shell.spec = model, hp, a.chunk_tokens, spec
    shell.scale = 1.0 / spec["accumulation"]
    shell.modules = [layer.block_sparse_moe for layer in model.model.layers]
    shell.reset_optimizer()
    model.train()
    report = []
    for u in range(a.updates):
        lr = protocol.lr_at(u, a.learning_rate, spec["sequences"], a.horizon_tokens)
        x, y = train.batch(u * spec["sequences"], spec["sequences"], pin=True)
        folder = out / f"update{u + 1:02d}"
        folder.mkdir(parents=True)
        torch.save(dict(parameters=host(dict(model.named_parameters())), optimizer_state=adam_slots(model, shell.optimizer),
                        learning_rate=lr, hyperparameters=hp, update_index=u), folder / "before.pt")
        for group in shell.optimizer.param_groups:
            group["lr"] = lr
        shell.optimizer.zero_grad(set_to_none=True)
        micro, active = spec["microbatch"], None
        for k in range(spec["accumulation"]):
            xs = x[k * micro:(k + 1) * micro].to(DEVICE)
            ys = y[k * micro:(k + 1) * micro].to(DEVICE)
            with precision_context(DEVICE, True):
                value = shell.microbatch(xs, ys)
            mask = torch.stack([m.active_experts for m in shell.modules])
            active = mask if active is None else active | mask
            torch.save(dict(tokens=xs.cpu(), targets=ys.cpu(), loss=value.detach().cpu(),
                            ce_coefficient=shell.scale, router_coefficient=protocol.ROUTER_COEFFICIENT * shell.scale,
                            selected=torch.stack([m.observed_selected.cpu() for m in shell.modules]),
                            probabilities=[m.observed_probabilities.cpu() for m in shell.modules],
                            active={f"active.{i}": mask[i].float().cpu() for i in range(len(shell.modules))}),
                       folder / f"mb{k:02d}.pt")
        for module, flags in zip(shell.modules, active.cpu().tolist()):
            for expert, flag in zip(module.experts, flags):
                if not flag:
                    for p in expert.parameters():
                        p.grad = None
        G = {n: (p.grad.detach().to("cpu", copy=True) if p.grad is not None else torch.zeros(p.shape))
             for n, p in model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip_norm"], foreach=True)
        shell.optimizer.step()
        clip = min(1.0, hp["clip_norm"] / (float(norm) + 1e-6))
        torch.save(dict(accumulated_gradient=G, participation={f"seen.{i}": active[i].float().cpu()
                                                              for i in range(len(shell.modules))},
                        parameters=host(dict(model.named_parameters())), optimizer_state=adam_slots(model, shell.optimizer),
                        norm=torch.tensor(float(norm)), clip=torch.tensor(clip)), folder / "after.pt")
        mismatched = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), reference[u][n])]
        report.append(dict(update=u + 1, learning_rate=lr, norm=float(norm),
                           bitwise_equal_to_uninstrumented=not mismatched, mismatched_parameters=mismatched[:20],
                           mismatched_count=len(mismatched)))
    return report


def observe_latch(a, hp, train, initial, out):
    import torch
    from megakernel_lab.tinymoe.optimizer import CLIP, NORM
    from .train import Latch
    spec = protocol.ARMS[a.arm]
    runner = Latch(a, hp, initial)
    program, report = runner.program, []
    for u in range(a.updates):
        lr = protocol.lr_at(u, a.learning_rate, spec["sequences"], a.horizon_tokens)
        x, y = train.batch(u * spec["sequences"], spec["sequences"], pin=True)
        before = dict(parameters=host(program.parameters), optimizer_state=host(program.optimizer_state),
                      model_state=host(program.state))
        loss, norm = runner.update(x, y, lr)
        torch.cuda.synchronize(DEVICE)
        after = dict(parameters=host(program.parameters), optimizer_state=host(program.optimizer_state),
                     model_state=host(program.state))
        layers = len([k for k in after["model_state"] if k.startswith("selected.")])
        payload = dict(schema="latch-tinymoe-smallbatch-native-observation-v1", arm=a.arm, update_index=u,
                       learning_rate=lr, hyperparameters=hp, tokens=x.clone(), targets=y.clone(),
                       before=before, after=after, gradients=host(program.grads),
                       loss=loss.detach().reshape(()).to("cpu", copy=True),
                       fixed_routes=torch.stack([after["model_state"][f"selected.{i}"] for i in range(layers)]),
                       identity=runner.identity())
        path = out / f"update{u + 1:02d}.pt"
        torch.save(payload, path)
        receipt = dict(bytes=path.stat().st_size, sha256=protocol.digest(path), completed_update=u + 1)
        path.with_suffix(".pt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        report.append(dict(update=u + 1, learning_rate=lr, loss=float(loss), norm=float(after["optimizer_state"][NORM]),
                           clip=float(after["optimizer_state"][CLIP]), observation=receipt))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=tuple(protocol.ARMS), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--artifact")
    p.add_argument("--out", required=True)
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--horizon-tokens", type=int, required=True)
    p.add_argument("--order-seed", type=int, required=True)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--chunk-tokens", type=int, default=512)
    a = p.parse_args()
    a.init_seed = 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    hp = protocol.hyperparameters(a.arm, a.learning_rate)
    train = Windows(a.data_dir, "train", order_seed=a.order_seed)
    initial, init_receipt = protocol.load_initial(a.init)
    rows = (observe_conventional if a.arm == "A" else observe_latch)(a, hp, train, initial, out)
    write_json(out / "observation.json", dict(schema="latch-tinymoe-smallbatch-observation-v1", arm=a.arm,
               hyperparameters=hp, init=init_receipt, data_manifest_sha256=train.manifest_sha256,
               order_sha256=train.order_sha256(), horizon_tokens=a.horizon_tokens, updates=rows))
    print(json.dumps(rows, default=repr), flush=True)


if __name__ == "__main__":
    main()
