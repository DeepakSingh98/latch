# Provenance: ConventionalCadence copied from research/latch_selective_cadence/conventional.py at latch commit
# 6fcad414ad86 (git blob 74c72b8962dfb7de3cea4a255c55b7a039faf128). The class body is unchanged; imports changed only
# (research.latch_tinymoe_smallbatch.protocol -> installed megakernel_lab.tinymoe.learning.protocol; `from . import
# runners/cadence_rule` -> local selective_runners / selective_cadence_rule).
# Removed: the donor's `agree` main (native and conventional from one private parent checkpoint in one process).
# Replaced by an example main (below the class): the same start state as selective_train.py (public pretrained
# snapshot or init-state file, zero optimizer state), 16 untimed warmup microbatches then an exact reload, the timed
# run under the same study Clock, and, if --native-final is given, the donor's final-parameter distance report
# against a selective_train.py --save-final run of the same microbatches.
"""Conventional (compiled-grouped PyTorch) execution of the same cadence semantics (GPU worker).

ConventionalCadence: the study's compiled-grouped backend (deterministic algorithms, TF32 off, BF16 autocast,
chunked FP32 CE + 0.02 x HF aux per 2-sequence pass, scale 1) and DESIGN.md semantics with ordinary PyTorch parts:

* fast group: torch.optim.AdamW(fused=True), betas (0.9, beta2_2), eps 1e-6, decoupled decay wd_2 on matrices, LR of
  the microbatch; global clip over the group (clip_grad_norm_); absent experts have grad None (HF semantics).
* slow group: FP32 accumulation buffers a += grad of present experts / all other tensors; at u mod 16 = 15,
  grad = a/16 (None for experts absent in the whole window), own clip, fused AdamW with lr = kappa x mean LR,
  betas (0.9, beta2_2^16), weight decay (1 - D_w)/lr so that p <- D_w p, and step state t/16 so that its bias
  correction 1 - (beta2_2^16)^(t/16) equals the native counter's 1 - beta2_2^t.

`agree` runs the native arm and this implementation for N microbatches from one parent state on the same GPU and
reports per-microbatch loss differences, final parameter distances per group and both clocks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from megakernel_lab.tinymoe.learning.protocol import ROUTER_COEFFICIENT, Clock, Windows, write_json

import selective_runners as runners
from selective_runners import DEVICE, SEQUENCES


class ConventionalCadence:
    def __init__(self, model_dir, slow_group, kappa, chunk=512):
        import torch
        from megakernel_lab.tinymoe.conventional import configure_execution, prepare_backend
        from megakernel_lab.tinymoe.model import GroupedExpertsMoe, load_model
        import selective_cadence_rule as cadence_rule
        self.execution = configure_execution()
        torch.manual_seed(0)
        self.rule, self.kappa, self.chunk, self.hp = cadence_rule, float(kappa), chunk, runners.hyperparameters()
        self.model = load_model(model_dir, device=DEVICE, attention="sdpa")
        self.compiled = prepare_backend(self.model, "compiled-grouped")
        self.modules = [layer.block_sparse_moe for layer in self.model.model.layers]
        assert all(isinstance(m, GroupedExpertsMoe) for m in self.modules)
        self.named = dict(self.model.named_parameters())
        names = list(self.named)
        self.slow = cadence_rule.slow_set(names, slow_group) if slow_group else set()
        self.slow_group = slow_group
        self.fast = [n for n in names if n not in self.slow]
        self.roles = {n: cadence_rule.EXPERT.match(n) for n in names}
        self.roles = {n: (int(m[1]), int(m[2])) if m else None for n, m in self.roles.items()}
        self.model.train()
        self.phase = None

    def _optimizer(self, names, lr, beta2, weight_decay):
        import torch
        params = [self.named[n] for n in names]
        decay = [p for p in params if p.ndim >= 2]
        plain = [p for p in params if p.ndim < 2]
        groups = [g for g in (dict(params=decay, weight_decay=weight_decay, decay=True),
                              dict(params=plain, weight_decay=0.0, decay=False)) if g["params"]]
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, beta2), eps=1e-6, fused=True)

    def load_parent(self, runner_state):
        import torch
        with torch.no_grad():
            for name, p in self.named.items():
                p.copy_(runner_state["parameters"][name])
                p.grad = None
        opt = runner_state["optimizer"]
        self.fast_opt = self._optimizer(self.fast, 0.0, self.hp["beta2"], self.hp["weight_decay"]) if self.fast else None
        self.slow_opt = (self._optimizer(sorted(self.slow, key=list(self.named).index), 0.0,
                                         math.exp(16 * math.log(self.hp["beta2"])), 0.0) if self.slow else None)
        for optimizer, divisor in ((self.fast_opt, 1.0), (self.slow_opt, 16.0)):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                for p in group["params"]:
                    name = next(n for n, q in self.named.items() if q is p)
                    optimizer.state[p] = dict(
                        step=torch.tensor(float(int(opt["step:" + name])) / divisor, dtype=torch.float32, device=DEVICE),
                        exp_avg=opt["exp_avg:" + name].to(DEVICE).clone(),
                        exp_avg_sq=opt["exp_avg_sq:" + name].to(DEVICE).clone())
        self.acc = {n: torch.zeros_like(self.named[n]) for n in self.slow}
        self.seen = torch.zeros(len(self.modules), self.modules[0].num_experts, dtype=torch.bool, device=DEVICE)

    def microbatch(self, x, y):
        import torch
        from torch.nn import functional as F
        from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func
        from megakernel_lab.tinymoe.model import features
        model = self.model
        final, _, routers = features(model, x)
        router = load_balancing_loss_func(routers, model.num_experts, model.num_experts_per_tok)
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
        return total + ROUTER_COEFFICIENT * router.detach()

    def update(self, x, y, u):
        import torch
        from megakernel_lab.tinymoe.conventional import precision_context
        self.phase = "apply" if u % 16 == 15 else ("accumulate" if self.slow else "update")
        for p in self.named.values():
            p.grad = None
        with precision_context(DEVICE, True):
            loss = self.microbatch(x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True))
        active = torch.stack([m.active_experts for m in self.modules])
        flags = active.cpu().tolist()   # one compact sync, as the study's arm A
        absent = {n for n, r in self.roles.items() if r and not flags[r[0]][r[1]]}
        self.fast_norm = None
        if self.fast:
            for n in absent:
                if n not in self.slow:
                    self.named[n].grad = None
            params = [self.named[n] for n in self.fast if self.named[n].grad is not None]
            self.fast_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
            for group in self.fast_opt.param_groups:
                group["lr"] = runners.lr_of(u)
            self.fast_opt.step()
        if self.slow:
            present = [n for n in self.slow if n not in absent]
            torch._foreach_add_([self.acc[n] for n in present], [self.named[n].grad for n in present])
            self.seen |= active
            if self.phase == "apply":
                seen = self.seen.cpu().tolist()
                for n in self.slow:
                    r = self.roles[n]
                    self.named[n].grad = (self.acc[n] * (1.0 / 16)) if (r is None or seen[r[0]][r[1]]) else None
                params = [self.named[n] for n in self.slow if self.named[n].grad is not None]
                self.slow_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
                scal = self.rule.slow_scalars([runners.lr_of(v) for v in range(u - 15, u + 1)],
                                              self.hp["weight_decay"], self.hp["beta2"], self.kappa)
                for group in self.slow_opt.param_groups:
                    group["lr"] = scal[self.rule.S_LR]
                    if group["decay"]:   # p <- (1 - lr wd) p = D_w p
                        group["weight_decay"] = (1.0 - scal[self.rule.S_DECAY]) / scal[self.rule.S_LR]
                self.slow_opt.step()
                torch._foreach_zero_(list(self.acc.values()))
                self.seen.zero_()
        return loss

    def state_host(self):
        return {n: p.detach().to("cpu", copy=True) for n, p in self.named.items()}




# ----------------------------------------------------------------------------- example main (new; see header)
def main():
    p = argparse.ArgumentParser(description="Conventional compiled-grouped PyTorch run of the A4 cadence semantics.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--init", help="init-state .pt; default: the public pretrained snapshot")
    p.add_argument("--slow-group", default="experts")
    p.add_argument("--kappa", type=float, default=4.0)
    p.add_argument("--start-update", type=int, default=0)
    p.add_argument("--order-seed", type=int, default=1729)
    p.add_argument("--microbatches", type=int, default=64)
    p.add_argument("--warm", type=int, default=16, help="untimed conventional warmup microbatches, then reload")
    p.add_argument("--native-run", help="selective_train.py output of the same microbatches (updates.npz, "
                                        "final-parameters.pt from --save-final) for the donor's agreement report")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    import selective_start
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    u0 = int(a.start_update)
    if u0 % 16 or a.microbatches % 16:
        raise ValueError("start update and length must be window aligned")
    initial, start_identity = selective_start.initial_parameters(a.model_dir, a.init)
    parent = dict(parameters=initial, optimizer=selective_start.zero_optimizer(initial))
    train = Windows(a.data_dir, "train", order_seed=a.order_seed)
    batches = [train.batch(i * SEQUENCES, SEQUENCES, pin=True) for i in range(a.microbatches)]
    conv = ConventionalCadence(a.model_dir, a.slow_group, a.kappa)
    conv.load_parent(parent)
    for i in range(a.warm):            # compile / first launches, then exact reload of the start state
        conv.update(*batches[i % len(batches)], u0 + i)
    conv.load_parent(parent)
    clock = Clock(DEVICE)
    loss_rows, norm_rows, slow_rows, sec, applies = [], [], [], [], []
    for i, (x, y) in enumerate(batches):
        with clock:
            loss = conv.update(x, y, u0 + i)
        sec.append(clock.last)
        loss_rows.append(float(loss))
        norm_rows.append(float(conv.fast_norm) if conv.fast_norm is not None else float("nan"))
        slow_rows.append(float(conv.slow_norm) if conv.phase == "apply" else float("nan"))
        applies.append(conv.phase == "apply")
        print(json.dumps(dict(microbatch=i + 1, update=u0 + i, phase=conv.phase, loss=loss_rows[-1],
                              fast_norm=norm_rows[-1], **({"slow_norm": slow_rows[-1]} if applies[-1] else {}),
                              ms=sec[-1] * 1000)), flush=True)
    sec, applies = np.array(sec), np.array(applies)
    cycles = len(sec) // 16 * 16
    report = dict(schema="latch-selective-update-example-conventional-v1", arm="A4", kappa=a.kappa,
                  slow_group=a.slow_group, start=start_identity, start_update=u0, order_seed=a.order_seed,
                  microbatches=a.microbatches, warm=a.warm,
                  conventional=dict(backend="compiled-grouped B2 + cadence (fused AdamW groups, FP32 accumulation)",
                                    execution=conv.execution, compiled_regions=len(conv.compiled)),
                  mean_loss=float(np.mean(loss_rows)),
                  ms=dict(mean=float(sec.mean() * 1000), median=float(np.median(sec) * 1000),
                          accumulate_mean=float(sec[~applies].mean() * 1000) if (~applies).any() else None,
                          apply_mean=float(sec[applies].mean() * 1000) if applies.any() else None,
                          complete_cycle_mean=float(sec[:cycles].mean() * 1000) if cycles else None),
                  ms_by_position=[float(np.mean(sec[k::16]) * 1000) for k in range(min(16, len(sec)))],
                  device=torch.cuda.get_device_name(DEVICE))
    if a.native_run:
        native = Path(a.native_run)
        n_rows = np.load(native / "updates.npz")
        n_loss = np.asarray(n_rows["loss"][:len(loss_rows)], dtype=np.float64)
        c_loss = np.array(loss_rows)
        report["loss_abs_diff"] = dict(max=float(np.abs(n_loss - c_loss).max()), mean=float(np.abs(n_loss - c_loss).mean()),
                                       first=float(abs(n_loss[0] - c_loss[0])))
        final_path = native / "final-parameters.pt"
        if final_path.exists():
            native_final = torch.load(final_path, map_location=DEVICE, weights_only=True)
            names = list(native_final)
            dist = {}
            for label, members in (("all", names), ("slow", [n for n in names if n in conv.slow]),
                                   ("fast", [n for n in names if n not in conv.slow])):
                if not members:
                    continue
                num = math.sqrt(sum(float((conv.named[n].detach().double() - native_final[n].double()).square().sum())
                                    for n in members))
                den = math.sqrt(sum(float(native_final[n].double().square().sum()) for n in members))
                moved = math.sqrt(sum(float((native_final[n].double() - initial[n].to(DEVICE).double()).square().sum())
                                      for n in members))
                dist[label] = dict(relative_l2=num / den, native_movement_l2=moved, distance_l2=num,
                                   distance_over_movement=num / moved if moved else None)
            report["final_parameter_distance"] = dist
    write_json(out / "conventional.json", report)
    np.savez(out / "rows.npz", loss=np.array(loss_rows), fast_norm=np.array(norm_rows), slow_norm=np.array(slow_rows),
             seconds=sec, apply=applies)
    print(json.dumps({k: report[k] for k in ("arm", "ms", "mean_loss") + (("loss_abs_diff",) if a.native_run else ())}),
          flush=True)


if __name__ == "__main__":
    main()
