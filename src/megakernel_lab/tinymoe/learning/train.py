"""One TinyMoE training run from random initialization for one arm (GPU worker).

  --arm A   compiled-grouped conventional (existing LATCH-TINYMOE backend; GA loop of
            LATCH-TINYMOE-GA): 16 microbatches x 8 x 1024 per optimizer update, fused AdamW.
  --arm B1  Latch generated program, 1 x 1024 per optimizer update, no accumulation.
  --arm B2  Latch generated program, 2 x 1024 per optimizer update, no accumulation.

Primary clock: accumulated synchronized wall seconds of complete optimizer updates,
from after host batch readiness through backend staging (token H2D copies, schedule
fills), forward/backward, accumulation, clipping, optimizer and absent-expert
bookkeeping, to update-completion synchronization. Excluded and reported separately:
compilation/warmup, evaluation, checkpoint/parameter writes and host data loading.
Every warmup update is undone (initial weights restored, optimizer state reset) so
the measured run starts from the exact seed initialization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from . import protocol
from .protocol import ROUTER_COEFFICIENT, SEQUENCE, Clock, Windows, write_json

DEVICE = "cuda"


# ----------------------------------------------------------------------------- arm A
class Conventional:
    def __init__(self, a, hp, initial):
        import torch
        from megakernel_lab.tinymoe.conventional import configure_execution, prepare_backend
        from megakernel_lab.tinymoe.model import GroupedExpertsMoe, load_model
        self.execution = configure_execution()
        torch.manual_seed(0)  # no stochastic operations in training (zero dropout/jitter)
        self.model = load_model(a.model_dir, device=DEVICE, attention="sdpa")
        self.initial = initial
        self.restore()
        self.compiled = prepare_backend(self.model, "compiled-grouped")
        self.modules = [layer.block_sparse_moe for layer in self.model.model.layers]
        assert all(isinstance(m, GroupedExpertsMoe) for m in self.modules)
        self.hp, self.chunk = hp, a.chunk_tokens
        self.spec = protocol.ARMS["A"]
        self.scale = 1.0 / self.spec["accumulation"]
        self.reset_optimizer()
        self.model.train()

    def restore(self):
        import torch
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(self.initial[name])
                p.grad = None

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
        return dict(parameters={n: p.detach().to("cpu", copy=True) for n, p in self.model.named_parameters()},
                    optimizer=self.optimizer.state_dict())

    def load(self, state):
        import torch
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(state["parameters"][name])
                p.grad = None
        self.reset_optimizer()
        self.optimizer.load_state_dict(state["optimizer"])

    def microbatch(self, x, y):
        """(CE_mean + 0.02 aux) / 16 for one 8-sequence pass; gradients accumulate in .grad."""
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
            (ce * self.scale).backward()
            total = total + ce.detach()
        torch.autograd.backward([final, (ROUTER_COEFFICIENT * self.scale) * router], [bridge.grad, None])
        return (total + ROUTER_COEFFICIENT * router.detach()) * self.scale

    def update(self, x, y, lr):
        import torch
        from megakernel_lab.tinymoe.conventional import precision_context
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        micro = self.spec["microbatch"]
        active, objective = None, None
        for k in range(self.spec["accumulation"]):
            xs = x[k * micro:(k + 1) * micro].to(DEVICE, non_blocking=True)
            ys = y[k * micro:(k + 1) * micro].to(DEVICE, non_blocking=True)
            with precision_context(DEVICE, True):
                value = self.microbatch(xs, ys)
            objective = value if objective is None else objective + value
            mask = torch.stack([m.active_experts for m in self.modules])
            active = mask if active is None else active | mask
        # HF absent-gradient semantics over the whole update: one compact sync.
        for module, flags in zip(self.modules, active.cpu().tolist()):
            for expert, flag in zip(module.experts, flags):
                if not flag:
                    for p in expert.parameters():
                        p.grad = None
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp["clip_norm"], foreach=True)
        self.optimizer.step()
        return objective, norm

    def identity(self):
        from megakernel_lab.tinymoe.conventional import source_identity
        return dict(backend="compiled-grouped + gradient accumulation", compiled_regions=self.compiled,
                    execution_policy=self.execution, optimizer="torch.optim.AdamW fused=True",
                    sources=source_identity(), microbatch=self.spec["microbatch"],
                    accumulation=self.spec["accumulation"], chunk_tokens=self.chunk)


# ----------------------------------------------------------------------------- arm B
class Latch:
    def __init__(self, a, hp, initial):
        import torch
        from megakernel_lab.operation_compiler import ModuleUpdate
        from megakernel_lab.tinymoe.native import artifact_identity
        from . import sb_rule
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.sb_rule, self.hp = sb_rule, hp
        self.directory = Path(a.artifact)
        build = json.loads((self.directory.parent / "sb-build.json").read_text()) \
            if (self.directory.parent / "sb-build.json").exists() else json.loads((self.directory / "sb-build.json").read_text())
        if build["batch_size"] != hp["sequences"] or build["sequence_length"] != SEQUENCE:
            raise ValueError("artifact shape differs from the requested arm")
        self.build = build
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
            for key, values in (("parameters", self.program.parameters), ("optimizer", self.program.optimizer_state),
                                ("model_state", self.program.state)):
                if set(values) != set(state[key]):
                    raise ValueError("checkpoint tensor names differ: " + key)
                for name, value in values.items():
                    value.copy_(state[key][name])

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
                    artifact=self.artifact, build_flags=self.build["flags"], lane_sources=self.build["lane_sources"],
                    blocks=int(runtime.blocks), multiprocessors=int(runtime.multiprocessors),
                    phases=len(runtime.manifest.get("phases", [])))


# ----------------------------------------------------------------------------- loop
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=tuple(protocol.ARMS), required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--init", required=True, help="init_state.py output (init-sSEED.pt)")
    p.add_argument("--artifact", help="Latch artifact directory (arms B1/B2)")
    p.add_argument("--out", required=True)
    p.add_argument("--learning-rate", type=float, required=True, help="peak LR")
    p.add_argument("--horizon-tokens", type=int, required=True)
    p.add_argument("--stop-tokens", type=int, default=0, help="stop early (schedule keeps the horizon); 0 = horizon")
    p.add_argument("--order-seed", type=int, required=True)
    p.add_argument("--eval-every-tokens", type=int, required=True)
    p.add_argument("--eval-windows", type=int, default=256)
    p.add_argument("--final-eval-windows", type=int, default=2048)
    p.add_argument("--warmup-updates", type=int, default=3)
    p.add_argument("--chunk-tokens", type=int, default=512)
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--checkpoint-every-tokens", type=int, default=0,
                   help="overwrite out/checkpoint.pt after the evaluation at each multiple (exact resume)")
    p.add_argument("--resume", "--resume-uri", dest="resume_uri",
                   help="local checkpoint.pt from a trusted run with identical settings")
    a = p.parse_args()
    import numpy as np
    import torch
    started = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    spec = protocol.ARMS[a.arm]
    hp = protocol.hyperparameters(a.arm, a.learning_rate)
    per_update = protocol.tokens_per_update(spec["sequences"])
    if a.horizon_tokens % per_update or a.eval_every_tokens % protocol.tokens_per_update(128):
        raise ValueError("horizon must be whole updates; evaluation points must be common to all arms")
    updates = a.horizon_tokens // per_update
    stop = (a.stop_tokens or a.horizon_tokens) // per_update
    if (a.horizon_tokens <= 0 or a.eval_every_tokens <= 0 or stop <= 0 or stop > updates
            or (a.stop_tokens and a.stop_tokens % per_update)):
        raise ValueError("positive horizons and whole-update stop points within the horizon are required")
    if a.checkpoint_every_tokens and a.checkpoint_every_tokens % a.eval_every_tokens:
        raise ValueError("checkpoint interval must be a multiple of the evaluation interval")
    train = Windows(a.data_dir, "train", order_seed=a.order_seed)
    dev = Windows(a.data_dir, "validation")
    initial, init_receipt = protocol.load_initial(a.init)
    from megakernel_lab.tinymoe.native import evaluator_config
    config = evaluator_config(a.model_dir)
    setup_start = time.perf_counter()
    runner = (Conventional if spec["backend"] == "conventional" else Latch)(a, hp, initial)
    setup_seconds = time.perf_counter() - setup_start
    sequences = spec["sequences"]

    # Warmup (compilation / first launches) on the run's first batches, then undo.
    warm_start = time.perf_counter()
    warm = []
    for u in range(a.warmup_updates):
        x, y = train.batch(u * sequences, sequences, pin=True)
        with Clock(DEVICE) as c:
            runner.update(x, y, protocol.lr_at(u, a.learning_rate, sequences, a.horizon_tokens))
        warm.append(c.last)
    runner.restore()
    runner.reset_optimizer()
    torch.cuda.synchronize(DEVICE)
    warmup_seconds = time.perf_counter() - warm_start

    identity = dict(arm=a.arm, hyperparameters=hp, runner=runner.identity(), init=init_receipt,
                    data=dict(manifest_sha256=train.manifest_sha256, order_seed=a.order_seed,
                              order_sha256=train.order_sha256(), train_windows=train.count, dev_windows=dev.count),
                    schedule=dict(kind="linear warmup + cosine to zero over tokens", horizon_tokens=a.horizon_tokens,
                                  tokens_per_update=per_update, updates=updates, warmup_fraction=hp["warmup_fraction"],
                                  evaluated_at="update token midpoint"),
                    objective="mean CE over update targets + 0.02 x mean over passes of HF load-balancing aux",
                    device=device_identity(), argv=vars(a), torch=torch.__version__)
    write_json(out / "identity.json", identity)
    clock = Clock(DEVICE)
    losses = np.full(stop, np.nan, dtype=np.float32)
    norms = np.full(stop, np.nan, dtype=np.float32)
    seconds = np.zeros(stop, dtype=np.float64)
    evals, eval_seconds = [], 0.0
    # Everything that must match for an exact continuation of the same run.
    contract = dict(arm=a.arm, hyperparameters=hp, horizon_tokens=a.horizon_tokens, order_seed=a.order_seed,
                    order_sha256=train.order_sha256(), data_manifest_sha256=train.manifest_sha256,
                    init_sha256=init_receipt["sha256"], eval_every_tokens=a.eval_every_tokens,
                    eval_windows=a.eval_windows, runner=identity["runner"])
    start, resumed = 0, None
    if a.resume_uri:
        import shutil
        receipt = json.loads(Path(a.resume_uri + ".json").read_text())
        shutil.copyfile(a.resume_uri, out / "resume.pt")
        if ((out / "resume.pt").stat().st_size != receipt["bytes"]
                or protocol.digest(out / "resume.pt") != receipt["sha256"]):
            raise ValueError("local resume checkpoint size/hash mismatch")
        payload = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if payload["contract"] != contract:
            raise ValueError("resume checkpoint contract differs from this run")
        runner.load(payload["runner"])
        start = payload["completed"]
        if type(start) is not int or not 0 <= start < stop:
            raise ValueError("checkpoint cursor must precede the requested stop point")
        clock.total, clock.count = payload["clock_total"], start
        losses[:start], norms[:start], seconds[:start] = payload["loss"], payload["norm"], payload["seconds"]
        evals, eval_seconds = payload["evals"], payload["eval_seconds"]
        with (out / "evaluation.jsonl").open("w") as f:
            for row in evals:
                f.write(json.dumps(row) + "\n")
        (out / "resume.pt").unlink()
        resumed = dict(uri=a.resume_uri, sha256=receipt["sha256"], completed=start, clock_total=clock.total)
        torch.cuda.synchronize(DEVICE)

    def checkpoint(completed):
        t = time.perf_counter()
        payload = dict(schema="latch-tinymoe-smallbatch-checkpoint-v1", contract=contract, completed=completed,
                       clock_total=clock.total, loss=losses[:completed].copy(), norm=norms[:completed].copy(),
                       seconds=seconds[:completed].copy(), evals=evals, eval_seconds=eval_seconds,
                       runner=runner.state())
        tmp = out / "checkpoint.tmp.pt"
        torch.save(payload, tmp)
        os.replace(tmp, out / "checkpoint.pt")
        receipt = dict(bytes=(out / "checkpoint.pt").stat().st_size, sha256=protocol.digest(out / "checkpoint.pt"),
                       completed=completed, tokens=completed * per_update, seconds=time.perf_counter() - t)
        write_json(out / "checkpoint.pt.json", receipt)
        checkpoints.append(receipt)
    checkpoints = []

    def evaluation(completed, windows, split_name="dev"):
        nonlocal eval_seconds
        if windows <= 0:
            return None
        t = time.perf_counter()
        torch.cuda.synchronize(DEVICE)
        row = protocol.evaluate(config, runner.parameters(), dev, windows)
        eval_seconds += time.perf_counter() - t
        row.update(split=split_name, completed_updates=completed, tokens=completed * per_update,
                   training_seconds=clock.total, lr_next=protocol.lr_at(completed, a.learning_rate, sequences,
                                                                        a.horizon_tokens) if completed < updates else 0.0)
        evals.append(row)
        with (out / "evaluation.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def flush():
        np.savez(out / "updates.tmp.npz", loss=losses, norm=norms, seconds=seconds)
        os.replace(out / "updates.tmp.npz", out / "updates.npz")

    if not resumed:
        evaluation(0, a.eval_windows)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    loop_start = time.perf_counter()
    status = "completed"
    next_batch = train.batch(start * sequences, sequences, pin=True)
    for u in range(start, stop):
        x, y = next_batch
        lr = protocol.lr_at(u, a.learning_rate, sequences, a.horizon_tokens)
        with clock:
            loss, norm = runner.update(x, y, lr)
        seconds[u] = clock.last
        losses[u], norms[u] = float(loss), float(norm)
        if not (np.isfinite(losses[u]) and np.isfinite(norms[u])):
            status = f"nonfinite at update {u + 1}"
            break
        completed = u + 1
        if completed < stop:
            next_batch = train.batch(completed * sequences, sequences, pin=True)
        if (completed * per_update) % a.eval_every_tokens == 0:
            evaluation(completed, a.eval_windows)
            flush()
            if a.checkpoint_every_tokens and (completed * per_update) % a.checkpoint_every_tokens == 0 and completed < stop:
                checkpoint(completed)
    completed = int(np.isfinite(losses).sum())
    flush()
    loop_seconds = time.perf_counter() - loop_start
    peak = dict(allocated=torch.cuda.max_memory_allocated(DEVICE), reserved=torch.cuda.max_memory_reserved(DEVICE))
    final = evaluation(completed, a.final_eval_windows, "dev-final") if status == "completed" else None
    saved = None
    if a.save_final and status == "completed":
        t = time.perf_counter()
        path = out / "final-parameters.pt"
        torch.save({k: v.detach().to("cpu", copy=True) for k, v in runner.parameters().items()}, path)
        saved = dict(path=str(path), bytes=path.stat().st_size, sha256=protocol.digest(path),
                     seconds=time.perf_counter() - t)
    measured = seconds[:completed]
    result = dict(schema="latch-tinymoe-smallbatch-run-v1", status=status, arm=a.arm, hyperparameters=hp,
                  completed_updates=completed, tokens=completed * per_update,
                  training_seconds=float(measured.sum()), training_gpu_hours=float(measured.sum()) / 3600,
                  median_update_ms=float(np.median(measured) * 1000) if completed else None,
                  mean_update_ms=float(measured.mean() * 1000) if completed else None,
                  tokens_per_training_second=completed * per_update / float(measured.sum()) if completed else None,
                  final_dev=final, evaluations=evals, final_parameters=saved, resumed_from=resumed,
                  checkpoints=checkpoints,
                  separate_clocks=dict(setup_seconds=setup_seconds, warmup_seconds=warmup_seconds, warmup_update_seconds=warm,
                                       evaluation_seconds=eval_seconds, loop_wall_seconds=loop_seconds,
                                       process_seconds=time.perf_counter() - started),
                  peak_memory_bytes=peak, identity_sha256=protocol.digest(out / "identity.json"))
    write_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("status", "arm", "completed_updates", "training_seconds",
                                              "median_update_ms")} | dict(final_dev=final and final["nll"])), flush=True)
    if status != "completed":
        raise SystemExit(3)


def device_identity():
    import subprocess
    import torch
    props = torch.cuda.get_device_properties(DEVICE)
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=uuid,driver_version,clocks.max.sm", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        smi = repr(exc)
    return dict(name=props.name, total_memory=props.total_memory, multiprocessors=props.multi_processor_count,
                nvidia_smi=smi, cuda=torch.version.cuda)


if __name__ == "__main__":
    main()
