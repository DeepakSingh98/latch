"""Runnable installed and bounded optimized conventional TinyMoE baselines."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

import torch

from .data import TokenBatches, digest
from .model import (Recipe, features, installed_objective, load_model, make_optimizer,
                    optimized_backward, optimizer_skip_unused, parameter_inventory,
                    use_grouped_experts, evaluation_view)
from .pins import public_identity
from .weight_ema import WeightEMA

BACKENDS = ("installed", "optimized", "grouped", "compiled-grouped")


def configure_execution():
    """Set deterministic CUDA arithmetic before constructing GPU state."""
    workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (None,":4096:8",":16:8"):
        raise ValueError("unsupported deterministic CUBLAS_WORKSPACE_CONFIG: "+workspace)
    if torch.cuda.is_initialized() and workspace is None:
        raise RuntimeError("configure deterministic TinyMoE execution before CUDA initialization")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return dict(deterministic_algorithms=True,cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                allow_tf32=False)


def source_identity():
    """Runtime bytes bind native resume, independently of a moving Git checkout."""
    from . import weight_ema
    paths = [Path(__file__).with_name(n) for n in ("conventional.py", "model.py", "data.py", "pins.py", "observation.py")]
    paths.append(Path(weight_ema.__file__))
    return {p.name: digest(p) for p in paths}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def precision_context(device, bf16=True):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16) if bf16 else nullcontext()


def evaluation_policy(windows,bf16=True):
    return dict(implementation="installed Transformers expert loops and SDPA; shared zero-copy weight view",
                precision="FP32 masters, BF16 autocast, FP32 cross entropy" if bf16 else "FP32",
                readout="final only",vocabulary_chunk_tokens=512,split="validation",start_window=0,windows=windows)


def prepare_backend(model, backend, precision=None):
    if backend not in BACKENDS:
        raise ValueError("unknown conventional backend")
    compiled = []
    if backend in ("grouped", "compiled-grouped"):
        use_grouped_experts(model)
    if precision not in (None, "release"):
        # LATCH-TRAIN-PRECISION: the policy's forward products (before compilation of the layer regions).
        from .model import apply_precision
        apply_precision(model, precision)
    if backend == "compiled-grouped":
        for index, layer in enumerate(model.model.layers):
            layer.block_sparse_moe.forward = torch.compile(layer.block_sparse_moe.forward, fullgraph=True)
            layer.self_attn.forward = torch.compile(layer.self_attn.forward, fullgraph=True)
            compiled += [f"layer{index}.moe", f"layer{index}.attention"]
    return compiled


def backward(model, backend, x, y, weight, chunk_tokens, precision=None):
    if backend == "installed":
        if precision not in (None, "release"):
            raise ValueError("training precision policies apply to the optimized conventional backends")
        loss, metrics = installed_objective(model, x, y, weight)
        loss.backward()
    else:
        loss, metrics = optimized_backward(model, x, y, weight, chunk_tokens=chunk_tokens,
                                           **({"precision": precision} if precision is not None else {}))
    optimizer_skip_unused(model)
    return loss.detach(), metrics


def update(model, optimizer, recipe, index, x, y, *, backend, chunk_tokens=512, ema=None, bf16=True, capture_gradients=None,
           precision=None):
    optimizer.zero_grad(set_to_none=True)
    lr, weight = recipe.lr(index), recipe.supervision_weight(index)
    for group in optimizer.param_groups:
        group["lr"] = lr
    with precision_context(x.device, bf16):
        loss, metrics = backward(model, backend, x, y, weight, chunk_tokens, precision)
    if capture_gradients is not None:
        capture_gradients()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip_norm, foreach=True)
    optimizer.step()
    if ema is not None:
        ema.update(index + 1)
    return dict(loss=loss, norm=norm, metrics=metrics, learning_rate=lr, supervision_weight=weight)


def scalar_metrics(result):
    return dict(objective=float(result["loss"]), gradient_norm=float(result["norm"]),
        final_ce=float(result["metrics"]["final_ce"]), router_aux=float(result["metrics"]["router_aux"]),
        exit_ce=[float(v) for v in result["metrics"]["exit_ce"]],
        learning_rate=result["learning_rate"], supervision_weight=result["supervision_weight"])


def save_checkpoint(path, model, optimizer, ema, completed, contract):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema="latch-tinymoe-conventional-checkpoint-v1", completed_updates=completed,
        contract=contract, model=model.state_dict(), optimizer=optimizer.state_dict(),
        ema=None if ema is None else ema.state_dict(), cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)
    receipt = dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path), completed_updates=completed)
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def restore_checkpoint(path, model, optimizer, ema, contract):
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if path.stat().st_size != receipt["bytes"] or digest(path) != receipt["sha256"]:
        raise ValueError("checkpoint hash/size mismatch")
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if payload["contract"] != contract:
        raise ValueError("checkpoint contract differs; exact resume requires same backend/config/data")
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
    completed = payload["completed_updates"]
    if ema is not None:
        ema.load_state_dict(payload["ema"], completed)
    elif payload["ema"] is not None:
        raise ValueError("EMA state supplied to unchanged recipe")
    torch.set_rng_state(payload["cpu_rng"].cpu())
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all([v.cpu() for v in payload["cuda_rng"]])
    return completed


@torch.no_grad()
def evaluate(model, batches, *, windows=8, bf16=True):
    previous = model.training
    model.eval()
    losses = []
    device = next(model.parameters()).device
    for i in range(min(windows, batches.count)):
        x, y = batches.batch(i, device)
        with precision_context(device, bf16):
            final, _, _ = features(model, x)
            # Final readout NLL only: router regularizer and intermediate losses
            # are training objectives, not reported held-out perplexity.
            chunks = []
            h, target = final.reshape(-1, final.shape[-1]), y.reshape(-1)
            for start in range(0, h.shape[0], 512):
                chunks.append(torch.nn.functional.cross_entropy(model.lm_head(h[start:start+512]).float(),
                    target[start:start+512], reduction="sum"))
            losses.append(float(sum(chunks) / target.numel()))
    model.train(previous)
    return dict(final_nll=statistics.mean(losses), windows=len(losses), targets=len(losses)*batches.width)


def train(a):
    execution_policy=configure_execution()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setup_start = time.perf_counter()
    recipe = Recipe(name=a.recipe, horizon=a.horizon, learning_rate=a.learning_rate, weight_decay=a.weight_decay)
    model = load_model(a.model_dir, device=a.device, attention=a.attention)
    compiled = prepare_backend(model, a.backend)
    optimizer = make_optimizer(model, recipe, fused=torch.device(a.device).type == "cuda")
    ema = WeightEMA(model, a.horizon) if a.recipe == "supervision_ema" else None
    training = TokenBatches(a.data_dir, "train", a.batch_size, a.sequence_length)
    validation = TokenBatches(a.data_dir, "validation", a.batch_size, a.sequence_length)
    contract = dict(**public_identity(), recipe=asdict(recipe), backend=a.backend, precision="fp32-master-bf16-autocast" if not a.fp32 else "fp32",
        batch_size=a.batch_size, sequence_length=a.sequence_length, chunk_tokens=a.chunk_tokens,
        data_manifest_sha256=digest(Path(a.data_dir)/"manifest.json"), attention=a.attention,
        torch_version=str(torch.__version__), seed=a.seed, optimizer="AdamW-fused" if torch.device(a.device).type=="cuda" else "AdamW",
        expert_optimizer_policy="skip absent gradients, including moments/step/decay", threads=a.threads,
        execution_policy=execution_policy,
        evaluation=evaluation_policy(a.eval_windows,not a.fp32),
        source_sha256=source_identity())
    start = restore_checkpoint(a.resume, model, optimizer, ema, contract) if a.resume else 0
    if ema is not None and not a.resume:
        ema.update(0)
    evaluator=evaluation_view(model.config,dict(model.named_parameters()),a.device) if a.eval_every else None
    synchronize(a.device)
    setup_seconds = time.perf_counter() - setup_start
    (out/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    rows, checks = [], []
    evaluation_start=time.perf_counter()
    if a.eval_every:
        initial=dict(completed_update=start,raw=evaluate(evaluator,validation,windows=a.eval_windows,bf16=not a.fp32))
        if ema is not None and ema.ready:
            with ema.apply():initial["ema"]=evaluate(evaluator,validation,windows=a.eval_windows,bf16=not a.fp32)
        with (out/"evaluation.jsonl").open("a") as f:f.write(json.dumps(initial)+"\n")
    initial_evaluation_seconds=time.perf_counter()-evaluation_start
    if torch.device(a.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(a.device)
    total_start = time.perf_counter()
    model.train()
    with (out/"updates.jsonl").open("a") as log:
        for index in range(start, min(a.updates, a.horizon)):
            x, y = training.batch(index, a.device)
            observation=None
            if index+1 in a.observe_at:
                from .observation import Observation
                observation=Observation(model,optimizer,ema,recipe,index,x,y)
            synchronize(a.device)
            tick = time.perf_counter()
            result = update(model, optimizer, recipe, index, x, y, backend=a.backend,
                chunk_tokens=a.chunk_tokens, ema=ema, bf16=not a.fp32,
                capture_gradients=None if observation is None else observation.capture_gradients)
            synchronize(a.device)
            seconds = time.perf_counter()-tick
            row = dict(completed_update=index+1, seconds=seconds, observation=observation is not None, **scalar_metrics(result))
            if not all(torch.isfinite(result[k]).item() for k in ("loss", "norm")):
                raise FloatingPointError(f"nonfinite training at {index+1}")
            rows.append(row); log.write(json.dumps(row)+"\n"); log.flush()
            if observation is not None:
                observation.save(out/f"observation-{index+1:06d}.pt",result)
                del observation
            if a.eval_every and (index+1) % a.eval_every == 0:
                val = dict(completed_update=index+1, raw=evaluate(evaluator, validation, windows=a.eval_windows, bf16=not a.fp32))
                if ema is not None and ema.ready:
                    with ema.apply():
                        val["ema"] = evaluate(evaluator, validation, windows=a.eval_windows, bf16=not a.fp32)
                with (out/"evaluation.jsonl").open("a") as f:
                    f.write(json.dumps(val)+"\n")
            if (a.checkpoint_every and (index+1) % a.checkpoint_every == 0) or index+1 in a.checkpoint_at:
                checks.append(save_checkpoint(out/f"checkpoint-{index+1:06d}.pt", model, optimizer, ema, index+1, contract))
    completed = rows[-1]["completed_update"] if rows else start
    if a.save_final:
        checks.append(save_checkpoint(out/f"checkpoint-{completed:06d}.pt", model, optimizer, ema, completed, contract))
    phases = {}
    for phase, predicate in (("supervision", lambda r:r["supervision_weight"]>0),
                             ("plain", lambda r:r["supervision_weight"]==0)):
        selected = [r["seconds"] for r in rows[a.timing_warmup:] if predicate(r) and not r["observation"]]
        if selected:
            phases[phase] = dict(updates=len(selected), median_ms=1000*statistics.median(selected), mean_ms=1000*statistics.mean(selected))
    report = dict(schema="latch-tinymoe-conventional-run-v1", contract=contract,
        parameter_count=sum(p.numel() for p in model.parameters()), setup_seconds=setup_seconds,
        initial_evaluation_seconds=initial_evaluation_seconds,
        training_process_seconds=time.perf_counter()-total_start, completed_updates=completed,
        phases=phases, compiled_regions=compiled, checkpoints=checks,
        gpu=torch.cuda.get_device_name(a.device) if torch.device(a.device).type=="cuda" else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device) if torch.device(a.device).type=="cuda" else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(a.device) if torch.device(a.device).type=="cuda" else None,
        timing_scope="synchronized complete update including routing, backward, clipping, optimizer and active EMA; excluding batch staging, evaluation/checkpoint and scalar logging; observed updates excluded from phase timing summaries; forward compilation may occur in separately measured initial evaluation, backward compilation is included in first update")
    (out/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",required=True); p.add_argument("--data-dir",required=True); p.add_argument("--out",required=True)
    p.add_argument("--backend",choices=BACKENDS,default="installed")
    p.add_argument("--recipe",choices=("unchanged","supervision_ema"),default="unchanged")
    p.add_argument("--device",default="cuda"); p.add_argument("--attention",default="sdpa")
    p.add_argument("--batch-size",type=int,default=2); p.add_argument("--sequence-length",type=int,default=512)
    p.add_argument("--horizon",type=int,default=400); p.add_argument("--updates",type=int,default=400)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=.01)
    p.add_argument("--chunk-tokens",type=int,default=512); p.add_argument("--threads",type=int,default=1)
    p.add_argument("--seed",type=int,default=1729); p.add_argument("--fp32",action="store_true")
    p.add_argument("--resume"); p.add_argument("--save-final",action="store_true")
    p.add_argument("--checkpoint-every",type=int,default=0); p.add_argument("--eval-every",type=int,default=50)
    p.add_argument("--checkpoint-at",type=int,nargs="*",default=[])
    p.add_argument("--observe-at",type=int,nargs="*",default=[])
    p.add_argument("--eval-windows",type=int,default=8); p.add_argument("--timing-warmup",type=int,default=4)
    train(p.parse_args())


if __name__ == "__main__":
    main()
