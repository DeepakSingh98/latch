# Provenance: copied from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch), original path examples/smelt_looped/example.py (blob 48ade3bdfd1f,
# sha256 65444ab8ae2b...; last changed at 13f223d4c).
# Changed: the donor inserted the repository root and its src/ directory into sys.path and imported
# research.smelt_latch.{looped,runners}; this copy imports the installed megakernel_lab package and the local
# smelt_looped / smelt_runners modules (this directory is sys.path[0] when the script runs). The docstring's
# build-module name is updated. Training loop, LR schedule, warmup-and-undo, timing, checkpoint format, dev
# evaluation and the bitwise compare are unchanged.
"""Runnable looped-MoE training example (SMELT-style loop of TinyMoE layers 4-9, r passes, shared weights).

The generated program is built on a CPU worker by the normal Latch API (``smelt_build.py`` here, i.e.
``megakernel_lab.tinymoe.build.prepare(rules="auto")`` around the looped capture). This script runs it on an H100,
or runs the same looped model conventionally (installed HF modules, one fullgraph torch.compile over every visit,
forward+backward replayed from one CUDA graph, fused AdamW).

    python examples/smelt_looped/example.py train --backend latch|conventional --model-dir M --data-dir D \
        [--artifact A] --batch-size 8 --sequence-length 512 --loop 3,6,2 --updates 32 --out OUT \
        [--save-at K] [--resume CKPT]
    python examples/smelt_looped/example.py compare LEFT.pt RIGHT.pt

Initial weights: the public TinyMoE-200m-2x16 checkpoint (continued training of the looped model). Objective:
mean CE + 0.02 x HF load-balancing aux over every MoE visit (one group per update at B <= 8). Optimizer:
participating AdamW (absent experts untouched) + global clip 1.0, beta1 0.9, eps 1e-6, beta2 0.999^(B/128),
learning rate linear warmup over --warmup updates to --learning-rate then constant (example schedule).
Data: ``latch-tinymoe data prepare`` token files (TokenBatches; contiguous fixed-length windows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time



def evaluate(runner, spec, model_dir, data_dir, batch, length, windows, device="cuda"):
    import torch
    from megakernel_lab.tinymoe.data import TokenBatches
    from megakernel_lab.tinymoe.model import evaluation_view
    from megakernel_lab.tinymoe.native import evaluator_config
    from smelt_looped import install_loop_experts, trunk_class
    dev = TokenBatches(data_dir, "validation", batch, length)
    model = evaluation_view(evaluator_config(model_dir), dict(runner.parameters()), device)
    install_loop_experts(model)
    trunk = trunk_class()(model, spec)
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(windows, dev.count)):
            x, y = dev.batch(i, device)
            final, _, _ = trunk(x)
            h, t = final.reshape(-1, final.shape[-1]), y.reshape(-1)
            total += float(torch.nn.functional.cross_entropy(model.lm_head(h).float(), t, reduction="sum"))
            count += t.numel()
    return total / count


def train(a):
    import torch
    from megakernel_lab.tinymoe.conventional import configure_execution
    from megakernel_lab.tinymoe.data import TokenBatches
    from megakernel_lab.tinymoe.model import load_model
    from smelt_runners import ConventionalLooped, LatchLooped, hyper, spec_of
    if a.backend == "conventional":
        configure_execution()
    spec = spec_of(a.loop)
    hp = hyper(a.learning_rate, a.batch_size)
    initial = {n: p.detach().clone() for n, p in load_model(a.model_dir, device="cpu").named_parameters()}
    if a.backend == "latch":
        runner = LatchLooped(artifact=a.artifact, hp=hp, initial=initial, spec=spec)
    else:
        runner = ConventionalLooped(model_dir=a.model_dir, spec=spec, hp=hp, initial=initial,
                                    mode="compiled-trunk", chunk=a.batch_size * a.sequence_length, graph=True)
    data = TokenBatches(a.data_dir, "train", a.batch_size, a.sequence_length)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    lr = lambda u: a.learning_rate * min(1.0, (u + 1) / a.warmup)
    start = 0
    if a.resume:
        payload = torch.load(a.resume, map_location="cpu", weights_only=False)
        runner.load(payload["runner"])
        start = payload["completed"]
    elif a.warm:  # execution warmup (first launch / compilation / graph capture), then undone
        x, y = data.batch(0)
        for _ in range(a.warm):
            runner.update(x, y, lr(0))
        runner.restore()
        runner.reset_optimizer()
    rows, seconds = [], 0.0
    for u in range(start, a.updates):
        x, y = data.batch(u)
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss, norm = runner.update(x, y, lr(u))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        seconds += dt
        rows.append(dict(update=u + 1, loss=float(loss), norm=float(norm), ms=dt * 1e3))
        print(json.dumps(rows[-1]), flush=True)
        if a.save_at and u + 1 == a.save_at:
            break
    completed = rows[-1]["update"] if rows else start
    torch.save(dict(runner=runner.state(), completed=completed), out / f"checkpoint-{completed:06d}.pt")
    result = dict(backend=a.backend, loop=spec.asdict(), completed=completed, update_seconds=seconds,
                  median_ms=sorted(r["ms"] for r in rows)[len(rows) // 2] if rows else None,
                  dev_nll=evaluate(runner, spec, a.model_dir, a.data_dir, a.batch_size, a.sequence_length,
                                   a.eval_windows),
                  identity=runner.identity(), rows=rows)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: result[k] for k in ("backend", "completed", "update_seconds", "median_ms", "dev_nll")}))


def compare(a):
    import torch
    left, right = (torch.load(p, map_location="cpu", weights_only=False)["runner"] for p in a.paths)
    report, equal = {}, True
    for kind in ("parameters", "optimizer", "model_state"):
        if kind not in left or kind not in right:
            continue
        names = sorted(set(left[kind]) & set(right[kind]))
        bad = [n for n in names if not torch.equal(left[kind][n].reshape(-1).view(torch.uint8),
                                                   right[kind][n].reshape(-1).view(torch.uint8))]
        report[kind] = dict(tensors=len(names), mismatched=bad[:10])
        equal = equal and not bad and set(left[kind]) == set(right[kind])
    print(json.dumps(dict(bitwise=equal, **report)))
    raise SystemExit(0 if equal else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "compare"))
    p.add_argument("paths", nargs="*")
    p.add_argument("--backend", choices=("latch", "conventional"), default="latch")
    p.add_argument("--model-dir")
    p.add_argument("--data-dir")
    p.add_argument("--artifact")
    p.add_argument("--loop", default="3,6,2")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--updates", type=int, default=32)
    p.add_argument("--warm", type=int, default=2)
    p.add_argument("--eval-windows", type=int, default=8)
    p.add_argument("--save-at", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--out")
    a = p.parse_args()
    (train if a.command == "train" else compare)(a)


if __name__ == "__main__":
    main()
