# Provenance: adapted from research/latch_selective_cadence/branch.py at latch commit 6fcad414ad86
# (git blob e28127c994cbe46165b7e271657c3e87fe7c168a; the branch runner behind SELECTIVE-10's restart_job.sh and
# equal_job.sh), with kernels() copied from research/latch_selective_10/cbench.py (blob
# 3b0b4ef2bb75e752c2743cccfe3134ee462b4011).
# Kept unchanged: the study Clock around runner.update (synchronize; LR/slow-scalar fills, token H2D, one program
# launch; synchronize), the per-microbatch rows (loss, fast/slow norm and clip), evaluation with the installed study
# evaluator, the mid-window checkpoint payload (runner state + rows + clock + evaluations), resume, checksums and
# the result fields compare_restart reads.
# Removed: the parent study checkpoint (receipt, B2 legacy-1.05B contract, order/manifest equality with the parent),
# the arm-A1 path and its bitwise parent replay. Replaced by: a start state from the public pretrained snapshot or
# an init-state file (selective_start.py), --start-update (schedule position, multiple of 16) with data windows
# consumed from the start of the PCG64(--order-seed) order, imports from the installed wheel.
# Added: one printed JSON row per microbatch (loss, clip norms, phase ms, program launches), launch counts from
# Runtime.launch_count, optional torch.profiler device-activity counts (--kernels), derived-copy (dcast) summary.
"""Selective-update (A4) complete training on the accumulate/apply Latch program pair (one H100).

    python selective_train.py --accumulate DIR --apply DIR --model-dir M --data-dir W --microbatches 64 \
        --out OUT [--save-at 24] [--resume OUT_OF_A_SAVED_RUN/checkpoint.pt] [--init INIT.pt] [--kernels]

Every microbatch (2 x 1024 tokens) runs the complete forward and backward pass and updates the shared weights
(attention, norms, routers, embedding, readout) with the study AdamW and its own clip. Experts accumulate their raw
FP32 gradients; at u mod 16 = 15 the apply program adds that microbatch's gradient, takes the window mean, clips it
by its own norm and applies one AdamW update (kappa x mean LR, beta2_2^16, counter +16; experts absent from the
whole window untouched). Clock = the study's (synchronized wall seconds of each microbatch's complete update:
scalar fills, token H2D, one generated launch); evaluation, checkpoint writes and host batch loading are excluded
and reported separately. Exact restart: --save-at K writes OUT/checkpoint.pt after K microbatches (any position in
a window) and continues; --stop-at K stops there; --resume CKPT continues a saved run in a new process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from megakernel_lab.tinymoe.learning import protocol
from megakernel_lab.tinymoe.learning.protocol import Clock, Windows, digest, write_json

import selective_runners as runners
import selective_start
from selective_runners import DEVICE, SEQUENCES


def launches(runner):
    """Generated-program launches issued by both phase runtimes so far (Runtime.launch_count)."""
    return sum(int(p.runtime.launch_count) for p in runner.programs.values())


def kernels(runner, batch, u):
    """Device activities of one microbatch launch (staging included). Copied from cbench.kernels."""
    import torch
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        runner.update(*batch, u)
        torch.cuda.synchronize()
    counts = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            key = "memcpy" if "Memcpy" in event.name else "memset" if "Memset" in event.name else "kernel"
            counts[key] = counts.get(key, 0) + 1
            counts.setdefault("names", {})
            counts["names"][event.name[:80]] = counts["names"].get(event.name[:80], 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accumulate", required=True, help="accumulate-phase artifact (selective_build.py --phase accumulate)")
    p.add_argument("--apply", required=True, help="apply-phase artifact (selective_build.py --phase apply)")
    p.add_argument("--kappa", type=float, default=4.0, help="slow-group LR multiplier (study-selected 4)")
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="fast group's LR x scale with weight decay / scale (study diagnostic A4m used 0.25)")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True, help="1025-token windows (megakernel_lab.tinymoe.learning quick-windows)")
    p.add_argument("--init", help="init-state .pt (random initialization); default: the public pretrained snapshot")
    p.add_argument("--start-update", type=int, default=0, help="schedule position u0 of the first microbatch")
    p.add_argument("--order-seed", type=int, default=1729)
    p.add_argument("--microbatches", type=int, required=True)
    p.add_argument("--eval-every", type=int, default=32)
    p.add_argument("--eval-windows", type=int, default=16)
    p.add_argument("--final-eval-windows", type=int, default=16)
    p.add_argument("--save-at", type=int, default=0)
    p.add_argument("--stop-at", type=int, default=0)
    p.add_argument("--resume")
    p.add_argument("--save-final", action="store_true")
    p.add_argument("--kernels", action="store_true",
                   help="after the run: torch.profiler device activities of one accumulate and one apply microbatch")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    started = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    hp = runners.hyperparameters()
    u0 = int(a.start_update)
    if u0 % 16 or a.eval_every % 16 or a.microbatches % 16:
        raise ValueError("start update, evaluation cadence and length must be window aligned")
    if a.kappa <= 0:
        raise ValueError("the cadence arm needs kappa > 0")
    runner = runners.Cadence(a.accumulate, a.apply, arm="A4", kappa=a.kappa, lr_scale=a.lr_scale)
    if runner.slow_group != "experts":
        raise ValueError("this example is arm A4 (slow group 'experts'); the artifacts were built for "
                         + str(runner.slow_group))
    initial, start_identity = selective_start.initial_parameters(a.model_dir, a.init)
    parent = selective_start.start_payload(runner, initial)
    del initial
    runner.load_parent(parent)
    train = Windows(a.data_dir, "train", order_seed=a.order_seed)
    dev = Windows(a.data_dir, "validation")
    from megakernel_lab.tinymoe.native import evaluator_config
    config = evaluator_config(a.model_dir)
    n = a.microbatches
    rows = {k: np.full(n, np.nan, dtype=np.float32) for k in ("loss", "fast_norm", "fast_clip", "slow_norm", "slow_clip")}
    rows["seconds"] = np.zeros(n, dtype=np.float64)
    rows["apply"] = np.zeros(n, dtype=np.int8)
    rows["launches"] = np.zeros(n, dtype=np.int32)
    clock = Clock(DEVICE)
    evals, eval_seconds, start = [], 0.0, 0
    identity = dict(schema="latch-selective-update-example-run-v1", arm="A4", runner=runner.identity,
                    hyperparameters=hp, start=start_identity, start_update=u0, order_seed=a.order_seed,
                    data=dict(manifest_sha256=train.manifest_sha256, order_sha256=train.order_sha256()),
                    schedule=dict(kind="legacy", peak=runners.PEAK, horizon_tokens=runners.HORIZON),
                    microbatches=n, argv=vars(a), torch=torch.__version__)
    write_json(out / "identity.json", identity)
    run_contract = dict(arm="A4", kappa=a.kappa, lr_scale=a.lr_scale, slow_group=runner.slow_group,
                        start=start_identity, start_update=u0, order_seed=a.order_seed,
                        data_manifest_sha256=train.manifest_sha256, microbatches=n, eval_every=a.eval_every,
                        eval_windows=a.eval_windows, artifacts=runner.identity["artifacts"])
    resumed = None
    if a.resume:
        payload = torch.load(a.resume, map_location="cpu", weights_only=False)
        if payload["contract"] != run_contract:
            raise ValueError("resume checkpoint contract differs")
        # Host writes of expert parameters: with dcast the next launch recasts the derived BF16 copy (Runtime.derived).
        runner.load(payload["runner"])
        start = payload["completed"]
        for k, v in payload["rows"].items():
            rows[k][:start] = v
        clock.total, clock.count = payload["clock_total"], start
        evals, eval_seconds = payload["evals"], payload["eval_seconds"]
        resumed = dict(path=a.resume, sha256=digest(a.resume), completed=start)
        torch.cuda.synchronize(DEVICE)

    def evaluation(completed, windows, split="dev"):
        nonlocal eval_seconds
        t = time.perf_counter()
        torch.cuda.synchronize(DEVICE)
        row = protocol.evaluate(config, runner.parameters, dev, windows)
        eval_seconds += time.perf_counter() - t
        row.update(split=split, microbatches=completed, update=u0 + completed,
                   tokens=(u0 + completed) * SEQUENCES * protocol.SEQUENCE,
                   branch_tokens=completed * SEQUENCES * protocol.SEQUENCE, training_seconds=clock.total)
        evals.append(row)
        with (out / "evaluation.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(dict(event="evaluation", **row)), flush=True)
        return row

    def save(completed):
        t = time.perf_counter()
        payload = dict(schema="latch-selective-update-example-checkpoint-v1", contract=run_contract,
                       completed=completed, clock_total=clock.total, evals=evals, eval_seconds=eval_seconds,
                       rows={k: v[:completed].copy() for k, v in rows.items()}, runner=runner.state())
        torch.save(payload, out / "checkpoint.tmp.pt")
        os.replace(out / "checkpoint.tmp.pt", out / "checkpoint.pt")
        write_json(out / "checkpoint.pt.json", dict(bytes=(out / "checkpoint.pt").stat().st_size,
                                                    sha256=digest(out / "checkpoint.pt"), completed=completed,
                                                    seconds=time.perf_counter() - t))

    if not resumed:
        evaluation(0, a.eval_windows)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    stop = a.stop_at or n
    loop_start = time.perf_counter()
    status = "completed" if stop == n else "stopped"
    completed = start
    batch = train.batch(start * SEQUENCES, SEQUENCES, pin=True)
    for i in range(start, stop):
        u = u0 + i
        x, y = batch
        before = launches(runner)
        with clock:
            loss = runner.update(x, y, u)
        rows["seconds"][i] = clock.last
        rows["launches"][i] = launches(runner) - before
        rows["apply"][i] = runner.phase == "apply"
        rows["loss"][i] = float(loss)
        for k, v in runner.diagnostics().items():
            if k.startswith("fast") or runner.phase == "apply":
                rows[k][i] = float(v)
        print(json.dumps(dict(microbatch=i + 1, update=u, phase=runner.phase, loss=float(rows["loss"][i]),
                              fast_norm=float(rows["fast_norm"][i]), fast_clip=float(rows["fast_clip"][i]),
                              **({"slow_norm": float(rows["slow_norm"][i]), "slow_clip": float(rows["slow_clip"][i])}
                                 if runner.phase == "apply" else {}),
                              ms=rows["seconds"][i] * 1000, launches=int(rows["launches"][i]))), flush=True)
        if not np.isfinite(rows["loss"][i]):
            status = f"nonfinite at microbatch {i + 1}"
            break
        completed = i + 1
        if completed < stop:
            batch = train.batch(completed * SEQUENCES, SEQUENCES, pin=True)
        if completed % a.eval_every == 0 and completed < n:
            evaluation(completed, a.eval_windows)
        if a.save_at and completed == a.save_at:
            save(completed)
    completed = int(np.isfinite(rows["loss"]).sum())
    loop_seconds = time.perf_counter() - loop_start
    peak = dict(allocated=torch.cuda.max_memory_allocated(DEVICE), reserved=torch.cuda.max_memory_reserved(DEVICE),
                persistent_bytes={k: sum(v.numel() * v.element_size() for v in getattr(runner, k).values())
                                  for k in ("parameters", "optimizer_state", "model_state")},
                derived_bytes=sum(v.numel() * v.element_size() for v in runner.program.runtime.derived.values()),
                arena_bytes=int(runner.arena.numel()) * 4)  # the pair shares one arena
    final = None
    if status == "completed":
        evaluation(completed, a.eval_windows)
        final = evaluation(completed, a.final_eval_windows, "dev-final")
    np.savez(out / "updates.npz", **rows)
    checks = runner.checksums()
    seconds = rows["seconds"][:completed]
    # LATCH-V0-NEXT: the first launch of a process also compiles the native launch-validation guard once (cached
    # in ~/.cache/latch_binding_guard), which can take seconds on a fresh host. Phase means therefore exclude this
    # process's first microbatch; it is reported separately as first_launch_ms.
    steady = np.arange(completed) >= 1
    phases = {name: dict(count=int((mask & steady).sum()),
                         median_ms=float(np.median(seconds[mask & steady]) * 1000) if (mask & steady).any() else None,
                         mean_ms=float(seconds[mask & steady].mean() * 1000) if (mask & steady).any() else None)
              for name, mask in (("accumulate", rows["apply"][:completed] == 0),
                                 ("apply", rows["apply"][:completed] == 1))}
    first_launch_ms = float(seconds[0] * 1000) if completed else None
    cycles = completed // 16 * 16   # u0 is window aligned, so the first `cycles` rows are complete 16-cycles
    cycle_mean_incl_first = float(seconds[:cycles].mean() * 1000) if cycles else None
    acc_ms, app_ms = phases["accumulate"]["mean_ms"], phases["apply"]["mean_ms"]
    cycle_mean = (15 * acc_ms + app_ms) / 16 if acc_ms is not None and app_ms is not None else cycle_mean_incl_first
    own = rows["launches"][start:completed]
    clip = dict(fast_active=int((rows["fast_clip"][:completed] < 1).sum()),
                fast_rows=int(np.isfinite(rows["fast_clip"][:completed]).sum()),
                slow_active=int((rows["slow_clip"][:completed] < 1).sum()),
                slow_rows=int(np.isfinite(rows["slow_clip"][:completed]).sum()))
    saved = None
    if a.save_final and status == "completed":
        path = out / "final-parameters.pt"
        torch.save({k: v.detach().to("cpu", copy=True) for k, v in runner.parameters.items()}, path)
        saved = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
    runtimes = {k: dict(launch_validation=prog.runtime.launch_validation,
                        launch_validation_path=getattr(prog.runtime, "launch_validation_path", None),
                        derived_keys=sorted(prog.runtime.derived), blocks=int(prog.runtime.blocks))
                for k, prog in runner.programs.items()}
    result = dict(schema="latch-selective-update-example-result-v1", status=status, arm="A4", kappa=a.kappa,
                  lr_scale=a.lr_scale, slow_group=runner.slow_group, start=start_identity, start_update=u0,
                  microbatches=completed, branch_tokens=completed * SEQUENCES * protocol.SEQUENCE,
                  training_seconds=float(seconds.sum()), mean_microbatch_ms=float(seconds.mean() * 1000),
                  median_microbatch_ms=float(np.median(seconds) * 1000), phases=phases,
                  complete_cycle_mean_ms=cycle_mean, first_launch_ms=first_launch_ms,
                  complete_cycle_mean_ms_incl_first_launch=cycle_mean_incl_first,
                  complete_cycle_note="single run, study clock, (15 x accumulate mean + apply mean) / 16 over this "
                                      "process's microbatches after its first launch; NOT the alternating multi-state "
                                      "cbench metric of LATCH-SELECTIVE-10",
                  launches_per_microbatch_this_process=sorted({int(v) for v in own}),
                  clip_activity=clip, final_dev=final, evaluations=evals, resumed_from=resumed,
                  final_checksums_sha256=hashlib.sha256(json.dumps(checks, sort_keys=True).encode()).hexdigest(),
                  final_parameters=saved, peak_memory=peak, runtimes=runtimes,
                  separate_clocks=dict(evaluation_seconds=eval_seconds, loop_wall_seconds=loop_seconds,
                                       process_seconds=time.perf_counter() - started),
                  device=torch.cuda.get_device_name(DEVICE), identity_sha256=digest(out / "identity.json"))
    write_json(out / "checksums.json", checks)
    write_json(out / "result.json", result)
    print(json.dumps(dict(status=status, arm="A4", kappa=a.kappa, microbatches=completed,
                          training_seconds=result["training_seconds"], accumulate_ms=phases["accumulate"]["mean_ms"],
                          apply_ms=phases["apply"]["mean_ms"], complete_cycle_mean_ms=cycle_mean,
                          first_launch_ms=first_launch_ms,
                          launches_per_microbatch=result["launches_per_microbatch_this_process"],
                          final_dev=final and final["nll"])), flush=True)
    if a.kernels and status == "completed":
        # Separate diagnostic pass (outputs above are already written): restore the start state, run 14 microbatches,
        # profile microbatch 15 (accumulate) and 16 (apply), as cbench --kernels.
        runner.load_parent(parent)
        batches = [train.batch(k * SEQUENCES, SEQUENCES, pin=True) for k in range(16)]
        for k in range(14):
            runner.update(*batches[k], u0 + k)
        report = dict(accumulate=kernels(runner, batches[14], u0 + 14), apply=kernels(runner, batches[15], u0 + 15),
                      note="torch.profiler CUDA activities of one microbatch including its scalar fills and token H2D")
        write_json(out / "kernels.json", report)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "names"} for k, v in report.items()
                          if isinstance(v, dict)}), flush=True)
    if status not in ("completed", "stopped"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
