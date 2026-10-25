"""Same-GPU alternating throughput comparison of the arms (GPU worker).

Each round runs every arm in a fresh process through the unchanged train.py path
(random init, same windows, same token schedule), over the SAME number of tokens per
arm, with evaluations disabled. Rounds alternate the arm order. Reports per-arm
median complete-update seconds (warmup excluded by train.py, first `--skip` measured
updates excluded here) and training seconds per token.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from . import protocol
from .protocol import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", default=["A", "B1", "B2"])
    p.add_argument("--artifact", action="append", default=[], help="ARM=DIR for Latch arms")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tokens", type=int, default=128 * 1024 * 24)
    p.add_argument("--horizon-tokens", type=int, default=128 * 1024 * 1000)
    p.add_argument("--skip-tokens", type=int, default=128 * 1024 * 4)
    p.add_argument("--learning-rate", action="append", default=[], help="ARM=LR (default: reference sqrt scaling)")
    p.add_argument("--order-seed", type=int, default=1729)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = dict(item.split("=", 1) for item in a.artifact)
    rates = {k: float(v) for k, v in (item.split("=", 1) for item in a.learning_rate)}
    rows = {arm: [] for arm in a.arms}
    for r in range(a.rounds):
        order = a.arms if r % 2 == 0 else list(reversed(a.arms))
        for arm in order:
            sequences = protocol.ARMS[arm]["sequences"]
            lr = rates.get(arm, protocol.REFERENCE["learning_rate"] * (sequences / 128) ** 0.5)
            run = out / f"round{r}-{arm}"
            argv = [sys.executable, "-B", "-m", "megakernel_lab.tinymoe.learning.train", "--arm", arm,
                    "--model-dir", a.model_dir, "--data-dir", a.data_dir, "--init", a.init, "--out", str(run),
                    "--learning-rate", str(lr), "--horizon-tokens", str(a.horizon_tokens),
                    "--stop-tokens", str(a.tokens), "--order-seed", str(a.order_seed),
                    "--eval-every-tokens", str(a.horizon_tokens), "--eval-windows", "0", "--final-eval-windows", "0"]
            if protocol.ARMS[arm]["backend"] == "latch":
                argv += ["--artifact", artifacts[arm]]
            subprocess.run(argv, check=True)
            import numpy as np
            seconds = np.load(run / "updates.npz")["seconds"]
            skip = a.skip_tokens // protocol.tokens_per_update(sequences)
            measured = seconds[skip:]
            result = json.loads((run / "result.json").read_text())
            rows[arm].append(dict(round=r, position=order.index(arm), updates=len(measured),
                                  median_ms=float(np.median(measured) * 1000), mean_ms=float(measured.mean() * 1000),
                                  seconds_per_million_tokens=float(measured.sum()) / (len(measured) * sequences * 1024) * 1e6,
                                  peak_memory_bytes=result["peak_memory_bytes"], device=json.loads(
                                      (run / "identity.json").read_text())["device"]))
    summary = {}
    for arm, values in rows.items():
        sequences = protocol.ARMS[arm]["sequences"]
        median = statistics.median(v["median_ms"] for v in values)
        summary[arm] = dict(sequences_per_update=sequences, tokens_per_update=sequences * 1024,
                            median_of_round_medians_ms=median, round_medians_ms=[v["median_ms"] for v in values],
                            seconds_per_million_tokens=statistics.median(v["seconds_per_million_tokens"] for v in values))
    if "A" in summary:
        base = summary["A"]["seconds_per_million_tokens"]
        for arm, value in summary.items():
            value["time_per_token_vs_A"] = value["seconds_per_million_tokens"] / base
    write_json(out / "paired.json", dict(schema="latch-tinymoe-smallbatch-paired-v1", rounds=a.rounds,
                                         tokens_per_arm_round=a.tokens, rows=rows, summary=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
