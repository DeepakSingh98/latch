"""Reduce every frozen fresh-seed confirmation target without selecting a favorable seed.

Pass each completed run as ARM:SEED=directory. Missing and incomplete runs remain
visible. This reducer never selects learning rates or changes target thresholds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import crossing, load_run
from .protocol import digest, write_json

SEEDS = (1730, 1731, 1732)
TARGETS = (4.30, 4.10, 3.90, 3.75, 3.60, 3.50, 3.40, 3.30)
HORIZON = 1_048_576_000
RATES = {"A": 2.4e-3, "B2": 3e-4}


def reduce(runs):
    records = {}
    for label, directory in runs.items():
        arm, seed_text = label.split(":")
        seed = int(seed_text)
        if arm not in RATES or seed not in SEEDS:
            raise ValueError("run is outside the frozen confirmation: " + label)
        run = load_run(directory)
        identity_path = Path(directory) / "identity.json"
        identity = json.loads(identity_path.read_text())
        if (run["arm"] != arm or run["lr"] != RATES[arm]
                or identity["data"]["order_seed"] != seed or identity["init"]["seed"] != seed
                or identity["schedule"]["horizon_tokens"] != HORIZON
                or identity["argv"]["eval_every_tokens"] != 8_388_608
                or identity["argv"]["eval_windows"] != 256):
            raise ValueError("run differs from the frozen confirmation: " + label)
        records[label] = dict(complete=run["status"] == "completed" and run["result"]["tokens"] == HORIZON,
                              tokens=run["result"]["tokens"], training_seconds=run["result"]["training_seconds"],
                              identity_sha256=digest(identity_path),
                              result_sha256=digest(Path(directory) / "result.json"),
                              init_sha256=identity["init"]["sha256"],
                              data_manifest_sha256=identity["data"]["manifest_sha256"],
                              order_sha256=identity["data"]["order_sha256"],
                              final_dev=run["final"],
                              targets={str(t): crossing(run["curve"], t) for t in TARGETS})
    pairs = {}
    for seed in SEEDS:
        a, b = (records.get(f"{arm}:{seed}") for arm in ("A", "B2"))
        if a and b and any(a[k] != b[k] for k in ("init_sha256", "data_manifest_sha256", "order_sha256")):
            raise ValueError("paired initialization or data identity differs: " + str(seed))
        values = {}
        for target in TARGETS:
            ca, cb = (r["targets"][str(target)] if r else None for r in (a, b))
            values[str(target)] = dict(A=ca, B2=cb,
                time_ratio_B2_over_A=cb["seconds"] / ca["seconds"] if ca and cb and ca["seconds"] else None,
                token_ratio_A_over_B2=ca["tokens"] / cb["tokens"] if ca and cb and cb["tokens"] else None)
        pairs[str(seed)] = dict(complete=bool(a and b and a["complete"] and b["complete"]), targets=values)
    return dict(schema="latch-tinymoe-frozen-confirmation-v1", targets=TARGETS, seeds=SEEDS,
                horizon_tokens=HORIZON, runs=records, pairs=pairs,
                missing=[f"{arm}:{seed}" for seed in SEEDS for arm in RATES if f"{arm}:{seed}" not in records])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", required=True, help="ARM:SEED=run-directory")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    runs = {}
    for item in a.run:
        label, directory = item.split("=", 1)
        if label in runs:
            raise ValueError("duplicate run: " + label)
        runs[label] = directory
    write_json(a.out, reduce(runs))


if __name__ == "__main__":
    main()
