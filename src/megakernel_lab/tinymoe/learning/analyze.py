"""Reduce pilot/confirmation run receipts: curves, LR selection, time and tokens to dev levels.

Inputs are the small run receipts collected by the host (run/result.json, run/evaluation.jsonl).
Crossings are first evaluation points at or below a level; the previous evaluation point gives the
evaluation-resolution lower bound. No interpolation, extrapolation or ratio multiplication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    rows = [json.loads(line) for line in (directory / "evaluation.jsonl").read_text().splitlines() if line.strip()]
    curve = [r for r in rows if r["split"] == "dev"]
    return dict(path=str(directory), arm=result["arm"], lr=result["hyperparameters"]["learning_rate"],
                status=result["status"], result=result, curve=curve,
                final=next((r for r in rows if r["split"] == "dev-final"), None))


def crossing(curve, level):
    previous = None
    for row in curve:
        if row["nll"] <= level:
            return dict(tokens=row["tokens"], seconds=row["training_seconds"],
                        tokens_lower=previous["tokens"] if previous else 0,
                        seconds_lower=previous["training_seconds"] if previous else 0.0, nll=row["nll"])
        previous = row
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run directories (each with result.json, evaluation.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--levels", type=float, nargs="*", default=None, help="dev levels; default from best A curve")
    a = p.parse_args()
    runs = [load_run(d) for d in a.runs]
    best = {}
    for run in runs:
        if run["final"] is None:
            continue
        key = run["arm"]
        if key not in best or run["final"]["nll"] < best[key]["final"]["nll"]:
            best[key] = run
    table = [dict(arm=r["arm"], lr=r["lr"], status=r["status"], final_dev=r["final"] and r["final"]["nll"],
                  last_curve_dev=r["curve"][-1]["nll"] if r["curve"] else None,
                  training_seconds=r["result"]["training_seconds"], median_update_ms=r["result"]["median_update_ms"],
                  tokens=r["result"]["tokens"]) for r in sorted(runs, key=lambda r: (r["arm"], r["lr"]))]
    levels = a.levels
    if levels is None and "A" in best:
        curve = best["A"]["curve"]
        half = [r["nll"] for r in curve if r["tokens"] >= curve[-1]["tokens"] / 2]
        levels = sorted({round(v, 3) for v in half[:: max(1, len(half) // 6)]}, reverse=True)
    crossings = {arm: {str(level): crossing(run["curve"], level) for level in levels or []} for arm, run in best.items()}
    ratios = {}
    if "A" in best:
        for arm in best:
            if arm == "A":
                continue
            ratios[arm] = {}
            for level in levels or []:
                ca, cb = crossings["A"][str(level)], crossings[arm][str(level)]
                if ca and cb:
                    ratios[arm][str(level)] = dict(time_ratio=cb["seconds"] / ca["seconds"],
                                                   token_ratio_A_over_B=ca["tokens"] / cb["tokens"])
    summary = dict(schema="latch-tinymoe-smallbatch-analysis-v1", runs=table,
                   selected={arm: dict(lr=r["lr"], final_dev=r["final"]["nll"], path=r["path"]) for arm, r in best.items()},
                   levels=levels, crossings=crossings, ratios_at_common_levels=ratios)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        plot(runs, best, out)
    except Exception as exc:  # plotting is optional
        summary["plot_error"] = repr(exc)
    print(json.dumps(dict(selected=summary["selected"], ratios=ratios), indent=2))


def plot(runs, best, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"A": "#1f6feb", "B1": "#d29922", "B2": "#2da44e"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for run in runs:
        c = run["curve"][1:]
        if not c:
            continue
        chosen = best.get(run["arm"]) is run
        kw = dict(color=colors.get(run["arm"], "gray"), alpha=1.0 if chosen else 0.3, lw=2 if chosen else 1,
                  label=f"{run['arm']} lr={run['lr']:.3g}" + (" (selected)" if chosen else ""))
        axes[0].plot([r["tokens"] / 1e6 for r in c], [r["nll"] for r in c], **kw)
        axes[1].plot([r["training_seconds"] / 60 for r in c], [r["nll"] for r in c], **kw)
    axes[0].set_xlabel("training tokens (M)")
    axes[1].set_xlabel("accumulated complete-update time (min)")
    axes[0].set_ylabel("dev NLL (256 windows)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylim(top=min(6.0, ax.get_ylim()[1]))
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)


if __name__ == "__main__":
    main()
