# Provenance: copied from research/latch_selective_cadence/compare_restart.py at latch commit 6fcad414ad86
# (git blob 23aeba34d9c0d9a3599aea881b4a5c51b10d33ce). Body unchanged except the usage line in the docstring (run as
# a script from this directory instead of `python -m research...`). Removed: nothing else.
"""Exact-restart comparison of two branch runs (straight vs resumed in a fresh process).

    python selective_compare_restart.py --straight RUN --resumed RUN --out OUT

Equal iff every per-microbatch row (loss, clip norms/coefficients; float32 bit patterns), every evaluation NLL and
every final parameter/optimizer-state checksum (all persistent tensors, including accumulators) are identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--straight", required=True)
    p.add_argument("--resumed", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import numpy as np
    s, r = Path(a.straight), Path(a.resumed)
    rows_s, rows_r = np.load(s / "updates.npz"), np.load(r / "updates.npz")
    rows = {}
    for key in ("loss", "fast_norm", "fast_clip", "slow_norm", "slow_clip"):
        x, y = rows_s[key], rows_r[key]
        rows[key] = bool(x.shape == y.shape and np.array_equal(x.view(np.uint32), y.view(np.uint32)))
    cs, cr = json.loads((s / "checksums.json").read_text()), json.loads((r / "checksums.json").read_text())
    differing = sorted(k for k in cs if cs[k] != cr.get(k)) + sorted(set(cr) - set(cs))
    result_s, result_r = (json.loads((x / "result.json").read_text()) for x in (s, r))
    es = [(e["microbatches"], e["windows"], e["nll"]) for e in result_s["evaluations"]]
    er = [(e["microbatches"], e["windows"], e["nll"]) for e in result_r["evaluations"]]
    report = dict(rows_bitwise=rows, checksum_tensors=len(cs), differing_tensors=differing[:20],
                  evaluations_equal=es == er, resumed_from=result_r.get("resumed_from"),
                  passed=all(rows.values()) and not differing and es == er)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "restart.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
