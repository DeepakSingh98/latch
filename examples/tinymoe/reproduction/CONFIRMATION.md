# Longer-horizon study: historical confirmation freeze

The historical protocol below is preserved under its original name. It changes
the learning-rate horizon from 131,072,000 to 1,048,576,000 tokens. With 5% warmup,
that changes warmup from 6,553,600 to 52,428,800 tokens and stretches the cosine
decay. The learning-rate function changes from the first update, so these runs
and their early target crossings are **not direct replications of the pilot**.

The separate direct original-schedule replication and seed-1729 schedule-isolation
study are frozen in [FOLLOWUP.md](FOLLOWUP.md). Their public commands and reducer
limitations are in [README.md](README.md). `confirmation.json` and the historical
`learning confirm` command continue to describe only this longer-horizon study.
The numerical settings and original source hashes below are unchanged.

---

Original heading: Confirmation freeze (2026-10-14 ~22:45 UTC, after the pilot, before any confirmation run)

## Pilot outcome that triggered confirmation (seed 1729, horizon 131,072,000 tokens)

Calibration finished with the equal allowance of four trials per arm (`results/pilot/analysis.json`):

| arm | trials (peak LR -> final dev NLL, 2,048 windows) | selected |
|---|---|---|
| A | 3e-4 -> 4.5446, 6e-4 -> 4.1019, 1.2e-3 -> 4.0509, 2.4e-3 -> 4.0246 | 2.4e-3 |
| B1 | 2.65e-5 -> 4.0367, 5.3e-5 -> 3.7311, 1.06e-4 -> 3.5770, 2.12e-4 -> 3.5261 | 2.12e-4 |
| B2 | 3.75e-5 -> 4.0728, 7.5e-5 -> 3.7288 (rerun after preemption), 1.5e-4 -> 3.5867, 3e-4 -> 3.5376 | 3e-4 |

Every arm's best is its highest tried LR (each arm's fourth trial extended its edge by one factor of
two, as the frozen rule required); optima may lie higher for all three arms.

At A's eight second-half dev levels (4.612 ... 4.107, 256-window curve), selected B2 needs 0.57-0.69x
A's accumulated training-update time with 2.7-3.2x fewer tokens; selected B1 needs 1.00-1.20x A's time.
The frozen continuation rule (a Latch arm reaching a second-half A level in no more time) is met by
B2 and not by B1. Same-GPU paired throughput: B2 needs 1.86x and B1 3.18x A's time per token.

## Frozen confirmation

* **Arms:** A = compiled conventional, 128 x 1024 per update (16 x 8 GA), peak LR 2.4e-3.
  B2 = Latch, 2 x 1024 per update, peak LR 3e-4. Everything else exactly as in PLAN.md
  (model, random init generator, data/windows, objective and router reduction, numerics,
  optimizer constants, token-defined beta2 and weight decay, warmup 5% + cosine to zero over tokens).
  Same artifacts (B2 `build-b2s1024-01`, library hash in run identities); training source `cfb782d8`,
  which reproduces the pilot runs bitwise (`r1-*`/`r2-*` resume qualification: dev evaluations equal
  the pilot receipts, resumed final parameters equal the uninterrupted run).
* **Seeds:** 1730, 1731, 1732 (fresh; init seed = data-order seed). Within a pair both arms load the
  same `init-sSEED.pt` and the same window order.
* **Horizon:** 1,048,576,000 tokens = 8,000 A updates = 512,000 B2 updates (8x the pilot; close to the
  published TinyMoE schedule of 8,070 x 131,072 tokens).
* **Evaluation:** first 256 dev windows (262,144 targets) every 8,388,608 tokens (125 points);
  2,048 dev windows at the end; test split (first 4,096 windows, 4,194,304 targets) once per run on the
  saved final parameters (`evaluate_final.py`), after all runs complete.
* **Targets (256-window dev NLL), frozen now:** 4.30, 4.10, 3.90, 3.75, 3.60, 3.50, 3.40, 3.30.
  3.40 and 3.30 are late targets that A may not reach; unreached targets are reported as such.
  For each seed and target: first evaluation at or below the target (tokens, accumulated training
  seconds, GPU-hours) with the previous evaluation as the resolution lower bound. No interpolation,
  extrapolation, post-hoc target choice or multiplication of speed and learning ratios.
* **Execution:** one run per single-H100 `a3-highgpu-1g` spot VM, pairs in parallel; checkpoint every
  67,108,864 tokens and exact resume (contract-bound) after preemption or lease end. Training clock
  accumulates across segments; compilation/warmup, evaluation, checkpointing and data loading excluded
  and reported. Per-run medians are compared with the pilot same-GPU paired throughput.

This public rendering omits operational spending details; the frozen experiment settings
and targets above are unchanged. Original source/hash is in docs/v0/LEARNING.md.
