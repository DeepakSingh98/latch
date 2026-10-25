# LATCH-TINYMOE-SMALLBATCH plan (written 2026-10-14 before any learning or throughput result)

**Question.** Can Latch's cheaper small-batch updates overcome their lower token throughput by
learning more efficiently, so that TinyMoE reaches a fixed held-out quality in less accumulated
training-update time than efficient large-batch compiled conventional training?

Public rendering of the frozen learning protocol. Operational budgets and cloud account details
are omitted; the selection and continuation rules are unchanged.

## Fixed across arms

| item | value |
|---|---|
| model | public `FlameF0X/TinyMoE-200m-2x16` @ `9b209f7439aadfac282e0bf44f2fda19771aba90`, unchanged architecture, 199,837,056 parameters, top-2 of 16 experts, no capacity/drops |
| initialization | random, `init_state.py` (Transformers `_init_weights`, N(0, 0.02), norms 1), identical tensors for both arms of a pair |
| auxiliary losses | none besides the model's router load-balancing loss (no intermediate supervision, no EMA) |
| tokenizer / data | TinyMoE tokenizer on the pinned learning-scaling FineWeb-Edu documents (corpus manifest `75e67515…8590`), re-read from the pinned parquet rows; train / validation(dev) / test document splits unchanged |
| windows | 1025 tokens inside one document, stride 1024: 1024 inputs + **1024 supplied next-token targets** for every sequence in every arm (resolves the donor 1023/1024 discrepancy, which came only from the HF Trainer's internal shift) |
| data order | one PCG64(order seed) permutation of training windows; arm A update u = windows [128u,128u+128), Latch B-sequence update j = windows [Bj, Bj+B) |
| objective per update | mean CE over the update's targets + 0.02 x mean over forward passes of the HF Mixtral load-balancing aux of that pass |
| router reduction | aux is computed per forward pass over that pass's tokens (all layers concatenated, pinned HF formula); arm A adds (CE_mean + 0.02 aux)/16 per 8-sequence microbatch; Latch adds CE_mean + 0.02 aux per pass. Relative coefficient 0.02 in both. The published HF Trainer added 0.02 aux undivided per microbatch (16 x 0.02 relative); that GA artefact is not reproduced. Statistic grouping differs by construction (8,192 vs 1,024/2,048 tokens per aux) |
| numerics | FP32 masters/moments, BF16 projections/attention/expert GEMMs, FP32 CE; deterministic algorithms and TF32 off on the conventional arm |
| optimizer | AdamW, beta1 0.9, eps 1e-6, global clip 1.0, decoupled decay on matrices only; absent-gradient experts unchanged |
| batch-dependent hyper-parameters, defined in tokens | beta2 = 0.999^(B/128) (constant second-moment half-life in tokens; 0.999 at 128 sequences); weight decay chosen so lr*wd per token equals the published 3e-4*0.1 per 131,072 tokens |
| schedule | linear warmup over the first 5% of the token horizon, cosine to zero at the horizon, evaluated at each update's token midpoint |
| evaluation | dev (validation split) final-readout NLL, installed Transformers modules over the live weights, BF16 autocast, router terms excluded; first 256 dev windows (262,144 targets) every 4,194,304 tokens, first 2,048 dev windows at the end; test split untouched until the final report |

## Arms

* **A — compiled conventional.** Existing LATCH-TINYMOE `compiled-grouped` backend with the
  LATCH-TINYMOE-GA accumulation loop: 16 microbatches x 8 x 1024 = 128 sequences per optimizer
  update, fused AdamW. Router term divided by 16 per microbatch (above).
* **B1 / B2 — Latch.** Generated TinyMoE programs built at 1 x 1024 and 2 x 1024 with the selected
  interval-unit flags (`megakernel_lab.tinymoe.learning.build`), one optimizer update per program step, no accumulation. Rule
  `megakernel_lab.tinymoe.learning.sb_rule` = donor participating AdamW + clip with LR, weight decay and beta2 staged as scalars.

## Clocks

Primary: accumulated synchronized wall seconds of complete optimizer updates (staging/H2D tokens,
schedule fills, forward/backward, accumulation, clipping, optimizer, absent-expert bookkeeping).
Excluded and reported separately: compilation/warmup (undone before the run), evaluation,
parameter writes, host data loading. All cloud activity counts toward spend.

## Audits (before trajectories are trusted)

One complete update from random weights (updates 1 and 2 of the pilot schedule) per arm, against the
unchanged donor policy the packaged `megakernel_lab/tinymoe/POLICY.json` with the independent FP64 reference at
the observed memberships: B1/B2 `megakernel_lab.tinymoe.learning.audit native`; A 16 `microbatch` audits + `accumulate`
(accumulated G vs FP64 sum, clip, AdamW formula). The instrumented arm-A update is checked bitwise
against the uninstrumented timed path.

## Pilot (seed 1729)

1. Same-GPU alternating throughput (`paired.py`): A, B1, B2 over equal tokens, 3 rounds. The
   time-per-token ratio B/A is the **sample-efficiency factor B must achieve** (tokens_A/tokens_B to a
   target) just to tie in training time.
2. Learning-rate calibration, equal allowance of **4 trials per arm** at one common token horizon
   H (planned 131,072,000 tokens = 1,000 A updates; revised only from measured throughput, before
   any learning result). First three trials per arm: A {3e-4, 6e-4, 1.2e-3}; B2 = A/8 and B1 = A/sqrt(128)
   (square-root scaling): B2 {3.75e-5, 7.5e-5, 1.5e-4}, B1 {2.65e-5, 5.3e-5, 1.06e-4}. The fourth trial per
   arm: one factor-2 step beyond the best if the best is at a grid edge, otherwise the geometric midpoint
   between the best and its better neighbour. Selection per arm: lowest final 2,048-window dev NLL.
3. Compare the selected runs: dev NLL vs tokens, dev NLL vs accumulated training seconds, and for dev
   levels reached by both, tokens and seconds to first reach the level (evaluation-resolution bounds).

**Continue to confirmation only if** the best Latch arm reaches some dev level in the second half of A's
selected curve in no more accumulated training time than A, or its time ratio at the final common
level is <= 1.15 AND decreasing across the last three common levels. Otherwise stop with the negative
result. Budget remaining is not a reason to continue.

## Confirmation (only if warranted; frozen before launch)

One Latch configuration and one conventional configuration frozen from the pilot; three fresh paired
seeds (init + data order identical within a pair); several dev targets including a late target frozen
before launch; a token horizon chosen from pilot throughput and curves within the remaining budget;
pairs parallel on matched single-H100 a3-highgpu-1g VMs. Report target attainment (with evaluation
resolution bounds), unreached targets, tokens, training seconds, GPU-hours and final dev/test loss
for every seed. No multiplication of separate speed and learning ratios.
