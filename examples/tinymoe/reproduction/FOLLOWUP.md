# Schedule-controlled follow-up (frozen 2026-10-15 ~01:50 UTC, before any follow-up run)

User request 2026-10-15: distinguish replication of the 131M-token pilot from the effect of the
8x-stretched schedule used by the 1.05B-token confirmation runs. Those runs (CONFIRMATION.md) are a
longer-horizon study, finish unchanged, and are preserved.

## Runs (8 new; no retuning, no new search)

Common: pilot protocol exactly (PLAN.md): model, random-init generator, batch sizes (A 128 x 1024 as
16 x 8 GA; B2 2 x 1024), AdamW constants, token-defined beta2/weight decay, 5% warmup + cosine to zero
over `--horizon-tokens`, precision, objective/router reduction, corpus/windows, B2 artifact
`build-b2s1024-01`, training source `cfb782d8` bytes (qualified bitwise against the pilot). Peak LRs
frozen from the pilot selection: **A 0.0024, B2 0.0003**. Evaluation at the pilot cadence: first 256 dev
windows every 4,194,304 tokens; first 2,048 dev windows once at the end (reported separately).

1. **Short-schedule replication, fresh seeds 1730, 1731, 1732** (A and B2 each, 6 runs):
   `--horizon-tokens 131072000`. Each pair reuses the confirmation seed's `init-sSEED.pt`
   (1730 `57c0f94c…`, 1731 `f64c78d7…`, 1732 `ef087d17…`) and data order (PCG64(seed)).
2. **Schedule isolation, pilot seed 1729** (A and B2, 2 runs): `--horizon-tokens 1048576000
   --stop-tokens 131072000` (the 1.05B schedule, stopped at 131M). `protocol.lr_at` depends only on
   the horizon; `--stop-tokens` only ends the loop, so the long schedule is preserved. Controls: the
   completed seed-1729 pilot runs `d4-a-lr2p4e-3` and `h1-b2-lr3e-4` (short schedule), not rerun.

Both arms of a pair run sequentially on the same single-H100 VM (matched GPU), one VM per seed.
Long-schedule comparisons for seeds 1730-1732 use the existing confirmation runs' prefixes
(256-window evaluations every 8,388,608 tokens up to 131M).

## Frozen comparison levels (256-window dev NLL)

The original eight pilot comparison levels, exactly as stored in `results/pilot/analysis.json`
(`levels`): **4.612, 4.466, 4.34, 4.254, 4.187, 4.141, 4.116, 4.107**. Also reported where reached:
the confirmation targets 4.30, 4.10, 3.90, 3.75, 3.60, 3.50, 3.40, 3.30.

## Reported

Dev NLL vs tokens and vs accumulated training-update seconds; dev NLL at matched token counts; first
recorded crossing of each level (tokens, seconds) with the previous evaluation as the resolution lower
bound; unreached levels marked; final 2,048-window dev separately; whether the short-run B2 benefit
repeats on fresh seeds, and the effect of the schedule change alone on each arm. No interpolation,
extrapolation or new headline selection.
