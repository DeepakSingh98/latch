"""Capture/build the small-batch Latch TinyMoE program at B x 1024 (CPU worker only).

Ordinary megakernel_lab.tinymoe.build.prepare with the selected interval-unit flags of
LATCH-TINYMOE-BUILD / -GA; only the tensor-update rule is replaced by sb_rule.make_rule
(plain participating AdamW + clip with staged LR/weight decay/beta2). The captured
objective is unchanged: CE_sum/targets * coefficient[0] + 0.02 * HF aux over the pass,
run with coefficients (1, 0) => mean CE + 0.02 aux. One optimizer update per program step;
no accumulation. Public weights are capture-time placeholders; runs overwrite every
parameter with the seed's random initialization before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--device-units", type=int, default=16)
    p.add_argument("--device-unit-jobs", type=int, default=16)
    p.add_argument("--device-unit-split", type=int, default=150000)
    p.add_argument("--device-unit-shared-pool", action="store_true")
    p.add_argument("--static-shared-reserve-bytes", type=int, default=8192)
    p.add_argument("--update-schedule", choices=("streamed", "streamed_prep", "streamed_ustate", "streamed_ustate_late",
                                                "streamed_prep_ustate", "streamed_prep_ustate_late"),
                   help="opt into ops.update_schedule interval rules (changed binaries need fresh qualification)")
    p.add_argument("--permutation-gradients", choices=("on", "off"), default="on",
                   help="ops.permutation_gradients for graph-proved dispatch/combine indices (off: general engine)")
    p.add_argument("--segmented-products",
                   help="opt into the flattened segmented-product body (ops.ragged_experts_lower)")
    p.add_argument("--norm-fusion",
                   help="opt into ops.norm_fusion rules: 'rows' or '+'-joined rule names (changed binaries need "
                        "fresh qualification)")
    p.add_argument("--attention-rules", choices=("streamed",),
                   help="opt into ops.attention_rules attention preparation/layout rules")
    p.add_argument("--matrix-schedule", choices=("underfilled",),
                   help="opt into ops.matrix_schedule rules for underfilled long-K matrix products")
    p.add_argument("--diagnostic-trace", action="store_true",
                   help="diagnostic build only: per-CTA phase-end stamps on profile launches (never timed)")
    p.add_argument("--screen-choices", default="",
                   help="research screen only: 'PROBLEM_KEY=body;...' existing matrix bodies forced through the step-16 "
                        "matrix_choices mechanism (admission still checked; inventory recorded)")
    p.add_argument("--screen-defines", default="",
                   help="research screen only: 'NAME VALUE;...' existing compile-time TMA body variants")
    p.add_argument("--screen-rules", default="",
                   help="research screen only: '+feature' group rules appended in the flat planner (recorded)")
    a = p.parse_args()
    import os
    if a.diagnostic_trace:
        os.environ["LATCH_DIAGNOSTIC_CTRACE"] = "1"
    if a.screen_rules:
        os.environ["LATCH_SCREEN_RULES"] = a.screen_rules
    if a.screen_defines:
        os.environ["LATCH_SCREEN_DEFINES"] = a.screen_defines

    import torch
    torch.set_num_threads(1)
    from megakernel_lab.operation_compiler import permutation_gradients
    permutation_gradients.ENABLED = a.permutation_gradients == "on"
    import megakernel_lab.tinymoe.build as build
    from megakernel_lab.tinymoe.model import Recipe
    from . import sb_rule
    from .protocol import COMMON

    # Recipe is build metadata only (horizon/LR are staged at run time); the rule carries
    # the common constants beta1/eps/clip.
    recipe = Recipe(name="unchanged", horizon=1000000, learning_rate=0.0, min_lr_fraction=0.0,
                    warmup_fraction=COMMON["warmup_fraction"], beta1=COMMON["beta1"], beta2=0.0,
                    eps=COMMON["eps"], weight_decay=0.0, clip_norm=COMMON["clip_norm"])
    flags = dict(compact_elementwise=True, outline_full_sums=True, native_code_only=True,
                 resource_phase_local_pointers=True, outline_pure_intervals=True,
                 device_units=a.device_units, device_unit_inline="isolate", device_unit_jobs=a.device_unit_jobs,
                 device_unit_split=a.device_unit_split, device_unit_shared_pool=a.device_unit_shared_pool,
                 static_shared_reserve_bytes=a.static_shared_reserve_bytes,
                 **({"update_schedule": a.update_schedule} if a.update_schedule else {}),
                 **({"segmented_products": a.segmented_products} if a.segmented_products else {}),
                 **({"norm_fusion": a.norm_fusion} if a.norm_fusion else {}),
                 **({"attention_rules": a.attention_rules} if a.attention_rules else {}),
                 **({"matrix_schedule": a.matrix_schedule} if a.matrix_schedule else {}))
    diagnostic = dict(diagnostic_trace=True) if a.diagnostic_trace else {}
    if a.screen_rules:
        diagnostic["screen_rules"] = a.screen_rules
    if a.screen_defines:
        diagnostic["screen_defines"] = a.screen_defines
    started = time.perf_counter()
    from contextlib import nullcontext
    inventory, choice_scope = [], nullcontext()
    if a.screen_choices:
        from megakernel_lab.operation_compiler.matrix_adapters.provider import matrix_choices

        # 'KEY=body;...' with KEY the provider's exact problem key (JSON list, e.g. from a recorded inventory)
        wanted = dict(item.rsplit("=", 1) for item in a.screen_choices.split(";"))
        choice_scope = matrix_choices(wanted, log=inventory)
    with choice_scope:
        summary = build.prepare(a.model, a.out, a.cache_dir, recipe=recipe, phase="plain",
                            batch_size=a.batch_size, sequence_length=a.sequence_length,
                            optimizer_factory=lambda model, recipe, ema_active=False: sb_rule.make_rule(model),
                            **flags)
    here = Path(__file__).resolve().parent
    record = dict(schema="latch-tinymoe-smallbatch-build-v1", wall_seconds=time.perf_counter() - started,
                  rule="sb_rule.make_rule", common=COMMON, batch_size=a.batch_size, sequence_length=a.sequence_length,
                  lane_sources={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                for name in ("sb_rule.py", "build.py", "protocol.py")},
                  flags=flags, permutation_gradients=a.permutation_gradients, summary=summary, **diagnostic,
                  **(dict(screen_choices=a.screen_choices, choice_inventory=inventory) if a.screen_choices else {}))
    (a.out / "sb-build.json").write_text(json.dumps(record, indent=2, default=repr) + "\n")
    print(json.dumps(dict(event="sb_build_complete", seconds=record["wall_seconds"]), default=repr), flush=True)


if __name__ == "__main__":
    main()
