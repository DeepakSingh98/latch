"""Capture/build the TinyMoE accumulating microbatch program (CPU worker only).

Uses the ordinary megakernel_lab.tinymoe.build.prepare path and its selected
interval-unit flags; only the tensor-update rule factory is replaced by
ga_rule.make_accumulating_rule. Shape: 8 sequences x 1024 tokens per microbatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .ga_common import PUBLISHED


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=PUBLISHED["per_device_train_batch_size"])
    p.add_argument("--sequence-length", type=int, default=PUBLISHED["sequence_length"])
    p.add_argument("--device-units", type=int, default=16)
    p.add_argument("--device-unit-jobs", type=int, default=16)
    p.add_argument("--device-unit-split", type=int, default=150000)
    p.add_argument("--device-unit-shared-pool", action="store_true")
    p.add_argument("--static-shared-reserve-bytes", type=int, default=8192)
    p.add_argument("--capture-only", action="store_true")
    a = p.parse_args()

    import megakernel_lab.tinymoe.build as build
    from megakernel_lab.tinymoe.model import Recipe
    from . import ga_rule

    # Recipe is build metadata only; the rule above carries the published values.
    recipe = Recipe(name="unchanged", horizon=PUBLISHED["max_steps"], learning_rate=PUBLISHED["learning_rate"],
                    min_lr_fraction=0.0, warmup_fraction=PUBLISHED["warmup_steps"] / PUBLISHED["max_steps"],
                    beta1=PUBLISHED["adam_beta1"], beta2=PUBLISHED["adam_beta2"], eps=PUBLISHED["adam_epsilon"],
                    weight_decay=PUBLISHED["weight_decay"], clip_norm=PUBLISHED["max_grad_norm"])
    started = time.perf_counter()
    summary = build.prepare(a.model, a.out, a.cache_dir, recipe=recipe, phase="plain",
        batch_size=a.batch_size, sequence_length=a.sequence_length, capture_only=a.capture_only,
        optimizer_factory=lambda model, recipe, ema_active=False: ga_rule.make_accumulating_rule(model),
        compact_elementwise=True, outline_full_sums=True, native_code_only=True,
        resource_phase_local_pointers=True, outline_pure_intervals=True,
        **({} if a.capture_only else dict(device_units=a.device_units, device_unit_inline="isolate",
            device_unit_jobs=a.device_unit_jobs, device_unit_split=a.device_unit_split,
            device_unit_shared_pool=a.device_unit_shared_pool,
            static_shared_reserve_bytes=a.static_shared_reserve_bytes)))
    here = Path(__file__).resolve().parent
    record = dict(schema="latch-tinymoe-ga-build-v1", wall_seconds=time.perf_counter() - started,
        rule="ga_rule.make_accumulating_rule", published=PUBLISHED,
        lane_sources={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                      for name in ("ga_rule.py", "build_ga.py", "ga_common.py")},
        flags=dict(compact_elementwise=True, outline_full_sums=True, native_code_only=True,
                   resource_phase_local_pointers=True, outline_pure_intervals=True,
                   device_units=a.device_units, device_unit_inline="isolate", device_unit_jobs=a.device_unit_jobs,
                   device_unit_split=a.device_unit_split, device_unit_shared_pool=a.device_unit_shared_pool,
                   static_shared_reserve_bytes=a.static_shared_reserve_bytes),
        summary=summary)
    (a.out / "ga-build.json").write_text(json.dumps(record, indent=2, default=repr) + "\n")
    print(json.dumps(dict(event="ga_build_complete", seconds=record["wall_seconds"]), default=repr), flush=True)


if __name__ == "__main__":
    main()
