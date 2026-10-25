# Provenance: copied from research/latch_selective_10/build10.py at latch commit 6fcad414ad86
# (git blob 2de2a63634d16363b862646e49c02d516ef74e4e; itself a copy of research/latch_selective_cadence/build_cadence.py
# blob acb37de2f624d3b1517c445326b93ff880a35e79 plus gradient_outputs/accumulation_rules).
# Changed: imports (research.latch_tinymoe_smallbatch.protocol.COMMON -> installed megakernel_lab.tinymoe.learning.
# protocol.COMMON, identical constants; research.latch_selective_cadence.cadence_rule -> local selective_cadence_rule)
# and the lane_sources record (hashes of this directory's selective_cadence_rule.py / selective_build.py and of the
# installed protocol.py instead of the donor's research/ paths). Build flags, defaults, the Recipe and the
# prepare(...) call are unchanged. Run on a CPU-only process: megakernel_lab.tinymoe.build.prepare refuses to run
# when CUDA is visible.
"""LATCH-SELECTIVE-10 copy of research.latch_selective_cadence.build_cadence (+ the normal build options
gradient_outputs and accumulation_rules; omitted = the study/combined lowering unchanged).

Capture/build one phase program of a selective-cadence arm at 2 x 1024 (CPU worker only).

Ordinary megakernel_lab.tinymoe.build.prepare with the flags of the frozen LATCH-TINYMOE-25 candidate
(`--update-schedule streamed` plus the interval-unit flags of the study's build_sb.py); only the tensor-update
rule is replaced by cadence_rule.make_rule(slow_group, phase), and the update schedule is
"streamed_ustate" (= streamed + unchanged-state commits elided; DESIGN.md "Execution"). The captured
objective is unchanged: CE_sum/targets * coefficient[0] + 0.02 * HF aux over the pass, run with
coefficients (1, 0). Both phases of an arm have identical persistent-state signatures.
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
    p.add_argument("--slow-group", required=True)
    p.add_argument("--phase", choices=("accumulate", "apply"), required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--update-schedule", default="streamed_ustate",
                   choices=("streamed", "streamed_ustate", "streamed_ustate_late", "streamed_prep", "streamed_prep_ustate",
                            "streamed_prep_ustate_late"))
    # LATCH-TINYMOE-COMBINED: the combined execution options (normal build API; omitted = unchanged lowering).
    p.add_argument("--segmented-products")
    p.add_argument("--norm-fusion")
    p.add_argument("--attention-rules")
    p.add_argument("--matrix-schedule")
    # LATCH-SELECTIVE-10: release step 15 gradient consumption and ops.accumulation_rules (normal build API).
    p.add_argument("--gradient-outputs", choices=("returned", "consumed"), default="returned")
    p.add_argument("--accumulation-rules")
    p.add_argument("--device-units", type=int, default=16)
    p.add_argument("--device-unit-jobs", type=int, default=16)
    p.add_argument("--device-unit-split", type=int, default=150000)
    p.add_argument("--static-shared-reserve-bytes", type=int, default=8192)
    p.add_argument("--capture-only", action="store_true")
    a = p.parse_args()

    import torch
    torch.set_num_threads(1)
    import megakernel_lab.tinymoe.build as build
    from megakernel_lab.tinymoe.model import Recipe
    from megakernel_lab.tinymoe.learning import protocol
    from megakernel_lab.tinymoe.learning.protocol import COMMON
    import selective_cadence_rule as cadence_rule

    recipe = Recipe(name="unchanged", horizon=1000000, learning_rate=0.0, min_lr_fraction=0.0,
                    warmup_fraction=COMMON["warmup_fraction"], beta1=COMMON["beta1"], beta2=0.0,
                    eps=COMMON["eps"], weight_decay=0.0, clip_norm=COMMON["clip_norm"])
    flags = dict(compact_elementwise=True, outline_full_sums=True, native_code_only=True,
                 resource_phase_local_pointers=True, outline_pure_intervals=True)
    if not a.capture_only:
        flags.update(device_units=a.device_units, device_unit_inline="isolate", device_unit_jobs=a.device_unit_jobs,
                     device_unit_split=a.device_unit_split, device_unit_shared_pool=False,
                     static_shared_reserve_bytes=a.static_shared_reserve_bytes, update_schedule=a.update_schedule)
        flags.update({k: v for k, v in dict(segmented_products=a.segmented_products, norm_fusion=a.norm_fusion,
                                            attention_rules=a.attention_rules,
                                            matrix_schedule=a.matrix_schedule,
                                            accumulation_rules=a.accumulation_rules).items() if v is not None})
        if a.gradient_outputs != "returned":
            flags["gradient_outputs"] = a.gradient_outputs
    started = time.perf_counter()
    summary = build.prepare(a.model, a.out, a.cache_dir, recipe=recipe, phase="plain",
                            batch_size=a.batch_size, sequence_length=a.sequence_length, capture_only=a.capture_only,
                            optimizer_factory=lambda model, recipe, ema_active=False: cadence_rule.make_rule(
                                model, slow_group=a.slow_group, phase=a.phase),
                            **flags)
    here = Path(__file__).resolve().parent
    record = dict(schema="latch-selective-cadence-build-v1", wall_seconds=time.perf_counter() - started,
                  rule="cadence_rule.make_rule", slow_group=a.slow_group, phase=a.phase, common=COMMON,
                  batch_size=a.batch_size, sequence_length=a.sequence_length,
                  lane_sources={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                for name in ("selective_cadence_rule.py", "selective_build.py")}
                  | {"megakernel_lab/tinymoe/learning/protocol.py":
                     hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest()},
                  flags=flags, summary=summary)
    (a.out / "cadence-build.json").write_text(json.dumps(record, indent=2, default=repr) + "\n")
    print(json.dumps(dict(event="cadence_build_complete", seconds=record["wall_seconds"]), default=repr), flush=True)


if __name__ == "__main__":
    main()
