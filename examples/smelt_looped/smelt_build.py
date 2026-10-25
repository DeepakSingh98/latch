# Provenance: copied from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch), original path research/smelt_latch/build.py (blob 75f2825551d4,
# sha256 c81c3cbaaf35...; last changed at e4a25d20e).
# Changed: imports (.looped -> smelt_looped, .runners -> smelt_runners), the docstring's invocation/module names,
# and the lane_sources record in smelt-build.json (hashes of this example's smelt_build.py / smelt_looped.py /
# smelt_runners.py / smelt_recipe.py instead of the donor build.py / looped.py / runners.py). Nothing removed.
# The recipe, optimizer rule (sb_rule.make_rule), prepare() flags, rules="auto" selection and the
# capture_model replacement are unchanged. Run as `python -B smelt_build.py ...` (see quickstart.sh).
"""Capture/build the looped TinyMoE update through the normal Latch API (CPU worker only).

    python -B smelt_build.py --model M --loop FIRST,SPAN,PASSES --out OUT/artifact \
        --cache-dir OUT/cache [--batch-size 32] [--rules auto|none] [--option KEY=VALUE ...]

Capture: smelt_looped.capture_factory(spec) replaces megakernel_lab.tinymoe.build.capture_model
(same signature; the packaged CapturedTinyMoE attention/expert/readout bodies visited in the SMELT order);
optimizer rule: megakernel_lab.tinymoe.learning.sb_rule.make_rule (participating AdamW + clip, staged LR/WD/beta2)
whose participation reads active.<layer> = union over that layer's visits. Build: megakernel_lab.tinymoe.build.prepare
with the LATCH-B32-TRANSFER delivered configuration (rules="auto" + the donor interval/device-unit flags).
No generated code is edited. --loop 0,1,1 (passes 1) is the unlooped recipe-Q program (diagnostic identity check).
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
    p.add_argument("--loop", required=True, help="FIRST,SPAN,PASSES (0-based first looped layer)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=1024)
    p.add_argument("--device-units", type=int, default=16)
    p.add_argument("--device-unit-jobs", type=int, default=18)
    p.add_argument("--device-unit-split", type=int, default=150000)
    p.add_argument("--static-shared-reserve-bytes", type=int, default=8192)
    p.add_argument("--rules", choices=("auto", "none"), default="auto")
    p.add_argument("--no-diagnostics", action="store_true", help="omit per-visit routing state outputs")
    p.add_argument("--option", action="append", default=[], help="extra prepare() option KEY=VALUE (string)")
    a = p.parse_args()

    import torch
    torch.set_num_threads(1)
    import megakernel_lab.tinymoe.build as build
    from megakernel_lab.tinymoe.learning import sb_rule
    from megakernel_lab.tinymoe.learning.protocol import COMMON
    from megakernel_lab.tinymoe.model import Recipe
    from smelt_looped import AUX_GROUP_SEQUENCES, capture_factory
    from smelt_runners import spec_of

    spec = spec_of(a.loop)
    if a.batch_size > AUX_GROUP_SEQUENCES and a.batch_size % AUX_GROUP_SEQUENCES:
        raise ValueError("batches above one aux group must hold whole groups")
    build.capture_model = capture_factory(spec)  # prepare() resolves this module global
    recipe = Recipe(name="unchanged", horizon=1000000, learning_rate=0.0, min_lr_fraction=0.0,
                    warmup_fraction=COMMON["warmup_fraction"], beta1=COMMON["beta1"], beta2=0.0,
                    eps=COMMON["eps"], weight_decay=0.0, clip_norm=COMMON["clip_norm"])
    extra = dict(item.split("=", 1) for item in a.option)
    options = (dict(rules="auto") if a.rules == "auto" else dict(update_schedule="streamed")) | extra
    flags = dict(compact_elementwise=True, outline_full_sums=True, native_code_only=True,
                 resource_phase_local_pointers=True, outline_pure_intervals=True,
                 device_units=a.device_units, device_unit_inline="isolate", device_unit_jobs=a.device_unit_jobs,
                 device_unit_split=a.device_unit_split, device_unit_shared_pool=False,
                 static_shared_reserve_bytes=a.static_shared_reserve_bytes, diagnostics=not a.no_diagnostics,
                 **options)
    started = time.perf_counter()
    summary = build.prepare(a.model, a.out, a.cache_dir, recipe=recipe, phase="plain",
                            batch_size=a.batch_size, sequence_length=a.sequence_length,
                            optimizer_factory=lambda model, recipe, ema_active=False: sb_rule.make_rule(model),
                            **flags)
    here = Path(__file__).resolve().parent
    packaged = Path(sb_rule.__file__).resolve().parent
    record = dict(schema="smelt-latch-build-v1", wall_seconds=time.perf_counter() - started,
                  loop=spec.asdict(), visits=spec.visits(12), rule="megakernel_lab.tinymoe.learning.sb_rule.make_rule",
                  common=COMMON, batch_size=a.batch_size, sequence_length=a.sequence_length,
                  aux_group_sequences=min(a.batch_size, AUX_GROUP_SEQUENCES), options=options,
                  lane_sources={name: hashlib.sha256((here / name).read_bytes()).hexdigest()
                                for name in ("smelt_build.py", "smelt_looped.py", "smelt_runners.py",
                                             "smelt_recipe.py")}
                  | {"learning/" + name: hashlib.sha256((packaged / name).read_bytes()).hexdigest()
                     for name in ("sb_rule.py", "protocol.py")},
                  flags={k: v for k, v in flags.items()}, summary=summary)
    text = json.dumps(record, indent=2, default=repr) + "\n"
    (a.out.parent / "smelt-build.json").write_text(text)
    (a.out / "smelt-build.json").write_text(text)  # travels with packed artifacts
    print(json.dumps(dict(event="smelt_build_complete", seconds=record["wall_seconds"], loop=spec.asdict(),
                          options=options), default=repr), flush=True)


if __name__ == "__main__":
    main()
