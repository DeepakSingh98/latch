# Provenance: copied from research/latch_selective_cadence/runners.py at latch commit 6fcad414ad86
# (git blob 916ae8a8e98273cbf69a82d0adc6e10f64397ba9; the runner SELECTIVE-10's cbench/equal_states/branch jobs used).
# Changed: imports only -- research.latch_tinymoe_smallbatch.{protocol,sb_rule} -> the installed
# megakernel_lab.tinymoe.learning.{protocol,sb_rule} (sb_rule.py is byte-identical to the donor's; protocol.py equals
# the donor's except that the donor appends schedule_spec/schedule_lr, which this code never calls), and
# `from . import cadence_rule` -> the local selective_cadence_rule. Arithmetic, staging order, PEAK/HORIZON schedule
# constants and the apply-at-u-mod-16==15 selection are unchanged. Baseline (arm A1) is kept because Cadence derives
# from it; the quickstart does not use A1 (it needs a separately built sb_rule program with sb-build.json).
"""Native runners for one B2 microbatch stream (GPU worker).

Baseline   A1: the frozen LATCH-TINYMOE-25 program (study sb_rule), one update per microbatch, staged exactly as the
           study trainer (Latch.update: LR fill, H2D tokens, one launch).
Cadence    A2-A4: the accumulate/apply program pair of one arm sharing every persistent tensor and one temporary
           arena (the TinyMoE two-phase loading pattern of megakernel_lab.tinymoe.native.Native). The host selects
           `apply` at u mod 16 = 15 and stages the slow window scalars only then.

Both load a parent study checkpoint's runner state (parameters, optimizer state, model state). Cadence-only state
(accumulators, participation, slow scalars) starts at zero at a window boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

DEVICE = "cuda"
SEQUENCES = 2
PEAK, HORIZON = 3e-4, 1_048_576_000  # parent legacy schedule


def lr_of(u):
    from megakernel_lab.tinymoe.learning import protocol
    return protocol.lr_at(u, PEAK, SEQUENCES, HORIZON)


def hyperparameters():
    from megakernel_lab.tinymoe.learning import protocol
    return protocol.hyperparameters("B2", PEAK)


def bits(value):
    import numpy as np
    return int(np.float32(value).view(np.uint32))


def checksums(groups):
    """Per-tensor bit-pattern sums (plain and position-weighted), as LATCH-TINYMOE-25 bench.checksums."""
    import torch
    out = {}
    for group, values in groups:
        for name, value in values.items():
            words = value.detach().reshape(-1)
            words = words.view(torch.int32) if words.element_size() == 4 else words.view(torch.int64)
            words = words.to(torch.int64)
            weights = torch.arange(1, words.numel() + 1, device=words.device, dtype=torch.int64) % 65521 + 1
            out[group + ":" + name] = [int(words.sum()), int((words * weights).sum())]
    return out


def _copy_into(values, saved, kind, *, allow_new=()):
    import torch
    missing = set(values) - set(saved)
    if set(saved) - set(values) or any(not any(name.startswith(p) for p in allow_new) for name in missing):
        raise ValueError(f"checkpoint {kind} names differ from the program: "
                         f"{sorted(set(saved) ^ set(values))[:6]}")
    with torch.no_grad():
        for name, value in values.items():
            if name in saved:
                value.copy_(saved[name])
            else:
                value.zero_()


class Baseline:
    arm = "A1"
    slow_group = None

    def __init__(self, artifact, lr_scale=1.0):
        import torch
        from megakernel_lab.operation_compiler import ModuleUpdate
        from megakernel_lab.tinymoe.native import artifact_identity
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from megakernel_lab.tinymoe.learning import sb_rule
        self.sb_rule, self.hp = sb_rule, hyperparameters()
        # Diagnostic LR control only (DESIGN.md decision rules use lr_scale = 1): every microbatch LR x lr_scale with
        # weight decay / lr_scale, so lr x wd per token (the decay) is unchanged.
        self.lr_scale = float(lr_scale)
        if self.lr_scale != 1.0:
            self.hp = dict(self.hp, weight_decay=self.hp["weight_decay"] / self.lr_scale)
        self.directory = Path(artifact)
        meta = self.directory / "sb-build.json"
        meta = meta if meta.exists() else self.directory.parent / "sb-build.json"
        build = json.loads(meta.read_text())
        if (build["batch_size"], build["sequence_length"]) != (SEQUENCES, 1024):
            raise ValueError("baseline artifact is not B2 x 1024")
        self.program = ModuleUpdate.load(self.directory, device=DEVICE)
        self.programs = {"update": self.program}
        self.identity = dict(arm="A1", lr_scale=self.lr_scale, artifacts={"update": artifact_identity(self.directory)},
                             build=dict(flags=build.get("flags"), lane_sources=build.get("lane_sources")))
        self.coefficients = torch.tensor([1.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.phase = None

    @property
    def parameters(self):
        return self.program.parameters

    @property
    def optimizer_state(self):
        return self.program.optimizer_state

    @property
    def model_state(self):
        return self.program.state

    def load_parent(self, runner_state):
        _copy_into(self.program.parameters, runner_state["parameters"], "parameters")
        _copy_into(self.program.optimizer_state, runner_state["optimizer"], "optimizer")
        _copy_into(self.program.state, runner_state["model_state"], "model_state")
        self.sb_rule.stage(self.program.optimizer_state, self.hp, self.hp["learning_rate"])

    def update(self, x, y, u):
        from megakernel_lab.tinymoe.optimizer import LR
        self.phase = "update"
        self.program.optimizer_state[LR].fill_(lr_of(u) * self.lr_scale if self.lr_scale != 1.0 else lr_of(u))
        self.program.stage((x.to(DEVICE, non_blocking=True),), (y.to(DEVICE, non_blocking=True), self.coefficients))
        self.program.step()
        return self.program.loss

    def diagnostics(self):
        from megakernel_lab.tinymoe.optimizer import CLIP, NORM
        opt = self.program.optimizer_state
        return dict(fast_norm=opt[NORM], fast_clip=opt[CLIP])

    def state(self):
        host = lambda values: {k: v.detach().to("cpu", copy=True) for k, v in values.items()}
        return dict(parameters=host(self.parameters), optimizer=host(self.optimizer_state),
                    model_state=host(self.model_state))

    def load(self, state):
        _copy_into(self.parameters, state["parameters"], "parameters")
        _copy_into(self.optimizer_state, state["optimizer"], "optimizer")
        _copy_into(self.model_state, state["model_state"], "model_state")

    def checksums(self):
        return checksums((("p", self.parameters), ("o", self.optimizer_state)))


class Cadence(Baseline):
    def __init__(self, accumulate, apply, *, arm, kappa, lr_scale=1.0):
        import torch
        from megakernel_lab.operation_compiler import ModuleUpdate
        from megakernel_lab.operation_compiler.variants import persistent_keys
        from megakernel_lab.tinymoe.native import artifact_directory, artifact_identity
        import selective_cadence_rule as cadence_rule
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.rule, self.hp, self.arm, self.kappa = cadence_rule, hyperparameters(), arm, float(kappa)
        # Diagnostic per-token-matched family: the FAST group's LR x lr_scale with its weight decay / lr_scale (decay
        # per token unchanged). The slow group (kappa x mean scheduled LR, base-clock decay) is not affected.
        self.lr_scale = float(lr_scale)
        self.fast_hp = (self.hp if self.lr_scale == 1.0
                        else dict(self.hp, weight_decay=self.hp["weight_decay"] / self.lr_scale))
        directories = {"accumulate": artifact_directory(accumulate), "apply": artifact_directory(apply)}
        builds = {k: json.loads((d / "cadence-build.json").read_text()) for k, d in directories.items()}
        groups = {b["slow_group"] for b in builds.values()}
        if len(groups) != 1 or {k: b["phase"] for k, b in builds.items()} != {"accumulate": "accumulate", "apply": "apply"}:
            raise ValueError("artifacts are not one arm's accumulate/apply pair")
        if any((b["batch_size"], b["sequence_length"]) != (SEQUENCES, 1024) for b in builds.values()):
            raise ValueError("cadence artifacts are not B2 x 1024")
        if len({json.dumps(b["lane_sources"], sort_keys=True) for b in builds.values()}) != 1:
            raise ValueError("phase programs come from different rule sources")
        self.slow_group = groups.pop()
        self.arena = ModuleUpdate.allocate_shared_arena(*directories.values(), device=DEVICE)
        self.programs, canonical = {}, None
        for phase, directory in directories.items():
            manifest = json.loads((directory / "manifest.json").read_text())
            receipt = json.loads((directory / "artifact.json").read_text())
            keys = persistent_keys(manifest, receipt["module_api"])
            if canonical is not None and set(keys.values()) != set(canonical):
                raise ValueError("phase programs disagree on persistent keys")
            bindings = None if canonical is None else {index: canonical[key] for index, key in keys.items()}
            program = ModuleUpdate.load(directory, device=DEVICE, bindings=bindings, arena=self.arena)
            if canonical is None:
                canonical = {key: program.runtime.args[index] for index, key in keys.items()}
            self.programs[phase] = program
        self.program = self.programs["accumulate"]
        for program in self.programs.values():
            for kind in ("parameters", "optimizer_state", "state"):
                left, right = getattr(program, kind), getattr(self.program, kind)
                if left.keys() != right.keys() or any(left[k].data_ptr() != right[k].data_ptr() for k in left):
                    raise ValueError("phase persistent tensors are not shared: " + kind)
        self.identity = dict(arm=arm, slow_group=self.slow_group, kappa=self.kappa, lr_scale=self.lr_scale,
                             artifacts={k: artifact_identity(d) for k, d in directories.items()},
                             build=dict(flags=builds["apply"]["flags"], lane_sources=builds["apply"]["lane_sources"]),
                             unchanged_state_writes={k: json.loads((d / "manifest.json").read_text())
                                                     .get("group_rules", {}).get("unchanged_state_writes")
                                                     for k, d in directories.items()},
                             blocks={k: int(p.runtime.blocks) for k, p in self.programs.items()})
        self.coefficients = torch.tensor([1.0, 0.0], dtype=torch.float32, device=DEVICE)
        self.phase = None

    def load_parent(self, runner_state):
        _copy_into(self.parameters, runner_state["parameters"], "parameters")
        _copy_into(self.optimizer_state, runner_state["optimizer"], "optimizer",
                   allow_new=("acc:", "seen:", "slow_"))
        _copy_into(self.model_state, runner_state["model_state"], "model_state")
        self.rule.stage(self.optimizer_state, self.fast_hp, self.hp["learning_rate"])
        self.rule.stage(self.optimizer_state, self.fast_hp, self.hp["learning_rate"],
                        self.rule.slow_scalars([0.0] * self.rule.WINDOW, self.hp["weight_decay"], self.hp["beta2"], 0.0))

    def update(self, x, y, u):
        window = self.rule.WINDOW
        self.phase = "apply" if u % window == window - 1 else "accumulate"
        program = self.programs[self.phase]
        opt = self.optimizer_state
        # The other fast scalars are constants staged at load (as the study).
        opt[self.rule.LR].fill_(lr_of(u) * self.lr_scale if self.lr_scale != 1.0 else lr_of(u))
        if self.phase == "apply":
            for name, value in self.rule.slow_scalars([lr_of(v) for v in range(u - window + 1, u + 1)],
                                                      self.hp["weight_decay"], self.hp["beta2"], self.kappa).items():
                opt[name].fill_(value)
        program.stage((x.to(DEVICE, non_blocking=True),), (y.to(DEVICE, non_blocking=True), self.coefficients))
        program.step()
        return program.loss

    def diagnostics(self):
        opt, r = self.optimizer_state, self.rule
        return dict(fast_norm=opt[r.NORM], fast_clip=opt[r.CLIP], slow_norm=opt[r.S_NORM], slow_clip=opt[r.S_CLIP])
