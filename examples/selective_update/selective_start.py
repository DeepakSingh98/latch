# Provenance: new example code (not a donor copy). start_payload follows research/latch_selective_10/cbench.py
# fresh_state at latch commit 6fcad414ad86 (git blob 3b0b4ef2bb75e752c2743cccfe3134ee462b4011): parameters from a
# supplied state, optimizer state zero except both clip coefficients one, model state zero, completed = 0. Removed
# relative to the donor: its study/full-run parent checkpoints (E/M/late states, private), which this example cannot
# ship; the start parameters come from the public pretrained snapshot or from an installed init-state file instead.
"""Start states for the selective-update example (public pretrained snapshot or a random-init file)."""
from __future__ import annotations

from pathlib import Path


def initial_parameters(model_dir, init_path=None):
    """FP32 CPU parameters by canonical name plus a small identity record.

    init_path=None: the public pretrained snapshot in model_dir (hash-verified by the installed load_model).
    init_path=FILE: a file written by `python -m megakernel_lab.tinymoe.learning init-state` (hash-verified receipt).
    """
    from megakernel_lab.tinymoe.learning import protocol
    if init_path:
        initial, receipt = protocol.load_initial(init_path)
        return initial, dict(kind="init_state", path=str(Path(init_path).resolve()), sha256=receipt["sha256"],
                             seed=receipt["seed"])
    from megakernel_lab.tinymoe.model import load_model
    model = load_model(model_dir, device="cpu", attention="sdpa")
    initial = {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters()}
    del model
    identity = Path(model_dir) / "latch_identity.json"
    return initial, dict(kind="public_pretrained", model_dir=str(Path(model_dir).resolve()),
                         snapshot_identity_sha256=protocol.digest(identity))


def start_payload(runner, initial):
    """cbench.fresh_state with supplied parameters: a runner state at a window boundary (completed = 0)."""
    import torch
    r = runner.rule
    params = {k: v.detach().to("cpu", copy=True) for k, v in runner.parameters.items()}
    if set(params) != set(initial):
        raise ValueError("start parameter names differ from the program")
    params = {k: initial[k].to(params[k].dtype).clone() for k in params}
    optimizer = {k: (torch.ones_like(v, device="cpu") if k in (r.CLIP, r.S_CLIP) else torch.zeros_like(v, device="cpu"))
                 for k, v in runner.optimizer_state.items()}
    model_state = {k: torch.zeros_like(v, device="cpu") for k, v in runner.model_state.items()}
    return dict(parameters=params, optimizer=optimizer, model_state=model_state)


def zero_optimizer(initial):
    """The same zero AdamW state in the native naming, for the conventional counterpart's load_parent."""
    import torch
    out = {}
    for name, p in initial.items():
        out["step:" + name] = torch.zeros((), dtype=torch.int64)
        out["exp_avg:" + name] = torch.zeros_like(p, dtype=torch.float32)
        out["exp_avg_sq:" + name] = torch.zeros_like(p, dtype=torch.float32)
    return out
