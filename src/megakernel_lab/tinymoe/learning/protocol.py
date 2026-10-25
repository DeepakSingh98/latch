"""LATCH-TINYMOE-SMALLBATCH common protocol: data windows, token schedules, hyper-parameter
derivations, initial weights, evaluation and the complete-update clock.

Both arms consume the SAME 1025-token windows in the SAME order: 1024 input positions
and 1024 supplied next-token targets (the final position's lookahead is the window's
last token), so every sequence has exactly 1024 targets in every arm. Arm A update u
uses ordered windows [128u, 128u+128) as 16 microbatches of 8; a Latch B-sequence update
j uses windows [Bj, Bj+B). Horizons, warmup, cosine decay, weight decay and the Adam
beta2 half-life are all defined in TOKENS so that changing the batch does not silently
change them; see derivations below.

Objective of one optimizer update, both arms:
    mean CE over all update targets + 0.02 * mean over forward passes of HF aux(pass)
where aux is the pinned Mixtral load-balancing loss over the pass's tokens (all layers
concatenated). Arm A: each 8-sequence microbatch contributes (CE_mean + 0.02 aux)/16.
Latch B: one pass per update, CE_mean + 0.02 aux. Relative router coefficient = 0.02 in
both. (The published HF Trainer reduction added 0.02*aux undivided per microbatch, i.e.
16 x 0.02 relative; that artefact of its accumulation is deliberately NOT reproduced.)
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

SEQUENCE = 1024
WINDOW = SEQUENCE + 1
ROUTER_COEFFICIENT = 0.02
COMMON = dict(beta1=0.9, eps=1e-6, clip_norm=1.0, warmup_fraction=0.05, final_lr_fraction=0.0,
              precision="FP32 masters/moments; BF16 projections, attention and expert GEMMs; FP32 CE",
              router_coefficient=ROUTER_COEFFICIENT)
# Published TinyMoE TrainingArguments (recovered by LATCH-TINYMOE-GA from training_args.bin).
REFERENCE = dict(sequences_per_update=128, learning_rate=3e-4, weight_decay=0.1, beta2=0.999)
ARMS = {
    "A": dict(backend="conventional", microbatch=8, accumulation=16, sequences=128),
    "B1": dict(backend="latch", microbatch=1, accumulation=1, sequences=1),
    "B2": dict(backend="latch", microbatch=2, accumulation=1, sequences=2),
}


def tokens_per_update(sequences):
    return int(sequences) * SEQUENCE


def beta2_for(sequences):
    """Hold the second-moment half-life fixed in TOKENS: beta2^(B/128) (0.999 at B=128)."""
    return REFERENCE["beta2"] ** (sequences / REFERENCE["sequences_per_update"])


def weight_decay(learning_rate, sequences):
    """Hold decoupled decay per TOKEN fixed at the published lr*wd per 131,072 tokens.

    lr*wd per update = 3e-5 * B/128, so every run shrinks weights by the same factor per
    token regardless of its peak LR or batch (decay timescale ~4.4B tokens)."""
    per_token = REFERENCE["learning_rate"] * REFERENCE["weight_decay"] / tokens_per_update(REFERENCE["sequences_per_update"])
    return per_token * tokens_per_update(sequences) / learning_rate


def hyperparameters(arm, learning_rate):
    spec = ARMS[arm]
    return dict(arm=arm, sequences=spec["sequences"], learning_rate=float(learning_rate),
                weight_decay=weight_decay(learning_rate, spec["sequences"]), beta2=beta2_for(spec["sequences"]),
                **{k: COMMON[k] for k in ("beta1", "eps", "clip_norm", "warmup_fraction", "final_lr_fraction",
                                          "router_coefficient")})


def lr_at(update, peak, sequences, horizon_tokens, warmup_fraction=COMMON["warmup_fraction"]):
    """Cosine with linear warmup over TOKENS, evaluated at the update's token midpoint."""
    x = (update + 0.5) * tokens_per_update(sequences) / horizon_tokens
    if x < warmup_fraction:
        return peak * x / warmup_fraction
    progress = min(1.0, (x - warmup_fraction) / (1.0 - warmup_fraction))
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=repr) + "\n")
    tmp.replace(path)


class Windows:
    """Hash-verified window corpus written by corpus.py ([N, 1025] little-endian uint16)."""

    def __init__(self, root, split, *, order_seed=None, verify=True):
        import numpy as np
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        spec = self.manifest["splits"][split]
        path = self.root / spec["file"]
        if verify and digest(path) != spec["sha256"]:
            raise ValueError("window corpus hash mismatch: " + split)
        self.windows = np.memmap(path, dtype="<u2", mode="r").reshape(-1, WINDOW)
        if len(self.windows) != spec["windows"]:
            raise ValueError("window count mismatch")
        self.order = None
        if order_seed is not None:
            if split != "train":
                raise ValueError("only training windows are permuted")
            self.order = np.random.Generator(np.random.PCG64(int(order_seed))).permutation(len(self.windows))
        self.count = len(self.windows)
        self.manifest_sha256 = digest(self.root / "manifest.json")

    def order_sha256(self):
        return None if self.order is None else hashlib.sha256(self.order.astype("<i8").tobytes()).hexdigest()

    def rows(self, start, count):
        import numpy as np
        if start < 0 or start + count > self.count:
            raise ValueError("window range exhausted; refusing repetition")
        index = np.arange(start, start + count) if self.order is None else self.order[start:start + count]
        return np.asarray(self.windows[index], dtype=np.int64)

    def batch(self, start, count, *, pin=False):
        import torch
        rows = torch.from_numpy(self.rows(start, count))
        x, y = rows[:, :SEQUENCE].contiguous(), rows[:, 1:].contiguous()
        return (x.pin_memory(), y.pin_memory()) if pin else (x, y)


def load_initial(path):
    """Random-initialization parameters written by init_state.py, hash-verified."""
    import torch
    path = Path(path)
    receipt = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if receipt["bytes"] != path.stat().st_size or receipt["sha256"] != digest(path):
        raise ValueError("initial state size/hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["parameters"], receipt


def evaluate(config, parameters, windows, count, *, batch=8, device="cuda"):
    """Final-readout NLL on the first `count` windows of a held-out split.

    Installed Transformers 4.55.4 modules over a zero-copy view of the live weights,
    BF16 autocast, FP32 CE in 512-token chunks; router/aux terms excluded. Identical
    for both arms (megakernel_lab.tinymoe.conventional.evaluate, batched by 8)."""
    import torch
    from megakernel_lab.tinymoe.model import evaluation_view, features
    model = evaluation_view(config, parameters, device)
    total, targets = 0.0, 0
    count = min(count, windows.count)
    with torch.no_grad():
        for start in range(0, count, batch):
            n = min(batch, count - start)
            x, y = windows.batch(start, n)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                final, _, _ = features(model, x)
                h, t = final.reshape(-1, final.shape[-1]), y.reshape(-1)
                for s in range(0, h.shape[0], 512):
                    total += float(torch.nn.functional.cross_entropy(model.lm_head(h[s:s + 512]).float(),
                                                                     t[s:s + 512], reduction="sum"))
            targets += int(t.numel())
    del model
    return dict(nll=total / targets, windows=count, targets=targets)


class Clock:
    """Accumulated synchronized wall seconds of complete optimizer updates."""

    def __init__(self, device):
        import torch
        self.torch, self.device, self.total, self.count = torch, device, 0.0, 0

    def __enter__(self):
        self.torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.torch.cuda.synchronize(self.device)
        self.last = time.perf_counter() - self.start
        if exc[0] is None:
            self.total += self.last
            self.count += 1
        return False
