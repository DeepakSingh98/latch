"""Pinned 25M recipe and schedules, with a tiny API smoke fixture. See SOURCE.json."""
from __future__ import annotations
import copy
import math
from .auxiliary import supervision_lambda

_COMMON = dict(arm="moe_intermediate_supervision", precision="bf16", optimizer="muon", optimizer_betas=[0.9, 0.95],
               weight_decay=0.1, optimizer_eps=1e-8, clip_norm=1.0, muon_gain=20.0, accumulation=1,
               head_chunk_tokens=1024, deterministic=True, allow_tf32=False)
RECIPES = {
    "25m": dict(_COMMON, size="25m",
                model=dict(d_model=384, n_layers=3, n_heads=6, seq_len=2048, vocab_size=50257, routing_group=32,
                           arm="moe_intermediate_supervision", seed=1729),
                target_tokens=500039680, schedule_tokens=500039680, learning_rate=0.0012,
                minimum_lr=0.00011999999999999999, warmup_tokens=5046272),
    "50m": dict(_COMMON, size="50m",
                model=dict(d_model=512, n_layers=8, n_heads=8, seq_len=2048, vocab_size=50257, routing_group=32,
                           arm="moe_intermediate_supervision", seed=1729),
                target_tokens=1000013824, schedule_tokens=1000013824, learning_rate=0.0008,
                minimum_lr=8e-05, warmup_tokens=10027008),
}
# Development fixture only (never timed or qualified): the 25M recipe's schedule with the tiny adapter model.
RECIPES["tiny"] = dict(RECIPES["25m"], size="tiny",
                       model=dict(d_model=128, n_layers=3, n_heads=2, seq_len=128, vocab_size=97, routing_group=32,
                                  arm="moe_intermediate_supervision", seed=1729))
TOKENS_PER_UPDATE = 32 * 2048


def recipe(size="25m", **model_overrides):
    """A copy of the recipe; ``model_overrides`` only for declared nearby-configuration transfer tests."""
    if size not in ('25m', 'tiny'):
        raise ValueError('only the 25M recipe and explicitly unqualified tiny fixture are included')
    r = copy.deepcopy(RECIPES[size])
    r["model"].update(model_overrides)
    return r


def learning_rate(step, config, tokens_per_update):
    """Absolute token clock, independent of pilot stop and resume boundaries."""
    peak, minimum = float(config.get("learning_rate", 3e-4)), float(config.get("minimum_lr", 3e-5))
    warmup, horizon = int(config.get("warmup_tokens", 1_048_576)), int(config.get("schedule_tokens", 200_000_000))
    tokens = (step + 1) * tokens_per_update
    if not 0 <= warmup < horizon or not 0 <= minimum <= peak:
        raise ValueError("invalid learning-rate schedule")
    if tokens <= warmup:
        return peak * tokens / warmup
    progress = min(1.0, (tokens - warmup) / (horizon - warmup))
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def aux_weight(r, step):
    return supervision_lambda(step, TOKENS_PER_UPDATE, int(r['schedule_tokens']))

def lr(r, step):
    return learning_rate(step, r, TOKENS_PER_UPDATE)
