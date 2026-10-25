# Provenance: copied from the private Latch repository (branch codex/smelt-latch) at commit 2c9ddb2e3d51
# (branch codex/smelt-latch). The two donor files below are byte-identical at LATCH-TINYMOE-BATCH-FRONTIER commit
# f9268c21b4dc (imported read-only into the SMELT lane at afd97c4eb107):
#   research/latch_tinymoe_batch_frontier/protocol.py  (blob 56521b5276eb, sha256 9ee0499a3329...)
#     kept unchanged: SEQUENCE, WINDOW, ROUTER_COEFFICIENT, COMMON, REFERENCE, ARMS (incl. the C<B>/L<B> loop),
#     AUX_GROUP_SEQUENCES, tokens_per_update, beta2_for, weight_decay, hyperparameters.
#     removed (not used by this example): arm_spec, lr_at, digest, write_json, Windows (window corpus reader),
#     load_initial, evaluate, Clock, schedule_spec/schedule_lr/cumulative_shrinkage and their imports.
#   research/latch_tinymoe_batch_frontier/observe.py   (blob 9a314272cc86, sha256 4bcfbd8ca3c9...)
#     kept unchanged: adam_slots (Latch-format exp_avg:/exp_avg_sq:/step: optimizer slots of a torch AdamW).
#     removed: the observation/audit driver (observe_conventional/observe_latch, snapshots, CLI).
# No constant, derivation or optimizer-slot format was changed. The donor module docstring is kept below.
"""LATCH-TINYMOE-BATCH-FRONTIER common protocol (from LATCH-TINYMOE-SMALLBATCH c1ef72ca, unchanged
arithmetic for the A/B1/B2 reference arms): data windows, token schedules, hyper-parameter
derivations, initial weights, evaluation and the complete-update clock.

New arms (this lane): effective batches of 8 and 32 (optionally 16) sequences x 1024 tokens,
each on two backends: C<B> = compiled conventional PyTorch (compiled-grouped TinyMoE backend,
optional microbatching with gradient accumulation), L<B> = Latch generated program (one
optimizer update per program step, no accumulation, streamed update schedule).

ROUTER AUXILIARY GROUPING (explicit, part of the learning recipe): for every new arm the
HF load-balancing aux is computed over GROUPS of AUX_GROUP_SEQUENCES = 8 consecutive
sequences of the update (8,192 tokens, all layers concatenated within the group) and the
update objective is
    mean CE over all update targets + 0.02 * mean over the update's groups of aux(group).
This is exactly the B128 reference's objective (A: 16 microbatches of 8 = 16 groups of 8).
B8 has one group; B32 has four. The grouping is a property of the recipe, NOT of the
executor: a conventional microbatch of m sequences (m a multiple of 8) evaluates aux per
8-sequence group inside the microbatch, and the Latch B32 program captures the same
grouped aux in one pass, so backend comparisons at one batch use identical objectives
and changing conventional microbatching does not change the objective. The B2 reference
used aux over its 2-sequence pass (group 2); B1 group 1.

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

SEQUENCE = 1024
WINDOW = SEQUENCE + 1
ROUTER_COEFFICIENT = 0.02
COMMON = dict(beta1=0.9, eps=1e-6, clip_norm=1.0, warmup_fraction=0.05, final_lr_fraction=0.0,
              precision="FP32 masters/moments; BF16 projections, attention and expert GEMMs; FP32 CE",
              router_coefficient=ROUTER_COEFFICIENT)
# Published TinyMoE TrainingArguments (recovered by LATCH-TINYMOE-GA from training_args.bin).
REFERENCE = dict(sequences_per_update=128, learning_rate=3e-4, weight_decay=0.1, beta2=0.999)
ARMS = {
    "A": dict(backend="conventional", microbatch=8, accumulation=16, sequences=128, aux_group=8),
    "B1": dict(backend="latch", microbatch=1, accumulation=1, sequences=1, aux_group=1),
    "B2": dict(backend="latch", microbatch=2, accumulation=1, sequences=2, aux_group=2),
}
AUX_GROUP_SEQUENCES = 8
for _b in (8, 16, 32, 128):  # C128 = the B128 recipe on faster conventional microbatching (same objective)
    # Conventional default microbatch = whole update; the measured fastest passing choice is
    # passed explicitly (train.py --microbatch) and recorded in the run identity.
    ARMS[f"C{_b}"] = dict(backend="conventional", microbatch=_b, accumulation=1, sequences=_b,
                          aux_group=AUX_GROUP_SEQUENCES)
    ARMS[f"L{_b}"] = dict(backend="latch", microbatch=_b, accumulation=1, sequences=_b,
                          aux_group=AUX_GROUP_SEQUENCES)


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


# ---------------------------------------------------------------- from observe.py (unchanged)
def adam_slots(model, optimizer):
    import torch
    slots = {}
    for name, p in model.named_parameters():
        live = optimizer.state.get(p, {})
        for key in ("exp_avg", "exp_avg_sq"):
            slots[key + ":" + name] = live[key].detach().to("cpu", copy=True) if key in live else torch.zeros(p.shape)
        slots["step:" + name] = torch.tensor(int(live.get("step", 0)), dtype=torch.int64)
    return slots
