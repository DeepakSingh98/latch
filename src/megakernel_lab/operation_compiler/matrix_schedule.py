"""Opt-in matrix-product execution rules for flat common compose programs (LATCH-TINYMOE-MATRIX).

Selected through the normal build API: ``CapturedModule.build(..., matrix_schedule="underfilled")``.
Omitting it (None) leaves lowering and generated source unchanged.

"underfilled" addresses provider pops4b TMA + wgmma products whose few output tiles leave most of the cooperative
grid idle while each busy CTA streams a long contraction (typically weight gradients dW = dY^T X over the tokens).
Both rules are structural (shape, tile count, contraction length, layout, precision policy); none consults a model
or parameter name.

* ``tdeep`` (existing release-step-14 rule, matrix_adapters.bodies.deepen): a product whose tiles fill at most half
  the grid keeps up to eight 64-wide k stages of its operand ring in flight. Tile shape, k order and rounding are
  unchanged (bitwise identical results).
* ``tsplit`` (ops/matrix_tma_split.py, practical precision only): a long-K product whose tiles fill at most a quarter
  of the grid runs as deterministic K slices on the same body (P6-B SPLIT mode, FP32 slice tiles), followed by a
  ``splitk_reduce`` phase that adds the slices in ascending order and rounds once to BF16. Only the FP32 association
  of the K sum changes, as the practical policy permits; results are deterministic and grid independent. Very long
  contractions (K >= 8192, e.g. a vocabulary-scale input gradient) kept by the chunked P5C/P8WS bodies use the same
  rule with equal slices of at most 2048 (the chunk length of the step-3 long-K repair).
* short single-wave contractions (K <= 512, tiles within one wave): the provider's step-13 TMA choice is replaced by
  the admitted cp.async wgmma body of the same bitwise class (identical values, shorter per-task fixed path).

"exact_split" (LATCH-B1-FAST) is "exact" plus ``tsplitx``: a very long contraction kept by a chunked P5C/P8WS body
(K >= 8192, PK-stage chunks summed in chunk order) runs as slices of exactly one chunk each on the TMA split body,
the last slice padded with zero-filled stages (matrix_tma_split.choose_long_exact); the chunk partials, their
ascending FP32 sum and the single BF16 rounding are the chunked body's, so results are bit for bit unchanged.

"exact" (LATCH-B1-FAST) selects only the bitwise rules of "underfilled": ``tdeep`` and the short single-wave
contraction body (and their joint tile planning). ``tsplit`` is not applied, so every product keeps the FP32
association of the unsplit body: results are bit for bit those of the unscheduled build.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

SCHEDULES = (None, "underfilled", "exact", "exact_split")
_SCHEDULE = ContextVar("latch_matrix_schedule", default=None)
_FEATURES = {"underfilled": "+tdeep+tsplit",  # short single-wave contractions: matrix_tma_split.short_contraction
             "exact": "+tdeep",  # LATCH-B1-FAST: the bitwise subset only (tdeep + short single-wave contractions)
             "exact_split": "+tdeep+tsplitx"}  # LATCH-B1-FAST: exact plus chunk-aligned slices of chunked long-K bodies
VERSION = 1


def validate(value):
    if value not in SCHEDULES:
        raise ValueError(f"matrix_schedule must be one of {SCHEDULES}")
    return value


def current():
    return _SCHEDULE.get()


@contextmanager
def scope(value):
    token = _SCHEDULE.set(validate(value))
    try:
        yield
    finally:
        _SCHEDULE.reset(token)


def features():
    """Extra group-rule features for the standalone flat common planner."""
    return _FEATURES.get(_SCHEDULE.get(), "")
