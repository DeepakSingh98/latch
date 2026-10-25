"""Pinned intermediate-supervision schedule and exit rules; see SOURCE.json."""
from __future__ import annotations
import math

def supervision_lambda(update_index: int, targets_per_update: int,
                       horizon_targets: int, *, initial_weight: float = 0.5,
                       decay_fraction: float = 0.5) -> float:
    """S97F's end-of-update, linearly decaying shared-readout weight.

Use the declared rounded horizon, never a short pilot's stopping update. Each
intermediate CE receives half this weight before division by ``1 + weight``.
"""
    if type(update_index) is not int or update_index < 0:
        raise ValueError("update_index must be a nonnegative integer")
    if type(targets_per_update) is not int or targets_per_update <= 0:
        raise ValueError("targets_per_update must be a positive integer")
    if type(horizon_targets) is not int or horizon_targets <= 0:
        raise ValueError("horizon_targets must be a positive integer")
    if not math.isfinite(initial_weight) or initial_weight < 0:
        raise ValueError("initial_weight must be finite and nonnegative")
    if not math.isfinite(decay_fraction) or not 0 < decay_fraction <= 1:
        raise ValueError("decay_fraction must be in (0, 1]")
    return initial_weight * max(
        0.0, 1.0 - (update_index + 1) * targets_per_update /
        (horizon_targets * decay_fraction))


def supervision_exit_layers(n_layers: int) -> tuple[int, int]:
    """One-based outputs at floor(depth/3), floor(2*depth/3).

The floor rule explicitly extends S51D's rule (then restricted to depths
divisible by three) to the common eight-block backbone. Neither exit is final.
"""
    if type(n_layers) is not int or n_layers < 3:
        raise ValueError("two distinct intermediate exits require at least three layers")
    return n_layers // 3, (2 * n_layers) // 3

