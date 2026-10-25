# Provenance: adapted from the Latch repository, commit df184a5f2579 (branch codex/latch-gradlev),
# research/latch_gradlev/gl/data.py (git blob 1d4be220207663c88bcec2a359d5d3f757e10fe4).
# Kept unchanged: WINDOW, T, blocks_from() (block construction: [EOT] + w[0:1023] -> w[0:1024],
# [EOT] + w[1024:2047] -> w[1024:2048]) and the EVAL protocol (dev = validation windows [0, 128),
# test = test windows [0, 512)).
# Changed: windows() reads the locally rebuilt public files written by assets.build_corpus() (window
# counts from data/data.json) instead of the private corpus; eval_blocks() takes an optional block
# limit and refuses (instead of silently shortening) when fewer windows were prepared.
# Removed: train_order() and train_batch() (helper training is not part of this example).
"""Blocks from the pinned 2049-token windows (one document per window).

Each window w[0:2049] yields two independent blocks of T = 1024 scored targets:
    block 0: inputs [EOT] + w[0:1023],    targets w[0:1024]
    block 1: inputs [EOT] + w[1024:2047], targets w[1024:2048]
(w[2048] is unused). EOT (50256) is a context-only BOS; every block starts with a fresh
trunk context and fresh fast state (paper reset policy: document / context boundaries).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .assets import data_dir
from .model import EOT

WINDOW = 2049
T = 1024


def windows(split):
    manifest = json.loads((data_dir() / 'data.json').read_text()) if (data_dir() / 'data.json').exists() else None
    if manifest is None or split not in manifest['splits']:
        raise SystemExit(f'no prepared {split} windows under {data_dir()}; run quickstart.sh prepare')
    n = manifest['splits'][split]['windows']
    return np.memmap(data_dir() / f'{split}.u16', dtype='<u2', mode='r', shape=(n, WINDOW))


def blocks_from(window_rows):
    """[N, 2049] uint16 -> (inputs [2N, 1024], targets [2N, 1024]) int64."""
    w = torch.from_numpy(np.asarray(window_rows, dtype=np.int64))
    n = w.shape[0]
    eot = torch.full((n, 1), EOT, dtype=torch.int64)
    a_in = torch.cat([eot, w[:, 0:T - 1]], 1)
    b_in = torch.cat([eot, w[:, T:2 * T - 1]], 1)
    inputs = torch.stack([a_in, b_in], 1).reshape(2 * n, T)
    targets = torch.stack([w[:, 0:T], w[:, T:2 * T]], 1).reshape(2 * n, T)
    return inputs, targets


# Frozen evaluation protocol (fixed before any training run; see the donor SPEC.md):
#   dev  = validation windows [0, 128)  -> 256 blocks, 262,144 targets (selection/diagnostics only)
#   test = test windows [0, 512)        -> 1,024 blocks, 1,048,576 targets (final comparison only)
EVAL = {'dev': ('validation', 0, 128), 'test': ('test', 0, 512)}


def eval_blocks(name, blocks=None):
    """The protocol's blocks, optionally only the first `blocks` of them (same order as the donor)."""
    split, lo, hi = EVAL[name]
    if blocks:
        hi = min(hi, lo + (int(blocks) + 1) // 2)
    w = windows(split)
    if w.shape[0] < hi:
        raise SystemExit(f'{name} needs {split} windows [{lo}, {hi}) but only {w.shape[0]} were prepared; '
                         'rerun prepare with more windows in a new GRADLEV_ROOT')
    inputs, targets = blocks_from(w[lo:hi])
    if blocks:
        inputs, targets = inputs[:blocks], targets[:blocks]
    return inputs, targets
