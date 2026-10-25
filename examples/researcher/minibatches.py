"""Minibatches for the workflow: synthetic bytes, or the prepared text written by ``examples/families/prepare_data.py``.

The prepared layout (``corpus.npy``, ``documents.jsonl``, ``SLICES.json``) is the one every text example shares.
Training windows come only from documents that are not reserved as development/held-out; the development
documents supply evaluation batches. The window order is a seeded permutation, so both execution paths see the
same bytes at the same step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import Config, synthetic_batch


class Synthetic:
    kind = 'synthetic'

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def identity(self):
        return dict(kind=self.kind, seed=self.cfg.seed)

    def batch(self, step):
        return synthetic_batch(self.cfg, step)

    def dev_batches(self, max_batches=8):
        return []


class PreparedText:
    kind = 'prepared-text'

    def __init__(self, cfg: Config, root, order_seed=20261010):
        self.cfg, self.root = cfg, Path(root)
        for name in ('corpus.npy', 'documents.jsonl', 'SLICES.json'):
            if not (self.root / name).is_file():
                raise ValueError(f'{self.root} is not a prepared text directory (missing {name}); '
                                 'see examples/families/prepare_data.py text')
        self.corpus = np.load(self.root / 'corpus.npy', mmap_mode='r')
        rows = [json.loads(line) for line in (self.root / 'documents.jsonl').read_text().splitlines() if line.strip()]
        slices = json.loads((self.root / 'SLICES.json').read_text())
        reserved = {seg['doc'] for key in ('development', 'heldout') for seg in slices.get(key, [])}
        span = cfg.seq_len + 1
        starts = []
        for row in rows:
            if row['doc'] in reserved:
                continue
            count = row['bytes'] // span
            if count:
                starts.append(row['offset'] + span * np.arange(count, dtype=np.int64))
        self.windows = np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)
        if len(self.windows) < cfg.batch:
            raise ValueError('corpus holds fewer training windows than one minibatch')
        self.order = np.random.default_rng(order_seed).permutation(len(self.windows))
        self.order_seed = order_seed
        self.dev_starts = [seg['start'] for seg in slices.get('development', [])]
        self.updates_available = len(self.windows) // cfg.batch

    def identity(self):
        manifest = self.root / 'manifest.json'
        return dict(kind=self.kind, root=str(self.root), windows=int(len(self.windows)),
                    updates_available=self.updates_available, order_seed=self.order_seed,
                    manifest=json.loads(manifest.read_text()) if manifest.exists() else None)

    def _window(self, start):
        chunk = torch.from_numpy(np.array(self.corpus[start:start + self.cfg.seq_len + 1], dtype=np.int64))
        return chunk[:-1], chunk[1:]

    def batch(self, step):
        if step >= self.updates_available:
            raise ValueError(f'step {step} exceeds the {self.updates_available} non-repeating updates in this corpus')
        picks = self.order[step * self.cfg.batch:(step + 1) * self.cfg.batch]
        pairs = [self._window(int(self.windows[i])) for i in picks]
        return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])

    def dev_batches(self, max_batches=8):
        """Evaluation minibatches from the reserved development documents (first bytes of each)."""
        batches, pairs = [], []
        for start in self.dev_starts:
            pairs.append(self._window(int(start)))
            if len(pairs) == self.cfg.batch:
                batches.append((torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])))
                pairs = []
                if len(batches) == max_batches:
                    break
        return batches


def source(cfg: Config, root=None):
    return PreparedText(cfg, root) if root else Synthetic(cfg)
