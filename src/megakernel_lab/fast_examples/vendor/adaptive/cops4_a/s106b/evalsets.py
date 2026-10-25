"""S106B evaluation conditions over 8193-byte units (target index t predicts byte t + 1).

  normal     the unit as is (every arm, every run).
  swap       distant-history test (8K arms): bytes [0, 2048) are replaced by the FIRST 2048 bytes of another document
             of the same slice (the next one, cyclically); bytes [2048, 8193) are the document's own. Scored region
             Q = targets [4096, 8192): every scored target keeps >= 2048 bytes of its own local history, more than the
             sliding-window stack can reach (6 x 255 = 1530), so W is unaffected by construction and
             loss(swap) - loss(normal) on Q is the value of the document's OWN distant history (>= 2048 bytes back).
  repeat     natural recall subset of Q: targets whose 16-byte string ending at the predicted byte occurs inside the
             distant region [0, 2048) and nowhere earlier in [2048, predicted byte): copying is possible only from
             distant history. Scored under normal and swap.
  passkey    synthetic retention: a random 32-letter string (a-z) overwrites bytes [512, 544); the same string
             ('same') or a fresh one ('diff') overwrites bytes [6144, 6176). Scored: targets predicting bytes
             [6152, 6176) (after 8 letters of cue). recall = CE(diff) - CE(same) in nats/byte.
"""
from __future__ import annotations

import numpy as np

DIST_END = 2048
Q0, Q1 = 4096, 8192
GRAM = 16
PK_A, PK_B, PK_LEN, PK_CUE = 512, 6144, 32, 8


def swap(units):
    out = units.copy()
    out[:, :DIST_END] = np.roll(units, -1, axis=0)[:, :DIST_END]
    return out


def region_mask(n):
    m = np.zeros((n, 8192), dtype=bool)
    m[:, Q0:Q1] = True
    return m


def _gram_hash(b, k=GRAM):
    b = b.astype(np.uint64)
    h = np.zeros(len(b) - k + 1, dtype=np.uint64)
    mult = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over='ignore'):
        for j in range(k):
            h = h * mult + b[j:j + len(h)]
    return h


def repeat_mask(units):
    """bool [N, 8192] over target indices (see module doc)."""
    N = units.shape[0]
    out = np.zeros((N, 8192), dtype=bool)
    for i in range(N):
        hv = _gram_hash(units[i])                          # gram starting at s covers bytes [s, s + 16)
        distant = hv[:DIST_END - GRAM + 1]
        local = hv[DIST_END:]
        _, first, inv = np.unique(local, return_index=True, return_inverse=True)
        is_first = first[inv] == np.arange(len(local))
        in_dist = np.isin(local, distant)
        s = np.arange(len(local)) + DIST_END
        t = s + GRAM - 2                                     # target index predicting the gram's last byte
        ok = is_first & in_dist & (t >= Q0) & (t < Q1)
        out[i, t[ok]] = True
    return out


def passkey(units, seed, same):
    rng = np.random.default_rng(seed)
    out = units.copy()
    letters = np.frombuffer(b'abcdefghijklmnopqrstuvwxyz', dtype=np.uint8)
    for i in range(out.shape[0]):
        a = letters[rng.integers(0, 26, PK_LEN)]
        b = letters[rng.integers(0, 26, PK_LEN)]
        out[i, PK_A:PK_A + PK_LEN] = a
        out[i, PK_B:PK_B + PK_LEN] = a if same else b
    return out


def passkey_mask(n):
    m = np.zeros((n, 8192), dtype=bool)
    m[:, PK_B + PK_CUE - 1:PK_B + PK_LEN - 1] = True
    return m


BUCKETS = ((0, 256), (256, 1024), (1024, 2048), (2048, 4096), (4096, 8192))


def summarize(pos_sum, n):
    """Mean CE overall, per bucket and in 64-target bins from a per-position sum over n units."""
    mean = pos_sum / n
    return dict(nats=float(mean.mean()),
                buckets={f'{a}-{b}': float(mean[a:b].mean()) for a, b in BUCKETS},
                bins64=[round(float(x), 5) for x in mean.reshape(128, 64).mean(1)])
