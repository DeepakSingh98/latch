"""Frozen numerical comparisons for independent optimizer/model references.

Extracted from the accepted Latch source; policies and comparison arithmetic
are unchanged. See provenance/NUMERICAL_REFERENCE_PACKAGING.json.
"""
from pathlib import Path
import json

import torch

POLICY = json.loads(Path(__file__).with_name("NUMERICAL_POLICY.json").read_text())
MIXED_POLICY = json.loads(Path(__file__).with_name("STAGE3_POLICY.json").read_text())["mixed_precision"]


def compare(actual, expected, category):
    policy = POLICY[category]
    if tuple(actual.shape) != tuple(expected.shape):
        return {'passed': False, 'shape_mismatch': [list(actual.shape), list(expected.shape)], 'policy': category}
    a, e = actual.detach().double(), expected.detach().double()
    d = (a - e).abs()
    bound = policy['atol'] + policy['rtol'] * e.abs()
    ratio = d / bound
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    return {'passed': finite and bool((ratio <= 1).all()), 'finite': finite,
            'max_ratio': float(ratio.max()) if a.numel() else 0.0,
            'max_abs': float(d.max()) if a.numel() else 0.0,
            'rms': float((d.square().mean()).sqrt()) if a.numel() else 0.0,
            'elements': a.numel(), 'policy': category}


def compare_arithmetic(actual, expected, category, arithmetic='float32'):
    if arithmetic == 'float32':
        return compare(actual, expected, category)
    if arithmetic != 'bfloat16':
        raise ValueError('unknown transformer comparison arithmetic')
    policy = MIXED_POLICY[category]
    if actual.shape != expected.shape:
        return {'passed': False, 'shape_mismatch': [list(actual.shape), list(expected.shape)]}
    a, e = actual.detach().double(), expected.detach().double()
    delta = (a - e).abs()
    ratio = delta / (policy['atol'] + policy['rtol'] * e.abs())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    return {'passed': finite and bool((ratio <= 1).all()), 'finite': finite,
            'max_ratio': float(ratio.max()) if a.numel() else 0.0,
            'max_abs': float(delta.max()) if a.numel() else 0.0,
            'rms': float(delta.square().mean().sqrt()) if a.numel() else 0.0,
            'elements': a.numel(), 'policy': 'STAGE3_POLICY.mixed_precision.' + category}
