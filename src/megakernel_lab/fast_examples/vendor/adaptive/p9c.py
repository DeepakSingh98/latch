"""P9-C harness: the P8-C harness (p8c.py, frozen C-OPS4-A fixtures, P8-C POLICY gates unchanged) plus P9-C tasks.

Task names route harness code only; the compiler never sees them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from megakernel_lab.fast_examples.vendor.adaptive import p8c
from megakernel_lab.fast_examples.vendor.adaptive.c4a import save

HERE = Path(__file__).resolve().parent










def task_fusion_check(spec, out):
    """CPU: capture with window_fusion on and off, replay both captured graphs on identical inputs, and compare
    loss, gradients, new parameters and optimizer state bitwise; records the rewrite's groups/refusals."""
    import torch
    torch.set_num_threads(1)
    from megakernel_lab.fast_examples.vendor.adaptive import c4a
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import capture
    a = c4a.recipe(spec['recipe'])
    P, inputs = c4a.fixture(a, seed=spec.get('seed', 106140))
    rows, caps = {}, {}
    for mode in ('off', 'auto'):
        t0 = time.perf_counter()
        cap = capture(a, P, inputs, storage_mode='save_body_tape', window_fusion=mode,
                      **spec.get('capture_options', {}))
        meta = cap.ir.metadata.get('init_attention_tape', {})
        rows[mode] = {'capture_seconds': time.perf_counter() - t0,
                      'fusion': {k: v for k, v in (meta.get('fusion') or {}).items()},
                      'init_targets': dict(sorted(__import__('collections').Counter(
                          str(op.target) for op in cap.ir.regions['init_saved'].operations).items()))
                      if 'init_saved' in cap.ir.regions else None}
        caps[mode] = cap
    count = torch.tensor(spec.get('count', a.chunks), dtype=torch.int64)
    results = {}
    for mode, cap in caps.items():
        flat = (*cap.example_args[:-1], count)
        t0 = time.perf_counter()
        results[mode] = cap.unpack(cap.graph_step(*flat))
        rows[mode]['graph_step_seconds'] = time.perf_counter() - t0
    r0, r1 = results['off'], results['auto']
    cmp = {'loss_equal': bool(torch.equal(r0['loss'], r1['loss'])), 'loss': [float(r0['loss']), float(r1['loss'])]}
    for key in ('grads', 'params', 'optimizer_state'):
        cmp[key + '_unequal'] = [k for k in r0[key] if not torch.equal(r0[key][k], r1[key][k])]
    rows['compare'] = cmp
    applied = len(rows['auto']['fusion'].get('groups', []))
    rows['passed'] = cmp['loss_equal'] and not any(cmp[k + '_unequal'] for k in ('grads', 'params', 'optimizer_state')) \
        and applied >= spec.get('expect_groups', 1)
    save(out / 'fusion-check.json', rows)
    return rows






TASKS = dict(p8c.TASKS, fusion_check=task_fusion_check)
GPU_TASKS = dict(p8c.GPU_TASKS)
