"""Conventional controls for C-OPS4-A: the unchanged S106B engine (vendored at cops4_a/s106b, tag s106b-final).

Arms (all CUDA-graph captured complete updates, same parameters/optimizer/schedule semantics):
  bf16          the donor policy C2b-long: BF16 autocast, torch.compile default, FlexAttention SWA (practical)
  bf16_ma       the same with torch.compile mode max-autotune-no-cudagraphs
  fp32          matched arithmetic: autocast off, TF32 off, torch.compile default
  fp32_ma       matched arithmetic with max-autotune-no-cudagraphs
The engine's own code is not modified; only the torch.compile mode is selected at construction.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import time

ARMS = {'bf16': ('bf16', None), 'bf16_ma': ('bf16', 'max-autotune-no-cudagraphs'),
        'fp32': ('fp32', None), 'fp32_ma': ('fp32', 'max-autotune-no-cudagraphs')}


@contextmanager
def compile_mode(mode):
    import torch
    original = torch.compile
    if mode is not None:
        torch.compile = functools.partial(original, mode=mode)
    try:
        yield
    finally:
        torch.compile = original


def configure(precision):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return {'tf32': False, 'precision': precision}


def make_engine(a, arm, *, backend='flex', seed=106140):
    """Build (compile + capture) one conventional engine for our Arch a (batch must equal 16384 // ctx)."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b import train as TR
    precision, mode = ARMS[arm]
    da = a.donor()
    if TR.TOKENS // a.ctx != a.batch:
        raise ValueError('S106B engine fixes 16,384 tokens per update')
    configure(precision)
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with compile_mode(mode):
        engine = TR.Engine(da, backend, gain=5.0, compile=True, capture=True, precision=precision)
    info = engine.build(seed=seed, warmup=3)
    info.update(construct_build_s=time.perf_counter() - t0, arm=arm, mode=mode, backend=backend,
                setup_peak_bytes=torch.cuda.max_memory_allocated() - before)
    return engine, info


def load_state(engine, P, state):
    import torch
    with torch.no_grad():
        for n in engine.P:
            engine.P[n].copy_(P[n])
        for n, t in engine.mom.items():
            t.copy_(state[f'momentum:{n}'])
        for n, t in engine.m.items():
            t.copy_(state[f'exp_avg:{n}'])
        for n, t in engine.v.items():
            t.copy_(state[f'exp_avg_sq:{n}'])


def read_state(engine):
    params = {n: t.detach().clone() for n, t in engine.P.items()}
    state = {f'momentum:{n}': t.clone() for n, t in engine.mom.items()}
    state.update({f'exp_avg:{n}': t.clone() for n, t in engine.m.items()})
    state.update({f'exp_avg_sq:{n}': t.clone() for n, t in engine.v.items()})
    return params, state


def stage(engine, x, y, sc):
    import torch
    with torch.no_grad():
        engine.x.copy_(x)
        engine.y.copy_(y)
        engine.sc.copy_(sc)
