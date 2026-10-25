# Provenance: copied from the Latch repository, commit df184a5f2579
# (branch codex/latch-gradlev), research/latch_gradlev/gl/latch_loop.py (git blob 8d7228875b00dec74b206d2bdfda3ebfc1bb36a1).
# Changes: none to the code (byte-identical below this header); imports megakernel_lab.operation_compiler.inference from the installed wheel.
"""The dependent GradLev adaptation loop captured through Latch's normal loop-inference API.

capture_loop_inference(init, body, output, params, inputs, count, max_steps, state):
  params  = frozen loop constants {b1, b2, lnf_w, lnf_b, E (BF16)}
  state   = fast weights {W1, W2} (FP32, committed after the loop; persists across calls)
  inputs  = per-position prefill rows X, U [T,B,d], targets [T,B], eta1/eta2 [T,B]
  carry   = {t, nll [T,B]} (restarts per call)
The body is deploy.step_math, the same function the conventional backends execute; every
iteration scores position t, forms the costates and writes both matrices before t+1 reads them.
"""
from __future__ import annotations

import torch

from .deploy import step_math
from .model import D, M, V


def loop_functions(T, head='mixed'):
    def init(params, X, U, TGT, E1, E2):
        return {'t': torch.zeros((), dtype=torch.int64, device=X.device),
                'nll': torch.zeros(X.shape[0], X.shape[1], dtype=torch.float32, device=X.device)}

    def body(params, carry, state, X, U, TGT, E1, E2):
        t = carry['t']
        i = t.reshape(1)
        x = X.index_select(0, i)[0]
        u = U.index_select(0, i)[0]
        tgt = TGT.index_select(0, i)[0]
        e1 = E1.index_select(0, i)[0]
        e2 = E2.index_select(0, i)[0]
        nll, W1, W2 = step_math(state['W1'], state['W2'], params, x, u, tgt, e1, e2, head)
        hit = (torch.arange(X.shape[0], device=X.device) == t)[:, None]
        return {'t': t + 1, 'nll': torch.where(hit, nll[None, :], carry['nll'])}, {'W1': W1, 'W2': W2}

    def output(params, carry, X, U, TGT, E1, E2):
        return carry['nll']

    return init, body, output


def random_model(d=D, m=M, v=V, seed=0):
    """Seeded random loop constants/state of arbitrary shape (reuse checks; values are not a model)."""
    gen = torch.Generator().manual_seed(seed)
    params = {'b1': torch.randn(m, generator=gen) * 0.02, 'b2': torch.randn(d, generator=gen) * 0.02,
              'lnf_w': 1 + 0.1 * torch.randn(d, generator=gen), 'lnf_b': 0.02 * torch.randn(d, generator=gen),
              'E': (torch.randn(v, d, generator=gen) * 0.05).to(torch.bfloat16)}
    W1 = torch.randn(d, m, generator=gen) / d ** 0.5
    W2 = torch.randn(m, d, generator=gen) / m ** 0.5
    return params, W1, W2


def example(B=1, T=1024, device='cpu', p=None, dims=None):
    """Example bindings (real GPT-2 values when p is given, else seeded random of the given dims)."""
    gen = torch.Generator().manual_seed(1)
    if p is not None:
        from .deploy import consts_from, initial_state
        params = consts_from(p, device)
        state = initial_state(p, device, B)
        d, m, v = D, M, V
    else:
        d, m, v = dims or (D, M, V)
        params, W1, W2 = random_model(d, m, v)
        state = {'W1': W1.expand(B, d, m).clone(), 'W2': W2.expand(B, m, d).clone()}
        params = {k: t.to(device) for k, t in params.items()}
        state = {k: t.to(device) for k, t in state.items()}
    inputs = (torch.randn(T, B, d, generator=gen).to(device), torch.randn(T, B, d, generator=gen).to(device),
              torch.randint(0, v, (T, B), generator=gen).to(device),
              torch.full((T, B), 0.9 / d, device=device), torch.full((T, B), 0.9 / m, device=device))
    return params, state, inputs


def capture(B=1, T=1024, p=None, device='cpu', dims=None, head='mixed'):
    from megakernel_lab.operation_compiler.inference import capture_loop_inference
    params, state, inputs = example(B, T, device, p, dims)
    init, body, output = loop_functions(T, head)
    return capture_loop_inference(init, body, output, params, inputs, torch.tensor(T, device=device),
                                  max_steps=T, state=state)


def build_artifact(p, out_dir, *, T=1024, B=1, lower_options=None, dims=None, head='mixed'):
    """Capture (CPU fake tensors) and prepare one artifact; returns (directory, receipt)."""
    import time
    from pathlib import Path
    from megakernel_lab.operation_compiler.inference import prepare_inference
    t0 = time.perf_counter()
    cap = capture(B, T, p=p, device='cpu', dims=dims, head=head)
    capture_s = time.perf_counter() - t0
    out_dir = Path(out_dir)
    receipt = prepare_inference(cap, out_dir / 'artifact', cache_dir=out_dir / 'cache',
                                lower_options=dict(lower_options or {'loop_schedule': 'vector_resident'}))
    receipt['capture_wall_seconds'] = capture_s
    receipt['setup_wall_seconds'] = time.perf_counter() - t0
    return out_dir / 'artifact', receipt
