"""C-OPS4-A GPU arm processes: one fresh process per arm (timing, memory, setup, real-text training, evaluation).

Invoked by c4a.py as ``python -B c4a_gpu.py '<json spec>'``; writes the JSON result to spec['out'].
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CORPUS = '/s106b/corpus/fwe-long-s106b-v1'


def device_used():
    import torch
    free, total = torch.cuda.mem_get_info()
    return total - free


def synthetic_batches(a, n=4, seed=31000):
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import synthetic_tokens
    return [synthetic_tokens(a, seed + i).cuda() for i in range(n)]


def time_loop(stage, step, batches, sc, warmup, iters):
    import torch
    for i in range(warmup):
        stage(batches[i % len(batches)], sc)
        step()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    w0 = time.perf_counter()
    for i in range(iters):
        stage(batches[i % len(batches)], sc)
        events[i][0].record()
        step()
        events[i][1].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - w0
    ms = sorted(s.elapsed_time(e) for s, e in events)
    return {'median_ms': ms[len(ms) // 2], 'min_ms': ms[0], 'max_ms': ms[-1], 'mean_ms': sum(ms) / len(ms),
            'iters': iters, 'loop_wall_s': wall}


def native_arm(spec, a):
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows
    used0 = device_used()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    native = c4a.Native(spec['folder'], blocks=spec.get('blocks'))
    setup = time.perf_counter() - t0
    sc = schedule_rows(64, 3e-3, 1.0)[7].cuda()

    def stage(tokens, row):
        native.x.copy_(tokens[:, :-1])
        native.y.copy_(tokens[:, 1:])
        native.sc.copy_(row)
    timing = time_loop(stage, native.runtime.step, synthetic_batches(a), sc, spec.get('warmup', 5), spec.get('iters', 30))
    profile = None
    if spec.get('profile'):
        profile = native.runtime.profile_step()
        torch.cuda.synchronize()
    return {'arm': 'native', 'setup': {'gpu_runtime_load_s': setup, 'cpu_build': native.receipt.get('build'),
                                       'cpu_capture_s': native.receipt.get('capture_seconds'),
                                       'cpu_lower_s': native.receipt.get('lower', {}).get('seconds')},
            'timing': timing, 'blocks': native.runtime.blocks,
            'memory': {'torch_peak_allocated': torch.cuda.max_memory_allocated(),
                       'torch_peak_reserved': torch.cuda.max_memory_reserved(),
                       'runtime_allocated_bytes': native.runtime.allocated_bytes,
                       'device_used_delta': device_used() - used0},
            'profile': profile, 'loss_last': native.runtime.loss.item()}


def conv_arm(spec, a):
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a_conv
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows
    used0 = device_used()
    engine, info = c4a_conv.make_engine(a, spec['arm'], backend=spec.get('backend', 'flex'))
    sc = schedule_rows(64, 3e-3, 1.0)[7].cuda()

    def stage(tokens, row):
        c4a_conv.stage(engine, tokens[:, :-1], tokens[:, 1:], row)
    torch.cuda.reset_peak_memory_stats()
    timing = time_loop(stage, engine.step, synthetic_batches(a), sc, spec.get('warmup', 5), spec.get('iters', 30))
    return {'arm': spec['arm'], 'setup': info, 'timing': timing,
            'memory': {'torch_peak_allocated_timed': torch.cuda.max_memory_allocated(),
                       'setup_peak_allocated': info['setup_peak_bytes'],
                       'torch_peak_reserved': torch.cuda.max_memory_reserved(),
                       'device_used_delta': device_used() - used0},
            'loss_last': engine.loss.item()}


def training_data(a, H, seed):
    import numpy as np
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b.corpus import Corpus
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b.train import segments
    corpus = Corpus(CORPUS)
    docs = corpus.stream(seed)
    units = corpus.units[docs[:2 * H]]
    seq = segments(units, a.ctx).reshape(H, a.batch, a.ctx + 1)
    return np.ascontiguousarray(seq), corpus


def train_native(spec, a):
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import schedule_rows
    H = spec['updates']
    data, corpus = training_data(a, H, spec['seed'])
    native = c4a.Native(spec['folder'])
    table = schedule_rows(H, spec['lr'], spec['eta']).cuda()
    gpu = torch.from_numpy(data).cuda().long()
    losses = torch.zeros(H, device='cuda')
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(H)]
    for i in range(H):
        native.x.copy_(gpu[i, :, :-1])
        native.y.copy_(gpu[i, :, 1:])
        native.sc.copy_(table[i])
        ev[i][0].record()
        native.runtime.step()
        ev[i][1].record()
        losses[i] = native.runtime.loss
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in ev]
    lh = losses.cpu().numpy()
    torch.save({k: v.detach().cpu() for k, v in native.params.items()}, spec['params_out'])
    return dict(arm='native', updates=H, train_seconds=sum(ms) / 1000, median_update_ms=float(np.median(ms)),
                loss_curve=[float(v) for v in lh[:: max(1, H // 64)]], final_train_loss=float(lh[-32:].mean()),
                finite=bool(np.isfinite(lh).all()), schedule_sha=c4a.hashlib.sha256(table.cpu().numpy().tobytes()).hexdigest())


def train_conv(spec, a):
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a
    from megakernel_lab.fast_examples.vendor.adaptive import c4a_conv
    H = spec['updates']
    data, corpus = training_data(a, H, spec['seed'])
    engine, info = c4a_conv.make_engine(a, spec['arm'], backend=spec.get('backend', 'flex'), seed=spec['seed'])
    docs = corpus.stream(spec['seed'])
    result = engine.train(spec['seed'], H, spec['lr'], spec['eta'], corpus.units[docs[:2 * H]])
    table = engine.schedule(H, spec['lr'], spec['eta'])
    torch.save({k: v.detach().cpu() for k, v in engine.P.items()}, spec['params_out'])
    result.pop('train_ms', None)
    return dict(arm=spec['arm'], setup=info, schedule_sha=c4a.hashlib.sha256(table.cpu().numpy().tobytes()).hexdigest(),
                **result)


def evaluate(spec, a):
    """Every arm's final parameters scored by one FP32 donor engine on the held-out E slice."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a_conv
    from megakernel_lab.fast_examples.vendor.adaptive.cops4_a.s106b.corpus import Corpus
    corpus = Corpus(CORPUS)
    units = corpus.units[corpus.slice_docs('E')[:spec.get('eval_units', 64)]]
    engine, _ = c4a_conv.make_engine(a, 'fp32', backend=spec.get('backend', 'flex'))
    out = {}
    for arm, path in spec['params'].items():
        P = torch.load(path, map_location='cuda')
        with torch.no_grad():
            for n in engine.P:
                engine.P[n].copy_(P[n])
        r = engine.evaluate(units, spec['eta'])
        out[arm] = {'E_ce_nats_per_byte': float(r['pos_sum'].sum() / (r['n'] * 8192)), 'units': r['n'],
                    'seconds': r['seconds']}
    return {'eval': out}


def update_check(spec, a):
    """One update from the same init/batch/schedule: arm deltas vs the FP64-gradient reference update
    (FP64 autograd of our formulation + the donor's Engine._update)."""
    import torch
    from megakernel_lab.fast_examples.vendor.adaptive import c4a
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import batch_inputs, init_params, schedule_rows, synthetic_tokens
    from megakernel_lab.operation_compiler.muon import prepare_muon
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import muon_config
    P0 = {k: v.cuda() for k, v in init_params(a, 106140).items()}
    S0 = {k: v.cuda() for k, v in prepare_muon(muon_config(a), init_params(a, 106140)).state.items()}
    tokens = synthetic_tokens(a, 777).cuda()
    sc = schedule_rows(64, 3e-3, 1.0)[7].cuda()
    inputs = tuple(t.cuda() for t in batch_inputs(a, tokens.cpu(), sc.cpu()))
    loss64, grads64 = c4a.own_reference(a, P0, inputs, torch.float64)
    ref_p, _, _ = c4a.donor_optimizer(a, P0, {k: v.float() for k, v in grads64.items()}, S0, sc)
    if spec['arm'].startswith('native'):
        native = c4a.Native(spec['folder'])
        native.restore((P0, S0))
        native.x.copy_(inputs[0]); native.y.copy_(inputs[1]); native.sc.copy_(sc)
        loss, _, params, _ = native.step()
    else:
        from megakernel_lab.fast_examples.vendor.adaptive import c4a_conv
        engine, _ = c4a_conv.make_engine(a, spec['arm'], backend=spec.get('backend', 'flex'))
        c4a_conv.load_state(engine, P0, S0)
        c4a_conv.stage(engine, inputs[0], inputs[1], sc)
        engine.step()
        torch.cuda.synchronize()
        loss, params = engine.loss.clone(), {k: v.detach().clone() for k, v in engine.P.items()}
    rows = {k: c4a.err(params[k] - P0[k], ref_p[k] - P0[k])['rel_l2'] for k in P0}
    return {'arm': spec['arm'], 'loss_rel_err': abs(loss.double().item() - loss64.item()) / abs(loss64.item()),
            'delta_rel_l2': rows, 'delta_worst': max((v, k) for k, v in rows.items()),
            'delta_median': sorted(rows.values())[len(rows) // 2]}


def main():
    spec = json.loads(sys.argv[1])
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from megakernel_lab.fast_examples.vendor.adaptive.examples.adaptive_memory import RECIPES
    a = RECIPES[spec['recipe']]
    started = time.perf_counter()
    try:
        fn = {'native': native_arm, 'conv': conv_arm, 'train_native': train_native, 'train_conv': train_conv,
              'evaluate': evaluate, 'update_check': update_check}[spec['kind']]
        result = fn(spec, a)
        result['passed'] = True
    except Exception as exc:  # noqa: BLE001
        import traceback
        result = {'passed': False, 'error': repr(exc)[:3000], 'traceback': traceback.format_exc()[-6000:]}
    result['process_seconds'] = time.perf_counter() - started
    Path(spec['out']).write_text(json.dumps(result, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
