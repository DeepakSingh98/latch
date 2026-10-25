"""P8-D GPU diagnostics (development only; never a qualification receipt)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from megakernel_lab.fast_examples.vendor.energy.cops4b import save

HERE = Path(__file__).resolve().parent




def _with_extra_controls(fn, payload, arg, out):
    """Run a frozen c_ops4_b GPU task with the P8-D control factory (harness-level class swap, restored after)."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import controls
    from megakernel_lab.fast_examples.vendor.energy.p8_d.controls_extra import EXTRA, ExtraControl
    original = controls.Control

    class Factory(original):
        def __new__(cls, cfg, variant):
            if variant in EXTRA:
                return ExtraControl(cfg, variant)
            return original(cfg, variant)

    controls.Control = Factory
    try:
        return fn(payload, arg, out)
    finally:
        controls.Control = original


def perfx(payload, arg, out):
    """c_ops4_b perf with the declared extra controls admitted (POLICY.json controls.added_before_measurement)."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import gpu_tasks
    return _with_extra_controls(gpu_tasks.perf, payload, arg, out)


def kscalex(payload, arg, out):
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import gpu_tasks
    return _with_extra_controls(gpu_tasks.kscale, payload, arg, out)


def train_control(payload, arg, out):
    """Learning gate for a practical control: N updates of `arg` vs C1 eager and C2 manual from the same init,
    data order and RNG (POLICY.json controls.validity.practical_bf16_controls)."""
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import State
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import shared_images
    from megakernel_lab.fast_examples.vendor.energy.cops4b import bindings, config
    from megakernel_lab.fast_examples.vendor.energy.p8_d.controls_extra import make_control
    cfg = config(payload.get('train_config', 'wb'))
    updates = payload.get('train_updates', 400)
    rows = max(2048, updates + 8)
    images = shared_images(cfg)
    params, opt, state, (imgs, order), _ = bindings(cfg, rows, device='cuda', images=images)
    finals, losses = {}, {}
    for variant in ('eager', 'manual', arg):
        S = State(cfg, params, opt, state, imgs, order)
        ctrl = make_control(cfg, variant)
        traj = []
        for _ in range(updates):
            traj.append(ctrl.run_update(S)['loss'].detach().clone())
        torch.cuda.synchronize()
        losses[variant] = torch.stack(traj).double().cpu().numpy()
        finals[variant] = {k: v.clone() for k, v in S.P.items()}
        del ctrl, S
        torch.cuda.empty_cache()
    names = list(params)
    flat = lambda d: torch.cat([d[n].double().flatten() for n in names])
    travel = float((flat(finals['eager']) - flat(params)).norm())
    dist = lambda a: float((flat(finals[a]) - flat(finals['eager'])).norm()) / max(travel, 1e-30)
    cal = dist('manual')
    finite = all(np.isfinite(v).all() for v in losses.values())
    result = {'control': arg, 'updates': updates, 'finite': bool(finite), 'd_control': dist(arg),
              'calibration_d_manual': cal, 'bound': max(2 * cal, 0.02),
              'loss_last50_mean': {k: float(v[-50:].mean()) for k, v in losses.items()},
              'loss_first': {k: float(v[0]) for k, v in losses.items()}}
    result['passed'] = bool(finite and result['d_control'] <= result['bound'])
    save(out / 'train_control.json', result)
    return result


def panel_profile(payload, arg, out):
    """Diagnostic: globaltimer stamps (CTA 0) at the top-level marks and at every segment/product boundary of the
    panel engine's first iteration. Intervals include waiting; they locate time, not isolated arithmetic."""
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import Native
    native = Native(payload, arg)
    manifest = native.rt.manifest
    engine = manifest.get('panel_engine', {})
    base = manifest['tick_layout']['body_first_iteration']
    snap = native.snapshot()
    rows = []
    for _ in range(payload.get('profile_repeats', 6)):
        native.restore(snap)
        native.count.fill_(native.K)
        native.rt.step(profile=True)
        torch.cuda.synchronize()
        rows.append(native.rt._ticks.cpu().tolist())
    native.restore(snap)
    st = np.array(rows[1:], dtype=np.float64)
    top = {k: float(np.median(st[:, i + 1] - st[:, i])) / 1e3
           for i, k in enumerate(('init', 'sampling_loop', 'train', 'commit'))}
    result = {'top_us': top, 'per_iteration_us': top['sampling_loop'] / max(native.K, 1)}
    labels = engine.get('tick_layout', {}).get('boundaries', [])
    if labels:
        spans = []
        for i, label in enumerate(labels):
            a, b = base + i, base + i + 1
            spans.append({'span': label, 'us': float(np.median(st[:, b] - st[:, a])) / 1e3})
        result['first_iteration'] = spans
        result['first_iteration_total_us'] = sum(x['us'] for x in spans)
        result['segments_us'] = sum(x['us'] for x in spans if x['span'].startswith('segment'))
        result['products_us'] = sum(x['us'] for x in spans if x['span'].startswith('product'))
    save(out / 'panel_profile.json', result)
    return {k: v for k, v in result.items() if k != 'first_iteration'} | {
        'largest': sorted(result.get('first_iteration', []), key=lambda x: -x['us'])[:8]}


def kscale2(payload, arg, out):
    """Count transfer: '<build>:<K1,K2>:<ctrl1+ctrl2>'. The native binary at runtime count K versus each control
    compiled for chain K (paired randomized rounds per K); validity of each control at each K by the same-state rule."""
    import math as _m
    import random as _r
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import checks
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import State, capture_update
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import Native, _time_graph
    from megakernel_lab.fast_examples.vendor.energy.p8_d.controls_extra import make_control
    import torch._dynamo.config as dynamo_config
    for key in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(dynamo_config, key):
            setattr(dynamo_config, key, max(getattr(dynamo_config, key), 1024))
    name, ks, ctrls = arg.split(':')
    native = Native(payload, name)
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in native.bound())
    snap = native.snapshot()
    rows = {}
    for K in [int(x) for x in ks.split(',') if x]:
        native.count.fill_(K)
        g_native = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_native):
            native.rt.step()
        native.restore(snap)
        cfg_k = dict(native.cfg, chain=K)
        arms = {'native': (g_native, lambda: native.restore(snap))}
        validity, setup = {}, {}
        keep = []
        for variant in ctrls.split('+'):
            started = time.perf_counter()
            S = State(cfg_k, P0, O0, S0, native.images, native.order)
            ctrl = make_control(cfg_k, variant)
            first = ctrl.run_update(S)
            torch.cuda.synchronize()
            xpos = native.images[native.order[int(S0['t'])]]
            obs, _ = checks.update_observables(cfg_k, P0, O0, xpos, first['xneg'])
            obs['control'] = checks.pack(native.names, first['loss'], first['grads'], S.P, S.m, S.v)
            res = checks.compare(obs, subject='control')
            validity[variant] = all(r['passed'] for r in res.values())
            S2 = State(cfg_k, P0, O0, S0, native.images, native.order)
            g = capture_update(ctrl, S2)
            setup[variant] = time.perf_counter() - started
            sn = S2.snapshot()
            arms[variant] = (g, (lambda s=S2, x=sn: s.restore(x)))
            keep.append((ctrl, S, S2))
        samples = {k: [] for k in arms}
        rng = _r.Random(4100 + K)
        for _ in range(payload.get('kscale_rounds', 10)):
            order = list(arms)
            rng.shuffle(order)
            for key in order:
                g, reset = arms[key]
                reset()
                samples[key].append(_time_graph(g, 10))
                reset()
        med = {k: float(np.median(v)) for k, v in samples.items()}
        valid = [k for k in med if k != 'native' and validity.get(k)]
        fastest = min(valid, key=lambda k: med[k]) if valid else None
        rows[K] = {'medians_ms': med, 'validity': validity, 'fastest_valid_control': fastest,
                   'ratio_native_over_fastest': med['native'] / med[fastest] if fastest else None,
                   'control_setup_seconds': setup}
        del keep, arms
        torch.cuda.empty_cache()
    native.restore(snap)
    result = {'build': name, 'by_count': rows}
    save(out / 'kscale2.json', result)
    return result


def checkk(payload, arg, out):
    """All c_ops4_b gates with the native's runtime count set to K ('<build>:<K>'): chain steps 1..K (sampled when
    K > 8) and every update observable from x_K. Same binary as the primary count."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import Native, _summary, gates
    name, _, k = arg.rpartition(':')
    native = Native(payload, name)
    K = int(k)
    if not 0 < K <= native.K:
        raise ValueError('count outside the native bound')
    native.K = K
    report = gates(native, label=f'fresh-count-{K}', limit=payload.get('step_limit', 8))
    report.update(count=K, setup_seconds=native.setup_seconds)
    save(out / 'check.json', report)
    return {'passed': report['passed'], 'count': K, 'summary': _summary(report)}
