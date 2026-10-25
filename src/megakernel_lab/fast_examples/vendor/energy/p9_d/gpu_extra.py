"""P9-D GPU tasks: SD learning/trained-state checks of practical controls and lane diagnostics.

Frozen c_ops4_b tasks and p8_d diagnostics are reused unchanged (p9d.py falls back to them)."""
from __future__ import annotations

from megakernel_lab.fast_examples.vendor.energy.cops4b import save


def train_control_cfg(payload, arg, out):
    """Learning gate of a practical control at a named configuration ('<config>:<control>'), POLICY.json
    controls.practical_bf16_validity: N updates of the control vs C1 eager and C2 manual from the same init, data
    order and RNG. Afterwards the same-state rule of ONE control update is evaluated (reported) at the fresh state and
    at the control's own trained state; R/C1/C2 come from c_ops4_b.checks at that state."""
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import checks
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import State
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import shared_images
    from megakernel_lab.fast_examples.vendor.energy.cops4b import bindings, config
    from megakernel_lab.fast_examples.vendor.energy.p8_d.controls_extra import make_control
    cfg_name, _, variant = arg.partition(':')
    cfg = config(cfg_name)
    updates = payload.get('train_updates', 400)
    rows = max(2048, updates + 8)
    images = shared_images(cfg)
    params, opt, state, (imgs, order), _ = bindings(cfg, rows, device='cuda', images=images)
    names = list(params)

    def same_state(P0, O0, S0):
        S = State(cfg, P0, O0, S0, imgs, order)
        ctrl = make_control(cfg, variant)
        first = ctrl.run_update(S)
        torch.cuda.synchronize()
        xpos = imgs[order[int(S0['t'])]]
        obs, _ = checks.update_observables(cfg, P0, O0, xpos, first['xneg'])
        obs['control'] = checks.pack(names, first['loss'], first['grads'], S.P, S.m, S.v)
        res = checks.compare(obs, subject='control')
        return {'passed': all(r['passed'] for r in res.values()),
                'rows': {q: {'errors': r['errors'], 'bound': r['bound'], 'passed': r['passed']} for q, r in res.items()}}

    report = {'config': cfg_name, 'control': variant, 'updates': updates}
    report['same_state_fresh'] = same_state(params, opt, state)
    finals, losses, trained = {}, {}, None
    for arm in ('eager', 'manual', variant):
        S = State(cfg, params, opt, state, imgs, order)
        ctrl = make_control(cfg, arm)
        traj = []
        for _ in range(updates):
            traj.append(ctrl.run_update(S)['loss'].detach().clone())
        torch.cuda.synchronize()
        losses[arm] = torch.stack(traj).double().cpu().numpy()
        finals[arm] = {k: v.clone() for k, v in S.P.items()}
        if arm == variant:
            trained = ({n: S.P[n].clone() for n in names},
                       {**{f'exp_avg:{n}': S.m[n].clone() for n in names},
                        **{f'exp_avg_sq:{n}': S.v[n].clone() for n in names},
                        **{f'step:{n}': S.step[n].clone() for n in names}},
                       {'buf': S.buf.clone(), 'ptr': S.ptr.clone(), 't': S.t.clone(), 'key': S.key.clone()})
        del ctrl, S
        torch.cuda.empty_cache()
    if payload.get('split_cpu_audit'):
        from megakernel_lab.fast_examples.cd_split import learning_observation
        learning_observation(out, initial=params, finals=finals, trajectories=losses, subject=variant)
    flat = lambda d: torch.cat([d[n].double().flatten() for n in names])
    travel = float((flat(finals['eager']) - flat(params)).norm())
    dist = lambda a: float((flat(finals[a]) - flat(finals['eager'])).norm()) / max(travel, 1e-30)
    cal = dist('manual')
    finite = all(np.isfinite(v).all() for v in losses.values())
    report.update({'finite': bool(finite), 'd_control': dist(variant), 'calibration_d_manual': cal,
                   'bound': max(2 * cal, 0.02), 'travel_eager': travel,
                   'loss_last50_mean': {k: float(v[-50:].mean()) for k, v in losses.items()},
                   'loss_first': {k: float(v[0]) for k, v in losses.items()}})
    report['learning_passed'] = bool(finite and report['d_control'] <= report['bound'])
    report['same_state_trained'] = same_state(*trained)
    report['passed'] = report['learning_passed']
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'trajectories.npz', **losses)
    save(out / 'train_control.json', report)
    return {k: report[k] for k in ('config', 'control', 'passed', 'learning_passed', 'd_control', 'bound',
                                   'calibration_d_manual', 'loss_last50_mean')} | {
        'same_state_fresh_passed': report['same_state_fresh']['passed'],
        'same_state_trained_passed': report['same_state_trained']['passed']}


def _region_inputs(abi, device):
    """Deterministic, realistic inputs of a standalone train region: s100l-init parameters and their casts, zero
    AdamW state (step 0), uniform [0, 1] negatives, the initial replay state and Fashion-MNIST images/order."""
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import shared_images
    from megakernel_lab.fast_examples.vendor.energy.cops4b import bindings
    cfg, m = abi['config'], abi['metadata']
    params, opt, state, (images, order), _ = bindings(cfg, 2048, device=device, images=shared_images(cfg))
    names = m['parameter_names']
    casts = [params[c['source']].to(getattr(torch, c['dtype'])) for c in m.get('invariant_casts', [])]
    gen = torch.Generator(device=device).manual_seed(9001)
    carry = []
    for n in m['carry_names']:
        spec = next(x for x in abi['inputs'] if x[0] == n) if any(x[0] == n for x in abi['inputs']) else None
        shape = [cfg['batch'], cfg['crop'] ** 2]
        carry.append(torch.rand(shape, generator=gen, device=device))
    values = ([params[n] for n in names] + casts + [opt[n] for n in m['optimizer_state_names']] + carry
              + [state[n] for n in m['state_names']] + [images, order])
    if len(values) != len(abi['inputs']):
        raise RuntimeError(f'train-region input arity {len(values)} vs {len(abi["inputs"])}')
    out = []
    for v, (name, shape, dtype) in zip(values, abi['inputs']):
        if list(v.shape) != shape or str(v.dtype).removeprefix('torch.') != dtype:
            raise RuntimeError(f'region input {name}: {list(v.shape)} {v.dtype} vs {shape} {dtype}')
        out.append(v.clone().contiguous())
    return out


def region_time(payload, arg, out):
    """Standalone timing of region builds ('<task dir>,<task dir>,...' under build_attempt; the first is the
    reference for output agreement): same inputs, `blocks` CTAs (default 112 = the composed panel grid)."""
    import json
    import random
    from pathlib import Path
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import _time_graph
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    from megakernel_lab.operation_compiler.runtime import Runtime, build
    blocks = payload.get('region_blocks', 112)
    rows, arms, outs = {}, {}, {}
    keep = []
    for name in [x for x in arg.split(',') if x]:
        directory, _, attempt = name.partition('^')
        base = Path(attempt or payload['build_attempt']) / directory / 'native'
        abi = json.loads((base / 'abi.json').read_text())
        program = GeneratedProgram.load(base)
        library = build(program, cache_dir=payload['cache_dir'], allow_compile=False)
        args = _region_inputs(abi, 'cuda')
        rt = Runtime(program, args, library=library, blocks=min(blocks, 10 ** 6))
        rt.step()
        torch.cuda.synchronize()
        outs[name] = [t.clone() for t in rt.outputs]
        snap = [a.clone() for a in args]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            rt.step()
        reset = (lambda a=args, s=snap: [x.copy_(y) for x, y in zip(a, s)])
        arms[name] = (g, reset)
        keep.append((rt, args))
        rows[name] = {'launch_policy': rt.launch_policy, 'arena_bytes': rt.manifest['arena_bytes'],
                      'selection': rt.manifest.get('compiler_selection', {}).get('schedule'),
                      'tensor_maps': len(rt.manifest.get('tensor_maps') or [])}
    first = next(iter(outs))
    for name, o in outs.items():
        rel = []
        for a, b in zip(o, outs[first]):
            a64, b64 = a.double(), b.double()
            rel.append(float((a64 - b64).norm() / max(float(b64.norm()), 1e-30)))
        rows[name]['max_rel_vs_first'] = max(rel)
        rows[name]['loss'] = float(o[0])
        rows[name]['finite'] = bool(all(torch.isfinite(t.double()).all() for t in o))
    rng = random.Random(5001)
    samples = {k: [] for k in arms}
    for _ in range(payload.get('region_rounds', 10)):
        order = list(arms)
        rng.shuffle(order)
        for k in order:
            g, reset = arms[k]
            reset()
            samples[k].append(_time_graph(g, 20))
    for k in arms:
        rows[k]['median_ms'] = float(np.median(samples[k]))
    result = {'blocks': blocks, 'rows': rows, 'samples_ms': samples}
    save(out / 'region_time.json', result)
    return {k: {q: v[q] for q in ('median_ms', 'max_rel_vs_first', 'finite', 'selection', 'tensor_maps')}
            for k, v in rows.items()}


def profile_blocks(payload, arg, out):
    """c_ops4_b profile (top-level marks; train phases when built with phase ticks) at a fixed grid size
    ('<build>:<blocks>'), so programs with different launch policies are compared on the same CTA count."""
    import numpy as np
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.gpu_tasks import Native
    name, _, blocks = arg.rpartition(':')
    native = Native(payload, name, blocks=int(blocks))
    snap = native.snapshot()
    rows = []
    for _ in range(payload.get('profile_repeats', 5)):
        native.restore(snap)
        native.count.fill_(native.K)
        native.rt.step(profile=True)
        torch.cuda.synchronize()
        rows.append(native.rt._ticks.cpu().tolist())
    native.restore(snap)
    st = np.array(rows[1:], dtype=np.float64)
    top = {k: float(np.median(st[:, i + 1] - st[:, i])) / 1e3
           for i, k in enumerate(('init', 'sampling_loop', 'train', 'commit'))}
    result = {'blocks': int(blocks), 'top_us': top, 'launch_policy': native.rt.launch_policy,
              'shared_bytes': native.rt.manifest.get('shared_bytes')}
    save(out / 'profile_blocks.json', result)
    return result
