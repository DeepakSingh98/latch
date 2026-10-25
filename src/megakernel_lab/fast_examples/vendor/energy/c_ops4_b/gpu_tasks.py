"""GPU tasks for C-OPS4-B: native gates (check), real-data training (train), complete-update timing (perf).

The native program is loaded from a CPU-built artifact (no compilation on the GPU worker). All arms share the
staged FP32 images and order table; parameters, optimizer state and explicit state are per arm.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import checks
from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import reference as R
from megakernel_lab.fast_examples.vendor.energy.cops4b import bindings, flat_args, images_fp32, save
from megakernel_lab.fast_examples.vendor.energy.examples import energy as recipe

_IMAGES = {}


def shared_images(cfg):
    key = cfg['crop']
    if key not in _IMAGES:
        _IMAGES.clear()
        _IMAGES[key] = torch.from_numpy(images_fp32(cfg)).cuda()
    return _IMAGES[key]


class Native:
    """A bound native program with snapshot/restore and structured outputs."""

    def __init__(self, payload, name, *, blocks=None):
        from megakernel_lab.operation_compiler.lower import GeneratedProgram
        from megakernel_lab.operation_compiler.runtime import Runtime, build
        # '<build name>^<attempt>' selects a build from another attempt (before/after comparisons in one job).
        name, _, override = name.partition('^')
        attempt = override or payload.get('native_builds', {}).get(name, payload['build_attempt'])
        base = Path(attempt) / f'capture-{name}' / 'native'
        self.meta = json.loads((base / 'abi.json').read_text())
        self.cfg, rows = self.meta['config'], self.meta['order_rows']
        program = GeneratedProgram.load(base)
        started = time.perf_counter()
        library = build(program, cache_dir=payload['cache_dir'], allow_compile=False)
        params, opt, state, inputs, count = bindings(self.cfg, rows, device='cuda', images=shared_images(self.cfg))
        args = flat_args(params, opt, state, inputs, count)
        for tensor, spec in zip(args, self.meta['args']):
            if list(tensor.shape) != spec['shape'] or str(tensor.dtype).removeprefix('torch.') != spec['dtype']:
                raise RuntimeError('native ABI mismatch')
        self.rt = Runtime(program, args, library=library, blocks=blocks)
        self.setup_seconds = time.perf_counter() - started
        self.names = self.meta['parameter_names']
        self.onames = self.meta['optimizer_state_names']
        self.snames = self.meta['state_names']
        self.cnames = self.meta.get('carry_names') or json.loads(
            (base / 'graph.json').read_text())['metadata']['carry_names']
        n, no, ns = len(self.names), len(self.onames), len(self.snames)
        self.mutable = self.rt.args[:n + no + ns]
        self.count = self.rt.args[-1]
        self.images, self.order = self.rt.args[n + no + ns], self.rt.args[n + no + ns + 1]
        self.K = self.cfg['chain']

    def snapshot(self):
        return [t.clone() for t in self.mutable]

    def restore(self, snap):
        for t, s in zip(self.mutable, snap):
            t.copy_(s)

    def bound(self):
        n, no = len(self.names), len(self.onames)
        m = self.mutable
        return (dict(zip(self.names, m[:n])), dict(zip(self.onames, m[n:n + no])), dict(zip(self.snames, m[n + no:])))

    def outputs(self):
        o = [t.clone() for t in self.rt.outputs]
        n, no, ns, nc = len(self.names), len(self.onames), len(self.snames), len(self.cnames)
        return {'loss': o[0], 'grads': dict(zip(self.names, o[1:1 + n])),
                'params': dict(zip(self.names, o[1 + n:1 + 2 * n])),
                'opt': dict(zip(self.onames, o[1 + 2 * n:1 + 2 * n + no])),
                'state': dict(zip(self.snames, o[1 + 2 * n + no:1 + 2 * n + no + ns])),
                'carry': dict(zip(self.cnames, o[1 + 2 * n + no + ns:-2])), 'executed': int(o[-2]), 'status': int(o[-1])}

    def run(self, count=None):
        self.count.fill_(self.K if count is None else count)
        self.rt.step()
        torch.cuda.synchronize()
        return self.outputs()


def _equal_outputs(a, b):
    flat = lambda o: [o['loss'], *o['grads'].values(), *o['params'].values(), *o['opt'].values(),
                      *o['state'].values(), *o['carry'].values()]
    return all(torch.equal(x, y) for x, y in zip(flat(a), flat(b))) and a['executed'] == b['executed']


def _step_indices(K, limit):
    if K <= limit:
        return list(range(1, K + 1))
    return sorted({1, 2, 3, 4, K // 4, K // 2, 3 * K // 4, K - 1, K})


def gates(native, *, label, limit=8):
    """Every native gate of POLICY.json from the native's current bindings (restored afterwards)."""
    cfg, K = native.cfg, native.K
    z = recipe.dims(cfg)
    snap = native.snapshot()
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in native.bound())
    t0, ptr0 = int(S0['t']), int(S0['ptr'])
    report = {'label': label, 't': t0}
    a = native.run()
    after_a = native.snapshot()
    native.restore(snap)
    b = native.run()
    after_b = native.snapshot()
    report['repeat_bitwise'] = _equal_outputs(a, b) and all(torch.equal(x, y) for x, y in zip(after_a, after_b))
    native.restore(snap)
    native.count.fill_(K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.rt.step()
    native.restore(snap)
    graph.replay()
    torch.cuda.synchronize()
    report['graph_replay_bitwise'] = _equal_outputs(a, native.outputs()) and all(
        torch.equal(x, y) for x, y in zip(after_a, native.snapshot()))
    report['status_zero'] = a['status'] == 0 and a['executed'] == K
    native.restore(snap)
    x0 = native.run(0)['carry']['x']
    x0_ref, fresh, idx = R.init_negatives(cfg, S0['buf'], t0, cfg['seed'])
    report['init_exact'] = bool(torch.equal(x0, x0_ref))
    report['init_fresh_rows'] = int(fresh.sum())
    xneg = a['carry']['x']
    slots = torch.as_tensor((ptr0 + np.arange(z['B'])) % z['R'], device=xneg.device)
    expected = S0['buf'].clone()
    expected[slots] = xneg
    new_state = dict(zip(native.snames, after_a[len(native.names) + len(native.onames):]))
    report['commit_exact'] = bool(torch.equal(new_state['buf'], expected) and int(new_state['ptr']) == (ptr0 + z['B']) % z['R']
                                  and int(new_state['t']) == t0 + 1 and torch.equal(new_state['key'], S0['key'])
                                  and all(torch.equal(a['state'][k], new_state[k]) for k in native.snames))
    steps = []
    for k in _step_indices(K, limit):
        native.restore(snap)
        prev = native.run(k - 1)['carry']['x']
        native.restore(snap)
        nxt = native.run(k)['carry']['x']
        inc = checks.step_increments(cfg, P0, prev, k - 1, t0, S0['key'], subject_next=nxt)
        errors = {lab: checks.rel(inc[lab], inc['R']) for lab in ('native', 'C1', 'C2')}
        steps.append({'k': k, **checks.judge(errors)})
    report['steps'] = steps
    xpos = native.images[native.order[t0]]
    obs, scalars = checks.update_observables(cfg, P0, O0, xpos, xneg)
    names = native.names
    obs['native'] = checks.pack(names, a['loss'], a['grads'], a['params'],
                                {n: a['opt'][f'exp_avg:{n}'] for n in names}, {n: a['opt'][f'exp_avg_sq:{n}'] for n in names})
    report['update'] = checks.compare(obs)
    report['update_scalars'] = scalars
    new, norm, coef = checks.optimizer_from_gradients(cfg, P0, O0, a['grads'])
    flat = lambda d: torch.cat([d[n].double().flatten() for n in names])
    tier = {'params_delta': checks.rel(flat(a['params']) - flat(P0), flat({n: new[n][0] for n in names}) - flat(P0)),
            'm': checks.rel(flat({n: a['opt'][f'exp_avg:{n}'] for n in names}), flat({n: new[n][1] for n in names})),
            'v': checks.rel(flat({n: a['opt'][f'exp_avg_sq:{n}'] for n in names}), flat({n: new[n][2] for n in names})),
            'steps_advanced': all(int(a['opt'][f'step:{n}']) == int(O0[f'step:{n}']) + 1 for n in names),
            'norm': norm, 'coef': coef}
    tier['passed'] = bool(tier['steps_advanced'] and max(tier['params_delta'], tier['m'], tier['v']) <= 1e-5)
    # Amendment v1.1 (POLICY.json): FP32 control optimizer on the native's own raw gradients as calibration.
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control
    names_, decay_ = checks.names_and_decay(cfg)
    ctl = Control(cfg, 'manual')._optimizer(P0, {n: O0[f'exp_avg:{n}'] for n in names},
                                            {n: O0[f'exp_avg_sq:{n}'] for n in names},
                                            {n: O0[f'step:{n}'] for n in names}, a['grads'], names, decay_)
    ref_delta = flat({n: new[n][0] for n in names}) - flat(P0)
    cal = {'params_delta': checks.rel(flat({n: ctl[n][0] for n in names}) - flat(P0), ref_delta),
           'm': checks.rel(flat({n: ctl[n][1] for n in names}), flat({n: new[n][1] for n in names})),
           'v': checks.rel(flat({n: ctl[n][2] for n in names}), flat({n: new[n][2] for n in names}))}
    tier['v1_1'] = {q: {'native': tier[q], 'fp32_control': cal[q], 'bound': max(2 * cal[q], 1e-6)} for q in cal}
    tier['v1_1_passed'] = bool(tier['steps_advanced'] and all(r['native'] <= r['bound'] for r in tier['v1_1'].values()))
    report['optimizer_tier'] = tier
    native.restore(snap)
    report['passed'] = bool(report['repeat_bitwise'] and report['graph_replay_bitwise'] and report['status_zero']
                            and report['init_exact'] and report['commit_exact']
                            and all(s['passed'] for s in steps) and all(r['passed'] for r in report['update'].values())
                            and tier['v1_1_passed'])
    report['passed_v1'] = bool(report['passed'] and tier['passed'])
    return report


def check(payload, arg, out):
    name, _, state_file = arg.partition('#')        # '<build name>[#<saved state file>]'
    native = Native(payload, name)
    if state_file:
        saved = torch.load(state_file, map_location='cuda')
        native.restore(saved['mutable'])
    report = gates(native, label=state_file or 'fresh', limit=payload.get('step_limit', 8))
    report.update(setup_seconds=native.setup_seconds, launch_policy=native.rt.launch_policy,
                  arena_bytes=native.rt.manifest['arena_bytes'])
    save(out / 'check.json', report)
    return {'passed': report['passed'], 'summary': _summary(report)}


def _summary(report):
    return {k: report[k] for k in ('repeat_bitwise', 'graph_replay_bitwise', 'status_zero', 'init_exact',
                                   'commit_exact', 'passed')} | {
        'steps_failed': [s['k'] for s in report['steps'] if not s['passed']],
        'update_failed': [q for q, r in report['update'].items() if not r['passed']],
        'optimizer_tier_v1': report['optimizer_tier']['passed'],
        'optimizer_tier_v1_1': report['optimizer_tier']['v1_1_passed'], 'passed_v1': report.get('passed_v1'),
        'worst_step_native': max(s['errors']['native'] for s in report['steps'])}


def train(payload, arg, out):
    """Native vs eager controls from the same init, data order and RNG for N updates; then trained-state gates."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control, State
    native = Native(payload, arg)
    cfg, updates = native.cfg, payload.get('train_updates', 400)
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in native.bound())
    arms = {'native': []}
    native.count.fill_(native.K)
    started = time.perf_counter()
    for _ in range(updates):
        native.rt.step()
        arms['native'].append(native.rt.outputs[0].clone())
    torch.cuda.synchronize()
    native_seconds = time.perf_counter() - started
    finals = {'native': {k: v.clone() for k, v in native.bound()[0].items()}}
    controls = {}
    for variant in payload.get('train_controls', ['eager', 'manual']):
        S = State(cfg, P0, O0, S0, native.images, native.order)
        ctrl = Control(cfg, variant)
        graph = None
        losses = []
        for _ in range(updates):
            losses.append(ctrl.run_update(S)['loss'].detach().clone())
        torch.cuda.synchronize()
        arms[variant] = losses
        finals[variant] = {k: v.clone() for k, v in S.P.items()}
        controls[variant] = S
    traj = {k: torch.stack(v).double().cpu().numpy() for k, v in arms.items()}
    if payload.get('split_cpu_audit'):
        from megakernel_lab.fast_examples.cd_split import learning_observation
        learning_observation(out, initial=P0, finals=finals, trajectories=traj, subject='native')
    flat = lambda d: torch.cat([d[n].double().flatten() for n in native.names])
    travel = lambda k: float((flat(finals[k]) - flat(P0)).norm())
    dist = lambda a, b: float((flat(finals[a]) - flat(finals[b])).norm())
    base = 'eager'
    others = [k for k in finals if k not in ('native', base)]
    d_native = dist('native', base) / max(travel(base), 1e-30)
    d_cal = max(dist(k, base) / max(travel(base), 1e-30) for k in others) if others else 0.0
    test = torch.from_numpy(images_fp32(cfg, 'test')).cuda()
    energies = {}
    for k, P in finals.items():
        buf = native.bound()[2]['buf'] if k == 'native' else controls[k].buf
        with torch.no_grad():
            energies[k] = {'test_mean_E': float(recipe.energy(P, test, cfg['blocks']).mean()),
                           'buffer_mean_E': float(recipe.energy(P, buf, cfg['blocks']).mean())}
    finite = all(np.isfinite(v).all() for v in traj.values())
    result = {'updates': updates, 'native_seconds': native_seconds, 'finite': finite,
              'loss_first': {k: float(v[0]) for k, v in traj.items()},
              'loss_last50_mean': {k: float(v[-50:].mean()) for k, v in traj.items()},
              'per_update_rel_dev_median': {k: float(np.median(np.abs(v - traj[base]) / np.maximum(np.abs(traj[base]), 1e-12)))
                                            for k, v in traj.items() if k != base},
              'travel': {k: travel(k) for k in finals}, 'param_distance_to_eager_over_travel':
                  {k: dist(k, base) / max(travel(base), 1e-30) for k in finals if k != base},
              'energies': energies, 'gate': {'d_native': d_native, 'calibration': d_cal,
                                             'bound': max(2 * d_cal, 0.02)}}
    result['passed'] = bool(finite and d_native <= max(2 * d_cal, 0.02))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'trajectories.npz', **traj)
    torch.save({'mutable': native.snapshot(), 'updates': updates}, out / 'trained.pt')
    save(out / 'train.json', result)
    trained = gates(native, label=f'trained-{updates}', limit=payload.get('step_limit', 8))
    save(out / 'check-trained.json', trained)
    result['trained_gates'] = _summary(trained)
    result['passed'] = bool(result['passed'] and trained['passed'])
    save(out / 'train.json', result)
    return {'passed': result['passed'], 'gate': result['gate'], 'trained_gates': result['trained_gates'],
            'loss_last50_mean': result['loss_last50_mean']}


def _time_graph(graph, reps):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / reps


def perf(payload, arg, out):
    """Paired randomized complete-update timing: native CUDA graphs (one or more lowering variants of the same
    program, comma-separated build names; the first is the primary) vs captured conventional controls."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control, State, capture_update
    import torch._dynamo.config as dynamo_config
    for key in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(dynamo_config, key):
            setattr(dynamo_config, key, max(getattr(dynamo_config, key), 1024))
    builds = payload.get('native_builds', {})
    names = [x for x in arg.split(',') if x]
    natives = {}
    for name in names:
        natives[name] = Native(payload, name)
    primary = natives[names[0]]
    cfg = primary.cfg
    for name, nat in natives.items():
        if nat.cfg != cfg:
            raise RuntimeError(f'native variant {name} is a different program configuration')
    data_bytes = primary.images.numel() * 4 + primary.order.numel() * 8
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in primary.bound())
    arms, memory, setup, validity, agreement = {}, {}, {}, {}, {}
    reference_out = None
    for name, nat in natives.items():
        snap = nat.snapshot()
        nat.count.fill_(nat.K)
        nat.rt.step()
        torch.cuda.synchronize()
        result = nat.outputs()
        nat.restore(snap)
        if reference_out is None:
            reference_out = result
        agreement[name] = {'bitwise_vs_primary': _equal_outputs(result, reference_out),
                           'loss': float(result['loss']), 'status': result['status']}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nat.rt.step()
        nat.restore(snap)
        arms['native:' + name] = (graph, (lambda n=nat, sn=snap: n.restore(sn)))
        memory['native:' + name] = {'arena_bytes': nat.rt.manifest['arena_bytes'],
                                    'binding_bytes': sum(t.numel() * t.element_size() for t in nat.mutable),
                                    'total_bytes': nat.rt.manifest['arena_bytes'] + sum(
                                        t.numel() * t.element_size() for t in nat.mutable) + data_bytes,
                                    'row_tasks': nat.rt.manifest.get('row_tasks', {}).get('selected', False)}
        setup['native:' + name] = {'load_seconds': nat.setup_seconds}
    for variant in payload.get('perf_controls', ['fused_autotune', 'fused_default', 'pieces_autotune', 'eager']):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        started = time.perf_counter()
        try:
            S = State(cfg, P0, O0, S0, primary.images, primary.order)
            ctrl = Control(cfg, variant)
            first = ctrl.run_update(S)
            torch.cuda.synchronize()
            compile_seconds = time.perf_counter() - started
            xpos = primary.images[primary.order[int(S0['t'])]]
            obs, _ = checks.update_observables(cfg, P0, O0, xpos, first['xneg'])
            obs['control'] = checks.pack(primary.names, first['loss'], first['grads'], S.P, S.m, S.v)
            rows = checks.compare(obs, subject='control')
            validity[variant] = {'passed': all(r['passed'] for r in rows.values()),
                                 'errors': {q: r['errors']['control'] for q, r in rows.items()}}
            S2 = State(cfg, P0, O0, S0, primary.images, primary.order)
            del S
            started = time.perf_counter()
            g = capture_update(ctrl, S2)
            snap2 = S2.snapshot()
            setup[variant] = {'compile_and_first_update_seconds': compile_seconds,
                              'capture_seconds': time.perf_counter() - started}
            g.replay()
            torch.cuda.synchronize()
            S2.restore(snap2)
            memory[variant] = {'peak_delta_bytes': torch.cuda.max_memory_allocated() - base,
                               'total_bytes': torch.cuda.max_memory_allocated() - base + data_bytes}
            arms[variant] = (g, (lambda s=S2, sn=snap2: s.restore(sn)))
        except Exception as exc:
            validity[variant] = {'passed': False, 'error': repr(exc)[-2000:]}
    rounds, rng = payload.get('perf_rounds', 20), random.Random(payload.get('perf_seed', 4001))
    warm = {}
    for key, (g, reset) in arms.items():
        reset()
        warm[key] = _time_graph(g, 3)
        reset()
    reps = {key: max(3, min(200, math.ceil(60.0 / max(ms, 1e-3)))) for key, ms in warm.items()}
    samples = {key: [] for key in arms}
    for _ in range(rounds):
        order = list(arms)
        rng.shuffle(order)
        for key in order:
            g, reset = arms[key]
            reset()
            samples[key].append(_time_graph(g, reps[key]))
            reset()
    med = {k: float(np.median(v)) for k, v in samples.items()}
    valid = [k for k in arms if not k.startswith('native:') and validity.get(k, {}).get('passed')]
    fastest = min(valid, key=lambda k: med[k]) if valid else None
    boot = random.Random(77)

    def ci(a, b):
        ratios, n = [], len(samples[a])
        for _ in range(2000):
            idx = [boot.randrange(n) for _ in range(n)]
            ratios.append(np.median([samples[a][i] for i in idx]) / np.median([samples[b][i] for i in idx]))
        return [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))]
    ratios = {k: {'ratio': med[k] / med[fastest], 'ci95': ci(k, fastest)} for k in arms
              if k.startswith('native:') and fastest} if fastest else {}
    scaling = {}
    for count in payload.get('native_counts', []):
        for name, nat in natives.items():
            snap = nat.snapshot()
            nat.count.fill_(count)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                nat.rt.step()
            times = []
            for _ in range(5):
                nat.restore(snap)
                times.append(_time_graph(g, max(3, reps['native:' + name])))
            nat.restore(snap)
            scaling.setdefault(name, {})[count] = float(np.median(times))
    result = {'medians_ms': med, 'samples_ms': samples, 'reps': reps, 'validity': validity,
              'native_agreement': agreement, 'fastest_valid_control': fastest, 'ratios_vs_fastest': ratios,
              'memory': memory, 'setup': setup, 'native_count_scaling_ms': scaling,
              'launch_policy': {k: n.rt.launch_policy for k, n in natives.items()},
              'device': torch.cuda.get_device_name()}
    save(out / 'perf.json', result)
    return {k: result[k] for k in ('medians_ms', 'fastest_valid_control', 'ratios_vs_fastest',
                                   'native_count_scaling_ms', 'native_agreement')} | {
        'validity': {k: v['passed'] for k, v in validity.items()}, 'memory': memory}


def profile(payload, arg, out):
    """Diagnostic: globaltimer stamps of CTA 0 at top-level marks and at each body phase of iteration 0.

    Intervals include synchronization; they locate time, they are not isolated arithmetic timings."""
    native = Native(payload, arg)
    layout = native.rt.manifest.get('tick_layout')
    snap = native.snapshot()
    rows = []
    for _ in range(payload.get('profile_repeats', 5)):
        native.restore(snap)
        native.count.fill_(native.K)
        native.rt.step(profile=True)
        torch.cuda.synchronize()
        rows.append(native.rt._ticks.cpu().tolist())
    native.restore(snap)
    stamps = np.array(rows[1:] if len(rows) > 1 else rows, dtype=np.float64)
    top = [float(np.median(stamps[:, i + 1] - stamps[:, i])) / 1e3 for i in range(4)]
    result = {'top_us': dict(zip(('init', 'sampling_loop', 'train', 'commit'), top))}
    if layout:
        base, phases = layout['body_first_iteration'], layout['body_phases']
        per = []
        for i, ph in enumerate(phases):
            a, b = base + i, base + i + 1
            if b < stamps.shape[1]:
                per.append({**ph, 'us': float(np.median(stamps[:, b] - stamps[:, a])) / 1e3})
        result['body_phases_first_iteration_us'] = per
        result['body_phase_sum_us'] = sum(p['us'] for p in per)
        if 'train' in layout:
            base, train = layout['train'], []
            for i, ph in enumerate(layout['train_phases']):
                if base + i + 1 < stamps.shape[1]:
                    train.append({**ph, 'us': float(np.median(stamps[:, base + i + 1] - stamps[:, base + i])) / 1e3})
            result['train_phases_us'] = train
            result['train_phase_sum_us'] = sum(p['us'] for p in train)
    save(out / 'profile.json', result)
    return {'top_us': result['top_us'], 'body_phase_sum_us': result.get('body_phase_sum_us'),
            'largest_phases': sorted(result.get('body_phases_first_iteration_us', []), key=lambda p: -p['us'])[:8],
            'train_phase_sum_us': result.get('train_phase_sum_us'),
            'largest_train_phases': sorted(result.get('train_phases_us', []), key=lambda p: -p['us'])[:10]}






def kscale(payload, arg, out):
    """Sampling-count scaling: the native binary at runtime count K vs a control compiled for chain K.
    arg: '<build name>:<K1,K2,...>'."""
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control, State, capture_update
    import torch._dynamo.config as dynamo_config
    for key in ('cache_size_limit', 'recompile_limit', 'accumulated_cache_size_limit', 'accumulated_recompile_limit'):
        if hasattr(dynamo_config, key):
            setattr(dynamo_config, key, max(getattr(dynamo_config, key), 1024))
    name, _, ks = arg.partition(':')
    native = Native(payload, name)
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in native.bound())
    snap = native.snapshot()
    variant = payload.get('kscale_control', 'fused_autotune')
    rows = {}
    for K in [int(x) for x in ks.split(',') if x]:
        if K > native.K:
            raise ValueError('requested count exceeds the native maximum')
        native.count.fill_(K)
        g_native = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_native):
            native.rt.step()
        native.restore(snap)
        cfg_k = dict(native.cfg, chain=K)
        started = time.perf_counter()
        S = State(cfg_k, P0, O0, S0, native.images, native.order)
        ctrl = Control(cfg_k, variant)
        g_ctl = capture_update(ctrl, S)
        compile_seconds = time.perf_counter() - started
        snap_c = S.snapshot()
        arms = {'native': (g_native, lambda: native.restore(snap)), variant: (g_ctl, lambda: S.restore(snap_c))}
        samples = {k: [] for k in arms}
        rng = random.Random(4100 + K)
        for _ in range(payload.get('kscale_rounds', 10)):
            order = list(arms)
            rng.shuffle(order)
            for key in order:
                g, reset = arms[key]
                reset()
                samples[key].append(_time_graph(g, 10))
                reset()
        med = {k: float(np.median(v)) for k, v in samples.items()}
        rows[K] = {'medians_ms': med, 'ratio_native_over_control': med['native'] / med[variant],
                   'control_compile_seconds': compile_seconds}
        del ctrl, S, g_ctl
        torch.cuda.empty_cache()
    native.restore(snap)
    result = {'build': name, 'control': variant, 'by_count': rows}
    save(out / 'kscale.json', result)
    return result
