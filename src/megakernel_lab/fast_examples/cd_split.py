"""Device observations and CPU-only audits of the frozen adaptive/energy gates.

GPU workers run native programs and the actual device calibration implementations.
They save inputs and outputs, never an FP64 reference or a qualified verdict. CPU
workers replay the *existing* gate functions against those immutable observations.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import random
from types import FunctionType, SimpleNamespace


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stage_energy():
    """CPU-only corpus conversion and exact recipe initialization for GPU loading."""
    import numpy as np
    import torch
    from .vendor.energy import cops4b
    root = cops4b.DATA.parent / 'energy-prepared'
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all(sha(root / row['file']) == row['sha256'] for row in manifest['files']):
            return manifest
        raise ValueError('CPU-staged energy data changed; preserve evidence and use a fresh root')
    files = []
    for crop in (8, 28):
        for which in ('train', 'test'):
            path = root / f'images-{crop}-{which}.npy'
            if path.exists():
                raise ValueError('incomplete energy data staging; use a fresh root')
            np.save(path, cops4b.images_fp32(dict(crop=crop), which))
            files.append(dict(file=path.name, sha256=sha(path)))
    for name in ('tiny', 'sd', 'wb'):
        cfg = cops4b.config(name)
        params, opt, state, (_, order), count = cops4b.bindings(cfg, 2048)
        path = root / f'bindings-{name}.pt'
        torch.save(dict(cfg=cfg, params=params, opt=opt, state=state, order=order, count=count), path)
        files.append(dict(file=path.name, sha256=sha(path)))
    manifest = dict(passed=True, files=files, scope='exact CPU recipe initialization and FP32 Fashion-MNIST crops')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


@contextmanager
def energy_device_inputs():
    """GPU host performs only checked file loads and device copies for initial inputs."""
    import numpy as np
    import torch
    from .vendor.energy import cops4b
    from .vendor.energy.c_ops4_b import gpu_tasks
    root = cops4b.DATA.parent / 'energy-prepared'
    manifest = json.loads((root / 'manifest.json').read_text())
    records = {r['file']: r for r in manifest['files']}
    checked = set()

    def checked_path(name):
        path = root / name
        if name not in checked:
            if sha(path) != records[name]['sha256']:
                raise ValueError(f'CPU-staged energy input hash mismatch: {name}')
            checked.add(name)
        return path

    def images(cfg, which='train'):
        return np.load(checked_path(f'images-{cfg["crop"]}-{which}.npy'))

    def bindings(cfg, rows, *, device='cpu', images=None):
        name = next((n for n in ('tiny', 'sd', 'wb') if cops4b.config(n) == cfg), None)
        if name is None or rows > 2048:
            raise ValueError('GPU energy inputs require a matching CPU-prepared configuration/order')
        data = torch.load(checked_path(f'bindings-{name}.pt'), weights_only=True, map_location='cpu')
        if data['cfg'] != cfg:
            raise ValueError('CPU-staged energy configuration mismatch')
        move = lambda values: {n: value.to(device) for n, value in values.items()}
        if images is None:
            images = torch.from_numpy(np.load(checked_path(f'images-{cfg["crop"]}-train.npy')))
        return (move(data['params']), move(data['opt']), move(data['state']),
                (images.to(device), data['order'][:rows].to(device)), data['count'].to(device))

    with patches([(cops4b, 'bindings', bindings), (gpu_tasks, 'bindings', bindings),
                  (cops4b, 'images_fp32', images), (gpu_tasks, 'images_fp32', images)]):
        yield


class Observations:
    def __init__(self, out):
        self.root = Path(out) / 'cpu-audit-inputs'
        self.root.mkdir(parents=True, exist_ok=False)
        self.rows = []

    def save(self, kind, data):
        import torch
        path = self.root / f'{len(self.rows):03d}-{kind}.pt'
        torch.save(cpu_tree(dict(kind=kind, **data)), path)
        row = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size, kind=kind)
        self.rows.append(row)
        (self.root / 'manifest.json').write_text(json.dumps(self.rows, indent=2) + '\n')
        return row


@contextmanager
def patches(items):
    old = [(obj, name, getattr(obj, name)) for obj, name, _ in items]
    try:
        for obj, name, value in items:
            setattr(obj, name, value)
        yield
    finally:
        for obj, name, value in reversed(old):
            setattr(obj, name, value)


def _adaptive_observe(store, native, a, label, *, counts=None, causality=False,
                      repeat=True, precision=None, **_):
    import torch
    from .vendor.adaptive import c4a, p8c
    snap = native.snapshot()
    data = dict(label=label, snapshot=snap, inputs=native.inputs(), counts=counts or [a.chunks],
                repeat=repeat, causality=causality, precision=precision, rows={})
    for count in data['counts']:
        native.count.fill_(count)
        reference = p8c.own_reference if precision is not None else c4a.own_reference
        options = dict(precision=precision) if precision is not None else {}
        ref32 = reference(a, snap[0], native.inputs(), torch.float32, count=count, **options)
        output = native.step()
        row = dict(output=output, ref32=ref32)
        if count == a.chunks:
            row['donor'] = c4a.donor_optimizer(a, snap[0], output[1], snap[1], native.sc)
            if repeat:
                native.restore(snap)
                row['again'] = native.step()
        data['rows'][count] = cpu_tree(row)
        native.restore(snap)
    if causality:
        data['causality_result'] = c4a.causality_check(native, a)
    native.count.fill_(a.chunks)
    native.restore(snap)
    receipt = store.save('adaptive-state', data)
    return dict(label=label, counts={}, passed=False, qualification_pending=True, observation=receipt)


def _adaptive_audit(data, case):
    import torch
    from .vendor.adaptive import c4a, p8c  # registers public nearby recipe
    a = c4a.recipe(case)
    precision = data.get('precision')
    gate_module = p8c if precision is not None else c4a
    original_ref, original_donor = gate_module.own_reference, c4a.donor_optimizer

    class Replay:
        def __init__(self):
            self.count = torch.tensor(a.chunks)
            self.sc = data['inputs'][2]
            self.calls = {}

        def snapshot(self):
            return data['snapshot']

        def restore(self, _):
            pass

        def inputs(self):
            return data['inputs']

        def step(self):
            count = int(self.count)
            call = self.calls.get(count, 0)
            self.calls[count] = call + 1
            return data['rows'][count]['again' if call else 'output']

    native = Replay()
    donor_calls = 0

    def ref(a, params, inputs, dtype, *, count, **kwargs):
        if dtype == torch.float32:
            if kwargs.get('precision') != precision:
                raise ValueError('adaptive calibration precision differs from its observation')
            return data['rows'][count]['ref32']
        return original_ref(a, params, inputs, dtype, count=count, **kwargs)

    def donor(*args, **kwargs):
        nonlocal donor_calls
        donor_calls += 1
        if precision is not None or donor_calls % 2:
            return data['rows'][a.chunks]['donor']
        return original_donor(*args, **kwargs)

    with patches([(gate_module, 'own_reference', ref), (c4a, 'donor_optimizer', donor),
                  (c4a, 'causality_check', lambda *_: data['causality_result'])]):
        options = dict(precision=precision) if precision is not None else {}
        return gate_module.check_state(native, a, data['label'], counts=data['counts'],
                                       causality=data['causality'], repeat=data['repeat'], **options)


def _energy_controls(cfg, params, opt, xpos, xneg):
    """Actual GPU calibration arithmetic, with no FP64 model reference."""
    from .vendor.energy.c_ops4_b import checks
    from .vendor.energy.c_ops4_b.controls import Control
    names, decay = checks.names_and_decay(cfg)
    obs = {}
    for label, variant in (('C1', 'eager'), ('C2', 'manual')):
        ctrl = Control(cfg, variant)
        loss, _, grads = ctrl._param_grads(params, xpos, xneg)
        result = ctrl._optimizer(params, {n: opt[f'exp_avg:{n}'] for n in names},
                                 {n: opt[f'exp_avg_sq:{n}'] for n in names},
                                 {n: opt[f'step:{n}'] for n in names}, grads, names, decay)
        obs[label] = checks.pack(names, loss, grads, {n: result[n][0] for n in names},
                                 {n: result[n][1] for n in names}, {n: result[n][2] for n in names})
    return obs


def _energy_reference(data):
    from .vendor.energy.c_ops4_b import checks, reference as R
    from .vendor.energy.examples import energy as recipe
    cfg, params, opt = data['cfg'], data['params'], data['opt']
    names, decay = checks.names_and_decay(cfg)
    loss, _, grads = R.loss_and_grads(checks.p64(params), data['xpos'].double(), data['xneg'].double(),
                                    cfg['alpha'], R.Arith('fp64'), recipe.dims(cfg)['L'])
    result, norm, coef = R.adamw(cfg, names, decay, checks.p64(params), grads,
                                {n: opt[f'exp_avg:{n}'] for n in names},
                                {n: opt[f'exp_avg_sq:{n}'] for n in names},
                                {n: int(opt[f'step:{n}']) for n in names})
    obs = dict(data['controls'])
    obs['R'] = checks.pack(names, loss, grads, {n: result[n][0] for n in names},
                           {n: result[n][1] for n in names}, {n: result[n][2] for n in names})
    return obs, dict(norm=float(norm), coef=float(coef))


def _pending_rows(obs, subject):
    return {q: dict(passed=False, qualification_pending=True, bound=None,
                    errors={k: None for k in (*obs, subject)}) for q in ('loss', 'grads', 'params', 'm', 'v')}


def _energy_observe(store, native, *, label, limit=8):
    import torch
    from .vendor.energy.c_ops4_b import checks, gpu_tasks
    from .vendor.energy.c_ops4_b.controls import Control
    snap = native.snapshot()
    P0, O0, S0 = (dict((k, v.clone()) for k, v in d.items()) for d in native.bound())
    data = dict(label=label, limit=limit, cfg=native.cfg, K=native.K, snapshot=snap,
                params=P0, opt=O0, state=S0, names=native.names, onames=native.onames,
                snames=native.snames, runs=[], increments={})

    def run(count=None, *, full=False):
        output = native.run(count)
        data['runs'].append(cpu_tree(dict(count=native.K if count is None else count,
                                          output=output if full else dict(carry=output['carry']),
                                          after=native.snapshot() if full else None)))
        return output

    a = run(full=True)
    native.restore(snap)
    run(full=True)
    native.restore(snap)
    native.count.fill_(native.K)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native.rt.step()
    native.restore(snap)
    graph.replay()
    torch.cuda.synchronize()
    data['graph'] = cpu_tree(dict(output=native.outputs(), after=native.snapshot()))
    native.restore(snap)
    run(0)
    for k in gpu_tasks._step_indices(native.K, limit):
        native.restore(snap)
        prev = run(k - 1)['carry']['x']
        native.restore(snap)
        nxt = run(k)['carry']['x']
        xi = checks.noise(native.cfg, S0['key'], k - 1, int(S0['t']))
        inc = {}
        for lab, variant in (('C1', 'eager'), ('C2', 'manual')):
            inc[lab] = Control(native.cfg, variant)._chain_step(checks.packed(P0), prev, xi).double() - prev.double()
        data['increments'][k - 1] = cpu_tree(inc)
    xpos = native.images[native.order[int(S0['t'])]]
    data['xpos'], data['xneg'] = xpos, a['carry']['x']
    data['controls'] = _energy_controls(native.cfg, P0, O0, xpos, data['xneg'])
    names, decay = checks.names_and_decay(native.cfg)
    data['optimizer_control'] = Control(native.cfg, 'manual')._optimizer(
        P0, {n: O0[f'exp_avg:{n}'] for n in names}, {n: O0[f'exp_avg_sq:{n}'] for n in names},
        {n: O0[f'step:{n}'] for n in names}, a['grads'], names, decay)
    native.restore(snap)
    receipt = store.save('energy-state', data)
    # Shape-compatible unfinished summary for the historical task wrappers.
    return dict(label=label, passed=False, qualification_pending=True, observation=receipt,
                repeat_bitwise=False, graph_replay_bitwise=False, status_zero=False,
                init_exact=False, commit_exact=False, steps=[dict(k=0, passed=False, errors=dict(native=0.0))],
                update={}, optimizer_tier=dict(passed=False, v1_1_passed=False))


def _energy_audit(data):
    """Replay the unchanged historical gates on CPU, including both optimizer tiers."""
    import torch
    from .vendor.energy.c_ops4_b import checks, gpu_tasks, reference as R
    from .vendor.energy.c_ops4_b.controls import Control
    from .vendor.energy.examples import energy as recipe

    class Replay:
        def __init__(self):
            for name in ('cfg', 'K', 'names', 'onames', 'snames'):
                setattr(self, name, data[name])
            self.count = torch.tensor(self.K)
            self.mutable = data['snapshot']
            self.cursor = 0
            self.rt = SimpleNamespace(step=lambda: None)
            self.images = data['xpos']
            self.order = {int(data['state']['t']): torch.arange(self.images.shape[0])}

        def snapshot(self):
            return self.mutable

        def restore(self, snap):
            self.mutable = snap

        def bound(self):
            return data['params'], data['opt'], data['state']

        def run(self, count=None):
            row = data['runs'][self.cursor]
            self.cursor += 1
            if row['count'] != (self.K if count is None else count):
                raise ValueError('energy native observation order differs from the frozen gate')
            self.current = row['output']
            if row['after'] is not None:
                self.mutable = row['after']
            return self.current

        def outputs(self):
            return self.current

    native = Replay()

    class Graph:
        def replay(self):
            native.current = data['graph']['output']
            native.mutable = data['graph']['after']

    class TorchProxy:
        cuda = SimpleNamespace(CUDAGraph=Graph, graph=lambda _: nullcontext(), synchronize=lambda: None)

        def __getattr__(self, name):
            return getattr(torch, name)

    def increments(cfg, params, prev, k, t, key, subject_next=None):
        _, _, xr = R.chain_step(checks.p64(params), prev.double(), int(k), int(t), cfg,
                                R.Arith('fp64'), recipe.dims(cfg)['L'], cfg['seed'])
        result = dict(data['increments'][k], R=xr - prev.double())
        if subject_next is not None:
            result['native'] = subject_next.double() - prev.double()
        return result

    gate = FunctionType(gpu_tasks.gates.__code__, dict(gpu_tasks.gates.__globals__, torch=TorchProxy()))
    with patches([(checks, 'step_increments', increments),
                  (checks, 'update_observables', lambda *_: _energy_reference(data)),
                  (Control, '_optimizer', lambda *_args, **_kwargs: data['optimizer_control'])]):
        report = gate(native, label=data['label'], limit=data['limit'])
    if native.cursor != len(data['runs']):
        raise ValueError('unused energy observations; gate identity changed')
    return report


@contextmanager
def observe(family, out):
    """Install process-scoped observation adapters; no persistent runtime modification."""
    store = Observations(out)
    if family == 'adaptive':
        from .vendor.adaptive import c4a, p8c
        observe_state = lambda *a, **k: _adaptive_observe(store, *a, **k)
        with patches([(c4a, 'check_state', observe_state), (p8c, 'check_state', observe_state)]):
            yield store
        return
    from .vendor.energy.c_ops4_b import checks, gpu_tasks
    pending = {}

    def update(cfg, params, opt, xpos, xneg):
        controls = _energy_controls(cfg, params, opt, xpos, xneg)
        obs = dict(controls)
        pending[id(obs)] = cpu_tree(dict(cfg=cfg, params=params, opt=opt, xpos=xpos, xneg=xneg, controls=controls))
        return obs, dict(qualification_pending=True)

    def compare(obs, subject='native', reference='R', skip=()):
        data = pending.pop(id(obs))
        data.update(subject=subject, observed=obs[subject], skip=list(skip))
        store.save('energy-update', data)
        return {q: row for q, row in _pending_rows(obs, subject).items() if q not in skip}

    with patches([(gpu_tasks, 'gates', lambda *a, **k: _energy_observe(store, *a, **k)),
                  (checks, 'update_observables', update), (checks, 'compare', compare)]):
        yield store
    if pending:
        raise ValueError('GPU calibration observations were not consumed')


def learning_observation(out, *, initial, finals, trajectories, subject, base='eager'):
    """Called by training harnesses after actual GPU updates, before CPU qualification."""
    import torch
    store = Observations(Path(out) / 'learning')
    return store.save('energy-learning', dict(initial=initial, finals=finals,
                      trajectories={k: torch.as_tensor(v) for k, v in trajectories.items()},
                      subject=subject, base=base))


def _learning_audit(data):
    import torch
    finals, base = data['finals'], data['base']
    names = list(data['initial'])
    flat = lambda p: torch.cat([p[n].double().flatten() for n in names])
    travel = float((flat(finals[base]) - flat(data['initial'])).norm())
    dist = {k: float((flat(p) - flat(finals[base])).norm()) / max(travel, 1e-30)
            for k, p in finals.items() if k != base}
    calibration = max((v for k, v in dist.items() if k != data['subject']), default=0.0)
    finite = all(bool(torch.isfinite(v).all()) for v in data['trajectories'].values())
    bound = max(2 * calibration, 0.02)
    return dict(passed=bool(finite and dist[data['subject']] <= bound), finite=finite,
                distances=dist, calibration=calibration, bound=bound, travel_eager=travel,
                updates={k: len(v) for k, v in data['trajectories'].items()})


def audit(spec, out):
    import torch
    from .vendor.energy.c_ops4_b import checks
    from .cd_driver import save
    root = Path(spec['check_folder'])
    rows, updates, task_directories = [], {}, set()
    for manifest in sorted(root.rglob('cpu-audit-inputs/manifest.json')):
        records = json.loads(manifest.read_text())
        for record in records:
            path = manifest.parent / record['file']
            if path.name != record['file'] or sha(path) != record['sha256']:
                raise ValueError(f'observation hash/path mismatch: {path}')
            data = torch.load(path, weights_only=True, map_location='cpu')
            if data['kind'] == 'adaptive-state':
                result = _adaptive_audit(data, spec['case'])
            elif data['kind'] == 'energy-state':
                result = _energy_audit(data)
            elif data['kind'] == 'energy-update':
                obs, scalars = _energy_reference(data)
                obs[data['subject']] = data['observed']
                comparisons = checks.compare(obs, subject=data['subject'], skip=data['skip'])
                result = dict(passed=all(r['passed'] for r in comparisons.values()),
                              comparisons=comparisons, reference_scalars=scalars)
                updates.setdefault(manifest.parent.parent, []).append(result)
            elif data['kind'] == 'energy-learning':
                result = _learning_audit(data)
            else:
                raise ValueError(f'unknown observation kind {data["kind"]}')
            # Practical-control same-state checks are qualification for the
            # performance arm, or reported diagnostics in the historical training
            # gate. They never became an all-controls-must-pass release gate.
            required = data['kind'] != 'energy-update'
            rows.append(dict(input=str(path.relative_to(root)), input_sha256=record['sha256'],
                             kind=data['kind'], required=required, **result))
            save(Path(out) / 'audit-progress.json', dict(passed=False, complete=False, rows=rows))
        directory = manifest.parent.parent
        task_directories.add(directory.parent if directory.name == 'learning' else directory)
    completed_tasks = {}
    for directory in sorted(task_directories):
        receipt_path = directory / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get('completed') or not receipt.get('result', {}).get('observation_complete'):
            raise ValueError(f'incomplete GPU observation task: {directory}')
        manifest_rows = json.loads((directory / 'cpu-audit-inputs' / 'manifest.json').read_text())
        if receipt['result']['observations'] != manifest_rows:
            raise ValueError(f'GPU receipt does not bind its observation manifest: {directory}')
        completed_tasks[str(directory.relative_to(root))] = sha(receipt_path)
    performance = {}
    for directory, results in updates.items():
        perf_path = directory / 'perf.json'
        if not perf_path.exists():
            continue
        perf = json.loads(perf_path.read_text())
        # Historical perf records every successful calibration call in insertion order.
        variants = [k for k, v in perf['validity'].items() if 'errors' in v]
        if len(variants) != len(results):
            raise ValueError('control observation count differs from performance receipt')
        valid = [k for k, v in zip(variants, results) if v['passed'] and k in perf['medians_ms']]
        fastest = min(valid, key=perf['medians_ms'].get) if valid else None
        ratios = {}
        if fastest:
            import numpy as np
            rng = random.Random(77)
            samples = perf['samples_ms']
            for name in samples:
                if not name.startswith('native:'):
                    continue
                n, bootstrap = len(samples[name]), []
                for _ in range(2000):
                    ids = [rng.randrange(n) for _ in range(n)]
                    bootstrap.append(np.median([samples[name][i] for i in ids]) /
                                     np.median([samples[fastest][i] for i in ids]))
                ratios[name] = dict(ratio=perf['medians_ms'][name] / perf['medians_ms'][fastest],
                                    ci95=[float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))])
        agreement = perf['native_agreement']
        perf_passed = bool(fastest and agreement and all(
            r['bitwise_vs_primary'] and r['status'] == 0 for r in agreement.values()))
        performance[str(directory.relative_to(root))] = dict(passed=perf_passed,
            fastest_valid_control=fastest, ratios_vs_fastest=ratios,
            validity={k: v['passed'] for k, v in zip(variants, results)})
    result = dict(passed=bool(rows) and all(r['passed'] for r in rows if r['required'])
                        and all(r['passed'] for r in performance.values()), complete=True,
                  qualification_pending=False, rows=rows, performance=performance,
                  completed_gpu_tasks=completed_tasks,
                  scope='CPU FP64 numerical gates with actual GPU calibration observations; original thresholds')
    save(Path(out) / 'numerical-audit.json', result)
    return result


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='CPU preparation of installed energy example inputs')
    parser.add_argument('action', choices=['stage-energy'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    os.environ['LATCH_FAST_ROOT'] = str(args.root.resolve())
    print(json.dumps(stage_energy(), indent=2))
