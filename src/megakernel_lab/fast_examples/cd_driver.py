"""Installed adaptive-memory and energy preparation, native checks, controls and training.

Every GPU task starts in a fresh Python process. Models, references and numerical
policies are package data; artifact locations are explicit and no import aliases
or historical dispatcher outputs are needed.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + '\n')


def single_kernel(spec):
    import torch
    if spec['family'] == 'adaptive':
        from .vendor.adaptive.c4a import Native
        native = Native(spec['folder'])
        runtime = native.runtime
    else:
        from .vendor.energy.c_ops4_b.gpu_tasks import Native
        from .cd_split import energy_device_inputs
        with energy_device_inputs():
            native = Native(spec['payload'], spec['case'])
        runtime = native.rt
    snapshot = native.snapshot()
    runtime.step()
    torch.cuda.synchronize()
    native.restore(snapshot)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        runtime.step()
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(passed=len(kernels) == 1, kernels=kernels, blocks=runtime.blocks,
                source_hash=runtime.manifest.get('source_hash'))


def child(spec, out):
    import torch
    torch.set_num_threads(1)
    if spec['device'] == 'gpu':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    family, task = spec['family'], spec['task']
    if task == 'cpu_audit':
        from .cd_split import audit
        result = audit(spec, out)
    elif task == 'single_kernel':
        result = single_kernel(spec)
    elif family == 'adaptive':
        from .vendor.adaptive import p9c
        if task != 'cpu_audit':
            adapted = {k: v for k, v in spec.items() if k not in ('family', 'case', 'device', 'name')}
            adapted['recipe'] = spec['case']
            if task == 'probe':
                adapted.update(build=True, lower=True, interpret=False,
                               capture_options={'init_storage': 'save_tape', 'precision': 'mixed_bf16'})
            if task == 'native_check':
                adapted.update(precision='mixed_bf16', defer_cpu_optimizer=True)
            if task == 'native_check':
                from .cd_split import observe
                with observe(family, out) as observations:
                    result = p9c.GPU_TASKS[task](adapted, out)
                result.update(passed=False, qualification_pending=True,
                              observation_complete=bool(observations.rows), observations=observations.rows)
            else:
                result = (p9c.TASKS if spec['device'] == 'cpu' else p9c.GPU_TASKS)[task](adapted, out)
    else:
        from .vendor.energy import p9d
        from .vendor.energy.c_ops4_b import gpu_tasks
        from .vendor.energy.p8_d import gpu_extra as p8_extra
        from .vendor.energy.p9_d import gpu_extra as p9_extra
        payload = spec['payload']
        if spec['device'] == 'cpu':
            result = p9d.CPU_TASKS[task](payload, spec['case'], out)
        else:
            fn = getattr(p9_extra, task, None) or getattr(p8_extra, task, None) or getattr(gpu_tasks, task)
            from .cd_split import observe, energy_device_inputs
            payload = dict(payload, split_cpu_audit=True)
            with energy_device_inputs(), observe(family, out) as observations:
                result = fn(payload, spec.get('arg', spec['case']), out)
            result.update(passed=False, qualification_pending=True,
                          observation_complete=bool(observations.rows), observations=observations.rows)
        if task == 'perfx' and 'passed' not in result:
            times, agreement = result.get('medians_ms', {}), result.get('native_agreement', {})
            result['passed'] = bool(times and result.get('fastest_valid_control') and agreement
                                    and all(math.isfinite(v) and v > 0 for v in times.values())
                                    and all(a['bitwise_vs_primary'] and a['status'] == 0 for a in agreement.values()))
            result['verdict_scope'] = 'timing completion; arithmetic/learning gates are separate receipts'
    from importlib.metadata import version
    receipt = dict(passed=bool(result.get('passed')), result=result, family=family, task=task,
                   qualification_pending=bool(result.get('qualification_pending')),
                   completed=bool(result.get('passed') or result.get('observation_complete')),
                   device=spec['device'], torch=torch.__version__, package_version=version('megakernel-lab'))
    save(out / 'receipt.json', receipt)
    return receipt


def one(spec, directory):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'task.json', spec)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1')
    env.pop('PYTHONPATH', None)
    if spec['device'] == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    started = time.perf_counter()
    try:
        with (out / 'task.log').open('w') as log:
            proc = subprocess.run([sys.executable, '-I', '-m', 'megakernel_lab.fast_examples.cd_driver',
                                   '--child', str(out / 'task.json'), str(out)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=14400)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = 'timeout'
    receipt = json.loads((out / 'receipt.json').read_text()) if (out / 'receipt.json').exists() else {}
    status = dict(passed=bool(receipt.get('passed')) and code == 0, returncode=code,
                  completed=bool(receipt.get('completed', receipt.get('passed'))) and code == 0,
                  qualification_pending=bool(receipt.get('qualification_pending')),
                  seconds=time.perf_counter() - started, receipt=str(out / 'receipt.json'))
    print(json.dumps(dict(task=spec['name'], **status)), flush=True)
    return status


def command(args):
    root = args.root.resolve()
    cases = args.cases or (['m8k', 'm2k', 'm8k_c64'] if args.family == 'adaptive' else ['sd', 'wb'])
    allowed = {'adaptive': {'m8k', 'm2k', 'm8k_c64'}, 'energy': {'sd', 'wb', 'tiny'}}[args.family]
    if set(cases) - allowed:
        raise ValueError('unknown cases: ' + ', '.join(sorted(set(cases) - allowed)))
    if args.action == 'report':
        rows = {str(p.relative_to(root)): json.loads(p.read_text())
                for case in cases for p in (root / args.family / case).rglob('receipt.json')}
        print(json.dumps(rows, indent=2))
        if args.json:
            save(args.json, rows)
        return 0
    tasks = []
    for case in cases:
        base = root / args.family / case
        cache = str(base / 'cache')
        prepared = base / 'prepare'
        common = dict(family=args.family, case=case, cache_dir=cache)
        if args.family == 'energy':
            common['payload'] = dict(build_attempt=str(prepared), cache_dir=cache, step_limit=8)
        if args.action == 'prepare':
            spec = dict(common, name='prepare', task='probe' if args.family == 'adaptive' else 'capture', device='cpu')
            if args.family == 'energy':
                spec['payload']['order_rows'] = 2048
            destination = prepared if args.family == 'adaptive' else prepared / ('capture-' + case)
            tasks.append((spec, destination))
        elif args.action == 'compare':
            if args.family == 'adaptive':
                for rnd in range(args.rounds):
                    arms = ['native_combined', 'bf16', 'bf16_ma', 'fp32']
                    if rnd % 2: arms.reverse()
                    spec = dict(common, name=f'compare-r{rnd}', task='timing', device='gpu',
                                arms=arms, folders={'native_combined': str(prepared)}, rounds=1, iters=30)
                    tasks.append((spec, base / spec['name']))
            else:
                spec = dict(common, name='compare', task='perfx', device='gpu')
                spec['payload'].update(perf_controls=['bf16_fused_autotune', 'fused_autotune_cd',
                                                      'fused_autotune', 'pieces_autotune'],
                                       perf_rounds=max(3, args.rounds), perf_seed=4001)
                tasks.append((spec, base / 'compare'))
        elif args.action == 'verify':
            # Numerical qualification is completed on CPU from GPU observations.
            if args.arms == 'cpu':
                spec = dict(common, name='verify-cpu', task='cpu_audit', device='cpu',
                            check_folder=str(base))
                tasks.append((spec, base / f'verify-cpu-{time.time_ns()}'))
            else:
                kernel = dict(common, name='kernel', task='single_kernel', device='gpu', folder=str(prepared))
                tasks.append((kernel, base / 'kernel'))
                spec = dict(common, name='verify', task='native_check' if args.family == 'adaptive' else 'check',
                            device='gpu', folder=str(prepared), train_updates=10)
                tasks.append((spec, base / 'verify'))
        elif args.action == 'train' and args.family == 'energy':
            spec = dict(common, name='train', task='train', device='gpu')
            spec['payload'].update(train_updates=400, train_controls=['eager', 'manual'])
            tasks.append((spec, base / 'train'))
            spec = dict(common, payload=dict(common['payload']), name='train-control', task='train_control_cfg',
                        arg=case + ':bf16_fused_autotune', device='gpu')
            spec['payload']['train_updates'] = 400
            tasks.append((spec, base / 'train-control'))
        else:
            raise ValueError('adaptive full-learning requires the separately staged corpus; this CLI ships synthetic checks')
    # Stage shared public data once before parallel captures: the historical downloader
    # uses one .part file per IDX and is not a concurrent download protocol.
    if args.action == 'prepare' and args.family == 'energy':
        staging = root / 'energy' / 'data-preparation'
        if not staging.exists():
            status = one(dict(family='energy', case='data', name='data', task='data',
                              device='cpu', payload={}), staging)
            if not status['passed']:
                return 1
        else:
            prior = json.loads((staging / 'receipt.json').read_text())
            if not prior.get('passed'):
                raise ValueError('prior data staging failed; preserve it and use a new root')
        from .cd_split import stage_energy
        stage_energy()
    workers = args.workers if all(s['device'] == 'cpu' for s, _ in tasks) else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda pair: one(*pair), tasks))
    save(root / args.family / f'{args.action}-{time.time_ns()}.json', results)
    return 0 if results and all(r['completed'] for r in results) else 1


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--child':
        raise SystemExit('Use python -m megakernel_lab.fast_examples instead')
    spec, out = json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3])
    try:
        answer = child(spec, out)
    except Exception as exc:
        answer = dict(passed=False, error=repr(exc), traceback=traceback.format_exc())
        save(out / 'receipt.json', answer)
    raise SystemExit(0 if answer.get('completed', answer['passed']) else 1)
