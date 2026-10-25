"""Fast-example driver: installed public API, shipped user models and conventional controls.

Actions (run inside the VM container, cwd = harness bundle):
  prepare-all FAMILY WORKERS [CASES...]   CPU: capture + build every native variant (parallel one-thread workers)
  gpu-all FAMILY ROUNDS [CASES...]        H100: every arm per case, fresh process per arm, rotated order per round
  audit-all FAMILY WORKERS [CASES...]     CPU: independent references at fresh/trained states vs every arm
  prepare-one / run-one / audit-one       single-case children (used by the *-all actions)

The compiler is frozen: this file never imports compiler internals, only megakernel_lab.operation_compiler public
names through the family modules. Families (fam_*.py) supply the user model, the capture/build call, conventional
control engines and the CPU reference. Native arms are named 'latch*'; each native variant is a separate build.
Layout: <root>/<family>/<case>/{prepare.json, initial.pt, <variant>/, <arm>-r<k>.json, <arm>-<state>.pt, audit.json}
"""
from pathlib import Path
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get('LATCH_FAST_ROOT', './latch-fast')).resolve()
WARM_UPDATES, CLOCKS, UPDATES_PER_CLOCK, POOL = 40, 3, 100, 8


def fam(name):
    return importlib.import_module('megakernel_lab.fast_examples.fam_' + name)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def init_torch():
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('highest')


def cp(d):
    return {n: v.detach().cpu().clone() for n, v in d.items()}


def peak():
    import torch
    return dict(allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                reserved_mib=torch.cuda.max_memory_reserved() / 2**20)


def case_dir(family, case):
    return ROOT / family / case


def prepare_one(family, case):
    import torch
    init_torch()
    F = fam(family)
    c = F.CASES[case]
    d = case_dir(family, case)
    if d.exists():
        raise FileExistsError(f'{d} already exists; use a new --root to preserve receipts')
    d.mkdir(parents=True)
    result = dict(family=family, case=case, config=c, variants={})
    started = time.perf_counter()
    try:
        captured, initial, info = F.capture(c)
        result.update(capture_seconds=time.perf_counter() - started, **info)
        torch.save(initial, d / 'initial.pt')
        recaptured = {}
        requested = os.environ.get('LATCH_FAST_VARIANTS', 'latch,latch-bf16proj').split(',')
        for variant, kwargs in F.native_variants(c).items():
            if variant not in requested:
                continue
            t = time.perf_counter()
            target = d / variant
            kwargs = dict(kwargs)
            capture_precision = kwargs.pop('_capture_precision', None)
            try:
                source = captured
                if capture_precision is not None:   # separately declared capture recipe (e.g. bf16_projections)
                    if capture_precision not in recaptured:
                        recaptured[capture_precision] = F.capture(c, precision=capture_precision)[0]
                    source = recaptured[capture_precision]
                receipt = F.build(source, target, d / ('cache-' + variant), kwargs) if hasattr(F, 'build') else \
                    source.build(target, cache_dir=d / ('cache-' + variant), arch='sm_90', **kwargs)
                if capture_precision is not None:
                    receipt = dict(receipt, capture_precision=capture_precision)
                result['variants'][variant] = dict(ok=True, kwargs=kwargs, seconds=time.perf_counter() - t,
                                                   receipt=receipt)
            except Exception:
                result['variants'][variant] = dict(ok=False, kwargs=kwargs, seconds=time.perf_counter() - t,
                                                   error=traceback.format_exc()[-6000:])
        result['ok'] = bool(result['variants']) and all(v['ok'] for v in result['variants'].values())
    except Exception:
        result.update(ok=False, error=traceback.format_exc()[-8000:])
    result['seconds'] = time.perf_counter() - started
    dump(d / 'prepare.json', result)
    print(json.dumps(dict(case=case, ok=result['ok'],
                          variants={k: v['ok'] for k, v in result['variants'].items()})), flush=True)
    return result['ok']


def add_variant(family, case, variant):
    """Build one more declared native variant into an existing case directory (keeps all prior receipts)."""
    import torch
    init_torch()
    F = fam(family)
    c = F.CASES[case]
    d = case_dir(family, case)
    prep = json.loads((d / 'prepare.json').read_text())
    kwargs = dict(F.native_variants(c)[variant])
    capture_precision = kwargs.pop('_capture_precision', None)
    t = time.perf_counter()
    try:
        captured = F.capture(c, **({'precision': capture_precision} if capture_precision else {}))[0]
        receipt = F.build(captured, d / variant, d / ('cache-' + variant), kwargs) if hasattr(F, 'build') else \
            captured.build(d / variant, cache_dir=d / ('cache-' + variant), arch='sm_90', **kwargs)
        prep['variants'][variant] = dict(ok=True, kwargs=kwargs, seconds=time.perf_counter() - t, receipt=receipt,
                                         added_later=True)
    except Exception:
        prep['variants'][variant] = dict(ok=False, kwargs=kwargs, seconds=time.perf_counter() - t,
                                         error=traceback.format_exc()[-6000:], added_later=True)
    prep['ok'] = any(v['ok'] for v in prep['variants'].values())
    dump(d / 'prepare.json', prep)
    print(json.dumps(dict(case=case, variant=variant, ok=prep['variants'][variant]['ok'])), flush=True)


def kernel_count(trace_path):
    events = json.loads(Path(trace_path).read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    names = {}
    for e in kernels:
        names[e.get('name', '?')[:120]] = names.get(e.get('name', '?')[:120], 0) + 1
    return len(kernels), names


def run_one(family, case, arm, rnd):
    import torch
    init_torch()
    F = fam(family)
    c = F.CASES[case]
    d = case_dir(family, case)
    out = d / f'{arm}-r{rnd}.json'
    t = time.perf_counter()
    r = dict(family=family, case=case, arm=arm, round=rnd)
    try:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        engine = F.engine(c, arm, d)
        torch.cuda.synchronize()
        r.update(setup_seconds=time.perf_counter() - t, setup_peak=peak(), compiled=engine.compiled,
                 captured=engine.graph is not None, device=torch.cuda.get_device_name(),
                 arm_label=getattr(engine, 'label', arm))
        initial = torch.load(d / 'initial.pt', weights_only=True)
        engine.load(initial)
        pool = F.pool(c, 'cuda', engine)
        losses = []
        for i in range(WARM_UPDATES):
            engine.stage(pool[i % POOL])
            engine.step()
            losses.append(float(engine.loss))
        trained = engine.state()
        r['losses40'] = losses
        common = d / 'common-trained.pt'
        if arm == F.REFERENCE_ARM and rnd == 0:
            torch.save(trained, common)
        if rnd == 0:
            for state, start in (('fresh', initial), ('trained', None)):
                if start is None:
                    if not common.exists():
                        r['trained_state_missing'] = True
                        continue
                    start = torch.load(common, weights_only=True)
                engine.load(start)
                engine.stage(pool[1])
                before = engine.state()
                engine.step()
                torch.cuda.synchronize()
                torch.save(dict(before=before, after=engine.state(), gradients=cp(engine.grads),
                                loss=engine.loss.detach().float().cpu().clone()), d / f'{arm}-{state}.pt')
        engine.load(trained)
        torch.cuda.reset_peak_memory_stats()
        times, stages = [], []
        for _ in range(CLOCKS):
            events = []
            for i in range(UPDATES_PER_CLOCK):
                t0 = time.perf_counter()
                engine.stage(pool[i % POOL])
                stages.append((time.perf_counter() - t0) * 1000)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                engine.step()
                b.record()
                events.append((a, b))
            torch.cuda.synchronize()
            times.append(sum(a.elapsed_time(b) for a, b in events) / UPDATES_PER_CLOCK)
        r.update(update_ms=times, median_update_ms=statistics.median(times),
                 stage_host_median_ms=statistics.median(stages), replay_peak=peak(),
                 finite=engine.finite(), final_loss=float(engine.loss))
        if rnd == 0:
            trace = d / f'{arm}-trace.json'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                engine.step()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace))
            r['cuda_kernel_count'], r['cuda_kernel_names'] = kernel_count(trace)
            trace.unlink()
            saved = engine.state()
            engine.stage(pool[2])
            engine.step()
            after = engine.state()
            engine.load(saved)
            engine.stage(pool[2])
            engine.step()
            again = engine.state()
            r['saved_restart_exact'] = all(torch.equal(after[k][n], again[k][n]) for k in after for n in after[k])
        r['ok'] = True
    except Exception:
        r.update(ok=False, error=traceback.format_exc()[-8000:])
    r['seconds'] = time.perf_counter() - t
    dump(out, r)
    print(json.dumps({k: r.get(k) for k in ('case', 'arm', 'round', 'ok', 'median_update_ms', 'cuda_kernel_count')}),
          flush=True)
    return r['ok'] and r.get('finite', False)


def rel(x, y):
    return float((x.double() - y.double()).norm() / y.double().norm().clamp_min(1e-30))


def rms(x, y):
    return float((x.double() - y.double()).square().mean().sqrt())


def audit_one(family, case):
    """Frozen predeclared gates (FAST_PREDECLARATION.md, unchanged): reference-calibrated loss/gradient/parameter/delta."""
    import torch
    init_torch()
    F = fam(family)
    c = F.CASES[case]
    d = case_dir(family, case)
    rows = []
    for state in ('fresh', 'trained'):
        source = d / ('initial.pt' if state == 'fresh' else 'common-trained.pt')
        cal_path = d / f'{F.REFERENCE_ARM}-{state}.pt'
        if not source.exists() or not cal_path.exists():
            rows.append(dict(state=state, missing_reference_input=True, passed=False))
            continue
        ref = F.reference(c, torch.load(source, weights_only=True))
        eager = torch.load(cal_path, weights_only=True)
        calibration = dict(loss=abs(float(eager['loss'] - ref['loss'])),
                           gradient=max(rel(eager['gradients'][n], g) for n, g in ref['gradients'].items()))
        prep = json.loads((d / 'prepare.json').read_text())
        expected = set(F.conventional_arms(c) if hasattr(F, 'conventional_arms') else F.CONVENTIONAL)
        expected.update(prep.get('variants', {}))
        requested = os.environ.get('LATCH_FAST_ARMS', '').split(',')
        if requested != ['']:
            expected &= set(requested)
        for arm in sorted(expected):
            path = d / f'{arm}-{state}.pt'
            if not path.exists():
                rows.append(dict(arm=arm, state=state, missing=True, passed=False))
                continue
            got = torch.load(path, weights_only=True)
            checks = []
            le = abs(float(got['loss'] - ref['loss']))
            lb = max(.01, 4 * calibration['loss'])
            checks.append(dict(kind='loss', error=le, bound=lb, passed=math.isfinite(le) and le <= lb))
            before = ref['before_parameters']
            for n, g in ref['gradients'].items():
                for kind, base, absolute in (('gradient', .02, 1e-6), ('parameter', .001, 0.), ('delta', .03, 1e-7)):
                    if kind == 'gradient' and n not in got['gradients']:
                        continue   # gradient_outputs='consumed' builds publish no gradients by design
                    if kind == 'gradient':
                        rr, ee, gg = g, eager['gradients'][n], got['gradients'][n]
                    else:
                        rr = ref['after_parameters'][n]
                        ee = eager['after']['parameters'][n]
                        gg = got['after']['parameters'][n]
                        if kind == 'delta':
                            p = before[n]
                            rr, ee, gg = rr - p, ee - p, gg - p
                    bound = max(base, 4 * rel(ee, rr))
                    error = rel(gg, rr)
                    absolute_error = rms(gg, rr)
                    checks.append(dict(name=n, kind=kind, error=error, bound=bound, reference_spread=rel(ee, rr),
                                       rms=absolute_error,
                                       passed=math.isfinite(error) and (error <= bound or absolute_error <= absolute)))
            failed = [x for x in checks if not x['passed']]
            rows.append(dict(arm=arm, state=state, passed=not failed, n_checks=len(checks), n_failed=len(failed),
                             worst_failed=sorted(failed, key=lambda x: -x['error'] / max(x['bound'], 1e-30))[:8],
                             reference_calibration=calibration,
                             inconclusive=calibration['gradient'] > .05 or calibration['loss'] > .02,
                             checks=checks))
    dump(d / 'audit.json', rows)
    print(json.dumps(dict(case=case, results={f"{r.get('arm')}:{r['state']}": r['passed'] for r in rows})), flush=True)
    return all(r['passed'] and not r.get('inconclusive') for r in rows if r.get('arm', 'latch').startswith('latch'))


def children(action, family, workers_or_rounds, cases):
    from concurrent.futures import ThreadPoolExecutor
    F = fam(family)
    cases = cases or list(F.CASES)
    log_dir = ROOT / family / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(args, timeout):
        started = time.time()
        try:
            p = subprocess.run([sys.executable, '-I', '-m', 'megakernel_lab.fast_examples.suite', *map(str, args)], text=True, capture_output=True,
                               timeout=timeout)
            code, text = p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired as e:
            code, text = 'timeout', str(e.stdout or '') + str(e.stderr or '')
        (log_dir / ('-'.join(map(str, args)) + '.log')).write_text(text)
        return dict(args=list(map(str, args)), returncode=code, seconds=round(time.time() - started, 2))

    if action == 'prepare-all':
        with ThreadPoolExecutor(max_workers=int(workers_or_rounds)) as pool:
            result = list(pool.map(lambda n: run(('prepare-one', family, n), F.PREPARE_TIMEOUT), cases))
    elif action == 'audit-all':
        with ThreadPoolExecutor(max_workers=int(workers_or_rounds)) as pool:
            result = list(pool.map(lambda n: run(('audit-one', family, n), F.AUDIT_TIMEOUT), cases))
    else:   # gpu-all: strictly sequential on one otherwise idle GPU; arm order rotates per round
        result = []
        # Longer verification: LATCH_FAST_ROUND_START>0 appends new rounds (receipts r<start>..) without touching the
        # first-pass receipts; LATCH_FAST_ARMS limits the arms (comma list). Round 0 alone saves audit states/profiles.
        start = int(os.environ.get('LATCH_FAST_ROUND_START', '0'))
        only = [x for x in os.environ.get('LATCH_FAST_ARMS', '').split(',') if x]
        for rnd in range(start, start + int(workers_or_rounds)):
            for n in cases:
                prep = case_dir(family, n) / 'prepare.json'
                if not prep.exists() or not json.loads(prep.read_text()).get('ok'):
                    continue   # no native variant built: recorded as unsupported, controls not timed
                arms = F.arms(F.CASES[n], json.loads(prep.read_text()))
                if only:
                    arms = [x for x in arms if x in only]
                if not arms:
                    raise ValueError(f'no requested arms exist for {n}')
                k = rnd % len(arms)
                order = arms[k:] + arms[:k]
                if rnd == 0 and F.REFERENCE_ARM in order:   # the common trained state must exist first
                    order.remove(F.REFERENCE_ARM)
                    order.insert(0, F.REFERENCE_ARM)
                for arm in order:
                    result.append(run(('run-one', family, n, arm, rnd), F.RUN_TIMEOUT))
    dump(ROOT / family / f'{action}-{int(time.time())}.json', result)
    print(json.dumps(dict(action=action, family=family, n=len(result),
                          failed=[r for r in result if r['returncode'] != 0])), flush=True)
    if action == 'gpu-all':
        # A failed optional control remains a failed receipt. Completion requires
        # every requested native/calibration arm and an actual compiled control.
        for case in cases:
            rows = [r for r in result if r['args'][2] == case]
            required = [r for r in rows if r['args'][3].startswith('latch') or r['args'][3] == F.REFERENCE_ARM]
            controls = [r for r in rows if r['args'][3].startswith(('compiled', 'amp-'))]
            if not required or any(r['returncode'] != 0 for r in required):
                return False
            if not controls or not any(r['returncode'] == 0 for r in controls):
                return False
        return bool(result)
    return bool(result) and all(r['returncode'] == 0 for r in result)


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare-one':
        raise SystemExit(0 if prepare_one(sys.argv[2], sys.argv[3]) else 1)
    elif action == 'run-one':
        raise SystemExit(0 if run_one(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])) else 1)
    elif action == 'add-variant':
        for case in sys.argv[4:]:
            add_variant(sys.argv[2], case, sys.argv[3])
    elif action == 'audit-one':
        raise SystemExit(0 if audit_one(sys.argv[2], sys.argv[3]) else 1)
    elif action in ('prepare-all', 'gpu-all', 'audit-all'):
        raise SystemExit(0 if children(action, sys.argv[2], sys.argv[3], sys.argv[4:]) else 1)
    else:
        raise SystemExit(f'unknown action {action}')
