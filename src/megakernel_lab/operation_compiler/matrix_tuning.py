"""Release steps 16 and 13: bounded automatic matrix-implementation selection with a reusable choice cache.

Step 13 adds ``strategy='mixed'`` (the default): per-group choices from ``ops.matrix_selection.MixedSearch``
(uniform families, then single-group deviations of the largest product groups, then their combination), all judged
by complete-update time. ``strategy='uniform'`` reproduces the step-16 search (whole-program body classes only).

Opt-in through the normal module API::

    artifact = captured.build(directory, cache_dir=..., execution="compose",
                              tuning=MatrixTuning(cache="/owned/latch-tuning"))

What it does (nothing here writes CUDA or plans a schedule):

1. Lowers the program once with the existing composed planner and records every BF16 product the matrix
   provider sees, including products the native path keeps (P5C), with the bodies that pass the existing
   structural/resource check (``matrix_adapters.candidates``).
2. Derives at most ``max_candidates`` whole-program variants from two existing body classes - masked TMA /
   warpgroup ``wgmma`` (P-OPS4-B) and ``mma.sync`` (popsc) - plus the existing legal selection. A variant is
   kept only if it changes at least one product to a different admitted body; duplicates are dropped.
3. Builds the variants (bounded worker pool) and runs a quick complete-update timing of each on private scratch
   copies of the artifact's bindings. A one-update sanity comparison with the existing selection (declared
   Stage-3 BF16 loss/parameter ceilings) excludes a variant that disagrees.
4. Verifies the fastest variant against the existing selection with interleaved complete-update rounds and
   adopts it only for a verified gain of at least ``min_gain``; otherwise the existing selection is kept.
5. Caches the verdict under the GPU/toolchain, implementation version, execution/precision and program
   structure (all product dimensions, strides, offsets, dtypes and surrounding operations). A repeat call
   reuses a valid entry without any measurement; entries whose bodies are no longer admitted are rejected.

Tuning never touches the caller's tensors or random state: variants load their own device copies of the
captured bindings, the final artifact's bindings are the untouched capture snapshot, and host/device RNG
states are checked unchanged. Timing scope is one complete generated update (forward, loss, backward,
optimizer and state writes); it is not a standalone matrix benchmark.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

TUNING_VERSION = 'step13-matrix-selection-v1'
WGMMA = ('pops4b_tma', 'pops4b_wg')
MMA_SYNC = ('popsc_bf16', 'popsc_bf16_tail', 'popsc_bf16_edge')
NATIVE_CLASS = {'p5c': 'bf16_wgmma_k16', 'popsc': 'bf16_mma_sync_k16'}
# Declared Stage-3 mixed-precision ceilings (atol, rtol) used by LATCH-MATRIX for BF16 products
# (research/latch_ops/STAGE3_POLICY.json); here only a sanity screen between existing bodies.
SANITY = {'loss': (2e-3, 2e-3), 'parameters': (5e-5, 2e-3)}


@dataclass(frozen=True)
class MatrixTuning:
    cache: str                       # directory holding one JSON entry per tuning key
    max_candidates: int = 4          # whole-program variants measured, including the existing selection
    max_seconds: float = 1800.0      # bound on builds + quick measurements (verification always completes)
    min_gain: float = 0.01           # verified complete-update gain needed to replace the existing selection
    quick_samples: int = 3
    quick_updates: int = 10
    verify_rounds: int = 7
    verify_updates: int = 20
    build_workers: int = 4
    numerics: str = 'declared'       # 'declared' (bodies agree within declared tolerance) | 'bitwise'
    keep_variants: bool = False      # keep per-variant artifacts (diagnostics) instead of removing them
    # Step 13
    strategy: str = 'mixed'          # 'mixed' (per-group choices, step 13) | 'uniform' (step-16 body classes)
    max_builds: int = 11             # mixed: distinct variants built, including the default selection
    top_groups: int = 3              # mixed: largest product groups (by flops) refined after the uniform round
    alternatives: int = 2            # mixed: other families tried per refined group
    baseline_policy: str | None = None   # default-selection policy (None: composed default; 'donor': frozen rule)

    def __post_init__(self):
        if not 2 <= self.max_candidates <= 4:
            raise ValueError('max_candidates must be 2..4 (the existing selection plus 1-3 alternatives)')
        if self.numerics not in ('declared', 'bitwise'):
            raise ValueError("numerics must be 'declared' or 'bitwise'")
        if self.min_gain < 0 or self.max_seconds <= 0 or self.build_workers < 1:
            raise ValueError('invalid tuning bounds')
        if self.strategy not in ('mixed', 'uniform'):
            raise ValueError("strategy must be 'mixed' or 'uniform'")
        if not 2 <= self.max_builds <= 16 or not 1 <= self.top_groups <= 6 or not 1 <= self.alternatives <= 3:
            raise ValueError('mixed search bounds: max_builds 2..16, top_groups 1..6, alternatives 1..3')
        if self.baseline_policy not in (None, 'donor', 'step13'):
            raise ValueError("baseline_policy must be None, 'donor' or 'step13'")


# ------------------------------------------------------------------------------------------------ identities
def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def implementation_version():
    """Hash of the compiler/body sources that decide what a choice means (any change invalidates entries)."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(TUNING_VERSION.encode())
    for path in sorted([*root.glob('*.py'), *root.glob('matrix_adapters/*.py'), *root.glob('kernels/*')]):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_fingerprint(arch):
    import torch
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    try:
        nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1]
    except Exception as error:      # recorded, never guessed
        nvcc = f'unavailable: {error!r}'
    return {'gpu': props.name, 'sm_count': props.multi_processor_count, 'capability': [props.major, props.minor],
            'cuda_runtime': torch.version.cuda, 'torch': torch.__version__, 'nvcc': nvcc, 'arch': arch}


def graph_signature(ir):
    """Program structure without parameter values or floating scalars (learning rate, momentum, scales)."""
    data = ir.to_dict() if hasattr(ir, 'to_dict') else ir
    values = data.get('values', {})

    def norm(item):
        if isinstance(item, dict) and set(item) == {'ref'}:
            v = values.get(item['ref'], {})
            return ['ref', v.get('shape'), v.get('dtype'), v.get('strides'), v.get('storage_offset', 0), v.get('role')]
        if isinstance(item, dict):
            return {k: norm(v) for k, v in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [norm(v) for v in item]
        if isinstance(item, float):
            return 'float'
        return item

    ops = [[op.get('target'), norm(op.get('args', [])), norm(op.get('kwargs', {}))]
           for op in data.get('operations', [])]
    inputs = [[values.get(n, {}).get(k) for k in ('shape', 'dtype', 'strides', 'role')] for n in data.get('inputs', [])]
    meta = data.get('metadata', {})
    return _sha({'ops': ops, 'inputs': inputs, 'precision_policy': meta.get('precision_policy'),
                 'optimizer': type(meta.get('optimizer')).__name__ if not isinstance(meta.get('optimizer'), dict)
                 else sorted(meta['optimizer'])})


def tuning_key(fingerprint, impl, execution, precision, arch, structure):
    """Products (dimensions, strides, offsets, dtypes) are a function of the program structure and compiler."""
    parts = {'version': TUNING_VERSION, 'environment': fingerprint, 'implementation': impl,
             'execution': execution, 'precision_policy': precision, 'arch': arch, 'program': structure}
    return _sha(parts), parts


# ------------------------------------------------------------------------------------------------ variants
def _class(row, body):
    if body == 'native':
        return NATIVE_CLASS.get(row['native'], 'native:' + str(row['native']))
    return row.get('classes', {}).get(body, body)


def variants(products, *, max_candidates=4, numerics='declared'):
    """Whole-program choice maps (``{problem_key: body}``) derived from the existing admitted bodies.

    Returns [(label, choices)] starting with the existing selection (empty map). Products with identical keys
    share one choice. ``numerics='bitwise'`` keeps only bodies of the existing selection's arithmetic class.
    """
    rows = {}
    for row in products:
        rows.setdefault(row['key'], row)

    def prefer(row, family, native_member):
        default = row['default']
        if default in family or (default == 'native' and row['native'] == native_member):
            return default
        for body in family:
            if body in row['admitted']:
                return body
        return default

    out, seen = [('existing', {})], {tuple(sorted((k, r['default']) for k, r in rows.items()))}
    for label, family, native_member in (('wgmma', WGMMA, 'p5c'), ('mma_sync', MMA_SYNC, 'popsc')):
        choices = {}
        for key, row in rows.items():
            body = prefer(row, family, native_member)
            if numerics == 'bitwise' and _class(row, body) != _class(row, row['default']):
                body = row['default']
            choices[key] = body
        identity = tuple(sorted(choices.items()))
        if identity in seen:
            continue
        seen.add(identity)
        out.append((label, {k: v for k, v in choices.items() if v != rows[k]['default']}))
        if len(out) == max_candidates:
            break
    return out


# ------------------------------------------------------------------------------------------------ cache
def load_entry(cache, key, parts):
    """Return (entry | None, reason). Never trusts an entry whose identity differs in any recorded field."""
    path = Path(cache) / f'{key}.json'
    if not path.exists():
        return None, 'miss'
    try:
        entry = json.loads(path.read_text())
    except ValueError as error:
        return None, f'rejected: unreadable entry ({error})'
    if entry.get('schema') != TUNING_VERSION or entry.get('key') != key or entry.get('parts') != parts:
        return None, 'rejected: entry identity differs (version, GPU/toolchain, implementation or program)'
    if entry.get('verdict') not in ('existing', 'tuned') or not isinstance(entry.get('choices'), dict):
        return None, 'rejected: malformed verdict'
    return entry, 'hit'


def validate_choices(entry, products):
    """Every cached body must still be admitted for its product (or be the native/default path).

    ``products`` is the inventory of the lowering that applied the cached choices.
    """
    rows = {r['key']: r for r in products}
    for key, body in entry['choices'].items():
        row = rows.get(key)
        if row is None:
            return f'rejected: cached product {key} not in this program'
        if body != 'native' and body != row['default'] and body not in row['admitted']:
            return f'rejected: cached body {body} is not admitted for {key}'
    return None


def store_entry(cache, entry):
    path = Path(cache)
    path.mkdir(parents=True, exist_ok=True)
    temp = path / f".{entry['key']}.{os.getpid()}.tmp"
    temp.write_text(json.dumps(entry, indent=2, sort_keys=True) + '\n')
    temp.replace(path / f"{entry['key']}.json")


# ------------------------------------------------------------------------------------------------ measurement
def _rng_snapshot():
    import torch
    return torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]


def _rng_equal(a, b):
    import torch
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


def _time(update, samples, updates):
    import torch
    values = []
    for _ in range(samples):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(updates):
            update.step()
        stop.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(stop) / updates)
    return sorted(values)


def _first_update(update):
    """One update from the private scratch copy of the initial state (loss + updated parameters)."""
    import torch
    update.step()
    torch.cuda.synchronize()
    loss = update.runtime.loss.detach().reshape(-1)[:1].double().cpu()
    params = {k: v.detach().double().cpu() for k, v in update.runtime.parameters.items()}
    return loss, params


def _agrees(candidate, reference):
    loss, params = candidate
    rloss, rparams = reference
    worst = 0.0

    def ratio(x, y, gate):
        atol, rtol = SANITY[gate]
        return float(((x - y).abs() / (atol + rtol * y.abs())).max()) if x.numel() else 0.0

    if not bool(loss.isfinite().all()):
        return False, math.inf
    worst = max(worst, ratio(loss, rloss, 'loss'))
    for name, value in params.items():
        if not bool(value.isfinite().all()):
            return False, math.inf
        worst = max(worst, ratio(value, rparams[name], 'parameters'))
    return worst <= 1.0, worst


# ------------------------------------------------------------------------------------------------ entry point
def prepare_tuned(captured, directory, *, cache_dir, arch, retain_all, policy, precision_policy, compact_codegen,
                  execution, compose_selection, tuning: MatrixTuning, gradient_outputs='returned',
                  fast_build=False, recurrent_gradients='inline'):
    """``prepare_module`` with bounded matrix tuning; writes the selected artifact to ``directory``."""
    import torch
    from . import api
    from .lower import LoweringError
    if execution != 'compose':
        raise LoweringError('matrix tuning selects among composed-planner matrix bodies; use execution="compose"')
    if retain_all or policy != 'auto' or compose_selection != 'auto' or compact_codegen != 'off':
        raise LoweringError('matrix tuning applies to the normal optimized lowering only')
    if not torch.cuda.is_available():
        raise LoweringError('matrix tuning measures complete updates on the target GPU; run it on the GPU worker')
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    started = time.perf_counter()
    rng_before = _rng_snapshot()
    work = directory.parent / f'.{directory.name}.tuning'
    work.mkdir(parents=True, exist_ok=False)
    common = dict(cache_dir=cache_dir, arch=arch, retain_all=retain_all, policy=policy,
                  precision_policy=precision_policy, compact_codegen=compact_codegen, execution=execution,
                  compose_selection=compose_selection, gradient_outputs=gradient_outputs,
                  fast_build=fast_build, recurrent_gradients=recurrent_gradients)
    clock, builds = {}, {}

    def build(label, choices):
        begin = time.perf_counter()
        log = []
        receipt = api.prepare_module(captured, work / label,
                                     matrix_choices_scope=(choices, log, tuning.baseline_policy), **common)
        return label, receipt, time.perf_counter() - begin, log

    fingerprint = environment_fingerprint(arch)
    impl = implementation_version()
    structure = graph_signature(captured.ir)
    key, parts = tuning_key(fingerprint, impl, execution, precision_policy, arch, structure)
    # Step 13: the search strategy and default policy decide what a cached verdict means.
    parts = dict(parts, strategy=tuning.strategy, baseline_policy=tuning.baseline_policy,
                 gradient_outputs=gradient_outputs, fast_build=bool(fast_build),
                 recurrent_gradients=recurrent_gradients,
                 search_bounds=[tuning.max_candidates, tuning.max_builds, tuning.top_groups, tuning.alternatives,
                                tuning.numerics, tuning.min_gain])
    key = _sha(parts)
    record = {'version': TUNING_VERSION, 'bounds': asdict(tuning), 'key': key, 'parts': parts}
    try:
        entry, cache_state = load_entry(tuning.cache, key, parts)
        final = None
        if entry is not None:
            # Reuse: one build with the cached choices, no measurement. The lowering re-runs the existing
            # structural/resource admission for every cached body; any refusal rejects the entry.
            label = 'cached' if entry['choices'] else 'existing'
            label, receipt, seconds, log = build(label, entry['choices'])
            builds[label] = (receipt, seconds, log)
            clock['build_seconds'] = seconds
            problem = validate_choices(entry, log)
            if problem is None and any(r.get('rejected') for r in log):
                problem = 'rejected: cached body refused by the structural/resource check'
            if problem is None:
                final = label
                record.update(products=log, verdict=entry['verdict'], choices=entry['choices'], selected=label,
                              reused_from={k: entry.get(k) for k in ('created_utc', 'existing_ms', 'selected_ms',
                                                                     'verified_gain')})
            else:
                cache_state = problem
                if label != 'existing':
                    shutil.rmtree(work / label, ignore_errors=True)
                    builds.pop(label)
        record['cache'] = cache_state
        if final is None:
            final = _search(build, builds, work, tuning, record, clock, started, key, parts)
        # Selected artifact: a copy of the chosen variant directory (bindings are the untouched capture snapshot).
        shutil.copytree(work / final, directory)
        rng_after = _rng_snapshot()
        record['rng_unchanged'] = _rng_equal(rng_before, rng_after)
        if not record['rng_unchanged']:
            raise RuntimeError('matrix tuning changed host/device random state')
        clock['total_seconds'] = time.perf_counter() - started
        record['clock'] = clock
        receipt = json.loads((directory / 'artifact.json').read_text())
        receipt['matrix_tuning'] = record
        (directory / 'artifact.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + '\n')
        return receipt
    finally:
        if not tuning.keep_variants:
            shutil.rmtree(work, ignore_errors=True)


def _quick(work, label, tuning, reference):
    """Quick complete-update timing of one built variant on a private copy of the bindings (+ sanity screen)."""
    import torch
    from . import api
    update = api.ModuleUpdate.load(work / label)
    first = _first_update(update)
    if reference is None:
        ok, worst = True, 0.0
    else:
        ok, worst = _agrees(first, reference)
    for _ in range(3):
        update.step()
    times = _time(update, tuning.quick_samples, tuning.quick_updates)
    del update
    torch.cuda.empty_cache()
    return first, {'median_ms': times[len(times) // 2], 'samples_ms': times, 'sanity_passed': ok,
                   'sanity_worst_ratio': worst}


def _search(build, builds, work, tuning, record, clock, started, key, parts):
    """Bounded search: existing selection + derived variants (uniform classes or mixed per-group rounds), quick
    complete-update timing, verification against the existing selection and a cache write."""
    from .matrix_selection import MixedSearch, describe, groups
    if 'existing' not in builds:
        label, receipt, seconds, log = build('existing', {})
        builds[label] = (receipt, seconds, log)
    products = builds['existing'][2]
    clock['existing_build_seconds'] = builds['existing'][1]
    record['products'] = products
    record['groups'] = {gid: {k: v for k, v in g.items() if k != 'keys'} for gid, g in groups(products).items()}
    measured, failures, maps = {}, {}, {'existing': {}}
    begin = time.perf_counter()
    reference, measured['existing'] = _quick(work, 'existing', tuning, None)
    quick_seconds = time.perf_counter() - begin
    if tuning.strategy == 'uniform':
        options = variants(products, max_candidates=tuning.max_candidates, numerics=tuning.numerics)
        pending_rounds = [[(l, c) for l, c in options if l != 'existing']]
        search = None
    else:
        search = MixedSearch(products, top_groups=tuning.top_groups, alternatives=tuning.alternatives,
                             max_builds=tuning.max_builds)
        pending_rounds = None
    rounds, build_seconds = [], 0.0
    while True:
        if search is None:
            if not pending_rounds:
                break
            batch = pending_rounds.pop(0)
        else:
            times = {l: (m.get('median_ms') if m.get('sanity_passed') else None) for l, m in measured.items()}
            times.update({l: None for l in failures})
            batch = search.next_round(times)
            if search.round > 3:
                break
            if tuning.numerics == 'bitwise':
                batch = [(l, c) for l, c in batch if _bitwise_ok(products, c)]
        for label, choices in batch:
            maps[label] = choices
        todo = [(l, c) for l, c in batch if l not in builds]
        rounds.append([l for l, _ in batch])
        if not todo:
            continue
        if time.perf_counter() - started > tuning.max_seconds:
            for label, _ in todo:
                measured[label] = {'skipped': 'tuning time bound reached'}
            break
        b0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=tuning.build_workers) as pool:
            futures = [pool.submit(build, l, c) for l, c in todo]
            for (l, _), future in zip(todo, futures):
                try:
                    label, receipt, seconds, log = future.result()
                    builds[label] = (receipt, seconds, log)
                    if any(r.get('rejected') for r in log):
                        failures[l] = 'a derived choice was refused during lowering'
                except Exception as error:      # a refused/failed variant is excluded and recorded
                    failures[l] = repr(error)[:2000]
        build_seconds += time.perf_counter() - b0
        b0 = time.perf_counter()
        for label, _ in todo:
            if label in failures or label not in builds:
                continue
            _, measured[label] = _quick(work, label, tuning, reference)
        quick_seconds += time.perf_counter() - b0
    clock['variant_build_seconds'] = {l: b[1] for l, b in builds.items()}
    clock['builds_wall_seconds'] = build_seconds
    clock['quick_measure_seconds'] = quick_seconds
    record['build_failures'] = failures
    record['rounds'] = rounds
    if search is not None and getattr(search, 'stopped', None):
        record['search_stopped'] = search.stopped
    record['variants'] = [{'label': l, 'choices': c, 'groups': describe(products, c)} for l, c in maps.items()]
    record['quick'] = measured
    eligible = {l: m['median_ms'] for l, m in measured.items() if 'median_ms' in m and m['sanity_passed']}
    best = min(eligible, key=eligible.get)
    verdict, final, verification = 'existing', 'existing', None
    if best != 'existing':
        begin = time.perf_counter()
        verification = _verify(work, best, tuning)
        clock['verify_seconds'] = time.perf_counter() - begin
        if verification['accepted']:
            verdict, final = 'tuned', best
    choices = maps[final]
    record.update(verification=verification, verdict=verdict, selected=final, choices=choices)
    existing_ms = (verification or {}).get('existing_ms', measured['existing']['median_ms'])
    store_entry(tuning.cache, {
        'schema': TUNING_VERSION, 'key': key, 'parts': parts, 'verdict': verdict, 'choices': choices,
        'selected_label': final, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'existing_ms': existing_ms,
        'selected_ms': verification['candidate_ms'] if verdict == 'tuned' else existing_ms,
        'verified_gain': (verification or {}).get('gain'), 'quick': measured,
        'products': [{k: r.get(k) for k in ('key', 'native', 'default', 'admitted', 'selected')} for r in products]})
    return final


def _bitwise_ok(products, choices):
    """``numerics='bitwise'``: every chosen body shares the default selection's arithmetic class."""
    rows = {r['key']: r for r in products}
    return all(_class(rows[k], b) == _class(rows[k], rows[k]['default']) for k, b in choices.items())


def _verify(work, label, tuning):
    """Interleaved complete-update rounds: existing selection vs candidate, both on private scratch state."""
    import torch
    from . import api
    a, b = api.ModuleUpdate.load(work / 'existing'), api.ModuleUpdate.load(work / label)
    for update in (a, b):
        for _ in range(5):
            update.step()
    torch.cuda.synchronize()
    rounds = []
    for index in range(tuning.verify_rounds):
        order = (a, b) if index % 2 == 0 else (b, a)
        result = {}
        for update in order:
            update.step()
            values = _time(update, 1, tuning.verify_updates)
            result['existing' if update is a else 'candidate'] = values[0]
        rounds.append(result)
    ratios = sorted(r['candidate'] / r['existing'] for r in rounds)
    median = ratios[len(ratios) // 2]
    wins = sum(r < 1.0 for r in ratios)
    needed = math.ceil(0.7 * len(ratios))
    existing = sorted(r['existing'] for r in rounds)[len(rounds) // 2]
    candidate = sorted(r['candidate'] for r in rounds)[len(rounds) // 2]
    del a, b
    torch.cuda.empty_cache()
    return {'candidate': label, 'rounds': rounds, 'median_ratio': median, 'wins': wins, 'wins_needed': needed,
            'existing_ms': existing, 'candidate_ms': candidate, 'gain': 1 - median,
            'accepted': median <= 1 - tuning.min_gain and wins >= needed,
            'rule': f'adopt only if median candidate/existing <= {1 - tuning.min_gain:.3f} in >= 70% of rounds'}
