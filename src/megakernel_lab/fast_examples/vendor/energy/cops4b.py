"""C-OPS4-B jobs: energy-model training through the common operation compiler.

CPU worker (mode cops4b_cpu, no GPU): data staging, capture, lowering, nvcc build, CPU recipe checks.
GPU worker (mode cops4b_gpu, one H100): native correctness gates, real-data training, complete-update timing.
Gates and thresholds are frozen in c_ops4_b/POLICY.json before any GPU measurement of the native update.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import struct
import time
import traceback
from pathlib import Path

import numpy as np

FMNIST = {'train-images-idx3-ubyte.gz': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
          'train-labels-idx1-ubyte.gz': '25c81989df183df01b3e8a0aad5dffbe',
          't10k-images-idx3-ubyte.gz': 'bef4ecab320f06d8554ea6380940ec79',
          't10k-labels-idx1-ubyte.gz': 'bb300cfdad3c16e7a12a480ee83cd310'}
MIRRORS = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
           'https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/']
DATA = Path(os.environ.get('LATCH_FAST_ROOT', './latch-fast')).resolve() / 'data' / 'fmnist'
N_TRAIN = 54000


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_jsonable) + '\n')


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ------------------------------------------------------------------------------------------------- data
def stage_fmnist(root=DATA):
    """S100L data staging (src/s100l_ebm/data.py): original IDX files, md5 as published by torchvision."""
    import urllib.request
    root.mkdir(parents=True, exist_ok=True)
    record = {}
    for name, md5 in FMNIST.items():
        path = root / name
        if not path.exists():
            error = None
            for mirror in MIRRORS:
                try:
                    tmp = path.with_suffix('.part')
                    urllib.request.urlretrieve(mirror + name, tmp)
                    if hashlib.md5(tmp.read_bytes()).hexdigest() == md5:
                        tmp.replace(path)
                        break
                    error = 'md5 mismatch from ' + mirror
                except Exception as exc:
                    error = repr(exc)
            if not path.exists():
                raise RuntimeError(f'could not stage {name}: {error}')
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != md5:
            raise RuntimeError('md5 mismatch: ' + name)
        record[name] = dict(md5=md5, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    return record


def read_idx(path):
    raw = gzip.decompress(Path(path).read_bytes())
    magic, = struct.unpack('>I', raw[:4])
    nd = magic & 0xFF
    shape = struct.unpack('>' + 'I' * nd, raw[4:4 + 4 * nd])
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(shape)


def split(data_seed):
    perm = np.random.Generator(np.random.PCG64(data_seed)).permutation(60000)
    return perm[:N_TRAIN], perm[N_TRAIN:]


def order_table(data_seed, B, updates):
    """S100L `s100l-fmnist-v1` batch order: [updates, B] indices into the 60,000 training images."""
    train, _ = split(data_seed)
    per = N_TRAIN // B
    out = np.empty((updates, B), np.int64)
    perm = None
    for t in range(updates):
        e, s = divmod(t, per)
        if s == 0:
            perm = np.random.Generator(np.random.PCG64(data_seed + 1 + e)).permutation(N_TRAIN)
        out[t] = train[perm[s * B:(s + 1) * B]]
    return out


def images_fp32(cfg, which='train'):
    """FP32 crop of every image, N[p] = fl32(p / 255) (the S100L LUT values)."""
    stage_fmnist()
    raw = read_idx(DATA / ('train-images-idx3-ubyte.gz' if which == 'train' else 't10k-images-idx3-ubyte.gz'))
    S = cfg['crop']
    o = (28 - S) // 2
    crop = raw[:, o:o + S, o:o + S].reshape(raw.shape[0], S * S)
    lut = (np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)
    return lut[crop]


# ------------------------------------------------------------------------------------------------- program
def config(name, overrides=None):
    from megakernel_lab.fast_examples.vendor.energy.examples import energy
    cfg = json.loads(json.dumps(energy.CONFIGS[name]))
    for key, value in (overrides or {}).items():
        if isinstance(value, dict):
            cfg[key] = {**cfg.get(key, {}), **value}
        else:
            cfg[key] = value
    return cfg


def bindings(cfg, rows, *, device='cpu', images=None):
    """(params, optimizer_state, state, inputs, count) at s100l-init-v1 and the initial replay buffer."""
    import torch
    from megakernel_lab.fast_examples.vendor.energy.examples import energy
    from megakernel_lab.operation_compiler.optim import initialize_optimizer_state
    params = energy.init_parameters(cfg)
    opt = initialize_optimizer_state(params, energy.optimizer_config(cfg))
    state = energy.initial_state(cfg)
    if images is None:
        images = torch.zeros((60000, cfg['crop'] ** 2), dtype=torch.float32)
    order = torch.from_numpy(order_table(cfg['data_seed'], cfg['batch'], rows))
    count = torch.tensor(cfg['chain'], dtype=torch.int64)
    move = lambda d: {k: v.to(device) for k, v in d.items()}
    return move(params), move(opt), move(state), (images.to(device), order.to(device)), count.to(device)


def capture(cfg, rows, images=None):
    from megakernel_lab.fast_examples.vendor.energy.examples import energy
    from megakernel_lab.operation_compiler.sampling import capture_sampled_training
    params, opt, state, inputs, count = bindings(cfg, rows, images=images)
    init, step, loss = energy.make_functions(cfg)
    return capture_sampled_training(init, step, loss, params, inputs, count, energy.optimizer_config(cfg),
                                    max_steps=cfg['chain'], state=state, optimizer_state=opt)


def abi(captured):
    return [{'shape': list(t.shape), 'dtype': str(t.dtype).removeprefix('torch.')} for t in captured.example_args]


def flat_args(params, opt, state, inputs, count):
    return (*params.values(), *opt.values(), *state.values(), *inputs, count)


# ------------------------------------------------------------------------------------------------- CPU




def cpu_example(payload, arg, out):
    from megakernel_lab.fast_examples.vendor.energy.examples.energy_training import prepare
    name, _, rows = arg.partition('@')
    result = prepare(out / 'native', config=name, rows=rows == 'rows')
    result['passed'] = True
    return result


def cpu_data(payload, arg, out):
    record = stage_fmnist()
    return {'files': record, 'passed': True}


def cpu_rng(payload, arg, out):
    """RNG contract on CPU: Random123 KAT, compiler field words vs the independent NumPy Philox, transforms."""
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import reference as R
    from megakernel_lab.operation_compiler.random import make_key, philox_field
    rows = []
    ok = R.kat_ok()
    for seed, c2, c3, shape in ((20260926, 0, 0, (8, 12)), (20260926, 0xFFFFFF01, 7, (5, 4)),
                                (2 ** 64 - 1, 63, 2 ** 32 - 1, (3, 9)), (123456789012, 5, 1, (17, 784))):
        key = make_key(seed)
        args = (key, torch.tensor(c2), torch.tensor(c3), list(shape))
        words = philox_field(*args, 'bits').numpy()
        ref = R.field_words(seed, c2, c3, *shape).astype(np.int64)
        uni = philox_field(*args, 'uniform').numpy()
        nor = philox_field(*args, 'normal').numpy()
        ref_n = R.normal64(seed, c2, c3, *shape)
        rows.append({'seed': seed, 'c2': c2, 'c3': c3, 'shape': shape,
                     'words_exact': bool(np.array_equal(words, ref)),
                     'uniform_exact': bool(np.array_equal(uni, R.uniform(ref).astype(np.float32))),
                     'normal_max_abs_diff_vs_fp64': float(np.abs(nor.astype(np.float64) - ref_n).max())})
    passed = ok and all(r['words_exact'] and r['uniform_exact'] and r['normal_max_abs_diff_vs_fp64'] < 1e-5
                        for r in rows)
    result = {'kat': ok, 'cases': rows, 'passed': passed}
    save(out / 'rng.json', result)
    return result


def cpu_capture(payload, arg, out):
    """Capture, lower and build one configuration; persist the program, IR and ABI."""
    import torch
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.runtime import build
    name, _, variant = arg.partition('@')
    cfg = config(name, payload.get('overrides', {}).get(name))
    rows = payload.get('order_rows', 2048)
    options = payload.get('lower_options', {}).get(variant or 'default', {})
    started = time.perf_counter()
    captured = capture(cfg, rows)
    capture_seconds = time.perf_counter() - started
    started = time.perf_counter()
    program = lower(captured.ir, **options)
    lower_seconds = time.perf_counter() - started
    native = out / 'native'
    program.write(native)
    (native / 'graph.json').write_text(json.dumps(captured.ir.to_dict(), sort_keys=True) + '\n')
    save(native / 'abi.json', {'config': cfg, 'order_rows': rows, 'args': abi(captured), 'lower_options': options,
                               'state_names': list(captured.state_names),
                               'parameter_names': list(captured.parameter_names),
                               'optimizer_state_names': list(captured.optimizer_state_names),
                               'carry_names': list(captured.carry_names)})
    started = time.perf_counter()
    library = build(program, cache_dir=payload['cache_dir'])
    build_seconds = time.perf_counter() - started
    summary = {'config': cfg, 'capture_seconds': capture_seconds, 'lower_seconds': lower_seconds,
               'build_seconds': build_seconds, 'library': str(library),
               'library_sha256': hashlib.sha256(Path(library).read_bytes()).hexdigest(),
               'source_hash': program.cache_key, 'arena_bytes': program.manifest['arena_bytes'],
               'shared_bytes': program.manifest['shared_bytes'],
               'region_operations': {k: len(g.operations) for k, g in captured.ir.regions.items()},
               'region_targets': {k: sorted({op.target for op in g.operations}) for k, g in captured.ir.regions.items()},
               'region_schedules': {k: v.get('compiler_selection', {}).get('schedule')
                                    for k, v in program.manifest['regions'].items()},
               'cast_reuse': captured.ir.metadata.get('cast_reuse'),
               'body_state_changes': captured.ir.metadata['body_state_changes'],
               'row_tasks': program.manifest.get('row_tasks'), 'passed': True}
    if payload.get('graph_check', {}).get(name):
        summary['graph_check'] = graph_check(captured, cfg, rows)
        summary['passed'] = summary['graph_check']['passed']
    save(out / 'summary.json', summary)
    return summary


def graph_check(captured, cfg, rows):
    """CPU: captured region graphs (native order) versus the eager user functions, real data, two updates."""
    import torch
    images = torch.from_numpy(images_fp32(cfg))
    params, opt, state, inputs, count = bindings(cfg, rows, images=images)
    args = list(flat_args(params, opt, state, inputs, count))
    n, no, ns = len(params), len(opt), len(state)
    report = []
    for update in range(2):
        graph = captured.graph_step(*args)
        eager = captured.step_fn(*args)
        diffs = [float((a.double() - b.double()).abs().max()) if a.numel() else 0.0 for a, b in zip(graph, eager)]
        report.append({'update': update, 'max_abs_diff': max(diffs), 'loss_graph': float(graph[0]),
                       'loss_eager': float(eager[0]), 'bitwise': all(torch.equal(a, b) for a, b in zip(graph, eager))})
        new = captured.unpack(graph)
        args[:n] = [new['params'][k] for k in captured.parameter_names]
        args[n:n + no] = [new['optimizer_state'][k] for k in captured.optimizer_state_names]
        args[n + no:n + no + ns] = [new['state'][k] for k in captured.state_names]
    return {'updates': report, 'passed': all(r['max_abs_diff'] < 1e-4 for r in report)}


def cpu_recipe(payload, arg, out):
    """CPU: the PyTorch recipe (eager user functions) versus the independent FP64 reference, same state."""
    import torch
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import checks
    cfg = config(arg, payload.get('overrides', {}).get(arg))
    images = torch.from_numpy(images_fp32(cfg))
    result = checks.recipe_vs_reference(cfg, images, payload.get('order_rows', 2048))
    save(out / 'recipe.json', result)
    return result


# ------------------------------------------------------------------------------------------------- GPU
