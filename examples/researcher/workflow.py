"""Researcher workflow: define -> conventional run -> Latch compile -> compare -> edit -> repeat.

    prepare   CPU worker with CUDA 12.8 nvcc: capture ``model.ByteLM``/``model.objective`` for a config
              (JSON file plus ``--set key=value`` edits), lower and compile the complete update.
    describe  Read a prepared directory: what was generated (one kernel, phases, selected implementation).
    run       H100: train one arm (``latch``, ``eager`` or ``compiled``) for N changing minibatches from the
              same initialization and data order; record losses, complete-update times, memory and setup;
              optional ``--save``/``--resume`` snapshots and development evaluation on prepared text.
    check     H100: first-update loss, gradients and updated parameters of the generated program against the
              eager module as written (same casts); the compiled control is compared the same way for scale.
    compare   Bitwise comparison of two snapshots (restart check).
    report    Merge run/check receipts into one table (CPU).

Every arm reads the same ``model.py``. A prepared directory is bound to one configuration; change the program
(dimensions, exit position, recurrence) by preparing again into a new directory. The auxiliary weight and the
minibatch are runtime inputs. This is an API/workflow demonstration with small models; it makes no learning or
speed claim beyond the numbers it prints.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ByteLM, Config, adamw_hyperparameters, objective, example_inputs, parameter_count   # noqa: E402
from minibatches import source   # noqa: E402

CONFIG_FILE = 'workflow-config.json'
SNAPSHOT_SCHEMA = 'latch-researcher-workflow-snapshot-v1'
SMOKE_TOLERANCE = 2e-2


def load_config(directory):
    record = json.loads((Path(directory) / CONFIG_FILE).read_text())
    return Config.from_dict(record['config'])


def prepare(directory, cache_dir, config, set, allow_gpu, indexed_policy='segmented_v1'):
    if torch.cuda.is_available() and not allow_gpu:
        raise RuntimeError('prepare on a CPU-only build worker (CUDA_VISIBLE_DEVICES=), or pass --allow-gpu')
    cfg = (Config.load(config) if config else Config()).edited(set or [])
    from megakernel_lab.operation_compiler import AdamWConfig, capture_module
    torch.set_num_threads(1)
    model = ByteLM(cfg)
    inputs, targets = example_inputs(cfg)
    rule = AdamWConfig(**adamw_hyperparameters(cfg), max_grad_norm=cfg.clip)
    # Embedding gradients are indexed accumulations. Left to itself ('auto') the compiler picks the fastest admitted
    # engine, global float atomics for compose programs, whose FP32 addition order is not repeatable run to run.
    # This workflow defaults to the deterministic ``segmented_v1`` engine (measured at the same complete-update time
    # on these shapes, bitwise restart); pass --indexed-policy auto to take the compiler's own choice.
    from contextlib import nullcontext
    from megakernel_lab.operation_compiler.indexed_reduction import policy as select_indexed_policy
    scope = nullcontext() if indexed_policy == 'auto' else select_indexed_policy(indexed_policy)
    began = time.perf_counter()
    with scope:
        captured = capture_module(model, objective, inputs, targets, optimizer=rule)
        capture_seconds = time.perf_counter() - began
        receipt = captured.build(directory, cache_dir=cache_dir, arch='sm_90', execution='compose')
    record = {'schema': 'latch-researcher-workflow-config-v1', 'config': cfg.to_dict(),
              'parameters': parameter_count(model), 'trainable': list(captured.parameter_names),
              'capture_seconds': capture_seconds, 'indexed_policy_requested': indexed_policy}
    (Path(directory) / CONFIG_FILE).write_text(json.dumps(record, indent=2) + '\n')
    return describe(directory)


def describe(directory):
    directory = Path(directory)
    receipt = json.loads((directory / 'artifact.json').read_text())
    record = json.loads((directory / CONFIG_FILE).read_text())
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    manifest = GeneratedProgram.load(directory).manifest
    selection = receipt.get('execution_selection', {})
    capabilities = selection.get('admitted_capabilities', {}) if isinstance(selection, dict) else {}
    phases = manifest.get('phases', [])
    from megakernel_lab.operation_compiler.indexed_compose import engine_of
    from megakernel_lab.operation_compiler.indexed_reduction import UNORDERED
    engine = engine_of(manifest)
    return {'directory': str(directory), 'config': record['config'], 'parameters': record['parameters'],
            'indexed_policy_requested': record.get('indexed_policy_requested', 'auto'),
            'indexed_engine': engine,
            'bitwise_repeatable_accumulation': None if engine is None else engine not in UNORDERED,
            'trainable': record['trainable'], 'source_hash': receipt['source_hash'],
            'execution': receipt['execution'], 'selected': selection.get('selected'),
            'selection_reasons': selection.get('reasons'),
            'attention_bodies': capabilities.get('attention'),
            'phases': len(phases), 'arena_words': manifest.get('arena_words'),
            'launch': 'one cooperative CUDA kernel per complete update (forward, loss, backward, clip, AdamW)',
            'optimizer': receipt['optimizer'].get('kind'),
            'capture_seconds': record.get('capture_seconds'), 'prepare_seconds': receipt['prepare_seconds'],
            'build_seconds': receipt['build_seconds'], 'library': receipt['library'],
            'scope': 'generation receipt; execution speed and numerical qualification are measured separately'}


class LatchArm:
    arm = 'latch'

    def __init__(self, directory, cfg):
        from megakernel_lab.operation_compiler import ModuleUpdate
        began = time.perf_counter()
        self.update = ModuleUpdate.load(directory)
        torch.cuda.synchronize()
        self.setup_seconds = time.perf_counter() - began
        self.cfg = cfg
        info = describe(directory)
        self.label = (f"one cooperative kernel launch per update; execution={info['execution']}, "
                      f"selected={info['selected']}, phases={info['phases']}, indexed={info['indexed_engine']}")
        self.source_hash = info['source_hash']

    def stage(self, tokens, labels, aux_weight):
        self.update.stage((tokens,), (labels, torch.tensor(float(aux_weight), dtype=torch.float32)))

    def step(self):
        self.update.step()

    @property
    def loss(self):
        return self.update.loss

    def parameters(self):
        return self.update.parameters

    def state_dict(self):
        return self.update.state_dict()

    def load_state_dict(self, state):
        self.update.load_state_dict(state)

    @property
    def launches(self):
        return self.update.runtime.launch_count


def make_arm(directory, arm, cfg):
    if arm == 'latch':
        return LatchArm(directory, cfg)
    from conventional import ConventionalUpdate
    return ConventionalUpdate(cfg, arm)


def peak():
    return {'allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'reserved_mib': torch.cuda.max_memory_reserved() / 2**20}


def run(directory, arm, steps, start_step, data, resume, save, warmup, out, eval_batches, aux_weight=None):
    if steps < 1 or start_step < 0 or not 0 <= warmup < steps:
        raise ValueError('steps positive, start-step nonnegative, warmup below steps')
    cfg = load_config(directory)
    weight = cfg.aux_weight if aux_weight is None else float(aux_weight)   # runtime input: no new program
    if weight and cfg.exit_after is None:
        raise ValueError('this program has no intermediate exit; aux_weight has no effect (prepare with exit_after)')
    src = source(cfg, data)
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats()
    engine = make_arm(directory, arm, cfg)
    setup_peak = peak()
    if resume is not None:
        snapshot = torch.load(resume, map_location='cpu', weights_only=True)
        if snapshot.get('schema') != SNAPSHOT_SCHEMA or snapshot['arm'] != engine.arm:
            raise ValueError('snapshot schema/arm differs from this run')
        if snapshot['next_step'] != start_step:
            raise ValueError(f"snapshot continues at step {snapshot['next_step']}, not {start_step}")
        engine.load_state_dict(snapshot['state'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events, losses = [], []
    began = time.perf_counter()
    for step in range(start_step, start_step + steps):
        tokens, labels = src.batch(step)
        engine.stage(tokens.to(device), labels.to(device), weight)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        engine.step()
        end.record()
        events.append((start, end))
        losses.append(engine.loss.detach().clone())
    torch.cuda.synchronize()
    wall = time.perf_counter() - began
    ms = [s.elapsed_time(e) for s, e in events]
    losses = [float(v) for v in losses]
    if not all(math.isfinite(v) for v in losses):
        raise RuntimeError('non-finite training loss')
    timed = ms[warmup:]
    result = {'schema': 'latch-researcher-workflow-run-v1', 'directory': str(directory), 'config': cfg.to_dict(),
              'arm': engine.arm, 'execution_label': engine.label, 'data': src.identity(),
              'aux_weight_used': weight, 'aux_weight_overridden': aux_weight is not None,
              'steps': list(range(start_step, start_step + steps)), 'losses': losses,
              'update_ms': ms, 'timed_updates': len(timed), 'warmup_updates': warmup,
              'median_update_ms': statistics.median(timed), 'mean_update_ms': statistics.fmean(timed),
              'min_update_ms': min(timed), 'wall_seconds_including_staging': wall,
              'setup_seconds': engine.setup_seconds, 'setup_peak': setup_peak, 'training_peak': peak(),
              'resumed_from': None if resume is None else str(resume), 'saved': None if save is None else str(save),
              'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'scope': 'short changing-minibatch run; complete-update CUDA-event times exclude minibatch staging; '
                       'not a learning-equivalence or benchmark claim'}
    if arm == 'latch':
        result['launches'] = engine.launches
        result['one_launch_per_update'] = engine.launches == steps
        result['source_hash'] = engine.source_hash
    else:
        result['graph_error'] = getattr(engine, 'graph_error', None)
    if data:
        from conventional import evaluate
        result['development'] = evaluate(cfg, engine.parameters(), src.dev_batches(eval_batches))
    if save is not None:
        torch.save({'schema': SNAPSHOT_SCHEMA, 'arm': engine.arm, 'next_step': start_step + steps,
                    'config': cfg.to_dict(), 'state': engine.state_dict()}, save)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
    return result


def relative(native, reference):
    reference = reference.float()
    return float((native.float() - reference).norm() / reference.norm().clamp_min(1e-12))


def check(directory, data, out, aux_weight=None):
    """First update: generated program versus the module as written (eager, same casts), compiled for scale."""
    from conventional import ConventionalUpdate
    cfg = load_config(directory)
    device = torch.device('cuda')
    tokens, labels = source(cfg, data).batch(0)
    tokens, labels = tokens.to(device), labels.to(device)
    used = cfg.aux_weight if aux_weight is None else float(aux_weight)
    weight = torch.tensor(used, dtype=torch.float32, device=device)

    latch = LatchArm(directory, cfg)
    latch.stage(tokens, labels, used)
    latch.step()
    torch.cuda.synchronize()
    latch_loss = float(latch.loss)
    latch_grads = {name: value.detach().clone() for name, value in latch.update.grads.items()}
    latch_after = {name: value.detach().clone() for name, value in latch.parameters().items()}

    eager = ConventionalUpdate(cfg, 'eager')
    loss = objective(eager.model(tokens), labels, weight)
    loss.backward()
    eager_grads = {name: p.grad.detach().clone() for name, p in eager.model.named_parameters()}
    if cfg.clip is not None:
        torch.nn.utils.clip_grad_norm_(eager.model.parameters(), cfg.clip)
    eager.optimizer.step()
    eager_after = {name: p.detach().clone() for name, p in eager.model.named_parameters()}
    eager_loss = float(loss.detach())

    compiled = ConventionalUpdate(cfg, 'compiled')
    compiled.stage(tokens, labels, used)
    compiled.step()
    torch.cuda.synchronize()
    compiled_loss = float(compiled.loss)
    compiled_after = {name: p.detach().clone() for name, p in compiled.model.named_parameters()}

    if set(latch_grads) != set(eager_grads):
        raise RuntimeError('trained parameter sets differ: ' + str(set(latch_grads) ^ set(eager_grads)))
    rows = {'loss': {'latch': latch_loss, 'eager': eager_loss, 'compiled': compiled_loss,
                     'latch_vs_eager_rel': abs(latch_loss - eager_loss) / max(abs(eager_loss), 1e-12),
                     'compiled_vs_eager_rel': abs(compiled_loss - eager_loss) / max(abs(eager_loss), 1e-12)},
            'gradients_latch_vs_eager': {name: relative(latch_grads[name], eager_grads[name]) for name in eager_grads},
            'parameters_after_update_latch_vs_eager': {name: relative(latch_after[name], eager_after[name]) for name in eager_after},
            'parameters_after_update_compiled_vs_eager': {name: relative(compiled_after[name], eager_after[name]) for name in eager_after}}
    worst_latch = max([rows['loss']['latch_vs_eager_rel'], *rows['gradients_latch_vs_eager'].values(),
                       *rows['parameters_after_update_latch_vs_eager'].values()])
    worst_compiled = max([rows['loss']['compiled_vs_eager_rel'], *rows['parameters_after_update_compiled_vs_eager'].values()])
    finite = all(math.isfinite(v) for v in (latch_loss, eager_loss, compiled_loss, worst_latch, worst_compiled))
    result = {'schema': 'latch-researcher-workflow-check-v1', 'directory': str(directory), 'config': cfg.to_dict(),
              'aux_weight_used': used, 'rows': rows, 'worst_latch_vs_eager': worst_latch, 'worst_compiled_vs_eager': worst_compiled,
              'tolerance': SMOKE_TOLERANCE, 'finite': finite,
              'passed': finite and worst_latch <= SMOKE_TOLERANCE,
              'trained_parameters': sorted(latch_grads), 'compiled_label': compiled.label,
              'scope': 'first-update smoke comparison with the eager module as written (same explicit BF16 casts); '
                       'catches a wrong objective, exit position or update rule; not independent numerical qualification'}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
    return result


def _same(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.contiguous().view(-1).view(torch.uint8), right.contiguous().view(-1).view(torch.uint8)))
    return left == right


def _differences(left, right, prefix=''):
    rows = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                rows.append({'key': prefix + str(key), 'missing': True})
            else:
                rows.extend(_differences(left[key], right[key], prefix + str(key) + '/'))
    elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not _same(left, right):
            same_shape = left.shape == right.shape and left.dtype == right.dtype
            delta = (left.float() - right.float()).abs() if same_shape else None
            rows.append({'key': prefix.rstrip('/'), 'dtype': str(left.dtype), 'shape': list(left.shape),
                         'elements_differing': int((delta != 0).sum()) if delta is not None else None,
                         'max_abs_difference': float(delta.max()) if delta is not None else None,
                         'max_abs_reference': float(right.float().abs().max()) if same_shape else None})
    elif left != right:
        rows.append({'key': prefix.rstrip('/'), 'left': str(left)[:80], 'right': str(right)[:80]})
    return rows


def compare(left, right, out=None):
    """Bitwise snapshot comparison; on a mismatch, the receipt lists every differing tensor."""
    a = torch.load(left, map_location='cpu', weights_only=True)
    b = torch.load(right, map_location='cpu', weights_only=True)
    result = {'bitwise_equal': _same(a, b), 'left': str(left), 'right': str(right), 'arm': a.get('arm'),
              'next_step': a.get('next_step')}
    if not result['bitwise_equal']:
        result['differences'] = _differences(a, b)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
    if not result['bitwise_equal']:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    return result


def report(runs, checks, out):
    rows = []
    by_config = {}
    for path in runs:
        record = json.loads(Path(path).read_text())
        # Group by prepared directory: edited variants keep the preset's config name.
        name = Path(record['directory']).name
        if record.get('aux_weight_overridden'):
            name += f"@w{record['aux_weight_used']:g}"
        by_config.setdefault(name, {})[record['arm']] = record
        development = record.get('development') or {}
        rows.append({'config': name, 'config_name': record['config']['name'], 'arm': record['arm'], 'steps': len(record['steps']),
                     'first_loss': record['losses'][0], 'last_loss': record['losses'][-1],
                     'median_update_ms': record['median_update_ms'], 'setup_seconds': record['setup_seconds'],
                     'training_peak_allocated_mib': record['training_peak']['allocated_mib'],
                     'training_peak_reserved_mib': record['training_peak']['reserved_mib'],
                     'dev_bits_per_byte': development.get('bits_per_byte'),
                     'execution_label': record['execution_label'], 'data': record['data']['kind']})
    ratios = {}
    for name, arms in by_config.items():
        if 'latch' in arms:
            fastest = min((a for a in arms if a != 'latch'), key=lambda a: arms[a]['median_update_ms'], default=None)
            agreement = {a: max(abs(x - y) for x, y in zip(arms['latch']['losses'], arms[a]['losses']))
                         for a in arms if a != 'latch' and len(arms[a]['losses']) == len(arms['latch']['losses'])}
            ratios[name] = {'fastest_conventional_arm': fastest,
                            'latch_over_fastest_conventional_time': (arms['latch']['median_update_ms'] / arms[fastest]['median_update_ms']
                                                                     if fastest else None),
                            'max_abs_loss_difference_over_run': agreement}
    checks_summary = []
    for path in checks or []:
        record = json.loads(Path(path).read_text())
        checks_summary.append({'config': Path(record['directory']).name, 'aux_weight': record.get('aux_weight_used'), 'passed': record['passed'],
                               'worst_latch_vs_eager': record['worst_latch_vs_eager'],
                               'worst_compiled_vs_eager': record['worst_compiled_vs_eager'], 'tolerance': record['tolerance']})
    lines = ['| config | arm | steps | first loss | last loss | median update ms | setup s | peak alloc MiB | dev bits/byte | execution |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        dev = '' if r['dev_bits_per_byte'] is None else f"{r['dev_bits_per_byte']:.4f}"
        lines.append(f"| {r['config']} | {r['arm']} | {r['steps']} | {r['first_loss']:.4f} | {r['last_loss']:.4f} | "
                     f"{r['median_update_ms']:.3f} | {r['setup_seconds']:.1f} | {r['training_peak_allocated_mib']:.0f} | {dev} | {r['execution_label']} |")
    result = {'schema': 'latch-researcher-workflow-report-v1', 'rows': rows, 'ratios': ratios, 'checks': checks_summary,
              'markdown': '\n'.join(lines),
              'scope': 'ratio = Latch median complete-update time / fastest conventional arm on the same GPU and recipe; '
                       'below one favors Latch; setup, memory and loss agreement are separate columns'}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2) + '\n')
        Path(out).with_suffix('.md').write_text(result['markdown'] + '\n\n' + json.dumps(ratios, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare', help='capture, lower and compile on a CPU worker with nvcc')
    p.add_argument('directory', type=Path)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, help='config JSON (examples/researcher/configs/*.json)')
    p.add_argument('--set', action='append', metavar='KEY=VALUE', help='edit a config field, e.g. --set exit_after=2 --set recurrence=3')
    p.add_argument('--allow-gpu', action='store_true')
    p.add_argument('--indexed-policy', default='segmented_v1', choices=('auto', 'segmented_v1', 'owner_v1', 'atomic_v1', 'frozen_base'),
                   help='indexed (embedding) gradient engine; default segmented_v1 is bitwise repeatable; '
                        'auto takes the compiler\'s fastest admitted engine (atomic, not repeatable)')
    p = sub.add_parser('describe', help='what was generated for a prepared directory')
    p.add_argument('directory', type=Path)
    p = sub.add_parser('run', help='train one arm on an H100')
    p.add_argument('directory', type=Path)
    p.add_argument('--arm', choices=('latch', 'eager', 'compiled'), required=True)
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--start-step', type=int, default=0)
    p.add_argument('--warmup', type=int, default=4, help='leading updates excluded from timing statistics')
    p.add_argument('--data', type=Path, help='prepared text directory (examples/families/prepare_data.py text); default synthetic bytes')
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--aux-weight', type=float, help='override the auxiliary weight at run time (no new program)')
    p.add_argument('--resume', type=Path)
    p.add_argument('--save', type=Path)
    p.add_argument('--out', type=Path)
    p = sub.add_parser('check', help='first-update comparison against the eager module (H100)')
    p.add_argument('directory', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--aux-weight', type=float)
    p.add_argument('--out', type=Path)
    p = sub.add_parser('compare', help='bitwise snapshot comparison')
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path)
    p.add_argument('--out', type=Path)
    p = sub.add_parser('report', help='merge receipts into one table')
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--checks', type=Path, nargs='*')
    p.add_argument('--out', type=Path)
    options = vars(parser.parse_args())
    action = options.pop('action')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    result = {'prepare': prepare, 'describe': describe, 'run': run, 'check': check,
              'compare': compare, 'report': report}[action](**options)
    print(json.dumps(result, indent=2, default=str))
    if action == 'check' and not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
