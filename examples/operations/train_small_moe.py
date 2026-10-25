"""Prepare and run the small MoE + intermediate-supervision training recipe.

The 25m configuration is the pinned three-layer, width-384 recipe with routing
group 32, context 2048 and vocabulary 50257. Both phases use normal capture/build
and share parameter/optimizer bindings and an arena. The tiny option is only an
API smoke fixture and does not establish the 25m model's numerical qualification.
Each step stages synthetic changing tokens, labels, auxiliary weight and the
optimizer schedule. Snapshots bind both generated programs and the host cursor.
No data, checkpoint, research checkout, credentials or cloud service is needed.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from megakernel_lab.operation_compiler import ModuleUpdate, capture_module
from megakernel_lab.operation_compiler.indexed_reduction import policy as indexed_policy
from megakernel_lab.operation_compiler.lower import GeneratedProgram
from moe_support.model import CaptureLM, make_config, objective
from moe_support.optimizer import make_rule, schedule_vector, SCHEDULE
from moe_support.recipe import recipe, aux_weight, lr


def batch_at(step, config):
    generator = torch.Generator().manual_seed(7100 + step)
    shape = (config['routing_group'], config['seq_len'])
    return tuple(torch.randint(config['vocab_size'], shape, generator=generator) for _ in range(2))


def prepare(directory, cache_dir, size):
    if torch.cuda.is_available():
        raise RuntimeError('Prepare on a CPU-only worker with CUDA 12.8 nvcc.')
    directory.mkdir(parents=True, exist_ok=False)
    r = recipe(size)
    x, labels = batch_at(0, r['model'])
    records = {}
    for phase in ('auxiliary', 'plain'):
        model = CaptureLM(make_config(size), auxiliary=phase == 'auxiliary')
        weight = torch.tensor(0.25 if phase == 'auxiliary' else 0.0, dtype=torch.float32)
        # Routing is recomputed each update. Segmented accumulation and top-k
        # ties resolved by ascending sequence index are explicit policy choices.
        with indexed_policy('segmented_v1'):
            captured = capture_module(model, objective, (x,), (labels.reshape(-1), weight),
                                      optimizer=make_rule(model))
            receipt = captured.build(directory / phase, cache_dir=cache_dir / phase,
                                     arch='sm_90', execution='compose')
        records[phase] = {key: receipt.get(key) for key in
                          ('source_hash', 'execution', 'execution_selection', 'prepare_seconds', 'build_seconds')}
        # Release the large first capture before constructing the other phase;
        # compilation stays sequential within this one CPU allocation.
        del captured, model, receipt
        gc.collect()
    config = {'schema': 'latch-small-moe-example-v1', 'size': size, 'recipe': r, 'programs': records,
              'scope': 'Synthetic API demonstration; independent numerical and performance evidence is separate.'}
    (directory / 'example-config.json').write_text(json.dumps(config, indent=2) + '\n')
    return config


def persistent_bindings(update):
    manifest = update.runtime.manifest
    return {(role['role'], role['parameter']): update.runtime.args[index]
            for index, name in enumerate(manifest['inputs'])
            if (role := manifest['input_roles'][name])['role'] in ('parameter', 'optimizer_state')}


def load_phases(directory):
    # Shared scratch is safe only for sequential launches on one stream. It does
    # not itself share persistent parameters or optimizer tensors.
    arena = ModuleUpdate.allocate_shared_arena(directory / 'auxiliary', directory / 'plain')
    auxiliary = ModuleUpdate.load(directory / 'auxiliary', arena=arena)
    shared = persistent_bindings(auxiliary)
    # Read the verified generated manifest to bind by role/name, not assumptions
    # about the two captured programs' different flat argument positions.
    manifest = GeneratedProgram.load(directory / 'plain').manifest
    bindings = {}
    for index, name in enumerate(manifest['inputs']):
        role = manifest['input_roles'][name]
        if role['role'] in ('parameter', 'optimizer_state'):
            bindings[index] = shared[(role['role'], role['parameter'])]
    if len(bindings) != len(shared):
        raise ValueError('auxiliary/plain programs disagree on persistent state')
    plain = ModuleUpdate.load(directory / 'plain', bindings=bindings, arena=arena)
    right = persistent_bindings(plain)
    if set(shared) != set(right) or any(shared[k].data_ptr() != right[k].data_ptr() for k in shared):
        raise RuntimeError('phase state was not shared')
    return {'auxiliary': auxiliary, 'plain': plain}, arena


def run(directory, steps, start_step, resume, save):
    if steps < 1 or start_step < 0:
        raise ValueError('steps must be positive and start-step nonnegative')
    config = json.loads((directory / 'example-config.json').read_text())
    phases, arena = load_phases(directory)
    identities = {phase: update.program.cache_key for phase, update in phases.items()}
    adam_step = 0
    if resume:
        snapshot = torch.load(resume, map_location='cpu', weights_only=True)
        if snapshot['schema'] != 'latch-small-moe-snapshot-v1' or snapshot['programs'] != identities:
            raise ValueError('snapshot belongs to different generated phase programs')
        if snapshot['recipe'] != config['recipe'] or snapshot['next_step'] != start_step:
            raise ValueError('snapshot recipe or absolute schedule position differs')
        phases['auxiliary'].load_state_dict(snapshot['state'])
        adam_step = snapshot['adam_step']
    r = config['recipe']
    rows = []
    pointers = {key: value.data_ptr() for key, value in persistent_bindings(phases['auxiliary']).items()}
    for step in range(start_step, start_step + steps):
        weight = aux_weight(r, step)
        phase = 'auxiliary' if weight > 0 else 'plain'
        update = phases[phase]
        x, labels = batch_at(step, r['model'])
        update.stage((x,), (labels.reshape(-1), torch.tensor(weight, dtype=torch.float32)))
        update.runtime.optimizer_state[SCHEDULE].copy_(schedule_vector(lr(r, step), adam_step + 1))
        update.step()
        adam_step += 1
        loss = float(update.loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError('nonfinite training loss')
        rows.append({'step': step, 'phase': phase, 'auxiliary_weight': weight, 'learning_rate': lr(r, step),
                     'adam_step': adam_step, 'loss': loss})
    shared = persistent_bindings(phases['auxiliary'])
    if any(value.data_ptr() != pointers[key] for key, value in shared.items()):
        raise RuntimeError('persistent bindings moved during phase transition')
    if save:
        torch.save({'schema': 'latch-small-moe-snapshot-v1', 'programs': identities, 'recipe': r,
                    'next_step': start_step + steps, 'adam_step': adam_step,
                    'state': phases['auxiliary'].state_dict()}, save)
    return {'size': config['size'], 'steps': rows, 'programs': identities,
            'persistent_state_shared': True, 'arena_bytes': arena.numel() * arena.element_size(),
            'saved': None if save is None else str(save),
            'scope': 'Synthetic changing-batch phase-transition demonstration, not a learning or speed measurement.'}


def compare(left, right):
    from train_modified_attention import compare as compare_snapshots
    return compare_snapshots(left, right)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cpu = sub.add_parser('prepare')
    cpu.add_argument('directory', type=Path)
    cpu.add_argument('--cache-dir', type=Path, required=True)
    cpu.add_argument('--size', choices=('25m', 'tiny'), default='25m')
    gpu = sub.add_parser('run')
    gpu.add_argument('directory', type=Path)
    gpu.add_argument('--steps', type=int, default=3)
    gpu.add_argument('--start-step', type=int, default=0)
    gpu.add_argument('--resume', type=Path)
    gpu.add_argument('--save', type=Path)
    same = sub.add_parser('compare')
    same.add_argument('left', type=Path)
    same.add_argument('right', type=Path)
    args = vars(parser.parse_args())
    action = args.pop('action')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    print(json.dumps({'prepare': prepare, 'run': run, 'compare': compare}[action](**args), indent=2))


if __name__ == '__main__':
    main()
