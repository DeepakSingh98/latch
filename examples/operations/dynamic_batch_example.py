"""Shared CLI for the two standalone bounded-batch module examples.

Build on a CPU worker; run/resume on a GPU worker with the recorded artifact
and native-cache paths. GPU byte comparisons demonstrate restart identity and
do not replace an independent numerical training oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from megakernel_lab.operation_compiler import DynamicBatch, ModuleUpdate, SGDConfig, capture_module


BATCHES = (1, 2, 7, 16, 31, 32, 3, 11, 23)
CHECKPOINT_AFTER = 4


def _fixture(batch, step, inputs, outputs, device):
    generator = torch.Generator(device=device).manual_seed(19000 + 37 * step + batch)
    x = .2 * torch.randn(batch, inputs, generator=generator, device=device)
    y = .1 * torch.randn(batch, outputs, generator=generator, device=device)
    return (x,), (y,)


def _capture_graph(update, fixture):
    """Warm up, capture exactly once, and restore all training state in place."""
    update.stage(*fixture)
    initial = update.state_dict()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        update.step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    update.load_state_dict(initial)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update.step()
    torch.cuda.synchronize()
    # Warmup/capture must not contribute an update to the demonstrated sequence.
    update.load_state_dict(initial)
    return graph


def _device_frame(update):
    result = {'loss': update.loss.detach().clone(),
              'active_batch': update.active_batch_tensor.detach().clone()}
    for group in ('grads', 'parameters', 'optimizer_state', 'state'):
        for name, value in getattr(update, group).items():
            result[f'{group}:{name}'] = value.detach().clone()
    return result


def _assert_same_bits(observed, expected, step):
    if set(observed) != set(expected):
        raise AssertionError(f'suffix step {step}: tensor names changed')
    for name, actual in observed.items():
        wanted = expected[name]
        if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
            raise AssertionError(f'suffix step {step}: {name} shape/dtype changed')
        # Compare raw bytes on the GPU, including signed zero and NaN payloads.
        a = actual.contiguous().reshape(-1).view(torch.uint8)
        b = wanted.contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            raise AssertionError(f'suffix step {step}: {name} changed bits after restore')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def _load_update(artifact, inputs, outputs):
    if not torch.cuda.is_available():
        raise RuntimeError('run/resume require a CUDA worker; use build on the CPU worker')
    update = ModuleUpdate.load(artifact)
    dynamic = update.manifest.get('dynamic_batch', {})
    positions = update.metadata['input_indices']
    if (dynamic.get('maximum') != 32 or len(positions['inputs']) != 1
            or len(positions['targets']) != 1
            or tuple(update.args[positions['inputs'][0]].shape) != (32, inputs)
            or tuple(update.args[positions['targets'][0]].shape) != (32, outputs)):
        raise ValueError('artifact does not match this example and its capacity-32 data contract')
    return update


def _run(name, artifact, output, inputs, outputs, *, resume):
    update = _load_update(artifact, inputs, outputs)
    device = update.runtime.device
    fixtures = [_fixture(batch, step, inputs, outputs, device)
                for step, batch in enumerate(BATCHES)]
    graph = _capture_graph(update, fixtures[0])
    pointers = tuple(value.data_ptr() for value in update.args)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path, suffix_path = output / 'checkpoint.pt', output / 'suffix.pt'
    losses = []

    if resume:
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        if (checkpoint.get('version') != 1 or checkpoint.get('example') != name
                or checkpoint.get('batches') != list(BATCHES)
                or checkpoint.get('checkpoint_after') != CHECKPOINT_AFTER
                or checkpoint.get('torch') != str(torch.__version__)):
            raise ValueError('checkpoint example, data sequence, or PyTorch version differs')
        update.load_state_dict(checkpoint['state'])
        saved = torch.load(suffix_path, weights_only=True, map_location='cpu')
        expected = [{key: value.to(device) for key, value in frame.items()} for frame in saved]
    else:
        expected = []
        checkpoint = None
        for step, fixture in enumerate(fixtures):
            update.stage(*fixture)
            graph.replay()
            loss = float(update.loss.item())
            if not bool(torch.isfinite(update.loss).item()):
                raise AssertionError(f'nonfinite loss at sequence step {step}')
            losses.append(loss)
            if step + 1 == CHECKPOINT_AFTER:
                checkpoint = {'version': 1, 'example': name, 'batches': list(BATCHES),
                              'checkpoint_after': CHECKPOINT_AFTER, 'torch': str(torch.__version__),
                              'state': update.state_dict()}
                torch.save(checkpoint, checkpoint_path)
            if step >= CHECKPOINT_AFTER:
                expected.append(_device_frame(update))
        torch.save([{key: value.cpu() for key, value in frame.items()} for frame in expected], suffix_path)
        # Restore into the same bindings, then replay the very same graph object.
        update.load_state_dict(checkpoint['state'])

    if len(expected) != len(BATCHES) - CHECKPOINT_AFTER:
        raise ValueError('saved suffix length differs from the demonstrated sequence')
    for suffix_index, step in enumerate(range(CHECKPOINT_AFTER, len(BATCHES))):
        update.stage(*fixtures[step])
        graph.replay()
        _assert_same_bits(_device_frame(update), expected[suffix_index], step)
    if pointers != tuple(value.data_ptr() for value in update.args):
        raise AssertionError('input, parameter, optimizer, RNG or batch-scalar binding changed address')
    torch.cuda.synchronize()
    receipt = {
        'passed': True, 'example': name, 'action': 'resume' if resume else 'run',
        'source_hash': update.program.cache_key, 'artifact': str(artifact.resolve()),
        'batches': list(BATCHES), 'checkpoint_after': CHECKPOINT_AFTER,
        'suffix_batches': list(BATCHES[CHECKPOINT_AFTER:]), 'suffix_bitwise_identical': True,
        'comparison': 'GPU equality of raw bytes for loss, gradients, parameters, optimizer, model/RNG state and active batch',
        'graph_captures_this_process': 1, 'stable_binding_pointers': True,
        'same_graph_after_inplace_restore': not resume,
        'loaded_checkpoint_in_new_runtime': resume,
        'checkpoint_sha256': _sha256(checkpoint_path), 'suffix_sha256': _sha256(suffix_path),
        'losses_first_pass': losses,
        'scope': 'synthetic API/restart demonstration; no independent numerical oracle or speed claim'}
    (output / ('resume.json' if resume else 'run.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main(name, model_factory, loss_fn, inputs, outputs, *, state_factory=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    cpu = sub.add_parser('build', help='capture and compile on a CPU build worker')
    cpu.add_argument('--artifact', type=Path, required=True)
    cpu.add_argument('--cache-dir', type=Path, required=True)
    cpu.add_argument('--arch', default='sm_90')
    for command in ('run', 'resume'):
        gpu = sub.add_parser(command, help='execute on a GPU worker; resume verifies a saved suffix')
        gpu.add_argument('--artifact', type=Path, required=True)
        gpu.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == 'build':
        torch.manual_seed(1907)
        state = state_factory() if state_factory is not None else None
        captured = capture_module(model_factory(), loss_fn,
                                  (torch.zeros(32, inputs),), (torch.zeros(32, outputs),),
                                  optimizer=SGDConfig(lr=.01, momentum=.9), state=state,
                                  dynamic_batch=DynamicBatch(maximum=32, axis=0))
        result = captured.build(args.artifact, cache_dir=args.cache_dir, arch=args.arch,
                                execution='compose')
    else:
        result = _run(name, args.artifact, args.output, inputs, outputs, resume=args.action == 'resume')
    print(json.dumps(result, indent=2))
