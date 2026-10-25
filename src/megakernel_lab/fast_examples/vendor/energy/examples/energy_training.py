"""Runnable energy-model training through the common operation compiler (C-OPS4-B).

The recipe is ordinary PyTorch (examples/energy.py): Langevin sampling with input gradients and a replay
buffer, a stop-gradient contrastive loss, global-norm clipping and AdamW. ``prepare`` captures and compiles it
on a CPU worker (no GPU); ``run`` loads the prebuilt single training kernel on an H100 and trains on
Fashion-MNIST, comparing the first update with the eager PyTorch control from the same state.

From research/latch_ops on owned cloud workers sharing /out:
    python -m examples.energy_training prepare /out/energy-example --config wb [--rows]
    python -m examples.energy_training run /out/energy-example --updates 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch


def prepare(directory, *, config='wb', rows=False, order_rows=2048):
    """CPU: capture the three regions, lower them into one cooperative kernel, build it with nvcc."""
    from megakernel_lab.fast_examples.vendor.energy import cops4b
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler.runtime import build
    torch.set_num_threads(1)
    cfg = cops4b.config(config)
    captured = cops4b.capture(cfg, order_rows)
    options = {'sampled_options': {'row_tasks': 'on'}} if rows else {}
    program = lower(captured.ir, **options)
    directory = Path(directory)
    program.write(directory)
    (directory / 'graph.json').write_text(json.dumps(captured.ir.to_dict(), sort_keys=True) + '\n')
    library = build(program, cache_dir=directory / 'cache')
    cops4b.save(directory / 'abi.json', {'config': cfg, 'order_rows': order_rows, 'args': cops4b.abi(captured),
                                         'lower_options': options,
                                         'parameter_names': list(captured.parameter_names),
                                         'optimizer_state_names': list(captured.optimizer_state_names),
                                         'state_names': list(captured.state_names),
                                         'carry_names': list(captured.carry_names)})
    return {'library': str(library), 'source_hash': program.cache_key, 'row_tasks': program.manifest['row_tasks']}


def run(directory, *, updates=200):
    """H100: load (no compilation), check one update against eager PyTorch, then train."""
    from megakernel_lab.fast_examples.vendor.energy import cops4b
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b import checks
    from megakernel_lab.fast_examples.vendor.energy.c_ops4_b.controls import Control
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    from megakernel_lab.operation_compiler.runtime import Runtime, build
    directory = Path(directory)
    meta = json.loads((directory / 'abi.json').read_text())
    cfg = meta['config']
    program = GeneratedProgram.load(directory)
    library = build(program, cache_dir=directory / 'cache', allow_compile=False)
    images = torch.from_numpy(cops4b.images_fp32(cfg)).cuda()
    params, opt, state, inputs, count = cops4b.bindings(cfg, meta['order_rows'], device='cuda', images=images)
    runtime = Runtime(program, cops4b.flat_args(params, opt, state, inputs, count), library=library)
    n, no = len(params), len(opt)
    before = [t.clone() for t in runtime.args[:n + no]]
    runtime.step()
    torch.cuda.synchronize()
    names = meta['parameter_names']
    loss = float(runtime.loss)
    # Same-state eager control: the native's own negatives, eager autograd gradients and optimizer.
    xneg = runtime.values[program.manifest['structured_outputs']['carry']['x']]
    xpos = images[inputs[1][0]]
    p0 = dict(zip(names, before[:n]))
    ctrl = Control(cfg, 'eager')
    control_loss, _, grads = ctrl._param_grads(p0, xpos, xneg)
    report = {'loss_native': loss, 'loss_eager_same_negatives': float(control_loss),
              'grad_relative_difference': checks.rel(torch.cat([runtime.grads[k].flatten() for k in names]),
                                                     torch.cat([grads[k].flatten() for k in names]))}
    losses, started = [loss], time.perf_counter()
    for _ in range(updates - 1):
        runtime.step()
        losses.append(runtime.loss.clone())
    torch.cuda.synchronize()
    report.update(updates=updates, seconds=time.perf_counter() - started,
                  loss_first=loss, loss_last=float(losses[-1]),
                  replay_pointer=int(runtime.state['ptr']), update_counter=int(runtime.state['t']))
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('directory')
    p.add_argument('--config', default='wb')
    p.add_argument('--rows', action='store_true', help='row-task sampling loop schedule')
    r = sub.add_parser('run')
    r.add_argument('directory')
    r.add_argument('--updates', type=int, default=200)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.directory, config=args.config, rows=args.rows), indent=2, default=str))
    else:
        run(args.directory, updates=args.updates)


if __name__ == '__main__':
    main()
