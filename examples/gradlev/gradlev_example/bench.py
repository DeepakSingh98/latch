# Provenance: adapted from the Latch repository, commit df184a5f2579 (branch codex/latch-gradlev),
# research/latch_gradlev/bench.py (git blob ad98f15e551bf0c78cc7ec022d40a0767f0cbfde).
# Changed: imports point at the local gradlev_example package; main() takes argv; the Latch arm
# 'latch=PATH' may name a prebuilt artifact directory (as in the donor) and 'latch' alone uses the
# artifact written by `quickstart.sh build` ($GRADLEV_ROOT/build_<name>/artifact); --blocks selects
# the first N dev blocks without loading the rest.
# Removed: Google Cloud Storage checkpoint download in load_model() (local checkpoint files in the donor format only).
# Timing, alternation, agreement and profile logic are unchanged; the printed summary also shows
# the profiled device-kernel count per block (already in the donor's bench.json).
"""Paired complete-sequence timing and agreement of the causal adaptation backends on one GPU.

python -m gradlev_example benchmark --out DIR --arms 'graph;graph_hybrid;latch'
       [--ckpt final.pt] [--blocks 16] [--rounds 5] [--split dev]

Per block (fresh state): prefill (trunk + gates) -> stage -> reset -> T adapted steps -> NLL sum
read back. 'frozen' times the no-write inference of the same block (prefill + batched head).
Arms alternate within every round on the same otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import deploy as gd
from . import model as gm
from .assets import ensure, root
from .conv import ConvLoop
from .data import eval_blocks


def load_model(ckpt, dev):
    p = gm.gpt2_params(ensure('gpt2.safetensors'))
    g = gm.gate_params()
    meta = {'model': 'gpt2-pretrained', 'gate': 'paper init mu=0.9'}
    if ckpt:
        state = torch.load(ckpt, map_location='cpu')
        p = {k: v.float() for k, v in state['params']['p'].items()}
        if 'g' in state['params']:
            g = {k: v.float() for k, v in state['params']['g'].items()}
        meta = {'model': str(ckpt), 'arm': state['args'].get('arm'), 'steps': state.get('steps')}
    return {k: v.to(dev) for k, v in p.items()}, {k: v.to(dev) for k, v in g.items()}, meta


def head_of(name):
    return 'mixed_gl32' if 'gl32' in name else 'mixed'


def default_artifact(name):
    """Artifact written by `quickstart.sh build` for this arm name (latch -> P1, latchgl32 -> P1b)."""
    return root() / f'build_{name}' / 'artifact'


def make_arm(spec, c, W1_0, W2_0, T, p=None, out=None, receipts=None):
    name, _, artifact = spec.partition('=')
    if name.startswith('latch'):
        from .latch_run import LatchLoop
        if artifact.startswith('build'):
            from .latch_loop import build_artifact
            opts = json.loads(artifact.partition(':')[2] or '{}') or None
            artifact, receipt = build_artifact({k: v.cpu() for k, v in p.items()}, Path(out) / f'build_{name}',
                                               T=T, lower_options=opts, head=head_of(name))
            receipts[name] = {k: receipt[k] for k in ('source_hash', 'build_seconds', 'lower_seconds',
                              'capture_wall_seconds', 'setup_wall_seconds', 'arena_bytes', 'shared_bytes')}
        elif not artifact:
            artifact = default_artifact(name)
            if not (Path(artifact) / 'artifact.json').exists():
                raise SystemExit(f'{artifact} is missing; run quickstart.sh build first')
            receipts[name] = {'artifact': str(artifact),
                              **{k: v for k, v in json.loads((Path(artifact) / 'artifact.json').read_text()).items()
                                 if k in ('source_hash', 'build_seconds', 'lower_seconds', 'arena_bytes',
                                          'shared_bytes')}}
        return name, LatchLoop(str(artifact), c, W1_0, W2_0, 1, T)
    return name, ConvLoop(name, c, W1_0, W2_0, 1, T)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gradlev_example benchmark')
    ap.add_argument('--out', required=True)
    ap.add_argument('--arms', default='graph;graph_hybrid;latch')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--blocks', type=int, default=16)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--profile', type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = 'cuda'
    p, g, meta = load_model(args.ckpt, dev)
    c = gd.consts_from(p, dev)
    W1_0, W2_0 = p[f'{gm.L-1}.fc_w'].contiguous(), p[f'{gm.L-1}.pr_w'].contiguous()
    inputs, targets = eval_blocks(args.split, args.blocks)
    inputs, targets = inputs[:args.blocks].to(dev), targets[:args.blocks].to(dev)
    T = inputs.shape[1]
    r = {'meta': meta, 'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'T': T,
         'blocks': args.blocks, 'rounds': args.rounds, 'arms': {}}
    try:
        torch.mm(torch.ones(1, 8, device=dev, dtype=torch.bfloat16), torch.ones(8, 4, device=dev, dtype=torch.bfloat16),
                 out_dtype=torch.float32)
        r['mm_out_dtype'] = True
    except Exception as exc:
        r['mm_out_dtype'] = f'{type(exc).__name__}: {exc}'
        raise

    # reference: eager FP32 policy on block 0 (also the agreement baseline for every arm)
    pre = [gd.prefill(p, g, inputs[i:i + 1]) for i in range(args.blocks)]
    torch.cuda.synchronize()
    st = gd.initial_state(p, dev)
    u, x, e1, e2 = pre[0]
    ref_nll = gd.adapt_eager(st, c, u, x, e1, e2, targets[:1])[0]
    ref_W1, ref_W2 = st['W1'][0].clone(), st['W2'][0].clone()
    st32 = gd.initial_state(p, dev)
    ref32 = {'nll': gd.adapt_eager(st32, c, u, x, e1, e2, targets[:1], head='mixed_gl32')[0]}
    ref32['W1'], ref32['W2'] = st32['W1'][0].clone(), st32['W2'][0].clone()
    r['reference_block0_p1b'] = {'adapted_mean_nll': float(ref32['nll'].mean())}
    frozen0 = gd.frozen_nll(p, g, inputs[:1], targets[:1])[0]
    r['reference_block0'] = {'adapted_mean_nll': float(ref_nll.mean()), 'frozen_mean_nll': float(frozen0.mean())}

    arms = {}
    r['latch_builds'] = {}
    for spec in args.arms.split(';' if ';' in args.arms else ','):
        name, arm = make_arm(spec, c, W1_0, W2_0, T, p=p, out=out, receipts=r['latch_builds'])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        arm.load(*pre[0], targets[:1])
        nll = arm.run_block().clone()
        torch.cuda.synchronize()
        setup = time.perf_counter() - t0
        arms[name] = arm
        rn, rw1, rw2 = ((ref32['nll'], ref32['W1'], ref32['W2']) if head_of(name) == 'mixed_gl32'
                        else (ref_nll, ref_W1, ref_W2))
        r['arms'][name] = {'first_block_seconds_incl_setup': setup, 'policy': 'P1b' if head_of(name) != 'mixed' else 'P1',
                           'peak_alloc_gib': torch.cuda.max_memory_allocated() / 2**30,
                           'nll_max_abs_vs_eager': float((nll[:, 0] - rn).abs().max()),
                           'nll_mean_diff_vs_eager': float((nll[:, 0] - rn).mean()),
                           'W1_rel_vs_eager': float((arm.W1[0] - rw1).norm() / rw1.norm()),
                           'W2_rel_vs_eager': float((arm.W2[0] - rw2).norm() / rw2.norm()),
                           'samples_ms': [], 'loop_ms': []}

    def frozen_block(i):
        return gd.frozen_nll(p, g, inputs[i:i + 1], targets[i:i + 1]).sum()

    r['arms']['frozen'] = {'samples_ms': []}
    names = list(arms)
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rnd in range(args.rounds):
        order = names[rnd % len(names):] + names[:rnd % len(names)] + ['frozen']
        for name in order:
            for i in range(args.blocks):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if name == 'frozen':
                    float(frozen_block(i))
                else:
                    arm = arms[name]
                    pr = gd.prefill(p, g, inputs[i:i + 1])
                    arm.load(*pr, targets[i:i + 1])
                    ev0.record()
                    nll = arm.run_block()
                    ev1.record()
                    float(nll.sum())
                torch.cuda.synchronize()
                r['arms'][name]['samples_ms'].append((time.perf_counter() - t0) * 1e3)
                if name != 'frozen':
                    r['arms'][name]['loop_ms'].append(ev0.elapsed_time(ev1))
    for name, row in r['arms'].items():
        s = sorted(row['samples_ms'])
        row['median_ms'] = s[len(s) // 2]
        if row.get('loop_ms'):
            l = sorted(row['loop_ms'])
            row['loop_median_ms'] = l[len(l) // 2]
            row['loop_us_per_token'] = 1e3 * row['loop_median_ms'] / T
    if args.profile:
        from torch.profiler import profile, ProfilerActivity
        for name, arm in arms.items():
            arm.load(*pre[0], targets[:1])
            arm.run_block()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                arm.run_block()
                torch.cuda.synchronize()
            ev = [e for e in prof.events() if e.device_type.name == 'CUDA']
            kern = [e for e in ev if 'memcpy' not in e.name.lower() and 'memset' not in e.name.lower()]
            agg = {}
            for e in kern:
                a = agg.setdefault(e.name[:90], [0, 0.0])
                a[0] += 1
                a[1] += e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total
            r['arms'][name]['profile'] = {'kernels': len(kern), 'events': len(ev),
                                          'top': sorted(([k, v[0], v[1] / 1e3] for k, v in agg.items()),
                                                        key=lambda z: -z[2])[:25]}
    (out / 'bench.json').write_text(json.dumps(r, indent=2))
    print(json.dumps({n: {**{k: v for k, v in row.items() if k not in ('samples_ms', 'loop_ms', 'profile')},
                          **({'device_kernels_per_block': row['profile']['kernels']} if 'profile' in row else {})}
                      for n, row in r['arms'].items()}, indent=1))


if __name__ == '__main__':
    main()
