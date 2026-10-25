# Provenance: adapted from the Latch repository, commit df184a5f2579 (branch codex/latch-gradlev),
# research/latch_gradlev/evaluate.py (git blob 92f19adb1d18921a1b0ea668244e54b8b2fc9ca2).
# Changed: imports point at the local gradlev_example package; main() takes argv; Latch backends load
# the artifact written by `quickstart.sh build` instead of rebuilding inside the evaluation (pass
# --rebuild 1 for the donor behaviour); --blocks loads only the first N blocks; the per-block token
# NLLs are also saved (adapted_nll.pt) so a Latch run and a conventional run can be compared offline,
# and block-0 means are reported (comparable with the donor bench's block-0 reference).
# The protocol (one prefill per block, frozen NLL, fresh state per block, score-before-update, FP64
# sums, position bins, carry-pairs deviation) is unchanged.
"""Held-out deployment evaluation: frozen vs adapted NLL on the conventional graph backend and Latch.

python -m gradlev_example evaluate --out DIR --split dev|test [--ckpt final.pt] [--blocks N]
       [--backends latch | graph_hybrid | graph,latch]

Every block: one prefill; frozen NLL (no writes, batched head); adapted NLL per backend
(fresh state, score-before-update, T serial steps). Per-token NLLs are summed in FP64.
Reports means, frozen-adapted deltas, position-binned deltas, per-block times, backend agreement.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import deploy as gd
from . import model as gm
from .bench import default_artifact, load_model
from .conv import ConvLoop
from .data import eval_blocks

BINS = (0, 64, 128, 256, 512, 768, 1024)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gradlev_example evaluate')
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--blocks', type=int, default=0)
    ap.add_argument('--backends', default='graph,latch')
    ap.add_argument('--tag', default='')
    ap.add_argument('--rebuild', type=int, default=0, help='build a fresh Latch artifact inside the output directory')
    ap.add_argument('--carry-pairs', type=int, default=0,
                    help='carry the fast state from block 2i into block 2i+1 (same document window); deviation from the reset policy')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    p, g, meta = load_model(args.ckpt, dev)
    c = gd.consts_from(p, dev)
    W1_0, W2_0 = p[f'{gm.L-1}.fc_w'].contiguous(), p[f'{gm.L-1}.pr_w'].contiguous()
    inputs, targets = eval_blocks(args.split, args.blocks or None)
    n = args.blocks or inputs.shape[0]
    inputs, targets = inputs[:n], targets[:n]
    T = inputs.shape[1]
    arms, setup = {}, {}
    for name in args.backends.split(','):
        t0 = time.perf_counter()
        if name.startswith('latch'):
            from .latch_run import LatchLoop
            head = 'mixed_gl32' if 'gl32' in name else 'mixed'
            if args.rebuild:
                from .latch_loop import build_artifact
                art, rec = build_artifact({k: v.cpu() for k, v in p.items()}, out / f'build_{args.tag or "m"}_{name}',
                                          T=T, head=head)
                setup[name] = {'build_receipt': {k: rec[k] for k in ('source_hash', 'build_seconds', 'setup_wall_seconds')}}
            else:
                art = default_artifact(name)
                if not (art / 'artifact.json').exists():
                    raise SystemExit(f'{art} is missing; run quickstart.sh build first')
                rec = json.loads((art / 'artifact.json').read_text())
                setup[name] = {'artifact': str(art), 'build_receipt': {k: rec[k] for k in ('source_hash', 'build_seconds')}}
            arms[name] = LatchLoop(str(art), c, W1_0, W2_0, 1, T)
        else:
            arms[name] = ConvLoop(name, c, W1_0, W2_0, 1, T)
        # warm (graph capture / first launch) on block 0
        u, x, e1, e2 = gd.prefill(p, g, inputs[:1].to(dev))
        arms[name].load(u, x, e1, e2, targets[:1].to(dev))
        arms[name].run_block()
        torch.cuda.synchronize()
        setup.setdefault(name, {})['setup_seconds'] = time.perf_counter() - t0
    tot = {'frozen': 0.0, **{k: 0.0 for k in arms}}
    pos = {k: torch.zeros(T, dtype=torch.float64) for k in ['frozen', *arms]}
    tokens = {k: torch.zeros(n, T, dtype=torch.float64) for k in ['frozen', *arms]}
    times = {k: [] for k in ['frozen', 'prefill', *arms]}
    agree = {'max_abs': 0.0, 'sum_abs': 0.0}
    odd = {k: 0.0 for k in ['frozen', *arms]}
    for i in range(n):
        tok, tgt = inputs[i:i + 1].to(dev), targets[i:i + 1].to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fz = gd.frozen_nll(p, g, tok, tgt)[0]
        torch.cuda.synchronize()
        times['frozen'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pr = gd.prefill(p, g, tok)
        torch.cuda.synchronize()
        times['prefill'].append(time.perf_counter() - t0)
        pos['frozen'] += fz.double().cpu()
        tokens['frozen'][i] = fz.double().cpu()
        if i % 2 == 1:
            odd['frozen'] += float(fz.double().sum())
        res = {}
        carry = bool(args.carry_pairs and i % 2 == 1)
        for name, arm in arms.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            arm.load(*pr, tgt)
            nll = arm.run_block(reset=not carry)[:, 0].double()
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t0)
            res[name] = nll.cpu()
            pos[name] += res[name]
            tokens[name][i] = res[name]
            if i % 2 == 1:
                odd[name] += float(res[name].sum())
        if len(res) == 2:
            a, b = res.values()
            agree['max_abs'] = max(agree['max_abs'], float((a - b).abs().max()))
            agree['sum_abs'] += float((a - b).abs().sum())
    r = {'meta': meta, 'split': args.split, 'blocks': n, 'targets': n * T, 'setup': setup,
         'carry_pairs': args.carry_pairs,
         'mean_nll': {k: float(v.sum() / (n * T)) for k, v in pos.items()},
         'position_bins': {}, 'median_block_seconds': {}, 'backend_agreement': {
             'max_abs_token_nll': agree['max_abs'], 'mean_abs_token_nll': agree['sum_abs'] / (n * T)}}
    for k in arms:
        r['mean_nll'][f'delta_{k}_minus_frozen'] = r['mean_nll'][k] - r['mean_nll']['frozen']
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        r['position_bins'][f'{lo}-{hi}'] = {k: float(v[lo:hi].sum() / (n * (hi - lo))) for k, v in pos.items()}
    for k, v in times.items():
        s = sorted(v)
        r['median_block_seconds'][k] = s[len(s) // 2]
    if args.carry_pairs or True:
        r['second_half_blocks_mean_nll'] = {k: float(v) / (n // 2 * T) for k, v in odd.items()} if n >= 2 else {}
    r['block0_mean_nll'] = {k: float(v[0].mean()) for k, v in tokens.items()}
    torch.save({k: v for k, v in pos.items()}, out / 'position_sums.pt')
    torch.save(tokens, out / 'adapted_nll.pt')
    (out / 'eval.json').write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
