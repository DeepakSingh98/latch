# Provenance: adapted from the Latch repository, commit df184a5f2579 (branch codex/latch-gradlev),
# research/latch_gradlev/check_state.py (git blob 0381f63bee86b44013cb8a592b386f9bf0b4b18d).
# Changed: imports point at the local gradlev_example package; main() takes argv; the Latch arm
# loads the artifact written by `quickstart.sh build` (built here only if it is missing, as in the
# donor); the resume subprocess re-enters this package (`python -m gradlev_example restart --phase
# resume`) with the example directory on PYTHONPATH; the inductor cache defaults to
# $GRADLEV_ROOT/inductor-cache instead of the cloud-worker path; only dev blocks 0 and 1 are loaded.
# Removed: Google Cloud Storage checkpoints. The checks themselves are unchanged.
"""Carried state, restart and reset checks for the adaptation loop (Latch and conventional).

python -m gradlev_example restart --out DIR [--ckpt final.pt]

* carried: one 1024-position call == two calls of 512 positions with the fast state carried
  between calls (Latch: explicit state persists across launches; conventional: same buffers).
* restart: after 512 positions the fast state is saved to disk; a NEW process loads the program
  (Latch: same artifact; conventional: same compiled step) and the state, and runs positions
  512..1023; NLL and final state must equal the uninterrupted run bit for bit.
* reset: block B after block A (reset between) == block B from a fresh state, bit for bit.
* short count: count = 1000 < max_steps gives the first 1000 NLLs and the 1000-step state.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from . import deploy as gd
from . import model as gm
from .assets import root
from .bench import default_artifact, load_model
from .conv import ConvLoop
from .data import eval_blocks

EXAMPLE_DIR = Path(__file__).resolve().parent.parent


def same(a, b):
    return bool(torch.equal(a.cpu(), b.cpu()))


def setup(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = 'cuda'
    p, g, _ = load_model(args.ckpt, dev)
    c = gd.consts_from(p, dev)
    W1_0, W2_0 = p[f'{gm.L-1}.fc_w'].contiguous(), p[f'{gm.L-1}.pr_w'].contiguous()
    inputs, targets = eval_blocks('dev', 2)
    inputs, targets = inputs[:2].to(dev), targets[:2].to(dev)
    pre = [gd.prefill(p, g, inputs[i:i + 1]) for i in range(2)]
    return p, g, c, W1_0, W2_0, pre, targets


def latch_arm(args, p, c, W1_0, W2_0):
    from .latch_run import LatchLoop
    art = default_artifact('latch')
    if not art.exists():
        from .latch_loop import build_artifact
        build_artifact({k: v.cpu() for k, v in p.items()}, art.parent, T=1024)
    return LatchLoop(str(art), c, W1_0, W2_0, 1, 1024)


def stage_shift(arm, pre, tgt, start):
    """Stage positions start.. of one block at positions 0.. (rest zero-filled; never executed)."""
    u, x, e1, e2 = pre
    T = x.shape[1]
    pad = lambda a: torch.cat([a[:, start:], torch.zeros_like(a[:, :start])], 1)
    arm.load(pad(u), pad(x), pad(e1), pad(e2), pad(tgt))


def main(argv=None):
    # exact conventional restart needs the same inductor kernel choices in the resumed process
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(root() / 'inductor-cache'))
    os.environ.setdefault('TORCHINDUCTOR_FX_GRAPH_CACHE', '1')
    ap = argparse.ArgumentParser(prog='gradlev_example restart')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--phase', default='main')
    ap.add_argument('--arm', default='')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p, g, c, W1_0, W2_0, pre, targets = setup(args)
    tgt0, tgt1 = targets[0:1], targets[1:2]
    if args.phase == 'resume':
        snap = torch.load(out / f'snap_{args.arm}.pt', map_location='cpu')
        if args.arm == 'latch':
            arm = latch_arm(args, p, c, W1_0, W2_0)
            arm.inf.load_state_dict(snap['latch_state'])
            stage_shift(arm, pre[0], tgt0, 512)
            nll = arm.run_block(reset=False, count=512)[:512, 0].clone()
        else:
            arm = ConvLoop('compiled', c, W1_0, W2_0, 1, 1024)
            arm.load(*pre[0], tgt0)
            arm.W1.copy_(snap['W1'].cuda())
            arm.W2.copy_(snap['W2'].cuda())
            nll = arm.run_block(reset=False, start=512, stop=1024)[512:, 0].clone()
        torch.save({'nll': nll.cpu(), 'W1': arm.W1.cpu(), 'W2': arm.W2.cpu()}, out / f'resumed_{args.arm}.pt')
        return
    r = {}
    for name in ('latch', 'compiled'):
        if name == 'latch':
            arm = latch_arm(args, p, c, W1_0, W2_0)
        else:
            arm = ConvLoop('compiled', c, W1_0, W2_0, 1, 1024)
        rr = {}
        # uninterrupted
        arm.load(*pre[0], tgt0)
        full = arm.run_block(**({'count': 1024} if name == 'latch' else {}))[:, 0].clone()
        Wf1, Wf2 = arm.W1.clone(), arm.W2.clone()
        # determinism: same block again
        arm.load(*pre[0], tgt0)
        again = arm.run_block(**({'count': 1024} if name == 'latch' else {}))[:, 0].clone()
        rr['repeat_bitwise'] = same(full, again) and same(arm.W1, Wf1)
        # carried: 512 + 512
        if name == 'latch':
            arm.load(*pre[0], tgt0)
            a = arm.run_block(reset=True, count=512)[:512, 0].clone()
            snap = {'latch_state': arm.inf.state_dict()}
            torch.save(snap, out / 'snap_latch.pt')
            stage_shift(arm, pre[0], tgt0, 512)
            b = arm.run_block(reset=False, count=512)[:512, 0].clone()
        else:
            arm.load(*pre[0], tgt0)
            a = arm.run_block(reset=True, start=0, stop=512)[:512, 0].clone()
            torch.save({'W1': arm.W1.cpu(), 'W2': arm.W2.cpu()}, out / 'snap_compiled.pt')
            b = arm.run_block(reset=False, start=512, stop=1024)[512:, 0].clone()
        rr['carried_nll_bitwise'] = same(torch.cat([a, b]), full)
        rr['carried_state_bitwise'] = same(arm.W1, Wf1) and same(arm.W2, Wf2)
        rr['carried_nll_max_abs'] = float((torch.cat([a, b]) - full).abs().max())
        # restart in a new process
        cmd = [sys.executable, '-B', '-m', 'gradlev_example', 'restart', '--out', str(out), '--phase', 'resume',
               '--arm', name] + (['--ckpt', args.ckpt] if args.ckpt else [])
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join([str(EXAMPLE_DIR)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        rr['restart_returncode'] = proc.returncode
        if proc.returncode == 0:
            res = torch.load(out / f'resumed_{name}.pt')
            rr['restart_nll_bitwise'] = same(res['nll'], full[512:])
            rr['restart_state_bitwise'] = same(res['W1'], Wf1.cpu()) and same(res['W2'], Wf2.cpu())
        else:
            rr['restart_stderr'] = proc.stderr[-3000:]
        # reset between blocks: block 1 after block 0 == block 1 fresh
        arm.load(*pre[1], tgt1)
        b1_after = arm.run_block(**({'count': 1024} if name == 'latch' else {}))[:, 0].clone()
        fresh = ConvLoop('compiled', c, W1_0, W2_0, 1, 1024) if name == 'compiled' else None
        if name == 'latch':
            arm2 = latch_arm(args, p, c, W1_0, W2_0)
            arm2.load(*pre[1], tgt1)
            b1_fresh = arm2.run_block(count=1024)[:, 0].clone()
        else:
            fresh.load(*pre[1], tgt1)
            b1_fresh = fresh.run_block()[:, 0].clone()
        rr['reset_bitwise'] = same(b1_after, b1_fresh)
        if name == 'latch':
            arm.load(*pre[0], tgt0)
            short = arm.run_block(reset=True, count=1000)[:1000, 0].clone()
            rr['short_count_bitwise'] = same(short, full[:1000])
            rr['executed_count'] = int(arm.inf.executed_count)
            rr['status_for_2000'] = None
            arm.inf.set_count(2000)
            arm.inf.run()
            rr['status_for_2000'] = int(arm.inf.loop_status)
            rr['executed_for_2000'] = int(arm.inf.executed_count)
        rr['mean_nll'] = float(full.mean())
        r[name] = rr
    # cross-backend agreement on the uninterrupted block
    (out / 'state_checks.json').write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == '__main__':
    main()
