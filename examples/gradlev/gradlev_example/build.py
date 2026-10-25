# Provenance: new file for the package example (not in the donor). It calls the donor's
# gl/latch_loop.build_artifact() (copied unchanged as latch_loop.py, latch@df184a5f2579) the way the
# donor's bench.py 'latch=build' arm does, but as a separate stage writing to $GRADLEV_ROOT.
"""Capture the GradLev adaptation loop and lower it with loop_schedule='vector_resident'.

python -m gradlev_example build [--policy P1|P1b]

capture_loop_inference (CPU, real GPT-2 values as example bindings) +
prepare_inference(..., lower_options={'loop_schedule': 'vector_resident'}) + nvcc.
Output: $GRADLEV_ROOT/build_latch/artifact (P1) or $GRADLEV_ROOT/build_latchgl32/artifact (P1b).
"""
from __future__ import annotations

import argparse
import json

from . import model as gm
from .assets import ensure, root


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gradlev_example build')
    ap.add_argument('--policy', choices=('P1', 'P1b'), default='P1',
                    help='P1: BF16-rounded softmax-gradient weights (primary); P1b: FP32 weights (single-pass head)')
    ap.add_argument('--lower-options', default='',
                    help="JSON lower_options (default {'loop_schedule': 'vector_resident'})")
    args = ap.parse_args(argv)
    name, head = ('latch', 'mixed') if args.policy == 'P1' else ('latchgl32', 'mixed_gl32')
    out = root() / f'build_{name}'
    if (out / 'artifact' / 'artifact.json').exists():
        print(f'reusing {out / "artifact"} (delete {out} to rebuild)')
        return
    if (out / 'artifact').exists():
        raise SystemExit(f'{out / "artifact"} exists without artifact.json (interrupted build); delete {out} and retry')
    import torch
    torch.set_num_threads(1)
    from .latch_loop import build_artifact
    p = gm.gpt2_params(ensure('gpt2.safetensors'))
    options = json.loads(args.lower_options) if args.lower_options else {'loop_schedule': 'vector_resident'}
    artifact, receipt = build_artifact(p, out, T=1024, B=1, lower_options=options, head=head)
    keep = {k: receipt.get(k) for k in ('source_hash', 'capture_wall_seconds', 'lower_seconds', 'build_seconds',
                                         'setup_wall_seconds', 'arena_bytes', 'shared_bytes', 'lower_options',
                                         'torch', 'arch')}
    keep.update(policy=args.policy, head=head, artifact=str(artifact),
                donor_source_hash={'P1': '84febc5eb2af0656674186200f871c08a7395e529fe821612c916761799b2304',
                                   'P1b': 'c6be88615968f0b4c0d06371f5c2fc14672d24dfc28c49ce9b10c8a6f493c6c9'}[args.policy])
    (out / 'build.json').write_text(json.dumps(keep, indent=2, sort_keys=True) + '\n')
    print(json.dumps(keep, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
