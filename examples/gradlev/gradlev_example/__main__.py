# Provenance: new file for the package example (not in the donor); dispatches to the stage modules.
"""python -m gradlev_example [--root ABS] prepare|build|evaluate|benchmark|restart [stage options]

--root (or the GRADLEV_ROOT environment variable) is the absolute output directory shared by all stages.
"""
from __future__ import annotations

import os
import sys

STAGES = ('prepare', 'build', 'evaluate', 'benchmark', 'restart')


def _prepare(argv):
    import argparse
    from .assets import build_corpus, ensure
    ap = argparse.ArgumentParser(prog='gradlev_example prepare')
    ap.add_argument('--dev-windows', type=int, default=128,
                    help='validation windows to rebuild (2 blocks each; 128 = the donor dev split, 2048 = hash-verified)')
    ap.add_argument('--test-windows', type=int, default=0,
                    help='test windows to rebuild (512 = the donor test protocol, 8192 = hash-verified)')
    args = ap.parse_args(argv)
    for name in ('gpt2.safetensors', 'gpt2-tokenizer.json'):
        print(name, ensure(name))
    build_corpus(args.dev_windows, args.test_windows)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == '--root':
        os.environ['GRADLEV_ROOT'] = argv[1]
        argv = argv[2:]
    if not argv or argv[0] not in STAGES:
        raise SystemExit(__doc__)
    stage, rest = argv[0], argv[1:]
    if stage == 'prepare':
        return _prepare(rest)
    if stage == 'build':
        from .build import main as run
    elif stage == 'evaluate':
        from .evaluate import main as run
    elif stage == 'benchmark':
        from .bench import main as run
    else:
        from .check_state import main as run
    return run(rest)


if __name__ == '__main__':
    main()
