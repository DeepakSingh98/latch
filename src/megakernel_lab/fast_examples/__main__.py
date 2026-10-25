"""Installed-wheel entry point for the complete-update examples."""
import argparse
import json
import os
from pathlib import Path

FAMILIES = ('mlp', 'ordinary', 'ffn', 'mixed', 'dynamic', 'adaptive', 'energy')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('LATCH_FAST_ROOT', './latch-fast')),
                        help='Owned build/results directory; preserve its absolute path between CPU and GPU hosts')
    sub = parser.add_subparsers(dest='action', required=True)
    ls = sub.add_parser('list', help='List shipped cases without building them')
    ls.add_argument('family', nargs='?', choices=FAMILIES)
    for name in ('prepare', 'compare', 'verify', 'train', 'report'):
        p = sub.add_parser(name)
        p.add_argument('family', choices=FAMILIES)
        p.add_argument('cases', nargs='*', help='Default: every shipped case in the family')
        p.add_argument('--workers', type=int, default=1, help='CPU processes; each process uses one compute thread')
        p.add_argument('--rounds', type=int, default=3, help='Paired GPU rounds, one arm at a time')
        p.add_argument('--arms', default='', help='Optional arm filter; adaptive/energy verify --arms cpu audits saved GPU observations; include eager for an additional control')
        p.add_argument('--variants', default='latch,latch-bf16proj', help='Native build variants; latch is the normal API')
        p.add_argument('--json', type=Path, help='Write the report JSON here')
    args = parser.parse_args(argv)
    os.environ['LATCH_FAST_ROOT'] = str(args.root.resolve())
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'MAX_JOBS'):
        os.environ[name] = '1'
    from . import suite
    suite.ROOT = args.root.resolve()
    if args.action == 'list':
        result = {}
        for family in ([args.family] if args.family else FAMILIES):
            if family == 'adaptive':
                result[family] = ['m8k', 'm2k', 'm8k_c64']
            elif family == 'energy':
                result[family] = ['sd', 'wb']
            else:
                result[family] = list(suite.fam(family).CASES)
        print(json.dumps(result, indent=2))
        return 0
    if args.workers < 1 or args.rounds < 1:
        parser.error('workers and rounds must be positive')
    os.environ['LATCH_FAST_ARMS'] = args.arms
    os.environ['LATCH_FAST_VARIANTS'] = args.variants
    if args.family in ('adaptive', 'energy'):
        from . import cd_driver
        return cd_driver.command(args)
    F = suite.fam(args.family)
    if unknown := set(args.cases) - F.CASES.keys():
        parser.error('unknown cases: ' + ', '.join(sorted(unknown)))
    if args.action == 'report':
        from .analyze import summarize, table
        report = summarize(suite.ROOT / args.family)
        if args.cases:
            report = {k: v for k, v in report.items() if k in args.cases}
        print(table(report))
        if args.json:
            suite.dump(args.json, report)
        return 0
    if args.action == 'train':
        parser.error('the train action is available for energy; compare includes 40 changing warm-up updates elsewhere')
    action = {'prepare': 'prepare-all', 'compare': 'gpu-all', 'verify': 'audit-all'}[args.action]
    if args.action in ('prepare', 'verify'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    return 0 if suite.children(action, args.family,
                               args.rounds if args.action == 'compare' else args.workers, args.cases) else 1


if __name__ == '__main__':
    raise SystemExit(main())
