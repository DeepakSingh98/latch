"""Summarize fetched suite receipts: per case, per arm median-of-round-medians, kernels, setup, memory, audit.

  python3 analyze.py results/fetch/builds/mlp [--json out.json]
Qualification requires complete clocks from a completed comparison and both passing,
conclusive CPU state audits. Classification uses the fastest eligible compiled/captured
control. Raw ratios against complete measured controls and matched arithmetic remain
diagnostics; they never qualify a native arm or a control with missing/failed gates.
"""
import json
import math
import statistics
import sys
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def comparison_rounds(root, case):
    """The public compare completion journals bind requested arms and round indices."""
    expected, outcomes = set(), {}
    for path in sorted(root.glob('gpu-all-*.json')):
        journal = load(path)
        if not isinstance(journal, list):
            continue
        for row in journal:
            args = row.get('args', []) if isinstance(row, dict) else []
            if len(args) != 5 or args[:3] != ['run-one', root.name, case]:
                continue
            try:
                index = int(args[4])
            except (TypeError, ValueError):
                continue
            if index < 0:
                continue
            expected.add(index)
            outcomes[(args[3], index)] = row.get('returncode') == 0
    return expected, outcomes


def audit_status(audit, arm):
    if audit is None:
        return 'pending'
    if not isinstance(audit, list):
        return 'incomplete'
    if any(row.get('missing_reference_input') for row in audit):
        return 'missing_reference'
    rows = [row for row in audit if row.get('arm') == arm]
    if not rows:
        return 'not_audited'
    if any(row.get('missing') for row in rows):
        return 'missing'
    if any(row.get('inconclusive') for row in rows):
        return 'inconclusive'
    if len(rows) != 2 or {row.get('state') for row in rows} != {'fresh', 'trained'}:
        return 'incomplete'
    return 'pass' if all(row.get('passed') is True for row in rows) else 'fail'


def positive_finite(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def summarize(root):
    root = Path(root)
    out = {}
    for case in sorted(p for p in root.iterdir() if p.is_dir() and p.name != 'logs'):
        runs = {}
        for f in case.glob('*-r[0-9]*.json'):
            r = load(f)
            if r is None:
                continue
            runs.setdefault(r['arm'], []).append(r)
        if not runs:
            continue
        audit = load(case / 'audit.json')
        expected, outcomes = comparison_rounds(root, case.name)
        arms = {}
        for arm, rows in runs.items():
            ok = [r for r in rows if r.get('ok')]
            meds = [r['median_update_ms'] for r in ok if positive_finite(r.get('median_update_ms'))]
            r0 = next((r for r in rows if r['round'] == 0), rows[0])
            observed = [r.get('round') for r in rows]
            complete = (bool(expected) and len(observed) == len(expected)
                        and set(observed) == expected and len(ok) == len(rows)
                        and len(meds) == len(rows) and all(r.get('finite') is True for r in rows)
                        and all(outcomes.get((arm, index)) is True for index in observed))
            status = audit_status(audit, arm)
            errors = [(r.get('error') or '').strip().splitlines()[-1:] for r in rows if not r.get('ok')]
            arms[arm] = dict(ok=len(ok) == len(rows) and bool(ok), rounds=len(rows),
                             median_ms=statistics.median(meds) if meds else None, round_ms=meds,
                             expected_round_indices=sorted(expected), observed_round_indices=sorted(observed),
                             timing_complete=bool(complete),
                             timing_status=('complete' if complete else 'missing_comparison_completion' if not expected else 'incomplete'),
                             kernels=r0.get('cuda_kernel_count'), restart_exact=r0.get('saved_restart_exact'),
                             finite=all(r.get('finite') for r in ok) if ok else None,
                             setup_s=r0.get('setup_seconds'), setup_peak_mib=(r0.get('setup_peak') or {}).get('allocated_mib'),
                             replay_peak_mib=(r0.get('replay_peak') or {}).get('allocated_mib'),
                             audit=True if status == 'pass' else False if status == 'fail' else None,
                             audit_status=status, qualified=bool(complete and status == 'pass'),
                             error=errors or None)
        controls = {a: v for a, v in arms.items() if not a.startswith('latch') and a != 'eager'
                    and v['timing_complete']}
        fp32_case = 'fp32' in case.name
        practical_only = lambda a: 'amp' in a or (fp32_case and a.endswith('-bf16'))
        matched = {a: v for a, v in controls.items() if not practical_only(a)}
        eligible = {a: v for a, v in controls.items() if v['qualified']}
        pick = lambda d: min(d, key=lambda a: d[a]['median_ms']) if d else None
        best, best_any, best_matched = pick(eligible), pick(controls), pick(matched)
        natives = {}
        for a, v in arms.items():
            if a.startswith('latch') and v['median_ms']:
                ratio = v['median_ms'] / arms[best_any]['median_ms'] if best_any else None
                qualified_ratio = v['median_ms'] / arms[best]['median_ms'] if best and v['qualified'] else None
                if v['audit_status'] == 'fail':
                    cls = 'numerical failure'
                elif not v['qualified']:
                    cls = 'unqualified'
                elif qualified_ratio is None:
                    cls = 'no eligible control'
                else:
                    cls = 'faster' if qualified_ratio < 0.99 else 'within10%' if qualified_ratio <= 1.10 else 'slower'
                natives[a] = dict(ratio=ratio, versus=best_any, classification=cls,
                                  ratio_scope='raw timing diagnostic, not qualification', qualified=v['qualified'],
                                  ratio_matched=v['median_ms'] / arms[best_matched]['median_ms'] if best_matched else None,
                                  versus_matched=best_matched,
                                  ratio_eligible=qualified_ratio,
                                  versus_eligible=best)
        prep = load(case / 'prepare.json') or {}
        out[case.name] = dict(arms=arms, best_eligible_control=best, best_control_any=best_any,
                              best_matched_control=best_matched, natives=natives,
                              build={k: dict(ok=v['ok'], seconds=v['seconds'],
                                             execution=(v.get('receipt') or {}).get('execution'),
                                             selected=((v.get('receipt') or {}).get('execution_selection') or {}).get('selected'))
                                     for k, v in prep.get('variants', {}).items()})
    return out


def table(summary):
    lines = []
    for case, s in summary.items():
        lines.append(f"## {case}  fastest control: {s['best_control_any']}  matched: {s['best_matched_control']}  "
                     f"gate-eligible: {s['best_eligible_control']}")
        for arm, v in sorted(s['arms'].items(), key=lambda kv: (kv[1]['median_ms'] or 1e9)):
            n = s['natives'].get(arm, {})
            ratio = ''
            if n:
                ratio = f"[{n['classification']}]"
                if n.get('ratio_eligible') is not None:
                    ratio += f" eligible {n['ratio_eligible']:.3f} vs {n['versus_eligible']}"
                if n.get('ratio') is not None:
                    ratio += f"; raw {n['ratio']:.3f} vs {n['versus']}"
                if n.get('ratio_matched') is not None:
                    ratio += f"; raw matched {n['ratio_matched']:.3f}"
            ms = f"{v['median_ms']:.4f}" if v['median_ms'] else 'FAIL'
            lines.append(f"  {arm:18s} {ms:>9s} ms k={v['kernels']} audit={v['audit_status']} "
                         f"timing={v['timing_status']} qualified={v['qualified']} restart={v['restart_exact']} "
                         f"rounds={[round(x, 4) for x in v['round_ms']]} {ratio} {v['error'] or ''}")
    return '\n'.join(lines)


if __name__ == '__main__':
    s = summarize(sys.argv[1])
    print(table(s))
    if '--json' in sys.argv:
        Path(sys.argv[sys.argv.index('--json') + 1]).write_text(json.dumps(s, indent=2, default=str) + '\n')
