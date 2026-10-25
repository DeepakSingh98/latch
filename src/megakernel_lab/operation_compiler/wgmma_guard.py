"""LATCH-COLD-PREP (megakernel-lab a5b94cb23d research/latch_ops/ops/wgmma_guard.py, copied unchanged
except this attribution and the census helper below): SASS check for loop-carried warpgroup-MMA register operands that the loop body overwrites.

CUDA 12.8 ptxas can reuse the registers of a loop-invariant HGMMA A fragment (``wgmma.mma_async`` with the A
operand in registers, defined before a loop and read by every iteration) for other values later in the loop body;
the next iteration then multiplies garbage. The PTX is correct and identical in every context; whether ptxas does
this depends on the surrounding code (observed for pflatlong::dq_item6 when a phase is compiled in a small entry,
never in the monolithic deep LM entry). This check finds the pattern in SASS:

  loop [head, latch] (a backward branch), HGMMA at p in the loop reading A registers X, some x in X written in
  (p, latch] and not written in [head, p)  ->  iteration 2 reads the value written after p.

``hazards(sass)`` returns one row per finding; ``check_cubin(path)`` runs cuobjdump. Builds that produce SASS
outside the anchor's monolithic entry (kernel split) refuse any cubin with a finding.
"""
from __future__ import annotations

import re
import subprocess

LINE = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
FUNC = re.compile(r'Function : (\S+)')
REG = re.compile(r'\bR(\d+)\b')
ACC = {'64x8': 4, '64x16': 8, '64x24': 12, '64x32': 16, '64x48': 24, '64x64': 32, '64x96': 48, '64x128': 64,
       '64x192': 96, '64x256': 128}
NO_DEST = ('ST', 'RED', 'ATOM', 'BAR', 'BRA', 'BSSY', 'BSYNC', 'WARPSYNC', 'EXIT', 'RET', 'CALL', 'MEMBAR',
           'DEPBAR', 'WARPGROUP', 'NOP', 'ERRBAR', 'CCTL', 'SYNCS', 'UTMA', 'UBLKCP', 'FENCE', 'LDGDEPBAR', 'YIELD')


def _parse(sass):
    funcs, name = {}, None
    for raw in sass.split('\n'):
        f = FUNC.search(raw)
        if f:
            name = f.group(1)
            funcs[name] = []
            continue
        m = LINE.search(raw)
        if m and name is not None:
            funcs[name].append((int(m.group(1), 16), m.group(2).strip()))
    return funcs


def _op(ins):
    parts = ins.split()
    if parts and parts[0].startswith('@'):
        parts = parts[1:]
    return (parts[0] if parts else ''), ' '.join(parts[1:])


def _written(ins):
    """Registers written by one SASS instruction (conservative width)."""
    op, rest = _op(ins)
    if not rest or op.startswith(NO_DEST):
        return set()
    if op.startswith('HGMMA'):
        m = re.match(r'HGMMA\.(\d+x\d+)x', op)
        r = REG.match(rest)
        if not (m and r):
            return set()
        return set(range(int(r.group(1)), int(r.group(1)) + ACC.get(m.group(1), 0)))
    first = rest.split(',')[0].strip()
    r = re.fullmatch(r'R(\d+)(?:\.\w+)*', first)
    if not r:
        return set()
    base = int(r.group(1))
    width = 1
    if op.startswith('LDSM'):
        width = int(op.rsplit('.', 1)[-1]) if op.rsplit('.', 1)[-1].isdigit() else 4
    elif '.128' in op:
        width = 4
    elif '.64' in op or 'WIDE' in op or op.startswith(('LDL.64', 'LDS.64', 'F2F.F64', 'DADD', 'DMUL', 'DFMA')):
        width = 2
    return set(range(base, base + width))


def _a_operand(ins):
    op, rest = _op(ins)
    if not op.startswith('HGMMA'):
        return set()
    operands = [x.strip() for x in rest.split(',')]
    if len(operands) < 2:
        return set()
    r = re.fullmatch(r'R(\d+)(?:\.\w+)*', operands[1])
    return set(range(int(r.group(1)), int(r.group(1)) + 4)) if r else set()


def hazards(sass):
    from bisect import bisect_left, bisect_right
    found = []
    # LATCH-BUILD-600: _op/_written/_a_operand are pure functions of the instruction text, and SASS repeats
    # instruction texts heavily, so each distinct text is decoded once per call.
    decoded = {}
    for name, rows in _parse(sass).items():
        index = {addr: k for k, (addr, _) in enumerate(rows)}
        loops, wpos, mma = [], {}, []
        for k, (addr, ins) in enumerate(rows):
            row = decoded.get(ins)
            if row is None:
                op, rest = _op(ins)
                row = decoded[ins] = (op, rest, tuple(sorted(_written(ins))), _a_operand(ins))
            op, rest, written, a = row
            if op.startswith('BRA'):
                t = re.search(r'0x([0-9a-f]+)', rest)
                if t and int(t.group(1), 16) <= addr and int(t.group(1), 16) in index:
                    loops.append((index[int(t.group(1), 16)], k))
            for r in written:
                wpos.setdefault(r, []).append(k)
            if a:
                mma.append((k, a))
        positions = [k for k, _ in mma]
        for head, latch in loops:
            for k, a in mma[bisect_left(positions, head):bisect_right(positions, latch)]:
                bad = []
                for r in sorted(a):
                    w = wpos.get(r, [])
                    written_before = bisect_left(w, k) > bisect_left(w, head)        # a write in [head, k)
                    written_after = bisect_right(w, latch) > bisect_right(w, k)       # a write in (k, latch]
                    if written_after and not written_before:
                        bad.append(r)
                if bad:
                    found.append({'function': name, 'hgmma': hex(rows[k][0]), 'instruction': rows[k][1][:100],
                                  'loop': [hex(rows[head][0]), hex(rows[latch][0])],
                                  'registers': [f'R{r}' for r in bad]})
    return found


def check_cubin(path, cuobjdump='cuobjdump'):
    proc = subprocess.run([cuobjdump, '-sass', str(path)], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f'cuobjdump failed on {path}: {proc.stderr[-2000:]}')
    return hazards(proc.stdout)


def census(sass):
    """Per-function warpgroup-MMA issue/serialization counts and the loop-clobber findings (LATCH-TINYMOE-BUILD)."""
    rows = {}
    for name, body in _parse(sass).items():
        ops = [_op(ins)[0] for _, ins in body]
        rows[name] = dict(instructions=len(ops), hgmma=sum(o.startswith('HGMMA') for o in ops),
                          warpgroup_depbar=sum(o.startswith('WARPGROUP.DEPBAR') for o in ops),
                          warpgroup_arrive=sum(o.startswith('WARPGROUP.ARRIVE') for o in ops),
                          calls=sum(o.startswith('CALL') for o in ops), local_loads=sum(o.startswith('LDL') for o in ops),
                          local_stores=sum(o.startswith('STL') for o in ops))
    return dict(functions=rows, hazards=hazards(sass))


if __name__ == '__main__':
    import json
    import sys
    for path in sys.argv[1:]:
        rows = check_cubin(path)
        print(json.dumps({'file': path, 'hazards': len(rows), 'rows': rows[:20]}, indent=1))
