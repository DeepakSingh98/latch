"""LATCH-BUILD-600: the build-time fast paths give exactly their reference implementations' results."""
from __future__ import annotations

from collections import namedtuple
import copy
import json
import random
import re

import pytest

from megakernel_lab.operation_compiler import arena_assign, elementwise_codegen, group_rules, wgmma_guard
from megakernel_lab.operation_compiler.vendor import plan as pinned


def _arena_case(rng):
    n, phases = rng.randint(1, 150), rng.randint(1, 30)
    order = [f'b{i}' for i in range(n)]
    size = {b: rng.choice([0, 1, 63, 64, 65, 128, 500, 4096, rng.randint(0, 10000)]) for b in order}
    cls = {}
    for b in order:
        r = rng.random()
        cls[b] = ('persistent' if r < 0.15 else 'unused' if r < 0.25
                  else ('live', *sorted((rng.randint(0, phases), rng.randint(0, phases)))))
    return order, size, cls, phases


@pytest.mark.parametrize('strategy', ['largest', 'adversarial'])
def test_arena_assign_matches_pinned_packer(strategy):
    rng = random.Random(7)
    for _ in range(400):
        order, size, cls, phases = _arena_case(rng)
        expected = pinned.assign(order, size, cls, phases, strategy)
        assert arena_assign.assign(order, size, cls, phases, strategy) == expected


def test_arena_assign_falls_back_for_non_integer_bounds():
    order, size = ['a', 'b'], {'a': 64, 'b': 64}
    cls = {'a': ('live', 0, 2.5), 'b': ('live', 1, 3)}
    assert arena_assign.assign(order, size, cls, 3) == pinned.assign(order, size, cls, 3)


def test_escape_scan_matches_suffix_regex():
    rng = random.Random(0)
    alphabet = '&+-( \t)\n=ab*/%|^\x85\xa0'
    for _ in range(50000):
        text = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        end = rng.randint(0, len(text))
        assert (bool(re.search(r'(?:\+\+|--|&)\s*\(*\s*$', text[:end]))
                == elementwise_codegen._escape_before(text, end, ('++', '--', '&')))
        assert bool(re.search(r'&\s*\(*\s*$', text[:end])) == elementwise_codegen._escape_before(text, end, ('&',))
        assert (bool(re.match(r'\s*\)*\s*(?:=(?!=)|[+*/%&|^\-]=|\+\+|--)', text[end:]))
                == bool(elementwise_codegen._MUTATION_AFTER.match(text, end)))


_Layout = namedtuple('_Layout', 'shape strides offset root')


def _contiguous(shape):
    out, stride = [], 1
    for d in reversed(shape):
        out.append(stride)
        stride *= d
    return tuple(reversed(out))


def _merge_case(rng, ssa):
    n, shapes = rng.randint(2, 60), [(4,), (2, 2), (8,)]
    phases, layouts, names = [], {}, []
    for k in range(n):
        kind = rng.choice(['elementwise'] * 6 + ['gemm'])
        shape = rng.choice(shapes)
        own = [f'x{k}_{j}' for j in range(rng.randint(1, 3))]
        for name in own:
            layouts[name] = _Layout(shape, _contiguous(shape), 0, name)
            if rng.random() < 0.2:
                strides = _contiguous(shape) if rng.random() < 0.7 else tuple(reversed(_contiguous(shape)))
                layouts[name + 'v'] = _Layout(shape, strides, 0, name)
        pool = names[:] if ssa or not names else names + [f'x{rng.randint(0, n - 1)}_0']
        reads = []
        for _ in range(rng.randint(0, 4)):
            if pool:
                c = rng.choice(pool)
                reads.append(c + 'v' if c + 'v' in layouts and rng.random() < 0.5 else c)
        reads = list(dict.fromkeys(reads))
        expr = ' + '.join(f'v_{r}[i]' for r in reads) or '0.0f'
        body = [f'const float r_{name} = {expr}; v_{name}[i] = r_{name};' for name in own]
        phases.append(dict(kind=kind, shape=list(shape), names=own, targets=own[:], reads=reads, writes=own[:],
                           body=body if kind == 'elementwise' else [], name=own[0]))
        names += own
    for phase in phases:
        for r in phase['reads']:
            layouts.setdefault(r, _Layout((4,), (1,), 0, r))
    inplace = set(rng.sample(names, k=min(len(names), rng.randint(0, 2))))
    return phases, layouts, inplace


def _index(layout, shape, i):
    if group_rules._identity(layout, tuple(shape)):
        return i
    if len(layout.shape) == 2:
        return f'{i}%2*2+{i}/2'
    raise ValueError


def test_stale_merge_matches_incremental_merge():
    rng = random.Random(1)
    merges = 0
    for _ in range(1500):
        phases, layouts, inplace = _merge_case(rng, ssa=rng.random() < 0.7)
        kw = dict(ident=lambda n: 'v_' + n, index=_index, inplace_roots=inplace)
        stale = group_rules._merge_elementwise_chains_stale(copy.deepcopy(phases), layouts, **kw)
        incremental = group_rules._merge_elementwise_chains_incremental(copy.deepcopy(phases), layouts, **kw)
        assert json.dumps(stale, sort_keys=True) == json.dumps(incremental, sort_keys=True)
        merges += len(stale[1])
    assert merges > 1000


def test_sass_hazard_decoding_memo_is_transparent():
    rng = random.Random(0)
    forms = ['FADD R{a}, R{b}, R{c}', 'LDS.128 R{a}, [R{b}]', 'HGMMA.64x64x16.F32.BF16 R{a}, R{b}, gdesc[UR4], R{a}',
             '@P0 BRA 0x{t:04x}', 'STS [R{a}], R{b}', 'MOV R{a}, R{b}', 'LDSM.16.M88.4 R{a}, [R{b}]',
             'HGMMA.64x128x16.F32.BF16 R{a}, R{b}.reuse, gdesc[UR8], R{a}']
    found = 0
    for _ in range(200):
        lines = []
        for f in range(rng.randint(1, 3)):
            lines.append(f'Function : fn{f}')
            n = rng.randint(5, 80)
            for k in range(n):
                ins = rng.choice(forms).format(a=rng.randint(0, 40), b=rng.randint(0, 40), c=rng.randint(0, 40),
                                               t=rng.randint(0, n - 1) * 16)
                lines.append(f'        /*{k * 16:04x}*/                   {ins} ;')
        sass = '\n'.join(lines)
        rows = wgmma_guard.hazards(sass)
        # Reference: the same scan with undecoded per-instruction calls.
        reference = []
        for name, body in wgmma_guard._parse(sass).items():
            index = {addr: k for k, (addr, _) in enumerate(body)}
            loops, wpos, mma = [], {}, []
            for k, (addr, ins) in enumerate(body):
                op, rest = wgmma_guard._op(ins)
                if op.startswith('BRA'):
                    t = re.search(r'0x([0-9a-f]+)', rest)
                    if t and int(t.group(1), 16) <= addr and int(t.group(1), 16) in index:
                        loops.append((index[int(t.group(1), 16)], k))
                for r in wgmma_guard._written(ins):
                    wpos.setdefault(r, []).append(k)
                a = wgmma_guard._a_operand(ins)
                if a:
                    mma.append((k, a))
            for head, latch in loops:
                for k, a in mma:
                    if not head <= k <= latch:
                        continue
                    bad = [r for r in sorted(a)
                           if any(k < w <= latch for w in wpos.get(r, ())) and not any(head <= w < k for w in wpos.get(r, ()))]
                    if bad:
                        reference.append((name, hex(body[k][0]), tuple(f'R{r}' for r in bad)))
        assert [(r['function'], r['hgmma'], tuple(r['registers'])) for r in rows] == reference
        found += len(rows)
    assert found
