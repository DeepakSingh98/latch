"""P-OPS5-C: elementwise register chains evaluated in the p5c warpgroup-MMA register epilogue (flat fused schedule).

Selection mirrors P-OPS2-B's attach_epilogues (pops2b_rules.py @ d8f8c0d98, integrated af3138238), restricted to
the p5c engine: a register-connected member set of one fused elementwise loop in the phase right after a p5c BF16
product moves into the product's phase when

* every member is an FP32/BF16 elementwise value over the product's element count;
* each member load is a same-index register read of another member, an identity read of the product, or a read of
  a value produced strictly before the product's phase (graph inputs included);
* no member participates in an in-place state commit and no member reads committed state.

Moving keeps every reader of a member in a later phase. The epilogue evaluates each member's identical per-operation
expression (fusion.py render) on the product's rounded-and-widened BF16 value, i.e. exactly the value a load of the
stored product yields, so results are bitwise equal to separate-phase execution. Members read later are stored by
the epilogue; the product is stored only if anything besides the epilogue reads it. No model names are inspected.
"""
from __future__ import annotations

import re

# LATCH-DENSE-EXEC: members may read the old value of in-place committed state whose commit runs at least two phases
# later (the member then moves from phase p + 1 to p, still strictly before the commit). Applied where the caller
# enables it (fusion_p6d: programs whose schedule is admitted by the long-attention protocol; the flat projection-only
# P6D proof certifies identity epilogue operands only). False: the original rule everywhere.
EARLY_STATE_READS = True



def attach(*, items, loads_of, phase_of, phases, nodes, layouts, in_place, inplace_source, register_load, numel,
           outputs=(), optimizer=False, early_state_reads=False):
    """Mutates phases/phase_of. Returns (epilogue_of: member -> product, epilogues: product -> [members])."""
    order = {node.name: k for k, node in enumerate(items)}
    readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            readers.setdefault(root, []).append((node.name, layout, shape, idx))
    committed = set(in_place)
    epilogue_of, epilogues = {}, {}
    # P6-D (optimizer=True): phases of every reader of each committed (old) state root.
    state_readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            if root in committed:
                state_readers.setdefault(root, []).append(node.name)

    def identity(layout, shape, idx, n):
        return (idx == 'i' and shape is not None and numel(shape) == n and layout.numel == n
                and layout.contiguous and layout.offset == 0)

    def component(start, q, n):
        members = set(phases[q]['loops'].get(n, ()))
        group, frontier = {start}, [start]
        while frontier:
            current = frontier.pop()
            neighbours = [root for root, layout, shape, idx in loads_of[current]
                          if root in members and shape is not None and register_load(nodes[current], root, layout, shape, idx)]
            neighbours += [reader for reader, layout, shape, idx in readers.get(current, ())
                           if reader in members and shape is not None
                           and register_load(nodes[reader], current, layout, shape, idx)]
            for other in neighbours:
                if other not in group:
                    group.add(other)
                    frontier.append(other)
        return sorted(group, key=order.get)

    for node in items:
        spec = (node.extra or {}).get('matrix_spec') if node.kind == 'gemm' else None
        if (not spec or spec.get('engine') != 'p5c' or (spec.get('out_f32') and not optimizer)
                or spec.get('batch') != 1):
            continue
        product = node.name
        layout = layouts[product]
        if layout.root != product or not layout.contiguous or layout.offset or len(layout.shape) != 2:
            continue
        p, n = phase_of[product], numel(layout.shape)
        if p + 1 >= len(phases):
            continue
        starts = [reader for reader, lay, shape, idx in readers.get(product, ())
                  if phase_of.get(reader) == p + 1 and nodes[reader].kind == 'elementwise'
                  and identity(lay, shape, idx, n) and reader not in epilogue_of]
        if not starts:
            continue
        chain = component(starts[0], p + 1, n)
        if optimizer:  # P6-D: every independent identity-reading chain of the product joins the same epilogue
            for start in starts[1:]:
                if start not in chain:
                    chain = sorted(set(chain) | set(component(start, p + 1, n)), key=order.get)
        members = set(chain)
        ok = True
        def state_ok(target):
            # P6-D: in-place state may be updated by this epilogue when its committing source is a member and every
            # non-member reader of the old value runs strictly earlier.
            return (optimizer and in_place.get(target) in members
                    and all(reader in members or phase_of.get(reader, p) < p
                            for reader in state_readers.get(target, ())))
        for name in chain:
            member = nodes[name]
            if (member.kind != 'elementwise' or member.dtype not in ('float32', 'bfloat16')
                    or numel(member.shape) != n or name in committed
                    or (name in inplace_source and not state_ok(inplace_source[name]))):
                ok = False
                break
            for root, lay, shape, idx in loads_of[name]:
                if (root in committed and early_state_reads and EARLY_STATE_READS
                        and phase_of.get(in_place[root], -1) > p + 1):
                    # LATCH-DENSE-EXEC: the old value of state committed in a strictly later phase (e.g. a bias read
                    # in a forward product's epilogue, committed by the optimizer at the end of the update): the
                    # same condition as any loop read before the commit (_check_schedule re-verifies it).
                    continue
                if root in committed:
                    ok = ok and state_ok(root) and identity(lay, shape, idx, n)
                elif root in members:
                    ok = ok and shape is not None and register_load(member, root, lay, shape, idx)
                elif root == product:
                    ok = ok and identity(lay, shape, idx, n)
                elif root in phase_of and phase_of[root] >= p:
                    ok = False
            if not ok:
                break
        if not ok:
            continue
        loop = phases[p + 1]['loops'][n]
        phases[p + 1]['loops'][n] = [m for m in loop if m not in members]
        if not phases[p + 1]['loops'][n]:
            del phases[p + 1]['loops'][n]
        for name in chain:
            phase_of[name] = p
            epilogue_of[name] = product
        epilogues[product] = chain
        phases[p].setdefault('epilogue', {})[product] = chain
    return epilogue_of, epilogues


# ------------------------------------------------------------------------------------------------ exact rewrites
def _balanced(text, start):
    """Index just past the parenthesis group opening at text[start] == '('."""
    depth = 0
    for k in range(start, len(text)):
        if text[k] == '(':
            depth += 1
        elif text[k] == ')':
            depth -= 1
            if depth == 0:
                return k + 1
    raise ValueError('unbalanced expression')


def reciprocal(expr):
    """`(1.0f / (x))` -> `__frcp_rn((x))`: rcp.rn and div.rn of 1 are the same correctly rounded IEEE result."""
    token = '(1.0f / ('
    while token in expr:
        a = expr.index(token)
        inner = a + len('(1.0f / ')
        b = _balanced(expr, inner)
        if b >= len(expr) or expr[b] != ')':
            break
        expr = expr[:a] + '__frcp_rn(' + expr[inner:b] + ')' + expr[b + 1:]
    return expr


_XDIV = re.compile(r'\(([A-Za-z_]\w*(?:\[\w*\])?|\d+\.\d*f) / \(')
# LATCH-MOESUP-ROW: `(N / (D))` with a compound numerator N (e.g. silu's `__bfloat162float(x) / (1.0f + expf(-x))`)
# whose value is evidently FP32 (a BF16 widening, an FP32 transcendental or an FP32 literal) and
# that has no top-level operator takes the same branch-free fast path; integer index arithmetic never matches.
XDIV_GENERAL = True
_FLOAT_EVIDENCE = re.compile(r'__bfloat162float\(|\bexpf\(|\b\d+\.\d*f\b')


def _top_level_operator(text):
    depth = 0
    for k, ch in enumerate(text):
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif depth == 0 and ch in '+-*/?:<>=!&|^%' and k > 0:
            return True
    return False


def _general_division(expr):
    start = 0
    while True:
        p = expr.find(' / (', start)
        if p < 0:
            return expr
        depth, k = 0, p - 1
        while k >= 0:
            if expr[k] == ')':
                depth += 1
            elif expr[k] == '(':
                if depth == 0:
                    break
                depth -= 1
            k -= 1
        den_open = p + 3
        try:
            den_end = _balanced(expr, den_open)
        except ValueError:
            return expr
        num = expr[k + 1:p]
        if (k < 0 or den_end >= len(expr) or expr[den_end] != ')' or not num.strip() or _top_level_operator(num)
                or not _FLOAT_EVIDENCE.search(num) or 'latch_xdiv' in num):
            start = p + 1
            continue
        expr = expr[:k] + f'latch_xdiv({num}, {expr[den_open:den_end]}, xok_)' + expr[den_end + 1:]
        start = k


def exact_division(expr):
    """LATCH-STEP10-SPEED: `(a / (b))` and `__frcp_rn(b)` -> `latch_xdiv(a, (b), xok_)` (kernels/latch_xdiv.cuh): the
    division's fast-path sequence without its branch; `xok_` is cleared when an operand is outside the window in which
    that sequence is the IEEE quotient, and the caller then recomputes with the original expression. Returns the new
    expression (unchanged when nothing matches)."""
    while True:
        m = _XDIV.search(expr)
        if not m:
            break
        inner = m.end() - 1
        b = _balanced(expr, inner)
        if b >= len(expr) or expr[b] != ')':
            break
        expr = expr[:m.start()] + f'latch_xdiv({m.group(1)}, {expr[inner:b]}, xok_)' + expr[b + 1:]
    if XDIV_GENERAL:
        expr = _general_division(expr)
    token = '__frcp_rn('
    while token in expr:
        a = expr.index(token)
        b = _balanced(expr, a + len(token) - 1)
        expr = expr[:a] + f'latch_xdiv(1.0f, {expr[a + len(token) - 1:b]}, xok_)' + expr[b:]
    return expr


def common_subexpressions(lines, prefix, minimum=24, limit=8):
    """Hoist identical transcendental subexpressions occurring more than once into temporaries.

    `lines` are 'lhs = expr;' statements of one element. Identical text evaluates the same operations in the same
    order on the same values, so computing it once is bitwise identical. Returns (temporary lines, new lines)."""
    temps = []
    for _ in range(limit):
        pool = '\n'.join(lines + [t for _, t in temps])
        best = None
        for k, ch in enumerate(pool):
            if ch != '(':
                continue
            if k and (pool[k - 1].isalnum() or pool[k - 1] == '_'):
                continue                         # P8-A: a call's argument list is not a standalone subexpression
            try:
                end = _balanced(pool, k)
            except ValueError:
                continue
            sub = pool[k:end]
            if len(sub) < minimum or 'expf' not in sub or '\n' in sub:
                continue
            if pool.count(sub) >= 2 and (best is None or len(sub) > len(best)):
                best = sub
        if best is None:
            break
        name = f'{prefix}{len(temps)}'
        lines = [line.replace(best, name) for line in lines]
        temps = [(n, t.replace(best, name)) for n, t in temps]
        temps.append((name, best))
    ordered, done = [], set()
    while len(ordered) < len(temps):
        for n, t in temps:
            if n in done:
                continue
            if all(m in done or m not in t for m, _ in temps if m != n):
                ordered.append(f'const float {n} = {t};')
                done.add(n)
    return ordered, lines
