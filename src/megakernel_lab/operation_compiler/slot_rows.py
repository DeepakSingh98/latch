"""Token-row groups (``accumulation_rules`` ``rslot``): slot-axis chains evaluated on chip.

Some models keep per-token FP32 state ``[T, W]`` and per-slot state ``[T, K, W]`` (K selected columns per token).
Between cross-token operations (attention, expert dispatch, matrix products) much of the work is a function of one
(token, channel) column across the K slots:

* elementwise phases of iteration shape ``[T, K..., W]`` (any K >= 1, including ``[T, W]``),
* exact slot-axis reductions ``[T, K, W] -> [T, W]`` (``exact_cta_tree_per_thread``, K <= 32; lower.py records
  ``slot_reduction`` on them when the rule is on),

plus exact last-dimension row reductions of such column values (``exact_cta_tree_per_warp``, wide form, W > 256)
and their row-shaped elementwise epilogues. Without the rule each of these is its own grid-wide pass that
materializes every intermediate.

This pass forms connected groups of such phases over one (T, W) and emits ONE warp-per-token-row phase. Lane ``l`` of
the warp owning row ``o`` evaluates, for each column ``c = 32 w + l``:

1. every column member, for each of its slots, in program order, with its unchanged rendered expression (operands of
   earlier members are registers; other operands are loaded at the member's own flat index for that slot);
2. each slot reduction's exact tree (``(0.0f + leaf)`` leaves, the xor-butterfly lane-0 tree of
   ``codegen_rules``, then the same scale/rounding) on those registers;
3. the leaf of each row reduction into ``v[w % 8]`` (first pass ``0.0f + leaf``, later passes ``v + leaf``; the
   per-thread order of the 256-thread CTA tree, as the warp-per-row engine and row_fusion evaluate it).

After the columns: the identical warp butterflies and pruned second stage of each row reduction, then the row-shaped
epilogues (every lane evaluates; lane 0 stores). A value is stored only when something outside the group reads it or
it is kept (graph output, commit, placement); every stored value is stored once, at its own address. Every operation
keeps its expression, operands and rounding (--fmad=false), every reduction its leaf order and tree: bitwise equal.

Admission is structural (shapes, layouts, dtypes, phase forms, readers); model names are never consulted. Only
forward phases (before the first backward or vocabulary-gradient phase) are candidates, and a phase whose output
feeds the source of a row reduction or row softmax that is not a candidate is left to row_fusion. A member read of
a group value must be column-local (the read layout maps member element (t, s, c) to the value's element
(t, s', c)); column members never read row values of their own group (no full epilogues in this version); members
with effects, in-place roots, accumulating stores or inlined producers are refused; a group whose contraction would
create a dependency cycle is refused.
"""
from __future__ import annotations

import math
import re

MIN_ROWS = 1024
MAX_SLOTS = 32
MAX_GROUP_EXPRESSIONS = 512          # rendered member lines x slots per column (code-size guard)
VERSION = 1
_BACKWARD = re.compile(r'backward|vocab_gradient')

_V_READ = re.compile(r'\bv_(\w+)\[')
_R_REF = re.compile(r'\br_(\w+)\b')


def enabled():
    from .accumulation_rules import active
    return active("rslot")


def _numel(shape):
    return math.prod(int(d) for d in shape)


def _contiguous(shape):
    out, step = [], 1
    for d in reversed(shape):
        out.append(step)
        step *= int(d)
    return tuple(reversed(out))


def _is_root_contiguous(layout, name):
    return (layout is not None and layout.root == name and not layout.offset
            and tuple(layout.strides) == _contiguous(layout.shape))


def _element(layout, shape, flat):
    """Root element addressed when iteration shape ``shape`` reads ``layout`` at flat index ``flat`` (lower._index)."""
    padded = (1,) * (len(shape) - len(layout.shape)) + tuple(layout.shape)
    strides = (0,) * (len(shape) - len(layout.shape)) + tuple(layout.strides)
    offset, rest = int(layout.offset), flat
    coords = []
    for b in reversed(shape):
        coords.append(rest % b)
        rest //= b
    coords.reverse()
    for c, a, st in zip(coords, padded, strides):
        if a != 1 and st:
            offset += c * st
    return offset


def _slot_map(layout, root_layout, shape, T, K, W):
    """For a reader of iteration shape ``shape`` (numel T*K*W) reading ``layout`` of a root [T, Kr, W]: the root slot
    s' read at member slot s, for s in range(K), if every read is column-local (same token, same channel); else None."""
    Kr = _numel(root_layout.shape) // (T * W)
    if Kr * T * W != _numel(root_layout.shape):
        return None
    if len(layout.shape) > len(shape):
        return None
    padded = (1,) * (len(shape) - len(layout.shape)) + tuple(layout.shape)
    if any(a not in (1, b) for a, b in zip(padded, shape)):
        return None
    out = []
    for s in range(K):
        seen = None
        for t in sorted({0, 1, T // 2, T - 1}):
            for c in sorted({0, 1, 31, W // 2, W - 1}):
                e = _element(layout, shape, (t * K + s) * W + c)
                if not 0 <= e < T * Kr * W:
                    return None
                t2, s2, c2 = e // (Kr * W), (e // W) % Kr, e % W
                if t2 != t or c2 != c or (seen is not None and s2 != seen):
                    return None
                seen = s2
        out.append(seen)
    return out


def _parse(line):
    """(ctype, name, expression, store) of a rendered elementwise body line, or None."""
    head, sep, tail = line.partition('; ')
    if not sep:
        head, tail = line.rstrip(), ''
        if not head.endswith(';'):
            return None
        head = head[:-1]
    m = re.match(r'const (float|__nv_bfloat16) r_(\w+) = (.*)$', head)
    if not m:
        return None
    ctype, name, expr = m.groups()
    tail = tail.strip()
    if tail and tail != f'v_{name}[i] = r_{name};':
        return None
    return ctype, name, expr, bool(tail)


def _member_kind(phase, layouts, dtypes):
    """('col', T, K, W) | ('slot', T, K, W) | ('rowred', T, W) | ('row', T) | reason string (refused) | None."""
    kind = phase.get('kind')
    if kind not in ('elementwise', 'reduction'):
        return None
    if phase.get('internal_mutations') or phase.get('counter_effects') or phase.get('requires_full_grid'):
        return 'effects'
    if kind == 'elementwise':
        shape = tuple(int(d) for d in phase.get('shape') or ())
        if not shape:
            return None
        if len(shape) >= 2 and shape[0] >= MIN_ROWS and shape[-1] % 32 == 0 and 256 < shape[-1] <= 2048:
            K = _numel(shape[1:-1])
            mk = ('col', shape[0], K, shape[-1]) if 1 <= K <= MAX_SLOTS else 'slots>max'
        elif shape[0] >= MIN_ROWS and _numel(shape[1:]) == 1:
            mk = ('row', shape[0])
        else:
            return None
        if isinstance(mk, str):
            return mk
        if not phase.get('body') or len(phase['body']) != len(phase['names']):
            return 'no body'
        parsed = [_parse(line) for line in phase['body']]
        if any(x is None for x in parsed):
            return 'unparsed line'
        if any(re.search(r'\b(?:threadIdx|blockIdx|blockDim|gridDim|lane|smem|__shfl\w*|atomic\w*)\b', line)
               for line in phase['body']):
            return 'thread identity'
        for (ctype, name, _, stored) in parsed:
            if stored:
                lay = layouts.get(name)
                if not _is_root_contiguous(lay, name) or _numel(lay.shape) != _numel(shape):
                    return 'stored name not a contiguous root'
                if dtypes.get(name) not in ('float32', 'bfloat16'):
                    return 'dtype'
        return mk
    info = phase.get('slot_reduction')
    if info:
        if phase.get('engine') != 'exact_cta_tree_per_thread' or len(phase['names']) != 1:
            return 'slot reduction form'
        src = layouts.get(info['source'])
        out = phase['names'][0]
        if (src is not None and len(src.shape) == 3 and info['axis'] == 1 and dtypes.get(out) == 'float32'
                and dtypes.get(info['source']) in ('float32', 'bfloat16')
                and src.shape[0] >= MIN_ROWS and src.shape[2] % 32 == 0 and 256 < src.shape[2] <= 2048
                and 1 <= src.shape[1] <= MAX_SLOTS and int(info['extent']) == src.shape[1]
                and out in layouts and _numel(layouts[out].shape) == src.shape[0] * src.shape[2]
                and _is_root_contiguous(layouts.get(out), out)):
            return ('slot', src.shape[0], src.shape[1], src.shape[2])
        return 'slot reduction shape'
    info = phase.get('row_reduction')
    if info:
        if phase.get('engine') != 'exact_cta_tree_per_warp' or len(phase['names']) != 1:
            return 'row reduction engine ' + str(phase.get('engine'))
        W, T = int(info['width']), int(info['rows'])
        code = phase.get('code') or ''
        out = phase['names'][0]
        if not (W % 32 == 0 and 256 < W <= 2048 and T >= MIN_ROWS):
            return None
        if ('v[w] = v[w] + ' in code and dtypes.get(out) == 'float32' and dtypes.get(info['source']) == 'float32'
                and _is_root_contiguous(layouts.get(out), out) and _numel(layouts[out].shape) == T):
            return ('rowred', T, W)
        return 'row reduction form'
    return None


def fuse_token_rows(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, probe=None):
    """Return (phases, records, elided_roots)."""
    from .codegen_rules import PRUNED_WARP_TREE, _butterfly0
    from .row_fusion import _dependency_order, _fusion_dependencies
    from .group_rules import _topological_dependencies
    if not PRUNED_WARP_TREE:
        return phases, [dict(refused='second stage not pruned')], set()
    root = lambda n: layouts[n].root if n in layouts else n
    writer, readers = {}, {}
    for k, p in enumerate(phases):
        for w in p['writes']:
            writer.setdefault(root(w), []).append(k)
        for r in p['reads']:
            readers.setdefault(root(r), set()).add(k)
    info, reasons, examples = {}, {}, {}
    first_backward = next((k for k, p in enumerate(phases) if _BACKWARD.search(str(p.get('kind', ''))) or any(
        _BACKWARD.search(str(t)) for t in p.get('targets', ()))), len(phases))
    sum1 = {p['names'][0]: k for k, p in enumerate(phases) if p.get('value_expr') and p.get('value_refs') is not None}
    for k, p in enumerate(phases):
        if k >= first_backward:
            break
        mk = _member_kind(p, layouts, dtypes)
        if mk is not None and not isinstance(mk, str):
            if any(root(n) in inplace_roots for n in p['names']):
                mk = 'inplace root'
            elif any(len(writer.get(root(n), ())) > 1 for n in p['writes']):
                mk = 'several writers'
        if isinstance(mk, str):
            reasons[mk] = reasons.get(mk, 0) + 1
            examples.setdefault(mk, []).append(p['names'][0]) if len(examples.get(mk, [])) < 6 else None
            continue
        if mk is None:
            continue
        info[k] = mk
    # A row reduction that row_fusion fuses with a full epilogue (e.g. x * rsqrt(mean(x * x))) or chains with further
    # stages stays with row_fusion (probe = row_fusion run on a copy of the phases); this pass has no full epilogues.
    if probe is not None:
        try:
            fused_records = probe(phases)
        except Exception:     # the probe is advisory: without it, keep every row reduction with row_fusion
            fused_records = None
        if fused_records is None:
            blocked = {phases[k]['names'][0] for k, mk in info.items() if mk[0] == 'rowred'}
        else:
            blocked = {r.get('reduction') for r in fused_records
                       if r.get('full_epilogue_phases') or r.get('chained_stages')}
            blocked |= {n for r in fused_records for n in (r.get('chained_stages') or ())}
        for k in [k for k, mk in info.items() if mk[0] == 'rowred' and phases[k]['names'][0] in blocked]:
            del info[k]
            reasons['row_fusion full epilogue'] = reasons.get('row_fusion full epilogue', 0) + 1
    kinds_count = {}
    for mk in info.values():
        kinds_count[mk[0]] = kinds_count.get(mk[0], 0) + 1

    def tw(k):
        mk = info[k]
        return (mk[1], mk[3]) if mk[0] in ('col', 'slot') else (mk[1], mk[2]) if mk[0] == 'rowred' else (mk[1], None)

    def col_shape(k):
        return tuple(int(d) for d in phases[k]['shape'])

    # value kind of a candidate's outputs: 'col' (slots K) or 'row'
    def value_kind(k):
        mk = info[k]
        if mk[0] == 'col':
            return 'col', mk[2]
        if mk[0] == 'slot':
            return 'col', 1
        return 'row', 1

    def compatible(q, n):
        """Read of name n (written by candidate p) by candidate q can be satisfied from registers."""
        y = root(n)
        p = writer[y][0]
        if p not in info or p == q:
            return False
        (Tq, Wq), (Tp, Wp) = tw(q), tw(p)
        if Tq != Tp:
            return False
        vk, Kv = value_kind(p)
        lay, rl = layouts.get(n), layouts.get(y)
        if lay is None or rl is None:
            return False
        mq = info[q]
        if mq[0] == 'col':
            if vk != 'col' or Wq != Wp:
                return False
            return _slot_map(lay, rl, col_shape(q), mq[1], mq[2], mq[3]) is not None
        if mq[0] == 'slot':
            if vk != 'col' or Wq != Wp or n != phases[q]['slot_reduction']['source']:
                return False
            return _slot_map(lay, rl, tuple(lay.shape), mq[1], mq[2], mq[3]) is not None
        if mq[0] == 'rowred':
            if vk != 'col' or Kv != 1 or Wq != Wp or n != phases[q]['row_reduction']['source']:
                return False
            return _slot_map(lay, rl, (mq[1], mq[2]), mq[1], 1, mq[2]) == [0]
        # row epilogue: identity read of a row value
        if mq[0] != 'row':
            return False
        return vk == 'row' and _numel(lay.shape) == Tq and lay.root == y and not lay.offset and \
            tuple(lay.strides) == _contiguous(lay.shape)

    original = _topological_dependencies(phases, root)
    order = _dependency_order(original)
    # Greedy topological grouping. taint[k] = groups reachable to k through a path that leaves the group (k is not
    # a member): k may join group g only if no parent outside g is tainted by g (no member -> non-member -> member
    # path, so contracting g keeps the DAG acyclic). Every admission check is made when a phase joins.
    group_of, members_of, taint, cost = {}, {}, {}, {}
    dropped = {}

    def drop(why):
        dropped[why] = dropped.get(why, 0) + 1

    def lines_of(k):
        mk = info[k]
        if mk[0] == 'col':
            return len(phases[k]['body']) * mk[2]
        return len(phases[k].get('body') or ()) or 1

    def admissible(q, g):
        mq = info[q]
        produced = {root(w) for m in members_of[g] for w in phases[m]['writes']}
        body = '\n'.join(phases[q].get('body') or ())
        shape = tuple(int(d) for d in phases[q].get('shape') or ())
        T0 = tw(members_of[g][0])[0]
        if tw(q)[0] != T0:
            return 'rows differ'
        Ws = {tw(m)[1] for m in members_of[g] if tw(m)[1] is not None}
        if tw(q)[1] is not None and Ws and tw(q)[1] not in Ws:
            return 'width differs'
        for n in phases[q]['reads']:
            if root(n) not in produced:
                continue
            if not compatible(q, n):
                return 'incompatible group read'
            if mq[0] in ('col', 'row'):
                try:
                    text = f'{ident(n)}[{index(layouts[n], shape, "i")}]'
                except Exception:
                    text = None
                if text is None or text not in body:
                    return 'group read text not found'
        if mq[0] == 'rowred' and root(phases[q]['row_reduction']['source']) not in produced:
            return 'row reduction source outside'
        if mq[0] == 'slot' and root(phases[q]['slot_reduction']['source']) not in produced:
            return 'slot reduction source outside'
        if cost[g] + lines_of(q) > MAX_GROUP_EXPRESSIONS:
            return 'code-size guard'
        return None

    for k in order:
        parents = original[k]
        t = set()
        for p_ in parents:
            t |= taint.get(p_, set())
            if p_ in group_of:
                t.add(group_of[p_])
        joined = None
        if k in info:
            guard = any(
                r not in info and root((phases[r].get('row_reduction') or phases[r].get('row_softmax') or {})
                                       .get('source', '')) == root(w)
                for w in phases[k]['writes'] for r in readers.get(root(w), ()))
            if guard:
                drop('outside row reduction prologue')
            else:
                cands = []
                for n in phases[k]['reads']:
                    y = root(n)
                    for p_ in writer.get(y, ()):
                        g = group_of.get(p_)
                        if g is not None and g not in cands:
                            cands.append(g)
                for g in cands:
                    if any(group_of.get(p_) != g and g in taint.get(p_, ()) for p_ in parents):
                        drop('dependency cycle')
                        continue
                    why = admissible(k, g)
                    if why:
                        drop(why)
                        continue
                    joined = g
                    break
                if joined is None and info[k][0] in ('col', 'slot'):
                    joined = len(members_of)          # a column/slot phase may start a group
                    members_of[joined], cost[joined] = [], 0
                if joined is not None:
                    group_of[k] = joined
                    members_of[joined].append(k)
                    cost[joined] += lines_of(k)
        if k in group_of:
            t.discard(group_of[k])
        taint[k] = t
    records = [dict(diagnostics=dict(candidates=kinds_count, refused=reasons, examples=examples, dropped=dropped))]
    groups = []
    for g, members in members_of.items():
        members = sorted(members)
        if len(members) < 2:
            continue
        produced = {root(w): m for m in members for w in phases[m]['writes']}
        internal = sum(1 for q in members for n in phases[q]['reads'] if root(n) in produced and produced[root(n)] != q)
        if not internal:
            continue
        if not any(info[m][0] == 'slot' or (info[m][0] == 'col' and info[m][2] > 1) for m in members):
            drop('no slot-axis work (left to row_fusion)')
            continue
        T = tw(members[0])[0]
        W = next(tw(m)[1] for m in members if tw(m)[1] is not None)
        groups.append((members, T, W))
    records[0]['diagnostics']['groups'] = len(groups)

    if not groups:
        return phases, records, set()

    fused, elided_all, member_of = {}, set(), {}
    inlined_sum1 = {}      # size-one sum value -> group members that now evaluate its expression
    for g, (members, T, W) in enumerate(groups):
        try:
            phase, stored, elided, expressions = _emit(phases, layouts, dtypes, members, info, T, W, ident, index,
                                                       kept, readers, root, _butterfly0, sum1, inlined_sum1)
        except _Refuse as e:
            records.append(dict(refused=str(e), members=[phases[m]['names'][0] for m in members]))
            continue
        fused[g] = phase
        elided_all |= elided
        for m in members:
            member_of[m] = g
        records.append(dict(rows=T, width=W, members=len(members),
                            member_kinds=[info[m][0] for m in members],
                            names=[n for m in members for n in phases[m]['names']],
                            stored=sorted(stored), elided=sorted(elided), expressions=expressions,
                            arithmetic='same per-operation expressions; slot and row reductions replay their exact trees'))
    if not fused:
        return phases, records, set()
    for v, ms in inlined_sum1.items():
        k = sum1[v]
        if v not in kept and set(readers.get(v, ())) <= ms and k not in member_of:
            member_of[k] = None           # removed: every reader evaluates its expression
            elided_all.add(v)
            records[0]['diagnostics'].setdefault('inlined_size_one_sums', []).append(v)
    result, placed, owners = [], {}, []
    removed = [k for k, g in member_of.items() if g is None]
    if removed:
        # a removed size-one sum: its readers' new reads of its operands keep the original producer edges
        phases_ = [p for k, p in enumerate(phases) if k not in removed]
        remap = {}
        j = 0
        for k in range(len(phases)):
            if k in removed:
                continue
            remap[k] = j
            j += 1
        member_of = {remap[k]: g for k, g in member_of.items() if g is not None}
        phases = phases_
    for k, phase in enumerate(phases):
        g = member_of.get(k)
        if g is None:
            owners.append(len(result))
            result.append(phase)
        else:
            if g not in placed:
                placed[g] = len(result)
                result.append(fused[g])
            owners.append(placed[g])
    try:
        deps = _fusion_dependencies(phases, result, owners, root)
        order = _dependency_order(deps)
    except ValueError as e:
        return phases, records + [dict(refused='contraction: ' + str(e))], set()
    return [result[k] for k in order], records, elided_all


class _Refuse(Exception):
    pass


def _emit(phases, layouts, dtypes, members, info, T, W, ident, index, kept, readers, root, butterfly0, sum1, inlined_sum1):
    mset = set(members)
    writers_of = {}
    for k_, p_ in enumerate(phases):
        for w_ in p_['writes']:
            writers_of.setdefault(root(w_), []).append(k_)
    produced = {}
    for m in members:
        for w in phases[m]['writes']:
            produced[root(w)] = m
    stored = {y for y in produced if y in kept or any(r not in mset for r in readers.get(y, ()))}
    elided = set(produced) - stored
    ktypes = {}
    for y, m in produced.items():
        mk = info[m]
        ktypes[y] = ('col', mk[2]) if mk[0] == 'col' else ('col', 1) if mk[0] == 'slot' else ('row', 1)
    col_lines, row_lines, decl = [], [], []
    slots_used = set()
    ident_names = {ident(n): n for n in layouts}
    expressions = 0

    def reg(y, s=None):
        return f'tp_{y}_{s}' if s is not None else f'tp_{y}'

    extra_reads, local_inlined, local_kept = [], set(), set()

    def substitute_reads(expr, q, shape, K, s):
        """Replace reads of group values in a rendered expression by registers (slot s of member q)."""
        for n in sorted(set(phases[q]['reads']), key=len, reverse=True):
            y = root(n)
            if y in sum1 and y not in produced and n == y:
                # accumulation_rules 'rsum1' at this reader: the one-leaf sum's own expression at the read element
                # (what lower.py's size-one-sum pass does for elementwise readers); operands must be final already.
                p1 = phases[sum1[y]]
                ops = {root(r) for r in p1['value_refs']}
                if all(w < sum1[y] for o_ in ops for w in writers_of.get(o_, ())):
                    text = f'{ident(n)}[{index(layouts[n], shape, "i")}]'
                    if text in expr:
                        idx = text[len(ident(n)) + 1:-1]
                        expr = expr.replace(text, '(' + re.sub(r'\bo\b', f'({idx})', p1['value_expr']) + ')')
                        local_inlined.add(y)
                        extra_reads.extend(p1['value_refs'])
                else:
                    local_kept.add(y)
                continue
            if y not in produced or produced[y] == q:
                continue
            text = f'{ident(n)}[{index(layouts[n], shape, "i")}]'
            if text not in expr:
                continue
            kind, Ky = ktypes[y]
            if info[q][0] == 'row':
                expr = expr.replace(text, reg(y))
                continue
            if kind != 'col':
                raise _Refuse('column member reads a row value')
            smap = _slot_map(layouts[n], layouts[y], shape, T, K, W)
            if smap is None:
                raise _Refuse('non column-local read')
            expr = expr.replace(text, reg(y, smap[s]))
        return expr

    for q in members:
        mk = info[q]
        p = phases[q]
        if mk[0] in ('col', 'row'):
            shape = tuple(int(d) for d in p['shape'])
            K = mk[2] if mk[0] == 'col' else 1
            own = sorted(p['names'], key=len, reverse=True)
            whole = '\n'.join(p['body'])
            for n in set(p['reads']):
                if root(n) in produced and produced[root(n)] != q and \
                        f'{ident(n)}[{index(layouts[n], shape, "i")}]' not in whole:
                    raise _Refuse(f'read text of {n} not found in {p["names"][0]}')
            for s in range(K):
                var = f'tp_ik{K}s{s}' if mk[0] == 'col' else 'o'
                if mk[0] == 'col':
                    slots_used.add(K)
                for line in p['body']:
                    ctype, name, expr, has_store = _parse(line)
                    expr = substitute_reads(expr, q, shape, K, s)
                    for n in own:
                        expr = re.sub(rf'\br_{re.escape(n)}\b', reg(n, s) if mk[0] == 'col' else reg(n), expr)
                    if _R_REF.search(expr):
                        raise _Refuse('unresolved register in ' + name)
                    for v in _V_READ.findall(expr):
                        if root(v) in produced:
                            raise _Refuse('group value still read from memory in ' + name)
                    expr = re.sub(r'\bi\b', f'({var})', expr)
                    target = reg(name, s) if mk[0] == 'col' else reg(name)
                    if root(name) in stored and not has_store:
                        raise _Refuse('kept value without a store line: ' + name)
                    if mk[0] == 'col':
                        col_lines.append(('decl', ctype, target, expr,
                                          f'{ident(name)}[{var}]' if root(name) in stored else None))
                    else:
                        text = f'const {ctype} {target} = {expr};'
                        if root(name) in stored:
                            text += f' if (lane == 0) {ident(name)}[o] = {target};'
                        row_lines.append('  ' + text)
                    expressions += 1
        elif mk[0] == 'slot':
            sr = p['slot_reduction']
            src, out = sr['source'], p['names'][0]
            K = mk[2]
            lay = layouts[src]
            y = root(src)
            leaves = []
            for r in range(K):
                if y in produced:
                    smap = _slot_map(lay, layouts[y], tuple(lay.shape), T, K, W)
                    if smap is None:
                        raise _Refuse('non column-local slot reduction source')
                    leaf = reg(y, smap[r])
                    if dtypes.get(src) == 'bfloat16':
                        leaf = f'__bfloat162float({leaf})'
                else:
                    slots_used.add(K)
                    leaf = f'{ident(src)}[{index(lay, tuple(lay.shape), f"tp_ik{K}s{r}")}]'
                    if dtypes.get(src) == 'bfloat16':
                        leaf = f'__bfloat162float({leaf})'
                leaves.append(f'(0.0f + {leaf})')
            tree = butterfly0(leaves)
            v = reg(root(out), 0)
            col_lines.append(('decl', 'float', f'tp_acc_{root(out)}', tree, None))
            col_lines.append(('decl', 'float', v, f'tp_acc_{root(out)}{sr["scale"]}',
                              f'{ident(out)}[(o * {W}LL + tp_c)]' if root(out) in stored else None))
            expressions += 1
        else:   # rowred
            rr = p['row_reduction']
            src, out = rr['source'], p['names'][0]
            y = root(src)
            leaf = reg(y, 0)
            acc = f'tp_v_{root(out)}'
            decl.append(f'  float {acc}[8];')
            col_lines.append(('acc', acc, leaf))
            second = butterfly0([f'{acc}[{w}]' for w in range(8)])
            text = (f'#pragma unroll\n  for (int w = 0; w < 8; ++w) {acc}[w] = s101h::warp_sum({acc}[w]);\n'
                    f'  const float tp_second_{root(out)} = {second}; const float {reg(root(out))} = '
                    f'tp_second_{root(out)}{rr["scale"]};')
            if root(out) in stored:
                text += f' if (lane == 0) {ident(out)}[o] = {reg(root(out))};'
            row_lines.append('  ' + text)
            expressions += 1
    if expressions > MAX_GROUP_EXPRESSIONS:
        raise _Refuse(f'{expressions} expressions exceed the code-size guard')
    body = _assemble(col_lines, slots_used, W, dtypes, ident_names)
    code = '\n'.join([
        f'for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o < {T}LL; '
        f'o += ((long long)gridDim.x * blockDim.x) >> 5) {{',
        '  const int lane = threadIdx.x & 31;',
        *decl,
        *body,
        *row_lines,
        '}'])
    names = [n for m in members for n in phases[m]['names']]
    # a size-one sum still read from memory anywhere in the group (text not found) keeps its read
    gone = set()
    for y in local_inlined:
        if y in local_kept or f'{ident(y)}[' in code:
            continue
        gone.add(y)
        inlined_sum1.setdefault(y, set()).update(m for m in members if any(root(r) == y for r in phases[m]['reads']))
    reads = list(dict.fromkeys([r for m in members for r in phases[m]['reads']
                                if root(r) not in produced and root(r) not in gone] + extra_reads))
    targets = [t for m in members for t in phases[m]['targets']]
    phase = {'kind': 'slot_token_rows', 'names': names, 'targets': targets, 'reads': reads, 'writes': names,
             'code': code, 'threads': 256, 'work_items': (T + 7) // 8, 'shared_bytes': 0,
             'engine': 'slot_token_rows', 'reduction_elements': W,
             'slot_rows': {'rows': T, 'width': W, 'members': len(members), 'stored': sorted(stored),
                          'elided': sorted(elided)},
             'arithmetic': 'per element the members\' rendered expressions; exact slot/row reduction trees'}
    return phase, stored, elided, expressions


_CTYPE = {'float32': 'float', 'bfloat16': '__nv_bfloat16', 'int64': 'long long', 'int32': 'int'}
HOIST_BUDGET = 96          # load registers per lane per column chunk


def _loads(expr):
    """Top-level array reads ``v_x[...]`` of an expression (bracket matched), in order."""
    out, i = [], 0
    while True:
        m = re.compile(r'\bv_\w+\[').search(expr, i)
        if not m:
            return out
        depth, j = 0, m.end() - 1
        while j < len(expr):
            if expr[j] == '[':
                depth += 1
            elif expr[j] == ']':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        out.append(expr[m.start():j + 1])
        i = j + 1


def _assemble(items, slots_used, W, dtypes, ident_names):
    """Unrolled column body in chunks: every chunk issues all its operand loads, then evaluates every member in
    program order (row reduction leaves in ascending column order), then stores. Loads only read values the group
    does not write (group values are registers), so issuing them before the chunk's stores reads the same values."""
    nw = W // 32
    suffix = lambda text, w: re.sub(r'\b(tp_(?!v_)\w+)\b', lambda m: f'{m.group(1)}_w{w}', text)

    def column(w):
        loads, comp, stores = {}, [], []
        for it in items:
            if it[0] == 'acc':
                _, acc, leaf = it
                leaf = suffix(leaf, w)
                comp.append(f'{acc}[{w}] = 0.0f + {leaf};' if w < 8 else f'{acc}[{w % 8}] = {acc}[{w % 8}] + {leaf};')
                continue
            _, ctype, target, expr, lhs = it
            expr = suffix(expr, w)
            for text in _loads(expr):
                arr = text[:text.index('[')]
                name = ident_names.get(arr)
                ct = _CTYPE.get(dtypes.get(name)) if name else None
                if ct is None:
                    continue
                if text not in loads:
                    loads[text] = (f'tp_L{len(loads)}_w{w}', ct)
                expr = expr.replace(text, loads[text][0])
            comp.append(f'const {ctype} {suffix(target, w)} = {expr};')
            if lhs:
                stores.append(f'{suffix(lhs, w)} = {suffix(target, w)};')
        index = [f'const long long tp_c_w{w} = {32 * w}LL + lane;'] + [
            f'const long long tp_ik{K}s{s}_w{w} = (o * {K}LL + {s}LL) * {W}LL + tp_c_w{w};'
            for K in sorted(slots_used) for s in range(K)]
        loads_ = [f'const {ct} {r} = {t};' for t, (r, ct) in loads.items()]
        return index, loads_, comp, stores, len(loads)
    cols = [column(w) for w in range(nw)]
    per = max(1, max(c[4] for c in cols))
    chunk = max(1, min(nw, HOIST_BUDGET // per))
    out = []
    for c0 in range(0, nw, chunk):
        part = cols[c0:c0 + chunk]
        out.append('  {')
        for key in range(4):
            for c in part:
                out.extend('    ' + line for line in c[key])
        out.append('  }')
    return out
