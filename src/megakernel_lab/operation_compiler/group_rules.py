"""P-OPS5-B operation grouping rules for staged regions (BF16 programs).

Two rules, both dispatched only on phase kind, dtype, shape, layout and dependencies:

* ``split_components``: an elementwise phase keeps one grid loop per independent
  register chain. Grouping by emission order used to place unrelated values (for
  example a weight cast next to an activation cast) in one loop, which delays the
  earlier-ready value and makes the loop read two products. Splitting restores the
  natural dependency level of each chain. Every line is unchanged.

* ``fuse_bf16_epilogues``: same-index elementwise consumers of a BF16 product
  (P-OPS-C ``popsc`` body, one product per phase, one batch, plain row-major
  result) are evaluated in the product's register-direct fragment store. The
  product value passed to the consumer is exactly the stored BF16 value
  (``cvt.rn.bf16x2`` of the FP32 accumulator), so every consumer expression sees
  the same operand bits as when it re-read the product from memory. Consumer
  stores whose values nothing else reads are elided, and the product store
  itself is skipped when only absorbed consumers read it. This removes the
  consumer's grid phase, its barrier, and the product write/read round trip.

Admission follows dependencies only: consumers must read the product (and any
earlier absorbed value) element-for-element in their own index space, and every
other operand must be produced strictly before the product's dependency level,
so no consumer of an outside value moves earlier than its producer.
"""
from __future__ import annotations

import math
import os
import re

_PRESETS = {True: {'rot'}, 'rot': {'rot'}, 'norot': set(), 'defer': {'rot', 'defer'},
            'chains': {'rot', 'chains'}, 'prologue': {'rot', 'chains', 'prologue'},
            'prologue_defer': {'rot', 'chains', 'prologue', 'defer'},
            'joint': {'rot', 'chains', 'prologue', 'defer', 'joint'},
            'all': {'rot', 'chains', 'prologue', 'defer', 'joint', 'softmax'}}
FEATURES = ('lsink', 'rot', 'defer', 'chains', 'prologue', 'joint', 'jointbig', 'softmax', 'place', 'alap', 'ilp', 'wrow', 'wrow2', 'part', 'jointlight', 'wsoft', 'deepred', 'ilp8', 'alapv', 'pre', 'rngel', 'vfill', 'vec4', 'rowepi', 'splitk', 'cluster', 'wide', 'idx32', 'rhoist', 'ring', 'slicek', 'slicek2', 'rsink', 'epilate', 'rclamp', 'nosync', 'epipair', 'epiearly', 'epism', 'epismh', 'idxid', 'vec4u', 'colsum', 'shk', 'lpt', 'lptl', 'epu1', 'epu2', 'l2pf', 'zmerge', 'lptx', 'partc', 'msum',
            # P7-C
            'p5c', 'p5cw', 'p5ce', 'powx', 'p5ck', 'zelide', 'sink', 'sinki', 'pelem', 'colsplit', 'pred',
            # P8-C
            'vecbf', 'vprod', 'vprodx', 'p5cv', 'splitk16', 'colreg',
            # release step 14
            'tmae', 'tmaj', 'ecta', 'emerge', 'w96', 'tdeep',
            # step-14 speed
            'ectaw', 'ectac', 'tla', 'teu4', 'teu8', 'tper', 'lptr', 'lsreg', 'ctrace', 'tepf', 'ectai', 'ectaq', 'btrace', 'dfast', 'tmpf', 'xdiv', 'fs6', 'fs8', 'tepr', 'tmaq', 'tfirst', 'rpipe', 'lsplit', 'pstream', 'tsplit',
            # LATCH-TINYMOE-25 (ops.update_schedule)
            'csink', 'fsbatch',
            # LATCH-TINYMOE-FUSE (ops.norm_fusion)
            'gatedbw',
            # TINYMOE-SELECTIVE-CADENCE (ops.update_schedule streamed_ustate / streamed_ustate_late)
            'ustate', 'ulate',
            # LATCH-SELECTIVE-10 (ops.accumulation_rules)
            'fsgather', 'fswarp', 'cmulti', 'rowchain', 'fsmix', 'fsfence', 'vec8', 'vec4m', 'tsplitx',
            # LATCH-TRAIN-SYNC (ops.accumulation_rules)
            'fsmixl', 'fsload', 'fscast', 'fsmix8', 'fsspread')


def features_of(value):
    """Rule set of a ``group_rules`` option: a preset name or '+'-joined features (base rules always on)."""
    if not value:
        return set()
    if value in _PRESETS:
        return set(_PRESETS[value])
    chosen = {part for part in str(value).split('+') if part and part != 'base'}
    if chosen - set(FEATURES):
        raise ValueError(f'unknown group rule features {sorted(chosen - set(FEATURES))}')
    return chosen


_LINE = re.compile(r'const (\w+) r_(\w+) = ')
_MEMORY = re.compile(r'\bv_(\w+)\[')
_REGISTER = re.compile(r'\br_(\w+)\b')


def _contiguous(shape):
    result, stride = [], 1
    for d in reversed(shape):
        result.append(stride)
        stride *= d
    return tuple(reversed(result))


def _identity(layout, shape):
    return (math.prod(layout.shape) == math.prod(shape) and layout.offset == 0
            and tuple(layout.shape) == tuple(shape)
            and all(d == 1 or s == c for d, s, c in zip(layout.shape, layout.strides, _contiguous(shape))))


def _store_free(line):
    """The line without its trailing store(s) (``; v_x[i] = ...;``)."""
    head = line.split('; ', 1)
    return head[0]


# Ported from frozen ROW donor c2a325e441f1c614b7cec4ec4ed94d43ec4e1895.
# LATCH-MOESUP-ROW: inside one connected register chain, lines whose own operands are ready earlier than the rest of
# the chain (e.g. an autograd scale w/(2(1+w))/count sharing w-only subexpressions with the loss it scales) are split
# into an earlier piece; the later piece reads their stored value (``v_x[i]``, the register's exact value) instead of
# the register. Without the split the whole chain waits for its latest operand and delays every reader of the early
# lines (there: every auxiliary readout gradient waited for the main readout's loss). Lines are unchanged.
LEVEL_SPLIT = True
# Only small chains (e.g. scalar loss/gradient-factor arithmetic): splitting a large chain stores and re-reads its
# early values and measured slower (plain update 58.97 -> 62.30 ms with every chain split, `match-r12`).
LEVEL_SPLIT_MAX_NUMEL = 65536
_STORE_OF = re.compile(r'; v_(\w+)\[i\] = r_(\w+);$')


def _level_split(piece, level_of, written_all, dtypes):
    """[piece] or level-ordered sub-pieces of one elementwise piece (explicit lsplit scheduling rule)."""
    names, body = list(piece['names']), list(piece['body'])
    index = {n: k for k, n in enumerate(names)}
    lv = []
    for k, line in enumerate(body):
        m = _LINE.match(line)
        expression = _store_free(line)[m.end():]
        # Only register-carried dependencies have the exact same-element
        # proof needed here. An internal memory view may have another index.
        if any(r in written_all for r in _MEMORY.findall(expression)):
            return [piece]
        if any(r in index and index[r] >= k for r in _REGISTER.findall(expression)):
            return [piece]
        ext = max((level_of(r) for r in _MEMORY.findall(expression) if r not in written_all), default=0)
        dep = max((lv[index[r]] for r in _REGISTER.findall(expression) if r in index and index[r] < k), default=0)
        lv.append(max(ext, dep))
    if len(set(lv)) < 2:
        return [piece]
    groups = {}
    for k, level in enumerate(lv):
        groups.setdefault(level, []).append(k)
    out = []
    for level in sorted(groups):
        members = groups[level]
        lines, extra = [], []
        for k in members:
            line = body[k]
            head, sep, tail = line.partition('; ')
            m = _LINE.match(head)
            prefix, expression = head[:m.end()], head[m.end():]
            for r in sorted(set(_REGISTER.findall(expression)), key=len, reverse=True):
                if r in index and lv[index[r]] != level:
                    store = _STORE_OF.search(body[index[r]])
                    if not store or store.group(1) != r or store.group(2) != r:
                        return [piece]          # the earlier value is not stored at the element index
                    expression = re.sub(rf'\br_{r}\b', f'v_{r}[i]', expression)
                    extra.append(r)
            lines.append(prefix + expression + sep + tail)
        member_names = [names[k] for k in members]
        text = ' '.join(_store_free(line)[_LINE.match(line).end():] for line in lines)
        reads = [r for r in piece['reads'] if f'v_{r}[' in text or (r in member_names and f'r_{r}' in text)]
        reads += [r for r in dict.fromkeys(extra) if r not in reads]
        sub = dict(piece)
        sub.update(names=member_names, targets=[piece['targets'][k] for k in members], reads=reads,
                   writes=member_names, body=lines, dtype=dtypes[member_names[0]],
                   registers={n: r for n, r in piece.get('registers', {}).items() if n in member_names})
        out.append(sub)
    return out


def split_components(phases, dtypes, *, ctypes, layouts=None, level_split=False, gated=False):
    """Split elementwise phases into independent register chains. Returns (phases, records).

    Chains stay together unless they differ in dependency level or in which matrix
    products they read: splitting is only useful when one chain would otherwise wait
    for another, or when a product's consumers must be separable for its epilogue.

    gated (LATCH-TINYMOE-FUSE ``gatedbw``): in a phase holding a gated activation's backward (a sigmoid or silu
    member), chains are keyed by their newest products only (the ones they wait for). The two cotangent chains of
    ``silu(u) * g`` both wait for the same new ``dy`` and read the same older ``u``; keyed by every product read
    they would be split and each would load ``dy`` and ``u`` separately. Lines are unchanged.
    """
    root = (lambda name: layouts[name].root if name in layouts else name) if layouts else (lambda name: name)
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    result, records = [], []
    for phase in phases:
        if phase.get('kind') != 'elementwise' or len(phase.get('names', ())) < 2:
            result.append(phase)
            continue
        names = list(phase['names'])
        body = list(phase['body'])
        if len(body) != len(names):
            result.append(phase)
            continue
        index = {n: k for k, n in enumerate(names)}
        parent = list(range(len(names)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ok = True
        for k, line in enumerate(body):
            match = _LINE.match(line)
            if not match or match.group(2) != names[k]:
                ok = False
                break
            expression = _store_free(line)[match.end():]
            used = set(_REGISTER.findall(expression)) | set(_MEMORY.findall(expression))
            for other in used:
                if other in index and index[other] != k:
                    parent[find(k)] = find(index[other])
        if not ok:
            result.append(phase)
            continue
        components = {}
        for k in range(len(names)):
            components.setdefault(find(k), []).append(k)
        if len(components) == 1:
            result.append(phase)
            continue
        written = set(names)
        keyed = {}
        gated_phase = gated and any(str(t).startswith(('aten.sigmoid.', 'aten.silu.')) for t in phase.get('targets', ()))
        for members in sorted(components.values()):
            lines = [body[k] for k in members]
            text = ' '.join(_store_free(line)[_LINE.match(line).end():] for line in lines)
            reads = [r for r in phase['reads'] if f'v_{r}[' in text or (r in written and f'r_{r}' in text)]
            outside = [writer[root(r)] for r in reads if root(r) in writer and root(r) not in written]
            products = {root(r) for r in reads if root(r) in writer and root(r) not in written
                        and phases[writer[root(r)]].get('kind') == 'gemm'}
            if gated_phase and products:
                newest = max(level[writer[p]] for p in products)
                products = {p for p in products if level[writer[p]] == newest}
            key = (1 + max((level[q] for q in outside), default=-1), tuple(sorted(products)))
            keyed.setdefault(key, []).extend(members)
        if len(keyed) == 1:
            result.append(phase)
            continue
        groups = {min(m): sorted(m) for m in keyed.values()}
        for _, members in sorted(groups.items()):
            member_names = [names[k] for k in members]
            lines = [body[k] for k in members]
            text = ' '.join(_store_free(line)[_LINE.match(line).end():] for line in lines)
            reads = [r for r in phase['reads'] if f'v_{r}[' in text or (r in written and f'r_{r}' in text)]
            piece = dict(phase)
            piece.update(names=member_names, targets=[phase['targets'][k] for k in members],
                         reads=reads, writes=member_names, body=lines,
                         dtype=dtypes[member_names[0]],
                         registers={n: r for n, r in phase.get('registers', {}).items() if n in member_names})
            result.append(piece)
        records.append({'names': names, 'components': [[names[k] for k in m] for _, m in sorted(groups.items())]})
    if level_split and LEVEL_SPLIT:
        def level_of(r):
            return 1 + level[writer[root(r)]] if root(r) in writer else 0
        split = []
        for phase in result:
            if (phase.get('kind') != 'elementwise' or len(phase.get('names', ())) < 2
                    or len(phase.get('body', ())) != len(phase['names'])
                    or math.prod(phase.get('shape') or ()) > LEVEL_SPLIT_MAX_NUMEL
                    or not all(_LINE.match(line) and _LINE.match(line).group(2) == name
                               for line, name in zip(phase['body'], phase['names']))):
                split.append(phase)
                continue
            pieces = _level_split(phase, level_of, set(phase['names']), dtypes)
            if len(pieces) > 1:
                records.append({'names': list(phase['names']), 'level_split': [p['names'] for p in pieces]})
            split.extend(pieces)
        result = split
    return result, records


def _topological(phases, root):
    import heapq
    produced_by = {}
    for k, phase in enumerate(phases):
        for w in phase['writes']:
            produced_by[root(w)] = k
    deps = [{produced_by[root(r)] for r in phase['reads'] if root(r) in produced_by and produced_by[root(r)] != k}
            for k, phase in enumerate(phases)]
    # The old scan selects the smallest currently ready original index. A
    # min-heap makes exactly that selection without rescanning the whole graph
    # once for every emitted phase.
    pending = [len(parents) for parents in deps]
    following = [[] for _ in phases]
    for child, parents in enumerate(deps):
        for parent in parents:
            following[parent].append(child)
    ready = [k for k, count in enumerate(pending) if count == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        for child in following[k]:
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, child)
    if len(order) != len(phases):
        raise ValueError('group rules produced a dependency cycle')
    return [phases[k] for k in order]


def _popsc_single(phase):
    spec = phase.get('matrix_spec') or {}
    code = phase.get('code') or ''
    return (phase.get('kind') == 'gemm' and spec.get('dtype') == 'bfloat16' and int(spec.get('batch', 1)) == 1
            and str(phase.get('engine', '')).startswith('popsc_bf16_') and code.count('popsc::bmm_phase') == 1
            and len(phase.get('members', [None])) == 1)


def _popsc_batched(phase, layouts, out):
    """P9-C: a batched BF16 small-tile product (one bmm_phase call) with a contiguous [batch, M, N] result: its
    epilogue's flat index includes the batch offset (kernels/popsc_mm.cuh), so same-index consumers can attach."""
    spec = phase.get('matrix_spec') or {}
    code = phase.get('code') or ''
    if not (phase.get('kind') == 'gemm' and spec.get('dtype') == 'bfloat16' and int(spec.get('batch', 1)) > 1
            and str(phase.get('engine', '')).startswith('popsc_bf16_') and code.count('popsc::bmm_phase<') == 1
            and len(phase.get('members', [None])) == 1):
        return False
    lay = layouts[out]
    return len(lay.shape) == 3 and _identity(lay, lay.shape)


JOINT_BATCHED = os.environ.get('P9C_JOINTB', '0') != '0'   # P9-C: batched joint plans (r10: no measured effect; opt-in)


def _popsc_batch_call(phase):
    """P9-C: a batched BF16 small-tile product phase with a single bmm_phase call (rotated by its stage's vrot)."""
    spec = phase.get('matrix_spec') or {}
    code = phase.get('code') or ''
    return (phase.get('kind') == 'gemm' and spec.get('dtype') == 'bfloat16' and int(spec.get('batch', 1)) > 1
            and str(phase.get('engine', '')).startswith('popsc_bf16_') and code.count('popsc::bmm_phase<') == 1
            and len(phase.get('members', [None])) == 1)


def _call(phase):
    """Structured form of a single BF16 product call; parsed once from the P-OPS-C emission."""
    call = phase.get('popsc_call')
    if call is None:
        code = phase['code']
        start = code.index('popsc::bmm_phase<')
        lt = code.index('<', start)
        gt = code.index('>(', lt)
        close = code.rindex(');')
        args = code[gt + 2:close].split(', ')
        if len(args) != 6:
            raise ValueError('unexpected BF16 product emission: ' + code[start:close + 2])
        call = {'prefix': code[:start], 'suffix': code[close + 2:], 'targs': code[lt + 1:gt].split(','),
                'args': args, 'store': True, 'pro': None, 'k': None, 'epi': None}
        phase['popsc_call'] = call
    return call


def _render(phase):
    call = phase['popsc_call']
    bm, bn, wm, wn, ak, bkc, ks, nst = call['targs'][:8]
    store = str(call['store']).lower()
    if call.get('split') is not None:
        sp = call['split']
        head = f'popsc::bmm_phase_sk<{bm},{bn},{wm},{wn},{ak},{bkc},{ks},{nst},{sp["S"]},{store}>'
        args = [*call['args'], sp['part'], sp['cnt'], *([call['epi']] if call.get('epi') else [])]
        phase['code'] = call['prefix'] + head + '(' + ', '.join(args) + ');' + call['suffix']
        return
    if call.get('cluster') is not None:
        head = f'popsc::bmm_phase_cl<{bm},{bn},{wm},{wn},{ak},{bkc},{ks},{nst},{store}>'
        phase['code'] = (call['prefix'] + head + '(' + ', '.join(['g', 'smem', call['args'][5], call['cluster']])
                         + ');' + call['suffix'])
        return
    if call.get('rowepi') is not None:
        head = f'popsc::bmm_phase_re<{bm},{bn},{wm},{wn},{ak},{bkc},{ks},{nst},{store}>'
        args = [*call['args'], call['rowepi']]
        phase['code'] = call['prefix'] + head + '(' + ', '.join(args) + ');' + call['suffix']
        return
    if call.get('se'):
        head = f'popsc::bmm_phase_se<{bm},{bn},{wm},{wn},{ak},{bkc},{ks},{nst},{store},{call.get("se_u", 4)}>'
        args = [*call['args'], *([call['epi']] if call.get('epi') is not None else []),
                *([call['pre']] if call.get('pre') is not None else [])]
        phase['code'] = call['prefix'] + head + '(' + ', '.join(args) + ');' + call['suffix']
        return
    if call.get('kg') is not None:
        kg = call['kg']
        head = f'popsc::bmm_phase_kg<{bm},{bn},{kg["wm"]},{kg["wn"]},{kg["g"]},{ak},{bkc},{ks},{nst},{store}>'
        args = [*call['args'], *([call['epi']] if call.get('epi') is not None else [])]
        phase['code'] = call['prefix'] + head + '(' + ', '.join(args) + ');' + call['suffix']
        return
    if call['pro'] is not None:
        head = f'popsc::bmm_phase_rp<{bm},{bn},{wm},{wn},{bkc},{ks},{nst},{call["k"]},{store}>'
        args = [*call['args'], call['pro'], call['epi'] or 'popsc::NoEpi{}']
    elif call['epi'] is not None:
        flags = (f',{str(bool(call.get("late"))).lower()},true' if call.get('pair')
                 else ',true' if call.get('late') else '')
        head = f'popsc::bmm_phase<{bm},{bn},{wm},{wn},{ak},{bkc},{ks},{nst},{store}{flags}>'
        args = [*call['args'], call['epi'], *([call['pre']] if call.get('pre') else [])]
    else:
        head = f'popsc::bmm_phase<{",".join(call["targs"][:8])}>'
        args = list(call['args'])
    phase['code'] = call['prefix'] + head + '(' + ', '.join(args) + ');' + call['suffix']


# P9-C: the same warp-per-row header with its loop variable captured (row groups use o, softmax phases row).
_ROW_HEADER_VAR = re.compile(r'for \(long long (o|row) = \(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5; '
                             r'(?:o|row) < (\d+)LL; (?:o|row) \+= \(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \{')
ROWEPI_BATCHED = os.environ.get('P9C_REBATCH', '1') != '0'
ROWEPI_SOFTMAX = os.environ.get('P9C_RESOFT', '1') != '0'
_ROW_HEADER = re.compile(r'for \(long long o = \(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5; '
                         r'o < (\d+)LL; o \+= \(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \{')


def fuse_row_prologues(phases, layouts, dtypes, *, ident, kept, shared_limit=192 * 1024):
    """Products whose A operand is a row group's BF16 output recompute it in their A prologue.

    Admission (properties only): a single one-batch BF16 product with a k-contiguous, plain row-major
    [M, K] A operand written by an exact row group (``row_fused``) of M rows and width K, stored by
    that group's full-shape epilogue. The product then no longer waits for the row group: its tasks
    replay the group's per-row operations (identical expressions and reduction tree) for their own
    BM rows straight into shared memory. The row group still runs (in the same level, on other CTAs)
    for every value something else reads; the operand's own store is elided when only prologue
    products read it. Returns (phases, records, elided_roots).
    """
    root = lambda name: layouts[name].root if name in layouts else name
    writer = {}
    for k, phase in enumerate(phases):
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    records, plans = [], {}
    for g, phase in enumerate(phases):
        if not _popsc_single(phase):
            continue
        spec = phase['matrix_spec']
        a = spec['a']
        if not spec.get('ak') or a not in layouts or dtypes.get(a) != 'bfloat16':
            continue
        sam, sak = spec['strides'][0], spec['strides'][1]
        if (sam, sak) != (spec['k'], 1) or not _identity(layouts[a], (spec['m'], spec['k'])):
            continue
        r = writer.get(root(a))
        if r is None or phases[r]['kind'] != 'row_fused':
            continue
        info = phases[r].get('row_fusion') or {}
        if info.get('width') != spec['k'] or info.get('rows') != spec['m'] or 'softmax' in info:
            continue
        code = phases[r]['code']
        header = _ROW_HEADER.search(code)
        target = f'{ident(a)}[i] = r_{root(a)};'
        if not header or int(header.group(1)) != spec['m'] or code.count(target) != 1:
            continue
        call = _call(phase)
        bm, bn, wm, wn, ak, bkc, ks, nst = call['targs'][:8]
        bm, bn, ks, nst, k = int(bm), int(bn), int(ks), int(nst), int(spec['k'])
        b_el = bn * (ks + 8) if bkc == 'true' else ks * (bn + 8)
        shared = 2 * (bm * (k + 8) + nst * b_el)
        if shared > shared_limit or spec['m'] % bm:
            continue
        plans[g] = (r, a, bm, k, shared)
    if not plans:
        return phases, [], set()
    elided = set()
    for g, (r, a, bm, k, shared) in plans.items():
        phase = phases[g]
        width = phases[r]['row_fusion']['width']
        body = phases[r]['code']
        body = _ROW_HEADER.sub(f'#pragma unroll\nfor (long long o = m0_ + (threadIdx.x >> 5); o < m0_ + {bm}; o += 8) {{',
                               body, count=1)
        target = f'{ident(a)}[i] = r_{root(a)};'
        body = body.replace(target, f'sA_[(o - m0_) * {k + 8}LL + (i - o * {width}LL)] = r_{root(a)};')
        # Every other effect stays with the row group: drop global stores (values remain for local use).
        body = re.sub(r'(if \(lane == 0\) )?v_\w+\[[io]\] = (?:v_\w+\[[io]\] \+ )?r_\w+;', ';', body)
        call = _call(phase)
        call['pro'] = f'[=](long long m0_, __nv_bfloat16* sA_) {{\n{body}\n}}'
        call['k'] = k
        _render(phase)
        phase['reads'] = list(dict.fromkeys([*(x for x in phase['reads'] if root(x) != root(a)),
                                             *phases[r]['reads']]))
        phase['shared_bytes'] = max(int(phase.get('shared_bytes', 0)), shared)
        phase['row_prologue'] = {'row_group': phases[r]['names'][:4], 'operand': a, 'rows_per_task': bm,
                                 'width': width}
        records.append({'product': phase['names'][0], 'operand': a, 'row_group': phases[r]['names'],
                        'rows_per_task': bm, 'shared_bytes': shared,
                        'arithmetic': "row group's per-row expressions and exact reduction tree replayed per task"})
    # Operands read only by prologue products need no store in their row group.
    prologued = {}
    for g, (r, a, _, _, _) in plans.items():
        prologued.setdefault(root(a), set()).add(g)
    for y, products in prologued.items():
        r = writer[y]
        if y in kept or (readers.get(y, set()) - products):
            continue
        phases[r]['code'] = phases[r]['code'].replace(f'{ident(y)}[i] = r_{y};', ';')
        phases[r]['writes'] = [w for w in phases[r]['writes'] if root(w) != y]
        phases[r]['names'] = [n for n in phases[r]['names'] if n != y]
        elided.add(y)
    return phases, records, elided


def _prefetchable(lines):
    """The single outside read text of an epilogue body, or None (reads before stores; one distinct read)."""
    reads = set()
    for line in lines:
        head, _, tail = line.partition('; ')
        reads |= set(re.findall(r'v_\w+\[[^\]]*\]', head))
        reads |= set(re.findall(r'= (v_\w+\[i\]) \+ ', tail))
    return next(iter(reads)) if len(reads) == 1 else None


def _with_prefetch(lines, read):
    out = []
    for line in lines:
        head, sep, tail = line.partition('; ')
        out.append(head.replace(read, 'p_') + sep + tail.replace(f'= {read} + ', '= p_ + '))
    return out


def _late_reads(lines):
    """Every distinct outside read text of an epilogue body (heads, and accumulate tails ``= v_x[i] + ``)."""
    reads = []
    for line in lines:
        head, _, tail = line.partition('; ')
        for text in re.findall(r'v_\w+\[[^\]]*\]', head) + re.findall(r'= (v_\w+\[i\]) \+ ', tail):
            if text not in reads:
                reads.append(text)
    return reads


_PAIR_STORE = re.compile(r'^(v_\w+)\[i\] = (.*);$')


def _pair_epilogue(lines, register, types, prefetched):
    """P6-C epipair: one epilogue call per adjacent column pair (i even, i + 1). Every line is evaluated for both
    elements exactly as before (registers renamed per element, element b at index i + 1); each per-element store
    ``v_x[i] = EXPR`` becomes one 8-byte FP32 or 4-byte BF16 pair store of the two unchanged values."""
    rename = lambda text, e: re.sub(r'\br_(\w+)\b', lambda m: f'r_{m.group(1)}_{e}', text).replace('p_.', f'p{e}_.')
    body, stores = [], {}
    for e, index in (('a', 'i'), ('b', '(i + 1)')):
        for line in lines:
            head, sep, tail = line.partition('; ')
            head = re.sub(r'\bi\b', index, rename(head, e))
            if not sep or not tail.strip() or tail.strip() == ';':
                body.append(head + (';' if not head.rstrip().endswith(';') else ''))
                continue
            st = _PAIR_STORE.match(tail.strip())
            if not st or types.get(st.group(1)) not in ('float', '__nv_bfloat16'):
                return None
            body.append(head + ';')
            stores.setdefault(st.group(1), {})[e] = re.sub(r'\bi\b', index, rename(st.group(2), e))
    pack = []
    for ptr, values in stores.items():
        if set(values) != {'a', 'b'}:
            return None
        if types[ptr] == 'float':
            pack.append(f'*reinterpret_cast<float2*>({ptr} + i) = make_float2({values["a"]}, {values["b"]});')
        else:
            pack.append(f'*reinterpret_cast<__nv_bfloat162*>({ptr} + i) = __halves2bfloat162({values["a"]}, {values["b"]});')
    params = f'long long i, __nv_bfloat16 {register}_a, __nv_bfloat16 {register}_b'
    if prefetched:
        params += ', const auto pa_, const auto pb_'
    return f'[=]({params}) {{ ' + ' '.join(body + pack) + ' }'


BATCHED_EPILOGUES = True   # P9-C: batched small-tile products take same-index epilogues too


def _epilogue_cost(phase):
    # Reuse the flat reference's cost units: an ordinary loaded expression
    # costs one, a transcendental costs four. Whole register chains are kept
    # together; constants conservatively share the containing phase's cost.
    from .fusion_reference import TRANSCENDENTAL
    return sum(4 if str(target).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
               in TRANSCENDENTAL else int(bool(phase.get("reads"))) for target in phase.get("targets", ()))


def fuse_bf16_epilogues(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, ctypes, prefetch=False,
                        pair=False, cost_limit=None):
    """Return (phases, records, elided_roots)."""
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)

    absorbed, plans = {}, {}
    for g, phase in enumerate(phases):
        results = list(phase['writes'])
        if len(results) != 1 or dtypes.get(results[0]) != 'bfloat16':
            continue
        out = results[0]
        batched = BATCHED_EPILOGUES and os.environ.get('P9C_BEPI', '1') != '0' and _popsc_batched(phase, layouts, out)
        if (not (_popsc_single(phase) or batched) or (phase.get('popsc_call') or {}).get('rowepi')
                or (phase.get('popsc_call') or {}).get('cluster')):
            continue
        out_layout = layouts[out]
        if len(out_layout.shape) != (3 if batched else 2) or not _identity(out_layout, out_layout.shape):
            continue
        numel = math.prod(out_layout.shape)
        group = {out}
        members = []
        cost = 0
        # Candidates in program order: any later elementwise phase reading the product or an absorbed value.
        for e in range(g + 1, len(phases)):
            consumer = phases[e]
            if e in absorbed or consumer['kind'] != 'elementwise':
                continue
            candidate_cost = _epilogue_cost(consumer) if cost_limit is not None else 0
            if cost_limit is not None and cost + candidate_cost > cost_limit:
                continue
            shape = tuple(consumer['shape'])
            if math.prod(shape) != numel or any(root(n) in inplace_roots for n in consumer['names']):
                continue
            if any(dtypes.get(n) not in ('float32', 'bfloat16') for n in consumer['names']):
                continue
            own = {root(w) for w in consumer['writes']}
            if not any(root(r) in group for r in consumer['reads'] if root(r) not in own):
                continue
            ok = True
            for r in consumer['reads']:
                y = root(r)
                if y in own:
                    continue
                if y in group:
                    if not _identity(layouts[r], shape):
                        ok = False
                elif y in writer and level[writer[y]] >= level[g]:
                    ok = False
                elif batched and os.environ.get('P9C_BEPI_MODE') == 'noouter':
                    ok = False
            if not ok:
                continue
            members.append(e)
            cost += candidate_cost
            absorbed[e] = g
            group |= own
        if members:
            plans[g] = (out, members, group)
    if not plans:
        return phases, [], set()

    records, elided, merged = [], set(), {}
    for g, (out, members, group) in plans.items():
        phase = phases[g]
        register = f'r_{out}'
        names_in_group = sorted((n for n in layouts if root(n) in group), key=len, reverse=True)
        outside = lambda y: y in kept or any(q != g and q not in members for q in readers.get(y, ()))
        lines = []
        for e in members:
            consumer = phases[e]
            shape = tuple(consumer['shape'])
            for line in consumer['body']:
                for n in names_in_group:
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    line = line.replace(text, f'r_{root(n)}')
                match = _LINE.match(line)
                name = match.group(2) if match else None
                if name and not outside(root(name)):
                    line = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', line)
                    elided.add(name)
                lines.append(line)
        store = outside(out)
        new = dict(phase)
        call = dict(_call(phase))
        pf_ = prefetch
        if os.environ.get('P9C_BEPI_MODE') == 'nopre' and int((phase.get('matrix_spec') or {}).get('batch', 1)) > 1:
            pf_ = False
        read = _prefetchable(lines) if pf_ is True else None
        late = _late_reads(lines) if pf_ in ('late', 'early') else []
        types = {ident(n): ctypes.get(dtypes.get(n)) for n in layouts} if late else {}
        if late and len(late) <= 4 and all(types.get(t.split('[', 1)[0]) for t in late):
            # P6-C 'epilate': every outside operand of the epilogue is loaded for all fragment positions after the
            # K loop (one struct per position), before any epilogue store; values and expressions unchanged.
            fields = ' '.join(f'{types[t.split("[", 1)[0]]} a{k};' for k, t in enumerate(late))
            body_lines = []
            order = sorted(range(len(late)), key=lambda k: len(late[k]), reverse=True)
            for line in lines:
                head, sep, tail = line.partition('; ')
                for k in order:
                    head = head.replace(late[k], f'p_.a{k}')
                    tail = tail.replace(f'= {late[k]} + ', f'= p_.a{k} + ')
                body_lines.append(head + sep + tail)
            call.update(store=store, late=prefetch != 'early',
                        pre=f'[=](long long i) {{ struct P_ {{ {fields} }}; return P_{{{", ".join(late)}}}; }}',
                        epi=f'[=](long long i, __nv_bfloat16 {register}, const auto p_) {{ '
                            + ' '.join(body_lines) + ' }')
        elif read is not None:
            # P-OPS5-B 'pre': the one outside operand is loaded for every fragment position before the K loop.
            call.update(store=store, pre=f'[=](long long i) {{ return {read}; }}',
                        epi=f'[=](long long i, __nv_bfloat16 {register}, const auto p_) {{ '
                            + ' '.join(_with_prefetch(lines, read)) + ' }')
        else:
            call.update(store=store, epi=f'[=](long long i, __nv_bfloat16 {register}) {{ ' + ' '.join(lines) + ' }')
        multi = bool(late) and 'struct P_' in (call.get('pre') or '')
        if pair and (not call.get('pre') or multi):
            ptypes = {ident(n): ctypes.get(dtypes.get(n)) for n in layouts}
            source_lines = body_lines if multi else lines
            paired = _pair_epilogue(source_lines, register, ptypes, multi)
            if paired is not None:
                call.update(pair=True, epi=paired)
        new['popsc_call'] = call
        _render(new)
        consumer_names = [n for e in members for n in phases[e]['names']]
        new.update(names=[*phase['names'], *consumer_names],
                   targets=[*phase['targets'], *(t for e in members for t in phases[e]['targets'])],
                   reads=list(dict.fromkeys([*phase['reads'], *(r for e in members for r in phases[e]['reads']
                                                                if root(r) not in group)])),
                   writes=[*phase['writes'], *consumer_names],
                   bf16_epilogue={'consumers': consumer_names, 'product_stored': store,
                                  'elided': sorted(n for n in consumer_names if n in elided)})
        merged[g] = new
        records.append({'product': out, 'consumer_names': consumer_names, 'product_stored': store,
                        'arithmetic': 'unchanged consumer expressions on the stored BF16 product value'})
    result = [merged.get(k, phase) for k, phase in enumerate(phases) if k not in absorbed]
    return _topological(result, root), records, elided


def rotate_products(phases):
    """Whole-grid single BF16 products take the stage's CTA rotation (``/*C_VBASE*/``).

    The lowering replaces the token with the caller's rotation plus the tiles of the
    pinned products placed before it in the same stage, and counts this product's
    tiles for the stage. Independent products sharing a level then start on
    different CTAs instead of all starting at CTA 0. Scheduling only: every task's
    arithmetic is unchanged (the rotation only selects which CTA runs a task).
    """
    rotated = []
    for phase in phases:
        if not _popsc_single(phase):
            continue
        call = _call(phase)
        if call['args'][2:6] != ['0', '1', '1', '0']:
            continue
        call['args'][5] = '/*C_VBASE*/0'
        _render(phase)
        rotated.append(phase['names'][0])
    return rotated


def merge_chains(phases, layouts, dtypes, *, ident, index, kept, inplace_roots):
    """Merge an elementwise phase into the elementwise phase producing its operand (same element count).

    Emission order only merges adjacent operations, so a chain such as a momentum update followed later
    by the parameter write that reads it occupies two dependency levels. Here a later phase joins its
    producer when it reads that producer's values element-for-element in its own index space and every
    other operand is produced strictly before the producer's level; its lines follow the producer's in
    the same loop iteration (so a same-element read precedes an in-place store). Values only the merged
    phase reads lose their stores. Returns (phases, records, elided_roots).
    """
    root = lambda name: layouts[name].root if name in layouts else name
    records, elided = [], set()
    changed = True
    while changed:
        changed = False
        writer, level = {}, []
        for k, phase in enumerate(phases):
            level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
            for w in phase['writes']:
                writer[root(w)] = k
        readers = {}
        for k, phase in enumerate(phases):
            for r in phase['reads']:
                readers.setdefault(root(r), set()).add(k)
        for e, phase in enumerate(phases):
            if phase['kind'] != 'elementwise' or len(phase.get('body', ())) != len(phase['names']):
                continue
            own = {root(w) for w in phase['writes']}
            producers = {writer[root(r)] for r in phase['reads'] if root(r) in writer and root(r) not in own}
            candidates = [a for a in producers if phases[a]['kind'] == 'elementwise'
                          and len(phases[a].get('body', ())) == len(phases[a]['names'])
                          and math.prod(phases[a]['shape']) == math.prod(phase['shape'])]
            if len(candidates) != 1:
                continue
            a = candidates[0]
            produced = {root(w) for w in phases[a]['writes']}
            shape = tuple(phase['shape'])
            ok = True
            for r in phase['reads']:
                y = root(r)
                if y in own:
                    continue
                if y in produced:
                    ok = ok and _identity(layouts[r], shape)
                elif y in writer and level[writer[y]] >= level[a]:
                    ok = False
            if not ok or any(dtypes.get(n) not in ('float32', 'bfloat16') for n in phase['names']):
                continue
            lines = []
            for line in phase['body']:
                for n in sorted((n for n in layouts if root(n) in produced), key=len, reverse=True):
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    line = line.replace(text, f'r_{root(n)}')
                lines.append(line)
            merged = dict(phases[a])
            body = list(phases[a]['body'])
            for k, name in enumerate(phases[a]['names']):
                y = root(name)
                if y in produced and y not in kept and y not in inplace_roots and readers.get(y, set()) <= {a, e}:
                    body[k] = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', body[k])
                    elided.add(name)
            merged.update(body=body + lines, names=[*phases[a]['names'], *phase['names']],
                          targets=[*phases[a]['targets'], *phase['targets']],
                          reads=list(dict.fromkeys([*phases[a]['reads'], *(r for r in phase['reads']
                                                                            if root(r) not in produced)])),
                          writes=[*phases[a]['writes'], *phase['writes']],
                          registers={**phases[a].get('registers', {}), **phase.get('registers', {})})
            records.append({'into': phases[a]['names'][0], 'merged': list(phase['names'])})
            phases = [merged if k == a else p for k, p in enumerate(phases) if k != e]
            phases = _topological(phases, root)
            changed = True
            break
    return phases, records, elided



def joint_plans(phases, intervals, *, grid=128, shared_cap=None, light=False):
    """Re-plan BF16 product tiles for each product's share of the grid within its dependency interval.

    Several independent products in one interval run concurrently on rotated CTAs; planning each for the
    whole grid makes them over-split (e.g. six 64-tile products run as three waves of small tasks, each
    paying pipeline fill and epilogue). Each rotated single product is re-planned with the P-OPS-C cost model
    for a grid share proportional to its tiles; arithmetic per output element is unchanged (one FP32
    accumulator, ascending k16, one rounding), so results stay bitwise equal.
    """
    from .matrix_rules import bf16_plan
    records = []
    # By default a re-plan may not raise the program's dynamic shared memory (the L1 carve-out is unchanged).
    cap = shared_cap or max([17408] + [int(p.get('shared_bytes', 0)) for p in phases])
    for members in intervals:
        products = [i for i in members if (_popsc_single(phases[i]) or (JOINT_BATCHED and _popsc_batch_call(phases[i])))
                    and 'popsc_call' in phases[i]
                    and (phases[i]['popsc_call']['args'][5].startswith('/*C_VBASE*/')
                         or (JOINT_BATCHED and 'vrot' in phases[i]['popsc_call']['args'][5]))
                    and not phases[i]['popsc_call'].get('rowepi') and not phases[i]['popsc_call'].get('cluster')]
        total = sum(int(phases[i].get('tiles', 0)) for i in products)
        if len(products) < 2 or total <= grid:
            continue
        for i in products:
            phase, spec = phases[i], phases[i]['matrix_spec']
            if light and any(fn in (phase['popsc_call'].get('epi') or '') for fn in ('expf', 'logf', 'powf', 'rsqrtf')):
                continue  # transcendental epilogues keep the finer whole-grid plan (more tasks share the work)
            share = max(16, grid * int(phase['tiles']) // total)
            plan = bf16_plan(spec['m'], spec['n'], spec['k'], int(spec.get('batch', 1)), spec['ak'], spec['bkc'],
                             grid=share)[0]
            call = phase['popsc_call']
            old = call['targs'][:8]
            new = [str(plan['bm']), str(plan['bn']), str(plan['wm']), str(plan['wn']), old[4], old[5],
                   str(plan['ks']), str(plan['nst'])]
            if new == old:
                continue
            if call['pro'] is not None:
                if spec['m'] % plan['bm']:
                    continue
                call['pro'] = call['pro'].replace(f'o < m0_ + {old[0]}; o += 8', f"o < m0_ + {plan['bm']}; o += 8")
                b_el = plan['bn'] * (plan['ks'] + 8) if old[5] == 'true' else plan['ks'] * (plan['bn'] + 8)
                shared = 2 * (plan['bm'] * (call['k'] + 8) + plan['nst'] * b_el)
            else:
                shared = plan['shared_bytes']
            if shared > cap:
                continue
            call['targs'] = new + call['targs'][8:]
            _render(phase)
            phase['shared_bytes'] = max(int(phase.get('shared_bytes', 0)), shared)
            records.append({'product': phase['names'][0], 'from': old[:4] + old[6:8], 'to': new[:4] + new[6:8],
                            'tiles': [int(phase['tiles']), plan['tiles']], 'grid_share': share})
            phase['tiles'] = plan['tiles']
            phase['matrix_plan'] = plan
    return records


def _tile_cost(bm, bn, k):
    """P6-C per-task time (us) of a BF16 popsc tile on the full resident grid with warm L2, fitted to the
    product-latency probes (gpu-11/gpu-27): a fixed fill/epilogue term plus a K-proportional operand/MMA term."""
    return 1.33 + k * (0.049 * (bm + bn) + 0.00049 * bm * bn) * 1e-3


def _best_offset(load, tiles, cost, grid):
    """Rotation r for `tiles` equal tasks of `cost` on CTAs (r + t) mod grid: minimizes the most loaded CTA that
    takes an extra task (sliding-window maximum over the circular load vector). Returns (offset, new max)."""
    base, extra = divmod(tiles, grid)
    if extra == 0:
        return 0, max(load) + base * cost
    ext = load + load
    window, best = [], None
    from collections import deque
    dq = deque()
    for c in range(len(ext)):
        while dq and ext[dq[-1]] <= ext[c]:
            dq.pop()
        dq.append(c)
        start = c - extra + 1
        if dq[0] < start:
            dq.popleft()
        if start >= 0 and start < grid:
            peak = ext[dq[0]]
            if best is None or peak < best[1]:
                best = (start, peak)
    r, peak = best
    return r, max(peak + (base + 1) * cost, max(load) + base * cost)


def np_cost(phase, footprint):
    """P6-C lptx: per-CTA time (us) of a non-matrix phase on its footprint CTAs, a coarse fit to the transfer-model
    phase profile (gpu-38): row groups scale with the elements each CTA owns; others by kind."""
    kind = str(phase.get('kind', ''))
    if kind == 'row_fused':
        rf = phase.get('row_fusion') or {}
        per_cta = int(rf.get('rows', 1)) * int(rf.get('width', phase.get('reduction_elements', 1))) / max(1, footprint)
        return round(0.5 + 0.0006 * per_cta * max(1, len(phase.get('reads', ()))), 3)
    if kind == 'reduction':
        return 2.5
    if kind.startswith('attention'):
        return 8.0
    if kind in ('softmax', 'log_softmax'):
        return 2.0
    return 1.0


def balanced_plans(phases, intervals, *, grid=128, shared_cap=192 * 1024, max_combos=512, hints=None):
    """P6-C lpt: co-scheduled BF16 products of one dependency interval are planned jointly for the makespan.

    The P-OPS5-B joint plan sizes each product for a grid share and the stage rotation then places the products
    back to back, so a CTA can receive two of the interval's heaviest tiles while others hold one light tile. Here
    each group of identical product shapes chooses a tile among its admitted plans, and products are placed in
    decreasing per-task cost at the rotation that least loads the CTAs taking their extra tasks (per-task cost from
    _tile_cost). Scheduling and tile shape only: every output element keeps one FP32 accumulator from +0, ascending
    k16 and one rounding, so results stay bitwise equal. Records phase['lpt_offset'] for the stage lowering.
    """
    from itertools import product as cartesian
    from .matrix_rules import bf16_plan
    records = []
    planned = {}                                                         # lptx: interval -> absolute product load
    for k_iv, members in enumerate(intervals):
        products = [i for i in members if _popsc_single(phases[i]) and 'popsc_call' in phases[i]
                    and phases[i]['popsc_call']['args'][5].startswith('/*C_VBASE*/')
                    and not phases[i]['popsc_call'].get('rowepi') and not phases[i]['popsc_call'].get('cluster')
                    and phases[i]['popsc_call'].get('split') is None]
        hint = (hints or {}).get(k_iv) or (hints or {}).get(str(k_iv))
        if len(products) < (1 if hint else 2):
            continue
        start = [0.0] * grid
        if hint:
            # lptx: the level's other work (non-matrix phases from CTA 0, other regions' products) and the
            # already planned products of this region's co-scheduled stages; offsets are then absolute CTAs.
            start = [float(x) for x in hint['load']]
            for peer in hint.get('peers', []):
                for c, x in enumerate(planned.get(int(peer), ())):
                    start[c] += x
        groups, options = {}, {}
        for i in products:
            spec, call = phases[i]['matrix_spec'], phases[i]['popsc_call']
            key = (spec['m'], spec['n'], spec['k'], spec['ak'], spec['bkc'], call['pro'] is None, call['targs'][4:6] == ['false', 'false'])
            groups.setdefault(key, []).append(i)
        for key, idx in groups.items():
            spec, call = phases[idx[0]]['matrix_spec'], phases[idx[0]]['popsc_call']
            current = (int(call['targs'][0]), int(call['targs'][1]))
            if call['pro'] is not None:
                options[key] = [None]                                    # prologue products keep their tile
                continue
            cands = [c for c in bf16_plan(spec['m'], spec['n'], spec['k'], 1, spec['ak'], spec['bkc'], grid=grid)[1]
                     if c['shared_bytes'] <= shared_cap and c['bm'] * (c['bn'] // 2 + 1) * 4 <= c['shared_bytes']]
            cands.sort(key=lambda c: ((c['bm'], c['bn']) != current, c['tiles']))
            options[key] = [None] + [c for c in cands if (c['bm'], c['bn']) != current]
        keys = list(groups)
        combos = list(cartesian(*(range(len(options[k])) for k in keys)))[:max_combos]

        def tile_of(i, choice):
            call = phases[i]['popsc_call']
            if choice is None:
                return int(call['targs'][0]), int(call['targs'][1]), int(phases[i]['tiles'])
            return choice['bm'], choice['bn'], choice['tiles']

        best = None
        for combo in combos:
            chosen = {k: options[k][c] for k, c in zip(keys, combo)}
            items = []
            for k in keys:
                for i in groups[k]:
                    bm, bn, tiles = tile_of(i, chosen[k])
                    items.append((-_tile_cost(bm, bn, int(phases[i]['matrix_spec']['k'])), -tiles, i))
            items.sort()
            load, offsets = list(start), {}
            for negcost, negtiles, i in items:
                cost, tiles = -negcost, -negtiles
                r, _ = _best_offset(load, tiles, cost, grid)
                base, extra = divmod(tiles, grid)
                for t in range(grid):
                    load[t] += base * cost
                for t in range(extra):
                    load[(r + t) % grid] += cost
                offsets[i] = r
            score = (round(max(load), 6), round(sum(load), 6), combo)
            if best is None or score < best[0]:
                best = (score, chosen, offsets)
        (makespan, _, _), chosen, offsets = best
        if hint:
            final = [0.0] * grid
            for k in keys:
                for i in groups[k]:
                    bm, bn, tiles = tile_of(i, chosen[k])
                    cost = _tile_cost(bm, bn, int(phases[i]['matrix_spec']['k']))
                    for t in range(tiles):
                        final[(offsets[i] + t) % grid] += cost
            planned[k_iv] = final
        for k in keys:
            choice = chosen[k]
            for i in groups[k]:
                phase, call = phases[i], phases[i]['popsc_call']
                old = list(call['targs'][:8])
                if choice is not None:
                    call['targs'] = [str(choice['bm']), str(choice['bn']), str(choice['wm']), str(choice['wn']),
                                     old[4], old[5], str(choice['ks']), str(choice['nst'])] + call['targs'][8:]
                    _render(phase)
                    phase['shared_bytes'] = max(int(phase.get('shared_bytes', 0)), choice['shared_bytes'])
                    phase['tiles'] = choice['tiles']
                    phase['matrix_plan'] = choice
                phase['lpt_offset'] = offsets[i]
                if hint:
                    phase['lpt_abs'] = True
                records.append({'product': phase['names'][0], 'from': old[:2], 'to': call['targs'][:2],
                                'tiles': int(phase['tiles']), 'offset': offsets[i], 'makespan_us': makespan})
    return records


def deep_rings(phases, *, cap=192 * 1024):
    """P6-C: deepen a BF16 product's cp.async stage ring up to every K slice (bounded by shared memory).

    The P-OPS-C plan keeps at most 4 stages in flight; when one CTA owns one output tile (the latency-bound
    regime of small recurrent products) a K/KS-step product then waits on about (K/KS)/3 dependent L2 round
    trips. With NST >= K/KS + 1 the prologue issues every slice before the first wait. The per-element
    arithmetic is unchanged (one FP32 accumulator, ascending k16, one rounding), so results stay bitwise equal.
    """
    records = []
    for phase in phases:
        if not _popsc_single(phase):
            continue
        call = _call(phase)
        if call.get('split') is not None or call.get('cluster') is not None or call.get('rowepi') is not None \
                or call['pro'] is not None:
            continue
        bm, bn, wm, wn, ak, bkc, ks, nst = call['targs'][:8]
        bm, bn, ks, nst = int(bm), int(bn), int(ks), int(nst)
        k = int(phase['matrix_spec']['k'])
        nk = k // ks
        a_el = bm * (ks + 8) if ak == 'true' else ks * (bm + 8)
        b_el = bn * (ks + 8) if bkc == 'true' else ks * (bn + 8)
        stage = 2 * (a_el + b_el)
        target = min(nk + 1, cap // stage)
        if target <= nst:
            continue
        call['targs'][7] = str(target)
        _render(phase)
        phase['shared_bytes'] = max(int(phase.get('shared_bytes', 0)), target * stage)
        records.append({'product': phase['names'][0], 'k_steps': nk, 'stages': [nst, target],
                        'shared_bytes': target * stage})
    return records


def _kg_choice(bm, bn, ks, groups_first=(4, 2), max_tile=16):
    """Sliced-K warp arrangement: k groups g (dividing KS/16) and a g-way smaller spatial warp grid whose warp tile
    keeps at most max_tile m16n8 accumulators (4 FP32 registers each)."""
    for g in groups_first:
        if (ks // 16) % g:
            continue
        spatial = 8 // g
        best = None
        for wm in (1, 2, 4, 8):
            if spatial % wm:
                continue
            wn = spatial // wm
            if bm % wm or bn % wn or (bm // wm) % 16 or (bn // wn) % 8:
                continue
            mt, nt = (bm // wm) // 16, (bn // wn) // 8
            if nt != 1 and nt % 2 or mt * nt > max_tile:
                continue
            key = (mt * nt, abs(mt - nt))
            if best is None or key < best[0]:
                best = (key, wm, wn)
        if best:
            return {'g': g, 'wm': best[1], 'wn': best[2]}
    return None


def staged_epilogues(phases, heavy_only=False, unroll=4):
    """P6-C epism: single BF16 products with no epilogue or a pair-form epilogue use popsc::btile_se (rounded tile
    staged in shared memory; epilogue as a compact loop over row-major column pairs with coalesced stores)."""
    records = []
    for phase in phases:
        if not _popsc_single(phase):
            continue
        call = _call(phase)
        if any(call.get(k) is not None for k in ('split', 'cluster', 'rowepi', 'kg')) or call['pro'] is not None:
            continue
        if call.get('epi') is not None and not call.get('pair'):
            continue
        if call.get('pre') is not None and not call.get('pair'):
            continue
        epi = call.get('epi') or ''
        if heavy_only and not any(fn in epi for fn in ('expf', 'logf', 'powf', 'tanhf', 'rsqrtf', 'sqrtf')):
            continue  # light epilogues keep the register-direct fragment stores
        call['se'] = True
        if unroll != 4:
            call['se_u'] = unroll  # P6-C epu1/epu2: pairs per thread per epilogue round (code size vs load batching)
        _render(phase)
        records.append(phase['names'][0])
    return records


def sliced_k(phases, *, groups_first=(4, 2)):
    """P6-C: single BF16 products use the sliced-K tile (popsc::btile_kg): KG warp groups split each stage's
    k16 blocks and their partials meet once in shared memory in fixed group order. Same tiles, rotation and stage
    ring; the K sum is reassociated in FP32 (declared numerical change, qualified by the unchanged gates)."""
    records = []
    for phase in phases:
        if not _popsc_single(phase):
            continue
        call = _call(phase)
        if any(call.get(k) is not None for k in ('split', 'cluster', 'rowepi', 'pre')) or call['pro'] is not None:
            continue
        bm, bn, wm, wn, ak, bkc, ks, nst = call['targs'][:8]
        bm, bn, ks, nst = int(bm), int(bn), int(ks), int(nst)
        choice = _kg_choice(bm, bn, ks, groups_first)
        if choice is None:
            continue
        a_el = bm * (ks + 8) if ak == 'true' else ks * (bm + 8)
        b_el = bn * (ks + 8) if bkc == 'true' else ks * (bn + 8)
        if choice['g'] * bm * (bn + 4) * 4 > nst * 2 * (a_el + b_el):
            continue
        call['kg'] = choice
        _render(phase)
        records.append({'product': phase['names'][0], 'tile': [bm, bn], 'from_warps': [int(wm), int(wn)], **choice})
    return records


_SOFTMAX_STORE = re.compile(r'for \(int w = 0; w < (\d+); \+\+w\) \{ const long long k = 32 \* w \+ lane; '
                            r'(v_\w+)\[row \* (\d+)LL \+ k\] = (.*); \}$', re.M)
_SOFTMAX_STORE_MULTI = re.compile(r'if \(live_\) \{ for \(int v_ = 0; v_ < (\d+); \+\+v_\) \{ const int w = w0_ \+ v_; '
                                  r'const long long k = 32 \* w \+ lane; (v_\w+)\[row \* (\d+)LL \+ k\] = (.*); \} \}$', re.M)


def fuse_softmax_epilogues(phases, layouts, dtypes, *, ident, index, kept, inplace_roots):
    """Same-size elementwise consumers of an exact warp-per-row (log-)softmax evaluated in its store loop.

    The consumer lines run per element right after the row's result is formed (flat index
    i = row * W + k), reading the result from its register; every other operand must be produced
    before the softmax's level. Expressions and rounding are unchanged. Returns (phases, records, elided).
    """
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    records, elided, absorbed = [], set(), set()
    for s, phase in enumerate(phases):
        if phase['kind'] not in ('softmax', 'log_softmax') or phase.get('engine') not in (
                'exact_cta_tree_per_warp', 'exact_cta_tree_multi_warp'):
            continue
        code = phase.get('code') or ''
        multi = phase.get('engine') == 'exact_cta_tree_multi_warp'
        match = (_SOFTMAX_STORE_MULTI if multi else _SOFTMAX_STORE).search(code)
        out = phase['names'][-1]
        if not match or match.group(2) != ident(out) or len(phase['writes']) < 1:
            continue
        width = int(match.group(3))
        numel = math.prod(layouts[out].shape)
        group, members = {root(out)}, []
        for e in range(s + 1, len(phases)):
            consumer = phases[e]
            if e in absorbed or consumer['kind'] != 'elementwise' or len(consumer.get('body', ())) != len(consumer['names']):
                continue
            shape = tuple(consumer['shape'])
            if math.prod(shape) != numel or any(root(n) in inplace_roots for n in consumer['names']):
                continue
            if any(dtypes.get(n) not in ('float32', 'bfloat16') for n in consumer['names']):
                continue
            own = {root(w) for w in consumer['writes']}
            if not any(root(r) in group for r in consumer['reads'] if root(r) not in own):
                continue
            ok = True
            for r in consumer['reads']:
                y = root(r)
                if y in own:
                    continue
                if y in group:
                    ok = ok and _identity(layouts[r], shape)
                elif y in writer and level[writer[y]] >= level[s]:
                    ok = False
            if ok:
                members.append(e)
                absorbed.add(e)
                group |= own
        if not members:
            continue
        outside = lambda y: y in kept or any(q != s and q not in members for q in readers.get(y, ()))
        lines = []
        for e in members:
            shape = tuple(phases[e]['shape'])
            for line in phases[e]['body']:
                for n in sorted((n for n in layouts if root(n) in group), key=len, reverse=True):
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    line = line.replace(text, f'r_{root(n)}')
                m = _LINE.match(line)
                if m and not outside(root(m.group(2))):
                    line = re.sub(r'; ' + re.escape(ident(m.group(2))) + r'\[i\] = [^;]*;$', ';', line)
                    elided.add(m.group(2))
                lines.append(line)
        store = f' {ident(out)}[i] = r_{root(out)};' if outside(root(out)) else ''
        if not store:
            elided.add(out)
        head = (f'if (live_) {{ for (int v_ = 0; v_ < {match.group(1)}; ++v_) {{ const int w = w0_ + v_; '
                if multi else f'for (int w = 0; w < {match.group(1)}; ++w) {{ ')
        replacement = (head + f'const long long k = 32 * w + lane; '
                       f'const long long i = row * {width}LL + k; const float r_{root(out)} = {match.group(4)};{store} '
                       + ' '.join(lines) + (' } }' if multi else ' }'))
        names = [n for e in members for n in phases[e]['names']]
        phase['code'] = code[:match.start()] + replacement + code[match.end():]
        phase['names'] = [*phase['names'], *names]
        phase['targets'] = [*phase['targets'], *(t for e in members for t in phases[e]['targets'])]
        phase['reads'] = list(dict.fromkeys([*phase['reads'], *(r for e in members for r in phases[e]['reads']
                                                                if root(r) not in group)]))
        phase['writes'] = [*phase['writes'], *names]
        records.append({'softmax': out, 'consumers': names, 'result_stored': bool(store)})
    if not absorbed:
        return phases, [], set()
    return _topological([p for k, p in enumerate(phases) if k not in absorbed], root), records, elided



def ilp_elementwise(phase, numel, ident, unroll=4):
    """Grid-stride elementwise loop with ``unroll`` independent elements per iteration.

    Every element's computations (and therefore loads) are emitted before any element's stores, so the
    loads of all copies are in flight together instead of one dependent round trip per element. Same
    per-element expressions, order of operations and rounding. Refused (None) when a line reads memory
    the phase itself writes, or when the line format is not the plain register/store form.
    """
    names = phase['names']
    if len(phase.get('body', ())) != len(names) or numel < 4096:
        return None
    written = {ident(n) for n in names}
    computes, stores = [], []
    for line, name in zip(phase['body'], names):
        match = _LINE.match(line)
        if not match or match.group(2) != name:
            return None
        head, _, tail = line.partition('; ')
        if any(f'{w}[' in head for w in written):
            return None
        if tail and not re.fullmatch(r'v_\w+\[i\] = (?:v_\w+\[i\] \+ )?r_\w+;', tail.strip()):
            return None
        computes.append(head + ';')
        stores.append(tail.strip())
    regs = sorted({f'r_{n}' for n in names}, key=len, reverse=True)

    def rename(text, u):
        text = re.sub(r'\bi\b', f'i{u}', text)
        for reg in regs:
            text = re.sub(rf'\b{re.escape(reg)}\b', f'{reg}_{u}', text)
        return text
    body = []
    for u in range(unroll):
        body.append(f'  const long long i{u} = b_ + {u}LL * s_; const bool k{u} = i{u} < {numel}LL;')
    for u in range(unroll):
        body.append(f'  const long long i{u}c = k{u} ? i{u} : b_;')
    for u in range(unroll):
        # Out-of-range copies recompute element b_ (always valid) and do not store.
        body += ['  ' + rename(line, u).replace(f'i{u}', f'i{u}c') for line in computes]
    for u in range(unroll):
        body += [f'  if (k{u}) {{ ' + ' '.join(rename(st, u) for st in stores if st) + ' }']
    return ('{ const long long s_ = (long long)gridDim.x * blockDim.x;\n'
            f'for (long long b_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; b_ < {numel}LL; b_ += {unroll}LL * s_) {{\n'
            + '\n'.join(body) + '\n} }')



_ZERO_STORE = re.compile(r'const float r_(\w+) = 0\.0f; v_(\w+)\[i\] = (v_(\w+)\[i\] \+ )?r_(\w+);$')


def vector_fill(phase, numel):
    """Constant-zero stores (``x = 0``) and zero accumulations (``x = x + 0``) as 16-byte vector loops.

    Same per-element operation (a +0.0f addition keeps its signed-zero effect), four elements per access.
    Only plain flat-index lines over a numel divisible by four qualify; returns None otherwise.
    """
    if numel % 4 or len(phase.get('body', ())) != len(phase['names']):
        return None
    stmts = []
    for line, name in zip(phase['body'], phase['names']):
        m = _ZERO_STORE.match(line)
        if not m or not (m.group(1) == m.group(2) == m.group(5) == name) or (m.group(3) and m.group(4) != name):
            return None
        if m.group(3):
            stmts.append(f'{{ float4 x_ = reinterpret_cast<float4*>(v_{name})[j]; x_.x = x_.x + 0.0f; x_.y = x_.y + 0.0f; '
                         f'x_.z = x_.z + 0.0f; x_.w = x_.w + 0.0f; reinterpret_cast<float4*>(v_{name})[j] = x_; }}')
        else:
            stmts.append(f'reinterpret_cast<float4*>(v_{name})[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);')
    return (f'for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < {numel // 4}LL; '
            f'j += (long long)gridDim.x * blockDim.x) {{\n' + '\n'.join(stmts) + '\n}')



def merged_fills(fills):
    """P6-C zmerge: several constant-zero store phases of one dependency interval as one 16-byte vector loop over
    their concatenated float4 index space (the same stores; only the loop structure and phase count change).
    fills: [(phase, numel)]; returns the merged code, or None when a phase is not a plain zero store."""
    targets = []
    for phase, numel in fills:
        if numel % 4 or len(phase.get('body', ())) != len(phase['names']):
            return None
        for line, name in zip(phase['body'], phase['names']):
            m = _ZERO_STORE.match(line)
            if not m or not (m.group(1) == m.group(2) == m.group(5) == name) or (m.group(3) and m.group(4) != name):
                return None
            targets.append((name, numel // 4, bool(m.group(3))))
    total = sum(n for _, n, _ in targets)
    lines = [f'for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < {total}LL; '
             f'j += (long long)gridDim.x * blockDim.x) {{', '  long long q_ = j;']
    for name, n, acc in targets:
        if acc:  # zero accumulation x = x + 0 (keeps its signed-zero effect), as vector_fill emits it
            body = (f'float4 x_ = reinterpret_cast<float4*>(v_{name})[q_]; x_.x = x_.x + 0.0f; x_.y = x_.y + 0.0f; '
                    f'x_.z = x_.z + 0.0f; x_.w = x_.w + 0.0f; reinterpret_cast<float4*>(v_{name})[q_] = x_;')
        else:
            body = f'reinterpret_cast<float4*>(v_{name})[q_] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);'
        lines.append(f'  if (q_ < {n}LL) {{ {body} continue; }}')
        lines.append(f'  q_ -= {n}LL;')
    lines.append('}')
    return '\n'.join(lines)


def _constant_address(text):
    """An access ``v_x[<integer literal>]`` (a broadcast scalar): every element reads the same address."""
    return re.fullmatch(r'v_\w+\[\d+(?:LL)?\]', text) is not None


def vector_elementwise(phase, numel, texts, unroll=1, bulk=False):
    """FP32 elementwise phase as one float4 load per operand and one float4 store per result per 4 elements.

    texts maps each identity-addressed read/store text (``v_x[<flat index>]``) to its pointer. Every memory
    access of the phase must be such a text; each lane evaluates the unchanged expression of its element
    (loads are issued before any store, and each element is stored once, so in-place updates read first).
    """
    names = phase['names']
    if numel % 4 or len(phase.get('body', ())) != len(names) or numel < 4096:
        return None
    pattern = re.compile(r'v_\w+\[[^\]]*\]')
    computes, stores, loads = [], [], {}
    bf16 = set()
    for line, name in zip(phase['body'], names):
        m = _LINE.match(line)
        if not m or m.group(1) not in ('float', '__nv_bfloat16') or m.group(2) != name:
            return None
        head, _, tail = line.partition('; ')
        st = re.fullmatch(r'v_(\w+)\[i\] = (?:(v_\w+\[i\]) \+ )?r_(\w+);', tail.strip()) if tail else None
        if tail and (not st or st.group(1) != name or st.group(3) != name):
            return None
        if m.group(1) == '__nv_bfloat16':
            # P9-B: a BF16 result is stored only plainly, to its own identity-addressed contiguous array.
            if st and (st.group(2) or f'v_{name}[i]' not in texts):
                return None
            bf16.add(name)
        for text in pattern.findall(head) + ([st.group(2)] if st and st.group(2) else []):
            if text not in texts and _constant_address(text):
                continue                                           # step 02: broadcast scalar read stays inline
            if text not in texts:
                # LATCH-STEP10-SPEED: a non-identity (e.g. broadcast) read of an array this phase does not write
                # stays a per-element scalar load inside the vector loop (its index is rewritten per element).
                array = text.partition('[')[0]
                from .lower import flat_exact_rules_active
                if (not VEC_BROADCAST or not flat_exact_rules_active() or (st and st.group(2) == text)
                        or any(array == f'v_{n}' for n in names)):
                    return None
                continue
            loads.setdefault(text, f'q{len(loads)}_')
        computes.append(head)
        stores.append((name, bool(st), st.group(2) if st else None))
    regs = sorted({f'r_{n}' for n in names}, key=len, reverse=True)

    def block(jv, sfx):
        # One float4 index jv: every load, then the unchanged per-element expressions, then the float4 stores.
        loads_, compute_, stores_ = [], [], []
        loads_ = [f'  const float4 {q}{sfx} = reinterpret_cast<const float4*>({texts[t]})[{jv}];' for t, q in loads.items()]
        for u, comp in enumerate('xyzw'):
            for head in computes:
                line = head
                for t in sorted(loads, key=len, reverse=True):
                    line = line.replace(t, f'{loads[t]}{sfx}.{comp}')
                line = re.sub(r'\bi\b', f'(4 * {jv} + {u})', line)
                for reg in regs:
                    line = re.sub(rf'\b{re.escape(reg)}\b', f'{reg}_{u}{sfx}', line)
                compute_.append('  ' + line + ';')
        for name, stored, acc in stores:
            if not stored:
                continue
            if name in bf16:
                h = [f'(uint32_t)__bfloat16_as_ushort(r_{name}_{u}{sfx})' for u in range(4)]
                stores_.append(f'reinterpret_cast<uint2*>(v_{name})[{jv}] = make_uint2({h[0]} | ({h[1]} << 16), '
                               f'{h[2]} | ({h[3]} << 16));')
                continue
            lanes = [f'{loads[acc]}{sfx}.{c} + r_{name}_{u}{sfx}' if acc else f'r_{name}_{u}{sfx}' for u, c in enumerate('xyzw')]
            stores_.append(f'reinterpret_cast<float4*>(v_{name})[{jv}] = make_float4({", ".join(lanes)});')
        return loads_, compute_, stores_
    if bulk and 1 <= len(loads) <= 4:
        # P9-B bulk streaming (kernels/p9_bulk.cuh): chunks of every operand land in shared memory through
        # cp.async.bulk (thread 0 issues, a transaction-count mbarrier per stage completes); every thread computes
        # its float4 indices of the chunk from shared memory with the unchanged expressions and stores directly.
        n4 = numel // 4
        nr = len(loads)
        cpt = min(2048, (180000 // (3 * nr * 16)) // 256 * 256)
        names_ = list(loads.items())
        _, compute_, stores_ = block('jv_', '')
        sload = [f'    const float4 {q} = p9sb_[(s_ * {nr} + {k}) * {cpt} + jj_];' for k, (t, q) in enumerate(names_)]
        issue = [f'      p9bulk::copy(p9sb_ + (ss_ * {nr} + {k}) * {cpt}, reinterpret_cast<const float4*>({texts[t]}) + cb_ * {cpt}LL, bytes_, ss_);'
                 for k, (t, q) in enumerate(names_)]
        return '\n'.join([
            '{',
            f'  const long long nch_ = ({n4}LL + {cpt - 1}) / {cpt};',
            '  float4* p9sb_ = reinterpret_cast<float4*>(smem);',
            '  p9bulk::init();',
            '  auto p9issue_ = [&](long long cb_, int ss_) {',
            '    if (threadIdx.x == 0) {',
            f'      const long long rem_ = {n4}LL - cb_ * {cpt};',
            f'      const unsigned bytes_ = (unsigned)((rem_ < {cpt} ? rem_ : {cpt}) * 16);',
            f'      p9bulk::expect(ss_, {nr}u * bytes_);',
            *issue,
            '    }',
            '  };',
            '  for (int s0_ = 0; s0_ < p9bulk::STAGES; ++s0_) {',
            '    const long long c0_ = (long long)blockIdx.x + (long long)s0_ * gridDim.x;',
            '    if (c0_ < nch_) p9issue_(c0_, s0_);',
            '  }',
            '  long long it_ = 0;',
            '  for (long long c_ = blockIdx.x; c_ < nch_; c_ += gridDim.x, ++it_) {',
            '    const int s_ = (int)(it_ % p9bulk::STAGES);',
            '    p9bulk::wait(s_, (unsigned)((it_ / p9bulk::STAGES) & 1));',
            f'    const long long base_ = c_ * {cpt};',
            f'    const int cnt_ = (int)({n4}LL - base_ < {cpt} ? {n4}LL - base_ : {cpt});',
            '    for (int jj_ = threadIdx.x; jj_ < cnt_; jj_ += blockDim.x) {',
            '    const long long jv_ = base_ + jj_;',
            *sload, *['  ' + x for x in compute_], *['    ' + x for x in stores_],
            '    }',
            '    __syncthreads();                                  // stage s_ fully read',
            '    const long long cn_ = c_ + (long long)p9bulk::STAGES * gridDim.x;',
            '    if (cn_ < nch_) p9issue_(cn_, s_);',
            '  }',
            '  p9bulk::finish();',
            '}'])
    if unroll <= 1:
        loads_, compute_, stores_ = block('j', '')
        body = loads_ + compute_ + ['  ' + x for x in stores_]
        return (f'for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < {numel // 4}LL; '
                f'j += (long long)gridDim.x * blockDim.x) {{\n' + '\n'.join(body) + '\n}')
    # P6-C vec4u: several float4 indices per iteration, all loads first (more bytes in flight per thread); indices
    # past the end are clamped for the loads and skipped for the stores. Same per-element expressions.
    n4 = numel // 4
    head = [f'  const long long s_ = (long long)gridDim.x * blockDim.x;']
    body, comps, sts = [], [], []
    for u in range(unroll):
        head.append(f'  const long long j{u} = jb_ + {u}LL * s_; const bool k{u} = j{u} < {n4}LL; '
                    f'const long long jc{u} = k{u} ? j{u} : jb_;')
        loads_, compute_, stores_ = block(f'jc{u}', f'_{u}')
        body += loads_
        comps += compute_
        sts += [f'  if (k{u}) {{ {x} }}' for x in stores_]
    return (f'for (long long jb_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; jb_ < {n4}LL; '
            f'jb_ += {unroll}LL * (long long)gridDim.x * blockDim.x) {{\n' + '\n'.join(head + body + comps + sts) + '\n}')




VEC_XDIV = True           # LATCH-STEP10-SPEED: vector_mixed loops use branch-free exact division (latch_xdiv;
                          # gpu-27: 0.2718 -> 0.2660 ms)
VEC_PIPELINE = True       # LATCH-MOESUP-ROW: on (MoE+sup 25M: -0.2 ms aux/plain, scr-sg, bitwise); LATCH-STEP10-SPEED: vector_mixed loads the next vector indices before the current compute


def _pipelined(index, loads, computes, stores, n4, unroll):
    """LATCH-STEP10-SPEED: the vector_mixed loop with its operand loads one grid-stride iteration ahead.

    Each iteration issues the 16-byte loads of the next iteration (into a second register set) before evaluating and
    storing the current one, so a thread's memory requests overlap its arithmetic. Every element still evaluates the
    same expression on the same loaded values and is stored once (identical results); loads of an iteration precede
    its stores and the next iteration's addresses are disjoint from the current one's (grid-stride indices).
    """
    rows = [re.fullmatch(r'\s*const (float4|uint4) (\w+) = (.*);', line) for line in loads]
    if not all(rows):
        return None
    step = f'{unroll}LL * (long long)gridDim.x * blockDim.x'
    at = lambda base: [line.replace('jb_', base) for line in index]
    decl = [f'  {t} {n}, n{n};' for t, n, _ in (r.groups() for r in rows)]
    first = [f'    {n} = {e};' for _, n, e in (r.groups() for r in rows)]
    nxt = [f'      n{n} = {e};' for _, n, e in (r.groups() for r in rows)]
    swap = [f'    {n} = n{n};' for _, n, _ in (r.groups() for r in rows)]
    return ('{\n  const long long s_ = (long long)gridDim.x * blockDim.x;\n'
            '  long long jb_ = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n' + '\n'.join(decl) + '\n'
            f'  if (jb_ < {n4}LL) {{\n' + '\n'.join(at('jb_') + first) + '\n  }\n'
            f'  for (; jb_ < {n4}LL; jb_ += {step}) {{\n' + '\n'.join(at('jb_')) + '\n'
            f'    const long long jn_ = jb_ + {step}; const bool more_ = jn_ < {n4}LL;\n'
            '    if (more_) {\n' + '\n'.join(at('jn_') + nxt) + '\n    }\n'
            + '\n'.join(computes + stores) + '\n    if (more_) {\n' + '\n'.join(swap) + '\n    }\n  }\n}')


VEC_BROADCAST = True      # LATCH-STEP10-SPEED: vector_mixed keeps non-identity reads as per-element loads
VEC_INDEX_HOIST = True    # LATCH-STEP10-SPEED: chunk-uniform broadcast index terms (exact; conv programs)
IM2COL_EPI_UNCAPPED = True  # LATCH-STEP10-SPEED: im2col-operand p5c products ignore the flat epilogue cost cap


def vector_mixed(phase, numel, texts, unroll=2, scattered=(), gathered=None):
    """P8-C vecbf: elementwise phase whose every memory access is identity-indexed FP32 or BF16, as 16-byte vectors.

    texts maps each identity-addressed access text (``v_x[i]`` or its layout-indexed form) to (pointer, dtype). A
    group of G elements (G = 8 when any access is BF16, else 4) per vector index: FP32 operands are loaded as float4
    (two per group of 8), BF16 operands as one uint4 (8 values) and unpacked exactly; every element evaluates the
    unchanged expression of its element; results are stored as float4 / packed BF16 pairs. Loads precede stores and
    every element is stored once, so in-place updates read first. ``unroll`` vector indices are in flight per thread.
    Data movement only: results are bit-identical to the scalar loop.
    """
    names = phase['names']
    if len(phase.get('body', ())) != len(names) or numel < 4096:
        return None
    dtypes = {t: d for t, (_, d) in texts.items()}
    pattern = re.compile(r'v_\w+\[[^\]]*\]')
    computes, stores, loads = [], [], {}
    gathers = {}
    result_types = {}
    for line, name in zip(phase['body'], names):
        m = re.match(r'const (float|__nv_bfloat16) r_(\w+) = ', line)
        if not m or m.group(2) != name:
            return None
        head, _, tail = line.partition('; ')
        st = re.fullmatch(r'v_(\w+)\[i\] = (?:(v_\w+\[i\]) \+ )?r_(\w+);', tail.strip()) if tail else None
        if tail and not st and name in scattered:
            # Step 02: a padded matrix operand is stored element by element at its padded index (a function of i).
            sc = re.fullmatch(r'v_(\w+)\[([^\]]+)\] = r_(\w+);', tail.strip())
            if sc and sc.group(1) == name and sc.group(3) == name:
                for text in pattern.findall(head):
                    if text not in texts and _constant_address(text):
                        continue
                    if text not in texts and gathered and text in gathered:
                        gathers.setdefault(text, True)
                        loads.setdefault(text, f'q{len(loads)}_')
                        continue
                    if text not in texts:
                        return None
                    loads.setdefault(text, f'q{len(loads)}_')
                computes.append(head)
                stores.append((name, 'scatter', sc.group(2)))
                result_types[name] = m.group(1)
                continue
        if tail and (not st or st.group(1) != name or st.group(3) != name):
            return None
        if st and st.group(2) and m.group(1) != 'float':
            return None
        for text in pattern.findall(head) + ([st.group(2)] if st and st.group(2) else []):
            if text not in texts and _constant_address(text):
                continue                                           # step 02: broadcast scalar read stays inline
            if text not in texts and gathered and text in gathered and not (st and text == st.group(2)):
                gathers.setdefault(text, True)                       # step 02: padded-pitch read (padacc::gather8)
                loads.setdefault(text, f'q{len(loads)}_')
                continue
            if text not in texts:
                # LATCH-STEP10-SPEED: a non-identity (e.g. broadcast) read of an array this phase does not write
                # stays a per-element scalar load inside the vector loop (its index is rewritten per element).
                array = text.partition('[')[0]
                from .lower import flat_exact_rules_active
                if (not VEC_BROADCAST or not flat_exact_rules_active() or (st and st.group(2) == text)
                        or any(array == f'v_{n}' for n in names)):
                    return None
                continue
            loads.setdefault(text, f'q{len(loads)}_')
        computes.append(head)
        stores.append((name, bool(st), st.group(2) if st else None))
        result_types[name] = m.group(1)
    for name, stored, _ in stores:
        if stored and stored != 'scatter':
            entry = texts.get(f'v_{name}[i]')
            want = 'float32' if result_types[name] == 'float' else 'bfloat16'
            if entry is None or entry[1] != want:
                return None
    bf = (any(d == 'bfloat16' for d in dtypes.values()) or any(t == '__nv_bfloat16' for t in result_types.values())
          or bool(gathers))
    G = 8 if bf else 4
    if numel % G:
        return None
    regs = sorted({f'r_{n}' for n in names}, key=len, reverse=True)
    comps = 'xyzw'

    def element(text, sfx, k):
        q, d = loads[text], ('bfloat16' if text in gathers else dtypes[text])
        if d == 'float32':
            return f'{q}{k // 4}{sfx}.{comps[k % 4]}'
        word = f'{q}0{sfx}.{comps[k // 2]}'
        return (f'__ushort_as_bfloat16((unsigned short)({word} >> 16))' if k % 2
                else f'__ushort_as_bfloat16((unsigned short)({word} & 0xffffu))')

    def block(jv, sfx):
        loads_, compute_, stores_ = [], [], []
        for t, q in loads.items():
            if t in gathers:
                ptr, d_, pitch_ = gathered[t]
                loads_.append(f'  const uint4 {q}0{sfx} = padacc::gather8<{d_}u, {pitch_}u>({ptr}, (unsigned)({G} * ({jv})));')
                continue
            ptr, d = texts[t]
            if d == 'float32':
                for h in range(G // 4):
                    loads_.append(f'  const float4 {q}{h}{sfx} = reinterpret_cast<const float4*>({ptr})[{G // 4} * ({jv}) + {h}];')
            else:
                loads_.append(f'  const uint4 {q}0{sfx} = reinterpret_cast<const uint4*>({ptr})[{jv}];')
        for k in range(G):
            for head in computes:
                line = head
                for t in sorted(loads, key=len, reverse=True):
                    line = line.replace(t, element(t, sfx, k))
                if VEC_INDEX_HOIST and VEC_BROADCAST:
                    # LATCH-STEP10-SPEED: element k of the chunk starting at base = G*jv (a multiple of G): for a
                    # divisor or modulus D that G divides, (base + k) / D == base / D and (base + k) % D ==
                    # base % D + k, so the remaining (broadcast) index terms are computed once per chunk. Exact.
                    from .lower import flat_exact_rules_active
                    if flat_exact_rules_active():
                        line = re.sub(r'\(unsigned\)\((?:i|\(i\))\) / (\d+)u',
                                      lambda m: (f'(unsigned)({G} * {jv}) / {m[1]}u' if int(m[1]) % G == 0
                                                 else m[0]), line)
                        line = re.sub(r'\(unsigned\)\((?:i|\(i\))\) % (\d+)u',
                                      lambda m: (f'((unsigned)({G} * {jv}) % {m[1]}u + {k}u)' if int(m[1]) % G == 0
                                                 else m[0]), line)
                line = re.sub(r'\bi\b', f'({G} * {jv} + {k})', line)
                for reg in regs:
                    line = re.sub(rf'\b{re.escape(reg)}\b', f'{reg}_{k}{sfx}', line)
                compute_.append('  ' + line + ';')
        for name, stored, acc in stores:
            if not stored:
                continue
            if stored == 'scatter':
                geometry = scattered.get(name) if isinstance(scattered, dict) else None
                if geometry and G == 8:
                    d, pitch = geometry
                    words = [f'((unsigned)__bfloat16_as_ushort(r_{name}_{2 * w}{sfx}) | '
                             f'((unsigned)__bfloat16_as_ushort(r_{name}_{2 * w + 1}{sfx}) << 16))' for w in range(4)]
                    stores_.append(f'padacc::scatter8<{d}u, {pitch}u>(v_{name}, (unsigned)({G} * ({jv})), '
                                   f'make_uint4({", ".join(words)}));')
                    continue
                where = [re.sub(r'\bi\b', f'({G} * ({jv}) + {k})', acc) for k in range(G)]
                stores_.append(' '.join(f'v_{name}[{where[k]}] = r_{name}_{k}{sfx};' for k in range(G)))
                continue
            if result_types[name] == 'float':
                for h in range(G // 4):
                    lanes = [(f'{element(acc, sfx, 4 * h + u)} + ' if acc else '') + f'r_{name}_{4 * h + u}{sfx}'
                             for u in range(4)]
                    stores_.append(f'reinterpret_cast<float4*>(v_{name})[{G // 4} * ({jv}) + {h}] = make_float4({", ".join(lanes)});')
            else:
                words = [f'((unsigned)__bfloat16_as_ushort(r_{name}_{2 * w}{sfx}) | '
                         f'((unsigned)__bfloat16_as_ushort(r_{name}_{2 * w + 1}{sfx}) << 16))' for w in range(G // 2)]
                stores_.append(f'reinterpret_cast<uint4*>(v_{name})[{jv}] = make_uint4({", ".join(words)});')
        return loads_, compute_, stores_
    n4 = numel // G
    head = ['  const long long s_ = (long long)gridDim.x * blockDim.x;']
    body, comps_, sts = [], [], []
    for u in range(max(1, unroll)):
        head.append(f'  const long long j{u} = jb_ + {u}LL * s_; const bool k{u} = j{u} < {n4}LL; '
                    f'const long long jc{u} = k{u} ? j{u} : jb_;')
        loads_, compute_, stores_ = block(f'jc{u}', f'_{u}')
        body += loads_
        comps_ += compute_
        sts += [f'  if (k{u}) {{ {x} }}' for x in stores_]
    from .lower import flat_exact_rules_active
    if VEC_XDIV and flat_exact_rules_active():
        # LATCH-STEP10-SPEED: the iteration's divisions without slow-path branches (latch_xdiv); its stores happen
        # only when every operand was in the exact window, otherwise the original expressions are evaluated and stored.
        from .p5c_epilogue import exact_division
        fast = [exact_division(line) for line in comps_]
        if fast != comps_:
            comps_ = ['  bool xok_ = true;', '  {'] + fast + ['  if (xok_) {'] + sts + ['  }', '  }', '  if (!xok_) {'] \
                + comps_ + sts + ['  }']
            sts = []
    if VEC_PIPELINE and flat_exact_rules_active():
        code = _pipelined(head[1:], body, comps_, sts, n4, max(1, unroll))
        if code:
            return code
    return (f'for (long long jb_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; jb_ < {n4}LL; '
            f'jb_ += {max(1, unroll)}LL * (long long)gridDim.x * blockDim.x) {{\n' + '\n'.join(head + body + comps_ + sts) + '\n}')


def fuse_row_epilogues(phases, layouts, dtypes, *, ident, kept, row_tile=16):
    """A row group that consumes one BF16 product's [M, N] result runs in that product's epilogue.

    The product is planned with full-row tiles (row_tile x N, eight warps across N), so each task owns
    complete rows: after its K loop the rounded BF16 tile stays in shared memory and the row group's
    unchanged one-warp-per-row code runs on the task's rows, reading the product from shared memory
    instead of global memory. The row group's grid level and the product write/re-read (when nothing
    else reads the product) are removed; nothing is recomputed. Admission: single one-batch BF16
    product, plain row-major [M, N] result, N a multiple of 128 up to 256, row group of M rows and
    width N reading the product element-for-element, every other row-group operand produced strictly
    before the product's level. Returns (phases, records, elided_roots).
    """
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    plans, used = {}, set()
    for r, phase in enumerate(phases):
        info = phase.get('row_fusion') or {}
        # P9-C: an exact one-warp-per-row softmax phase (same header form) is admitted like a row group.
        soft = (ROWEPI_SOFTMAX and phase['kind'] in ('softmax', 'log_softmax')
                and phase.get('engine') == 'exact_cta_tree_per_warp')
        if soft:
            head = _ROW_HEADER_VAR.search(phase.get('code') or '')
            if not head:
                continue
            rows = int(head.group(2))
        elif phase['kind'] != 'row_fused' or 'code_r1' not in phase or 'softmax' in info:
            continue
        else:
            width, rows = info.get('width'), info.get('rows')
        products = {writer[root(x)] for x in phase['reads'] if root(x) in writer and phases[writer[root(x)]]['kind'] == 'gemm'}
        if len(products) != 1:
            continue
        g = products.pop()
        gp = phases[g]
        out = gp['writes'][0] if gp['writes'] else None
        # P9-C: batched BF16 products with a contiguous [batch, M, N] result (rows = batch * M).
        batched = ROWEPI_BATCHED and out is not None and _popsc_batched(gp, layouts, out)
        if g in used or not (_popsc_single(gp) or batched) or (gp.get('popsc_call') or {}).get('pro') or (gp.get('popsc_call') or {}).get('epi'):
            continue
        spec = gp['matrix_spec']
        nb = int(spec.get('batch', 1))
        if soft:
            width = int(spec['n'])
        shape = (nb, int(spec['m']), int(spec['n'])) if batched else (rows, width)
        if (len(gp['writes']) != 1 or dtypes.get(out) != 'bfloat16' or nb * spec['m'] != rows or spec['n'] != width
                or width % 128 or width > 256 or spec['m'] % row_tile or not _identity(layouts[out], shape)):
            continue
        if any(level[writer[root(x)]] >= level[g] for x in phase['reads']
               if root(x) in writer and root(x) != root(out)):
            continue
        if not (_ROW_HEADER_VAR if soft else _ROW_HEADER).search(phase['code'] if soft else phase['code_r1']):
            continue
        plans[r] = g
        used.add(g)
    if not plans:
        return phases, [], set()
    records, elided, merged, removed = [], set(), {}, set()
    for r, g in plans.items():
        rp, gp = phases[r], phases[g]
        spec, out = gp['matrix_spec'], gp['writes'][0]
        width = int(spec['n'])
        rows = int(spec.get('batch', 1)) * int(spec['m'])
        if 'code_r1' in rp and rp['kind'] == 'row_fused':
            body = _ROW_HEADER.sub(f'for (long long o = m0_ + (threadIdx.x >> 5); o < m0_ + {row_tile}; o += 8) {{',
                                   rp['code_r1'], count=1)
        else:
            body = _ROW_HEADER_VAR.sub(lambda m: f'for (long long {m.group(1)} = m0_ + (threadIdx.x >> 5); '
                                                 f'{m.group(1)} < m0_ + {row_tile}; {m.group(1)} += 8) {{',
                                       rp['code'], count=1)
        for n in sorted((n for n in layouts if root(n) == root(out)), key=len, reverse=True):
            body = re.sub(re.escape(ident(n)) + r'\[[^\]]*\]', f'sC_[i - m0_ * {width}LL]', body)
        k = int(spec['k'])
        ks = 32 if k % 32 == 0 else 16
        nk = k // ks
        nst = min(4, nk + 1) if nk > 1 else 2
        bkc = bool(spec['bkc'])
        a_el = row_tile * (ks + 8) if spec['ak'] else ks * (row_tile + 8)
        b_el = width * (ks + 8) if bkc else ks * (width + 8)
        shared = max(2 * nst * (a_el + b_el), 1024 + 2 * row_tile * width)
        call = _call(gp)
        store = root(out) in kept or bool(readers.get(root(out), set()) - {g, r})
        call.update(store=store, rowepi=f'[=](long long m0_, const __nv_bfloat16* sC_) {{\n{body}\n}}')
        call['targs'] = [str(row_tile), str(width), '1', '8', str(spec['ak']).lower(), str(bkc).lower(), str(ks),
                         str(nst)] + call['targs'][8:]
        new = dict(gp)
        new['popsc_call'] = call
        _render(new)
        new.update(names=[*gp['names'], *rp['names']], targets=[*gp['targets'], *rp['targets']],
                   reads=list(dict.fromkeys([*gp['reads'], *(x for x in rp['reads'] if root(x) != root(out))])),
                   writes=[*gp['writes'], *rp['writes']], tiles=rows // row_tile,
                   shared_bytes=max(int(gp.get('shared_bytes', 0)), shared),
                   row_epilogue={'row_group': rp['names'][:4], 'product_stored': store, 'row_tile': row_tile})
        merged[g] = new
        removed.add(r)
        records.append({'product': out, 'row_group': rp['names'], 'product_stored': store, 'row_tile': row_tile,
                        'shared_bytes': shared})
    result = [merged.get(k, p) for k, p in enumerate(phases) if k not in removed]
    return _topological(result, root), records, elided



def split_k_plans(phases, *, grid=128, min_steps=16, choices=(4, 2)):
    """Rotated single BF16 products with a long K chain: [(phase index, splits)], splits from ``choices``.

    Chosen when K / KS >= min_steps, each slice keeps at least four K steps, and tiles x splits fits the grid.
    P8-C splitk16: choices (16, 8, 4, 2) for weight-gradient products whose few output tiles leave most CTAs idle.
    """
    plans = []
    for i, phase in enumerate(phases):
        call = phase.get('popsc_call')
        if (not _popsc_single(phase) or not call or call.get('pro') or call.get('rowepi') or call.get('pre')
                or call.get('cluster')
                or not call['args'][5].startswith('/*C_VBASE*/')):
            continue
        k, ks, tiles = int(phase['matrix_spec']['k']), int(call['targs'][6]), int(phase.get('tiles', 0))
        nk = k // ks
        if nk < min_steps or not tiles:
            continue
        for splits in choices:
            if nk % splits == 0 and nk // splits >= 4 and tiles * splits <= grid:
                plans.append((i, splits))
                break
    return plans



CLUSTER = 8


def fuse_cluster_rows(phases, layouts, dtypes, *, ident, kept, block_rows=32):
    """A row group consuming one BF16 product runs in that product's epilogue, distributed over a cluster.

    The product's 8 column tiles of width W/8 (= 32, one frozen warp column block each) for a block of 32 rows
    run on the 8 CTAs of one thread-block cluster (column tile = cluster rank). Each CTA evaluates the row
    group's per-element prologue for its columns from its rounded tile in shared memory, forms the frozen
    per-column-block warp sum of every row, and publishes it; after a cluster barrier every CTA reads the
    eight partials through distributed shared memory and replays the identical second-stage butterfly, then
    evaluates the row and per-element epilogues for its own columns. Statistics, values and stores are
    bitwise identical to the separate row group; its grid level and the product write/re-read disappear,
    with the same task parallelism as the product. Requires W = 256 and a cluster launch.
    """
    from .matrix_rules import bf16_plan
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    plans, used = {}, set()
    for r, phase in enumerate(phases):
        parts = phase.get('row_parts')
        if phase['kind'] != 'row_fused' or not parts or parts['width'] != 256 or parts['nw'] != 8:
            continue
        products = {writer[root(x)] for x in phase['reads'] if root(x) in writer and phases[writer[root(x)]]['kind'] == 'gemm'}
        if len(products) != 1:
            continue
        g = products.pop()
        gp = phases[g]
        call = gp.get('popsc_call') or {}
        if g in used or not _popsc_single(gp) or call.get('pro') or call.get('epi') or call.get('rowepi'):
            continue
        spec = gp['matrix_spec']
        out = gp['writes'][0]
        if (len(gp['writes']) != 1 or dtypes.get(out) != 'bfloat16' or spec['n'] != 256 or spec['m'] != parts['rows_count']
                or spec['m'] % block_rows or not _identity(layouts[out], (spec['m'], spec['n']))):
            continue
        if any(level[writer[root(x)]] >= level[g] for x in phase['reads'] if root(x) in writer and root(x) != root(out)):
            continue
        plans[r] = g
        used.add(g)
    if not plans:
        return phases, [], set()
    records, merged, removed = [], {}, set()
    for r, g in plans.items():
        rp, gp = phases[r], phases[g]
        parts, spec, out = rp['row_parts'], gp['matrix_spec'], gp['writes'][0]
        names = sorted((n for n in layouts if root(n) == root(out)), key=len, reverse=True)

        def local(line):
            for n in names:
                line = re.sub(re.escape(ident(n)) + r'\[[^\]]*\]', 'sC_[(o - m0_) * 32 + lane]', line)
            return line.replace('if (lane == 0)', 'if (lane == 0 && w == 0)')
        per = block_rows // 8
        code = ['[=](long long m0_, const __nv_bfloat16* sC_, float* part_) {',
                '  auto cl_ = cooperative_groups::this_cluster();',
                '  const int lane = threadIdx.x & 31, wr_ = threadIdx.x >> 5;',
                '  const int w = (int)cl_.block_rank();']
        code += [f'  float keep_{n}_r[{per}];' for n in parts['carried']]
        code += ['#pragma unroll', f'  for (int q_ = 0; q_ < {per}; ++q_) {{',
                 f'    const long long o = m0_ + wr_ * {per} + q_;',
                 '    const long long i = o * 256LL + 32 * w + lane;']
        code += ['    ' + local(line) for line in parts['prologue']]
        code += [f'    keep_{n}_r[q_] = r_{n};' for n in parts['carried']]
        code += [f'    const float s_ = s101h::warp_sum(0.0f + {local(parts["leaf"])});',
                 f'    if (lane == 0) part_[wr_ * {per} + q_] = s_;', '  }', '  cl_.sync();',
                 '#pragma unroll', f'  for (int q_ = 0; q_ < {per}; ++q_) {{',
                 f'    const long long o = m0_ + wr_ * {per} + q_;',
                 f'    float second = lane < 8 ? cl_.map_shared_rank(part_, lane)[wr_ * {per} + q_] : 0.0f;',
                 '    second = s101h::warp_sum(second);',
                 f'    const float r_{parts["out"]} = second{parts["scale"]};']
        if parts['out_stored']:
            code.append(f'    if (lane == 0 && w == 0) {ident(parts["out"])}[o] = r_{parts["out"]};')
        code += ['    {', '    const long long i = o;']
        code += ['    ' + local(line) for line in parts['rows']]
        code += ['    {', '    const long long i = o * 256LL + 32 * w + lane;']
        code += [f'    const float r_{n} = keep_{n}_r[q_];' for n in parts['carried']]
        code += ['    ' + local(line) for line in parts['full']]
        code += ['    }', '    }', '  }', '  cl_.sync();', '}']
        plan = bf16_plan(spec['m'], spec['n'], spec['k'], 1, spec['ak'], spec['bkc'], force={'bm': 32, 'bn': 32})[0]
        call = _call(gp)
        store = root(out) in kept or bool(readers.get(root(out), set()) - {g, r})
        call.update(store=store, cluster='\n'.join(code))
        call['targs'] = ['32', '32', str(plan['wm']), str(plan['wn']), str(spec['ak']).lower(),
                         str(spec['bkc']).lower(), str(plan['ks']), str(plan['nst'])] + call['targs'][8:]
        new = dict(gp)
        new['popsc_call'] = call
        _render(new)
        new.update(names=[*gp['names'], *rp['names']], targets=[*gp['targets'], *rp['targets']],
                   reads=list(dict.fromkeys([*gp['reads'], *(x for x in rp['reads'] if root(x) != root(out))])),
                   writes=[*gp['writes'], *rp['writes']], tiles=(spec['m'] // 32) * 8,
                   shared_bytes=max(int(gp.get('shared_bytes', 0)), plan['shared_bytes'], 1024 + 2 * 32 * 32 + 4 * 32),
                   cluster_rows={'row_group': rp['names'][:4], 'product_stored': store, 'cluster': CLUSTER})
        merged[g] = new
        removed.add(r)
        records.append({'product': out, 'row_group': rp['names'], 'product_stored': store})
    result = [merged.get(k, p) for k, p in enumerate(phases) if k not in removed]
    return _topological(result, root), records, set()



def vector_epilogue_run(value, lines, types):
    """P8-C 'p5cv': the p5c register-epilogue lambda over 8 contiguous outputs with 16-byte operand loads and result
    stores. ``types`` maps each pointer identifier to 'float32' / 'bfloat16'. Identity reads ``v_Y[(i)]`` / ``v_Y[i]``
    of float32/bfloat16 operands become one uint4 / two float4 loads before the 8 elements; ``v_X[i] = r_X;`` stores
    become register slots stored as two float4 / one uint4 afterwards. Elements are evaluated by the unchanged lines in
    the same order; every element is loaded before any is stored, and each output element has one writer, so the
    values are bit-identical to the scalar lambda (kept as the fallback when a pointer is not 16-byte aligned)."""
    scalar = ('#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n        const long long i = (long long)i0 + j;\n        '
              + value + '\n        ' + '\n        '.join(lines) + '\n      }')
    store_re = re.compile(r'; (v_\w+)\[i\] = (r_\w+);$')
    load_re = re.compile(r'\b(v_\w+)\[\(?i\)?\]')
    loads, stores, body = [], [], []
    for line in lines:
        m = store_re.search(line)
        head = line[:m.start()] + ';' if m else line
        for ptr in load_re.findall(head):
            if ptr not in types:
                return None
            if ptr not in loads:
                loads.append(ptr)
        if m:
            ptr, reg = m.group(1), m.group(2)
            lm = _LINE.match(line)
            if ptr not in types or not lm or f'r_{lm.group(2)}' != reg:
                return None
            want = 'float32' if lm.group(1) == 'float' else 'bfloat16' if lm.group(1) == '__nv_bfloat16' else None
            if want != types[ptr]:
                return None
            stores.append((ptr, reg, want))
            head = head[:-1] + f'; s{ptr}[j] = {reg};'
        body.append(load_re.sub(lambda mm: f'l{mm.group(1)}[j]', head))
    if not stores and not loads:
        return None
    if len({p for p, _, _ in stores}) != len(stores) or set(loads) & {p for p, _, _ in stores}:
        return None   # in-place or repeated targets keep the scalar lambda
    ptrs = list(dict.fromkeys(loads + [p for p, _, _ in stores]))
    aligned = ' | '.join(f'reinterpret_cast<uintptr_t>({p})' for p in ptrs)
    pre = []
    for ptr in loads:
        if types[ptr] == 'float32':
            pre.append(f'float l{ptr}[8]; {{ const float4 a_ = reinterpret_cast<const float4*>({ptr} + i0)[0], '
                       f'b_ = reinterpret_cast<const float4*>({ptr} + i0)[1]; l{ptr}[0] = a_.x; l{ptr}[1] = a_.y; '
                       f'l{ptr}[2] = a_.z; l{ptr}[3] = a_.w; l{ptr}[4] = b_.x; l{ptr}[5] = b_.y; l{ptr}[6] = b_.z; '
                       f'l{ptr}[7] = b_.w; }}')
        else:
            pre.append(f'__nv_bfloat16 l{ptr}[8]; {{ const uint4 a_ = *reinterpret_cast<const uint4*>({ptr} + i0); '
                       f'const unsigned w_[4] = {{a_.x, a_.y, a_.z, a_.w}}; for (int k_ = 0; k_ < 8; ++k_) '
                       f'l{ptr}[k_] = __ushort_as_bfloat16((unsigned short)((k_ & 1) ? (w_[k_ >> 1] >> 16) : (w_[k_ >> 1] & 0xffffu))); }}')
    for ptr, _, t in stores:
        pre.append(f'{"float" if t == "float32" else "__nv_bfloat16"} s{ptr}[8];')
    post = []
    for ptr, _, t in stores:
        if t == 'float32':
            post.append(f'reinterpret_cast<float4*>({ptr} + i0)[0] = make_float4(s{ptr}[0], s{ptr}[1], s{ptr}[2], s{ptr}[3]); '
                        f'reinterpret_cast<float4*>({ptr} + i0)[1] = make_float4(s{ptr}[4], s{ptr}[5], s{ptr}[6], s{ptr}[7]);')
        else:
            words = ', '.join(f'((unsigned)__bfloat16_as_ushort(s{ptr}[{2 * w}]) | ((unsigned)__bfloat16_as_ushort(s{ptr}[{2 * w + 1}]) << 16))'
                              for w in range(4))
            post.append(f'*reinterpret_cast<uint4*>({ptr} + i0) = make_uint4({words});')
    vector = ('\n      '.join(pre) + '\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n        const long long i = (long long)i0 + j;\n        '
              + value + '\n        ' + '\n        '.join(body) + '\n      }\n      ' + '\n      '.join(post))
    return ('[&](int e, unsigned i0, const float (&gv)[8]) {\n      if (((' + aligned + ') & 15) == 0) {\n      '
            + vector + '\n      } else {\n' + scalar + '\n      }\n    }')


# P8-B: declared fast activation arithmetic (capture precision policy activation='fast', set by lower() from the
# graph metadata): exponentials in register epilogues use __expf and the SiLU / sigmoid divisions __fdividef.
import contextvars as _contextvars
FAST_ACTIVATION = _contextvars.ContextVar('p8b_fast_activation', default=False)
_EPI_SILU = re.compile(r'\((r_\w+) / \(1\.0f \+ __expf\(([^()]*)\)\)\)')
_EPI_RCP = re.compile(r'__frcp_rn\(\(1\.0f \+ __expf\(([^()]*)\)\)\)')


def _fast_activation(line):
    line = re.sub(r'(?<![\w_])expf\(', '__expf(', line)
    line = _EPI_SILU.sub(lambda m: f'__fdividef({m.group(1)}, 1.0f + __expf({m.group(2)}))', line)
    return _EPI_RCP.sub(lambda m: f'__fdividef(1.0f, 1.0f + __expf({m.group(1)}))', line)


# P8-B: vector access in p5c register epilogues. Each epilogue call covers 8 contiguous outputs (i0 multiple of 8,
# identity index), so identity-indexed BF16 operand loads become one 16-byte load per chunk issued in the
# prefetch callback (all chunks of a group before any is computed) and identity-indexed stores become one 16- or
# 32-byte store per chunk after the per-element lines. Same values and expressions (data movement only).
P5C_EPI_VECTOR = True
EPI_XDIV = True             # LATCH-STEP10-SPEED: conv-program epilogues use branch-free exact division (latch_xdiv;
                            # gpu-25: 0.2833 -> 0.2720 ms; probe: 0 mismatches vs `/` over all silu/sigmoid inputs)
EPI_VECTOR_VIEWS = True     # LATCH-STEP10-SPEED: aligned view identifiers qualify for the vector epilogue loads
P5C_EPI_VECTOR_F32 = True   # LATCH-STEP10-SPEED: also FP32 identity side inputs (16-byte loads per chunk)
_EPI_LOAD =re.compile(r'(\b[A-Za-z_]\w*)\[\(?i\)?\]')
_EPI_STORE = re.compile(r'; (\b[A-Za-z_]\w*)\[i\] = (r_\w+);$')


def _vector_epilogue(lines, types):
    """(prelude, ld callback, run lines, tail) for identity-indexed loads/stores of known BF16/FP32 identifiers,
    or None when nothing qualifies. `types` maps identifiers to 'bfloat16' / 'float32'."""
    stores, loads, floads = {}, {}, {}
    out = []
    for line in lines:
        match = _EPI_STORE.search(line)
        if match and types.get(match.group(1)) in ('bfloat16', 'float32'):
            name = match.group(1)
            k = stores.setdefault(name, len(stores))
            line = line[:match.start()] + f'; s8{k}_[j] = {match.group(2)};'
        head, sep, tail = line.partition(' = ')
        def load(m):
            name = m.group(1)
            if m.group(0) not in (f'{name}[(i)]', f'{name}[i]'):
                return m.group(0)
            from .lower import flat_exact_rules_active
            if types.get(name) == 'float32' and P5C_EPI_VECTOR_F32 and flat_exact_rules_active():
                # LATCH-STEP10-SPEED: FP32 side inputs as two 16-byte loads per 8-element chunk (before its lines).
                return f'p8f{floads.setdefault(name, len(floads))}_[j]'
            if types.get(name) != 'bfloat16':
                return m.group(0)
            k = loads.setdefault(name, len(loads))
            return f'p5c::upkb(p8l{k}_[e], j)'
        out.append(head + sep + _EPI_LOAD.sub(load, tail) if sep else line)
    if not stores and not loads and not floads:
        return None
    ctype = {'bfloat16': '__nv_bfloat16', 'float32': 'float'}
    prelude = ' '.join(f'uint4 p8l{k}_[8];' for k in loads.values())
    ld = ('[&](int e, unsigned i0) { ' + ' '.join(
        f'p8l{k}_[e] = __ldg(reinterpret_cast<const uint4*>({name} + i0));' for name, k in loads.items()) + ' }')
    decl = ' '.join(f'{ctype[types[name]]} s8{k}_[8];' for name, k in stores.items())
    decl += ''.join(f' float p8f{k}_[8]; *reinterpret_cast<float4*>(p8f{k}_) = __ldcg(reinterpret_cast<const float4*>('
                    f'{name} + i0)); *reinterpret_cast<float4*>(p8f{k}_ + 4) = __ldcg(reinterpret_cast<const float4*>('
                    f'{name} + i0 + 4));' for name, k in floads.items())
    tail = ' '.join(f'p5c::st8({name} + i0, s8{k}_);' for name, k in stores.items())
    return prelude, ld, decl, out, tail


def fuse_p5c_epilogues(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, vector=False,
                       cost_limit=None):
    """P7-C 'p5ce': same-index elementwise consumers of a warpgroup-MMA (p5c) product run in its register epilogue.

    The analysis is fuse_bf16_epilogues' (P-OPS5-B): later elementwise phases of the product's element count that
    read the product (or an absorbed value) with the identity index, every other operand produced strictly before the
    product's level. Their lines run per element on the product's rounded value (gv[j]: the exact widening of the
    stored BF16 result, or the FP32-converted value), in the order of the original phases; values read only by the
    absorbed lines lose their stores, and the product's own store is omitted when nothing else reads it. Expressions,
    operand bits and rounding are unchanged. Returns (phases, records, elided_roots)."""
    from .matrix_rules import emit_group
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    absorbed, plans = {}, {}
    for g, phase in enumerate(phases):
        spec = phase.get('matrix_spec') or {}
        if (phase['kind'] != 'gemm' or spec.get('engine') != 'p5c' or spec.get('epi') or spec.get('out_acc')
                or spec.get('splitk') or spec.get('out_raw')):
            continue
        results = list(phase['writes'])
        if len(results) != 1 or dtypes.get(results[0]) not in ('bfloat16', 'float32'):
            continue
        out = results[0]
        out_layout = layouts[out]
        if len(out_layout.shape) != 2 or not _identity(out_layout, out_layout.shape):
            continue
        numel = math.prod(out_layout.shape)
        group, members = {out}, []
        cost = 0
        # P9-C (diagnostic P9C_HEAVYEPI=split): transcendental consumers of a many-wave product stay separate phases
        # (the p5c register epilogue does not overlap the MMAs, so its per-tile cost repeats every wave).
        heavy_split = os.environ.get('P9C_HEAVYEPI') == 'split' and numel >= 512 * 128 * 128
        for e in range(g + 1, len(phases)):
            consumer = phases[e]
            if e in absorbed or consumer['kind'] != 'elementwise' or not consumer.get('body') \
                    or len(consumer['body']) != len(consumer['names']):
                continue
            if heavy_split and any('expf' in line for line in consumer['body']):
                continue
            # LATCH-STEP10-SPEED: an implicit-GEMM (im2col operand) product takes its consumers without the flat cost
            # cap: otherwise each consumer chain re-reads the full pixel-major activation in a separate phase
            # (measured on the step-10 development block: 0.478 -> 0.413 ms).
            limit = None if (spec.get('virtual_a') and IM2COL_EPI_UNCAPPED) else cost_limit
            candidate_cost = _epilogue_cost(consumer) if limit is not None else 0
            if limit is not None and cost + candidate_cost > limit:
                continue
            shape = tuple(consumer['shape'])
            if math.prod(shape) != numel or any(root(n) in inplace_roots for n in consumer['names']):
                continue
            if any(dtypes.get(n) not in ('float32', 'bfloat16') for n in consumer['names']):
                continue
            own = {root(w) for w in consumer['writes']}
            if not any(root(r) in group for r in consumer['reads'] if root(r) not in own):
                continue
            ok = True
            for r in consumer['reads']:
                y = root(r)
                if y in own:
                    continue
                if y in group:
                    if not _identity(layouts[r], shape):
                        ok = False
                elif y in writer and level[writer[y]] >= level[g]:
                    ok = False
            if ok:
                members.append(e)
                cost += candidate_cost
                absorbed[e] = g
                group |= own
        if members:
            plans[g] = (out, members, group)
    if not plans:
        return phases, [], set()
    records, elided, merged = [], set(), {}
    for g, (out, members, group) in plans.items():
        phase = phases[g]
        names_in_group = sorted((n for n in layouts if root(n) in group), key=len, reverse=True)
        outside = lambda y: y in kept or any(q != g and q not in members for q in readers.get(y, ()))
        lines = []
        for e in members:
            consumer = phases[e]
            shape = tuple(consumer['shape'])
            for line in consumer['body']:
                for n in names_in_group:
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    line = line.replace(text, f'r_{root(n)}')
                match = _LINE.match(line)
                name = match.group(2) if match else None
                if name and not outside(root(name)):
                    line = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', line)
                    elided.add(name)
                lines.append(line)
        store = outside(out)
        value = (f'const float r_{out} = gv[j];' if dtypes[out] == 'float32'
                 else f'const __nv_bfloat16 r_{out} = __float2bfloat16_rn(gv[j]);')
        # P9 union: p5cv retains C's vector order; B's staged access applies otherwise.
        if vector:
            run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n'
                   '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                   + '\n        '.join(lines) + '\n      }\n    }')
            types = {ident(n): dtypes[n] for n in layouts if dtypes.get(n) in ('float32', 'bfloat16')}
            run = vector_epilogue_run(value, lines, types) or run
            prelude, ld = '', '[&](int e, unsigned i0) {}'
        else:
            from . import p5c_matrix
            if p5c_matrix.P5C_EXACT_REWRITES:
                # P8-B: the flat path's exact epilogue rewrites (common transcendental subexpressions once,
                # (1.0f / (x)) -> __frcp_rn(x): both correctly rounded) in loop-region epilogues too.
                from .p5c_epilogue import common_subexpressions, reciprocal
                temps, lines = common_subexpressions(lines, f'p5c_t{g}_')
                lines = [reciprocal(line) for line in temps + lines]
            if FAST_ACTIVATION.get():
                lines = [_fast_activation(line) for line in lines]
            bvec = None
            if P5C_EPI_VECTOR:
                from .lower import flat_exact_rules_active
                # LATCH-STEP10-SPEED: also view identifiers whose pointer stays 16-byte aligned (offset in elements
                # a multiple of 8); the access text already indexes the view's own pointer element by element.
                views = EPI_VECTOR_VIEWS and flat_exact_rules_active()
                types = {ident(n): dtypes[n] for n in layouts if n in dtypes
                         and (layouts[n].root == n or (views and layouts[n].offset % 8 == 0))}
                bvec = _vector_epilogue(lines, types)
            if bvec:
                prelude, ld, decl, lines, tail = bvec
                run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n      ' + decl + '\n#pragma unroll\n'
                       '      for (int j = 0; j < 8; ++j) {\n'
                       '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                       + '\n        '.join(lines) + '\n      }\n      ' + tail + '\n    }')
            else:
                prelude, ld, decl, tail = '', '[&](int e, unsigned i0) {}', '', ''
                run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n'
                       '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                       + '\n        '.join(lines) + '\n      }\n    }')
            from .lower import flat_exact_rules_active
            if EPI_XDIV and flat_exact_rules_active():
                # LATCH-STEP10-SPEED: the chunk's divisions without the per-division slow-path branch; the whole
                # chunk is recomputed with the original expressions when any operand left the exact window.
                from .p5c_epilogue import exact_division
                fast = [exact_division(line) for line in lines]
                if fast != lines and not any(re.search(r'\bv_\w+\[[^\]]*\] = ', line) for line in lines):
                    loop = lambda ls: ('#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n'
                                       '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                                       + '\n        '.join(ls) + '\n      }\n      ')
                    run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n      ' + decl + '\n      bool xok_ = true;'
                           '\n      ' + loop(fast) + 'if (!xok_) {\n      ' + loop(lines) + '}\n      ' + tail + '\n    }')
        spec = dict(phase['matrix_spec'], epi={'prelude': prelude, 'run': run, 'ld': ld},
                    out=(phase['matrix_spec']['out'] if store else 'nullptr'))
        code, extra = emit_group([spec], ident=ident)
        new = dict(phase)
        new.update(extra)
        consumer_names = [n for e in members for n in phases[e]['names']]
        new.update(code=code, matrix_spec=spec, names=[*phase['names'], *consumer_names],
                   targets=[*phase['targets'], *(t for e in members for t in phases[e]['targets'])],
                   reads=list(dict.fromkeys([*phase['reads'], *(r for e in members for r in phases[e]['reads']
                                                                if root(r) not in group)])),
                   writes=[*phase['writes'], *consumer_names],
                   p5c_epilogue={'consumers': consumer_names, 'product_stored': store,
                                 'elided': sorted(n for n in consumer_names if n in elided)})
        if not store:
            elided.add(out)
        merged[g] = new
        records.append({'product': out, 'consumer_names': consumer_names, 'product_stored': store,
                        'arithmetic': 'unchanged consumer expressions on the rounded product value in registers'})
    result = [merged.get(k, phase) for k, phase in enumerate(phases) if k not in absorbed]
    return _topological(result, root), records, elided


# ------------------------------------------------------------------------------------------------ release step 14
# Cost bound of the same-index consumers absorbed into one staged TMA epilogue (None: no bound). The staged epilogue
# runs once per output chunk after the main loop, on a tile already in shared memory; unlike the p5c/popsc register
# epilogues it does not lengthen the accumulator's live range. Measured on the development program (step 14).
TMA_EPI_COST_LIMIT = None
# Evaluate a single-reader elementwise producer inside the epilogue that reads it (release step 14).
INLINE_PRODUCERS = True
# Store the BF16 product instead of its exact FP32 widening for readers outside the epilogue (release step 14).
NARROW_WIDENINGS = True


# LATCH-MOE-EMA-100: batch-stacked provider TMA products ([batch, M, N] contiguous results; the staged epilogue's
# flat index is the stacked row times N plus the column, i.e. the contiguous result index) take epilogues too.
# e01 batch-stacked TMA epilogues: opt-in (expert and Newton-Schulz regions measured slower, receipt c50/c52).
BATCHED_TMA_EPILOGUES = os.environ.get("LATCH_OPS_BATCHED_TMA_EPILOGUES", "0") == "1"


# e01b register-direct pair epilogues: opt-in (expert regions measured 0.2-0.5 ms slower, receipt c50).
PAIR_FALLBACK = os.environ.get("LATCH_OPS_TMA_PAIR_EPILOGUES", "0") == "1"


def _tma_record(phase):
    record = phase.get('tma_respec')
    if record is None and BATCHED_TMA_EPILOGUES:
        record = phase.get('tma_respec_batched')
    return record


def spec_of(phase):
    return _tma_record(phase)['spec']


def fuse_tma_epilogues(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, cost_limit=None,
                       prefetch=False, xdiv=False, prefetch_rows=False):
    """Release step 14: same-index elementwise consumers of a provider pops4b TMA + wgmma product (masked edges,
    batch 1) run in that product's staged epilogue (pops4b::tma_body STAGED, epi.vec8 over 8-column chunks).

    Admission and line rewriting are fuse_p5c_epilogues' (P7-C): later elementwise phases of the product's element
    count that read the product (or an absorbed value) with the identity index, every other operand produced strictly
    before the product's dependency level. Lines evaluate on the rounded product value (the widened BF16 result), in
    original phase order; values read only by absorbed lines lose their stores; the product store is omitted when
    nothing else reads it. Expressions, operand bits and rounding are unchanged. Returns (phases, records, elided)."""
    from .matrix_adapters.bodies import Refusal, respec_tma
    root = lambda name: layouts[name].root if name in layouts else name
    writer, level = {}, []
    for k, phase in enumerate(phases):
        level.append(1 + max((level[writer[root(r)]] for r in phase['reads'] if root(r) in writer), default=-1))
        for w in phase['writes']:
            writer[root(w)] = k
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    absorbed, plans = {}, {}

    def _inlinable(q, e, ref, shape, numel):
        # An elementwise producer read only by consumer e (identity index, same element count), whose stores
        # nothing else reads, and whose own inputs are ready by the product's level: its lines run first at the
        # same element inside the epilogue; its phase, barrier interval and store disappear (values unchanged).
        producer = phases[q]
        if (q in absorbed or producer['kind'] != 'elementwise' or not producer.get('body')
                or len(producer['body']) != len(producer['names']) or producer.get('vprod_inline')
                or math.prod(tuple(producer['shape'])) != numel or not _identity(layouts[ref], shape)):
            return False
        writes_ = {root(w) for w in producer['writes']}
        if writes_ & set(kept) or any(readers.get(w, set()) - {e, q} for w in writes_):
            return False
        if any(root(n) in inplace_roots for n in producer['names']):
            return False
        # The merged phase waits for the producer's inputs instead of the producer; they must be ready before the
        # product's level, so the product is never delayed.
        return not any(root(x) in group_now or (root(x) not in writes_ and root(x) in writer
                                                 and level[writer[root(x)]] >= level[g])
                       for x in producer['reads'])
    for g, phase in enumerate(phases):
        record = _tma_record(phase)
        if phase['kind'] != 'gemm' or not record or phase.get('tma_epilogue'):
            continue
        results = list(phase['writes'])
        if len(results) != 1 or dtypes.get(results[0]) != 'bfloat16':
            continue
        out = results[0]
        out_layout = layouts[out]
        spec = record['spec']
        expected = ((int(spec['batch']), spec['m'], spec['n']) if spec.get('tma_batch') else (spec['m'], spec['n']))
        if (not _identity(out_layout, out_layout.shape) or spec['n'] % 8
                or tuple(out_layout.shape) != expected):
            continue
        numel = math.prod(out_layout.shape)
        group, members, cost = {out}, [], 0
        group_now = group
        for e in range(g + 1, len(phases)):
            consumer = phases[e]
            if e in absorbed or consumer['kind'] != 'elementwise' or not consumer.get('body') \
                    or len(consumer['body']) != len(consumer['names']):
                continue
            candidate_cost = _epilogue_cost(consumer) if cost_limit is not None else 0
            if cost_limit is not None and cost + candidate_cost > cost_limit:
                continue
            shape = tuple(consumer['shape'])
            if math.prod(shape) != numel or any(root(n) in inplace_roots for n in consumer['names']):
                continue
            if any(dtypes.get(n) not in ('float32', 'bfloat16') for n in consumer['names']):
                continue
            own = {root(w) for w in consumer['writes']}
            if not any(root(r) in group for r in consumer['reads'] if root(r) not in own):
                continue
            ok, inline = True, []
            for r in consumer['reads']:
                y = root(r)
                if y in own:
                    continue
                if y in group:
                    if not _identity(layouts[r], shape):
                        ok = False
                elif y in writer and INLINE_PRODUCERS and _inlinable(writer[y], e, r, shape, numel):
                    if writer[y] not in inline:
                        inline.append(writer[y])
                elif y in writer and level[writer[y]] >= level[g]:
                    ok = False
            if ok:
                for q in inline:
                    members.append(q)
                    absorbed[q] = g
                    group |= {root(w) for w in phases[q]['writes']}
                members.append(e)
                cost += candidate_cost
                absorbed[e] = g
                group |= own
        if members:
            plans[g] = (out, sorted(members), group)
    if not plans:
        return phases, [], set()
    records, elided, merged, refused = [], set(), {}, set()
    for g, (out, members, group) in plans.items():
        phase = phases[g]
        names_in_group = sorted((n for n in layouts if root(n) in group), key=len, reverse=True)
        outside = lambda y: y in kept or any(q != g and q not in members for q in readers.get(y, ()))
        lines, elided_here = [], set()
        for e in members:
            consumer = phases[e]
            shape = tuple(consumer['shape'])
            for line in consumer['body']:
                head, sep, tail = line.partition('; ')
                for n in names_in_group:
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    head = head.replace(text, f'r_{root(n)}')
                    # split_components materializes cross-component reads
                    # with the loop index. Restore them before the vector
                    # prefetch pass can treat an internal value as an input.
                    if _identity(layouts[n], shape):
                        head = head.replace(f'{ident(n)}[i]', f'r_{root(n)}')
                line = head + sep + tail
                match = _LINE.match(line)
                name = match.group(2) if match else None
                if name and not outside(root(name)):
                    line = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', line)
                    elided_here.add(name)
                lines.append(line)
        store = outside(out)
        if NARROW_WIDENINGS:
            # An exact widening of the product (r_v = float(bf16 product)) stored only for elementwise readers outside
            # this epilogue: store the BF16 product instead and let those readers widen it on load (same values).
            pattern = re.compile(r'^const float r_(\w+) = __bfloat162float\(r_' + re.escape(out)
                                 + r'\); (\w+)\[i\] = r_\1;$')
            for k_, line in enumerate(lines):
                match = pattern.match(line)
                if not match or match.group(2) != ident(match.group(1)):
                    continue
                v = match.group(1)
                if root(v) != v or v in kept or not outside(v) or dtypes.get(v) != 'float32':
                    continue
                if [n for n in layouts if root(n) == v] != [v] or tuple(layouts[v].shape) != tuple(layouts[out].shape):
                    continue
                outer = [q for q in readers.get(v, ()) if q != g and q not in members]
                if not outer or any(phases[q]['kind'] != 'elementwise' or not phases[q].get('body')
                                    or len(phases[q]['body']) != len(phases[q]['names'])
                                    or absorbed.get(q, len(phases)) < g for q in outer):
                    continue
                load = re.compile(r'\b' + re.escape(ident(v)) + r'\[([^\[\]]*)\]')
                for q in outer:
                    phases[q]['body'] = [load.sub(lambda m: f'__bfloat162float({ident(out)}[{m.group(1)}])', x)
                                         for x in phases[q]['body']]
                    phases[q]['reads'] = list(dict.fromkeys(out if r_ == v else r_ for r_ in phases[q]['reads']))
                    readers.setdefault(out, set()).add(q)
                readers[v] = set()
                lines[k_] = line[:line.index('; ')] + ';'
                elided_here.add(v)
                store = True
        value = f'const __nv_bfloat16 r_{out} = __float2bfloat16_rn(gv[j]);'
        scalar_lines = list(lines)          # per-element order before rewrites (state placement proof)
        from . import p5c_matrix
        if p5c_matrix.P5C_EXACT_REWRITES:
            from .p5c_epilogue import common_subexpressions, reciprocal
            temps, lines = common_subexpressions(lines, f'tma_t{g}_')
            lines = [reciprocal(line) for line in temps + lines]
        if FAST_ACTIVATION.get():
            lines = [_fast_activation(line) for line in lines]
        types = {ident(n): dtypes[n] for n in layouts if n in dtypes and layouts[n].root == n}
        bvec = _vector_epilogue_tma(lines, types) if P5C_EPI_VECTOR else None
        pf = pr = None
        if bvec:
            prelude, ld, decl, lines, tail = bvec
            if prefetch_rows:
                # Side inputs produced two or more dependency levels before the product (or program inputs / state)
                # are likely out of L2: one bulk L2 prefetch per tile row when the task starts. Cache hint only.
                by_ident = {ident(n): n for n in layouts if layouts[n].root == n}
                old_sides = []
                for name in dict.fromkeys(re.findall(r'reinterpret_cast<const (?:uint4|float4)\*>\((\w+) \+ i0\)', ld)):
                    r_ = by_ident.get(name)
                    if r_ is None or dtypes.get(r_) not in ('bfloat16', 'float32'):
                        continue
                    if r_ not in writer or level[g] - level[writer[r_]] >= 2:
                        old_sides.append((name, 2 if dtypes[r_] == 'bfloat16' else 4))
                if old_sides:
                    pr = '[&](unsigned i0, unsigned n) { ' + ' '.join(
                        f'asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"({name} + i0), '
                        f'"r"(n * {size}u) : "memory");' for name, size in old_sides) + ' }'

            if prefetch:
                # Step-14 speed ('tepf'): L2 prefetch of the identity-indexed side inputs when the product's task
                # starts (pops4b P6-B pre hook), so the epilogue's loads after the last k step hit L2. Values unchanged.
                sides = list(dict.fromkeys(re.findall(r'reinterpret_cast<const (?:uint4|float4)\*>\((\w+) \+ i0\)', ld)))
                if sides:
                    pf = '[&](unsigned i0) { ' + ' '.join(
                        f'asm volatile("prefetch.global.L2 [%0];" :: "l"({name} + i0));' for name in sides) + ' }'

            ld, tail = ld.replace('p5c::', 'tmaepi::'), tail.replace('p5c::', 'tmaepi::')
            lines = [line.replace('p5c::', 'tmaepi::') for line in lines]
            fast, changed = xdiv_lines(lines) if xdiv else (lines, False)
            if changed:
                # Straight-line fast paths for every element, then (rarely) the whole chunk again with the original
                # expressions when any operand left the fast-path domain: identical stored values.
                loop = lambda body: ('#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n'
                                     '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                                     + '\n        '.join(body) + '\n      }\n      ')
                run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n      ' + decl + '\n      bool ok_ = true;\n'
                       + loop(fast) + 'if (!ok_) {\n' + loop(lines) + '}\n      ' + tail + '\n    }')
            else:
                run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n      ' + decl + '\n#pragma unroll\n'
                       '      for (int j = 0; j < 8; ++j) {\n'
                       '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                       + '\n        '.join(lines) + '\n      }\n      ' + tail + '\n    }')
        else:
            prelude, ld = '', '[&](int e, unsigned i0) {}'
            run = ('[&](int e, unsigned i0, const float (&gv)[8]) {\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n'
                   '        const long long i = (long long)i0 + j;\n        ' + value + '\n        '
                   + '\n        '.join(lines) + '\n      }\n    }')
        epi = {'code': (f'tmaepi::vec8p({ld}, {run}, {pf})' if pf else f'tmaepi::vec8r({ld}, {run}, {pr})' if pr
                        else f'tmaepi::vec8({ld}, {run})'), 'store': store, 'staged': True}
        try:
            code, staging = respec_tma(_tma_record(phase), ident=ident, epi=epi)
            if 'tmaepi::' not in code:
                raise Refusal('re-emitted product body does not evaluate the epilogue')
        except Refusal as error:
            code = None
            if PAIR_FALLBACK and spec.get('pair', True) and spec['n'] % 2 == 0:
                # LATCH-MOE-EMA-100: when the staged tile does not fit beside the operand ring, the register-direct
                # pair epilogue (pops4b::tma_body EPI on each even-column pair of rounded, widened values) evaluates
                # the original per-element lines (before any expression rewrite) at the two indices.
                pair_run = ('[&](unsigned i0, float x0, float x1) {\n#pragma unroll\n      for (int j = 0; j < 2; ++j) {\n'
                            '        const long long i = (long long)i0 + j;\n'
                            f'        const __nv_bfloat16 r_{out} = __float2bfloat16_rn(j ? x1 : x0);\n        '
                            + '\n        '.join(scalar_lines) + '\n      }\n    }')
                pair_epi = {'code': pair_run, 'store': store, 'staged': False, 'pair_direct': True}
                try:
                    code, staging = respec_tma(_tma_record(phase), ident=ident, epi=pair_epi)
                    if 'decltype(epi)' not in code:
                        raise Refusal('re-emitted product body does not evaluate the pair epilogue')
                    epi, prelude = pair_epi, ''
                except Refusal:
                    code = None
            if code is None:
                refused.add(g)
                records.append({'product': out, 'refused': str(error)})
                continue
        code = '{\n' + prelude + '\n' + code + '\n}' if prelude else code
        new = dict(phase)
        consumer_names = [n for e in members for n in phases[e]['names']]
        headers = list(dict.fromkeys([*phase.get('headers', []), 'latch_tma_epi.cuh']))
        new.update(code=code, names=[*phase['names'], *consumer_names],
                   targets=[*phase['targets'], *(t for e in members for t in phases[e]['targets'])],
                   reads=list(dict.fromkeys([*phase['reads'], *(r for e in members for r in phases[e]['reads']
                                                                if root(r) not in group)])),
                   writes=[*phase['writes'], *consumer_names],
                   shared_bytes=int(phase.get('shared_bytes', 0)) + staging, headers=headers,
                   tma_epilogue={'consumers': consumer_names, 'product_stored': store,
                                 'elided': sorted(n for n in consumer_names if n in elided_here),
                                 'staging_bytes': staging, 'epi': epi, 'prelude': prelude,
                                 'lines': scalar_lines,
                                 'shape': [spec_of(phase)['m'] * int(spec_of(phase).get('batch', 1)
                                                                     if spec_of(phase).get('tma_batch') else 1),
                                           spec_of(phase)['n']]})
        elided |= elided_here
        if not store:
            elided.add(out)
        merged[g] = new
        records.append({'product': out, 'consumer_names': consumer_names, 'product_stored': store,
                        'arithmetic': 'unchanged consumer expressions on the rounded product value (staged tile)'})
    keep = {e for e, g in absorbed.items() if g in refused}
    result = [merged.get(k, phase) for k, phase in enumerate(phases) if k not in absorbed or k in keep]
    return _topological(result, root), records, elided


def xdiv_lines(lines):
    """Step-14 speed ('xdiv'): FP32 epilogue lines with IEEE divisions (A / B) and __frcp_rn(X) rewritten to the
    straight-line fast paths latch_div_f(ok_, A, B) / latch_rcp_f(ok_, X) (kernels/latch_tma_epi.cuh). Only
    `const float r_*` lines, only divisions outside index brackets with their own grouping parentheses. Returns
    (lines, changed)."""
    out, changed = [], False
    for line in lines:
        if not line.startswith('const float r_'):
            out.append(line)
            continue
        if '__frcp_rn(' in line:
            line = line.replace('__frcp_rn(', 'latch_rcp_f(ok_, ')
            changed = True
        start = 0
        while True:
            pos = line.find(' / ', start)
            if pos < 0:
                break
            start = pos + 3
            if line[:pos].count('[') != line[:pos].count(']'):
                continue
            depth, left = 0, None
            for k in range(pos - 1, -1, -1):
                if line[k] == ')':
                    depth += 1
                elif line[k] == '(':
                    if depth == 0:
                        left = k
                        break
                    depth -= 1
            if left is None or (left > 0 and (line[left - 1].isalnum() or line[left - 1] == '_')):
                continue
            depth, right = 0, None
            for k in range(pos + 3, len(line)):
                if line[k] == '(':
                    depth += 1
                elif line[k] == ')':
                    if depth == 0:
                        right = k
                        break
                    depth -= 1
            if right is None:
                continue
            a, b = line[left + 1:pos], line[pos + 3:right]
            if any(_top_level_comma(x) for x in (a, b)) or 'LL' in a + b or 'u)' in a + b:
                continue
            line = line[:left] + f'latch_div_f(ok_, {a}, {b})' + line[right + 1:]
            changed = True
            start = left
        out.append(line)
    return out, changed


def _top_level_comma(text):
    depth = 0
    for ch in text:
        if ch in '([':
            depth += 1
        elif ch in ')]':
            depth -= 1
        elif ch == ',' and depth == 0:
            return True
    return False


def joint_tma_plans(phases, intervals, *, ident, grid=None):
    """Release step 14: tile choice for the provider pops4b TMA products that share one dependency interval.

    Products of one interval run back to back on rotated CTA offsets (each starts after the previous one's tiles), so
    the interval lasts as long as the most loaded CTA. Each product's plan is chosen among its body's own admitted
    tile plans (matrix_rules_p7b.wg_plan) to minimize that estimated makespan (the planner's per-tile cost; ties keep
    the provider's plan). Only tile shape/stage count change; every product keeps its arithmetic (FP32 wgmma
    accumulation over ascending 64-wide k steps, one BF16 rounding) and its epilogue. Returns (phases, records).

    LATCH-TINYMOE-MATRIX (ops.matrix_schedule): cp.async wgmma products chosen for short contractions ('wg_respec',
    same planner and arithmetic) take part too, and deterministic split-K tasks of the interval count as fixed loads.
    A choice that would newly confine the interval's elementwise phases to the CTAs the products leave idle (the
    step-14 'ecta' restriction, total tasks below the grid) is not taken."""
    import itertools
    from .matrix_adapters.bodies import BY_NAME, Refusal, pinned, respec_tma
    tma_body = BY_NAME['pops4b_tma']
    wg_body = BY_NAME['pops4b_wg']
    records = []
    with pinned(tma_body.module, tma_body.settings) as mod:
        grid = mod.TARGET_GRID if grid is None else grid
        limit = mod.SHARED_LIMIT

    def record_of(i):
        return phases[i].get('tma_respec') or phases[i].get('wg_respec')

    for k, members in enumerate(intervals):
        tma_products = [i for i in members if phases[i]['kind'] == 'gemm' and phases[i].get('tma_respec')
                        and '/*B_VBASE*/' in phases[i].get('code', '')]
        wg_products = [i for i in members if phases[i]['kind'] == 'gemm' and phases[i].get('wg_respec')
                       and '/*B_VBASE*/' in phases[i].get('code', '')]
        products = tma_products + wg_products      # the flat emitter starts TMA products first
        # LATCH-TINYMOE-MATRIX: deterministic split-K products (ops/matrix_tma_split.py) of the interval run after
        # the TMA products on the following rotated CTAs; their (slice, tile) tasks are fixed loads of the makespan.
        fixed = [i for i in members if phases[i]['kind'] == 'gemm' and phases[i].get('tma_split_k')
                 and '/*B_VBASE*/' in phases[i].get('code', '')]
        if not products or len(products) + len(fixed) < 2 or len(products) + len(fixed) > 4:
            continue
        fixed_loads = []
        for i in fixed:
            record = phases[i]['tma_split_k']
            with pinned(tma_body.module, tma_body.tile_settings()) as mod:
                try:
                    _, candidates = mod.wg_plan(record['shape'][0], record['shape'][1], record['slice_k'], 1,
                                                record['ak'], record['bkc'], grid=grid, shared_limit=limit)
                except mod.MatrixRuleError:
                    candidates = []
            match = [c for c in candidates if [c['bm'], c['bn'], c['wm']] == record['tile']]
            if not match:
                break
            fixed_loads.append((int(phases[i]['tiles']), match[0]['local_us'] / max(1, match[0]['waves'])))
        if len(fixed_loads) != len(fixed):
            continue
        options = []
        for i in products:
            body = tma_body if phases[i].get('tma_respec') else wg_body
            spec, plan = record_of(i)['spec'], record_of(i)['plan']
            staging = int((phases[i].get('tma_epilogue') or {}).get('staging_bytes', 0))
            settings = body.tile_settings() if body is tma_body else body.settings
            with pinned(body.module, settings) as mod:
                try:
                    _, candidates = mod.wg_plan(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'],
                                                spec['bkc'], grid=grid, shared_limit=limit)
                    if body is tma_body:
                        from .matrix_adapters.bodies import deepen
                        candidates = [deepen(c, grid, limit) for c in candidates]
                except mod.MatrixRuleError:
                    candidates = []
            rows = []
            for c in candidates:
                c_staging = c['bm'] * (c['bn'] + 8) * 2 if staging else 0
                if c['shared_bytes'] + c_staging > limit or (spec['n'] % 8 and staging):
                    continue
                rows.append((c, c['local_us'] / max(1, c['waves'])))
            if not any((c['bm'], c['bn'], c['wm']) == (plan['bm'], plan['bn'], plan['wm']) for c, _ in rows):
                rows = []
            options.append(rows)
        if any(not rows for rows in options):
            continue
        fixed_tasks = sum(tasks for tasks, _ in fixed_loads)
        elementwise = any(phases[i]['kind'] == 'elementwise' for i in members)

        def makespan(choice):
            load = [0.0] * grid
            offset = 0
            for (c, per_tile) in choice:
                for t in range(c['tiles']):
                    load[(offset + t) % grid] += per_tile
                offset += c['tiles']
            for tasks, per_task in fixed_loads:
                for t in range(tasks):
                    load[(offset + t) % grid] += per_task
                offset += tasks
            return max(load)
        current = [next(r for r in rows if (r[0]['bm'], r[0]['bn'], r[0]['wm']) == (record_of(i)['plan']['bm'],
                        record_of(i)['plan']['bn'], record_of(i)['plan']['wm']))
                   for i, rows in zip(products, options)]
        current_tasks = sum(c['tiles'] for c, _ in current) + fixed_tasks
        best, best_cost = current, makespan(current)
        for choice in itertools.product(*options):
            tasks = sum(c['tiles'] for c, _ in choice) + fixed_tasks
            if elementwise and fixed and tasks < ECTA_GRID <= current_tasks:
                continue
            cost = makespan(choice)
            if cost < best_cost * 0.97:
                best, best_cost = list(choice), cost
        if best == current:
            continue
        changed = []
        for i, (c, _), (old, _) in zip(products, best, current):
            if (c['bm'], c['bn'], c['wm']) == (old['bm'], old['bn'], old['wm']):
                continue
            key = 'tma_respec' if phases[i].get('tma_respec') else 'wg_respec'
            body = tma_body if key == 'tma_respec' else wg_body
            record = dict(phases[i][key], plan=dict(c))
            epi = (phases[i].get('tma_epilogue') or {}).get('epi')
            try:
                if epi is not None:
                    code, staging = respec_tma(record, ident=ident, epi=epi)
                    prelude = phases[i]['tma_epilogue'].get('prelude')
                    code = '{\n' + prelude + '\n' + code + '\n}' if prelude else code
                else:
                    with pinned(body.module, body.settings) as mod:
                        depth = mod._LOWER_DEPTH[0]
                        mod._LOWER_DEPTH[0] = 1
                        try:
                            code = mod._code(record['spec'], record['plan'], ident, 0, 1, 1, record['rot'])
                        finally:
                            mod._LOWER_DEPTH[0] = depth
                    staging = 0
            except Refusal:
                continue
            phases[i] = dict(phases[i], code=code, tiles=c['tiles'], work_items=c['tiles'],
                             shared_bytes=int(c['shared_bytes']) + staging, **{key: record},
                             matrix_plan={key_: v for key_, v in c.items() if isinstance(v, (int, float, str, bool))})
            changed.append({'phase': i, 'from': [old['bm'], old['bn'], old['wm']], 'to': [c['bm'], c['bn'], c['wm']]})
        if changed:
            records.append({'interval': k, 'estimated_us': [round(makespan(current), 3), round(best_cost, 3)],
                            'changed': changed})
    return phases, records


# Release step 14 ('ecta'): elementwise work on the CTAs that an interval's pinned matrix tiles leave idle.
ECTA_GRID = 128          # the planners' target grid (TARGET_GRID); the restriction is skipped at runtime on smaller grids
_ECTA_UNSAFE = ('smem', '__syncthreads', 'grid.', '::', 'atomic', '__shfl')


def ecta_eligible(phase):
    """A self-contained elementwise loop: CTAs only distribute its independent elements (any partition of the grid
    computes the same values); no shared memory, CTA/grid synchronization, helpers, atomics or warp exchange."""
    code = phase.get('code') or ''
    if phase.get('kind') != 'elementwise' or not code or any(t in code for t in _ECTA_UNSAFE):
        return False
    stripped = code.replace('blockIdx.x', '').replace('gridDim.x', '')
    return 'blockIdx' not in stripped and 'gridDim' not in stripped


def ecta_restrict(code, tiles):
    """Run ``code`` on CTAs [tiles, gridDim.x) (all CTAs when the grid is not larger than ``tiles``)."""
    body = code.replace('blockIdx.x', 'ecta_b_').replace('gridDim.x', 'ecta_g_')
    return (f'if (blockIdx.x >= {tiles}u || gridDim.x <= {tiles}u) {{\n'
            f'  const unsigned ecta_b_ = gridDim.x > {tiles}u ? blockIdx.x - {tiles}u : blockIdx.x;\n'
            f'  const unsigned ecta_g_ = gridDim.x > {tiles}u ? gridDim.x - {tiles}u : gridDim.x;\n'
            f'{body}\n}}')


def _per_task_us(phase):
    plan = phase.get('matrix_plan') or {}
    return float(plan.get('local_us', 0.0)) / max(1, int(plan.get('waves', 1) or 1))


def fp32_idle_placement(phases, members, pinned_tiles, *, policy='capacity'):
    """Step-14 speed: CTA placement of an ecta interval's FP32 small-tile products (popsc_fp32 engines).

    The interval's pinned TMA tiles occupy CTAs [0, pinned_tiles) (rotated back to back, one tile per CTA per pass).
    The donor rule confined every FP32 product to the CTAs those tiles leave idle; when the FP32 work is large relative
    to that idle capacity (many tasks, few idle CTAs) the interval waits for a handful of CTAs. Two placements are
    compared with the planners' own per-task estimates (body plan ``local_us / waves``):

    * ``idle``:  the exact idle slice [pinned_tiles, gridDim.x), tasks continuing round-robin across products;
    * ``whole``: the whole grid, rotated so the idle CTAs take each round's first tasks (busy CTAs pick up the rest
      after their tiles), tasks again continuing across products.

    ``policy='capacity'`` keeps the smaller estimated interval makespan (ties keep ``idle``, the donor behaviour);
    ``policy='whole'`` always takes the whole grid. Placement changes only which CTA runs a task: every task's
    arithmetic and every stored value are unchanged (no task accumulates into another's output).
    Returns ({phase: (mode, rotation)}, record)."""
    from .matrix_rules import TARGET_GRID
    fp32 = [i for i in members if phases[i]['kind'] == 'gemm'
            and str(phases[i].get('engine', '')).startswith('popsc_fp32')]
    if not fp32:
        return {}, None
    pinned = [i for i in members if phases[i]['kind'] == 'gemm' and any(
        t in phases[i].get('code', '') for t in ('/*VBASE*/', '/*B_VBASE*/', '/*C_VBASE*/'))]
    grids = [int((phases[i].get('matrix_plan') or {}).get('grid', 0)) for i in pinned]
    grid = max([g for g in grids if g > 0] or [TARGET_GRID])
    busy = [0.0] * grid
    offset = 0
    for i in pinned:
        cost, tiles = _per_task_us(phases[i]), int(phases[i].get('tiles', 0))
        for t in range(tiles):
            busy[(offset + t) % grid] += cost
        offset += tiles
    idle = grid - min(pinned_tiles, grid)
    work = [(i, int(phases[i].get('tiles', 0)), _per_task_us(phases[i])) for i in fp32]

    def simulate(mode):
        load = list(busy)
        first, count = (pinned_tiles, idle) if mode == 'idle' and idle > 0 else (0, grid)
        rot = 0 if mode == 'idle' else pinned_tiles
        rotations = {}
        for i, tasks, cost in work:
            rotations[i] = rot
            for task in range(tasks):
                # slice_of: CTA first + (me + rot) % count runs task index me (strided by count).
                load[first + (task + rot) % count] += cost
            rot += tasks
        return max(load), rotations
    idle_span, idle_rot = simulate('idle')
    whole_span, whole_rot = simulate('whole')
    mode = policy if policy in ('whole', 'idle') else 'whole' if whole_span < idle_span else 'idle'
    rotations = whole_rot if mode == 'whole' else idle_rot
    record = {'fp32_phases': [phases[i]['names'][0] for i in fp32], 'pinned_tiles': pinned_tiles,
              'model_grid': grid, 'fp32_tasks': sum(t for _, t, _ in work), 'mode': mode, 'policy': policy,
              'estimated_us': {'idle': round(idle_span, 3), 'whole': round(whole_span, 3)}}
    return {i: (mode, rotations[i]) for i in fp32}, record


def sink_to_idle(phases, intervals, layouts, pinned_tiles):
    """Move ecta-eligible elementwise phases to the interval just before their earliest reader when that interval's
    pinned matrix tiles leave CTAs idle (no complete-contract successor is crossed; values unchanged)."""
    from .execution_plan import dependency_successors
    successors = dependency_successors(phases, layouts)
    level = {i: k for k, members in enumerate(intervals) for i in members}
    load = [sum(pinned_tiles(i) for i in members) for members in intervals]
    moved = []
    for index in sorted(level, reverse=True):
        phase = phases[index]
        if phase.get('kind') != 'elementwise' or not phase.get('body') or phase.get('vprod_inline'):
            continue
        later = [level[s] for s in successors[index] if s in level]
        if not later:
            continue
        target = min(later) - 1
        if target <= level[index] or not 0 < load[target] < ECTA_GRID:
            continue
        moved.append({'phase': index, 'from': level[index], 'to': target})
        level[index] = target
    result = [[] for _ in intervals]
    for i, k in level.items():
        result[k].append(i)
    return [sorted(m) for m in result if m], moved


# LATCH-MOESUP-ROW ('pstream'): a memory-streaming row engine (one warp per row, rows distributed only by
# blockIdx/gridDim: the vocabulary gradient) shares the interval of an independent store-bound TMA product (short K,
# at least 2^28 outputs, e.g. another readout's logits) on a disjoint CTA slice: the product on [0, G * share), the
# stream on the rest. Placement keeps every complete-contract predecessor earlier and every successor later; each
# phase computes the same values. Standalone production shapes: the logits product scales with its CTA count, while
# the gradient stream draws more bandwidth per SM on fewer SMs (32 SMs: 12.9 ms, 64: 7.0 ms, 132: 4.8 ms).
PAIR_SHARE = 700          # per mille of the grid given to the product
PAIR_MIN_OUTPUT = 2 ** 28
PAIR_MAX_K = 8 * 64
_PAIR_CALL = re.compile(r'tma_phase_inline<[^>]*>\(g, smem, 0, 1, 1, (?:/\*[BC]_VBASE\*/0?|0)\)')
# LATCH-MOESUP-ROW: a statically partitioned streaming row pass (each warp a fixed share of rows) sharing a barrier
# interval with a long-attention phase (per-CTA work items of unequal causal cost) runs after the attention items of
# each CTA, so the interval ends at the slowest CTA's attention share plus a full row share (measured: attention
# forward 0.81 ms + vocabulary gradient 4.63 ms took 5.62 ms together). Such row passes get their own interval
# directly after (they are independent of the interval's other phases, and their readers are later intervals).
SPLIT_STREAM_ROWS = True
_SPLIT_ROW_KINDS = ('vocab_gradient',)


def split_stream_rows(phases, intervals):
    out, records = [], []
    for interval in intervals:
        attention = [i for i in interval if str(phases[i].get('kind', '')).startswith('attention_long')]
        rows = [i for i in interval if phases[i].get('kind') in _SPLIT_ROW_KINDS]
        if attention and rows and len(rows) < len(interval):
            out.append([i for i in interval if i not in rows])
            out.append(list(rows))
            records.append({'rows': [phases[i]['names'][0] for i in rows],
                            'attention': [phases[i]['names'][0] for i in attention]})
        else:
            out.append(interval)
    return out, records


_STREAM_ENGINES = ('vocab_gradient_warp_pipelined',)


def _stream_safe(phase):
    code = phase.get('code') or ''
    return (phase.get('engine') in _STREAM_ENGINES and bool(code)
            and not any(t in code for t in ('smem', '__syncthreads', 'grid.', 'atomic')))


def pair_streams(phases, intervals, layouts):
    """Return (intervals, pairs, records): pairs maps a stream phase to its product phase (see 'pstream').

    A stream is placed by its root-level data hazards (RAW, WAR, WAW) only: it has no internal grid synchronization,
    counters or global effects, so the program-order edges that grid-fence bodies add to every earlier body carry no
    hazard for it; it never joins an interval that holds such a fence body."""
    from .execution_plan import requires_grid_fence
    root = lambda name: layouts[name].root if name in layouts else name
    rroots = [{root(r) for r in p.get('reads', ())} for p in phases]
    wroots = [{root(w) for w in p.get('writes', ())} | {root(m) for m in p.get('internal_mutations', ())}
              for p in phases]

    def hazards(i, j):
        return bool(wroots[i] & (rroots[j] | wroots[j]) or rroots[i] & wroots[j])
    level = {i: k for k, members in enumerate(intervals) for i in members}
    fenced = {k for k, members in enumerate(intervals) if any(requires_grid_fence(phases[i]) for i in members)}

    def product(i):
        p = phases[i]
        spec = p.get('matrix_spec') or {}
        if p.get('kind') != 'gemm' or len(_PAIR_CALL.findall(p.get('code') or '')) != 1:
            return False
        m, n, k = (int(spec.get(x, 0) or 0) for x in ('m', 'n', 'k'))
        return int(spec.get('batch', 1) or 1) == 1 and m * n >= PAIR_MIN_OUTPUT and 0 < k <= PAIR_MAX_K

    hosts = {}
    for k, members in enumerate(intervals):
        big = [i for i in members if product(i)]
        if len(big) == 1:
            hosts[k] = big[0]
    fences = [i for i in range(len(phases)) if i in level and requires_grid_fence(phases[i])]

    def movable(i):
        # Counter/scratch bodies (e.g. column splits) stay ordered after their initializer by the scratch root's
        # write/mutation hazards; only grid fences and global effects are refused.
        p = phases[i]
        return not (requires_grid_fence(p) or p.get('global_effects'))

    def cascade(index, k):
        """New levels when the stream moves to interval k and every data descendant follows (None if refused)."""
        moved = {index: k}
        for j in range(index + 1, len(phases)):
            if j not in level:
                continue
            bound = max((moved[i] + 1 for i in moved if hazards(i, j)), default=-1)
            if bound > level[j]:
                if not movable(j) or bound >= len(intervals) or bound in fenced:
                    return None
                moved[j] = bound
        for j, new in moved.items():   # keep every grid-fence body after all bodies before it in program order
            if any(f > j and level[f] <= new for f in fences):
                return None
        return moved

    pairs, records, taken = {}, [], set()
    for index in sorted(level):
        if not _stream_safe(phases[index]) or requires_grid_fence(phases[index]):
            continue
        earliest = 1 + max((level[j] for j in range(index) if j in level and hazards(j, index)), default=-1)
        plan = None
        for k in sorted(hosts):
            if k < earliest or k <= level[index] or k in taken or k in fenced or hazards(index, hosts[k]):
                continue
            plan = cascade(index, k)
            if plan is not None:
                break
        if plan is None:
            continue
        taken.add(k)
        records.append({'stream': phases[index]['names'][0], 'product': phases[hosts[k]]['names'][0],
                        'from': level[index], 'to': k, 'share_per_mille': PAIR_SHARE,
                        'descendants_moved': len(plan) - 1})
        level.update(plan)
        pairs[index] = hosts[k]
    if not pairs:
        return intervals, {}, []
    result = [[] for _ in intervals]
    for i, k in level.items():
        result[k].append(i)
    return [sorted(m) for m in result if m], pairs, records


def pair_restrict(code, share):
    """Run ``code`` on the CTAs [G * share / 1000, G) that the paired product's slice leaves (slice_of bounds)."""
    body = code.replace('blockIdx.x', 'pair_b_').replace('gridDim.x', 'pair_g_')
    return (f'{{ const unsigned pair_0_ = (unsigned)((unsigned long long)gridDim.x * {share}ULL / 1000ULL);\n'
            f'if (blockIdx.x >= pair_0_) {{\n'
            f'  const unsigned pair_b_ = blockIdx.x - pair_0_, pair_g_ = gridDim.x - pair_0_;\n'
            f'{body}\n}} }}')


def _merge_elementwise_chains_reference(phases, layouts, *, ident, index, inplace_roots):
    """Release step 14 ('emerge'): a later elementwise phase B joins an earlier elementwise phase A of the same shape
    when B reads A's results at the identity index and every other operand of B is produced by a phase before A at a
    lower dependency level (so A is never delayed). B's lines run after A's at the same element, reading A's register
    values; program-order adjacency no longer decides fusion. Expressions and roundings are unchanged.
    Returns (phases, records)."""
    roots = {name: layout.root for name, layout in layouts.items()}
    root = lambda name: roots.get(name, name)
    aliases, layout_order = {}, {}
    for position, name in enumerate(layouts):
        aliases.setdefault(root(name), []).append(name)
        layout_order[name] = position
    # These facts survive every merge. Rechecking all names in every growing
    # chain on each restart makes large optimizer graphs cubic to traverse.
    plain = [(p['kind'] == 'elementwise' and bool(p.get('body')) and len(p['body']) == len(p['names'])
              and not p.get('vprod_inline') and not any(root(n) in inplace_roots for n in p['names']))
             for p in phases]
    reads = [tuple(root(r) for r in p['reads']) for p in phases]
    writes = [tuple(root(w) for w in p['writes']) for p in phases]
    records = []
    changed = True
    while changed:
        changed = False
        writer, level = {}, []
        for k, phase in enumerate(phases):
            level.append(1 + max((level[writer[r]] for r in reads[k] if r in writer), default=-1))
            for w in writes[k]:
                writer[w] = k
        for b, phase_b in enumerate(phases):
            if not plain[b]:
                continue
            shape = tuple(phase_b['shape'])
            sources = {writer[r] for r in reads[b] if r in writer and writer[r] != b}
            for a in sorted(sources, reverse=True):
                phase_a = phases[a]
                if a >= b or not plain[a] or tuple(phase_a['shape']) != shape:
                    continue
                own_a = set(writes[a])
                # When A consumes a matrix product, B's other operands must precede that product's level too, so a
                # later epilogue pass (which admits A only on that condition) still absorbs the merged chain.
                bound = level[a] - (1 if any(r in writer and phases[writer[r]]['kind'] == 'gemm'
                                             for r in reads[a]) else 0)
                own_b = set(writes[b])
                ok = True
                for r, y in zip(phase_b['reads'], reads[b]):
                    if y in own_b:
                        continue
                    if y in own_a:
                        if not _identity(layouts[r], shape):
                            ok = False
                    elif y in writer and (writer[y] >= a or level[writer[y]] >= bound):
                        ok = False
                if not ok:
                    continue
                names_a = sorted((n for y in own_a for n in aliases.get(y, ())),
                                 key=lambda n: (-len(n), layout_order[n]))
                lines = []
                for line in phase_b['body']:
                    head, sep, tail = line.partition('; ')
                    for n in names_a:
                        try:
                            text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                        except ValueError:
                            continue
                        head = head.replace(text, f'r_{root(n)}')
                        # Level splitting materializes an earlier register at
                        # the loop index, rather than its canonical layout
                        # expression (for scalars that expression is 0).
                        # Rejoining the pieces must restore that register read
                        # before the now-private producer store is elided.
                        if _identity(layouts[n], shape):
                            head = head.replace(f'{ident(n)}[i]', f'r_{root(n)}')
                    lines.append(head + sep + tail)
                merged = dict(phase_a)
                merged.update(names=[*phase_a['names'], *phase_b['names']],
                              targets=[*phase_a['targets'], *phase_b['targets']],
                              reads=list(dict.fromkeys([*phase_a['reads'], *(r for r in phase_b['reads']
                                                                            if root(r) not in own_a)])),
                              writes=[*phase_a['writes'], *phase_b['writes']], body=[*phase_a['body'], *lines])
                if not records:
                    phases = list(phases)  # Keep the caller's list immutable.
                phases[a] = merged
                phases.pop(b)
                reads[a] = tuple(root(r) for r in merged['reads'])
                writes[a] = tuple(root(w) for w in merged['writes'])
                # A and B were both plain; their concatenated body/names stay
                # plain and retain A's shape and non-inline metadata.
                plain.pop(b)
                reads.pop(b)
                writes.pop(b)
                records.append({'into': phase_a['names'][0], 'merged': list(phase_b['names'])})
                changed = True
                break
            if changed:
                break
    return phases, records


# LATCH-E2TTT-OUTER: the reference above merges one pair, then recomputes every writer/level and rescans from the
# first phase (O(phases x merges): 409 of 966 s building an 8-layer E2-TTT update). The incremental form below makes
# the same merges in the same order (False restores the reference for comparisons).
MERGE_INCREMENTAL = True


def merge_elementwise_chains(phases, layouts, *, ident, index, inplace_roots):
    """Release step 14 ('emerge'); same contract and results as ``_merge_elementwise_chains_reference``.

    Exactness: phases keep stable ids (their input positions); each root keeps the ordered list of its writers, so
    the reference's global last writer (``writer``) and its forward 'latest writer before k' (``level``) are both
    recovered exactly. A merge of B into A leaves every decision for phases before A unchanged (their operands'
    writers and levels precede A, and every value moved to A is written after them), so scanning resumes at A rather
    than at the first phase. Levels are computed once in scan order: a merge cannot change the level of A (B's other
    operands are below A's level) or of any phase between A and B (none reads B); later levels are computed after
    the merge. Should a merged phase's level differ (e.g. B reading its own in-place value), all levels are rebuilt.
    """
    if not MERGE_INCREMENTAL:
        return _merge_elementwise_chains_reference(phases, layouts, ident=ident, index=index,
                                                   inplace_roots=inplace_roots)
    if MERGE_STALE:
        result = _merge_elementwise_chains_stale(phases, layouts, ident=ident, index=index,
                                                 inplace_roots=inplace_roots)
        if MERGE_CHECK and result != _merge_elementwise_chains_incremental(phases, layouts, ident=ident, index=index,
                                                                           inplace_roots=inplace_roots):
            raise AssertionError('stale-set merge_elementwise_chains differs from the incremental form')
        return result
    return _merge_elementwise_chains_incremental(phases, layouts, ident=ident, index=index,
                                                 inplace_roots=inplace_roots)


# LATCH-BUILD-600: after each merge the incremental form re-examines every phase from A to the furthest phase
# examined so far, although almost none of them can decide differently (260 of 1,239 s in the 4-layer 8,192-token
# E2-TTT build). The stale-set form below re-examines only phases whose decision inputs a merge changed, in the same
# order, so it makes the same merges in the same order (False keeps the incremental form for comparisons).
MERGE_STALE = True
MERGE_CHECK = os.environ.get('LATCH_MERGE_CHECK') == '1'   # validation runs: also run the incremental form, compare


def _merge_elementwise_chains_incremental(phases, layouts, *, ident, index, inplace_roots):
    import bisect
    roots = {name: layout.root for name, layout in layouts.items()}
    root = lambda name: roots.get(name, name)
    aliases, layout_order = {}, {}
    for position, name in enumerate(layouts):
        aliases.setdefault(root(name), []).append(name)
        layout_order[name] = position
    original = phases
    phases = list(phases)                 # the caller's list stays unchanged
    n = len(phases)
    plain = [(p['kind'] == 'elementwise' and bool(p.get('body')) and len(p['body']) == len(p['names'])
              and not p.get('vprod_inline') and not any(root(n_) in inplace_roots for n_ in p['names']))
             for p in phases]
    reads = [tuple(root(r) for r in p['reads']) for p in phases]
    writes = [tuple(root(w) for w in p['writes']) for p in phases]
    alive = [True] * n
    writers = {}                          # root -> ascending ids of live writers
    for k in range(n):
        for w in writes[k]:
            lst = writers.setdefault(w, [])
            if not lst or lst[-1] != k:
                if k not in lst:
                    lst.append(k)

    def last_writer(r):
        lst = writers.get(r)
        return lst[-1] if lst else None

    def writer_before(r, k):
        lst = writers.get(r)
        if not lst:
            return None
        i = bisect.bisect_left(lst, k)
        return lst[i - 1] if i else None

    level = [None] * n
    frontier = [0]                        # levels are valid for every live id < frontier[0]

    def level_of(k):
        return 1 + max((level[w] for w in (writer_before(r, k) for r in reads[k]) if w is not None), default=-1)

    def advance(limit):
        while frontier[0] <= limit and frontier[0] < n:
            k = frontier[0]
            if alive[k]:
                level[k] = level_of(k)
            frontier[0] += 1

    def rebuild_levels():
        for k in range(frontier[0]):
            if alive[k]:
                level[k] = level_of(k)

    records = []
    b = 0
    while b < n:
        if not alive[b]:
            b += 1
            continue
        advance(b)
        if not plain[b]:
            b += 1
            continue
        phase_b = phases[b]
        shape = tuple(phase_b['shape'])
        sources = set()
        for r in reads[b]:
            w = last_writer(r)
            if w is not None and w != b:
                sources.add(w)
        merged_into = None
        for a in sorted(sources, reverse=True):
            phase_a = phases[a]
            if a >= b or not plain[a] or tuple(phase_a['shape']) != shape:
                continue
            own_a = set(writes[a])
            bound = level[a] - (1 if any(last_writer(r) is not None and phases[last_writer(r)]['kind'] == 'gemm'
                                         for r in reads[a]) else 0)
            own_b = set(writes[b])
            ok = True
            for r, y in zip(phase_b['reads'], reads[b]):
                if y in own_b:
                    continue
                if y in own_a:
                    if not _identity(layouts[r], shape):
                        ok = False
                else:
                    w = last_writer(y)
                    if w is not None and (w >= a or level[w] >= bound):
                        ok = False
            if not ok:
                continue
            names_a = sorted((n_ for y in own_a for n_ in aliases.get(y, ())),
                             key=lambda n_: (-len(n_), layout_order[n_]))
            lines = []
            for line in phase_b['body']:
                head, sep, tail = line.partition('; ')
                for n_ in names_a:
                    try:
                        text = f"{ident(n_)}[{index(layouts[n_], shape, 'i')}]"
                    except ValueError:
                        continue
                    head = head.replace(text, f'r_{root(n_)}')
                    if _identity(layouts[n_], shape):
                        head = head.replace(f'{ident(n_)}[i]', f'r_{root(n_)}')
                lines.append(head + sep + tail)
            merged = dict(phase_a)
            merged.update(names=[*phase_a['names'], *phase_b['names']],
                          targets=[*phase_a['targets'], *phase_b['targets']],
                          reads=list(dict.fromkeys([*phase_a['reads'], *(r for r in phase_b['reads']
                                                                        if root(r) not in own_a)])),
                          writes=[*phase_a['writes'], *phase_b['writes']], body=[*phase_a['body'], *lines])
            phases[a] = merged
            for w in set(writes[b]):
                lst = writers[w]
                lst.remove(b)
                if a not in lst:
                    bisect.insort(lst, a)
            alive[b] = False
            reads[a] = tuple(root(r) for r in merged['reads'])
            writes[a] = tuple(root(w) for w in merged['writes'])
            records.append({'into': phase_a['names'][0], 'merged': list(phase_b['names'])})
            if level_of(a) != level[a]:
                rebuild_levels()
            merged_into = a
            break
        b = merged_into if merged_into is not None else b + 1
    if not records:
        return original, records
    return [p for k, p in enumerate(phases) if alive[k]], records


def _merge_elementwise_chains_stale(phases, layouts, *, ident, index, inplace_roots):
    """``_merge_elementwise_chains_incremental`` without its redundant re-examinations; identical results.

    The incremental scan examines phase ids in ascending order; a merge of B into A makes it resume at A, i.e. re-
    examine every live id from A up to ``front`` (the furthest id examined so far) before continuing beyond. An
    examination is a pure function of the scan state, and an examination that finds no merge changes nothing. So a
    re-examination can only differ from the phase's previous (merge-free) examination if a merge since then changed
    one of its inputs. A merge of B into A changes exactly: A's phase (body, reads, writes), the writer lists of the
    roots W that B wrote (B removed, A inserted), B's liveness, and - only when A's level changes, which rebuilds all
    levels - levels. Examining phase k reads: the last writers of k's reads (changed only if k reads a root in W),
    for each source A' of k (a last writer of one of k's reads) its phase, writes and level and the last writers of
    A''s reads (changed only if A' is A, i.e. k reads a root A writes, or A' reads a root in W), and the last
    writers / levels of k's other operands (W again). Every examined id that may be affected is marked stale: A,
    the readers of A's writes (which include W), and the readers of the writes of every reader of W; after a level
    rebuild, every id. A restart at A re-examines exactly the stale live ids >= A in ascending order (the others
    would repeat their previous merge-free examination), then continues beyond ``front`` as before; stale ids below
    A stay stale until a later restart reaches them, as the incremental scan would re-examine them only then.
    """
    import bisect
    import heapq
    roots = {name: layout.root for name, layout in layouts.items()}
    root = lambda name: roots.get(name, name)
    aliases, layout_order = {}, {}
    for position, name in enumerate(layouts):
        aliases.setdefault(root(name), []).append(name)
        layout_order[name] = position
    original = phases
    phases = list(phases)
    n = len(phases)
    plain = [(p['kind'] == 'elementwise' and bool(p.get('body')) and len(p['body']) == len(p['names'])
              and not p.get('vprod_inline') and not any(root(n_) in inplace_roots for n_ in p['names']))
             for p in phases]
    reads = [tuple(root(r) for r in p['reads']) for p in phases]
    writes = [tuple(root(w) for w in p['writes']) for p in phases]
    alive = [True] * n
    writers = {}
    for k in range(n):
        for w in writes[k]:
            lst = writers.setdefault(w, [])
            if not lst or lst[-1] != k:
                if k not in lst:
                    lst.append(k)
    readers = {}                          # root -> live ids whose reads include it
    for k in range(n):
        for r in reads[k]:
            readers.setdefault(r, set()).add(k)

    def last_writer(r):
        lst = writers.get(r)
        return lst[-1] if lst else None

    def writer_before(r, k):
        lst = writers.get(r)
        if not lst:
            return None
        i = bisect.bisect_left(lst, k)
        return lst[i - 1] if i else None

    level = [None] * n
    frontier = [0]

    def level_of(k):
        return 1 + max((level[w] for w in (writer_before(r, k) for r in reads[k]) if w is not None), default=-1)

    def advance(limit):
        while frontier[0] <= limit and frontier[0] < n:
            k = frontier[0]
            if alive[k]:
                level[k] = level_of(k)
            frontier[0] += 1

    def rebuild_levels():
        for k in range(frontier[0]):
            if alive[k]:
                level[k] = level_of(k)

    def examine(b):
        """The incremental scan's examination of live id b: the id merged into, or None (state unchanged)."""
        advance(b)
        if not plain[b]:
            return None
        phase_b = phases[b]
        shape = tuple(phase_b['shape'])
        sources = set()
        for r in reads[b]:
            w = last_writer(r)
            if w is not None and w != b:
                sources.add(w)
        for a in sorted(sources, reverse=True):
            phase_a = phases[a]
            if a >= b or not plain[a] or tuple(phase_a['shape']) != shape:
                continue
            own_a = set(writes[a])
            bound = level[a] - (1 if any(last_writer(r) is not None and phases[last_writer(r)]['kind'] == 'gemm'
                                         for r in reads[a]) else 0)
            own_b = set(writes[b])
            ok = True
            for r, y in zip(phase_b['reads'], reads[b]):
                if y in own_b:
                    continue
                if y in own_a:
                    if not _identity(layouts[r], shape):
                        ok = False
                else:
                    w = last_writer(y)
                    if w is not None and (w >= a or level[w] >= bound):
                        ok = False
            if not ok:
                continue
            names_a = sorted((n_ for y in own_a for n_ in aliases.get(y, ())),
                             key=lambda n_: (-len(n_), layout_order[n_]))
            lines = []
            for line in phase_b['body']:
                head, sep, tail = line.partition('; ')
                for n_ in names_a:
                    try:
                        text = f"{ident(n_)}[{index(layouts[n_], shape, 'i')}]"
                    except ValueError:
                        continue
                    head = head.replace(text, f'r_{root(n_)}')
                    if _identity(layouts[n_], shape):
                        head = head.replace(f'{ident(n_)}[i]', f'r_{root(n_)}')
                lines.append(head + sep + tail)
            merged = dict(phase_a)
            merged.update(names=[*phase_a['names'], *phase_b['names']],
                          targets=[*phase_a['targets'], *phase_b['targets']],
                          reads=list(dict.fromkeys([*phase_a['reads'], *(r for r in phase_b['reads']
                                                                        if root(r) not in own_a)])),
                          writes=[*phase_a['writes'], *phase_b['writes']], body=[*phase_a['body'], *lines])
            phases[a] = merged
            moved = set(writes[b])
            for w in moved:
                lst = writers[w]
                lst.remove(b)
                if a not in lst:
                    bisect.insort(lst, a)
            alive[b] = False
            for r in reads[b]:
                readers[r].discard(b)
            reads[a] = tuple(root(r) for r in merged['reads'])
            writes[a] = tuple(root(w) for w in merged['writes'])
            for r in reads[a]:
                readers.setdefault(r, set()).add(a)
            records.append({'into': phase_a['names'][0], 'merged': list(phase_b['names'])})
            rebuilt = level_of(a) != level[a]
            if rebuilt:
                rebuild_levels()
            return a, moved, rebuilt
        return None

    records = []
    stale = set()                         # examined ids whose inputs changed since their last examination
    queue, queued = [], set()             # ids to re-examine (all <= front), ascending
    front = -1
    while True:
        if queue:
            b = heapq.heappop(queue)
            queued.discard(b)
        elif front + 1 < n:
            front += 1
            b = front
        else:
            break
        if not alive[b]:
            continue
        stale.discard(b)
        result = examine(b)
        if result is None:
            continue
        a, moved, rebuilt = result
        if rebuilt:
            affected = range(front + 1)
        else:
            affected = {a}
            for y in writes[a]:
                affected |= readers.get(y, set())
            for y in moved:
                for q in readers.get(y, ()):
                    for z in writes[q]:
                        affected |= readers.get(z, set())
        stale.update(k for k in affected if k <= front and alive[k])
        for k in [k for k in stale if k >= a]:
            stale.discard(k)
            if k not in queued:
                queued.add(k)
                heapq.heappush(queue, k)
    if not records:
        return original, records
    return [p for k, p in enumerate(phases) if alive[k]], records


def sink_multi_reader_chains(phases, layouts, *, ident, index, inplace=None):
    """LATCH-SELECTIVE-10 ('cmulti'): ``csink`` for a plain elementwise phase A read by k >= 2 phases B1..Bk.

    Every Bi must be a plain elementwise phase of A's shape and dtype reading A's results only at the identity
    index, no Bi may read another Bj's results, and no phase between A and the last Bi (in phase order) other than
    the Bi may read a merged result or write anything A or a Bi reads (so moving A and the earlier Bi to the last
    Bi's position delays no other value and reorders no access). The merged loop runs A's lines, then B1..Bk's in
    phase order, each Bi reading A's registers: per element, the same expressions on the same operand values.
    An in-place overwrite of a value A reads must be one of the merged results (A's lines precede it), as for
    csink. Typical case: an optimizer's new moments read by both the parameter update and the participation
    select that commits the moments in place. Returns (phases, records)."""
    roots = {name: layout.root for name, layout in layouts.items()}
    root = lambda name: roots.get(name, name)
    aliases, layout_order = {}, {}
    for position, name in enumerate(layouts):
        aliases.setdefault(root(name), []).append(name)
        layout_order[name] = position
    inplace = dict(inplace or {})
    plain = lambda p: (p['kind'] == 'elementwise' and bool(p.get('body')) and len(p['body']) == len(p['names'])
                       and not p.get('vprod_inline') and not p.get('counter_effects')
                       and not p.get('internal_mutations'))
    order = list(range(len(phases)))
    current = dict(enumerate(phases))
    position = {k: k for k in order}
    readers, writers = {}, {}
    for k, phase in current.items():
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
        for w in phase['writes']:
            writers.setdefault(root(w), set()).add(k)
    records, refused = [], {}

    def refuse(reason):
        refused[reason] = refused.get(reason, 0) + 1

    for a in reversed(list(order)):
        if a not in current:
            continue
        phase_a = current[a]
        if not plain(phase_a):
            continue
        own_a = {root(w) for w in phase_a['writes']}
        bs = sorted(set().union(*(readers.get(w, set()) for w in own_a)) - {a}, key=lambda k: position[k])
        if len(bs) < 2:
            continue
        if position[bs[0]] < position[a]:
            refuse('reader before producer')
            continue
        group = [current[k] for k in bs]
        shape = tuple(phase_a['shape'])
        if any(not plain(p) or tuple(p['shape']) != shape or p.get('dtype', 'float32') != phase_a.get('dtype', 'float32')
               for p in group):
            refuse('reader form')
            continue
        own_b = [{root(w) for w in p['writes']} for p in group]
        if any(root(r) in own_b[j] for i_, p in enumerate(group) for j in range(len(group)) if i_ != j
               for r in p['reads']):
            refuse('dependent readers')
            continue
        members = {a, *bs}
        hi = position[bs[-1]]
        # Every moved body (A and the earlier readers) moves to the last reader's position: no other phase between
        # a moved body and that position may write what the moved body reads or read what it writes.
        crossing = False
        for k in [a, *bs[:-1]]:
            reads_k = {root(r) for r in current[k]['reads']}
            writes_k = {root(w) for w in current[k]['writes']}
            touched = set().union(*(writers.get(r, set()) for r in reads_k), *(readers.get(w, set()) for w in writes_k))
            if any(position[k] < position[q] < hi for q in touched - members):
                crossing = True
                break
        if crossing:
            refuse('intervening access')
            continue
        written = {root(w) for k in members for w in current[k]['writes']}
        if any(root(r) in inplace and inplace[root(r)] not in written for r in phase_a['reads']):
            refuse('in-place target written elsewhere')
            continue
        if any(root(r) in own_a and not _identity(layouts[r], shape) for p in group for r in p['reads']):
            refuse('non-identity read')
            continue
        names_a = sorted((n for y in own_a for n in aliases.get(y, ())), key=lambda n: (-len(n), layout_order[n]))
        bodies, ok = [], True
        for p in group:
            lines = []
            for line in p['body']:
                head, sep, tail = line.partition('; ')
                for n in names_a:
                    try:
                        text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                    except ValueError:
                        continue
                    head = head.replace(text, f'r_{root(n)}')
                    if _identity(layouts[n], shape):
                        head = head.replace(f'{ident(n)}[i]', f'r_{root(n)}')
                lines.append(head + sep + tail)
            if any(f"{ident(n)}[" in line.partition('; ')[0] for line in lines for n in names_a):
                ok = False
                break
            bodies.append(lines)
        if not ok:
            refuse('memory read remains')
            continue
        last = bs[-1]
        merged = dict(group[-1])
        merged.update(names=[*phase_a['names'], *(n for p in group for n in p['names'])],
                      targets=[*phase_a['targets'], *(t for p in group for t in p['targets'])],
                      reads=list(dict.fromkeys([*phase_a['reads'],
                                                *(r for p in group for r in p['reads'] if root(r) not in own_a)])),
                      writes=[*phase_a['writes'], *(w for p in group for w in p['writes'])],
                      body=[*phase_a['body'], *(line for lines in bodies for line in lines)])
        if 'registers' in phase_a or any('registers' in p for p in group):
            merged['registers'] = {**phase_a.get('registers', {}),
                                   **{k: v for p in group for k, v in p.get('registers', {}).items()}}
        for k in members:
            for r in current[k]['reads']:
                readers.get(root(r), set()).discard(k)
            for w in current[k]['writes']:
                writers.get(root(w), set()).discard(k)
            if k != last:
                del current[k]
                order.remove(k)
        current[last] = merged
        for r in merged['reads']:
            readers.setdefault(root(r), set()).add(last)
        for w in merged['writes']:
            writers.setdefault(root(w), set()).add(last)
        for w in own_a:
            readers.get(w, set()).discard(last)
        records.append({'sunk': list(phase_a['names'][:4]), 'readers': [p['names'][0] for p in group]})
    if refused:
        records.append({'refused_multi_reader_candidates': refused})
    return [current[k] for k in order], records


def sink_single_reader_chains(phases, layouts, *, ident, index, inplace=None):
    """LATCH-TINYMOE-25 ('csink'): a plain elementwise phase A whose results are read by exactly one other phase B,
    itself a plain elementwise phase of the same shape and dtype that reads them only at the identity index, joins B
    at B's position: A's lines run first inside B's loop and B reads A's registers. Nothing else reads A, so the
    move delays no other value; it removes the store/reload of A's unkept results (the ordinary dead-store pass
    elides them) and the dependency interval A would otherwise occupy. A's kept results are still stored. A never
    moves past an in-place overwrite of a value it reads: ``inplace`` maps each in-place target root to the
    root of the result written over it, and A may read a target only when A itself or B writes that result
    (A's lines precede B's in the merged loop). Expressions and roundings are unchanged.
    Returns (phases, records)."""
    roots = {name: layout.root for name, layout in layouts.items()}
    root = lambda name: roots.get(name, name)
    aliases, layout_order = {}, {}
    for position, name in enumerate(layouts):
        aliases.setdefault(root(name), []).append(name)
        layout_order[name] = position
    inplace = dict(inplace or {})
    plain = lambda p: (p['kind'] == 'elementwise' and bool(p.get('body')) and len(p['body']) == len(p['names'])
                       and not p.get('vprod_inline') and not p.get('counter_effects')
                       and not p.get('internal_mutations'))
    order = list(range(len(phases)))
    current = dict(enumerate(phases))
    readers = {}
    for k, phase in current.items():
        for r in phase['reads']:
            readers.setdefault(root(r), set()).add(k)
    records, refused = [], {}
    for a in reversed(list(order)):
        phase_a = current[a]
        if not plain(phase_a):
            continue
        own_a = {root(w) for w in phase_a['writes']}
        targets = set().union(*(readers.get(w, set()) for w in own_a)) - {a}
        if len(targets) != 1:
            continue
        b = next(iter(targets))
        phase_b = current[b]
        written = {w for p in (phase_a, phase_b) for w in p['writes']}
        written |= {root(w) for w in written}
        if any(root(r) in inplace and inplace[root(r)] not in written for r in phase_a['reads']):
            refused['in-place target written elsewhere'] = refused.get('in-place target written elsewhere', 0) + 1
            continue
        if (order.index(b) <= order.index(a) or not plain(phase_b)
                or tuple(phase_b['shape']) != tuple(phase_a['shape'])
                or phase_b.get('dtype', 'float32') != phase_a.get('dtype', 'float32')):
            refused['reader form'] = refused.get('reader form', 0) + 1
            if len(refused.setdefault('samples', [])) < 6 and len(phase_a['names']) > 1:
                refused['samples'].append(dict(
                    a=phase_a['names'][:4], b=phase_b['names'][:4], b_kind=phase_b.get('kind'),
                    order=order.index(b) > order.index(a), b_plain=bool(plain(phase_b)),
                    b_body=len(phase_b.get('body') or ()), b_names=len(phase_b.get('names', ())),
                    b_inline=bool(phase_b.get('vprod_inline')), shapes=[phase_a.get('shape'), phase_b.get('shape')],
                    dtypes=[phase_a.get('dtype'), phase_b.get('dtype')]))
            continue
        shape = tuple(phase_b['shape'])
        if any(root(r) in own_a and not _identity(layouts[r], shape) for r in phase_b['reads']):
            refused['non-identity read'] = refused.get('non-identity read', 0) + 1
            continue
        names_a = sorted((n for y in own_a for n in aliases.get(y, ())), key=lambda n: (-len(n), layout_order[n]))
        lines = []
        for line in phase_b['body']:
            head, sep, tail = line.partition('; ')
            for n in names_a:
                try:
                    text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                except ValueError:
                    continue
                head = head.replace(text, f'r_{root(n)}')
                if _identity(layouts[n], shape):
                    head = head.replace(f'{ident(n)}[i]', f'r_{root(n)}')
            lines.append(head + sep + tail)
        if any(f"{ident(n)}[" in line.partition('; ')[0] for line in lines for n in names_a):
            refused['memory read remains'] = refused.get('memory read remains', 0) + 1
            continue  # an A value would still be read through memory: keep both phases
        merged = dict(phase_b)
        merged.update(names=[*phase_a['names'], *phase_b['names']],
                      targets=[*phase_a['targets'], *phase_b['targets']],
                      reads=list(dict.fromkeys([*phase_a['reads'],
                                                *(r for r in phase_b['reads'] if root(r) not in own_a)])),
                      writes=[*phase_a['writes'], *phase_b['writes']], body=[*phase_a['body'], *lines])
        if 'registers' in phase_a or 'registers' in phase_b:
            merged['registers'] = {**phase_a.get('registers', {}), **phase_b.get('registers', {})}
        current[b] = merged
        del current[a]
        order.remove(a)
        for r in phase_a['reads']:
            readers.setdefault(root(r), set()).discard(a)
            readers[root(r)].add(b)
        for w in own_a:
            readers.pop(w, None)
        records.append({'into': phase_b['names'][0], 'sunk': list(phase_a['names'])})
    if refused:
        records.append({'refused_single_reader_candidates': refused})
    return [current[k] for k in order], records



def _vector_epilogue_tma(lines, types):
    """Release step 14: the p5c vector-epilogue rewrite for the staged TMA epilogue (one 8-element chunk per call):
    identity-indexed BF16 and FP32 side inputs load as 16-byte vectors before the chunk is evaluated, identity
    stores leave as 16-byte vectors after it. Same values; only data movement changes."""
    stores, loads, floads = {}, {}, {}
    out = []
    for line in lines:
        match = _EPI_STORE.search(line)
        if match and types.get(match.group(1)) in ('bfloat16', 'float32'):
            name = match.group(1)
            k = stores.setdefault(name, len(stores))
            line = line[:match.start()] + f'; s8{k}_[j] = {match.group(2)};'
        head, sep, tail = line.partition(' = ')

        def load(m):
            name = m.group(1)
            if m.group(0) not in (f'{name}[(i)]', f'{name}[i]'):
                return m.group(0)
            if types.get(name) == 'bfloat16':
                k = loads.setdefault(name, len(loads))
                return f'tmaepi::upkb(p8l{k}_, j)'
            if types.get(name) == 'float32':
                k = floads.setdefault(name, len(floads))
                return f'p8f{k}_[j]'
            return m.group(0)
        out.append(head + sep + _EPI_LOAD.sub(load, tail) if sep else line)
    if not stores and not loads and not floads:
        return None
    ctype = {'bfloat16': '__nv_bfloat16', 'float32': 'float'}
    prelude = ' '.join([f'uint4 p8l{k}_;' for k in loads.values()] + [f'float p8f{k}_[8];' for k in floads.values()])
    ld = ('[&](int e, unsigned i0) { ' + ' '.join(
        [f'p8l{k}_ = __ldg(reinterpret_cast<const uint4*>({name} + i0));' for name, k in loads.items()]
        + [f'*reinterpret_cast<float4*>(&p8f{k}_[0]) = __ldg(reinterpret_cast<const float4*>({name} + i0)); '
           f'*reinterpret_cast<float4*>(&p8f{k}_[4]) = __ldg(reinterpret_cast<const float4*>({name} + i0 + 4));'
           for name, k in floads.items()]) + ' }')
    decl = ' '.join(f'{ctype[types[name]]} s8{k}_[8];' for name, k in stores.items())
    tail = ' '.join(f'tmaepi::st8({name} + i0, s8{k}_);' for name, k in stores.items())
    return prelude, ld, decl, out, tail
