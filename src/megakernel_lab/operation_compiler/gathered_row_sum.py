"""LATCH-MOE-EMA-FAST: one warp-per-row pass for a gathered summand and its pinned vector4 row sum.

    G = index_select(T, 0, idx)                 FP32 rows (index_select_warp_rows), read only by E
    E: elementwise over G and other operands    one summand x read only by S, other results kept as needed
    S: sum(x, dim=-1)                           torch_cuda_vector4_v1 (one warp per row)

S's loop already visits every element of its row once (lane, vector slot, column block). Here it evaluates E's
lines at that element, reading G through its gather (table[idx[row]][col], the stored form's value and range
trap), adds the same summand value to the same accumulator, and stores E's other results. The FP32 gathered rows
and the summand are no longer written or re-read. Every value, rounding and the reduction tree are unchanged;
only where E's element operations run changes. ``ENABLED = False`` keeps the three passes.
"""
from __future__ import annotations

import math
import re

from .group_rules import _identity, _topological

ENABLED = True
LEAF = re.compile(r'latch_acc4\[c\] = latch_acc4\[c\] \+ (?P<leaf>[^;]+);')


def fuse(phases, layouts, dtypes, *, ident, index, kept, inplace_roots):
    """Returns (phases, records, elided_roots)."""
    if not ENABLED:
        return phases, [], set()
    root = lambda name: layouts[name].root if name in layouts else name
    records, elided = [], set()
    changed = True
    while changed:
        changed = False
        writer, readers = {}, {}
        for k, phase in enumerate(phases):
            for w in phase['writes']:
                writer[root(w)] = k
            for r in phase['reads']:
                readers.setdefault(root(r), set()).add(k)
        for s, summ in enumerate(phases):
            plan = _plan(s, summ, phases, layouts, dtypes, writer, readers, kept, inplace_roots, root, ident, index)
            if plan is None:
                continue
            e, g, merged, dead, record = plan
            elided |= dead
            records.append(record)
            phases = [merged if k == s else p for k, p in enumerate(phases) if k not in (e, g)]
            phases = _topological(phases, root)
            changed = True
            break
    return phases, records, elided


def _plan(s, summ, phases, layouts, dtypes, writer, readers, kept, inplace_roots, root, ident, index):
    if (summ.get('kind') != 'reduction' or summ.get('engine') not in ENGINES
            or summ.get('row_arithmetic') != 'torch_cuda_vector4_v1' or len(summ['reads']) != 1
            or len(summ['writes']) != 1 or len(LEAF.findall(summ.get('code', ''))) != 1):
        return None
    x = summ['reads'][0]
    summand = root(x)
    e = writer.get(summand)
    if e is None or e >= s:
        return None
    elem = phases[e]
    if elem.get('kind') != 'elementwise' or len(elem.get('body', ())) != len(elem['names']):
        return None
    shape = tuple(elem['shape'])
    extent = summ.get('reduction_elements')
    if type(extent) is not int or extent <= 0 or math.prod(shape) % extent:
        return None
    rows = math.prod(shape) // extent
    if (summand in kept or summand in inplace_roots or readers.get(summand, set()) != {s}
            or summand not in {root(n) for n in elem['names']} or not _identity(layouts[x], shape)
            or dtypes.get(summand) != 'float32'):
        return None
    # The leaf is one read of the summand's (contiguous, identity) storage at row prow, column pcol.
    leaf = LEAF.search(summ['code']).group('leaf').strip()
    if not (leaf.startswith(ident(x) + '[') and leaf.endswith(']') and leaf.count('[') == 1
            and 'prow' in leaf and 'pcol' in leaf):
        return None
    gathers = {writer[root(r)] for r in elem['reads'] if root(r) in writer
               and phases[writer[root(r)]].get('kind') == 'index_select'
               and phases[writer[root(r)]].get('engine') == 'index_select_warp_rows'}
    if len(gathers) > 1 or (not gathers and not PLAIN_PRODUCERS):
        return None
    if not gathers:
        return _plan_plain(s, summ, e, elem, phases, layouts, dtypes, readers, kept, inplace_roots, root, ident,
                           x, summand, rows, extent, shape)
    g = gathers.pop()
    gather = phases[g]
    rows_value = root(gather['names'][0])
    table, idx = gather.get('gather_table'), gather.get('gather_index')
    if (len(gather['names']) != 1 or table is None or idx is None or rows_value in kept or rows_value in inplace_roots
            or readers.get(rows_value, set()) != {e} or dtypes.get(rows_value) != 'float32'
            or dtypes.get(table) != 'float32' or dtypes.get(idx) != 'int64'):
        return None
    lt, li = layouts[table], layouts[idx]
    if (len(lt.shape) < 2 or math.prod(lt.shape[1:]) != extent or tuple(li.shape) != (rows,)
            or not _identity(lt, tuple(lt.shape)) or not _identity(li, tuple(li.shape))):
        return None
    # The moved results are read only after S; every other operand of E precedes E already.
    for name in elem['names']:
        if any(k < s and k != e for k in readers.get(root(name), set())):
            return None
    aliases = sorted((n for n in layouts if root(n) == rows_value), key=len, reverse=True)
    if any(layouts[n].offset for n in aliases) or not _identity(layouts[rows_value], (rows, extent)):
        return None
    # A read of the stored rows at flat storage position f is table[idx[f / W]][f % W] (with its range trap).
    lines = [_gather_reads(line, [ident(n) for n in aliases],
                           lambda f: (f'latch_row_gather({ident(table)}, {ident(idx)}, (long long)({f}), '
                                      f'{lt.shape[0]}LL, {extent}LL)'))
             for line in elem['body']]
    if any(line is None for line in lines):
        return None
    dead = {rows_value, summand}
    stored = []
    for k, name in enumerate(elem['names']):
        y = root(name)
        if y == summand or (y not in kept and y not in inplace_roots and readers.get(y, set()) <= {e, s}):
            lines[k] = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', lines[k])
            dead.add(y)
        else:
            stored.append(name)
    register = next(f'r_{n}' for n in elem['names'] if root(n) == summand)

    def leaf(text_lines):
        body = '\n'.join('            ' + line for line in text_lines)
        return ('{ const long long i = (long long)prow * ' + f'{extent}LL + pcol;\n' + body
                + f'\n            latch_acc4[c] = latch_acc4[c] + {register}; }}')

    code = LEAF.sub(lambda m: leaf(lines), summ['code'], count=1)
    if VECTOR_ROWS:
        vector = _vector_rows(summ['code'], lines, leaf, table=ident(table), index=ident(idx), rows=lt.shape[0],
                              extent=extent, layouts=layouts, dtypes=dtypes, ident=ident, shape=shape)
        if vector is not None:
            code = vector.replace('/*SCALAR*/', code)
    merged = dict(summ)
    merged.update(code=code, names=[*stored, *summ['names']], writes=[*stored, *summ['writes']],
                  targets=[*elem['targets'], *gather['targets'], *summ['targets']],
                  reads=list(dict.fromkeys([table, idx, *(r for r in elem['reads']
                                                         if root(r) not in (rows_value, summand))])),
                  gathered_row_sum=dict(gather=gather['names'][0], table=table, index=idx,
                                        elementwise=list(elem['names']), summand=x, rows=rows, extent=extent,
                                        preserves='summand values, vector4 accumulation order and warp tree'))
    record = dict(sum=summ['names'][0], gather=gather['names'][0], elementwise=list(elem['names']),
                  stored=stored, elided=sorted(dead))
    return e, g, merged, dead, record


def _plan_plain(s, summ, e, elem, phases, layouts, dtypes, readers, kept, inplace_roots, root, ident,
                x, summand, rows, extent, shape):
    """The same row pass for an elementwise summand without a gather (e.g. a norm's square and its row mean)."""
    for name in elem['names']:
        if any(k < s and k != e for k in readers.get(root(name), set())):
            return None
    lines = list(elem['body'])
    dead, stored = {summand}, []
    for k, name in enumerate(elem['names']):
        y = root(name)
        if y == summand or (y not in kept and y not in inplace_roots and readers.get(y, set()) <= {e, s}):
            lines[k] = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', lines[k])
            dead.add(y)
        else:
            stored.append(name)
    register = next(f'r_{n}' for n in elem['names'] if root(n) == summand)

    def leaf(text_lines):
        body = '\n'.join('            ' + line for line in text_lines)
        return ('{ const long long i = (long long)prow * ' + f'{extent}LL + pcol;\n' + body
                + f'\n            latch_acc4[c] = latch_acc4[c] + {register}; }}')

    code = LEAF.sub(lambda m: leaf(lines), summ['code'], count=1)
    if VECTOR_ROWS:
        vector = _vector_rows(summ['code'], lines, leaf, table=None, index=None, rows=0, extent=extent,
                              layouts=layouts, dtypes=dtypes, ident=ident, shape=shape)
        if vector is not None:
            code = vector.replace('/*SCALAR*/', code)
    merged = dict(summ)
    merged.update(code=code, names=[*stored, *summ['names']], writes=[*stored, *summ['writes']],
                  targets=[*elem['targets'], *summ['targets']],
                  reads=list(dict.fromkeys(r for r in elem['reads'] if root(r) != summand)),
                  gathered_row_sum=dict(gather=None, elementwise=list(elem['names']), summand=x, rows=rows,
                                        extent=extent,
                                        preserves='summand values, vector4 accumulation order and warp tree'))
    record = dict(sum=summ['names'][0], gather=None, elementwise=list(elem['names']), stored=stored,
                  elided=sorted(dead))
    return e, None, merged, dead, record


# LATCH-MOE-EMA-FAST: torch vector4 row sums and means share the row loop (mean multiplies after the tree).
ENGINES = ('torch_cuda_vector4_sum', 'torch_cuda_vector4_mean')
PLAIN_PRODUCERS = True
VECTOR_ROWS = True
_C_LOOP = 'for (unsigned c = 0; c < 4u; ++c) {'
_PROW = 'const unsigned prow = g;'
_LANES = ('auto latch_vlane4 = [](const float4 v, unsigned c) -> float { '
          'return c == 0u ? v.x : c == 1u ? v.y : c == 2u ? v.z : v.w; };\n'
          'auto latch_hlane4 = [](const uint2 v, unsigned c) -> __nv_bfloat16 { const unsigned w = c < 2u ? v.x : v.y; '
          'return __ushort_as_bfloat16((unsigned short)((c & 1u) ? (w >> 16) : (w & 0xffffu))); };\n')


_ROW_DIV = re.compile(r'\(unsigned\)\(i\) / (\d+)u')


def hoist_quad_invariant(lines, base, prefix):
    """Reads `A[SUB]` whose subscript uses i only as `(unsigned)(i) / D` with D % 4 == 0 take one value for the
    four elements base..base+3 (base % 4 == 0): (base + c) / D == base / D. Returns (lines, preload statements)."""
    preload, cache = [], {}
    out = []
    for line in lines:
        position = 0
        while True:
            match = re.search(r'\b(v_\w+)\[', line[position:])
            if match is None:
                break
            at = position + match.start()
            array = match.group(1)
            depth, k = 0, at + len(array)
            while k < len(line):
                depth += {'[': 1, ']': -1}.get(line[k], 0)
                if depth == 0:
                    break
                k += 1
            if k >= len(line):
                return lines, []
            sub = line[at + len(array) + 1:k]
            divisors = [int(d) for d in _ROW_DIV.findall(sub)]
            uses = len(re.findall(r'\bi\b', sub))
            if divisors and all(d % 4 == 0 for d in divisors) and uses == len(divisors) and '[' not in sub:
                text = f'{array}[{sub}]'
                if text not in cache:
                    cache[text] = f'{prefix}{len(cache)}'
                    preload.append(f'const auto {cache[text]} = {array}[{_ROW_DIV.sub(lambda m: f"(unsigned)({base}) / {m.group(1)}u", sub)}];')
                line = line[:at] + cache[text] + line[k + 1:]
                position = at + len(cache[text])
            else:
                position = k + 1
        out.append(line)
    return out, preload


def _vector_rows(code, lines, leaf, *, table, index, rows, extent, layouts, dtypes, ident, shape):
    """The aligned form of the fused row loop: each lane's four contiguous elements of a column block are loaded
    once as float4 (FP32) / uint2 (four BF16), and the gathered row's address and range trap are resolved once per
    row. Every element value, the leaf order, the vector4 accumulators and the stores are those of the scalar
    form, which is retained (`/*SCALAR*/`) for unaligned storage. None when the loop is not of this form."""
    if code.count(_C_LOOP) != 1 or code.count(_PROW) != 1 or len(LEAF.findall(code)) != 1:
        return None
    vec = list(lines)
    if table is not None:
        gather = re.compile(re.escape(f'latch_row_gather({table}, {index}, (long long)(') + r'\(?i\)?'
                            + re.escape(f'), {rows}LL, {extent}LL)'))
        vec = [gather.sub('latch_vlane4(latch_vg, c)', line) for line in vec]
    if any('latch_row_gather(' in line for line in vec):
        return None
    names = {ident(n): n for n in layouts}
    preload, masks = [], ([f'(unsigned long long)({table})'] if table is not None else [])
    narrow = []
    for array in sorted({m.group(1) for line in vec for m in re.finditer(r'\b(v_\w+)\[\(i\)\]', line)}):
        name = names.get(array)
        if name is None or not _identity(layouts[name], shape):
            return None
        k = len(preload)
        if dtypes.get(name) == 'float32':
            preload.append(f'const float4 latch_v{k} = *reinterpret_cast<const float4*>({array} + latch_i0);')
            masks.append(f'(unsigned long long)({array})')
            value = f'latch_vlane4(latch_v{k}, c)'
        elif dtypes.get(name) == 'bfloat16':
            preload.append(f'const uint2 latch_v{k} = *reinterpret_cast<const uint2*>({array} + latch_i0);')
            narrow.append(f'(unsigned long long)({array})')
            value = f'latch_hlane4(latch_v{k}, c)'
        else:
            return None
        vec = [line.replace(f'{array}[(i)]', value) for line in vec]
    # Each lane's four stores of a column block become one vector store after its last element (c == 3).
    for k, line in enumerate(vec):
        match = re.search(r'; (v_\w+)\[i\] = (r_\w+);$', line)
        if match is None:
            continue
        array, value = match.groups()
        name = names.get(array)
        if name is None or not _identity(layouts[name], shape) or dtypes.get(name) not in ('float32', 'bfloat16'):
            return None
        s = f'latch_s{k}'
        if dtypes[name] == 'bfloat16':
            preload.append(f'unsigned {s}[4];')
            narrow.append(f'(unsigned long long)({array})')
            store = (f'{s}[c] = (unsigned)__bfloat16_as_ushort({value}); if (c == 3u) '
                     f'*reinterpret_cast<uint2*>({array} + latch_i0) = make_uint2({s}[0] | ({s}[1] << 16), '
                     f'{s}[2] | ({s}[3] << 16));')
        else:
            preload.append(f'float {s}[4];')
            masks.append(f'(unsigned long long)({array})')
            store = (f'{s}[c] = {value}; if (c == 3u) *reinterpret_cast<float4*>({array} + latch_i0) = '
                     f'make_float4({s}[0], {s}[1], {s}[2], {s}[3]);')
        vec[k] = line[:match.start()] + '; ' + store
    vec, hoisted = hoist_quad_invariant(vec, 'latch_i0', 'latch_u')
    preload += hoisted
    row = (f'{_PROW}\n        const long long latch_gr = (long long){index}[prow]; '
           f'if (latch_gr < 0 || latch_gr >= {rows}LL) __trap();\n'
           f'        const float* latch_grow = {table} + latch_gr * {extent}LL;') if table is not None else _PROW
    unrolled = re.compile(r'#pragma unroll\n(\s*)' + re.escape(_C_LOOP))
    if len(unrolled.findall(code)) != 1:
        return None
    block = ('const long long latch_i0 = (long long)prow * ' + f'{extent}LL + 128u*j + 4u*lane;\n'
             + ('          const float4 latch_vg = *reinterpret_cast<const float4*>(latch_grow + (128u*j + 4u*lane));\n'
                if table is not None else '')
             + ''.join('          ' + p + '\n' for p in preload) + '#pragma unroll\n          ' + _C_LOOP)
    fast = LEAF.sub(lambda m: leaf(vec), unrolled.sub(lambda m: block, code.replace(_PROW, row)), count=1)
    if not masks and not narrow:
        return None
    aligned = ' && '.join([f'(({" | ".join(masks)}) & 15ULL) == 0ULL'] * bool(masks)
                          + [f'(({" | ".join(narrow)}) & 7ULL) == 0ULL'] * bool(narrow))
    return '{\n' + _LANES + f'if ({aligned}) {{\n' + fast + '\n} else {\n/*SCALAR*/\n}\n}'


def _gather_reads(line, arrays, render):
    """Replace every `array[EXPR]` (balanced brackets) of the given arrays; None if a read cannot be parsed."""
    for array in arrays:
        start = 0
        while True:
            at = line.find(array + '[', start)
            if at < 0:
                break
            if at > 0 and (line[at - 1].isalnum() or line[at - 1] == '_'):
                start = at + 1
                continue
            depth, k = 0, at + len(array)
            while k < len(line):
                depth += {'[': 1, ']': -1}.get(line[k], 0)
                if depth == 0:
                    break
                k += 1
            if k >= len(line):
                return None
            replacement = render(line[at + len(array) + 1:k])
            line = line[:at] + replacement + line[k + 1:]
            start = at + len(replacement)
    return line
