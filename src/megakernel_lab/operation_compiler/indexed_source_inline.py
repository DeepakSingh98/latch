"""LATCH-MOE-EMA-FAST: an ordered indexed gradient evaluates its single-line elementwise source at each read.

    x = EXPR(i)                         elementwise FP32, one line, read only by P (e.g. expert output * weight)
    P: embedding_gradient over rows of x (ordered row sums, float4 rows when aligned, scalar rows otherwise)

P reads x only through `g_[...]` (float4 or float view of x's contiguous storage). Each read becomes the same
expression evaluated at those flat elements (a float4 read is its four elements), so every addend and the
ordered FP32 accumulation are unchanged; x itself is not written or re-read. ``ENABLED = False`` keeps x.
"""
from __future__ import annotations

import math
import re

from .gathered_row_sum import _LANES, _gather_reads, hoist_quad_invariant
from .group_rules import _identity, _topological

ENABLED = True
VECTOR_SOURCES = True
# The aligned branch keeps RAW_BATCH source rows in flight as raw operands (identity quads and row-invariant
# reads) and evaluates each addend right before its ordered addition (same value, same order).
RAW_ROWS = True
RAW_BATCH = 8


def _raw_rows_head(head, arrays, kinds, value, hoisted, names, dtypes):
    """Replace the float4 branch of an ordered warp-row body by the raw-operand form; None if not of that form."""
    grab = lambda pattern: (lambda m: m.group(1) if m else None)(re.search(pattern, head))
    out, ordered = grab(r'float4\* o_ = reinterpret_cast<float4\*>\((\w+)\)'), grab(r'\? (\w+)\[q_ \+ lane_\]')
    bounds, rows = grab(r'const long long b_ = (\w+)\[2 \* row\]'), grab(r'row < (\d+)LL')
    padding, units = grab(r'row != (-?\d+)LL'), grab(r'c_ < (\d+)\)')
    cond = re.search(r'\n  if \((.*)\) \{\n', head)
    if None in (out, ordered, bounds, rows, padding, units) or cond is None or re.search(r'\bi\b', value):
        return None
    units = int(units)
    J = -(-units // 32)
    loads_u, loads_j = [], []
    decl = []
    for k, a in enumerate(arrays):
        t = 'float4' if kinds[a] == 'float32' else 'uint2'
        decl.append(f'{t} r{k}_[{RAW_BATCH}][{J}];')
        loads_j.append(f'r{k}_[u][j] = *reinterpret_cast<const {t}*>({a} + 4 * (o_ * {units}LL + c_));')
        value = value.replace(f'latch_w{k}', f'r{k}_[u][j]')
    for k, line in enumerate(hoisted):
        m = re.match(r'const auto latch_u(\d+) = (v_\w+)\[(.*)\];$', line)
        if m is None or dtypes.get(names.get(m.group(2))) != 'float32':
            return None
        divisors = [int(d) for d in re.findall(r'\(unsigned\)\(4 \* q\) / (\d+)u', m.group(3))]
        if not divisors or any(d % (4 * units) for d in divisors):
            return None
        decl.append(f'float s{m.group(1)}_[{RAW_BATCH}];')
        sub = m.group(3).replace('(unsigned)(4 * q)', f'(unsigned)(4 * o_ * {units}LL)')
        loads_u.append(f's{m.group(1)}_[u] = live_ ? {m.group(2)}[{sub}] : 0.0f;')
        value = value.replace(f'latch_u{m.group(1)}', f's{m.group(1)}_[u]')
    lanes = ' '.join(f'{{ const unsigned c = {c}u; w_.{f} = {value}; }}' for c, f in enumerate('xyzw'))
    body = f"""{{
    float4* o_ = reinterpret_cast<float4*>({out});
    for (long long row = wid_; row < {rows}LL; row += nw_) {{
      float4 a_[{J}];
#pragma unroll
      for (int j = 0; j < {J}; ++j) a_[j] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (row != {padding}LL) {{
        const long long b_ = {bounds}[2 * row], e_ = {bounds}[2 * row + 1];
#pragma unroll 1
        for (long long q_ = b_; q_ < e_; q_ += 32) {{
          const long long mine_ = q_ + lane_ < e_ ? {ordered}[q_ + lane_] : 0LL;
          const int cnt_ = (int)(e_ - q_ < 32 ? e_ - q_ : 32);
#pragma unroll 1
          for (int u0_ = 0; u0_ < cnt_; u0_ += {RAW_BATCH}) {{
            {' '.join(decl)}
#pragma unroll
            for (int u = 0; u < {RAW_BATCH}; ++u) {{
              const long long o_ = __shfl_sync(0xffffffffu, mine_, (u0_ + u) & 31);
              const bool live_ = u0_ + u < cnt_;
              {' '.join(loads_u)}
#pragma unroll
              for (int j = 0; j < {J}; ++j) {{
                const int c_ = lane_ + 32 * j;
                if (live_ && c_ < {units}) {{ {' '.join(loads_j)} }}
              }}
            }}
#pragma unroll
            for (int u = 0; u < {RAW_BATCH}; ++u)
              if (u0_ + u < cnt_) {{
#pragma unroll
                for (int j = 0; j < {J}; ++j) {{
                  float4 w_; {lanes}
                  a_[j].x = __fadd_rn(a_[j].x, w_.x); a_[j].y = __fadd_rn(a_[j].y, w_.y);
                  a_[j].z = __fadd_rn(a_[j].z, w_.z); a_[j].w = __fadd_rn(a_[j].w, w_.w);
                }}
              }}
          }}
        }}
      }}
#pragma unroll
      for (int j = 0; j < {J}; ++j) {{
        const int c_ = lane_ + 32 * j;
        if (c_ < {units}) o_[row * {units}LL + c_] = a_[j];
      }}
    }}
  """
    return head[:cond.end() - 2] + body
LINE = re.compile(r'^const float r_(?P<name>\w+) = (?P<expr>.+); (?P<array>\w+)\[i\] = r_(?P=name);$')


def fuse(phases, layouts, dtypes, *, ident, kept, inplace_roots):
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
        for p, phase in enumerate(phases):
            plan = _plan(p, phase, phases, layouts, dtypes, writer, readers, kept, inplace_roots, root, ident)
            if plan is None:
                continue
            e, merged, dead, record = plan
            elided |= dead
            records.append(record)
            phases = [merged if k == p else q for k, q in enumerate(phases) if k != e]
            phases = _topological(phases, root)
            changed = True
            break
    return phases, records, elided


def _plan(p, phase, phases, layouts, dtypes, writer, readers, kept, inplace_roots, root, ident):
    code = phase.get('code') or ''
    if phase.get('kind') != 'embedding_gradient' or code.count('} else {') != 1:
        return None
    for source in phase['reads']:
        x = root(source)
        e = writer.get(x)
        if e is None or e >= p:
            continue
        elem = phases[e]
        if (elem.get('kind') != 'elementwise' or len(elem.get('names', ())) != 1 or len(elem.get('body', ())) != 1
                or root(elem['names'][0]) != x or x in kept or x in inplace_roots or readers.get(x, set()) != {p}
                or dtypes.get(x) != 'float32'):
            continue
        shape = tuple(elem['shape'])
        layout = layouts[source]
        if layout.offset or math.prod(layout.shape) != math.prod(shape) or not _identity(layout, tuple(layout.shape)):
            continue
        match = LINE.match(elem['body'][0].strip())
        if match is None or match.group('array') != ident(x) or match.group('name') != elem['names'][0]:
            continue
        src = ident(source)
        vector = f'reinterpret_cast<const float4*>({src})'
        scalar = f'reinterpret_cast<const float*>({src})'
        if code.count(vector) != 1 or code.count(scalar) != 1:
            continue
        head, tail = code.split('} else {')
        if vector not in head or scalar not in tail:
            continue
        head = head.replace(f'const float4* g_ = {vector};', '')
        tail = tail.replace(f'const float* g_ = {scalar};', '')
        head = _gather_reads(head, ['g_'], lambda q: f'latch_src4({q})')
        tail = _gather_reads(tail, ['g_'], lambda q: f'latch_src1({q})')
        if head is None or tail is None:
            continue
        head = head.replace(f'(unsigned long long)({src}) | ', '')
        if re.search(r'\b' + re.escape(src) + r'\b', head + tail):
            continue
        expr = match.group('expr')
        prologue = 'auto latch_src1 = [&](long long i) -> float { return ' + expr + '; };\n'
        names = {ident(n): n for n in layouts}
        arrays = sorted({m.group(1) for m in re.finditer(r'\b(v_\w+)\[\(i\)\]', expr)})
        kinds = {a: dtypes.get(names.get(a)) for a in arrays}
        shape_e = tuple(elem['shape'])
        if (VECTOR_SOURCES and arrays and all(k in ('float32', 'bfloat16') for k in kinds.values())
                and all(_identity(layouts[names[a]], shape_e) for a in arrays)
                and head.count('== 0ULL) {') == 1):
            # The float4 rows' four elements read each identity operand once (float4 / four BF16); the
            # other reads and every operation stay per element. Their alignment joins the float4 branch test.
            loads, value = [], expr
            for k, a in enumerate(arrays):
                if kinds[a] == 'float32':
                    loads.append(f'const float4 latch_w{k} = *reinterpret_cast<const float4*>({a} + 4 * q);')
                    value = value.replace(f'{a}[(i)]', f'latch_vlane4(latch_w{k}, c)')
                else:
                    loads.append(f'const uint2 latch_w{k} = *reinterpret_cast<const uint2*>({a} + 4 * q);')
                    value = value.replace(f'{a}[(i)]', f'latch_hlane4(latch_w{k}, c)')
            (value,), hoisted = hoist_quad_invariant([value], '4 * q', 'latch_u')
            loads += hoisted
            lanes = ''.join(f'{{ const long long i = 4 * q + {c}; const unsigned c = {c}u; latch_r.{f} = {value}; }} '
                            for c, f in enumerate('xyzw'))
            prologue += (_LANES + 'auto latch_src4 = [&](long long q) -> float4 { ' + ' '.join(loads)
                         + ' float4 latch_r; ' + lanes + 'return latch_r; };\n')
            wide = [f'(unsigned long long)({a})' for a in arrays if kinds[a] == 'float32']
            narrow = [f'(unsigned long long)({a})' for a in arrays if kinds[a] == 'bfloat16']
            extra = ''.join(f' && (({" | ".join(group)}) & {mask}ULL) == 0ULL'
                            for group, mask in ((wide, 15), (narrow, 7)) if group)
            head = head.replace('== 0ULL) {', '== 0ULL' + extra + ') {', 1)
            if RAW_ROWS:
                rows8 = _raw_rows_head(head, arrays, kinds, value, hoisted, names, dtypes)
                if rows8 is not None:
                    head = rows8
        else:
            prologue += ('auto latch_src4 = [&](long long q) -> float4 { return make_float4(latch_src1(4 * q), '
                         'latch_src1(4 * q + 1), latch_src1(4 * q + 2), latch_src1(4 * q + 3)); };\n')
        merged = dict(phase)
        merged.update(code='{\n' + prologue + head + '} else {' + tail + '\n}',
                      reads=list(dict.fromkeys([*elem['reads'], *(r for r in phase['reads'] if root(r) != x)])),
                      targets=[*elem['targets'], *phase['targets']],
                      indexed_source_inline=dict(source=source, expression_of=elem['names'][0],
                                                 preserves='each addend value and the ordered FP32 accumulation'))
        dead = {x} | {root(n) for n in layouts if root(n) == x}
        record = dict(gradient=phase['names'][0], source=elem['names'][0])
        return e, merged, dead, record
    return None
