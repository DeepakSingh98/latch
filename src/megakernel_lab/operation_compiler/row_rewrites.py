"""Exact rewrites of warp-per-row group bodies (row/normalization/elementwise chains) and literal divisions.

Every rewrite is opt-in (``accumulation_rules`` / ``elementwise_rules`` names below); with none selected the emitted
source is unchanged.

Row groups (row_fusion.py) evaluate element ``i = o * W + 32 * w + lane`` of row ``o`` (W % 32 == 0, w < W / 32, so
``k = 32 * w + lane`` is in [0, W)). Index text substituted into these bodies after row fusion -- notably the
select/slice_backward placements of indexed_backward.py, which arrive with their own flat-index coordinate algebra
-- recomputes per element quantities that are row constants: the slot guard ``(i / W) % K == j`` and the gradient
row ``i / (K * W)``. The compiler cannot prove ``k < W`` through the 32-bit division sequences, so every element
pays the division chains and the guarded loads are not batched.

* ``accumulation_rules`` ``ridx`` -- inside a row-loop scope that declares ``const long long i_row = o;``
  (row_fusion's ROW_INDEX_REUSE form), and only when ``rows * W < 2**32`` (so ``(unsigned)(i) == i``):

  - ``(unsigned)(i) / N u`` with ``N % W == 0``  ->  ``((unsigned)(i_row) / (N / W) u)``
    (``o = c*q + r``, ``r < c``: ``o*W + k = c*W*q + (r*W + k)`` with ``r*W + k < c*W``, so the quotient is ``q``);
  - ``(unsigned)(i) % W u``  ->  ``(unsigned)(32 * w + lane)``  (``o*W + k`` mod ``W`` is ``k``);
  - ``(unsigned)(i) / 1u`` -> ``(unsigned)(i)`` and the no-op casts ``(unsigned)(((long long)((unsigned)(X))))`` ->
    ``(unsigned)(X)`` / ``(unsigned)(((i)))`` -> ``(unsigned)(i)`` (value-identical: X < 2**32).

  Every rewritten subexpression has exactly the original unsigned value, so every address, guard and value is
  unchanged; only how the index is computed changes (the guard becomes a row-uniform expression nvcc hoists).

* ``accumulation_rules`` ``rearly`` -- row_fusion also takes consumers of a group's prologue values that come before
  its reduction as epilogue candidates; ``feeds_group`` is the cycle guard.

* ``accumulation_rules`` ``rremat`` -- a row-scaled FP32 group value ``y = a * s[row]`` read only at identity by later
  elementwise phases is recomputed in those readers from ``a`` and ``s`` and is not stored (``remat_products``).

* ``elementwise_rules`` ``divc`` -- IEEE single-precision division by a literal constant, ``(P / C)``, is compiled by
  nvcc (-prec-div=true) as an inline fast path plus an out-of-line slow-path CALL. The call inside a row body's
  unrolled element loop stops nvcc from batching the loop's loads (a power-of-two divisor, which nvcc turns into an
  exact multiply, avoids it). The rewrite replaces the quotient by ``latch_divc(P, C, RN(1/C), (double)C,
  RN_d(1/C))``, an inline function with no call:

  - |P| in [2^-100, 2^100]: q = RN(P*y), e = fma(-q, C, P), result fma(e, y, q) (float, y = RN(1/C));
  - other nonzero finite P (tiny/huge): the same correction in double from (double)P, then one RN to float;
  - +-0 and +-inf: P (C > 0); NaN: the canonical NaN (what the emitted IEEE division returns for every NaN).
  All values are computed and combined by selects (no branch), so nothing stops nvcc batching the loop's loads.

  It is applied only for divisors in ``VERIFIED_DIVISORS`` (see there), and never for powers of two (already an
  exact multiply). The shipped table lists the literals verified so far; every other literal keeps its IEEE
  division. Per-literal rewrite counts are recorded in the program manifest under
  ``group_rules["exact_divisions"]``. Only parenthesized quotients whose left operand is a float register/array
  element/member/parenthesized float expression are rewritten, so the C++ operand types (float, or BF16/integers
  converted to float) are unchanged.
"""
from __future__ import annotations

import re

VERSION = 1


def _elementwise(name):
    from .elementwise_rules import active
    return active(name)


def _accumulation(name):
    from .accumulation_rules import active
    return active(name)


def rowidx():
    return _accumulation("ridx")


def earlyepi():
    return _accumulation("rearly")


_SCOPE = re.compile(r'const long long i = (o|on_) \* (\d+)LL \+ 32 \* w \+ lane; const long long i_row = \1;')
_DIV = re.compile(r'\(unsigned\)\(i\) / (\d+)u')
_MOD = re.compile(r'\(unsigned\)\(i\) % (\d+)u')
_NOOP = re.compile(r'\(unsigned\)\(\(\(long long\)\(\(unsigned\)\(((?:i|i_row))\)\)\)\)')


def _rewrite_line(line, width):
    for _ in range(4):
        before = line
        line = line.replace('(unsigned)(((i)))', '(unsigned)(i)').replace('(unsigned)(i) / 1u', '(unsigned)(i)')
        line = _NOOP.sub(lambda m: f'(unsigned)({m.group(1)})', line)
        line = _DIV.sub(lambda m: ('(unsigned)(i_row)' if int(m.group(1)) == width else
                                   f'((unsigned)(i_row) / {int(m.group(1)) // width}u)')
                        if int(m.group(1)) % width == 0 else m.group(0), line)
        line = _MOD.sub(lambda m: '(unsigned)(32 * w + lane)' if int(m.group(1)) == width else m.group(0), line)
        if line == before:
            break
    return line


def row_index_rewrite(code, rows):
    """Apply the 'ridx' rewrites to one row-group body (text). ``rows`` bounds the row index."""
    if not rowidx():
        return code
    out, scopes, depth = [], [], 0
    for line in code.split('\n'):
        m = _SCOPE.search(line)
        if m and int(m.group(2)) % 32 == 0 and rows * int(m.group(2)) < 2 ** 32:
            scopes.append((depth, int(m.group(2))))
        elif m or re.search(r'\blong long i = ', line):
            scopes.append((depth, None))   # another index binding: no rewrite until it closes
        if scopes and scopes[-1][1] is not None and not m:
            line = _rewrite_line(line, scopes[-1][1])
        out.append(line)
        depth += line.count('{') - line.count('}')
        while scopes and depth < scopes[-1][0]:
            scopes.pop()
    return '\n'.join(out)


def divc():
    return _elementwise("divc")


# Literal divisors verified exhaustively: an H100 kernel evaluated latch_divc(x, C, ...) and the emitted IEEE x / C
# (nvcc -prec-div=true) for all 2^32 FP32 bit patterns x and compared the result bits; every divisor listed here had
# zero differing results. The table lists only the literals verified so far (3, 5, 6, 7, 480, sqrt(480) and 16380,
# the divisors met while optimizing one model); a division by any other literal keeps its IEEE division. No checking
# harness ships with the package: adding a divisor requires an exhaustive comparison of latch_divc against IEEE x / C
# (nvcc -prec-div=true) over all 2^32 FP32 inputs x on the target GPU, with zero differing result bits.
# Keys are the literal exactly as it appears in emitted source; values are (RN_float(1/C), RN_double(1/C)) as the
# hex-float and decimal text passed to latch_divc.
VERIFIED_DIVISORS = {
    "480.0f": ("0x1.1111120000000p-9f", "0.0020833333333333333"),
    "21.908902300206645f": ("0x1.75e9740000000p-5f", "0.045643544746802246"),
    "6.0f": ("0x1.5555560000000p-3f", "0.16666666666666666"),
    "3.0f": ("0x1.5555560000000p-2f", "0.3333333333333333"),
    "7.0f": ("0x1.24924a0000000p-3f", "0.14285714285714285"),
    "5.0f": ("0x1.99999a0000000p-3f", "0.2"),
    "16380.0f": ("0x1.0010020000000p-14f", "6.105006105006105e-05"),
}

# Branch-free (selects only): a branch around the double path would bring the scheduling barrier back.
DIVC_HELPER = r"""#ifndef LATCH_EXACTDIV
#define LATCH_EXACTDIV
// latch_divc (row_rewrites.py): IEEE x / c for an exhaustively verified literal c > 0, inline and branch-free.
__device__ __forceinline__ float latch_divc(float x, float c, float yf, double cd, double yd) {
  const double d_ = (double)x; const double q_ = __dmul_rn(d_, yd); const double r_ = __fma_rn(-q_, cd, d_);
  float z_ = __double2float_rn(__fma_rn(r_, yd, q_));
  const float q = __fmul_rn(x, yf); const float e = __fmaf_rn(-q, c, x); const float f = __fmaf_rn(e, yf, q);
  const float a = fabsf(x);
  z_ = (a >= 0x1p-100f && a <= 0x1p100f) ? f : z_;
  z_ = (x == 0.0f || isinf(x)) ? x : z_;
  return isnan(x) ? __int_as_float(0x7fffffff) : z_;
}
#endif
"""

_CDIV = re.compile(r'/ ([0-9]+\.[0-9]+f)\)')
_BAD_OPERAND = re.compile(r'\bdouble\b|(?<![\w.])[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?(?![0-9eE.f])')


def _primary_start(text, end):
    """Start of the primary expression that ends at ``end``: a balanced ``( )``, a name with ``[ ]``/``.`` suffixes."""
    k = end
    while k > 0 and text[k - 1] == ' ':
        k -= 1
    stop = k
    while True:
        if k > 0 and text[k - 1] in ')]':
            close = text[k - 1]
            open_ = '(' if close == ')' else '['
            depth, j = 0, k - 1
            while j >= 0:
                if text[j] == close:
                    depth += 1
                elif text[j] == open_:
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            if j < 0:
                return None
            k = j
            if close == ')':
                while k > 0 and (text[k - 1].isalnum() or text[k - 1] in '_:'):
                    k -= 1
                return k if k < stop else None
            continue
        while k > 0 and (text[k - 1].isalnum() or text[k - 1] in '_.'):
            k -= 1
        return k if k < stop else None


def _literal(text):
    import struct
    v = struct.unpack('<f', struct.pack('<f', float(text[:-1])))[0]
    return v, struct.unpack('<f', struct.pack('<f', 1.0 / v))[0] if v else None


def division_rewrite(code, counts=None):
    """'divc': rewrite ``(P / C)`` for verified literal C (see module docstring). ``counts`` collects them."""
    if not divc() or '/ ' not in code:
        return code
    out, pos, parts = code, 0, []
    for m in _CDIV.finditer(code):
        lit = m.group(1)
        if lit not in VERIFIED_DIVISORS:
            continue
        start = _primary_start(code, m.start())
        if start is None or start < pos or start == 0 or code[start - 1] != '(':
            continue
        expr = code[start:m.start()].rstrip()
        if _BAD_OPERAND.search(expr) or not re.match(r'[A-Za-z_(]', expr):
            continue
        yf, yd = VERIFIED_DIVISORS[lit]
        parts.append(code[pos:start - 1])
        parts.append(f'latch_divc({expr}, {lit}, {yf}, (double){lit}, {yd})')
        pos = m.end()
        if counts is not None:
            counts[lit] = counts.get(lit, 0) + 1
    if not parts:
        return code
    parts.append(code[pos:])
    return ''.join(parts)


def feeds_group(phases, e, k, member_reads, root, readers):
    """'rearly' cycle guard: True when a value written by phase e (< k) reaches a root the group reads,
    directly or through phases e < x < k (the only phases that can run between e and the group)."""
    frontier = [root(w) for w in phases[e]['writes']]
    seen, visited = set(frontier), set()
    while frontier:
        y = frontier.pop()
        if y in member_reads:
            return True
        for x in readers.get(y, ()):
            if e < x < k and x not in visited:
                visited.add(x)
                for w in phases[x]['writes']:
                    rw = root(w)
                    if rw not in seen:
                        seen.add(rw)
                        frontier.append(rw)
    return False


def remat():
    return _accumulation("rremat")


_ROWSCALE = r'const float r_{y} = \(r_(\w+) \* (v_\w+)\[([^\[\]]*)\]\);'


def remat_products(phases, layouts, *, kept_roots, ident, numel, dtypes):
    """'rremat' (see module docstring). Returns (records, values no longer stored, values now stored)."""
    if not remat():
        return [], set(), set()
    root = lambda n: layouts[n].root if n in layouts else n
    readers = {}
    for j, q in enumerate(phases):
        for r in q.get('reads', ()):
            readers.setdefault(root(r), set()).add(j)
    records, dead, stored_now = [], set(), set()
    for k, p in enumerate(phases):
        rf = p.get('row_fusion') or {}
        if p.get('kind') != 'row_fused' or not isinstance(p.get('code'), str) or not rf.get('width'):
            continue
        width, code = int(rf['width']), p['code']
        if int(rf.get('rows', 0)) * width >= 2 ** 32:
            continue
        by_ident = {ident(n): n for n in p.get('reads', ())}
        stored = list(rf.get('stored', ()))
        for y in list(stored):
            lay = layouts.get(y)
            if (lay is None or lay.root != y or y in kept_roots or dtypes.get(y) != 'float32'
                    or y == rf.get('reduction') or lay.numel != int(rf['rows']) * width):
                continue
            m = re.search(_ROWSCALE.format(y=re.escape(y)), code)
            if not m or code.count(f'const float r_{y} = ') != 1:
                continue
            a, sp, idx = m.group(1), m.group(2), m.group(3)
            s = by_ident.get(sp)
            la = layouts.get(a)
            if (s is None or 'i_row' not in idx or re.search(r'(?<![\w.])i(?![\w])', idx)
                    or a not in p.get('names', ()) or la is None or la.root != a or a in kept_roots
                    or dtypes.get(a) != 'float32' or la.numel != lay.numel or not la.contiguous or la.offset):
                continue
            ys = [j for j in readers.get(y, ()) if j != k]
            name_re = re.compile(rf'(?<![\w.]){re.escape(ident(y))}(?!\w)')
            identity_re = re.compile(rf'(?<![\w.]){re.escape(ident(y))}\[\s*\(?i\)?\s*\]')
            ok = bool(ys)
            for j in ys:
                q = phases[j]
                texts = [q[key] for key in ('code', 'code_r1') if isinstance(q.get(key), str)] + list(q.get('body') or [])
                joined = '\n'.join(texts)
                if (j < k or q.get('kind') != 'elementwise' or not q.get('body') or q.get('vprod_inline')
                        or numel(q['shape']) != lay.numel or any(root(r) == y and r != y for r in q['reads'])
                        or {root(w) for w in q['writes']} & {y, a, root(s)}
                        or not name_re.search(joined) or len(name_re.findall(joined)) != len(identity_re.findall(joined))):
                    ok = False
                    break
            if not ok:
                continue
            row_text = f'((unsigned)(i) / {width}u)'
            expr = f'({ident(a)}[(i)] * {sp}[{idx.replace("(i_row)", "(" + row_text + ")").replace("i_row", row_text)}])'
            for j in ys:
                q = phases[j]
                for key in ('code', 'code_r1'):
                    if isinstance(q.get(key), str):
                        q[key] = identity_re.sub(lambda _m: expr, q[key])
                if q.get('body'):
                    q['body'] = [identity_re.sub(lambda _m: expr, line) for line in q['body']]
                q['reads'] = list(dict.fromkeys([r for r in q['reads'] if root(r) != y] + [a, s]))
                q.pop('row_parts', None)
                q.setdefault('row_scale_inputs', []).append(y)
            # producer: drop y's stores, store a once where it is computed
            store_re = re.compile(rf'\s*{re.escape(ident(y))}\[i\] = [^;]+;')
            for key in ('code', 'code_r1'):
                if isinstance(p.get(key), str):
                    p[key] = store_re.sub('', p[key])
            if a not in stored and a not in stored_now:
                def add_store(text):
                    lines = text.split('\n')
                    for t, line in enumerate(lines):
                        mm = re.search(rf'const float r_{re.escape(a)} = (?!keep_|ck_)[^;]+;', line)
                        if mm:
                            lines[t] = line[:mm.end()] + f' {ident(a)}[i] = r_{a};' + line[mm.end():]
                            return '\n'.join(lines), True
                    return text, False
                p['code'], done = add_store(p['code'])
                if not done:
                    raise RuntimeError(f'row-scale remat: no computing definition of {a} in its row group')
                if isinstance(p.get('code_r1'), str):
                    p['code_r1'], _ = add_store(p['code_r1'])
                stored_now.add(a)
                rf.setdefault('stored', []).append(a)
                if a in rf.get('elided', []):
                    rf['elided'] = [n for n in rf['elided'] if n != a]
            rf['stored'] = [n for n in rf['stored'] if n != y]
            rf.setdefault('elided', []).append(y)
            p.pop('row_parts', None)
            dead.add(y)
            code = p['code']
            records.append(dict(value=y, factor=a, scale=s, readers=[phases[j]['names'][0] for j in ys],
                                elements_not_stored=lay.numel))
    return records, dead, stored_now
