"""Exact body-efficiency rewrites for vectorized (float4) elementwise phases and exact row reductions.

Every rewrite is opt-in (``elementwise_rules`` / ``accumulation_rules`` names below); with none selected the emitted
source is unchanged. Each rewrite changes only how the same values are addressed, loaded or kept: every element
evaluates the same expression text on the same operand values (programs are compiled with --fmad=false), and every
result is stored once, after all loads of its round, exactly as before.

* ``elementwise_rules`` ``lanecse`` -- lane-invariant row index. In a float4 round the four lanes are elements
  4*j + u (u = 0..3). A broadcast read whose index is ``(unsigned)(i) / C`` (or ``i / C`` in 64-bit form) with
  C % 4 == 0 has the same quotient for all four lanes: 4*j is a multiple of 4 (also after the 32-bit truncation,
  since 2**32 is), so (4*j + u) / C == (4*j) / C. Writing the index as ``(4 * j)`` makes the four lanes' load
  addresses and every expression on them textually identical, so nvcc CSEs the per-row load and e.g. a per-row
  IEEE division (pure operations on identical operands). Values unchanged.
* ``elementwise_rules`` ``bf16vec`` -- identity-indexed contiguous BF16 operands of an FP32 float4 phase are loaded
  as one 8-byte ``uint2`` per round (4 BF16 values, base 8-byte aligned: offset 0 of a 64-word-aligned arena root,
  index a multiple of 4) instead of four 2-byte loads, and unpacked exactly: ``__bfloat162float(h)`` is the float
  whose bits are ``h << 16``. Only reads that appear exclusively as ``__bfloat162float(TEXT)`` are converted.
* ``elementwise_rules`` ``rolling`` -- (1) an unroll-1 float4 body evaluates its four lanes interleaved line by line
  and reloads each hoisted operand register for the thread's next grid-stride index right after that register's
  last use (operands read by an in-place store are reloaded after the stores). One register set; the next round's
  loads overlap the current arithmetic. Indices are grid-stride disjoint, so a reload of index j + s reads the
  value the original loop reads in that later round (no store of this phase touches index j + s before then); past
  the end the reload is clamped to the current index and unused. (2) Lane-invariant broadcast reads of the current
  index (text containing ``(4 * j)``, no lane term; e.g. a per-row value) also become rolling registers (``auto``
  keeps the element type), loaded with the operand set and reloaded for the next index after their last use.
  vector_elementwise refuses broadcast reads of arrays the phase writes, so the reloaded addresses are never stored
  by this phase. (3) The vec4u grid-stride unroll (4) is capped so that the round's hoisted operand registers (4
  per float4 operand, 2 per uint2 operand) times the unroll stay within ``UNROLL_BUDGET`` (halving 4 -> 2 -> 1).
  Same expressions, same values, each element stored once.
* ``accumulation_rules`` ``rwarp`` -- exact CTA-per-row reductions (256 threads: thread t sums r = t, t+256, ... in
  order; xor-butterfly warp_sum; lanes 0..7 of warp 0 carry the 8 warp partials, other lanes +0.0f, second
  warp_sum) are evaluated one warp per row: lane l replays virtual threads l + 32k (k = 0..7) in the same order, the
  same butterfly per virtual warp (a xor butterfly leaves every lane with identical bits), then the same final
  butterfly. No shared memory or __syncthreads; 8 rows per CTA round. Only for >= 1024 rows and at most 32 rounds
  per virtual thread.
* ``accumulation_rules`` ``rstage`` -- ROW_CHAIN groups: (1) the per-row result store of each chained row stage
  (``if (lane == 0) v_X[o] = r_X;``) is emitted once after the last stage instead of between stages (every r_X is
  in scope there). The group never reads its own stores (internal values are registers), so only the store
  position changes; no global store then separates one stage's leaf loads from the next stage's. (2) Up to
  ``STAGE_GROUP`` consecutive chained stages emit their leaf loops (each into its own leaf array) before their
  reduction trees, when every member but the last has no row/element tail and no member's leaf loop names an
  earlier member's row result. Every stage keeps its leaves, tree and rounding; only the issue order and register
  names change, so one stage's loads overlap the previous stages' shuffle trees.
"""
from __future__ import annotations

import re

VERSION = 1
UNROLL_BUDGET = 64       # 'rolling': hoisted operand registers x vec4u unroll
STAGE_GROUP = 4          # 'rstage': consecutive chained stages whose leaf loops issue together


def _elementwise(name):
    from .elementwise_rules import active
    return active(name)


def _accumulation(name):
    from .accumulation_rules import active
    return active(name)


def rowcse():
    return _elementwise("lanecse")


def bf16vec():
    return _elementwise("bf16vec")


def roll():
    return _elementwise("rolling")


def rowwarp():
    return _accumulation("rwarp")


def rdefer():
    return _accumulation("rstage")


def rgroup():
    return STAGE_GROUP if _accumulation("rstage") else 0


def unroll_budget():
    return UNROLL_BUDGET if _elementwise("rolling") else None


_ROW32 = re.compile(r'\(unsigned\)\(i\) / (\d+)u')
_ROW64 = re.compile(r'(?<![\w.])i / (\d+)LL')


def lane_invariant_rows(line, jv):
    """Rewrite lane-invariant quotients of the element index (before ``i`` -> ``(4 * jv + u)`` substitution)."""
    if not rowcse():
        return line
    line = _ROW32.sub(lambda m: f'(unsigned)((4 * {jv})) / {m.group(1)}u' if int(m.group(1)) % 4 == 0 else m.group(0), line)
    return _ROW64.sub(lambda m: f'(4 * {jv}) / {m.group(1)}LL' if int(m.group(1)) % 4 == 0 else m.group(0), line)


def bf16_operands(heads, texts_bf16, pattern):
    """BF16 identity reads converted to uint2 rounds: {text: name}. A text qualifies only when every occurrence in
    every line is the whole argument of a ``__bfloat162float(...)`` call."""
    if not bf16vec() or not texts_bf16:
        return {}
    found = {}
    for head in heads:
        for text in pattern.findall(head):
            if text in texts_bf16:
                found.setdefault(text, None)
    out = {}
    for text in found:
        wrapped = sum(h.count(f'__bfloat162float({text})') for h in heads)
        total = sum(h.count(text) for h in heads)
        if wrapped and wrapped == total:
            out[text] = f'hb{len(out)}_'
    return out


def bf16_lane(name, sfx, lane):
    word = f'{name}{sfx}.{"x" if lane < 2 else "y"}'
    return f'__uint_as_float({word} << 16)' if lane % 2 == 0 else f'__uint_as_float({word} & 0xffff0000u)'


def choose_unroll(unroll, float4_operands, uint2_operands):
    budget = unroll_budget()
    if not budget:
        return unroll
    regs = 4 * float4_operands + 2 * uint2_operands
    while unroll > 1 and unroll * regs > budget:
        unroll //= 2
    return unroll


_LOAD = re.compile(r'\s*const (float4|uint2) (\w+) = reinterpret_cast<const (?:float4|uint2)\*>\((.+)\)\[j\];$')


_ROWREAD = re.compile(r'v_\w+\[[^\[\]]*\(4 \* j\)[^\[\]]*\]')


def rolling(loads_, compute_, stores_, n4):
    """'rolling' body for the unroll-1 float4 loop (see module docstring); None when the shape is unexpected."""
    if not roll() or not loads_ or len(compute_) % 4:
        return None
    regs = []
    for line in loads_:
        m = _LOAD.match(line)
        if not m:
            return None
        regs.append(m.groups())
    lanes = len(compute_) // 4
    inter = [compute_[u * lanes + t] for t in range(lanes) for u in range(4)]
    use = {name: re.compile(rf'\b{re.escape(name)}\b') for _, name, _ in regs}
    last = {}
    for k, line in enumerate(inter):
        for _, name, _ in regs:
            if use[name].search(line):
                last[name] = k
    late = {name for _, name, _ in regs if any(use[name].search(st) for st in stores_) or name not in last}
    reload = lambda ty, name, ptr: f'    {name} = reinterpret_cast<const {ty}*>({ptr})[jnc_];'
    rows, row_last = {}, {}
    for k, line in enumerate(inter):
        for text in _ROWREAD.findall(line):
            if '4 * j +' not in text:
                rows.setdefault(text, f'rw{len(rows)}_')
                row_last[text] = k
    if any(re.search(rf'\b{n}\b', x) for n in rows.values() for x in inter + stores_):
        rows, row_last = {}, {}
    out = ['{', '  const long long s_ = (long long)gridDim.x * blockDim.x;',
           '  long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;',
           f'  const long long jp_ = j < {n4}LL ? j : 0;']
    out += [f'  {ty} {name} = reinterpret_cast<const {ty}*>({ptr})[jp_];' for ty, name, ptr in regs]
    out += [f'  auto {n} = {t.replace("(4 * j)", "(4 * jp_)")};' for t, n in rows.items()]
    out += [f'  for (; j < {n4}LL; j += s_) {{', f'    const long long jn_ = j + s_; const long long jnc_ = jn_ < {n4}LL ? jn_ : j;']
    for k, line in enumerate(inter):
        for t in sorted(rows, key=len, reverse=True):
            line = line.replace(t, rows[t])
        out.append('  ' + line)
        out += [reload(ty, name, ptr) for ty, name, ptr in regs if name not in late and last.get(name) == k]
        out += [f'    {n} = {t.replace("(4 * j)", "(4 * jnc_)")};' for t, n in rows.items() if row_last[t] == k]
    out += ['    ' + st for st in stores_]
    out += [reload(ty, name, ptr) for ty, name, ptr in regs if name in late]
    out += ['  }', '}']
    return '\n'.join(out)


def warp_row_reduction(rows, n, read, store):
    """'rwarp' replacement of the exact CTA-per-row reduction (see module docstring); ``read`` is the
    per-element read text in ``r`` and ``o``; ``store`` the unchanged result statement on ``acc`` and ``o``."""
    rounds = (n + 255) // 256
    if not rowwarp() or rows < 1024 or rounds > 32:
        return None
    return f"""{{
const int lane_ = threadIdx.x & 31;
for (long long o = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5); o < {rows}LL; o += (long long)gridDim.x * (blockDim.x >> 5)) {{
  float a_[8];
#pragma unroll
  for (int k_ = 0; k_ < 8; ++k_) {{
    float acc = 0.0f;
#pragma unroll
    for (int m_ = 0; m_ < {rounds}; ++m_) {{
      const long long r = lane_ + 32 * k_ + 256LL * m_;
      if (r < {n}LL) acc += {read};
    }}
    a_[k_] = s101h::warp_sum(acc);
  }}
  float acc = 0.0f;
#pragma unroll
  for (int k_ = 0; k_ < 8; ++k_) if (lane_ == k_) acc = a_[k_];
  acc = s101h::warp_sum(acc);
  if (lane_ == 0) {store}
}}
}}"""
