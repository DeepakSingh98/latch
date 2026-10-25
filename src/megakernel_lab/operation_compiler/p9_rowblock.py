"""P9-A staged row-block engine for exact reduction phases (new in this lane).

A phase whose heavy items are all P6-D exact reductions over one row-major domain of R rows x E columns -- row
reductions (``p6d_exact_row_per_warp``: one output per row) and column reductions (``p6d_exact_column_wide``: one
output per column) -- plus the row consumers attached to its row reductions, runs as follows:

* Rows are split into GROUPS = 128 groups; group g owns rows o = g + 128 * sl (sl = 0, 1, ...). A CTA takes groups
  g = blockIdx.x, blockIdx.x + gridDim.x, ... (one group per CTA on a full H100 grid).
* The rows of a group of every *staged* tensor (a contiguous R x E operand read by the phase at its own row-major
  index) are copied into shared memory with ``cp.async.bulk`` (one bulk copy per tensor row, one mbarrier).
* Column reductions: the frozen tree's virtual thread t (0..255) sums rows t, t + 256, ... sequentially from +0.
  Rows t + 256 j with t = g + 128 u belong to group g, so the CTA of group g evaluates exactly the partial sums
  S_t[c] of its two virtual threads (same leaf expression, same order) from shared memory and stores them to an
  arena scratch buffer. After an in-phase grid barrier one warp per column applies the frozen butterflies
  (s101h::warp_sum over each virtual warp of 32 S_t, then over the 8 virtual-warp partials): bitwise equal.
* Row reductions: warp w evaluates the frozen per-row order (lane l carries virtual threads 32 k + l over
  r = 256 j + 32 k + l) for staged rows sl = w, w + 8, ... from shared memory, stores the value and runs the attached
  row consumers (unchanged expressions; staged reads from shared memory, others from global memory).

Only accesses whose index is the canonical own-row-major index are redirected; every other access (row scalars,
column broadcasts, non-staged tensors) is untouched. Values and arithmetic order are identical, so the results are
bitwise equal to the warp-per-row / column-wide engines. Greedy staging under SMEM_BUDGET picks the most-read
tensors first; a phase with no staged leaf operand keeps the inherited engines.
"""
from __future__ import annotations

import re

ROWBLOCK = True
SPLIT_STREAMS = False       # independent members of large elementwise loops are emitted as one loop per member
SPLIT_STREAMS_MIN = 1 << 20
LEAF_INLINE = True         # values read only by exact reduction leaves are evaluated in the leaves (fusion_p6d)
LEAF_INLINE_LIMIT = 24
FANIN_STORE = True         # store FP32 sums of single-use FP32 operands instead of inlining them (fusion_p6d)
WARP_SOFTMAX = True        # P-OPS2-D exact warp-per-row (log_)softmax rows in the P6-D schedule (bitwise)
GROUPS = 128
ENGINE = 3                 # 6: CTA-cooperative rows (engine 3 staging); 5: engine 3 with per-warp issue and per-row barriers; 1: one staged block per group; 2: double-buffered row chunks, staged consumers
CHUNK = 4                  # rows per chunk (engine 2)
ROW_WARPS = 2              # warps per row in the row consumers (engine 2)
SMEM_BUDGET = 192 * 1024
SMEM_BUDGET_V3 = 222 * 1024
PREFETCH_REST = False      # engine 3: L2 bulk prefetch of row-contiguous operands that are not staged
DISCARD = False            # P9-A: dead planned buffers are discarded from L2 after their last reader (no write-back)
DISCARD_MIN = 1 << 20
SIDE_CASTS = False         # P9-A: phase-0 FP32->BF16 input casts run as side work of earlier product phases (exact)
SIDE_MIN = 1 << 18
SIDE_WINDOW = 4            # candidate host phases: the last SIDE_WINDOW product phases before the first reader
CONSUMER_UNROLL = 4        # engine 6: rows per unrolled round of CTA element consumers (no prefetch)
HOIST_CONS = True          # engine 3: consumer row scalars in registers, column broadcasts as quad loads (exact)
REG_PREFETCH = False       # engine 3: non-staged vector consumer operands loaded into registers at group entry (exact)
REG_PREFETCH_WORDS = 64    # register budget (32-bit words per thread) for those operands
ALIGN = 128
DIAG = ()                  # timing diagnostics only (wrong results): 'no_consumer', 'no_rows', 'no_cols'
TRACE = False              # diagnostic: CTA 0 globaltimer stamps inside staged phases (p9_trace_read)
_TRACE_IDS = [0]
DEFER_FINALIZE = False     # engine 3: column finalize runs in the next product phase after each CTA's tiles (exact)
PART_T = False             # engine 3: column partials stored column-major (coalesced finalize reads; exact)
ROW_UNROLL = 2             # cap of the quad unroll of warp-row element consumers (code size)
COMPACT = True             # diagnostic/code-size option: row reduction j loop and column partial j loop not unrolled
# LATCH-DENSE-EXEC: engine 1 (the fallback when engines 3+ refuse more than 16 staged rows per group, i.e. more than
# 2048 token rows) unrolls its compact column-partial j loop by this factor, so each chain keeps that many leaf loads
# in flight instead of one memory latency per row. The chain still adds rows t + 256 j in ascending j (exact).
COLUMN_UNROLL = 8

HELPERS = r'''#ifndef P9_ROWBLOCK_HELPERS
#define P9_ROWBLOCK_HELPERS
__shared__ __align__(8) unsigned long long p9_bar;
__shared__ __align__(8) unsigned long long p9_bars[2];
__shared__ __align__(8) unsigned long long p9_rbars[17];  // P9-A engine 5: rows 0..15, broadcasts 16
__device__ __forceinline__ void p9_pf(const void* p, unsigned bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
template <class T> __device__ __forceinline__ void p9_cp4(T (&d)[4], const T (&s)[4]) {
  d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
}
__device__ __forceinline__ unsigned p9_sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void p9_bulk(unsigned dst, const void* src, unsigned bytes, unsigned bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void p9_wait(unsigned bar, unsigned parity) {
  asm volatile("{\n.reg .pred P1;\nP9_WAIT:\nmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n@P1 bra P9_DONE;\n"
               "bra P9_WAIT;\nP9_DONE:\n}\n" ::"r"(bar), "r"(parity) : "memory");
}
#endif
#ifndef P9_DIRECT_HELPERS
#define P9_DIRECT_HELPERS
// LATCH-FFN-RETAIN direct column engine: element q of a 16-byte register vector (bit-exact reinterpretation).
__device__ __forceinline__ __nv_bfloat16 p9d_bf(const uint4 v, const int q) {
  const unsigned w = q < 2 ? v.x : q < 4 ? v.y : q < 6 ? v.z : v.w;
  return __ushort_as_bfloat16((unsigned short)((q & 1) ? (w >> 16) : (w & 0xffffu)));
}
__device__ __forceinline__ float p9d_f(const uint4 v, const int q) {
  return __uint_as_float(q == 0 ? v.x : q == 1 ? v.y : q == 2 ? v.z : v.w);
}
#endif
#ifndef P9_TRACE_HELPERS
#define P9_TRACE_HELPERS
#ifdef P9_TRACE
__device__ unsigned long long p9_trace_buf[256][16];
#define P9T(ph, st) do { if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long t_; \
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); p9_trace_buf[ph][st] = t_; } } while (0)
#define P9TW(ph, st, tid) do { if (blockIdx.x == 0 && threadIdx.x == (tid)) { unsigned long long t_; \
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); p9_trace_buf[ph][st] = t_; } } while (0)
extern "C" int p9_trace_read(void* dst) { return (int)cudaMemcpyFromSymbol(dst, p9_trace_buf, sizeof(p9_trace_buf)); }
#ifndef P9_ATTN_DECL
#define P9_ATTN_DECL
__device__ unsigned long long p9_attn_buf[2][512][6];
__device__ unsigned p9_attn_seq[2];
#endif
#define P9AS(kind, st) do { if (blockIdx.x == 0 && (threadIdx.x & 127) == 0) { const int w_ = threadIdx.x >> 7; \
  unsigned long long t_; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_)); const unsigned s_ = p9_attn_seq[w_]; \
  if (s_ < 512) { p9_attn_buf[w_][s_][st] = t_; p9_attn_buf[w_][s_][2] = (kind); } if (st) p9_attn_seq[w_] = s_ + 1; } } while (0)
extern "C" int p9_attn_read(void* dst) { return (int)cudaMemcpyFromSymbol(dst, p9_attn_buf, sizeof(p9_attn_buf)); }
extern "C" int p9_trace_reset() { unsigned z[2] = {0u, 0u}; return (int)cudaMemcpyToSymbol(p9_attn_seq, z, sizeof(z)); }
#else
#define P9T(ph, st) do { } while (0)
#define P9TW(ph, st, tid) do { } while (0)
#define P9AS(kind, st) do { } while (0)
#endif
#endif
'''

_ACCESS = re.compile(r'\bv_\w+\[')
_VEC = re.compile(r'latch_ld4\((v_\w+) \+ 4u \* q, ')


def _norm(expr):
    return re.sub(r'[()\s]|unsigned', '', expr)


def _accesses(code):
    """(start, end, ident, index) of every v_X[...] access (bracket matched)."""
    out = []
    for m in _ACCESS.finditer(code):
        depth, k = 1, m.end()
        while depth:
            depth += {'[': 1, ']': -1}.get(code[k], 0)
            k += 1
        out.append((m.start(), k, m.group(0)[:-1], code[m.end():k - 1]))
    return out


def _rewrite(code, staged, canon, replacement):
    """Replace staged accesses whose normalized index equals `canon` by replacement(ident) (loads only)."""
    pieces, last = [], 0
    for start, end, ident, index in _accesses(code):
        store = code[end:end + 3] == ' = '
        if ident in staged and _norm(index) == canon and not store:
            pieces.append(code[last:start])
            pieces.append(replacement(ident))
            last = end
    pieces.append(code[last:])
    return ''.join(pieces)


def scratch_words(phases, nodes):
    """Arena words for the column partial sums of the largest eligible phase (256 x E per column reduction)."""
    best = 0
    for phase in phases:
        specs = [(nodes[n].extra or {}).get('p9') for n in phase['heavy']]
        if specs and all(specs):
            cols = [s for s in specs if s['kind'] == 'col']
            if cols:
                best = max(best, len(cols) * 256 * cols[0]['extent'])
        if COLUMN_DIRECT:
            # Direct engine: column reductions of any widths, beside inherited warp-per-row reductions.
            cols = [s for s in specs if s and s['kind'] == 'col']
            others = [n for n, s in zip(phase['heavy'], specs) if not (s and s['kind'] == 'col')]
            if cols and all(direct_inherited(nodes[n]) for n in others):
                best = max(best, 256 * sum(s['extent'] for s in cols))
    return best


def direct_inherited(node):
    """A heavy item the direct column engine runs unchanged in its phase (before the in-phase grid barrier)."""
    return (getattr(node, 'kind', 'reduction') == 'reduction'
            and (node.extra or {}).get('engine') == 'p6d_exact_row_per_warp' and not (node.extra or {}).get('p9'))


# LATCH-FFN-RETAIN: unstaged column engine for phases whose heavy items are all exact column reductions over the
# same row count (any mix of widths), e.g. a 2560-wide bias gradient beside a 640-wide LayerNorm weight gradient,
# where the staged engines refuse (rows do not fit shared memory, or the widths differ). The frozen tree is kept:
# virtual thread t (0..255) of output column oc sums leaves r = t + 256 j in ascending j from +0 (one register chain
# per (t, oc)). A lane owns DIRECT_VEC consecutive columns of DIRECT_CHAINS consecutive virtual threads; leaf operands
# read at the canonical own-row-major index r * E + oc are fetched as 16-byte vectors into registers for a window of
# rows (all loads of the window issue before its additions), every other access is unchanged. Partials go to the
# arena scratch; after one in-phase grid barrier, a CTA per 32 outputs transposes 256 x 32 partials through shared
# memory and one warp per output applies the unchanged butterflies (warp_sum over each virtual warp of 32 partials,
# then over the 8 virtual-warp results; lanes >= 8 contribute +0). Same additions in the same order: bitwise equal
# to the inherited engines.
COLUMN_DIRECT = True
DIRECT_CHAINS = 1
DIRECT_VEC = 8
DIRECT_WORDS = 96           # register words of loaded leaf operands per window (sets the row window per reduction)
DIRECT_MAX_WINDOW = 16
DIRECT_PAD = 33
DIRECT_SKIP = ()           # diagnostic timing only (wrong results): reductions whose partial loop is not emitted


def _row_of_column_index(index, E):
    """True when `index` is exactly ((unsigned)r * E + (unsigned)oc [* 1]) / E (C precedence, unsigned casts only).

    For 0 <= oc < E and r * E + oc < 2**31 (the engine's admitted positions) this equals r, so per-row operands
    (LayerNorm statistics, row scales) read once per row instead of through a division per element: same element.
    """
    import ast
    text = re.sub(r'\(unsigned\)', '', index)
    text = re.sub(r'\b(\d+)u\b', r'\1', text)
    if re.search(r'[^\w\s()*+/]', text):
        return False
    try:
        node = ast.parse(text, mode='eval').body
    except SyntaxError:
        return False
    const = lambda n, v: isinstance(n, ast.Constant) and n.value == v
    name = lambda n, v: isinstance(n, ast.Name) and n.id == v
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and const(node.right, E)):
        return False
    add = node.left
    if not (isinstance(add, ast.BinOp) and isinstance(add.op, ast.Add)):
        return False
    rows, cols = add.left, add.right
    row_ok = isinstance(rows, ast.BinOp) and isinstance(rows.op, ast.Mult) and name(rows.left, 'r') and const(rows.right, E)
    col_ok = name(cols, 'oc') or (isinstance(cols, ast.BinOp) and isinstance(cols.op, ast.Mult)
                                  and name(cols.left, 'oc') and const(cols.right, 1))
    return row_ok and col_ok


def _hoist_rows(leaf, E):
    pieces, last = [], 0
    for start, end, ident, index in _accesses(leaf):
        if _row_of_column_index(index, E):
            pieces.append(leaf[last:start])
            pieces.append(f'{ident}[(unsigned)r]')
            last = end
    pieces.append(leaf[last:])
    return ''.join(pieces)


DIRECT_ROW_HOIST = True


def emit_phase_direct(heavy, consumer_code, *, ident_dtype, ident_ok, written, elsize, scratch=True,
                      resolve=lambda t: t, inherited=(), **_):
    """inherited: (name, code) of the phase's other heavy items (direct_inherited), emitted unchanged after the
    column partials and before the grid barrier (they neither use shared memory nor read the partials)."""
    if not COLUMN_DIRECT or not scratch:
        return None
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s and s['kind'] == 'col' for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R = specs[0][1]['rows']
    U, V = DIRECT_CHAINS, DIRECT_VEC
    if (any(s['rows'] != R for _, s in specs) or R % 256 or R < 256 or 256 % U or V not in (4, 8)
            or any(s['extent'] % V for _, s in specs)):
        return None
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    plans = []
    for name, s in specs:
        E = s['extent']
        canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
        vectors = []
        for _, _, ident, index in _accesses(s['leaf']):
            dtype = ident_dtype(ident)
            if (_norm(index) == canon and ident not in vectors and dtype in elsize and ident not in written
                    and (V * elsize[dtype]) % 16 == 0 and ident_ok(ident, R, E)):
                vectors.append(ident)
        words = sum(V * elsize[ident_dtype(i)] // 4 for i in vectors) * U
        window = 1
        while window * 2 <= DIRECT_MAX_WINDOW and window * 2 * max(words, 1) <= DIRECT_WORDS and (R // 256) % (window * 2) == 0:
            window *= 2
        plans.append(dict(name=name, spec=s, E=E, canon=canon, vectors=vectors, window=window))
    bases, base = [], 0
    for plan in plans:
        bases.append(base)
        base += 256 * plan['E']
    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    lines = [f'// P9 direct column engine: {R} rows; ' + ', '.join(f"{p['name']} ({p['E']}, window {p['window']})" for p in plans),
             '{',
             '  const int lane = threadIdx.x & 31;',
             '  const long long gw = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;',
             '  const long long nw = ((long long)gridDim.x * blockDim.x) >> 5;',
             f'  P9T({tid}, 0);']
    start = 0
    for c, plan in enumerate(plans):
        E, J, name, s = plan['E'], plan['window'], plan['name'], plan['spec']
        blocks = (E + 32 * V - 1) // (32 * V)
        items = blocks * (256 // U)
        if name in DIRECT_SKIP:
            start += items
            continue
        regs = {i: f'p9d_{i[2:]}' for i in plan['vectors']}
        per = {i: 16 // elsize[ident_dtype(i)] for i in regs}            # elements per 16-byte vector
        unpack = {i: 'p9d_bf' if ident_dtype(i) == 'bfloat16' else 'p9d_f' for i in regs}
        leaf = _rewrite(s['leaf'], set(regs), plan['canon'],
                        lambda ident: f'{unpack[ident]}({regs[ident]}[jj][u][q / {per[ident]}], q % {per[ident]})')
        if DIRECT_ROW_HOIST:
            leaf = _hoist_rows(leaf, E)
        lines += [f'  // column partials: {name}',
                  # One grid-stride sequence over the concatenated items of every reduction (load balance).
                  f'  for (long long item = ((gw - {start}LL) % nw + nw) % nw; item < {items}LL; item += nw) {{',
                  f'    const long long oc0 = (item % {blocks}LL) * {32 * V} + lane * {V};',
                  f'    const long long t0 = (item / {blocks}LL) * {U};',
                  f'    const bool live = oc0 < {E}LL;',
                  f'    float acc[{U}][{V}];',
                  '#pragma unroll',
                  f'    for (int u = 0; u < {U}; ++u)',
                  '#pragma unroll',
                  f'      for (int q = 0; q < {V}; ++q) acc[u][q] = 0.0f;',
                  '#pragma unroll 1',
                  f'    for (int j = 0; j < {R // 256}; j += {J}) {{']
        for ident in plan['vectors']:
            lines.append(f'      uint4 {regs[ident]}[{J}][{U}][{V // per[ident]}];')
        lines += ['#pragma unroll',
                  f'      for (int jj = 0; jj < {J}; ++jj)',
                  '#pragma unroll',
                  f'        for (int u = 0; u < {U}; ++u) {{',
                  '          const long long r = t0 + u + 256LL * (j + jj);',
                  f'          const unsigned at = (unsigned)r * {E}u + (unsigned)oc0;',
                  '          if (live) {']
        for ident in plan['vectors']:
            for w in range(V // per[ident]):
                lines.append(f'            {regs[ident]}[jj][u][{w}] = *reinterpret_cast<const uint4*>({ident} + at + {w * per[ident]}u);')
        lines += ['          }',
                  '        }',
                  '      if (live) {',
                  '#pragma unroll',
                  f'      for (int jj = 0; jj < {J}; ++jj)',
                  '#pragma unroll',
                  f'        for (int u = 0; u < {U}; ++u) {{',
                  '          const long long r = t0 + u + 256LL * (j + jj);',
                  '#pragma unroll',
                  f'          for (int q = 0; q < {V}; ++q) {{',
                  '            const long long oc = oc0 + q;',
                  f'            acc[u][q] += {leaf};',
                  '          }',
                  '        }',
                  '      }',
                  '    }',
                  '    if (live) {',
                  '#pragma unroll',
                  f'      for (int u = 0; u < {U}; ++u)',
                  '#pragma unroll',
                  f'        for (int q = 0; q < {V}; q += 4)',
                  f'          *reinterpret_cast<float4*>(v_p9part + {bases[c]}LL + (t0 + u) * {E}LL + oc0 + q) = '
                  'make_float4(acc[u][q], acc[u][q + 1], acc[u][q + 2], acc[u][q + 3]);',
                  '    }',
                  '  }']
        start += items
    for name, code in inherited:
        lines += [f'  // inherited reduction: {name}', '  {', code, '  }']
    tiles = sum((p['E'] + 31) // 32 for p in plans)
    lines += [f'  P9T({tid}, 1);',
              '  grid.sync();                                                 // every partial sum is stored',
              f'  for (int tile = blockIdx.x; tile < {tiles}; tile += gridDim.x) {{',
              '    int c = 0, o0 = tile * 32;']
    for c, plan in enumerate(plans[:-1]):
        lines.append(f'    if (c == {c} && o0 >= {(plan["E"] + 31) // 32 * 32}) {{ o0 -= {(plan["E"] + 31) // 32 * 32}; c = {c + 1}; }}')
    lines += ['    long long base = 0, E = 0;']
    for c, plan in enumerate(plans):
        lines.append(f'    if (c == {c}) {{ base = {bases[c]}LL; E = {plan["E"]}LL; }}')
    lines += ['    __syncthreads();                                           // earlier shared-memory users are done',
              '    for (int t = threadIdx.x >> 5; t < 256; t += blockDim.x >> 5)',
              f'      smem[t * {DIRECT_PAD} + lane] = o0 + lane < E ? v_p9part[base + (long long)t * E + o0 + lane] : 0.0f;',
              '    __syncthreads();',
              '    for (int oo = threadIdx.x >> 5; oo < 32; oo += blockDim.x >> 5) {',
              '      float part = 0.0f;',
              '#pragma unroll',
              '      for (int i = 0; i < 8; ++i) {',
              f'        const float x = s101h::warp_sum(smem[(32 * i + lane) * {DIRECT_PAD} + oo]);',
              '        if (lane == i) part = x;',
              '      }',
              '      float a = lane < 8 ? part : 0.0f;',
              '      a = s101h::warp_sum(a);',
              '      const long long o = o0 + oo;']
    for c, plan in enumerate(plans):
        lines.append(f'      if (lane == 0 && c == {c} && o < {plan["E"]}LL) {plan["spec"]["out"]}[o] = {plan["spec"]["store"]};')
    lines += ['    }',
              '  }',
              '  __syncthreads();',
              f'  P9T({tid}, 2);',
              '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_column_direct',
              'p9_rowblock': {'rows': R, 'extents': [p['E'] for p in plans], 'staged': [],
                              'column_reductions': [p['name'] for p in plans], 'row_reductions': [],
                              'inherited_reductions': [name for name, _ in inherited],
                              'chains': U, 'vector': V, 'windows': [p['window'] for p in plans],
                              'vector_operands': [p['vectors'] for p in plans]},
              'shared_bytes': 256 * DIRECT_PAD * 4 + 1024}
    return '\n'.join(lines), record


def emit_phase(heavy, consumer_code, *, ident_dtype, ident_ok, written, elsize, scratch=True, resolve=lambda t: t):
    """Return (code, record) for an eligible phase, else None.

    heavy: heavy nodes of the phase; consumer_code(name) -> attached row-consumer code (or '');
    ident_dtype(ident) -> 'float32' | 'bfloat16' | None; ident_ok(ident, rows, extent) -> True when the identifier is a
    contiguous rows x extent operand of that dtype; written: identifiers written by this phase.
    """
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 8 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    consumers = {n: consumer_code(n) for n, _ in rows}
    counts = {}
    for _, s in specs:
        canon = row_canon if s['kind'] == 'row' else col_canon
        for _, _, ident, index in _accesses(s['leaf']):
            if _norm(index) == canon:
                counts[ident] = counts.get(ident, 0) + 4        # leaf reads weigh more than consumer reads
    for code in consumers.values():
        for ident in _VEC.findall(code):
            counts[ident] = counts.get(ident, 0) + 1
        for _, _, ident, index in _accesses(code):
            if _norm(index) == 'i':
                counts[ident] = counts.get(ident, 0) + 1
    candidates = [i for i in counts if i not in written and ident_dtype(i) in elsize and ident_ok(i, R, E)]
    candidates.sort(key=lambda i: (-counts[i], i))
    staged, offset = {}, 0
    for ident in candidates:
        size = (nrows_max * E * elsize[ident_dtype(ident)] + ALIGN - 1) // ALIGN * ALIGN
        if offset + size <= SMEM_BUDGET:
            staged[ident] = offset
            offset += size
    leaf_staged = any(counts.get(i, 0) >= 4 for i in staged)
    if not leaf_staged:
        return None
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    order = list(staged)
    lines = [f'// P9 staged row-block engine: {R} x {E}; staged {", ".join(order)}',
             '{',
             '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in order:
        t = ctype[ident_dtype(i)]
        lines.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {staged[i]});')
    lines += ['  const unsigned p9_b = p9_sa(&p9_bar);',
              '  __syncthreads();                                             // earlier shared-memory users are done',
              '  if (threadIdx.x == 0) {',
              '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b) : "memory");',
              '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
              '  }',
              '  __syncthreads();',
              f'  P9T({_TRACE_IDS[0]}, 0);',
              '  unsigned p9_parity = 0;',
              f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x, p9_parity ^= 1u) {{',
              f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
              '    if (threadIdx.x < 32) {',
              '      asm volatile("fence.proxy.async;" ::: "memory");        // generic accesses -> bulk copies',
              f'      if (threadIdx.x == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b), "r"((unsigned)nrows * {sum(row_bytes.values())}u) : "memory");',
              '      __syncwarp();',
              f'      for (int k = threadIdx.x; k < nrows * {len(order)}; k += 32) {{',
              f'        const int sl = k / {len(order)}, which = k % {len(order)};',
              f'        const long long row = (long long)g + {GROUPS}LL * sl;']
    for n, i in enumerate(order):
        lines.append(f'        if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)sl * {E}), {i} + row * {E}LL, {row_bytes[i]}u, p9_b);')
    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    lines += ['      }',
              '    }',
              '    p9_wait(p9_b, p9_parity);',
              f'    P9T({tid}, 1);']

    if 'no_cols' in DIAG:
        cols = []
    if 'no_rows' in DIAG:
        rows = []
    for c, (name, s) in enumerate(cols):
        leaf = _rewrite(s['leaf'], staged, col_canon, lambda ident: f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned)oc]')
        lines += [f'    // column partials: {name}',
                  f'    for (int idx = threadIdx.x; idx < {256 // GROUPS} * {E}; idx += blockDim.x) {{',
                  f'      const int u = idx / {E};',
                  f'      const long long oc = idx % {E};',
                  f'      const long long t = g + {GROUPS}LL * u;',
                  '      float acc = 0.0f;',
                  ((f'#pragma unroll {COLUMN_UNROLL}' if COLUMN_UNROLL > 1 else '#pragma unroll 1') if COMPACT
                   else '#pragma unroll'),
                  f'      for (int j = 0; j < {R // 256}; ++j) {{',
                  '        const long long r = t + 256LL * j;',
                  f'        const unsigned sl = (unsigned)(r - g) / {GROUPS}u;',
                  f'        acc += {leaf};',
                  '      }',
                  f'      v_p9part[{c * 256 * E}LL + t * {E}LL + oc] = acc;',
                  '    }']
    lines.append(f'    P9T({tid}, 2);')
    if rows:
        lines += ['    for (int sl = threadIdx.x >> 5; sl < nrows; sl += blockDim.x >> 5) {',
                  f'      const long long o = (long long)g + {GROUPS}LL * sl;',
                  '      const int lane = threadIdx.x & 31;']
        for name, s in rows:
            leaf = _rewrite(s['leaf'], staged, row_canon, lambda ident: f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned)rr]')
            consumer = '' if 'no_consumer' in DIAG else consumers[name]
            for ident in staged:
                consumer = consumer.replace(f'latch_ld4({ident} + 4u * q, ',
                                            f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
            consumer = _rewrite(consumer, staged, 'i',
                                lambda ident: f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)sl * {E}u)]')
            lines += [f'      {{ // row reduction: {name}',
                      '      float v[8];',
                      '#pragma unroll',
                      '      for (int k = 0; k < 8; ++k) v[k] = 0.0f;',
                      '#pragma unroll 1' if COMPACT else '#pragma unroll',
                      f'      for (int j = 0; j < {(E + 255) // 256}; ++j) {{',
                      '#pragma unroll',
                      '        for (int k = 0; k < 8; ++k) {',
                      '          const long long rr = 256LL * j + 32 * k + lane;',
                      f'          v[k] += {leaf};' if E % 256 == 0 else
                      f'          v[k] += rr < {E}LL ? {leaf} : 0.0f;',
                      '        }',
                      '      }',
                      '#pragma unroll',
                      '      for (int k = 0; k < 8; ++k) v[k] = s101h::warp_sum(v[k]);',
                      '      float second = 0.0f;',
                      '#pragma unroll',
                      '      for (int k = 0; k < 8; ++k) if (lane == k) second = v[k];',
                      '      second = s101h::warp_sum(second);',
                      f'      if (lane == 0) {s["out"]}[o] = {s["store"]};',
                      f'      {consumer}',
                      '      }']
        lines.append('    }')
    lines += [f'    P9T({tid}, 3);',
              '    __syncthreads();                                           // staged rows are reused by the next group',
              f'    P9T({tid}, 4);',
              '  }']
    if cols:
        lines += ['  grid.sync();                                                 // every partial sum is stored',
                  f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
                  'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
                  '    const int lane = threadIdx.x & 31;',
                  f'    const long long c = w / {E}, o = w % {E};',
                  '    float part = 0.0f;',
                  '#pragma unroll',
                  '    for (int i = 0; i < 8; ++i) {',
                  f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
                  '      if (lane == i) part = x;',
                  '    }',
                  '    float a = lane < 8 ? part : 0.0f;',
                  '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            lines.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        lines.append('  }')
    lines += [f'  P9T({tid}, 5);',
              '  __syncthreads();',
              '  if (threadIdx.x == 0) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b) : "memory");',
              '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock', 'p9_rowblock': {'rows': R, 'extent': E, 'staged': order,
                                                        'staged_bytes': offset, 'column_reductions': [n for n, _ in cols],
                                                        'row_reductions': [n for n, _ in rows]},
              'shared_bytes': offset + 1024}
    return '\n'.join(lines), record


def _blocks(consumer):
    """Split attached row-consumer code into its loop blocks ('{ // N elements ...' ... '}')."""
    head, rest = consumer.split('\n', 1) if consumer.startswith('__syncwarp();') else ('', consumer)
    blocks, current = [], []
    for line in rest.split('\n'):
        if line.startswith('{ // ') and current:
            blocks.append('\n'.join(current))
            current = []
        current.append(line)
    if current and any(l.strip() for l in current):
        blocks.append('\n'.join(current))
    return [b for b in blocks if b.strip()]


def emit_phase_v2(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t):
    """Engine 2: double-buffered chunks of CHUNK staged rows per group; row scalars and column broadcasts in shared
    memory; consumers staged and run by ROW_WARPS warps per row. Same expressions and orders as engine 1 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 8 or R < 256 or E % 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if (cols and (R % 256 or not scratch)) or len(rows) > CHUNK:
        return None
    if 'no_cols' in DIAG:
        cols = []
    nrows_max = (R + GROUPS - 1) // GROUPS
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'): 'rr',
                 _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'): 'oc', 'i%' + f'{E}u': f'(i % {E}u)'}
    consumers = {n: ('' if 'no_consumer' in DIAG else consumer_code(n)) for n, _ in rows}
    texts = [s['leaf'] for _, s in specs] + list(consumers.values())
    outs = {s['out'] for _, s in specs}
    # Classify every load.
    staged_count, scal, bcast = {}, set(), set()
    for text in texts:
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                staged_count[ident] = staged_count.get(ident, 0) + 1
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                staged_count[ident] = staged_count.get(ident, 0) + 1
    if any(i in written and i not in outs for i in scal | bcast):
        scal = {i for i in scal if i not in written or i in outs}
        bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in staged_count if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-staged_count[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    fixed = len(bcast) * E * 4 + len(scal) * nrows_max * 4 + 1024
    chunk = CHUNK
    staged = []
    while True:
        staged, used = [], fixed
        for ident in candidates:
            size = 2 * chunk * E * elsize[ident_dtype(ident)]
            if used + size <= SMEM_BUDGET:
                staged.append(ident)
                used += size
        if len(staged) == len(candidates) or chunk == 2:
            break
        chunk //= 2
    if not staged:
        return None
    if len(rows) > chunk:
        return None
    nchunks = (nrows_max + chunk - 1) // chunk
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())
    # Shared layout: column broadcasts, row scalars, then two buffers per staged tensor.
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    for i in staged:
        offsets[('s', i)] = off
        off += (2 * chunk * row_bytes[i] + ALIGN - 1) // ALIGN * ALIGN
    total = off

    def sub(text, canon_map):
        """canon_map: normalized-index -> replacement(ident) for the identifiers it covers."""
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = canon_map(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(local, column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9b_{ident[2:]}[(unsigned)({local}) * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal:
                return f'p9r_{ident[2:]}[r0 + ({local})]'
            if ident in bcast and form in row_bcast:
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9b_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)p9_r * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[r0 + p9_r]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    def consumer_text(name):
        text = consumers[name]
        if not text:
            return ''
        out = []
        for block in _blocks(text):
            elementwise = f'qend = (unsigned)o * 1u + 1u' not in block
            if elementwise:
                block = block.replace('const unsigned gs = 32u;', f'const unsigned gs = {32 * ROW_WARPS}u;')
                block = re.sub(r'for \(unsigned base = \(unsigned\)o \* (\d+)u \+ \(unsigned\)lane;',
                               lambda m: f'for (unsigned base = (unsigned)o * {m.group(1)}u + (unsigned)lane + 32u * (unsigned)p9_h;',
                               block)
                for ident in staged:
                    block = block.replace(f'latch_ld4({ident} + 4u * q, ',
                                          f'latch_ld4(p9b_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)p9_r * {E}u), ')
                block = sub(block, cons_map)
                out.append(block)
            else:
                out.append('if (p9_h == 0) ' + sub(block, cons_map))
        return '\n'.join(out)

    L = [f'// P9 staged row-block engine v2: {R} x {E}, {chunk}-row chunks x 2 buffers; staged {", ".join(staged)}',
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  const unsigned p9_b0 = p9_sa(&p9_bars[0]), p9_b1 = p9_sa(&p9_bars[1]);',
          '  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b1) : "memory");',
          '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  }']
    for i in sorted(bcast):
        L.append(f'  for (int c = threadIdx.x; c < {E}; c += blockDim.x) p9c_{i[2:]}[c] = {i}[c];')
    L += ['  __syncthreads();',
          '  unsigned p9_use0 = 0, p9_use1 = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          f'    const int nchunks = (nrows + {chunk - 1}) / {chunk};',
          '    auto p9_issue = [&](int k) {                               // warp 0: bulk copies of chunk k',
          '      if (threadIdx.x < 32 && k < nchunks) {',
          '        asm volatile("fence.proxy.async;" ::: "memory");',
          f'        const int r0 = k * {chunk}, nr = nrows - r0 < {chunk} ? nrows - r0 : {chunk};',
          '        const unsigned bar = (k & 1) ? p9_b1 : p9_b0;',
          f'        if (threadIdx.x == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"((unsigned)nr * {per_row}u) : "memory");',
          '        __syncwarp();',
          f'        for (int x = threadIdx.x; x < nr * {len(staged)}; x += 32) {{',
          f'          const int r = x / {len(staged)}, which = x % {len(staged)};',
          f'          const long long row = (long long)g + {GROUPS}LL * (r0 + r);']
    for n, i in enumerate(staged):
        L.append(f'          if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)((k & 1) * {chunk} + r) * {E}), {i} + row * {E}LL, {row_bytes[i]}u, bar);')
    L += ['        }',
          '      }',
          '    };',
          '    p9_issue(0);',
          '    p9_issue(1);']
    if scal - outs:
        L.append(f'    for (int x = threadIdx.x; x < nrows * {len(scal - outs)}; x += blockDim.x) {{')
        L.append('      const int r = x % nrows, which = x / nrows;')
        L.append(f'      const long long row = (long long)g + {GROUPS}LL * r;')
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'      if (which == {n}) p9r_{i[2:]}[r] = {i}[row];')
        L.append('    }')
        L.append('    __syncthreads();                                         // row scalars of the group')
    col_threads = 256 - 32 * chunk if rows else 256       # warps without a row of the chunk run the column partials
    col_first = 256 - col_threads
    nacc = 2 * E // col_threads
    for c, (name, s) in enumerate(cols):
        L.append(f'    float p9a{c}[{nacc}];')
        L.append('#pragma unroll')
        L.append(f'    for (int m = 0; m < {nacc}; ++m) p9a{c}[m] = 0.0f;')
    L += ['    for (int k = 0; k < nchunks; ++k) {',
          f'      const int r0 = k * {chunk}, nr = nrows - r0 < {chunk} ? nrows - r0 : {chunk};',
          '      if (k & 1) { p9_wait(p9_b1, p9_use1 & 1u); ++p9_use1; } else { p9_wait(p9_b0, p9_use0 & 1u); ++p9_use0; }']
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'      const {t}* p9b_{i[2:]} = p9s_{i[2:]} + (k & 1) * {chunk * E};')
    for c, (name, s) in enumerate(cols):
        leaf = sub(s['leaf'], leaf_map('p9_rl', 'oc'))
        L += [f'      // column partials: {name} (virtual thread t = g + {GROUPS} u sums rows t + 256 j in order of j)',
              f'      if (threadIdx.x >= {col_first}) {{',
              '#pragma unroll',
              f'      for (int m = 0; m < {nacc}; ++m) {{',
              f'        const int idx = (threadIdx.x - {col_first}) + {col_threads} * m, u = idx / {E};',
              f'        const long long oc = idx % {E};',
              f'        for (int p9_rl = u; p9_rl < nr; p9_rl += 2) {{   // chunk rows of parity u, ascending (chunk start is even)',
              f'          const long long r = (long long)g + {GROUPS}LL * (r0 + p9_rl);',
              f'          p9a{c}[m] += {leaf};',
              '        }',
              '      }',
              '      }']
    if rows:
        L += ['      {',
              '        const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;',
              f'        if (w < {len(rows)} * 0 + {chunk} && w < nr) {{',
              '          const int p9_r = w;',
              f'          const long long o = (long long)g + {GROUPS}LL * (r0 + p9_r);']
        for name, s in rows:
            leaf = sub(s['leaf'], leaf_map('p9_r', 'rr'))
            L += [f'          {{ // row reduction: {name}',
                  '          float v[8];',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) v[k8] = 0.0f;',
                  '#pragma unroll',
                  f'          for (int j = 0; j < {E // 256}; ++j) {{',
                  '#pragma unroll',
                  '            for (int k8 = 0; k8 < 8; ++k8) {',
                  '              const long long rr = 256LL * j + 32 * k8 + lane;',
                  f'              v[k8] += {leaf};',
                  '            }',
                  '          }',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) v[k8] = s101h::warp_sum(v[k8]);',
                  '          float second = 0.0f;',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) if (lane == k8) second = v[k8];',
                  '          second = s101h::warp_sum(second);',
                  f'          const float p9_val = {s["store"]};',
                  f'          if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'          if (lane == 0) p9r_{s["out"][2:]}[r0 + p9_r] = p9_val;')
            L.append('          }')
        L += ['        }',
              '      }',
              '      __syncthreads();                                         // row values and partial sums complete']
        cons = '\n'.join(consumer_text(n) for n, _ in rows)
        if cons.strip():
            L += ['      {',
                  '        const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;',
                  f'        const int p9_r = w % {chunk}, p9_h = w / {chunk};',
                  f'        if (p9_r < nr && p9_h < {ROW_WARPS}) {{',
                  f'          const long long o = (long long)g + {GROUPS}LL * (r0 + p9_r);',
                  cons,
                  '        }',
                  '      }']
    L += ['      __syncthreads();                                         // chunk buffer free',
          '      p9_issue(k + 2);',
          '    }']
    for c, (name, s) in enumerate(cols):
        L += [f'    if (threadIdx.x >= {col_first}) {{',
              '#pragma unroll',
              f'    for (int m = 0; m < {nacc}; ++m) {{',
              f'      const int idx = (threadIdx.x - {col_first}) + {col_threads} * m, u = idx / {E};',
              f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a{c}[m];',
              '    }',
              '    }']
    L.append('  }')
    if cols:
        L += ['  grid.sync();                                                 // every partial sum is stored',
              f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            L.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        L.append('  }')
    L += ['  __syncthreads();',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b1) : "memory");',
          '  }',
          '}']
    record = {'engine': 'p9_rowblock2', 'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'chunk': chunk,
                                                        'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast),
                                                        'staged_bytes': total, 'column_reductions': [n for n, _ in cols],
                                                        'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    return '\n'.join(L), record



def hoist_consumer(text, E):
    """P9-A (exact): row scalars p9r_X[sl] of a warp-row consumer are read once into registers; column broadcasts
    p9c_X[(i % E)] of a 4-wide quad are one vector shared load per quad. Every value is unchanged."""
    names = sorted(set(re.findall(r'\bp9r_(\w+)\[sl\]', text)))
    for n in names:
        text = text.replace(f'p9r_{n}[sl]', f'p9h_{n}')
    cols = sorted(set(re.findall(r'\bp9c_(\w+)\[\(i % ' + str(E) + r'u\)\]', text)))
    marker = '      if (q < qend) {\n'
    if cols and marker in text and 'latch_ld4(' in text:
        loads = ''.join(f'        float p9q_{n}[4]; latch_ld4(p9c_{n} + (4u * q) % {E}u, p9q_{n});\n' for n in cols)
        text = text.replace(marker, marker + loads, 1)
        for n in cols:
            text = text.replace(f'p9c_{n}[(i % {E}u)]', f'p9q_{n}[j]')
    head = ''.join(f'const float p9h_{n} = p9r_{n}[sl]; ' for n in names)
    if not head:
        return text
    sync = '__syncwarp();'
    k = text.find(sync)
    k = k + len(sync) if k >= 0 and text[:k].strip() == '' else 0   # after the row value's warp barrier
    return text[:k] + '\n{ ' + head + '\n' + text[k:] + '\n}'


def emit_phase_v3(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t, defer_finalize=False):
    """Engine 3: all rows of a group staged at once under SMEM_BUDGET_V3 (two half-group barriers: warp w computes its
    first row sl = w while rows 8.. still land), L2 prefetch of row operands that do not fit, row scalars and column
    broadcasts in shared memory, warp-per-row reductions and consumers, then ILP-unrolled column partials. Every value
    and every addition order is that of engines 1/2 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 256 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    if nrows_max > 16:
        return None
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'),
                 'i%' + f'{E}u'}
    consumers = {n: consumer_code(n) for n, _ in rows}
    outs = {s['out'] for _, s in specs}
    counts, scal, bcast = {}, set(), set()
    for kind, text in [('leaf', s['leaf']) for _, s in specs] + [('cons', c) for c in consumers.values()]:
        weight = 4 if kind == 'leaf' else 1
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
    scal = {i for i in scal if i not in written or i in outs}
    bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in counts if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-counts[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    staged, rest = [], []
    for i in candidates:
        size = (nrows_max * E * elsize[ident_dtype(i)] + ALIGN - 1) // ALIGN * ALIGN
        if off + size <= SMEM_BUDGET_V3:
            offsets[('s', i)] = off
            off += size
            staged.append(i)
        else:
            rest.append(i)
    if not any(counts[i] >= 4 for i in staged):
        return None
    total = off
    half = 8
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())

    def sub(text, f):
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = f(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal and form not in ('i', 'i/' + f'{E}u'):
                return f'p9r_{ident[2:]}[sl]'
            if ident in bcast and form in row_bcast and form != 'i%' + f'{E}u':
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)sl * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[sl]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    Q = E // 4
    H = (nrows_max + 7) // 8
    header = re.compile(r'for \(unsigned base = \(unsigned\)o \* ' + str(Q) + r'u \+ \(unsigned\)lane; base < qend; '
                        r'base \+= (\d+)u \* gs\) \{')
    gsdecl = f'const unsigned gs = 32u; const unsigned qend = (unsigned)o * {Q}u + {Q}u;'

    def vector_form(text):
        m = header.search(text)
        if not m or gsdecl not in text or len(header.findall(text)) != 1:
            return None
        U = int(m.group(1))
        return U if Q % (32 * U) == 0 else None

    # Non-staged operands read only as vector consumer loads of the own row: issued into registers at group entry.
    pref = []
    if REG_PREFETCH and 'no_consumer' not in DIAG:
        words = 0
        forms = {n: vector_form(c) for n, c in consumers.items() if c}
        for i in rest:
            uses = [n for n, c in consumers.items() if c and f'{i}[' in c]
            vec = [n for n, c in consumers.items() if c and f'latch_ld4({i} + 4u * q, ' in c]
            if uses or not vec or any(forms.get(n) is None for n in vec) or ident_dtype(i) not in ctype:
                continue
            w = H * (Q // 32) * 4 * elsize[ident_dtype(i)] // 4
            if words + w <= REG_PREFETCH_WORDS:
                pref.append(i)
                words += w

    def consumer_text(name):
        text = consumers[name]
        if not text or 'no_consumer' in DIAG:
            return ''
        for ident in staged:
            text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
        if pref and any(f'latch_ld4({i} + 4u * q, ' in text for i in pref):
            U = vector_form(text)
            text = header.sub(f'#pragma unroll\n  for (unsigned p9it = 0; p9it < {Q // (32 * U)}u; ++p9it) {{\n'
                              f'    const unsigned base = (unsigned)o * {Q}u + (unsigned)lane + p9it * {32 * U}u;', text)
            for i in pref:
                text = re.sub(r'latch_ld4\(' + i + r' \+ 4u \* q, (\w+)\);',
                              lambda m: f'p9_cp4({m.group(1)}, p9g_{i[2:]}[p9h][p9it * {U}u + (unsigned)u]);', text)
        text = sub(text, cons_map)
        return hoist_consumer(text, E) if HOIST_CONS else text

    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    L = [f'// P9 staged row-block engine v3: {R} x {E}; staged {", ".join(staged)}'
         + (f'; register-prefetched {", ".join(pref)}' if pref else '')
         + (f'; L2-prefetched {", ".join(r for r in rest if r not in pref)}' if PREFETCH_REST and rest else ''),
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  const unsigned p9_b0 = p9_sa(&p9_bars[0]), p9_b1 = p9_sa(&p9_bars[1]);',
          '  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b1) : "memory");',
          '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  }',
          '  __syncthreads();',
          f'  P9T({tid}, 0);',
          '  unsigned p9_parity = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x, p9_parity ^= 1u) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          f'    const int n0 = nrows < {half} ? nrows : {half}, n1 = nrows - n0;',
          '    if (threadIdx.x < 32) {                                    // rows [0, 8) on barrier 0, the rest on barrier 1',
          '      asm volatile("fence.proxy.async;" ::: "memory");',
          '      if (threadIdx.x == 0) {',
          f'        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b0), "r"((unsigned)n0 * {per_row}u + {len(bcast) * E * 4}u) : "memory");',
          f'        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b1), "r"((unsigned)n1 * {per_row}u) : "memory");']
    for i in sorted(bcast):                                   # column broadcasts first: one bulk copy each, barrier 0
        L.append(f'        p9_bulk(p9_sa(p9c_{i[2:]}), {i}, {E * 4}u, p9_b0);')
    L += ['      }',
          '      __syncwarp();',
          f'      for (int k = threadIdx.x; k < nrows * {len(staged)}; k += 32) {{',
          f'        const int sl = k / {len(staged)}, which = k % {len(staged)};',
          f'        const long long row = (long long)g + {GROUPS}LL * sl;',
          f'        const unsigned bar = sl < {half} ? p9_b0 : p9_b1;']
    for n, i in enumerate(staged):
        L.append(f'        if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)sl * {E}), {i} + row * {E}LL, {row_bytes[i]}u, bar);')
    L += ['      }',
          '    }']
    if rest and PREFETCH_REST:
        L += ['    else if (threadIdx.x < 64) {                               // L2 prefetch of the non-staged row operands',
              f'      for (int k = threadIdx.x - 32; k < nrows * {len(rest)}; k += 32) {{',
              f'        const int sl = k / {len(rest)}, which = k % {len(rest)};',
              f'        const long long row = (long long)g + {GROUPS}LL * sl;']
        for n, i in enumerate(rest):
            L.append(f'        if (which == {n}) p9_pf({i} + row * {E}LL, {E * elsize[ident_dtype(i)]}u);')
        L += ['      }',
              '    }']
    if pref and rows and 'no_rows' not in DIAG:                # own-row consumer operands into registers
        for i in pref:
            L.append(f'    {ctype[ident_dtype(i)]} p9g_{i[2:]}[{H}][{Q // 32}][4];')
        L += ['    {',
              '      const unsigned lane = threadIdx.x & 31;',
              '#pragma unroll',
              f'      for (int p9h = 0; p9h < {H}; ++p9h) {{',
              '        const int sl = (threadIdx.x >> 5) + 8 * p9h;',
              f'        const unsigned o4 = (unsigned)(g + {GROUPS} * sl) * {Q}u + lane;',
              '        if (sl < nrows) {',
              '#pragma unroll',
              f'          for (int k = 0; k < {Q // 32}; ++k) {{']
        for i in pref:
            L.append(f'            latch_ld4({i} + 4u * (o4 + 32u * k), p9g_{i[2:]}[p9h][k]);')
        L += ['          }',
              '        }',
              '      }',
              '    }']
    if scal - outs:                                            # row scalars by warps 1.. while warp 0 issues copies
        L.append(f'    for (int x = threadIdx.x - 32; x >= 0 && x < nrows * {len(scal - outs)}; x += blockDim.x - 32) {{')
        L.append('      const int r = x % nrows, which = x / nrows;')
        L.append(f'      const long long row = (long long)g + {GROUPS}LL * r;')
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'      if (which == {n}) p9r_{i[2:]}[r] = {i}[row];')
        L.append('    }')
    L.append(f'    P9TW({tid}, 13, 0);')
    L.append('    __syncthreads();                                         // shared row scalars and broadcasts')
    L.append(f'    P9T({tid}, 12);')
    if rows and 'no_rows' not in DIAG:
        loop = (['#pragma unroll',
                 f'      for (int p9h = 0; p9h < {H}; ++p9h) {{',
                 '        const int sl = (threadIdx.x >> 5) + 8 * p9h;',
                 '        if (sl >= nrows) break;'] if pref else
                [f'      for (int sl = threadIdx.x >> 5; sl < nrows; sl += blockDim.x >> 5) {{'])
        L += ['    for (int p9rep = 0; p9rep < 2; ++p9rep)' if 'twice' in DIAG else '',
              '    {',
              '      const int lane = threadIdx.x & 31;'] + loop + [
              f'        p9_wait(p9_b0, p9_parity);',
              f'        if (sl >= {half}) p9_wait(p9_b1, p9_parity);',
              f'        if (sl == 0) P9T({tid}, 5); else if (sl == 8) P9T({tid}, 7);',
              f'        const long long o = (long long)g + {GROUPS}LL * sl;']
        for name, s in rows:
            leaf = sub(s['leaf'], leaf_map('rr'))
            L += [f'        {{ // row reduction: {name}',
                  '        float v[8];',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = 0.0f;',
                  '#pragma unroll 1' if COMPACT else '#pragma unroll',
                  f'        for (int j = 0; j < {E // 256}; ++j) {{',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) {',
                  '            const long long rr = 256LL * j + 32 * k8 + lane;',
                  f'            v[k8] += {leaf};',
                  '          }',
                  '        }',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = s101h::warp_sum(v[k8]);',
                  '        float second = 0.0f;',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) if (lane == k8) second = v[k8];',
                  '        second = s101h::warp_sum(second);',
                  f'        const float p9_val = {s["store"]};',
                  f'        if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'        if (lane == 0) p9r_{s["out"][2:]}[sl] = p9_val;')
            L += [f'        if (sl == 0) P9T({tid}, 8);',
                  f'        {consumer_text(name)}',
                  '        }']
        L += [f'        if (sl == 0) P9T({tid}, 6);',
              f'        if (sl == 1) P9TW({tid}, 9, 32);',
              f'        if (sl == 7) P9TW({tid}, 10, 224);',
              f'        if (sl == 15) P9TW({tid}, 11, 224);',
              '      }',
              '    }']
    L.append(f'    P9T({tid}, 1);')
    if cols and 'no_cols' not in DIAG:
        nacc = 2 * E // 256
        L += ['    p9_wait(p9_b0, p9_parity);',
              '    if (n1) p9_wait(p9_b1, p9_parity);']
        for c, (name, s) in enumerate(cols):
            leaf = sub(s['leaf'], leaf_map('oc'))
            L += [f'    {{ // column partials: {name} (virtual thread t = g + {GROUPS} u sums rows t + 256 j, ascending j)',
                  f'    float p9a[{nacc}];',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) p9a[m] = 0.0f;',
                  f'    for (int j = 0; j < {R // 256}; ++j) {{',
                  '#pragma unroll',
                  f'      for (int m = 0; m < {nacc}; ++m) {{',
                  f'        const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'        const long long oc = idx % {E};',
                  '        const int sl = u + 2 * j;',
                  f'        const long long r = (long long)g + {GROUPS}LL * sl;',
                  f'        p9a[m] += {leaf};',
                  '      }',
                  '    }',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) {{',
                  f'      const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  (f'      v_p9part[{c * 256 * E}LL + (long long)(idx % {E}) * 256LL + (g + {GROUPS} * u)] = p9a[m];' if PART_T else
                   f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a[m];'),
                  '    }',
                  '    }']
    L += [f'    P9T({tid}, 2);',
          '    __syncthreads();                                         // staged rows are reused by the next group',
          '  }',
          f'  P9T({tid}, 3);']
    finalize = []
    if cols and 'no_cols' not in DIAG:
        if not defer_finalize:
            L.append('  grid.sync();                                                 // every partial sum is stored')
        finalize += [f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              (f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + o * 256LL + 32LL * i + lane]);' if PART_T else
               f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);'),
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            finalize.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        finalize.append('  }')
        if not defer_finalize:
            L += finalize
    L += [f'  P9T({tid}, 4);',
          '  __syncthreads();',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b1) : "memory");',
          '  }',
          '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock3',
              'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'prefetched': rest,
                              'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast), 'staged_bytes': total, 'register_prefetched': pref,
                              'column_reductions': [n for n, _ in cols], 'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    if defer_finalize and finalize:
        record['p9_deferred_finalize'] = '\n'.join(finalize)
        record['p9_deferred_outputs'] = [s['out'] for _, s in cols]
    return '\n'.join(L), record


def emit_phase_v6(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t):
    """Engine 6 = engine 3 staging with CTA-cooperative rows: thread t accumulates the frozen-order partial of virtual
    thread t (elements t + 256 j, ascending j) for every row of a half group (8 independent rows), the warps reduce
    them together, warp w finishes rows w and w + 8 (the final 8-value tree, scalar consumers) and every element
    consumer runs over the whole CTA. Values and addition orders are those of engine 3 (bitwise). Engine 3: all rows of a group staged at once under SMEM_BUDGET_V3 (two half-group barriers: warp w computes its
    first row sl = w while rows 8.. still land), L2 prefetch of row operands that do not fit, row scalars and column
    broadcasts in shared memory, warp-per-row reductions and consumers, then ILP-unrolled column partials. Every value
    and every addition order is that of engines 1/2 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 256 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    if nrows_max > 16:
        return None
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'),
                 'i%' + f'{E}u'}
    consumers = {n: consumer_code(n) for n, _ in rows}
    outs = {s['out'] for _, s in specs}
    counts, scal, bcast = {}, set(), set()
    for kind, text in [('leaf', s['leaf']) for _, s in specs] + [('cons', c) for c in consumers.values()]:
        weight = 4 if kind == 'leaf' else 1
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
    scal = {i for i in scal if i not in written or i in outs}
    bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in counts if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-counts[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    for name, _ in rows:
        offsets[('w', name)] = off
        off += 16 * 8 * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    staged, rest = [], []
    for i in candidates:
        size = (nrows_max * E * elsize[ident_dtype(i)] + ALIGN - 1) // ALIGN * ALIGN
        if off + size <= SMEM_BUDGET_V3:
            offsets[('s', i)] = off
            off += size
            staged.append(i)
        else:
            rest.append(i)
    if not any(counts[i] >= 4 for i in staged):
        return None
    total = off
    half = 8
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())

    def sub(text, f):
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = f(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal and form not in ('i', 'i/' + f'{E}u'):
                return f'p9r_{ident[2:]}[sl]'
            if ident in bcast and form in row_bcast and form != 'i%' + f'{E}u':
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)sl * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[sl]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    Q = E // 4
    H = (nrows_max + 7) // 8
    header = re.compile(r'for \(unsigned base = \(unsigned\)o \* ' + str(Q) + r'u \+ \(unsigned\)lane; base < qend; '
                        r'base \+= (\d+)u \* gs\) \{')
    gsdecl = f'const unsigned gs = 32u; const unsigned qend = (unsigned)o * {Q}u + {Q}u;'

    def vector_form(text):
        m = header.search(text)
        if not m or gsdecl not in text or len(header.findall(text)) != 1:
            return None
        U = int(m.group(1))
        return U if Q % (32 * U) == 0 else None

    # Non-staged operands read only as vector consumer loads of the own row: issued into registers at group entry.
    pref = []
    if REG_PREFETCH and 'no_consumer' not in DIAG:
        words = 0
        forms = {n: vector_form(c) for n, c in consumers.items() if c}
        for i in rest:
            uses = [n for n, c in consumers.items() if c and f'{i}[' in c]
            vec = [n for n, c in consumers.items() if c and f'latch_ld4({i} + 4u * q, ' in c]
            if uses or not vec or any(forms.get(n) is None for n in vec) or ident_dtype(i) not in ctype:
                continue
            w = H * (Q // 32) * 4 * elsize[ident_dtype(i)] // 4
            if words + w <= REG_PREFETCH_WORDS:
                pref.append(i)
                words += w

    def consumer_text(name):
        text = consumers[name]
        if not text or 'no_consumer' in DIAG:
            return ''
        for ident in staged:
            text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
        if pref and any(f'latch_ld4({i} + 4u * q, ' in text for i in pref):
            U = vector_form(text)
            text = header.sub(f'#pragma unroll\n  for (unsigned p9it = 0; p9it < {Q // (32 * U)}u; ++p9it) {{\n'
                              f'    const unsigned base = (unsigned)o * {Q}u + (unsigned)lane + p9it * {32 * U}u;', text)
            for i in pref:
                text = re.sub(r'latch_ld4\(' + i + r' \+ 4u \* q, (\w+)\);',
                              lambda m: f'p9_cp4({m.group(1)}, p9g_{i[2:]}[p9h][p9it * {U}u + (unsigned)u]);', text)
        return sub(text, cons_map)

    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    L = [f'// P9 staged row-block engine v6: {R} x {E}; staged {", ".join(staged)}'
         + (f'; register-prefetched {", ".join(pref)}' if pref else '')
         + (f'; L2-prefetched {", ".join(r for r in rest if r not in pref)}' if PREFETCH_REST and rest else ''),
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for name, _ in rows:
        L.append(f'  float* p9w_{name} = reinterpret_cast<float*>(p9_smem + {offsets[("w", name)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  const unsigned p9_b0 = p9_sa(&p9_bars[0]), p9_b1 = p9_sa(&p9_bars[1]);',
          '  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b1) : "memory");',
          '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  }',
          '  __syncthreads();',
          f'  P9T({tid}, 0);',
          '  unsigned p9_parity = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x, p9_parity ^= 1u) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          *([] if nrows_max <= 16 else ['#error engine 6 needs at most 16 rows per group']),
          f'    const int n0 = nrows < {half} ? nrows : {half}, n1 = nrows - n0;',
          '    if (threadIdx.x < 32) {                                    // rows [0, 8) on barrier 0, the rest on barrier 1',
          '      asm volatile("fence.proxy.async;" ::: "memory");',
          '      if (threadIdx.x == 0) {',
          f'        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b0), "r"((unsigned)n0 * {per_row}u + {len(bcast) * E * 4}u) : "memory");',
          f'        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b1), "r"((unsigned)n1 * {per_row}u) : "memory");']
    for i in sorted(bcast):                                   # column broadcasts first: one bulk copy each, barrier 0
        L.append(f'        p9_bulk(p9_sa(p9c_{i[2:]}), {i}, {E * 4}u, p9_b0);')
    L += ['      }',
          '      __syncwarp();',
          f'      for (int k = threadIdx.x; k < nrows * {len(staged)}; k += 32) {{',
          f'        const int sl = k / {len(staged)}, which = k % {len(staged)};',
          f'        const long long row = (long long)g + {GROUPS}LL * sl;',
          f'        const unsigned bar = sl < {half} ? p9_b0 : p9_b1;']
    for n, i in enumerate(staged):
        L.append(f'        if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)sl * {E}), {i} + row * {E}LL, {row_bytes[i]}u, bar);')
    L += ['      }',
          '    }']
    if rest and PREFETCH_REST:
        L += ['    else if (threadIdx.x < 64) {                               // L2 prefetch of the non-staged row operands',
              f'      for (int k = threadIdx.x - 32; k < nrows * {len(rest)}; k += 32) {{',
              f'        const int sl = k / {len(rest)}, which = k % {len(rest)};',
              f'        const long long row = (long long)g + {GROUPS}LL * sl;']
        for n, i in enumerate(rest):
            L.append(f'        if (which == {n}) p9_pf({i} + row * {E}LL, {E * elsize[ident_dtype(i)]}u);')
        L += ['      }',
              '    }']
    if pref and rows and 'no_rows' not in DIAG:                # own-row consumer operands into registers
        for i in pref:
            L.append(f'    {ctype[ident_dtype(i)]} p9g_{i[2:]}[{H}][{Q // 32}][4];')
        L += ['    {',
              '      const unsigned lane = threadIdx.x & 31;',
              '#pragma unroll',
              f'      for (int p9h = 0; p9h < {H}; ++p9h) {{',
              '        const int sl = (threadIdx.x >> 5) + 8 * p9h;',
              f'        const unsigned o4 = (unsigned)(g + {GROUPS} * sl) * {Q}u + lane;',
              '        if (sl < nrows) {',
              '#pragma unroll',
              f'          for (int k = 0; k < {Q // 32}; ++k) {{']
        for i in pref:
            L.append(f'            latch_ld4({i} + 4u * (o4 + 32u * k), p9g_{i[2:]}[p9h][k]);')
        L += ['          }',
              '        }',
              '      }',
              '    }']
    if scal - outs:                                            # row scalars by warps 1.. while warp 0 issues copies
        L.append(f'    for (int x = threadIdx.x - 32; x >= 0 && x < nrows * {len(scal - outs)}; x += blockDim.x - 32) {{')
        L.append('      const int r = x % nrows, which = x / nrows;')
        L.append(f'      const long long row = (long long)g + {GROUPS}LL * r;')
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'      if (which == {n}) p9r_{i[2:]}[r] = {i}[row];')
        L.append('    }')
    L.append(f'    P9TW({tid}, 13, 0);')
    L.append('    __syncthreads();                                         // shared row scalars and broadcasts')
    L.append(f'    P9T({tid}, 12);')
    if rows and 'no_rows' not in DIAG:
        parts = {name: consumer_code.parts(name) for name, _ in rows}
        L += ['    {',
              '      const int t = threadIdx.x, lane = t & 31, wq = t >> 5;']
        for h, bar in ((0, 'p9_b0'), (1, 'p9_b1')):
            if h == 1:
                L.append('      if (n1) {')
            L += [f'      p9_wait({bar}, p9_parity);',
                  *( [f'      P9T({tid}, 5);'] if h == 0 else [])]
            for name, s in rows:
                leaf = sub(s['leaf'], leaf_map('rr'))
                L += [f'      {{ // row partials: {name}, rows {8 * h}..{8 * h + 7} (virtual thread t: elements t + 256 j, ascending j)',
                      '      float pa[8];',
                      '#pragma unroll',
                      '      for (int r = 0; r < 8; ++r) {',
                      f'        const int sl = {8 * h} + r;',
                      f'        const long long o = (long long)g + {GROUPS}LL * sl;',
                      '        float a = 0.0f;',
                      '        if (sl < nrows) {',
                      '#pragma unroll',
                      f'          for (int j = 0; j < {E // 256}; ++j) {{',
                      '            const long long rr = 256LL * j + t;',
                      f'            a += {leaf};',
                      '          }',
                      '        }',
                      '        pa[r] = a;',
                      '      }',
                      '#pragma unroll',
                      '      for (int r = 0; r < 8; ++r) pa[r] = s101h::warp_sum(pa[r]);',
                      '      if (lane == 0) {',
                      '#pragma unroll',
                      f'        for (int r = 0; r < 8; ++r) p9w_{name}[({8 * h} + r) * 8 + wq] = pa[r];',
                      '      }',
                      '      }']
            if h == 1:
                L.append('      }')
        elems_raw = [parts[name][1] for name, _ in rows if parts[name][1]]
        gpref = []
        if elems_raw and E // 4 == 256 and 'no_consumer' not in DIAG:
            for i in rest:
                if (ident_dtype(i) in ctype and any(f'latch_ld4({i} + 4u * q, ' in x for x in elems_raw)
                        and not any(f'{i}[' in x for x in elems_raw)):
                    gpref.append(i)
        for i in gpref:
            L += [f'      {ctype[ident_dtype(i)]} p9g_{i[2:]}[{nrows_max}][4];   // this thread\'s quad of every row',
                  '#pragma unroll',
                  f'      for (int r = 0; r < {nrows_max}; ++r)',
                  f'        if (r < nrows) latch_ld4({i} + 4u * ((unsigned)(g + {GROUPS} * r) * 256u + (unsigned)t), p9g_{i[2:]}[r]);']
        L += [f'      P9T({tid}, 8);',
              '      __syncthreads();                                       // every warp partial of every row',
              f'      for (int sl = wq; sl < nrows; sl += 8) {{',
              f'        const long long o = (long long)g + {GROUPS}LL * sl;']
        for name, s in rows:
            L += [f'        {{ // row value: {name} (the 8 warp sums of the frozen 256-thread tree)',
                  f'        float second = lane < 8 ? p9w_{name}[sl * 8 + lane] : 0.0f;',
                  '        second = s101h::warp_sum(second);',
                  f'        const float p9_val = {s["store"]};',
                  f'        if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'        if (lane == 0) p9r_{s["out"][2:]}[sl] = p9_val;')
            L.append('        }')
        L.append('        __syncwarp();  // row values are stored by lane 0 before the scalar consumers read them')
        for name, _ in rows:
            if parts[name][0] and 'no_consumer' not in DIAG:
                L.append('        ' + sub(parts[name][0], cons_map))
        L += ['      }',
              f'      P9T({tid}, 6);',
              '      __syncthreads();                                       // row values and scalar consumers']
        elems = [parts[name][1] for name, _ in rows if parts[name][1]]
        if elems and 'no_consumer' not in DIAG:
            texts = []
            for text in elems:
                for ident in staged:
                    text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                        f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
                for ident in gpref:
                    text = re.sub(r'latch_ld4\(' + ident + r' \+ 4u \* q, (\w+)\);',
                                  lambda m: f'p9_cp4({m.group(1)}, p9g_{ident[2:]}[sl]);', text)
                texts.append(sub(text, cons_map))
            if gpref:
                L += ['#pragma unroll',
                      f'      for (int sl = 0; sl < {nrows_max}; ++sl) {{               // element consumers over the CTA',
                      '        if (sl >= nrows) break;']
            else:
                L += [f'#pragma unroll {CONSUMER_UNROLL}',
                      '      for (int sl = 0; sl < nrows; ++sl) {                   // element consumers over the CTA']
            L += [f'        const long long o = (long long)g + {GROUPS}LL * sl;',
                  *('        ' + x for x in texts),
                  '      }']
        L.append('    }')
    L.append(f'    P9T({tid}, 1);')
    if cols and 'no_cols' not in DIAG:
        nacc = 2 * E // 256
        L += ['    p9_wait(p9_b0, p9_parity);',
              '    if (n1) p9_wait(p9_b1, p9_parity);']
        for c, (name, s) in enumerate(cols):
            leaf = sub(s['leaf'], leaf_map('oc'))
            L += [f'    {{ // column partials: {name} (virtual thread t = g + {GROUPS} u sums rows t + 256 j, ascending j)',
                  f'    float p9a[{nacc}];',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) p9a[m] = 0.0f;',
                  f'    for (int j = 0; j < {R // 256}; ++j) {{',
                  '#pragma unroll',
                  f'      for (int m = 0; m < {nacc}; ++m) {{',
                  f'        const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'        const long long oc = idx % {E};',
                  '        const int sl = u + 2 * j;',
                  f'        const long long r = (long long)g + {GROUPS}LL * sl;',
                  f'        p9a[m] += {leaf};',
                  '      }',
                  '    }',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) {{',
                  f'      const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a[m];',
                  '    }',
                  '    }']
    L += [f'    P9T({tid}, 2);',
          '    __syncthreads();                                         // staged rows are reused by the next group',
          '  }',
          f'  P9T({tid}, 3);']
    if cols and 'no_cols' not in DIAG:
        L += ['  grid.sync();                                                 // every partial sum is stored',
              f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            L.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        L.append('  }')
    L += [f'  P9T({tid}, 4);',
          '  __syncthreads();',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b1) : "memory");',
          '  }',
          '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock6',
              'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'prefetched': rest,
                              'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast), 'staged_bytes': total,
                              'register_prefetched': gpref if rows and 'no_rows' not in DIAG else [],
                              'column_reductions': [n for n, _ in cols], 'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    return '\n'.join(L), record



def emit_phase_v5(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t):
    """Engine 5 = engine 3 where every warp issues the bulk copies and loads the row scalars of its own rows, each row
    on its own barrier (no CTA barrier before the rows; a warp starts as soon as its row has landed). Engine 3: all rows of a group staged at once under SMEM_BUDGET_V3 (two half-group barriers: warp w computes its
    first row sl = w while rows 8.. still land), L2 prefetch of row operands that do not fit, row scalars and column
    broadcasts in shared memory, warp-per-row reductions and consumers, then ILP-unrolled column partials. Every value
    and every addition order is that of engines 1/2 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 256 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    if nrows_max > 16:
        return None
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'),
                 'i%' + f'{E}u'}
    consumers = {n: consumer_code(n) for n, _ in rows}
    outs = {s['out'] for _, s in specs}
    counts, scal, bcast = {}, set(), set()
    for kind, text in [('leaf', s['leaf']) for _, s in specs] + [('cons', c) for c in consumers.values()]:
        weight = 4 if kind == 'leaf' else 1
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
    scal = {i for i in scal if i not in written or i in outs}
    bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in counts if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-counts[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    staged, rest = [], []
    for i in candidates:
        size = (nrows_max * E * elsize[ident_dtype(i)] + ALIGN - 1) // ALIGN * ALIGN
        if off + size <= SMEM_BUDGET_V3:
            offsets[('s', i)] = off
            off += size
            staged.append(i)
        else:
            rest.append(i)
    if not any(counts[i] >= 4 for i in staged):
        return None
    total = off
    half = 8
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())

    def sub(text, f):
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = f(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal and form not in ('i', 'i/' + f'{E}u'):
                return f'p9r_{ident[2:]}[sl]'
            if ident in bcast and form in row_bcast and form != 'i%' + f'{E}u':
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)sl * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[sl]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    Q = E // 4
    H = (nrows_max + 7) // 8
    header = re.compile(r'for \(unsigned base = \(unsigned\)o \* ' + str(Q) + r'u \+ \(unsigned\)lane; base < qend; '
                        r'base \+= (\d+)u \* gs\) \{')
    gsdecl = f'const unsigned gs = 32u; const unsigned qend = (unsigned)o * {Q}u + {Q}u;'

    def vector_form(text):
        m = header.search(text)
        if not m or gsdecl not in text or len(header.findall(text)) != 1:
            return None
        U = int(m.group(1))
        return U if Q % (32 * U) == 0 else None

    # Non-staged operands read only as vector consumer loads of the own row: issued into registers at group entry.
    pref = []
    if False:                                                  # engine 5: no register prefetch
        words = 0
        forms = {n: vector_form(c) for n, c in consumers.items() if c}
        for i in rest:
            uses = [n for n, c in consumers.items() if c and f'{i}[' in c]
            vec = [n for n, c in consumers.items() if c and f'latch_ld4({i} + 4u * q, ' in c]
            if uses or not vec or any(forms.get(n) is None for n in vec) or ident_dtype(i) not in ctype:
                continue
            w = H * (Q // 32) * 4 * elsize[ident_dtype(i)] // 4
            if words + w <= REG_PREFETCH_WORDS:
                pref.append(i)
                words += w

    def consumer_text(name):
        text = consumers[name]
        if not text or 'no_consumer' in DIAG:
            return ''
        for ident in staged:
            text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
        if pref and any(f'latch_ld4({i} + 4u * q, ' in text for i in pref):
            U = vector_form(text)
            text = header.sub(f'#pragma unroll\n  for (unsigned p9it = 0; p9it < {Q // (32 * U)}u; ++p9it) {{\n'
                              f'    const unsigned base = (unsigned)o * {Q}u + (unsigned)lane + p9it * {32 * U}u;', text)
            for i in pref:
                text = re.sub(r'latch_ld4\(' + i + r' \+ 4u \* q, (\w+)\);',
                              lambda m: f'p9_cp4({m.group(1)}, p9g_{i[2:]}[p9h][p9it * {U}u + (unsigned)u]);', text)
        return sub(text, cons_map)

    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    L = [f'// P9 staged row-block engine v5: {R} x {E}; staged {", ".join(staged)}'
         + (f'; register-prefetched {", ".join(pref)}' if pref else '')
         + (f'; L2-prefetched {", ".join(r for r in rest if r not in pref)}' if PREFETCH_REST and rest else ''),
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x < 17) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_sa(&p9_rbars[threadIdx.x])) : "memory");',
          '  if (threadIdx.x == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  __syncthreads();',
          f'  P9T({tid}, 0);',
          '  unsigned p9_parity = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x, p9_parity ^= 1u) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          f'    const int n0 = nrows < {half} ? nrows : {half}, n1 = nrows - n0;',
          ]
    nsc = len(scal - outs)
    L += ['    {                                                          // every warp issues its own rows',
          '      const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;',
          '      asm volatile("fence.proxy.async;" ::: "memory");']
    if bcast:
        L += ['      if (threadIdx.x == 0) {',
              f'        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_sa(&p9_rbars[16])), "r"({len(bcast) * E * 4}u) : "memory");']
        for i in sorted(bcast):
            L.append(f'        p9_bulk(p9_sa(p9c_{i[2:]}), {i}, {E * 4}u, p9_sa(&p9_rbars[16]));')
        L.append('      }')
    L += ['#pragma unroll',
          f'      for (int h = 0; h < {H}; ++h) {{',
          '        const int sl = w + 8 * h;',
          '        if (sl < nrows && lane == 0)',
          f'          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_sa(&p9_rbars[sl])), "r"({per_row}u) : "memory");',
          '      }',
          '      __syncwarp();',
          f'      if (lane < {H * len(staged)}) {{',
          f'        const int sl = w + 8 * (lane / {len(staged)}), which = lane % {len(staged)};',
          f'        const long long row = (long long)g + {GROUPS}LL * sl;',
          '        if (sl < nrows) {',
          '          const unsigned bar = p9_sa(&p9_rbars[sl]);']
    for n, i in enumerate(staged):
        L.append(f'          if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)sl * {E}), {i} + row * {E}LL, {row_bytes[i]}u, bar);')
    L += ['        }',
          '      }']
    if nsc:
        L += [f'      if (lane < {H * nsc}) {{                                 // this warp\'s row scalars',
              f'        const int sl = w + 8 * (lane / {nsc}), which = lane % {nsc};',
              f'        const long long row = (long long)g + {GROUPS}LL * sl;',
              '        if (sl < nrows) {']
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'          if (which == {n}) p9r_{i[2:]}[sl] = {i}[row];')
        L += ['        }',
              '      }']
    L += ['      __syncwarp();',
          '    }',
          f'    P9T({tid}, 13);',
          f'    P9T({tid}, 12);']
    if rows and 'no_rows' not in DIAG:
        loop = (['#pragma unroll',
                 f'      for (int p9h = 0; p9h < {H}; ++p9h) {{',
                 '        const int sl = (threadIdx.x >> 5) + 8 * p9h;',
                 '        if (sl >= nrows) break;'] if pref else
                [f'      for (int sl = threadIdx.x >> 5; sl < nrows; sl += blockDim.x >> 5) {{'])
        L += ['    {',
              '      const int lane = threadIdx.x & 31;'] + loop + [
              f'        p9_wait(p9_sa(&p9_rbars[sl]), p9_parity);',
              *([f'        p9_wait(p9_sa(&p9_rbars[16]), p9_parity);'] if bcast else []),
              f'        if (sl == 0) P9T({tid}, 5); else if (sl == 8) P9T({tid}, 7);',
              f'        const long long o = (long long)g + {GROUPS}LL * sl;']
        for name, s in rows:
            leaf = sub(s['leaf'], leaf_map('rr'))
            L += [f'        {{ // row reduction: {name}',
                  '        float v[8];',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = 0.0f;',
                  '#pragma unroll',
                  f'        for (int j = 0; j < {E // 256}; ++j) {{',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) {',
                  '            const long long rr = 256LL * j + 32 * k8 + lane;',
                  f'            v[k8] += {leaf};',
                  '          }',
                  '        }',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = s101h::warp_sum(v[k8]);',
                  '        float second = 0.0f;',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) if (lane == k8) second = v[k8];',
                  '        second = s101h::warp_sum(second);',
                  f'        const float p9_val = {s["store"]};',
                  f'        if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'        if (lane == 0) p9r_{s["out"][2:]}[sl] = p9_val;')
            L += [f'        if (sl == 0) P9T({tid}, 8);',
                  f'        {consumer_text(name)}',
                  '        }']
        L += [f'        if (sl == 0) P9T({tid}, 6);',
              f'        if (sl == 1) P9TW({tid}, 9, 32);',
              f'        if (sl == 7) P9TW({tid}, 10, 224);',
              f'        if (sl == 15) P9TW({tid}, 11, 224);',
              '      }',
              '    }']
    if not (rows and 'no_rows' not in DIAG):
        L += ['    for (int sl = threadIdx.x >> 5; sl < nrows; sl += blockDim.x >> 5) {',
              '      p9_wait(p9_sa(&p9_rbars[sl]), p9_parity);',
              *(['      p9_wait(p9_sa(&p9_rbars[16]), p9_parity);'] if bcast else []),
              '    }']
    L.append(f'    P9T({tid}, 1);')
    if cols and 'no_cols' not in DIAG:
        nacc = 2 * E // 256
        L += ['    __syncthreads();                                         // every row landed (waited by its warp); scalars']
        for c, (name, s) in enumerate(cols):
            leaf = sub(s['leaf'], leaf_map('oc'))
            L += [f'    {{ // column partials: {name} (virtual thread t = g + {GROUPS} u sums rows t + 256 j, ascending j)',
                  f'    float p9a[{nacc}];',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) p9a[m] = 0.0f;',
                  f'    for (int j = 0; j < {R // 256}; ++j) {{',
                  '#pragma unroll',
                  f'      for (int m = 0; m < {nacc}; ++m) {{',
                  f'        const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'        const long long oc = idx % {E};',
                  '        const int sl = u + 2 * j;',
                  f'        const long long r = (long long)g + {GROUPS}LL * sl;',
                  f'        p9a[m] += {leaf};',
                  '      }',
                  '    }',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) {{',
                  f'      const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a[m];',
                  '    }',
                  '    }']
    L += [f'    P9T({tid}, 2);',
          '    __syncthreads();                                         // staged rows are reused by the next group',
          '  }',
          f'  P9T({tid}, 3);']
    if cols and 'no_cols' not in DIAG:
        L += ['  grid.sync();                                                 // every partial sum is stored',
              f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            L.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        L.append('  }')
    L += [f'  P9T({tid}, 4);',
          '  __syncthreads();',
          '  if (threadIdx.x < 17) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_sa(&p9_rbars[threadIdx.x])) : "memory");',
          '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock5',
              'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'prefetched': rest,
                              'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast), 'staged_bytes': total, 'register_prefetched': pref,
                              'column_reductions': [n for n, _ in cols], 'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    return '\n'.join(L), record



def emit_phase_v7(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t):
    """Engine 7 = engine 5 with a producer warp: warp 0 issues every bulk copy in row order (one barrier per row) and
    computes no row; warps 1..7 load their rows' scalars and evaluate rows w - 1 + 7 i as each row lands, so landing
    overlaps compute. Values and orders are those of engine 3 (bitwise). Engine 5 = engine 3 where every warp issues the bulk copies and loads the row scalars of its own rows, each row
    on its own barrier (no CTA barrier before the rows; a warp starts as soon as its row has landed). Engine 3: all rows of a group staged at once under SMEM_BUDGET_V3 (two half-group barriers: warp w computes its
    first row sl = w while rows 8.. still land), L2 prefetch of row operands that do not fit, row scalars and column
    broadcasts in shared memory, warp-per-row reductions and consumers, then ILP-unrolled column partials. Every value
    and every addition order is that of engines 1/2 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 256 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    if nrows_max > 16:
        return None
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'),
                 'i%' + f'{E}u'}
    consumers = {n: consumer_code(n) for n, _ in rows}
    outs = {s['out'] for _, s in specs}
    counts, scal, bcast = {}, set(), set()
    for kind, text in [('leaf', s['leaf']) for _, s in specs] + [('cons', c) for c in consumers.values()]:
        weight = 4 if kind == 'leaf' else 1
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
    scal = {i for i in scal if i not in written or i in outs}
    bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in counts if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-counts[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    staged, rest = [], []
    for i in candidates:
        size = (nrows_max * E * elsize[ident_dtype(i)] + ALIGN - 1) // ALIGN * ALIGN
        if off + size <= SMEM_BUDGET_V3:
            offsets[('s', i)] = off
            off += size
            staged.append(i)
        else:
            rest.append(i)
    if not any(counts[i] >= 4 for i in staged):
        return None
    total = off
    half = 8
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())

    def sub(text, f):
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = f(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9s_{ident[2:]}[(unsigned)sl * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal and form not in ('i', 'i/' + f'{E}u'):
                return f'p9r_{ident[2:]}[sl]'
            if ident in bcast and form in row_bcast and form != 'i%' + f'{E}u':
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)sl * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[sl]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    Q = E // 4
    H = (nrows_max + 7) // 8
    header = re.compile(r'for \(unsigned base = \(unsigned\)o \* ' + str(Q) + r'u \+ \(unsigned\)lane; base < qend; '
                        r'base \+= (\d+)u \* gs\) \{')
    gsdecl = f'const unsigned gs = 32u; const unsigned qend = (unsigned)o * {Q}u + {Q}u;'

    def vector_form(text):
        m = header.search(text)
        if not m or gsdecl not in text or len(header.findall(text)) != 1:
            return None
        U = int(m.group(1))
        return U if Q % (32 * U) == 0 else None

    # Non-staged operands read only as vector consumer loads of the own row: issued into registers at group entry.
    pref = []
    if False:                                                  # engine 5: no register prefetch
        words = 0
        forms = {n: vector_form(c) for n, c in consumers.items() if c}
        for i in rest:
            uses = [n for n, c in consumers.items() if c and f'{i}[' in c]
            vec = [n for n, c in consumers.items() if c and f'latch_ld4({i} + 4u * q, ' in c]
            if uses or not vec or any(forms.get(n) is None for n in vec) or ident_dtype(i) not in ctype:
                continue
            w = H * (Q // 32) * 4 * elsize[ident_dtype(i)] // 4
            if words + w <= REG_PREFETCH_WORDS:
                pref.append(i)
                words += w

    def consumer_text(name):
        text = consumers[name]
        if not text or 'no_consumer' in DIAG:
            return ''
        for ident in staged:
            text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)sl * {E}u), ')
        if pref and any(f'latch_ld4({i} + 4u * q, ' in text for i in pref):
            U = vector_form(text)
            text = header.sub(f'#pragma unroll\n  for (unsigned p9it = 0; p9it < {Q // (32 * U)}u; ++p9it) {{\n'
                              f'    const unsigned base = (unsigned)o * {Q}u + (unsigned)lane + p9it * {32 * U}u;', text)
            for i in pref:
                text = re.sub(r'latch_ld4\(' + i + r' \+ 4u \* q, (\w+)\);',
                              lambda m: f'p9_cp4({m.group(1)}, p9g_{i[2:]}[p9h][p9it * {U}u + (unsigned)u]);', text)
        return sub(text, cons_map)

    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    L = [f'// P9 staged row-block engine v7: {R} x {E}; staged {", ".join(staged)}'
         + (f'; register-prefetched {", ".join(pref)}' if pref else '')
         + (f'; L2-prefetched {", ".join(r for r in rest if r not in pref)}' if PREFETCH_REST and rest else ''),
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x < 17) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_sa(&p9_rbars[threadIdx.x])) : "memory");',
          '  if (threadIdx.x == 0) asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  __syncthreads();',
          f'  P9T({tid}, 0);',
          '  unsigned p9_parity = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x, p9_parity ^= 1u) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          f'    const int n0 = nrows < {half} ? nrows : {half}, n1 = nrows - n0;',
          ]
    nsc = len(scal - outs)
    L += ['    {                                                          // warp 0: producer of every row',
          '      const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;',
          '      if (w == 0) {',
          '        asm volatile("fence.proxy.async;" ::: "memory");']
    if bcast:
        L += ['        if (lane == 0) {',
              f'          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_sa(&p9_rbars[16])), "r"({len(bcast) * E * 4}u) : "memory");']
        for i in sorted(bcast):
            L.append(f'          p9_bulk(p9_sa(p9c_{i[2:]}), {i}, {E * 4}u, p9_sa(&p9_rbars[16]));')
        L.append('        }')
    L += ['        if (lane < nrows)',
          f'          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_sa(&p9_rbars[lane])), "r"({per_row}u) : "memory");',
          '        __syncwarp();',
          f'        for (int k = lane; k < nrows * {len(staged)}; k += 32) {{   // row order',
          f'          const int sl = k / {len(staged)}, which = k % {len(staged)};',
          f'          const long long row = (long long)g + {GROUPS}LL * sl;',
          '          const unsigned bar = p9_sa(&p9_rbars[sl]);']
    for n, i in enumerate(staged):
        L.append(f'          if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)sl * {E}), {i} + row * {E}LL, {row_bytes[i]}u, bar);')
    L += ['        }',
          '      }']
    if nsc:
        L += [f'      else if (lane < {3 * nsc}) {{                            // this compute warp\'s row scalars',
              f'        const int sl = (w - 1) + 7 * (lane / {nsc}), which = lane % {nsc};',
              f'        const long long row = (long long)g + {GROUPS}LL * sl;',
              '        if (sl < nrows) {']
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'          if (which == {n}) p9r_{i[2:]}[sl] = {i}[row];')
        L += ['        }',
              '      }']
    L += ['      __syncwarp();',
          '    }',
          f'    P9T({tid}, 13);',
          f'    P9T({tid}, 12);']
    if rows and 'no_rows' not in DIAG:
        loop = (['#pragma unroll',
                 f'      for (int p9h = 0; p9h < {H}; ++p9h) {{',
                 '        const int sl = (threadIdx.x >> 5) + 8 * p9h;',
                 '        if (sl >= nrows) break;'] if pref else
                [f'      for (int sl = (threadIdx.x >> 5) - 1; sl >= 0 && sl < nrows; sl += 7) {{'])
        L += ['    {',
              '      const int lane = threadIdx.x & 31;'] + loop + [
              f'        p9_wait(p9_sa(&p9_rbars[sl]), p9_parity);',
              *([f'        p9_wait(p9_sa(&p9_rbars[16]), p9_parity);'] if bcast else []),
              f'        if (sl == 0) P9TW({tid}, 5, 32); else if (sl == 7) P9TW({tid}, 7, 32);',
              f'        const long long o = (long long)g + {GROUPS}LL * sl;']
        for name, s in rows:
            leaf = sub(s['leaf'], leaf_map('rr'))
            L += [f'        {{ // row reduction: {name}',
                  '        float v[8];',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = 0.0f;',
                  '#pragma unroll',
                  f'        for (int j = 0; j < {E // 256}; ++j) {{',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) {',
                  '            const long long rr = 256LL * j + 32 * k8 + lane;',
                  f'            v[k8] += {leaf};',
                  '          }',
                  '        }',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) v[k8] = s101h::warp_sum(v[k8]);',
                  '        float second = 0.0f;',
                  '#pragma unroll',
                  '        for (int k8 = 0; k8 < 8; ++k8) if (lane == k8) second = v[k8];',
                  '        second = s101h::warp_sum(second);',
                  f'        const float p9_val = {s["store"]};',
                  f'        if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'        if (lane == 0) p9r_{s["out"][2:]}[sl] = p9_val;')
            L += [f'        if (sl == 0) P9TW({tid}, 8, 32);',
                  f'        {consumer_text(name)}',
                  '        }']
        L += [f'        if (sl == 0) P9TW({tid}, 6, 32);',
              f'        if (sl == 1) P9TW({tid}, 9, 64);',
              f'        if (sl == 6) P9TW({tid}, 10, 224);',
              f'        if (sl == 13) P9TW({tid}, 11, 224);',
              '      }',
              '    }']
    if not (rows and 'no_rows' not in DIAG):
        L += ['    for (int sl = threadIdx.x >> 5; sl < nrows; sl += blockDim.x >> 5) {',
              '      p9_wait(p9_sa(&p9_rbars[sl]), p9_parity);',
              *(['      p9_wait(p9_sa(&p9_rbars[16]), p9_parity);'] if bcast else []),
              '    }']
    L.append(f'    P9T({tid}, 1);')
    if cols and 'no_cols' not in DIAG:
        nacc = 2 * E // 256
        L += ['    __syncthreads();                                         // every row landed (waited by its warp); scalars']
        for c, (name, s) in enumerate(cols):
            leaf = sub(s['leaf'], leaf_map('oc'))
            L += [f'    {{ // column partials: {name} (virtual thread t = g + {GROUPS} u sums rows t + 256 j, ascending j)',
                  f'    float p9a[{nacc}];',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) p9a[m] = 0.0f;',
                  f'    for (int j = 0; j < {R // 256}; ++j) {{',
                  '#pragma unroll',
                  f'      for (int m = 0; m < {nacc}; ++m) {{',
                  f'        const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'        const long long oc = idx % {E};',
                  '        const int sl = u + 2 * j;',
                  f'        const long long r = (long long)g + {GROUPS}LL * sl;',
                  f'        p9a[m] += {leaf};',
                  '      }',
                  '    }',
                  '#pragma unroll',
                  f'    for (int m = 0; m < {nacc}; ++m) {{',
                  f'      const int idx = threadIdx.x + 256 * m, u = idx / {E};',
                  f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a[m];',
                  '    }',
                  '    }']
    L += [f'    P9T({tid}, 2);',
          '    __syncthreads();                                         // staged rows are reused by the next group',
          '  }',
          f'  P9T({tid}, 3);']
    if cols and 'no_cols' not in DIAG:
        L += ['  grid.sync();                                                 // every partial sum is stored',
              f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            L.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        L.append('  }')
    L += [f'  P9T({tid}, 4);',
          '  __syncthreads();',
          '  if (threadIdx.x < 17) asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_sa(&p9_rbars[threadIdx.x])) : "memory");',
          '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock7',
              'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'prefetched': rest,
                              'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast), 'staged_bytes': total, 'register_prefetched': pref,
                              'column_reductions': [n for n, _ in cols], 'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    return '\n'.join(L), record




def emit_phase_v4(heavy, consumer_code, *, ident_dtype, ident_ok, ident_numel, written, elsize, scratch=True,
                  resolve=lambda t: t):
    """Engine 4 (engine 3 with every row operand staged: one pass of all rows when they fit, else two passes of 8 rows,
    single buffer, column partials accumulated across passes in row order). Engine 3: all rows of a group staged at once under SMEM_BUDGET_V3 (two half-group barriers: warp w computes its
    first row sl = w while rows 8.. still land), L2 prefetch of row operands that do not fit, row scalars and column
    broadcasts in shared memory, warp-per-row reductions and consumers, then ILP-unrolled column partials. Every value
    and every addition order is that of engines 1/2 (exact)."""
    specs = [(node.name, (node.extra or {}).get('p9')) for node in heavy]
    if not specs or not all(s for _, s in specs):
        return None
    specs = [(n, dict(s, leaf=resolve(s['leaf']))) for n, s in specs]
    R, E = specs[0][1]['rows'], specs[0][1]['extent']
    if any((s['rows'], s['extent']) != (R, E) for _, s in specs) or E % 256 or R < 256:
        return None
    cols = [(n, s) for n, s in specs if s['kind'] == 'col']
    rows = [(n, s) for n, s in specs if s['kind'] == 'row']
    if cols and (R % 256 or not scratch):
        return None
    nrows_max = (R + GROUPS - 1) // GROUPS
    if nrows_max > 16:
        return None
    row_canon = _norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u')
    col_canon = _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u')
    row_scal = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u / {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u / {E}u'),
                'i/' + f'{E}u', 'i'}
    row_bcast = {_norm(f'(unsigned)rr * 1u + (unsigned)o * {E}u % {E}u'), _norm(f'(unsigned)r * {E}u + (unsigned)oc * 1u % {E}u'),
                 'i%' + f'{E}u'}
    consumers = {n: consumer_code(n) for n, _ in rows}
    outs = {s['out'] for _, s in specs}
    counts, scal, bcast = {}, set(), set()
    for kind, text in [('leaf', s['leaf']) for _, s in specs] + [('cons', c) for c in consumers.values()]:
        weight = 4 if kind == 'leaf' else 1
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            n, form = ident_numel(ident), _norm(index)
            if n is None:
                continue
            if form in (row_canon, col_canon, 'i') and n == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
            elif form in row_scal and n == R:
                scal.add(ident)
            elif form in row_bcast and n == E:
                bcast.add(ident)
        for ident in _VEC.findall(text):
            if ident_numel(ident) == R * E and ident_ok(ident, R, E):
                counts[ident] = counts.get(ident, 0) + weight
    scal = {i for i in scal if i not in written or i in outs}
    bcast = {i for i in bcast if i not in written}
    candidates = sorted((i for i in counts if i not in written and ident_dtype(i) in elsize),
                        key=lambda i: (-counts[i], i))
    ctype = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}
    offsets, off = {}, 0
    for i in sorted(bcast):
        offsets[('c', i)] = off
        off += E * 4
    for i in sorted(scal):
        offsets[('r', i)] = off
        off += nrows_max * 4
    off = (off + ALIGN - 1) // ALIGN * ALIGN
    row_all = sum(E * elsize[ident_dtype(i)] for i in candidates)
    if off + nrows_max * row_all + 128 * len(candidates) <= SMEM_BUDGET_V3:
        prow = nrows_max                                         # one pass of every row
    elif off + 8 * row_all + 128 * len(candidates) <= SMEM_BUDGET_V3:
        prow = 8                                                 # passes of 8 rows (one row per warp)
    else:
        return None
    staged, rest = list(candidates), []
    for i in staged:
        offsets[('s', i)] = off
        off += (prow * E * elsize[ident_dtype(i)] + ALIGN - 1) // ALIGN * ALIGN
    if not any(counts[i] >= 4 for i in staged):
        return None
    total = off
    half = 8
    npass = (nrows_max + prow - 1) // prow
    row_bytes = {i: E * elsize[ident_dtype(i)] for i in staged}
    per_row = sum(row_bytes.values())

    def sub(text, f):
        pieces, last = [], 0
        for start, end, ident, index in _accesses(text):
            if text[end:end + 3] == ' = ':
                continue
            rep = f(ident, _norm(index))
            if rep is not None:
                pieces.append(text[last:start])
                pieces.append(rep)
                last = end
        pieces.append(text[last:])
        return ''.join(pieces)

    def leaf_map(column):
        def f(ident, form):
            if ident in staged and form in (row_canon, col_canon):
                return f'p9s_{ident[2:]}[(unsigned)(sl - p9_r0) * {E}u + (unsigned){column}]'
            if ident in scal and form in row_scal and form not in ('i', 'i/' + f'{E}u'):
                return f'p9r_{ident[2:]}[sl]'
            if ident in bcast and form in row_bcast and form != 'i%' + f'{E}u':
                return f'p9c_{ident[2:]}[{column}]'
            return None
        return f

    def cons_map(ident, form):
        if ident in staged and form == 'i':
            return f'p9s_{ident[2:]}[(i - (unsigned)o * {E}u + (unsigned)(sl - p9_r0) * {E}u)]'
        if ident in scal and form in ('i/' + f'{E}u', 'i'):
            return f'p9r_{ident[2:]}[sl]'
        if ident in bcast and form == 'i%' + f'{E}u':
            return f'p9c_{ident[2:]}[(i % {E}u)]'
        return None

    def consumer_text(name):
        text = consumers[name]
        if not text or 'no_consumer' in DIAG:
            return ''
        for ident in staged:
            text = text.replace(f'latch_ld4({ident} + 4u * q, ',
                                f'latch_ld4(p9s_{ident[2:]} + (4u * q - (unsigned)o * {E}u + (unsigned)(sl - p9_r0) * {E}u), ')
        return sub(text, cons_map)

    tid = _TRACE_IDS[0]
    _TRACE_IDS[0] += 1
    L = [f'// P9 staged row-block engine v3: {R} x {E}; staged {", ".join(staged)}'
         + (f'; L2-prefetched {", ".join(rest)}' if rest else ''),
         '{',
         '  char* p9_smem = reinterpret_cast<char*>(smem);']
    for i in sorted(bcast):
        L.append(f'  float* p9c_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("c", i)]});')
    for i in sorted(scal):
        L.append(f'  float* p9r_{i[2:]} = reinterpret_cast<float*>(p9_smem + {offsets[("r", i)]});')
    for i in staged:
        t = ctype[ident_dtype(i)]
        L.append(f'  {t}* p9s_{i[2:]} = reinterpret_cast<{t}*>(p9_smem + {offsets[("s", i)]});')
    L += ['  const unsigned p9_b0 = p9_sa(&p9_bars[0]), p9_b1 = p9_sa(&p9_bars[1]);',
          '  __syncthreads();                                             // earlier shared-memory users are done',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(p9_b1) : "memory");',
          '    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");',
          '  }',
          '  __syncthreads();',
          f'  P9T({tid}, 0);',
          '  unsigned p9_parity = 0;',
          f'  for (int g = blockIdx.x; g < {GROUPS}; g += gridDim.x) {{',
          f'    const int nrows = ({R} - g + {GROUPS - 1}) / {GROUPS};',
          '    const int npass = (nrows + %(PROW)s - 1) / %(PROW)s;' % {'PROW': prow},
          ]
    if scal - outs:                                            # row scalars of the group by warps 1..
        L.append(f'    for (int x = threadIdx.x - 32; x >= 0 && x < nrows * {len(scal - outs)}; x += blockDim.x - 32) {{')
        L.append('      const int r = x % nrows, which = x / nrows;')
        L.append(f'      const long long row = (long long)g + {GROUPS}LL * r;')
        for n, i in enumerate(sorted(scal - outs)):
            L.append(f'      if (which == {n}) p9r_{i[2:]}[r] = {i}[row];')
        L.append('    }')
    nacc = 2 * E // 256
    for c, (name, s) in enumerate(cols if (cols and 'no_cols' not in DIAG) else []):
        L += [f'    float p9a{c}[{nacc}];',
              '#pragma unroll',
              f'    for (int m = 0; m < {nacc}; ++m) p9a{c}[m] = 0.0f;']
    L += ['    for (int p9_pass = 0; p9_pass < npass; ++p9_pass) {',
          f'      const int p9_r0 = p9_pass * {prow}, p9_nr = nrows - p9_r0 < {prow} ? nrows - p9_r0 : {prow};',
          '      if (threadIdx.x < 32) {',
          '        asm volatile("fence.proxy.async;" ::: "memory");',
          '        if (threadIdx.x == 0) {',
          f'          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(p9_b0), "r"((unsigned)p9_nr * {per_row}u + (p9_pass ? 0u : {len(bcast) * E * 4}u)) : "memory");']
    for i in sorted(bcast):
        L.append(f'          if (p9_pass == 0) p9_bulk(p9_sa(p9c_{i[2:]}), {i}, {E * 4}u, p9_b0);')
    L += ['        }',
          '        __syncwarp();',
          f'        for (int k = threadIdx.x; k < p9_nr * {len(staged)}; k += 32) {{',
          f'          const int r = k / {len(staged)}, which = k % {len(staged)};',
          f'          const long long row = (long long)g + {GROUPS}LL * (p9_r0 + r);']
    for n, i in enumerate(staged):
        L.append(f'          if (which == {n}) p9_bulk(p9_sa(p9s_{i[2:]} + (long long)r * {E}), {i} + row * {E}LL, {row_bytes[i]}u, p9_b0);')
    L += ['        }',
          '      }',
          '      if (p9_pass == 0) __syncthreads();                      // shared row scalars',
          '      p9_wait(p9_b0, p9_parity);',
          '      p9_parity ^= 1u;']
    if rows and 'no_rows' not in DIAG:
        L += ['      {',
              '        const int lane = threadIdx.x & 31;',
              f'        for (int sl = p9_r0 + (threadIdx.x >> 5); sl < p9_r0 + p9_nr; sl += blockDim.x >> 5) {{',
              f'          if (sl == 0) P9T({tid}, 5);',
              f'          const long long o = (long long)g + {GROUPS}LL * sl;']
        for name, s in rows:
            leaf = sub(s['leaf'], leaf_map('rr'))
            L += [f'          {{ // row reduction: {name}',
                  '          float v[8];',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) v[k8] = 0.0f;',
                  '#pragma unroll',
                  f'          for (int j = 0; j < {E // 256}; ++j) {{',
                  '#pragma unroll',
                  '            for (int k8 = 0; k8 < 8; ++k8) {',
                  '              const long long rr = 256LL * j + 32 * k8 + lane;',
                  f'              v[k8] += {leaf};',
                  '            }',
                  '          }',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) v[k8] = s101h::warp_sum(v[k8]);',
                  '          float second = 0.0f;',
                  '#pragma unroll',
                  '          for (int k8 = 0; k8 < 8; ++k8) if (lane == k8) second = v[k8];',
                  '          second = s101h::warp_sum(second);',
                  f'          const float p9_val = {s["store"]};',
                  f'          if (lane == 0) {s["out"]}[o] = p9_val;']
            if s['out'] in scal:
                L.append(f'          if (lane == 0) p9r_{s["out"][2:]}[sl] = p9_val;')
            L += [f'          {consumer_text(name)}',
                  '          }']
        L += [f'          if (sl == 0) P9T({tid}, 6);',
              '        }',
              '      }']
    for c, (name, s) in enumerate(cols if (cols and 'no_cols' not in DIAG) else []):
        leaf = sub(s['leaf'], leaf_map('oc'))
        L += [f'      {{ // column partials: {name} (rows of this pass, ascending j per virtual thread)',
              '#pragma unroll',
              f'      for (int m = 0; m < {nacc}; ++m) {{',
              f'        const int idx = threadIdx.x + 256 * m, u = idx / {E};',
              f'        const long long oc = idx % {E};',
              f'        for (int sl = p9_r0 + ((u - p9_r0) & 1); sl < p9_r0 + p9_nr; sl += 2) {{',
              f'          const long long r = (long long)g + {GROUPS}LL * sl;',
              f'          p9a{c}[m] += {leaf};',
              '        }',
              '      }',
              '      }']
    L += ['      __syncthreads();                                       // the pass buffer is reused',
          '    }']
    for c, (name, s) in enumerate(cols if (cols and 'no_cols' not in DIAG) else []):
        L += ['#pragma unroll',
              f'    for (int m = 0; m < {nacc}; ++m) {{',
              f'      const int idx = threadIdx.x + 256 * m, u = idx / {E};',
              f'      v_p9part[{c * 256 * E}LL + (long long)(g + {GROUPS} * u) * {E}LL + idx % {E}] = p9a{c}[m];',
              '    }']
    L.append(f'    P9T({tid}, 1);')
    L += [f'    P9T({tid}, 2);',
          '    __syncthreads();                                         // staged rows are reused by the next group',
          '  }',
          f'  P9T({tid}, 3);']
    if cols and 'no_cols' not in DIAG:
        L += ['  grid.sync();                                                 // every partial sum is stored',
              f'  for (long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; w < {len(cols) * E}LL; '
              'w += ((long long)gridDim.x * blockDim.x) >> 5) {',
              '    const int lane = threadIdx.x & 31;',
              f'    const long long c = w / {E}, o = w % {E};',
              '    float part = 0.0f;',
              '#pragma unroll',
              '    for (int i = 0; i < 8; ++i) {',
              f'      const float x = s101h::warp_sum(v_p9part[c * {256 * E}LL + (32LL * i + lane) * {E}LL + o]);',
              '      if (lane == i) part = x;',
              '    }',
              '    float a = lane < 8 ? part : 0.0f;',
              '    a = s101h::warp_sum(a);']
        for c, (name, s) in enumerate(cols):
            L.append(f'    if (lane == 0 && c == {c}) {s["out"]}[o] = {s["store"]};')
        L.append('  }')
    L += [f'  P9T({tid}, 4);',
          '  __syncthreads();',
          '  if (threadIdx.x == 0) {',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b0) : "memory");',
          '    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(p9_b1) : "memory");',
          '  }',
          '}']
    record = {'p9_trace_id': tid, 'engine': 'p9_rowblock4', 'p9_rows_per_pass': prow,
              'p9_rowblock': {'rows': R, 'extent': E, 'staged': staged, 'prefetched': rest,
                              'row_scalars': sorted(scal), 'column_broadcasts': sorted(bcast), 'staged_bytes': total,
                              'column_reductions': [n for n, _ in cols], 'row_reductions': [n for n, _ in rows]},
              'shared_bytes': total + 1024}
    return '\n'.join(L), record
