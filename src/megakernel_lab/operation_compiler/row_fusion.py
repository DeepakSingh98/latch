"""Row-local fusion of exact last-dimension reductions with their pointwise producers/consumers (P-OPS2-D).

A contiguous FP32 reduction over the last dimension of width W (32 < W <= 256, or W <= 2048 under P6-C wide,
W % 32 == 0, at least 128 rows) is emitted by codegen_rules.exact_reduction as one
warp per row that replays the original CTA butterfly exactly. This pass lets that
same warp also evaluate, for its own row:

* prologue: elementwise phases of the reduction's full shape that produce its
  source (and their same-shape elementwise producers), evaluated per element
  j = 32*w + lane exactly as before;
* the identical reduction tree (leaf 0.0f + x, warp butterflies, second level);
* epilogue: elementwise phases of the row shape (evaluated redundantly by every
  lane, stored by lane 0) and of the full shape (per element), whose reads of
  group values are same-element or row-broadcast.

Every operation keeps its rendered expression, operands and rounding; only
which threads evaluate it and whether an intermediate is stored change. Values
read outside the group (or returned, committed or placed) are still stored.
Group membership follows dependency levels: no member's outside consumer can
move later, and outside inputs must be produced strictly before the group.
"""
from __future__ import annotations

import bisect
import math
import re


def _contiguous(shape):
    result, stride = [], 1
    for d in reversed(shape):
        result.append(stride)
        stride *= d
    return tuple(reversed(result))


def _identity(layout, shape):
    """layout addresses its root element-for-element in this flat order."""
    return (tuple(layout.shape) == tuple(shape) and layout.offset == 0
            and all(d == 1 or s == c for d, s, c in zip(layout.shape, layout.strides, _contiguous(shape))))


def _row_broadcast(layout, full, rows):
    """Reading layout at full-shape flat index o*W + j yields root element o (row-local broadcast)."""
    pad = len(full) - len(layout.shape)
    if pad < 0 or layout.offset != 0:
        return False
    shape = (1,) * pad + tuple(layout.shape)
    strides = (0,) * pad + tuple(layout.strides)
    if shape[-1] not in (1, full[-1]) or (shape[-1] == full[-1] and full[-1] != 1 and strides[-1] != 0):
        return False
    expected = _contiguous(full[:-1])
    for d in range(len(full) - 1):
        if full[d] == 1:
            continue
        if shape[d] != full[d] or strides[d] != expected[d]:
            return False
    return math.prod(full[:-1]) == rows


# P8-B: lane-strided rows (element 32 w + lane of a row of `width`, width % 128 == 0) compute each Philox block once
# per four consecutive elements: lane L computes blocks 32 m + L of the row before the sweep, consumers take their
# word by warp shuffle (ops/random_cuda.py latch_philox_mask_pre). Bit-identical masks. Module setting.
PHILOX_ROW_BLOCKS = True
# P9-B: warp-per-row loops prefetch the warp's next row of every own-row operand into L2 at the top of each row (a
# cache hint only: values unchanged). Off by default (diagnostic until measured).
P9_ROW_PREFETCH = False
# LATCH-MOESUP: warp-row index_select gathers join a row group's prologue (evaluated per element from the table;
# the gathered rows are not stored unless read outside the group). Module setting.
ROW_GATHERS = True
ROW_GATHER_HELPER = """#ifndef LATCH_ROW_GATHER
#define LATCH_ROW_GATHER
template <typename T, typename I>
__device__ __forceinline__ T latch_row_gather(const T* table, const I* index, long long i, long long rows, long long width) {
  const long long r = (long long)index[i / width];
  if (r < 0 || r >= rows) __trap();
  return table[r * width + i % width];
}
// LATCH-MOESUP-ROW: the same element when the caller knows its row (i / width == row): one index load per row.
template <typename T, typename I>
__device__ __forceinline__ T latch_row_gather_rc(const T* table, const I* index, long long row, long long i, long long rows,
                                                 long long width) {
  const long long r = (long long)index[row];
  if (r < 0 || r >= rows) __trap();
  return table[r * width + (i - row * width)];
}
#endif
"""
# LATCH-MOESUP-ROW: in warp-per-row forms (i = o * W + 32 * w + lane, W a multiple of 32) the element's row is o, so
# per-row reads (row gathers' index loads, row-broadcast operands (unsigned)(i) / W % R) use the row directly instead
# of a 64-bit division per element. Same addresses and values; one index/broadcast load per row after CSE.
ROW_INDEX_REUSE = True
# LATCH-MOESUP-ROW: chained row stages. A later row reduction of the same width/rows whose source a group already
# computes in registers (e.g. a norm backward's second row sum) joins the group as a stage: its reduction replays
# the same exact tree on the kept values and its row/full epilogues follow, so the intermediate values are never
# written and read back and a grid barrier disappears. Values and reduction orders are unchanged.
ROW_CHAIN = True
_ROW_I_DECL = re.compile(r'^(\s*)const long long i = (o|on_) \* (\d+)LL \+ 32 \* w \+ lane;\s*$')
_ANY_I_DECL = re.compile(r'\blong long i = ')


_R_DECL = re.compile(r'\bconst ([\w:]+) r_(\w+) = ')
_R_USE = re.compile(r'\br_(\w+)\b')


def _chain_stages(code, chain, phases, body, stored, ident, root, nw, width, reduce_lines, group_loops):
    """LATCH-MOESUP-ROW ROW_CHAIN: append chained row stages to a group's per-row code (inside the row loop, after
    the group's own epilogues). Per-element values a later loop reads are kept in registers (ck_*), assigned where
    they are computed. Each stage replays the group's exact reduction tree on its kept source."""
    loops = [list(group_loops[0]), list(group_loops[1])]   # per-element loop bodies, in order
    stage_parts = []
    for j, (k2, er2, ef2) in enumerate(chain):
        info = phases[k2]['row_reduction']
        out2, s2 = phases[k2]['names'][0], root(info['source'])
        leaf = [f'leaf_[w] = r_{s2};']
        loops.append(leaf)
        full_lines = [line for e in ef2 for line in body(e, 'full')]
        loops.append(full_lines)
        stage_parts.append((out2, info['scale'], leaf, [line for e in er2 for line in body(e, 'rows')], full_lines))
    declared = [{m.group(2): m.group(1) for line in lp for m in _R_DECL.finditer(line)} for lp in loops]
    every = {}
    for d in declared:
        every.update(d)
    loads, stores = [[] for _ in loops], [[] for _ in loops]
    kept = {}
    for c in range(2, len(loops)):
        used = {m.group(1) for line in loops[c] for m in _R_USE.finditer(line)}
        for n in sorted(used - set(declared[c])):
            producers = [q for q in range(c) if n in declared[q]]
            if not producers:
                continue   # a row value (declared outside the per-element loops)
            q = producers[-1]
            kept[n] = declared[q][n]
            if f'ck_{n}[w] = r_{n};' not in stores[q]:
                stores[q].append(f'ck_{n}[w] = r_{n};')
            loads[c].append(f'const {kept[n]} r_{n} = ck_{n}[w];')
    head = f'    const long long i = o * {width}LL + 32 * w + lane;'
    out = []
    for line in code:
        if line == '/*CKDECL*/':
            out += [f'  {t} ck_{n}[{nw}];' for n, t in kept.items()]
        elif line in ('/*CKSTORE0*/', '/*CKSTORE1*/'):
            out += ['    ' + x for x in stores[0 if line == '/*CKSTORE0*/' else 1]]
        else:
            out.append(line)
    for j, (out2, scale2, leaf, row_lines, full_lines) in enumerate(stage_parts):
        c = 2 + 2 * j
        v, sec = f'v_c{j}', f'second_c{j}'
        out += [f'  // chained row stage {j}: {out2}', '#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{', head]
        out += ['    ' + x for x in loads[c] + leaf + stores[c]] + ['  }']
        out += [f'  float {v}[8];', '#pragma unroll'] + [x.replace('v[w]', f'{v}[w]') for x in reduce_lines]
        out += ['#pragma unroll', f'  for (int w = 0; w < 8; ++w) {v}[w] = s101h::warp_sum({v}[w]);',
                f'  float {sec} = 0.0f;', '#pragma unroll',
                '  for (int w = 0; w < 8; ++w) {', f'    const float partial = __shfl_sync(0xffffffffu, {v}[w], 0);',
                f'    if (lane == w) {sec} = partial;', '  }', f'  {sec} = s101h::warp_sum({sec});',
                f'  const float r_{out2} = {sec}{scale2};']
        if out2 in stored:
            out.append(f'  if (lane == 0) {ident(out2)}[o] = r_{out2};')
        out += ['  {', '  const long long i = o;'] + ['  ' + x for x in row_lines]
        if full_lines or loads[c + 1] or stores[c + 1]:
            out += ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{', head]
            out += ['    ' + x for x in loads[c + 1] + full_lines + stores[c + 1]] + ['  }']
    out += ['  }'] * len(stage_parts)
    return out


def _row_index_reuse(code):
    out, width = [], None
    for line in code.split('\n'):
        m = _ROW_I_DECL.match(line)
        if m:
            width = int(m.group(3)) if int(m.group(3)) % 32 == 0 else None
            if width is not None:
                line = line + f' const long long i_row = {m.group(2)};'
            out.append(line)
            continue
        if _ANY_I_DECL.search(line):
            width = None
        if width is not None:
            line = re.sub(rf'latch_row_gather\((v_\w+), (v_\w+), i, (\d+)LL, {width}LL\)',
                          rf'latch_row_gather_rc(\1, \2, i_row, i, \3LL, {width}LL)', line)
            line = line.replace(f'(unsigned)(i) / {width}u', '(unsigned)(i_row)')
        out.append(line)
    return '\n'.join(out)
_PHILOX_CALL = re.compile(r'latch_philox_mask\((\w+), i, (\d+)ULL, ([^)]+)\)')


def _philox_rows(code, width):
    """Rewrite per-element Philox mask calls of one lane-strided row group (list of code lines)."""
    text = '\n'.join(code)
    stride = f'o * {width}LL + 32 * w + lane;'
    if (not PHILOX_ROW_BLOCKS or width % 128 or 'latch_philox_mask(' not in text or stride not in text
            or any('const long long i = ' in line and stride not in line and 'const long long i = o;' not in line
                   for line in code)):
        return code
    states = []
    for match in _PHILOX_CALL.finditer(text):
        if match.group(1) not in states:
            states.append(match.group(1))
    if not states or any(f'latch_philox_mask({s}, ' in text.replace(f'latch_philox_mask({s}, i, ', '') for s in states):
        return code
    blocks = width // 128
    pre = []
    for k, state in enumerate(states):
        pre += [f'  uint32_t pw{k}_[{blocks}][4];', '#pragma unroll',
                f'  for (int m_ = 0; m_ < {blocks}; ++m_) latch_philox_block({state}, '
                f'((unsigned long long)(o * {width}LL) >> 2) + 32ULL * m_ + (unsigned long long)lane, pw{k}_[m_]);']
    out = []
    for line in code:
        for k, state in enumerate(states):
            line = re.sub(r'latch_philox_mask\(' + re.escape(state) + r', i, (\d+)ULL, ([^)]+)\)',
                          lambda m: (f'latch_philox_mask_pre(pw{k}_[w >> 2], 8 * (w & 3) + (lane >> 2), lane & 3, '
                                     f'{m.group(1)}ULL, {m.group(2)})'), line)
        out.append(line)
        if line.strip() == 'const int lane = threadIdx.x & 31;':
            out += pre
    return out


def _warps_per_row(rows, nw, blocks=128, cap=8):
    """Warps sharing a row so one round keeps every block's eight warps busy (P-OPS3-C rule); divides nw."""
    per_block = -(-rows // blocks)
    group = 1
    while group < 8 and group < per_block:
        group *= 2
    r = min(cap, 8 // group)
    while r > 1 and nw % r:
        r //= 2
    return r


_READ = re.compile(r'\bv_\w+\[[^\]]*\]')
# LATCH-MOESUP-ROW rpipe: a warp-row gather of the element (ROW_GATHER_HELPER) is a read like any other.
_GATHER_READ = re.compile(r'latch_row_gather\((v_\w+), v_\w+, i, \d+LL, \d+LL\)')
_CT = {'float32': 'float', 'bfloat16': '__nv_bfloat16'}


def _hoist(groups_of_lines, elements, slot, types, budget=96, gathers=False):
    """P6-C rhoist: external reads of a row group's per-element lines, issued for all of a thread's elements
    before any of them computes or stores. A row group writes only its own produced values (single assignment,
    no in-place roots), so every remaining memory read is of a value produced before the group and cannot be
    affected by the group's stores; only the compiler's aliasing assumption serialized them per element.
    Returns (declarations, load statements, rewritten line groups) or None when over the register budget."""
    reads = []
    for lines in groups_of_lines:
        for line in lines:
            head = line.split('; ', 1)[0]
            found = _READ.findall(head)
            if gathers:
                found += [m.group(0) for m in _GATHER_READ.finditer(head)]
            for text in found:
                if text not in reads:
                    reads.append(text)
    if not reads or elements * len(reads) > budget:
        return None
    ctypes = []
    for text in reads:
        g = _GATHER_READ.match(text)
        ctype = types.get(g.group(1) if g else text.split('[', 1)[0])
        if ctype is None:
            return None
        ctypes.append(ctype)
    decls = [f'{c} h{j}_[{elements}];' for j, c in enumerate(ctypes)]
    loads = [f'h{j}_[{slot}] = {text};' for j, text in enumerate(reads)]
    order = sorted(range(len(reads)), key=lambda j: len(reads[j]), reverse=True)

    def rewrite(line):
        head, sep, tail = line.partition('; ')
        for j in order:
            head = head.replace(reads[j], f'h{j}_[{slot}]')
        return head + sep + tail
    return decls, loads, [[rewrite(line) for line in lines] for lines in groups_of_lines]


_STORE_TAIL = re.compile(r'^(v_\w+)\[i\] = (r_\w+);$')
_DECL = re.compile(r'^const (\w+) (r_\w+) = ')


def _sink(groups_of_lines, elements, slot):
    """P6-C rsink: a row group's plain per-element stores (``v_x[i] = r_x;``) are deferred until after its
    element loop: the stored values stay in registers and are written by a separate loop over the same elements.
    The group never reads its own stored values from memory (substituted by registers), so only the store issue
    point moves; with no store between them, every element's loads may be issued before any compute completes.
    Returns (declarations, rewritten line groups, flush statements per group) or None."""
    types, out_groups, flushes, decls, count = {}, [], [], [], 0
    for lines in groups_of_lines:
        new_lines, flush = [], []
        for line in lines:
            head, sep, tail = line.partition('; ')
            m = _DECL.match(head)
            if m:
                types[m.group(2)] = m.group(1)
            st = _STORE_TAIL.match(tail.strip()) if sep else None
            if st and st.group(2) in types:
                name = f'sk{count}_'
                count += 1
                decls.append(f'{types[st.group(2)]} {name}[{elements}];')
                flush.append(f'{st.group(1)}[i] = {name}[{slot}];')
                line = f'{head}; {name}[{slot}] = {st.group(2)};'
            new_lines.append(line)
        out_groups.append(new_lines)
        flushes.append(flush)
    if not count:
        return None
    return decls, out_groups, flushes


def fuse_rows(phases, layouts, dtypes, *, ident, index, kept, inplace_roots, flat=False, multi_warp=False,
              absorb_early=False, hoist=False, sink=False, clamp=False, pipe=False, orders=None):
    """Return (phases, records, elided_roots). phases keep their dict format.

    flat (P-OPS5-B group rules): members may have any shape with the reduction's
    element count when every read of a group value is the identity in the member's
    own shape (so the warp's flat index addresses the same element), and full-shape
    epilogue members may produce BF16 values (e.g. the operand cast of a normalized
    row). Prologue members stay FP32 (their values may be carried in FP32 registers).
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
    names_by_root, layout_position = {}, {}
    for position, name in enumerate(layouts):
        names_by_root.setdefault(root(name), []).append(name)
        layout_position[name] = position
    # LATCH-BUILD-600: the epilogue scans below visit phases in ascending order and skip every phase that reads no
    # value the group has produced so far. ``readers_after`` yields exactly the phases that read a produced value
    # (static ``readers``, extended as members are added), in the same ascending order, so the skipped phases are
    # never visited and every visited phase sees the group state the full scan would (additions come only from
    # earlier positions). The checks applied to each visited phase are unchanged.
    import heapq

    class readers_after:
        def __init__(self, start, values):
            self.start, self.heap, self.queued = start, [], set()
            self.add(values, start)

        def add(self, values, after):
            for y in values:
                for e in readers.get(y, ()):
                    if e > after and e not in self.queued:
                        self.queued.add(e)
                        heapq.heappush(self.heap, e)

        def __iter__(self):
            while self.heap:
                yield heapq.heappop(self.heap)
    row_reduction_phases = [c for c, p_ in enumerate(phases) if p_.get('row_reduction')]
    program_types = None

    def elementwise(k, shape, bf16=False, prologue=False):
        p = phases[k]
        if p['kind'] != 'elementwise' or any(root(n) in inplace_roots for n in p['names']):
            return False
        if flat:
            # LATCH-MOESUP (ROW_GATHERS): a prologue phase may also store BF16 values (e.g. the operand cast of a
            # scaled gathered row); values carried into the epilogue stay FP32 (checked when the group forms).
            allowed = ('float32', 'bfloat16') if bf16 or (prologue and ROW_GATHERS) else ('float32',)
            return (math.prod(p['shape']) == math.prod(shape)
                    and all(dtypes.get(n) in allowed for n in p['names']))
        return (tuple(p['shape']) == tuple(shape)
                and p.get('dtype', 'float32') == 'float32'
                and all(dtypes.get(n) == 'float32' for n in p['names']))

    def own_shape(m, ctx_shape):
        """Index shape a member's rendered lines use: its own under flat rules."""
        return tuple(phases[m]['shape']) if flat and phases[m]['kind'] == 'elementwise' else tuple(ctx_shape)

    gather_lines = {}

    def gather(k):
        """LATCH-MOESUP (ROW_GATHERS): a dim-0 row gather of a contiguous FP32 table (extra_ops warp-row
        index_select) evaluated per element at the group's flat index: table[index[i / W]][i % W], the stored
        form's value and range trap. The caller requires the group to read it in its own flat order."""
        p = phases[k]
        if (not ROW_GATHERS or not flat or p['kind'] != 'index_select' or p.get('engine') != 'index_select_warp_rows'
                or len(p['names']) != 1 or 'gather_table' not in p):
            return False
        g, table, idx = p['names'][0], p['gather_table'], p['gather_index']
        lg, lt, li = layouts.get(g), layouts.get(table), layouts.get(idx)
        if (lg is None or lt is None or li is None or root(g) in inplace_roots or dtypes.get(g) != 'float32'
                or dtypes.get(table) != 'float32' or dtypes.get(idx) != 'int64' or len(lt.shape) < 2
                or not _identity(lg, tuple(lg.shape)) or not _identity(lt, tuple(lt.shape))
                or not _identity(li, tuple(li.shape)) or len(li.shape) != 1 or li.shape[0] != lg.shape[0]):
            return False
        width = math.prod(lt.shape[1:])
        gather_lines[k] = [f'const float r_{g} = latch_row_gather({ident(table)}, {ident(idx)}, i, {lt.shape[0]}LL, '
                           f'{width}LL); {ident(g)}[i] = r_{g};']
        return True

    consumed, records, groups = set(), [], []
    chains = {}   # LATCH-MOESUP-ROW ROW_CHAIN: group index -> [(reduction phase, row epilogues, full epilogues)]
    for k, phase in enumerate(phases):
        info = phase.get('row_reduction')
        if not info or k in consumed:
            continue
        out = phase['names'][0]
        source, width, rows = info['source'], info['width'], info['rows']
        full = tuple(layouts[source].shape)
        rshape = tuple(layouts[out].shape)
        # ---- prologue closure: same-shape elementwise producers of the source
        prologue = []
        frontier = [(source, full)]
        while frontier:
            value, reader_shape = frontier.pop()
            q = writer.get(root(value))
            if (q is None or q in prologue or q in consumed or q == k
                    or not (elementwise(q, full, prologue=True) or gather(q))
                    or not _identity(layouts[value], reader_shape)):
                continue
            prologue.append(q)
            if q not in gather_lines:   # a gather's table is read indirectly: never a same-element producer
                frontier.extend((r, own_shape(q, full)) for r in phases[q]['reads'])
        # Drop prologue members whose outside consumers the group would delay.
        while True:
            members = set(prologue) | {k}
            produced = {root(w) for m in members for w in phases[m]['writes']}
            external = [writer[root(r)] for m in members for r in phases[m]['reads']
                        if root(r) in writer and root(r) not in produced]
            group_level = 1 + max((level[q] for q in external), default=-1)
            late = [q for q in prologue if level[q] < group_level and not (
                flat and absorb_early and all(r in members for w in phases[q]['writes'] for r in readers.get(root(w), ())))]
            if not late:
                break
            prologue = [q for q in prologue if q not in late]
        prologue.sort()
        members = set(prologue) | {k}
        produced = {root(w) for m in members for w in phases[m]['writes']}
        external = [writer[root(r)] for m in members for r in phases[m]['reads']
                    if root(r) in writer and root(r) not in produced]
        group_level = 1 + max((level[q] for q in external), default=-1)

        epilogue_rows, epilogue_full = [], []

        def substitutable(m, ctx):
            mine = own_shape(m, full)
            for r in phases[m]['reads']:
                if root(r) not in produced:
                    continue
                y = root(r)
                lay = layouts[r]
                if ctx == 'full':
                    if writer[y] in prologue or writer[y] in epilogue_full:
                        if not _identity(lay, mine):
                            return False
                    elif mine != full:
                        return False
                    elif not _row_broadcast(lay, full, rows) and not (rshape == full and _identity(lay, full)):
                        return False
                else:
                    if writer[y] != k and writer[y] not in epilogue_rows:
                        return False
                    if not _identity(lay, rshape):
                        return False
            return True

        if not all(substitutable(m, 'full') for m in prologue):
            continue
        # ---- epilogue: later elementwise phases of row or full shape fed only by the group/earlier work
        # ROW_GATHERS: with a gathered prologue row, earlier-placed consumers of group values (e.g. the scaled cast
        # of the gathered row, split from the reduction's producer chain) also join the epilogue: every group
        # input precedes the gather's level, so nothing they produce can feed the group (no cycle), and nothing
        # they read is produced later (checked below). Phases a member reads are never taken.
        gathered = any(q in gather_lines for q in prologue)
        member_reads = {root(r) for m in members for r in phases[m]['reads']}
        candidates = readers_after(min(members) if gathered else k, produced)
        for e in candidates:
            if e in consumed or e in members:
                continue
            if gathered and e < k and any(root(w) in member_reads for w in phases[e]['writes']):
                continue
            ctx = 'rows' if elementwise(e, rshape) else 'full' if elementwise(e, full, bf16=True) else None
            if ctx == 'rows' and flat and tuple(phases[e]['shape']) != rshape:
                ctx = None
            if ctx is None:
                continue
            own = {root(w) for w in phases[e]['writes']}
            reads = [root(r) for r in phases[e]['reads'] if root(r) not in own]
            if not any(r in produced for r in reads):
                continue
            if any(r in writer and r not in produced and level[writer[r]] >= group_level for r in reads):
                continue
            # Row-shaped values never read full-shape group values (substitutable
            # rejects that), so they are emitted before the full-shape epilogue.
            if not substitutable(e, ctx):
                continue
            (epilogue_rows if ctx == 'rows' else epilogue_full).append(e)
            members.add(e)
            added = {root(w) for w in phases[e]['writes']}
            produced |= added
            candidates.add(added, e)
        if not prologue and not epilogue_rows and not epilogue_full:
            continue
        # Values carried from the prologue into the full epilogue are kept in FP32 registers.
        epilogue_reads = {root(r) for e in epilogue_full for r in phases[e]['reads']}
        if any(dtypes.get(n) != 'float32' for q in prologue for n in phases[q]['names'] if n in epilogue_reads):
            continue
        chain = []
        nw_ = width // 32
        rpw_ = (_warps_per_row(rows, 8 if nw_ > 8 else nw_, cap=int(multi_warp)) if multi_warp else 1)
        if ROW_CHAIN and rpw_ == 1 and width % 32 == 0:
            full_set, row_set = set(prologue) | set(epilogue_full), {k} | set(epilogue_rows)

            def chain_substitutable(m, ctx):
                mine = own_shape(m, full)
                for r in phases[m]['reads']:
                    if root(r) not in produced:
                        continue
                    y, lay = root(r), layouts[r]
                    if ctx == 'full':
                        if writer[y] in full_set:
                            if not _identity(lay, mine):
                                return False
                        elif mine != full:
                            return False
                        elif not _row_broadcast(lay, full, rows) and not (rshape == full and _identity(lay, full)):
                            return False
                    elif writer[y] not in row_set or not _identity(lay, rshape):
                        return False
                return True

            while True:
                k2 = None
                for c in row_reduction_phases[bisect.bisect_right(row_reduction_phases, k):]:
                    i2 = phases[c].get('row_reduction')
                    if c in consumed or c in members or not i2 or i2['width'] != width or i2['rows'] != rows:
                        continue
                    s2, o2 = i2['source'], phases[c]['names'][0]
                    if (root(s2) in produced and writer.get(root(s2)) in full_set and dtypes.get(s2) == 'float32'
                            and tuple(layouts[s2].shape) == full and _identity(layouts[s2], full)
                            and tuple(layouts[o2].shape) == rshape and dtypes.get(o2) == 'float32'
                            and len(phases[c]['names']) == 1
                            and all(root(r) == root(s2) for r in phases[c]['reads'])
                            and not any(root(n) in inplace_roots for n in phases[c]['names'])):
                        k2 = c
                        break
                if k2 is None:
                    break
                members.add(k2)
                produced |= {root(w) for w in phases[k2]['writes']}
                row_set.add(k2)
                er2, ef2 = [], []
                candidates = readers_after(k2, produced)
                for e in candidates:
                    if e in consumed or e in members:
                        continue
                    ctx = 'rows' if elementwise(e, rshape) else 'full' if elementwise(e, full, bf16=True) else None
                    if ctx == 'rows' and flat and tuple(phases[e]['shape']) != rshape:
                        ctx = None
                    if ctx is None:
                        continue
                    own = {root(w) for w in phases[e]['writes']}
                    reads = [root(r) for r in phases[e]['reads'] if root(r) not in own]
                    if not any(r in produced for r in reads):
                        continue
                    if any(r in writer and r not in produced and level[writer[r]] >= group_level for r in reads):
                        continue
                    if not chain_substitutable(e, ctx):
                        continue
                    (er2 if ctx == 'rows' else ef2).append(e)
                    (row_set if ctx == 'rows' else full_set).add(e)
                    members.add(e)
                    added = {root(w) for w in phases[e]['writes']}
                    produced |= added
                    candidates.add(added, e)
                chain.append((k2, er2, ef2))
        if chain:
            chains[len(groups)] = chain
        consumed |= members
        groups.append((k, prologue, epilogue_rows, epilogue_full, produced))

    # Softmax rows (P-OPS2-D): the exact warp-per-row softmax may evaluate its
    # input's same-shape elementwise producers per element before its trees.
    softmax_groups = []
    for k, phase in enumerate(phases):
        info = phase.get('row_softmax')
        if not info or k in consumed:
            continue
        source, width, rows = info['source'], info['width'], info['rows']
        full = tuple(layouts[source].shape)
        q = writer.get(root(source))
        if q is None or q in consumed or not elementwise(q, full) or not _identity(layouts[source], full):
            continue
        external = [writer[root(r)] for r in phases[q]['reads'] if root(r) in writer
                    and root(r) not in {root(w) for w in phases[q]['writes']}]
        if 1 + max((level[x] for x in external), default=-1) > level[q]:
            continue
        if any(root(r) in {root(w) for w in phases[q]['writes']} and not _identity(layouts[r], full)
               for r in phases[q]['reads']):
            continue
        consumed |= {k, q}
        softmax_groups.append((k, q))
    if not groups and not softmax_groups:
        return phases, [], set()
    member_of = {m: g for g, (k, pro, er, ef, _) in enumerate(groups) for m in (k, *pro, *er, *ef)}
    member_of.update({m: g for g, chain in chains.items() for (k2, er2, ef2) in chain for m in (k2, *er2, *ef2)})
    elided = set()
    fused = {}
    for offset, (k, q) in enumerate(softmax_groups):
        g = len(groups) + offset
        member_of[k] = member_of[q] = g
        info = phases[k]['row_softmax']
        source, width = info['source'], info['width']
        full = tuple(layouts[source].shape)
        produced = {root(w) for w in phases[q]['writes']}
        stored = {y for y in produced if y in kept or any(r not in (k, q) for r in readers.get(y, ()))}
        elided |= produced - stored
        lines = []
        for line in phases[q]['body']:
            for n in sorted([n for n in layouts if root(n) in produced], key=len, reverse=True):
                try:
                    text = f"{ident(n)}[{index(layouts[n], full, 'i')}]"
                except ValueError:
                    continue
                line = line.replace(text, f'r_{root(n)}')
            match = re.match(r'const (\w+) r_(\w+) = ', line)
            if match and match.group(2) not in stored:
                line = re.sub(r'; ' + re.escape(ident(match.group(2))) + r'\[i\] = [^;]*;$', ';', line)
            lines.append(line)
        load = (f'  for (int w = 0; w < {width // 32}; ++w) {{ const long long k = 32 * w + lane; '
                f'const long long i = row * {width}LL + k; ' + ' '.join(lines) + f' xv[w] = r_{root(source)}; }}')
        code = phases[k]['code']
        assert info['load_line'] in code
        names = [*phases[q]['names'], *phases[k]['names']]
        fused[g] = {**phases[k], 'code': code.replace(info['load_line'], load, 1), 'names': names,
                    'targets': [*phases[q]['targets'], *phases[k]['targets']],
                    'reads': list(dict.fromkeys(r for m in (q, k) for r in phases[m]['reads'] if root(r) not in produced)),
                    'writes': names, 'row_fusion': {'softmax': phases[k]['names'][0], 'prologue_names': phases[q]['names'],
                                                    'stored': sorted(stored), 'elided': sorted(produced - stored)}}
        fused[g].pop('row_softmax', None)
        records.append(fused[g]['row_fusion'])
    for g, (k, prologue, epilogue_rows, epilogue_full, produced) in enumerate(groups):
        members = {k, *prologue, *epilogue_rows, *epilogue_full}
        chain = chains.get(g, [])
        members |= {m for (k2, er2, ef2) in chain for m in (k2, *er2, *ef2)}
        info = phases[k]['row_reduction']
        out = phases[k]['names'][0]
        source, width, rows, scale = info['source'], info['width'], info['rows'], info['scale']
        full = tuple(layouts[source].shape)
        rshape = tuple(layouts[out].shape)
        nw = width // 32
        outside = lambda y: (y in kept or any(member_of.get(q) != g for q in readers.get(y, ())))
        stored = {y for y in produced if outside(y)}
        elided |= produced - stored

        # LATCH-E2TTT-OUTER: the produced names, longest first then layout order (exactly the former per-line
        # `sorted([n for n in layouts if root(n) in produced], key=len, reverse=True)`), computed once per group
        # from a root index instead of rescanning every program layout for every body line.
        produced_names = sorted((n for y in produced for n in names_by_root.get(y, ())),
                                key=lambda n: (-len(n), layout_position[n]))

        def substitute(line, ctx, m=None):
            # split_components may materialize a crossing register as v_x[i],
            # whereas the ordinary index renderer spells the identity v_x[(i)].
            # Both are the same element. Rewrite reads only: the trailing store
            # still decides whether an externally observed value materializes.
            head, separator, tail = line.partition('; ')
            for n in produced_names:
                shape = (own_shape(m, full) if m is not None else full) if ctx == 'full' else rshape
                try:
                    text = f"{ident(n)}[{index(layouts[n], shape, 'i')}]"
                except ValueError:
                    continue  # this view cannot be read in this context
                register = f"r_{root(n)}"
                head = head.replace(text, register)
                if _identity(layouts[n], shape):
                    head = head.replace(f"{ident(n)}[i]", register)
            return head + separator + tail

        def body(m, ctx):
            lines = []
            for line in gather_lines.get(m) or phases[m]['body']:
                line = substitute(line, ctx, m)
                match = re.match(r'const (\w+) r_(\w+) = ', line)
                name = match.group(2) if match else None
                if name and name not in stored:
                    line = re.sub(r'; ' + re.escape(ident(name)) + r'\[i\] = [^;]*;$', ';', line)
                elif name and ctx == 'rows':
                    line = line.replace(f'; {ident(name)}[i] =', f'; if (lane == 0) {ident(name)}[i] =', 1)
                lines.append(line)
            return lines

        pro_values = [n for q in prologue for n in phases[q]['names']]
        later_reads = {root(r) for e in (*epilogue_full,) for r in phases[e]['reads']}
        carried = [n for n in pro_values if n in later_reads]
        source_register = root(source) in produced and writer.get(root(source)) in prologue
        wide = nw > 8   # P6-C: W > 256 replays the CTA fallback's folded leaves (slot w & 7, passes w >> 3)
        rpw = (_warps_per_row(rows, 8 if wide else nw, cap=int(multi_warp)) if multi_warp else 1)
        # LATCH-MOESUP-ROW gather order: when a gathered table row is read through an index whose occurrences an
        # indexed engine already sorted (keys row << 32 | ordinal, every ordinal once), the warps visit the group's
        # rows in that order, so rows gathering the same table row run close together (L2 reuse of the table).
        # Rows are independent: each still computes, reduces and stores exactly its own values.
        order_keys = None
        if orders and rpw == 1:
            for q in prologue:
                if q not in gather_lines:
                    continue
                gi = phases[q].get('gather_index')
                entry = orders.get(root(gi)) if gi in layouts else None
                if (entry and entry[1] == rows and len(layouts[gi].shape) == 1 and layouts[gi].shape[0] == rows
                        and _identity(layouts[gi], tuple(layouts[gi].shape))):
                    order_keys = entry[0]
                    break
        ordinal = lambda pv: f'(long long)((unsigned long long){ident(order_keys)}[{pv}] & 0xffffffffULL)'
        code = [f'for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o < {rows}LL; '
                f'o += ((long long)gridDim.x * blockDim.x) >> 5) {{',
                '  const int lane = threadIdx.x & 31;',
                f'  float leaf_[{nw}];']
        code += [f'  float keep_{n}[{nw}];' for n in carried]
        if chain:
            code.append('/*CKDECL*/')
        if program_types is None:   # LATCH-BUILD-600: one program-wide map (read only), not one per group
            program_types = {ident(n): _CT[dtypes[n]] for n in layouts if dtypes.get(n) in _CT}
        types = program_types
        pro_lines = [line for q in prologue for line in body(q, 'full')]
        epi_lines = [line for e in epilogue_full for line in body(e, 'full')]
        # A load of an internally produced value cannot be issued before the
        # group, even when that value also has a materialized outside consumer.
        # Fail at lowering rather than emit an undefined pointer or stale load.
        internal_reads = {ident(n) for y in produced for n in names_by_root.get(y, ())}
        for line in pro_lines + epi_lines:
            if any(read.split('[', 1)[0] in internal_reads
                   for read in _READ.findall(line.split('; ', 1)[0])):
                from .lower import LoweringError
                raise LoweringError("row fusion left an internal value as a global read: " + line)
        hoisted = _hoist([pro_lines, epi_lines], nw, 'w', types, gathers=pipe) if hoist else None
        piped = None
        if hoisted:
            decls, loads, (pro_lines, epi_lines) = hoisted
            code += ['  ' + d for d in decls]
            if pipe and rpw == 1 and 'latch_philox_mask(' not in '\n'.join(pro_lines + epi_lines):
                # LATCH-MOESUP-ROW rpipe: the hoisted reads of the warp's NEXT row are issued (into p*_) before this
                # row computes, reduces and stores. The group stores only values it produces and never reads them
                # back from memory, so no read of any row can observe a store of the group; values are unchanged.
                stride = '(((long long)gridDim.x * blockDim.x) >> 5)'
                renamed = [re.sub(r'^h(\d+)_\[', r'p\1_[', x) for x in loads]
                piped = ([re.sub(r'\bh(\d+)_\[', r'p\1_[', d) for d in decls], renamed)
                pipe_block = ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{']
                pipe_block += [f'    h{j}_[w] = p{j}_[w];' for j in range(len(loads))] + ['  }']
                pipe_block += (['  {', f'  const long long pn_ = p_ + {stride};',
                                 f'  const long long on_ = pn_ < {rows}LL ? {ordinal("pn_")} : 0LL;', f'  if (pn_ < {rows}LL) {{']
                               if order_keys else ['  {', f'  const long long on_ = o + {stride};', f'  if (on_ < {rows}LL) {{'])
                pipe_block += [
                               '#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                               f'    const long long i = on_ * {width}LL + 32 * w + lane;'] + ['    ' + x for x in renamed]
                pipe_block += ['  }', '  }', '  }']
                plain_block = (['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                                f'    const long long i = o * {width}LL + 32 * w + lane;'] + ['    ' + x for x in loads] + ['  }'])
                piped = piped + (pipe_block, plain_block)
                code.append('/*RPIPE*/')
            else:
                code += ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                         f'    const long long i = o * {width}LL + 32 * w + lane;'] + ['    ' + x for x in loads] + ['  }']
        sunk = _sink([pro_lines, epi_lines], nw, 'w') if sink else None
        if P9_ROW_PREFETCH:
            own = sorted(set(re.findall(r'\b(v_\w+)\[\(i\)\]',
                                        '\n'.join(pro_lines + epi_lines + (list(loads) if hoisted else [])))))
            sizes = {'float': 4, '__nv_bfloat16': 2}
            pf = []
            for name_ in own:
                size_ = sizes.get(types.get(name_), 4)
                lines_ = (int(width) * size_ + 127) // 128
                pf.append(f'    if (lane < {lines_}) asm volatile("prefetch.global.L2 [%0];" :: '
                          f'"l"(reinterpret_cast<const char*>({name_} + on_ * {width}LL) + 128 * lane));')
            if pf:
                code += ['  {', '    const long long on_ = o + (((long long)gridDim.x * blockDim.x) >> 5);',
                         f'    if (on_ < {rows}LL) {{'] + ['  ' + x for x in pf] + ['    }', '  }']
        flush_loop = lambda stmts: (['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                                     f'    const long long i = o * {width}LL + 32 * w + lane;']
                                    + ['    ' + x for x in stmts] + ['  }']) if stmts else []
        pro_flush = epi_flush = []
        if sunk:
            sdecls, (pro_lines, epi_lines), (pro_flush, epi_flush) = sunk
            code += ['  ' + d for d in sdecls]
        code += ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                 f'    const long long i = o * {width}LL + 32 * w + lane;']
        code += ['    ' + line for line in pro_lines]
        code += [f'    keep_{n}[w] = r_{n};' for n in carried]
        if chain:
            code.append('/*CKSTORE0*/')
        leaf = f'r_{root(source)}' if source_register else f'{ident(source)}[i]'
        if wide:
            folds = ' '.join(f'if (w + {8 * p} < {nw}) v[w] = v[w] + leaf_[w + {8 * p} < {nw} ? w + {8 * p} : 0];'
                             for p in range(1, -(-nw // 8)))
            reduce_lines = [f'  for (int w = 0; w < 8; ++w) {{ v[w] = 0.0f + leaf_[w]; {folds} }}']
        else:
            reduce_lines = [f'  for (int w = 0; w < 8; ++w) v[w] = w < {nw} ? 0.0f + leaf_[w < {nw} ? w : 0] : 0.0f;']
        code += [f'    leaf_[w] = {leaf};', '  }', *flush_loop(pro_flush),
                 '  float v[8];', '#pragma unroll', *reduce_lines,
                 '#pragma unroll', '  for (int w = 0; w < 8; ++w) v[w] = s101h::warp_sum(v[w]);',
                 '  float second = 0.0f;', '#pragma unroll',
                 '  for (int w = 0; w < 8; ++w) {', '    const float partial = __shfl_sync(0xffffffffu, v[w], 0);',
                 '    if (lane == w) second = partial;', '  }', '  second = s101h::warp_sum(second);',
                 f'  const float r_{out} = second{scale};']
        if out in stored:
            code.append(f'  if (lane == 0) {ident(out)}[o] = r_{out};')
        if epilogue_rows:
            code.append('  {')
            code.append('  const long long i = o;')
            code += ['  ' + line for e in epilogue_rows for line in body(e, 'rows')]
        if epilogue_full:
            code += ['#pragma unroll', f'  for (int w = 0; w < {nw}; ++w) {{',
                     f'    const long long i = o * {width}LL + 32 * w + lane;']
            code += [f'    const float r_{n} = keep_{n}[w];' for n in carried]
            code += ['    ' + line for line in epi_lines]
            if chain:
                code.append('/*CKSTORE1*/')
            code.append('  }')
            code += flush_loop(epi_flush)
        if chain:
            code = _chain_stages(code, chain, phases, body, stored, ident, root, nw, width, reduce_lines,
                                 [pro_lines + [f'keep_{n}[w] = r_{n};' for n in carried],
                                  [f'const float r_{n} = keep_{n}[w];' for n in carried] + epi_lines if epilogue_full else []])
        if epilogue_rows:
            code.append('  }')
        code.append('}')
        code_plain = None
        if piped:
            pdecls, ploads, pipe_block, plain_block = piped
            at = code.index('/*RPIPE*/')
            code_plain = '\n'.join(_philox_rows(code[:at] + plain_block + code[at + 1:], width))
            code = code[:at] + pipe_block + code[at + 1:]
            start = '(((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5)'
            stride = '(((long long)gridDim.x * blockDim.x) >> 5)'
            if order_keys:
                head = (['{', '  const int lane = threadIdx.x & 31;'] + ['  ' + d for d in pdecls]
                        + [f'  long long p_ = {start};', f'  long long o = p_ < {rows}LL ? {ordinal("p_")} : 0LL;',
                           f'  if (p_ < {rows}LL) {{', '#pragma unroll',
                           f'  for (int w = 0; w < {nw}; ++w) {{', f'    const long long i = o * {width}LL + 32 * w + lane;']
                        + ['    ' + x for x in ploads]
                        + ['  }', '  }', f'for (; p_ < {rows}LL; p_ += {stride}) {{', f'  o = {ordinal("p_")};'])
            else:
                head = (['{', '  const int lane = threadIdx.x & 31;'] + ['  ' + d for d in pdecls]
                        + [f'  long long o = {start};', f'  if (o < {rows}LL) {{', '#pragma unroll',
                           f'  for (int w = 0; w < {nw}; ++w) {{', f'    const long long i = o * {width}LL + 32 * w + lane;']
                        + ['    ' + x for x in ploads] + ['  }', '  }', f'for (; o < {rows}LL; o += {stride}) {{'])
            assert code[0].startswith('for (long long o = ')
            code = head + code[1:] + ['}']
        elif order_keys:
            assert code[0].startswith('for (long long o = ')
            code[0] = (f'for (long long p_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5); p_ < {rows}LL; '
                       f'p_ += (((long long)gridDim.x * blockDim.x) >> 5)) {{ const long long o = {ordinal("p_")};')
        code = _philox_rows(code, width)
        ordered = sorted(members)
        names = [n for m in ordered for n in phases[m]['names']]
        reads = list(dict.fromkeys(r for m in ordered for r in phases[m]['reads'] if root(r) not in produced))
        if order_keys and order_keys not in reads:
            reads.append(order_keys)
        fused[g] = {'kind': 'row_fused', 'names': names,
                    'targets': [t for m in ordered for t in phases[m]['targets']],
                    'reads': reads, 'writes': names, 'code': '\n'.join(code),
                    'work_items': (rows + 7) // 8, 'reduction_elements': width,
                    'row_fusion': {'reduction': out, 'width': width, 'rows': rows,
                                   'prologue_phases': len(prologue), 'row_epilogue_phases': len(epilogue_rows),
                                   'full_epilogue_phases': len(epilogue_full),
                                   'gather_prologue': [phases[q]['names'][0] for q in prologue if q in gather_lines],
                                   'stored': sorted(stored), 'elided': sorted(produced - stored),
                                   'arithmetic': 'same per-operation expressions; reduction replays the exact CTA tree'}}
        fused[g]['code_r1'] = code_plain or fused[g]['code']   # unchanged form for P-OPS5-B row epilogues
        # Structured pieces for cluster-distributed rows (P-OPS5-B): per-element prologue, leaf, statistic and
        # row/full epilogue lines, exactly as rendered above.
        if wide:
            fused[g]['row_fusion']['folded_passes'] = -(-nw // 8)
        if hoisted:
            fused[g]['row_fusion']['hoisted_reads'] = len(hoisted[1])
        if piped:
            fused[g]['row_fusion']['next_row_pipelined'] = True
        if order_keys:
            fused[g]['row_fusion']['gather_order'] = order_keys
        if sunk:
            fused[g]['row_fusion']['sunk_stores'] = len(sunk[0])
        fused[g]['row_parts'] = {
            'prologue': [line for q in prologue for line in body(q, 'full')], 'carried': list(carried),
            'leaf': leaf, 'out': out, 'scale': scale, 'out_stored': out in stored,
            'rows': [line for e in epilogue_rows for line in body(e, 'rows')],
            'full': [line for e in epilogue_full for line in body(e, 'full')],
            'width': width, 'rows_count': rows, 'nw': nw}
        if rpw > 1:
            multi = (_multi_warp_wide if wide else _multi_warp_group)(
                phases, layouts, k, prologue, epilogue_rows, epilogue_full, produced,
                stored, carried, source_register, body, out, source, width, rows, scale,
                nw, rpw, ident, root, hoist=types if hoist else None, sink=sink,
                **({'clamp': clamp} if wide else {}))
            multi['code_r1'] = fused[g]['code']
            multi['row_parts'] = fused[g]['row_parts']
            fused[g] = multi
        if chain:
            # LATCH-MOESUP-ROW ROW_CHAIN: the structured one-stage forms (GEMM row epilogues, cluster rows) do not
            # describe a chained group.
            fused[g].pop('code_r1', None)
            fused[g].pop('row_parts', None)
            fused[g]['row_fusion']['chained_stages'] = [phases[k2]['names'][0] for (k2, _, _) in chain]
        if ROW_INDEX_REUSE:
            for key in ('code', 'code_r1'):
                if isinstance(fused[g].get(key), str):
                    fused[g][key] = _row_index_reuse(fused[g][key])
        records.append(fused[g]['row_fusion'])
    # Replace members by their fused phase at the first member position, then
    # restore a dependency-respecting order (stable topological sort).
    result, placed = [], set()
    for k, phase in enumerate(phases):
        g = member_of.get(k)
        if g is None:
            result.append(phase)
        elif g not in placed:
            placed.add(g)
            result.append(fused[g])
    produced_by = {}
    for k, phase in enumerate(result):
        for w in phase['writes']:
            produced_by[root(w)] = k
    deps = [{produced_by[root(r)] for r in phase['reads'] if root(r) in produced_by and produced_by[root(r)] != k}
            for k, phase in enumerate(result)]
    # LATCH-BUILD-600: the same order as repeatedly taking the lowest-index phase whose dependencies are placed
    # (formerly a rescan from phase 0 per placement, O(phases^2)): a min-heap of ready phases.
    import heapq
    pending = [len(d) for d in deps]
    dependents = {}
    for k, d in enumerate(deps):
        for q in d:
            dependents.setdefault(q, []).append(k)
    ready = [k for k, c in enumerate(pending) if not c]
    heapq.heapify(ready)
    order = []
    while ready:
        k = heapq.heappop(ready)
        order.append(k)
        for q in dependents.get(k, ()):
            pending[q] -= 1
            if not pending[q]:
                heapq.heappush(ready, q)
    if len(order) < len(result):
        raise ValueError('row fusion produced a dependency cycle')
    return [result[k] for k in order], records, elided



def _multi_warp_group(phases, layouts, k, prologue, epilogue_rows, epilogue_full, produced, stored, carried,
                      source_register, body, out, source, width, rows, scale, nw, rpw, ident, root, hoist=None,
                      sink=False):
    """P-OPS5-B: ``rpw`` warps share a row (row engine of P-OPS3-C, adapted to row fusion groups).

    Warp k of a row owns frozen threads 32 w + lane for w in [k nw/rpw, (k+1) nw/rpw): it evaluates those
    elements' prologue members, the frozen per-w butterflies (0.0f + leaf, xor warp_sum) and writes the w
    partials to shared memory; after one block barrier every warp of the row forms the identical second
    stage (lane < nw ? partial[lane] : 0, warp_sum). Row statistic, epilogue values and stores are therefore
    bitwise identical to the one-warp engine; only which warp evaluates which element changes.
    """
    nwr, slots = nw // rpw, 8 // rpw
    code = ['{',
            '  const int lane = threadIdx.x & 31;',
            f'  const int wk_ = (threadIdx.x >> 5) % {rpw}, slot_ = (threadIdx.x >> 5) / {rpw}, w0_ = wk_ * {nwr};',
            '  unsigned buf_ = 0u;',
            f'  for (long long ob_ = (long long)blockIdx.x * {slots}; ob_ < {rows}LL; ob_ += (long long)gridDim.x * {slots}) {{',
            '    const long long o = ob_ + slot_;',
            f'    const bool live_ = o < {rows}LL;',
            f'    float leaf_[{nwr}];']
    code += [f'    float keep_{n}[{nwr}];' for n in carried]
    pro_lines = [line for q in prologue for line in body(q, 'full')]
    epi_lines = [line for e in epilogue_full for line in body(e, 'full')]
    hoisted = _hoist([pro_lines, epi_lines], nwr, 'v_', hoist) if hoist is not None else None
    if hoisted:
        decls, loads, (pro_lines, epi_lines) = hoisted
        code += ['    ' + d for d in decls]
        code += ['    if (live_) {', '#pragma unroll', f'    for (int v_ = 0; v_ < {nwr}; ++v_) {{',
                 '      const int w = w0_ + v_;', f'      const long long i = o * {width}LL + 32 * w + lane;']
        code += ['      ' + x for x in loads] + ['    }', '    }']
    sunk = _sink([pro_lines, epi_lines], nwr, 'v_') if sink else None
    flush_loop = lambda stmts: (['    if (live_) {', '#pragma unroll', f'    for (int v_ = 0; v_ < {nwr}; ++v_) {{',
                                 '      const int w = w0_ + v_;', f'      const long long i = o * {width}LL + 32 * w + lane;']
                                + ['      ' + x for x in stmts] + ['    }', '    }']) if stmts else []
    pro_flush = epi_flush = []
    if sunk:
        sdecls, (pro_lines, epi_lines), (pro_flush, epi_flush) = sunk
        code += ['    ' + d for d in sdecls]
    code += ['    if (live_) {', '#pragma unroll', f'    for (int v_ = 0; v_ < {nwr}; ++v_) {{',
             '      const int w = w0_ + v_;', f'      const long long i = o * {width}LL + 32 * w + lane;']
    code += ['      ' + line for line in pro_lines]
    code += [f'      keep_{n}[v_] = r_{n};' for n in carried]
    leaf = f'r_{root(source)}' if source_register else f'{ident(source)}[i]'
    code += [f'      leaf_[v_] = {leaf};', '    }', '    }', *flush_loop(pro_flush),
             '    float* part_ = smem + buf_ * 64u + slot_ * 8;',
             '    if (live_) {', '#pragma unroll', f'      for (int v_ = 0; v_ < {nwr}; ++v_) {{',
             '        const float s_ = s101h::warp_sum(0.0f + leaf_[v_]);',
             '        if (lane == 0) part_[w0_ + v_] = s_;', '      }', '    }',
             '    __syncthreads();',
             '    if (live_) {',
             f'    float second = lane < {nw} ? part_[lane] : 0.0f;',
             '    second = s101h::warp_sum(second);',
             f'    const float r_{out} = second{scale};']
    if out in stored:
        code.append(f'    if (lane == 0 && wk_ == 0) {ident(out)}[o] = r_{out};')
    if epilogue_rows:
        code += ['    {', '    const long long i = o;']
        code += ['    ' + line.replace('if (lane == 0)', 'if (lane == 0 && wk_ == 0)')
                 for e in epilogue_rows for line in body(e, 'rows')]
    if epilogue_full:
        code += ['#pragma unroll', f'    for (int v_ = 0; v_ < {nwr}; ++v_) {{', '      const int w = w0_ + v_;',
                 f'      const long long i = o * {width}LL + 32 * w + lane;']
        code += [f'      const float r_{n} = keep_{n}[v_];' for n in carried]
        code += ['      ' + line for line in epi_lines]
        code.append('    }')
        if epi_flush:
            code += ['#pragma unroll', f'    for (int v_ = 0; v_ < {nwr}; ++v_) {{', '      const int w = w0_ + v_;',
                     f'      const long long i = o * {width}LL + 32 * w + lane;'] + ['      ' + x for x in epi_flush] + ['    }']
    if epilogue_rows:
        code.append('    }')
    code += ['    }', '    buf_ ^= 1u;', '  }', '}']
    members = {k, *prologue, *epilogue_rows, *epilogue_full}
    ordered = sorted(members)
    names = [n for m in ordered for n in phases[m]['names']]
    reads = list(dict.fromkeys(r for m in ordered for r in phases[m]['reads'] if root(r) not in produced))
    return {'kind': 'row_fused', 'names': names,
            'targets': [t for m in ordered for t in phases[m]['targets']],
            'reads': reads, 'writes': names, 'code': '\n'.join(code),
            'work_items': -(-rows // slots), 'reduction_elements': width,
            'row_fusion': {'reduction': out, 'width': width, 'rows': rows, 'warps_per_row': rpw,
                           'prologue_phases': len(prologue), 'row_epilogue_phases': len(epilogue_rows),
                           'full_epilogue_phases': len(epilogue_full),
                           'stored': sorted(stored), 'elided': sorted(produced - stored),
                           **({'hoisted_reads': len(hoisted[1])} if hoisted else {}),
                           **({'sunk_stores': len(sunk[0])} if sunk else {}),
                           'arithmetic': 'same per-operation expressions; reduction replays the exact CTA tree'}}


def _multi_warp_wide(phases, layouts, k, prologue, epilogue_rows, epilogue_full, produced, stored, carried,
                     source_register, body, out, source, width, rows, scale, nw, rpw, ident, root, hoist=None,
                     sink=False, clamp=False):
    """P6-C: multi-warp row engine for W > 256 (W % 32 == 0), bitwise equal to the 256-thread CTA fallback.

    The CTA fallback gives thread t = 32 s + lane (slot s < 8) the leaf acc = 0.0f + x[t] + x[t + 256] + ...
    (ascending passes). Here warp k of a row owns slots [k 8/rpw, (k+1) 8/rpw) with all their passes, forms
    the identical folded leaves and per-slot butterflies, writes the 8 slot partials to shared memory and,
    after one block barrier, every warp of the row forms the identical second stage.
    """
    sl, passes, slots = 8 // rpw, -(-nw // 8), 8 // rpw
    n = sl * passes
    code = ['{',
            '  const int lane = threadIdx.x & 31;',
            f'  const int wk_ = (threadIdx.x >> 5) % {rpw}, slot_ = (threadIdx.x >> 5) / {rpw}, s0_ = wk_ * {sl};',
            '  unsigned buf_ = 0u;',
            f'  for (long long ob_ = (long long)blockIdx.x * {slots}; ob_ < {rows}LL; ob_ += (long long)gridDim.x * {slots}) {{',
            '    const long long o = ob_ + slot_;',
            f'    const bool live_ = o < {rows}LL;',
            f'    float leaf_[{n}];']
    code += [f'    float keep_{x}[{n}];' for x in carried]
    head = ['#pragma unroll', f'    for (int v_ = 0; v_ < {sl}; ++v_) {{', '#pragma unroll',
            f'    for (int p_ = 0; p_ < {passes}; ++p_) {{',
            f'      const int w = s0_ + v_ + 8 * p_;', f'      if (w < {nw}) {{',
            f'      const long long i = o * {width}LL + 32 * w + lane;']
    pro_lines = [line for q in prologue for line in body(q, 'full')]
    epi_lines = [line for e in epilogue_full for line in body(e, 'full')]
    hoisted = _hoist([pro_lines, epi_lines], n, f'v_ * {passes} + p_', hoist) if hoist is not None else None
    if hoisted:
        decls, loads, (pro_lines, epi_lines) = hoisted
        code += ['    ' + d for d in decls]
        code += ['    if (live_) {', *head] + ['      ' + x for x in loads] + ['      }', '    }', '    }', '    }']
    sunk = _sink([pro_lines, epi_lines], n, f'v_ * {passes} + p_') if sink else None
    pro_flush = epi_flush = []
    if sunk:
        sdecls, (pro_lines, epi_lines), (pro_flush, epi_flush) = sunk
        code += ['    ' + d for d in sdecls]
    compute_head = head
    if sunk and clamp and not any(re.search(r'\bv_\w+\[[^\]]*\] [+\-*/]?= ', line.partition('; ')[2])
                                  for line in (*pro_lines, *epi_lines)):
        # P6-C rclamp: with every element store sunk, absent elements (w >= nw) compute discarded values at a
        # clamped, valid index instead of branching, so all elements' loads form one straight-line block.
        compute_head = ['#pragma unroll', f'    for (int v_ = 0; v_ < {sl}; ++v_) {{', '#pragma unroll',
                        f'    for (int p_ = 0; p_ < {passes}; ++p_) {{',
                        f'      const int w = s0_ + v_ + 8 * p_ < {nw} ? s0_ + v_ + 8 * p_ : {nw - 1};', '      {',
                        f'      const long long i = o * {width}LL + 32 * w + lane;']
    code += ['    if (live_) {', *compute_head]
    code += ['      ' + line for line in pro_lines]
    code += [f'      keep_{x}[v_ * {passes} + p_] = r_{x};' for x in carried]
    leaf = f'r_{root(source)}' if source_register else f'{ident(source)}[i]'
    folds = ' '.join(f'if (s0_ + v_ + {8 * p} < {nw}) s_ = s_ + leaf_[v_ * {passes} + {p}];' for p in range(1, passes))
    code += [f'      leaf_[v_ * {passes} + p_] = {leaf};', '      }', '    }', '    }', '    }']
    if pro_flush:
        code += ['    if (live_) {', *head] + ['      ' + x for x in pro_flush] + ['      }', '    }', '    }', '    }']
    code += [
             '    float* part_ = smem + buf_ * 64u + slot_ * 8;',
             '    if (live_) {', '#pragma unroll', f'      for (int v_ = 0; v_ < {sl}; ++v_) {{',
             f'        float s_ = 0.0f + leaf_[v_ * {passes}]; {folds}',
             '        s_ = s101h::warp_sum(s_);',
             '        if (lane == 0) part_[s0_ + v_] = s_;', '      }', '    }',
             '    __syncthreads();',
             '    if (live_) {',
             '    float second = lane < 8 ? part_[lane] : 0.0f;',
             '    second = s101h::warp_sum(second);',
             f'    const float r_{out} = second{scale};']
    if out in stored:
        code.append(f'    if (lane == 0 && wk_ == 0) {ident(out)}[o] = r_{out};')
    if epilogue_rows:
        code += ['    {', '    const long long i = o;']
        code += ['    ' + line.replace('if (lane == 0)', 'if (lane == 0 && wk_ == 0)')
                 for e in epilogue_rows for line in body(e, 'rows')]
    if epilogue_full:
        code += compute_head
        code += [f'      const float r_{x} = keep_{x}[v_ * {passes} + p_];' for x in carried]
        code += ['      ' + line for line in epi_lines]
        code += ['      }', '    }', '    }']
        if epi_flush:
            code += head + ['      ' + x for x in epi_flush] + ['      }', '    }', '    }']
    if epilogue_rows:
        code.append('    }')
    code += ['    }', '    buf_ ^= 1u;', '  }', '}']
    members = {k, *prologue, *epilogue_rows, *epilogue_full}
    ordered = sorted(members)
    names = [m for q in ordered for m in phases[q]['names']]
    reads = list(dict.fromkeys(r for q in ordered for r in phases[q]['reads'] if root(r) not in produced))
    return {'kind': 'row_fused', 'names': names,
            'targets': [t for q in ordered for t in phases[q]['targets']],
            'reads': reads, 'writes': names, 'code': '\n'.join(code),
            'work_items': -(-rows // slots), 'reduction_elements': width,
            'row_fusion': {'reduction': out, 'width': width, 'rows': rows, 'warps_per_row': rpw,
                           'folded_passes': passes, **({'hoisted_reads': len(hoisted[1])} if hoisted else {}),
                           **({'sunk_stores': len(sunk[0])} if sunk else {}),
                           'prologue_phases': len(prologue), 'row_epilogue_phases': len(epilogue_rows),
                           'full_epilogue_phases': len(epilogue_full),
                           'stored': sorted(stored), 'elided': sorted(produced - stored),
                           'arithmetic': 'same per-operation expressions; reduction replays the exact 256-thread CTA tree (folded leaves)'}}
