"""P7-E: compact code generation for large flat fused training programs.

The flat fused emitter writes one straight-line block per phase, so a model with L identical layers carries L
inlined copies of every phase body (including every warpgroup-MMA product main loop and epilogue). nvcc time grows
with that code. ``fold`` keeps one copy of each distinct phase body:

* The kernel becomes one uniform phase loop: ``for (pc) { switch (p7e_op[pc]) { case k: <body k> } grid.sync();
  ticks[pc + 1] = stamp(); }``. Phase order, barriers, ticks, launch and every phase's code are unchanged; the loop
  index and switch value are CTA- and grid-uniform (constant-bank loads), so no warpgroup MMA sits in a divergent path
  and no call is introduced (wgmma pipelining is preserved; see P-OPS5-C C7510/C7518 notes).
* Two phases share a body when their text is identical up to (a) kernel-scope pointer declarations (``v_*`` =
  ``arena + C`` or ``args[k]``, with the same declared type and kind), (b) tensor-map table offsets and (c)
  alpha-renaming of identifiers bound inside the phase. Shared bodies read (a) and (b) from a ``__constant__`` table
  at the phase's slot base; all other literals (shapes, strides, tile configurations, slices) stay compile-time.
* Every member is verified by re-expansion: substituting its own bindings into the shared body must reproduce its
  original text token for token (comments aside). Singletons keep their text, with pointer declarations substituted
  by their defining constant expression.

``P7E_MODE = 'off'`` returns the program unchanged (parent bytes).
"""
from __future__ import annotations

import hashlib
import json
import re

P7E_MODE = 'off'               # 'off' | 'fold' (phase bodies) | 'items' (top-level blocks of phases)
P7E_MIN_GROUP = 2              # a body is shared when at least this many phases have it
P7E_TABLE = 'constant'         # slot table storage: 'constant' (__constant__ bank)
P7E_PRESYNC = False            # dispatch first, then the grid barrier at the start of the selected body (the
                               # instruction stream after the barrier is the next body's code, as in straight-line)
P7E_SHARE_ONLY = ()            # if set, only units containing one of these substrings are shared (e.g. 'p5c::phase')
P7E_SEGMENTS = False           # consecutive unshared units form one straight-line case (parent text and barriers)
P7E_NOSHARE_PURE = False       # units made only of fused elementwise loops keep their own body with constant pointers
                               # (latency-bound gathers lose immediate addressing when pointers come from the table)
P7E_NOSHARE_EXCEPT = ()        # ...unless they also contain one of these (e.g. 'p5c::': matrix bodies stay shared)
P7E_NOSHARE = ()               # units whose text contains one of these substrings keep their own body (constant pointers)
P7E_OPAQUE = False             # bodies use per-iteration opaque copies of arena/args: no address derived from them is
                               # hoisted out of the phase loop (keeps per-body register pressure as in straight-line)
P7E_OPAQUE_SREG = False        # also per-iteration opaque threadIdx/blockIdx/blockDim/gridDim .x in bodies (no hoisted
                               # thread-index expressions live across the phase loop)
P7E_OUTLINE = False            # bodies without warpgroup-MMA products become __noinline__ device functions called from the
                               # switch (own register allocation; no wgmma behind any call); wgmma bodies stay inline
P7E_OUTLINE_KEEP = ('p5c::', 'p8ws::')  # substrings marking bodies that must stay entry-resident (inline in the kernel)
P7E_PREFETCH = False           # warm the next step's constant-bank slots (and op/base) before each grid barrier

KERNEL = 'extern "C" __global__ void latch_ops(float** args, float* arena, unsigned long long* ticks) {'
TICK0 = '  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();'
PHASE = re.compile(r'// Phase (\d+): ([^\n]*)\n\{\n(.*?)\n\}\ngrid\.sync\(\);\n'
                   r'if \(ticks && blockIdx\.x == 0 && threadIdx\.x == 0\) ticks\[(\d+)\] = stamp\(\);\n?', re.S)
DECL = re.compile(r'^(?P<ty>(?:float|__nv_bfloat16|int64_t|int|unsigned)\*) (?P<name>v_\w+) = (?P<expr>[^;\n]+);$')
TOKEN = re.compile(r'(?P<comment>//[^\n]*)|(?P<string>"(?:\\.|[^"\\\n])*")|(?P<char>\'(?:\\.|[^\'\\\n])*\')'
                   r'|(?P<ident>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)|(?P<number>\d[\w.]*)|(?P<space>\s+)|(?P<other>.)', re.S)
TMAP = re.compile(r'\(reinterpret_cast<const char\*>\(args\[(\d+)\]\) \+ (\d+)\)')
TYPES = {'float', 'int', 'unsigned', 'uint4', 'uint2', 'uint32_t', 'uint64_t', 'int64_t', 'int32_t', '__nv_bfloat16',
         '__nv_bfloat162', 'bool', 'double', 'auto', 'long', 'short', 'char', 'size_t', 'float4', 'float2',
         'p5c::Args', 'popsc::Slice', 'pops4b::TArgs', 'popsc::BArgs'}
RESERVED = {'smem', 'grid', 'args', 'arena', 'ticks', 'threadIdx', 'blockIdx', 'blockDim', 'gridDim', 'warpSize'}


P7E_OPQ_HELPER = ('template <class T> __device__ __forceinline__ T* p7e_opq(T* p) { asm volatile("" : "+l"(p)); return p; }\n'
                  '__device__ __forceinline__ unsigned p7e_opq_u(unsigned v) { asm volatile("" : "+r"(v)); return v; }\n')
P7E_SREG_HEAD = ('  const unsigned p7e_tx = p7e_opq_u(threadIdx.x), p7e_bx = p7e_opq_u(blockIdx.x), '
                 'p7e_nt = p7e_opq_u(blockDim.x), p7e_nb = p7e_opq_u(gridDim.x);\n')
P7E_OPAQUE_HEAD = '  float* p7e_arena = p7e_opq(arena); float** p7e_args = p7e_opq(args);\n'


class FoldRefusal(ValueError):
    pass


def _tokens(text):
    out = []
    for m in TOKEN.finditer(text):
        kind = m.lastgroup
        out.append((kind, m.group(0)))
    if ''.join(t for _, t in out) != text:
        raise FoldRefusal('tokenizer does not cover the text')
    return out


def _significant(tokens, i, step):
    j = i + step
    while 0 <= j < len(tokens) and tokens[j][0] in ('space', 'comment'):
        j += step
    return tokens[j] if 0 <= j < len(tokens) else (None, None)


def _bound(tokens):
    """Identifiers declared inside the phase (type keyword before, declarator punctuation after)."""
    sig = [i for i, (k, _) in enumerate(tokens) if k not in ('space', 'comment')]
    bound = set()
    for n, i in enumerate(sig):
        kind, text = tokens[i]
        if kind != 'ident' or text in RESERVED or '::' in text or n == 0 or n + 1 >= len(sig):
            continue
        prev = n - 1
        while prev >= 0 and tokens[sig[prev]][1] in ('*', '&'):
            prev -= 1
        if prev >= 0 and tokens[sig[prev]][1] in TYPES and tokens[sig[n + 1]][1] in ('[', '=', ';', ',', ')', '{'):
            bound.add(text)
    return bound


def _is_member(tokens, i):
    """True for a member name after '.' or '->' (never renamed or substituted)."""
    j = i - 1
    while j >= 0 and tokens[j][0] in ('space', 'comment'):
        j -= 1
    if j < 0:
        return False
    if tokens[j][1] == '.':
        return True
    return tokens[j][1] == '>' and j >= 1 and tokens[j - 1][1] == '-'


class Declarations:
    """Kernel-scope pointer declarations: name -> (type, base, byte offset, defining expression)."""

    def __init__(self, lines):
        self.rows = {}
        for line in lines:
            m = DECL.match(line)
            if not m:
                raise FoldRefusal('unrecognized kernel-scope declaration: ' + line[:120])
            self.rows[m['name']] = self._parse(m['ty'], m['expr'])

    @staticmethod
    def _parse(ty, expr):
        elem = {'float*': 4, '__nv_bfloat16*': 2, 'int64_t*': 8, 'int*': 4, 'unsigned*': 4}[ty]
        e = expr.strip()
        m = re.fullmatch(r'(?:reinterpret_cast<(?P<cast>[\w]+\*)>\((?P<inner>[^()]*)\)|(?P<plain>[^()]*))'
                         r'(?: \+ (?P<post>\d+)LL)?', e)
        if not m:
            raise FoldRefusal('unrecognized declaration expression: ' + e)
        inner = m['inner'] if m['inner'] is not None else m['plain']
        if m['cast'] is not None and m['cast'] != ty:
            raise FoldRefusal('declaration cast differs from its type: ' + e)
        if m['post'] is not None and m['cast'] is None:
            raise FoldRefusal('unexpected element offset without cast: ' + e)
        a = re.fullmatch(r'args\[(\d+)\]', inner)
        if a:
            if m['post'] is not None:
                raise FoldRefusal('offset input pointer: ' + e)
            return {'ty': ty, 'base': 'args', 'value': int(a.group(1)), 'expr': e}
        r = re.fullmatch(r'arena \+ (\d+)LL(?: \+ (\d+)LL)?', inner)
        if not r:
            raise FoldRefusal('unrecognized pointer base: ' + e)
        words = int(r.group(1)) + int(r.group(2) or 0)
        offset = 4 * words + elem * int(m['post'] or 0)
        return {'ty': ty, 'base': 'arena', 'value': offset, 'expr': e}


def _canonical(code, decls):
    """Canonical token string, the ordered slot bindings and the bound-identifier order of one phase."""
    tokens = [t for t in _tokens(code) if t[0] != 'comment']
    bound = _bound(tokens)
    slots, slot_index, local = [], {}, {}
    out = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == 'other' and text == '(':
            # Tensor-map table entry: (reinterpret_cast<const char*>(args[N]) + OFF)
            rest = ''.join(t for _, t in tokens[i:i + 24])
            m = TMAP.match(rest)
            if m:
                key = ('map', int(m.group(1)), int(m.group(2)))
                if key not in slot_index:
                    slot_index[key] = len(slots)
                    slots.append(key)
                out.append(f'@M{m.group(1)}:{slot_index[key]}')
                consumed = 0
                while consumed < len(m.group(0)):
                    consumed += len(tokens[i][1])
                    i += 1
                if consumed != len(m.group(0)):
                    raise FoldRefusal('tensor-map reference split across tokens')
                continue
        if kind == 'ident':
            member = _is_member(tokens, i)
            if text in decls.rows and not member:
                row = decls.rows[text]
                key = ('ptr', text)
                if key not in slot_index:
                    slot_index[key] = len(slots)
                    slots.append(key)
                out.append(f'@{row["base"]}:{row["ty"]}:{slot_index[key]}')
                i += 1
                continue
            if text in bound and not member:
                if text not in local:
                    local[text] = len(local)
                out.append(f'@L{local[text]}')
                i += 1
                continue
        out.append(text)
        i += 1
    return ''.join(out), slots, list(local), tokens, bound


def _slot_value(slot, decls):
    if slot[0] == 'map':
        return slot[2]
    return decls.rows[slot[1]]['value']


def _emit_shared(tokens, bound, decls, slots, map_arg):
    """Representative text with pointer/map slots read from the table at the phase's base."""
    index = {s: k for k, s in enumerate(slots)}
    out = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == 'other' and text == '(':
            rest = ''.join(t for _, t in tokens[i:i + 24])
            m = TMAP.match(rest)
            if m:
                k = index[('map', int(m.group(1)), int(m.group(2)))]
                out.append(f'(reinterpret_cast<const char*>(args[{map_arg}]) + p7e_tab[p7e_b + {k}])')
                consumed = 0
                while consumed < len(m.group(0)):
                    consumed += len(tokens[i][1])
                    i += 1
                continue
        if kind == 'ident' and text in decls.rows:
            if not _is_member(tokens, i):
                row = decls.rows[text]
                k = index[('ptr', text)]
                if row['base'] == 'arena':
                    out.append(f'reinterpret_cast<{row["ty"]}>(reinterpret_cast<char*>(arena) + p7e_tab[p7e_b + {k}])')
                else:
                    out.append(f'reinterpret_cast<{row["ty"]}>(args[p7e_tab[p7e_b + {k}]])')
                i += 1
                continue
        out.append(text)
        i += 1
    return ''.join(out)


def _inline(code, decls):
    """Singleton/tail text with each pointer declaration replaced by its defining constant expression."""
    out = []
    tokens = _tokens(code)
    for i, (kind, text) in enumerate(tokens):
        if kind == 'ident' and text in decls.rows and not _is_member(tokens, i):
            out.append('(' + decls.rows[text]['expr'] + ')')
        else:
            out.append(text)
    return ''.join(out)


def _expand(shared_tokens, rep_bound_order, member_bound_order, rep_slots, member_slots, decls):
    """Member text implied by the representative tokens and the member's bindings (for verification)."""
    rename = dict(zip(rep_bound_order, member_bound_order))
    slot_of = dict(zip(rep_slots, member_slots))
    out = []
    i = 0
    tokens = shared_tokens
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == 'other' and text == '(':
            rest = ''.join(t for _, t in tokens[i:i + 24])
            m = TMAP.match(rest)
            if m:
                target = slot_of[('map', int(m.group(1)), int(m.group(2)))]
                out.append(f'(reinterpret_cast<const char*>(args[{target[1]}]) + {target[2]})')
                consumed = 0
                while consumed < len(m.group(0)):
                    consumed += len(tokens[i][1])
                    i += 1
                continue
        if kind == 'ident' and not _is_member(tokens, i):
            if text in decls.rows:
                out.append(slot_of[('ptr', text)][1])
                i += 1
                continue
            if text in rename:
                out.append(rename[text])
                i += 1
                continue
        out.append(text)
        i += 1
    return ''.join(out)


SREGS = {'threadIdx': 'p7e_tx', 'blockIdx': 'p7e_bx', 'blockDim': 'p7e_nt', 'gridDim': 'p7e_nb'}


def _opaque(body, bases=True, sregs=False):
    """Replace kernel-parameter tokens arena/args (and optionally the .x special registers) by per-iteration copies."""
    tokens = _tokens(body)
    out = []
    i = 0
    while i < len(tokens):
        kind, text = tokens[i]
        if kind == 'ident' and not _is_member(tokens, i):
            if bases and text in ('arena', 'args'):
                out.append('p7e_' + text)
                i += 1
                continue
            if (sregs and text in SREGS and i + 2 < len(tokens) and tokens[i + 1][1] == '.'
                    and tokens[i + 2][1] == 'x'):
                out.append(SREGS[text])
                i += 3
                continue
        out.append(text)
        i += 1
    return ''.join(out)


def split_kernel(source):
    start = source.find(KERNEL)
    if start < 0 or source.count('extern "C" __global__') != 1:
        raise FoldRefusal('expected exactly one latch_ops kernel entry')
    body_start = start + len(KERNEL)
    head, rest = source[:start], source[body_start:]
    t0 = rest.find('\n' + TICK0 + '\n')
    if t0 < 0:
        raise FoldRefusal('kernel tick-0 marker not found')
    prologue = rest[:t0].split('\n')
    after = rest[t0 + len(TICK0) + 2:]
    if prologue[0] != '' or prologue[1] != '  extern __shared__ float smem[];' or prologue[2] != '  cg::grid_group grid = cg::this_grid();':
        raise FoldRefusal('unexpected kernel prologue')
    decl_lines = [l for l in prologue[3:] if l.strip()]
    phases, pos = [], 0
    while True:
        m = PHASE.match(after, pos)
        if not m:
            break
        k, names, code, tick = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
        if k != len(phases) or tick != k + 1:
            raise FoldRefusal('phase numbering/ticks are not consecutive')
        phases.append({'index': k, 'names': names, 'code': code})
        pos = m.end()
    if not phases:
        raise FoldRefusal('no phases recognized')
    tail = after[pos:]
    return head, decl_lines, phases, tail


def fold(program, mode='fold'):
    from .lower import GeneratedProgram
    source = program.source
    head, decl_lines, phases, tail = split_kernel(source)
    decls = Declarations(decl_lines)
    # Reconstruction check: the parsed parts reproduce the kernel exactly.
    rebuilt = (head + KERNEL + '\n  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n'
               + '\n'.join(decl_lines) + '\n' + TICK0 + '\n'
               + ''.join(f'// Phase {p["index"]}: {p["names"]}\n{{\n{p["code"]}\n}}\ngrid.sync();\n'
                         f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{p["index"] + 1}] = stamp();\n'
                         for p in phases) + tail)
    if rebuilt != source:
        raise FoldRefusal('kernel parse does not reproduce the source')
    for p in phases + [{'code': tail}]:
        tokens = [t for t in _tokens(p['code']) if t[0] not in ('space', 'comment')]
        for i, (kind, text) in enumerate(tokens[:-1]):
            nxt = tokens[i + 1][1] + (tokens[i + 2][1] if i + 2 < len(tokens) else '')
            if kind == 'ident' and text in decls.rows and (nxt[:1] == '=' and nxt[:2] != '=='
                                                           or nxt[:2] in ('++', '--', '+=', '-=')):
                raise FoldRefusal('kernel-scope pointer reassigned: ' + text)
    map_arg = len(program.manifest['inputs'])
    items = mode == 'items'
    units = []
    for p in phases:
        parts = _split_items(p['code']) if items else [p['code']]
        for j, code in enumerate(parts):
            units.append({'index': len(units), 'phase': p['index'], 'names': p['names'], 'code': code,
                          'last': j == len(parts) - 1})
    groups = {}
    for u in units:
        canon, slots, order, tokens, bound = _canonical(u['code'], decls)
        unshare = any(x in u['code'] for x in P7E_NOSHARE) and not any(x in u['code'] for x in P7E_NOSHARE_EXCEPT)
        unshare = unshare or (P7E_NOSHARE_PURE and _pure_elementwise(u['code']))
        if unshare or (P7E_SHARE_ONLY and not any(x in u['code'] for x in P7E_SHARE_ONLY)):
            canon = f'@unshared{u["index"]}:' + canon
        u.update(canon=canon, slots=slots, order=order, tokens=tokens, bound=bound)
        for slot in slots:
            if slot[0] == 'map' and slot[1] != map_arg:
                raise FoldRefusal('tensor-map table is not args[inputs]')
        groups.setdefault(canon, []).append(u['index'])
    cases, op_of, base_of, tick_of, count_of, table = [], [], [], [], [], []
    case_of_group = {}
    stats = {'phases': len(phases), 'units': len(units), 'groups': 0, 'shared_groups': 0, 'shared_units': 0,
             'singletons': 0}
    for u in units:
        members = groups[u['canon']]
        tick_of.append(u['phase'] + 1 if u['last'] else 0)
        count_of.append(len(u['slots']) if len(members) >= P7E_MIN_GROUP else 0)
        if len(members) >= P7E_MIN_GROUP:
            rep = units[members[0]]
            if u['canon'] not in case_of_group:
                case_of_group[u['canon']] = len(cases)
                body = _emit_shared(rep['tokens'], rep['bound'], decls, rep['slots'], map_arg)
                cases.append({'members': sorted({units[m]['phase'] for m in members}), 'shared': True,
                              'names': rep['names'], 'body': body})
                stats['shared_groups'] += 1
            expanded = _expand(rep['tokens'], rep['order'], u['order'], rep['slots'], u['slots'], decls)
            if expanded != ''.join(t for _, t in u['tokens']):
                raise FoldRefusal(f'unit {u["index"]} (phase {u["phase"]}) does not re-expand from its shared body')
            op_of.append(case_of_group[u['canon']])
            base_of.append(len(table))
            table.extend(_slot_value(slot, decls) for slot in u['slots'])
            stats['shared_units'] += 1
        elif P7E_SEGMENTS and cases and cases[-1].get('segment') and op_of[-1] == len(cases) - 1:
            # Extend the open straight-line segment: the previous unit's barrier (if it ended a phase) stays inline.
            seg = cases[-1]
            prev = tick_of[-2]
            if prev:
                seg['body'] += ('\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) '
                                f'ticks[{prev}] = stamp();\n')
            seg['body'] += '\n// Phase %d: %s\n{\n%s\n}' % (u['phase'], u['names'][:160], _inline(u['code'], decls))
            seg['members'].append(u['phase'])
            tick_of.pop(-2)
            count_of.pop()
            stats['singletons'] += 1
            stats['segment_units'] = stats.get('segment_units', 0) + 1
        else:
            op_of.append(len(cases))
            base_of.append(0)
            cases.append({'members': [u['phase']], 'shared': False, 'names': u['names'], 'segment': P7E_SEGMENTS,
                          'body': ('// Phase %d: %s\n{\n%s\n}' % (u['phase'], u['names'][:160], _inline(u['code'], decls))
                                   if P7E_SEGMENTS else _inline(u['code'], decls))})
            stats['singletons'] += 1
    stats['groups'] = len(cases)
    stats['table_entries'] = len(table)
    if 8 * max(1, len(table)) + 16 * len(units) > 60 * 1024:
        raise FoldRefusal('slot table exceeds the constant-bank budget')
    case_text, outlined = [], []
    for k, c in enumerate(cases):
        label = (f'// Shared body {k}: phases {c["members"]} (first: {c["names"][:160]})' if c['shared']
                 else f'// Phase {c["members"][0]}: {c["names"]}')
        enter = (f'  asm volatile("// p7e body {k}" ::: "memory");\n'
                 '  if (p7e_s) {\n  grid.sync();\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[p7e_s] = stamp();\n  }\n'
                 if P7E_PRESYNC else '')
        body = _opaque(c['body'], P7E_OPAQUE, P7E_OPAQUE_SREG) if (P7E_OPAQUE or P7E_OPAQUE_SREG) else c['body']
        if P7E_OUTLINE and not any(x in c['body'] for x in P7E_OUTLINE_KEEP):
            idents = {t for kind, t in _tokens(c['body']) if kind == 'ident'}
            if idents & {'return', 'grid', 'ticks', 'p7e_pc', 'p7e_s'} or P7E_OPAQUE or P7E_OPAQUE_SREG:
                raise FoldRefusal(f'body {k} cannot be outlined (uses kernel control state)')
            outlined.append(f'// {label[3:]}\n__device__ __noinline__ void p7e_body_{k}(float** args, float* arena, int p7e_b) {{\n'
                            f'  extern __shared__ float smem[];\n{{\n{body}\n}}\n}}\n')
            case_text.append(f'  case {k}: {label}\n{enter}  p7e_body_{k}(args, arena, p7e_b);\n  break;')
            stats['outlined'] = stats.get('outlined', 0) + 1
        else:
            case_text.append(f'  case {k}: {label}\n{enter}{{\n{body}\n}}\n  break;')
    n = len(op_of)
    if len(tick_of) != n or len(count_of) != n or len(base_of) != n:
        raise FoldRefusal('internal: step tables disagree')
    ticked = items or P7E_SEGMENTS
    tables = ('// P7-E fold: per-step body index and slot base; slots hold arena byte offsets, input indices or\n'
              '// tensor-map table byte offsets of the step\'s pointers (read only through uniform constant-bank loads).\n'
              f'__constant__ long long p7e_tab[{max(1, len(table))}] = {{{", ".join(str(v) for v in table) or "0"}}};\n'
              f'__constant__ int p7e_op[{n}] = {{{", ".join(map(str, op_of))}}};\n'
              f'__constant__ int p7e_base[{n}] = {{{", ".join(map(str, base_of))}}};\n')
    if P7E_PREFETCH and P7E_PRESYNC:
        raise FoldRefusal('prefetch and presync layouts are exclusive')
    if P7E_PREFETCH:
        # Each phase end warms the constant-bank entries (body index, slot base and slots) of every step of the next
        # phase while the grid barrier is pending: the step-start chain (op/base -> slot -> address) then hits.
        pf_lo, pf_n = [0] * n, [0] * n
        ends = [i for i in range(n) if tick_of[i] or not ticked]
        for j, e in enumerate(ends):
            nxt = range(e + 1, ends[j + 1] + 1) if j + 1 < len(ends) else range(0)
            shared = [i for i in nxt if count_of[i]]
            if shared:
                pf_lo[e] = min(base_of[i] for i in shared)
                pf_n[e] = max(base_of[i] + count_of[i] for i in shared) - pf_lo[e]
        tables += (f'__constant__ int p7e_pfb[{n}] = {{{", ".join(map(str, pf_lo))}}};\n'
                   f'__constant__ int p7e_pfn[{n}] = {{{", ".join(map(str, pf_n))}}};\n')
        prefetch = ('  if (p7e_pc + 1 < ' + str(n) + ') {\n'
                    '    const int p7e_no = p7e_op[p7e_pc + 1] + p7e_base[p7e_pc + 1]' + (' + p7e_tick[p7e_pc + 1]' if ticked else '') + ';\n'
                    '    asm volatile("" :: "r"(p7e_no));\n'
                    '    const int p7e_nb = p7e_pfb[p7e_pc], p7e_nc = p7e_pfn[p7e_pc];\n'
                    '    for (int p7e_i = 4 * (int)threadIdx.x; p7e_i < p7e_nc; p7e_i += 4 * (int)blockDim.x) {\n'
                    '      const long long p7e_w = p7e_tab[p7e_nb + p7e_i]; asm volatile("" :: "l"(p7e_w));\n    }\n  }\n')
    else:
        prefetch = ''
    if ticked:
        tables += f'__constant__ int p7e_tick[{n}] = {{{", ".join(map(str, tick_of))}}};\n'
        step_end = ('  const int p7e_t = p7e_tick[p7e_pc];\n  if (p7e_t) {\n' + prefetch + '  grid.sync();\n'
                    '  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[p7e_t] = stamp();\n  }\n  }\n')
    else:
        step_end = (prefetch + '  grid.sync();\n'
                    '  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[p7e_pc + 1] = stamp();\n  }\n')
    if P7E_PRESYNC:
        # Barrier (and tick) that precedes step pc: the previous step's phase end, or none.
        sync_of = [0] + [tick_of[i] if ticked else i + 1 for i in range(n - 1)]
        tables += f'__constant__ int p7e_sync[{n}] = {{{", ".join(map(str, sync_of))}}};\n'
        last = tick_of[-1] if ticked else n
        step_end = ('  }\n  grid.sync();\n'
                    f'  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{last}] = stamp();\n')
        loop_head = '  const int p7e_b = p7e_base[p7e_pc], p7e_s = p7e_sync[p7e_pc];\n'
    else:
        loop_head = '  const int p7e_b = p7e_base[p7e_pc];\n'
    kernel = (KERNEL + '\n  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n' + TICK0 + '\n'
              f'#pragma unroll 1\n  for (int p7e_pc = 0; p7e_pc < {n}; ++p7e_pc) {{\n'
              + loop_head + (P7E_OPAQUE_HEAD if P7E_OPAQUE else '') + (P7E_SREG_HEAD if P7E_OPAQUE_SREG else '') + '  switch (p7e_op[p7e_pc]) {\n'
              + '\n'.join(case_text)
              + '\n  default: __trap();\n  }\n' + step_end
              + _inline(tail, decls))
    if P7E_OPAQUE or P7E_OPAQUE_SREG:
        tables += P7E_OPQ_HELPER
    new_source = head + tables + ''.join(outlined) + kernel
    manifest = dict(program.manifest)
    stats.update(mode=mode, prefetch=P7E_PREFETCH, presync=P7E_PRESYNC, opaque=P7E_OPAQUE, opaque_sreg=P7E_OPAQUE_SREG, outline=P7E_OUTLINE, segments=P7E_SEGMENTS, share_only=list(P7E_SHARE_ONLY), steps=len(op_of), noshare=list(P7E_NOSHARE), noshare_except=list(P7E_NOSHARE_EXCEPT), noshare_pure=P7E_NOSHARE_PURE, source_bytes_before=len(source), source_bytes_after=len(new_source),
                 p5c_sites_before=source.count('p5c::phase'), p5c_sites_after=new_source.count('p5c::phase'))
    manifest['p7e'] = stats
    manifest['source_hash'] = hashlib.sha256((new_source + json.dumps(manifest['header_hashes'], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(new_source, manifest, program.kernel_body, program.helpers, program.functions)


PURE_ITEM = re.compile(r'\{ // \d+ elements(?: \(x4 vector\))?: ')


def _pure_elementwise(code):
    """True when every top-level item of the unit is a fused elementwise loop (no matrix, reduction or attention)."""
    try:
        items = _split_items(code)
    except FoldRefusal:
        return False
    return all(PURE_ITEM.match(item) for item in items)


def _split_items(code):
    """Top-level blocks of one phase (each with its leading comments), executed in order by every CTA."""
    items, depth, start = [], 0, None
    tokens = _tokens(code)
    pos = 0
    for kind, text in tokens:
        if kind not in ('space', 'comment'):
            if text == '{':
                if depth == 0:
                    start = pos
                depth += 1
            elif text == '}':
                depth -= 1
                if depth == 0:
                    items.append(code[start:pos + 1])
            elif depth == 0:
                raise FoldRefusal('phase has statements outside top-level blocks')
        pos += len(text)
    if depth or not items:
        raise FoldRefusal('unbalanced phase blocks')
    return items


P7E_STRICT = False             # raise on unrecognized emission instead of keeping the program unchanged


def apply(program):
    if P7E_MODE == 'off':
        return program
    if P7E_MODE not in ('fold', 'items'):
        raise ValueError('unknown P7E_MODE ' + str(P7E_MODE))
    try:
        return fold(program, P7E_MODE)
    except FoldRefusal as refusal:
        # Emission this pass does not recognize (other generators, loops, new declaration forms): keep the program
        # exactly as lowered and record why, so combinations with other emitter changes degrade to the parent path.
        if P7E_STRICT:
            raise
        program.manifest['p7e'] = {'mode': P7E_MODE, 'refused': str(refusal)}
        return program
