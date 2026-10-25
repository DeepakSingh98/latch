"""STEP17 interval partition: whole publication intervals as separately compiled device code.

Large compose programs (TinyMoE: 723 publication intervals, one ~485K-instruction PTX entry)
spend hours in ptxas on the single cooperative entry function. Compose lowering emits the kernel
body as a sequence of top-level brace blocks, each closing its interval with ``grid.sync()`` and
its tick stanza, and each re-deriving every pointer and hardware index it uses from ``args`` /
``arena`` / thread registers. Kernel scope itself declares only ``smem`` and ``grid``.

This pass moves each such block, byte for byte, into ``static __device__ __noinline__ void
latch_s17_block_N(float** args, float* arena, unsigned long long* ticks)`` that redeclares the
kernel-scope ``smem``/``grid`` it uses, and calls it at the same place. The entry keeps every
top-level statement between blocks (prologue tick, the final barrier/tick), so the program's
order of intervals, barriers, ticks and arithmetic is unchanged; there is still exactly one
``__global__`` entry and one cooperative launch. The functions are then distributed over
``units`` relocatable translation units (``-rdc=true``) that compile concurrently and are
device-linked by nvlink into one cubin (``step17_build``'s driver-API shim embeds it).

Refusals keep a block inline (never a different program):
  * any kernel-scope declaration other than ``smem``/``grid`` (a block could capture it);
  * ``return`` outside a lambda, ``goto``, or a label;
  * preprocessor directives other than ``#pragma`` and self-contained ``#define``/``#undef`` pairs,
    or use of a macro defined elsewhere in the kernel body;
  * blocks matching an explicit ``keep_inline`` pattern (e.g. inline warpgroup-MMA bodies, which
    ptxas serializes inside ``__noinline__`` callees).
The whole pass refuses programs with ``__launch_bounds__`` (callee register limits would not be
known per unit), host markers needing the anchor runtime and namespace-scope device variables in
the program prefix. Secondary units are internal-linkage copies, so the build also refuses any
secondary unit that defines global or constant variables beyond a read-only allowlist (they would
be private to that unit); see ``unit_state``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

VERSION = 'step17-intervals-v1'
KERNEL = re.compile(r'extern "C" __global__ void latch_ops\(float\*\* args, float\* arena, '
                    r'unsigned long long\* ticks\) \{')
TEXT_STATEMENTS = (
    re.compile(r'extern __shared__ float smem\[\];'),
    re.compile(r'cg::grid_group grid = cg::this_grid\(\);'),
    re.compile(r'if \(ticks && blockIdx\.x == 0 && threadIdx\.x == 0\) ticks\[\d+\] = stamp\(\);'),
    re.compile(r'grid\.sync\(\);'),
    # LATCH-E2TTT-OUTER: lower.py's 'ectaq' claim-counter zeroing at kernel entry (CTA 0, before the first grid
    # barrier). Kernel-scope text stays verbatim in the entry ahead of every outlined interval call, so the
    # zeroing keeps its order relative to the first barrier and to every claim.
    re.compile(r'if \(blockIdx\.x == 0 && threadIdx\.x < \d+u\) '
               r'reinterpret_cast<unsigned\*>\(arena \+ \d+LL\)\[threadIdx\.x\] = 0u;'),
)
SIGNATURE = 'float** args, float* arena, unsigned long long* ticks'
NAMESPACE_VARIABLE = re.compile(r'^(?:static )?(?:__device__|__constant__|__managed__)(?: __device__)? [^(;{]*;', re.M)
WGMMA_INLINE = (r'pflatlong::(?:forward_item6|kv_item6|dq_item6|forward_item|kv_item|dq_item)\b',
                r'pops4b::tma_phase_inline\b')
# LATCH-TINYMOE-XMAT: the flattened segmented-product body inlines its warpgroup MMA. Added only for programs that
# emit it, so other programs keep their unit sources and cache identity.
RAGGED_FLAT = r'latch_ragged::(?:t?flat2?|touter)<'
# LATCH-TRAIN-EXPERTS: claimed segmented products (feature "claim"); likewise added only for programs that emit them.
RAGGED_CLAIM = r'latch_ragged::(?:tflat2?|touter)_claim<'


def wgmma_inline(source):
    return (WGMMA_INLINE + ((RAGGED_FLAT,) if re.search(RAGGED_FLAT, source) else ())
            + ((RAGGED_CLAIM,) if re.search(RAGGED_CLAIM, source) else ()))


class Refusal(Exception):
    pass


def kernel_parts(source):
    """(prefix, head, body, rest): prefix before the entry, its signature line, body, closing brace + host tail."""
    matches = list(KERNEL.finditer(source))
    if len(matches) != 1 or source.count('__global__') != 1:
        raise Refusal('expected exactly one recognized latch_ops entry without launch bounds')
    tail = source.find('\nextern "C" int latch_query(', matches[0].end())
    if tail < 0:
        raise Refusal('kernel end marker missing')
    close = source.rfind('}', matches[0].end(), tail)
    return (source[:matches[0].start()], source[matches[0].start():matches[0].end()],
            source[matches[0].end():close], source[close:])


def scan(body):
    """Top-level pieces [(kind, start, end)] with kind 'text' or 'block'; skips comments, literals, directives."""
    pieces, depth, i, n, start, block = [], 0, 0, len(body), 0, None
    line_start = True
    while i < n:
        c = body[i]
        if line_start and c in ' \t':
            i += 1
            continue
        if line_start and c == '#':
            j = body.find('\n', i)
            while j > 0 and body[j - 1] == '\\':
                j = body.find('\n', j + 1)
            i = n if j < 0 else j
            continue
        line_start = c == '\n'
        if body.startswith('//', i):
            j = body.find('\n', i)
            i = n if j < 0 else j
            continue
        if body.startswith('/*', i):
            j = body.find('*/', i + 2)
            if j < 0:
                raise Refusal('unterminated block comment')
            i = j + 2
            continue
        if c in '"\'':
            j = i + 1
            while j < n and body[j] != c:
                j += 2 if body[j] == '\\' else 1
            i = j + 1
            continue
        if c == '{':
            if depth == 0:
                pieces.append(('text', start, i))
                block = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                raise Refusal('unbalanced kernel body')
            if depth == 0:
                pieces.append(('block', block, i + 1))
                start = i + 1
        i += 1
    if depth:
        raise Refusal('unbalanced kernel body')
    pieces.append(('text', start, n))
    return [p for p in pieces if p[0] == 'block' or body[p[1]:p[2]].strip()]


def _statements(text):
    clean = re.sub(r'//[^\n]*', '', re.sub(r'/\*.*?\*/', '', text, flags=re.S))
    if '#' in clean:
        raise Refusal('preprocessor directive at kernel scope')
    return [s.strip() + ';' for s in ' '.join(clean.split()).split(';') if s.strip()]


def check_kernel_scope(body, pieces):
    """Every top-level statement must be a known prologue/epilogue form; returns the smem declaration seen."""
    for kind, a, b in pieces:
        if kind != 'text':
            continue
        for statement in _statements(body[a:b]):
            if not any(p.fullmatch(statement) for p in TEXT_STATEMENTS):
                raise Refusal('unrecognized kernel-scope statement: ' + statement[:100])


_BLANKED = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)


def _strip(text):
    """Text with comments and string/char literals blanked (same length)."""
    return _BLANKED.sub(lambda m: ' ' * len(m.group(0)), text)


LAMBDA_HEAD = re.compile(r'\]\s*(?:\([^()]*\))?\s*(?:mutable\s*)?(?:->\s*[\w:<>,\s\*&]+?)?\s*$')


def block_refusal(block, body_macros, keep_inline):
    clean = _strip(block)
    for pattern in keep_inline:
        if re.search(pattern, clean):
            return 'kept inline by pattern ' + pattern
    lines = [l.strip() for l in clean.split('\n') if l.strip().startswith('#')]
    defined, undefined = [], []
    for line in lines:
        m = re.match(r'#\s*(\w+)\s*(\w*)', line)
        if not m:
            return 'unrecognized directive'
        if m.group(1) == 'pragma':
            continue
        if m.group(1) == 'define':
            defined.append(m.group(2))
        elif m.group(1) == 'undef':
            undefined.append(m.group(2))
        else:
            return 'preprocessor directive #' + m.group(1)
    if sorted(defined) != sorted(undefined):
        return 'macro defined in the block outlives it'
    idents = set(re.findall(r'[A-Za-z_]\w*', clean))
    if (idents & body_macros) - set(defined):
        return 'uses a macro defined elsewhere in the kernel body'
    if re.search(r'\bgoto\b', clean):
        return 'goto'
    # Labels: `name:` at the start of a line or statement (not `::`; case/default belong to a moved switch).
    if re.search(r'(?:^|[;{}])[ \t]*(?!case\b|default\b)[A-Za-z_]\w*[ \t]*:(?!:)', clean, re.M):
        return 'label'
    # return is allowed only inside lambda bodies.
    stack = []
    for m in re.finditer(r'[{}]|\breturn\b', clean):
        token = m.group(0)
        if token == '{':
            stack.append(bool(LAMBDA_HEAD.search(clean[max(0, m.start() - 300):m.start()])))
        elif token == '}':
            stack.pop()
        elif not any(stack):
            return 'return outside a lambda'
    return None


def _header_names(prefix, header_names):
    from .lower import _header_path
    seen, texts = set(), []
    pending = [n for n in header_names if n.endswith('.cuh')] + re.findall(r'#include "([\w.]+\.cuh)"', prefix)
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        text = _header_path(name).read_text()
        texts.append((name, text))
        pending += re.findall(r'#include "([\w.]+\.cuh)"', text)
    return texts


PRELUDE_STATEMENTS = (
    re.compile(r'float\*\* latch_pa = latch_opq\(args\);'),
    re.compile(r'float\* latch_pr = latch_opq\(arena\);'),
    re.compile(r'unsigned (lsr_tx|lsr_bx|lsr_nt|lsr_nb) = (threadIdx|blockIdx|blockDim|gridDim)\.x;'),
    re.compile(r'asm volatile\("" : "\+r"\(lsr_(?:tx|bx|nt|nb)\)\);'),
    re.compile(r'(?:const )?[\w:]+(?: ?\*)+ ?v_\w+ = (?:reinterpret_cast<[\w:]+(?: ?\*)+>\()?'
               r'(?:latch_pa\[\d+\]|latch_pr \+ \d+LL)\)?;'),
    # LATCH-TINYMOE-XMAT: views at constant offsets inside an arena root (`latch_pr + root + view`) are the same pure
    # re-derivation; accepting them keeps such intervals splittable (isolated warpgroup-MMA pieces).
    re.compile(r'(?:const )?[\w:]+(?: ?\*)+ ?v_\w+ = (?:reinterpret_cast<[\w:]+(?: ?\*)+>\()?'
               r'latch_pr(?: \+ \d+LL){2,}\)?;'),
)


# LATCH-TRAIN-MATMUL (matrix_bodies 'isolate', signalled by its never-matching sentinel in ``isolate``): a pointer
# view at a constant element offset from an arena/argument view is the same pure re-derivation.
OFFSET_VIEW = re.compile(r'(?:const )?[\w:]+(?: ?\*)+ ?v_\w+ = reinterpret_cast<[\w:]+(?: ?\*)+>\('
                         r'(?:latch_pr(?: \+ \d+LL)+|latch_pa\[\d+\])\) \+ \d+LL;')


def split_interval(name, block, limit, isolate=()):
    """Split one interval's consecutive phases (``{...}`` separated only by ``__syncthreads();``) into piece
    functions of about ``limit`` bytes. The interval's prelude must consist only of pure re-derivations
    (opaque base pointers, hardware indices, arena/argument pointer declarations); each piece repeats it.
    Phases matching an ``isolate`` pattern each become their own piece (no other calls in that function:
    under relocatable compilation any call makes ptxas serialize the function's warpgroup MMA, C7509).
    Returns (pieces[(name, text, weight)], outer_block) or None when the interval does not have this form.
    """
    inner_start, inner_end = block.index('{') + 1, block.rindex('}')
    inner = block[inner_start:inner_end]
    try:
        rows = scan(inner)
    except Refusal:
        return None
    subs = [(a, b) for kind, a, b in rows if kind == 'block']
    if len(subs) < 2:
        return None
    prelude = inner[:subs[0][0]]
    try:
        statements = _statements(prelude)
    except Refusal:
        return None
    accepted = PRELUDE_STATEMENTS + ((OFFSET_VIEW,) if any('matrix_bodies_offset_views' in p for p in isolate) else ())
    if not all(any(p.fullmatch(st) for p in accepted) for st in statements):
        return None
    # Phases: the maximal run of sub-blocks separated only by __syncthreads(); anything after (the closing
    # grid barrier and tick stanza) stays in the outer interval function.
    count = 1
    for (_, b0), (a1, _) in zip(subs, subs[1:]):
        try:
            if _statements(inner[b0:a1]) != ['__syncthreads();']:
                break
        except Refusal:
            break
        count += 1
    subs = subs[:count]
    if len(subs) < 2:
        return None
    groups, current, size = [], [], 0
    for a, b in subs:
        alone = any(re.search(pattern, _strip(inner[a:b])) for pattern in isolate)
        if current and (alone or size + (b - a) > limit):
            groups.append(current)
            current, size = [], 0
        current.append((a, b))
        size += b - a
        if alone:
            groups.append(current)
            current, size = [], 0
    if current:
        groups.append(current)
    if len(groups) < 2:
        return None
    pieces, out, last = [], [], 0
    for k, group in enumerate(groups):
        a, b = group[0][0], group[-1][1]
        run = inner[a:b]
        piece = f'{name}_p{k}'
        clean = _strip(run)
        decls = ('  extern __shared__ float smem[];\n' if re.search(r'\bsmem\b', clean) else '') + \
                ('  cg::grid_group grid = cg::this_grid();\n' if re.search(r'\bgrid\b', clean) else '')
        pieces.append((piece, f'static __device__ __noinline__ void {piece}({SIGNATURE}) {{\n{decls}{{\n{prelude}'
                              f'\n{run}\n}}\n}}\n', len(run)))
        out += [inner[last:a], f'{piece}(args, arena, ticks);']
        last = b
    out.append(inner[last:])
    return pieces, block[:inner_start] + ''.join(out) + block[inner_end:]


def outline(source, header_names=(), *, keep_inline=WGMMA_INLINE, split_bytes=None, isolate=()):
    """Return (prefix, functions[(name, text, weight)], kernel_with_calls_and_rest, stats) or raise Refusal.

    ``split_bytes`` additionally splits intervals larger than that into phase-run pieces (split_interval);
    ``isolate`` patterns put each matching phase of a multi-phase interval in a piece of its own.
    """
    prefix, head, body, rest = kernel_parts(source)
    for marker in ('__managed__', 'cudaMemcpyToSymbol', 'latch_cargs'):
        if marker in source:
            raise Refusal('host-visible marker ' + marker)
    if NAMESPACE_VARIABLE.search(prefix):
        raise Refusal('namespace-scope device variable in the program prefix')
    # Header variables are often inside inactive diagnostic #if regions; the build instead refuses any
    # secondary unit whose compiled PTX defines global/constant variables beyond a read-only allowlist (unit_state).
    pieces = scan(body)
    check_kernel_scope(body, pieces)
    body_macros = set(re.findall(r'^\s*#\s*define\s+(\w+)', body, re.M))
    functions, out, refusals, last, blocks_outlined, split_rows = [], [], {}, 0, 0, []
    for kind, a, b in pieces:
        if kind != 'block':
            continue
        block = body[a:b]
        why = block_refusal(block, body_macros, keep_inline)
        if why:
            refusals[why] = refusals.get(why, 0) + 1
            continue
        name = f'latch_s17_block_{blocks_outlined}'
        blocks_outlined += 1
        isolating = any(re.search(pattern, _strip(block)) for pattern in isolate)
        if (split_bytes and len(block) > split_bytes) or isolating:
            split = split_interval(name, block, split_bytes or len(block), isolate)
            if split is not None:
                pieces_, block = split
                functions += pieces_
                split_rows.append(dict(function=name, bytes=b - a, pieces=len(pieces_)))
        clean = _strip(block)
        decls = ''
        if re.search(r'\bsmem\b', clean):
            decls += '  extern __shared__ float smem[];\n'
        if re.search(r'\bgrid\b', clean):
            decls += '  cg::grid_group grid = cg::this_grid();\n'
        text = f'static __device__ __noinline__ void {name}({SIGNATURE}) {{\n{decls}{block}\n}}\n'
        functions.append((name, text, len(block)))
        out.append(body[last:a])
        out.append(f'{name}(args, arena, ticks);')
        last = b
    out.append(body[last:])
    if not functions:
        raise Refusal('no outlinable interval')
    blocks = sum(1 for p in pieces if p[0] == 'block')
    stats = dict(version=VERSION, blocks=blocks, outlined=blocks_outlined, inline=blocks - blocks_outlined,
                 functions=len(functions), split_intervals=split_rows, split_bytes=split_bytes, isolate=list(isolate),
                 refusals=refusals, body_bytes=len(body), keep_inline=list(keep_inline))
    return prefix, functions, head + ''.join(out) + rest, stats


def outlined_program_source(source, header_names=(), *, keep_inline=WGMMA_INLINE, split_bytes=None, isolate=(),
                            shared_pool=False):
    """One translation unit: functions before the entry (whole-program compilation, no relocatable link)."""
    prefix, functions, kernel, stats = outline(source, header_names, keep_inline=keep_inline, split_bytes=split_bytes,
                                               isolate=isolate)
    if shared_pool:
        functions, slots = pool_shared_flags(functions)
        prefix += f'__shared__ __align__(16) unsigned {POOL}[{max(1, slots)}];\n'
        stats = dict(stats, shared_pool_slots=slots)
    return prefix + f'// STEP17 outlined intervals ({VERSION})\n' + ''.join(t for _, t, _ in functions) + kernel, stats


def weights(prefix, functions):
    """Approximate device-code volume per function: text, called prefix helpers and heavy template sites."""
    from .step17_outline import code_weight, prefix_function_sizes
    sizes = prefix_function_sizes(prefix)
    return [code_weight(text, sizes) + 20000 * len(re.findall(r'pflatlong::\w+_item\d*<|pops4b::\w+<', text))
            for _, text, _ in functions]


def partition_sources(source, header_names, units, *, keep_inline=WGMMA_INLINE, split_bytes=None, isolate=(),
                      shared_pool=False):
    """([(file, text)], stats): unit0 holds the entry; units 1..k-1 hold balanced interval functions."""
    from .step17_build import _header_texts, _internalized_unit, _secondary_prefix
    prefix, functions, kernel, stats = outline(source, header_names, keep_inline=keep_inline, split_bytes=split_bytes,
                                               isolate=isolate)
    pool = ''
    if shared_pool:
        functions, slots = pool_shared_flags(functions)
        pool = f'__shared__ __align__(16) unsigned {POOL}[{max(1, slots)}];\n'
        stats = dict(stats, shared_pool_slots=slots)
    secondary = _secondary_prefix(prefix)
    weight = weights(prefix, functions)
    bins = [[] for _ in range(max(1, min(units - 1, len(functions))))]
    loads = [0] * len(bins)
    for index in sorted(range(len(functions)), key=lambda i: -weight[i]):
        target = loads.index(min(loads))
        bins[target].append(index)
        loads[target] += weight[index]
    declarations = '\n'.join(f'__device__ void {name}({SIGNATURE});' for name, _, _ in functions)
    sources = [('unit0.cu', prefix + pool + f'// STEP17 interval unit 0 ({VERSION}): kernel entry\n' + declarations
                + '\n' + kernel)]
    system = ''.join(line + '\n' for line in sorted({' '.join(m.split()) for m in re.findall(
        r'^[ \t]*#[ \t]*include[ \t]*<[^>\n]+>', '\n'.join(_header_texts(prefix, header_names)), re.M)}))
    # Every unit declares every interval function at global scope (outside its internal namespace), so an
    # interval split into pieces calls its pieces' external definitions wherever they were placed.
    external = '\n'.join(f'__device__ void {name}({SIGNATURE});' for name, _, _ in functions) + '\n'
    # Relocatable (-rdc) units refer to unit 0's single pool definition (a sized extern static shared).
    external += pool.replace('__shared__', 'extern __shared__', 1)
    for k, members in enumerate(bins, 1):
        texts = [functions[i][1] for i in sorted(members)]
        sources.append((f'unit{k}.cu', _internalized_unit(secondary, texts, system + external, k)))
    stats = dict(stats, units=len(sources), unit_weights=loads, unit_members=[sorted(m) for m in bins],
                 unit_sha256=[hashlib.sha256(t.encode()).hexdigest() for _, t in sources])
    return sources, stats


def identity(units, keep_inline=WGMMA_INLINE, unit_flags=('-rdc=true',), split_bytes=None, isolate=(), shared_pool=False):
    """Every input that changes the generated unit sources or their compile commands (cache-key material)."""
    from .step17_build import PARTITION_VERSION
    return {'version': VERSION, 'partition_version': PARTITION_VERSION, 'units': int(units),
            **({'split_bytes': int(split_bytes)} if split_bytes else {}),
            **({'isolate': list(isolate)} if isolate else {}),
            **({'shared_pool': 'block-scope 4-byte __shared__ flags pooled across functions v1'} if shared_pool else {}),
            'keep_inline': list(keep_inline), 'unit_flags': list(unit_flags), 'internal_linkage': True,
            'unit_header_patch': RENEW_A_VERSION, 'sass_guard': 'wgmma_guard loop-carried register-A'}




PTX_VARIABLE = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+)?\.(global|const)\s+(?:\.align\s+\d+\s+)?\.\w+\s+'
                          r'([\w$]+)(\[\d*\])?(\s*=)?', re.M)


def unit_state(ptx_text):
    """Global/constant variables a relocatable unit defines, classified against a read-only allowlist.

    Secondary units are internal-linkage copies, so any variable they define is private to that unit.
    Allowed: libcu++'s stateless one-byte tag objects (cuda::std::piecewise_construct, ranges CPOs) and
    CUDA math's initialized read-only tables (__cudart_*, e.g. __cudart_i2opi_f for trig reduction) and
    the compiler's initialized anonymous literals ($str*, __unnamed_N: string literals such as assert
    messages and function names), and ops.prologue_batch's initialized block-scope ``static const`` member
    tables (names containing ``latch_ptab_``; read-only, so a private copy per unit is the same data).
    Everything else (program/header __device__ or __constant__ state, function statics) is refused.
    """
    allowed, refused = [], []
    for space, name, extent, init in PTX_VARIABLE.findall(ptx_text):
        row = f'.{space} {name}{extent}{" =" if init else ""}'
        if (space == 'global' and 'cuda3std' in name and extent == '[1]' and not init) or \
                (name.startswith('__cudart_') and init) or \
                (init and re.fullmatch(r'\$str(?:\$\d+)?|__unnamed_\d+', name)) or \
                (init and 'latch_ptab_' in name):
            allowed.append(row)
        else:
            refused.append(row)
    return dict(allowed=sorted(set(allowed)), refused=sorted(set(refused)))


RENEW_A_VERSION = 'pflatlong-renew-a-v1 (LATCH-COLD-PREP megakernel-lab a495c1b730)'
_RENEW_A = """// LATCH-TINYMOE-BUILD unit copy: LATCH-COLD-PREP's PFLATLONG_RENEW_A (megakernel-lab a495c1b730). CUDA 12.8 ptxas
// can reuse the registers of a loop-invariant register A fragment of wgmma for other values later in the loop body
// when the attention item is compiled in a small context (separately compiled interval functions); each wrapper call
// issues its products from a copy (fragment XOR a runtime zero from redux.sync, which ptxas cannot fold). Values are
// unchanged; the build additionally refuses any linked SASS with a loop-carried A-operand finding (wgmma_guard).
PFLATLONG_DEV void renew_a(uint32_t (&f)[4][4], const uint32_t (&a)[4][4]) {
  unsigned lane, zero;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
  asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(zero) : "r"(lane));
#pragma unroll
  for (int k = 0; k < 4; ++k)
#pragma unroll
    for (int e = 0; e < 4; ++e) f[k][e] = a[k][e] ^ zero;
  asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}
"""
_WG_NT = ("PFLATLONG_DEV void wg_nt(float (&acc)[8][4], const uint32_t (&a)[4][4], const char* tile, bool accumulate) {\n",
          "PFLATLONG_DEV void wg_nt(float (&acc)[8][4], const uint32_t (&a_in)[4][4], const char* tile, bool accumulate) {\n"
          "  uint32_t a[4][4];\n  renew_a(a, a_in);\n")
_WG_NT6 = ("PFLATLONG_DEV void wg_nt6(float (&acc)[8][4], const uint32_t (&a)[4][4], uint64_t d0) {\n",
           "PFLATLONG_DEV void wg_nt6(float (&acc)[8][4], const uint32_t (&a_in)[4][4], uint64_t d0) {\n"
           "  uint32_t a[4][4];\n  renew_a(a, a_in);\n")
_ANCHOR = '// acc (64 rows x 64) (+)= A (64 x 64, registers, 4 k16 steps) * B where B[k][n] = tile[n][k] (K-major tile).\n'


def unit_headers(prefix, header_names):
    """{header: text} copies placed beside the unit sources (quoted includes find them first).

    Only attention_long_flat.cuh is patched (renew_a before its register-A wgmma wrappers); the program's
    manifest, header identities and every other header are unchanged. Refuses a header whose exact
    wrapper text differs from the audited form.
    """
    rows = {}
    for name, text in _header_names(prefix, header_names):
        if name != 'attention_long_flat.cuh':
            continue
        for old, _ in (_WG_NT, _WG_NT6):
            if text.count(old) != 1:
                raise Refusal('attention_long_flat.cuh register-A wrapper differs from the audited form')
        if text.count(_ANCHOR) != 1 or 'renew_a(' in text:
            raise Refusal('attention_long_flat.cuh renew_a anchor differs from the audited form')
        text = text.replace(_ANCHOR, _RENEW_A + _ANCHOR).replace(*_WG_NT).replace(*_WG_NT6)
        rows[name] = text
    return rows


POOL = 'latch_s17_pool'
FLAG_DECLARATION = re.compile(r'^([ \t]*)__shared__ (unsigned|int|float) (\w+);[ \t]*$', re.M)


def pool_shared_flags(functions):
    """Opt-in: block-scope 4-byte ``__shared__`` scalars of outlined functions share one static pool.

    The lowering's dynamic shared plan for programs above 256 tokens leaves 2,048 bytes for static
    shared, but each interval declares its own block-scope reduction flags (TinyMoE: 627 x 4 bytes).
    Outlined functions run one after another and every function boundary is a barrier (intervals end
    with ``grid.sync()``; split pieces are separated by ``__syncthreads()`` in their interval), while a
    block-scope flag is only reachable inside its own function. So the k-th flag declaration of every
    function can use pool slot k: declarations within one function keep distinct storage. Each
    ``__shared__ T name;`` becomes ``T& name = *reinterpret_cast<T*>(&latch_s17_pool[k]);``.
    Returns (functions, slots); refuses an interval function without a closing grid barrier.
    """
    out, slots = [], 0
    for name, text, weight in functions:
        found = FLAG_DECLARATION.findall(text)
        if not found:
            out.append((name, text, weight))
            continue
        if '_p' not in name.rsplit('block_', 1)[-1] and 'grid.sync();' not in _strip(text):
            raise Refusal('pooled interval function lacks its closing grid barrier: ' + name)
        counter = iter(range(len(found)))

        def replace(m):
            k = next(counter)
            return f'{m.group(1)}{m.group(2)}& {m.group(3)} = *reinterpret_cast<{m.group(2)}*>(&{POOL}[{k}]);'
        out.append((name, FLAG_DECLARATION.sub(replace, text), weight))
        slots = max(slots, len(found))
    return out, slots
