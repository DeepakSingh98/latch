"""STEP17: outline large self-contained kernel blocks into separately allocated device functions.

ptxas inlines every generated phase into the single cooperative entry (the suite's
transformer/recurrent entries are 44K/77K SASS instructions) and its time grows faster
than linearly with that function. This post-lowering pass moves large brace blocks of
the kernel body, unchanged, into ``static __device__ __noinline__`` functions called at
the same place. A block's code is not edited: kernel-scope pointer declarations it
uses (``T* v = arena + C;`` / ``args[k]`` forms, ``extern __shared__``, the grid
group) are reproduced verbatim inside the function, so addresses keep their constant
immediates; scalar kernel-scope values (loop indices, clamped counts) are passed by
value. Blocks are refused (kept inline) when they could change meaning out of line:
control transfer (return/break/continue/goto), assignment to or address of a passed
value, arrays or unrecognized declarations from an enclosing scope, or any program
using warpgroup MMA (P-OPS5-C/P7-E: calls serialize HGMMA), tensor-map/TMA bodies or
trace exports. The whole pass refuses unknown kernel framing and returns the input.

Opt-in: ``LATCH_STEP17_OUTLINE=on`` (or ``outline(program)``). The changed source has a
new source hash, so its native cache key differs from the inline program's.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from .p7e_codegen import _bound, _is_member, _tokens

VERSION = 'step17-outline-v1'
MIN_CHARS = int(os.environ.get('LATCH_STEP17_OUTLINE_MIN', '1500'))
KERNEL = re.compile(r'extern "C" __global__ (?:__launch_bounds__\(\d+, \d+\) )?void (?:__launch_bounds__\(\d+, \d+\) )?'
                    r'latch_ops\(float\*\* args, '
                    r'float\* arena, unsigned long long\* ticks\) \{')
# Program-wide refusals: asynchronous warpgroup MMA, TMA/bulk copies and trace exports must stay entry-resident.
PROGRAM_REFUSE = ('wgmma', 'cp.async.bulk', 'mbarrier', 'cuTensorMap', 'CUtensorMap', 'P5C_TMAP', 'POPS4B_TMAP',
                  'latch_cargs', 'TRACE')
CONTROL = {'return', 'break', 'continue', 'goto', 'case', 'default'}
SCALARS = r'(?:const )?(?:int|unsigned|long long|unsigned long long|int64_t|uint64_t|int32_t|uint32_t|bool|float)'
PURE_IDENTS = {'args', 'arena', 'reinterpret_cast', 'float', 'int', 'int64_t', 'unsigned', '__nv_bfloat16', 'uint32_t',
               'int32_t', 'uint64_t', 'long', 'char', 'const', 'double', 'uint2', 'uint4', 'float2', 'float4', 'unsigned'}


class Refusal(Exception):
    pass


def enabled():
    return os.environ.get('LATCH_STEP17_OUTLINE', '').lower() in {'1', 'on', 'true', 'yes'}


def _significant(tokens):
    return [i for i, (k, _) in enumerate(tokens) if k not in ('space', 'comment')]


def _match(tokens, sig, start, open_, close):
    """Index into sig of the token closing the bracket at sig[start]."""
    depth = 0
    for n in range(start, len(sig)):
        text = tokens[sig[n]][1]
        if text == open_:
            depth += 1
        elif text == close:
            depth -= 1
            if depth == 0:
                return n
    raise Refusal(f'unbalanced {open_}{close}')


def _classify(text):
    """Kernel-scope declaration statement -> (names, info) where info kind is pure/shared/grid/value/opaque."""
    s = ' '.join(text.split())
    if s == 'extern __shared__ float smem[];' or re.fullmatch(r'extern __shared__ [\w ]+ \w+\[\];', s):
        name = re.search(r'(\w+)\[\];$', s).group(1)
        return {name: dict(kind='shared', decl=s)}
    if re.fullmatch(r'cg::grid_group (\w+) = cg::this_grid\(\);', s):
        return {re.search(r'grid_group (\w+)', s).group(1): dict(kind='grid', decl=s)}
    m = re.fullmatch(r'(?:const )?((?:unsigned |signed |long |const )*[\w:]+(?: ?\*)+) (\w+) = (.+);', s)
    if m:
        expr_idents = set(re.findall(r'[A-Za-z_]\w*', re.sub(r'\b\d[\w.]*', ' ', m.group(3))))
        if expr_idents <= PURE_IDENTS and ('arena' in expr_idents or 'args' in expr_idents):
            return {m.group(2): dict(kind='pure', decl=s)}
        return {m.group(2): dict(kind='value', type=m.group(1).replace(' *', '*'))}
    m = re.fullmatch(SCALARS + r' (\w+) = [^,;]+;', s)
    if m:
        ty = re.match(SCALARS, s).group(0).replace('const ', '')
        return {m.group(1): dict(kind='value', type=ty)}
    return None


KEYWORDS = {'return', 'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'sizeof', 'new', 'delete', 'goto',
            'long', 'int', 'unsigned', 'signed', 'short', 'char', 'float', 'double', 'bool', 'void', 'const',
            'volatile', 'static', 'extern', 'auto', 'inline', 'unroll', 'pragma', 'template', 'typename',
            '__shared__', '__device__', '__restrict__', 'constexpr', 'struct', 'asm', 'true', 'false'}


def _declared_like(tokens):
    """Over-approximate every identifier declared anywhere in a token list (used only to refuse)."""
    sig = _significant(tokens)
    names = set(_bound(tokens))
    for n in range(1, len(sig) - 1):
        kind, text = tokens[sig[n]]
        if kind != 'ident' or _is_member(tokens, sig[n]) or '::' in text or text in KEYWORDS:
            continue
        before_kind, before = tokens[sig[n - 1]]
        after = tokens[sig[n + 1]][1]
        if after in ('=', '[', ';', ',', ')', '(', '{') and (before in ('*', '&', '>') or
                                                            (before_kind == 'ident' and before not in KEYWORDS)):
            names.add(text)
    return names


class _Outliner:
    def __init__(self, body, min_chars=MIN_CHARS):
        self.min_chars = min_chars
        self.body = body
        self.tokens = _tokens(body)
        self.sig = _significant(self.tokens)
        self.functions = []
        self.replacements = []   # (start offset, end offset, call text)
        self.refusals = {}
        offsets, total = [], 0
        for _, text in self.tokens:
            offsets.append(total)
            total += len(text)
        self.offsets = offsets + [total]
        self.kernel_names = _declared_like(self.tokens) - KEYWORDS

    def text(self, a, b):
        """Source text of significant tokens a..b inclusive (with interior whitespace/comments)."""
        return self.body[self.offsets[self.sig[a]]:self.offsets[self.sig[b] + 1]]

    def tok(self, n):
        return self.tokens[self.sig[n]][1]

    def compound(self, start, end, visible):
        """Walk statements of the compound whose contents are sig[start..end] (exclusive of braces)."""
        visible = dict(visible)
        n = start
        while n <= end:
            t = self.tok(n)
            if t == '{':
                close = _match(self.tokens, self.sig, n, '{', '}')
                self.candidate(n, close, visible)
                n = close + 1
                continue
            if t in ('for', 'while'):
                paren = n + 1
                if self.tok(paren) != '(':
                    raise Refusal('loop without header')
                pclose = _match(self.tokens, self.sig, paren, '(', ')')
                inner = dict(visible)
                if t == 'for':
                    header = ' '.join(self.text(paren + 1, pclose - 1).split())
                    m = re.match(SCALARS + r' (\w+) = ', header)
                    if m:
                        inner[m.group(1)] = dict(kind='value', type=re.match(SCALARS, header).group(0).replace('const ', ''))
                    elif _bound(self.tokens[self.sig[paren]:self.sig[pclose]]):
                        for name in _bound(self.tokens[self.sig[paren]:self.sig[pclose]]):
                            inner[name] = dict(kind='opaque')
                if pclose + 1 <= end and self.tok(pclose + 1) == '{':
                    bclose = _match(self.tokens, self.sig, pclose + 1, '{', '}')
                    self.compound(pclose + 2, bclose - 1, inner)
                    n = bclose + 1
                else:
                    n = self.skip_statement(pclose + 1, end)
                continue
            if t in ('switch', 'do'):
                # Never outlined or entered; skip the whole construct exactly.
                k = n + 1
                if t == 'switch':
                    k = _match(self.tokens, self.sig, k, '(', ')') + 1
                if self.tok(k) != '{':
                    raise Refusal(t + ' without braces')
                k = _match(self.tokens, self.sig, k, '{', '}') + 1
                if t == 'do':
                    k = self.skip_statement(k, end)
                n = k
                continue
            if t == 'if':
                paren = n + 1
                pclose = _match(self.tokens, self.sig, paren, '(', ')')
                n = pclose + 1
                if n <= end and self.tok(n) == '{':
                    close = _match(self.tokens, self.sig, n, '{', '}')
                    self.candidate(n, close, visible)
                    n = close + 1
                else:
                    n = self.skip_statement(n, end)
                if n <= end and self.tok(n) == 'else':
                    n += 1
                continue
            # Plain statement: record declarations.
            stop = self.skip_statement(n, end)
            statement = self.text(n, stop - 1)
            stokens = self.tokens[self.sig[n]:self.sig[stop - 1] + 1]
            declared = _bound(stokens)
            info = _classify(statement) if declared or statement.lstrip().startswith(('extern', 'cg::')) else None
            if info:
                visible.update(info)
            for name in declared:
                if not info or name not in info:
                    visible[name] = dict(kind='opaque')
            n = stop
        return visible

    def skip_statement(self, n, end):
        depth = 0
        while n <= end:
            t = self.tok(n)
            if t in '([{':
                depth += 1
            elif t in ')]}':
                depth -= 1
            elif t == ';' and depth == 0:
                return n + 1
            n += 1
        return n

    def candidate(self, open_, close, visible):
        text = self.text(open_, close)
        if len(text) < self.min_chars:
            return
        try:
            call, function = self.outline(open_, close, text, visible)
        except Refusal as why:
            self.refusals[str(why)] = self.refusals.get(str(why), 0) + 1
            self.compound(open_ + 1, close - 1, visible)
            return
        self.functions.append(function)
        self.replacements.append((self.offsets[self.sig[open_]], self.offsets[self.sig[close] + 1], call))

    def outline(self, open_, close, text, visible):
        btokens = self.tokens[self.sig[open_]:self.sig[close] + 1]
        bsig = _significant(btokens)
        used = []
        for i in bsig:
            kind, value = btokens[i]
            if kind == 'ident' and not _is_member(btokens, i):
                used.append((i, value))
            if value in CONTROL:
                raise Refusal(f'control transfer {value}')
        # Strict declarations, plus declaration-like names that are not visible from an enclosing scope
        # (e.g. comma declarators); a visible name is never treated as block-local here.
        bound = set(_bound(btokens)) | (_declared_like(btokens) - set(visible))
        free = {v for _, v in used if v not in bound and v not in KEYWORDS}
        needed = sorted(free & set(visible))
        unscoped = sorted((free - set(visible)) & self.kernel_names)
        if unscoped:
            raise Refusal('kernel-local name not tracked in scope: ' + ','.join(unscoped[:4]))
        params, redecl = [], []
        for name in needed:
            row = visible[name]
            if row['kind'] == 'param':
                continue
            if row['kind'] == 'opaque':
                raise Refusal('enclosing array/unrecognized declaration')
            if row['kind'] in ('pure', 'shared', 'grid'):
                redecl.append(row['decl'])
            else:
                params.append((row['type'], name))
        for _, name in params:
            for n, i in enumerate(bsig):
                kind, value = btokens[i]
                if value != name or kind != 'ident' or _is_member(btokens, i):
                    continue
                after = btokens[bsig[n + 1]][1] if n + 1 < len(bsig) else ''
                after2 = btokens[bsig[n + 2]][1] if n + 2 < len(bsig) else ''
                before = btokens[bsig[n - 1]][1] if n else ''
                if (after == '=' and after2 != '=') or after in ('+', '-', '*', '/', '|', '&', '^', '<', '>') and after2 == '=' \
                        or after in ('++', '--') or before in ('++', '--') or (after == '+' and after2 == '+') \
                        or (after == '-' and after2 == '-') or before == '&':
                    raise Refusal('passed value is assigned or addressed')
        index = len(self.functions)
        name = f'latch_s17_block_{index}'
        signature = ', '.join(['float** args', 'float* arena', 'unsigned long long* ticks'] +
                              [f'{ty} {pname}' for ty, pname in params])
        decls = ''.join(f'  {line}\n' for line in redecl)
        if os.environ.get('LATCH_STEP17_ASSUME_GLOBAL', 'on').lower() in {'1', 'on', 'true', 'yes'}:
            # The kernel entry knows its pointer parameters address global memory; an out-of-line callee does not.
            # Restate that fact so loads/stores through args/arena keep their global (LDG/STG) forms.
            decls = '  __builtin_assume(__isGlobal(args)); __builtin_assume(__isGlobal(arena));\n' + decls
        function = (f'static __device__ __noinline__ void {name}({signature}) {{\n{decls}{text}\n}}\n')
        call = f'{name}({", ".join(["args", "arena", "ticks"] + [p for _, p in params])});'
        return call, function


def prefix_function_sizes(prefix):
    """Text size of each forceinline device function in the prefix (a proxy for its inlined code volume)."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(
        r'^__device__ __forceinline__ [\w:<>, *&]+? (\w+)\(', prefix, re.M)]
    sizes = {}
    for (start, name), nxt in zip(starts, starts[1:] + [(len(prefix), None)]):
        sizes[name] = sizes.get(name, 0) + nxt[0] - start
    return sizes


def code_weight(text, sizes):
    """Approximate device-code volume of a statement range: text, called prefix functions, template bodies."""
    called = set(re.findall(r'\b(\w+)\(', text)) & set(sizes)
    return len(text) + sum(sizes[c] for c in called) + 3000 * len(re.findall(r'::\w+_(?:phase|item)<', text))


class _Chunker(_Outliner):
    """Coarse mode: contiguous runs of statements (phases and their barriers) become a few large functions.

    A run ends only where nothing it declares is used by a later statement of the same compound, so a run
    never needs to return values. Heavy compound statements ({...}, for/if bodies) are split recursively.
    """

    def __init__(self, body, chunks, sizes):
        super().__init__(body, 0)
        self.chunks, self.target, self.sizes = int(chunks), None, sizes

    def statements(self, start, end):
        rows, n = [], start
        while n <= end:
            t = self.tok(n)
            child = None
            if t == '{':
                stop = _match(self.tokens, self.sig, n, '{', '}') + 1
                child = (n + 1, stop - 2, None)
            elif t in ('for', 'while'):
                pclose = _match(self.tokens, self.sig, n + 1, '(', ')')
                if pclose + 1 <= end and self.tok(pclose + 1) == '{':
                    stop = _match(self.tokens, self.sig, pclose + 1, '{', '}') + 1
                    child = (pclose + 2, stop - 2, (n + 2, pclose - 1) if t == 'for' else None)
                else:
                    stop = self.skip_statement(pclose + 1, end)
            elif t in ('switch', 'do'):
                k = n + 1
                if t == 'switch':
                    k = _match(self.tokens, self.sig, k, '(', ')') + 1
                k = _match(self.tokens, self.sig, k, '{', '}') + 1
                stop = self.skip_statement(k, end) if t == 'do' else k
            elif t == 'if':
                pclose = _match(self.tokens, self.sig, n + 1, '(', ')')
                k = pclose + 1
                if k <= end and self.tok(k) == '{':
                    stop = _match(self.tokens, self.sig, k, '{', '}') + 1
                    child = (k + 1, stop - 2, None)
                else:
                    stop = self.skip_statement(k, end)
                if stop <= end and self.tok(stop) == 'else':
                    child = None   # never split if/else pairs; the whole statement moves as one unit
                    k = stop + 1
                    if self.tok(k) == '{':
                        stop = _match(self.tokens, self.sig, k, '{', '}') + 1
                    else:
                        stop = self.skip_statement(k, end)
            else:
                stop = self.skip_statement(n, end)
            text = self.text(n, stop - 1)
            rows.append(dict(a=n, b=stop - 1, child=child, weight=code_weight(text, self.sizes), kind=t))
            n = stop
        return rows

    def declare(self, row, visible):
        if row['kind'] in ('{', 'for', 'while', 'if', 'switch', 'do'):
            return
        statement = self.text(row['a'], row['b'])
        stokens = self.tokens[self.sig[row['a']]:self.sig[row['b']] + 1]
        declared = _bound(stokens)
        info = _classify(statement) if declared or statement.lstrip().startswith(('extern', 'cg::')) else None
        if info:
            visible.update(info)
        for name in declared:
            if not info or name not in info:
                visible[name] = dict(kind='opaque')

    def names_used(self, a, b):
        tokens = self.tokens[self.sig[a]:self.sig[b] + 1]
        return {v for i, (k, v) in enumerate(tokens) if k == 'ident' and not _is_member(tokens, i)}

    def split(self, start, end, visible):
        visible = dict(visible)
        rows = self.statements(start, end)
        if self.target is None:
            self.target = max(1, sum(r['weight'] for r in rows)) / self.chunks
        run = []

        def flush(run, later_rows):
            if not run or sum(r['weight'] for r in run) < 0.05 * self.target:
                return
            a, b = run[0]['a'], run[-1]['b']
            text = '{\n' + self.text(a, b) + '\n}'
            try:
                call, function = self.outline(a, b, text, run_visible)
            except Refusal as why:
                self.refusals[str(why)] = self.refusals.get(str(why), 0) + 1
                return
            self.functions.append(function)
            self.replacements.append((self.offsets[self.sig[a]], self.offsets[self.sig[b] + 1], call))

        run_visible = dict(visible)
        weight = 0
        def closable(later):
            declared = {k for k, v in visible.items() if run_visible.get(k) is not v
                        and v['kind'] not in ('pure', 'shared', 'grid')}
            used_later = self.names_used(later[0]['a'], later[-1]['b']) if later else set()
            return not declared & used_later

        for index, row in enumerate(rows):
            if row['weight'] > 1.5 * self.target and row['child'] is not None:
                if closable(rows[index:]):
                    flush(run, rows[index:])
                else:
                    self.refusals['run keeps a declaration used later'] = \
                        self.refusals.get('run keeps a declaration used later', 0) + 1
                run, weight = [], 0
                cstart, cend, header = row['child']
                inner = dict(visible)
                if header is not None:
                    text = ' '.join(self.text(header[0], header[1]).split())
                    m = re.match(SCALARS + r' (\w+) = ', text)
                    if not m:
                        self.declare(row, visible)
                        continue
                    inner[m.group(1)] = dict(kind='value', type=re.match(SCALARS, text).group(0).replace('const ', ''))
                self.split(cstart, cend, inner)
                run_visible = dict(visible)
                continue
            if not run:
                before = dict(visible)
                self.declare(row, visible)
                added = {k: v for k, v in visible.items() if before.get(k) is not v}
                if added and all(v['kind'] in ('pure', 'value', 'shared', 'grid') for v in added.values()):
                    # A leading declaration stays in the caller: later runs redeclare or receive it.
                    continue
                visible = before
                run_visible = dict(visible)
            run.append(row)
            weight += row['weight']
            self.declare(row, visible)
            if weight >= self.target and closable(rows[index + 1:]):
                # Close the run only where later statements do not read a name it declared (recomputable
                # pointer declarations are reproduced by later runs and never block a cut).
                flush(run, rows[index + 1:])
                run, weight = [], 0
        flush(run, [])


def outline_parts(source, header_names=(), min_chars=None, chunks=None):
    """Split a generated source into (prefix, functions, kernel_with_calls_and_tail) or raise Refusal.

    ``functions`` is a list of (name, text) static __noinline__ definitions; the kernel text calls them in
    place of the original blocks. The prefix is everything before the kernel entry, unchanged. With
    ``chunks`` the kernel body is cut into about that many contiguous statement runs (coarse mode).
    """
    from .lower import _header_path
    min_chars = int(os.environ.get('LATCH_STEP17_OUTLINE_MIN', MIN_CHARS)) if min_chars is None else int(min_chars)
    for marker in PROGRAM_REFUSE:
        if marker in source:
            raise Refusal(f'program contains {marker}')
    for name in header_names:
        if name.endswith('.cuh'):
            text = _header_path(name).read_text()
            if 'wgmma' in text or 'cp.async.bulk' in text:
                raise Refusal(f'header {name} contains asynchronous MMA/bulk copies')
    matches = list(KERNEL.finditer(source))
    if len(matches) != 1 or source.count('extern "C" __global__') != 1:
        raise Refusal('expected one recognized kernel entry')
    head_end = matches[0].end()
    tail = source.find('\nextern "C" int latch_query(', head_end)
    if tail < 0:
        raise Refusal('kernel end marker missing')
    close = source.rfind('}', head_end, tail)
    body = source[head_end:close]
    params = {'args': dict(kind='param'), 'arena': dict(kind='param'), 'ticks': dict(kind='param')}
    if chunks:
        sizes = prefix_function_sizes(source[:matches[0].start()])
        worker = _Chunker(body, chunks, sizes)
        worker.split(0, len(worker.sig) - 1, params)
    else:
        worker = _Outliner(body, min_chars)
        worker.compound(0, len(worker.sig) - 1, params)
    if not worker.functions:
        raise Refusal('no outlinable block')
    pieces, last = [], 0
    for start, end, call in sorted(worker.replacements):
        pieces.append(body[last:start])
        pieces.append(call)
        last = end
    pieces.append(body[last:])
    entry = matches[0].start()
    functions = [(re.search(r'void (latch_s17_block_\d+)\(', text).group(1), text) for text in worker.functions]
    stats = dict(functions=len(functions), block_refusals=worker.refusals, body_chars=len(body),
                 outlined_chars=sum(e - s for s, e, _ in worker.replacements), min_chars=min_chars,
                 mode='chunks' if chunks else 'blocks', chunks=chunks)
    return source[:entry], functions, source[entry:head_end] + ''.join(pieces) + source[close:], stats


def outline(program, min_chars=None):
    """Return (program, record). Unrecognized programs are returned unchanged with the refusal reason."""
    from .lower import GeneratedProgram
    record = dict(version=VERSION, applied=False)
    try:
        prefix, functions, kernel, stats = outline_parts(program.source, program.manifest.get('header_hashes', {}),
                                                         min_chars)
    except Refusal as why:
        record['refusal'] = str(why)
        return program, record
    new_source = prefix + '// STEP17 outlined kernel blocks (' + VERSION + ')\n' + ''.join(t for _, t in functions) + kernel
    manifest = dict(program.manifest)
    record.update(applied=True, parent_source_hash=program.cache_key, **stats)
    manifest['step17_outline'] = record
    manifest['source_hash'] = hashlib.sha256((new_source + json.dumps(manifest['header_hashes'], sort_keys=True))
                                             .encode()).hexdigest()
    result = GeneratedProgram(new_source, manifest, program.kernel_body, program.helpers, program.functions)
    for attribute in ('_selection_contracts',):
        if hasattr(program, attribute):
            setattr(result, attribute, getattr(program, attribute))
    return result, record


def maybe_outline(program):
    if not enabled():
        return program
    result, _ = outline(program)
    return result
