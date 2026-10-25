"""P7-E CPU contracts for the compact code generator (ops/p7e_codegen.py).

* P7E_MODE='off' reproduces the parent lowering byte for byte.
* fold: each phase re-derived independently from the emitted text (case body of p7e_op[pc] with its p7e_tab slots
  substituted) equals the original phase up to alpha-renaming of phase-bound identifiers, with every pointer and
  tensor-map reference resolved to the same storage address; manifest phase records, inputs and arena are unchanged;
  exactly one kernel entry; deterministic output.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

GRAPHS = {'o-l-bf16': '/donor5c/stage/o-l-bf16/graph.json', 'o-m-bf16': '/donor5c/stage/o-m-bf16/graph.json'}
PTR_SHARED = re.compile(r'reinterpret_cast<(?P<ty>[\w]+\*)>\(reinterpret_cast<char\*>\(arena\) \+ p7e_tab\[p7e_b \+ (?P<k>\d+)\]\)')
ARG_SHARED = re.compile(r'reinterpret_cast<(?P<ty>[\w]+\*)>\(args\[p7e_tab\[p7e_b \+ (?P<k>\d+)\]\]\)')
MAP_SHARED = re.compile(r'\(reinterpret_cast<const char\*>\(args\[(?P<n>\d+)\]\) \+ p7e_tab\[p7e_b \+ (?P<k>\d+)\]\)')


def _lower(graph, mode, options=None):
    from megakernel_lab.operation_compiler.ir import TrainingIR
    from megakernel_lab.operation_compiler.lower import lower
    from megakernel_lab.operation_compiler import p7e_codegen
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK
    # The integrated public entry uses compact_codegen instead of a persistent
    # module global. Generate its ordinary program, then exercise this pass's
    # explicit diagnostic modes under the same build lock.
    options = {'P7E_OUTLINE_KEEP': ('p5c::', 'pflatlong::', 'plong::', 'pops4b::', 'p7a4b::', 'p6e::'),
               **dict(options or {}), 'P7E_MODE': mode}
    with _LOCK:
        parent = lower(TrainingIR.from_dict(json.loads(Path(graph).read_text())), compact_codegen='off')
        previous = {k: getattr(p7e_codegen, k) for k in options}
        for k, v in options.items():
            setattr(p7e_codegen, k, v)
        try:
            return p7e_codegen.apply(parent)
        finally:
            for k, v in previous.items():
                setattr(p7e_codegen, k, v)


def _normalize_original(code, decls):
    """Original phase: comments dropped, pointers -> PTR(type, base, value), maps -> MAP(arg, offset)."""
    from megakernel_lab.operation_compiler import p7e_codegen as P
    tokens = [t for t in P._tokens(code) if t[0] != 'comment']
    text = []
    for i, (kind, tok) in enumerate(tokens):
        if kind == 'ident' and tok in decls.rows and not P._is_member(tokens, i):
            r = decls.rows[tok]
            text.append(f'PTR({r["ty"]},{r["base"]},{r["value"]})')
        else:
            text.append(tok)
    text = ''.join(text)
    return P.TMAP.sub(lambda m: f'MAP({m.group(1)},{m.group(2)})', text)


def _normalize_shared(body, table, base):
    body = re.sub(r'//[^\n]*', '', body)
    body = PTR_SHARED.sub(lambda m: f'PTR({m["ty"]},arena,{table[base + int(m["k"])]})', body)
    body = ARG_SHARED.sub(lambda m: f'PTR({m["ty"]},args,{table[base + int(m["k"])]})', body)
    return MAP_SHARED.sub(lambda m: f'MAP({m["n"]},{table[base + int(m["k"])]})', body)


def _alpha(text):
    """Rename phase-bound identifiers in order of first appearance (shared bodies keep their first member's names)."""
    from megakernel_lab.operation_compiler import p7e_codegen as P
    tokens = P._tokens(text)
    bound = P._bound(tokens)
    names, out = {}, []
    for i, (kind, tok) in enumerate(tokens):
        if kind == 'ident' and tok in bound and not P._is_member(tokens, i):
            out.append('@L%d' % names.setdefault(tok, len(names)))
        else:
            out.append(tok)
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def check_fold(graph, mode='fold', options=None):
    parent = _lower(graph, 'off')
    folded = _lower(graph, mode, options)
    again = _lower(graph, mode, options)
    return check_programs(parent, folded, again=again)


def check_programs(parent, folded, *, again=None):
    """Independent emitted-slot audit of already generated programs, including P6D.

    Supplying existing sources keeps this diagnostic independent of recapture or
    changed current lowering defaults. Omitted `again` does not assert determinism.
    """
    from megakernel_lab.operation_compiler import p7e_codegen as P
    rows = {**({'deterministic': folded.source == again.source} if again is not None else {}),
            'one_entry': folded.source.count('extern "C" __global__') == 1,
            'manifest_phases_unchanged': folded.manifest['phases'] == parent.manifest['phases'],
            'inputs_unchanged': folded.manifest['inputs'] == parent.manifest['inputs'],
            'arena_unchanged': folded.manifest['arena_bytes'] == parent.manifest['arena_bytes']}
    _, decl_lines, phases, _ = P.split_kernel(parent.source)
    decls = P.Declarations(decl_lines)
    src = folded.source
    table = [int(x) for x in re.search(r'p7e_tab\[\d+\] = \{([^}]*)\}', src).group(1).split(',')]
    ops_ = [int(x) for x in re.search(r'p7e_op\[\d+\] = \{([^}]*)\}', src).group(1).split(',')]
    bases = [int(x) for x in re.search(r'p7e_base\[\d+\] = \{([^}]*)\}', src).group(1).split(',')]
    cases = {}
    enter = (r'(?:  asm volatile\("// p7e body \d+" ::: "memory"\);\n  if \(p7e_s\) \{\n  grid\.sync\(\);\n'
             r'  if \(ticks && blockIdx\.x == 0 && threadIdx\.x == 0\) ticks\[p7e_s\] = stamp\(\);\n  \}\n)?')
    for m in re.finditer(r'\n  case (\d+): [^\n]*\n' + enter + r'\{\n(.*?)\n\}\n  break;', src, re.S):
        cases[int(m.group(1))] = m.group(2)
    bodies = {}
    head_re = re.compile(r'__device__ __noinline__ void p7e_body_(\d+)\(float\*\* args, float\* arena, int p7e_b\) \{\n'
                         r'  extern __shared__ float smem\[\];\n\{\n')
    for m in head_re.finditer(src):
        depth, pos = 1, m.end()
        for kind, tok in P._tokens(src[m.end():src.index('extern "C" __global__')]):
            if kind not in ('comment', 'string', 'char'):
                depth += tok.count('{') if tok == '{' else 0
                depth -= 1 if tok == '}' else 0
            if depth == 0:
                break
            pos += len(tok)
        text = src[m.end():pos]
        if not text.endswith('\n'):
            raise AssertionError('outlined body framing')
        bodies[int(m.group(1))] = text[:-1]
    for m in re.finditer(r'\n  case (\d+): [^\n]*\n' + enter + r'  p7e_body_(\d+)\(args, arena, p7e_b\);\n  break;', src):
        if m.group(1) != m.group(2) or int(m.group(1)) not in bodies:
            raise AssertionError('outlined call does not name its own body')
        cases[int(m.group(1))] = bodies[int(m.group(1))]
    rows['outlined_bodies_wgmma_free'] = all(not any(marker in b for marker in
        ('p5c::', 'pflatlong::', 'plong::', 'pops4b::', 'p7a4b::', 'p6e::', 'wgmma'))
        for b in bodies.values())
    tick = re.search(r'p7e_tick\[\d+\] = \{([^}]*)\}', src)
    ticks = [int(x) for x in tick.group(1).split(',')] if tick else [k + 1 for k in range(len(ops_))]

    def resolve(body, step):
        body = re.sub(r'\bp7e_(arena|args)\b', r'\1', body)
        for name, sreg in (('p7e_tx', 'threadIdx.x'), ('p7e_bx', 'blockIdx.x'), ('p7e_nt', 'blockDim.x'), ('p7e_nb', 'gridDim.x')):
            body = re.sub(r'\b' + name + r'\b', sreg, body)
        if 'p7e_tab' in body:
            return _normalize_shared(body, table, bases[step])
        derived = body
        for name, r in sorted(decls.rows.items(), key=lambda kv: -len(kv[1]['expr'])):
            derived = derived.replace('(' + r['expr'] + ')', f'PTR({r["ty"]},{r["base"]},{r["value"]})')
        derived = re.sub(r'//[^\n]*', '', derived)
        return P.TMAP.sub(lambda m: f'MAP({m.group(1)},{m.group(2)})', derived)

    # Each step's text, assigned to phases in order; a segment step holds whole phases separated by inline barriers.
    derived_of = {k: [] for k in range(len(phases))}
    expected_phase, inline_ok = 0, True
    seg_block = re.compile(r'// Phase (\d+): [^\n]*\n\{\n(.*?)\n\}(?:\ngrid\.sync\(\);\nif \(ticks && blockIdx\.x == 0 && '
                           r'threadIdx\.x == 0\) ticks\[(\d+)\] = stamp\(\);\n)?(?=\n// Phase |\Z)', re.S)
    for step, t in enumerate(ticks):
        body = cases[ops_[step]]
        if body.startswith('// Phase '):
            blocks = list(seg_block.finditer(body))
            if not blocks or '\n'.join(b.group(0) for b in blocks) != body:
                inline_ok = False
            for j, b in enumerate(blocks):
                k = int(b.group(1))
                derived_of[k].append(resolve(b.group(2), step))
                if j + 1 < len(blocks):
                    following = int(blocks[j + 1].group(1))
                    # Same phase (items of one phase): no barrier; next phase: its barrier and tick inline.
                    if not ((following == k and b.group(3) is None)
                            or (following == k + 1 and b.group(3) is not None and int(b.group(3)) == k + 1)):
                        inline_ok = False
            if blocks:
                expected_phase = int(blocks[-1].group(1)) + (1 if t else 0)
        else:
            derived_of[expected_phase].append(resolve(body, step))
            if t:
                expected_phase += 1
        if t and t != expected_phase:
            inline_ok = False
    rows['steps_cover_phases_in_order'] = inline_ok and expected_phase == len(phases)
    presync = re.search(r'p7e_sync\[\d+\] = \{([^}]*)\}', src)
    if presync:
        syncs = [int(x) for x in presync.group(1).split(',')]
        rows['presync_barriers_match'] = syncs == [0] + ticks[:-1] and src.count('ticks[p7e_s] = stamp()') == len(cases)
    mismatches = []
    for p in phases:
        k = p['index']
        if _alpha('\n'.join(derived_of[k])) != _alpha(_normalize_original(p['code'], decls)):
            mismatches.append(k)
    rows['phases_rederived'] = len(phases)
    rows['phase_mismatches'] = mismatches
    passed = all(v for key, v in rows.items() if key not in ('phases_rederived', 'phase_mismatches')) and not mismatches
    return {'passed': passed, 'checks': rows, 'stats': folded.manifest.get('p7e'),
            'parent_source_sha256': __import__('hashlib').sha256(parent.source.encode()).hexdigest()}


def check_refusal():
    """Unrecognized emission keeps the program unchanged (refusal recorded); strict mode raises."""
    from megakernel_lab.operation_compiler import p7e_codegen as P
    from megakernel_lab.operation_compiler.lower import GeneratedProgram
    program = GeneratedProgram('extern "C" __global__ void other(float* x) {}\n', {'inputs': [], 'header_hashes': {}})
    saved = (P.P7E_MODE, P.P7E_STRICT)
    try:
        P.P7E_MODE, P.P7E_STRICT = 'fold', False
        kept = P.apply(program)
        unchanged = kept.source == program.source and 'refused' in kept.manifest.get('p7e', {})
        P.P7E_STRICT = True
        try:
            P.apply(GeneratedProgram(program.source, {'inputs': [], 'header_hashes': {}}))
            strict = False
        except P.FoldRefusal:
            strict = True
    finally:
        P.P7E_MODE, P.P7E_STRICT = saved
    return {'passed': unchanged and strict, 'unchanged_with_refusal': unchanged, 'strict_raises': strict}


def run():
    results = {'refusal': check_refusal()}
    for name, graph in GRAPHS.items():
        if Path(graph).exists():
            for mode, options in (('fold', {}), ('items', {}), ('fold', {'P7E_PRESYNC': True}),
                                  ('items', {'P7E_PRESYNC': True}),
                                  ('fold', {'P7E_PRESYNC': True, 'P7E_NOSHARE': ('pattn::forward_item',)}),
                                  ('fold', {'P7E_OPAQUE': True}), ('items', {'P7E_OPAQUE': True}),
                                  ('fold', {'P7E_OPAQUE': True, 'P7E_PRESYNC': True}),
                                  ('fold', {'P7E_SHARE_ONLY': ('p5c::phase',), 'P7E_SEGMENTS': True}),
                                  ('items', {'P7E_SHARE_ONLY': ('p5c::phase',), 'P7E_SEGMENTS': True}),
                                  ('fold', {'P7E_OPAQUE': True, 'P7E_OPAQUE_SREG': True}),
                                  ('items', {'P7E_OPAQUE': True, 'P7E_OPAQUE_SREG': True}),
                                  ('fold', {'P7E_OUTLINE': True}), ('items', {'P7E_OUTLINE': True}),
                                  ('fold', {'P7E_OUTLINE': True, 'P7E_NOSHARE': (' elements',), 'P7E_NOSHARE_EXCEPT': ('p5c::',)}),
                                  ('fold', {'P7E_OUTLINE': True, 'P7E_NOSHARE_PURE': True})):
                label = ':'.join([name, mode] + [f'{k}={v}' for k, v in options.items()])
                results[label] = check_fold(graph, mode, options)
    return {'passed': bool(results) and all(r['passed'] for r in results.values()), 'cases': results}


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        GRAPHS = dict(arg.split('=', 1) for arg in sys.argv[1:])
    print(json.dumps(run(), indent=1)[:4000])
