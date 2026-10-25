"""LATCH-STEP-06: loop-composer support for deferred shared-weight gradients (ops/loop_defer_grads.py).

Narrow hooks used by ``loop_pipeline``: which reverse-body outputs remain accumulating parameter gradients, the
per-iteration reverse-tape buffers that receive the deferred cotangent operands, the tensor maps spanning every
iteration slot of each operand, and the single post-loop stage (``kernels/step06_dgrad.cuh``).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

HEADER = 'step06_dgrad.cuh'
DSUM8 = True    # LATCH-STEP-06: out-of-line ordered sum, 8 elements per thread, every load first (g-11: 6.007 vs 6.090 ms, bitwise)
BG = False      # LATCH-STEP-06: chunks of the ordered sum run in grid-barrier waits of the final reverse step
BG_CHUNK_QUADS = 512   # quads per chunk (2048 elements: two per thread of one CTA)
BG_SITES = 'attention'  # 'all' barriers of the final reverse step, or only those ending attention-backward levels
BG_INLINE = False       # barrier-wait chunk sums inlined at each site (no out-of-line call in the loops)
NST = 6
_BYTES = {'bfloat16': 2, 'float32': 4}


def info(metadata):
    return metadata.get('deferred_grads') or None


def kept_parameters(metadata, np_):
    dg = info(metadata)
    deferred = set(dg['params']) if dg else set()
    return [k for k in range(np_) if k not in deferred]


def header_hash():
    path = Path(__file__).parent / 'kernels' / HEADER
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header_closure():
    """{name: sha256} of every vendored header the deferred stage includes, transitively (LATCH-TRAIN-CORRECTNESS).

    step06_dgrad.cuh includes the P5C warpgroup body (p5c_wg.cuh and its own includes). A program records, writes and
    hashes only the headers named in its manifest, and runtime.build selects sm_90a from those names, so a loop whose
    regions use no warpgroup body must still record these: otherwise its build cannot find p5c_wg.cuh (and would
    target sm_90 for wgmma code)."""
    import re
    from .lower import _header_path
    seen, todo = {}, [HEADER]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        text = _header_path(name).read_bytes()
        seen[name] = hashlib.sha256(text).hexdigest()
        todo += [n for n in re.findall(r'#include\s+"([^"/]+)"', text.decode()) if n.endswith('.cuh')]
    return seen


def maps(dg, body_tape, reverse_tape, max_steps):
    """One 128-byte map per operand spanning all iteration slots (rows of consecutive slots are contiguous)."""
    out, index = [], []
    if dg.get('mode', 'batch') == 'tape':
        return out, index
    slots = max(1, max_steps)
    for e in dg['entries']:
        a_start, a_stride = body_tape[e['tape_index']]
        b_start, b_stride = reverse_tape[e['operand_index']]
        T, M, N = e['T'], e['M'], e['N']
        if 4 * a_stride != T * M * 2 or 4 * b_stride != T * N * 2:
            raise ValueError('deferred gradient operands need contiguous iteration slots')
        ids = []
        for start, inner, name in ((a_start, M, 'tape:' + e['tape']), (b_start, N, 'reverse_tape:' + e['operand'])):
            out.append({'root': f'step06_dgrad:{name}', 'storage': 'arena', 'word_offset': start, 'element_offset': 0,
                        'dims': [inner, slots * T], 'stride_bytes': inner * 2, 'box': [64, 64], 'dtype': 'bfloat16',
                        'swizzle': 128, 'oob_fill': 'zero'})
            ids.append(len(out) - 1)
        index.append(ids)
    return out, index


def _lit(x):
    return x if isinstance(x, str) else repr(float(x)) + 'f'


def _opt_body(opt):
    """C statements of one optimizer element update (generated elementwise form: a + (b * 1.0f))."""
    lines, env = [], {}
    for e in opt['exprs']:
        a, b = (env.get(x, x) if isinstance(x, str) else _lit(x) for x in (e['a'], e['b']))
        expr = f'({a} * {b})' if e['op'] == '*' else f'({a} {e["op"]} ({b} * 1.0f))'
        name = 'r_' + e['out']
        lines.append(f'const float {name} = {expr};')
        env[e['out']] = name
    lines.append(f'pn = {env[opt["outputs"]["p_new"]]}; mn = {env[opt["outputs"]["m_new"]]};')
    return ' '.join(lines)


def _opt_struct(bodies):
    if not bodies:
        return '', 'step06::NoOpt'
    cases = '\n'.join(f'    case {i}: {{ {b} }} break;' for i, b in enumerate(bodies))
    return ('struct step06_opt {\n  __device__ __forceinline__ static void apply(int c, float g, float p, float m, '
            'float& pn, float& mn) {\n    switch (c) {\n' + cases + '\n    default: pn = p; mn = m;\n    }\n  }\n};\n',
            'step06_opt')


def stage_tape(dg, gradient_offsets, reverse_tape, fuse=False):
    """MODE 'tape': (table + optimizer declarations, stage code, element count) of the ordered post-loop sum."""
    rows, bodies, total = [], [], 0
    for e in dg['entries']:
        opt = e.get('optimizer') if fuse else None
        code_index = -1
        if opt:
            body = _opt_body(opt)
            if body not in bodies:
                bodies.append(body)
            code_index = bodies.index(body)
        start, stride = reverse_tape[e['operand_index']]
        n = e['M'] * e['N']
        if n % 4 or stride % 2:
            raise ValueError('deferred gradient sums need 16-byte vectors')
        rows.append(f'{{{gradient_offsets[e["param"]]}LL, {start}LL, {stride}LL, {n // 4}LL, '
                    f'{1 if e.get("zero_init") else 0}, {code_index}, {opt["p_arg"] if opt else 0}, '
                    f'{opt["m_arg"] if opt else 0}}}')
        total += n
    opt_decl, opt_name = _opt_struct(bodies)
    dg['_opt_name'] = opt_name
    table = (f'__device__ const step06::DSum step06_ds_tab[{len(rows)}] = {{\n  ' + ',\n  '.join(rows) + '\n};\n'
             + opt_decl)
    entry = 'dsum_out' if DSUM8 else 'dsum'   # the 8-wide sum runs out of line (own registers)
    code = f'step06::{entry}<{len(rows)}, {opt_name}>(step06_ds_tab, count, args, arena);'
    return table, code, total


def stage(dg, gradient_offsets, map_index, map_base, fuse=False):
    """(table declaration, stage code) of the post-loop deferred gradient stage."""
    entries, tile0, bodies = [], 0, []
    T = {e['T'] for e in dg['entries']}
    if len(T) != 1:
        raise ValueError('deferred gradients need one iteration row count')
    (T,) = T
    for e, (ia, ib) in zip(dg['entries'], map_index):
        opt = e.get('optimizer') if fuse else None
        code_index = -1
        if opt:
            body = _opt_body(opt)
            if body not in bodies:
                bodies.append(body)
            code_index = bodies.index(body)
        entries.append(f'{{{map_base + ia}, {map_base + ib}, {gradient_offsets[e["param"]]}LL, {e["M"]}, {e["N"]}, {tile0}, '
                       f'{1 if e.get("zero_init") else 0}, {code_index}, '
                       f'{opt["p_arg"] if opt else 0}, {opt["m_arg"] if opt else 0}}}')
        tile0 += (e['M'] // 128) * (e['N'] // 128)
    n = len(entries)
    table = (f'__device__ const step06::DEnt step06_dg_tab[{n}] = {{\n  ' + ',\n  '.join(entries) + '\n};\n')
    opt = 'step06::NoOpt'
    if bodies:
        cases = '\n'.join(f'    case {i}: {{ {b} }} break;' for i, b in enumerate(bodies))
        table += ('struct step06_opt {\n  __device__ __forceinline__ static void apply(int c, float g, float p, float m, '
                  'float& pn, float& mn) {\n    switch (c) {\n' + cases + '\n    default: pn = p; mn = m;\n    }\n  }\n};\n')
        opt = 'step06_opt'
    code = ('{ extern __shared__ float smem[]; '
            f'step06::dgrad<{n}, {NST}, {opt}>(step06_dg_tab, {tile0}, {T}, p5c_maps_sh, arena, count, smem, args); }}')
    return table, code, tile0


def tiles_of(dg):
    return sum((e['M'] // 128) * (e['N'] // 128) for e in dg['entries'])


__all__ = ['info', 'kept_parameters', 'maps', 'stage', 'header_hash', 'header_closure', 'HEADER', 'tiles_of']


def bg_tables(dg, order):
    """(declaration, chunk count, first chunk index per entry) of the chunked ordered sum. `order`: entry indices in
    the order their final products complete (chunks of earlier entries come first)."""
    rows, first = [], {}
    for k in order:
        e = dg['entries'][k]
        quads = e['M'] * e['N'] // 4
        first[k] = len(rows)
        for q0 in range(0, quads, BG_CHUNK_QUADS):
            rows.append(f'{{{k}, {q0}, {min(quads, q0 + BG_CHUNK_QUADS)}}}')
    decl = (f'__device__ const step06::BgChunk step06_bg_tab[{len(rows)}] = {{\n  ' + ',\n  '.join(rows) + '\n};\n')
    return decl, len(rows), first


def bg_barrier(opt_name):
    """Counter barrier whose waiting CTAs sum ready chunks (helpers after the dgrad table)."""
    return (
        '__device__ __forceinline__ void latch_loop_barrier_bg(unsigned long long* counter, int ready, int count, '
        'float** args, float* arena) {\n'
        '  __syncthreads();\n'
        '  if (threadIdx.x == 0) {\n'
        '    latch_loop_bar_target += gridDim.x;\n'
        '    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(counter) : "memory");\n'
        '  }\n'
        '  if (ready > 0) {\n'
        '    __syncthreads();\n'
        f'    step06::bg_wait<{opt_name}>(counter, latch_loop_bar_target, ready, step06_ds_tab, step06_bg_tab, count, '
        'args, arena);\n'
        '  }\n'
        '  if (threadIdx.x == 0) {\n'
        '    while (s101h::ld_acq_u64(counter) < latch_loop_bar_target) {}\n'
        '  }\n'
        '  __syncthreads();\n'
        '}\n')
