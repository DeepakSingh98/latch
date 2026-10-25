"""P-OPS-E pipelined lowering adapted to the Stage 4 optimizer ABI.

Pinned donor: 578fa6ea14f12b04b85148c73cf6ef44415910e1,
research/latch_ops/ops/loop_pipeline.py. Copied source, no donor runtime import.
Integration adds named optimizer-state bindings, typed commits, static-connectivity
participation for AdamW, and per-runtime barrier storage. Conditional/recomputed/
diagnostic regions keep the existing ordered path in loop_lower.

The same PyTorch-derived regions as loop_lower.lower_loop, lowered as *staged*
regions: each call executes one dependency interval (no grid barrier inside).
This module owns only region control: which interval of which region instance
runs between two grid barriers. It never changes an operation, its operands or
the order of any accumulation into one buffer.

Every region instance (a "track") declares, per stage, which loop-level buffers
it reads and writes (parameters, momentum, explicit state, data, boundary and
tape slots, carry cotangents, running gradients, loop outputs). Tracks listed in
program order are merged level by level: a stage runs after its own previous
stage and after every earlier-track stage it conflicts with (read-after-write,
write-after-read or write-after-write on one buffer; an accumulation is both).
Region-local scratch is private to each concurrently live instance.

Straight-line blocks merge neighbouring regions (initialization with the first
forward prefix; the last forward suffix, terminal objective and first reverse
prefix; the last reverse suffix, initialization VJP, optimizer and final commit).
The runtime loops are software-pipelined without unrolling or specializing the
count: the forward body runs the suffix of iteration j with the prefix of
iteration j+1, and the reverse body runs the suffix of reverse step e with the
prefix of step e+1. Split points minimize the merged body's barrier levels.
Counts 0..max_steps run the same code; uniform device guards disable absent
iterations. Ideas: dX/dW co-phasing, vbase CTA rotation and one-invocation-late
dW issue in Latch latch-v0.6.0 s108g/rlt and p0d/rlt generators (hand-written
there; derived here from buffer dependencies).
"""
from __future__ import annotations

P9_INLINE = 'all'     # 'all' (parent) | 'wgmma': inline only stages with warpgroup-MMA products
# P9-B (from P9-A's P9_LOCAL_PTRS): each stage call re-derives its pointer tables from opaque copies of arena / args /
# the loop index inside the call's scope, so no block-wide table entry stays live across stages and grid barriers.
P9_LOCAL_PTRS = True
# P9-B (P9-A's P9_SREG_MACRO): special-register reads after the vendor headers are volatile PTX reads (no derived
# thread/block index is kept live across stages).
P9_SREG_MACRO = True
P9_OPQ_HELPER = ('#ifndef P9_OPQ_DEFINED\n#define P9_OPQ_DEFINED\n'
                 'template <class T> __device__ __forceinline__ T* p9_opq(T* p) { asm volatile("" : "+l"(p)); return p; }\n'
                 '__device__ __forceinline__ int p9_opqi(int v) { asm volatile("" : "+r"(v)); return v; }\n#endif')
P9_SREG_HELPER = r'''#ifndef P9_SREG_MACRO_DEFINED
#define P9_SREG_MACRO_DEFINED
struct p9_u3 { unsigned x, y, z; };
__device__ __forceinline__ p9_u3 p9_sr_tid() { p9_u3 r; asm volatile("mov.u32 %0, %%tid.x;" : "=r"(r.x)); r.y = 0u; r.z = 0u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_ctaid() { p9_u3 r; asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r.x)); r.y = 0u; r.z = 0u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_ntid() { p9_u3 r; asm volatile("mov.u32 %0, %%ntid.x;" : "=r"(r.x)); r.y = 1u; r.z = 1u; return r; }
__device__ __forceinline__ p9_u3 p9_sr_nctaid() { p9_u3 r; asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r.x)); r.y = 1u; r.z = 1u; return r; }
#define threadIdx p9_sr_tid()
#define blockIdx p9_sr_ctaid()
#define blockDim p9_sr_ntid()
#define gridDim p9_sr_nctaid()
#endif'''


def _local_tables(track, t, var, scratch):
    """P9-B: scope-local pointer tables of one stage call (P9_LOCAL_PTRS)."""
    import re as _re
    def opq(expr):
        expr = _re.sub(r'\bargs\[', 'p9_ag[', expr)
        expr = _re.sub(r'\barena\b', 'p9_ar', expr)
        if var:
            expr = _re.sub(r'\b' + _re.escape(var) + r'\b', 'p9_iv', expr)
        return expr
    head = 'float* p9_ar = p9_opq(arena); float** p9_ag = p9_opq(args); '
    if var:
        head += f'const int p9_iv = p9_opqi({var}); '
    return (head + f'float* a{t}[] = {{' + ', '.join(opq(ptr) for _, ptr in track.args) + '}; '
            + f'float* o{t}[] = {{' + ', '.join(opq(ptr) for _, ptr in track.outs) + '}; '
            + f'float* x{t} = {opq(scratch)}; ')

import hashlib
import json
import math

from .lower import LoweringError, TensorLayout, _BYTES, _CTYPES, _index, _translation_unit


# Adapted from the donor's S102G/bar2 release/acquire barrier. The arrival
# counter belongs to this Runtime's arena, not a module-global device symbol.
# Its sole initialization barrier also makes concurrent independent runtimes safe.
COUNTER_BARRIER = r'''
#ifndef LATCH_OPS_REGION_COUNTER_BARRIER
#define LATCH_OPS_REGION_COUNTER_BARRIER
__shared__ unsigned long long latch_loop_bar_target;
__device__ __forceinline__ void latch_loop_bar_begin(unsigned long long* counter) {
  if (blockIdx.x == 0 && threadIdx.x == 0) *counter = 0;
  cg::this_grid().sync();
  if (threadIdx.x == 0) latch_loop_bar_target = 0;
  __syncthreads();
}
__device__ __forceinline__ void latch_loop_barrier(unsigned long long* counter) {
  __syncthreads();
  if (threadIdx.x == 0) {
    latch_loop_bar_target += gridDim.x;
    asm volatile("red.release.gpu.global.add.u64 [%0], 1;" :: "l"(counter) : "memory");
    while (s101h::ld_acq_u64(counter) < latch_loop_bar_target) {}
  }
  __syncthreads();
}
#endif
'''



PELEM_MAX_WORK = 4096        # P9-C: partitioned stages carry at most 4096 x 256 elements of work
COPY_PARTITION = True   # P9-C: copy stages split over disjoint CTA ranges with vector copies
_BYTES_OF = {'float32': 4, 'bfloat16': 2, 'int64': 8, 'int32': 4, 'bool': 1, 'uint8': 1, 'float64': 8}

def _participation_code(connectivity, parameters, carries, pointers):
    """Device-only boolean reverse propagation for a nonconditional body.

    Connectivity is structural, not inferred from numerical gradient values.
    Count can change at every replay. No tensor/model/RNG value is consumed.
    """
    if not connectivity:
        return ''
    if isinstance(connectivity['body'], dict):
        raise LoweringError('conditional gradient participation requires ordered replay')
    np_, nc = len(parameters), len(carries)
    if (len(connectivity['terminal']) != np_ + nc or len(connectivity['body']) != nc
            or any(len(row) != np_ + nc for row in connectivity['body'])
            or len(connectivity['init']) != nc or any(len(row) != np_ for row in connectivity['init'])):
        raise LoweringError('loop gradient connectivity ABI mismatch')
    literal = lambda value: str(bool(value)).lower()
    lines = ['if (blockIdx.x == 0 && threadIdx.x == 0) {',
             f'bool active_p[{max(1, np_)}] = {{' + ', '.join(map(literal, connectivity['terminal'][:np_])) + '};',
             f'bool active_c[{max(1, nc)}] = {{' + ', '.join(map(literal, connectivity['terminal'][np_:])) + '};',
             f'bool next_c[{max(1,nc)}];', 'for (int reverse = count - 1; reverse >= 0; --reverse) {']
    def edges(rows, column):
        return ' || '.join(f'active_c[{i}]' for i,row in enumerate(rows) if row[column]) or 'false'
    lines += [f'active_p[{i}] |= ({edges(connectivity["body"], i)});' for i in range(np_)]
    lines += [f'next_c[{i}] = ({edges(connectivity["body"], np_+i)});' for i in range(nc)]
    lines += [f'active_c[{i}] = next_c[{i}];' for i in range(nc)]
    lines += ['}']
    lines += [f'*reinterpret_cast<bool*>({pointer}) = active_p[{i}] || ({edges(connectivity["init"], i)});'
              for i,pointer in enumerate(pointers)]
    return '\n'.join(lines + ['}'])


def _form(cn=0, cv=0, c0=0):
    return (cn, cv, c0)


def _expr(form, var):
    cn, cv, c0 = form
    terms = []
    if cn:
        terms.append('count' if cn == 1 else f'({cn})*count')
    if cv:
        terms.append(var if cv == 1 else f'({cv})*{var}')
    if c0 or not terms:
        terms.append(str(c0))
    return '(' + ' + '.join(terms) + ')'


def _parity(form):
    return tuple(x % 2 for x in form)


class _Track:
    def __init__(self, label, region, stages, args, outs, scratch, guard=None, copy_code=None, copy_rw=None):
        self.label, self.region, self.stages = label, region, list(stages)
        self.args, self.outs = args, outs          # [(key, pointer expression)]
        self.scratch, self.guard = scratch, guard
        self.copy_code, self.copy_rw = copy_code, copy_rw


# P-OPS2-D: product-engine selection for one-resident-CTA staged programs. One
# CTA per SM leaves shared memory for deeper rings with 32-deep stages; separate
# (noinline) product bodies get their own register allocation instead of
# spilling inside the inlined kernel; accumulating epilogues and a reduced
# grid share for accumulate-only products enable their slack placement.
# P9-B: a parameter gradient zero-filled by the terminal VJP whose only reverse-body writer is one FP32-accumulating
# product skips the zero fill when count > 0; that product stores (instead of adding to +0) in reverse step 0.
P9_ACC_FIRST = False
# LATCH-STEP-06: warp-per-row phases (one warp walks rows o, o + warps, ...) L2-prefetch the next row's
# row-contiguous operands while they compute the current row. Cache hint only: loads, values and order unchanged.
STEP06_ROW_PF = False
# LATCH-STEP-06: loop-invariant attention mask tables. The shared mask-tile table of an immutable mask input is
# rebuilt by the first forward attention of every body iteration (PRE 0) and by every backward attention call (own
# slot, PRE 0). When the mask is one launch input, the table is built once at kernel entry and every forward and
# backward call reads it (PRE 1). Same table contents, same item arithmetic.
STEP06_MASK_HOIST = True    # g-10: 6.064 vs fo2 6.173 ms (bitwise 12/12)
# LATCH-STEP-06: hoisted backward calls run backward_phase6t (call-epoch publication tags and exact completion targets
# instead of the in-phase flag clear and grid synchronization; kernels/attention_p9.cuh).
STEP06_BWD_TAGS = False


def _hoist_mask_tables(helpers, functions, body):
    """(helpers, body, record): hoist the mask-tile table builds of P9 attention wrappers to kernel entry."""
    import re
    def split_top(text):
        parts, depth, cur = [], 0, ''
        for ch in text:
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            if ch == ',' and depth == 0:
                parts.append(cur.strip())
                cur = ''
            else:
                cur += ch
        if cur.strip():
            parts.append(cur.strip())
        return parts

    kernel = '\n'.join(body)

    def kernel_mask(name, params):
        """Kernel-scope expressions of the mask operand over every call site of wrapper `name`."""
        index = [p.split()[-1] for p in split_top(params)].index('p_mask')
        exprs = set()
        for fn in functions:
            fname = re.match(r'__device__ __\w+__ void (\w+)\(', fn)
            for call in re.finditer(re.escape(name) + r'\(([^;]*)\);', fn):
                var = split_top(call.group(1))[index]
                decl = re.search(r'(?:const )?float\* ' + re.escape(var) + r' = args\[(\d+)\];', fn)
                if not fname or not decl:
                    return None
                region_index = int(decl.group(1))
                sites = list(re.finditer(re.escape(fname.group(1)) + r'\((a\d+), ', kernel))
                if not sites:
                    return None
                for site in sites:
                    table = kernel.rfind(f'float* {site.group(1)}[] = {{', 0, site.start())
                    if table < 0:
                        return None
                    start = table + len(f'float* {site.group(1)}[] = {{')
                    depth, j = 1, start
                    while depth:
                        depth += {'{': 1, '}': -1}.get(kernel[j], 0)
                        j += 1
                    items = split_top(kernel[start:j - 1])
                    if region_index >= len(items):
                        return None
                    exprs.add(items[region_index])
        return exprs

    def signature(fields):
        # The fields mtab_build reads: mask operand and strides (A.mask/m_b/m_i/m_j/has_mask) and the sequence A.t.
        return tuple(sorted(line.strip() for line in fields.split('\n')
                            if ('A.mask = ' in line or re.search(r'\bA\.t = ', line))
                            and set(re.findall(r'\bp_\w+', line)) <= {'p_mask'}))

    text = '\n'.join(helpers)
    found = []
    for h in re.finditer(r'__device__ __noinline__ void (p9w_(forward|backward)_\w+)\(([^)]*)\) \{\n', text):
        end = text.index('\n}\n', h.end())
        inner = text[h.end():end]
        call = re.search(r'(p9att::(forward_phase4|backward_phase6)<0>\(([^;]*)\);)\s*$', inner)
        if call and inner.startswith('pshort::Args A{};\n'):
            found.append({'name': h.group(1), 'which': h.group(2), 'params': h.group(3), 'whole': text[h.start():end + 3],
                          'fields': inner[len('pshort::Args A{};\n'):call.start()], 'call': call.group(1),
                          'args': split_top(call.group(3))})
    builders = [w for w in found if w['which'] == 'forward' and w['args'][-1] != '-1']
    record = []
    for b in builders:
        slot = b['args'][-1]
        expr = kernel_mask(b['name'], b['params'])
        if not expr or len(expr) != 1 or not re.fullmatch(r'p9_ag\[\d+\]', next(iter(expr))):
            continue
        (expr,) = expr
        sig = signature(b['fields'])
        users = [b] + [w for w in found if w['which'] == 'backward' and w['args'][-1] == '-1'
                       and signature(w['fields']) == sig and kernel_mask(w['name'], w['params']) == {expr}]
        for w in users:
            if w['which'] == 'forward':
                new = w['call'].replace('forward_phase4<0>(', 'forward_phase4<1>(', 1)
            else:
                phase_ = 'backward_phase6t' if STEP06_BWD_TAGS else 'backward_phase6'
                new = f'p9att::{phase_}<1>({", ".join(w["args"][:-1] + [slot])});'
            text = text.replace(w['whole'], w['whole'].replace(w['call'], new), 1)
        fields = '\n'.join(sig)
        text += (f'\n__device__ __noinline__ void p9w_mtab_{slot}(const float* p_mask) {{\npshort::Args A{{}};\n{fields}\n'
                 f'plong::mtab_build(A, {slot}, 3);\n}}\n')
        body = [body[0], f'{{ float** p9_ag = p9_opq(args); p9w_mtab_{slot}({expr}); }}', *body[1:]]
        record.append({'slot': int(slot), 'mask': expr, 'forward': b['name'], 'backward': [w['name'] for w in users[1:]]})
    return [text], body, record


def _row_prefetch(text):
    """Insert, at the top of every warp-per-row loop of a stage function, one bulk L2 prefetch per row-contiguous
    operand (``v_x[(i)]`` with ``i = o * W + 32 * w + lane``) of the warp's next row."""
    import re
    head = re.compile(r'for \(long long o = \(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5; '
                      r'o < (\d+)LL; o \+= \(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \{')
    sizes = {name: 4 if kind == 'float' else 2
             for kind, name in re.findall(r'(float|__nv_bfloat16)\* (v_\w+) = ', text)}
    def block(start):
        depth, j = 1, start
        while depth and j < len(text):
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            j += 1
        return text[start:j]

    out, pos = [], 0
    for m in head.finditer(text):
        loop = block(m.end())
        widths, names = set(), []
        # Element reads only: inside `for (int w ...)` blocks that declare i = o * W + 32 * w + lane (per-row
        # scalars such as rsqrt are read with i = o outside them and are never prefetched as rows).
        for wl in re.finditer(r'for \(int w = 0; w < \d+; \+\+w\) \{', loop):
            body = block(m.end() + wl.end())
            decl = re.match(r'\s*const long long i = o \* (\d+)LL \+ 32 \* w \+ lane;', body)
            if not decl:
                continue
            widths.add(int(decl.group(1)))
            names += re.findall(r'\b(v_\w+)\[\(i\)\]', body)
        if len(widths) != 1:
            continue
        (w,) = widths
        names = [n for n in dict.fromkeys(names) if n in sizes]
        calls = [f'asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"({n} + on_ * {w}LL), '
                 f'"r"({w * sizes[n]}u) : "memory");' for n in names if (w * sizes[n]) % 16 == 0]
        if not calls:
            continue
        out.append(text[pos:m.end()])
        out.append(f'\n  if ((threadIdx.x & 31) == 0) {{ const long long on_ = o + (((long long)gridDim.x * blockDim.x) >> 5); '
                   f'if (on_ < {m.group(1)}LL) {{ ' + ' '.join(calls) + ' } }')
        pos = m.end()
    return ''.join(out) + text[pos:]


def _acc_first(functions, function_region, reverse_region, terminal_region='terminal_vjp'):
    """P9-B: (covered gradient indices, rewritten functions). Gradient k is out[k] of the reverse body and out[1 + k]
    of the terminal VJP. Text-level and conservative: any other reference keeps the gradient uncovered."""
    import re
    decl = re.compile(r'float\* (v_\w+) = out\[(\d+)\];')
    prod = re.compile(r'(p5c::Args g\{[^;]*?, (v_\w+), \d+LL, \d+, \d+, \d+\};)(.{0,400}?)p5c::phase(?:_epi)?<\d+,\d+,\d+,'
                      r'\w+,\w+,\d+,(\d+),', re.S)
    rev = [i for i, r in function_region.items() if r == reverse_region]
    term = [i for i, r in function_region.items() if r == terminal_region]
    if not rev or not term:
        return set(), functions
    writers, others = {}, set()
    for i in rev:
        text = functions[i]
        for fn in re.split(r'(?=__device__ __forceinline__ void loop_)', text):
            names = {v: int(k) for v, k in decl.findall(fn)}
            for v, k in names.items():
                uses = len(re.findall(r'\b' + re.escape(v) + r'\b', fn)) - 1
                accs = [m for m in prod.finditer(fn) if m.group(2) == v and m.group(4) == '2']
                if uses == 0:
                    continue
                if uses != len(accs):
                    others.add(k)
                writers[k] = writers.get(k, 0) + len(accs)
    candidates = {k for k, n in writers.items() if n == 1 and k not in others}
    zero_store = 'reinterpret_cast<float4*>({v})[q_] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);'
    covered = set()
    for i in term:
        text = functions[i]
        for fn in re.split(r'(?=__device__ __forceinline__ void loop_)', text):
            for v, j in decl.findall(fn):
                k = int(j) - 1
                if k not in candidates:
                    continue
                uses = len(re.findall(r'\b' + re.escape(v) + r'\b', fn)) - 1
                if uses == 1 and zero_store.format(v=v) in fn:
                    covered.add(k)
    if not covered:
        return set(), functions
    out = list(functions)
    for i in term:
        text = out[i]
        parts = re.split(r'(?=__device__ __forceinline__ void loop_)', text)
        for n_, fn in enumerate(parts):
            for v, j in decl.findall(fn):
                if int(j) - 1 in covered:
                    fn = fn.replace(zero_store.format(v=v), 'if (!p5c::p5c_skipzero) ' + zero_store.format(v=v))
            parts[n_] = fn
        out[i] = ''.join(parts)
    for i in rev:
        parts = re.split(r'(?=__device__ __forceinline__ void loop_)', out[i])
        for n_, fn in enumerate(parts):
            names = {v: int(k) for v, k in decl.findall(fn)}
            def mark(m):
                if m.group(4) == '2' and names.get(m.group(2)) in covered:
                    return m.group(1) + ' g.acc_first = 1;' + m.group(3) + m.group(0)[len(m.group(1)) + len(m.group(3)):]
                return m.group(0)
            parts[n_] = prod.sub(mark, fn)
        out[i] = ''.join(parts)
    return covered, out


PIPELINED_MATRIX_ENGINE = {'stages': 4, 'stage_depths': [16, 32], 'inline': False, 'accumulate': True,
                           'deferred_grid': 96, 'selector': 'auto', 'bodies': ['simt', 'popsb']}


def lower_pipelined(graph, *, retain_all, fuse_elementwise, options):
    from .group_rules import features_of
    if 'lptx' not in features_of(options.get('group_rules')):
        return _lower_pipelined_once(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise, options=options)
    # P6-C lptx: a first lowering fixes the level schedule (which stages share a level); the loop-body regions are
    # then lowered again with each stage's level context, so their products are planned around that work.
    first = _lower_pipelined_once(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise, options=options)
    hints = _lptx_hints(first.manifest)
    program = _lower_pipelined_once(graph, retain_all=retain_all, fuse_elementwise=fuse_elementwise, options=options,
                                    hints_by_region=hints)
    if [e['members'] for e in program.manifest['lpt_log']] != [e['members'] for e in first.manifest['lpt_log']] and False:
        raise LoweringError('lptx: level schedule changed between passes')
    return program


def _lptx_hints(manifest):
    """{region: {stage: {'load': [128 us], 'peers': [stage]}}} for the loop-body regions from a first lowering."""
    regions = manifest['regions']
    body = {manifest['splits'].get('forward_region'), manifest['splits'].get('reverse_region')} - {None}
    occurrences = {}
    for entry in manifest['lpt_log']:
        np_load, products = [0.0] * 128, {}
        for region, stage, rot in entry['members']:
            stages = regions.get(region, {}).get('stages', []) if region else []
            if stage >= len(stages):
                continue
            for footprint, cost in stages[stage].get('np_load', []):
                for c in range(min(128, footprint)):
                    np_load[c] += cost
            load = stages[stage].get('gemm_load')
            if load:
                products[(region, stage)] = [load[(c - rot) % 128] for c in range(128)]
        for region, stage, rot in entry['members']:
            if region not in body:
                continue
            load = list(np_load)
            for (r2, _), vec in products.items():
                if r2 != region:
                    load = [a + b for a, b in zip(load, vec)]
            peers = [s2 for r2, s2, _ in entry['members'] if r2 == region and s2 != stage]
            occurrences.setdefault((region, stage), []).append((entry['block'], load, peers))
    hints = {}
    for (region, stage), found in occurrences.items():
        loop = [x for x in found if x[0] in ('forward', 'reverse')] or found
        hints.setdefault(region, {})[stage] = {'load': [round(max(x[1][c] for x in loop), 3) for c in range(128)],
                                               'peers': sorted({p for x in loop for p in x[2]})}
    return hints


def _lower_pipelined_once(graph, *, retain_all, fuse_elementwise, options, hints_by_region=None):
    from .lower import lower, _LPT_HINTS

    one_cta = (options['launch_bounds'] or options['residency'] == 'one'
               or (options['residency'] == 'auto' and options['inline_stages']))
    region_options = dict(options['region_options'])
    if one_cta and 'matrix_engine' not in region_options:
        region_options['matrix_engine'] = dict(PIPELINED_MATRIX_ENGINE)
    elif region_options.get('matrix_engine') == {}:
        region_options.pop('matrix_engine')
    options = {**options, 'region_options': region_options, 'one_cta': one_cta}

    if graph.storage_mode == 'recompute_boundaries':
        raise LoweringError('pipelined loop lowering requires saved boundaries')
    if retain_all:
        raise LoweringError('pipelined loop lowering keeps only stable outputs; use the diagnostic path')
    saved_body = graph.storage_mode == 'save_body_tape'
    forward_region = 'body_saved' if saved_body else 'body'
    reverse_region = 'body_vjp_saved' if saved_body else 'body_vjp'
    init_saved = getattr(graph, 'init_saved', None) is not None
    init_name, init_vjp_name = ('init_saved', 'init_vjp_saved') if init_saved else ('init', 'init_vjp')
    region_names = (init_name, forward_region, 'terminal_vjp', reverse_region, init_vjp_name, 'optimizer')
    metadata = graph.metadata
    casts = metadata.get('invariant_casts', [])
    if casts:
        region_names = ('invariant_casts', *region_names)
    parameters, carries, states = (tuple(metadata[k]) for k in ('parameter_names', 'carry_names', 'state_names'))
    optimizer_names = tuple(metadata.get('optimizer_state_names', parameters))
    optimizer_group = 'optimizer_state' if 'optimizer_state' in graph.outputs else 'momentum'
    np_, no, nc, ns = len(parameters), len(optimizer_names), len(carries), len(states)
    inputs, outputs = list(graph.inputs), list(graph.output_names())
    ni = len(inputs) - np_ - no - ns - 1
    values = graph.values
    layouts = {}
    for name in inputs + outputs:
        value = values[name]
        layouts[name] = {'shape': list(value.shape), 'strides': list(value.strides), 'root': name, 'offset': 0,
                         'dtype': str(value.dtype).removeprefix('torch.')}
        if layouts[name]['dtype'] not in _CTYPES:
            raise LoweringError(f'loop value {name}: unsupported {layouts[name]["dtype"]}')
    for name in inputs[:np_+no+ns]:
        if not TensorLayout(tuple(values[name].shape), tuple(values[name].strides), name).contiguous:
            raise LoweringError(f'loop mutable binding {name} must be contiguous before final commit')
    if layouts[inputs[-1]]['dtype'] != 'int64' or math.prod(layouts[inputs[-1]]['shape']) != 1:
        raise LoweringError('loop runtime count must be one int64 element')
    structured = graph.outputs

    from .group_rules import features_of

    def lower_region(name, bindings, matrix_grids=None):
        program = lower_region_once(name, bindings, matrix_grids)
        if options['joint_matrix'] and matrix_grids is None:
            # P-OPS2-D joint selection: products sharing a dependency interval
            # split the grid, so each is planned for its share of resident CTAs.
            manifest = program.manifest
            phases = manifest['phases']
            grids = {}
            for members in manifest['barrier_intervals']:
                products = [phases[i].get('product', phases[i]['names'][0]) for i in members
                            if phases[i].get('engine') in ('popsb_cpasync', 'pops2_fp32')]
                for product in products:
                    if len(products) > 1:
                        grids[product] = max(16, 128 // len(products))
            if grids:
                program = lower_region_once(name, bindings, grids)
        return program

    def lower_region_once(name, bindings, matrix_grids):
        # Straight-line regions outside the loops may expose one stage per phase.
        granularity = 'phase' if options['phase_stages'] and name in ('init', 'terminal_vjp', 'init_vjp', 'optimizer', 'init_saved', 'init_vjp_saved') else True
        token = _LPT_HINTS.set((hints_by_region or {}).get(name))
        try:
            return _lower_region_call(name, bindings, matrix_grids, granularity)
        finally:
            _LPT_HINTS.reset(token)

    def _lower_region_call(name, bindings, matrix_grids, granularity):
        from .matrix_rules import FP32_EDGE_SCOPE
        edge_token = FP32_EDGE_SCOPE.set('region')     # LATCH-RESEARCH-EDITS: masked FP32 edge tiles in regions
        try:
            return _lower_region(name, bindings, matrix_grids, granularity)
        finally:
            FP32_EDGE_SCOPE.reset(edge_token)

    def _lower_region(name, bindings, matrix_grids, granularity):
        return lower(getattr(graph, name), retain_all=False, fuse_elementwise=fuse_elementwise,
                     **({'matrix_grids': matrix_grids} if matrix_grids else {}),
                     output_bindings=bindings, region_body=True, barrier_schedule='dependency',
                     elide_stores=bool(options['elide_stores']), staged=granularity,
                     partition_phases=bool(options['partition_ctas']) or (
                         name in ('optimizer', 'init_vjp') and 'part' in features_of(options.get('group_rules'))
                         and (name == 'optimizer' or 'alapv' in features_of(options.get('group_rules')))) or (
                         # P6-C partc: the once-per-launch weight casts run concurrently on disjoint CTA ranges.
                         name == 'invariant_casts' and 'partc' in features_of(options.get('group_rules'))),
                     policy=options['region_policy'],
                     **({'split_counters': 'persistent'} if options['persistent_counters'] else {}),
                     **({'row_fusion': True} if options['row_fusion'] else {}),
                     **({'warp_softmax': True} if options['warp_softmax'] else {}),
                     **({'batched_reduction_loads': True} if options['batched_reduction_loads'] else {}),
                     **({'defer_accumulate': True, 'defer_policy': options['defer_policy']}
                        if options['defer_accumulate'] else {}),
                     **({'epilogue_fusion': True} if options['epilogue_fusion'] else {}),
                     **({'write_through': True} if options['write_through'] else {}),
                     **({'copy_bypass': True} if options['copy_bypass'] else {}),
                     **({'group_rules': region_rules(name)} if options['group_rules'] else {}),
                     **options['region_options'])

    def region_rules(name):
        # P8-C: with sequence inputs the per-iteration body regions read slice t of a sequence buffer, an operand
        # address the tensor-map table cannot enumerate; only those regions drop the tensor-map product rules.
        rules = options['group_rules']
        if graph.metadata.get('sequence_specs') and name in ('body', 'body_vjp', 'body_saved', 'body_vjp_saved'):
            kept = [f for f in str(rules).split('+') if f not in ('p5c', 'p5cw', 'p5ce', 'p5ck')]
            return '+'.join(kept) if kept else False
        return rules

    if options['group_rules']:
        # P-OPS5-B rules address BF16 conversion chains: programs without BF16 values lower unchanged.
        def has_bf16(region):
            values = region.values.values() if isinstance(region.values, dict) else region.values
            return any(str(getattr(v, 'dtype', '')).endswith('bfloat16') for v in values)
        if not any(has_bf16(getattr(graph, name)) for name in region_names if hasattr(getattr(graph, name), 'values')):
            options = {**options, 'group_rules': False}
    # LATCH-STEP-06: deferred shared-weight gradients leave the reverse body's accumulating outputs.
    from . import loop_dgrad
    dgrad = loop_dgrad.info(metadata) if saved_body else None
    kept_grads = loop_dgrad.kept_parameters(metadata, np_)
    n_acc = {reverse_region: len(kept_grads)}
    programs = {}
    for name in region_names:
        count = len(getattr(graph, name).outputs)
        if name in (reverse_region, init_vjp_name):
            bindings = ['accumulate'] * n_acc.get(name, np_) + ['store'] * (count - n_acc.get(name, np_))
        else:
            bindings = ['store'] * count
        zero_init = [e for e in (dgrad['entries'] if dgrad else []) if e.get('zero_init')]
        if name == 'terminal_vjp' and zero_init:
            # LATCH-STEP-06: the +0 fills of +0-initialized deferred gradients are elided (zero accumulate);
            # the deferred stage stores those gradients. Refuse the elision unless lower() omitted every fill.
            for e in zero_init:
                bindings[1 + e['param']] = 'accumulate'
            program = lower_region(name, bindings)
            placement = program.manifest.get('output_placement', {})
            outs = program.manifest['outputs']
            elided = {e['param'] for e in zero_init
                      if any(v.get('mode') == 'accumulate' and v.get('zero_omitted') and int(v['output']) == 1 + e['param']
                             for v in placement.values())}
            if elided != {e['param'] for e in zero_init}:
                for e in zero_init:
                    e['zero_init'] = False
                program = lower_region(name, ['store'] * count)
            programs[name] = program
            continue
        programs[name] = lower_region(name, bindings)
        if programs[name].manifest['state_writes'] or programs[name].functions:
            raise LoweringError('staged loop regions must be pure flat operation regions')
    inplace = False
    if options['inplace_commit']:
        # The optimizer's new parameters/momentum overwrite their inputs only when
        # lower() proves every other read of an input precedes its overwrite.
        try:
            programs['optimizer'] = lower_region('optimizer', [f'inplace:{i}' for i in range(np_+no)])
            inplace = len(programs['optimizer'].manifest['placed_outputs']) == np_+no
            if not inplace:
                programs['optimizer'] = lower_region('optimizer', ['store'] * (np_+no))
        except LoweringError:
            programs['optimizer'] = lower_region('optimizer', ['store'] * (np_+no))

    # ---------------------------------------------------------------- storage
    align = lambda size: (size + 63) // 64 * 64
    words = {name: align(p.manifest['arena_words']) for name, p in programs.items()}
    x_words = max(words[forward_region], words[reverse_region])
    y_words = max(words[init_name], words['terminal_vjp'], words[init_vjp_name])
    top = 0

    def take(count):
        nonlocal top
        start = top
        top += count
        return start
    X = [take(x_words), take(x_words)]
    Y = take(y_words)
    Z = X[0] if words['optimizer'] <= x_words else take(align(words['optimizer']))
    # The once-per-launch cast region may share Block A with init; private scratch.
    W_scratch = take(words['invariant_casts']) if casts else None
    scratch_words = top
    offsets = {}

    def allocate(desc, count=1):
        size = align((math.prod(desc['shape']) * _BYTES[desc['dtype']] + 3) // 4)
        return take(size * count), size

    p = [f'args[{i}]' for i in range(np_)]
    m = [f'args[{np_+i}]' for i in range(no)]
    s = [f'args[{np_+no+i}]' for i in range(ns)]
    data = [f'args[{np_+no+ns+i}]' for i in range(ni)]
    for name in outputs:
        role = values[name].output_roles[0] if values[name].output_roles else ''
        if inplace and role.split(':')[0] in ('params', optimizer_group):
            group, key = role.split(':', 1)
            source = inputs[parameters.index(key) if group == 'params' else np_ + optimizer_names.index(key)]
            layouts[name] = {**layouts[name], 'root': source}   # committed state is the returned value
            continue
        offsets[name], _ = allocate(layouts[name])
    ptr = lambda name: f'arena + {offsets[name]}LL'
    loss_ptr = ptr(structured['loss'])
    gp = [ptr(structured['grads'][k]) for k in parameters]
    pp = None if inplace else [ptr(structured['params'][k]) for k in parameters]
    mp = None if inplace else [ptr(structured[optimizer_group][k]) for k in optimizer_names]
    sp = [ptr(structured['state'][k]) for k in states]
    cp = [ptr(structured['carry'][k]) for k in carries]
    carry_desc = [layouts[structured['carry'][k]] for k in carries]
    state_desc = [layouts[structured['state'][k]] for k in states]
    param_desc = [layouts[structured['grads'][k]] for k in parameters]
    optimizer_desc = [layouts[structured[optimizer_group][k]] for k in optimizer_names]
    # Parameter gradients accumulate across iterations in FP32. Carry (and its
    # cotangent, which PyTorch's VJP returns in the carry dtype) may be FP32 or
    # BF16; storage and copies follow the declared dtype (C-OPS2-A).
    if any(d['dtype'] != 'float32' for d in param_desc):
        raise LoweringError('loop parameter-gradient accumulation requires FP32')
    if any(d['dtype'] not in ('float32', 'bfloat16') for d in carry_desc):
        raise LoweringError('loop differentiable carry must be FP32 or BF16')
    cot = [[f'arena + {allocate(d)[0]}LL' for d in carry_desc] for _ in range(2)]
    # P-OPS4-F: one buffer per update-invariant cast, written by the first block
    # of every launch and only read afterwards (never by the optimizer/commit).
    cast_desc = [{'shape': list(d['shape']), 'dtype': d['dtype']} for d in casts]
    cast_buffers = [f'arena + {allocate(d)[0]}LL' for d in cast_desc]
    tape_size = graph.max_steps + 1
    boundaries = [allocate(d, tape_size) for d in carry_desc + state_desc]
    tape_desc = metadata.get('body_tape', []) if saved_body else []
    body_tape = [allocate(d, max(1, graph.max_steps)) for d in tape_desc]
    # LATCH-STEP-06: per-iteration BF16 cotangent operands of the deferred gradients (reverse tape).
    reverse_tape = [allocate(d, max(1, graph.max_steps)) for d in (dgrad['reverse_tape'] if dgrad else [])]
    # C-OPS4-A sequence inputs: full buffers written by init; reverse step r writes cotangent slice r.
    seq_specs = metadata.get('sequence_specs', [])
    seq_desc = [{'shape': list(d['shape']), 'dtype': d['dtype']} for d in seq_specs]
    for d in seq_desc:
        if (math.prod(d['shape'][1:]) * _BYTES[d['dtype']]) % 16:
            raise LoweringError('sequence slices must be 16-byte multiples')
    seq_full = [f'arena + {allocate(d)[0]}LL' for d in seq_desc]
    init_tape = [f'arena + {allocate(d)[0]}LL' for d in metadata.get('init_tape', [])] if init_saved else []
    seq_diff = [i for i, d in enumerate(seq_specs) if d['differentiable']]
    seq_cot = [f'arena + {allocate(seq_desc[i])[0]}LL' for i in seq_diff]
    slice_words = [math.prod(d['shape'][1:]) * _BYTES[d['dtype']] // 4 for d in seq_desc]
    connectivity = metadata.get('gradient_connectivity')
    active = [f'arena + {allocate({"shape": [], "dtype": "bool"})[0]}LL' for _ in parameters] if connectivity else []
    barrier_offset = take(align(2)) if options['barrier'] == 'counter' else None
    # P-OPS2-D persistent split-K counters: one never-reused area per region,
    # zeroed once per launch before the first grid barrier. Each product's last
    # arriver restores its counters to zero, so successive instances (which never
    # run the same stage concurrently) reuse them without a producer phase.
    counter_area = {name: take(align(program.manifest['persistent_counter_words']))
                    for name, program in programs.items() if program.manifest.get('persistent_counter_words')}

    # ------------------------------------------------- keys and pointer helpers
    def bkey(k, form, var):
        start, stride = boundaries[k]
        return (('b', k, form), f'arena + {start}LL + {_expr(form, var)} * {stride}LL')

    def tkey(k, form, var):
        start, stride = body_tape[k]
        return (('t', k, form), f'arena + {start}LL + {_expr(form, var)} * {stride}LL')

    def rtkey(k, form, var):
        start, stride = reverse_tape[k]
        return (('rt', k, form), f'arena + {start}LL + {_expr(form, var)} * {stride}LL')

    def ckey(c, form, var):
        parity = _parity(form)
        e = _expr(form, var)
        return (('cot', c, parity), f'((({e}) & 1) == 0 ? {cot[0][c]} : {cot[1][c]})')

    def xs_slice(form, var):
        return [(('x', k), f'{base} + {_expr(form, var)} * {words}LL') for k, (base, words) in enumerate(zip(seq_full, slice_words))]

    def dxs_slice(form, var):
        return [(('dx', j), f'{seq_cot[j]} + {_expr(form, var)} * {slice_words[i]}LL') for j, i in enumerate(seq_diff)]

    XF = [(('x', k), base) for k, base in enumerate(seq_full)]
    IT = [(('it', k), base) for k, base in enumerate(init_tape)]
    DXF = [(('dx', j), base) for j, base in enumerate(seq_cot)]
    P = [(('p', i), p[i]) for i in range(np_)]
    M = [(('m', i), m[i]) for i in range(no)]
    S = [(('s', i), s[i]) for i in range(ns)]
    D = [(('d', i), data[i]) for i in range(ni)]
    G = [(('g', i), gp[i]) for i in range(np_)]
    A = [(('participation', i), pointer) for i,pointer in enumerate(active)]
    W = [(('w', i), pointer) for i, pointer in enumerate(cast_buffers)]
    PW = P + W      # every non-optimizer region reads cast storage after the parameters

    def scratch_ptr(area):
        return area if isinstance(area, str) else f'arena + {area}LL'

    def x_area(form, var):
        e = _expr(form, var)
        return f'((({e}) & 1) == 0 ? arena + {X[0]}LL : arena + {X[1]}LL)'

    def forward_track(label, form, var, stages, guard=None):
        # Iteration index j = form: reads boundary j, writes boundary j+1 and tape j.
        nxt = (form[0], form[1], form[2] + 1)
        args = PW + [bkey(k, form, var) for k in range(nc + ns)] + xs_slice(form, var) + D
        outs = [bkey(k, nxt, var) for k in range(nc + ns)] + [tkey(k, form, var) for k in range(len(body_tape))]
        return _Track(label, forward_region, stages, args, outs, x_area(form, var), guard)

    def reverse_track(label, eform, var, stages, guard=None):
        # Execution step e: boundary/tape index r = n-1-e; cotangent parities e, e+1.
        r = (1 - eform[0], -eform[1], -1 - eform[2])
        nxt = (eform[0], eform[1], eform[2] + 1)
        args = (PW + [bkey(k, r, var) for k in range(nc + ns)] + xs_slice(r, var) + D
                + [tkey(k, r, var) for k in range(len(body_tape))] + [ckey(c, eform, var) for c in range(nc)])
        outs = ([G[k] for k in kept_grads] + [ckey(c, nxt, var) for c in range(nc)] + dxs_slice(r, var)
                + [rtkey(k, r, var) for k in range(len(reverse_tape))])
        area = x_area((1 + eform[0], eform[1], eform[2]), var)     # private parity n+e
        track = _Track(label, reverse_region, stages, args, outs, area, guard)
        track.p9_step = _expr(eform, var)                        # P9-B: execution step of this track
        return track

    def all_stages(name):
        return list(range(_stage_count(name)))

    copy_stage = {}
    functions, helpers = [], []
    function_region, emitted_tracks = {}, []   # P7-C: region of each function text; every emitted track
    event_regions = []  # P6-C phase_ticks: region index -> name

    def _stage_count(name):
        return len(programs[name].manifest['stages']) + (1 if copy_stage.get(name) else 0)

    for name, program in programs.items():
        mf = program.manifest
        if program.helpers and program.helpers not in helpers:
            helpers.append(program.helpers)
        placed = set(mf.get('placed_outputs', ()))
        accumulating = name in (reverse_region, init_vjp_name)
        ctr_param = ', float* latch_ctr' if name in counter_area else ''
        copies, reads, writes = [], set(), set()
        for i, value_name in enumerate(mf['outputs']):
            if i in placed:
                continue
            desc = mf['layouts'][value_name]
            root = desc['root']
            if root in mf['inputs']:
                pointer = f'args[{mf["inputs"].index(root)}]'
                reads.add(('arg', mf['inputs'].index(root)))
            else:
                pointer = f'arena + {mf["offsets"][root]}LL'
            layout = TensorLayout(tuple(desc['shape']), tuple(desc['strides']), root, desc['offset'])
            ctype = _CTYPES[desc['dtype']]
            op = '+=' if accumulating and i < n_acc.get(name, np_) else '='
            if op == '+=':
                reads.add(('out', i))
            writes.add(('out', i))
            copies.append(
                f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {layout.numel}LL; '
                f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>(out[{i}])[i] {op} '
                f'reinterpret_cast<const {ctype}*>({pointer})[{layout.offset}LL + {_index(layout, layout.shape)}];')
        function_region[len(functions)] = name
        functions.append(program.kernel_body.replace('__REGION_ID__', str(len(event_regions)))
                         .replace('__REGION__', f'loop_{name}'))
        event_regions.append(name)
        if copies:
            copy_stage[name] = {'reads': reads, 'writes': writes}
            functions.append(f'__device__ __noinline__ void loop_{name}_s{len(mf["stages"])}(float** args, float* arena, float** out, int vrot, int cta0, int ctas{ctr_param}) {{\n'
                             + '\n'.join(copies) + '\n}\n')

    def stage_rw(track, stage):
        from .execution_plan import current_execution, region_stage_effects
        compose = current_execution() == "compose"
        if track.copy_rw is not None:
            return region_stage_effects(*track.copy_rw) if compose else track.copy_rw
        mf = programs[track.region].manifest
        if stage < len(mf['stages']):
            info = mf['stages'][stage]
            r = ({track.args[i][0] for i in info['reads_inputs']} | {track.outs[o][0] for o in info['accumulates']}
                 | {track.outs[o][0] for o in info.get('reads_outputs', ())})
            w = {track.outs[o][0] for o in info['writes_outputs']}
        else:
            info = copy_stage[track.region]
            r = {(track.args[i][0] if kind == 'arg' else track.outs[i][0]) for kind, i in info['reads']}
            w = {track.outs[i][0] for _, i in info['writes']}
        return (region_stage_effects(r, w, global_fence=bool(info.get('global_fence')))
                if compose else (frozenset(r), frozenset(w)))

    def interval_of(track, stage):
        if track.region is None:
            return stage
        stages = programs[track.region].manifest['stages']
        return stages[stage]['interval'] if stage < len(stages) else (stages[-1]['interval'] + 1 if stages else 0)

    from .group_rules import features_of
    _features = features_of(options.get('group_rules'))
    alap_regions = ({'optimizer'} if 'alap' in _features else set()) | ({'init_vjp'} if 'alapv' in _features else set())
    part_regions = {'optimizer'} | ({'init_vjp'} if 'alapv' in _features else set())

    def merge(tracks):
        # A stage follows every stage of its region's previous interval (interval
        # order keeps the region's interval-unit arena lifetimes valid) and every
        # conflicting stage of an earlier track.
        placed = []
        for t, track in enumerate(tracks):
            # A region without scratch storage exposes every internal dependency
            # as a buffer key, so its stages need no interval floor.
            free = track.region is not None and programs[track.region].manifest['arena_words'] == 0 \
                and not copy_stage.get(track.region)
            done, current, current_max = -1, None, -1
            for stage in track.stages:
                r, w = stage_rw(track, stage)
                k = interval_of(track, stage)
                if k != current:
                    done, current = max(done, current_max), k
                level = 0 if free else done + 1
                for other_level, other_t, _, other_r, other_w in placed:
                    if (other_t != t or free) and ((other_w & (r | w)) or (other_r & w)):
                        level = max(level, other_level + 1)
                placed.append((level, t, stage, r, w))
                current_max = max(current_max, level)
        if alap_regions and placed:
            # P-OPS5-B slack placement: stages of commit-only regions (the optimizer) move to the latest
            # level before every conflicting later stage, so their streams stop sharing levels with the
            # critical last reverse iteration. Dependencies (buffer conflicts, region interval order) hold.
            last = max(entry[0] for entry in placed)
            for index in range(len(placed) - 1, -1, -1):
                level, t, stage, r, w = placed[index]
                if tracks[t].region not in alap_regions:
                    continue
                k = interval_of(tracks[t], stage)
                bound = last
                for later, (other_level, other_t, other_stage, other_r, other_w) in enumerate(placed):
                    if later == index:
                        continue
                    conflict = (other_w & (r | w)) or (other_r & w)
                    ordered = later > index and (other_t != t or interval_of(tracks[other_t], other_stage) > k)
                    if ordered and conflict or (other_t == t and later > index
                                                  and interval_of(tracks[other_t], other_stage) > k):
                        bound = min(bound, other_level - 1)
                if bound > level:
                    placed[index] = (bound, t, stage, r, w)
        levels = sorted({entry[0] for entry in placed})
        return [[(t, stage) for level2, t, stage, _, _ in placed if level2 == level] for level in levels]

    def copy_track(label, pairs, guard=None):
        # Independent element copies sharing one stage: [(dest key, ptr, src key, ptr, desc)].
        code = []
        if len(pairs) > 1 and COPY_PARTITION:
            # P9-C: each copy runs on its own CTA range (proportional to its size) as 16-byte, four-way unrolled
            # vector copies when its element size and pointers allow, instead of every copy back to back on every
            # CTA. The same elements are copied.
            sizes = [math.prod(desc['shape']) * _BYTES_OF.get(desc['dtype'], 4) for *_, desc in pairs]
            weights = [max(1, n // 16384) for n in sizes]
            total, prefix = sum(weights), 0
            for (_, dptr, _, sptr, desc), w, nbytes in zip(pairs, weights, sizes):
                ctype = _CTYPES[desc['dtype']]
                n = math.prod(desc['shape'])
                scalar = (f'for (long long i = lb_ * blockDim.x + threadIdx.x; i < {n}LL; i += lg_ * blockDim.x) '
                          f'reinterpret_cast<{ctype}*>({dptr})[i] = reinterpret_cast<const {ctype}*>({sptr})[i];')
                body = scalar
                if nbytes % 16 == 0:
                    v = nbytes // 16
                    body = (f'if (((reinterpret_cast<uintptr_t>({dptr}) | reinterpret_cast<uintptr_t>({sptr})) & 15) == 0) {{\n'
                            f'  const uint4* s_ = reinterpret_cast<const uint4*>({sptr}); uint4* d_ = reinterpret_cast<uint4*>({dptr});\n'
                            f'  for (long long c0 = lb_ * blockDim.x + threadIdx.x; c0 < {v}LL; c0 += 4 * lg_ * blockDim.x) {{\n'
                            f'    uint4 t_[4];\n'
                            f'#pragma unroll\n    for (int u = 0; u < 4; ++u) {{ const long long c = c0 + u * lg_ * blockDim.x; if (c < {v}LL) t_[u] = s_[c]; }}\n'
                            f'#pragma unroll\n    for (int u = 0; u < 4; ++u) {{ const long long c = c0 + u * lg_ * blockDim.x; if (c < {v}LL) d_[c] = t_[u]; }}\n'
                            f'  }}\n}} else {{ {scalar} }}')
                code.append(f'{{ const int lo_ = (int)((long long)gridDim.x * {prefix} / {total});\n'
                            f'  const int hi_ = max(lo_ + 1, (int)((long long)gridDim.x * {prefix + w} / {total}));\n'
                            f'  if ((int)blockIdx.x >= lo_ && (int)blockIdx.x < hi_) {{\n'
                            f'  const long long lb_ = (int)blockIdx.x - lo_, lg_ = hi_ - lo_;\n{body}\n}} }}')
                prefix += w
            pairs_code = None
        for _, dptr, _, sptr, desc in (pairs if not code else ()):
            ctype = _CTYPES[desc['dtype']]
            n = math.prod(desc['shape'])
            code.append(f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {n}LL; '
                        f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<{ctype}*>({dptr})[i] = '
                        f'reinterpret_cast<const {ctype}*>({sptr})[i];')
        rw = (frozenset(src for _, _, src, _, _ in pairs), frozenset(dst for dst, _, _, _, _ in pairs))
        return _Track(label, None, [0], [], [], None, guard, copy_code='\n'.join(code), copy_rw=rw)

    tick_layout, tick_top = [], 8
    partition = bool(options['partition_ctas'])
    part_feature = 'part' in features_of(options.get('group_rules'))

    lpt_log = []

    def emit(block, tracks, var=None, iterations=1, iteration_expr='0', bg=None):
        nonlocal tick_top
        emitted_tracks.extend(tracks)
        schedule = merge(tracks)
        base = tick_top
        tick_top += iterations * (len(schedule) + 1)
        tick_layout.append({'block': block, 'base': base, 'iterations': iterations, 'levels': len(schedule),
                            'members': [[[tracks[t].label, stage] for t, stage in level] for level in schedule]})
        # P6-C diagnostic (member_ticks): per (iteration, level, member) the latest CTA finish stamp (atomicMax
        # after the member's CTA-local barrier). Default off; never part of a timed build.
        mbase = mwidth = None
        if options.get('member_ticks'):
            mwidth = max(len(level) for level in schedule)
            mbase = tick_top
            tick_top += iterations * len(schedule) * mwidth
            tick_layout[-1].update(member_base=mbase, member_width=mwidth)
        slot = f'{base}LL + ({iteration_expr}) * {len(schedule) + 1}LL'
        lines = ['{']
        fills = []
        for t, track in enumerate(tracks):
            if track.region is None:
                continue
            if options['shared_tables']:
                # Pointer tables in shared memory: inlined stages load what they use
                # instead of keeping every block-wide pointer live in registers.
                lines.append(f'__shared__ float* a{t}[{max(1, len(track.args))}]; __shared__ float* o{t}[{max(1, len(track.outs))}];')
                fills += [f'a{t}[{k}] = {ptr};' for k, (_, ptr) in enumerate(track.args)]
                fills += [f'o{t}[{k}] = {ptr};' for k, (_, ptr) in enumerate(track.outs)]
            elif P9_LOCAL_PTRS:
                continue
            else:
                lines.append(f'float* a{t}[] = {{' + ', '.join(ptr for _, ptr in track.args) + '};')
                lines.append(f'float* o{t}[] = {{' + ', '.join(ptr for _, ptr in track.outs) + '};')
            lines.append(f'float* x{t} = {scratch_ptr(track.scratch)};')
        if fills:
            lines.append('if (threadIdx.x == 0) {\n' + '\n'.join(fills) + '\n}\n__syncthreads();')
        lines.append(f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{slot}] = stamp();')
        if options.get('phase_ticks'):
            lines.append('P6C_LEVEL();')
        def stage_meta(track, stage):
            if track.region is None:
                return 0, 1
            stages = programs[track.region].manifest['stages']
            if stage >= len(stages):
                return 0, 1
            return stages[stage]['gemm_tiles'], stages[stage].get('jobs', 0)

        for index, level in enumerate(schedule):
            parts, rotation = [], 0
            # P-OPS5-B placement: products of every member start after the largest CTA footprint of the
            # level's non-matrix work (which always starts at CTA 0), when the level's tiles fit after it.
            footprints = [programs[tracks[t].region].manifest['stages'][stage].get('footprint')
                          for t, stage in level if tracks[t].region is not None
                          and stage < len(programs[tracks[t].region].manifest['stages'])]
            if any(f is not None for f in footprints):
                tiles = sum(stage_meta(tracks[t], stage)[0] for t, stage in level)
                base = max(f or 0 for f in footprints)
                rotation = base if base + tiles <= 128 else 0
            # P6-C lpt: stages of one level take the rotations that least load the busiest CTA under the stages'
            # modeled product loads (heaviest stage first); scheduling only.
            lpt_rot = None
            def member_stage(t, stage):
                track_ = tracks[t]
                if track_.region is None or stage >= len(programs[track_.region].manifest['stages']):
                    return {}
                return programs[track_.region].manifest['stages'][stage]
            absolute = {k_ for k_, (t, stage) in enumerate(level) if member_stage(t, stage).get('lpt_absolute')}
            if 'lptl' in features_of(options.get('group_rules')):
                loads = []
                for k_, (t, stage) in enumerate(level):
                    st = (programs[tracks[t].region].manifest['stages'][stage]
                          if tracks[t].region is not None and stage < len(programs[tracks[t].region].manifest['stages'])
                          else {})
                    loads.append(st.get('gemm_load'))
                if sum(1 for x in loads if x and max(x) > 0) > 1:
                    total_, lpt_rot = [0.0] * 128, {}
                    for k_ in absolute:
                        if loads[k_]:
                            total_ = [a + b for a, b in zip(total_, loads[k_])]
                    for k_ in sorted((k_ for k_, x in enumerate(loads) if x and max(x) > 0 and k_ not in absolute),
                                     key=lambda k_: -max(loads[k_])):
                        x = loads[k_]
                        best_ = min(range(128), key=lambda r: (round(max(total_[c] + x[(c - r) % 128] for c in range(128)), 4),
                                                              (r - rotation) % 128))
                        lpt_rot[k_] = best_
                        total_ = [total_[c] + x[(c - best_) % 128] for c in range(128)]
            # Pinned GEMMs take CTAs by rotation; with partition_ctas, the other
            # CTAs are divided among members' remaining work by jobs.
            metas = [stage_meta(tracks[t], stage) for t, stage in level]
            gemm_total = sum(m[0] for m in metas)
            weights = [min(m[1], 1 << 20) for m in metas]
            busy = [k for k, w in enumerate(weights) if w]
            total, prefix = sum(weights), 0
            # P9-C pelem: the self-contained stages of a level (stage records flagged 'pelem') take disjoint CTA
            # spans proportional to their jobs; every other member keeps the full grid.
            # Only small stages are partitioned (P9-C r9: large fills and ticketed reductions on a proportional span
            # stalled their level); large self-contained stages keep the full grid.
            safe_members = [k_ for k_, (t, stage) in enumerate(level) if member_stage(t, stage).get('pelem')
                            and weights[k_] and member_stage(t, stage).get('pelem_work', 1 << 30) <= PELEM_MAX_WORK
                            ] if 'pelem' in features_of(options.get('group_rules')) else []
            if len(safe_members) < 2:
                safe_members = []
            safe_w = {k_: max(1, int(member_stage(*level[k_]).get('pelem_work', weights[k_]))) for k_ in safe_members}
            safe_total, safe_prefix = sum(safe_w.values()), 0
            log_members = []
            for k, (t, stage) in enumerate(level):
                track = tracks[t]
                rot_k = 0 if k in absolute else (lpt_rot[k] if lpt_rot and k in lpt_rot else rotation)
                log_members.append([track.region, stage, rot_k])
                # P-OPS5-B: a level made only of commit-only optimizer stages runs them on disjoint CTA ranges.
                part_level = part_feature and all(tracks[t].region in part_regions for t, _ in level)
                if k in safe_members:
                    w_ = safe_w[k]
                    lo = f'(int)((long long)(int)gridDim.x * {safe_prefix} / {safe_total})'
                    hi = f'(int)((long long)(int)gridDim.x * {safe_prefix + w_} / {safe_total})'
                    span = (lo, f'max(1, {hi} - {lo})')
                    safe_prefix += w_
                elif (partition or part_level) and len(busy) > 1 and weights[k]:
                    lo = f'{gemm_total} + (int)((long long)((int)gridDim.x - {gemm_total}) * {prefix} / {total})'
                    hi = f'{gemm_total} + (int)((long long)((int)gridDim.x - {gemm_total}) * {prefix + weights[k]} / {total})'
                    fits = f'(int)gridDim.x >= {gemm_total + len(busy)}'
                    span = (f'({fits} ? {lo} : 0)', f'({fits} ? max(1, {hi} - ({lo})) : (int)gridDim.x)')
                    prefix += weights[k]
                else:
                    span = ('0', '(int)gridDim.x')
                call = (track.copy_code if track.region is None
                        else f'loop_{track.region}_s{stage}(a{t}, x{t}, o{t}, {rot_k}, {span[0]}, {span[1]}'
                             + (f', arena + {counter_area[track.region]}LL' if track.region in counter_area else '') + ');')
                if P9_LOCAL_PTRS and track.region is not None and not options['shared_tables']:
                    call = _local_tables(track, t, var, scratch_ptr(track.scratch)) + call
                if acc_cov and track.region == reverse_region and getattr(track, 'p9_step', None) is not None:
                    call = f'p5c::p5c_first = (({track.p9_step}) == 0) ? 1 : 0; ' + call
                if track.region is not None and stage < len(programs[track.region].manifest['stages']):
                    rotation += programs[track.region].manifest['stages'][stage]['gemm_tiles']
                parts.append(f'if ({track.guard}) {{ {call} }}' if track.guard else f'{{ {call} }}')
                if mbase is not None:
                    parts[-1] += (f'\n__syncthreads();\nif (ticks && threadIdx.x == 0) atomicMax(ticks + {mbase}LL + '
                                  f'(({iteration_expr}) * {len(schedule)}LL + {index}LL) * {mwidth}LL + {k}LL, stamp());')
            lpt_log.append({'block': block, 'members': log_members})
            guards = {tracks[t].guard for t, _ in level}
            barrier = 'grid.sync();'
            if bg is not None:
                # LATCH-STEP-06 (loop_dgrad.BG): chunks of weights whose final product ran in an earlier level of the
                # final reverse step (track bg[0] when bg[1] holds) may be summed by CTAs waiting at this barrier.
                done_ = {stage for lv in schedule[:index] for t_, stage in lv if tracks[t_].label == bg[0]}
                ready_ = bg_plan['ready'](done_)
                # Each such site is an out-of-line call (register spills around it): loop_dgrad.BG_SITES 'attention'
                # keeps only the barriers that end a level holding an attention-backward phase.
                site_ = loop_dgrad.BG_SITES == 'all' or any(
                    programs[tracks[t_].region].manifest['phases'][ph_].get('kind') == 'attention_long_backward'
                    for t_, stage in level if tracks[t_].region is not None
                    and stage < len(programs[tracks[t_].region].manifest['stages'])
                    for ph_ in programs[tracks[t_].region].manifest['stages'][stage]['phases'])
                if ready_ and site_:
                    barrier = f'STEP06_BG_SYNC(({bg[1]}) ? {ready_} : 0);'
            if len(guards) == 1 and None not in guards:
                barrier = f'if ({next(iter(guards))}) {barrier}'
            lines.append('\n__syncthreads();\n'.join(parts))
            lines.append(barrier)
            lines.append(f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{slot} + {index + 1}] = stamp();')
            if options.get('phase_ticks'):
                lines.append('P6C_LEVEL();')
        lines.append('}')
        return '\n'.join(lines), schedule

    # ---------------------------------------------------------- split choice
    acc_cov = set()
    if P9_ACC_FIRST:
        acc_cov, functions = _acc_first(functions, function_region, reverse_region)
    S_F, S_R = _stage_count(forward_region), _stage_count(reverse_region)

    def forward_body(split):
        return [forward_track('F(j)', _form(0, 1, 0), 'j', range(split, S_F)),
                forward_track('F(j+1)', _form(0, 1, 1), 'j', range(0, split))]

    def reverse_body(split):
        return [reverse_track('R(e)', _form(0, 1, 0), 'e', range(split, S_R)),
                reverse_track('R(e+1)', _form(0, 1, 1), 'e', range(0, split))]

    def best(builder, total):
        scores = [(len(merge(builder(split))), -split, split) for split in range(total + 1)]
        return min(scores)[2]
    skew = options['pipeline'] == 'full'
    split_f = best(forward_body, S_F) if skew else S_F
    split_r = best(reverse_body, S_R) if skew else 0
    # S_F (all prefix) / 0 (all suffix) reproduce an unpipelined loop body.

    nonzero = 'count > 0'
    blocks = []
    count_input = len(inputs) - 1
    body = ['cg::grid_group grid = cg::this_grid();',
            'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();',
            f'const int64_t requested_count = *reinterpret_cast<const int64_t*>(args[{count_input}]);',
            f'const int count = int(requested_count < 0 ? 0 : (requested_count > {graph.max_steps}LL ? {graph.max_steps}LL : requested_count));',
            f'if (blockIdx.x == 0 && threadIdx.x == 0) {{ '
            f'*reinterpret_cast<int64_t*>(arena + {offsets[structured["executed_count"]]}LL) = count; '
            f'*reinterpret_cast<int64_t*>(arena + {offsets[structured["status"]]}LL) = requested_count == count ? 0 : 1; }}']
    if acc_cov:
        body.append('p5c::p5c_skipzero = (count > 0) ? 1 : 0; p5c::p5c_first = 0;')
    if connectivity:
        body += [_participation_code(connectivity, parameters, carries, active), 'grid.sync();']
    zero = _form()
    # Block A: initialization, explicit-state boundary copy, first forward prefix.
    init = _Track('init', init_name, all_stages(init_name), PW + D, [bkey(k, zero, None) for k in range(nc)] + XF + IT, Y)
    state0 = copy_track('state0', [(*bkey(nc + k, zero, None), *S[k], state_desc[k]) for k in range(ns)])
    zero_lines = []
    for j, i in enumerate(seq_diff):
        words = slice_words[i]
        zero_lines.append(f'for (long long i = (long long)count * {words}LL + (long long)blockIdx.x * blockDim.x + threadIdx.x; '
                          f'i < {graph.max_steps * words}LL; i += (long long)gridDim.x * blockDim.x) '
                          f'reinterpret_cast<unsigned int*>({seq_cot[j]})[i] = 0u;')
    zero_dx = _Track('zero_dx', None, [0], [], [], None, None, copy_code='\n'.join(zero_lines),
                     copy_rw=(frozenset(), frozenset(key for key, _ in DXF)))
    cast_track = ([_Track('casts', 'invariant_casts', all_stages('invariant_casts'), P + D, W, W_scratch)]
                  if casts else [])
    tracks = (cast_track + [init] + ([state0] if ns else []) + ([zero_dx] if seq_diff else [])
              + [forward_track('F(0)', zero, None, range(0, split_f), nonzero)])
    code, schedule = emit('A', [t for t in tracks if t.stages])
    body += [code, 'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();']
    # Pipelined forward loop: suffix of j with prefix of j+1.
    loop_tracks = [t for t in forward_body(split_f) if t.stages]
    code, schedule = emit('forward', loop_tracks, 'j', max(1, graph.max_steps), 'j')
    body += ['for (int j = 0; j + 1 < count; ++j) ' + code,
             'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[2] = stamp();']
    # LATCH-STEP-06 (loop_dgrad.BG): the reverse-body stage that writes each deferred weight's final product, chunk
    # order by that stage, and the ready-chunk count after a set of completed final-step stages.
    bg_plan = None
    if (dgrad and dgrad.get('mode') == 'tape' and loop_dgrad.BG and options['barrier'] == 'counter'
            and split_r > 0):
        rev_stages = programs[reverse_region].manifest['stages']
        n_out = len(reverse_track('bg', zero, None, range(0)).outs)
        first_rt = n_out - len(reverse_tape)
        prod = {}
        for k_, e_ in enumerate(dgrad['entries']):
            hits_ = [st_ for st_, rec_ in enumerate(rev_stages) if first_rt + e_['operand_index'] in rec_.get('writes_outputs', ())]
            prod[k_] = max(hits_) if hits_ else None
        if all(v_ is not None for v_ in prod.values()):
            # Only +0-initialized gradients (no other writer of G in the final reverse step) are summed early;
            # the others stay for block C, after every terminal/initialization contribution.
            zi_ = {k_: bool(e_.get('zero_init')) for k_, e_ in enumerate(dgrad['entries'])}
            order_ = sorted(prod, key=lambda k_: (not zi_[k_], prod[k_], k_))
            bg_decl, bg_count, bg_first = loop_dgrad.bg_tables(dgrad, order_)
            chunks_ = [bg_first[k_] for k_ in order_] + [bg_count]

            def _ready(done_, order_=order_, chunks_=chunks_, prod=prod, zi_=zi_):
                n_ = 0
                while (n_ < len(order_) and zi_[order_[n_]] and prod[order_[n_]] in done_
                       and prod[order_[n_]] < split_r):
                    n_ += 1
                return chunks_[n_]
            bg_plan = {'decl': bg_decl, 'count': bg_count, 'ready': _ready, 'order': order_, 'prod': prod}
    # Block B: last forward suffix, terminal objective/VJP, first reverse prefix.
    last = _form(1, 0, -1)
    final = _form(1, 0, 0)
    terminal = _Track('terminal', 'terminal_vjp', all_stages('terminal_vjp'),
                      PW + [bkey(k, final, None) for k in range(nc)] + D,
                      [(('loss',), loss_ptr)] + G + [ckey(c, zero, None) for c in range(nc)], Y)
    tracks = [forward_track('F(n-1)', last, None, range(split_f, S_F), nonzero), terminal,
              reverse_track('R(0)', zero, None, range(0, split_r), nonzero)]
    code, schedule = emit('B', [t for t in tracks if t.stages], bg=('R(0)', 'count == 1') if bg_plan else None)
    body += [code, 'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[3] = stamp();']
    # Pipelined reverse loop: suffix of step e with prefix of step e+1.
    loop_tracks = [t for t in reverse_body(split_r) if t.stages]
    code, schedule = emit('reverse', loop_tracks, 'e', max(1, graph.max_steps), 'e',
                          bg=('R(e+1)', 'e + 2 == count') if bg_plan else None)
    body += ['for (int e = 0; e + 1 < count; ++e) ' + code,
             'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[4] = stamp();']
    # Block C: last reverse suffix, initialization VJP, optimizer, final commit.
    init_vjp = _Track('init_vjp', init_vjp_name, all_stages(init_vjp_name),
                      PW + D + IT + [ckey(c, final, None) for c in range(nc)] + DXF, G, Y)
    if inplace:
        opt_outs = [(('p', i), p[i]) for i in range(np_)] + [(('m', i), m[i]) for i in range(no)]
    else:
        opt_outs = [(('pp', i), pp[i]) for i in range(np_)] + [(('mp', i), mp[i]) for i in range(no)]
    optimizer = _Track('optimizer', 'optimizer', all_stages('optimizer'), P + M + G + A
                       + [D[i] for i in metadata.get('optimizer_inputs', [])], opt_outs, Z)
    commit = []
    for k in range(ns):
        src = bkey(nc + k, final, None)
        commit.append((('s', k), s[k], *src, state_desc[k]))
        commit.append((('sp', k), sp[k], *src, state_desc[k]))
    for k in range(nc):
        commit.append((('cp', k), cp[k], *bkey(k, final, None), carry_desc[k]))
    if not inplace:
        commit += [(('p', i), p[i], ('pp', i), pp[i], param_desc[i]) for i in range(np_)]
        commit += [(('m', i), m[i], ('mp', i), mp[i], optimizer_desc[i]) for i in range(no)]
    dgrad_tracks, fuse_flag = [], False
    if dgrad:
        # LATCH-STEP-06: every deferred gradient once, after the last reverse step (iteration 0) and before the
        # initialization VJP and the optimizer read or add to those gradients.
        reads = ({G[e['param']][0] for e in dgrad['entries']} | {rtkey(k, zero, None)[0] for k in range(len(reverse_tape))}
                 | {tkey(e['tape_index'], zero, None)[0] for e in dgrad['entries']})
        writes = {G[e['param']][0] for e in dgrad['entries']}
        offsets_g = {k: int(offsets[structured['grads'][parameters[k]]]) for k in range(np_)}
        # LATCH-STEP-06 (loop_defer_grads.FUSE_OPTIMIZER): deferred weights whose optimizer updates make up whole
        # in-place optimizer stages (a stage may group several same-shape weights) get those updates in the deferred
        # stage's epilogue; the stages are dropped.
        fused_stages = set()
        fuse_flag = bool(inplace) and any(e.get('optimizer') for e in dgrad['entries'])
        if fuse_flag:
            token = ('compose_global_grid_fence',)
            owners = {}
            for e in dgrad['entries']:
                if not e.get('optimizer'):
                    continue
                k, j = e['param'], e['optimizer']['m_arg'] - np_
                own = {('p', k), ('m', j)}
                hits = [st for st in all_stages('optimizer') if stage_rw(optimizer, st)[1] - {token} & own]
                if len(hits) == 1:
                    owners.setdefault(hits[0], []).append((e, own))
                else:
                    e['optimizer'] = None
            for st, members in owners.items():
                owned = set().union(*(own for _, own in members))
                if (stage_rw(optimizer, st)[1] - {token}) == owned:
                    fused_stages.add(st)
                    reads |= owned
                    writes |= owned
                else:
                    for e, _ in members:
                        e['optimizer'] = None
            fuse_flag = bool(fused_stages)
            if fused_stages:
                optimizer.stages = [st for st in optimizer.stages if st not in fused_stages]
        if dgrad.get('mode', 'batch') == 'tape':
            dgrad_table, dgrad_code, _ = loop_dgrad.stage_tape(dgrad, offsets_g, reverse_tape, fuse=fuse_flag)
            if bg_plan:
                # Every chunk not summed in a barrier wait of the final reverse step.
                dgrad_table += bg_plan['decl']
                dgrad_code = (f'step06::bg_drain<{dgrad["_opt_name"]}>(step06_ds_tab, step06_bg_tab, '
                              f'{bg_plan["count"]}, count, args, arena);')
        else:
            dgrad_table, dgrad_code, _ = loop_dgrad.stage(
                dgrad, offsets_g, [(2 * i, 2 * i + 1) for i in range(len(dgrad['entries']))], 0, fuse=fuse_flag)
        any_zero = any(e.get('zero_init') for e in dgrad['entries'])
        dgrad_tracks = [_Track('dgrad', None, [0], [], [], None, None if any_zero else nonzero, copy_code=dgrad_code,
                               copy_rw=(frozenset(reads | writes), frozenset(writes)))]
    if dgrad and fuse_flag:
        # The fused stage writes parameters and optimizer state: it follows the initialization VJP (whose
        # contributions to these gradients are +0 and elided).
        tracks = [reverse_track('R(n-1)', last, None, range(split_r, S_R), nonzero), init_vjp, *dgrad_tracks,
                  optimizer, copy_track('commit', commit)]
    else:
        tracks = [reverse_track('R(n-1)', last, None, range(split_r, S_R), nonzero), *dgrad_tracks, init_vjp,
                  optimizer, copy_track('commit', commit)]
    code, schedule = emit('C', [t for t in tracks if t.stages])
    body += [code] + [f'if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i}] = stamp();' for i in (5, 6, 7)]

    if options['barrier'] not in ('cg', 'counter'):
        raise LoweringError(f"unknown loop barrier {options['barrier']}")
    if options['inline_stages']:
        # P9-B 'wgmma': only stages holding warpgroup-MMA products are inlined (ptxas serializes wgmma in out-of-line
        # functions); every other stage keeps its own register allocation instead of adding to the single kernel
        # function's pressure (spills).
        keep = ('p5c::', 'p8ws::', 'p9ws::', 'pops4b::')
        functions = [text.replace('__device__ __noinline__ void loop_', '__device__ __forceinline__ void loop_')
                     if P9_INLINE == 'all' or any(k in text for k in keep) else text
                     for text in functions]
    if STEP06_ROW_PF:
        functions = [_row_prefetch(text) for text in functions]
    if any('grid.sync();' in text for text in functions):
        raise LoweringError('staged regions must not contain grid barriers')
    if counter_area:
        # Zero every persistent split-K counter before the launch's first barrier.
        zero = [f'for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {align(programs[name].manifest["persistent_counter_words"])}LL; '
                f'i += (long long)gridDim.x * blockDim.x) reinterpret_cast<unsigned*>(arena + {offset}LL)[i] = 0u;'
                for name, offset in counter_area.items()]
    else:
        zero = []
    if options['barrier'] == 'counter':
        # The zeroing precedes latch_loop_bar_begin, whose cooperative grid sync orders it.
        body = [line.replace('grid.sync();', 'latch_loop_barrier(loop_counter);') for line in body]
        import re as _rebg
        body = [_rebg.sub(r'STEP06_BG_SYNC\((.*?)\);', r'latch_loop_barrier_bg(loop_counter, \1, count, args, arena);', line)
                for line in body]
        bg_zero = ['if (blockIdx.x == 0 && threadIdx.x == 0) step06::step06_bg_next = 0u;'] if bg_plan else []
        body = [body[0], f'unsigned long long* loop_counter = reinterpret_cast<unsigned long long*>(arena + {barrier_offset}LL);',
                *zero, *bg_zero, 'latch_loop_bar_begin(loop_counter);', *body[1:]]
        helpers.append(COUNTER_BARRIER)
    elif zero:
        body = [body[0], *zero, 'grid.sync();', *body[1:]]
    phase_events = None
    if options.get('phase_ticks'):
        # P6-C diagnostic event log after all tick words: (stamp, id) pairs in a sequence every CTA advances
        # identically (uniform guards); phase ends take the latest CTA stamp, level markers CTA 0's stamp.
        capacity = 8192
        phase_events = {'base': tick_top, 'capacity': capacity, 'regions': list(event_regions)}
        body = [body[0], f'if (threadIdx.x == 0) {{ p6c_ev = ticks ? ticks + {tick_top}LL : nullptr; p6c_seq = 0u; }}',
                '__syncthreads();', *body[1:]]
        tick_top += 2 * capacity
        helpers.append('''
__shared__ unsigned long long* p6c_ev;
__shared__ unsigned p6c_seq;
#define P6C_EV(R, P) do { __syncthreads(); if (threadIdx.x == 0) { if (p6c_ev && p6c_seq < 8192u) { atomicMax(p6c_ev + 2ull * p6c_seq, stamp()); if (blockIdx.x == 0) p6c_ev[2ull * p6c_seq + 1] = ((unsigned long long)(R) << 16) | (unsigned long long)(P); } ++p6c_seq; } } while (0)
#define P6C_LEVEL() do { if (threadIdx.x == 0) { if (p6c_ev && p6c_seq < 8192u && blockIdx.x == 0) { p6c_ev[2ull * p6c_seq] = stamp(); p6c_ev[2ull * p6c_seq + 1] = ~0ull; } ++p6c_seq; } } while (0)
''')
    tensor_maps = None
    from .p5c_matrix import TMAP_REGION
    dgrad_maps = loop_dgrad.maps(dgrad, body_tape, reverse_tape, graph.max_steps)[0] if dgrad else []
    if any(TMAP_REGION + '(' in text for text in functions) or dgrad_maps:
        functions, tensor_maps, extra_base = _region_tensor_maps(
            functions, function_region, emitted_tracks, programs, len(inputs),
            boundaries=boundaries, body_tape=body_tape, tape_size=tape_size, max_steps=graph.max_steps,
            body_regions=(forward_region, reverse_region), x_areas=X, reverse_tape=reverse_tape,
            extra_maps=dgrad_maps)
        if dgrad and dgrad_maps:
            # Map indices of the deferred-gradient table are relative to the extra maps' base.
            dgrad_table = loop_dgrad.stage(dgrad, offsets_g,
                                           [(extra_base + 2 * i, extra_base + 2 * i + 1)
                                            for i in range(len(dgrad['entries']))], 0, fuse=fuse_flag)[0]
        from .p5c_matrix import MAP_ADDR_SHARED_MAX, TRAP_ASSERT, region_map_helpers
        shared_addresses = 0 < len(tensor_maps) <= MAP_ADDR_SHARED_MAX
        body = [body[0], 'if (threadIdx.x == 0) p5c_maps_sh = reinterpret_cast<const char*>(args[%d]);' % len(inputs),
                '__syncthreads();', *(['p5c_load_addresses();', '__syncthreads();'] if shared_addresses else []),
                *body[1:]]
        helpers.append(TRAP_ASSERT + region_map_helpers(len(tensor_maps)))
    if dgrad:
        helpers.append(('#define STEP06_DSUM8 1\n' if loop_dgrad.DSUM8 else '')
                       + ('#define STEP06_BG_INLINE 1\n' if loop_dgrad.BG_INLINE else '')
                       + f'#include "{loop_dgrad.HEADER}"\n' + dgrad_table)
        if bg_plan:
            helpers.append(loop_dgrad.bg_barrier(dgrad['_opt_name']))
    if P9_LOCAL_PTRS:
        helpers.insert(0, P9_OPQ_HELPER)
    if P9_SREG_MACRO:
        helpers.insert(0, P9_SREG_HELPER)
    hoisted = None
    if STEP06_MASK_HOIST:
        helpers, body, hoisted = _hoist_mask_tables(helpers, functions, body)
    kernel_body = '\n'.join(body)
    helper_source = '\n'.join(helpers)
    shared_bytes = max(program.manifest.get('shared_bytes', 17408) for program in programs.values())
    from .resources import select_program_resources
    resource = select_program_resources(shared_bytes, children=[p.manifest for p in programs.values()],
        arch=options['region_options'].get('target_arch','sm_90'), allow_two_cta=not options['one_cta'])
    source = _translation_unit(kernel_body, helpers=helper_source, functions='\n'.join(functions), shared_bytes=shared_bytes,
                               resident_blocks_per_sm=resource['resident_blocks_per_sm'],
                               cluster=(8 if any('bmm_phase_cl<' in text for text in functions)
                                        else 2 if any('/*P5C_CLUSTER2*/' in text for text in functions) else 0))
    if options['one_cta']:
        # One resident 256-thread CTA per SM is all the cooperative grid needs
        # (<= 128 blocks on 132 SMs); let the compiler use the register file.
        source = source.replace('extern "C" __global__ void latch_ops(',
                                'extern "C" __global__ void __launch_bounds__(256, 1) latch_ops(', 1)
    from .group_rules import features_of
    if 'shk' in features_of(options.get('group_rules')):
        source = '#define POPSC_SHK 1  // P6-C shk: one shared K-loop body per product tile configuration\n' + source
    header_hashes = {}
    for program in programs.values():
        for name, digest in program.manifest['header_hashes'].items():
            if name in header_hashes and header_hashes[name] != digest:
                raise LoweringError(f'loop region header identities differ: {name}')
            header_hashes[name] = digest
    if dgrad:
        # LATCH-TRAIN-CORRECTNESS: the deferred stage's header and everything it includes (loop_dgrad.header_closure);
        # regions without warpgroup bodies do not record p5c_wg.cuh and its includes themselves.
        for name, digest in loop_dgrad.header_closure().items():
            if name in header_hashes and header_hashes[name] != digest:
                raise LoweringError(f'loop region header identities differ: {name}')
            header_hashes[name] = digest
    source_hash = hashlib.sha256((source + json.dumps(header_hashes, sort_keys=True)).encode()).hexdigest()
    writes = dict(zip(inputs[:np_+no+ns], [structured['params'][k] for k in parameters]
                      + [structured[optimizer_group][k] for k in optimizer_names] + [structured['state'][k] for k in states]))
    manifest = {
        'format': 'latch-ops-cuda-loop-pipelined-v1', 'source_hash': source_hash, 'header_hashes': header_hashes,
        'inputs': inputs, 'outputs': outputs, 'structured_outputs': structured,
        'input_roles': {n: {'role': values[n].role, 'parameter': values[n].parameter} for n in inputs},
        'state_writes': writes, 'layouts': layouts, 'offsets': offsets,
        'optimizer_state_schema': metadata.get('optimizer_state_schema', []),
        'gradient_participation': 'device reverse structural propagation from runtime count' if connectivity else 'legacy SGD zero-gradient policy',
        'persistent_counters': {name: {'arena_word_offset': offset,
                                       'words': programs[name].manifest['persistent_counter_words']}
                                for name, offset in counter_area.items()},
        'barrier_storage': {'kind': options['barrier'], 'arena_word_offset': barrier_offset,
                            'initialization': 'one cooperative grid barrier per launch' if barrier_offset is not None else None,
                            'module_global_state': False},
        'arena_words': top, 'arena_bytes': 4*top, 'region_scratch_bytes': 4*scratch_words,
        'scratch_areas': {'X': [4*X[0], 4*X[1], 4*x_words], 'Y': [4*Y, 4*y_words], 'Z': 4*Z},
        'shared_bytes': shared_bytes,
        'boundary_tape_bytes': sum(stride*tape_size*4 for _, stride in boundaries),
        'body_tape_bytes': sum(stride*max(1, graph.max_steps)*4 for _, stride in body_tape),
        'init_tape_bytes': sum(math.prod(d['shape']) * _BYTES[d['dtype']] for d in metadata.get('init_tape', [])) if init_saved else 0,
        'retained_all': False, 'phase_count': 7, 'loop_options': options,
        'inplace_commit': inplace, 'splits': {'forward': split_f, 'forward_stages': S_F,
                                              'reverse': split_r, 'reverse_stages': S_R,
                                              **({'forward_region': forward_region, 'reverse_region': reverse_region}
                                                 if 'lptx' in features_of(options.get('group_rules')) else {})},
        **({'lpt_log': lpt_log} if 'lptx' in features_of(options.get('group_rules')) else {}),
        'tick_blocks': tick_layout, 'tick_words': tick_top, 'phase_events': phase_events,
        'profile_intervals': [
            {'start_tick': 0, 'end_tick': 1, 'kind': 'pipeline_A', 'name': 'initialization + first forward prefix'},
            {'start_tick': 1, 'end_tick': 2, 'kind': 'pipeline_forward', 'name': 'pipelined forward iterations'},
            {'start_tick': 2, 'end_tick': 3, 'kind': 'pipeline_B', 'name': 'last forward suffix + terminal VJP + first reverse prefix'},
            {'start_tick': 3, 'end_tick': 4, 'kind': 'pipeline_reverse', 'name': 'pipelined reverse iterations'},
            {'start_tick': 4, 'end_tick': 5, 'kind': 'pipeline_C', 'name': 'last reverse suffix + initial VJP + optimizer + state commit'}],
        'phases': [{'kind': n, 'names': [n]} for n in ('init', 'forward_loop', 'terminal_vjp', 'reverse_loop', 'init_vjp', 'optimizer')],
        'max_gemm_tiles': max(p.manifest['max_gemm_tiles'] for p in programs.values()),
        'regions': {n: p.manifest for n, p in programs.items()},
        'compiled_regions': list(region_names),
        'runtime_loop': {'max_steps': graph.max_steps, 'count_input': inputs[-1], 'storage_mode': graph.storage_mode,
                         'count_policy': 'clamp to [0,max_steps]; nonzero status exposes out-of-range request',
                         'body_backward': 'PyTorch joint VJP partition; staged regions merged by buffer dependencies'},
        'numerical_policy': programs[forward_region].manifest['numerical_policy'],
        'execution': 'one cooperative kernel; software-pipelined runtime forward/reverse loops; staged regions; one final commit',
        'memory_plan': 'disjoint stable outputs/cotangents/boundaries/tape; double-buffered loop-region scratch; one scratch area per concurrent straight-line region',
    }
    from .lower import GeneratedProgram
    if casts:
        manifest.update(invariant_casts=casts, invariant_cast_bytes=sum(
            math.prod(d['shape']) * _BYTES[d['dtype']] for d in cast_desc))
    if resource['has_popsb']:
        manifest['resource_policy'] = resource
    if tensor_maps:
        manifest.update(tensor_maps=tensor_maps, tensor_map_addresses=True,
                        index_checks='trap (assert overridden for wgmma programs)')
    if hoisted:
        manifest['step06_mask_hoist'] = hoisted
    return GeneratedProgram(source, manifest, kernel_body, helper_source, '\n'.join(functions))


def _region_tensor_maps(functions, function_region, tracks, programs, input_count, *, boundaries, body_tape,
                        tape_size, max_steps, body_regions, x_areas, reverse_tape=(), extra_maps=()):
    """P7-C: resolve wgmma operand-map placeholders of loop regions.

    A region's operand may be bound to different storage per region instance (boundary/tape slots, scratch parity,
    cotangent parity). Every possible binding of each placeholder becomes one map of the table; the code selects the
    map whose recorded address equals the operand pointer it receives (p5c_pick_map, trap on no match). Maps hold
    addresses/extents only."""
    import re
    from .p5c_matrix import TMAP_REGION
    fixed = re.compile(r'args\[(\d+)\]|arena \+ (\d+)LL')

    def parse(pointer):
        found = []
        for m in fixed.finditer(pointer):
            found.append(('input', int(m.group(1))) if m.group(1) is not None else ('arena', int(m.group(2))))
        return found

    def key_candidates(key, pointer):
        if key[0] == 'b':
            start, stride = boundaries[key[1]]
            return [('arena', start + i * stride) for i in range(tape_size)]
        if key[0] == 't':
            start, stride = body_tape[key[1]]
            return [('arena', start + i * stride) for i in range(max(1, max_steps))]
        if key[0] == 'rt':
            start, stride = reverse_tape[key[1]]
            return [('arena', start + i * stride) for i in range(max(1, max_steps))]
        found = parse(pointer)
        if not found:
            raise LoweringError(f'p5c region map: unresolved binding {key} -> {pointer}')
        return found

    per_region = {}
    for track in tracks:
        if track.region is None:
            continue
        entry = per_region.setdefault(track.region, {'args': {}, 'outs': {}, 'scratch': set()})
        for i, (key, pointer) in enumerate(track.args):
            entry['args'].setdefault(i, set()).update(key_candidates(key, pointer))
        for i, (key, pointer) in enumerate(track.outs):
            entry['outs'].setdefault(i, set()).update(key_candidates(key, pointer))
        if track.region in body_regions:
            entry['scratch'].update(x_areas)
        elif isinstance(track.scratch, int):
            entry['scratch'].add(track.scratch)
        elif track.scratch is not None:
            entry['scratch'].update(word for _, word in parse(str(track.scratch)))
    maps, index = [], {}
    marker = re.compile(re.escape(TMAP_REGION) + r'\(([^()|]*)\|([^()]*)\)')
    out = list(functions)
    for f, text in enumerate(functions):
        if TMAP_REGION + '(' not in text:
            continue
        region = function_region.get(f)
        if region is None:
            raise LoweringError('p5c region map placeholder outside a region function')
        mf, bind = programs[region].manifest, per_region.get(region)
        if bind is None:
            raise LoweringError(f'p5c region {region} has no emitted instance')
        placement = mf.get('output_placement', {})

        def replace(match):
            root, offset, d0, d1, stride, b0, b1, *_swizzle = match.group(1).split(',')
            pointer = match.group(2)
            if root in mf['inputs']:
                bases = sorted(bind['args'].get(mf['inputs'].index(root), ()))
            elif root in placement and placement[root]['mode'] in ('store', 'accumulate'):
                bases = sorted(bind['outs'].get(int(placement[root]['output']), ()))
            elif root in mf['offsets']:
                bases = sorted(('arena', area + int(mf['offsets'][root])) for area in bind['scratch'])
            else:
                raise LoweringError(f'p5c region map: operand root {root} of {region} has no storage')
            if not bases:
                raise LoweringError(f'p5c region map: no binding for {root} of {region}')
            first = None
            for storage, where in bases:
                spec = (storage, where, int(offset), int(d0), int(d1), int(stride), int(b0), int(b1))
                if spec not in index:
                    index[spec] = len(maps)
                    maps.append({'root': f'{region}:{root}', 'storage': storage,
                                 **({'index': where} if storage == 'input' else {'word_offset': where}),
                                 'element_offset': int(offset), 'dims': [int(d0), int(d1)], 'stride_bytes': int(stride),
                                 'box': [int(b0), int(b1)], 'dtype': 'bfloat16', 'swizzle': 128, 'oob_fill': 'zero'})
                first = index[spec] if first is None else first
            ids = [index[(storage, where, int(offset), int(d0), int(d1), int(stride), int(b0), int(b1))]
                   for storage, where in bases]
            if ids != list(range(ids[0], ids[0] + len(ids))):
                # Maps of one placeholder are contiguous unless an identical map was created earlier: add copies.
                ids = []
                for storage, where in bases:
                    maps.append({**maps[index[(storage, where, int(offset), int(d0), int(d1), int(stride), int(b0),
                                               int(b1))]]})
                    ids.append(len(maps) - 1)
            return f'p5c_pick_map({ids[0]}, {len(ids)}, (const void*)({pointer} + {int(offset)}))'
        out[f] = marker.sub(replace, text)
        if TMAP_REGION + '(' in out[f]:
            raise LoweringError('unresolved p5c region map placeholder')
    extra_base = len(maps)
    maps.extend(dict(m) for m in extra_maps)     # LATCH-STEP-06: fixed maps addressed by index (extra_base + i)
    total = len(maps)
    out = [text.replace('p5c_pick_map(', f'p5c_pick_map({total}, ') for text in out]
    return out, maps, extra_base
