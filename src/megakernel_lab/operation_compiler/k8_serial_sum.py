"""Fuse a proved private ordered sum of contiguous BF16 k8 products.

This consumes the existing graph-bound arithmetic declaration and verifies the
rendered matrices, widenings and additions. The caller's partial boundaries,
BF16 roundings and ascending FP32 addition sequence remain unchanged. Each CTA
owns one output tile through the entire sequence, retaining partials in shared
memory instead of materializing every full matrix in global memory.
"""
from __future__ import annotations

import re

ENABLED = True
HEADER = 'k8_serial_sum.cuh'


def fuse(phases, layouts, dtypes, contract, *, ident, index, kept, inplace_roots):
    if not ENABLED or not contract:
        return phases, [], set()
    root = lambda n: layouts[n].root if n in layouts else n
    compact = lambda t: re.sub(r'\s+', '', t)
    producer = {n: i for i, phase in enumerate(phases) for n in phase['writes']}
    readers = {}
    for i, phase in enumerate(phases):
        for name in phase['reads']:
            readers.setdefault(root(name), set()).add(i)
    replaced, removed, dead, records = {}, set(), set(), []
    for weight in contract.get('weight_gradients', []):
        partials = weight['partials']
        widened = weight['widened_partials']
        sums = weight['subtotal_additions']
        chunk = weight['chunk_rows']
        if (len(partials) < 2 or len(widened) != len(partials) or len(sums) != len(partials)-1
                or chunk % 16 or weight.get('source_only_removed_subtotals')):
            continue
        selected = set(partials + widened + sums)
        if any(name not in producer or name not in layouts or root(name) != name for name in selected):
            continue
        members = {producer[name] for name in selected}
        final = sums[-1]
        private = selected - {final}
        if (members & removed or private & kept or selected & inplace_roots
                or any(phases[i].get('active_values') for i in members)
                or any(readers.get(name, set()) - members for name in private)):
            continue
        shape = tuple(layouts[final].shape)
        if (len(shape) != 2 or any(tuple(layouts[n].shape) != shape or not layouts[n].contiguous
                                     or layouts[n].offset for n in selected)
                or any(dtypes[n] != ('bfloat16' if n in partials else 'float32') for n in selected)):
            continue
        bindings, matrix_phases = [], []
        ok = True
        for partial in partials:
            phase = phases[producer[partial]]
            binding = phase.get('serial_bf16_contract', {})
            if (phase['kind'] != 'gemm' or phase['writes'] != [partial]
                    or phase.get('engine') != 'latch_matrix:mma_k8_readout'
                    or binding != contract.get('products', {}).get(partial)
                    or binding.get('role') != 'weight_partial' or binding.get('partition_k') != 0
                    or phase.get('k8_widen_respec', {}).get('pitch') != shape[-1]):
                ok = False
                break
            bindings.append(binding)
            matrix_phases.append(producer[partial])
        if not ok:
            continue
        first = bindings[0]
        la, lb = layouts[first['a']], layouts[first['b']]
        if la.root in inplace_roots or lb.root in inplace_roots:
            continue
        ak, bkc = la.strides[-1] == 1, lb.strides[-2] == 1
        end = 0
        for binding in bindings:
            start, stop = binding.get('token_interval', (-1, -1))
            a, b = layouts[binding['a']], layouts[binding['b']]
            if (start != end or stop <= start or stop-start > chunk
                    or (end and start % chunk) or binding['shape'] != [*shape,stop-start]
                    or a.root != la.root or b.root != lb.root or a.strides != la.strides or b.strides != lb.strides
                    or a.offset != la.offset+start*la.strides[-1]
                    or b.offset != lb.offset+start*lb.strides[-2]):
                ok = False
                break
            end = stop
        if not ok:
            continue
        # Validate the exact emitted scalar chain. Register loads are admitted
        # only when defined earlier in that same phase; no expression matching
        # relies on graph-generated names or names of model parameters.
        expected = {wide: ('cast', partial) for partial, wide in zip(partials, widened)}
        prior = widened[0]
        for total, wide in zip(sums, widened[1:]):
            expected[total] = ('add', prior, wide)
            prior = total
        for member in sorted(members - set(matrix_phases)):
            phase = phases[member]
            if phase['kind'] != 'elementwise' or len(phase.get('body', [])) != len(phase['names']):
                ok = False
                break
            regs = set()
            for name, target, line in zip(phase['names'], phase['targets'], phase['body']):
                if name in selected:
                    def read(value):
                        return f'r_{value}' if value in regs else f'{ident(value)}[{index(layouts[value],shape)}]'
                    rule = expected.get(name)
                    if rule is None:
                        ok = False
                        break
                    if rule[0] == 'cast':
                        expression = f'__bfloat162float({read(rule[1])})'
                        want = 'aten._to_copy.default'
                    else:
                        expression = f'({read(rule[1])} + ({read(rule[2])} * 1.0f))'
                        want = 'aten.add.Tensor'
                    rendered = f'const float r_{name} = {expression}; {ident(name)}[i] = r_{name};'
                    if target != want or compact(line) != compact(rendered):
                        ok = False
                        break
                regs.add(name)
            if not ok:
                break
        if not ok:
            continue
        # Unabsorbed statements can reload a private result through either its
        # materialized pointer or a storage alias, as well as a local register.
        # Phase-level reader sets cannot distinguish these uses from the chain
        # being removed, so reject any such token in a surviving statement.
        private_names = private | {n for n in layouts if root(n) in private}
        private_tokens = {token for n in private_names for token in (f'r_{n}', ident(n))}
        private_use = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(private_tokens))) + r')\b')
        for member in members - set(matrix_phases):
            phase = phases[member]
            leftover = [line for name,line in zip(phase['names'],phase['body']) if name not in selected]
            if any(private_use.search(line) for line in leftover):
                ok = False
                break
        if not ok:
            continue
        at = max(matrix_phases)
        # All removed scalar definitions must precede the final matrix or be
        # part of its downstream scalar phase. Any unrelated earlier reader of
        # the final sum would make delaying the sequence unsafe.
        if any(i < at for i in readers.get(final, set())):
            continue
        from .matrix_adapters import mma_k8 as k8_body
        wgmma = k8_body.WIDE and k8_body.WGMMA and 'weight_partial' in k8_body.WGMMA_ROLES
        if wgmma:
            # LATCH-MOE-EMA-FAST: the same ordered partial sum, accumulated by wgmma zero-half pairs with the
            # running FP32 total held in registers (kernels/wgmma_k8.cuh phase_serial_sum).
            shared = k8_body.wgmma_shared(ak, 128)
            if k8_body.WGMMA_TMA:
                ta, tb = k8_body.tma_markers(la.root, la.offset, ak, la.strides[-2] if ak else la.strides[-1],
                                             lb.root, lb.offset, bkc, lb.strides[-1] if bkc else lb.strides[-2],
                                             shape[0], shape[1], end, 128)
                call = (f'latch_wgk8::phase_serial_sum_tma<{str(ak).lower()},{str(bkc).lower()},{chunk},{end}>'
                        f'(g,{ta},{tb},smem);')
            else:
                call = f'latch_wgk8::phase_serial_sum<{str(ak).lower()},{str(bkc).lower()},{chunk},{end}>(g,smem);'
        else:
            shared = 8 * ((128*40 if ak else 32*136) + (128*40 if bkc else 32*136)) + 2*128*136*4
            call = f'latch_mma_k8::phase_serial_sum<{str(ak).lower()},{str(bkc).lower()},{chunk}>(g,smem);'
        tiles = ((shape[0]+127)//128)*((shape[1]+127)//128)
        code = f'''{{
  popsc::BArgs g{{}};
  g.A=reinterpret_cast<const s101h::bf16*>({ident(la.root)})+{la.offset}LL;
  g.B=reinterpret_cast<const s101h::bf16*>({ident(lb.root)})+{lb.offset}LL;
  g.C=reinterpret_cast<s101h::bf16*>({ident(final)});
  g.sam={la.strides[-2]}LL; g.sak={la.strides[-1]}LL; g.sbk={lb.strides[-2]}LL; g.sbn={lb.strides[-1]}LL;
  g.M={shape[0]}; g.N={shape[1]}; g.K={end}; g.ldc={shape[1]}; g.nb=1;
  {call}
}}'''
        record = dict(output=final, partials=list(partials), widened_partials=list(widened),
            sums=list(sums), chunk_rows=chunk, shape=[*shape,end], shared_bytes=shared,
            arithmetic='each original ascending-k8 partial rounded BF16; exact widening; ascending FP32 RN additions without initial +0',
            instruction=('mma.sync.m16n8k8 arithmetic issued as wgmma.m64n128k16 register-A zero-half pairs'
                         if wgmma else 'mma.sync.m16n8k8'),
            global_intermediates_elided=sorted(private))
        fused = dict(phases[at], names=[final], writes=[final], targets=['aten.add.Tensor'], code=code,
            reads=list(dict.fromkeys(v for binding in bindings for v in (binding['a'],binding['b']))),
            engine='latch_matrix:mma_k8_serial_sum', shared_bytes=shared, tiles=tiles, work_items=tiles,
            k=end, matrix_shape=[*shape,end], k8_serial_sum=record,
            arithmetic=record['arithmetic'], matrix_spec=dict(engine='latch_matrix:mma_k8_serial_sum',
                a=first['a'],b=first['b'],out=ident(final),m=shape[0],n=shape[1],k=end,batch=1,
                roots=[(la.root,int(la.offset)),(lb.root,int(lb.offset))]))
        for key in ('k8_widen_respec','serial_bf16_contract','matrix_plan','latch_matrix'):
            fused.pop(key, None)
        replaced[at] = fused
        removed.update(set(matrix_phases)-{at})
        for member in members - set(matrix_phases):
            phase = phases[member]
            keep_positions = [i for i,n in enumerate(phase['names']) if n not in selected]
            if not keep_positions:
                removed.add(member)
                continue
            # Only the final sum can flow to an unabsorbed operation; private
            # intermediates have no users outside this proven chain.
            body = [re.sub(r'\br_' + re.escape(final) + r'\b',
                           f'{ident(final)}[{index(layouts[final],shape)}]', phase['body'][i])
                    for i in keep_positions]
            if any(private_use.search(line) for line in body):
                raise ValueError('private k8 subtotal unexpectedly read by an unabsorbed operation')
            replaced[member] = dict(phase, names=[phase['names'][i] for i in keep_positions],
                writes=[n for n in phase['writes'] if n not in selected],
                targets=[phase['targets'][i] for i in keep_positions],body=body,
                reads=list(dict.fromkeys([n for n in phase['reads'] if root(n) not in private]+[final])),
                registers={n:r for n,r in phase.get('registers', {}).items() if n not in selected})
        dead.update(private)
        records.append(record)
    return [replaced.get(i,p) for i,p in enumerate(phases) if i not in removed], records, dead


# LATCH-MOE-EMA-100: flat shared-readout accumulation chains (serial_bf16 weight_accumulation_groups,
# 'last-forward-first'): prefix + w_h1_0 + ... + w_h1_15 + w_h2_0 + ... + w_h2_15. Where one head's partial products
# and the chain's additions are lowered in the same region, while the prefix total and the remaining heads' widened
# partials are stored region inputs, one fused phase runs the head's ascending chunk partials on a running FP32 total
# that starts at the prefix (phase_serial_chain_tma) and then adds the stored partials in chain order. Every
# product, rounding and addition, and their order, are the chain's; partials and intermediate sums lose storage.
CHAINS = True


def fuse_chains(phases, layouts, dtypes, contract, *, ident, index, kept, inplace_roots):
    if not CHAINS or not contract:
        return phases, [], set()
    from .matrix_adapters import mma_k8 as k8_body
    if not (k8_body.WIDE and k8_body.WGMMA and 'weight_partial' in k8_body.WGMMA_ROLES and k8_body.WGMMA_TMA):
        return phases, [], set()
    root = lambda n: layouts[n].root if n in layouts else n
    compact = lambda t: re.sub(r'\s+', '', t)
    producer = {n: i for i, phase in enumerate(phases) for n in phase['writes']}
    readers = {}
    for i, phase in enumerate(phases):
        for name in phase['reads']:
            readers.setdefault(root(name), set()).add(i)
    by_widened = {}
    for weight in contract.get('weight_gradients', []):
        for i, name in enumerate(weight.get('widened_partials', [])):
            by_widened[name] = (weight, i)
    replaced, removed, dead, records = {}, set(), set(), []
    for group in contract.get('weight_accumulation_groups', []):
        additions = list(group.get('additions') or [])
        if group.get('single_head_unchanged') or len(additions) < 2:
            continue
        prefix = group.get('prefix_subtotal')
        flat = list(group.get('widened_partials', []))[-len(additions):]
        first = by_widened.get(flat[0])
        if prefix is None or first is None or first[1] != 0:
            continue
        weight = first[0]
        partials, widened, chunk = list(weight['partials']), list(weight['widened_partials']), weight['chunk_rows']
        n = len(widened)
        if len(partials) != n or n < 2 or flat[:n] != widened or chunk % 16:
            continue
        extras = flat[n:]
        if (any(x not in producer for x in partials + widened + additions)
                or prefix in producer or any(x in producer for x in extras)
                or any(x not in layouts for x in [prefix, *extras])):
            continue
        final = additions[-1]
        selected = set(partials + widened + additions)
        private = selected - {final}
        members = {producer[x] for x in selected}
        if (members & removed or private & kept or selected & inplace_roots
                or any(root(x) != x for x in selected)
                or any(phases[i].get('active_values') for i in members)
                or any(readers.get(x, set()) - members for x in private)):
            continue
        shape = tuple(layouts[final].shape)
        if (len(shape) != 2 or any(tuple(layouts[x].shape) != shape or not layouts[x].contiguous or layouts[x].offset
                                   for x in [*selected, prefix, *extras])
                or any(dtypes[x] != ('bfloat16' if x in partials else 'float32')
                       for x in [*selected, prefix, *extras])):
            continue
        bindings, matrix_phases, ok = [], [], True
        for partial in partials:
            phase = phases[producer[partial]]
            binding = phase.get('serial_bf16_contract', {})
            if (phase['kind'] != 'gemm' or phase['writes'] != [partial]
                    or phase.get('engine') != 'latch_matrix:mma_k8_readout'
                    or binding != contract.get('products', {}).get(partial)
                    or binding.get('role') != 'weight_partial' or binding.get('partition_k') != 0
                    or phase.get('k8_widen_respec', {}).get('pitch') != shape[-1]):
                ok = False
                break
            bindings.append(binding)
            matrix_phases.append(producer[partial])
        if not ok:
            continue
        la, lb = layouts[bindings[0]['a']], layouts[bindings[0]['b']]
        if la.root in inplace_roots or lb.root in inplace_roots:
            continue
        ak, bkc = la.strides[-1] == 1, lb.strides[-2] == 1
        end = 0
        for binding in bindings:
            start, stop = binding.get('token_interval', (-1, -1))
            a, b = layouts[binding['a']], layouts[binding['b']]
            if (start != end or stop <= start or stop-start > chunk or (end and start % chunk)
                    or binding['shape'] != [*shape, stop-start] or a.root != la.root or b.root != lb.root
                    or a.strides != la.strides or b.strides != lb.strides
                    or a.offset != la.offset+start*la.strides[-1] or b.offset != lb.offset+start*lb.strides[-2]):
                ok = False
                break
            end = stop
        if not ok:
            continue
        expected = {wide: ('cast', partial) for partial, wide in zip(partials, widened)}
        prior = prefix
        for total, addend in zip(additions, flat):
            expected[total] = ('add', prior, addend)
            prior = total
        for member in sorted(members - set(matrix_phases)):
            phase = phases[member]
            if phase['kind'] != 'elementwise' or len(phase.get('body', [])) != len(phase['names']):
                ok = False
                break
            regs = set()
            for name, target, line in zip(phase['names'], phase['targets'], phase['body']):
                if name in selected:
                    def read(value):
                        return f'r_{value}' if value in regs else f'{ident(value)}[{index(layouts[value], shape)}]'
                    rule = expected.get(name)
                    if rule is None:
                        ok = False
                        break
                    if rule[0] == 'cast':
                        expression, want = f'__bfloat162float({read(rule[1])})', 'aten._to_copy.default'
                    else:
                        expression, want = f'({read(rule[1])} + ({read(rule[2])} * 1.0f))', 'aten.add.Tensor'
                    rendered = f'const float r_{name} = {expression}; {ident(name)}[i] = r_{name};'
                    if target != want or compact(line) != compact(rendered):
                        ok = False
                        break
                regs.add(name)
            if not ok:
                break
        if not ok:
            continue
        private_names = private | {x for x in layouts if root(x) in private}
        private_tokens = {token for x in private_names for token in (f'r_{x}', ident(x))}
        private_use = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(private_tokens))) + r')\b')
        if any(private_use.search(line) for member in members - set(matrix_phases)
               for name, line in zip(phases[member]['names'], phases[member]['body']) if name not in selected):
            continue
        at = max(matrix_phases)
        if any(i < at for i in readers.get(final, set())):
            continue
        shared = k8_body.wgmma_shared(ak, 128)
        ta, tb = k8_body.tma_markers(la.root, la.offset, ak, la.strides[-2] if ak else la.strides[-1],
                                     lb.root, lb.offset, bkc, lb.strides[-1] if bkc else lb.strides[-2],
                                     shape[0], shape[1], end, 128)
        table = ', '.join(ident(x) for x in extras) or 'nullptr'
        code = f'''{{
  popsc::BArgs g{{}};
  g.A=reinterpret_cast<const s101h::bf16*>({ident(la.root)})+{la.offset}LL;
  g.B=reinterpret_cast<const s101h::bf16*>({ident(lb.root)})+{lb.offset}LL;
  g.C=reinterpret_cast<s101h::bf16*>({ident(final)});
  g.sam={la.strides[-2]}LL; g.sak={la.strides[-1]}LL; g.sbk={lb.strides[-2]}LL; g.sbn={lb.strides[-1]}LL;
  g.M={shape[0]}; g.N={shape[1]}; g.K={end}; g.ldc={shape[1]}; g.nb=1;
  const float* const chain_extra_[{max(1, len(extras))}] = {{{table}}};
  latch_wgk8::phase_serial_chain_tma<{str(ak).lower()},{str(bkc).lower()},{chunk},{end},{len(extras)}>(g,{ta},{tb},smem,{ident(prefix)},chain_extra_);
}}'''
        record = dict(output=final, prefix=prefix, partials=partials, widened_partials=widened, additions=additions,
                      stored_addends=extras, chunk_rows=chunk, shape=[*shape, end], shared_bytes=shared,
                      arithmetic='prefix, then each ascending-k8 partial rounded BF16 and exactly widened, then the '
                                 'stored widened partials, added left to right in FP32 RN in the chain order',
                      global_intermediates_elided=sorted(private))
        tiles = ((shape[0]+127)//128)*((shape[1]+127)//128)
        fused = dict(phases[at], names=[final], writes=[final], targets=['aten.add.Tensor'], code=code,
                     reads=list(dict.fromkeys([v for binding in bindings for v in (binding['a'], binding['b'])]
                                              + [prefix, *extras])),
                     engine='latch_matrix:mma_k8_serial_sum', shared_bytes=shared, tiles=tiles, work_items=tiles,
                     k=end, matrix_shape=[*shape, end], k8_serial_chain=record, arithmetic=record['arithmetic'],
                     matrix_spec=dict(engine='latch_matrix:mma_k8_serial_sum', a=bindings[0]['a'], b=bindings[0]['b'],
                                      out=ident(final), m=shape[0], n=shape[1], k=end, batch=1,
                                      roots=[(la.root, int(la.offset)), (lb.root, int(lb.offset))]))
        for key in ('k8_widen_respec', 'serial_bf16_contract', 'matrix_plan', 'latch_matrix'):
            fused.pop(key, None)
        replaced[at] = fused
        removed.update(set(matrix_phases) - {at})
        for member in members - set(matrix_phases):
            phase = phases[member]
            keep_positions = [i for i, x in enumerate(phase['names']) if x not in selected]
            if not keep_positions:
                removed.add(member)
                continue
            body = [re.sub(r'\br_' + re.escape(final) + r'\b', f'{ident(final)}[{index(layouts[final], shape)}]',
                           phase['body'][i]) for i in keep_positions]
            if any(private_use.search(line) for line in body):
                raise ValueError('private chain value unexpectedly read by an unabsorbed operation')
            replaced[member] = dict(phase, names=[phase['names'][i] for i in keep_positions],
                                    writes=[x for x in phase['writes'] if x not in selected],
                                    targets=[phase['targets'][i] for i in keep_positions], body=body,
                                    reads=list(dict.fromkeys([x for x in phase['reads'] if root(x) not in private]
                                                             + [final])),
                                    registers={k: r for k, r in phase.get('registers', {}).items()
                                               if k not in selected})
        dead.update(private)
        records.append(record)
    return [replaced.get(i, p) for i, p in enumerate(phases) if i not in removed], records, dead
