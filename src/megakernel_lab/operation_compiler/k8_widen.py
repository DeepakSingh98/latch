"""Store the exactly BF16-rounded k8 result directly into its FP32 widening.

This is the same storage optimization as matrix_widen for the explicit k8
arithmetic body. It proves a sole, identity FP32 widening and leaves every
later operation and every BF16 rounding boundary intact.
"""
from __future__ import annotations

import re

ENABLED = True


def fuse(phases, layouts, dtypes, *, ident, index, kept, inplace_roots):
    if not ENABLED:
        return phases, [], set()
    root = lambda name: layouts[name].root if name in layouts else name
    readers = {}
    for i, phase in enumerate(phases):
        for name in phase['reads']:
            readers.setdefault(root(name), set()).add(i)
    replaced, removed, dead, records = {}, set(), set(), []
    for i, original in enumerate(phases):
        phase = replaced.get(i, original)
        spec = phase.get('k8_widen_respec')
        if (phase['kind'] != 'gemm' or not spec or not spec.get('enabled')
                or len(phase['writes']) != 1):
            continue
        source = phase['writes'][0]
        shape = tuple(spec['shape'])
        if (source not in layouts or root(source) != source or source in kept
                or source in inplace_roots or dtypes.get(source) != 'bfloat16'
                or tuple(layouts[source].shape) != shape or layouts[source].offset
                or not layouts[source].contiguous or spec['pitch'] != shape[-1]):
            continue
        users = readers.get(source, set())
        if len(users) != 1:
            continue
        j = next(iter(users))
        consumer = replaced.get(j, phases[j])
        if (j <= i or j in removed or consumer['kind'] != 'elementwise'
                or not consumer.get('names') or not consumer.get('body')
                or len(consumer['body']) != len(consumer['names'])
                or consumer.get('targets', [None])[0] != 'aten._to_copy.default'):
            continue
        out = consumer['names'][0]
        if (out not in layouts or root(out) != out or out in inplace_roots
                or dtypes.get(out) != 'float32' or tuple(layouts[out].shape) != shape
                or not layouts[out].contiguous or layouts[out].offset):
            continue
        source_index = index(layouts[source], shape)
        expected = (f'const float r_{out} = __bfloat162float({ident(source)}[{source_index}]); '
                    f'{ident(out)}[i] = r_{out};')
        compact = lambda text: re.sub(r'\s+', '', text)
        if compact(consumer['body'][0]) != compact(expected):
            continue
        # Remaining consumer lines may use the widened register, but no
        # register or pointer naming the BF16 storage being eliminated. The
        # sole phase reader can still contain a second use through an alias.
        private_names = {source} | {name for name in layouts if root(name) == source}
        private_tokens = {token for name in private_names for token in (f'r_{name}', ident(name))}
        private_use = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(private_tokens))) + r')\b')
        if any(private_use.search(line) for line in consumer['body'][1:]):
            continue
        output_text = f'g.C = reinterpret_cast<s101h::bf16*>({spec["output"]});'
        if phase['code'].count(output_text) != 1 or phase['code'].count('/*K8_ROUNDED_F32*/false') != 1:
            continue
        code = phase['code'].replace(output_text,
            f'g.C = reinterpret_cast<s101h::bf16*>({ident(out)});').replace(
            '/*K8_ROUNDED_F32*/false', '/*K8_ROUNDED_F32*/true')
        record = dict(source=source, output=out, shape=list(shape), product_stored=False,
            arithmetic='unchanged ascending k8 product; BF16 RN then exact FP32 widening',
            avoided_bf16_store_and_load_bytes=4 * layouts[source].numel)
        replacement = dict(phase, code=code, names=[*phase['names'], out],
            targets=[*phase.get('targets', []), 'aten._to_copy.default'], writes=[out],
            k8_widening=record,
            arithmetic=phase.get('arithmetic', '') + '; exact FP32 widening after BF16 RN')
        replacement.pop('k8_widen_respec', None)
        # Matrix-spec output bookkeeping is not a rescheduling permission:
        # this phase now owns the FP32 destination while its bound contract
        # continues to describe the original BF16-rounded product.
        replacement['matrix_spec'] = {**phase['matrix_spec'], 'out': ident(out)}
        replaced[i] = replacement
        if len(consumer['names']) == 1:
            removed.add(j)
        else:
            widened_load = f'{ident(out)}[{index(layouts[out], shape)}]'
            body = [re.sub(r'\br_' + re.escape(out) + r'\b', widened_load, line)
                    for line in consumer['body'][1:]]
            registers = dict(consumer.get('registers', {}))
            registers.pop(out, None)
            replaced[j] = dict(consumer, names=consumer['names'][1:], targets=consumer['targets'][1:],
                writes=[name for name in consumer['writes'] if name != out],
                reads=list(dict.fromkeys([name for name in consumer['reads'] if name != source] + [out])),
                body=body, registers=registers)
        dead.add(source)
        records.append(record)
    return [replaced.get(i, phase) for i, phase in enumerate(phases) if i not in removed], records, dead
