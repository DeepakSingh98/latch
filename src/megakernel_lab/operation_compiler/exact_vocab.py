"""Elide private unpadded CE intermediates while retaining the CUDA ILP4 contract.

The frozen ROW donor c2a325e441 separates statistics, selected log-probability,
and gradient rows. This adapter uses that storage strategy only; the emitter
retains the corrected torch_cuda_logsoftmax_ilp4_v1 arithmetic instead of the
donor's online reduction or padded matrix extent. No model-name dispatch.
"""
from __future__ import annotations

import copy
import math

from . import vocab_head as head
from .torch_logsoftmax import CONTRACT as LOG_SOFTMAX_CONTRACT

EXACT_ROWS = True
KEY = 'exact_vocab_rows'
TARGETS = {kind: f'latch_ops.exact_vocab_{kind}.default' for kind in ('stats', 'pick', 'grad')}


def _names(item):
    """Inventory nested output and state-write names, including string leaves."""
    if isinstance(item, str):
        return {item}
    if isinstance(item, dict):
        return set().union(*(_names(v) for v in item.values())) if item else set()
    if isinstance(item, (list, tuple)):
        return set().union(*(_names(v) for v in item)) if item else set()
    return set()


def _escaped(data):
    result = _names(data.get('outputs', {})) | _names(data.get('state_writes', {}))
    result.update(data.get('state_writes', {}))
    result.update(name for name, value in data['values'].items() if value.get('output_roles'))
    return result


def _dense(value, shape, dtype):
    return (value.get('dtype') == dtype and list(value.get('shape', ())) == list(shape)
            and list(value.get('strides', head._contiguous(shape))) == head._contiguous(shape)
            and not value.get('alias_of') and int(value.get('storage_offset', 0)) == 0)


def _match(data, op):
    plan = head._match(data, op)
    if plan is None:
        raise head.Refused('below vocabulary admission threshold')
    rows, vocab = plan['rows'], plan['vocab']
    if (type(rows) is not int or rows <= 0 or type(vocab) is not int
            or vocab <= 9216 or vocab % 4 == 0 or vocab >= 2**31):
        raise head.Refused('exact rows require unpadded rank-2 regular ILP4 width>9216 and width%4!=0')
    if plan['dtype'] != 'bfloat16' or plan['cast_op'] is None:
        raise head.Refused('exact rows require an explicit BF16 product widened to FP32')
    values = data['values']
    shape = [rows, vocab]
    for name, dtype in ((plan['product'], 'bfloat16'), (plan['logits'], 'float32'),
                        (op['name'], 'float32')):
        if not _dense(values[name], shape, dtype):
            raise head.Refused('exact rows require material contiguous offset-zero logits and log-probabilities')
    gather = plan['gather']
    index = head._ref(gather['args'][2])
    if values[index]['dtype'] != 'int64' or not _dense(values[gather['name']], [rows,1], 'float32'):
        raise head.Refused('exact pick requires int64 row indices and a dense FP32 result')
    remove_ops = [plan['cast_op'], op]
    if plan['back'] is not None:
        back = plan['back']
        if str(head._arg(back, 3, 'input_dtype', None)).removeprefix('torch.') != 'float32':
            raise head.Refused('exact log-softmax backward input dtype must be FP32')
        if '/' in plan['chain']:
            raise head.Refused('division in a sparse gradient chain is not admitted')
        if set(plan['chain']) - {'*', '~'}:
            raise head.Refused('unsupported sparse gradient chain')
        zeros = plan['leaf']['zeros']
        if zeros['target'] in head._FULL_FILLS or zeros['target'] == 'aten.fill.Scalar':
            fill = (zeros['args'][1] if zeros['target'] in ('aten.full.default', 'aten.fill.Scalar')
                    else zeros['args'][-1])
            if math.copysign(1.0, float(fill)) < 0:
                raise head.Refused('exact sparse rows require positive-zero destination fill')
        if not _dense(values[plan['dlogits']], shape, 'bfloat16'):
            raise head.Refused('exact gradient requires material contiguous BF16 output')
        for name in [back['name'], head._ref(back['args'][0]), plan['leaf']['zeros']['name'],
                     *[item['name'] for item in plan['chain_ops']]]:
            if not _dense(values[name], shape, 'float32'):
                raise head.Refused('exact sparse gradient chain requires dense offset-zero FP32 tensors')
        remove_ops += [plan['leaf']['zeros'], plan['leaf']['scatter'], *plan['chain_ops'], back]
    removed = {item['name'] for item in remove_ops}
    escaped = _escaped(data)
    if removed & escaped:
        raise head.Refused('an eliminated CE intermediate escapes through outputs or state writes')
    allowed = removed | {gather['name']}
    if plan['back'] is not None:
        allowed.add(plan['dcast']['name'])
    # Legacy matching alone does not check terminal G readers, arbitrary output
    # nesting, state writes, or passive aliases. Require every eliminated value
    # to be private to this exact matched subgraph.
    for item in data['operations']:
        if removed.intersection(head._refs(item.get('args', ())) + head._refs(item.get('kwargs', {}))):
            if item['name'] not in allowed:
                raise head.Refused('an eliminated CE intermediate has another consumer')
    for name, value in values.items():
        if value.get('alias_of') in removed and name not in removed:
            raise head.Refused('an eliminated CE intermediate has an escaping alias')
    for item in [*remove_ops, gather, *([plan['dcast']] if plan['back'] else [])]:
        if (item.get('effect', 'pure') != 'pure'
                or list(item.get('outputs') or [item['name']]) != [item['name']]):
            raise head.Refused('exact rows require pure single-output matched operations')
    plan['removed'] = removed
    return plan


def _operation(data, name, target, args, kwargs):
    refs = list(dict.fromkeys(head._refs(args) + head._refs(kwargs)))
    return dict(name=name, target=target, args=args, kwargs=kwargs, inputs=refs, outputs=[name],
                dependencies=[r for r in refs if r not in data.get('inputs', ())], reads=refs,
                writes=[name], reduction_axes=[], synchronization='global_phase_boundary', effect='pure')


def _apply(data, plan):
    product, rows, vocab = plan['product'], plan['rows'], plan['vocab']
    stats = 'exact_vocab_stats_' + product
    if stats in data['values']:
        raise head.Refused('exact vocabulary statistics name already exists')
    head._new_value(data, stats, [rows,2], 'float32')
    ref = lambda name: {'ref': name}
    stats_op = _operation(data, stats, TARGETS['stats'], [ref(product)], {'vocab': vocab})
    gather = plan['gather']
    pick = _operation(data, gather['name'], TARGETS['pick'],
                      [ref(product), ref(stats), gather['args'][2]], {'vocab': vocab})
    replacements = {plan['ls_op']['name']: stats_op, gather['name']: pick}
    if plan['back'] is not None:
        leaf = plan['leaf']
        supplied = ref(leaf['source']) if 'source' in leaf else leaf['value']
        args = [ref(product), ref(stats), ref(leaf['index']), supplied,
                *[ref(name) if name is not None else None for name in plan['factors']]]
        kwargs = dict(vocab=vocab, chain=plan['chain'], leaf_add=bool(leaf.get('add', False)))
        if 'value' in leaf:
            kwargs['leaf_value'] = leaf['value']
        replacements[plan['dlogits']] = _operation(data, plan['dlogits'], TARGETS['grad'], args, kwargs)
    data['operations'] = [replacements.get(op['name'], op) for op in data['operations']
                          if op['name'] not in plan['removed'] or op['name'] in replacements]
    for name in plan['removed']:
        del data['values'][name]
    return dict(product=product, stats=stats, pick=gather['name'], dlogits=plan.get('dlogits'),
                rows=rows, vocabulary=vocab, padded=False, chain=plan.get('chain', ''),
                log_softmax_source=plan['ls_op']['name'], removed=sorted(plan['removed']),
                arithmetic=LOG_SOFTMAX_CONTRACT, logical_fp32_alignment='(row*vocabulary)%4')


# LATCH-MOE-EMA-FAST: the sparse-cotangent operands of an exact gradient (target mask, loss scale) are ordered
# before its statistics when they do not depend on them, so the statistics row pass can also emit the gradient
# (exact_vocab_emit.fuse_rows). A scalar ones_like reads its input only for shape/dtype; within that operand
# chain it becomes the same constant without the input edge. Only operation order and that edge change.
HOIST_GRADIENT_OPERANDS = True
_COLLECTIVES = ('latch_ops.distributed_redistribute.default', 'latch_ops.distributed_sum.default')


def _hoist_gradient_operands(data, record):
    ops = data['operations']
    pos = {name: i for i, op in enumerate(ops) for name in (op.get('outputs') or [op['name']])}
    producer = {name: op for op in ops for name in (op.get('outputs') or [op['name']])}
    stats, grad = record['stats'], record.get('dlogits')
    if grad is None or stats not in pos or grad not in pos:
        return dict(refused='no gradient or statistics operation')
    first = pos[stats]
    gradient = producer[grad]
    if any(op.get('target') in _COLLECTIVES for op in ops[first:pos[grad]]):
        # A gradient in a later region would become a cross-region boundary value (logical row pitch, which
        # the k8 readout operands do not admit); it stays with its own products.
        return dict(refused='a collective separates the statistics from the gradient')
    closure, constants, seen = [], [], set()
    stack = [name for name in gradient.get('inputs', ()) if name not in (stats, record['product'])]
    while stack:
        name = stack.pop()
        if name in seen or name not in producer:
            continue
        seen.add(name)
        if pos[name] < first:
            continue
        if name == stats:
            return dict(refused='operand chain reads the statistics', chain=closure)
        op = producer[name]
        if (op.get('effect', 'pure') != 'pure' or list(op.get('outputs') or [name]) != [name]
                or op.get('target') in _COLLECTIVES):
            return dict(refused='impure, multi-output or collective operand producer', operation=name)
        value = data['values'].get(name, {})
        if (op['target'] == 'aten.ones_like.default' and value.get('shape') is not None
                and len(value['shape']) == 0 and value.get('dtype') == 'float32' and not value.get('alias_of')):
            constants.append(name)
            closure.append(name)
            continue
        closure.append(name)
        stack.extend(op.get('inputs', ()))
    for name in constants:
        op = producer[name]
        op.update(target='aten.scalar_tensor.default', args=[1.0], kwargs={'dtype': 'float32', 'layout': 'strided'},
                  inputs=[], dependencies=[], reads=[])
    # Order: earlier operations, the operand chain, the statistics, then the gradient directly after them (in the
    # statistics' region when a collective follows). Every moved operation's inputs precede it.
    moved = sorted(closure, key=pos.__getitem__)
    members = set(moved) | {stats, grad}
    data['operations'] = (ops[:first] + [producer[name] for name in moved] + [producer[stats], gradient]
                          + [op for op in ops[first:] if (op.get('outputs') or [op['name']])[0] not in members])
    return dict(operations=moved, shape_only_constants=constants, gradient_after=stats,
                gradient_moved_by=pos[grad] - first)


# LATCH-MOE-EMA-FAST: when a collective separates a head's statistics from its gradient (an intermediate head:
# forward product, backward gradient), the statistics, the pick and their consumers ordered before the gradient
# (the head's loss terms) move to just before the gradient, i.e. into the gradient's region. The forward product
# keeps its position (the audited head order); the logits already cross into that region for the gradient.
SINK_STATISTICS = True


def _scalar_ones_constants(data):
    """A 0-dim FP32 ones_like is the constant 1.0 whatever its input; drop that input edge (exact)."""
    names = []
    for op in data['operations']:
        value = data['values'].get(op['name'], {})
        if (op.get('target') == 'aten.ones_like.default' and value.get('shape') is not None
                and len(value['shape']) == 0 and value.get('dtype') == 'float32' and not value.get('alias_of')):
            op.update(target='aten.scalar_tensor.default', args=[1.0], kwargs={'dtype': 'float32', 'layout': 'strided'},
                      inputs=[], dependencies=[], reads=[])
            names.append(op['name'])
    return names


def _sink_statistics(data, record):
    ops = data['operations']
    pos = {name: i for i, op in enumerate(ops) for name in (op.get('outputs') or [op['name']])}
    stats, grad = record['stats'], record.get('dlogits')
    if grad is None or stats not in pos or grad not in pos:
        return None
    first, last = pos[stats], pos[grad]
    if not any(op.get('target') in _COLLECTIVES for op in ops[first:last]):
        return None
    consumers = {}
    for op in ops:
        for name in op.get('inputs', ()):
            consumers.setdefault(name, []).append(op)
    alias = {name: value.get('alias_of') for name, value in data['values'].items() if value.get('alias_of')}
    moved, stack = {stats}, [stats]
    while stack:
        name = stack.pop()
        readers = list(consumers.get(name, ()))
        readers += [op for view, base in alias.items() if base == name for op in consumers.get(view, ())]
        for op in readers:
            out = (op.get('outputs') or [op['name']])[0]
            if out == grad or out in moved or pos[out] > last:
                continue
            if (op.get('target') in _COLLECTIVES or op.get('effect', 'pure') != 'pure'
                    or len(op.get('outputs') or [out]) != 1):
                return dict(refused='a statistics consumer before the gradient is collective or impure', operation=out)
            moved.add(out)
            stack.append(out)
    order = [op for op in ops if (op.get('outputs') or [op['name']])[0] in moved]
    rest = [op for op in ops if (op.get('outputs') or [op['name']])[0] not in moved]
    at = next(i for i, op in enumerate(rest) if (op.get('outputs') or [op['name']])[0] == grad)
    data['operations'] = rest[:at] + order + rest[at:]
    return dict(operations=[(op.get('outputs') or [op['name']])[0] for op in order], before=grad)


def rewrite(graph, enabled=None):
    """Return a nonmutating unpadded rewrite and its selection/refusal report.

    The normal rule only applies when the original exact CUDA log-softmax
    contract is explicit. Turning the feature off returns the identical object.
    """
    if enabled is None:
        enabled = EXACT_ROWS
    if not isinstance(enabled, bool):
        raise ValueError('exact vocabulary rows toggle must be Boolean')
    if not enabled:
        return graph, None
    to_dict = getattr(graph, 'to_dict', None)
    raw = to_dict() if to_dict else graph
    if (not isinstance(raw, dict) or raw.get('metadata', {}).get('log_softmax_contract') != LOG_SOFTMAX_CONTRACT
            or not any(op.get('target') == 'aten._log_softmax.default' for op in raw.get('operations', ()))):
        return graph, None
    data = copy.deepcopy(raw)
    report = dict(version=1, log_softmax_contract=LOG_SOFTMAX_CONTRACT, rewritten=[], refused=[])
    for op in list(data['operations']):
        if op['target'] != 'aten._log_softmax.default':
            continue
        try:
            plan = _match(data, op)
            record = _apply(data, plan)
        except head.Refused as exc:
            report['refused'].append(dict(log_softmax=op['name'], reason=str(exc)))
            continue
        report['rewritten'].append(record)
    if not report['rewritten']:
        return graph, report
    if HOIST_GRADIENT_OPERANDS:
        if SINK_STATISTICS:
            report['shape_only_constants'] = _scalar_ones_constants(data)
            for record in report['rewritten']:
                record['statistics_sunk'] = _sink_statistics(data, record)
        for record in report['rewritten']:
            record['gradient_operands_hoisted'] = _hoist_gradient_operands(data, record)
    data.setdefault('metadata', {})[KEY] = report
    return (type(graph).from_dict(data) if to_dict else data), report
