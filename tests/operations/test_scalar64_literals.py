"""CPU-worker checks for exact scalar-FP64 literals across all elementwise emitters."""
import copy
import json

from megakernel_lab.operation_compiler import fusion_reference, fusion_p6d, fusion_p6e, fusion_p7a, fusion_p7b
from megakernel_lab.operation_compiler.lower import LoweringError, lower
from .test_lower_safety import _ref, _value


VALUE = float.fromhex('0x1.0000000000001p+0')  # One double ULP above 1; rounds to FP32 1.


def fixture(kind):
    graph = dict(inputs=['a', 'condition'], outputs=['out'],
                 values=dict(a=_value((), 'float64'), condition=_value((), 'bool'),
                             out=_value((), 'float64')), operations=[])
    if kind == 'scalar_tensor':
        args, kwargs = [VALUE], dict(dtype='float64')
    elif kind == 'full':
        args, kwargs = [[], VALUE], dict(dtype='float64')
    elif kind == 'where':
        args, kwargs = [_ref('condition'), _ref('a'), VALUE], {}
    elif kind == 'clone':
        args, kwargs = [_ref('a')], {}
    else:
        raise AssertionError(kind)
    graph['operations'] = [dict(name='out', target='aten.' + kind + '.default', args=args, kwargs=kwargs)]
    return graph


def run():
    checks = []
    emitters = [('common', lambda graph: lower(graph, policy='reference'))]
    emitters += [(module.__name__.rsplit('.', 1)[-1], module.lower_fused)
                 for module in (fusion_reference, fusion_p6d, fusion_p6e, fusion_p7a, fusion_p7b)]
    for name, emit in emitters:
        for kind in ('scalar_tensor', 'full', 'where', 'clone'):
            program = emit(fixture(kind))
            assert 'double*' in program.source
            if kind != 'clone':
                assert VALUE.hex() in program.source
                assert VALUE.hex() + 'f' not in program.source
            checks.append(dict(emitter=name, operation=kind, exact_scalar64=True))
        for value in (float('inf'), float('nan')):
            bad = copy.deepcopy(fixture('scalar_tensor'))
            bad['operations'][0]['args'] = [value]
            try:
                emit(bad)
            except LoweringError:
                checks.append(dict(emitter=name, nonfinite_refused=True))
            else:
                raise AssertionError('Nonfinite scalar64 constant admitted')
    return dict(passed=True, checks=checks)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
