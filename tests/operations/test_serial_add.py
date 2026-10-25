"""CPU source-contract tests for bounded-live-vector ordered addition."""
import copy

from megakernel_lab.operation_compiler.serial_add import emit


def fixture(count=8, right_chain=False):
    names = [f'sum_{i}' for i in range(count)]
    body, previous = [], 'v_leaf_0[(i)]'
    texts = {f'v_leaf_{i}[(i)]': f'v_leaf_{i}' for i in range(count + 1)}
    for i, name in enumerate(names):
        a, b = previous, f'v_leaf_{i + 1}[(i)]'
        if right_chain:
            a, b = b, a
        body.append(f'const float r_{name} = ({a} + ({b} * 1.0f));'
                    + (f' v_{name}[i] = r_{name};' if i == count - 1 else ''))
        previous = f'r_{name}'
    texts[f'v_{names[-1]}[i]'] = f'v_{names[-1]}'
    return dict(names=names, body=body, targets=['aten.add.Tensor'] * count), texts


def test_ordered_chain_streams_each_vector_once_in_add_order():
    phase, texts = fixture(32)
    source = emit(phase, 65536, texts)
    assert source is not None
    assert source.count('const float4 ') == 33
    assert source.count('__fadd_rn(') == 128
    assert source.count('] = serial_acc_;') == 1
    assert source.index('serial_acc_.w =') < source.index('(v_leaf_2)')
    assert source.index('(v_leaf_2)') < source.index('(v_leaf_3)')
    assert phase['serial_add_stream']['additions'] == 32


def test_right_chain_preserves_operand_order():
    phase, texts = fixture(right_chain=True)
    source = emit(phase, 65536, texts)
    assert '__fadd_rn(serial_leaf_0.x, serial_acc_.x)' in source


def test_scaled_branched_materialized_and_nonidentity_chains_refused():
    for modification in ('alpha', 'branch', 'store', 'missing', 'tail'):
        phase, texts = fixture()
        if modification == 'alpha':
            phase['body'][2] = phase['body'][2].replace('1.0f', '0.5f')
        elif modification == 'branch':
            phase['body'][2] = phase['body'][2].replace('r_sum_1', 'r_sum_0')
        elif modification == 'store':
            phase['body'][2] += ' v_sum_2[i] = r_sum_2;'
        elif modification == 'missing':
            del texts['v_leaf_2[(i)]']
        else:
            phase['body'][-1] += ' v_elsewhere[i] = r_sum_7;'
        before = copy.deepcopy(phase)
        assert emit(phase, 65536, texts) is None
        assert phase == before


def test_short_or_unaligned_chains_use_original_emitter():
    phase, texts = fixture()
    assert emit(phase, 65535, texts) is None
    assert emit(phase, 1024, texts) is None
    phase, texts = fixture(3)
    assert emit(phase, 65536, texts) is None
