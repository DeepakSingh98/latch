"""Independent small matrix/reduction structural contracts; run on Modal CPU.

These check admitted layouts, safe fallbacks and the original FP32 reduction
tree. Native arithmetic is separately gated on the complete model GPU updates.
"""
from __future__ import annotations

import ast
import re
import struct
import traceback

from megakernel_lab.operation_compiler.lower import TensorLayout, _contiguous
from megakernel_lab.operation_compiler.matrix_codegen import select, split_k
from megakernel_lab.operation_compiler.codegen_rules import exact_reduction


def layout(shape, name, *, strides=None, offset=0):
    return TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), name, offset)


def matrix_admission_and_fallback():
    a, b = layout((96, 320), 'a'), layout((320, 256), 'b')
    chosen, reason = select('a', 'b', 'out', a, b, (96, 256), 'bfloat16')
    assert chosen and chosen['dtype'] == 'bfloat16', reason
    assert (chosen['m'], chosen['n'], chosen['k']) == (96, 256, 320)
    transposed = layout((96, 320), 'a', strides=(1, 96))
    assert select('a', 'b', 'out', transposed, b, (96, 256), 'bfloat16')[0]
    for operand in (layout((96, 320), 'a', offset=1),
                    layout((96, 320), 'a', strides=(640, 2))):
        candidate, why = select('a', 'b', 'out', operand, b, (96, 256), 'bfloat16')
        assert candidate is None and 'retained' in why, why
    tail_a, tail_b = layout((16, 48), 'a'), layout((48, 10), 'b')
    assert select('a', 'b', 'out', tail_a, tail_b, (16, 10), 'bfloat16')[0] is None
    assert select('a', 'b', 'out', a, b, (96, 256), 'bfloat16', arch='sm_80')[0] is None
    # FP32 rank2 stays on the existing/split-K path; only admitted underfilled
    # batched shapes use C's exact-order small body.
    assert select('a', 'b', 'out', a, b, (96, 256), 'float32')[0] is None
    ba, bb = layout((2, 64, 64), 'a'), layout((2, 64, 64), 'b')
    assert select('a', 'b', 'out', ba, bb, (2, 64, 64), 'float32')[0]
    full_a, full_b = layout((64, 64, 64), 'a'), layout((64, 64, 64), 'b')
    assert select('a', 'b', 'out', full_a, full_b, (64, 64, 64), 'float32')[0] is None
    return {'aligned_bf16': True, 'transposed_bf16': True, 'tail_and_unaligned_fallback': True,
            'non_sm90_fallback': True, 'underfilled_batched_fp32': True}


def split_k_admission_is_bounded():
    a, b = layout((64, 1024), 'a'), layout((1024, 64), 'b')
    spec = split_k('a', 'b', 'out', a, b, ident=lambda name: 'v_' + name)
    assert spec and spec['metadata']['ksplit'] == 8
    assert spec['elements'] == 8 * 64 * 64
    assert 'ascending slice sum' in spec['metadata']['arithmetic']
    assert split_k('a', 'b', 'out', a, b, ident=str, enabled=False) is None
    assert split_k('a', 'b', 'out', layout((63, 1024), 'a'), b, ident=str) is None
    assert split_k('a', 'b', 'out', layout((64, 1024), 'a', offset=1), b, ident=str) is None
    assert split_k('a', 'b', 'out', layout((64, 64), 'a'), layout((64, 64), 'b'), ident=str) is None


def _f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def _butterfly(values):
    values = list(values)
    for offset in (16, 8, 4, 2, 1):
        previous = values
        values = [_f32(previous[lane] + previous[lane ^ offset]) for lane in range(32)]
    return values[0]


def exact_small_reduction_tree():
    # Evaluate generated scalar expression with an independent FP32 interpreter.
    # Alternating large/cancelling/small terms distinguish association changes.
    pattern = [1e20, 1., -1e20, -1., 1e-20, -1e-20, -0., 3.25]
    tested = 0
    for extent in (1, 2, 3, 7, 8, 15, 16, 17, 31, 32):
        for shift in range(len(pattern)):
            leaves = [_f32(pattern[(i + shift) % len(pattern)]) for i in range(extent)]
            def read(position):
                return 'leaf' + re.search(r'\((\d+)LL /', position).group(1)
            code, metadata = exact_reduction(out='out', position='(r / 1)', extent=extent,
                results=13, read=read, rounded=lambda expression: expression)
            expression = re.search(r'const float acc = (.*);', code).group(1).replace('0.0f', '0.0')
            def evaluate(node):
                if isinstance(node, ast.Name):
                    return leaves[int(node.id[4:])]
                if isinstance(node, ast.Constant):
                    return _f32(node.value)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    return _f32(evaluate(node.left) + evaluate(node.right))
                raise AssertionError('unexpected generated scalar expression')
            actual = evaluate(ast.parse(expression, mode='eval').body)
            warp0 = _butterfly([_f32(0.0 + value) for value in leaves] + [0.0] * (32-extent))
            expected = _butterfly([warp0] + [0.0] * 31)
            assert struct.pack('f', actual) == struct.pack('f', expected), (extent, shift)
            assert metadata['engine'] == 'exact_cta_tree_per_thread'
            tested += 1
    return {'bitwise_scalar_tree_fixtures': tested}


def exact_warp_reduction_boundaries():
    # Interpret the emitted lane-zero expression independently of codegen's
    # symbolic butterfly. Compare IEEE FP32 bytes, including the zero sign, to
    # the original two-stage 256-thread CTA algorithm. Each thread starts at +0
    # and folds its leaves in order before either original or pruned stage.
    patterns = ([1e20, 1., -1e20, -1., 1e-20, -1e-20, -0., 3.25],
                [0., -0.], [-0.], [2.**-149, -2.**-149, 2.**-126, -2.**-126],
                [16777216., 1., -16777216., -1.])
    tested = 0
    for extent in (33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160,
                   161, 191, 192, 193, 223, 224, 225, 255, 256, 257, 511, 512, 2048):
        code, metadata = exact_reduction(out='out', position='(r / 1)', extent=extent,
            results=128, read=lambda pos: 'input[' + pos + ']', rounded=str, wide=extent > 256)
        assert metadata['engine'] == 'exact_cta_tree_per_warp'
        assert '0.0f + input[(rr / 1)]' in code
        live = min(8, (extent + 31) // 32)
        assert f'for (int w = 0; w < {live}; ++w) v[w] = s101h::warp_sum(v[w]);' in code
        assert 's101h::warp_sum(second)' not in code
        expression = re.search(r'const float second = (.*?);', code).group(1)
        tree = ast.parse(expression, mode='eval').body
        for pattern in patterns:
            for shift in range(len(pattern)):
                leaves = [_f32(pattern[(i + shift) % len(pattern)]) for i in range(extent)]
                threads = [0.0] * 256
                for i, value in enumerate(leaves):
                    threads[i % 256] = _f32(threads[i % 256] + value)
                partials = [_butterfly(threads[w * 32:(w + 1) * 32]) for w in range(8)]
                def evaluate(node):
                    if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                            and node.value.id == 'v' and isinstance(node.slice, ast.Constant)):
                        return partials[node.slice.value]
                    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                        return _f32(evaluate(node.left) + evaluate(node.right))
                    raise AssertionError('unexpected generated warp expression')
                actual = evaluate(tree)
                expected = _butterfly(partials + [0.0] * 24)
                assert struct.pack('f', actual) == struct.pack('f', expected), (extent, pattern, shift)
                tested += 1
    for extent, results in ((33, 127), (257, 128), (0, 128)):
        assert exact_reduction(out='out', position='(r / 1)', extent=extent, results=results,
                               read=str, rounded=str) is None
    return {'bitwise_warp_tree_fixtures': tested, 'comparison': 'IEEE FP32 bytes including signed zero'}


def exact_coalesced_unsigned_leaves():
    # The convolution donor fixes unsigned index substitution in the scalar
    # replay of the same CTA tree. Canonicalize only each leaf's index spelling:
    # the resulting source, leaf order and complete tree must be identical.
    fixtures = 0
    for extent in (33, 63, 64, 65, 127, 128, 129, 255, 256):
        sources, sequences = [], []
        for position, pattern in (('(r / 1)', r'\((\d+)LL / 1\)'),
                                  ('(unsigned)(r)', r'\(unsigned\)\((\d+)LL\)')):
            leaves = []
            def read(index):
                match = re.fullmatch(pattern, index)
                assert match is not None, index
                leaves.append(int(match.group(1)))
                return 'leaf' + match.group(1)
            code, metadata = exact_reduction(out='out', position=position, extent=extent,
                results=1024, read=read, rounded=str, coalesced_threads=True)
            assert metadata['engine'] == 'exact_cta_tree_per_thread_column'
            sources.append(code)
            sequences.append(leaves)
        assert sequences[0] == sequences[1] == list(range(extent))
        assert sources[0] == sources[1]
        fixtures += 1
    return {'identical_leaf_and_tree_source_fixtures': fixtures}


def scratch_collision_keeps_supported_matrix():
    from megakernel_lab.operation_compiler.lower import lower, _lower_serial
    shapes = {'a': (64,1024), 'b': (1024,64), 'result': (64,64), 'result__splitk': (1,)}
    graph = {'inputs':['a','b','result__splitk'], 'outputs':['result'],
        'values': {name: {'shape':list(shape),'strides':list(_contiguous(shape)), 'dtype':'float32'}
                   for name,shape in shapes.items()},
        'operations':[{'name':'result','target':'aten.mm.default','args':[{'ref':'a'},{'ref':'b'}],'kwargs':{}}]}
    for generate in (lower,_lower_serial):
        # Isolate D's split-K fallback: automatic B matrices have different
        # scratch names and may safely handle this graph without D's scratch.
        program = generate(graph, matrix_policy='frozen')
        assert program.manifest['phases'][0]['engine'] == 'pinned_simt_64'
        assert any('scratch name collision' in row['reason'] for row in program.manifest['matrix_decisions'])
    return {'existing_body_retained':True,'both_schedulers':True}


def run():
    tests = []
    for function in (matrix_admission_and_fallback, split_k_admission_is_bounded,
                     exact_small_reduction_tree, exact_warp_reduction_boundaries,
                     exact_coalesced_unsigned_leaves,
                     scratch_collision_keeps_supported_matrix):
        try:
            detail = function()
            tests.append({'name': function.__name__, 'passed': True, 'detail': detail})
        except Exception:
            tests.append({'name': function.__name__, 'passed': False, 'error': traceback.format_exc()})
    return {'kind': 'integration-matrix-reduction-contracts', 'tests': tests,
            'passed': all(test['passed'] for test in tests)}
