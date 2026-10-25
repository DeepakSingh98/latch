"""Metadata contracts for CUDA-module ownership; no numerical computation."""
from megakernel_lab.operation_compiler.lower import cuda_module_state_policy


def test_legacy_constant_binding_remains_private():
    result = cuda_module_state_policy({'p7_const_args': True})
    assert result['requires_private_module']
    assert result['reasons'] == ['constant argument bindings']


def test_emitted_attention_headers_require_private_module():
    for name in ('attention_long.cuh', 'attention_p9.cuh'):
        result = cuda_module_state_policy({'header_hashes': {name: 'hash-bound-by-loader'}})
        assert result['requires_private_module'] and result['headers'] == [name]


def test_old_or_stale_annotations_cannot_disable_required_isolation():
    manifest = {'header_hashes': {'attention_p9.cuh': 'hash-bound-by-loader'},
                'cuda_module_state': {'requires_private_module': False}}
    assert cuda_module_state_policy(manifest)['requires_private_module']


def test_stateless_headers_do_not_force_module_copy():
    assert not cuda_module_state_policy({'header_hashes': {'p5c_wg.cuh': 'hash'}})['requires_private_module']
    assert not cuda_module_state_policy({})['requires_private_module']


def run():
    tests = {}
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            tests[name] = True
    return {'passed': True, 'tests': tests, 'scope': 'module ownership metadata only'}
