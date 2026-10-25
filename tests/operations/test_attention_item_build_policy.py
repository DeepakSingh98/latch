"""The preserved O1 attention-wrapper failure restricts compilation to exact O0."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import api, runtime
from megakernel_lab.operation_compiler.lower import GeneratedProgram, cuda_module_state_policy


MODE = 'elementwise_full_sum_resource_pure_runs_attention_items'


def program(marker):
    manifest = dict(source_hash='attention-wrapper-policy-fixture', target_arch='sm_90', header_hashes={})
    if marker == 'proof':
        manifest['attention_item_codegen'] = {}
    elif marker == 'header':
        manifest['header_hashes']['attention_item_wrappers.cuh'] = 'fixture'
    else:
        raise AssertionError(marker)
    manifest['cuda_module_state'] = cuda_module_state_policy(manifest)
    return GeneratedProgram('// policy fixture\n', manifest)


@pytest.mark.parametrize('level', [None, 1, 2, 3])
@pytest.mark.parametrize('marker', ['proof', 'header'])
def test_saved_program_refused_before_toolchain_cache_or_writes(tmp_path, monkeypatch, marker, level):
    def forbidden(*args, **kwargs):
        pytest.fail('rejected attention policy reached toolchain')
    monkeypatch.setattr(runtime.subprocess, 'run', forbidden)
    with pytest.raises(ValueError, match='requires explicit ptxas_optimization_level=0'):
        runtime.build(program(marker), cache_dir=tmp_path, allow_compile=False,
                      ptxas_optimization_level=level)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('level', [None, 1, 2, 3])
def test_fresh_api_refuses_before_capture_access_or_writes(tmp_path, level):
    with pytest.raises(ValueError, match='requires explicit ptxas_optimization_level=0'):
        api.prepare_module(None, tmp_path / 'artifact', cache_dir=tmp_path / 'cache',
                           compact_codegen=MODE, ptxas_optimization_level=level)
    assert not list(tmp_path.iterdir())


def test_explicit_o0_keeps_existing_cache_identity_command_and_receipt(tmp_path, monkeypatch):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if '--version' in argv:
            return SimpleNamespace(stdout='nvcc fixture\n', returncode=0)
        Path(argv[argv.index('-o') + 1]).write_bytes(b'unchanged fixture binary')
        return SimpleNamespace(stdout='', returncode=0)
    monkeypatch.setattr(runtime.subprocess, 'run', run)
    monkeypatch.delenv('LATCH_STEP17_BUILD', raising=False)
    value = program('proof')
    before = json.dumps(value.manifest, sort_keys=True)
    library = runtime.build(value, cache_dir=tmp_path, native_code_only=True,
                            disable_expensive_optimizations=True, ptxas_optimization_level=0)
    flags = [*runtime.BUILD_FLAGS, '-Xptxas=--allow-expensive-optimizations=false', '-Xptxas=--opt-level=0']
    key = dict(source_hash=value.cache_key, arch='sm_90', flags=flags, nvcc='nvcc fixture\n',
               code_generation=dict(virtual_arch='compute_90', native_arch='sm_90', ptx_fallback=False))
    assert library.parent.name == hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    receipt = json.loads((library.parent / 'build.json').read_text())
    assert receipt['flags'] == flags
    assert receipt['command'][:-3] == ['nvcc', *flags, '--gpu-architecture=compute_90', '--gpu-code=sm_90']
    assert set(receipt) == {*key, 'cache_key', 'command', 'returncode', 'build_seconds'}
    assert json.dumps(value.manifest, sort_keys=True) == before
    assert runtime.build(value, cache_dir=tmp_path, native_code_only=True,
                         disable_expensive_optimizations=True, ptxas_optimization_level=0,
                         allow_compile=False) == library
    assert len([c for c in calls if '-o' in c]) == 1


@pytest.mark.parametrize('level', [False, True, 0.0, '0'])
def test_zero_lookalikes_do_not_bypass_validation(tmp_path, level):
    with pytest.raises(ValueError, match='integer'):
        runtime.build(program('proof'), cache_dir=tmp_path, ptxas_optimization_level=level)
    with pytest.raises(ValueError, match='integer'):
        api.prepare_module(None, tmp_path / 'artifact', cache_dir=tmp_path,
                           compact_codegen=MODE, ptxas_optimization_level=level)
    assert not list(tmp_path.iterdir())
