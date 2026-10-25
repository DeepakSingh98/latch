"""Explicit assembler optimization: unchanged defaults and exact build identity."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import api, runtime
from megakernel_lab.operation_compiler.lower import GeneratedProgram


def program():
    return GeneratedProgram('// assembler level fixture\n', dict(source_hash='fixture',
        target_arch='sm_90', header_hashes={}, panel_engine={'products_engine':'wgmma_fixture'}))


def compiler(monkeypatch, fail=False):
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        if '--version' in argv: return SimpleNamespace(stdout='nvcc fixture\n',returncode=0)
        if fail: return SimpleNamespace(stdout='retained assembler failure',returncode=2)
        Path(argv[argv.index('-o')+1]).write_bytes(b'fixture library')
        return SimpleNamespace(stdout='',returncode=0)
    monkeypatch.setattr(runtime.subprocess,'run',run)
    monkeypatch.delenv('LATCH_STEP17_BUILD',raising=False)
    return calls


def test_default_is_unchanged_and_all_explicit_levels_have_separate_exact_cache_keys(tmp_path,monkeypatch):
    calls=compiler(monkeypatch);value=program();source=value.source
    default=runtime.build(value,cache_dir=tmp_path)
    manifest=json.dumps(value.manifest,sort_keys=True)
    key=dict(source_hash='fixture',arch='sm_90a',flags=list(runtime.BUILD_FLAGS),nvcc='nvcc fixture\n')
    assert default.parent.name==hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    assert calls[1][:-3]==['nvcc',*runtime.BUILD_FLAGS,'-arch=sm_90a']
    libraries={default}
    for level in range(4):
        library=runtime.build(value,cache_dir=tmp_path,ptxas_optimization_level=level)
        receipt=json.loads((library.parent/'build.json').read_text())
        flag=f'-Xptxas=--opt-level={level}'
        assert receipt['flags']==[*runtime.BUILD_FLAGS,flag]
        assert receipt['command'][:-3]==['nvcc',*runtime.BUILD_FLAGS,flag,'-arch=sm_90a']
        assert '--fmad=false' in receipt['flags'] and '-O3' in receipt['flags']
        assert value.source==source and json.dumps(value.manifest,sort_keys=True)==manifest
        assert library not in libraries;libraries.add(library)
        assert runtime.build(value,cache_dir=tmp_path,ptxas_optimization_level=level)==library
    assert runtime.build(value,cache_dir=tmp_path)==default
    assert len([c for c in calls if '-o' in c])==5


def test_level_combines_with_native_target_and_disabled_expensive_passes(tmp_path,monkeypatch):
    compiler(monkeypatch)
    library=runtime.build(program(),cache_dir=tmp_path,native_code_only=True,
        disable_expensive_optimizations=True,ptxas_optimization_level=1)
    receipt=json.loads((library.parent/'build.json').read_text())
    assert receipt['flags']==[*runtime.BUILD_FLAGS,'-Xptxas=--allow-expensive-optimizations=false','-Xptxas=--opt-level=1']
    assert receipt['code_generation']==dict(virtual_arch='compute_90a',native_arch='sm_90a',ptx_fallback=False)


@pytest.mark.parametrize('value',[True,False,'1',1.0,-1,4])
def test_invalid_level_refused_before_output(tmp_path,value):
    with pytest.raises(ValueError,match='integer'):
        runtime.build(program(),cache_dir=tmp_path,ptxas_optimization_level=value)
    with pytest.raises(ValueError,match='integer'):
        api.prepare_module(None,tmp_path/'artifact',cache_dir=tmp_path,ptxas_optimization_level=value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('environment',[False,True])
def test_step17_refused_including_environment(tmp_path,monkeypatch,environment):
    calls=compiler(monkeypatch)
    if environment: monkeypatch.setenv('LATCH_STEP17_BUILD','on')
    with pytest.raises(ValueError,match='STEP17'):
        runtime.build(program(),cache_dir=tmp_path,ptxas_optimization_level=1,
            step17=None if environment else 'qualified')
    assert not any('-o' in c for c in calls) and not list(tmp_path.iterdir())


def test_failure_is_preserved_without_optimizer_fallback(tmp_path,monkeypatch):
    calls=compiler(monkeypatch,fail=True)
    with pytest.raises(RuntimeError,match='retained assembler failure'):
        runtime.build(program(),cache_dir=tmp_path,ptxas_optimization_level=1)
    assert len([c for c in calls if '-o' in c])==1
    receipt,=tmp_path.glob('*/build.json')
    assert json.loads(receipt.read_text())['returncode']==2
    assert 'retained assembler failure' in receipt.with_name('build.log').read_text()
    assert not list(tmp_path.glob('*/program.so'))


def test_public_api_records_only_explicit_level_and_preserves_generated_source(tmp_path,monkeypatch):
    captured=api.capture_module(torch.nn.Linear(3,2),lambda y:y.square().mean(),(torch.ones(2,3),))
    calls=[]
    def build(value,**kwargs):
        calls.append((value.source,kwargs));path=tmp_path/'library.so';path.write_bytes(b'fixture');return path
    monkeypatch.setattr(runtime,'build',build)
    ordinary=captured.build(tmp_path/'ordinary',cache_dir=tmp_path/'cache')
    candidate=captured.build(tmp_path/'candidate',cache_dir=tmp_path/'cache',ptxas_optimization_level=1)
    assert calls[0][0]==calls[1][0]
    assert calls[0][1]==dict(cache_dir=tmp_path/'cache',arch='sm_90')
    assert calls[1][1]==dict(cache_dir=tmp_path/'cache',arch='sm_90',ptxas_optimization_level=1)
    assert 'ptxas_optimization_level' not in ordinary
    assert candidate['ptxas_optimization_level']==1
    assert json.loads((tmp_path/'candidate/artifact.json').read_text())['ptxas_optimization_level']==1
    for kwargs in (dict(tuning=object()),dict(fast_build=True),dict(fast_device_compile=True)):
        with pytest.raises(ValueError,match='matrix tuning|host-shim|fast-device'):
            captured.build(tmp_path/'refused',cache_dir=tmp_path/'cache',ptxas_optimization_level=1,**kwargs)
        assert not (tmp_path/'refused').exists()


def test_runtime_refuses_conflicting_optimizer_preferences(tmp_path):
    with pytest.raises(ValueError,match='fast_device_compile'):
        runtime.build(program(),cache_dir=tmp_path,ptxas_optimization_level=1,fast_device_compile=True)
    assert not list(tmp_path.iterdir())
