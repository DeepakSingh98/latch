"""Compile preference identity and forwarding; no actual nvcc or GPU needed."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import api,runtime,step17_build
from megakernel_lab.operation_compiler.lower import GeneratedProgram


def program():
    return GeneratedProgram('// fixture\n',dict(source_hash='fixture',header_hashes={},target_arch='sm_90'))


def compiler(monkeypatch):
    calls=[]
    def run(argv,**kwargs):
        calls.append(argv)
        if '--version' in argv:
            return SimpleNamespace(stdout='nvcc fixture\n',returncode=0)
        Path(argv[argv.index('-o')+1]).write_bytes(b'fixture binary')
        return SimpleNamespace(stdout='',returncode=0)
    monkeypatch.setattr(runtime.subprocess,'run',run)
    monkeypatch.delenv('LATCH_STEP17_BUILD',raising=False)
    return calls


def test_default_command_and_cache_identity_remain_unchanged(tmp_path,monkeypatch):
    calls=compiler(monkeypatch);flags=list(runtime.BUILD_FLAGS)
    library=runtime.build(program(),cache_dir=tmp_path)
    key=dict(source_hash='fixture',arch='sm_90',flags=flags,nvcc='nvcc fixture\n')
    expected=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    assert library.parent.name==expected
    assert calls[0]==['nvcc','--version']
    assert calls[1][:-4]==['nvcc',*flags]
    assert not any('Ofast' in arg for call in calls for arg in call)
    assert runtime.BUILD_FLAGS==flags


def test_preference_has_separate_cache_and_receipt_without_changing_source(tmp_path,monkeypatch):
    calls=compiler(monkeypatch);value=program();original=value.source
    baseline=runtime.build(value,cache_dir=tmp_path)
    fast=runtime.build(value,cache_dir=tmp_path,fast_device_compile=True)
    assert baseline!=fast and value.source==original
    receipt=json.loads((fast.parent/'build.json').read_text())
    assert receipt['flags']==[*runtime.BUILD_FLAGS,'--Ofast-compile=max']
    assert '--Ofast-compile=max' in receipt['command']
    assert '--fmad=false' in receipt['flags'] and '-lineinfo' in receipt['flags']
    count=sum('-o' in call for call in calls)
    assert runtime.build(value,cache_dir=tmp_path,fast_device_compile=True)==fast
    assert sum('-o' in call for call in calls)==count
    assert runtime.build(value,cache_dir=tmp_path)==baseline


def test_preference_reaches_single_unit_host_shim(tmp_path,monkeypatch):
    compiler(monkeypatch);seen={}
    monkeypatch.setattr(step17_build,'host_parameters',lambda source:(0,0,None))
    monkeypatch.setattr(step17_build,'toolchain_identity',lambda nvcc:{'fixture':'toolchain'})
    def build(value,directory,**kwargs):
        seen.update(kwargs);path=directory/'candidate.so';path.write_bytes(b'fixture')
        return dict(temporary=str(path),commands=['fixture'])
    monkeypatch.setattr(step17_build,'build',build)
    library=runtime.build(program(),cache_dir=tmp_path,step17='qualified',fast_device_compile=True)
    assert seen['units']==1
    assert seen['flags']==[*runtime.BUILD_FLAGS,'--Ofast-compile=max']
    assert json.loads((library.parent/'build.json').read_text())['flags']==seen['flags']


def test_unsupported_toolchain_fails_without_silent_fallback_or_artifact(tmp_path,monkeypatch):
    def unsupported(argv,**kwargs):
        assert argv==['nvcc','--Ofast-compile=max','--version']
        raise subprocess.CalledProcessError(1,argv,stderr='unsupported preference')
    monkeypatch.setattr(runtime.subprocess,'run',unsupported)
    with pytest.raises(subprocess.CalledProcessError):
        runtime.build(program(),cache_dir=tmp_path,fast_device_compile=True)
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('option',[1,'max',None])
def test_preference_requires_boolean(tmp_path,option):
    with pytest.raises(ValueError,match='boolean'):
        runtime.build(program(),cache_dir=tmp_path,fast_device_compile=option)
    with pytest.raises(ValueError,match='boolean'):
        api.prepare_module(None,tmp_path/'artifact',cache_dir=tmp_path,fast_device_compile=option)


def test_public_api_forwards_preference_separately_from_host_shim(tmp_path,monkeypatch):
    captured=api.capture_module(torch.nn.Linear(3,2),lambda out,target:(out-target).square().mean(),
                                (torch.ones(2,3),),(torch.zeros(2,2),))
    calls=[]
    def build(value,**kwargs):
        calls.append((value.source,kwargs));path=tmp_path/'library.so';path.write_bytes(b'fixture');return path
    monkeypatch.setattr(runtime,'build',build)
    plain=captured.build(tmp_path/'plain',cache_dir=tmp_path/'cache')
    fast=captured.build(tmp_path/'fast',cache_dir=tmp_path/'cache',fast_device_compile=True)
    assert calls[0][0]==calls[1][0]
    assert calls[0][1]==dict(cache_dir=tmp_path/'cache',arch='sm_90')
    assert calls[1][1]==dict(cache_dir=tmp_path/'cache',arch='sm_90',fast_device_compile=True)
    assert 'fast_device_compile' not in plain
    assert fast['fast_device_compile'] is True and fast['fast_build'] is False
    with pytest.raises(ValueError,match='matrix tuning'):
        captured.build(tmp_path/'tuned',cache_dir=tmp_path/'cache',tuning=object(),fast_device_compile=True)
    assert not (tmp_path/'tuned').exists()


def test_tinymoe_cli_forwards_explicit_preference(monkeypatch):
    from megakernel_lab.tinymoe import build
    calls=[]
    monkeypatch.setattr(build,'prepare',lambda *args,**kwargs:calls.append(kwargs) or {})
    monkeypatch.setattr('sys.argv',['build','--model','model','--out','artifact','--cache-dir','cache',
                                  '--fast-device-compile'])
    build.main()
    assert calls[0]['fast_device_compile'] is True
