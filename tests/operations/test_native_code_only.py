"""Explicit GPU code targets must not share fallback-image cache entries."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import api, runtime
from megakernel_lab.operation_compiler.lower import GeneratedProgram


def program(arch="sm_90", accelerated=False):
    return GeneratedProgram("// native-code fixture\n", dict(source_hash="fixture", target_arch=arch,
        header_hashes={}, panel_engine={"products_engine": "wgmma_fixture"} if accelerated else {}))


def compiler(monkeypatch):
    calls=[]
    def run(argv, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return SimpleNamespace(stdout="nvcc fixture\n", returncode=0)
        Path(argv[argv.index("-o")+1]).write_bytes(b"fixture library")
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.delenv("LATCH_STEP17_BUILD", raising=False)
    return calls


@pytest.mark.parametrize("arch,accelerated,target", [
    ("sm_80",False,"sm_80"), ("sm_86",False,"sm_86"), ("sm_89",False,"sm_89"),
    ("sm_90",False,"sm_90"), ("sm_90",True,"sm_90a"),
])
def test_explicit_virtual_real_pair_has_distinct_cache(tmp_path, monkeypatch, arch, accelerated, target):
    calls=compiler(monkeypatch);value=program(arch, accelerated)
    default=runtime.build(value, cache_dir=tmp_path, arch=arch)
    key=dict(source_hash="fixture", arch=target, flags=runtime.BUILD_FLAGS, nvcc="nvcc fixture\n")
    assert default.parent.name==hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    assert calls[-1][:-3]==["nvcc",*runtime.BUILD_FLAGS,"-arch="+target]
    native=runtime.build(value, cache_dir=tmp_path, arch=arch, native_code_only=True)
    assert native!=default
    receipt=json.loads((native.parent/"build.json").read_text())
    virtual="compute_"+target.removeprefix("sm_")
    assert receipt["code_generation"]==dict(virtual_arch=virtual,native_arch=target,ptx_fallback=False)
    assert receipt["command"][:-3]==["nvcc",*runtime.BUILD_FLAGS,
        "--gpu-architecture="+virtual,"--gpu-code="+target]
    assert receipt["flags"]==runtime.BUILD_FLAGS
    assert value.source=="// native-code fixture\n"
    count=sum("-o" in call for call in calls)
    assert runtime.build(value,cache_dir=tmp_path,arch=arch,native_code_only=True)==native
    assert runtime.build(value,cache_dir=tmp_path,arch=arch)==default
    assert sum("-o" in call for call in calls)==count


def test_native_target_can_combine_with_explicit_device_preference(tmp_path,monkeypatch):
    compiler(monkeypatch)
    library=runtime.build(program(accelerated=True),cache_dir=tmp_path,
                          fast_device_compile=True,native_code_only=True)
    receipt=json.loads((library.parent/"build.json").read_text())
    assert "--Ofast-compile=max" in receipt["command"]
    assert "--gpu-code=sm_90a" in receipt["command"]
    assert "--fmad=false" in receipt["command"]


@pytest.mark.parametrize("value",[1,"true",None])
def test_non_boolean_option_fails_before_build(tmp_path,value):
    with pytest.raises(ValueError,match="boolean"):
        runtime.build(program(),cache_dir=tmp_path,native_code_only=value)
    with pytest.raises(ValueError,match="boolean"):
        api.prepare_module(None,tmp_path/"artifact",cache_dir=tmp_path,native_code_only=value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("environment",[False,True])
def test_host_shim_combination_fails_without_cache_output(tmp_path,monkeypatch,environment):
    compiler(monkeypatch)
    if environment:monkeypatch.setenv("LATCH_STEP17_BUILD","on")
    with pytest.raises(ValueError,match="STEP17"):
        runtime.build(program(),cache_dir=tmp_path,native_code_only=True,
                      step17=None if environment else "qualified")
    assert not list(tmp_path.iterdir())


def test_public_module_api_forwards_and_records_only_explicit_option(tmp_path,monkeypatch):
    captured=api.capture_module(torch.nn.Linear(3,2),lambda out:out.square().mean(),(torch.ones(2,3),))
    calls=[]
    def build(value,**kwargs):
        calls.append((value.source,kwargs));path=tmp_path/"fixture.so";path.write_bytes(b"fixture");return path
    monkeypatch.setattr(runtime,"build",build)
    ordinary=captured.build(tmp_path/"ordinary",cache_dir=tmp_path/"cache")
    native=captured.build(tmp_path/"native",cache_dir=tmp_path/"cache",native_code_only=True)
    assert calls[0][0]==calls[1][0]
    assert calls[0][1]==dict(cache_dir=tmp_path/"cache",arch="sm_90")
    assert calls[1][1]==dict(cache_dir=tmp_path/"cache",arch="sm_90",native_code_only=True)
    assert "native_code_only" not in ordinary and native["native_code_only"] is True
    for kwargs in (dict(tuning=object()),dict(fast_build=True)):
        with pytest.raises(ValueError,match="matrix tuning|host-shim"):
            captured.build(tmp_path/"refused",cache_dir=tmp_path/"cache",native_code_only=True,**kwargs)
        assert not (tmp_path/"refused").exists()


def test_tinymoe_cli_forwards_native_code_option(monkeypatch):
    from megakernel_lab.tinymoe import build
    calls=[]
    monkeypatch.setattr(build,"prepare",lambda *args,**kwargs:calls.append(kwargs) or {})
    monkeypatch.setattr("sys.argv",["build","--model","model","--out","artifact","--cache-dir","cache",
                                  "--native-code-only"])
    build.main()
    assert calls[0]["native_code_only"] is True
