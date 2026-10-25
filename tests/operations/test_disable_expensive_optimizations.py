"""Explicit ptxas memory preference: cache, defaults, refusals and propagation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megakernel_lab.operation_compiler import api, runtime
from megakernel_lab.operation_compiler.lower import GeneratedProgram


FLAG = "-Xptxas=--allow-expensive-optimizations=false"


def program():
    return GeneratedProgram("// ptxas preference fixture\n", dict(source_hash="fixture",
        target_arch="sm_90", header_hashes={}, panel_engine={"products_engine": "wgmma_fixture"}))


def compiler(monkeypatch, *, fail_candidate=False):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if "--version" in argv:
            return SimpleNamespace(stdout="nvcc fixture\n", returncode=0)
        if fail_candidate and FLAG in argv:
            return SimpleNamespace(stdout="preserved ptxas preference failure", returncode=2)
        Path(argv[argv.index("-o") + 1]).write_bytes(b"fixture library")
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(runtime.subprocess, "run", run)
    monkeypatch.delenv("LATCH_STEP17_BUILD", raising=False)
    return calls


def test_default_identity_and_source_unchanged_candidate_is_explicit(tmp_path, monkeypatch):
    calls = compiler(monkeypatch)
    value = program()
    original_source = value.source
    default = runtime.build(value, cache_dir=tmp_path)
    # Ordinary GeneratedProgram.write annotates the initially minimal fixture
    # with cuda_module_state. The opt-in must preserve that normal result.
    original_manifest = json.dumps(value.manifest, sort_keys=True)
    key = dict(source_hash="fixture", arch="sm_90a", flags=list(runtime.BUILD_FLAGS), nvcc="nvcc fixture\n")
    assert default.parent.name == hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    assert calls[0] == ["nvcc", "--version"]
    assert calls[1][:-3] == ["nvcc", *runtime.BUILD_FLAGS, "-arch=sm_90a"]
    candidate = runtime.build(value, cache_dir=tmp_path, disable_expensive_optimizations=True)
    assert candidate != default
    receipt = json.loads((candidate.parent / "build.json").read_text())
    assert receipt["flags"] == [*runtime.BUILD_FLAGS, FLAG]
    assert receipt["command"][:-3] == ["nvcc", *runtime.BUILD_FLAGS, FLAG, "-arch=sm_90a"]
    assert "--fmad=false" in receipt["flags"] and "-O3" in receipt["flags"]
    assert value.source == original_source and json.dumps(value.manifest, sort_keys=True) == original_manifest
    count = sum("-o" in call for call in calls)
    assert runtime.build(value, cache_dir=tmp_path, disable_expensive_optimizations=True) == candidate
    assert runtime.build(value, cache_dir=tmp_path) == default
    assert sum("-o" in call for call in calls) == count


def test_combines_with_explicit_native_target_and_compile_preference(tmp_path, monkeypatch):
    compiler(monkeypatch)
    library = runtime.build(program(), cache_dir=tmp_path, native_code_only=True,
        fast_device_compile=True, disable_expensive_optimizations=True)
    receipt = json.loads((library.parent / "build.json").read_text())
    assert receipt["flags"] == [*runtime.BUILD_FLAGS, "--Ofast-compile=max", FLAG]
    assert receipt["code_generation"] == dict(virtual_arch="compute_90a", native_arch="sm_90a", ptx_fallback=False)
    assert "--gpu-code=sm_90a" in receipt["command"]


@pytest.mark.parametrize("value", [1, "false", None])
def test_strict_boolean_before_output(tmp_path, value):
    with pytest.raises(ValueError, match="boolean"):
        runtime.build(program(), cache_dir=tmp_path, disable_expensive_optimizations=value)
    with pytest.raises(ValueError, match="boolean"):
        api.prepare_module(None, tmp_path / "artifact", cache_dir=tmp_path,
                           disable_expensive_optimizations=value)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("environment", [False, True])
def test_refuses_unqualified_host_shim_combinations(tmp_path, monkeypatch, environment):
    calls = compiler(monkeypatch)
    if environment:
        monkeypatch.setenv("LATCH_STEP17_BUILD", "on")
    with pytest.raises(ValueError, match="STEP17"):
        runtime.build(program(), cache_dir=tmp_path, disable_expensive_optimizations=True,
                      step17=None if environment else "qualified")
    assert not any("-o" in call for call in calls)
    assert not list(tmp_path.iterdir())


def test_compile_failure_is_preserved_without_default_fallback(tmp_path, monkeypatch):
    calls = compiler(monkeypatch, fail_candidate=True)
    with pytest.raises(RuntimeError, match="preserved ptxas preference failure"):
        runtime.build(program(), cache_dir=tmp_path, disable_expensive_optimizations=True)
    compile_calls = [call for call in calls if "-o" in call]
    assert len(compile_calls) == 1 and FLAG in compile_calls[0]
    reports = list(tmp_path.glob("*/build.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["returncode"] == 2 and FLAG in report["flags"]
    assert "preserved ptxas preference failure" in reports[0].with_name("build.log").read_text()
    assert not list(tmp_path.glob("*/program.so"))


def test_public_api_preserves_source_and_records_opt_in(tmp_path, monkeypatch):
    captured = api.capture_module(torch.nn.Linear(3, 2), lambda out: out.square().mean(), (torch.ones(2, 3),))
    calls = []
    def build(value, **kwargs):
        calls.append((value.source, kwargs))
        path = tmp_path / "library.so"
        path.write_bytes(b"fixture")
        return path
    monkeypatch.setattr(runtime, "build", build)
    ordinary = captured.build(tmp_path / "ordinary", cache_dir=tmp_path / "cache")
    candidate = captured.build(tmp_path / "candidate", cache_dir=tmp_path / "cache",
                               disable_expensive_optimizations=True)
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] == dict(cache_dir=tmp_path / "cache", arch="sm_90")
    assert calls[1][1] == dict(cache_dir=tmp_path / "cache", arch="sm_90", disable_expensive_optimizations=True)
    assert "disable_expensive_optimizations" not in ordinary
    assert candidate["disable_expensive_optimizations"] is True
    assert json.loads((tmp_path / "candidate/artifact.json").read_text())["disable_expensive_optimizations"] is True
    for kwargs in (dict(tuning=object()), dict(fast_build=True)):
        with pytest.raises(ValueError, match="matrix tuning|host-shim"):
            captured.build(tmp_path / "refused", cache_dir=tmp_path / "cache",
                           disable_expensive_optimizations=True, **kwargs)
        assert not (tmp_path / "refused").exists()
