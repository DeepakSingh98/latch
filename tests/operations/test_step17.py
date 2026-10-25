"""STEP17 contracts: the opt-in fast build never reuses a binary across a relevant change.

CPU-only (needs nvcc and gcc, no GPU). Run on the lane's cloud CPU worker:
python -m pytest -q research/latch_ops/ops/test_step17.py
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import pytest

from megakernel_lab.operation_compiler import runtime, step17_build
from megakernel_lab.operation_compiler.lower import GeneratedProgram, _translation_unit


def _program(body='  if (threadIdx.x == 0 && blockIdx.x == 0) arena[0] = 1.0f;\n', shared=17408, **kw):
    import hashlib
    from megakernel_lab.operation_compiler.lower import _header_path
    source = _translation_unit(body, shared_bytes=shared, **kw)
    headers = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
               for name in ('core.cuh', 'gemm.cuh', 'epi.cuh')}
    manifest = {'header_hashes': headers, 'inputs': [], 'outputs': [], 'layouts': {}, 'target_arch': 'sm_90'}
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(headers, sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest)


def _path(program, cache, **kw):
    """Library path the build would use, without compiling."""
    with pytest.raises(FileNotFoundError) as error:
        runtime.build(program, cache_dir=cache, allow_compile=False, **kw)
    return Path(str(error.value).split(': ', 1)[1])


def test_host_template_parameters_and_refusals():
    assert step17_build.host_parameters(_program().source) == (17408, 0, None)
    assert step17_build.host_parameters(_program(shared=98304, cluster=2, max_grid_blocks=64).source) == (98304, 2, 64)
    program = _program()
    for edit in ('cudaMemcpyToSymbol(x, y, 4);', '#define P5C_TRACE 1', 'extern "C" int other_export() { return 0; }'):
        with pytest.raises(step17_build.Unsupported):
            step17_build.host_parameters(edit + '\n' + program.source)
    changed = program.source.replace('cudaLaunchCooperativeKernel((const void*)latch_ops, dim3(blocks)',
                                     'cudaLaunchKernel((const void*)latch_ops, dim3(blocks)', 1)
    with pytest.raises(step17_build.Unsupported):
        step17_build.host_parameters(changed)


def test_cache_keys_separate_every_relevant_change(tmp_path, monkeypatch):
    program = _program()
    monkeypatch.delenv('LATCH_STEP17_UNITS', raising=False)
    anchor = _path(program, tmp_path, step17=False)
    fast = _path(program, tmp_path, step17=True)
    assert anchor != fast
    monkeypatch.setenv('LATCH_STEP17_UNITS', '8')
    units8 = _path(program, tmp_path, step17=True)
    assert units8 not in (anchor, fast)
    monkeypatch.setenv('LATCH_STEP17_ASSUME_GLOBAL', 'off')
    assert _path(program, tmp_path, step17=True) != units8
    monkeypatch.delenv('LATCH_STEP17_ASSUME_GLOBAL')
    monkeypatch.setenv('LATCH_STEP17_OUTLINE_MIN', '900')
    assert _path(program, tmp_path, step17=True) != units8
    monkeypatch.delenv('LATCH_STEP17_OUTLINE_MIN')
    monkeypatch.delenv('LATCH_STEP17_UNITS')
    # Source and header identity are the program's source hash.
    other = _program(body='  if (threadIdx.x == 0 && blockIdx.x == 0) arena[0] = 2.0f;\n')
    assert _path(other, tmp_path, step17=True) != fast
    header = GeneratedProgram(program.source, {**program.manifest, 'source_hash': 'f' * 64})
    assert _path(header, tmp_path, step17=True) != fast
    # Flags, host compiler and shim template identity.
    monkeypatch.setattr(runtime, 'BUILD_FLAGS', [*runtime.BUILD_FLAGS, '-DLATCH_STEP17_TEST=1'])
    assert _path(program, tmp_path, step17=True) != fast
    monkeypatch.undo()
    identity = step17_build.toolchain_identity
    monkeypatch.setattr(step17_build, 'toolchain_identity', lambda nvcc='nvcc': {**identity(nvcc), 'cc': 'other'})
    assert _path(program, tmp_path, step17=True) != fast


def test_built_modes_never_cross_and_shim_loads_without_driver(tmp_path):
    program = _program()
    fast = runtime.build(program, cache_dir=tmp_path, step17=True)
    receipt = json.loads((fast.parent / 'build.json').read_text())
    assert receipt['step17_build']['mode'] == step17_build.SHIM_VERSION
    assert (fast.parent / 'program.cubin').exists()
    # The anchor key is still absent: a fast build never satisfies an anchor request.
    with pytest.raises(FileNotFoundError):
        runtime.build(program, cache_dir=tmp_path, allow_compile=False, step17=False)
    anchor = runtime.build(program, cache_dir=tmp_path, step17=False)
    assert anchor.parent != fast.parent and 'step17_build' not in json.loads((anchor.parent / 'build.json').read_text())
    # Repeat requests hit their own entries.
    assert runtime.build(program, cache_dir=tmp_path, step17=True) == fast
    library = ctypes.CDLL(str(fast))
    library.latch_error.restype = ctypes.c_char_p
    blocks, sms = ctypes.c_int(), ctypes.c_int()
    if not Path('/dev/nvidia0').exists():
        code = library.latch_query(ctypes.byref(blocks), ctypes.byref(sms))
        assert code != 0 and library.latch_error(code)


def test_refused_host_code_keeps_anchor_build(tmp_path):
    program = _program()
    traced = GeneratedProgram('#define P5C_TRACE_TEST 1\n' + program.source.replace('// Generated', '// TRACE gen', 1),
                              {**program.manifest, 'source_hash': 'e' * 64})
    library = runtime.build(traced, cache_dir=tmp_path, step17=True)
    receipt = json.loads((library.parent / 'build.json').read_text())
    assert 'step17_build' not in receipt and 'trace' in receipt['step17_refusal']


def test_partition_failure_falls_back_to_one_device_unit(tmp_path, monkeypatch):
    body = ''.join('  { // padding ' + 'x' * 400 + '\n  if (threadIdx.x == %d) { float acc = 0.0f; '
                   'for (int i = 0; i < 64; ++i) acc += arena[i] * %d.0f; arena[%d] = acc; } }\n' % (k, k + 1, 100 + k)
                   for k in range(4))
    program = _program(body=body)
    monkeypatch.setenv('LATCH_STEP17_UNITS', '3')
    library = runtime.build(program, cache_dir=tmp_path, step17=True)
    record = json.loads((library.parent / 'step17.json').read_text())
    assert record['partition'] is not None and not record['partition'].get('failed'), record['partition']
    assert record['partition']['units'] == 3 and (library.parent / 'program.cubin').exists()
    # A namespace-scope device variable cannot be split; the build still succeeds through one unit.
    variable = GeneratedProgram('__device__ int latch_step17_counter;\n' + program.source,
                                {**program.manifest, 'source_hash': 'd' * 64})
    library = runtime.build(variable, cache_dir=tmp_path, step17=True)
    record = json.loads((library.parent / 'step17.json').read_text())
    assert record['partition']['failed'] and 'device_nvcc' in record['timings']
