"""device_unit_max_registers: an explicit per-unit register target in every unit's nvcc command (CPU only)."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import runtime, step17_build, step17_intervals
from megakernel_lab.operation_compiler.lower import GeneratedProgram, _header_path, _translation_unit

FLAGS = ['-O3', '--use_fast_math']
NAMES = ['unit0.cu', 'unit1.cu', 'unit2.cu']


def _settings(units, **extra):
    return dict(step17_intervals.identity(units), **extra)


def _commands(units, **extra):
    return step17_build._unit_commands('nvcc', FLAGS, 'sm_90a', '/build', _settings(units, **extra), NAMES[:units])


@pytest.mark.parametrize('units', [1, 3])
def test_register_flag_follows_unit_flags_and_precedes_include(units):
    plain = _commands(units)
    explicit = _commands(units, max_registers=255)
    unit_flags = ['-rdc=true'] if units > 1 else []
    for name, before, after in zip(NAMES, plain, explicit):
        head = ['nvcc', *FLAGS, '-arch=sm_90a', *unit_flags]
        tail = ['-I/build', '-cubin', name, '-o', name[:-3] + '.cubin']
        assert before == head + tail                                    # unset: the release argv
        assert after == head + ['-maxrregcount=255'] + tail
        assert after[after.index('-maxrregcount=255') + 1] == '-I/build'
    assert _commands(units, max_registers=168)[0][len(FLAGS) + 2 + len(unit_flags)] == '-maxrregcount=168'
    assert _commands(units, max_registers=None) == plain


def test_register_target_changes_unit_cache_key():
    def key(command):
        return step17_build._unit_key('unit text', {'core.cuh': 'h'}, command[1:-5] + [command[-4]], 'nvcc 12.8',
                                      header_staging={})
    plain, explicit = _commands(3)[1], _commands(3, max_registers=255)[1]
    assert key(plain) != key(explicit)
    assert key(explicit) != key(_commands(3, max_registers=248)[1])


def _fake_unit_build(monkeypatch, tmp_path, settings):
    """Run step17_build._build_intervals with stand-in compilers; returns the record's unit commands."""
    sources = [('unit0.cu', '// entry'), ('unit1.cu', '// first'), ('unit2.cu', '// second')]
    monkeypatch.setattr(step17_intervals, 'partition_sources', lambda *args, **kwargs: (sources, {'units': 3}))
    monkeypatch.setattr(step17_intervals, '_header_names', lambda *args: [])
    monkeypatch.setattr(step17_intervals, 'unit_headers', lambda *args: {})

    def measured(argv, cwd):
        output = Path(argv[argv.index('-o') + 1])
        output = output if output.is_absolute() else Path(cwd) / output
        output.write_bytes(b'fixture-cubin')
        unit = next((v for v in argv if v.endswith('.cu')), None)
        if unit is not None:
            (Path(cwd) / ('keep_' + unit[:-3]) / 'fixture.ptx').write_text('// no secondary global state\n')
        return 0, '', 0.1, 1, 0.1, 0.0

    def run(argv, **kwargs):
        return SimpleNamespace(stdout='fixture nvcc' if argv[-1] == '--version' else '', returncode=0)
    monkeypatch.setattr(step17_build, '_run_measured', measured)
    monkeypatch.setattr(step17_build.subprocess, 'run', run)
    monkeypatch.setattr(step17_build, '_cuda_home', lambda nvcc: tmp_path)
    record = step17_build._build_intervals(_program(), tmp_path, 'sm_90', list(FLAGS), 'nvcc', settings, 1,
                                          tmp_path / 'program.cubin', {})
    return [command for command in record['commands']['units'] if any(v.endswith('.cu') for v in command)]


def test_unit_build_commands_carry_register_target(tmp_path, monkeypatch):
    commands = _fake_unit_build(monkeypatch, tmp_path, _settings(3, max_registers=255))
    assert len(commands) == 3
    for command in commands:
        at = command.index('-maxrregcount=255')
        assert command[at - 1] == '-rdc=true' and command[at + 1] == f'-I{tmp_path}'
        assert command.count('-maxrregcount=255') == 1


def test_unit_build_commands_unchanged_without_register_target(tmp_path, monkeypatch):
    commands = _fake_unit_build(monkeypatch, tmp_path, _settings(3))
    assert len(commands) == 3 and not any(v.startswith('-maxrregcount') for c in commands for v in c)
    for command in commands:
        assert command[command.index('-rdc=true') + 1] == f'-I{tmp_path}'


def _program():
    source = _translation_unit('  grid.sync();\n', shared_bytes=17408)
    headers = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
               for name in ('core.cuh', 'gemm.cuh', 'epi.cuh')}
    manifest = {'header_hashes': headers, 'inputs': [], 'outputs': [], 'layouts': {}, 'target_arch': 'sm_90'}
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(headers, sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest)


@pytest.fixture
def toolchain(monkeypatch):
    """A stand-in nvcc/cc identity so cache keys can be derived without CUDA."""
    monkeypatch.setattr(step17_build, 'toolchain_identity', lambda nvcc='nvcc', cc='gcc': {})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'nvcc test', ''))


def test_native_cache_key_binds_register_target(tmp_path, toolchain):
    program = _program()

    def path(**kw):
        with pytest.raises(FileNotFoundError) as error:
            runtime.build(program, cache_dir=tmp_path, allow_compile=False, device_units=4, **kw)
        return str(error.value)
    default = path()
    assert path(device_unit_max_registers=None) == default
    assert len({default, path(device_unit_max_registers=255), path(device_unit_max_registers=128)}) == 3


@pytest.mark.parametrize('value', [0, 56, 63, 65, 167, 249, 254, 256, 1024, True, 255.0, '255', 'r255'])
def test_invalid_register_targets_raise(tmp_path, value):
    with pytest.raises(ValueError, match='device_unit_max_registers'):
        runtime.build(_program(), cache_dir=tmp_path, allow_compile=False, device_units=4,
                      device_unit_max_registers=value)


@pytest.mark.parametrize('value', [64, 128, 168, 248, 255])
def test_valid_register_targets_pass_validation(tmp_path, toolchain, value):
    with pytest.raises(FileNotFoundError):
        runtime.build(_program(), cache_dir=tmp_path, allow_compile=False, device_units=4,
                      device_unit_max_registers=value)


def test_register_target_requires_device_units(tmp_path):
    with pytest.raises(ValueError, match='requires device_units'):
        runtime.build(_program(), cache_dir=tmp_path, allow_compile=False, device_unit_max_registers=255)
