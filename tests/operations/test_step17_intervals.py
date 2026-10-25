"""Interval-unit build contracts (CPU: needs nvcc/nvlink/gcc, no GPU)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from megakernel_lab.operation_compiler import runtime, step17_build, step17_intervals as si
from megakernel_lab.operation_compiler.lower import GeneratedProgram, _header_path, _translation_unit

INTERVAL = '''{{
unsigned lsr_tx = threadIdx.x; unsigned lsr_bx = blockIdx.x; unsigned lsr_nt = blockDim.x; unsigned lsr_nb = gridDim.x;
float* v_{i} = arena + {offset}LL;
auto latch_term = [&](long long r) -> float {{ return v_{i}[r] * 2.0f; }};
for (long long k = (long long)lsr_bx * lsr_nt + lsr_tx; k < 64LL; k += (long long)lsr_nb * lsr_nt) {{ v_{i}[k] = latch_term(k) + {i}.0f; }}
smem[lsr_tx] = 1.0f;
__syncthreads();
grid.sync();
if (ticks && lsr_bx == 0 && lsr_tx == 0) {{ const unsigned long long t = stamp(); ticks[{tick}] = t; }}
}}
'''


def _body(count=5, extra=''):
    blocks = ''.join(INTERVAL.format(i=i, offset=64 * i, tick=i + 1) for i in range(count))
    return ('  extern __shared__ float smem[];\n  cg::grid_group grid = cg::this_grid();\n\n'
            '  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();\n' + blocks + extra +
            '  grid.sync();\n  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[99] = stamp();\n')


def _program(body):
    source = _translation_unit(body, shared_bytes=17408)
    headers = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
               for name in ('core.cuh', 'gemm.cuh', 'epi.cuh')}
    manifest = {'header_hashes': headers, 'inputs': [], 'outputs': [], 'layouts': {}, 'target_arch': 'sm_90'}
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(headers, sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest)


def test_outline_moves_every_interval_byte_for_byte():
    program = _program(_body())
    prefix, functions, kernel, stats = si.outline(program.source, program.manifest['header_hashes'])
    assert stats['blocks'] == stats['outlined'] == 5 and stats['inline'] == 0
    assert kernel.count('latch_s17_block_') == 5 and 'grid.sync();' in kernel   # final barrier stays in the entry
    # Re-inserting the moved block texts reproduces the original source exactly.
    rebuilt = kernel
    for name, text, _ in functions:
        head = f'static __device__ __noinline__ void {name}({si.SIGNATURE}) {{\n'
        assert text.startswith(head) and text.endswith('\n}\n')
        block = text[len(head):-3]
        for decl in ('  extern __shared__ float smem[];\n', '  cg::grid_group grid = cg::this_grid();\n'):
            block = block.removeprefix(decl)
        rebuilt = rebuilt.replace(f'{name}(args, arena, ticks);', block, 1)
    assert prefix + rebuilt == program.source


@pytest.mark.parametrize('edit,reason', [
    ('{ if (arena) return; grid.sync(); }\n', 'return outside a lambda'),
    ('{ goto done; done: grid.sync(); }\n', 'goto'),
    ('{\n#if 1\n grid.sync();\n#endif\n}\n', 'preprocessor directive #if'),
    ('{\n#define LATCH_T 1\n grid.sync();\n}\n', 'macro defined in the block outlives it'),
])
def test_unsafe_blocks_stay_inline(edit, reason):
    program = _program(_body(2, edit))
    _, _, kernel, stats = si.outline(program.source, program.manifest['header_hashes'])
    assert stats['outlined'] == 2 and stats['refusals'] == {reason: 1}
    assert edit.strip().split('\n')[0] in kernel


def test_kernel_scope_declaration_refuses_the_program():
    body = _body(2).replace('  cg::grid_group grid', '  float* shared_ptr = arena + 4;\n  cg::grid_group grid')
    with pytest.raises(si.Refusal):
        si.outline(_program(body).source)


def test_keep_inline_pattern_and_macro_use():
    body = _body(2, '{\n#define LATCH_LOAD(x) (x)\n arena[0] = LATCH_LOAD(1.0f);\n#undef LATCH_LOAD\n grid.sync(); }\n'
                 '{\n#define LATCH_OTHER 2.0f\n arena[2] = 1.0f; grid.sync(); }\n'
                 '{ arena[1] = LATCH_OTHER; grid.sync(); }\n'
                 '{ pflatlong::forward_item6(arena); grid.sync(); }\n')
    _, _, _, stats = si.outline(_program(body).source)
    assert stats['outlined'] == 3 and stats['refusals'] == {
        'macro defined in the block outlives it': 1, 'uses a macro defined elsewhere in the kernel body': 1,
        'kept inline by pattern ' + si.WGMMA_INLINE[0]: 1}
    _, _, _, stats = si.outline(_program(body).source, keep_inline=())
    assert stats['outlined'] == 4
    with pytest.raises(si.Refusal):
        si.outline(_program('#define LATCH_X 1\n' + _body(2)).source)


def _sass_functions(cubin):
    sass = subprocess.run(['cuobjdump', '-sass', str(cubin)], capture_output=True, text=True, check=True).stdout
    return re.findall(r'Function : (\S+)', sass), sass


@pytest.mark.parametrize('units', [1, 3])
def test_unit_build_links_one_entry_and_caches_units(tmp_path, units):
    program = _program(_body(6))
    cache = tmp_path / 'units-cache'
    library = runtime.build(program, cache_dir=tmp_path / 'a', arch='sm_90', device_units=units,
                            device_unit_cache=cache)
    receipt = json.loads((library.parent / 'build.json').read_text())
    partition = receipt['step17_build']['partition']
    assert partition['outlined'] == 6 and partition['units'] == units and not partition['unit_cache_hits']
    names, sass = _sass_functions(library.parent / 'program.cubin')
    # Whole-program cubins list callees inside the entry's listing; relocatable links list each function.
    assert names.count('latch_ops') == 1 and len(re.findall(r'\bCALL\.', sass)) >= 6
    assert units == 1 or sum('latch_s17_block_' in n for n in names) >= 6
    # A second directory with the same unit identities reuses every unit cubin.
    again = runtime.build(program, cache_dir=tmp_path / 'b', arch='sm_90', device_units=units, device_unit_cache=cache)
    hits = json.loads((again.parent / 'build.json').read_text())['step17_build']['partition']['unit_cache_hits']
    assert len(hits) == units
    assert (again.parent / 'program.cubin').read_bytes() == (library.parent / 'program.cubin').read_bytes()


def test_cache_keys_separate_unit_settings(tmp_path):
    program = _program(_body(3))

    def path(**kw):
        with pytest.raises(FileNotFoundError) as error:
            runtime.build(program, cache_dir=tmp_path, allow_compile=False, **kw)
        return str(error.value).split(': ', 1)[1]
    keys = {path(), path(device_units=1), path(device_units=4), path(device_units=4, device_unit_inline='none'),
            path(device_units=4, ptxas_optimization_level=1), path(step17=True)}
    assert len(keys) == 6
    with pytest.raises(ValueError):
        runtime.build(program, cache_dir=tmp_path, device_units=0)
    with pytest.raises(ValueError):
        runtime.build(program, cache_dir=tmp_path, device_units=2, step17=True)


def test_unit_state_allowlist():
    ptx = (".global .align 1 .b8 __nv_static_29__x__ZN38_INTERNAL_x4cuda3std3__419piecewise_constructE[1];\n"
           ".global .align 4 .b8 __cudart_i2opi_f[24] = {65, 144};\n"
           ".global .align 1 .b8 $str$1[9] = {1};\n.global .align 1 .b8 __unnamed_1[80] = {2};\n"
           ".extern .shared .align 4 .b8 smem[];\n")
    assert si.unit_state(ptx) == dict(allowed=[
        '.global $str$1[9] =', '.global __cudart_i2opi_f[24] =',
        '.global __nv_static_29__x__ZN38_INTERNAL_x4cuda3std3__419piecewise_constructE[1]',
        '.global __unnamed_1[80] ='], refused=[])
    bad = ".global .align 4 .b8 $str$1[9];\n.global .align 4 .b8 p9_attn_seq[8];\n.const .align 4 .f32 table[4] = {1.0};\n.visible .global .align 8 .u64 counter;\n"
    assert si.unit_state(bad)['refused'] == ['.const table[4] =', '.global $str$1[9]', '.global counter',
                                             '.global p9_attn_seq[8]']


def test_unit_header_patch_is_exact_and_local():
    patched = si.unit_headers('#include "attention_long_flat.cuh"\n', {})
    original = _header_path('attention_long_flat.cuh').read_text()
    text = patched['attention_long_flat.cuh']
    assert text.count('PFLATLONG_DEV void renew_a(') == 1 and text.count('renew_a(a, a_in);') == 2
    # Removing exactly the inserted lines restores the installed header byte for byte.
    restored = text.replace(si._RENEW_A, '').replace(si._WG_NT[1], si._WG_NT[0]).replace(si._WG_NT6[1], si._WG_NT6[0])
    assert restored == original
    assert si.unit_headers('#include "core.cuh"\n', {}) == {}


SPLIT_INTERVAL = '''{
float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);
unsigned lsr_tx = threadIdx.x; asm volatile("" : "+r"(lsr_tx));
unsigned lsr_bx = blockIdx.x; asm volatile("" : "+r"(lsr_bx));
unsigned lsr_nt = blockDim.x; asm volatile("" : "+r"(lsr_nt));
unsigned lsr_nb = gridDim.x; asm volatile("" : "+r"(lsr_nb));
float* v_a = latch_pr + 4096LL;
''' + '__syncthreads();\n'.join(
    f'// Phase {i}\n{{\nfor (unsigned k = lsr_bx * lsr_nt + lsr_tx; k < 64u; k += lsr_nb * lsr_nt) v_a[k] += {i}.0f;\n}}\n'
    for i in range(6)) + '''grid.sync();
if (ticks && lsr_bx == 0 && lsr_tx == 0) { const unsigned long long t = stamp(); ticks[50] = t; }
}
'''


def _split_body(pad=0):
    return _body(2, SPLIT_INTERVAL.replace('{\nfor', '{\n// ' + 'x' * pad + '\nfor') if pad else SPLIT_INTERVAL)


def _opq_program(body):
    opq = ('template <class T> __device__ __forceinline__ T* latch_opq(T* p) { asm volatile("" : "+l"(p)); '
           'return p; }\n')
    program = _program(body)
    source = program.source.replace('extern "C" __global__', opq + 'extern "C" __global__', 1)
    manifest = dict(program.manifest)
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(manifest['header_hashes'], sort_keys=True))
                                             .encode()).hexdigest()
    return GeneratedProgram(source, manifest)


def test_split_interval_pieces_repeat_only_the_pure_prelude():
    program = _opq_program(_split_body())
    _, functions, _, stats = si.outline(program.source, program.manifest['header_hashes'], split_bytes=200)
    assert stats['split_intervals'] == [dict(function='latch_s17_block_2', bytes=len(SPLIT_INTERVAL.strip()), pieces=3)]
    texts = dict((n, t) for n, t, _ in functions)
    outer = texts['latch_s17_block_2']
    assert outer.count('latch_s17_block_2_p') == 3 and 'grid.sync();' in outer and 'v_a[k] +=' not in outer
    for k in range(3):
        piece = texts[f'latch_s17_block_2_p{k}']
        assert 'float* v_a = latch_pr + 4096LL;' in piece and 'grid.sync' not in piece
    # Phases appear exactly once across the pieces, in order.
    joined = ''.join(texts[f'latch_s17_block_2_p{k}'] for k in range(3))
    assert [int(x) for x in re.findall(r'v_a\[k\] \+= (\d+)\.0f', joined)] == list(range(6))
    # A prelude with any other statement is not split.
    impure = _split_body().replace('float* v_a = latch_pr + 4096LL;', 'float* v_a = latch_pr + 4096LL; v_a[0] = 1;')
    _, _, _, stats = si.outline(_opq_program(impure).source, split_bytes=200)
    assert stats['split_intervals'] == []


def test_split_build_links(tmp_path):
    program = _opq_program(_split_body(400))
    library = runtime.build(program, cache_dir=tmp_path, arch='sm_90', device_units=3, device_unit_split=1000)
    partition = json.loads((library.parent / 'build.json').read_text())['step17_build']['partition']
    assert partition['split_intervals'][0]['pieces'] == 3 and partition['functions'] == 6
    names, sass = _sass_functions(library.parent / 'program.cubin')
    assert names.count('latch_ops') == 1 and sum('latch_s17_block_2_p' in n for n in names) == 3
    one = runtime.build(program, cache_dir=tmp_path, arch='sm_90', device_units=1, device_unit_split=1000)
    assert one.exists() and one != library


def test_isolate_puts_each_wgmma_phase_in_a_call_free_piece():
    interval = SPLIT_INTERVAL.replace('v_a[k] += 3.0f;', 'v_a[k] += 3.0f; pflatlong::kv_item6(v_a);', 1)
    program = _opq_program(_body(2, interval))
    _, functions, _, stats = si.outline(program.source, keep_inline=(), isolate=si.WGMMA_INLINE)
    texts = dict((n, t) for n, t, _ in functions)
    assert stats['split_intervals'] == [dict(function='latch_s17_block_2', bytes=len(interval.strip()), pieces=3)]
    assert [t.count('pflatlong::kv_item6') for n, t in sorted(texts.items()) if '_p' in n] == [0, 1, 0]
    assert 'v_a[k] += 3.0f; pflatlong::kv_item6' in texts['latch_s17_block_2_p1'] and \
        'v_a[k] += 2.0f' not in texts['latch_s17_block_2_p1']


FLAG_INTERVAL = '''{{
unsigned lsr_tx = threadIdx.x; unsigned lsr_bx = blockIdx.x;
{{
  __shared__ unsigned msum_last_;
  if (lsr_tx == 0) msum_last_ = atomicAdd(reinterpret_cast<unsigned*>(arena + {i}), 1u) == 0u;
  __syncthreads();
  if (msum_last_) arena[64 + {i}] = 1.0f;
}}
__syncthreads();
{{
  __shared__ unsigned msum_last_;
  if (lsr_tx == 0) msum_last_ = 1u;
  __syncthreads();
  if (msum_last_ && lsr_tx == 0) arena[128 + {i}] = 2.0f;
}}
grid.sync();
}}
'''


def _entry_shared(library):
    usage = subprocess.run(['cuobjdump', '-res-usage', str(library.parent / 'program.cubin')], capture_output=True,
                           text=True, check=True).stdout
    lines = usage.splitlines()
    entry = next(lines[i + 1] for i, l in enumerate(lines) if 'Function latch_ops' in l)
    return int(re.search(r'SHARED:(\d+)', entry).group(1))


def test_shared_pool_gives_each_function_distinct_slots_and_shrinks_static_shared(tmp_path):
    body = _body(0, ''.join(FLAG_INTERVAL.format(i=i) for i in range(6)))
    program = _program(body)
    _, functions, _, _ = si.outline(program.source)
    pooled, slots = si.pool_shared_flags(functions)
    assert slots == 2
    for _, text, _ in pooled:
        assert '__shared__ unsigned msum_last_' not in text
        assert text.count('msum_last_ = *reinterpret_cast<unsigned*>(&latch_s17_pool[0]);') == 1
        assert text.count('msum_last_ = *reinterpret_cast<unsigned*>(&latch_s17_pool[1]);') == 1
    plain = runtime.build(program, cache_dir=tmp_path, arch='sm_90', device_units=3)
    pool = runtime.build(program, cache_dir=tmp_path, arch='sm_90', device_units=3, device_unit_shared_pool=True)
    one = runtime.build(program, cache_dir=tmp_path, arch='sm_90', device_units=1, device_unit_shared_pool=True)
    # nvlink already lays some per-unit flags out compactly; pooling must still strictly reduce static shared.
    assert _entry_shared(pool) < _entry_shared(plain)
    assert _entry_shared(one) <= _entry_shared(pool)


def test_interval_build_preserves_each_units_own_diagnostics(tmp_path, monkeypatch):
    """A warning in an earlier unit must not be replaced by the last unit's log."""
    from types import SimpleNamespace

    program = _program(_body(2))
    sources = [('unit0.cu', '// first'), ('unit1.cu', '// second')]
    monkeypatch.setattr(si, 'partition_sources', lambda *args, **kwargs: (sources, {'units': 2}))
    monkeypatch.setattr(si, '_header_names', lambda *args: [])
    monkeypatch.setattr(si, 'unit_headers', lambda *args: {})

    def measured(argv, cwd):
        output = Path(argv[argv.index('-o') + 1])
        if not output.is_absolute():
            output = Path(cwd) / output
        output.write_bytes(b'fixture-cubin')
        unit = next((v for v in argv if v.endswith('.cu')), None)
        if unit is None:
            return 0, 'link diagnostics\n', 0.1, 1, 0.1, 0.0
        keep = Path(cwd) / ('keep_' + unit[:-3])
        (keep / 'fixture.ptx').write_text('// no secondary global state\n')
        return 0, f'{unit}: distinct diagnostics\n', 0.1, 1, 0.1, 0.0

    def run(argv, **kwargs):
        assert argv[-1] == '--version' or '-sass' in argv
        return SimpleNamespace(stdout='fixture nvcc' if argv[-1] == '--version' else '', returncode=0)

    monkeypatch.setattr(step17_build, '_run_measured', measured)
    monkeypatch.setattr(step17_build.subprocess, 'run', run)
    monkeypatch.setattr(step17_build, '_cuda_home', lambda nvcc: tmp_path)
    record = step17_build._build_intervals(program, tmp_path, 'sm_90', [], 'nvcc',
                                          si.identity(2), 1, tmp_path / 'program.cubin', {})
    for unit, _ in sources:
        expected = f'{unit}: distinct diagnostics\n'
        assert (tmp_path / 'units' / (unit[:-3] + '.log')).read_text() == expected
        assert record['log'].count(expected) == 1
