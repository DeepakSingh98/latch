"""device_unit_inline='isolate_all': warpgroup-MMA engine phases and argument offset-view preludes (CPU only)."""
import hashlib
import json
import re
import subprocess

import pytest

from megakernel_lab.operation_compiler import runtime, step17_build, step17_intervals as s, unit_isolation
from megakernel_lab.operation_compiler.lower import GeneratedProgram, _header_path, _translation_unit
from megakernel_lab.operation_compiler.matrix_bodies import VIEW_SENTINEL, isolate_patterns

PRELUDE = """float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);
unsigned lsr_tx = threadIdx.x; asm volatile("" : "+r"(lsr_tx));
unsigned lsr_nb = gridDim.x; asm volatile("" : "+r"(lsr_nb));
__nv_bfloat16* v_a = reinterpret_cast<__nv_bfloat16*>(latch_pr + 4096LL);
__nv_bfloat16* v_b = reinterpret_cast<__nv_bfloat16*>(latch_pr + 4096LL) + 1048576LL;
float* v_s = latch_pa[3] + 64LL;
"""
GEMM = "{\n  pops4b::TArgs g{0};\n  pops4b::tma_phase_inline<128,64,2,false,false,4,true,true>(g, smem, 0, 1, 1, 0);\n}\n"
WMM = "{\n  popsc::BArgs g{v_a};\n  pops4b::wmm_phase_inline<128,64,2,true,true,4,true>(g, smem, 0, 1, 1, 0);\n}\n"
ELEM = "{\nfor (long long i = lsr_tx; i < n; i += lsr_nb) v_s[i] = 0.0f;\n}\n"


def block(*phases, prelude=PRELUDE):
    return "{\n" + prelude + "__syncthreads();\n".join(phases) + "grid.sync();\n}"


def test_offset_view_prelude_refused_by_default_and_split_with_isolate_all():
    b = block(GEMM, ELEM, GEMM)
    base = s.wgmma_inline(b)
    assert s.split_interval("f", b, len(b), base) is None
    iso = base + unit_isolation.patterns(b)
    pieces, outer = s.split_interval("f", b, len(b), iso)
    assert len(pieces) == 3
    assert [bool(re.search("tma_phase_inline", t)) for _, t, _ in pieces] == [True, False, True]
    assert outer.count("f_p") == 3 and "grid.sync();" in outer


def test_argument_offset_views_need_their_own_sentinel():
    b = block(GEMM, ELEM, GEMM)
    engines = s.wgmma_inline(b) + tuple(p for p in unit_isolation.patterns(b) if p != unit_isolation.SENTINEL)
    assert s.split_interval("f", b, len(b), engines) is None            # `latch_pa[3] + 64LL` is not accepted
    assert s.split_interval("f", b, len(b), engines + (unit_isolation.SENTINEL,)) is not None
    cast_only = PRELUDE.replace("float* v_s = latch_pa[3] + 64LL;\n", "")
    c = block(GEMM, ELEM, GEMM, prelude=cast_only)
    assert s.split_interval("f", c, len(c), engines) is not None         # cast views: the matrix_bodies sentinel
    assert not any(re.search(p, "arg_offset_views " + b) for p in (VIEW_SENTINEL, unit_isolation.SENTINEL))


def test_wmm_engine_isolated_only_with_isolate_all():
    b = block(WMM, ELEM, WMM, ELEM)
    assert not any(re.search(p, b) for p in s.wgmma_inline(b))
    pats = unit_isolation.patterns(b)
    assert any(re.search(p, b) for p in pats)
    pieces, _ = s.split_interval("f", b, len(b), s.wgmma_inline(b) + pats)
    wmm = [t for _, t, _ in pieces if "wmm_phase_inline" in t]
    assert len(wmm) == 2 and all(t.count("// ") == 0 and "for (long long" not in t for t in wmm)


def test_patterns_only_for_emitted_engines():
    assert unit_isolation.patterns("int x;") == (VIEW_SENTINEL, unit_isolation.SENTINEL)
    found = unit_isolation.patterns("p5c::phase_chain<1>(); p9ws::bitem2<3>();")
    assert found == (unit_isolation.ENGINES[1], unit_isolation.ENGINES[3], VIEW_SENTINEL, unit_isolation.SENTINEL)


# ------------------------------------------------------------------------- runtime.build isolate tuple (no nvcc)
EMITTED = ("  // latch_ragged::flat<1> pops4b::wmm_phase_inline<1> p5c::phase_chain<1> p8ws::group<1>\n"
           "  grid.sync();\n")


def _program(group_rules):
    source = _translation_unit(EMITTED, shared_bytes=17408)
    headers = {name: hashlib.sha256(_header_path(name).read_bytes()).hexdigest()
               for name in ('core.cuh', 'gemm.cuh', 'epi.cuh')}
    manifest = {'header_hashes': headers, 'inputs': [], 'outputs': [], 'layouts': {}, 'target_arch': 'sm_90',
                'group_rules': group_rules}
    manifest['source_hash'] = hashlib.sha256((source + json.dumps(headers, sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest)


@pytest.fixture
def identities(monkeypatch):
    """Record the step17 interval identity runtime.build binds, with a stand-in toolchain."""
    seen = []
    real = s.identity

    def identity(*args, **kwargs):
        seen.append(kwargs.get('isolate', ()))
        return real(*args, **kwargs)
    monkeypatch.setattr(s, 'identity', identity)
    monkeypatch.setattr(step17_build, 'toolchain_identity', lambda nvcc='nvcc', cc='gcc': {})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 0, 'nvcc test', ''))
    return seen


def _isolate(program, inline, tmp_path, seen):
    with pytest.raises(FileNotFoundError):
        runtime.build(program, cache_dir=tmp_path, allow_compile=False, device_units=4, device_unit_inline=inline)
    return tuple(seen[-1])


def test_isolate_all_tuple_order(tmp_path, identities):
    program = _program({'matrix_bodies': ['exact', 'isolate']})
    src = program.source
    got = _isolate(program, 'isolate_all', tmp_path, identities)
    expected = tuple(dict.fromkeys(s.wgmma_inline(src) + isolate_patterns(program.manifest)
                                   + unit_isolation.patterns(src)))
    assert got == expected
    assert got.count(VIEW_SENTINEL) == 1 and got[-1] == unit_isolation.SENTINEL
    assert got.index(VIEW_SENTINEL) < got.index(unit_isolation.ENGINES[0])
    assert s.RAGGED_FLAT in got[:got.index(VIEW_SENTINEL)]
    assert _isolate(program, 'isolate', tmp_path, identities) == s.wgmma_inline(src) + isolate_patterns(program.manifest)
    assert _isolate(program, 'wgmma', tmp_path, identities) == ()


def test_isolate_all_without_matrix_isolate_rule(tmp_path, identities):
    program = _program({})
    src = program.source
    got = _isolate(program, 'isolate_all', tmp_path, identities)
    assert got == s.wgmma_inline(src) + unit_isolation.patterns(src)


def test_isolate_all_has_its_own_cache_key(tmp_path, identities):
    program = _program({})

    def path(inline):
        with pytest.raises(FileNotFoundError) as error:
            runtime.build(program, cache_dir=tmp_path, allow_compile=False, device_units=4, device_unit_inline=inline)
        return str(error.value)
    assert len({path('isolate'), path('isolate_all'), path('wgmma')}) == 3
    with pytest.raises(ValueError, match="'isolate_all'"):
        runtime.build(program, cache_dir=tmp_path, allow_compile=False, device_units=4, device_unit_inline='all')
