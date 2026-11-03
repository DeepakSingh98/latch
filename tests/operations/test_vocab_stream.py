"""Streamed vocabulary-head logits product (ops/vocab_stream.py, vocab_rules ``stream``/``views``): code generation,
the statistics-free form, admission refusals, option scopes and the device-unit inline pattern."""
import pytest

from megakernel_lab.operation_compiler import vocab_rules
from megakernel_lab.operation_compiler import vocab_stream as vs
from megakernel_lab.operation_compiler import step17_intervals as s17
from megakernel_lab.operation_compiler.lower import TensorLayout

ROWS, PADDED, VOCAB, WIDTH = 4000, 8192, 8185, 448      # 63 128-column tiles + a padded one; 7 k blocks


def record(**over):
    spec = {"a": "x_rows", "ak": True, "b": "head_weight_t", "batch": 1, "bkc": True,
            "dtype": "bfloat16", "edge": True, "k": WIDTH, "m": ROWS, "n": PADDED, "out": "v_head_logits",
            "pair": True, "strides": [WIDTH, 1, 1, WIDTH, 0, 0],
            "tma": {"a": ["x_rows", 0], "b": ["head_weight", 64]}, "wg": True}
    spec.update(over)
    return {"plan": {"bm": 128, "bn": 256, "nst": 3, "wm": 2}, "rot": "/*B_VBASE*/", "spec": spec}


def logits_layout(rows=ROWS, padded=PADDED, root="head_logits", offset=0):
    return TensorLayout((rows, padded), (padded, 1), root, offset)


def test_forward_code_with_fused_statistics():
    code, need, meta = vs.forward_code(record(), stats="v_head_stats", vocab=VOCAB)
    assert ("  // Vocab head: logits product + online row statistics (vocab_stream.cuh); replaces the statistics "
            "phase.\n") in code
    assert "latch_vocab_stream::forward<4, 7>(g, smem);" in code
    assert f"POPS4B_TMAP(x_rows,0,{WIDTH},{ROWS},{2 * WIDTH},64,128)" in code
    assert f"POPS4B_TMAP(head_weight,64,{WIDTH},{PADDED},{2 * WIDTH},64,128)" in code
    assert f"v_head_logits, {PADDED}LL, v_head_stats, {ROWS}, {PADDED}, {WIDTH}, {VOCAB}}}" in code
    assert need == vs.shared_bytes() == 1024 + 8 * 16384 + 4 * 16384 + 2048
    assert meta["engine"] == vs.ENGINE and meta["shared_bytes"] == need
    assert meta["vocab_stream"]["column_tiles"] == PADDED // 128
    assert meta["vocab_stream"]["row_blocks"] == -(-ROWS // 128)
    assert meta["vocab_stream"]["diagnostic"] == 0 and meta["vocab_stream"]["state"] == "registers"
    assert s17.VOCAB_STREAM in s17.wgmma_inline(code)


def test_statistics_free_form():
    code, _, _ = vs.forward_code(record(), stats="nullptr", vocab=VOCAB)
    new = vs.stats_free(code, logits_layout())
    assert "latch_vocab_stream::forward<4, 7, 0, false, false, 0, 2>(g, smem);" in new
    assert "forward<4, 7>(" not in new
    # The logits tensor map (contiguous [R, Vp], 64 x 64 box) is appended to FArgs.
    assert (f"v_head_logits, {PADDED}LL, nullptr, {ROWS}, {PADDED}, {WIDTH}, {VOCAB}, "
            f"POPS4B_TMAP(head_logits,0,{PADDED},{ROWS},{2 * PADDED},64,64)}};") in new
    # Everything else is the fused-statistics text.
    assert new.splitlines()[:2] == code.splitlines()[:2] and new.count("\n") == code.count("\n")
    assert s17.VOCAB_STREAM in s17.wgmma_inline(new)
    # One K block; the tensor map carries the logits' own offset.
    small, _, _ = vs.forward_code(record(k=64, n=2048, m=100), stats="nullptr", vocab=2000)
    small = vs.stats_free(small, logits_layout(100, 2048, "logits_root", 4096))
    assert "latch_vocab_stream::forward<4, 1, 0, false, false, 0, 2>(g, smem);" in small
    assert "POPS4B_TMAP(logits_root,4096,2048,100,4096,64,64)};" in small


def test_statistics_free_shared_bytes():
    # 1 KiB slack + eight resident 16 KiB X blocks + 4-stage W ring + two 8 KiB store-staging quarters.
    assert vs.stats_free_shared(vs.shared_bytes()) == 214016
    assert vs.stats_free_shared(0) == 214016


@pytest.mark.parametrize("layout", [None, TensorLayout((ROWS, PADDED), (PADDED + 8, 1), "head_logits"),
                                    TensorLayout((ROWS * PADDED,), (1,), "head_logits")])
def test_statistics_free_refusals(layout):
    code, _, _ = vs.forward_code(record(), stats="nullptr", vocab=VOCAB)
    with pytest.raises(vs.Refused):
        vs.stats_free(code, layout)
    with pytest.raises(vs.Refused):           # an already rewritten call is not rewritten again
        vs.stats_free(vs.stats_free(code, logits_layout()), logits_layout())


@pytest.mark.parametrize("over", [dict(ak=False), dict(bkc=False), dict(batch=2), dict(dtype="float32"),
                                  dict(n=PADDED - 7), dict(k=520), dict(k=444), dict(ldc=PADDED + 64), dict(tma={}),
                                  dict(strides=[WIDTH, 1, WIDTH, 1, 0, 0])])
def test_refusals(over):
    with pytest.raises(vs.Refused):
        vs.forward_code(record(**over), stats="s", vocab=VOCAB)


def test_vocab_bounds():
    with pytest.raises(vs.Refused):
        vs.forward_code(record(), stats="s", vocab=PADDED + 1)
    with pytest.raises(vs.Refused):
        vs.forward_code(record(), stats="s", vocab=VOCAB, budget=190000)
    code, _, _ = vs.forward_code(record(k=64, n=2048, m=100), stats="s", vocab=2000)
    assert "forward<4, 1>" in code


def test_rules_select_the_stream_and_views():
    assert not vs.enabled() and not vs.split_enabled()
    with vocab_rules.scope("stream"):
        assert vs.enabled() and not vs.split_enabled()
    with vocab_rules.scope("views"):
        assert vs.split_enabled() and not vs.enabled()
    with vocab_rules.scope("rows"):
        assert not vs.enabled() and not vs.split_enabled()
    assert not vs.enabled() and not vs.split_enabled()


def test_other_programs_keep_unit_patterns():
    assert s17.VOCAB_STREAM not in s17.wgmma_inline("pops4b::tma_phase_inline<128,256,2,true,true,3,true,true>(g, smem);")


class _L:
    def __init__(self, root, offset):
        self.root, self.offset = root, offset


def _view_phases():
    code = "pops4b::TArgs g{m0, m1, v_vocab_chunk_a, v_vocab_chunk_b, v_out, 448LL, 8192, 448, 4000};"
    return [{"kind": "gemm", "names": ["vocab_chunk"], "reads": ["vocab_chunk_a", "vocab_chunk_b"], "code": code},
            {"kind": "gemm", "names": ["other_mm"], "reads": ["other_a"], "code": "f(v_other_a);"}]


def test_root_views():
    layouts = {"vocab_chunk_a": _L("vocab_dlogits", 16777216), "vocab_chunk_b": _L("x", 917504),
               "other_a": _L("other_root", 64)}
    ident = lambda n: "v_" + n
    phases = _view_phases()
    assert vs.root_views(phases, layouts, ident) == 0
    assert phases == _view_phases()
    with vocab_rules.scope("views"):
        assert vs.root_views(phases, layouts, ident) == 2
    assert "(v_vocab_dlogits + 16777216LL), (v_x + 917504LL)" in phases[0]["code"]
    assert phases[0]["reads"] == ["vocab_dlogits", "x"]
    assert phases[1] == _view_phases()[1]          # not a vocabulary-head phase


def test_root_views_matches_the_phases_of_a_lowered_vocabulary_head(monkeypatch):
    """root_views picks phases by the ``vocab_`` value names that the vocabulary-head rewrite gives its products; a
    real lowering must produce phases it selects, or 'views' would silently do nothing."""
    import copy
    import re
    from megakernel_lab.operation_compiler.lower import lower
    from .test_vocab_head import _capture
    calls = []
    original = vs.root_views

    def record(phases, layouts, ident):
        calls.append((copy.deepcopy(phases), ident))
        return original(phases, layouts, ident)

    monkeypatch.setattr(vs, "root_views", record)
    captured = _capture(tokens=64, width=64, vocab=1013)
    with vocab_rules.scope("views"):
        on = lower(captured.ir, execution="compose", vocabulary_head="force", target_arch="sm_90")
    off = lower(captured.ir, execution="compose", vocabulary_head="force", target_arch="sm_90")
    assert calls
    phases, ident = calls[0]
    selected = [p for p in phases if p.get("kind") == "gemm" and any(
        "vocab_" in str(x) for x in (*p.get("names", ()), *p.get("reads", ())))]
    assert len(selected) >= 2                           # the logits product and a backward product
    # This head has no offset-view operands, so the rule changes nothing ...
    assert on.source == off.source
    # ... and an offset view of a selected phase's operand is rewritten to its root.
    phase = copy.deepcopy(selected[0])
    value = next(r for r in phase["reads"] if re.search(r"\b" + re.escape(ident(r)) + r"\b", phase["code"]))
    layouts = {value: _L("vocab_root", 4096)}
    with vocab_rules.scope("views"):
        assert original([phase], layouts, ident) == 1
    assert f"({ident('vocab_root')} + 4096LL)" in phase["code"] and "vocab_root" in phase["reads"]
