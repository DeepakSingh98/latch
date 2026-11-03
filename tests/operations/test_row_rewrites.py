"""Row-body rewrites (operation_compiler/row_rewrites.py): accumulation_rules 'ridx'/'rearly'/'rremat' and
elementwise_rules 'divc' text rules and their exactness arguments."""
import math
import random
import struct
from contextlib import ExitStack, contextmanager

import pytest

from megakernel_lab.operation_compiler import accumulation_rules, elementwise_rules
from megakernel_lab.operation_compiler import row_rewrites as rr


@contextmanager
def rules(elementwise=None, accumulation=None):
    with ExitStack() as stack:
        stack.enter_context(elementwise_rules.scope(elementwise))
        stack.enter_context(accumulation_rules.scope(accumulation))
        yield


ROW = """for (long long o = 0; o < 65536LL; o += 1) {
  const int lane = lsr_tx & 31;
#pragma unroll
  for (int w = 0; w < 12; ++w) {
    const long long i = o * 384LL + 32 * w + lane; const long long i_row = o;
    const float r_a = ((((long long)((unsigned)(((long long)((unsigned)(i) / 384u))) % 5u)) == 4LL) ? v_g[((long long)((unsigned)(((long long)((unsigned)(i) / 1920u))) % 13107u)) * 384LL + ((long long)((unsigned)(((long long)((unsigned)(i) / 1u))) % 384u)) * 1LL] : 0.0f);
    const float r_b = (v_e[(long long)((unsigned)(i) / 1920u % 13107u) * 384LL + (long long)((unsigned)(i) % 384u) * 1LL] / 7.0f);
  }
  {
  const long long i = o;
  const float r_c = (v_s[(unsigned)(i) / 384u] / 3.0f);
  }
}"""


def test_default_off_identity():
    with rules():
        assert rr.row_index_rewrite(ROW, 65536) == ROW
        assert rr.division_rewrite(ROW) == ROW
        assert not (rr.rowidx() or rr.earlyepi() or rr.remat() or rr.divc())


def test_tokens_select_the_rewrites():
    with rules("divc", "ridx+rearly+rremat"):
        assert rr.rowidx() and rr.earlyepi() and rr.remat() and rr.divc()
    with rules("lanecse", "rstage"):
        assert not (rr.rowidx() or rr.earlyepi() or rr.remat() or rr.divc())


def test_ridx_rewrites_only_inside_row_scope():
    with rules(accumulation="ridx"):
        out = rr.row_index_rewrite(ROW, 65536)
        too_many = rr.row_index_rewrite(ROW, 1 << 24)
    loop, rows = out.split("  {\n  const long long i = o;")
    assert "(unsigned)(i) /" not in loop and "(unsigned)(i) %" not in loop
    assert "(unsigned)(i_row) % 5u" in loop and "((unsigned)(i_row) / 5u)" in loop and "(unsigned)(32 * w + lane)" in loop
    assert rows == ROW.split("  {\n  const long long i = o;")[1]   # the i = o scope is untouched
    assert too_many == ROW                                         # too many rows for 32-bit indices: unchanged


def test_ridx_identities_exhaustive_small():
    """The rewritten forms equal the originals for every row/lane (the module's arithmetic argument)."""
    rnd = random.Random(0)
    for width in (32, 96, 384):
        for c in (1, 2, 5, 7, 8):
            rows = 65536
            for o in [0, 1, 6, 7, rows - 1] + [rnd.randrange(rows) for _ in range(200)]:
                for k in range(width):
                    i = (o * width + k) & 0xFFFFFFFF
                    assert i // (width * c) == o // c
                    assert i // width == o
                    assert i % width == k


def test_divc_rewrite_guards(monkeypatch):
    monkeypatch.setattr(rr, "VERIFIED_DIVISORS", {"7.0f": ("0x1.24924a0000000p-3f", "0.14285714285714285"),
                                                    "3.0f": ("0x1.5555560000000p-2f", "0.3333333333333333")})
    counts = {}
    with rules("divc"):
        out = rr.division_rewrite("const float r = (v_e[(i) * 2LL] / 7.0f); const float s = (r_x / 3.0f);", counts)
        keep = [
            "(a + b / 7.0f)",                 # quotient is not the whole parenthesis (precedence)
            "x = b / 7.0f;",                  # not parenthesized
            "((double)(r_x) / 7.0f)",         # double operand
            "((r_x * 0.5) / 7.0f)",           # double literal inside the operand
            "(r_x / 8.0f)",                   # power of two / unverified literal
            "(r_x / 9.0f)",                   # unverified literal
        ]
        kept = [rr.division_rewrite(text) for text in keep]
        nested = rr.division_rewrite("((r_a * v_b[i]) / 7.0f)")
        member = rr.division_rewrite("(latch_ec_l7.w / 3.0f)")
    assert out == ("const float r = latch_divc(v_e[(i) * 2LL], 7.0f, 0x1.24924a0000000p-3f, (double)7.0f, "
                   "0.14285714285714285); const float s = latch_divc(r_x, 3.0f, 0x1.5555560000000p-2f, "
                   "(double)3.0f, 0.3333333333333333);")
    assert counts == {"7.0f": 1, "3.0f": 1}
    assert kept == keep
    assert nested.startswith("latch_divc((r_a * v_b[i]), 7.0f,")
    assert member.startswith("latch_divc(latch_ec_l7.w, 3.0f,")


def test_verified_table_constants():
    """Every table entry carries RN_float(1/c) and RN_double(1/c) of the float literal, and no power of two."""
    for lit, (yf, yd) in rr.VERIFIED_DIVISORS.items():
        c = struct.unpack("<f", struct.pack("<f", float(lit[:-1])))[0]
        assert c > 0 and math.log2(c) != int(math.log2(c))
        assert float.fromhex(yf[:-1]) == _f32_div(1.0, c)
        assert float(yd) == 1.0 / c


def _f32_div(a, b):
    import numpy as np
    return float(np.float32(a) / np.float32(b))


def test_divc_helper_text():
    helper = rr.DIVC_HELPER
    assert helper.startswith("#ifndef LATCH_EXACTDIV\n#define LATCH_EXACTDIV\n"
                             "// latch_divc (row_rewrites.py): IEEE x / c for ")
    assert helper.rstrip().endswith("#endif")
    assert "isnan(x)" in helper and "isinf(x)" in helper and "x == 0.0f" in helper
    assert "if (" not in helper                            # selects only: no branch


def test_feeds_group_cycle_guard():
    """An early epilogue candidate whose output reaches a group input through the span is refused."""
    phases = [
        {"writes": ["a"], "reads": []},        # 0 prologue member (reads nothing)
        {"writes": ["e"], "reads": ["a"]},     # 1 candidate e
        {"writes": ["x"], "reads": ["e"]},     # 2 between e and k, reads e
        {"writes": ["s"], "reads": ["a"]},     # 3 reduction k
        {"writes": ["z"], "reads": ["e"]},     # 4 after k
    ]
    readers = {}
    for i, p in enumerate(phases):
        for r in p["reads"]:
            readers.setdefault(r, set()).add(i)
    root = lambda n: n
    assert not rr.feeds_group(phases, 1, 3, {"a"}, root, readers)
    assert rr.feeds_group(phases, 1, 3, {"a", "x"}, root, readers)      # a member reads x, which depends on e
    assert rr.feeds_group(phases, 1, 3, {"e"}, root, readers)           # a member reads e itself
    assert not rr.feeds_group(phases, 1, 3, {"a", "z"}, root, readers)  # z is produced after k: not in the span


class _L:
    def __init__(self, root, numel):
        self.root, self.numel, self.contiguous, self.offset = root, numel, True, 0


GROUP = """for (long long o = 0; o < 64LL; o += 1) {
  const int lane = lsr_tx & 31;
#pragma unroll
  for (int w = 0; w < 12; ++w) {
    const long long i = o * 384LL + 32 * w + lane; const long long i_row = o;
    const float r_g = (v_x[(i)] * 2.0f);
    const float r_y = (r_g * v_s[(long long)((unsigned)(i_row) % 64u) * 40LL]); sk0_[w] = r_y;
  }
#pragma unroll
  for (int w = 0; w < 12; ++w) {
    const long long i = o * 384LL + 32 * w + lane; const long long i_row = o;
    v_y[i] = sk0_[w];
  }
}"""


def _remat_fixture():
    n = 64 * 384
    layouts = {"x": _L("x", n), "g": _L("g", n), "y": _L("y", n), "s": _L("s", 64 * 40), "z": _L("z", n)}
    dtypes = {k: "float32" for k in layouts}
    group = dict(kind="row_fused", code=GROUP, names=["g", "y"], reads=["x", "s"], writes=["g", "y"],
                 row_fusion=dict(width=384, rows=64, stored=["y"], elided=["g"], reduction="r"))
    reader = dict(kind="elementwise", shape=[64, 384], names=["z"], reads=["y"], writes=["z"],
                  body=["const float r_z = (v_y[(i)] + 1.0f); v_z[i] = r_z;"])
    return [group, reader], layouts, dtypes


def _remat(phases, layouts, dtypes, kept=()):
    return rr.remat_products(phases, layouts, kept_roots=set(kept), ident=lambda v: "v_" + v,
                             numel=lambda s: math.prod(s), dtypes=dtypes)


def test_rremat_recomputes_row_scaled_values_at_readers():
    phases, layouts, dtypes = _remat_fixture()
    group, reader = phases
    with rules():
        assert _remat(phases, layouts, dtypes) == ([], set(), set())
    assert group["code"] == GROUP
    with rules(accumulation="rremat"):
        recs, dead, now = _remat(phases, layouts, dtypes)
    assert dead == {"y"} and now == {"g"} and recs[0]["factor"] == "g"
    assert "v_y[i] =" not in group["code"] and "v_g[i] = r_g;" in group["code"]
    assert reader["body"][0] == ("const float r_z = ((v_g[(i)] * v_s[(long long)((unsigned)(((unsigned)(i) / 384u)) "
                                 "% 64u) * 40LL]) + 1.0f); v_z[i] = r_z;")
    assert set(reader["reads"]) == {"g", "s"} and reader["row_scale_inputs"] == ["y"]
    assert group["row_fusion"]["stored"] == ["g"]


def test_rremat_keeps_kept_values():
    phases, layouts, dtypes = _remat_fixture()
    with rules(accumulation="rremat"):
        assert _remat(phases, layouts, dtypes, kept={"y"})[1] == set()


@pytest.mark.parametrize("lit", sorted(rr.VERIFIED_DIVISORS))
def test_verified_divisors_are_finite_positive_literals(lit):
    assert lit.endswith("f") and float(lit[:-1]) > 0
