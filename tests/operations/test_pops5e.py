"""Contract tests for the P-OPS5-E staged dense reduction rules (ops/pops5e_rules.py) and their lowering hooks."""
from __future__ import annotations

import pytest

from megakernel_lab.operation_compiler import pops5e_rules as P5


def test_simplify_rewrites_row_and_column_identities():
    text = ("v_m[(((((unsigned)((prow * 576u + pcol))) / 576u)))] + v_g[((((unsigned)((prow * 576u + pcol))) % 576u))]"
            " + v_x[(((unsigned)((prow * 576u + pcol))))]")
    out = P5.simplify(text, 576)
    assert "v_m[(((prow)))]" in out and "v_g[((pcol))]" in out
    assert "(prow * 576u + pcol)" in out          # identity reads are left for staging


def test_reads_and_stage_rewrite_only_identity_reads():
    leaf = "(__bfloat162float(v_a[(((unsigned)((prow * 64u + pcol))))]) * v_g[(pcol)])"
    reads = P5.reads([leaf])
    canon = P5._normalize(P5.position(64))
    assert reads["v_a"] == [canon] and reads["v_g"] != [canon]
    out = P5.stage_rewrite([leaf], {"v_a": "latch_s0", "__E__": 64})[0]
    assert "latch_s0[latch_lr * 64u + pcol]" in out and "v_g[(pcol)]" in out


def test_lift_classifies_invariants():
    row = "((__bfloat162float(latch_s0[latch_lr * 576u + pcol]) * (v_g[(pcol)] + (v_k[0u] * 0.0f))) * rsqrtf((v_m[prow] + 1e-05f)))"
    leaves, const, rows, cols = P5.lift([row])
    assert any("rsqrtf" in s for _, s in rows)
    assert any("v_g" in s for _, s in cols)
    assert "latch_s0" in leaves[0]
    # nothing element-dependent is lifted
    assert all("latch_s0" not in s for _, s in const + rows + cols)


def test_lift_terminates_on_temporaries_and_respects_allow_col():
    leaves, const, rows, cols = P5.lift(["((latch_s1[latch_lr * 8u + pcol]) * v_g[(pcol)])"], allow_col=False)
    assert not cols
    leaves, _, _, cols = P5.lift(["((latch_s1[latch_lr * 8u + pcol]) * v_g[(pcol)])"])
    assert len(cols) == 1 and "latch_hk" in leaves[0]


def test_share_extracts_common_element_subexpressions():
    r, c, decls = P5.share("((latch_s1[latch_lr * 8u + pcol] + latch_q) * a)", "((latch_s1[latch_lr * 8u + pcol] + latch_q) * b)")
    assert decls and decls[0][0] == "latch_e0" and "latch_e0" in r and "latch_e0" in c


@pytest.mark.parametrize("row_bytes", [2304, 4608, 8064, 13888])
def test_plans_fit_budget(row_bytes):
    for prefer in ("flight", "flight3", None, "stages"):
        plan = P5.plan(16384, 576, row_bytes, prefer=prefer)
        assert plan["tile"] * plan["stages"] * row_bytes <= P5.STAGE_BUDGET
        if prefer == "flight3":
            assert plan["stages"] >= 3
        if prefer == "flight":
            assert plan["tile"] in (1, 2, 4, 8, 16)


def test_emit_fused_single_barrier_and_row_only_exact_tree():
    row = "(__bfloat162float(v_a[(((unsigned)((prow * 576u + pcol))))]) * v_g[((((unsigned)((prow * 576u + pcol))) % 576u))])"
    col = "(v_b[((unsigned)((prow * 576u + pcol)))] * rsqrtf(v_m[(((((unsigned)((prow * 576u + pcol))) / 576u)))]))"
    code, shared = P5.emit(rows=16384, extent=576, row_out="v_out", row_leaf=row, col_part="v_part", col_leaf=col,
                           rounded=lambda e: e, scale="", staged=[("v_a", "__nv_bfloat16", 2), ("v_b", "float", 4)],
                           idents_bytes=None, tile=4, stages=3)
    assert "every warp is done with this stage" not in code       # fused: one barrier per tile
    assert "latch_pb" in code and shared > 0
    code2, _ = P5.emit(rows=16384, extent=576, row_out="v_out", row_leaf=row, col_part=None, col_leaf=None,
                       rounded=lambda e: e, scale="", staged=[("v_a", "__nv_bfloat16", 2)], idents_bytes=None,
                       tile=8, stages=3)
    # frozen tree: eight per-slot butterflies then the ordered second stage
    assert code2.count("latch_v[") >= 8 and "second = s101h::warp_sum(second)" in code2


def test_lowering_defaults_unchanged_and_stream_dispatch():
    """The retained legacy stream rule preserves its off/on dispatch contracts."""
    # The facade's fusion_reference predates the stream rule. P6E retains the
    # rule with independent on/off settings; production default tests remain.
    from megakernel_lab.operation_compiler import fusion_p6e
    from megakernel_lab.operation_compiler.execution_plan import execution_scope
    from megakernel_lab.operation_compiler.performance_schedule import _LOCK, _configuration
    def lower(graph, **options):
        with _LOCK, execution_scope('legacy'), _configuration('p6e', {}):
            return fusion_p6e.lower_fused(graph, **{'pops3c': True, 'pops4a': 'auto', **options})
    rows, ext = 2048, 512
    values = {
        "x": {"name": "x", "shape": [rows, ext], "dtype": "float32", "strides": [ext, 1], "role": "input"},
        "y": {"name": "y", "shape": [rows, ext], "dtype": "float32", "strides": [ext, 1], "role": "input"},
        "p": {"name": "p", "shape": [rows, ext], "dtype": "float32", "strides": [ext, 1], "role": "temporary"},
        "q": {"name": "q", "shape": [rows, ext], "dtype": "float32", "strides": [ext, 1], "role": "temporary"},
        "r": {"name": "r", "shape": [rows, 1], "dtype": "float32", "strides": [1, 1], "role": "temporary"},
        "c": {"name": "c", "shape": [1, ext], "dtype": "float32", "strides": [ext, 1], "role": "temporary"},
    }
    ops = [
        {"name": "p", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, {"ref": "y"}], "kwargs": {}},
        {"name": "q", "target": "aten.add.Tensor", "args": [{"ref": "x"}, {"ref": "y"}], "kwargs": {}},
        {"name": "c", "target": "aten.sum.dim_IntList", "args": [{"ref": "q"}, [0], True], "kwargs": {}},
        {"name": "r", "target": "aten.sum.dim_IntList", "args": [{"ref": "p"}, [1], True], "kwargs": {}},
    ]
    graph = {"inputs": ["x", "y"], "outputs": ["r", "c"], "values": values, "operations": ops}
    base = lower(graph)
    off = lower(graph, pops5e_stream=False)
    assert base.cache_key == off.cache_key
    on = lower(graph, pops5e_stream=True, pops5e_min_elements=0)
    decisions = on.manifest.get("pops5e_stream", [])
    assert decisions and decisions[0]["mode"] == "fused"
