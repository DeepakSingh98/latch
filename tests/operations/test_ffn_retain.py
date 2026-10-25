"""LATCH-FFN-RETAIN CPU contracts (lowering structure only; GPU bitwise A/B evidence is in the lane receipts).

1. Normal compose compilation retains the established P6D schedule for an attention-free explicit-cast program whose
   P5C products are proven by the homogeneous protocol proof; policy-captured programs are not candidates.
2. The direct column engine handles column reductions of different widths in one phase and keeps the frozen tree.
3. Per-row operands are re-indexed only for the exact (r * E + oc) / E form.
4. Literal / literal divisions do not make an epilogue ALU-heavy; raw FP32 products plan onto the warp-specialized
   body and never share a group with rounded products.
"""
from __future__ import annotations

import contextlib

import pytest

from megakernel_lab.operation_compiler import compose, p5c_matrix, p9_rowblock
from .test_dense_exec import _graph, exact_product_graph


@contextlib.contextmanager
def _set(module, **values):
    old = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


def explicit_graph(m=4096, n=2560, k=640):
    """The opmath form of a BF16 linear with bias (FP32 product of widened BF16 operands, one BF16 rounding), then an
    explicit BF16 product of its rounded result (as in the FFN backward): a P6D program without attention."""
    def build(value, op, r):
        value("x", (m, k), role="input")
        value("w", (n, k), role="input")
        value("b", (n,), role="input")
        value("xb", (m, k), dtype="bfloat16"); op("xb", "aten._to_copy.default", [r("x")], {"dtype": "torch.bfloat16"})
        value("wb", (n, k), dtype="bfloat16"); op("wb", "aten._to_copy.default", [r("w")], {"dtype": "torch.bfloat16"})
        value("xf", (m, k)); op("xf", "aten._to_copy.default", [r("xb")], {"dtype": "torch.float32"})
        value("wf", (n, k)); op("wf", "aten._to_copy.default", [r("wb")], {"dtype": "torch.float32"})
        value("wt", (k, n), strides=(1, k)); op("wt", "aten.t.default", [r("wf")])
        value("p", (m, n)); op("p", "aten.mm.default", [r("xf"), r("wt")])
        value("y", (m, n)); op("y", "aten.add.Tensor", [r("p"), r("b")])
        value("yb", (m, n), dtype="bfloat16"); op("yb", "aten._to_copy.default", [r("y")], {"dtype": "torch.bfloat16"})
        value("g", (m, k), dtype="bfloat16"); op("g", "aten.mm.default", [r("yb"), r("wb")])
        return ["x", "w", "b"], ["g"]
    return _graph(build)


def _compose(graph):
    from megakernel_lab.operation_compiler.lower import lower
    return lower(graph, execution="compose", target_arch="sm_90")


def test_explicit_cast_flat_program_retains_p6d():
    program = _compose(explicit_graph())
    selection = program.manifest["execution_selection"]
    assert selection["selected"] == "legacy_static", selection
    assert selection.get("protocol_retention_candidate") == "p6d_explicit_flat"
    assert selection["retained_protocol"]["attention_groups"] == []
    assert selection["retained_protocol"]["matrix_bodies"]
    assert "attention-free explicit-cast" in selection["reasons"][0]


def test_explicit_flat_switch_reproduces_common_planning():
    with _set(compose, EXPLICIT_FLAT_PROTOCOL=False):
        program = _compose(explicit_graph())
    assert program.manifest["execution_selection"]["selected"] == "common_static"


def test_policy_captured_programs_are_not_explicit_candidates():
    graph = exact_product_graph()
    manifest = {"phases": [{"kind": "gemm", "matrix_spec": {"engine": "p5c"}}], "attention_groups": []}
    assert compose.explicit_flat_candidate(graph, manifest)
    graph["metadata"] = {"precision_policy": {"name": "declared"}}
    assert not compose.explicit_flat_candidate(graph, manifest)
    manifest["phases"].append({"kind": "attention_long_forward", "body_interface": "compose-v1"})
    del graph["metadata"]
    assert not compose.explicit_flat_candidate(graph, manifest)


@pytest.mark.parametrize("index,expected", [
    ("(((unsigned)((unsigned)r * 640u + (unsigned)oc * 1u)) / 640u)", True),
    ("((((((unsigned)((unsigned)r * 640u + (unsigned)oc * 1u)) / 640u))))", True),
    ("((unsigned)r * 640u + (unsigned)oc) / 640u", True),
    ("(unsigned)r * 640u + ((unsigned)oc * 1u / 640u)", False),     # precedence differs: not the row
    ("((unsigned)r * 640u + (unsigned)oc * 1u) / 320u", False),
    ("((unsigned)r * 641u + (unsigned)oc * 1u) / 640u", False),
    ("((unsigned)rr * 640u + (unsigned)oc) / 640u", False),
    ("((unsigned)r * 640u + (unsigned)oc * 2u) / 640u", False),
    ("((unsigned)r * 640u + (unsigned)oc * 1u) % 640u", False),
])
def test_row_index_rewrite_only_for_the_exact_form(index, expected):
    assert p9_rowblock._row_of_column_index(index, 640) is expected


def column_sums_graph(rows=512, widths=(2560, 640)):
    """BF16 column sums of different widths over the same rows (bias gradients of two projections)."""
    def build(value, op, r):
        inputs, outputs = [], []
        for i, width in enumerate(widths):
            value(f"x{i}", (rows, width), dtype="bfloat16", role="input")
            value(f"s{i}", (1, width), dtype="bfloat16")
            op(f"s{i}", "aten.sum.dim_IntList", [r(f"x{i}"), [0], True])
            value(f"f{i}", (1, width))
            op(f"f{i}", "aten._to_copy.default", [r(f"s{i}")], {"dtype": "torch.float32"})
            inputs.append(f"x{i}")
            outputs.append(f"f{i}")
        return inputs, outputs
    return _graph(build)


def test_direct_engine_runs_mixed_width_column_sums():
    from megakernel_lab.operation_compiler.lower import lower
    program = lower(column_sums_graph(), schedule="p6d")
    direct = [p for p in program.manifest["phases"] if p.get("engine") == "p9_column_direct"]
    assert direct, [p.get("engine") for p in program.manifest["phases"]]
    info = direct[0]["p9_rowblock"]
    assert info["rows"] == 512 and sorted(info["extents"]) == [640, 2560]
    source = program.source
    # Frozen leaf order: virtual thread t sums rows t + 256 j (ascending j) from +0; then the fixed butterflies.
    assert "const long long r = t0 + u + 256LL * (j + jj);" in source
    assert "s101h::warp_sum(smem[(32 * i + lane) * 33 + oo])" in source
    with _set(p9_rowblock, COLUMN_DIRECT=False):
        legacy = lower(column_sums_graph(), schedule="p6d")
    assert not any(p.get("engine") == "p9_column_direct" for p in legacy.manifest["phases"])


def test_constant_divisions_are_not_alu_heavy():
    assert not p5c_matrix._alu_heavy("r[j] = ((1.0f / 10485760.0f) * (x * 2.0f));")
    assert not p5c_matrix._alu_heavy("r[j] = (x * (2.5e-3f / 3.0f));")
    assert p5c_matrix._alu_heavy("r[j] = (x / y);")
    assert p5c_matrix._alu_heavy("r[j] = (x / 640.0f);")
    assert p5c_matrix._alu_heavy("r[j] = tanhf(x);")


def _raw_spec(**extra):
    spec = {"engine": "p5c", "m": 16384, "n": 640, "k": 2560, "ak": True, "bkc": True, "out_raw": True,
            "ws_ok": True, "out": "nullptr", "epi": {"run": "[&](int e, unsigned i0, const float (&gv)[8]) { }",
                                                      "ld": "[&](int e, unsigned i0) { }", "prelude": ""}}
    spec.update(extra)
    return spec


def test_raw_products_plan_onto_the_warp_specialized_body():
    plan = p5c_matrix.plan(_raw_spec(), 132)
    assert plan.get("ws") and plan.get("out_raw") and plan["bm"] in (64, 128)
    with _set(p5c_matrix, P8WS_RAW=False):
        plain = p5c_matrix.plan(_raw_spec(), 132)
    assert not plain.get("ws") and plain.get("out_raw")
    heavy = _raw_spec(epi={"run": "[&](int e, unsigned i0, const float (&gv)[8]) { r[j] = tanhf(gv[j]); }",
                           "ld": "[&](int e, unsigned i0) { }", "prelude": ""})
    assert not p5c_matrix.plan(heavy, 132).get("ws")
    short = _raw_spec(n=2560, k=640)          # 10 k-steps per tile: the FP32 epilogue would not hide under the mainloop
    assert not p5c_matrix.plan(short, 132).get("ws")


def test_step13_upgrades_keep_common_planning():
    # A common candidate whose products the step-13 structural default would move to provider bodies keeps common
    # planning: the established emitter cannot run those bodies (mixed-operation programs of the integration suite).
    with _set(compose, step13_would_upgrade=lambda provider: True):
        program = _compose(explicit_graph())
    selection = program.manifest["execution_selection"]
    assert selection["selected"] == "common_static"
    assert any("step-13 structural default" in reason for reason in selection["reasons"]), selection["reasons"]
    assert not compose.step13_would_upgrade(None)


def explicit_graph_with_norm_columns():
    graph = explicit_graph()
    def columns(value, op, r):
        value("norm_grad", (4096, 640), role="input")
        value("norm_sum", (640,))
        op("norm_sum", "aten.sum.dim_IntList", [r("norm_grad"), [0]])
        return ["norm_grad"], ["norm_sum"]
    other = _graph(columns)
    graph["values"].update(other["values"])
    for key in ("operations", "inputs", "outputs"):
        graph[key].extend(other[key])
    return graph


def test_optional_norm_columns_do_not_preempt_explicit_flat_retention():
    from megakernel_lab.operation_compiler.lower import lower, _FLAT_COLSPLIT_PROBE
    graph = explicit_graph_with_norm_columns()
    program = _compose(graph)
    assert program.manifest["execution_selection"]["selected"] == "legacy_static"
    assert program.manifest["execution_selection"]["retained_protocol"]["matrix_bodies"]
    assert program.source == lower(graph).source
    assert _FLAT_COLSPLIT_PROBE.get() is None


@pytest.mark.parametrize("refusal", ["unfused", "body_proof"])
def test_common_program_keeps_original_column_split_source(monkeypatch, refusal):
    from megakernel_lab.operation_compiler.lower import lower, _FLAT_COLSPLIT_PROBE
    graph = explicit_graph_with_norm_columns()
    options = {"fuse_elementwise": False} if refusal == "unfused" else {}
    if refusal == "body_proof":
        monkeypatch.setattr(compose, "verify_legacy_long_protocol",
                            lambda *args: (None, "injected mismatched body contract"))
    program = lower(graph, execution="compose", **options)
    common = lower(graph, execution="compose", compose_selection="common", **options)
    selection = program.manifest["execution_selection"]
    assert selection["selected"] == "common_static"
    assert selection["common_body_upgrade"]["column_splits"] == ["norm_sum"]
    assert any(p.get("engine") == "flat_colsplit" for p in program.manifest["phases"])
    assert program.source == common.source and program.cache_key == common.cache_key
    assert _FLAT_COLSPLIT_PROBE.get() is None


def test_convolution_column_admission_is_not_deferred():
    from types import SimpleNamespace
    from megakernel_lab.operation_compiler.lower import lower
    graph = SimpleNamespace(**explicit_graph_with_norm_columns(), metadata={"channels_last": True})
    program = lower(graph, execution="compose")
    common = lower(graph, execution="compose", compose_selection="common")
    assert program.manifest["execution_selection"]["selected"] == "common_static"
    assert "common_body_upgrade" not in program.manifest["execution_selection"]
    assert any(p.get("engine") == "flat_colsplit" for p in program.manifest["phases"])
    assert program.cache_key == common.cache_key


def test_column_probe_context_restores_after_emitter_failure(monkeypatch):
    import functools
    import importlib
    module = importlib.import_module('megakernel_lab.operation_compiler.lower')
    @functools.wraps(module._lower_serial)
    def fail(*args, **kwargs):
        assert module._FLAT_COLSPLIT_PROBE.get() == []
        raise RuntimeError("injected emitter failure")
    monkeypatch.setattr(module, "_lower_serial", fail)
    outer = ["outer probe"]
    token = module._FLAT_COLSPLIT_PROBE.set(outer)
    try:
        with pytest.raises(RuntimeError, match="injected emitter failure"):
            compose.lower_composed_flat(explicit_graph_with_norm_columns(), selection_probe=True)
        assert module._FLAT_COLSPLIT_PROBE.get() is outer
    finally:
        module._FLAT_COLSPLIT_PROBE.reset(token)
