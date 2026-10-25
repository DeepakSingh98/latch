"""LATCH-DENSE-EXEC CPU contracts: P6D retention for clipped ordinary updates (lowering structure only).

1. P9-A late lone loops never merge a loop with a same-phase reader that reads it through a broadcast index.
2. Long single-output FP32 sums in P6D use the existing multi_cta_full body with declared scratch and contract.
3. compose.view_read_hazards accepts exactly the same-loop register reads the emitter declares.
"""
from __future__ import annotations

import contextlib

import pytest

from megakernel_lab.operation_compiler import compose, fusion_p6d


def _graph(build):
    values, ops = {}, []

    def value(name, shape, dtype="float32", strides=None, role="intermediate"):
        if strides is None:
            strides, acc = [], 1
            for d in reversed(shape):
                strides.insert(0, acc)
                acc *= d
        values[name] = {"name": name, "shape": list(shape), "dtype": dtype, "strides": list(strides), "role": role}

    def op(name, target, args, kwargs=None):
        ops.append({"name": name, "target": target, "args": args, "kwargs": kwargs or {}})
    inputs, outputs = build(value, op, lambda n: {"ref": n})
    return {"values": values, "operations": ops, "inputs": inputs, "outputs": outputs}


def clip_graph(n=65536, terms=24):
    """Global-norm clip chain as captured for AdamW max_grad_norm: per-tensor long sums of squares, an add chain,
    sqrt, +eps, reciprocal, NaN-preserving clamp, then the broadcast scale of every gradient."""
    def build(value, op, r):
        names = [f"g{i}" for i in range(terms)]
        for name in names:
            value(name, (n,), role="input")
        total = None
        for i, name in enumerate(names):
            value(f"q{i}", (n,)); op(f"q{i}", "aten.mul.Tensor", [r(name), r(name)])
            value(f"s{i}", ()); op(f"s{i}", "aten.sum.default", [r(f"q{i}")])
            if total is None:
                total = f"s{i}"
            else:
                value(f"t{i}", ()); op(f"t{i}", "aten.add.Tensor", [r(total), r(f"s{i}")])
                total = f"t{i}"
        value("nrm", ()); op("nrm", "aten.sqrt.default", [r(total)])
        value("den", ()); op("den", "aten.add.Tensor", [r("nrm"), 1e-06])
        value("full", ()); op("full", "aten.full_like.default", [r("den"), 1.0])
        value("coef", ()); op("coef", "aten.div.Tensor", [r("full"), r("den")])
        value("nan", (), dtype="bool"); op("nan", "aten.ne.Tensor", [r("coef"), r("coef")])
        value("cl", ()); op("cl", "aten.clamp.default", [r("coef"), None, 1.0])
        value("w", ()); op("w", "aten.where.self", [r("nan"), r("coef"), r("cl")])
        for i, name in enumerate(names):
            value(f"o{i}", (n,)); op(f"o{i}", "aten.mul.Tensor", [r(name), r("w")])
        return names, [f"o{i}" for i in range(terms)] + [total]
    return _graph(build)


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


def _lower(graph):
    from megakernel_lab.operation_compiler.lower import lower
    return lower(graph, schedule="p6d")


def test_late_loops_keep_broadcast_readers_out_of_the_producer_loop():
    # The captured AdamW global-norm chain: the NaN test reads the 0-d coefficient (inlined from the norm) at
    # broadcast index 0u, a memory read. Merging the norm's lone loop into the reader's phase made P6D's own
    # schedule check refuse the whole program (the base compiler raises here).
    program = _lower(clip_graph())
    phase_of = {}
    for index, phase in enumerate(program.manifest["phases"]):
        for name in phase["names"]:
            phase_of[name] = index
    # The materialized end of the norm chain (sqrt and +eps inline into its readers) precedes every 0-d reader.
    producer = next(n for n in ("nrm", "den", "t23") if n in phase_of)
    readers = [n for n in ("nan", "coef", "w") if n in phase_of]
    assert readers, sorted(phase_of)
    assert all(phase_of[producer] < phase_of[n] for n in readers)


def test_long_scalar_sums_use_the_multi_cta_full_body():
    program = _lower(clip_graph())
    m = program.manifest
    full = [p for p in m["phases"] if p.get("engine") == "multi_cta_full"]
    assert sorted(p["names"][0] for p in full) == sorted(f"s{i}" for i in range(24))
    assert all(p["reduction_elements"] == 65536 and p["requires_full_grid"] and p["max_grid_blocks"] == 1024
               and p["arithmetic"] == "per-CTA contiguous chunk trees, then fixed-order combine of CTA partials"
               for p in full)
    scratch = m["matrix_scratch"]
    assert scratch["counter_words"] == 24 and scratch["partial_words"] == 24 * 1024
    assert scratch["counter_word_offset"] + 24 + 24 * 1024 <= m["arena_words"]
    assert "v_fullctr[i_] = 0u" in program.source and program.source.count("msum_last_ = atomicAdd") == 24
    assert "(v_fullctr + 0)" in program.source and "(v_fullctr + 23)" in program.source
    with _set(fusion_p6d, P6D_FULL_REDUCTION=False):
        old = _lower(clip_graph())
    assert not any(p.get("engine") == "multi_cta_full" for p in old.manifest["phases"])
    assert "matrix_scratch" not in old.manifest and "v_fullctr" not in old.source


def test_short_sums_keep_the_existing_tree():
    program = _lower(clip_graph(n=4096))      # below FULL_REDUCTION_MIN (as the common planner)
    assert not any(p.get("engine") == "multi_cta_full" for p in program.manifest["phases"])


def test_common_clip_chain_keeps_clamp_in_register_after_level_remerge():
    from megakernel_lab.operation_compiler.lower import lower
    # Select the same common emitter that the full ordinary model requires.
    # Auto correctly retains the legacy emitter for this small isolated graph.
    program = lower(clip_graph(n=8, terms=2), execution="compose", compose_selection="common")
    # The installed ordinary researcher update previously emitted v_cl[i]
    # after removing its producer storage. Preserve NaN selection and the
    # separately rounded clamp value through split/remerge scheduling.
    where = next(line for line in program.source.splitlines() if 'const float r_w =' in line)
    assert '(r_nan ? r_coef : r_cl)' in where
    assert 'v_cl[' not in program.source


def identity_view_graph(rows=256, width=64):
    def build(value, op, r):
        value("a", (rows, width), role="input")
        value("b", (rows, width), role="input")
        value("y", (rows, width)); op("y", "aten.add.Tensor", [r("a"), r("b")])
        value("v", (rows * width,)); op("v", "aten.view.default", [r("y"), [rows * width]])
        value("c", (rows * width,), dtype="bfloat16")
        op("c", "aten._to_copy.default", [r("v")], {"dtype": "torch.bfloat16"})
        return ["a", "b"], ["c", "y"]
    return _graph(build)


def test_declared_register_reads_clear_only_their_view_hazards():
    graph = identity_view_graph()
    program = _lower(graph)
    m = program.manifest
    assert "c" in m["register_reads"].get("y", [])
    assert compose.view_read_hazards(graph, m)["same_phase_view_consumers"] == []
    undeclared = {**m}
    undeclared.pop("register_reads")
    same = [i for i, p in enumerate(m["phases"]) if {"y", "c"} <= set(p["names"])]
    if same:      # both in one loop: without the declaration the conservative rule still reports the read
        assert compose.view_read_hazards(graph, undeclared)["same_phase_view_consumers"] == ["c"]
    wrong = {**m, "register_reads": {"y": ["someone_else"]}}
    if same:
        assert compose.view_read_hazards(graph, wrong)["same_phase_view_consumers"] == ["c"]


def exact_product_graph(m=4096, n=2560, k=640, exact=True):
    """FP32 mm of widened BF16 operands (+ bias, one BF16 rounding), the opmath form of a BF16 linear with bias."""
    def build(value, op, r):
        value("x", (m, k), role="input")
        value("w", (n, k), role="input")
        value("b", (n,), role="input")
        value("xb", (m, k), dtype="bfloat16"); op("xb", "aten._to_copy.default", [r("x")], {"dtype": "torch.bfloat16"})
        value("wb", (n, k), dtype="bfloat16"); op("wb", "aten._to_copy.default", [r("w")], {"dtype": "torch.bfloat16"})
        if exact:
            value("xf", (m, k)); op("xf", "aten._to_copy.default", [r("xb")], {"dtype": "torch.float32"})
            value("wf", (n, k)); op("wf", "aten._to_copy.default", [r("wb")], {"dtype": "torch.float32"})
        else:
            value("xf", (m, k)); op("xf", "aten.mul.Tensor", [r("x"), 1.5])
            value("wf", (n, k)); op("wf", "aten.mul.Tensor", [r("w"), 1.5])
        value("wt", (k, n), strides=(1, k)); op("wt", "aten.t.default", [r("wf")])
        value("p", (m, n)); op("p", "aten.mm.default", [r("xf"), r("wt")])
        value("y", (m, n)); op("y", "aten.add.Tensor", [r("p"), r("b")])
        value("yb", (m, n), dtype="bfloat16"); op("yb", "aten._to_copy.default", [r("y")], {"dtype": "torch.bfloat16"})
        return ["x", "w", "b"], ["yb"]
    return _graph(build)


@pytest.mark.parametrize("schedule", ["p6d", "compose"])
def test_bf16_exact_fp32_product_uses_p5c_raw_output(schedule):
    from megakernel_lab.operation_compiler.lower import lower
    program = (lower(exact_product_graph(), schedule="p6d") if schedule == "p6d"
               else lower(exact_product_graph(), execution="compose", target_arch="sm_90"))
    decisions = [d for d in program.manifest["matrix_decisions"] if d.get("output") == "p"]
    assert any("BF16-exact FP32 product" in d.get("reason", "") and d.get("selected") for d in decisions), decisions
    assert ",3," in program.source or ",3>" in program.source        # p5c OUT == 3 (unrounded FP32 result)


def test_non_exact_fp32_product_keeps_fp32_bodies():
    from megakernel_lab.operation_compiler.lower import lower
    program = lower(exact_product_graph(exact=False), schedule="p6d")
    assert not any("BF16-exact" in d.get("reason", "") for d in program.manifest["matrix_decisions"])


def test_bf16_exact_reuse_rewrites_only_proven_readers():
    import torch
    from megakernel_lab.operation_compiler.ir import Operation, Ref, TrainingIR, Value
    from megakernel_lab.operation_compiler.bf16_exact import reuse

    def v(name, shape, dtype="float32", role="temporary"):
        strides, acc = [], 1
        for d in reversed(shape):
            strides.insert(0, acc)
            acc *= d
        return Value(name, tuple(shape), dtype, tuple(strides), role=role)

    def o(name, target, args, kwargs=None):
        reads = tuple(a.name for a in args if isinstance(a, Ref))
        return Operation(name, target, tuple(args), kwargs or {}, reads, (name,), (), reads, ())
    shape = (64, 64)
    values = {n: v(n, shape, d, role) for n, d, role in (("s", "bfloat16", "input"), ("g", "float32", "input"),
                                                        ("w", "bfloat16", "input"), ("f", "float32", "temporary"),
                                                        ("r", "float32", "temporary"), ("n", "bfloat16", "temporary"),
                                                        ("p", "bfloat16", "temporary"), ("m", "float32", "temporary"))}
    ops = [o("f", "aten._to_copy.default", [Ref("s")], {"dtype": torch.float32}),
           o("r", "aten.relu.default", [Ref("f")]),
           o("n", "aten._to_copy.default", [Ref("r")], {"dtype": torch.bfloat16}),
           o("p", "aten.mm.default", [Ref("n"), Ref("w")]),
           o("m", "aten.threshold_backward.default", [Ref("g"), Ref("r"), 0])]
    graph = TrainingIR(values, ops, ("s", "g", "w"), {"p": Ref("p"), "m": Ref("m")})
    rewritten, report = reuse(graph)
    assert [row["value"] for row in report["rewritten"]] == ["r"]
    mask = next(op for op in rewritten.operations if op.name == "m")
    assert mask.args[1] == Ref("bf16x_n") and "bf16x_n" in rewritten.values
    names = [op.name for op in rewritten.operations]
    assert names.index("bf16x_n") == names.index("n") + 1
    # Without a product reading the BF16 copy, nothing changes (a widened read could add storage).
    ops2 = [op for op in ops if op.name != "p"]
    unchanged, report2 = reuse(TrainingIR(values, ops2, ("s", "g", "w"), {"m": Ref("m"), "n": Ref("n")}))
    assert not report2["rewritten"]
