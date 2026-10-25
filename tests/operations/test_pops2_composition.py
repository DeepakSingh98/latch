"""Metadata contracts for A products inside D's existing staged loop ABI.

These check resource ownership and fallback selection, not numerical accuracy.
Native complete-update qualification remains a separate cloud gate.
"""
from __future__ import annotations

import re
import traceback

from megakernel_lab.operation_compiler.lower import TensorLayout, _contiguous, _lower_serial
from megakernel_lab.operation_compiler.matrix_codegen import fp32_async
from megakernel_lab.operation_compiler.loop_pipeline import PIPELINED_MATRIX_ENGINE


def _layout(shape, name):
    return TensorLayout(shape, _contiguous(shape), name)


def _plan(name="out", **options):
    return fp32_async("a", "b", name, _layout((64, 4096), "a"),
                      _layout((4096, 64), "b"), (64, 64),
                      ident=lambda value: "v_" + value,
                      engine=PIPELINED_MATRIX_ENGINE, **options)


def persistent_selected_product():
    plan, reason = _plan(counters_at="latch_ctr + 64LL")
    assert plan, reason
    meta = plan["metadata"]
    assert meta["engine"] == "pops2_fp32" and meta["ksplit"] > 1
    assert meta["persistent_counter_words"] > 0
    assert plan["counter_init"] is None
    scratch, = plan["scratch"]
    assert scratch["role"] == "partials" and scratch["counter_words"] == 0
    assert scratch["shape"] == (scratch["partial_words"],)
    assert "reinterpret_cast<unsigned*>(latch_ctr + 64LL)" in plan["code"]
    assert "/*P2_" not in plan["code"] and "/*B_VBASE*/" in plan["code"]
    assert "#define POPS_INLINE\n" not in plan["helpers"]
    if "fp32_simt.cuh" in plan["headers"]:
        assert "#define POPS2_NOINLINE" in plan["helpers"]
    return {"engine": meta["engine"], "body": meta["body"],
            "counter_words": meta["persistent_counter_words"]}


def private_scratch_and_fallback():
    plan, reason = _plan()
    assert plan, reason
    scratch, = plan["scratch"]
    assert plan["counter_init"]["writes"] == [scratch["name"]]
    assert scratch["shape"] == (scratch["counter_words"] + scratch["partial_words"],)
    # A collision cannot bind a graph value as private mutable scratch. The
    # D adapter may choose its independently named, admitted B scratch instead.
    fallback, reason = _plan(occupied_names=[scratch["name"]])
    assert fallback and fallback["metadata"]["engine"] == "popsb_cpasync", reason
    assert fallback["scratch"][0]["name"] != scratch["name"]
    accumulating, reason = _plan(accumulate=True, counters_at="latch_ctr")
    assert accumulating, reason
    assert accumulating["metadata"]["engine"] == "popsb_cpasync"
    assert ",true>(" in accumulating["code"]
    assert "direct accumulated" in accumulating["metadata"]["matrix_selector_fallback"]
    return {"collision_retains_disjoint_scratch": True, "D_accumulation_retained": True}


def staged_codegen_keeps_full_grid():
    graph = {"inputs": ["a", "b"], "outputs": ["y"], "values": {
        "a": {"shape": [64, 4096], "dtype": "float32"},
        "b": {"shape": [4096, 64], "dtype": "float32"},
        "y": {"shape": [64, 64], "dtype": "float32"}},
        "operations": [{"name": "y", "target": "aten.mm.default", "args": ["a", "b"]}]}
    program = _lower_serial(graph, output_bindings=["store"], region_body=True,
                            barrier_schedule="dependency", staged=True, partition_phases=True,
                            split_counters="persistent", matrix_engine=PIPELINED_MATRIX_ENGINE)
    matrix, = program.manifest["phases"]
    assert matrix["engine"] == "pops2_fp32"
    assert program.manifest["persistent_counter_words"] > 0
    assert "float* latch_ctr" in program.source
    assert "/*P2_" not in program.source and "/*B_VBASE*/" not in program.source
    stage, = program.manifest["stages"]
    assert stage["gemm_tiles"] > 0 and stage["jobs"] == 0
    assert "(vrot + 0), smem)" in program.source
    return {"persistent_counter_words": program.manifest["persistent_counter_words"],
            "full_grid_stage": True}


def canonical_functors_are_region_safe():
    from megakernel_lab.operation_compiler.fusion import lower_fused
    from .test_pops2b import _sgd_bf16_graph
    from megakernel_lab.operation_compiler import pops2b_rules
    old = pops2b_rules.EPI_COST_LIMIT
    pops2b_rules.EPI_COST_LIMIT = 64
    try:
        program = lower_fused(_sgd_bf16_graph(), pops2b_casts=True)
    finally:
        pops2b_rules.EPI_COST_LIMIT = old
    names = re.findall(r"struct (latch_fn_[0-9a-f]+) ", program.helpers)
    assert names
    for name in names:
        assert f"#ifndef {name.upper()}_DEFINED\n#define {name.upper()}_DEFINED" in program.helpers
    return {"guarded_canonical_functors": len(names)}


def run():
    results = []
    for test in (persistent_selected_product, private_scratch_and_fallback,
                 staged_codegen_keeps_full_grid, canonical_functors_are_region_safe):
        try:
            results.append({"name": test.__name__, "passed": True, "detail": test()})
        except Exception:
            results.append({"name": test.__name__, "passed": False, "error": traceback.format_exc()})
    return {"kind": "pops2-region-composition-contract", "tests": results,
            "passed": all(row["passed"] for row in results)}
