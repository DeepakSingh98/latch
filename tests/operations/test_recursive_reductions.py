"""Structural preservation of the ordered donor's full-grid reduction body.

Run on the lane's CPU worker. Native equality still requires matched physical
grid sizes and the complete-update GPU checks; strict FP32 alone is not proof
that two reduction trees evaluate the same floating-point calculation.
"""
import pytest
from types import SimpleNamespace

from megakernel_lab.operation_compiler.lower import LoweringError, lower
from megakernel_lab.operation_compiler.loop_region_executor import wrap_region


def _graph(extent, target="mean"):
    return {"inputs": ["x"], "outputs": ["result"], "values": {
        "x": {"shape": [extent], "strides": [1], "dtype": "float32"},
        "result": {"shape": [], "strides": [], "dtype": "float32"}},
        "operations": [{"name": "result", "target": f"aten.{target}.default", "args": [{"ref": "x"}]}]}


def _staged(graph, binding="store", **options):
    return lower(graph, execution="compose", output_bindings=[binding], region_body=True,
                 staged=True, barrier_schedule="dependency", preserve_full_reductions=True, **options)


def _full_body(source):
    start = source.index("if (gridDim.x <= 1024) {")
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += {"{": 1, "}": -1}.get(source[end], 0)
        end += 1
    return source[start:end]


@pytest.mark.parametrize("target", ["sum", "mean"])
@pytest.mark.parametrize("extent", [16384, 65536])
@pytest.mark.parametrize("index_rules", [False, "idx32+idxid"])
def test_staged_store_retains_identical_donor_full_grid_calculation(target, extent, index_rules):
    graph = _graph(extent, target)
    # Match indexing settings explicitly: standalone compose otherwise selects
    # uint32 identity addressing while a staged region defaults to int64.
    donor = lower(graph, execution="compose", compose_selection="common", group_rules=index_rules)
    staged = _staged(graph, group_rules=index_rules)
    before = next(p for p in donor.manifest["phases"] if p.get("engine") == "multi_cta_full")
    after = next(p for p in staged.manifest["phases"] if p.get("engine") == "multi_cta_full")
    for key in ("engine", "reduction_elements", "arithmetic", "requires_full_grid", "max_grid_blocks"):
        assert before[key] == after[key]
    assert _full_body(donor.source) == _full_body(staged.source)
    assert staged.manifest["region_reduction_policy"]["outputs"] == ["result"]
    assert staged.manifest["max_grid_blocks"] == 1024
    assert "if (*max_blocks > 1024) *max_blocks = 1024;" in staged.source


@pytest.mark.parametrize("binding", ["store", "accumulate"])
def test_counter_initializer_precedes_reduction_and_scratch_is_private(binding):
    program = _staged(_graph(32768), binding)
    manifest = program.manifest
    init, body = manifest["phases"]
    scratch = init["writes"][0]
    assert init["kind"] == "reduction_counter_init" and init["reads"] == []
    assert body["engine"] == "multi_cta_full" and body["internal_mutations"] == [scratch]
    assert scratch in body["reads"] and scratch not in manifest["inputs"] + manifest["outputs"]
    assert body["private_scratch"] == [{"name": scratch, "counter_words": 1,
                                       "partial_word_offset": 64, "partial_words": 1024}]
    assert body["counter_effects"] == [{"storage": scratch, "word_offset": 0, "words": 1,
                                       "initialization": "zeroed by dependency predecessor",
                                       "completion": "last arriver restores zero"}]
    assert manifest["barrier_intervals"] == [[0], [1]]
    first, second = manifest["stages"]
    assert first["interval"] < second["interval"]
    assert second["writes_outputs"] == [0]
    assert second["accumulates"] == ([0] if binding == "accumulate" else [])
    assert manifest["offsets"][scratch] + 1088 <= manifest["arena_words"]
    body_source = _full_body(program.source)
    assert body_source.count("v_result[0] =") == 1
    assert ("v_result[0] = v_result[0] + (acc" in body_source) == (binding == "accumulate")
    assert "float acc = 0.0f;" in body_source
    assert "grid.sync();" not in program.kernel_body


def test_small_scalar_and_existing_staged_policy_are_unchanged():
    small = _staged(_graph(8192))
    assert small.manifest["region_reduction_policy"]["outputs"] == []
    assert not any(p.get("engine") == "multi_cta_full" for p in small.manifest["phases"])
    graph = _graph(32768)
    existing = lower(graph, execution="compose", output_bindings=["store"], region_body=True,
                     staged=True, barrier_schedule="dependency")
    assert "region_reduction_policy" not in existing.manifest
    assert not any(p.get("engine") == "multi_cta_full" for p in existing.manifest["phases"])


def test_full_grid_body_does_not_become_a_partitioned_legacy_reduction():
    program = lower(_graph(32768), output_bindings=["store"], region_body=True, staged=True,
                    barrier_schedule="dependency", preserve_full_reductions=True, partition_phases=True)
    body_source = _full_body(program.source)
    assert "gridDim.x" in body_source and "blockIdx.x" in body_source
    assert "lb_" not in body_source and "lg_" not in body_source


def test_preservation_flag_requires_a_staged_region():
    with pytest.raises(LoweringError, match="staged region"):
        lower(_graph(32768), preserve_full_reductions=True, policy="reference")


def test_persistent_full_reduction_retains_tree_with_an_executor_owned_counter():
    program = _staged(_graph(32768), "accumulate", split_counters="persistent")
    manifest = program.manifest
    body, = manifest["phases"]
    assert body["engine"] == "multi_cta_full" and body["requires_full_grid"]
    assert manifest["persistent_counter_words"] == 1
    assert body["counter_effects"] == [{"storage": "runtime_counter_arena", "word_offset": 0,
        "words": 1, "initialization": "zero once per enclosing executor invocation",
        "completion": "last arriver restores zero"}]
    assert len(manifest["stages"]) == 1
    source = _full_body(program.source)
    assert "float acc = 0.0f;" in source
    assert "v_result[0] = v_result[0] + (acc" in source
    assert "reinterpret_cast<unsigned*>(latch_ctr + 0LL)" in source
    assert body["arithmetic"] == "per-CTA contiguous chunk trees, then fixed-order combine of CTA partials"


def test_wrapper_forwards_dedicated_counters_without_resetting_nested_children():
    program = SimpleNamespace(manifest={
        "inputs": ["x"], "outputs": ["y"], "state_writes": {}, "placed_outputs": [0],
        "output_placement": {"y": {"output": 0, "mode": "store"}}, "stages": [{}],
        "persistent_counter_words": 7, "executor_counter_word_offset": 128,
    }, functions="", kernel_body="__device__ void __REGION___s0() {}")
    ordinary = wrap_region(program, "ordinary", ["store"], counter=True)
    nested = wrap_region(program, "nested", ["store"], counter=True, counter_offset=4096,
                         initialize_counters=False, final_barrier=False)
    assert "reinterpret_cast<unsigned*>(arena + 128LL)[i]=0u" in ordinary
    assert ordinary.count("latch_loop_barrier(loop_counter);") == 2
    assert "stage_s0(args, arena, out, 0, 0, (int)gridDim.x, arena + 4096LL);" in nested
    assert "[i]=0u" not in nested and "latch_loop_barrier(loop_counter);" not in nested


def test_inline_recursive_stage_calls_preserve_reduction_body_grid_and_barriers():
    program = _staged(_graph(32768), "accumulate", split_counters="persistent")
    program.manifest["executor_counter_word_offset"] = 4096
    ordinary = wrap_region(program, "ordinary", ["accumulate"], counter=True)
    program.manifest["recursive_stage_calls"] = "inline_local_tables"
    inlined = wrap_region(program, "inlined", ["accumulate"], counter=True)
    assert _full_body(ordinary) == _full_body(inlined)
    assert ordinary.count("latch_loop_barrier(loop_counter);") == inlined.count("latch_loop_barrier(loop_counter);")
    assert "__device__ __noinline__ void stage_s0" in ordinary
    assert "__device__ __forceinline__ void stage_s0" in inlined
    assert "float** p9_og = p9_opq(out);" in inlined
    assert "float* a0[] = {p9_ag[0]};" in inlined
    assert "float* o0[] = {p9_og[0]};" in inlined
    assert "stage_s0(a0, x0, o0, 0, 0, (int)gridDim.x, x0 + 4096LL); }\nlatch_loop_barrier" in inlined
    assert "P9_OPQ_DEFINED" not in inlined  # Every wrapper namespace declares its own helpers.


def test_recursive_inline_option_is_consumed_inherited_and_scoped(monkeypatch):
    import importlib
    from megakernel_lab.operation_compiler.loop_region_executor import lower_region
    lower_module = importlib.import_module("megakernel_lab.operation_compiler.lower")
    leaf = SimpleNamespace(outputs=["y"])
    nested = SimpleNamespace(outputs=["y"], nested_operation=True)

    def fake_lower(graph, **options):
        assert "recursive_inline_stages" not in options
        if getattr(graph, "nested_operation", False):
            return lower_region(leaf, ["store"])
        return SimpleNamespace(manifest={"arena_words": 64})

    monkeypatch.setattr(lower_module, "lower", fake_lower)
    assert lower_region(nested, ["store"], recursive_inline_stages=True).manifest[
        "recursive_stage_calls"] == "inline_local_tables"
    assert "recursive_stage_calls" not in lower_region(leaf, ["store"]).manifest
    assert "recursive_stage_calls" not in lower_region(nested, ["store"], recursive_inline_stages=False).manifest
    for invalid in (1, "all", None):
        with pytest.raises(LoweringError, match="recursive_inline_stages must be boolean"):
            lower_region(leaf, ["store"], recursive_inline_stages=invalid)


def test_nested_full_reductions_allocate_raw_counters_outside_child_scratch():
    from .test_nested_loop import fixture
    # Unlike the tiny general fixture, these real inner mean reductions cross
    # the full-grid threshold and exercise allocation of persistent counters.
    captured, _ = fixture(width=128, batch=128)
    region = captured.ir.body_saved
    program = lower(region, execution="compose", output_bindings=["store"] * len(region.outputs),
                    region_body=True, loop_counter=True)
    manifest = program.manifest
    counters = manifest["recursive_executor"]["persistent_counters"]
    assert counters["body_saved"]["words"] >= 2
    occupied = []
    for name, counter in counters.items():
        start, words = counter["word_offset"], counter["words"]
        assert words == manifest["regions"][name]["persistent_counter_words"]
        assert start % 64 == 0 and start >= manifest["region_scratch_bytes"] // 4
        assert start + words <= manifest["arena_words"]
        assert all(start + words <= low or start >= high for low, high in occupied)
        occupied.append((start, start + words))
        zero = f"reinterpret_cast<unsigned*>(arena + {start}LL)[i]=0u;"
        assert program.kernel_body.count(zero) == 1
        assert program.kernel_body.index(zero) < program.kernel_body.index("for (int inner_iteration=")
        assert zero not in program.functions
    assert "int32" not in {desc["dtype"] for desc in manifest["layouts"].values()}


def test_explicit_loop_region_options_override_inherited_defaults(monkeypatch):
    from megakernel_lab.operation_compiler import loop_lower
    graph = SimpleNamespace(max_steps=3, body_vjp=None, metadata={"nested_loop": True},
                            storage_mode="save_body_tape")
    received = {}

    class ReachedLoop(Exception):
        pass

    def inspect_options(graph, **kwargs):
        received.update(kwargs["options"]["region_options"])
        raise ReachedLoop

    monkeypatch.setattr(loop_lower, "lower_loop", inspect_options)
    overrides = {"row_fusion": False, "warp_softmax": False, "batched_reduction_loads": False,
                 "defer_accumulate": False, "epilogue_fusion": False, "write_through": False,
                 "split_counters": "phase"}
    with pytest.raises(ReachedLoop):
        lower(graph, row_fusion=True, loop_options={"region_options": overrides})
    assert all(received[key] == value for key, value in overrides.items())
    assert received["target_arch"] == "sm_90"
    with pytest.raises(LoweringError, match="compiler-option mapping"):
        lower(graph, loop_options={"region_options": []})
