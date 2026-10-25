"""Normal compiler API structural checks, run with the pinned Torch runtime."""
import pytest

from megakernel_lab.operation_compiler.lower import lower
from megakernel_lab.operation_compiler.performance_schedule import select_schedule


def _value(shape, dtype):
    stride, strides = 1, []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {"shape": shape, "strides": list(reversed(strides)), "dtype": dtype}


def test_auxiliary_bmm_does_not_suppress_dense_body():
    values = {"x": _value([1024, 512], "bfloat16"), "w": _value([512, 1024], "bfloat16"),
              "h": _value([1024, 1024], "bfloat16"), "a": _value([2, 17, 19], "float32"),
              "b": _value([2, 19, 23], "float32"), "aux": _value([2, 17, 23], "float32")}
    graph = {"inputs": ["x", "w", "a", "b"], "outputs": ["h", "aux"], "values": values,
             "operations": [
                 {"name": "h", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]},
                 {"name": "aux", "target": "aten.bmm.default", "args": [{"ref": "a"}, {"ref": "b"}]}]}
    assert select_schedule(graph)[0] == "reference"
    program = lower(graph, execution="compose", compose_selection="common")
    engines = {p.get("matrix_spec", {}).get("engine", p.get("engine")) for p in program.manifest["phases"]}
    assert "p5c" in engines
    assert "generic_simt_16" in engines
    assert program.manifest["tensor_maps"]
    assert "P5C_TMAP(" not in program.source and "P5C_TMAPR(" not in program.source
    assert program.manifest["execution_plan"]["stages"] == [[0, 1]]


def test_practical_activation_is_explicit_and_scoped():
    graph = {"inputs": ["x"], "outputs": ["y"], "values": {
        "x": _value([64], "float32"), "y": _value([64], "float32")},
        "operations": [{"name": "y", "target": "aten.sigmoid.default", "args": [{"ref": "x"}]}]}
    strict = lower(graph, execution="compose", compose_selection="common")
    practical = lower(graph, execution="compose", compose_selection="common", precision_policy="practical")
    again = lower(graph, execution="compose", compose_selection="common")
    assert "__fdividef" not in strict.kernel_body
    assert "__fdividef" in practical.kernel_body
    assert strict.source == again.source
    assert practical.manifest["execution_plan"]["arithmetic"] == "practical"


def test_default_remains_legacy():
    graph = {"inputs": ["x"], "outputs": ["y"], "values": {
        "x": _value([64], "float32"), "y": _value([64], "float32")},
        "operations": [{"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, 2]}]}
    first = lower(graph)
    lower(graph, execution="compose", compose_selection="common")
    assert lower(graph).source == first.source
    assert "execution_plan" not in first.manifest


@pytest.mark.parametrize("extent,selected", [(8192, False), (16384, True), (131072, True)])
def test_flat_scalar_reduction_retains_full_grid_body_and_private_counter(extent, selected):
    graph = {"inputs": ["x"], "outputs": ["mean"], "values": {
        "x": _value([extent], "float32"), "mean": _value([], "float32")},
        "operations": [{"name": "mean", "target": "aten.mean.default", "args": [{"ref": "x"}]}]}
    program = lower(graph, execution="compose", compose_selection="common")
    manifest = program.manifest
    reduction = next(p for p in manifest["phases"] if "mean" in p["writes"])
    assert (reduction.get("engine") == "multi_cta_full") == selected
    if selected:
        assert manifest["max_grid_blocks"] == 1024
        assert "if (*max_blocks > 1024) *max_blocks = 1024;" in program.source
        plan = manifest["execution_plan"]
        body = next(b for b in plan["bodies"] if b["body"] == "multi_cta_full")
        scratch = body["private_scratch"][0]["name"]
        initializer = next(b for b in plan["bodies"] if any(w["name"] == scratch for w in b["writes"]))
        assert initializer["index"] in body["dependencies"]
        assert [m["root"] for m in body["internal_mutations"]] == [scratch]
        assert body["counter_effects"][0]["storage"] == scratch
        assert scratch in manifest["offsets"]
        assert tuple(manifest["layouts"][scratch]["shape"]) == (1088,)
        lifetime = plan["scratch"][scratch]["lifetime"]
        stages = {p: stage for stage, members in enumerate(plan["stages"]) for p in members}
        assert lifetime[1] <= stages[initializer["index"]] < stages[body["index"]] <= lifetime[2]
        assert "fixed-order combine" in body["precision"]
    reference = lower(graph, execution="compose", compose_selection="common", exact_reductions=False).manifest
    assert all(p.get("engine") != "multi_cta_full" for p in reference["phases"])


@pytest.mark.parametrize("m,n,k,warpgroup", [(1024, 512, 512, False), (512, 512, 1024, False),
                                            (2048, 1024, 1024, True), (1536, 1152, 1152, True)])
def test_flat_matrix_admission_preserves_occupied_tiles_and_existing_ws_body(m, n, k, warpgroup):
    from megakernel_lab.operation_compiler import p5c_matrix
    before = {key: getattr(p5c_matrix, key) for key in p5c_matrix.FLAT_SELECTED_DEFAULTS}
    graph = {"inputs": ["x", "w"], "outputs": ["y"], "values": {
        "x": _value([m, k], "bfloat16"), "w": _value([k, n], "bfloat16"),
        "y": _value([m, n], "bfloat16")},
        "operations": [{"name": "y", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]}]}
    phase = next(p for p in lower(graph, execution="compose", compose_selection="common").manifest["phases"] if "y" in p["writes"])
    assert (phase.get("matrix_spec", {}).get("engine") == "p5c") == warpgroup
    if warpgroup:
        assert phase["matrix_plan"]["ws_engine"] == "p8ws_flat"
    assert before == {key: getattr(p5c_matrix, key) for key in before}


@pytest.mark.parametrize("m,n,k", [(128, 256, 128), (2048, 1024, 1024)])
def test_flat_expensive_activation_keeps_full_grid_elementwise_body(m, n, k):
    graph = {"inputs": ["x", "w"], "outputs": ["y"], "values": {
        "x": _value([m, k], "bfloat16"), "w": _value([k, n], "bfloat16"),
        "h": _value([m, n], "bfloat16"), "f": _value([m, n], "float32"),
        "y": _value([m, n], "float32")}, "operations": [
            {"name": "h", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]},
            {"name": "f", "target": "aten._to_copy.default", "args": [{"ref": "h"}],
             "kwargs": {"dtype": "torch.float32"}},
            {"name": "y", "target": "aten.silu.default", "args": [{"ref": "f"}]}]}
    phases = lower(graph, execution="compose", compose_selection="common").manifest["phases"]
    assert next(p for p in phases if "y" in p["writes"])["kind"] == "elementwise"
    matrix = next(p for p in phases if p["kind"] == "gemm")
    if matrix.get("matrix_spec", {}).get("engine") == "p5c":
        assert matrix["matrix_plan"]["ws_engine"] == "p8ws_flat"


def test_flat_last_use_state_commit_aliases_result_and_orders_old_read():
    graph = {"inputs": ["p", "g"], "outputs": ["loss", "updated"],
             "values": {"p": _value([64], "float32"), "g": _value([64], "float32"),
                        "loss": _value([], "float32"), "updated": _value([64], "float32")},
             "state_writes": {"p": "updated"}, "operations": [
                 {"name": "loss", "target": "aten.sum.default", "args": [{"ref": "p"}]},
                 {"name": "updated", "target": "aten.sub.Tensor", "args": [{"ref": "p"}, {"ref": "g"}]}]}
    program = lower(graph, execution="compose", compose_selection="common")
    manifest = program.manifest
    assert manifest["in_place_commits"] == {"p": "updated"}
    assert not manifest["copied_commits"]
    assert manifest["layouts"]["updated"]["root"] == "p"
    assert "updated" not in manifest["offsets"]
    bodies = manifest["execution_plan"]["bodies"]
    loss = next(b for b in bodies if any(w["name"] == "loss" for w in b["writes"]))
    update = next(b for b in bodies if any(w["name"] == "updated" for w in b["writes"]))
    assert loss["index"] in update["dependencies"]
    retained = lower(graph, execution="compose", compose_selection="common", retain_all=True).manifest
    assert not retained["in_place_commits"] and retained["layouts"]["updated"]["root"] == "updated"


@pytest.mark.parametrize("extent", [64, 4096])
@pytest.mark.parametrize("fuse", [False, True])
def test_state_result_identity_view_reads_new_value_before_scalar_or_vector_store(extent, fuse):
    # The fused vector body loads memory before computing/storing any result.
    # Its view of updated must therefore resolve to the producer's register,
    # not a prefetched read of either the old state or uninitialized arena.
    graph = {"inputs": ["p"], "outputs": ["updated", "q"],
             "values": {name: _value([extent], "float32") for name in ("p", "updated", "view", "q")},
             "state_writes": {"p": "updated"}, "operations": [
                 {"name": "updated", "target": "aten.add.Tensor", "args": [{"ref": "p"}, 1]},
                 {"name": "view", "target": "aten.view.default", "args": [{"ref": "updated"}, [extent]]},
                 {"name": "q", "target": "aten.add.Tensor", "args": [{"ref": "view"}, 2]}]}
    program = lower(graph, execution="compose", compose_selection="common", fuse_elementwise=fuse)
    manifest = program.manifest
    assert manifest["in_place_commits"] == {"p": "updated"}
    assert manifest["layouts"]["view"]["root"] == "p"
    bodies = manifest["execution_plan"]["bodies"]
    update = next(b for b in bodies if any(w["name"] == "updated" for w in b["writes"]))
    consumer = next(b for b in bodies if any(w["name"] == "q" for w in b["writes"]))
    if fuse:
        assert update["index"] == consumer["index"]
        assert "v_view[" not in program.kernel_body
        if extent >= 4096:
            assert "const float4" in program.kernel_body
            assert "r_q_0 = (r_updated_0 +" in program.kernel_body
        else:
            assert "r_q = (r_updated +" in program.kernel_body
    else:
        assert update["index"] in consumer["dependencies"]


@pytest.mark.parametrize("late_reader", ["same_phase", "later_phase", "strided_alias", "bound_matrix_root"])
def test_state_alias_refuses_old_reads_after_or_across_update(late_reader):
    from megakernel_lab.operation_compiler.state_placement import last_use_state_aliases
    from megakernel_lab.operation_compiler.lower import TensorLayout
    layouts = {name: TensorLayout((64,), (1,), name) for name in ("p", "g", "updated", "other")}
    layouts["old_view"] = TensorLayout((64,), (2,), "p")
    phases = [{"kind": "elementwise", "shape": (64,), "names": ["updated"],
               "reads": ["p", "g"], "writes": ["updated"],
               "body": ["const float r_updated = v_p[i] - v_g[i]; v_updated[i] = r_updated;"]}]
    if late_reader == "same_phase":
        phases[0]["reads"].append("p")
        phases[0]["body"].append("const float r_other = v_p[i]; v_other[i] = r_other;")
    elif late_reader == "later_phase":
        phases.append({"kind": "reduction", "reads": ["p"], "writes": ["other"]})
    elif late_reader == "strided_alias":
        phases[0]["reads"] = ["old_view", "g"]
        phases[0]["body"] = ["const float r_updated = v_old_view[2*i] - v_g[i]; v_updated[i] = r_updated;"]
    else:
        phases.append({"kind": "gemm", "reads": ["updated"], "writes": ["other"],
                       "matrix_spec": {"roots": [("updated", 0)]}})
    assert not last_use_state_aliases(phases, layouts, {name: "float32" for name in layouts},
                                     {"p": "updated"}, ["p", "g"], ident=lambda name: "v_" + name)


@pytest.mark.parametrize("retain_intermediate", [False, True])
def test_runtime_layouts_exclude_elided_roots_and_aliases_but_preserve_outputs(retain_intermediate):
    graph = {"inputs": ["x"], "outputs": ["a", "b"] if retain_intermediate else ["b"],
             "values": {name: _value([64], "float32") for name in ("x", "a", "av", "b")},
             "operations": [
                 {"name": "a", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, 2]},
                 {"name": "av", "target": "aten.view.default", "args": [{"ref": "a"}, [64]]},
                 {"name": "b", "target": "aten.add.Tensor", "args": [{"ref": "a"}, 1]}]}
    manifest = lower(graph, execution="compose", compose_selection="common").manifest
    for name, layout in manifest["layouts"].items():
        assert layout["root"] in set(manifest["inputs"]) | set(manifest["offsets"]), name
    assert set(manifest["outputs"]) <= set(manifest["layouts"])
    if retain_intermediate:
        assert "a" in manifest["layouts"] and "av" in manifest["layouts"]
        assert "a" in manifest["offsets"]
    else:
        assert "a" not in manifest["layouts"] and "av" not in manifest["layouts"]
        assert {"a", "av"} <= set(manifest["elided_layouts"])
        assert "a" not in manifest["offsets"]


def test_captured_module_runtime_values_all_have_bound_storage():
    import torch
    from megakernel_lab.operation_compiler.api import capture_module

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((64, 64), 0.02))

        def forward(self, x):
            hidden = (x.to(torch.bfloat16) @ self.weight.to(torch.bfloat16)).float()
            return hidden + 0.25 * torch.nn.functional.silu(hidden)

    captured = capture_module(Model(), lambda prediction, target: (prediction - target).square().mean(),
                              (torch.full((64, 64), 0.1),), (torch.zeros((64, 64)),))
    manifest = lower(captured.ir, execution="compose", compose_selection="common").manifest
    assert manifest["elided_layouts"]
    assert set(manifest["outputs"]) <= set(manifest["layouts"])
    assert set(manifest["state_writes"].values()) <= set(manifest["layouts"])
    for name, layout in manifest["layouts"].items():
        assert layout["root"] in set(manifest["inputs"]) | set(manifest["offsets"]), name
