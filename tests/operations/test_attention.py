"""Structural attention/ordered-index contracts; execute on the admitted CPU worker.

These inspect generated operations and dependency storage, not CUDA arithmetic.
The integration GPU qualification separately checks loss, all gradients and state.
"""
from __future__ import annotations

import traceback
import math

from megakernel_lab.operation_compiler import attention_lower as attention
from megakernel_lab.operation_compiler.lower import TensorLayout, _contiguous, _field, _ident, _ref


def fixture(*, width=64, length=96, dtype="float32", delayed_dout=False):
    values, operations, layouts = {}, [], {}

    def value(name, shape, *, root=None, strides=None):
        values[name] = {"name": name, "shape": list(shape), "dtype": dtype}
        layouts[name] = TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), root or name)

    def op(name, target, args, shape, *, root=None, strides=None, kwargs=None):
        value(name, shape, root=root, strides=strides)
        operations.append({"output": name, "target": "aten." + target,
                           "args": args, "kwargs": kwargs or {}})

    head, score = (2, length, width), (2, length, length)
    for name in ("q", "k", "v", "dout"):
        value(name, head)
    value("mask", (length, length))
    op("kt", "transpose.int", ["k", 1, 2], (2, width, length),
       root="k", strides=(length * width, 1, width))
    op("score", "bmm.default", ["q", "kt"], score)
    op("scaled", "mul.Tensor", ["score", .125], score)
    op("masked", "add.Tensor", ["scaled", "mask"], score)
    op("prob", "_softmax.default", ["masked", -1, False], score)
    op("out", "bmm.default", ["prob", "v"], head)
    op("pt", "transpose.int", ["prob", 1, 2], score,
       root="prob", strides=(length * length, 1, length))
    dout = "dout"
    if delayed_dout:
        op("computed_dout", "mul.Tensor", ["dout", 1.1], head)
        dout = "computed_dout"
    op("dv", "bmm.default", ["pt", dout], head)
    op("vt", "transpose.int", ["v", 1, 2], (2, width, length),
       root="v", strides=(length * width, 1, width))
    op("dp", "bmm.default", [dout, "vt"], score)
    op("ds", "_softmax_backward_data.default", ["dp", "prob", -1, dtype], score)
    op("dscaled", "mul.Tensor", ["ds", .125], score)
    op("qt", "transpose.int", ["q", 1, 2], (2, width, length),
       root="q", strides=(length * width, 1, width))
    op("dkt", "bmm.default", ["qt", "dscaled"], (2, width, length))
    op("kv", "transpose.int", ["kt", 1, 2], head,
       root="k", strides=(length * width, width, 1))
    op("dq", "bmm.default", ["dscaled", "kv"], head)
    return operations, values, layouts


def match(operations, values, protected=()):
    return attention.plan(operations, values, set(values), ref=_ref, field=_field,
                          output_name=lambda operation: operation["output"], protected=set(protected))


def run():
    results = {}

    def check(name, fn):
        try:
            fn()
            results[name] = {"passed": True}
        except Exception:
            results[name] = {"passed": False, "traceback": traceback.format_exc()}

    def selection():
        operations, values, layouts = fixture(delayed_dout=True)
        groups, refusals = match(operations, values, {"out", "dq", "dv"})
        assert len(groups) == 1 and not refusals
        group = groups[0]
        assert group["anchor_backward"] == "dv", "must wait until dO exists"
        assert group["last_backward"] == "dq"
        assert {"prob", "score", "masked", "dp", "ds"}.issubset(group["removed"])
        attention.validate_views(group, layouts)
        dtypes = {name: "float32" for name in values}
        emitted = [attention.emit(group, layouts, dtypes, _ident, which=which)
                   for which in ("prepare", "forward", "delta", "backward")]
        assert max(item[2].get("shared_bytes", 0) for item in emitted) == 131840
        assert all(item[2]["work_items"] > 0 for item in emitted)
        assert "pattn::backward_kv_item<64>" in emitted[-1][1]

    check("complete_backward_and_safe_anchor", selection)

    def escaped():
        operations, values, _ = fixture()
        for protected in ({"prob"}, {"dkt"}):
            groups, refusals = match(operations, values, protected)
            assert not groups and refusals
        operations.append({"output": "mask_grad", "target": "aten.sum.dim_IntList",
                           "args": ["ds", [0], False], "kwargs": {}})
        values["mask_grad"] = {"shape": [96, 96], "dtype": "float32"}
        groups, refusals = match(operations, values)
        assert not groups and refusals, "trainable mask gradient must remain ordinary operations"

    check("escaped_probability_and_trainable_mask_fallback", escaped)

    def restrictions():
        for width, dtype, misaligned in ((48, "float32", False), (64, "bfloat16", False),
                                         (64, "float32", True)):
            operations, values, layouts = fixture(width=width)
            groups, _ = match(operations, values)
            if misaligned:
                layout = layouts["q"]
                layouts["q"] = TensorLayout(layout.shape, layout.strides, "q", 1)
            try:
                attention.emit(groups[0], layouts, {name: dtype for name in values}, _ident, which="forward")
            except attention.AttentionRefused:
                pass
            else:
                raise AssertionError((width, dtype, misaligned))

    check("dtype_head_width_and_alignment_fallback", restrictions)

    def collisions_and_scale():
        operations, values, _ = fixture()
        values["out__lse"] = {"shape": [2, 96], "dtype": "float32"}
        groups, refusals = match(operations, values)
        assert not groups and "synthetic" in refusals[0]["reason"]
        del values["out__lse"]
        for operation in operations:
            if operation["output"] in {"scaled", "dscaled"}:
                operation["target"] = "aten.div.Tensor"
            if operation["output"] == "dscaled":
                operation["kwargs"] = {"rounding_mode": "floor"}
        groups, refusals = match(operations, values)
        assert not groups and refusals

    check("name_collision_and_backward_scale_fallback", collisions_and_scale)

    def packed_phases():
        operations, values, layouts = fixture(delayed_dout=True)
        groups, _ = match(operations, values)
        group = groups[0]
        # Pretend lowering has seen all ordinary aliases; fused intermediates
        # are intentionally absent, just as in the real operation walk.
        layouts = {name: layout for name, layout in layouts.items() if name not in group["skip"]}
        dtypes = {name: "float32" for name in values}
        nodes, views, decisions = [], [], []

        def define(name, shape, root=None, strides=None):
            layouts[name] = TensorLayout(tuple(shape), tuple(strides or _contiguous(shape)), root or name)
            dtypes[name] = "float32"

        def add_view(name, target, source):
            views.append({"name": name, "target": target, "source": source})

        def add_heavy(kind, op, name, code, refs, **extra):
            nodes.append({"kind": kind, "name": name, "code": code, "reads": refs, **extra})

        def add_phase(kind, code, metadata, reads, writes):
            assert all(name in layouts for name in reads), (kind, reads)
            attention.add_fused_nodes(kind, code, metadata, reads, writes,
                layouts=layouts, dtypes=dtypes, make_layout=TensorLayout,
                add_heavy=add_heavy, add_view=add_view, reserved_names=set(values))

        for operation in operations:
            name = operation["output"]
            if name in group["skip"]:
                attention.emit_member(group, name, layouts, dtypes, values,
                    field=_field, ident=_ident, define=define, add_phase=add_phase,
                    add_view=add_view, record=decisions.append)
        assert len(nodes) == 4 and len(decisions) == 1
        for node in nodes:
            root = node["name"]
            segments = sorted((segment["offset"], segment["offset"] +
                               math.prod(segment["shape"]))
                              for segment in node["packed_outputs"].values())
            assert all(start % 4 == 0 for start, _ in segments)
            assert all(left[1] <= right[0] for left, right in zip(segments, segments[1:]))
            assert layouts[root].numel == segments[-1][1]
            assert all(layouts[name].root == root for name in node["phase_writes"])
        assert layouts["dkt"].root == layouts["dq"].root == layouts["dv"].root
        assert layouts["dkt"].offset == layouts["dkt__dk"].offset
        assert layouts["dkt"].strides == (96 * 64, 1, 64)
        assert "prob" not in layouts and "score" not in layouts

    check("packed_atomic_phase_aliases_and_lifetimes", packed_phases)

    def integrated_codegen_and_fallback():
        from megakernel_lab.operation_compiler.lower import _lower_serial
        from megakernel_lab.operation_compiler.fusion import lower_fused
        operations, values, _ = fixture(delayed_dout=True)
        graph = {"operations": operations, "values": values,
                 "inputs": ["q", "k", "v", "dout", "mask"],
                 "outputs": ["out", "dq", "dv"], "state_writes": {}}
        for lower in (_lower_serial, lower_fused):
            program = lower(graph, **({"attention_min_keys": 0} if lower is lower_fused else {}))
            assert len(program.manifest["attention_groups"]) == 1
            assert "attention_fp32.cuh" in program.manifest["header_hashes"]
            assert program.manifest["shared_bytes"] == 131840
            assert "prob" not in program.manifest["layouts"]
            assert "cudaFuncSetAttribute" in program.source
            # Exporting P prevents its elimination and proves ordinary fallback.
            fallback = lower(dict(graph, outputs=["out", "prob", "dq", "dv"]),
                             **({"attention_min_keys": 0} if lower is lower_fused else {}))
            assert not fallback.manifest["attention_groups"]
            assert fallback.manifest["attention_refusals"]
            assert "prob" in fallback.manifest["layouts"]

    check("both_schedulers_codegen_and_exported_probability_fallback", integrated_codegen_and_fallback)

    def downstream_view_restores_original_dkt_layout():
        from megakernel_lab.operation_compiler.lower import _lower_serial
        from megakernel_lab.operation_compiler.fusion import lower_fused
        operations, values, _ = fixture()
        for name, target, args, shape in (
                ("dkt_split", "view.default", ["dkt", [1, 2, 64, 96]], [1, 2, 64, 96]),
                ("dkt_permuted", "permute.default", ["dkt_split", [0, 3, 1, 2]], [1, 96, 2, 64]),
                ("merged", "view.default", ["dkt_permuted", [1, 96, 128]], [1, 96, 128])):
            values[name] = {"shape": shape, "dtype": "float32"}
            operations.append({"output": name, "target": "aten." + target, "args": args, "kwargs": {}})
        groups, refusals = match(operations, values)
        assert len(groups) == 1 and not refusals
        assert groups[0]["restore_dkt_reason"].startswith("merged:")
        graph = {"operations": operations, "values": values,
                 "inputs": ["q", "k", "v", "dout", "mask"],
                 "outputs": ["out", "dq", "dv", "merged"], "state_writes": {}}
        reference = _lower_serial(graph, fuse_attention=False, ordered_index_gradients=False,
                                  matrix_policy="frozen", split_k=False, exact_reductions=False)
        original_strides = reference.manifest["layouts"]["merged"]["strides"]
        # A singleton axis has no address effect; _reshape_strides chooses its
        # canonical stride (96 here), rather than retaining the old batch span.
        # Compare the complete emitted layout to the original operation graph.
        assert original_strides[1:] == [1, 96]
        for lower in (_lower_serial, lower_fused):
            program = lower(graph, **({"attention_min_keys": 0} if lower is lower_fused else {}))
            assert len(program.manifest["attention_groups"]) == 1
            assert program.manifest["attention_groups"][0]["layout_restore"].startswith("merged:")
            assert program.manifest["layouts"]["dkt"]["strides"] == [64 * 96, 96, 1]
            assert program.manifest["layouts"]["merged"]["strides"] == original_strides
            assert any(phase["kind"] == "attention_layout_restore" for phase in program.manifest["phases"])
            assert "v_dkt[i] = v_dkt__dk[" in program.source
        # The copied index maps the same (batch, key, channel) value to the
        # original bmm's layout, with no change of FP32 arithmetic.
        for batch, channel, key in ((0, 0, 0), (0, 63, 95), (1, 31, 17), (1, 63, 95)):
            index = (batch * 64 + channel) * 96 + key
            source = (index // (64 * 96)) * (64 * 96) + (index % 96) * 64 + (index // 96) % 64
            assert source == (batch * 96 + key) * 64 + channel

    check("downstream_view_restores_original_dkt_layout", downstream_view_restores_original_dkt_layout)

    def exact_integer_reduction_in_fused_schedule():
        from megakernel_lab.operation_compiler.fusion import lower_fused
        for dtype in ("bool", "int64"):
            graph = {"inputs": ["x"], "outputs": ["total"],
                "values": {"x": {"shape": [2, 17], "dtype": dtype},
                           "total": {"shape": [2], "dtype": "int64"}},
                "operations": [{"name": "total", "target": "aten.sum.dim_IntList",
                                "args": ["x", [1], False], "kwargs": {"dtype": "int64"}}]}
            program = lower_fused(graph)
            assert "unsigned long long acc = 0" in program.source
            assert "static_cast<int64_t>(acc)" in program.source
            assert program.manifest["phases"][0]["arithmetic"].startswith("integer modulo2**64")
            assert "float acc =" not in program.source

    check("stage4_boolean_and_int64_reduction_preserves_exact_bits", exact_integer_reduction_in_fused_schedule)

    def ordered_indices():
        from megakernel_lab.operation_compiler.transformer_lower import ordered_embedding_backward
        names = {"grad", "ids", "result"}
        layouts = {"grad": TensorLayout((2, 7, 64), (7 * 64, 64, 1), "grad"),
                   "ids": TensorLayout((2, 7), (7, 1), "ids")}
        dtypes = {"grad": "float32", "ids": "int64", "result": "float32"}
        emitted = ordered_embedding_backward(["grad", "ids", 101, -1, False], {},
                                              "result", (101, 64), layouts, dtypes, names)
        assert emitted["threads"] == 256 and len(emitted["phases"]) == 2
        assert len(emitted["auxiliary"]) == 3
        assert "acc += g[u]" in emitted["phases"][1][2]
        assert "atomic" not in emitted["phases"][1][2]
        names.add("result__occ")
        assert ordered_embedding_backward(["grad", "ids", 101, -1, False], {},
                                           "result", (101, 64), layouts, dtypes, names) is None

    check("ordered_embedding_fixed_order_and_collision_fallback", ordered_indices)

    def captured_adamw_dropout_conditional():
        import torch
        from megakernel_lab.operation_compiler.capture import capture_conditional_training
        from megakernel_lab.operation_compiler.optim import AdamWConfig, initialize_optimizer_state
        from megakernel_lab.operation_compiler.random import dropout, make_rng_state
        torch.manual_seed(714)
        params = {name: torch.randn(2, 16, 32) * .1 for name in ("q", "k", "v")}
        optimizer = AdamWConfig(eps=1e-4)
        state = {"rng": make_rng_state(731)}
        target = torch.randn(2, 16, 32) * .1

        def objective(scale):
            def loss(parameters, state, target):
                score = torch.bmm(parameters["q"], parameters["k"].transpose(1, 2)) * scale
                output = torch.bmm(torch.softmax(score, -1), parameters["v"])
                output, rng = dropout(output, state["rng"], .125)
                return (output - target).square().mean(), {"rng": rng}
            return loss

        captured = capture_conditional_training(objective(.125), objective(.25), params, None,
            (target,), torch.tensor(True), optimizer, state=state,
            optimizer_state=initialize_optimizer_state(params, optimizer))
        for graph in (captured.ir.true_branch, captured.ir.false_branch):
            groups, refusals = attention.plan(graph.operations, graph.values, set(graph.values),
                ref=_ref, field=_field, output_name=lambda operation: operation.outputs[0],
                protected=set(graph.output_names()))
            assert len(groups) == 1 and not refusals, refusals
            assert groups[0]["backward"], "must match automatically derived full backward"
            assert any(operation.target.startswith("latch_ops.philox_") for operation in graph.operations)
            assert graph.metadata["optimizer_state_names"], "optimizer ABI must remain separate"
            assert not set(groups[0]["skip"]).intersection(graph.outputs["optimizer_state"].values())
            assert not set(groups[0]["skip"]).intersection(graph.outputs["state"].values())

    check("pytorch_adamw_and_post_attention_dropout_in_both_regions", captured_adamw_dropout_conditional)
    return {"passed": all(result["passed"] for result in results.values()), "tests": results,
            "scope": "CPU structural/code-generation contracts; CUDA numerics qualified separately"}
