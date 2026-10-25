"""Automatic integration policy contracts, executed on the admitted CPU worker.

GPU qualification separately checks the emitted arithmetic and committed state.
These fixtures exercise the ordinary entry point, not model-specific overrides.
"""
from __future__ import annotations

import traceback

from megakernel_lab.operation_compiler import attention_lower
from megakernel_lab.operation_compiler.lower import lower
from .test_attention import fixture


def graph(length, *, forward=False, bf16_input=False, width=64):
    operations, values, _ = fixture(length=length, width=width)
    inputs = ["q", "k", "v", "dout", "mask"]
    if forward:
        operations = operations[:next(i for i, op in enumerate(operations) if op["output"] == "out") + 1]
        inputs.remove("dout")
        active = set(inputs) | {op["output"] for op in operations}
        values = {name: value for name, value in values.items() if name in active}
    if bf16_input:
        values["q_bf16"] = dict(values["q"], name="q_bf16", dtype="bfloat16")
        operations.insert(0, {"output": "q", "target": "aten._to_copy.default",
                             "args": ["q_bf16"], "kwargs": {"dtype": "float32"}})
        inputs[0] = "q_bf16"
    return {"operations": operations, "values": values, "inputs": inputs,
            "outputs": ["out"] if forward else ["out", "dq", "dv"], "state_writes": {}}


def run():
    checks = {}

    def check(name, fn):
        try:
            fn()
            checks[name] = {"passed": True}
        except Exception:
            checks[name] = {"passed": False, "traceback": traceback.format_exc()}

    def boundaries():
        for length, expected in ((96, None), (512, None), (768, "v1"), (1024, "v2")):
            program = lower(graph(length))
            groups = program.manifest["attention_groups"]
            assert [x["variant"] for x in groups] == ([] if expected is None else [expected])
            assert program.manifest["compiler_policy"] == "integrated_v3"
            if expected == "v2":
                assert "attention_fp32_v2.cuh" in program.manifest["header_hashes"]
                assert groups[0]["source_commit"] == attention_lower.SOURCE_COMMIT_V2
            if expected is None:
                assert "prob" in program.manifest["layouts"]
                assert any("product bodies" in x["reason"] for x in program.manifest["attention_refusals"])

    check("automatic_fp32_context_boundaries", boundaries)

    def precision():
        program = lower(graph(512, bf16_input=True))
        assert program.manifest["attention_selection_policy"]["has_bfloat16"]
        assert [g["variant"] for g in program.manifest["attention_groups"]] == ["v1"]
        assert "prob" not in program.manifest["layouts"]
        override = lower(graph(512, bf16_input=True), attention_min_keys=512)
        assert not override.manifest["attention_groups"]
        assert override.manifest["attention_selection_policy"]["explicit_override"]

    check("mixed_precision_preserves_qualified_bf16_attention", precision)

    def forward_memory():
        training = lower(graph(1024))
        inference = lower(graph(1024, forward=True))
        assert [g["variant"] for g in inference.manifest["attention_groups"]] == ["v2"]
        assert any("__xs" in key for key in training.manifest["layouts"])
        assert not any("__xs" in key or "__ds" in key for key in inference.manifest["layouts"])
        assert inference.manifest["arena_words"] < training.manifest["arena_words"]
        assert not inference.manifest["state_writes"]

    # Inference qualification is deferred by the training-only v0 scope update.

    def fallback():
        for options in ({"policy": "reference"}, {"target_arch": "sm_80"}):
            program = lower(graph(1024), **options)
            assert not program.manifest["attention_groups"]
            assert "prob" in program.manifest["layouts"]
            assert program.manifest["compiler_selection"]["reasons"]
        assert attention_lower.select_variant(32, 2048, 2) == "v1"
        assert attention_lower.select_variant(64, 512, 2) == "v1"
        assert attention_lower.select_variant(64, 512, 2, "force") == "v2"

    check("reference_architecture_and_head_width_fallback", fallback)
    return {"kind": "integration2-policy-contracts", "checks": checks,
            "passed": all(value["passed"] for value in checks.values())}
