"""ops.attention_rules (LATCH-TINYMOE-ATTN): fake capture, tape and source checks on a CPU worker.

Two grouped-query width-48 causal attention calls (the TinyMoE spelling: real-width operands, BF16
explicit SDPA with GQA expansion and zero padding inside) share one captured causal constant.
"""
import pytest
import torch

from megakernel_lab.operation_compiler import attention_rules, bf16_sdpa
from megakernel_lab.operation_compiler.compose_attention import rewrite_tape


def two_layer_fixture(*, sequence=256):
    from megakernel_lab.operation_compiler.capture import capture_training
    from megakernel_lab.operation_compiler.user_optimizer import tensor_update
    shapes = {"q0": 8, "k0": 4, "v0": 4, "q1": 8, "k1": 4, "v1": 4}
    parameters = {name: torch.zeros(1, heads, sequence, 48) for name, heads in shapes.items()}
    cotangent = torch.zeros(1, 8, sequence, 48, dtype=torch.bfloat16)

    positions = torch.arange(sequence, dtype=torch.float32)[:, None] * torch.linspace(0.0, 1.0, 24)[None, :]
    angles = torch.cat((positions, positions), dim=-1)
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]

    def rotate_half(x):
        return torch.cat((-x[..., 24:], x[..., :24]), dim=-1)

    def objective(params, cotangent):
        total = 0.0
        for layer in (0, 1):
            q, k, v = (params[f"{name}{layer}"].to(torch.bfloat16) for name in ("q", "k", "v"))
            if layer == 0:  # the TinyMoE spelling: BF16 projections times FP32 rotary tables, then the BF16 cast
                q = (q * cos + rotate_half(q) * sin).to(torch.bfloat16)
                k = (k * cos + rotate_half(k) * sin).to(torch.bfloat16)
            y = bf16_sdpa.sdpa(q, k, v, is_causal=True, enable_gqa=True, scale=48 ** -0.5, sliding_window=1024)
            total = total + (y.float() * cotangent.float()).sum()
        return total

    rule = tensor_update(lambda p, g, s, hp: ({name: value - 0.001 * g[name] for name, value in p.items()}, {}),
                         name="attention_rules_probe")
    with bf16_sdpa.capture_bf16_sdpa():
        return capture_training(objective, parameters, None, inputs=(cotangent,), config=rule)


def _producer_targets(ir):
    return {name: op.target for op in ir.operations for name in op.outputs}


def test_validate_and_default_unchanged():
    with pytest.raises(ValueError):
        attention_rules.validate("fast")
    captured = two_layer_fixture()
    _, default = rewrite_tape(captured.ir, version=7)
    assert "attention_rules" not in default
    targets = _producer_targets(captured.ir)
    for record in default["groups"]:
        plan = record["plan"]
        assert all(targets[plan["sources"][key]] == "aten.clone.default" for key in ("q", "k", "v", "do"))
        assert "mask_tables" not in plan and not plan.get("tape_keynorm_item")
        assert not plan.get("tape_merged_backward")
        assert any(name.endswith("__long_kn") for name in record["forward"]["outputs"])
        assert any(name.endswith("__long_mmax") for name in record["forward"]["outputs"])


def test_streamed_tape():
    captured = two_layer_fixture()
    targets = _producer_targets(captured.ir)
    with attention_rules.scope("streamed"):
        rewritten, report = rewrite_tape(captured.ir, version=7)
    assert report["attention_rules"] == "streamed"
    first, second = report["groups"]
    for record in (first, second):
        plan = record["plan"]
        # Clones removed: operands are the zero-pad concatenations, the cotangent its slice_backward pad.
        assert {targets[plan["sources"][key]] for key in ("q", "k", "v")} == {"aten.cat.default"}
        assert targets[plan["sources"]["do"]] == "aten.slice_backward.default"
        assert len(plan["elided_clones"]) == 4
        assert plan["tape_merged_backward"] and not plan.get("tape_keynorm_item")
        pack = plan["tape_pack"]
        roped = record is first
        assert {key: (pack[key]["fp32"], pack[key]["group"], pack[key]["width"], bool(pack[key]["rope"]))
                for key in ("q", "k", "v")} == {
            "q": (not roped, 1, 48, roped), "k": (not roped, 2, 48, roped), "v": (True, 2, 48, False)}
        if roped:
            assert pack["q"]["rope"]["half"] == 24 and pack["q"]["rope"]["cos"] != pack["q"]["rope"]["sin"]
            assert targets[pack["q"]["src"]] == "aten._to_copy.default"  # the BF16 projection, read by the pack
        outputs = record["forward"]["outputs"]
        assert outputs[:3] == [plan["sources"][key] for key in ("q", "k", "v")]
        assert any(name.endswith("__long_kn") for name in outputs)
        assert set(record["forward"]["inputs"][:3]) == {pack[key]["src"] for key in ("q", "k", "v")}
        assert any(name.endswith("__long_delta") for name in record["backward"]["outputs"])
    remaining = {op.target for op in rewritten.operations}
    # Every padding concatenation, zero fill, repetition copy, rotary product and contiguity clone of the
    # operands is gone (the backward of the rotation keeps its own ops).
    assert not remaining & {"aten.zeros.default", "aten.unsqueeze.default"}
    assert not [op for op in rewritten.operations if op.target == "aten.cat.default"
                and op.outputs[0] in set(first["plan"]["removed"])]
    from megakernel_lab.operation_compiler.compose_attention import TAPE_TARGET
    produced = {name for op in rewritten.operations if op.target != TAPE_TARGET for name in op.outputs}
    assert not produced & (set(first["plan"]["removed"]) | set(second["plan"]["removed"]))
    assert bf16_sdpa.FORWARD not in remaining
    assert not [op for op in rewritten.operations if op.target == "aten.clone.default"
                and set(op.outputs) & {first["plan"]["sources"][k] for k in ("q", "k", "v")}]
    # The first group owns the constant mask's tables; the second reads them and has no prepare outputs.
    assert "mask_tables" not in first["plan"]
    owner = second["plan"]["mask_tables"]
    assert set(owner.values()) <= set(first["forward"]["outputs"])
    assert set(owner.values()) <= set(second["forward"]["inputs"]) and set(owner.values()) <= set(second["backward"]["inputs"])
    assert not any(name.endswith(("__long_mmax", "__long_mzero", "__long_rm")) for name in second["forward"]["outputs"])


def test_streamed_source():
    from megakernel_lab.operation_compiler.lower import lower
    captured = two_layer_fixture()
    rewritten, _ = rewrite_tape(captured.ir, version=7)
    base = lower(rewritten, target_arch="sm_90", execution="compose", vocabulary_head="off").source
    with attention_rules.scope("streamed"):
        rewritten, _ = rewrite_tape(captured.ir, version=7)
        program = lower(rewritten, target_arch="sm_90", execution="compose", vocabulary_head="off")
    source = program.source
    assert base.count("pflatlong::prepare_unit(") == 2 and "forward_item6<1, false>" in base
    assert "pflatlong::prepare_unit(" not in source and source.count("pattnpack::mask_unit(A, u);") == 1
    assert source.count("pattnpack::pack_copy<8, 256, 8>(") == 2 and source.count("pattnpack::key_norm_unit<256, 8>(P, u);") == 2
    assert source.count("forward_item6<1, false>") == 2 and "forward_item6k" not in source
    assert source.count("pflatlong::delta_unit6(") == 2 and source.count("dq_item6n<1, false>") == 2
    assert source.count("kv_item6<1, false>") == 2 and "dq_item6<1, false>" not in source
    assert '#include "attention_pack.cuh"' in source or "attention_pack.cuh" in program.manifest.get("headers", [])
    groups = [row for row in program.manifest["attention_groups"] if row.get("tape_half") == "forward"]
    assert [row["keynorm"].startswith("operand pack") for row in groups] == [True, True]
    # The packed sources are read through their lowered layouts: FP32 parameters, query heads 8 / key heads 4.
    assert source.count("pattnpack::Src{") == 6
    assert source.count(", 24};") == 2  # layer 0's rotary Q and K sources carry their tables and half width


def test_kn_item_without_pack():
    """The kn_item rule alone (pack refused: shared padded operand) keeps forward_item6k."""
    from megakernel_lab.operation_compiler import flat_long_attention
    captured = two_layer_fixture()
    plan = next(iter(bf16_sdpa.plans(captured.ir).values()))
    _, code, extra = flat_long_attention.emit(plan, lambda n: n, None, which="forward", recipe="bf16", version=7,
                                              keynorm_in_item=True)
    assert "forward_item6k<1, false>" in code
    assert flat_long_attention.emit(plan, lambda n: n, None, which="prepare", recipe="bf16", version=7,
                                    keynorm_external=True) is None
