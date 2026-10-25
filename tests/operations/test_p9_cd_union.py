"""P9 B/C/D integration contracts; execute numerical cases on the CPU worker."""
from __future__ import annotations


def test_sequence_and_recurrent_presets_stay_independent():
    from megakernel_lab.operation_compiler.execution_policy import RECURRENT_RULES, SEQUENCE_RULES
    from megakernel_lab.operation_compiler.group_rules import features_of

    recurrent, sequence = map(features_of, (RECURRENT_RULES, SEQUENCE_RULES))
    assert {"colreg", "rhoist", "ilp8", "part"} <= recurrent
    assert "alap" not in recurrent
    assert {"alap", "vecbf", "splitk", "colsplit", "p5cv"} <= sequence
    assert not {"part", "colreg", "rhoist", "pelem", "pred", "rowepi"} & sequence


def test_precision_recipes_remain_explicit_and_distinct():
    from megakernel_lab.operation_compiler.precision import resolve

    assert resolve(None) is None
    assert resolve("float32") is None
    mixed = resolve("mixed_bf16")
    operands = resolve("bf16_attention_operands")
    fast = resolve("bf16_attention_operands_fast_activation")
    assert mixed.attention == "bfloat16" and mixed.activation == "exact"
    assert operands.attention == "bf16_operands" and operands.activation == "exact"
    assert fast.attention == "bf16_operands" and fast.activation == "fast"
    assert mixed.products and not operands.products and not fast.products


def test_shared_casts_preserve_values_and_dtype():
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx
    from megakernel_lab.operation_compiler.precision import share_float_casts

    # Two uses of the same BF16 buffer through a transpose: keep exact values
    # while eliminating the redundant preserve-format conversion after AD.
    def joint(x):
        ordinary = x.float()
        transposed = x.transpose(0, 1).float()
        duplicate = x.float()
        return ordinary + duplicate, transposed + ordinary.transpose(0, 1)

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16)
    graph = make_fx(joint)(x)
    before = graph(x)
    report = share_float_casts(graph)
    after = graph(x)
    assert report["commuted"] and report["merged"]
    assert all(a.dtype == torch.float32 and torch.equal(a, b) for a, b in zip(before, after))


def test_selected_panel_defaults_leave_rejected_modes_off():
    from megakernel_lab.operation_compiler.sampling_panels import PANEL_DEFAULTS
    from megakernel_lab.operation_compiler.sampling_lower import SAMPLED_OPTIONS
    from megakernel_lab.operation_compiler import fusion_reference

    assert SAMPLED_OPTIONS["panels"] == "auto"
    assert SAMPLED_OPTIONS["commit"] == "alias" and SAMPLED_OPTIONS["train_place"] == "auto"
    assert PANEL_DEFAULTS["spill"] == "auto" and PANEL_DEFAULTS["share_segments"] is True
    assert PANEL_DEFAULTS["mma"] == "auto" and PANEL_DEFAULTS["pack"] == "fused"
    assert PANEL_DEFAULTS["elementwise"] == "exact" and PANEL_DEFAULTS["wg_desc"] == "a"
    assert not PANEL_DEFAULTS["field_ahead"] and not PANEL_DEFAULTS["wg_inline"]
    assert fusion_reference.REDUCTION_STACK is True
    assert not fusion_reference.LOOP_PARTITION
    assert not fusion_reference.LOOP_SPLIT and not fusion_reference.SCALAR_HOIST
