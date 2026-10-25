"""LATCH-B32-TRANSFER: the automatic rule selection for one-update TinyMoE programs (CPU, no CUDA)."""
import pytest

from megakernel_lab.operation_compiler import accumulation_rules as ar
from megakernel_lab.tinymoe import build


def test_update_preset_is_exact_subset():
    chosen = ar.rules_of("update")
    assert chosen == ar.UPDATE_PRESET_RULES
    assert set(chosen) <= set(ar.RULES)
    assert not {"castview", "cmulti", "accfuse"} & set(chosen)
    with ar.scope("update"):
        assert ar.features() == "".join("+" + r for r in ("fsgather", "fswarp", "cmulti", "rowchain", "fsmix", "fsfence")
                                        if r in chosen)


def test_resolve_rules():
    explicit = dict(update_schedule="streamed", segmented_products=None, norm_fusion=None, attention_rules=None,
                    accumulation_rules=None)
    assert build.resolve_rules(None, **explicit) == explicit
    auto = build.resolve_rules("auto", **dict(explicit, update_schedule=None))
    assert auto == build.UPDATE_RULES == build.b32_rules()
    assert build.resolve_rules("auto", **dict(explicit, update_schedule="streamed_prep")) == build.UPDATE_RULES
    with pytest.raises(ValueError):
        build.resolve_rules("auto", **explicit)  # explicit "streamed" differs from the selection
    with pytest.raises(ValueError):
        build.resolve_rules("all", **explicit)
