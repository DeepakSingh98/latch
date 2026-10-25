"""Integration4: retain a safe flat body unless the convolution split-K path admits its replacement."""

import pytest

from megakernel_lab.operation_compiler import conv_ops, matrix_rules
from megakernel_lab.operation_compiler.lower import TensorLayout


def _candidate(m=256, n=2304, k=18432):
    taps = conv_ops.TapsOperands(enabled=True)
    taps.defer(conv_ops.IM2COL_TARGET, "taps", None, [], [], {}, {"taps": "bfloat16"})
    # Weight-gradient layout: transposed output gradient times the pixel-major im2col operand.
    la = TensorLayout((m, k), (1, m), "grad")
    lb = TensorLayout((k, n), (n, 1), "taps")
    spec = matrix_rules.bf16_spec("grad", "taps", "out", la, lb, (m, n), engine_ok=True)
    assert spec.get("engine") == "p5c", spec
    return taps, spec, la, lb, (m, n)


def _select(candidate, **options):
    taps, spec, la, lb, shape = candidate
    return taps.select(spec, "flat admitted", "grad", "taps", "out", la, lb, shape,
                       {"grad": "bfloat16", "taps": "bfloat16", "out": "bfloat16"},
                       p5c_ok=options.get("p5c_ok", True), accumulate=options.get("accumulate", False))


def test_underfilled_flat_conv_weight_gradient_uses_existing_split_body():
    candidate = _candidate()
    spec, reason = _select(candidate)
    assert spec["conv_splitk"] == 3 and spec["p5c_tile"] == (128, 128, 2), spec
    assert (spec["m"], spec["n"], spec["k"]) == (256, 2304, 18432)
    assert "flat admission relied on a long-K split" in reason


@pytest.mark.parametrize("restriction", ("split_disabled", "not_taps", "accumulate", "p5c_disabled"))
def test_flat_conv_body_is_kept_when_split_path_is_ineligible(monkeypatch, restriction):
    candidate = _candidate()
    options = {}
    if restriction == "split_disabled":
        monkeypatch.setattr(conv_ops, "SPLITK", False)
    elif restriction == "not_taps":
        candidate[0].deferred.clear()
    elif restriction == "accumulate":
        options["accumulate"] = True
    else:
        options["p5c_ok"] = False
    spec, reason = _select(candidate, **options)
    assert spec is candidate[1] and reason == "flat admitted"


def test_flat_conv_body_with_enough_tiles_is_unchanged():
    candidate = _candidate(m=512, n=4608)
    spec, reason = _select(candidate)
    assert spec is candidate[1] and reason == "flat admitted"
