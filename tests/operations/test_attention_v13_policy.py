"""Source-only legacy attention and explicit device-policy regressions.

Golden identities were obtained from the owned pre-resumption checkpoint
e3536ae68966048a1d50568588dfe0c38817c9d4 and independently compared with
the current emitter. The finite-mask online-softmax overflow repair deliberately
changes one legacy body: normalize only that exact edit before checking that all
other legacy bytes remain unchanged. Its behavior is checked separately by the
FP64 window oracle and old/new causal bitwise GPU gate in window_regression.py.
No CUDA compilation or tensor computation is needed for this source-only lock.
"""
import hashlib
import json
from pathlib import Path

import pytest


LEGACY_HEADER = "554da18225e98a524ef442f7b118778e876d17f2b9681e90cef9519fbcaaedc7"
LEGACY_EMIT = {
    3: "2e5133c69d3749c3266606f92f58918d31d23ebfc6b50db75f2e38ded587cf31",
    7: "2b13252de274c893033cedf68f68114bfbd6bcc302766f54f56bce0f8036d796",
}


def test_all_other_legacy_header_bytes_are_unchanged():
    from megakernel_lab.operation_compiler import flat_long_attention
    header = (Path(flat_long_attention.__file__).parent / "kernels" / "attention_long_flat.cuh").read_text()
    replacement = """  // A finite additive mask may use -FLT_MAX. In a leading fully masked row/tile,
  // multiplying that maximum by log2(e) overflows even though score - maximum
  // is finite. Subtract first only in that exceptional case; ordinary online
  // softmax retains its existing fused arithmetic and rounding.
  const bool shift0 = isinf(ml0) && isfinite(n0), shift1 = isinf(ml1) && isfinite(n1);
  const float c0 = m0 == -INFINITY ? 0.f : ex2(shift0 ? (m0 - n0) * L2E : fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : ex2(shift1 ? (m1 - n1) * L2E : fmaf(m1, L2E, -ml1));
"""
    original = """  const float c0 = m0 == -INFINITY ? 0.f : ex2(fmaf(m0, L2E, -ml0));
  const float c1 = m1 == -INFINITY ? 0.f : ex2(fmaf(m1, L2E, -ml1));
"""
    assert header.count(replacement) == 1
    header = header.replace(replacement, original)
    for element in range(4):
        row = element // 2
        replacement = (f"    s[n][{element}] = ex2(shift{row} ? ((FOLD ? scaled(a, s[n][{element}]) : "
                       f"s[n][{element}]) - n{row}) * L2E : fmaf(s[n][{element}], k, -ml{row}));")
        original = f"    s[n][{element}] = ex2(fmaf(s[n][{element}], k, -ml{row}));"
        assert header.count(replacement) == 1
        header = header.replace(replacement, original)
    start = header.index("// ===================================================================================================== v13 forward")
    end = header.index("// Products of one 16-wide chunk:", start)
    header = header[:start] + header[end:]
    start = header.index("// v13 backward:")
    end = header.index("template <int NS>\nPFLATLONG_DEV void dq_item3(", start)
    header = header[:start] + header[end:]
    assert hashlib.sha256(header.encode()).hexdigest() == LEGACY_HEADER


@pytest.mark.parametrize("version", (3, 7))
def test_legacy_phase_code_and_metadata_are_unchanged(version):
    from megakernel_lab.operation_compiler import flat_long_attention as flat
    assert flat.DEFAULT_VERSION == 7 and flat.DEFAULT_RECIPE == "exact3"
    stride = (1572864, 131072, 64)
    plan = dict(bh=96, t=2048, hd=64, heads=12,
        group=dict(o="o", label="policy", scale_mode=1, scale=.125, mask="mask"),
        sources=dict(q="q", k="k", v="v", do="do"),
        finals=dict(o="o", dq="dq", dkt="dk", dv="dv"),
        in_strides={key: stride for key in ("q", "k", "v", "do")},
        out_strides={key: stride for key in ("o", "dq", "dkt", "dv")})
    emitted = []
    for recipe in ("exact3", "bf16"):
        for mask in (None, (0, 2048, 1)):
            current = {**plan, "group": {**plan["group"], "mask": None if mask is None else "mask"}}
            emitted.append(flat.defines(recipe, version))
            for phase in ("prepare", "forward", "dq", "kv"):
                emitted.append(flat.emit(current, lambda name: "v_" + name, mask,
                    which=phase, recipe=recipe, version=version, keynorm_epilogue=False))
    digest = hashlib.sha256(json.dumps(emitted, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == LEGACY_EMIT[version]


def test_sm_guard_applies_only_to_the_explicit_new_contract():
    from megakernel_lab.operation_compiler.runtime import _attention_target_sms
    assert _attention_target_sms({}) is None
    assert _attention_target_sms({"attention_groups": [{"version": 13}]}) is None
    assert _attention_target_sms({"attention_groups": [{"version": 7, "deterministic_dq_target_sms": 1}]}) is None
    row = dict(backward_arithmetic="flash-bf16-deterministic-v1", deterministic_dq_target_sms=132)
    assert _attention_target_sms({"attention_groups": [row, dict(row)]}) == 132
    for bad in (None, True, 0, -1, "132"):
        with pytest.raises(ValueError, match="metadata"):
            _attention_target_sms({"attention_groups": [{**row, "deterministic_dq_target_sms": bad}]})
    with pytest.raises(ValueError, match="different"):
        _attention_target_sms({"attention_groups": [row, {**row, "deterministic_dq_target_sms": 114}]})
