"""Opt-in runtime interval probes (ops.runtime_trace): source form, device-unit form, decoding."""
import re

import pytest

from megakernel_lab.operation_compiler import runtime_trace as rt
from megakernel_lab.operation_compiler import step17_intervals as si
from megakernel_lab.operation_compiler.lower import lower


def _value(shape, dtype):
    stride, strides = 1, []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {"shape": shape, "strides": list(reversed(strides)), "dtype": dtype}


GRAPH = {"inputs": ["x", "w", "a", "b"], "outputs": ["h", "aux", "s"],
         "values": {"x": _value([1024, 512], "bfloat16"), "w": _value([512, 1024], "bfloat16"),
                    "h": _value([1024, 1024], "bfloat16"), "a": _value([2, 17, 19], "float32"),
                    "b": _value([2, 19, 23], "float32"), "aux": _value([2, 17, 23], "float32"),
                    "s": _value([2, 17, 23], "float32")},
         "operations": [
             {"name": "h", "target": "aten.mm.default", "args": [{"ref": "x"}, {"ref": "w"}]},
             {"name": "aux", "target": "aten.bmm.default", "args": [{"ref": "a"}, {"ref": "b"}]},
             {"name": "s", "target": "aten.mul.Tensor", "args": [{"ref": "aux"}, 2]}]}


def _programs(mode=True):
    base = lower(GRAPH, execution="compose", compose_selection="common")
    with rt.scope(mode):
        return base, lower(GRAPH, execution="compose", compose_selection="common")


def _kernel_blocks(program):
    _, _, body, _ = si.kernel_parts(program.source)
    return [body[a:b] for kind, a, b in si.scan(body) if kind == "block"]


def test_off_leaves_the_program_unchanged():
    first = lower(GRAPH, execution="compose", compose_selection="common")
    with rt.scope(False):
        second = lower(GRAPH, execution="compose", compose_selection="common")
    assert first.source == second.source and rt.KEY not in second.manifest


def test_interval_text_is_unchanged_up_to_its_closing_barrier():
    base, probed = _programs()
    planned = base.manifest["shared_bytes"]
    assert probed.manifest["shared_bytes"] == planned + rt.GUARD_BYTES
    assert probed.manifest["runtime_trace_planned_shared_bytes"] == planned
    before, after = _kernel_blocks(base), _kernel_blocks(probed)
    assert len(before) == len(after) and "grid.sync" not in "".join(after)
    for old, new in zip(before, after):
        assert old.startswith(new[:-1])
    assert probed.manifest["source_hash"] != base.manifest["source_hash"]


def test_every_interval_has_one_land_and_one_done_and_a_tick_area():
    _, probed = _programs()
    record = probed.manifest[rt.KEY]
    count = len(record["intervals"])
    assert count == probed.source.count("latch_rtp::done(ticks,") and count >= 2
    assert probed.source.count("latch_rtp::land(ticks,") == count
    assert probed.source.count("latch_rtp::arm(ticks)") == 1
    assert probed.source.count("latch_rtp::publish<") == count
    assert probed.manifest["tick_words"] == record["fault_base"] + rt.STRIDE + 1
    assert record["base"] >= probed.manifest["phase_count"] + 2
    assert sorted(p for row in record["intervals"] for p in row["phases"]) == sorted(
        {p for row in record["intervals"] for p in row["phases"]})


def test_no_define_the_device_unit_build_refuses_and_kernel_scope_stays_recognized():
    _, probed = _programs()
    assert not re.search(r'^\s*#\s*define\s+\w*TRACE\w*', probed.source, re.M)
    _, _, body, _ = si.kernel_parts(probed.source)
    si.check_kernel_scope(body, si.scan(body))


def test_split_interval_sees_the_same_phases():
    block = ("{\nfloat** latch_pa = latch_opq(args);\n// Phase 0: a\n{\nwork(0);\n}\n__syncthreads();\n// Phase 1: b\n{\n"
             "pops4b::tma_phase_inline<1>();\n}\n__syncthreads();\n// Phase 2: c\n{\nwork(2);\n}\ngrid.sync();\n"
             "if (ticks && blockIdx.x == 0 && threadIdx.x == 0) { const unsigned long long t = stamp(); ticks[1] = t; ticks[2] = t; }\n}")
    inject, start, end, words = rt._sites(block)
    assert words == [1, 2] and block[start:end] == "grid.sync();" and block[inject:].startswith("work(0);")
    kept = block[:start] + "}"
    plain = si.split_interval("f", block, 150000, si.WGMMA_INLINE)
    split = si.split_interval("f", kept, 150000, si.WGMMA_INLINE)
    assert plain is not None and split is not None and len(plain[0]) == len(split[0]) == 3
    assert [text for _, text, _ in plain[0]] == [text for _, text, _ in split[0]]


def test_a_chained_block_injects_at_its_top_and_an_unknown_tail_refuses():
    block = "{\nif (blockIdx.x == 0) {\nwork();\n}\ngrid.sync();\n}"
    assert rt._sites(block)[0] == 1
    assert rt._sites("{\nno_barrier();\n}") is None
    with pytest.raises(rt.Refusal):
        rt._sites("{\nwork();\ngrid.sync();\nmore();\n}")


def test_inject_mode_adds_only_the_test_sites():
    _, on = _programs()
    _, injected = _programs("inject")
    record = injected.manifest[rt.KEY]
    assert record["inject_word"] == record["fault_base"] + rt.STRIDE
    assert injected.source.count("latch_rtp::inject(ticks, smem,") == len(record["intervals"])
    assert "inject" not in on.source and on.manifest[rt.KEY]["inject_word"] is None


def test_validate_refuses_unknown_values():
    with pytest.raises(ValueError):
        rt.validate("on")


def test_decode_and_report_name_the_faulting_interval():
    _, probed = _programs()
    manifest = probed.manifest
    record = manifest[rt.KEY]
    words = [0] * manifest["tick_words"]
    base, stride = record["base"], record["stride"]
    for k in range(len(record["intervals"])):
        for c in range(2):
            words[base + rt.SLOTS * k * stride + c] = 1000 + 100 * k + c
            words[base + (rt.SLOTS * k + 1) * stride + c] = 1040 + 100 * k + 2 * c
            words[base + (rt.SLOTS * k + 2) * stride + c] = 1050 + 100 * k + 2 * c
    words[record["fault_base"] + 1] = (1 << 63) | (1 << 62) | 1
    decoded = rt.decode(manifest, [w - (1 << 64) if w >= 1 << 63 else w for w in words], 2)
    assert decoded["faults"] == {1: dict(interval=1, first_word=True, last_word=False)}
    assert decoded["intervals"][0]["ctas"] == 2 and decoded["intervals"][0]["done"][1] == 1052
    assert decoded["intervals"][0]["end"][1] == 1042
    lines = rt.report(decoded)
    assert lines[-1].startswith("FAULT shared-memory guard overwritten in interval 1 (CTA 1")
    assert "no fault" in rt.report(rt.decode(manifest, [0] * manifest["tick_words"], 2))[-1]
