"""Pure-exclusive relocation keeps unsupported/shared bindings in place."""
import re

import pytest
import torch

from megakernel_lab.operation_compiler import capture_module
from megakernel_lab.operation_compiler.elementwise_codegen import _phase_bodies
from megakernel_lab.operation_compiler.lower import GeneratedProgram, lower
from megakernel_lab.operation_compiler import phase_pointer_codegen as prior
from megakernel_lab.operation_compiler.mixed_phase_pointer_codegen import fold, KEY, MARKER, Refusal
from megakernel_lab.operation_compiler.user_optimizer import TensorUpdateRule
from .fixtures.tinymoe_source_proofs import original_declarations, relocated_names

from .test_phase_pointer_codegen import fixture


def mixed_fixture():
    program = fixture(prior.MIN_PHASES + 1)
    phase, start, end = next(_phase_bodies(program.source))
    assert phase == 0
    body = ("if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_y_0) = 0u;\n"
            "v_x_1 = v_x_0;\n")
    program.source = program.source[:start] + body + program.source[end:]
    program.manifest["phases"][0] = dict(kind="reduction_counter_init", private_scratch=["private"],
                                        counter_effects=[dict(storage="private")])
    return program


def independently_expand(before, after):
    old = {i: before.source[a:b] for i, a, b in _phase_bodies(before.source)}
    source, edits, prefixes = after.source, [], {}
    for i, a, b in _phase_bodies(source):
        prefix = re.match(r"// " + MARKER + rf": {i}\n.*?// {MARKER}: end\n", source[a:b], re.S)
        body = source[a:b] if prefix is None else source[a + prefix.end():b]
        assert body == old[i]
        if prefix:
            prefixes[i] = prefix[0]; edits.append((a, a + prefix.end()))
    assert prefixes.keys() == {i for group in after.manifest[KEY]["groups"] for i in group["phases"]}
    for a, b in reversed(edits): source = source[:a] + source[b:]
    for group in reversed(after.manifest[KEY]["groups"]):
        pattern = re.compile(r"// Phase " + str(group["first_phase"]) + r":")
        a, b = pattern.search(before.source).start(), pattern.search(source).start()
        old_start, new_start = before.source.rfind("{\n", 0, a) + 2, source.rfind("{\n", 0, b) + 2
        declaration = re.compile(r"^(?:float|double|__nv_bfloat16|int64_t|bool)\* (v_\w+) = [^\n]+;\n", re.M)
        old_prelude = before.source[old_start:a]
        declarations = original_declarations(old_prelude)
        moved = set().union(*(relocated_names(prefixes[i], declarations, MARKER, i) for i in group["phases"]))
        expected = "".join(line for line in old_prelude.splitlines(keepends=True)
                           if not ((match := declaration.fullmatch(line)) and match[1] in moved))
        assert source[new_start:b] == expected, "retained declaration changed"
        for i in group["unchanged_phases"]:
            assert not moved & set(re.findall(r"\bv_\w+\b", old[i]))
        source = source[:new_start] + before.source[old_start:a] + source[b:]
    assert source == before.source
    assert {k: v for k, v in before.manifest.items() if k != "source_hash"} == {
        k: v for k, v in after.manifest.items() if k not in ("source_hash", KEY)}


def test_shared_reassigned_pointer_and_resource_phase_remain_outer():
    before = mixed_fixture(); after = fold(before); group = after.manifest[KEY]["groups"][0]
    assert group["first_phase"] == 0 and group["phases"] == list(range(1, 33))
    assert group["localized_pointer_count"] == 63 and group["retained_pointer_count"] == 3
    assert group["max_phase_local_pointer_count"] == 2
    prefix = after.source[:after.source.index("// Phase 0:")]
    assert "float* v_x_1 = latch_pa[1];" in prefix
    assert "float* v_x_0 = latch_pa[0];" in prefix and "float* v_y_0 = latch_pr + 0LL;" in prefix
    assert "float* v_y_1 = latch_pr + 8LL;" not in prefix
    original_bodies = {i: before.source[a:b] for i, a, b in _phase_bodies(before.source)}
    for i, a, b in _phase_bodies(after.source):
        if i == 0: assert after.source[a:b] == original_bodies[0]
        else: assert "float* v_x_1 =" not in after.source[a:b]
    independently_expand(before, after)
    corrupted = GeneratedProgram(after.source.replace("= 0u;", "= 1u;", 1), after.manifest)
    with pytest.raises(AssertionError): independently_expand(before, corrupted)
    corrupted = GeneratedProgram(after.source.replace("float* v_x_1 = latch_pa[1];", "float* v_x_1 = latch_pa[2];", 1), after.manifest)
    with pytest.raises(AssertionError, match="retained declaration"): independently_expand(before, corrupted)


@pytest.mark.parametrize("mutation", ["rhs", "type", "base"])
def test_independent_prefix_proof_rejects_moved_binding_corruption(mutation):
    before = mixed_fixture(); after = fold(before)
    if mutation == "rhs": after.source = after.source.replace("float* v_y_1 = latch_pr + 8LL;", "float* v_y_1 = latch_pr + 16LL;", 1)
    elif mutation == "type": after.source = after.source.replace("float* v_y_1 =", "double* v_y_1 =", 1)
    else:
        marker = after.source.index("// " + MARKER)
        after.source = after.source[:marker] + after.source[marker:].replace("float* latch_pr = latch_opq(arena);", "float* latch_pr = latch_opq(args[0]);", 1)
    with pytest.raises(AssertionError): independently_expand(before, after)


@pytest.mark.parametrize("where", ["unsupported", "interphase", "commit", "address"])
def test_any_external_pointer_use_pins_its_original_binding(where):
    before = mixed_fixture()
    if where == "unsupported": before.source = before.source.replace("v_x_1 = v_x_0;", "v_x_1 = v_x_0; v_y_2[0] = 1.0f;")
    elif where == "address": before.source = before.source.replace("v_x_1 = v_x_0;", "v_x_1 = v_x_0; use_binding(&v_y_2);")
    elif where == "interphase": before.source = before.source.replace("__syncthreads();", "use_value(v_y_2);\n__syncthreads();", 1)
    else: before.source = before.source.replace("grid.sync();", "v_y_2[0] = 1.0f;\ngrid.sync();", 1)
    after = fold(before)
    assert after.manifest[KEY]["groups"][0]["retained_pointer_count"] == 4
    assert "float* v_y_2 = latch_pr + 16LL;" in after.source[:after.source.index("// Phase 0:")]
    independently_expand(before, after)


@pytest.mark.parametrize("replacement", ["const float* v_x_1", "float* volatile v_x_1", "float* __restrict__ v_x_1"])
def test_even_retained_pointer_qualifiers_refuse_the_interval(replacement):
    before = mixed_fixture(); before.source = before.source.replace("float* v_x_1", replacement, 1)
    after = fold(before)
    assert after.source == before.source and after.manifest[KEY]["localized_phases"] == 0


@pytest.mark.parametrize("change", ["dependent", "table", "direct_arena", "too_few", "missing_scalar"])
def test_unproven_admission_does_not_change_source(change):
    before = mixed_fixture()
    if change == "dependent": before.source = before.source.replace("latch_pa[1]", "v_x_2", 1)
    elif change == "table": before.source = before.source.replace("v_x_1 = v_x_0;", "args[1] = v_x_0;")
    elif change == "direct_arena": before.source = before.source.replace("v_x_1 = v_x_0;", "arena[0] = 0.0f;")
    elif change == "missing_scalar": before.source = before.source.replace('unsigned lsr_tx = threadIdx.x; asm volatile("" : "+r"(lsr_tx));\n', '')
    else: before.manifest["phases"][1]["kind"] = "row_fused"
    after = fold(before)
    assert after.source == before.source and after.manifest[KEY]["localized_phases"] == 0


def test_refused_pure_body_is_unsupported_and_pins_all_its_bindings():
    before = fixture(prior.MIN_PHASES + 2)
    before.source = before.source.replace("v_x_0[j] * 2.0f", "custom_effect(v_x_0, v_x_1)")
    after = fold(before); group = after.manifest[KEY]["groups"][0]
    assert 0 in group["unchanged_phases"] and group["retained_pointer_count"] == 3
    independently_expand(before, after)


def test_old_mode_remains_exact_and_modes_cannot_be_stacked():
    before = fixture(); original = prior.fold(before)
    mixed = fold(before)
    assert prior.fold(before).source == original.source and prior.fold(before).manifest == original.manifest
    with pytest.raises(Refusal): fold(original)
    with pytest.raises(Refusal): fold(mixed)
    independently_expand(before, mixed)


def test_ordinary_lowering_modes_and_off_identity(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(8 + i)) for i in range(40)])
        def forward(self, x): return sum((p * x[:p.numel()]).sum() for p in self.p)
    def update(params, grads, state, hyper): return {n: p - .001 * grads[n] for n, p in params.items()}, state
    captured = capture_module(Model(), lambda y: y.square().mean(), (torch.ones(48),), optimizer=TensorUpdateRule(update))
    options = dict(execution="compose", compose_selection="common", precision_policy="practical")
    baseline = lower(captured.ir, compact_codegen="elementwise_full_sum", **options)
    old = lower(captured.ir, compact_codegen="elementwise_full_sum_phase_pointers", **options)
    mixed = lower(captured.ir, compact_codegen="elementwise_full_sum_mixed_phase_pointers", **options)
    assert mixed.manifest[KEY]["localized_phases"] >= 40
    independently_expand(baseline, mixed)
    again = lower(captured.ir, compact_codegen="elementwise_full_sum_phase_pointers", **options)
    assert old.source == again.source and old.manifest == again.manifest
    direct = lower(captured.ir, compact_codegen="mixed_phase_pointers", **options)
    independently_expand(lower(captured.ir, **options), direct)
