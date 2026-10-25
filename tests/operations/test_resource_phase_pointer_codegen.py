"""Finite resource forms, use-set pinning and independent emitted-byte inverse."""
import copy
import re

import pytest
from megakernel_lab.operation_compiler import extra_ops
from megakernel_lab.operation_compiler.lower import GeneratedProgram, TensorLayout
from megakernel_lab.operation_compiler.elementwise_codegen import _phase_bodies
from megakernel_lab.operation_compiler import mixed_phase_pointer_codegen as mixed
from megakernel_lab.operation_compiler.resource_phase_pointer_codegen import fold, KEY, MARKER, Refusal, _cat_body
from .fixtures.tinymoe_source_proofs import original_declarations, relocated_names

from .test_phase_pointer_codegen import fixture


def resource_fixture():
    program = fixture(36)
    cat = dict(kind="cat", names=["y_1"], reads=["x_0", "x_1"], writes=["y_1"],
               targets=["aten.cat.default"], vector_copy=True)
    layouts = {n: dict(shape=[4, 8] if n != "y_1" else [8, 8], strides=[8, 1], root=n, offset=0, dtype="float32")
               for n in ("x_0", "x_1", "y_1")}
    program.manifest["layouts"] = layouts
    # Use the ordinary emitter, independently of resource admission.
    ls = {n: TensorLayout(tuple(v["shape"]), tuple(v["strides"]), n) for n, v in layouts.items()}
    body = extra_ops.emit("cat.default", (["x_0", "x_1"], 0), {}, "y_1", (8, 8), ls,
                          {n:"float32" for n in ls}, set(ls), None)[1] + "\n"
    from megakernel_lab.operation_compiler.phase_pointer_codegen import SCALAR_SOURCE
    for reg, builtin in SCALAR_SOURCE.items(): body = body.replace(builtin + ".x", reg)
    replacements = list(_phase_bodies(program.source))[:2]
    for i, a, b in reversed(replacements):
        text = body if i else "if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_y_0) = 0u;\n"
        program.source = program.source[:a] + text + program.source[b:]
    program.manifest["phases"][0] = dict(kind="reduction_counter_init", names=["y_0"], reads=[], targets=[], writes=["y_0"])
    program.manifest["phases"][1] = cat
    return mixed.fold(program)


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


def test_exact_counter_and_cat_move_only_bindings():
    before = resource_fixture(); after = fold(before)
    proof = after.manifest[KEY]
    assert proof["localized_phases"] == 2
    group = proof["groups"][0]
    assert group["localized_pointer_count"] == 4
    assert group["phase_kinds"] == {"reduction_counter_init":1, "cat":1}
    independently_expand(before, after)
    assert after.manifest[mixed.KEY] == before.manifest[mixed.KEY]


@pytest.mark.parametrize("suffix", [
    "v_x_0 = v_x_1;", "++v_x_0;", "v_x_0--;", "take(&v_x_0);", "float*& alias = v_x_0;",
    "take(reinterpret_cast<float**>(v_x_0));", "take(reinterpret_cast<float*&>(v_x_0));",
    "unknown(v_x_0);", "MACRO(v_x_0);", "take(&v_x_0[0]);", "__shared__ float scratch[4];",
])
def test_any_extra_cat_form_is_refused_and_its_bindings_stay_outer(suffix):
    before = resource_fixture()
    _, a, b = list(_phase_bodies(before.source))[1]
    before.source = before.source[:b] + suffix + "\n" + before.source[b:]
    after = fold(before)
    assert after.manifest[KEY]["localized_phases"] == 1
    assert after.manifest[KEY]["groups"][0]["phases"] == [0]
    independently_expand(before, after)


@pytest.mark.parametrize("mutation", ["counter_value", "counter_type", "counter_condition", "cat_index", "cat_load", "cat_store", "cat_zero"])
def test_exact_template_mutations_are_refused(mutation):
    before = resource_fixture()
    replacements = {
        "counter_value": (" = 0u;", " = 1u;"),
        "counter_type": ("reinterpret_cast<unsigned*>", "reinterpret_cast<int*>"),
        "counter_condition": ("lsr_bx == 0 && lsr_tx == 0", "lsr_tx == 0"),
        "cat_index": ("v_ < 16u", "v_ < 15u"),
        "cat_load": ("reinterpret_cast<const float*>(v_x_0)", "reinterpret_cast<const float*>(v_x_0) + 1"),
        "cat_store": ("reinterpret_cast<float*>(v_y_1)", "reinterpret_cast<float*>(v_y_1) + 1"),
        "cat_zero": ("make_float4(0.f, 0.f, 0.f, 0.f)", "make_float4(1.f, 0.f, 0.f, 0.f)")}
    old, new = replacements[mutation]; assert old in before.source
    before.source = before.source.replace(old, new, 1)
    after = fold(before)
    assert after.manifest[KEY]["localized_phases"] == 1
    independently_expand(before, after)


@pytest.mark.parametrize("qualifier", ["const float*", "float* volatile", "float* const", "float* __restrict__", "float*&", "float**"])
def test_qualified_or_dependent_outer_bindings_are_not_moved(qualifier):
    before = resource_fixture(); before.source = before.source.replace("float* v_x_0 =", qualifier + " v_x_0 =", 1)
    after = fold(before); assert after.source == before.source
    assert after.manifest[KEY]["localized_phases"] == 0


@pytest.mark.parametrize("extra", ["v_y_0[0] = 1.f;", "take(&v_y_0);", "v_y_0 = v_x_2;"])
def test_unsupported_phase_references_pin_resource_binding(extra):
    before = resource_fixture(); _, a, b = list(_phase_bodies(before.source))[2]
    before.source = before.source[:b] + extra + "\n" + before.source[b:]
    after = fold(before)
    assert after.manifest[KEY]["groups"][0]["phases"] == [1]
    independently_expand(before, after)


@pytest.mark.parametrize("extra", ["args[0] = v_x_0;", "arena[0] = 0.f;", "latch_pa[0] = v_x_0;"])
def test_pointer_table_or_base_mutations_refuse_interval(extra):
    before = resource_fixture(); _, _, b = list(_phase_bodies(before.source))[1]
    before.source = before.source[:b] + extra + "\n" + before.source[b:]
    after = fold(before); assert after.source == before.source


@pytest.mark.parametrize("which", ["rhs", "type", "base", "retained", "body", "manifest"])
def test_independent_inverse_rejects_corruption(which):
    before = resource_fixture(); after = fold(before)
    start = after.source.index("// " + MARKER)
    left, right = after.source[:start], after.source[start:]
    if which == "rhs": right = right.replace("v_y_0 = latch_pr + 0LL", "v_y_0 = latch_pr + 4LL", 1)
    elif which == "type": right = right.replace("float* v_y_0", "double* v_y_0", 1)
    elif which == "base": right = right.replace("latch_opq(arena)", "latch_opq(args[0])", 1)
    elif which == "retained": left = left.replace("latch_opq(args)", "latch_opq(out)", 1)
    elif which == "body": right = right.replace(" = 0u;", " = 1u;", 1)
    else:
        after.manifest = copy.deepcopy(after.manifest)
        after.manifest["phases"][0]["private_scratch"] = ["changed"]
    after.source = left + right
    with pytest.raises(AssertionError): independently_expand(before, after)


@pytest.mark.parametrize("kind", ["counter_effects", "private_scratch", "requires_full_grid"])
def test_resource_metadata_is_never_reinterpreted(kind):
    before = resource_fixture(); before.manifest["phases"][1][kind] = ["unproven"]
    after = fold(before); assert after.manifest[KEY]["groups"][0]["phases"] == [0]
    independently_expand(before, after)


def test_required_prior_mode_and_double_application_fail_closed():
    with pytest.raises(Refusal): fold(fixture())
    with pytest.raises(Refusal): fold(fold(resource_fixture()))
