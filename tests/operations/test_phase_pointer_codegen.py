"""Phase-local opaque bindings: inverse bytes, short lifetimes, refusal edges."""
import hashlib
import json
import re

import pytest
import torch

from megakernel_lab.operation_compiler import capture_module
from megakernel_lab.operation_compiler.elementwise_codegen import fold as share, _phase_bodies
from megakernel_lab.operation_compiler.lower import GeneratedProgram, LoweringError, lower
from megakernel_lab.operation_compiler.phase_pointer_codegen import (
    fold, Refusal, KEY, MARKER, MIN_PHASES, OPAQUE_HELPER, SCALAR_SOURCE)
from megakernel_lab.operation_compiler.user_optimizer import TensorUpdateRule
from .fixtures.tinymoe_source_proofs import original_declarations, relocated_names


def fixture(count=MIN_PHASES, *, kind="elementwise", typed=False):
    declarations, bodies = [], []
    for i in range(count):
        if typed and i == 0:
            declarations += ["__nv_bfloat16* v_x_0 = reinterpret_cast<__nv_bfloat16*>(latch_pa[0]) + 2LL;\n",
                             "float* v_y_0 = latch_pr + 64LL + 3LL;\n"]
            expression = "__bfloat162float(v_x_0[j])"
        else:
            declarations += [f"float* v_x_{i} = latch_pa[{i}];\n", f"float* v_y_{i} = latch_pr + {i * 8}LL;\n"]
            expression = f"v_x_{i}[j]"
        bodies.append(f"for (long long j = lsr_tx; j < 8LL; j += lsr_nt) {{\n"
                      f"const float phase_temporary_long_register_{i} = {expression} * 2.0f;\n"
                      f"v_y_{i}[j] = phase_temporary_long_register_{i};\n}}\n")
    prelude = "float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n"
    prelude += "".join(f'unsigned {name} = {source}.x; asm volatile("" : "+r"({name}));\n'
                       for name, source in SCALAR_SOURCE.items()) + "".join(declarations) + "\n"
    source = OPAQUE_HELPER + '\nextern "C" __global__ void latch_ops(float** args, float* arena, unsigned long long* ticks) {\n'
    source += "cg::grid_group grid = cg::this_grid();\n{\n" + prelude
    source += "\n__syncthreads();\n".join(f"// Phase {i}: independent\n{{\n{body}}}" for i, body in enumerate(bodies))
    source += "\ngrid.sync();\nif (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[1] = stamp();\n}\n"
    source += "// Commit remains caller owned\ngrid.sync();\n}\n"
    manifest = dict(phases=[dict(kind=kind) for _ in bodies], header_hashes={},
                    execution_selection={"selected": "common_static"}, barrier_schedule="dependency",
                    group_rules={"local_pointers": ["lptr", "lsreg"]}, barrier_intervals=[list(range(count))])
    manifest["source_hash"] = hashlib.sha256((source + json.dumps({}, sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest)


def independently_expand(before, after):
    old = {i: before.source[a:b] for i, a, b in _phase_bodies(before.source)}
    source, replacements, prefixes = after.source, [], {}
    for i, a, b in _phase_bodies(source):
        prefix = re.match(r"// " + MARKER + rf": {i}\n.*?// {MARKER}: end\n", source[a:b], re.S)
        stripped = source[a:b] if prefix is None else source[a + prefix.end():b]
        assert stripped == old[i]
        if prefix:
            prefixes[i] = prefix[0]
            replacements.append((a, a + prefix.end(), ""))
    for a, b, value in reversed(replacements):
        source = source[:a] + value + source[b:]
    for group in reversed(after.manifest[KEY]["groups"]):
        i = group["phases"][0]
        pattern = re.compile(r"// Phase " + str(i) + r":")
        a = pattern.search(before.source).start(); b = pattern.search(source).start()
        old_start = before.source.rfind("{\n", 0, a) + 2
        new_start = source.rfind("{\n", 0, b) + 2
        declarations = original_declarations(before.source[old_start:a])
        for phase in group["phases"]:
            relocated_names(prefixes[phase], declarations, MARKER, phase)
        source = source[:new_start] + before.source[old_start:a] + source[b:]
    assert source == before.source
    assert {k: v for k, v in before.manifest.items() if k != "source_hash"} == {
        k: v for k, v in after.manifest.items() if k not in ("source_hash", KEY)}


@pytest.mark.parametrize("compacted", [False, True])
def test_scopes_shrink_and_every_body_barrier_tick_and_commit_reexpands(compacted):
    before = fixture()
    if compacted:
        before = share(before)
        assert before.manifest["elementwise_codegen"]["shared_phases"] == MIN_PHASES
    after = fold(before); proof = after.manifest[KEY]
    assert proof["localized_intervals"] == 1 and proof["localized_phases"] == MIN_PHASES
    assert proof["groups"][0]["interval_pointer_count"] == 2 * MIN_PHASES
    assert proof["groups"][0]["max_phase_pointer_count"] == 2
    independently_expand(before, after)
    for _, a, b in _phase_bodies(after.source):
        assert after.source[a:b].count("latch_opq(args)") == 1
        assert after.source[a:b].count("latch_opq(arena)") == 1
    damaged = GeneratedProgram(after.source.replace("2.0f", "3.0f", 1), after.manifest)
    with pytest.raises(AssertionError):
        independently_expand(before, damaged)


@pytest.mark.parametrize("mutation", ["rhs", "type", "base"])
def test_independent_prefix_proof_rejects_moved_binding_corruption(mutation):
    before = fixture(); after = fold(before)
    if mutation == "rhs": after.source = after.source.replace("float* v_x_0 = latch_pa[0];", "float* v_x_0 = latch_pa[1];", 1)
    elif mutation == "type": after.source = after.source.replace("float* v_x_0 =", "double* v_x_0 =", 1)
    else: after.source = after.source.replace("float** latch_pa = latch_opq(args);", "float** latch_pa = latch_opq(out);", 1)
    with pytest.raises(AssertionError): independently_expand(before, after)


def test_alias_addresses_typed_views_and_identifiers_are_preserved():
    before = fixture(typed=True)
    before.source = before.source.replace("float* v_y_1 = latch_pr + 8LL;", "float* v_y_1 = latch_pa[1];")
    after = fold(before)
    assert after.manifest[KEY]["localized_phases"] == MIN_PHASES
    assert "reinterpret_cast<__nv_bfloat16*>(latch_pa[0]) + 2LL" in after.source
    assert "float* v_y_1 = latch_pa[1];" in after.source
    independently_expand(before, after)


@pytest.mark.parametrize("replacement", [
    "volatile float* v_x_0", "const float* v_x_0", "float* volatile v_x_0", "float* const v_x_0",
    "float* __restrict__ v_x_0", "float*& v_x_0", "float** v_x_0", "float*volatile v_x_0",
])
def test_qualifiers_are_never_erased(replacement):
    before = fixture(); before.source = before.source.replace("float* v_x_0", replacement, 1)
    after = fold(before)
    assert after.source == before.source and after.manifest[KEY]["localized_phases"] == 0


@pytest.mark.parametrize("mutation", ["dependent", "reassign", "escape", "table", "shared", "unknown_call",
                                     "cross_phase", "scalar_width", "scalar_mismatch", "unsupported_prelude"])
def test_unsupported_scope_forms_stay_byte_identical(mutation):
    before = fixture()
    if mutation == "dependent": before.source = before.source.replace("latch_pa[0]", "v_y_1", 1)
    elif mutation == "reassign": before.source = before.source.replace("// Phase 0: independent\n{\n", "// Phase 0: independent\n{\nv_x_0 = v_y_0;\n")
    elif mutation == "escape": before.source = before.source.replace("v_x_0[j]", "*(&v_x_0)")
    elif mutation == "table": before.source = before.source.replace("v_x_0[j]", "args[0][j]")
    elif mutation == "shared": before.source = before.source.replace("v_x_0[j]", "smem[j]")
    elif mutation == "unknown_call": before.source = before.source.replace("v_x_0[j]", "side_effect(v_x_0)")
    elif mutation == "cross_phase": before.source = before.source.replace("grid.sync();", "v_x_0[0] = 1.0f;\ngrid.sync();", 1)
    elif mutation == "scalar_width": before.source = before.source.replace("unsigned lsr_tx =", "unsigned long long lsr_tx =")
    elif mutation == "scalar_mismatch": before.source = before.source.replace("lsr_tx = threadIdx.x", "lsr_tx = blockIdx.x")
    else: before.source = before.source.replace("// Phase 0:", "long long retained_counter = 0;\n// Phase 0:")
    after = fold(before)
    assert after.source == before.source and after.manifest[KEY]["localized_phases"] == 0


@pytest.mark.parametrize("mutation", ["kind", "counter", "scratch", "grid", "too_small"])
def test_initializers_and_other_cross_phase_resources_are_excluded(mutation):
    before = fixture(MIN_PHASES - 1 if mutation == "too_small" else MIN_PHASES)
    if mutation == "kind": before.manifest["phases"][0]["kind"] = "reduction_counter_init"
    if mutation == "counter": before.manifest["phases"][0]["counter_effects"] = [{"storage": "counter"}]
    if mutation == "scratch": before.manifest["phases"][0]["private_scratch"] = ["scratch"]
    if mutation == "grid": before.manifest["phases"][0]["requires_full_grid"] = True
    after = fold(before)
    assert after.source == before.source and after.manifest[KEY]["localized_phases"] == 0


@pytest.mark.parametrize("mutation", ["helper_effect", "macro", "interval", "region", "double"])
def test_unproven_program_contracts_fail_closed(mutation):
    before = fixture()
    if mutation == "helper_effect": before.source = before.source.replace("return p;", "return p + 1;")
    elif mutation == "macro": before.source = "#define latch_opq other\n" + before.source
    elif mutation == "interval": before.manifest["barrier_intervals"][0] = []
    elif mutation == "region": before.manifest["region_body"] = True
    else: before = fold(before)
    with pytest.raises(Refusal): fold(before)


def test_ordinary_capture_mode_and_default_identity():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(8 + i)) for i in range(40)])
        # Distinct extents prevent the ordinary elementwise fusion pass from
        # collapsing this liveness fixture to one common-shape update phase.
        def forward(self, x): return sum(((i + 1) * p * x[:p.numel()]).sum() for i, p in enumerate(self.weights))
    def update(params, grads, state, hyper):
        return {n: p - .001 * grads[n] for n, p in params.items()}, state
    captured = capture_module(Model(), lambda y: y.square().mean(), (torch.ones(48),),
                              optimizer=TensorUpdateRule(update))
    options = dict(execution="compose", compose_selection="common", precision_policy="practical")
    ordinary = lower(captured.ir, **options)
    explicit = lower(captured.ir, compact_codegen="off", **options)
    assert ordinary.source == explicit.source and ordinary.manifest == explicit.manifest
    candidate = lower(captured.ir, compact_codegen="phase_pointers", **options)
    assert candidate.manifest[KEY]["localized_phases"] >= 40, candidate.manifest[KEY]
    independently_expand(ordinary, candidate)
    compact = lower(captured.ir, compact_codegen="elementwise_full_sum", **options)
    composed = lower(captured.ir, compact_codegen="elementwise_full_sum_phase_pointers", **options)
    assert composed.manifest[KEY]["localized_phases"] >= 40
    independently_expand(compact, composed)
    again = lower(captured.ir, compact_codegen="elementwise_full_sum", **options)
    assert again.source == compact.source and again.manifest == compact.manifest
    with pytest.raises(LoweringError, match="standalone static common"):
        lower(captured.ir, compact_codegen="phase_pointers")
