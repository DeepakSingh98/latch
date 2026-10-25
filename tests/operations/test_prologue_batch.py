"""ops.prologue_batch table rules on synthetic generated-program text (CPU only)."""
from __future__ import annotations

import re

import pytest

from megakernel_lab.operation_compiler import prologue_batch, update_schedule
from megakernel_lab.operation_compiler.lower import GeneratedProgram

_BIAS = """static __device__ __noinline__ void latch_ec_body8(unsigned latch_ec_p0, unsigned latch_ec_p1, unsigned latch_ec_p2, unsigned latch_ec_p3, int64_t* latch_ec_p4, int64_t* latch_ec_p5, float* latch_ec_p6, float* latch_ec_p7, float* latch_ec_p8, float* latch_ec_p9) {
for (long long latch_ec_l0 = (long long)latch_ec_p0 * latch_ec_p1 + latch_ec_p2; latch_ec_l0 < 1LL; latch_ec_l0 += (long long)latch_ec_p3 * latch_ec_p1) {
const int64_t latch_ec_l5 = static_cast<int64_t>(static_cast<uint64_t>(latch_ec_p4[0]) + (1ULL * 1ULL)); latch_ec_p5[latch_ec_l0] = latch_ec_l5;
const float latch_ec_l8 = latch_ec_l5;
const float latch_ec_l12 = (latch_ec_l8 * latch_ec_p6[0]);
const float latch_ec_l16 = (latch_ec_p7[0] / latch_ec_l12); latch_ec_p8[latch_ec_l0] = latch_ec_l16;
const float latch_ec_l19 = sqrtf(latch_ec_l12); latch_ec_p9[latch_ec_l0] = latch_ec_l19;
}
}
"""
_DECAY = """static __device__ __noinline__ void latch_ec_body9(unsigned latch_ec_p0, unsigned latch_ec_p1, unsigned latch_ec_p2, unsigned latch_ec_p3, float* latch_ec_p4, float* latch_ec_p5, float* latch_ec_p6) {
for (long long latch_ec_l0 = (long long)latch_ec_p0 * latch_ec_p1 + latch_ec_p2; latch_ec_l0 < 1LL; latch_ec_l0 += (long long)latch_ec_p3 * latch_ec_p1) {
const float latch_ec_l5 = (latch_ec_p4[0] * latch_ec_p5[0]);
const float latch_ec_l8 = (1.0f - (latch_ec_l5 * 1.0f)); latch_ec_p6[latch_ec_l0] = latch_ec_l8;
}
}
"""


def _bias_phase(k, step_arg, div_off, sqrt_off):
    return (f"// Phase {k}: add_{k},div_{k},sqrt_{k}\n{{\n// latch_mixed_phase_pointers: {k}\n"
            "float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n"
            "float* v_arg663_1 = latch_pa[663];\nfloat* v_arg667_1 = latch_pa[667];\n"
            f"int64_t* v_arg{step_arg}_1 = reinterpret_cast<int64_t*>(latch_pa[{step_arg}]);\n"
            f"int64_t* v_add_{k} = reinterpret_cast<int64_t*>(latch_pa[{step_arg}]);\n"
            f"float* v_div_{k} = latch_pr + {div_off}LL;\nfloat* v_sqrt_{k} = latch_pr + {sqrt_off}LL;\n"
            "// latch_mixed_phase_pointers: end\n"
            f"latch_ec_body8(lsr_bx, lsr_nt, lsr_tx, lsr_nb, v_arg{step_arg}_1, v_add_{k}, v_arg667_1, v_arg663_1, "
            f"v_div_{k}, v_sqrt_{k});\n}}\n__syncthreads();\n")


def _decay_phase(k, off):
    return (f"// Phase {k}: mul_{k},rsub_{k}\n{{\n// latch_mixed_phase_pointers: {k}\n"
            "float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);\n"
            "float* v_arg663_1 = latch_pa[663];\nfloat* v_arg664_1 = latch_pa[664];\n"
            f"float* v_rsub_{k} = latch_pr + {off}LL;\n// latch_mixed_phase_pointers: end\n"
            f"latch_ec_body9(lsr_bx, lsr_nt, lsr_tx, lsr_nb, v_arg663_1, v_arg664_1, v_rsub_{k});\n\n}}\n__syncthreads();\n")


def _counter_phase(k, off):
    return (f"// Phase {k}: compose_full_sum_{k}\n{{\n// latch_resource_phase_pointers: {k}\n"
            f"float* latch_pr = latch_opq(arena);\nfloat* v_compose_full_sum_{k} = latch_pr + {off}LL;\n"
            "// latch_resource_phase_pointers: end\n"
            f"if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_compose_full_sum_{k}) = 0u;\n"
            "}\n__syncthreads();\n")


def _program(bias=10, decay=9, counters=12, step_reads=None):
    phases, text = [], []
    for k in range(counters):
        phases.append(dict(kind="reduction_counter_init", names=[f"compose_full_sum_{len(phases)}"], reads=[],
                           targets=[], writes=[f"compose_full_sum_{len(phases)}"]))
        text.append(_counter_phase(len(phases) - 1, 1000 + 64 * k))
    for k in range(bias):
        p = len(phases)
        arg = 672 + 3 * k
        phases.append(dict(kind="elementwise", names=[f"add_{p}"], reads=[f"arg{arg}_1", "arg667_1", "arg663_1"],
                           writes=[f"add_{p}", f"div_{p}", f"sqrt_{p}"], targets=[]))
        text.append(_bias_phase(p, arg, 5000 + 704 * k, 90000 + 704 * ((k * 7) % bias)))
    for k in range(decay):
        p = len(phases)
        phases.append(dict(kind="elementwise", names=[f"mul_{p}"], reads=["arg663_1", "arg664_1"],
                           writes=[f"mul_{p}", f"rsub_{p}"], targets=[]))
        text.append(_decay_phase(p, 200000 + 704 * k))
    if step_reads is not None:            # a dependent member: reads another member's output
        phases[step_reads[0]]["reads"].append(step_reads[1])
    views = [dict(name=f"add_{p}", source=phases[p]["reads"][0], target="in_place")
             for p in range(len(phases)) if phases[p]["names"][0].startswith("add_")]
    body = "".join(text)
    source = (_BIAS + _DECAY + 'extern "C" __global__ void latch_ops(float** args, float* arena, '
              "unsigned long long* ticks) {\n{\n" + body + "grid.sync();\n}\n}\n")
    manifest = dict(barrier_schedule="dependency", phases=phases, barrier_intervals=[list(range(len(phases)))],
                    views=views, source_hash="x", header_hashes={})
    return GeneratedProgram(source, manifest)


def test_runs_become_table_phases():
    program = prologue_batch.batch(_program())
    record = program.manifest["prologue_batch"]
    # Adjacent bias-correction and decay-factor calls form one phase with one loop per helper.
    assert [r["rule"] for r in record["runs"]] == ["counter_table", "scalar_table"]
    assert [g["helper"] for g in record["runs"][1]["groups"]] == ["latch_ec_body8", "latch_ec_body9"]
    assert record["batched_phases"] == 12 + 10 + 9 and record["refused"] == []
    source = program.source
    assert source.count("// Phase ") == 2
    assert source.count("__syncthreads();") == 2          # one barrier per table phase (was one per phase)
    # Counter table: arena float offsets, one store per counter.
    table = re.search(r"static const long long latch_ptab_0_0\[12\] = \{([^}]*)\};", source)
    assert [int(v[:-2]) for v in table[1].split(", ")] == [1000 + 64 * k for k in range(12)]
    # Bias corrections: step column shared by the in-place output; common lr/log-beta2 stay literal.
    bias = re.search(r"static const long long latch_ptab_12_0\[30\] = \{([^}]*)\};", source)
    rows = [int(v[:-2]) for v in bias[1].split(", ")]
    assert rows[:3] == [2 * 672 + 1, 2 * 5000, 2 * 90000]
    assert "latch_ec_body8(0u, 1u, 0u, 1u, " in source
    assert "reinterpret_cast<float*>(latch_pa[667])" in source and "reinterpret_cast<float*>(latch_pa[663])" in source
    # Decay factors: only the output pointer varies.
    assert re.search(r"static const long long latch_ptab_12_1\[9\] = ", source)
    assert source.count("latch_ec_body9(0u, 1u, 0u, 1u, reinterpret_cast<float*>(latch_pa[663]), "
                        "reinterpret_cast<float*>(latch_pa[664]), ") == 1


def test_dependent_member_refuses_its_run():
    program = prologue_batch.batch(_program(step_reads=(14, "div_13")))
    record = program.manifest["prologue_batch"]
    assert [r["rule"] for r in record["runs"]] == ["counter_table"]
    assert record["refused"] == [dict(phases=[12, 30], reason="members not independent")]
    assert "latch_ec_body8(lsr_bx, lsr_nt, lsr_tx, lsr_nb" in program.source


def test_short_runs_are_unchanged():
    original = _program(bias=3, decay=2, counters=4)
    assert prologue_batch.batch(original) is original


def test_queue_keeps_table_phases_on_the_full_grid():
    with update_schedule.scope("streamed_prep"):
        program = update_schedule.rotate(prologue_batch.batch(_program()))
    assert program.manifest["update_schedule"]["queued_runs"] == 0
    assert "update_schedule queue" not in program.source


def test_schedule_value_is_validated():
    assert update_schedule.validate("streamed_prep") == "streamed_prep"
    with pytest.raises(ValueError):
        update_schedule.validate("prep")
