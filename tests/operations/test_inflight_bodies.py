"""More loads in flight in memory-bound bodies (operation_compiler/inflight_bodies.py): elementwise_rules 'permrows',
'embunroll' and 'rowcopy', accumulation_rules 'rnarrow' and 'rmid'. Source-level checks (no GPU)."""
import copy
import re
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest

from megakernel_lab.operation_compiler import accumulation_rules, elementwise_rules, row_chain
from megakernel_lab.operation_compiler import inflight_bodies as ib


@contextmanager
def rules(elementwise=None, accumulation=None):
    with ExitStack() as stack:
        stack.enter_context(elementwise_rules.scope(elementwise))
        stack.enter_context(accumulation_rules.scope(accumulation))
        yield


def canonical(K=3, units=16, rows=64, n=192, add="latch_pgrad::add4(acc_, v_[s_]);"):
    """A permutation embedding-gradient body in the canonical one-index-per-thread form."""
    total = rows * units
    return (f"for (long long u_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; u_ < {total}LL;\n"
            f"     u_ += (long long)gridDim.x * blockDim.x) {{\n"
            f"  const long long row = u_ / {units}LL, col = u_ % {units}LL;\n"
            f"  float4 acc_ = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
            f"  if (row != -1LL) {{\n"
            f"    long long q_[{K}];\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < {K}; ++s_) q_[s_] = v_inv[row * {K}LL + s_];\n"
            f"    float4 v_[{K}];\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < {K}; ++s_) {{\n"
            f"      if (q_[s_] < 0 || q_[s_] >= {n}LL || (col == 0 && v_ord[q_[s_]] != row)) __trap();\n"
            f"      v_[s_] = reinterpret_cast<const float4*>(v_g + q_[s_] * {4 * units}LL)[col];\n"
            f"    }}\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < {K}; ++s_) {add}\n"
            f"  }}\n"
            f"  reinterpret_cast<float4*>(v_out + row * {4 * units}LL)[col] = acc_;\n"
            f"}}")


def gather(units=8, rows=64, n=64):
    """K = 1 BF16 -> BF16 widening permutation gather in the canonical form."""
    total = rows * units
    return (f"for (long long u_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; u_ < {total}LL;\n"
            f"     u_ += (long long)gridDim.x * blockDim.x) {{\n"
            f"  const long long row = u_ / {units}LL, col = u_ % {units}LL;\n"
            f"  float4 acc_ = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
            f"  if (row != -1LL) {{\n"
            f"    long long q_[1];\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < 1; ++s_) q_[s_] = v_inv[row];\n"
            f"    float4 v_[1];\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < 1; ++s_) {{\n"
            f"      if (q_[s_] < 0 || q_[s_] >= {n}LL || (col == 0 && v_ord[q_[s_]] != row)) __trap();\n"
            f"      v_[s_] = latch_pgrad::widen4(reinterpret_cast<const uint2*>(v_x + (q_[s_] * {4 * units}LL))[col]);\n"
            f"    }}\n"
            f"#pragma unroll\n"
            f"    for (int s_ = 0; s_ < 1; ++s_) latch_pgrad::add4(acc_, v_[s_]);\n"
            f"  }}\n"
            f"  reinterpret_cast<uint2*>(v_out + row * {4 * units}LL)[col] = latch_pgrad::round4(acc_);\n"
            f"}}")


def phase(code, engine="permutation_fixed_sum_v1", name="eg"):
    return dict(kind="embedding_gradient", engine=engine, names=[name], writes=[name], code=code)


def bf16_rows(width, offset=0):
    return SimpleNamespace(shape=(64, width), strides=(width, 1), offset=offset)


def test_accessors_follow_the_tokens():
    with rules():
        assert not (ib.rows() or ib.rowcopy() or ib.narrow() or ib.midrow() or ib.emb() or ib.any_enabled())
    with rules("permrows+embunroll+rowcopy", "rnarrow+rmid"):
        assert ib.rows() and ib.rowcopy() and ib.narrow() and ib.midrow() and ib.emb() and ib.any_enabled()
    with rules("embunroll+rowcopy", "rnarrow+rmid"):
        assert not ib.any_enabled()                              # only 'permrows' rewrites finished phase code


def test_off_leaves_phase_code_unchanged():
    phases = [phase(canonical())]
    before = copy.deepcopy(phases)
    with rules("embunroll+rowcopy", "rnarrow"):
        assert ib.rewrite(phases) == {}
    assert phases == before


def test_permrows_warp_rows_form():
    phases = [phase(canonical()), dict(kind="elementwise", names=["z"], code="z")]
    with rules("permrows"):
        record = ib.rewrite(phases)
    assert record == {"rows": [["eg"]]}
    p = phases[0]
    assert p["inflight_rules"] == ["rows"] and p["inflight_rows"] == dict(R=2, B=3, J=1, K=3, ops=1, rows=64)
    code = p["code"]
    assert code.startswith("{  // Wide-issue warp rows: 2 rows per warp round, 3 occurrences per load batch")
    assert code.count("__trap()") == 1 and code.rstrip().endswith("if (mlp_bad_) __trap();\n}")
    # every occurrence is added in ascending ordinal with the unchanged per-occurrence add
    assert "latch_pgrad::add4(mlp_acc_[i_][j_], mlp_v_[i_][s_ - mlp_b0_][j_]);" in code
    assert "reinterpret_cast<const float4*>(v_g + mlp_q_[i_][s_] * 64LL)[mlp_c_]" in code
    assert phases[1] == dict(kind="elementwise", names=["z"], code="z")


def test_permrows_batches_follow_the_load_target():
    assert (ib.ROW_LOADS, ib.ROW_BATCH, ib.GATHER_ROWS) == (8, 4, 8)
    phases = [phase(canonical(K=8, units=16))]
    with rules("permrows"):
        ib.rewrite(phases)
    assert phases[0]["inflight_rows"]["B"] == 4 and phases[0]["inflight_rows"]["R"] == 2   # 8 // (4 * 1)


def test_permrows_refusal_keeps_the_body_with_a_deferred_trap():
    phases = [phase(canonical(add="acc_.x += v_[s_].x;"))]
    with rules("permrows"):
        record = ib.rewrite(phases)
    assert record == {"rows_refused": [[["eg"], "unknown add"]], "defer": [["eg"]]}
    code = phases[0]["code"]
    assert phases[0]["inflight_rules"] == ["defer"]
    assert code.startswith("{\nbool mlp_bad_ = false;\n") and code.endswith("\nif (mlp_bad_) __trap();\n}")
    assert ("const bool out_ = q_[s_] < 0 || q_[s_] >= 192LL; if (out_) q_[s_] = 0LL; "
            "mlp_bad_ |= out_ || (col == 0 && v_ord[q_[s_]] != row);") in code
    assert code.count("__trap()") == 1
    assert "acc_.x += v_[s_].x;" in code                          # the body itself is unchanged


def test_permrows_bf16_gather_form_needs_aligned_rows():
    aligned = [phase(gather(), engine="permutation_gather_v1")]
    with rules("permrows"):
        ib.rewrite(aligned, {"x": bf16_rows(32), "out": bf16_rows(32)})
    assert aligned[0]["inflight_rows"] == dict(R=8, K=1, gather=True, chunks=4, rows=64)
    assert aligned[0]["code"].startswith("{  // Wide-issue gather rows: warp per 8 rows, 16-byte chunks")
    unaligned = [phase(gather(), engine="permutation_gather_v1")]
    with rules("permrows"):
        record = ib.rewrite(unaligned, {"x": bf16_rows(32, offset=4), "out": bf16_rows(32)})
    assert record == {"rows": [["eg"]]}                          # the general warp-rows form instead
    assert "gather" not in unaligned[0]["inflight_rows"]
    assert unaligned[0]["code"].startswith("{  // Wide-issue warp rows: ")


@pytest.mark.parametrize("width", [1, 3, 5, 8, 16])
@pytest.mark.parametrize("backward", [False, True])
def test_rnarrow_softmax_groups(width, backward):
    code = ib.narrow_softmax_code(rows=4096, width=width, x="v_x[row * 5LL + k]", g="v_g[row * 5LL + k]",
                                  out="v_o", backward=backward, logged=False, safe=False)
    G = 1 << (width - 1).bit_length()
    assert code.startswith(f"{{  // Wide-issue narrow rows: {32 // G} rows per warp step ({G}-lane groups), "
                           f"{ib.NARROW_UNROLL} steps per iteration")
    offsets = [int(o) for o in re.findall(r"__shfl_xor_sync\(0xffffffffu, total, (\d+)\)", code)]
    expected = []
    o = G // 2
    while o >= 1:
        expected.append(o)
        o //= 2
    assert offsets == expected                                   # the group butterfly G/2 .. 1
    assert ib.PACK_HEAD.search(code) and code.count("blockIdx.x") == 1 and code.count("gridDim.x") == 1


def test_rnarrow_refuses_rows_wider_than_sixteen():
    assert ib.narrow_softmax_code(rows=4096, width=17, x="x", g=None, out="v_o", backward=False, logged=True,
                                  safe=False) is None
    assert ib.narrow_argmax_code(rows=4096, width=17, x="x", out="v_o", better="b") is None
    code = ib.narrow_argmax_code(rows=4096, width=6, x="v_x[row * 6LL + k]", out="v_o", better="latch_better")
    assert code.startswith("{  // Wide-issue narrow argmax: 4 rows per warp step (8-lane groups)")


def test_rnarrow_row_chain_form_restricts_the_packed_loop():
    code = ib.narrow_softmax_code(rows=4096, width=5, x="v_x[row * 5LL + k]", g=None, out="v_o", backward=False,
                                  logged=True, safe=False)
    phase_ = dict(kind="softmax", engine="exact_cta_tree_per_warp_narrow", narrow_rows=True)
    form = row_chain._form(phase_, code)
    assert form == ("warp_pack", 4096)
    restricted = row_chain.restrict(phase_, form, code, 5)
    assert restricted == ib.pack_restrict(code)
    assert "mlp_r0_ = latch_rc_r0 + (long long)(threadIdx.x >> 5) * 32LL" in restricted
    assert "const long long mlp_rhi_ = latch_rc_r1;" in restricted and "gridDim.x" not in restricted
    assert row_chain._form(dict(phase_, narrow_rows=False), code) is None   # not the warp-row form


def test_embunroll_keeps_eight_elements_in_flight():
    assert ib.EMB_UNROLL == 8
    code = ib.scalar_embedding_code(numel=12345, index="v_i[i / 22LL]", table="v_t[row * 22LL + (i % 22LL)]",
                                    out="v_o", bound=777, ctype="float")
    assert code.startswith("{  // Wide-issue scalar embedding: 8 elements per thread in flight")
    assert "int64_t mlp_r_[8]; float mlp_v_[8];" in code and "assert(row >= 0 && row < 777LL);" in code
    # all index loads, then all table loads, then the stores
    assert code.index("mlp_r_[u_] = i < 12345LL ? v_i[i / 22LL] : 0;") < code.index("mlp_v_[u_] = v_t[")
    assert code.index("mlp_v_[u_] = v_t[") < code.index("v_o[i] = mlp_v_[u_];")


def test_rowcopy_warp_per_rows_form():
    assert ib.ROWCOPY_ROWS == 16
    code = ib.rowcopy_code(rows=1031, per_row=48, stride=56, total=1031 * 48, src="v_x", dst="v_y", order="v_o",
                           inverse="v_q", indent="    scalar();")
    assert code.startswith("{  // Wide-issue row copy: warp per 16 rows")
    assert "uint4 x_[16][2];" in code and "pgx_[s_ * 56LL + ch_]" in code
    assert "mlp_bad_ |= pgq_[sl_] != r0_ + mlp_lane_;" in code and code.count("__trap()") == 1
    assert code.rstrip().endswith("} else {\n    scalar();\n  }\n}")   # run-time alignment fallback


def test_rmid_accessor():
    with rules(accumulation="rmid"):
        assert ib.midrow() and not ib.narrow()
    with rules(accumulation="rnarrow"):
        assert ib.narrow() and not ib.midrow()


def _emit_row_softmax(target, width, rows, warp_rows=True):
    import torch
    from megakernel_lab.operation_compiler.lower import TensorLayout
    from megakernel_lab.operation_compiler.transformer_lower import emit_operation
    layouts = {name: TensorLayout((rows, width), (width, 1), name) for name in ("x", "g")}
    dtypes = {"x": "float32", "g": "float32", "out": "float32"}
    if "backward" in target:
        args = ({"ref": "g"}, {"ref": "x"}, -1, torch.float32)
    else:
        args = ({"ref": "x"}, -1, False)
    return emit_operation(target, args, {}, "out", (rows, width), layouts, dtypes, {"x": "x", "g": "g"}, read=None,
                          warp_rows=warp_rows)


@pytest.mark.parametrize("target", ["_softmax.default", "_log_softmax.default", "_softmax_backward_data.default"])
def test_rmid_selects_warp_rows_only_inside_its_gate(target):
    # 'rmid' admits 32 < W <= 256 with W % 32 != 0 and at least 128 rows, and gives those rows the warp-per-row form
    # that wider rows already take; every other shape, and every shape with the token off, keeps its release form.
    for width, rows in ((48, 256), (100, 128), (32, 256), (64, 256), (48, 64)):
        with rules():
            off = _emit_row_softmax(target, width, rows)
            warp_form = _emit_row_softmax(target, width, rows, warp_rows="pops2a")
        with rules(accumulation="rmid"):
            on = _emit_row_softmax(target, width, rows)
            not_warp_rows = _emit_row_softmax(target, width, rows, warp_rows=False)
        with rules():
            assert not_warp_rows == _emit_row_softmax(target, width, rows, warp_rows=False)
        if (width, rows) in ((48, 256), (100, 128)):
            assert on != off and on == warp_form, (target, width, rows)
        else:
            assert on == off, (target, width, rows)
