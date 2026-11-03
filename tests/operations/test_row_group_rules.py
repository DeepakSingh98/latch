"""Row-group issue rules of accumulation_rules ('rilp', 'rinl', 'rleaf', 'rhoist2', 'rsum1'): structural and
host-executed semantic tests of the exact memory-traffic rules (CPU, no CUDA).

The semantic tests compile the generated reduction bodies as host C++ (each CTA/thread of a simulated cooperative grid
runs in turn; the bodies use no inter-thread communication) and require bitwise-equal results to the unbatched body.
"""
import random
import shutil
import struct
import subprocess

import pytest

from megakernel_lab.operation_compiler import accumulation_rules as ar
from megakernel_lab.operation_compiler import group_rules as gr
from megakernel_lab.operation_compiler.codegen_rules import batched_results, exact_reduction


def test_rilp_rule_selection():
    assert "rilp" in ar.rules_of("update+fsload+fsspread+rilp")
    with ar.scope("update+rilp"):
        assert "+rilp" in ar.features()
        assert "rilp" in gr.features_of("base" + ar.features())
    assert ar.features() == ""


def test_batched_results_policy():
    threads = 132 * 256
    assert batched_results(1, 62914560) == 8 and batched_results(3, 7864320) == 8
    assert batched_results(6, 7864320) == 4 and batched_results(12, 7864320) == 2
    assert batched_results(17, 7864320) == 1 and batched_results(0, 7864320) == 1
    assert batched_results(6, threads) == 1 and batched_results(6, 2 * threads) == 2


def _position(extent, rows, width):
    # The middle-axis form lower.py renders for a [rows, extent, width] -> [rows, width] reduction.
    return (f"(long long)((unsigned)(r) % {extent}u) * {width}LL + (long long)((unsigned)(o) / {width}u % {rows}u) * "
            f"{extent * width}LL + (long long)((unsigned)(o) % {width}u) * 1LL")


def _render(extent, rows, width, scale, batch, vector=False):
    results = rows * width
    code, extra = exact_reduction(out="out", position=_position(extent, rows, width), extent=extent, results=results,
                                  read=lambda p: f"x[{p}]", rounded=lambda e: e, scale=scale, batch=batch,
                                  vector_read=(lambda p: f"*reinterpret_cast<const float4*>(x + ({p}))") if vector else None)
    return code, extra


def test_rilp_body_structure():
    plain, extra = _render(6, 64, 32, " / 6.0f", 1)
    batched, bextra = _render(6, 64, 32, " / 6.0f", 4)
    assert extra == {"engine": "exact_cta_tree_per_thread", "reduction_elements": 6}
    assert bextra["engine"] == "exact_cta_tree_per_thread" and bextra["batched_results"] == 4
    # The tree text of one result is unchanged; all leaf loads of a round precede its stores.
    tree = plain.split("const float acc = ")[1].split(";\n")[0]
    assert tree in batched
    assert batched.index("acc_[u_] = ") < batched.index("out[o] = ")
    assert "4LL * (long long)gridDim.x * blockDim.x" in batched
    assert batched.count("blockIdx.x") == 1 and batched.count("threadIdx.x") == 1


HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
struct D { unsigned x; };
struct float4 { float x, y, z, w; };
static float4 make_float4(float a, float b, float c, float d) { return float4{a, b, c, d}; }
static void plain(D blockIdx, D threadIdx, D gridDim, D blockDim, const float* x, float* out) { PLAIN }
static void batched(D blockIdx, D threadIdx, D gridDim, D blockDim, const float* x, float* out) { BATCHED }
int main(int argc, char** argv) {
  const long long n = NIN, m = NOUT; const unsigned grid = GRID, block = BLOCK;
  float* x = (float*)malloc(n * sizeof(float)); float* a = (float*)malloc(m * 4); float* b = (float*)malloc(m * 4);
  FILE* f = fopen(argv[1], "rb"); if (fread(x, 4, n, f) != (size_t)n) return 2; fclose(f);
  memset(a, 0x7f, m * 4); memset(b, 0x3f, m * 4);
  for (unsigned bx = 0; bx < grid; ++bx) for (unsigned tx = 0; tx < block; ++tx) {
    plain(D{bx}, D{tx}, D{grid}, D{block}, x, a);
    batched(D{bx}, D{tx}, D{grid}, D{block}, x, b);
  }
  for (long long i = 0; i < m; ++i) if (memcmp(&a[i], &b[i], 4)) { printf("DIFF %lld\n", i); return 1; }
  printf("EQUAL %lld\n", m); return 0;
}
"""


@pytest.mark.skipif(shutil.which("c++") is None, reason="host C++ compiler required")
@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("extent,scale", [(1, ""), (2, ""), (3, " / 3.0f"), (6, " / 6.0f"), (8, ""), (12, "")])
def test_rilp_batched_body_is_bitwise_equal_on_host(tmp_path, extent, scale, vector):
    rows, width = 37, 48           # 1776 results; a 5 x 64 grid leaves a partial final round
    batch = 8 if extent <= 3 else 4 if extent <= 8 else 2
    plain, _ = _render(extent, rows, width, scale, 1)
    batched, extra = _render(extent, rows, width, scale, batch, vector)
    assert extra.get("vector4", False) == vector
    rng = random.Random(extent)
    values = [rng.choice([0.0, -0.0, 1e-30, -3.5, 7.25, 1e30, -1e30]) if rng.random() < 0.2 else rng.uniform(-4, 4)
              for _ in range(rows * extent * width)]
    (tmp_path / "x.bin").write_bytes(struct.pack(f"<{len(values)}f", *values))
    source = (HARNESS.replace("PLAIN", plain.replace("/*RED_EPILOGUE*/", "")).replace("BATCHED", batched)
              .replace("NIN", str(len(values))).replace("NOUT", str(rows * width)).replace("GRID", "5")
              .replace("BLOCK", "64"))
    (tmp_path / "h.cpp").write_text(source)
    build = subprocess.run(["c++", "-std=c++17", "-O1", "-ffp-contract=off", "-Wno-unknown-pragmas", "-o", str(tmp_path / "h"),
                            str(tmp_path / "h.cpp")], capture_output=True, text=True, timeout=50)
    assert build.returncode == 0, build.stderr[-2000:]
    run = subprocess.run([str(tmp_path / "h"), str(tmp_path / "x.bin")], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0 and run.stdout.startswith("EQUAL"), run.stdout + run.stderr


def _value(shape, dtype="float32"):
    stride, strides = 1, []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {"shape": shape, "strides": list(reversed(strides)), "dtype": dtype}


def _size_one_sum_graph(n=4096, d=448):
    # s = sum(a * w, axis of size 1): a weighted pool over a single slot. a and w are also read elsewhere.
    return {"inputs": ["a", "w"], "outputs": ["s", "na", "nw"], "values": {
        "a": _value([n, 1, d]), "w": _value([n, 1, 1]), "p": _value([n, 1, d]), "s": _value([n, d]),
        "na": _value([n, 1, d]), "nw": _value([n, 1, 1])},
        "operations": [
            {"name": "p", "target": "aten.mul.Tensor", "args": [{"ref": "a"}, {"ref": "w"}]},
            {"name": "s", "target": "aten.sum.dim_IntList", "args": [{"ref": "p"}, [1]]},
            {"name": "na", "target": "aten.neg.default", "args": [{"ref": "a"}]},
            {"name": "nw", "target": "aten.neg.default", "args": [{"ref": "w"}]}]}


def _reduction(manifest, name):
    return next(p for p in manifest["phases"] if name in p["writes"])


def test_rinl_and_rilp_lowering_structure():
    from megakernel_lab.operation_compiler.lower import lower
    graph = _size_one_sum_graph()
    plain = lower(graph, execution="compose", compose_selection="common").manifest
    assert "p" in _reduction(plain, "s")["reads"] and "p" not in plain.get("elided_stores", [])
    program = lower(graph, execution="compose", compose_selection="common", group_rules="base+rilp+rinl")
    manifest = program.manifest
    red = _reduction(manifest, "s")
    assert "p" not in red["reads"] and {"a", "w"} <= set(red["reads"])
    assert red.get("batched_results") == 8 and red.get("engine") == "exact_cta_tree_per_thread"
    assert "p" in manifest.get("elided_stores", [])
    assert {"reduction": "s", "source": "p"} in manifest["group_rules"]["inlined_reduction_sources"]
    assert manifest["group_rules"]["batched_reductions"] == [
        {"result": "s", "batched_results": 8, "reduction_elements": 1, "restored": False}]
    assert "acc_[u_]" in program.source


def test_rilp_restores_the_unbatched_body_when_the_result_is_committed_in_place():
    from megakernel_lab.operation_compiler.lower import lower
    n, d = 4096, 448
    graph = {"inputs": ["x", "t"], "outputs": ["loss"], "state_writes": {"t": "s"}, "values": {
        "x": _value([n, 2, d]), "t": _value([n, d]), "s": _value([n, d]), "loss": _value([])},
        "operations": [{"name": "s", "target": "aten.sum.dim_IntList", "args": [{"ref": "x"}, [1]]},
                       {"name": "loss", "target": "aten.sum.default", "args": [{"ref": "x"}]}]}
    manifest = lower(graph, execution="compose", compose_selection="common", group_rules="base+rilp").manifest
    record = manifest["group_rules"]["batched_reductions"][0]
    assert record["result"] == "s" and record["restored"] is True
    assert "batched_results" not in _reduction(manifest, "s")


def _stream_row_graph(rows=16384, width=448, streams=8):
    # m = mean(a0 * a0, -1) with eight stored same-row products y_k = a_k * a0: 8 reads x 14 elements per lane
    # exceed rhoist's 96 registers and fit rhoist2's 128.
    names = [f"a{k}" for k in range(streams)]
    values = {n: _value([rows, width]) for n in names}
    ops = []
    for k in range(streams):
        values[f"y{k}"] = _value([rows, width])
        ops.append({"name": f"y{k}", "target": "aten.mul.Tensor", "args": [{"ref": f"a{k}"}, {"ref": "a0"}]})
    values["sq"] = _value([rows, width]); values["m"] = _value([rows, 1])
    ops += [{"name": "sq", "target": "aten.mul.Tensor", "args": [{"ref": "a0"}, {"ref": "a0"}]},
            {"name": "m", "target": "aten.mean.dim", "args": [{"ref": "sq"}, [-1], True]}]
    return {"inputs": names, "outputs": ["m"] + [f"y{k}" for k in range(streams)], "values": values, "operations": ops}


def _chained_dot_graph(rows=16384, width=448, k=4):
    # t = x * rsqrt(mean(x * x) + eps); d_j = sum(t * s_j, -1): chained row stages whose leaves t * s_j share t.
    v = {"x": _value([rows, width]), "sq": _value([rows, width]), "m": _value([rows, 1]), "e": _value([rows, 1]),
         "r": _value([rows, 1]), "t": _value([rows, width])}
    ops = [{"name": "sq", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, {"ref": "x"}]},
           {"name": "m", "target": "aten.mean.dim", "args": [{"ref": "sq"}, [-1], True]},
           {"name": "e", "target": "aten.add.Scalar", "args": [{"ref": "m"}, 1e-5]},
           {"name": "r", "target": "aten.rsqrt.default", "args": [{"ref": "e"}]},
           {"name": "t", "target": "aten.mul.Tensor", "args": [{"ref": "x"}, {"ref": "r"}]}]
    for j in range(k):
        v[f"s{j}"] = _value([rows, width]); v[f"p{j}"] = _value([rows, width]); v[f"d{j}"] = _value([rows, 1])
        ops += [{"name": f"p{j}", "target": "aten.mul.Tensor", "args": [{"ref": "t"}, {"ref": f"s{j}"}]},
                {"name": f"d{j}", "target": "aten.sum.dim_IntList", "args": [{"ref": f"p{j}"}, [-1], True]}]
    return {"inputs": ["x"] + [f"s{j}" for j in range(k)], "outputs": [f"d{j}" for j in range(k)],
            "values": v, "operations": ops}


@pytest.mark.parametrize("again", [False, True])
def test_rleaf_recomputes_shared_operand_stage_leaves(again):
    import re
    from megakernel_lab.operation_compiler.lower import lower
    rules = "base+wide+rhoist+rsink+rpipe" + ("+rleaf" if again else "")
    program = lower(_chained_dot_graph(), execution="compose", compose_selection="common", group_rules=rules,
                    row_fusion=True)
    record = next(r for r in program.manifest["row_fusions"] if r["reduction"] == "m")
    assert len(record.get("chained_stages", [])) == 4
    kept = set(re.findall(r"float ck_(\w+)\[", program.source))
    if again:
        assert kept == {"t"}
        # re-evaluated from r_t and s_j (a global load, or the group's hoisted register of the same load)
        assert all(f"const float r_p{j} = (r_t * " in program.source for j in range(4))
    else:
        assert {f"p{j}" for j in range(4)} <= kept


def _chained_square_graph(rows=16384, width=448, k=3):
    # u_j = t * s_j are stored; q_j = u_j * u_j are used only as the leaves of d_j = sum(q_j, -1).
    g = _chained_dot_graph(rows, width, k)
    v, ops = g["values"], [op for op in g["operations"] if not op["name"].startswith("d")]
    for j in range(k):
        v[f"q{j}"] = _value([rows, width])
        ops += [{"name": f"q{j}", "target": "aten.mul.Tensor", "args": [{"ref": f"p{j}"}, {"ref": f"p{j}"}]},
                {"name": f"d{j}", "target": "aten.sum.dim_IntList", "args": [{"ref": f"q{j}"}, [-1], True]}]
    g["operations"] = ops
    g["outputs"] = [f"d{j}" for j in range(k)] + [f"p{j}" for j in range(k)]
    return g


def test_rleaf_sums_leaf_only_values_into_tree_partials():
    import re
    from megakernel_lab.operation_compiler.lower import lower
    base = "base+wide+rhoist+rsink+rpipe"
    plain = lower(_chained_square_graph(), execution="compose", compose_selection="common", group_rules=base,
                  row_fusion=True).source
    lean = lower(_chained_square_graph(), execution="compose", compose_selection="common", group_rules=base + "+rleaf",
                 row_fusion=True).source
    assert {f"q{j}" for j in range(3)} <= set(re.findall(r"float ck_(\w+)\[", plain))
    assert not set(re.findall(r"float ck_(q\d)\[", lean))
    for j in range(3):
        assert f"float cp_q{j}[8] = {{}};" in lean and f"(leaf partials of r_q{j})" in lean
        assert f"else cp_q{j}[w & 7] = cp_q{j}[w & 7] + r_q{j};" in lean


@pytest.mark.parametrize("wide", [False, True])
def test_rhoist2_hoists_wider_groups_without_pipelining_them(wide):
    from megakernel_lab.operation_compiler.lower import lower
    rules = "base+wide+rhoist+rsink+rpipe" + ("+rhoist2" if wide else "")
    program = lower(_stream_row_graph(), execution="compose", compose_selection="common", group_rules=rules,
                    row_fusion=True)
    record = next(r for r in program.manifest["row_fusions"] if r["reduction"] == "m")
    assert not record.get("next_row_pipelined")
    assert bool(record.get("wide_hoist")) == wide
    assert bool(record.get("hoisted_reads")) == wide


def test_rsum1_evaluates_a_size_one_sum_at_its_readers():
    from megakernel_lab.operation_compiler.lower import lower
    n, d = 4096, 448
    graph = _size_one_sum_graph(n, d)
    graph["values"]["b"] = _value([n, d]); graph["values"]["y"] = _value([n, d])
    graph["inputs"].append("b")
    graph["operations"].append({"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": "s"}, {"ref": "b"}]})
    graph["outputs"] = ["y", "na", "nw"]
    rules = "base+vec4+rilp+rinl"
    plain = lower(graph, execution="compose", compose_selection="common", group_rules=rules).manifest
    assert any("s" in p["writes"] for p in plain["phases"])
    program = lower(graph, execution="compose", compose_selection="common", group_rules=rules + "+rsum1")
    m = program.manifest
    record = m["group_rules"]["size_one_sums_at_readers"]
    assert record[0]["value"] == "s"
    if "refused" in record[0]:
        # The guard admits only readers that index the value at their own element (identity text); this broadcast
        # reader renders another index form, so the sum stays stored, unchanged.
        assert record[0]["refused"] == "non-identity read" and any("s" in p["writes"] for p in m["phases"])
    else:
        assert not any("s" in p["writes"] for p in m["phases"]) and "s" in m["elided_stores"]
        reader = next(p for p in m["phases"] if "y" in p["writes"])
        assert {"a", "w", "b"} <= set(reader["reads"]) and "s" not in reader["reads"]
        assert "v_s[" not in program.source
    assert m["phase_count"] == len(m["phases"]) + 1
    # a returned sum is kept
    graph["outputs"] = ["y", "s", "na", "nw"]
    kept = lower(graph, execution="compose", compose_selection="common", group_rules=rules + "+rsum1").manifest
    assert any("s" in p["writes"] for p in kept["phases"])
    assert kept["group_rules"]["size_one_sums_at_readers"][0]["refused"] == "kept or non-root result"
