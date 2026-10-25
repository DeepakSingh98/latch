"""LATCH-MOESUP-ROW rewrites and emitters (CPU only: graph rewrites, generated text and a small lowering)."""
from __future__ import annotations

import copy


def _value(name, shape, dtype, alias_of=None):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= max(int(extent), 1)
    return {"name": name, "dtype": dtype, "shape": list(shape), "strides": list(reversed(strides)),
            "role": "temporary", "parameter": None, "alias_of": alias_of, "storage_offset": 0, "producer": name,
            "output_roles": []}


def _op(name, target, args, kwargs=None, view=False):
    refs = []

    def walk(a):
        if isinstance(a, dict) and set(a) == {"ref"}:
            refs.append(a["ref"])
        elif isinstance(a, (list, tuple)):
            for x in a:
                walk(x)
    walk(args)
    return {"name": name, "target": target, "args": args, "kwargs": kwargs or {}, "inputs": list(refs),
            "outputs": [name], "dependencies": list(refs), "reads": list(refs), "writes": [] if view else [name],
            "reduction_axes": [], "synchronization": "view" if view else "global_phase_boundary", "effect": "pure"}


def _topk_graph(rounds=3, width=8):
    r = lambda n: {"ref": n}
    values = {"x": _value("x", [4, 16, width], "float32"), "ar": _value("ar", [width], "int64")}
    ops = [_op("ar", "aten.arange.default", [width])]
    cur, outs = "x", []
    for k in range(rounds):
        a = f"a{k}"
        values[a] = _value(a, [4, 16], "int64")
        ops.append(_op(a, "aten.argmax.default", [r(cur), -1]))
        u = f"u{k}"
        values[u] = _value(u, [4, 16, 1], "int64", alias_of=a)
        ops.append(_op(u, "aten.unsqueeze.default", [r(a), -1], view=True))
        g = f"g{k}"
        values[g] = _value(g, [4, 16, 1], "float32")
        ops.append(_op(g, "aten.gather.default", [r("x"), -1, r(u)]))
        outs.append(g)
        if k == rounds - 1:
            break
        e, s, w = f"e{k}", f"s{k}", f"w{k}"
        ue = f"ue{k}"
        values[ue] = _value(ue, [4, 16, 1], "int64", alias_of=a)
        ops.append(_op(ue, "aten.unsqueeze.default", [r(a), -1], view=True))
        values[e] = _value(e, [4, 16, width], "bool")
        ops.append(_op(e, "aten.eq.Tensor", [r("ar"), r(ue)]))
        values[s] = _value(s, [], "float32")
        ops.append(_op(s, "aten.scalar_tensor.default", [-1.0], {"dtype": "float32"}))
        values[w] = _value(w, [4, 16, width], "float32")
        ops.append(_op(w, "aten.where.self", [r(e), r(s), r(cur)]))
        cur = w
    return {"operations": ops, "values": values, "inputs": ["x"], "outputs": [{"ref": g} for g in outs]}


def test_topk_rounds_rewrite():
    from megakernel_lab.operation_compiler.topk_rounds import TARGET, rewrite
    graph = _topk_graph()
    out, report = rewrite(copy.deepcopy(graph))
    assert report and report["rewritten"][0]["rounds"] == 3
    ops = {o["name"]: o for o in out["operations"]}
    t = report["rewritten"][0]["indices"]
    assert ops[t]["target"] == TARGET and out["values"][t]["shape"] == [3, 4, 16]
    for k in range(3):
        a = ops[f"a{k}"]
        assert a["target"] == "aten.select.int" and a["args"] == [{"ref": t}, 0, k] and a["synchronization"] == "view"
        assert out["values"][f"a{k}"]["storage_offset"] == k * 64
    assert not any(n in ops for n in ("e0", "e1", "w0", "w1"))
    position = {o["name"]: i for i, o in enumerate(out["operations"])}
    for o in out["operations"]:
        for r in o["reads"]:
            assert r not in position or position[r] < position[o["name"]], (o["name"], r)


def test_topk_rounds_refuses_outside_reader():
    from megakernel_lab.operation_compiler.topk_rounds import rewrite
    graph = _topk_graph()
    graph["outputs"].append({"ref": "w0"})     # the masked row is observable: the chain must stay unchanged
    out, report = rewrite(copy.deepcopy(graph))
    assert report is None and out == graph


def test_level_split_moves_early_lines():
    from megakernel_lab.operation_compiler.group_rules import split_components
    names = ["div_6", "add_30", "mul_36", "div_7", "mul_37", "mul_38", "div_8"]
    body = ["const float r_div_6 = (v_sum_9[0] / v_cnt[0]); v_div_6[i] = r_div_6;",
            "const float r_add_30 = (v_w[0] + (1.0f * 1.0f)); v_add_30[i] = r_add_30;",
            "const float r_mul_36 = (r_add_30 * 2.0f); v_mul_36[i] = r_mul_36;",
            "const float r_div_7 = (v_w[0] / r_mul_36); v_div_7[i] = r_div_7;",
            "const float r_mul_37 = (r_div_6 * r_div_7); v_mul_37[i] = r_mul_37;",
            "const float r_mul_38 = (1.0f * r_div_7); v_mul_38[i] = r_mul_38;",
            "const float r_div_8 = (r_mul_38 / v_cnt[0]); v_div_8[i] = r_div_8;"]
    phases = [{"kind": "x", "names": ["c"], "reads": [], "writes": ["cnt"]},
              {"kind": "x", "names": ["s"], "reads": ["cnt"], "writes": ["sum_9"]},
              {"kind": "elementwise", "names": names, "targets": ["t"] * 7, "reads": ["sum_9", "cnt", "w"],
               "writes": names, "body": body, "shape": [], "registers": {n: "r_" + n for n in names}}]
    out, records = split_components(phases, {n: "float32" for n in names}, ctypes={}, level_split=True)
    pieces = [p["names"] for p in out[2:]]
    assert pieces == [["add_30", "mul_36", "div_7", "mul_38"], ["div_8"], ["div_6", "mul_37"]]
    late = out[4]
    assert "v_div_7[i]" in late["body"][1] and "div_7" in late["reads"]
    assert any("level_split" in r for r in records)


def test_general_exact_division():
    from megakernel_lab.operation_compiler.p5c_epilogue import exact_division
    silu = ("const float r_s = (__bfloat162float(q.x) / (1.0f + expf(-__bfloat162float(q.x))));")
    assert exact_division(silu) == ("const float r_s = latch_xdiv(__bfloat162float(q.x), "
                                    "(1.0f + expf(-__bfloat162float(q.x))), xok_);")
    assert exact_division("const long long z = ((unsigned)(i) / (384u));") == "const long long z = ((unsigned)(i) / (384u));"
    assert exact_division("const float r_y = ((r_a + r_b) / (r_c));") == "const float r_y = ((r_a + r_b) / (r_c));"


def test_level_split_remerge_restores_materialized_register_reads():
    from megakernel_lab.operation_compiler.group_rules import merge_elementwise_chains, split_components
    from megakernel_lab.operation_compiler.lower import TensorLayout, _ident, _index

    # A bool predicate breaks the initial float register chain. Splitting
    # clamp from where then introduces v_clamp[i], even for a scalar [0]
    # layout. Rejoining all three chains must consume the clamp register.
    for shape in ((), (4,)):
        strides = () if not shape else (1,)
        names = ('x', 'coef', 'nan', 'clamp', 'where')
        layouts = {n: TensorLayout(shape, strides, n) for n in names}
        dtypes = {n: 'bool' if n == 'nan' else 'float32' for n in names}
        at = _index(layouts['x'], shape)

        def phase(outputs, reads, body):
            return dict(kind='elementwise', names=outputs, targets=['t'] * len(outputs),
                        reads=reads, writes=outputs, body=body, shape=shape,
                        dtype=dtypes[outputs[0]], registers={n: 'r_' + n for n in outputs})

        phases = [phase(['coef'], ['x'], [
            f'const float r_coef = v_x[{at}]; v_coef[i] = r_coef;']),
            phase(['nan'], ['coef'], [
                f'const bool r_nan = (v_coef[{at}] != v_coef[{at}]); v_nan[i] = r_nan;']),
            phase(['clamp', 'where'], ['coef', 'nan', 'clamp'], [
                f'const float r_clamp = latch_clamp_max(v_coef[{at}], 1.0f); v_clamp[i] = r_clamp;',
                f'const float r_where = (v_nan[{at}] ? v_coef[{at}] : r_clamp); v_where[i] = r_where;'])]
        split, records = split_components(phases, dtypes, ctypes={}, layouts=layouts, level_split=True)
        assert any('level_split' in r for r in records)
        assert any(': v_clamp[i]' in line for p in split for line in p.get('body', []))
        merged, records = merge_elementwise_chains(split, layouts, ident=_ident, index=_index, inplace_roots=set())
        assert records and len(merged) == 1
        assert merged[0]['body'][-1] == 'const float r_where = (r_nan ? r_coef : r_clamp); v_where[i] = r_where;'
        assert 'clamp' not in merged[0]['reads']


def test_elementwise_remerge_refuses_broadcast_transpose_and_offset_views():
    from megakernel_lab.operation_compiler.group_rules import merge_elementwise_chains
    from megakernel_lab.operation_compiler.lower import TensorLayout, _ident, _index

    shape = (2, 2)
    for view_shape, strides, offset in (((2, 2), (1, 2), 0),
                                         ((2, 1), (2, 1), 0),
                                         ((2, 1), (2, 1), 1)):
        layouts = {n: TensorLayout(shape, (2, 1), n) for n in ('x', 'p', 'out')}
        layouts['view'] = TensorLayout(view_shape, strides, 'p', offset)
        phases = [dict(kind='elementwise', names=['p'], targets=['t'], reads=['x'], writes=['p'],
                       shape=shape, body=['const float r_p = v_x[i]; v_p[i] = r_p;']),
                  dict(kind='elementwise', names=['out'], targets=['t'], reads=['view'], writes=['out'],
                       shape=shape, body=[f'const float r_out = v_view[{_index(layouts["view"], shape)}]; v_out[i] = r_out;'])]
        original = copy.deepcopy(phases)
        merged, records = merge_elementwise_chains(phases, layouts, ident=_ident, index=_index, inplace_roots=set())
        assert not records and merged == original


def test_widened_segmented_merge_rewrite():
    from megakernel_lab.operation_compiler.scaled_merge import rewrite
    r = lambda n: {"ref": n}
    values = {"b": _value("b", [4, 8, 16], "bfloat16"), "c": _value("c", [4, 8, 16], "float32"),
              "g": _value("g", [32, 16], "float32", alias_of="c"), "idx": _value("idx", [32], "int64"),
              "e": _value("e", [10, 16], "float32")}
    ops = [_op("c", "aten._to_copy.default", [r("b")], {"dtype": "float32"}),
           _op("g", "aten.view.default", [r("c"), [32, 16]], view=True),
           _op("e", "aten.embedding_dense_backward.default", [r("g"), r("idx"), 10, -1, False])]
    graph = {"operations": ops, "values": values, "inputs": ["b", "idx"], "outputs": [r("e")]}
    out, report = rewrite(copy.deepcopy(graph))
    assert report and report["rewritten"][0]["widened"] == "c"
    names = {o["name"]: o for o in out["operations"]}
    assert "c" not in names and names["e"]["kwargs"].get("latch_row_widen") is True
    assert names["g"]["args"][0] == {"ref": "b"} and out["values"]["g"]["dtype"] == "bfloat16"


def test_pair_restrict_partition():
    from megakernel_lab.operation_compiler.group_rules import pair_restrict
    code = pair_restrict("for (long long row = blockIdx.x; row < 8; row += gridDim.x) {}", 700)
    assert "pair_0_ = (unsigned)((unsigned long long)gridDim.x * 700ULL / 1000ULL)" in code
    assert "row = pair_b_" in code and "row += pair_g_" in code


def test_row_index_reuse_by_row():
    from megakernel_lab.operation_compiler.row_fusion import _row_index_reuse
    code = "\n".join([
        "  const long long i = on_ * 384LL + 32 * w + lane;",
        "  p0_[w] = latch_row_gather(v_t, v_ix, i, 65536LL, 384LL);",
        "  p1_[w] = v_s[(long long)((unsigned)(i) / 384u % 524288u) * 1LL];",
        "  const long long i = k + 5;",
        "  q = latch_row_gather(v_t, v_ix, i, 65536LL, 384LL);"])
    out = _row_index_reuse(code).split("\n")
    assert out[0].endswith("const long long i_row = on_;")
    assert out[1] == "  p0_[w] = latch_row_gather_rc(v_t, v_ix, i_row, i, 65536LL, 384LL);"
    assert out[2] == "  p1_[w] = v_s[(long long)((unsigned)(i_row) % 524288u) * 1LL];"
    assert out[4] == "  q = latch_row_gather(v_t, v_ix, i, 65536LL, 384LL);"   # other index forms untouched


def test_chain_stage_replays_tree_and_keeps_values():
    from megakernel_lab.operation_compiler.row_fusion import _chain_stages
    phases = {10: {"names": ["s1"], "row_reduction": {"source": "a", "scale": ""}}}
    code = ["for (o) {", "  float leaf_[12];", "/*CKDECL*/", "  for (w) {", "    const float r_x = h0_[w];",
            "/*CKSTORE0*/", "  }", "  for (w) {", "    const float r_a = (r_x * 2.0f);", "/*CKSTORE1*/", "  }"]
    reduce_lines = ["  for (int w = 0; w < 8; ++w) { v[w] = 0.0f + leaf_[w]; if (w + 8 < 12) v[w] = v[w] + leaf_[w + 8 < 12 ? w + 8 : 0]; }"]
    body = lambda e, ctx: ["const float r_y = (r_x * r_s1); v_y[i] = r_y;"]
    out = _chain_stages(code, [(10, [], [20])], phases, body, {"y"}, lambda n: "v_" + n, lambda n: n, 12, 384,
                        reduce_lines, [["const float r_x = h0_[w];"], ["const float r_a = (r_x * 2.0f);"]])
    text = "\n".join(out)
    assert "float ck_a[12];" in text and "float ck_x[12];" in text
    assert "ck_a[w] = r_a;" in text and "ck_x[w] = r_x;" in text
    assert "v_c0[w] = 0.0f + leaf_[w]; if (w + 8 < 12) v_c0[w] = v_c0[w] + leaf_[w + 8 < 12 ? w + 8 : 0];" in text
    assert "const float r_s1 = second_c0;" in text and "if (lane == 0) v_s1[o]" not in text
    assert text.count("{") == text.count("}") + 1   # the row loop itself stays open


def test_split_stream_rows():
    from megakernel_lab.operation_compiler.group_rules import split_stream_rows
    phases = [{"kind": "attention_long_forward", "names": ["att"]}, {"kind": "vocab_gradient", "names": ["g"]},
              {"kind": "elementwise", "names": ["e"]}, {"kind": "vocab_gradient", "names": ["g2"]}]
    out, records = split_stream_rows(phases, [[0, 1, 2], [3]])
    assert out == [[0, 2], [1], [3]] and records == [{"rows": ["g"], "attention": ["att"]}]
