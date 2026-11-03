"""Depth-read region rules (graph_rules 'depth_reads') on synthetic joint-IR fragments (no torch/CUDA needed)."""
from __future__ import annotations

import copy

import pytest

from megakernel_lab.operation_compiler import depth_read_region as drr
from megakernel_lab.operation_compiler import graph_rules

T, D = 64, 16


def _op(name, target, args, kwargs=None, outputs=None):
    refs = []

    def walk(a):
        if isinstance(a, dict) and set(a) == {"ref"}:
            refs.append(a["ref"])
        elif isinstance(a, (list, tuple)):
            for x in a:
                walk(x)
        elif isinstance(a, dict):
            for x in a.values():
                walk(x)
    walk(args)
    walk(kwargs or {})
    outs = outputs or [name]
    return dict(name=name, target=target, args=args, kwargs=kwargs or {}, inputs=list(dict.fromkeys(refs)),
                reads=list(dict.fromkeys(refs)), outputs=outs, writes=outs, dependencies=list(dict.fromkeys(refs)),
                effect="pure")


def R(x):
    return {"ref": x}


def _graph(sources_per_read, eps=(1e-6,)):
    """Reads over sources (names); each read gets its own cast/stat chain per source, as captured."""
    ops, values = [], {}

    def val(name, shape, dtype="float32"):
        values[name] = dict(name=name, shape=list(shape), dtype=dtype, strides=None, alias_of=None, producer=name)
    for s in sorted({s for r in sources_per_read for s in r}):
        val(s, (T, D))
    inputs = sorted(values)
    k = 0
    for j, srcs in enumerate(sources_per_read):
        q = f"q{j}"
        val(q, (D,))
        inputs.append(q)
        qb = f"qcast{j}"
        ops.append(_op(qb, drr.CAST, [R(q)], {"dtype": "torch.bfloat16"}))
        val(qb, (D,), "bfloat16")
        qt = f"qt{j}"
        ops.append(_op(qt, "aten.unsqueeze.default", [R(qb), 0]))
        val(qt, (1, D), "bfloat16")
        qtt = f"qtt{j}"
        ops.append(_op(qtt, "aten.t.default", [R(qt)]))
        val(qtt, (D, 1), "bfloat16")
        logits = []
        for s in srcs:
            k += 1
            e = eps[j % len(eps)]
            names = dict(sq=f"sq{k}", mean=f"mean{k}", add=f"add{k}", rs=f"rsqrt{k}", cast=f"cast{k}", mm=f"mm{k}",
                         lg=f"logit{k}")
            ops.append(_op(names["sq"], drr.MUL, [R(s), R(s)]))
            val(names["sq"], (T, D))
            ops.append(_op(names["mean"], "aten.mean.dim", [R(names["sq"]), [-1], True]))
            val(names["mean"], (T, 1))
            ops.append(_op(names["add"], "aten.add.Tensor", [R(names["mean"]), e]))
            val(names["add"], (T, 1))
            ops.append(_op(names["rs"], drr.RSQRT, [R(names["add"])]))
            val(names["rs"], (T, 1))
            ops.append(_op(names["cast"], drr.CAST, [R(s)], {"dtype": "torch.bfloat16"}))
            val(names["cast"], (T, D), "bfloat16")
            ops.append(_op(names["mm"], drr.MM, [R(names["cast"]), R(qtt)]))
            val(names["mm"], (T, 1), "bfloat16")
            ops.append(_op(names["lg"], drr.MUL, [R(names["mm"]), R(names["rs"])]))
            val(names["lg"], (T, 1))
            logits.append(names["lg"])
        cat, div, sm = f"cat{j}", f"div{j}", f"softmax{j}"
        ops.append(_op(cat, drr.CAT, [[R(x) for x in logits], -1]))
        val(cat, (T, len(srcs)))
        ops.append(_op(div, drr.DIV, [R(cat), 4.0]))
        val(div, (T, len(srcs)))
        ops.append(_op(sm, "aten._softmax.default", [R(div), -1, False]))
        val(sm, (T, len(srcs)))
        # A backward-like reader of each read's statistic and BF16 copy.
        for kk in range(k - len(srcs) + 1, k + 1):
            ops.append(_op(f"vjp{kk}", drr.MUL, [R(f"rsqrt{kk}"), R(sm)]))
            val(f"vjp{kk}", (T, len(srcs)))
            ops.append(_op(f"qgrad{kk}", drr.MM, [R(f"cast{kk}"), R(qtt)]))
            val(f"qgrad{kk}", (T, 1), "bfloat16")
    outs = [o["outputs"][0] for o in ops if o["name"].startswith(("vjp", "qgrad", "softmax"))]
    return dict(operations=ops, values=values, inputs=inputs, outputs=[R(x) for x in outs], metadata={})


def _mixture_graph(reads_sources, backward_readers=2):
    """Depth reads over shared per-source statistics and BF16 copies, each followed by an RMSNorm (M, r, N = M * r,
    Y = N * w) whose normalized value N is also read by ``backward_readers`` backward-like products."""
    ops, values, inputs, outputs = [], {}, [], []

    def val(name, shape, dtype="float32"):
        values[name] = dict(name=name, shape=list(shape), dtype=dtype, strides=None, alias_of=None, producer=name)

    def add(name, target, args, shape, dtype="float32", kwargs=None):
        ops.append(_op(name, target, args, kwargs))
        val(name, shape, dtype)
    for s in sorted({s for r in reads_sources for s in r}):
        val(s, (T, D))
        inputs.append(s)
        add(f"sq_{s}", drr.MUL, [R(s), R(s)], (T, D))
        add(f"mean_{s}", "aten.mean.dim", [R(f"sq_{s}"), [-1], True], (T, 1))
        add(f"add_{s}", "aten.add.Tensor", [R(f"mean_{s}"), 1e-6], (T, 1))
        add(f"rs_{s}", drr.RSQRT, [R(f"add_{s}")], (T, 1))
        add(f"cast_{s}", drr.CAST, [R(s)], (T, D), "bfloat16", {"dtype": "torch.bfloat16"})
    for name, shape in (("g_all", (8,)), ("wn", (D,)), ("dy", (T, D))):
        val(name, shape)
        inputs.append(name)
    for j, srcs in enumerate(reads_sources):
        n = len(srcs)
        val(f"q{j}", (D,))
        inputs.append(f"q{j}")
        add(f"qc{j}", drr.CAST, [R(f"q{j}")], (D,), "bfloat16", {"dtype": "torch.bfloat16"})
        add(f"qu{j}", "aten.unsqueeze.default", [R(f"qc{j}"), 0], (1, D), "bfloat16")
        add(f"qt{j}", "aten.t.default", [R(f"qu{j}")], (D, 1), "bfloat16")
        for s in srcs:
            add(f"mm{j}_{s}", drr.MM, [R(f"cast_{s}"), R(f"qt{j}")], (T, 1), "bfloat16")
            add(f"lg{j}_{s}", drr.MUL, [R(f"mm{j}_{s}"), R(f"rs_{s}")], (T, 1))
        add(f"cat{j}", drr.CAT, [[R(f"lg{j}_{s}") for s in srcs], -1], (T, n))
        add(f"div{j}", drr.DIV, [R(f"cat{j}"), 4.0], (T, n))
        add(f"sm{j}", "aten._softmax.default", [R(f"div{j}"), -1, False], (T, n))
        add(f"gains{j}", "aten.slice.Tensor", [R("g_all"), 0, 0, n], (n,))
        add(f"w{j}", drr.MUL, [R(f"sm{j}"), R(f"gains{j}")], (T, n))
        add(f"sc{j}", drr.CAT, [[R(f"rs_{s}") for s in srcs], -1], (T, n))
        add(f"al{j}", drr.MUL, [R(f"w{j}"), R(f"sc{j}")], (T, n))
        mix = []
        for i, s in enumerate(srcs):
            sl = f"al{j}"
            if n > 1:
                sl = f"sl{j}_{i}"
                add(sl, "aten.slice.Tensor", [R(f"al{j}"), 1, i, i + 1], (T, 1))
            add(f"fm{j}_{i}", drr.MUL, [R(s), R(sl)], (T, D))
            mix.append(f"fm{j}_{i}")
        cur = mix[0]
        for i in range(1, n):
            add(f"ad{j}_{i}", "aten.add.Tensor", [R(cur), R(mix[i])], (T, D))
            cur = f"ad{j}_{i}"
        add(f"msq{j}", drr.MUL, [R(cur), R(cur)], (T, D))
        add(f"mmean{j}", "aten.mean.dim", [R(f"msq{j}"), [-1], True], (T, 1))
        add(f"madd{j}", "aten.add.Tensor", [R(f"mmean{j}"), 1e-6], (T, 1))
        add(f"mr{j}", drr.RSQRT, [R(f"madd{j}")], (T, 1))
        add(f"N{j}", drr.MUL, [R(cur), R(f"mr{j}")], (T, D))
        add(f"Y{j}", drr.MUL, [R(f"N{j}"), R("wn")], (T, D))
        outputs.append(R(f"Y{j}"))
        for b in range(backward_readers):
            add(f"bw{j}_{b}", drr.MUL, [R("dy"), R(f"N{j}")], (T, D))
            outputs.append(R(f"bw{j}_{b}"))
    return dict(operations=ops, values=values, inputs=inputs, outputs=outputs, metadata={})


def _rewrite(data):
    with graph_rules.scope("depth_reads"):
        return drr.rewrite(data)


def _check_order(new):
    seen = set(new["inputs"])
    for op in new["operations"]:
        assert set(op["reads"]) <= seen, op["name"]
        seen |= set(op["outputs"])


def test_recognizes_reads():
    g = drr._Graph(_graph([["x0"], ["x0", "a1"], ["x0", "a1", "a2"]]))
    reads = drr.depth_reads(g)
    assert [r["sources"] for r in reads] == [1, 2, 3]
    assert [t["source"] for t in reads[2]["terms"]] == ["x0", "a1", "a2"]
    tails = drr._Graph(_mixture_graph([["x0"], ["x0", "a1"]]))
    found = drr.depth_reads(tails)
    assert [drr.norm_tail(tails, r)["normalized"] for r in found] == ["N0", "N1"]


def test_disabled_is_identity():
    data = _graph([["x0"], ["x0"]])
    assert drr.rewrite(data) == (data, None) and drr.rewrite(data)[0] is data
    with graph_rules.scope("cse+castshare"):
        new, report = drr.rewrite(data)
    assert new is data and report is None


def test_remat_clones_normalized_value_per_backward_reader():
    data = _mixture_graph([["x0"], ["x0", "a1"]], backward_readers=2)
    before = copy.deepcopy(data)
    new, report = _rewrite(data)
    assert data == before                                   # the input is never mutated
    assert report["depth_reads"] == 2 and report["tails"] == 2 and report["remat"] == 4
    assert new["metadata"]["depth_read"] is report and report["kinds"] == ["fbatch", "remat"]
    order = [op["name"] for op in new["operations"]]
    by = {op["name"]: op for op in new["operations"]}
    for j, M in ((0, "fm0_0"), (1, "ad1_1")):
        # reader 0 of N is the forward Y = N * w (keeps N); backward readers 1 and 2 read their own clone.
        for k, reader in ((1, f"bw{j}_0"), (2, f"bw{j}_1")):
            clone = f"N{j}__nsr_rm{k}"
            assert order.index(clone) == order.index(reader) - 1
            assert by[clone]["target"] == drr.MUL and by[clone]["args"] == [R(M), R(f"mr{j}")]
            assert by[clone]["outputs"] == by[clone]["writes"] == [clone]
            assert by[reader]["args"] == [R("dy"), R(clone)] and clone in by[reader]["reads"]
            assert new["values"][clone]["producer"] == clone and new["values"][clone]["shape"] == [T, D]
        assert by[f"Y{j}"]["args"][0] == R(f"N{j}")
    _check_order(new)


def test_forward_logit_products_sharing_an_operand_are_stacked():
    new, report = _rewrite(_mixture_graph([["x0"], ["x0", "a1"], ["x0", "a1"]], backward_readers=0))
    assert report["fbatch_groups"] == 2 and report["fbatch_products"] == 5
    by = {op["name"]: op for op in new["operations"]}
    Q, B, P = "mm0_x0__nsr_fq", "mm0_x0__nsr_fb", "mm0_x0__nsr_fp"
    assert by[Q]["target"] == drr.CAT and by[Q]["args"] == [[R("qu0"), R("qu1"), R("qu2")], 0]
    assert by[B]["target"] == "aten.t.default" and new["values"][B]["alias_of"] == Q
    assert by[P]["target"] == drr.MM and by[P]["args"] == [R("cast_x0"), R(B)]
    for r, mm in enumerate(("mm0_x0", "mm1_x0", "mm2_x0")):
        assert by[mm]["target"] == "aten.slice.Tensor" and by[mm]["args"] == [R(P), 1, r, r + 1]
        assert new["values"][mm]["storage_offset"] == r and new["values"][mm]["strides"] == [3, 1]
    assert "mm1_a1__nsr_fp" in by
    _check_order(new)


def test_operation_lists_are_normalized():
    """With the rule on, every operation's inputs/reads/dependencies lose duplicates (order kept), even when no
    depth read is recognized; the report is recorded in the graph metadata."""
    data = _graph([["x0"]])
    data["operations"] = [op for op in data["operations"] if op["target"] != "aten._softmax.default"]
    for op in data["operations"]:
        for key in ("inputs", "reads", "dependencies"):
            op[key] = op[key] + op[key][:1]
    new, report = _rewrite(data)
    assert report["depth_reads"] == 0 and new["metadata"]["depth_read"] is report
    assert data["operations"][0]["reads"] == data["operations"][0]["reads"][:1] * 2   # input untouched
    for op, old in zip(new["operations"], data["operations"]):
        for key in ("inputs", "reads", "dependencies"):
            assert op[key] == list(dict.fromkeys(old[key]))
        assert op["args"] == old["args"] and op["args"] is not old["args"]
    assert [op["name"] for op in new["operations"]] == [op["name"] for op in data["operations"]]


@pytest.mark.parametrize("early_single", [False, True])
def test_fbatch_cse_shared_queries_across_sources_chunks_and_single(early_single):
    """A's second chunk is planned before B's earlier batch; both share q8/q9 after CSE.

    Optionally an unbatched source also reads q8 before either batch. Query prerequisites must be emitted once
    before every reader, regardless of source-group planning order; product operands/column mapping stay exact.
    """
    ops, values, inputs = [], {}, ["A", "B", "single"] + [f"q{i}" for i in range(10)]

    def val(name, shape, dtype="bfloat16"):
        values[name] = dict(name=name, shape=list(shape), dtype=dtype, strides=None, alias_of=None, producer=name)

    for source in ("A", "B", "single"):
        val(source, (T, D))
    for i in range(10):
        val(f"q{i}", (D,), "float32")
    queries = {}

    def query(i):
        if i in queries:
            return
        chain = [_op(f"qc{i}", drr.CAST, [R(f"q{i}")], {"dtype": "torch.bfloat16"}),
                 _op(f"qu{i}", "aten.unsqueeze.default", [R(f"qc{i}"), 0]),
                 _op(f"qt{i}", "aten.t.default", [R(f"qu{i}")])]
        ops.extend(chain)
        queries[i] = chain
        for name, shape in ((f"qc{i}", (D,)), (f"qu{i}", (1, D)), (f"qt{i}", (D, 1))):
            val(name, shape)

    def product(source, i):
        name = f"mm{source}{i}"
        ops.append(_op(name, drr.MM, [R(source), R(f"qt{i}")]))
        val(name, (T, 1))

    query(0)
    product("A", 0)
    if early_single:
        query(8)
        product("single", 8)
    for i in range(1, 10):
        query(i)
        if i in (1, 8, 9):
            product("B", i)
        product("A", i)
    report = {}
    new = drr.forward_batches(ops, values, [{"rows": T}], report, {"inputs": inputs})
    assert report == {"fbatch_groups": 3, "fbatch_products": 13}
    assert len({op["name"] for op in new}) == len(new)
    seen = set(inputs)
    for op in new:
        assert set(op["reads"]) <= seen, op["name"]
        seen.update(op["outputs"])
    by = {op["name"]: op for op in new}
    # Every original query operation survives exactly, including its cast and view semantics.
    for chain in queries.values():
        for op in chain:
            assert by[op["name"]] == op
    originals = {op["name"]: op for op in ops if op["target"] == drr.MM}
    for name, original in originals.items():
        rewritten = by[name]
        if name.startswith("mmsingle"):
            assert rewritten == original
            continue
        assert rewritten["target"] == "aten.slice.Tensor"
        prod = by[rewritten["args"][0]["ref"]]
        assert prod["args"][0] == original["args"][0]
        transposed = by[prod["args"][1]["ref"]]
        cat = by[transposed["args"][0]["ref"]]
        column = rewritten["args"][2]
        original_transpose = by[original["args"][1]["ref"]]
        assert cat["args"][0][column] == original_transpose["args"][0]
        assert values[name]["storage_offset"] == column
        assert values[name]["strides"] == [len(cat["args"][0]), 1]
