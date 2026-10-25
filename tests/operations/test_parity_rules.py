"""LATCH-MOE4-PARITY rule tests (graph-level; no GPU)."""
import copy


def _value(name, shape, dtype):
    strides, step = [], 1
    for extent in reversed(shape):
        strides.append(step)
        step *= max(extent, 1)
    return {"name": name, "dtype": dtype, "shape": list(shape), "strides": list(reversed(strides)), "role": "temporary",
            "parameter": None, "alias_of": None, "storage_offset": 0, "producer": name, "output_roles": []}


def _op(name, target, args, kwargs=None, sync="global_phase_boundary"):
    refs = [a["ref"] for a in args if isinstance(a, dict)]
    return {"name": name, "target": target, "args": args, "kwargs": kwargs or {}, "inputs": refs, "outputs": [name],
            "dependencies": refs, "reads": refs, "writes": [name], "reduction_axes": [], "synchronization": sync,
            "effect": "pure"}


def _keepdim_graph(rounds=3, width=32, reader=True):
    """torch_topk32 spelling: argmax(keepdim=True), eq(arange, index), where(eq, s, masked), gathers of index."""
    r = lambda n: {"ref": n}
    values = {"x": _value("x", [4, 16, width], "float32"), "ar": _value("ar", [width], "int64")}
    ops = [_op("ar", "aten.arange.default", [width], {"dtype": "int64"})]
    cur, outs = "x", []
    for k in range(rounds):
        a = f"a{k}"
        values[a] = _value(a, [4, 16, 1], "int64")
        ops.append(_op(a, "aten.argmax.default", [r(cur), -1, True]))
        if reader:
            g = f"g{k}"
            values[g] = _value(g, [4, 16, 1], "float32")
            ops.append(_op(g, "aten.gather.default", [r("x"), -1, r(a)]))
            outs.append(g)
        if k == rounds - 1:
            outs.append(a)
            break
        e, s, w = f"e{k}", f"s{k}", f"w{k}"
        values[e] = _value(e, [4, 16, width], "bool")
        ops.append(_op(e, "aten.eq.Tensor", [r("ar"), r(a)]))
        values[s] = _value(s, [], "float32")
        ops.append(_op(s, "aten.scalar_tensor.default", [-100.0], {"dtype": "float32"}))
        values[w] = _value(w, [4, 16, width], "float32")
        ops.append(_op(w, "aten.where.self", [r(e), r(s), r(cur)]))
        cur = w
    return {"operations": ops, "values": values, "inputs": ["x"], "outputs": [{"ref": g} for g in outs]}


def test_topk_rounds_keepdim_chain_with_index_readers():
    from megakernel_lab.operation_compiler.topk_rounds import TARGET, rewrite
    graph = _keepdim_graph()
    out, report = rewrite(copy.deepcopy(graph))
    assert report and report["rewritten"][0]["rounds"] == 3 and report["rewritten"][0]["keepdim"]
    ops = {o["name"]: o for o in out["operations"]}
    t = report["rewritten"][0]["indices"]
    assert ops[t]["target"] == TARGET and out["values"][t]["shape"] == [3, 4, 16, 1]
    for k in range(3):
        a = ops[f"a{k}"]
        assert a["target"] == "aten.select.int" and a["args"] == [{"ref": t}, 0, k]
        assert out["values"][f"a{k}"]["storage_offset"] == k * 64
        assert out["values"][f"a{k}"]["shape"] == [4, 16, 1]
        assert ops[f"g{k}"]["target"] == "aten.gather.default"
    assert not any(n in ops for n in ("e0", "e1", "w0", "w1"))
    position = {o["name"]: i for i, o in enumerate(out["operations"])}
    for o in out["operations"]:
        for r in o["reads"]:
            assert r not in position or position[r] < position[o["name"]], (o["name"], r)


def test_topk_rounds_keepdim_refuses_observable_mask():
    from megakernel_lab.operation_compiler.topk_rounds import rewrite
    graph = _keepdim_graph()
    graph["outputs"].append({"ref": "w0"})
    out, report = rewrite(copy.deepcopy(graph))
    assert report is None and out == graph


def test_topk_rounds_second_chain_from_unrelated_where():
    """A chain whose source is an ordinary where (priority) still starts; continuation wheres do not."""
    from megakernel_lab.operation_compiler.topk_rounds import rewrite
    graph = _keepdim_graph(rounds=2, reader=False)
    r = lambda n: {"ref": n}
    values, ops = graph["values"], graph["operations"]
    values["c"] = _value("c", [4, 16, 32], "bool")
    ops.insert(1, _op("c", "aten.gt.Tensor", [r("x"), r("x")]))
    values["p"] = _value("p", [4, 16, 32], "float32")
    ops.insert(2, _op("p", "aten.where.self", [r("c"), r("x"), r("x")]))
    by = {o["name"]: o for o in ops}
    by["a0"]["args"][0] = r("p")
    for key in ("inputs", "reads", "dependencies"):
        by["a0"][key] = ["p"]
    by["w0"]["args"][2] = r("p")
    for key in ("inputs", "reads", "dependencies"):
        by["w0"][key] = ["e0", "s0", "p"]
    out, report = rewrite(copy.deepcopy(graph))
    assert report and report["rewritten"][0]["source"] == "p" and report["rewritten"][0]["rounds"] == 2
