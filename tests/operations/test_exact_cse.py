"""Exact CSE on the captured IR (operation_compiler/exact_cse.py; graph_rules 'cse', applied by prepare_module)."""
from megakernel_lab.operation_compiler import exact_cse
from megakernel_lab.operation_compiler.ir import Operation, Ref, TrainingIR, Value


def _v(name, role="temporary", producer=None, alias_of=None):
    return Value(name, (4, 8), "float32", (8, 1), role=role, producer=producer, alias_of=alias_of)


def _op(name, target, args, outs=None, effect="pure"):
    reads = tuple(a.name for a in args if isinstance(a, Ref))
    outs = outs or (name,)
    return Operation(name, target, tuple(args), {}, reads, outs, (), reads, outs, effect=effect)


def _graph():
    values = {n: _v(n, producer=n) for n in ("m1", "m2", "r1", "r2", "c1", "c2", "u", "rnd1", "rnd2", "out")}
    values["x"] = _v("x", role="input")
    ops = [
        _op("m1", "aten.mul.Tensor", (Ref("x"), Ref("x"))),
        _op("m2", "aten.mul.Tensor", (Ref("x"), Ref("x"))),          # duplicate of m1
        _op("r1", "aten.rsqrt.default", (Ref("m1"),)),
        _op("r2", "aten.rsqrt.default", (Ref("m2"),)),               # duplicate after renaming m2 -> m1
        _op("c1", "aten._to_copy.default", (Ref("x"), "bfloat16")),
        _op("c2", "aten._to_copy.default", (Ref("x"), "float16")),   # different attribute: kept
        _op("rnd1", "latch_ops.philox_mask.default", (Ref("x"),)),
        _op("rnd2", "latch_ops.philox_mask.default", (Ref("x"),)),   # random target: kept
        _op("u", "aten.add.Tensor", (Ref("r1"), Ref("r2"))),
        _op("out", "aten.add.Tensor", (Ref("u"), Ref("c2"))),
    ]
    values["out"] = _v("out", role="output", producer="out")
    return TrainingIR(values, ops, ("x",), {"loss": "out"}, {})


def test_nothing_to_merge_returns_same_object():
    values = {"a": _v("a", producer="a"), "x": _v("x", role="input"), "out": _v("out", role="output", producer="out")}
    ops = [_op("a", "aten.mul.Tensor", (Ref("x"), Ref("x"))), _op("out", "aten.add.Tensor", (Ref("a"), Ref("x")))]
    g = TrainingIR(values, ops, ("x",), {"loss": "out"}, {})
    got, report = exact_cse.apply(g)
    assert got is g and report == {"merged": 0}
    assert exact_cse.apply({"operations": []}) == ({"operations": []}, None)


def test_merges_identical_pure_ops_only():
    g, report = exact_cse.apply(_graph())
    names = [op.name for op in g.operations]
    assert names == ["m1", "r1", "c1", "c2", "rnd1", "rnd2", "u", "out"]
    u = next(op for op in g.operations if op.name == "u")
    assert u.args == (Ref("r1"), Ref("r1")) and u.reads == ("r1",)
    assert "m2" not in g.values and "r2" not in g.values and "out" in g.values
    assert report == {"merged": 2, "values_removed": 2,
                      "by_target": {"aten.mul.Tensor": 1, "aten.rsqrt.default": 1}}


def test_protected_outputs_are_not_removed():
    g = _graph()
    g.values["m2"] = _v("m2", role="output", producer="m2")
    g2, report = exact_cse.apply(g)
    assert "m2" in [op.name for op in g2.operations]


def test_operand_order_is_significant():
    values = {n: _v(n, producer=n) for n in ("a", "b", "u")}
    values["x"] = _v("x", role="input")
    values["y"] = _v("y", role="input")
    values["out"] = _v("out", role="output", producer="out")
    ops = [_op("a", "aten.mul.Tensor", (Ref("x"), Ref("y"))), _op("b", "aten.mul.Tensor", (Ref("y"), Ref("x"))),
           _op("u", "aten.sub.Tensor", (Ref("a"), Ref("b"))), _op("out", "aten.add.Tensor", (Ref("u"), Ref("u")))]
    g = TrainingIR(values, ops, ("x", "y"), {"loss": "out"}, {})
    assert exact_cse.apply(g)[1]["merged"] == 0


def test_opaque_targets_are_not_merged():
    values = {n: _v(n, producer=n) for n in ("t1", "t2", "u")}
    values["x"] = _v("x", role="input")
    values["out"] = _v("out", role="output", producer="out")
    ops = [_op("t1", "latch_ops.composed_attention_tape.default", (Ref("x"),)),
           _op("t2", "latch_ops.composed_attention_tape.default", (Ref("x"),)),
           _op("u", "aten.add.Tensor", (Ref("t1"), Ref("t2"))), _op("out", "aten.mul.Tensor", (Ref("u"), Ref("u")))]
    g = TrainingIR(values, ops, ("x",), {"loss": "out"}, {})
    assert exact_cse.apply(g)[1]["merged"] == 0
