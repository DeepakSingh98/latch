"""Tiny metadata-only admission tests; importing this file does not import torch."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

_spec = importlib.util.spec_from_file_location("_latch_layer_norm_metadata_test",
    Path(next(iter(importlib.util.find_spec("megakernel_lab").submodule_search_locations)))
    / "operation_compiler" / "torch_layer_norm.py")
ln = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ln
_spec.loader.exec_module(ln)


def fixture(prefix=""):
    values, ops = {}, []
    def name(n): return prefix+n
    def value(n, shape):
        stride, strides = 1, []
        for size in reversed(shape): strides.insert(0, stride); stride *= size
        values[name(n)] = dict(name=name(n), shape=shape, strides=strides, dtype="float32",
                              storage_offset=0, alias_of=None, output_roles=[])
    def ref(n): return {"ref": name(n)}
    def op(n, target, args, shape):
        value(n, shape)
        refs = list(ln._references(args))
        ops.append(dict(name=name(n), target=target, args=args, kwargs={}, inputs=refs,
                        outputs=[name(n)], reads=refs, writes=[name(n)], effect="pure"))
    value("x", [3, 8]); value("gain", [8]); value("bias", [8])
    op("mean", "aten.mean.dim", [ref("x"), [-1], True], [3, 1])
    op("center", "aten.sub.Tensor", [ref("x"), ref("mean")], [3, 8])
    op("square", "aten.mul.Tensor", [ref("center"), ref("center")], [3, 8])
    op("sum", "aten.sum.dim_IntList", [ref("square"), [1], True], [3, 1])
    op("var", "aten.div.Tensor", [ref("sum"), 8], [3, 1])
    op("eps", "aten.add.Tensor", [ref("var"), 1e-5], [3, 1])
    op("rstd", "aten.rsqrt.default", [ref("eps")], [3, 1])
    op("center2", "aten.sub.Tensor", [ref("x"), ref("mean")], [3, 8])
    op("norm", "aten.mul.Tensor", [ref("center2"), ref("rstd")], [3, 8])
    op("scaled", "aten.mul.Tensor", [ref("norm"), ref("gain")], [3, 8])
    op("out", "aten.add.Tensor", [ref("scaled"), ref("bias")], [3, 8])
    return dict(values=values, operations=ops, inputs=[name(n) for n in ("x", "gain", "bias")],
                outputs={"result": name("out"), "stats": [name("mean"), name("rstd")]},
                metadata={ln.KEY: ln.CONTRACT})


class LayerNormMetadataTests(unittest.TestCase):
    def refused(self, graph):
        self.assertEqual(ln.find_plans(graph), [])
        self.assertTrue(ln.analyze(graph)["refused"])

    def test_default_is_noop_and_does_not_modify(self):
        g = fixture(); g["metadata"] = {}; before = copy.deepcopy(g)
        self.assertEqual(ln.find_plans(g), []); self.assertEqual(g, before)

    def test_positive_exact_members_and_no_mutation(self):
        g = fixture(); before = copy.deepcopy(g); p, = ln.find_plans(g)
        self.assertEqual((p.rows, p.width, p.x, p.gain, p.bias), (3, 8, "x", "gain", "bias"))
        self.assertEqual(p.finals, ("out", "mean", "rstd"))
        self.assertEqual(p.members, tuple(op["name"] for op in g["operations"]))
        self.assertEqual(set(p.intermediates), set(g["values"]) - {"x", "gain", "bias", *p.finals})
        self.assertEqual(g, before)

    def test_multiple_independent_groups(self):
        a, b = fixture("a_"), fixture("b_")
        a["values"].update(b["values"]); a["operations"].extend(b["operations"])
        a["inputs"].extend(b["inputs"]); a["outputs"]["second"] = b["outputs"]
        self.assertEqual([p.anchor for p in ln.find_plans(a)], ["a_mean", "b_mean"])

    def test_intermediate_external_reader(self):
        g = fixture(); op = copy.deepcopy(g["operations"][-1]); op.update(
            name="consumer", outputs=["consumer"], args=[{"ref": "center"}, {"ref": "bias"}],
            inputs=["center", "bias"], reads=["center", "bias"], writes=["consumer"])
        g["operations"].append(op); g["values"]["consumer"] = copy.deepcopy(g["values"]["out"])
        self.refused(g)

    def test_intermediate_escape_forms(self):
        for mode in ("output", "state_write", "role", "alias"):
            with self.subTest(mode=mode):
                g = fixture()
                if mode == "output": g["outputs"]["extra"] = "center"
                elif mode == "state_write": g["state_writes"] = {"saved": "center"}
                elif mode == "role": g["values"]["center"]["output_roles"] = ["saved"]
                else:
                    g["values"]["external_alias"] = dict(g["values"]["center"], alias_of="center")
                self.refused(g)

    def test_storage_and_dtype_refusals(self):
        cases = [("x", "dtype", "bfloat16"), ("gain", "dtype", "float64"),
                 ("x", "strides", [9, 1]), ("gain", "strides", [2]),
                 ("x", "storage_offset", 1), ("gain", "storage_offset", 1),
                 ("out", "storage_offset", 1), ("rstd", "strides", [2, 1]),
                 ("x", "shape", [3, 7]), ("x", "shape", [3, 4100]),
                 ("bias", "shape", [1, 8]), ("sum", "dtype", "float64")]
        for value, field, replacement in cases:
            with self.subTest(value=value, field=field):
                g = fixture(); g["values"][value][field] = replacement; self.refused(g)

    def test_numerical_equation_refusals(self):
        for op_name, index, replacement in [("mean", 1, [0]), ("mean", 2, False),
                                            ("var", 1, 7), ("eps", 1, float("nan")),
                                            ("eps", 1, 0)]:
            with self.subTest(op=op_name, index=index):
                g = fixture(); next(o for o in g["operations"] if o["name"] == op_name)["args"][index] = replacement
                self.refused(g)
        g = fixture(); g["operations"][-1]["kwargs"]["alpha"] = 2; self.refused(g)

    def test_source_must_precede_anchor(self):
        g = fixture(); g["inputs"].remove("gain")
        g["inputs"].append("source"); g["values"]["source"] = copy.deepcopy(g["values"]["gain"])
        g["operations"].insert(1, dict(name="late_gain", target="aten.clone.default",
            args=[{"ref": "source"}], kwargs={}, inputs=["source"], outputs=["gain"], reads=["source"],
            writes=["gain"], effect="pure"))
        self.refused(g)

    def test_alias_and_effect_safety(self):
        g = fixture(); g["operations"].insert(1, dict(name="write", target="custom.write",
            args=[], kwargs={}, inputs=[], outputs=[], writes=["x"], effect="write"))
        self.refused(g)

    def test_ambiguous_equation_and_unknown_contract(self):
        g = fixture(); duplicate = copy.deepcopy(g["operations"][-1]); duplicate.update(
            name="out2", outputs=["out2"]); g["operations"].append(duplicate)
        self.refused(g)
        g = fixture(); g["metadata"][ln.KEY] = "unknown"
        with self.assertRaises(ValueError): ln.find_plans(g)


if __name__ == "__main__":
    unittest.main()
