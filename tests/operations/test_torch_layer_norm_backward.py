"""Small metadata-only LayerNorm dX proofs; no Torch import or arithmetic run."""
import copy
import importlib.util
from pathlib import Path
import sys
import types
import unittest

_package = "_latch_ln_backward_metadata_test"
if _package not in sys.modules:
    module = types.ModuleType(_package)
    module.__path__ = [str(Path(next(iter(importlib.util.find_spec("megakernel_lab").submodule_search_locations)))
                           / "operation_compiler")]
    sys.modules[_package] = module
_spec = importlib.util.spec_from_file_location(_package+".torch_layer_norm_backward",
    Path(sys.modules[_package].__path__[0]) / "torch_layer_norm_backward.py")
ln = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ln
_spec.loader.exec_module(ln)


def fixture(prefix=""):
    values, operations = {}, []
    def n(name): return prefix+name
    def ref(name): return {"ref": n(name)}
    def value(name, shape):
        stride, strides = 1, []
        for size in reversed(shape): strides.insert(0,stride); stride*=size
        values[n(name)] = dict(shape=shape,strides=strides,dtype="float32",storage_offset=0,
                               alias_of=None,output_roles=[])
    def op(name,target,args,shape):
        value(name,shape);refs=list(ln._references(args))
        operations.append(dict(name=n(name),target=target,args=args,kwargs={},inputs=refs,reads=refs,
                               outputs=[n(name)],writes=[n(name)],effect="pure"))
    for name,shape in [("x",[3,8]),("dy",[3,8]),("gain",[8]),("mean",[3,1]),("rstd",[3,1])]:
        value(name,shape)
    op("center","aten.sub.Tensor",[ref("x"),ref("mean")],[3,8])
    op("norm","aten.mul.Tensor",[ref("center"),ref("rstd")],[3,8])
    op("weighted","aten.mul.Tensor",[ref("dy"),ref("gain")],[3,8])
    op("scaled","aten.mul.Tensor",[ref("weighted"),8],[3,8])
    op("sum1","aten.sum.dim_IntList",[ref("weighted"),[-1],True],[3,1])
    op("mixed","aten.mul.Tensor",[ref("weighted"),ref("norm")],[3,8])
    op("sum2","aten.sum.dim_IntList",[ref("mixed"),[1],True],[3,1])
    op("correction","aten.mul.Tensor",[ref("norm"),ref("sum2")],[3,8])
    op("subtract1","aten.sub.Tensor",[ref("scaled"),ref("sum1")],[3,8])
    op("subtract2","aten.sub.Tensor",[ref("subtract1"),ref("correction")],[3,8])
    op("scale","aten.div.Tensor",[ref("rstd"),8],[3,1])
    op("output","aten.mul.Tensor",[ref("scale"),ref("subtract2")],[3,8])
    op("gain_product","aten.mul.Tensor",[ref("dy"),ref("norm")],[3,8])
    return dict(values=values,operations=operations,inputs=[n(x) for x in ("x","dy","gain","mean","rstd")],
        outputs={"dx":n("output"),"gain_input":n("gain_product")},metadata={ln.KEY:ln.CONTRACT})


class LayerNormBackwardMetadataTests(unittest.TestCase):
    def refused(self,g):
        self.assertEqual(ln.find_plans(g),[])
        self.assertTrue(ln.analyze(g)["refused"])

    def test_default_noop(self):
        g=fixture();g["metadata"]={};before=copy.deepcopy(g)
        self.assertEqual(ln.find_plans(g),[]);self.assertEqual(g,before)

    def test_positive_preserves_gain_tape(self):
        g=fixture();before=copy.deepcopy(g);p,=ln.find_plans(g)
        self.assertEqual((p.anchor,p.rows,p.width,p.output),("weighted",3,8,"output"))
        self.assertEqual(p.preserved,("center","norm"))
        self.assertEqual(len(p.members),10);self.assertEqual(len(p.intermediates),9)
        self.assertEqual(g,before)

    def test_escaping_ancestors_preserved(self):
        for mode in ("output","state_write","role","alias"):
            g=fixture()
            if mode=="output":g["outputs"]["saved"]="mixed"
            elif mode=="state_write":g["state_writes"]={"saved":"mixed"}
            elif mode=="role":g["values"]["mixed"]["output_roles"]=["saved"]
            else:g["values"]["alias"]=dict(g["values"]["mixed"],alias_of="mixed")
            p,=ln.find_plans(g)
            self.assertTrue({"center","norm","weighted","mixed"}<=set(p.preserved))
            self.assertFalse(set(p.preserved)&set(p.members))

    def test_all_intermediates_can_escape(self):
        g=fixture();g["outputs"]["saved"]=[o["outputs"][0] for o in g["operations"]]
        p,=ln.find_plans(g);self.assertEqual(p.members,("output",));self.assertEqual(p.intermediates,())

    def test_numerical_refusals(self):
        for operation,index,value in [("scale",1,float("nan")),("scale",1,7),("scaled",1,7),
                                       ("sum1",1,[0]),("sum2",2,False),
                                       ("mixed",1,{"ref":"center"})]:
            g=fixture();next(o for o in g["operations"] if o["name"]==operation)["args"][index]=value
            self.refused(g)
        g=fixture();g["operations"][8]["kwargs"]={"alpha":2};self.refused(g)

    def test_layout_dtype_refusals(self):
        for name,key,value in [("dy","dtype","bfloat16"),("gain","shape",[1,8]),
                                ("x","strides",[9,1]),("x","storage_offset",1),
                                ("mean","shape",[3]),("rstd","dtype","float64"),
                                ("output","storage_offset",1)]:
            g=fixture();g["values"][name][key]=value;self.refused(g)

    def test_effect_refusal(self):
        g=fixture();g["operations"].insert(8,dict(name="effect",target="custom.write",args=[],kwargs={},
            inputs=[],outputs=[],reads=[],writes=["x"],effect="write"));self.refused(g)

    def test_missing_binding_and_late_source(self):
        g=fixture();g["inputs"].remove("gain");self.refused(g)
        g=fixture();g["inputs"].remove("gain");g["inputs"].append("source")
        g["values"]["source"]=dict(g["values"]["gain"])
        g["operations"].insert(4,dict(name="late",target="aten.clone.default",args=[{"ref":"source"}],
            kwargs={},inputs=["source"],reads=["source"],outputs=["gain"],writes=["gain"],effect="pure"))
        self.refused(g)

    def test_multiple_groups(self):
        g,h=fixture("a_"),fixture("b_");g["values"].update(h["values"])
        g["operations"].extend(h["operations"]);g["inputs"].extend(h["inputs"]);g["outputs"]["second"]=h["outputs"]
        self.assertEqual([p.anchor for p in ln.find_plans(g)],["a_weighted","b_weighted"])

    def test_unknown_contract(self):
        g=fixture();g["metadata"][ln.KEY]="unknown"
        with self.assertRaises(ValueError):ln.find_plans(g)


if __name__=="__main__":unittest.main()
