"""Segmented-attention rotary pass recognizer (segmented_attention='rotary') on a synthetic joint-IR attention layer."""
import pytest

from megakernel_lab.operation_compiler import segmented_attention_rules as rules
from megakernel_lab.operation_compiler import segmented_rotary

R, H, P, D = 64, 3, 32, 28


def _v(name, shape, dtype, alias_of=None):
    strides, step = [], 1
    for d in reversed(shape):
        strides.append(step)
        step *= d
    return dict(name=name, shape=list(shape), dtype=dtype, strides=list(reversed(strides)), role="temporary",
                parameter=None, alias_of=alias_of, storage_offset=0, producer=name, output_roles=[])


def _op(name, target, args, reads, outputs=None, kwargs=None):
    outputs = outputs or [name]
    return dict(name=name, target=target, args=args, kwargs=kwargs or {}, inputs=list(reads), outputs=list(outputs),
                dependencies=list(reads), reads=list(reads), writes=list(outputs), reduction_axes=[],
                synchronization="global_phase_boundary", effect="pure")


def ref(n):
    return {"ref": n}


def layer(add_swap=False, outer_swap=False, leak=False, widen=False):
    V, O = {}, []

    def val(name, shape, dtype, alias=None):
        V[name] = _v(name, shape, dtype, alias)

    def op(name, target, args, reads, shape, dtype, alias=None, **kw):
        val(name, shape, dtype, alias)
        O.append(_op(name, target, args, reads, **kw))

    for n, shp, dt in (("proj", (R, H * P), "bfloat16"), ("cos", (R, D), "float32"), ("sin", (R, D), "float32"),
                       ("offs", (5,), "int64"), ("v", (R, H, P), "bfloat16"), ("dout", (R, H, P), "bfloat16")):
        val(n, shp, dt)
    h = D // 2
    for role in ("q", "k"):
        x, x32, t = f"x_{role}", f"x32_{role}", f"t_{role}"
        op(x, "aten.view.default", [ref("proj"), [R, H, P]], ["proj"], (R, H, P), "bfloat16", "proj")
        op(x32, "aten._to_copy.default", [ref(x)], [x], (R, H, P), "float32", kwargs={"dtype": "torch.float32"})
        op(t, "aten.slice.Tensor", [ref(x32), 2, 0, D], [x32], (R, H, D), "float32", x32)
        op(f"hi_{role}", "aten.slice.Tensor", [ref(t), -1, h, D], [t], (R, H, h), "float32", t)
        op(f"lo_{role}", "aten.slice.Tensor", [ref(t), -1, 0, h], [t], (R, H, h), "float32", t)
        op(f"neg_{role}", "aten.neg.default", [ref(f"hi_{role}")], [f"hi_{role}"], (R, H, h), "float32")
        op(f"rot_{role}", "aten.cat.default", [[ref(f"neg_{role}"), ref(f"lo_{role}")], -1],
           [f"neg_{role}", f"lo_{role}"], (R, H, D), "float32")
        cu, su = f"cu_{role}", f"su_{role}"
        op(cu, "aten.unsqueeze.default", [ref("cos"), 1], ["cos"], (R, 1, D), "float32", "cos")
        op(su, "aten.unsqueeze.default", [ref("sin"), 1], ["sin"], (R, 1, D), "float32", "sin")
        op(f"m1_{role}", "aten.mul.Tensor", [ref(t), ref(cu)], [t, cu], (R, H, D), "float32")
        op(f"m2_{role}", "aten.mul.Tensor", [ref(f"rot_{role}"), ref(su)], [f"rot_{role}", su], (R, H, D), "float32")
        pair = [ref(f"m2_{role}"), ref(f"m1_{role}")] if add_swap else [ref(f"m1_{role}"), ref(f"m2_{role}")]
        op(f"a_{role}", "aten.add.Tensor", pair, [f"m1_{role}", f"m2_{role}"], (R, H, D), "float32")
        op(f"b_{role}", "aten._to_copy.default", [ref(f"a_{role}")], [f"a_{role}"], (R, H, D), "bfloat16",
           kwargs={"dtype": "torch.bfloat16"})
        op(f"z_{role}", "aten.new_zeros.default", [ref(f"b_{role}"), [R, H, P - D]], [f"b_{role}"],
           (R, H, P - D), "bfloat16")
        op(f"cat_{role}", "aten.cat.default", [[ref(f"b_{role}"), ref(f"z_{role}")], -1], [f"b_{role}", f"z_{role}"],
           (R, H, P), "bfloat16")
    for n, shp, dt in (("o", (R, H, P), "bfloat16"), ("lse", (H, R), "float32"), ("wf", (4, H, 3), "int64"),
                       ("dq", (R, H, P), "bfloat16"), ("dk", (R, H, P), "bfloat16"), ("dv", (R, H, P), "bfloat16"),
                       ("wb", (4, H, 6), "int64")):
        val(n, shp, dt)
    O.append(_op("fwd", "latch_ops.segmented_sdpa_fwd.default",
                 [ref("cat_q"), ref("cat_k"), ref("v"), ref("offs"), 64, 0.125], ["cat_q", "cat_k", "v", "offs"],
                 outputs=["o", "lse", "wf"]))
    O.append(_op("bwd", "latch_ops.segmented_sdpa_bwd.default",
                 [ref("cat_q"), ref("cat_k"), ref("v"), ref("o"), ref("lse"), ref("dout"), ref("offs"), 64, 0.125],
                 ["cat_q", "cat_k", "v", "o", "lse", "dout", "offs"], outputs=["dq", "dk", "dv", "wb"]))
    outs = []
    for role, grad in (("q", "dq"), ("k", "dk")):
        gs, ms, mc = f"gs_{role}", f"ms_{role}", f"mc_{role}"
        op(gs, "aten.slice.Tensor", [ref(grad), 2, 0, D], [grad], (R, H, D), "bfloat16", grad)
        src = gs
        if widen:
            src = f"gw_{role}"
            op(src, "aten._to_copy.default", [ref(gs)], [gs], (R, H, D), "float32", kwargs={"dtype": "torch.float32"})
        op(ms, "aten.mul.Tensor", [ref(src), ref(f"su_{role}")], [src, f"su_{role}"], (R, H, D), "float32")
        op(mc, "aten.mul.Tensor", [ref(src), ref(f"cu_{role}")], [src, f"cu_{role}"], (R, H, D), "float32")
        op(f"sl_{role}", "aten.slice.Tensor", [ref(ms), -1, 0, h], [ms], (R, H, h), "float32", ms)
        op(f"sh_{role}", "aten.slice.Tensor", [ref(ms), -1, h, D], [ms], (R, H, h), "float32", ms)
        op(f"bneg_{role}", "aten.neg.default", [ref(f"sl_{role}")], [f"sl_{role}"], (R, H, h), "float32")
        op(f"sb1_{role}", "aten.slice_backward.default", [ref(f"sh_{role}"), [R, H, D], -1, 0, h, 1], [f"sh_{role}"],
           (R, H, D), "float32")
        op(f"sb2_{role}", "aten.slice_backward.default", [ref(f"bneg_{role}"), [R, H, D], -1, h, 9223372036854775807, 1],
           [f"bneg_{role}"], (R, H, D), "float32")
        op(f"ba1_{role}", "aten.add.Tensor", [ref(mc), ref(f"sb1_{role}")], [mc, f"sb1_{role}"], (R, H, D), "float32")
        pair = [ref(f"sb2_{role}"), ref(f"ba1_{role}")] if outer_swap else [ref(f"ba1_{role}"), ref(f"sb2_{role}")]
        op(f"ba2_{role}", "aten.add.Tensor", pair, [f"ba1_{role}", f"sb2_{role}"], (R, H, D), "float32")
        op(f"pad_{role}", "aten.slice_backward.default", [ref(f"ba2_{role}"), [R, H, P], -1, 0, D, 1],
           [f"ba2_{role}"], (R, H, P), "float32")
        op(f"g_{role}", "aten._to_copy.default", [ref(f"pad_{role}")], [f"pad_{role}"], (R, H, P), "bfloat16",
           kwargs={"dtype": "torch.bfloat16"})
        outs.append(f"g_{role}")
    if leak:
        op("leak", "aten.mul.Tensor", [ref("m1_q"), ref("m1_q")], ["m1_q"], (R, H, D), "float32")
        outs.append("leak")
    return dict(values=V, operations=O, inputs=["proj", "cos", "sin", "offs", "v", "dout"],
                outputs={"loss": "o", "grads": outs}, metadata={}, format_version=1)


@pytest.fixture()
def rotary():
    with rules.scope("rotary"):
        yield segmented_rotary


@pytest.mark.parametrize("add_swap,outer_swap,widen", [(False, False, False), (True, True, False), (False, False, True)])
def test_rewrites_both_operands_and_gradients(rotary, add_swap, outer_swap, widen):
    data, report = rotary.rewrite(layer(add_swap, outer_swap, widen=widen))
    assert len(report["packs"]) == 2 and len(report["unpacks"]) == 2, report["refused"]
    targets = [op["target"] for op in data["operations"]]
    assert targets.count(rotary.PACK_TARGET) == 2 and targets.count(rotary.UNPACK_TARGET) == 2
    names = {op["name"] for op in data["operations"]}
    assert not names & {"x32_q", "rot_k", "a_q", "z_k", "mc_q", "ba2_k", "pad_q"}
    assert {"cat_q", "cat_k", "g_q", "g_k", "fwd", "bwd", "cu_q", "su_k"} <= names
    pack = next(op for op in data["operations"] if op["name"] == "cat_q")
    assert pack["args"][:3] == [{"ref": "x_q"}, {"ref": "cu_q"}, {"ref": "su_q"}]
    assert pack["args"][4] == ("rotated_first" if add_swap else "direct_first")
    unpack = next(op for op in data["operations"] if op["name"] == "g_q")
    assert unpack["args"][:3] == [{"ref": "dq"}, {"ref": "cu_q"}, {"ref": "su_q"}]
    assert unpack["args"][4] == ("ch|li" if outer_swap else "ch|il")
    seen = set(data["inputs"])
    for op in data["operations"]:
        assert all(r in data["values"] for r in op["reads"]) and set(op["reads"]) <= seen, op["name"]
        seen |= set(op["outputs"])


def test_outside_reader_keeps_chain(rotary):
    _, report = rotary.rewrite(layer(leak=True))
    assert {r["operand"] for r in report["packs"]} == {"cat_k"}
    assert any(r.get("operand") == "cat_q" for r in report["refused"])


def test_disabled_is_identity():
    graph = layer()
    assert segmented_rotary.rewrite(graph) == (graph, None)
    with rules.scope("claim+folded_forward+fused_backward"):
        assert segmented_rotary.rewrite(graph) == (graph, None)


def test_targets_and_report_metadata(rotary):
    assert rotary.TARGETS == ("latch_ops.segmented_rotary_pack.default", "latch_ops.segmented_rotary_unpack.default")
    data, report = rotary.rewrite(layer())
    assert data["metadata"]["segmented_rotary"] is report and report["kinds"] == ["pack", "unpack"]
