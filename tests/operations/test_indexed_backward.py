"""select/slice_backward placements evaluated at their readers (ops/indexed_backward.py; elementwise_rules 'indexed')."""
from __future__ import annotations

import re

import pytest

from megakernel_lab.operation_compiler import elementwise_rules
from megakernel_lab.operation_compiler import indexed_backward as ib


def _value(shape, dtype="float32"):
    stride, strides = 1, []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return {"shape": shape, "strides": list(reversed(strides)), "dtype": dtype}


def _stack_graph(t=256, k=4, d=96, mean=True, extra_reader=False, out_select=False):
    """grad of a [t, k, d] stack = sum_j select_backward(g_j, dim=1, index=j) (+ mean backward)."""
    values = {f"g{j}": _value([t, d]) for j in range(k)}
    values.update({f"s{j}": _value([t, k, d]) for j in range(k)})
    values.update({"m": _value([t, 1, d]), "w": _value([t, k, d])})
    ops = [{"name": f"s{j}", "target": "aten.select_backward.default", "args": [{"ref": f"g{j}"}, [t, k, d], 1, j]}
           for j in range(k)]
    acc = "s0"
    for j in range(1, k):
        values[f"a{j}"] = _value([t, k, d])
        ops.append({"name": f"a{j}", "target": "aten.add.Tensor", "args": [{"ref": acc}, {"ref": f"s{j}"}]})
        acc = f"a{j}"
    if mean:
        values["e"] = _value([t, k, d]); values["q"] = _value([t, k, d]); values["r"] = _value([t, k, d])
        ops += [{"name": "e", "target": "aten.expand.default", "args": [{"ref": "m"}, [t, k, d]]},
                {"name": "q", "target": "aten.div.Scalar", "args": [{"ref": "e"}, float(k)]},
                {"name": "r", "target": "aten.add.Tensor", "args": [{"ref": acc}, {"ref": "q"}]}]
        acc = "r"
    values["y"] = _value([t, k, d])
    ops.append({"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": acc}, {"ref": "w"}]})
    outputs = ["y"]
    if extra_reader:
        values["z"] = _value([t, k, d])
        ops.append({"name": "z", "target": "aten.neg.default", "args": [{"ref": "s1"}]})
        outputs.append("z")
    if out_select:
        outputs.append("s0")
    # inputs g_j, m, w; e is a view of m in the IR (expand)
    graph = {"inputs": [*(f"g{j}" for j in range(k)), "m", "w"], "outputs": outputs, "values": values,
             "operations": ops}
    return graph


def _lower(graph, on):
    from megakernel_lab.operation_compiler.lower import lower
    with elementwise_rules.scope("indexed" if on else None):
        return lower(graph, execution="compose", compose_selection="common", group_rules="base+vec4")


def test_disabled_by_default():
    assert not ib.enabled()
    with elementwise_rules.scope("indexed"):
        assert ib.enabled()
    with elementwise_rules.scope("lanecse"):
        assert not ib.enabled()
    assert ib.KINDS == ("select_backward", "slice_backward")


def _check_records(program, target):
    """Every record is consistent: admitted -> not stored and absent from the source; refused -> still stored."""
    m = program.manifest
    stored = {p["writes"][0] for p in m["phases"] if p["targets"] == [target]}
    for r in m["group_rules"]["indexed_backward_at_readers"]:
        assert (r["value"] in stored) == ("refused" in r), r
        if "refused" not in r:
            assert not re.search(rf"\bv_{r['value']}\b", program.source)
            reader = [p for p in m["phases"] if p["names"][0] in r["readers"]]
            assert reader and all(set(r["operands"]) <= set(p["reads"]) for p in reader)
    return {r["value"]: r for r in m["group_rules"]["indexed_backward_at_readers"]}


def test_select_backward_chain_records():
    graph = _stack_graph()
    plain = _lower(graph, on=False)
    assert len([p for p in plain.manifest["phases"] if p["targets"] == ["aten.select_backward.default"]]) == 4
    assert "indexed_backward_at_readers" not in plain.manifest["group_rules"]
    program = _lower(graph, on=True)
    rec = _check_records(program, "aten.select_backward.default")
    assert sorted(rec) == ["s0", "s1", "s2", "s3"]
    # This small graph's reader is an already vectorized elementwise body (float4 loads): a non-identity read,
    # so the placements stay stored.
    assert all(r.get("refused") in (None, "non-identity read") for r in rec.values())
    assert program.manifest["phase_count"] == len(program.manifest["phases"]) + 1


def test_kept_results_are_refused():
    graph = _stack_graph(out_select=True)
    program = _lower(graph, on=True)
    rec = _check_records(program, "aten.select_backward.default")
    assert rec["s0"]["refused"] == "kept or non-root result"


def test_multiple_readers_records():
    program = _lower(_stack_graph(extra_reader=True), on=True)
    _check_records(program, "aten.select_backward.default")


def test_slice_backward_column_placements():
    t, k = 128, 6
    values = {"x": _value([t, 1]) for _ in range(1)}
    values = {f"g{j}": _value([t, 1]) for j in range(k)}
    values.update({f"s{j}": _value([t, k]) for j in range(k)})
    ops = [{"name": f"s{j}", "target": "aten.slice_backward.default", "args": [{"ref": f"g{j}"}, [t, k], 1, j, j + 1, 1]}
           for j in range(k)]
    acc = "s0"
    for j in range(1, k):
        values[f"a{j}"] = _value([t, k])
        ops.append({"name": f"a{j}", "target": "aten.add.Tensor", "args": [{"ref": acc}, {"ref": f"s{j}"}]})
        acc = f"a{j}"
    values["w"] = _value([t, k]); values["y"] = _value([t, k])
    ops.append({"name": "y", "target": "aten.mul.Tensor", "args": [{"ref": acc}, {"ref": "w"}]})
    graph = {"inputs": [*(f"g{j}" for j in range(k)), "w"], "outputs": ["y"], "values": values, "operations": ops}
    program = _lower(graph, on=True)
    rec = program.manifest["group_rules"]["indexed_backward_at_readers"]
    stored = {p["writes"][0] for p in program.manifest["phases"] if p["targets"] == ["aten.slice_backward.default"]}
    for r in rec:
        # admitted placements disappear; a refused one (e.g. a reader that renders another index form) stays stored
        assert (r["value"] in stored) == ("refused" in r), r
        if "refused" in r:
            assert r["refused"] == "non-identity read"


class _Layout:
    def __init__(self, name, shape):
        self.shape, self.root, self.offset, self.contiguous = shape, name, 0, True
        n = 1
        for e in shape:
            n *= e
        self.numel = n


def _placement_phases(target="aten.select_backward.default"):
    code = ("for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < 1024LL; i += 1) {\n"
            "  const long long c = ((i / 64LL) % 4LL);\n"
            "  v_s[i] = (c == 2LL) ? v_g[((i / 256LL) % 4LL) * 64LL + ((i / 1LL) % 64LL) * 1LL] : 0.0f;\n}")
    layouts = {n: _Layout(n, shape) for n, shape in (("s", (4, 4, 64)), ("g", (4, 64)), ("r", (4, 4, 64)))}
    phases = [{"kind": "elementwise", "targets": [target], "names": ["s"], "writes": ["s"], "reads": ["g"],
               "code": code, "shape": [4, 4, 64]},
              {"kind": "elementwise", "targets": ["aten.add.Tensor"], "names": ["r"], "writes": ["r"],
               "reads": ["s", "x"], "shape": [4, 4, 64], "body": ["v_r[(i)] = v_s[(i)] + v_x[(i)];"]}]
    return phases, layouts


@pytest.mark.parametrize("target", ["aten.select_backward.default", "aten.slice_backward.default", "aten.add.Tensor"])
def test_placement_substituted_at_identity_reader(target):
    phases, layouts = _placement_phases(target)
    kwargs = dict(kept_roots=set(), numel=lambda shape: 1024, ident=lambda n: "v_" + n)
    assert ib.place_at_readers(phases, layouts, **kwargs) == ([], set(), None)      # token absent: unchanged
    assert len(phases) == 2
    with elementwise_rules.scope("indexed"):
        records, dead, helpers = ib.place_at_readers(phases, layouts, **kwargs)
    assert helpers is None
    if target == "aten.add.Tensor":
        assert (records, dead, len(phases)) == ([], set(), 2)
        return
    assert dead == {"s"} and [p["names"] for p in phases] == [["r"]]
    assert records == [{"value": "s", "kind": target.split(".")[1], "shape": [4, 4, 64], "elements_not_stored": 1024,
                        "readers": ["r"], "operands": ["g"]}]
    assert phases[0]["body"] == ["v_r[(i)] = (((((i / 64LL) % 4LL)) == 2LL) ? v_g[((i / 256LL) % 4LL) * 64LL + "
                                 "((i / 1LL) % 64LL) * 1LL] : 0.0f) + v_x[(i)];"]
    assert phases[0]["reads"] == ["x", "g"] and phases[0]["indexed_backward_inputs"] == ["s"]


@pytest.mark.parametrize("code,expect", [
    ("for (long long i = 0; i < 8LL; i += 1) {\n  const long long c = ((i / 2LL) % 4LL);\n"
     "  v_sb[i] = (c == 3LL) ? v_g[((i / 8LL) % 1LL) * 2LL + ((i / 1LL) % 2LL) * 1LL] : 0.0f;\n}",
     "((((i / 2LL) % 4LL)) == 3LL) ? v_g[((i / 8LL) % 1LL) * 2LL + ((i / 1LL) % 2LL) * 1LL] : 0.0f)"),
    ("if (aligned) {\nvector\n} else {\nfor (long long i = 0; i < 8LL; i += 1) { const long long c_ = ((i / 1LL) % 8LL); "
     "v_sb[i] = (c_ >= 2LL && (c_ - 2LL) % 1LL == 0 && ((c_ - 2LL) / 1LL) < 1LL) ? v_g[0LL] : 0.0f; }\n}",
     "((((i / 1LL) % 8LL)) >= 2LL && ((((i / 1LL) % 8LL)) - 2LL) % 1LL == 0 && (((((i / 1LL) % 8LL)) - 2LL) / 1LL) "
     "< 1LL) ? v_g[0LL] : 0.0f)"),
])
def test_element_expression_parses_both_scalar_forms(code, expect):
    phase = {"code": code, "writes": ["sb"], "reads": ["g"], "names": ["sb"]}
    expr, why = ib.element_expression(phase, lambda n: "v_" + n)
    assert why is None and expr == "(" + expect


def test_element_expression_refuses_foreign_identifiers():
    code = ("for (long long i = 0; i < 8LL; i += 1) {\n  const long long c = (i % 4LL);\n"
            "  v_sb[i] = (c == 3LL) ? v_other[i] : 0.0f;\n}")
    expr, why = ib.element_expression({"code": code, "writes": ["sb"], "reads": ["g"], "names": ["sb"]},
                                      lambda n: "v_" + n)
    assert expr is None and "v_other" in why
