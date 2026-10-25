"""Integration4 contracts for exact gather/cast and scaled-merge rewrites (cloud CPU)."""
import copy
import math

import pytest

from megakernel_lab.operation_compiler import gather_cast, scaled_merge, widen_cast


def _value(name, shape, dtype="float32", alias=None):
    return {"name": name, "shape": list(shape), "strides": gather_cast._contiguous(shape),
            "dtype": dtype, "alias_of": alias, "storage_offset": 0, "output_roles": []}


def _op(name, target, args, reads, *, kwargs=None, view=False):
    return {"name": name, "target": target, "args": args, "kwargs": kwargs or {},
            "inputs": list(reads), "reads": list(reads), "dependencies": list(reads),
            "outputs": [name], "writes": [name],
            "synchronization": "view" if view else "global_phase_boundary"}


def _r(name):
    return {"ref": name}


def _check_refs(graph):
    names = set(graph["values"])

    def refs(item):
        if isinstance(item, dict):
            if set(item) == {"ref"}:
                yield item["ref"]
            else:
                for value in item.values():
                    yield from refs(value)
        elif isinstance(item, (tuple, list)):
            for value in item:
                yield from refs(value)
    for op in graph["operations"]:
        assert set(refs((op["args"], op["kwargs"]))) <= names, op
        for field in ("reads", "inputs", "dependencies"):
            assert set(op[field]) <= names, (field, op)
    assert all(value.get("alias_of") in names or not value.get("alias_of") for value in graph["values"].values())


def _merge_graph(with_view):
    shape = (2, 3, 8) if with_view else (6, 8)
    values = {name: _value(name, dims, dtype) for name, dims, dtype in (
        ("b", shape, "bfloat16"), ("s", shape[:-1] + (1,), "float32"),
        ("idx", (6,), "int64"), ("m", shape, "float32"), ("out", (3, 8), "float32"))}
    ops = [_op("m", scaled_merge.MUL, [_r("b"), _r("s")], ["b", "s"])]
    source = "m"
    if with_view:
        values["view"] = _value("view", (6, 8), alias="m")
        ops.append(_op("view", "aten.view.default", [_r("m"), [6, 8]], ["m"], view=True))
        source = "view"
    ops.append(_op("out", scaled_merge.MERGE, [_r(source), _r("idx"), 3, -1, False], [source, "idx"]))
    return {"inputs": ["b", "s", "idx"], "outputs": ["out"], "values": values, "operations": ops}


@pytest.mark.parametrize("with_view", [False, True])
def test_scaled_merge_preserves_references_and_fp32_contributions(with_view):
    import torch
    graph = _merge_graph(with_view)
    saved = copy.deepcopy(graph)
    rewritten, report = scaled_merge.rewrite(graph)
    assert graph == saved and report and "m" not in rewritten["values"]
    _check_refs(rewritten)
    merge = rewritten["operations"][-1]
    assert merge["args"][0] == _r("view" if with_view else "b")
    assert merge["kwargs"]["latch_row_scale"] == _r("s")
    shape = graph["values"]["b"]["shape"]
    b = (torch.arange(math.prod(shape)).reshape(shape).float() / 7).to(torch.bfloat16)
    scale = torch.tensor([-.3, .1, 1.3, -2.1, .75, 0.]).reshape(shape[:-1] + [1])
    before = (b * scale).reshape(6, 8)
    after = b.float().reshape(6, 8) * scale.reshape(6, 1)
    assert torch.equal(before, after)
    idx = torch.tensor([0, 1, 1, 2, 0, 2])
    expected = torch.ops.aten.embedding_dense_backward.default(before, idx, 3, -1, False)
    actual = torch.ops.aten.embedding_dense_backward.default(after, idx, 3, -1, False)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("exposure", ["output", "other_reader", "nonidentity_view"])
def test_scaled_merge_refuses_exposed_or_reordered_products(exposure):
    graph = _merge_graph(True)
    if exposure == "output":
        graph["outputs"].append("m")
    elif exposure == "other_reader":
        graph["values"]["other"] = _value("other", (2, 3, 8))
        graph["operations"].append(_op("other", "aten.neg.default", [_r("m")], ["m"]))
    else:
        graph["values"]["view"]["strides"] = [1, 6]
    rewritten, report = scaled_merge.rewrite(graph)
    assert rewritten is graph and report is None


@pytest.mark.parametrize("width", [8, 2048])
def test_normal_compose_scaled_merge_keeps_segmented_shape_refusal(width):
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    from megakernel_lab.operation_compiler.lower import lower
    graph = _merge_graph(False)
    for name in ("b", "m", "out"):
        shape = graph["values"][name]["shape"]
        shape[-1] = width
        graph["values"][name]["strides"] = gather_cast._contiguous(shape)
    rewritten, report = scaled_merge.rewrite(graph)
    assert bool(report) is (width <= 1024)
    with policy("segmented_v1"):
        program = lower(graph, execution="compose")
    assert "out" in program.manifest["outputs"]
    if width > 1024:
        assert rewritten is graph


def _gather_graph():
    values = {name: _value(name, shape, dtype) for name, shape, dtype in (
        ("table", (5, 8), "float32"), ("idx", (6,), "int64"), ("g", (6, 8), "float32"),
        ("cast", (6, 8), "bfloat16"), ("out", (6, 8), "float32"))}
    ops = [_op("g", "aten.index_select.default", [_r("table"), 0, _r("idx")], ["table", "idx"]),
           _op("cast", gather_cast.CAST, [_r("g")], ["g"], kwargs={"dtype": "torch.bfloat16"}),
           _op("out", gather_cast.CAST, [_r("cast")], ["cast"], kwargs={"dtype": "torch.float32"})]
    return {"inputs": ["table", "idx"], "outputs": ["out"], "values": values, "operations": ops}


def test_gather_cast_moves_rounding_before_repeated_indices():
    import torch
    graph = _gather_graph()
    saved = copy.deepcopy(graph)
    rewritten, report = gather_cast.rewrite(graph)
    assert graph == saved and report and "cast" not in rewritten["values"]
    _check_refs(rewritten)
    assert rewritten["values"]["g"]["dtype"] == "bfloat16"
    table = torch.arange(40).reshape(5, 8).float() / 13
    idx = torch.tensor([4, 0, 4, 2, 1, 0])
    assert torch.equal(table[idx].bfloat16(), table.bfloat16()[idx])


@pytest.mark.parametrize("exposure", ["gather_output", "cast_output", "second_reader"])
def test_gather_cast_refuses_exposed_intermediates(exposure):
    graph = _gather_graph()
    if exposure.endswith("output"):
        graph["outputs"].append("g" if exposure == "gather_output" else "cast")
    else:
        graph["values"]["other"] = _value("other", (6, 8))
        graph["operations"].append(_op("other", "aten.neg.default", [_r("g")], ["g"]))
    assert gather_cast.rewrite(graph) == (graph, None)


def test_widen_cast_elision_keeps_fp32_results_and_rejects_views():
    graph = {"inputs": ["b", "s"], "outputs": ["out"], "values": {
        "b": _value("b", (6, 8), "bfloat16"), "s": _value("s", (6, 1)),
        "cast": _value("cast", (6, 8)), "out": _value("out", (6, 8))}, "operations": [
        _op("cast", widen_cast.CAST, [_r("b")], ["b"], kwargs={"dtype": "torch.float32"}),
        _op("out", "aten.mul.Tensor", [_r("cast"), _r("s")], ["cast", "s"])]}
    rewritten, report = widen_cast.rewrite(graph)
    assert report and "cast" not in rewritten["values"]
    assert rewritten["operations"][0]["args"][0] == _r("b")
    _check_refs(rewritten)
    graph["values"]["view"] = _value("view", (6, 8), alias="cast")
    assert widen_cast.rewrite(graph) == (graph, None)


def test_widen_cast_keeps_materialized_serial_prefix_but_elides_unrelated_casts():
    for materialized in (True, False):
        graph = {"inputs": ["b", "s"], "outputs": ["out", "ordinary_out"], "values": {
            "b": _value("b", (6, 8), "bfloat16"), "s": _value("s", (6, 1)),
            **{name: _value(name, (6, 8)) for name in ("cast", "out", "ordinary", "ordinary_out")}},
            "operations": [
                _op("cast", widen_cast.CAST, [_r("b")], ["b"], kwargs={"dtype": "torch.float32"}),
                _op("out", "aten.add.Tensor", [_r("cast"), _r("s")], ["cast", "s"]),
                _op("ordinary", widen_cast.CAST, [_r("b")], ["b"], kwargs={"dtype": "torch.float32"}),
                _op("ordinary_out", "aten.mul.Tensor", [_r("ordinary"), _r("s")], ["ordinary", "s"])],
            "metadata": {"serial_bf16_readout": {"weight_gradients": [{
                "partials": ["b", "other_partial"], "widened_partials": ["cast", "other_widened"],
                "source_only_removed_subtotals": [] if materialized else ["old_subtotal"]}]}}}
        original = copy.deepcopy(graph)
        rewritten, report = widen_cast.rewrite(graph)
        assert graph == original and rewritten["metadata"]["serial_bf16_readout"] == original["metadata"]["serial_bf16_readout"]
        assert ("cast" in rewritten["values"]) is materialized
        assert "ordinary" not in rewritten["values"]
        assert {row["cast"] for row in report["rewritten"]} == ({"ordinary"} if materialized else {"cast", "ordinary"})
        _check_refs(rewritten)


@pytest.mark.parametrize("rewrite", [gather_cast.rewrite, widen_cast.rewrite, scaled_merge.rewrite])
def test_optional_rewrites_preserve_list_valued_ir(rewrite):
    graph = _gather_graph()
    graph["values"] = list(graph["values"].values())
    rewritten, report = rewrite(graph)
    assert rewritten is graph and report is None


def _dispatch_merge_graph():
    """Small instance of the 25M recipe's explicit BF16 dispatch/weighted merge."""
    graph = _merge_graph(False)
    values = graph['values']
    values.update(table=_value('table', (5, 8)), gidx=_value('gidx', (6,), 'int64'),
                  gather=_value('gather', (6, 8)), wide=_value('wide', (6, 8)))
    graph['inputs'] = ['table', 'gidx', 's', 'idx']
    graph['operations'][0]['args'][0] = _r('wide')
    for key in ('inputs', 'reads', 'dependencies'):
        graph['operations'][0][key] = ['wide', 's']
    graph['operations'][:0] = [
        _op('gather', 'aten.index_select.default', [_r('table'), 0, _r('gidx')], ['table', 'gidx']),
        _op('b', gather_cast.CAST, [_r('gather')], ['gather'], kwargs={'dtype': 'torch.bfloat16'}),
        _op('wide', widen_cast.CAST, [_r('b')], ['b'], kwargs={'dtype': 'torch.float32'})]
    return graph


def test_strict_normal_compose_keeps_static_dispatch_and_merge_rewrites(monkeypatch):
    import importlib
    lower_module = importlib.import_module('megakernel_lab.operation_compiler.lower')
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    original = lower_module._rewrite_compose_casts
    seen = []

    def observed(graph, **kwargs):
        result = original(graph, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(lower_module, '_rewrite_compose_casts', observed)
    with policy('segmented_v1'):
        program = lower_module.lower(_dispatch_merge_graph(), execution='compose', precision_policy='strict')
    assert program.manifest['outputs'] == ['out']
    assert len(seen) == 1
    rewritten = seen[0]
    assert all(rewritten['metadata'][name]['rewritten'] for name in ('gather_cast', 'widen_cast', 'scaled_merge'))
    assert not {'b', 'wide', 'm'} & set(rewritten['values'])
    _check_refs(rewritten)


@pytest.mark.parametrize('contract', ['precision', 'dynamic', 'retain_all', 'reference', 'architecture'])
def test_optional_cast_rewrites_preserve_prior_lowering_contracts(contract):
    from megakernel_lab.operation_compiler.indexed_reduction import policy
    from megakernel_lab.operation_compiler.lower import _rewrite_compose_casts
    from megakernel_lab.operation_compiler.precision import BF16_PROJECTIONS
    graph = _dispatch_merge_graph()
    kwargs = dict(retain_all=False, policy='auto', target_arch='sm_90')
    if contract == 'precision':
        graph['metadata'] = {'precision_policy': BF16_PROJECTIONS.describe()}
    elif contract == 'dynamic':
        graph['metadata'] = {'dynamic_batch': {'maximum': 6, 'value_axes': {'gather': 0, 'b': 0, 'wide': 0}}}
    elif contract == 'retain_all':
        kwargs['retain_all'] = True
    elif contract == 'reference':
        kwargs['policy'] = 'reference'
    else:
        kwargs['target_arch'] = 'sm_80'
    before = copy.deepcopy(graph)
    with policy('segmented_v1'):
        result = _rewrite_compose_casts(graph, **kwargs)
    assert result is graph and graph == before
