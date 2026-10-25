"""Proof for retaining an exact captured BF16 projection schedule.

Practical flat precision permits approximate bodies; it does not require them.
This path keeps the stronger, exact activation contract already captured before
AD. It admits only the existing POPSC body family and explicit cast boundaries.
"""
from __future__ import annotations


ARITHMETIC = ("explicit BF16 operands/output; FP32 MMA accumulation from +0; ascending k16; no split-K; "
              "one cvt.rn.bf16x2 rounding (bitwise = pinned 64x64 body)")


def exact_projection_capture(graph):
    from .lower import _field
    from .precision import PrecisionPolicy, COVERED
    policy = _field(graph, "metadata", {}).get("precision_policy")
    if policy is None:
        return bool(explicit_projection_contract(graph))
    if not isinstance(policy, dict):
        return False
    products = policy.get("products")
    if (not isinstance(products, (tuple, list)) or not products
            or any(not isinstance(item, str) for item in products)
            or not set(products) <= set(COVERED) or len(set(products)) != len(products)):
        return False
    expected = PrecisionPolicy("", products=tuple(products)).describe()
    # Names and capture receipts are descriptive; unknown semantic fields are
    # not interpreted as permission to change the declared arithmetic.
    if set(policy) - set(expected) - {"covered_calls", "calls", "exact_round_trips_removed"}:
        return False
    return all(policy.get(key) == value for key, value in expected.items() if key != "name")


def explicit_projection_contract(graph):
    """Recognize existing BF16 projection arithmetic without changing the IR.

    This is a body-selection candidate, not a new forward precision policy.
    The differentiated graph remains authoritative, including cast sharing and
    each gradient accumulation. Actual bodies/layouts/cast expressions must
    still pass verify_legacy_bf16. Batched/mixed products and BF16 arithmetic
    between projections remain outside this narrow equivalence class.
    """
    from .lower import _field, _ref
    metadata = _field(graph, "metadata", {})
    if (metadata.get("precision_policy") is not None or metadata.get("backward_autocast")
            or "distributed_event" in metadata):
        return None
    values = _field(graph, "values", {})
    if not isinstance(values, dict):
        values = {_field(value, "name"): value for value in values}
    names = set(values)
    dtype = lambda n: str(_field(values.get(n, {}), "dtype", "")).removeprefix("torch.")
    products, casts = [], []
    views = {"t", "transpose", "permute", "view", "reshape", "_unsafe_view", "squeeze", "unsqueeze", "expand", "alias", "detach"}
    for op in _field(graph, "operations", _field(graph, "nodes", ())):
        base = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        output = _field(op, "outputs", [_field(op, "name")])[0]
        args = _field(op, "args", ())
        if base in {"mm", "matmul", "bmm", "addmm"}:
            operands = [_ref(arg, names) for arg in args[:2]]
            if (base not in {"mm", "matmul"} or len(operands) != 2 or None in operands
                    or any(dtype(n) != "bfloat16" or len(_field(values[n], "shape", ())) != 2
                           for n in [*operands, output])):
                return None
            products.append(output)
        elif dtype(output) == "bfloat16" and base not in views | {"_to_copy"}:
            return None
        if base == "_to_copy" and args:
            source = _ref(args[0], names)
            if "bfloat16" in (dtype(source), dtype(output)):
                if (dtype(source), dtype(output)) not in (("float32", "bfloat16"), ("bfloat16", "float32")):
                    return None
                casts.append(output)
    return ({"origin": "explicit captured ATen graph; unchanged differentiation and state",
             "products": products, "casts": casts,
             "product_arithmetic": "BF16 operands, FP32 accumulation, one BF16 result rounding"}
            if products and casts else None)


def verify_legacy_bf16(graph, common, legacy, *, contracts=None):
    """Check actual emitted products, views, roundings and scalar cast recipes.

Different POPSC tile partitions keep the same ascending k16 MMA chain. No
P5C, split-K, tail/edge adapter, prologue absorption or legacy epilogue is
    admitted by this proof. Common exact BF16->FP32 epilogues and subsequent
    FP32 gradient additions may materialize in the retained schedule, keeping
    each per-use BF16 rounding before its ordered FP32 addition.
"""
    from .lower import _field, _ref, _ident
    from .matrix_rules import BF16_TILES
    if not exact_projection_capture(graph):
        return None, "captured BF16 projection arithmetic is not the exact supported contract"
    layouts = {**common.get("elided_layouts", {}), **common["layouts"]}
    operations = _field(graph, "operations", _field(graph, "nodes", ()))
    names = set(layouts)
    values = _field(graph, "values", {})
    if not isinstance(values, dict):
        values = {_field(value, "name"): value for value in values}
    names.update(values)
    opname = lambda op: _field(op, "outputs", [_field(op, "name")])[0]
    base = lambda op: str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
    by_name = {opname(op): op for op in operations}
    emitted = (contracts or {}).get("bf16", [])
    casts = (contracts or {}).get("bf16_casts", [])
    additions = (contracts or {}).get("bf16_adds", [])
    products = [op for op in operations if base(op) in ("mm", "bmm", "matmul", "addmm")]
    if not products or len(emitted) != len(products):
        return None, "established BF16 product coverage is not individually proven"
    if legacy.get("performance_schedule", {}).get("precision_policy") != "strict":
        return None, "established BF16 retention requires effective strict flat arithmetic"
    for header in ("popsc_mm.cuh", "bgemm.cuh", "tgemm.cuh", "core.cuh", "epi.cuh"):
        if (not common.get("header_hashes", {}).get(header)
                or common["header_hashes"][header] != legacy.get("header_hashes", {}).get(header)):
            return None, "established BF16 calculation header does not match: " + header
    proof, addition_proof = [], []
    for op in products:
        name = opname(op)
        found = [item for item in emitted if item.get("name") == name]
        phases = [p for p in common["phases"] if p.get("matrix_spec", {}).get("out") == _ident(name)]
        if len(found) != 1 or len(phases) != 1:
            return None, "established BF16 output lacks unique emitted-body evidence: " + name
        item, phase = found[0], phases[0]
        spec, old = phase["matrix_spec"], item.get("spec", {})
        if (spec != old or spec.get("dtype") != "bfloat16" or spec.get("engine")
                or spec.get("tail") or set(spec) != {"dtype", "a", "b", "out", "m", "n", "k", "batch", "strides", "ak", "bkc"}
                or base(op) == "addmm"):
            return None, "established BF16 emitted matrix specification does not match: " + name
        members = phase.get("members", [])
        if len(members) != 1:
            return None, "common BF16 matrix lacks one declared member: " + name
        body = item.get("member", {})
        for member in (members[0], body):
            plan = member.get("matrix_plan", {})
            tile = tuple(plan.get(key) for key in ("bm", "bn", "wm", "wn"))
            if (tile not in BF16_TILES or plan.get("ks") not in (32, 64)
                    or member.get("engine") != f"popsc_bf16_{tile[0]}x{tile[1]}"
                    or member.get("arithmetic") != ARITHMETIC
                    or member.get("output") != spec["out"]
                    or member.get("matrix_shape") != [spec[k] for k in ("m", "n", "k")]
                    or member.get("batch") != spec["batch"]
                    or member.get("operand_layout") != {"a_k_contiguous": spec["ak"], "b_k_contiguous": spec["bkc"]}
                    or any(key in member for key in ("epilogue", "a_prologue", "b_prologue"))):
                return None, "established BF16 body or one-rounding arithmetic is unproven: " + name
        old_phase = legacy["phases"][item["phase"]]
        if (name not in old_phase.get("writes", [])
                or [(i.get("kind"), i.get("name")) for i in old_phase.get("items", [])].count(("gemm", name)) != 1):
            return None, "established BF16 rounded output is not materialized: " + name
        args = _field(op, "args", ())
        if [_ref(arg, names) for arg in args[:2]] != [spec["a"], spec["b"]]:
            return None, "established BF16 ordered operands differ: " + name
        for operand in (spec["a"], spec["b"], name):
            before, after = layouts.get(operand), legacy["layouts"].get(operand)
            if not before or before != after or before.get("dtype") != "bfloat16":
                return None, "established BF16 operand/output view does not match: " + operand
        epi = phase.get("bf16_epilogue")
        if epi:
            records = [row for row in common.get("group_rules", {}).get("bf16_epilogues", []) if row.get("product") == name]
            if (len(records) != 1 or records[0].get("arithmetic") != "unchanged consumer expressions on the stored BF16 product value"
                    or records[0].get("consumer_names") != epi.get("consumers")
                    or records[0].get("product_stored") != epi.get("product_stored")):
                return None, "common BF16 epilogue rounding evidence does not match: " + name
            widened = set()
            for consumer in epi.get("consumers", []):
                op_ = by_name.get(consumer)
                refs = [_ref(arg, names) for arg in _field(op_, "args", ())]
                if (op_ is not None and base(op_) == "_to_copy" and refs == [name]
                        and layouts.get(consumer, {}).get("dtype") == "float32"):
                    widened.add(consumer)
                    continue
                rows = [row for row in additions if row.get("name") == consumer]
                if (op_ is None or base(op_) != "add" or len(refs) != 2 or None in refs
                        or _field(op_, "kwargs", {}).get("alpha", 1) != 1 or not widened.intersection(refs)
                        or len(rows) != 1 or rows[0].get("inputs") != refs
                        or rows[0].get("expression") != "(operand0 + (operand1 * 1.0f))"
                        or any(layouts.get(ref, {}).get("dtype") != "float32" for ref in [*refs, consumer])
                        or rows[0].get("input_layouts") != [layouts[ref] for ref in refs]
                        or rows[0].get("output_layout") != layouts[consumer]):
                    return None, "common BF16 epilogue is not an exact widening and ordered FP32 addition: " + name
                widened.add(consumer)
                addition_proof.append({"output": consumer, "inputs": refs, "arithmetic": rows[0]["expression"]})
        proof.append({"output": name, "legacy_phase": item["phase"], "engine": body["engine"],
                      "arithmetic": ARITHMETIC})
    cast_proof = []
    for op in operations:
        if base(op) != "_to_copy":
            continue
        name = opname(op)
        source = _ref(_field(op, "args", [None])[0], names)
        dtype = lambda value: str(_field(values.get(value, {}), "dtype", "")).removeprefix("torch.")
        before, after = dtype(source), dtype(name)
        if "bfloat16" not in (before, after):
            continue
        found = [item for item in casts if item.get("name") == name]
        expected = ("__float2bfloat16_rn(operand)" if (before, after) == ("float32", "bfloat16")
                    else "__bfloat162float(operand)" if (before, after) == ("bfloat16", "float32") else None)
        if (len(found) != 1 or expected is None or found[0].get("input") != source
                or found[0].get("expression") != expected
                or any(not layouts.get(key) or layouts[key] != found[0].get(field)
                       for key, field in ((source, "input_layout"), (name, "output_layout")))):
            return None, "established BF16 cast boundary is not proven: " + name
        cast_proof.append({"output": name, "input": source, "arithmetic": expected})
    if len(cast_proof) != len(casts) or not cast_proof:
        return None, "established BF16 cast coverage differs"
    return {"matrix_bodies": proof, "casts": cast_proof, "fp32_additions": addition_proof,
            "capture_precision": dict(_field(graph, "metadata", {}).get("precision_policy")
                                      or explicit_projection_contract(graph)),
            "capture_activation": "exact", "capture_product_rounding": "one BF16 round then exact widening"}, None
