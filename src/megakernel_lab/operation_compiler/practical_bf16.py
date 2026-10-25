"""Execution permission for BF16 programs, independent of the captured learning recipe.

This permission does not insert casts into a forward, derivative or optimizer.
Only already-BF16 products may use BF16 tensor-core bodies; FP32 products remain
FP32. It permits different valid reduction association, padding, BF16 partial
rounding and the existing practical activation implementations. Qualification
against independent numerical references belongs to the complete workload.
"""
from __future__ import annotations

import copy

KEY = "execution_precision"
CONTRACT = "practical_bf16_v1"
CHUNKS = "practical_bf16_weight_chunks"


def validate(policy, chunk_k=None):
    if policy not in (None, "strict", "practical"):
        raise ValueError("precision_policy must be strict or practical")
    if chunk_k is not None:
        if policy != "practical":
            raise ValueError("bf16_weight_chunk_k requires precision_policy='practical'")
        if type(chunk_k) is not int or chunk_k <= 0 or chunk_k % 16:
            raise ValueError("bf16_weight_chunk_k must be a positive multiple of 16")
    return "strict" if policy is None else policy


def describe(chunk_k=None):
    return dict(contract=CONTRACT, policy="practical", bf16_weight_chunk_k=chunk_k,
        operands="captured dtypes unchanged; BF16 products use FP32 accumulation",
        permitted=["full-rate BF16 matrix instructions", "zero-padded execution extents",
                   "reassociated reductions", "BF16 matrix partials accumulated in FP32",
                   "existing practical activation approximations", "rowwise gather/product commutation"],
        preserved=["mathematical objective", "routing and tie-selection rule", "data order",
                   "optimizer roles and state updates", "captured FP32 products and master state"],
        qualification="explicit execution permission; independent numerical and learning validation required")


def bind(graph, policy, chunk_k=None):
    """Copy only metadata; reject conflicting arithmetic rather than silently discard it."""
    selected = validate(policy, chunk_k)
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})
    previous = metadata.get(KEY)
    if previous is not None:
        if not isinstance(previous, dict) or previous != describe(previous.get("bf16_weight_chunk_k")):
            raise ValueError("unknown or modified execution precision contract")
        validate(previous.get("policy"), previous.get("bf16_weight_chunk_k"))
        if selected != "practical":
            raise ValueError("practical BF16 graph requires precision_policy='practical'")
        if chunk_k is None:
            chunk_k = previous.get("bf16_weight_chunk_k")
        elif previous.get("bf16_weight_chunk_k") != chunk_k:
            raise ValueError("bf16_weight_chunk_k differs from the bound graph")
    if selected != "practical":
        return graph
    incompatible = [name for name in ("serial_bf16_readout", "matrix_accumulation_contract",
                                      "matrix_splitk_contract")
                    if metadata.get(name) not in (None, "legacy")]
    if incompatible:
        raise ValueError("practical BF16 conflicts with explicit historical matrix arithmetic: "
                         + ", ".join(incompatible))
    metadata = dict(metadata, **{KEY: describe(chunk_k)})
    result = copy.copy(graph)
    if isinstance(result, dict):
        result["metadata"] = metadata
    else:
        result.metadata = metadata
    return result


def chunk_weights(graph):
    """Split private BF16 weight-like contractions after vocabulary padding.

    Selection uses operation, layout, dtype and exclusive widening dataflow.
    K must exceed the selected chunk and the shorter output extent. Square
    Newton--Schulz products are outside this rule. No product with a BF16
    consumer, escaping partial, or nontransparent intermediate is rewritten.
    """
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})
    existing = chunk_record(graph)
    contract = metadata.get(KEY)
    chunk_k = contract.get("bf16_weight_chunk_k") if contract else None
    if chunk_k is None:
        return graph, None
    validate(contract.get("policy"), chunk_k)
    from . import vocab_head as head
    from .serial_bf16 import _chunk_weight
    to_dict = getattr(graph, "to_dict", None)
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    values = data["values"]
    output_names = set(head._outputs(data))
    # State writes and passive alias readers also make an intermediate public.
    def names(item):
        if isinstance(item, str):
            return {item}
        if isinstance(item, dict):
            return set().union(*(names(x) for x in item.values())) if item else set()
        if isinstance(item, (tuple, list)):
            return set().union(*(names(x) for x in item)) if item else set()
        return set()
    output_names.update(names(data.get("outputs", {})))
    output_names.update(names(data.get("state_writes", {})))
    output_names.update(data.get("state_writes", {}))
    output_names.update(n for n, v in values.items() if v.get("output_roles"))
    report = existing or dict(contract=CONTRACT, chunk_k=chunk_k, products=[])
    previous_count = len(report["products"])
    views = head._TRANSPOSING_VIEWS | {"aten.slice.Tensor"}
    for op in list(data["operations"]):
        if op.get("target") != "aten.mm.default" or op.get("kwargs") or len(op.get("args", ())) != 2:
            continue
        a, b = (head._ref(x) for x in op["args"])
        outputs = op.get("outputs", ())
        if a is None or b is None or len(outputs) != 1:
            continue
        out = outputs[0]
        if any(values[x].get("dtype") != "bfloat16" for x in (a, b, out)):
            continue
        av, bv = values[a], values[b]
        if len(av["shape"]) != 2 or len(bv["shape"]) != 2:
            continue
        m, k = av["shape"]
        kb, n = bv["shape"]
        if (kb != k or k <= chunk_k or k <= min(m, n)
                or av["strides"][0] != 1 or av["strides"][1] == 1
                or bv["strides"][1] != 1 or bv["strides"][0] == 1
                or list(values[out]["shape"]) != [m, n]):
            continue
        _, consumers = head._index(data)
        current, chain, seen = out, [], set()
        widening = None
        while current not in seen and current not in output_names:
            seen.add(current)
            users = consumers.get(current, [])
            if len(users) != 1:
                break
            user = users[0]
            user_outputs = user.get("outputs", ())
            if len(user_outputs) != 1:
                break
            name = user_outputs[0]
            # An alias absent from the explicit consumer chain is still a use.
            if any(v.get("alias_of") == current and alias != name for alias, v in values.items()):
                break
            if user["target"] == "aten._to_copy.default":
                if (values[name]["dtype"] == "float32"
                        and list(values[name]["shape"]) == list(values[current]["shape"])):
                    widening = name
                break
            if user["target"] not in views or values[name]["dtype"] != "bfloat16":
                break
            chain.append(name)
            current = name
        if widening is None:
            continue
        plan = dict(operation=op, a=a, b=b, output=out, shape=[m, n, k], views=chain, cast=widening,
                    head=dict(product=out, dlogits=a, root=b))
        # Reuse the transparent ATen slice/product/widen/add construction. The
        # historical helper's product bindings are intentionally NOT installed:
        # ordinary full-rate matrix selection applies to every new partial.
        row = _chunk_weight(data, plan, chunk_k, {})
        row.update(shape=[m, n, k], instruction_selection="ordinary full-rate BF16 planner")
        report["products"].append(row)
    if len(report["products"]) == previous_count:
        return graph, None
    data.setdefault("metadata", {})[CHUNKS] = report
    return (type(graph).from_dict(data) if to_dict is not None else data), report


def chunk_record(graph):
    """Return the validated complete inventory, including a prior rewrite.

    The pass's second return value describes whether this invocation changed
    the graph. Prepared manifests must instead retain this persistent record.
    Copy it so emitted metadata cannot mutate a caller's already-bound IR.
    """
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})
    if metadata.get(CHUNKS) is None:
        return None
    validate_chunks(graph)
    return copy.deepcopy(metadata[CHUNKS])


def validate_artifact(receipt, training_ir, manifests):
    """Bind permission to hash-checked IR and every emitted region at load time."""
    from .square_sum_association import validate_artifact as validate_square_sums
    validate_square_sums(receipt, training_ir, manifests)
    record = receipt.get(KEY)
    if record != training_ir.get("metadata", {}).get(KEY):
        raise ValueError("distributed execution precision differs from its training IR")
    if receipt.get(CHUNKS) != training_ir.get("metadata", {}).get(CHUNKS):
        raise ValueError("distributed BF16 chunk record differs from its training IR")
    validate_chunks(training_ir)
    if record is None:
        if receipt.get("precision_policy") == "practical":
            raise ValueError("practical distributed artifact is missing its execution precision contract")
        if any(manifest.get(KEY) is not None or manifest.get(CHUNKS) is not None for manifest in manifests):
            raise ValueError("region execution precision is absent from its distributed contract")
        return
    required = {"training-ir.json", *(region["directory"] + "/manifest.json"
                                     for region in receipt.get("regions", []))}
    if not required <= set(receipt.get("files", {})):
        raise ValueError("practical execution requires hash-bound training IR and every region manifest")
    if len(manifests) != len(receipt.get("regions", [])):
        raise ValueError("practical region manifest inventory differs")
    if not isinstance(record, dict) or record != describe(record.get("bf16_weight_chunk_k")):
        raise ValueError("unsupported distributed execution precision contract")
    validate(record.get("policy"), record.get("bf16_weight_chunk_k"))
    if receipt.get("precision_policy") != "practical":
        raise ValueError("distributed execution precision request differs")
    for manifest in manifests:
        if manifest.get(KEY) != record:
            raise ValueError("region execution precision differs from distributed contract")


def validate_chunks(graph, record=None):
    """Prove complete BF16 partial coverage and ordered FP32 accumulation.

    Returns partial-name -> logical-product proof for consumers inspecting a
    partitioned graph. Exact widening elision is admitted only at the declared
    FP32-result add; every contraction slice and every sum operand is checked.
    No shape-selected inference can substitute for the bound transformation.
    """
    data = graph if isinstance(graph, dict) else graph.to_dict()
    metadata = data.get("metadata", {})
    declared = metadata.get(CHUNKS)
    if record is not None and record != declared:
        raise ValueError("practical BF16 chunk record differs from retained IR")
    if declared is None:
        return {}
    execution = metadata.get(KEY, {})
    chunk_k = execution.get("bf16_weight_chunk_k")
    if (execution != describe(chunk_k) or chunk_k is None or declared.get("contract") != CONTRACT
            or declared.get("chunk_k") != chunk_k or not declared.get("products")):
        raise ValueError("practical BF16 chunk declaration lacks its execution contract")
    validate("practical", chunk_k)
    values = data["values"]
    producers = {name: op for op in data["operations"] for name in op["outputs"]}
    from .vocab_head import _ref
    result, logical = {}, set()
    def require(condition, reason):
        if not condition:
            raise ValueError("practical BF16 chunk proof: " + reason)
    def typed(name, shape, dtype):
        value = values.get(name, {})
        return list(value.get("shape", ())) == list(shape) and value.get("dtype") == dtype
    def operation(name, target, args, kwargs=None):
        op = producers.get(name, {})
        require(op.get("target") == target and op.get("outputs") in ([name], (name,))
                and list(op.get("args", ())) == args and op.get("kwargs", {}) == (kwargs or {}),
                "operation/operands differ: " + str(name))
    for row in declared["products"]:
        m, n, k = row["shape"]
        product, partials, widened, sums = (row[key] for key in
            ("product", "partials", "widened_partials", "subtotal_additions"))
        intervals = [[start, min(start + chunk_k, k)] for start in range(0, k, chunk_k)]
        require(product not in logical and row.get("chunk_rows") == chunk_k and k > chunk_k
                and row.get("token_intervals") == intervals and len(partials) == len(intervals)
                and len(widened) == len(partials) and len(sums) == len(partials)-1,
                "logical identity or full contraction coverage differs")
        require(row.get("accumulation") == "ascending FP32 addition of individually BF16-rounded partials"
                and row.get("instruction_selection") == "ordinary full-rate BF16 planner",
                "partial arithmetic differs")
        logical.add(product)
        parents, slices, terms = None, [], []
        for partial, wide, (start, stop) in zip(partials, widened, intervals):
            require(partial not in result and typed(partial, [m, n], "bfloat16"), "partial identity or dtype differs")
            op = producers.get(partial, {})
            operands = list(op.get("args", ()))
            require(len(operands) == 2, "partial matrix arity differs")
            left, right = map(_ref, operands)
            operation(partial, "aten.mm.default", [{"ref": left}, {"ref": right}])
            source_pair = []
            for name, axis, shape, whole in ((left, 1, [m, stop-start], [m, k]),
                                             (right, 0, [stop-start, n], [k, n])):
                sliced = producers.get(name, {})
                args = list(sliced.get("args", ()))
                require(len(args) == 5, "contraction slice arity differs")
                source = _ref(args[0])
                operation(name, "aten.slice.Tensor", [{"ref": source}, axis, start, stop, 1])
                require(typed(source, whole, "bfloat16") and typed(name, shape, "bfloat16"),
                        "slice or source typed extent differs")
                base, view = values[source], values[name]
                require(view.get("alias_of") == source and view["strides"] == base["strides"]
                        and view.get("storage_offset", 0) == base.get("storage_offset", 0)+start*base["strides"][axis],
                        "contraction slice layout differs")
                source_pair.append(source)
                slices.append(name)
            require(parents is None or parents == source_pair, "partial contractions use different original operands")
            parents = source_pair
            if wide in values or wide in producers:
                operation(wide, "aten._to_copy.default", [{"ref": partial}], {"dtype": "float32"})
                require(typed(wide, [m, n], "float32"), "partial widening dtype differs")
                terms.append(wide)
            else:
                # widen_cast's exact implicit conversion at an FP32 add.
                terms.append(partial)
        total = terms[0]
        for name, term in zip(sums, terms[1:]):
            operation(name, "aten.add.Tensor", [{"ref": total}, {"ref": term}])
            require(typed(name, [m, n], "float32"), "subtotal does not accumulate in FP32")
            total = name
        operation(product, "aten._to_copy.default", [{"ref": total}], {"dtype": "float32"})
        require(typed(product, [m, n], "float32"), "logical result does not preserve FP32 subtotal")
        proof = dict(product=product, shape=[m, n, k], inputs=parents, partials=list(partials),
                     slices=slices, intervals=intervals, sums=list(sums),
                     accumulation="BF16-rounded full-K disjoint partials, ascending FP32 sum")
        for partial in partials:
            result[partial] = proof
    return result
