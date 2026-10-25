"""Independent source-audit helper retained for compiler regression coverage.

Unchanged function extracted from TinyMoE tested commit
adde6ca52b556df8585edfb6c7ade3a241964620,
src/megakernel_lab/moe4/arithmetic_build_audit.py. The excluded MoE4
application and its CLI are not runtime/test dependencies of the public package.
"""

def softmax_vjp_matrix_dtypes(graph, *, chunk_products=None):
    """Inspect typed AD products downstream of softmax, without name matching."""
    producers = {name: op for op in graph["operations"] for name in op["outputs"]}
    softmax_outputs = {name for op in graph["operations"]
        if op["target"] == "aten._softmax_backward_data.default" for name in op["outputs"]}
    view_targets = {"aten.view.default", "aten._unsafe_view.default", "aten.reshape.default",
        "aten.transpose.int", "aten.t.default", "aten.permute.default", "aten._to_copy.default",
        "aten.clone.default", "aten.detach.default", "aten.contiguous.default"}
    practical = graph.get("metadata", {}).get("execution_precision", {}).get("policy") == "practical"
    chunk_products = chunk_products or {}
    if chunk_products and not practical:
        raise ValueError("chunked softmax derivatives require practical execution permission")
    validated_slices = {name for row in chunk_products.values() for name in row["slices"]}
    def origin(name):
        seen = set()
        while name not in seen:
            if name in softmax_outputs:
                return name
            seen.add(name)
            op = producers.get(name)
            if practical and name in validated_slices:
                if op is None or op["target"] != "aten.slice.Tensor":
                    raise ValueError("proved contraction slice is absent from the derivative region")
                name = op["args"][0]["ref"]
                continue
            # Practical mode admits the explicitly recorded gather/product
            # commutation. Follow its data operand, never the integer index,
            # while still verifying the actual typed matrix derivatives below.
            if practical and op is not None and op["target"] == "aten.index_select.default":
                args = op.get("args", [])
                if len(args) >= 3 and args[1] == 0 and isinstance(args[0], dict) and set(args[0]) == {"ref"}:
                    name = args[0]["ref"]
                    continue
            if op is None or op["target"] not in view_targets or len(op["inputs"]) != 1:
                return None
            name = op["inputs"][0]
        return None
    result, partial_groups = [], {}
    for op in graph["operations"]:
        if op["target"] != "aten.mm.default" or not any(origin(n) for n in op["inputs"]):
            continue
        row = dict(operation=op["name"], inputs=op["inputs"], outputs=op["outputs"],
            input_dtypes=[graph["values"][n]["dtype"] for n in op["inputs"]],
            output_dtypes=[graph["values"][n]["dtype"] for n in op["outputs"]])
        partial = op["outputs"][0]
        if partial in chunk_products:
            proof = chunk_products[partial]
            partial_groups.setdefault(proof["product"], []).append(row)
        else:
            result.append(row)
    for product, rows in partial_groups.items():
        proof = chunk_products[rows[0]["outputs"][0]]
        if ([row["outputs"][0] for row in rows] != proof["partials"]
                or any(row["input_dtypes"] != ["bfloat16", "bfloat16"]
                       or row["output_dtypes"] != ["bfloat16"] for row in rows)):
            raise ValueError("logical softmax VJP lacks every ordered typed BF16 partial")
        result.append(dict(operation=product, inputs=proof["inputs"], outputs=[product],
            input_dtypes=["bfloat16", "bfloat16"], output_dtypes=["float32"],
            partial_products=rows, partial_output_dtype="bfloat16", chunk_proof=proof))
    return result

