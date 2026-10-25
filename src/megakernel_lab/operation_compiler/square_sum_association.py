"""Explicit practical FP32 association for full sums of squares.

Scalar norm terms use stable adjacent pairs. Long full scalar square reductions
use eight FP32 per-thread lanes before the existing CTA trees. Squared leaves,
all dtypes and mathematical consumers are unchanged. Qualification is separate.
"""
from __future__ import annotations

import copy
import hashlib
import math

KEY = "practical_square_sum_association"
CONTRACT = "balanced_fp32_square_sums_v1"
LANES = 8
MIN_ELEMENTS = 64 * 256


def validate_option(precision_policy, option):
    if option not in (None, "balanced"):
        raise ValueError("square_sum_association must be None or balanced")
    if option is not None and precision_policy != "practical":
        raise ValueError("square_sum_association requires precision_policy='practical'")


def _ref(arg):
    return arg.get("ref") if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _strings(item):
    if isinstance(item, str):
        return {item}
    if isinstance(item, dict):
        return set().union(*(_strings(x) for x in item.values())) if item else set()
    if isinstance(item, (tuple, list)):
        return set().union(*(_strings(x) for x in item)) if item else set()
    return set()


def _data(graph):
    return graph if isinstance(graph, dict) else graph.to_dict()


def _metadata(graph):
    return graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})


def _permission(data):
    from . import practical_bf16
    declared = data.get("metadata", {}).get(practical_bf16.KEY)
    if (not isinstance(declared, dict)
            or declared != practical_bf16.describe(declared.get("bf16_weight_chunk_k"))):
        raise ValueError("square-sum association requires bound practical execution permission")


def _index(data):
    producers = {name: op for op in data["operations"] for name in op["outputs"]}
    users = {}
    for op in data["operations"]:
        for name in _strings(op.get("args", ())) | _strings(op.get("kwargs", {})):
            users.setdefault(name, set()).add(op["name"])
    public = _strings(data.get("outputs", {})) | _strings(data.get("state_writes", {}))
    public.update(data.get("state_writes", {}))
    public.update(n for n, v in data["values"].items() if v.get("output_roles"))
    aliases = {v["alias_of"] for v in data["values"].values() if v.get("alias_of")}
    return producers, users, public, aliases


def _scalar(data, name):
    value = data["values"].get(name, {})
    return (value.get("dtype") == "float32" and value.get("shape") in ([], ())
            and not value.get("alias_of") and value.get("storage_offset", 0) == 0)


def _plain(op, name, target, arity):
    return (op.get("target") == target and op.get("outputs") in ([name], (name,))
            and len(op.get("args", ())) == arity and not op.get("kwargs")
            and op.get("effect", "pure") == "pure")


def _add(data, producers, name):
    op = producers.get(name, {})
    if not (_scalar(data, name) and _plain(op, name, "aten.add.Tensor", 2)):
        return None
    children = [_ref(arg) for arg in op["args"]]
    return children if all(child is not None and _scalar(data, child) for child in children) else None


def _leaf(data, producers, name):
    op = producers.get(name, {})
    if not (_scalar(data, name) and _plain(op, name, "aten.sum.default", 1)):
        return None
    source = _ref(op["args"][0])
    square = producers.get(source, {})
    if not _plain(square, source, "aten.mul.Tensor", 2):
        return None
    operand = _ref(square["args"][0])
    if operand is None or square["args"][0] != square["args"][1]:
        return None
    values = data["values"]
    if (values.get(source, {}).get("dtype") != "float32"
            or values.get(operand, {}).get("dtype") != "float32"
            or values[source].get("shape") != values[operand].get("shape")
            or not values[source].get("shape")
            or any(type(n) is not int or n <= 0 for n in values[source]["shape"])):
        return None
    return dict(sum=name, square=source, operand=operand, shape=list(values[source]["shape"]))


def _balanced(leaves, names):
    """Stable adjacent pairs, carrying an unpaired rightmost term unchanged."""
    pending, result, index = list(leaves), [], 0
    while len(pending) > 1:
        following = []
        for offset in range(0, len(pending)-1, 2):
            name = names[index]
            result.append(dict(name=name, left=pending[offset], right=pending[offset+1]))
            following.append(name)
            index += 1
        if len(pending) % 2:
            following.append(pending[-1])
        pending = following
    if index != len(names):
        raise ValueError("square-sum association addition inventory differs")
    return result


def validate(graph):
    """Validate retained association against typed scalar and squared-leaf IR."""
    if _metadata(graph).get(KEY) is None:
        return None
    data = _data(graph)
    record = data.get("metadata", {}).get(KEY)
    if record is None:
        return None
    _permission(data)
    def require(ok, why):
        if not ok:
            raise ValueError("square-sum association proof: " + why)
    require(isinstance(record, dict) and set(record) == {"contract", "trees", "full_sums", "thread_lanes"}
            and record["contract"] == CONTRACT and record["thread_lanes"] == LANES
            and (record["trees"] or record["full_sums"]), "invalid declaration")
    producers, users, public, aliases = _index(data)
    partial = "distributed_event" in data.get("metadata", {})
    full_names = [leaf["sum"] for leaf in record["full_sums"]]
    require(len(full_names) == len(set(full_names)), "duplicate full tensor sum")
    for leaf in record["full_sums"]:
        require(math.prod(leaf["shape"]) >= MIN_ELEMENTS, "full sum below long-reduction threshold")
        if partial and leaf["sum"] not in producers:
            continue
        require(_leaf(data, producers, leaf["sum"]) == leaf, "full squared reduction differs")
    occupied = set()
    for row in record["trees"]:
        leaves, original = row["leaves"], row["original_additions"]
        names = [entry["name"] for entry in original]
        leaf_names = [leaf["sum"] for leaf in leaves]
        require(len(leaves) >= 3 and len(set(leaf_names)) == len(leaves)
                and len(names) == len(leaves)-1 and len(set(names)) == len(names)
                and not occupied.intersection(names) and row["root"] == names[-1], "tree inventory differs")
        occupied.update(names)
        expanded = {name: [name] for name in leaf_names}
        for entry in original:
            require(entry["left"] in expanded and entry["right"] in expanded
                    and entry["name"] not in expanded, "original tree is not topological")
            expanded[entry["name"]] = expanded[entry["left"]] + expanded[entry["right"]]
        require(expanded[row["root"]] == leaf_names, "original tree leaf order/coverage differs")
        balanced = _balanced(leaf_names, names)
        require(row["additions"] == balanced, "stable adjacent-pair association differs")
        if partial and row["root"] not in producers:
            continue
        require(all((_scalar(data, leaf["sum"]) and leaf["sum"] in data["inputs"])
                    if partial and leaf["sum"] not in producers else
                    _leaf(data, producers, leaf["sum"]) == leaf for leaf in leaves), "squared reduction differs")
        for entry in balanced:
            name = entry["name"]
            require(_add(data, producers, name) == [entry["left"], entry["right"]], "typed scalar add differs")
            if name != row["root"]:
                expected_users = {producers[x["name"]]["name"] for x in balanced
                                  if name in (x["left"], x["right"])}
                require(name not in public and name not in aliases and users.get(name, set()) == expected_users,
                        "internal subtotal escapes")
    return copy.deepcopy(record)


def rewrite(graph, *, option=None):
    """Return an immutable rewrite and persistent proof, or the original graph.

    Enabling requires explicit practical permission. None preserves an unmarked
    graph or inherits an already bound record; it never erases its arithmetic.
    """
    if option not in (None, "balanced"):
        raise ValueError("square_sum_association must be None or balanced")
    if option is None and _metadata(graph).get(KEY) is None:
        return graph, None
    data = _data(graph)
    existing = validate(data)
    if existing is not None:
        return graph, existing
    if option is None:
        return graph, None
    _permission(data)
    producers, users, public, aliases = _index(data)
    full_sums = [leaf for name in producers if (leaf := _leaf(data, producers, name)) is not None
                 and math.prod(leaf["shape"]) >= MIN_ELEMENTS]
    candidates = {name for name in producers if _add(data, producers, name) is not None}
    candidate_operations = {producers[name]["name"] for name in candidates}
    rows, replacements, removed = [], {}, set()
    for root in producers:
        if root not in candidates or any(user in candidate_operations for user in users.get(root, ())):
            continue
        leaves, additions, seen = [], [], set()
        # Capture emits a left-deep chain for global gradient norms. Visit it
        # iteratively so model size cannot consume Python's recursion budget.
        # Exit frames retain the original left-to-right postorder and the same
        # no-repeated-node/private-subtotal proof as recursive traversal.
        pending, valid = [(root, None)], True
        while pending:
            name, children = pending.pop()
            if children is not None:
                additions.append(dict(name=name, left=children[0], right=children[1]))
                continue
            if name in seen:
                valid = False
                break
            seen.add(name)
            children = _add(data, producers, name)
            if children is None:
                leaf = _leaf(data, producers, name)
                if leaf is None:
                    valid = False
                    break
                leaves.append(leaf)
                continue
            if name != root and (name in public or name in aliases or len(users.get(name, ())) != 1):
                valid = False
                break
            pending.append((name, children))
            pending.extend((child, None) for child in reversed(children))
        if not valid or len(leaves) < 3:
            continue
        balanced = _balanced([leaf["sum"] for leaf in leaves], [add["name"] for add in additions])
        if balanced == additions:
            continue
        emitted = []
        for entry in balanced:
            op = copy.deepcopy(producers[entry["name"]])
            operands = [entry["left"], entry["right"]]
            op.update(args=[{"ref": name} for name in operands], inputs=operands, reads=operands,
                      dependencies=operands)
            emitted.append(op)
        replacements[root] = emitted
        removed.update(add["name"] for add in additions)
        rows.append(dict(root=root, leaves=leaves, original_additions=additions, additions=balanced))
    if not rows and not full_sums:
        return graph, None
    result = copy.deepcopy(data)
    result["operations"] = [emitted for op in data["operations"]
        for emitted in (replacements.get(op["outputs"][0], []) if op["outputs"][0] in removed else [copy.deepcopy(op)])]
    result.setdefault("metadata", {})[KEY] = dict(contract=CONTRACT, trees=rows,
                                                  full_sums=full_sums, thread_lanes=LANES)
    record = validate(result)
    return (type(graph).from_dict(result) if not isinstance(graph, dict) else result), record


def selected_full_sums(graph):
    """Typed local outputs selected by a previously validated proof."""
    if _metadata(graph).get(KEY) is None:
        return set()
    data = _data(graph)
    record = data.get("metadata", {}).get(KEY)
    if record is None:
        return set()
    produced = {name for op in data["operations"] for name in op["outputs"]}
    return {leaf["sum"] for leaf in record["full_sums"] if leaf["sum"] in produced}


def validate_artifact(receipt, training_ir, manifests):
    """Bind full typed IR, every region's permission and emitted lane inventory."""
    record = validate(training_ir)
    if receipt.get(KEY) != record:
        raise ValueError("distributed square-sum association differs from its training IR")
    if any(manifest.get(KEY) != record for manifest in manifests):
        raise ValueError("region square-sum association differs from distributed contract")
    emitted = [name for manifest in manifests for name in manifest.get("square_sum_lane_outputs", [])]
    expected = [] if record is None else [leaf["sum"] for leaf in record["full_sums"]]
    if len(emitted) != len(set(emitted)) or set(emitted) != set(expected):
        raise ValueError("emitted square-sum lane inventory differs from typed IR")


DEEP_ROUNDS = 3
DEEPX_ROUNDS = 8    # LATCH-B1-FAST ``fsdeepx``


def lane_loop(reduction_read):
    """CUDA body: preserve each FP32 leaf, shorten only thread-local addition."""
    declarations = "; ".join(f"float lane{lane} = 0.0f" for lane in range(LANES)) + ";"
    additions = "\n".join(f"    lane{lane} += latch_term(r + {lane}LL * blockDim.x);" for lane in range(LANES))
    tail = "\n".join(f"  if (r + {lane}LL * blockDim.x < hi_) lane{lane} += latch_term(r + {lane}LL * blockDim.x);"
                     for lane in range(LANES))
    combine = "((lane0 + lane1) + (lane2 + lane3)) + ((lane4 + lane5) + (lane6 + lane7))"
    from . import accumulation_rules
    deep = ""
    if accumulation_rules.active("fsdeep"):
        # LATCH-SELECTIVE-10 ``fsdeep``: three rounds' terms are loaded before any is added (24 independent loads
        # in flight per thread instead of 8); each lane then adds its three terms in the original round order, so
        # every lane sees the same summands in the same sequence. The 8-wide loop below finishes the rest.
        rounds = DEEPX_ROUNDS if accumulation_rules.active("fsdeepx") else DEEP_ROUNDS
        loads = "\n".join(f"    const float t{q} = latch_term(r + {q}LL * blockDim.x);" for q in range(rounds * LANES))
        adds = "\n".join(f"    lane{q % LANES} += t{q};" for q in range(rounds * LANES))
        deep = (f"  for (; r + {rounds * LANES - 1}LL * blockDim.x < hi_; r += {rounds * LANES}LL * blockDim.x) {{\n"
                f"{loads}\n{adds}\n  }}\n")
    return f'''  auto latch_term = [&](long long r) -> float {{ return {reduction_read}; }};
  {declarations}
  long long r = lo_ + threadIdx.x;
{deep}  for (; r + {LANES-1}LL * blockDim.x < hi_; r += {LANES}LL * blockDim.x) {{
{additions}
  }}
{tail}
  acc = {combine};'''


def marker(name):
    return "/* square_sum_lanes8:" + hashlib.sha256(name.encode()).hexdigest() + " */"


def annotate_program(graph, program):
    if _metadata(graph).get(KEY) is None:
        return program
    data = _data(graph)
    record = data.get("metadata", {}).get(KEY)
    if record is None:
        return program
    expected = sorted(selected_full_sums(data))
    actual = [name for name in expected if marker(name) in program.source]
    if actual != expected:
        from .lower import LoweringError
        raise LoweringError("selected square sums require emitted eight-lane full reductions: "
                            + ", ".join(sorted(set(expected)-set(actual))))
    program.manifest[KEY] = copy.deepcopy(record)
    program.manifest["square_sum_lane_outputs"] = actual
    program.manifest["square_sum_distributed_region"] = "distributed_event" in data.get("metadata", {})
    return program


def validate_manifest(manifest, source):
    record = manifest.get(KEY)
    if record is None:
        if manifest.get("square_sum_lane_outputs"):
            raise ValueError("square-sum emission has no bound association")
        return
    _permission(dict(metadata={"execution_precision": manifest.get("execution_precision")}))
    if record.get("contract") != CONTRACT or record.get("thread_lanes") != LANES:
        raise ValueError("unknown square-sum manifest contract")
    names = [leaf["sum"] for leaf in record["full_sums"]]
    actual = sorted(name for name in names if marker(name) in source)
    if actual != manifest.get("square_sum_lane_outputs"):
        raise ValueError("square-sum manifest differs from emitted source")
    if not manifest.get("square_sum_distributed_region", False) and set(actual) != set(names):
        raise ValueError("standalone square-sum emission has incomplete coverage")
