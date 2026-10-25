"""Compatibility proof for the established flat projection-only P6D schedule."""
from __future__ import annotations

from .bf16_retention import ARITHMETIC, exact_projection_capture, explicit_projection_contract

WGMMA_ARITHMETIC = ("explicit BF16 operands/output; wgmma FP32 accumulation from +0 in ascending k16 steps; "
    "no split-K; one cvt.rn.bf16x2 rounding (tolerance-gated, not bitwise to mma.sync)")

# These are the established exact-tree emitters, not a claim that arbitrary
# future reduction rewrites preserve the tree. A changed calculation source
# requires a new compatibility proof before this optional selection applies.
# The codegen_rules hash includes frozen step04's pruned warp second stage and
# epilogue markers. The arithmetic keeps the original XOR tree, removing only
# +0 lanes: partials start with +0 + leaf and cannot be -0. The independent
# bitwise FP32 interpreter in test_matrix_integration checks cancellation,
# subnormals and both input zero signs through all live-warp boundaries and
# the wide folded path. Per-reduction source/protocol checks below remain exact.
# The convolution donor ae8e7c9e6 additionally fixes the unsigned leaf index
# spelling in the coalesced scalar replay. exact_coalesced_unsigned_leaves
# verifies identical leaves and complete emitted trees after index normalization;
# it does not change the tree or any existing signed-index emission.
# LATCH-DENSE-EXEC re-pinned p9_rowblock.py (was 8019ccfe...): engine 1's compact column-partial j loop gained
# "#pragma unroll COLUMN_UNROLL" (8). Leaves, their ascending j order per virtual thread, partial storage and the
# finalize are unchanged (bitwise A/B over 376 update tensors on the 16384-row ordinary model); engines 2-7 and the
# engine-3 protocol this proof admits are byte-identical.
# Integration3 re-pins only row-consumer scheduling: the accepted consumer set must
# be closed over same-phase reads. Reduction arithmetic/emission is unchanged;
# test_p6d_row_domains checks the rejected optimizer race and safe attachments.
# LATCH-FFN-RETAIN re-pinned p9_rowblock.py (was d6aabd4a...): engines 1-7 and their protocol are byte-identical; the
# file gains the direct column engine (emit_phase_direct: the same leaves r = t + 256 j summed in ascending j from +0
# per virtual thread, partials in the arena scratch, the unchanged two-level butterflies; bitwise A/B over three
# complete updates of the FFN program), its 16-byte register-unpack helpers, the exact (r * E + oc) / E -> r
# re-indexing of per-row leaf operands, and scratch_words() counts direct phases (sum of 256 x E over their column
# reductions). _reduction_proof below binds direct records explicitly (p9_column_direct branch).
REDUCTION_SOURCES = {
    "p9_rowblock.py": "bd51603fc7dab1ee778cc7346f8896552562c53673647c439b0712d766be22be",
    "p6d_rules.py": "08c34e48b66c71f3c357e3de240dc711e0c05b6f46999cda290da3f7d7cd0585",
    "codegen_rules.py": "349be2aedbdcfdfcc4aac85bcfaac64b8b02328f4c94df62449393046936d6b9",
}


def candidate(graph, manifest):
    if not exact_projection_capture(graph) or manifest.get("attention_groups"):
        return False
    matrices = [phase for phase in manifest["phases"] if phase.get("kind") == "gemm"]
    return (any(phase.get("matrix_spec", {}).get("engine") == "p5c" for phase in matrices)
            and all(phase.get("matrix_spec", {}).get("dtype") == "bfloat16"
                    and phase["matrix_spec"].get("engine") in (None, "p5c")
                    and phase["matrix_spec"].get("batch") == 1
                    and not phase["matrix_spec"].get("tail") for phase in matrices))


def _parts(graph):
    from .lower import _field
    values = _field(graph, "values", {})
    if not isinstance(values, dict):
        values = {_field(value, "name"): value for value in values}
    operations = _field(graph, "operations", _field(graph, "nodes", ()))
    output = lambda op: _field(op, "outputs", [_field(op, "name")])[0]
    return values, {output(op): op for op in operations}


def _base(op):
    from .lower import _field
    return str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]


def _canonical(value, aliases):
    from .ir import Ref
    if isinstance(value, Ref):
        return {"ref": aliases.get(value.name, value.name)}
    if isinstance(value, dict):
        return {key: aliases.get(item, item) if key == "ref" and isinstance(item, str)
                else _canonical(item, aliases) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item, aliases) for item in value]
    return value


def _cse_proof(graph, lowered, report, protected):
    """Only identical pure operand casts/views may disappear in this scope."""
    from .lower import _field
    values, operations = _parts(graph)
    new_values, new_operations = _parts(lowered)
    aliases = {}
    merged = report.get("merged_values", [])
    if report.get("merged", 0) != len(merged):
        return None
    for pair in merged:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        old, new = pair
        if old in aliases or old in protected or old not in operations or new not in operations:
            return None
        a, b = operations[old], operations[new]
        if (_base(a) not in ("_to_copy", "t", "transpose", "permute", "view", "reshape")
                or any(_field(op, "effect") != "pure" for op in (a, b))
                or _field(a, "target") != _field(b, "target")
                or _canonical(_field(a, "args"), aliases) != _canonical(_field(b, "args"), aliases)
                or _canonical(_field(a, "kwargs"), aliases) != _canonical(_field(b, "kwargs"), aliases)):
            return None
        for key in ("dtype", "shape", "strides", "storage_offset", "alias_of"):
            va, vb = _field(values[old], key), _field(values[new], key)
            if key == "alias_of":
                va, vb = aliases.get(va, va), aliases.get(vb, vb)
            if _canonical(va, aliases) != _canonical(vb, aliases):
                return None
        aliases[old] = aliases.get(new, new)
    if set(new_operations) != set(operations) - set(aliases):
        return None
    for name, op in new_operations.items():
        original = operations[name]
        if (_field(op, "target") != _field(original, "target")
                or _canonical(_field(original, "args"), aliases) != _canonical(_field(op, "args"), {})
                or _canonical(_field(original, "kwargs"), aliases) != _canonical(_field(op, "kwargs"), {})):
            return None
    if set(new_values) != set(values) - set(aliases):
        return None
    return aliases


def _scalar_expression(op, values, policy):
    from .lower import _field, _float, _ref
    args, kwargs = _field(op, "args", ()), _field(op, "kwargs", {})
    names = set(values)
    dtype = lambda name: str(_field(values[name], "dtype")).removeprefix("torch.")
    def read(index):
        arg = args[index]
        name = _ref(arg, names)
        if name is None:
            return _float(arg) if isinstance(arg, (int, float, bool)) else None
        expression = f"REF({name})"
        return f"__bfloat162float({expression})" if dtype(name) == "bfloat16" else expression
    base = _base(op)
    if base in ("_to_copy", "clone", "contiguous"):
        expression = read(0)
    elif base in ("add", "sub"):
        expression = f"({read(0)} {'+' if base == 'add' else '-'} ({read(1)} * {_float(kwargs.get('alpha', args[2] if len(args) > 2 else 1))}))"
    elif base in ("mul", "div") and not kwargs.get("rounding_mode"):
        expression = f"({read(0)} {'*' if base == 'mul' else '/'} {read(1)})"
    elif base == "pow" and args[1] in (1, 2):
        expression = read(0) if args[1] == 1 else f"({read(0)} * {read(0)})"
    elif base == "tanh":
        expression = f"tanhf({read(0)})"
    elif base == "tanh_backward":
        expression = f"({read(0)} * (1.0f - {read(1)} * {read(1)}))"
    elif base in ("ones_like", "zeros_like"):
        expression = "1.0f" if base == "ones_like" else "0.0f"
    else:
        return None
    name = _field(op, "outputs", [_field(op, "name")])[0]
    return f"__float2bfloat16_rn({expression})" if dtype(name) == "bfloat16" else expression


def _body_ok(spec, body):
    from .matrix_rules import BF16_TILES
    plan = body.get("matrix_plan", {})
    allowed = {"dtype", "a", "b", "out", "m", "n", "k", "batch", "strides", "ak", "bkc", "engine", "roots", "ws_ok", "rot_token", "epi"}
    if set(spec) - allowed:
        return False
    if any(spec.get(key) for key in ("splitk", "out_acc", "out_f32", "pointers", "tail")):
        return False
    if body.get("matrix_shape") != [spec.get(k) for k in ("m", "n", "k")] or body.get("batch") != 1:
        return False
    if body.get("operand_layout") != {"a_k_contiguous": spec.get("ak"), "b_k_contiguous": spec.get("bkc")}:
        return False
    if spec.get("engine") == "p5c":
        if (plan.get("mc", 1) != 1 or plan.get("out_f32") or plan.get("release")
                or any(plan.get(k) for k in ("splitk", "pingpong", "pp", "cluster"))
                or body.get("arithmetic") != WGMMA_ARITHMETIC or plan.get("bn") != 128):
            return False
        if plan.get("ws"):
            return (plan.get("ws_engine") == "p8ws_flat" and plan.get("wm") == 0 and plan.get("bm") in (64, 128)
                    and body.get("engine") == f"p5c_ws_{plan['bm']}x128")
        return (plan.get("bm") == 128 and plan.get("wm") == 2 and body.get("engine") == "p5c_128x128")
    tile = tuple(plan.get(key) for key in ("bm", "bn", "wm", "wn"))
    return (tile in BF16_TILES and plan.get("ks") in (32, 64) and body.get("arithmetic") == ARITHMETIC
            and body.get("engine") == f"popsc_bf16_{tile[0]}x{tile[1]}")


def verify(graph, common, legacy, *, contracts=None, policy):
    try:
        return _verify(graph, common, legacy, contracts=contracts, policy=policy)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None, "established P6D emission evidence is incomplete or malformed"


def _verify(graph, common, legacy, *, contracts=None, policy):
    from .lower import _field, _ident
    proof = (contracts or {}).get("p6d")
    schedule = legacy.get("performance_schedule", {})
    if (not candidate(graph, common) or not proof or schedule.get("selected") != "p6d"
            or schedule.get("precision_policy") != policy or legacy.get("attention_groups")
            or common.get("numerical_policy") != legacy.get("numerical_policy")):
        return None, "established flat P6D protocol or declared precision does not match"
    protected = set(common["inputs"]) | set(common["outputs"]) | set(common["state_writes"]) | set(common["state_writes"].values())
    aliases = _cse_proof(graph, proof.get("lowered_graph"), proof.get("cse_report", {}), protected)
    if aliases is None:
        return None, "established P6D cast/view common-subexpression proof failed"
    canonical = lambda name: aliases.get(name, name)
    values, operations = _parts(proof["lowered_graph"])
    common_layouts = {**common.get("elided_layouts", {}), **common["layouts"]}
    logical = proof.get("logical_layouts", {})
    state_sources = set(common["state_writes"].values())
    for name, layout in common_layouts.items():
        if canonical(name) not in logical:
            continue
        expected = {**layout, "root": canonical(layout["root"])}
        if name in state_sources and layout["root"] in common["state_writes"]:
            expected["root"] = canonical(name)
        if expected != logical[canonical(name)]:
            return None, "established P6D logical view differs: " + name
    for header in ("popsc_mm.cuh", "p5c_wg.cuh", "p8ws.cuh", "s107a_wgmma.cuh", "pops4b_tma.cuh",
                   "pops4b_wg.cuh", "bgemm.cuh", "tgemm.cuh", "core.cuh", "epi.cuh"):
        if not common.get("header_hashes", {}).get(header) or common["header_hashes"][header] != legacy.get("header_hashes", {}).get(header):
            return None, "established P6D calculation header differs: " + header
    scalar_names = set()
    scalar_bases = {"_to_copy", "clone", "contiguous", "add", "sub", "mul", "div", "pow", "tanh", "tanh_backward", "ones_like", "zeros_like"}
    other_bases = {"mm", "matmul", "t", "transpose", "permute", "view", "reshape", "unsqueeze", "squeeze", "expand", "sum", "mean"}
    if any(_base(op) not in scalar_bases | other_bases for op in operations.values()):
        return None, "established P6D graph contains an unproven operation family"
    for scalar in proof.get("scalars", []):
        name = scalar.get("name")
        if name in scalar_names or name not in operations:
            return None, "established P6D scalar has no unique captured operation"
        scalar_names.add(name)
        expression = _scalar_expression(operations[name], values, policy)
        if not expression or expression != scalar.get("expression"):
            return None, "established P6D scalar/cast arithmetic differs: " + name
    if scalar_names != {name for name, op in operations.items() if _base(op) in scalar_bases}:
        return None, "established P6D scalar/cast coverage differs"
    products, checked, matrix_proof = proof.get("products", []), set(), []
    expected_products = {canonical(p["names"][0]) for p in common["phases"] if p.get("kind") == "gemm"}
    if len(products) != len(expected_products):
        return None, "established P6D matrix coverage differs"
    for phase in common["phases"]:
        if phase.get("kind") != "gemm":
            continue
        name = canonical(phase["names"][0])
        matches = [item for item in products if item.get("name") == name]
        if len(matches) != 1:
            return None, "established P6D matrix has no unique emission evidence: " + name
        item = matches[0]
        emitted_phase = legacy["phases"][item["phase"]]
        emitted_names = [row["name"] for row in emitted_phase.get("items", []) if row.get("kind") == "gemm"]
        if (emitted_names.count(name) != 1 or emitted_phase.get("members", [])[emitted_names.index(name)] != item.get("member")):
            return None, "established P6D matrix member is not bound to the final manifest: " + name
        spec, old_spec = phase.get("matrix_spec", {}), item.get("base_spec", {})
        if len(phase.get("members", [])) != 1:
            return None, "common P5C matrix lacks individual body metadata: " + name
        fields = ("dtype", "m", "n", "k", "batch", "strides", "ak", "bkc", "engine")
        if (any(spec.get(key) != old_spec.get(key) for key in fields)
                or any(canonical(spec.get(key)) != old_spec.get(key) for key in ("a", "b"))
                or old_spec.get("out") != _ident(name)
                or not _body_ok(spec, phase["members"][0]) or not _body_ok(item.get("spec", {}), item.get("member", {}))):
            return None, "established P6D matrix specification, family or one-rounding arithmetic differs: " + name
        if ({key: value for key, value in item["spec"].items() if key not in ("out", "epi")}
                != {key: value for key, value in old_spec.items() if key != "out"}):
            return None, "established P6D final emitted operands differ from admitted operands: " + name
        for operand in (spec["a"], spec["b"], phase["names"][0]):
            before = common_layouts.get(operand)
            if not before or {**before, "root": canonical(before["root"])} != logical.get(canonical(operand)):
                return None, "established P6D matrix operand or output layout differs: " + operand
        if spec.get("engine") == "p5c":
            expected_roots = [[canonical(root), offset] for root, offset in spec.get("roots", [])]
            if _canonical(old_spec.get("roots"), {}) != expected_roots:
                return None, "established P6D matrix TMA roots differ: " + name
            plan = item["member"]["matrix_plan"]
            sam, sak, sbk, sbn, _, _ = old_spec["strides"]
            for (root, offset), kmajor, rows, stride, box_rows in (
                    (old_spec["roots"][0], old_spec["ak"], old_spec["m"], sam if old_spec["ak"] else sak, plan["bm"]),
                    (old_spec["roots"][1], old_spec["bkc"], old_spec["n"], sbn if old_spec["bkc"] else sbk, plan["bn"])):
                dims, box = ([old_spec["k"], rows], [64, box_rows]) if kmajor else ([rows, old_spec["k"]], [64, 64])
                maps = [entry for entry in legacy.get("tensor_maps", []) if entry.get("root") == root
                        and entry.get("element_offset") == offset and entry.get("dims") == dims
                        and entry.get("box") == box and entry.get("stride_bytes") == 2 * stride]
                if (len(maps) != 1 or maps[0].get("dtype") != "bfloat16" or maps[0].get("swizzle") != 128
                        or maps[0].get("oob_fill") != "zero" or maps[0].get("storage") != "arena"
                        or maps[0].get("word_offset") != legacy["offsets"].get(root)):
                    return None, "established P6D matrix tensor-map storage binding differs: " + name
        checked.add(name)
        matrix_proof.append({"output": name, "legacy_phase": item["phase"], "engine": item["member"]["engine"],
                             "arithmetic": item["member"]["arithmetic"]})
    if checked != expected_products:
        return None, "established P6D product coverage is incomplete"
    # Additional emission/synchronization checks are below; a successful
    # candidate must bind all private evidence to the final saved manifest.
    return _finish_proof(graph, common, legacy, proof, aliases, matrix_proof, scalar_names, policy)


def _reduction_proof(legacy, proof, operations):
    import hashlib
    from types import SimpleNamespace
    from .lower import _field, _ref, _numel, _float, _ident, TensorLayout
    from . import p6d_rules, p9_rowblock
    from .codegen_rules import exact_reduction
    settings = proof["settings"]
    if (proof.get("calculation_sources") != REDUCTION_SOURCES
            or settings.get("p9_rowblock_engine") != 3 or settings.get("p9_defer_finalize")
            or proof["allocation"].get("pending_finalize") or proof["allocation"].get("p9_kn_words")):
        return None, "established P6D exact reduction source or protocol differs"
    records = proof.get("reductions", [])
    expected = {name for name, op in operations.items() if _base(op) in ("sum", "mean")}
    if len(records) != len(expected) or {row["name"] for row in records} != expected:
        return None, "established P6D reduction coverage differs"
    recipes, result = {}, []
    for row in records:
        name, op = row["name"], operations[row["name"]]
        args, kwargs = _field(op, "args", ()), _field(op, "kwargs", {})
        source = _ref(args[0], set(proof["logical_layouts"]))
        binding = proof["logical_layouts"][source]
        src = TensorLayout(tuple(binding["shape"]), tuple(binding["strides"]), binding["root"], binding["offset"])
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        dims = list(range(len(src.shape))) if dims is None or dims == [] or dims == () else ([dims] if isinstance(dims, int) else list(dims))
        dims = sorted({int(dim) % len(src.shape) for dim in dims})
        kept = [axis for axis in range(len(src.shape)) if axis not in dims]
        keep = bool(args[2] if len(args) > 2 else kwargs.get("keepdim", False))
        shape = tuple(1 if axis in dims else size for axis, size in enumerate(src.shape)) if keep else tuple(src.shape[axis] for axis in kept)
        dtype = proof["logical_layouts"][name]["dtype"]
        if (binding["dtype"] not in ("float32", "bfloat16") or dtype not in ("float32", "bfloat16")
                or row.get("target") != _field(op, "target") or row.get("shape") != list(shape) or row.get("dtype") != dtype):
            return None, "established P6D reduction type, shape or operation differs: " + name
        reduction_shape, kept_shape = tuple(src.shape[axis] for axis in dims), tuple(src.shape[axis] for axis in kept)
        terms = [f"(({index} / {_numel(subshape[j+1:])}LL) % {src.shape[axis]}LL) * {src.strides[axis]}LL"
                 for sequence, subshape, index in [(dims, reduction_shape, "r"), (kept, kept_shape, "o")]
                 for j, axis in enumerate(sequence)]
        position = " + ".join(terms) or "0"
        extent, results = _numel(reduction_shape), _numel(shape)
        scale = f" / {_float(extent)}" if _base(op) == "mean" else ""
        rounded = (lambda text: f"__float2bfloat16_rn({text})") if dtype == "bfloat16" else (lambda text: text)
        reads = row.get("storage_reads", [])
        if any(read.get("kind") != "storage" or read.get("source") != source or read.get("shape") is not None
               or read.get("token") != f"@@LOAD{read.get('token_index')}@@" for read in reads):
            return None, "established P6D reduction source loads differ: " + name
        def read(position):
            tokens = {entry["token"] for entry in reads if entry["position"] == position}
            if len(tokens) != 1:
                raise ValueError("unproven reduction leaf")
            return tokens.pop()
        out = _ident(name)
        selected = p6d_rules.reduction_engine(out=out, position=position, extent=extent, results=results,
            read=read, rounded=rounded, scale=scale, src=src, dims=dims, kept=kept)
        if selected is None:
            selected = exact_reduction(out=out, position=position, extent=extent, results=results,
                                       read=read, rounded=rounded, scale=scale)
        if selected is None:
            code = f'''for (long long o = blockIdx.x; o < {results}LL; o += gridDim.x) {{
  float acc = 0.0f;
  for (long long r = threadIdx.x; r < {extent}LL; r += blockDim.x)
    acc += {read(position)};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {out}[o] = {rounded('acc' + scale)};
  }}
  __syncthreads();
}}'''
            selected = code, {"reduction_elements": extent}
        code, extra = selected
        if extra != row.get("extra") or hashlib.sha256(code.encode()).hexdigest() != row.get("node_code_sha256"):
            return None, "established P6D exact reduction tree or rounding differs: " + name
        recipes[name] = SimpleNamespace(name=name, extra=extra)
        result.append({"output": name, "source": source, "dimensions": dims, "elements": extent,
                       "node_engine": extra.get("engine", "fixed_cta_tree"), "node_code_sha256": row["node_code_sha256"]})
    phase_rows = proof.get("reduction_phases", [])
    phases = {row["phase"]: row for row in phase_rows}
    if len(phases) != len(phase_rows) or set(phases) != {row["phase"] for row in records}:
        return None, "established P6D reduction phase coverage differs"
    for index, row in phases.items():
        emitted = legacy["phases"][index]
        names = [item["name"] for item in emitted["items"] if item["kind"] == "reduction"]
        if names != row["names"] or names != [item["name"] for item in records if item["phase"] == index]:
            return None, "established P6D reduction phase ownership differs"
        record = row.get("p9_record")
        if record is not None and record.get("engine") == "p9_column_direct":
            # LATCH-FFN-RETAIN direct column engine: same leaves, chains and butterflies as the column engine; the
            # phase's other reductions are the inherited warp-per-row bodies, emitted unchanged.
            if any(emitted.get(key) != value for key, value in record.items()):
                return None, "established P6D emitted reduction record differs"
            info = record["p9_rowblock"]
            cols, rest = info.get("column_reductions", []), info.get("inherited_reductions", [])
            specs = [recipes[name].extra.get("p9", {}) for name in cols]
            if (not cols or sorted(cols + rest) != sorted(names) or info.get("staged") or info.get("row_reductions")
                    or any(spec.get("kind") != "col" or spec.get("rows") != info.get("rows") for spec in specs)
                    or info.get("extents") != [spec.get("extent") for spec in specs]
                    or any(not p9_rowblock.direct_inherited(recipes[name]) for name in rest)):
                return None, "established P6D direct column reductions or inherited bodies differ"
            continue
        if record is not None:
            if record.get("engine") not in ("p9_rowblock", "p9_rowblock3") or any(emitted.get(key) != value for key, value in record.items()):
                return None, "established P6D emitted reduction record differs"
            info = record["p9_rowblock"]
            specs = [recipes[name].extra.get("p9", {}) for name in names]
            if (any(spec.get("rows") != info.get("rows") or spec.get("extent") != info.get("extent") for spec in specs)
                    or info.get("row_reductions") != [name for name, spec in zip(names, specs) if spec.get("kind") == "row"]
                    or info.get("column_reductions") != [name for name, spec in zip(names, specs) if spec.get("kind") == "col"]):
                return None, "established P6D staged reduction dimensions or direction differ"
            for pointer in info.get("staged", []):
                bindings = [layout for name, layout in proof["layouts"].items() if _ident(name) == pointer]
                if len(bindings) != 1:
                    return None, "established P6D staged reduction pointer is unbound"
                layout = bindings[0]
                view = TensorLayout(tuple(layout["shape"]), tuple(layout["strides"]), layout["root"], layout["offset"])
                if not view.contiguous or view.numel != info["rows"] * info["extent"]:
                    return None, "established P6D staged reduction view differs"
            if not 0 < info.get("staged_bytes", 0) <= legacy["shared_bytes"]:
                return None, "established P6D staged reduction shared storage differs"
        elif emitted.get("engine") != recipes[names[0]].extra.get("engine"):
            return None, "established P6D fixed reduction engine differs"
    # The unchanged P9 source replays the same virtual 256-thread leaves and
    # butterflies. Bind its scratch request to those admitted node recipes.
    nodes = {name: recipes.get(name, SimpleNamespace(extra={}))
             for phase in proof["phases"] for name in phase["heavy"]}
    if p9_rowblock.scratch_words(proof["phases"], nodes) != proof["allocation"].get("p9_words"):
        return None, "established P6D reduction scratch extent differs"
    return {"nodes": result, "calculation_sources": dict(REDUCTION_SOURCES),
            "replay": "established fixed 256-thread tree, ascending per-thread leaves from +0 and unchanged butterflies"}, None


def _finish_proof(graph, common, legacy, proof, aliases, matrices, scalars, policy):
    import hashlib
    import re
    from .lower import _field, _CTYPES, _ident
    from .resources import select_program_resources
    from .vendor import plan as arena_plan
    from .p5c_epilogue import common_subexpressions, reciprocal
    values, operations = _parts(proof["lowered_graph"])
    logical, layouts = proof["logical_layouts"], proof["layouts"]
    settings = proof.get("settings", {})
    if (proof.get("precision_policy") != policy or settings.get("p9_diag_no_state_stores")
            or settings.get("p9_rowblock_diag") or settings.get("p5c_diag_call")):
        return None, "established P6D has diagnostic or mismatched arithmetic settings"
    canonical = lambda name: aliases.get(name, name)
    writes = {canonical(target): canonical(source) for target, source in common["state_writes"].items()}
    if writes != legacy.get("state_writes") or writes != proof.get("state_writes"):
        return None, "established P6D state output coverage differs"
    if ([canonical(name) for name in common["inputs"]] != legacy["inputs"]
            or [canonical(name) for name in common["outputs"]] != legacy["outputs"]):
        return None, "established P6D public input/output coverage differs"
    scalar_by_name = {item["name"]: item for item in proof["scalars"]}
    inline, stored = set(proof.get("inline", [])), set(proof.get("stored", []))
    epi_proof, epi_owner = [], {}
    for item in proof["products"]:
        name, spec, epi = item["name"], item["spec"], item.get("epilogue")
        if spec.get("out") not in (_ident(name), "nullptr") or item["member"].get("output") != spec["out"]:
            return None, "established P6D output binding differs: " + name
        if item.get("stored_product") != (spec["out"] != "nullptr"):
            return None, "established P6D product store evidence differs: " + name
        if epi is None:
            if spec.get("epi") or not item["stored_product"]:
                return None, "established P6D omitted a product without a proven epilogue: " + name
            continue
        members = epi.get("members", [])
        if (not members or len(set(members)) != len(members) or not set(members) <= scalars
                or spec.get("epi") != epi.get("code")):
            return None, "established P6D epilogue has unproven members or emitted code: " + name
        size = logical[name]["shape"]
        vector = {row["name"]: row for row in epi.get("vector_reads", [])}
        if len(vector) != len(epi.get("vector_reads", [])):
            return None, "established P6D epilogue has duplicate input descriptors: " + name
        for ref in [*vector, *epi.get("store_names", [])]:
            if not layouts.get(ref) or legacy["layouts"].get(ref) != layouts[ref]:
                return None, "established P6D epilogue load/store pointer binding differs: " + ref
        def identity(ref):
            layout = logical.get(ref, {})
            from .lower import _contiguous
            return (layout.get("shape") == size and layout.get("strides") == list(_contiguous(size))
                    and layout.get("offset") == 0)
        def expression(output, active=(), owner=None):
            if output in active or output not in scalar_by_name:
                raise ValueError("unknown or cyclic scalar")
            owner = output if owner is None else owner
            text = scalar_by_name[output]["expression"]
            def value(ref, widened):
                if not identity(ref):
                    raise ValueError("nonidentity epilogue operand " + ref + " in " + output)
                origin = logical[ref]["root"]
                dtype = logical[ref]["dtype"]
                if origin in inline:
                    result = expression(origin, (*active, output), owner)
                elif origin == name:
                    # Both admitted WGMMA headers convert accumulator pairs to
                    # BF16 before widening gv for the generated epilogue.
                    if dtype != "bfloat16" or not widened:
                        raise ValueError("product rounding boundary " + ref + " in " + output)
                    return "gv[j]"
                elif origin in members:
                    if members.index(origin) >= members.index(owner):
                        raise ValueError("epilogue member dependency order " + ref + " in " + output)
                    result = f"r_{origin}[j]"
                elif ref in vector:
                    row = vector[ref]
                    if row.get("index") != "i0" or row.get("dtype") != dtype:
                        raise ValueError("nonidentity vector descriptor " + ref + " in " + output)
                    result = f"p5c::{'upkb' if dtype == 'bfloat16' else 'upk'}({row['register']}[e], j)"
                else:
                    raise ValueError("unproven epilogue memory read " + ref + " in " + output)
                return f"__bfloat162float({result})" if widened else result
            text = re.sub(r"__bfloat162float\(REF\((\w+)\)\)", lambda m: value(m[1], True), text)
            text = re.sub(r"REF\((\w+)\)", lambda m: value(m[1], False), text)
            if "REF(" in text:
                raise ValueError("indexed symbolic epilogue operand")
            return text
        try:
            expressions = [{"name": member, "dtype": scalar_by_name[member]["dtype"], "expression": expression(member)}
                           for member in members]
        except (ValueError, KeyError) as error:
            return None, "established P6D epilogue operand/index/rounding proof failed: " + name + ": " + str(error)
        before = [f"r_{row['name']}[j] = {row['expression']};" for row in expressions]
        after = before
        if epi.get("exact_rewrites"):
            temporaries, rewritten = common_subexpressions(before, 'p5c_t_')
            after = [reciprocal(line) for line in temporaries + rewritten]
        expected_stores = [member for member in members if member in stored]
        stores = [" ".join(f"p5c::st8{suffix}({_ident(member)} + i0, r_{member});" for member in expected_stores)
                  for suffix in ("", "cs")]
        if (expressions != epi.get("expressions") or before != epi.get("compute_before_rewrites")
                or after != epi.get("compute_after_rewrites") or expected_stores != epi.get("store_names")
                or epi.get("store_code") not in stores):
            return None, "established P6D epilogue arithmetic or stored outputs differ: " + name
        declarations = " ".join(f"{_CTYPES[scalar_by_name[member]['dtype']]} r_{member}[8];" for member in members)
        run = ("[&](int e, unsigned i0, const float (&gv)[8]) {\n"
               f"      {declarations}\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {{\n"
               "        const unsigned i = i0 + (unsigned)j;\n        " + "\n        ".join(after) +
               f"\n      }}\n      {epi['store_code']}\n    }}")
        if epi["code"].get("run") != run:
            return None, "established P6D epilogue emitted execution differs: " + name
        prelude = " ".join(f"uint4 {row['register']}[8]{'[2]' if row['dtype'] == 'float32' else ''};"
                           for row in vector.values())
        loads = " ".join((f"{row['register']}[e] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0);"
                         if row["dtype"] == "bfloat16" else
                         f"{row['register']}[e][0] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0); "
                         f"{row['register']}[e][1] = *reinterpret_cast<const uint4*>({_ident(ref)} + i0 + 4);")
                         for ref, row in vector.items())
        streamed = re.sub(r"\*reinterpret_cast<const uint4\*>\(([^;]*?)\);", r"__ldcs(reinterpret_cast<const uint4*>(\1));", loads)
        if (epi["code"].get("prelude") != prelude
                or epi["code"].get("ld") not in [f"[&](int e, unsigned i0) {{ {text} }}" for text in (loads, streamed)]):
            return None, "established P6D epilogue emitted input loads differ: " + name
        for member in members:
            if member in epi_owner or proof.get("phase_of", {}).get(member) != item["phase"]:
                return None, "established P6D epilogue phase ownership differs: " + member
            epi_owner[member] = name
        epi_proof.append({"product": name, "members": members, "stores": expected_stores,
                          "product_stored": item["stored_product"], "rounding": "BF16 rounded product widened before scalar expressions"})
    for item in proof["products"]:
        name = item["name"]
        if item["spec"]["out"] != "nullptr":
            continue
        if (name not in proof.get("elided_products", []) or name in legacy["outputs"]
                or name in writes.values() or name in proof["storage_roots"]
                or any(epi_owner.get(consumer) != name for consumer, loads in proof["loads_of"].items()
                       if any(load["root"] == name for load in loads))):
            return None, "established P6D omitted product has an escaped reader or public store: " + name
    phases, phase_of = proof["phases"], proof["phase_of"]
    if proof.get("in_place") != legacy.get("in_place_commits"):
        return None, "established P6D state commit bindings differ"
    for target, source in proof["in_place"].items():
        if (writes.get(target) != source or layouts[source] != {**logical[source], "root": target, "offset": 0}
                or layouts[target]["shape"] != layouts[source]["shape"] or layouts[source]["dtype"] != "float32"):
            return None, "established P6D in-place state destination differs: " + source
        q = phase_of[source]
        for consumer, loads in proof["loads_of"].items():
            for load in loads:
                if load["root"] != target:
                    continue
                p = phase_of.get(consumer, -1)
                if p > q:
                    return None, "established P6D reads old state after its commit: " + target
                if p == q:
                    same_epi = epi_owner.get(consumer) and epi_owner.get(consumer) == epi_owner.get(source)
                    same_loop = any(consumer in members and source in members for members in phases[q]["loops"].values())
                    if (not (same_epi or same_loop) or load.get("index") != "i"
                            or load.get("shape") != layouts[target]["shape"] or load.get("layout") != layouts[target]):
                        return None, "established P6D same-phase state reads lack load-before-store proof: " + target
    copied = {target: source for target, source in writes.items() if target not in proof["in_place"]}
    if copied != legacy.get("copied_commits") or copied != proof.get("copied_commits", {}).get("mapping"):
        return None, "established P6D final-copy state coverage differs"
    for target, source in copied.items():
        if (source not in stored or source not in legacy["layouts"]
                or layouts[source]["shape"] != layouts[target]["shape"]
                or layouts[source]["dtype"] != layouts[target]["dtype"]):
            return None, "established P6D final state copy lacks a matching stored source: " + source
    from .fusion_p6d import _index32, _u
    from .lower import TensorLayout, _numel
    copy_code = []
    for target, source in copied.items():
        layout = layouts[source]
        view = TensorLayout(tuple(layout["shape"]), tuple(layout["strides"]), layout["root"], layout["offset"])
        value = f"{_ident(source)}[{_index32(view, tuple(layouts[target]['shape']), 'i')}]"
        if layout["dtype"] == "bfloat16":
            value = f"__bfloat162float({value})"
        if layouts[target]["dtype"] == "bfloat16":
            value = f"__float2bfloat16_rn({value})"
        copy_code.append(f"for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < {_u(_numel(layouts[target]['shape']))}; i += gridDim.x * blockDim.x) "
                         f"{_ident(target)}[i] = {value};")
    copy_evidence = proof["copied_commits"]
    if (copy_code != copy_evidence.get("code")
            or hashlib.sha256("\n".join(copy_code).encode()).hexdigest() != copy_evidence.get("code_sha256")):
        return None, "established P6D final state copy emitted code differs"
    reductions, refusal = _reduction_proof(legacy, proof, operations)
    if refusal:
        return None, refusal
    allocation = proof.get("allocation", {})
    if allocation.get("arena_words") != legacy.get("arena_words") or allocation.get("offsets") != legacy.get("offsets"):
        return None, "established P6D arena binding differs"
    if arena_plan.check(proof["storage_roots"], allocation["sizes"], allocation["lifetimes"],
                        allocation["offsets"], allocation["tensor_arena_words"]):
        return None, "established P6D tensor arena proof failed"
    ranges = []
    for prefix in ("p9", "p9_kn", "p9_ctr"):
        start, count = allocation.get(prefix + "_offset"), allocation.get(prefix + "_words", 0)
        if start is None:
            if count:
                return None, "established P6D scratch size has no binding"
            continue
        if (type(start) is not int or type(count) is not int or count < 1
                or start < allocation["tensor_arena_words"] or start + count > allocation["arena_words"]):
            return None, "established P6D scratch overlaps tensors or exceeds arena"
        ranges.append((start, start + count))
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        return None, "established P6D scratch bindings overlap"
    counters = []
    for q, (_, counter) in allocation.get("pending_finalize", {}).items():
        q = int(q)
        if (q <= 0 or q >= len(phases) or counter in counters or not 0 <= counter < allocation.get("p9_ctr_words", 0)
                or not phases[q]["heavy"] or any(name not in {p["name"] for p in proof["products"]} for name in phases[q]["heavy"])):
            return None, "established P6D deferred reduction synchronization is unproven"
        earlier = set(phases[q - 1]["heavy"])
        early = set(phases[q]["heavy"]) | {member for group in phases[q].get("epilogue", {}).values() for member in group}
        if any(load["root"] in earlier for member in early for load in proof["loads_of"].get(member, [])):
            return None, "established P6D consumes deferred reduction before finalize"
        counters.append(counter)
    try:
        resource = select_program_resources(legacy["shared_bytes"], phases=legacy["phases"], arch=legacy["target_arch"])
    except (ValueError, TypeError):
        return None, "established P6D resource admission failed"
    if proof.get("shared_bytes") != legacy["shared_bytes"] or proof.get("resource_policy") != resource:
        return None, "established P6D whole-program resource evidence differs"
    if legacy.get("cluster_dims") or resource["resident_blocks_per_sm"] != 1:
        return None, "established P6D requires an unsupported launch protocol"
    return {"matrix_bodies": matrices, "cse_aliases": aliases, "epilogues": epi_proof,
            "reductions": reductions, "copied_commits": copied,
            "precision_policy": policy, "capture_precision": dict(_field(graph, "metadata", {}).get("precision_policy")
                                                                  or explicit_projection_contract(graph)),
            "resources": {"shared_bytes": legacy["shared_bytes"], "resident_blocks_per_sm": 1},
            "synchronization": {"barrier_count": legacy["barrier_count"], "deferred_reduction_counters": counters}}, None
