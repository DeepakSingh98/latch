"""Common static scheduler adapter for existing operation and region bodies."""
from __future__ import annotations

import contextvars

import inspect
from contextlib import nullcontext

from .execution_plan import current_precision


def view_read_hazards(graph, manifest):
    """Find new-value view reads which require the common register-read fix.

    Inspect the emitted material phases, including the established candidate:
    the two emitters may group the same graph differently. Input views and
    views crossing a material phase boundary do not trigger this rule.
    """
    from .lower import _field, _flatten_refs, _ref, _VIEW_BASES
    operations = _field(graph, "operations", _field(graph, "nodes", ()))
    names = set(manifest["layouts"]) | set(manifest.get("elided_layouts", {}))
    names.update(_field(op, "outputs", [_field(op, "name")])[0] for op in operations)
    phases = {}
    for index, phase in enumerate(manifest["phases"]):
        # A legacy phase may co-schedule matrix bodies and multiple independent
        # elementwise loops. Only values in the same loop share its prefetch.
        groups = ([item.get("names", ()) for item in phase["items"] if item.get("kind") == "elementwise"]
                  if "items" in phase else [phase.get("names", ())] if phase.get("kind") == "elementwise" else [])
        for group_index, group in enumerate(groups):
            for name in group:
                phases.setdefault(name, set()).add((index, group_index))
    state_results = set(manifest.get("state_writes", {}).values())
    # LATCH-DENSE-EXEC: an emitter that declares its same-loop register reads (P6D: root -> consumers, by the
    # predicate its own schedule checker enforces) reads those roots from the producer's register, not memory.
    declared = manifest.get("register_reads")
    aliases, state_reads, same_phase_reads = {}, [], []
    for op in operations:
        target = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
        output = _field(op, "outputs", [_field(op, "name")])[0]
        arguments = _field(op, "args", ())
        if target in _VIEW_BASES and arguments:
            source = _ref(arguments[0], names)
            if source is not None:
                aliases[output] = aliases.get(source, source)
        else:
            roots = {aliases[name] for name in _flatten_refs((arguments, _field(op, "kwargs", {})), names)
                     if name in aliases}
            if roots & state_results:
                state_reads.append(output)
            same = [root for root in roots if phases.get(root, set()) & phases.get(output, set())]
            if same and not (declared is not None and all(output in declared.get(root, ()) for root in same)):
                same_phase_reads.append(output)
    return {"state_result_view_consumers": state_reads, "same_phase_view_consumers": same_phase_reads}


def reduction_retention_candidates(manifest):
    """Recognize only the common full-reduction adapter's private effects.

    This permits trying the established emitter, not selecting it. Its actual
    reduction engine, arithmetic and allocated scratch are checked afterwards.
    Unknown effects and shared scratch remain common requirements.
    """
    candidates = []
    phases = manifest["phases"]
    for index, phase in enumerate(phases):
        if phase.get("engine") != "multi_cta_full" or phase.get("kind") != "reduction":
            continue
        private = phase.get("private_scratch", [])
        if len(private) != 1 or not isinstance(private[0], dict):
            continue
        scratch = private[0].get("name")
        if private != [{"name": scratch, "counter_words": 1, "partial_word_offset": 64, "partial_words": 1024}]:
            continue
        if (not isinstance(scratch, str) or phase.get("internal_mutations") != [scratch]
                or phase.get("global_effects") or phase.get("max_grid_blocks") != 1024
                or not phase.get("requires_full_grid") or len(phase.get("writes", [])) != 1
                or phase.get("counter_effects") != [{"storage": scratch, "word_offset": 0, "words": 1,
                    "initialization": "zeroed by dependency predecessor", "completion": "last arriver restores zero"}]):
            continue
        users = [i for i, other in enumerate(phases) if scratch in
                 set(other.get("reads", ())) | set(other.get("writes", ())) | set(other.get("internal_mutations", ()))]
        initializers = [i for i in users if i != index]
        if len(initializers) != 1:
            continue
        init = phases[initializers[0]]
        if (initializers[0] >= index or init.get("kind") != "reduction_counter_init"
                or init.get("writes") != [scratch] or init.get("reads")
                or any(init.get(key) for key in ("internal_mutations", "private_scratch", "counter_effects", "global_effects"))):
            continue
        candidates.append({"phase": index, "output": phase["writes"][0], "scratch": scratch,
                           "initializer": initializers[0], "reduction_elements": phase.get("reduction_elements"),
                           "arithmetic": phase.get("arithmetic")})
    return candidates


def verify_legacy_reductions(common, legacy, candidates):
    """Return a declared-body proof, or a refusal; never infer it from CUDA text."""
    proofs = []
    for candidate in candidates:
        output = candidate["output"]
        # LATCH-STEP-04: a clip-norm term the established emitter computes inside the indexed phase
        # (declared fused arithmetic, same output layout) replaces this standalone reduction body.
        # The indexed body may share its phase with other bodies (co-scheduling), so match the body item.
        fused = [i for i, p in enumerate(legacy["phases"])
                 if any(item.get("kind") == "embedding_gradient" and output in item.get("phase_writes", ())
                        for item in p.get("items", ()))]
        if len(fused) == 1 and common["layouts"].get(output, {}).get("shape") == legacy["layouts"].get(output, {}).get("shape"):
            proofs.append({**candidate, "legacy_phase": fused[0], "engine": "indexed_fused_norm",
                           "arithmetic": "indexed engine fused sum(g*g): per-thread squares, CTA tree, "
                                         "last-arriver sum of CTA partials in CTA order"})
            continue
        matches = [(i, p) for i, p in enumerate(legacy["phases"]) if output in p.get("writes", ())]
        if len(matches) != 1:
            return [], "established reduction output has no unique material producer: " + output
        index, phase = matches[0]
        reductions = [item.get("name") for item in phase.get("items", ()) if item.get("kind") == "reduction"]
        # Legacy aggregate phase metadata describes its first heavy body. With
        # multiple reductions that is insufficient proof for each individual
        # body, so retain common rather than guessing from the aggregate.
        # LATCH-STEP-04: a body that carries its own reduction contract is proven item by item.
        item = next((it for it in phase.get("items", ()) if it.get("kind") == "reduction" and it.get("name") == output
                     and "engine" in it), None)
        body = item if item is not None else phase
        heavy = [it for it in phase.get("items", ())
                 if it.get("kind") not in {"elementwise", "epilogue", "operand_cast", "row_members"}]
        if (item is not None and heavy == [item]
                and any(phase.get(key) != item.get(key)
                        for key in ("engine", "reduction_elements", "arithmetic"))):
            return [], "established sole reduction body and aggregate contract disagree: " + output
        if (body.get("engine") != "multi_cta_full" or (item is None and reductions != [output])
                or body.get("reduction_elements") != candidate["reduction_elements"]
                or not candidate["arithmetic"] or body.get("arithmetic") != candidate["arithmetic"]):
            return [], "established reduction engine, extent or arithmetic does not match: " + output
        before, after = common["layouts"].get(output, {}), legacy["layouts"].get(output, {})
        if not before or not after or any(before.get(key) != after.get(key) for key in ("shape", "dtype")):
            return [], "established reduction output layout does not match: " + output
        source_phase = common["phases"][candidate["phase"]]
        if output not in phase.get("names", ()):
            return [], "established reduction output has no body name: " + output
        position = phase.get("names", []).index(output)
        if phase.get("targets", [])[position:position + 1] != source_phase.get("targets"):
            return [], "established reduction target does not match: " + output
        proofs.append({**candidate, "legacy_phase": index, "engine": "multi_cta_full"})
    if candidates:
        scratch = legacy.get("matrix_scratch", {})
        offset, counters, partials = (scratch.get(key) for key in ("counter_word_offset", "counter_words", "partial_words"))
        if (any(type(value) is not int for value in (offset, counters, partials))
                or offset < 0 or counters < 1 or partials < 1024
                or offset + counters + partials > legacy.get("arena_words", 0)):
            return [], "established reduction counter/partial scratch allocation is unproven"
    return proofs, None


def matrix_retention_candidates(manifest):
    """Recognize the existing POPS2 split-K adapter, not arbitrary effects.

    Common planning gives every product private counter/partial storage and a
    dependency predecessor which initializes it. The established emitter can
    bind that same calculation to its reusable, per-launch initialized arena.
    Recognition only permits trying it; emitted bodies and bindings are proved
    separately before selecting the established schedule.
    """
    candidates = []
    phases = manifest["phases"]
    for index, phase in enumerate(phases):
        if phase.get("kind") != "gemm" or phase.get("engine") != "pops2_fp32":
            continue
        plan = phase.get("matrix_plan", {})
        fields = [plan.get(key) for key in ("tiles", "tasks", "bm", "bn", "ksplit")]
        if any(type(value) is not int or value <= 0 for value in fields) or fields[-1] <= 1:
            continue
        tiles, tasks, bm, bn, splits = fields
        if tasks != tiles * splits:
            continue
        counters, partials = (tiles + 63) // 64 * 64, (tasks * bm * bn + 63) // 64 * 64
        private = phase.get("private_scratch", [])
        if len(private) != 1 or not isinstance(private[0], dict):
            continue
        scratch = private[0].get("name")
        expected = {"name": scratch, "shape": [counters + partials], "dtype": "float32",
                    "role": "counter_and_partials", "counter_words": counters, "partial_words": partials,
                    "initialization": "partial words defined by matrix slice stores"}
        normalized = {**private[0], "shape": list(private[0].get("shape", []))}
        if (not isinstance(scratch, str) or normalized != expected
                or phase.get("internal_mutations") != [scratch]
                or phase.get("persistent_counter_words") != 0
                or phase.get("counter_effects") or phase.get("global_effects")
                or not phase.get("requires_full_grid") or len(phase.get("writes", [])) != 1):
            continue
        output = phase["writes"][0]
        if (phase.get("names") != [output] or phase.get("product") != output
                or scratch not in phase.get("reads", [])):
            continue
        users = [i for i, other in enumerate(phases) if scratch in
                 set(other.get("reads", ())) | set(other.get("writes", ())) | set(other.get("internal_mutations", ()))]
        initializers = [i for i in users if i != index]
        if len(initializers) != 1:
            continue
        init = phases[initializers[0]]
        operands = [name for name in phase["reads"] if name != scratch]
        if (initializers[0] >= index or init.get("kind") != "matrix_scratch_init"
                or init.get("targets") != ["p_ops_b.private_counter_zero"]
                or init.get("writes") != [scratch] or init.get("reads") != operands
                or init.get("counter_words") != counters or init.get("partial_words") != partials
                or any(init.get(key) for key in ("internal_mutations", "private_scratch", "counter_effects", "global_effects"))):
            continue
        candidates.append({"phase": index, "output": output, "scratch": scratch,
                           "initializer": initializers[0], "counter_words": counters, "partial_words": partials})
    return candidates


def verify_legacy_matrices(common, legacy, candidates, *, contracts=None, common_contracts=None):
    """Prove each retained split-K body and the emitter's actual scratch binding.

    Match physical body plans, arithmetic, targets, operands and outputs rather
    than the first heavy body's aggregate phase metadata or CUDA text. Grid
    estimates are planner inputs; only the selected physical plan is compared.
    """
    from .matrix_select import HEADERS
    if not candidates:
        return [], None
    proofs = []
    checked_phases = set()
    split_candidates = {candidate["output"]: candidate for candidate in candidates}
    # Retaining an entire schedule also retains its unsplit companions. Prove
    # them too: a grid-cost change must not silently pick a different body.
    for source_index, source in enumerate(common["phases"]):
        if source.get("engine") != "pops2_fp32":
            continue
        output = source.get("product")
        split_candidate = split_candidates.get(output)
        candidate = split_candidate or {"phase": source_index, "output": output}
        matches = [(i, phase, item) for i, phase in enumerate(legacy["phases"])
                   for item in phase.get("items", []) if item.get("kind") == "gemm" and item.get("name") == output]
        if len(matches) != 1:
            return [], "established matrix output has no unique emitted body: " + output
        index, phase, item = matches[0]
        declared = [(row, body) for row in (contracts or {}).get("pops2", []) if row.get("phase") == index
                    for body in row.get("items", []) if body.get("name") == output]
        if len(declared) != 1:
            return [], "established matrix lacks unique fresh emission evidence: " + output
        evidence, contract = declared[0]
        body = contract.get("body", {})
        source = {**source, "body": source.get("matrix_plan", {}).get("body")}
        fields = ("engine", "body", "matrix_shape", "arithmetic", "transposed_product", "requires_full_grid", "shared_bytes")
        physical = lambda plan: {key: value for key, value in plan.items() if key != "estimate_us"}
        if (any(key not in body or body[key] != source.get(key) for key in fields)
                or not body.get("arithmetic") or not body.get("matrix_plan")
                or physical(body["matrix_plan"]) != physical(source.get("matrix_plan", {}))):
            return [], "established matrix body, shape, plan or arithmetic does not match: " + output
        layout = (common_contracts or {}).get("pops2_layouts", {}).get(output)
        if not layout or contract.get("layout") != layout:
            return [], "established matrix emitted operand/destination layout does not match: " + output
        header = HEADERS.get(body["matrix_plan"].get("body"))
        if not header or not common.get("header_hashes", {}).get(header) or (
                common["header_hashes"][header] != legacy.get("header_hashes", {}).get(header)):
            return [], "established matrix calculation header does not match: " + output
        operands = [name for name in source["reads"] if name != candidate.get("scratch")]
        if contract.get("operand_roots") != operands or output not in phase.get("writes", []) or output not in phase.get("names", []):
            return [], "established matrix operands or material output does not match: " + output
        position = phase["names"].index(output)
        if phase.get("targets", [])[position:position + 1] != source.get("targets"):
            return [], "established matrix target does not match: " + output
        for name in [*operands, output]:
            before, after = common["layouts"].get(name, {}), legacy["layouts"].get(name, {})
            keys = ("shape", "strides", "offset", "dtype", "root")
            if not before or not after or any(before.get(key) != after.get(key) for key in keys):
                return [], "established matrix operand/output layout does not match: " + name
        if split_candidate is None:
            if body["matrix_plan"].get("ksplit") != 1:
                return [], "established split-K matrix has no recognized effect proof: " + output
            proofs.append({**candidate, "legacy_phase": index, "engine": body["engine"],
                           "body": body["body"], "header": header})
            continue
        binding = evidence.get("split_bindings", {}).get(output, {})
        if any(binding.get(key) != candidate[key] for key in ("counter_words", "partial_words")):
            return [], "established matrix split-K storage requirement does not match: " + output
        if index not in checked_phases:
            arena = legacy.get("matrix_scratch", {})
            offset, counters, partials = (arena.get(key) for key in ("counter_word_offset", "counter_words", "partial_words"))
            if (any(type(value) is not int or value < 0 for value in (offset, counters, partials))
                    or offset != (contracts or {}).get("matrix_scratch_base")
                    or offset + counters + partials > legacy.get("arena_words", 0)):
                return [], "established matrix counter/partial scratch allocation is unproven"
            for kind, capacity in (("counter", counters), ("partial", partials)):
                intervals = []
                for bound in evidence["split_bindings"].values():
                    start, count = bound.get(kind + "_word_offset"), bound.get(kind + "_words")
                    if (type(start) is not int or type(count) is not int or start < 0 or count < 1
                            or start + count > capacity):
                        return [], "established matrix split-K binding exceeds its allocation"
                    intervals.append((start, start + count))
                intervals.sort()
                if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                    return [], "established co-scheduled split-K bindings overlap"
            checked_phases.add(index)
        proofs.append({**candidate, "legacy_phase": index, "engine": body["engine"],
                       "body": body["body"], "binding": binding, "header": header})
    return proofs, None


def homogeneous_long_candidate(manifest):
    groups = manifest.get("attention_groups", [])
    if not groups or any(group.get("decision") != "compose_long_bf16_boundary_attention" for group in groups):
        return False
    admitted = {group["group"] for group in groups}
    phases = [p for p in manifest["phases"] if "attention" in p.get("kind", "")]
    kinds = {"attention_long_prepare", "attention_long_forward", "attention_long_dq", "attention_long_kv"}
    return (len(phases) == 4 * len(admitted)
            and all(p.get("body_interface") == "compose-v1" and p.get("group") in admitted
                    and p["kind"] in kinds for p in phases)
            and all({p["kind"] for p in phases if p["group"] == group} == kinds for group in admitted))


def explicit_flat_candidate(graph, manifest):
    """LATCH-FFN-RETAIN: an attention-free flat program whose BF16 arithmetic is spelled by explicit captured casts.

    The homogeneous protocol proof below never relies on an attention group to certify a P5C body: it binds each
    admitted common P5C product to the established emitter's body with the same shape, operand layout, batch and
    one-rounding arithmetic (or the sole-reader FP32 conversion of that rounding). Programs captured under a declared
    precision policy keep their separate, stricter flat P6D proof (p6d_retention), and programs with any attention
    phase keep the attention protocol requirements. No model name, size or new numerical threshold enters.
    """
    from .lower import _field
    if (manifest.get("attention_groups") or _field(graph, "metadata", {}).get("precision_policy")
            or any("attention" in p.get("kind", "") or p.get("body_interface") == "compose-v1"
                   for p in manifest["phases"])):
        return False
    return any(p.get("matrix_spec", {}).get("engine") == "p5c" for p in manifest["phases"])


def verify_legacy_long_protocol(graph, common, legacy):
    """Prove body coverage by the existing homogeneous long-attention protocol."""
    from .lower import _field, _flatten_refs, _ref
    protocol = legacy.get("performance_schedule", {})
    if (protocol.get("selected") != "p6d"
            or protocol.get("precision_policy") != current_precision()
            or common.get("numerical_policy") != legacy.get("numerical_policy")):
        return None, "established long-attention protocol or declared arithmetic does not match"
    old_groups = {g["group"]: g for g in legacy.get("attention_groups", [])}
    fields = ("group", "head_width", "sequence", "source_commit", "version", "recipe", "sources", "outputs", "removed")
    for group in common["attention_groups"]:
        old = old_groups.get(group["group"], {})
        if (old.get("decision") != "long_bf16_boundary_attention"
                or any(group.get(key) != old.get(key) for key in fields)):
            return None, "established long-attention group does not match: " + group["group"]
    if len(old_groups) != len(common["attention_groups"]):
        return None, "established long-attention group coverage differs"
    preparations = []
    for group in old_groups:
        bodies = [item for phase in legacy["phases"] for item in phase.get("items", [])
                  if item.get("group") == group and item.get("kind", "").startswith("attention_long_")]
        prepare = [item for item in bodies if item["kind"] == "attention_long_prepare"]
        if (sorted(item["kind"] for item in bodies if item["kind"] != "attention_long_prepare") !=
                ["attention_long_dq", "attention_long_forward", "attention_long_kv"]
                or len(prepare) > 1
                or any(item.get("source_commit") != old_groups[group]["source_commit"] for item in bodies)):
            return None, "established long-attention emitted body coverage differs: " + group
        if prepare:
            # P6D can fuse key norms into the K product while the first owner
            # still emits preparation of shared mask proof tables. Those
            # attributed outputs are a subset of the common prepare body;
            # packed arena offsets may differ, but their layouts may not.
            common_prepare = [phase for phase in common["phases"]
                              if phase.get("kind") == "attention_long_prepare" and phase.get("group") == group]
            written = prepare[0].get("phase_writes", [])
            if (len(common_prepare) != 1 or not written or len(set(written)) != len(written)
                    or not set(written).issubset(common_prepare[0].get("writes", []))
                    or any(not common["layouts"].get(name) or not legacy["layouts"].get(name)
                           or any(common["layouts"][name].get(key) != legacy["layouts"][name].get(key)
                                  for key in ("shape", "strides", "dtype")) for name in written)):
                return None, "established long-attention preparation outputs differ: " + group
            preparations.append({"group": group, "outputs": written})

    def products(manifest):
        result = {}
        for phase in manifest["phases"]:
            if phase.get("kind") != "gemm":
                continue
            names = ([item["name"] for item in phase["items"] if item.get("kind") == "gemm"]
                     if "items" in phase else [name for name, target in zip(phase.get("names", []), phase.get("targets", []))
                         if target in ("aten.mm.default", "aten.bmm.default", "aten.matmul.default")])
            members = phase.get("members", [phase])
            if len(names) == len(members):
                result.update(zip(names, members))
        return result

    old_products, new_products = products(legacy), products(common)
    operations = _field(graph, "operations", _field(graph, "nodes", ()))
    names = set(common["layouts"]) | set(common.get("elided_layouts", {}))
    readers = {}
    for op in operations:
        for ref in set(_flatten_refs((_field(op, "args", ()), _field(op, "kwargs", {})), names)):
            readers.setdefault(ref, []).append(op)
    proof = []
    for output, body in new_products.items():
        if not body.get("engine", "").startswith("p5c"):
            continue
        old_output, old = output, old_products.get(output)
        if old is None and len(readers.get(output, [])) == 1:
            consumer = readers[output][0]
            target = str(_field(consumer, "target", "")).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
            converted = _field(consumer, "outputs", [_field(consumer, "name")])[0]
            source_layout = common["layouts"].get(output, common.get("elided_layouts", {}).get(output, {}))
            converted_layout = legacy["layouts"].get(converted, {})
            converted_body = old_products.get(converted, {})
            # Existing P5C_CONVERT writes float(bf16_rn(acc)), preserving the
            # intermediate BF16 rounding. Admit only its original sole-reader
            # conversion rule, never a new FP32 accumulation recipe.
            if (target == "_to_copy" and _ref(_field(consumer, "args", [None])[0], names) == output
                    and output not in common.get("outputs", [])
                    and output not in common.get("state_writes", {}).values()
                    and source_layout.get("dtype") == "bfloat16" and converted_layout.get("dtype") == "float32"
                    and source_layout.get("shape") == converted_layout.get("shape")
                    and converted_body.get("matrix_plan", {}).get("out_f32")
                    and converted_body.get("engine", "").endswith("_f32")):
                old_output, old = converted, converted_body
        if (not old or not old.get("engine", "").startswith("p5c")
                or any(body.get(key) != old.get(key) for key in ("matrix_shape", "operand_layout", "batch", "arithmetic"))):
            return None, "established P5C body coverage does not match: " + output
        proof.append({"output": output, "legacy_output": old_output, "engine": old["engine"],
                      "conversion": "float(bf16_rn(acc))" if output != old_output else None})
    if len(proof) != sum(p.get("matrix_spec", {}).get("engine") == "p5c" for p in common["phases"]):
        return None, "common P5C body metadata is not individually proven"
    return {"protocol": "p6d", "precision_policy": protocol["precision_policy"],
            "attention_groups": sorted(old_groups), "matrix_bodies": proof,
            **({"preparation_bodies": preparations} if preparations else {})}, None


# LATCH-FFN-RETAIN: attention-free explicit-cast programs reuse the homogeneous P5C protocol proof (A/B switch).
EXPLICIT_FLAT_PROTOCOL = True
# LATCH-RESEARCH-EDITS (A/B switch): masked FP32 edge bodies of the common candidate require common planning.
EDGE_BODY_CAPABILITY = True


def flat_selection(graph, program, *, mode, depth, explicit_options, fuse_elementwise):
    """Choose from bodies actually admitted by the common candidate.

    The measured small-body regression does not justify replacing established
    static emission where common planning adds no body or required safety rule.
    Preserve all regions, explicit planner controls and opaque body effects.
    """
    from .lower import _field
    from .bf16_retention import exact_projection_capture
    from .p6d_retention import candidate as p6d_matrix_candidate
    phases = program.manifest["phases"]
    capabilities = {
        "p5c": sum(p.get("matrix_spec", {}).get("engine", p.get("engine")) in ("p5c", "p8ws") for p in phases),
        "attention": sum("attention" in p.get("kind", "") or p.get("body_interface") == "compose-v1" for p in phases),
        "matrix_adapters": sum(p.get("engine", "").startswith("latch_matrix:") for p in phases),
        # Frozen ROW transfer: retain a new scheduling body only after its
        # structural admission actually emitted it. Untouched FP32 programs
        # continue through the established legacy-selection policy.
        "pipelined_rows": sum(bool(p.get("row_fusion", {}).get("next_row_pipelined")) for p in phases),
        "multi_cta_full": sum(p.get("engine") == "multi_cta_full" for p in phases),
        "opaque_effects": sum(bool(p.get("internal_mutations") or p.get("counter_effects")
                                    or p.get("private_scratch") or p.get("global_effects")) for p in phases),
        "global_fences": sum(bool(body.get("global_fence")) for body in program.manifest["execution_plan"]["bodies"]),
        # LATCH-STEP-04: reusable indexed accumulation is a common-path body.
        "indexed_accumulation": sum(p.get("indexed_engine") not in (None, "") for p in phases),
        # LATCH-RESEARCH-EDITS: step-14 masked FP32 edge tiles are planned only by the standalone common planner;
        # the established static emitter runs those products on the generic 16 x 16 tile.
        "fp32_edge_bodies": sum(bool((p.get("matrix_spec") or {}).get("edge_fp32"))
                                or any(bool((m.get("matrix_spec") or {}).get("edge_fp32")) for m in p.get("members", []))
                                for p in phases) if EDGE_BODY_CAPABILITY else 0,
    }
    capabilities.update(view_read_hazards(graph, program.manifest))
    reduction_candidates = reduction_retention_candidates(program.manifest)
    matrix_candidates = matrix_retention_candidates(program.manifest)
    required = dict(capabilities)
    # LATCH-STEP-04: the established static emitter receives the same indexed engine explicitly
    # (verified after its lowering), so indexed accumulation alone does not require common planning.
    required["indexed_accumulation"] = 0
    # Internal grid barriers of an indexed engine body (atomic_v1: zero, accumulate, norm stages) do not require
    # common planning either: the static emitter runs every indexed heavy body on all CTAs, never a CTA subset.
    required["global_fences"] -= sum(1 for body in program.manifest["execution_plan"]["bodies"]
                                     if body.get("global_fence") and type(body.get("index")) is int
                                     and 0 <= body["index"] < len(phases)
                                     and phases[body["index"]].get("indexed_engine"))
    required["multi_cta_full"] -= len(reduction_candidates)
    required["opaque_effects"] -= len(reduction_candidates)
    required["opaque_effects"] -= len(matrix_candidates)
    protocol_candidate = homogeneous_long_candidate(program.manifest)
    if protocol_candidate:
        required["p5c"] = required["attention"] = 0
        required["pipelined_rows"] = 0
    explicit_candidate = (EXPLICIT_FLAT_PROTOCOL and not protocol_candidate
                          and explicit_flat_candidate(graph, program.manifest))
    if explicit_candidate:
        required["p5c"] = 0
        required["pipelined_rows"] = 0
    flat_p6d_candidate = not explicit_candidate and p6d_matrix_candidate(graph, program.manifest)
    if flat_p6d_candidate:
        required["p5c"] = 0
        # A row-load prefetch alone does not displace the established whole
        # P6D schedule. This only permits the candidate: lower still proves
        # its complete arithmetic/storage protocol or keeps the common body.
        required["pipelined_rows"] = 0
    reasons = []
    if mode == "common":
        reasons.append("explicit common planner diagnostic")
    if depth != 1:
        reasons.append("nested region retains common planning")
    if explicit_options:
        reasons.append("explicit lowering controls: " + ", ".join(sorted(explicit_options)))
    if not fuse_elementwise:
        reasons.append("unfused diagnostic retains common operation lowering")
    from .torch_row_reduction import CONTRACT as row_contract
    from .embedding_backward_contract import CONTRACT as embedding_contract
    if any(phase.get("indexed_arithmetic") == embedding_contract for phase in phases):
        reasons.append("explicit learned embedding ten-row arithmetic requires common lowering")
    from .torch_column_sum import CONTRACT as column_contract
    if any(phase.get("column_arithmetic") == column_contract for phase in phases):
        reasons.append("explicit CUDA vector4 column-sum arithmetic requires common lowering")
    if any(phase.get("row_arithmetic") == row_contract for phase in phases):
        reasons.append("explicit CUDA vector4 row-reduction arithmetic requires common lowering")
    from .torch_layer_norm import CONTRACT as layer_norm_contract
    if any(phase.get("row_arithmetic") == layer_norm_contract for phase in phases):
        reasons.append("explicit CUDA vector4 LayerNorm arithmetic requires common lowering")
    from .torch_layer_norm_backward import ROW_ARITHMETIC as layer_norm_backward_contract
    if any(phase.get("row_arithmetic") == layer_norm_backward_contract for phase in phases):
        reasons.append("explicit CUDA vector4 LayerNorm dX arithmetic requires common lowering")
    from .torch_layer_norm_backward import AFFINE_CONTRACT as layer_norm_affine_contract
    if any(phase.get("column_arithmetic") == layer_norm_affine_contract for phase in phases):
        reasons.append("explicit CUDA LayerNorm affine arithmetic requires common lowering")
    from .torch_logsoftmax import CONTRACT as log_softmax_contract
    if any(phase.get("row_arithmetic") in (log_softmax_contract+"_forward",log_softmax_contract+"_backward") for phase in phases):
        reasons.append("explicit CUDA ILP4 log-softmax arithmetic requires common lowering")
    if any(phase.get("exact_vocab_contract", {}).get("contract") == log_softmax_contract for phase in phases):
        reasons.append("exact unpadded ILP4 vocabulary rows require common lowering")
    from .torch_softmax_backward import ROW_ARITHMETIC as softmax_backward_contract
    if any(phase.get("row_arithmetic") == softmax_backward_contract for phase in phases):
        reasons.append("explicit CUDA warp32 softmax backward arithmetic requires common lowering")
    if program.manifest.get("scalar_divisions"):
        reasons.append("emitted CUDA scalar reciprocal arithmetic requires common lowering")
    bf16_candidate = (not explicit_candidate and exact_projection_capture(graph)
                      and (current_precision() != "strict" or _field(graph, "metadata", {}).get("precision_policy")))
    if not bf16_candidate and ((current_precision() != "strict" and not (protocol_candidate or explicit_candidate))
            or _field(graph, "metadata", {}).get("precision_policy")):
        reasons.append("declared arithmetic retains common planning")
    reasons.extend("admitted common capability: " + name for name, count in required.items() if count)
    return {"version": 1, "requested": "compose", "mode": mode,
            "selected": "common_static" if reasons else "legacy_static",
            "reasons": reasons or ["no additional admitted body or common-only safety requirement; retain established static emitter"],
            "admitted_capabilities": capabilities,
            **({"reduction_retention_candidates": reduction_candidates} if reduction_candidates else {}),
            **({"matrix_retention_candidates": matrix_candidates} if matrix_candidates else {}),
            **({"protocol_retention_candidate": "p6d"} if protocol_candidate else {}),
            **({"protocol_retention_candidate": "p6d_explicit_flat"} if explicit_candidate else {}),
            **({"bf16_retention_candidate": "exact_projection_capture"} if bf16_candidate else {}),
            **({"p6d_matrix_retention_candidate": True} if flat_p6d_candidate else {}),
            "policy": "actual body admission and effects; no model names, workload sizes or new numerical thresholds"}


def step13_would_upgrade(provider):
    """LATCH-FFN-RETAIN: True when the step-13 structural default would move some product of the common candidate to a
    different admitted provider body (the same conditions as step13_common_upgrade, without re-lowering). The
    established static emitter cannot run those bodies, so an explicit flat program keeps common planning then."""
    from .matrix_adapters.candidates import COMPOSE_COMMON_POLICY
    from .matrix_adapters.provider import _CHOICES
    scope = _CHOICES.get()
    if (COMPOSE_COMMON_POLICY is None or provider is None or provider.policy == COMPOSE_COMMON_POLICY
            or (scope is not None and scope.get("policy") is not None) or provider.force is not None):
        return False
    return any(d.get("problem_key") is not None and d.get("step13_body") not in (None, d.get("selected"))
               for d in provider.decisions)


def step13_common_upgrade(provider, relower):
    """Release step 13 structural default (a selection rule; no new body or schedule).

    Called only when the frozen-rule program already selected the common planner. If the step-13 policy would
    choose a different admitted body for some provider product, the program is lowered once more under that
    policy and returned with a record; otherwise ``None``. Explicit tuning/diagnostic policies are respected, and
    General established static programs never reach this rule (adding a provider body would force common
    planning, which the development suite measured as 1.4x slower on such a program). The separately checked
    homogeneous-long protocol rule below can also call it; flat matrix retention is not sufficient.
    """
    from .matrix_adapters.candidates import COMPOSE_COMMON_POLICY
    from .matrix_adapters.provider import _CHOICES, default_policy
    scope = _CHOICES.get()
    if (COMPOSE_COMMON_POLICY is None or provider is None or provider.policy == COMPOSE_COMMON_POLICY
            or (scope is not None and scope.get("policy") is not None) or provider.force is not None):
        return None
    changed = [d for d in provider.decisions if d.get("problem_key") is not None
               and d.get("step13_body") not in (None, d.get("selected"))]
    if not changed:
        return None
    if provider.inventory is not None:          # the re-lowering re-inventories every product
        del provider.inventory[provider.inventory_start:]
    with default_policy(COMPOSE_COMMON_POLICY):
        program, new_provider = relower()
    return program, new_provider, {"policy": COMPOSE_COMMON_POLICY, "changed_products": [
        {"output": d["output"], "from": d.get("selected") or "native", "to": d["step13_body"]} for d in changed]}


def long_protocol_common_upgrade(provider, legacy_manifest, protocol, relower):
    """Retain admitted WGMMA upgrades when newly legal P6D would discard them.

    This runs only after the caller proves the entire homogeneous long-attention
    protocol and its safety contracts. P5C-containing established programs keep
    their fused schedule. For the remaining programs, the existing step-13
    product admission and FLOP boundary decide whether a common candidate adds
    an eligible body. Its actual whole-program grid-barrier count must not
    increase: a faster product alone does not justify additional global
    synchronization. No model identity or new size threshold enters selection.
    """
    if (not protocol or protocol.get("protocol") != "p6d"
            or not protocol.get("attention_groups") or protocol.get("matrix_bodies")):
        return None

    def has_p5c(value):
        if isinstance(value, dict):
            engine = value.get("engine", "")
            return (isinstance(engine, str) and engine.startswith(("p5c", "p8ws"))) or any(
                has_p5c(child) for child in value.values())
        return isinstance(value, (list, tuple)) and any(has_p5c(child) for child in value)

    # Inspect actual emitted members, including co-scheduled products; an empty
    # common-to-established proof alone does not exclude additional P5C bodies.
    if has_p5c(legacy_manifest.get("phases", [])):
        return None
    inventory = getattr(provider, "inventory", None)
    prior_inventory = list(inventory) if inventory is not None else None
    upgraded = step13_common_upgrade(provider, relower)
    if upgraded is None:
        return None
    program, new_provider, rule = upgraded
    established_barriers = legacy_manifest.get("barrier_count")
    common_barriers = program.manifest.get("barrier_count")
    if (type(established_barriers) is not int or established_barriers < 0
            or type(common_barriers) is not int or common_barriers < 0
            or common_barriers > established_barriers):
        if inventory is not None:
            inventory[:] = prior_inventory
        return None
    # Preserve the exact accepted program and metadata when the cost guard
    # passes, so unaffected artifacts retain their complete byte identity.
    return program, new_provider, rule


# Release step 14: structural rules of the standalone flat common planner (group_rules features). Each rule is
# exact (values unchanged) or a declared ordered FP32 reassociation; none reads model names or dimensions.
# STEP14_RULES = False reproduces the step-13 lowering (research A/B arms only; not an API).
STEP14_RULES = True
STEP14_FEATURES = "+idx32+idxid+vecbf+tmae+tmaj+emerge+ecta+powx"
# Step-14 speed: FP32 small-tile products in intervals with pinned matrix tiles claim tasks from a per-launch queue
# (planned by the list-scheduling bound), and staged-epilogue IEEE divisions/reciprocals run their straight-line fast
# paths with an exact per-chunk fallback. STEP14S_RULES = False reproduces the step-14 lowering (A/B arms only).
STEP14S_RULES = True
STEP14S_FEATURES = "+ectaq+xdiv"


def STEP14_FLAT_RULES():
    return (STEP14_FEATURES + (STEP14S_FEATURES if STEP14S_RULES else "")) if STEP14_RULES else ""


# LATCH-MOESUP: existing exact engines the standalone flat planner did not name. 'wide': one warp per result replays
# the frozen 256-thread CTA tree for reductions of 257..2048 leaves (bitwise; P6-C). 'ilp8': elementwise phases the
# vector rules refuse (e.g. a broadcast operand) keep eight independent elements in flight per thread (same
# expressions and rounding; P8-C). MOESUP_RULES = False reproduces the previous lowering (A/B arms).
MOESUP_RULES = True
MOESUP_FEATURES = "+wide+ilp8+rhoist+rsink"
# Frozen ROW donor c2a325e441f1c614b7cec4ec4ed94d43ec4e1895: scheduling only.
# Keep the prior defaults available for an isolated source/measurement ablation.
ROW_RULES = True
ROW_FEATURES = "+lptr+lsreg+rpipe+deepred+lsplit"
# (+rhoist+rsink: P9-B row-fusion load hoisting / store sinking, exact; plain update 93.2 -> 81.0 ms, prof-plain-c5-hs.)
# (LATCH-MOESUP-ROW: +lptr+lsreg: P9-A per-interval opaque pointers / special registers (no live range crosses a grid
# barrier; the 255-register kernel no longer serializes row loops), plain 72.43 -> 63.68 ms with the widened merge;
# +rpipe next-row pipelined row groups; +deepred 32 batched loads in long single-CTA reductions. All exact.
# 'pstream' (streaming readout gradients on a disjoint CTA slice of a later readout's logits product) is exact but
# measured slower -- aux 106.12 -> 107.69 ms, `match-final`: both phases are DRAM-bound -- so it is not a default.)
# (+vprod with virtual row gathers measured slower on the MoE recipe: inline readers lose 16-byte vector loops,
# plain update 93.2 -> 134.5 ms (prof-plain-c4); kept available as an explicit rule, not a default.)
# (+vprod: P8-C virtual producers -- single-consumer casts/arithmetic/index maps, now including row gathers,
# evaluated inline by their elementwise readers with identical operations and roundings; bit-identical.)

# LATCH-MOESUP-ROW: CTA-tree reductions with few outputs and long per-thread chains (e.g. Muon's per-matrix norms,
# 442k leaves per CTA) issue their strided loads in batches before adding them in the unchanged per-thread order
# (existing P-OPS2-D batched loads; 32 in flight with 'deepred'). Exact. The compose planner never enabled it.
BATCHED_REDUCTIONS = True


def MOESUP_FLAT_RULES():
    return MOESUP_FEATURES + (ROW_FEATURES if ROW_RULES else "") if MOESUP_RULES else ""


# Set while the standalone flat common planner lowers (depth 1): scopes the step-14 matrix-selection rules (FP32
# edge tiles / ordered split / transposed orientation) to that planner; loops, regions and legacy paths unchanged.
STEP14_FLAT_ACTIVE = contextvars.ContextVar('latch_step14_flat_active', default=False)


def step14_flat():
    return STEP14_RULES and STEP14_FLAT_ACTIVE.get()


# Bodies the composed flat planner takes from the matrix provider (module constant: diagnostics may extend it).
PROVIDER_BODIES = ("pops4b_tma", "pops4b_wg", "popsc_bf16_edge", "p8ws")


def batch_tma_policy(graph):
    """Record normal stacked-TMA admission without the obsolete readout veto.

    The accepted row (0541553960) and matrix (6c72bf2c6d) donors both
    remeasured the original multi-readout slowdown. Their qualified normal
    selection admits stacked TMA for either phase. Retain structural inventory
    and the explicit context override for diagnostics; do not infer a new speed
    claim from either donor's timing.
    """
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})
    heads = (metadata.get("vocabulary_head") or {}).get("rewritten", [])
    large = [head for head in heads
             if head.get("dtype") == "bfloat16" and head.get("backward")
             and int(head.get("rows", 0)) * int(head.get("padded", 0)) >= 2 ** 28]
    return {"version": 2, "batch_stacked_tma": True,
            "large_training_readouts": len(large),
            "reason": "qualified normal stacked-TMA admission; prior multi-readout veto superseded"}


def static_shared_budget(reserve_bytes):
    """Dynamic budget below H100's total per-CTA opt-in limit.

    This is a caller-declared reserve, not an estimate of compiled static
    storage. Actual static + dynamic storage still needs CUDA admission.
    """
    if type(reserve_bytes) is not int or not 0 <= reserve_bytes <= 232448 - 17408:
        raise ValueError("static_shared_reserve_bytes must be a nonnegative integer leaving at least 17408 dynamic bytes")
    return 232448 - reserve_bytes


def lower_composed_flat(graph, *, fuse_elementwise=True, selection_probe=False,
                        static_shared_reserve_bytes=0, **options):
    from .lower import LoweringError, _LOWERING_DEPTH, _lower_serial, _validate_flat_precision_policy, _FLAT_COLSPLIT_PROBE
    from .group_rules import FAST_ACTIVATION
    from .p5c_matrix import FLAT_SELECTED_DEFAULTS
    from .performance_schedule import _configuration
    standalone = not options.get("staged") and not options.get("region_body")
    shared_limit = static_shared_budget(static_shared_reserve_bytes)
    if static_shared_reserve_bytes and (not standalone or _LOWERING_DEPTH.get() != 1):
        raise LoweringError("static shared reserve requires a standalone common compose program")
    # These are independent body/local storage rules, not a model protocol.
    # Wide P5C admission belongs to rotated loop stages; flat products retain
    # the existing flat tile-occupancy threshold and small-tile fallback.
    defaults = dict(barrier_schedule="dependency", group_rules=("p5c+p5ce+vec4+vec4u" + STEP14_FLAT_RULES() + MOESUP_FLAT_RULES()
                                                                if standalone
                    else "p5c+p5cw+p5ce+vec4+vec4u"),
                    elide_stores=True, row_fusion=True, warp_softmax=True,
                    epilogue_fusion=True, write_through=True)
    if standalone:
        defaults["matrix_engine"] = "auto"
        if MOESUP_RULES and ROW_RULES and BATCHED_REDUCTIONS:
            # Prefetch leaves, then add in the unchanged per-thread order.
            defaults["batched_reduction_loads"] = True
        from . import flat_elementwise
        from . import update_schedule
        from . import matrix_schedule, norm_fusion, accumulation_rules, memory_rules
        defaults["group_rules"] = (flat_elementwise.flat_rules(defaults["group_rules"]) + update_schedule.features()
                                   + norm_fusion.features() + matrix_schedule.features()
                                   + accumulation_rules.features() + memory_rules.features())
        # LATCH-TINYMOE-MATRIX research screens only (never a released selection path): LATCH_DIAGNOSTIC_CTRACE=1
        # adds per-CTA phase-end globaltimer stamps written on profile launches; LATCH_SCREEN_RULES='+a+b' appends
        # existing group-rule features so their effect can be measured before a structural rule is adopted.
        import os
        if os.environ.get("LATCH_DIAGNOSTIC_CTRACE") in ("1", "interval"):
            # LATCH-TRAIN-SYNC: "interval" stamps only each interval's start/end (lower._ctrace_phase_stamps), so
            # the update-schedule work queues, which need runs of phases separated only by CTA barriers, survive.
            defaults["group_rules"] += "+ctrace"
        if os.environ.get("LATCH_SCREEN_RULES"):
            from .group_rules import features_of
            features_of(os.environ["LATCH_SCREEN_RULES"])   # validates the names
            defaults["group_rules"] += os.environ["LATCH_SCREEN_RULES"]
    defaults.update(options)
    matrix_provider = None
    if standalone and _LOWERING_DEPTH.get() == 1:
        from .matrix_adapters import MatrixProvider
        from .matrix_adapters.requirements import ProgramResources
        from .matrix_adapters.provider import compose_policy, current_choices
        from .serial_bf16 import KEY as serial_key
        from .matrix_accumulation import KEY as matrix_accumulation_key
        from .matrix_splitk import KEY as matrix_splitk_key
        metadata = graph.get('metadata', {}) if isinstance(graph, dict) else graph.metadata
        serial_record = metadata.get(serial_key)
        # current_choices() is empty unless ops.matrix_tuning installed explicit per-problem choices (it may also
        # name the default policy). Step 13: compose_policy() resolves the structural default rule.
        matrix_provider = MatrixProvider(**{**current_choices(), "policy": compose_policy()},
                                         **({"resources": ProgramResources(dynamic_shared_limit=shared_limit)}
                                            if static_shared_reserve_bytes else {}),
                                         only=set(PROVIDER_BODIES), serial_bf16=serial_record,
                                         matrix_accumulation_contract=metadata.get(matrix_accumulation_key),
                                         matrix_splitk_contract=metadata.get(matrix_splitk_key))
        defaults["matrix_provider"] = matrix_provider
    unknown = set(defaults) - set(inspect.signature(_lower_serial).parameters)
    if unknown:
        raise LoweringError(f"unsupported compose body options: {sorted(unknown)}")
    _validate_flat_precision_policy(current_precision(), {}, {})
    # 'dfast' (step-14 speed diagnostic only, never a default): approximate activations to bound epilogue math cost.
    token = FAST_ACTIVATION.set(current_precision() == "practical" or FAST_ACTIVATION.get()
                                or "dfast" in str(options.get("group_rules") or ""))
    # Release step 14: the flat planner honours the loop planner's 32-bit / identity index rules when its group
    # rules name them (the same _INDEX_MODE the bounded-loop path sets; exact addresses, index spaces < 2**31).
    from .group_rules import features_of
    from .lower import _INDEX_MODE
    rule_features = features_of(defaults.get("group_rules"))
    index_token = (_INDEX_MODE.set("u32id" if "idxid" in rule_features else "u32")
                   if "idx32" in rule_features else None)
    from .matrix_adapters.bodies import TMA_DEEP_RINGS, TMA_EXTRA_TILES, BATCH_TMA_PROGRAM
    tiles_token = TMA_EXTRA_TILES.set("w96" in rule_features)
    flat_token = STEP14_FLAT_ACTIVE.set(standalone and _LOWERING_DEPTH.get() == 1)
    deep_token = TMA_DEEP_RINGS.set("tdeep" in rule_features)
    from .matrix_rules import FP32_EDGE_QUEUE, FP32_EDGE_SPLIT_FORCE
    split_token = FP32_EDGE_SPLIT_FORCE.set(8 if "fs8" in rule_features else 6 if "fs6" in rule_features else None)
    queue_token = FP32_EDGE_QUEUE.set("ectaq" in rule_features and "ecta" in rule_features)
    batch_policy = batch_tma_policy(graph)
    batch_token = BATCH_TMA_PROGRAM.set(batch_policy["batch_stacked_tma"])
    column_probe = [] if selection_probe else None
    column_token = _FLAT_COLSPLIT_PROBE.set(column_probe)
    from .vocab_head import _STATS_PROBE
    stats_probe = [] if selection_probe else None
    stats_token = _STATS_PROBE.set(stats_probe)
    try:
        # lower() already holds the shared lowering lock; the established
        # configuration context restores these donor settings on every exit.
        configuration_values = {"p5c_matrix." + key: value for key, value in FLAT_SELECTED_DEFAULTS.items()}
        if static_shared_reserve_bytes:
            from . import p5c_matrix
            # The provider and native/grouped P5C path must use the same cap.
            # Other bodies retain their existing plans; the final check below
            # refuses any requirement for which no capped planner exists.
            for key in ("P4B_SHARED_LIMIT", "P5C_SHARED_LIMIT", "P8WS_SHARED_LIMIT", "P9WS_SHARED_LIMIT"):
                configuration_values["p5c_matrix." + key] = min(getattr(p5c_matrix, key), shared_limit)
            configuration_values["p5c_matrix.P8WS_FIT_SHARED_LIMIT"] = True
        configuration = (_configuration("reference", configuration_values) if standalone else nullcontext())
        from . import norm_fusion
        fusion_report = None
        if standalone and _LOWERING_DEPTH.get() == 1 and norm_fusion.current() is not None:
            # LATCH-TINYMOE-FUSE: recognized normalization/activation IR rules (ops.norm_fusion).
            graph, fusion_report = norm_fusion.apply_ir(graph)
        from . import accumulation_rules
        accumulation_report = None
        if standalone and _LOWERING_DEPTH.get() == 1 and accumulation_rules.current() is not None:
            # LATCH-SELECTIVE-10: gradient-traffic IR rules (ops.accumulation_rules).
            graph, accumulation_report = accumulation_rules.apply_ir(graph)
        from . import memory_rules
        memory_report = None
        if standalone and _LOWERING_DEPTH.get() == 1 and memory_rules.remat_mode() is not None:
            # LATCH-TRAIN-MEMORY: late-operand rematerialization (ops.memory_rules "remat"; placed by lsink).
            graph, memory_report = memory_rules.apply_ir(graph)
        with configuration, norm_fusion.column_sums((fusion_report or {}).get("column_sums", ())), \
                norm_fusion.rope_values(((fusion_report or {}).get("rope") or {}).get("values", ())):
            program = _lower_serial(graph, fuse_elementwise=fuse_elementwise, **defaults)
            if fusion_report is not None:
                program.manifest["norm_fusion"] = fusion_report
            if accumulation_report is not None:
                program.manifest["accumulation_rules"] = accumulation_report
            if memory_report is not None:
                program.manifest["memory_rules"] = memory_report
            if static_shared_reserve_bytes:
                required = program.manifest["shared_bytes"]
                if required > shared_limit:
                    raise LoweringError(f"program needs {required} dynamic shared bytes; static reserve "
                                        f"{static_shared_reserve_bytes} leaves {shared_limit}")
                program.manifest["shared_memory_budget"] = {
                    "version": 1, "total_optin_limit_bytes": 232448,
                    "static_reserve_bytes": static_shared_reserve_bytes,
                    "dynamic_limit_bytes": shared_limit, "dynamic_required_bytes": required,
                    "admission": "declared reserve; actual compiled static plus dynamic CUDA admission remains mandatory",
                }
            program.manifest["batch_tma_selection"] = batch_policy
            if column_probe:
                program.manifest["deferred_column_splits"] = column_probe
            if stats_probe:
                program.manifest["deferred_vocabulary_stats"] = stats_probe
            return program, matrix_provider
    finally:
        FAST_ACTIVATION.reset(token)
        if index_token is not None:
            _INDEX_MODE.reset(index_token)
        TMA_EXTRA_TILES.reset(tiles_token)
        STEP14_FLAT_ACTIVE.reset(flat_token)
        TMA_DEEP_RINGS.reset(deep_token)
        FP32_EDGE_SPLIT_FORCE.reset(split_token)
        FP32_EDGE_QUEUE.reset(queue_token)
        BATCH_TMA_PROGRAM.reset(batch_token)
        _FLAT_COLSPLIT_PROBE.reset(column_token)
        _STATS_PROBE.reset(stats_token)
