"""Explicit pointer localization for two exact ordinary resource-phase forms.

After mixed pure-phase localization, admit only the literal counter zero-store
and the existing FP32 vector-cat emitter's complete body. No resource operation,
counter lifetime, shared storage, synchronization or arithmetic moves.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from . import elementwise_codegen as ec
from . import phase_pointer_codegen as pp
from . import mixed_phase_pointer_codegen as mixed

Refusal = pp.Refusal
KEY = "resource_phase_pointer_codegen"
MARKER = "latch_resource_phase_pointers"


def _prefix(marker, phase, declarations):
    text = "".join(declarations.values())
    bases = []
    if "latch_pa[" in text: bases.append("float** latch_pa = latch_opq(args);")
    if "latch_pr +" in text: bases.append("float* latch_pr = latch_opq(arena);")
    if "latch_po[" in text: bases.append("float** latch_po = latch_opq(out);")
    return f"// {marker}: {phase}\n" + " ".join(bases) + "\n" + text + f"// {marker}: end\n"


def _without_prior_prefix(phase, body, prior_phases):
    if phase not in prior_phases:
        return body
    prefix = re.match(rf"// {mixed.MARKER}: {phase}\n([^\n]*)\n(.*?)// {mixed.MARKER}: end\n", body, re.S)
    if not prefix:
        raise Refusal("missing prior mixed pointer prefix")
    declarations = {}
    for line in prefix[2].splitlines(keepends=True):
        name, _ = pp._pointer(line)
        if name in declarations: raise Refusal("duplicate prior local binding")
        declarations[name] = line
    if not declarations or prefix[0] != _prefix(mixed.MARKER, phase, declarations):
        raise Refusal("prior mixed prefix changed")
    return body[prefix.end():]


def _cat_body(metadata, manifest):
    """Reconstruct the finite vector+scalar FP32 cat template, without env knobs."""
    from . import extra_ops
    from .lower import TensorLayout, _ident
    reads, writes = metadata.get("reads", []), metadata.get("writes", [])
    if (metadata.get("targets") != ["aten.cat.default"] or not metadata.get("vector_copy")
            or metadata.get("inplace_pieces") or len(writes) != 1 or metadata.get("names") != writes
            or len(reads) < 2 or len(set(reads + writes)) != len(reads) + 1):
        raise Refusal("unsupported cat metadata")
    descs = manifest.get("layouts", {})
    try:
        if any(descs[n]["dtype"] != "float32" for n in reads + writes):
            raise Refusal("cat must use FP32 storage")
        layouts = {n: TensorLayout(tuple(descs[n]["shape"]), tuple(descs[n]["strides"]),
                                  descs[n]["root"], descs[n].get("offset", 0)) for n in reads + writes}
    except (KeyError, TypeError, ValueError) as error:
        raise Refusal("cat layout unavailable") from error
    name, shape = writes[0], layouts[writes[0]].shape
    if not shape or any(type(x) is not int or x <= 0 for ls in layouts.values() for x in ls.shape):
        raise Refusal("cat requires positive static extents")
    candidates = [d for d in range(len(shape))
        if all(len(layouts[n].shape) == len(shape) and all(layouts[n].shape[j] == shape[j]
                   for j in range(len(shape)) if j != d) for n in reads)
        and sum(layouts[n].shape[d] for n in reads) == shape[d]]
    if len(candidates) != 1: raise Refusal("cat dimension is not uniquely reconstructible")
    args, dtypes, names = (list(reads), candidates[0]), {n: "float32" for n in layouts}, set(layouts)
    fast = extra_ops._vector_copy("cat", args, {}, name, shape, layouts, dtypes, names)
    if fast is None: raise Refusal("cat vector template not admitted")
    code, pointers = fast
    scalar = extra_ops.emit("cat.default", args, {"_p8c_scalar": True}, name, shape,
                            layouts, dtypes, names, None)
    if scalar is None or scalar[0] != "cat": raise Refusal("cat scalar template missing")
    aligned = " && ".join(f"((reinterpret_cast<uintptr_t>({p}) & 15u) == 0)" for p in pointers)
    body = f"if ({aligned}) {{\n{code}\n}} else {{\n{scalar[1]}\n}}\n"
    for register, builtin in pp.SCALAR_SOURCE.items():
        body = body.replace(builtin + ".x", register)
    return body, {_ident(n) for n in reads + writes}


def _admit(body, metadata, manifest, types, scalar_names):
    if any(metadata.get(k) for k in ("counter_effects", "private_scratch", "requires_full_grid")):
        raise Refusal("unsupported resource metadata")
    kind = metadata.get("kind")
    if kind == "reduction_counter_init":
        writes = metadata.get("writes", [])
        if len(writes) != 1 or metadata.get("names") != writes or metadata.get("reads") or metadata.get("targets"):
            raise Refusal("unsupported counter initialization metadata")
        from .lower import _ident
        name = _ident(writes[0])
        expected = f"if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>({name}) = 0u;\n"
        uses = {name}
    elif kind == "cat":
        expected, uses = _cat_body(metadata, manifest)
    else:
        raise Refusal("unsupported phase kind")
    if body != expected: raise Refusal("resource body differs from exact ordinary template")
    if any(types.get(name) != "float*" for name in uses):
        raise Refusal("resource pointer must be an outer plain FP32 binding")
    if not set(re.findall(r"\blsr_(?:tx|bx|nt|nb)\b", body)) <= scalar_names:
        raise Refusal("unbound scalar register")
    return uses


def fold(program):
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if KEY in manifest or MARKER in source: raise Refusal("already resource-localized or reserved marker")
    if mixed.KEY not in manifest or pp.KEY in manifest:
        raise Refusal("requires preceding mixed pointer pass")
    if (manifest.get("execution_selection", {}).get("selected") != "common_static"
            or manifest.get("barrier_schedule") != "dependency"
            or not {"lptr", "lsreg"} <= set(manifest.get("group_rules", {}).get("local_pointers", ()))
            or any(manifest.get(k) for k in ("region_body", "stages", "dynamic_batch", "output_placement"))
            or len(list(ec.KERNEL.finditer(source))) != 1):
        raise Refusal("requires ordinary static common Runtime")
    definitions = re.findall(r"(?m)^[^\n]*\blatch_opq\([^;\n]*\) \{[^\n]*", source)
    if definitions != [pp.OPAQUE_HELPER] or re.search(r"(?m)^\s*#\s*define\s+latch_opq\b", source):
        raise Refusal("opaque helper differs")
    pairs, bodies, phases = pp._braces(source), list(ec._phase_bodies(source)), manifest["phases"]
    if {i for i, _, _ in bodies} != set(range(len(phases))) or len(bodies) != len(phases):
        raise Refusal("phase source/manifest identity differs")
    scopes = [(m.start() - 2, pairs[m.start() - 2], m) for m in pp.OPAQUE.finditer(source)
              if source[m.start() - 2:m.start()] == "{\n" and m.start() - 2 in pairs]
    intervals = manifest["barrier_intervals"]
    if len(scopes) not in (len(intervals), len(intervals) + 1): raise Refusal("publication scope count differs")
    prior_phases = {i for g in manifest[mixed.KEY]["groups"] for i in g["phases"]}
    groups, edits, refused, cursor = [], [], Counter(), 0
    for interval, (opening, closing, opaque) in enumerate(scopes):
        members = []
        while cursor < len(bodies) and bodies[cursor][1] < closing:
            row = bodies[cursor]
            if row[1] <= opening or row[2] >= closing: raise Refusal("phase escapes publication scope")
            members.append(row); cursor += 1
        if not members: continue
        if interval >= len(intervals) or sorted(i for i, _, _ in members) != sorted(intervals[interval]):
            raise Refusal("publication membership differs")
        try:
            head = source.rfind("// Phase ", opaque.end(), members[0][1])
            if head < opaque.end(): raise Refusal("phase prelude missing")
            declarations, types, prelude, scalar_names = {}, {}, [], set()
            for line in source[opaque.end():head].splitlines(keepends=True):
                scalar = pp.SCALAR.fullmatch(line)
                if not line.strip(): prelude.append((None, line))
                elif scalar and pp.SCALAR_SOURCE[scalar[1]] == scalar[2] and scalar[1] not in scalar_names:
                    scalar_names.add(scalar[1]); prelude.append((None, line))
                else:
                    name, ty = pp._pointer(line)
                    if name in declarations: raise Refusal("duplicate pointer binding")
                    declarations[name], types[name] = line, ty; prelude.append((name, line))
            admitted, uses, outside, pos, reasons = set(), {}, [], head, Counter()
            for phase, start, end in members:
                body = source[start:end]
                clean = _without_prior_prefix(phase, body, prior_phases)
                outside.append(source[pos:start]); pos = end
                # Emitter-level segmented outlining moves the complete body,
                # including comments. Keep this pass's established raw-text
                # admission rule even when those lexical tokens live in a helper.
                outlined_bases = phases[phase].get("indexed_segmented_codegen", {}).get("resource_pointer_lexical_bases")
                if outlined_bases or re.search(r"\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b", clean):
                    raise Refusal("unproven direct base or pointer-table access")
                uses[phase] = set(re.findall(r"\bv_\w+\b", body)) & declarations.keys()
                if phases[phase].get("kind") not in ("cat", "reduction_counter_init"): continue
                try:
                    exact = _admit(body, phases[phase], manifest, types, scalar_names)
                    if exact != uses[phase]: raise Refusal("resource use-set differs")
                    admitted.add(phase)
                except Refusal as error: reasons[str(error)] += 1
            outside.append(source[pos:closing])
            outside_text = "".join(outside)
            if re.search(r"\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b", outside_text):
                raise Refusal("unproven interphase pointer-table access")
            retained = set(re.findall(r"\bv_\w+\b", outside_text))
            for phase, _, _ in members:
                if phase not in admitted: retained.update(uses[phase])
            exclusive = (set().union(*(uses[i] for i in admitted)) if admitted else set()) - retained
            if not exclusive:
                refused.update(reasons); continue
            selected, counts, phase_edits = [], [], []
            for phase, start, _ in members:
                local = uses[phase] & exclusive if phase in admitted else set()
                if not local: continue
                selected.append(phase); counts.append(len(local))
                text = _prefix(MARKER, phase, {n: d for n, d in declarations.items() if n in local})
                phase_edits.append((start, start, text, ""))
            replacement = source[opaque.start():opaque.end()] + "".join(line for n, line in prelude if n not in exclusive)
            edits.extend([(opaque.start(), head, replacement, source[opaque.start():head]), *phase_edits])
            groups.append(dict(interval=interval, first_phase=members[0][0], phases=selected,
                unchanged_phases=[i for i, _, _ in members if i not in selected],
                interval_pointer_count=len(declarations), localized_pointer_count=len(exclusive),
                retained_pointer_count=len(declarations)-len(exclusive), max_phase_local_pointer_count=max(counts),
                phase_local_pointer_counts=counts, phase_kinds=dict(Counter(phases[i]["kind"] for i in selected)),
                refused_resource_phases=dict(sorted(reasons.items())),
                original_prelude_sha256=hashlib.sha256(source[opaque.start():head].encode()).hexdigest()))
        except Refusal as error: refused[str(error)] += 1
    if cursor != len(bodies): raise Refusal("unscoped phase")
    pieces, inverse, cursor = [], [], 0
    for start, end, replacement, old in sorted(edits):
        if start < cursor or source[start:end] != old: raise Refusal("overlapping or changed edit")
        pieces.extend((source[cursor:start], replacement)); inverse.extend((source[cursor:start], old)); cursor = end
    pieces.append(source[cursor:]); inverse.append(source[cursor:])
    if "".join(inverse) != source: raise Refusal("resource pointer inverse failed")
    emitted, result = "".join(pieces), dict(manifest)
    result[KEY] = dict(version=1, source_before=manifest["source_hash"], source_bytes_before=len(source.encode()),
        source_bytes_after=len(emitted.encode()), localized_intervals=len(groups),
        localized_phases=sum(len(g["phases"]) for g in groups), groups=groups,
        refused_intervals=dict(sorted(refused.items())),
        proof="exact ordinary FP32 vector-cat and counter-zero templates; only exclusive plain pointer declarations move; "
              "all prior manifest semantics, bodies, resources, counters, barriers, ticks and commits byte-preserved",
        pointer_table_contract="ordinary Runtime creates separate immutable pointer tables not aliased by tensor data")
    result["source_hash"] = hashlib.sha256((emitted + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted, result)
