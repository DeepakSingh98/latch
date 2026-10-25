"""Opt-in relocation of pure-exclusive pointers inside mixed intervals.

Resource phases and every binding they can reference stay byte-identical.
Only a pointer whose complete interval use set is confined to proven pure
elementwise bodies moves into those bodies. The ordinary Runtime's separate
immutable pointer table permits the delayed reads, as in phase_pointer_codegen.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from . import elementwise_codegen as ec
from . import phase_pointer_codegen as pp

Refusal = pp.Refusal
KEY = "mixed_phase_pointer_codegen"
MARKER = "latch_mixed_phase_pointers"
MIN_PHASES = pp.MIN_PHASES


def fold(program):
    """Move only independently bound pointers invisible to unsupported work."""
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if any(key in manifest for key in (KEY, pp.KEY)) or any(marker in source for marker in (MARKER, pp.MARKER)):
        raise Refusal("already localized or reserved marker")
    if (manifest.get("execution_selection", {}).get("selected") != "common_static"
            or manifest.get("barrier_schedule") != "dependency"
            or not {"lptr", "lsreg"} <= set(manifest.get("group_rules", {}).get("local_pointers", ()))
            or any(manifest.get(k) for k in ("region_body", "stages", "dynamic_batch", "output_placement"))
            or len(list(ec.KERNEL.finditer(source))) != 1):
        raise Refusal("requires ordinary static common Runtime with opaque interval pointers")
    definitions = re.findall(r"(?m)^[^\n]*\blatch_opq\([^;\n]*\) \{[^\n]*", source)
    if definitions != [pp.OPAQUE_HELPER] or re.search(r"(?m)^\s*#\s*define\s+latch_opq\b", source):
        raise Refusal("opaque helper is not the side-effect-free ordinary identity")
    pairs = pp._braces(source)
    helpers = pp._helper_bodies(source, manifest, pairs)
    phases, bodies = manifest["phases"], list(ec._phase_bodies(source))
    if {i for i, _, _ in bodies} != set(range(len(phases))) or len(bodies) != len(phases):
        raise Refusal("phase source/manifest identity differs")
    scopes = []
    for opaque in pp.OPAQUE.finditer(source):
        opening = opaque.start() - 2
        if source[opening:opaque.start()] != "{\n" or opening not in pairs:
            raise Refusal("opaque prelude lacks exact local scope")
        scopes.append((opening, pairs[opening], opaque))
    intervals = manifest["barrier_intervals"]
    if len(scopes) not in (len(intervals), len(intervals) + 1):
        raise Refusal("opaque publication scope count differs")
    groups, edits, refused = [], [], Counter()
    cursor = 0
    for interval, (opening, closing, opaque) in enumerate(scopes):
        members = []
        while cursor < len(bodies) and bodies[cursor][1] < closing:
            row = bodies[cursor]
            if row[1] <= opening or row[2] >= closing:
                raise Refusal("phase escapes its publication scope")
            members.append(row); cursor += 1
        if not members:
            continue
        if interval >= len(intervals) or sorted(i for i, _, _ in members) != sorted(intervals[interval]):
            raise Refusal("publication interval membership differs")
        if len(members) < MIN_PHASES:
            continue
        try:
            head = source.rfind("// Phase ", opaque.end(), members[0][1])
            if head < opaque.end():
                raise Refusal("phase prelude boundary missing")
            # Retain the original raw-text admission context of emitter bodies
            # moved into typed helpers, including lexical tokens in comments.
            outlined_bases = any(phases[i].get("indexed_segmented_codegen", {}).get("resource_pointer_lexical_bases")
                                 for i, _, _ in members)
            if outlined_bases or re.search(r"\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b", source[head:closing]):
                raise Refusal("unproven direct base or pointer-table access")
            declarations, types, prelude, scalar_names = {}, {}, [], set()
            for line in source[opaque.end():head].splitlines(keepends=True):
                scalar = pp.SCALAR.fullmatch(line)
                if not line.strip():
                    prelude.append((None, line))
                elif scalar and pp.SCALAR_SOURCE[scalar[1]] == scalar[2] and scalar[1] not in scalar_names:
                    scalar_names.add(scalar[1]); prelude.append((None, line))
                else:
                    name, ty = pp._pointer(line)
                    if name in declarations:
                        raise Refusal("duplicate pointer binding")
                    declarations[name], types[name] = line, ty
                    prelude.append((name, line))
            admitted, uses, phase_refusals = set(), {}, Counter()
            outside, pos = [], head
            for phase, start, end in members:
                body, metadata = source[start:end], phases[phase]
                uses[phase] = set(re.findall(r"\bv_\w+\b", body))
                outside.append(source[pos:start]); pos = end
                if metadata.get("kind") != "elementwise" or any(metadata.get(k) for k in
                        ("counter_effects", "private_scratch", "requires_full_grid")):
                    continue
                try:
                    pp._pure_body(phase, body, types, helpers)
                    if not uses[phase] <= declarations.keys():
                        raise Refusal("undeclared phase pointer")
                    if not set(re.findall(r"\blsr_(?:tx|bx|nt|nb)\b", body)) <= scalar_names:
                        raise Refusal("unbound scalar register")
                    admitted.add(phase)
                except Refusal as error:
                    phase_refusals[str(error)] += 1
            outside.append(source[pos:closing])
            if len(admitted) < MIN_PHASES:
                raise Refusal("too few independently proven pure phases")
            retained = set(re.findall(r"\bv_\w+\b", "".join(outside)))
            for phase, _, _ in members:
                if phase not in admitted:
                    retained.update(uses[phase])
            exclusive = set().union(*(uses[i] for i in admitted)) - retained
            if not exclusive:
                raise Refusal("no pure-exclusive pointers")
            phase_edits, selected, counts, total_counts = [], [], [], []
            for phase, start, _ in members:
                local = uses[phase] & exclusive if phase in admitted else set()
                if not local:
                    continue
                pointer_text = "".join(line for name, line in declarations.items() if name in local)
                bases = []
                if "latch_pa[" in pointer_text: bases.append("float** latch_pa = latch_opq(args);")
                if "latch_pr +" in pointer_text: bases.append("float* latch_pr = latch_opq(arena);")
                if "latch_po[" in pointer_text: bases.append("float** latch_po = latch_opq(out);")
                prefix = f"// {MARKER}: {phase}\n" + " ".join(bases) + "\n" + pointer_text + f"// {MARKER}: end\n"
                phase_edits.append((start, start, prefix, ""))
                selected.append(phase); counts.append(len(local)); total_counts.append(len(uses[phase]))
            # Keep the original opaque bases and every non-moved declaration
            # in place and in order. Unsupported bodies never see a new scope.
            replacement = source[opaque.start():opaque.end()] + "".join(
                line for name, line in prelude if name not in exclusive)
            edits.extend([(opaque.start(), head, replacement, source[opaque.start():head]), *phase_edits])
            groups.append(dict(interval=interval, first_phase=members[0][0], phases=selected, admitted_phases=sorted(admitted),
                unchanged_phases=[i for i, _, _ in members if i not in selected],
                interval_pointer_count=len(declarations), localized_pointer_count=len(exclusive),
                retained_pointer_count=len(declarations) - len(exclusive),
                max_phase_local_pointer_count=max(counts), max_phase_total_pointer_count=max(total_counts),
                phase_local_pointer_counts=counts, refused_pure_phases=dict(sorted(phase_refusals.items())),
                original_prelude_sha256=hashlib.sha256(source[opaque.start():head].encode()).hexdigest()))
        except Refusal as error:
            refused[str(error)] += 1
    if cursor != len(bodies):
        raise Refusal("unscoped manifest phase")
    pieces, inverse, cursor = [], [], 0
    for start, end, replacement, old in sorted(edits):
        if start < cursor or source[start:end] != old:
            raise Refusal("overlapping or changed relocation")
        pieces.extend((source[cursor:start], replacement)); inverse.extend((source[cursor:start], old)); cursor = end
    pieces.append(source[cursor:]); inverse.append(source[cursor:])
    if "".join(inverse) != source:
        raise Refusal("pointer relocation byte inverse failed")
    emitted, result = "".join(pieces), dict(manifest)
    result[KEY] = dict(version=1, minimum_phases=MIN_PHASES, source_before=manifest["source_hash"],
        source_bytes_before=len(source.encode()), source_bytes_after=len(emitted.encode()),
        localized_intervals=len(groups), localized_phases=sum(len(g["phases"]) for g in groups), groups=groups,
        refused_intervals=dict(sorted(refused.items())),
        proof="only pure-exclusive plain pointer declarations move; every resource/shared binding, body, helper call, "
              "barrier, counter, tick and commit remains byte-identical; exact inverse and prior manifest semantics",
        pointer_table_contract="ordinary Runtime creates a separate immutable pointer table; tensor data does not alias its storage")
    result["source_hash"] = hashlib.sha256((emitted + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted, result)
