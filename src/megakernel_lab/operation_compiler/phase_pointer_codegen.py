"""Optional phase-local opaque pointers for large pure elementwise intervals.

Only pointer derivation moves. Phase bodies, their calls, publication order,
barriers, ticks and commits are byte-preserved. Runtime's argument pointer table
is a separate immutable allocation (runtime.Runtime); tensor data cannot alias
that allocation. This permits delaying its reads to each consuming phase.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

from . import elementwise_codegen as ec

Refusal = ec.Refusal
KEY = "phase_pointer_codegen"
MARKER = "latch_phase_pointers"
MIN_PHASES = 32
OPAQUE = re.compile(r"(?m)^float\*\* latch_pa = latch_opq\(args\); float\* latch_pr = latch_opq\(arena\);"
                    r"(?: float\*\* latch_po = latch_opq\(out\);)?\n")
POINTER = re.compile(r"(float|double|__nv_bfloat16|int64_t|bool)\* (v_\w+) = (.+);\n")
SCALAR = re.compile(r'unsigned (lsr_tx|lsr_bx|lsr_nt|lsr_nb) = (threadIdx|blockIdx|blockDim|gridDim)\.x; '
                    r'asm volatile\("" : "\+r"\(\1\)\);\n')
SCALAR_SOURCE = dict(lsr_tx="threadIdx", lsr_bx="blockIdx", lsr_nt="blockDim", lsr_nb="gridDim")
HELPER = re.compile(r"static __device__ __noinline__ void (latch_ec_body\d+)\(([^\n]*)\) \{\n")
OPAQUE_HELPER = ('template <class T> __device__ __forceinline__ T* latch_opq(T* p) '
                 '{ asm volatile("" : "+l"(p)); return p; }')


def _braces(source):
    stack, pairs = [], {}
    # Ignore comments and quoted assembly. No regex substitution of C scopes.
    for token in ec.LEX.finditer(source):
        if token.lastgroup != "other":
            continue
        if token[0] == "{":
            stack.append(token.start())
        elif token[0] == "}":
            if not stack:
                raise Refusal("unbalanced source scope")
            pairs[stack.pop()] = token.start()
    if stack:
        raise Refusal("unbalanced source scope")
    return pairs


def _pointer(line):
    match = POINTER.fullmatch(line)
    if not match:
        raise Refusal("unproven pointer declaration or qualifier")
    ty, name, expression = match.groups()
    # Exact ordinary emitter forms, with typed reinterpretation and a literal
    # view offset. Never evaluate a dependent tensor-pointer expression.
    base = r"(?:latch_pa\[\d+\]|latch_po\[\d+\]|latch_pr \+ \d+LL)"
    address = base if ty == "float" else rf"reinterpret_cast<{ty}\*>\({base}\)"
    if not re.fullmatch(address + r"(?: \+ \d+LL)?", expression):
        raise Refusal("unproven or dependent pointer expression")
    return name, ty + "*"


def _helper_bodies(source, manifest, pairs):
    records = {g["helper"]: g for g in manifest.get("elementwise_codegen", {}).get("groups", ())}
    found = {}
    for match in HELPER.finditer(source):
        name = match[1]
        if name in found or name not in records:
            raise Refusal("unbound or duplicate compact helper")
        body = source[match.end():pairs[match.end() - 2]]
        if hashlib.sha256(body.encode()).hexdigest() != records[name]["canonical_sha256"]:
            raise Refusal("compact helper identity differs")
        parameters = []
        for item in match[2].split(", ") if match[2] else []:
            parameter = re.fullmatch(r"(float\*|double\*|__nv_bfloat16\*|int64_t\*|bool\*|unsigned) (latch_ec_p\d+)", item)
            if not parameter:
                raise Refusal("unproven compact helper qualifier or type")
            parameters.append(parameter.groups())
        if [ty for ty, _ in parameters] != records[name]["parameter_types"]:
            raise Refusal("compact helper signature differs")
        found[name] = parameters, body, set(records[name]["phases"])
    if found.keys() != records.keys():
        raise Refusal("compact helper declaration missing")
    return found


def _pure_body(phase, body, types, helpers):
    call = re.fullmatch(r"(latch_ec_body\d+)\(([^;\n]*)\);\n", body)
    expanded = body
    if call:
        if call[1] not in helpers:
            raise Refusal("unknown compact helper")
        parameters, code, phases = helpers[call[1]]
        actuals = call[2].split(", ") if call[2] else []
        if phase not in phases or len(actuals) != len(parameters):
            raise Refusal("compact helper call differs")
        bindings = {}
        for (ty, formal), actual in zip(parameters, actuals):
            if ty == "unsigned":
                if actual not in ec.SCALARS:
                    raise Refusal("unproven compact scalar binding")
            elif types.get(actual) != ty:
                raise Refusal("unproven compact pointer binding")
            bindings[formal] = actual
        # Validation only: restore bound names and rename helper-local registers
        # away from canonicalize's reserved prefix. Original call stays literal.
        expanded = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: bindings.get(m[0],
            "phase_validation_" + m[0][len(ec.PREFIX):] if m[0].startswith(ec.PREFIX) else m[0]), code)
    ec.canonicalize(expanded, types)
    return set(re.findall(r"\bv_\w+\b", body))


def fold(program):
    """Relocate plain pointer declarations in admitted common-runtime scopes."""
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if KEY in manifest or MARKER in source:
        raise Refusal("already localized or reserved marker")
    if (manifest.get("execution_selection", {}).get("selected") != "common_static"
            or manifest.get("barrier_schedule") != "dependency"
            or not {"lptr", "lsreg"} <= set(manifest.get("group_rules", {}).get("local_pointers", ()))
            or any(manifest.get(k) for k in ("region_body", "stages", "dynamic_batch", "output_placement"))
            or len(list(ec.KERNEL.finditer(source))) != 1):
        raise Refusal("requires ordinary static common Runtime with opaque interval pointers")
    definitions = re.findall(r"(?m)^[^\n]*\blatch_opq\([^;\n]*\) \{[^\n]*", source)
    if definitions != [OPAQUE_HELPER] or re.search(r"(?m)^\s*#\s*define\s+latch_opq\b", source):
        raise Refusal("opaque helper is not the side-effect-free ordinary identity")
    pairs = _braces(source)
    helpers = _helper_bodies(source, manifest, pairs)
    phases = manifest["phases"]
    bodies = list(ec._phase_bodies(source))
    if {i for i, _, _ in bodies} != set(range(len(phases))) or len(bodies) != len(phases):
        raise Refusal("phase source/manifest identity differs")
    scopes = []
    for opaque in OPAQUE.finditer(source):
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
        if not members:  # Remaining state-copy scope is never transformed.
            continue
        if interval >= len(intervals) or sorted(i for i, _, _ in members) != sorted(intervals[interval]):
            raise Refusal("publication interval membership differs")
        if len(members) < MIN_PHASES:
            continue
        if any(phases[i].get("kind") != "elementwise" or phases[i].get("counter_effects")
               or phases[i].get("private_scratch") or phases[i].get("requires_full_grid") for i, _, _ in members):
            refused["non-pure interval or cross-phase resources"] += 1; continue
        try:
            head = source.rfind("// Phase ", opaque.end(), members[0][1])
            if head < opaque.end():
                raise Refusal("phase prelude boundary missing")
            prelude = source[opaque.end():head]
            declarations, types, kept, scalar_names = {}, {}, [], set()
            for line in prelude.splitlines(keepends=True):
                scalar = SCALAR.fullmatch(line)
                if not line.strip():
                    kept.append(line)
                elif scalar and SCALAR_SOURCE[scalar[1]] == scalar[2] and scalar[1] not in scalar_names:
                    scalar_names.add(scalar[1]); kept.append(line)
                else:
                    name, ty = _pointer(line)
                    if name in declarations:
                        raise Refusal("duplicate pointer binding")
                    declarations[name], types[name] = line, ty
            phase_edits, counts = [], []
            for phase, start, end in members:
                body = source[start:end]
                used = _pure_body(phase, body, types, helpers)
                if not used <= declarations.keys():
                    raise Refusal("undeclared phase pointer")
                if not (set(re.findall(r"\blsr_(?:tx|bx|nt|nb)\b", body)) <= scalar_names):
                    raise Refusal("unbound scalar register")
                lines = [line for name, line in declarations.items() if name in used]
                pointer_text = "".join(lines)
                bases = []
                if "latch_pa[" in pointer_text:
                    bases.append("float** latch_pa = latch_opq(args);")
                if "latch_pr +" in pointer_text:
                    bases.append("float* latch_pr = latch_opq(arena);")
                if "latch_po[" in pointer_text:
                    bases.append("float** latch_po = latch_opq(out);")
                prefix = f"// {MARKER}: {phase}\n" + " ".join(bases) + "\n" + pointer_text + f"// {MARKER}: end\n"
                phase_edits.append((start, start, prefix, "")); counts.append(len(used))
            # Nothing outside phase bodies may use the delayed pointers/bases.
            remaining, pos = [], head
            for _, start, end in members:
                remaining.append(source[pos:start]); pos = end
            remaining.append(source[pos:closing])
            if re.search(r"\bv_\w+\b|\blatch_p[aro]\b", "".join(remaining)):
                raise Refusal("pointer live outside a pure phase")
            replacement = "".join(kept)
            scope_edits = [(opaque.start(), head, replacement, source[opaque.start():head]), *phase_edits]
            edits.extend(scope_edits)
            groups.append(dict(interval=interval, phases=[i for i, _, _ in members],
                interval_pointer_count=len(declarations), max_phase_pointer_count=max(counts, default=0),
                phase_pointer_counts=counts, original_prelude_sha256=hashlib.sha256(source[opaque.start():head].encode()).hexdigest()))
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
    emitted = "".join(pieces)
    result = dict(manifest)
    result[KEY] = dict(version=1, minimum_phases=MIN_PHASES, source_before=manifest["source_hash"],
        source_bytes_before=len(source.encode()), source_bytes_after=len(emitted.encode()),
        localized_intervals=len(groups), localized_phases=sum(len(g["phases"]) for g in groups),
        groups=groups, refused_intervals=dict(sorted(refused.items())),
        proof="unchanged bodies, helper calls, order, barriers, ticks and commits; original plain pointer declarations; "
              "fresh opaque bases per pure phase; byte-exact relocation inverse; all prior manifest semantics preserved",
        pointer_table_contract="ordinary Runtime creates a separate immutable pointer table; tensor data does not alias its storage")
    result["source_hash"] = hashlib.sha256((emitted + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(emitted, result)
