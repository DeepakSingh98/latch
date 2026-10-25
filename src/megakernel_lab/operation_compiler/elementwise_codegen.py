"""Checked sharing of pure elementwise bodies in the common compose emitter.

This is deliberately separate from P7-E's phase-loop folding and STEP17's whole
program outlining. Publication intervals, pointer declarations, attention/matrix
bodies, barriers and commits remain in the entry kernel. Only repeated,
self-contained elementwise bodies become typed, non-restrict device calls.
Every replacement is checked by byte-exact inverse expansion before emission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import re

PREFIX = "latch_ec_"
PHASE = re.compile(r"// Phase (\d+):[^\n]*\n\{\n")
LEX = re.compile(r'(?P<comment>//[^\n]*|/\*.*?\*/)|(?P<string>"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')'
                 r'|(?P<ident>[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)|(?P<number>\d[\w.]*)'
                 r'|(?P<space>\s+)|(?P<other>.)', re.S)
POINTER = re.compile(r"(?m)^\s*(float|double|__nv_bfloat16|int64_t|bool)\* (v_\w+) = [^;\n]+;")
KERNEL = re.compile(r'extern "C" __global__ (?:__launch_bounds__\(\d+, ?\d+\) )?void '
                    r'(?:__launch_bounds__\(\d+, ?\d+\) )?latch_ops\(')
TYPES = frozenset("float double int unsigned signed long short char bool int64_t int32_t uint64_t uint32_t "
                  "size_t __nv_bfloat16 __nv_bfloat162 float2 float4 uint2 uint4".split())
QUALIFIERS = frozenset(("const",))
KEYWORDS = frozenset(("for", "if", "else", "true", "false", "reinterpret_cast", "static_cast"))
PURE_CALLS = frozenset("sqrtf rsqrtf expf expm1f logf log1pf powf fabsf fmaxf fminf fmaf sinf cosf tanhf "
                       "floorf ceilf truncf __expf __fdividef __frcp_rn __int_as_float __float_as_int "
                       "__float2bfloat16_rn __bfloat162float __bfloat16_as_ushort __ushort_as_bfloat16 "
                       "make_float2 make_float4 make_uint2 make_uint4".split())
SCALARS = {"lsr_tx": "unsigned", "lsr_bx": "unsigned", "lsr_nt": "unsigned", "lsr_nb": "unsigned"}
BUILTINS = frozenset(("threadIdx", "blockIdx", "blockDim", "gridDim"))
FORBIDDEN = re.compile(r"\b(?:asm|volatile|__shared__|smem|grid|ticks|args|arena|out|return|break|continue|goto|"
                       r"switch|while|do|auto|new|delete|throw|try|__syncthreads\w*|__shfl\w*|atomic\w*|"
                       r"mbarrier\w*|wgmma\w*|__syncwarp|__threadfence\w*)\b|cp\.async")


class Refusal(ValueError):
    """A body or program does not satisfy this narrow source contract."""


@dataclass(frozen=True)
class Body:
    phase: int
    start: int
    end: int
    original: str
    canonical: str
    parameters: tuple[tuple[str, str], ...]  # type, original argument
    inverse: dict[str, str]


def _tokens(text):
    tokens = [(m.lastgroup, m[0], m.start(), m.end()) for m in LEX.finditer(text)]
    if "".join(t[1] for t in tokens) != text:
        raise Refusal("incomplete tokenization")
    return tokens


def _structure(tokens):
    """Matched delimiters and lexical lifetimes, including braced for loops."""
    stack, pairs, scopes = [], {}, [(-1, len(tokens))]
    for i, (_, word, _, _) in enumerate(tokens):
        if word in ("(", "[", "{"):
            stack.append((word, i))
        elif word in (")", "]", "}"):
            if not stack or stack[-1][0] != {")": "(", "]": "[", "}": "{"}[word]:
                raise Refusal("unbalanced delimiters")
            opening, start = stack.pop(); pairs[start] = i
            if opening == "{":
                scopes.append((start, i))
    if stack:
        raise Refusal("unbalanced delimiters")
    for i, (_, word, _, _) in enumerate(tokens):
        if word == "for":
            if i + 1 not in pairs or tokens[i + 1][1] != "(":
                raise Refusal("unrecognized loop")
            close = pairs[i + 1]
            if close + 1 not in pairs or tokens[close + 1][1] != "{":
                raise Refusal("only braced for loops are admitted")
            scopes.append((i, pairs[close + 1]))
        elif word in ("if", "else"):
            if word == "if":
                if i + 1 not in pairs or tokens[i + 1][1] != "(":
                    raise Refusal("unrecognized conditional")
                close = pairs[i + 1]
                # The emitter uses expression conditions, never C++ condition
                # declarations/initializers with an additional lexical lifetime.
                condition = tokens[i + 2:close]
                if any(t[1] == ";" for t in condition):
                    raise Refusal("conditional initializer")
                for j, token in enumerate(condition[:-2]):
                    if token[1] in TYPES and condition[j + 1][0] == "ident" and condition[j + 2][1] == "=":
                        raise Refusal("conditional declaration")
                start = close + 1
            else:
                start = i + 1
            if start >= len(tokens):
                raise Refusal("missing conditional body")
            if tokens[start][1] == "{":
                continue  # Already represented by the matched brace scope.
            if word == "else" and tokens[start][1] == "if":
                continue  # The nested conditional is checked independently.
            if tokens[start][1] in ("if", "for"):
                raise Refusal("unbraced nested control")
            end = start
            while end < len(tokens) and tokens[end][1] != ";":
                end = pairs[end] + 1 if end in pairs else end + 1
            if end == len(tokens):
                raise Refusal("unterminated conditional statement")
            scopes.append((start - 1, end))
    return pairs, scopes


def _declarations(tokens, pairs, scopes):
    """Finite emitted-C declaration grammar, with comma declarators and scope."""
    declared = defaultdict(list)
    i = 0
    while i < len(tokens):
        if tokens[i][1] not in TYPES | QUALIFIERS:
            i += 1; continue
        j = i
        while j < len(tokens) and tokens[j][1] in TYPES | QUALIFIERS:
            j += 1
        # Casts/type uses are not declarations. Local pointers/references are
        # intentionally not admitted; vector pointer casts remain expressions.
        if (j >= len(tokens) or tokens[j][0] != "ident" or j + 1 >= len(tokens)
                or tokens[j + 1][1] not in ("=", ";", ",", "[")):
            i = j + 1; continue
        end = min(b for a, b in scopes if a < j < b)
        while True:
            name = tokens[j][1]
            if tokens[j + 1][1] == "[":
                raise Refusal("local array address may escape by decay")
            if name.startswith(("v_", PREFIX)) or name in set(SCALARS) | BUILTINS:
                raise Refusal("shadowed external binding")
            declared[name].append((j, end))
            k = j + 1
            while k < len(tokens) and tokens[k][1] not in (",", ";"):
                if k in pairs:
                    k = pairs[k] + 1
                else:
                    k += 1
            if k == len(tokens):
                raise Refusal("unterminated declaration")
            if tokens[k][1] == ";":
                i = k + 1; break
            j = k + 1
            if (j + 1 >= len(tokens) or tokens[j][0] != "ident"
                    or tokens[j + 1][1] not in ("=", ";", ",", "[")):
                raise Refusal("unrecognized comma declarator")
    return declared


def _substitute(text, mapping):
    out, previous = [], None
    for kind, word, _, _ in _tokens(text):
        out.append(mapping.get(word, word) if kind == "ident" and previous != "." else word)
        if kind not in ("space", "comment"):
            previous = word
    return "".join(out)


_MUTATION_AFTER = re.compile(r"\s*\)*\s*(?:=(?!=)|[+*/%&|^\-]=|\+\+|--)")


def _escape_before(text, end, operators):
    """LATCH-BUILD-600: ``re.search(r"(?:OPS)\\s*\\(*\\s*$", text[:end])`` without copying or rescanning the
    prefix: the match is an operator followed by whitespace, open parentheses, whitespace and the end, and the
    longest such tail (scanned backwards) is the only one an operator can precede."""
    j = end
    while j and text[j - 1].isspace():
        j -= 1
    while j and text[j - 1] == "(":
        j -= 1
    while j and text[j - 1].isspace():
        j -= 1
    return any(text.endswith(op, 0, j) for op in operators)


def canonicalize(text, pointer_types):
    """Return checked canonical text, typed arguments and inverse names."""
    tokens = _tokens(text)
    executable = "".join(word if kind != "comment" else " " * len(word) for kind, word, _, _ in tokens)
    if FORBIDDEN.search(executable) or PREFIX in text or any(t[0] == "string" for t in tokens):
        raise Refusal("control/shared/async/atomic or reserved source")
    # Only this harmless directive is admitted; retain its exact original bytes.
    directives = list(re.finditer(r"(?m)^[ \t]*#[^\n]*", executable))
    if any(not re.fullmatch(r"\s*#pragma\s+unroll(?:\s+\d+)?", d[0]) for d in directives):
        raise Refusal("unknown preprocessor directive")
    sig = [t for t in tokens if t[0] not in ("space", "comment")]
    pairs, scopes = _structure(sig)
    declared = _declarations(sig, pairs, scopes)
    mapping, inverse, parameters = {}, {}, []
    for i, (kind, word, _, _) in enumerate(sig):
        if kind != "ident":
            continue
        previous = sig[i - 1][1] if i else None
        following = sig[i + 1][1] if i + 1 < len(sig) else None
        if previous == ".":
            if word not in ("x", "y", "z", "w"):
                raise Refusal("unknown vector/builtin member")
            continue
        if word.startswith("v_") or word in SCALARS:
            ty = pointer_types.get(word) if word.startswith("v_") else SCALARS[word]
            if ty is None:
                raise Refusal("unbound or qualified tensor pointer")
            # A function parameter must be used as a value. Mutation/address
            # escape would change the caller's live variable rather than its data.
            if (_MUTATION_AFTER.match(executable, sig[i][3])
                    or _escape_before(executable, sig[i][2], ("++", "--", "&"))):
                raise Refusal("external binding mutation or address escape")
            if word not in mapping:
                replacement = f"{PREFIX}p{len(parameters)}"
                mapping[word] = replacement; inverse[replacement] = word
                parameters.append((ty, word))
            continue
        if word in declared:
            if _escape_before(executable, sig[i][2], ("&",)):
                raise Refusal("local address escape")
            if not any(start <= i < end for start, end in declared[word]):
                raise Refusal("local identifier outside its lexical lifetime")
            if word not in mapping:
                replacement = f"{PREFIX}l{len(mapping)}"
                mapping[word] = replacement; inverse[replacement] = word
            continue
        if word in TYPES | QUALIFIERS | KEYWORDS:
            continue
        if word in ("pragma", "unroll") and any(d.start() <= sig[i][2] < d.end() for d in directives):
            continue
        if word in BUILTINS and following == ".":
            if _escape_before(executable, sig[i][2], ("&",)):
                raise Refusal("builtin address escape")
            continue
        if word in PURE_CALLS and following == "(":
            continue
        raise Refusal("unbound identifier or non-admitted call: " + word)
    if len(parameters) > 64:
        raise Refusal("too many typed arguments")
    canonical = _substitute(text, mapping)
    if _substitute(canonical, inverse) != text:
        raise Refusal("byte-exact identifier re-expansion failed")
    return canonical, tuple(parameters), inverse


def _phase_bodies(source):
    heads = list(PHASE.finditer(source))
    for i, head in enumerate(heads):
        limit = heads[i + 1].start() if i + 1 < len(heads) else len(source)
        depth = 1
        for token in LEX.finditer(source, head.end(), limit):
            if token.lastgroup != "other":
                continue
            if token[0] == "{": depth += 1
            elif token[0] == "}": depth -= 1
            if depth == 0:
                yield int(head[1]), head.end(), token.start()
                break
        else:
            raise Refusal("phase boundary could not be proven")


def fold(program):
    """Apply optional sharing after ordinary common lowering/finalization."""
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if PREFIX in source or "elementwise_codegen" in manifest:
        raise Refusal("already folded or reserved helper prefix")
    kernels = list(KERNEL.finditer(source))
    if len(kernels) != 1:
        raise Refusal("requires one ordinary entry kernel")
    pointer_types = {}
    for match in POINTER.finditer(source):
        ty, name = match[1] + "*", match[2]
        if name in pointer_types and pointer_types[name] != ty:
            raise Refusal("inconsistent tensor pointer type")
        pointer_types[name] = ty
    # A differently qualified declaration of the same spelling in another
    # interval invalidates that binding globally; never discard volatile/const.
    # Scan the complete finite declaration spelling, including qualifiers on
    # either side of '*'. A post-star volatile pointer cannot become an ordinary
    # by-value helper binding merely because another interval used plain T*.
    for match in re.finditer(r"(?m)^[ \t]*([A-Za-z_][A-Za-z_0-9 \t*&]*?)\b(v_\w+)[ \t]*=", source):
        ty = re.sub(r"\s+", " ", match[1]).strip()
        if match[2] in pointer_types and pointer_types[match[2]] != ty:
            pointer_types.pop(match[2])
    for match in re.finditer(r"(?m)^[ \t]*((?:\w+[ \t]+)*\w+)[ \t]+(lsr_(?:tx|bx|nt|nb))[ \t]*=", source):
        if match[1] != "unsigned":
            raise Refusal("unproven special-register scalar type")
    for name in SCALARS:
        if re.search(r"\bunsigned " + name + r" = ", source) is None:
            # Bodies using this scalar will be refused rather than guessed.
            # Scalar declarations are an invariant of the local-register emitter.
            if re.search(r"\b" + name + r"\b", source):
                raise Refusal("unproven special-register scalar type")
    phases = manifest.get("phases", ())
    groups, refusals, seen = defaultdict(list), Counter(), set()
    for phase, start, end in _phase_bodies(source):
        if phase in seen or not 0 <= phase < len(phases):
            raise Refusal("phase/manifest identity mismatch")
        seen.add(phase)
        if phases[phase].get("kind") != "elementwise":
            continue
        original = source[start:end]
        try:
            canonical, parameters, inverse = canonicalize(original, pointer_types)
        except Refusal as error:
            refusals[str(error)] += 1; continue
        key = canonical, tuple(ty for ty, _ in parameters)
        groups[key].append(Body(phase, start, end, original, canonical, parameters, inverse))
    if len(seen) != len(phases):
        raise Refusal("not every manifest phase has one source body")
    helpers, replacements, records = [], [], []
    for (canonical, types), members in groups.items():
        if len(members) < 2:
            continue
        name = f"{PREFIX}body{len(helpers)}"
        params = ", ".join(f"{ty} {PREFIX}p{i}" for i, ty in enumerate(types))
        helper = f"static __device__ __noinline__ void {name}({params}) {{\n{canonical}}}\n"
        calls = [name + "(" + ", ".join(arg for _, arg in member.parameters) + ");\n" for member in members]
        if len(helper) + sum(map(len, calls)) >= sum(len(m.original) for m in members):
            continue
        helpers.append(helper)
        for member, call in zip(members, calls):
            if _substitute(canonical, member.inverse) != member.original:
                raise Refusal("member re-expansion failed")
            replacements.append((member.start, member.end, call, member.original))
        records.append(dict(helper=name, phases=[m.phase for m in members], parameter_types=list(types),
                            canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest()))
    # Build and independently invert the patch stream: all bytes outside exact
    # eligible bodies remain identical, including interval fences and commits.
    cursor, pieces, restored = 0, [], []
    for start, end, call, original in sorted(replacements):
        pieces.extend((source[cursor:start], call))
        restored.extend((source[cursor:start], original)); cursor = end
    pieces.append(source[cursor:]); restored.append(source[cursor:])
    if "".join(restored) != source:
        raise Refusal("whole-source re-expansion failed")
    transformed = "".join(pieces)
    insertion = kernels[0].start()
    # Kernel declaration precedes every phase, so body edits cannot move it.
    transformed = transformed[:insertion] + "".join(helpers) + transformed[insertion:]
    result = dict(manifest)
    result["elementwise_codegen"] = dict(version=1, mode="elementwise", source_before=manifest["source_hash"],
        source_bytes_before=len(source.encode()), source_bytes_after=len(transformed.encode()),
        shared_bodies=len(helpers), shared_phases=len(replacements), groups=records,
        refused_bodies=dict(sorted(refusals.items())),
        proof="typed non-restrict arguments; unchanged literals; byte-exact member and whole-source re-expansion; "
              "all non-elementwise source, declarations, fences, ticks and commits preserved")
    result["source_hash"] = hashlib.sha256((transformed + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(transformed, result)
