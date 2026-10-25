"""Default-off emitter-level outlining of deterministic segmented reductions.

Only the two complete bodies rendered by indexed_segmented.emit are moved.
The planner still sees their original dependencies, resource metadata and phase
kinds. Typed arguments retain pointer aliasing; shared offsets remain float
offsets. No arithmetic, CTA synchronization or grid publication is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import re

KEY = "indexed_segmented_codegen"
_PREFERENCE = ContextVar("latch_indexed_segmented_codegen", default="off")
_KINDS = frozenset({"index_segment_partials", "embedding_gradient"})
_TYPES = frozenset({"float*", "int64_t*", "__nv_bfloat16*", "unsigned"})
_SCALARS = {f"{name}.x": "unsigned" for name in ("threadIdx", "blockIdx", "blockDim", "gridDim")}
# Tokens, comments and quoted strings are disjoint. Only emitter-owned symbol
# tokens are renamed; a comment or pragma string never changes its spelling.
_TOKENS = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*(?:\.x\b)?')
_IDENT = re.compile(r"[A-Za-z_]\w*\Z")
# The existing resource-pointer pass intentionally checks raw source text,
# including comments. Preserve that exact admission context when its body moves.
_RESOURCE_BASES = re.compile(r"\b(?:args|out|arena|latch_pa|latch_po|latch_pr)\b")


class Refusal(ValueError):
    pass


@contextmanager
def preference(mode="off"):
    """Select compile-time body outlining independently of numerical policy.

    Use around the normal capture.build/lower invocation. It never selects an
    indexed arithmetic engine, changes assembler flags, or invokes a runtime.
    """
    if mode not in ("off", "typed_v1"):
        raise Refusal("indexed codegen preference must be off or typed_v1")
    token = _PREFERENCE.set(mode)
    try:
        yield mode
    finally:
        _PREFERENCE.reset(token)


def current_preference():
    return _PREFERENCE.get()


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _symbols(text):
    return [m.group() for m in _TOKENS.finditer(text) if _IDENT.fullmatch(m.group()) or m.group() in _SCALARS]


def _rename(text, mapping):
    return _TOKENS.sub(lambda m: mapping.get(m.group(), m.group()), text)


def outline(phase, bindings):
    """Return (phase, helper) from a complete emitter-owned body and typed ABI.

    bindings is a sequence so duplicated aliases can be checked before dedup.
    Order is determined by first actual body use, not the caller's mapping order.
    """
    if phase.get("kind") not in _KINDS or KEY in phase:
        raise Refusal("requires one unoutlined segmented partial or final phase")
    body = phase.get("code")
    if not isinstance(body, str) or not body.startswith("{") or not body.endswith("}"):
        raise Refusal("requires a complete emitter-owned scoped body")
    symbols = _symbols(body)
    if any(s.startswith("latch_is_") for s in symbols):
        raise Refusal("reserved indexed helper identifier")
    types = dict(_SCALARS)
    for name, dtype in bindings:
        if not _IDENT.fullmatch(name) or dtype not in _TYPES or dtype == "unsigned":
            raise Refusal("invalid typed pointer binding")
        if name in types and types[name] != dtype:
            raise Refusal("conflicting types for aliased binding")
        types[name] = dtype
    if types.get("smem") != "float*":
        raise Refusal("shared storage must retain float pointer offsets")
    unknown = {s for s in symbols if (s.startswith("v_") and s != "v_") and s not in types}
    if unknown:
        raise Refusal("unbound tensor pointer: " + ", ".join(sorted(unknown)))
    actuals = list(dict.fromkeys(s for s in symbols if s in types))
    if "smem" not in actuals:
        raise Refusal("segmented helper must retain shared storage")
    formals = [f"latch_is_arg{i}" for i in range(len(actuals))]
    normalized = _rename(body, dict(zip(actuals, formals)))
    parameter_types = [types[n] for n in actuals]
    identity = _sha("\n".join(parameter_types) + "\n" + normalized)
    helper = "latch_is_" + phase["kind"] + "_" + identity
    definition = ("static __device__ __noinline__ void " + helper + "(" +
                  ", ".join(t + " " + n for t, n in zip(parameter_types, formals)) + ") " + normalized + "\n")
    call = helper + "(" + ", ".join(actuals) + ");"
    proof = dict(version=1, mode="typed_v1", helper=helper, argument_names=actuals,
                 formal_names=formals, parameter_types=parameter_types,
                 original_body_sha256=_sha(body), normalized_body_sha256=_sha(normalized),
                 definition_sha256=_sha(definition), call_sha256=_sha(call),
                 body_barriers=body.count("__syncthreads();"),
                 resource_pointer_lexical_bases=sorted(set(_RESOURCE_BASES.findall(body))))
    return {**phase, "code": call, KEY: proof}, definition


def validate_lowering(graph, *, compose, common, depth, retain_all, options):
    if current_preference() == "off":
        return
    metadata = graph.get("metadata", {}) if isinstance(graph, dict) else getattr(graph, "metadata", {})
    if (not compose or not common or depth != 1 or retain_all
            or any(hasattr(graph, k) for k in ("max_steps", "true_branch", "nested_operation"))
            or (metadata or {}).get("dynamic_batch")
            or any(k in options for k in ("output_bindings", "region_body", "staged", "barrier_schedule"))):
        raise Refusal("indexed helper outlining requires standalone static common composition")


def annotate(program):
    if current_preference() == "off":
        return program
    if KEY in program.manifest:
        raise Refusal("indexed helper manifest already annotated")
    proofs = [p[KEY] for p in program.manifest.get("phases", ()) if KEY in p]
    if not proofs:
        raise Refusal("indexed helper preference selected no segmented phases")
    if program.manifest.get("execution_selection", {}).get("selected") != "common_static":
        raise Refusal("indexed helper preference requires common_static execution")
    helpers = {p["helper"] for p in proofs}
    for name in helpers:
        if program.source.count("static __device__ __noinline__ void " + name + "(") != 1:
            raise Refusal("missing or duplicate indexed helper definition")
    program.manifest[KEY] = dict(version=1, mode="typed_v1", outlined_phases=len(proofs),
                                unique_helpers=len(helpers), helpers=sorted(helpers),
                                arithmetic="unchanged emitter body and typed ABI")
    return program
