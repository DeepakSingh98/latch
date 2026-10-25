"""Checked outlining of the existing balanced FP32 full-grid sum body.

This is separate from pure elementwise sharing: shared memory, fences and
atomics are admitted only through an exact known reduction template. Caller
flags, initializer phases, publication intervals and all resource metadata stay
in place. Ordinary and elementwise-only lowering never invoke this pass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re

from . import square_sum_association as sums
from .elementwise_codegen import KERNEL, POINTER, SCALARS, Refusal, _phase_bodies


PREFIX = "latch_fs_"
MARKER = "/* latch_fs_member_marker */"
HEAD = "{ const long long o = 0;\nif (lsr_nb <= 1024) {\n  __shared__ unsigned msum_last_;\n"
TAIL = "\n}\n}\n"
ARITHMETIC = ("eight FP32 strided thread lanes, adjacent-pair lane combine; "
              "per-CTA contiguous chunk trees, then fixed-order combine of CTA partials")


def calculation(elements, input_, scratch, output, marker):
    """Exact admitted ordinary body, excluding its caller-owned shared flag."""
    read = f"{input_}[(long long)((unsigned)(r) % {elements}u) * 1LL]"
    lane_loop = sums.lane_loop(read)
    result = f'''  const long long chunk_ = ({elements}LL + gridDim.x - 1) / gridDim.x, lo_ = (long long)blockIdx.x * chunk_;
  const long long hi_ = lo_ + chunk_ < {elements}LL ? lo_ + chunk_ : {elements}LL;
  float acc = 0.0f;
{marker}
{lane_loop}
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {{
      ({scratch} + 64LL)[blockIdx.x] = acc;
      __threadfence();
      msum_last_ = atomicAdd(reinterpret_cast<unsigned*>({scratch}), 1u) == gridDim.x - 1u;
    }}
  }}
  __syncthreads();
  if (msum_last_) {{
    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < gridDim.x; b += blockDim.x) acc += __ldcg(({scratch} + 64LL) + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {{
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) {{ {output}[0] = acc; *reinterpret_cast<unsigned*>({scratch}) = 0u; }}
    }}
  }}
  __syncthreads();'''
    for old, new in (("threadIdx.x", "lsr_tx"), ("blockIdx.x", "lsr_bx"),
                     ("blockDim.x", "lsr_nt"), ("gridDim.x", "lsr_nb")):
        result = result.replace(old, new)
    return result


def ordinary_body(elements, input_, scratch, output, marker):
    return HEAD + calculation(elements, input_, scratch, output, marker) + TAIL


def _bindings(source):
    pointers = {}
    for match in POINTER.finditer(source):
        ty = match[1] + "*"
        if match[2] in pointers and pointers[match[2]] != ty:
            raise Refusal("inconsistent tensor pointer type")
        pointers[match[2]] = ty
    for match in re.finditer(r"(?m)^[ \t]*([A-Za-z_][A-Za-z_0-9 \t*&]*?)\b(v_\w+)[ \t]*=", source):
        ty = re.sub(r"\s+", " ", match[1]).strip()
        if match[2] in pointers and pointers[match[2]] != ty:
            pointers.pop(match[2])
    for match in re.finditer(r"(?m)^[ \t]*((?:\w+[ \t]+)*\w+)[ \t]+(lsr_(?:tx|bx|nt|nb))[ \t]*=", source):
        if match[1] != "unsigned":
            raise Refusal("unproven full-sum special-register type")
    if any(re.search(r"\bunsigned " + name + " = ", source) is None for name in SCALARS):
        raise Refusal("missing full-sum special-register bindings")
    if "extern __shared__ float smem[];" not in source:
        raise Refusal("full-sum shared arena binding is missing")
    return pointers


def _admit(phase, body, pointers, phases, bodies, offsets, index):
    """Validate the exact complete topology before admitting shared operations."""
    elements = phase.get("reduction_elements")
    if (phase.get("requires_full_grid") is not True or phase.get("max_grid_blocks") != 1024
            or phase.get("arithmetic") != ARITHMETIC or type(elements) is not int
            or not sums.MIN_ELEMENTS <= elements <= 0xffffffff
            or phase.get("targets") != ["aten.sum.default"]
            or len(phase.get("names", ())) != 1 or phase.get("writes") != phase["names"]
            or len(phase.get("reads", ())) != 2):
        raise Refusal("non-admitted full-sum arithmetic, shape or topology")
    input_, scratch = phase["reads"]
    output = phase["names"][0]
    if (phase.get("internal_mutations") != [scratch]
            or phase.get("private_scratch") != [dict(name=scratch, counter_words=1,
                partial_word_offset=64, partial_words=1024)]
            or phase.get("counter_effects") != [dict(storage=scratch, word_offset=0, words=1,
                initialization="zeroed by dependency predecessor", completion="last arriver restores zero")]):
        raise Refusal("full-sum private counter contract differs")
    if any(not re.fullmatch(r"[A-Za-z_]\w*", name) for name in (input_, scratch, output)):
        raise Refusal("invalid full-sum tensor name")
    arguments = tuple("v_" + name for name in (input_, scratch, output))
    if any(pointers.get(name) != "float*" for name in arguments):
        raise Refusal("full-sum tensor binding is not plain FP32")
    initializers = [i for i, row in enumerate(phases) if row.get("writes") == [scratch]
                    and row.get("kind") == "reduction_counter_init"]
    if len(initializers) != 1 or offsets[initializers[0]] >= offsets[index]:
        raise Refusal("full-sum counter initializer is missing or follows its consumer")
    initializer = f"if (lsr_bx == 0 && lsr_tx == 0) *reinterpret_cast<unsigned*>(v_{scratch}) = 0u;\n"
    if bodies[initializers[0]] != initializer:
        raise Refusal("full-sum counter initializer body differs")
    marker = sums.marker(output)
    expected = ordinary_body(elements, *arguments, marker)
    if body != expected:
        raise Refusal("full-sum body differs from the exact admitted template")
    return elements, arguments, marker


def fold(program):
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if PREFIX in source or "full_sum_codegen" in manifest:
        raise Refusal("already outlined or reserved full-sum prefix")
    if (manifest.get("execution_selection", {}).get("selected") != "common_static"
            or manifest.get("full_grid") is not True):
        raise Refusal("full-sum outlining requires standalone common full-grid execution")
    kernels = list(KERNEL.finditer(source))
    if len(kernels) != 1:
        raise Refusal("full-sum outlining requires one ordinary entry kernel")
    pointers = _bindings(source)
    phases = manifest.get("phases", ())
    positions, bodies, offsets = {}, {}, {}
    for index, start, end in _phase_bodies(source):
        if index in positions or not 0 <= index < len(phases):
            raise Refusal("full-sum phase/manifest identity mismatch")
        positions[index] = start, end
        offsets[index] = start
        bodies[index] = source[start:end]
    if len(positions) != len(phases):
        raise Refusal("full-sum phase inventory is incomplete")
    groups, refused = defaultdict(list), Counter()
    for index, phase in enumerate(phases):
        if phase.get("kind") != "reduction" or phase.get("engine") != "multi_cta_full":
            continue
        try:
            elements, arguments, marker = _admit(phase, bodies[index], pointers, phases, bodies, offsets, index)
        except Refusal as error:
            refused[str(error)] += 1
            continue
        groups[elements].append((index, arguments, marker))
    helpers, edits, records = [], [], []
    for elements, members in groups.items():
        if len(members) < 2:
            continue
        name = PREFIX + "body" + str(len(helpers))
        canonical = calculation(elements, PREFIX+"input", PREFIX+"scratch", PREFIX+"output", MARKER)
        canonical = canonical.replace("msum_last_", "(*"+PREFIX+"last)")
        canonical = canonical.replace(MARKER+"\n", "")
        helper = (f"static __device__ __noinline__ void {name}(float* {PREFIX}input, float* {PREFIX}scratch, "
                  f"float* {PREFIX}output, float* smem, unsigned* {PREFIX}last, "
                  "unsigned lsr_tx, unsigned lsr_bx, unsigned lsr_nt, unsigned lsr_nb) {\n"
                  + canonical + "\n}\n")
        replacements = []
        for index, arguments, marker in members:
            call = HEAD + marker + "\n  " + name + "(" + ", ".join(arguments) + ", smem, &msum_last_, lsr_tx, lsr_bx, lsr_nt, lsr_nb);" + TAIL
            # Independently undo the helper's pointer/flag binding and reinsert
            # this output's original association marker at its original point.
            expanded = canonical.replace("(*"+PREFIX+"last)", "msum_last_")
            for formal, actual in zip((PREFIX+"input", PREFIX+"scratch", PREFIX+"output"), arguments):
                expanded = re.sub(r"\b"+formal+r"\b", actual, expanded)
            expanded = expanded.replace("  auto latch_term =", marker+"\n  auto latch_term =", 1)
            if HEAD+expanded+TAIL != bodies[index]:
                raise Refusal("full-sum member re-expansion failed")
            replacements.append((*positions[index], call, bodies[index]))
        if len(helper)+sum(len(e[2]) for e in replacements) >= sum(len(e[3]) for e in replacements):
            continue
        helpers.append(helper); edits.extend(replacements)
        records.append(dict(helper=name, elements=elements, phases=[m[0] for m in members],
            canonical_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            shared_flag="original caller declaration; typed unsigned pointer; no restrict"))
    cursor, pieces, inverse = 0, [], []
    for start, end, replacement, original in sorted(edits):
        pieces.extend((source[cursor:start], replacement))
        inverse.extend((source[cursor:start], original)); cursor=end
    pieces.append(source[cursor:]); inverse.append(source[cursor:])
    if "".join(inverse) != source:
        raise Refusal("full-sum whole-source inversion failed")
    result_source = "".join(pieces)
    point = kernels[0].start()
    result_source = result_source[:point]+"".join(helpers)+result_source[point:]
    result = dict(manifest)
    result["full_sum_codegen"] = dict(version=1, mode="balanced_full_sum", source_before=manifest["source_hash"],
        source_bytes_before=len(source.encode()), source_bytes_after=len(result_source.encode()),
        shared_bodies=len(helpers), shared_phases=len(edits), groups=records,
        refused_bodies=dict(sorted(refused.items())),
        proof="exact admitted FP32 eight-lane full-grid template and private counter topology; "
              "caller shared flags retained; typed non-restrict bindings; byte-exact member/whole-source inversion; "
              "unchanged arithmetic, barriers, publication intervals and all original resource metadata")
    result["source_hash"] = hashlib.sha256((result_source+json.dumps(result["header_hashes"],sort_keys=True)).encode()).hexdigest()
    sums.validate_manifest(result, result_source)
    return GeneratedProgram(result_source, result)
