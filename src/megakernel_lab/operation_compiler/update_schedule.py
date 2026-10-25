"""Opt-in update-schedule rules for flat common compose programs (LATCH-TINYMOE-25).

Selected through the normal build API: ``CapturedModule.build(..., update_schedule="streamed")``.
Omitting it (None) leaves lowering and generated source unchanged.

"streamed_ustate" (TINYMOE-SELECTIVE-CADENCE) is "streamed" plus ``ustate``: a committed parameter or
state value that a tensor rule returned unchanged (capture materializes it as a copy of its own binding)
commits nothing; the copy chain is removed and the returned output names the unchanged binding. Two
programs sharing one persistent state (``ModuleUpdate.load(bindings=)``) can then skip a parameter group's
optimizer work entirely on the steps where that group does not update.

"streamed_ustate_late" adds ``ulate`` (late_updates below): bodies that no loss or gradient output depends on
start no earlier than the interval after the last loss/gradient interval. A rule whose fast-group clip depends on
only part of the gradients (e.g. experts) otherwise lets that group's optimizer start inside the backward
intervals, interleaved with the remaining backward bodies; ``ulate`` keeps the update after the backward, where a
whole-model rule places it anyway.

"streamed" adds three structural rules. None depends on a model name or shape table:

* ``csink`` (group_rules.sink_single_reader_chains): a same-shape elementwise chain read by exactly
  one later chain joins that chain. For a tensor optimizer this places decay, moments and the
  parameter update of each tensor in one loop and one dependency interval.
* (``nosync``, the P6-C barrier elision between shared-memory-free neighbours, is available in the
  flat emitter but not selected: on TinyMoE it slowed the optimizer interval from 4.4 to 7.7 ms.)
* ``fsbatch`` (batch_full_sums): consecutive multi-CTA full sums of one interval share one body;
  each keeps its lanes, CTA tree, partial/fence/ticket order and fixed-order combine.
* ``rotate`` (below): grid-stride elementwise phases of one dependency interval start where the
  previous phase's partial last round ended, so small and odd-sized tensors are spread over all CTAs
  instead of always loading the low CTAs. Every element is still computed exactly once by the same
  expression; only the thread that computes it changes.

"streamed_prep" is "streamed" plus ops.prologue_batch's table rules (LATCH-TINYMOE-PROLOGUE), applied just
before the work queue; their table phases keep the full grid (never queued or rotated).

"streamed_prep_ustate" / "streamed_prep_ustate_late" (LATCH-TINYMOE-COMBINED) are "streamed_ustate" /
"streamed_ustate_late" plus the same prologue table rules (``prologue_tables()``).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import re

SCHEDULES = (None, "streamed", "streamed_prep", "streamed_ustate", "streamed_ustate_late",
             "streamed_prep_ustate", "streamed_prep_ustate_late")
_SCHEDULE = ContextVar("latch_update_schedule", default=None)
_FEATURES = {"streamed": "+csink+fsbatch", "streamed_prep": "+csink+fsbatch",
             "streamed_ustate": "+csink+fsbatch+ustate", "streamed_ustate_late": "+csink+fsbatch+ustate+ulate",
             "streamed_prep_ustate": "+csink+fsbatch+ustate", "streamed_prep_ustate_late": "+csink+fsbatch+ustate+ulate"}
VERSION = 1


def validate(value):
    if value not in SCHEDULES:
        raise ValueError(f"update_schedule must be one of {SCHEDULES}")
    return value


def current():
    return _SCHEDULE.get()


@contextmanager
def scope(value):
    token = _SCHEDULE.set(validate(value))
    try:
        yield
    finally:
        _SCHEDULE.reset(token)


def prologue_tables():
    """ops.prologue_batch table rules are selected (the "streamed_prep" family)."""
    return (_SCHEDULE.get() or "").startswith("streamed_prep")


def features():
    """Extra group-rule features for the standalone flat common planner."""
    return _FEATURES.get(_SCHEDULE.get(), "")


class Refusal(ValueError):
    """The program does not satisfy the narrow rotation source contract."""


_PHASE = re.compile(r"// Phase (\d+):[^\n]*\n\{\n")
_START = re.compile(r"\(long long\)\s*lsr_bx \* lsr_nt \+ lsr_tx|lsr_bx \* lsr_nt \+ lsr_tx")
_CALL = re.compile(r"^(latch_ec_body\d+)\(([^;]*)\);$", re.M)
_HELPER = re.compile(r"static __device__ __noinline__ void (latch_ec_body\d+)\(([^)]*)\) \{\n(.*?)\n\}\n", re.S)
_HELPER_START = re.compile(r"\(long long\)(latch_ec_p\d+) \* (latch_ec_p\d+) \+ (latch_ec_p\d+)")
_HELPER_EXTENT = re.compile(r"< (\d+)LL")
# Flat-index lanes per grid-stride round (vec4u/ilp unrolled bodies): ``+= U LL * (long long)...`` in a helper,
# ``jb_ += U LL * (long long)gridDim.x * blockDim.x`` in an inline vector loop.
_ROUND = re.compile(r"\+= (\d+)LL \* \(long long\)")
_UNSAFE = re.compile(r"\b(?:__shared__|smem|__syncthreads\w*|__shfl\w*|atomic\w*|asm|grid|ticks|cp\.async|"
                     r"mbarrier\w*|wgmma\w*|__syncwarp|__threadfence\w*|threadIdx|blockIdx|gridDim|blockDim|"
                     r"latch_pi_\w+|latch_fs_\w+|lb_|lg_|cta0|ctas)\b")


def _blocks(source):
    heads = list(_PHASE.finditer(source))
    for i, head in enumerate(heads):
        depth, j = 1, head.end()
        limit = heads[i + 1].start() if i + 1 < len(heads) else len(source)
        while j < limit and depth:
            c = source[j]
            depth += (c == "{") - (c == "}")
            j += 1
        if depth:
            raise Refusal("phase boundary could not be proven")
        yield int(head[1]), head.end(), j - 1


def _helpers(source):
    """Folded elementwise helpers that use their CTA/thread indices only as one flat index bx*nt+tx.

    Such a body is a pure function of (bx*nt+tx, grid, pointers) without inter-thread communication;
    permuting the flat index over [0, nb*nt) permutes which thread performs each unit of work and leaves
    the complete set of work and every value unchanged, whatever the loop structure (unrolled, pipelined).
    """
    result = {}
    for match in _HELPER.finditer(source):
        name, params, body = match[1], match[2], match[3]
        starts = _HELPER_START.findall(body)
        extents = [int(x) for x in _HELPER_EXTENT.findall(body)]
        if len(starts) != 1 or not extents or _UNSAFE.search(body):
            continue
        b, n, t = starts[0]
        if len(re.findall(r"\b" + b + r"\b", body)) != 1 or len(re.findall(r"\b" + t + r"\b", body)) != 1:
            continue
        names = [p.split()[-1] for p in params.split(",")]
        rounds = [int(x) for x in _ROUND.findall(body)]
        result[name] = dict(extent=max(extents), block=names.index(b), thread=names.index(t),
                            round=max(rounds, default=1))
    return result


def _extent(text, helpers):
    """Grid-stride extent of an eligible phase body, or None."""
    if _UNSAFE.search(text):
        return None
    calls = _CALL.findall(text)
    if calls:
        total = 0
        for name, args in calls:
            helper = helpers.get(name)
            if helper is None:
                return None
            args = [a.strip() for a in args.split(",")]
            if args[helper["block"]] != "lsr_bx" or args[helper["thread"]] != "lsr_tx":
                return None
            total += helper["extent"]
        if text.count("lsr_bx") != len(calls) or text.count("lsr_tx") != len(calls):
            return None
        return total
    starts = len(_START.findall(text))
    extents = [int(x) for x in re.findall(r"< (\d+)(?:LL|ULL|u)\b", text)]
    if not starts or not extents or text.count("lsr_bx") != starts or text.count("lsr_tx") != starts:
        return None
    return max(extents)


_FS_OPEN = "{ const long long o = 0;\nif (gridDim.x <= 1024) {\n  __shared__ unsigned msum_last_;\n"
_FS_MID = ("  acc = s101h::warp_sum(acc);\n  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;\n"
           "  __syncthreads();\n  if (threadIdx.x < 32) {\n    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;\n"
           "    acc = s101h::warp_sum(acc);\n    if (threadIdx.x == 0) {\n      ")
_FS_REST = re.compile(r"(?P<part>[^\n]+?)\[blockIdx\.x\] = acc;\n      __threadfence\(\);\n"
                      r"      msum_last_ = atomicAdd\((?P<cnt>[^\n]+?), 1u\) == gridDim\.x - 1u;\n    \}\n  \}\n"
                      r"  __syncthreads\(\);\n  if \(msum_last_\) \{\n(?P<combine>.*)\n  \}\n  __syncthreads\(\);\n\}\n\}\Z",
                      re.S)
FS_BATCH = 32


def _full_sum_parts(code):
    """(prelude, calculation, partials, counter, combine) of one exact multi-CTA full-sum body, or None."""
    at = code.find(_FS_OPEN)
    if at < 0 or code.count(_FS_OPEN) != 1:
        return None
    prelude, body = code[:at], code[at + len(_FS_OPEN):]
    if prelude:
        if not (prelude.startswith("{\n") and body.endswith("\n}")):
            return None
        prelude, body = prelude[2:], body[:-2]
    mid = body.find(_FS_MID)
    if mid < 0 or body.count(_FS_MID) != 1:
        return None
    calculation, rest = body[:mid], body[mid + len(_FS_MID):]
    match = _FS_REST.match(rest)
    if match is None or "msum_last_" in calculation + match["combine"] + prelude:
        return None
    rebuilt = (("{\n" + prelude if prelude else "") + _FS_OPEN + calculation + _FS_MID + rest
               + ("\n}" if prelude else ""))
    if rebuilt != code:
        return None
    return prelude, calculation, match["part"], match["cnt"], match["combine"]


_COMBINE = re.compile(
    r"\A    __threadfence\(\);\n    acc = 0\.0f;\n"
    r"    for \(unsigned b = threadIdx\.x; b < gridDim\.x; b \+= blockDim\.x\) acc \+= __ldcg\((?P<part>[^\n]+?) \+ b\);\n"
    r"    acc = s101h::warp_sum\(acc\);\n    if \(\(threadIdx\.x & 31\) == 0\) smem\[threadIdx\.x >> 5\] = acc;\n"
    r"    __syncthreads\(\);\n    if \(threadIdx\.x < 32\) \{\n      acc = threadIdx\.x < 8 \? smem\[threadIdx\.x\] : 0\.0f;\n"
    r"      acc = s101h::warp_sum\(acc\);\n      if \(threadIdx\.x == 0\) \{ (?P<out>[^\n]+?)\[0\] = acc; "
    r"\*(?P<cnt>[^\n]+?) = 0u; \}\n    \}\Z")


def _warp_combine(combine):
    """One warp's exact replica of a full sum's block combine (``fswarp``), or None if the form differs.

    The block combine (256 threads, 8 warps): thread t sums partials b = t, t + 256, ... from +0 in ascending b;
    warp w's butterfly total goes to smem[w]; warp 0 then butterflies smem[0..7] (lanes 8..31 contribute +0).
    The replica walks the 8 virtual warps in one warp: lane l of virtual warp w sums the same partials in the
    same order, the same xor butterfly gives the same total in every lane (each step adds the same two values,
    commutatively), and the final butterfly adds the same 8 totals. Requires blockDim.x == 256 (checked)."""
    m = _COMBINE.match(combine)
    if m is None:
        return None
    part, out, cnt = m["part"], m["out"], m["cnt"]
    # All eight virtual-thread partial loads are issued before the butterflies (independent registers).
    return ("      __threadfence();\n"
            "      float latch_fv_[8];\n"
            "      #pragma unroll\n"
            "      for (unsigned latch_w_ = 0; latch_w_ < 8u; ++latch_w_) {\n"
            "        float acc = 0.0f;\n"
            f"        for (unsigned b = latch_w_ * 32u + (threadIdx.x & 31u); b < gridDim.x; b += 256u) acc += __ldcg({part} + b);\n"
            "        latch_fv_[latch_w_] = acc;\n"
            "      }\n"
            "      float latch_fw_ = 0.0f;\n"
            "      #pragma unroll\n"
            "      for (unsigned latch_w_ = 0; latch_w_ < 8u; ++latch_w_) {\n"
            "        const float acc = s101h::warp_sum(latch_fv_[latch_w_]);\n"
            "        if ((threadIdx.x & 31u) == latch_w_) latch_fw_ = acc;\n"
            "      }\n"
            "      latch_fw_ = s101h::warp_sum(latch_fw_);\n"
            f"      if ((threadIdx.x & 31u) == 0u) {{ {out}[0] = latch_fw_; *{cnt} = 0u; }}\n")


_LANES_CALC = re.compile(
    r"\A  const long long chunk_ = \((?P<n>\d+)LL \+ gridDim\.x - 1\) / gridDim\.x, lo_ = \(long long\)blockIdx\.x \* chunk_;\n"
    r"  const long long hi_ = lo_ \+ chunk_ < (?P=n)LL \? lo_ \+ chunk_ : (?P=n)LL;\n"
    r"  float acc = 0\.0f;\n(?P<marker>/\* square_sum_lanes8:[0-9a-f]+ \*/)\n"
    r"  auto latch_term = \[&\]\(long long r\) -> float \{ return (?P<expr>.+?); \};\n"
    r"  float lane0 = 0\.0f;[^\n]*\n  long long r = lo_ \+ threadIdx\.x;\n.*"
    r"  acc = \(\(lane0 \+ lane1\) \+ \(lane2 \+ lane3\)\) \+ \(\(lane4 \+ lane5\) \+ \(lane6 \+ lane7\)\);\s*\Z", re.S)
FS_MIX_GROUP = 4


# LATCH-TRAIN-SYNC ``fsmixl``: a sum whose summand is evaluated through one-line leaf lambdas defined before its
# exact body (e.g. a participation-masked square ``where(flag, g, 0)**2`` of an expert gradient) is admitted to the
# round-interleaved loop too. Each lambda is defined once per sum under a sum-unique name (word-boundary renaming in
# its own definition and in the summand expression only); its text, and so every summand's value, is unchanged.
_LEAF_DEF = re.compile(r"auto (latch_\w+) = \[&\]\(long long i\) -> float \{[^\n]*\};\n")


def _leaf_prelude(prelude):
    """[(name, definition)] of a prelude made only of one-line ``auto latch_*`` leaf lambdas, or None."""
    defs, at = [], 0
    for m in _LEAF_DEF.finditer(prelude):
        if m.start() != at:
            return None
        defs.append((m[1], m[0]))
        at = m.end()
    if at != len(prelude) or not defs or len({n for n, _ in defs}) != len(defs):
        return None
    return defs


# LATCH-TRAIN-SYNC ``fsload``: a leaf's masked operand ``(FLAG[0] ? P[(i)] : Z)`` (a where-mask over a participation flag)
# is loaded before the selection (``const auto latch_ldJ_ = P[(i)]; ... (FLAG[0] ? latch_ldJ_ : Z)``): i is always an
# index of the summed view (the term wraps r modulo the view's extent), so the load is in bounds; the selected value
# is the same. The compiler may then issue every load of a round together instead of behind its flag test.
_MASKED_LOAD = re.compile(r"\((v_\w+)\[0\] \? (v_\w+)\[\(i\)\] : (r_\w+)\)")


def _hoist_masked_loads(definition):
    head = " -> float { "
    at = definition.find(head)
    if at < 0:
        return definition
    loads, count = [], [0]

    def sub(m):
        name = f"latch_ld{count[0]}_"
        count[0] += 1
        loads.append(f"const auto {name} = {m[2]}[(i)]; ")
        return f"({m[1]}[0] ? {name} : {m[3]})"
    body = _MASKED_LOAD.sub(sub, definition[at + len(head):])
    return definition[:at + len(head)] + "".join(loads) + body if loads else definition


def _wrapped(expr, n):
    """The summand indexes its view only through r wrapped modulo the view's extent n (always in bounds)."""
    return (f"(unsigned)(r) % {n}u" in expr and
            len(re.findall(r"\br\b", expr)) == expr.count(f"(unsigned)(r) % {n}u"))


def _rename(text, names, q):
    for name in names:
        text = re.sub(r"\b" + re.escape(name) + r"\b", f"{name}_m{q}", text)
    return text


def _mixed_calculations(group, unconditional=False):
    """``fsmix``: the partial (thread-lane) sums of several eight-lane square sums in one round-interleaved loop.

    Each sum keeps its chunk, its eight lanes and every lane's summands in ascending r (one 8-wide round at a time,
    then the same predicated tail): only the issue order of independent loads across sums changes, so up to
    FS_MIX_GROUP sums' loads are in flight together. ``group``: [(slot, calculation)] or [(slot, calculation,
    prelude)] (``fsmixl``: one-line leaf lambdas the summand calls, renamed per sum). Returns code or None."""
    parsed = []
    for item in group:
        slot, calc = item[0], item[1]
        prelude = item[2] if len(item) > 2 else ""
        fused = item[3] if len(item) > 3 else None
        m = _LANES_CALC.match(calc)
        if m is None:
            return None
        defs = _leaf_prelude(prelude) if prelude else []
        if defs is None:
            return None
        if unconditional and not _wrapped(m["expr"], m["n"]):
            return None
        if fused is not None:
            # LATCH-TRAIN-SYNC ops.cast_sums: the leaf takes the producer value; the lanes compute and store it.
            from .cast_sums import leaf_value
            value = leaf_value(prelude, "v_" + fused["view"]) if unconditional else None
            if value is None or int(m["n"]) != int(fused["n"]) or m["expr"] != (
                    f"{value[0]}((long long)((unsigned)(r) % {m['n']}u) * 1LL)"):
                return None
            defs = [(value[0], value[1] + "\n")]
        parsed.append((slot, m, defs, fused))
    lines = ["{ // update_schedule fsmix: round-interleaved partial sums"]
    for q, (slot, m, defs, fused) in enumerate(parsed):
        n = m["n"]
        names = [name for name, _ in defs]
        lines += [_rename(_hoist_masked_loads(text) if unconditional and fused is None else text, names, q).rstrip("\n")
                  for _, text in defs]
        if fused is not None:
            index = f"const long long i = {int(fused['offset'])}LL + (long long)((unsigned)(r) % {n}u);"
            term = (f"  auto latch_term{q} = [&](long long r) -> float {{ {index} return {fused['expression']}; }};\n"
                    f"  auto latch_put{q} = [&](long long r, float latch_w) {{ {index} {fused['store']}[i] = latch_w; }};")
        else:
            term = f"  auto latch_term{q} = [&](long long r) -> float {{ return {_rename(m['expr'], names, q)}; }};"
        lines += [f"  const long long chunk{q}_ = ({n}LL + gridDim.x - 1) / gridDim.x, lo{q}_ = (long long)blockIdx.x * chunk{q}_;",
                  f"  const long long hi{q}_ = lo{q}_ + chunk{q}_ < {n}LL ? lo{q}_ + chunk{q}_ : {n}LL;",
                  m["marker"],
                  term,
                  "  " + " ".join(f"float l{q}_{k} = 0.0f;" for k in range(8)),
                  f"  long long r{q} = lo{q}_ + threadIdx.x;"]
    lines.append("  for (;;) {")
    lines.append("    bool latch_more_ = false;")
    if unconditional:
        # fsload: leave before evaluating a round in which no sum of the group has a full round left (the summands
        # are evaluated unconditionally, so an empty final round would load a whole round for nothing).
        lines += [f"    const bool f{q}_ = r{q} + 7LL * blockDim.x < hi{q}_;" for q, _ in enumerate(parsed)]
        lines.append("    if (!(" + " || ".join(f"f{q}_" for q, _ in enumerate(parsed)) + ")) break;")
    for q, _ in enumerate(parsed):
        if not unconditional:
            lines.append(f"    const bool f{q}_ = r{q} + 7LL * blockDim.x < hi{q}_;")
        for k in range(8):
            # fsload: wrapped (in-bounds) summands are evaluated unconditionally; only the additions are guarded.
            lines.append(f"    const float t{q}_{k} = latch_term{q}(r{q} + {k}LL * blockDim.x);" if unconditional else
                         f"    const float t{q}_{k} = f{q}_ ? latch_term{q}(r{q} + {k}LL * blockDim.x) : 0.0f;")
    for q, (_, _, defs, fused) in enumerate(parsed):
        if fused is not None:
            leaf = f"{defs[0][0]}_m{q}"
            lines.append(f"    if (f{q}_) {{ " + " ".join(f"latch_put{q}(r{q} + {k}LL * blockDim.x, t{q}_{k}); "
                                                         f"l{q}_{k} += {leaf}(t{q}_{k});" for k in range(8))
                         + f" r{q} += 8LL * blockDim.x; latch_more_ = true; }}")
            continue
        lines.append(f"    if (f{q}_) {{ " + " ".join(f"l{q}_{k} += t{q}_{k};" for k in range(8))
                     + f" r{q} += 8LL * blockDim.x; latch_more_ = true; }}")
    lines += ["    if (!latch_more_) break;", "  }"]
    if unconditional:
        for q, _ in enumerate(parsed):
            for k in range(8):
                lines.append(f"  const float u{q}_{k} = latch_term{q}(r{q} + {k}LL * blockDim.x);")
    for q, (slot, _, defs, fused) in enumerate(parsed):
        for k in range(8):
            if fused is not None:
                lines.append(f"  if (r{q} + {k}LL * blockDim.x < hi{q}_) {{ latch_put{q}(r{q} + {k}LL * blockDim.x, u{q}_{k}); "
                             f"l{q}_{k} += {defs[0][0]}_m{q}(u{q}_{k}); }}")
                continue
            lines.append(f"  if (r{q} + {k}LL * blockDim.x < hi{q}_) l{q}_{k} += u{q}_{k};" if unconditional else
                         f"  if (r{q} + {k}LL * blockDim.x < hi{q}_) l{q}_{k} += latch_term{q}(r{q} + {k}LL * blockDim.x);")
        lines.append(f"  latch_fsb_acc_[{slot}] = ((l{q}_0 + l{q}_1) + (l{q}_2 + l{q}_3)) + ((l{q}_4 + l{q}_5) + (l{q}_6 + l{q}_7));")
    lines.append("}")
    return "\n".join(lines)


def batch_full_sums(parts, part_ids, phases, warp_combine=False, mix=False, fence=False, leaves=False, stamp=None,
                    unconditional=False, fused=None, group=None, spread=False):
    """'fsbatch': consecutive multi-CTA full sums of one interval share one body (values unchanged).

    Each sum keeps its own chunk lanes, CTA tree (lane-0 warp values), partial write, fence, ticket
    and fixed-order combine by its last-arriving CTA. Only the CTA barriers and the serialization of
    one ticket round trip per sum are shared: up to 32 sums issue their tickets from distinct lanes.
    warp_combine (LATCH-SELECTIVE-10 ``fswarp``): the combines a CTA owns run on its 8 warps in parallel, each an
    exact one-warp replica of the block combine (``_warp_combine``), instead of one after another on the block.
    Returns (parts, records).
    """
    out, records, k = list(parts), [], 0
    head = re.compile(r"\A(// Phase \d+:[^\n]*\n\{\n)(.*)(\n\}(?:\nif \(ticks && threadIdx\.x == 0 && blockIdx\.x < 256u\) "
                      r"ticks\[[^\n]*\] = stamp\(\);)?)\Z", re.S)
    while k < len(parts):
        run = []
        while (k + len(run) < len(parts) and len(run) < FS_BATCH
               and phases[part_ids[k + len(run)]].get("engine") == "multi_cta_full"):
            match = head.match(parts[k + len(run)])
            pieces = _full_sum_parts(match[2]) if match else None
            if pieces is None:
                break
            run.append((match, pieces))
        if len(run) < 2:
            k += max(1, len(run))
            continue
        K = len(run)
        lines = ["{ // update_schedule fsbatch", "if (gridDim.x <= 1024) {",
                 f"  __shared__ unsigned latch_fsb_last_[{K}];", f"  float latch_fsb_acc_[{K}];"]
        if stamp is not None:   # LATCH-TRAIN-SYNC diagnostic: per-CTA time at the batch's start
            lines.append(stamp(f"fsbs:{part_ids[k]}"))
        mixed_ = set()
        if mix:
            # LATCH-SELECTIVE-10 fsmix: prelude-free eight-lane sums, FS_MIX_GROUP at a time, round-interleaved.
            # LATCH-TRAIN-SYNC fsmixl (leaves): sums whose prelude is only one-line leaf lambdas join the groups.
            plain_ = [t for t, (_, (prelude, calculation, _, _, _)) in enumerate(run)
                      if _LANES_CALC.match(calculation)
                      and (not prelude.strip() or (leaves and _leaf_prelude(prelude) is not None))]
            fused = fused or {}
            size_ = group or FS_MIX_GROUP
            for g in range(0, len(plain_), size_):
                group_ = plain_[g:g + size_]
                if len(group_) < 2 and not any(part_ids[k + t] in fused for t in group_):
                    break
                code_ = _mixed_calculations([(t, run[t][1][1], run[t][1][0] if run[t][1][0].strip() else "",
                                              fused.get(part_ids[k + t])) for t in group_], unconditional=unconditional)
                if code_ is None and any(part_ids[k + t] in fused for t in group_):
                    raise ValueError("cast_sums: a fused sum's batch group could not be generated")
                if code_ is None and unconditional:
                    code_ = _mixed_calculations([(t, run[t][1][1], run[t][1][0] if run[t][1][0].strip() else "")
                                                 for t in group_])
                if code_ is not None:
                    lines += ["{", "const long long o = 0;", code_, "}"]
                    mixed_.update(group_)
        for t, (_, (prelude, calculation, _, _, _)) in enumerate(run):
            if t in mixed_:
                continue
            lines += ["{", prelude + "const long long o = 0;", calculation + f"  latch_fsb_acc_[{t}] = acc;", "}"]
        if stamp is not None:   # LATCH-TRAIN-SYNC diagnostic: per-CTA time after the thread-lane partial sums
            lines.append(stamp(f"fsbl:{part_ids[k]}"))
        for t in range(K):
            lines.append(f"  {{ const float acc = s101h::warp_sum(latch_fsb_acc_[{t}]); "
                         f"if ((threadIdx.x & 31) == 0) smem[{8 * t} + (threadIdx.x >> 5)] = acc; }}")
        lines += ["  __syncthreads();", "  if (threadIdx.x < 32) {"]
        if fence:
            # LATCH-SELECTIVE-10 ``fsfence``: the same per-sum second-level butterflies; then lane t writes sum t's
            # partial, every lane fences its own write once, and lane t takes sum t's ticket -- one round instead of
            # K serialized (divergent) write/fence/atomic blocks. Each sum's partial still precedes its fence and
            # ticket, so the last-arriving CTA's combine sees every partial as before.
            lines.append("    float latch_fsb_mine_ = 0.0f;")
            for t in range(K):
                lines.append(f"    {{ float acc = threadIdx.x < 8 ? smem[{8 * t} + threadIdx.x] : 0.0f; "
                             f"acc = s101h::warp_sum(acc); acc = __shfl_sync(0xffffffffu, acc, 0); "
                             f"if (threadIdx.x == {t}) latch_fsb_mine_ = acc; }}")
            lines.append(f"    float* const latch_fsb_parts_[{K}] = {{" + ", ".join(
                f"({part})" for _, (_, _, part, _, _) in run) + "};")
            lines.append(f"    unsigned* const latch_fsb_counts_[{K}] = {{" + ", ".join(
                f"({cnt})" for _, (_, _, _, cnt, _) in run) + "};")
            lines += [f"    if (threadIdx.x < {K}) latch_fsb_parts_[threadIdx.x][blockIdx.x] = latch_fsb_mine_;",
                      "    __threadfence();",
                      f"    if (threadIdx.x < {K}) latch_fsb_last_[threadIdx.x] = "
                      "atomicAdd(latch_fsb_counts_[threadIdx.x], 1u) == gridDim.x - 1u;"]
        else:
            for t, (_, (_, _, part, cnt, _)) in enumerate(run):
                lines.append(f"    {{ float acc = threadIdx.x < 8 ? smem[{8 * t} + threadIdx.x] : 0.0f; "
                             f"acc = s101h::warp_sum(acc); acc = __shfl_sync(0xffffffffu, acc, 0);\n"
                             f"      if (threadIdx.x == {t}) {{ {part}[blockIdx.x] = acc; __threadfence(); "
                             f"latch_fsb_last_[{t}] = atomicAdd({cnt}, 1u) == gridDim.x - 1u; }} }}")
        lines += ["  }", "  __syncthreads();"]
        if stamp is not None:   # after the CTA trees, partial writes, fences and tickets
            lines.append(stamp(f"fsbt:{part_ids[k]}"))
        warps = [_warp_combine(combine) for _, (_, _, _, _, combine) in run] if warp_combine else None
        if warps and all(w is not None for w in warps) and spread:
            # LATCH-TRAIN-SYNC ``fsspread``: sum t is combined by one designated warp (t % 8) of CTA t % gridDim.x, which
            # waits until all gridDim.x tickets of the sum are in (every CTA writes its partial, fences and takes its
            # ticket before any CTA waits: no CTA waits for a waiting CTA), then runs the same exact one-warp combine
            # the last arriver would run (same partials, order and butterflies; it resets the ticket). The last
            # arriving CTA no longer runs every combine of the batch after its own partials.
            lines.append("  if (blockDim.x == 256u) {")
            for t, text in enumerate(warps):
                cnt = run[t][1][3]
                lines += [f"    if (blockIdx.x == {t}u % gridDim.x && (threadIdx.x >> 5) == {t % 8}u) {{",
                          f"      if ((threadIdx.x & 31u) == 0u) while (*reinterpret_cast<volatile unsigned*>({cnt}) "
                          "< gridDim.x) { }",
                          "      __syncwarp();", text, "    }"]
            lines += ["  } else {"]
            for t, (_, (_, _, _, _, combine)) in enumerate(run):
                lines += [f"  if (latch_fsb_last_[{t}]) {{", "    float acc;", combine, "  }", "  __syncthreads();"]
            lines += ["  }", "  __syncthreads();", "}", "}"]
            records_mode = "spread"
        elif warps and all(w is not None for w in warps):
            # Warp v takes the CTA's owned sums v, v + 8, ... (in batch order); combines touch no shared memory.
            lines.append("  if (blockDim.x == 256u) {")
            for t, text in enumerate(warps):
                lines += [f"    if (latch_fsb_last_[{t}] && (threadIdx.x >> 5) == "
                          f"(unsigned)__popc(0u{''.join(f' | (latch_fsb_last_[{q}] ? {1 << q}u : 0u)' for q in range(t))}) % 8u) {{",
                          text, "    }"]
            lines += ["  } else {"]
            for t, (_, (_, _, _, _, combine)) in enumerate(run):
                lines += [f"  if (latch_fsb_last_[{t}]) {{", "    float acc;", combine, "  }", "  __syncthreads();"]
            lines += ["  }", "  __syncthreads();", "}", "}"]
            records_mode = "warp"
        else:
            for t, (_, (_, _, _, _, combine)) in enumerate(run):
                # The original phases ended with a CTA barrier; consecutive combines reuse smem[0..7].
                lines += [f"  if (latch_fsb_last_[{t}]) {{", "    float acc;", combine, "  }", "  __syncthreads();"]
            lines += ["}", "}"]
            records_mode = "block"
        if stamp is not None:   # after the owned combines (inside the batch's outer blocks)
            lines.insert(len(lines) - 2, stamp(f"fsbc:{part_ids[k]}"))
        first = run[0][0]
        out[k] = first[1] + "\n".join(lines) + first[3]
        for j in range(1, K):
            out[k + j] = run[j][0][1] + "// joined the update_schedule fsbatch body above" + run[j][0][3]
        records.append(dict(phases=[part_ids[k + j] for j in range(K)], combine=records_mode, mixed=len(mixed_),
                            **({"fused": [part_ids[k + t] for t in sorted(mixed_) if part_ids[k + t] in (fused or {})]}
                               if fused else {})))
        k += K
    return out, records


QUEUE_MIN = 8          # shortest run of eligible phases that becomes one CTA work queue
QUEUE_ITEM = 1024      # flat-index units per virtual block (one 4-wide unrolled round of 256 threads)
QUEUE_THREADS = 256    # threads per CTA of the queued bodies


def _round(text, helpers):
    """Flat-index lanes of one grid-stride round of an eligible block (1 without an unrolled loop)."""
    rounds = [helpers[name]["round"] for name, _ in _CALL.findall(text) if name in helpers]
    rounds += [int(x) for x in _ROUND.findall(text)]
    return max(rounds, default=1)


def _queue(run, source, helpers):
    """One CTA work queue over (phase, virtual block) items for a run of eligible phase blocks.

    Item j belongs to phase k and virtual block c; CTA b takes items b, b + nb, ... Each item runs the
    phase's unchanged block with shadowed indices lsr_bx = c and lsr_nb = V_k (its virtual grid of V_k
    blocks). Every eligible body is a pure function of the flat index bx*nt+tx over a grid of nb*nt
    threads, so the V_k virtual blocks compute each element exactly once with the same expression.
    The phase of an item is found by a balanced comparison tree on constants (no device tables).
    """
    starts, total, cases = [], 0, []
    for k, (phase, head, start, end, extent) in enumerate(run):
        # LATCH-TINYMOE-EGRAD: an item spans at least one whole unrolled round of its body, so wider rounds
        # (more independent 16-byte loads in flight per thread) are not cut into masked partial rounds. For
        # rounds of <= 4 lanes this is QUEUE_ITEM, unchanged.
        item = max(QUEUE_ITEM, _round(source[start:end], helpers) * QUEUE_THREADS)
        blocks = max(1, -(-extent // item))
        starts.append(total)
        cases.append(f"case {k}: {{ const unsigned lsr_bx = latch_q_ - {total}u, lsr_nb = {blocks}u;\n"
                     f"{source[start:end]}\n}} break;")
        total += blocks

    def tree(lo, hi):
        if hi - lo == 1:
            return f"{lo}u"
        mid = (lo + hi) // 2
        return f"(latch_q_ < {starts[mid]}u ? {tree(lo, mid)} : {tree(mid, hi)})"
    body = (f"// update_schedule queue: phases {run[0][0]}..{run[-1][0]} ({len(run)} bodies, {total} items)\n{{\n"
            f"for (unsigned latch_q_ = lsr_bx; latch_q_ < {total}u; latch_q_ += lsr_nb) {{\n"
            f"  switch ({tree(0, len(run))}) {{\n" + "\n".join(cases) + "\n  default: break;\n  }\n}\n}")
    return body, dict(phases=[r[0] for r in run], items=total)


def rotate(program):
    """Queue runs of eligible elementwise bodies; rotate the remaining ones (values unchanged)."""
    from .lower import GeneratedProgram
    source, manifest = program.source, program.manifest
    if manifest.get("barrier_schedule") != "dependency" or "update_schedule" in manifest:
        raise Refusal("requires one dependency-scheduled flat program")
    phases = manifest["phases"]
    interval = {p: k for k, members in enumerate(manifest["barrier_intervals"]) for p in members}
    helpers = _helpers(source)
    heads = {m.end(): m.start() for m in _PHASE.finditer(source)}
    tables = set(manifest.get("prologue_batch", {}).get("heads", ()))
    blocks = []
    for phase, start, end in _blocks(source):
        extent = (_extent(source[start:end], helpers)
                  if phases[phase].get("kind") == "elementwise" and phase not in tables else None)
        blocks.append((phase, heads[start], start, end, extent))
    # Maximal runs of eligible blocks of one interval separated only by CTA barriers.
    runs, run_ = [], []
    for block in blocks:
        if block[4] is None:
            if run_:
                runs.append(run_)
            run_ = []
            continue
        if run_ and (interval[run_[-1][0]] != interval[block[0]]
                     or source[run_[-1][3] + 1:block[1]] not in ("\n__syncthreads();\n", "\n")):
            runs.append(run_)
            run_ = []
        run_.append(block)
    if run_:
        runs.append(run_)
    queued = [run for run in runs if len(run) >= QUEUE_MIN]
    in_queue = {block[0] for run in queued for block in run}
    edits, offsets, records, queues = [], {}, [], []
    for run in queued:
        body, record = _queue(run, source, helpers)
        queues.append(record)
        edits.append((run[0][1], run[-1][3] + 1, body))
    for phase, head, start, end, extent in blocks:
        if extent is None or phase in in_queue:
            continue
        k = interval[phase]
        offset = offsets.get(k, 0)
        offsets[k] = offset + -(-extent // 32) * 32  # whole warps per phase: tiny phases take distinct warps
        rotation = offset                       # a multiple of 32: lanes stay consecutive (coalesced)
        if rotation:
            edits.append((start, start,
                          "const unsigned latch_rg_ = lsr_nb * lsr_nt; const unsigned latch_rt_ = "
                          f"(lsr_bx * lsr_nt + lsr_tx + latch_rg_ - {rotation}u % latch_rg_) % latch_rg_;\n"
                          "const unsigned lsr_bx = latch_rt_ / lsr_nt, lsr_tx = latch_rt_ % lsr_nt;\n"))
            records.append(dict(phase=phase, interval=k, extent=extent, rotation=rotation))
    pieces, cursor = [], 0
    for start, end, text in sorted(edits):
        if start < cursor:
            raise Refusal("overlapping update-schedule edits")
        pieces += [source[cursor:start], text]
        cursor = end
    pieces.append(source[cursor:])
    transformed = "".join(pieces)
    result = dict(manifest)
    result["update_schedule"] = dict(
        version=VERSION, schedule=current(), rotated_phases=len(records),
        rotated_intervals=len({r["interval"] for r in records}),
        queued_runs=len(queues), queued_phases=sum(len(q["phases"]) for q in queues),
        eligible_helpers=len(helpers), source_before=manifest["source_hash"],
        rule=("runs of >= %d eligible bodies of one interval: CTA work queue over (body, virtual block) items; "
              "other eligible bodies: per-interval running grid-stride rotation" % QUEUE_MIN),
        queues=queues, records=records)
    result["source_hash"] = hashlib.sha256(
        (transformed + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(transformed, result)


def late_updates(phases, intervals, layouts, targets):
    """``ulate``: ASAP levels with a floor for bodies that no target (loss/gradient output) depends on.

    Bodies that some target transitively depends on keep their level (their dependencies are themselves needed);
    every other body is placed at max(its ASAP level over the new levels of its dependencies, last needed level
    + 1). Dependencies come from the same effect contracts as the ordinary dependency schedule, so every RAW/WAR/
    WAW and fence order is preserved and each body's arithmetic is unchanged. Returns (intervals, record)."""
    from .execution_plan import contracts
    items = contracts(phases, layouts)
    deps = [tuple(item.dependencies) for item in items]
    roots = {layouts[name].root if name in layouts else name for name in targets}
    needed = set()
    stack = [i for i, phase in enumerate(phases)
             if any((layouts[w].root if w in layouts else w) in roots for w in phase["writes"])]
    while stack:
        i = stack.pop()
        if i in needed:
            continue
        needed.add(i)
        stack.extend(deps[i])
    stage = {p: k for k, members in enumerate(intervals) for p in members}
    if not needed:
        return intervals, dict(moved=0, reason="no target producers")
    last = max(stage[i] for i in needed)
    level, moved = [], 0
    for i in range(len(phases)):
        if i in needed:      # needed bodies depend only on needed bodies: keep the ordinary placement
            base = stage[i]
        else:
            base = max(1 + max((level[d] for d in deps[i]), default=-1), last + 1, stage[i])
        moved += base != stage[i]
        level.append(base)
    out = [[] for _ in range(max(level) + 1)]
    for i, k in enumerate(level):
        out[k].append(i)
    out = [members for members in out if members]
    return out, dict(moved=moved, last_target_interval=last, needed=len(needed), bodies=len(phases),
                     intervals_before=len(intervals), intervals_after=len(out))
