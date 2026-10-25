"""Opt-in prologue batching rules for flat common compose programs (LATCH-TINYMOE-PROLOGUE).

Selected through the normal build API: ``CapturedModule.build(..., update_schedule="streamed_prep")`` is
"streamed" (ops.update_schedule) plus the rules below, applied to the generated program just before the
update-schedule work queue. Other schedules leave lowering and generated source unchanged.

A training update's first dependency interval carries many tiny independent phases: per-tensor optimizer
scalars (step counter increment, bias corrections, ``1 - lr * wd``) and the reset of every multi-CTA
reduction ticket. Each was its own phase: one code block per phase that every CTA walks, followed by a CTA
barrier, or one work-queue case per phase. Two structural rules replace a run of such phases by one phase
whose members are listed in a read-only table. Nothing depends on a model name or shape table, and every
member's arithmetic is unchanged:

* ``scalar table``: a run of >= ``MIN_RUN`` adjacent phases of one dependency interval, separated only by
  CTA barriers, each exactly one call of an eligible elementwise helper (``update_schedule._helpers``: a
  pure function of the flat index ``bx*nt+tx`` with no shared memory, barriers, shuffles or atomics) whose
  extent is at most ``MAX_EXTENT``, and whose other arguments are pointers declared from ``args`` or the
  arena. Grid thread ``t`` runs member ``t``'s unchanged helper on a one-thread virtual grid (bx = tx = 0,
  nt = nb = 1), which visits the member's flat indices 0..extent-1 itself. A helper computes each flat
  index from the same expression for any grid, so every member writes the same values as before.
  Members of different helpers form one loop per helper inside the same phase.
* ``counter table``: a run of >= ``MIN_RUN`` adjacent reduction-counter initializations (the exact
  ``if (lsr_bx == 0 && lsr_tx == 0) *counter = 0u;`` template) becomes one phase in which grid thread ``t``
  zeroes counter ``t``. Every counter is still reset in the same interval, before its first use.

Admission additionally checks, from the manifest's per-phase reads and writes and the declared pointer of
every argument, that no member writes storage another member of its run reads or writes (phases of one
dependency interval are independent by construction; this re-proves it for the members being regrouped).
Phases outside admitted runs are unchanged. A run that fails any check is left as it was and recorded.

Tables are block-scope ``static const`` arrays: initialized read-only device data private to the unit that
holds the phase (``step17_intervals.unit_state`` admits them by their ``latch_ptab_`` name). Each entry is a
pointer code: ``2*k+1`` for ``args[k]`` and ``2*f`` for arena float offset ``f``.
"""
from __future__ import annotations

import hashlib
import json
import re

VERSION = 1
MIN_RUN = 8            # shortest run of phases that becomes one table phase
MAX_EXTENT = 64        # largest helper extent (flat indices) a member may serialize on one thread
TABLE_PREFIX = "latch_ptab_"


class Refusal(ValueError):
    """The program does not satisfy the narrow source contract."""


_MARKERS = ("latch_mixed_phase_pointers", "latch_resource_phase_pointers")
_BASE_LINES = {
    "float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);": ("pa", "pr"),
    "float* latch_pr = latch_opq(arena);": ("pr",),
    "float** latch_pa = latch_opq(args);": ("pa",),
}
_LOCATION = r"(?:latch_pa\[(?P<{0}a>\d+)\]|latch_pr \+ (?P<{0}r>\d+)LL)"
_DECL = re.compile(r"(?P<type>[A-Za-z_][\w:]*\*) (?P<name>v_\w+) = (?:reinterpret_cast<(?P<cast>[A-Za-z_][\w:]*\*)>\("
                   + _LOCATION.format("c") + r"\)|" + _LOCATION.format("p") + r");")
_CALL = re.compile(r"(latch_ec_body\d+)\(([^;()]*)\);")
_INDEX = ("lsr_bx", "lsr_nt", "lsr_tx", "lsr_nb")
_COUNTER = re.compile(r"if \(lsr_bx == 0 && lsr_tx == 0\) \*reinterpret_cast<unsigned\*>\((v_\w+)\) = 0u;")
_SEPARATORS = ("\n__syncthreads();\n", "\n")


def _parse_body(text):
    """(declarations {name: (type, code)}, statement) of a phase body in pointer-prefix form, or None."""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 4:
        return None
    head = re.fullmatch(r"// (\w+): (\d+)", lines[0])
    if head is None or head[1] not in _MARKERS or lines[1] not in _BASE_LINES:
        return None
    try:
        end = lines.index(f"// {head[1]}: end")
    except ValueError:
        return None
    declarations = {}
    for line in lines[2:end]:
        match = _DECL.fullmatch(line)
        if match is None or match["name"] in declarations:
            return None
        if match["cast"] is not None and match["cast"] != match["type"]:
            return None
        arg, offset = match["ca"] or match["pa"], match["cr"] or match["pr"]
        if arg is not None and "pa" not in _BASE_LINES[lines[1]] or offset is not None and "pr" not in _BASE_LINES[lines[1]]:
            return None
        code = 2 * int(arg) + 1 if arg is not None else 2 * int(offset)
        declarations[match["name"]] = (match["type"], code)
    statements = lines[end + 1:]
    if len(statements) != 1 or not declarations:
        return None
    return declarations, statements[0]


def _helper_roles(source, helpers):
    """{helper: (extent, {parameter position: index register})} for eligible helpers."""
    from .update_schedule import _HELPER, _HELPER_START
    roles = {}
    for match in _HELPER.finditer(source):
        name = match[1]
        if name not in helpers:
            continue
        params = [p.split()[-1] for p in match[2].split(",")]
        b, n, t = _HELPER_START.findall(match[3])[0]
        index = [0, 1, 2, 3]
        if sorted(params[:4]) != sorted(f"latch_ec_p{i}" for i in range(4)) or len({b, n, t}) != 3:
            continue
        nb = [p for p in params[:4] if p not in (b, n, t)]
        if len(nb) != 1 or not re.search(r"\b" + nb[0] + r"\b", match[3]):
            continue
        role = {params.index(b): "lsr_bx", params.index(n): "lsr_nt", params.index(t): "lsr_tx",
                params.index(nb[0]): "lsr_nb"}
        if sorted(role) != index:
            continue
        roles[name] = (helpers[name]["extent"], role)
    return roles


def _scalar_member(text, roles):
    parsed = _parse_body(text)
    if parsed is None:
        return None
    declarations, statement = parsed
    call = _CALL.fullmatch(statement)
    if call is None or call[1] not in roles:
        return None
    extent, role = roles[call[1]]
    args = [a.strip() for a in call[2].split(",")]
    if extent > MAX_EXTENT or len(args) < 5 or any(args[i] != r for i, r in role.items()):
        return None
    pointers = args[4:]
    if any(a not in declarations for a in pointers) or set(pointers) != set(declarations):
        return None
    return dict(helper=call[1], types=[declarations[a][0] for a in pointers],
                codes=[declarations[a][1] for a in pointers], names=pointers)


def _counter_member(text):
    parsed = _parse_body(text)
    if parsed is None:
        return None
    declarations, statement = parsed
    match = _COUNTER.fullmatch(statement)
    if match is None or set(declarations) != {match[1]} or declarations[match[1]][0] != "float*":
        return None
    code = declarations[match[1]][1]
    if code & 1:
        return None                       # counters live in the arena
    return dict(codes=[code], names=[match[1]])


def _roots(manifest):
    sources = {v["name"]: v["source"] for v in manifest.get("views", [])}

    def root(name):
        seen = set()
        while name in sources and name not in seen:
            seen.add(name)
            name = sources[name]
        return name
    return root


def _independent(members, phases, root):
    """No member writes storage that another member reads or writes (manifest names and declared pointers)."""
    touched_by, written_by = {}, {}
    for k, member in enumerate(members):
        phase = phases[member["phase"]]
        written = {root(n) for n in phase.get("writes", [])}
        keys = {("value", root(n)) for n in phase.get("reads", []) + phase.get("writes", [])}
        keys |= {("pointer", code) for code in member["codes"]}
        writes = {("value", n) for n in written}
        writes |= {("pointer", code) for name, code in zip(member["names"], member["codes"])
                   if root(_manifest_name(name)) in written}
        for key in keys:
            touched_by.setdefault(key, set()).add(k)
        for key in writes:
            written_by.setdefault(key, set()).add(k)
    return all(touched_by.get(key, set()) <= writers and len(writers) == 1 for key, writers in written_by.items())


def _manifest_name(identifier):
    return identifier[2:] if identifier.startswith("v_") else identifier


def _decode(code_expression, ctype):
    return (f"reinterpret_cast<{ctype}>(({code_expression}) & 1 ? reinterpret_cast<char*>(latch_pa[({code_expression}) >> 1]) "
            f": reinterpret_cast<char*>(latch_pr + (({code_expression}) >> 1)))")


def _constant(code, ctype):
    location = f"latch_pa[{code >> 1}]" if code & 1 else f"latch_pr + {code >> 1}LL"
    return f"reinterpret_cast<{ctype}>({location})"


def _table(name, rows):
    values = ", ".join(f"{v}LL" for row in rows for v in row)
    return f"static const long long {name}[{sum(len(r) for r in rows)}] = {{{values}}};"


def _loop(count, rotation, body):
    """Grid-stride loop over table rows; consecutive loops of one phase start on different threads."""
    start = "lsr_bx * lsr_nt + lsr_tx"
    if rotation:
        start = f"(lsr_bx * lsr_nt + lsr_tx + latch_pg_ - {rotation}u % latch_pg_) % latch_pg_"
    return (f"for (unsigned latch_pt_ = {start}; latch_pt_ < {count}u; latch_pt_ += latch_pg_) {{\n"
            f"{body}\n}}")


def _scalar_phase(run, roles, label):
    """One phase body for a run of scalar-helper members (one loop per helper, in first-appearance order)."""
    groups = {}
    for member in run:
        groups.setdefault(member["helper"], []).append(member)
    lines = ["float** latch_pa = latch_opq(args); float* latch_pr = latch_opq(arena);",
             "const unsigned latch_pg_ = lsr_nb * lsr_nt;"]
    rotation, records = 0, []
    for g, (helper, members) in enumerate(groups.items()):
        types = members[0]["types"]
        if any(m["types"] != types for m in members):
            raise Refusal("pointer types differ between calls of one helper")
        _, role = roles[helper]
        # Columns: constant across members -> literal; otherwise one table column (shared by equal columns).
        columns, table_columns, arguments = [], [], []
        for position in range(len(types)):
            column = tuple(m["codes"][position] for m in members)
            columns.append(column)
        for position, column in enumerate(columns):
            if len(set(column)) == 1:
                arguments.append(_constant(column[0], types[position]))
                continue
            if column not in table_columns:
                table_columns.append(column)
            arguments.append(_decode(f"latch_pe_[{table_columns.index(column)}]", types[position]))
        width = len(table_columns)
        index_values = {"lsr_bx": "0u", "lsr_tx": "0u", "lsr_nt": "1u", "lsr_nb": "1u"}
        call_args = [index_values[role[i]] for i in range(4)] + arguments
        name = f"{TABLE_PREFIX}{label}_{g}"
        body = []
        if width:
            body.append(_table(name, [[c[k] for c in table_columns] for k in range(len(members))]))
            body.append(f"const long long* latch_pe_ = {name} + {width}u * latch_pt_;")
        body.append(f"{helper}({', '.join(call_args)});")
        lines.append("{\n" + _loop(len(members), rotation, "\n".join(body)) + "\n}")
        rotation += -(-len(members) // 32) * 32
        records.append(dict(helper=helper, members=len(members), table_columns=width))
    return "\n".join(lines), records


def _counter_phase(run, label):
    name = f"{TABLE_PREFIX}{label}_0"
    body = [_table(name, [[m["codes"][0] >> 1] for m in run]),
            f"*reinterpret_cast<unsigned*>(latch_pr + {name}[latch_pt_]) = 0u;"]
    lines = ["float* latch_pr = latch_opq(arena);", "const unsigned latch_pg_ = lsr_nb * lsr_nt;",
             "{\n" + _loop(len(run), 0, "\n".join(body)) + "\n}"]
    return "\n".join(lines)


def batch(program):
    """Apply both table rules to an admitted program; returns a new GeneratedProgram (or the same one)."""
    from .lower import GeneratedProgram
    from .update_schedule import _PHASE, _blocks, _helpers
    source, manifest = program.source, program.manifest
    if manifest.get("barrier_schedule") != "dependency" or "prologue_batch" in manifest or "update_schedule" in manifest:
        raise Refusal("requires one dependency-scheduled flat program before the update-schedule queue")
    phases = manifest["phases"]
    interval = {p: k for k, members in enumerate(manifest["barrier_intervals"]) for p in members}
    roles = _helper_roles(source, _helpers(source))
    root = _roots(manifest)
    heads = {m.end(): m.start() for m in _PHASE.finditer(source)}
    blocks = []
    for phase, start, end in _blocks(source):
        text = source[start:end]
        kind = phases[phase].get("kind")
        member = None
        if kind == "elementwise":
            member = _scalar_member(text, roles)
            family = "scalar"
        elif kind == "reduction_counter_init":
            member = _counter_member(text)
            family = "counter"
        if member is not None:
            member = dict(member, phase=phase, family=family)
        blocks.append((phase, heads[start], start, end, member))
    runs, current = [], []
    for block in blocks:
        member = block[4]
        if member is None:
            if current:
                runs.append(current)
            current = []
            continue
        if current and (current[-1][4]["family"] != member["family"]
                        or interval[current[-1][0]] != interval[block[0]]
                        or source[current[-1][3] + 1:block[1]] not in _SEPARATORS):
            runs.append(current)
            current = []
        current.append(block)
    if current:
        runs.append(current)
    edits, records, refused = [], [], []
    for run in runs:
        if len(run) < MIN_RUN:
            continue
        members = [b[4] for b in run]
        family, first = members[0]["family"], run[0][0]
        ids = [b[0] for b in run]
        if not _independent(members, phases, root):
            refused.append(dict(phases=[ids[0], ids[-1]], reason="members not independent"))
            continue
        if family == "counter" and len({m["codes"][0] for m in members}) != len(members):
            refused.append(dict(phases=[ids[0], ids[-1]], reason="repeated counter"))
            continue
        label = str(first)
        try:
            if family == "scalar":
                body, groups = _scalar_phase(members, roles, label)
            else:
                body, groups = _counter_phase(members, label), []
        except Refusal as error:
            refused.append(dict(phases=[ids[0], ids[-1]], reason=str(error)))
            continue
        text = (f"// Phase {first}: prologue_batch {family} table of {len(run)} phases ({ids[0]}..{ids[-1]})\n"
                f"{{\n{body}\n}}")
        edits.append((run[0][1], run[-1][3] + 1, text))
        records.append(dict(rule=family + "_table", interval=interval[first], head=first, phases=ids,
                            groups=groups))
    if not edits:
        return program
    pieces, cursor = [], 0
    for start, end, text in edits:
        pieces += [source[cursor:start], text]
        cursor = end
    pieces.append(source[cursor:])
    transformed = "".join(pieces)
    result = dict(manifest)
    result["prologue_batch"] = dict(
        version=VERSION, min_run=MIN_RUN, max_extent=MAX_EXTENT, source_before=manifest["source_hash"],
        heads=[r["head"] for r in records], batched_phases=sum(len(r["phases"]) for r in records),
        rule=("runs of >= %d adjacent independent scalar-helper calls (extent <= %d) or reduction-counter "
              "initializations of one interval: one phase over a read-only member table" % (MIN_RUN, MAX_EXTENT)),
        runs=records, refused=refused)
    result["source_hash"] = hashlib.sha256(
        (transformed + json.dumps(result["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(transformed, result)
