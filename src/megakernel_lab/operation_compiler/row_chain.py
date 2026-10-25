"""Row-block chains of small row-local dependency intervals (LATCH-SELECTIVE-10, ops.accumulation_rules ``rowchain``).

A run of consecutive dependency intervals whose phases are all row-local over one leading dimension R (a token
dimension: every result row r is computed only from row r of the run's earlier results, or from values that existed
before the run) needs no grid barrier between its intervals if every CTA computes the same rows in every phase:
CTA b owns rows [b*B, min(R, (b+1)*B)), B = ceil(R / gridDim.x), and runs the run's phases one after another on its
rows with CTA barriers in between; one grid barrier closes the run. A row's values are computed by the same
expression on the same operands as before -- only which thread computes them changes -- so results are bitwise
identical (the update_schedule queue argument, applied to row blocks).

Admitted phase forms (emitter-specific and checked textually; anything else ends a run):

* ``elementwise`` with per-element body lines over the output index ``i`` (output [R, ...]; every run-produced
  operand has rank and leading extent equal to the output's, so broadcasting aligns rows) -- re-emitted as a scalar
  loop over the CTA's element range.
* flat element loops (``gather``, ``cat``) ``for (long long i = blockIdx.x*blockDim.x + threadIdx.x; i < N; ...)``
  whose run-produced operands are read only at ``((i / D) % R)`` row indices with D = N / R.
* the narrow exact warp-per-row softmax (``exact_cta_tree_per_warp_narrow``): rows walk the CTA's block, one warp
  per row (lanes are physical lanes, as before).
* ``topk_rounds_lane_row`` (one thread per row, no inter-thread communication).
"""
from __future__ import annotations

import math
import re

MAX_ELEMENTS = 1 << 18
MIN_RUN = 2
# TINYMOE-FINER-TRANSFER: cut runs at arena storage reuse (arena_safe_runs). False restores the unsafe historical
# planning only for reproducing old builds; it can race.
ARENA_SAFE = True
_FLAT_HEAD = re.compile(r"for \(long long i = \(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x; i < (\d+)LL; "
                        r"i \+= \(long long\)gridDim\.x \* blockDim\.x\) \{")
_WARP_ROW_HEAD = re.compile(r"for \(long long row = \(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5; "
                            r"row < (\d+)LL; row \+= \(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \{")
_LANE_ROW_HEAD = re.compile(r"for \(long long g = \(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5; "
                            r"g < (\d+)LL; g \+= \(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \{\n"
                            r"  const int lane = threadIdx\.x & 31;\n  long long c0_ = g;\n  c0_ = c0_ \* 32 \+ lane;\n"
                            r"  const long long row = c0_ \* 1LL;\n")
_FORBIDDEN = re.compile(r"\b(?:__shared__|smem|__syncthreads\w*|atomic\w*|grid\.|cp\.async|mbarrier\w*|wgmma\w*|"
                        r"__threadfence\w*|ticks|latch_ctr|asm)\b")


def _numel(shape):
    return math.prod(int(d) for d in shape)


def _rows_of(phase, layouts):
    """Leading extent R shared by every result of the phase (and its elements per row), or None."""
    shapes = [tuple(layouts[w].shape) for w in phase["writes"] if w in layouts]
    if not shapes or any(len(s) == 0 for s in shapes):
        return None
    R = int(shapes[0][0])
    if any(int(s[0]) != R or _numel(s) != _numel(shapes[0]) for s in shapes):
        return None
    return R, _numel(shapes[0]) // R if R else None


def _form(phase, code):
    """(form, extent) of an admitted phase body, or None."""
    kind, engine = phase.get("kind"), str(phase.get("engine") or "")
    if phase.get("requires_full_grid") or phase.get("counter_effects") or phase.get("internal_mutations"):
        return None
    if kind == "elementwise":
        body = phase.get("body") or ()
        if not body or len(body) != len(phase.get("names", ())) or phase.get("vprod_inline"):
            return None
        text = "\n".join(body)
        if _FORBIDDEN.search(text) or re.search(r"\b(?:threadIdx|blockIdx|gridDim|blockDim)\b", text):
            return None
        return "elementwise", None
    if _FORBIDDEN.search(code):
        return None
    if kind == "softmax" and engine == "exact_cta_tree_per_warp_narrow":
        m = _WARP_ROW_HEAD.search(code)
        if m and code.count("blockIdx.x") == 1 and code.count("gridDim.x") == 1:
            return "warp_row", int(m[1])
        return None
    if "__shfl" in code:
        return None
    if kind == "reduction" and engine == "topk_rounds_lane_row":
        m = _LANE_ROW_HEAD.search(code)
        if m and code.count("blockIdx.x") == 1 and code.count("gridDim.x") == 1 and code.count("threadIdx.x") == 2:
            return "lane_row", int(m[1])
        return None
    if kind in ("gather", "cat"):
        m = _FLAT_HEAD.search(code)
        if m and code.count("blockIdx.x") == 1 and code.count("gridDim.x") == 1 and code.count("threadIdx.x") == 1:
            return "flat", int(m[1])
    return None


def _ident(name):
    return "v_" + name


def _local_reads(phase, form, code, R, per_row, produced, layouts):
    """Every read of a run-produced value is row-local for this phase form."""
    out_rank = len(layouts[phase["writes"][0]].shape)
    for r in phase["reads"]:
        if r not in produced:
            continue
        lay = layouts.get(r)
        if lay is None or not lay.shape or int(lay.shape[0]) != R:
            return False
        if form == "elementwise":
            if len(lay.shape) != out_rank:
                return False
        elif form == "flat":
            # Every access of r must index the row ((i / D) % R) of the output element (or element i of an [R]
            # operand when the output has one element per row).
            accesses = re.findall(re.escape(_ident(r)) + r"\[(.{0,40})", code)
            plain_ok = per_row == 1 and _numel(lay.shape) == R
            if not accesses or any(not (a.startswith(f"((i / {per_row}LL) % {R}LL)")
                                        or plain_ok and (a.startswith("(i)]") or a.startswith("i]")))
                                   for a in accesses):
                return False
        else:   # warp_row / lane_row bodies read their operand rows by construction ([R, W] rows)
            if len(lay.shape) != 2:
                return False
    return True


def plan_runs(intervals, phases, layouts, phase_code):
    """{first interval: [interval indices]}, skipped intervals, records."""
    info = []
    for members in intervals:
        entry = None
        forms = []
        for i in members:
            phase = phases[i]
            rows = _rows_of(phase, layouts)
            code = phase_code(i, 0)
            form = _form(phase, code)
            if form is not None and form[0] == "lane_row":
                # Lane-row top-k results are stored per round at [k * R + row] (rows in the last extent).
                R_ = form[1] * 32
                stores = re.findall(r"v_\w+\[(\d+)LL \+ row\] = ", code)
                if stores and all(int(o) % R_ == 0 for o in stores) and all(
                        _numel(layouts[w].shape) % R_ == 0 for w in phase["writes"]):
                    rows = (R_, 1)
                else:
                    rows = None
            if rows is None or form is None or max(_numel(layouts[w].shape) for w in phase["writes"]) > MAX_ELEMENTS:
                forms = None
                break
            R, per_row = rows
            if form[0] == "warp_row" and (form[1] != R or per_row >= 32):
                forms = None
                break
            if form[0] == "lane_row" and form[1] * 32 != R:
                forms = None
                break
            if form[0] == "flat" and form[1] != R * per_row:
                forms = None
                break
            forms.append((i, R, per_row, form, code))
        if forms and len({R for _, R, _, _, _ in forms}) == 1:
            entry = forms
        info.append(entry)
    runs, skip, records = {}, set(), []
    k = 0
    while k < len(intervals):
        if info[k] is None:
            k += 1
            continue
        R = info[k][0][1]
        run, produced = [], set()
        j = k
        while j < len(intervals) and info[j] is not None and info[j][0][1] == R:
            ok = all(_local_reads(phases[i], form[0], code, R, per_row, produced, layouts)
                     for i, _, per_row, form, code in info[j])
            if not ok:
                break
            run.append(j)
            for i, *_ in info[j]:
                produced.update(phases[i]["writes"])
            j += 1
        if len(run) >= MIN_RUN:
            runs[run[0]] = run
            skip.update(run[1:])
            records.append(dict(intervals=len(run), rows=R,
                                phases=[phases[i]["names"][0] for kk in run for i, *_ in info[kk]]))
            k = j
        else:
            k += 1
    return runs, skip, records, {kk: info[kk] for r in runs.values() for kk in r}


def arena_safe_runs(runs, intervals, phases, layouts, offsets, sizes):
    """Split runs at every interval that writes arena storage an earlier interval of the run accessed.

    TINYMOE-FINER-TRANSFER fix. Arena lifetimes are planned in interval units BEFORE runs are merged: storage whose
    last reader is interval j may be reused by a value produced in interval j+1, relying on the grid barrier between
    them. Inside a run that barrier is gone, and row locality does not protect storage reuse between DIFFERENT roots
    (their rows map to different addresses: a CTA writing its rows of the new root can overwrite rows of the old root
    that another CTA has not read yet). Observed on TinyMoE B8 (8,192-row top-k chains): run-to-run nondeterministic
    losses and gradients. A run is therefore cut before any interval that writes a root overlapping the storage of a
    different root read or written by an earlier interval of the same run. Returns ({first: run}, records)."""
    def roots(k, written_only):
        found = {}
        for i in intervals[k]:
            names = phases[i]["writes"] if written_only else list(phases[i]["reads"]) + list(phases[i]["writes"])
            for name in names:
                lay = layouts.get(name)
                root = lay.root if lay is not None else None
                if root in offsets and root in sizes:
                    found[root] = (int(offsets[root]), int(offsets[root]) + int(sizes[root]))
        return found

    safe, records, cuts = {}, [], 0
    for run in runs.values():
        pieces, current, seen = [], [], {}
        for k in run:
            clash = any(r not in seen and any(lo < hi2 and lo2 < hi for r2, (lo2, hi2) in seen.items() if r2 != r)
                        for r, (lo, hi) in roots(k, True).items())
            if clash and current:
                pieces.append(current)
                current, seen = [], {}
                cuts += 1
            current.append(k)
            seen.update(roots(k, False))
        pieces.append(current)
        for piece in pieces:
            if len(piece) >= MIN_RUN:
                safe[piece[0]] = piece
    for first, piece in sorted(safe.items()):
        records.append(dict(intervals=len(piece), first_interval=first,
                            phases=[phases[i]["names"][0] for kk in piece for i in intervals[kk]]))
    return safe, records, cuts


def restrict(phase, form, code, per_row):
    """The phase's code restricted to the CTA's row block [latch_rc_r0, latch_rc_r1)."""
    name = form[0]
    if name == "elementwise":
        return (f"for (long long i = latch_rc_r0 * {per_row}LL + threadIdx.x; i < latch_rc_r1 * {per_row}LL; "
                f"i += blockDim.x) {{\n" + "\n".join(phase["body"]) + "\n}")
    if name == "flat":
        return _FLAT_HEAD.sub(lambda m: (f"for (long long i = latch_rc_r0 * {per_row}LL + threadIdx.x; "
                                         f"i < latch_rc_r1 * {per_row}LL; i += blockDim.x) {{"), code, count=1)
    if name == "warp_row":
        return _WARP_ROW_HEAD.sub(lambda m: ("for (long long row = latch_rc_r0 + (threadIdx.x >> 5); row < latch_rc_r1; "
                                             "row += blockDim.x >> 5) {"), code, count=1)
    if name == "lane_row":
        return _LANE_ROW_HEAD.sub(lambda m: ("for (long long latch_rc_q = latch_rc_r0 + threadIdx.x; latch_rc_q < latch_rc_r1; "
                                             "latch_rc_q += blockDim.x) {\n  const long long g = latch_rc_q >> 5;\n"
                                             "  const int lane = (int)(latch_rc_q & 31);\n  long long c0_ = g;\n"
                                             "  c0_ = c0_ * 32 + lane;\n  const long long row = c0_ * 1LL;\n"), code, count=1)
    raise ValueError(name)


def open_block(R):
    return ("{ // row-block chain (ops.accumulation_rules rowchain)\n"
            f"const long long latch_rc_b = ({R}LL + gridDim.x - 1) / gridDim.x;\n"
            "const long long latch_rc_r0 = (long long)blockIdx.x * latch_rc_b < " f"{R}LL ? (long long)blockIdx.x * latch_rc_b : {R}LL;\n"
            f"const long long latch_rc_r1 = latch_rc_r0 + latch_rc_b < {R}LL ? latch_rc_r0 + latch_rc_b : {R}LL;\n")
