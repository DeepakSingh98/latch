"""Per-row-block readiness across dependency-interval boundaries (``tile_readiness``, opt-in).

With ``tile_readiness="attention"`` (default None: every program is unchanged) the grid barrier after interval k is
removed when EVERY read that crosses the boundary k -> k+1 is a supported row-local producer -> consumer edge in a
common row space [R, ...]:

Producers: width-32 segmented attention phases (forward O; backward dK and dV). Each finished item publishes its
element count to the 128-row blocks of the results it covers (``A.tr_cnt``/``A.tr_rb`` of the attention arguments,
kernels/attention_segmented.cuh). An item publishes only after its final stores.

Consumers (wait until EVERY required producer block counter is complete):
* an elementwise phase of more than 2**18 elements whose results have leading extent R and whose reads of producer
  results are row-local (same leading extent and rank): re-emitted as a loop over row blocks, assigned by static
  round robin over the grid in ascending order (its own emitted grid-stride loop restricted to the block,
  ``restrict_flat``); thread 0 waits for the block's counters, then a CTA barrier admits the block's threads.
* a claimed segmented right product (also the scatter-epilogue form) whose A operand is a producer's published
  [R, K] root: before issuing a task's first operand stage, the issuing thread waits for the task's rows, then fences
  generic -> async proxy so the TMA loads see other CTAs' generic stores (``latch_tr::ReadyT<false, true>``).

Rules: every read crossing the boundary is such an edge; other phases of interval k+1 read nothing produced in k and
run first; phases of k+1 write no root read or written in k (no WAR/WAW across the open boundary) and no in-place
update of k's inputs; no grid-barrier protocol inside either interval's phases. Consecutive boundaries may open
(k and k+1 both open). Arena lifetimes are planned in barrier epochs (intervals joined by open boundaries form one
epoch), so no storage is reused between intervals that overlap in time. The boundary becomes a CTA barrier
(``latch_tr::elided_barrier``). Deadlock freedom: a CTA waits only for items of an EARLIER interval; all of those are
taken before any CTA leaves that interval's item loops, and their holders wait only for even earlier intervals, so
the earliest pending work always progresses.

Counters live in the per-launch 'ectaq' words past the arena (zeroed at kernel entry, ordered by the first grid
barrier). Every consumer computes the same values as behind the barrier.

Observation compatibility: a retained int64 state clone of a forward attention phase's native work-counter table
cannot consume per-row readiness. It may be copied one interval later, after the unchanged consumer interval's
closing grid barrier, only when the source cannot be rewritten and the state destination has no phase readers or
other writers. The clone body, returned state and work-counter events remain unchanged; no training computation
moves.
"""
from __future__ import annotations

import math
import re

VERSION = "tile-ready-v2"
_FORBIDDEN = re.compile(r"\b(?:__shared__|smem|__syncthreads\w*|atomic\w*|grid\.|cp\.async|mbarrier\w*|wgmma\w*|"
                        r"__threadfence\w*|ticks|latch_ctr|asm|threadIdx|blockIdx|gridDim|blockDim)\b")
LAST_CENSUS = []
# Rows per published block of attention results.
ROWS_PER_BLOCK = 128
# Phases at or below row_chain.MAX_ELEMENTS can be merged by the row-chain pass (its own row-local schedule);
# row-block loops are for larger row-structured phases only.
MIN_LOOP_ELEMENTS = (1 << 18) + 1
# Resident CTAs the row-block size of a loop is chosen for (at least two blocks per CTA).
_LOOP_CTAS = 132


def mode():
    """2 under ``tile_readiness`` 'attention', else 0 (off)."""
    from .tile_readiness import active
    return 2 if active("attention") else 0


def enabled():
    return mode() > 0


def _numel(shape):
    return math.prod(int(d) for d in shape)


_START = "(long long)blockIdx.x * blockDim.x + threadIdx.x"
_STRIDE = "(long long)gridDim.x * blockDim.x"
_BOUND = re.compile(r"\b(jb_|jn_|j\d+|i) < (\d+)LL\b")


def restrict_flat(code, numel, per_row, rb):
    """The phase's own emitted grid-stride loop (scalar, vector_mixed or its pipelined form) restricted to the
    elements of rows [latch_rc_r0, latch_rc_r1): start at the block's first index, stride by the CTA's threads, stop
    at the block's end. Every element keeps its emitted expression and its G-element vector grouping; None when the
    code is not exactly one such loop over the phase's [rows, per_row] index space (G | per_row)."""
    if not code or code.count(_START) != 1 or "gridDim" not in code:
        return None
    bounds = {int(m.group(2)) for m in _BOUND.finditer(code)}
    if len(bounds) != 1:
        return None
    n = bounds.pop()
    if n <= 0 or numel % n:
        return None
    g = numel // n
    if g not in (1, 2, 4, 8) or per_row % g or (per_row * rb) % g:
        return None
    out = code.replace(_START, "(tr_v0_ + (long long)threadIdx.x)").replace(_STRIDE, "(long long)blockDim.x")
    out = _BOUND.sub(lambda m: f"{m.group(1)} < tr_v1_", out)
    if re.search(r"\b(?:gridDim|blockIdx)\b", out) or re.search(rf"\b{n}LL\b", out):
        return None
    return (f"{{ const long long tr_v0_ = latch_rc_r0 * {per_row // g}LL, tr_v1_ = latch_rc_r1 * {per_row // g}LL;\n"
            + out + "\n}")


def _barrier_text(phase):
    """Phases whose code contains a grid-wide barrier protocol keep both neighbouring barriers."""
    return bool(re.search(r"grid\.sync|grid_sync|plong_mcnt|internal_grid", phase.get("code") or ""))


def attention_outputs(phase):
    """Width-32 segmented attention row results it can PUBLISH per item (forward O; dK and dV), else None."""
    kind, code = phase.get("kind"), phase.get("code") or ""
    if kind == "segmented_attention_forward" and "run_forward<32," in code:
        return list(phase["writes"][:1])
    if kind == "segmented_attention_kv" and "run_kv<32," in code:
        return list(phase["writes"][:2])
    return None


def product_waits_ok(phase):
    """A claimed right product (plain, paired or scatter-epilogue) whose A operand rows can be WAITED for.

    K-concatenated hosts are excluded: their bodies take no readiness argument."""
    engine = str(phase.get("engine") or "")
    return (phase.get("kind") == "gemm" and phase.get("flat_claim") and not phase.get("tma_outer")
            and "outer" not in engine and "gather" not in engine and "kcat" not in engine
            and not phase.get("tile_consumers") and not phase.get("register_consumers")
            and bool(phase.get("reads")))


def _rows(phase, layouts):
    shapes = [tuple(layouts[w].shape) for w in phase["writes"] if w in layouts]
    if not shapes or len(shapes) != len(phase["writes"]) or any(len(s) == 0 for s in shapes):
        return None
    R = int(shapes[0][0])
    if R == 0 or any(int(s[0]) != R or _numel(s) != _numel(shapes[0]) for s in shapes):
        return None
    return R, _numel(shapes[0]) // R


def loop_form(phase, layouts):
    """(R, per_row) for an elementwise phase that can run as a row-block loop, else None."""
    if phase.get("kind") != "elementwise" or phase.get("counter_effects"):
        return None
    if phase.get("internal_mutations") or phase.get("vprod_inline") or phase.get("active_values"):
        return None
    body = phase.get("body") or ()
    if not body or len(body) != len(phase.get("names", ())):
        return None
    if _FORBIDDEN.search("\n".join(body)):
        return None
    rows = _rows(phase, layouts)
    if rows is None or rows[0] * rows[1] < MIN_LOOP_ELEMENTS:
        return None
    return rows


class Plan:
    """The chosen open boundaries and the per-phase rewrites the interval emitter applies."""

    def __init__(self, boundaries, phases, rb, loops, products):
        self.boundaries = boundaries
        self.phases = phases
        self.rb = rb
        self.loops = loops            # elementwise consumer phase -> {"rows", "per_row", "waits": [(p, cols)]}
        self.products = products      # attention producer -> {"rows", "cols", "publish": True};
        #                               waiting product -> {"rows", "cols", "wait": (p, K)}
        self.consumer = {q: True for q in loops}          # every loop is re-emitted (it waits for producer rows)
        self.waiting_products = {q for q, v in products.items() if v.get("wait")}
        self.open_after = {frozenset(b["members"]) for b in boundaries}
        self.counter = {}             # producer phase -> counter pointer expression (allocated when it is emitted)
        self.emitted, self.opened, self.forms, self.bound = [], [], {}, []
        # Rows per block, per phase: attention producers publish rb-row blocks; row-block loops take blocks small
        # enough for >= 2 blocks per resident CTA (a 16,384-row phase at 128 rows has only 128 blocks for 132 CTAs).
        self.rb_of = {i: rb for i in products}
        for i, v in loops.items():
            target = max(1, v["rows"] // (2 * _LOOP_CTAS))
            self.rb_of[i] = max(16, min(rb, 1 << (target.bit_length() - 1)))

    def epoch_of(self, intervals):
        open_k = {b["interval"] for b in self.boundaries}
        epoch, out = 0, {}
        for k, members in enumerate(intervals):
            for i in members:
                out[i] = epoch
            if k not in open_k:
                epoch += 1
        return out

    def order(self, members):
        """Row-block loops and waiting products of an interval run after its independent phases."""
        late = [i for i in members if (i in self.loops and self.loops[i]["waits"]) or i in self.waiting_products]
        return [i for i in members if i not in late] + late if late else members

    def closes_open(self, members):
        return frozenset(members) in self.open_after

    def _publisher(self, i, rows, alloc):
        blocks = -(-rows // self.rb_of[i])
        ptr = alloc(blocks)
        self.counter[i] = ptr
        return ptr

    def _wait_blocks(self, p, R, cols, rbc):
        """Wait for every producer block (producer's own rows per block) overlapping consumer block tr_j_."""
        rbp, cnt = self.rb_of[p], self.counter[p]
        need = f"(unsigned)(((tr_q_ + 1) * {rbp}LL < {R}LL ? {rbp}LL : {R}LL - tr_q_ * {rbp}LL) * {cols}LL)"
        end = f"((tr_j_ + 1) * {rbc}LL < {R}LL ? (tr_j_ + 1) * {rbc}LL : {R}LL)"
        return (f"for (long long tr_q_ = (tr_j_ * {rbc}LL) / {rbp}LL; tr_q_ * {rbp}LL < {end}; ++tr_q_) "
                f"latch_tr::wait_rows({cnt}, tr_q_, {need});")

    def product_code(self, i, body, alloc):
        """Bind row publication to an attention call, or an A-row readiness functor to a claimed product call."""
        from .ragged_experts_lower import CLAIM_SLOT
        info = self.products.get(i)
        if info is None:
            return body
        if str(self.phases[i].get("kind", "")).startswith("segmented_attention"):
            if info.get("wait") or not info.get("publish") or body.count("latch_segmented::run_") != 1:
                raise RuntimeError("tile_ready: unexpected attention producer form")
            cnt = self._publisher(i, info["rows"], alloc)
            return body.replace("latch_segmented::run_", f"A.tr_cnt = {cnt}; A.tr_rb = {self.rb_of[i]}; "
                                "A.tr_delay = 0;  "
                                "// tile-ready per-item row publication\nlatch_segmented::run_", 1)
        if body.count(CLAIM_SLOT) != 1:
            raise RuntimeError("tile_ready: unexpected claimed product call form")
        cnt = self._publisher(i, info["rows"], alloc) if info.get("publish") else "nullptr"
        wcnt, wcols, wrb = "nullptr", 0, self.rb_of[i]
        if info.get("wait"):
            p, wcols = info["wait"]
            if p not in self.counter:
                raise RuntimeError("tile_ready: waiting product emitted before its producer")
            wcnt, wrb = self.counter[p], self.rb_of[p]
        name = f"tr_rdy_{i}"
        body = body.replace(CLAIM_SLOT, f"{CLAIM_SLOT}, {name}", 1)
        pub_, wait_ = str(bool(info.get("publish"))).lower(), str(bool(info.get("wait"))).lower()
        decl = (f"const latch_tr::ReadyT<{pub_}, {wait_}> {name}{{{cnt}, {self.rb_of[i]}, nullptr, 0, "
                f"{wcnt}, {wcols}u, {wrb}, nullptr, 0}};  "
                f"// tile-ready readiness ({'publish' if info.get('publish') else ''}"
                f"{' + ' if info.get('publish') and info.get('wait') else ''}{'wait A rows' if info.get('wait') else ''})\n")
        head, brace, tail = body.partition("{")
        if not brace:
            raise RuntimeError("tile_ready: product body is not a block")
        return head + brace + "\n" + decl + tail

    def loop_code(self, i, alloc, code=None):
        """The consumer phase re-emitted as a row-block loop; each block first waits for its producer rows."""
        from . import row_chain
        info = self.loops[i]
        R, per_row, rb = info["rows"], info["per_row"], self.rb_of[i]
        blocks = -(-R // rb)
        waits = []
        for p, cols in info["waits"]:
            if p not in self.counter:
                raise RuntimeError("tile_ready: row-block loop emitted before its producer")
            waits.append(self._wait_blocks(p, R, cols, rb))
        body = restrict_flat(code, R * per_row, per_row, rb)
        self.forms[i] = "emitted_flat_loop" if body else "scalar_body"
        if body is None:
            body = row_chain.restrict(self.phases[i], ("elementwise", None), "", per_row)
        wait = ""
        if waits:
            wait = ("  if (threadIdx.x == 0) {\n    const long long tr_j_ = tr_b_;\n    "
                    + "\n    ".join(waits) + "\n    __threadfence();\n  }\n  __syncthreads();\n")
        return ("// tile-ready row-block loop: " f"{rb}-row blocks, static round robin over the grid in ascending order"
                f"{'; waits for producer rows' if waits else ''}\n"
                "#pragma unroll 1\n"
                f"for (long long tr_s_ = (long long)blockIdx.x; tr_s_ < {blocks}LL; tr_s_ += gridDim.x) {{\n"
                "  const long long tr_b_ = tr_s_;\n"
                + wait
                + f"  const long long latch_rc_r0 = tr_b_ * {rb}LL;\n"
                f"  const long long latch_rc_r1 = latch_rc_r0 + {rb}LL < {R}LL ? latch_rc_r0 + {rb}LL : {R}LL;\n"
                + body + "\n}")

    def consumer_code(self, i, alloc, code=None):
        return self.loop_code(i, alloc, code)

    def record(self):
        return dict(version=VERSION, mode=mode(), rows_per_block=self.rb,
                    rows_per_block_of={self.phases[i]["names"][0]: v for i, v in self.rb_of.items()},
                    boundaries=[dict(interval=b["interval"], edges=b["edges"],
                                     producers=sorted({e[0] for e in b["edges"]}),
                                     consumers=sorted({e[1] for e in b["edges"]}),
                                     independent=[self.phases[i]["names"][0] for i in b["independent"]],
                                     sunk=b.get("sunk", []),
                                     **({"counter_snapshots": b["counter_snapshots"]} if b.get("counter_snapshots") else {}))
                                for b in self.boundaries])


def plan(phases, intervals, layouts, *, placement=None, inputs=(), bodies=None, excluded=(),
         state_writes=None, state_inputs=(), returned=()):
    """Choose open boundaries; returns a Plan or None. ``excluded``: phases already taken by other rewrites."""
    global LAST_CENSUS
    LAST_CENSUS = []
    if not mode():
        return None
    rb = ROWS_PER_BLOCK
    excluded = set(excluded)
    root = lambda n: layouts[n].root if n in layouts else n
    names = lambda i: phases[i]["names"][0]
    writer_of = {w: i for i, ph in enumerate(phases) for w in ph["writes"]}
    inplace_targets = {}
    for res, (_index, how) in (placement or {}).items():
        if isinstance(how, str) and how.startswith("inplace:") and res in writer_of:
            inplace_targets.setdefault(writer_of[res], set()).add(root(inputs[int(how[8:])]))
    loops, products = {}, {}
    chosen, ordinal, census = [], 0, []
    where = {i: k for k, members in enumerate(intervals) for i in members}
    readers_of, writers_of_root = {}, {}
    for i, ph in enumerate(phases):
        for r in ph["reads"]:
            readers_of.setdefault(root(r), set()).add(i)
        for w in ph["writes"]:
            writers_of_root.setdefault(root(w), set()).add(i)
    state_targets = {value: target for target, value in (state_writes or {}).items() if target in state_inputs}
    returned = set(returned)

    def counter_snapshot(q, k, A, B):
        """Retained counter-only state copy after the consumer's full-completion barrier, or None.

        This recognizes instrumentation by its native counter source and state-write identity, never by an old
        phase number. No data/control consumer sees this snapshot until the kernel returns. Both the crossed and
        destination intervals leave the source untouched, and no other phase accesses the state destination.
        """
        if (k + 2 >= len(intervals) or q in excluded
                or _barrier_text(phases[q]) or any(attention_outputs(phases[i]) for i in B)):
            return None
        ph = phases[q]
        if (ph.get("kind") != "elementwise" or ph.get("targets") != ["aten.clone.default"]
                or ph.get("dtype") != "int64" or len(ph.get("names", ())) != 1
                or ph.get("writes") != ph["names"] or len(ph.get("reads", ())) != 1
                or ph.get("counter_effects") or ph.get("internal_mutations") or ph.get("active_values")):
            return None
        value, counter = ph["writes"][0], ph["reads"][0]
        target = state_targets.get(value)
        if target is None or value not in returned or root(value) != target:
            return None
        if readers_of.get(target) or writers_of_root.get(target) != {q}:
            return None
        source_layout, copy_layout, target_layout = (layouts.get(n) for n in (counter, value, target))
        if (source_layout is None or copy_layout is None or target_layout is None
                or source_layout.root != counter or target_layout.root != target
                or source_layout.offset or copy_layout.offset or target_layout.offset
                or not all(x.contiguous for x in (source_layout, copy_layout, target_layout))
                or tuple(source_layout.shape) != tuple(copy_layout.shape)
                or tuple(source_layout.shape) != tuple(target_layout.shape)
                or tuple(ph.get("shape", ())) != tuple(source_layout.shape)):
            return None
        publishers = [p for p in A if phases[p].get("kind") == "segmented_attention_forward"
                      and phases[p].get("head_width") == 32 and phases[p].get("work_counters") == counter
                      and counter in phases[p]["writes"] and attention_outputs(phases[p])]
        if len(publishers) != 1:
            return None
        # The copy will execute in interval k+2, so either concurrent or crossed writes would race/change its value.
        if any(where[i] in (k + 1, k + 2) for i in writers_of_root.get(counter, ())):
            return None
        return dict(copy=names(q), producer=names(publishers[0]), counter=counter, state_target=target,
                    shape=list(source_layout.shape), dtype="int64", from_interval=k + 1, to_interval=k + 2,
                    completion_barrier_after=k + 1,
                    proof="retained pure counter clone; no destination readers/other writers; no crossed source writes")

    def row_local(r, R, roots):
        """r views one of the published roots row by row: same R rows, the root's row pitch, and every element of
        view row i inside root row i (whole reshapes, in-row slices, unsqueezes)."""
        lay = layouts.get(r)
        if lay is None or lay.root not in roots or not lay.shape or int(lay.shape[0]) != R:
            return False
        rl = layouts.get(lay.root)
        if rl is None or not rl.contiguous or rl.offset or not rl.shape or int(rl.shape[0]) != R:
            return False
        pitch = _numel(rl.shape) // R
        strides = [int(x) for x in getattr(lay, "strides", ())] or None
        if strides is None:
            return lay.contiguous and not lay.offset and _numel(lay.shape) == _numel(rl.shape)
        if any(x < 0 for x in strides) or lay.offset < 0 or (R > 1 and strides[0] != pitch):
            return False
        reach = int(lay.offset) + sum((int(n) - 1) * x for n, x in zip(lay.shape[1:], strides[1:]))
        return reach < pitch

    # Interval 0 never produces: the per-launch counters are zeroed at kernel entry and ordered by its barrier.
    for k in range(1, len(intervals) - 1):
        A, B = intervals[k], intervals[k + 1]
        wrote = {}
        for p in A:
            for w in phases[p]["writes"]:
                wrote[root(w)] = p
        fed = sorted({wrote[root(r)] for q in B for r in phases[q]["reads"] if root(r) in wrote})
        if not fed:
            continue
        # Census only boundaries a publishing producer feeds.
        if not any(attention_outputs(phases[p]) for p in fed):
            continue
        reason = None
        touched_A = set(wrote) | {root(r) for p in A for r in phases[p]["reads"]}
        new_loops, new_products, independent, edges, sinks, snapshots = {}, {}, [], [], [], []
        for q in list(B):
            if q in excluded:
                reason = f"{names(q)}: taken by another rewrite"
            elif _barrier_text(phases[q]) or (bodies and bodies[q].get("internal_grid_barrier")):
                reason = f"{names(q)}: grid-barrier protocol inside a phase after the boundary"
            elif {root(w) for w in phases[q]["writes"]} & touched_A or inplace_targets.get(q, set()) & touched_A:
                reason = f"{names(q)}: writes storage the producer interval touches (WAR/WAW)"
            if reason:
                break
            deps = sorted({wrote[root(r)] for r in phases[q]["reads"] if root(r) in wrote})
            if not deps:
                independent.append(q)
                continue
            lf = loop_form(phases[q], layouts)
            if lf is not None:
                R, per_row = lf
                waits = []
                for p in deps:
                    outs = attention_outputs(phases[p])
                    if outs is None or p in excluded:
                        reason = (f"{names(q)} reads {names(p)} ({phases[p].get('kind')}:{phases[p].get('engine')}), "
                                  "not a readiness producer")
                        break
                    prows = int(layouts[outs[0]].shape[0])
                    pcols = _numel(layouts[outs[0]].shape[1:])
                    roots_p = {root(o) for o in outs}
                    for r in [r for r in phases[q]["reads"] if root(r) in wrote and wrote[root(r)] == p]:
                        lay = layouts.get(r)
                        if (lay is None or prows != R or not row_local(r, R, roots_p)
                                or len(lay.shape) != len(layouts[phases[q]["writes"][0]].shape)):
                            reason = f"{names(q)} reads {r} of {names(p)} not row-locally"
                            break
                    if reason:
                        break
                    new_products.setdefault(p, {}).update(rows=prows, cols=pcols, publish=True)
                    waits.append((p, pcols))
                    edges.append((names(p), names(q)))
                if reason:
                    break
                new_loops.setdefault(q, {}).update(rows=R, per_row=per_row, waits=waits)
                continue
            if product_waits_ok(phases[q]):
                a = phases[q]["reads"][0]
                if [p for p in deps if any(root(r) in wrote and wrote[root(r)] == p and r != a
                                           for r in phases[q]["reads"])] or len(deps) != 1:
                    reason = f"{names(q)}: product reads interval-k values other than its A operand"
                    break
                p = deps[0]
                outs = attention_outputs(phases[p])
                lay = layouts.get(a)
                if outs is None:
                    reason = f"{names(q)}: A operand from {names(p)} ({phases[p].get('kind')}), not a readiness producer"
                    break
                prows = int(layouts[outs[0]].shape[0])
                roots_p = {root(o) for o in outs}
                if (lay is None or len(lay.shape) < 2 or not row_local(a, prows, roots_p)
                        or _numel(layouts[lay.root].shape) != _numel(lay.shape)):
                    reason = f"{names(q)}: A operand {a} is not a whole row view of a published root"
                    break
                kcols = _numel(lay.shape[1:])
                new_products.setdefault(p, {}).update(rows=prows, cols=kcols, publish=True)
                new_products.setdefault(q, {}).update(rows=prows, wait=(p, kcols),
                                                      cols=new_products.get(q, {}).get("cols", 0))
                edges.append((names(p), names(q)))
                continue
            snapshot = counter_snapshot(q, k, A, B)
            if snapshot is not None:
                sinks.append(q)
                snapshots.append(snapshot)
                continue
            reason = f"{names(q)} ({phases[q].get('kind')}:{phases[q].get('engine') or ''}): unsupported consumer"
            break
        if not reason and not edges:
            reason = "no consumer after the boundary"
        if not reason:
            for p in A + B:
                if p in excluded or _barrier_text(phases[p]) or (bodies and bodies[p].get("internal_grid_barrier")):
                    reason = f"{names(p)}: grid-barrier protocol or excluded phase in the pair"
                    break
        if not reason:
            this = ordinal
            ordinal += 1
            for q in sinks:          # apply the moves only for an opened boundary
                intervals[k + 1].remove(q)
                intervals[k + 2].append(q)
                intervals[k + 2].sort()
                where[q] = k + 2
            chosen.append(dict(interval=k, ordinal=this, members=list(A), edges=edges, independent=independent,
                               sunk=[names(q) for q in sinks],
                               **({"counter_snapshots": snapshots} if snapshots else {})))
            for d_, n_ in ((loops, new_loops), (products, new_products)):
                for ph_, v_ in n_.items():
                    merged = d_.setdefault(ph_, {})
                    for key_, val_ in v_.items():
                        if key_ == "waits":
                            merged.setdefault("waits", []).extend(val_)
                        elif key_ == "publish":
                            merged["publish"] = merged.get("publish", False) or val_
                        else:
                            merged[key_] = val_
        census.append(dict(interval=k, producers=[names(p) for p in fed], opened=reason is None, reason=reason,
                           sunk=[names(q) for q in sinks] if reason is None else []))
    LAST_CENSUS = census
    if not chosen:
        return None
    for v in loops.values():
        v.setdefault("waits", [])
        v.setdefault("publish", False)
    return Plan(chosen, phases, rb, loops, products)
