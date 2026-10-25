"""Opt-in gradient-traffic rules for flat common compose training programs (LATCH-SELECTIVE-10).

Selected through the normal build API: ``CapturedModule.build(..., accumulation_rules="castview")`` (or a
'+'-joined subset of ``RULES``). Omitting it (None) leaves lowering and generated source unchanged.

Every rule is dispatched on recognized operation structure (targets, dtypes, shapes, views and readers), never on
model names, and leaves values bit for bit unchanged: it moves where a value is computed or whether it is stored,
never an expression, rounding or reduction order.

* ``castview`` -- an exact floating widening cast (BF16/FP16 -> FP32) of a whole tensor whose every reader is a
  ``select`` view (each read by non-view operations) is replaced by one cast per view of the source's matching
  view: ``select(cast(x), d, i)`` -> ``cast(select(x, d, i))``. A cast is elementwise and a select only chooses
  elements, so every element has the same bits. Typical case: the FP32 copy of a stacked expert weight gradient
  (``float(ragged_outer(dy, x))``) read per expert by the update. The per-view casts then join their readers'
  elementwise chains, so the FP32 stacked copy is no longer written and re-read; the readers load the BF16 product.
  Requires that the cast and its views are not program outputs (e.g. ``gradient_outputs="consumed"``): a public
  FP32 gradient must be stored.
* ``fsgather`` -- (with update_schedule's ``fsbatch``) an interval's exact multi-CTA full sums are emitted
  consecutively after its other phases (phases of one dependency interval are independent), so one fsbatch body
  holds up to 32 of them instead of runs of 4-5 broken by interleaved single-CTA reductions.
* ``fswarp`` -- the combines an fsbatch body's CTA owns (it was last to arrive) run in parallel on its 8 warps,
  each an exact one-warp replica of the block combine: the same per-thread partial sums in the same order, the same
  xor butterflies and the same final 8-term butterfly (update_schedule._warp_combine).
* ``wsoftn`` -- softmax / softmax-backward rows narrower than a warp (e.g. a 16-expert router) use the exact
  warp-per-row replay of the CTA-per-row tree (transformer_lower; P-OPS2-D form) instead of one 256-thread CTA per
  row with six CTA barriers.
* ``fsdeep`` -- eight-lane FP32 square sums (square_sum_association.lane_loop) load three rounds of terms before
  adding them, each lane in its original order (24 loads in flight per thread instead of 8).
* ``rorder`` -- the stable bucket permutation of segmented (expert) dispatch uses one block scan per bucket
  (``latch_ragged::order_scan``) instead of 16 chunked Hillis-Steele scans; the permutation is unique, so the
  integers are identical.
* ``cmulti`` -- (with update_schedule's ``csink``) an elementwise chain read by several same-shape elementwise
  chains joins all of them in one loop (group_rules.sink_multi_reader_chains): e.g. new AdamW moments read by the
  parameter update and by the participation select that commits them; the moments are no longer stored and re-read.
* ``rowchain`` -- runs of consecutive small row-local intervals (e.g. a router's cast/softmax/top-k/gather/mask
  chain over tokens) share one grid barrier: each CTA computes its own token rows in every phase (ops/row_chain.py).
* ``fsfence`` -- (with fsbatch) a batch's partial writes, fences and tickets are issued by all its lanes in one
  round (one ``__threadfence`` per lane) instead of one divergent write/fence/atomic block per sum.
* ``accfuse`` -- (after castview) a BF16 segmented outer product whose every per-segment result is read only as
  ``acc_e + where(flag_e, float(result_e), zeros_like)`` becomes one ``ragged_outer_acc`` whose epilogue adds the
  same exactly-widened, masked BF16 result to acc_e (kernels/ragged_tma_acc.cuh); the BF16 product is no longer
  stored and re-read, and an accumulator state is updated in place (same-index read then write).
* ``fsdeepx`` -- (LATCH-B1-FAST, with fsdeep) eight rounds of terms are loaded before adding instead of three (64
  independent loads in flight per thread); each lane still adds its terms in the original round order. The long
  per-thread chains of the 12.3M-element embedding/readout square sums are latency-bound at three rounds.
* ``vec8`` / ``vec4m`` -- (LATCH-B1-FAST) wider float4 elementwise loops: ``vec8`` keeps eight float4 indices per
  thread in flight (instead of four) for phases above 65,536 float4; ``vec4m`` keeps four for phases of 16,384 to
  65,536 float4 (instead of one). The update-schedule work queue then cuts such bodies into items of one wider
  round, so each queued item moves twice the bytes per memory latency. Same per-element expressions, same single
  store per element; only how many elements a thread has in flight changes.
* ``fsmix`` -- (with fsbatch) up to four eight-lane square sums of a batch compute their partial lanes in one
  round-interleaved loop (update_schedule._mixed_calculations): every lane adds the same terms in the same order;
  the independent loads of several sums are in flight together instead of one small sum after another.
* ``fsmixl`` -- (LATCH-TRAIN-SYNC, with fsmix) sums whose summand is evaluated through one-line leaf lambdas defined
  before the exact body (e.g. the participation-masked square ``where(flag, g, 0)**2`` of each expert gradient in a
  clipped update) join fsmix's round-interleaved groups; each lambda is defined once per sum under a sum-unique name.
  Without it every such sum of a batch issues its own dependent load rounds one sum after another (latency-bound:
  each CTA owns ~9 summands per thread of a 294,912-element expert gradient). Same lanes, same order per lane.
* ``fsload`` -- (LATCH-TRAIN-SYNC, with fsmix; implies fsmixl's admission) in the round-interleaved groups, summands that
  index their view only through r wrapped modulo its extent are evaluated unconditionally (in-bounds loads) and only
  the additions keep their round/tail guards; a leaf's where-masked operand ``(FLAG[0] ? P[(i)] : Z)`` is loaded before
  the selection. Every lane still adds the same summands in the same order; loads behind the flag tests and the tail
  guards were issued one dependent round at a time (measured: 32 expert-gradient sums 96.7 -> 33.5 us in isolation).
* ``fscast`` -- (LATCH-TRAIN-SYNC, with fsbatch + fsmix + fsload; ops/cast_sums.py) exact square sums over the
  leading-dimension slices of a root written by a single-line elementwise producer (e.g. the FP32 copy of a stacked
  expert-weight gradient, summed per expert for the clip norm) move into the producer's interval; each sum's lanes
  compute and store exactly the producer elements of its own CTA chunk and add the leaf of the same register value.
  The root is no longer re-read by the sums; every value and every addition order is unchanged.
* ``fsmix8`` -- (LATCH-TRAIN-SYNC, with fsmix) eight sums per round-interleaved group instead of four.
* ``fsspread`` -- (LATCH-TRAIN-SYNC, with fsbatch + fswarp) sum t of a batch is combined by a designated warp of CTA
  t % gridDim.x once all its tickets are in, instead of every combine running on the last-arriving CTA after its own
  partials: same partials, same order, same butterflies, same ticket reset. No CTA waits before its own deposits.
* ``dcast`` -- (capture ABI, ops/derived_casts.py) a ``cat_cast`` of FP32 parameters becomes persistent derived state
  read by every reader; a program that leaves those parameters unchanged commits nothing, one that updates them
  commits the cast of the updated values, and the runtime recasts after host writes. Same values every launch.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace

UPDATE_PRESET_RULES = ("fsgather", "fswarp", "wsoftn", "fsdeep", "rorder", "rowchain", "fsmix", "fsfence")
RULES = ("castview", "fsgather", "fswarp", "wsoftn", "fsdeep", "rorder", "cmulti", "rowchain", "fsmix", "accfuse", "fsfence", "dcast",
         "fsdeepx", "vec8", "vec4m", "fsmixl", "fsload", "fscast", "fsmix8", "fsspread")
# LATCH-B32-TRANSFER: "update" = the rules measured non-regressing on a one-update training program (no
# accumulator state; TinyMoE recipe Q, B32 x 1024, same idle H100, paired with the program without them). Not in it:
# castview (+0.34 ms there: the FP32 expert-gradient copy has two readers, AdamW and the clip square sums, and each
# re-reads the BF16 product instead), cmulti (+0.06 ms, neutral), accfuse (needs an accumulator; slower where it
# applies). Selecting a preset never changes values: every rule is exact by construction.
PRESETS = {"update": UPDATE_PRESET_RULES}
_RULES = ContextVar("latch_accumulation_rules", default=None)
VERSION = 1
EXACT_WIDENING = {("bfloat16", "float32"), ("float16", "float32")}
CAST = "aten._to_copy.default"
SELECT = "aten.select.int"


def rules_of(value):
    if value is None:
        return ()
    if not isinstance(value, str) or not value:
        raise ValueError(f"accumulation_rules must be None, one of {sorted(PRESETS)} or '+'-joined rules from {RULES}")
    if value in PRESETS:
        return tuple(PRESETS[value])
    # LATCH-TRAIN-SYNC: a preset name inside a '+'-joined list expands in place (e.g. "update+fsmixl").
    chosen = tuple(dict.fromkeys(rule for part in value.split("+") for rule in PRESETS.get(part, (part,))))
    if not chosen or set(chosen) - set(RULES):
        raise ValueError(f"accumulation_rules must be None, one of {sorted(PRESETS)} or '+'-joined rules from {RULES}")
    return chosen


def validate(value):
    rules_of(value)
    return value


def current():
    return _RULES.get()


def active(rule):
    return rule in rules_of(_RULES.get())


@contextmanager
def scope(value):
    token = _RULES.set(validate(value))
    try:
        yield
    finally:
        _RULES.reset(token)


def features():
    """Extra group-rule features for the standalone flat common planner."""
    chosen = rules_of(_RULES.get())
    return "".join("+" + rule for rule in ("fsgather", "fswarp", "cmulti", "rowchain", "fsmix", "fsfence", "vec8", "vec4m",
                                           "fsmixl", "fsload", "fscast", "fsmix8", "fsspread")
                   if rule in chosen)


def _protected(graph):
    from .remat import _names
    protected = set(graph.inputs)
    _names(graph.outputs, protected)
    _names(getattr(graph, "state_writes", None) or {}, protected)
    return protected


def _contiguous_strides(shape):
    strides, acc = [], 1
    for d in reversed(shape):
        strides.insert(0, acc)
        acc *= max(int(d), 1)
    return tuple(strides)


def _fresh(stem, used):
    name, k = f"{stem}__cv", 1
    while name in used:
        k += 1
        name = f"{stem}__cv{k}"
    used.add(name)
    return name


def _cast_views(graph, report):
    """castview (module docstring)."""
    from .ir import Operation, Ref, Value
    values, operations = graph.values, graph.operations
    protected = _protected(graph)
    readers = {}
    for op in operations:
        for name in dict.fromkeys(op.reads):
            readers.setdefault(name, []).append(op)
    used = set(values) | {op.name for op in operations}
    plan, records = {}, []
    for op in operations:
        if (op.target != CAST or len(op.args) != 1 or len(op.outputs) != 1 or op.effect != "pure"
                or set(op.kwargs) - {"dtype", "layout", "device"}
                or str(op.kwargs.get("layout", "strided")).split(".")[-1] != "strided"):
            continue
        source = op.args[0].name if isinstance(op.args[0], Ref) else None
        cast = op.outputs[0]
        if source is None or source not in values or cast in protected:
            continue
        src, out = values[source], values[cast]
        if ((src.dtype, out.dtype) not in EXACT_WIDENING or out.alias_of is not None
                or tuple(src.shape) != tuple(out.shape)):
            continue
        views = readers.get(cast, [])
        if not views or any(v.target != SELECT or v.synchronization != "view" or len(v.outputs) != 1
                            or len(v.args) != 3 or not isinstance(v.args[0], Ref) or v.args[0].name != cast
                            or v.outputs[0] in protected for v in views):
            continue
        if any(not readers.get(v.outputs[0]) or any(r.synchronization == "view" for r in readers[v.outputs[0]])
               for v in views):
            continue
        rank = len(src.shape)
        root = src.alias_of or source
        rewritten = []
        for v in views:
            dim, index = int(v.args[1]), int(v.args[2])
            dim = dim % rank
            size = int(src.shape[dim])
            index = index % size if index < 0 else index
            if not 0 <= index < size:
                break
            shape = tuple(int(s) for k, s in enumerate(src.shape) if k != dim)
            strides = tuple(int(s) for k, s in enumerate(src.strides) if k != dim)
            offset = int(src.storage_offset) + index * int(src.strides[dim])
            name = _fresh(v.outputs[0], used)
            view_op = Operation(name, SELECT, (Ref(source), v.args[1], v.args[2]), {}, (source,), (name,),
                                tuple(op.dependencies), (source,), (), synchronization="view", effect="pure")
            view_value = Value(name, shape, src.dtype, strides, role="temporary", alias_of=root,
                               storage_offset=offset, producer=name)
            target = v.outputs[0]
            cast_op = Operation(v.name, CAST, (Ref(name),), dict(op.kwargs), (name,), (target,), (name,),
                                (name,), (target,), synchronization=op.synchronization, effect="pure")
            cast_value = Value(target, shape, out.dtype, _contiguous_strides(shape), role="temporary",
                               producer=v.name)
            rewritten.append((v, view_op, view_value, cast_op, cast_value))
        else:
            plan[op.name] = []
            for v, view_op, view_value, cast_op, cast_value in rewritten:
                plan[v.name] = (view_op, view_value, cast_op, cast_value)
            numel = 1
            for s in out.shape:
                numel *= int(s)
            records.append(dict(cast=cast, source=source, views=len(views), shape=list(out.shape),
                                fp32_bytes_not_stored=4 * numel))
    if not records:
        report["castview"] = []
        return graph
    new_values = dict(values)
    new_ops = []
    # Each per-view cast is emitted immediately before its first reader, so it joins that reader's elementwise
    # chain (lowering merges an elementwise operation into the preceding same-shape phase); its source view
    # follows the original product, which precedes every reader.
    pending = {}
    for op in operations:
        entry = plan.get(op.name)
        if entry == []:
            new_values.pop(op.outputs[0], None)
            continue
        if entry is not None:
            view_op, view_value, cast_op, cast_value = entry
            pending[cast_value.name] = (view_op, cast_op)
            new_values[view_value.name] = view_value
            new_values[cast_value.name] = cast_value
            continue
        for name in dict.fromkeys(op.reads):
            if name in pending:
                new_ops += list(pending.pop(name))
        new_ops.append(op)
    if pending:
        raise RuntimeError("castview: a rewritten view has no reader")
    report["castview"] = records
    report["castview_fp32_bytes_not_stored"] = sum(r["fp32_bytes_not_stored"] for r in records)
    return replace(graph, values=new_values, operations=new_ops)


def apply_ir(graph):
    """(graph, report) after the selected IR rules; unchanged graph when nothing is recognized."""
    from .ir import TrainingIR
    chosen = rules_of(_RULES.get())
    report = {"version": VERSION, "rules": list(chosen)}
    if not chosen or not isinstance(graph, TrainingIR) or not isinstance(graph.outputs, dict):
        report["reason"] = "no rules selected or not a structured flat training graph"
        return graph, report
    if "castview" in chosen:
        graph = _cast_views(graph, report)
    if "accfuse" in chosen:
        if "castview" not in chosen:
            report["accfuse"] = {"reason": "requires castview (per-segment casts of the product)"}
        else:
            graph = _accumulating_products(graph, report)
    return graph, report


OUTER = "latch_ops.ragged_outer.default"
OUTER_ACC = "latch_ops.ragged_outer_acc"


def _accumulating_products(graph, report):
    """accfuse (module docstring): ragged_outer + its E per-segment accumulate chains -> one ragged_outer_acc."""
    from .ir import Operation, Ref
    values, operations = graph.values, graph.operations
    protected = _protected(graph)
    readers, producer = {}, {}
    for op in operations:
        for name in dict.fromkeys(op.reads):
            readers.setdefault(name, []).append(op)
        for out in op.outputs:
            producer[out] = op
    position = {op.name: k for k, op in enumerate(operations)}
    ref = lambda x: x.name if isinstance(x, Ref) else None
    replace_at, removed, records = {}, set(), []
    for op in operations:
        if op.target != OUTER or len(op.outputs) != 1 or len(op.args) != 3 or op.kwargs:
            continue
        ro = op.outputs[0]
        v = values[ro]
        if ro in protected or v.dtype != "bfloat16" or len(v.shape) != 3 or v.alias_of is not None:
            continue
        E, M, N = (int(d) for d in v.shape)
        views = readers.get(ro, [])
        if len(views) != E:
            continue
        chains = {}
        for view in views:
            if (view.target != SELECT or view.synchronization != "view" or len(view.args) != 3
                    or ref(view.args[0]) != ro or int(view.args[1]) % 3 != 0):
                break
            e = int(view.args[2]) % E
            vr = readers.get(view.outputs[0], [])
            if len(vr) != 1 or vr[0].target != CAST or len(vr[0].args) != 1:
                break
            cast = vr[0]
            c = cast.outputs[0]
            cr = readers.get(c, [])
            zl = [r for r in cr if r.target == "aten.zeros_like.default"]
            wh = [r for r in cr if r.target == "aten.where.self"]
            if len(cr) != 2 or len(zl) != 1 or len(wh) != 1:
                break
            zl, wh = zl[0], wh[0]
            if (readers.get(zl.outputs[0], []) != [wh] or len(zl.args) != 1 or ref(zl.args[0]) != c
                    or set(zl.kwargs) - {"pin_memory", "dtype", "layout", "device", "memory_format"}
                    or len(wh.args) != 3 or ref(wh.args[1]) != c or ref(wh.args[2]) != zl.outputs[0]
                    or ref(wh.args[0]) is None):
                break
            flag = ref(wh.args[0])
            ar = readers.get(wh.outputs[0], [])
            if len(ar) != 1 or ar[0].target != "aten.add.Tensor" or ar[0].kwargs or len(ar[0].args) != 2:
                break
            add = ar[0]
            acc = ref(add.args[0])
            if acc is None or ref(add.args[1]) != wh.outputs[0] or len(add.outputs) != 1:
                break
            fv, av, ov, cv = values[flag], values[acc], values[add.outputs[0]], values[c]
            if (fv.dtype != "bool" or int(fv.numel) != 1 or av.dtype != "float32" or tuple(av.shape) != (M, N)
                    or ov.dtype != "float32" or tuple(ov.shape) != (M, N) or cv.dtype != "float32" or e in chains
                    or any(x in protected for x in (view.outputs[0], c, zl.outputs[0], wh.outputs[0]))
                    or (acc in producer and position[producer[acc].name] > position[add.name])):
                break
            chains[e] = (view, cast, zl, wh, add, flag, acc)
        else:
            if len(chains) != E:
                continue
            order = [chains[e] for e in range(E)]
            flags = tuple(Ref(ch[5]) for ch in order)
            accs = tuple(Ref(ch[6]) for ch in order)
            outs = tuple(ch[4].outputs[0] for ch in order)
            reads = tuple(dict.fromkeys([*op.reads, *(f.name for f in flags), *(a.name for a in accs)]))
            deps = tuple(dict.fromkeys(producer[r].name for r in reads if r in producer))
            new = Operation(ro + "__acc", OUTER_ACC, (op.args[0], op.args[1], op.args[2], flags, accs), {},
                            reads, outs, deps, reads, outs, synchronization=op.synchronization, effect="pure")
            # Emitted after the latest producer of its inputs (the flags are usually computed with the update).
            anchor = max([op.name] + [producer[r].name for r in reads if r in producer], key=lambda x: position[x])
            replace_at[anchor] = replace_at.get(anchor, []) + [new]
            removed.add(op.name)
            for ch in order:
                removed.update(x.name for x in ch[:5])
            records.append(dict(product=ro, segments=E, shape=[M, N], results=list(outs[:2]) + ["..."],
                                bf16_bytes_not_stored=2 * E * M * N))
    if not records:
        report["accfuse"] = []
        return graph
    new_list = [new for group in replace_at.values() for new in group]
    kept = {out for new in new_list for out in new.outputs}
    renamed = {}                     # removed add operation -> the new operation now producing its result
    new_values = dict(values)
    for op in operations:
        if op.name in removed:
            for out in op.outputs:
                if out not in kept:
                    new_values.pop(out, None)
    for new in new_list:
        for out in new.outputs:
            new_values[out] = replace(values[out], producer=new.name)
            renamed[producer[out].name] = new.name
    fixed = []
    for op in operations:
        if op.name not in removed:
            if any(d in renamed for d in op.dependencies):
                op = replace(op, dependencies=tuple(dict.fromkeys(renamed.get(d, d) for d in op.dependencies)))
            fixed.append(op)
        fixed.extend(replace_at.get(op.name, ()))
    report["accfuse"] = records
    report["accfuse_bf16_bytes_not_stored"] = sum(r["bf16_bytes_not_stored"] for r in records)
    return replace(graph, values=new_values, operations=fixed)
