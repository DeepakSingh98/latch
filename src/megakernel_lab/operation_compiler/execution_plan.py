"""Shared execution contracts and a deterministic static cooperative planner.

Bodies remain opaque CUDA calculations. This module owns dependencies and
publication, never capture, differentiation or the bodies' arithmetic.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass

_EXECUTION = ContextVar("latch_execution", default="legacy")
_REQUESTED_EXECUTION = ContextVar("latch_requested_execution", default=None)
_PRECISION = ContextVar("latch_execution_precision", default="strict")
VERSION = 1


def current_execution():
    return _EXECUTION.get()


def requested_execution():
    """Explicit enclosing diagnostic scope, if any; omitted public calls use auto."""
    return _REQUESTED_EXECUTION.get()


def current_precision():
    return _PRECISION.get()


@contextmanager
def execution_scope(execution=None, precision=None):
    from .lower import LoweringError
    selected = _EXECUTION.get() if execution is None else execution
    if selected not in ("legacy", "compose"):
        raise LoweringError("execution must be legacy or compose")
    policy = _PRECISION.get() if precision is None else precision
    if policy not in ("strict", "practical"):
        raise LoweringError("execution precision must be strict or practical")
    token, arithmetic = _EXECUTION.set(selected), _PRECISION.set(policy)
    requested = _REQUESTED_EXECUTION.set(execution if execution is not None else _REQUESTED_EXECUTION.get())
    try:
        yield
    finally:
        _PRECISION.reset(arithmetic)
        _EXECUTION.reset(token)
        _REQUESTED_EXECUTION.reset(requested)


@dataclass(frozen=True)
class Operand:
    name: str
    root: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    offset: int
    dtype: str


@dataclass(frozen=True)
class BodyContract:
    index: int
    body: str
    reads: tuple[Operand, ...]
    writes: tuple[Operand, ...]
    internal_mutations: tuple[Operand, ...]
    counter_effects: tuple[dict, ...]
    private_scratch: tuple[dict, ...]
    global_effects: tuple[dict, ...]
    dependencies: tuple[int, ...]
    shared_bytes: int
    threads: int
    work_items: int
    grid_scope: str
    internal_grid_barrier: bool
    global_fence: bool
    precision: str


def _value(layout, key, default=None):
    return layout.get(key, default) if isinstance(layout, dict) else getattr(layout, key, default)


def requires_grid_fence(phase):
    return bool(phase.get("global_fence") or phase.get("internal_grid_barrier")
                or phase.get("internal_grid_barriers") or "grid.sync(" in phase.get("code", ""))


def contracts(phases, layouts, dtypes=None):
    """Resolve aliases and RAW/WAR/WAW hazards in captured effect order.

An operand produced and consumed inside a single body is local to that body.
Writes through aliases conservatively conflict at whole-root granularity. A
body with internal grid synchronization is an ordering fence; no CTA subset
may execute it. CUDA register occupancy still requires the runtime query.
"""
    from .lower import LoweringError
    dtypes = dtypes or {}
    writer, readers, result = {}, {}, []
    fence = None

    def operand(name):
        if name not in layouts:
            raise LoweringError(f"execution contract operand has no layout: {name}")
        layout = layouts[name]
        return Operand(name, str(_value(layout, "root", name)), tuple(_value(layout, "shape")),
                       tuple(_value(layout, "strides")), int(_value(layout, "offset", 0)),
                       str(dtypes.get(name, _value(layout, "dtype", "unknown"))))

    for index, phase in enumerate(phases):
        rd = tuple(operand(name) for name in dict.fromkeys(phase.get("reads", ())))
        wr = tuple(operand(name) for name in dict.fromkeys(phase.get("writes", ())))
        mutations = tuple(operand(name) for name in dict.fromkeys(phase.get("internal_mutations", ())))
        rroots = {x.root for x in rd} | {x.root for x in mutations}
        wroots = {x.root for x in wr} | {x.root for x in mutations}
        deps = {writer[root] for root in rroots | wroots if root in writer}
        deps.update(r for root in wroots for r in readers.get(root, ()))
        if fence is not None:
            deps.add(fence)
        internal = bool(phase.get("internal_grid_barrier") or phase.get("internal_grid_barriers")
                        or "grid.sync(" in phase.get("code", ""))
        global_fence = requires_grid_fence(phase)
        if global_fence:
            # LATCH-STEP-06 (ops/loop_slack.py): a slack-placed side product shares the fence's stage and runs after
            # the fenced body on every CTA (stage order); the fence does not wait for it. Its own data dependencies
            # and any earlier fence still order it.
            deps.update(j for j in range(index) if not phases[j].get("step06_slack"))
            fence = index
        shared, threads = int(phase.get("shared_bytes", 0)), int(phase.get("threads", 256))
        if shared < 0 or shared > 232448 or threads != 256:
            raise LoweringError(f"body {index} exceeds the common 256-thread cooperative resource ABI")
        result.append(BodyContract(index, str(phase.get("engine", phase.get("kind", "unknown"))),
            rd, wr, mutations, tuple(phase.get("counter_effects", ())), tuple(phase.get("private_scratch", ())),
            tuple(phase.get("global_effects", ())),
            tuple(sorted(deps)), shared, threads,
            int(phase.get("work_items", phase.get("tiles", 0))),
            str(phase.get("execution_grid_scope", "physical_cooperative_grid")), internal, global_fence,
            str(phase.get("arithmetic", current_precision()))))
        for root in rroots - wroots:
            readers.setdefault(root, set()).add(index)
        for root in wroots:
            writer[root] = index
            readers[root] = set()
    return result


def static_intervals(phases, layouts):
    """ASAP levels; each body keeps its native physical-grid work assignment."""
    items = contracts(phases, layouts)
    levels, stages = [], []
    for item in items:
        level = 1 + max((levels[p] for p in item.dependencies), default=-1)
        levels.append(level)
        while len(stages) <= level:
            stages.append([])
        stages[level].append(item.index)
    return stages


def dependency_successors(phases, layouts):
    """Reverse the complete effect edges for safe later placement of a body.

    A delayed body must remain strictly before every successor's current stage.
    Checking direct successors preserves transitive order too; callers update
    their stage map after every accepted move.
    """
    result = [[] for _ in phases]
    for item in contracts(phases, layouts):
        for predecessor in item.dependencies:
            result[predecessor].append(item.index)
    return result


def region_stage_effects(reads, writes, *, global_fence=False):
    """Carry a global fence through the loop's cross-region merge scheduler.

    All stages read the token; only a fence writes it. Therefore ordinary stages
    stay independent while every earlier/later track waits across a fenced body.
    The token denotes synchronization ownership, not allocated tensor storage.
    """
    token = ("compose_global_grid_fence",)
    return (frozenset(reads) | {token}, frozenset(writes) | ({token} if global_fence else set()))


def record_plan(manifest, *, phases=None, layouts=None, dtypes=None):
    """Bind contracts to the *emitted* intervals and checked arena lifetimes."""
    from .lower import LoweringError
    phases = manifest.get("phases", ()) if phases is None else phases
    layouts = manifest.get("layouts", {}) if layouts is None else layouts
    items = contracts(phases, layouts, dtypes)
    intervals = manifest.get("barrier_intervals", [])
    assigned = {p: stage for stage, members in enumerate(intervals) for p in members}
    if len(assigned) != len(items) or sum(map(len, intervals)) != len(items):
        raise LoweringError("execution plan must schedule each body exactly once")
    # LATCH-SELECTIVE-10 rowchain (ops/row_chain.py): a merged interval runs ordered row-local sub-stages separated by
    # CTA barriers, each CTA on its own rows; a body may depend on an earlier sub-stage of its interval.
    order = {p: (assigned[p], 0) for p in assigned}
    # LATCH-TRAIN-SYNC ops.cast_sums: the fused sums of an interval follow its other bodies; each CTA's sum lanes read
    # only the producer values its own threads computed (cast_sums.py).
    for stage, groups in list((manifest.get("row_chain_substages") or {}).items()) + list(
            (manifest.get("cast_sum_substages") or {}).items()):
        stage = int(stage)
        if sorted(p for g in groups for p in g) != sorted(intervals[stage]):
            raise LoweringError("row-chain sub-stages differ from their interval")
        for sub, group in enumerate(groups):
            for p in group:
                order[p] = (stage, sub)
    for item in items:
        conflicts = [p for p in item.dependencies if order[p] >= order[item.index]]
        if conflicts:
            import json

            def describe(index):
                body = items[index]
                return {"index": index, "body": body.body, "stage": assigned[index],
                        "names": phases[index].get("names", ()),
                        "reads": sorted({x.root for x in body.reads}),
                        "writes": sorted({x.root for x in body.writes}),
                        "internal_mutations": sorted({x.root for x in body.internal_mutations}),
                        "global_fence": body.global_fence}

            detail = {"body": describe(item.index), "dependencies": [describe(p) for p in conflicts],
                      "deferred_products": manifest.get("deferred_products", []),
                      "sunk_recomputation": manifest.get("group_rules", {}).get("sunk_recomputation", [])}
            raise LoweringError(f"execution plan violates dependency of body {item.index}: "
                                + json.dumps(detail, sort_keys=True))
    scratch = {name: {"word_offset": int(offset), "lifetime": manifest.get("lifetimes", {}).get(name)}
               for name, offset in manifest.get("offsets", {}).items()}
    from .group_rules import FAST_ACTIVATION
    manifest["execution_plan"] = {
        "version": VERSION, "selected": "compose", "scheduler": "static_dependency_intervals",
        "bodies": [asdict(item) for item in items], "stages": intervals,
        "scratch": scratch, "arena_bytes": manifest.get("arena_bytes", 0),
        "persistent_counter_words": manifest.get("persistent_counter_words", 0),
        "resources": {"shared_bytes": manifest.get("shared_bytes", 0), "threads": 256,
                      "resident_blocks_per_sm": manifest.get("resource_policy", {}).get("resident_blocks_per_sm", 1),
                      "registers": "compiler allocation; actual runtime occupancy admission required"},
        "publication": "cooperative grid barrier between dependency intervals; CTA shared reuse barriers inside intervals",
        "state_commit": ("validated same-index last-use aliases; remaining writes wait for final commit"
                         if manifest.get("in_place_commits") else "all old-state reads complete before the final commit"),
        "arithmetic": current_precision(), "fast_activation": bool(FAST_ACTIVATION.get()), "queue": False,
        "arithmetic_scope": "flat execution recipe; capture_precision separately records region AD policy",
        "qualification": "structural contract only; fresh complete-update numerical and runtime checks required",
    }
    return manifest["execution_plan"]


def record_composed_regions(manifest):
    children = manifest.get("regions", manifest.get("branches", {}))
    manifest["execution_plan"] = {
        "version": VERSION, "selected": "compose", "scheduler": "static_region_composition",
        "regions": {name: child.get("execution_plan") for name, child in children.items()},
        "arena_bytes": manifest.get("arena_bytes", 0), "shared_bytes": manifest.get("shared_bytes", 0),
        "publication": "existing uniform branch / runtime loop dependency boundaries",
        "queue": False, "qualification": "fresh complete-update qualification required",
    }
