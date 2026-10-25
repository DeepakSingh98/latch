"""Last-use state placement for common flat phases.

The admission is adapted from fusion_reference's elementwise in-place commit
rule. It changes storage and dependencies, never the update expression. Opaque
matrix/attention bodies retain a final copy unless a separate adapter proves
their read-before-store contract.
"""
from __future__ import annotations

from collections import Counter
import re

# LATCH-MOE4-PARITY: admit any producer kind when the destination's old value is never read in the region.
UNREAD_TARGET_ALIASES = True
# LATCH-MOE-EMA-FAST: an unread destination may also take a value read through tensor-map descriptors when every
# such descriptor is a provider (POPS4B_TMAP) map: the provider binds each map through the final layouts, so the
# map then addresses the destination's input storage (encoded once with the bound arguments). Literal-root P5C
# maps keep their storage. False restores the exclusion of every mapped value.
PROVIDER_MAPPED_ALIASES = True


def last_use_state_aliases(phases, layouts, dtypes, writes, inputs, *, ident):
    """Return target -> new value for proved same-index elementwise commits.

    Run after body fusion, before dependency and allocation planning. Layouts
    are not mutated here. The caller aliases every view of an admitted result
    to its target; the common RAW/WAR/WAW planner then orders all old readers
    before the writing body and all new readers after it.
    """
    producer, readers = {}, {}
    for k, phase in enumerate(phases):
        for name in phase.get("writes", ()):
            producer[layouts[name].root] = k
        for name in phase.get("reads", ()):
            readers.setdefault(layouts[name].root, set()).add(k)
        for name in phase.get("internal_mutations", ()):
            # Opaque read/write ownership is also a prior use, even if it has
            # no ordinary result or operand in this phase's public signature.
            readers.setdefault(layouts[name].root, set()).add(k)
    multiplicity = Counter(writes.values())
    # TMA descriptors were rendered before storage planning and bind literal
    # roots. Retain storage when a later body has such an already-bound root;
    # ordinary pointer operands are rebound by the shared declarations below.
    bound_roots = {root for phase in phases for root, _ in
                   (phase.get("matrix_spec") or {}).get("roots", ())
                   if not (PROVIDER_MAPPED_ALIASES and str((phase.get("matrix_spec") or {}).get("engine", ""))
                           .startswith("latch_matrix:pops4b"))}
    # Roots literally bound into rendered tensor-map descriptors keep their storage.
    mapped, literal = set(), set()
    for phase in phases:
        for match in re.finditer(r"(POPS4B_TMAP|P5C_TMAPR?)\(\s*(\w+)\s*,", str(phase.get("code", ""))):
            mapped.add(match.group(2))
            if match.group(1) != "POPS4B_TMAP":
                literal.add(match.group(2))
    if PROVIDER_MAPPED_ALIASES:
        mapped = literal
    selected = {}
    for target, source in writes.items():
        dst, src = layouts[target], layouts[source]
        writer = producer.get(src.root)
        if (target not in inputs or source in inputs or source in bound_roots
                or src.root != source or multiplicity[source] != 1
                or writer is None or not dst.contiguous or not src.contiguous or dst.offset or src.offset
                or dst.shape != src.shape or dtypes[target] != dtypes[source]):
            continue
        if (UNREAD_TARGET_ALIASES and not readers.get(target)
                and not any(layouts[name].root == target for phase in phases for name in phase.get("reads", ()))
                and source not in mapped and ident(source) not in mapped):
            # LATCH-MOE4-PARITY: no phase reads the destination's old value (e.g. a distributed boundary export),
            # so the producing body may store into the destination directly, whatever its kind. Storage and
            # dependencies change; every stored value and expression is unchanged. The commit copy disappears.
            selected[target] = source
            continue
        phase = phases[writer]
        rmw = phase.get("same_index_rmw") if phase.get("kind") == "gemm" else None
        if rmw is not None:
            # LATCH-SELECTIVE-10 accfuse: a product epilogue that reads element (r, c) of the old state and then writes
            # element (r, c) of its result, in the same thread (kernels/ragged_tma_acc.cuh), for a declared pair.
            if rmw.get(source) == target and not any(k > writer for k in readers.get(target, ())):
                selected[target] = source
            continue
        if phase.get("kind") == "cat_cast":
            # LATCH-SELECTIVE-10 dcast (ops.accumulation_rules): a derived copy's refresh reads only its updated
            # pieces, never the old copy, and every old reader precedes it; it stores into the copy directly.
            from . import accumulation_rules
            old_names = {name for name, layout in layouts.items() if layout.root == target}
            if (accumulation_rules.active("dcast") and list(phase.get("writes", ())) == [source]
                    and not old_names & set(phase.get("reads", ())) and not any(k > writer for k in readers.get(target, ()))):
                selected[target] = source
            continue
        epilogue = phase.get("tma_epilogue") if phase.get("kind") == "gemm" else None
        if epilogue is not None:
            # Release step 14: a same-index consumer evaluated in a staged TMA epilogue (group_rules
            # fuse_tma_epilogues). Each element of the staged chunk is loaded and stored by one thread, its lines in
            # the order below, with the stores of a chunk after its loads; the same proof as an elementwise loop.
            if tuple(epilogue.get("shape", ())) != dst.shape or source not in epilogue.get("consumers", ()):
                continue
            lines = epilogue.get("lines", ())
        elif phase.get("kind") != "elementwise" or tuple(phase.get("shape", ())) != dst.shape:
            continue
        else:
            lines = phase.get("body", ())
        if any(k > writer for k in readers.get(target, ())):
            continue
        old_names = {name for name, layout in layouts.items() if layout.root == target}
        same_reads = [layouts[name] for name in phase.get("reads", ()) if name in old_names]
        if any(not layout.contiguous or layout.offset or layout.shape != dst.shape for layout in same_reads):
            continue
        stores = [k for k, line in enumerate(lines) if re.search(
            rf"\b{re.escape(ident(source))}\[i\] = r_{re.escape(source)};", line)]
        if len(stores) != 1:
            continue
        # Generated elementwise lines evaluate their expression before their
        # trailing store. Any later old-state load in this loop is unsafe.
        if any(re.search(rf"\b{re.escape(ident(name))}\[", line)
               for name in old_names for line in lines[stores[0] + 1:]):
            continue
        selected[target] = source
    return selected
