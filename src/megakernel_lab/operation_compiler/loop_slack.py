"""LATCH-STEP-06: slack placement of side products into imbalanced levels of a staged loop region (scheduling only).

A product of a staged region whose results are region outputs that no phase of the region reads (in a saved-tape
reverse body: the per-iteration weight-gradient products of ``loop_defer_grads`` tape mode) has no successor inside
the region. The as-soon-as-possible schedule co-phases it with the critical dX products of its own dependency
interval. Here such a product may instead move to a later interval of the region whose modeled per-CTA load leaves
the most room for its tiles: in particular the long-attention levels, whose static item order leaves part of the
grid idle (one item versus two per CTA). The stage lowering then places the moved product's tiles with a
longest-first table on top of that per-CTA load and runs them after the attention items.

Only the interval of a side product changes. Every phase keeps its arithmetic; tiles keep their k order and
rounding; results are bitwise equal.
"""
from __future__ import annotations

ENABLED = True
GRID = 128
REDUCTIONS = True      # exact column reductions of side outputs (norm-gain gradients) move as well
GAIN = 0.8              # move only when the modeled makespan increase drops below this fraction of staying
# Coarse per-item attention costs (us) for the static P9 item order, from the P9-B in-kernel traces: forward item
# setup + tail and per 64-key tile of 128 query rows; backward prologue, key/value item fixed + per 128 x 128 unit,
# and the work-queued dQ items (spread evenly).
FWD_ITEM_US, FWD_TILE_US = 15.0, 1.3
BWD_PRE_US, KV_ITEM_US, KV_UNIT_US, DQ_US = 13.0, 8.0, 7.5, 20.0


def attention_load(phase, grid=GRID):
    """Per-CTA modeled time (us) of a P9 long-attention phase under its static item order, or None."""
    info = phase.get("step06_attention")
    if not info or info.get("engine") != "p9":
        return None
    t, bh, which = int(info["t"]), int(info["bh"]), info["which"]
    load = [0.0] * grid
    if which == "forward":
        blocks = t // 128
        items = blocks * bh
        for c in range(grid):
            for r in range(items // grid + 2):
                j = r * grid + ((grid - 1 - c) if r & 1 else c)
                if j >= items:
                    continue
                qblk = blocks - 1 - j // bh
                load[c] += FWD_ITEM_US + FWD_TILE_US * 2 * (qblk + 1)
        return load
    nkb = t // 128
    items = nkb * bh
    dq = DQ_US * items / (nkb * 48) if bh else 0.0
    for c in range(grid):
        load[c] = BWD_PRE_US + dq
        for r in range(items // grid + 2):
            j = r * grid + ((grid - 1 - c) if r & 1 else c)
            if j >= items:
                continue
            load[c] += KV_ITEM_US + KV_UNIT_US * (nkb - j // bh)
    return load


def phase_load(phase, layouts, footprint, np_cost, grid=GRID):
    load = attention_load(phase, grid)
    if load is not None:
        return load
    fp = min(grid, max(1, footprint(phase, layouts)))
    cost = np_cost(phase, fp)
    return [cost if c < fp else 0.0 for c in range(grid)]


def tile_cost(phase):
    plan = phase.get("matrix_plan") or {}
    tiles = max(1, int(phase.get("tiles", plan.get("tiles", GRID))))
    est = float(plan.get("estimate_us", 8.0))
    waves = max(1, -(-tiles // int(plan.get("grid", GRID))))
    return est / waves, tiles


def _lpt(load, tiles, cost):
    """Makespan after greedily placing `tiles` equal tasks on the least loaded CTAs."""
    import heapq
    heap = [(v, c) for c, v in enumerate(load)]
    heapq.heapify(heap)
    for _ in range(tiles):
        v, c = heapq.heappop(heap)
        heapq.heappush(heap, (v + cost, c))
    return max(v for v, _ in heap)


def place(phases, intervals, layouts, side_outputs, *, footprint, np_cost, grid=GRID):
    """(intervals, records): side products (gemm phases whose writes are all in `side_outputs` and read by no other
    phase) move to the later interval where their modeled makespan increase is smallest."""
    if not ENABLED:
        return intervals, []
    root = lambda n: layouts[n].root if n in layouts else n
    readers = {}
    for k, phase in enumerate(phases):
        for r in phase["reads"]:
            readers.setdefault(root(r), set()).add(k)
    intervals = [list(m) for m in intervals]
    level = {i: k for k, members in enumerate(intervals) for i in members}

    def interval_load(members, skip=None):
        load = [0.0] * grid
        for j in members:
            if j == skip:
                continue
            phase = phases[j]
            if phase["kind"] == "gemm":
                cost, tiles = tile_cost(phase)
                for t in range(tiles):           # rotation-placed products: count evenly from CTA 0
                    load[t % grid] += cost
            else:
                for c, v in enumerate(phase_load(phase, layouts, footprint, np_cost, grid)):
                    load[c] += v
        return load

    records = []
    for i, phase in enumerate(phases):
        product = phase["kind"] == "gemm" and (phase.get("matrix_spec") or {}).get("engine") == "p5c"
        reduction = REDUCTIONS and phase["kind"] == "reduction" and phase.get("engine") == "exact_col_registers"
        if not (product or reduction):
            continue
        writes = list(phase["writes"])
        if not writes or any(w not in side_outputs for w in writes):
            continue
        if any(k != i for w in writes for k in readers.get(root(w), ())):
            continue
        start = level[i]
        here = intervals[start]
        base_here = interval_load(here, skip=i)
        if product:
            cost, tiles = tile_cost(phase)
            stay = _lpt(base_here, tiles, cost) - max(base_here)
        else:
            own = phase_load(phase, layouts, footprint, np_cost, grid)   # fixed CTAs (column blocks from CTA 0)
            stay = max(a + b for a, b in zip(base_here, own)) - max(base_here)
        best = None
        for target in range(start + 1, len(intervals)):
            members = intervals[target]
            if not any(phases[j].get("step06_attention") for j in members):
                continue
            load = interval_load(members)
            inc = ((_lpt(load, tiles, cost) if product else max(a + b for a, b in zip(load, own))) - max(load))
            if best is None or inc < best[0]:
                best = (inc, target)
        if best is None or best[0] >= GAIN * stay:
            continue
        intervals[start].remove(i)
        intervals[best[1]].append(i)
        level[i] = best[1]
        phase["step06_slack"] = True
        records.append({"phase": i, "product": phase["names"][0], "from": start, "to": best[1],
                        "modeled_increase_us": round(best[0], 2), "modeled_stay_us": round(stay, 2)})
    return [sorted(m) for m in intervals if m], records


__all__ = ["place", "attention_load"]
