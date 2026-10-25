"""Opt-in non-expert matrix body rules (LATCH-TRAIN-MATMUL).

Selected through the normal build API: ``CapturedModule.build(..., matrix_bodies="tailrot+taildeep")`` (a '+'-joined
subset of RULES, or the preset "exact" = tailrot+taildeep+fill+p5c; "isolate" is opt-in only). Omitting it (None) leaves lowering and generated source
unchanged. Independent of ``matrix_schedule`` (both may be given). Every rule is structural: it reads product shapes,
layouts, tile counts and the program's shared-memory limit, never a model, module or parameter name.

* ``tailrot``: native mma.sync tail products (``popsc::bmm_tail``: M, N, K multiples of 16 but not all of 32, e.g. a
  16-row router weight gradient) that a single-member matrix phase emits on the whole grid start on the interval's
  rotated CTA offset, after the tiles of the interval's earlier rotated matrix products, exactly as the release step-14
  rotation already places provider TMA products. Before, several such products of one dependency interval all started
  on CTA 0 and ran back to back on the same few CTAs while the rest of the grid idled. Placement only: every tile is
  computed whole by one CTA with unchanged arithmetic (bitwise identical results).
* ``taildeep``: a long-contraction tail product (at least 32 64-wide k steps) keeps more operand bytes in flight. The
  same body (``popsc::bmm_tail``) and tile shape run with the k-stage width (64/128/256) and ring depth (<= 8) that
  maximize the ring's bytes in flight within the program's shared-memory limit. The product is latency bound on its
  operand loads (a 32 x 32 tile streams two 64-wide k stages per ~0.6 us round trip with the default 4-stage ring).
  Every element still owns one FP32 accumulator that receives one mma.sync.m16n8k16 product per k16 block in
  ascending k from +0 and is rounded once, so results are bitwise identical.
* ``fill``: in a dependency interval whose rotated matrix products are long-contraction tasks on at most half of the
  grid (e.g. the K partitions of a weight gradient) followed by one wide TMA product (at least one tile per CTA), the
  wide product no longer starts its round-robin tiles on the CTAs that still run the long tasks. With at least NST
  64-wide k steps per tile it claims its tiles from a per-launch counter (the existing release step-14 claim mode,
  one task ahead) so CTAs that finish early take more; otherwise it runs on the CTA slice the long tasks leave idle.
  The long tasks keep their rotated placement and run first: in any interval that also holds a wide product they
  are emitted ahead of the interval's other rotated products (so they take the first rotated CTAs instead of
  landing on CTAs that already carry wide-product tiles). Each tile is still computed whole by one CTA with
  unchanged arithmetic (bitwise identical results); only which CTA computes it changes.
* ``p5c``: a provider TMA product left with the plain whole-grid call (no fused epilogue, contiguous batch-1 BF16
  result, no serial/split/segment contract) is re-emitted on the existing P-OPS5-C body (``p5c::phase``, the same
  m64nNk16 wgmma chains from +0 in ascending k16 steps and one BF16 rounding: the same bitwise class, results bit for
  bit unchanged) when its shape is one the step-13 rule routed to the TMA body only because the frozen flat P5C
  admission wants >= 8 k steps and half a grid of tiles: short contractions (2..7 64-wide k steps) with at least one
  128 x 128 (or 128 x 192 for N = 192) tile per CTA take a 6-stage (4 for 128 x 192) ring; long contractions (>= 32
  k steps) on at most half a grid of tiles take 128 x 64 P5C tiles with a 6-stage ring (64 x 128, one warpgroup
  per 64-row slab, 8 stages, when M is a multiple of 64 but not of 128); where an interval's long tasks would need
  more tiles than the grid has CTAs (a CTA would run two of them back to back), the largest 128 x 64 partitions take
  128 x 128 tiles until they fit. (The warp-specialized P8-A body is 32% faster in isolation but slower inside the
  program, where many partitions share L2: not used.) Store-bound short-K products (the step-13 rule's p8ws choice) take
  P5C 256 x 128 (else 128 x 256) tiles with per-stage release. Rings never exceed RING_LIMIT. Measured on the
  interval replay of the real programs (projection products 18-33% faster, weight-gradient partitions 10-29%,
  readout logits 11%).
* ``isolate``: the device-unit build (``device_unit_inline="isolate"``) already moves each inline-wgmma phase of an
  outlined interval into a call-free function piece, because under relocatable compilation any call in a function
  makes ptxas serialize its warpgroup MMA (C7509). With this rule the P5C / P8WS bodies count as such phases too, and
  an interval whose prelude declares operand views at a constant offset from an arena view
  (``reinterpret_cast<T*>(latch_pr + A) + B``, e.g. the K partitions of a serial-BF16 weight gradient) is accepted as
  splittable: those statements are the same pure pointer re-derivations each piece repeats. Function boundaries
  only; generated arithmetic unchanged (bitwise). The release build serialized the readout dW/dX interval and every
  router-gradient interval this way.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

RULES = ("tailrot", "taildeep", "fill", "p5c", "isolate")
# "isolate" (device-unit outlining of the P5C/p8ws phases) stays opt-in: it removed the ptxas serialization record
# but did not shorten the measured update (mb6b vs mb6a).
PRESETS = {"exact": "tailrot+taildeep+fill+p5c"}
_BODIES = ContextVar("latch_matrix_bodies", default=frozenset())
VERSION = 1


def rules_of(value):
    if value is None:
        return frozenset()
    if not isinstance(value, str) or not value:
        raise ValueError("matrix_bodies must be None, a preset or a '+'-joined rule string")
    value = PRESETS.get(value, value)
    names = value.split("+")
    unknown = [n for n in names if n not in RULES]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"matrix_bodies rules must be distinct names from {RULES} (or a preset {sorted(PRESETS)}); "
                         f"got {value!r}")
    return frozenset(names)


def validate(value):
    rules_of(value)
    return value


def active(name):
    if name not in RULES:
        raise ValueError(name)
    return name in _BODIES.get()


@contextmanager
def scope(value):
    token = _BODIES.set(rules_of(value))
    try:
        yield
    finally:
        _BODIES.reset(token)


# ----------------------------------------------------------------------------------------------- taildeep
DEEP_MIN_STEPS = 32          # long contractions only: >= 32 64-wide k steps
DEEP_MAX_STAGES = 8          # cp.async groups pending in the ring (cp.async.wait_group immediate <= 7)
DEEP_WIDTHS = (64, 128, 256)


def deepen_tail(plan, m, n, k, ak, bkc, *, stage_elements, shared_limit):
    """The tail plan with the k-stage width / ring depth of the most bytes in flight (ties: narrower stages).

    Bitwise: only the operand staging granularity changes (k16 blocks are consumed in the same ascending order by the
    same warps; absent k beyond K is staged as +0 exactly as before). Returns the original plan when not admitted."""
    if not active("taildeep") or k < 64 * DEEP_MIN_STEPS:
        return plan
    best = None
    for ks in DEEP_WIDTHS:
        if k % ks:
            continue
        stage = 2 * stage_elements(plan["bm"], plan["bn"], ak, bkc, ks)
        nk = k // ks
        nst = min(DEEP_MAX_STAGES, nk, shared_limit // stage)
        if nst < 3:
            continue
        key = ((nst - 1) * stage, -ks)
        if best is None or key > best[0]:
            best = (key, ks, nst, stage)
    if best is None:
        return plan
    _, ks, nst, stage = best
    if (ks, nst) == (plan["ks"], plan["nst"]):
        return plan
    return dict(plan, ks=ks, nst=nst, shared_bytes=nst * stage,
                taildeep={"from": [plan["ks"], plan["nst"]], "in_flight_bytes": (nst - 1) * stage})


# ----------------------------------------------------------------------------------------------- tailrot
TAIL_VBASE = "/*T_VBASE*/"   # replaced by the interval's rotated CTA offset (lower.phase_code); 0 elsewhere


def tail_rotation_token(spec):
    """The rot argument of a single-member native tail product phase (a token under 'tailrot', else 0)."""
    if active("tailrot") and spec.get("tail") and spec.get("dtype") == "bfloat16" and not spec.get("active"):
        return TAIL_VBASE
    return 0


# ----------------------------------------------------------------------------------------------- fill
_ROTATED = ("/*VBASE*/", "/*B_VBASE*/", "/*C_VBASE*/", TAIL_VBASE)


def fill_plan(phases, members, *, grid):
    """None, or {"wide": phase, "fixed": [phases], "fixed_tiles": F, "mode": "claim" | "slice"} for one interval."""
    import re
    if not active("fill"):
        return None

    def code(i):
        return phases[i].get("code") or ""

    def tiles(i):
        return int(phases[i].get("tiles", 0) or 0)

    def k_of(i):
        shape = phases[i].get("matrix_shape") or ()
        return int(shape[2]) if len(shape) == 3 else 0

    rotated = [i for i in members if phases[i]["kind"] == "gemm" and any(t in code(i) for t in _ROTATED)]
    wide = [i for i in rotated if tiles(i) >= grid and "g.claim" not in code(i) and (
            (code(i).count("tma_phase_inline<") == 1 and "pops4b::TArgs g{" in code(i))
            or (code(i).count("p5c::phase<") == 1 and "p5c::Args g{" in code(i)))]
    if len(wide) != 1:
        return None
    fixed = [i for i in rotated if i != wide[0]]
    fixed_tiles = sum(tiles(i) for i in fixed)
    if not fixed or not 0 < fixed_tiles <= grid // 2:
        return None
    if any(k_of(i) < 64 * DEEP_MIN_STEPS for i in fixed):
        return None
    if members.index(wide[0]) < max(members.index(i) for i in fixed):
        return None                 # the long tasks must take the first rotated CTAs (emitted before the wide product)
    steps = -(-k_of(wide[0]) // 64)
    m = re.search(r"tma_phase_inline<\d+,\d+,\d+,\w+,\w+,(\d+),", code(wide[0]))
    if m is not None:
        mode = "claim" if steps >= max(2, int(m.group(1))) else "slice"
    elif "p5c::phase<" in code(wide[0]):
        mode = "slice"              # the p5c body has no claim mode: the CTAs the long tasks leave idle
    else:
        return None
    return {"wide": wide[0], "fixed": fixed, "fixed_tiles": fixed_tiles, "mode": mode}


def fill_slice(body, fixed_tiles):
    """The wide product's whole-grid rotated call restricted to the CTAs [fixed_tiles, gridDim.x)."""
    import re
    body, count = re.subn(r"((?:tma_phase_inline|p5c::phase)<[^>]*>)\(g, smem, 0, 1, 1, (\d+)\)",
                          rf"\1(g, smem, (int)gridDim.x > {fixed_tiles} ? {fixed_tiles} : 0, (int)gridDim.x, "
                          r"(int)gridDim.x, 0)", body)
    if count != 1:
        raise RuntimeError("matrix_bodies fill: unexpected TMA product call form")
    return body


# ----------------------------------------------------------------------------------------------- p5c
_PLAIN_TMA = None
# Largest operand ring (incl. alignment / staging) a rule may plan: the release program's own largest product ring
# (p8ws 128 x 128 x 4 stages with two staging tiles, P5C 256 x 128 x 4 stages): 197,632 bytes.
RING_LIMIT = 197632


def _long_tile_plan(spec):
    """(bm, bn) the ``p5c`` rule gives a long-contraction partition, or None (not a long few-tile product)."""
    m, n, k = spec["m"], spec["n"], spec["k"]
    if k % 64 or k // 64 < DEEP_MIN_STEPS or m % 64 or n % 64:
        return None
    if m % 128 == 0:
        return (128, 64)
    if n % 128 == 0:
        return (64, 128)
    return None


def coarse_partitions(phases, intervals, *, grid):
    """Phases whose 128 x 64 partition tiles are coarsened to 128 x 128 so that the long tasks of their interval
    fit one per CTA (largest first, until the interval's long tiles fit the grid or nothing can be coarsened)."""
    coarse = set()
    for members in intervals or ():
        long = []
        for i in members:
            rec = phases[i].get("tma_respec")
            if phases[i].get("kind") != "gemm" or not rec:
                continue
            tile = _long_tile_plan(rec["spec"])
            if tile is not None:
                spec = rec["spec"]
                long.append((i, spec["m"], spec["n"], tile))
        total = sum((m // bm) * (n // bn) for _, m, n, (bm, bn) in long)
        for i, m, n, (bm, bn) in sorted(long, key=lambda r: -(r[1] * r[2])):
            if total <= grid:
                break
            if (bm, bn) == (128, 64) and n % 128 == 0:
                coarse.add(i)
                total -= (m // 128) * (n // 64) - (m // 128) * (n // 128)
    return coarse


def respec_p5c(phases, *, ident, grid, intervals=None):
    """(phases, records, headers): plain provider TMA products re-emitted on the P5C body (rule ``p5c``)."""
    import re
    global _PLAIN_TMA
    if not active("p5c"):
        return phases, [], ()
    coarse = coarse_partitions(phases, intervals, grid=grid)
    from .matrix_adapters.bodies import BY_NAME, pinned
    if _PLAIN_TMA is None:
        _PLAIN_TMA = re.compile(r"\{\n  pops4b::TArgs g\{[^\n;]*\};\n"
                                r"  pops4b::tma_phase_inline<[^>]*>\(g, smem, 0, 1, 1, /\*B_VBASE\*/\);\n\}")
    limit = RING_LIMIT
    body = BY_NAME["p5c"]
    records, out = [], list(phases)
    for i, ph in enumerate(phases):
        rec = ph.get("tma_respec")
        if ph.get("kind") != "gemm" or not rec or ph.get("tma_epilogue") or not _PLAIN_TMA.fullmatch(ph.get("code", "")):
            continue
        spec = rec["spec"]
        if (spec.get("dtype") != "bfloat16" or spec.get("batch", 1) != 1 or spec.get("ldc") or spec.get("tma_batch")
                or any(spec.get(key) for key in ("serial_bf16_k", "ksegs", "split", "epi", "rowstats"))):
            continue
        m, n, k, ak, bkc = spec["m"], spec["n"], spec["k"], spec["ak"], spec["bkc"]
        if m % 64 or n % 64 or k % 64:
            continue
        steps = k // 64
        if 2 <= steps < 8 and m % 128 == 0:
            if n % 128 == 0:
                bm, bn, deepest = 128, 128, 6
            elif n % 192 == 0 and n < 256:
                bm, bn, deepest = 128, 192, 4
            else:
                continue
            if (m // bm) * (n // bn) < grid:
                continue                                    # wide products only (at least one tile per CTA)
        elif steps >= DEEP_MIN_STEPS and m % 128 == 0 and (m // 128) * (n // 64) <= grid // 2:
            bm, bn, deepest = 128, 64, 6
            if coarse and i in coarse and n % 128 == 0:
                bm, bn = 128, 128                    # interval holds more long tasks than the grid has CTAs
        elif steps >= DEEP_MIN_STEPS and m % 64 == 0 and n % 128 == 0 and (m // 64) * (n // 128) <= grid // 2:
            bm, bn, deepest = 64, 128, 8            # one warpgroup per 64-row slab (WM = 1)
        else:
            continue
        ws = False
        wm = 1 if bm == 64 else 2
        stage = (128 + 128) * 128 if ws else (bm + bn) * 128     # p8ws ring slots are 128 + BN rows
        staging = 2 * 128 * 128 * 2 if ws else 0                  # p8ws double-buffered epilogue staging
        nst = min(deepest, max(4, min(steps, deepest)), (limit - 1024 - staging) // stage)
        if nst < (4 if ws else 2):
            continue
        p5spec = {key: value for key, value in spec.items() if key not in ("tma", "tma_batch")}
        p5spec.update(engine="p5c", roots=[tuple(spec["tma"]["a"]), tuple(spec["tma"]["b"])])
        if ws:
            p5spec["ws_ok"] = True
        tiles = (m // bm) * (n // bn)
        plan = dict(bm=bm, bn=bn, wm=wm, nst=nst, release=False, out_f32=False, mc=1, tiles=tiles,
                    shared_bytes=nst * stage + staging + 1024, matrix_bodies="p5c",
                    **({"ws": True, "ws_engine": "p8ws_flat"} if ws else {}))
        with pinned(body.module, body.settings) as mod:
            code = mod.code(p5spec, plan, ident, 0, 1, 1, "/*B_VBASE*/")
        new = dict(ph, code=code, tiles=tiles, work_items=tiles, shared_bytes=plan["shared_bytes"],
                   engine="latch_matrix:p8ws" if ws else "latch_matrix:p5c", matrix_plan={key: v for key, v in plan.items()
                                                          if isinstance(v, (int, float, str, bool))})
        new.pop("tma_respec", None)
        if isinstance(new.get("latch_matrix"), dict):
            new["latch_matrix"] = dict(new["latch_matrix"], body="p5c")
        out[i] = new
        records.append({"phase": i, "output": (ph.get("names") or ["?"])[0], "shape": [m, n, k],
                        "from": [rec["plan"].get("bm"), rec["plan"].get("bn"), rec["plan"].get("nst")],
                        "to": [bm, bn, nst]})
    return out, records, tuple(body.headers) if records else ()


def long_first(phases, members, *, grid):
    """Rule ``fill``: an interval's long rotated tasks emitted before its other members (only when it also holds a
    wide rotated product). Phases of one interval are independent; only CTA placement changes."""
    if not active("fill"):
        return members

    def code(i):
        return phases[i].get("code") or ""

    def tiles(i):
        return int(phases[i].get("tiles", 0) or 0)

    def k_of(i):
        shape = phases[i].get("matrix_shape") or ()
        return int(shape[2]) if len(shape) == 3 else 0

    rotated = [i for i in members if phases[i]["kind"] == "gemm" and any(t in code(i) for t in _ROTATED)]
    long = [i for i in rotated if k_of(i) >= 64 * DEEP_MIN_STEPS and tiles(i) <= grid // 2]
    if not long or not any(tiles(i) >= grid for i in rotated if i not in long):
        return members
    return long + [i for i in members if i not in long]


# ----------------------------------------------------------------------------------------------- isolate
VIEW_SENTINEL = r"(?!)matrix_bodies_offset_views"     # never matches code; tells the splitter to accept offset views
ISOLATE = (r"p5c::phase(?:_k|_sk|_epi\w*)?<", r"p8ws::phase_mode<", r"p8ws::group<")


def active_rules():
    """Sorted active rule names (recorded in the program manifest by lower.py), or None."""
    rules = _BODIES.get()
    return sorted(rules) if rules else None


def isolate_patterns(manifest):
    """Extra step-17 isolate patterns for a program lowered with the ``isolate`` rule (empty otherwise)."""
    rules = (manifest.get("group_rules") or {}).get("matrix_bodies") or ()
    return ISOLATE + (VIEW_SENTINEL,) if "isolate" in rules else ()
