"""Device-unit isolation for ``device_unit_inline='isolate_all'``: keep auxiliary calls out of warpgroup-MMA pieces.

Under relocatable (-rdc) device-unit compilation, ptxas serializes every ``wgmma.mma_async`` of a function that
contains any call (C7509: "presence of Extern calls"): each HGMMA is followed by a full ``WARPGROUP.DEPBAR``. The
calls come from code that has nothing to do with the products in the same publication interval: the compact
codegen's ``__noinline__`` elementwise/full-sum helpers, the CUDA runtime's 64-bit division subroutine, and so on.
``step17_intervals.split_interval`` already moves each inline-wgmma phase of such an interval into a function piece
of its own (``device_unit_inline='isolate'``), but only

  * for phases matching an explicit isolate pattern (the program's inline-wgmma engine entry points), and
  * when the interval's prelude is made only of pure re-derivations it repeats in every piece.

``'isolate_all'`` widens both, structurally and without changing any generated arithmetic:

  * ``patterns(source)``: every entry-resident warpgroup-MMA engine form the program emits (namespace-qualified
    ``*_phase_inline`` / ``*_split_phase_inline`` templates whose headers issue wgmma, the P5C / P8WS phase bodies),
    added only when the program contains them, so other programs keep their unit sources and cache identity;
  * the view sentinels: prelude declarations of operand views at a constant element offset from an arena/argument
    view (``reinterpret_cast<T*>(latch_pr + A) + B`` and ``latch_pa[i] + B``) are the same pure pointer
    re-derivations.

Every piece is called at the original place in the original order, inside the same interval; pieces of one interval
are separated by the interval's own ``__syncthreads()`` (unchanged); the closing grid barrier and tick stanza stay in
the outer interval function. Address arithmetic, memory visibility, synchronization and dependency order are
unchanged; only function boundaries move.
"""
from __future__ import annotations

import re

# Entry-resident warpgroup-MMA engine forms (their headers issue wgmma inside the inlined body).
ENGINES = (
    r"\b(?:pops4b|p7a4b)::(?:tma_phase|tma_split_phase|wmm_phase)_inline<",
    r"\bp5c::phase(?:_k|_sk|_chain|_epi\w*)?<",
    r"\bp8ws::(?:phase_mode|group|item\w*)<",
    r"\bp9ws::(?:group|item|bitem)\w*<",
)


SENTINEL = r"(?!)arg_offset_views"   # never matches code; tells the splitter to accept uncast argument offset views


def patterns(source):
    """Isolate patterns this program needs (only those it emits), plus the offset-view sentinels."""
    from .matrix_bodies import VIEW_SENTINEL
    found = tuple(p for p in ENGINES if re.search(p, source))
    return found + (VIEW_SENTINEL, SENTINEL)
