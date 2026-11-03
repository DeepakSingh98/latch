"""Streamed vocabulary-head logits product (``vocab_rules`` ``stream`` and ``views``).

``stream``: the padded vocabulary logits product (a provider ``pops4b_tma`` body, recorded in its phase's
``tma_respec``) is re-emitted as ``kernels/vocab_stream.cuh`` ``latch_vocab_stream::forward``, which keeps each CTA's
128 X rows resident in shared memory and streams W through a TMA ring with the two warpgroups on alternate 128-column
tiles, storing the same BF16 logits. Two forms are emitted:

* statistics-free (``stats_free``): ``forward<4, NK, 0, false, false, 0, 2>`` does no statistics work and leaves its
  64 x 64 result tiles through a TMA-store epilogue (the logits tensor map is appended to ``FArgs``). The row-statistics
  phase is emitted unchanged after it.
* fused statistics (used where the statistics-free form is refused): ``forward<4, NK>`` also writes the row statistics
  ``vocab_stats`` in its epilogue, in the exact per-lane order of the online warp statistics pass
  (``vocab_head._emit_stats_online``). The statistics phase is then not emitted; its readers (the target pick and the
  logits gradient) read the product phase's second output.

Values are bitwise those of the replaced phases (see the header). Selection uses only the recorded product spec/plan
and operand layouts.

``views``: vocabulary-head GEMM phases address offset-view operands inline (``root_views``).
"""
from __future__ import annotations

from .vocab_rules import VOCAB_DYNAMIC_SHARED_BOUND

HEADER = "vocab_stream.cuh"
ENGINE = "latch_vocab_stream:forward_stats"
NST = 4                 # W ring stages
BM = BN = 128


class Refused(Exception):
    pass


def enabled():
    """vocab_rules ``stream``: re-emit the padded logits product on the streamed body."""
    from .vocab_rules import active
    return active("stream")


def shared_bytes(nst=NST):
    """Dynamic shared memory of the fused-statistics form: 1 KiB alignment slack, eight resident 16 KiB X blocks, the
    W ring and the tail chunks."""
    return 1024 + 8 * BM * 128 + nst * BN * 128 + BM * 16


def forward_code(record, *, stats, vocab, budget=None):
    """(code, shared bytes, metadata) re-emitting the recorded logits product with fused row statistics.

    ``stats`` is the statistics value's C identifier and ``vocab`` its logical width (<= padded N)."""
    spec, plan = dict(record.get("spec") or {}), dict(record.get("plan") or {})
    tma = spec.get("tma") or {}
    m, n, k = int(spec.get("m", 0)), int(spec.get("n", 0)), int(spec.get("k", 0))
    if (spec.get("dtype") != "bfloat16" or not spec.get("ak") or not spec.get("bkc") or spec.get("batch", 1) != 1
            or spec.get("ldc") or "a" not in tma or "b" not in tma or not spec.get("out")):
        raise Refused("vocabulary stream: contiguous batch-1 BF16 K-major operands with TMA roots required")
    if m <= 0 or n % BN or not 0 < k <= 512 or k % 8 or not 0 < int(vocab) <= n:
        raise Refused("vocabulary stream: N % 128 == 0, K <= 512 (resident rows), 0 < V <= N")
    sam, sak, sbk, sbn = spec["strides"][:4]
    if sak != 1 or sbk != 1 or (2 * sam) % 16 or (2 * sbn) % 16:
        raise Refused("vocabulary stream: 16-byte aligned K-contiguous rows")
    nst = NST
    need = shared_bytes(nst)
    if (budget is not None and need > int(budget)) or need > VOCAB_DYNAMIC_SHARED_BOUND:
        raise Refused(f"vocabulary stream: {need} shared bytes exceed the {budget}-byte budget")
    (a_root, a_off), (b_root, b_off) = tma["a"], tma["b"]
    nk = -(-k // 64)
    tx = f"POPS4B_TMAP({a_root},{int(a_off)},{k},{m},{2 * sam},64,{BM})"
    tw = f"POPS4B_TMAP({b_root},{int(b_off)},{k},{n},{2 * sbn},64,{BN})"
    code = f"""{{
  // Vocab head: logits product + online row statistics (vocab_stream.cuh); replaces the statistics phase.
  const latch_vocab_stream::FArgs g{{{tx}, {tw}, {spec['out']}, {n}LL, {stats}, {m}, {n}, {k}, {int(vocab)}}};
  latch_vocab_stream::forward<{nst}, {nk}>(g, smem);
}}"""
    meta = dict(engine=ENGINE, shared_bytes=need, threads=256,
                vocab_stream=dict(version=1, body=1, stages=nst, diagnostic=0, state="registers", tile=[BM, BN], k_blocks=nk, resident_rows=BM,
                                  vocabulary=int(vocab), padded=n, rows=m, replaced_plan=plan,
                                  row_blocks=-(-m // BM), column_tiles=n // BN),
                arithmetic=("wgmma.mma_async m64n128k16; FP32 accumulation from +0, ascending k16 steps over the "
                            "pops4b_tma zero-filled 64-wide k blocks; one round-to-nearest BF16 rounding. Row "
                            "statistics in the epilogue: exact row max and the online warp pass's per-lane "
                            "rescaled exp-sums in ascending chunk order, lane 0's partial chunk, xor-butterfly merge"))
    return code, need, meta


# ------------------------------------------------------------------------------- statistics-free streamed product
def stats_free(code, logits_layout):
    """Turn ``forward_code``'s call (emitted with ``stats="nullptr"``) into the statistics-free form.

    The call becomes ``forward<4, NK, 0, false, false, 0, 2>`` (STATS = false: the bare streamed logits product, no
    statistics output; the online statistics phase is kept), whose epilogue stages each warpgroup's rounded 64 x 64
    quarters in shared memory and stores them with TMA bulk tensor stores. The logits tensor map (contiguous
    [R, Vp] logits, 64 x 64 box) is appended to ``FArgs``. Same values, same rounding."""
    import re
    new, n = re.subn(r"latch_vocab_stream::forward<(\d+), (\d+)>\(",
                     r"latch_vocab_stream::forward<4, \2, 0, false, false, 0, 2>(", code)
    if n != 1:
        raise Refused("statistics-free vocabulary stream: unexpected forward call form")
    lay = logits_layout
    if lay is None or len(lay.shape) != 2 or tuple(lay.strides) != (lay.shape[1], 1):
        raise Refused("statistics-free vocabulary stream: contiguous [R, Vp] logits required")
    rows, vp = lay.shape
    tc = f"POPS4B_TMAP({lay.root},{lay.offset},{vp},{rows},{2 * vp},64,64)"
    new, n = re.subn(r"(latch_vocab_stream::FArgs g\{[^;]*?)\};", lambda m: m.group(1) + ", " + tc + "};", new, count=1)
    if n != 1:
        raise Refused("statistics-free vocabulary stream: unexpected argument form")
    return new


def stats_free_shared(recorded):
    """Dynamic shared bytes of the statistics-free form (``recorded`` is the fused form's figure, not used): 1 KiB
    alignment slack, eight resident 16 KiB X blocks, the 4-stage W ring and two 8 KiB store-staging quarters; no
    tail-chunk buffer. 214,016 bytes."""
    return 1024 + 8 * 128 * 128 + 4 * 128 * 128 + 2 * 8192


# -------------------------------------------------------------------------------- splittable product intervals
def split_enabled():
    """vocab_rules ``views``: vocabulary-head product phases address offset views inline."""
    from .vocab_rules import active
    return active("views")


def root_views(phases, layouts, ident):
    """Vocabulary-head product phases read offset views (for example a serial weight-gradient chunk's operand views
    `latch_pr + root + offset`); such declarations keep step17 from splitting the interval into per-phase pieces, so
    an unrelated call in the interval (e.g. a 64-bit loop-count division) serializes every wgmma in it (C7509). Here
    each such view is addressed as `(root + offset)` inside the phase code and the phase reads the root instead: the
    same address, the same tensor maps (bound separately), no runtime work. Returns the number of rewritten views."""
    import re
    if not split_enabled():
        return 0
    count = 0
    for phase in phases:
        code = phase.get("code") or ""
        if phase.get("kind") != "gemm" or not any("vocab_" in str(x) for x in (*phase.get("names", ()),
                                                                                 *phase.get("reads", ()))):
            continue
        reads = list(phase.get("reads", ()))
        for i, value in enumerate(reads):
            layout = layouts.get(value)
            if layout is None or layout.root == value or not layout.offset:
                continue
            pattern = r"\b" + re.escape(ident(value)) + r"\b"
            if not re.search(pattern, code):
                continue
            code = re.sub(pattern, f"({ident(layout.root)} + {int(layout.offset)}LL)", code)
            reads[i] = layout.root
            count += 1
        phase["code"] = code
        phase["reads"] = list(dict.fromkeys(reads))
    return count
