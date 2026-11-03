"""Short-K BF16 outer products evaluated at their readers instead of stored (matrix_bodies ``shortk``).

A product ``[T, K] x [K, D]`` with K <= 2 (for example the gradient of a [T, 1] GEMV with respect to its [T, K] operand)
is lowered by the skinny-K body, which computes every output element as

    acc = +0.0f; for z ascending: acc = fmaf(widen(A[row, z]), widen(B[z, col]), acc); out = __float2bfloat16_rn(acc)

and stores the [T, D] BF16 result, which its single elementwise / row-group reader then loads and widens once. This
pass substitutes the producer's own element expression at each identity read of its reader,

    v_out[(i)]  ->  __float2bfloat16_rn(fmaf(a1, b1, fmaf(a0, b0, 0.0f)))      (K = 2; K = 1 keeps one fmaf)

with ``az = __bfloat162float(A[row * sam + z * sak])`` and ``bz = __bfloat162float(B[z * sbk + col * sbn])``,
row = i / D and col = i % D. The reader therefore sees exactly the BF16 value it loaded before (same fmaf chain from
+0 in ascending z, same single RN rounding, programs compiled with --fmad=false and no FTZ), and its own arithmetic is
unchanged. The [T, D] array is never written or read; the reader instead reads one A value per row (lane-invariant)
and the D-element B row (one 8-byte vector per four lanes in the float4 bodies when B is 8-byte aligned arena storage,
else scalar loads), both cache resident.

Admission (otherwise the product stays stored, with the refusal recorded):

* the producer is a skinny-K product of the ``skinny`` bodies (``skinny_outer`` metadata: shapes, strides and operand
  identities) with K <= KMAX, BF16 operands and result, numel < 2**31;
* its result is its own contiguous offset-0 root, not returned, committed, placed or written in place;
* it has at most MAX_READERS reader (recomputation is never duplicated), an elementwise body or a row group of the same
  extent that reads the value directly (not through a view), and every textual occurrence of the value is an identity
  read;
* no phase between the producer and the reader writes the root of an operand, and the reader writes neither an operand
  root nor the result root.

The pass runs at the same point as the indexed-backward placements at readers (after row fusion and the existing exact
rewrites; before store elision, vector code generation, barrier scheduling and arena assignment), so the new reads
are planned like any other (operand lifetimes extend to the reader). Without ``shortk`` every program is unchanged.
"""
from __future__ import annotations

import re

VERSION = 1
KMAX = 2               # largest admitted K
MAX_READERS = 1        # readers per admitted product

# Scalar element of B's column ``(I) % N`` at element offset OFF (unit column stride). Vector bodies rewrite each lane's
# call (vector_lanes) into one 8-byte load per four lanes; every other body uses this scalar form.
HELPERS = r"""// Low-K BF16 outer products at readers (ops/outer_readers.py)
#ifndef LATCH_OUT_COL
#define LATCH_OUT_COL
#define latch_out_col(P, OFF, N, I) ((P)[(long long)((unsigned)(I) % (N)) + (OFF)])
#endif
"""


def enabled():
    from .matrix_bodies import active
    return active("shortk")


def element_expression(o, column_vectors):
    """BF16 element expression of the product at flat index ``(i)`` (the skinny-K arithmetic).

    o: skinny_outer metadata. column_vectors: B rows may be read as 8-byte vectors in float4 bodies."""
    m, n, k = int(o["m"]), int(o["n"]), int(o["k"])
    sam, sak, sbk, sbn = (int(o[x]) for x in ("sam", "sak", "sbk", "sbn"))
    row = f"(long long)((unsigned)(i) / {n}u)"
    acc = "0.0f"
    for z in range(k):
        a = f"__bfloat162float({o['a']}[{row} * {sam}LL + {z * sak}LL])"
        if column_vectors:
            b = f"__bfloat162float(latch_out_col({o['b']}, {z * sbk}LL, {n}u, (i)))"
        else:
            b = f"__bfloat162float({o['b']}[(long long)((unsigned)(i) % {n}u) * {sbn}LL + {z * sbk}LL])"
        acc = f"fmaf({a}, {b}, {acc})"
    return f"__float2bfloat16_rn({acc})"


_LANE = re.compile(r"__bfloat162float\(latch_out_col\((v_\w+), (\d+)LL, (\d+)u, \(\(4 \* (\w+) \+ ([0-3])\)\)\)\)")


def vector_lanes(code):
    """float4 bodies: lane u of index jv reads column (4*jv + u) % N = (4*jv) % N + u (N % 4 == 0), i.e. half u of the
    8-byte vector at element (4*jv) % N; the four lanes' load texts are identical (one load per round). The widened
    value is the float whose bits are the BF16 bits << 16, exactly __bfloat162float."""
    def one(mt):
        p, off, n, jv, u = mt.group(1), mt.group(2), mt.group(3), mt.group(4), int(mt.group(5))
        word = f"reinterpret_cast<const uint2*>({p} + {off}LL)[(unsigned)((4 * {jv})) % {n}u / 4u].{'x' if u < 2 else 'y'}"
        return f"__uint_as_float({word} << 16)" if u % 2 == 0 else f"__uint_as_float({word} & 0xffff0000u)"
    return _LANE.sub(one, code)


def place_at_readers(phases, layouts, *, kept_roots, external_roots, numel, ident, dtypes, retain_all=False):
    """Rewrite admitted products into their readers. Returns (records, dead value names, helper text or None)."""
    if retain_all or not enabled():
        return [], set(), None
    root = lambda n: layouts[n].root if n in layouts else n
    records, drop = [], set()
    by_ident = {}
    for name in layouts:
        by_ident.setdefault(ident(name), name)
    for k, ph in enumerate(phases):
        o = ph.get("skinny_outer")
        if ph.get("kind") != "gemm" or not o:
            continue
        v = ph["writes"][0] if len(ph.get("writes", ())) == 1 else None
        lay = layouts.get(v) if v else None
        refuse = None
        a_name, b_name = by_ident.get(o.get("a")), by_ident.get(o.get("b"))
        if v is None or lay is None:
            refuse = "not a single-result product"
        elif ident(v) != o.get("out"):
            refuse = "result identity differs"
        elif int(o["k"]) > KMAX:
            refuse = f"K {o['k']} > {KMAX}"
        elif a_name is None or b_name is None or not {a_name, b_name} <= set(ph.get("reads", ())):
            refuse = "operand identities not among the product reads"
        elif any(dtypes.get(x) != "bfloat16" for x in (v, a_name, b_name)):
            refuse = "non-BF16 operand or result"
        elif lay.root != v or not lay.contiguous or lay.offset or v in kept_roots:
            refuse = "kept or non-root result"
        elif lay.numel >= (1 << 31) or tuple(lay.shape) != (int(o["m"]), int(o["n"])):
            refuse = "result extent"
        readers = [j for j, q in enumerate(phases) if j != k and any(root(r) == v for r in q["reads"])]
        if refuse is None and not readers:
            refuse = "no reader"
        if refuse is None and len(readers) > MAX_READERS:
            refuse = f"{len(readers)} readers > {MAX_READERS}"
        operands = [a_name, b_name]
        operand_roots = {root(r) for r in operands}
        name_re = re.compile(rf"(?<![\w.]){re.escape(ident(v))}(?!\w)")
        identity_re = re.compile(rf"(?<![\w.]){re.escape(ident(v))}\[\s*\(?i\)?\s*\]")
        for j in readers:
            if refuse:
                break
            q = phases[j]
            if j < k:
                refuse = "reader before producer"
            elif any(root(r) == v and r != v for r in q["reads"]):
                refuse = "read through a view"
            elif q["kind"] not in ("elementwise", "row_fused") or q.get("vprod_inline") or (
                    q["kind"] == "elementwise" and not q.get("body")):
                refuse = "reader kind " + q["kind"]
            elif q["kind"] == "elementwise" and numel(q["shape"]) != lay.numel:
                refuse = "elementwise reader of another extent"
            elif q["kind"] == "row_fused" and int((q.get("row_fusion") or {}).get("rows", 0)) * int(
                    (q.get("row_fusion") or {}).get("width", 0)) != lay.numel:
                refuse = "row group of another extent"
            elif {root(w) for w in q["writes"]} & (operand_roots | {v}):
                refuse = "reader writes an operand or result root"
            elif any(k < w < j for w, wq in enumerate(phases) for nm in wq["writes"] if root(nm) in operand_roots):
                refuse = "operand rewritten before a reader"
            else:
                texts = [q[key] for key in ("code", "code_r1") if isinstance(q.get(key), str)] + list(q.get("body") or [])
                joined = "\n".join(texts)
                total, identity = len(name_re.findall(joined)), len(identity_re.findall(joined))
                if not total or total != identity:
                    refuse = "non-identity read"
        if refuse:
            records.append({"value": v, "engine": ph.get("engine"), "refused": refuse})
            continue
        lb = layouts[b_name]
        vec_ok = (int(o["sbn"]) == 1 and int(o["n"]) % 4 == 0 and int(o["sbk"]) % 4 == 0
                  and lb.offset % 4 == 0 and lb.root not in external_roots)
        expr = element_expression(o, vec_ok)
        for j in readers:
            q = phases[j]
            for key in ("code", "code_r1"):
                if isinstance(q.get(key), str):
                    q[key] = identity_re.sub(lambda _m: f"({expr})", q[key])
            if q.get("body"):
                q["body"] = [identity_re.sub(lambda _m: f"({expr})", line) for line in q["body"]]
            q["reads"] = list(dict.fromkeys([r for r in q["reads"] if root(r) != v] + operands))
            q.pop("row_parts", None)
            q.setdefault("short_k_inputs", []).append(v)
        drop.add(k)
        records.append({"value": v, "engine": ph.get("engine"), "shape": list(lay.shape), "k": int(o["k"]),
                        "elements_not_stored": lay.numel, "readers": [phases[j]["names"][0] for j in readers],
                        "reader_kinds": [phases[j]["kind"] for j in readers], "operands": operands,
                        "column_vectors": bool(vec_ok)})
    dead = {phases[k]["writes"][0] for k in drop}
    if drop:
        phases[:] = [ph for k, ph in enumerate(phases) if k not in drop]
    return records, dead, (HELPERS if drop else None)
