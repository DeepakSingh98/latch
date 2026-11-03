"""Claimed segmented-product tokens of ``segmented_products`` (kernels/claim_tasks.cuh, kernels/bucketing.cuh).

Every rewrite is selected by a structural predicate over an emitted ragged product phase (engine, body call form,
template extents), never by operation names, and is off unless its feature is named in ``segmented_products``; with
none of them named the emitted phases, and therefore the generated source, are unchanged.

* ``bsearch``: claimed right products (single and paired) take ``latch_grp::tflat_claim_h`` / ``tflat2_claim_h``: the
  per-launch task table is built by a one-warp prefix scan and a ticket finds its segment by binary search. N-major B
  tiles with ``BN % 64 != 0`` are staged as whole 64-column chunks (exact shared bytes: ``_claimed_shared``). Tasks,
  their order and every stored bit are unchanged.
* ``scatter`` with ``bsearch``: right products with the row-scatter epilogue (ops/pack_fusion.py,
  ``latch_pack::tflat_claim_scatter``) take ``latch_grp::tflat_claim_scatter_rb``: identical arithmetic, stores and
  trap condition; each result row's unpacked destination and bijection check are resolved once per row into a
  per-warpgroup shared table (``ROW_TABLE_BYTES``) instead of once per 16-byte chunk.
* ``fit``: claimed tiles fit the product's padded area (``claimed_tile``); every output element keeps its whole
  ascending k16 chain in one task, so results are bitwise independent of the tile.
* ``route``: routing counts and the stable order run on every CTA (``latch_grp::route_counts`` / ``route_order``, with
  entry-zeroed words); the first interval keeps the single-CTA body (``route_fallback``). Identical integers.
* ``packsum``: an FP32 elementwise sum whose only reader is a routing-pack permutation fixed-k sum is evaluated in that
  sum's gathered load and not stored (``fuse_pack_sums``). Bitwise.
"""
from __future__ import annotations

import re as _re

HEADER = "claim_tasks.cuh"
ROUTE_HEADER = "bucketing.cuh"
SCATTER_HEAD = "latch_pack::tflat_claim_scatter<"
ROW_TABLE_BYTES = 2 * 64 * 4          # two warpgroups x 64 rows x int destination


def _feature(name):
    from .ragged_experts_lower import _feature as feature
    return feature(name)


# Claimed-product tile fit by padded MMA area (feature "fit").
#   Outer (weight-gradient) products take the candidate with the strictly smallest padded area; ties prefer smaller
#     tiles (more parallel k chains) for long segments (mean rows >= LONG_CHAIN_ROWS) and larger tiles (fewer operand
#     re-reads) otherwise. Candidates keep the body's constraints (64-wide MN-major chunks per warpgroup, MT <= 2) and a
#     ring of 3, 4 or 6 stages that fits shared memory.
#   With "bsearch" (whose bodies stage BN % 64 != 0 as whole 64-column chunks), N-major right products with n <= 192
#     choose BN in NMAJOR_BN by padded columns (ties: the wider tile); paired products may then take up to 160 columns
#     (pair_bn_limit).
OUTER_TILES = ((128, 128, 2, 4), (128, 64, 2, 6), (128, 192, 2, 4), (128, 256, 2, 3), (64, 128, 1, 6), (64, 256, 1, 4))
NMAJOR_BN = (64, 96, 128, 160)
LONG_CHAIN_ROWS = 4096       # mean packed rows per segment at which a weight-gradient task's k chain bounds its interval


def pair_bn_limit():
    """Widest column tile paired (flat2) products may take: 160 with "fit" and "bsearch", 128 with exactly one of them,
    None (no limit) with neither."""
    fit, bsearch = _feature("fit"), _feature("bsearch")
    if fit and bsearch:
        return 160
    if fit or bsearch:
        return 128
    return None


_H_CALL = _re.compile(r"latch_grp::(tflat2?)_claim_h<(\d+),(\d+),(\d+),(\d+),(true|false),(\d+)")


def _claimed_shared(code, phase):
    """Exact shared memory of a binary-search claimed right body whose N-major B tile is staged in 64-column chunks."""
    m = _H_CALL.search(code)
    if not m:
        return None
    kind, e, bm, bn, wm, bkc, nst = m.group(1), *map(int, m.group(2, 3, 4, 5)), m.group(6) == "true", int(m.group(7))
    if bkc or bn % 64 == 0:
        return None
    p = 2 if kind == "tflat2" else 1
    bnc = -(-bn // 64) * 64
    stage = bm * 128 + p * bnc * 128
    table = (8 * (e + 2) + 127) & ~127
    wtm, wtn = bm // wm, bn if wm == 2 else bn // 2
    sepi = 2 * p * wtm * (wtn * 2 + 16)
    fits = wm == 2 and nst * stage + 128 + table + sepi <= 220 * 1024
    return nst * stage + 1024 + 8 * (e + 2) + (128 + table + sepi if fits else 0)


def _area(m, n, bm, bn):
    return -(-m // bm) * bm * (-(-n // bn) * bn)


def claimed_tile(*, outer, kmajor, m, n, current, rows=None, groups=None):
    """(BM, BN, WM, stages) of a claimed product under feature "fit"; ``current`` when "fit" is not selected."""
    bm, bn, wm, stages = current
    if not _feature("fit"):
        return current
    if outer:
        long_chain = rows is not None and groups and rows / groups >= LONG_CHAIN_ROWS
        sign = 1 if long_chain else -1
        here = _area(m, n, bm, bn)
        pool = [c for c in OUTER_TILES if _area(m, n, c[0], c[1]) < here]
        if pool:
            return min(pool, key=lambda c: (_area(m, n, c[0], c[1]), sign * c[0] * c[1]))
        return current
    if not kmajor and (bm, wm) == (128, 2) and _feature("bsearch") and n <= 192:
        # Narrow widths only; wider products keep their tile.
        pool = [c for c in NMAJOR_BN if _area(1, n, 1, c) < _area(1, n, 1, bn)]
        if pool:
            return bm, min(pool, key=lambda c: (_area(1, n, 1, c), -c)), wm, stages
        return current
    return current


# Feature "packsum": routing-pack gradients read their summed products directly. In the backward of a pack
# rows = embedding(order // k, x) feeding several grouped products, the products' input gradients are summed by one
# elementwise phase E whose only reader is the pack's permutation fixed-k sum P. When E's result is FP32 (a widened sum
# of BF16 products), E's per-element body is evaluated inside P's gathered load instead (same expressions, same
# operands, same rounding; P's ordered sum is unchanged), so E's result is neither written nor re-read and E's phase
# disappears. Structural conditions only (single reader, whole contiguous operands, the canonical vectorized load
# form); anything else is left unchanged.
_BODY_LINE = _re.compile(r"^(const (\w+) (r_\w+) = (.*?);)(?: (\S+) = \3;)?$")


def _whole(layouts, name, numel):
    lay = layouts.get(name)
    return (lay is not None and lay.root == name and lay.contiguous and not lay.offset
            and int(getattr(lay, "numel", -1)) == numel)


def fuse_pack_sums(phases, layouts, dtypes, ident, kept):
    """Returns ([(E names, P names)], removed phase indices, removed value names)."""
    if not _feature("packsum"):
        return [], set(), set()
    readers, writer = {}, {}
    for k, p in enumerate(phases):
        for r in p.get("reads", ()):
            readers.setdefault(layouts[r].root if r in layouts else r, set()).add(k)
        for w in p.get("writes", ()):
            writer.setdefault(w, k)
    fused, remove, removed = [], set(), set()
    for pk, P in enumerate(phases):
        if P.get("kind") != "embedding_gradient" or P.get("engine") not in ("permutation_fixed_sum_v1",
                                                                             "permutation_gather_v1"):
            continue
        code, G = P.get("code", ""), P["reads"][0]
        ek = writer.get(G)
        if ek is None or ek in remove or ek == pk:
            continue
        E = phases[ek]
        body = E.get("body") or []
        if (E.get("kind") != "elementwise" or not body or len(body) != len(E["names"]) or E.get("vprod_inline")
                or E.get("active_values") or G in kept or readers.get(G) != {pk}):
            continue
        numel = int(getattr(layouts[G], "numel", -1))
        if not all(_whole(layouts, w, numel) for w in E["writes"]) or not all(_whole(layouts, r, numel) for r in E["reads"]):
            continue
        if any(w != G and (w in kept or readers.get(w, set()) - {ek}) for w in E["writes"]):
            continue
        ins = [r for r in E["reads"] if r not in E["writes"]]
        if not ins or any(dtypes.get(r) not in ("bfloat16", "float32") for r in ins):
            continue
        if dtypes.get(G) != "float32":
            continue                                       # only sums whose eliminated intermediate is FP32
        ig = _re.escape(ident(G))
        pat = _re.compile(r"v_\[s_\] = reinterpret_cast<const float4\*>\(" + ig + r" \+ \((.+?)\)\)\[col\];")
        found = pat.findall(code)
        if len(found) != 1:
            continue
        lines, ok = [], True
        g_reg = None
        for line in body:
            m = _BODY_LINE.match(line.strip())
            if not m:
                ok = False
                break
            lines.append(m.group(1))
            if m.group(3) == "r_" + G:
                g_reg = m.group(3)
        if not ok or g_reg is None:
            continue
        loads, elem = [], {}
        for j, r in enumerate(ins):
            if dtypes[r] == "bfloat16":
                loads.append(f"const uint2 in{j}_ = reinterpret_cast<const uint2*>({ident(r)} + rb_)[col];")
                elem[r] = [f"__ushort_as_bfloat16((unsigned short)(in{j}_.{'xy'[c // 2]}{' & 0xffffu' if c % 2 == 0 else ' >> 16'}))"
                           for c in range(4)]
            else:
                loads.append(f"const float4 in{j}_ = reinterpret_cast<const float4*>({ident(r)} + rb_)[col];")
                elem[r] = [f"in{j}_.{'xyzw'[c]}" for c in range(4)]
        parts = []
        for c in range(4):
            sub = []
            for line in lines:
                for r in ins:
                    for form in (f"{ident(r)}[(i)]", f"{ident(r)}[i]"):
                        line = line.replace(form, f"({elem[r][c]})")
                sub.append(line)
            text = " ".join(sub)
            if any(ident(r) + "[" in text for r in [*ins, *E["writes"]]):
                ok = False
                break
            parts.append(f"{{ {text} t4_.{'xyzw'[c]} = {g_reg}; }}")
        if not ok:
            continue
        block = ("{ const long long rb_ = (" + found[0] + "); " + " ".join(loads) + " float4 t4_; " + " ".join(parts)
                 + " v_[s_] = t4_; }")
        P["code"] = pat.sub(lambda _m: block, code, count=1)
        P["reads"] = list(dict.fromkeys(ins + list(P["reads"][1:])))
        P["fused_pack_sum"] = list(E["names"])
        P["arithmetic"] = P.get("arithmetic", "") + "; the summed input gradients' elementwise body evaluated at each gathered row (same expressions)"
        remove.add(ek)
        removed.update(E["writes"])
        fused.append((list(E["names"]), list(P["names"])))
    return fused, remove, removed


def rewrite(emitted):
    """Rewrite emitted ragged phases in place; returns the headers they now need (claim_tasks.cuh includes
    pack_fusion.cuh, so its chain is staged with it even in programs without row-scatter products)."""
    headers = _rewrite(emitted)
    chain = ["pops4b_tma.cuh", "ragged_tma.cuh", "ragged_claim.cuh", "pack_fusion.cuh"] if HEADER in headers else []
    return [*chain, *headers]


_COUNTS = _re.compile(r"^latch_ragged::counts<(\d+)>\((.*), smem\);$")
_ORDER = _re.compile(r"^latch_ragged::(?:order|order_scan)<(\d+)>\((.*), smem\);$")
ROUTE_ORDER_SHARED = 4 * (256 + 8 * 256 + 256) + 64     # bases, per-warp counts, round totals


def _rewrite(emitted):
    headers = []
    if _feature("route"):
        # Routing preparation (route_order's counts and stable order) on every CTA; same integers. The engine name is
        # kept (artifact inspection keys on it); route_rows marks the body.
        for phase in emitted:
            if phase.get("engine") != "stable_integer_bucket_order":
                continue
            code = phase.get("code", "").strip()
            mc, mo = _COUNTS.match(code), _ORDER.match(code)
            if phase.get("kind") == "ragged_counts" and mc:
                e = int(mc.group(1))
                # Whole 128-byte lines (32 words): later claimed products' ticket counters keep their line alignment.
                phase["code"] = f"latch_grp::route_counts<{e}>({mc.group(2)}, smem, /*GSKC:{-(-(e + 1) // 32) * 32}*/);"
            elif phase.get("kind") == "ragged_order" and mo:
                e = int(mo.group(1))
                phase["code"] = f"latch_grp::route_order<{e}>({mo.group(2)}, smem, /*GSKC:32*/);"
                phase["shared_bytes"] = max(int(phase.get("shared_bytes", 0)), ROUTE_ORDER_SHARED)
            else:
                continue
            phase["route_rows"] = True
            phase["route_fallback"] = code                   # first interval (no entry-zeroed words): original body
            phase["arithmetic"] = phase.get("arithmetic", "") + "; every CTA counts / places a row chunk (same integers)"
            if ROUTE_HEADER not in headers:
                headers.append(ROUTE_HEADER)
    if _feature("bsearch"):
        # Claimed right products (single, paired) look up their ticket's segment by binary search.
        for phase in emitted:
            code = phase.get("code", "")
            if phase.get("kind") != "gemm" or not phase.get("flat_claim"):
                continue
            new = (code.replace("latch_ragged::tflat_claim<", "latch_grp::tflat_claim_h<")
                   .replace("latch_ragged::tflat2_claim<", "latch_grp::tflat2_claim_h<"))
            if new != code:
                phase["code"] = new
                phase["task_bsearch"] = True
                exact = _claimed_shared(new, phase)
                if exact is not None:
                    phase["shared_bytes"] = exact
                if HEADER not in headers:
                    headers.append(HEADER)
        # Row-scatter products take the binary-search body with a per-row destination table.
        for phase in emitted:
            code = phase.get("code", "")
            if (phase.get("kind") == "gemm" and phase.get("pack_scatter")
                    and phase.get("engine") == "ragged_flat_tma_claim_scatter_wgmma" and code.count(SCATTER_HEAD) == 1):
                phase["code"] = code.replace(SCATTER_HEAD, "latch_grp::tflat_claim_scatter_rb<")
                phase["shared_bytes"] = int(phase.get("shared_bytes", 0)) + ROW_TABLE_BYTES
                phase["engine"] = "ragged_flat_tma_claim_scatter_rows_wgmma"
                phase["arithmetic"] = phase.get("arithmetic", "") + (
                    "; each row's destination and bijection check resolved once per row (same bits)")
                if HEADER not in headers:
                    headers.append(HEADER)
    return headers
