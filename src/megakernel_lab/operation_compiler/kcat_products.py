"""K-concatenated sibling claimed products (``segmented_products`` feature ``kcat``; default off).

Requires the ``bsearch`` feature (the fused bodies are binary-search claimed bodies) and ``precision_policy=
'practical'``.

When two claimed right products over the same packed rows are only ever added, e.g. the input gradient
``dY_a W_a^T + dY_b W_b^T`` of two sibling projections of the same rows, the lowered graph computes the two products
separately (two BF16 [rows, N] results) and then one elementwise ``aten.add`` reads both back and writes their BF16
sum, which is the only reader of either product. ``fuse`` replaces the pair and the add with ONE claimed product over
the concatenated reduction axis (kernels/claim_tasks.cuh ``tflat_kcat_claim_h``): the same tiles accumulate both
products' k stages into one FP32 accumulator and round once to BF16, written directly to the add's result. Not
materialized: both product results and the add's re-read of them (per pair 3 x rows x N x 2 bytes of traffic).

Arithmetic: the same sum of the same products; one BF16 rounding instead of three (each product's result and the
sum), so not bitwise to the separate form (practical precision allows the regrouping; the fused form is more accurate).

Structural admission only (no model names):
* the add has exactly two operands, each the single-result output of a claimed TMA right product
  (``ragged_flat_tma_claim_wgmma``) whose only reader is the add;
* both products share segment offsets, packed rows, N, tile template and K-contiguous B operands;
* result layouts: same shape, BF16, contiguous; the add's result likewise.
Anything else is left unchanged and recorded as a refusal.
"""
from __future__ import annotations

import re

_CALL = re.compile(
    r"latch_(?:ragged|grp)::tflat_claim(?:_h)?<(?P<tmpl>\d+,\d+,\d+,\d+,(?:true|false),\d+),(?P<s0>/\*XCLAIM_S0\*/(?:true|false))>"
    r"\((?P<ma>POPS4B_TMAP\([^()]*\)), (?P<mb>POPS4B_TMAP\([^()]*\)), g, (?P<off>[^,]+), (?P<rows>\d+), smem, "
    r"/\*XMAT_ROT\*/0, (?P<slot>/\*XCLAIM\*/nullptr)\);")
_INIT = re.compile(r"popsc::BArgs g\{(?P<init>[^;]*)\};")


def enabled():
    from .ragged_experts_lower import _feature
    return _feature("kcat")


def _split(init):
    out, depth, cur = [], 0, ""
    for ch in init:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    out.append(cur.strip())
    return out


def _contiguous(lay):
    shape, strides = list(lay.shape), list(lay.strides)
    step = 1
    for e, s in reversed(list(zip(shape, strides))):
        if e != 1 and s != step:
            return False
        step *= e
    return True


def fuse(phases, layouts, dtypes, ident, kept):
    """Returns (records, removed phase indices, removed value names)."""
    if not enabled():
        return [], set(), set()
    root = lambda n: layouts[n].root if n in layouts else n
    producer = {w: k for k, p in enumerate(phases) for w in p.get("writes", ())}
    readers = {}
    for k, p in enumerate(phases):
        for r in p.get("reads", ()):
            readers.setdefault(root(r), set()).add(k)
    records, remove, removed = [], set(), set()
    for ke, E in enumerate(phases):
        if E.get("kind") != "elementwise" or len(E.get("writes", ())) != 1:
            continue
        reads = list(dict.fromkeys(E.get("reads", ())))
        if len(reads) != 2 or [str(t).removeprefix("torch.ops.") for t in E.get("targets", ())] not in (
                ["aten.add.Tensor"],):
            continue
        a, b = reads
        out = E["writes"][0]
        why = None
        ka, kb = producer.get(a), producer.get(b)
        if ka is None or kb is None or ka in remove or kb in remove or ke in remove:
            continue
        Pa, Pb = phases[ka], phases[kb]
        if any(p.get("engine") != "ragged_flat_tma_claim_wgmma" or p.get("tma_outer") or not p.get("flat_claim")
               or len(p.get("writes", ())) != 1 or p.get("claim_merged_into") is not None for p in (Pa, Pb)):
            continue
        if readers.get(root(a)) != {ke} or readers.get(root(b)) != {ke} or {a, b, out} & set(kept):
            why = "other readers or kept values"
        lays = [layouts.get(x) for x in (a, b, out)]
        if why is None and (any(l is None for l in lays) or any(dtypes.get(x) != "bfloat16" for x in (a, b, out))
                            or len({tuple(l.shape) for l in lays}) != 1 or not all(_contiguous(l) for l in lays)
                            or any(int(l.offset) % 8 for l in lays)):
            why = "result layouts"
        ma, mb = _CALL.search(Pa.get("code", "")), _CALL.search(Pb.get("code", ""))
        ia, ib = _INIT.search(Pa.get("code", "")), _INIT.search(Pb.get("code", ""))
        if why is None and not (ma and mb and ia and ib):
            why = "product form"
        if why is None and (ma["tmpl"] != mb["tmpl"] or ma["off"] != mb["off"] or ma["rows"] != mb["rows"]
                            or ma["s0"] != mb["s0"]):
            why = "different tiles / segments"
        if why is None:
            ga, gb = _split(ia["init"]), _split(ib["init"])
            if len(ga) != 15 or len(gb) != 15 or ga[9:13] != gb[9:13] or ga[11:13] != gb[11:13]:
                why = "different extents"

        if why is not None:
            records.append(dict(add=E.get("names", [out])[0], refused=why))
            continue
        host, other = (kb, ka) if kb > ka else (ka, kb)
        gh = _split(ia["init"]) if host == ka else _split(ib["init"])
        go = _split(ib["init"]) if host == ka else _split(ia["init"])
        mh, mo = (ma, mb) if host == ka else (mb, ma)
        lo = layouts[out]
        gh[8] = f"({ident(lo.root)} + {int(lo.offset)}LL)"
        go[8] = gh[8]       # the second pair's own result is not materialized (its C field is unused by the kernel)
        H = phases[host]
        H["code"] = f"""{{
  popsc::BArgs g0{{{', '.join(gh)}}};
  popsc::BArgs g1{{{', '.join(go)}}};
  latch_grp::tflat_kcat_claim_h<{mh["tmpl"]},{mh["s0"]}>({mh["ma"]}, {mh["mb"]}, {mo["ma"]}, {mo["mb"]}, g0, g1, {mh["off"]}, {mh["rows"]}, smem, /*XMAT_ROT*/0, {mh["slot"]});
}}"""
        H["reads"] = list(dict.fromkeys(list(H["reads"]) + list(phases[other]["reads"])))
        H["writes"] = [out]
        H["names"] = list(dict.fromkeys(list(H["names"]) + list(phases[other]["names"]) + list(E["names"])))
        H["targets"] = list(H.get("targets", ())) + list(phases[other].get("targets", ())) + list(E.get("targets", ()))
        H["engine"] = "ragged_flat_tma_claim_kcat_wgmma"
        H["kcat"] = dict(products=[Pa["names"][0], Pb["names"][0]], add=E["names"][0])
        H["arithmetic"] = (H.get("arithmetic", "") + "; kcat: two sibling products' k stages accumulated "
                           "into one FP32 tile, one BF16 rounding (the elementwise sum of the two results is not "
                           "materialized)")
        H.pop("flat_consumable", None)
        H.pop("flat_numel", None)
        remove |= {other, ke}
        removed |= {a, b}
        records.append(dict(add=E["names"][0], products=[Pa["names"][0], Pb["names"][0]], host=H["names"][0]))
    return records, remove, removed


# ---------------------------------------------------------------------------------------------- q/k/v triples
# Three-way form (feature "kcat", acting only where feature "packsum" fused the sum): three claimed right products over
# the same packed rows whose results are summed only inside the fused pack-sum load of a routing-pack gradient P
# (``fused_pack_sum``: widened FP32 sum of the three BF16 results at each gathered row). The three products become one
# K-concatenated product (tflat_kcat3_claim_h) writing the BF16 rounding of the FP32 sum into the first product's
# buffer, and P's fused load block becomes a plain widened load of that buffer. Not materialized: the other two results
# (2 x rows x N x 2 bytes written and read). Arithmetic: same sum; one BF16 rounding of the FP32 sum instead of three
# BF16 results summed in FP32 (not bitwise).
_BLOCK = re.compile(r"\{ const long long rb_ = (?P<x>.*?); (?P<loads>(?:const uint2 in\d_ = reinterpret_cast<const "
                    r"uint2\*>\(v_\w+ \+ rb_\)\[col\]; )+)float4 t4_; (?P<rest>.*?) v_\[s_\] = t4_; \}")
_LOADV = re.compile(r"reinterpret_cast<const uint2\*>\((v_\w+) \+ rb_\)\[col\]")
_REST_OK = re.compile(r"__bfloat162float|__ushort_as_bfloat16|unsigned short|in\d_\.[xy]|0xffffu|>> 16|& |"
                      r"\* 1\.0f|const float r_\w+ = |r_\w+|t4_\.[xyzw] = |[(){};+ ]")


def fuse_qkv(phases, layouts, dtypes, ident, kept):
    """Returns (records, removed phase indices, removed value names)."""
    if not enabled():
        return [], set(), set()
    root = lambda n: layouts[n].root if n in layouts else n
    producer = {w: k for k, p in enumerate(phases) for w in p.get("writes", ())}
    readers = {}
    for k, p in enumerate(phases):
        for r in p.get("reads", ()):
            readers.setdefault(root(r), set()).add(k)
    records, remove, removed = [], set(), set()
    for kp, P in enumerate(phases):
        if not P.get("fused_pack_sum") or P.get("kind") != "embedding_gradient":
            continue
        code = P.get("code", "")
        m = _BLOCK.search(code)
        if not m:
            records.append(dict(pack=P["names"][0], refused="fused load form"))
            continue
        vals = _LOADV.findall(m["loads"])
        rest = _REST_OK.sub("", m["rest"])
        names = [v[2:] for v in vals]
        if len(names) != 3 or rest.strip() or len(set(names)) != 3:
            records.append(dict(pack=P["names"][0], refused="not a plain three-operand sum", rest=rest[:80]))
            continue
        ks = [producer.get(n) for n in names]
        if any(k is None or k in remove for k in ks):
            continue
        Ps = [phases[k] for k in ks]
        why = None
        if any(p.get("engine") != "ragged_flat_tma_claim_wgmma" or p.get("tma_outer") or not p.get("flat_claim")
               or len(p.get("writes", ())) != 1 for p in Ps):
            why = "products"
        elif any(readers.get(root(n)) != {kp} for n in names) or set(names) & set(kept):
            why = "other readers"
        lays = [layouts.get(n) for n in names]
        if why is None and (any(l is None for l in lays) or any(dtypes.get(n) != "bfloat16" for n in names)
                            or len({tuple(l.shape) for l in lays}) != 1 or not all(_contiguous(l) for l in lays)):
            why = "layouts"
        ms = [_CALL.search(p.get("code", "")) for p in Ps]
        found = [_INIT.search(p.get("code", "")) for p in Ps]
        if why is None and (not all(ms) or not all(found)):
            why = "product form"
        if why is None and len({(x["tmpl"], x["off"], x["rows"], x["s0"]) for x in ms}) != 1:
            why = "different tiles / segments"
        inits = [_split(g["init"]) for g in found] if why is None else []
        if why is None and (any(len(g) != 15 for g in inits) or len({tuple(g[9:13]) for g in inits}) != 1):
            why = "different extents"

        if why is not None:
            records.append(dict(pack=P["names"][0], refused=why))
            continue
        host = max(ks)
        hi = ks.index(host)
        order = [hi] + [i for i in range(3) if i != hi]          # host pair first (its claim slot / STATIC0)
        target = names[0]                                        # the fused BF16 sum lives in the first buffer
        lt = layouts[target]
        cptr = f"({ident(lt.root)} + {int(lt.offset)}LL)"
        for g in inits:
            g[8] = cptr
        x = [ms[i] for i in order]
        H = phases[host]
        H["code"] = f"""{{
  popsc::BArgs g0{{{', '.join(inits[order[0]])}}};
  popsc::BArgs g1{{{', '.join(inits[order[1]])}}};
  popsc::BArgs g2{{{', '.join(inits[order[2]])}}};
  latch_grp::tflat_kcat3_claim_h<{x[0]["tmpl"]},{x[0]["s0"]}>({x[0]["ma"]}, {x[0]["mb"]}, {x[1]["ma"]}, {x[1]["mb"]}, {x[2]["ma"]}, {x[2]["mb"]}, g0, g1, g2, {x[0]["off"]}, {x[0]["rows"]}, smem, /*XMAT_ROT*/0, {x[0]["slot"]});
}}"""
        H["reads"] = list(dict.fromkeys(r for p in Ps for r in p["reads"]))
        H["writes"] = [target]
        H["names"] = list(dict.fromkeys(n for p in Ps for n in p["names"]))
        H["targets"] = [t for p in Ps for t in p.get("targets", ())]
        H["engine"] = "ragged_flat_tma_claim_kcat_wgmma"
        H["kcat"] = dict(products=[p["names"][0] for p in Ps], pack=P["names"][0])
        H["arithmetic"] = (H.get("arithmetic", "") + "; kcat3: three sibling products' k stages "
                           "accumulated into one FP32 tile, one BF16 rounding")
        H.pop("flat_consumable", None)
        H.pop("flat_numel", None)
        new_load = f"v_[s_] = latch_pgrad::widen4(reinterpret_cast<const uint2*>({ident(target)} + ({m['x']}))[col]);"
        P["code"] = code[:m.start()] + new_load + code[m.end():]
        P["reads"] = [r for r in P["reads"] if r not in names[1:]]
        P["arithmetic"] = P.get("arithmetic", "") + "; kcat3: reads the fused BF16 sum"
        others = [k for k in ks if k != host]
        remove |= set(others)
        removed |= set(names[1:])
        records.append(dict(pack=P["names"][0], products=[p["names"][0] for p in Ps], host=H["names"][0]))
    return records, remove, removed
