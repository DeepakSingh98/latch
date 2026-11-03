"""Routing-pack rewrites for segmented products: shared pack casts and row-scattered product results.

Two joint-IR rewrites applied by ``lower`` next to the cast commutations (``gather_cast``), and the emission of the
products they mark (``kernels/pack_fusion.cuh``):

1. ``share_gather_casts`` (``graph_rules`` rule ``castshare``). An FP32 row gather read only by several
   identity-shaped BF16 casts

       g = embedding(T, idx)  FP32 [R, D];   c_j = _to_copy(g, bfloat16)  (j = 1..n, n >= 2)

   becomes  Tb = _to_copy(T, bfloat16) [rows of T];  g' = embedding(Tb, idx) BF16;  every reader of every c_j reads g'.
   Rounding is elementwise, so every c_j equals g' bit for bit (``gather_cast`` makes the same commutation for one
   cast). The FP32 rows and n - 1 duplicate BF16 copies are neither written nor read. Gradients are unaffected: the
   joint IR already holds each cast's VJP (the widened cotangent of each reader, summed in FP32 before the gather's
   VJP), and the gather's VJP reads only its indices. Example: gathered rows cast once per projection role that
   reads them.

2. ``scatter_outputs`` (``segmented_products`` feature ``scatter``; GEMM -> unpack): a right product whose result is
   read only by a bijective row unpack (``permutation_gather`` with its validated inverse, or ``embedding`` by
   route_order's inverse) stores each result row at its unpacked row, with the unpack's own bijection check; the packed
   result and the copy are not materialized. Products whose lowering does not select a claimed TMA body store their
   packed rows and are followed by the standalone checked row scatter (``emit``: fallback).

Both rewrites are off unless selected; with neither selected the graph is returned unchanged. The scheduler is
unchanged; removing a pack phase only removes its own phase from the dependency graph.
"""
from __future__ import annotations

import copy

PERMUTATION = "latch_ops.permutation_gather.default"
ORDER = "latch_ops.ragged_order.default"
GATHER = "aten.embedding.default"
CAST = "aten._to_copy.default"
HEADER = "pack_fusion.cuh"
RIGHT = ("latch_ops.ragged_mm.default", "latch_ops.ragged_mm_observed.default")


def _target(op):
    return str(op.get("target") or "").removeprefix("torch.ops.")


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _replace(obj, old, new):
    if isinstance(obj, dict):
        if set(obj) == {"ref"} and obj["ref"] == old:
            return {"ref": new}
        return {k: _replace(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace(v, old, new) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace(v, old, new) for v in obj)
    return obj


def _swap(seq, old, new):
    return [new if x == old else x for x in seq] if isinstance(seq, (list, tuple)) else seq


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return list(reversed(strides))


def _is_contiguous(value):
    shape, strides = list(value.get("shape") or ()), value.get("strides")
    return strides is None or all(d == 1 or s == t for d, s, t in zip(shape, strides, _contiguous(shape)))


def _reads(op):
    return set(op.get("reads") or op.get("inputs") or ())


def share_gather_casts(data, outputs, report):
    ops, values = data["operations"], data["values"]
    for op in list(ops):
        if _target(op) != GATHER:
            continue
        g = op["name"]
        vg = values.get(g) or {}
        if vg.get("dtype") != "float32" or g in outputs or len(vg.get("shape") or ()) != 2:
            continue
        if any(v.get("alias_of") == g for n, v in values.items() if n != g):
            continue                                                # views of g: left to gather_cast / unchanged
        readers = [o for o in ops if o is not op and g in _reads(o)]
        if len(readers) < 2 or any(_target(o) != CAST for o in readers):
            continue
        casts = []
        for cast in readers:
            c = cast["name"]
            vc = values.get(c) or {}
            if (c in outputs or vc.get("dtype") != "bfloat16" or _ref((cast.get("args") or [None])[0]) != g
                    or list(vc.get("shape") or ()) != list(vg["shape"]) or not _is_contiguous(vc)
                    or any(k not in ("dtype", "layout", "device", "memory_format") for k in (cast.get("kwargs") or {}))
                    or any(v.get("alias_of") == c and n in outputs for n, v in values.items())):
                break
            casts.append(cast)
        else:
            table = _ref(op["args"][0])
            if table is None or (values.get(table) or {}).get("dtype") != "float32":
                continue
            tb = f"{table}__bf16_pack_{g}"
            if tb in values:
                continue
            first = casts[0]
            values[tb] = {**copy.deepcopy(values[table]), "name": tb, "dtype": "bfloat16", "role": "temporary",
                          "parameter": None, "alias_of": None, "storage_offset": 0, "producer": tb, "output_roles": [],
                          "strides": _contiguous(values[table]["shape"])}
            cast_table = {**copy.deepcopy(first), "name": tb, "args": [{"ref": table}, *copy.deepcopy(first["args"][1:])],
                          "inputs": [table], "outputs": [tb],
                          "dependencies": [] if table in data.get("inputs", ()) else [table],
                          "reads": [table], "writes": [tb]}
            ops.insert(ops.index(op), cast_table)
            op["args"] = _replace(op["args"], table, tb)
            for key in ("inputs", "dependencies", "reads"):
                op[key] = _swap(op.get(key), table, tb)
            vg["dtype"] = "bfloat16"
            vg["strides"] = _contiguous(vg["shape"])
            removed = []
            for cast in casts:
                c = cast["name"]
                ops.remove(cast)
                for o in ops:
                    o["args"] = _replace(o.get("args"), c, g)
                    o["kwargs"] = _replace(o.get("kwargs"), c, g)
                    for key in ("inputs", "dependencies", "reads"):
                        o[key] = _swap(o.get(key), c, g)
                for v in values.values():
                    if v.get("alias_of") == c:
                        v["alias_of"] = g
                del values[c]
                removed.append(c)
            report["shared_casts"].append({"gather": g, "table": table, "table_cast": tb, "removed_casts": removed,
                                           "rows": vg["shape"][0], "width": vg["shape"][1]})


def _claim_admissible(rows, groups):
    from . import ragged_experts_lower as rel
    return rel._feature("tma") and rel._feature("claim") and rows >= rel.claim_min_rows() * groups


def scatter_outputs(data, outputs, report, target_arch):
    """GEMM -> unpack: a right ragged product whose BF16 result C is read only by a bijective row unpack U of it,
    either ``permutation_gather(C, order_arg, inverse_arg)`` (U[j] = C[order_arg[j]], runtime-checked
    inverse_arg[order_arg[j]] == j) or ``embedding(C, inv)`` with ``inv`` the inverse output of a ``ragged_order``
    (route_order) whose order output is ``ord``, writes U directly: row r of C goes to U[dest[r]] with
    dest = inverse_arg (resp. ord), and the epilogue checks check[dest[r]] == r with check = order_arg (resp. inv), the
    unpack's own bijection condition. One store per row, no reduction, no atomics; the unpack's VJP (a gather of the
    cotangent) is untouched. Claimed TMA bodies only; otherwise the standalone unpack is emitted after the product."""
    from . import ragged_experts_lower as rel
    if target_arch != "sm_90" or rel.current_products() is None:
        return
    ops, values = data["operations"], data["values"]
    producer = {}
    for o in ops:
        for w in o.get("outputs") or ():
            producer[w] = o
    for u in list(ops):
        t = _target(u)
        args = u.get("args") or ()
        if t == PERMUTATION and len(args) == 3:
            c, dest, check = _ref(args[0]), _ref(args[2]), _ref(args[1])
        elif t == GATHER and len(args) >= 2 and all(a in (None, -1, False) for a in args[2:]):
            c, inv = _ref(args[0]), _ref(args[1])
            r = producer.get(inv)
            if r is None or _target(r) != ORDER or list(r.get("outputs") or ())[2:3] != [inv]:
                continue
            dest, check = list(r["outputs"])[0], inv
        else:
            continue
        p = producer.get(c)
        out = (u.get("outputs") or [None])[0]
        if (p is None or _target(p) not in RIGHT or c in outputs or out in outputs or out is None
                or (p.get("kwargs") or {}) or (values.get(c) or {}).get("dtype") != "bfloat16"):
            continue
        vc, vu = values.get(c) or {}, values.get(out) or {}
        if (list(vc.get("shape") or ()) != list(vu.get("shape") or ()) or len(vc.get("shape") or ()) != 2
                or not _is_contiguous(vc) or not _is_contiguous(vu) or vu.get("dtype") != "bfloat16"):
            continue
        if any(v.get("alias_of") == c for n, v in values.items() if n != c):
            continue
        if [o for o in ops if o is not u and c in _reads(o)]:
            continue                                                # C has another reader
        rows = vc["shape"][0]
        groups = ((values.get(_ref((p.get("args") or (None, None, None))[2])) or {}).get("shape") or [0])[0] - 1
        if groups < 1 or not _claim_admissible(rows, groups):
            continue
        # Pairing (flat2) needs a partner with the same A; the marked product is emitted unpaired (emit: phases=None),
        # and any other body falls back to the standalone checked scatter.
        ops.remove(u)
        p["outputs"] = [out if x == c else x for x in p["outputs"]]
        p["writes"] = [out if x == c else x for x in (p.get("writes") or ())]
        for key in ("inputs", "reads"):
            seq = list(p.get(key) or ())
            p[key] = seq + [x for x in (dest, check) if x not in seq]
        p["kwargs"] = {"pack_scatter": {"dest": dest, "check": check, "packed": c, "rows": rows,
                                        "unpack": t, "unpack_op": u["name"]}}
        values[out]["producer"] = p["name"]
        del values[c]
        report["scattered_outputs"].append({"product": p["name"], "target": _target(p), "packed": c, "unpacked": out,
                                            "unpack": t, "dest": dest, "check": check, "rows": rows})


def rewrite(graph, *, target_arch="sm_90"):
    """Return (graph, report); the input is never mutated. Identity unless graph_rules 'castshare' or
    segmented_products 'scatter' is selected."""
    from . import graph_rules
    from . import ragged_experts_lower as rel
    share = graph_rules.active("castshare")
    scatter = rel._feature("scatter")
    if not (share or scatter):
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or not isinstance(values, dict) or not any(_target(op) in (GATHER, PERMUTATION) for op in ops):
        return graph, None
    from .gather_cast import _output_names
    outputs = _output_names(data)
    report = {"version": 1, "shared_casts": [], "scattered_outputs": []}
    if share:
        share_gather_casts(data, outputs, report)
    if scatter:
        scatter_outputs(data, outputs, report, target_arch)
    if not report["shared_casts"] and not report["scattered_outputs"]:
        return graph, None
    data.setdefault("metadata", {})["pack_fusion"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def marked(op, field):
    kwargs = field(op, "kwargs", None) or {}
    return kwargs.get("pack_scatter") if isinstance(kwargs, dict) else None


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define, target_arch, phases=None, producers=None,
         operations=None):
    """Emit a product marked by ``scatter_outputs``: the claimed TMA body storing rows at their unpacked positions,
    or (fallback) the unchanged product followed by the standalone checked row scatter."""
    return _emit_scatter(op, values=values, layouts=layouts, dtypes=dtypes, names=names, ref=ref, field=field,
                         ident=ident, define=define, target_arch=target_arch, producers=producers,
                         operations=operations)


def _emit_scatter(op, *, values, layouts, dtypes, names, ref, field, ident, define, target_arch, producers, operations):
    """Emit a product marked by ``scatter_outputs``: the claimed body storing rows at their unpacked positions, or
    (fallback) the unchanged product into the packed rows followed by the standalone checked row scatter."""
    import dataclasses
    import re
    from .lower import LoweringError, TensorLayout
    from . import ragged_experts_lower as rel
    info = (field(op, "kwargs", None) or {})["pack_scatter"]
    packed, dest, check, rows = info["packed"], info["dest"], info["check"], info["rows"]
    outs = list(field(op, "outputs", ()))
    unpacked = outs[0]
    shape = tuple(_value_field(values[unpacked], "shape"))
    width = shape[1]
    define(unpacked, shape, None, "bfloat16")

    def ptr(name):
        layout = layouts[name]
        return f"({ident(layout.root)} + {layout.offset}LL)"
    ld, lc = layouts[dest], layouts[check]
    if dtypes[dest] != "int64" or dtypes[check] != "int64" or ld.shape != (rows,) or lc.shape != (rows,) \
            or not ld.contiguous or not lc.contiguous:
        raise LoweringError("pack_fusion: unpack permutation storage changed after the rewrite")
    proxy_values = dict(values)
    proxy_values[packed] = values[unpacked]
    new_outs = [packed if x == unpacked else x for x in outs]
    rest = {k: v for k, v in (field(op, "kwargs", None) or {}).items() if k != "pack_scatter"}
    if isinstance(op, dict):
        proxy = {**op, "outputs": new_outs, "kwargs": rest}
    else:
        proxy = dataclasses.replace(op, outputs=tuple(new_outs), kwargs=rest)
    layouts[packed] = TensorLayout(shape, (width, 1), packed)
    dtypes[packed] = "bfloat16"
    materialize = []

    def proxy_define(value, value_shape, value_strides, value_dtype):
        if value == packed:
            if materialize:
                define(value, value_shape, value_strides, value_dtype)
            return
        define(value, value_shape, value_strides, value_dtype)
    emitted = rel.emit(proxy, values=proxy_values, layouts=layouts, dtypes=dtypes, names=names, ref=ref, field=field,
                       ident=ident, define=proxy_define, target_arch=target_arch, phases=None, producers=producers,
                       operations=operations)
    product = next((p for p in emitted if p.get("kind") == "gemm"), None)
    head = "latch_ragged::tflat_claim<"
    if product is None or product.get("engine") != "ragged_flat_tma_claim_wgmma" or product["code"].count(head) != 1:
        # Fallback: the product stores its packed rows, then the standalone checked row scatter (U[dest[r]] = C[r]).
        define(packed, shape, None, "bfloat16")
        vec = width // 8
        scatter = dict(kind="indexed_permutation", names=[unpacked], targets=[info["unpack"]], reads=[packed, dest, check],
                       writes=[unpacked], code=f"""{{
  const uint4* src_ = reinterpret_cast<const uint4*>({ptr(packed)});
  uint4* dst_ = reinterpret_cast<uint4*>({ptr(unpacked)});
  for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < {rows * vec}LL;
       v += (long long)gridDim.x * blockDim.x) {{
    const long long r = v / {vec}LL, c = v % {vec}LL;
    const int64_t d = {ptr(dest)}[r];
    if (d < 0 || d >= {rows} || {ptr(check)}[d] != r) __trap();
    dst_[d * {vec}LL + c] = src_[v];
  }}
}}""", threads=256, requires_full_grid=True, work_items=max(1, rows * vec // 256), shared_bytes=0,
                       engine="pack_fusion_fallback_row_scatter",
                       arithmetic="storage-preserving checked row scatter (the unpack's bijection condition)")
        for p in emitted:
            p["pack_scatter_fallback"] = True
        return emitted + [scatter]
    code = product["code"]
    start = code.index(head)
    close = code.index(">(", start)
    code = code[:start] + "latch_pack::tflat_claim_scatter<" + code[start + len(head):close] + \
        f">({ptr(dest)}, {ptr(check)}, " + code[close + 2:]
    packed_ptr = f"({ident(packed)} + 0LL)"
    if code.count(packed_ptr) != 1:
        raise LoweringError("pack_fusion: unexpected product result pointer form")
    product["code"] = code.replace(packed_ptr, ptr(unpacked))
    if re.search(rf"\b{ident(packed)}\b", product["code"]):
        raise LoweringError("pack_fusion: the packed result is still referenced")
    product["names"] = [unpacked if n == packed else n for n in product["names"]]
    product["writes"] = [unpacked if n == packed else n for n in product["writes"]]
    product["reads"] = list(dict.fromkeys(list(product["reads"]) + [dest, check]))
    for key in ("flat_consumable", "flat_numel", "flat_pair_key"):
        product.pop(key, None)                                     # tile consumers index the packed rows
    if product.get("work_output") == packed:
        product["work_output"] = unpacked
    product["pack_scatter"] = {"unpacked": unpacked, "dest": dest, "check": check, "rows": rows,
                               "packed_value_not_materialized": packed, "unpack": info["unpack"]}
    product["engine"] = product["engine"].replace("_wgmma", "_scatter_wgmma")
    product["arithmetic"] += ("; result rows stored at their unpacked rows dest[r] with the unpack's bijection check "
                              "(the packed result and the unpack copy are not materialized)")
    del layouts[packed], dtypes[packed]
    return emitted


def _value_field(value, key):
    return value.get(key) if isinstance(value, dict) else getattr(value, key)
