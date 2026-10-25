"""Operation-based lowering of packed segmented products; no model recognizer.

``segmented_products="flat"`` (opt-in, ``CapturedModule.build``; LATCH-TINYMOE-XMAT) selects the
flattened WGMMA body ``latch_ragged::flat`` for aligned BF16 sm_90 products: a CTA's (segment, tile)
tasks share one operand pipeline, stages prefetch across task boundaries and the body is inlined
(isolated in its own call-free device-unit piece). Tile shape comes from ``flat_tile`` (structural:
product kind and extents only). Two products with the same A value, offsets, extents and strides (gate
and up projections, their weight gradients) are computed by one ``flat2`` phase sharing A stages when
the second product's B is already available ("flat"; "flat_single" keeps them separate). Per-element
arithmetic, task enumeration and zero-filled staging are those of ``latch_ragged::product``, so results
are bitwise unchanged. None leaves lowering unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math

from .ragged_experts import ORDER, MM, MM_RAW, OUTER, OUTER_WIDE, OUTER_ACC, TARGETS

# Option value -> features of the flattened body. "flat" pairs products sharing A ("pair"); "flat_single" does not.
# Further "+"-joined features: "tma" (TMA-fed right products), "prefetch" (L2 prefetch of a product's older operands
# one dependency level ahead), "rotate" (consecutive flat products of one interval continue the round-robin CTA
# assignment where the earlier ones ended), "consume" (identity-indexed elementwise consumers of a right product run
# tile-locally in the CTA that stored the tile; see absorb_consumers). Earlier spellings flat_tma, flat_pf and flat_tma_pf remain accepted.
_BASES = {"flat": ("pair",), "flat_single": ()}
# "narrow" (LATCH-B1-FAST): the right (token-row) products take 64-row tiles (BM 64, BN 128, one warpgroup per
# 64-row slab), twice the tasks of the 128-row tile; for small microbatches (B1 x 1024: ~128 routed rows per expert)
# the 128-row tile leaves a third of the grid idle and most tiles half-filled. Every result keeps its wgmma k16
# chain, so values are bitwise unchanged; only which CTA computes a tile changes. Weight-gradient (outer) tiles
# are unchanged.
# "claim" (LATCH-TRAIN-EXPERTS): TMA-fed products take their tasks from a per-launch counter (kernels/ragged_claim.cuh):
# CTAs that finish early take more tasks, a CTA whose queue is empty moves on to the interval's next phase, and
# weight-gradient tasks go longest segment first. Each task is still computed whole by one CTA with unchanged
# arithmetic, so results are bitwise those of the static bodies; "rotate" is moot for claimed products.
# "repi" (LATCH-TRAIN-EXPERTS, with "claim"): identity-indexed elementwise consumers of a claimed right product (e.g.
# SwiGLU) run in the product's store loop on the values the thread just rounded from its accumulators (no re-read of the
# stored tile, no tile walk); their generated per-element bodies are unchanged. Unlike "consume" (tile-local re-read).
# "sepi" (with "claim"): claimed right products stage each warpgroup's BF16 tile in shared memory and store whole
# 16-byte row chunks (kernels/ragged_claim.cuh LATCH_CLAIM_SEPI); paired products take 3 stages to fit the staging.
_EXTRA = ("tma", "prefetch", "rotate", "consume", "narrow", "claim", "repi", "sepi")
_ALIASES = {"flat_tma": "flat+tma", "flat_pf": "flat+prefetch", "flat_tma_pf": "flat+tma+prefetch"}


def _features(value):
    if value is None:
        return None
    base, *extra = _ALIASES.get(value, value).split("+")
    if base not in _BASES or any(e not in _EXTRA for e in extra) or len(set(extra)) != len(extra):
        raise ValueError(f"segmented_products must be None or flat|flat_single with optional +{'/+'.join(_EXTRA)}")
    return _BASES[base] + tuple(extra)


PRODUCTS = (None, "flat", "flat_single", *_ALIASES, "flat+tma+prefetch+rotate", "flat+tma+rotate+narrow",
            "flat+tma+rotate+claim", "flat+tma+rotate+narrow+claim")   # documented examples
_PRODUCTS = ContextVar("latch_segmented_products", default=None)


def _feature(name):
    value = _PRODUCTS.get()
    return value is not None and name in _features(value)


def validate_products(value):
    _features(value)
    return value


def current_products():
    return _PRODUCTS.get()


@contextmanager
def products_scope(value):
    token = _PRODUCTS.set(validate_products(value))
    try:
        yield
    finally:
        _PRODUCTS.reset(token)


def flat_tile(outer, m, n, k):
    """(BM, BN, WM, stages) of the flattened body; structural rule over the product's static extents.

    Two warpgroups each own a 64-row slab of a 128-row tile (m64nBNk16 wgmma); BN 128 keeps two or
    three column tiles for the 384/768-wide expert products and four 48 KiB-or-smaller stages.
    Screening only: LATCH_RAGGED_FLAT_TILE="mm=BM,BN,WM,S;outer=BM,BN,WM,S" overrides the rule (the
    generated source, and therefore the build identity, records the chosen tile).
    """
    import os
    override = os.environ.get("LATCH_RAGGED_FLAT_TILE")
    if override:
        table = dict(item.split("=") for item in override.split(";"))
        bm, bn, wm, stages = (int(v) for v in table["outer" if outer else "mm"].split(","))
        if wm not in (1, 2) or bm % 64 or bn % 32 or not 2 <= stages <= 8:
            raise ValueError("invalid LATCH_RAGGED_FLAT_TILE")
        return bm, bn, wm, stages
    if not outer and _feature("narrow"):
        return 64, 128, 1, 4
    return 128, 128, 2, 4

HEADERS = ("tgemm.cuh", "bgemm.cuh", "popsc_mm.cuh", "s107a_wgmma.cuh", "pops4b_wg.cuh", "ragged_experts.cuh")
TMA_HEADERS = ("pops4b_tma.cuh", "ragged_tma.cuh")
CLAIM_MIN_ROWS = 1024                 # average packed rows per segment for a claimed product (structural rule)
# Claimed outer products: experts holding >= CLAIM_HOT x the mean packed rows (measured TinyMoE routing: the largest
# expert holds 3-8x the mean) take CLAIM_HOT_TILE (BM, BN, WM, stages) tiles, so their k chains, which bound the
# product, are spread over more CTAs. 0 disables the hot class.
CLAIM_HOT = 0                         # measured: hot tiles 0.80x in isolation but +0.4 ms in the complete B32 update
CLAIM_WIDE = 1                        # wide claimed tiles (128 x 256 for K 384, 256 x 128 for K 768 input gradients):
                                      # with 'sepi' 72.24 -> 72.06 ms (B32, bitwise); without it within spread
CLAIM_MERGE = 1                       # one queue for the two claimed dW products of an interval: measured
                                      # 74.64 -> 74.19 ms bitwise (LATCH_CLAIM_MERGE=0 screens it off)
CLAIM_HOT_TILE = (128, 64, 2, 6)
CLAIM_HEADER = "ragged_claim.cuh"     # added only to programs that emit a claimed product (feature "claim")
CLAIM_SLOT = "/*XCLAIM*/nullptr"       # replaced by the phase's counter pointer when its interval is emitted
CLAIM_STATIC0 = "/*XCLAIM_S0*/true"    # true: static first task (first claimed product of its interval)


# Pending (deferred) flat products per lowering, keyed by the phase list they will join (identity-checked, so a
# lowering that raised never leaks a pending product into a later one).
_PENDING = {}


def _pending(phases):
    entry = _PENDING.get(id(phases))
    if entry is None or entry[0] is not phases:
        entry = _PENDING[id(phases)] = (phases, {})
    return entry[1]


def _partner_ahead(op, operations, field, ref, names, values, key_a, offsets, b, output):
    """True when a later product pairs with ``op`` and no operation between them reads ``op``'s result.

    The partner has the same target and offsets value, an A operand with the same structural identity and a B
    value with the same shape and strides; ``emit`` re-checks the complete pairing key when it arrives.
    """
    if operations is None:
        return False
    target = str(field(op, "target")).removeprefix("torch.ops.")
    position = next((i for i, other in enumerate(operations) if other is op), None)
    if position is None:
        return False
    vb = values.get(b)
    for other in operations[position + 1:]:
        if output in tuple(field(other, "inputs", ()) or ()):
            return False
        if str(field(other, "target")).removeprefix("torch.ops.") != target:
            continue
        args = list(field(other, "args", ()))
        if len(args) != 3 or ref(args[2], names) != offsets:
            continue
        a2, b2 = ref(args[0], names), ref(args[1], names)
        if a2 is None or b2 is None or _canonical(a2, values, field, ref, names, operations) != key_a:
            continue
        vb2 = values.get(b2)
        if vb is not None and vb2 is not None and (tuple(field(vb, "shape")) == tuple(field(vb2, "shape"))
                                                   and tuple(field(vb, "strides")) == tuple(field(vb2, "strides"))):
            return True
    return False


def flush(phases):
    """Deferred products whose partner never matched, in deferral order (called after the operation loop)."""
    entry = _PENDING.pop(id(phases), None)
    return list(entry[1].values()) if entry is not None and entry[0] is phases else []


# Pure single-input casts: two values produced by the same cast of the same value are equal (common
# subexpressions), so their products may share one A stage. Only these targets are recognized.
_CASTS = ("aten._to_copy.default", "_to_copy.default")
_OPS = {}


def _canonical(value, values, field, ref, names, operations):
    """Structural identity of a product operand: the value, or (cast, source, attributes) of a pure cast."""
    if operations is not None:
        index = _OPS.get(id(operations))
        if index is None or index[0] is not operations:
            index = (operations, {field(o, "name"): o for o in operations})
            _OPS.clear()
            _OPS[id(operations)] = index
        producer = index[1].get(field(values[value], "producer", None)) if value in values else None
        if (producer is not None and str(field(producer, "target")).removeprefix("torch.ops.") in _CASTS
                and field(producer, "effect", "pure") == "pure" and len(field(producer, "outputs", ())) == 1):
            args = field(producer, "args", ())
            sources = [ref(a, names) for a in args]
            if len(args) == 1 and sources[0] is not None:
                kwargs = sorted((k, repr(v)) for k, v in dict(field(producer, "kwargs", {}) or {}).items())
                v = values[value]
                return ["cast", sources[0], repr(kwargs), list(field(v, "shape")), str(field(v, "dtype")),
                        list(field(v, "strides"))]
    return ["value", value]


def _prefetch_phase(phase, ptr, layouts, phases, producers, operands):
    """L2 prefetch of a product's older operands, placed one dependency level ahead of the product.

    The freshest operand (latest producing phase) anchors the placement: the prefetch phase reads that producer's
    own inputs, so the ASAP planner puts it in the producer's interval, and it reads the prefetched operands. It
    writes nothing and changes no value: each CTA's thread 0 issues bulk L2 prefetch hints for a slice of each
    older operand's storage span (16-byte aligned spans only).
    """
    if phases is None or producers is None:
        return []
    root = lambda name: layouts[name].root if name in layouts else name
    # RAW dependency levels of the phases emitted so far (phase order is not level order).
    writer, levels = {}, []
    for body in phases:
        levels.append(1 + max((levels[writer[root(r)]] for r in body.get("reads", ()) if root(r) in writer), default=-1))
        for w in body.get("writes", ()):
            writer[root(w)] = len(levels) - 1
    def producer(name):
        return producers.get(name, producers.get(root(name), -1))
    def level(name):
        return levels[producer(name)] if producer(name) >= 0 else -1
    order = sorted(operands, key=level)
    fresh = order[-1]
    anchor = producer(fresh)
    if anchor < 0 or not phases[anchor].get("reads"):
        return []
    lines, targets = [], []
    for name in dict.fromkeys(order[:-1]):
        layout = layouts[name]
        if level(name) >= levels[anchor] or any(s <= 0 for s in layout.strides):
            continue
        span = 1 + sum((d - 1) * s for d, s in zip(layout.shape, layout.strides))
        if (2 * layout.offset) % 16 or (2 * span) % 16:
            continue
        lines.append(f"  latch_ragged::l2_prefetch({ptr(name)}, {2 * span}LL);")
        targets.append(name)
    if not lines:
        return []
    reads = list(dict.fromkeys(list(phases[anchor]["reads"]) + targets))
    return [phase("ragged_prefetch", "{\n" + "\n".join(lines) + "\n}", [], reads, work_items=1, shared_bytes=0,
                  engine="ragged_l2_prefetch", prefetched=targets,
                  arithmetic="none (L2 prefetch hints only; no value read into registers or written)")]


def rotate_intervals(phases, intervals):
    """Feature "rotate": in each interval, a flat product starts its round-robin task assignment at the CTA after
    the last task of the interval's earlier flat products (their task counts are recomputed from the segment
    offsets at run time), so the extra tasks of consecutive products land on different CTAs. Task arithmetic and
    every result are unchanged; only which CTA computes a tile changes. Returns the rotated products."""
    if not _feature("rotate"):
        return []
    rotated = []
    for members in intervals:
        earlier = {}               # per segment-offsets value: earlier products' task counts (same pointers in scope)
        for i in members:
            body = phases[i]
            if "flat_tasks" not in body or "/*XMAT_ROT*/0" not in body.get("code", "") or body.get("flat_claim"):
                continue
            prior = earlier.setdefault(body["flat_offsets"], [])
            if prior:
                body["code"] = body["code"].replace(
                    "/*XMAT_ROT*/0", f"latch_ragged::rot_after({' + '.join(prior)})")
                rotated.append(body["names"][0])
            prior.append(body["flat_tasks"])
    return rotated


def absorb_consumers(phases, layouts, ident):
    """Feature "consume": move identity-indexed elementwise consumers into the right product that feeds them.

    An elementwise phase qualifies for a flat right product G (earliest such product first) when it reads at least
    one of G's results, every read and write is a whole contiguous root of G's [rows, N] element count indexed
    only at [i] in its body, and every read other than G's results is produced before G. Its unchanged body then
    runs as G's tile-local consumer (latch_ragged::tile_consumers: after each tile's store, in the same CTA, over
    the tile's elements). Values and their arithmetic are unchanged; the consumer phase and its pass over memory
    (and its interval, when it was alone) disappear. Returns the absorbed phase indices.
    """
    import math
    repi = _feature("repi") and _feature("claim")
    if not _feature("consume") and not repi:
        return set()
    writer = {}
    for index, body in enumerate(phases):
        for w in body.get("writes", ()):
            writer.setdefault(w, index)
    absorbed = set()

    def whole(name, numel):
        layout = layouts.get(name)
        return (layout is not None and layout.root == name and layout.contiguous and not layout.offset
                and layout.numel == numel)

    for g, gp in enumerate(phases):
        if not gp.get("flat_consumable") or g in absorbed or (repi and not gp.get("flat_claim")):
            continue
        numel = gp["flat_numel"]
        products = list(gp["writes"][:2 if gp.get("flat_paired") else 1])   # the product's own results, in order
        lines, used = [], set()
        for e in range(g + 1, len(phases)):
            ce = phases[e]
            if (e in absorbed or ce.get("kind") != "elementwise" or not ce.get("body")
                    or len(ce["body"]) != len(ce["names"]) or ce.get("vprod_inline") or ce.get("active_values")
                    or math.prod(ce.get("shape") or (0,)) != numel):
                continue
            outs = set(gp["writes"])
            if not any(r in outs for r in ce["reads"]):
                continue
            if not all(whole(v, numel) for v in [*ce["reads"], *ce["writes"]]):
                continue
            if not all(r in outs or r in ce["writes"] or writer.get(r, -1) < g for r in ce["reads"]):
                continue
            text = "\n".join(ce["body"])
            if any(text.count(ident(v) + "[") != text.count(ident(v) + "[i]") + text.count(ident(v) + "[(i)]")
                   for v in [*ce["reads"], *ce["writes"]]):
                continue
            if any(text.count(ident(w) + "[") != 1 for w in ce["writes"]):
                continue                     # one store per result and no reread: repeated elements are idempotent
            body = list(ce["body"])
            if repi:
                # The product's results come from registers (v0, v1: the BF16 bits the thread just stored).
                for k, v in enumerate(products):
                    body = [line.replace(ident(v) + "[(i)]", f"v{k}").replace(ident(v) + "[i]", f"v{k}") for line in body]
                if any(ident(v) + "[" in line for v in products for line in body):
                    continue
            lines.extend(body)
            used.update(v for v in [*ce["reads"], *ce["writes"]] if not (repi and v in products))
            gp["names"] = gp["names"] + ce["names"]
            gp["targets"] = gp["targets"] + ce["targets"]
            gp["writes"] = gp["writes"] + ce["writes"]
            gp["reads"] = list(dict.fromkeys(gp["reads"] + [r for r in ce["reads"] if r not in outs and r not in ce["writes"]]))
            gp.setdefault("tile_consumers", []).append(list(ce["names"]))
            for w in ce["writes"]:
                writer[w] = g
            absorbed.add(e)
        if lines:
            # A local functor (members: the pointers the bodies use) with a force-inlined call operator: a lambda's
            # operator may stay out of line under relocatable compilation, and any call serializes the product's
            # warpgroup MMA (C7509).
            names_ = sorted(ident(v) for v in used)
            signature = ("long long i, const s101h::bf16 v0, const s101h::bf16 v1" if repi else "long long i")
            lam = ("  struct XmatEpi {\n" + ("    using reg_tag = void;\n" if repi else "")
                   + "".join(f"    decltype({n}) {n};\n" for n in names_)
                   + f"    __device__ __forceinline__ void operator()({signature}) const {{\n      "
                   + "\n      ".join(lines) + "\n    }\n  };\n"
                   + f"  const XmatEpi xmat_epi{{{', '.join(names_)}}};\n")
            if repi:
                gp["register_consumers"] = True
            code = gp["code"]
            head, tail = code.split("\n", 1)
            # Static bodies end "/*XMAT_ROT*/0);"; claimed bodies (feature "claim") take the consumer after the counter.
            tail = tail.replace(f"{CLAIM_SLOT});", f"{CLAIM_SLOT}, xmat_epi);") if CLAIM_SLOT + ");" in tail else \
                tail.replace("/*XMAT_ROT*/0);", "/*XMAT_ROT*/0, xmat_epi);")
            gp["code"] = head + "\n" + lam + tail
            if "xmat_epi);" not in gp["code"]:
                raise ValueError("tile-local consumer could not be bound to its product call")
    return absorbed


def finalize_maps(program):
    """Bind POPS4B_TMAP placeholders left by TMA-fed segmented products when no matrix provider bound them.

    Same table ABI as matrix_adapters.provider.finalize (128-byte map entries after the program inputs; storage
    by input index or arena word offset; BF16, 128-byte swizzle, zero fill), and the same trap-for-assert rule
    for warpgroup-MMA programs. Programs without placeholders are returned unchanged.
    """
    import hashlib
    import json
    import re
    if "POPS4B_TMAP(" not in program.source:
        return program
    from .lower import GeneratedProgram
    from . import p5c_matrix
    manifest = dict(program.manifest)
    inputs = manifest["inputs"]
    maps, index = list(manifest.get("tensor_maps") or ()), {}

    def replace(match):
        key = match.group(1)
        if key not in index:
            root, offset, d0, d1, stride, b0, b1, *rest = key.split(",")
            layout = manifest.get("layouts", {}).get(root, {})
            storage_root = layout.get("root", root) if isinstance(layout, dict) else root
            element_offset = int(offset) + (int(layout.get("offset", 0)) if isinstance(layout, dict) else 0)
            if storage_root in inputs:
                where = {"storage": "input", "index": inputs.index(storage_root)}
            elif storage_root in manifest["offsets"]:
                where = {"storage": "arena", "word_offset": int(manifest["offsets"][storage_root])}
            else:
                raise ValueError(f"tensor-map operand root {root} has no storage")
            index[key] = len(maps)
            maps.append({"root": storage_root, **where, "element_offset": element_offset,
                         "dims": [int(d0), int(d1)], "stride_bytes": int(stride), "box": [int(b0), int(b1)],
                         "dtype": "bfloat16", "swizzle": int(rest[0]) if rest else 128, "oob_fill": "zero"})
        return f"(reinterpret_cast<const char*>(args[{len(inputs)}]) + {128 * index[key]})"

    source = re.sub(r"POPS4B_TMAP\(([^()]*)\)", replace, program.source)
    entry = 'extern "C" __global__'
    if source.count(entry) == 1 and "P-OPS5-C: generated index bounds checks" not in source:
        source = source.replace(entry, p5c_matrix.TRAP_ASSERT + entry, 1)
        manifest["index_checks"] = "trap (assert overridden for wgmma programs)"
    manifest["tensor_maps"] = maps
    manifest["source_hash"] = hashlib.sha256(
        (source + json.dumps(manifest["header_hashes"], sort_keys=True)).encode()).hexdigest()
    return GeneratedProgram(source, manifest, program.kernel_body, program.helpers, program.functions)


def _emit_outer_acc(args, outputs, *, layouts, dtypes, names, ref, ptr, phase, target_arch, LoweringError):
    """LATCH-SELECTIVE-10 accfuse: ``ragged_outer_acc(a, b, offsets, flags, accs)`` -> E FP32 results
    ``accs[e] + where(flags[e], float(ragged_outer(a, b, offsets)[e]), +0)`` from the TMA outer body
    (kernels/ragged_tma_acc.cuh): the product's tasks and arithmetic, and per element the cast, select and FP32
    addition the replaced operations evaluated. Outputs may alias their accumulators (same-index read then write)."""
    if len(args) != 5 or not isinstance(args[3], (list, tuple)) or not isinstance(args[4], (list, tuple)):
        raise LoweringError("ragged_outer_acc: invalid operation ABI")
    a, b, offsets = (ref(x, names) for x in args[:3])
    flags, accs = [ref(x, names) for x in args[3]], [ref(x, names) for x in args[4]]
    la, lb, ls = layouts[a], layouts[b], layouts[offsets]
    groups = ls.shape[0] - 1
    if (len(ls.shape) != 1 or dtypes[offsets] != "int64" or not ls.contiguous or not 1 <= groups <= 256
            or len(flags) != groups or len(accs) != groups or len(outputs) != groups or None in flags or None in accs):
        raise LoweringError("ragged_outer_acc: one flag, accumulator and result per segment")
    if len(la.shape) != 2 or len(lb.shape) != 2 or lb.shape[0] != la.shape[0]:
        raise LoweringError("ragged_outer_acc: operand extents")
    rows, m = la.shape
    n, k = lb.shape[1], rows
    if dtypes[a] != "bfloat16" or dtypes[b] != "bfloat16" or target_arch != "sm_90" or _PRODUCTS.get() is None:
        raise LoweringError("ragged_outer_acc requires BF16 operands and the flat sm_90 products")
    sam, sak = la.strides[1], la.strides[0]
    sbk, sbn = lb.strides
    if not (sam == 1 and sbn == 1 and (sak * 2) % 16 == 0 and (sbk * 2) % 16 == 0 and (la.offset * 2) % 16 == 0
            and (lb.offset * 2) % 16 == 0 and n % 2 == 0 and rows < 2**31):
        raise LoweringError("ragged_outer_acc requires row-major TMA outer operands")
    for f in flags:
        if dtypes[f] != "bool" or layouts[f].numel != 1:
            raise LoweringError("ragged_outer_acc flags are boolean scalars")
    for x in (*accs, *outputs):
        lay = layouts[x]
        if dtypes[x] != "float32" or tuple(lay.shape) != (m, n) or not lay.contiguous or lay.offset % 2:
            raise LoweringError("ragged_outer_acc accumulators and results are contiguous FP32 [M, N]")
    bm, bn, wm, stages = flat_tile(True, m, n, k)
    marker = lambda layout, d0, d1, stride, b0, b1: (
        f"POPS4B_TMAP({layout.root},{layout.offset},{d0},{d1},{2 * stride},{b0},{b1})")
    map_a, map_b = marker(la, m, rows, sak, 64, 64), marker(lb, n, rows, sbk, 64, 64)
    init = f"""{{{ptr(a)}, {sam}LL, {sak}LL, 0LL,
                 {ptr(b)}, {sbk}LL, {sbn}LL, 0LL,
                 nullptr, {n}LL, {m * n}LL, {m}, {n}, {k}, 1}}"""
    code = f"""{{
  const bool* const latch_oaf_[{groups}] = {{{', '.join(ptr(x) for x in flags)}}};
  const float* const latch_oai_[{groups}] = {{{', '.join(ptr(x) for x in accs)}}};
  float* const latch_oao_[{groups}] = {{{', '.join(ptr(x) for x in outputs)}}};
  popsc::BArgs g{init};
  latch_ragged::touter_acc<{groups},{bm},{bn},{wm},{stages}>({map_a}, {map_b}, g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0, latch_oaf_, latch_oai_, latch_oao_);
}}"""
    tiles = groups * math.ceil(m / bm) * math.ceil(n / bn)
    return phase("gemm", code, list(outputs), [a, b, offsets, *flags, *accs], work_items=max(1, tiles),
                 tiles=max(1, tiles), shared_bytes=stages * (bm + bn) * 128 + 1024 + 8 * (groups + 2),
                 engine="ragged_flat_tma_outer_acc_wgmma", matrix_shape=[m, n, k], groups=groups, packed_rows=rows,
                 variable_extent="K", empty_segment="adds +0 (masked) results", tma_operands=True, tma_outer=True,
                 flat_tasks=f"latch_ragged::tasks<{groups},{bm},{bn},true>({ptr(offsets)}, {m}, {n})",
                 flat_offsets=ptr(offsets), same_index_rmw=dict(zip(outputs, accs)),
                 arithmetic=("BF16 WGMMA ascending k16 blocks with zero-filled tails; FP32 accumulation and one BF16 "
                             "result rounding; exactly widened, masked by the segment flag (+0) and added in FP32 to "
                             "the segment's accumulator (bitwise the replaced cast/where/add)"),
                 runtime_rows="exact segment lengths; no per-expert capacity")


def emit(op, *, values, layouts, dtypes, names, ref, field, ident, define, target_arch, phases=None, producers=None,
         operations=None):
    from .lower import LoweringError
    target = str(field(op, "target")).removeprefix("torch.ops.")
    args = list(field(op, "args", ()))
    outputs = tuple(field(op, "outputs", ()))
    if not outputs:
        outputs = (field(op, "output", field(op, "name")),)
    reads = [ref(a, names) for a in args if ref(a, names) is not None]
    for name in outputs:
        value = values[name]
        shape = tuple(field(value, "shape"))
        define(name, shape, None, dtypes[name])

    def ptr(name):
        layout = layouts[name]
        return f"({ident(layout.root)} + {layout.offset}LL)"

    def phase(kind, code, writes, inputs, **metadata):
        return dict(kind=kind, names=list(writes), targets=[target], reads=list(inputs),
                    writes=list(writes), code=code, threads=256,
                    requires_full_grid=True, **metadata)

    if target == ORDER:
        if len(args) != 2 or len(outputs) != 3:
            raise LoweringError("ragged_order: invalid operation ABI")
        selected, groups = ref(args[0], names), args[1]
        sl = layouts[selected]
        if type(groups) is not int or not 1 <= groups <= 256 or len(sl.shape) != 2 or dtypes[selected] != "int64" or not sl.contiguous:
            raise LoweringError("ragged_order requires contiguous int64 choices and 1..256 groups")
        rows = math.prod(sl.shape)
        order, offsets, inverse = outputs
        if rows >= 2**31 or layouts[order].shape != (rows,) or layouts[inverse].shape != (rows,) or layouts[offsets].shape != (groups + 1,):
            raise LoweringError("ragged_order output extent mismatch")
        from . import accumulation_rules
        # LATCH-SELECTIVE-10 rorder: the one-pass block-scan body (identical integers; kernels/ragged_experts.cuh).
        order_body = "order_scan" if accumulation_rules.active("rorder") else "order"
        common = dict(engine="stable_integer_bucket_order", shared_bytes=1024,
                      packed_rows=rows, groups=groups, padding_rows=0,
                      arithmetic="integer counts and stable ascending-choice placement; no floating atomics")
        return [
            phase("ragged_counts", f"latch_ragged::counts<{groups}>({ptr(selected)}, {ptr(offsets)}, {rows}, smem);",
                  [offsets], [selected], work_items=1, **common),
            phase("ragged_order", f"latch_ragged::{order_body}<{groups}>({ptr(selected)}, {ptr(offsets)}, {ptr(order)}, {ptr(inverse)}, {rows}, smem);",
                  [order, inverse], [selected, offsets], work_items=groups, **common),
        ]
    if target == OUTER_ACC:
        return [_emit_outer_acc(args, outputs, layouts=layouts, dtypes=dtypes, names=names, ref=ref, ptr=ptr,
                                phase=phase, target_arch=target_arch, LoweringError=LoweringError)]
    if target not in (MM, MM_RAW, OUTER, OUTER_WIDE) or len(args) != 3 or len(outputs) != 1:
        raise LoweringError("unknown ragged product ABI")
    # LATCH-TRAIN-PRECISION: ragged_mm_raw = BF16 operands, FP32 accumulation, UNROUNDED FP32 result (opt-in training
    # precision policies). Admitted on the TMA flat body only (tflat_raw / tflat2_raw); never hosts tile consumers.
    raw = target == MM_RAW
    a, b, offsets = [ref(arg, names) for arg in args]
    name = outputs[0]
    la, lb, ls = layouts[a], layouts[b], layouts[offsets]
    outer = target in (OUTER, OUTER_WIDE)
    # Widened outer product: BF16 operands, one BF16 result rounding, stored exactly widened in FP32.
    wide = target == OUTER_WIDE
    if len(la.shape) != 2 or len(lb.shape) != (2 if outer else 3):
        raise LoweringError("ragged product ranks differ from declared ABI")
    rows, width = la.shape
    groups = ls.shape[0] - 1
    if len(ls.shape) != 1 or dtypes[offsets] != "int64" or not ls.contiguous or not 1 <= groups <= 256:
        raise LoweringError("ragged product needs contiguous group offsets")
    dtype = "bfloat16" if (wide or raw) else dtypes[name]
    if (wide or raw) and dtypes[name] != "float32":
        raise LoweringError("widened ragged_outer / ragged_mm_raw store FP32")
    if dtypes[a] != dtype or dtypes[b] != dtype or dtype not in ("float32", "bfloat16"):
        raise LoweringError("generated ragged products require matching BF16 or FP32 storage")
    if outer:
        if lb.shape[0] != rows:
            raise LoweringError("ragged_outer row extent mismatch")
        m, n, k = width, lb.shape[1], rows
        sam, sak = la.strides[1], la.strides[0]
        sbk, sbn = lb.strides
        bsb, bsc = 0, m * n
        expected = (groups, m, n)
    else:
        if lb.shape[:2] != (groups, width):
            raise LoweringError("ragged_mm group or contracted extent mismatch")
        m, n, k = rows, lb.shape[2], width
        sam, sak = la.strides
        bsb, sbk, sbn = lb.strides
        bsc = 0
        expected = (rows, n)
    if layouts[name].shape != expected or rows >= 2**31:
        raise LoweringError("ragged product output extent mismatch")
    aligned = ((sam == 1 or sak == 1) and (sbk == 1 or sbn == 1)
               and all(v % 8 == 0 for v in (sak if sam == 1 else sam,
                                            sbn if sbk == 1 else sbk, la.offset, lb.offset, bsb))
               and n % 2 == 0 and (not (wide or raw) or layouts[name].offset % 2 == 0))
    flat_meta, prefetch = {}, []
    if raw and not (dtype == "bfloat16" and aligned and target_arch == "sm_90" and _PRODUCTS.get() is not None):
        raise LoweringError("ragged_mm_raw needs aligned BF16 operands on sm_90 with segmented_products enabled")
    if dtype == "bfloat16" and aligned and target_arch == "sm_90" and _PRODUCTS.get() is not None and not wide:
        bm, bn, wm, stages = flat_tile(outer, m, n, k)
        import os
        claim_min_rows = int(os.environ.get("LATCH_CLAIM_MIN_ROWS", CLAIM_MIN_ROWS))   # screening override only
        if (_feature("claim") and int(os.environ.get("LATCH_CLAIM_WIDE", CLAIM_WIDE)) and not outer
                and (sbk != 1 or not _feature("pair"))
                and rows >= claim_min_rows * groups and k % 64 == 0):
            # Claimed input-gradient products (N-major stacked weights, never paired with the forward products): wide
            # tiles halve the operand re-reads per output (measured: K 384 x N 768 at 128 x 256, K 768 x N 384 at
            # 256 x 128). Per-element k chains are unchanged.
            if k <= 384 and n >= 512:
                # With the staged epilogue ("sepi") three stages leave room for its two 64 x 256 slabs.
                bm, bn, wm, stages = (128, 256, 2, 3) if _feature("sepi") else (128, 256, 2, 4)
            elif n <= 384 and k >= 512:
                bm, bn, wm, stages = 256, 128, 2, 3
        ak, bkc = str(sak == 1).lower(), str(sbk == 1).lower()
        prefetch = _prefetch_phase(phase, ptr, layouts, phases, producers, (a, b)) if _feature("prefetch") else []
        # flat_tma: right products whose operands are 2D tensor-map views (A K-major rows; stacked B as one 2D
        # matrix, N-major B only with whole k stages per segment); 16-byte aligned addresses and row strides.
        tma = (_feature("tma") and not outer and sak == 1 and (sam * 2) % 16 == 0
               and (la.offset * 2) % 16 == 0 and (lb.offset * 2) % 16 == 0
               and ((sbk == 1 and bsb == n * sbn and (sbn * 2) % 16 == 0)
                    or (sbn == 1 and bsb == k * sbk and (sbk * 2) % 16 == 0 and k % 64 == 0)))
        # Outer products: A [rows, M] and B [rows, N] both row-major (M-/N-contiguous) 2D maps over the packed rows.
        tma_outer = (_feature("tma") and outer and sam == 1 and sbn == 1 and (sak * 2) % 16 == 0
                     and (sbk * 2) % 16 == 0 and (la.offset * 2) % 16 == 0 and (lb.offset * 2) % 16 == 0)
        tma = tma or tma_outer
        if raw and (outer or not tma):
            raise LoweringError("ragged_mm_raw needs the TMA flat body (segmented_products with +tma and aligned "
                                "K-major operands)")
        # Feature "claim": every TMA-fed product (right, paired, outer) of this lowering takes claimed tasks; the
        # claimed bodies keep >= 3 stages so the issuing thread hands task ids over before a CTA barrier.
        # Structural admission: claiming pays where tasks are long (>= CLAIM_MIN_ROWS packed rows per segment on
        # average; B32 x 1024 TinyMoE: 4,096); short products (B1/B2: 128-256 rows per expert) keep the static round
        # robin, whose first tasks need no counter round trip. Outer results must be contiguous [M, N] per segment.
        claim = (tma and not raw and _feature("claim") and stages >= 3 and rows >= claim_min_rows * groups
                 and (not outer or (layouts[name].contiguous and layouts[name].offset % 2 == 0)))
        import os
        # Screening only: LATCH_CLAIM_HOT overrides CLAIM_HOT (recorded in the generated source's template arguments).
        claim_hot = int(os.environ.get("LATCH_CLAIM_HOT", CLAIM_HOT))
        hbm, hbn, hwm, hst = CLAIM_HOT_TILE
        hot_args = f",{claim_hot},{hbm},{hbn},{hwm},{hst}" if claim_hot else ""
        hot_ring = hst * (hbm + hbn) * 128 if claim_hot else 0
        pair_stages = 3 if claim and _feature("sepi") and bm == 128 and wm == 2 else stages
        if tma:
            marker = lambda layout, d0, d1, stride, b0, b1: (
                f"POPS4B_TMAP({layout.root},{layout.offset},{d0},{d1},{2 * stride},{b0},{b1})")
            if outer:
                map_a, map_b = marker(la, m, rows, sak, 64, 64), marker(lb, n, rows, sbk, 64, 64)
            else:
                map_a = marker(la, k, rows, sam, 64, bm)
                map_b = (marker(lb, k, groups * n, sbn, 64, bn) if sbk == 1 else marker(lb, n, groups * k, sbk, 64, 64))
        out_ptr = f"reinterpret_cast<s101h::bf16*>({ptr(name)})" if raw else ptr(name)
        init = f"""{{{ptr(a)}, {sam}LL, {sak}LL, 0LL,
                 {ptr(b)}, {sbk}LL, {sbn}LL, {bsb}LL,
                 {out_ptr}, {n}LL, {bsc}LL, {m}, {n}, {k}, 1}}"""
        suffix = "_raw" if raw else ""
        tmpl = f"{groups},{bm},{bn},{wm},{ak},{bkc},{stages},{str(outer).lower()}"
        # Paired products (flat2): the same A value, offsets, extents, strides and tile, differing only in B and C.
        key = [target, _canonical(a, values, field, ref, names, operations), offsets, list(la.shape),
               list(la.strides), list(lb.shape), list(lb.strides), lb.offset % 8, list(expected), dtype, tmpl, tma]
        pairing = phases is not None and _feature("pair") and not tma_outer
        pending = _pending(phases) if pairing else {}
        slot = repr(key)
        if slot in pending:
            first = pending.pop(slot)
            first["code"] = (f"""{{
  popsc::BArgs g0{first["flat_init"]};
  popsc::BArgs g1{init};
  latch_ragged::tflat2_claim<{groups},{bm},{bn},{wm},{bkc},{pair_stages},{CLAIM_STATIC0}>({map_a}, {first["flat_map_b"]}, {map_b}, g0, g1, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0, {CLAIM_SLOT});
}}""" if tma and claim else f"""{{
  popsc::BArgs g0{first["flat_init"]};
  popsc::BArgs g1{init};
  latch_ragged::tflat2{suffix}<{groups},{bm},{bn},{wm},{bkc},{stages}>({map_a}, {first["flat_map_b"]}, {map_b}, g0, g1, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0);
}}""" if tma else f"""{{
  popsc::BArgs g0{first["flat_init"]};
  popsc::BArgs g1{init};
  latch_ragged::flat2<{tmpl}>(g0, g1, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0);
}}""")
            first["names"].append(name)
            first["writes"].append(name)
            first["targets"].append(target)
            first["reads"] = list(dict.fromkeys(first["reads"] + reads))
            first["shared_bytes"] = (pair_stages if tma and claim else stages) * (bm + 2 * bn) * 128 + 1024 + 8 * (groups + 2) + (
                sepi_bytes(groups, bn, 2) if tma and claim and _feature("sepi") else 0)
            first["engine"] = ("ragged_flat2_tma_claim_wgmma" if tma and claim else
                               ("ragged_flat2_tma_wgmma" if tma else "ragged_flat2_wgmma") + ("_raw" if raw else ""))
            first["flat_paired"] = True
            first["arithmetic"] += "; paired with a second product sharing each A stage (same per-output arithmetic)"
            return prefetch + [first]
        code = (f"""{{
  popsc::BArgs g{init};
  latch_ragged::touter_claim<{groups},{bm},{bn},{wm},{stages},{CLAIM_STATIC0}{hot_args}>({map_a}, {map_b}, g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0, {CLAIM_SLOT});
}}""" if tma_outer and claim else f"""{{
  popsc::BArgs g{init};
  latch_ragged::touter<{groups},{bm},{bn},{wm},{stages}>({map_a}, {map_b}, g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0);
}}""" if tma_outer else f"""{{
  popsc::BArgs g{init};
  latch_ragged::tflat_claim<{groups},{bm},{bn},{wm},{bkc},{stages},{CLAIM_STATIC0}>({map_a}, {map_b}, g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0, {CLAIM_SLOT});
}}""" if tma and claim else f"""{{
  popsc::BArgs g{init};
  latch_ragged::tflat{suffix}<{groups},{bm},{bn},{wm},{bkc},{stages}>({map_a}, {map_b}, g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0);
}}""" if tma else f"""{{
  popsc::BArgs g{init};
  latch_ragged::flat<{tmpl}>(g, {ptr(offsets)}, {rows}, smem, /*XMAT_ROT*/0);
}}""")
        flat_meta = dict(flat_pair_key=key, flat_init=init, **({"flat_map_b": map_b, "tma_operands": True} if tma else {}),
                         **({"flat_claim": True} if tma and claim else {}),
                         **({"claim_outer": dict(key=repr([offsets, m, n, rows, list(la.strides), list(lb.strides), la.offset % 8,
                                                          lb.offset % 8, groups, bm, bn, wm, stages]),
                                                 tmpl=f"{groups},{bm},{bn},{wm},{stages}", map_a=map_a, map_b=map_b,
                                                 init=init, offsets=ptr(offsets), rows=rows, hot=hot_args)}
                            if tma_outer and claim else {}),
                         **({"tma_outer": True} if tma_outer else {}),
                         flat_tasks=f"latch_ragged::tasks<{groups},{bm},{bn},{str(outer).lower()}>({ptr(offsets)}, {m}, {n})",
                         flat_offsets=ptr(offsets),
                         **({"flat_consumable": True, "flat_numel": rows * n} if not outer and bsc == 0 and not raw
                            else {}))
        shared = max(stages * (bm + bn) * 128,
                     hot_ring if tma_outer and claim else 0) + 1024 + 8 * (groups + 2) + (
            sepi_bytes(groups, bn, 1, bm) if tma and claim and not outer and _feature("sepi") else 0)
        engine = ("ragged_flat_tma_outer_claim_wgmma" if tma_outer and claim else
                  "ragged_flat_tma_claim_wgmma" if tma and claim else
                  ("ragged_flat_tma_outer_wgmma" if tma_outer else "ragged_flat_tma_wgmma" if tma else "ragged_flat_wgmma")
                  + ("_raw" if raw else ""))
        arithmetic = ("BF16 WGMMA ascending k16 blocks with zero-filled tails; FP32 accumulation and one BF16 result "
                      "rounding (bitwise the ragged_pops4b_wgmma body; operand stages prefetched across tasks)")
        if raw:
            arithmetic = ("BF16 WGMMA ascending k16 blocks with zero-filled tails; FP32 accumulation; the FP32 "
                          "accumulators are stored unrounded (ragged_mm_raw); operand stages prefetched across tasks")
            engine = engine + "_raw"
    elif dtype == "bfloat16" and aligned and target_arch == "sm_90":
        bm, bn, wm, stages = (128, 128, 2, 3) if outer else (64, 128, 1, 3)
        ak, bkc = str(sak == 1).lower(), str(sbk == 1).lower()
        code = f"""{{
  popsc::BArgs g{{{ptr(a)}, {sam}LL, {sak}LL, 0LL,
                 {ptr(b)}, {sbk}LL, {sbn}LL, {bsb}LL,
                 {f"reinterpret_cast<s101h::bf16*>({ptr(name)})" if wide else ptr(name)}, {n}LL, {bsc}LL, {m}, {n}, {k}, 1}};
  latch_ragged::product<{groups},{bm},{bn},{wm},{ak},{bkc},{stages},{str(outer).lower()}{",true" if wide else ""}>(g, {ptr(offsets)}, {rows}, smem);
}}"""
        shared = stages * (bm + bn) * 128 + 1024
        engine = "ragged_pops4b_wgmma"
        arithmetic = "BF16 WGMMA ascending k16 blocks with zero-filled tails; FP32 accumulation and one BF16 result rounding"
    else:
        bm = bn = 16
        scalar = "s101h::bf16" if dtype == "bfloat16" else "float"
        code = f"""latch_ragged::simt<{scalar},{groups},{str(outer).lower()}{",float" if wide else ""}>(
  {ptr(a)}, {ptr(b)}, {ptr(offsets)}, {ptr(name)}, {rows}, {m}, {n}, {k},
  {sam}LL, {sak}LL, {sbk}LL, {sbn}LL, {bsb}LL, {bsc}LL, smem);"""
        shared, engine = 2048, "ragged_simt16"
        arithmetic = "ascending-K FP32 fmaf; BF16 operands exactly widened and result rounded once when applicable"
    if wide:
        arithmetic += "; stored exactly widened to FP32 (the bits of a separate BF16 -> FP32 cast)"
    # A bound, not padded runtime work: actual tile counts use the segment sizes.
    tiles = groups * math.ceil(m / bm) * math.ceil(n / bn) if outer else (math.ceil(rows / bm) + groups - 1) * math.ceil(n / bn)
    result = phase("gemm", code, [name], reads, work_items=max(1, tiles), tiles=max(1, tiles),
                   shared_bytes=shared, engine=engine, matrix_shape=[m, n, k],
                   groups=groups, packed_rows=rows, variable_extent="K" if outer else "M",
                   empty_segment="zero weight gradient" if outer else "no row tasks",
                   arithmetic=arithmetic, runtime_rows="exact segment lengths; no per-expert capacity", **flat_meta)
    # Pairing defers the first product to its partner's position (its inputs stay available; nothing in between
    # reads its result), so the pair is emitted where both B operands exist.
    if flat_meta and phases is not None and _feature("pair") and not flat_meta.get("tma_outer") and _partner_ahead(
            op, operations, field, ref, names, values, flat_meta["flat_pair_key"][1], offsets, b, name):
        _pending(phases)[repr(flat_meta["flat_pair_key"])] = result
        return prefetch
    return prefetch + [result]


def merge_claimed_outer(phases, members):
    """Feature "claim" (+ CLAIM_MERGE): two claimed weight-gradient phases of one interval with the same segment offsets,
    extents, operand strides and tile take one queue (kernels/ragged_claim.cuh touter2_claim), so both products' largest
    expert tiles are claimed first. The first phase's code computes both results; the second phase keeps its manifest
    record (same writes, same interval) with an empty body. Returns [(first, second)] merged pairs."""
    import os
    if not int(os.environ.get("LATCH_CLAIM_MERGE", CLAIM_MERGE)):
        return []
    pending, merged = {}, []
    for i in members:
        info = phases[i].get("claim_outer")
        if not info or phases[i].get("claim_merged_into") is not None:
            continue
        j = pending.pop(info["key"], None)
        if j is None:
            pending[info["key"]] = i
            continue
        a, b = phases[j]["claim_outer"], info
        phases[j]["code"] = (f"""{{
  popsc::BArgs g0{a["init"]};
  popsc::BArgs g1{b["init"]};
  latch_ragged::touter2_claim<{a["tmpl"]},{CLAIM_STATIC0}{a["hot"]}>({a["map_a"]}, {a["map_b"]}, {b["map_a"]}, {b["map_b"]}, g0, g1, {a["offsets"]}, {a["rows"]}, smem, /*XMAT_ROT*/0, {CLAIM_SLOT});
}}""")
        phases[j]["names"] = phases[j]["names"]          # writes stay with their own phase records
        phases[j]["claim_merged_with"] = phases[i]["names"][0]
        phases[i]["code"] = f"/* {phases[i]['names'][0]}: computed by the merged claimed product of phase {j} */"
        phases[i]["claim_merged_into"] = j
        phases[i]["flat_claim"] = False
        merged.append((j, i))
    return merged


def sepi_bytes(groups, bn, products, bm=128):
    """Shared memory of the staged epilogue (kernels/ragged_claim.cuh): table alignment slack plus the two warpgroups'
    bm / 2-row slabs per product at a padded pitch."""
    return 128 + ((8 * (groups + 2) + 127) & ~127) + 2 * products * (bm // 2) * (bn * 2 + 16)
