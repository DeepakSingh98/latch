"""Indexed backward placements evaluated at their readers instead of stored expanded (elementwise_rules ``indexed``).

``select_backward(g, sizes, dim, index)`` and ``slice_backward(g, sizes, dim, start, end, step)`` place a gradient
into a zero tensor of the forward input's shape. Autograd emits one such placement per forward ``x[:, j]`` /
``x[:, a:b]`` read, and the reader then adds the k placements together. Stored, each placement writes the full
expanded FP32 tensor (mostly +0) and every reader reads all of them back: for a [T, k, D] stack, k placements of
k*T*D elements written and read, instead of k*T*D gradient elements read once.

A placement phase is replaced by its own element expression,

    ((c == index) ? g[<grad index of i>] : 0.0f)

substituted at each identity read ``v_x[i]`` / ``v_x[(i)]`` of its readers. The expression is taken verbatim from
the phase's existing scalar form (the form the 16-byte vector copy is already bit-identical to), so every reader
element sees exactly the value it loaded before: the same gradient element or the same +0.0f. Reader arithmetic
(the order of the additions of the placements, their roundings and the signs of zeros) is unchanged. Nothing is
accumulated, so there are no atomics and no reassociation; a many-to-one placement does not exist here (each output
element has at most one source element by construction of select/slice_backward).

Admission (otherwise the placement stays stored, with the refusal recorded):

* the producer is a single-result ``select_backward``/``slice_backward`` phase with a recognized scalar form whose
  expression references only its own operand pointers and the flat index;
* its result is its own contiguous root, not returned, committed, placed or written in place;
* every reader is an elementwise body or a row group of the same extent, reads the value directly (not through a
  view), and every textual occurrence of the value is an identity read;
* no phase between the producer and a reader writes the root of an operand (the operand keeps its value until the
  reader), and no reader writes an operand root (no in-place hazard at the remapped index).

The pass runs after row fusion and the existing exact rewrites, before store elision, vector code generation,
barrier scheduling and arena assignment, so the new reads are planned like any other.

Enabled by the ``indexed`` elementwise rule; both placement kinds (``select_backward`` and ``slice_backward``) are
admitted. Off by default: every existing program is unchanged.
"""
from __future__ import annotations

import re

KINDS = ("select_backward", "slice_backward")
# Identifiers that may appear in a placement's scalar element expression besides operand pointers and the index.
_ALLOWED_WORDS = {"i", "long", "unsigned", "int", "float", "static_cast", "int64_t", "__nv_bfloat16"}
_WORD = re.compile(r"(?<![\w.])([A-Za-z_]\w*)")


def enabled():
    from .elementwise_rules import active
    return active("indexed")


def _placement_kind(phase):
    targets = phase.get("targets") or []
    if len(targets) != 1 or len(phase.get("names", ())) != 1 or len(phase.get("writes", ())) != 1:
        return None
    base = str(targets[0]).removeprefix("aten.").split(".")[0]
    return base if base in KINDS else None


def _scalar_loop(code, out):
    """The text of the scalar loop that stores ``out[i]`` (the vector copy's fallback branch, or the whole code)."""
    store = re.search(rf"(?<![\w.]){re.escape(out)}\[i\] = ", code)
    if store is None:
        return None
    start = code.rfind("for (long long i =", 0, store.start())
    if start < 0:
        return None
    return code[start:]


def element_expression(phase, ident):
    """(expression, operand pointer idents) of the placement's element at flat index ``i``, or (None, reason)."""
    out = ident(phase["writes"][0])
    code = phase.get("code")
    if not isinstance(code, str):
        return None, "no scalar code"
    loop = _scalar_loop(code, out)
    if loop is None:
        return None, "no scalar store"
    store = re.search(rf"(?<![\w.]){re.escape(out)}\[i\] = ([^;]+);", loop)
    if store is None:
        return None, "unparsed scalar store"
    rhs = store.group(1).strip()
    defs = dict(re.findall(r"const long long (\w+) = ([^;]+);", loop[:store.start()]))
    # every local is a coordinate of the flat index; substitute (deepest names first)
    for name in sorted(defs, key=len, reverse=True):
        if name == "i" or not re.fullmatch(r"c_?|[a-z]\w*_?", name):
            return None, "unexpected local"
        rhs = re.sub(rf"(?<![\w.]){re.escape(name)}(?!\w)", f"({defs[name].strip()})", rhs)
    pointers = {ident(r) for r in phase.get("reads", ())}
    for word in _WORD.findall(rhs):
        if word in _ALLOWED_WORDS or word in pointers:
            continue
        return None, f"unexpected identifier {word}"
    if not any(p in rhs for p in pointers):
        return None, "no operand read"
    if rhs.count("?") != 1:
        return None, "not a single guarded placement"
    return f"({rhs})", None


def place_at_readers(phases, layouts, *, kept_roots, numel, ident, dtypes=None, retain_all=False):
    """Rewrite admitted placements into their readers. Returns (records, dead value names, None).

    The third item (helper text) is always None: the substituted expressions need no helpers. ``dtypes`` is accepted
    for the caller's uniform signature and not used."""
    if retain_all or not enabled():
        return [], set(), None
    root = lambda n: layouts[n].root if n in layouts else n
    records, drop = [], set()
    for k, ph in enumerate(phases):
        kind = _placement_kind(ph)
        if kind is None:
            continue
        v = ph["writes"][0]
        lay = layouts.get(v)
        refuse = None
        if lay is None or lay.root != v or not lay.contiguous or lay.offset or v in kept_roots:
            refuse = "kept or non-root result"
        expr, operands = None, list(ph["reads"])
        if refuse is None:
            expr, why = element_expression(ph, ident)
            if expr is None:
                refuse = why
        readers = [j for j, q in enumerate(phases) if j != k and any(root(r) == v for r in q["reads"])]
        if refuse is None and not readers:
            refuse = "no reader"
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
            elif {root(w) for w in q["writes"]} & operand_roots:
                refuse = "reader writes an operand root"
            elif any(k < w < j for w, wq in enumerate(phases) for n in wq["writes"] if root(n) in operand_roots):
                refuse = "operand rewritten before a reader"
            else:
                texts = [q[key] for key in ("code", "code_r1") if isinstance(q.get(key), str)] + list(q.get("body") or [])
                joined = "\n".join(texts)
                total, identity = len(name_re.findall(joined)), len(identity_re.findall(joined))
                if not total or total != identity:
                    refuse = "non-identity read"
        if refuse:
            records.append({"value": v, "kind": kind, "refused": refuse})
            continue

        def subst(text):
            return identity_re.sub(lambda _m: expr, text)
        for j in readers:
            q = phases[j]
            for key in ("code", "code_r1"):
                if isinstance(q.get(key), str):
                    q[key] = subst(q[key])
            if q.get("body"):
                q["body"] = [subst(line) for line in q["body"]]
            q["reads"] = list(dict.fromkeys([r for r in q["reads"] if root(r) != v] + operands))
            q.pop("row_parts", None)
            q.setdefault("indexed_backward_inputs", []).append(v)
        drop.add(k)
        records.append({"value": v, "kind": kind, "shape": list(lay.shape), "elements_not_stored": lay.numel,
                        "readers": [phases[j]["names"][0] for j in readers], "operands": operands})
    dead = {phases[k]["writes"][0] for k in drop}
    if drop:
        phases[:] = [ph for k, ph in enumerate(phases) if k not in drop]
    return records, dead, None
