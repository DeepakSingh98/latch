"""Depth-read region recognition and two exact graph rules (graph_rules ``depth_reads``).

A *depth read* mixes ``n`` earlier source activations ``S_i`` (each ``[T, D]``) with learned per-source scores:

    r_i      = rsqrt(mean(S_i * S_i, -1, keepdim) + eps)                      FP32 [T, 1]
    p_i      = mm(_to_copy(S_i, bf16), t(_to_copy(q, bf16)))                   BF16 [T, 1] (BF16 GEMV)
    l_i      = p_i * r_i                                                       FP32 [T, 1]
    w        = softmax(cat(l_1..l_n) / c) * gains[:n]                          FP32 [T, n]
    alpha    = w * cat(r_1..r_n)
    out      = S_1 * alpha[:, 0:1] + S_2 * alpha[:, 1:2] + ...                 FP32 [T, D]

followed by an RMSNorm of ``out``. It is recognized from graph structure only: a softmax over the last axis whose
input is a constant division of a concatenation of ``[T, 1]`` products of a ``[T, 1]`` BF16 matrix product and a
reciprocal square root. Model names, interval ids and phase indices are never consulted.

Two rules run on the recognized reads (``RULES``); both keep every value bit for bit:

* ``remat`` -- the RMSNorm's FP32 normalized value ``N = M * r`` is recomputed in front of each backward reader as a
  clone of the same operation (``<N>__nsr_rm<k>``) instead of being stored for the backward;
* ``fbatch`` -- up to ``FBATCH_MAX`` forward logit GEMVs that share one BF16 source operand become one stacked product
  (``<mm>__nsr_fq`` / ``fb`` / ``fp``) whose columns are read as views.

Every operation's ``inputs``/``reads``/``dependencies`` lists are normalized (duplicates removed, order kept) whenever
the rules run, and the report is recorded in the graph metadata (``depth_read``). Without ``depth_reads`` the graph
is returned unchanged.
"""
from __future__ import annotations

import copy

VERSION = 1
RULES = frozenset({"remat", "fbatch"})
SOFTMAX = ("aten._softmax.default",)
CAT = "aten.cat.default"
MUL = "aten.mul.Tensor"
DIV = "aten.div.Tensor"
MM = "aten.mm.default"
CAST = "aten._to_copy.default"
RSQRT = "aten.rsqrt.default"
VIEWS = {"aten.view.default", "aten._unsafe_view.default", "aten.t.default", "aten.unsqueeze.default",
         "aten.reshape.default", "aten.permute.default", "aten.transpose.int", "aten.squeeze.dim",
         "aten.expand.default", "aten.alias.default"}


def enabled():
    from .graph_rules import active
    return active("depth_reads")


def _target(op):
    return str(op.get("target") or "").removeprefix("torch.ops.")


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _reads(op):
    return list(dict.fromkeys(op.get("reads") or op.get("inputs") or ()))


def _replace(obj, old, new):
    if isinstance(obj, dict):
        if set(obj) == {"ref"} and obj["ref"] in old:
            return {"ref": old[obj["ref"]]}
        return {k: _replace(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace(v, old, new) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_replace(v, old, new) for v in obj)
    return obj


def _output_names(data):
    """Every value name in the IR's (nested) outputs structure, plus values carrying output roles (as gather_cast)."""
    names = set()

    def walk(item):
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            if set(item) == {"ref"}:
                names.add(item["ref"])
            else:
                for v in item.values():
                    walk(v)
        elif isinstance(item, (list, tuple)):
            for v in item:
                walk(v)
    walk(data.get("outputs", {}))
    names |= {n for n, v in (data.get("values") or {}).items() if v.get("output_roles")}
    return names


class _Graph:
    def __init__(self, data):
        self.data = data
        self.ops = data["operations"]
        self.values = data["values"]
        self.producer, self.readers = {}, {}
        for op in self.ops:
            for name in op.get("outputs") or ():
                self.producer[name] = op
            for name in _reads(op):
                self.readers.setdefault(name, []).append(op)
        self.outputs = _output_names(data)
        self.inputs = set(data.get("inputs") or ())

    def shape(self, name):
        return tuple((self.values.get(name) or {}).get("shape") or ())

    def dtype(self, name):
        return (self.values.get(name) or {}).get("dtype")

    def op(self, name):
        return self.producer.get(name)

    def through_views(self, name):
        """The non-view producer value of name (views followed back to their source)."""
        seen = 0
        while seen < 32:
            op = self.producer.get(name)
            if op is None or _target(op) not in VIEWS:
                return name
            src = _ref((op.get("args") or [None])[0])
            if src is None:
                return name
            name, seen = src, seen + 1
        return name


def depth_reads(g):
    """Recognized depth reads: [{softmax, terms: [{logit, product, cast, source, stat}]}] (structure only)."""
    found = []
    for op in g.ops:
        if _target(op) not in SOFTMAX:
            continue
        args = op.get("args") or []
        x = _ref(args[0]) if args else None
        shape = g.shape(op["outputs"][0])
        if x is None or len(shape) != 2 or g.dtype(op["outputs"][0]) != "float32":
            continue
        T, n = shape
        div = g.op(x)
        if div is None or _target(div) != DIV:
            continue
        cat_name = _ref((div.get("args") or [None])[0])
        cat = g.op(cat_name) if cat_name else None
        if cat is None or _target(cat) != CAT or not isinstance(cat["args"][0], (list, tuple)):
            continue
        parts = [_ref(a) for a in cat["args"][0]]
        if len(parts) != n or None in parts:
            continue
        terms = []
        for part in parts:
            mul = g.op(part)
            if mul is None or _target(mul) != MUL or g.shape(part) != (T, 1):
                break
            a, b = (_ref(v) for v in (mul.get("args") or [None, None])[:2])
            prod = stat = None
            for u, v in ((a, b), (b, a)):
                pu, pv = g.op(g.through_views(u)) if u else None, g.op(v) if v else None
                if pu is not None and _target(pu) == MM and pv is not None and _target(pv) == RSQRT:
                    prod, stat = pu, v
            if prod is None:
                break
            lhs = g.through_views(_ref(prod["args"][0]))
            cop = g.op(lhs)
            if cop is None or _target(cop) != CAST or g.dtype(lhs) != "bfloat16":
                break
            source = _ref(cop["args"][0])
            if source is None or len(g.shape(source)) != 2 or g.shape(source)[0] != T:
                break
            query = g.through_views(_ref(prod["args"][1]))
            terms.append(dict(logit=part, product=prod["outputs"][0], cast=lhs, source=source, stat=stat, query=query))
        else:
            found.append(dict(softmax=op["outputs"][0], sources=n, rows=T, terms=terms))
    return found


def _alpha(g, read):
    """alpha = mul(mul(softmax, gains[:n]), cat(r_1..r_n)): the forward chain from the read's softmax (or None)."""
    T, n = read["rows"], read["sources"]
    alphas = set()
    for w_op in g.readers.get(read["softmax"], ()):
        w = w_op["outputs"][0]
        if _target(w_op) != MUL or g.shape(w) != (T, n):
            continue
        for a_op in g.readers.get(w, ()):
            if _target(a_op) != MUL or g.shape(a_op["outputs"][0]) != (T, n):
                continue
            others = [_ref(x) for x in a_op["args"][:2] if _ref(x) != w]
            if len(others) == 1 and others[0] and g.op(others[0]) is not None and _target(g.op(others[0])) == CAT:
                alphas.add(a_op["outputs"][0])
    return alphas.pop() if len(alphas) == 1 else None


def norm_tail(g, read):
    """The read's mixture M, its RMSNorm statistic r, N = M * r and output Y = N * w (dict) or None.

    M = sum_i S_i * slice(alpha, i) (left-nested adds), then r = rsqrt(mean(M * M) + eps), N = mul(M, r) and
    Y = mul(N, w) for a 1-D weight w of width D (the RMSNorm that follows every depth read)."""
    alpha = _alpha(g, read)
    if alpha is None:
        return None
    T = read["rows"]
    per = [op["outputs"][0] for op in g.readers.get(alpha, ()) if _target(op) == "aten.slice.Tensor"]
    if read["sources"] == 1:
        per.append(alpha)
    sources = {t["source"] for t in read["terms"]}
    fmix = []
    for sl in per:
        for op in g.readers.get(sl, ()):
            args = [_ref(a) for a in op["args"][:2]]
            if _target(op) == MUL and sl in args and ({a for a in args if a != sl} & sources):
                fmix.append(op["outputs"][0])
    if not fmix:
        return None
    order = {op["name"]: i for i, op in enumerate(g.ops)}
    cur = min(fmix, key=lambda v: order[g.op(v)["name"]])
    D = g.shape(cur)[1] if len(g.shape(cur)) == 2 else None
    for _ in range(read["sources"] + 1):
        sq = [op for op in g.readers.get(cur, ()) if _target(op) == MUL and [_ref(a) for a in op["args"][:2]] == [cur, cur]]
        if sq:
            break
        nxt = [op for op in g.readers.get(cur, ()) if _target(op) == "aten.add.Tensor" and g.shape(op["outputs"][0]) == (T, D)
               and any(_ref(a) in fmix for a in op["args"][:2] if _ref(a) != cur)]
        if len(nxt) != 1:
            return None
        cur = nxt[0]["outputs"][0]
    else:
        return None
    M = cur
    N = r = Y = None
    for op in g.readers.get(M, ()):
        args = [_ref(a) for a in op["args"][:2]]
        if _target(op) == MUL and args[0] == M and args[1] and g.op(args[1]) is not None \
                and _target(g.op(args[1])) == RSQRT and g.shape(args[1]) == (T, 1):
            N, r = op["outputs"][0], args[1]
    if N is None:
        return None
    for op in g.readers.get(N, ()):
        args = [_ref(a) for a in op["args"][:2]]
        if _target(op) == MUL and args[0] == N and args[1] and g.shape(args[1]) == (D,):
            Y = op["outputs"][0]
    if Y is None:
        return None
    return dict(mixture=M, rstd=r, normalized=N, output=Y, alpha=alpha)


def rewrite(graph):
    """(graph, report); the input is never mutated. Unchanged (same object, report None) when disabled."""
    if not enabled():
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or not isinstance(values, dict):
        return graph, None
    g = _Graph(data)
    reads = depth_reads(g)
    for read in reads:
        read["tail"] = norm_tail(g, read)
    report = {"version": VERSION, "kinds": sorted(RULES), "depth_reads": len(reads),
              "source_terms": sum(r["sources"] for r in reads)}
    mapping = {}
    remats = remat(g, reads, report)
    report["tails"] = sum(1 for r in reads if r.get("tail"))
    # Normalize every operation (argument copies; de-duplicated inputs/reads/dependencies in their order).
    new_ops = []
    for op in ops:
        op = dict(op)
        op["args"] = _replace(op.get("args"), mapping, None)
        op["kwargs"] = _replace(op.get("kwargs"), mapping, None)
        for key in ("inputs", "reads", "dependencies"):
            if isinstance(op.get(key), (list, tuple)):
                op[key] = list(dict.fromkeys(mapping.get(x, x) for x in op[key]))
        new_ops.append(op)
    if remats:
        new_ops = _apply_remats(new_ops, values, remats, mapping)
    new_ops = forward_batches(new_ops, values, reads, report, data)
    data["operations"] = new_ops
    data.setdefault("metadata", {})["depth_read"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


def _value(g, name, shape, dtype):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return {"name": name, "shape": list(shape), "dtype": dtype, "strides": list(reversed(strides)), "role": "temporary",
            "parameter": None, "alias_of": None, "storage_offset": 0, "producer": name, "output_roles": []}


def _new_op(name, target, args, kwargs, inputs_set):
    refs = []

    def walk(a):
        if isinstance(a, dict) and set(a) == {"ref"}:
            refs.append(a["ref"])
        elif isinstance(a, (list, tuple)):
            for x in a:
                walk(x)
        elif isinstance(a, dict):
            for x in a.values():
                walk(x)
    walk(args)
    walk(kwargs)
    refs = list(dict.fromkeys(refs))
    return {"name": name, "target": target, "args": args, "kwargs": kwargs, "inputs": refs, "outputs": [name],
            "dependencies": [r for r in refs if r not in inputs_set], "reads": refs, "writes": [name],
            "reduction_axes": [], "synchronization": "global_phase_boundary", "effect": "pure"}


def remat(g, reads, report):
    """Rule ``remat``: the RMSNorm's FP32 normalized value N = mul(M, r) is recomputed at each backward reader.

    The forward stores N (FP32 [T, D]) only for the backward (the norm-weight gradient reads ``dY * N``), while the
    backward already reads M and r for the RMSNorm derivative. Each backward reader instead reads a clone of the same
    operation ``mul(M, r)`` placed immediately before it (same operands, one IEEE multiply, --fmad=false), so N is
    neither written nor read back. The forward reader (Y = N * w) keeps N in registers.
    Returns {reader op name: [(N, clone op, clone value)]}."""
    out = {}
    for read in reads:
        tail = read.get("tail")
        if not tail:
            continue
        N, Y = tail["normalized"], tail["output"]
        if N in g.outputs:
            continue
        producer = g.op(N)
        for k, op in enumerate(g.readers.get(N, ())):
            if Y in (op.get("outputs") or ()):
                continue
            clone = f"{N}__nsr_rm{k}"
            nop = dict(copy.deepcopy(producer), name=clone, outputs=[clone], writes=[clone])
            out.setdefault(op["name"], []).append((N, nop, clone))
    report["remat"] = sum(len(v) for v in out.values())
    return out


FBATCH_MAX = 8   # queries per stacked forward product (N = 8 is the shape the edge mma.sync body already serves)


def forward_batches(ops, values, reads, report, data):
    """Rule ``fbatch``: the forward logit GEMVs of one source for several reads become one product.

    Read j's logit for source i is ``mm(C_i, t(unsqueeze(cast(q_j), 0)))`` ([T, D] x [D, 1], C_i the source's BF16
    copy). The selected body (``popsc_bf16_edge``: mma.sync m16n8k16, one FP32 accumulator
    per element from +0, ascending k16 blocks, one RN rounding) computes every output element from its own A row and
    B column only. Stacking the g <= 8 queries of the reads that use C_i as the columns of B,
    ``P = mm(C_i, t(cat([unsqueeze(cast(q_j), 0)], 0)))`` ([T, D] x [D, g]), and reading logit j as the column view
    ``P[:, r:r+1]`` keeps every element's instruction sequence and rounding while C_i is streamed once. The query
    casts depend only on parameters, so they are hoisted to the first member's position (stable order otherwise).
    The selected engine of every stacked product must be re-checked in the manifest (bitwise class mma_sync_k16)."""
    producer = {o: op for op in ops for o in op.get("outputs") or ()}
    inputs_set = set(data.get("inputs") or ())

    def shape(v):
        return tuple((values.get(v) or {}).get("shape") or ())

    def dtype(v):
        return (values.get(v) or {}).get("dtype")
    T = reads[0]["rows"] if reads else None
    by_c, order = {}, {op["name"]: i for i, op in enumerate(ops)}
    for op in ops:
        if _target(op) != MM:
            continue
        c, b = (_ref(x) for x in op["args"][:2])
        out = op["outputs"][0]
        if c is None or b is None or shape(out) != (T, 1) or dtype(out) != "bfloat16" or dtype(c) != "bfloat16":
            continue
        D = shape(c)[1] if len(shape(c)) == 2 else None
        tb = producer.get(b)
        un = producer.get(_ref(tb["args"][0])) if tb is not None and _target(tb) == "aten.t.default" else None
        qc = producer.get(_ref(un["args"][0])) if un is not None and _target(un) == "aten.unsqueeze.default" \
            and un["args"][1] == 0 else None
        if qc is None or _target(qc) != CAST or shape(qc["outputs"][0]) != (D,) or dtype(qc["outputs"][0]) != "bfloat16":
            continue
        if not set(qc.get("reads") or ()) <= inputs_set or (values.get(out) or {}).get("output_roles"):
            continue
        by_c.setdefault(c, []).append(dict(mm=op["name"], t=tb["name"], un=un["name"], qc=qc["name"], D=D))
    insert_at, removed, query_ops, groups, members_n = {}, set(), set(), 0, 0
    for c, members in by_c.items():
        members.sort(key=lambda m: order[m["mm"]])
        for i in range(0, len(members), FBATCH_MAX):
            chunk = members[i:i + FBATCH_MAX]
            if len(chunk) < 2:
                continue
            g, D = len(chunk), chunk[0]["D"]
            groups += 1
            members_n += g
            tag = f"{chunk[0]['mm']}__nsr_f"
            Q, B, P = tag + "q", tag + "b", tag + "p"
            values[Q] = _value(None, Q, (g, D), "bfloat16")
            values[B] = {**_value(None, B, (D, g), "bfloat16"), "strides": [1, D], "alias_of": Q}
            values[P] = _value(None, P, (T, g), "bfloat16")
            for m in chunk:
                for key in ("qc", "un", "t"):
                    query_ops.add(m[key])
            new = [_new_op(Q, CAT, [[{"ref": m["un"]} for m in chunk], 0], {}, inputs_set),
                    dict(_new_op(B, "aten.t.default", [{"ref": Q}], {}, inputs_set), writes=[], synchronization="view"),
                    _new_op(P, MM, [{"ref": c}, {"ref": B}], {}, inputs_set)]
            for r, m in enumerate(chunk):
                values[m["mm"]] = {**values[m["mm"]], "strides": [g, 1], "alias_of": P, "storage_offset": r}
                new.append(dict(_new_op(m["mm"], "aten.slice.Tensor", [{"ref": P}, 1, r, r + 1], {}, inputs_set),
                                writes=[], synchronization="view"))
                removed.add(m["mm"])
            insert_at[chunk[0]["mm"]] = new
    report["fbatch_groups"] = groups
    report["fbatch_products"] = members_n
    if not groups:
        return ops
    # CSE can share a query chain between source groups/chunks and unbatched products. Source-group iteration
    # order is not execution order: assigning the shared chain to the first group visited can put it after an
    # earlier reader. Keep every original query op once, hoisting only these admitted parameter-only chains
    # when a new batch first needs them. All ordinary ops keep their original order and expressions.
    out, defined, emitted_queries = [], set(inputs_set), set()

    def emit(op):
        if op["name"] in emitted_queries:
            return
        for r in op.get("reads") or ():
            prereq = producer.get(r)
            if r not in defined and prereq is not None and prereq["name"] in query_ops:
                emit(prereq)
        out.append(op)
        defined.update(op.get("outputs") or ())
        if op["name"] in query_ops:
            emitted_queries.add(op["name"])

    for op in ops:
        if op["name"] in insert_at:
            for new_op in insert_at[op["name"]]:
                emit(new_op)
        elif op["name"] not in removed:
            emit(op)
    defined = set(inputs_set)
    for op in out:
        missing = [r for r in op.get("reads") or () if r not in defined]
        if missing:
            raise AssertionError(f"depth_reads fbatch: {op['name']} reads {missing[:3]} before definition")
        defined |= set(op.get("outputs") or ())
    return out


def _apply_remats(ops, values, remats, mapping):
    out = []
    for op in ops:
        for N, nop, clone in remats.get(op["name"], ()):
            values[clone] = {**copy.deepcopy(values[N]), "name": clone, "producer": clone, "alias_of": None}
            nop = dict(nop, args=_replace(nop["args"], mapping, None))
            for key in ("inputs", "reads", "dependencies"):
                nop[key] = [mapping.get(x, x) for x in nop.get(key) or ()]
            out.append(nop)
            op = dict(op, args=_replace(op.get("args"), {N: clone}, None),
                      kwargs=_replace(op.get("kwargs"), {N: clone}, None))
            for key in ("inputs", "reads", "dependencies"):
                if isinstance(op.get(key), (list, tuple)):
                    op[key] = [clone if x == N else x for x in op[key]]
        out.append(op)
    return out
