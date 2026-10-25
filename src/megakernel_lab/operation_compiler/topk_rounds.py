"""Iterative row selection as one row pass (LATCH-MOESUP-ROW): a graph rewrite applied by ``lower`` before schedule
selection, and its warp-per-row emitter.

    a0 = argmax(x, -1)                                   x FP32 [..., W], W <= 256
    e0 = eq(ar, unsqueeze(a0, -1)); c0 = where(e0, s0, x) ar int64 [W] (e.g. arange), s0 FP32 []
    a1 = argmax(c0, -1); e1 = eq(ar, unsqueeze(a1, -1)); c1 = where(e1, s1, c0)
    ...
    a(K-1) = argmax(c(K-2), -1)

(the selection spelling of a stable top-K by K masked argmax rounds) becomes one op
``t = latch_ops.topk_rounds(x, ar, s0, ..., s(K-2); rounds=K)`` with t int64 [K, ...] and every ``ak`` a view
``select(t, 0, k)``. One warp per row keeps the row in registers (lane l holds columns l, l + 32, ...) and replays,
per round, the exact argmax_warp_row selection (same per-lane column order, same xor butterfly, same total order:
NaN maximal, then value, then lowest index), the exact mask ``ar[k] == a`` and replacement by ``s``. Every index is
therefore identical to the K-phase form; the masks and masked rows (read only inside the chain) are neither stored
nor re-read, and K argmax + K-1 mask + K-1 replacement phases (3K-2 grid barriers) become one phase.
Recognized from operations, shapes, dtypes and data flow only. ``ENABLED = False`` disables it.

LATCH-MOE4-PARITY port (from LATCH-MOESUP-ROW 0541553960, unchanged emitted arithmetic): also admits the
``argmax(..., keepdim=True)`` spelling, where ``e = eq(ar, a)`` reads the [..., 1] index directly and ``t`` is
[K, ..., 1]; the index values may have other readers (e.g. gathers), which then read the same indices through a
view of ``t``. A chain may start at an argmax whose source is produced by an unrelated ``where``.
"""
from __future__ import annotations

import copy
import math

ENABLED = True
LANE_ROWS = True     # rows of <= 32 columns: one lane per row (see emit)
TARGET = "latch_ops.topk_rounds.default"
MAX_WIDTH = 256
ARGMAX = "aten.argmax.default"
UNSQUEEZE = "aten.unsqueeze.default"
EQ = "aten.eq.Tensor"
WHERE = "aten.where.self"


def _ref(arg):
    return arg["ref"] if isinstance(arg, dict) and set(arg) == {"ref"} else None


def _contiguous(shape):
    strides, step = [], 1
    for extent in reversed(list(shape)):
        strides.append(step)
        step *= max(int(extent), 1)
    return list(reversed(strides))


def _output_names(data):
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


def rewrite(graph):
    """Return (graph, report); the input is never mutated."""
    if not ENABLED:
        return graph, None
    to_dict = getattr(graph, "to_dict", None)
    if to_dict is None and not isinstance(graph, dict):
        return graph, None
    data = to_dict() if to_dict is not None else copy.deepcopy(graph)
    ops, values = data.get("operations"), data.get("values")
    if not ops or values is None or not any(op.get("target") == ARGMAX for op in ops):
        return graph, None
    outputs = _output_names(data)
    report = {"version": 1, "rewritten": []}

    def reads(op):
        return set(op.get("reads") or op.get("inputs") or ())

    def readers_of(name):
        return [o for o in ops if name in reads(o) and o.get("name") != name]

    def argmax_of(op):
        """(source, lead shape, width) of a last-dim, no-keepdim FP32 argmax, or None."""
        args = op.get("args") or []
        if op.get("target") != ARGMAX or len(args) < 2:
            return None
        keepdim = bool(args[2]) if len(args) > 2 else bool(op.get("kwargs", {}).get("keepdim"))
        x = _ref(args[0])
        vx = values.get(x)
        if x is None or vx is None or vx.get("dtype") != "float32" or not vx.get("shape"):
            return None
        shape = list(vx["shape"])
        if int(args[1]) % len(shape) != len(shape) - 1 or values.get(op["name"], {}).get("dtype") != "int64":
            return None
        if not 1 <= shape[-1] <= MAX_WIDTH:
            return None
        if list(values[op["name"]].get("shape") or []) != (shape[:-1] + [1] if keepdim else shape[:-1]):
            return None
        return x, shape[:-1], shape[-1], keepdim

    def round_after(a, x, cur, lead, width, keepdim=False):
        """(eq op, where op, scalar, ar, unsqueeze views, next argmax) continuing the chain from argmax ``a``."""
        if keepdim:
            views = []
            eqs = [o for o in readers_of(a) if o.get("target") == EQ]
        else:
            views = [o for o in readers_of(a) if o.get("target") == UNSQUEEZE and o.get("synchronization") == "view"
                     and int((o.get("args") or [None, 0])[1]) % (len(lead) + 1) == len(lead)]
            eqs = [o for v in views for o in readers_of(v["name"]) if o.get("target") == EQ]
        if len(eqs) != 1:
            return None
        e = eqs[0]
        ea = [_ref(v) for v in (e.get("args") or [])]
        if len(ea) != 2 or None in ea:
            return None
        ar, u = ea
        if (u not in ({a} if keepdim else {v["name"] for v in views}) or values.get(ar, {}).get("dtype") != "int64"
                or list(values.get(ar, {}).get("shape") or []) != [width]
                or values.get(e["name"], {}).get("dtype") != "bool"
                or list(values[e["name"]]["shape"]) != lead + [width]):
            return None
        wheres = readers_of(e["name"])
        if len(wheres) != 1 or wheres[0].get("target") != WHERE:
            return None
        w = wheres[0]
        wa = [_ref(v) for v in (w.get("args") or [])]
        if len(wa) != 3 or wa[0] != e["name"] or wa[2] != cur or None in wa:
            return None
        s = wa[1]
        if (values.get(s, {}).get("dtype") != "float32" or list(values.get(s, {}).get("shape", [1])) != []
                or values.get(w["name"], {}).get("dtype") != "float32"
                or list(values[w["name"]]["shape"]) != lead + [width]):
            return None
        nxt = [o for o in readers_of(w["name"]) if o.get("target") == ARGMAX]
        if len(nxt) != 1 or argmax_of(nxt[0]) != (w["name"], lead, width, keepdim):
            return None
        # The mask and the masked row may be read only by the chain itself (next replacement / next argmax).
        if e["name"] in outputs or w["name"] in outputs:
            return None
        for o in readers_of(w["name"]):
            if o is nxt[0]:
                continue
            if o.get("target") != WHERE:
                return None
        used_views = [v for v in views if v["name"] == u]
        return e, w, s, ar, used_views, nxt[0]

    consumed = set()
    for start in [op for op in ops if op.get("target") == ARGMAX]:
        if start["name"] in consumed or start not in ops:
            continue
        info = argmax_of(start)
        if info is None:
            continue
        x, lead, width, keepdim = info
        producer = next((o for o in ops if o.get("name") == x), None)
        if producer is not None and producer.get("target") == WHERE:
            # A continuation round's source is where(eq(ar, argmax ...), s, previous row).
            cond = _ref((producer.get("args") or [None])[0])
            cond_op = next((o for o in ops if o.get("name") == cond), None)
            if cond_op is not None and cond_op.get("target") == EQ and any(
                    (q := next((o for o in ops if o.get("name") == _ref(arg)), None)) is not None
                    and q.get("target") in (ARGMAX, UNSQUEEZE) for arg in (cond_op.get("args") or [])):
                continue   # not a chain start
        rounds, cur, a, ar0 = [start], x, start, None
        removed = []
        while True:
            step = round_after(a["name"], x, cur, lead, width, keepdim)
            if step is None:
                break
            e, w, s, ar, views, nxt = step
            if ar0 is not None and ar != ar0:
                break
            ar0 = ar
            # the next where (if any) must read this row and nothing outside the chain may read it
            removed.append((e, w, s, views))
            rounds.append(nxt)
            cur, a = w["name"], nxt
        if len(rounds) < 2:
            continue
        # A trailing replacement whose masked row feeds no further argmax is left unchanged by construction.
        for e, w, s, views in removed:
            later = [o for o in readers_of(w["name"]) if o.get("target") == WHERE and o not in [r[1] for r in removed]]
            if later:
                break
        else:
            K = len(rounds)
            rows = math.prod(lead)
            t = start["name"] + "__topk"
            while t in values:
                t += "_"
            scalars = [s for _, _, s, _ in removed]
            refs = [x, ar0, *scalars]
            position = ops.index(start)
            # Operands produced after the first round must be input-free constants (e.g. scalar_tensor): they move
            # ahead of the fused op; anything else keeps the chain unchanged.
            late = [o for o in ops[position:] if o.get("name") in refs]
            if any(reads(o) for o in late):
                continue
            for o in late:
                ops.remove(o)
                ops.insert(position, o)
                position += 1
            t_op = {"name": t, "target": TARGET, "args": [{"ref": r} for r in refs], "kwargs": {"rounds": K},
                    "inputs": list(dict.fromkeys(refs)), "outputs": [t], "dependencies": list(dict.fromkeys(refs)),
                    "reads": list(dict.fromkeys(refs)), "writes": [t], "reduction_axes": [],
                    "synchronization": "global_phase_boundary", "effect": "pure"}
            tshape = [K, *lead, 1] if keepdim else [K, *lead]
            values[t] = {"name": t, "dtype": "int64", "shape": tshape, "strides": _contiguous(tshape),
                         "role": "temporary", "parameter": None, "alias_of": None, "storage_offset": 0,
                         "producer": t, "output_roles": []}
            ops.insert(position, t_op)
            for k, a_op in enumerate(rounds):
                name = a_op["name"]
                a_op.update(target="aten.select.int", args=[{"ref": t}, 0, k], kwargs={}, inputs=[t], dependencies=[t],
                            reads=[t], writes=[], reduction_axes=[], synchronization="view")
                values[name].update(alias_of=t, storage_offset=k * rows, strides=_contiguous(tshape[1:]))
                consumed.add(name)
            gone = []
            for e, w, s, views in removed:
                gone += [e, w]
                for v in views:
                    if not [o for o in readers_of(v["name"]) if o is not e]:
                        gone.append(v)
            for o in gone:
                if o in ops:
                    ops.remove(o)
                    values.pop(o["name"], None)
            report["rewritten"].append({"rounds": K, "source": x, "rows": rows, "width": width, "indices": t,
                                        "keepdim": keepdim,
                                        "selections": [r["name"] for r in rounds]})
    if not report["rewritten"]:
        return graph, None
    data.setdefault("metadata", {})["topk_rounds"] = report
    if to_dict is not None:
        return type(graph).from_dict(data), report
    return data, report


_HELPER = '''#ifndef LATCH_ARGMAX_HELPERS
#define LATCH_ARGMAX_HELPERS
__device__ __forceinline__ bool latch_argmax_better(float v, long long i, float b, long long j) {
  if (isnan(v)) return !isnan(b) || i < j;
  if (isnan(b)) return false;
  return v > b || (v == b && i < j);
}
#endif
'''


def emit(target, args, kwargs, name, shape, layouts, dtypes, names, read):
    """(kind, code, metadata, helpers) for latch_ops.topk_rounds, or None."""
    if target != TARGET:
        return None
    from .lower import LoweringError, _ident, _index, _ref as ref_of
    refs = [ref_of(a, names) for a in args]
    if len(refs) < 2 or None in refs:
        raise LoweringError("topk_rounds: tensor operands required")
    x, ar, scalars = refs[0], refs[1], refs[2:]
    K = int(kwargs["rounds"])
    lx, la = layouts[x], layouts[ar]
    width = lx.shape[-1]
    rows = math.prod(lx.shape[:-1])
    if (dtypes[x] != "float32" or dtypes[ar] != "int64" or tuple(la.shape) != (width,) or dtypes[name] != "int64"
            or tuple(shape) not in ((K, *lx.shape[:-1]), (K, *lx.shape[:-1], 1)) or len(scalars) != K - 1
            or width > MAX_WIDTH
            or any(dtypes[s] != "float32" or math.prod(layouts[s].shape) != 1 for s in scalars)):
        raise LoweringError("topk_rounds: FP32 [..., W<=256] rows, int64 [W] mask operand, K-1 FP32 scalars")
    out = _ident(name)
    arr = f'{_ident(ar)}[{_index(la, la.shape, "k")}]'
    lead, lead_strides = list(lx.shape[:-1]), list(lx.strides[:-1])
    lane_dims = [d for d in range(len(lead)) if lead[d] % 32 == 0 and lead_strides[d] > 0]
    if LANE_ROWS and width <= 32 and lane_dims:
        # Lane per row: the argmax is the unique maximum of a strict total order (NaN maximal, then value, then the
        # lowest index), so a lane's sequential scan selects exactly the element the warp butterfly selects. Lanes
        # take consecutive rows along the row dimension of smallest stride (coalesced row loads), each lane keeps its
        # row and the mask operand in registers; no shuffles.
        d = min(lane_dims, key=lambda j: lead_strides[j])
        gdims = [n // 32 if j == d else n for j, n in enumerate(lead)]
        groups = math.prod(gdims)
        unravel, rem = [], "g"
        for j in range(len(gdims) - 1, -1, -1):
            unravel.append(f"  long long c{j}_ = {rem} % {gdims[j]}LL;" if j else f"  long long c{j}_ = {rem};")
            if j:
                unravel.append(f"  const long long q{j}_ = {rem} / {gdims[j]}LL;")
                rem = f"q{j}_"
        unravel.append(f"  c{d}_ = c{d}_ * 32 + lane;")
        row_expr = " + ".join(f"c{j}_ * {math.prod(lead[j + 1:])}LL" for j in range(len(lead))) or "0LL"
        lines = [f"for (long long g = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; g < {groups}LL; "
                 f"g += ((long long)gridDim.x * blockDim.x) >> 5) {{",
                 "  const int lane = threadIdx.x & 31;", *unravel,
                 f"  const long long row = {row_expr};",
                 f"  float v_[{width}]; long long a_[{width}];",
                 "#pragma unroll",
                 f"  for (int k = 0; k < {width}; ++k) {{ v_[k] = {_ident(x)}[{_index(lx, lx.shape, f'(row * {width}LL + k)')}]; "
                 f"a_[k] = {arr}; }}"]
        for r in range(K):
            lines += ["  {",
                      "    float best = -INFINITY; long long index = 0x7fffffffffffffffLL;",
                      "#pragma unroll",
                      f"    for (int k = 0; k < {width}; ++k) if (latch_argmax_better(v_[k], k, best, index)) {{ best = v_[k]; index = k; }}",
                      f"    {out}[{r * rows}LL + row] = static_cast<int64_t>(index);"]
            if r < K - 1:
                s = scalars[r]
                lines += [f"    const float s_ = {_ident(s)}[{_index(layouts[s], layouts[s].shape, '0')}];",
                          "#pragma unroll",
                          f"    for (int k = 0; k < {width}; ++k) if (a_[k] == static_cast<int64_t>(index)) v_[k] = s_;"]
            lines.append("  }")
        lines.append("}")
        return "reduction", "\n".join(lines), {"reduction_elements": width, "engine": "topk_rounds_lane_row",
                                               "rounds": K,
                                               "arithmetic": "exact comparisons (argmax total order: NaN maximal, "
                                                             "first index on ties); masked replacement by the given "
                                                             "scalars; no arithmetic"}, _HELPER
    nw = -(-width // 32)
    xr = f'{_ident(x)}[{_index(lx, lx.shape, f"(row * {width}LL + k)")}]'
    lines = [f"for (long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row < {rows}LL; "
             f"row += ((long long)gridDim.x * blockDim.x) >> 5) {{",
             "  const int lane = threadIdx.x & 31;",
             f"  float v_[{nw}];",
             "#pragma unroll",
             f"  for (int j = 0; j < {nw}; ++j) {{ const long long k = lane + 32 * j; v_[j] = k < {width}LL ? {xr} : 0.0f; }}"]
    for r in range(K):
        lines += ["  {",
                  "    float best = -INFINITY; long long index = 0x7fffffffffffffffLL;",
                  "#pragma unroll",
                  f"    for (int j = 0; j < {nw}; ++j) {{ const long long k = lane + 32 * j;",
                  f"      if (k < {width}LL && latch_argmax_better(v_[j], k, best, index)) {{ best = v_[j]; index = k; }} }}",
                  "    for (int delta = 16; delta > 0; delta /= 2) {",
                  "      const float v = __shfl_xor_sync(0xffffffffU, best, delta);",
                  "      const long long k = __shfl_xor_sync(0xffffffffU, index, delta);",
                  "      if (latch_argmax_better(v, k, best, index)) { best = v; index = k; }",
                  "    }",
                  f"    if (lane == 0) {out}[{r * rows}LL + row] = static_cast<int64_t>(index);"]
        if r < K - 1:
            s = scalars[r]
            sv = f'{_ident(s)}[{_index(layouts[s], layouts[s].shape, "0")}]'
            lines += [f"    const float s_ = {sv};",
                      "#pragma unroll",
                      f"    for (int j = 0; j < {nw}; ++j) {{ const long long k = lane + 32 * j;",
                      f"      if (k < {width}LL && {arr} == static_cast<int64_t>(index)) v_[j] = s_; }}"]
        lines.append("  }")
    lines.append("}")
    return "reduction", "\n".join(lines), {"reduction_elements": width, "engine": "topk_rounds_warp_row",
                                           "rounds": K,
                                           "arithmetic": "exact comparisons (argmax total order: NaN maximal, "
                                                         "first index on ties); masked replacement by the given "
                                                         "scalars; no arithmetic"}, _HELPER
