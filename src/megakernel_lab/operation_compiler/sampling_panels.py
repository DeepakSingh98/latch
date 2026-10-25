"""Cluster-resident row-panel execution of a nondifferentiated sampling loop (P8-D panel engine rule).

The generic sampling-loop schedule materializes every intermediate of the step in global memory and separates
its phases with whole-grid barriers (C-OPS4-B: ~37 phases of ~4 us per WB Langevin step). Splitting rows over
CTA groups with the same per-phase structure did not help (row tasks, +5-14%). This rule changes WHERE row state
lives and HOW products exchange it, derived from graph properties only:

  * row separability (ops.sampling_rows.row_axes): every value of the step is row-indexed along dimension 0 of the
    carry, or row-free (loop-invariant parameters/casts, constants, broadcasts of them);
  * every matrix product contracts a row-indexed BF16 panel [B, K] with a loop-invariant BF16 operand [K, N]
    (a strided view of a bound input); everything else is elementwise per (row, column) or a counter-grid random
    field addressed by absolute (row, column).

Execution: B rows are split into G groups of R rows; each group runs its whole chain on one thread-block cluster
of C CTAs. Each CTA owns whole 16-column units of every row-indexed value (balanced split per width) and keeps
the values that live across products in shared memory (BF16 product outputs, FP32 values chosen by a
recomputation-cost rule, the carry). Between two products only elementwise work on owned slices runs; a product's
BF16 operand is pushed to every peer with one bulk DSMEM copy per peer and an mbarrier transaction count (no grid
or cluster barrier per phase); each CTA computes its owned output features over the full K from streamed weight
chunks (ops/kernels/panel_engine.cuh). The whole grid meets again only after the loop.

Arithmetic: products keep the declared policy (BF16 operands, FP32 accumulation, one BF16 rounding); elementwise
expressions are the same C expressions as ops.lower renders, evaluated once per element in FP32 (generated code is
compiled --fmad=false); summation order inside a product differs from the generic engine (k-split partial chains in
fixed order). Results are deterministic and independent of timing.

Refusals (with a recorded reason, falling back to the ordered schedule): step-changed state, non-row-separable
ops, reductions, products whose right operand is row-indexed or not a strided view of a BF16 input, widths that are
not multiples of 16, row-free values that vary along rows, or no geometry that fits shared memory and residency.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import math

from .ir import Ref
from .lower import LoweringError, _float, _header_path
from .sampling_rows import FIELD, REDUCED, RowRefusal, _base, _refs, row_axes

HEADER = "panel_engine.cuh"
PANEL_DEFAULTS = {"cluster": "auto", "rows": "auto", "stages": "auto", "chunk_k": 64, "remat_limit": 4,
                  "ksplit": "auto", "max_tasks": 16, "diagnostics": [], "mode": "auto", "fuse": True,
                  "overlap": False, "spin": False, "exchange": "multicast", "tiling": "mma",
                  "early_consumed": True, "elementwise": "exact", "unroll": False,
                  "spill": "auto", "spill_gap": 3, "spill_ahead": 2, "pack": "fused", "mma": "auto",
                  "wg_desc": "a", "wg_inline": False, "field_ahead": False, "share_segments": True}
# P9-D share_segments: segment bodies are canonicalized (local value names renamed by first use; shared-memory offsets,
# slot parities, barrier indices and input pointers lifted into arguments) and identical bodies become ONE noinline
# function called per segment, shrinking the per-iteration instruction footprint (the loop body otherwise exceeds the
# instruction cache: every unique segment is fetched from L2 each iteration). Same statements, same arithmetic.
# P9-D field_ahead: a counter-grid random field evaluated in one segment (e.g. Langevin noise) is generated for the
# next iteration in slices during the exchange waits of the following products into a shared buffer (iteration 0 in
# the prologue); its segment then reads the buffer. Same Philox words and transform at the same (row, column, counter).
# P9-D mma='wgmma': products run as two warpgroups of m64nRk16 wgmma (k-split 2) on the packed swizzled chunks and the
# panel's core matrices; wg_desc selects the no-swizzle B descriptor order (a: LBO = R x 16, SBO = 128; b: swapped).
# P9-D spill: 'auto' keeps every value in shared memory when the slice layout fits (byte-identical P8-D program) and
# otherwise splits long idle intervals of stored values out to a per-CTA global (L2-resident) area: the value is
# written there when produced and brought back by ONE bulk copy issued `spill_ahead` hops before its next use.
DIAGNOSTICS = {"skip_exchange", "skip_mma", "skip_weights"}   # timing probes only; results are invalid
SHARED_LIMIT = 232448
# Co-resident thread-block clusters at one CTA per SM on H100 SXM (measured by this lane's probe, gpu-01,
# cudaOccupancyMaxActiveClusters with >= 100 KiB dynamic shared memory). The runtime query still admits the grid.
MAX_CLUSTERS = {2: 66, 4: 30, 8: 15, 12: 8, 16: 7}   # 12: not probed; the runtime occupancy query still admits
COST = {"field": 24, "sigmoid": 4, "silu": 4, "silu_backward": 6, "exp": 4, "expm1": 4, "log1p": 4, "log": 4, "tanh": 4,
        "erf": 4, "sin": 4, "cos": 4, "sqrt": 2, "rsqrt": 2, "div": 2, "reciprocal": 2, "pow": 4}
IDENTITY = {"clone", "contiguous", "alias", "detach", "lift_fresh_copy"}
CONSTRUCTORS = {"full", "full_like", "ones_like", "zeros_like", "empty_like", "new_full", "new_zeros", "new_ones",
                "zeros", "ones", "scalar_tensor"}
UNARY = {"neg", "exp", "expm1", "log1p", "log", "tanh", "sqrt", "rsqrt", "abs", "sigmoid", "silu", "relu", "reciprocal",
         "erf", "sin", "cos"}
BINARY = {"add", "sub", "mul", "div", "rsub", "minimum", "maximum", "silu_backward", "sigmoid_backward",
          "tanh_backward"}
CLAMPS = {"clamp", "clamp_min", "clamp_max"}


class PanelRefusal(ValueError):
    pass


@dataclass
class Node:
    name: str
    kind: str                 # carry | product | elem | field | const | rowfree
    width: int = 0
    dtype: str = "float32"
    op: object = None
    inputs: list = field(default_factory=list)   # row-indexed input names (dependencies)


def _scalar_arg(op, index, key=None, default=None):
    if len(op.args) > index:
        return op.args[index]
    return op.kwargs.get(key, default) if key else default


def _ref(x):
    return x.name if isinstance(x, Ref) else None


class _Graph:
    """Classified live step operations."""

    def __init__(self, graph):
        m = graph.metadata
        region = graph.body
        self.region = region
        names, casts = m["parameter_names"], m.get("invariant_casts", [])
        np_, nw, nc, ns = len(names), len(casts), len(m["carry_names"]), len(m["state_names"])
        if m.get("body_state_changes"):
            raise PanelRefusal("the step changes explicit state (only the carry is row-resident)")
        inputs = list(region.inputs)
        self.carry_in = inputs[np_ + nw:np_ + nw + nc]
        self.k_input = inputs[np_ + nw + nc + ns]
        self.inputs = inputs
        values = region.values
        self.values = values
        if not self.carry_in:
            raise PanelRefusal("no carry")
        B = values[self.carry_in[0]].shape[0] if values[self.carry_in[0]].shape else 0
        self.B = B
        try:
            self.axes = row_axes(region, set(self.carry_in), B)
        except RowRefusal as exc:
            raise PanelRefusal(f"row separability: {exc}") from exc
        if len(region.outputs) != nc:
            raise PanelRefusal("step outputs other than the carry")
        producers = {}
        for op in region.operations:
            for out in (op.outputs or [op.name]):
                producers[out] = op
        self.producers = producers
        live, stack = set(), list(region.outputs)
        while stack:
            name = stack.pop()
            if name in live:
                continue
            live.add(name)
            op = producers.get(name)
            if op is not None:
                stack.extend(r for r in dict.fromkeys(_refs((op.args, op.kwargs))))
        self.live = live
        self.nodes, self.rowfree, self.products = {}, {}, []
        for name in self.carry_in:
            self._row_value(name)
            self.nodes[name] = Node(name, "carry", self.width(name), str(values[name].dtype).removeprefix("torch."))
        for op in region.operations:
            out = op.outputs[0] if op.outputs else op.name
            if out not in live:
                continue
            self._classify(op, out)
        for i, out in enumerate(region.outputs):
            node = self.nodes.get(out)
            if node is None or node.kind == "const" or node.width != self.nodes[self.carry_in[i]].width:
                raise PanelRefusal(f"output {out} is not a row panel of the carry width")

    def dtype(self, name):
        return str(self.values[name].dtype).removeprefix("torch.")

    def width(self, name):
        return int(self.values[name].shape[1])

    def _row_value(self, name):
        shape = tuple(self.values[name].shape)
        if self.axes.get(name) != 0 or len(shape) != 2 or shape[0] != self.B:
            raise PanelRefusal(f"{name}: row-indexed values must be [B, n] panels")
        if shape[1] % 16:
            raise PanelRefusal(f"{name}: panel width {shape[1]} is not a multiple of 16")

    # ---------------------------------------------------------------- row-free values as column expressions
    def column_expr(self, name, width):
        """C expression of a row-free value broadcast to [*, width] at column `col` (or a constant)."""
        if name in self.rowfree:
            kind, payload, w = self.rowfree[name]
        else:
            kind, payload, w = self._resolve_rowfree(name)
            self.rowfree[name] = (kind, payload, w)
        if kind == "const":
            return payload
        if w not in (None, width):
            raise PanelRefusal(f"{name}: row-free broadcast width {w} differs from {width}")
        return payload

    def _resolve_rowfree(self, name):
        values = self.values
        shape = tuple(values[name].shape)
        dtype = self.dtype(name)
        if name in self.inputs:
            if name == self.k_input:
                raise PanelRefusal(f"iteration index {name} used outside random-field counters")
            if dtype not in ("float32",):
                raise PanelRefusal(f"row-free input {name} of dtype {dtype} in elementwise use")
            if math.prod(shape) == 1:
                return ("expr", f"__ldg(&IN({name})[0])", None)
            if len(shape) >= 1 and all(d == 1 for d in shape[:-1]):
                return ("expr", f"VEC({name})", shape[-1])
            raise PanelRefusal(f"row-free input {name} {shape} varies along rows")
        op = self.producers[name]
        base = _base(op.target)
        refs = list(dict.fromkeys(_refs((op.args, op.kwargs))))
        if base in CONSTRUCTORS or base == "fill":
            value = (op.args[1] if base == "fill" else
                     op.args[2] if base == "new_full" else
                     op.args[1] if base in ("full", "full_like") else
                     op.args[0] if base == "scalar_tensor" else
                     1 if "ones" in base else 0)
            if base == "empty_like":
                return ("const", "0.0f", None)   # value never read except through fill (checked by users)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PanelRefusal(f"{name}: non-literal constant")
            return ("const", _float(value), None)
        if base in {"expand", "unsqueeze", "view", "_unsafe_view", "reshape"} | IDENTITY or (
                base == "_to_copy" and self.dtype(refs[0]) == dtype):
            source = refs[0]
            kind, payload, w = self._resolve_rowfree(source)
            if w is not None and (not shape or shape[-1] != w):
                raise PanelRefusal(f"{name}: view moves a broadcast column axis")
            return kind, payload, w
        if base in UNARY | BINARY | CLAMPS:
            parts, widths = [], set()
            for arg in op.args:
                r = _ref(arg)
                if r is None:
                    parts.append(arg)
                    continue
                kind, payload, w = self._resolve_rowfree(r)
                parts.append(("E", payload))
                if w is not None:
                    widths.add(w)
            if len(widths) > 1:
                raise PanelRefusal(f"{name}: row-free operands disagree on width")
            expression = _render(base, op, [p[1] if isinstance(p, tuple) else p for p in parts],
                                 [isinstance(p, tuple) for p in parts])
            return ("expr", expression, widths.pop() if widths else None)
        raise PanelRefusal(f"{name}: row-free {op.target} has no column-expression rule")

    # ---------------------------------------------------------------- classification
    def _classify(self, op, out):
        base = _base(op.target)
        axis = self.axes.get(out)
        refs = list(dict.fromkeys(_refs((op.args, op.kwargs))))
        def is_const(r):
            node = self.nodes.get(r)
            return (node is not None and node.kind == "const") or (node is None and self.axes.get(r) is None)
        if base not in ("mm", "matmul") and op.target != FIELD and refs and all(is_const(r) for r in refs):
            axis = None   # e.g. a constant broadcast along the row extent: a column expression, not row state
        if base in CONSTRUCTORS or base == "fill" or (axis is None):
            # Row-free or shape-only constructors: resolved lazily as column expressions by their users.
            if axis == REDUCED:
                raise PanelRefusal(f"{out}: reduction over rows")
            self.nodes[out] = Node(out, "const", op=op)
            return
        if axis == REDUCED:
            raise PanelRefusal(f"{out}: reduction over rows")
        self._row_value(out)
        width, dtype = self.width(out), self.dtype(out)
        if op.target == FIELD:
            key, c2, c3 = (_ref(a) for a in op.args[:3])
            transform = op.args[4]
            if len(op.args) > 5:
                raise PanelRefusal("row-based random field is not admitted")
            for c in (c2, c3):
                if c is None or (c != self.k_input and c not in self.inputs):
                    raise PanelRefusal("random-field counters must be the iteration index or bound scalars")
            if key not in self.inputs or tuple(self.values[key].shape) != (2,):
                raise PanelRefusal("random-field key must be a bound int64[2] input")
            if transform not in ("normal", "uniform"):
                raise PanelRefusal(f"random-field transform {transform} is not admitted")
            self.nodes[out] = Node(out, "field", width, dtype, op=op)
            return
        if base in ("mm", "matmul"):
            a, b = refs[0], refs[1]
            if self.axes.get(a) != 0 or self.dtype(a) != "bfloat16" or dtype != "bfloat16":
                raise PanelRefusal(f"{out}: products need a row-indexed BF16 left operand and BF16 output")
            weight, sk, sn = self._weight_view(b)
            K, N = tuple(self.values[b].shape)
            if K % 16 or N % 16:
                raise PanelRefusal(f"{out}: product dimensions must be multiples of 16")
            node = Node(out, "product", width, dtype, op=op, inputs=[a])
            node.weight, node.sk, node.sn, node.K = weight, sk, sn, K
            self.nodes[out] = node
            self.products.append(out)
            return
        if base in IDENTITY or (base == "_to_copy" and len(refs) == 1) or base in UNARY | BINARY | CLAMPS | {"where"}:
            if dtype not in ("float32", "bfloat16", "bool"):
                raise PanelRefusal(f"{out}: dtype {dtype} in the panel step")
            inputs = [r for r in refs if self.axes.get(r) == 0 and self.nodes.get(r, Node(r, "const")).kind != "const"]
            for r in inputs:
                if self.width(r) != width:
                    raise PanelRefusal(f"{out}: operand widths differ")
            self.nodes[out] = Node(out, "elem", width, dtype, op=op, inputs=inputs)
            return
        raise PanelRefusal(f"{out}: {op.target} has no panel rule")

    def _weight_view(self, name):
        """(input name, sk, sn) such that B(k, n) = input[k * sk + n * sn] for a strided view chain."""
        shape = tuple(self.values[name].shape)
        strides = list(self.values[name].strides)
        chain = []
        current = name
        while current not in self.inputs:
            op = self.producers.get(current)
            if op is None:
                raise PanelRefusal(f"{name}: product operand is not a view of a bound input")
            base = _base(op.target)
            if base not in {"t", "transpose", "permute"} | IDENTITY:
                raise PanelRefusal(f"{name}: product operand view {op.target} is not admitted")
            chain.append(op)
            current = _ref(op.args[0])
        if self.dtype(current) != "bfloat16" or len(self.values[current].shape) != 2:
            raise PanelRefusal(f"{name}: product operand base must be a BF16 matrix input")
        bshape, bstrides = tuple(self.values[current].shape), tuple(self.values[current].strides)
        if bstrides != (bshape[1], 1):
            raise PanelRefusal(f"{name}: product operand base must be contiguous")
        # Recompute the view strides from the base (independent of the recorded meta strides).
        dims = [(bshape[0], bstrides[0]), (bshape[1], bstrides[1])]
        for op in reversed(chain):
            base = _base(op.target)
            if base == "t":
                dims = dims[::-1]
            elif base == "transpose":
                i, j = int(op.args[1]) % 2, int(op.args[2]) % 2
                dims[i], dims[j] = dims[j], dims[i]
            elif base == "permute":
                dims = [dims[int(d) % 2] for d in op.args[1]]
        if tuple(d for d, _ in dims) != shape:
            raise PanelRefusal(f"{name}: view chain shape mismatch")
        sk, sn = dims[0][1], dims[1][1]
        if 1 not in (sk, sn):
            raise PanelRefusal(f"{name}: product operand must be contiguous along k or n")
        return current, sk, sn


def _render(base, op, args, is_expr, fast=False):
    """C expression of one elementwise operation from operand expressions (ops.lower formulas).

    fast=True (declared POLICY amendment p1, opt-in): sigmoid/silu use __expf and a correctly rounded reciprocal."""
    def lit(i):
        return args[i] if is_expr[i] else _float(args[i])
    if base in ("add", "sub"):
        alpha = op.kwargs.get("alpha", op.args[2] if len(op.args) > 2 else 1)
        right = lit(1) if alpha == 1 else f"({lit(1)} * {_float(alpha)})"
        return f"({lit(0)} {'+' if base == 'add' else '-'} {right})"
    if base == "rsub":
        alpha = op.kwargs.get("alpha", op.args[2] if len(op.args) > 2 else 1)
        return f"({lit(1)} - ({lit(0)} * {_float(alpha)}))"
    if base in ("mul", "div"):
        if base == "div" and op.kwargs.get("rounding_mode") is not None:
            raise PanelRefusal("rounded division")
        return f"({lit(0)} {'*' if base == 'mul' else '/'} {lit(1)})"
    if base == "neg":
        return f"(-{lit(0)})"
    if base == "reciprocal":
        return f"(1.0f / {lit(0)})"
    if base in ("tanh", "exp", "expm1", "log1p", "log", "sqrt", "rsqrt", "abs", "erf", "sin", "cos"):
        fn = "fabsf" if base == "abs" else base + "f"
        return f"{fn}({lit(0)})"
    if base == "sigmoid":
        if fast:
            return f"__frcp_rn(1.0f + __expf(-{lit(0)}))"
        return f"(1.0f / (1.0f + expf(-{lit(0)})))"
    if base == "silu":
        if fast:
            return f"({lit(0)} * __frcp_rn(1.0f + __expf(-{lit(0)})))"
        return f"({lit(0)} / (1.0f + expf(-{lit(0)})))"
    if base == "relu":
        return f"fmaxf({lit(0)}, 0.0f)"
    if base == "silu_backward":
        s = (f"__frcp_rn(1.0f + __expf(-{lit(1)}))" if fast else f"(1.0f / (1.0f + expf(-{lit(1)})))")
        return f"({lit(0)} * {s} * (1.0f + {lit(1)} * (1.0f - {s})))"
    if base == "sigmoid_backward":
        return f"({lit(0)} * {lit(1)} * (1.0f - {lit(1)}))"
    if base == "tanh_backward":
        return f"({lit(0)} * (1.0f - {lit(1)} * {lit(1)}))"
    if base in ("minimum", "maximum"):
        sym = "<" if base == "minimum" else ">"
        a, b = lit(0), lit(1)
        return f"(({a} != {a}) || ({b} != {b}) ? __int_as_float(0x7fc00000) : ({a} {sym} {b} ? {a} : {b}))"
    if base in CLAMPS:
        low = op.args[1] if base in ("clamp", "clamp_min") and len(op.args) > 1 else op.kwargs.get("min")
        high = ((op.args[2] if len(op.args) > 2 else op.kwargs.get("max")) if base == "clamp" else
                (op.args[1] if base == "clamp_max" else None))
        if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in (low, high)):
            raise PanelRefusal("clamp admits static scalar bounds only")
        value = lit(0)
        if low is not None:
            value = f"({value} < {_float(low)} ? {_float(low)} : {value})"
        if high is not None:
            value = f"({value} > {_float(high)} ? {_float(high)} : {value})"
        return value
    if base == "where":
        return f"({lit(0)} ? {lit(1)} : {lit(2)})"
    raise PanelRefusal(f"no elementwise rendering for {base}")


def _cost(node):
    if node.kind == "field":
        return COST["field"]
    if node.kind != "elem":
        return 0
    base = _base(node.op.target)
    if base in IDENTITY or (base == "_to_copy" and node.dtype == "float32"):
        return 0
    return COST.get(base, 1)


@dataclass
class Geometry:
    cluster: int
    rows: int
    groups: int
    stages: int
    chunk_k: int


def _units(n):
    return n // 16


def _wmax(n, C):
    return 16 * -(-_units(n) // C)


class PanelSchedule:
    def __init__(self, g, geometry, ksplit, maxt, plan, shared_bytes, layout):
        self.g, self.geo, self.ksplit, self.maxt = g, geometry, ksplit, maxt
        self.plan, self.shared_bytes, self.layout = plan, shared_bytes, layout
        self.cluster = geometry.cluster


def _schedule_values(g, remat_limit):
    """Segments, materialization and per-segment evaluation sets."""
    products = g.products
    index = {p: j for j, p in enumerate(products)}
    m = len(products)
    nodes = g.nodes
    roots = [(j, nodes[p].inputs[0], "exchange", p) for j, p in enumerate(products)]
    roots += [(m, out, "output", i) for i, out in enumerate(g.region.outputs)]
    roots.sort(key=lambda x: x[0])
    leaf = lambda v: nodes[v].kind in ("carry", "product")

    def demands(M):
        demand, first = defaultdict(set), {}

        def visit(v, s):
            node = nodes[v]
            if node.kind == "const":
                return
            if leaf(v):
                demand[v].add(s)
                return
            if v in M:
                demand[v].add(s)
                if v in first:
                    return
                first[v] = s
            else:
                if s in demand[v]:
                    return
                demand[v].add(s)
            for u in node.inputs:
                visit(u, s)
        for s, v, _, _ in roots:
            visit(v, s)
        return demand, first

    def cone_cost(v, M, seen):
        if v in seen or leaf(v) or v in M or nodes[v].kind == "const":
            return 0
        seen.add(v)
        return _cost(nodes[v]) + sum(cone_cost(u, M, seen) for u in nodes[v].inputs)

    M = set()
    order = [v for v in nodes if nodes[v].kind in ("elem", "field")]
    for v in reversed(order):   # consumers first: keep the values many segments read, recompute their cones
        demand, _ = demands(M)
        if len(demand.get(v, ())) > 1 and cone_cost(v, M, set()) > remat_limit:
            M.add(v)
    demand, first = demands(M)
    # Availability check: an evaluated value's product inputs must precede the segment.
    for v, segs in demand.items():
        node = nodes[v]
        if node.kind == "product":
            if min(segs) <= index[v]:
                raise PanelRefusal(f"{v} is read before its product")
    evaluate = defaultdict(list)    # segment -> values computed there (operation order)
    for v in order:
        if v not in demand:
            continue
        segs = sorted(demand[v])
        if v in M:
            evaluate[first[v]].append(v)
        else:
            for s in segs:
                evaluate[s].append(v)
    return {"m": m, "index": index, "roots": roots, "materialized": M, "demand": demand, "first": first,
            "evaluate": evaluate}


def _lifetimes(g, plan, chained=True):
    """Stored values -> (start, end, dtype, width) in segment units (products at j + 0.5).

    The carry slot holds the loop-entry carry until its last read and receives the new carry at the last segment;
    with `chained`, other same-shaped values may occupy it in between (element-local handoff), modelled by the
    pseudo value '<carry>@out' pinned to the carry's slot."""
    nodes, demand, index, m = g.nodes, plan["demand"], plan["index"], plan["m"]
    lives = {}
    for v in g.carry_in:
        dtype = "float32" if nodes[v].dtype == "float32" else nodes[v].dtype
        last = float(max(demand[v])) if v in demand else 0.0
        if chained:
            lives[v] = (0.0, last, dtype, nodes[v].width)
            lives[v + "@out"] = (float(m), float(m) + 0.75, dtype, nodes[v].width)
        else:
            lives[v] = (0.0, float(m) + 0.75, dtype, nodes[v].width)
    for p in g.products:
        if p in demand:
            lives[p] = (index[p] + 0.5, float(max(demand[p])), "bfloat16", nodes[p].width)
    for v in plan["materialized"]:
        if nodes[v].dtype not in ("float32", "bfloat16"):
            raise PanelRefusal(f"{v}: only FP32/BF16 values are stored across products")
        if v in demand:
            lives[v] = (float(plan["first"][v]), float(max(demand[v])), nodes[v].dtype, nodes[v].width)
    return lives


def _allocate(lives, R, C, base):
    """First-fit interval allocation of shared slices [R][WMAX(width)]; same-segment handoff only for equal
    dtype/width (element-local: each element is read before it is overwritten by the same thread).
    '<v>@out' pseudo values must land in the slot of <v> (else PanelRefusal)."""
    def fits(slot, start, end, dtype, width):
        for (s0, e0, d0, w0) in slot["users"]:
            handoff = (e0 == start and d0 == dtype and w0 == width) or (end == s0 and d0 == dtype and w0 == width)
            if not (e0 < start or end < s0 or handoff):
                return False
        return True
    slots, assignment, slot_of = [], {}, {}
    pinned = [v for v in lives if v.endswith("@out")]
    order = sorted((v for v in lives if not v.endswith("@out")), key=lambda v: (lives[v][0], -_wmax(lives[v][3], C)))
    for v in order:
        start, end, dtype, width = lives[v]
        nbytes = R * _wmax(width, C) * (2 if dtype == "bfloat16" else 4)
        nbytes = (nbytes + 127) // 128 * 128
        target = None
        for slot in slots:
            if slot["bytes"] >= nbytes and fits(slot, start, end, dtype, width):
                target = slot
                break
        if target is None:
            target = {"offset": base + sum(x["bytes"] for x in slots), "bytes": nbytes, "users": []}
            slots.append(target)
        target["users"].append((start, end, dtype, width))
        assignment[v] = target["offset"]
        slot_of[v] = target
    for v in pinned:
        carry = v[:-4]
        slot = slot_of[carry]
        start, end, dtype, width = lives[v]
        if not fits(slot, start, end, dtype, width):
            raise PanelRefusal("carry slot is not free for the new carry")
        slot["users"].append((start, end, dtype, width))
        assignment[v] = slot["offset"]
    total = sum(x["bytes"] for x in slots)
    return assignment, total


def plan_panels(graph, options, *, batch_rows=None):
    """PanelSchedule for the graph's sampling step, or PanelRefusal."""
    opts = {**PANEL_DEFAULTS, **(options or {})}
    g = _Graph(graph)
    if not g.products:
        raise PanelRefusal("no matrix products in the step")
    plan = _schedule_values(g, opts["remat_limit"])
    B = g.B
    widths = {g.nodes[p].width for p in g.products} | {g.nodes[p].K for p in g.products}
    KC = int(opts["chunk_k"])
    candidates = []
    clusters = [opts["cluster"]] if opts["cluster"] != "auto" else [8, 16, 4]
    rows_options = [opts["rows"]] if opts["rows"] != "auto" else [32, 24, 40, 16, 48, 64]
    for C in clusters:
        for R in rows_options:
            if R % 8:
                continue
            G = -(-B // R)
            if G > MAX_CLUSTERS.get(C, 0):
                continue
            if opts.get("mode", "auto") in ("auto", "slice"):
                spill = opts.get("spill", "auto")
                if spill not in ("auto", "on", "off"):
                    raise PanelRefusal("spill must be auto, on or off")
                fitted = None
                if spill in ("auto", "off"):
                    try:
                        fitted = _fit(g, plan, Geometry(C, R, G, 2, KC), opts, mode="slice")
                    except PanelRefusal:
                        pass
                if fitted is None and spill in ("auto", "on"):
                    try:
                        fitted = _fit(g, plan, Geometry(C, R, G, 2, KC), opts, mode="slice", spill=True)
                    except PanelRefusal:
                        pass
                if fitted is not None:
                    candidates.append(fitted)
            if opts.get("mode", "auto") == "slice":
                continue
            for stages in ([opts["stages"]] if opts["stages"] != "auto" else [8, 6, 5, 4, 3]):
                if stages < 3:
                    continue
                geo = Geometry(C, R, G, stages, KC)
                try:
                    candidates.append(_fit(g, plan, geo, opts))
                except PanelRefusal:
                    continue
    if not candidates:
        raise PanelRefusal("no cluster/row geometry fits shared memory and co-residency")
    candidates.sort(key=lambda s: s.layout["estimate_us_per_iteration"])
    return candidates[0]


def _tiling(mt, NT, KQ, R, ld, scratch_budget, model="mma", splits=(1, 2, 4)):
    """(MTW, WM, S) for one product: warps = WM x S (<= 8), S k16 phases, under the register (<= 64 accumulators)
    and scratch budgets. model 'mma' minimizes per-warp MMA count; model 'smem' also counts shared-memory operand
    traffic (the panel fragment is re-read by each of the WM warp columns; fused k-phase partials are written and
    read once each) at ~128 bytes per clock and takes the larger of the two (both are rough cycle estimates)."""
    best = None
    for S in splits:
        for WM in range(1, 9):
            if WM * S > 8 or WM > max(mt, 1):
                continue
            MTW = -(-max(mt, 1) // WM)
            if MTW * NT * 4 > 64:
                continue
            scratch = (S - 1) * R * ld * 4
            if scratch > scratch_budget:
                continue
            work = MTW * NT * -(-KQ // S)
            if model == "smem":
                smem = (WM * R * KQ * 16 * 2 + ld * KQ * 16 * 2 + 2 * S * R * ld * 4) / 128.0
                key = (max(work * 8.0, smem), scratch, S)
            else:
                key = (work, scratch, S)
            if best is None or key < best[0]:
                best = (key, (MTW, WM, S))
    if best is None:
        raise PanelRefusal("no warp tiling fits the register/scratch budget")
    return best[1]


def _vectors(g):
    """Bound row-free column vectors read by panel elementwise operations: name -> width."""
    vectors = {}
    for v, node in g.nodes.items():
        if node.kind != "elem":
            continue
        for r in dict.fromkeys(_refs((node.op.args, node.op.kwargs))):
            other = g.nodes.get(r)
            if other is None or other.kind == "const" or g.axes.get(r) is None:
                expr = g.column_expr(r, node.width)
                i = 0
                while True:
                    j = expr.find("VEC(", i)
                    if j < 0:
                        break
                    k = expr.index(")", j)
                    vectors[expr[j + 4:k]] = node.width
                    i = k + 1
    return vectors


def _split_lifetimes(g, plan, lives, gap, ahead):
    """Stored values -> shared pieces [(start, end)]: a demand gap longer than `gap` segments ends the current piece;
    the next piece starts at the hop `ahead` segments before the resuming demand (its prefetch point, + 0.25: after
    that product block's opening barrier, so every read of a value ending at that segment has completed)."""
    if ahead < 1 or gap < ahead:
        raise PanelRefusal("spill needs 1 <= spill_ahead <= spill_gap")
    pieces, spilled = {}, set()
    for v, (start, end, dtype, width) in lives.items():
        if v.endswith("@out") or v in g.carry_in:
            pieces[v] = [(start, end)]
            continue
        demand = sorted(x for x in plan["demand"].get(v, ()) if x >= start)
        segs, lo, prev = [], start, start
        for x in demand:
            if x - prev > gap:
                segs.append((lo, prev))
                lo = x - ahead + 0.25
                spilled.add(v)
            prev = x
        segs.append((lo, prev if demand else end))
        pieces[v] = segs
    return pieces, spilled


def _fit(g, plan, geo, opts, mode="ring", spill=False):
    C, R, KC, STAGES = geo.cluster, geo.rows, geo.chunk_k, geo.stages
    if KC != 64:
        raise PanelRefusal("packed chunks are 64 deep (128-byte swizzled rows)")
    products = g.products
    nodes = g.nodes
    NT = R // 8
    wmax = max(_wmax(nodes[p].width, C) for p in products)
    slot_bytes = wmax * 128
    kpad = {p: -(-nodes[p].K // KC) * KC for p in products}
    m = len(products)
    # Panel buffers alternate by product parity (global product counter); an odd count needs both at max size.
    Ks = [kpad[p] for p in products]
    kb = ([max(Ks[0::2]), max(Ks[1::2]) if m > 1 else 64] if m % 2 == 0 else [max(Ks)] * 2)
    vectors = _vectors(g)
    off = 0
    layout = {"mode": mode}
    lives = _lifetimes(g, plan)
    chained = True
    try:
        _allocate(lives, R, C, 0)
    except PanelRefusal:
        lives, chained = _lifetimes(g, plan, chained=False), False
    pieces, spilled = ({v: [(a[0], a[1])] for v, a in lives.items()}, set())
    if spill:
        if mode != "slice" or not opts.get("fuse", True):
            raise PanelRefusal("spill needs the fused slice mode")
        pieces, spilled = _split_lifetimes(g, plan, lives, int(opts["spill_gap"]), int(opts["spill_ahead"]))
        if not spilled:
            raise PanelRefusal("no stored value has a spillable idle interval")
    prefetches = []
    for v in sorted(spilled, key=lambda v: lives[v][0]):
        for i, (a, b) in enumerate(pieces[v][1:], start=1):
            hop = int(a - 0.25)
            first_use = min(x for x in plan["demand"][v] if x >= a)
            prefetches.append({"value": v, "piece": i, "hop": hop, "use": int(first_use)})
    layout["bars"] = off; off += 8 * (2 + 2 * STAGES + C + len(prefetches)); off = (off + 127) // 128 * 128
    layout["prods"] = off; off += ((24 * m + 127) // 128) * 128
    nch = sum(kpad[p] // KC for p in products)
    if m > 255:
        raise PanelRefusal("more than 255 products per iteration")
    if mode == "slice":
        # One panel buffer (cluster flow control) and two whole-product operand slots (128-byte aligned rows).
        if C < 2:
            raise PanelRefusal("slice mode needs a cluster of at least 2")
        layout["owner"] = off
        layout["panel0"] = layout["panel1"] = off
        off += R * max(Ks) * 2
        off = (off + 127) // 128 * 128
        sizes = [_wmax(nodes[p].width, C) * kpad[p] * 2 for p in products]
        if m % 2 == 0:   # product j always uses slot j % 2: size each slot for its own products
            slot_sizes = [max(sizes[0::2]), max(sizes[1::2]) if m > 1 else 128]
        else:
            slot_sizes = [max(sizes)] * 2
        slot_sizes = [(x + 127) // 128 * 128 for x in slot_sizes]
        if opts.get("mma", "auto") in ("wgmma", "auto"):
            # 128-byte swizzle atoms need 1024-byte aligned chunk bases (chunks are 2048 x units bytes apart)
            off = (off + 1023) // 1024 * 1024
            slot_sizes = [(x + 1023) // 1024 * 1024 for x in slot_sizes]
        layout["slots"] = [off, off + slot_sizes[0]]
        layout["slot_sizes"] = slot_sizes
        layout["slice_bytes"] = max(slot_sizes)
        layout["ring"] = off
        off += sum(slot_sizes)
    else:
        layout["owner"] = off; off += (nch + 127) // 128 * 128
        layout["panel0"] = off; off += R * kb[0] * 2
        layout["panel1"] = off; off += R * kb[1] * 2
        off = (off + 127) // 128 * 128
        layout["ring"] = off; off += STAGES * slot_bytes
    layout["fields"] = {}
    if opts.get("field_ahead", False) and mode == "slice":
        for v, node in nodes.items():
            segs = [s_ for s_, vals in plan["evaluate"].items() if v in vals]
            if node.kind == "field" and len(segs) == 1 and segs[0] < m and node.op.args[4] in ("normal", "uniform"):
                layout["fields"][v] = {"offset": off, "segment": segs[0]}
                off += (R * _wmax(node.width, C) * 4 + 127) // 128 * 128
    layout["vectors"] = {}
    for name, width in sorted(vectors.items()):
        layout["vectors"][name] = off
        off += (_wmax(width, C) * 4 + 15) // 16 * 16
    off = (off + 127) // 128 * 128
    plives = {}
    for v, segs in pieces.items():
        if len(segs) == 1:
            plives[v] = lives[v]
        else:
            for i, (a, b) in enumerate(segs):
                plives[f"{v}#{i}"] = (float(a), float(b), lives[v][2], lives[v][3])
    _, value_bytes = _allocate(plives, R, C, 0)
    budget = SHARED_LIMIT - off - value_bytes - 128
    if mode == "slice":
        budget = None   # partials reuse the product's own slot (bounded per product below)
    tiling, scratch = {}, 0
    requested = opts.get("mma", "auto")
    admissible = (mode == "slice" and opts.get("fuse", True) and not opts.get("overlap", False)
                  and wmax <= 64 and R % 8 == 0 and R <= 256 and opts.get("wg_desc", "a") == "a")
    if requested == "wgmma" and not admissible:
        raise PanelRefusal("wgmma products need the fused slice mode, owned widths <= 64 and R a multiple of 8")
    wgmma = requested == "wgmma" or (requested == "auto" and admissible)
    for p in products:
        ld = _wmax(nodes[p].width, C)
        if wgmma:
            j = products.index(p)
            own_slot = layout["slot_sizes"][j % 2] if m % 2 == 0 else layout["slice_bytes"]
            if 2 * R * ld * 4 > own_slot:
                raise PanelRefusal("wgmma k-split partials exceed the product slot")
            tiling[p] = (1, 1, 2)
            continue
        if mode == "slice":
            # fused partials need S tiles in the slot ((S - 1) is the bound _tiling applies): reserve one more tile
            j = products.index(p)
            own_slot = layout["slot_sizes"][j % 2] if m % 2 == 0 else layout["slice_bytes"]
            tiling[p] = _tiling(ld // 16, NT, kpad[p] // 16, R, ld, own_slot - R * ld * 4,
                                model=opts.get("tiling", "mma"),
                                splits=tuple(x for x in (1, 2, 4, 8) if x <= int(opts.get("max_ksplit", 8))))
        else:
            tiling[p] = _tiling(ld // 16, NT, kpad[p] // 16, R, ld, max(0, budget))
            scratch = max(scratch, (tiling[p][2] - 1) * R * ld * 4)
    layout["scratch"] = off
    off += (scratch + 127) // 128 * 128
    assignment, total = _allocate(plives, R, C, off)
    off += total
    if wgmma:
        # m64 A tiles read up to 64 rows of the last chunk (padding rows) and the shared base is realigned to 1024
        off += 64 * 128 + 1024
    # Spill area (global, per CTA) and prefetch records: one barrier per prefetch, one completion per iteration.
    spill_offsets, spill_bytes = {}, 0
    for v in sorted(spilled, key=lambda v: lives[v][0]):
        spill_offsets[v] = spill_bytes
        nbytes = R * _wmax(lives[v][3], C) * (2 if lives[v][2] == "bfloat16" else 4)
        spill_bytes += (nbytes + 127) // 128 * 128
    for i, pf in enumerate(prefetches):
        v = pf["value"]
        pf.update(bar=2 + 2 * STAGES + C + i, shared=assignment[f"{v}#{pf['piece']}"], spill=spill_offsets[v],
                  bytes=R * _wmax(lives[v][3], C) * (2 if lives[v][2] == "bfloat16" else 4))
    layout["pieces"] = {v: [(a, b, (f"{v}#{i}" if len(segs) > 1 else v)) for i, (a, b) in enumerate(segs)]
                        for v, segs in pieces.items()}
    layout["spill"] = spill_offsets
    layout["spill_bytes"] = spill_bytes
    layout["prefetches"] = prefetches
    if off > SHARED_LIMIT:
        raise PanelRefusal(f"shared memory {off} exceeds {SHARED_LIMIT}")
    # Rough per-iteration estimate (us): per product ~1.2 us exchange/synchronization plus MMA at ~2048 flop/clk
    # (1.7 GHz) per SM and weight bytes at ~40 GB/s per SM, whichever is larger.
    est = 0.0
    for p in products:
        w = _wmax(nodes[p].width, C)
        mma = 2.0 * R * w * kpad[p] / (2048 * 1.7e3)
        stream = w * kpad[p] * 2 / 40e3
        est += max(mma, stream) + 1.2 + (0.3 * kpad[p] / KC if mode == "ring" else 0.0)
    layout.update(values=assignment, kb=kb, kpad=kpad, wmax=wmax, slot_bytes=slot_bytes, shared_bytes=off,
                  estimate_us_per_iteration=round(est, 2), groups=geo.groups, vector_widths=vectors)
    sched = PanelSchedule(g, geo, tiling, 0, plan, off, layout)
    sched.diagnostics = sorted(set(opts["diagnostics"]))
    sched.fused = mode == "slice" and bool(opts.get("fuse", True))
    # per-source readiness inside the MMA loop measured slower (gpu-13..15: 119 vs 91 us/iteration); opt-in only
    sched.overlap = sched.fused and bool(opts.get("overlap", False)) and "skip_exchange" not in sched.diagnostics
    sched.spin = bool(opts.get("spin", False))
    sched.early_consumed = bool(opts.get("early_consumed", True))
    if opts.get("elementwise", "exact") not in ("exact", "fast"):
        raise PanelRefusal("elementwise must be exact or fast")
    sched.fast = opts.get("elementwise", "exact") == "fast"
    sched.unroll = bool(opts.get("unroll", False))   # unroll-2 measured slower (gpu-22)
    if opts.get("pack", "fused") not in ("fused", "serial"):
        raise PanelRefusal("pack must be fused or serial")
    sched.pack = opts.get("pack", "fused")
    sched.wgmma = wgmma
    if opts.get("mma", "auto") not in ("auto", "sync", "wgmma") or opts.get("wg_desc", "a") not in ("a", "b"):
        raise PanelRefusal("mma must be auto, sync or wgmma; wg_desc a or b")
    sched.wg_swap = 1 if opts.get("wg_desc", "a") == "b" else 0
    sched.wg_inline = bool(opts.get("wg_inline", False)) and sched.wgmma
    sched.share_segments = bool(opts.get("share_segments", False))
    sched.multicast = (not sched.overlap and opts.get("exchange", "multicast") == "multicast")
    if set(sched.diagnostics) - DIAGNOSTICS:
        raise PanelRefusal("unknown panel diagnostic")
    return sched


# ------------------------------------------------------------------------------------------------ code generation
import re as _re

_CANON_RULES = (
    (_re.compile(r"\bsm \+ (\d+)"), "sm + {}", "int"),
    (_re.compile(r"\bspill_cta \+ (\d+)"), "spill_cta + {}", "int"),
    (_re.compile(r"\bbars \+ (\d+)"), "bars + {}", "int"),
    (_re.compile(r"\(pg \+ (\d+)\)"), "(pg + {})", "int"),
)


def _canonical_segment(code):
    """(canonical body, [(kind, value)] arguments) of a segment block; equal bodies may share one function."""
    code = _re.sub(r"//[^\n]*", "", code)
    names = {}

    def rename(match):
        prefix, base = match.group(1), match.group(2)
        names.setdefault(base, f"x{len(names)}")
        return prefix + names[base]
    body = _re.sub(r"\b([nq]_)([A-Za-z0-9_]+)\b", rename, code)
    args, index = [], {}

    def lift(pattern, template, kind):
        def repl(match):
            # one argument per occurrence: bodies differing only in which offsets coincide still share
            alias = f"a{len(args)}"
            args.append((kind, match.group(1)))
            return template.format(alias)
        return repl
    for pattern, template, kind in _CANON_RULES:
        body = pattern.sub(lift(pattern, template, kind), body)
    pointers = {}

    def ptr(match):
        token = match.group(0)
        if token not in pointers:
            pointers[token] = f"p{len(pointers)}"
        return pointers[token]
    body = _re.sub(r"\b(?:key_)?in_\d+\b", ptr, body)
    return body, args, pointers


def _cname(v):
    return "n_" + v


def generate(sched, *, input_exprs, count_expr, first_carry, final_carry, ticks_expr="nullptr"):
    """Device function + call for the whole sampling loop.

    input_exprs: body-region input name -> C pointer expression (float*) at the composition call site.
    first_carry / final_carry: per carry, C pointer expressions of the loop-entry and loop-exit boundaries."""
    g, geo, plan, L = sched.g, sched.geo, sched.plan, sched.layout
    C, R, KC, STAGES, G = geo.cluster, geo.rows, geo.chunk_k, geo.stages, geo.groups
    nodes, products, index, m = g.nodes, g.products, plan["index"], plan["m"]
    wmax = L["wmax"]
    B = g.B
    used_inputs = []

    def IN(name):
        if name not in used_inputs:
            used_inputs.append(name)
        return f"in_{used_inputs.index(name)}"

    def colexpr(name, width):
        expr = g.column_expr(name, width)
        for token, render in (("IN(", lambda x: f"reinterpret_cast<const float*>({IN(x)})"),
                              ("VEC(", lambda x: f"reinterpret_cast<const float*>(sm + {L['vectors'][x]})[col - c0]")):
            out, i = [], 0
            while True:
                j = expr.find(token, i)
                if j < 0:
                    out.append(expr[i:])
                    break
                k = expr.index(")", j)
                out.append(expr[i:j])
                out.append(render(expr[j + len(token):k]))
                i = k + 1
            expr = "".join(out)
        return expr

    def ctype(dtype):
        return {"float32": "float", "bfloat16": "float", "bool": "bool"}[dtype]

    pieces = L.get("pieces", {})

    def slice_offset(v, s):
        segs = pieces.get(v)
        if not segs or len(segs) == 1:
            return L["values"][v]
        if s is None:   # leading dimension queries only (spill implies the fused slice mode)
            return L["values"][segs[0][2]]
        for a, b, name in segs:
            if a <= s <= b:
                return L["values"][name]
        raise PanelRefusal(f"{v}: no resident piece at segment {s}")

    def slice_ptr(v, s=None):
        dtype = "bfloat16" if nodes[v].kind == "product" else nodes[v].dtype
        t = "panel::bf16" if dtype == "bfloat16" else "float"
        return f"reinterpret_cast<{t}*>(sm + {slice_offset(v, s)})", dtype, _wmax(nodes[v].width, C)

    def stored(v):
        return v in L["values"] or v in pieces

    spill_offsets = L.get("spill", {})

    def spill_store(v, n):
        """Global copy of a spilled value (same [R][ld] slice layout) when it is formed."""
        ld = _wmax(nodes[v].width, C)
        dtype = "bfloat16" if nodes[v].kind == "product" else nodes[v].dtype
        if dtype == "bfloat16":
            return (f"panel::bf16_store4(reinterpret_cast<panel::bf16*>(spill_cta + {spill_offsets[v]}) + r * {ld} + j, "
                    f"{n}[0], {n}[1], {n}[2], {n}[3]);")
        return (f"*reinterpret_cast<float4*>(reinterpret_cast<float*>(spill_cta + {spill_offsets[v]}) + r * {ld} + j) = "
                f"make_float4({n}[0], {n}[1], {n}[2], {n}[3]);")

    fused_mode = L["mode"] == "slice" and sched.fused
    from_base = 2 + 2 * STAGES   # per-source panel barriers (overlap) follow the ring barriers
    wait_fn = "mbar_spin" if sched.spin else "mbar_wait"
    multicast = sched.multicast
    early = bool(sched.early_consumed) and "skip_exchange" not in sched.diagnostics
    stage_bytes = R * max(L["kpad"].values()) * 2
    def bufexpr(j):
        return f"{j % 2}" if m % 2 == 0 else f"((pg + {j}) & 1)"
    # ---- segment code
    def segment(s):
        values = plan["evaluate"].get(s, [])
        roots = [r for r in plan["roots"] if r[0] == s]
        by_width = defaultdict(list)
        for v in values:
            by_width[nodes[v].width].append(v)
        exch = [r for r in roots if r[2] == "exchange"]
        outs = [r for r in roots if r[2] == "output"]
        for r in exch + outs:
            by_width.setdefault(nodes[r[1]].width, [])
        # Slice mode: the previous product's output is formed here from its k-phase partials (fused reduction).
        fused = products[s - 1] if (fused_mode and s >= 1) else None
        if fused is not None and (s in plan["demand"].get(fused, ()) or stored(fused)):
            by_width.setdefault(nodes[fused].width, [])
        code = []
        for width, vals in by_width.items():
            u = _units(width)
            body = []
            defined = set()

            def operand(name):
                node = nodes.get(name)
                if node is None or node.kind == "const" or g.axes.get(name) is None:
                    e = colexpr(name, width)
                    return e, True
                if name == fused and name not in defined:
                    MTW_, WM_, S_ = sched.ksplit[name]
                    ld_ = _wmax(nodes[name].width, C)
                    slot_ = f"(((pg + {s - 1}) & 1) ? {L['slots'][1]} : {L['slots'][0]})"
                    body.append(f"float4 q_{name} = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(sm + {slot_}) + r * {ld_} + j);")
                    for t_ in range(1, S_):
                        body.append(f"{{ const float4 t_ = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(sm + {slot_}) + {t_ * R * ld_} + r * {ld_} + j); "
                                    f"q_{name}.x += t_.x; q_{name}.y += t_.y; q_{name}.z += t_.z; q_{name}.w += t_.w; }}")
                    body.append(f"const float {_cname(name)}[4] = {{panel::bf16_round(q_{name}.x), panel::bf16_round(q_{name}.y), "
                                f"panel::bf16_round(q_{name}.z), panel::bf16_round(q_{name}.w)}};")
                    defined.add(name)
                    return _cname(name) + "[l]", False
                if name not in defined:
                    # stored value (carry, product output or earlier materialized value)
                    ptr, dtype, ld = slice_ptr(name, s)
                    if dtype == "bfloat16":
                        body.append(f"const float4 q_{name} = panel::bf16_load4({ptr} + r * {ld} + j);")
                    else:
                        body.append(f"const float4 q_{name} = *reinterpret_cast<const float4*>({ptr} + r * {ld} + j);")
                    body.append(f"const float {_cname(name)}[4] = {{q_{name}.x, q_{name}.y, q_{name}.z, q_{name}.w}};")
                    defined.add(name)
                return _cname(name) + "[l]", False
            for v in vals:
                node = nodes[v]
                op = node.op
                base = _base(op.target)
                if node.kind == "field" and v in L.get("fields", {}):
                    fld = L["fields"][v]
                    ld_f = _wmax(node.width, C)
                    body.append(f"const float4 q_{v} = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(sm + {fld['offset']}) + r * {ld_f} + j);")
                    body.append(f"const float {_cname(v)}[4] = {{q_{v}.x, q_{v}.y, q_{v}.z, q_{v}.w}};")
                    defined.add(v)
                    continue
                if node.kind == "field":
                    key, c2, c3 = (_ref(a) for a in op.args[:3])
                    cexpr = lambda c: "(uint32_t)iteration" if c == g.k_input else f"(uint32_t)(*reinterpret_cast<const int64_t*>({IN(c)}))"
                    body.append(f"float {_cname(v)}[4];")
                    body.append("{ uint32_t w_[4]; latch_philox_block((uint32_t)(col4 >> 2), (uint32_t)grow, "
                                f"{cexpr(c2)}, {cexpr(c3)}, (uint32_t)key_{IN(key)}[0], (uint32_t)key_{IN(key)}[1], w_);")
                    tr = op.args[4]
                    body.append("#pragma unroll\n for (int l = 0; l < 4; ++l) " + _cname(v) + "[l] = " +
                                ("latch_field_normal(w_, col4 + l);" if tr == "normal" else "latch_field_uniform(w_[l]);") + " }")
                    defined.add(v)
                    continue
                if base == "fill" or base in CONSTRUCTORS:
                    continue
                args, is_expr = [], []
                for a in op.args:
                    r_ = _ref(a)
                    if r_ is None:
                        args.append(a)
                        is_expr.append(False)
                    else:
                        e, _ = operand(r_)
                        args.append(e)
                        is_expr.append(True)
                if base in IDENTITY:
                    expr = args[0]
                elif base == "_to_copy":
                    target = node.dtype
                    expr = f"panel::bf16_round({args[0]})" if target == "bfloat16" and g.dtype(_ref(op.args[0])) == "float32" else args[0]
                else:
                    expr = _render(base, op, args, is_expr, fast=sched.fast)
                body.append(f"{ctype(node.dtype)} {_cname(v)}[4];")
                body.append(f"#pragma unroll\n for (int l = 0; l < 4; ++l) {{ const int col = col4 + l; (void)col; {_cname(v)}[l] = {expr}; }}")
                defined.add(v)
            stores = []
            spilled_here = False
            if fused is not None and width == nodes[fused].width and stored(fused):
                e_, _ = operand(fused)
                ptr, dtype, ld = slice_ptr(fused, s)
                n_ = _cname(fused)
                stores.append(f"panel::bf16_store4({ptr} + r * {ld} + j, {n_}[0], {n_}[1], {n_}[2], {n_}[3]);")
                if fused in spill_offsets:
                    stores.append(spill_store(fused, n_))
                    spilled_here = True
            for v in vals:
                if v in plan["materialized"] and plan["first"].get(v) == s and stored(v):
                    ptr, dtype, ld = slice_ptr(v, s)
                    n = _cname(v)
                    if dtype == "bfloat16":
                        stores.append(f"panel::bf16_store4({ptr} + r * {ld} + j, {n}[0], {n}[1], {n}[2], {n}[3]);")
                    else:
                        stores.append(f"*reinterpret_cast<float4*>({ptr} + r * {ld} + j) = make_float4({n}[0], {n}[1], {n}[2], {n}[3]);")
                    if v in spill_offsets:
                        stores.append(spill_store(v, n))
                        spilled_here = True
            for r_ in exch:
                if nodes[r_[1]].width != width:
                    continue
                e, _ = operand(r_[1])
                n = e.replace("[l]", "")
                j_ = index[r_[3]]
                if multicast:
                    stores.append(f"panel::stage_store4(stage + ((pg + {j_}) & 1) * {stage_bytes}LL, {R}, r, c0 + j, "
                                  f"{n}[0], {n}[1], {n}[2], {n}[3]);")
                else:
                    stores.append(f"panel::panel_store4(pbuf[{bufexpr(j_)}], {R}, r, c0 + j, {n}[0], {n}[1], {n}[2], {n}[3]);")
            for r_ in outs:
                if nodes[r_[1]].width != width:
                    continue
                e, _ = operand(r_[1])
                n = e.replace("[l]", "")
                carry = g.carry_in[r_[3]]
                ptr, dtype, ld = slice_ptr(carry)
                stores.append(f"*reinterpret_cast<float4*>({ptr} + r * {ld} + j) = make_float4({n}[0], {n}[1], {n}[2], {n}[3]);")
            if not stores:
                continue
            code.append(f"{{ // segment {s}, width {width}\n"
                        f"const int u_lo = ({u} * rank) / {C}, u_hi = ({u} * (rank + 1)) / {C};\n"
                        f"const int c0 = 16 * u_lo, wq = 4 * (u_hi - u_lo);\n"
                        + ("#pragma unroll 2\n" if sched.unroll else "") +
                        f"for (int e = threadIdx.x; e < rows_valid * wq; e += blockDim.x) {{\n"
                        f"const int r = e / wq, j = (e - r * wq) * 4, col4 = c0 + j, grow = row0 + r; (void)grow;\n"
                        + "\n".join(body) + "\n" + "\n".join(stores) + "\n}\n"
                        # generic-proxy spill stores become visible to the later bulk (async-proxy) prefetch
                        + ('asm volatile("fence.proxy.async.global;" ::: "memory");\n' if spilled_here else "")
                        + "}")
        waits = [f"panel::{wait_fn}(bars + {pf['bar']}, (uint32_t)(iteration & 1));"
                 for pf in L.get("prefetches", []) if pf["use"] == s]
        return "\n".join(waits + code)

    # ---- products
    prods_init, pack_code = [], []
    chunk0, pack_offset = 0, 0
    for j, p in enumerate(products):
        node = nodes[p]
        kmajor = 1 if node.sk == 1 else 0
        stride = node.sn if kmajor else node.sk
        u = _units(node.width)
        kpad = L["kpad"][p]
        nch = kpad // KC
        prods_init.append(
            f"if (threadIdx.x == {j % 256}) {{ panel::Prod& P = prods[{j}]; "
            f"P.pack = pack + {pack_offset}LL + 16LL * (({u} * rank) / {C}) * {kpad}; "
            f"P.width = 16 * ((({u} * (rank + 1)) / {C}) - (({u} * rank) / {C})); P.chunk0 = {chunk0}; "
            f"P.nchunks = {nch}; P.pad = 0; }}")
        pack_code.append(f"panel::pack_operand(__PANEL_PACK__ + {pack_offset}LL, reinterpret_cast<const panel::bf16*>({input_exprs[node.weight]}), "
                         f"{kmajor}, {stride}LL, {node.K}, {kpad}, {node.width}, {C});")
        chunk0 += nch
        pack_offset += node.width * kpad
    nchunks_iter = chunk0
    pack_elements = pack_offset
    if sched.pack == "fused":
        # P9-D: one grid-strided loop over the 16-byte vectors of every product's pack (independent writes).
        cases, cum, offset = [], 0, 0
        for j, p in enumerate(products):
            node = nodes[p]
            kmajor = 1 if node.sk == 1 else 0
            stride = node.sn if kmajor else node.sk
            kpad = L["kpad"][p]
            vectors = node.width * (kpad // 8)
            cases.append(f"{'if' if j == 0 else 'else if'} (v < {cum + vectors}LL) panel::pack_vector(__PANEL_PACK__ + {offset}LL, "
                         f"reinterpret_cast<const panel::bf16*>({input_exprs[node.weight]}), {kmajor}, {stride}LL, {node.K}, "
                         f"{kpad}, {node.width}, {C}, v - {cum}LL);")
            cum += vectors
            offset += node.width * kpad
        pack_code = [f"for (long long v = (long long)blockIdx.x * blockDim.x + threadIdx.x; v < {cum}LL; "
                     f"v += (long long)gridDim.x * blockDim.x) {{\n" + "\n".join(cases) + "\n}"]
    starts = []
    acc_ = 0
    for p in products:
        starts.append(acc_)
        acc_ += L["kpad"][p] // KC
    owner_search = " ".join(f"if (e >= {c0_}) p_ = {j};" for j, c0_ in enumerate(starts))
    if L["mode"] == "slice":
        prologue_loads = " ".join(
            f"if (threadIdx.x == 0 && {i} < total_products) panel::load_slice(sm + {L['slots'][i]}, bars + 2 + {i}, "
            f"prods[{i % m}].pack, (uint32_t)(prods[{i % m}].width * {L['kpad'][products[i % m]] * 2}));"
            for i in range(2))
    else:
        prologue_loads = (f"if (threadIdx.x == 0) for (int i = 0; i < {STAGES} - panel::LAG && i < total; ++i) "
                          f"panel::issue<{STAGES}, {L['slot_bytes']}>(ring, prods, {m}, {nchunks_iter}, i);")
    step = []
    stamp = lambda i: (f"if (ticks && iteration == 0 && blockIdx.x == 0 && threadIdx.x == 0) ticks[{i}] = stamp();")

    def field_fill(v, it_expr, part=None):
        """Generate field v at iteration it_expr into its buffer: all owned elements, or slice part = (a, n)."""
        node, fld = nodes[v], L["fields"][v]
        op = node.op
        key, c2, c3 = (_ref(a) for a in op.args[:3])
        cex = lambda c: f"(uint32_t)({it_expr})" if c == g.k_input else f"(uint32_t)(*reinterpret_cast<const int64_t*>({IN(c)}))"
        u_, ld_f = _units(node.width), _wmax(node.width, C)
        rng = ("const int lo_ = 0, hi_ = E_;" if part is None else
               f"const int lo_ = (E_ * {part[0]}) / {part[1]}, hi_ = (E_ * {part[0] + 1}) / {part[1]};")
        value = "latch_field_normal(w_, col4 + l)" if op.args[4] == "normal" else "latch_field_uniform(w_[l])"
        return (f"{{ const int u_lo = ({u_} * rank) / {C}, u_hi = ({u_} * (rank + 1)) / {C};\n"
                f"const int c0 = 16 * u_lo, wq = 4 * (u_hi - u_lo), E_ = rows_valid * wq; {rng}\n"
                f"float* fb_ = reinterpret_cast<float*>(sm + {fld['offset']});\n"
                f"for (int e = lo_ + threadIdx.x; e < hi_; e += blockDim.x) {{\n"
                f"const int r = e / wq, j = (e - r * wq) * 4, col4 = c0 + j, grow = row0 + r;\n"
                f"uint32_t w_[4]; latch_philox_block((uint32_t)(col4 >> 2), (uint32_t)grow, {cex(c2)}, {cex(c3)}, "
                f"(uint32_t)key_{IN(key)}[0], (uint32_t)key_{IN(key)}[1], w_);\n"
                f"#pragma unroll\nfor (int l = 0; l < 4; ++l) fb_[r * {ld_f} + j + l] = {value};\n}}\n}}")

    def field_slices(s_):
        code = []
        for v, fld in L.get("fields", {}).items():
            first = fld["segment"]
            if s_ >= first:
                code.append(f"if (iteration + 1 < count) " + field_fill(v, "iteration + 1", (s_ - first, m - first)))
        return "\n".join(code)
    shared_fns, shared_order = {}, []

    def emit_segment(code):
        if not sched.share_segments or not code.strip():
            return code
        body, args, pointers = _canonical_segment(code)
        key = (body, tuple(k for k, _ in args), tuple(sorted(pointers.values())))
        if key not in shared_fns:
            name = f"panel_seg_{len(shared_fns)}"
            params = ["unsigned char* sm", "unsigned char* stage", "unsigned char* spill_cta", "uint64_t* bars",
                      "int rank", "int rows_valid", "int row0", "int iteration", "int pg", "int count"]
            params += [f"int a{i}" for i in range(len(args))]
            for token, alias in pointers.items():
                params.append((f"const int64_t* {alias}" if token.startswith("key_") else f"float* {alias}"))
            shared_fns[key] = name
            shared_order.append(f"__device__ __noinline__ void {name}({', '.join(params)}) {{\n{body}\n}}\n")
        call_args = ["sm", "stage", "spill_cta", "bars", "rank", "rows_valid", "row0", "iteration", "pg", "count"]
        call_args += [value for _, value in args]
        call_args += list(pointers)
        return f"{shared_fns[key]}({', '.join(call_args)});"

    for s in range(m + 1):
        step.append(emit_segment(segment(s)))
        step.append(stamp(1 + 4 * s))
        if s == m:
            break
        p = products[s]
        node = nodes[p]
        K = node.K
        uK = _units(K)
        out_ptr, _, ld = slice_ptr(p)
        buf = bufexpr(s)
        kpad = L["kpad"][p]
        MTW, WM, S = sched.ksplit[p]
        tail = ""
        if kpad != K:
            # zero the panel's padded k-range [K, Kpad) (no CTA owns it; another product may have used the buffer)
            tail = (f"for (int e = threadIdx.x; e < {R * (kpad - K) // 8}; e += blockDim.x) "
                    f"reinterpret_cast<uint4*>(pbuf[b_] + {(K // 8) * R * 16})[e] = make_uint4(0, 0, 0, 0);\n"
                    + ("panel::fence_proxy_async();\n" if sched.wgmma else ""))
        if L["mode"] == "slice":
            nxt = (s + 2) % m
            step.append(
                f"{{ // product {s}: {p} [{node.width}] <- K {K} (slice mode)\n"
                f"const int P_ = pg + {s}; const int b_ = 0;\n"
                + tail +
                f"__syncthreads();\n"
                + "".join(f"if (threadIdx.x == 96) panel::load_slice(sm + {pf['shared']}, bars + {pf['bar']}, "
                          f"reinterpret_cast<const panel::bf16*>(spill_cta + {pf['spill']}), {pf['bytes']}u);\n"
                          for pf in L.get("prefetches", []) if pf["hop"] == s)
                + (f"if (threadIdx.x == 0 && P_ >= 1 && P_ + 1 < total_products) panel::load_slice("
                   f"sm + ((P_ + 1) & 1 ? {L['slots'][1]} : {L['slots'][0]}), bars + 2 + ((P_ + 1) & 1), prods[{(s + 1) % m}].pack, "
                   f"(uint32_t)(prods[{(s + 1) % m}].width * {L['kpad'][products[(s + 1) % m]] * 2}));\n"
                   if sched.fused else "")
                # arm the arrival barrier(s) first: the pushing threads may wait on cluster flow control
                + ("" if "skip_exchange" in sched.diagnostics else
                   (f"if (threadIdx.x >= 32 && threadIdx.x < 32 + {C} && threadIdx.x - 32 != rank) "
                    f"panel::mbar_arrive_expect_tx(bars + {from_base} + (threadIdx.x - 32), "
                    f"(uint32_t)({32 * R} * ((({uK} * (threadIdx.x - 32 + 1)) / {C}) - (({uK} * (threadIdx.x - 32)) / {C}))));\n"
                    if sched.overlap else
                    f"if (threadIdx.x == 0) panel::mbar_arrive_expect_tx(bars + 0, (uint32_t)({32 * R * uK}));\n"
                    if multicast else
                    f"if (threadIdx.x == 0) panel::mbar_arrive_expect_tx(bars + 0, (uint32_t)({32 * R} * ({uK} - "
                    f"((({uK} * (rank + 1)) / {C}) - (({uK} * rank) / {C})))));\n"))
                + (f"panel::multicast_slice(pbuf[0], stage + (P_ & 1) * {stage_bytes}LL, {R}, ({uK} * rank) / {C}, "
                   f"({uK} * (rank + 1)) / {C}, {C}, bars + 0, bars + 1, P_ > 0, (uint32_t)((P_ - 1) & 1));\n"
                   if multicast else
                   f"panel::push_slice_flow(pbuf[0], {R}, ({uK} * rank) / {C}, ({uK} * (rank + 1)) / {C}, {C}, rank, "
                   f"bars + {from_base} + rank, bars + 1, P_ > 0, (uint32_t)((P_ - 1) & 1));\n"
                   if sched.overlap else
                   f"panel::push_slice_flow(pbuf[0], {R}, ({uK} * rank) / {C}, ({uK} * (rank + 1)) / {C}, {C}, rank, "
                   f"bars + 0, bars + 1, P_ > 0, (uint32_t)((P_ - 1) & 1));\n")
                + stamp(2 + 4 * s) + "\n"
                + field_slices(s) + "\n"
                + ("" if ("skip_exchange" in sched.diagnostics or sched.overlap) else
                   f"panel::{wait_fn}(bars + 0, (uint32_t)(P_ & 1));\n")
                + f"panel::{wait_fn}(bars + 2 + (P_ & 1), (uint32_t)((P_ >> 1) & 1));\n"
                + stamp(3 + 4 * s) + "\n"
                + (f"panel::product_slice_wg<{R}, {kpad // KC}>(sm + (P_ & 1 ? {L['slots'][1]} : {L['slots'][0]}), "
                   f"prods[{s}].width, pbuf[0], {ld}, {'bars + 1' if early else 'nullptr'}, rank, {C});\n"
                   if sched.wgmma else
                   f"panel::product_slice_partials<{R}, {MTW}, {WM}, {S}, {1 if sched.overlap else 0}>(sm + (P_ & 1 ? {L['slots'][1]} : {L['slots'][0]}), "
                   f"prods[{s}].width, {kpad // KC}, pbuf[0], {ld}, bars + {from_base}, (uint32_t)(P_ & 1), {uK}, rank, {C}, "
                   f"{'bars + 1' if early else 'nullptr'});\n"
                   if sched.fused else
                   f"panel::product_slice<{R}, {MTW}, {WM}, {S}>(sm + (P_ & 1 ? {L['slots'][1]} : {L['slots'][0]}), "
                   f"prods[{s}].width, {kpad // KC}, pbuf[0], {out_ptr}, {ld});\n")
                + f"__syncthreads();\n"
                + ("" if ("skip_exchange" in sched.diagnostics or (sched.fused and early)) else
                   f"panel::signal_consumed(bars + 1, {C}, rank);\n")
                + ("" if sched.fused else
                   f"if (threadIdx.x == 0 && P_ + 2 < total_products) panel::load_slice(sm + (P_ & 1 ? {L['slots'][1]} : {L['slots'][0]}), "
                   f"bars + 2 + (P_ & 1), prods[{nxt}].pack, (uint32_t)(prods[{nxt}].width * {L['kpad'][products[nxt]] * 2}));\n")
                + f"}}")
            step.append(stamp(4 + 4 * s))
            continue
        step.append(
            f"{{ // product {s}: {p} [{node.width}] <- K {K}\n"
            f"const int b_ = {buf};\n"
            + tail +
            f"__syncthreads();\n"
            + (f"panel::multicast_slice(pbuf[b_], stage + ((pg + {s}) & 1) * {stage_bytes}LL, {R}, ({uK} * rank) / {C}, "
               f"({uK} * (rank + 1)) / {C}, {C}, &bars[b_], bars, 0, 0u);\n"
               if multicast else
               f"panel::push_slice(pbuf[b_], {R}, ({uK} * rank) / {C}, ({uK} * (rank + 1)) / {C}, {C}, rank, &bars[b_]);\n")
            + ("" if "skip_exchange" in sched.diagnostics else
               f"if (threadIdx.x == 0) panel::mbar_arrive_expect_tx(&bars[b_], (uint32_t)({32 * R * uK}));\n"
               if multicast else
               "" if "skip_exchange" in sched.diagnostics else
               f"if (threadIdx.x == 0) panel::mbar_arrive_expect_tx(&bars[b_], (uint32_t)({32 * R} * ({uK} - "
               f"((({uK} * (rank + 1)) / {C}) - (({uK} * rank) / {C})))));\n")
            + stamp(2 + 4 * s) + "\n"
            + ("" if "skip_exchange" in sched.diagnostics else
               f"panel::mbar_wait(&bars[b_], (parity >> b_) & 1u); parity ^= (1u << b_);\n")
            + stamp(3 + 4 * s) + "\n" +
            f"panel::product<{R}, {STAGES}, {L['slot_bytes']}, {MTW}, {WM}, {S}>(prods, {m}, {s}, {nchunks_iter}, g, total, "
            f"ring, pbuf[b_], {out_ptr}, {ld}, scratch);\n"
            f"__syncthreads();\n}}")
        step.append(stamp(4 + 4 * s))
    # ---- function
    carry_load, carry_store = [], []
    for i, v in enumerate(g.carry_in):
        ptr, dtype, ld = slice_ptr(v)
        if dtype != "float32":
            raise PanelRefusal("carry must be FP32")
        u = _units(nodes[v].width)
        width = nodes[v].width
        for target, lines_, direction in ((first_carry[i], carry_load, "load"), (final_carry[i], carry_store, "store")):
            idx = len(used_inputs)
            body = (f"{{ const int u_lo = ({u} * rank) / {C}, u_hi = ({u} * (rank + 1)) / {C};\n"
                    f"const int c0 = 16 * u_lo, wq = 4 * (u_hi - u_lo);\n"
                    f"float* gp = reinterpret_cast<float*>(carry_{direction}_{i});\n"
                    f"for (int e = threadIdx.x; e < rows_valid * wq; e += blockDim.x) {{\n"
                    f"const int r = e / wq, j = (e - r * wq) * 4;\n")
            if direction == "load":
                body += (f"*reinterpret_cast<float4*>({ptr} + r * {ld} + j) = "
                         f"*reinterpret_cast<const float4*>(gp + (long long)(row0 + r) * {width} + c0 + j);\n}}\n}}")
            else:
                body += (f"*reinterpret_cast<float4*>(gp + (long long)(row0 + r) * {width} + c0 + j) = "
                         f"*reinterpret_cast<const float4*>({ptr} + r * {ld} + j);\n}}\n}}")
            lines_.append(body)
    vector_loads = []
    for name, off in L["vectors"].items():
        width = L["vector_widths"][name]
        u = _units(width)
        vector_loads.append(
            f"{{ const int c0 = 16 * (({u} * rank) / {C}), w = 16 * (({u} * (rank + 1)) / {C}) - c0;\n"
            f"for (int e = threadIdx.x; e < w; e += blockDim.x) reinterpret_cast<float*>(sm + {off})[e] = "
            f"reinterpret_cast<const float*>({IN(name)})[c0 + e]; }}")
    spilling = bool(L.get("spill_bytes", 0))
    params = ([f"float* in_{i}" for i in range(len(used_inputs))] + ["const panel::bf16* pack", "unsigned char* stage_all"]
              + (["unsigned char* spill_all"] if spilling else []))
    carry_params = ([f"float* carry_load_{i}" for i in range(len(g.carry_in))]
                    + [f"float* carry_store_{i}" for i in range(len(g.carry_in))])
    key_decls = []
    for name in list(used_inputs):
        if g.values[name].shape and tuple(g.values[name].shape) == (2,) and g.dtype(name) == "int64":
            i = used_inputs.index(name)
            key_decls.append(f"const int64_t key_in_{i}[2] = {{reinterpret_cast<const int64_t*>(in_{i})[0], "
                             f"reinterpret_cast<const int64_t*>(in_{i})[1]}};")
    panel_bytes = (R * max(L["kpad"].values()) * 2 if L["mode"] == "slice" else R * (L['kb'][0] + L['kb'][1]) * 2)
    zero_panels = (f"for (int e = threadIdx.x; e < {panel_bytes // 16}; e += blockDim.x) "
                   f"reinterpret_cast<uint4*>(sm + {L['panel0']})[e] = make_uint4(0, 0, 0, 0);"
                   + (" panel::fence_proxy_async();" if sched.wgmma else ""))
    align_expr = (" + ((1024u - (panel::saddr(panel_smem) & 1023u)) & 1023u)" if sched.wgmma else "")
    wg_helper = ""
    if sched.wgmma:
        n = R // 2
        regs = ", ".join(f"%{i}" for i in range(n))
        outs = ", ".join(f'"+f"(d[{i}])' for i in range(n))
        if sched.wg_swap:
            raise PanelRefusal("wg_desc b was measured wrong (gpu-08 gates); only a is emitted")
        parts_ = []
        for nc in sorted({L["kpad"][p_] // KC for p_ in products}):
            lines = ["{", ".reg .pred p;", ".reg .b64 a, b;", f"setp.ne.b32 p, %{n + 3}, 0;", f"mov.b64 a, %{n};",
                     f"mov.b64 b, %{n + 1};", "wgmma.fence.sync.aligned;"]
            mma = f"wgmma.mma_async.sync.aligned.m64n{R}k16.f32.bf16.bf16 {{{regs}}}, a, b, p, 1, 1, 0, 0;"
            for c in range(nc):
                lines += [mma, "add.s64 a, a, 2;", f"add.s64 b, b, {2 * R};", mma]
                if c + 1 < nc:
                    lines += [f"add.s64 a, a, %{n + 2};", f"add.s64 b, b, {6 * R};"]
            lines += ["wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 0;", "}"]
            text = "\\n".join(lines)
            parts_.append("template <> __device__ __forceinline__ void wg_steps<" + str(R) + ", " + str(nc) + ">(float (&d)["
                          + str(n) + "], uint64_t da, uint64_t db, uint64_t jump) {\n  asm volatile(\"" + text + "\\n\"\n  : "
                          + outs + "\n  : \"l\"(da), \"l\"(db), \"l\"(jump), \"r\"(1) : \"memory\");\n}\n")
        wg_helper = "namespace panel {\n" + "".join(parts_) + "}  // namespace panel\n"
    spill_decl = (f"\n  unsigned char* spill_cta = spill_all + (long long)blockIdx.x * {L['spill_bytes']}LL;"
                  if spilling else ("\n  unsigned char* spill_cta = nullptr;" if sched.share_segments else ""))
    function = wg_helper + "".join(shared_order) + f"""
__device__ {'__forceinline__' if sched.wg_inline else '__noinline__'} void panel_loop({', '.join(params + carry_params + ['int count', 'unsigned long long* ticks'])}) {{
  extern __shared__ __align__(128) unsigned char panel_smem[];
  unsigned char* sm = panel_smem{align_expr};
  const int rank = (int)panel::cluster_rank();
  const int group = (int)(blockIdx.x / {C});
  const int row0 = group * {R};
  const int rows_valid = ({B} - row0) < {R} ? ({B} - row0) : {R};
  unsigned char* stage = stage_all + (long long)group * {2 * stage_bytes}LL; (void)stage;{spill_decl}
  uint64_t* bars = reinterpret_cast<uint64_t*>(sm + {L['bars']});
  panel::Prod* prods = reinterpret_cast<panel::Prod*>(sm + {L['prods']});
  unsigned char* pbuf[2] = {{sm + {L['panel0']}, sm + {L['panel1']}}};
  panel::Ring ring;
  ring.slots = sm + {L['ring']}; ring.full = bars + 2; ring.empty = bars + 2 + {STAGES};
  ring.owner = sm + {L['owner']};
  for (int e = threadIdx.x; e < {nchunks_iter if L["mode"] == "ring" else 0}; e += blockDim.x) {{
    int p_ = 0; {owner_search}
    (sm + {L['owner']})[e] = (unsigned char)p_;
  }}
  float* scratch = reinterpret_cast<float*>(sm + {L['scratch']});
  {' '.join(key_decls)}
  {zero_panels}
  {chr(10).join(prods_init)}
  {chr(10).join(vector_loads)}
  if (threadIdx.x == 0) {{
    panel::mbar_init(bars + 0, 1); panel::mbar_init(bars + 1, {max(C - 1, 1) if L["mode"] == "slice" else 1});
    for (int i = 0; i < {STAGES}; ++i) {{ panel::mbar_init(ring.full + i, 1); panel::mbar_init(ring.empty + i, blockDim.x / 32); }}
    for (int i = 0; i < {C + len(L.get("prefetches", []))}; ++i) panel::mbar_init(bars + {2 + 2 * STAGES} + i, 1);
    panel::mbar_init_fence();
  }}
  {chr(10).join(carry_load)}
  {chr(10).join("if (count > 0) " + field_fill(v, "0") for v in L.get("fields", {}))}
  __syncthreads();
  panel::cluster_sync();
  const int total = count * {nchunks_iter};
  int g = 0;
  const int total_products = count * {m}; (void)total_products;
  {prologue_loads}
  uint32_t parity = 0u;
  int pg = 0; (void)pg;
  if (ticks && blockIdx.x == 0 && threadIdx.x == 0) ticks[0] = stamp();
  for (int iteration = 0; iteration < count; ++iteration) {{
{chr(10).join(step)}
    __syncthreads();
    pg += {m};
  }}
  {chr(10).join(carry_store)}
  __syncthreads();
  panel::cluster_sync();
}}
"""
    call_args = ([f"(float*)({input_exprs[n]})" for n in used_inputs] + ["(const panel::bf16*)(__PANEL_PACK__)"]
                 + ["(unsigned char*)(__PANEL_STAGE__)"] + (["(unsigned char*)(__PANEL_SPILL__)"] if spilling else []))
    call_args += [f"(float*)({c})" for c in first_carry] + [f"(float*)({c})" for c in final_carry]
    call = (f"if (blockIdx.x < {G * C}) panel_loop({', '.join(call_args + [count_expr, ticks_expr])});\n"
            f"if (gridDim.x < {G * C} && blockIdx.x == 0 && threadIdx.x == 0) __trap();\n")
    header_hash = hashlib.sha256(_header_path(HEADER).read_bytes()).hexdigest()
    manifest = {
        "selected": True, "rule": "p8d-panel-engine-v1", "mode": L["mode"], "fused_reduction": bool(sched.fused),
        "per_source_overlap": bool(getattr(sched, "overlap", False)),
        "elementwise": "fast (POLICY amendment p1)" if sched.fast else "exact (ops.lower expressions)", "cluster": C, "rows_per_group": R, "groups": G,
        "engine_ctas": G * C, "stages": STAGES, "chunk_k": KC, "products": [
            {"name": p, "K": nodes[p].K, "N": nodes[p].width, "weight": nodes[p].weight,
             "orientation": "k-major" if nodes[p].sk == 1 else "n-major",
             "tiling": dict(zip(("mtw", "wm", "ksplit"), sched.ksplit[p]))}
            for p in products],
        "materialized": sorted(plan["materialized"]),
        "stored_slices": {v: int(o) for v, o in L["values"].items()},
        "spill": {"values": {v: int(o) for v, o in spill_offsets.items()}, "bytes_per_cta": int(L.get("spill_bytes", 0)),
                  "prefetches": [{k: pf[k] for k in ("value", "hop", "use", "bytes")} for pf in L.get("prefetches", [])],
                  "rule": "idle intervals longer than spill_gap segments live in a per-CTA global area; one bulk "
                          "copy brings the exact stored bits back spill_ahead hops before the next use"},
        "spill_area_bytes": int(L.get("spill_bytes", 0)) * G * C,
        "segments": {str(s): plan["evaluate"].get(s, []) for s in range(m + 1)},
        "shared_bytes": sched.shared_bytes, "estimate_us_per_iteration": L["estimate_us_per_iteration"],
        "chunks_per_iteration": nchunks_iter, "diagnostic_only": bool(sched.diagnostics),
        "pack_elements": pack_elements, "pack_code": "\n".join(pack_code),
        "exchange": "multicast" if multicast else ("overlap-push" if sched.overlap else "push"),
        "field_ahead": {v: f["segment"] for v, f in L.get("fields", {}).items()},
        "shared_segments": {"functions": len(shared_fns), "segments": m + 1} if sched.share_segments else None,
        "products_engine": (f"wgmma m64n{R}k16, k-split 2, B descriptor {'b' if sched.wg_swap else 'a'}"
                            if sched.wgmma else "mma.sync m16n8k16"),
        "stage_bytes": (2 * stage_bytes * G) if multicast else 0,
        "pack": "loop-invariant product operands re-laid out once per update into per-rank swizzled 64-deep chunks",
        "defines": "".join(f"#define PANEL_DIAG_{d.upper()}\n" for d in sorted(sched.diagnostics)),
        "diagnostics": sched.diagnostics,
        "tick_layout": {"offset": "ticks_expr base", "start": 0,
                        "boundaries": [f"{('segment', 'push', 'wait', 'product')[i]} {s}" for s in range(m + 1)
                                       for i in range(4) if not (s == m and i > 0)]},
        "arithmetic": "products: BF16 operands, FP32 accumulation (k-split partial chains summed in fixed order), "
                      "one BF16 rounding; elementwise: ops.lower FP32 expressions, --fmad=false",
    }
    return function, call, header_hash, manifest


def plan_and_generate(graph, options, **kw):
    sched = plan_panels(graph, options)
    function, call, header_hash, manifest = generate(sched, **kw)
    return sched, function, call, header_hash, manifest
