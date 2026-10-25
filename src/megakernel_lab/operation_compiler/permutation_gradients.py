"""Permutation-aware embedding gradients (LATCH-DISPATCH-GRAD): inverse gathers and fixed-k row sums.

``aten.embedding_dense_backward(g, idx, V, padding_idx, False)`` sums, for every table row r, the gradient rows of
all occurrences ``c`` with ``idx[c] == r``. The general engines must discover the occurrences at run time (sort,
segment, reduce). When the graph itself proves how the occurrences are arranged this lowering reads them directly:

* ``perm(n, inverse=Q)``: ``idx`` is a permutation of ``[0, n)`` whose inverse ``Q`` is a graph value
  (``idx[Q[r]] == r``). Every row has exactly one occurrence, ``Q[r]``: ``out[r] = +0.0 + g[Q[r]]``.
* ``repeat(n, d, inverse=Q)``: ``idx = P // d`` (floor division by a static positive integer) where ``P`` is a
  permutation of ``[0, n*d)`` with inverse ``Q``. Row r occurs exactly d times, at ``Q[r*d + s]`` for s < d, and
  nowhere else (``P[c] // d == r`` iff ``P[c]`` lies in ``[r*d, r*d + d)``). The d positions are sorted ascending
  in registers and summed from +0.0 in that order.

Facts come only from operation semantics, never from names or shapes of a model. The single source of permutations
is ``latch_ops.ragged_order`` (stable bucket order ``order`` and its inverse ``inverse``; the routing ids are range
checked at run time, so both outputs are mutually inverse permutations of ``[0, tokens * choices)``); identity views
or copies of a 1-D int64 vector carry the fact; ``floor_divide`` / ``div(rounding_mode='floor')`` by a static
integer that divides the extent turns a permutation into a repeat. Anything else (token embeddings, arbitrary or
repeated indices, other index arithmetic) keeps the selected general engine.

Arithmetic: each row adds its contributions in ascending occurrence ordinal from +0.0 with round-to-nearest FP32
additions, the ordered policy. The general deterministic engine (``segmented_v1``) sums a row in exactly that order
whenever the row's occurrences fall inside one of its chunks; with the d occurrences of every row consecutive in
sorted order this holds whenever d divides the chunk size (d = 1, 2 always; powers of two <= 4 for every extent),
so those programs are bitwise unchanged. Otherwise only the association differs from the segmented engine
(``reassociated_fp32_v1`` covers both). Contributions use the same per-element values as the general engines:
FP32 rows, exactly widened BF16 rows (``latch_row_widen``), or ``float(bf16) * s`` rounded once
(``latch_row_scale``); BF16 results round the completed FP32 sum once. A negative-zero contribution still yields
+0 (``+0 + -0``), as the general engines. Every claimed occurrence is checked at run time (``idx[q] == r``); a
stale or corrupt inverse traps instead of producing a wrong gradient. No scratch, sorting or grid barriers.

``ENABLED = False`` (or ``LATCH_OPS_PERMUTATION_GRADIENTS=0``) restores the general engines.
"""
from __future__ import annotations

import os

ENABLED = os.environ.get("LATCH_OPS_PERMUTATION_GRADIENTS", "1") != "0"
ORDER = "latch_ops.ragged_order.default"
EDB = "aten.embedding_dense_backward.default"
MAX_REPEAT = 16
# Identity movement of a whole int64 vector (contiguous input and output, no offset, same element count).
_IDENTITY = {"view", "_unsafe_view", "reshape", "alias", "detach", "clone", "contiguous", "_to_copy", "lift_fresh_copy"}

HELPERS = r"""#include <cassert>
// LATCH-DISPATCH-GRAD permutation-aware embedding gradients (FP32 adds from +0.0, no FMA).
namespace latch_pgrad {
__device__ __forceinline__ float4 widen4(const uint2 x) {
  return make_float4(__uint_as_float(x.x << 16), __uint_as_float(x.x & 0xffff0000u),
                     __uint_as_float(x.y << 16), __uint_as_float(x.y & 0xffff0000u));
}
__device__ __forceinline__ float4 scaled4(const uint2 x, const float s) {
  return make_float4(__fmul_rn(__uint_as_float(x.x << 16), s), __fmul_rn(__uint_as_float(x.x & 0xffff0000u), s),
                     __fmul_rn(__uint_as_float(x.y << 16), s), __fmul_rn(__uint_as_float(x.y & 0xffff0000u), s));
}
__device__ __forceinline__ void add4(float4& a, const float4 b) {
  a.x = __fadd_rn(a.x, b.x); a.y = __fadd_rn(a.y, b.y); a.z = __fadd_rn(a.z, b.z); a.w = __fadd_rn(a.w, b.w);
}
__device__ __forceinline__ uint2 round4(const float4 a) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(a.x, a.y), hi = __floats2bfloat162_rn(a.z, a.w);
  return make_uint2(*reinterpret_cast<const unsigned*>(&lo), *reinterpret_cast<const unsigned*>(&hi));
}
}  // namespace latch_pgrad
"""


def _base(target):
    return str(target).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]


def _contiguous_vector(value):
    from .lower import _field
    shape = tuple(int(x) for x in _field(value, "shape", ()))
    strides = tuple(int(x) for x in _field(value, "strides", _field(value, "stride", ())) or ())
    if int(_field(value, "storage_offset", 0) or 0) != 0:
        return None
    step = 1
    for extent, stride in reversed(list(zip(shape, strides))):
        if extent != 1 and stride != step:
            return None
        step *= extent
    if len(strides) != len(shape):
        return None
    numel = 1
    for extent in shape:
        numel *= extent
    return numel


def _int64(value):
    from .lower import _field
    return str(_field(value, "dtype", "")).removeprefix("torch.") == "int64"


def analyze(operations, values, names):
    """{value: fact} for int64 vectors whose element arrangement the graph proves (see module docstring)."""
    from .lower import _field, _ref
    facts = {}
    for op in operations:
        target = str(_field(op, "target", _field(op, "op"))).removeprefix("torch.ops.")
        args = list(_field(op, "args", ()) or ())
        kwargs = dict(_field(op, "kwargs", {}) or {})
        outputs = tuple(_field(op, "outputs", ()) or ())
        if not outputs:
            single = _field(op, "output", _field(op, "name"))
            outputs = (single,) if single is not None else ()
        outputs = tuple(_ref(o, names) or o for o in outputs)
        if target == ORDER:
            selected = _ref(args[0], names) if args else None
            if (len(args) != 2 or len(outputs) != 3 or selected is None or not _int64(values[selected])
                    or type(args[1]) is not int):
                continue
            rows = 1
            for extent in _field(values[selected], "shape", ()):
                rows *= int(extent)
            order, _, inverse = outputs
            if any(o not in values or not _int64(values[o]) or _contiguous_vector(values[o]) != rows
                   or len(tuple(_field(values[o], "shape", ()))) != 1 for o in (order, inverse)):
                continue
            facts[order] = dict(kind="perm", extent=rows, inverse=inverse, source=ORDER)
            facts[inverse] = dict(kind="perm", extent=rows, inverse=order, source=ORDER)
            continue
        if len(outputs) != 1 or outputs[0] not in values:
            continue
        out = outputs[0]
        base = _base(target)
        source = _ref(args[0], names) if args else None
        if source is None or source not in facts or not _int64(values[out]):
            continue
        fact = facts[source]
        if base in _IDENTITY:
            # Both values are int64 (checked above); require identical flat element order.
            count = _contiguous_vector(values[out])
            if (count is not None and count == _contiguous_vector(values[source])
                    and len(tuple(_field(values[out], "shape", ()))) == 1):
                facts[out] = dict(fact)
            continue
        floor = base == "floor_divide" or (base == "div" and kwargs.get("rounding_mode") == "floor")
        if floor and fact["kind"] == "perm" and len(args) == 2:
            divisor = args[1]
            if type(divisor) is not int or not 1 <= divisor <= MAX_REPEAT or fact["extent"] % divisor:
                continue
            # Elementwise: the quotient keeps the permutation's length; its values span extent // divisor rows.
            if _contiguous_vector(values[out]) != fact["extent"] or len(tuple(_field(values[out], "shape", ()))) != 1:
                continue
            facts[out] = dict(kind="repeat", rows=fact["extent"] // divisor, repeat=divisor,
                              inverse=fact["inverse"], source=fact["source"])
    return facts


def _vector_ok(layout, width):
    """Rows of `width` contiguous elements whose starts are multiples of four elements."""
    if layout.strides[-1] != 1 or width % 4 or layout.offset % 4:
        return False
    return all(s % 4 == 0 for s, e in zip(layout.strides[:-1], layout.shape[:-1]) if e != 1)


def emit(args, kwargs, name, shape, layouts, dtypes, names, facts, *, ident, unravel):
    """Indexed-compose result for a proved index, or None (general engine). Raises nothing for refusals."""
    from .lower import _ref
    if not ENABLED or len(args) < 3:
        return None
    grad, ids = _ref(args[0], names), _ref(args[1], names)
    if grad is None or ids is None or ids not in facts:
        return None
    fact = facts[ids]
    rows = int(args[2])
    padding = int(args[3] if len(args) > 3 else kwargs.get("padding_idx", -1))
    if bool(args[4] if len(args) > 4 else kwargs.get("scale_grad_by_freq", False)):
        return None
    if set(kwargs) - {"latch_row_scale", "latch_row_widen", "latch_unique_rows", "padding_idx", "scale_grad_by_freq"}:
        return None
    lg, li = layouts[grad], layouts[ids]
    inverse = fact["inverse"]
    if inverse not in layouts or dtypes.get(inverse) != "int64" or dtypes.get(ids) != "int64":
        return None
    lq = layouts[inverse]
    n = li.numel
    if len(shape) != 2 or shape[0] != rows or tuple(lg.shape) != (*li.shape, shape[1]) or lq.numel != n:
        return None
    if fact["kind"] == "perm":
        repeat = 1
        if fact["extent"] != rows or n != rows:
            return None
    else:
        repeat = fact["repeat"]
        if fact["rows"] != rows or n != rows * repeat:
            return None
        if kwargs.get("latch_unique_rows"):
            return None
    scale = _ref(kwargs.get("latch_row_scale"), names) if kwargs.get("latch_row_scale") is not None else None
    widen = bool(kwargs.get("latch_row_widen"))
    out_dtype, grad_dtype = dtypes.get(name), dtypes.get(grad)
    if scale is not None:
        if widen or grad_dtype != "bfloat16" or dtypes.get(scale) != "float32" or layouts[scale].numel != n:
            return None
        mode = "scaled"
    elif widen:
        if grad_dtype != "bfloat16":
            return None
        mode = "widen"
    elif grad_dtype == "float32":
        mode = "fp32"
    elif grad_dtype == "bfloat16" and out_dtype == "bfloat16":
        mode = "widen"       # BF16 VJP: exact widening, FP32 row sum, one final rounding
    else:
        return None
    if out_dtype not in ("float32", "bfloat16") or (out_dtype == "bfloat16" and mode != "widen"):
        return None
    if n >= (1 << 31) - 1 or rows >= (1 << 31) - 1:
        return None
    width = shape[1]
    lo = layouts[name]
    vector = _vector_ok(lg, width) and _vector_ok(lo, width) and lo.contiguous
    w = 4 if vector else 1
    units = width // w
    G, I, Q, OUT = ident(grad), ident(ids), ident(inverse), ident(name)
    id_at = unravel(li.shape, li.strides, "q_[s_]")
    inv_at = unravel(lq.shape, lq.strides, f"(row * {repeat}LL + s_)")
    grad_row = unravel(lg.shape[:-1], lg.strides[:-1], "q_[s_]")
    scale_at = unravel(layouts[scale].shape, layouts[scale].strides, "q_[s_]") if scale else None
    if w == 4:
        if mode == "fp32":
            load = f"reinterpret_cast<const float4*>({G} + ({grad_row}))[col]"
        elif mode == "widen":
            load = f"latch_pgrad::widen4(reinterpret_cast<const uint2*>({G} + ({grad_row}))[col])"
        else:
            load = (f"latch_pgrad::scaled4(reinterpret_cast<const uint2*>({G} + ({grad_row}))[col], "
                    f"{ident(scale)}[{scale_at}])")
        zero, add = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)", "latch_pgrad::add4(acc_, v_[s_]);"
        vt = "float4"
        store = (f"reinterpret_cast<uint2*>({OUT} + row * {width}LL)[col] = latch_pgrad::round4(acc_);"
                 if out_dtype == "bfloat16" else f"reinterpret_cast<float4*>({OUT} + row * {width}LL)[col] = acc_;")
    else:
        sc = lg.strides[-1]
        element = f"{G}[({grad_row}) + col * {sc}LL]"
        if mode == "fp32":
            load = element
        elif mode == "widen":
            load = f"__bfloat162float({element})"
        else:
            load = f"__fmul_rn(__bfloat162float({element}), {ident(scale)}[{scale_at}])"
        zero, add, vt = "0.0f", "acc_ = __fadd_rn(acc_, v_[s_]);", "float"
        target = f"{OUT}[{unravel(lo.shape, lo.strides, f'(row * {width}LL + col)')}]"
        store = f"{target} = {'__float2bfloat16_rn(acc_)' if out_dtype == 'bfloat16' else 'acc_'};"
    # Insertion network over the d registers (compile-time indices): ascending occurrence ordinals.
    sort = "" if repeat == 1 else f"""
#pragma unroll
    for (int a_ = 1; a_ < {repeat}; ++a_) {{
#pragma unroll
      for (int b_ = a_; b_ > 0; --b_) {{
        const long long x_ = q_[b_ - 1], y_ = q_[b_];
        q_[b_ - 1] = x_ < y_ ? x_ : y_; q_[b_] = x_ < y_ ? y_ : x_;
      }}
    }}"""
    code = f"""for (long long u_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; u_ < {rows * units}LL;
     u_ += (long long)gridDim.x * blockDim.x) {{
  const long long row = u_ / {units}LL, col = u_ % {units}LL;
  {vt} acc_ = {zero};
  if (row != {padding}LL) {{
    long long q_[{repeat}];
#pragma unroll
    for (int s_ = 0; s_ < {repeat}; ++s_) q_[s_] = {Q}[{inv_at}];{sort}
    {vt} v_[{repeat}];
#pragma unroll
    for (int s_ = 0; s_ < {repeat}; ++s_) {{
      if (q_[s_] < 0 || q_[s_] >= {n}LL || (col == 0 && {I}[{id_at}] != row)) __trap();
      v_[s_] = {load};
    }}
#pragma unroll
    for (int s_ = 0; s_ < {repeat}; ++s_) {add}
  }}
  {store}
}}"""
    reads = [grad, ids, inverse] + ([scale] if scale else [])
    phase = dict(kind="embedding_gradient", writes=[name], reads=reads, code=code,
                 work_items=-(-(rows * units) // 256), shared_bytes=0,
                 engine="permutation_gather_v1" if repeat == 1 else "permutation_fixed_sum_v1",
                 reduction_elements=repeat,
                 arithmetic=("one occurrence per row: +0.0 + contribution" if repeat == 1 else
                             f"{repeat} occurrences per row, ascending ordinal FP32 sum from +0.0")
                             + ("; one BF16 rounding" if out_dtype == "bfloat16" else ""))
    from .indexed_reduction import requested_policy
    return {"auxiliary": [], "phases": [phase], "helpers": HELPERS,
            "decision": "permutation_inverse_gather" if repeat == 1 else "permutation_fixed_k_sum",
            "source_commit": None, "indexed_reduction_policy": requested_policy(),
            "engine": phase["engine"], "arithmetic_policy": "ordered_fp32_v1",
            "accumulation_order": "ascending occurrence ordinals from +0.0 (the ordered policy)",
            "index_proof": dict(fact, index=ids, occurrences_per_row=repeat, runtime_check="idx[q] == row"),
            "contribution": mode, "vector_columns": w == 4, "max_live_index_elements": 0,
            "indexed_selection": {"requested": requested_policy(), "selected": phase["engine"],
                                  "rule": "graph-proved permutation index (ragged_order) with a known inverse"}}
