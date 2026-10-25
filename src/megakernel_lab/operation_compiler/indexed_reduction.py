"""Deterministic occurrence-indexed FP32 reductions for ordinary embedding AD.

Unlike the historical per-vocabulary-row occurrence scan, construction never
visits the Cartesian product of rows and positions. Integer stable merge passes
order input ordinals by (index, ordinal); row ranges then select only matching
gradient rows. The arithmetic contract follows the old ascending-position rule
(775505d79b1d7debc9b2865e092e30405eee8455), but the sorting implementation is new.

Work is O(N log^2 N + V log N + V*D + N*D). Scratch live ranges require at most
two N-element integer permutations and V pairs of bounds; retain_all diagnostics
deliberately retain each merge pass, O(N log N). All intermediates are SSA, so
dependency scheduling and arena reuse remain ordinary compiler mechanisms.
"""
from __future__ import annotations

import contextvars
from .explain.collect import note as _explain_note  # LATCH-EXPLAIN: no-op unless a collector is active
from contextlib import contextmanager
import math


POLICIES = ("baseline", "batched_v1", "grouped_v1", "batched8_v1", "segmented_v1", "segmented_tile_v1",
            # LATCH-STEP-04: the frozen base P-OPS-D emitter (transformer_lower), selectable under compose,
            # and the single-phase shared-histogram engine (indexed_rowscan); rowscan_dbgK are timing ablations
            # (stop after internal step K; numerically invalid by construction).
            "frozen_base", "rowscan_v1", "rowscan_dbg1", "rowscan_dbg2", "rowscan_dbg3", "rowscan_dbg4",
            # single-phase row-ownership engine (indexed_owner); owner_dbgK: timing ablations (invalid numerics)
            "owner_v1", "owner_dbg0", "owner_dbg1", "owner_dbg2", "owner_dbg3", "owner_dbg4", "owner_dbg5",
            # global float-atomic accumulation (indexed_atomic): unordered FP32, not bitwise repeatable
            "atomic_v1")
# Implementations whose declared arithmetic is a fixed FP32 reassociation, not ascending order.
REASSOCIATED = frozenset({"segmented_v1", "segmented_tile_v1", "rowscan_v1", "owner_v1", "atomic_v1"})
# Implementations whose FP32 addition order depends on hardware arrival order (not bitwise repeatable).
UNORDERED = frozenset({"atomic_v1"})
_POLICY = contextvars.ContextVar("latch_indexed_reduction_policy", default="baseline")
# LATCH-STEP-04: whether a caller selected a policy explicitly (None: automatic selection).
_REQUESTED = contextvars.ContextVar("latch_indexed_reduction_requested", default=None)


def requested_policy():
    return _REQUESTED.get()


# LATCH-MOE4-160: batched ordered reductions over contiguous FP32 rows use one warp per output
# row (memory-level parallelism only; the per-coordinate addition order is unchanged).
WARP_ROWS = True


def _warp_rows(out, grad, ordered, bounds, rows, width, padding, batch):
    """Warp-per-row ordered reduction body (grid-stride over rows, warp-uniform control flow)."""
    def body(vector):
        T = "float4" if vector else "float"
        units = width // 4 if vector else width
        J = -(-units // 32)
        zero = "make_float4(0.0f, 0.0f, 0.0f, 0.0f)" if vector else "0.0f"
        add = ("a_[j].x = __fadd_rn(a_[j].x, v_[u][j].x); a_[j].y = __fadd_rn(a_[j].y, v_[u][j].y); "
               "a_[j].z = __fadd_rn(a_[j].z, v_[u][j].z); a_[j].w = __fadd_rn(a_[j].w, v_[u][j].w);"
               if vector else "a_[j] = __fadd_rn(a_[j], v_[u][j]);")
        return f"""{{
    const {T}* g_ = reinterpret_cast<const {T}*>({grad});
    {T}* o_ = reinterpret_cast<{T}*>({out});
    for (long long row = wid_; row < {rows}LL; row += nw_) {{
      {T} a_[{J}];
#pragma unroll
      for (int j = 0; j < {J}; ++j) a_[j] = {zero};
      if (row != {padding}LL) {{
        const long long b_ = {bounds}[2 * row], e_ = {bounds}[2 * row + 1];
#pragma unroll 1
        for (long long q_ = b_; q_ < e_; q_ += 32) {{
          const long long mine_ = q_ + lane_ < e_ ? {ordered}[q_ + lane_] : 0LL;
          const int cnt_ = (int)(e_ - q_ < 32 ? e_ - q_ : 32);
#pragma unroll 1
          for (int u0_ = 0; u0_ < cnt_; u0_ += {batch}) {{
            {T} v_[{batch}][{J}];
#pragma unroll
            for (int u = 0; u < {batch}; ++u) {{
              const long long o_ = __shfl_sync(0xffffffffu, mine_, (u0_ + u) & 31);
              const bool live_ = u0_ + u < cnt_;
#pragma unroll
              for (int j = 0; j < {J}; ++j) {{
                const int c_ = lane_ + 32 * j;
                v_[u][j] = (live_ && c_ < {units}) ? g_[o_ * {units}LL + c_] : {zero};
              }}
            }}
#pragma unroll
            for (int u = 0; u < {batch}; ++u)
              if (u0_ + u < cnt_) {{
#pragma unroll
                for (int j = 0; j < {J}; ++j) {{ {add} }}
              }}
          }}
        }}
      }}
#pragma unroll
      for (int j = 0; j < {J}; ++j) {{
        const int c_ = lane_ + 32 * j;
        if (c_ < {units}) o_[row * {units}LL + c_] = a_[j];
      }}
    }}
  }}"""
    head = f"""{{
  const int lane_ = threadIdx.x & 31;
  const long long wid_ = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const long long nw_ = ((long long)gridDim.x * blockDim.x) >> 5;
"""
    if width % 4 == 0:
        aligned = f"(((unsigned long long)({grad}) | (unsigned long long)({out})) & 15ULL) == 0ULL"
        return head + f"  if ({aligned}) {body(True)} else {body(False)}\n}}"
    return head + "  " + body(False) + "\n}"


@contextmanager
def policy(name=None):
    """Select an exact-arithmetic implementation when generating the program.

    The baseline remains the frozen donor implementation. Candidates change
    memory-level parallelism/scheduling only, never the FP32 addition order.
    """
    name = _POLICY.get() if name is None else name
    if name not in POLICIES:
        raise ValueError("unknown indexed reduction policy: " + repr(name))
    token, requested = _POLICY.set(name), _REQUESTED.set(name)
    try:
        yield name
    finally:
        _REQUESTED.reset(requested)
        _POLICY.reset(token)


def ordered_embedding_backward(args, kwargs, name, shape, layouts, dtypes, names, *, cache=None, norm=None,
                               flags=None, read=None, arithmetic=None):
    from .lower import LoweringError, _ident, _index, _ref

    grad, ids = _ref(args[0], names), _ref(args[1], names)
    lg, li = layouts[grad], layouts[ids]
    rows = int(args[2])
    padding = int(args[3] if len(args) > 3 else kwargs.get("padding_idx", -1))
    # LATCH-MOESUP (ops/scaled_merge.py): a per-occurrence FP32 factor applied to BF16 rows inside the segmented
    # engine; every other engine refuses it.
    scale = _ref(kwargs.get("latch_row_scale"), names) if kwargs.get("latch_row_scale") is not None else None
    # LATCH-MOESUP-ROW (ops/scaled_merge.py): BF16 rows widened exactly at load by the segmented engine.
    widen = bool(kwargs.get("latch_row_widen"))
    if widen:
        if (scale is not None or dtypes[grad] != "bfloat16"
                or _POLICY.get() not in ("segmented_v1", "segmented_tile_v1")):
            raise LoweringError("widened embedding backward requires BF16 rows and the segmented engine")
    elif scale is not None:
        if (dtypes[grad] != "bfloat16" or dtypes[scale] != "float32" or layouts[scale].numel != layouts[ids].numel
                or _POLICY.get() not in ("segmented_v1", "segmented_tile_v1")):
            raise LoweringError("row-scaled embedding backward requires BF16 rows, FP32 factors and the segmented engine")
    elif dtypes[grad] != "float32":
        raise LoweringError("ordered embedding backward requires FP32 gradients and int64 indices")
    if dtypes[name] != "float32" or dtypes[ids] != "int64":
        raise LoweringError("ordered embedding backward requires FP32 gradients and int64 indices")
    if bool(args[4] if len(args) > 4 else kwargs.get("scale_grad_by_freq", False)):
        raise LoweringError("frequency-scaled embedding gradients are not admitted")
    if len(shape) != 2 or shape[0] != rows or lg.shape != (*li.shape, shape[1]):
        raise LoweringError("ordered embedding backward shape mismatch")
    n, width = li.numel, shape[1]
    if n < 1 or rows < 1:
        raise LoweringError("ordered embedding backward requires nonempty index and row dimensions")
    if n > (1 << 31) - 1 or rows > (1 << 31) - 1:
        raise LoweringError("ordered embedding backward index dimensions exceed admitted signed-32-bit bounds")
    auxiliary, phases = [], []
    implementation = _POLICY.get()
    # LATCH-MOE-EMA-100 (row_scatter_mm): a scatter whose rows must each receive at most one occurrence takes the
    # exact ordered baseline and traps on a repeated index (checked on every execution from the row bounds).
    unique_rows = bool(kwargs.get("latch_unique_rows"))
    if unique_rows:
        if scale is not None or widen or arithmetic is not None or norm is not None or flags is not None:
            raise LoweringError("unique-row scatter admits plain FP32 rows only")
        implementation = "baseline"
    if arithmetic is not None:
        from .embedding_backward_contract import CONTRACT
        if (arithmetic.get("contract") != CONTRACT or arithmetic.get("output") != name
                or arithmetic.get("partial_rows") != 10 or n <= 3072
                or arithmetic.get("indices_count") != n or arithmetic.get("rows") != rows
                or arithmetic.get("width") != width or norm is not None or flags is not None):
            raise LoweringError("embedding ten-row arithmetic differs from the admitted operation")
        # Reuse stable occurrence construction; no automatic atomic fallback.
        implementation = "baseline"
    if implementation == "atomic_v1" and not flags:
        from .indexed_atomic import HELPERS, emit, supported
        if supported(n, rows, width)[0]:
            emitted, decision = emit(name, grad, ids, lg, li, rows, padding, width, _ident, norm=norm, read=read)
            auxiliary = decision.pop("auxiliary", [])
            return {"auxiliary": auxiliary, "phases": emitted, "helpers": HELPERS,
                    "decision": "global_atomic_accumulation", "source_commit": None,
                    "indexed_reduction_policy": implementation, **decision, "max_live_index_elements": 0}
        implementation = "owner_v1"
    if implementation == "owner_v1" or implementation.startswith("owner_dbg"):
        from .indexed_owner import HELPERS, emit, supported
        admitted, reason = supported(n, rows, width)
        if not admitted:
            _explain_note("indexed_fallback", output=name, engine=_POLICY.get(), reason=reason, fallback="next admitted engine or baseline")
        if admitted:
            stop = int(implementation[-1]) if implementation.startswith("owner_dbg") else 99
            emitted, decision = emit(name, grad, ids, lg, li, rows, padding, width, _ident, norm=norm, flags=flags,
                                     debug_stop=stop, read=read)
            return {"auxiliary": auxiliary, "phases": emitted, "helpers": HELPERS,
                    "decision": "owner_row_units", "source_commit": None,
                    "indexed_reduction_policy": implementation, **decision, "max_live_index_elements": 0}
        implementation = "segmented_v1"   # outside the owner engine's index/shared-memory bounds
    if implementation == "rowscan_v1" or implementation.startswith("rowscan_dbg"):
        from .indexed_rowscan import HELPERS, emit, supported
        admitted, reason = supported(n, rows, width)
        if not admitted:
            _explain_note("indexed_fallback", output=name, engine=_POLICY.get(), reason=reason, fallback="next admitted engine or baseline")
        if admitted:
            stop = int(implementation[-1]) if implementation.startswith("rowscan_dbg") else 99
            emitted, decision = emit(name, grad, ids, lg, li, rows, padding, width, _ident, debug_stop=stop)
            return {"auxiliary": auxiliary, "phases": emitted, "helpers": HELPERS,
                    "decision": "rowscan_occurrence_units", "source_commit": None,
                    "indexed_reduction_policy": implementation, **decision,
                    "max_live_index_elements": 0}
        implementation = "segmented_v1"   # shape outside the shared-histogram engine: load-balanced fallback
    if implementation in REASSOCIATED:
        from .indexed_segmented import HELPERS, emit, supported
        admitted, reason = supported(n, rows, width)
        if not admitted:
            _explain_note("indexed_fallback", output=name, engine=_POLICY.get(), reason=reason, fallback="next admitted engine or baseline")
        if admitted:
            def allocate_typed(suffix, extent, dtype):
                value = name + "__" + suffix
                if value in names or value in layouts:
                    raise LoweringError("generated indexed scratch name collides with graph value: " + value)
                auxiliary.append((value, extent, dtype))
                return value
            emitted, decision = emit(name, grad, ids, lg, li, rows, padding, width, _ident, allocate_typed,
                                     cache if cache is not None else {},
                                     sorter="tile" if implementation == "segmented_tile_v1" else "radix",
                                     scale=(scale, layouts[scale]) if scale is not None else None, widen=widen)
            return {"auxiliary": auxiliary, "phases": emitted, "helpers": HELPERS,
                    "decision": "segmented_occurrence_pieces", "reused_occurrences": decision["reused_occurrences"],
                    "source_commit": None, "indexed_reduction_policy": implementation, **decision,
                    "max_live_index_elements": 2 * n + 2 * rows}

    def allocate(suffix, extent):
        value = name + "__" + suffix
        if value in names or value in layouts:
            raise LoweringError("generated ordered-index scratch name collides with graph value: " + value)
        auxiliary.append((value, extent, "int64"))
        return value

    def at_id(ordinal):
        return f"{_ident(ids)}[{_index(li, li.shape, ordinal)}]"

    def loop(extent, body):
        return (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {extent}LL; "
                f"i += (long long)gridDim.x * blockDim.x) {{\n{body}\n}}")

    def phase(kind, output, code, reads, **extra):
        phases.append({"kind": kind, "writes": [output], "reads": reads, "code": code, **extra})

    # Reuse integer occurrence order for multiple tables indexed by the same
    # graph value. Padding affects only the final floating reduction.
    key = (ids, rows)
    reused = cache is not None and key in cache
    from . import ordered_index_radix
    radix = ordered_index_radix.supported(n, rows)
    linear_bounds = radix and ordered_index_radix.LINEAR_BOUNDS
    radix_scratch = 0
    if reused:
        ordered, bounds = cache[key]
    else:
        if radix:
            if linear_bounds:
                bounds = allocate("row_bounds", (rows, 2))
            radix_phases, ordered, _, radix_scratch = ordered_index_radix.emit(
                name, ids, n, rows, _index(li, li.shape, "o"), _ident, allocate,
                bounds=bounds if linear_bounds else None)
            phases.extend(radix_phases)
            span, pass_index = n, 1
        else:
            ordered = allocate("ordinal_0", (n,))
            phase("index_ordinals", ordered, loop(n, f"""  const int64_t id = {at_id('i')};
  assert(id >= 0 && id < {rows}LL);
  {_ident(ordered)}[i] = i;"""), [ids])
            span, pass_index = 1, 1
        while span < n:
            merged = allocate(f"ordinal_{pass_index}", (n,))
            # Every input element computes its unique rank in the union of two
            # already-sorted runs. The (id, original ordinal) key is unique, so
            # both halves use lower_bound without a duplicate-write race.
            code = loop(n, f"""  const long long begin = (i / {2 * span}LL) * {2 * span}LL;
  const long long middle = begin + {span}LL < {n}LL ? begin + {span}LL : {n}LL;
  const long long end = begin + {2 * span}LL < {n}LL ? begin + {2 * span}LL : {n}LL;
  const bool left = i < middle;
  const long long own_begin = left ? begin : middle;
  const long long other_begin = left ? middle : begin;
  const long long other_end = left ? end : middle;
  const long long ordinal = {_ident(ordered)}[i];
  const int64_t id = {at_id('ordinal')};
  long long lo = other_begin, hi = other_end;
  while (lo < hi) {{
    const long long m = lo + (hi - lo) / 2;
    const long long other = {_ident(ordered)}[m];
    const int64_t other_id = {at_id('other')};
    if (other_id < id || (other_id == id && other < ordinal)) lo = m + 1;
    else hi = m;
  }}
  const long long destination = begin + (i - own_begin) + (lo - other_begin);
  {_ident(merged)}[destination] = ordinal;""")
            phase("index_stable_merge", merged, code, [ids, ordered], merge_span=span)
            ordered, span, pass_index = merged, span * 2, pass_index + 1
        if not linear_bounds:
            bounds = allocate("row_bounds", (rows, 2))
            code = loop(rows, f"""  long long lo = 0, hi = {n}LL;
  while (lo < hi) {{
    const long long m = lo + (hi - lo) / 2, ordinal = {_ident(ordered)}[m];
    if ({at_id('ordinal')} < i) lo = m + 1; else hi = m;
  }}
  {_ident(bounds)}[2 * i] = lo;
  hi = {n}LL;
  while (lo < hi) {{
    const long long m = lo + (hi - lo) / 2, ordinal = {_ident(ordered)}[m];
    if ({at_id('ordinal')} <= i) lo = m + 1; else hi = m;
  }}
  {_ident(bounds)}[2 * i + 1] = lo;""")
            phase("index_row_bounds", bounds, code, [ids, ordered])
        if cache is not None:
            cache[key] = (ordered, bounds)
    grad_index = _index(lg, lg.shape, f"(ordinal * {width}LL + col)")
    code = loop(rows * width, f"""  const long long row = i / {width}LL, col = i % {width}LL;
  float acc = 0.0f;
  if (row != {padding}LL) {{
    const long long end = {_ident(bounds)}[2 * row + 1];
    for (long long p = {_ident(bounds)}[2 * row]; p < end; ++p) {{
      const long long ordinal = {_ident(ordered)}[p];
      acc += {_ident(grad)}[{grad_index}];
    }}
  }}
  {_ident(name)}[i] = acc;""")
    extra = {}
    if scale is not None:
        raise LoweringError("row-scaled embedding backward: the segmented engine did not admit this shape")
    if implementation in REASSOCIATED:
        implementation = "baseline"  # unsupported shape: exact ordered fallback, recorded below
        extra = {"indexed_fallback": "baseline (shape not admitted by the requested engine)"}
    if implementation != "baseline":
        batch_elements = 8 if implementation == "batched8_v1" else 32
        # Reuse the P-OPS-D/P6-E batched-load implementation retained in
        # transformer_lower.ordered_embedding_backward, without its O(V*N)
        # occurrence construction. Loading ahead is independent; adding stays
        # in ascending ordinal order and starts at the same positive zero.
        batches = f"""    long long p = {_ident(bounds)}[2 * row];
    for (; p + {batch_elements} <= end; p += {batch_elements}) {{
      float values[{batch_elements}];
#pragma unroll
      for (int u = 0; u < {batch_elements}; ++u) {{
        const long long ordinal = {_ident(ordered)}[p + u];
        values[u] = {_ident(grad)}[{grad_index}];
      }}
#pragma unroll
      for (int u = 0; u < {batch_elements}; ++u) acc += values[u];
    }}
    for (; p < end; ++p) {{
      const long long ordinal = {_ident(ordered)}[p];
      acc += {_ident(grad)}[{grad_index}];
    }}"""
        body = f"""  float acc = 0.0f;
  if (row != {padding}LL) {{
    const long long end = {_ident(bounds)}[2 * row + 1];
{batches}
  }}
  {_ident(name)}[row * {width}LL + col] = acc;"""
        grouped = implementation == "grouped_v1" and width >= 32 and lg.strides[-1] == 1
        if grouped:
            # A warp owns one 32-column group, including odd-width tails.
            # Rotate row ownership by column tile so a hot row's independent
            # columns occupy different resident warps, even when V is a power
            # of two. Multiplication is a bijection modulo V; it distributes
            # nearby IDs without consulting values or model names.
            multiplier = 17
            while math.gcd(multiplier, rows) != 1:
                multiplier += 2
            tiles = (width + 31) // 32
            code = f"""for (long long job = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
     job < {rows * tiles}LL; job += (long long)gridDim.x * (blockDim.x / 32)) {{
  const long long tile = job / {rows}LL;
  const long long row = ((job % {rows}LL) * {multiplier}LL + tile * 37LL) % {rows}LL;
  const long long col = tile * 32 + (threadIdx.x & 31);
  if (col < {width}LL) {{
{body}
  }}
}}"""
        else:
            code = loop(rows * width, f"  const long long row = i / {width}LL, col = i % {width}LL;\n" + body)
        # `engine` participates in generic scheduler dispatch (notably P6-B
        # reduction sharing). A descriptive field must not change those rules.
        extra = {"indexed_engine": "ordered_warp_tiles_batched32" if grouped else f"ordered_batched{batch_elements}",
                 "indexed_reduction_policy": implementation,
                 "prefetch_elements": batch_elements,
                 "initialization": "each output coordinate written once from positive zero; inactive and padding rows included",
                 "parallelism": "independent row/column coordinates only; serial ascending additions per coordinate",
                 "adapted_from": "transformer_lower.ordered_embedding_backward P-OPS-D/P6-E batched loads"}
        if WARP_ROWS and not grouped and lg.contiguous and width >= 32:
            # LATCH-MOE4-160: one warp owns an output row. Lanes fetch up to 32 occurrence
            # ordinals at once and shuffle them; every lane loads its column slice of four
            # occurrences before adding them. Each output coordinate still starts at positive
            # zero and adds its occurrences one at a time in ascending position.
            code = _warp_rows(_ident(name), _ident(grad), _ident(ordered), _ident(bounds), rows, width,
                              padding, 4)
            extra.update(indexed_engine="ordered_warp_rows4", prefetch_elements=4,
                         parallelism=("one warp per output row; lanes own column slices; "
                                      "serial ascending additions per coordinate"),
                         vector_columns="float4 when width % 4 == 0 and pointers are 16-byte aligned (runtime check)")
    if arithmetic is not None:
        # FP32 register partials have the same per-add rounding as the source
        # kernel's stored partials; no additional floating scratch is required.
        code = loop(rows * width, f"""  const long long row = i / {width}LL, col = i % {width}LL;
  float acc = 0.0f;
  if (row != {padding}LL) {{
    const long long end = {_ident(bounds)}[2 * row + 1];
    for (long long begin = {_ident(bounds)}[2 * row]; begin < end; begin += 10) {{
      float partial = 0.0f;
      const long long stop = begin + 10 < end ? begin + 10 : end;
      for (long long p = begin; p < stop; ++p) {{
        const long long ordinal = {_ident(ordered)}[p];
        partial = __fadd_rn(partial, {_ident(grad)}[{grad_index}]);
      }}
      acc = __fadd_rn(acc, partial);
    }}
  }}
  {_ident(name)}[i] = acc;""")
        extra = {"indexed_engine": "ordered_embedding_segment10", "indexed_arithmetic": arithmetic["contract"],
                 "embedding_backward_contract": arithmetic, "partial_rows": 10,
                 "partial_dtype": "float32", "final_dtype": "float32",
                 "initialization": "positive FP32 zero per partial and per output coordinate",
                 "parallelism": "independent output coordinates; ascending stable ten-occurrence partials"}
    if unique_rows:
        code = (loop(rows, f"  if ({_ident(bounds)}[2 * i + 1] - {_ident(bounds)}[2 * i] > 1LL) __trap();")
                + "\n" + code)
        extra = {**extra, "unique_rows": "trap when a row receives more than one occurrence"}
    phase("embedding_gradient", name, code, [grad, ordered, bounds],
          reduction_elements="ascending occurrence ordinals", accumulation_dtype="float32", **extra)
    return {"auxiliary": auxiliary, "phases": phases, "helpers": "#include <cassert>\n",
            "decision": "stable_radix_occurrence_list" if radix else "stable_merge_occurrence_list", "reused_occurrences": reused,
            "row_bounds": "sorted_neighbor_scatter" if linear_bounds else "binary_search",
            "source_commit": None, "contract_source_commit": "775505d79b1d7debc9b2865e092e30405eee8455",
            "accumulation_order": (arithmetic["accumulation"] if arithmetic is not None
                                   else "ascending input position, starting from FP32 zero"),
            "work_bound": ("N ceil(log2(V)/8) + V + V*D + N*D" if linear_bounds else
                           "N ceil(log2(V)/8) + V log N + V*D + N*D" if radix
                           else "N log^2 N + V log N + V*D + N*D"),
            "max_live_index_elements": 2 * n + 2 * rows + radix_scratch}
