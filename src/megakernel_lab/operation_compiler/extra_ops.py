"""C-OPS4-B operation rules shared by the ordered (lower._lower_serial) and fused (fusion.lower_fused) lowerings.

Generic ATen/explicit-random operations needed by sampling-based training programs; none is specific to a
model. Elementwise rules return one C expression per output element; indexed/heavy rules return complete
grid-stride phases through transformer_lower.emit_operation (shared by both lowerings).

Elementwise: clamp / clamp_min / clamp_max (FP32, NaN propagating like torch), isnan, integer remainder,
floor division and right shift (int64), arange.
Views: slice (step >= 1) and select, as stride/offset views of their source.
Indexed: latch_ops.philox_field (counter-grid random field), cat, slice_backward, index_select (any stored
dtype), index_copy (rows replaced, unique-index contract checked on the device).
"""
from __future__ import annotations

import math

ELEMENTWISE = {"clamp", "clamp_min", "clamp_max", "isnan", "remainder", "floor_divide", "bitwise_right_shift",
               "__rshift__", "arange"}
VIEWS = {"slice", "select"}
INDEXED = {"cat", "slice_backward", "index_select", "index_copy"}
# LATCH-MOESUP: warp-per-row 16-byte gathers for contiguous index_select rows (False: scalar loop only).
INDEX_SELECT_ROWS = True
FIELD_TARGET = "latch_ops.philox_field.default"
U64 = 1 << 64


def _int_literal(value):
    return f"static_cast<int64_t>({int(value) % U64}ULL)"


def view(base, args, src, shape):
    """(strides, offset) of a slice/select view of layout ``src``."""
    from .lower import LoweringError
    strides, offset = list(src.strides), src.offset
    dim = int(args[1]) % len(src.shape)
    if base == "select":
        index = int(args[2])
        index = index + src.shape[dim] if index < 0 else index
        if not 0 <= index < src.shape[dim]:
            raise LoweringError("select index out of range")
        offset += index * strides[dim]
        del strides[dim]
    else:
        start = args[2] if len(args) > 2 and args[2] is not None else 0
        step = int(args[4]) if len(args) > 4 else 1
        start = max(0, min(int(start) + src.shape[dim] if int(start) < 0 else int(start), src.shape[dim]))
        if step < 1:
            raise LoweringError("slice requires a positive step")
        offset += start * strides[dim]
        strides[dim] *= step
    if len(strides) != len(shape):
        raise LoweringError("slice/select rank mismatch")
    return tuple(strides), offset


def uses_expression(base, kwargs, dtype, target=""):
    return (base in ELEMENTWISE or target == FIELD_TARGET
            or (base == "div" and kwargs.get("rounding_mode") == "floor" and dtype == "int64"))


def field_inputs_admitted(args, names, input_names):
    """Elementwise random-field form: the key must be a bound input (read directly as two words)."""
    from .lower import _ref
    key = _ref(args[0], names)
    return key is not None and key in input_names


def expression(base, target, args, kwargs, name, dtypes, names, rd, index, shape, input_names=()):
    """C expression for one element of an extra elementwise operation, or None."""
    from .lower import LoweringError, _float, _ident, _ref
    if target == FIELD_TARGET:
        if not field_inputs_admitted(args, names, input_names):
            return None
        rows, cols = (int(x) for x in args[3])
        transform = args[4]
        base_term = f" + (long long)({rd(5)})" if len(args) > 5 else ""
        row = f"((long long)({index}) / {cols}LL{base_term})"
        col = f"((long long)({index}) % {cols}LL)"
        fn = {"bits": "latch_field_bits_at", "uniform": "latch_field_uniform_at", "normal": "latch_field_normal_at"}[transform]
        return f"{fn}({_ident(_ref(args[0], names))}, (uint32_t)({rd(1)}), (uint32_t)({rd(2)}), {row}, {col})"
    if base == "div" and kwargs.get("rounding_mode") == "floor" and dtypes[name] == "int64":
        base = "floor_divide"
    if base not in ELEMENTWISE:
        return None
    out = dtypes[name]
    if base in {"clamp", "clamp_min", "clamp_max"}:
        if out not in ("float32", "bfloat16"):
            raise LoweringError("clamp requires a floating tensor")
        if base == "clamp":
            low = args[1] if len(args) > 1 else kwargs.get("min")
            high = args[2] if len(args) > 2 else kwargs.get("max")
        else:
            low = args[1] if base == "clamp_min" else None
            high = args[1] if base == "clamp_max" else None
        for bound in (low, high):
            if bound is not None and (_ref(bound, names) is not None or not isinstance(bound, (int, float))):
                raise LoweringError("clamp bounds must be static scalars")
        # torch.clamp propagates NaN; both comparisons are false for NaN, which then passes through.
        if low is not None and high is not None:
            return f"latch_clamp({rd(0)}, {_float(low)}, {_float(high)})"
        if low is not None:
            return f"latch_clamp_min({rd(0)}, {_float(low)})"
        return f"latch_clamp_max({rd(0)}, {_float(high)})"
    if base == "isnan":
        if out != "bool":
            raise LoweringError("isnan returns bool")
        return f"isnan({rd(0)})"
    if base == "arange":
        if out != "int64":
            raise LoweringError("arange is admitted for int64 results")
        values = [a for a in args if isinstance(a, (int, float))]
        start, step = (0, 1)
        if target.endswith("start_step"):
            start, step = int(args[0]), int(args[2]) if len(args) > 2 else 1
        elif target.endswith("start"):
            start = int(args[0])
        if any(type(v) is not int for v in values):
            raise LoweringError("int64 arange requires integer literals")
        return f"static_cast<int64_t>({start % U64}ULL + static_cast<uint64_t>({index}) * {step % U64}ULL)"
    # integer arithmetic
    if out != "int64":
        raise LoweringError(f"{target}: admitted for int64 operands/results")

    def operand(i):
        ref = _ref(args[i], names)
        if ref is None:
            if type(args[i]) is not int:
                raise LoweringError(f"{target}: integer scalar literals required")
            return _int_literal(args[i])
        if dtypes[ref] != "int64":
            raise LoweringError(f"{target}: implicit mixed integer arithmetic is unsupported")
        return f"static_cast<int64_t>({rd(i)})"
    if base == "remainder":
        return f"latch_imod({operand(0)}, {operand(1)})"
    if base == "floor_divide" or (base == "div" and kwargs.get("rounding_mode") == "floor"):
        return f"latch_ifloordiv({operand(0)}, {operand(1)})"
    if base in {"bitwise_right_shift", "__rshift__"}:
        return f"latch_ishr({operand(0)}, {operand(1)})"
    raise LoweringError(f"unsupported extra operation {target}")


EXTRA_HELPERS = r'''
#ifndef LATCH_OPS_EXTRA_HELPERS
#define LATCH_OPS_EXTRA_HELPERS
__device__ __forceinline__ float latch_clamp(float x, float lo, float hi) {
  // PyTorch applies the lower bound, then the upper: lo > hi yields hi
  // for finite x, while a NaN input continues to propagate.
  x = x < lo ? lo : x; return x > hi ? hi : x;
}
__device__ __forceinline__ float latch_clamp_min(float x, float lo) { return x < lo ? lo : x; }
__device__ __forceinline__ float latch_clamp_max(float x, float hi) { return x > hi ? hi : x; }
// torch.int64 semantics: floor division and remainder with the divisor's sign; division by zero traps.
__device__ __forceinline__ int64_t latch_ifloordiv(int64_t a, int64_t b) {
  if (b == 0) __trap();
  int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q;
}
__device__ __forceinline__ int64_t latch_imod(int64_t a, int64_t b) {
  if (b == 0) __trap();
  int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r;
}
__device__ __forceinline__ int64_t latch_ishr(int64_t a, int64_t s) {
  return s >= 64 ? (a < 0 ? -1 : 0) : (s < 0 ? a : (a >> s));
}
#endif
'''


FIELD_HELPERS = r'''
#ifndef LATCH_OPS_PHILOX_FIELD
#define LATCH_OPS_PHILOX_FIELD
// Philox4x32-10 block (c0, c1, c2, c3) under key (k0, k1); Random123 constants (see ops/random.py).
__device__ __forceinline__ void latch_philox_block(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                                                   uint32_t k0, uint32_t k1, uint32_t* w) {
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint64_t p0 = 0xD2511F53ULL * (uint64_t)c0, p1 = 0xCD9E8D57ULL * (uint64_t)c2;
    const uint32_t n0 = (uint32_t)(p1 >> 32) ^ c1 ^ k0, n2 = (uint32_t)(p0 >> 32) ^ c3 ^ k1;
    c1 = (uint32_t)p1; c3 = (uint32_t)p0; c0 = n0; c2 = n2;
    k0 += 0x9E3779B9U; k1 += 0xBB67AE85U;
  }
  w[0] = c0; w[1] = c1; w[2] = c2; w[3] = c3;
}
// Field element (row, col): block (col >> 2, row, c2, c3), word col & 3. Transform 0 bits, 1 uniform, 2 normal.
__device__ __forceinline__ uint32_t latch_field_word(const int64_t* key, uint32_t c2, uint32_t c3,
                                                     long long row, long long col, uint32_t* w) {
  latch_philox_block((uint32_t)(col >> 2), (uint32_t)row, c2, c3, (uint32_t)key[0], (uint32_t)key[1], w);
  return w[col & 3];
}
__device__ __forceinline__ float latch_field_uniform(uint32_t word) {
  return (float)(word >> 8) * 5.9604644775390625e-08f;
}
// S100L Box-Muller: libdevice logf/cospif/sinpif, IEEE sqrtf (compiled without fast math).
__device__ __forceinline__ float latch_field_normal(const uint32_t* w, long long col) {
  const int pair = (int)(col & 3) >> 1;
  const float a = (float)((w[2 * pair] >> 8) + 1u) * 5.9604644775390625e-08f;
  const float b2 = (float)(w[2 * pair + 1] >> 8) * 5.9604644775390625e-08f * 2.0f;
  const float r = sqrtf(-2.0f * logf(a));
  return (col & 1) ? r * sinpif(b2) : r * cospif(b2);
}
// Elementwise forms (one element per call; fused with consumers by the elementwise lowerings).
__device__ __forceinline__ int64_t latch_field_bits_at(const int64_t* key, uint32_t c2, uint32_t c3, long long row, long long col) {
  uint32_t w[4];
  return (int64_t)latch_field_word(key, c2, c3, row, col, w);
}
__device__ __forceinline__ float latch_field_uniform_at(const int64_t* key, uint32_t c2, uint32_t c3, long long row, long long col) {
  uint32_t w[4];
  return latch_field_uniform(latch_field_word(key, c2, c3, row, col, w));
}
__device__ __forceinline__ float latch_field_normal_at(const int64_t* key, uint32_t c2, uint32_t c3, long long row, long long col) {
  uint32_t w[4];
  latch_field_word(key, c2, c3, row, col, w);
  return latch_field_normal(w, col);
}
#endif
'''


def emit(target, args, kwargs, name, shape, layouts, dtypes, names, read, input_names=()):
    """(kind, code, extra, helper) for an indexed extra operation, or None.

    A random field whose key is a bound input is elementwise (expression() below) and fuses with its consumers;
    otherwise it is an indexed phase here."""
    if target == FIELD_TARGET and field_inputs_admitted(args, names, input_names):
        return None
    from .lower import LoweringError, _CTYPES, _contiguous, _ident, _index, _ref, TensorLayout
    base = target.split(".")[0]
    out = _ident(name)
    numel = math.prod(shape)
    loop = (f"for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < {numel}LL; "
            f"i += (long long)gridDim.x * blockDim.x)")
    ref = lambda i: _ref(args[i], names)

    def at(value, expression_index="i", out_shape=None):
        layout = layouts[value]
        return f"{_ident(value)}[{_index(layout, out_shape or layout.shape, expression_index)}]"

    if target == FIELD_TARGET and not field_inputs_admitted(args, names, input_names):
        key, c2, c3 = ref(0), ref(1), ref(2)
        if dtypes[key] != "int64" or layouts[key].shape != (2,) or not layouts[key].contiguous:
            raise LoweringError("Philox field key must be contiguous int64[2]")
        if any(dtypes[c] != "int64" or math.prod(layouts[c].shape) != 1 for c in (c2, c3)):
            raise LoweringError("Philox field counters must be int64 scalars")
        transform = args[4]
        rows, cols = (int(x) for x in args[3])
        if tuple(shape) != (rows, cols):
            raise LoweringError("Philox field shape mismatch")
        expected = "int64" if transform == "bits" else "float32"
        if dtypes[name] != expected:
            raise LoweringError(f"Philox field {transform} result must be {expected}")
        scalar = lambda c: f"(uint32_t){_ident(c)}[0]"
        # Optional sixth argument (row tasks, ops/sampling_rows.py): int64 scalar row base of a row-sliced field.
        base = _ref(args[5], names) if len(args) > 5 else None
        if base is not None and (dtypes[base] != "int64" or math.prod(layouts[base].shape) != 1):
            raise LoweringError("Philox field row base must be an int64 scalar")
        row_base = f" + (long long){_ident(base)}[0]" if base is not None else ""
        value = {"bits": "(int64_t)word", "uniform": "latch_field_uniform(word)",
                 "normal": "latch_field_normal(w, col)"}[transform]
        code = f'''{{
const uint32_t c2_ = {scalar(c2)}, c3_ = {scalar(c3)};
{loop} {{
  const long long row = i / {cols}LL{row_base}, col = i % {cols}LL;
  uint32_t w[4];
  const uint32_t word = latch_field_word({_ident(key)}, c2_, c3_, row, col, w);
  (void)word;
  {out}[i] = {value};
}}
}}'''
        return "random_field", code, {"field_transform": transform, "field_shape": [rows, cols]}, FIELD_HELPERS

    if base == "cat" and INPLACE_CAT_PIECES and not kwargs.get("_p8c_scalar"):
        guarded = _piecewise_flat_cat(args, kwargs, name, shape, layouts, dtypes, names)
        if guarded is not None:
            return "cat", guarded, {"vector_copy": True, "inplace_pieces": True}, ""
    import os as _os
    _kinds = _os.environ.get("P8C_VCOPY_KINDS", "cat,slice_backward,select_backward").split(",")  # diagnostic switch
    if base in ("cat", "slice_backward", "select_backward") and base in _kinds and not kwargs.get("_p8c_scalar"):
        fast = _vector_copy(base, args, kwargs, name, shape, layouts, dtypes, names)
        if fast is not None:
            code, pointers = fast
            if base == "select_backward":
                from .standard_ops import emit as emit_standard
                scalar = emit_standard(target, args, kwargs, name, shape, layouts, dtypes, names, read)
                kind = "embed_gradient"
            else:
                scalar = emit(target, args, {**kwargs, "_p8c_scalar": True}, name, shape, layouts, dtypes, names, read,
                              input_names)
                kind = base
            if scalar is not None:
                aligned = " && ".join(f"((reinterpret_cast<uintptr_t>({p_}) & 15u) == 0)" for p_ in pointers)
                return kind, f"if ({aligned}) {{\n{code}\n}} else {{\n{scalar[1]}\n}}", {"vector_copy": True}, ""
    kwargs = {k: v for k, v in kwargs.items() if k != "_p8c_scalar"}
    if base not in INDEXED:
        return None
    ctype = _CTYPES[dtypes[name]]

    def coordinates(index="i"):
        return [f"(({index} / {math.prod(shape[d + 1:])}LL) % {shape[d]}LL)" for d in range(len(shape))]

    if base == "cat":
        sources = [_ref(a, names) for a in args[0]]
        dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % len(shape)
        if any(s is None or dtypes[s] != dtypes[name] for s in sources):
            raise LoweringError("cat requires same-dtype tensor operands")
        coords = coordinates()
        parts, start = [], 0
        for s in sources:
            ls = layouts[s]
            if len(ls.shape) != len(shape) or any(ls.shape[d] != shape[d] for d in range(len(shape)) if d != dim):
                raise LoweringError("cat operand shape mismatch")
            terms = [f"{(coords[d] if d != dim else f'(c_ - {start}LL)')} * {ls.strides[d]}LL"
                     for d in range(len(shape)) if ls.shape[d] != 1 or d == dim]
            parts.append((start, start + ls.shape[dim], f"{_ident(s)}[{' + '.join(terms) or '0'}]"))
            start += ls.shape[dim]
        if start != shape[dim]:
            raise LoweringError("cat extent mismatch")
        select = "".join(f"c_ < {stop}LL ? {value} : " for _, stop, value in parts[:-1]) + parts[-1][2]
        code = f"{loop} {{ const long long c_ = {coords[dim]}; {out}[i] = {select}; }}"
        return "cat", code, {}, ""

    if base == "slice_backward":
        grad = ref(0)
        sizes, dim, start, end, step = list(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5])
        dim %= len(shape)
        if tuple(sizes) != tuple(shape) or step < 1:
            raise LoweringError("slice_backward shape/step mismatch")
        start = start + shape[dim] if start < 0 else start
        start = max(0, min(start, shape[dim]))
        lg = layouts[grad]
        coords = coordinates()
        local = f"((c_ - {start}LL) / {step}LL)"
        terms = [f"{(coords[d] if d != dim else local)} * {lg.strides[d]}LL" for d in range(len(shape))]
        zero = "0.0f" if dtypes[name] == "float32" else f"({ctype})0"
        code = (f"{loop} {{ const long long c_ = {coords[dim]}; "
                f"{out}[i] = (c_ >= {start}LL && (c_ - {start}LL) % {step}LL == 0 && {local} < {lg.shape[dim]}LL) ? "
                f"{_ident(grad)}[{' + '.join(terms)}] : {zero}; }}")
        return "slice_backward", code, {}, ""

    if base == "index_select":
        source, index = ref(0), ref(2)
        dim = int(args[1]) % len(shape)
        ls, li = layouts[source], layouts[index]
        if dtypes[index] != "int64" or len(li.shape) != 1 or dtypes[source] != dtypes[name]:
            raise LoweringError("index_select requires a rank-1 int64 index and matching dtypes")
        coords = coordinates()
        terms = [f"{coords[d]} * {ls.strides[d]}LL" for d in range(len(shape)) if d != dim]
        terms.append(f"r_ * {ls.strides[dim]}LL")
        code = (f"{loop} {{ const int64_t r_ = {_ident(index)}[{coords[dim]} * {li.strides[0]}LL]; "
                f"if (r_ < 0 || r_ >= {ls.shape[dim]}LL) __trap(); {out}[i] = {_ident(source)}[{' + '.join(terms)}]; }}")
        lo = layouts.get(name)
        width = math.prod(shape[1:]) if len(shape) > 1 else 1
        unit = {"float32": 4, "bfloat16": 8}.get(dtypes[name])
        if (INDEX_SELECT_ROWS and dim == 0 and unit and len(shape) > 1 and width % unit == 0
                and ls.contiguous and not ls.offset and lo is not None and lo.contiguous and not lo.offset
                and shape[0] >= 256):
            # LATCH-MOESUP: row gather of contiguous rows, one warp per output row, 16-byte copies (data movement
            # only; identical values). The scalar loop remains the fallback for unaligned base pointers.
            vectors, ctype = width // unit, "float4"
            code = (f"if (((((unsigned long long)({_ident(source)})) | ((unsigned long long)({out}))) & 15ULL) == 0ULL) {{\n"
                    f"  const float4* src_ = reinterpret_cast<const float4*>({_ident(source)});\n"
                    f"  float4* dst_ = reinterpret_cast<float4*>({out});\n"
                    f"  const int lane_ = threadIdx.x & 31;\n"
                    f"  for (long long row_ = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; row_ < {shape[0]}LL; "
                    f"row_ += ((long long)gridDim.x * blockDim.x) >> 5) {{\n"
                    f"    const int64_t r_ = {_ident(index)}[row_ * {li.strides[0]}LL];\n"
                    f"    if (r_ < 0 || r_ >= {ls.shape[0]}LL) __trap();\n"
                    f"    for (int c_ = lane_; c_ < {vectors}; c_ += 32) dst_[row_ * {vectors}LL + c_] = src_[r_ * {vectors}LL + c_];\n"
                    f"  }}\n}} else {{\n{code}\n}}")
            return "index_select", code, {"engine": "index_select_warp_rows", "gather_table": source,
                                          "gather_index": index}, ""
        return "index_select", code, {}, ""

    if base == "index_copy":
        # out = self with rows index[j] (along dim 0) replaced by source[j]. torch leaves duplicate indices
        # unspecified; this rule traps on duplicates. One warp per output row finds its source row.
        destination, index, source = ref(0), ref(2), ref(3)
        dim = int(args[1]) % len(shape)
        ld, li, lsrc = layouts[destination], layouts[index], layouts[source]
        if dim != 0 or len(li.shape) != 1 or dtypes[index] != "int64" or dtypes[destination] != dtypes[name] \
                or dtypes[source] != dtypes[name] or lsrc.shape != (li.shape[0], *shape[1:]):
            raise LoweringError("index_copy admits dim 0, a rank-1 int64 index and matching row shapes")
        rows, width, count = shape[0], math.prod(shape[1:]), li.shape[0]
        row_d = TensorLayout(tuple(shape[1:]), ld.strides[1:], ld.root)
        row_s = TensorLayout(tuple(shape[1:]), lsrc.strides[1:], lsrc.root)
        staged = count * 8 <= 16384
        # Index list staged once per CTA in shared memory (the CTA's other rows reuse it).
        load_index = (f"  int64_t* idx_s = reinterpret_cast<int64_t*>(smem);\n"
                      f"  for (long long j = threadIdx.x; j < {count}LL; j += blockDim.x) idx_s[j] = {_ident(index)}[j * {li.strides[0]}LL];\n"
                      f"  __syncthreads();\n") if staged else ""
        index_read = "idx_s[j]" if staged else f"{_ident(index)}[j * {li.strides[0]}LL]"
        vector = (ld.contiguous and lsrc.contiguous and width % 4 == 0 and dtypes[name] == "float32"
                  and ld.offset % 4 == 0 and lsrc.offset % 4 == 0)
        if vector:
            copy = (f"  for (long long c = lane; c < {width // 4}LL; c += 32)\n"
                    f"    reinterpret_cast<float4*>({out})[row * {width // 4}LL + c] = found >= 0\n"
                    f"      ? reinterpret_cast<const float4*>({_ident(source)})[found * {width // 4}LL + c]\n"
                    f"      : reinterpret_cast<const float4*>({_ident(destination)})[row * {width // 4}LL + c];")
        else:
            copy = (f"  for (long long c = lane; c < {width}LL; c += 32)\n"
                    f"    {out}[row * {width}LL + c] = found >= 0 ? {_ident(source)}[found * {lsrc.strides[0]}LL + {_index(row_s, row_s.shape, 'c')}]\n"
                    f"                                       : {_ident(destination)}[row * {ld.strides[0]}LL + {_index(row_d, row_d.shape, 'c')}];")
        code = f'''{{
{load_index}for (long long row = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); row < {rows}LL; row += (long long)gridDim.x * 8) {{
  const int lane = threadIdx.x & 31;
  long long found = -1;
  for (long long j = lane; j < {count}LL; j += 32) {{
    const int64_t target = {index_read};
    if (target < 0 || target >= {rows}LL) __trap();
    if (target == row) {{ if (found >= 0) __trap(); found = j; }}
  }}
  for (int delta = 16; delta > 0; delta >>= 1) {{
    const long long other = __shfl_xor_sync(0xffffffffU, found, delta);
    if (other >= 0 && found >= 0 && other != found) __trap();
    found = other > found ? other : found;
  }}
{copy}
}}
__syncthreads();
}}'''
        # P9-D: the same rule when the result is committed in place over `self` (fusion_reference in-place commit):
        # only the indexed rows change, so one warp per index row copies its source row; unchanged rows are not
        # touched. Range and duplicate checks trap as above (every index is compared with every other index).
        if vector:
            scatter_copy = (f"  for (long long c = lane; c < {width // 4}LL; c += 32)\n"
                            f"    reinterpret_cast<float4*>({out})[target * {width // 4}LL + c] = "
                            f"reinterpret_cast<const float4*>({_ident(source)})[j * {width // 4}LL + c];")
        else:
            scatter_copy = (f"  for (long long c = lane; c < {width}LL; c += 32)\n"
                            f"    {out}[target * {ld.strides[0]}LL + {_index(row_d, row_d.shape, 'c')}] = "
                            f"{_ident(source)}[j * {lsrc.strides[0]}LL + {_index(row_s, row_s.shape, 'c')}];")
        scatter = f'''{{
for (long long j = (long long)blockIdx.x * 8 + (threadIdx.x >> 5); j < {count}LL; j += (long long)gridDim.x * 8) {{
  const int lane = threadIdx.x & 31;
  const int64_t target = {_ident(index)}[j * {li.strides[0]}LL];
  if (target < 0 || target >= {rows}LL) __trap();
  for (long long k = lane; k < {count}LL; k += 32)
    if (k != j && {_ident(index)}[k * {li.strides[0]}LL] == target) __trap();
{scatter_copy}
}}
}}'''
        return "index_copy", code, {"rows": rows, "replaced_rows": count, "shared_bytes": 17408 if staged else 0,
                                    "duplicate_policy": "device trap (torch leaves duplicates unspecified)",
                                    "in_place_scatter": scatter, "self_operand": destination}, ""
    return None


# LATCH-MOE-EMA-100 (b01): a flat concatenation of contiguous 1-D pieces copies piece by piece, and a piece whose
# source already is its destination slice (boundary placement nested the producer's storage inside the result,
# distributed.partition) is not copied. Same bytes in the result; data movement only. False restores the select form.
INPLACE_CAT_PIECES = __import__("os").environ.get("LATCH_OPS_INPLACE_CAT_PIECES", "1") != "0"


def _piecewise_flat_cat(args, kwargs, name, shape, layouts, dtypes, names):
    from .lower import _CTYPES, _ident, _ref
    if len(shape) != 1 or not isinstance(args[0], (list, tuple)) or len(args[0]) < 2:
        return None
    out_layout = layouts.get(name)
    if out_layout is None or not out_layout.contiguous or dtypes[name] not in _CTYPES:
        return None
    ctype = _CTYPES[dtypes[name]]
    pieces, start = [], 0
    for a in args[0]:
        s = _ref(a, names)
        if s is None or dtypes[s] != dtypes[name] or len(layouts[s].shape) != 1:
            return None
        n = layouts[s].shape[0]
        if n and layouts[s].strides[0] != 1:
            return None
        if n:
            pieces.append((s, start, n))
        start += n
    if start != shape[0]:
        return None
    out = _ident(name)
    lines = ["{ const unsigned long long t_ = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;",
             "  const unsigned long long s_ = (unsigned long long)gridDim.x * blockDim.x;"]
    for s, lo, n in pieces:
        lines.append(f"  {{ const {ctype}* p_ = reinterpret_cast<const {ctype}*>({_ident(s)}); {ctype}* q_ = {out} + {lo}LL;\n"
                     f"    if (p_ != q_) {{\n"
                     f"      if (((reinterpret_cast<uintptr_t>(p_) | reinterpret_cast<uintptr_t>(q_)) & 15u) == 0) {{\n"
                     f"        const unsigned long long nv_ = {n}ULL * sizeof({ctype}) / 16ULL, done_ = nv_ * 16ULL / sizeof({ctype});\n"
                     f"        for (unsigned long long v_ = t_; v_ < nv_; v_ += s_) reinterpret_cast<uint4*>(q_)[v_] = reinterpret_cast<const uint4*>(p_)[v_];\n"
                     f"        for (unsigned long long e_ = done_ + t_; e_ < {n}ULL; e_ += s_) q_[e_] = p_[e_];\n"
                     f"      }} else {{\n"
                     f"        for (unsigned long long e_ = t_; e_ < {n}ULL; e_ += s_) q_[e_] = p_[e_];\n"
                     f"      }}\n    }}\n  }}")
    lines.append("}")
    return "\n".join(lines)


def _vector_copy(base, args, kwargs, name, shape, layouts, dtypes, names):
    """P8-C: 16-byte vector form of cat / unit-step slice_backward / select_backward, or None.

    The result (a fresh contiguous value) is a sequence of rows: every index up to the concatenated / sliced /
    selected dimension, each row a contiguous run of ``inner`` elements (the product of the later dimensions). A
    row comes from one source row (contiguous in the later dimensions, any strides before) or is +0 padding. Each
    element is the same source element (or the same +0) as in the scalar form: a data-movement change only,
    bit-identical results. Admitted when runs, strides and offsets are 16-byte aligned and extents fit 31 bits.
    """
    from .lower import _CTYPES, _ident, _ref
    dtype = dtypes[name]
    if dtype not in ("float32", "bfloat16") or len(shape) < 1:
        return None
    width = 4 if dtype == "float32" else 8
    vtype = "float4" if dtype == "float32" else "uint4"
    numel = math.prod(shape)
    if numel % width or numel >= 2**31:
        return None
    out_layout = layouts.get(name)
    if out_layout is not None and (not out_layout.contiguous or out_layout.offset % width):
        return None
    if base == "cat":
        dim = int(args[1] if len(args) > 1 else kwargs.get("dim", 0)) % len(shape)
        sources = [_ref(a, names) for a in args[0]]
        if any(s is None or dtypes[s] != dtype for s in sources):
            return None
        pieces, start = [], 0
        for s in sources:
            ls = layouts[s]
            if len(ls.shape) != len(shape) or any(ls.shape[d] != shape[d] for d in range(len(shape)) if d != dim):
                return None
            pieces.append((s, start, start + ls.shape[dim], ls.strides[dim]))
            start += ls.shape[dim]
        if start != shape[dim]:
            return None
        squeezed = False
    elif base == "slice_backward":
        grad = _ref(args[0], names)
        sizes, dim, start, end, step = list(args[1]), int(args[2]), int(args[3]), int(args[4]), int(args[5])
        dim %= len(shape)
        if tuple(sizes) != tuple(shape) or step != 1 or grad is None or dtypes[grad] != dtype:
            return None
        start = start + shape[dim] if start < 0 else start
        start = max(0, min(start, shape[dim]))
        ls = layouts[grad]
        if len(ls.shape) != len(shape) or ls.shape[dim] == 0:
            return None
        pieces = [(grad, start, start + ls.shape[dim], ls.strides[dim])]
        squeezed = False
    else:
        grad = _ref(args[0], names)
        sizes, dim, index = list(args[1]), int(args[2]), int(args[3])
        dim %= len(shape)
        index %= shape[dim]
        if tuple(sizes) != tuple(shape) or grad is None or dtypes[grad] != dtype:
            return None
        if len(layouts[grad].shape) != len(shape) - 1:
            return None
        pieces = [(grad, index, index + 1, 0)]
        squeezed = True
    inner = math.prod(shape[dim + 1:])
    # Innermost-dimension pieces (inner == 1): a vector covers `width` consecutive positions of one piece.
    last = inner == 1 and not squeezed and shape[dim] % width == 0 and all(
        lo % width == 0 and hi % width == 0 and (dstride == 1 or hi - lo == 1) for _, lo, hi, dstride in pieces)
    if inner % width and not last:
        return None
    terms_by_piece = []
    for s, lo, hi, dstride in pieces:
        ls = layouts[s]
        full = list(ls.shape[:dim]) + [None] + list(ls.shape[dim:]) if squeezed else list(ls.shape)
        strides = list(ls.strides[:dim]) + [0] + list(ls.strides[dim:]) if squeezed else list(ls.strides)
        expect = 1
        for d in range(len(shape) - 1, dim, -1):
            if full[d] != 1 and strides[d] != expect:
                return None
            expect *= full[d]
        if ls.offset % width or any(strides[d] % width for d in range(dim if last else dim + 1) if (full[d] or 1) != 1):
            return None
        span = ls.offset + sum((e - 1) * st for e, st in zip(ls.shape, ls.strides))
        if span >= 2**31:
            return None
        terms = []
        rest = "o_"
        for d in range(dim - 1, -1, -1):
            coord = f"({rest} % {shape[d]}u)" if d > 0 else rest
            if full[d] != 1 and strides[d]:
                terms.append(f"{coord} * {strides[d]}u")
            rest = f"({rest} / {shape[d]}u)"
        if dstride and hi - lo > 1:
            terms.append(f"(c_ - {lo}u) * {dstride}u")
        offset = " + ".join(terms) or "0u"   # value pointers already include the layout offset
        terms_by_piece.append((s, lo, hi, offset))
    ctype = _CTYPES[dtype]
    zero = "make_uint4(0, 0, 0, 0)" if vtype == "uint4" else "make_float4(0.f, 0.f, 0.f, 0.f)"
    expr = zero
    for s, lo, hi, offset in reversed(terms_by_piece):
        load = f"*reinterpret_cast<const {vtype}*>(reinterpret_cast<const {ctype}*>({_ident(s)}) + ({offset}) + w_)"
        cond = f"c_ >= {lo}u && c_ < {hi}u" if (base != "cat" or lo) else f"c_ < {hi}u"
        expr = f"({cond}) ? {load} : {expr}"
    index = (f"  const unsigned f_ = v_ * {width}u, o_ = f_ / {shape[dim]}u, c_ = f_ - o_ * {shape[dim]}u, w_ = 0u;\n"
             if last else
             f"  const unsigned f_ = v_ * {width}u, r_ = f_ / {inner}u, w_ = f_ - r_ * {inner}u;\n"
             f"  const unsigned c_ = r_ % {shape[dim]}u, o_ = r_ / {shape[dim]}u;\n")
    nv = numel // width
    unroll = 4 if nv >= 4 * 32768 else 1
    if unroll == 1:
        code = (f"for (unsigned v_ = blockIdx.x * blockDim.x + threadIdx.x; v_ < {nv}u; v_ += gridDim.x * blockDim.x) {{\n"
                + index +
                f"  *reinterpret_cast<{vtype}*>(reinterpret_cast<{ctype}*>({_ident(name)}) + f_) = {expr};\n}}")
    else:
        # Four independent vectors per thread and iteration, every load before any store (more bytes in flight).
        loads, stores = [], []
        for u in range(unroll):
            def rn(text):
                import re as _re
                return _re.sub(r"\b([frwcov])_\b", lambda m: f"{m.group(1)}{u}_", text)
            body = rn(index)
            e = rn(expr)
            loads.append(f"  const unsigned v{u}_ = vb_ + {u}u * s_;\n  if (v{u}_ < {nv}u) {{\n" + body
                         + f"    x{u}_ = {e};\n  }}")
            stores.append(f"  if (v{u}_ < {nv}u) *reinterpret_cast<{vtype}*>(reinterpret_cast<{ctype}*>({_ident(name)}) + v{u}_ * {width}u) = x{u}_;")
        decl = " ".join(f"{vtype} x{u}_;" for u in range(unroll))
        code = (f"{{ const unsigned s_ = gridDim.x * blockDim.x;\n"
                f"for (unsigned vb_ = blockIdx.x * blockDim.x + threadIdx.x; vb_ < {nv}u; vb_ += {unroll}u * s_) {{\n"
                f"  {decl}\n" + "\n".join(loads) + "\n" + "\n".join(stores) + "\n}\n}")
    return code, [_ident(name)] + [_ident(s) for s, _, _, _ in terms_by_piece]
