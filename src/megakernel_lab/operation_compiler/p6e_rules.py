"""P6-E compiler rules for complete state-space training updates (research compiler only; all opt-in).

Rules are selected from operations, shapes, layouts and dependencies only; nothing inspects model or value names.

* ``p6e_trap``: in programs with warpgroup-MMA products, generated index checks abort through ``__trap()`` instead of
  ``assert()``. assert's ``__assertfail`` system call is a function call inside the kernel, and ptxas then serializes
  every wgmma of the function (C7510, "wgmma pipeline crossing function boundary at a function call"), which the
  P-OPS5-E h30 build logs. Same abort-on-violation semantics, no message. Lineage: P-OPS5-C ``p5c_matrix.TRAP_ASSERT``
  (4e2b74545), re-applied here to the P-OPS4-B TMA/wgmma programs.
"""
from __future__ import annotations

TRAP_ASSERT = '''// P6-E (from P-OPS5-C): generated index bounds checks use assert(); its __assertfail system call makes ptxas
// serialize every wgmma of the function (C7510). In programs with warpgroup-MMA products a failed check aborts the
// launch through __trap() instead (same abort-on-violation, no message).
#ifdef assert
#undef assert
#endif
#define assert(c) do { if (!(c)) __trap(); } while (0)
'''


# ------------------------------------------------------------------------------------------------ slice assembly
# ``p6e_assembly``: an elementwise loop whose only member is a sum (add chains with alpha 1, every intermediate of the
# member's dtype) of slice_backward terms along the trailing axis with step 1 and pairwise-disjoint column ranges,
# each term a stored contiguous source. (The gradient of a value consumed through several column slices, e.g. a fused
# projection's [rows, 3E+2N] gradient assembled from a scan's separate outputs.) Every output element is covered by at
# most one term, so the reference chain evaluates, per element, sums of exact zeros with at most one source value x:
# each partial sum is exactly +0 or x, except that IEEE x + (+0) turns -0 into +0 and NaN into a quiet NaN, and each
# rounding of a representable value is the identity. The emitted copy therefore stores round(float(x) + 0.0f) inside a
# segment and +0 outside, which is bitwise the reference result for every input, including signed zeros and NaN. It
# moves 16-byte chunks along rows (one segment lookup per chunk) instead of evaluating the nested conditional chain
# and two integer divisions per element. Dispatch: operation kinds, axes, extents, layouts and dtypes only.
_BYTES = {"float32": 4, "bfloat16": 2}


def _norm(v, size):
    v = int(v)
    if v < 0:
        v += size
    return max(0, min(size, v))


def assembly_pattern(name, *, nodes, ew_args, inline, layouts, dtypes, declared, ref):
    node = nodes.get(name)
    if node is None or node.kind != "elementwise" or node.dtype not in _BYTES or not node.shape:
        return None
    shape = tuple(int(x) for x in node.shape)
    width = shape[-1]
    numel = 1
    for d in shape:
        numel *= d
    dtype = node.dtype
    segments = []

    def walk(r, top):
        n = nodes.get(r)
        if n is None or n.kind != "elementwise" or n.dtype != dtype or r not in ew_args:
            return False
        if tuple(int(x) for x in n.shape) != shape:
            return False
        if not top and (r not in inline or layouts[r].root != r or not layouts[r].contiguous or layouts[r].offset):
            return False
        base, args, kwargs = ew_args[r]
        if base == "add":
            alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
            if alpha != 1 or len(args) < 2:
                return False
            a, b = ref(args[0]), ref(args[1])
            if a is None or b is None:
                return False
            return walk(a, False) and walk(b, False)
        if base == "slice_backward":
            if len(args) < 5:
                return False
            grad = ref(args[0])
            sizes = tuple(int(x) for x in args[1])
            dim = int(args[2]) % len(shape)
            step = int(args[5]) if len(args) > 5 else 1
            if grad is None or sizes != shape or dim != len(shape) - 1 or step != 1:
                return False
            start, end = _norm(args[3], width), _norm(args[4], width)
            lay = layouts.get(grad)
            if (lay is None or grad not in declared or lay.root in inline or dtypes.get(grad) != dtype
                    or not lay.contiguous or tuple(int(x) for x in lay.shape) != shape[:-1] + (end - start,)
                    or end <= start):
                return False
            segments.append((start, end, grad, int(lay.offset)))
            return True
        return False

    if not walk(name, True) or len(segments) < 2:
        return None
    segments.sort()
    if any(segments[k][1] > segments[k + 1][0] for k in range(len(segments) - 1)):
        return None
    vec = 16 // _BYTES[dtype]
    if width % vec or any(s % vec or e % vec or (off * _BYTES[dtype]) % 16 for s, e, _, off in segments):
        return None
    if numel // vec >= 2 ** 31:
        return None
    return {"rows": numel // width, "width": width, "dtype": dtype, "vector": vec,
            "segments": [(s, e, g) for s, e, g, _ in segments]}


def assembly_code(pattern, out_ident, *, exclude, ident):
    rows, width, vec = pattern["rows"], pattern["width"], pattern["vector"]
    per_row = width // vec
    chunks = rows * per_row
    bf16 = pattern["dtype"] == "bfloat16"
    canon = "latch_p6e_canon_bf16" if bf16 else "latch_p6e_canon_f32"
    lookups = []
    for k, (s, e, g) in enumerate(pattern["segments"]):
        cond = f"col >= {s}u && col < {e}u"
        load = (f"{canon}(*reinterpret_cast<const uint4*>({ident(g)} + (size_t)row * {e - s}u + (col - {s}u)))")
        lookups.append(("if" if k == 0 else "else if") + f" ({cond}) x[w] = {load};")
    if exclude:
        head = (f"  const unsigned skip = 2u * {exclude}u <= gridDim.x ? {exclude}u : 0u;\n"
                f"  const unsigned vb = (blockIdx.x + gridDim.x - skip) % gridDim.x;\n"
                f"  const unsigned gs = (gridDim.x - skip) * blockDim.x;\n"
                f"  if (vb < gridDim.x - skip)\n")
        start = "vb"
    else:
        head = "  const unsigned gs = gridDim.x * blockDim.x;\n"
        start = "blockIdx.x"
    segs = ", ".join(f"[{s},{e})" for s, e, _ in pattern["segments"])
    return (f"{{ // P6-E disjoint slice assembly: {rows} rows x {width}, segments {segs}, 16-byte chunks: {out_ident}\n"
            + head +
            f"  for (unsigned base = {start} * blockDim.x + threadIdx.x; base < {chunks}u; base += 4u * gs) {{\n"
            f"    uint4 x[4];\n"
            f"#pragma unroll\n"
            f"    for (int w = 0; w < 4; ++w) {{\n"
            f"      const unsigned c = base + (unsigned)w * gs;\n"
            f"      x[w] = make_uint4(0u, 0u, 0u, 0u);\n"
            f"      if (c < {chunks}u) {{\n"
            f"        const unsigned row = c / {per_row}u, col = (c % {per_row}u) * {vec}u;\n"
            f"        " + "\n        ".join(lookups) + "\n"
            f"      }}\n"
            f"    }}\n"
            f"#pragma unroll\n"
            f"    for (int w = 0; w < 4; ++w) {{\n"
            f"      const unsigned c = base + (unsigned)w * gs;\n"
            f"      if (c < {chunks}u) *reinterpret_cast<uint4*>({out_ident} + (size_t)c * {vec}u) = x[w];\n"
            f"    }}\n"
            f"  }}\n}}")


ASSEMBLY_HELPERS = r'''
#ifndef LATCH_P6E_CANON
#define LATCH_P6E_CANON
// round(float(x) + 0.0f) per element: exact for representable x, turns -0 into +0 and NaN into the conversion's NaN,
// as the reference chain of exact-zero additions does.
__device__ __forceinline__ uint32_t latch_p6e_canon_pair(uint32_t w) {
  const float lo = __uint_as_float(w << 16) + 0.0f, hi = __uint_as_float(w & 0xffff0000u) + 0.0f;
  const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint4 latch_p6e_canon_bf16(uint4 v) {
  return make_uint4(latch_p6e_canon_pair(v.x), latch_p6e_canon_pair(v.y), latch_p6e_canon_pair(v.z),
                    latch_p6e_canon_pair(v.w));
}
__device__ __forceinline__ uint4 latch_p6e_canon_f32(uint4 v) {
  return make_uint4(__float_as_uint(__uint_as_float(v.x) + 0.0f), __float_as_uint(__uint_as_float(v.y) + 0.0f),
                    __float_as_uint(__uint_as_float(v.z) + 0.0f), __float_as_uint(__uint_as_float(v.w) + 0.0f));
}
#endif
'''


# ------------------------------------------------------------------------------------------------ loop invariants
# ``p6e_licm``: in a fused elementwise loop, a parenthesized subexpression (or call) of a member's per-element
# expression that reads only constants, graph scalars (v_x[0u]) and outlined pure functions of them is evaluated once
# per thread before the loop and referenced by name (``const auto``: its C++ type, hence every rounding, is kept). The
# loop writes only its own members and reads every scalar from an earlier phase, so the value is the one each element
# computed; results are bitwise unchanged. (AdamW bias corrections 1 - beta^step, their square roots, lr / bc1 and
# the decoupled-decay factor are such spans: computed per element before.)
import re as _re

_ELEMENT = _re.compile(r"\b(i|j|v|q|u|base|e|i0|gv)\b|\bin_\d+\b|\br_\w+|\bc_\w+|\bp6in\d+")


def _spans(text):
    out, stack = [], []
    for k, c in enumerate(text):
        if c in '([':
            stack.append((k, c))
        elif c in ')]' and stack:
            a, opener = stack.pop()
            if (opener, c) not in (('(', ')'), ('[', ']')):
                continue
            while a > 0 and (text[a - 1].isalnum() or text[a - 1] == '_' or text[a - 1] == ':'):
                a -= 1
            if opener == '[' and a == k:
                continue
            out.append((a, k + 1))
    return out


def licm(statements, prefix):
    """(declarations, statements') with maximal invariant spans of the right-hand sides hoisted."""
    names, decls, out = {}, [], []
    for stmt in statements:
        eq = stmt.find(' = ')
        if eq < 0:
            out.append(stmt)
            continue
        lhs, text = stmt[:eq + 3], stmt[eq + 3:]
        while True:
            best = None
            for a, b in _spans(text):
                span = text[a:b]
                if _ELEMENT.search(span) or not _re.search(r"\[|\w\(", span) or len(span) <= 8:
                    continue
                if _re.fullmatch(r"\(?" + prefix + r"\d+\)?", span):
                    continue
                if best is None or b - a > best[1] - best[0]:
                    best = (a, b)
            if best is None:
                break
            a, b = best
            span = text[a:b]
            if span not in names:
                names[span] = f"{prefix}{len(names)}"
                decls.append(f"const auto {names[span]} = {span};")
            text = text[:a] + f"({names[span]})" + text[b:]
        out.append(lhs + text)
    return decls, out


# ------------------------------------------------------------------------------------------------ fast activations
# ``p6e_fast_act`` (numerical policy, tolerance-gated): sigmoid / silu / silu_backward operations whose result dtype
# is BF16 are evaluated inline with the hardware exponential (ex2.approx via __expf) and fast division instead of the
# out-of-line IEEE expf/division calls. The FP32 value differs from the IEEE one by a few FP32 ulps (relative error
# ~1e-6), three orders below the BF16 rounding that follows immediately; a rounded result changes only when the
# exact value lies within that error of a BF16 rounding boundary. This is the precision the conventional control's
# Triton kernels use for the same operations.
FAST_ACT = r'''
#ifndef LATCH_P6E_FAST_ACT
#define LATCH_P6E_FAST_ACT
__device__ __forceinline__ float latch_fsig(float x) { return __fdividef(1.0f, 1.0f + __expf(-x)); }
__device__ __forceinline__ float latch_fsilu(float x) { return x * latch_fsig(x); }
__device__ __forceinline__ float latch_fsilu_bwd(float g, float x) {
  const float s = latch_fsig(x);
  return g * s * (1.0f + x * (1.0f - s));
}
#endif
'''


# ------------------------------------------------------------------------------------------------ scan placement
# ``p6e_scan_place``: a backward scan whose BF16 gradients du | ddelta | dB | dC | dz are each read only by one
# slice_backward along the trailing axis (step 1) into the same [B, T, W] shape, combined by an add chain (alpha 1,
# intermediates read only by the chain) whose final sum has no other terms, with disjoint ranges covering [0, W),
# writes those gradients directly into the final sum's storage (row stride W). The final sum becomes a view of the
# scan's packed result and the slice_backward/add operations are not emitted. The scan stores -0 as +0 there
# (round(x) + 0 in the reference chain); every value is bitwise the reference result. Dispatch: operation kinds,
# axes, ranges, shapes and dtypes only.
SCAN_GRADIENTS = (0, 1, 3, 4, 6)       # du, ddelta, dB, dC, dz of latch_ops::selective_scan_bwd


def scan_placements(operations, values, *, names, ref, field, output_name, flatten_refs, bwd_target):
    readers = {}
    producer = {}
    for op in operations:
        for o in field(op, "outputs", None) or [output_name(op)]:
            producer[o] = op
        args, kwargs = list(field(op, "args", [])), dict(field(op, "kwargs", {}))
        for r in dict.fromkeys(flatten_refs((args, kwargs), names)):
            readers.setdefault(r, []).append(op)

    def base(op):
        return str(field(op, "target", field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.").split(".")[0]
    plans = {}
    for op in operations:
        target = str(field(op, "target", field(op, "op"))).removeprefix("torch.ops.").removeprefix("aten.")
        if target != bwd_target:
            continue
        results = list(field(op, "outputs"))
        cols, skip, finals, shape = {}, set(), set(), None
        ok = True
        for index in SCAN_GRADIENTS:
            name = results[index]
            rs = readers.get(name, [])
            if len(rs) != 1 or base(rs[0]) != "slice_backward":
                ok = False
                break
            sb = rs[0]
            a = list(field(sb, "args", []))
            if len(a) < 5 or ref(a[0]) != name:
                ok = False
                break
            sizes = tuple(int(x) for x in a[1])
            dim = int(a[2]) % len(sizes)
            step = int(a[5]) if len(a) > 5 else 1
            start, end = int(a[3]), int(a[4])
            width = sizes[-1]
            if start < 0:
                start += width
            end = min(end if end >= 0 else end + width, width)
            if dim != len(sizes) - 1 or step != 1 or (shape is not None and sizes != shape):
                ok = False
                break
            shape = sizes
            cols[index] = start
            skip.add(output_name(sb))
            # climb the add chain to its final sum
            cur = output_name(sb)
            while True:
                rs2 = readers.get(cur, [])
                if len(rs2) != 1 or base(rs2[0]) != "add":
                    break
                add = rs2[0]
                aa = list(field(add, "args", []))
                alpha = dict(field(add, "kwargs", {})).get("alpha", aa[2] if len(aa) > 2 else 1)
                if alpha != 1:
                    ok = False
                    break
                skip.add(output_name(add))
                cur = output_name(add)
            finals.add(cur)
        if not ok or len(finals) != 1 or shape is None:
            continue
        final = finals.pop()
        if final not in skip:
            continue
        # every add in the chain must read only chain values (no other terms); ranges disjoint and covering [0, W)
        chain_ok = True
        for nm in skip:
            p = producer.get(nm)
            if p is None:
                chain_ok = False
                break
            if base(p) == "add":
                for r in flatten_refs((list(field(p, "args", [])), dict(field(p, "kwargs", {}))), names):
                    if r not in skip:
                        chain_ok = False
            elif nm != final and any(output_name(q) not in skip for q in readers.get(nm, [])):
                chain_ok = False
            if nm not in values or str(field(values[nm], "dtype")) != "bfloat16":
                chain_ok = False
        spans = []
        for index, start in cols.items():
            rshape = tuple(int(x) for x in field(values[results[index]], "shape"))
            spans.append((start, start + rshape[-1]))
        spans.sort()
        if not chain_ok or spans[0][0] != 0 or spans[-1][1] != shape[-1] or any(
                spans[k][1] != spans[k + 1][0] for k in range(len(spans) - 1)):
            continue
        plans[output_name(op)] = {"name": final, "width": shape[-1], "cols": cols, "skip": skip, "shape": shape}
    return plans


# ``p6e_inline_div``: force-inlined copy of the outlined IEEE division used inside fused elementwise loops.
INLINE_DIV = r'''
#ifndef LATCH_P6E_INLINE_DIV
#define LATCH_P6E_INLINE_DIV
__device__ __forceinline__ float latch_fdiv_i(float a, float b) { return a / b; }
#endif
'''
