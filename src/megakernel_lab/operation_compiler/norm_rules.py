"""P-OPS3-C normalization rules: exact row/column reduction engines and row-consumer attachment.

Normalization layers lower to a reduction over a row (RMSNorm statistics, the row term of its derivative),
a reduction over the leading rows (the normalization weight gradient) and elementwise consumers that broadcast
the row statistic back over the row. The fused schedule previously gave every non-matrix heavy operation its own
grid phase: a CTA-per-row reduction, then a separate elementwise phase that re-read the row from memory, and the
weight-gradient reduction serialized next to it with lanes striding across rows (uncoalesced).

These rules change only scheduling and the assignment of work to threads, never the arithmetic:

* ``row engine``: a warp owns one output row. Lane l accumulates the leaves of the frozen CTA threads
  t = 32 w + l (w = 0..7) exactly as the frozen loop does (``acc = 0; acc += leaf(t + 256 j)``), then replays the
  frozen tree: an xor butterfly per w (the frozen per-warp ``warp_sum``) and the second-stage butterfly over the
  eight warp partials with zero lanes. The stored statistic is bitwise identical to the frozen CTA tree.
* ``column engine``: a reduction over leading rows (kept trailing columns). Lanes own columns, so every load is a
  contiguous segment; each thread evaluates the lane-0 butterfly subtree of four frozen threads per group
  (levels 16 and 8 in registers), levels 4/2/1 by register and two xor shuffles, and the second stage from shared
  memory with the frozen zero lanes. Bitwise identical to the frozen tree.
* ``row attachment``: an elementwise operation over the same [rows, extent] index space that reads the row
  statistic only at its own row (and otherwise only earlier values, graph inputs or members already attached) is
  evaluated by the owning warp right after the statistic, from a register. A same-shaped operation in the
  statistic's phase that shares a loaded operand with the reduction leaf may attach too (sibling), so its
  consumers can follow. Attached expressions are the unchanged per-operation expressions.
* A row reduction and one column reduction may share a phase on disjoint block ranges.

Dispatch uses only reduction axes, extents, layouts and dependencies; no model names or layer roles.
"""
from __future__ import annotations

import re

ROW_MIN_ROWS = 64          # fewer rows keep the CTA tree (a warp per row would leave the grid idle)
ROW_MAX_LEAVES = 16        # leaves per frozen thread (extent <= 4096)
COL_MIN_COLUMNS = 32
COL_MAX_LEAVES = 16        # rows per frozen thread for the column engine (reduced extent <= 4096)
COL_GROUP = 8              # columns per CTA iteration (one 32-byte segment per row and lane group)
GRID_CAP = 128


def position(src_shape, src_strides, dims, kept, rvar, ovar):
    """Storage offset of source element (reduced flat index rvar, kept flat index ovar); frozen formula.

    Emitted in 32-bit unsigned arithmetic (callers admit only sources below 2**31 elements)."""
    from .lower import _numel
    reduction_shape = tuple(src_shape[i] for i in dims)
    kept_shape = tuple(src_shape[i] for i in kept)
    terms = []
    for sequence, subshape, index in [(dims, reduction_shape, rvar), (kept, kept_shape, ovar)]:
        for j, axis in enumerate(sequence):
            inner, extent, stride = _numel(subshape[j+1:]), src_shape[axis], src_strides[axis]
            coord = index if inner == 1 else f"({index} / {inner}u)"
            if j != 0:
                coord = f"({coord} % {extent}u)"
            terms.append(coord if stride == 1 else f"{coord} * {stride}u")
    return " + ".join(terms) or "0u"


def _span(shape, strides):
    return 1 + sum((d - 1) * s for d, s in zip(shape, strides))


def warps_per_row(rows, blocks):
    """Warps cooperating on one row so a block's eight warps are busy in one round (power of two)."""
    per_block = -(-rows // max(1, blocks))
    group = 1
    while group < 8 and group < per_block:
        group *= 2
    return 8 // group


def row_eligible(src_shape, src_strides, dims, results, extent):
    """Reduced axes are the trailing axes, innermost unit stride; enough rows; bounded extent."""
    n = len(src_shape)
    if _span(src_shape, src_strides) >= 2**31:
        return False
    if not dims or list(dims) != list(range(n - len(dims), n)):
        return False
    if src_strides[-1] != 1 or extent <= 32 or -(-extent // 256) > ROW_MAX_LEAVES or results < ROW_MIN_ROWS:
        return False
    # Reduced axes merge into one contiguous run (coalesced rows).
    stride = 1
    for axis in reversed(dims):
        if src_strides[axis] != stride:
            return False
        stride *= src_shape[axis]
    return True


def column_eligible(src_shape, src_strides, dims, results, extent):
    """Reduced axes are the leading axes; kept trailing axes are contiguous with unit innermost stride."""
    n = len(src_shape)
    if _span(src_shape, src_strides) >= 2**31:
        return False
    if not dims or list(dims) != list(range(len(dims))) or len(dims) == n:
        return False
    if extent <= 32 or -(-extent // 256) > COL_MAX_LEAVES or results < COL_MIN_COLUMNS:
        return False
    stride = 1
    for axis in reversed(range(len(dims), n)):
        if src_strides[axis] != stride:
            return False
        stride *= src_shape[axis]
    return True


def row_code(*, out, rows, extent, leaf, rounded, scale, blocks=GRID_CAP):
    """Row engine. W warps share a row; warp k of a row group owns frozen threads 32 w + lane for its 8/W values
    of w. `leaf` reads source element (prow, pcol). Markers: /*ROW_LO*/, /*ROW_MEMBERS*/ (members see g, lane,
    latch_w0, latch_nw; one unrolled element loop per owned w)."""
    leaves = -(-extent // 256)
    return f'''{{ // P-OPS3-C row engine: {rows} rows x {extent} (frozen CTA tree, /*ROW_W*/ warps per row)
  const unsigned latch_lo = /*ROW_LO*/0u;
  if (blockIdx.x >= latch_lo) {{
    constexpr unsigned latch_wr = /*ROW_WR*/8u, latch_nw = 8u / latch_wr, latch_rb = 8u / latch_wr;
    const unsigned latch_nb = gridDim.x - latch_lo;
    const unsigned lane = threadIdx.x & 31u, latch_slot = (threadIdx.x >> 5) / latch_wr;
    const unsigned latch_w0 = ((threadIdx.x >> 5) % latch_wr) * latch_nw;
    unsigned latch_buf = 0u;
    for (unsigned latch_base = (blockIdx.x - latch_lo) * latch_rb; latch_base < {rows}u; latch_base += latch_nb * latch_rb) {{
      const unsigned g = latch_base + latch_slot;
      float* latch_part = smem + latch_buf * 64u + latch_slot * 8u;
      if (g < {rows}u) {{
        const unsigned prow = g;
#pragma unroll
        for (unsigned v = 0; v < latch_nw; ++v) {{
          float acc = 0.0f;
#pragma unroll
          for (int j = 0; j < {leaves}; ++j) {{
            const unsigned pcol = 32u * (latch_w0 + v) + lane + 256u * j;
            if (pcol < {extent}u) acc += {leaf};
          }}
          acc = s101h::warp_sum(acc);
          if (lane == 0u) latch_part[latch_w0 + v] = acc;
        }}
      }}
      __syncthreads();
      if (g < {rows}u) {{
        float second = lane < 8u ? latch_part[lane] : 0.0f;
        second = s101h::warp_sum(second);
        if (latch_w0 == 0u && lane == 0u) {out}[g] = {rounded('second' + scale)};
/*ROW_MEMBERS*/
      }}
      latch_buf ^= 1u;
    }}
  }}
}}'''


def single_warp_rows(code):
    """P6-B: a row engine with one warp per row keeps its eight frozen per-warp partials in registers (lane v holds
    partial v after the xor butterflies) instead of shared memory, so the second stage needs no block barrier and
    the CTA's warps run their rows independently. Same values, same second-stage butterfly: bitwise identical."""
    stage = """          acc = s101h::warp_sum(acc);
          if (lane == 0u) latch_part[latch_w0 + v] = acc;
        }
      }
      __syncthreads();
"""
    second = "        float second = lane < 8u ? latch_part[lane] : 0.0f;\n"
    head = "      float* latch_part = smem + latch_buf * 64u + latch_slot * 8u;\n"
    if code.count(stage) != 1 or code.count(second) != 1 or code.count(head) != 1:
        raise ValueError("row engine text changed; single-warp register rule not applicable")
    code = code.replace(head, "      float latch_mine = 0.0f;  // lane v < 8: frozen per-warp partial v\n")
    code = code.replace(stage, """          acc = s101h::warp_sum(acc);
          if (lane == v) latch_mine = acc;
        }
      }
""")
    return code.replace(second, "        float second = lane < 8u ? latch_mine : 0.0f;\n")


def column_code(*, out, columns, extent, leaf, rounded, scale, group=None):
    """Coalesced exact column engine. `leaf` reads source element (reduced pr, kept o). Marker: /*COL_SPAN*/.

    group 8: lanes = 4 sub-groups x 8 columns, each thread two lane-0 subtrees of four frozen threads.
    group 4: lanes = 8 sub-groups x 4 columns, each thread one subtree (half the leaves, twice the blocks)."""
    COL_GROUP = group or globals()['COL_GROUP']
    if COL_GROUP == 4:
        return _column_code4(out=out, columns=columns, extent=extent, leaf=leaf, rounded=rounded, scale=scale)
    leaves = -(-extent // 256)
    groups = -(-columns // COL_GROUP)
    tail = rounded('s0' + scale)
    return f'''{{ // P-OPS3-C column engine: {columns} columns x {extent} rows (frozen CTA tree, lanes over columns)
  const unsigned latch_cn = /*COL_SPAN*/min({groups}u, gridDim.x);
  if (blockIdx.x < latch_cn) {{
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5, sub = lane >> 3, cl = lane & 7u;
    for (unsigned cg = blockIdx.x; cg < {groups}u; cg += latch_cn) {{
      const unsigned o = cg * {COL_GROUP}u + cl;
      float t2[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {{
        float x[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {{
          const unsigned t = 32u * w + sub + 4u * h + 8u * q;
          float acc = 0.0f;
#pragma unroll
          for (int j = 0; j < {leaves}; ++j) {{
            const unsigned pr = t + 256u * j;
            if (pr < {extent}u && o < {columns}u) acc += {leaf};
          }}
          x[q] = acc;
        }}
        t2[h] = (x[0] + x[2]) + (x[1] + x[3]);
      }}
      float v = t2[0] + t2[1];
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      if (sub == 0u) smem[w * {COL_GROUP}u + cl] = v;
      __syncthreads();
      if (threadIdx.x < {COL_GROUP}u && cg * {COL_GROUP}u + threadIdx.x < {columns}u) {{
        float s[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] = (smem[l * {COL_GROUP} + threadIdx.x] + 0.0f) + 0.0f;
        const float s0 = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
        {out}[cg * {COL_GROUP}u + threadIdx.x] = {tail};
      }}
      __syncthreads();
    }}
  }}
}}'''


def _column_code4(*, out, columns, extent, leaf, rounded, scale):
    leaves = -(-extent // 256)
    groups = -(-columns // 4)
    tail = rounded('s0' + scale)
    return f'''{{ // P-OPS3-C column engine: {columns} columns x {extent} rows (frozen CTA tree, 4 columns per CTA)
  const unsigned latch_cn = /*COL_SPAN*/min({groups}u, gridDim.x);
  if (blockIdx.x < latch_cn) {{
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5, sub = lane >> 2, cl = lane & 3u;
    for (unsigned cg = blockIdx.x; cg < {groups}u; cg += latch_cn) {{
      const unsigned o = cg * 4u + cl;
      float x[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) {{
        const unsigned t = 32u * w + sub + 8u * q;
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < {leaves}; ++j) {{
          const unsigned pr = t + 256u * j;
          if (pr < {extent}u && o < {columns}u) acc += {leaf};
        }}
        x[q] = acc;
      }}
      float v = (x[0] + x[2]) + (x[1] + x[3]);
      v += __shfl_xor_sync(0xffffffffu, v, 16);
      v += __shfl_xor_sync(0xffffffffu, v, 8);
      v += __shfl_xor_sync(0xffffffffu, v, 4);
      if (sub == 0u) smem[w * 4u + cl] = v;
      __syncthreads();
      if (threadIdx.x < 4u && cg * 4u + threadIdx.x < {columns}u) {{
        float s[8];
#pragma unroll
        for (int l = 0; l < 8; ++l) s[l] = (smem[l * 4 + threadIdx.x] + 0.0f) + 0.0f;
        const float s0 = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
        {out}[cg * 4u + threadIdx.x] = {tail};
      }}
      __syncthreads();
    }}
  }}
}}'''


def normalized_index(text):
    return re.sub(r"[\s()]", "", text)


def row_index_text(extent):
    return f"i/{extent}u"


OUTLINED = r"""
#ifndef LATCH_OPS_OUTLINED
#define LATCH_OPS_OUTLINED
__device__ __noinline__ float latch_powf(float x, float y) { return powf(x, y); }
__device__ __noinline__ float latch_expf(float x) { return expf(x); }
__device__ __noinline__ float latch_logf(float x) { return logf(x); }
__device__ __noinline__ float latch_fdiv(float a, float b) { return a / b; }
__device__ __noinline__ float latch_sigmoid(float x) { return (1.0f / (1.0f + expf(-x))); }
__device__ __noinline__ float latch_silu(float x) { return (x / (1.0f + expf(-x))); }
__device__ __noinline__ float latch_silu_backward(float g, float x) {
  const float s = (1.0f / (1.0f + expf(-x)));
  return (g * s * (1.0f + x * (1.0f - s)));
}
#endif
"""
_OUTLINE = re.compile(r"(?<![\w:.])(powf|expf|logf)\(")


def outline(body):
    """Route generated libdevice transcendental calls to their single out-of-line copies."""
    count = len(_OUTLINE.findall(body)) + body.count("latch_fdiv(") + body.count("latch_sigmoid(") + body.count("latch_silu")
    return _OUTLINE.sub(lambda m: "latch_" + m.group(1) + "(", body), count


FACTORABLE = {"add", "sub", "mul", "div", "neg", "pow", "rsqrt", "sqrt", "exp", "expm1", "log1p", "log", "reciprocal",
              "tanh", "sigmoid", "silu", "relu", "abs", "clone", "_to_copy"}


PURE_CALLS = ("latch_sigmoid", "latch_silu_backward", "latch_silu", "latch_fdiv", "powf", "expf", "logf")


# P6-B: further pure single-result math calls (same function on the same argument text gives the same bits).
PURE_CALLS_P6B = PURE_CALLS + ("tanhf", "sqrtf", "rsqrtf", "erff", "expm1f", "log1pf", "sinf", "cosf", "latch_expf",
                                 "latch_logf", "latch_powf", "latch_sigmoid")


def _call_spans(text, calls):
    found = []
    for name in calls:
        for match in re.finditer(r"(?<![\w:.])" + re.escape(name) + r"\(", text):
            depth, k = 0, match.end() - 1
            while k < len(text):
                depth += text[k] == "("
                depth -= text[k] == ")"
                if depth == 0:
                    break
                k += 1
            found.append(text[match.start():k + 1])
    return found


def cse_lines(lines, prefix, calls=PURE_CALLS_P6B, rounds=2):
    """P6-B: block-level CSE of pure calls repeated across the sequential lines of one element's chain. Each repeated
    call text is evaluated once into a temporary right before the first line using it (its arguments are defined by
    then) and that value is reused (same function, same argument: identical bits). Inner calls first; a second round
    catches outer calls that became identical after the inner replacement."""
    lines = list(lines)
    counter = 0
    for _ in range(rounds):
        counts = {}
        for line in lines:
            for call in _call_spans(line, calls):
                counts[call] = counts.get(call, 0) + 1
        repeated = sorted((c for c, n in counts.items() if n > 1), key=len)
        if not repeated:
            break
        for call in repeated:
            pattern = re.compile(r"(?<![\w:.])" + re.escape(call))
            hits = [k for k, line in enumerate(lines) if pattern.search(line)]
            if len(hits) < 2 and not (hits and len(pattern.findall(lines[hits[0]])) > 1):
                continue
            temp = f"{prefix}_{counter}"
            counter += 1
            first = hits[0]
            lines = (lines[:first] + [f"const float {temp} = {call};"]
                     + [pattern.sub(temp, line) for line in lines[first:]])
    return lines


def cse_calls(expression, prefix, calls=PURE_CALLS):
    """Evaluate each repeated pure outlined call of one element expression once (its value is unchanged)."""
    prelude = []
    while True:
        counts = {}
        for name in calls:
            for match in re.finditer(r"(?<![\w:.])" + re.escape(name) + r"\(", expression):
                depth, k = 0, match.end() - 1
                while k < len(expression):
                    depth += expression[k] == "("
                    depth -= expression[k] == ")"
                    if depth == 0:
                        break
                    k += 1
                call = expression[match.start():k + 1]
                counts[call] = counts.get(call, 0) + 1
        repeated = [c for c, n in counts.items() if n > 1]
        if not repeated:
            return prelude, expression
        call = min(repeated, key=len)            # innermost first; outer calls are found again afterwards
        temp = f"{prefix}_{len(prelude)}"
        prelude.append(f"const float {temp} = {call};")
        expression = expression.replace(call, temp)
