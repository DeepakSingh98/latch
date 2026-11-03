"""More bytes in flight in memory-bound bodies of the persistent program.

Every form is opt-in (``elementwise_rules`` / ``accumulation_rules`` names below); with none selected nothing is
rewritten and the emitted source is unchanged. ``rewrite`` runs on finished phase code, after the routing-pack sum
fusion, so fused gathered loads are kept as they are.

* ``elementwise_rules`` ``permrows`` -- permutation embedding gradients (``permutation_fixed_sum_v1`` and
  ``permutation_gather_v1``) as warp-per-row(s) bodies: one warp owns R whole output rows; lanes 0..K-1 load the
  row's K inverse entries once (shuffled to the warp, sorted with the same network) and check the bijection in
  parallel; every lane then loads its column units of B occurrences x R rows before any addition (R x B x
  ceil(units/32) x operands independent loads in flight per lane, target ``ROW_LOADS`` row-operand loads per batch,
  ``ROW_BATCH`` occurrences per batch). Each element still adds its occurrences in ascending ordinal from +0.0 with
  the same per-occurrence expression text (including fused pack sums) and stores once. Requires the canonical
  one-index-per-thread form. One occurrence per row (permutation gathers) of BF16 rows into BF16 rows with the plain
  widening contribution use a register-light form: one warp per ``GATHER_ROWS`` rows, 16-byte moves, all rows'
  loads issued before any store; the value stored is the BF16 rounding of +0.0 + widen(x), i.e. x with -0 replaced
  by +0 (exact). The bijection check is deferred: the per-occurrence check ``if (q out of range || (col == 0 &&
  idx[q] != row)) __trap();`` becomes a flag (out-of-range entries are clamped to 0 before the address is formed)
  and one ``__trap()`` after the body's loop. A phase that refuses the warp-rows form gets only this deferred check:
  in the column-0 lanes the original check loads ``idx[q]`` and branches on it before the next occurrence's loads
  can issue. Same values, same order of additions, same checks (a violated bijection still traps, after the loop
  instead of at the element).
* ``elementwise_rules`` ``rowcopy`` -- ``validated_bijective_row_copy`` (latch_ops.permutation_gather, 16-byte row
  form, see row_permutation.py): rows of at most 128 vectors take one warp per ``ROWCOPY_ROWS`` destination rows;
  lanes 0..R-1 load each row's source index once and check it (range + inverse) into a deferred-trap flag; all R
  rows' 16-byte loads issue before any store. Same bytes moved, same checks; the run-time alignment test and scalar
  fallback are kept.
* ``accumulation_rules`` ``rnarrow`` -- rows narrower than a warp (``exact_cta_tree_per_warp_narrow`` softmax /
  log-softmax and their backward, ``argmax_warp_row`` with width <= 16): one row per G-lane group (G = next power
  of two >= width), 32/G rows per warp step and ``NARROW_UNROLL`` steps whose loads are issued before any
  reduction. The original warp form combines lane l with lanes l^16 .. l^G first, which all hold the identity
  (0.0f / -INFINITY / no candidate); x + 0.0f == x for the non-(-0) leaves (+0.0f + v), fmaxf(x, -INFINITY) == x
  for the non-NaN leaves (fmaxf(-INFINITY, x)), and the second level only re-adds identities to lane 0's value. The
  group butterfly (offsets G/2 .. 1) therefore pairs the same values in the same order: identical bits. Argmax uses
  the same total order (NaN maximal, value, lowest index), whose maximum is order independent.
* ``elementwise_rules`` ``embunroll`` -- the scalar per-element ``embedding`` gather (tables whose width is not a
  multiple of a 16-byte chunk): ``EMB_UNROLL`` grid-stride elements per thread with every index load, then every
  table load, issued before the stores (same element copies, same checks).
* ``accumulation_rules`` ``rmid`` -- softmax rows with 32 < width <= 256 and width % 32 != 0 take the existing
  warp-per-row form (bitwise equal to the 256-thread block-per-row tree) instead of one CTA per row with four
  block barriers per row.
"""
from __future__ import annotations

import re

VERSION = 1
NARROW_UNROLL = 8        # 'rnarrow': warp steps per iteration
EMB_UNROLL = 8           # 'embunroll': elements per thread in flight
ROWCOPY_ROWS = 16        # 'rowcopy': destination rows per warp
ROW_LOADS = 8            # 'permrows': row-operand loads per batch
GATHER_ROWS = 8          # 'permrows': rows per warp of the K = 1 BF16 gather
ROW_BATCH = 4            # 'permrows': occurrences per load batch


def _elementwise(name):
    from .elementwise_rules import active
    return active(name)


def _accumulation(name):
    from .accumulation_rules import active
    return active(name)


def rows():
    return _elementwise("permrows")


def rowcopy():
    return _elementwise("rowcopy")


def narrow():
    return _accumulation("rnarrow")


def _narrow_geometry(width):
    G = 1
    while G < width:
        G *= 2
    return G, 32 // G, NARROW_UNROLL


PACK_HEAD = re.compile(r"const long long mlp_rhi_ = (\d+)LL;\n"
                       r"for \(long long mlp_r0_ = \(\(\(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x\) >> 5\) \* (\d+)LL; "
                       r"mlp_r0_ < mlp_rhi_; mlp_r0_ \+= \(\(\(long long\)gridDim\.x \* blockDim\.x\) >> 5\) \* \2LL\) \{")


def pack_restrict(code):
    """row_chain: the packed narrow body restricted to the CTA's row block [latch_rc_r0, latch_rc_r1)."""
    return PACK_HEAD.sub(lambda m: ("const long long mlp_rhi_ = latch_rc_r1;\n"
                                    f"for (long long mlp_r0_ = latch_rc_r0 + (long long)(threadIdx.x >> 5) * {m[2]}LL; "
                                    f"mlp_r0_ < mlp_rhi_; mlp_r0_ += (long long)(blockDim.x >> 5) * {m[2]}LL) {{"),
                         code, count=1)


def narrow_softmax_code(*, rows, width, x, g, out, backward, logged, safe):
    """Packed narrow-row softmax (see module docstring); x / g are element texts in `row` and `k`."""
    if width < 1 or width > 16:
        return None
    G, P, U = _narrow_geometry(width)
    step = P * U
    offs = []
    o = G // 2
    while o >= 1:
        offs.append(o)
        o //= 2

    def red(var, op, first):
        comb = (f"{var} = fmaxf({var}, __shfl_xor_sync(0xffffffffu, {var}, {{o}}));" if op == "max" else
                f"{var} += __shfl_xor_sync(0xffffffffu, {var}, {{o}});")
        return f"float {var} = {first};" + "".join(" " + comb.format(o=o) for o in offs)
    lines = [f"{{  // Wide-issue narrow rows: {P} rows per warp step ({G}-lane groups), {U} steps per iteration",
             "const int mlp_lane_ = threadIdx.x & 31;",
             f"const int mlp_sub_ = mlp_lane_ & {G - 1}, mlp_grp_ = mlp_lane_ / {G};",
             f"const long long mlp_rhi_ = {rows}LL;",
             f"for (long long mlp_r0_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * {step}LL; mlp_r0_ < mlp_rhi_; "
             f"mlp_r0_ += (((long long)gridDim.x * blockDim.x) >> 5) * {step}LL) {{",
             f"  float mlp_x_[{U}];" + (f" float mlp_g_[{U}];" if backward else ""),
             "#pragma unroll",
             f"  for (int u_ = 0; u_ < {U}; ++u_) {{",
             f"    const long long row = mlp_r0_ + u_ * {P} + mlp_grp_; const long long k = mlp_sub_;",
             f"    const bool ok_ = mlp_sub_ < {width} && row < mlp_rhi_;",
             f"    mlp_x_[u_] = ok_ ? {x} : 0.0f;" + (f" mlp_g_[u_] = ok_ ? {g} : 0.0f;" if backward else ""),
             "  }",
             "#pragma unroll",
             f"  for (int u_ = 0; u_ < {U}; ++u_) {{",
             f"    const long long row = mlp_r0_ + u_ * {P} + mlp_grp_; const long long k = mlp_sub_;",
             "    const float xv = mlp_x_[u_];" + (" const float gv = mlp_g_[u_];" if backward else "")]
    if backward:
        summand = "gv" if logged else "(gv * xv)"
        lines.append("    " + red("total", "sum", f"mlp_sub_ < {width} ? 0.0f + {summand} : 0.0f"))
        result = "(gv - expf(xv) * total)" if logged else "(xv * (gv - total))"
    else:
        lines.append("    " + red("maximum", "max", f"mlp_sub_ < {width} ? fmaxf(-INFINITY, xv) : -INFINITY"))
        lines.append("    " + red("total", "sum", f"mlp_sub_ < {width} ? 0.0f + expf(xv - maximum) : 0.0f"))
        result = "(xv - maximum - logf(total))" if logged else "(expf(xv - maximum) / total)"
        if safe:
            result = f"(maximum == -INFINITY ? 0.0f : {result})"
    lines += [f"    if (mlp_sub_ < {width} && row < mlp_rhi_) {out}[row * {width}LL + k] = {result};", "  }", "}", "}"]
    return "\n".join(lines)


def narrow_argmax_code(*, rows, width, x, out, better):
    """Packed narrow-row argmax over the last axis (width <= 16)."""
    if width < 1 or width > 16:
        return None
    G, P, U = _narrow_geometry(width)
    step = P * U
    offs = []
    o = G // 2
    while o >= 1:
        offs.append(o)
        o //= 2
    comb = "".join(f"""
      {{ const float v = __shfl_xor_sync(0xffffffffu, best, {o}); const long long kk = __shfl_xor_sync(0xffffffffu, index, {o});
        if ({better}(v, kk, best, index)) {{ best = v; index = kk; }} }}""" for o in offs)
    return f"""{{  // Wide-issue narrow argmax: {P} rows per warp step ({G}-lane groups), {U} steps per iteration
const int mlp_lane_ = threadIdx.x & 31;
const int mlp_sub_ = mlp_lane_ & {G - 1}, mlp_grp_ = mlp_lane_ / {G};
for (long long mlp_r0_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * {step}LL; mlp_r0_ < {rows}LL;
     mlp_r0_ += (((long long)gridDim.x * blockDim.x) >> 5) * {step}LL) {{
  float mlp_x_[{U}];
#pragma unroll
  for (int u_ = 0; u_ < {U}; ++u_) {{
    const long long row = mlp_r0_ + u_ * {P} + mlp_grp_; const long long k = mlp_sub_;
    mlp_x_[u_] = (mlp_sub_ < {width} && row < {rows}LL) ? {x} : -INFINITY;
  }}
#pragma unroll
  for (int u_ = 0; u_ < {U}; ++u_) {{
    const long long row = mlp_r0_ + u_ * {P} + mlp_grp_;
    float best = -INFINITY; long long index = 0x7fffffffffffffffLL;
    if (mlp_sub_ < {width} && {better}(mlp_x_[u_], (long long)mlp_sub_, best, index)) {{ best = mlp_x_[u_]; index = mlp_sub_; }}{comb}
    if (mlp_sub_ == 0 && row < {rows}LL) {out}[row] = static_cast<int64_t>(index);
  }}
}}
}}"""


def midrow():
    return _accumulation("rmid")


def emb():
    return _elementwise("embunroll")


def scalar_embedding_code(*, numel, index, table, out, bound, ctype):
    """`index` / `table` are element texts in `i` / `row` (see module docstring)."""
    U = EMB_UNROLL
    st = "(long long)gridDim.x * blockDim.x"
    return f"""{{  // Wide-issue scalar embedding: {U} elements per thread in flight
for (long long mlp_i0_ = (long long)blockIdx.x * blockDim.x + threadIdx.x; mlp_i0_ < {numel}LL; mlp_i0_ += {U}LL * {st}) {{
  int64_t mlp_r_[{U}]; {ctype} mlp_v_[{U}];
#pragma unroll
  for (int u_ = 0; u_ < {U}; ++u_) {{ const long long i = mlp_i0_ + u_ * ({st}); mlp_r_[u_] = i < {numel}LL ? {index} : 0; }}
#pragma unroll
  for (int u_ = 0; u_ < {U}; ++u_) {{
    const long long i = mlp_i0_ + u_ * ({st}); const int64_t row = mlp_r_[u_];
    assert(row >= 0 && row < {bound}LL);
    if (i < {numel}LL) mlp_v_[u_] = {table};
  }}
#pragma unroll
  for (int u_ = 0; u_ < {U}; ++u_) {{ const long long i = mlp_i0_ + u_ * ({st}); if (i < {numel}LL) {out}[i] = mlp_v_[u_]; }}
}}
}}"""


def any_enabled():
    return rows()


def rowcopy_code(*, rows, per_row, stride, total, src, dst, order, inverse, indent):
    """Warp-per-rows form of row_permutation's 16-byte row copy (see module docstring)."""
    R = ROWCOPY_ROWS
    P = -(-per_row // 32)
    return f"""{{  // Wide-issue row copy: warp per {R} rows
  const uint4* pgx_ = reinterpret_cast<const uint4*>({src});
  uint4* pgy_ = reinterpret_cast<uint4*>({dst});
  const int64_t* pgo_ = {order};
  const int64_t* pgq_ = {inverse};
  if (((reinterpret_cast<unsigned long long>(pgx_) | reinterpret_cast<unsigned long long>(pgy_)) & 15ULL) == 0ULL) {{
    bool mlp_bad_ = false;
    const int mlp_lane_ = threadIdx.x & 31;
    for (long long r0_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * {R}LL; r0_ < {rows}LL;
         r0_ += (((long long)gridDim.x * blockDim.x) >> 5) * {R}LL) {{
      int64_t sl_ = 0;
      if (mlp_lane_ < {R} && r0_ + mlp_lane_ < {rows}LL) {{
        sl_ = pgo_[r0_ + mlp_lane_];
        if (sl_ < 0 || sl_ >= {rows}LL) {{ mlp_bad_ = true; sl_ = 0; }}
        mlp_bad_ |= pgq_[sl_] != r0_ + mlp_lane_;
      }}
      uint4 x_[{R}][{P}];
#pragma unroll
      for (int i_ = 0; i_ < {R}; ++i_) {{
        const long long s_ = __shfl_sync(0xffffffffu, sl_, i_);
#pragma unroll
        for (int c_ = 0; c_ < {P}; ++c_) {{
          const int ch_ = mlp_lane_ + 32 * c_;
          x_[i_][c_] = (ch_ < {per_row} && r0_ + i_ < {rows}LL) ? pgx_[s_ * {stride}LL + ch_] : make_uint4(0u, 0u, 0u, 0u);
        }}
      }}
#pragma unroll
      for (int i_ = 0; i_ < {R}; ++i_)
#pragma unroll
        for (int c_ = 0; c_ < {P}; ++c_) {{
          const int ch_ = mlp_lane_ + 32 * c_;
          if (ch_ < {per_row} && r0_ + i_ < {rows}LL) pgy_[(r0_ + i_) * {per_row}LL + ch_] = x_[i_][c_];
        }}
    }}
    if (mlp_bad_) __trap();
  }} else {{
{indent}
  }}
}}"""


# Canonical permutation_gradients.emit body (one grid-stride index per thread), possibly with the fused pack-sum block
# in place of the per-occurrence load statement.
_CANON = re.compile(
    r"^for \(long long u_ = \(long long\)blockIdx\.x \* blockDim\.x \+ threadIdx\.x; u_ < (?P<total>\d+)LL;\n"
    r"     u_ \+= \(long long\)gridDim\.x \* blockDim\.x\) \{\n"
    r"  const long long row = u_ / (?P<units>\d+)LL, col = u_ % (?P=units)LL;\n"
    r"  (?P<vt>float4|float) acc_ = (?P<zero>[^;]+);\n"
    r"  if \(row != (?P<pad>-?\d+)LL\) \{\n"
    r"    long long q_\[(?P<k>\d+)\];\n"
    r"#pragma unroll\n"
    r"    for \(int s_ = 0; s_ < (?P=k); \+\+s_\) q_\[s_\] = (?P<qload>[^\n]+);(?P<sort>.*?)\n"
    r"    (?P=vt) v_\[(?P=k)\];\n"
    r"#pragma unroll\n"
    r"    for \(int s_ = 0; s_ < (?P=k); \+\+s_\) \{\n"
    r"      if \(q_\[s_\] < 0 \|\| q_\[s_\] >= (?P<n>\d+)LL \|\| \(col == 0 && (?P<idx>v_\w+\[[^\n]*?\]) != row\)\) __trap\(\);\n"
    r"      (?P<load>[^\n]+)\n"
    r"    \}\n"
    r"#pragma unroll\n"
    r"    for \(int s_ = 0; s_ < (?P=k); \+\+s_\) (?P<add>[^\n]+)\n"
    r"  \}\n"
    r"  (?P<store>[^\n]+)\n"
    r"\}$", re.S)


_GLOAD = re.compile(r"^v_\[s_\] = latch_pgrad::widen4\(reinterpret_cast<const uint2\*>\((v_\w+) \+ \((.+)\)\)\[col\]\);$")
_GSTORE = re.compile(r"^reinterpret_cast<uint2\*>\((v_\w+) \+ row \* (\d+)LL\)\[col\] = latch_pgrad::round4\(acc_\);$")


def _gather_code(g, U, rows_, aligned16=None):
    """K = 1, BF16 -> BF16 widening gather: warp per R rows, 16-byte chunks (see module docstring)."""
    lm, sm = _GLOAD.match(g["load"].strip()), _GSTORE.match(g["store"].strip())
    if not lm or not sm or int(sm.group(2)) != 4 * U or (4 * U) % 8:
        return None, "gather form"
    G, at, OUT, W = lm.group(1), lm.group(2), sm.group(1), int(sm.group(2))
    if aligned16 is None or not (aligned16(G[2:]) and aligned16(OUT[2:])):
        return None, "16-byte alignment not proven"
    C = W // 8                       # 16-byte chunks per row
    P = -(-C // 32)
    R = GATHER_ROWS
    at_ = at.replace("q_[s_]", "mlp_qq_")
    idx_ = g["idx"].replace("q_[s_]", "mlp_ql_")
    pz = ("((W_ & 0xffffu) == 0x8000u ? (W_ & 0xffff0000u) : W_)", "((V_ >> 16) == 0x8000u ? (V_ & 0x0000ffffu) : V_)")

    def fix(w):
        return f"[&]() {{ const unsigned W_ = {w}; const unsigned V_ = {pz[0]}; return {pz[1]}; }}()"
    vrow = f"(row < {rows_}LL && row != {g['pad']}LL)"
    body = f"""{{  // Wide-issue gather rows: warp per {R} rows, 16-byte chunks
bool mlp_bad_ = false;
const int mlp_lane_ = threadIdx.x & 31;
for (long long mlp_r0_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * {R}LL; mlp_r0_ < {rows_}LL;
     mlp_r0_ += ((((long long)gridDim.x * blockDim.x) >> 5)) * {R}LL) {{
  long long mlp_ql_ = 0LL;
  {{ const long long row = mlp_r0_ + mlp_lane_;
    if (mlp_lane_ < {R} && {vrow}) {{
      const int s_ = 0; long long q_[1]; q_[0] = {g['qload']}; mlp_ql_ = q_[0];
      const bool out_ = mlp_ql_ < 0 || mlp_ql_ >= {g['n']}LL; if (out_) {{ mlp_bad_ = true; mlp_ql_ = 0LL; }}
      mlp_bad_ |= {idx_} != row;
    }} }}
  uint4 mlp_x_[{R}][{P}];
#pragma unroll
  for (int i_ = 0; i_ < {R}; ++i_) {{
    const long long row = mlp_r0_ + i_;
    const long long mlp_qq_ = __shfl_sync(0xffffffffu, mlp_ql_, i_);
#pragma unroll
    for (int c_ = 0; c_ < {P}; ++c_) {{
      const int ch_ = mlp_lane_ + 32 * c_;
      mlp_x_[i_][c_] = ({vrow} && ch_ < {C}) ? reinterpret_cast<const uint4*>({G} + ({at_}))[ch_] : make_uint4(0u, 0u, 0u, 0u);
    }}
  }}
#pragma unroll
  for (int i_ = 0; i_ < {R}; ++i_) {{
    const long long row = mlp_r0_ + i_;
    if (row < {rows_}LL) {{
#pragma unroll
      for (int c_ = 0; c_ < {P}; ++c_) {{
        const int ch_ = mlp_lane_ + 32 * c_;
        if (ch_ < {C}) {{ const uint4 x_ = mlp_x_[i_][c_];
          reinterpret_cast<uint4*>({OUT} + row * {W}LL)[ch_] = make_uint4({fix('x_.x')}, {fix('x_.y')}, {fix('x_.z')}, {fix('x_.w')}); }}
      }}
    }}
  }}
}}
if (mlp_bad_) __trap();
}}"""
    return body, dict(R=R, K=1, gather=True, chunks=C, rows=rows_)


def _rows_code(code, aligned16=None):
    m = _CANON.match(code.strip())
    if not m:
        return None, "not canonical"
    g = m.groupdict()
    U, K, total = int(g["units"]), int(g["k"]), int(g["total"])
    if total % U:
        return None, "partial rows"
    rows_ = total // U
    if K == 1:
        out, why = _gather_code(g, U, rows_, aligned16)
        if out is not None:
            return out, why
    vt, load, add, store = g["vt"], g["load"], g["add"], g["store"]
    if vt != "float4" or "v_[s_]" not in load or "[col]" not in load:
        return None, "not a vector load"
    if add.strip() != "latch_pgrad::add4(acc_, v_[s_]);":
        return None, "unknown add"
    if re.search(r"\bacc_\b", store) is None or "[col]" not in store:
        return None, "unknown store"
    ops = len(re.findall(r"reinterpret_cast<const (?:uint2|float4|uint4)\*>", load))
    if ops < 1:
        return None, "no row loads"
    J = -(-U // 32)
    B = min(K, ROW_BATCH)
    R = max(1, ROW_LOADS // (B * ops))
    R = max(1, min(R, 200 // (J * (B + 1) * 4)))     # float4 accumulators + batch registers stay below ~200
    nt = f"((long long)gridDim.x * blockDim.x) >> 5"
    load_ = load.replace("q_[s_]", "mlp_q_[i_][s_]").replace("v_[s_]", "mlp_v_[i_][s_ - mlp_b0_][j_]")
    load_ = re.sub(r"\[col\]", "[mlp_c_]", load_)
    store_ = re.sub(r"\bacc_\b", "mlp_acc_[i_][j_]", store)
    store_ = re.sub(r"\[col\]", "[mlp_c_]", store_)
    sort_ = g["sort"].replace("q_[", "mlp_q_[i_][")
    idx_ = g["idx"].replace("q_[s_]", "mlp_mine_")
    vrow = f"(row < {rows_}LL && row != {g['pad']}LL)"
    body = f"""{{  // Wide-issue warp rows: {R} rows per warp round, {B} occurrences per load batch, {J} column units per lane
bool mlp_bad_ = false;
const int mlp_lane_ = threadIdx.x & 31;
for (long long mlp_r0_ = (((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * {R}LL; mlp_r0_ < {rows_}LL;
     mlp_r0_ += ({nt}) * {R}LL) {{
  long long mlp_q_[{R}][{K}];
#pragma unroll
  for (int i_ = 0; i_ < {R}; ++i_) {{
    const long long row = mlp_r0_ + i_;
    long long mlp_ql_ = 0LL;
    if (mlp_lane_ < {K} && {vrow}) {{ const int s_ = mlp_lane_; mlp_ql_ = {g['qload']}; }}
#pragma unroll
    for (int s_ = 0; s_ < {K}; ++s_) mlp_q_[i_][s_] = __shfl_sync(0xffffffffu, mlp_ql_, s_);{sort_}
    long long mlp_mine_ = 0LL;
#pragma unroll
    for (int s_ = 0; s_ < {K}; ++s_) {{
      const bool out_ = mlp_q_[i_][s_] < 0 || mlp_q_[i_][s_] >= {g['n']}LL;
      if (out_) {{ mlp_bad_ |= {vrow}; mlp_q_[i_][s_] = 0LL; }}
      if (mlp_lane_ == s_) mlp_mine_ = mlp_q_[i_][s_];
    }}
    if (mlp_lane_ < {K} && {vrow}) mlp_bad_ |= {idx_} != row;
  }}
  {vt} mlp_acc_[{R}][{J}];
#pragma unroll
  for (int i_ = 0; i_ < {R}; ++i_)
#pragma unroll
    for (int j_ = 0; j_ < {J}; ++j_) mlp_acc_[i_][j_] = {g['zero']};
#pragma unroll
  for (int mlp_b0_ = 0; mlp_b0_ < {K}; mlp_b0_ += {B}) {{
    {vt} mlp_v_[{R}][{B}][{J}];
#pragma unroll
    for (int i_ = 0; i_ < {R}; ++i_) {{
      const long long row = mlp_r0_ + i_;
      if ({vrow}) {{
#pragma unroll
        for (int s_ = mlp_b0_; s_ < mlp_b0_ + {B} && s_ < {K}; ++s_) {{
#pragma unroll
          for (int j_ = 0; j_ < {J}; ++j_) {{
            const long long mlp_c_ = mlp_lane_ + 32 * j_ < {U} ? mlp_lane_ + 32 * j_ : 0;
            {load_}
          }}
        }}
      }}
    }}
#pragma unroll
    for (int i_ = 0; i_ < {R}; ++i_) {{
      const long long row = mlp_r0_ + i_;
      if ({vrow}) {{
#pragma unroll
        for (int s_ = mlp_b0_; s_ < mlp_b0_ + {B} && s_ < {K}; ++s_)
#pragma unroll
          for (int j_ = 0; j_ < {J}; ++j_) latch_pgrad::add4(mlp_acc_[i_][j_], mlp_v_[i_][s_ - mlp_b0_][j_]);
      }}
    }}
  }}
#pragma unroll
  for (int i_ = 0; i_ < {R}; ++i_) {{
    const long long row = mlp_r0_ + i_;
    if (row < {rows_}LL) {{
#pragma unroll
      for (int j_ = 0; j_ < {J}; ++j_) {{
        const long long mlp_c_ = mlp_lane_ + 32 * j_;
        if (mlp_c_ < {U}) {{ {store_} }}
      }}
    }}
  }}
}}
if (mlp_bad_) __trap();
}}"""
    return body, dict(R=R, B=B, J=J, K=K, ops=ops, rows=rows_)


# `if (Q[s_] < 0 || Q[s_] >= N || (COL == 0 && IDX[...] != ROW)) __trap();`
_CHECK = re.compile(
    r"if \((q\w*_)\[s_\] < 0 \|\| \1\[s_\] >= (\d+)LL \|\| \((col\w*) == 0 && (v_\w+\[[^\n]*?\]) != (row\w*)\)\) __trap\(\);")


def _defer_code(code):
    found = _CHECK.findall(code)
    if not found:
        return None

    def sub(m):
        q, n, col, idx, row = m.groups()
        return (f"const bool out_ = {q}[s_] < 0 || {q}[s_] >= {n}LL; if (out_) {q}[s_] = 0LL; "
                f"mlp_bad_ |= out_ || ({col} == 0 && {idx} != {row});")
    body = _CHECK.sub(sub, code)
    if "__trap()" in body:
        return None
    return "{\nbool mlp_bad_ = false;\n" + body + "\nif (mlp_bad_) __trap();\n}"


def _aligned16(layouts):
    """BF16 view whose row starts are 16-byte aligned: offset and every non-unit outer stride multiples of 8 elements,
    unit inner stride (arena roots are 64-word aligned)."""
    def ok(name):
        lay = layouts.get(name) if layouts is not None else None
        if lay is None:
            return False
        shape, strides = list(getattr(lay, "shape", ())), list(getattr(lay, "strides", ()))
        if not shape or strides[-1] != 1 or int(getattr(lay, "offset", 1)) % 8:
            return False
        return all(st % 8 == 0 for st, e in zip(strides[:-1], shape[:-1]) if e != 1)
    return ok


def rewrite(phases, layouts=None):
    """Rewrite phase code in place; returns a record {rule: [phase names]}."""
    record = {}
    aligned16 = _aligned16(layouts)
    if not any_enabled():
        return record
    for p in phases:
        if p.get("kind") != "embedding_gradient" or p.get("engine") not in ("permutation_fixed_sum_v1",
                                                                             "permutation_gather_v1"):
            continue
        new, why = _rows_code(p.get("code", ""), aligned16)
        if new is not None:
            p["code"] = new
            p["inflight_rules"] = [*p.get("inflight_rules", []), "rows"]
            p["inflight_rows"] = why
            record.setdefault("rows", []).append(list(p.get("names", p.get("writes", []))))
            continue
        record.setdefault("rows_refused", []).append([list(p.get("names", p.get("writes", []))), why])
        new = _defer_code(p.get("code", ""))
        if new is not None:
            p["code"] = new
            p["inflight_rules"] = [*p.get("inflight_rules", []), "defer"]
            record.setdefault("defer", []).append(list(p.get("names", p.get("writes", []))))
    return record
