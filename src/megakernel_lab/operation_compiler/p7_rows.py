"""P7-A register row engine: chained row statistics and their row members from one load of each row.

The P-OPS3-C row engine (and P6-A's chained form) evaluates every statistic of a phase in its own pass: the leaf
of each statistic reloads the row from memory, a statistic is stored to global memory and re-read by the next
statistic's leaf and by the attached members, and each reduction replays the frozen 256-thread CTA tree (eight
xor butterflies per row). For layer normalization forward (mean, then variance, then normalized outputs) and
backward (two row sums, then the input gradient) this is three to four dependent memory round trips per row
batch, issued by eight warps per SM: the phases are latency bound (P6-A ncu: no eligible warp 86% of cycles).

Rule (operation/dependency/layout based; no model names):

* Admission: the row engines of one phase share (rows, extent), extent is a multiple of 128 and at most 2048,
  every statistic's leaf/members only read (a) element-identity values of the same [rows, extent] index space,
  (b) per-column values (index == column), (c) per-row values, (d) earlier statistics of the group, or (e) any
  other memory location (left as an ordinary load). Classification of an index expression is a numeric proof
  over the whole index space (as P6-A's _row_index_exact), not pattern matching on names.
* Execution: a warp owns a row; lane l owns the four-column vectors 128 j + 4 l .. +3. Element-identity operands
  that no member of the group writes are loaded once per row with 16-byte (FP32) / 8-byte (BF16) vector loads;
  per-column operands once per warp. Each statistic sums its lane's leaves in fixed (j, t) order and finishes with
  a five-step xor butterfly, so every lane holds the statistic in a register; the stored statistic and every
  member are the unchanged per-operation expressions evaluated on the rounded statistic. Member results are
  stored with vector stores. RB rows are processed together for memory-level parallelism.
* Arithmetic: per-element member and leaf expressions are unchanged; only the FP32 summation order of each row
  statistic changes (deterministic, a fixed per-lane order and butterfly instead of the frozen CTA tree). This is
  the FP32 reassociation the frozen selection admits; it is not bitwise to the frozen tree.
"""
from __future__ import annotations

import re

import numpy as np

MAX_EXTENT = 2048
REG_BUDGET = 64          # 32-bit registers of cached row values per lane across the RB rows of one iteration

_CT = {"float32": "float", "bfloat16": "__nv_bfloat16"}
_LOAD = re.compile(r"\bv_\w+\[")


def eligible(rows, extent):
    return extent % 128 == 0 and 128 <= extent <= MAX_EXTENT and rows >= 1 and rows * extent < 2 ** 31


class Classifier:
    """Numeric proof over the whole [rows, extent] index space, in exact 32-bit unsigned arithmetic (numpy uint32
    wraps, divides and reduces like CUDA unsigned int): an index text equals the element / row / column coordinate,
    or an integer subexpression depends on the row only."""

    VARS = {"prow", "pcol", "g", "c", "i", "pr", "o"}

    def __init__(self, rows, extent):
        self.rows, self.extent = rows, extent
        self.cache, self.row_cache = {}, {}
        prow = np.arange(rows, dtype=np.uint32)[:, None]
        pcol = np.arange(extent, dtype=np.uint32)[None, :]
        element = prow * np.uint32(extent) + pcol
        self.env = {"prow": prow, "pcol": pcol, "g": prow, "c": pcol, "i": element, "pr": prow, "o": pcol}
        zero = np.zeros_like(pcol)
        self.env0 = {"prow": prow, "pcol": zero, "g": prow, "c": zero, "i": prow * np.uint32(extent), "pr": prow,
                     "o": zero}
        self.targets = {"element": element, "row": prow, "col": pcol}

    def _value(self, text, env):
        expr = text.replace("(unsigned)", "")
        if "long" in expr or "?" in expr or "<" in expr or ">" in expr or "=" in expr or "." in expr:
            return None
        expr = re.sub(r"(\d+)(u|U)\b", r"\1", expr)
        if re.search(r"[^\sA-Za-z_0-9()+\-*/%]", expr):
            return None
        names = set(re.findall(r"[A-Za-z_]\w*", expr))
        if not names or not names <= self.VARS:
            return None
        if re.search(r"\b\d{10,}\b", expr):
            return None
        try:
            with np.errstate(all="ignore"):
                value = eval(expr.replace("/", "//"), {"__builtins__": {}}, dict(env))
            value = np.asarray(value)
            if value.dtype != np.uint32:
                return None
            return np.broadcast_to(value, (self.rows, self.extent))
        except Exception:
            return None

    def __call__(self, text):
        if text not in self.cache:
            result, value = None, self._value(text, self.env)
            if value is not None:
                for kind, target in self.targets.items():
                    if np.array_equal(value, np.broadcast_to(target, (self.rows, self.extent))):
                        result = kind
                        break
            self.cache[text] = result
        return self.cache[text]

    def row_only(self, text):
        """True when the integer expression is independent of the column everywhere (equal to its value at
        column 0)."""
        if text not in self.row_cache:
            a, b = self._value(text, self.env), self._value(text, self.env0)
            self.row_cache[text] = a is not None and b is not None and np.array_equal(a, b)
        return self.row_cache[text]


_COL_VARS = re.compile(r"\b(pcol|c|o)\b")


def simplify(text, classify, extent):
    """Rewrite maximal integer subexpressions proven equal to the element/row/column coordinate as i / g / pcol,
    and row-only ones as the same expression at column 0 (identical value, column-invariant text that the CUDA
    compiler evaluates once per row). Arithmetic on data is untouched."""
    out, k, n = [], 0, len(text)
    while k < n:
        if text[k] == "(":
            depth, m = 0, k
            while m < n:
                depth += text[m] == "("
                depth -= text[m] == ")"
                if depth == 0:
                    break
                m += 1
            sub = text[k:m + 1]
            if m < n and classify._value(sub, classify.env) is not None:
                kind = classify(sub)
                if kind == "element":
                    out.append("i")
                elif kind == "row":
                    out.append("g")
                elif kind == "col":
                    out.append("pcol")
                elif classify.row_only(sub) and _COL_VARS.search(sub) or (classify.row_only(sub) and
                                                                          re.search(r"\bi\b", sub)):
                    zeroed = _COL_VARS.sub("0u", sub)
                    zeroed = re.sub(r"\bi\b", f"(g * {extent}u)", zeroed)
                    out.append(zeroed)
                else:
                    out.append(sub if "(" not in sub[1:] else "(" + simplify(sub[1:-1], classify, extent) + ")")
                k = m + 1
                continue
        out.append(text[k])
        k += 1
    return "".join(out)


def _loads(text):
    """(start, end, ident, index text) of every v_<ident>[...] load with a balanced, load-free index."""
    out = []
    for m in _LOAD.finditer(text):
        depth, k = 0, m.end() - 1
        while k < len(text):
            depth += text[k] == "["
            depth -= text[k] == "]"
            if depth == 0:
                break
            k += 1
        index = text[m.end():k]
        if "[" in index or k >= len(text):
            continue
        out.append((m.start(), k + 1, m.group(0)[:-1], index))
    return out


class Group:
    """Register naming for one admitted row group."""

    def __init__(self, rows, extent, ident_dtype, written, stats):
        self.rows, self.extent = rows, extent
        self.ident_dtype = ident_dtype
        self.written = set(written)           # identifiers stored by members/statistics of the group
        self.stats = dict(stats)              # identifier of a statistic -> register base name (indexed by [r])
        self.classify = Classifier(rows, extent)
        self.element, self.column = {}, {}   # identifier -> register base

    def regify(self, text):
        pieces, last = [], 0
        for start, end, ident, index in _loads(text):
            kind = self.classify(index)
            dtype = self.ident_dtype.get(ident)
            repl = None
            if ident in self.stats and kind == "row":
                repl = f"{self.stats[ident]}[r]"
            elif dtype in _CT and ident not in self.written and ident not in self.stats:
                if kind == "element":
                    repl = f"{self.element.setdefault(ident, 'x' + ident[1:])}[r][e]"
                elif kind == "col":
                    repl = f"{self.column.setdefault(ident, 'k' + ident[1:])}[e]"
            if repl is not None:
                pieces.append(text[last:start])
                pieces.append(repl)
                last = end
        pieces.append(text[last:])
        return simplify("".join(pieces), self.classify, self.extent)

    def row_registers(self):
        return sum((4 if self.ident_dtype[x] == "float32" else 2) for x in self.element) * (self.extent // 128)


def code(group, entries, *, rb, lo="0u", tiles=None, columns=(), virtuals=(), prefetch=False):
    """entries: [(stat_ident, stat_ctype, leaf_text, store_rhs_text, members)] in dependency order; members:
    [(name_ident, ctype, expression_text, stored, per_row)] (texts already regified, variables g/i/c/e/r/
    prow/pcol and s_<k>[r] statistics). Returns the engine's CUDA text.

    tiles = (T, RT): slab-major ownership instead of grid-strided rows - CTA owns tiles t of RT rows, warp w the
    rows r0 + w + 8 k (k ascending). columns = [(leaf_texts, part_idents)] are folded column-partial reductions
    over the same rows (P-OPS4-A vector column items): each lane accumulates its 4-column vectors over its rows
    in k order and the eight warps combine in warp order into part[t * extent + o], exactly the P-OPS4-A vector
    column body's order (bitwise identical partials).

    prefetch (grid-strided rows only): the element operands of a warp's next RB rows are loaded into a second
    register set while the current rows are computed (one row batch of loads in flight behind the arithmetic,
    without unrolling the arithmetic). Element operands are never written by the group, so the early loads read
    the same values; results are bitwise identical."""
    rows, extent = group.rows, group.extent
    nj = extent // 128
    ne = 4 * nj
    lines = [f"{{ // P7-A register row engine: {len(entries)} statistic(s) of {rows} rows x {extent} "
             f"(warp per row, lane 4-column vectors, {rb} rows per iteration)",
             f"  const unsigned latch_lo = {lo};",
             "  if (blockIdx.x >= latch_lo) {",
             "  const unsigned lane = threadIdx.x & 31u;",
             "  const unsigned p7_w = (blockIdx.x - latch_lo) * (blockDim.x >> 5) + (threadIdx.x >> 5);",
             "  const unsigned p7_nw = (gridDim.x - latch_lo) * (blockDim.x >> 5);"]
    for ident, reg in group.column.items():
        ct = _CT[group.ident_dtype[ident]]
        lines.append(f"  {ct} {reg}[{ne}];")
        lines.append(f"#pragma unroll\n  for (unsigned j = 0; j < {nj}u; ++j) {{ {ct} t4[4]; "
                     f"latch_ld4({ident} + 128u * j + 4u * lane, t4); "
                     f"{reg}[4 * j] = t4[0]; {reg}[4 * j + 1] = t4[1]; {reg}[4 * j + 2] = t4[2]; {reg}[4 * j + 3] = t4[3]; }}")
    ncol = sum(len(leaves) for leaves, _ in columns)
    if tiles:
        nt, rt = tiles
        kr = -(-rt // 8)
        for q in range(ncol):
            lines.append(f"  float p7_c{q}[{ne}];")
        lines.append(f"  for (unsigned p7_t = blockIdx.x - latch_lo; p7_t < {nt}u; p7_t += gridDim.x - latch_lo) {{")
        lines.append(f"    const unsigned p7_r0 = p7_t * {rt}u, p7_r1 = min(p7_r0 + {rt}u, {rows}u);")
        for q in range(ncol):
            lines.append(f"#pragma unroll\n    for (unsigned e = 0; e < {ne}u; ++e) p7_c{q}[e] = 0.0f;")
        k_loop = f"  for (unsigned p7_k = 0; p7_k < {kr}u; p7_k += {rb}u) {{"
        row_of = lambda r: f"p7_r0 + (threadIdx.x >> 5) + 8u * (p7_k + {r})"
        row_at = lambda k_, r: f"p7_r0 + (threadIdx.x >> 5) + 8u * ({k_} + {r})"
        guard = "g < p7_r1"
        first0, next0 = "0u", f"p7_k + {rb}u"
    else:
        k_loop = f"  for (unsigned p7_b = p7_w; p7_b < {rows}u; p7_b += p7_nw * {rb}u) {{"
        row_of = lambda r: f"p7_b + {r} * p7_nw"
        row_at = lambda b_, r: f"{b_} + {r} * p7_nw"
        guard = f"g < {rows}u"
        first0, next0 = "p7_w", f"p7_b + {rb}u * p7_nw"
    prefetch = bool(prefetch and group.element)

    def load_rows(suffix, first):
        out = [f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
               f"      const unsigned g = {row_at(first, 'r')};\n      if ({guard}) {{"]
        for ident, reg in group.element.items():
            ct = _CT[group.ident_dtype[ident]]
            out.append(f"#pragma unroll\n        for (unsigned j = 0; j < {nj}u; ++j) {{ {ct} t4[4]; "
                       f"latch_ld4({ident} + g * {extent}u + 128u * j + 4u * lane, t4); "
                       f"{reg}{suffix}[r][4 * j] = t4[0]; {reg}{suffix}[r][4 * j + 1] = t4[1]; "
                       f"{reg}{suffix}[r][4 * j + 2] = t4[2]; {reg}{suffix}[r][4 * j + 3] = t4[3]; }}")
        out.append("      }\n    }")
        return out
    if prefetch:
        for ident, reg in group.element.items():
            lines.append(f"  {_CT[group.ident_dtype[ident]]} {reg}[{rb}][{ne}];")
        lines.extend(load_rows("", first0))
    lines.append(k_loop)
    if prefetch:
        for ident, reg in group.element.items():
            lines.append(f"    {_CT[group.ident_dtype[ident]]} {reg}_n[{rb}][{ne}];")
        lines.extend(load_rows("_n", next0))
    else:
        for ident, reg in group.element.items():
            ct = _CT[group.ident_dtype[ident]]
            lines.append(f"    {ct} {reg}[{rb}][{ne}];")
    if group.element and not prefetch:
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"      const unsigned g = {row_of('r')};\n      if ({guard}) {{")
        for ident, reg in group.element.items():
            ct = _CT[group.ident_dtype[ident]]
            lines.append(f"#pragma unroll\n        for (unsigned j = 0; j < {nj}u; ++j) {{ {ct} t4[4]; "
                         f"latch_ld4({ident} + g * {extent}u + 128u * j + 4u * lane, t4); "
                         f"{reg}[r][4 * j] = t4[0]; {reg}[r][4 * j + 1] = t4[1]; {reg}[r][4 * j + 2] = t4[2]; "
                         f"{reg}[r][4 * j + 3] = t4[3]; }}")
        lines.append("      }\n    }")
    elem_head = (f"#pragma unroll\n        for (unsigned j = 0; j < {nj}u; ++j) {{\n#pragma unroll\n"
                 f"          for (unsigned t = 0; t < 4u; ++t) {{\n"
                 f"            const unsigned e = 4u * j + t, pcol = 128u * j + 4u * lane + t, c = pcol, prow = g;\n"
                 f"            const unsigned i = g * {extent}u + pcol, pr = g, o = pcol;\n"
                 f"            (void)i; (void)c; (void)prow; (void)pr; (void)o;\n")
    for reg, text in virtuals:
        # Inline producer read by several statistics/members: evaluated once per element (identical expression).
        lines.append(f"    float {reg}[{rb}][{ne}];")
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"      const unsigned g = {row_of('r')};\n      if ({guard}) {{\n" + elem_head
                     + f"            {reg}[r][e] = {text};\n          }}\n        }}\n      }}\n    }}")
    if columns:
        # Folded column partials: rows in k order per lane (r ascending within the batch), as the P-OPS4-A body.
        flat = [leaf for leaves, _ in columns for leaf in leaves]
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"      const unsigned g = {row_of('r')};\n      if ({guard}) {{\n" + elem_head
                     + "".join(f"            p7_c{q}[e] += {leaf};\n" for q, leaf in enumerate(flat))
                     + "          }\n        }\n      }\n    }")
    for stat, ct, leaf, rhs, members in entries:
        lines.append(f"    float p7_a_{stat}[{rb}];")
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"      const unsigned g = {row_of('r')}; float acc = 0.0f;\n      if ({guard}) {{\n"
                     + elem_head + f"            acc += {leaf};\n          }}\n        }}\n      }}\n"
                     f"      p7_a_{stat}[r] = acc;\n    }}")
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"#pragma unroll\n      for (int d = 16; d > 0; d >>= 1) "
                     f"p7_a_{stat}[r] += __shfl_xor_sync(0xffffffffu, p7_a_{stat}[r], d);\n    }}")
        lines.append(f"    {ct} s_{stat}[{rb}];")
        lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                     f"      const unsigned g = {row_of('r')}; const float second = p7_a_{stat}[r];\n"
                     f"      s_{stat}[r] = {rhs};\n      if (lane == 0u && {guard}) {stat}[g] = s_{stat}[r];\n    }}")
        if members:
            per_row = [m for m in members if m[4]]
            elem = [m for m in members if not m[4]]
            body = [f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r) {{\n"
                    f"      const unsigned g = {row_of('r')};\n      if ({guard}) {{"]
            for name, mct, expr, stored, _ in per_row:
                body.append(f"        const {mct} q{name[1:]} = {expr};")
                if stored:
                    body.append(f"        if (lane == 0u) {name}[g] = q{name[1:]};")
            for name, mct, _, _, _ in elem:
                body.append(f"        {mct} r{name[1:]}[{ne}];")
            if elem:
                body.append(elem_head + "".join(f"            r{name[1:]}[e] = {expr};\n" for name, mct, expr, _, _ in elem)
                            + "          }\n        }")
                for name, mct, _, stored, _ in elem:
                    if stored:
                        body.append(f"#pragma unroll\n        for (unsigned j = 0; j < {nj}u; ++j) "
                                    f"latch_st4({name} + g * {extent}u + 128u * j + 4u * lane, r{name[1:]}[4 * j], "
                                    f"r{name[1:]}[4 * j + 1], r{name[1:]}[4 * j + 2], r{name[1:]}[4 * j + 3]);")
            body.append("      }\n    }")
            lines.extend(body)
    if prefetch:
        for ident, reg in group.element.items():
            lines.append(f"#pragma unroll\n    for (unsigned r = 0; r < {rb}u; ++r)\n#pragma unroll\n"
                         f"      for (unsigned e = 0; e < {ne}u; ++e) {reg}[r][e] = {reg}_n[r][e];")
    if tiles:
        lines.append("  }")                                   # p7_k
        if columns:
            parts = [part for _, ps in columns for part in ps]
            lines.append("#pragma unroll\n    for (unsigned e = 0; e < %du; ++e) {" % ne)
            for q in range(ncol):
                lines.append(f"      smem[({q}u * {ne}u + e) * 256u + threadIdx.x] = p7_c{q}[e];")
            lines.append("    }\n    __syncthreads();\n    if ((threadIdx.x >> 5) == 0u) {")
            lines.append(f"#pragma unroll\n      for (unsigned e = 0; e < {ne}u; ++e) {{\n"
                         f"        const unsigned o = 128u * (e / 4u) + 4u * lane + (e % 4u);")
            for q, part in enumerate(parts):
                lines.append(f"        {{ float s = smem[({q}u * {ne}u + e) * 256u + lane];\n"
                             f"#pragma unroll\n          for (unsigned w = 1; w < 8u; ++w) s += smem[({q}u * {ne}u + e) * 256u + w * 32u + lane];\n"
                             f"          {part}[p7_t * {extent}u + o] = s; }}")
            lines.append("      }\n    }\n    __syncthreads();")
    lines.append("  }\n  }\n}")
    return "\n".join(lines)


def choose_rb(rows, row_registers, grid_warps):
    per_warp = -(-rows // max(1, grid_warps))
    rb = 1
    while rb * 2 <= 8 and rb * 2 <= per_warp and rb * 2 * max(1, row_registers) <= REG_BUDGET:
        rb *= 2
    return rb
