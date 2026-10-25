"""P-OPS5-E staged dense reductions: bulk-copy row tiles, fused row/column passes.

In the one-CTA-per-SM cooperative program (255 registers, eight warps per SM) the existing reduction engines keep
too few loads in flight to approach HBM bandwidth on long reductions over large dense operands (P-OPS4-A measured
39% DRAM utilisation with no eligible warp in 90% of cycles). This rule moves the operand stream off the register
file: one thread issues ``cp.async.bulk`` copies of whole contiguous row tiles of every dense operand the reduction
leaf reads into a shared-memory ring guarded by mbarriers, and all warps evaluate the leaf from shared memory.

Applies to a reduction whose source is a contiguous ``[rows, extent]`` index space (flat element ``prow*extent +
pcol``) reduced over its trailing part (``row``: one result per row) or its leading part (``column``: one result
per column). A row and a column reduction over the same space whose operands are ready together are evaluated in
ONE pass (``fused``): the operand tiles are read once for both.

* Row results keep the frozen CTA-tree order exactly (lane l accumulates leaves 32w+l+256j, a warp butterfly per w,
  the second-stage butterfly with zero lanes; as P-OPS3-C's engine): bitwise identical to the frozen tree.
* Column results: thread t owns columns t + 256q; each block sums its tiles' rows in row order into registers,
  stores one partial per block; a finalize item sums the block partials in block order (fixed order, different from
  the frozen tree: admitted only through the declared numerical checks).
* Staging is decided at emission, after inlining: a root is staged only if every read of it in the leaves is the
  identity element ``v[prow*extent + pcol]`` of a contiguous allocation; other reads (broadcast vectors, per-row
  statistics) stay ordinary loads. A misaligned base falls back to a cooperative synchronous copy (same values).

Dispatch uses only reduction axes, extents, layouts, dependencies and byte counts.
"""
from __future__ import annotations

import re

GRID_MAX = 256              # partial rows reserved for column partials (>= any cooperative grid on H100)
STAGE_BUDGET = 150 * 1024   # operand ring; 192 KB measured 6% slower whole-update (program shared 194 KB > 165 KB: less L1 for other phases)
MIN_ROWS = 256
MIN_ELEMENTS = 1 << 20  # smaller operands: pipeline fill/sync overhead exceeds the transfer (tiny16 measured 2.7x slower)
MAX_STAGES = 4
THREADS = 256

HELPERS = r"""
#ifndef LATCH_P5E_HELPERS
#define LATCH_P5E_HELPERS
__device__ __forceinline__ unsigned latch_p5e_saddr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void latch_p5e_bar_init(unsigned long long* bar) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(latch_p5e_saddr(bar)) : "memory");
}
__device__ __forceinline__ void latch_p5e_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void latch_p5e_fence_proxy() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void latch_p5e_expect(unsigned long long* bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(latch_p5e_saddr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void latch_p5e_copy(void* dst, const void* src, unsigned bytes, unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(latch_p5e_saddr(dst)), "l"(src), "r"(bytes), "r"(latch_p5e_saddr(bar)) : "memory");
}
__device__ __forceinline__ void latch_p5e_wait(unsigned long long* bar, unsigned parity) {
  asm volatile("{\n .reg .pred p;\n LATCH_P5E_WAIT_%=:\n"
               " mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               " @!p bra LATCH_P5E_WAIT_%=;\n}" :: "r"(latch_p5e_saddr(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void latch_p5e_bar_inval(unsigned long long* bar) {
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(latch_p5e_saddr(bar)) : "memory");
}
#endif
"""

LEAF_MARK = "/*P5E_LEAF_{}*/"
_IDENT_READ = re.compile(r"\b(v_[A-Za-z0-9_]+)\[")


def position(extent):
    return f"(prow * {extent}u + pcol)"


def _balanced(text, start):
    """Index of the ']' closing the '[' at text[start - 1]."""
    depth = 1
    k = start
    while k < len(text):
        c = text[k]
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                return k
        k += 1
    raise ValueError('unbalanced index expression')


def _normalize(index):
    s = index.replace('(unsigned)', '').replace(' ', '')
    while s.startswith('(') and s.endswith(')'):
        depth, ok = 0, True
        for k, c in enumerate(s):
            depth += c == '('
            depth -= c == ')'
            if depth == 0 and k != len(s) - 1:
                ok = False
                break
        if not ok:
            break
        s = s[1:-1]
    return s


def reads(leaves):
    """{ident: [normalized index, ...]} for every v_X[...] read in the leaf texts."""
    out = {}
    for text in leaves:
        for m in _IDENT_READ.finditer(text):
            end = _balanced(text, m.end())
            out.setdefault(m.group(1), []).append(_normalize(text[m.end():end]))
    return out


def stage_rewrite(leaves, staged):
    """Replace identity reads v_X[P] of staged identifiers with shared-tile reads."""
    out = []
    for text in leaves:
        pieces, k = [], 0
        for m in _IDENT_READ.finditer(text):
            if m.start() < k:
                continue
            ident = m.group(1)
            if ident not in staged:
                continue
            end = _balanced(text, m.end())
            pieces.append(text[k:m.start()])
            pieces.append(f"{staged[ident]}[latch_lr * {staged['__E__']}u + pcol]")
            k = end + 1
        pieces.append(text[k:])
        out.append(''.join(pieces))
    return out


def simplify(text, extent):
    """Rewrite W / E -> prow and W % E -> pcol where W is the canonical position (possibly wrapped in parentheses
    and (unsigned) casts): every leaf element has 0 <= pcol < E, so both identities hold exactly."""
    core = f"(prow * {extent}u + pcol)"
    out, k = [], 0
    while True:
        at = text.find(core, k)
        if at < 0:
            out.append(text[k:])
            return ''.join(out)
        a, b = at, at + len(core)
        while True:                       # grow over enclosing (...) and (unsigned) casts
            if a >= 10 and text[a - 10:a] == '(unsigned)':
                a -= 10
                continue
            if a >= 1 and b < len(text) and text[a - 1] == '(' and text[b] == ')':
                a, b = a - 1, b + 1
                continue
            break
        rest = text[b:]
        if rest.startswith(f" / {extent}u"):
            out.append(text[k:a] + "prow")
            k = b + len(f" / {extent}u")
        elif rest.startswith(f" % {extent}u"):
            out.append(text[k:a] + "pcol")
            k = b + len(f" % {extent}u")
        else:
            out.append(text[k:b])
            k = b


_CONST_READ = re.compile(r"\b(v_[A-Za-z0-9_]+)\[0u\]")


def hoist_constants(leaves, ctypes):
    """Loop-invariant v_X[0u] reads -> registers loaded once (the engine never writes its leaf inputs)."""
    names, out = {}, []
    for text in leaves:
        out.append(_CONST_READ.sub(lambda m: names.setdefault(m.group(1), f"latch_k{len(names)}"), text))
    decls = [f"const {ctypes.get(ident, 'float')} {reg} = {ident}[0u];" for ident, reg in names.items()]
    return out, decls


def plan(rows, extent, row_bytes, budget=STAGE_BUDGET, prefer=None):
    """Tile rows and ring depth for `row_bytes` staged bytes per row.

    prefer None: largest tile, then deepest ring; 'stages': deepest ring (>= 3 when possible), then largest tile;
    {'tile': t, 'stages': s}: forced when it fits."""
    if row_bytes == 0:
        return {'tile': 8, 'stages': 0}
    if isinstance(prefer, dict):
        if prefer['tile'] * row_bytes * prefer['stages'] <= budget:
            return {'tile': prefer['tile'], 'stages': prefer['stages']}
        prefer = None
    if isinstance(prefer, dict) and 'budget' in prefer:
        budget = prefer['budget']
    if prefer in ('flight', 'flight3'):
        # maximise bytes in flight (prefetch distance x tile bytes), then the larger tile; fused pipelines
        # ('flight3') keep >= 3 stages (prefetch distance S-2), row-only pipelines prefetch S-1 tiles.
        best = None
        for stages in range(2, MAX_STAGES + 1):
            if prefer == 'flight3' and stages < 3:
                continue
            ahead = stages - (2 if prefer == 'flight3' else 1)
            # row-only passes give a warp to each row of the tile: keep tiles balanced over the 8 warps
            for tile in (range(1, 17) if prefer == 'flight3' else (1, 2, 4, 8, 16)):
                if tile * row_bytes * stages > budget or rows < tile * 4:
                    continue
                key = (ahead * tile * row_bytes, tile)
                if best is None or key > best[0]:
                    best = (key, {'tile': tile, 'stages': stages})
        if best is not None:
            return best[1]
        prefer = None
    if prefer == 'stages':
        order = [(t, st) for st in range(MAX_STAGES, 1, -1) for t in (16, 8, 4, 2, 1)]
    elif prefer == 'three':     # fused single-barrier pipelines need >= 3 stages; then the largest tile
        order = [(t, 3) for t in (16, 8, 4, 2, 1)] + [(t, st) for t in (16, 8, 4, 2, 1) for st in (4, 2)]
    else:
        order = [(t, st) for t in (16, 8, 4, 2, 1) for st in range(MAX_STAGES, 1, -1)]
    for tile, stages in order:
        if tile * row_bytes * stages <= budget and rows >= tile * 4 and (prefer != 'stages' or tile >= 4 or stages <= 2):
            return {'tile': tile, 'stages': stages}
    return None


def emit(*, rows, extent, row_out, row_leaf, col_part, col_leaf, rounded, scale, staged, idents_bytes,
         tile, stages, rotate=0, ctypes=None):
    """Engine source. staged: [(ident, ctype, elem_bytes)] identity-read roots (shared ring members)."""
    E = extent
    q = -(-E // THREADS)
    stage_bytes = sum(tile * E * b for _, _, b in staged)
    offsets, off = {}, 0
    for ident, ctype, b in staged:
        offsets[ident] = off
        off += tile * E * b
    ring = stages * stage_bytes
    bar_off = -(-ring // 16) * 16
    names = {ident: f"latch_s{k}" for k, (ident, _, _) in enumerate(staged)}
    names['__E__'] = E
    leaves = [simplify(x, E) for x in (row_leaf, col_leaf) if x is not None]
    leaves = stage_rewrite(leaves, names)
    leaves, hoisted = hoist_constants(leaves, ctypes or {})
    row_leaf2 = leaves[0] if row_leaf is not None else None
    col_leaf2 = leaves[-1] if col_leaf is not None else None
    lifted = [x for x in (row_leaf2, col_leaf2) if x is not None]
    lifted, c_decl, r_decl, k_decl = lift(lifted, allow_col=col_leaf2 is not None)
    row_leaf2 = lifted[0] if row_leaf2 is not None else None
    col_leaf2 = lifted[-1] if col_leaf2 is not None else None
    e_decl = []
    if row_leaf2 is not None and col_leaf2 is not None:
        row_leaf2, col_leaf2, e_decl = share(row_leaf2, col_leaf2)
    qn = -(-E // THREADS)
    hoisted += [f"const auto {reg} = {span};" for reg, span in c_decl]
    for reg, span in k_decl:
        hoisted.append(f"float {reg}[{qn}];")
        hoisted.append(f"#pragma unroll\n  for (unsigned k = 0; k < {qn}u; ++k) {{ const unsigned pcol = threadIdx.x + 256u * k; "
                       f"{reg}[k] = pcol < {E}u ? (float)({span}) : 0.0f; }}")
    row_hoist = " ".join(f"const auto {reg} = {span};" for reg, span in r_decl)
    elem_share = " ".join(f"const auto {reg} = {span};" for reg, span in e_decl)
    hoist_code = "\n  ".join(hoisted)
    pr_off = bar_off + 16 * stages
    # Fused passes keep one block barrier per tile: the barrier after the partials also frees the stage read two
    # tiles earlier (prefetch distance S-2, S >= 3; attached members read the current stage after it).
    single = col_leaf is not None and row_leaf is not None and stages >= 3
    ahead = stages - 2 if single else stages - 1
    end_sync = "" if single else "__syncthreads();   // every warp is done with this stage before it is refilled"
    ntiles = -(-rows // tile)
    decl_stage = "\n".join(
        f"    {ctype}* latch_s{k} = reinterpret_cast<{ctype}*>(latch_sb + latch_st * {stage_bytes}u + {offsets[ident]}u);"
        for k, (ident, ctype, _) in enumerate(staged))
    issue_copies = "\n".join(
        f"        latch_p5e_copy(latch_dst + {offsets[ident]}u, {ident} + (size_t)latch_t * {tile * E}u, latch_nr * {E * b}u, latch_bar + latch_is);"
        for ident, ctype, b in staged)
    sync_copies = "\n".join(
        f"        {{ const unsigned latch_n = latch_nr * {E * b // 4}u; const unsigned* latch_g = reinterpret_cast<const unsigned*>({ident} + (size_t)latch_t * {tile * E}u);"
        f" unsigned* latch_d = reinterpret_cast<unsigned*>(latch_dst + {offsets[ident]}u);"
        f" for (unsigned latch_k = threadIdx.x; latch_k < latch_n; latch_k += {THREADS}u) latch_d[latch_k] = latch_g[latch_k]; }}"
        for ident, ctype, b in staged)
    aligned = " && ".join(f"((reinterpret_cast<size_t>({ident}) & 15u) == 0u)" for ident, _, _ in staged) or "true"
    q = -(-E // THREADS)
    row_part = ""
    col_decl = col_init = col_part_code = col_store = ""
    members = f"""
        {{ const unsigned g = prow; constexpr unsigned latch_w0 = 0u, latch_nw = 8u; (void)latch_w0; (void)latch_nw;
/*ROW_MEMBERS*/
        }}"""
    if col_leaf2 is not None:
        col_decl = f"float latch_c[{q}];"
        col_init = f"#pragma unroll\n    for (unsigned k = 0; k < {q}u; ++k) latch_c[k] = 0.0f;"
        col_store = f"""
#pragma unroll
    for (unsigned k = 0; k < {q}u; ++k) {{
      const unsigned pcol = threadIdx.x + 256u * k;
      if (pcol < {E}u) {col_part}[blockIdx.x * {E}u + pcol] = latch_c[k];
    }}"""
    if E % 32 == 0:
        # lane < 32 and E % 32 == 0: element 32w + lane + 256j is in range iff 32w + 256j < E (static)
        slots = []
        for w in range(8):
            terms = [j for j in range(-(-E // 256)) if 32 * w + 256 * j < E]
            body = " ".join(f"{{ const unsigned pcol = {32 * w}u + lane + {256 * j}u; acc += {row_leaf2}; }}"
                            for j in terms) if row_leaf2 is not None else ""
            slots.append(f"        {{ float acc = 0.0f; {body} latch_v[{w}] = s101h::warp_sum(acc); }}")
        row_slots = "\n".join(slots)
    else:
        row_slots = f"""#pragma unroll
        for (unsigned w = 0; w < 8u; ++w) {{
          float acc = 0.0f;
#pragma unroll
          for (unsigned j = 0; j < {-(-E // 256)}u; ++j) {{
            const unsigned pcol = 32u * w + lane + 256u * j;
            if (pcol < {E}u) acc += {row_leaf2};
          }}
          latch_v[w] = s101h::warp_sum(acc);
        }}"""
    if col_leaf2 is None:
        # Row results only: a warp per row (lane l, slots w = 0..7, elements 32w + l + 256j in j order), eight
        # independent per-slot butterflies, the ordered second stage: the frozen CTA tree exactly.
        row_part = f"""
      for (unsigned latch_lr = threadIdx.x >> 5; latch_lr < latch_nr; latch_lr += 8u) {{
        const unsigned prow = latch_r0 + latch_lr;
        {row_hoist}
        float latch_v[8];
{row_slots}
        float second = 0.0f;
#pragma unroll
        for (unsigned w = 0; w < 8u; ++w) if (lane == w) second = latch_v[w];
        second = s101h::warp_sum(second);
        if (lane == 0u) {row_out}[prow] = {rounded('second' + scale)};{members}
      }}"""
    else:
        # Column results (and row results of the same space): one thread<->column mapping. Thread t = 32w + l
        # evaluates elements t + 256k in k order -- for rows this is the frozen tree's per-thread accumulation;
        # the per-warp butterflies and the ordered second stage complete it exactly. The tile's rows are
        # unrolled so their butterflies and leaf chains overlap.
        ra_decl = f"float latch_ra[{tile}];" if row_leaf2 is not None else ""
        ra_init = "latch_ra[latch_lr] = 0.0f;" if row_leaf2 is not None else ""
        inner = []
        if row_leaf2 is not None:
            inner.append(f"latch_ra[latch_lr] += {row_leaf2};")
        inner.append(f"latch_c[k] += {col_leaf2};")
        second = ""
        if row_leaf2 is not None:
            second = f"""
#pragma unroll
      for (unsigned latch_lr = 0; latch_lr < {tile}u; ++latch_lr) latch_ra[latch_lr] = s101h::warp_sum(latch_ra[latch_lr]);
      if (lane == 0u) {{
#pragma unroll
        for (unsigned latch_lr = 0; latch_lr < {tile}u; ++latch_lr) latch_pr[latch_pb + latch_lr * 8u + (threadIdx.x >> 5)] = latch_ra[latch_lr];
      }}"""
        full_k = E // 256
        elems = []
        for k in range(q):
            stmt = f"{{ constexpr unsigned k = {k}u; const unsigned pcol = threadIdx.x + {256 * k}u; {elem_share} {' '.join(inner)} }}"
            elems.append(stmt if k < full_k else f"if (threadIdx.x < {E - 256 * k}u) {stmt}")
        elem_code = "\n            ".join(elems)

        def rows_loop(guard):
            return f"""
#pragma unroll
      for (unsigned latch_lr = 0; latch_lr < {tile}u; ++latch_lr) {{
        const unsigned prow = latch_r0 + latch_lr;
        {ra_init}
        {'if (latch_lr < latch_nr) {' if guard else '{'}
          {row_hoist}
            {elem_code}
        }}
      }}"""
        col_part_code = f"""
      {ra_decl}
      if (latch_nr == {tile}u) {{ {rows_loop(False)}
      }} else {{ {rows_loop(True)}
      }}{second}"""
        if row_leaf2 is not None:
            row_part = f"""
      __syncthreads();
      for (unsigned latch_lr = threadIdx.x >> 5; latch_lr < latch_nr; latch_lr += 8u) {{
        const unsigned prow = latch_r0 + latch_lr;
        float second = lane < 8u ? latch_pr[latch_pb + latch_lr * 8u + lane] : 0.0f;
        second = s101h::warp_sum(second);
        if (lane == 0u) {row_out}[prow] = {rounded('second' + scale)};{members}
      }}"""
    if stages == 0:
        # Nothing to stage: plain tile loop over global operands.
        body = f"""{{ // P-OPS5-E dense reduction (no staged operand): {rows} rows x {E}
  const unsigned lane = threadIdx.x & 31u;
  float* latch_pr = smem;
  {hoist_code}
  __syncthreads();
  {col_decl}
  {col_init}
  for (unsigned latch_t = blockIdx.x; latch_t < {ntiles}u; latch_t += gridDim.x) {{
    const unsigned latch_r0 = latch_t * {tile}u;
    const unsigned latch_nr = min({tile}u, {rows}u - latch_r0);
    const unsigned latch_pb = 0u;
    {col_part_code}
    {row_part}
    __syncthreads();
  }}
  {col_store}
}}"""
        return body, 4 * tile * 8
    body = f"""{{ // P-OPS5-E staged dense reduction: {rows} rows x {E}, tile {tile} rows, {stages}-stage bulk-copy ring, {len(staged)} staged operand(s)
  unsigned char* latch_sb = reinterpret_cast<unsigned char*>(smem);
  unsigned long long* latch_bar = reinterpret_cast<unsigned long long*>(latch_sb + {bar_off}u);
  float* latch_pr = reinterpret_cast<float*>(latch_sb + {pr_off}u);
  const unsigned lane = threadIdx.x & 31u;
  const bool latch_bulk = {aligned};
  {hoist_code}
  __syncthreads();   // previous users of shared memory are done
  if (threadIdx.x == 0) {{
    for (unsigned s = 0; s < {stages}u; ++s) latch_p5e_bar_init(latch_bar + s);
    latch_p5e_fence_init();
  }}
  __syncthreads();
  {col_decl}
  {col_init}
  const unsigned latch_mine = blockIdx.x < {ntiles}u ? ({ntiles}u - 1u - blockIdx.x) / gridDim.x + 1u : 0u;
  // prologue: issue the first stages
  if (latch_bulk && threadIdx.x == 0) {{
    for (unsigned latch_i = 0; latch_i < {ahead}u && latch_i < latch_mine; ++latch_i) {{
      const unsigned latch_t = blockIdx.x + latch_i * gridDim.x, latch_is = latch_i % {stages}u;
      const unsigned latch_nr = min({tile}u, {rows}u - latch_t * {tile}u);
      unsigned char* latch_dst = latch_sb + latch_is * {stage_bytes}u;
      latch_p5e_expect(latch_bar + latch_is, latch_nr * {stage_bytes // tile}u);
{issue_copies}
    }}
  }}
  for (unsigned latch_i = 0; latch_i < latch_mine; ++latch_i) {{
    const unsigned latch_t = blockIdx.x + latch_i * gridDim.x;
    const unsigned latch_st = latch_i % {stages}u;
    const unsigned latch_pb = (latch_i & 1u) * {tile * 8}u;
    const unsigned latch_r0 = latch_t * {tile}u;
    const unsigned latch_nr = min({tile}u, {rows}u - latch_r0);
    if (latch_bulk) {{
      if (threadIdx.x == 0 && latch_i + {ahead}u < latch_mine) {{
        const unsigned latch_j = latch_i + {ahead}u;
        const unsigned latch_t = blockIdx.x + latch_j * gridDim.x, latch_is = latch_j % {stages}u;
        const unsigned latch_nr = min({tile}u, {rows}u - latch_t * {tile}u);
        unsigned char* latch_dst = latch_sb + latch_is * {stage_bytes}u;
        latch_p5e_fence_proxy();
        latch_p5e_expect(latch_bar + latch_is, latch_nr * {stage_bytes // tile}u);
{issue_copies}
      }}
      latch_p5e_wait(latch_bar + latch_st, (latch_i / {stages}u) & 1u);
    }} else {{
      unsigned char* latch_dst = latch_sb + latch_st * {stage_bytes}u;
{sync_copies}
      __syncthreads();
    }}
    {{
{decl_stage}
      {col_part_code}
      {row_part}
    }}
    {end_sync}
  }}
  {col_store}
  if (threadIdx.x == 0) {{
    for (unsigned s = 0; s < {stages}u; ++s) latch_p5e_bar_inval(latch_bar + s);
  }}
  __syncthreads();
}}"""
    return body, pr_off + 2 * 4 * tile * 8


FIN_BATCH = 0   # P6-E lowering option p6e_fin_batch: block partials loaded in batches of this many (same order)


def finalize(*, part, out, extent, rounded, scale):
    if FIN_BATCH > 1:
        # P6-E: the same block-order sum; each batch of FIN_BATCH partials is loaded before it is added, so the
        # loads overlap instead of forming one dependent chain of gridDim L2 round trips (bitwise identical).
        B = int(FIN_BATCH)
        return f"""{{ // P-OPS5-E column finalize: {extent} results, block partials in block order (P6-E batches of {B})
  for (unsigned o = blockIdx.x * blockDim.x + threadIdx.x; o < {extent}u; o += gridDim.x * blockDim.x) {{
    float s = {part}[o];
    unsigned b = 1;
    for (; b + {B}u <= gridDim.x; b += {B}u) {{
      float t[{B}];
#pragma unroll
      for (unsigned u = 0; u < {B}u; ++u) t[u] = {part}[(b + u) * {extent}u + o];
#pragma unroll
      for (unsigned u = 0; u < {B}u; ++u) s += t[u];
    }}
    for (; b < gridDim.x; ++b) s += {part}[b * {extent}u + o];
    {out}[o] = {rounded('s' + scale)};
  }}
}}"""
    return f"""{{ // P-OPS5-E column finalize: {extent} results, block partials in block order
  for (unsigned o = blockIdx.x * blockDim.x + threadIdx.x; o < {extent}u; o += gridDim.x * blockDim.x) {{
    float s = {part}[o];
    for (unsigned b = 1; b < gridDim.x; ++b) s += {part}[b * {extent}u + o];
    {out}[o] = {rounded('s' + scale)};
  }}
}}"""


def full_finalize(*, part, out, rows, rounded, scale, block=0):
    """Scalar from per-row partials on block `block % gridDim`: 256 threads sum rows t, t+256, ... in order,
    warp butterflies, then the eight warp sums in warp order (fixed order)."""
    return f"""{{ // P-OPS5-E full finalize: {rows} row partials
  if (blockIdx.x == {block}u % gridDim.x) {{
    float s = 0.0f;
#pragma unroll 4
    for (unsigned r = threadIdx.x; r < {rows}u; r += 256u) s += {part}[r];
    s = s101h::warp_sum(s);
    __syncthreads();
    if ((threadIdx.x & 31u) == 0u) smem[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0u) {{
      float t = smem[0];
#pragma unroll
      for (unsigned w = 1; w < 8u; ++w) t += smem[w];
      {out}[0] = {rounded('t' + scale)};
    }}
    __syncthreads();
  }}
}}"""


# ------------------------------------------------------------------ leaf loop-invariant motion and sharing
_VARS = ("prow", "pcol", "latch_lr")


def _spans(text):
    """(start, end) of balanced (...) spans and ident[...] reads; a span directly after an identifier (a call or an
    array) includes the identifier."""
    out, stack = [], []
    for k, c in enumerate(text):
        if c in '([':
            stack.append((k, c))
        elif c in ')]' and stack:
            a, opener = stack.pop()
            if (opener, c) not in (('(', ')'), ('[', ']')):
                continue
            while a > 0 and (text[a - 1].isalnum() or text[a - 1] == '_'):
                a -= 1
            if opener == '[' and a == k:   # bare bracket without identifier: not an expression
                continue
            out.append((a, k + 1))
    return out


def _deps(span):
    d = {v for v in _VARS if re.search(r"\b" + v + r"\b", span)}
    if 'latch_hk' in span:
        d.add('pcol')                     # per-column temporaries are indexed by the column slot k
    if 'latch_hr' in span:
        d.add('prow')
    if re.search(r"\blatch_(e\d|s\d|lr)", span):
        d.update(('prow', 'pcol'))      # per-element temporaries and staged tiles
    return d


def _worth(span):
    # memory reads or calls; trivial literals/variables are left in place
    return bool(re.search(r"\[|\w\(", span)) and len(span) > 6


def _kind(span):
    d = _deps(span)
    if not d:
        return 'const'
    if d == {'prow'}:
        return 'row'
    if d == {'pcol'} and not span.startswith('__float2bfloat16_rn('):
        return 'col'                      # column temporaries are FP32 arrays: BF16-valued spans stay inline
    return None


PREFIX = {'const': 'latch_hc', 'row': 'latch_hr', 'col': 'latch_hk'}


def lift(leaves, allow_col=True):
    """Loop-invariant motion on pure leaf texts over prow / pcol / staged reads. Returns (leaves', const, row,
    col) declarations: const spans depend on no index (once per engine), row spans only on prow (once per row),
    col spans only on pcol (once per owned column, FP32 array indexed by k). Maximal spans first."""
    names, decls = {}, {'const': [], 'row': [], 'col': []}
    out = []
    for text in leaves:
        while True:
            best = None
            for a, b in _spans(text):
                span = text[a:b]
                if re.fullmatch(r"\(?latch_h\w+(\[k\])?\)?", span) or not _worth(span):
                    continue
                kind = _kind(span)
                if kind == 'col' and not allow_col:
                    kind = None
                if kind is not None and (best is None or b - a > best[1] - best[0]):
                    best = (a, b, kind)
            if best is None:
                break
            a, b, kind = best
            span = text[a:b]
            key = (kind, span)
            if key not in names:
                names[key] = f"{PREFIX[kind]}{len(names)}"
                decls[kind].append((names[key], span))
            text = text[:a] + (f"({names[key]}[k])" if kind == 'col' else f"({names[key]})") + text[b:]
        out.append(text)
    return out, decls['const'], decls['row'], decls['col']


def share(row_leaf, col_leaf, min_len=24):
    """Common subexpressions of the row and column leaves (same element) evaluated once: (row', col', decls)."""
    decls, k = [], 0
    while True:
        best = None
        for a, b in _spans(row_leaf):
            span = row_leaf[a:b]
            if len(span) >= min_len and _worth(span) and span in col_leaf and (best is None or len(span) > len(best)):
                best = span
        if best is None:
            return row_leaf, col_leaf, decls
        reg = f"latch_e{k}"
        k += 1
        decls.append((reg, best))
        row_leaf = row_leaf.replace(best, f"({reg})")
        col_leaf = col_leaf.replace(best, f"({reg})")
