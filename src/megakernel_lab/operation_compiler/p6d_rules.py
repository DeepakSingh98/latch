"""P6-D whole-update rules (defaults = the qualified candidate v6a, built in cpu-06) for the operation compiler (flat fused schedule).

Every rule is selected from operation, dimension, layout and dependency properties only (never model names) and
is exact unless stated otherwise. Module flags select rules for A/B builds; defaults are the qualified candidate.

POW_ONE                 aten.pow(x, 1) renders as x. IEEE pow(x, 1) = x for every x (incl. +-0, inf, NaN); the
                        CUDA powf identity is checked exhaustively on the GPU (all 2^32 inputs, bitwise).
BROADCAST_MATERIALIZE   a powf-class producer (pow with an exponent other than 1 or 2) whose readers see it through
                        a broadcast (reader extent >= 2x its own) is stored once instead of being re-evaluated per
                        reader element. Same expression, same rounding: bitwise equal.
OPTIMIZER_EPILOGUE      p5c register epilogues also admit (a) products with the exact FP32-converting store and
                        (b) members committed in place into input state, when every other reader of that state runs in
                        an earlier phase and members read it at their own index (load-before-store per element, the
                        same condition as the fused loops' in-place commit). The SGD update of a weight then runs in its
                        gradient product's epilogue on the converted gradient value: bitwise equal.
"""
from __future__ import annotations

# Late-operand rematerialization (ops/remat.py) is off by default in lower(): see LATE_REMAT.
LATE_REMAT = False
POW_ONE = True
BROADCAST_MATERIALIZE = True
OPTIMIZER_EPILOGUE = True   # in-place elementwise state updates (SGD) join an FP32-converting product's epilogue



def powf_exponent(exponent):
    """True when aten.pow renders through powf (any exponent other than the exact 1 and 2 special cases)."""
    return not (isinstance(exponent, (int, float)) and exponent in (1, 2))


# ------------------------------------------------------------------------------------------ reductions
ROW_REDUCTION = True        # warp-per-output replay of the frozen 256-thread CTA tree (exact, bitwise)
COLUMN_REDUCTION = True     # CTA-per-8-outputs replay with coalesced kept dimension (exact, bitwise)
ROW_MAX_EXTENT = 4096
ROW_PAIR = False            # two rows per warp concurrently in the row engine (exact)
COLUMN_OUTPUTS = 8
COLUMN_WIDE = True          # coalesced column engine (lanes over outputs; exact via shared-memory butterflies)


def _r_stride_one(src, dims):
    """The innermost reduced axis has unit stride (consecutive r are adjacent in memory)."""
    return bool(dims) and src.strides[dims[-1]] == 1 and src.shape[dims[-1]] > 1


def _o_stride_one(src, kept):
    return bool(kept) and src.strides[kept[-1]] == 1 and src.shape[kept[-1]] > 1


def _merged_stride(src, axes):
    """Single element stride of `axes` (outer -> inner) when they merge into one strided run, else None."""
    axes = [a for a in axes if src.shape[a] != 1]
    if not axes:
        return 0
    for outer, inner in zip(axes, axes[1:]):
        if src.strides[outer] != src.strides[inner] * src.shape[inner]:
            return None
    return src.strides[axes[-1]]


def _simple_position(src, dims, kept, r, o):
    """Exact 32-bit storage position r * sr + o * so (+ offset) when both index groups merge; else None."""
    sr, so = _merged_stride(src, dims), _merged_stride(src, kept)
    span = 1 + sum((d - 1) * st for d, st in zip(src.shape, src.strides)) + getattr(src, 'offset', 0)
    if sr is None or so is None or span >= (1 << 31):
        return None
    terms = [f'(unsigned){r} * {sr}u' if sr else None, f'(unsigned){o} * {so}u' if so else None]
    return ' + '.join(t for t in terms if t) or '0u'


def reduction_engine(*, out, position, extent, results, read, rounded, scale, src, dims, kept):
    """Return (code, extra) for an exact FP32 sum/mean over `dims` of `src`, or None.

    The frozen order (fusion.py generic path): thread t of a 256-thread CTA accumulates leaves r = t, t + 256, ...
    sequentially from +0; s101h::warp_sum (xor butterfly, identical in every lane) per warp; warp 0 sums the 8 warp
    partials (lanes >= 8 hold +0) with the same butterfly. Both engines evaluate exactly these additions.
    """
    if results < 64 or extent <= 256:
        return None
    if ROW_REDUCTION and _r_stride_one(src, dims) and extent <= ROW_MAX_EXTENT:
        # One warp per output. Lane l carries the 8 virtual threads t = 32k + l (k = 0..7) in v[k].
        simple = _simple_position(src, dims, kept, 'rr', 'o')
        leaf = read(simple if simple is not None else position.replace('(r / ', '(rr / '))
        steps = (extent + 255) // 256
        guard = '' if extent % 256 == 0 else f'rr < {extent}LL ? '
        tail = '' if extent % 256 == 0 else ' : 0.0f'
        code = f'''for (long long o = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o < {results}LL; o += ((long long)gridDim.x * blockDim.x) >> 5) {{
  const int lane = threadIdx.x & 31;
  float v[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = 0.0f;
#pragma unroll
  for (int j = 0; j < {steps}; ++j) {{
#pragma unroll
    for (int k = 0; k < 8; ++k) {{
      const long long rr = 256LL * j + 32 * k + lane;
      v[k] += {guard}{leaf}{tail};
    }}
  }}
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = s101h::warp_sum(v[k]);
  float second = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) if (lane == k) second = v[k];
  second = s101h::warp_sum(second);
  if (lane == 0) {out}[o] = {rounded('second' + scale)};
  {ROW_MARK}
}}'''
        if ROW_PAIR:
            # Two rows per warp in flight (o and o + W): twice the loads outstanding per warp; each row keeps its own
            # frozen order. The second row is evaluated at a clamped index when absent and not stored (exact).
            simple1 = _simple_position(src, dims, kept, 'rr', 'o1c')
            leaf1 = read(simple1 if simple1 is not None else position.replace('(r / ', '(rr / ').replace('(o / ', '(o1c / '))
            code = f'''{{
const long long W = ((long long)gridDim.x * blockDim.x) >> 5;
for (long long o0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5; o0 < {results}LL; o0 += 2 * W) {{
  const int lane = threadIdx.x & 31;
  const long long o1 = o0 + W;
  const bool has1 = o1 < {results}LL;
  const long long o1c = has1 ? o1 : o0;
  float v[8], u[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {{ v[k] = 0.0f; u[k] = 0.0f; }}
  {{
  const long long o = o0;
#pragma unroll
  for (int j = 0; j < {steps}; ++j) {{
#pragma unroll
    for (int k = 0; k < 8; ++k) {{
      const long long rr = 256LL * j + 32 * k + lane;
      v[k] += {guard}{leaf}{tail};
      u[k] += {guard}{leaf1}{tail};
    }}
  }}
  }}
#pragma unroll
  for (int k = 0; k < 8; ++k) {{ v[k] = s101h::warp_sum(v[k]); u[k] = s101h::warp_sum(u[k]); }}
  float second = 0.0f, second1 = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) if (lane == k) {{ second = v[k]; second1 = u[k]; }}
  second = s101h::warp_sum(second);
  second1 = s101h::warp_sum(second1);
  if (lane == 0) {out}[o0] = {rounded('second' + scale)};
  if (lane == 0 && has1) {out}[o1] = {rounded('second1' + scale)};
  {{ const long long o = o0; {ROW_MARK} }}
  if (has1) {{ const long long o = o1; {ROW_MARK} }}
}}
}}'''
        extra = {'engine': 'p6d_exact_row_per_warp', 'reduction_elements': extent}
        if simple is not None and not guard:
            # P9-A staged row-block engine (ops/p9_rowblock.py): the same leaf, order and store, from staged rows.
            extra['p9'] = {'kind': 'row', 'leaf': leaf, 'out': out, 'store': rounded('second' + scale),
                           'rows': results, 'extent': extent}
        return code, extra
    if COLUMN_REDUCTION and COLUMN_WIDE and _o_stride_one(src, kept) and not _r_stride_one(src, dims):
        # Lanes span 8 consecutive outputs x 4 virtual threads: a warp load touches 4 rows (not 32). Physical thread p
        # holds virtual threads t = (p >> 3) + 32 k (k = 0..7) of output (p & 7); the per-virtual-warp butterflies run
        # over the identical lane order through shared memory, then the 8 warp partials as before (exact).
        blocks = (results + 7) // 8
        simple = _simple_position(src, dims, kept, 'r', 'oc')
        leaf = read(simple if simple is not None else position.replace('(o / ', '(oc / '))
        steps = (extent + 255) // 256
        guard = ('' if extent % 256 == 0 and results % 8 == 0 else
                 f'(r < {extent}LL && oc < {results}LL) ? ')
        tail = '' if not guard else ' : 0.0f'
        code = f'''for (long long ob = blockIdx.x; ob < {blocks}LL; ob += gridDim.x) {{
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, c = threadIdx.x & 7, g = threadIdx.x >> 3;
  const long long oc = ob * 8 + c;
  float v[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] = 0.0f;
#pragma unroll 2
  for (int j = 0; j < {steps}; ++j) {{
#pragma unroll
    for (int k = 0; k < 8; ++k) {{
      const long long r = 256LL * j + 32 * k + g;
      v[k] += {guard}{leaf}{tail};
    }}
  }}
  __syncthreads();
#pragma unroll
  for (int k = 0; k < 8; ++k) smem[(c * 8 + k) * 32 + g] = v[k];
  __syncthreads();
  float part = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {{
    const float x = s101h::warp_sum(smem[(w * 8 + i) * 32 + lane]);
    if (lane == i) part = x;
  }}
  float a = lane < 8 ? part : 0.0f;
  a = s101h::warp_sum(a);
  const long long o = ob * 8 + w;
  if (lane == 0 && o < {results}LL) {out}[o] = {rounded('a' + scale)};
  __syncthreads();
}}'''
        extra = {'engine': 'p6d_exact_column_wide', 'reduction_elements': extent, 'outputs_per_cta': 8}
        if simple is not None and not guard:
            extra['p9'] = {'kind': 'col', 'leaf': leaf, 'out': out, 'store': rounded('a' + scale),
                           'rows': extent, 'extent': results}
        return code, extra
    if COLUMN_REDUCTION and _o_stride_one(src, kept) and not _r_stride_one(src, dims):
        c_out = COLUMN_OUTPUTS
        blocks = (results + c_out - 1) // c_out
        simple = _simple_position(src, dims, kept, 'r', 'oc')
        leaf = read(simple if simple is not None else position.replace('(o / ', '(oc / '))
        bound = '' if results % c_out == 0 else f' && oc < {results}LL'
        code = f'''for (long long ob = blockIdx.x; ob < {blocks}LL; ob += gridDim.x) {{
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  float acc[{c_out}];
#pragma unroll
  for (int c = 0; c < {c_out}; ++c) acc[c] = 0.0f;
#pragma unroll 2
  for (long long r = threadIdx.x; r < {extent}LL; r += 256) {{
#pragma unroll
    for (int c = 0; c < {c_out}; ++c) {{
      const long long oc = ob * {c_out} + c;
      if (true{bound}) acc[c] += {leaf};
    }}
  }}
#pragma unroll
  for (int c = 0; c < {c_out}; ++c) {{
    const float s = s101h::warp_sum(acc[c]);
    if (lane == 0) smem[c * 8 + w] = s;
  }}
  __syncthreads();
  if (w < {c_out}) {{
    float a = lane < 8 ? smem[w * 8 + lane] : 0.0f;
    a = s101h::warp_sum(a);
    const long long o = ob * {c_out} + w;
    if (lane == 0 && o < {results}LL) {out}[o] = {rounded('a' + scale)};
  }}
  __syncthreads();
}}'''
        return code, {'engine': 'p6d_exact_column_cta', 'reduction_elements': extent, 'outputs_per_cta': c_out}
    return None


# ------------------------------------------------------------------------------------------ graph rewrites
EMBEDDING_INDEX = False     # (measured neutral, gpu-11 v10a) contiguous embedding-gradient reads index pos * width + col directly (exact)
CSE = True                  # merge identical pure single-output operations (exact: same operation, same inputs)
_NO_CSE = ("empty", "empty_like", "rand", "randn", "dropout", "philox")


def cse(graph):
    """Common-subexpression elimination over captured TrainingIR graphs, applied before remat and lowering.

    Two pure single-output operations with the same target, the same (already canonicalized) arguments and keyword
    arguments, and the same output dtype/shape/strides compute the same value; readers of the later one read the
    earlier one. Protected values (graph outputs, inputs, state writes) are never removed. Returns (graph, report).
    """
    from dataclasses import replace
    from .ir import TrainingIR
    from .remat import _names, _rewrite
    report = {"merged": 0, "merged_values": []}
    if not CSE or not isinstance(graph, TrainingIR) or not isinstance(graph.outputs, dict):
        return graph, report
    protected = _names(graph.outputs, set(graph.inputs))
    _names(getattr(graph, "state_writes", None) or {}, protected)
    mapping, op_map, seen, result = {}, {}, {}, []
    values = dict(graph.values)
    for op in graph.operations:
        if mapping:
            op = replace(op, args=_rewrite(op.args, mapping), kwargs=_rewrite(op.kwargs, mapping),
                         inputs=tuple(mapping.get(x, x) for x in op.inputs),
                         reads=tuple(mapping.get(x, x) for x in op.reads),
                         dependencies=tuple(dict.fromkeys(op_map.get(d, d) for d in op.dependencies)))
        base = op.target.removeprefix("aten.").split(".")[0]
        if (op.effect == "pure" and len(op.outputs) == 1 and tuple(op.writes) in ((), tuple(op.outputs))
                and op.outputs[0] not in protected and not any(base.startswith(x) for x in _NO_CSE)
                and "latch_ops" not in op.target):
            out = values[op.outputs[0]]
            key = (op.target, repr(op.args), repr(sorted(op.kwargs.items())), out.dtype, tuple(out.shape),
                   tuple(out.strides), out.storage_offset,
                   mapping.get(out.alias_of, out.alias_of) if out.alias_of else None)
            if key in seen:
                canonical_op, canonical_value = seen[key]
                mapping[op.outputs[0]] = canonical_value
                op_map[op.name] = canonical_op
                values.pop(op.outputs[0], None)
                report["merged"] += 1
                report["merged_values"].append([op.outputs[0], canonical_value])
                continue
            seen[key] = (op.name, op.outputs[0])
        result.append(op)
    if not report["merged"]:
        return graph, report
    for name, value in list(values.items()):
        if value.alias_of in mapping:
            values[name] = replace(value, alias_of=mapping[value.alias_of])
    metadata = dict(graph.metadata)
    metadata["p6d_cse"] = {"merged": report["merged"]}
    return replace(graph, values=values, operations=result, metadata=metadata), report


# ------------------------------------------------------------------------------------------ row consumers
ROW_CONSUMERS = True        # elementwise consumers of a warp-per-row reduction run in the reduction's phase (exact)
ROW_MARK = "/*P6D_ROW_CONSUMERS*/"


def attach_row_consumers(*, items, loads_of, phase_of, phases, nodes, layouts, in_place, inplace_source, numel,
                         index32):
    """Move elementwise consumers of a row-engine reduction into the reduction's phase.

    A reduction R (engine p6d_exact_row_per_warp, `rows` outputs over a unit-stride extent E) is computed by one warp
    per row. A register-connected member set of the loop over rows * E elements (or of the loop over `rows` elements)
    in the phase right after R moves into R's warp, after R's value for that row is stored, when every member load is
    (a) R through the row broadcast (element index i / E) or at its own index (row loop), (b) another member of the
    set at the same index, or (c) a value produced strictly before R's phase (graph inputs included). No member may be
    an in-place state source or read committed state. Members are evaluated with their unchanged per-operation
    expressions on the same values, so results are bitwise equal; readers outside the set stay in later phases and
    read members from memory. Returns ({reduction: {"elements": [...], "scalars": [...]}}, {member: reduction}).
    """
    committed = set(in_place)
    attached, owner = {}, {}
    order = {node.name: k for k, node in enumerate(items)}
    readers = {}
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, set()).add(node.name)
    for node in items:
        if node.kind != "reduction" or (node.extra or {}).get("engine") != "p6d_exact_row_per_warp":
            continue
        r = node.name
        rows, extent = numel(layouts[r].shape), int(node.extra["reduction_elements"])
        p = phase_of[r]
        if p + 1 >= len(phases):
            continue
        groups = {}
        for size, kind in ((rows * extent, "elements"), (rows, "scalars")):
            # Loops of R's own phase join too when nothing else in that phase reads them (they become part of the
            # warp's per-row work); loops of the next phase join when their inputs are ready.
            same = [m for m in phases[p]["loops"].get(size, ()) if m not in owner]
            candidates = same + [m for m in phases[p + 1]["loops"].get(size, ()) if m not in owner]
            chosen = []
            for name in sorted(candidates, key=order.get):
                member = nodes[name]
                if (member.kind != "elementwise" or member.dtype not in ("float32", "bfloat16")
                        or name in inplace_source or name in committed or numel(member.shape) != size):
                    continue
                if name in same and any(phase_of.get(reader, p + 1) <= p and reader not in candidates
                                        for reader in readers.get(name, ())):
                    continue
                ok, reads_r = True, False
                for root, lay, shape, idx in loads_of[name]:
                    if root in committed:
                        # Moving a reader of old state earlier keeps it before the commit (which is after p).
                        ok = ok and phase_of.get(in_place[root], -1) > p
                    elif root == r:
                        try:
                            position = index32(lay, shape, idx) if shape is not None else None
                        except Exception:  # noqa: BLE001 - any unrecognized index keeps the member in place
                            position = None
                        wanted = f"i/{extent}u" if kind == "elements" else "i"
                        flat = position.replace("(", "").replace(")", "").replace(" ", "") if position else None
                        ok = ok and flat == wanted
                        reads_r = True
                    elif root in chosen:
                        ok = ok and shape is not None and idx == "i" and lay.contiguous and lay.offset == 0
                    elif root in phase_of and phase_of[root] >= p:
                        ok = False
                if ok:
                    chosen.append(name)
            # The proposed loop may contain state commits or other members that
            # cannot move into the row's warp. Membership in `candidates` is not
            # enough: a remaining same-phase loop uses a different CTA/index
            # mapping and cannot read a row member before a grid barrier.
            # Close the accepted set in both directions. Removing a producer
            # also removes consumers that would now read it from this phase.
            while chosen:
                accepted = set(chosen)
                rejected = {
                    name for name in chosen
                    if any(phase_of.get(reader, p + 1) <= p and reader not in accepted
                           for reader in readers.get(name, ()))
                    or any(root != r and root not in committed and root in phase_of
                           and phase_of[root] >= p and root not in accepted
                           for root, _, _, _ in loads_of[name])
                }
                if not rejected:
                    break
                chosen = [name for name in chosen if name not in rejected]
            if chosen:
                groups[kind] = chosen
        if not groups:
            continue
        for kind, members in groups.items():
            size = rows * extent if kind == "elements" else rows
            for q in (p, p + 1):
                if size in phases[q]["loops"]:
                    phases[q]["loops"][size] = [m for m in phases[q]["loops"][size] if m not in members]
                    if not phases[q]["loops"][size]:
                        del phases[q]["loops"][size]
            for m in members:
                phase_of[m] = p
                owner[m] = r
        attached[r] = {"elements": groups.get("elements", []), "scalars": groups.get("scalars", []),
                       "rows": rows, "extent": extent}
    return attached, owner


SINK_TINY = True            # phases holding only tiny elementwise loops join the following phase (exact)
MATRIX_GROUP_MAX = 6        # independent matrix products per phase (P-OPS5-C used 3)
MULTI_REDUCTION = 2         # up to this many independent exact-engine reductions per phase (0: one, as P-OPS5-C)
TINY_ELEMENTS = 65536


def sink_tiny_loops(*, items, loads_of, phase_of, phases, in_place, inplace_source):
    """Move a phase made only of tiny elementwise loops (no heavy item) into the next phase.

    Legal when no other item reads a member in the next phase (readers stay strictly later), and for a member that
    commits state in place, no other reader of that old state runs in the next phase. Members keep their loops and
    expressions; only the phase (one grid barrier fewer) changes. Returns the number of phases emptied.
    """
    readers = {}
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, set()).add(node.name)
    moved = 0
    for q in range(len(phases) - 1):
        phase = phases[q]
        if phase["heavy"] or not phase["loops"] or phase.get("epilogue"):
            continue
        if any(size > TINY_ELEMENTS for size in phase["loops"]):
            continue
        members = {m for loop in phase["loops"].values() for m in loop}
        ok = True
        for m in members:
            for reader in readers.get(m, ()):
                if reader not in members and phase_of.get(reader, q + 2) <= q + 1:
                    ok = False
            target = inplace_source.get(m)
            if target is not None:
                for reader in readers.get(target, ()):
                    if reader not in members and phase_of.get(reader, -1) == q + 1:
                        ok = False
        if not ok:
            continue
        for size, loop in phase["loops"].items():
            phases[q + 1]["loops"].setdefault(size, []).extend(loop)
            for m in loop:
                phase_of[m] = q + 1
        phase["loops"] = {}
        moved += 1
    return moved


# ------------------------------------------------------------------------------------------ compile structure
ATTENTION_SITES = False     # identical attention item loops share one entry-resident site (goto), fewer bodies


class AttentionSites:
    """Route each long-attention item loop to one shared site per distinct loop text (P5C_SHARED_SITES pattern).

    A phase's `plong::Args A{}` initialization stays in place; A is copied to a kernel-scope variable and control
    jumps to the site emitted after the kernel's final statement, which runs the identical loop and returns through
    a uniform switch. Same function (no call: wgmma stays in the entry function), same arithmetic and schedule.
    """
    import re as _re
    BLOCK = _re.compile(r'plong::Args A\{\};\n(?P<fields>(?:A\..*\n)+)(?P<comments>(?://.*\n)*)'
                        r'(?P<loop>for \(long long r = 0;; \+\+r\) \{\n(?:  .*\n)+?\})')

    def __init__(self):
        self.sites, self.returns = {}, 0

    def route(self, code):
        def replace(match):
            loop = match.group('loop')
            site = self.sites.setdefault(loop, len(self.sites))
            r = self.returns
            self.returns += 1
            return ('{\nplong::Args A{};\n' + match.group('fields') + f'p6d_sa = A; p6d_sr = {r};\n}}\n'
                    + match.group('comments') + f'goto P6D_ASITE_{site};\nP6D_ARET_{r}: ;')
        return self.BLOCK.sub(replace, code)

    def declarations(self):
        return '\n  plong::Args p6d_sa{}; int p6d_sr = 0;\n' if self.sites else ''

    def definitions(self):
        if not self.sites:
            return ''
        cases = ' '.join(f'case {r}: goto P6D_ARET_{r};' for r in range(self.returns))
        out = ['  return;']
        for loop, site in self.sites.items():
            out.append(f'P6D_ASITE_{site}:\n{{\nconst plong::Args& A = p6d_sa;\n{loop}\n}}\n'
                       f'  switch (p6d_sr) {{ {cases} default: return; }}')
        return '\n'.join(out) + '\n'



SILU_SAVE = False           # sigmoid nodes are stored where their input is produced, not re-evaluated in each reader
