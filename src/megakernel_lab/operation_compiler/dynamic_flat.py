"""LATCH-STEP19-SPEED: bounded active extents inside the established flat scheduler.

The flat effects scheduler (fusion_reference.lower_fused) owns inlining, register
values, ASAP/slack placement, matrix co-scheduling and in-place state commits. A
bounded dynamic batch program reuses all of it over capacity layouts. This module
supplies only what depends on the device active extent ``latch_active_batch``:

* loop keys: an active loop (its values carry the leading batch axis) never shares
  a loop, register or epilogue with a static loop of equal capacity element count;
* loop bounds: active loops visit exactly ``B * row`` elements (row = numel / capacity);
* reductions over active values: fixed-order CTA trees over the active elements,
  or one ascending per-thread sum per kept output when only the batch axis is reduced;
* matrices: the existing masked-edge BF16 body (popsc::bmm_edge_phase) under the
  capacity tile plan, with runtime M (active rows) or K (active reduction extent);
* explicit Philox masks/advances: active element counts and counter offsets.

Static programs never construct this object; their emission is unchanged.
"""
from __future__ import annotations

import math

# Normal compilation selects the flat schedule for admitted dynamic programs; False keeps the
# ordered dynamic lowering (the qualified STEP19 baseline) for paired comparisons.
ENABLED = True
# Flat-scheduler rules not yet given active-extent contracts stay off for dynamic programs.
REFUSED_RULES = {"pops2b_defer": False, "pops2b_prefetch": False, "pops2b_plans": False}
# BF16 operand casts computed in product staging (masked prologues; removes the separate cast phase).
# Measured slower (gpu-v8-01: the converting mainloop takes ~13 us for the forward product); off.
CAST_PROLOGUES = False


def flat_options(pops2b_rules):
    """Flat-scheduler rule settings for a dynamic program (explicit unsupported requests are refused)."""
    from .lower import LoweringError
    for key, value in pops2b_rules.items():
        if key in REFUSED_RULES and value != REFUSED_RULES[key]:
            raise LoweringError(f"{key} has no bounded active-extent contract")
    return {**pops2b_rules, **REFUSED_RULES, "pops2b_casts": bool(CAST_PROLOGUES)}


# Split-K for active-row BF16 products with a static K: slice counts tried in order; each slice keeps at
# least SPLIT_MIN_STEPS pipeline steps and tiles x slices fits the modeled grid. () disables it.
SPLIT_CHOICES = (8, 4, 2)
SPLIT_MIN_STEPS = 2
SPLIT_MIN_CHAIN = 16                # unsplit K steps below this keep the single chain (gpu-flat5/v6: 8 slowed mm)
# Co-scheduling choices for dynamic programs; selected from paired H100 variants gpu-v6-01/gpu-v8-01 (HANDOFF).
HALF_GRID_PLAN = False              # a full-grid product may take its half-grid plan to host bounded reductions
REDUCTIONS_AFTER_TILES = True       # bounded reductions may follow matrix tiles on the same CTAs (no half plan)
EPILOGUE_STREAM_GATE = False        # state-streaming epilogue chains must fit the product mainloop time
# Full reductions over at least this many capacity elements use several CTAs (16 when placed on idle CTAs).
FULL_REDUCTION_MIN = 2048
FULL_REDUCTION_CTAS = 16
# Bounded reductions needing at most this many CTAs may run on CTAs a small matrix group leaves idle.
IDLE_REDUCTION_CTAS = 16
# Reductions run with a virtual CTA index/count; the whole grid unless placed on idle CTAs.
CTA_WRAP = "{ const unsigned latch_rb = blockIdx.x, latch_rg = gridDim.x;\n"

# Generated epilogue admission for small products: chain expression cost x elements per product CTA
# (inline-style units; a transcendental counts 4): at most 128 units per thread of a 256-thread CTA,
# far below the grid barrier and whole-grid loop a separate phase would cost.
EPILOGUE_TILE_WORK = 32768


class FlatActive:
    epilogue_tile_work = None

    def __init__(self, declaration, values, *, retain_all=False):
        from .lower import LoweringError, _field
        # retain_all keeps every intermediate in storage (no inlining, epilogues, prologues or compaction);
        # active loop bounds, reductions and masked/split matrix bodies are the same as the packed program's.
        self.retain_all = retain_all
        self.declaration = declaration
        self.axes = dict(declaration["value_axes"])
        self.maximum = declaration["maximum"]
        self.input = declaration["input"]
        if type(self.maximum) is not int or not 2 <= self.maximum <= 32:
            raise LoweringError("invalid bounded batch capacity")
        for name, axis in self.axes.items():
            shape = tuple(_field(values[name], "shape"))
            if type(axis) is not int or not 0 <= axis < len(shape) or shape[axis] != self.maximum:
                raise LoweringError(f"{name}: active axis must have the declared capacity")
        self.used_matrix = []
        self.epilogue_tile_work = EPILOGUE_TILE_WORK
        self.epilogue_stream_gate = EPILOGUE_STREAM_GATE

    # ------------------------------------------------------------------ classification
    def active(self, name):
        return self.axes.get(name) == 0

    def axis(self, name):
        return self.axes.get(name)

    def key(self, name, numel):
        """Loop key: plain capacity count for static loops, a tagged count for active ones."""
        return ("active", numel) if self.active(name) else numel

    @staticmethod
    def numel_of_key(key):
        return key[1] if isinstance(key, tuple) else key

    @staticmethod
    def is_active_key(key):
        return isinstance(key, tuple)

    def count(self, numel, lanes=1):
        """Unsigned active iteration count of a leading-axis active loop over numel capacity elements."""
        row = numel // self.maximum
        if numel % self.maximum or row % lanes:
            from .lower import LoweringError
            raise LoweringError("active loop rows must divide into the vector width")
        return f"((unsigned)latch_active_batch * {row // lanes}u)"

    def check_elementwise(self, name, shape, loads, layouts):
        from .lower import LoweringError
        axis = self.axes.get(name)
        if axis not in (None, 0):
            raise LoweringError(f"{name}: bounded batch elementwise values need a leading active axis")
        for ref in loads:
            ref_axis = self.axes.get(ref, self.axes.get(layouts[ref].root) if ref in layouts else None)
            if ref_axis is None:
                continue
            if ref_axis != 0:
                raise LoweringError(f"{ref}: nonleading active view read elementwise")
            if axis is None:
                raise LoweringError(f"{name}: static elementwise value reads active {ref}")

    # ------------------------------------------------------------------ reductions
    def reduction(self, *, name, source, src, dims, kept, out, read, rounded, mean, out_shape):
        """Return (code, extra) for a floating sum/mean over an active source, or None if source is static."""
        from .lower import LoweringError, _numel
        axis = self.axes.get(source)
        if axis is None:
            if self.axes.get(name) is not None:
                raise LoweringError(f"{name}: reduction of a static value cannot produce an active value")
            return None
        if self.axes.get(name) not in (None, 0):
            raise LoweringError(f"{name}: reduction result needs a leading active axis")

        def extent(axes):
            static = math.prod(src.shape[i] for i in axes if i != axis)
            return f"((long long)latch_active_batch * {static}LL)" if axis in axes else f"{static}LL"
        reduced, results = extent(dims), extent(kept)
        scale = f" / (float){reduced}" if mean else ""
        if list(dims) == [axis] and axis not in kept:
            # Only the batch axis is reduced: one thread per kept output sums its
            # active rows in ascending order (bounded reads, fixed order per B).
            kept_shape = [src.shape[i] for i in kept]
            terms = []
            for j, a in enumerate(kept):
                tail = math.prod(kept_shape[j + 1:])
                coordinate = "o" if tail == 1 else f"(o / {tail}LL)"
                if j:
                    coordinate = f"({coordinate} % {src.shape[a]}LL)"
                if src.strides[a]:
                    terms.append(coordinate if src.strides[a] == 1 else f"{coordinate} * {src.strides[a]}LL")
            base = " + ".join(terms) or "0LL"
            step = src.strides[axis]
            code = CTA_WRAP + f"""for (long long o = (long long)latch_rb * blockDim.x + threadIdx.x; o < {results}; o += (long long)latch_rg * blockDim.x) {{
  float acc = 0.0f;
  for (long long r = 0; r < (long long)latch_active_batch; ++r) acc += {read(f"({base}) + r * {step}LL")};
  {out}[o] = {rounded('acc' + scale)};
  /*RED_EPI*/
}}}}"""
            ctas = max(1, math.ceil(math.prod(src.shape[i] for i in kept) / 256))
            return code, {"reduction_elements": self.maximum, "engine": "bounded_active_rows_per_thread",
                          "work_items": ctas, "idle_ok": ctas <= IDLE_REDUCTION_CTAS,
                          "arithmetic": "ascending active-row FP32 sum per kept output; no atomics"}
        if not kept and src.contiguous and list(dims) == list(range(len(src.shape))) \
                and math.prod(src.shape) >= FULL_REDUCTION_MIN:
            # One output over all active elements: contiguous per-CTA chunks with the CTA tree, then the
            # last arriving CTA combines the partials in fixed CTA order (the C-OPS4-B/P6-C msum protocol of
            # the static flat schedule, with the virtual CTA count of its placement).
            read_r = read("r")
            code = CTA_WRAP + f"""__shared__ unsigned latch_full_last_;
  const long long extent_ = {reduced};
  const long long chunk_ = (extent_ + latch_rg - 1) / latch_rg, lo_ = (long long)latch_rb * chunk_;
  const long long hi_ = lo_ + chunk_ < extent_ ? lo_ + chunk_ : extent_;
  float acc = 0.0f;
#pragma unroll 4
  for (long long r = lo_ + threadIdx.x; r < hi_; r += blockDim.x) acc += {read_r};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {{
      (/*P2_PART*/)[latch_rb] = acc;
      __threadfence();
      latch_full_last_ = atomicAdd(/*P2_CNT*/, 1u) == latch_rg - 1u;
    }}
  }}
  __syncthreads();
  if (latch_full_last_) {{
    __threadfence();
    acc = 0.0f;
    for (unsigned b = threadIdx.x; b < latch_rg; b += blockDim.x) acc += __ldcg((/*P2_PART*/) + b);
    acc = s101h::warp_sum(acc);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {{
      acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
      acc = s101h::warp_sum(acc);
      if (threadIdx.x == 0) {{ {out}[0] = {rounded('acc' + scale)}; *(/*P2_CNT*/) = 0u; }}
    }}
  }}
  __syncthreads();
}}"""
            return code, {"reduction_elements": _numel(src.shape), "engine": "bounded_active_multi_cta_full",
                          "work_items": FULL_REDUCTION_CTAS, "idle_ok": True,
                          "p2_split": {"counter_words": 1, "partial_words": 1024},
                          "arithmetic": "per-CTA contiguous active chunks with the CTA tree, then a fixed-order "
                                        "combine of the CTA partials (deterministic for its CTA count)"}
        terms = []
        for axes, index in ((dims, "r"), (kept, "o")):
            for j, a in enumerate(axes):
                if src.strides[a] == 0:
                    continue
                size = "(long long)latch_active_batch" if a == axis else f"{src.shape[a]}LL"
                tail = extent(axes[j + 1:])
                coordinate = index if tail == "1LL" else f"({index} / {tail})"
                if j:
                    coordinate = f"({coordinate} % {size})"
                terms.append(coordinate if src.strides[a] == 1 else f"{coordinate} * {src.strides[a]}LL")
        position = ("r" if src.contiguous and list(dims) == list(range(len(src.shape)))
                    else " + ".join(terms) or "0")
        code = CTA_WRAP + f"""for (long long o = latch_rb; o < {results}; o += latch_rg) {{
  float acc = 0.0f;
  for (long long r = threadIdx.x; r < {reduced}; r += blockDim.x) acc += {read(position)};
  acc = s101h::warp_sum(acc);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x < 32) {{
    acc = threadIdx.x < 8 ? smem[threadIdx.x] : 0.0f;
    acc = s101h::warp_sum(acc);
    if (threadIdx.x == 0) {out}[o] = {rounded('acc' + scale)};
  }}
  __syncthreads();
}}}}"""
        outputs = math.prod(src.shape[i] for i in kept)     # capacity bound of the kept outputs
        return code, {"reduction_elements": _numel([src.shape[i] for i in dims]), "engine": "bounded_active_cta_tree",
                      "work_items": outputs, "idle_ok": outputs <= IDLE_REDUCTION_CTAS,
                      "arithmetic": "fixed CTA tree over active elements; FP32 accumulation"}

    # ------------------------------------------------------------------ matrices
    def matrix_spec(self, spec, a, b, name):
        """Annotate a capacity BF16 spec with its runtime extents; refuse other active layouts."""
        from .lower import LoweringError
        aa, ba, oa = self.axes.get(a), self.axes.get(b), self.axes.get(name)
        if spec is None or spec.get("dtype") != "bfloat16" or spec.get("engine") or spec.get("batch") != 1:
            raise LoweringError("bounded dynamic flat schedule admits capacity BF16 small-tile products")
        if (aa == 1) != (ba == 0):
            raise LoweringError("both matrix reduction operands must share their active K extent")
        if aa == 0 and oa != 0 or ba == 1 or (aa != 0 and oa is not None):
            raise LoweringError("matrix output active-axis contract mismatch")
        spec = dict(spec)
        spec.pop("tail", None)            # the masked-edge body covers every capacity tile plan
        spec["active"] = {"m": aa == 0, "k": aa == 1}
        if aa == 0 and SPLIT_CHOICES:
            # Few active-row tiles over a long static K: the existing split-K protocol (ordered slice
            # partials, last-arriver fixed-order sum, one BF16 rounding) puts more CTAs on the K chain.
            from .matrix_rules import _plan, TARGET_GRID
            plan = _plan(spec, TARGET_GRID)
            nk = spec["k"] // plan["ks"]
            for splits in (SPLIT_CHOICES if nk >= SPLIT_MIN_CHAIN else ()):
                if nk % splits == 0 and nk // splits >= SPLIT_MIN_STEPS and plan["tiles"] * splits <= TARGET_GRID:
                    spec["split"] = splits
                    break
        self.used_matrix.append(name)
        return spec

    # ------------------------------------------------------------------ explicit Philox
    def philox_offset(self, value):
        from .dynamic_codegen import scalar
        return scalar(value, integer=True)

    # ------------------------------------------------------------------ manifest
    def manifest(self, manifest, layouts, phase_records):
        import copy
        manifest["dynamic_batch"] = copy.deepcopy(self.declaration)
        manifest["dynamic_batch"]["storage_policy"] = "capacity layouts and strides; active material rows only"
        manifest["dynamic_batch"]["schedule"] = ("established flat dependency scheduler; active loop bounds, "
                                                 "reductions and masked-edge BF16 products")
        for name, axis in self.axes.items():
            if name in manifest["layouts"]:
                manifest["layouts"][name]["active_axis"] = axis
        for record in phase_records:
            touched = {v: self.axes[v] for v in [*record.get("reads", ()), *record.get("writes", ())]
                       if v in self.axes}
            record["reads"] = list(dict.fromkeys([*record.get("reads", ()), self.input]))
            if touched:
                record["active_values"] = touched


def idle_heavy(items, offset):
    """Reductions on consecutive CTA ranges after the matrix tasks of the phase ([offset, grid)).

    With a grid too small for the ranges, ranges wrap and one CTA may run several reductions in
    sequence; the barrier between items is outside every CTA condition.
    """
    lines, cursor = ["__syncthreads();  // matrix staging done before reduction shared use"], 0
    for index, (name, code, ctas) in enumerate(items):
        if not code.startswith(CTA_WRAP):
            from .lower import LoweringError
            raise LoweringError(f"{name}: reduction has no virtual CTA form")
        body = code[len(CTA_WRAP):].rstrip()
        if not body.endswith("}"):
            from .lower import LoweringError
            raise LoweringError(f"{name}: malformed virtual CTA reduction")
        body = body[:-1]                    # the wrapper's closing brace; the idle block supplies its own
        if index:
            lines.append("__syncthreads();")
        lines.append(f"""// reduction on idle CTAs: {name}
{{ const unsigned latch_g0 = gridDim.x, latch_n = {ctas}u < latch_g0 ? {ctas}u : latch_g0;
  const unsigned latch_s = ({offset}u + {cursor}u + latch_n <= latch_g0) ? {offset}u + {cursor}u : {cursor}u % latch_g0;
  const unsigned latch_d = (blockIdx.x + latch_g0 - latch_s) % latch_g0;
  if (latch_d < latch_n) {{ const unsigned latch_rb = latch_d, latch_rg = latch_n;
{body}
  }}
}}""")
        cursor += ctas
    return "\n".join(lines)


def compact_heavy(*, phases, phase_of, items, loads_of, nodes, epilogues, in_place, inplace_source, can_join,
                  register_load):
    """Move heavy items (with their generated epilogue members) to the earliest legal phase.

    Epilogue attachment moves elementwise chains into earlier matrix phases; reductions and products
    reading those values may then start earlier too. Only earlier moves are made; every memory read
    stays after its producer's phase, old-state reads stay before in-place commits, and _check_schedule
    revalidates the result.
    """
    readers = {}
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, set()).add(node.name)
    moved = True
    while moved:
        moved = False
        for p in range(len(phases)):
            for name in list(phases[p]["heavy"]):
                group = [name, *epilogues.get(name, ())]
                members = set(group)
                earliest = 0
                for member in group:
                    for root, _, _, _ in loads_of.get(member, ()):
                        if root in members or root not in phase_of:
                            continue
                        earliest = max(earliest, phase_of[root] + 1)
                    target = inplace_source.get(member)
                    if target is not None:
                        for reader in readers.get(target, ()):
                            if reader not in members and reader in phase_of:
                                earliest = max(earliest, phase_of[reader] + 1)
                # Old-state reads stay at or before their in-place commit phase.
                commit_limits = [phase_of[in_place[root]] for member in group
                                 for root, _, _, _ in loads_of.get(member, ())
                                 if root in in_place and in_place[root] not in members and in_place[root] in phase_of]
                for q in range(earliest, p):
                    if commit_limits and q > min(commit_limits):
                        break
                    if not can_join(q, nodes[name]):
                        continue
                    phases[p]["heavy"].remove(name)
                    phases[q]["heavy"].append(name)
                    phases[q]["heavy"].sort(key=lambda n: nodes[n].order)
                    if name in epilogues:
                        source = phases[p].get("epilogue", {}).pop(name)
                        phases[q].setdefault("epilogue", {})[name] = source
                    for member in group:
                        phase_of[member] = q
                    moved = True
                    break
        order = {node.name: k for k, node in enumerate(items)}
        for p in range(len(phases)):
            if phases[p]["heavy"] or phases[p].get("epilogue"):
                continue        # loops placed beside matrix/reduction work (often on idle CTAs) stay there
            for key in list(phases[p]["loops"]):
                for chain in _components(phases[p]["loops"].get(key, ()), loads_of, nodes, register_load):
                    members = set(chain)
                    earliest = 0
                    commit_limits = []
                    for member in chain:
                        for root, _, _, _ in loads_of.get(member, ()):
                            if root in members or root not in phase_of:
                                if root in in_place and in_place[root] not in members and in_place[root] in phase_of:
                                    commit_limits.append(phase_of[in_place[root]] - 1)
                                continue
                            earliest = max(earliest, phase_of[root] + 1)
                        target = inplace_source.get(member)
                        if target is not None:
                            for reader in readers.get(target, ()):
                                if reader not in members and reader in phase_of:
                                    earliest = max(earliest, phase_of[reader] + 1)
                    limit = min(commit_limits) if commit_limits else p
                    q = earliest
                    if q >= p or q > limit:
                        continue
                    loop = phases[p]["loops"][key]
                    for member in chain:
                        loop.remove(member)
                        phase_of[member] = q
                    if not loop:
                        del phases[p]["loops"][key]
                    destination = phases[q]["loops"].setdefault(key, [])
                    destination.extend(chain)
                    destination.sort(key=order.get)
                    moved = True
        empty = [k for k, ph in enumerate(phases) if not ph["heavy"] and not ph["loops"] and not ph.get("epilogue")]
        if empty:
            remap, kept = {}, []
            for k, ph in enumerate(phases):
                if ph["heavy"] or ph["loops"] or ph.get("epilogue"):
                    remap[k] = len(kept)
                    kept.append(ph)
            phases[:] = kept
            for n in phase_of:
                phase_of[n] = remap[phase_of[n]]


def _components(members, loads_of, nodes, register_load):
    """Register-connected components of one loop's members (operation order kept)."""
    member_set, parent = set(members), {n: n for n in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for name in members:
        for root, layout, shape, idx in loads_of.get(name, ()):
            if root in member_set and shape is not None and register_load(nodes[name], root, layout, shape, idx):
                parent[find(root)] = find(name)
    groups = {}
    for name in members:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values())


def attach_reduction_epilogues(*, items, loads_of, phase_of, phases, nodes, layouts, in_place, inplace_source,
                               register_load, key_of, epilogue_of):
    """Elementwise register chains consuming a per-output reduction at the same flat index run in that
    reduction's thread right after its result (e.g. a bias gradient followed by its optimizer update).

    The conditions mirror pops2b_rules.attach_epilogues: every other operand is produced in an earlier
    phase (or is a graph input), the chain's results are read only in later phases, and in-place commits
    stay after every other old-state read. Returns {reduction: [members in operation order]}.
    """
    order = {node.name: k for k, node in enumerate(items)}
    readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            readers.setdefault(root, []).append((node.name, layout, shape, idx))
    commit_phase = {target: phase_of[source] for target, source in in_place.items()}

    def identity(layout, shape, idx, n):
        return (idx == "i" and shape is not None and math.prod(shape) == n and layout.numel == n
                and layout.contiguous and layout.offset == 0)

    def component(start):
        q, key = phase_of[start], key_of(start)
        members = set(phases[q]["loops"].get(key, ()))
        group, frontier = {start}, [start]
        while frontier:
            current = frontier.pop()
            neighbours = [root for root, layout, shape, idx in loads_of[current]
                          if root in members and shape is not None and register_load(nodes[current], root, layout, shape, idx)]
            neighbours += [reader for reader, layout, shape, idx in readers.get(current, ())
                           if reader in members and shape is not None and register_load(nodes[reader], current, layout, shape, idx)]
            for other in neighbours:
                if other not in group:
                    group.add(other)
                    frontier.append(other)
        return sorted(group, key=order.get)

    def valid(chain, red, p, n, key):
        members = set(chain)
        for name in chain:
            node = nodes[name]
            if node.kind != "elementwise" or node.dtype not in ("float32", "bfloat16") or key_of(name) != key:
                return False
            if name in epilogue_of:
                return False
            for root, layout, shape, idx in loads_of[name]:
                if root in members and shape is not None and register_load(node, root, layout, shape, idx):
                    continue
                if root == red:
                    if not identity(layout, shape, idx, n):
                        return False
                    continue
                if root in commit_phase:
                    source = in_place[root]
                    if source in members:
                        if not identity(layout, shape, idx, n):
                            return False
                    elif not commit_phase[root] > p:
                        return False
                    continue
                if root not in phase_of:
                    continue
                if phase_of[root] >= p:
                    return False
            if name in inplace_source:
                for reader, layout, shape, idx in readers.get(inplace_source[name], ()):
                    if reader not in members and phase_of[reader] >= p:
                        return False
            for reader, layout, shape, idx in readers.get(name, ()):
                if reader not in members and phase_of[reader] <= p:
                    return False
        return True

    attached = {}
    for red in sorted((n.name for n in items if n.kind == "reduction"
                       and n.extra.get("engine") == "bounded_active_rows_per_thread"), key=lambda r: phase_of[r]):
        layout = layouts[red]
        if not layout.contiguous or layout.offset or layout.root != red:
            continue
        p, n, key = phase_of[red], layout.numel, key_of(red)
        progress = True
        while progress:
            progress = False
            for reader, rlayout, shape, idx in readers.get(red, ()):
                if (reader in epilogue_of or nodes[reader].kind != "elementwise" or phase_of[reader] <= p
                        or not identity(rlayout, shape, idx, n)):
                    continue
                chain = component(reader)
                if not valid(chain, red, p, n, key):
                    continue
                q = phase_of[reader]
                loop = phases[q]["loops"][key]
                for name in chain:
                    loop.remove(name)
                    phase_of[name] = p
                    epilogue_of[name] = red
                    if name in inplace_source:
                        commit_phase[inplace_source[name]] = p
                if not loop:
                    del phases[q]["loops"][key]
                attached.setdefault(red, []).extend(chain)
                attached[red].sort(key=order.get)
                progress = True
                break
    empty = [k for k, ph in enumerate(phases) if not ph["heavy"] and not ph["loops"] and not ph.get("epilogue")]
    if empty:
        remap, kept = {}, []
        for k, ph in enumerate(phases):
            if ph["heavy"] or ph["loops"] or ph.get("epilogue"):
                remap[k] = len(kept)
                kept.append(ph)
        phases[:] = kept
        for name in phase_of:
            phase_of[name] = remap[phase_of[name]]
    return attached
