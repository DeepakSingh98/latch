# Adapted from frozen P7-A f36b4737cee732e6915d11e233bcc068126b3e64; private scheduling/resource protocol.
# Imported from P-OPS2-B d8f8c0d981354e3b81ef4c8671ae6cc2481a3af9.
"""P-OPS2-B rules for BF16 matrix execution and operand preparation (flat fused schedule, sm_90).

Selection uses only graph dependencies, dtype, static shapes, strides/contiguity, storage effects and the declared
target. No model or value names are inspected.

Matrix epilogue attachment
--------------------------
An elementwise register chain (members of one fused loop joined by same-index register reads) that reads the BF16
result of an admitted exact-order BF16 product at the identical flat index moves into that product's matrix phase as
a generated epilogue when every other operand it reads already exists before the product's phase (graph inputs or
values stored by earlier phases), in-place state commits keep every old-state read before the commit, and every
reader of the chain runs in a later phase. The epilogue evaluates each member's identical per-operation expression
on the rounded-and-widened BF16 result (exactly what a load of the stored result yields), so results are bitwise
identical to separate-phase execution. The product's own store is omitted when only epilogue members read it.
Phases left empty disappear together with their grid barrier.
"""
from __future__ import annotations

import math

from . import matrix_rules_p7a as rules

HEADER = 'pops2b_mm.cuh'

# Epilogue streaming model (H100, used only for the grouped CTA-slice choice; correctness never depends on it).
HBM_BYTES_PER_US = 2.6e6        # sustained aggregate DRAM bandwidth
EPI_SM_BYTES_PER_US = 30e3      # sustained per-SM epilogue stream with scattered fragment-order accesses
EPI_FIXED_US = 0.05


def _identity(layout, shape, idx, numel):
    return (idx == 'i' and shape is not None and math.prod(shape) == numel and layout.numel == numel
            and layout.contiguous and layout.offset == 0)


def permutation_inverse(layout, shape):
    """For a bijective permuted view of a contiguous root, return per-dimension (stride, extent) of the root.

    The member's element at contiguous index j (over `shape`) reads root flat p = sum(c_d * stride_d); given p, the
    coordinates are c_d = (p / stride_d) % extent_d. Returns None unless the view is such a permutation.
    """
    if layout.offset or math.prod(shape) != layout.numel or tuple(layout.shape) != tuple(shape):
        return None
    dims = sorted(((st, ex) for st, ex in zip(layout.strides, layout.shape) if ex != 1))
    expected = 1
    for stride, extent in dims:
        if stride != expected:
            return None
        expected *= extent
    return list(zip(layout.strides, layout.shape))


def eligible_product(node, layouts, wg_epilogue=False):
    """BF16 exact-order small-tile product with a contiguous rank-2 result. P6-A: with wg_epilogue, aligned TMA
    warpgroup-MMA products with an even N also take (register-direct pair) epilogues."""
    spec = node.extra.get('matrix_spec') if node.extra else None
    if node.kind != 'gemm' or not spec or spec.get('dtype') != 'bfloat16' or spec.get('batch') != 1:
        return None
    if spec.get('edge') or spec.get('splitk'):
        return None             # masked-edge / split-K partial products keep their plain body
    if spec.get('wg') and not (wg_epilogue and spec.get('tma') and spec.get('pair') and rules.tma_allowed()):
        return None             # wgmma products keep their plain body unless the pair epilogue is admitted
    layout = layouts[node.name]
    if not layout.contiguous or layout.offset or layout.root != node.name or len(layout.shape) != 2:
        return None
    return spec


EPI_COST_LIMIT = 3              # summed per-element expression cost (inline-style units; transcendental = 4)


def attach_epilogues(*, items, loads_of, phase_of, phases, nodes, layouts, dtypes, in_place, inplace_source,
                     register_load, numel_of, permuted_copies=True, cost_of=None, cost_limit=None, stream_gate=True,
                     background=(), fixed=(), wg_epilogue=False):
    """background: deferred products running on idle blocks; their epilogues are off the critical path."""
    cost_limit = EPI_COST_LIMIT if cost_limit is None else cost_limit
    """Move qualifying register chains into matrix epilogues. Mutates phases/phase_of; returns epilogue maps."""
    epilogue_of, epilogues, permuted = {}, {}, {}
    order = {node.name: k for k, node in enumerate(items)}
    readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            readers.setdefault(root, []).append((node.name, layout, shape, idx))
    commit_phase = {target: phase_of[source] for target, source in in_place.items()}

    def component(start):
        """Register-connected loop members of start's (phase, numel) loop."""
        q, n = phase_of[start], numel_of(nodes[start].shape)
        members = set(phases[q]['loops'].get(n, ()))
        group, frontier = {start}, [start]
        while frontier:
            current = frontier.pop()
            neighbours = []
            for root, layout, shape, idx in loads_of[current]:
                if root in members and shape is not None and register_load(nodes[current], root, layout, shape, idx):
                    neighbours.append(root)
            for reader, layout, shape, idx in readers.get(current, ()):
                if reader in members and shape is not None and register_load(nodes[reader], current, layout, shape, idx):
                    neighbours.append(reader)
            for other in neighbours:
                if other not in group:
                    group.add(other)
                    frontier.append(other)
        return sorted(group, key=order.get)

    def valid(chain, product, p, n):
        members = set(chain)
        # Epilogue work runs only on the product's CTA slice; expression-heavy chains (transcendentals) keep the
        # whole grid in their own phase.
        if product not in background and cost_of is not None and sum(cost_of(name) for name in chain) > cost_limit:
            return False
        for name in chain:
            node = nodes[name]
            if node.kind != 'elementwise' or node.dtype not in ('float32', 'bfloat16') or numel_of(node.shape) != n:
                return False
            for root, layout, shape, idx in loads_of[name]:
                if root in members and shape is not None and register_load(node, root, layout, shape, idx):
                    continue
                if root == product:
                    if not _identity(layout, shape, idx, n):
                        return False
                    continue
                if epilogue_of.get(root) == product and root not in permuted and _identity(layout, shape, idx, n):
                    continue
                if dtypes.get(root) not in ('float32', 'bfloat16'):
                    return False
                if root in commit_phase:
                    source = in_place[root]
                    if source in members or epilogue_of.get(source) == product:
                        # Old state at the same index: every epilogue load precedes its stores.
                        if not _identity(layout, shape, idx, n):
                            return False
                    elif not commit_phase[root] > p:
                        return False
                    continue
                if root not in phase_of:
                    continue                                   # graph input
                if phase_of[root] >= p:
                    return False
            if name in inplace_source:
                target = inplace_source[name]
                for reader, layout, shape, idx in readers.get(target, ()):
                    if reader in members:
                        continue
                    if epilogue_of.get(reader) == product and _identity(layout, shape, idx, n):
                        continue
                    if phase_of[reader] >= p:
                        return False
            for reader, layout, shape, idx in readers.get(name, ()):
                if reader not in members and phase_of[reader] <= p:
                    return False
        if stream_gate and product not in background and any(name in inplace_source for name in chain):
            # A state-streaming chain (in-place commits) is attached only when its modeled streaming time fits within
            # the product's own mainloop; longer streams are left to idle-block placement.
            spec = nodes[product].extra['matrix_spec']
            plan_ = rules._plan(spec, rules.TARGET_GRID)
            loaded = {root for name in chain for root, layout, shape, idx in loads_of[name]
                      if root != product and root not in members and _identity(layout, shape, idx, n)}
            nbytes = sum(SIZES.get(dtypes.get(r), 4) for r in loaded) + sum(SIZES.get(nodes[m].dtype, 4) for m in chain)
            active = max(1, min(rules.TARGET_GRID, plan_['tiles']))
            stream_us = n * nbytes / min(HBM_BYTES_PER_US, active * EPI_SM_BYTES_PER_US)
            if stream_us > plan_['local_us']:
                return False
        return True

    def permuted_member(reader, product, p, n):
        """A lone loop member that only copies a bijectively permuted view of the product (e.g. a head layout)."""
        node = nodes[reader]
        if node.kind != 'elementwise' or node.dtype not in ('float32', 'bfloat16') or numel_of(node.shape) != n:
            return None
        loads = loads_of[reader]
        if not loads or any(root != product for root, _, _, _ in loads):
            return None
        if component(reader) != [reader] or len(node.loads or ()) != 1:
            return None
        view = layouts[node.loads[0]]
        source = view.root
        if source != product:
            # Through one inlined, identity-indexed elementwise producer of the product (e.g. the widening cast).
            inner = nodes.get(source)
            if (inner is None or source in phase_of or inner.kind != 'elementwise' or len(inner.loads or ()) != 1
                    or numel_of(inner.shape) != n):
                return None
            inner_view = layouts[inner.loads[0]]
            if inner_view.root != product or not inner_view.contiguous or inner_view.offset or inner_view.numel != n:
                return None
            if not layouts[source].contiguous:
                return None
        inverse = permutation_inverse(view, node.shape)
        if not inverse:
            return None
        if reader in inplace_source:
            return None
        if any(phase_of[r] <= p for r, _, _, _ in readers.get(reader, ())):
            return None
        return inverse

    products = sorted((node.name for node in items if eligible_product(node, layouts, wg_epilogue)),
                      key=lambda g: phase_of[g])
    for product in products:
        p = phase_of[product]
        spec = nodes[product].extra['matrix_spec']
        n = spec['m'] * spec['n']
        progress = True
        while progress:
            progress = False
            for reader, layout, shape, idx in readers.get(product, ()):
                if (reader in epilogue_of or reader in fixed or nodes[reader].kind != 'elementwise'
                        or phase_of[reader] <= p):
                    continue
                if not _identity(layout, shape, idx, n) or numel_of(nodes[reader].shape) != n:
                    inverse = permuted_member(reader, product, p, n) if permuted_copies else None
                    if inverse is None:
                        continue
                    permuted[reader] = inverse
                    chain = [reader]
                else:
                    chain = component(reader)
                    if not valid(chain, product, p, n):
                        continue
                q = phase_of[reader]
                loop = phases[q]['loops'][n]
                for name in chain:
                    loop.remove(name)
                    phase_of[name] = p
                    epilogue_of[name] = product
                    if name in inplace_source:
                        commit_phase[inplace_source[name]] = p
                if not loop:
                    del phases[q]['loops'][n]
                epilogues.setdefault(product, []).extend(chain)
                epilogues[product].sort(key=order.get)
                phases[p].setdefault('epilogue', {})[product] = epilogues[product]
                progress = True
                break
    empty = [k for k, ph in enumerate(phases) if not ph['heavy'] and not ph['loops'] and not ph.get('epilogue')]
    if empty:
        remap, kept = {}, []
        for k, ph in enumerate(phases):
            if ph['heavy'] or ph['loops'] or ph.get('epilogue'):
                remap[k] = len(kept)
                kept.append(ph)
        phases[:] = kept
        for name in phase_of:
            phase_of[name] = remap[phase_of[name]]
    return epilogue_of, epilogues, permuted


# ------------------------------------------------------------------------------------------------ operand casts
CAST_COST_LIMIT = 2             # recomputed expression work per staged element (inline-style cost units)
CAST_SOURCE_LIMIT = 2           # identity-indexed sources staged per absorbed cast


def select_absorbed_casts(*, items, loads_of, nodes, layouts, dtypes, protected, mutable, cost_of):
    """BF16 elementwise casts computed in exact-order product operand staging.

    recompute: every reader is an admitted BF16 product reading the cast through exactly one operand view and the
      cast reads no mutable state; each reader computes the cast in its staging and the cast has no storage.
    materialize: the first reader (operation order) is such a product, every other reader comes later; the first
      reader's staging also stores the rounded operand once (cast reuse) and later readers load it.
    Casts that could be a product epilogue are left to the epilogue rule. Returns {cast: {mode, readers}}.
    """
    order = {node.name: k for k, node in enumerate(items)}
    readers = {}
    for node in items:
        for root, layout, shape, idx in loads_of[node.name]:
            readers.setdefault(root, []).append(node.name)

    def operand_of(product, name):
        node = nodes[product]
        spec = node.extra.get('matrix_spec') if node.extra else None
        if node.kind != 'gemm' or not spec or spec.get('dtype') != 'bfloat16' or spec.get('edge') or spec.get('wg'):
            return None
        operands = [key for key in ('a', 'b') if layouts[spec[key]].root == name]
        if len(operands) != 1:
            return None
        view = layouts[spec[operands[0]]]
        return operands[0] if view.offset == 0 and view.numel == layouts[name].numel else None

    chosen, taken = {}, set()
    for node in items:
        name = node.name
        if (node.kind != 'elementwise' or node.dtype != 'bfloat16' or name in protected
                or layouts[name].root != name or not layouts[name].contiguous or cost_of(name) > CAST_COST_LIMIT):
            continue
        sources = {root for root, _, _, _ in loads_of[name]}
        if any(dtypes.get(root) not in ('float32', 'bfloat16') for root in sources):
            continue
        # Every source must be stageable by cp.async: read at the cast's own flat index (no broadcast/gather loads,
        # which would put synchronous global latency inside every K-stage).
        size = layouts[name].numel
        if len(sources) > CAST_SOURCE_LIMIT or not all(
                idx == 'i' and shape is not None and math.prod(shape) == size and layout.numel == size
                and layout.contiguous and layout.offset == 0 for _, layout, shape, idx in loads_of[name]):
            continue
        # A cast that could be a product epilogue (it reads a product result and otherwise only older values) is
        # computed once there instead of being recomputed by every consumer tile's staging.
        if any(root in nodes and nodes[root].kind == 'gemm'
               and all(other == root or other not in order or order[other] < order[root] for other in sources)
               for root in sources):
            continue
        users = sorted(set(readers.get(name, ())), key=order.get)
        if not users:
            continue
        operands = [operand_of(user, name) for user in users]
        if any((user, which) in taken for user, which in zip(users, operands) if which):
            continue
        if all(operands) and not sources & mutable:
            chosen[name] = {'mode': 'recompute', 'readers': list(zip(users, operands))}
            taken.update(zip(users, operands))
        elif operands[0] and all(order[user] > order[users[0]] for user in users[1:]):
            chosen[name] = {'mode': 'materialize', 'readers': [(users[0], operands[0])]}
            taken.add((users[0], operands[0]))
    return chosen


# ------------------------------------------------------------------------------------------------ emission
# Dynamic shared memory admitted per CTA for matrix stage rings: the integrated base's maximum (P-OPS-C 128x64x5
# ring). Larger carve-outs shrink L1 for every phase of the single kernel (P-OPS-D measured the penalty).
SHARED_BUDGET = 133120
SIZES = {'float32': 4, 'bfloat16': 2}
CONVERT_CYCLES = 24             # per converted element pair and thread (smem loads, expression, cvt, smem store)


def _src_bytes(dtypes, rows, rk, ks):
    total = 0
    for dtype in dtypes:
        size = SIZES[dtype]
        pad = 16 // size
        elements = rows * (ks + pad) if rk else ks * (rows + pad)
        total += (elements * size + 15) // 16 * 16
    return total


# Diagnostic-only overrides keyed by emitted output identifier (microbenchmarks): {"bm", "bn", "ks", "nst"}.
# Empty in every compiler use; the selected implementation never depends on it.
FORCE_PLAN: dict = {}
STAGE_CAP = 4                   # ring depth cap for K slices longer than four steps (P-OPS-C measurement)


def tile_plan(spec, grid, extra=None):
    """BF16 tile/stage plan with staged prologue sources, tile epilogue staging and their costs (H100 model)."""
    extra = extra or {}
    m, n, k, batch, ak, bkc = spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc']
    force = FORCE_PLAN.get(spec['out'])
    a_src = extra['pa']['sources'] if extra.get('pa') else []
    b_src = extra['pb']['sources'] if extra.get('pb') else []
    epi_bytes = extra['epi']['bytes_per_element'] if extra.get('epi') else 0
    ks = 64 if k % 64 == 0 else 32
    if force and force.get('ks'):
        ks = int(force['ks'])
    nk = k // ks
    candidates = []
    for bm, bn, wm, wn in rules.BF16_TILES:
        if m % bm or n % bn or (force and (bm, bn) != (force['bm'], force['bn'])):
            continue
        operand = 2 * rules._bf16_stage_elements(bm, bn, ak, bkc, ks)
        stage = (operand + 15) // 16 * 16 + _src_bytes(a_src, bm, ak, ks) + _src_bytes(b_src, bn, bkc, ks)
        prefetch = sum((bm * bn * size + 15) // 16 * 16 for size in
                       (extra['epi'].get('prefetch_sizes', ()) if extra.get('epi') else ()))
        nst = min(nk + 1 if nk <= 4 else STAGE_CAP, (SHARED_BUDGET - prefetch) // stage)
        if force and force.get('nst'):
            nst = min(int(force['nst']), (SHARED_BUDGET - prefetch) // stage)
        if nst < 2:
            continue
        if extra.get('epi') and bm * (bn + 4) * 4 > nst * stage:
            continue
        mt, nt = (bm // wm) // 16, (bn // wn) // 8
        tiles, waves, local, traffic = rules._estimate(m, n, k, batch, bm, bn, grid, 2,
                                                       (k // 16) * max(rules.MMA_CHAIN_CYCLES, 4 + 3 * mt * nt),
                                                       rules.MMA_MACS_PER_CYCLE)
        pairs = (len(a_src) > 0) * bm * ks / 2 + (len(b_src) > 0) * bn * ks / 2
        convert = waves * nk * (pairs / 256) * CONVERT_CYCLES / rules.CLOCK_MHZ
        epilogue = 0.0
        if epi_bytes:
            active = max(1, min(grid, tiles))
            epilogue = EPI_FIXED_US + m * n * batch * epi_bytes / min(HBM_BYTES_PER_US, active * EPI_SM_BYTES_PER_US)
        local += convert + epilogue
        candidates.append({'bm': bm, 'bn': bn, 'wm': wm, 'wn': wn, 'ks': ks, 'nst': nst, 'tiles': tiles,
                           'waves': waves, 'grid': grid, 'shared_bytes': nst * stage + prefetch, 'stage_bytes': stage,
                           'convert_us': round(convert, 4), 'epilogue_us': round(epilogue, 4),
                           'local_us': round(local, 4), 'traffic_us': round(traffic, 4),
                           'estimate_us': round(local + traffic, 4)})
    if not candidates:
        raise rules.MatrixRuleError(f'no admitted BF16 tile within the shared budget for {(m, n, k)}')
    return min(candidates, key=lambda c: (c['estimate_us'], c['tiles'], -c['bm']))


def _code(spec, plan, ident, lo, hi, den, rot, extra):
    """bmm_phase2 call: optional epilogue and A/B operand prologues (with materializing side-store)."""
    sam, sak, sbk, sbn, bsa, bsb = spec['strides']
    m, n, k, batch = spec['m'], spec['n'], spec['k'], spec['batch']
    t = lambda flag: str(flag).lower()
    epi, pa, pb = extra.get('epi'), extra.get('pa'), extra.get('pb')
    store = epi['store'] if epi else True
    out = spec['out'] if store else 'nullptr'
    a_ptr = 'nullptr' if pa else ident(spec['a'])
    b_ptr = 'nullptr' if pb else ident(spec['b'])
    call = (f'pops2b::bmm_phase2<{plan["bm"]},{plan["bn"]},{plan["wm"]},{plan["wn"]},{t(spec["ak"])},'
            f'{t(spec["bkc"])},{plan["ks"]},{plan["nst"]},{t(store)},{t(bool(pa))},{t(bool(pb))}>')
    lines = ['{',
             f'  popsc::BArgs g{{{a_ptr}, {sam}LL, {sak}LL, {bsa}LL, {b_ptr}, {sbk}LL, {sbn}LL, {bsb}LL,',
             f'              {out}, {n}LL, {m*n}LL, {m}, {n}, {k}, {batch}}};',
             f'  const auto epi = {epi["code"] if epi else "pops2b::NoEpi{}"};',
             f'  const auto pa = {pa["code"] if pa else "pops2b::NoPro{}"};',
             f'  const auto pb = {pb["code"] if pb else "pops2b::NoPro{}"};',
             f'  {call}(g, smem, epi, pa, pb, {pa["offset"] if pa else 0}LL, {pb["offset"] if pb else 0}LL, '
             f'{pa["out"] if pa else "nullptr"}, {pb["out"] if pb else "nullptr"}, {lo}, {hi}, {den}, {rot});',
             '}']
    return '\n'.join(lines)


def emit_group(specs, extras, *, ident, grid=None, plans=False):
    grid = rules.TARGET_GRID if grid is None else grid
    """matrix_rules.emit_group with generated epilogues/prologues; slice choice includes epilogue streaming time.

    plans=True selects this lane's tile planner (and bmm_phase2 body) for every product of the phase."""
    if not any(extras) and not plans:
        return rules.emit_group(specs, ident=ident, grid=grid)
    extras = [e or ({'plain': True} if plans and not (spec.get('edge') or spec.get('wg')) else {}) for spec, e in zip(specs, extras)]
    def planner(spec, g, e):
        if e and not spec.get('wg'):
            return tile_plan(spec, g, e)
        if e and e.get('epi') and e['epi'].get('vec') and rules.WG_VEC_EPI_MAX_WTN:
            # P7-A: vector staged epilogues hold no accumulators; wider warpgroup tiles, staging bytes reserved.
            ncs = max(1, int(e['epi'].get('colsum') or 0))

            def staging(bm, bn, wm):
                wtn, wtm = bn // (2 // wm), bm // wm
                return 2048 + 2 * 4 * ncs * wtn * 4 + 2 * wtm * (wtn + 8) * 2
            return rules.wg_plan(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'], grid=g,
                                 max_wtn=rules.WG_VEC_EPI_MAX_WTN, extra_bytes=staging, wtn_set=(64, 128, 256))[0]
        if e and e.get('epi') and rules.WG_EPI_MAX_WTN:
            # P6-A: epilogue-carrying warpgroup products use narrower warpgroup tiles (register pressure).
            return rules.wg_plan(spec['m'], spec['n'], spec['k'], spec['batch'], spec['ak'], spec['bkc'], grid=g,
                                 max_wtn=rules.WG_EPI_MAX_WTN)[0]
        return rules._plan(spec, g)
    whole = [planner(spec, grid, e) for spec, e in zip(specs, extras)]
    best = ('serial', sum(p['estimate_us'] for p in whole), None)
    if rules.GROUP_PARTITION and 2 <= len(specs) <= 3:
        step = grid // 16
        shares = range(step, grid, step)
        options = ([(g, grid - g) for g in shares] if len(specs) == 2 else
                   [(g1, g2, grid - g1 - g2) for g1 in shares for g2 in shares if g1 + g2 < grid])
        for option in options:
            plans = [planner(spec, g, e) for spec, g, e in zip(specs, option, extras)]
            us = max(p['local_us'] for p in plans) + sum(p['traffic_us'] for p in plans)
            if us < best[1]:
                best = ('partition', us, (option, plans))
    plain = lambda spec, plan, lo, hi, den, rot: rules._code(spec, plan, ident, lo, hi, den, rot)
    blocks, members, rot = [], [], 0
    chosen = (zip(specs, whole, [None] * len(specs)) if best[0] == 'serial' else zip(specs, best[2][1], best[2][0]))
    lo = 0
    chosen = list(chosen)
    for idx_, ((spec, plan, g), extra) in enumerate(zip(chosen, extras)):
        slice_ = [0, 1, 1] if g is None else [lo, lo + g, grid]
        r = rot if g is None else 0
        dyn = bool(rules.WG_DYN_TILES and g is None and idx_ == len(chosen) - 1 and idx_ > 0 and spec.get('wg')
                   and spec.get('tma') and rules.tma_allowed() and not spec.get('splitk')
                   and not (extra and extra.get('epi') and (extra['epi'].get('ld_bytes') or extra['epi'].get('st_bytes'))))
        if extra and spec.get('wg'):
            blocks.append(rules._code(spec, plan, ident, *slice_, r, epi=extra.get('epi'), dyn=dyn))
        elif dyn and not extra:
            blocks.append(rules._code(spec, plan, ident, *slice_, r, dyn=True))
        else:
            blocks.append(_code(spec, plan, ident, *slice_, r, extra) if extra else plain(spec, plan, *slice_, r))
        record = rules._member_record(spec, plan, slice_)
        if extra.get('epi'):
            epi = extra['epi']
            record.update(engine=record['engine'] + '_epi', epilogue=epi['members'], epilogue_store=epi['store'],
                          arithmetic=record['arithmetic'] + '; generated epilogue on the rounded BF16 result '
                          '(identical per-operation expressions)')
        for key in ('pa', 'pb'):
            if extra.get(key):
                record[{'pa': 'a_prologue', 'pb': 'b_prologue'}[key]] = extra[key]['cast']
        members.append(record)
        if g is None:
            rot += plan['tiles']
        else:
            lo += g
    engine = members[0]['engine'] if len(members) == 1 else f'popsc_group_{best[0]}'
    extra = {'tiles': sum(m['matrix_plan']['tiles'] for m in members), 'engine': engine,
             'batch': members[0]['batch'], 'shared_bytes': max(m['matrix_plan']['shared_bytes'] for m in members),
             'members': members, 'group_schedule': best[0], 'group_estimate_us': round(best[1], 4),
             'matrix_shape': members[0]['matrix_shape'], 'matrix_plan': members[0]['matrix_plan'],
             'operand_layout': members[0]['operand_layout'], 'arithmetic': members[0]['arithmetic']}
    return '\n'.join(blocks), extra


# ------------------------------------------------------------------------------------------------ L2 prefetch
PREFETCH_MIN_BYTES = 64 * 1024        # smaller operands are not worth an issue loop
PREFETCH_PHASE_BYTES = 24 * 1024**2   # per issuing phase, well under the 50 MB H100 L2
PREFETCH_CHUNK = 4096                 # bytes per cp.async.bulk.prefetch.L2 instruction (16-byte multiple)


def prefetch_plan(phase_records, inputs, layouts, dtypes, sizes):
    """{issuing phase k: [(input root, bytes, chunk)]} for graph-input operands read by phase k+1.

    Graph inputs (parameters, optimizer state, data) are read-only or committed in place by their final reader,
    so prefetching them one phase early never observes a different value; it only warms L2. Inputs already read
    by phase k itself are skipped (they are being fetched anyway).
    """
    plan = {}
    input_set = set(inputs)
    for k in range(len(phase_records) - 1):
        if not any(str(member.get('engine', '')).startswith('popsc_bf16')
                   for member in phase_records[k + 1].get('members', ())):
            continue                          # staging rule of BF16 matrix phases only
        current = set(phase_records[k]['reads'])
        chunks, total = [], 0
        for root in phase_records[k + 1]['reads']:
            if root not in input_set or root in current or layouts[root].root != root:
                continue
            nbytes = layouts[root].numel * sizes[dtypes[root]]
            if nbytes < PREFETCH_MIN_BYTES or nbytes % 16 or total + nbytes > PREFETCH_PHASE_BYTES:
                continue
            chunks.append((root, nbytes, PREFETCH_CHUNK))
            total += nbytes
        if chunks:
            plan[k] = chunks
    return plan


# ------------------------------------------------------------------------------------------------ idle blocks
# Heavy phase kinds whose generated code assigns item k to block k (items on blocks [0, n), grid-strided): blocks at or
# above n are idle and can run independent elementwise loops. Durations are an H100 fit of the integrated attention
# bodies at sequence 64 / head width 64 (base profile), scaled by (t / 64)^2; used for placement only.
IDLE_HEAVY_KINDS = ('attention_prepare', 'attention_forward', 'attention_delta', 'attention_backward')
IDLE_HEAVY_US = {'attention_prepare': 7.3, 'attention_forward': 14.1, 'attention_delta': 12.4,
                 'attention_backward': 33.0}


IDLE_FILL = 1.0                 # share of the modeled heavy duration offered to idle-block loops


def idle_heavy_model(fill=None):
    import re
    fill = IDLE_FILL if fill is None else fill

    def duration(kind):
        def model(node):
            match = re.search(r'A\.t = (\d+);', node.code or '')
            t = int(match.group(1)) if match else 64
            return fill * IDLE_HEAVY_US[kind] * (t / 64) ** 2
        return model
    return {kind: duration(kind) for kind in IDLE_HEAVY_KINDS}


# ------------------------------------------------------------------------------------------------ deferred products
DEFER_FILL = 0.85               # share of an idle-capable phase's modeled duration given to deferred products


def defer_products(*, items, loads_of, phase_of, phases, nodes, idle_model, numel_of, grid=None,
                   inplace_source=None, dtypes=None):
    grid = rules.TARGET_GRID if grid is None else grid
    """Move off-critical-path BF16 products into later item-per-block heavy phases (their idle CTA slice).

    A product qualifies when every transitive reader is elementwise (gradient outputs, optimizer chains: sinks with
    no heavy consumer). It moves to the first later idle-capable phase whose modeled heavy duration covers the
    product's modeled time on the idle blocks; its reader chains move after it. Values and arithmetic are
    unchanged; only the phase and CTA slice change. Returns ({product: {phase, slice, plan_grid}}, {phase: us}).
    """
    readers = {}
    for node in items:
        for root, _, _, _ in loads_of[node.name]:
            readers.setdefault(root, set()).add(node.name)
    order = {node.name: k for k, node in enumerate(items)}
    deferred, preload = {}, {}
    products = sorted((node.name for node in items if node.kind == 'gemm' and node.extra
                       and (node.extra.get('matrix_spec') or {}).get('dtype') == 'bfloat16'),
                      key=lambda g: phase_of[g])
    for product in products:
        closure, frontier, ok = set(), list(readers.get(product, ())), bool(readers.get(product))
        while frontier and ok:
            name = frontier.pop()
            if name in closure:
                continue
            if nodes[name].kind != 'elementwise':
                ok = False
                break
            closure.add(name)
            frontier.extend(readers.get(name, ()))
        if not ok:
            continue
        spec = nodes[product].extra['matrix_spec']
        start = phase_of[product]
        target = None
        for q in range(start + 1, len(phases)):
            heavy = [nodes[h] for h in phases[q]['heavy']]
            main = [h for h in heavy if h.name not in deferred]
            if not main or not all(h.kind in idle_model and h.extra.get('work_items') for h in main):
                continue
            busy = sum(int(h.extra['work_items']) for h in main)
            if 2 * busy > grid:
                continue
            idle = grid - busy
            heavy_us = max(idle_model[h.kind](h) for h in main)
            plan_ = rules._plan(spec, idle)
            # State-streaming readers (optimizer chains) also run on the idle blocks, as the product's epilogue.
            stream = 0.0
            if inplace_source and any(name in inplace_source for name in closure):
                roots = {r for name in closure for r, _, _, _ in loads_of[name] if r != product and r not in closure}
                nbytes = sum(SIZES.get((dtypes or {}).get(r), 4) for r in roots) + sum(
                    SIZES.get(nodes[name].dtype, 4) for name in closure)
                stream = spec['m'] * spec['n'] * nbytes / min(HBM_BYTES_PER_US, idle * EPI_SM_BYTES_PER_US)
            if preload.get(q, 0.0) + plan_['estimate_us'] + stream <= DEFER_FILL * heavy_us:
                target = (q, busy, idle, plan_['estimate_us'] + stream)
                break
        if target is None:
            continue
        q, busy, idle, us = target
        phases[start]['heavy'].remove(product)
        phases[q]['heavy'].append(product)
        phase_of[product] = q
        while q + 1 >= len(phases):
            phases.append({'heavy': [], 'loops': {}})
        for name in sorted(closure, key=order.get):
            if phase_of[name] <= q:
                numel = numel_of(nodes[name].shape)
                loop = phases[phase_of[name]]['loops'][numel]
                loop.remove(name)
                if not loop:
                    del phases[phase_of[name]]['loops'][numel]
                phases[q + 1]['loops'].setdefault(numel, []).append(name)
                phases[q + 1]['loops'][numel].sort(key=order.get)
                phase_of[name] = q + 1
        deferred[product] = {'phase': q, 'slice': [busy, grid, grid], 'plan_grid': idle, 'estimate_us': us}
        preload[q] = preload.get(q, 0.0) + us
    # Phases left without work disappear (with their barrier).
    keep = [k for k, ph in enumerate(phases) if ph['heavy'] or ph['loops'] or ph.get('epilogue')]
    if len(keep) != len(phases):
        remap = {old: new for new, old in enumerate(keep)}
        phases[:] = [phases[k] for k in keep]
        for name in phase_of:
            phase_of[name] = remap[phase_of[name]]
        for info in deferred.values():
            info['phase'] = remap[info['phase']]
        preload = {remap[k]: v for k, v in preload.items() if k in remap}
    return deferred, preload
