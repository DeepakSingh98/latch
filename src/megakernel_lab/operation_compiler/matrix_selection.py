"""Release step 13: structural matrix-body groups, predeclared candidate sets and bounded mixed choices.

This module only *names* choices among existing, resource-checked bodies (``matrix_adapters.candidates``); it
writes no CUDA and plans no schedule. ``ops.matrix_tuning`` builds and measures the resulting complete updates.

Vocabulary
----------
product row   one entry of the provider inventory (``matrix_adapters.provider.MatrixProvider._choose``): problem
              key, native engine (``p5c`` / ``popsc`` / ``fallback``), the default body, admitted bodies and flops.
family        an existing body class applied to one product:
              ``D`` the default selection;
              ``W`` masked TMA / cp.async warpgroup ``wgmma`` (P-OPS4-B ``pops4b_tma`` / ``pops4b_wg``);
              ``S`` ``mma.sync`` (the native popsc body, else ``popsc_bf16`` / ``_tail`` / ``_edge``);
              ``P`` the P-OPS5-C engine (native P5C, else ``p8ws`` / ``p5c`` where admitted).
group         products with the same native engine, operand orientation (A/B row or column major) and flop
              bucket (factor-4 bands).  Groups are purely structural: no model names, values or seeds.

Two choice generators use these groups:

``candidate_set``  the predeclared, bounded evaluation set (uniform families, single-group deviations from the
                   D/W/S uniform bases, size splits and orientation splits; at most ``cap`` distinct maps).
``MixedSearch``    the bounded automatic search: uniform families first, then single-group deviations of the
                   largest groups from the fastest base, then one combination of the improving deviations.
                   Every round is judged by complete-update time, never by an isolated product timing.
"""
from __future__ import annotations

from itertools import permutations
import json
import math

FAMILIES = ('D', 'W', 'S', 'P')
WGMMA_BODIES = ('pops4b_tma', 'pops4b_wg')
MMA_SYNC_BODIES = ('popsc_bf16', 'popsc_bf16_tail', 'popsc_bf16_edge')
ENGINE_BODIES = ('p8ws', 'p5c')
BUCKET_BASE = 26                 # flop bands [2^26, 2^28), [2^28, 2^30), ... (factor 4)


# ------------------------------------------------------------------------------------------------ structure
def orientation(row):
    """'NN' / 'TN' / 'NT' / 'TT': A (M x K) and B (K x N) row-major (N) or transposed (T) storage."""
    key = json.loads(row['key'])
    a, b = key[4], key[5]
    return ('N' if a[1] == 1 else 'T') + ('N' if b[1] == 1 else 'T')


def bucket(flops):
    return max(0, (int(math.log2(max(flops, 1))) - BUCKET_BASE) // 2)


def family_body(row, family):
    """The existing body ``family`` names for this product, or None when the family has no admitted member."""
    admitted, native = set(row['admitted']), row['native']
    if family == 'D':
        return row['default']
    if family == 'W':
        for body in (WGMMA_BODIES if row['batch'] == 1 else ('pops4b_wg',)):
            if body in admitted:
                return body
        return None
    if family == 'S':
        if native == 'popsc':
            return 'native'
        return next((b for b in MMA_SYNC_BODIES if b in admitted), None)
    if family == 'P':
        if native == 'p5c':
            return 'native'
        return next((b for b in ENGINE_BODIES if b in admitted), None) if row['batch'] == 1 else None
    raise ValueError(family)


def family_of(row, body):
    """Inverse of ``family_body`` (the family a selected body belongs to)."""
    if body == 'native':
        return {'popsc': 'S', 'p5c': 'P'}.get(row['native'], 'D')
    if body in WGMMA_BODIES:
        return 'W'
    if body in MMA_SYNC_BODIES:
        return 'S'
    if body in ENGINE_BODIES:
        return 'P'
    return 'D'


def unique_rows(products):
    rows = {}
    for row in products:
        rows.setdefault(row['key'], row)
    return rows


def groups(products):
    """{group_id: {'keys': [...], 'flops': total, 'options': [families with a body differing in some product]}}.

    Products whose admitted bodies offer no alternative to the default are left out (they never change).
    Total flops count every occurrence of a product (repeated keys share one choice).
    """
    rows = unique_rows(products)
    count = {}
    for row in products:
        count[row['key']] = count.get(row['key'], 0) + 1
    out = {}
    for key, row in rows.items():
        bodies = {f: family_body(row, f) for f in FAMILIES}
        alternatives = [f for f in FAMILIES[1:] if bodies[f] is not None and bodies[f] != row['default']]
        if not alternatives:
            continue
        gid = f"{row['native']}:{orientation(row)}:b{bucket(row['flops'])}"
        g = out.setdefault(gid, {'keys': [], 'flops': 0.0, 'options': set(), 'outputs': []})
        g['keys'].append(key)
        g['outputs'].append(row['output'])
        g['flops'] += row['flops'] * count[key]
        g['options'].update(alternatives)
    for g in out.values():
        g['options'] = sorted(g['options'], key=FAMILIES.index)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]['flops']))


def assignment_choices(products, assign):
    """Resolve ``{group_id: family}`` (missing groups keep D) into the provider choice map ``{key: body}``.

    A family without a member for some product of the group leaves that product at its default.
    """
    rows = unique_rows(products)
    gmap = {}
    for gid, g in groups(products).items():
        for key in g['keys']:
            gmap[key] = gid
    choices = {}
    for key, row in rows.items():
        family = assign.get(gmap.get(key), 'D')
        body = family_body(row, family)
        if body is not None and body != row['default']:
            choices[key] = body
    return choices


def uniform(products, family):
    return {gid: family for gid in groups(products)}


def describe(products, choices):
    """Per-group family counts of a choice map (diagnostic, for receipts)."""
    rows = unique_rows(products)
    out = {}
    for gid, g in groups(products).items():
        fams = {}
        for key in g['keys']:
            fam = family_of(rows[key], choices.get(key, rows[key]['default']))
            fams[fam] = fams.get(fam, 0) + 1
        out[gid] = fams
    return out


# ------------------------------------------------------------------------------------------------ evaluation set
def candidate_set(products, *, cap=64):
    """Predeclared bounded evaluation set: [(label, assignment, choices)] with distinct choice maps.

    Order (truncated at ``cap``): uniform D/W/S/P; single-group deviations from the uniform D, W and S bases;
    size splits (products at or above a group-flop bucket boundary in one family, below it in another);
    orientation splits (NN products in one family, transposed-operand products in another). The first entry is
    always the default selection (empty choice map).
    """
    gs = groups(products)
    out, seen, overflow = [], set(), set()

    def add(label, assign):
        choices = assignment_choices(products, assign)
        identity = tuple(sorted(choices.items()))
        if identity in seen:
            return
        if len(out) >= cap:
            overflow.add(identity)
            return
        seen.add(identity)
        out.append((label, dict(assign), choices))

    for fam in FAMILIES:
        add(f'uniform:{fam}', uniform(products, fam))
    for base in ('D', 'W', 'S'):
        for gid, g in gs.items():
            for fam in FAMILIES:
                if fam != base:
                    add(f'{base}+{gid}={fam}', {**uniform(products, base), gid: fam})
    levels = sorted({int(gid.rsplit(':b', 1)[1]) for gid in gs})
    for level in levels[1:]:
        for big, small in permutations(FAMILIES, 2):
            add(f'size>=b{level}:{big}/{small}',
                {gid: big if int(gid.rsplit(':b', 1)[1]) >= level else small for gid in gs})
    for fwd, other in permutations(FAMILIES, 2):
        add(f'orient:NN={fwd}/T={other}', {gid: fwd if gid.split(':')[1] == 'NN' else other for gid in gs})
    return out, len(overflow)


# ------------------------------------------------------------------------------------------------ automatic search
class MixedSearch:
    """Bounded per-group search judged by complete-update times supplied by the caller.

    Rounds (each a batch of builds that the caller measures):
      1. uniform families ``first`` (default D, W, P: on the development suite uniform mma.sync was 1.17-1.39x
         slower on every program that uses the common planner, so S is tried only as a group alternative);
      2. from the fastest round-1 base, move each of the ``top_groups`` largest groups (by flops) to each of the
         ``alternatives`` best-ranked other families (families ranked by their round-1 uniform time);
         (skipped when every round-1 alternative is more than ``stop_margin`` slower than the default);
      3. if round-2 deviations of two or more groups improved on the base by at least ``min_step``, their union
         (the fastest improving family per group).
    At most ``max_builds`` distinct variants are built in total (round order decides what is dropped).
    """

    def __init__(self, products, *, top_groups=3, alternatives=2, min_step=0.005, max_builds=11,
                 first=('D', 'W', 'P'), stop_margin=0.10):
        self.products = products
        self.first = first
        self.stop_margin = stop_margin
        self.groups = groups(products)
        self.top_groups, self.alternatives = top_groups, alternatives
        self.min_step, self.max_builds = min_step, max_builds
        self.variants = {}            # label -> (assignment, choices)
        self._seen = {}
        self.round = 0
        self.rounds = []

    def _add(self, batch, label, assign):
        choices = assignment_choices(self.products, assign)
        identity = tuple(sorted(choices.items()))
        if identity in self._seen:
            return
        if len(self._seen) >= self.max_builds:
            return
        self._seen[identity] = label
        self.variants[label] = (dict(assign), choices)
        batch.append((label, choices))

    def next_round(self, times):
        """Return the next [(label, choices)] batch to build and measure ([] when finished).

        ``times`` maps every already measured label to its quick complete-update time (None: failed/incorrect).
        """
        batch = []
        self.round += 1
        if self.round == 1:
            for fam in self.first:
                self._add(batch, 'existing' if fam == 'D' else f'uniform:{fam}', uniform(self.products, fam))
        elif self.round == 2:
            valid = {l: t for l, t in times.items() if t is not None and l in self.variants}
            if not valid:
                return []
            base = min(valid, key=valid.get)
            others = [t for l, t in valid.items() if l != 'existing']
            if 'existing' in valid and others and min(others) > valid['existing'] * (1 + self.stop_margin):
                # Every uniform alternative is far slower (development suite: a program kept on the established
                # static emitter pays the common planner's cost for any provider body); no refinement round.
                self.stopped = 'every round-1 alternative > {:.0%} slower than the default'.format(self.stop_margin)
                self.rounds.append([])
                return []
            base_assign = self.variants[base][0]
            measured = [f for f in FAMILIES if (f'uniform:{f}' if f != 'D' else 'existing') in valid]
            rank = sorted(measured, key=lambda f: valid['existing' if f == 'D' else f'uniform:{f}'])
            rank += [f for f in FAMILIES if f not in measured]      # unmeasured families rank last
            self.base = base
            for gid in list(self.groups)[:self.top_groups]:
                current = base_assign.get(gid, 'D')
                options = [f for f in rank if f != current and (f == 'D' or f in self.groups[gid]['options'])]
                for fam in options[:self.alternatives]:
                    self._add(batch, f'{base}|{gid}={fam}', {**base_assign, gid: fam})
        elif self.round == 3:
            base = getattr(self, 'base', None)
            if base is None or times.get(base) is None:
                return []
            best = {}                     # group -> (time, family) of its fastest improving deviation
            for label in self.variants:
                if not label.startswith(base + '|') or label.endswith('|combined') or times.get(label) is None:
                    continue
                if times[label] > times[base] * (1 - self.min_step):
                    continue
                gid, fam = label.split('|', 1)[1].rsplit('=', 1)
                if gid not in best or times[label] < best[gid][0]:
                    best[gid] = (times[label], fam)
            if len(best) >= 2:
                merged = dict(self.variants[base][0])
                merged.update({gid: fam for gid, (_, fam) in best.items()})
                self._add(batch, f'{base}|combined', merged)
        self.rounds.append([label for label, _ in batch])
        return batch
