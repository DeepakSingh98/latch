"""Candidate enumeration, resource admission and selection policies.

``candidates(problem, resources)`` lists every existing body that structurally admits the problem, each with its
best donor plan, alternative plans, explicit requirements and the resource-check verdict. ``select`` applies one
named policy; policies read only operation properties (shape, strides, dtypes, result mode) and program resources.

Policies
--------
``frozen_reference``  the decisions the frozen ab4703ee3 reference schedule makes (popsc 32-aligned body, popsc
                      tail for 16-multiples, otherwise the generic fallback; FP32 via P-OPS2-A). Used to prove the
                      adapter reproduces the frozen compiler byte for byte.
``donor``             the union of already-qualified donor admission rules, in donor precedence: the flat P-OPS5-C
                      rule (p8ws, else p5c) for aligned products it admits; P-OPS4-B masked-edge wgmma for awkward
                      products of at least 2^30 flops (TMA form for batch 1, cp.async form for batches); the
                      mma.sync masked-edge body for smaller awkward products; then the frozen-reference bodies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bodies import BODIES, BY_NAME, BODY_INFO, Refusal
from .problem import MatrixProblem
from .requirements import ProgramResources, Requirements, check

POLICIES = ('frozen_reference', 'donor', 'step13')
# Default policy of the composed planner's provider (compose.lower_composed_flat); ``matrix_choices(policy=...)``
# overrides it for one lowering (e.g. 'donor' = frozen rule).
COMPOSE_DEFAULT_POLICY = 'donor'
# Step 13 structural default: when the frozen-rule program already needs the common planner, the program is
# lowered once more with this policy (compose.step13_common_upgrade). None disables the rule.
COMPOSE_COMMON_POLICY = 'step13'
STEP13_REPLACED = ('popsc_bf16', 'popsc_bf16_tail', 'popsc_bf16_edge', 'mixed_generic')
# Development suite: products of 2^28-2^30 flops gained from wgmma inside the complete update, while moving the
# [2^26, 2^28) groups back to mma.sync was 1.1-1.4% faster; smaller products were not measured, so they keep
# the frozen rule.
STEP13_MIN_FLOPS = 2 ** 28
# Store-bound rule from 773d6bec7, measured separately in research/latch_matrix_moesup:
# batch-1 products with few 64-wide K stages and a large output take p8ws when admitted.
STORE_BOUND_WS = True
STORE_BOUND_MAX_K_STAGES = 8
STORE_BOUND_MIN_OUTPUT = 2 ** 28
WG_EDGE_MIN_FLOPS = 2 ** 30      # matrix_rules_p7b.WG_MIN_FLOPS (P-OPS4-B gpu-07 selected default)
WG_ALIGNED_MIN_FLOPS = 2 ** 28   # performance_defaults.P7B_SELECTED WG_MIN_FLOPS with WGMMA_ALIGNED (practical)


@dataclass
class Candidate:
    body: str
    problem: MatrixProblem
    plan: dict
    requirements: Requirements
    admitted: bool
    reasons: list
    alternatives: list = field(default_factory=list)
    donor_rule: str = ''

    @property
    def estimate_us(self):
        return self.plan.get('estimate_us')

    def emit(self, *, ident, a, b, out, lo=0, hi=1, den=1, rot=0, ldc=None):
        if not self.admitted:
            raise Refusal(f'{self.body} not admitted: {self.reasons}')
        extra = {'ldc': ldc} if ldc is not None else {}
        return BY_NAME[self.body].emit(self.problem, self.plan, ident=ident, a=a, b=b, out=out,
                                       lo=lo, hi=hi, den=den, rot=rot, **extra)

    def record(self):
        return {'body': self.body, 'admitted': self.admitted, 'reasons': self.reasons,
                'plan': {k: v for k, v in self.plan.items() if not isinstance(v, (list, dict)) or k == 'padded'},
                'alternatives': self.alternatives, 'requirements': self.requirements.to_dict(),
                'donor_rule': self.donor_rule, 'provenance': BODY_INFO.get(self.body, {})}


def candidates(problem: MatrixProblem, resources: ProgramResources = ProgramResources(), bodies=BODIES):
    """Every structurally admissible body with its best admitted plan (or the refusal reason)."""
    result, refused = [], {}
    for body in bodies:
        ok, why = body.structural(problem)
        if not ok:
            refused[body.name] = why
            continue
        try:
            plans = body.plans(problem, resources.grid if resources.grid else 128, resources.dynamic_shared_limit)
        except Refusal as error:
            refused[body.name] = str(error)
            continue
        best, best_reasons, admitted = None, None, False
        for plan in plans:
            req = body.requirements(problem, plan)
            ok, reasons = check(req, resources)
            if ok:
                best, best_reasons, admitted = (plan, req), [], True
                break
            if best is None:
                best, best_reasons = (plan, req), reasons
        plan, req = best
        alts = [{k: p.get(k) for k in ('bm', 'bn', 'body', 'nst', 'tiles', 'shared_bytes', 'estimate_us') if k in p}
                for p in plans if p is not plan][:4]
        result.append(Candidate(body.name, problem, plan, req, admitted, best_reasons, alts))
    return result, refused


def _store_bound_p5c(cands, problem, resources):
    """matrix_bodies 'p5c' only: a P5C candidate with a 256 x 128 or 128 x 256 plan of >= 4 stages, stage release."""
    from .. import matrix_bodies
    if not matrix_bodies.active('p5c'):
        return None
    base = _first(cands, 'p5c')
    if base is None:
        return None
    body = BY_NAME['p5c']
    try:
        plans = body.plans(problem, resources.grid if resources.grid else 128, resources.dynamic_shared_limit)
    except Refusal:
        return None
    for tile in ((256, 128), (128, 256)):
        plan = next((x for x in plans if (x['bm'], x['bn']) == tile and x['nst'] >= 4), None)
        if plan is None:
            continue
        plan = dict(plan, release=True, matrix_bodies='p5c')
        req = body.requirements(problem, plan)
        ok, reasons = check(req, resources)
        if ok:
            return Candidate('p5c', problem, plan, req, True, [], base.alternatives)
    return None


def _first(cands, *names):
    for name in names:
        for c in cands:
            if c.body == name and c.admitted:
                return c
    return None


def _exact_bf16_tensor():
    from ..lower import _EXACT_BF16_PRODUCTS
    return _EXACT_BF16_PRODUCTS.get() == 'tensor'


def select(problem: MatrixProblem, resources: ProgramResources = ProgramResources(), policy='donor'):
    """Return (candidate | None, decision record)."""
    if policy not in POLICIES:
        raise ValueError(f'unknown policy {policy}')
    cands, refused = candidates(problem, resources)
    by = {c.body: c for c in cands}
    chosen, rule = None, ''
    if problem.dtype == 'float32':
        chosen, rule = _first(cands, 'fp32_pops2'), 'strict FP32 P-OPS2-A body selection'
    elif policy == 'frozen_reference':
        chosen = _first(cands, 'popsc_bf16', 'popsc_bf16_tail', 'mixed_generic')
        rule = 'frozen reference order: 32-aligned popsc, 16-multiple tail, generic fallback'
    else:
        aligned32 = not (problem.m % 32 or problem.n % 32 or problem.k % 32)
        p5c_ok = 'p5c' in by and by['p5c'].admitted and BY_NAME['p5c'].donor_admitted(problem)
        if p5c_ok:
            chosen = _first(cands, 'p8ws', 'p5c')
            rule = 'flat P-OPS5-C admission (FLAT_SELECTED_DEFAULTS): warp-specialized body when its plan exists'
        elif not aligned32 and problem.flops >= WG_EDGE_MIN_FLOPS:
            chosen = _first(cands, 'pops4b_tma', 'pops4b_wg')
            rule = 'P-OPS4-B masked-edge wgmma for awkward products >= 2^30 flops'
        elif problem.batch > 1 and problem.flops >= WG_ALIGNED_MIN_FLOPS:
            chosen = _first(cands, 'pops4b_tma', 'pops4b_wg')
            rule = ('P7-B practical WGMMA_ALIGNED rule (>= 2^28 flops); TMA form for batch-stacked products, '
                    'cp.async form for other batched products')
        if chosen is None and not aligned32 and problem.m % 16 + problem.n % 16 + problem.k % 16:
            chosen = _first(cands, 'popsc_bf16_edge')
            rule = rule or 'P-OPS4-B masked-edge mma.sync for awkward products'
        if chosen is None:
            chosen = _first(cands, 'popsc_bf16', 'popsc_bf16_tail', 'popsc_bf16_edge', 'mixed_generic')
            rule = (rule + '; ' if rule else '') + 'frozen reference bodies'
    if chosen is None and problem.out_mode == 'store_raw' and _exact_bf16_tensor():
        # LATCH-TRAIN-PRECISION (opt-in exact_bf16_products='tensor'): an unrounded FP32 result of a BF16-exact
        # product is implemented only by the masked TMA / cp.async wgmma bodies (store_raw); take them for every
        # admitted shape instead of leaving the product to the strict FP32 SIMT body.
        chosen = _first(cands, 'pops4b_tma', 'pops4b_wg')
        rule = (rule + '; ' if rule else '') + 'exact_bf16_products=tensor: unrounded FP32 result on the wgmma body'
    if (policy == 'step13' and chosen is not None and chosen.body in STEP13_REPLACED and problem.batch == 1
            and problem.flops >= STEP13_MIN_FLOPS):
        # Step 13 (development evidence, research/latch_release/HANDOFF.md): inside programs that already use the
        # common planner, the masked TMA wgmma body beat mma.sync for non-P5C batch-1 products of at least 2^28
        # flops on the complete update; P5C-admitted products keep their engine (moving them was slower).
        upgraded = _first(cands, 'pops4b_tma', 'pops4b_wg')
        if upgraded is not None:
            chosen, rule = upgraded, rule + '; step13 common-path rule: masked wgmma replaces mma.sync (batch 1)'
    if (policy == 'step13' and STORE_BOUND_WS and chosen is not None and chosen.body == 'pops4b_tma'
            and problem.batch == 1 and problem.k // 64 < STORE_BOUND_MAX_K_STAGES
            and problem.m * problem.n >= STORE_BOUND_MIN_OUTPUT):
        # A short-K product with a very large output spends most of each tile in its store; the
        # warp-specialized body (dedicated TMA producer, epilogue warps on a double-buffered staging tile) overlaps
        # that store with the next tile's MMAs. The same m64n128k16 chains preserve arithmetic;
        # qualification and complete-update measurements belong to the matrix campaign report.
        ws = _first(cands, 'p8ws')
        if ws is not None:
            chosen, rule = ws, rule + '; store-bound short-K rule: warp-specialized body overlaps the store'
        wide = _store_bound_p5c(cands, problem, resources)
        if wide is not None:
            # LATCH-TRAIN-MATMUL (opt-in matrix_bodies 'p5c'): the same-class P5C body with 256 x 128 (else
            # 128 x 256) tiles and per-stage release halves one operand's L2 re-reads of a large short-K output;
            # interval replay of the B32 readout logits: 1.74 -> 1.55 ms, bitwise.
            chosen, rule = wide, rule + '; matrix_bodies p5c: wide-tile P5C body for a store-bound short-K product'
    if chosen is not None:
        chosen.donor_rule = rule
    decision = {'policy': policy, 'problem': problem.to_dict(), 'selected': chosen.body if chosen else None,
                'rule': rule, 'resources': resources.to_dict(),
                'candidates': [c.record() for c in cands], 'refused': refused}
    return chosen, decision
